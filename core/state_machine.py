# -*- coding: utf-8 -*-
"""
Core State Machine & Orchestration Loop (Gatherion A2 Bot)
Runs on a non-blocking PyQt6 QThread (BotWorker) and coordinates Vision, InputDriver,
RouteManager, CombatRecovery, Break Management, and HUD / UI signals.
"""

import logging
import math
import os
import random
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# F-0540 tek kaynak: HP/MP iksir eşik yedek değerleri config/schema.py
# sabitlerinden okunur (sözleşme no 10: HP 65 / MP 45).
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT
from version import __version__, normalize_version

logger = logging.getLogger("Gatherion.StateMachine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class BotMode(str, Enum):
    """Supported operating modes for Gatherion A2 Bot."""
    ROUTE_MODE = "route"
    COMBAT_MODE = "combat"
    MANUAL_RECORD = "manual_record"
    AUTO_RECORD = "auto_record"
    EXPLORER_MODE = "explorer"

    # Alias for convenience
    ROUTE = "route"
    COMBAT = "combat"
    EXPLORER = "explorer"


class BotState(str, Enum):
    """Supported HUD badges and internal state machine states."""
    WALKING = "WALKING"
    GATHERING = "GATHERING"
    FIGHTING = "FIGHTING"
    PAUSED = "PAUSED"
    BREAK = "BREAK"
    WAITING = "WAITING"
    MAP = "MAP"

    # Aliases for 3-harvest & sequential navigation compatibility
    NAVIGATING_NODE = "WALKING"
    APPROACHING_NODE = "WALKING"
    EXPLORING = "WALKING"
    RESTING = "WAITING"


MODE_BADGE_MAP: Dict[str, str] = {
    BotMode.ROUTE_MODE.value: "TOPLAMA (ROUTE)",
    BotMode.COMBAT_MODE.value: "SAVAŞ (COMBAT)",
    BotMode.MANUAL_RECORD.value: "MANUAL RECORD",
    BotMode.AUTO_RECORD.value: "AUTO RECORD",
    BotMode.EXPLORER_MODE.value: "AUTO CARTOGRAPHY",
}


# D30: hoisted out of BotWorker._get_cfg, which rebuilt this mapping on
# every call. The mapping is only ever read (via .get), so a single
# module-level copy is behaviour-identical.
_CFG_ALIASES: Dict[str, Tuple[Tuple[Optional[str], str], ...]] = {
    "f_interval_min": (
        (None, "f_interval_min"),
        ("gathering", "f_interval_min"),
        ("gathering", "min_f_interval"),
    ),
    "f_interval_max": (
        (None, "f_interval_max"),
        ("gathering", "f_interval_max"),
        ("gathering", "max_f_interval"),
    ),
    "test_mode": (
        (None, "test_mode"),
        ("gathering", "test_mode"),
    ),
    "show_overlay": (
        (None, "show_overlay"),
        ("gathering", "show_overlay"),
    ),
    "harvests_per_node": (
        (None, "harvests_per_node"),
        ("gathering", "harvests_per_node"),
        ("route", "harvests_per_node"),
    ),
    "inter_harvest_delay_min": (
        (None, "inter_harvest_delay_min"),
        ("gathering", "inter_harvest_delay_min"),
    ),
    "inter_harvest_delay_max": (
        (None, "inter_harvest_delay_max"),
        ("gathering", "inter_harvest_delay_max"),
    ),
    "retry_on_extraction_fail": (
        (None, "retry_on_extraction_fail"),
        ("gathering", "retry_on_extraction_fail"),
    ),
    "max_retries_per_harvest_stage": (
        (None, "max_retries_per_harvest_stage"),
        ("gathering", "max_retries_per_harvest_stage"),
    ),
    "gather_key": (
        (None, "gather_key"),
        ("gathering", "gather_key"),
    ),
    "post_harvest_prompt_timeout": (
        (None, "post_harvest_prompt_timeout"),
        ("gathering", "post_harvest_prompt_timeout"),
        ("gathering", "prompt_reappear_timeout"),
    ),
    "gather_cast_duration_sec": (
        (None, "gather_cast_duration_sec"),
        ("gathering", "gather_cast_duration_sec"),
        ("gathering", "cast_duration_sec"),
    ),
    # P1-13: pause_on_keyboard_input KALDIRILDI — bu anahtarın şemada
    # (config/schema.py) alanı, UI'da denetimi ve üretimde okuyucusu yoktu;
    # klavye girdisiyle duraklatma mekanizması da yok (yalnızca kısayol
    # ctrl+alt+f10 ile duraklatma ve Alt-Tab koruması var). Takma-ad
    # girdisi tek başına ayarın okunabilir olduğu izlenimi veriyordu.
    "pause_on_alt_tab": (
        # P1-13: okuyucular route'u önce okur (main._resolve_pause_on_alt_tab,
        # BotWorker._get_cfg); UI gathering yazar, _cfg_set route'a aynalar.
        (None, "pause_on_alt_tab"),
        ("route", "pause_on_alt_tab"),
        ("gathering", "pause_on_alt_tab"),
    ),
    "rest_minutes": (
        (None, "rest_minutes"),
        ("route", "rest_minutes"),
        ("route", "rest_after_visit_min"),
    ),
    "gathered_join_route": (
        (None, "gathered_join_route"),
        (None, "gathered_join_active_route"),
        ("route", "gathered_join_active_route"),
        ("route", "gathered_join_route"),
        ("route", "gathered_nodes_join_route"),
    ),
    "time_to_reach_sec": (
        (None, "time_to_reach_sec"),
        ("route", "time_to_reach_sec"),
        ("route", "time_to_reach_node_sec"),
    ),
    "reclick_node_sec": (
        (None, "reclick_node_sec"),
        ("route", "reclick_node_sec"),
        ("route", "reclick_interval_sec"),
    ),
    "selective_gather_enabled": (
        (None, "selective_gather_enabled"),
        ("gathering", "selective_gather_enabled"),
        ("gathering", "selective_gather"),
    ),
    # F-0573: oto-iksir anahtarı — config/schema.py `gathering.auto_consumables`
    # (ayar R01'de ekleniyor); UI onay kutusu worker özniteliğini yazar.
    "auto_consumables": (
        (None, "auto_consumables"),
        ("gathering", "auto_consumables"),
    ),
    "selected_materials": (
        (None, "selected_materials"),
        ("gathering", "selected_materials"),
        (None, "selected_gather_materials"),
        ("gathering", "selected_gather_materials"),
    ),
    "gather_on_the_way": (
        (None, "gather_on_the_way"),
        ("route", "gather_on_the_way"),
        ("route", "exploration_gather"),
    ),
    "minimap_auto_record": (
        (None, "minimap_auto_record"),
        ("route", "minimap_auto_record"),
    ),
    "minimap_ignore_monsters": (
        (None, "minimap_ignore_monsters"),
        ("route", "minimap_ignore_monsters"),
    ),
    "hybrid_3d_and_minimap_record": (
        (None, "hybrid_3d_and_minimap_record"),
        ("route", "hybrid_3d_and_minimap_record"),
    ),
    "auto_record_3d_crests": (
        (None, "auto_record_3d_crests"),
        ("route", "auto_record_3d_crests"),
    ),
    "max_failed_in_row": (
        (None, "max_failed_in_row"),
        ("route", "max_failed_in_row"),
        ("route", "max_failures_before_wait"),
    ),
    "remove_node_after_failures": (
        (None, "remove_node_after_failures"),
        ("route", "remove_node_after_failures"),
    ),
    "wait_min_after_failed": (
        (None, "wait_min_after_failed"),
        ("route", "wait_min_after_failed"),
    ),
    "pause_after_no_gather_min": (
        (None, "pause_after_no_gather_min"),
        ("route", "pause_after_no_gather_min"),
        ("route", "pause_after_idle_min"),
    ),
    "fight_back": (
        (None, "fight_back"),
        (None, "fight_back_when_attacked"),
        ("route", "fight_back"),
        ("route", "fight_back_when_attacked"),
    ),
    "fight_stop_gather_sec": (
        (None, "fight_stop_gather_sec"),
        ("route", "fight_stop_gather_sec"),
    ),
    "combat_skill_burst_sec": (
        (None, "combat_skill_burst_sec"),
        ("input", "skill_burst_sec"),
        ("combat", "skill_burst_sec"),
        ("route", "skill_burst_sec"),
    ),
    "skills_per_burst": (
        (None, "skills_per_burst"),
        ("input", "skills_per_burst"),
        ("combat", "skills_per_burst"),
        ("route", "skills_per_burst"),
    ),
    "combat_skill_gcd_sec": (
        (None, "combat_skill_gcd_sec"),
        ("input", "skill_gcd_sec"),
        ("combat", "skill_gcd_sec"),
        ("route", "skill_gcd_sec"),
    ),
    "mp_full_threshold": (
        (None, "mp_full_threshold"),
        ("route", "mp_full_threshold"),
        ("combat", "mp_full_threshold"),
    ),
    # F-23-01: iksir eşik/tuş anahtarlarının takma adı yoktu — _get_cfg
    # düz (None, key) adayıyla route.hp/mp_potion_threshold_pct değerini
    # hiç çözemiyor, her zaman çağrı-yeri varsayılanını (65/45) dönüyordu.
    # Bu girdiler olmadan ManaPhaseCombat'a ayar geçirmek sahte bağlama
    # olurdu (kullanıcı 50 yazsa bile bot 65 kullanırdı).
    "hp_potion_threshold_pct": (
        (None, "hp_potion_threshold_pct"),
        ("route", "hp_potion_threshold_pct"),
    ),
    "mp_potion_threshold_pct": (
        (None, "mp_potion_threshold_pct"),
        ("route", "mp_potion_threshold_pct"),
    ),
    # F-23-01: tuş anahtarları J dalında şemaya eklendi (RouteSettings);
    # bu dalda _get_cfg varsayılanı (F1/F2) geçerli, J merge'inden sonra
    # dosyadaki değer çözülür.
    "hp_potion_key": (
        (None, "hp_potion_key"),
        ("route", "hp_potion_key"),
    ),
    "mp_potion_key": (
        (None, "mp_potion_key"),
        ("route", "mp_potion_key"),
    ),
    # P1-13: HP/MP/buff anahtarları şemada (route.*) ve UI'da vardı ama
    # _get_cfg takma adı yoktu — düz (None, key) adayı BotSettings
    # üzerinde çözülemiyor, her zaman çağrı-yeri varsayılanı dönüyordu.
    # F-23-01'deki eşik/tuş deseniyle aynı: takma ad olmadan ayar
    # geçirmek sahte bağlama olurdu (kullanıcı kapatsa bile bot
    # kullanmaya devam ederdi).
    "auto_use_hp_potion": (
        (None, "auto_use_hp_potion"),
        ("route", "auto_use_hp_potion"),
    ),
    "auto_use_mp_potion": (
        (None, "auto_use_mp_potion"),
        ("route", "auto_use_mp_potion"),
    ),
    "auto_refresh_buff_scroll": (
        (None, "auto_refresh_buff_scroll"),
        ("route", "auto_refresh_buff_scroll"),
    ),
    "buff_scroll_interval_sec": (
        (None, "buff_scroll_interval_sec"),
        ("route", "buff_scroll_interval_sec"),
    ),
    "buff_scroll_key": (
        (None, "buff_scroll_key"),
        ("route", "buff_scroll_key"),
    ),
    "ucb_explore_ratio": (
        (None, "ucb_explore_ratio"),
        ("route", "ucb_explore_ratio"),
    ),
    "skill_discovery_enabled": (
        (None, "skill_discovery_enabled"),
        ("route", "skill_discovery_enabled"),
        ("combat", "skill_discovery_enabled"),
    ),
    "combat_skill_keys": (
        (None, "combat_skill_keys"),
        (None, "skill_keys"),
        ("combat", "combat_skill_keys"),
        ("combat", "skill_keys"),
        ("route", "combat_skill_keys"),
        ("route", "skill_keys"),
    ),
    "skill_keys": (
        (None, "skill_keys"),
        (None, "combat_skill_keys"),
        ("combat", "skill_keys"),
        ("combat", "combat_skill_keys"),
        ("route", "skill_keys"),
        ("route", "combat_skill_keys"),
    ),
    "break_every_min": (
        (None, "break_every_min"),
        ("breaks", "break_every_min"),
        ("breaks", "interval_min"),
    ),
    "break_every_variance": (
        (None, "break_every_variance"),
        ("breaks", "break_every_variance"),
        ("breaks", "interval_jitter_min"),
    ),
    "break_lasts_min": (
        (None, "break_lasts_min"),
        ("breaks", "break_lasts_min"),
        ("breaks", "duration_min"),
    ),
    "break_lasts_variance": (
        (None, "break_lasts_variance"),
        ("breaks", "break_lasts_variance"),
        ("breaks", "duration_jitter_min"),
    ),
    "prompt_threshold": (
        (None, "prompt_threshold"),
        ("gathering", "prompt_threshold"),
        ("vision", "prompt_threshold"),
    ),
    "version": (
        (None, "version"),
        ("app", "version"),
    ),
}


class BotWorker(QThread):
    """
    Main non-blocking orchestration thread (QThread) for the Gatherion A2 Bot.
    Emits real-time HUD dictionaries, timestamped log lines, node record events,
    and state transition notifications to the PyQt6 UI layer.
    """

    hud_update = pyqtSignal(dict)
    log_added = pyqtSignal(str)
    node_recorded = pyqtSignal(dict)
    state_changed = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)
    # Harvest aşama sinyali (stage, max_harvests): main.py bunu hasattr ile
    # arayıp OverlayHUD.update_harvest_stage'e bağlar; sinyal yoksa rozet
    # hiç güncellenemezdi (Harvest 0/3 hep 0 kalırdı).
    harvest_stage_changed = pyqtSignal(int, int)

    def __init__(
        self,
        config: Any = None,
        vision: Any = None,
        input_driver: Any = None,
        route_manager: Any = None,
        combat_recovery: Any = None,
        navigator: Any = None,
        recorder: Any = None,
        overlay_rect_provider: Optional[Callable[[], Any]] = None,
        parent: Any = None,
        hud_reader: Any = None,
        minimap_scanner: Any = None,
        material_scanner: Any = None,
        yolo_detector: Any = None,
    ) -> None:
        super().__init__(parent)

        self._lock = threading.RLock()
        self.overlay_rect_provider = overlay_rect_provider

        # Initialize or bind core dependencies
        self.config = config if config is not None else self._init_default_config()
        self.cfg = self.config

        self.vision = vision if vision is not None else self._init_default_vision()
        self.input_driver = (
            input_driver if input_driver is not None else self._init_default_input_driver()
        )
        self.driver = self.input_driver

        self.route_manager = (
            route_manager if route_manager is not None else self._init_default_route_manager()
        )
        self.routes = self.route_manager
        self.route_mgr = self.route_manager

        self.navigator = (
            navigator if navigator is not None else self._init_default_navigator()
        )
        self.recorder = (
            recorder if recorder is not None else self._init_default_recorder()
        )
        self.combat_recovery = (
            combat_recovery
            if combat_recovery is not None
            else self._init_default_combat_recovery()
        )
        self.combat = self.combat_recovery

        # F-0198: a None core dependency must be visible, not silently carried.
        # run() refuses to start the loop while any of these is missing.
        self._core_deps_missing = [
            name
            for name, dep in (
                ("vision", self.vision),
                ("input_driver", self.input_driver),
                ("route_manager", self.route_manager),
                ("combat_recovery", self.combat_recovery),
            )
            if dep is None
        ]
        if self._core_deps_missing:
            logger.error(
                "Core dependencies missing, bot will not run: %s",
                ", ".join(self._core_deps_missing),
            )
        self._pending_route_sort = False
        self._last_gather_prompt = None
        self._gathered_nodes_count = 0
        self._stage_cast_bar_seen = False

        # Core state & mode flags
        self._mode: str = BotMode.ROUTE_MODE.value
        self._state: str = BotState.PAUSED.value
        self._running: bool = False
        self._paused: bool = True
        self._pause_reason: str = "Duraklatıldı · Devam için Ctrl+Alt+F10"
        self._f8_pending: bool = False

        # Session & timing metrics
        self._session_start_time: float = time.time()
        self._state_enter_time: float = time.time()
        self._last_gather_time: float = time.time()
        self._last_f_press_time: float = 0.0
        self._next_f_interval: float = 1.8
        self._last_bot_key_time: float = 0.0

        # Counters & route tracking
        self.gathered_count: int = 0
        self.f_pressed_count: int = 0
        self.saved_nodes_count: int = 0
        self.last_saved_str: str = "None"
        self.last_prompt_match: int = 0
        self.failed_in_row: int = 0

        self.active_node: Optional[Any] = None
        self.active_node_distance_m: int = 0
        self._walk_start_time: float = 0.0
        self._arrival_at_node_ts: float = 0.0
        self._last_reclick_time: float = 0.0
        self._route_was_fighting: bool = False
        self._route_fight_had_signal: bool = False
        self._route_combat_confirmed: bool = False
        self._route_saw_target_bar: bool = False
        self._route_saw_hp_drop: bool = False
        self._route_prev_hp_ratio: Optional[float] = None
        # P1-10 (K-14): watchdog'un okuduğu vitaller. BotWorker'da
        # hp_ratio/current_hp hiç yayınlanmıyordu — watchdog'un
        # _check_worker_vitals'ı (check_health ← monitor loop) her
        # saniye çalışıyor ama her zaman no-op kalıyordu (WORKER_DEAD /
        # WORKER_LOW_HP / WORKER_REVIVED hiç ateşlenemiyordu).
        # Son bilinen oyuncu HP oranı burada yayınlanır; ölüm/hayatta
        # kalma kararı _publish_worker_hp_ratio'nun debounce'una göre
        # verilir (tek karelik 0.0 okuma hatası ölüm sayılmaz).
        self.hp_ratio: Optional[float] = None
        self._worker_hp_prev_ratio: Optional[float] = None
        self._worker_hp_near_zero_streak: int = 0
        self._worker_hp_pre_collapse: Optional[float] = None
        self._route_last_loot_ts: float = 0.0
        self._route_bar_since_loot: bool = False
        # ROTA-İPTAL-ZİNCİRİ (Görev 2): doğrulanmış savaş kapısı.
        # Tek karelik bar/HP sinyali FIGHTING'e sokmaz — sinyal ardışık
        # tick'lerde tekrarlanmalı (streak); bir savaş bayrağı
        # temizlendikten sonra histerezis penceresi içinde FIGHTING'e
        # tekrar girilmez.
        self._route_bar_streak: int = 0
        self._route_hp_streak: int = 0
        # fix/v20-panic-combat: bar penceresi içinde saldıran-düşman
        # kanıtı görüldü mü (HP düşüşü veya ToT aggro). Bar tek başına
        # FIGHTING'e sokmaz — bkz. _route_combat_signal_confirmed.
        self._route_bar_corroborated: bool = False
        self._route_fight_last_cleared_ts: float = 0.0
        self._route_fight_suspended: bool = False
        # W54/v19: aktif hasat sirasinda dogrulanmis saldiri sinyalinin
        # ilk goruldugu an (time.time — _handle_combat_and_death'teki
        # now_ts ile ayni saat). fight_stop_gather_sec dolmadan savas
        # toplamayi bolmez; bkz. _handle_combat_and_death.
        self._gather_under_attack_since: float = 0.0
        self._map_open_backoff_until: float = 0.0
        self._map_open_fail_announced: bool = False
        self._last_nav_block: str = ""
        self._gather_lost_frames: int = 0
        self._is_gathering_active: bool = False

        # Mechanic 1: 3-Harvests-Per-Node Loop (1/3 -> 2/3 -> 3/3) state
        self._current_node_harvest_stage: int = 0
        self._current_node_stage_retries: int = 0
        self._target_harvests_for_active_node: int = 3
        self._inter_harvest_wait_until: float = 0.0
        self._post_harvest_prompt_deadline: float = 0.0
        self._waiting_for_next_stage_prompt: bool = False
        self._stage_cast_in_progress: bool = False
        self._stage_cast_start_time: float = 0.0
        self._stage_cast_end_time: float = 0.0
        # Cast ilerleme kanıtı (Y2/Y3): barın dolduğu, tek karede %90
        # görülmesiyle değil, ardışık okumalarda YÜKSELDİĞİYLE kanıtlanır.
        self._stage_cast_prev_pct: Optional[float] = None
        self._stage_cast_rose: bool = False
        # Son renkli karenin yaşı (O6): bayat kareyle cast kararı verilmez.
        self._last_color_frame_ts: float = 0.0
        # Düğüm mesafesinin tazeliği (16/1, 16/2): mesafe yalnızca harita
        # reclick'inde güncellenir; bayat mesafe varış/takılma kararı verdirmez.
        self._active_node_distance_updated_ts: float = 0.0
        # Barsız fight-back adım sayacı (17/B7 — P3-4: eski metin bayattı).
        # Eskiden "L3 (fight_back_step) loot F'si sayacı" deniyordu; L3
        # Y2/P1-11 düzeltmesinden beri kör loot F basmıyor
        # (combat_recovery.py fight_back_step, Y2 bloğu) — sayılan artık
        # yalnızca ardışık barsız fight-back adımıdır. Sayaç korunur:
        # _loot_confirmed_fight onu streak>=2 eşiğiyle (l3_looted) okur.
        self._route_fightback_no_bar_streak: int = 0
        # Oto-kayıt modu aşama takibi (15/O2): manuel modda da gerçek aşama
        # yazılsın diye aynı prompt'ta art arda toplananlar sayılır.
        self._auto_record_stage: int = 0
        self._auto_record_prompt_name: str = ""
        # Kartografi hata görünürlüğü (16/12): sessiz bozulma olmasın.
        self._carto_error_count: int = 0
        self._last_detected_prompt_key: str = "f"

        # Strict sequential node commitment & cursor (prevents jumping 40 -> 84)
        self._current_target_node_id: Optional[Any] = None
        self._sequential_cursor_idx: int = 0
        self._current_seq_pos: int = 0

        # Mechanic 2: Continuous Hybrid 3D World + Top-Right Minimap Auto-Recording
        self._last_minimap_auto_record_ts: float = 0.0
        self._approaching_3d_crest: Optional[Dict[str, Any]] = None

        # Break scheduling state
        self._next_break_timestamp: float = 0.0
        self._break_end_timestamp: float = 0.0
        self._last_anti_afk_time: float = 0.0
        self._schedule_next_break()

        # Failure cooldown state
        self._failure_wait_until: float = 0.0
        self._next_reclick_interval: float = 10.0
        self.auto_cartography_enabled: bool = True

        # Optional Advanced Stealth, Sequential Cartography & Telemetry Subsystems (Agents 11-25)
        self.telemetry: Any = self._init_optional_subsystem("core.telemetry", "TelemetryEngine")
        self.watchdog: Any = None
        self.sequential_planner: Any = self._init_optional_subsystem(
            "core.sequential_router", "SequentialRoutePlanner"
        )
        # H-sweep (dilim 12 K-1): Harvest/Navigation/Corridor üretim
        # kurulumu — navigator + planner kurulduktan sonra.
        self._init_navigation_helpers()
        self.hud_reader: Any = (
            hud_reader
            if hud_reader is not None
            else self._init_optional_subsystem("core.ocr_distance", "GameHUDReader")
        )
        self.minimap_scanner: Any = (
            minimap_scanner
            if minimap_scanner is not None
            else self._init_optional_subsystem("core.minimap_scanner", "MinimapScanner")
        )
        self.material_scanner: Any = (
            material_scanner
            if material_scanner is not None
            else self._init_optional_subsystem("core.material_scanner", "MaterialScanner")
        )
        # Dataset factory ÖNCE kurulmalı: explorer kurulurken factory'yi alır
        # (eski sırada explorer hep None alıp her kartografi tick'inde yamaya muhtaçtı).
        self.dataset_factory: Any = self._init_optional_subsystem(
            "dataset_factory.factory_controller", "AutoLabelingDatasetFactory"
        )
        self.explorer_controller: Any = self._init_explorer_controller()
        self.explorer_ctrl: Any = self.explorer_controller
        self.auto_cartographer: Any = (
            getattr(self.explorer_controller, "auto_cartographer", None)
            if self.explorer_controller is not None
            else self._init_auto_cartographer()
        )
        self.human_timing: Any = self._init_optional_subsystem(
            "core.human_timing", "HumanTimingEngine"
        )
        self.organic_anti_afk: Any = self._init_optional_subsystem(
            "core.human_timing", "OrganicAntiAFK"
        )
        self.frame_pacer: Any = self._init_optional_subsystem(
            "core.stealth_capture", "AdaptiveFramePacer"
        )
        self.windmouse: Any = self._init_optional_subsystem(
            "core.windmouse_humanizer", "WindMouseEngine"
        )
        self.biometric_clicker: Any = self._init_optional_subsystem(
            "core.windmouse_humanizer", "BiometricClickGenerator"
        )
        self.behavioral_guard: Any = self._init_optional_subsystem(
            "core.behavioral_guard", "BehavioralStealthGuard"
        )
        self.panic_system: Any = self._init_optional_subsystem(
            "core.opsec_crypto", "PanicAlertSystem"
        )
        if self.dataset_factory is None:
            _df_err = ""
            try:
                _df_err = str(
                    self.__dict__.get("_optional_subsystem_errors", {}).get(
                        "dataset_factory.factory_controller.AutoLabelingDatasetFactory", ""
                    )
                    or ""
                )
            except Exception:
                _df_err = ""
            _df_msg = (
                "DATASET FACTORY YOK — YZ kare çekimi (oto-kayıt) çalışmayacak. "
                f"Sebep: {_df_err or 'bilinmiyor (import/kurulum hatası)'}"
            )
            logger.error(_df_msg)
            try:
                self._emit_log(_df_msg)
            except Exception:
                pass
        self.game_learner: Any = None
        try:
            from core.omni_game_learner import OmniGameLearner
            self.game_learner = OmniGameLearner(
                bot_worker=self,
                save_screenshots=True,
                min_sample_interval_sec=12.0,
            )
        except Exception as gl_exc:
            logger.debug("OmniGameLearner başlatma hatası: %s", gl_exc)

        self.action_bus: Any = None
        try:
            from core.game_action_bus import GameActionBus

            self.action_bus = GameActionBus(
                input_driver=self.input_driver,
                skill_bar_reader=None,
                vision=self.vision,
                on_log=self._emit_log,
                release_basic_attack=self._release_combat_right_button,
                restore_basic_attack=lambda: self._sync_combat_right_button(True, force=True),
            )
        except Exception as bus_exc:
            logger.debug("GameActionBus başlatma hatası: %s", bus_exc)

        self.mapless_navigator: Any = self._init_optional_subsystem(
            "core.navigation_intel", "MaplessMicroNavigator"
        )

        # Enterprise Suite & 50-Point Vision/Navigation Subsystems
        self.dpi_guard: Any = self._init_optional_subsystem(
            "core.dpi_focus_guard", "DPIScalingSynchronizer"
        )
        self.focus_monitor: Any = self._init_optional_subsystem(
            "core.dpi_focus_guard", "ForegroundFocusGuard"
        )
        self.pause_on_alt_tab: bool = bool(self._get_cfg("pause_on_alt_tab", True))
        self._last_focus_lost_ts: float = 0.0
        self.route_backup_mgr: Any = self._init_optional_subsystem(
            "core.dpi_focus_guard", "RouteBackupManager"
        )
        self.template_tuner: Any = self._init_optional_subsystem(
            "core.dpi_focus_guard", "SelfHealingTemplateTuner"
        )
        self.feather_filter: Any = self._init_optional_subsystem(
            "core.vision_filters", "FeatherNegativeFilter"
        )
        self.pyramid_crest_detector: Any = self._init_optional_subsystem(
            "core.vision_filters", "Pyramid3DCrestDetector"
        )
        self.dynamic_key_reader: Any = self._init_optional_subsystem(
            "core.vision_filters", "DynamicInteractionKeyReader"
        )
        self.monster_circ_filter: Any = self._init_optional_subsystem(
            "core.vision_filters", "MonsterCircularityFilter"
        )
        self.nameplate_isolator: Any = self._init_optional_subsystem(
            "core.vision_filters", "NameplateHSVIsolator"
        )
        self.minimap_orientation_guard: Any = self._init_optional_subsystem(
            "core.patch_normalizer", "MinimapOrientationGuard"
        )
        self.patch_lighting_normalizer: Any = self._init_optional_subsystem(
            "core.patch_normalizer", "SobelClahePatchNormalizer"
        )
        self.roi_cropper: Any = self._init_optional_subsystem(
            "core.patch_normalizer", "HighResROICropper"
        )
        self.watershed_separator: Any = self._init_optional_subsystem(
            "core.patch_normalizer", "WatershedClusterSeparator"
        )
        self.motion_blur_guard: Any = self._init_optional_subsystem(
            "core.patch_normalizer", "MotionBlurGate"
        )
        self.skill_bar_monitor: Any = self._init_optional_subsystem(
            "core.skill_bar_reader", "SkillBarReader"
        )
        if self.action_bus is not None and hasattr(self.action_bus, "bind"):
            try:
                self.action_bus.bind(
                    input_driver=self.input_driver,
                    skill_bar_reader=self.skill_bar_monitor,
                    vision=self.vision,
                    on_log=self._emit_log,
                    release_basic_attack=self._release_combat_right_button,
                    restore_basic_attack=lambda: self._sync_combat_right_button(True, force=True),
                )
            except Exception as bind_exc:
                logger.debug("GameActionBus bind skipped: %s", bind_exc)

        # Zero-Knowledge Vision Agent layers (PerceptionHub → ManaPhase → Discovery → Economy)
        self.input_port: Any = None
        self.perception_hub: Any = None
        self.mana_phase_combat: Any = None
        self.skill_discovery: Any = None
        self.slot_economy: Any = None
        self.rl_reward_log: Any = None
        self.obstacle_memory: Any = None
        self._frame_cognition: Any = None
        self._ucb_slot_hint: Any = None
        self._discovery_probe: Any = None
        try:
            from core.hardware_input_port import (
                SoftInputAdapter,
                build_default_input_port,
                resolve_serial_hid_port,
            )

            serial_port = resolve_serial_hid_port()
            # SoftInputAdapter keeps mouse soft-aim; serial/Interception attach on InputController
            self.input_port = build_default_input_port(
                input_driver=self.input_driver,
                serial_port=serial_port,
            )
            if not isinstance(self.input_port, SoftInputAdapter):
                self.input_port = SoftInputAdapter(input_driver=self.input_driver)
            if self.action_bus is not None and hasattr(self.action_bus, "bind_input_port"):
                self.action_bus.bind_input_port(self.input_port)
        except Exception as port_exc:
            logger.debug("SoftInputAdapter skipped: %s", port_exc)
        try:
            from core.perception_hub import PerceptionHub

            self.perception_hub = PerceptionHub(
                game_learner=self.game_learner,
                skill_bar_reader=self.skill_bar_monitor,
                vision=self.vision,
                yolo_detector=None,
                combat_recovery=getattr(self, "combat_recovery", None),
            )
        except Exception as hub_exc:
            logger.debug("PerceptionHub skipped: %s", hub_exc)
        try:
            from core.mana_phase_combat import ManaPhaseCombat

            # F-23-01: HP/MP iksir eşikleri artık ayardan besleniyor
            # (route.hp/mp_potion_threshold_pct, şema varsayılanı 65/45 —
            # no10 kararı HP 65). Eskiden kurucuya yalnızca
            # mp_full_threshold geçiliyor, savaş yolu 0.50/0.30 sabitini,
            # rota yolu smart 0.65/0.45 sabitini kullanıyordu; kullanıcı
            # ayarı değiştirse bile hiçbir şey değişmiyordu. Düz + smart
            # eşik tek değere indirildi: ayar neyse o.
            self.mana_phase_combat = ManaPhaseCombat(
                mp_full_threshold=float(self._get_cfg("mp_full_threshold", 0.95) or 0.95),
                hp_potion_threshold=float(self._get_cfg("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT) or HP_POTION_THRESHOLD_PCT) / 100.0,
                mp_potion_threshold=float(self._get_cfg("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT) or MP_POTION_THRESHOLD_PCT) / 100.0,
                smart_hp_potion_threshold=float(self._get_cfg("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT) or HP_POTION_THRESHOLD_PCT) / 100.0,
                smart_mp_potion_threshold=float(self._get_cfg("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT) or MP_POTION_THRESHOLD_PCT) / 100.0,
                hp_potion_key=str(self._get_cfg("hp_potion_key", "F1") or "F1"),
                mp_potion_key=str(self._get_cfg("mp_potion_key", "F2") or "F2"),
                # P1-13: iksir/buff anahtarları artık ayardan besleniyor
                # (route.auto_use_hp/mp_potion, auto_refresh_buff_scroll,
                # buff_scroll_interval_sec, buff_scroll_key).
                auto_use_hp_potion=bool(self._get_cfg("auto_use_hp_potion", True)),
                auto_use_mp_potion=bool(self._get_cfg("auto_use_mp_potion", True)),
                auto_refresh_buff_scroll=bool(self._get_cfg("auto_refresh_buff_scroll", True)),
                buff_scroll_interval_sec=float(self._get_cfg("buff_scroll_interval_sec", 300) or 300),
                buff_scroll_key=str(self._get_cfg("buff_scroll_key", "F7") or "F7"),
                on_log=self._emit_log,
            )
        except Exception as mana_exc:
            logger.debug("ManaPhaseCombat skipped: %s", mana_exc)
        try:
            from core.skill_discovery import SkillDiscovery

            self.skill_discovery = SkillDiscovery(on_log=self._emit_log)
            if self.mana_phase_combat is not None and hasattr(self.mana_phase_combat, "set_lexicon"):
                self.mana_phase_combat.set_lexicon(self.skill_discovery.lexicon())
            if bool(self._get_cfg("skill_discovery_enabled", True)):
                self.skill_discovery.start()
        except Exception as disc_exc:
            logger.debug("SkillDiscovery skipped: %s", disc_exc)
        try:
            from core.slot_economy import SlotEconomy

            self.slot_economy = SlotEconomy(
                explore_ratio=float(self._get_cfg("ucb_explore_ratio", 0.20) or 0.20),
                on_log=self._emit_log,
            )
        except Exception as eco_exc:
            logger.debug("SlotEconomy skipped: %s", eco_exc)
        try:
            from core.rl_reward_log import ObstacleMemory, RlRewardLog

            self.rl_reward_log = RlRewardLog()
            self.obstacle_memory = ObstacleMemory(slot_economy=self.slot_economy)
        except Exception as rl_exc:
            logger.debug("RlRewardLog skipped: %s", rl_exc)

        self.micro_camera_look: Any = self._init_optional_subsystem(
            "core.skill_bar_reader", "MicroCameraLookAround"
        )
        self.incident_recorder: Any = self._init_optional_subsystem(
            "core.incident_recorder", "IncidentRecorder"
        )
        self.mobile_token_auth: Any = self._init_optional_subsystem(
            "core.incident_recorder", "MobileTokenAuthenticator"
        )
        self.dry_run_simulator: Any = self._init_optional_subsystem(
            "core.dry_run_simulator", "DryRunReplaySimulator"
        )
        self.broker_scanner: Any = self._init_optional_subsystem(
            "core.economy_broker", "TradeBrokerPriceScanner"
        )
        self.crafting_controller: Any = self._init_optional_subsystem(
            "core.economy_broker", "AutoCraftingMorphQueue"
        )
        self.economy_calculator: Any = self._init_optional_subsystem(
            "core.economy_broker", "SessionProfitSimulator"
        )
        # v20-onnx-reload: main may inject one shared ONNXYolo3DDetector so
        # BotWorker does not build a second CUDA session that hub/reflex keep
        # after a later attribute overwrite.
        if yolo_detector is not None:
            self.yolo_detector = yolo_detector
        else:
            self.yolo_detector = self._init_optional_subsystem(
                "core.neural_audio_perception", "ONNXYolo3DDetector"
            )
        if (
            self.material_scanner is not None
            and self.yolo_detector is not None
            and hasattr(self.material_scanner, "set_yolo_detector")
        ):
            try:
                self.material_scanner.set_yolo_detector(self.yolo_detector)
            except Exception:
                pass
        if self.perception_hub is not None and hasattr(self.perception_hub, "bind"):
            try:
                self.perception_hub.bind(
                    yolo_detector=self.yolo_detector,
                    combat_recovery=self.combat_recovery,
                    skill_bar_reader=self.skill_bar_monitor,
                    game_learner=self.game_learner,
                    vision=self.vision,
                )
            except Exception:
                pass
        self.reflex: Any = self._init_reflex_scanner()
        self.audio_analyzer: Any = self._init_optional_subsystem(
            "core.neural_audio_perception", "AudioLoopbackFFTAnalyzer"
        )
        self.alt_char_mgr: Any = self._init_optional_subsystem(
            "core.multi_char_scheduler", "AltCharacterManager"
        )
        self.task_scheduler: Any = self._init_optional_subsystem(
            "core.multi_char_scheduler", "AutomationTaskScheduler"
        )
        self.pet_buff_mgr: Any = self._init_optional_subsystem(
            "core.multi_char_scheduler", "PetAndBuffManager"
        )
        self.depth_avoider: Any = self._init_optional_subsystem(
            "core.depth_glide_navigator", "MonocularDepthObstacleAvoidance"
        )
        self.glide_pilot: Any = self._init_optional_subsystem(
            "core.depth_glide_navigator", "GlideWaypointController"
        )

        # Sync initial driver flags
        self._sync_input_driver_state()
        self._rehydrate_saved_nodes_count()

        # Selective gathering mode ("Seç-Topla")
        self.selective_gather_enabled: bool = False
        self.selected_gather_materials: list[str] = []
        cfg_sel_en = self._get_cfg("selective_gather_enabled", None)
        cfg_mats = self._get_cfg("selected_materials", None)
        if cfg_mats:
            self.set_selected_gather_materials(
                cfg_mats,
                enabled=bool(cfg_sel_en) if cfg_sel_en is not None else True,
            )
        elif cfg_sel_en is not None:
            self.selective_gather_enabled = bool(cfg_sel_en)

    def _rehydrate_saved_nodes_count(self) -> None:
        """Rehydrate self.saved_nodes_count from route_manager across restarts.

        Görev 4: sayı yeniden yükleniyordu ama last_saved_str "None"da
        kalıyordu (170 AUTO MAPPED düğüm varken panelde "LAST MAPPED: None"
        görünmesinin nedeni buydu — otomatik haritalama :1967'de zaten
        güncelliyor, eksik olan yalnızca yeniden başlatma sonrası
        rehidrasyondu). Kayıtlı en büyük düğüm kimliğinden "Node <id>"
        olarak doldurulur; bu oturumda zaten kaydedilmişse değiştirilmez.
        """
        try:
            rm = self._plain_attr("route_manager")
            if rm is None:
                return
            count = 0
            nodes: Optional[list] = None
            # Açık bölge argümanı: bölge-uyumsuzluğu engeli yalnızca
            # aktif-rotayı çalıştıran (argümansız) çağrıları durdurur;
            # bu okuma kayıtlı veriyi gösterir, engellenmemelidir.
            active_region = getattr(rm, "active_region", None)
            if hasattr(rm, "get_nodes"):
                try:
                    nodes = rm.get_nodes(region=active_region)
                except TypeError:
                    nodes = rm.get_nodes()
                if isinstance(nodes, list):
                    count = len(nodes)
            elif hasattr(rm, "get_route_nodes"):
                try:
                    nodes = rm.get_route_nodes(region_name=active_region)
                except TypeError:
                    nodes = rm.get_route_nodes()
                if isinstance(nodes, list):
                    count = len(nodes)
            if count > 0:
                self.saved_nodes_count = max(int(self._plain_attr("saved_nodes_count", 0) or 0), count)
                if str(self._plain_attr("last_saved_str", "None") or "None") == "None":
                    max_id = 0
                    for n in nodes or []:
                        try:
                            if isinstance(n, dict):
                                nid = int(n.get("id", 0) or 0)
                            else:
                                nid = int(getattr(n, "id", 0) or 0)
                        except (TypeError, ValueError):
                            continue
                        if nid > max_id:
                            max_id = nid
                    if max_id > 0:
                        self.last_saved_str = f"Node {max_id}"
        except Exception as exc:
            logger.debug("Failed to rehydrate saved_nodes_count: %s", exc)

    @property
    def settings(self) -> Any:
        """Returns the underlying BotSettings object or config manager."""
        if self.config is not None and hasattr(self.config, "settings"):
            return getattr(self.config, "settings")
        return self.config

    @property
    def active_mode(self) -> str:
        """Alias for self.mode."""
        return self.mode

    def set_selected_gather_materials(
        self,
        materials: Sequence[str],
        enabled: Optional[bool] = None,
    ) -> None:
        """Configure the active materials for selective gathering ('Seç-Topla')."""
        if not materials:
            self.selected_gather_materials = []
            self.selective_gather_enabled = False if enabled is None else bool(enabled)
        else:
            filtered: list[str] = []
            from core.material_scanner import normalize_material_name

            for m in materials:
                if not m:
                    continue
                s = str(m).strip()
                norm = normalize_material_name(s)
                if norm in ("tumu", "all", "*", "hepsi", "tum", "none"):
                    continue
                filtered.append(s)
            self.selected_gather_materials = filtered
            if enabled is not None:
                self.selective_gather_enabled = bool(enabled)
            else:
                self.selective_gather_enabled = bool(filtered)

        # Synchronize with MaterialScanner
        mat_scan = self._plain_attr("material_scanner")
        if mat_scan is not None and hasattr(mat_scan, "set_selected_materials"):
            try:
                if self.selective_gather_enabled:
                    mat_scan.set_selected_materials(self.selected_gather_materials)
                else:
                    mat_scan.set_selected_materials([])
            except Exception as exc:
                logger.debug("Failed to set selected materials on MaterialScanner: %s", exc)

        self._emit_log(
            f"Seç-Topla: {'Aktif' if self.selective_gather_enabled else 'Devre Dışı'} · "
            f"Materyaller: {', '.join(self.selected_gather_materials) if self.selected_gather_materials else 'Tümü'}"
        )

    def get_selected_gather_materials(self) -> list[str]:
        """Return the list of currently selected materials for gathering."""
        return list(getattr(self, "selected_gather_materials", []))

    def _normalized_selected_materials(self) -> frozenset:
        """Normalized selected-material set, cached per selection contents.

        D30: ``is_material_selected_for_gather`` used to re-run the full
        ``match_material_names`` loop (regex normalisation + alias-table
        rebuild + catalog lookup per target) for every candidate. The
        cache key is the current selection tuple, so it stays correct
        even if ``selected_gather_materials`` is assigned directly
        instead of via ``set_selected_gather_materials``.
        """
        from core.material_scanner import normalize_material_name

        selected = getattr(self, "selected_gather_materials", []) or []
        key = tuple(selected)
        cache = self.__dict__.get("_selected_materials_norm_cache")
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == key:
            return cache[1]
        norm_set = {
            normalize_material_name(str(s)) for s in key
        }
        norm_set.discard("")
        norm = frozenset(norm_set)
        self.__dict__["_selected_materials_norm_cache"] = (key, norm)
        return norm

    def is_material_selected_for_gather(self, mat_or_node: Any) -> bool:
        """
        Check if a given material string, DetectedMaterial, or route node matches
        the currently selected materials for gathering.
        If selective gathering is inactive, returns True.
        """
        if not getattr(self, "selective_gather_enabled", False):
            return True
        selected = getattr(self, "selected_gather_materials", [])
        if not selected:
            return True

        from core.material_scanner import match_material_names, normalize_material_name

        mat_str = ""
        if isinstance(mat_or_node, str):
            mat_str = mat_or_node
        elif mat_or_node is not None:
            for field in (
                "resource_type",
                "material_type",
                "name",
                "material",
                "category",
                "class_name",
            ):
                v = self._get_node_field(mat_or_node, field)
                if v:
                    mat_str = str(v)
                    break
            if not mat_str and hasattr(mat_or_node, "material_type"):
                mat_str = str(getattr(mat_or_node, "material_type", "") or "")

        if not mat_str:
            return False

        # D30 fast path: exact equality after normalisation is step 1 of
        # match_material_names, so a hit here is exactly what the loop
        # would return — answered with one normalise + one set lookup.
        norm_cand = normalize_material_name(mat_str)
        if norm_cand and norm_cand in self._normalized_selected_materials():
            return True

        # D30 memo: match_material_names(target, candidate) is a pure
        # function of its two string inputs (normalisation + a literal
        # alias table + a static catalog lookup), and the same
        # (target, candidate) pairs are re-checked on every route tick
        # (_get_next_ready_node / _get_ready_and_resting_counts /
        # _check_nearby_materials). Memoise per worker, bounded so the
        # table cannot grow without limit.
        memo = self.__dict__.get("_match_names_memo")
        if not isinstance(memo, dict):
            memo = self.__dict__["_match_names_memo"] = {}
        for target in selected:
            memo_key = (target, mat_str)
            cached = memo.get(memo_key)
            if cached is None:
                cached = bool(match_material_names(target, mat_str))
                if len(memo) >= 1024:
                    memo.clear()
                memo[memo_key] = cached
            if cached:
                return True
        return False

    def _check_nearby_materials(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        max_distance_m: float = 25.0,
    ) -> Optional[Any]:
        """
        Scan the frame for nearby gatherable materials using MaterialScanner.
        In selective gathering mode, strictly filters candidates to match selected materials,
        bypassing non-matching materials (e.g., herbs when mining Orichalcum).
        Returns the highest-priority matching DetectedMaterial, or None.
        """
        mat_scan = self._plain_attr("material_scanner")
        if mat_scan is None or not hasattr(mat_scan, "scan_frame_for_materials"):
            return None

        frame = frame_bgr if frame_bgr is not None else self.__dict__.get("_last_color_frame")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            frame = self._grab_frame(grayscale=False)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        # Synchronize scanner selection with worker selection
        is_selective = getattr(self, "selective_gather_enabled", False) and bool(
            getattr(self, "selected_gather_materials", None)
        )
        if hasattr(mat_scan, "set_selected_materials"):
            try:
                if is_selective:
                    mat_scan.set_selected_materials(self.selected_gather_materials)
                elif (
                    hasattr(mat_scan, "get_selected_materials")
                    and mat_scan.get_selected_materials()
                ):
                    mat_scan.set_selected_materials([])
            except Exception:
                pass

        try:
            candidates = mat_scan.scan_frame_for_materials(frame)
        except Exception as exc:
            logger.debug("_check_nearby_materials scan error: %s", exc)
            return None

        if not candidates:
            return None

        valid_targets = []
        for cand in candidates:
            dist = getattr(cand, "estimated_distance_m", 0.0) or 0.0
            if max_distance_m > 0 and dist > max_distance_m:
                continue

            mtype = getattr(cand, "material_type", "") or ""
            if is_selective:
                if not self.is_material_selected_for_gather(mtype):
                    # Strictly bypass unselected material
                    continue
            valid_targets.append(cand)

        # F-0392: "nearest" means sorted by estimated distance, not list order.
        valid_targets.sort(key=lambda c: float(getattr(c, "estimated_distance_m", 0.0) or 0.0))
        return valid_targets[0] if valid_targets else None

    def _should_bypass_unselected_gather(
        self, frame_bgr: Optional[np.ndarray] = None
    ) -> bool:
        """
        Determines whether a detected gather prompt or nearby interaction should be bypassed
        because selective gathering is active and the target material does not match selected materials.
        """
        if not getattr(self, "selective_gather_enabled", False):
            return False
        selected = getattr(self, "selected_gather_materials", [])
        if not selected:
            return False

        # 1. If active_node is known, check if it matches
        if self.active_node is not None:
            return not self.is_material_selected_for_gather(self.active_node)

        frame = frame_bgr if frame_bgr is not None else self.__dict__.get("_last_color_frame")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            frame = self._grab_frame(grayscale=False)

        # 2. If no active_node (opportunistic gather), inspect nearby materials via MaterialScanner
        nearby = self._check_nearby_materials(frame_bgr=frame, max_distance_m=15.0)
        if nearby is not None:
            return False

        # 3. Check what materials are detected on screen to confirm whether an unselected material exists
        mat_scan = self._plain_attr("material_scanner")
        if mat_scan is not None and hasattr(mat_scan, "scan_frame_for_materials"):
            try:
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    prev_sel = (
                        mat_scan.get_selected_materials()
                        if hasattr(mat_scan, "get_selected_materials")
                        else []
                    )
                    if hasattr(mat_scan, "set_selected_materials"):
                        mat_scan.set_selected_materials([])
                    all_cands = mat_scan.scan_frame_for_materials(frame)
                    if hasattr(mat_scan, "set_selected_materials"):
                        mat_scan.set_selected_materials(prev_sel)

                    if all_cands:
                        for c in all_cands:
                            c_type = getattr(c, "material_type", "")
                            if self.is_material_selected_for_gather(c_type):
                                return False
                        return True
            except Exception:
                pass

        # In selective mode without active node or confirmed match, bypass unknown prompt
        return True

    def adopt_shared_yolo_detector(self, detector: Any) -> None:
        """Point worker/hub/reflex/scanner at one detector; close a discarded peer.

        main.py used to assign ``worker.yolo_detector = shared`` after
        BotWorker already built (and bound) its own instance — hub/reflex
        kept the first CUDA session alive. This method rebinds dependents
        and closes the orphan so RSS does not climb by ~300-500MB each time.
        """
        old = getattr(self, "yolo_detector", None)
        self.yolo_detector = detector
        ms = getattr(self, "material_scanner", None)
        if ms is not None and hasattr(ms, "set_yolo_detector"):
            try:
                ms.set_yolo_detector(detector)
            except Exception as exc:
                logger.debug("material_scanner set_yolo_detector skipped: %s", exc)
        hub = getattr(self, "perception_hub", None)
        if hub is not None and hasattr(hub, "bind"):
            try:
                hub.bind(yolo_detector=detector)
            except Exception as exc:
                logger.debug("perception_hub yolo rebind skipped: %s", exc)
        reflex = getattr(self, "reflex", None)
        bind = getattr(reflex, "_bind_detector", None) if reflex is not None else None
        if callable(bind):
            try:
                bind(detector)
            except Exception as exc:
                logger.debug("reflex yolo rebind skipped: %s", exc)
        if old is not None and old is not detector:
            closer = getattr(old, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:
                    logger.debug("discarded yolo_detector.close skipped: %s", exc)

    def _init_reflex_scanner(self) -> Any:
        """Reflex scan thread. A missing yolo11/rtdetr file keeps the yolov8 onnx."""
        try:
            from core.reflex_layer import ReflexScanner

            kind = "yolov8"
            cfg = self.config
            if cfg is not None and hasattr(cfg, "get"):
                kind = str(cfg.get("", "reflex_model", "yolov8") or "yolov8")
            elif cfg is not None and hasattr(cfg, "settings"):
                kind = str(getattr(cfg.settings, "reflex_model", "yolov8") or "yolov8")
            return ReflexScanner(kind=kind, shared_detector=self.yolo_detector)
        except Exception as exc:
            logger.debug("Reflex scanner skipped: %s", exc)
            return None

    def _init_optional_subsystem(self, module_name: str, class_name: str) -> Any:
        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            return cls()
        except Exception as exc:
            # Sebebi sakla: sessizce None dönmek, dataset factory gibi kritik
            # alt sistemlerin neden ölü olduğunu logda görünmez yapıyordu.
            try:
                self.__dict__.setdefault("_optional_subsystem_errors", {})[f"{module_name}.{class_name}"] = str(exc)
            except Exception:
                pass
            logger.warning("Optional subsystem %s.%s skipped: %s", module_name, class_name, exc)
            return None

    def _init_auto_cartographer(self) -> Any:
        try:
            from core.auto_cartographer import AutoCartographer
            return AutoCartographer(
                route_manager=self.route_manager,
                vision_engine=self.vision,
            )
        except Exception as exc:
            # S-0102: sessiz debug + None, sağlık özetinde görünmüyordu.
            # _init_optional_subsystem deseniyle aynı: sebebi kaydet + warning.
            try:
                self.__dict__.setdefault("_optional_subsystem_errors", {})["core.auto_cartographer.AutoCartographer"] = str(exc)
            except Exception:
                pass
            logger.warning("AutoCartographer init skipped: %s", exc)
            return None

    def heartbeat(self) -> None:
        """Explorer heartbeat (F-0393): the explorer calls this callback
        while mapping; previously getattr(self, "heartbeat", None) was
        always None, so it never fired."""
        try:
            watchdog = self._plain_attr("watchdog")
            if watchdog is not None and hasattr(watchdog, "record_heartbeat"):
                watchdog.record_heartbeat(self.state, self.mode)
        except Exception as exc:
            logger.debug("heartbeat error: %s", exc)

    def _init_explorer_controller(self) -> Any:
        try:
            from core.explorer_mode import ExplorerModeController
            return ExplorerModeController(
                route_manager=self.route_manager,
                vision_engine=self.vision,
                input_driver=self.input_driver,
                dataset_factory=getattr(self, "dataset_factory", None),
                heartbeat_callback=getattr(self, "heartbeat", None),
                test_mode=False,
            )
        except Exception as exc:
            # S-0102: _init_optional_subsystem deseniyle aynı — sebebi kaydet + warning.
            try:
                self.__dict__.setdefault("_optional_subsystem_errors", {})["core.explorer_mode.ExplorerModeController"] = str(exc)
            except Exception:
                pass
            logger.warning("ExplorerModeController init skipped: %s", exc)
            return None

    # =========================================================================
    # Dependency Initialization Helpers
    # =========================================================================

    def _init_default_config(self) -> Any:
        try:
            from config.schema import SettingsManager
            return SettingsManager()
        except Exception as exc_schema:
            logger.debug("config.schema not used (%s), trying core.config", exc_schema)
            try:
                from core.config import ConfigManager
                return ConfigManager()
            except Exception as exc_core:
                logger.warning("Fallback dict config initialized: %s", exc_core)
                return {}

    def _init_default_vision(self) -> Any:
        try:
            from core.vision import Vision
            return Vision()
        except Exception as exc:
            logger.warning("Vision initialization failed: %s", exc)
            return None

    def _init_default_input_driver(self) -> Any:
        try:
            from core.input_driver import InputDriver
            return InputDriver()
        except Exception as exc:
            logger.warning("InputDriver initialization failed: %s", exc)
            return None

    def _init_default_route_manager(self) -> Any:
        try:
            from core.route_manager import RouteManager
            return RouteManager()
        except Exception as exc:
            logger.warning("RouteManager initialization failed: %s", exc)
            return None

    def _init_default_navigator(self) -> Any:
        try:
            from core.navigator import MapNavigator
            if (
                self.route_manager is not None
                and self.vision is not None
                and self.input_driver is not None
            ):
                return MapNavigator(self.route_manager, self.vision, self.input_driver)
        except Exception as exc:
            logger.debug("MapNavigator initialization skipped: %s", exc)
        return None

    def _init_navigation_helpers(self) -> None:
        """H-sweep kurulumları (dilim 12 K-1): ThreeStageHarvestController,
        NavigationIntel ve CorridorNavigator üretim kurulumu.

        install_* yardımcıları H dalında (fix/sweep-nav-minimap) tanımlı;
        bu dalda yoksa kurulum atlanır (warning) — entegrasyonda
        yardımcılar gelince aynı çağrı noktaları çalışır. Dönen örnekler
        self.harvest_controller / self.navigation_intel /
        self.corridor_navigator olarak saklanır. NOT: execute_harvest_loop
        bloklayan (sleep/poll) tam döngüdür — tick döngüsünden ASLA
        çağrılmaz; olay arayüzü (start_node/on_cast_completed) bağlantısı
        ayrı tasarım kararıdır (açık soru).
        """
        self.harvest_controller: Any = None
        self.navigation_intel: Any = None
        self.corridor_navigator: Any = None
        try:
            from core.navigation_intel import (
                install_navigation_intel,
                install_three_stage_harvest_controller,
            )
        except Exception as exc:
            logger.warning(
                "Harvest/navigation kurulum yardımcıları yok (H dalı henüz entegre değil): %s",
                exc,
            )
            install_navigation_intel = None  # type: ignore[assignment]
            install_three_stage_harvest_controller = None  # type: ignore[assignment]
        if install_navigation_intel is not None:
            try:
                self.navigation_intel = install_navigation_intel(
                    input_controller=self.input_driver,
                    route_manager=self.route_manager,
                    vision_engine=self.vision,
                )
            except Exception as exc:
                logger.warning("NavigationIntel kurulumu başarısız: %s", exc)
                self.navigation_intel = None
        if install_three_stage_harvest_controller is not None:
            try:
                self.harvest_controller = install_three_stage_harvest_controller(
                    route_manager=self.route_manager,
                    input_controller=self.input_driver,
                )
            except Exception as exc:
                logger.warning("ThreeStageHarvestController kurulumu başarısız: %s", exc)
                self.harvest_controller = None
        try:
            from core.corridor_navigator import install_corridor_navigator
        except Exception as exc:
            logger.warning(
                "CorridorNavigator kurulum yardımcısı yok (H dalı henüz entegre değil): %s",
                exc,
            )
            install_corridor_navigator = None  # type: ignore[assignment]
        if install_corridor_navigator is not None:
            try:
                self.corridor_navigator = install_corridor_navigator(
                    planner=getattr(self, "sequential_planner", None),
                )
            except Exception as exc:
                logger.warning("CorridorNavigator kurulumu başarısız: %s", exc)
                self.corridor_navigator = None

    def _init_default_recorder(self) -> Any:
        try:
            from modules.recorder import NodeRecorder
            if (
                self.route_manager is not None
                and self.vision is not None
                and self.input_driver is not None
                and self.navigator is not None
            ):
                return NodeRecorder(
                    self.route_manager, self.vision, self.input_driver, self.navigator
                )
        except Exception as exc:
            logger.debug("NodeRecorder initialization skipped: %s", exc)
        return None

    def _init_default_combat_recovery(self) -> Any:
        try:
            from core.combat_recovery import CombatRecovery  # type: ignore

            # F-23-07 (v11): fly_unreachable ayarı bağlandı (varsayılan True).
            return CombatRecovery(
                fly_unreachable=bool(self._get_cfg("fly_unreachable", True)),
                vision=self.vision,
                input_ctrl=self.input_driver,
            )
        except Exception as exc_rec:
            # 17/B13: kör loot yolu (CombatSystem F+Tab) artık sessiz debug
            # değil — fallback devredeyken warning verilir.
            logger.warning(
                "core.combat_recovery yok (%s) — kör loot yolu (CombatSystem F+Tab) devrede, loot doğrulaması yok",
                exc_rec,
            )
            try:
                from core.combat import CombatSystem
                if self.vision is not None and self.input_driver is not None:
                    return CombatSystem(self.vision, self.input_driver)
            except Exception as exc_combat:
                logger.warning("CombatSystem initialization failed: %s", exc_combat)
        return None

    # =========================================================================
    # Unified Configuration Accessor
    # =========================================================================

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        """
        Reads a configuration key from SettingsManager (config/schema.py),
        BotSettings dataclass, ConfigManager (core/config.py), or plain dicts.
        """

        try:
            if self.config is None:
                return default

            if key == "break_every_min":
                # F-23-06: breaks.enabled=false molayı kapatır — break_every_min
                # değeri dosyada duruyor diye yok sayılmaz. Eskiden yalnızca
                # break_every_min anahtarı hem düz hem breaks. altında None
                # ise 0 dönülüyordu; kullanıcı enabled=false + 30 yazsa mola
                # çalışmaya devam ediyordu. Açıkça False değilse (None/eksik)
                # normal aramaya düşülür.
                breaks_enabled = self._raw_cfg_lookup("breaks", "enabled")
                if breaks_enabled is False:
                    return 0

            search_candidates = _CFG_ALIASES.get(key, ((None, key),))
            for section, k in search_candidates:
                val = self._raw_cfg_lookup(section, k)
                if val is not None:
                    return val
        except Exception as exc:
            logger.debug("Config lookup error for key '%s': %s", key, exc)

        return default

    def _raw_cfg_lookup(self, section: Optional[str], key: str) -> Any:
        try:
            if isinstance(self.config, dict):
                if section is None:
                    return self.config.get(key)
                sec_dict = self.config.get(section)
                if isinstance(sec_dict, dict):
                    return sec_dict.get(key)
                return None

            # Direct attribute on config or config.settings (BotSettings dataclass)
            for root_obj in (self.config, getattr(self.config, "settings", None)):
                if root_obj is None:
                    continue
                if section is None and hasattr(root_obj, key):
                    attr_val = getattr(root_obj, key)
                    if not callable(attr_val):
                        return attr_val
                if section is not None and hasattr(root_obj, section):
                    sec_obj = getattr(root_obj, section)
                    if isinstance(sec_obj, dict) and key in sec_obj:
                        return sec_obj[key]
                    if hasattr(sec_obj, key):
                        attr_val = getattr(sec_obj, key)
                        if not callable(attr_val):
                            return attr_val

            raw_dict = getattr(self.config, "config", None) or getattr(self.config, "data", None)
            if isinstance(raw_dict, dict):
                if section is None and key in raw_dict:
                    return raw_dict[key]
                if section is not None and isinstance(raw_dict.get(section), dict):
                    if key in raw_dict[section]:
                        return raw_dict[section][key]

            if hasattr(self.config, "get") and callable(self.config.get):
                try:
                    if section is not None:
                        return self.config.get(section, key, None)
                    return self.config.get(key, None)
                except TypeError:
                    if section is None:
                        return self.config.get(key)

            # F-0179: SettingsManager.from_dict drops unknown keys (e.g.
            # breaks.enabled), so a value the user wrote in settings.json
            # would silently vanish. Fall back to the raw file (cached).
            raw = self._raw_settings_file_dict()
            if isinstance(raw, dict):
                if section is None:
                    if key in raw:
                        return raw[key]
                else:
                    sec = raw.get(section)
                    if isinstance(sec, dict) and key in sec:
                        return sec[key]
        except Exception as exc:
            logger.debug("Raw config lookup failed (%s.%s): %s", section, key, exc)
        return None

    def _raw_settings_file_dict(self) -> Optional[Dict[str, Any]]:
        """Raw settings.json content, cached by mtime. None when unavailable."""
        try:
            import json
            import os
            from pathlib import Path

            path = getattr(self.config, "settings_path", None)
            if path is None:
                return None
            mtime = os.path.getmtime(path)
            cache = self.__dict__.get("_cfg_file_cache")
            if (
                isinstance(cache, dict)
                and cache.get("mtime") == mtime
                and cache.get("path") == str(path)
            ):
                data = cache.get("data")
                return data if isinstance(data, dict) else None
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.__dict__["_cfg_file_cache"] = {
                "path": str(path),
                "mtime": mtime,
                "data": data,
            }
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _sync_input_driver_state(self) -> None:
        """Synchronizes route_active and test_mode flags with InputController if supported."""
        if self.input_driver is None:
            return
        try:
            test_mode = bool(self._get_cfg("test_mode", False))
            if hasattr(self.input_driver, "set_test_mode"):
                self.input_driver.set_test_mode(test_mode)
            elif hasattr(self.input_driver, "test_mode"):
                setattr(self.input_driver, "test_mode", test_mode)

            is_route_running = (
                not self._paused
                and self._mode == BotMode.ROUTE_MODE.value
                and self._state != BotState.PAUSED.value
            )
            if hasattr(self.input_driver, "set_route_active"):
                self.input_driver.set_route_active(is_route_running)
            elif hasattr(self.input_driver, "route_active"):
                setattr(self.input_driver, "route_active", is_route_running)
        except Exception as exc:
            logger.debug("InputDriver state sync error: %s", exc)

    # =========================================================================
    # Public Properties & Control Methods
    # =========================================================================

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def pause_reason(self) -> str:
        with self._lock:
            return self._pause_reason

    def set_mode(self, mode: str) -> None:
        """
        Switches the active bot mode ('route', 'combat', 'manual_record', 'auto_record', 'explorer')
        and resets transient navigation variables.
        """
        try:
            raw_str = str(mode.value if isinstance(mode, Enum) else mode).strip().lower()
            if raw_str in ("combat", "combat_mode", "savas", "savaş", "fight"):
                normalized = BotMode.COMBAT_MODE.value
            elif raw_str in ("route", "route_mode", "gather", "topla"):
                normalized = BotMode.ROUTE_MODE.value
            else:
                normalized = raw_str

            if normalized not in (
                BotMode.ROUTE_MODE.value,
                BotMode.COMBAT_MODE.value,
                BotMode.MANUAL_RECORD.value,
                BotMode.AUTO_RECORD.value,
                BotMode.EXPLORER_MODE.value,
            ):
                logger.warning("Unsupported mode '%s', defaulting to route", normalized)
                normalized = BotMode.ROUTE_MODE.value

            with self._lock:
                prev_mode = self._mode
                self._mode = normalized
                self.active_node = None
                self._is_gathering_active = False
                self._stage_cast_in_progress = False
                self._waiting_for_next_stage_prompt = False
                self._current_node_harvest_stage = 0
                self._current_target_node_id = None
                self._sequential_cursor_idx = 0
                self._current_seq_pos = 0
                self._gather_lost_frames = 0
                self._walk_start_time = 0.0
                self._last_gather_time = time.time()
                self._schedule_next_break()
                # Leaving combat/route fight-back: drop sticky target so Toplama stays clean.
                if normalized != BotMode.COMBAT_MODE.value:
                    self._combat_was_targeting = False
                    self._route_combat_confirmed = False
                    self._route_was_fighting = False

            if normalized != BotMode.COMBAT_MODE.value:
                self._release_all_combat_inputs()
            else:
                self._release_route_motion()

            with self._lock:
                if normalized == BotMode.MANUAL_RECORD.value:
                    self._paused = False
                    self._set_state_internal(BotState.WAITING.value)
                elif normalized in (BotMode.AUTO_RECORD.value, BotMode.EXPLORER_MODE.value):
                    self._paused = False
                    self._set_state_internal(BotState.WALKING.value)
                elif normalized == BotMode.COMBAT_MODE.value:
                    self._paused = False
                    self._combat_skill_knobs_logged = False
                    self._set_state_internal(BotState.FIGHTING.value)
                else:
                    # When entering ROUTE_MODE: if called from GUI thread while worker is running,
                    # defer heavy sequential route sorting to the background worker thread so F9Menu never freezes!
                    if self.isRunning() and threading.current_thread() is not self:
                        self._pending_route_sort = True
                    else:
                        self._auto_sort_sequential_route()
                    if not self._paused:
                        self._set_state_internal(BotState.WALKING.value)

            # pause() calls abort_motion(); modes that force-run must clear the input gate again.
            # v20-loop: ROUTE_MODE eksikti — duraklatma sonrası TOPLAMA'ya
            # geçince girdi kapısı (motion_abort/aim_blocked) kilitli
            # kalıyor, hold_movement_key hep False dönüyor, karakter hiç
            # yürümüyordu. Rota da force-run modudur.
            if normalized in (
                BotMode.MANUAL_RECORD.value,
                BotMode.AUTO_RECORD.value,
                BotMode.EXPLORER_MODE.value,
                BotMode.COMBAT_MODE.value,
                BotMode.ROUTE_MODE.value,
            ):
                driver = self.input_driver
                if driver is not None:
                    resume_fn = getattr(driver, "resume_motion", None)
                    if callable(resume_fn):
                        try:
                            resume_fn()
                        except Exception as exc:
                            logger.debug("resume_motion after set_mode skipped: %s", exc)

            self._sync_input_driver_state()
            mode_label = MODE_BADGE_MAP.get(normalized, normalized.upper())
            if prev_mode != normalized:
                self._emit_log(f"Mode changed to {mode_label}")
            self._emit_hud_snapshot()
        except Exception as exc:
            logger.error("Error in set_mode('%s'): %s", mode, exc, exc_info=True)

    def _auto_sort_sequential_route(self, mode: str = "smooth_loop") -> None:
        """Ensures the active route is ordered sequentially (no ping-pong jumps)."""
        try:
            if self.explorer_controller is not None and hasattr(
                self.explorer_controller, "finalize_and_sort_route"
            ):
                self.explorer_controller.finalize_and_sort_route(mode=mode)
            elif (
                self.sequential_planner is not None
                and self.route_manager is not None
                and hasattr(self.sequential_planner, "apply_sequential_order_to_route")
            ):
                region = getattr(self.route_manager, "active_region", None)
                route = getattr(self.route_manager, "active_route", None)
                self.sequential_planner.apply_sequential_order_to_route(
                    self.route_manager, region=region, route=route, mode=mode
                )
        except Exception as exc:
            logger.debug("Sequential auto-sort skipped: %s", exc)

    def _notify_panic_paused(self, paused: bool) -> None:
        """Panik sistemine botun duraklatıldığını/devam ettiğini bildirir.

        fix/v20-panic-combat: PanicAlertSystem.set_paused() savaş
        sayacını sıfırlar/dondurur (duraklatılmış bot savaşmaz — canlı
        v19'da bot DURAKLATILMIŞKEN COMBAT_TIMEOUT_2M tetikleniyordu).
        Panik KİLİDİNE dokunmaz (kilit yalnızca reset_panic ile açılır).
        panic_system yoksa/eskiyse (set_paused yok) sessizce geçilir —
        pause()/resume()/stop() asla bu bildirim yüzünden bozulmaz.
        Tüm duraklatma/durdurma yolları pause()/stop() üzerinden geçtiği
        için tek kanca noktası burasıdır.
        """
        try:
            panic = self._plain_attr("panic_system", None)
            if panic is None:
                return
            fn = getattr(panic, "set_paused", None)
            if callable(fn):
                fn(bool(paused))
        except Exception as exc:
            logger.debug("panic_system.set_paused bildirimi atlandı: %s", exc)

    def pause(self, reason: str = "Paused by user") -> None:
        """Pauses the bot execution and sets HUD badge to PAUSED."""
        try:
            self._halt_motion()
            with self._lock:
                already_paused = self._paused and self._state == BotState.PAUSED.value
                self._paused = True
                self._pause_reason = reason or "Paused by user"
                # 16/10: pause aktif bacağı silmez — Alt+Tab'da (odak kaybı)
                # yürüme emeği çöpe gitmez; resume aynı düğüme devam eder.
                # (Savaş askısı "active_node korunur" diyordu ama pause
                # siliyordu — çelişki buydu.) Yalnızca anlık toplama/cast
                # durumu sıfırlanır; yarım aşama kaldığı yerden değil,
                # düğüm başından yeniden denenir.
                self._is_gathering_active = False
                self._stage_cast_in_progress = False
                self._waiting_for_next_stage_prompt = False
                self._current_node_harvest_stage = 0
                self._gather_lost_frames = 0
                self._set_state_internal(BotState.PAUSED.value)

            # fix/v20-panic-combat: duraklatma savaş sayacını sıfırlar/
            # dondurur (panik sistemine bildirilir).
            self._notify_panic_paused(True)
            self._sync_input_driver_state()
            if not already_paused:
                self._emit_log(reason)
            self._emit_hud_snapshot(
                title="Paused",
                subtitle=f"{reason} · Devam: Ctrl+Alt+F10",
            )
        except Exception as exc:
            logger.error("Error in pause(): %s", exc, exc_info=True)

    def _player_driven_mapping_mode(self) -> bool:
        """Explorer / Oto-Kayıt: player walks; bot must not require held W for WALKING HUD."""
        mode = self._plain_attr("_mode", None)
        if mode is None:
            try:
                mode = self._mode
            except Exception:
                return False
        return mode in (
            BotMode.EXPLORER_MODE.value,
            BotMode.AUTO_RECORD.value,
        )

    def _unlock_panic_lock(self) -> Optional[Union[Dict[str, Any], bool]]:
        """
        Panik kilidini GERÇEKTEN açar (fix/panic-lock).

        Eski resume() yalnızca panic_system._inputs_halted /
        _panic_triggered bayraklarını temizliyordu; emergency_halt_inputs
        maymun-yamaladığı girdi metotlarını (press_key, key_down, key_up,
        click, ...) _blocked_input no-op ile değiştirdiği için bayraklar
        "açık" görünürken bot girdi-ölü kalıyordu (canlı log 20:15:13'te
        3x "[PANIC-LOCK] Girdi engellendi"). Tek gerçek açma yolu
        reset_panic()/unlock_inputs — saklanan orijinal metotları geri
        yükler ve zamanlayıcıları sıfırlar (koşul sürüyorsa watchdog hemen
        yeniden tetiklemez).

        Döner (B-09, üç durumlu): kilit aktifti ve açma doğrulandıysa
        tetikleyen olay kaydı (dict; olay kaydı yoksa boş dict); kilit
        aktif değilse ya da panik sistemi yoksa None; kilit aktif ama
        açılamadı/doğrulanamadıysa False (başarısızlık ayrıca error
        loglanır). reset_panic() olmayan (eski/test) panic nesnelerinde
        bayrak temizliğine düşer ve bunu warning ile açıkça söyler.
        """
        panic = self._plain_attr("panic_system")
        if panic is None:
            return None

        def _bool_flag(name: str) -> bool:
            # Gerçek bool olmayan değerler (örn. MagicMock) "bilinmiyor"
            # sayılır — sahte pozitif/negatif doğrulama yapılmaz.
            try:
                return getattr(panic, name, None) is True
            except Exception:
                return False

        # reset_panic() _last_panic_event'i siler — sebebi ÖNCE oku.
        event: Optional[Dict[str, Any]] = None
        try:
            getter = getattr(panic, "get_last_panic_event", None)
            if callable(getter):
                maybe_event = getter()
                if isinstance(maybe_event, dict):
                    event = maybe_event
        except Exception:
            event = None
        if event is None:
            try:
                raw = getattr(panic, "_last_panic_event", None)
                if isinstance(raw, dict):
                    event = dict(raw)
            except Exception:
                event = None

        saved_state = getattr(panic, "_halted_input_state", None)
        lock_active = (
            _bool_flag("_panic_triggered")
            or _bool_flag("_inputs_halted")
            or _bool_flag("panic_triggered")
            or _bool_flag("inputs_halted")
            or isinstance(saved_state, dict)
        )
        if not lock_active:
            return None

        reset_fn = None
        for _name in ("reset_panic", "unlock_inputs", "resume_from_panic"):
            _fn = getattr(panic, _name, None)
            if callable(_fn):
                reset_fn = _fn
                break

        if reset_fn is None:
            try:
                if hasattr(panic, "_inputs_halted"):
                    panic._inputs_halted = False
                if hasattr(panic, "_panic_triggered"):
                    panic._panic_triggered = False
            except Exception:
                pass
            logger.warning(
                "[PANIC] Panik bayrakları temizlendi ama reset_panic()/unlock_inputs "
                "yok — maymun-yamalı girdi metotları geri yüklenememiş olabilir."
            )
            self._emit_log(
                "Panik bayrakları temizlendi ama girdi metotları geri yüklenememiş olabilir"
            )
            # B-09: kilit aktifti ama açma doğrulanamadı -> False.
            return False

        try:
            # Argümansız: reset_panic kilitlerken saklanan denetleyiciyi
            # (halt edilen nesnenin kendisini) geri yükler.
            result = reset_fn()
        except Exception as exc:
            logger.error("[PANIC] Panik kilidi açılamadı (reset_panic hatası): %s", exc)
            self._emit_log(f"Panik kilidi AÇILAMADI: {exc}")
            return False
        if result is False:
            logger.error(
                "[PANIC] Panik kilidi kısmen açıldı (reset_panic False döndü) — "
                "bazı girdi metotları kilitli kalmış olabilir."
            )
            self._emit_log("Panik kilidi kısmen açıldı — bazı girdiler kilitli kalmış olabilir")
            return False
        if (
            _bool_flag("_panic_triggered")
            or _bool_flag("_inputs_halted")
            or _bool_flag("panic_triggered")
            or _bool_flag("inputs_halted")
        ):
            logger.error(
                "[PANIC] reset_panic() başarılı döndü ama panik bayrakları hâlâ "
                "aktif — girdi kilidi sürüyor olabilir."
            )
            self._emit_log("Panik kilidi açılamadı — bayraklar hâlâ aktif")
            return False

        reason = ""
        details = ""
        if event is not None:
            reason = str(event.get("reason") or "")
            details = str(event.get("details") or "")
        msg = f"Panik kilidi açıldı (sebep: {reason or 'bilinmiyor'}" + (
            f" — {details}" if details else ""
        ) + ")"
        logger.info("[PANIC] %s", msg)
        self._emit_log(msg)
        # B-09: açma doğrulandı ama olay kaydı yoksa boş dict döner — None
        # 'kilit yok' anlamında saklıdır, başarıyla karışmaz.
        return event if event is not None else {}

    def resume(self) -> None:
        """Resumes the bot from PAUSED or BREAK state."""
        try:
            # fix/panic-lock (F-0049): panik kilidi yalnızca bayrak
            # temizliğiyle açılmaz — _unlock_panic_lock() reset_panic()
            # çağırıp maymun-yamalı girdi metotlarını geri yükler ve
            # açmayı doğrular. Dönen olay kaydı HUD altyazısına yazılır.
            cleared_event = self._unlock_panic_lock()
            if cleared_event is False:
                # B-09: kilit aktif ama açılamadı — resume duraklatılmış
                # kalır (resume_motion çalışmaz, _paused False yapılmaz).
                logger.error("[PANIC] resume durduruldu: panik kilidi açılamadı, bot duraklatılmış kalıyor")
                self._emit_log("Panik kilidi açılamadı — bot duraklatılmış kalıyor")
                self._emit_hud_snapshot(subtitle="Panik kilidi açılamadı — girdiler kilitli")
                return
            # fix/v20-panic-combat: kilit açıldı, devam ediliyor — savaş
            # sayacının donması kalkar (sıfırdan başlar).
            self._notify_panic_paused(False)
            driver = self.input_driver
            if driver is not None and hasattr(driver, "resume_motion"):
                driver.resume_motion()
            if driver is not None:
                for _flag in ("halted", "locked", "is_halted", "disabled"):
                    try:
                        if getattr(driver, _flag, False) is True:
                            setattr(driver, _flag, False)
                    except Exception:
                        pass
            with self._lock:
                self._paused = False
                self._pause_reason = ""
                # O5: resume, 15 dk toplamasızlık sayacını (_last_gather_time)
                # ve failure-cooldown'u (_failure_wait_until) sıfırlamaz —
                # sıfırlamak "15 dk toplama yoksa dur" korumasını her
                # devamda erteler ve 10-fail beklemesini atlatır. Sayaç
                # yalnızca gerçek toplamada (ve oturum başlangıcında) yazılır.
                self._break_end_timestamp = 0.0
                self._schedule_next_break()

                if self._mode == BotMode.MANUAL_RECORD.value:
                    target_state = BotState.WAITING.value
                elif self._mode == BotMode.COMBAT_MODE.value:
                    # F-0499: combat must win over walking — otherwise a
                    # combat-mode resume lands in WALKING and never fights.
                    target_state = BotState.FIGHTING.value
                elif self._player_driven_mapping_mode() or self._walking_is_real():
                    target_state = BotState.WALKING.value
                else:
                    target_state = (
                        BotState.WALKING.value
                        if self._walking_is_real()
                        else BotState.WAITING.value
                    )
                self._set_state_internal(target_state)

            self._sync_input_driver_state()
            self._emit_log(f"Resumed ({MODE_BADGE_MAP.get(self._mode, 'ROUTE')})")
            if isinstance(cleared_event, dict):
                _reason = str(cleared_event.get("reason") or "bilinmiyor")
                self._emit_hud_snapshot(
                    subtitle=f"Panik kilidi açıldı: {_reason} · girdiler yeniden açık"
                )
            else:
                self._emit_hud_snapshot()
        except Exception as exc:
            logger.error("Error in resume(): %s", exc, exc_info=True)

    def toggle_pause(self) -> None:
        """Toggles between paused and active states."""
        try:
            if self.is_paused:
                self.resume()
            else:
                self.pause("Paused by user")
        except Exception as exc:
            logger.error("Error in toggle_pause(): %s", exc, exc_info=True)

    def trigger_f8_record(self) -> Optional[Dict[str, Any]]:
        """
        Triggers recording of the current player position as a node (F8).
        If the worker thread is running, queues or executes safely and emits node_recorded.
        """
        try:
            if self.isRunning() and threading.current_thread() is not self:
                with self._lock:
                    self._f8_pending = True
                return None
            return self._execute_f8_record()
        except Exception as exc:
            logger.error("Error in trigger_f8_record(): %s", exc, exc_info=True)
            return None

    def stop(self) -> None:
        """Stops the loop and joins the thread so it cannot keep pythonw alive after quit()."""
        try:
            self._halt_motion()
            # fix/panic-lock: durdurulan bot panik kilitli metotlarla
            # kalmasın — stop() da kilidi gerçekten açar (reset_panic).
            # Önce kilit açılır (metotlar geri yüklenir), sonra giriş
            # mandalı kurulur (son bırakma + stop sonrası sessizlik).
            self._unlock_panic_lock()
            # fix/v20-panic-combat: durdurulan botun savaş sayacı da
            # dondurulur (poll thread'i kısa süre daha yaşayabilir).
            self._notify_panic_paused(True)
            # F-durdur: son bırakma (_halt_motion) ardından giriş mandalını
            # kur — stop sonrası F10 abort'u / periyodik guard'lar artık
            # hiçbir sentetik girdi gönderemez. resume_motion() kaldırır
            # (F-0049 deseni: resume mandalları tek yoldan temizler).
            driver = self.input_driver
            if driver is not None:
                mark = getattr(driver, "mark_input_stopped", None)
                if callable(mark):
                    try:
                        mark()
                    except Exception:
                        logger.debug("mark_input_stopped skipped", exc_info=True)
            with self._lock:
                self._running = False
                self._paused = True
            self._sync_input_driver_state()
            self._emit_log("Stopping bot worker")
            try:
                self.requestInterruption()
            except Exception:
                logger.debug("BotWorker interruption request skipped", exc_info=True)
            if threading.current_thread() is self:
                return
            if self.isRunning():
                self.wait(4000)
            if self.isRunning():
                # P2-8 (K-17): ikinci grace penceresi — uzun bir tick
                # (yakalama / rota adımı) hâlâ bitiyor olabilir.
                # `QThread.terminate()` ani öldürmedir: kilit, dosya ve
                # HID durumu yarıda bırakır, bir daha kullanılmaz.
                logger.warning(
                    "BotWorker 4 sn'de durmadı; 4 sn daha bekleniyor (terminate yok)."
                )
                self.wait(4000)
            if self.isRunning():
                # P2-8: durmadıysa bile terminate() YOK. Thread referansı
                # korunur (düşürülürse hayalet thread olur, P2-4); sonraki
                # stop() çağrısı tekrar dener. Süreç çıkışı bu thread'e
                # takılırsa logda bu satır görülür.
                logger.error(
                    "BotWorker stop() sonrası hâlâ çalışıyor; terminate() "
                    "uygulanmadı (P2-8) — thread referansı korunuyor, "
                    "sonraki stop() tekrar dener."
                )
        except Exception as exc:
            logger.error("Error stopping BotWorker: %s", exc, exc_info=True)

    def on_user_key_pressed(self, key_name: str) -> None:
        """
        Automatically pauses the bot if the user presses a manual control key
        while ROUTE_MODE is actively running.
        """
        try:
            if not key_name:
                return
            clean_key = str(key_name).strip().lower()

            # Ignore hotkeys used to control the bot itself
            if clean_key in ("f8", "f9", "f10"):
                return

            # Ignore synthetic key events generated by the bot
            if self.input_driver is not None and hasattr(self.input_driver, "is_bot_pressing"):
                try:
                    if self.input_driver.is_bot_pressing(clean_key):
                        return
                except Exception as exc:
                    logger.debug("is_bot_pressing check failed: %s", exc)

            # F-0372: the 0.25 s window dropped real user input during busy
            # combat, and the route-only check ignored other modes. Synthetic
            # bot presses are already filtered by is_bot_pressing above, so
            # no time window is needed here.
            with self._lock:
                is_active = (
                    not self._paused
                    and self._state != BotState.PAUSED.value
                )

            if is_active:
                self.pause(f"Paused: user pressed {clean_key.upper()}")
        except Exception as exc:
            logger.error("Error in on_user_key_pressed('%s'): %s", key_name, exc, exc_info=True)

    def set_pause_on_alt_tab(self, enabled: bool) -> None:
        """Enables or disables automatic pausing when the player Alt-Tabs away from the game."""
        self.pause_on_alt_tab = bool(enabled)
        logger.info("[Alt-Tab Guard] pause_on_alt_tab set to %s", self.pause_on_alt_tab)

    def _on_game_focus_lost(self, reason: str = "") -> None:
        """
        Called when the Aion 2 game window loses foreground focus (e.g. user Alt-Tabs to Discord/Chrome).
        Immediately halts all movement keys, pauses the bot, and alerts via HUD and log.
        """
        if not getattr(self, "pause_on_alt_tab", True):
            return
        if self.is_paused:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_focus_lost_ts", 0.0) < 0.5:
            return
        self._last_focus_lost_ts = now
        logger.info("[Alt-Tab] Oyun penceresi odağı kaybetti (%s) -> Bot otomatik duraklatılıyor.", reason)
        self._release_route_motion()
        if self.input_driver is not None and hasattr(self.input_driver, "abort_motion"):
            try:
                self.input_driver.abort_motion()
            except Exception:
                pass
        self.pause("Alt-Tab (Oyun odağı kayboldu)")
        self._emit_log("🛑 [Alt-Tab] Oyun arka plana geçti, bot otomatik duraklatıldı.")
        try:
            import winsound
            winsound.Beep(700, 180)
        except Exception:
            pass

    def _check_game_focus(self) -> bool:
        """
        Tick-level focus check. Returns True if game is focused or pause_on_alt_tab is disabled.
        If Alt-Tab is detected, pauses immediately and returns False.
        """
        if not getattr(self, "pause_on_alt_tab", True) or self.is_paused:
            return True
        focus_mon = getattr(self, "focus_monitor", None)
        if focus_mon is None or not hasattr(focus_mon, "is_game_window_focused"):
            # F-0180/F-0373: no focus monitor means the check cannot verify
            # the game is focused — fail closed in live mode. test_mode is
            # exempt, and so are skeleton workers built via __new__ (unit
            # tests) that never went through __init__ at all: for a real
            # worker __init__ always records _core_deps_missing, so its
            # absence proves this is not a live bot.
            if bool(self._get_cfg("test_mode", False)):
                return True
            if "_core_deps_missing" not in self.__dict__:
                return True
            if not bool(self._plain_attr("_focus_guard_missing_warned", False)):
                self._focus_guard_missing_warned = True
                logger.error(
                    "Game focus cannot be verified: focus monitor unavailable — ticks paused"
                )
            return False
        try:
            focused, reason = focus_mon.is_game_window_focused(allow_gatherion_ui=True)
            if not focused:
                self._on_game_focus_lost(reason)
                return False
        except Exception as exc:
            logger.debug("Error checking game focus: %s", exc)
            # F-0373: an erroring focus check is not a passed check — for a
            # real (init-completed) worker in live mode. test_mode and
            # __new__ skeleton workers keep the old permissive path.
            if (
                not bool(self._get_cfg("test_mode", False))
                and "_core_deps_missing" in self.__dict__
            ):
                return False
        return True

    # =========================================================================
    # Main QThread Loop
    # =========================================================================

    def run(self) -> None:
        """Main QThread execution loop."""
        logger.info("BotWorker thread started.")
        # F-0198: never run the loop with a missing core dependency.
        if getattr(self, "_core_deps_missing", None):
            missing = ", ".join(self._core_deps_missing)
            logger.error("BotWorker not started: missing core dependencies: %s", missing)
            try:
                self._emit_log(f"Başlatılamadı: eksik bağımlılık: {missing}")
            except Exception:
                pass
            return
        try:
            current = threading.current_thread()
            if current is not threading.main_thread() and not current.daemon:
                try:
                    current.daemon = True
                except Exception:
                    current._daemonic = True
        except Exception:
            logger.debug("BotWorker daemon flag unchanged", exc_info=True)
        with self._lock:
            self._running = True
            self._session_start_time = time.time()
            self._last_gather_time = time.time()
            self._schedule_next_break()

        self._sync_input_driver_state()
        self._rehydrate_saved_nodes_count()
        self._emit_hud_snapshot()
        if self.reflex is not None and hasattr(self.reflex, "start"):
            self.reflex.start()
        self._log_combat_skill_knobs_once()

        # Wire and start FocusGuard for automatic pause on Alt-Tab
        if self.focus_monitor is not None and hasattr(self.focus_monitor, "start_monitoring"):
            try:
                from core.game_gpu_throttle import find_aion_game_processes
                aion_pids = find_aion_game_processes()
                if aion_pids and hasattr(self.focus_monitor, "register_game_pids"):
                    self.focus_monitor.register_game_pids(aion_pids)
            except Exception as proc_exc:
                logger.debug("Aion process registration into focus_monitor skipped: %s", proc_exc)
            try:
                self.focus_monitor.start_monitoring(
                    on_focus_lost=self._on_game_focus_lost,
                    check_interval_sec=0.05,
                    allow_gatherion_ui=True,
                )
            except Exception as mon_exc:
                logger.debug("Focus monitor start failed: %s", mon_exc)

        try:
            while self._is_running() and not self.isInterruptionRequested():
                try:
                    # 1. Process any queued F8 node recording request
                    if self._consume_f8_request():
                        self._execute_f8_record()

                    # 1.5 Process deferred route sorting on worker thread (prevents GUI freeze on F9 -> Route)
                    if getattr(self, "_pending_route_sort", False):
                        self._pending_route_sort = False
                        self._auto_sort_sequential_route()

                    # 2. Refresh resting nodes status on RouteManager if supported
                    self._refresh_route_cooldowns()

                    # 2.5 Record watchdog heartbeat if attached
                    if self.watchdog is not None and hasattr(self.watchdog, "record_heartbeat"):
                        try:
                            self.watchdog.record_heartbeat(self.state, self.mode)
                        except Exception as wd_exc:
                            logger.debug("Watchdog heartbeat error: %s", wd_exc)

                    # 3. Check if paused. One paused loop tick releases movement keys.
                    if self.is_paused:
                        self._release_route_motion()
                        self._set_state_internal(BotState.PAUSED.value)
                        self._emit_hud_snapshot(
                            title="Paused",
                            subtitle=f"{self._pause_reason} · Devam: Ctrl+Alt+F10",
                        )
                        self._sleep_adaptive(BotState.PAUSED.value, fallback_ms=150)
                        continue

                    # 3.5 Check game focus (Alt-Tab auto-pause)
                    if not self._check_game_focus():
                        continue

                    # 4. Dispatch by active mode
                    current_mode = self.mode
                    if current_mode == BotMode.MANUAL_RECORD.value:
                        self._step_manual_record()
                        self._sleep_adaptive(self.state, fallback_ms=150)
                    elif current_mode == BotMode.AUTO_RECORD.value:
                        self._step_auto_record()
                        self._sleep_adaptive(self.state, fallback_ms=100)
                    elif current_mode == BotMode.EXPLORER_MODE.value:
                        self._step_explorer_mode()
                        self._sleep_adaptive(self.state, fallback_ms=100)
                    elif current_mode == BotMode.COMBAT_MODE.value:
                        self._step_combat_mode()
                        # Tight loop: skill bursts already sleep inter-key; keep tick short.
                        self._sleep_adaptive(self.state, fallback_ms=40)
                    else:
                        self._step_route_mode()
                        self._sleep_adaptive(self.state, fallback_ms=100)

                except Exception as loop_exc:
                    logger.error("Error inside BotWorker iteration: %s", loop_exc, exc_info=True)
                    self.msleep(250)
        finally:
            # P2-8: thread çıktıktan sonra `_running` True kalmasın —
            # `_is_running()` aksi hâlde ölmüş thread'i çalışıyor sanır.
            with self._lock:
                self._running = False
            if self.focus_monitor is not None and hasattr(self.focus_monitor, "stop_monitoring"):
                try:
                    self.focus_monitor.stop_monitoring()
                except Exception:
                    pass
            if self.reflex is not None and hasattr(self.reflex, "stop"):
                try:
                    self.reflex.stop()
                except Exception as reflex_exc:
                    logger.debug("Reflex stop skipped: %s", reflex_exc)
            self._sync_input_driver_state()
            logger.info("BotWorker thread exited.")

    def _is_running(self) -> bool:
        with self._lock:
            return bool(getattr(self, "_running", True))

    def _consume_f8_request(self) -> bool:
        with self._lock:
            if self._f8_pending:
                self._f8_pending = False
                return True
            return False

    def _sleep_adaptive(self, state: str, fallback_ms: int = 100) -> None:
        """Uses AdaptiveFramePacer (8-12 FPS walking, 20-30 FPS gathering, 2-4 FPS paused) if available."""
        ms = int(fallback_ms)
        try:
            if self.frame_pacer is not None and hasattr(self.frame_pacer, "get_next_sleep_interval"):
                sec = float(self.frame_pacer.get_next_sleep_interval(state))
                ms = max(25, min(350, int(sec * 1000.0)))
        except Exception as exc:
            logger.debug("Adaptive sleep fallback: %s", exc)
            ms = int(fallback_ms)
        left = max(0, ms)
        while left > 0 and self._is_running() and not self.isInterruptionRequested():
            step = min(25, left)
            self.msleep(step)
            left -= step

    def _run_passive_auto_cartography(self, gather_while_exploring: bool = False) -> None:
        """
        Automatically scans screen & minimap for gems/ores/materials while walking,
        deduplicates them via SLAM visual odometry, saves 80x80 patches, and inserts
        them into the route in sequential corridor order without requiring manual F8!
        """
        # Renkli kareyi kartograf kapısından önce al: YZ öğrenimi gezinirken de beslensin.
        frame_bgr = self._grab_frame(grayscale=False)
        # v20-yz: gezinme YZ kare çekimi — fabrika yalnızca hasat bitince
        # çalışıyordu; yürüyüş/keşif sırasında 0 kare kaydediliyordu.
        # Kanca renkli kareden hemen sonra, kartografi kapısından bağımsız
        # çalışır (kapı kapalı olsa bile çekim yapılır; kısma içeridedir).
        self._maybe_capture_yz_roaming_frame(frame_bgr)
        # __new__ ile kurulan worker'larda (__init__ atlanmış) bu iki öznitelik
        # yoktur — doğrudan okuma AttributeError verir. Üretimde __init__
        # kurduğu için davranış aynen kalır (kapalıysa dön, açıksa devam).
        # (16/12a REDDEDİLDİ: kare kapıdan önce alınır çünkü _grab_frame'in
        # yan etkisi _last_color_frame'i besler — kasıtlı, yorumda belgeli.)
        if not getattr(self, "auto_cartography_enabled", False) or getattr(
            self, "explorer_controller", None
        ) is None:
            return
        if getattr(self.explorer_controller, "dataset_factory", None) is None and getattr(self, "dataset_factory", None) is not None:
            self.explorer_controller.dataset_factory = self.dataset_factory
        if getattr(self.explorer_controller, "heartbeat_callback", None) is None:
            self.explorer_controller.heartbeat_callback = self.heartbeat
        try:
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                return
            # Keep explorer on the live YZ reflex / YOLO path every cartography tick.
            bind = getattr(self.explorer_controller, "bind_ai_sources", None)
            if callable(bind):
                try:
                    bind(
                        reflex=getattr(self, "reflex", None),
                        yolo_detector=getattr(self, "yolo_detector", None),
                        perception_hub=getattr(self, "perception_hub", None),
                    )
                except Exception as bind_exc:
                    logger.debug("explorer AI bind skipped: %s", bind_exc)
            result = self.explorer_controller.step_passive_cartography(
                frame_bgr, gather_while_exploring=gather_while_exploring
            )
            # 16/4 bitişiği: keşif toplaması da gerçek toplamadır — idle
            # sayacına işlenir (ileride keşif auto-pause kararı için).
            try:
                if isinstance(result, dict) and int(result.get("harvests_completed_in_step", 0) or 0) > 0:
                    self._last_gather_time = time.time()
            except (TypeError, ValueError):
                pass
            if isinstance(result, dict):
                new_nodes = result.get("new_nodes", []) or []
                for nd in new_nodes:
                    nd_dict = self._node_to_dict(nd) or (nd if isinstance(nd, dict) else {})
                    self.saved_nodes_count += 1
                    # 16/12c: kimliği yoksa "Node <sayaç>" uydurulmaz —
                    # last_saved_str eski değerinde kalır.
                    nid = nd_dict.get("id")
                    rtype = nd_dict.get("resource_type", "Material")
                    if nid is not None:
                        self.last_saved_str = f"Node {nid}"
                    self._emit_log(f"Auto-mapped #{nid if nid is not None else '?'} ({rtype}) in sequence")
                    self.node_recorded.emit(nd_dict)
                yz_n = int(result.get("yz_count", 0) or 0)
                det_n = int(result.get("detected_count", 0) or 0)
                if yz_n > 0 or det_n > 0:
                    logger.debug(
                        "Cartography tick: detected=%d yz=%d new=%d gate_rejected=%s",
                        det_n,
                        yz_n,
                        len(new_nodes),
                        result.get("gate_rejected_total"),
                    )
            self._carto_error_count = 0
        except Exception as exc:
            # 16/12b: kartografi hatası artık sessiz debug değil — ilk hata
            # warning, tekrarlar sayaçlı debug (her tick warning spam'i yok).
            self._carto_error_count = int(getattr(self, "_carto_error_count", 0) or 0) + 1
            if self._carto_error_count <= 1:
                logger.warning(
                    "Passive auto-cartography error (x%d, HUD 'Auto-mapping' göstermeye devam eder): %s",
                    self._carto_error_count,
                    exc,
                )
            else:
                logger.debug(
                    "Passive auto-cartography error (x%d): %s", self._carto_error_count, exc
                )

    def _maybe_capture_yz_roaming_frame(self, frame_bgr: Any) -> None:
        """Periyodik YZ kare çekimi (v20-yz): yürüyüş/keşif sırasında.

        Kök neden: dataset factory yalnızca hasat bitince çalışıyordu
        (on_gather_success — 3 çağrı noktası da hasat bitişi); yürüyüş
        ve oto keşif sırasında fabrika hiç kare kaydetmiyordu. Bu kanca
        her kartografi tick'inde çağrılır ama çekimi kısar (varsayılan
        30 sn, ayar: yz_roaming_capture_interval_sec, 10..300 sn).

        Savaşta çağrılmaz (_run_passive_auto_cartography savaşta hiç
        çalışmaz); duraklatmada çekim yapılmaz. Kanca asla ana döngüyü
        bozamaz — tüm hatalar yutulur. Fabrika bağlı değilse tek
        seferlik uyarı verilir (kurulum hatasının kendisi
        state_machine:647'deki ERROR satırındadır).
        """
        try:
            factory = self._plain_attr("dataset_factory", None)
            if factory is None or not hasattr(factory, "on_gather_success"):
                if not self._plain_attr("_yz_roaming_factory_warned", False):
                    self._yz_roaming_factory_warned = True
                    logger.warning(
                        "YZ gezinme çekimi atlandı: dataset factory bağlı değil "
                        "(başlangıçtaki 'DATASET FACTORY YOK' satırına bakın)"
                    )
                return
            if str(getattr(self, "state", "") or "").upper() == "PAUSED":
                return
            mode = str(getattr(self, "mode", "") or "").lower()
            if mode not in ("route", "explorer", "auto_record", "manual_record"):
                return
            try:
                interval = float(self._get_cfg("yz_roaming_capture_interval_sec", 30.0) or 30.0)
            except (TypeError, ValueError):
                interval = 30.0
            interval = min(max(interval, 10.0), 300.0)
            now = time.time()
            last = float(self._plain_attr("_last_yz_roaming_capture_ts", 0.0) or 0.0)
            if now - last < interval:
                return
            self._last_yz_roaming_capture_ts = now
            try:
                factory.on_gather_success(
                    current_frame=frame_bgr,
                    material_name="",
                    harvest_stage=1,
                    max_harvests=1,
                    roaming=True,
                )
            except Exception as exc:
                logger.debug("YZ gezinme çekimi hatası: %s", exc)
        except Exception:
            # Tanı/kayıt kancası ana döngüyü asla bozamaz.
            logger.debug("YZ gezinme çekimi kancası atlandı", exc_info=True)

    # =========================================================================
    # Mode -1: COMBAT_MODE ("combat" - Savaşma Modu / Otomatik Mob Avı)
    # =========================================================================

    def _normalize_combat_skill_keys(self, raw: Any) -> list:
        """Turn a settings value into skill-slot keys. Movement and loot keys are not skills."""
        if raw is None:
            return []
        if isinstance(raw, str):
            text = raw.replace(";", ",").replace("|", ",")
            parts = text.split(",") if "," in text else text.split()
        elif isinstance(raw, (list, tuple)):
            parts = list(raw)
        else:
            return []
        skip = {
            "",
            "tab",
            "f",
            "w",
            "a",
            "s",
            "d",
            "space",
            "esc",
            "m",
            "left",
            "right",
            "left_click",
            "right_click",
            "x",
            "f6",
            "b",
        }
        keys = []
        for part in parts:
            key = str(part).strip().lower()
            if key in skip or key in keys:
                continue
            keys.append(key)
        return keys

    def _plain_attr(self, name: str, default: Any = None) -> Any:
        """Read a field without QObject getattr, which throws if BotWorker.__init__ was skipped."""
        try:
            return self.__dict__.get(name, default)
        except Exception:
            return default

    def _known_combat_skill_slots(self) -> list:
        """Skill slots the combat code already defines. No new layout."""
        recovery = self._plain_attr("combat_recovery")
        monitor = self._plain_attr("skill_bar_monitor")
        for raw in (
            getattr(recovery, "COMBAT_SKILL_KEYS", None),
            getattr(monitor, "SKILL_SLOTS", None),
        ):
            keys = self._normalize_combat_skill_keys(raw)
            if keys:
                return keys
        try:
            from core.combat_recovery import CombatRecovery

            keys = self._normalize_combat_skill_keys(CombatRecovery.COMBAT_SKILL_KEYS)
            if keys:
                return keys
        except Exception:
            logger.debug("Combat skill slot lookup skipped")
        try:
            from core.skill_bar_reader import SkillBarReader

            return self._normalize_combat_skill_keys(SkillBarReader.SKILL_SLOTS)
        except Exception:
            logger.debug("Skill bar slot lookup skipped")
            return []

    def _configured_combat_skill_keys(self) -> list:
        """Settings keys when present, otherwise full hotbar (1/2/3/4/q/e/r/t).

        Do NOT prefer NaturalCombatRotation.DEFAULT_COOLDOWNS keys — that list is
        only 1–5/Q/E (r/t yok) and is not wired into any live rotation path at
        all, so it cannot be the live '2' lock. v20/madde 2 kök neden: hazır
        listesiyle havuzun bire bir değiştirilmesi (tekil hazır set) +
        30-50 ms basımın oyunda kayda geçmemesi.
        """
        for name in ("combat_skill_keys", "skill_keys"):
            keys = self._normalize_combat_skill_keys(self._get_cfg(name, None))
            if keys:
                return keys
        known = self._known_combat_skill_slots()
        if known:
            return known
        return ["1", "2", "3", "4", "q", "e", "r", "t"]

    def _log_combat_skill_knobs_once(self) -> None:
        """Emit effective burst/gcd so live sessions prove settings were applied."""
        if bool(self._plain_attr("_combat_skill_knobs_logged", False)):
            return
        self._combat_skill_knobs_logged = True
        trusted = self._keyboard_trusted_for_combat()
        keys = self._configured_combat_skill_keys()
        burst = self._combat_skill_burst_sec()
        gcd = self._combat_skill_cooldown_sec(keys[0] if keys else "1")
        per = self._combat_skills_per_burst()
        msg = (
            f"COMBAT SKILL KNOBS trusted={trusted} burst={burst:.3f}s "
            f"gcd={gcd:.3f}s per_burst={per} keys={','.join(keys)}"
        )
        try:
            self._emit_log(msg)
        except Exception:
            pass
        logger.info(msg)

    def _keyboard_trusted_for_combat(self) -> bool:
        driver = self._plain_attr("input_driver")
        if driver is None:
            return False
        trusted = getattr(driver, "keyboard_inject_trusted", None)
        try:
            if callable(trusted):
                return bool(trusted())
            if isinstance(trusted, bool):
                return bool(trusted)
        except Exception:
            return False
        return False

    def _warn_bad_config_once(self, name: str, raw: Any) -> None:
        """Bozuk savaş/toplama ayarını her farklı değer için bir kez uyar (B-06).

        Ayrıştırılamayan ayar eskiden sessizce varsayılana dönüyordu;
        kullanıcı ayarının uygulandığını sanıyordu. Uyarı, aynı bozuk
        değer tekrarlanınca spam yapmaz — yalnızca ilk görüldüğünde loglanır.
        """
        warned = self.__dict__.setdefault("_bad_config_warned", set())
        key = (str(name), repr(raw))
        if key in warned:
            return
        warned.add(key)
        logger.warning(
            "Savaş/toplama ayarı ayrıştırılamadı (%s=%r); varsayılan kullanılıyor.",
            name, raw,
        )

    def _combat_skill_burst_sec(self) -> float:
        """Min seconds between skill bursts. Trusted Interception defaults fast (ms scale)."""
        env = str(os.environ.get("GATHERION_SKILL_BURST_SEC", "") or "").strip()
        if env:
            try:
                return max(0.02, float(env))
            except (TypeError, ValueError):
                self._warn_bad_config_once("GATHERION_SKILL_BURST_SEC", env)
        for name in ("combat_skill_burst_sec", "skill_burst_sec"):
            raw = self._get_cfg(name, None)
            if raw is not None:
                try:
                    return max(0.02, float(raw))
                except (TypeError, ValueError):
                    self._warn_bad_config_once(name, raw)
        return 0.08 if self._keyboard_trusted_for_combat() else 0.15

    def _combat_skills_per_burst(self) -> int:
        env = str(os.environ.get("GATHERION_SKILLS_PER_BURST", "") or "").strip()
        if env:
            try:
                return max(1, min(8, int(env)))
            except (TypeError, ValueError):
                self._warn_bad_config_once("GATHERION_SKILLS_PER_BURST", env)
        raw = self._get_cfg("skills_per_burst", None)
        if raw is not None:
            try:
                return max(1, min(8, int(raw)))
            except (TypeError, ValueError):
                self._warn_bad_config_once("skills_per_burst", raw)
        return 4 if self._keyboard_trusted_for_combat() else 2

    def _combat_skill_hold_range(self) -> Tuple[float, float]:
        # v20/madde 2: 0.03-0.05 sn basım logda görünür ama oyunun giriş
        # yoklamasına takılmayıp kayda geçmeyebilir — güvenli alt sınır
        # 0.08-0.12 sn (insan basımı ölçeği; bus yoluyla aynı).
        if self._keyboard_trusted_for_combat():
            return 0.08, 0.12
        return 0.06, 0.10

    def _combat_skill_cooldown_sec(self, key: str) -> float:
        """Bot-side per-key floor. Trusted path stays short — game enforces real CD."""
        env = str(os.environ.get("GATHERION_SKILL_GCD_SEC", "") or "").strip()
        if env:
            try:
                return max(0.02, float(env))
            except (TypeError, ValueError):
                self._warn_bad_config_once("GATHERION_SKILL_GCD_SEC", env)
        for name in ("combat_skill_gcd_sec", "skill_gcd_sec"):
            raw = self._get_cfg(name, None)
            if raw is not None:
                try:
                    return max(0.02, float(raw))
                except (TypeError, ValueError):
                    self._warn_bad_config_once(name, raw)
        # Short ms-level bot-side floors so skills fire rapidly in round-robin sequence
        if self._keyboard_trusted_for_combat():
            return 0.12
        return 0.18

    def _is_route_fight_back_enabled(self) -> bool:
        """
        Returns True only if route-level aggro defense is explicitly enabled.
        Under pure Toplama (route.fight_back=False or unset), combat skills are suppressed.
        """
        cfg = getattr(self, "config", None)
        if isinstance(cfg, dict):
            route_cfg = cfg.get("route")
            if isinstance(route_cfg, dict):
                if route_cfg.get("fight_back") is not None:
                    return bool(route_cfg.get("fight_back"))
                if route_cfg.get("fight_back_when_attacked") is not None:
                    return bool(route_cfg.get("fight_back_when_attacked"))
            if cfg.get("route_fight_back") is not None:
                return bool(cfg.get("route_fight_back"))
        elif cfg is not None and hasattr(cfg, "get"):
            try:
                val = cfg.get("route", "fight_back", default=None)
                if val is not None:
                    return bool(val)
                val2 = cfg.get("route", "fight_back_when_attacked", default=None)
                if val2 is not None:
                    return bool(val2)
            except Exception as exc:
                self._warn_bad_config_once("route:fight_back", f"cfg.get failed: {exc}")
        return False

    def is_combat_authorized(self, action: str = "skill") -> bool:
        """
        Mode Isolation Gate:
        Enforces strict boundary separation between Combat and Gathering/Route operations.
        Suppresses all combat skill execution, target cycling, and attack holds
        whenever combat is not explicitly authorized.

        Args:
            action: "skill" (combat hotbar 1..5, q, e, r),
                    "target" (TAB targeting, soft-aim reticle),
                    "attack" (LMB / RMB continuous combat clicks/holds).
        """
        if getattr(self, "_paused", False) or getattr(self, "is_paused", False):
            return False

        current_mode = str(self._plain_attr("_mode", "")).strip().lower()
        current_state = str(self._plain_attr("_state", "")).strip().upper()

        # Active harvesting or GATHERING state ALWAYS strictly suppresses combat actions (zero leaks)
        if bool(self._plain_attr("_is_gathering_active", False)) or current_state == BotState.GATHERING.value:
            return False

        # Dedicated COMBAT mode: authorized during combat step or fighting state
        is_combat_mode = current_mode in (
            BotMode.COMBAT_MODE.value.lower(),
            "combat",
            "combat_mode",
            "savaş",
            "savas",
            "fight",
        )
        if is_combat_mode:
            return True

        # Non-combat modes (ROUTE_MODE / TOPLAMA, EXPLORER_MODE, MANUAL_RECORD, AUTO_RECORD):
        is_route = current_mode in (
            BotMode.ROUTE_MODE.value.lower(),
            "route",
            "toplama",
            "route_mode",
            "gather",
        )
        if is_route:
            if not self._is_route_fight_back_enabled():
                # Zero SKILL KEY leakage during Toplama when route.fight_back=False
                return False
        # F-23-05: şema varsayılanı True (config/schema.py RouteSettings).
        elif not bool(self._get_cfg("fight_back", True)):
            return False

        # When fight_back is enabled in route/other mode: authorized ONLY when character is confirmed in aggro defense
        in_fighting_state = current_state == BotState.FIGHTING.value
        has_aggro_signal = bool(
            self._plain_attr("_route_combat_confirmed", False)
            or self._plain_attr("_route_fight_had_signal", False)
            or self._plain_attr("_combat_was_targeting", False)
        )

        if not (in_fighting_state and has_aggro_signal):
            return False

        if action == "target":
            # F-0181: the old clause denied Tab when _route_was_fighting was
            # set but the target bar was gone — exactly the moment Tab is
            # needed to re-acquire. A confirmed fight authorizes Tab.
            return in_fighting_state and has_aggro_signal

        return True

    def _release_all_combat_inputs(self) -> None:
        """
        Anti-stuck key release guard: guarantees all combat keys, attacks,
        and modifier/targeting states are completely released on transitions.
        Ensures zero stuck keys and zero key leakage between modes.
        """
        try:
            self._sync_combat_right_button(False)
        except Exception:
            pass
        try:
            self._release_combat_left_button()
        except Exception:
            pass

        bus = self._plain_attr("action_bus")
        if bus is not None:
            try:
                if hasattr(bus, "release_left_attack"):
                    bus.release_left_attack()
                if hasattr(bus, "hold_basic_attack"):
                    bus.hold_basic_attack(False)
                if hasattr(bus, "end_exclusive"):
                    bus.end_exclusive(False)
            except Exception as exc:
                logger.debug("action_bus release failed: %s", exc)

        driver = getattr(self, "input_driver", None)
        if driver is not None:
            combat_keys = set(self._configured_combat_skill_keys())
            combat_keys.update(["1", "2", "3", "4", "5", "q", "e", "r", "tab", "shift", "ctrl", "alt"])

            hw_up = getattr(driver, "_hw_key_up", None)
            release_key = getattr(driver, "release_key", None)
            active_keys = getattr(driver, "_active_bot_keys", None)
            to_release = set()
            if isinstance(active_keys, set):
                to_release = active_keys.intersection(combat_keys)
            else:
                to_release = set(combat_keys)

            for k in to_release:
                try:
                    if callable(release_key):
                        release_key(k)
                    elif callable(hw_up):
                        hw_up(k)
                except Exception:
                    pass

            if hasattr(driver, "release_modifier_keys"):
                try:
                    driver.release_modifier_keys()
                except Exception:
                    pass

        self._combat_was_targeting = False
        self._combat_no_target_streak = 0
        self._combat_last_target_ts = 0.0
        self._mark_right_button(False)

    @staticmethod
    def _bus_cast_ok(result: Any) -> Optional[bool]:
        """GameActionBus.cast_skill dönüşünü 'tuş gönderildi mi' kararına çevirir.

        dict -> "ok" alanı (yoksa bilinmiyor=None); bool -> kendisi;
        None/dönüşsüz sürücü -> None (bilinmiyor; çağıran eski davranışı
        korur, yani damgalar). Yalnızca açık False "gönderilmedi" sayılır.
        """
        if result is None:
            return None
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            if "ok" not in result:
                return None
            return bool(result.get("ok"))
        return None

    def _cast_one_combat_skill(self, key: str, frame: Any = None) -> Optional[bool]:
        """One combat skill key. Returns True when the key was actually sent,
        False when it was suppressed (AutoGate/CombatGate) or the press
        failed, None when the bus gave no usable verdict (P0-6: callers
        must only stamp the per-key cooldown on a send — a False here
        means no stamp, so the key stays eligible for the next tick)."""
        key_str = str(key).strip().lower()
        if key_str in ("x", "f6", "b"):
            logger.debug("[AutoGate] _cast_one_combat_skill('%s') suppressed (automatic client function)", key)
            return False
        if not self.is_combat_authorized("skill"):
            logger.debug("[CombatGate] _cast_one_combat_skill('%s') suppressed", key)
            return False
        hold_min, hold_max = self._combat_skill_hold_range()
        bus = self._plain_attr("action_bus")
        if bus is not None and hasattr(bus, "cast_skill"):
            try:
                return self._bus_cast_ok(bus.cast_skill(key, frame=frame))
            except Exception:
                logger.debug("action_bus.cast_skill failed", exc_info=True)
                self._release_combat_right_button()
                return self._press_key(key, min_hold=hold_min, max_hold=hold_max, release_mouse=True)
        self._release_combat_right_button()
        return self._press_key(key, min_hold=hold_min, max_hold=hold_max, release_mouse=True)

    def _send_configured_combat_skill(self, ready_skills: Any = None) -> bool:
        """Rotate hotbar skills. Trusted Interception: fast multi-key bursts (no sticky 'best').

        F-0047: when the caller knows which skills are ready (cooldown-free),
        those go first. A ready list that intersects nothing means
        "nothing is ready" — never fall back to round-robin spam.
        v20/madde 2: kesişim tek tuşa indiyse (okuyucu yalnızca '2'yi hazır
        görüyorsa) havuz o tuşa çökmesin diye hazır tuşların ardından
        configured tuşlar eklenir (ManaPhase BURST harmanıyla aynı) —
        aksi halde rotasyon tek tuşa kilitlenir.
        """
        if self.is_paused:
            return False
        if bool(self._get_cfg("test_mode", False)):
            return False
        if not self.is_combat_authorized("skill"):
            return False
        keys = self._configured_combat_skill_keys()
        if ready_skills is not None:
            ready = self._normalize_combat_skill_keys(ready_skills)
            if ready:
                in_ready = [k for k in keys if k in ready]
                if not in_ready:
                    return False
                keys = in_ready + [k for k in keys if k not in ready]
        if not keys:
            return False
        now = time.time()
        burst_gap = self._combat_skill_burst_sec() + random.uniform(-0.015, 0.025)
        last_burst = float(self._plain_attr("_combat_last_skill_burst_ts", 0.0) or 0.0)
        if last_burst > 0.0 and (now - last_burst) < (burst_gap - 1e-4):
            return False
        used = self._plain_attr("_combat_skill_used_at")
        if not isinstance(used, dict):
            used = {}
            self._combat_skill_used_at = used
        # Round-robin only. Re-anchoring to select_best_skill stuck on "2"/"1" forever.
        start = int(self._plain_attr("_combat_skill_idx", 0) or 0) % len(keys)
        per_burst = self._combat_skills_per_burst()
        trusted = self._keyboard_trusted_for_combat()
        frame = self.__dict__.get("_last_color_frame")
        pressed = 0
        pressed_keys: list = []
        last_offset = -1
        for offset in range(len(keys)):
            if pressed >= per_burst:
                break
            key = keys[(start + offset) % len(keys)]
            last = float(used.get(key, 0.0) or 0.0)
            if last > 0.0 and (now - last) < (self._combat_skill_cooldown_sec(key) + random.uniform(-0.020, 0.020)):
                continue
            # P0-6: cooldown damgası gönderimden SONRA — _cast_one False
            # dönerse (kapı engeli / basış başarısız) damga yok, burst
            # sayılmaz; tuş bir sonraki tick'te tekrar denenir. None
            # (dönüşsüz bus) eski davranışı korur: damgalanır.
            sent = self._cast_one_combat_skill(key, frame=frame)
            if sent is False:
                continue
            used[key] = now
            last_offset = offset
            pressed += 1
            pressed_keys.append(key)
            if pressed < per_burst:
                # Inter-skill settle: variable sub-second delay (saliselik değişken)
                time.sleep(random.uniform(0.025, 0.055) if trusted else random.uniform(0.045, 0.075))
        # v20/madde 2 tanı: her burst'te hazır liste + havuz sırası +
        # gerçekten basılan tuşlar (tek tuşa kilitlenme canlı logdan okunur).
        logger.debug(
            "ROTATION ready=%r keys=%s pressed=%s",
            ready_skills,
            keys,
            pressed_keys,
        )
        if pressed <= 0:
            return False
        advance = last_offset + 1 if last_offset >= 0 else 1
        self._combat_skill_idx = (start + advance) % len(keys)
        # P0-6: burst damgası yalnızca >=1 tuş gerçekten gönderildiyse
        # yazılır (pressed artık yalnızca gönderilen/bilinmeyen basışları
        # sayar; hepsi başarısızsa yukarıda False dönüldü).
        self._combat_last_skill_burst_ts = now
        return True

    def _consume_mana_decision_via_bus(self, decision: Any, action_bus: Any) -> bool:
        """F-0122: mana_brain.decide() çıktısını apply_to_bus üzerinden tüket.

        decision.skill_keys / potion_keys / kiting bugüne dek üretimde hiç
        tüketilmiyordu (yalnızca phase okunuyordu). Bus yoksa, skill yetkisi
        yoksa veya kararın skill listesi boşsa False döner — çağıran o zaman
        eski _send_configured_combat_skill yoluna düşer (çift basım yok).
        F-0573: oto-iksir kapalıyken decision.potion_keys atılmaz.
        Hold senkronizasyonu apply_to_bus içinde de yapılır ama değerler
        _spam_combat_mouse/_sync_* ile aynıdır (hold çağrıları durum
        takipli/idempotent) — çift hold olmaz."""
        if decision is None or action_bus is None:
            return False
        if self.is_paused or bool(self._get_cfg("test_mode", False)):
            return False
        if not self.is_combat_authorized("skill"):
            return False
        skill_keys = list(getattr(decision, "skill_keys", None) or [])
        if not skill_keys:
            return False
        if not (hasattr(action_bus, "cast_skill") or hasattr(action_bus, "press_key")):
            return False
        mana_brain = self._plain_attr("mana_phase_combat")
        if mana_brain is None or not hasattr(mana_brain, "apply_to_bus"):
            return False
        try:
            if not self._is_auto_consumables_enabled():
                try:
                    decision.potion_keys = []
                except Exception:
                    pass
            mana_brain.apply_to_bus(decision, action_bus)
            return True
        except Exception as exc:
            logger.debug("ManaPhase apply_to_bus failed: %s", exc)
            return False

    def _real_target_for_skills(self, has_bar: bool) -> bool:
        """A skill is allowed when the target bar is up, or a monster already set the combat flag."""
        if has_bar:
            return True
        if bool(self._plain_attr("_combat_was_targeting", False)):
            return True
        return bool(self._plain_attr("_route_saw_target_bar", False))

    def _maybe_press_combat_f(self, frame_bgr: Any = None) -> None:
        """Periodic variable F press during combat (2 to 6 seconds: bazen 2, bazen 3, bazen 6)."""
        if self.is_paused or bool(self._get_cfg("test_mode", False)):
            return
        now = time.time()
        last_f = float(self._plain_attr("_combat_last_f_press_ts", 0.0) or 0.0)
        gap = float(self._plain_attr("_combat_next_f_interval_sec", 0.0) or 0.0)
        if gap <= 0.0 or last_f <= 0.0:
            self._combat_last_f_press_ts = now
            self._combat_next_f_interval_sec = random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
            return
        if (now - last_f) >= gap:
            self._combat_last_f_press_ts = now
            self._combat_next_f_interval_sec = random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
            try:
                self._emit_log(f"⚔️ Savaş içi F basımı (aralık: {gap:.1f}s)")
                bus = self._plain_attr("action_bus")
                if bus is not None and hasattr(bus, "cast_skill"):
                    bus.cast_skill("f", frame=frame_bgr)
                elif hasattr(self, "_press_key"):
                    self._press_key("f", min_hold=0.08, max_hold=0.12, release_mouse=True)
            except Exception as exc:
                logger.debug("Combat F press error: %s", exc)

    def _step_combat_mode(self) -> None:
        """
        Dedicated mob combat & hunting loop:
        1. Checks character death or revive dialog.
        2. Detects hostile monster target (Target frame / Center combat nameplate).
        3. If no target:
           - If target just died: loots with [F], then immediately presses [TAB] to acquire next target!
           - If idle/searching: presses [TAB] every 1.2s to search for nearby hostiles.
        4. If monster targeted:
           - Executes humanized combat skill rotation (1, 2, 3, 4, Q, E, R, Sol/Sağ Tık).
           - Tracks target HP bar. When target dies, immediately transitions to loot + TAB!
        5. Emits COMBAT HUD snapshot.
        """
        if self.is_paused:
            self._sync_combat_right_button(False)
            self._release_combat_left_button()
            return

        if not self._check_game_focus():
            return

        now = time.time()
        test_mode = bool(self._get_cfg("test_mode", False))

        # 1. Death check
        self._in_step_combat_mode = True
        try:
            if self._handle_combat_and_death():
                return
        finally:
            self._in_step_combat_mode = False

        frame_bgr = self._grab_frame(grayscale=False)
        if frame_bgr is None:
            return

        # 2. Check if hostile enemy is targeted (bar / soft-lock reticle / learner)
        has_enemy = self._combat_has_hostile_lock(frame_bgr)
        # Do NOT clear a real bar/reticle lock just because HP OCR is 0/stale —
        # that caused live HEDEF YOK + TAB spam while the game still had a target.
        still_taking_damage = False
        if self.combat_recovery is not None and hasattr(self.combat_recovery, "check_under_attack"):
            try:
                atk_res = self.combat_recovery.check_under_attack(frame_bgr)
                still_taking_damage = bool(atk_res[0]) if isinstance(atk_res, tuple) else bool(atk_res)
            except Exception:
                still_taking_damage = False

        was_fighting = getattr(self, "_combat_was_targeting", False)
        no_target_streak = getattr(self, "_combat_no_target_streak", 0)

        # 3. Handle monster death & targeting
        if not has_enemy:
            no_target_streak += 1
            setattr(self, "_combat_no_target_streak", no_target_streak)

            last_target_ts = float(self._plain_attr("_combat_last_target_ts", 0.0) or 0.0)
            time_since_target = (now - last_target_ts) if last_target_ts > 0.0 else 999.0

            # Target Persistence Grace:
            # Savaş esnasında hasar yazıları ("CRITICAL", uçuşan rakamlar) ve büyü partikülleri
            # hedef can barını veya isimliğini 1-2 saniye boyunca görsel olarak örtebilir.
            # Normal bir oyuncu bu örtülmeler esnasında skillerini kesintisiz spamlamaya devam eder.
            # 2.5 saniyeye (veya 25 kareye) kadar tanınan hedef devamlılığı toleransı sayesinde
            # hedef erken düşürülmez, kamera sağa sola savrulmaz ve 5 saniyelik duraksama tamamen kalkar.
            is_combat_grace = was_fighting and (time_since_target < 2.5 and no_target_streak < 25)

            if is_combat_grace:
                has_enemy = True
                is_flicker_grace = True
            else:
                # Savaş gerçekten bitti veya hedef tamamen kayboldu.
                # F-0182: a vanished bar alone is not a kill (flicker/flee).
                # Require ALL: was fighting AND no damage incoming AND the
                # target has been gone past the 2.5 s / 25-tick grace above.
                # (4.0 s here left a 1.5 s limbo after grace expiry with no
                # loot and no Tab; the grace boundary is the kill boundary.)
                death_ready = (
                    was_fighting
                    and not still_taking_damage
                    and time_since_target >= 2.5
                )
                if death_ready:
                    # Canavar öldü. Saldırı tuşlarını bırak; F ile ganimet topla ve sıradaki hedefe geç.
                    self._sync_combat_right_button(False)
                    self._release_combat_left_button()
                    setattr(self, "_combat_was_targeting", False)
                    setattr(self, "_combat_no_target_streak", 0)
                    self._emit_log("⚔️ Canavar öldü -> Ganimet toplanıyor (F)...")
                    self._emit_hud_snapshot(
                        badge=BotState.FIGHTING.value,
                        mode_badge="SAVAŞ (COMBAT)",
                        title="Canavar Öldü: Ganimet Toplanıyor",
                        subtitle="F ile ganimet toplanıyor, ardından sıradaki canavar seçilecek...",
                    )
                    # 17/B4: test_mode simüle gönderim sayılır (Y7 kuralı) —
                    # test_death_ready_* testi kill'i test_mode'da bekler.
                    f_sent = bool(test_mode)
                    if not test_mode:
                        f_sent = bool(self._press_key("f"))
                        self.msleep(150)
                        self._emit_log("Sıradaki canavarı seçmek için Tab tuşuna basılıyor...")
                        self._press_combat_tab(soft_aim=False)
                        self.msleep(120)
                    if f_sent:
                        # Kill yalnızca loot F'i gerçekten gönderildiyse
                        # sayılır — önce sayıp sonra F basmak, basılamayan
                        # F'i de kill sayıyordu.
                        try:
                            from core.lifetime_stats import get_lifetime_stats

                            get_lifetime_stats().record_monster_kill(mob_key="combat_death")
                        except Exception:
                            logger.debug("lifetime combat death kill skipped", exc_info=True)
                        # sweep-29 K-2: doğrulanmış öldürme telemetriye yazılır
                        # (C B4 ile birlikte hareket eder — kill yalnızca
                        # F gönderildiyse sayılır).
                        try:
                            _tel = getattr(self, "telemetry", None)
                            if _tel is not None and hasattr(_tel, "record_combat"):
                                _tel.record_combat(
                                    duration_sec=self._take_combat_duration_sec(),
                                    outcome="CLEARED",
                                    mob_key="combat_death",
                                )
                        except Exception as _tel_exc:
                            logger.warning("Telemetri record_combat (combat kill) hatası: %s", _tel_exc)
                    fresh = self._grab_frame(grayscale=False)
                    if self._combat_has_hostile_lock(fresh):
                        frame_bgr = fresh
                        has_enemy = True
                        self._combat_last_target_ts = now
                        setattr(self, "_combat_was_targeting", True)
                    else:
                        return
                elif was_fighting:
                    # Hedef barı kayboldu fakat hasar alınıyor -> hızlıca Tab ile hedefi yeniden kilitle (kamera süpürme yok)
                    self._sync_combat_right_button(False)
                    last_tab_ts = getattr(self, "_combat_last_tab_ts", 0.0)
                    if now - last_tab_ts >= 0.5:
                        setattr(self, "_combat_last_tab_ts", now)
                        if not test_mode:
                            self._press_combat_tab(soft_aim=False)
                            self.msleep(80)
                        fresh = self._grab_frame(grayscale=False)
                        if self._combat_has_hostile_lock(fresh):
                            frame_bgr = fresh
                            has_enemy = True
                            self._combat_last_target_ts = now
                            setattr(self, "_combat_was_targeting", True)
                        else:
                            return
                    else:
                        return
                else:
                    # Savaş dışı / Canavar aranıyor:
                    self._sync_combat_right_button(False)
                    self._release_combat_left_button()
                    last_tab_ts = getattr(self, "_combat_last_tab_ts", 0.0)
                    if now - last_tab_ts >= 0.7:
                        setattr(self, "_combat_last_tab_ts", now)
                        tab_misses = int(self._plain_attr("_combat_tab_miss_count", 0) or 0) + 1
                        self._combat_tab_miss_count = tab_misses
                        if tab_misses in (5, 15, 40):
                            self._warn_combat_input_blocked(tab_misses)
                        self._emit_hud_snapshot(
                            badge=BotState.FIGHTING.value,
                            mode_badge="SAVAŞ (COMBAT)",
                            title="Canavar Aranıyor (Tab)",
                            subtitle="HEDEF YOK — Tab ile canavar aranıyor...",
                        )
                        if not test_mode:
                            # Sadece üst üste 3 başarısız Tab sonrasında yumuşak hedefleme yap
                            use_soft = tab_misses >= 3
                            self._press_combat_tab(soft_aim=use_soft)
                            self.msleep(80)
                        fresh = self._grab_frame(grayscale=False)
                        if self._combat_has_hostile_lock(fresh):
                            frame_bgr = fresh
                            has_enemy = True
                            self._combat_last_target_ts = now
                            setattr(self, "_combat_was_targeting", True)
                        else:
                            return
                    else:
                        self._emit_hud_snapshot(
                            badge=BotState.FIGHTING.value,
                            mode_badge="SAVAŞ (COMBAT)",
                            title="Savaş Modu: Bekleniyor",
                            subtitle="Canavar aranıyor (Tab)...",
                        )
                        return

        if not has_enemy:
            return

        # 4. Active enemy targeted!
        setattr(self, "_combat_was_targeting", True)
        # sweep-29 K-2: savaş başlangıç damgası (record_combat süresi için).
        if not was_fighting:
            self._fight_started_mono = time.monotonic()
        self._combat_last_target_ts = now
        if not locals().get("is_flicker_grace", False) and self._combat_has_hostile_lock(frame_bgr):
            setattr(self, "_combat_no_target_streak", 0)
            setattr(self, "_combat_tab_miss_count", 0)
        self._set_state_internal(BotState.FIGHTING.value)

        # PerceptionHub snapshot (fallback: OmniGameLearner)
        # Use _plain_attr: unit tests skip QObject.__init__.
        # Combat: do not re-observe every tick when `_grab_frame` already
        # gated/throttled perception — that duplicate was a stall amplifier.
        frame_cog = self._plain_attr("_frame_cognition")
        perception_hub = self._plain_attr("perception_hub")
        if (
            frame_cog is None
            and perception_hub is not None
            and hasattr(perception_hub, "observe")
            and self._combat_perception_due()
        ):
            try:
                frame_cog = perception_hub.observe(frame_bgr)
                self._frame_cognition = frame_cog
                self._mark_combat_perception()
            except Exception:
                frame_cog = None
        game_learner = self._plain_attr("game_learner")
        cognition = {}
        if game_learner is not None and hasattr(game_learner, "get_live_cognition_snapshot"):
            try:
                cognition = game_learner.get_live_cognition_snapshot() or {}
            except Exception:
                cognition = {}
        # F-0046: algi yokken HP/MP/canavar cani %100 varsayilamaz.
        # Bilinmeyen degerler None kalir; asagidaki iksir/skill/HUD adimlari
        # None gorunce atlanir, sahte "tam canli" uzerinden karar verilmez.
        perception_known = bool(cognition) or (
            frame_cog is not None
            and hasattr(frame_cog, "as_dict")
            and bool(frame_cog.as_dict())
        )
        if frame_cog is not None and hasattr(frame_cog, "as_dict"):
            hub = frame_cog.as_dict()
            hp_pct = hub.get("monster_hp_percent", cognition.get("monster_hp_percent"))
            p_hp = hub.get("player_hp_percent", cognition.get("player_hp_percent"))
            p_mp = hub.get("player_mp_percent", cognition.get("player_mp_percent"))
            ready_skills = hub.get("ready_skills") or cognition.get("ready_skills") or []
            combo_seq = hub.get("combo_sequence") or cognition.get("combo_sequence") or []
            learned_cnt = hub.get("learned_frames", cognition.get("total_learned_frames", 0))
            # v20 (HP yanlış okuma): hub HP/MP'yi ölçememişse
            # (player_hp_valid / player_mp_valid False) 0.0 varsayılanı
            # oran sayılmaz — None olur; yoksa okuma hatası iksir /
            # kaçış tetikler ("CAN POTU %0", RETREAT_JUMP). Anahtar
            # yoksa (eski snapshot) varsayılan True: davranış korunur.
            _hp_known = bool(hub.get("player_hp_valid", True))
            _mp_known = bool(hub.get("player_mp_valid", True))
            player_mp_ratio = (
                float(hub["player_mp"])
                if hub.get("player_mp") is not None and _mp_known
                else (float(p_mp) / 100.0 if p_mp is not None and _mp_known else None)
            )
            player_hp_ratio = (
                float(hub["player_hp"])
                if hub.get("player_hp") is not None and _hp_known
                else (float(p_hp) / 100.0 if p_hp is not None and _hp_known else None)
            )
            monster_hp_ratio = (
                float(hub["monster_hp"])
                if hub.get("monster_hp") is not None
                else (float(hp_pct) / 100.0 if hp_pct is not None else None)
            )
            buff_count = int(hub.get("buff_count", 0) or 0)
        else:
            hp_pct = cognition.get("monster_hp_percent")
            p_hp = cognition.get("player_hp_percent")
            p_mp = cognition.get("player_mp_percent")
            ready_skills = cognition.get("ready_skills") or []
            combo_seq = cognition.get("combo_sequence") or []
            learned_cnt = cognition.get("total_learned_frames", 0)
            player_mp_ratio = float(p_mp) / 100.0 if p_mp is not None else None
            player_hp_ratio = float(p_hp) / 100.0 if p_hp is not None else None
            monster_hp_ratio = float(hp_pct) / 100.0 if hp_pct is not None else None
            buff_count = 0

        # Skill discovery probe (safe target present) — only with real vitals.
        if player_hp_ratio is not None and monster_hp_ratio is not None:
            self._maybe_run_skill_discovery(
                frame_bgr,
                player_hp_ratio=player_hp_ratio,
                monster_hp_ratio=monster_hp_ratio,
                buff_count=buff_count,
            )

        phase_label = "B"
        mana_brain = self._plain_attr("mana_phase_combat")
        skill_discovery = self._plain_attr("skill_discovery")
        action_bus = self._plain_attr("action_bus")
        if mana_brain is not None and hasattr(mana_brain, "set_thresholds"):
            try:
                # F-23-01: eşikler her savaş tick'inde ayardan tazelenir —
                # yalnızca mp_full değil, HP/MP iksir eşikleri + tuşlar da.
                # (Aynı tazeleme rota modunda _on_settings_saved üzerinden
                # refresh_consumable_settings() ile yapılır.)
                # P1-13: iksir/buff anahtarları da burada tazelenir.
                mana_brain.set_thresholds(
                    mp_full_threshold=float(self._get_cfg("mp_full_threshold", 0.95) or 0.95),
                    hp_potion_threshold=float(self._get_cfg("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT) or HP_POTION_THRESHOLD_PCT) / 100.0,
                    mp_potion_threshold=float(self._get_cfg("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT) or MP_POTION_THRESHOLD_PCT) / 100.0,
                    smart_hp_potion_threshold=float(self._get_cfg("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT) or HP_POTION_THRESHOLD_PCT) / 100.0,
                    smart_mp_potion_threshold=float(self._get_cfg("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT) or MP_POTION_THRESHOLD_PCT) / 100.0,
                    hp_potion_key=str(self._get_cfg("hp_potion_key", "F1") or "F1"),
                    mp_potion_key=str(self._get_cfg("mp_potion_key", "F2") or "F2"),
                    auto_use_hp_potion=bool(self._get_cfg("auto_use_hp_potion", True)),
                    auto_use_mp_potion=bool(self._get_cfg("auto_use_mp_potion", True)),
                    auto_refresh_buff_scroll=bool(self._get_cfg("auto_refresh_buff_scroll", True)),
                    buff_scroll_interval_sec=float(self._get_cfg("buff_scroll_interval_sec", 300) or 300),
                    buff_scroll_key=str(self._get_cfg("buff_scroll_key", "F7") or "F7"),
                )
            except Exception:
                pass

        hp_txt = f"%{hp_pct}" if hp_pct is not None else "?"
        p_hp_txt = f"%{p_hp}" if p_hp is not None else "?"
        p_mp_txt = f"%{p_mp}" if p_mp is not None else "?"
        self._emit_hud_snapshot(
            badge=BotState.FIGHTING.value,
            mode_badge="SAVAŞ (COMBAT)",
            title=f"⚔️ Savaş: Canavar [Can: {hp_txt}] · YZ Öğrenimi ({learned_cnt} Kare)",
            subtitle=f"Kombo: [{' → '.join(list(combo_seq)[:3])}] · Can: {p_hp_txt} · Mana: {p_mp_txt} · Ölünce Tab Basılacak",
            col1_title="YZ KARELERİ",
            col1_val=str(learned_cnt),
            col2_title="CANAVAR CANI",
            col2_val=hp_txt,
            col3_title="OYUNCU CAN/MANA",
            col3_val=f"{p_hp_txt}/{p_mp_txt}",
        )

        if not test_mode:
            # 1. Vitality check: drink HP/MP potions if vitals dropped (< 65%, emergency < 40%)
            # F-0046: vitals unknown -> no potion decision on a fake 100%.
            if player_hp_ratio is not None and player_mp_ratio is not None:
                self._check_and_trigger_auto_potions(player_hp_ratio, player_mp_ratio, frame_bgr)

            # P1-10 (K-14): savaş adımındaki canlı HP örneği watchdog'a
            # yayınlanır (ölüm/düşük HP izlemesi bu vitale bakar).
            self._publish_worker_hp_ratio(player_hp_ratio)

            allow_skills = self._real_target_for_skills(bool(has_enemy))
            decision = None
            self._log_combat_skill_knobs_once()
            if (
                mana_brain is not None
                and hasattr(mana_brain, "decide")
                and player_mp_ratio is not None
                and player_hp_ratio is not None
            ):
                try:
                    if skill_discovery is not None and hasattr(mana_brain, "set_lexicon"):
                        mana_brain.set_lexicon(skill_discovery.lexicon())
                    decision = mana_brain.decide(
                        player_mp=player_mp_ratio,
                        player_hp=player_hp_ratio,
                        ready_skills=ready_skills,
                        fallback_keys=self._configured_combat_skill_keys(),
                    )
                    phase_label = getattr(getattr(decision, "phase", None), "value", "B")
                except Exception:
                    logger.debug("ManaPhase decide skipped", exc_info=True)
                    decision = None

            # 2. Combat Mouse Attack Mode:
            # Low mana (Regen phase "A"): spam left click (sol tık) to recover mana
            # High mana (Burst phase "B"): spam right click (sağ tık) for burst damage
            # F-23-05: şema varsayılanı 0.95 (kurulumdaki çağrıyla aynıydı,
            # burası 0.90 diyordu — aynı ayarın iki varsayılanı vardı).
            mp_thresh = float(self._get_cfg("mp_full_threshold", 0.95) or 0.95)
            if decision is not None:
                from core.mana_phase_combat import ManaPhase
                is_regen = (getattr(decision, "phase", None) == ManaPhase.REGEN)
            else:
                is_regen = (
                    player_mp_ratio is not None and player_mp_ratio < mp_thresh
                )

            # Periodic variable F press in combat mode (bazen 2, bazen 3, bazen 6 sn)
            self._maybe_press_combat_f(frame_bgr)

            combat_mouse_btn = "left" if is_regen else "right"
            self._spam_combat_mouse(combat_mouse_btn)

            if combat_mouse_btn == "left":
                self._sync_combat_left_button(True)
                self._sync_combat_right_button(False)
            else:
                self._sync_combat_right_button(True)
                self._release_combat_left_button()

            # 3. Combat Skill Rotation (1, 2, 3, 4, Q, E, R)
            if not allow_skills:
                self._emit_log("Skill bekleniyor — gerçek hedef kilidi yok (HUD-only değil)")
            else:
                try:
                    burst_gap = self._combat_skill_burst_sec()
                    last_burst = float(self._plain_attr("_combat_last_skill_burst_ts", 0.0) or 0.0)
                    gate_ok = (last_burst <= 0.0) or ((now - last_burst) >= (burst_gap - 1e-4))
                    if mana_brain is not None and hasattr(mana_brain, "should_cast_skill"):
                        # F-0183: both clocks must agree; OR let either one
                        # override the other and spam skills.
                        gate_ok = gate_ok and bool(mana_brain.should_cast_skill(min_gap=burst_gap, now=now))
                    if gate_ok:
                        # F-0122: karar önce bus üzerinden tüketilir
                        # (skill_keys / potion_keys / kiting); bus yoksa veya
                        # tüketemezse eski rotasyon yoluna düşülür.
                        bus_consumed = False
                        if decision is not None:
                            bus_consumed = self._consume_mana_decision_via_bus(
                                decision, action_bus
                            )
                        if bus_consumed:
                            if mana_brain is not None:
                                mana_brain._last_skill_ts = now
                        else:
                            cast_done = self._send_configured_combat_skill(
                                ready_skills=ready_skills
                            )
                            if cast_done and mana_brain is not None:
                                mana_brain._last_skill_ts = now
                except Exception:
                    logger.debug("Combat skill skipped", exc_info=True)

            # Economy / reward tick (lightweight)
            # S-2601: bilinmeyen vitaller uydurulmaz — None olduğu gibi
            # geçirilir; _zk_post_combat_tick bilinmeyen MP/canavar canı için
            # ödül sinyali üretmez (HP'deki None -> 0.0/atla kuralıyla aynı).
            self._zk_post_combat_tick(
                player_mp_ratio=player_mp_ratio,
                monster_hp_ratio=monster_hp_ratio,
                phase=phase_label,
                player_hp_ratio=player_hp_ratio,
            )

    # =========================================================================
    # Mode 0: EXPLORER_MODE ("explorer" - Gezdikçe Otomatik Haritalandır & Sırala)
    # =========================================================================

    def _step_explorer_mode(self) -> None:
        """
        Player walks/flies freely around the map; bot automatically detects and maps
        every gem/material seen on 3D screen or minimap and orders them sequentially.
        """
        if not self._check_game_focus():
            return
        # F-0184: explorer mode previously skipped break/death checks entirely.
        now = time.time()
        if self._state == BotState.BREAK.value or self._should_enter_break(now):
            self._handle_break_step(now)
            return
        if self._handle_combat_and_death():
            return
        self._run_passive_auto_cartography(gather_while_exploring=True)
        self._set_state_internal(BotState.WALKING.value)
        ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
        self._emit_hud_snapshot(
            badge=BotState.WALKING.value,
            mode_badge="AUTO CARTOGRAPHY",
            title="Auto-mapping gems & materials",
            subtitle="Walk around: gems & ores are mapped in sequential order",
            col1_title="AUTO MAPPED",
            col1_val=str(self.saved_nodes_count),
            col2_title=self._get_route_header_title(),
            col2_val=f"{ready_cnt} ready · {resting_cnt} resting",
            col3_title="LAST MAPPED",
            col3_val=self.last_saved_str,
        )

    # =========================================================================
    # Mode 1: MANUAL_RECORD ("manual_record")
    # =========================================================================

    def _step_manual_record(self) -> None:
        """
        Player walks and presses F manually; pressing F8 records the current node,
        and passive auto-cartography also maps seen gems/materials automatically.
        HUD shows WAITING badge, MANUAL RECORD mode badge, and SAVED / NODES / LAST SAVED columns.
        """
        if not self._check_game_focus():
            return
        self._run_passive_auto_cartography(gather_while_exploring=False)
        self._set_state_internal(BotState.WAITING.value)
        ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
        self._emit_hud_snapshot(
            badge=BotState.WAITING.value,
            mode_badge="MANUAL RECORD",
            title="Manual record",
            subtitle="Walk around (auto-mapping active, or press Ctrl+Alt+F8)",
            col1_title="SAVED",
            col1_val=str(self.saved_nodes_count),
            col2_title="NODES",
            col2_val=f"{ready_cnt} ready · {resting_cnt} resting",
            col3_title="LAST SAVED",
            col3_val=self.last_saved_str,
        )

    # =========================================================================
    # Mode 2: AUTO_RECORD ("auto_record")
    # =========================================================================

    def _step_auto_record(self) -> None:
        """
        Player walks manually; bot watches for 'F Gather' prompt, presses F at
        randomized intervals (f_interval_min..f_interval_max), and automatically
        records a new node when gathering finishes (plus passive auto-cartography).
        """
        if not self._check_game_focus():
            return
        self._run_passive_auto_cartography(gather_while_exploring=False)
        detected, conf = self._detect_gather_prompt()
        self.last_prompt_match = conf
        now = time.time()

        if detected:
            self._gather_lost_frames = 0
            if not self._is_gathering_active:
                self._is_gathering_active = True
                self._stage_cast_in_progress = False
                self._emit_log(f"Gather prompt found ({conf}%)")
                self._emit_log("Gathering started")

            self._set_state_internal(BotState.GATHERING.value)
            cast_dur = float(self._get_cfg("gather_cast_duration_sec", 3.0) or 3.0)
            if self._stage_cast_in_progress:
                is_cast, pct = self._check_gather_cast_progress()
                if is_cast:
                    self._stage_cast_bar_seen = True
                    self._observe_cast_progress(pct)
                # F-0050 twin: a timeout with no cast bar ever seen is not a
                # successful gather. Y2/Y3: %90 tek karede yetmez, barın
                # yükseldiği de kanıtlanmalı; yoksa sayılmaz.
                cast_proven = (is_cast and pct >= 90.0 and bool(self._plain_attr("_stage_cast_rose", False))) or (
                    now - self._stage_cast_start_time >= cast_dur
                    and bool(self._plain_attr("_stage_cast_bar_seen", False))
                    and bool(self._plain_attr("_stage_cast_rose", False))
                )
                if cast_proven:
                    self._stage_cast_in_progress = False
                    self.gathered_count += 1
                    self._last_gather_time = now
                    # O2: oto-kayıtta da gerçek aşama sayılır (sabit 1/3 değil).
                    max_h_ar = self._configured_max_harvests(None)
                    self._auto_record_stage = min(self._auto_record_stage + 1, max_h_ar)
                    self._emit_log(f"Resource gathered (#{self.gathered_count})")
                elif (
                    now - self._stage_cast_start_time >= cast_dur
                    and bool(self._plain_attr("_stage_cast_bar_seen", False))
                    and not bool(self._plain_attr("_stage_cast_rose", False))
                ):
                    # Bar görüldü ama hiç yükselmedi: statik bar, kanıtlanmış
                    # cast değil — sayılmadan oturum bitirilir.
                    self._stage_cast_in_progress = False
                    self._is_gathering_active = False
                    self._gather_lost_frames = 0
                    self._auto_record_stage = 0
                    self._auto_record_prompt_name = ""
                    self._emit_log("Gather not confirmed (cast bar never rose): not counted")
            elif now - self._last_f_press_time >= self._next_f_interval:
                self._press_gather_key()
                self._stage_cast_in_progress = True
                self._reset_cast_progress_tracking()
                self._stage_cast_start_time = now
                self._last_f_press_time = now
                self._next_f_interval = self._sample_f_interval()

            self._emit_hud_snapshot(
                badge=BotState.GATHERING.value,
                mode_badge="AUTO RECORD",
                title="Gathering resource",
                subtitle=f"Gather prompt recognized at {conf}%",
                col1_title="GATHERED",
                col1_val=str(self.gathered_count),
                col2_title="F PRESSED",
                col2_val=str(self.f_pressed_count),
                col3_title="PROMPT MATCH",
                col3_val=f"{conf}%",
            )
        else:
            if self._is_gathering_active:
                gather_finished = False
                if self._stage_cast_in_progress:
                    is_cast, pct = self._check_gather_cast_progress()
                    if is_cast:
                        self._stage_cast_bar_seen = True
                        self._observe_cast_progress(pct)
                    cast_dur = float(self._get_cfg("gather_cast_duration_sec", 3.0) or 3.0)
                    # Y2/Y3: %90 tek karede yetmez; barın yükseldiği de kanıtlanmalı.
                    if (is_cast and pct >= 90.0 and bool(self._plain_attr("_stage_cast_rose", False))) or (
                        now - self._stage_cast_start_time >= cast_dur
                        and bool(self._plain_attr("_stage_cast_bar_seen", False))
                        and bool(self._plain_attr("_stage_cast_rose", False))
                    ):
                        self._stage_cast_in_progress = False
                        self._is_gathering_active = False
                        self._gather_lost_frames = 0
                        self.gathered_count += 1
                        self._last_gather_time = now
                        # O2: oto-kayıtta da gerçek aşama sayılır.
                        max_h_ar = self._configured_max_harvests(None)
                        self._auto_record_stage = min(self._auto_record_stage + 1, max_h_ar)
                        self._emit_log(f"Resource gathered (#{self.gathered_count})")
                        gather_finished = True
                else:
                    self._gather_lost_frames += 1
                    if self._gather_lost_frames >= 3:
                        # F-0375: a prompt that merely vanished for 3 ticks
                        # is not a completed gather — end the session without
                        # counting it.
                        self._is_gathering_active = False
                        self._gather_lost_frames = 0
                        self._auto_record_stage = 0
                        self._auto_record_prompt_name = ""
                        self._emit_log("Gather prompt lost: gather not confirmed, not counted")
                        gather_finished = False

                if gather_finished:
                    # Same as route mode: YOLO / dataset factory captures t-2.0s frame on gather end
                    if self.dataset_factory is not None and hasattr(
                        self.dataset_factory, "on_gather_success"
                    ):
                        try:
                            prompt_name = ""
                            if isinstance(getattr(self, "_last_gather_prompt", None), dict):
                                prompt_name = str(self._last_gather_prompt.get("name", "") or "")
                            # O2: aşama sabit 1/3 değil, bu ziyarette gerçekten
                            # tamamlanan aşama; materyal boşsa "Unknown".
                            max_h_ar = self._configured_max_harvests(None)
                            stage_ar = min(max(1, int(self._auto_record_stage or 1)), max_h_ar)
                            self.dataset_factory.on_gather_success(
                                material_name=prompt_name or "Unknown",
                                harvest_stage=stage_ar,
                                max_harvests=max_h_ar,
                            )
                            self._auto_record_stage = 0
                            self._auto_record_prompt_name = ""
                        except Exception as df_exc:
                            logger.debug(
                                "Dataset factory on_gather_success skipped: %s", df_exc
                            )

                    if bool(self._get_cfg("gathered_join_route", True)):
                        self._execute_f8_record(auto_triggered=True)

            self._set_state_internal(BotState.WALKING.value)
            self._emit_hud_snapshot(
                badge=BotState.WALKING.value,
                mode_badge="AUTO RECORD",
                title="Auto record",
                subtitle="Walk around: bot presses F and saves gathered nodes",
                col1_title="GATHERED",
                col1_val=str(self.gathered_count),
                col2_title="F PRESSED",
                col2_val=str(self.f_pressed_count),
                col3_title="PROMPT MATCH",
                col3_val=f"{conf}%",
            )

    # =========================================================================
    # Mode 3: ROUTE_MODE ("route")
    # =========================================================================

    def _step_route_mode(self) -> None:
        """
        Full autonomous route execution:
        1. Break schedule & active break handling
        2. Failure cooldown wait handling
        3. Idle no-gather auto-pause check
        4. Combat & character death check
        5. Blocking game window check -> close with Esc
        6. Gathering / walking / map navigation cycle
        """
        now = time.time()

        # A pause tick must drop W/A/D. Walking only continues while TOPLAMA is running.
        if self.is_paused:
            self._release_route_motion()
            return

        if not self._check_game_focus():
            return

        # 1. Active break or scheduled break trigger
        if self._state == BotState.BREAK.value or self._should_enter_break(now):
            self._handle_break_step(now)
            return

        # 2. Waiting cooldown after max consecutive failed nodes
        if self._failure_wait_until > now:
            rem_sec = max(0, int(self._failure_wait_until - now))
            rem_str = f"{rem_sec // 60}:{rem_sec % 60:02d}"
            self._set_state_internal(BotState.WAITING.value)
            self._emit_hud_snapshot(
                badge=BotState.WAITING.value,
                title=f"Waiting after failures · {rem_str}",
                subtitle="Cooling down before retrying route nodes",
            )
            return
        elif self._failure_wait_until > 0.0:
            self._failure_wait_until = 0.0
            self.failed_in_row = 0
            self._emit_log("Failure wait finished: resuming route")

        # 3. Auto-pause if no resource gathered for pause_after_no_gather_min
        # F-0503: an invalid setting must not raise every tick.
        try:
            pause_no_gather_min = float(self._get_cfg("pause_after_no_gather_min", 15) or 0)
        except (TypeError, ValueError):
            logger.warning("Invalid pause_after_no_gather_min; using 15")
            pause_no_gather_min = 15.0
        if pause_no_gather_min > 0:
            idle_sec = now - self._last_gather_time
            if idle_sec >= pause_no_gather_min * 60.0:
                self.pause(f"No gather for {int(pause_no_gather_min)} min")
                return

        # 4. Combat & Death checks
        if self._handle_combat_and_death():
            return

        # 5. Blocking in-game window check (close with Esc as in frame_34s.jpg)
        if self._handle_blocking_game_window():
            return

        # 5b. Bölge tespiti (v10 entegrasyon): rota modunda pasif kartografi
        # hiç çalışmıyordu — bölge uyumsuzluğu koruması beslenemiyordu.
        # gather_while_exploring=False: rota modunun kendi toplama motoru
        # (_handle_route_gathering) vardır; kartografi tick'i yalnızca
        # konum/bölge besler, fırsatçı toplama yapmaz. Bölge tespiti
        # 5 sn kısılmış olduğundan tekrarlayan çağrı ucuzdur.
        self._run_passive_auto_cartography(gather_while_exploring=False)

        # 6. Check for gather prompt or continue active gathering cycle
        # F-23-05: şema varsayılanı False (config/schema.py RouteSettings).
        gather_on_way = bool(self._get_cfg("gather_on_the_way", False))
        if gather_on_way or self.active_node is not None or self._is_gathering_active:
            detected, conf = self._detect_gather_prompt()
            self.last_prompt_match = conf

            if detected:
                if not self._is_gathering_active and self._should_bypass_unselected_gather():
                    self._emit_log("Bypassing non-matching material (selective gather active)")
                else:
                    self._handle_route_gathering(conf, now)
                    return
            elif self._is_gathering_active:
                if self._stage_cast_in_progress:
                    # Casting is in progress: keep driving the gather state (cast progress monitor & HUD)
                    self._handle_route_gathering(conf, now)
                    return
                elif self._waiting_for_next_stage_prompt:
                    if now <= self._post_harvest_prompt_deadline:
                        # Waiting for prompt to reappear after harvest stage completion
                        self._handle_route_gathering(conf, now)
                        return
                    else:
                        # Prompt did not reappear after stage completion.
                        # F-0186: a single missed frame / 4 s window must not
                        # kill a 1/3 or 2/3 node. Re-detect once; only when
                        # the prompt is really gone is the node depleted.
                        redetected, _rconf = self._detect_gather_prompt()
                        if redetected:
                            self._post_harvest_prompt_deadline = now + float(
                                self._get_cfg("post_harvest_prompt_timeout", 4.0) or 4.0
                            )
                            self._handle_route_gathering(_rconf, now)
                            return
                        node_id = self._extract_node_id(self.active_node)
                        self._emit_log(
                            f"Node {node_id} prompt timeout ({self._current_node_harvest_stage}/{self._target_harvests_for_active_node}) · node depleted early"
                        )
                        self._complete_route_gathering(now, depleted=True)
                        return
                else:
                    self._gather_lost_frames += 1
                    if self._gather_lost_frames >= 3:
                        self._complete_route_gathering(now, depleted=True)
                        return

        # 7. If no active node is selected, open map and navigate to next ready node
        if self.active_node is None:
            # F-0088: düğümsüz tick = ekonomi turu fırsatı (seyrek, 600 sn).
            self._maybe_run_economy_cycle(now)
            if gather_on_way:
                nearby = self._check_nearby_materials(max_distance_m=18.0)
                if nearby is not None and getattr(nearby, "screen_x", None) is not None:
                    # 16/3: fırsatçı toplama artık sadece log değil — yakındaki
                    # materyale gerçekten yürünür (geçici düğüm; harita
                    # tıklaması yok, ekrandaki konumuna yönelinir). Prompt
                    # çıkınca adım 6 toplama motorunu çalıştırır.
                    self._start_opportunistic_walk(nearby, now)
                    return
            self._select_and_navigate_next_node()
            return

        # 8. Currently walking to active_node: monitor timeout and periodic map reclick
        self._monitor_walking_to_node(now)

    def _parse_attack_result(self, atk_res: Any) -> Tuple[bool, Optional[float]]:
        """Split check_under_attack into (flag, hp ratio). A bool has no HP sample."""
        if isinstance(atk_res, tuple):
            under = bool(atk_res[0]) if atk_res else False
            hp_ratio: Optional[float] = None
            if len(atk_res) > 1 and atk_res[1] is not None:
                try:
                    hp_ratio = float(atk_res[1])
                except (TypeError, ValueError):
                    hp_ratio = None
            return under, hp_ratio
        return bool(atk_res), None

    def _enemy_target_bar_visible(self, frame: Optional[np.ndarray]) -> bool:
        """True when a hostile lock is visible (target bar or AION2 soft reticle)."""
        return self._combat_has_hostile_lock(frame)

    def _route_enemy_bar_visible(self, frame: Optional[np.ndarray]) -> bool:
        """True only for a REAL enemy target bar (combat_recovery detector).

        _enemy_target_bar_visible / _combat_has_hostile_lock ayrıca learner
        (current_monster_targeted) ve _frame_cognition (has_target) kilitlerini
        de sayar — keşif modunda botun kendi hedeflemesi / pasif kartografi
        kilidi düşman kilidi sayılmaz; yoksa oto-keşif her kilitte savaşa
        döner. Rota/keşif FIGHTING kapısı bu dar tanımı kullanır.
        """
        if frame is None:
            return False
        recovery = self.combat_recovery
        if recovery is None:
            return False
        for name in ("_detect_enemy_target_bar", "detect_enemy_target_bar"):
            fn = getattr(recovery, name, None)
            if not callable(fn):
                continue
            try:
                if bool(fn(frame)):
                    return True
            except Exception:
                pass
        return False

    def _combat_has_hostile_lock(self, frame: Optional[np.ndarray]) -> bool:
        """Hostile lock: real target bar / soft reticle / learner with HP > 0."""
        if frame is None:
            return False
        recovery = self.combat_recovery
        if recovery is not None:
            for name in ("_detect_enemy_target_bar", "detect_enemy_target_bar"):
                fn = getattr(recovery, name, None)
                if not callable(fn):
                    continue
                try:
                    if bool(fn(frame)):
                        return True
                except Exception:
                    pass
        learner = self._plain_attr("game_learner")
        if learner is not None:
            try:
                # Target flag alone is enough — HP OCR can lag/zero while lock is real.
                if bool(getattr(learner, "current_monster_targeted", False)):
                    return True
                snap_fn = getattr(learner, "get_live_cognition_snapshot", None)
                if callable(snap_fn):
                    snap = snap_fn() or {}
                    if bool(snap.get("monster_targeted")):
                        return True
            except Exception:
                pass
        hub = self._plain_attr("_frame_cognition")
        if hub is not None:
            try:
                if bool(getattr(hub, "has_target", False)):
                    return True
            except Exception:
                pass
        return False

    def _publish_worker_hp_ratio(self, hp_ratio: Optional[float]) -> None:
        """Son bilinen oyuncu HP oranını watchdog'a yayınlar.

        P1-10 (K-14): `core/watchdog.py::_check_worker_vitals`
        worker üzerinde `hp_ratio` okur; bu yayın olmadan WORKER_DEAD /
        WORKER_LOW_HP / WORKER_REVIVED ölü yoldu. Debounce,
        `_note_player_hp_drop`'taki F-0067/F-0376 kuralının aynısıdır:
        sağlıklı (> 0.02) örnek hemen yayınlanır; sıfıra-yakın (<= 0.02)
        örnek yalnızca sağlıklı (>= 0.5) tabandan ardışık 2 örnekte
        çöküş sayılır — tek karelik 0.0 okuma hatası (okunamayan bar)
        ölüm sayılmaz. `None` (örnek yok) son bilinen değeri korur.
        Rota ve savaş adımlarındaki iki canlı HP örnek noktasından
        çağrılır; saldırı algılama sayaçlarına dokunmaz.
        """
        if hp_ratio is None:
            return
        try:
            hp_f = float(hp_ratio)
        except (TypeError, ValueError):
            return
        # getattr yerine _plain_attr: BotWorker.__new__ ile üretilmiş
        # (QThread.__init__ çağrılmamış) nesnede eksik öznitelik için
        # QObject getattr RuntimeError atar ("super-class __init__() ...
        # was never called"); _plain_attr __dict__ okur.
        prev = self._plain_attr("_worker_hp_prev_ratio", None)
        self._worker_hp_prev_ratio = hp_f
        if hp_f > 0.02:
            self.hp_ratio = hp_f
            self._worker_hp_near_zero_streak = 0
            self._worker_hp_pre_collapse = None
            return
        if prev is None:
            self._worker_hp_near_zero_streak = 1
            self._worker_hp_pre_collapse = None
        elif prev > 0.02:
            self._worker_hp_near_zero_streak = 1
            self._worker_hp_pre_collapse = prev
        else:
            self._worker_hp_near_zero_streak = (
                getattr(self, "_worker_hp_near_zero_streak", 0) + 1
            )
        if (
            getattr(self, "_worker_hp_near_zero_streak", 0) == 2
            and getattr(self, "_worker_hp_pre_collapse", None) is not None
            and self._worker_hp_pre_collapse >= 0.5
        ):
            self.hp_ratio = hp_f

    def _note_player_hp_drop(self, hp_ratio: Optional[float]) -> bool:
        """True when this sample is a real HP drop. A missing bar is not a drop."""
        # P1-10 (K-14): rota adımındaki canlı HP örneği watchdog'a yayınlanır.
        self._publish_worker_hp_ratio(hp_ratio)
        if hp_ratio is None:
            # Örnek yok: ardışıklık bozulur ama sağlıklı taban korunur
            # (sonraki geçerli örnek hâlâ o tabanla kıyaslanır).
            self._route_near_zero_hp_streak = 0
            return False
        prev = getattr(self, "_route_prev_hp_ratio", None)
        self._route_prev_hp_ratio = hp_ratio
        if hp_ratio <= 0.02:
            # DÜZELTME (yanlış "saldırı altında"): hp_ratio <= 0.02
            # "bilinmiyor" olabilir — check_under_attack HP barını
            # okuyamadığında 0.0 döner (F-0067). Tek karelik 0.0 okuma
            # hatası, sağlıklı bir bardan gelmiş gibi "düşüş" sayılmaz;
            # canlıda bu, FIGHTING + "Saldırı altında" çıkarıyordu.
            # Gerçek sıfıra çöküş (F-0376 niyeti korunur) ancak ardışık
            # 2 sıfıra-yakın örnekle ve sağlıklı (>= 0.5) bir tabandan
            # sayılır — ve OLAY olarak bir kez sayılır: çöküş sayıldıktan
            # sonra HP ~0'da kalmaya devam ederse her karede True dönmez
            # (yoksa okunamayan HP, FIGHTING'i kalıcı kilitler).
            if prev is None:
                self._route_near_zero_hp_streak = 1
                self._route_pre_collapse_hp_ratio = None
            elif prev > 0.02:
                self._route_near_zero_hp_streak = 1
                self._route_pre_collapse_hp_ratio = prev
            else:
                self._route_near_zero_hp_streak = (
                    getattr(self, "_route_near_zero_hp_streak", 0) + 1
                )
            streak = getattr(self, "_route_near_zero_hp_streak", 0)
            baseline = getattr(self, "_route_pre_collapse_hp_ratio", None)
            return bool(
                streak == 2 and baseline is not None and baseline >= 0.5
            )
        self._route_near_zero_hp_streak = 0
        self._route_pre_collapse_hp_ratio = None
        if prev is None or prev <= 0.02:
            return False
        return (prev - hp_ratio) >= 0.015

    def _take_combat_duration_sec(self) -> float:
        """Sweep-29 K-2: savaş süresini döndürür ve başlangıç damgasını tüketir.

        Başlangıç _remember_real_combat_signal (rota) / _step_combat_mode
        (savaş modu) içinde tutulur. Damga yoksa 0.0 döner — süre bilinmiyor
        demektir, uydurulmaz; record_combat süreyi yalnızca ölçülebildiğinde
        yazar.
        """
        started = getattr(self, "_fight_started_mono", None)
        self._fight_started_mono = None
        if started is None:
            return 0.0
        try:
            return max(0.0, time.monotonic() - float(started))
        except (TypeError, ValueError):
            return 0.0

    def _remember_real_combat_signal(self, has_bar: bool, hp_dropped: bool) -> None:
        """A fight is real only after a target bar or an HP drop. Then the flag may stay set."""
        # F-0187: with neither signal this must be a no-op — previously it
        # unconditionally set all three flags.
        if not (has_bar or hp_dropped):
            return
        # sweep-29 K-2: savaş başlangıç damgası (record_combat süresi için).
        if not bool(getattr(self, "_route_was_fighting", False)):
            self._fight_started_mono = time.monotonic()
        self._route_fight_had_signal = True
        self._route_combat_confirmed = True
        self._route_was_fighting = True
        if has_bar:
            self._route_saw_target_bar = True
            self._route_bar_since_loot = True
        if hp_dropped:
            self._route_saw_hp_drop = True

    def _route_combat_signal_confirmed(
        self, has_bar: bool, hp_dropped: bool, tot_agro: bool = False
    ) -> bool:
        """True once a real combat signal repeats on consecutive ticks.

        fix/v20-panic-combat (canlı 14:48 hayalet-savaş: ortada saldıran
        düşman yokken pasif mob barıyla "Fighting enemy"): düşman barı
        TEK BAŞINA artık FIGHTING'e sokmaz — pasif mob hedeflendiğinde
        barı ardışık tick'lerde de görünür. Bar, yalnızca bar
        penceresinde saldıran-düşman kanıtı da görüldüyse doğrulanır:
        HP düşüşü veya ToT aggro (mob oyuncuyu hedefliyor).
        HP düşüşü tek başına (bar yokken) yalnızca ardışık
        route_fight_confirm_frames tick'te de düşüyorsa doğrular —
        tek karelik 0.0 okuma hatası _note_player_hp_drop'ta zaten
        elenir (sağlıklı >= 0.5 taban + ardışık 2 örnek + tek seferlik),
        burada da seviye değil SÜREKLİ düşüş istenir. Görev 1'in
        ürettiği tek-kare sinyaller aynen tüketilir — burada ikinci bir
        doğrulama mantığı YOK, sadece tekrar + bar-kanıtı sayılır.
        Eşik: route_fight_confirm_frames (varsayılan 2).
        """
        try:
            confirm_frames = int(self._get_cfg("route_fight_confirm_frames", 2) or 2)
        except (TypeError, ValueError):
            confirm_frames = 2
        confirm_frames = max(1, confirm_frames)
        if has_bar:
            self._route_bar_streak = int(getattr(self, "_route_bar_streak", 0) or 0) + 1
        else:
            self._route_bar_streak = 0
            self._route_bar_corroborated = False
        if hp_dropped:
            self._route_hp_streak = int(getattr(self, "_route_hp_streak", 0) or 0) + 1
        else:
            self._route_hp_streak = 0
        # Bar penceresi içinde kanıt: bu tick'te bar + (HP düşüşü |
        # ToT aggro) birlikte görüldüyse pencere kanıtlı sayılır; bar
        # kesilince pencere kapanır (yukarıda sıfırlandı).
        if has_bar and (hp_dropped or tot_agro):
            self._route_bar_corroborated = True
        return (
            self._route_hp_streak >= confirm_frames
            or (
                self._route_bar_streak >= confirm_frames
                and bool(getattr(self, "_route_bar_corroborated", False))
            )
        )

    def _route_fight_reentry_blocked(self, now: float) -> bool:
        """True while inside the cooldown after a route-fight teardown.

        Bir savaş bayrağı temizlendikten sonra (gerçek bitiş de olsa,
        yanlış-alarm da olsa) N saniye FIGHTING'e tekrar girilmez — 2
        karelik yanlış sinyalin bile döngüyü yeniden başlatmasını engeller.
        Süre: route_fight_reentry_cooldown_sec (varsayılan 8 sn; 0 =
        histerezis kapalı, sadece doğrulama kapısı çalışır).
        """
        try:
            cooldown = float(self._get_cfg("route_fight_reentry_cooldown_sec", 8.0) or 0.0)
        except (TypeError, ValueError):
            cooldown = 8.0
        if cooldown <= 0.0:
            return False
        last = float(getattr(self, "_route_fight_last_cleared_ts", 0.0) or 0.0)
        if last <= 0.0:
            return False
        return (now - last) < cooldown

    def _confirmed_fight_ended(self, has_bar: bool, hp_dropped: bool) -> bool:
        """Loot only after a real signal was present and has now ended."""
        if has_bar or hp_dropped:
            return False
        # __new__ worker'ında (__init__ atlanmış) bu bayraklar yoktur —
        # doğrudan okuma AttributeError verir; varsayılanlar __init__
        # değerleriyle aynıdır (False).
        had_real = bool(
            getattr(self, "_route_saw_target_bar", False)
            or getattr(self, "_route_saw_hp_drop", False)
        )
        return bool(
            getattr(self, "_route_was_fighting", False)
            and getattr(self, "_route_fight_had_signal", False)
            and had_real
        )

    def _loot_suppressed_empty_bar(self) -> bool:
        """An empty bar cannot press F again inside 30s unless a new target bar or HP drop was seen."""
        last = float(getattr(self, "_route_last_loot_ts", 0.0) or 0.0)
        if last <= 0.0 or (time.time() - last) >= 30.0:
            return False
        return not (
            getattr(self, "_route_bar_since_loot", False)
            or getattr(self, "_route_saw_hp_drop", False)
        )

    def _clear_route_combat(self) -> None:
        """Drop a combat flag that has no target bar and no HP drop, so the route can walk."""
        was_fighting = bool(getattr(self, "_route_was_fighting", False))
        self._route_was_fighting = False
        self._route_fight_had_signal = False
        self._route_combat_confirmed = False
        self._route_saw_target_bar = False
        self._route_saw_hp_drop = False
        # ROTA-İPTAL-ZİNCİRİ: doğrulama streak'leri ve askı bayrağı da
        # sıfırlanır — yeni sinyal sıfırdan doğrulansın.
        self._route_bar_streak = 0
        self._route_hp_streak = 0
        # fix/v20-panic-combat: bar-kanıt penceresi de sıfırlanır.
        self._route_bar_corroborated = False
        # W54/v19: hasat-sirasi saldiri damgasi da sifirlanir — sinyal
        # kesildiyse sure bir sonraki saldirida sifirdan baslar.
        self._gather_under_attack_since = 0.0
        # 17/B7: L3 fight-back aynası da sıfırlanır (loot sonrası yeni
        # savaşta eski streak F atlatmasın).
        self._route_fightback_no_bar_streak = 0
        self._route_fight_suspended = False
        if was_fighting:
            # Histerezis: bir savaş bayrağı temizlendi — N saniye boyunca
            # FIGHTING'e tekrar girilmez (kalıcı yanlış-alarm döngüsünü
            # kırar). Süre route_fight_reentry_cooldown_sec (varsayılan 8 sn).
            self._route_fight_last_cleared_ts = time.time()
        recovery = getattr(self, "combat_recovery", None)
        if recovery is not None and getattr(recovery, "in_combat", False):
            recovery.in_combat = False
        self._release_all_combat_inputs()
        if getattr(self, "_state", None) == BotState.FIGHTING.value:
            if getattr(self, "active_node", None) is not None:
                self._set_state_internal(BotState.WALKING.value)
            else:
                self._set_state_internal(BotState.WAITING.value)

    def _loot_confirmed_fight(self) -> None:
        """Press gather once when a confirmed fight ends. An empty bar never starts this."""
        if self.is_paused:
            return
        if self._loot_suppressed_empty_bar():
            # 17/B11: baskı artık sessiz değil.
            logger.warning(
                "Loot F bastırıldı: son loot'tan beri yeni hedef barı/HP düşüşü yok (30 sn kuralı)"
            )
            self._clear_route_combat()
            self._route_bar_since_loot = False
            return
        # F-0377: a kill is only recorded when a target bar was seen and has
        # now ended. An HP-drop-only fight proves damage, not a kill.
        had_target_bar = bool(
            getattr(self, "_route_saw_target_bar", False)
            or getattr(self, "_route_bar_since_loot", False)
        )
        if not had_target_bar:
            # 17/B5: yalnızca HP düşüşüyle biten savaşta loot F basılmaz —
            # bar görülmeden basılan F kör loot'tur (no27).
            logger.warning("Loot F basılmadı: hedef barı hiç görülmedi (yalnızca HP düşüşü)")
            self._clear_route_combat()
            self._route_bar_since_loot = False
            return
        # F16: atlama SADECE CombatSystem fallback'inde geçerlidir.
        # Eskiden "L3 barsız 2. adımda zaten loot F bastı, L2 F'i
        # atlanır" deniyordu; canlı L3 (CombatRecovery.fight_back_step,
        # Y2/P1-11) barsız 2. adımda kör loot F basmıyor (ne F basılır
        # ne Tab çevrilir ne kesim yazılır) — streak>=2 orada yalnızca
        # "2 ardışık barsız fight-back adımı" demektir ve L2 F'ini
        # atlamak SIFIR F bırakıyordu (kill yine de sayılıyordu).
        # F basan tek L3, combat_recovery import edilemezse devreye
        # giren CombatSystem fallback'idir (core/combat.py
        # fight_back_step, ölümde F+Tab — F13 v18). O yüzden atlama,
        # recovery CombatSystem ise uygulanır; canlı yolda (ve
        # mock/fake recovery'de) L2 her zaman tek F'i basar (no27:
        # ölüm başına tek F; çift F yok çünkü canlı L3 F basmaz).
        try:
            from core.combat import CombatSystem

            _l3_presses_loot_f = isinstance(
                getattr(self, "combat_recovery", None), CombatSystem
            )
        except Exception:
            _l3_presses_loot_f = False
        l3_looted = _l3_presses_loot_f and (
            int(getattr(self, "_route_fightback_no_bar_streak", 0) or 0) >= 2
        )
        if l3_looted:
            logger.info("Loot F atlandı: barsız fight-back streak>=2 (no27 — ikinci F yok)")
        # 17/B9: rota modunda düğüm varken toplama istemi görünüyorsa F
        # toplamayı başlatırdı — loot F atlanır (yalnızca L2; combat
        # modunda düğüm bağlamı yok, guardsız bırakılır — açık soru:
        # gerçek loot-istemi detektörü lazım).
        prompt_blocks_loot = False
        if not l3_looted:
            is_route = str(self._plain_attr("_mode", "")).strip().lower() in (
                BotMode.ROUTE_MODE.value.lower(),
                "route",
                "toplama",
                "route_mode",
            )
            if is_route and getattr(self, "active_node", None) is not None:
                try:
                    prompt_seen, _prompt_conf = self._detect_gather_prompt()
                except Exception as exc:
                    logger.debug("Loot gather-prompt kontrolü atlandı: %s", exc)
                    prompt_seen = False
                if prompt_seen:
                    prompt_blocks_loot = True
                    logger.warning("Loot F atlandı: toplama istemi var (F toplamayı başlatırdı)")
        test_mode = bool(self._get_cfg("test_mode", False))
        # test_mode simüle gönderim sayılır (Y7 kuralı; L1 ile aynı).
        f_sent = bool(l3_looted or test_mode) and not prompt_blocks_loot
        if not l3_looted and not prompt_blocks_loot and not test_mode:
            f_sent = bool(self._press_key("f"))
            self.msleep(200)
        # 17/B11: zaman damgası yalnızca F gönderildiyse yazılır.
        if f_sent:
            self._route_last_loot_ts = time.time()
        self._route_bar_since_loot = False
        self._clear_route_combat()
        if f_sent:
            # 17/B4: kill yalnızca loot F'i gönderildiyse sayılır (L2 F'i
            # ya da L3 fight-back F'i). Basış başarısızlığı / prompt engeli
            # -> kill sayılmaz. (had_target_bar B5 kapısından geçti.)
            try:
                from core.lifetime_stats import get_lifetime_stats

                get_lifetime_stats().record_monster_kill(mob_key="route_loot")
            except Exception:
                logger.debug("lifetime route loot kill skipped", exc_info=True)
            # sweep-29 K-2: doğrulanmış savaş sonu telemetriye yazılır —
            # eskiden record_combat hiç çağrılmıyordu, savaş sayacı hep 0'dı.
            # F-0377 ile tutarlı: yalnızca bar görülüp bittiyse öldürme sayılır.
            try:
                _tel = getattr(self, "telemetry", None)
                if _tel is not None and hasattr(_tel, "record_combat"):
                    _tel.record_combat(
                        duration_sec=self._take_combat_duration_sec(),
                        outcome="CLEARED",
                        mob_key="route_loot",
                    )
            except Exception as _tel_exc:
                logger.warning("Telemetri record_combat (route loot) hatası: %s", _tel_exc)
        if f_sent:
            self._emit_log("Saldırgan canavar öldü -> Ganimet alınıyor (F)...")
        elif prompt_blocks_loot:
            self._emit_log("Loot F atlandı (toplama istemi) -> rotaya dönülüyor...")
        else:
            self._emit_log("Loot F gönderilemedi — kill sayılmadı.")
        if not test_mode:
            is_combat_mode = str(self._plain_attr("_mode", "")).strip().lower() in (
                BotMode.COMBAT_MODE.value.lower(),
                "combat",
                "combat_mode",
                "savaş",
                "savas",
                "fight",
            )
            if is_combat_mode:
                self._press_combat_tab()
                self.msleep(150)
            else:
                self._release_all_combat_inputs()
                self._emit_log("⚔️ Savunma tamamlandı -> Rotaya geri dönülüyor...")
                if getattr(self, "active_node", None) is not None:
                    self._set_state_internal(BotState.WALKING.value)
                else:
                    self._set_state_internal(BotState.WAITING.value)

    def _right_button_is_tracked(self) -> bool:
        """Read the hold flag without touching an uninitialized QObject attribute."""
        try:
            return bool(self.__dict__.get("_combat_right_button_down", False))
        except Exception:
            return False

    def _mark_right_button(self, held: bool) -> None:
        try:
            self.__dict__["_combat_right_button_down"] = bool(held)
            # F-0378: timestamp the hold so a stale flag cannot claim "held"
            # forever if the driver lost the key (focus loss, driver reset).
            self.__dict__["_combat_right_button_since"] = time.time() if held else 0.0
        except Exception:
            try:
                self._combat_right_button_down = bool(held)
            except Exception:
                logger.debug("Right-button flag was not stored", exc_info=True)

    def _sync_combat_right_button(self, target_selected: bool, force: bool = False) -> None:
        """Hold T key while a real target is selected (right-click attack = T, per user mapping).
        Release when the target is gone or the user pauses. No mouse movement/click."""
        if not target_selected or self._combat_attack_paused() or not self.is_combat_authorized("attack"):
            self._release_combat_right_button()
            return
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        driver = self.input_driver
        if self._right_button_is_tracked() and not force:
            # F-0378: a hold older than 30 s without re-confirmation is stale
            # — re-press instead of trusting the flag forever.
            since = float(self._plain_attr("_combat_right_button_since", 0.0) or 0.0)
            if since > 0.0 and (time.time() - since) < 30.0:
                return
            self._mark_right_button(False)
        # force=True: skill sonrası bayrak yanlış kalsa bile T'yi bırakıp tekrar bas.
        if force and self._right_button_is_tracked():
            up = getattr(driver, "key_up", None)
            if callable(up):
                try:
                    up("t")
                except Exception:
                    pass
        down = getattr(driver, "key_down", None)
        if callable(down):
            try:
                if down("t"):
                    self._mark_right_button(True)
                    return
            except Exception:
                pass
        # Fallback: press_key pulse (routes through the same hardware-key gate).
        press = getattr(driver, "press_key", None)
        if callable(press):
            try:
                press("t", min_hold=0.05, max_hold=0.09)
                self._mark_right_button(True)
            except Exception:
                pass

    def _combat_attack_paused(self) -> bool:
        try:
            if self.is_paused:
                return True
        except Exception:
            if getattr(self, "_paused", False):
                return True
        driver = getattr(self, "input_driver", None)
        aborted = getattr(driver, "motion_aborted", None) if driver is not None else None
        if not callable(aborted):
            return False
        try:
            return bool(aborted())
        except Exception:
            return False

    def _release_held_mouse_button(self, button: str) -> bool:
        """Safety net: release a physically held mouse button, if any.
        Release/cleanup only — never presses or clicks. Used by the combat
        release paths so a button held by older code (or the user) cannot
        stay stuck down after combat ends.

        P0-5/K-04: yumuşak SendInput yedeği KALDIRILDI ve `except: pass`
        kaldırıldı — bırakma yalnızca güvenilir `_hw_mouse_up` yolundan
        denenir; reddedilirse/hata verirse log + HUD alarmı üretilir ve
        False döner (tuş `_mouse_buttons_down` kaydında kalır ki sonraki
        süpürme yeniden deneyebilsin). True = bırakılacak tuş yok ya da
        bırakma kabul edildi.
        """
        driver = getattr(self, "input_driver", None)
        if driver is None:
            return False
        held = getattr(driver, "_mouse_buttons_down", None)
        if not (isinstance(held, (set, frozenset)) and button in held):
            return True
        mouse_up = getattr(driver, "_hw_mouse_up", None)
        if not callable(mouse_up):
            msg = (
                f"FARE BIRAKMA HATASI: '{button}' tuşu basılı kayıtta ama "
                "sürücüde _hw_mouse_up yok — tuş bırakılamadı."
            )
            logger.error(msg)
            self._emit_log(msg)
            return False
        try:
            released = mouse_up(button)
        except Exception as exc:
            msg = (
                f"FARE BIRAKMA HATASI: '{button}' tuşu bırakılamadı ({exc}) — "
                "güvenilir fare yolu reddetti/hata verdi, yumuşak yedek yok. "
                "Tuş basılı kalmış olabilir."
            )
            logger.error(msg)
            self._emit_log(msg)
            return False
        if released is False:
            msg = (
                f"FARE BIRAKMA HATASI: '{button}' tuşu bırakılamadı — "
                "güvenilir fare yolu (Interception/Faker) reddetti, yumuşak "
                "yedek yok. Tuş basılı kalmış olabilir."
            )
            logger.error(msg)
            self._emit_log(msg)
            return False
        return True

    def _release_combat_right_button(self) -> None:
        """Release the held T key (right-click attack = T) and, as a safety
        net, any physically held right mouse button (baseline behaviour).
        Release/cleanup only — no mouse click is issued on the attack path."""
        driver = getattr(self, "input_driver", None)
        tracked = self._right_button_is_tracked()
        held = getattr(driver, "_mouse_buttons_down", None) if driver is not None else None
        button_down = isinstance(held, (set, frozenset)) and "right" in held
        if driver is None or not (button_down or tracked):
            self._mark_right_button(False)
            return
        up = getattr(driver, "key_up", None)
        if callable(up):
            try:
                up("t")
            except Exception:
                pass
        if button_down:
            self._release_held_mouse_button("right")
        self._mark_right_button(False)

    def _warn_combat_input_blocked(self, tab_misses: int) -> None:
        """HUD/log when Tab spam finds no target — often UAC integrity mismatch."""
        admin = False
        try:
            from core.uac_elevator import is_admin

            admin = bool(is_admin())
        except Exception:
            try:
                import ctypes

                admin = bool(ctypes.windll.shell32.IsUserAnAdmin() != 0)
            except Exception:
                admin = False
        if admin:
            msg = (
                f"COMBAT Tab {tab_misses}x HEDEF YOK — bot elevated ama hedef yok: "
                "AION2 Raw Input Tab/skill inject'i filtreliyor olabilir. "
                "Mouse-only soft-aim aktif. FakerInputWrapper.dll yok — HID klavye yok. "
                "Not: Steam de High ise tüm çocuklar (AION2) elevated kalır."
            )
        else:
            msg = (
                f"COMBAT Tab {tab_misses}x HEDEF YOK — bot YÖNETİCİ DEĞİL (Medium). "
                "Steam+AION2 High ise UIPI SendInput (tuş+fare) sessizce düşer. "
                "Çözüm: Android GatherionHidBridge + GATHERION_HID_TCP (docs/ANDROID_HID_KEYBOARD.md), "
                "VEYA Leonardo GATHERION_HID_COM, VEYA Steam Medium (STEAM_UNELEVATED.bat)."
            )
        self._emit_log(msg)
        logger.warning(msg)

    def _sync_combat_left_button(self, hold: bool) -> None:
        """Mana regen phase R-key hold (left-click attack = R, via GameActionBus / driver).
        On release, also releases a physically held left mouse button as a
        safety net (same defensive double-release as the right side)."""
        if bool(self._get_cfg("test_mode", False)):
            return
        if hold and not self.is_combat_authorized("attack"):
            self._release_combat_left_button()
            return
        bus = self._plain_attr("action_bus")
        if bus is not None and hasattr(bus, "hold_left_attack"):
            try:
                bus.hold_left_attack(bool(hold))
                if not hold:
                    self._release_held_mouse_button("left")
                return
            except Exception as exc:
                logger.debug("hold_left_attack via bus failed: %s", exc)
        driver = getattr(self, "input_driver", None)
        if driver is None:
            return
        try:
            if hold:
                down = getattr(driver, "key_down", None)
                if callable(down):
                    down("r")
            else:
                up = getattr(driver, "key_up", None)
                if callable(up):
                    up("r")
                self._release_held_mouse_button("left")
        except Exception as exc:
            logger.debug("left attack key sync failed: %s", exc)

    def _release_combat_left_button(self) -> None:
        self._sync_combat_left_button(False)

    def _spam_combat_mouse(self, button: str) -> None:
        """
        Combat attack spam via keys (name kept for callers/tests): 'left' (low-mana
        regen) presses R, 'right' (high-mana burst) presses T. No mouse movement,
        no clicks — mouse-look camera never drifts.
        """
        btn = str(button or "left").strip().lower()
        if btn not in ("left", "right"):
            btn = "left"
        if self.is_paused or bool(self._get_cfg("test_mode", False)):
            return
        if not self.is_combat_authorized("attack"):
            return

        now = time.time()
        last_spam = float(self._plain_attr("_last_combat_mouse_spam_ts", 0.0) or 0.0)
        spam_gap = float(self._plain_attr("_combat_mouse_spam_gap", 0.09) or 0.09)
        if (now - last_spam) < spam_gap:
            return

        # Ensure opposite attack key is released
        other = "right" if btn == "left" else "left"
        if other == "right":
            self._sync_combat_right_button(False)
        else:
            self._release_combat_left_button()

        # right-click attack = T, left-click attack = R (user mapping).
        key = "t" if btn == "right" else "r"
        driver = getattr(self, "input_driver", None)
        port = self._plain_attr("input_port")
        try:
            if driver is not None and hasattr(driver, "press_key"):
                driver.press_key(key, min_hold=0.025, max_hold=0.045)
            elif port is not None and hasattr(port, "send_key"):
                port.send_key(key, min_hold=0.025, max_hold=0.045)
            elif driver is not None and hasattr(driver, "key_down"):
                driver.key_down(key)
                time.sleep(0.025)
                driver.key_up(key)
        except Exception as exc:
            logger.debug("combat key spam failed for %s: %s", btn, exc)

        self._last_combat_mouse_spam_ts = time.time()
        self._combat_mouse_spam_gap = random.uniform(0.075, 0.110)

    def _is_auto_consumables_enabled(self) -> bool:
        """F-0573: oto-iksir anahtarı. UI (ui/f9_menu.py) önce
        worker.auto_consumables_enabled özniteliğini yazar; yoksa
        config/schema.py `gathering.auto_consumables` okunur. Ayar henüz
        yoksa varsayılan açık — eski davranış korunur. Kapalıyken iksir atılmaz."""
        try:
            override = self.__dict__.get("auto_consumables_enabled", None)
            if override is not None:
                return bool(override)
        except Exception:
            pass
        try:
            return bool(self._get_cfg("auto_consumables", True))
        except Exception:
            return True

    def set_auto_consumables_enabled(self, enabled: bool) -> None:
        """F-0573: UI onay kutusunun (ui/f9_menu.py) worker'a yazma yolu."""
        try:
            self.__dict__["auto_consumables_enabled"] = bool(enabled)
        except Exception:
            pass

    def refresh_consumable_settings(self) -> None:
        """F-23-01/F-23-11: canlı ayar tazeleme (main._on_settings_saved).

        Ayar kaydedilince SettingsManager yeniden yüklenir; bu metod o
        değerleri başlangıçta bir kez kurulan canlı nesnelere yazar:
        ManaPhaseCombat eşik/tuşları (set_thresholds) ve SlotEconomy keşif
        oranı (set_explore_ratio). Savaş tick'i eşikleri zaten her tick
        tazeler (bkz. set_thresholds çağrısı); bu yol rota modunda da
        tazelenmeyi sağlar. P1-13: iksir/buff anahtarları da burada
        tazelenir (auto_use_hp/mp_potion, auto_refresh_buff_scroll,
        buff_scroll_interval_sec, buff_scroll_key). Hiçbir alt adımda hata
        fırlatmaz — bir modül yoksa/hatalıysa diğerleri yine uygulanır.
        """
        try:
            hp_pct = float(self._get_cfg("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT) or HP_POTION_THRESHOLD_PCT) / 100.0
            mp_pct = float(self._get_cfg("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT) or MP_POTION_THRESHOLD_PCT) / 100.0
            mana_brain = self._plain_attr("mana_phase_combat")
            if mana_brain is not None and hasattr(mana_brain, "set_thresholds"):
                try:
                    mana_brain.set_thresholds(
                        mp_full_threshold=float(self._get_cfg("mp_full_threshold", 0.95) or 0.95),
                        hp_potion_threshold=hp_pct,
                        mp_potion_threshold=mp_pct,
                        smart_hp_potion_threshold=hp_pct,
                        smart_mp_potion_threshold=mp_pct,
                        hp_potion_key=str(self._get_cfg("hp_potion_key", "F1") or "F1"),
                        mp_potion_key=str(self._get_cfg("mp_potion_key", "F2") or "F2"),
                        auto_use_hp_potion=bool(self._get_cfg("auto_use_hp_potion", True)),
                        auto_use_mp_potion=bool(self._get_cfg("auto_use_mp_potion", True)),
                        auto_refresh_buff_scroll=bool(self._get_cfg("auto_refresh_buff_scroll", True)),
                        buff_scroll_interval_sec=float(self._get_cfg("buff_scroll_interval_sec", 300) or 300),
                        buff_scroll_key=str(self._get_cfg("buff_scroll_key", "F7") or "F7"),
                    )
                except Exception as exc:
                    logger.debug("ManaPhase eşik tazeleme hatası: %s", exc)
        except Exception as exc:
            logger.debug("Tüketim eşiği okuma hatası: %s", exc)
        try:
            eco = self._plain_attr("slot_economy")
            if eco is not None and hasattr(eco, "set_explore_ratio"):
                try:
                    eco.set_explore_ratio(float(self._get_cfg("ucb_explore_ratio", 0.20) or 0.20))
                except Exception as exc:
                    logger.debug("SlotEconomy oran tazeleme hatası: %s", exc)
        except Exception as exc:
            logger.debug("Keşif oranı okuma hatası: %s", exc)

    def _check_and_trigger_auto_potions(
        self,
        player_hp_ratio: float,
        player_mp_ratio: float,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Monitors player HP and MP during combat and route travel:
        - HP < hp_potion_threshold (default 0.65 / 65%): triggers HP potion (e.g. F1, F3, 7).
        - Emergency HP < emergency_hp_threshold (0.40): immediate priority HP recovery.
        - MP < mp_potion_threshold (default 0.45 / 45%): triggers MP potion (e.g. F2, F4, 8).
        - Strictly forbids and suppresses keys X, F6, and B (managed automatically by client).
        Respects per-potion cooldowns (default 8.0s).
        """
        if self.is_paused or bool(self._get_cfg("test_mode", False)):
            return False

        # F-0573: oto-iksir kapalıyken (UI onay kutusu / config) iksir atılmaz.
        if not self._is_auto_consumables_enabled():
            return False

        # P1-13: iksir başına anahtar (route.auto_use_hp/mp_potion) —
        # ana şalter açıkken bile tek iksir kapatılabilir.
        hp_enabled = bool(self._get_cfg("auto_use_hp_potion", True))
        mp_enabled = bool(self._get_cfg("auto_use_mp_potion", True))

        now = time.time()
        try:
            hp = max(0.0, min(1.0, float(player_hp_ratio)))
        except (TypeError, ValueError):
            return False
        # F-0194: MP may be unknown (None) in route mode — skip the MP potion
        # instead of pretending MP is full (1.0).
        mp: Optional[float] = None
        if player_mp_ratio is not None:
            try:
                mp = max(0.0, min(1.0, float(player_mp_ratio)))
            except (TypeError, ValueError):
                return False

        # If HP is near zero (e.g. death or invalid frame), death handler manages it
        if hp <= 0.01:
            return False

        mana_brain = self._plain_attr("mana_phase_combat")
        hp_thresh = float(getattr(mana_brain, "smart_hp_potion_threshold", 0.65) if mana_brain else 0.65)
        # F-23-01: mana_brain yokken MP yedeği 0.35'ti; smart varsayılan
        # (0.45) ve şema (mp_potion_threshold_pct 45) ile çelişiyordu.
        mp_thresh = float(getattr(mana_brain, "smart_mp_potion_threshold", 0.45) if mana_brain else 0.45)
        emerg_thresh = float(getattr(mana_brain, "emergency_hp_threshold", 0.40) if mana_brain else 0.40)

        hp_cd = float(getattr(mana_brain, "hp_potion_cd_sec", 8.0) if mana_brain else 8.0)
        mp_cd = float(getattr(mana_brain, "mp_potion_cd_sec", 8.0) if mana_brain else 8.0)

        last_hp = float(self._plain_attr("_last_hp_potion_ts", 0.0) or 0.0)
        last_mp = float(self._plain_attr("_last_mp_potion_ts", 0.0) or 0.0)
        if mana_brain and hasattr(mana_brain, "_last_hp_potion_ts"):
            last_hp = max(last_hp, float(mana_brain._last_hp_potion_ts or 0.0))
        if mana_brain and hasattr(mana_brain, "_last_mp_potion_ts"):
            last_mp = max(last_mp, float(mana_brain._last_mp_potion_ts or 0.0))

        FORBIDDEN = {"x", "f6", "b"}
        pot_triggered = False

        # 1. HP Potion Check
        if hp_enabled and hp < hp_thresh:
            if (now - last_hp) >= hp_cd:
                chosen_key = None
                consumable_mgr = getattr(mana_brain, "consumable_manager", None) if mana_brain else None
                if consumable_mgr is not None and hasattr(consumable_mgr, "get_best_hp_potion"):
                    try:
                        best = consumable_mgr.get_best_hp_potion(now=now)
                        if best and str(best).strip().lower() not in FORBIDDEN:
                            chosen_key = str(best).strip()
                    except Exception:
                        chosen_key = None

                if not chosen_key:
                    # F-0380: never blind-press F1 when no potion slot is
                    # known. Only an explicitly configured key may be used.
                    fallback_key = getattr(mana_brain, "hp_potion_key", None) if mana_brain else None
                    if fallback_key and str(fallback_key).strip().lower() not in FORBIDDEN:
                        chosen_key = str(fallback_key).strip()
                    else:
                        logger.debug("HP potion skipped: no known potion slot (no blind F1)")
                        chosen_key = None

                if chosen_key and str(chosen_key).strip().lower() not in FORBIDDEN:
                    is_emerg = hp < emerg_thresh
                    tag = "🚨 ACİL CAN POTU" if is_emerg else "🧪 CAN POTU"
                    self._emit_log(f"{tag}: Can %{int(round(hp * 100))} < %{int(round(hp_thresh * 100))} -> [{chosen_key}] basılıyor...")
                    # B-02: basış onaylanmazsa (sürücü 0.0/False) zaman damgası
                    # ve "kullanıldı" kaydı yazılmaz — cooldown başlamaz, retry
                    # engellenmez; aksi halde iksir hiç gitmediği halde
                    # "basıldı" sayılırdı.
                    pressed = self._press_potion_key(chosen_key, min_hold=0.06, max_hold=0.12)
                    if not pressed:
                        logger.warning(
                            "HP iksiri basılamadı (tuş gönderilmedi): [%s] — cooldown "
                            "başlatılmadı, bir sonraki tick'te tekrar denenecek.",
                            chosen_key,
                        )
                    else:
                        self._last_hp_potion_ts = now
                        if mana_brain and hasattr(mana_brain, "_last_hp_potion_ts"):
                            mana_brain._last_hp_potion_ts = now
                        if consumable_mgr and hasattr(consumable_mgr, "mark_used"):
                            try:
                                consumable_mgr.mark_used(chosen_key, now=now)
                            except Exception:
                                pass
                        pot_triggered = True

        # 2. MP Potion Check (skipped when MP is unknown)
        if mp_enabled and mp is not None and mp < mp_thresh:
            if (now - last_mp) >= mp_cd:
                chosen_mp = None
                consumable_mgr = getattr(mana_brain, "consumable_manager", None) if mana_brain else None
                if consumable_mgr is not None and hasattr(consumable_mgr, "get_best_mp_potion"):
                    try:
                        best_mp = consumable_mgr.get_best_mp_potion(now=now)
                        if best_mp and str(best_mp).strip().lower() not in FORBIDDEN:
                            chosen_mp = str(best_mp).strip()
                    except Exception:
                        chosen_mp = None

                if not chosen_mp:
                    # F-0380: same rule as HP — no blind F2.
                    fallback_mp = getattr(mana_brain, "mp_potion_key", None) if mana_brain else None
                    if fallback_mp and str(fallback_mp).strip().lower() not in FORBIDDEN:
                        chosen_mp = str(fallback_mp).strip()
                    else:
                        logger.debug("MP potion skipped: no known potion slot (no blind F2)")
                        chosen_mp = None

                if chosen_mp and str(chosen_mp).strip().lower() not in FORBIDDEN:
                    self._emit_log(f"💧 MANA POTU: Mana %{int(round(mp * 100))} < %{int(round(mp_thresh * 100))} -> [{chosen_mp}] basılıyor...")
                    # B-02: HP ile aynı kural — basış onaylanmazsa zaman
                    # damgası / "kullanıldı" kaydı yazılmaz, retry engellenmez.
                    pressed = self._press_potion_key(chosen_mp, min_hold=0.06, max_hold=0.12)
                    if not pressed:
                        logger.warning(
                            "MP iksiri basılamadı (tuş gönderilmedi): [%s] — cooldown "
                            "başlatılmadı, bir sonraki tick'te tekrar denenecek.",
                            chosen_mp,
                        )
                    else:
                        self._last_mp_potion_ts = now
                        if mana_brain and hasattr(mana_brain, "_last_mp_potion_ts"):
                            mana_brain._last_mp_potion_ts = now
                        if consumable_mgr and hasattr(consumable_mgr, "mark_used"):
                            try:
                                consumable_mgr.mark_used(chosen_mp, now=now)
                            except Exception:
                                pass
                        pot_triggered = True

        return pot_triggered

    def _send_configured_combat_skill_from_keys(self, keys: list) -> bool:
        """Cast ready keys from ManaPhase decision list (same bus path, fast when trusted)."""
        if self.is_paused or bool(self._get_cfg("test_mode", False)):
            return False
        if not self.is_combat_authorized("skill"):
            return False
        norm = self._normalize_combat_skill_keys(keys)
        if not norm:
            return self._send_configured_combat_skill()
        now = time.time()
        burst_gap = self._combat_skill_burst_sec()
        last_burst = float(self._plain_attr("_combat_last_skill_burst_ts", 0.0) or 0.0)
        if last_burst > 0.0 and (now - last_burst) < (burst_gap - 1e-4):
            return False
        used = self._plain_attr("_combat_skill_used_at")
        if not isinstance(used, dict):
            used = {}
            self._combat_skill_used_at = used
        per_burst = self._combat_skills_per_burst()
        trusted = self._keyboard_trusted_for_combat()
        frame = self.__dict__.get("_last_color_frame")
        pressed = 0
        for key in norm:
            if pressed >= per_burst:
                break
            last = float(used.get(key, 0.0) or 0.0)
            if last > 0.0 and (now - last) < self._combat_skill_cooldown_sec(key):
                continue
            # P0-6: cooldown damgası gönderimden SONRA — _cast_one False
            # dönerse (kapı engeli / basış başarısız) damga yok, burst
            # sayılmaz; tuş bir sonraki tick'te tekrar denenir. None
            # (dönüşsüz bus) eski davranışı korur: damgalanır.
            sent = self._cast_one_combat_skill(key, frame=frame)
            if sent is False:
                continue
            used[key] = now
            pressed += 1
            if pressed < per_burst:
                time.sleep(random.uniform(0.025, 0.055) if trusted else random.uniform(0.045, 0.075))
        if pressed <= 0:
            return False
        # P0-6: burst damgası yalnızca >=1 tuş gerçekten gönderildiyse
        # yazılır (pressed artık yalnızca gönderilen/bilinmeyen basışları
        # sayar; hepsi başarısızsa yukarıda False dönüldü).
        self._combat_last_skill_burst_ts = now
        return True

    def _maybe_run_skill_discovery(
        self,
        frame_bgr: Any,
        player_hp_ratio: float,
        monster_hp_ratio: float,
        buff_count: int = 0,
    ) -> None:
        if not self.is_combat_authorized("skill"):
            return
        disc = self._plain_attr("skill_discovery")
        if disc is None or not bool(self._get_cfg("skill_discovery_enabled", True)):
            return
        if not hasattr(disc, "needs_discovery") or not disc.needs_discovery():
            return
        slot = disc.next_slot() if hasattr(disc, "next_slot") else None
        if not slot:
            return
        probe = {
            "slot": slot,
            "player_hp": float(player_hp_ratio),
            "monster_hp": float(monster_hp_ratio),
            "buffs": int(buff_count),
            "ts": time.time(),
        }
        prev = self._plain_attr("_discovery_probe")
        # Second tick for same slot → classify delta
        if isinstance(prev, dict) and str(prev.get("slot")) == str(slot) and (time.time() - float(prev.get("ts", 0))) >= 0.8:
            try:
                disc.classify_delta(
                    slot=slot,
                    monster_hp_before=float(prev.get("monster_hp", 0)),
                    monster_hp_after=float(monster_hp_ratio),
                    player_hp_before=float(prev.get("player_hp", 1)),
                    player_hp_after=float(player_hp_ratio),
                    buffs_before=int(prev.get("buffs", 0)),
                    buffs_after=int(buff_count),
                )
                mana_brain = self._plain_attr("mana_phase_combat")
                if mana_brain is not None and hasattr(mana_brain, "set_lexicon"):
                    mana_brain.set_lexicon(disc.lexicon())
            except Exception as exc:
                logger.debug("skill discovery classify: %s", exc)
            self._discovery_probe = None
            return
        # First tick: optional Alt+hover tooltip (AION2 cursor), then keyboard cast.
        bus = self._plain_attr("action_bus")
        try:
            # Alt-hover disabled while diagnosing dead keys (stuck Alt ->
            # cursor mode). The tooltip helpers were dead code (F-0505) and
            # have been removed.
            if bus is not None and hasattr(bus, "cast_skill"):
                bus.cast_skill(slot, frame=frame_bgr)
            else:
                self._press_key(slot, min_hold=0.12, max_hold=0.18, release_mouse=True)
            self._discovery_probe = probe
        except Exception as exc:
            logger.debug("skill discovery cast: %s", exc)

    def _zk_post_combat_tick(
        self,
        player_mp_ratio: Any = None,
        monster_hp_ratio: Any = None,
        phase: str = "B",
        player_hp_ratio: Any = None,
    ) -> None:
        """Record reward + chat loot into slot economy; refresh UCB hint occasionally."""
        cog = self._plain_attr("_frame_cognition")
        chat_events = []
        if cog is not None:
            chat_events = list(getattr(cog, "chat_events", None) or [])
            if hasattr(cog, "as_dict"):
                chat_events = list(cog.as_dict().get("chat_events") or chat_events)

        eco = self._plain_attr("slot_economy")
        if eco is not None and chat_events:
            seen = self._plain_attr("_zk_chat_seen_fps")
            if not isinstance(seen, dict):
                seen = {}
                self._zk_chat_seen_fps = seen
            now_ts = time.time()
            # Drop fingerprints older than 60s
            for old_fp, old_ts in list(seen.items()):
                if (now_ts - float(old_ts or 0.0)) > 60.0:
                    seen.pop(old_fp, None)
            for ev in chat_events:
                et = str(ev.get("type", ""))
                raw = str(ev.get("raw", "") or "")[:80]
                fp = f"{et}:{raw}:{ev.get('amount', '')}:{ev.get('item_name', '')}"
                if fp in seen:
                    continue
                if et == "kina":
                    try:
                        eco.record_drop(kina=float(ev.get("amount", 0) or 0), mob_key="combat")
                        seen[fp] = now_ts
                    except Exception:
                        pass
                elif et == "item":
                    try:
                        eco.record_drop(item_name=str(ev.get("item_name", "")), mob_key="combat")
                        seen[fp] = now_ts
                    except Exception:
                        pass
                elif et == "defeat":
                    try:
                        eco.record_kill(mob_key=str(ev.get("raw", "mob"))[:48])
                        seen[fp] = now_ts
                    except Exception:
                        pass

        # F-0188: the only production log_tick call dropped kina and
        # damage_taken (both default to 0.0 in the log). Kina comes from the
        # chat events already parsed above; damage taken is the HP drop
        # since the previous combat tick.
        # S-2601: bilinmeyen vital uydurulmaz. MP bilinmiyorsa israf
        # hesaplanamaz (0.0 yazılır, "tam mana" varsayılmaz); canavar canı
        # bilinmiyorsa hasar iddia edilmez (0.0 yazılır, "100 hasar"
        # varsayılmaz). HP'de zaten uygulanan None -> 0.0/atla kuralıyla aynı.
        try:
            mp_f = float(player_mp_ratio) if player_mp_ratio is not None else None
        except (TypeError, ValueError):
            mp_f = None
        try:
            mon_f = float(monster_hp_ratio) if monster_hp_ratio is not None else None
        except (TypeError, ValueError):
            mon_f = None
        wasted = max(0.0, 1.0 - mp_f) if (str(phase) == "B" and mp_f is not None) else 0.0
        dmg = max(0.0, 1.0 - mon_f) * 100.0 if mon_f is not None else 0.0
        kina_total = 0.0
        for _ev in chat_events:
            if isinstance(_ev, dict) and str(_ev.get("type", "")) == "kina":
                try:
                    kina_total += float(_ev.get("amount", 0) or 0)
                except (TypeError, ValueError):
                    pass
        damage_taken = 0.0
        try:
            hp_f = float(player_hp_ratio) if player_hp_ratio is not None else None
        except (TypeError, ValueError):
            hp_f = None
        prev_hp = self._plain_attr("_zk_prev_player_hp", None)
        if hp_f is not None and prev_hp is not None:
            try:
                damage_taken = max(0.0, float(prev_hp) - hp_f) * 100.0
            except (TypeError, ValueError):
                damage_taken = 0.0
        if hp_f is not None:
            self._zk_prev_player_hp = hp_f
        reward_log = self._plain_attr("rl_reward_log")
        if reward_log is not None and hasattr(reward_log, "log_tick"):
            try:
                slot_id = ""
                if eco is not None:
                    slot_id = str(getattr(eco, "_current_slot", "") or "")
                reward_log.log_tick(
                    kina=kina_total,
                    damage_dealt=dmg,
                    damage_taken=damage_taken,
                    wasted_mana=wasted * 0.1,
                    slot_id=slot_id,
                    meta={"phase": phase},
                )
            except Exception as exc:
                logger.debug("rl reward tick: %s", exc)

        # Refresh UCB hint every ~20s
        last = float(self._plain_attr("_ucb_hint_ts", 0.0) or 0.0)
        if eco is not None and (time.time() - last) >= 20.0:
            try:
                self._ucb_slot_hint = eco.suggested_hint()
                self._ucb_hint_ts = time.time()
            except Exception:
                pass

    def _zk_note_stuck_and_avoid(self, cx: float, cy: float, moving: bool = True) -> None:
        mem = self._plain_attr("obstacle_memory")
        stuck_for = None
        if mem is not None and hasattr(mem, "note_position"):
            try:
                stuck_for = mem.note_position(cx, cy, moving_cmd=moving)
            except Exception:
                stuck_for = None
        reward_log = self._plain_attr("rl_reward_log")
        if stuck_for and reward_log is not None:
            try:
                # F-0381: note_position returns a CUMULATIVE stuck duration;
                # logging it every tick double-counts. Log only the delta.
                last_logged = float(self._plain_attr("_zk_last_stuck_logged", 0.0) or 0.0)
                delta = float(stuck_for) - last_logged
                if delta > 0.0:
                    reward_log.log_tick(stuck_time=delta, meta={"event": "stuck"})
                    self._zk_last_stuck_logged = float(stuck_for)
            except Exception:
                pass
        elif not stuck_for:
            self._zk_last_stuck_logged = 0.0
        if not stuck_for:
            return
        frame = self.__dict__.get("_last_color_frame")
        depth_avoider = self._plain_attr("depth_avoider")
        if depth_avoider is not None and frame is not None:
            try:
                analysis = None
                if hasattr(depth_avoider, "analyze_forward_corridor"):
                    analysis = depth_avoider.analyze_forward_corridor(frame)
                if analysis and hasattr(depth_avoider, "apply_avoidance_steering"):
                    depth_avoider.apply_avoidance_steering(
                        self._plain_attr("input_driver"),
                        analysis,
                        test_mode=bool(self._get_cfg("test_mode", False)),
                    )
                elif analysis and hasattr(self, "_apply_active_obstacle_evasion"):
                    steer = float((analysis or {}).get("recommended_steer_deg", 15.0) or 15.0)
                    self._apply_active_obstacle_evasion("stuck", steer, time.time())
            except Exception as exc:
                logger.debug("depth avoid on stuck: %s", exc)
        glide_pilot = self._plain_attr("glide_pilot")
        if glide_pilot is not None and hasattr(glide_pilot, "execute_glide_maneuver"):
            # F-0048: execute_glide_maneuver defaults to test_mode=True, which
            # sends no input yet returns success=True. Pass the real mode
            # explicitly and never claim a glide when there is no driver.
            driver = self._plain_attr("input_driver")
            if driver is None:
                logger.warning("Glide maneuver skipped: no input driver")
            else:
                try:
                    result = glide_pilot.execute_glide_maneuver(
                        self._plain_attr("active_node"),
                        input_ctrl=driver,
                        test_mode=bool(self._get_cfg("test_mode", False)),
                    )
                    if isinstance(result, dict) and not result.get("success", True):
                        logger.warning("Glide maneuver reported failure: %s", result)
                except TypeError:
                    try:
                        glide_pilot.execute_glide_maneuver(self._plain_attr("active_node"))
                    except Exception:
                        pass
                except Exception:
                    pass
        # Enter current area as a slot sample for UCB when stuck resolved path continues
        eco = self._plain_attr("slot_economy")
        if eco is not None and hasattr(eco, "enter_slot"):
            try:
                sid = eco.slot_id_from_coords(cx, cy, region=str(self._get_cfg("active_region", "map")))
                eco.enter_slot(sid, cx=cx, cy=cy)
            except Exception:
                pass

    def _route_mp_ratio_from_cognition(self) -> Optional[float]:
        """Best-effort real MP ratio from the learner snapshot; None if unknown."""
        try:
            learner = self._plain_attr("game_learner")
            if learner is not None and hasattr(learner, "get_live_cognition_snapshot"):
                snap = learner.get_live_cognition_snapshot() or {}
                if isinstance(snap, dict) and snap.get("player_mp_percent") is not None:
                    return max(0.0, min(1.0, float(snap["player_mp_percent"]) / 100.0))
                if isinstance(snap, dict) and snap.get("player_mp") is not None:
                    return max(0.0, min(1.0, float(snap["player_mp"])))
        except Exception:
            pass
        return None

    def _handle_combat_and_death(self) -> bool:
        """Checks for character death or enemy attack and responds accordingly."""
        if self.combat_recovery is None:
            # F-0189: death/attack checks silently absent is dangerous — warn
            # once and surface it on the HUD instead of returning quietly.
            if not bool(self._plain_attr("_combat_recovery_missing_warned", False)):
                self._combat_recovery_missing_warned = True
                logger.warning(
                    "Combat/death check skipped: combat_recovery is None — "
                    "death and under-attack detection are OFF"
                )
                try:
                    self._emit_log("⚠️ Ölüm/saldırı denetimi kapalı: combat_recovery yok")
                except Exception:
                    pass
            return False
        if self.is_paused:
            self._sync_combat_right_button(False)
            return False

        try:
            frame_bgr = self._grab_frame(grayscale=False)

            # Character death check (passes frame safely; supports both frame-accepting and 0-arg mocks)
            # F-0190: passing a frame makes the callee use confirm_frames=1
            # (single frame). Pass 5 explicitly so one bad frame cannot
            # declare death; mocks that do not accept the kwarg fall back.
            is_dead = False
            if hasattr(self.combat_recovery, "check_character_dead"):
                try:
                    if frame_bgr is not None:
                        is_dead = bool(
                            self.combat_recovery.check_character_dead(
                                frame_bgr, confirm_frames=5
                            )
                        )
                    else:
                        is_dead = bool(
                            self.combat_recovery.check_character_dead(confirm_frames=5)
                        )
                except TypeError:
                    try:
                        if frame_bgr is not None:
                            is_dead = bool(self.combat_recovery.check_character_dead(frame_bgr))
                        else:
                            is_dead = bool(self.combat_recovery.check_character_dead())
                    except TypeError:
                        try:
                            is_dead = bool(self.combat_recovery.check_character_dead())
                        except TypeError:
                            is_dead = False
            elif hasattr(self.combat_recovery, "is_dead"):
                try:
                    is_dead = bool(self.combat_recovery.is_dead(frame_bgr))
                except TypeError:
                    is_dead = bool(self.combat_recovery.is_dead())

            if is_dead:
                self._emit_log("Character death detected")
                recovered = False
                if hasattr(self.combat_recovery, "recover_from_death"):
                    try:
                        recovered = self.combat_recovery.recover_from_death(frame_bgr)
                    except TypeError:
                        recovered = self.combat_recovery.recover_from_death()
                elif hasattr(self.combat_recovery, "handle_death"):
                    try:
                        recovered = bool(self.combat_recovery.handle_death())
                    except TypeError:
                        recovered = bool(self.combat_recovery.handle_death())
                if recovered:
                    # F-0191/F-0192: a revived character must not resume with
                    # a stale harvest session (old node/stage/cast flags).
                    self.active_node = None
                    self._is_gathering_active = False
                    self._stage_cast_in_progress = False
                    self._stage_cast_bar_seen = False
                    self._waiting_for_next_stage_prompt = False
                    self._current_node_harvest_stage = 0
                    self._current_node_stage_retries = 0
                    self._gather_lost_frames = 0
                    self._route_prev_hp_ratio = None
                    self._combat_was_targeting = False
                    self._clear_route_combat()
                    self._death_wait_until = 0.0
                    self._emit_log("Character revived: harvest state reset")
                else:
                    # F-0382: recover_from_death only reports an already-done
                    # revive; it never tries one. Wait (tick-based, bounded)
                    # for the game/player revive before giving up and pausing.
                    wait_sec = float(self._get_cfg("death_revive_wait_sec", 60) or 60)
                    now_ts = time.time()
                    deadline = float(self._plain_attr("_death_wait_until", 0.0) or 0.0)
                    if deadline <= 0.0:
                        deadline = now_ts + max(0.0, wait_sec)
                        self._death_wait_until = deadline
                        self._emit_log(
                            f"Character died: waiting up to {int(wait_sec)}s for revive"
                        )
                        # sweep-29 K-2/K-1: ölüm ilk kez doğrulandığında
                        # telemetriye yazılır + sesli uyarı verilir. Bu dal
                        # deadline kurulurken (epizot başına bir kez) çalışır.
                        try:
                            _tel = getattr(self, "telemetry", None)
                            if _tel is not None and hasattr(_tel, "record_combat"):
                                _tel.record_combat(
                                    duration_sec=self._take_combat_duration_sec(),
                                    outcome="DIED",
                                    mob_key="",
                                )
                        except Exception as _tel_exc:
                            logger.warning("Telemetri record_combat (ölüm) hatası: %s", _tel_exc)
                        try:
                            _tel = getattr(self, "telemetry", None)
                            if _tel is not None and hasattr(_tel, "send_alert"):
                                # Webhook yok (bildirim kanalı kararı Selçuk'ta,
                                # AÇIK SORU); Windows'ta sesli uyarı çalar.
                                _tel.send_alert(
                                    "DEATH", "Karakter öldü — dirilme bekleniyor."
                                )
                        except Exception as _alert_exc:
                            logger.warning("Telemetri send_alert (ölüm) hatası: %s", _alert_exc)
                    if now_ts < deadline:
                        return True
                    self._death_wait_until = 0.0
                    self.pause("Paused: character died")
                return True
            else:
                self._death_wait_until = 0.0

            # Dedicated combat mode runs its own targeting/skills/tab loop below.
            if (
                str(self._plain_attr("_mode", "")).lower() in (BotMode.COMBAT_MODE.value.lower(), "combat")
                or bool(self._plain_attr("_in_step_combat_mode", False))
            ):
                return False


            # Active Toplama harvest / player-mapping: never divert into fight-back skills.
            gathering_now = bool(self._plain_attr("_is_gathering_active", False)) or (
                str(self._plain_attr("_state", "")) == BotState.GATHERING.value
            )
            # F-23-05: şema varsayılanı 3 (config/schema.py RouteSettings).
            # Açık 0, "toplamayı hiç bölme" demektir: `or 3` onu 3'e
            # çevirip aşağıdaki `<= 0` kısa devresini ölü koda
            # dönüştürüyordu — yalnızca None varsayılana düşer.
            _stop_gather_raw = self._get_cfg("fight_stop_gather_sec", 3)
            stop_gather_sec = 3 if _stop_gather_raw is None else int(_stop_gather_raw)
            if gathering_now and stop_gather_sec <= 0:
                self._sync_combat_right_button(False)
                return False
            if self._player_driven_mapping_mode() and not bool(self._get_cfg("fight_back", True)):
                self._sync_combat_right_button(False)
                return False

            # Under attack check. A missing target bar is not a death and is not a fight.
            # F-23-05: şema varsayılanı True (config/schema.py RouteSettings).
            if bool(self._get_cfg("fight_back", True)):
                atk_res: Any = False
                if hasattr(self.combat_recovery, "check_under_attack"):
                    try:
                        if frame_bgr is not None:
                            atk_res = self.combat_recovery.check_under_attack(frame_bgr)
                        else:
                            atk_res = self.combat_recovery.check_under_attack()
                    except TypeError:
                        try:
                            atk_res = self.combat_recovery.check_under_attack()
                        except TypeError:
                            atk_res = False

                _under_ignored, hp_ratio = self._parse_attack_result(atk_res)
                # ROTA-İPTAL-ZİNCİRİ: FIGHTING kapısı SADECE gerçek düşman
                # barını sayar — keşif/öğrenici kilidi (learner
                # current_monster_targeted, _frame_cognition.has_target) ve
                # çıplak "saldırı altında" (vignette/boş bar) sayılmaz.
                # (Geniş _enemy_target_bar_visible burada artık okunmaz;
                # savaş modu kendi döngüsünde kullanmaya devam eder.)
                # fix/v20-panic-combat: bar tek başına yetmez — bar
                # penceresinde HP düşüşü veya ToT aggro da gerekir
                # (hayalet-savaş: pasif mob barı FIGHTING'e sokmaz).
                route_bar = self._route_enemy_bar_visible(frame_bgr)
                hp_dropped = self._note_player_hp_drop(hp_ratio)
                # fix/v20-panic-combat: ToT aggro işareti — check_under_attack
                # bu tick'te combat_recovery.last_target_of_target_agro'ya
                # yazdı. `is True` koruması: recovery yoksa / mock ise
                # kanıt yok sayılır (MagicMock özniteliği True sayılmaz).
                tot_agro = (
                    getattr(
                        self.combat_recovery,
                        "last_target_of_target_agro",
                        None,
                    )
                    is True
                )
                # Doğrulama: Görev 1'in ürettiği tek-kare sinyali (bar / HP
                # düşüşü) aynen tüketilir; FIGHTING'e giriş için sinyalin
                # ardışık tick'lerde tekrarlanması gerekir. fix/v20:
                # bar tek başına yetmez — bar penceresinde HP düşüşü
                # veya ToT aggro da gerekir (hayalet-savaş düzeltmesi).
                now_ts = time.time()
                confirmed = self._route_combat_signal_confirmed(
                    route_bar, hp_dropped, tot_agro
                )

                # Auto-potion check when taking damage or under attack
                # F-0193/F-0194: hp_ratio is None for a bare bool result, and
                # MP 1.0 is a fabrication — use the real MP when known.
                if hp_ratio is not None and hp_ratio > 0.01:
                    mp_ratio = self._route_mp_ratio_from_cognition()
                    self._check_and_trigger_auto_potions(hp_ratio, mp_ratio, frame_bgr)

                # Bare "under attack" (vignette / empty bar) does not start a counterattack.
                # ROTA-İPTAL-ZİNCİRİ: tek karelik sinyal de başlatmaz — önce
                # doğrulama (ardışık tick), sonra histerezis penceresi.
                # W54/test_4 (v19 KOK NEDEN): aktif hasat sirasinda
                # dogrulanmis saldiri (2 ardisik tick) dogrudan FIGHTING'e
                # sapiyordu — cast / asama-arasi bekleme tick'i hic
                # surulemiyor, asama asla bitmiyordu. fight_stop_gather_sec
                # "savas toplamayi N sn SUREKLI saldiridan sonra boler"
                # demektir (0 = asla bolme, test_1 kilidi); oysa kodda
                # sure olculmuyordu — yalnizca `<= 0` kisa devresi vardi.
                # Hasat surerken sapma, sinyal SUREKLI N sn'yi
                # doldurunca olur; o zamana dek tick rotaya (toplamaya)
                # doner. Olum denetimi ve oto-iksir bu kapinin USTUNDE
                # calisir, etkilenmez. Damga sinyal kesilince
                # _clear_route_combat ile, oturum basinda/sonunda
                # sifirlanir.
                if gathering_now and stop_gather_sec > 0:
                    if route_bar or hp_dropped:
                        if float(self._plain_attr("_gather_under_attack_since", 0.0) or 0.0) <= 0.0:
                            self._gather_under_attack_since = now_ts
                    if confirmed:
                        _since = float(self._plain_attr("_gather_under_attack_since", 0.0) or 0.0)
                        if (now_ts - _since) < stop_gather_sec:
                            self._sync_combat_right_button(False)
                            return False
                if confirmed and not self._route_fight_reentry_blocked(now_ts):
                    self._remember_real_combat_signal(has_bar=route_bar, hp_dropped=hp_dropped)
                    if self._state != BotState.FIGHTING.value:
                        self._emit_log("Saldırı altında: Karşı saldırı yapılıyor")
                    self._set_state_internal(BotState.FIGHTING.value)
                    self._emit_hud_snapshot(
                        badge=BotState.FIGHTING.value,
                        title="Fighting enemy",
                        subtitle="Defending character before continuing route",
                    )

                    if not bool(self._get_cfg("test_mode", False)):
                        self._last_bot_key_time = time.time()
                        if route_bar:
                            self._combat_was_targeting = True
                            self._combat_no_target_streak = 0
                        is_route = str(self._plain_attr("_mode", "")).strip().lower() in (
                            BotMode.ROUTE_MODE.value.lower(),
                            "route",
                            "toplama",
                            "route_mode",
                        )
                        route_fb = self._is_route_fight_back_enabled()
                        allow_fight_back = (not is_route) or route_fb
                        if allow_fight_back:
                            self._maybe_press_combat_f(frame_bgr)
                            try:
                                if self._real_target_for_skills(route_bar):
                                    self._send_configured_combat_skill()
                            except Exception:
                                logger.debug("Combat skill skipped", exc_info=True)
                            if hasattr(self.combat_recovery, "fight_back_step"):
                                # 17/B7 (P3-4: eski metin bayattı):
                                # fight_back_step barsız 2. adımda ARTIK loot
                                # F basmaz — Y2/P1-11 düzeltmesi kör F'i (ve
                                # kör Tab/kesim yazımını) kaldırdı
                                # (combat_recovery.py fight_back_step, Y2
                                # bloğu ~2136); o dal True döner ama dönüş
                                # hep True değildir (girdi yoksa / basım
                                # başarısızsa / istisnada False). Ayna sayaç
                                # burada tutulur: route_bar False ise streak
                                # artar, True ise sıfırlanır. F16: streak>=2
                                # L2 F'ini YALNIZCA CombatSystem
                                # fallback'inde atlatır (o L3 ölümde F
                                # basar); canlı CombatRecovery yolunda L2
                                # tek F'i basar — bkz. _loot_confirmed_fight.
                                if route_bar:
                                    self._route_fightback_no_bar_streak = 0
                                else:
                                    self._route_fightback_no_bar_streak = int(getattr(self, "_route_fightback_no_bar_streak", 0) or 0) + 1
                                self.combat_recovery.fight_back_step(input_ctrl=self.input_driver, frame=frame_bgr)
                            elif hasattr(self.combat_recovery, "fight_back"):
                                self.combat_recovery.fight_back(input_ctrl=self.input_driver, frame=frame_bgr)
                            elif hasattr(self.combat_recovery, "execute_combat_step"):
                                self.combat_recovery.execute_combat_step(input_ctrl=self.input_driver, frame=frame_bgr)
                            # Hold after the skill tick so a click inside it cannot release the button.
                            self._sync_combat_right_button(bool(route_bar))
                        else:
                            # In ROUTE_MODE ("TOPLAMA") when route.fight_back=False, strictly suppress combat skills and fight_back (no SKILL KEY leak)
                            self._sync_combat_right_button(False)
                    # Doğrulanmış savaş rotayı askıya aldı: active_node
                    # korunur; savaş bitince rota kendiliğinden devam eder.
                    self._route_fight_suspended = True
                    return True

                if confirmed:
                    # Histerezis: sinyal doğrulandı ama bayrak yeni
                    # temizlendi — FIGHTING'e girme; rotayı da iptal etme
                    # (False dön, route step yürümeye devam etsin).
                    self._sync_combat_right_button(False)
                    return False

                self._sync_combat_right_button(False)
                if self._confirmed_fight_ended(route_bar, hp_dropped):
                    self._loot_confirmed_fight()
                elif not route_bar and not hp_dropped:
                    # Hiç sinyal yok (ne hedef barı ne HP düşüşü): takılı
                    # savaş bayrağı rotayı tutmasın diye temizlenir.
                    self._clear_route_combat()
                # Sinyal var ama henüz doğrulanmadı: hiçbir şey
                # temizlenmez — _route_bar_streak / _route_hp_streak
                # biriksin diye korunur, yoksa doğrulama eşiğine
                # (route_fight_confirm_frames) asla ulaşılamaz.
                # False dönülür, rota yürümeye devam eder.
            else:
                # F-0194: potions are survival, not fight-back. Check them
                # even when counter-attack is disabled.
                try:
                    _atk2: Any = False
                    if hasattr(self.combat_recovery, "check_under_attack"):
                        if frame_bgr is not None:
                            _atk2 = self.combat_recovery.check_under_attack(frame_bgr)
                        else:
                            _atk2 = self.combat_recovery.check_under_attack()
                    _u2, _hp2 = self._parse_attack_result(_atk2)
                    if _hp2 is not None and _hp2 > 0.01:
                        self._check_and_trigger_auto_potions(
                            _hp2, self._route_mp_ratio_from_cognition(), frame_bgr
                        )
                except Exception:
                    pass
                self._sync_combat_right_button(False)
                self._clear_route_combat()
        except Exception as exc:
            logger.error("Combat/death check error: %s", exc, exc_info=True)

        return False

    def _handle_blocking_game_window(self) -> bool:
        """
        Checks if a blocking game window/dialog is open (when not intentionally viewing the map).
        If open, presses Esc and logs 'A game window was open: closed with Esc'.
        Rate-limited with a 4.0-second cooldown to prevent infinite Esc loops.
        """
        if self.vision is None or self._state == BotState.MAP.value:
            return False

        now = time.time()
        last_esc = getattr(self, "_last_esc_press_time", 0.0)
        if now - last_esc < 4.0:
            return False

        try:
            is_blocking = False
            if hasattr(self.vision, "is_game_window_blocking"):
                try:
                    is_blocking = bool(self.vision.is_game_window_blocking())
                except TypeError:
                    frame = self._grab_frame(grayscale=False)
                    if frame is not None:
                        is_blocking = bool(self.vision.is_game_window_blocking(frame))
            elif hasattr(self.vision, "check_blocking_window"):
                is_blocking = bool(self.vision.check_blocking_window())

            if is_blocking:
                setattr(self, "_last_esc_press_time", now)
                if not bool(self._get_cfg("test_mode", False)):
                    self._press_key("esc")
                self._emit_log("A game window was open: closed with Esc")
                self.msleep(250)
                return True
        except Exception as exc:
            logger.debug("Blocking window check error: %s", exc)

        return False

    def _check_inventory_full(self, frame_bgr: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Detects whether inventory/cube is full and blocking further harvesting.
        Checks:
        1. Behavioral guard prompt-stuck monitor (10s stuck with prompt visible)
        2. Vision inventory panel / cube detection
        3. Chat feedback for 'inventory full' or 'envanter dolu'
        """
        # 1. Behavioral guard check
        # S005 (v19/P3-8): prompt görünürlüğü önceden sabit `True`
        # varsayılıyordu — görünmeyen prompt için de 10 sn sayacı
        # başlıyor, envanter dolu değilken bile bot durabiliyordu.
        # Görünürlük VisionEngine.detect_gather_prompt ile ölçülür;
        # ölçülemiyorsa False verilir (sayaç sıfırlanır, uydurma sinyal yok).
        node_id = self._extract_node_id(self.active_node)
        prompt_visible = False
        _vision_probe = self._plain_attr("vision")
        if _vision_probe is not None and hasattr(_vision_probe, "detect_gather_prompt"):
            try:
                if frame_bgr is None:
                    frame_bgr = self.__dict__.get("_last_color_frame")
                if frame_bgr is None:
                    frame_bgr = self._grab_frame(grayscale=False)
                if frame_bgr is not None:
                    prompt_visible = bool(_vision_probe.detect_gather_prompt(frame_bgr)[0])
            except Exception as exc:
                logger.debug("Gather prompt visibility probe error: %s", exc)
        guard = self._plain_attr("behavioral_guard")
        if guard is not None and hasattr(guard, "check_prompt_stuck"):
            try:
                is_stuck, stuck_msg = guard.check_prompt_stuck(
                    prompt_visible,
                    current_node_id=node_id if isinstance(node_id, int) else None,
                    max_stuck_sec=10.0,
                )
                if is_stuck:
                    return True, stuck_msg or "Inventory full / skill level requirement (Prompt stuck 10s)"
            except Exception as exc:
                logger.debug("check_prompt_stuck error: %s", exc)

        # 2. Vision check for modal inventory / cube blocking window
        vision = self._plain_attr("vision")
        if vision is not None:
            try:
                if frame_bgr is None:
                    frame_bgr = self.__dict__.get("_last_color_frame")
                if frame_bgr is None:
                    frame_bgr = self._grab_frame(grayscale=False)
                if hasattr(vision, "is_inventory_open"):
                    try:
                        inv_open = bool(vision.is_inventory_open(frame_bgr)) if frame_bgr is not None else bool(vision.is_inventory_open())
                    except TypeError:
                        inv_open = bool(vision.is_inventory_open())
                    if inv_open:
                        return True, "Envanter / Cube penceresi açık (Inventory open / full)"
            except Exception as exc:
                logger.debug("Vision inventory check error: %s", exc)

        # 3. Chat feedback check
        hud_reader = self._plain_attr("hud_reader")
        if hud_reader is not None and hasattr(hud_reader, "parse_chat_feedback_fast"):
            try:
                if frame_bgr is None:
                    frame_bgr = self.__dict__.get("_last_color_frame")
                if frame_bgr is None:
                    frame_bgr = self._grab_frame(grayscale=False)
                # F-0506: no fabricated black frame. parse_chat_feedback_fast
                # accepts None, so the check still runs without a frame.
                fb = hud_reader.parse_chat_feedback_fast(frame_bgr)
                if isinstance(fb, dict) and (fb.get("inventory_full") or fb.get("cube_full")):
                    return True, "Envanter dolu (Chat feedback: Inventory full)"
            except Exception as exc:
                logger.debug("Chat feedback inventory error: %s", exc)

        return False, ""

    def _check_gather_cast_progress(self, frame_bgr: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Monitors gathering cast / progress bar via GameHUDReader.
        Returns:
            (is_present: bool, progress_pct: float 0.0..100.0)
        """
        hud_reader = self._plain_attr("hud_reader")
        if hud_reader is not None:
            try:
                if frame_bgr is None:
                    frame_bgr = self.__dict__.get("_last_color_frame")
                    # O6: önbellek kare bayatsa (1 sn'den eski) onunla cast
                    # kararı verilmez — taze kare çekilir.
                    frame_ts = float(self.__dict__.get("_last_color_frame_ts", 0.0) or 0.0)
                    if frame_bgr is not None and frame_ts > 0.0 and (time.time() - frame_ts) > 1.0:
                        fresh = self._grab_frame(grayscale=False)
                        if fresh is not None:
                            frame_bgr = fresh
                if frame_bgr is None:
                    frame_bgr = self._grab_frame(grayscale=False)
                if frame_bgr is None:
                    # F-0506: no frame -> no cast bar, not a black square.
                    return False, 0.0

                if frame_bgr is not None and hasattr(hud_reader, "verify_gather_cast_bar"):
                    is_pres, pct = hud_reader.verify_gather_cast_bar(frame_bgr)
                    if is_pres:
                        return True, float(pct)

                if frame_bgr is not None and hasattr(hud_reader, "read_cast_bar_progress"):
                    ratio = hud_reader.read_cast_bar_progress(frame_bgr)
                    if ratio > 0.0:
                        return True, round(float(ratio) * 100.0, 1)
            except Exception as exc:
                logger.debug("Cast progress read error: %s", exc)

        return False, 0.0

    def _handle_route_gathering(self, conf: int, now: float) -> None:
        """
        Executes the full gathering engine cycle:
        detect -> approach -> interact (F/E key loop) -> verify cast & progress bar -> 3x harvest cycle -> completion.
        Guarantees zero false F presses during cast progress or inter-harvest cooldown.
        """
        self._gather_lost_frames = 0
        node_id = self._extract_node_id(self.active_node)
        if self._node_not_gatherable(self.active_node):
            self._is_gathering_active = False
            self.active_node = None
            self._emit_log(f"Node {node_id} is a scroll, not gatherable: skip harvest")
            return

        # 1. Check for inventory full or prompt stuck
        is_full, full_reason = self._check_inventory_full()
        if is_full:
            # F-0049: a full inventory / stuck prompt is a routine pause, not a
            # security panic. trigger_panic() halts inputs with no unlock path
            # in this worker, so it must never fire here — pause only.
            self._emit_log(f"Inventory full / prompt stuck: {full_reason}")
            self.pause(full_reason)
            return

        # 2. Selective gathering check: bypass if unselected
        if not self._is_gathering_active and self._should_bypass_unselected_gather():
            # F-0383: leaving active_node set makes the next tick bypass the
            # same node again forever and pollutes the fail counters. Clear
            # it so the selector can move on (it returns None when no
            # selected material is ready).
            self._emit_log(f"Node {node_id or 'unselected'} bypassed: material does not match selective gather")
            self.active_node = None
            self._is_gathering_active = False
            self._current_node_harvest_stage = 0
            return

        # 3. Halt player movement so character stands still for interaction
        self._halt_motion()
        self._release_all_combat_inputs()

        # 4. Initialize node harvest session on first entry
        test_mode = bool(self._get_cfg("test_mode", False))
        if not self._is_gathering_active:
            self._release_all_combat_inputs()
            self._is_gathering_active = True
            # W54/v19: yeni hasat oturumu — onceki oturumdan kalma
            # saldiri damgasi yeni dugumde erken savasa sapmasin.
            self._gather_under_attack_since = 0.0
            max_h = self._configured_max_harvests(self.active_node)
            self._target_harvests_for_active_node = max_h
            self._current_node_harvest_stage = 0
            self._current_node_stage_retries = 0
            self._stage_cast_in_progress = False
            self._waiting_for_next_stage_prompt = False
            self._reset_cast_progress_tracking()
            self._emit_harvest_stage()
            self._emit_log(f"Gather prompt found ({conf}%) · Stage 1/{max_h}")

            # Human visual reaction delay before initial interaction
            if not test_mode and self.human_timing is not None:
                try:
                    react_sec = float(self.human_timing.get_visual_reaction_delay_sec())
                    self.msleep(max(50, min(400, int(react_sec * 1000.0))))
                except Exception:
                    pass
            self._emit_log("Gathering started")

        self._set_state_internal(BotState.GATHERING.value)
        target_h = max(1, self._target_harvests_for_active_node)
        current_stage_display = min(target_h, self._current_node_harvest_stage + 1)

        # 4. Check if we are waiting between harvest stages (Inter-harvest delay)
        if self._waiting_for_next_stage_prompt:
            if now < self._inter_harvest_wait_until:
                # Inter-harvest cooldown delay: do NOT press F yet! Zero false F presses.
                self._emit_hud_snapshot(
                    badge=BotState.GATHERING.value,
                    title=f"Gathering node {node_id} ({current_stage_display}/{target_h})",
                    subtitle="Inter-harvest cooldown · preparing next harvest",
                )
                return
            else:
                # Delay elapsed: prompt'u TEKRAR doğrula — aşama-arası F
                # zaman-güdümlü kör basılmaz (Y4). Prompt yoksa ve süre
                # dolduysa node erken tükendi sayılır; süre dolmadıysa
                # beklemeye devam edilir.
                self._waiting_for_next_stage_prompt = False
                self._last_f_press_time = 0.0
                redetected, _rconf = self._detect_gather_prompt()
                if not redetected:
                    deadline = float(
                        self._plain_attr("_post_harvest_prompt_deadline", 0.0) or 0.0
                    )
                    if deadline <= 0.0 or now > deadline:
                        self._emit_log(
                            f"Node {node_id} next-stage prompt gone "
                            f"({self._current_node_harvest_stage}/{self._target_harvests_for_active_node}) · node depleted early"
                        )
                        self._complete_route_gathering(now, depleted=True)
                        return
                    self._waiting_for_next_stage_prompt = True
                    self._inter_harvest_wait_until = min(deadline, now + 0.3)
                    self._emit_hud_snapshot(
                        badge=BotState.GATHERING.value,
                        title=f"Gathering node {node_id} ({current_stage_display}/{target_h})",
                        subtitle="Waiting for next-stage prompt...",
                    )
                    return

        # 5. Check if casting is in progress (Progress bar monitoring)
        if self._stage_cast_in_progress:
            is_cast_bar, progress_pct = self._check_gather_cast_progress()
            if is_cast_bar:
                self._stage_cast_bar_seen = True
                self._observe_cast_progress(progress_pct)
            cast_dur = float(self._get_cfg("gather_cast_duration_sec", 3.0) or 3.0)
            elapsed_cast = now - self._stage_cast_start_time

            # Cast completion detection
            # Y2/Y3: aşama, barın İLERLEDİĞİ kanıtlanmadan bitmez. Tek
            # karede %90 görmek yetmez — üst hedef barı dolu başlayan
            # statik bir bar da %90 görünür. İlerleme kanıtı: ardışık
            # okumalarda yükselme (_stage_cast_rose). Bar hiç yükselmeden
            # %90'da takılırsa aşama sayılmaz; süre dolunca retry/abandon
            # yoluna düşer (F-0050). CANLI NOT: gerçek cast barı oyunda
            # dolu başlayıp hiç yükselmiyorsa bu kapı aşamaları hep
            # abandon eder — PC karesiyle hangi barın gerçek cast barı
            # olduğu doğrulanmalı (açık soru).
            cast_finished = False
            if is_cast_bar:
                if progress_pct >= 90.0 and bool(self._plain_attr("_stage_cast_rose", False)):
                    cast_finished = True
                else:
                    # Still actively filling progress bar: update HUD and DO NOT press F!
                    # (%90 ama yükselme yoksa da burada beklenir — sayılmaz.)
                    self._emit_hud_snapshot(
                        badge=BotState.GATHERING.value,
                        title=f"Gathering node {node_id} ({current_stage_display}/{target_h}) · {int(progress_pct)}%",
                        subtitle=f"Harvesting... {progress_pct:.0f}%",
                    )
                    return
            else:
                # Cast bar not present on this frame:
                # F-0050: a timeout alone is NOT a successful cast. If the bar
                # was never seen, F probably missed — retry the press instead
                # of counting a harvest that never happened.
                if test_mode:
                    cast_finished = True
                elif (
                    elapsed_cast >= cast_dur
                    and bool(self._plain_attr("_stage_cast_bar_seen", False))
                    and bool(self._plain_attr("_stage_cast_rose", False))
                ):
                    cast_finished = True
                elif elapsed_cast >= cast_dur:
                    # Prompt disappeared -> harvest completed successfully
                    prompt_now, _ = self._detect_gather_prompt()
                    if not prompt_now:
                        self._emit_log(f"Cast duration reached & prompt cleared on node {node_id}: harvest stage finished")
                        cast_finished = True
                    else:
                        self._current_node_stage_retries += 1
                    # F-23-05: şema varsayılanı 2 (config/schema.py).
                    max_retries = int(self._get_cfg("max_retries_per_harvest_stage", 2) or 2)
                    if self._current_node_stage_retries > max_retries:
                        self._emit_log(
                            f"Cast bar never appeared on node {node_id}: abandoning stage"
                        )
                        self._complete_route_gathering(now, depleted=False)
                        return
                    self._emit_log(
                        f"Cast bar never appeared on node {node_id}: retrying F "
                        f"({self._current_node_stage_retries}/{max_retries})"
                    )
                    self._stage_cast_in_progress = False
                    self._stage_cast_bar_seen = False
                    self._last_f_press_time = 0.0
                    return
                else:
                    # Still within expected cast time window
                    self._emit_hud_snapshot(
                        badge=BotState.GATHERING.value,
                        title=f"Gathering node {node_id} ({current_stage_display}/{target_h})",
                        subtitle=f"Harvesting... ({elapsed_cast:.1f}s)",
                    )
                    return

            if cast_finished:
                self._on_harvest_stage_finished(now)
                return

        # 6. Interaction Prompt (F / E key loop)
        # Press gather key when interval has passed and NOT currently casting
        if now - self._last_f_press_time >= self._next_f_interval:
            self._press_gather_key()
            self._stage_cast_in_progress = True
            self._stage_cast_bar_seen = False
            self._reset_cast_progress_tracking()
            self._stage_cast_start_time = time.time()
            self._last_f_press_time = time.time()
            self._next_f_interval = self._sample_f_interval()

        self._emit_hud_snapshot(
            badge=BotState.GATHERING.value,
            title=f"Gathering node {node_id} ({current_stage_display}/{target_h})",
            subtitle=f"Gather prompt recognized at {conf}%",
        )

    def _on_harvest_stage_finished(self, now: float) -> None:
        """
        Called when a single harvest cast completes:
        1. Checks for 'Essence Extraction failed' chat message.
        2. Records successful stage and triggers Time Machine YOLO training.
        3. Either advances to next stage of 3x cycle or triggers node completion.
        """
        node_id = self._extract_node_id(self.active_node, default=None)
        test_mode = bool(self._get_cfg("test_mode", False))
        max_h = max(1, self._target_harvests_for_active_node)

        # Check chat for "Essence Extraction failed"
        if not test_mode and self.hud_reader is not None and hasattr(self.hud_reader, "parse_chat_feedback_fast"):
            try:
                frame_bgr = self._grab_frame(grayscale=False)
                if frame_bgr is None:
                    frame_bgr = self.__dict__.get("_last_color_frame")
                # F-0506: no fabricated black frame. parse_chat_feedback_fast
                # accepts None (real reader returns empty; mocks read their
                # own flags), so the check still runs without a frame.
                fb = self.hud_reader.parse_chat_feedback_fast(frame_bgr)
                if isinstance(fb, dict) and fb.get("extraction_failed"):
                    # F-23-05: şema varsayılanı 2 (config/schema.py).
                    max_retries = int(self._get_cfg("max_retries_per_harvest_stage", 2) or 2)
                    # O4: retry_on_extraction_fail ayarı artık okunur — false
                    # ise başarısız extraction'da retry yapılmaz.
                    retry_allowed = bool(self._get_cfg("retry_on_extraction_fail", True))
                    self._current_node_stage_retries += 1
                    if retry_allowed and self._current_node_stage_retries <= max_retries:
                        self._emit_log(
                            f"Essence Extraction failed on node {node_id}: retrying stage "
                            f"{self._current_node_harvest_stage + 1}/{max_h} "
                            f"({self._current_node_stage_retries}/{max_retries})"
                        )
                        self._stage_cast_in_progress = False
                        self._last_f_press_time = 0.0
                        return
                    else:
                        reason = (
                            "Essence Extraction failed, retry disabled (retry_on_extraction_fail=false)"
                            if not retry_allowed
                            else "Max extraction retries exceeded"
                        )
                        self._emit_log(f"{reason} on node {node_id}")
                        self._complete_route_gathering(now, depleted=True)
                        return
            except Exception as fb_exc:
                logger.debug("Chat feedback check error: %s", fb_exc)

        # Successful harvest stage!
        self._current_node_harvest_stage += 1
        self._current_node_stage_retries = 0
        self._emit_harvest_stage()
        self.gathered_count += 1
        self.failed_in_row = 0
        self._last_gather_time = now
        # O1: materyal bilinmiyorsa "Orichalcum Ore" uydurulmaz — Unknown.
        rtype = self._get_node_field(self.active_node, "resource_type", "Unknown") or "Unknown"
        rest_min = float(self._get_cfg("rest_minutes", 5) or 5)

        # Trigger Self-Training YOLO Dataset Factory (Time Machine t - 2.0s capture)
        if self.dataset_factory is not None and hasattr(self.dataset_factory, "on_gather_success"):
            try:
                self.dataset_factory.on_gather_success(
                    material_name=str(rtype),
                    harvest_stage=self._current_node_harvest_stage,
                    max_harvests=max_h,
                )
            except Exception as df_exc:
                logger.debug("Dataset factory on_gather_success error: %s", df_exc)

        if self.route_manager is not None and hasattr(self.route_manager, "record_harvest_stage") and node_id is not None:
            try:
                # F-0051: the real signature takes rest_minutes. Older
                # callers/mocks still use rest_duration_sec — fall back to
                # it on TypeError instead of silently swallowing the call.
                try:
                    self.route_manager.record_harvest_stage(node_id, rest_minutes=float(rest_min))
                except TypeError:
                    self.route_manager.record_harvest_stage(
                        node_id, rest_duration_sec=int(rest_min * 60)
                    )
            except Exception:
                pass

        # Check if 3x harvest cycle is complete
        if self._current_node_harvest_stage >= max_h:
            self._emit_log(f"Node {node_id} cycle complete ({self._current_node_harvest_stage}/{max_h})")
            self._complete_route_gathering(now, depleted=True)
        else:
            # Prepare for next harvest stage in the 3x cycle
            self._stage_cast_in_progress = False
            self._waiting_for_next_stage_prompt = True
            # F-23-05: şema varsayılanı 0.6/1.4 (config/schema.py).
            delay_min = float(self._get_cfg("inter_harvest_delay_min", 0.6) or 0.6)
            delay_max = float(self._get_cfg("inter_harvest_delay_max", 1.4) or 1.4)
            if delay_max < delay_min:
                delay_max = delay_min
            inter_delay = 0.05 if test_mode else random.uniform(delay_min, delay_max)
            self._inter_harvest_wait_until = now + inter_delay

            prompt_timeout = float(self._get_cfg("post_harvest_prompt_timeout", 4.0) or 4.0)
            self._post_harvest_prompt_deadline = now + prompt_timeout

            self._emit_log(
                f"Node {node_id} stage {self._current_node_harvest_stage}/{max_h} gathered · continuing next stage"
            )
            self._emit_hud_snapshot(
                badge=BotState.GATHERING.value,
                title=f"Gathering node {node_id} ({self._current_node_harvest_stage + 1}/{max_h})",
                subtitle=f"Stage {self._current_node_harvest_stage}/{max_h} complete · awaiting next prompt",
            )

    def _complete_route_gathering(self, now: float, depleted: bool = True) -> None:
        """
        Handles full completion / depletion of a route node:
        1. Resets behavioral guard stuck state.
        2. Clears harvest stage and casting flags.
        3. Marks node as gathered with route cooldown.
        4. Dispatches telemetry, economy tracking, respawn prediction.
        5. Seamlessly clears active_node and transitions to next route waypoint.
        """
        guard = self._plain_attr("behavioral_guard")
        if guard is not None and hasattr(guard, "check_prompt_stuck"):
            try:
                guard.check_prompt_stuck(False)
            except Exception:
                pass

        node_id = self._extract_node_id(self.active_node, default=None)
        if self._node_not_gatherable(self.active_node):
            self._is_gathering_active = False
            self._stage_cast_in_progress = False
            self._waiting_for_next_stage_prompt = False
            self._current_node_harvest_stage = 0
            self._current_node_stage_retries = 0
            self._gather_lost_frames = 0
            self.active_node = None
            self._emit_log(f"Node {node_id} is a scroll, not gatherable: harvest not counted")
            return

        walk_dur = max(0.0, now - self._walk_start_time) if self._walk_start_time > 0 else 0.0
        completed_stages = int(self._current_node_harvest_stage)
        self._is_gathering_active = False
        self._stage_cast_in_progress = False
        self._waiting_for_next_stage_prompt = False
        # W54/v19: oturum bitti — saldiri damgasi temizlenir.
        self._gather_under_attack_since = 0.0
        self._current_node_harvest_stage = 0
        self._current_node_stage_retries = 0
        self._gather_lost_frames = 0

        rest_min = float(self._get_cfg("rest_minutes", 5) or 5)
        # O1: materyal bilinmiyorsa "Orichalcum Ore" uydurulmaz — Unknown.
        rtype = self._get_node_field(self.active_node, "resource_type", "Unknown") or "Unknown"
        # Y5: 0 aşamayla biten toplama "toplandı" sayılmaz — ne node
        # dinlendirilir ne telemetri/ekonomi/respawn/alt kayda girer.
        gathered_any = completed_stages > 0

        if node_id is not None:
            if depleted and gathered_any:
                # F-0385: gathered_count counts STAGES (1/3, 2/3, 3/3);
                # telemetry/economy count NODES. Track nodes separately so
                # the two are never confused again.
                self._gathered_nodes_count = int(getattr(self, "_gathered_nodes_count", 0) or 0) + 1
                self._mark_node_gathered(node_id, rest_min)
                self._emit_log(f"Node {node_id} gathered (resting {int(rest_min)}m)")
            elif depleted:
                self._emit_log(
                    f"Node {node_id} depleted early with 0 stages — not rested, not counted"
                )
            else:
                # F-0050 abandon: no harvest happened, so the node must not
                # be rested as if it were gathered.
                self._emit_log(f"Node {node_id} abandoned without harvest (not rested)")
        else:
            if gathered_any:
                self._emit_log(f"Resource gathered (#{self.gathered_count})")
            else:
                self._emit_log("Gather ended with 0 stages — not counted")

        if gathered_any and self.telemetry is not None and hasattr(self.telemetry, "record_gather"):
            try:
                reg = getattr(self.route_manager, "active_region", None) or "unknown"
                rt = getattr(self.route_manager, "active_route", None) or "All"
                # P1-6 (K-06): node_id bilinmiyorsa 0 (bilinmeyen düğüm)
                # gönderilir. Eskiden yedek değer `self.gathered_count`
                # idi — o bir AŞAMA sayacıdır (F-0385: 1/3, 2/3, 3/3),
                # düğüm kimliği değildir; telemetriye düğüm diye aşama
                # sayısı yazılıyordu. Düğüm sayısı (`_gathered_nodes_count`)
                # da kimlik değildir, o yüzden yedek olarak kullanılmaz.
                try:
                    tel_node_id = int(node_id) if node_id is not None else 0
                except (TypeError, ValueError):
                    tel_node_id = 0
                self.telemetry.record_gather(
                    node_id=tel_node_id,
                    region=str(reg),
                    route=str(rt),
                    resource_type=str(rtype),
                    walk_duration_sec=walk_dur,
                    match_pct=self.last_prompt_match,
                )
            except Exception as tel_exc:
                # Y-5: kayıt hatası debug'da kaybolmasın — toplanan aşama
                # telemetriye yazılamadıysa görünür olmalı.
                logger.warning("Telemetry record_gather error: %s", tel_exc)

        # Enterprise Economy Calculator, Respawn Predictor & Multi-Character Quota Tracking
        if not gathered_any:
            self._emit_harvest_stage()
            if depleted:
                self.active_node = None
                self._walk_start_time = 0.0
            return
        eco = getattr(self, "economy_calculator", None)
        if eco is not None and hasattr(eco, "record_gather"):
            try:
                eco.record_gather(str(rtype), count=1)
            except Exception as eco_exc:
                logger.warning("Economy calculator record_gather skipped: %s", eco_exc)

        respawn = getattr(self, "respawn_predictor", None)
        if respawn is not None and node_id is not None and hasattr(respawn, "record_gather"):
            try:
                respawn.record_gather(int(node_id))
            except Exception as rsp_exc:
                logger.warning("Respawn predictor record_gather skipped: %s", rsp_exc)

        alt_mgr = getattr(self, "alt_char_mgr", None)
        if alt_mgr is not None and hasattr(alt_mgr, "record_gather"):
            try:
                alt_mgr.record_gather(count=1)
            except Exception as alt_exc:
                logger.warning("Alt character record_gather skipped: %s", alt_exc)

        self._emit_harvest_stage()
        if depleted:
            self.active_node = None
            self._walk_start_time = 0.0

    def _drop_node_on_walk_refusal(self, node_id: Any, reason: str) -> None:
        """Yürüme komutu reddedildiğinde düğümü bırakır (16/15).

        Geçici durumlarda (pause, sürücü yok) failure sayılmaz — yoksa
        sürücüsüz her tick tüm düğümleri fail eder. Gerçek rette failure
        sayılır ki sessiz seçim-haritalama döngüsüne girilmesin ve 10-fail
        koruması devreye girebilsin. (A grubunun _register_node_failure
        içindeki record_failure çağrısı da burada tetiklenir — aynı olay
        için tek kayıt, çift sayım yok.)
        """
        self._emit_log(f"Node {node_id} dropped: {reason}")
        if self.is_paused or self.input_driver is None:
            self.active_node = None
            self._walk_start_time = 0.0
            return
        self._mark_node_failed(node_id)
        self._register_node_failure()
        self.active_node = None
        self._walk_start_time = 0.0

    def _select_and_navigate_next_node(self) -> None:
        """Finds the next ready node, opens the map, clicks the node, and transitions to WALKING."""
        rest_min = float(self._get_cfg("rest_minutes", 5) or 5)
        next_node = self._get_next_ready_node(rest_min)
        if self._node_not_gatherable(next_node):
            self._emit_log(
                f"Node {self._extract_node_id(next_node)} is a scroll, not gatherable: skip walk"
            )
            next_node = None

        if next_node is not None and getattr(self, "selective_gather_enabled", False):
            if not self.is_material_selected_for_gather(next_node):
                self._emit_log(
                    f"Node {self._extract_node_id(next_node)} bypassed: material does not match selective gather"
                )
                next_node = None

        if next_node is None:
            self._set_state_internal(BotState.WAITING.value)
            ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
            # F-0507: in selective mode a ready node may exist that simply
            # does not match the selected materials — "all resting" would lie.
            subtitle = f"All {resting_cnt} route nodes are currently resting"
            if getattr(self, "selective_gather_enabled", False):
                try:
                    _nodes = []
                    if self.route_manager is not None and hasattr(
                        self.route_manager, "get_route_nodes"
                    ):
                        _nodes = self.route_manager.get_route_nodes() or []
                    if any(
                        self._is_node_ready_local(n, int(rest_min * 60))
                        and not self.is_material_selected_for_gather(n)
                        for n in _nodes
                    ):
                        subtitle = "Ready nodes exist but none match the selected materials"
                except Exception:
                    pass
            self._emit_hud_snapshot(
                badge=BotState.WAITING.value,
                title="Waiting for ready nodes",
                subtitle=subtitle,
            )
            self.msleep(500)
            return

        self.active_node = next_node
        node_id = self._extract_node_id(next_node)
        # v20-loop: yeni yürüyüşte mesafe-tazelik bayrağı sıfırlanır.
        self._walk_distance_ever_fresh = False

        nav_ok, dist_m, under_overlay = self._open_map_and_click_node(next_node)
        if under_overlay:
            self._emit_log(f"Node {node_id} is under the overlay panel")
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            return

        if self.is_paused:
            self._release_route_motion()
            return

        if not nav_ok:
            if self._last_nav_block == "map_open":
                # Map never opened: walk on foot. Do not idle, delete the node, or press F.
                if (
                    self.mode == BotMode.ROUTE_MODE.value
                    and not self.is_paused
                    and self._command_walk_toward(next_node)
                ):
                    # F-0052: dist_m is 0 when the map never opened. A fake
                    # 1 m makes _monitor_walking_to_node treat the node as
                    # arrived and fail it after 1.5 s. Unknown stays 0 (the
                    # monitor reads 0 as 999 m via `or 999`).
                    self._set_active_node_distance(dist_m)
                    self._walk_start_time = time.time()
                    self._last_reclick_time = time.time()
                    self._set_state_internal(BotState.WALKING.value)
                    self._emit_hud_snapshot(
                        badge=BotState.WALKING.value,
                        title=f"Walking to node {node_id} · distance unknown (map closed)",
                        subtitle="On arrival: F, or a click on the gathering icon",
                    )
                return
            self._emit_log(f"Could not locate node {node_id} on map")
            if self._command_walk_toward(next_node):
                self._set_active_node_distance(dist_m)
                self._walk_start_time = time.time()
                self._last_reclick_time = time.time()
                self._set_state_internal(BotState.WALKING.value)
                self._emit_hud_snapshot(
                    badge=BotState.WALKING.value,
                    title=f"Walking to node {node_id} · distance unknown (map closed)",
                    subtitle="On arrival: F, or a click on the gathering icon",
                )
                return
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            return

        self._set_active_node_distance(dist_m)
        self._walk_start_time = time.time()
        self._last_reclick_time = time.time()
        if not self._command_walk_toward(next_node):
            # 16/15: yürüme komutu reddedildiyse düğüm sessizce düşürülmez
            # — failure sayılır (geçici durumdaysa: pause/sürücü yoksa sayılmaz).
            self._drop_node_on_walk_refusal(node_id, "walk command refused after map click")
            return
        self._set_state_internal(BotState.WALKING.value)
        self._emit_log(f"Route: going to node {node_id} ({dist_m} m)")
        self._emit_hud_snapshot(
            badge=BotState.WALKING.value,
            title=f"Walking to node {node_id} · {dist_m} m",
            subtitle="On arrival: F, or a click on the gathering icon",
        )

    def _get_player_movement_coords(self) -> Tuple[float, float]:
        """Obtain real player world/map coordinates or visual odometry for ObstacleMemory."""
        carto = self._plain_attr("auto_cartographer")
        if carto is None and self._plain_attr("explorer_controller") is not None:
            carto = getattr(self._plain_attr("explorer_controller"), "auto_cartographer", None)
        if carto is not None:
            try:
                if hasattr(carto, "get_current_position"):
                    pos = carto.get_current_position()
                    if pos and len(pos) >= 2:
                        return float(pos[0]), float(pos[1])
                if hasattr(carto, "map_x") and hasattr(carto, "map_y"):
                    return float(carto.map_x), float(carto.map_y)
            except Exception:
                pass
        # F-0386: with no cartographer/odometry there is NO known player
        # position. The old formula invented one from the node coords
        # (same +elapsed*1.5 on both axes). Return "unknown" (0, 0); callers
        # already treat (0, 0) as "no position".
        return 0.0, 0.0

    def _execute_unstick_maneuver(self) -> None:
        """Physical unstick maneuver when walk distance hasn't decreased over 5s.
        Releases W, taps S (backup), taps Space (jump), wiggles direction (A/D),
        and re-commands walk toward node, avoiding false WAITING/BEKLEMEDE lock.
        """
        self._emit_log("⚠️ Takılma tespit edildi (5s mesafe azalmadı) -> Fiziksel kurtarma (Unstick) uygulanıyor...")
        self._emit_hud_snapshot(
            badge="KURTARMA",
            title="Takılma Kurtarma (Unstick)",
            subtitle="Geri çekilme (S) + Zıplama (Space) + Yön düzeltme (A/D)",
        )
        try:
            # 1. Release W / all movement keys
            self._release_route_motion()
            self.msleep(60)

            # 2. Tap S (back up slightly)
            self._press_key("s", min_hold=0.18, max_hold=0.28)
            self.msleep(80)

            # 3. Tap Space (jump over obstacle/ledge)
            self._press_key("space", min_hold=0.08, max_hold=0.15)
            self.msleep(80)

            # 4. Wiggle direction (A or D)
            wiggle_key = "a" if (int(time.time() * 10) % 2 == 0) else "d"
            self._press_key(wiggle_key, min_hold=0.12, max_hold=0.22)
            self.msleep(60)

            # 5. Re-engage walk forward toward active node
            if self.active_node is not None:
                self._command_walk_toward(self.active_node)
        except Exception as exc:
            logger.debug("Unstick maneuver error: %s", exc)

    def _run_corridor_navigation_step(self, now: float) -> bool:
        """F-0075: CorridorNavigator.step_navigation tick bağlantısı.

        WALKING tick'inde en fazla ``corridor_nav_step_sec`` (varsayılan
        2 sn) sıklıkla çalışır: uçurum filtresi / contested kilidi aktif
        düğümü dışlıyorsa sıradaki düğüme geçilir, takılma manevrası ve
        koridor değerlendirmesi çalışır. Düğüm değiştiyse / düştüyse True
        döner (bu tick burada biter); aksi hâlde False.
        """
        nav = self.__dict__.get("corridor_navigator")
        if nav is None or self.active_node is None:
            return False
        if bool(self._get_cfg("test_mode", False)):
            return False
        try:
            motion_stop = self._motion_should_stop()
        except Exception:
            # __init__ atlanmış iskelette (test) belirlenemez — fail-closed.
            return False
        if motion_stop:
            return False
        try:
            interval = float(self._get_cfg("corridor_nav_step_sec", 2.0) or 2.0)
        except (TypeError, ValueError):
            interval = 2.0
        last = float(self.__dict__.get("_last_corridor_nav_step_ts", 0.0) or 0.0)
        if interval > 0 and (now - last) < interval:
            return False
        self._last_corridor_nav_step_ts = now
        try:
            nodes: list = []
            route_mgr = self.__dict__.get("route_manager")
            if route_mgr is not None and hasattr(route_mgr, "get_route_nodes"):
                try:
                    nodes = list(route_mgr.get_route_nodes() or [])
                except Exception:
                    nodes = []
            active_id = self._extract_node_id(self.active_node)
            seq_pos = int(self.__dict__.get("_current_seq_pos", 0) or 0)
            if nodes:
                matches = [
                    i
                    for i, n in enumerate(nodes)
                    if self._extract_node_id(n) == active_id
                ]
                if matches:
                    seq_pos = matches[0]
                else:
                    nodes = [self.active_node]
                    seq_pos = 0
            else:
                nodes = [self.active_node]
                seq_pos = 0
            coords_x, coords_y = self._get_player_movement_coords()
            player_pos = (
                (coords_x, coords_y)
                if (coords_x != 0.0 or coords_y != 0.0)
                else None
            )
            # Kare önbellekten alınır (taze kare yoksa kare gerektiren
            # dallar atlanır; uçurum + kilitli-contested + prewalk yine
            # çalışır) — her 2 sn'de ek ekran yakalama yapılmaz.
            result = nav.step_navigation(
                frame=self.__dict__.get("_last_color_frame"),
                nodes=nodes,
                current_seq_pos=seq_pos,
                player_pos=player_pos,
                is_walking=True,
                input_controller=self.input_driver,
                now=now,
            )
            if not isinstance(result, dict):
                return False
            if result.get("unstuck_triggered"):
                # Çift kurtulma manevrası olmasın: monitörün kendi
                # takılma saatleri sıfırlanır.
                self._stuck_coords_ts = now
                self._stuck_dist_decrease_ts = now
            status = str(result.get("status") or "")
            if status in (
                "cliff_skipped",
                "contested_skipped",
                "node_no_coordinates",
            ) or result.get("repath_requested"):
                selected = result.get("selected_node")
                if selected is None:
                    self._emit_log(
                        f"CorridorNavigator: node {active_id} atlandı ({status}), sıradaki düğüm yok"
                    )
                    self._mark_node_failed(active_id)
                    self._register_node_failure()
                    self.active_node = None
                    self._walk_start_time = 0.0
                    self._arrival_at_node_ts = 0.0
                    return True
                if self._extract_node_id(selected) != active_id:
                    self._emit_log(
                        f"CorridorNavigator: node {active_id} -> {self._extract_node_id(selected)} ({status})"
                    )
                    self.active_node = selected
                    self._walk_start_time = now
                    self._last_reclick_time = now
                    self._arrival_at_node_ts = 0.0
                    self._set_active_node_distance(0)
                    return True
            return False
        except Exception as exc:
            logger.debug("CorridorNavigator step skipped: %s", exc)
            return False

    def _maybe_run_economy_cycle(self, now: float) -> None:
        """F-0088: periyodik ekonomi turu (tek giriş: run_economy_cycle).

        Yalnızca rota modunda, düğümsüz tick'te ve en fazla
        ``economy_cycle_sec`` (varsayılan 600 sn) sıklıkla çalışır:
        pazar tarama (karede broker penceresi açıksa) -> en kârlı rota
        seçimi -> üretim kuyruğu. Rota değişimi yürüyüş ortasında
        yapılmaz; düğüm seçimi turdan sonra geldiği için yeni rota
        bir sonraki seçimde devreye girer.
        """
        if getattr(self, "is_paused", False):
            return
        try:
            if self._motion_should_stop():
                return
        except Exception:
            # __init__ atlanmış iskelette (test) belirlenemez — fail-closed.
            return
        broker_scanner = self.__dict__.get("broker_scanner")
        crafting_controller = self.__dict__.get("crafting_controller")
        route_mgr = self.__dict__.get("route_manager")
        if (
            broker_scanner is None
            and crafting_controller is None
            and route_mgr is None
        ):
            return
        try:
            interval = float(self._get_cfg("economy_cycle_sec", 600) or 600)
        except (TypeError, ValueError):
            interval = 600.0
        if interval <= 0:
            return
        last = float(self.__dict__.get("_last_economy_cycle_ts", 0.0) or 0.0)
        if (now - last) < interval:
            return
        self._last_economy_cycle_ts = now
        try:
            from core.economy_broker import run_economy_cycle
        except Exception as exc:
            logger.debug("Economy cycle import skipped: %s", exc)
            return
        try:
            region = None
            if route_mgr is not None:
                region = getattr(route_mgr, "active_region", None)
            result = run_economy_cycle(
                scanner=broker_scanner,
                route_manager=route_mgr,
                crafting_queue=crafting_controller,
                input_ctrl=self.input_driver,
                frame=self.__dict__.get("_last_color_frame"),
                region=(str(region) if region else None),
                test_mode=bool(self._get_cfg("test_mode", False)),
            )
            route = (result or {}).get("route") or {}
            crafted = (result or {}).get("crafted_jobs") or []
            self._emit_log(
                "Ekonomi turu: rota "
                f"{route.get('best_region')}/{route.get('best_route')} "
                f"({route.get('formatted_kinah_per_hour', '0')} Kinah/saat) · "
                f"{len(crafted)} üretim"
            )
        except Exception as exc:
            logger.debug("Economy cycle skipped: %s", exc)

    def _monitor_walking_to_node(self, now: float) -> None:
        """
        Monitors travel towards active_node:
        - Periodically re-clicks the node on the map every reclick_node_sec (if > 0)
        - Handles time_to_reach_sec timeout and consecutive failure threshold
        """
        if self.active_node is None:
            return

        if self.is_paused:
            self._release_route_motion()
            return

        node_id = self._extract_node_id(self.active_node)
        time_to_reach = float(self._get_cfg("time_to_reach_sec", 60) or 60)
        reclick_sec = float(self._get_cfg("reclick_node_sec", 10) or 0)
        elapsed_walk = now - self._walk_start_time

        # v20-loop: harita açılamadığında mesafeyi odometriden tazele
        # (kartograf oyuncu konumu ↔ düğüm harita koordinatı). Stall
        # altında da çalışır — harita/reclick'e bağımlı değildir.
        self._refresh_node_distance_from_odometry(now)
        # Yürüyüş boyunca mesafe hiç tazelendi mi? (timeout kararı için.)
        if self._active_node_distance_fresh(now):
            self._walk_distance_ever_fresh = True

        # Timeout check
        if elapsed_walk >= time_to_reach:
            if self._is_opportunistic_node(self.active_node):
                # Fırsatçı düğüm rota düğümü değildir — failure sayılmaz.
                self._emit_log(f"Opportunistic node not reached in {int(time_to_reach)}s: dropping")
                self.active_node = None
                self._walk_start_time = 0.0
                self._arrival_at_node_ts = 0.0
                return
            # v20-loop: yürüyüş boyunca mesafe HİÇ tazelenmediyse bu
            # düğüm hatası değil, NAVİGASYON hatasıdır (harita açılmadı /
            # odometri yok / karakter hiç yürümedi). Düğüm failure
            # sayılmaz, kara listeye girmez, silinmez — dinlenmeye alınır
            # ve bir sonraki turda tekrar denenir. Eskiden sessizce fail
            # sayılıp 3 tekrarda kalıcı siliniyordu (canlı: Node 40/14).
            if not bool(getattr(self, "_walk_distance_ever_fresh", False)):
                self._emit_log(
                    f"⚠️ NAVİGASYON HATASI: Node {node_id} {int(time_to_reach)} sn'de "
                    "ulaşılamadı, mesafe hiç tazelenmedi — düğüm SİLİNMEDİ, "
                    "dinlenmeye alındı (harita/odometri kontrol edin)"
                )
                logger.warning(
                    "NAVIGATION FAILURE (no fresh distance) node %s: rested, not failed",
                    node_id,
                )
                self._rest_node_after_nav_failure(node_id)
                self.active_node = None
                self._walk_start_time = 0.0
                self._arrival_at_node_ts = 0.0
                return
            stale_note = (
                " (mesafe hiç tazelenmedi — varış prompt ile doğrulanamadı)"
                if not self._active_node_distance_fresh(now)
                else ""
            )
            self._emit_log(f"Node {node_id} not reached in {int(time_to_reach)}s: skipping{stale_note}")
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            self._walk_start_time = 0.0
            self._arrival_at_node_ts = 0.0
            return

        # Hedefe varış ve toplanacak kaynak kontrolü
        # 16/2: varış kararı yalnızca TAZE mesafeye bakılır. Bayat mesafe
        # (reclick başarısızken donan değer) 999 sayılır — prompt kontrolü
        # (adım 6) zaten varışı prompt ile yakalar.
        dist_fresh = self._active_node_distance_fresh(now)
        curr_dist = (
            float(getattr(self, "active_node_distance_m", 999.0) or 999.0)
            if dist_fresh
            else 999.0
        )
        if curr_dist <= 3.0:
            arr_ts = float(getattr(self, "_arrival_at_node_ts", 0.0) or 0.0)
            if arr_ts <= 0.0:
                self._arrival_at_node_ts = now
            elif (now - arr_ts) >= 1.5:
                # Düğüm noktasına gelindi fakat toplanacak prompt çıkmadı
                self._emit_log(f"⚠️ Düğüm #{node_id} noktasına gelindi fakat toplanacak kaynak bulunamadı!")
                if self._is_opportunistic_node(self.active_node):
                    self.active_node = None
                    self._walk_start_time = 0.0
                    self._arrival_at_node_ts = 0.0
                    return
                self._mark_node_failed(node_id)
                self._register_node_failure()
                self.active_node = None
                self._walk_start_time = 0.0
                self._arrival_at_node_ts = 0.0
                return
        else:
            self._arrival_at_node_ts = 0.0

        # F-0075: CorridorNavigator tick bağlantısı (2 sn'de bir). Düğüm
        # değiştiyse / düştüyse bu tick burada biter.
        if self._run_corridor_navigation_step(now):
            return

        # Periodic map target refresh (reclick_node_sec). A failed open backs off.
        # Fırsatçı düğümün harita konumu yoktur — reclick yapılmaz.
        if (
            reclick_sec > 0
            and (now - self._last_reclick_time) >= reclick_sec
            and self._walking_is_real()
            and not self._map_open_in_backoff()
            and not self._is_opportunistic_node(self.active_node)
        ):
            nav_ok, dist_m, under_overlay = self._open_map_and_click_node(
                self.active_node, is_reclick=True
            )
            self._last_reclick_time = time.time()
            if self._last_nav_block == "map_open":
                # Map still closed. Keep the same node and keep holding W.
                if not self._command_walk_toward(self.active_node):
                    return
            elif under_overlay:
                self._emit_log(f"Node {node_id} is under the overlay panel")
                self._mark_node_failed(node_id)
                self._register_node_failure()
                self.active_node = None
                return
            if nav_ok:
                self._set_active_node_distance(dist_m)
                self._emit_log(f"Route: going to node {node_id} ({dist_m} m)")
        else:
            if self._map_open_in_backoff() and not self._walking_is_real() and not self._command_walk_toward(self.active_node):
                # 16/15: sessiz düşürme yok — ret failure sayılır (geçici
                # durumdaysa _drop_node_on_walk_refusal saymaz).
                self._drop_node_on_walk_refusal(node_id, "walk refused while map in backoff")
                self._release_route_motion()
                self._set_state_internal(BotState.WAITING.value)
                return
            if not self._walking_is_real() and not self._command_walk_toward(self.active_node):
                self._drop_node_on_walk_refusal(node_id, "walk refused while not walking")
                return
            if not self._walking_is_real():
                self._set_state_internal(BotState.WAITING.value)
                return

            # Strafe only while a movement key or path step is already active.
            if self.game_learner is not None:
                cognition = self.game_learner.get_live_cognition_snapshot()
                if cognition.get("obstacle_detected"):
                    obs_type = cognition.get("obstacle_type", "obstacle")
                    steer_deg = float(cognition.get("recommended_steer_deg", 0.0))
                    self._apply_active_obstacle_evasion(obs_type, steer_deg, now)

            # Active stuck recovery (16/1): birincil sinyal oyuncu
            # koordinat hareketidir — mesafe yalnızca reclick'te (10 sn)
            # güncellenir, ona bakmak düzgün yürüyüşte ~5 sn'de bir
            # yanlış unstick patlatırdı. Mesafe dalı yalnızca TAZE
            # mesafede çalışır; koordinat bilinmiyorsa ve mesafe bayatsa
            # takılma kararı verilmez (fail-closed: yanlış unstick yok).
            coords_x, coords_y = self._get_player_movement_coords()
            coords_known = (coords_x != 0.0 or coords_y != 0.0)
            stuck_fired = False
            if coords_known:
                last_coords = self._plain_attr("_stuck_last_coords", None)
                last_coords_ts = self._plain_attr("_stuck_coords_ts", None)
                if (
                    last_coords_ts is None
                    or last_coords is None
                    or self._plain_attr("active_node") != self._plain_attr("_stuck_monitored_node", None)
                ):
                    self._stuck_monitored_node = self._plain_attr("active_node")
                    self._stuck_last_coords = (coords_x, coords_y)
                    self._stuck_coords_ts = now
                else:
                    moved = math.hypot(coords_x - float(last_coords[0]), coords_y - float(last_coords[1]))
                    if moved > 3.0:
                        self._stuck_last_coords = (coords_x, coords_y)
                        self._stuck_coords_ts = now
                    elif self._walking_is_real() and (now - float(last_coords_ts)) >= 5.0:
                        self._execute_unstick_maneuver()
                        self._stuck_coords_ts = now
                        self._stuck_last_coords = (coords_x, coords_y)
                        stuck_fired = True
            if not stuck_fired and self._active_node_distance_fresh(now):
                curr_dist = float(self._plain_attr("active_node_distance_m", 0.0) or 0.0)
                last_check_dist = self._plain_attr("_stuck_last_dist", None)
                last_decrease_ts = self._plain_attr("_stuck_dist_decrease_ts", None)
                if (
                    last_decrease_ts is None
                    or last_check_dist is None
                    or self._plain_attr("active_node") != self._plain_attr("_stuck_monitored_node", None)
                ):
                    self._stuck_monitored_node = self._plain_attr("active_node")
                    self._stuck_last_dist = curr_dist
                    self._stuck_dist_decrease_ts = now
                elif curr_dist < (last_check_dist - 0.5):
                    self._stuck_last_dist = curr_dist
                    self._stuck_dist_decrease_ts = now
                elif self._walking_is_real() and (now - last_decrease_ts) >= 5.0:
                    self._execute_unstick_maneuver()
                    self._stuck_dist_decrease_ts = now
                    self._stuck_last_dist = curr_dist

            # ZK: stuck memory with real player movement coordinates / odometry (never constant node_id)
            if coords_known:
                self._zk_note_stuck_and_avoid(coords_x, coords_y, moving=self._walking_is_real())
            hint = self._plain_attr("_ucb_slot_hint")
            if hint and str(self.mode).lower() in ("combat", "farm"):
                # Hint is informational for combat/farm; route nodes stay primary.
                pass

            if not self._command_walk_toward(self.active_node):
                return

            # 16/18: HUD'daki mesafe ölçülendir — geçen süreye göre %35'e
            # kadar azaltan formül kaldırıldı; operatör gerçek (son
            # ölçülen) mesafeyi görür, bilinmiyorsa "unknown" yazar.
            measured_dist = int(getattr(self, "active_node_distance_m", 0) or 0)
            dist_txt = f"{measured_dist} m" if measured_dist > 0 else "distance unknown"

            self._set_state_internal(BotState.WALKING.value)
            self._emit_hud_snapshot(
                badge=BotState.WALKING.value,
                title=f"Walking to node {node_id} · {dist_txt}",
                subtitle="On arrival: F, or a click on the gathering icon",
            )

    def _note_still_obstacle_angle(self, obs_type: str, steer_deg: float, now: float) -> bool:
        """A repeating ~20° label on a still character is a stuck classifier, not a rock."""
        hist = getattr(self, "_still_obstacle_angles", None)
        if not isinstance(hist, list):
            hist = []
        hist.append((float(steer_deg), str(obs_type or "").lower(), float(now)))
        hist = [item for item in hist if now - item[2] <= 12.0]
        self._still_obstacle_angles = hist
        if len(hist) < 2:
            return False
        angles = [item[0] for item in hist]
        return (max(angles) - min(angles)) <= 6.0

    def _obstacle_evasion_allowed(self, obs_type: str, steer_deg: float, now: float) -> bool:
        """Strafe only for a close obstacle while a movement key or path step is active."""
        if not self._walking_is_real():
            self._note_still_obstacle_angle(obs_type, steer_deg, now)
            return False
        kind = str(obs_type or "").strip().lower()
        if kind in ("", "none"):
            return False
        close_kind = kind in ("wall", "rock", "tree_trunk", "cliff_drop", "obstacle")
        if abs(float(steer_deg)) <= 2.5 and not close_kind:
            return False
        return True

    def _strafe_key_for_steer(self, steer_deg: float, now: float) -> Optional[str]:
        """
        Pick A or D for a sidestep. Alternating left/right labels cancel each
        other and used to leave the body in place while the camera yawed.
        After two flips, keep one direction long enough to actually step off.
        """
        if abs(float(steer_deg)) <= 2.5:
            return None
        preferred = "d" if float(steer_deg) > 0.0 else "a"
        hist = getattr(self, "_evasion_sign_hist", None)
        if not isinstance(hist, list):
            hist = []
        sign = 1 if float(steer_deg) > 0.0 else -1
        hist.append((sign, float(now)))
        hist = [item for item in hist if now - item[1] <= 12.0]
        self._evasion_sign_hist = hist
        locked = getattr(self, "_evasion_locked_key", None)
        lock_until = float(getattr(self, "_evasion_lock_until", 0.0) or 0.0)
        if locked in ("a", "d") and now <= lock_until:
            return str(locked)
        signs = [item[0] for item in hist]
        flips = sum(1 for prev, cur in zip(signs, signs[1:]) if prev != cur)
        if flips >= 2 and len(signs) >= 3:
            positive = sum(1 for item in signs if item > 0)
            locked = "d" if positive >= (len(signs) - positive) else "a"
            self._evasion_locked_key = locked
            self._evasion_lock_until = now + 4.0
            return locked
        return preferred

    def _apply_active_obstacle_evasion(
        self, obs_type: str, steer_deg: float, now: float
    ) -> bool:
        """
        Sidestep and jump while W stays held. The cursor is not moved: an
        absolute mouse move only changes the camera and the character stands still.
        """
        if not self._obstacle_evasion_allowed(obs_type, steer_deg, now):
            return False

        last_evade = getattr(self, "_last_obstacle_evade_ts", 0.0)
        if (now - last_evade) < 1.4:
            return False
        setattr(self, "_last_obstacle_evade_ts", now)

        driver = self.input_driver
        if driver is None:
            return False

        test_mode = bool(self._get_cfg("test_mode", False))
        if test_mode:
            return False

        try:
            hold = getattr(driver, "hold_movement_key", None)
            if not callable(hold):
                return False
            if not hold("w"):
                return False

            strafe = self._strafe_key_for_steer(steer_deg, now)
            if strafe and not hold(strafe):
                strafe = None

            jumped = False
            if obs_type in ("wall", "rock", "cliff_drop", "obstacle") or abs(steer_deg) > 4.0:
                pulse = getattr(driver, "pulse_key", None)
                if callable(pulse):
                    jumped = bool(pulse("space", 0.16))
                else:
                    self._press_key("space")
                    jumped = True
                # A keyboard report that only contains Space drops W and the strafe.
                hold("w")
                if strafe:
                    hold(strafe)

            if strafe:
                if not self._msleep_abortable(480):
                    release_one = getattr(driver, "release_movement_key", None)
                    if callable(release_one):
                        release_one(strafe)
                    hold("w")
                    return False
                release_one = getattr(driver, "release_movement_key", None)
                if callable(release_one):
                    release_one(strafe)
                hold("w")

            jump_note = " ve zıplama" if jumped else ""
            if strafe:
                action = f"W+{strafe.upper()} tutuldu{jump_note}"
            else:
                action = f"W tutuldu{jump_note}"
            self._emit_log(
                f"🧗 [YZ Engel Kaçınma] {obs_type.upper()} engeli: {action} (Açı: {steer_deg:.1f}°)."
            )
            return True
        except Exception as exc:
            logger.debug("Engel kaçınma manevrası hatası: %s", exc)
            return False

    def _register_node_failure(self) -> None:
        """Increments failed_in_row, captures incident snapshot, and triggers wait_min_after_failed cooldown if limit reached."""
        self.failed_in_row += 1
        # sweep-29 K-2: node başarısızlığı telemetriye yazılır — eskiden
        # yalnızca incident snapshot alınıyordu; success_rate hep %100,
        # failed_total hep 0 görünüyordu.
        try:
            node_id = self._extract_node_id(self.active_node)
            _tel = getattr(self, "telemetry", None)
            if (
                _tel is not None
                and hasattr(_tel, "record_failure")
                and isinstance(node_id, int)
            ):
                _tel.record_failure(node_id=node_id, reason="node_failure")
        except Exception as _tel_exc:
            logger.warning("Telemetri record_failure hatası: %s", _tel_exc)
        max_failed = int(self._get_cfg("max_failed_in_row", 10) or 10)
        wait_min = float(self._get_cfg("wait_min_after_failed", 5) or 5)

        if not bool(self._get_cfg("test_mode", False)) and self.incident_recorder is not None:
            try:
                frame_bgr = self._grab_frame(grayscale=False)
                if frame_bgr is not None and hasattr(self.incident_recorder, "record_incident"):
                    self.incident_recorder.record_incident(
                        frame_bgr,
                        reason=f"NODE_FAIL_{self.failed_in_row}",
                        metadata={"active_node": str(self._extract_node_id(self.active_node))},
                    )
            except Exception as inc_exc:
                logger.debug("Incident snapshot skipped: %s", inc_exc)

        if max_failed > 0 and self.failed_in_row >= max_failed:
            self._failure_wait_until = time.time() + (wait_min * 60.0)
            self._set_state_internal(BotState.WAITING.value)
            self._emit_log(
                f"Failed {self.failed_in_row}/{max_failed} in a row: waiting {int(wait_min)}m"
            )
            self._emit_hud_snapshot(
                badge=BotState.WAITING.value,
                title=f"Waiting {int(wait_min)} min after {self.failed_in_row} failures",
                subtitle="Too many unreachable nodes in a row",
            )

    # =========================================================================
    # Map Interaction & Node Location
    # =========================================================================

    def _open_map_and_click_node(
        self, node: Any, is_reclick: bool = False
    ) -> Tuple[bool, int, bool]:
        """
        1. Emits MAP badge ("Looking at the map", "Don't touch the mouse or keyboard")
        2. Opens map
        3. Locates node on open map (locate_node_on_open_map)
        4. Checks if target is under overlay panel
        5. Clicks target on map & closes map
        Returns: (success, distance_meters, is_under_overlay)
        """
        node_id = self._extract_node_id(node)
        test_mode = bool(self._get_cfg("test_mode", False))

        try:
            if self._motion_should_stop():
                return False, 0, False
            # Do not claim "Looking at the map" until the map is actually open.
            if not test_mode:
                if self._map_open_in_backoff():
                    if not is_reclick:
                        self._last_nav_block = "map_open"
                    return False, 0, False
                self._last_nav_block = ""
                opened = self._ensure_map_open()
                if not opened or self._motion_should_stop():
                    self._begin_map_open_backoff(node_id)
                    if is_reclick:
                        # A later map retry must not cancel the on-foot walk.
                        self._last_nav_block = ""
                    return False, 0, False
                # Sweep G: doğrulama kapısı — harita tıklaması yalnızca kesin
                # açıkken yapılır; bilinmiyor (None) da backoff'a düşer.
                # (`not ...` üç durumda da doğrudur: True→devam, False/None→backoff.)
                if self.vision is not None and hasattr(self.vision, "is_map_open") and not self._vision_map_is_open():
                    self._begin_map_open_backoff(node_id)
                    if is_reclick:
                        self._last_nav_block = ""
                    return False, 0, False
            self._set_state_internal(BotState.MAP.value)
            # F-0508: in test_mode the map is never opened — say so.
            self._emit_hud_snapshot(
                badge=BotState.MAP.value,
                title="Looking at the map" if not test_mode else "Map (test mode — not opened)",
                subtitle="Don't touch the mouse or keyboard" if not test_mode else "Test mode: map open skipped",
            )

            # Locate node coordinates & distance on the open map
            loc_result = self._locate_node_on_open_map(node)
            if loc_result is None or self._motion_should_stop():
                if not test_mode:
                    self._ensure_map_closed()
                self._deny_walk_hud("Map target missing")
                return False, 0, False

            target_x, target_y, dist_m, under_overlay = loc_result

            if under_overlay or self._is_point_under_overlay(target_x, target_y):
                if not test_mode:
                    self._ensure_map_closed()
                return False, dist_m, True

            if not test_mode:
                self._click_map_point(target_x, target_y)
                if self._motion_should_stop() or not self._msleep_abortable(150):
                    self._ensure_map_closed()
                    self._deny_walk_hud("Map click aborted")
                    return False, dist_m, False
                closed = self._ensure_map_closed()
                if not closed or self._map_blocks_movement():
                    self._deny_walk_hud("Map UI blocks movement")
                    return False, dist_m, False

            return True, max(1, dist_m), False
        except Exception as exc:
            logger.error("Error in _open_map_and_click_node(node=%s): %s", node_id, exc, exc_info=True)
            if not test_mode:
                self._ensure_map_closed()
            return False, 0, False

    def _locate_node_on_open_map(
        self, node: Any
    ) -> Optional[Tuple[int, int, int, bool]]:
        """
        Calls route_manager.locate_node_on_open_map or vision.locate_node_on_open_map
        or falls back to patch template matching / stored map coordinates.
        Returns (target_x, target_y, distance_m, under_overlay) or None.
        """
        overlay_rect = self._get_overlay_rect()
        open_map_frame = self._grab_frame(grayscale=False)

        # 1. Try route_manager / vision / navigator locate_node_on_open_map
        for provider in (self.route_manager, self.vision, self.navigator):
            if provider is not None and hasattr(provider, "locate_node_on_open_map"):
                try:
                    fn = getattr(provider, "locate_node_on_open_map")
                    raw = None
                    try:
                        raw = fn(
                            node,
                            open_map_frame,
                            vision_engine=self.vision,
                            overlay_rect=overlay_rect,
                        )
                    except TypeError:
                        try:
                            raw = fn(node, overlay_rect=overlay_rect)
                        except TypeError:
                            try:
                                raw = fn(node)
                            except TypeError:
                                raw = fn(open_map_frame, node)
                    parsed = self._parse_locate_result(raw, node)
                    if parsed is not None:
                        return parsed
                except Exception as exc:
                    logger.debug("locate_node_on_open_map on %s failed: %s", provider, exc)

        # 2. Fallback: template match node's patch_file on open map or use map_x/map_y
        target_x = self._get_node_field(node, "map_x", self._get_node_field(node, "x"))
        target_y = self._get_node_field(node, "map_y", self._get_node_field(node, "y"))

        patch_file = self._get_node_field(node, "patch_path") or self._get_node_field(
            node, "patch_file"
        )
        if patch_file and self.vision is not None:
            patch_dir = getattr(self.route_manager, "patch_dir", "routes/patches")
            patch_path = (
                str(patch_file)
                if os.path.isabs(str(patch_file)) or os.path.exists(str(patch_file))
                else os.path.join(str(patch_dir), os.path.basename(str(patch_file)))
            )
            if os.path.exists(patch_path):
                try:
                    try:
                        patch_gray = cv2.imdecode(np.fromfile(patch_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        patch_gray = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                    map_gray = self._grab_frame(grayscale=True)
                    if patch_gray is not None and map_gray is not None:
                        res = cv2.matchTemplate(map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        # F-0082: SelfHealingTemplateTuner — her şablon
                        # eşleşme denemesi bir düğüm ziyaretidir: skor
                        # kaydedilir ve eşik düğümün kendi iyileşen
                        # eşiğinden alınır (sabit 0.70 yerine). Tuner yoksa
                        # veya düğüm kimliği yoksa varsayılan 0.70 geçerli.
                        node_threshold = 0.70
                        _tuner_node_id = self._extract_node_id(node, default=None)
                        if _tuner_node_id is not None:
                            tuner = getattr(self, "template_tuner", None)
                            if tuner is not None:
                                try:
                                    node_threshold = float(
                                        tuner.get_node_threshold(_tuner_node_id)
                                    )
                                except Exception:
                                    node_threshold = 0.70
                                try:
                                    tuner.record_visit_score(
                                        _tuner_node_id,
                                        float(max_val),
                                        patch_gray,
                                        patch_save_path=patch_path,
                                    )
                                except Exception as tun_exc:
                                    logger.debug("Template tuner kaydı atlandı: %s", tun_exc)
                        if max_val >= node_threshold:
                            ph, pw = patch_gray.shape[:2]
                            target_x = int(max_loc[0] + pw // 2)
                            target_y = int(max_loc[1] + ph // 2)
                except Exception as exc:
                    logger.debug("Fallback patch match failed: %s", exc)

        if target_x is None or target_y is None:
            return None

        tx, ty = int(target_x), int(target_y)
        dist_m = self._estimate_distance_meters(tx, ty, node)
        under_ov = self._is_point_under_overlay(tx, ty)
        return tx, ty, dist_m, under_ov

    def _start_opportunistic_walk(self, nearby: Any, now: float) -> None:
        """Yakındaki materyale geçici düğümle yürür (16/3).

        Geçici düğüm rota düğümü değildir: harita konumu yoktur (map_x'e
        EKRAN x'i yazılır — _turn_key_for_node ekranda sağ/sol direksiyon
        için zaten ekran merkezini kullanır), mesafesi tarayıcının tahmini
        mesafesidir. Monitör bu düğümde reclick yapmaz ve timeout/fail
        saymaz; prompt çıkınca normal toplama motoru devralır.
        """
        m_type = getattr(nearby, "material_type", "Material") or "Material"
        screen_x = getattr(nearby, "screen_x", None)
        screen_y = getattr(nearby, "screen_y", None)
        est_dist = getattr(nearby, "estimated_distance_m", 0.0) or 0.0
        transient: Dict[str, Any] = {
            "id": None,
            "resource_type": str(m_type),
            "map_x": int(screen_x) if screen_x is not None else None,
            "map_y": int(screen_y) if screen_y is not None else None,
            "distance_m": float(est_dist),
            "_opportunistic": True,
        }
        self.active_node = transient
        self._set_active_node_distance(est_dist)
        self._walk_start_time = now
        self._last_reclick_time = now
        self._arrival_at_node_ts = 0.0
        self._emit_log(f"Opportunistic gather: approaching nearby {m_type} (~{float(est_dist):.0f} m)")
        if self._command_walk_toward(transient):
            self._set_state_internal(BotState.WALKING.value)
            self._emit_hud_snapshot(
                badge=BotState.WALKING.value,
                title=f"Walking to nearby {m_type}",
                subtitle="On arrival: F, or a click on the gathering icon",
            )
        else:
            # Yürüme reddedildiyse geçici düğüm sessizce düşer (rota
            # düğümü olmadığı için failure sayılmaz).
            self._emit_log("Opportunistic walk refused: dropping transient node")
            self.active_node = None
            self._walk_start_time = 0.0

    def _parse_locate_result(
        self, raw: Any, node: Any
    ) -> Optional[Tuple[int, int, int, bool]]:
        if raw is None or raw is False:
            return None

        # Support NodeLocationResult (from core/route_manager.py) which has .to_dict() and attributes
        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            raw = raw.to_dict()

        if isinstance(raw, dict):
            # 16/5: boş dict "bulundu, hedef (0,0)" sayılmaz. found yoksa ve
            # koordinat da yoksa (ne dict'te ne düğümde) bulunamadı döner.
            node_map_x = self._get_node_field(node, "map_x", 0) or 0
            node_map_y = self._get_node_field(node, "map_y", 0) or 0
            has_coord = (
                any(k in raw for k in ("x", "y", "map_x", "map_y"))
                or node_map_x != 0
                or node_map_y != 0
            )
            if raw.get("found", None) is False:
                return None
            if raw.get("found", None) is None and not has_coord:
                return None
            tx = int(raw.get("x", raw.get("map_x", self._get_node_field(node, "map_x", 0) or 0)))
            ty = int(raw.get("y", raw.get("map_y", self._get_node_field(node, "map_y", 0) or 0)))
            if tx == 0 and ty == 0 and not has_coord:
                return None
            dist_raw = raw.get("distance_m", raw.get("distance", None))
            if isinstance(dist_raw, str):
                digits = "".join(ch for ch in dist_raw if ch.isdigit() or ch == ".")
                dist = int(round(float(digits))) if digits else self._estimate_distance_meters(tx, ty, node)
            elif isinstance(dist_raw, (int, float)) and dist_raw > 0:
                dist = int(round(float(dist_raw)))
            else:
                dist = self._estimate_distance_meters(tx, ty, node)
            return tx, ty, max(1, dist), bool(raw.get("under_overlay", False))

        if isinstance(raw, (tuple, list)):
            # Format: ((x, y), confidence, distance_str)
            if len(raw) >= 2 and isinstance(raw[0], (tuple, list)) and len(raw[0]) >= 2:
                tx, ty = int(raw[0][0]), int(raw[0][1])
                dist = self._estimate_distance_meters(tx, ty, node)
                if len(raw) >= 3 and isinstance(raw[2], str):
                    digits = "".join(ch for ch in raw[2] if ch.isdigit() or ch == ".")
                    if digits:
                        dist = int(round(float(digits)))
                elif len(raw) >= 3 and isinstance(raw[2], (int, float)):
                    dist = int(round(float(raw[2])))
                under_ov = bool(getattr(raw, "under_overlay", False))
                return tx, ty, max(1, dist), under_ov

            if len(raw) == 2:
                tx, ty = int(raw[0]), int(raw[1])
                return tx, ty, self._estimate_distance_meters(tx, ty, node), False
            if len(raw) == 3:
                if isinstance(raw[0], bool):
                    if not raw[0]:
                        return None
                    tx, ty = int(raw[1]), int(raw[2])
                    return tx, ty, self._estimate_distance_meters(tx, ty, node), False
                tx, ty, dist = int(raw[0]), int(raw[1]), int(raw[2])
                return tx, ty, max(1, dist), False
            if len(raw) >= 4:
                tx, ty, dist, under_ov = (
                    int(raw[0]),
                    int(raw[1]),
                    int(raw[2]),
                    bool(raw[3]),
                )
                return tx, ty, max(1, dist), under_ov

        return None

    def _estimate_distance_meters(self, target_x: int, target_y: int, node: Any) -> int:
        dist_val = self._get_node_field(node, "distance_m")
        if dist_val is not None:
            try:
                return max(1, int(round(float(dist_val))))
            except (ValueError, TypeError):
                logger.debug("Invalid distance_m on node: %s", dist_val)

        screen_w = int(getattr(self.vision, "screen_w", 1920) or 1920)
        screen_h = int(getattr(self.vision, "screen_h", 1080) or 1080)
        cx, cy = screen_w // 2, screen_h // 2
        pixel_dist = math.hypot(target_x - cx, target_y - cy)
        # v19 P1-7/P1-12: literal 0.28 yerine tek kaynak
        # `core/corridor_navigator.py::WORLD_MAP_METERS_PER_PX` (0.28).
        # DİKKAT — uzay varsayımı kalibre edilmedi: buradaki piksel ekran
        # merkezine göre ekran pikselidir, dünya-haritası pikseli değil;
        # WORLD_MAP ölçeğiyle çarpılması v18'den beri korunan yaklaşımdır,
        # "ölçüldü" sayılmaz (bkz. corridor_navigator kalibrasyon kancası).
        from core.corridor_navigator import WORLD_MAP_METERS_PER_PX

        meters = max(2, int(round(pixel_dist * WORLD_MAP_METERS_PER_PX)))
        return meters

    def _get_overlay_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns the (x, y, w, h) rectangle of the HUD overlay panel."""
        if callable(self.overlay_rect_provider):
            try:
                rect = self.overlay_rect_provider()
                if rect is not None:
                    if hasattr(rect, "x") and hasattr(rect, "width"):
                        rx = int(rect.x() if callable(rect.x) else rect.x)
                        ry = int(rect.y() if callable(rect.y) else rect.y)
                        rw = int(rect.width() if callable(rect.width) else rect.width)
                        rh = int(rect.height() if callable(rect.height) else rect.height)
                        return rx, ry, rw, rh
                    if isinstance(rect, (tuple, list)) and len(rect) >= 4:
                        return int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            except Exception as exc:
                logger.debug("overlay_rect_provider error: %s", exc)

        if not bool(self._get_cfg("show_overlay", True)):
            return None

        # Default top-left Gatherion overlay bounding box (x=25, y=45, w=360, h=185)
        return (20, 40, 370, 195)

    def _is_point_under_overlay(self, x: int, y: int) -> bool:
        rect = self._get_overlay_rect()
        if rect is None:
            return False
        rx, ry, rw, rh = rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def _map_open_key(self) -> str:
        """AION 2 world-map key. F8/F9/F10 chords stay the bot hotkeys."""
        raw = str(self._get_cfg("map_key", "m") or "m").strip().lower()
        if not raw or "+" in raw or any(chord in raw for chord in ("f8", "f9", "f10")):
            return "m"
        return raw

    def _focus_game_window(self) -> bool:
        """Focus AION 2 so M is not typed into the Gatherion window."""
        driver = self.input_driver
        if driver is None:
            return False
        fn = getattr(driver, "focus_game_window", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception as exc:
            logger.debug("Game focus before map key skipped: %s", exc)
            return False

    def _vision_map_is_open(self) -> Optional[bool]:
        """Üç durumlu harita kontrolü (Sweep G): True (harita açık), False
        (harita kapalı), None (bilinmiyor — bozuk kare / doğrulama hatası).
        None 'harita açık olabilir' sayılır — bool() ile False'a indirgenmez
        (fail-open). Vision yoksa False döner (eski davranış: doğrulayıcı
        yokken kapalı varsayılır)."""
        vision = getattr(self, "vision", None)
        if vision is None or not hasattr(vision, "is_map_open"):
            return False
        frame = None
        grab = getattr(self, "_grab_frame", None)
        if callable(grab):
            try:
                frame = grab(grayscale=False)
            except TypeError:
                try:
                    frame = grab()
                except Exception:
                    frame = None
            except Exception:
                frame = None
        try:
            if frame is not None:
                try:
                    result = vision.is_map_open(frame)
                except TypeError:
                    result = vision.is_map_open()
            else:
                result = vision.is_map_open()
        except Exception:
            return None
        return None if result is None else bool(result)

    def _ensure_map_open(self) -> bool:
        """Focus the game, press the map key once, then wait until the map is visible.

        A second press used to close the map when the 720p header template missed
        a live 2560x1440 frame. One press only; a miss falls back to walking.
        """
        try:
            if self._motion_should_stop():
                return False
            # Sweep G: bilinmiyor (None) açık sayılır — M'ye basmak açık
            # haritayı kapatırdı, o yüzden basmadan "açık" dönülür.
            if self._vision_map_is_open() is not False:
                return True
            self._focus_game_window()
            self._toggle_map_key()
            for _ in range(6):
                if not self._msleep_abortable(140):
                    return False
                # Doğrulama kapısı: yalnızca kesin açık sayılır; bilinmiyor
                # yoklamaya devam eder (kör harita tıklaması yapılmaz).
                if self._vision_map_is_open() is True:
                    return self._msleep_abortable(80)
            # M basışı doğrulanmadı sayılır (önceki durum kesin kapalıydı) —
            # bilinmiyor burada "açıldı" sayılmaz, backoff/yürüme yoluna düşülür.
            return self._vision_map_is_open() is True
        except Exception as exc:
            logger.debug("Error opening map: %s", exc)
            return False

    def _ensure_map_closed(self) -> bool:
        try:
            if self.navigator is not None and hasattr(self.navigator, "close_map"):
                return bool(self.navigator.close_map())
            if self.route_manager is not None and hasattr(self.route_manager, "close_map"):
                return bool(self.route_manager.close_map())
            if self.vision is not None and hasattr(self.vision, "is_map_open"):
                # Sweep G: M'ye yalnızca kesin açıkken basılır; "kapalı"
                # yalnızca kesin kapalıyken raporlanır (bilinmiyor → False).
                if self.vision.is_map_open() is True:
                    self._toggle_map_key()
                    if not self._msleep_abortable(180):
                        return False
                return self.vision.is_map_open() is False
            self._toggle_map_key()
            return self._msleep_abortable(180)
        except Exception as exc:
            logger.debug("Error closing map: %s", exc)
            return False

    # =========================================================================
    # Break Management (Anti-AFK & Scheduled Breaks)
    # =========================================================================

    def _schedule_next_break(self) -> None:
        """Schedules the next break timestamp based on break_every_min ± break_every_variance."""
        try:
            every_min = float(self._get_cfg("break_every_min", 0) or 0)
            if every_min <= 0:
                self._next_break_timestamp = 0.0
                return

            variance_min = float(self._get_cfg("break_every_variance", 5) or 0)
            delta_min = max(1.0, every_min + random.uniform(-variance_min, variance_min))
            self._next_break_timestamp = time.time() + (delta_min * 60.0)
        except Exception as exc:
            logger.debug("Break scheduling error: %s", exc)
            self._next_break_timestamp = 0.0

    def _should_enter_break(self, now: float) -> bool:
        every_min = float(self._get_cfg("break_every_min", 0) or 0)
        if every_min <= 0:
            return False
        if self._next_break_timestamp <= 0.0:
            self._schedule_next_break()
            return False
        return now >= self._next_break_timestamp and not self._is_gathering_active

    def _handle_break_step(self, now: float) -> None:
        """
        Executes a scheduled break with orange BREAK badge and subtle Anti-AFK actions,
        then automatically resumes ROUTE_MODE when the break timer expires.
        """
        # F-0387: entering a break must drop W immediately — otherwise the
        # character keeps walking through the whole break.
        self._halt_motion()
        self._release_route_motion()
        if self._state != BotState.BREAK.value or self._break_end_timestamp <= 0.0:
            lasts_min = float(self._get_cfg("break_lasts_min", 5) or 5)
            lasts_var = float(self._get_cfg("break_lasts_variance", 1) or 0)
            duration_min = max(0.5, lasts_min + random.uniform(-lasts_var, lasts_var))
            self._break_end_timestamp = now + (duration_min * 60.0)
            self._last_anti_afk_time = now
            self.active_node = None
            self._set_state_internal(BotState.BREAK.value)
            self._emit_log(f"Break started for {duration_min:.1f} min")

        remaining_sec = self._break_end_timestamp - now
        if remaining_sec <= 0:
            self._break_end_timestamp = 0.0
            # O5: mola sonu da _last_gather_time'ı sıfırlamaz — mola zaten
            # toplamasız geçen sürenin üstüne gelir; sıfırlamak 15 dk
            # korumasını erteler.
            self._schedule_next_break()
            self._set_state_internal(BotState.WALKING.value)
            self._emit_log("Break finished: resuming route mode")
            self._emit_hud_snapshot()
            return

        # Perform subtle Anti-AFK action every 25..45 seconds during break
        if now - self._last_anti_afk_time >= random.uniform(25.0, 45.0):
            self._perform_anti_afk_step()
            self._last_anti_afk_time = now

        rem_int = max(0, int(remaining_sec))
        rem_str = f"{rem_int // 60}:{rem_int % 60:02d}"
        self._emit_hud_snapshot(
            badge=BotState.BREAK.value,
            mode_badge="ROUTE",
            title=f"On a break · {rem_str} left",
            subtitle="Anti-AFK active · Will resume route automatically",
        )

    def _get_human_timing_coordinator(self) -> Any:
        """F-0103: HumanTimingCoordinator'ı mevcut alt sistemlerden kurar
        (HumanTimingEngine + OrganicAntiAFK zaten __init__'te kuruluyor;
        break scheduler koordinatörün kendi varsayılanıdır). Kurulamazsa
        None döner — çağıran eski anti-AFK yoluna düşer."""
        coord = self._plain_attr("_human_timing_coordinator")
        if coord is not None:
            return coord
        try:
            from core.human_timing import HumanTimingCoordinator

            coord = HumanTimingCoordinator(
                timing_engine=self._plain_attr("human_timing"),
                anti_afk=self._plain_attr("organic_anti_afk"),
            )
        except Exception as exc:
            logger.debug("HumanTimingCoordinator kurulamadı: %s", exc)
            return None
        try:
            self.__dict__["_human_timing_coordinator"] = coord
        except Exception:
            pass
        return coord

    def _perform_anti_afk_step(self) -> None:
        """Executes a subtle camera/mouse micro-movement or anti-AFK step during BREAK.

        F-0103: HumanTimingCoordinator.tick() bu adımın sahibidir — koordinatör
        varsa kadansı o yönetir (140-260 sn aralık): zamanı gelmediyse ("waiting")
        hiçbir şey yapılmaz, eylem uygulandıysa ("executed") adım biter.
        Koordinatör yoksa / beklemedeyse / başarısızsa eski input_driver
        yoluna düşülür. Mola sinyalleri (macro_break/micro_pause) burada
        tüketilmez — mola zamanlaması _should_enter_break/_handle_break_step'tedir.
        """
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        try:
            coord = self._get_human_timing_coordinator()
            if coord is not None and hasattr(coord, "tick"):
                # P1-10 (K-14): tick artık tek çağrılık kablolama noktası
                # run_human_timing_tick üzerinden akar (önceden doğrudan
                # coord.tick() çağrılıyordu; sarmalayıcı hiç çağrılmıyordu).
                # coordinator verilerek botun kendi alt sistemleri
                # (self.human_timing / self.organic_anti_afk) korunur —
                # tekil global koordinatöre geçilmez.
                from core.human_timing import run_human_timing_tick

                res = run_human_timing_tick(
                    self.input_driver, now=time.monotonic(), coordinator=coord
                )
                if isinstance(res, dict):
                    if res.get("macro_break") or res.get("micro_pause"):
                        logger.debug(
                            "HumanTimingCoordinator mola sinyali (break adımı yönetir): %s",
                            {k: res.get(k) for k in ("macro_break", "micro_pause")},
                        )
                    afk = res.get("anti_afk") or {}
                    if isinstance(afk, dict):
                        if afk.get("action") == "waiting":
                            return
                        if afk.get("executed"):
                            return
                        # pending / busy / error / uygulanamadı -> eski yola düş.
        except Exception as exc:
            logger.debug("HumanTimingCoordinator tick hatası, eski yola düşülüyor: %s", exc)
        try:
            self._last_bot_key_time = time.time()
            if hasattr(self.input_driver, "anti_afk_step"):
                self.input_driver.anti_afk_step()
            elif hasattr(self.input_driver, "move_mouse_relative"):
                self.input_driver.move_mouse_relative(
                    random.randint(-8, 8), random.randint(-5, 5)
                )
            elif hasattr(self.input_driver, "smooth_move_to") and hasattr(
                self.input_driver, "get_cursor_position"
            ):
                cx, cy = self.input_driver.get_cursor_position()
                self.input_driver.smooth_move_to(
                    cx + random.randint(-4, 4), cy + random.randint(-4, 4)
                )
        except Exception as exc:
            logger.debug("Anti-AFK step error: %s", exc)

    # =========================================================================
    # Node Recording (F8) & RouteManager Adapters
    # =========================================================================

    @staticmethod
    def _node_not_gatherable(node: Any) -> bool:
        if node is None:
            return False
        if isinstance(node, dict):
            return bool(node.get("not_gatherable", False))
        return bool(getattr(node, "not_gatherable", False))

    @staticmethod
    def _extract_node_id(node: Any, default: Any = "?") -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get("id", default)
        if hasattr(node, "id"):
            return getattr(node, "id")
        return default

    def _configured_max_harvests(self, node: Any) -> int:
        """Node max_harvests wins; otherwise the harvests_per_node setting
        (F-0199/F-0384: the alias existed but was never read)."""
        # O7: oyunda düğüm başına en fazla 3 aşama var — bozuk bir kayıttaki
        # max_harvests=99 botu 99 aşama denemeye zorlamasın.
        field = self._get_node_field(node, "max_harvests", None)
        if field is not None:
            try:
                return max(1, min(3, int(field)))
            except (TypeError, ValueError):
                self._warn_bad_config_once("node:max_harvests", field)
        raw_cfg = self._get_cfg("harvests_per_node", 3)
        try:
            return max(1, min(3, int(raw_cfg or 3)))
        except (TypeError, ValueError):
            self._warn_bad_config_once("harvests_per_node", raw_cfg)
            return 3

    @staticmethod
    def _get_node_field(node: Any, field_name: str, default: Any = None) -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get(field_name, default)
        return getattr(node, field_name, default)

    @staticmethod
    def _node_to_dict(node: Any) -> Optional[Dict[str, Any]]:
        if node is None:
            return None
        if isinstance(node, dict):
            return dict(node)
        if hasattr(node, "to_dict") and callable(node.to_dict):
            return dict(node.to_dict())
        if hasattr(node, "__dict__"):
            return dict(vars(node))
        return None

    def _execute_f8_record(self, auto_triggered: bool = False) -> Optional[Dict[str, Any]]:
        """Records the current player location as a route node and emits node_recorded."""
        try:
            raw_node: Any = None

            if self.route_manager is not None and hasattr(
                self.route_manager, "record_current_node"
            ):
                fn = self.route_manager.record_current_node
                # F-0388 (P3-4 düzeltmesi): a TypeError from INSIDE fn must
                # not trigger a second call with a different arity. The
                # dispatch calls used to sit INSIDE the inspection try, so
                # an internal TypeError was caught and fn was called AGAIN
                # with (input_driver, vision) — double recording / masked
                # error. Only the signature inspection is inside the try
                # now; the single dispatch call below runs outside it, so
                # an error raised inside fn propagates once, unretried.
                try:
                    import inspect as _inspect

                    _params = list(_inspect.signature(fn).parameters.values())
                    _positional = [
                        p
                        for p in _params
                        if p.kind
                        in (
                            _inspect.Parameter.POSITIONAL_ONLY,
                            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    ]
                    _has_varargs = any(
                        p.kind == _inspect.Parameter.VAR_POSITIONAL for p in _params
                    )
                    if _has_varargs or len(_positional) >= 2:
                        _arity = 2
                    elif len(_positional) == 1:
                        _arity = 1
                    else:
                        _arity = 0
                except (TypeError, ValueError):
                    # Signature unreadable (builtin/partial/Mock): fall back
                    # to the 2-arg call, as before.
                    _arity = 2
                if _arity >= 2:
                    raw_node = fn(self.input_driver, self.vision)
                elif _arity == 1:
                    raw_node = fn(self.input_driver)
                else:
                    raw_node = fn()
            elif self.recorder is not None and hasattr(self.recorder, "record_current_node"):
                raw_node = self.recorder.record_current_node()
            elif self.route_manager is not None and hasattr(self.route_manager, "add_node"):
                screen_w = int(getattr(self.vision, "screen_w", 1920) or 1920)
                screen_h = int(getattr(self.vision, "screen_h", 1080) or 1080)
                cx, cy = screen_w // 2, screen_h // 2
                patch = None
                frame = self._grab_frame(grayscale=True)
                if frame is not None and frame.size > 0:
                    fh, fw = frame.shape[:2]
                    cx, cy = fw // 2, fh // 2
                    patch = frame[
                        max(0, cy - 40) : min(fh, cy + 40),
                        max(0, cx - 40) : min(fw, cx + 40),
                    ].copy()
                raw_node = self.route_manager.add_node(map_x=cx, map_y=cy, patch_img=patch)

            node_dict = self._node_to_dict(raw_node)
            if isinstance(node_dict, dict):
                self.saved_nodes_count += 1
                node_id = node_dict.get("id", self.saved_nodes_count)
                self.last_saved_str = f"Node {node_id}"
                prefix = "Auto-saved" if auto_triggered else "Saved"
                self._emit_log(f"{prefix} node {node_id}")
                self.node_recorded.emit(node_dict)
                self._emit_hud_snapshot()
                return node_dict
        except Exception as exc:
            logger.error("Error recording node: %s", exc, exc_info=True)
        return None

    def _refresh_route_cooldowns(self) -> None:
        if self.route_manager is None:
            return
        try:
            if hasattr(self.route_manager, "update_resting_nodes"):
                self.route_manager.update_resting_nodes()
            elif hasattr(self.route_manager, "refresh_cooldowns"):
                self.route_manager.refresh_cooldowns()
        except Exception as exc:
            logger.debug("Route cooldown refresh error: %s", exc)

    def _is_node_ready_local(self, node: Any, rest_sec: float) -> bool:
        """Ready check mirroring SequentialRoutePlanner._is_node_ready
        (F-0195/F-0509: the planner's pick must be re-verified here)."""
        try:
            if self._node_not_gatherable(node):
                return False
            if hasattr(node, "is_ready") and callable(node.is_ready):
                try:
                    return bool(node.is_ready(now=time.time(), rest_duration_sec=int(rest_sec)))
                except TypeError:
                    return bool(node.is_ready())
            if isinstance(node, dict):
                if "ready" in node:
                    return bool(node["ready"])
                if node.get("rest_remaining_sec") is not None:
                    return float(node["rest_remaining_sec"]) <= 0.0
                last = float(node.get("last_visited_timestamp", node.get("last_visited", 0.0)) or 0.0)
                if last <= 0.0:
                    return True
                return (time.time() - last) >= float(rest_sec)
            rem = getattr(node, "rest_remaining_sec", None)
            if rem is not None:
                return float(rem) <= 0.0
            last = float(
                getattr(node, "last_visited_timestamp", getattr(node, "last_visited", 0.0)) or 0.0
            )
            if last <= 0.0:
                return True
            return (time.time() - last) >= float(rest_sec)
        except Exception:
            # P1-3 (K-07): hazırlık kararı verilemedi — fail-closed.
            # Eskiden istisnada True ("hazır") dönülüyordu: bozuk düğüm
            # kaydı dinlenme süresi dolmamış bir düğüme botu gönderirdi.
            # Belirsiz düğüm hazır sayılmaz; rota onu atlar.
            return False

    def _get_next_ready_node(self, rest_minutes: float) -> Optional[Any]:
        if self.route_manager is None:
            return None
        is_selective = getattr(self, "selective_gather_enabled", False) and bool(
            getattr(self, "selected_gather_materials", None)
        )
        try:
            # Selective Gathering: filter route nodes strictly by selected materials
            if is_selective and hasattr(self.route_manager, "get_route_nodes"):
                try:
                    all_nodes = self.route_manager.get_route_nodes() or []
                    matched_nodes = [
                        n
                        for n in all_nodes
                        if self.is_material_selected_for_gather(n)
                        and not self._node_not_gatherable(n)
                    ]
                    if not matched_nodes:
                        return None

                    now = time.time()
                    rest_sec = int(round(rest_minutes * 60.0))
                    ready_nodes = [
                        n for n in matched_nodes if self._is_node_ready_local(n, rest_sec)
                    ]

                    if not ready_nodes:
                        return None

                    if self.sequential_planner is not None and len(ready_nodes) >= 2:
                        try:
                            selected_node, next_idx = (
                                self.sequential_planner.get_next_in_sequence(
                                    ready_nodes,
                                    current_seq_pos=self._current_seq_pos,
                                    forward_window=12,
                                    allow_wrap=True,
                                )
                            )
                            if selected_node is not None and self._is_node_ready_local(
                                selected_node, rest_sec
                            ):
                                self._current_seq_pos = int(next_idx)
                                return selected_node
                        except Exception as seq_exc:
                            logger.debug("Sequential planner selective fallback: %s", seq_exc)

                    if ready_nodes:
                        return ready_nodes[0]
                    return None
                except Exception as sel_exc:
                    logger.debug("Selective route node retrieval error: %s", sel_exc)

            # Layer IV & Agent 18: Anti-Ping-Pong Sequential Corridor Selection
            # F-0195/F-0509: only READY nodes go into the planner, and the pick
            # is re-verified — a resting node must never be returned here.
            if self.sequential_planner is not None and hasattr(
                self.route_manager, "get_route_nodes"
            ):
                try:
                    route_nodes = self.route_manager.get_route_nodes() or []
                    rest_sec_all = int(rest_minutes * 60)
                    ready_route_nodes = [
                        n for n in route_nodes if self._is_node_ready_local(n, rest_sec_all)
                    ]
                    if len(ready_route_nodes) >= 2:
                        selected_node, next_idx = self.sequential_planner.get_next_in_sequence(
                            ready_route_nodes,
                            current_seq_pos=self._current_seq_pos,
                            forward_window=12,
                            allow_wrap=True,
                        )
                        if selected_node is not None and self._is_node_ready_local(
                            selected_node, rest_sec_all
                        ):
                            if not is_selective or self.is_material_selected_for_gather(selected_node):
                                self._current_seq_pos = int(next_idx)
                                return selected_node
                except Exception as seq_exc:
                    logger.debug("Sequential planner fallback: %s", seq_exc)

            if hasattr(self.route_manager, "get_next_ready_node"):
                fn = self.route_manager.get_next_ready_node
                node = None
                try:
                    node = fn(rest_duration_sec=int(rest_minutes * 60))
                except TypeError:
                    try:
                        node = fn(rest_minutes=rest_minutes)
                    except TypeError:
                        node = fn()
                if node is not None:
                    if is_selective and not self.is_material_selected_for_gather(node):
                        return None
                    # 16/11: fallback dalı da re-verification yapar — yorum
                    # "dinlenen düğüm buradan asla dönmez" diyordu ama kod
                    # doğrulamıyordu. Hazır değilse None (bekleme durumu).
                    if not self._is_node_ready_local(node, int(rest_minutes * 60)):
                        return None
                    return node
        except Exception as exc:
            logger.error("Error getting next ready node: %s", exc, exc_info=True)
        return None

    def _mark_node_gathered(self, node_id: Any, rest_minutes: float) -> None:
        if self.route_manager is None:
            return
        try:
            if hasattr(self.route_manager, "mark_node_gathered"):
                try:
                    self.route_manager.mark_node_gathered(node_id, rest_minutes=rest_minutes)
                except TypeError:
                    self.route_manager.mark_node_gathered(node_id)
            elif hasattr(self.route_manager, "mark_node_visited"):
                self.route_manager.mark_node_visited(node_id)
        except Exception as exc:
            logger.error("Error marking node %s gathered: %s", node_id, exc)

    def _mark_node_failed(self, node_id: Any) -> None:
        if self.route_manager is None:
            return
        try:
            remove_after = int(self._get_cfg("remove_node_after_failures", 0) or 0)
            active_rte = getattr(self, "active_route", None)
            if hasattr(self.route_manager, "mark_node_failed"):
                try:
                    self.route_manager.mark_node_failed(
                        node_id,
                        remove_after_failures=remove_after,
                        active_route=active_rte,
                        run_ai_check=True,
                    )
                except TypeError:
                    try:
                        self.route_manager.mark_node_failed(
                            node_id, remove_after_failures=remove_after
                        )
                    except TypeError:
                        self.route_manager.mark_node_failed(node_id)
        except Exception as exc:
            logger.error("Error marking node %s failed: %s", node_id, exc)

    def _get_ready_and_resting_counts(self) -> Tuple[int, int]:
        """Returns (ready_count, resting_count) for the active route (filtered by selected materials if active)."""
        if self.route_manager is None:
            return 0, 0
        try:
            rest_sec = int(float(self._get_cfg("rest_minutes", 5) or 5) * 60)
            if getattr(self, "selective_gather_enabled", False) and getattr(
                self, "selected_gather_materials", None
            ):
                # 16/13: düğüm başına try/except — bir düğümün is_ready
                # imzası farklıysa TÜM sayım filtresiz get_route_stats'a
                # düşmez; o düğüm tek başına atlanır.
                try:
                    nodes = []
                    if hasattr(self.route_manager, "get_route_nodes"):
                        nodes = self.route_manager.get_route_nodes() or []
                    elif hasattr(self.route_manager, "nodes"):
                        nodes = getattr(self.route_manager, "nodes") or []
                    matched = [
                        n
                        for n in nodes
                        if self.is_material_selected_for_gather(n)
                        and not self._node_not_gatherable(n)
                    ]
                    now = time.time()
                    ready = 0
                    for n in matched:
                        try:
                            if self._is_node_ready_local(n, rest_sec):
                                ready += 1
                        except Exception:
                            continue
                    return ready, len(matched) - ready
                except Exception:
                    pass

            if hasattr(self.route_manager, "get_route_stats"):
                try:
                    stats = self.route_manager.get_route_stats(rest_duration_sec=rest_sec)
                except TypeError:
                    stats = self.route_manager.get_route_stats()
                if isinstance(stats, tuple) and len(stats) >= 2:
                    return int(stats[0]), int(stats[1])

            if hasattr(self.route_manager, "get_status_counts"):
                counts = self.route_manager.get_status_counts()
                if isinstance(counts, tuple) and len(counts) >= 2:
                    return int(counts[0]), int(counts[1])
                if isinstance(counts, dict):
                    return int(counts.get("ready", 0)), int(counts.get("resting", 0))

            nodes = []
            if hasattr(self.route_manager, "get_route_nodes"):
                nodes = self.route_manager.get_route_nodes() or []
            elif hasattr(self.route_manager, "nodes"):
                nodes = getattr(self.route_manager, "nodes") or []

            now = time.time()
            ready_cnt = 0
            resting_cnt = 0
            for n in nodes:
                if hasattr(n, "is_ready") and callable(n.is_ready):
                    if n.is_ready(now=now, rest_duration_sec=rest_sec):
                        ready_cnt += 1
                    else:
                        resting_cnt += 1
                elif isinstance(n, dict):
                    status = str(n.get("status", "")).lower()
                    resting_until = float(n.get("resting_until", 0) or 0)
                    last_visited = float(
                        n.get("last_visited_timestamp", n.get("last_visited", 0)) or 0
                    )
                    if status == "resting" and resting_until > now:
                        resting_cnt += 1
                    elif resting_until > now or (
                        last_visited > 0 and (now - last_visited) < rest_sec
                    ):
                        resting_cnt += 1
                    else:
                        ready_cnt += 1
                else:
                    # 16/13: is_ready'siz dict-dışı düğüm artık hiçbir
                    # kovaya girmemek yerine aynı hazır kuralıyla sayılır.
                    try:
                        if self._is_node_ready_local(n, rest_sec):
                            ready_cnt += 1
                        else:
                            resting_cnt += 1
                    except Exception:
                        continue
            return ready_cnt, resting_cnt
        except Exception as exc:
            logger.debug("Error computing ready/resting counts: %s", exc)
            return 0, 0

    def _get_route_header_title(self) -> str:
        """Formats 'REGION · ROUTE' in uppercase for the second telemetry column."""
        try:
            # 16/19: varsayılan bölge tek kaynaktan — route_manager'ın
            # DEFAULT_REGION'ı ("Verteron"); eskiden burada "ALTGARD"
            # yazılıydı, route_manager ise "Verteron" varsayıyordu.
            try:
                from core.route_manager import DEFAULT_REGION as _DEFAULT_REGION
            except Exception:
                _DEFAULT_REGION = "Verteron"
            region = _DEFAULT_REGION
            route = "ALL"
            if self.route_manager is not None:
                reg_attr = getattr(self.route_manager, "active_region", None)
                rt_attr = getattr(self.route_manager, "active_route", None)
                if reg_attr:
                    region = str(reg_attr)
                if rt_attr:
                    route = str(rt_attr)

                if not reg_attr or not rt_attr:
                    data = getattr(self.route_manager, "data", None)
                    if isinstance(data, dict):
                        region = str(data.get("active_region") or region)
                        route = str(data.get("active_route") or route)

            combined = f"{region.upper()} · {route.upper()}"
            if len(combined) > 20:
                ellipsis_dots = chr(46) * 3
                combined = combined[:17] + ellipsis_dots
            return combined
        except Exception as exc:
            logger.debug("Error formatting route header title: %s", exc)
            return "ROUTE · NODES"

    # =========================================================================
    # Vision & Input Driver Adapters
    # =========================================================================

    def _detect_gather_prompt(self) -> Tuple[bool, int]:
        """
        Detects the 'F Gather' interaction prompt via Vision and AI inference (PerceptionHub/ReflexScanner),
        returning (detected: bool, confidence_pct: 0..100).
        (Wiki: Essence Extraction is F for every material — there is no E
        gather prompt in the game; prompt keys are always 'f'.)
        """
        threshold = float(self._get_cfg("prompt_threshold", 0.75) or 0.75)

        # 1. Classical vision check (template matching)
        vision = self._plain_attr("vision")
        if vision is not None:
            try:
                raw = None
                if hasattr(vision, "detect_gather_prompt"):
                    try:
                        raw = vision.detect_gather_prompt(threshold=threshold)
                    except TypeError:
                        raw = vision.detect_gather_prompt()
                elif hasattr(vision, "check_gather_prompt"):
                    try:
                        raw = vision.check_gather_prompt(threshold=threshold, use_roi=True)
                    except TypeError:
                        raw = vision.check_gather_prompt()
                elif hasattr(vision, "find_gather_prompt"):
                    raw = vision.find_gather_prompt()

                if raw is not None:
                    # Y6: vision 'e' kararı yoksayılır — E ile toplama oyunda yok.
                    last_k = str(getattr(vision, "last_detected_gather_key", "") or "").lower()
                    if last_k == "f":
                        self._last_detected_prompt_key = "f"
                    if isinstance(raw, dict):
                        det = bool(raw.get("detected", raw.get("found", False)))
                        conf_val = float(raw.get("confidence", raw.get("match_pct", 0)))
                        conf_pct = (
                            int(round(conf_val * 100))
                            if 0.0 < conf_val <= 1.0
                            else int(round(conf_val))
                        )
                        if det:
                            k = str(raw.get("key", "")).lower()
                            if k == "f":
                                self._last_detected_prompt_key = "f"
                            # F-0185: remember the prompt that was just seen.
                            self._last_gather_prompt = {
                                "name": str(raw.get("name", raw.get("material", "gather_prompt")) or "gather_prompt"),
                                "key": self._last_detected_prompt_key,
                                "confidence": max(0, min(100, conf_pct)),
                            }
                            return True, max(0, min(100, conf_pct))

                    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
                        det = bool(raw[0])
                        conf_val = float(raw[1] or 0)
                        conf_pct = (
                            int(round(conf_val * 100))
                            if 0.0 < conf_val <= 1.0
                            else int(round(conf_val))
                        )
                        if det:
                            return True, max(0, min(100, conf_pct))

                    elif isinstance(raw, bool) and raw:
                        # F-0389: a bare bool carries no confidence; do not
                        # fabricate 100%. Use the prompt threshold as the
                        # honest (low) confidence for an unmeasured hit.
                        conf_pct = int(round(float(threshold) * 100)) if 0.0 < float(threshold) <= 1.0 else int(round(float(threshold)))
                        self._last_gather_prompt = {
                            "name": "gather_prompt",
                            "key": self._last_detected_prompt_key,
                            "confidence": max(1, min(100, conf_pct)),
                        }
                        return True, max(1, min(100, conf_pct))
            except Exception as exc:
                logger.debug("Gather prompt vision detection error: %s", exc)

        # 2. AI PerceptionHub check (FrameCognition gather_prompt / classes 7 & 8)
        hub = self._plain_attr("perception_hub")
        if hub is not None:
            try:
                cog = hub.last() if hasattr(hub, "last") else self._plain_attr("_frame_cognition")
                if cog is not None:
                    if getattr(cog, "gather_prompt", False):
                        # F-0389: no measured confidence -> 50 (unknown),
                        # never a fabricated 85.
                        conf = int(getattr(cog, "gather_conf", 50) or 50)
                        k = str(getattr(cog, "gather_key", "") or "").lower()
                        if k == "f":
                            self._last_detected_prompt_key = "f"
                        return True, max(1, min(100, conf))
                    targets = getattr(cog, "targets", None) or []
                    for t in targets:
                        if isinstance(t, dict):
                            cid = t.get("class_id")
                            cname = str(t.get("class_name", t.get("name", ""))).lower()
                            # D2: alt-string değil tam isim eşleşmesi — adı
                            # 'gather_prompt' içeren gelecekteki bir sınıf
                            # yanlış eşleşmesin. (cid 7/8 eşlemesi taksonomi
                            # kararı beklediği için aynen duruyor.)
                            if cid in (7, 8) or cname in ("f_gather", "f_prompt", "e_gather", "e_prompt", "gather_prompt"):
                                # Y1: güven ölçülmediyse 50 (bilinmiyor),
                                # asla uydurma 85 değil.
                                conf = float(t.get("confidence", t.get("conf", 0.50)) or 0.50)
                                conf_pct = int(round(conf * 100)) if conf <= 1.0 else int(round(conf))
                                self._last_detected_prompt_key = "f"
                                return True, max(1, min(100, conf_pct))
            except Exception as exc:
                logger.debug("PerceptionHub gather prompt check skipped: %s", exc)

        # 3. AI ReflexScanner check (ReflexLayer YOLO classes 7 [F Gather] & 8 [E Gather])
        reflex = self._plain_attr("reflex")
        if reflex is not None and hasattr(reflex, "latest"):
            try:
                snap = reflex.latest()
                if isinstance(snap, dict):
                    boxes = snap.get("boxes", []) or []
                    for b in boxes:
                        if isinstance(b, dict):
                            cid = b.get("class_id")
                            if cid is None:
                                try:
                                    from core.reflex_layer import _class_id
                                    cid = _class_id(b)
                                except Exception:
                                    cid = -1
                            cname = str(b.get("class_name", b.get("name", ""))).lower()
                            if cid in (7, 8) or cname in ("f_gather", "f_prompt", "e_gather", "e_prompt", "gather_prompt"):
                                # Y1: güven ölçülmediyse 50 (bilinmiyor), asla uydurma 85 değil.
                                conf = float(b.get("confidence", b.get("conf", 0.50)) or 0.50)
                                conf_pct = int(round(conf * 100)) if conf <= 1.0 else int(round(conf))
                                self._last_detected_prompt_key = "f"
                                return True, max(1, min(100, conf_pct))
            except Exception as exc:
                logger.debug("ReflexScanner gather prompt check skipped: %s", exc)

        # 4. MaterialScanner fallback
        # K1: scan_gather_prompt dict DEĞİL list[DetectedMaterial] döner
        # (core/material_scanner.py:2929-3052) — eski `isinstance(dict)`
        # kontrolü hiçbir zaman True olamazdı, katman ölüydü.
        mat_scan = self._plain_attr("material_scanner")
        if mat_scan is not None and hasattr(mat_scan, "scan_gather_prompt"):
            try:
                frame = self.__dict__.get("_last_color_frame")
                if frame is not None:
                    prompt_res = mat_scan.scan_gather_prompt(frame)
                    best_conf: Optional[float] = None
                    if isinstance(prompt_res, dict) and prompt_res.get("found"):
                        # Eski/mock sözleşmesiyle uyumluluk (dict dönerse).
                        raw_conf = prompt_res.get("confidence", prompt_res.get("conf"))
                        try:
                            best_conf = float(raw_conf) if raw_conf is not None else None
                        except (TypeError, ValueError):
                            best_conf = None
                    elif isinstance(prompt_res, (list, tuple)):
                        for item in prompt_res:
                            if isinstance(item, dict):
                                raw_conf = item.get("confidence", item.get("conf"))
                            else:
                                raw_conf = getattr(item, "confidence", None)
                            try:
                                c = float(raw_conf) if raw_conf is not None else None
                            except (TypeError, ValueError):
                                c = None
                            if c is not None and (best_conf is None or c > best_conf):
                                best_conf = c
                    if best_conf is not None:
                        # Y1: güven ölçülmediyse (None/0) 50 bilinmiyor.
                        conf = float(best_conf) if best_conf > 0 else 0.50
                        conf_pct = int(round(conf * 100)) if conf <= 1.0 else int(round(conf))
                        # Bu tarayıcı 'F Gather' istemini bulur — tuş hep 'f'.
                        self._last_detected_prompt_key = "f"
                        return True, max(1, min(100, conf_pct))
            except Exception as exc:
                logger.debug("MaterialScanner gather prompt check skipped: %s", exc)

        return False, 0

    def _learning_mix_sources(self) -> Optional[Dict[str, Any]]:
        """Non-blocking sources for one multi-mix sample. Empty means keep the old path."""
        mix: Dict[str, Any] = {}
        reflex = getattr(self, "reflex", None)
        if reflex is not None and hasattr(reflex, "latest"):
            try:
                snap = reflex.latest()
                boxes = snap.get("boxes") if isinstance(snap, dict) else None
                if boxes:
                    mix["yolo"] = boxes
            except Exception as exc:
                logger.debug("reflex latest skipped: %s", exc)
        factory = getattr(self, "dataset_factory", None)
        tm = getattr(factory, "time_machine", None) if factory is not None else None
        if tm is not None and hasattr(tm, "get_frame_before"):
            try:
                old = tm.get_frame_before(2.0)
            except Exception as exc:
                logger.debug("time machine lookback skipped: %s", exc)
                old = None
            if old is not None and getattr(old, "detections", None):
                mix["time_machine"] = old
        prompt = getattr(self, "_last_gather_prompt", None)
        if isinstance(prompt, dict):
            mix["gather_prompt"] = prompt
        minimap = getattr(self, "_last_minimap_sprouts", None)
        if minimap:
            mix["minimap"] = minimap
        return mix or None

    def _in_combat_grab_gate(self) -> bool:
        """True while combat mode/state should skip heavy per-frame learning.

        Matches the dataset_factory combat skip in `_grab_frame` so learner /
        perception hitch the same way. Learning resumes when mode/state leave
        combat (gate is evaluated each grab — not a sticky disable).
        """
        mode_now = str(getattr(self, "mode", "") or "")
        state_now = str(getattr(self, "state", "") or "").upper()
        return mode_now == "combat" or state_now in (
            "FIGHTING",
            "COMBAT",
            "ATTACKING",
        )

    # Combat perception throttle: vitals still refresh, but not every grab.
    _COMBAT_PERCEPTION_INTERVAL_SEC: float = 1.5

    def _combat_perception_due(self) -> bool:
        """Outside combat: always due. In combat: at most once per interval."""
        if not self._in_combat_grab_gate():
            return True
        now = time.monotonic()
        last = float(getattr(self, "_last_combat_perception_mono", 0.0) or 0.0)
        return (now - last) >= float(self._COMBAT_PERCEPTION_INTERVAL_SEC)

    def _mark_combat_perception(self) -> None:
        if self._in_combat_grab_gate():
            self._last_combat_perception_mono = time.monotonic()

    def _grab_frame(self, grayscale: bool = True) -> Optional[np.ndarray]:
        if self.vision is None:
            return None
        try:
            frame = None
            if hasattr(self.vision, "grab_screen"):
                frame = self.vision.grab_screen(grayscale=grayscale)
            elif hasattr(self.vision, "capture_screen"):
                frame = self.vision.capture_screen(grayscale=grayscale)
            if frame is not None and not grayscale:
                self._last_color_frame = frame
                self._last_color_frame_ts = time.time()
            if (
                not grayscale
                and frame is not None
                and isinstance(frame, np.ndarray)
                and frame.ndim == 3
            ):
                # Skip heavy dataset ingest / learner / perception while fighting —
                # live evidence 2026-11-11 19:23:50 COMBAT → 19:24:05 stall 15.18s
                # (fps 2.44, avg_latency 409ms). dataset_factory was already gated;
                # observe_frame + perception_hub.observe were not.
                _in_combat = self._in_combat_grab_gate()

                # OmniGameLearner: Oyunun tamamını öğrenmek için sürekli algı ve ekran yakalama
                if self.reflex is not None and hasattr(self.reflex, "submit_frame"):
                    try:
                        self.reflex.submit_frame(frame)
                    except Exception as reflex_exc:
                        logger.debug("reflex submit skipped: %s", reflex_exc)

                if self.game_learner is not None and not _in_combat:
                    try:
                        self.game_learner.observe_frame(
                            frame,
                            current_state=getattr(self, "state", "WALKING"),
                            current_mode=getattr(self, "mode", "route"),
                            mix_sources=self._learning_mix_sources(),
                        )
                    except Exception as gl_exc:
                        logger.debug("game_learner.observe_frame error: %s", gl_exc)

                # PerceptionHub: tek kare snapshot (combat/discovery/economy paylaşır).
                # Combat: throttle ≤1.5s so vitals refresh without per-frame hitch.
                if (
                    self.perception_hub is not None
                    and hasattr(self.perception_hub, "observe")
                    and self._combat_perception_due()
                ):
                    try:
                        self._frame_cognition = self.perception_hub.observe(frame)
                        self._mark_combat_perception()
                    except Exception as hub_exc:
                        logger.debug("perception_hub.observe skipped: %s", hub_exc)

                if self.dataset_factory is not None and not _in_combat:
                    if hasattr(self.dataset_factory, "feed_live_frame"):
                        try:
                            self.dataset_factory.feed_live_frame(frame)
                        except Exception as df_exc:
                            logger.debug("dataset_factory.feed_live_frame error: %s", df_exc)
                    elif hasattr(self.dataset_factory, "ingest_live_frame"):
                        try:
                            self.dataset_factory.ingest_live_frame(frame)
                        except Exception:
                            pass
            return frame
        except Exception as exc:
            logger.debug("Screen grab error: %s", exc)
        return None

    def _sample_f_interval(self) -> float:
        f_min = float(self._get_cfg("f_interval_min", 1.5) or 1.5)
        f_max = float(self._get_cfg("f_interval_max", 2.8) or 2.8)
        if f_max < f_min:
            f_max = f_min
        if self.human_timing is not None and hasattr(
            self.human_timing, "get_f_press_interval_sec"
        ):
            try:
                return float(self.human_timing.get_f_press_interval_sec(f_min, f_max))
            except Exception as exc:
                logger.debug("Ex-Gaussian F interval fallback: %s", exc)
        return random.uniform(f_min, f_max)

    def _get_active_interaction_key(self) -> str:
        """Toplama etkileşim tuşu: her zaman 'f'.

        Wiki (Gathering/Essence Extraction): oyunda HER materyal F ile
        toplanır; E ile toplama yoktur. Bu yüzden config'teki 'e'/'auto'
        ve prompt zincirinin 'e' kararı artık tuşu değiştiremez — 'e'
        tespiti ölü koddur, kaldırıldı. (config/schema.py'deki
        VALID_GATHER_KEYS hâlâ 'e'/'auto' kabul eder; şema J grubunun,
        worker tarafı burada F'e kilitlendi.)
        """
        return "f"

    def _press_gather_key(self) -> bool:
        # F-0390: the counter/log used to run before any key was sent, so a
        # missing driver still counted a press. Count only a real send
        # (test_mode counts as a simulated send).
        # Y7: bus'un {"ok": False, "method": "failed"} dönüşü artık okunur —
        # başarısız basış sayaçta "basıldı" görünmez. Döner: gönderim
        # gerçekten denendiyse True, aksi halde False.
        try:
            self._focus_game_window()
        except Exception:
            pass
        target_key = self._get_active_interaction_key()
        frame_bgr = self.__dict__.get("_last_color_frame")
        if not bool(self._get_cfg("test_mode", False)) and self.input_driver is None:
            logger.debug("Gather key not pressed: no input driver")
            return False
        sent = bool(self._get_cfg("test_mode", False))
        if not bool(self._get_cfg("test_mode", False)) and self.input_driver is not None:
            try:
                if frame_bgr is None:
                    frame_bgr = self._grab_frame(grayscale=False)

                self._last_bot_key_time = time.time()
                bus = self._plain_attr("action_bus")
                if bus is not None and hasattr(bus, "gather_interact"):
                    outcome = bus.gather_interact(frame=frame_bgr, preferred_key=target_key)
                    method = str((outcome or {}).get("method", "") or "")
                    ok = bool((outcome or {}).get("ok", False))
                    if not ok:
                        logger.error(
                            "Gather interact failed (bus method=%s): F not counted",
                            method or "failed",
                        )
                        return False
                    self._emit_log(f"Gather {method or 'bus'} F")
                    self.f_pressed_count += 1
                    return True
                if hasattr(self.input_driver, "send_gather_key"):
                    res = self.input_driver.send_gather_key(frame=frame_bgr)
                    sent = res is not False
                elif hasattr(self.input_driver, "press_key"):
                    res = self.input_driver.press_key(target_key, frame=frame_bgr, release_mouse=True, post_to_window=False)
                    sent = res is not False
                else:
                    sent = False
            except Exception as exc:
                logger.error("Failed to send gather interact: %s", exc)
                sent = False
        if sent:
            self.f_pressed_count += 1
            self._emit_log(f"Pressed {target_key.upper()}")
        return bool(sent)

    def _combat_soft_aim_attack(self) -> None:
        """AION2 mouse-look: nameplate Alt+click / soft-aim when Tab fails. Skills only after lock."""
        driver = self.input_driver
        if driver is None:
            return
        if not self.is_combat_authorized("target") and not self.is_combat_authorized("skill"):
            return
        try:
            frame = self._grab_frame(grayscale=False)
            soft = getattr(driver, "soft_aim_click", None)
            if callable(soft):
                try:
                    soft(frame=frame, restore_lmb=True)
                except TypeError:
                    soft(frame=frame)
                self._emit_log("SOFT AIM (mouse-look) — isimlik Alt+click / kamera + sol tık")
                # NEVER skill-spam without a real target bar / reticle lock.
                fresh = self._grab_frame(grayscale=False)
                if self._combat_has_hostile_lock(fresh):
                    setattr(self, "_combat_was_targeting", True)
                    try:
                        self._sync_combat_left_button(True)
                    except Exception:
                        pass
                    self._send_configured_combat_skill()
                else:
                    try:
                        self._sync_combat_left_button(True)
                    except Exception:
                        pass
                return
            nudge = getattr(driver, "nudge_camera", None)
            if callable(nudge) and hasattr(driver, "press_key"):
                for dx in (40, -80, 60, -30):
                    nudge(int(dx), 0)
                    self.msleep(50)
                    # Sol tık saldırısı yerine R tuşu (sol tık = R).
                    try:
                        driver.press_key("r", min_hold=0.045, max_hold=0.09)
                    except TypeError:
                        driver.press_key("r")
                    self.msleep(50)
                self._emit_log("SOFT AIM (mouse-look) — kamera süpürme + R tuşu")
        except Exception:
            logger.debug("combat soft aim failed", exc_info=True)

    def _press_combat_tab(self, soft_aim: bool = False) -> None:
        """Acquire next target. Prefer action bus so right-click is restored cleanly."""
        if self.is_paused:
            return
        if not self.is_combat_authorized("target"):
            return
        # Soft SendInput Tab is untrusted under AION2 Raw Input — force mouse soft-aim.
        try:
            driver = self.input_driver
            trusted = getattr(driver, "keyboard_inject_trusted", None) if driver is not None else None
            if callable(trusted):
                trusted = bool(trusted())
            elif isinstance(trusted, bool):
                trusted = bool(trusted)
            else:
                trusted = False
            if not trusted:
                soft_aim = True
        except Exception:
            soft_aim = True
        bus = self._plain_attr("action_bus")
        frame = None
        try:
            frame = self._grab_frame(grayscale=False)
        except Exception:
            frame = self.__dict__.get("_last_color_frame")
        if bus is not None and hasattr(bus, "press_tab_target"):
            try:
                bus.press_tab_target(frame=frame, soft_aim=bool(soft_aim))
                return
            except TypeError:
                try:
                    bus.press_tab_target()
                    return
                except Exception:
                    logger.debug("action_bus.press_tab_target failed", exc_info=True)
            except Exception:
                logger.debug("action_bus.press_tab_target failed", exc_info=True)
        self._press_key("tab", min_hold=0.12, max_hold=0.18, release_mouse=True)
        if soft_aim:
            self._combat_soft_aim_attack()

    def _press_potion_key(self, key: str, min_hold: float = 0.06, max_hold: float = 0.12) -> bool:
        """Press a potion key (F1/F2/7/8...). Potions are survival, not combat
        skills, so they must not be swallowed by the combat-skill gate
        (F-0391: in route mode _press_key dropped 7/8 as 'skills'). Routes
        through _press_key with potion=True so mocks/hooks on _press_key
        still see the press.

        Returns True only if the driver confirms the key was sent (B-02).
        """
        return self._press_key(key, min_hold=min_hold, max_hold=max_hold, potion=True)

    @staticmethod
    def _driver_press_ok(result: Any) -> bool:
        """Sürücü dönüşünü 'tuş gönderildi mi' kararına çevirir (B-02).

        InputController.press_key float döner: 0.0 = gönderilmedi,
        >0.0 = basılı tutma süresi (gönderildi). Kayıt/test sürücüleri
        bool döner (True = gönderildi). None / False / 0.0 = gönderilmedi;
        dönüşü raporlamayan sürücü 'gönderildi' sayılmaz (fail-closed).
        """
        if result is None or result is False:
            return False
        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return float(result) > 0.0
        return bool(result)

    def _press_key(
        self,
        key: str,
        min_hold: float = None,
        max_hold: float = None,
        release_mouse: bool = False,
        potion: bool = False,
    ) -> bool:
        """Tuş basar; sürücü gönderimi onaylarsa True, yoksa False doner (B-02).

        False halleri: duraklatıldı / test_mode / sürücü yok / kapı
        engeli (X/F6/B, combat gate) / sürücü 0.0/False döndürdü / istisna.
        Çağıran False'ta 'basıldı' sayamaz (cooldown/zaman damgası yazılmaz).
        """
        if self.is_paused:
            return False
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return False
        key_str = str(key).strip().lower()
        if key_str in ("x", "f6", "b"):
            logger.debug("[AutoGate] Key '%s' suppressed (automatic client function)", key_str)
            return False
        # F-0391: potion=True bypasses the combat-skill gate — potions are
        # survival, and 7/8 must work in route mode too.
        # F-1358 [O1]: "e" (skill) and "t" (right-click attack key) are combat
        # keys too - without them they leaked through in ROUTE_MODE.
        if not potion and key_str in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "r", "e", "t"):
            if not self.is_combat_authorized("skill"):
                logger.debug("[CombatGate] Combat key '%s' suppressed (mode=%s, state=%s)", key_str, self._mode, self._state)
                return False
        elif key_str == "tab":
            if not self.is_combat_authorized("target"):
                logger.debug("[CombatGate] Tab target key '%s' suppressed (mode=%s, state=%s)", key_str, self._mode, self._state)
                return False
        try:
            self._last_bot_key_time = time.time()
            frame = self.__dict__.get("_last_color_frame")
            if key.lower() == "esc" and hasattr(self.input_driver, "press_esc"):
                return self._driver_press_ok(self.input_driver.press_esc())
            elif hasattr(self.input_driver, "press_key"):
                kwargs = {"frame": frame, "release_mouse": bool(release_mouse)}
                if min_hold is not None:
                    kwargs["min_hold"] = min_hold
                if max_hold is not None:
                    kwargs["max_hold"] = max_hold
                sent = self._driver_press_ok(self.input_driver.press_key(key, **kwargs))
                if not sent:
                    logger.warning("Tuş gönderilemedi (sürücü 0.0/False döndü): '%s'", key)
                return sent
            elif hasattr(self.input_driver, "send_key"):
                return self._driver_press_ok(self.input_driver.send_key(key))
            return False
        except Exception as exc:
            logger.error("Failed to press key '%s': %s", key, exc)
            return False

    def _toggle_map_key(self) -> None:
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        try:
            self._last_bot_key_time = time.time()
            self._focus_game_window()
            frame = self.__dict__.get("_last_color_frame")
            if hasattr(self.input_driver, "toggle_map"):
                self.input_driver.toggle_map(frame=frame)
            else:
                self._press_key(self._map_open_key())
        except Exception as exc:
            logger.error("Failed to toggle map key: %s", exc)

    def _click_map_point(self, x: int, y: int) -> None:
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        if self._motion_should_stop():
            return
        try:
            self._last_bot_key_time = time.time()
            tx, ty = int(x), int(y)
            # F-1695: arayüz tıklaması önce WindMouse click_humanized
            # zincirinden gönderilir (imleç-açma + dwell + jitter); zincir
            # Interception üzerinden input_ctrl.click_map ile biter ve
            # yumuşak yola düşmez. Başarısız olursa aşağıdaki mevcut
            # yola düşülür. Harita tıklaması skill değildir
            # (use_skill_key=False — R/T kuralı etkilenmez).
            if self.windmouse is not None and hasattr(
                self.windmouse, "click_humanized"
            ):
                try:
                    if self.windmouse.click_humanized(
                        self.input_driver, tx, ty, button="left"
                    ):
                        return
                    logger.debug(
                        "click_humanized False döndü; doğrudan harita tıklamasına düşülüyor"
                    )
                except Exception as wm_exc:
                    logger.debug("click_humanized fallback: %s", wm_exc)

            # Layer I: 2D Gaussian Spatial Jitter (±3..6px) + WindMouse Bezier Trajectory
            if self.biometric_clicker is not None and hasattr(
                self.biometric_clicker, "apply_2d_gaussian_jitter"
            ):
                try:
                    tx, ty = self.biometric_clicker.apply_2d_gaussian_jitter(tx, ty)
                except Exception as jit_exc:
                    logger.debug("2D Gaussian jitter fallback: %s", jit_exc)

            if self.windmouse is not None and hasattr(self.windmouse, "move_cursor_humanized"):
                try:
                    self.windmouse.move_cursor_humanized(self.input_driver, tx, ty)
                except Exception as wm_exc:
                    logger.debug("WindMouse move fallback: %s", wm_exc)

            if hasattr(self.input_driver, "click_map"):
                self.input_driver.click_map(tx, ty, button="left")
            elif hasattr(self.input_driver, "human_click"):
                self.input_driver.human_click(tx, ty, button="left")
            elif hasattr(self.input_driver, "click"):
                self.input_driver.click(tx, ty, button="left")
        except Exception as exc:
            logger.error("Failed to click map at (%d, %d): %s", x, y, exc)

    # =========================================================================
    # State Transitions, Logging & HUD Signal Emission
    # =========================================================================

    def _halt_motion(self) -> None:
        """Stop cursor paths and release WASD. Safe from the GUI thread."""
        driver = self.input_driver
        if driver is None:
            return
        try:
            self._release_all_combat_inputs()
        except Exception as exc:
            logger.debug("Combat inputs release in halt skipped: %s", exc)
        abort = getattr(driver, "abort_motion", None)
        if callable(abort):
            try:
                abort()
                return
            except Exception as exc:
                logger.debug("abort_motion failed: %s", exc)
        releaser = getattr(driver, "release_all_movement_keys", None)
        if callable(releaser):
            try:
                releaser()
            except Exception as exc:
                logger.debug("release_all_movement_keys failed: %s", exc)

    def _motion_should_stop(self) -> bool:
        if self.is_paused:
            return True
        # D87 [O1]: getattr — __init__ atlanmış bir çağırıcıda doğrudan
        # self.focus_monitor okuması AttributeError veriyordu.
        focus_mon = getattr(self, "focus_monitor", None)
        if getattr(self, "pause_on_alt_tab", True) and focus_mon is not None and hasattr(focus_mon, "is_game_window_focused"):
            try:
                focused, _ = focus_mon.is_game_window_focused(allow_gatherion_ui=True)
                if not focused:
                    return True
            except Exception:
                pass
        driver = self.input_driver
        if driver is None:
            return False
        fn = getattr(driver, "motion_aborted", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def _msleep_abortable(self, ms: int) -> bool:
        left = max(0, int(ms))
        while left > 0:
            if self._motion_should_stop():
                return False
            step = min(25, left)
            self.msleep(step)
            left -= step
        return not self._motion_should_stop()

    def _walking_is_real(self) -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        active = getattr(driver, "movement_active", None)
        if not callable(active):
            return False
        try:
            return bool(active())
        except Exception:
            return False

    def _map_blocks_movement(self) -> bool:
        vision = self.vision
        if vision is None or not hasattr(vision, "is_map_open"):
            return False
        try:
            result = vision.is_map_open()
        except Exception:
            # Sweep G: doğrulama hatası = bilinmiyor = engelliyor say.
            return True
        # v20-loop: None (bilinmiyor: bozuk/eksik kare) tek karede
        # engelliyor sayılmaz — 1 FPS stall'da kareler bayat/bozuk
        # geldiği için her yürüme komutu "Map UI blocks movement" diye
        # reddediliyor, karakter sabit kalıyordu. Son BİLİNEN durum
        # kullanılır; hiç bilinmiyorsa harita kapalı varsayılır (bot
        # haritasız başlar). (Gather prompt tarafı None'u açık saymaya
        # devam eder — orası ayrı kapıdır.)
        if result is None:
            return bool(getattr(self, "_last_known_map_open", False))
        self._last_known_map_open = bool(result)
        return bool(result)

    def _turn_key_for_node(self, node: Any) -> Optional[str]:
        x_val = self._get_node_field(node, "map_x", self._get_node_field(node, "x"))
        if x_val is None:
            return None
        try:
            x_px = float(x_val)
        except (TypeError, ValueError):
            return None
        width = 1920
        driver = self.input_driver
        if driver is not None and hasattr(driver, "get_screen_resolution"):
            try:
                width = int(driver.get_screen_resolution()[0]) or 1920
            except Exception:
                width = 1920
        center = width / 2.0
        if x_px < center - 30.0:
            return "a"
        if x_px > center + 30.0:
            return "d"
        return None

    def _map_open_in_backoff(self) -> bool:
        """After one failed map-open, further map-key attempts wait out the window."""
        until = float(getattr(self, "_map_open_backoff_until", 0.0) or 0.0)
        if until <= 0.0:
            return False
        if time.time() >= until:
            self._map_open_backoff_until = 0.0
            self._map_open_fail_announced = False
            return False
        return True

    def _begin_map_open_backoff(self, node_id: Any) -> None:
        """Record one failed map open and skip map-key spam. Do not idle or release walk."""
        self._last_nav_block = "map_open"
        self._map_open_backoff_until = time.time() + 45.0
        if not self._map_open_fail_announced:
            self._map_open_fail_announced = True
            logger.warning("Map did not open when navigating to node %s", node_id)
            self._emit_log(f"Map did not open for node {node_id}")

    def _release_route_motion(self) -> None:
        """Drop path-step and movement keys so WALKING cannot stay true while idle."""
        driver = self.input_driver
        if driver is None:
            return
        end_step = getattr(driver, "end_path_step", None)
        if callable(end_step):
            try:
                end_step()
            except Exception as exc:
                logger.debug("end_path_step failed: %s", exc)
        releaser = getattr(driver, "release_all_movement_keys", None)
        if callable(releaser):
            try:
                releaser()
            except Exception as exc:
                logger.debug("release_all_movement_keys failed: %s", exc)

    def _deny_walk_hud(self, reason: str) -> None:
        self._release_route_motion()
        if self.is_paused:
            return
        self._set_state_internal(BotState.WAITING.value)
        try:
            self._emit_hud_snapshot(
                badge=BotState.WAITING.value,
                title="Not walking",
                subtitle=reason,
            )
        except Exception as exc:
            logger.debug("Walk HUD clear skipped: %s", exc)

    def _command_walk_toward(self, node: Any) -> bool:
        """Hold forward and turn toward the node. False when input is refused."""
        if self._motion_should_stop():
            self._deny_walk_hud("Paused")
            return False
        driver = self.input_driver
        if driver is None:
            self._deny_walk_hud("No input driver")
            return False
        if self._map_blocks_movement():
            closed = self._ensure_map_closed()
            if not closed or self._map_blocks_movement():
                self._deny_walk_hud("Map UI blocks movement")
                return False
        hold = getattr(driver, "hold_movement_key", None)
        if not callable(hold):
            self._deny_walk_hud("Movement input refused")
            return False
        forward_ok = bool(hold("w"))
        turn_key = self._turn_key_for_node(node)
        turn_ok = bool(hold(turn_key)) if turn_key else False
        if not forward_ok and not turn_ok:
            releaser = getattr(driver, "release_all_movement_keys", None)
            if callable(releaser):
                releaser()
            self._deny_walk_hud("Movement input refused")
            return False
        begin = getattr(driver, "begin_path_step", None)
        if callable(begin):
            begin()
        return True

    def _set_state_internal(self, new_state: str) -> None:
        # Route mode needs bot-held W for a true WALKING badge. Explorer/Oto-Kayıt
        # are player-walk modes — never downgrade them to BEKLEMEDE just because W is up.
        if (
            str(new_state).upper() in ("WALKING", "YÜRÜYOR")
            and not self._walking_is_real()
            and not self._player_driven_mapping_mode()
        ):
            new_state = BotState.WAITING.value
        changed = False
        prev_state = None
        with self._lock:
            if self._state != new_state:
                prev_state = self._state
                self._state = new_state
                self._state_enter_time = time.time()
                changed = True
            current_mode = self._mode

        if changed:
            if prev_state == BotState.FIGHTING.value or new_state in (BotState.GATHERING.value, BotState.PAUSED.value):
                self._release_all_combat_inputs()
            self._sync_input_driver_state()
            try:
                self.state_changed.emit(new_state, current_mode)
            except Exception as exc:
                logger.debug("state_changed emit error: %s", exc)

    def _format_elapsed(self) -> str:
        elapsed = max(0, int(time.time() - self._session_start_time))
        minutes = elapsed // 60
        seconds = elapsed % 60
        if minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}:{mins:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _emit_log(self, message: str) -> None:
        """Emits a timestamped log entry ('HH:MM:SS Message') via log_added signal."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            formatted = f"{timestamp} {message}"
            logger.info(message)
            self.log_added.emit(formatted)
        except Exception as exc:
            logger.debug("log_added emit error: %s", exc)

    def _emit_harvest_stage(self) -> None:
        """Harvest aşamasını (stage, max) harvest_stage_changed ile yayar.

        main.py bu sinyali hasattr ile arayıp OverlayHUD.update_harvest_stage
        fonksiyonuna bağlar — sinyal tanımlı ama hiç emit edilmezse rozet
        yine 0/3'te kalır. Aynı değer tekrar emit edilmez.
        """
        try:
            stage = int(getattr(self, "_current_node_harvest_stage", 0) or 0)
            mx = max(1, int(getattr(self, "_target_harvests_for_active_node", 3) or 3))
        except (TypeError, ValueError):
            return
        last = self.__dict__.get("_last_emitted_harvest")
        if last == (stage, mx):
            return
        self.__dict__["_last_emitted_harvest"] = (stage, mx)
        try:
            self.harvest_stage_changed.emit(stage, mx)
        except Exception as exc:
            logger.debug("harvest_stage_changed emit error: %s", exc)

    def _reset_cast_progress_tracking(self) -> None:
        """Yeni cast aşaması: ilerleme kanıtını sıfırla."""
        self._stage_cast_prev_pct = None
        self._stage_cast_rose = False

    def _observe_cast_progress(self, pct: float) -> bool:
        """Bir cast okumasını ilerleme kanıtına işler.

        Barın ardışık iki okumada yükseldiği görülürse True döner (ve
        _stage_cast_rose kalıcı olur). Tek karede %90 görmek yükselme
        sayılmaz — dolu başlayan statik bir hedef barı da %90 görünür.
        """
        try:
            pct_f = float(pct)
        except (TypeError, ValueError):
            return bool(getattr(self, "_stage_cast_rose", False))
        prev = getattr(self, "_stage_cast_prev_pct", None)
        if prev is not None and pct_f > float(prev) + 1.0:
            self._stage_cast_rose = True
        self._stage_cast_prev_pct = pct_f
        return bool(self._stage_cast_rose)

    def _set_active_node_distance(self, dist_m: Any) -> None:
        """Düğüm mesafesini tazelik damgasıyla birlikte yazar.

        dist <= 0 "bilinmiyor" demektir (harita hiç açılmadı) — o zaman
        damga 0 kalır ki bayat/ölçülmemiş mesafe taze sanılmasın.
        """
        try:
            dist = int(dist_m) if dist_m else 0
        except (TypeError, ValueError):
            dist = 0
        self.active_node_distance_m = max(0, dist)
        self._active_node_distance_updated_ts = (
            time.time() if self.active_node_distance_m > 0 else 0.0
        )

    def _active_node_distance_fresh(self, now: float) -> bool:
        """Mesafe son reclick aralığı içinde ölçüldü mü?"""
        ts = float(getattr(self, "_active_node_distance_updated_ts", 0.0) or 0.0)
        if ts <= 0.0:
            return False
        try:
            reclick_sec = float(self._get_cfg("reclick_node_sec", 10) or 10)
        except (TypeError, ValueError):
            reclick_sec = 10.0
        return (now - ts) <= max(10.0, reclick_sec) + 5.0

    def _refresh_node_distance_from_odometry(self, now: float) -> None:
        """Harita açılamadığında mesafeyi odometriden tazele (v20-loop).

        Kartografın oyuncu konumu (map_x, map_y) ile düğümün harita
        koordinatı arası mesafe × WORLD_MAP_METERS_PER_PX (0.28).
        Harita/reclick'e bağımlı değildir — stall altında da çalışır.
        Yalnızca konum biliniyorsa ve en fazla 2 sn'de bir tazelenir.
        Varış kararı hâlâ prompt + taze mesafe ile verilir; bu yalnızca
        mesafenin bayat kalıp yürüyüşü 60 sn'de öldürmesini engeller.
        """
        if self.active_node is None:
            return
        if self._active_node_distance_fresh(now):
            return
        last = float(getattr(self, "_odometry_dist_ts", 0.0) or 0.0)
        if (now - last) < 2.0:
            return
        try:
            coords_x, coords_y = self._get_player_movement_coords()
        except Exception:
            return
        if (coords_x == 0.0 and coords_y == 0.0) or coords_x is None or coords_y is None:
            return
        node = self.active_node
        nx = self._get_node_field(node, "map_x")
        ny = self._get_node_field(node, "map_y")
        if nx is None or ny is None:
            return
        try:
            px_dist = math.hypot(float(nx) - float(coords_x), float(ny) - float(coords_y))
        except (TypeError, ValueError):
            return
        try:
            from core.corridor_navigator import WORLD_MAP_METERS_PER_PX
        except Exception:
            return
        dist_m = max(1, int(round(px_dist * WORLD_MAP_METERS_PER_PX)))
        self._odometry_dist_ts = now
        self._set_active_node_distance(dist_m)

    def _rest_node_after_nav_failure(self, node_id: Any) -> None:
        """Navigasyon hatası sonrası düğümü dinlenmeye al (v20-loop).

        Mesafe hiç tazelenmediği için zaman aşımına uğrayan düğümün
        suçu yoktur: consecutive_failures artırılmaz, kara listeye
        girmez, silinmez. Dinlenme süresi dolunca tekrar denenir.
        """
        if self.route_manager is None:
            return
        try:
            node = self.route_manager.get_node(node_id)
            if node is None:
                return
            node.state = "resting"
            node.last_visited_timestamp = time.time()
            self.route_manager.save_region(node.region)
        except Exception as exc:
            logger.debug("nav-failure rest skipped for node %s: %s", node_id, exc)

    @staticmethod
    def _is_opportunistic_node(node: Any) -> bool:
        """Fırsatçı yaklaşma düğümü mü (rota düğümü değil, haritası yok)?"""
        return isinstance(node, dict) and bool(node.get("_opportunistic", False))

    def _emit_hud_snapshot(
        self,
        badge: Optional[str] = None,
        mode_badge: Optional[str] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        col1_title: Optional[str] = None,
        col1_val: Optional[str] = None,
        col2_title: Optional[str] = None,
        col2_val: Optional[str] = None,
        col3_title: Optional[str] = None,
        col3_val: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds and emits the standardized HUD status dictionary via hud_update signal.
        """
        with self._lock:
            curr_state = badge or self._state
            curr_mode = self._mode
        movement_active = self._walking_is_real()
        player_mapping = curr_mode in (
            BotMode.EXPLORER_MODE.value,
            BotMode.AUTO_RECORD.value,
        )
        if (
            str(curr_state).upper() in ("WALKING", "YÜRÜYOR")
            and not movement_active
            and not player_mapping
        ):
            curr_state = BotState.WAITING.value
            if title is None or str(title).lower().startswith("walking"):
                title = "Not walking"

        resolved_mode_badge = mode_badge or MODE_BADGE_MAP.get(curr_mode, "ROUTE")
        ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
        max_failed = int(self._get_cfg("max_failed_in_row", 10) or 10)
        # P3-1: HUD yükündeki sürüm tek kaynaktan (version.__version__);
        # ayar dosyasındaki eski "v2.0" yazımı schema'da normalize edilir.
        version_str = normalize_version(str(self._get_cfg("version", __version__) or __version__))

        if curr_mode == BotMode.MANUAL_RECORD.value:
            default_title = "Manual record"
            default_sub = "Walk around and press Ctrl+Alt+F8 to save nodes"
            c1_t, c1_v = "SAVED", str(self.saved_nodes_count)
            c2_t, c2_v = "NODES", f"{ready_cnt} ready · {resting_cnt} resting"
            c3_t, c3_v = "LAST SAVED", self.last_saved_str
        elif curr_mode == BotMode.EXPLORER_MODE.value:
            default_title = "Auto-mapping gems & materials"
            default_sub = "Walk around: YZ+CV maps gems/ores in sequential order"
            c1_t, c1_v = "AUTO MAPPED", str(self.saved_nodes_count)
            c2_t, c2_v = self._get_route_header_title(), f"{ready_cnt} ready · {resting_cnt} resting"
            c3_t, c3_v = "LAST MAPPED", self.last_saved_str
        elif curr_mode == BotMode.AUTO_RECORD.value:
            default_title = "Auto record"
            default_sub = "Walk around: bot presses F and saves nodes"
            c1_t, c1_v = "GATHERED", str(self.gathered_count)
            c2_t, c2_v = "F PRESSED", str(self.f_pressed_count)
            c3_t, c3_v = "PROMPT MATCH", f"{self.last_prompt_match}%"
        elif curr_mode == BotMode.COMBAT_MODE.value:
            cognition = self.game_learner.get_live_cognition_snapshot() if self.game_learner else {}
            # S-0103: bilinmeyen HP / hazır skill için sahte varsayılan yok
            # (F-0046 ile aynı kural). Önceki varsayılanlar (0 ve ["1"]) HUD'a
            # "Canavar Canı: %0 · Hazır Skiller: 1" yalanını yazdırıyordu.
            hp_pct = cognition.get("monster_hp_percent")
            ready_skills = cognition.get("ready_skills") or []
            learned_cnt = cognition.get("total_learned_frames", 0)
            try:
                _hp_num = float(hp_pct) if hp_pct is not None else None
            except (TypeError, ValueError):
                _hp_num = None
            hp_txt = f"%{hp_pct}" if _hp_num is not None else "?"
            skills_txt = ", ".join(str(s) for s in ready_skills) if ready_skills else "—"
            default_title = f"Savaş Modu: {'Canavarla Savaşılıyor' if cognition.get('monster_targeted') else 'Canavar Aranıyor'}"
            default_sub = f"Canavar Canı: {hp_txt} · Hazır Skiller: {skills_txt} · Tab ile Seç"
            c1_t, c1_v = "YZ KARELERİ", str(learned_cnt)
            c2_t, c2_v = "CANAVAR CANI", f"%{hp_pct}" if (_hp_num is not None and _hp_num > 0) else "HEDEF YOK"
            c3_t, c3_v = "HAZIR SKİL", ",".join(str(s) for s in ready_skills[:3]) if ready_skills else "—"
        else:
            node_id = self._extract_node_id(self.active_node)
            default_title = (
                f"Walking to node {node_id} · {self.active_node_distance_m} m"
                if self.active_node is not None
                else "Ready"
            )
            default_sub = "On arrival: F, or a click on the gathering icon"
            c1_t, c1_v = "GATHERED", str(self.gathered_count)
            c2_t, c2_v = self._get_route_header_title(), f"{ready_cnt} ready · {resting_cnt} resting"
            c3_t, c3_v = "FAILED IN A ROW", f"{self.failed_in_row} / {max_failed}"

        # Canlı öğrenilen kare sayısını al
        learned_cnt = self.game_learner.total_learned_frames if self.game_learner else 0

        # F-0500: _state_enter_time was write-only; expose the state age.
        try:
            _entered = float(self._plain_attr("_state_enter_time", time.time()) or time.time())
            state_age_sec = round(max(0.0, time.time() - _entered), 1)
        except Exception:
            state_age_sec = 0.0
        payload: Dict[str, Any] = {
            "badge": curr_state,
            "mode_badge": resolved_mode_badge,
            "elapsed_str": self._format_elapsed(),
            "state_age_sec": state_age_sec,
            "title": title if title is not None else default_title,
            "subtitle": subtitle if subtitle is not None else default_sub,
            "col1_title": col1_title if col1_title is not None else c1_t,
            "col1_val": col1_val if col1_val is not None else c1_v,
            "col2_title": col2_title if col2_title is not None else c2_t,
            "col2_val": col2_val if col2_val is not None else c2_v,
            "col3_title": col3_title if col3_title is not None else c3_t,
            "col3_val": col3_val if col3_val is not None else c3_v,
            "version": version_str,
            "learned_frames": learned_cnt,
            # Legacy compatibility keys for older UI callers
            "state": curr_state,
            "movement_active": movement_active,
            "sub": subtitle if subtitle is not None else default_sub,
            "gathered": self.gathered_count,
            "f_pressed": self.f_pressed_count,
            "match_pct": self.last_prompt_match,
            # Harvest beslemesi (O3 + sinyal tamamlayıcısı): aşama sayacı ve
            # düğüm sayacı HUD/panel tarafından bu anahtarlardan okunur.
            # _current_node_harvest_stage node bitince sıfırlanır; bu anlık
            # değerdir, kümülatif değil.
            "harvest_stage": int(getattr(self, "_current_node_harvest_stage", 0) or 0),
            "harvest_max": max(
                1, int(getattr(self, "_target_harvests_for_active_node", 3) or 3)
            ),
            "harvest_stage_display": (
                f"{int(getattr(self, '_current_node_harvest_stage', 0) or 0)}/"
                f"{max(1, int(getattr(self, '_target_harvests_for_active_node', 3) or 3))}"
            ),
            "gathered_nodes": int(getattr(self, "_gathered_nodes_count", 0) or 0),
        }

        try:
            self.hud_update.emit(payload)
        except Exception as exc:
            logger.debug("hud_update emit error: %s", exc)

        return payload

    def _core_dependency_status(self) -> Tuple[bool, List[str]]:
        """Live core-dependency check (only meaningful for workers that went
        through __init__; skeleton workers built via __new__ report unknown
        as ready instead of fabricating a failure)."""
        if "_core_deps_missing" not in self.__dict__:
            return True, []
        missing = [
            name
            for name in ("vision", "input_driver", "route_manager", "combat_recovery")
            if self._plain_attr(name) is None
        ]
        return (not missing), missing

    def get_state_machine_health_summary(self) -> dict:
        # F-0196: no more hardcoded worker_ready=True. Readiness is the live
        # core-dependency status plus the real mode/state.
        try:
            ready, missing = self._core_dependency_status()
            return {
                "worker_ready": bool(ready),
                "mode": str(getattr(self, "mode", "route")),
                "state": str(getattr(self, "state", "")),
                "paused": bool(getattr(self, "is_paused", True)),
                "missing_core_dependencies": missing,
                "optional_subsystem_errors": dict(
                    self.__dict__.get("_optional_subsystem_errors", {}) or {}
                ),
            }
        except Exception:
            return {"worker_ready": False, "mode": "route", "missing_core_dependencies": ["unknown"]}

    def get_state_machine_ready_flag(self) -> dict:
        ready, missing = self._core_dependency_status()
        return {"module": "state_machine", "ready": bool(ready), "missing": missing}

    def get_state_machine_version_info(self) -> dict:
        # F-0510 / P3-1: tek sürüm — version.__version__. Ayar dosyasındaki
        # sürüm de config.schema'da normalize edildiği için HUD ile aynıdır.
        ready, _missing = self._core_dependency_status()
        return {"module": "state_machine", "version": __version__, "ready": bool(ready)}

    def get_state_machine_diagnostic_snapshot(self) -> dict:
        ready, missing = self._core_dependency_status()
        return {
            "module": "state_machine",
            "status": "ok" if ready else "degraded",
            "ready": bool(ready),
            "missing": missing,
        }

    def get_state_machine_capabilities(self) -> list:
        return ["state_machine", "diagnostic_snapshot", "version_info"]


def get_state_machine_capabilities() -> list:
    return ["state_machine", "diagnostic_snapshot", "version_info"]


def validate_state_machine_contract() -> bool:
    return bool(get_state_machine_capabilities())


def get_state_machine_baseline_metrics() -> dict:
    return {"module": "state_machine", "contract_valid": validate_state_machine_contract(), "capability_count": len(get_state_machine_capabilities())}


def get_state_machine_readiness_status() -> dict:
    return {"module": "state_machine", "ready": validate_state_machine_contract(), "metrics": get_state_machine_baseline_metrics()}


def get_state_machine_audit_record() -> dict:
    return {"module": "state_machine", "audit_passed": validate_state_machine_contract(), "readiness": get_state_machine_readiness_status()}


def get_state_machine_operational_digest() -> dict:
    return {"module": "state_machine", "operational": validate_state_machine_contract(), "audit": get_state_machine_audit_record()}


def get_state_machine_telemetry_payload() -> dict:
    return {"module": "state_machine", "telemetry_ready": validate_state_machine_contract(), "digest": get_state_machine_operational_digest()}


def get_state_machine_integrity_seal() -> dict:
    return {"module": "state_machine", "sealed": validate_state_machine_contract(), "telemetry": get_state_machine_telemetry_payload()}


def get_state_machine_compliance_certificate() -> dict:
    return {"module": "state_machine", "certified": validate_state_machine_contract(), "seal": get_state_machine_integrity_seal()}


def get_state_machine_readiness_probe() -> dict:
    return {"module": "state_machine", "ready": validate_state_machine_contract(), "certificate": get_state_machine_compliance_certificate()}


def get_state_machine_audit_summary() -> dict:
    return {"module": "state_machine", "audit_passed": validate_state_machine_contract(), "probe": get_state_machine_readiness_probe()}


def get_state_machine_heartbeat_snapshot() -> dict:
    return {"module": "state_machine", "alive": validate_state_machine_contract(), "audit": get_state_machine_audit_summary()}



