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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

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
    MANUAL_RECORD = "manual_record"
    AUTO_RECORD = "auto_record"
    EXPLORER_MODE = "explorer"

    # Alias for convenience
    ROUTE = "route"
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
    BotMode.ROUTE_MODE.value: "ROUTE",
    BotMode.MANUAL_RECORD.value: "MANUAL RECORD",
    BotMode.AUTO_RECORD.value: "AUTO RECORD",
    BotMode.EXPLORER_MODE.value: "AUTO CARTOGRAPHY",
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

        # Core state & mode flags
        self._mode: str = BotMode.ROUTE_MODE.value
        self._state: str = BotState.PAUSED.value
        self._running: bool = False
        self._paused: bool = True
        self._pause_reason: str = "Duraklatıldı · Başlatmak için F9'a basın"
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
        self._last_reclick_time: float = 0.0
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

        # Strict sequential node commitment & cursor (prevents jumping 40 -> 84)
        self._current_target_node_id: Optional[Any] = None
        self._sequential_cursor_idx: int = 0
        self._current_seq_pos: int = 0

        # Death confirmation tick counter (requires >= 3 consecutive ticks + revive_dialog check)
        self._death_confirm_ticks: int = 0

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
        self.dataset_factory: Any = self._init_optional_subsystem(
            "dataset_factory.factory_controller", "AutoLabelingDatasetFactory"
        )
        self.mapless_navigator: Any = self._init_optional_subsystem(
            "core.navigation_intel", "MaplessMicroNavigator"
        )

        # Sync initial driver flags
        self._sync_input_driver_state()

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

    def _init_optional_subsystem(self, module_name: str, class_name: str) -> Any:
        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name)
            return cls()
        except Exception as exc:
            logger.debug("Optional subsystem %s.%s skipped: %s", module_name, class_name, exc)
            return None

    def _init_auto_cartographer(self) -> Any:
        try:
            from core.auto_cartographer import AutoCartographer
            return AutoCartographer(
                route_manager=self.route_manager,
                vision_engine=self.vision,
            )
        except Exception as exc:
            logger.debug("AutoCartographer init skipped: %s", exc)
            return None

    def _init_explorer_controller(self) -> Any:
        try:
            from core.explorer_mode import ExplorerModeController
            return ExplorerModeController(
                route_manager=self.route_manager,
                vision_engine=self.vision,
                input_driver=self.input_driver,
            )
        except Exception as exc:
            logger.debug("ExplorerModeController init skipped: %s", exc)
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
            return CombatRecovery(self.vision, self.input_driver)
        except Exception as exc_rec:
            logger.debug("core.combat_recovery not found (%s), trying core.combat", exc_rec)
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
        aliases: Dict[str, Sequence[Tuple[Optional[str], str]]] = {
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
            "pause_on_keyboard_input": (
                (None, "pause_on_keyboard_input"),
                ("gathering", "pause_on_keyboard_input"),
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
            "minimap_auto_filter_check": (
                (None, "minimap_auto_filter_check"),
                ("route", "minimap_auto_filter_check"),
            ),
            "minimap_cluster_stack_bonus": (
                (None, "minimap_cluster_stack_bonus"),
                ("route", "minimap_cluster_stack_bonus"),
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

        try:
            if self.config is None:
                return default

            if key == "break_every_min":
                breaks_enabled = self._raw_cfg_lookup("breaks", "enabled")
                if (
                    breaks_enabled is False
                    and self._raw_cfg_lookup(None, "break_every_min") is None
                    and self._raw_cfg_lookup("breaks", "break_every_min") is None
                ):
                    return 0

            search_candidates = aliases.get(key, ((None, key),))
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
        except Exception as exc:
            logger.debug("Raw config lookup failed (%s.%s): %s", section, key, exc)
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

    def set_mode(self, mode: str) -> None:
        """
        Switches the active bot mode ('route', 'manual_record', 'auto_record', 'explorer')
        and resets transient navigation variables.
        """
        try:
            normalized = str(mode.value if isinstance(mode, Enum) else mode).strip().lower()
            if normalized not in (
                BotMode.ROUTE_MODE.value,
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
                self._gather_lost_frames = 0
                self._walk_start_time = 0.0
                self._last_gather_time = time.time()
                self._schedule_next_break()

                if normalized == BotMode.MANUAL_RECORD.value:
                    self._paused = False
                    self._set_state_internal(BotState.WAITING.value)
                elif normalized in (BotMode.AUTO_RECORD.value, BotMode.EXPLORER_MODE.value):
                    self._paused = False
                    self._set_state_internal(BotState.WALKING.value)
                else:
                    # When entering ROUTE_MODE, auto-sort the route into a smooth sequential loop
                    self._auto_sort_sequential_route()
                    if not self._paused:
                        self._set_state_internal(BotState.WALKING.value)

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

    def pause(self, reason: str = "Paused by user") -> None:
        """Pauses the bot execution and sets HUD badge to PAUSED."""
        try:
            with self._lock:
                already_paused = self._paused and self._state == BotState.PAUSED.value
                self._paused = True
                self._pause_reason = reason or "Paused by user"
                self.active_node = None
                self._is_gathering_active = False
                self._set_state_internal(BotState.PAUSED.value)

            self._sync_input_driver_state()
            if not already_paused:
                self._emit_log(reason)
            self._emit_hud_snapshot(
                title="Paused",
                subtitle=f"{reason} · Press F9 to resume",
            )
        except Exception as exc:
            logger.error("Error in pause(): %s", exc, exc_info=True)

    def resume(self) -> None:
        """Resumes the bot from PAUSED or BREAK state."""
        try:
            with self._lock:
                self._paused = False
                self._pause_reason = ""
                self._last_gather_time = time.time()
                self._failure_wait_until = 0.0
                self._break_end_timestamp = 0.0
                self._schedule_next_break()

                if self._mode == BotMode.MANUAL_RECORD.value:
                    target_state = BotState.WAITING.value
                else:
                    target_state = BotState.WALKING.value
                self._set_state_internal(target_state)

            self._sync_input_driver_state()
            self._emit_log(f"Resumed ({MODE_BADGE_MAP.get(self._mode, 'ROUTE')})")
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
        """Gracefully stops the QThread loop and waits briefly for termination."""
        try:
            with self._lock:
                self._running = False
                self._paused = True
            self._sync_input_driver_state()
            self._emit_log("Stopping bot worker")
            if self.isRunning() and threading.current_thread() is not self:
                self.wait(2500)
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
            if clean_key in ("f8", "f9"):
                return

            # Ignore synthetic key events generated by the bot
            if self.input_driver is not None and hasattr(self.input_driver, "is_bot_pressing"):
                try:
                    if self.input_driver.is_bot_pressing(clean_key):
                        return
                except Exception as exc:
                    logger.debug("is_bot_pressing check failed: %s", exc)

            if time.time() - self._last_bot_key_time < 0.25:
                return

            with self._lock:
                is_active_route = (
                    not self._paused
                    and self._mode == BotMode.ROUTE_MODE.value
                    and self._state != BotState.PAUSED.value
                )

            if is_active_route:
                self.pause(f"Paused: user pressed {clean_key.upper()}")
        except Exception as exc:
            logger.error("Error in on_user_key_pressed('%s'): %s", key_name, exc, exc_info=True)

    # =========================================================================
    # Main QThread Loop
    # =========================================================================

    def run(self) -> None:
        """Main QThread execution loop."""
        logger.info("BotWorker thread started.")
        with self._lock:
            self._running = True
            self._session_start_time = time.time()
            self._last_gather_time = time.time()
            self._schedule_next_break()

        self._sync_input_driver_state()
        self._emit_hud_snapshot()

        try:
            while self._is_running():
                try:
                    # 1. Process any queued F8 node recording request
                    if self._consume_f8_request():
                        self._execute_f8_record()

                    # 2. Refresh resting nodes status on RouteManager if supported
                    self._refresh_route_cooldowns()

                    # 2.5 Record watchdog heartbeat if attached
                    if self.watchdog is not None and hasattr(self.watchdog, "record_heartbeat"):
                        try:
                            self.watchdog.record_heartbeat(self.state, self.mode)
                        except Exception as wd_exc:
                            logger.debug("Watchdog heartbeat error: %s", wd_exc)

                    # 3. Check if paused
                    if self.is_paused:
                        self._set_state_internal(BotState.PAUSED.value)
                        self._emit_hud_snapshot(
                            title="Paused",
                            subtitle=f"{self._pause_reason} · Press F9 to resume",
                        )
                        self._sleep_adaptive(BotState.PAUSED.value, fallback_ms=150)
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
                    else:
                        self._step_route_mode()
                        self._sleep_adaptive(self.state, fallback_ms=100)

                except Exception as loop_exc:
                    logger.error("Error inside BotWorker iteration: %s", loop_exc, exc_info=True)
                    self.msleep(250)
        finally:
            self._sync_input_driver_state()
            logger.info("BotWorker thread exited.")

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _consume_f8_request(self) -> bool:
        with self._lock:
            if self._f8_pending:
                self._f8_pending = False
                return True
            return False

    def _sleep_adaptive(self, state: str, fallback_ms: int = 100) -> None:
        """Uses AdaptiveFramePacer (8-12 FPS walking, 20-30 FPS gathering, 2-4 FPS paused) if available."""
        try:
            if self.frame_pacer is not None and hasattr(self.frame_pacer, "get_next_sleep_interval"):
                sec = float(self.frame_pacer.get_next_sleep_interval(state))
                ms = max(25, min(350, int(sec * 1000.0)))
                self.msleep(ms)
                return
        except Exception as exc:
            logger.debug("Adaptive sleep fallback: %s", exc)
        self.msleep(fallback_ms)

    def _run_passive_auto_cartography(self, gather_while_exploring: bool = False) -> None:
        """
        Automatically scans screen & minimap for gems/ores/materials while walking,
        deduplicates them via SLAM visual odometry, saves 80x80 patches, and inserts
        them into the route in sequential corridor order without requiring manual F8!
        """
        if not self.auto_cartography_enabled or self.explorer_controller is None:
            return
        try:
            frame_bgr = self._grab_frame(grayscale=False)
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                return
            result = self.explorer_controller.step_passive_cartography(
                frame_bgr, gather_while_exploring=gather_while_exploring
            )
            if isinstance(result, dict):
                new_nodes = result.get("new_nodes", []) or []
                for nd in new_nodes:
                    nd_dict = self._node_to_dict(nd) or (nd if isinstance(nd, dict) else {})
                    self.saved_nodes_count += 1
                    nid = nd_dict.get("id", self.saved_nodes_count)
                    rtype = nd_dict.get("resource_type", "Material")
                    self.last_saved_str = f"Node {nid}"
                    self._emit_log(f"Auto-mapped #{nid} ({rtype}) in sequence")
                    self.node_recorded.emit(nd_dict)
        except Exception as exc:
            logger.debug("Passive auto-cartography error: %s", exc)

    # =========================================================================
    # Mode 0: EXPLORER_MODE ("explorer" - Gezdikçe Otomatik Haritalandır & Sırala)
    # =========================================================================

    def _step_explorer_mode(self) -> None:
        """
        Player walks/flies freely around the map; bot automatically detects and maps
        every gem/material seen on 3D screen or minimap and orders them sequentially.
        """
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
        self._run_passive_auto_cartography(gather_while_exploring=False)
        self._set_state_internal(BotState.WAITING.value)
        ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
        self._emit_hud_snapshot(
            badge=BotState.WAITING.value,
            mode_badge="MANUAL RECORD",
            title="Manual record",
            subtitle="Walk around (auto-mapping active, or press F8)",
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
        self._run_passive_auto_cartography(gather_while_exploring=False)
        detected, conf = self._detect_gather_prompt()
        self.last_prompt_match = conf
        now = time.time()

        if detected:
            self._gather_lost_frames = 0
            if not self._is_gathering_active:
                self._is_gathering_active = True
                self._emit_log(f"Gather prompt found ({conf}%)")
                self._emit_log("Gathering started")

            self._set_state_internal(BotState.GATHERING.value)
            if now - self._last_f_press_time >= self._next_f_interval:
                self._press_gather_key()
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
                self._gather_lost_frames += 1
                if self._gather_lost_frames >= 3:
                    self._is_gathering_active = False
                    self._gather_lost_frames = 0
                    self.gathered_count += 1
                    self._last_gather_time = now
                    self._emit_log(f"Resource gathered (#{self.gathered_count})")

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
        pause_no_gather_min = float(self._get_cfg("pause_after_no_gather_min", 15) or 0)
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

        # 6. Check for 'F Gather' prompt if gathering on the way is enabled or active node exists
        gather_on_way = bool(self._get_cfg("gather_on_the_way", True))
        if gather_on_way or self.active_node is not None or self._is_gathering_active:
            detected, conf = self._detect_gather_prompt()
            self.last_prompt_match = conf

            if detected:
                self._handle_route_gathering(conf, now)
                return
            elif self._is_gathering_active:
                self._gather_lost_frames += 1
                if self._gather_lost_frames >= 3:
                    self._complete_route_gathering(now)
                    return

        # 7. If no active node is selected, open map and navigate to next ready node
        if self.active_node is None:
            self._select_and_navigate_next_node()
            return

        # 8. Currently walking to active_node: monitor timeout and periodic map reclick
        self._monitor_walking_to_node(now)

    def _handle_combat_and_death(self) -> bool:
        """Checks for character death or enemy attack and responds accordingly."""
        if self.combat_recovery is None:
            return False

        try:
            frame_bgr = self._grab_frame(grayscale=False)

            # Character death check (passes frame safely; supports both frame-accepting and 0-arg mocks)
            is_dead = False
            if hasattr(self.combat_recovery, "check_character_dead"):
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
                if hasattr(self.combat_recovery, "recover_from_death"):
                    try:
                        recovered = self.combat_recovery.recover_from_death(frame_bgr)
                    except TypeError:
                        recovered = self.combat_recovery.recover_from_death()
                    if not recovered:
                        self.pause("Paused: character died")
                    else:
                        self.active_node = None
                elif hasattr(self.combat_recovery, "handle_death"):
                    self.combat_recovery.handle_death()
                    self.active_node = None
                else:
                    self.pause("Paused: character died")
                return True

            # Under attack check
            if bool(self._get_cfg("fight_back", True)):
                under_attack = False
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
                    under_attack = bool(atk_res[0]) if isinstance(atk_res, tuple) else bool(atk_res)

                if under_attack:
                    if self._state != BotState.FIGHTING.value:
                        self._emit_log("Under attack: fighting back")
                    self._set_state_internal(BotState.FIGHTING.value)
                    self._emit_hud_snapshot(
                        badge=BotState.FIGHTING.value,
                        title="Fighting enemy",
                        subtitle="Defending character before continuing route",
                    )

                    if not bool(self._get_cfg("test_mode", False)):
                        self._last_bot_key_time = time.time()
                        if hasattr(self.combat_recovery, "fight_back"):
                            self.combat_recovery.fight_back()
                        elif hasattr(self.combat_recovery, "fight_back_step"):
                            self.combat_recovery.fight_back_step()
                        elif hasattr(self.combat_recovery, "execute_combat_step"):
                            self.combat_recovery.execute_combat_step()
                    return True
        except Exception as exc:
            logger.error("Combat/death check error: %s", exc, exc_info=True)

        return False

    def _handle_blocking_game_window(self) -> bool:
        """
        Checks if a blocking game window/dialog is open (when not intentionally viewing the map).
        If open, presses Esc and logs 'A game window was open: closed with Esc'.
        """
        if self.vision is None or self._state == BotState.MAP.value:
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
                if not bool(self._get_cfg("test_mode", False)):
                    self._press_key("esc")
                self._emit_log("A game window was open: closed with Esc")
                self.msleep(200)
                return True
        except Exception as exc:
            logger.debug("Blocking window check error: %s", exc)

        return False

    def _handle_route_gathering(self, conf: int, now: float) -> None:
        """Handles pressing F and updating HUD while 'F Gather' prompt is visible in ROUTE_MODE."""
        self._gather_lost_frames = 0
        node_id = self._extract_node_id(self.active_node)

        # Layer IV: 10s Prompt-Stuck Protection (Inventory full / skill level requirement)
        if self.behavioral_guard is not None and hasattr(self.behavioral_guard, "check_prompt_stuck"):
            try:
                is_stuck, stuck_msg = self.behavioral_guard.check_prompt_stuck(
                    True, current_node_id=node_id if isinstance(node_id, int) else None, max_stuck_sec=10.0
                )
                if is_stuck:
                    if self.panic_system is not None and hasattr(self.panic_system, "trigger_panic"):
                        try:
                            self.panic_system.trigger_panic("PROMPT_STUCK_10S", stuck_msg, input_ctrl=self.input_driver)
                        except Exception:
                            logger.debug("Panic trigger skipped")
                    self.pause(stuck_msg)
                    return
            except Exception as guard_exc:
                logger.debug("check_prompt_stuck error: %s", guard_exc)

        if not self._is_gathering_active:
            self._is_gathering_active = True
            self._emit_log(f"Gather prompt found ({conf}%)")
            # Layer III: Human Visual Reaction Delay (180ms - 380ms Ex-Gaussian) before first F press
            if not bool(self._get_cfg("test_mode", False)) and self.human_timing is not None:
                try:
                    react_sec = float(self.human_timing.get_visual_reaction_delay_sec())
                    self.msleep(max(50, min(400, int(react_sec * 1000.0))))
                except Exception:
                    logger.debug("Visual reaction delay fallback")
            self._emit_log("Gathering started")

        self._set_state_internal(BotState.GATHERING.value)

        if now - self._last_f_press_time >= self._next_f_interval:
            self._press_gather_key()
            self._last_f_press_time = time.time()
            self._next_f_interval = self._sample_f_interval()

        self._emit_hud_snapshot(
            badge=BotState.GATHERING.value,
            title=f"Gathering node {node_id}",
            subtitle=f"Gather prompt recognized at {conf}%",
        )

    def _complete_route_gathering(self, now: float) -> None:
        """
        Handles completion of a gather stage (supports 3x harvest `1/3 -> 2/3 -> 3/3`,
        `Essence Extraction failed` retry without losing stages, partial node depletion,
        and automatic 2s Pre-Gather Time Machine YOLO dataset labeling).
        """
        if self.behavioral_guard is not None and hasattr(self.behavioral_guard, "check_prompt_stuck"):
            try:
                self.behavioral_guard.check_prompt_stuck(False)
            except Exception:
                logger.debug("Prompt stuck reset error")

        node_id = self._extract_node_id(self.active_node, default=None)
        test_mode = bool(self._get_cfg("test_mode", False))

        # Check chat for "Essence Extraction failed" (Item 23: do not consume harvest stage)
        if not test_mode and self.hud_reader is not None and hasattr(self.hud_reader, "parse_chat_feedback_fast"):
            try:
                frame_bgr = self._grab_frame(grayscale=False)
                if frame_bgr is not None:
                    fb = self.hud_reader.parse_chat_feedback_fast(frame_bgr)
                    if isinstance(fb, dict) and fb.get("extraction_failed"):
                        self._is_gathering_active = False
                        self._gather_lost_frames = 0
                        self._emit_log(f"Essence Extraction failed on node {node_id}: retrying same stage")
                        self._press_gather_key()
                        return
            except Exception as fb_exc:
                logger.debug("Chat feedback check skipped: %s", fb_exc)

        walk_dur = max(0.0, now - self._walk_start_time) if self._walk_start_time > 0 else 0.0
        self._is_gathering_active = False
        self._gather_lost_frames = 0
        self.gathered_count += 1
        self.failed_in_row = 0
        self._last_gather_time = now

        rest_min = float(self._get_cfg("rest_minutes", 5) or 5)
        rtype = self._get_node_field(self.active_node, "resource_type", "Orichalcum Ore") or "Orichalcum Ore"

        # Trigger Self-Training YOLO Dataset Factory (captures frame from t - 2.0s!)
        if self.dataset_factory is not None and hasattr(self.dataset_factory, "on_gather_success"):
            try:
                stage_num = max(1, min(3, self._current_node_harvest_stage + 1))
                self.dataset_factory.on_gather_success(
                    material_name=str(rtype),
                    harvest_stage=stage_num,
                    max_harvests=3,
                )
            except Exception as df_exc:
                logger.debug("Dataset factory on_gather_success skipped: %s", df_exc)

        # In live gameplay, support 3x harvests per node if the prompt reappears after 400ms
        depleted = True
        if not test_mode and self.active_node is not None:
            self._current_node_harvest_stage += 1
            max_h = int(self._get_node_field(self.active_node, "max_harvests", 3) or 3)
            if self._current_node_harvest_stage < max_h:
                self.msleep(400)
                re_detected, re_conf = self._detect_gather_prompt()
                if re_detected:
                    depleted = False
                    if self.route_manager is not None and hasattr(self.route_manager, "record_harvest_stage") and node_id is not None:
                        try:
                            self.route_manager.record_harvest_stage(node_id, rest_duration_sec=int(rest_min * 60))
                        except Exception:
                            logger.debug("record_harvest_stage incremental call skipped")
                    self._emit_log(
                        f"Node {node_id} stage {self._current_node_harvest_stage}/{max_h} gathered · continuing next stage"
                    )
                    self._handle_route_gathering(re_conf, time.time())
                    return

        self._current_node_harvest_stage = 0
        if node_id is not None:
            self._mark_node_gathered(node_id, rest_min)
            self._emit_log(f"Node {node_id} gathered (resting {int(rest_min)}m)")
        else:
            self._emit_log(f"Resource gathered (#{self.gathered_count})")

        if self.telemetry is not None and hasattr(self.telemetry, "record_gather"):
            try:
                reg = getattr(self.route_manager, "active_region", "Altgard") or "Altgard"
                rt = getattr(self.route_manager, "active_route", "All") or "All"
                self.telemetry.record_gather(
                    node_id=int(node_id) if isinstance(node_id, int) else self.gathered_count,
                    region=str(reg),
                    route=str(rt),
                    resource_type=str(rtype),
                    walk_duration_sec=walk_dur,
                    match_pct=self.last_prompt_match,
                )
            except Exception as tel_exc:
                logger.debug("Telemetry record_gather error: %s", tel_exc)

        if depleted:
            self.active_node = None
            self._walk_start_time = 0.0

    def _select_and_navigate_next_node(self) -> None:
        """Finds the next ready node, opens the map, clicks the node, and transitions to WALKING."""
        rest_min = float(self._get_cfg("rest_minutes", 5) or 5)
        next_node = self._get_next_ready_node(rest_min)

        if next_node is None:
            self._set_state_internal(BotState.WAITING.value)
            ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
            self._emit_hud_snapshot(
                badge=BotState.WAITING.value,
                title="Waiting for ready nodes",
                subtitle=f"All {resting_cnt} route nodes are currently resting",
            )
            self.msleep(500)
            return

        self.active_node = next_node
        node_id = self._extract_node_id(next_node)

        nav_ok, dist_m, under_overlay = self._open_map_and_click_node(next_node)
        if under_overlay:
            self._emit_log(f"Node {node_id} is under the overlay panel")
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            return

        if not nav_ok:
            self._emit_log(f"Could not locate node {node_id} on map")
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            return

        self.active_node_distance_m = dist_m
        self._walk_start_time = time.time()
        self._last_reclick_time = time.time()
        self._set_state_internal(BotState.WALKING.value)
        self._emit_log(f"Route: going to node {node_id} ({dist_m} m)")
        self._emit_hud_snapshot(
            badge=BotState.WALKING.value,
            title=f"Walking to node {node_id} · {dist_m} m",
            subtitle="On arrival: F, or a click on the gathering icon",
        )

    def _monitor_walking_to_node(self, now: float) -> None:
        """
        Monitors travel towards active_node:
        - Periodically re-clicks the node on the map every reclick_node_sec (if > 0)
        - Handles time_to_reach_sec timeout and consecutive failure threshold
        """
        if self.active_node is None:
            return

        node_id = self._extract_node_id(self.active_node)
        time_to_reach = float(self._get_cfg("time_to_reach_sec", 60) or 60)
        reclick_sec = float(self._get_cfg("reclick_node_sec", 10) or 0)
        elapsed_walk = now - self._walk_start_time

        # Timeout check
        if elapsed_walk >= time_to_reach:
            self._emit_log(f"Node {node_id} not reached in {int(time_to_reach)}s: skipping")
            self._mark_node_failed(node_id)
            self._register_node_failure()
            self.active_node = None
            self._walk_start_time = 0.0
            return

        # Periodic map target refresh (reclick_node_sec)
        if reclick_sec > 0 and (now - self._last_reclick_time) >= reclick_sec:
            nav_ok, dist_m, under_overlay = self._open_map_and_click_node(
                self.active_node, is_reclick=True
            )
            self._last_reclick_time = time.time()
            if under_overlay:
                self._emit_log(f"Node {node_id} is under the overlay panel")
                self._mark_node_failed(node_id)
                self._register_node_failure()
                self.active_node = None
                return
            if nav_ok:
                self.active_node_distance_m = dist_m
                self._emit_log(f"Route: going to node {node_id} ({dist_m} m)")
        else:
            # Estimate decreasing distance smoothly while walking
            if self.active_node_distance_m > 1 and time_to_reach > 0:
                progress = min(0.95, elapsed_walk / time_to_reach)
                est_dist = max(1, int(round(self.active_node_distance_m * (1.0 - progress * 0.35))))
            else:
                est_dist = max(1, self.active_node_distance_m)

            self._set_state_internal(BotState.WALKING.value)
            self._emit_hud_snapshot(
                badge=BotState.WALKING.value,
                title=f"Walking to node {node_id} · {est_dist} m",
                subtitle="On arrival: F, or a click on the gathering icon",
            )

    def _register_node_failure(self) -> None:
        """Increments failed_in_row and triggers wait_min_after_failed cooldown if limit reached."""
        self.failed_in_row += 1
        max_failed = int(self._get_cfg("max_failed_in_row", 10) or 10)
        wait_min = float(self._get_cfg("wait_min_after_failed", 5) or 5)

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
        self._set_state_internal(BotState.MAP.value)
        self._emit_hud_snapshot(
            badge=BotState.MAP.value,
            title="Looking at the map",
            subtitle="Don't touch the mouse or keyboard",
        )

        test_mode = bool(self._get_cfg("test_mode", False))

        try:
            # Open the map if not in test mode
            if not test_mode:
                opened = self._ensure_map_open()
                if not opened:
                    logger.warning("Map did not open when navigating to node %s", node_id)

            # Locate node coordinates & distance on the open map
            loc_result = self._locate_node_on_open_map(node)
            if loc_result is None:
                if not test_mode:
                    self._ensure_map_closed()
                return False, 0, False

            target_x, target_y, dist_m, under_overlay = loc_result

            if under_overlay or self._is_point_under_overlay(target_x, target_y):
                if not test_mode:
                    self._ensure_map_closed()
                return False, dist_m, True

            if not test_mode:
                self._click_map_point(target_x, target_y)
                self.msleep(150)
                self._ensure_map_closed()

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
                    patch_gray = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                    map_gray = self._grab_frame(grayscale=True)
                    if patch_gray is not None and map_gray is not None:
                        res = cv2.matchTemplate(map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= 0.70:
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

    def _parse_locate_result(
        self, raw: Any, node: Any
    ) -> Optional[Tuple[int, int, int, bool]]:
        if raw is None or raw is False:
            return None

        # Support NodeLocationResult (from core/route_manager.py) which has .to_dict() and attributes
        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            raw = raw.to_dict()

        if isinstance(raw, dict):
            if raw.get("found", True) is False:
                return None
            tx = int(raw.get("x", raw.get("map_x", self._get_node_field(node, "map_x", 0) or 0)))
            ty = int(raw.get("y", raw.get("map_y", self._get_node_field(node, "map_y", 0) or 0)))
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
        meters = max(2, int(round(pixel_dist * 0.28)))
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

    def _ensure_map_open(self) -> bool:
        try:
            if self.navigator is not None and hasattr(self.navigator, "open_map"):
                return bool(self.navigator.open_map())
            if self.route_manager is not None and hasattr(self.route_manager, "open_map"):
                return bool(self.route_manager.open_map())
            if self.vision is not None and hasattr(self.vision, "is_map_open"):
                if not self.vision.is_map_open():
                    self._toggle_map_key()
                    self.msleep(250)
                return bool(self.vision.is_map_open())
            self._toggle_map_key()
            self.msleep(250)
            return True
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
                if self.vision.is_map_open():
                    self._toggle_map_key()
                    self.msleep(180)
                return not bool(self.vision.is_map_open())
            self._toggle_map_key()
            self.msleep(180)
            return True
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
            self._last_gather_time = now
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

    def _perform_anti_afk_step(self) -> None:
        """Executes a subtle camera/mouse micro-movement or anti-AFK step during BREAK."""
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
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
    def _extract_node_id(node: Any, default: Any = "?") -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get("id", default)
        if hasattr(node, "id"):
            return getattr(node, "id")
        return default

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
                try:
                    raw_node = fn(self.input_driver, self.vision)
                except TypeError:
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

    def _get_next_ready_node(self, rest_minutes: float) -> Optional[Any]:
        if self.route_manager is None:
            return None
        try:
            # Layer IV & Agent 18: Anti-Ping-Pong Sequential Corridor Selection
            if self.sequential_planner is not None and hasattr(
                self.route_manager, "get_route_nodes"
            ):
                try:
                    route_nodes = self.route_manager.get_route_nodes() or []
                    if len(route_nodes) >= 2:
                        selected_node, next_idx = self.sequential_planner.get_next_in_sequence(
                            route_nodes,
                            current_seq_pos=self._current_seq_pos,
                            forward_window=12,
                            allow_wrap=True,
                        )
                        if selected_node is not None:
                            self._current_seq_pos = int(next_idx)
                            return selected_node
                except Exception as seq_exc:
                    logger.debug("Sequential planner fallback: %s", seq_exc)

            if hasattr(self.route_manager, "get_next_ready_node"):
                fn = self.route_manager.get_next_ready_node
                try:
                    return fn(rest_duration_sec=int(rest_minutes * 60))
                except TypeError:
                    try:
                        return fn(rest_minutes=rest_minutes)
                    except TypeError:
                        return fn()
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
            if hasattr(self.route_manager, "mark_node_failed"):
                try:
                    self.route_manager.mark_node_failed(
                        node_id, remove_after_failures=remove_after
                    )
                except TypeError:
                    self.route_manager.mark_node_failed(node_id)
        except Exception as exc:
            logger.error("Error marking node %s failed: %s", node_id, exc)

    def _get_ready_and_resting_counts(self) -> Tuple[int, int]:
        """Returns (ready_count, resting_count) for the active route."""
        if self.route_manager is None:
            return 0, 0
        try:
            rest_sec = int(float(self._get_cfg("rest_minutes", 5) or 5) * 60)

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
            return ready_cnt, resting_cnt
        except Exception as exc:
            logger.debug("Error computing ready/resting counts: %s", exc)
            return 0, 0

    def _get_route_header_title(self) -> str:
        """Formats 'REGION · ROUTE' in uppercase for the second telemetry column."""
        try:
            region = "ALTGARD"
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
        Detects the 'F Gather' prompt via Vision and returns (detected, confidence_pct: 0..100).
        """
        if self.vision is None:
            return False, 0

        threshold = float(self._get_cfg("prompt_threshold", 0.75) or 0.75)
        try:
            raw = None
            if hasattr(self.vision, "detect_gather_prompt"):
                try:
                    raw = self.vision.detect_gather_prompt(threshold=threshold)
                except TypeError:
                    raw = self.vision.detect_gather_prompt()
            elif hasattr(self.vision, "check_gather_prompt"):
                try:
                    raw = self.vision.check_gather_prompt(threshold=threshold, use_roi=True)
                except TypeError:
                    raw = self.vision.check_gather_prompt()
            elif hasattr(self.vision, "find_gather_prompt"):
                raw = self.vision.find_gather_prompt()

            if raw is None:
                return False, 0

            if isinstance(raw, dict):
                det = bool(raw.get("detected", raw.get("found", False)))
                conf_val = float(raw.get("confidence", raw.get("match_pct", 0)))
                conf_pct = (
                    int(round(conf_val * 100))
                    if 0.0 < conf_val <= 1.0
                    else int(round(conf_val))
                )
                return det, max(0, min(100, conf_pct))

            if isinstance(raw, (tuple, list)) and len(raw) >= 2:
                det = bool(raw[0])
                conf_val = float(raw[1] or 0)
                conf_pct = (
                    int(round(conf_val * 100))
                    if 0.0 < conf_val <= 1.0
                    else int(round(conf_val))
                )
                return det, max(0, min(100, conf_pct))

            if isinstance(raw, bool):
                return raw, (100 if raw else 0)
        except Exception as exc:
            logger.debug("Gather prompt detection error: %s", exc)

        return False, 0

    def _grab_frame(self, grayscale: bool = True) -> Optional[np.ndarray]:
        if self.vision is None:
            return None
        try:
            frame = None
            if hasattr(self.vision, "grab_screen"):
                frame = self.vision.grab_screen(grayscale=grayscale)
            elif hasattr(self.vision, "capture_screen"):
                frame = self.vision.capture_screen(grayscale=grayscale)
            if (
                not grayscale
                and frame is not None
                and isinstance(frame, np.ndarray)
                and frame.ndim == 3
                and self.dataset_factory is not None
                and hasattr(self.dataset_factory, "feed_live_frame")
            ):
                try:
                    self.dataset_factory.feed_live_frame(frame)
                except Exception as df_exc:
                    logger.debug("dataset_factory.feed_live_frame error: %s", df_exc)
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

    def _press_gather_key(self) -> None:
        self.f_pressed_count += 1
        if not bool(self._get_cfg("test_mode", False)) and self.input_driver is not None:
            try:
                self._last_bot_key_time = time.time()
                if hasattr(self.input_driver, "send_gather_key"):
                    self.input_driver.send_gather_key()
                elif hasattr(self.input_driver, "press_f"):
                    self.input_driver.press_f()
                elif hasattr(self.input_driver, "press_key"):
                    self.input_driver.press_key("f")
            except Exception as exc:
                logger.error("Failed to send gather key F: %s", exc)
        self._emit_log("Pressed F")

    def _press_key(self, key: str) -> None:
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        try:
            self._last_bot_key_time = time.time()
            if key.lower() == "esc" and hasattr(self.input_driver, "press_esc"):
                self.input_driver.press_esc()
            elif hasattr(self.input_driver, "press_key"):
                self.input_driver.press_key(key)
            elif hasattr(self.input_driver, "send_key"):
                self.input_driver.send_key(key)
        except Exception as exc:
            logger.error("Failed to press key '%s': %s", key, exc)

    def _toggle_map_key(self) -> None:
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        try:
            self._last_bot_key_time = time.time()
            if hasattr(self.input_driver, "toggle_map"):
                self.input_driver.toggle_map()
            else:
                self._press_key("m")
        except Exception as exc:
            logger.error("Failed to toggle map key: %s", exc)

    def _click_map_point(self, x: int, y: int) -> None:
        if bool(self._get_cfg("test_mode", False)) or self.input_driver is None:
            return
        try:
            self._last_bot_key_time = time.time()
            tx, ty = int(x), int(y)
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

    def _set_state_internal(self, new_state: str) -> None:
        changed = False
        with self._lock:
            if self._state != new_state:
                self._state = new_state
                self._state_enter_time = time.time()
                changed = True
            current_mode = self._mode

        if changed:
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

        resolved_mode_badge = mode_badge or MODE_BADGE_MAP.get(curr_mode, "ROUTE")
        ready_cnt, resting_cnt = self._get_ready_and_resting_counts()
        max_failed = int(self._get_cfg("max_failed_in_row", 10) or 10)
        version_str = str(self._get_cfg("version", "v1.6.15") or "v1.6.15")

        if curr_mode == BotMode.MANUAL_RECORD.value:
            default_title = "Manual record"
            default_sub = "Walk around and press F8 to save nodes"
            c1_t, c1_v = "SAVED", str(self.saved_nodes_count)
            c2_t, c2_v = "NODES", f"{ready_cnt} ready · {resting_cnt} resting"
            c3_t, c3_v = "LAST SAVED", self.last_saved_str
        elif curr_mode == BotMode.AUTO_RECORD.value:
            default_title = "Auto record"
            default_sub = "Walk around: bot presses F and saves nodes"
            c1_t, c1_v = "GATHERED", str(self.gathered_count)
            c2_t, c2_v = "F PRESSED", str(self.f_pressed_count)
            c3_t, c3_v = "PROMPT MATCH", f"{self.last_prompt_match}%"
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

        payload: Dict[str, Any] = {
            "badge": curr_state,
            "mode_badge": resolved_mode_badge,
            "elapsed_str": self._format_elapsed(),
            "title": title if title is not None else default_title,
            "subtitle": subtitle if subtitle is not None else default_sub,
            "col1_title": col1_title if col1_title is not None else c1_t,
            "col1_val": col1_val if col1_val is not None else c1_v,
            "col2_title": col2_title if col2_title is not None else c2_t,
            "col2_val": col2_val if col2_val is not None else c2_v,
            "col3_title": col3_title if col3_title is not None else c3_t,
            "col3_val": col3_val if col3_val is not None else c3_v,
            "version": version_str,
            # Legacy compatibility keys for older UI callers
            "state": curr_state,
            "sub": subtitle if subtitle is not None else default_sub,
            "gathered": self.gathered_count,
            "f_pressed": self.f_pressed_count,
            "match_pct": self.last_prompt_match,
        }

        try:
            self.hud_update.emit(payload)
        except Exception as exc:
            logger.debug("hud_update emit error: %s", exc)

        return payload
