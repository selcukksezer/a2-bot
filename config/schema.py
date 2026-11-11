import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

from version import __version__, normalize_version

Field = field

logger = logging.getLogger(__name__)

VALID_GATHER_KEYS = ("f", "e", "auto")
# P3-1: tek sürüm kaynağı version.__version__ ("2.0.0"). Eski "v2.0"
# yazımı ayar dosyası sürümü olarak da artık kullanılmaz; eski dosyalar
# yüklenirken normalize_version() ile kanoniğe çevrilir.
DEFAULT_VERSION = __version__

# F-0540 tek kaynak (sözleşme no 10): HP/MP iksir eşikleri yalnızca burada
# tanımlanır (yüzde, 0-100). ui/settings_window.py DEFAULT_SETTINGS bu
# sabitlerden türetilir; core/combat.py, core/mana_phase_combat.py,
# core/skill_bar_reader.py, ui/overlay_hud.py ve core/state_machine.py
# aynı sabitleri okur — 65/45 sayısı depoda tek bir yerde yazılır.
HP_POTION_THRESHOLD_PCT: int = 65
MP_POTION_THRESHOLD_PCT: int = 45

# Legacy setting keys still used by older UI/state-machine code. They map to
# the canonical schema keys so a save through an old key never silently drops.
LEGACY_KEY_ALIASES = {
    ("gathering", "min_f_interval"): "f_interval_min",
    ("gathering", "max_f_interval"): "f_interval_max",
    ("route", "rest_after_visit_min"): "rest_minutes",
    ("route", "time_to_reach_node_sec"): "time_to_reach_sec",
    ("route", "fight_back_when_attacked"): "fight_back",
    ("breaks", "interval_min"): "break_every_min",
    ("breaks", "duration_min"): "break_lasts_min",
    ("breaks", "interval_jitter_min"): "break_every_variance",
    ("breaks", "duration_jitter_min"): "break_lasts_variance",
    # F-23-10: UI LEGACY_KEY_MAP'te olup şemada eksik olan 5 eski ad — UI
    # eski adla okuyup yazıyor, şema tanımayınca SettingsManager.set()
    # bilinmeyen anahtarı reddediyordu (sessiz kayıp).
    ("route", "gathered_nodes_join_route"): "gathered_join_active_route",
    ("route", "reclick_interval_sec"): "reclick_node_sec",
    ("route", "exploration_gather"): "gather_on_the_way",
    ("route", "max_failures_before_wait"): "max_failed_in_row",
    ("route", "pause_after_idle_min"): "pause_after_no_gather_min",
    # F-23-06: SettingsWindow düz (flat) "breaks_enabled" anahtarıyla yazar;
    # yönetici yolundan (get/set) da "enabled" alanına çözülsün.
    ("breaks", "breaks_enabled"): "enabled",
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off", ""}


def _clean_consumable_key(value: Any, fallback: str, field_name: str) -> str:
    """İksir/parşömen tuşunu temizler; yasaklı tuşa düşerse yedeğe döner.

    Yasaklı küme core/mana_phase_combat.py FORBIDDEN_CONSUMABLE_KEYS ile
    aynıdır ({"x", "f6", "b"} — F6 otomatik istemci slotudur). Şema bu
    kümeyi içe aktarmaz (döngüsel bağımlılık riski); liste burada
    aynalanır, kaynak yorumda belirtilir.
    """
    raw = str(value if value is not None else fallback or "").strip()
    if not raw:
        return fallback
    if raw.lower() in ("x", "f6", "b"):
        logger.warning(
            "%s yasaklı tuş '%s' içeriyor; '%s' kullanılacak.",
            field_name,
            raw,
            fallback,
        )
        return fallback
    return raw


def _coerce_bool(value: Any, default: bool) -> bool:
    """Parse a bool without bool("false") == True. Strings are matched explicitly."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        logger.warning("Geçersiz bool değeri %r, varsayılan %r kullanılıyor.", value, default)
        return bool(default)
    return bool(value)


def _clamp(value: Any, low: float, high: float, default: float, name: str) -> float:
    """Clamp a numeric setting, logging when the stored value is corrected."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("Geçersiz sayısal değer %r (%s), varsayılan %r kullanılıyor.", value, name, default)
        return default
    if number < low or number > high:
        clamped = max(low, min(high, number))
        logger.warning("Ayar %s=%r sınır dışı, %r olarak sınırlandı.", name, value, clamped)
        return clamped
    return number


def _clamp_int(value: Any, low: int, high: int, default: int, name: str) -> int:
    return int(_clamp(value, low, high, default, name))


def _coerce(value: Any, caster: Any, default: Any) -> Any:
    """Use default for this field only. A bad value must not reset the whole file."""
    try:
        return caster(value)
    except (TypeError, ValueError):
        return default


def merge_settings_payload(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay incoming values. Keys that exist only in the file stay."""
    if not isinstance(existing, dict):
        return dict(incoming) if isinstance(incoming, dict) else {}
    if not isinstance(incoming, dict):
        return dict(existing)
    merged = dict(existing)
    for key, value in incoming.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_settings_payload(current, value)
        else:
            merged[key] = value
    return merged


@dataclass
class GatheringSettings:
    f_interval_min: float = 1.5
    f_interval_max: float = 3.0
    test_mode: bool = False
    show_overlay: bool = True
    harvests_per_node: int = 3
    inter_harvest_delay_min: float = 0.6
    inter_harvest_delay_max: float = 1.4
    retry_on_extraction_fail: bool = True
    max_retries_per_harvest_stage: int = 2
    gather_key: str = "f"
    selective_gather_enabled: bool = False
    selected_materials: List[str] = field(default_factory=list)
    proximity_route_auto: bool = True
    pause_on_alt_tab: bool = True
    # F-0573 (Karar 7): iksir/parşömen oto-kullanım ana şalteri. Varsayılan
    # açık — mevcut davranışı korur. UI kutusu ui/f9_menu.py'de zaten var
    # (gathering.auto_consumables okur); bayrağı okuma işi R26'da
    # (core/state_machine.py).
    auto_consumables: bool = True

    def validate(self) -> "GatheringSettings":
        self.f_interval_min = _clamp(self.f_interval_min, 0.1, 30.0, 1.5, "f_interval_min")
        self.f_interval_max = _clamp(
            self.f_interval_max, self.f_interval_min, 60.0, max(self.f_interval_min, 3.0), "f_interval_max"
        )

        self.test_mode = _coerce_bool(self.test_mode, False)
        self.show_overlay = _coerce_bool(self.show_overlay, True)

        self.harvests_per_node = _clamp_int(self.harvests_per_node, 1, 10, 3, "harvests_per_node")

        self.inter_harvest_delay_min = _clamp(
            self.inter_harvest_delay_min, 0.05, 15.0, 0.6, "inter_harvest_delay_min"
        )
        self.inter_harvest_delay_max = _clamp(
            self.inter_harvest_delay_max,
            self.inter_harvest_delay_min,
            30.0,
            max(self.inter_harvest_delay_min, 1.4),
            "inter_harvest_delay_max",
        )

        self.retry_on_extraction_fail = _coerce_bool(self.retry_on_extraction_fail, True)

        self.max_retries_per_harvest_stage = _clamp_int(
            self.max_retries_per_harvest_stage, 0, 10, 2, "max_retries_per_harvest_stage"
        )

        raw_key = str(self.gather_key if self.gather_key is not None else "f").strip().lower()
        self.gather_key = raw_key if raw_key in VALID_GATHER_KEYS else "f"

        self.selective_gather_enabled = _coerce_bool(self.selective_gather_enabled, False)
        if isinstance(self.selected_materials, str):
            # A single string is one material, not one character per material.
            cleaned = self.selected_materials.strip()
            self.selected_materials = [cleaned] if cleaned else []
        elif not isinstance(self.selected_materials, (list, tuple, set)):
            self.selected_materials = list(self.selected_materials) if self.selected_materials else []
        self.selected_materials = [str(m).strip() for m in self.selected_materials if str(m).strip()]
        self.proximity_route_auto = _coerce_bool(self.proximity_route_auto, True)
        self.pause_on_alt_tab = _coerce_bool(getattr(self, "pause_on_alt_tab", True), True)
        self.auto_consumables = _coerce_bool(getattr(self, "auto_consumables", True), True)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatheringSettings":
        if not isinstance(data, dict):
            return cls().validate()
        raw_mats = data.get("selected_materials", [])
        if not isinstance(raw_mats, (list, tuple, set)):
            raw_mats = [raw_mats] if raw_mats else []
        instance = cls(
            f_interval_min=_coerce(data.get("f_interval_min", 1.5), float, 1.5),
            f_interval_max=_coerce(data.get("f_interval_max", 3.0), float, 3.0),
            test_mode=_coerce_bool(data.get("test_mode", False), False),
            show_overlay=_coerce_bool(data.get("show_overlay", True), True),
            harvests_per_node=_coerce(data.get("harvests_per_node", 3), int, 3),
            inter_harvest_delay_min=_coerce(data.get("inter_harvest_delay_min", 0.6), float, 0.6),
            inter_harvest_delay_max=_coerce(data.get("inter_harvest_delay_max", 1.4), float, 1.4),
            retry_on_extraction_fail=_coerce_bool(data.get("retry_on_extraction_fail", True), True),
            max_retries_per_harvest_stage=_coerce(data.get("max_retries_per_harvest_stage", 2), int, 2),
            gather_key=str(data.get("gather_key", "f")).strip().lower(),
            selective_gather_enabled=_coerce_bool(data.get("selective_gather_enabled", False), False),
            selected_materials=[str(m).strip() for m in raw_mats if str(m).strip()],
            proximity_route_auto=_coerce_bool(data.get("proximity_route_auto", True), True),
            pause_on_alt_tab=_coerce_bool(data.get("pause_on_alt_tab", True), True),
            auto_consumables=_coerce_bool(data.get("auto_consumables", True), True),
        )
        return instance.validate()


GatheringConfig = GatheringSettings


@dataclass
class RouteSettings:
    rest_minutes: int = 5
    climb_air_max_sec: int = 10
    gathered_join_active_route: bool = True
    fly_unreachable: bool = True
    check_odyle_minimap: bool = True
    reclick_node_sec: int = 10
    time_to_reach_sec: int = 60
    gather_on_the_way: bool = False
    wait_min_after_failed: int = 5
    max_failed_in_row: int = 10
    remove_node_after_failures: int = 0
    pause_after_no_gather_min: int = 15
    fight_back: bool = True
    fight_stop_gather_sec: int = 3
    harvests_per_node: int = 3
    minimap_auto_record: bool = True
    minimap_ignore_monsters: bool = True
    hybrid_3d_and_minimap_record: bool = True
    auto_record_3d_crests: bool = True
    mp_full_threshold: float = 0.95
    # Tüketim (HP/MP/buff) ayarları — F-0540: settings_window _cfg_set bu
    # anahtarları BotSettings.route üzerine setattr ile yazar; dataclass
    # alanı değillerse asdict() kayıtta düşürür ve yeniden yüklemede
    # kaybolurlar (sahte kalıcılık). Varsayılanlar bu modülün başındaki
    # HP/MP_POTION_THRESHOLD_PCT tek-kaynak sabitleridir.
    auto_use_hp_potion: bool = True
    hp_potion_threshold_pct: int = HP_POTION_THRESHOLD_PCT
    # F-23-03: iksir/parşömen tuşları şemada yoktu — UI API'si (get/set_
    # consumable_settings) bunları bölüm-dışı düz öznitelik olarak yazıyor,
    # SettingsManager.save() asdict() ile yazdığı için diskte kayboluyordu
    # (sahte kalıcılık). Artık gerçek dataclass alanlarıdır.
    # buff_scroll_key varsayılanı F6'ydı; F6 core/mana_phase_combat.py
    # FORBIDDEN_CONSUMABLE_KEYS'te yasaklı ("automatic client slot") — F7.
    hp_potion_key: str = "F1"
    auto_use_mp_potion: bool = True
    mp_potion_threshold_pct: int = MP_POTION_THRESHOLD_PCT
    mp_potion_key: str = "F2"
    auto_refresh_buff_scroll: bool = True
    buff_scroll_interval_sec: int = 300
    buff_scroll_key: str = "F7"
    ucb_explore_ratio: float = 0.20
    skill_discovery_enabled: bool = True
    pause_on_alt_tab: bool = True
    # Minimap / dünya-haritası metre/piksel kalibrasyonu (v19, P1-7/P1-12).
    # None = kalibre edilmedi -> `core/corridor_navigator.py` yerleşik
    # sabiti kullanılır (MINIMAP 0.55 / WORLD_MAP 0.28, ikisi de canlı
    # ölçümle kalibre EDİLMEDİ). PC'de ölçüm yapılıp değer buraya yazılır;
    # ölçüm yöntemi: `calibrate_meters_per_px(bilinen_m, piksel)` — asla
    # tahminle doldurulmaz. Aralık dışı / sayı olmayan değer validate()
    # tarafından None'a çevrilir (sessizce kabul edilmez).
    minimap_meters_per_px: Optional[float] = None
    world_map_meters_per_px: Optional[float] = None

    def validate(self) -> "RouteSettings":
        self.rest_minutes = _clamp_int(self.rest_minutes, 0, 240, 5, "rest_minutes")

        self.climb_air_max_sec = _clamp_int(self.climb_air_max_sec, 1, 120, 10, "climb_air_max_sec")

        self.gathered_join_active_route = _coerce_bool(self.gathered_join_active_route, True)
        self.fly_unreachable = _coerce_bool(self.fly_unreachable, True)
        self.check_odyle_minimap = _coerce_bool(self.check_odyle_minimap, True)

        self.reclick_node_sec = _clamp_int(self.reclick_node_sec, 1, 120, 10, "reclick_node_sec")

        self.time_to_reach_sec = _clamp_int(self.time_to_reach_sec, 5, 600, 60, "time_to_reach_sec")

        self.gather_on_the_way = _coerce_bool(self.gather_on_the_way, False)

        self.wait_min_after_failed = _clamp_int(self.wait_min_after_failed, 0, 240, 5, "wait_min_after_failed")

        self.max_failed_in_row = _clamp_int(self.max_failed_in_row, 1, 100, 10, "max_failed_in_row")

        self.remove_node_after_failures = _clamp_int(self.remove_node_after_failures, 0, 100, 0, "remove_node_after_failures")

        self.pause_after_no_gather_min = _clamp_int(self.pause_after_no_gather_min, 0, 240, 15, "pause_after_no_gather_min")

        self.fight_back = _coerce_bool(self.fight_back, True)

        self.fight_stop_gather_sec = _clamp_int(self.fight_stop_gather_sec, 0, 60, 3, "fight_stop_gather_sec")

        self.harvests_per_node = _clamp_int(self.harvests_per_node, 1, 10, 3, "harvests_per_node")

        self.minimap_auto_record = _coerce_bool(self.minimap_auto_record, True)
        self.minimap_ignore_monsters = _coerce_bool(self.minimap_ignore_monsters, True)
        self.hybrid_3d_and_minimap_record = _coerce_bool(self.hybrid_3d_and_minimap_record, True)
        self.auto_record_3d_crests = _coerce_bool(self.auto_record_3d_crests, True)

        self.mp_full_threshold = _clamp(self.mp_full_threshold, 0.5, 1.0, 0.95, "mp_full_threshold")

        self.auto_use_hp_potion = _coerce_bool(self.auto_use_hp_potion, True)
        self.hp_potion_threshold_pct = _clamp_int(self.hp_potion_threshold_pct, 1, 100, HP_POTION_THRESHOLD_PCT, "hp_potion_threshold_pct")
        self.hp_potion_key = _clean_consumable_key(self.hp_potion_key, "F1", "hp_potion_key")
        self.auto_use_mp_potion = _coerce_bool(self.auto_use_mp_potion, True)
        self.mp_potion_threshold_pct = _clamp_int(self.mp_potion_threshold_pct, 1, 100, MP_POTION_THRESHOLD_PCT, "mp_potion_threshold_pct")
        self.mp_potion_key = _clean_consumable_key(self.mp_potion_key, "F2", "mp_potion_key")
        self.auto_refresh_buff_scroll = _coerce_bool(self.auto_refresh_buff_scroll, True)
        self.buff_scroll_interval_sec = _clamp_int(self.buff_scroll_interval_sec, 1, 3600, 300, "buff_scroll_interval_sec")
        self.buff_scroll_key = _clean_consumable_key(self.buff_scroll_key, "F7", "buff_scroll_key")

        self.ucb_explore_ratio = _clamp(self.ucb_explore_ratio, 0.05, 0.5, 0.20, "ucb_explore_ratio")

        self.skill_discovery_enabled = _coerce_bool(self.skill_discovery_enabled, True)
        self.pause_on_alt_tab = _coerce_bool(getattr(self, "pause_on_alt_tab", True), True)

        # Ölçek kalibrasyon geçersiz kılmaları: None = kalibre edilmedi.
        # Sayı değilse veya makul aralık (0.05..5.0 m/px) dışındaysa None'a
        # çevrilir ve uyarı loglanır; uydurma ölçek kalıcılaşamaz.
        for _key in ("minimap_meters_per_px", "world_map_meters_per_px"):
            _raw = getattr(self, _key, None)
            if _raw is None:
                continue
            try:
                _val = float(_raw)
            except (TypeError, ValueError):
                logger.warning("Ölçek kalibrasyonu sayı değil (%s=%r); yok sayılıyor.", _key, _raw)
                setattr(self, _key, None)
                continue
            if not (0.05 <= _val <= 5.0):
                logger.warning(
                    "Ölçek kalibrasyonu aralık dışı (%s=%r, aralık 0.05..5.0); yok sayılıyor.",
                    _key,
                    _raw,
                )
                setattr(self, _key, None)
            else:
                setattr(self, _key, _val)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteSettings":
        if not isinstance(data, dict):
            return cls().validate()
        instance = cls(
            rest_minutes=_coerce(data.get("rest_minutes", 5), int, 5),
            climb_air_max_sec=_coerce(data.get("climb_air_max_sec", 10), int, 10),
            gathered_join_active_route=_coerce_bool(data.get("gathered_join_active_route", True), True),
            fly_unreachable=_coerce_bool(data.get("fly_unreachable", True), True),
            check_odyle_minimap=_coerce_bool(data.get("check_odyle_minimap", True), True),
            reclick_node_sec=_coerce(data.get("reclick_node_sec", 10), int, 10),
            time_to_reach_sec=_coerce(data.get("time_to_reach_sec", 60), int, 60),
            gather_on_the_way=_coerce_bool(data.get("gather_on_the_way", False), False),
            wait_min_after_failed=_coerce(data.get("wait_min_after_failed", 5), int, 5),
            max_failed_in_row=_coerce(data.get("max_failed_in_row", 10), int, 10),
            remove_node_after_failures=_coerce(data.get("remove_node_after_failures", 0), int, 0),
            pause_after_no_gather_min=_coerce(data.get("pause_after_no_gather_min", 15), int, 15),
            fight_back=_coerce_bool(data.get("fight_back", True), True),
            fight_stop_gather_sec=_coerce(data.get("fight_stop_gather_sec", 3), int, 3),
            harvests_per_node=_coerce(data.get("harvests_per_node", 3), int, 3),
            minimap_auto_record=_coerce_bool(data.get("minimap_auto_record", True), True),
            minimap_ignore_monsters=_coerce_bool(data.get("minimap_ignore_monsters", True), True),
            hybrid_3d_and_minimap_record=_coerce_bool(data.get("hybrid_3d_and_minimap_record", True), True),
            auto_record_3d_crests=_coerce_bool(data.get("auto_record_3d_crests", True), True),
            mp_full_threshold=_coerce(data.get("mp_full_threshold", 0.95), float, 0.95),
            auto_use_hp_potion=_coerce_bool(data.get("auto_use_hp_potion", True), True),
            hp_potion_threshold_pct=_coerce(data.get("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT), int, HP_POTION_THRESHOLD_PCT),
            hp_potion_key=str(data.get("hp_potion_key", "F1") or "F1"),
            auto_use_mp_potion=_coerce_bool(data.get("auto_use_mp_potion", True), True),
            mp_potion_threshold_pct=_coerce(data.get("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT), int, MP_POTION_THRESHOLD_PCT),
            mp_potion_key=str(data.get("mp_potion_key", "F2") or "F2"),
            auto_refresh_buff_scroll=_coerce_bool(data.get("auto_refresh_buff_scroll", True), True),
            buff_scroll_interval_sec=_coerce(data.get("buff_scroll_interval_sec", 300), int, 300),
            buff_scroll_key=str(data.get("buff_scroll_key", "F7") or "F7"),
            ucb_explore_ratio=_coerce(data.get("ucb_explore_ratio", 0.20), float, 0.20),
            skill_discovery_enabled=_coerce_bool(data.get("skill_discovery_enabled", True), True),
            pause_on_alt_tab=_coerce_bool(data.get("pause_on_alt_tab", True), True),
            # Ölçek kalibrasyonu: None = kalibre edilmedi (yerleşik sabit).
            # Sayı olmayan / aralık dışı değer validate() ile None'a döner.
            minimap_meters_per_px=_coerce(data.get("minimap_meters_per_px"), float, None),
            world_map_meters_per_px=_coerce(data.get("world_map_meters_per_px"), float, None),
        )
        return instance.validate()


@dataclass
class BreakSettings:
    enabled: bool = True
    break_every_min: int = 0
    break_every_variance: int = 1
    break_lasts_min: int = 10
    break_lasts_variance: int = 1

    def validate(self) -> "BreakSettings":
        self.enabled = _coerce_bool(self.enabled, True)
        self.break_every_min = _clamp_int(self.break_every_min, 0, 1440, 0, "break_every_min")

        self.break_every_variance = _clamp_int(self.break_every_variance, 0, 120, 1, "break_every_variance")

        self.break_lasts_min = _clamp_int(self.break_lasts_min, 1, 240, 10, "break_lasts_min")

        self.break_lasts_variance = _clamp_int(self.break_lasts_variance, 0, 120, 1, "break_lasts_variance")

        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BreakSettings":
        if not isinstance(data, dict):
            return cls().validate()
        instance = cls(
            enabled=_coerce_bool(data.get("enabled", True), True),
            break_every_min=_coerce(data.get("break_every_min", 0), int, 0),
            break_every_variance=_coerce(data.get("break_every_variance", 1), int, 1),
            break_lasts_min=_coerce(data.get("break_lasts_min", 10), int, 10),
            break_lasts_variance=_coerce(data.get("break_lasts_variance", 1), int, 1),
        )
        return instance.validate()


DEFAULT_NODE_CHORD = "ctrl+alt+f8"
DEFAULT_MENU_CHORD = "ctrl+alt+f9"
DEFAULT_PAUSE_CHORD = "ctrl+alt+f10"
BARE_BLOCKED_HOTKEYS = frozenset({"f8", "f9", "f10"})

_MOD_ORDER = ("ctrl", "alt", "shift", "win")
_MOD_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "ctl": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "win",
    "windows": "win",
    "meta": "win",
    "super": "win",
    "cmd": "win",
}
_MOD_LABELS = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}


@lru_cache(maxsize=256)
def parse_hotkey_chord(spec: str) -> Tuple[str, FrozenSet[str]]:
    """Split 'ctrl+alt+f10' into ('f10', {'ctrl','alt'}). Bare 'f10' has no modifiers.

    Pure function, cached: the same handful of chords is re-parsed on every
    key event (see match_hotkey_action / HotkeyListener) and on every
    HotkeysSettings.validate() call, so the split + alias lookup cost is paid
    once per distinct chord instead of once per call. The returned frozenset
    is immutable, so sharing it across callers is safe.
    """
    text = str(spec or "").strip().lower().replace(" ", "")
    if not text:
        return "", frozenset()
    mods = []
    key = ""
    for part in text.split("+"):
        if not part:
            continue
        alias = _MOD_ALIASES.get(part)
        if alias:
            if alias not in mods:
                mods.append(alias)
        else:
            key = part
    return key, frozenset(mods)


def canonical_hotkey_chord(spec: str, fallback: str) -> str:
    """Normalize a chord. Bare F8/F9/F10 fall back so the bot cannot bind them alone."""
    key, mods = parse_hotkey_chord(spec)
    if not key or (key in BARE_BLOCKED_HOTKEYS and not mods):
        key, mods = parse_hotkey_chord(fallback)
    ordered = [name for name in _MOD_ORDER if name in mods]
    return "+".join(ordered + ([key] if key else []))


def format_hotkey_label(spec: str) -> str:
    """User-facing chord, e.g. ctrl+alt+f10 -> Ctrl+Alt+F10."""
    key, mods = parse_hotkey_chord(spec)
    parts = [_MOD_LABELS[name] for name in _MOD_ORDER if name in mods]
    if key.startswith("f") and key[1:].isdigit():
        parts.append(key.upper())
    elif len(key) == 1:
        parts.append(key.upper())
    elif key:
        parts.append(key.upper())
    return "+".join(parts)


def match_hotkey_action(
    key_name: str,
    held_modifiers: Iterable[str],
    chords: Dict[str, str],
) -> Optional[str]:
    """
    Return the action id when key+modifiers equal a configured chord.
    Bare F8/F9/F10 never match, even if a stored spec forgot its modifiers.
    """
    name = str(key_name or "").strip().lower()
    held = frozenset(str(item).strip().lower() for item in (held_modifiers or ()) if item)
    if name in BARE_BLOCKED_HOTKEYS and not held:
        return None
    for action, spec in chords.items():
        key, mods = parse_hotkey_chord(spec)
        if not key or not mods:
            continue
        if name == key and held == mods:
            return action
    return None


@dataclass
class HotkeysSettings:
    toggle_menu_f9: str = DEFAULT_MENU_CHORD
    record_node_f8: str = DEFAULT_NODE_CHORD
    pause_resume_f10: str = DEFAULT_PAUSE_CHORD

    def validate(self) -> "HotkeysSettings":
        self.toggle_menu_f9 = canonical_hotkey_chord(self.toggle_menu_f9, DEFAULT_MENU_CHORD)
        self.record_node_f8 = canonical_hotkey_chord(self.record_node_f8, DEFAULT_NODE_CHORD)
        self.pause_resume_f10 = canonical_hotkey_chord(self.pause_resume_f10, DEFAULT_PAUSE_CHORD)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HotkeysSettings":
        if not isinstance(data, dict):
            return cls().validate()
        return cls(
            toggle_menu_f9=str(data.get("toggle_menu_f9") or DEFAULT_MENU_CHORD),
            record_node_f8=str(data.get("record_node_f8") or DEFAULT_NODE_CHORD),
            pause_resume_f10=str(data.get("pause_resume_f10") or DEFAULT_PAUSE_CHORD),
        ).validate()


@dataclass
class DisplaySettings:
    monitor_idx: int = 1
    last_tab: str = "settings"

    def validate(self) -> "DisplaySettings":
        self.monitor_idx = _clamp_int(self.monitor_idx, 1, 8, 1, "monitor_idx")
        self.last_tab = str(self.last_tab or "settings").strip() or "settings"
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisplaySettings":
        if not isinstance(data, dict):
            return cls().validate()
        return cls(
            monitor_idx=_coerce(data.get("monitor_idx", 1), int, 1),
            last_tab=str(data.get("last_tab", "settings") or "settings"),
        ).validate()


def _bot_settings_as_dict(settings: "BotSettings") -> Dict[str, Any]:
    """asdict() with the partial-dict fallback for a dataclass that fails
    to convert. Shared by BotSettings.to_dict() and
    SettingsManager.export_payload() so both paths stay identical."""
    try:
        return asdict(settings)
    except Exception as exc:
        logger.error("Error converting BotSettings to dict: %s", exc)
        return {
            "gathering": asdict(settings.gathering),
            "route": asdict(settings.route),
            "breaks": asdict(settings.breaks),
            "hotkeys": asdict(settings.hotkeys),
            "display": asdict(settings.display),
            "active_region": settings.active_region,
            "active_route": settings.active_route,
            "version": settings.version,
            "reflex_model": settings.reflex_model,
            "blacklist": list(settings.blacklist) if isinstance(settings.blacklist, list) else [],
        }


@dataclass
class BotSettings:
    gathering: GatheringSettings = field(default_factory=GatheringSettings)
    route: RouteSettings = field(default_factory=RouteSettings)
    breaks: BreakSettings = field(default_factory=BreakSettings)
    hotkeys: HotkeysSettings = field(default_factory=HotkeysSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    active_region: str = "Altgard"
    active_route: str = "Dredgion Crash Site"
    version: str = DEFAULT_VERSION
    reflex_model: str = "yolov8"
    blacklist: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> "BotSettings":
        if not isinstance(self.gathering, GatheringSettings):
            self.gathering = GatheringSettings.from_dict(
                self.gathering if isinstance(self.gathering, dict) else {}
            )
        else:
            self.gathering.validate()

        if not isinstance(self.route, RouteSettings):
            self.route = RouteSettings.from_dict(
                self.route if isinstance(self.route, dict) else {}
            )
        else:
            self.route.validate()

        if not isinstance(self.breaks, BreakSettings):
            self.breaks = BreakSettings.from_dict(
                self.breaks if isinstance(self.breaks, dict) else {}
            )
        else:
            self.breaks.validate()

        if not isinstance(self.hotkeys, HotkeysSettings):
            self.hotkeys = HotkeysSettings.from_dict(
                self.hotkeys if isinstance(self.hotkeys, dict) else {}
            )
        else:
            self.hotkeys.validate()

        if not isinstance(self.display, DisplaySettings):
            self.display = DisplaySettings.from_dict(
                self.display if isinstance(self.display, dict) else {}
            )
        else:
            self.display.validate()

        # Keep harvests_per_node synchronized between gathering and route.
        # The sync is no longer silent: a mismatch is logged so a config
        # mistake is visible instead of one side being overwritten quietly.
        if self.gathering.harvests_per_node != self.route.harvests_per_node:
            logger.warning(
                "harvests_per_node uyuşmazlığı (gathering=%s, route=%s); tek değere eşitleniyor.",
                self.gathering.harvests_per_node,
                self.route.harvests_per_node,
            )
            if self.route.harvests_per_node != 3 and self.gathering.harvests_per_node == 3:
                self.gathering.harvests_per_node = self.route.harvests_per_node
            else:
                self.route.harvests_per_node = self.gathering.harvests_per_node

        self.active_region = str(self.active_region or "Altgard").strip() or "Altgard"
        self.active_route = (
            str(self.active_route or "Dredgion Crash Site").strip() or "Dredgion Crash Site"
        )
        # P3-1: eski "v2.0" yazımı kanonik "2.0.0"a çevrilir — eski ayar
        # dosyası sürüm uyumsuzluğu uyarısı vermeden yüklenir.
        self.version = normalize_version(str(self.version or DEFAULT_VERSION).strip() or DEFAULT_VERSION)
        reflex = str(self.reflex_model or "yolov8").strip().lower().replace("-", "").replace("_", "")
        if reflex in ("yolo11", "yolov11", "v11"):
            self.reflex_model = "yolo11"
        elif reflex in ("rtdetr", "rtdetrl"):
            self.reflex_model = "rtdetr"
        else:
            self.reflex_model = "yolov8"

        if not isinstance(self.blacklist, list):
            self.blacklist = []
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _bot_settings_as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotSettings":
        try:
            if not isinstance(data, dict):
                return cls().validate()
            gathering_data = data.get("gathering", {})
            route_data = data.get("route", {})
            breaks_data = data.get("breaks", {})

            if isinstance(gathering_data, dict) and isinstance(route_data, dict):
                if "harvests_per_node" in gathering_data and "harvests_per_node" not in route_data:
                    route_data = dict(route_data)
                    route_data["harvests_per_node"] = gathering_data["harvests_per_node"]
                elif "harvests_per_node" in route_data and "harvests_per_node" not in gathering_data:
                    gathering_data = dict(gathering_data)
                    gathering_data["harvests_per_node"] = route_data["harvests_per_node"]

            raw_bl = data.get("blacklist", [])
            instance = cls(
                gathering=GatheringSettings.from_dict(gathering_data),
                route=RouteSettings.from_dict(route_data),
                breaks=BreakSettings.from_dict(breaks_data),
                hotkeys=HotkeysSettings.from_dict(data.get("hotkeys", {})),
                display=DisplaySettings.from_dict(data.get("display", {})),
                active_region=str(data.get("active_region") or "Altgard"),
                active_route=str(data.get("active_route") or "Dredgion Crash Site"),
                version=str(data.get("version") or DEFAULT_VERSION),
                reflex_model=str(data.get("reflex_model", "yolov8")),
                blacklist=list(raw_bl) if isinstance(raw_bl, list) else [],
            )
            return instance.validate()
        except Exception as exc:
            logger.error("Error parsing BotSettings from dict: %s", exc)
            raise


# Alias AppSettings to BotSettings for callers referencing AppSettings.validate()
AppSettings = BotSettings


class SettingsManager:
    def __init__(self, settings_path: Union[str, Path] = "config/settings.json") -> None:
        self.settings_path = Path(settings_path)
        if not self.settings_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            candidate = project_root / self.settings_path
            if candidate.parent.exists() or str(settings_path).startswith("config"):
                self.settings_path = candidate

        self.settings: BotSettings = BotSettings()
        self._allow_overwrite = True
        self.load_error: Optional[str] = None
        # F-23-10: dosyadaki sürüm DEFAULT_VERSION ("2.0.0") değilse True —
        # SettingsWindow açılışta bunu görür ve kullanıcıyı uyarır (eski
        # v1.6.15 dosyasının sessizce varsayılanlarla değişmesi engellenir).
        self.version_mismatch: bool = False
        try:
            if not self.settings_path.exists():
                logger.info(
                    "Settings file not found at %s. Creating default settings.",
                    self.settings_path,
                )
                self.reset_to_defaults()
            else:
                self.load()
        except Exception as exc:
            logger.error("Failed to initialize SettingsManager: %s", exc)
            self._allow_overwrite = not self.settings_path.exists()
            self.settings = BotSettings().validate()

    def load(self) -> BotSettings:
        try:
            if not self.settings_path.exists():
                logger.warning(
                    "Settings file %s does not exist during load(). Creating defaults.",
                    self.settings_path,
                )
                self._allow_overwrite = True
                return self.reset_to_defaults()

            raw_text = self.settings_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ValueError("settings root must be an object")
            self.settings = BotSettings.from_dict(data)
            self._allow_overwrite = True
            self.load_error = None
            # F-23-10: eski sürüm dosyası (örn. v1.6.15) sessizce
            # varsayılanlarla değişmesin — bayrak + uyarı; UI bunu gösterir.
            self.version_mismatch = str(self.settings.version or "") != DEFAULT_VERSION
            if self.version_mismatch:
                logger.warning(
                    "Ayar dosyası sürümü '%s', beklenen '%s' — eski dosya; "
                    "bilinmeyen anahtarlar korunur, eksikler varsayılanla dolar.",
                    self.settings.version,
                    DEFAULT_VERSION,
                )
            logger.info("Successfully loaded settings from %s", self.settings_path)
            return self.settings
        except Exception as exc:
            logger.error("Error loading settings from %s: %s", self.settings_path, exc)
            self.load_error = str(exc)
            # A corrupt file is never silently discarded: keep a backup copy
            # next to it so the user's values can be recovered.
            # 26-F3 (L devri): yedek zaman damgalıdır — damgasız
            # "settings.json.corrupt.bak" her yeni bozulmada önceki yedeği
            # eziyor, kullanıcının bir önceki ayarları kayboluyordu.
            raw_text = locals().get("raw_text")
            if isinstance(raw_text, str) and raw_text:
                try:
                    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                    backup_path = self.settings_path.with_name(
                        f"{self.settings_path.name}.corrupt.{stamp}.bak"
                    )
                    backup_path.write_text(raw_text, encoding="utf-8")
                    logger.error("Bozuk ayar dosyası yedeklendi: %s", backup_path)
                except Exception as backup_exc:
                    logger.error("Bozuk ayar dosyası yedeklenemedi: %s", backup_exc)
            # Keep the file. A later close must not replace it with defaults.
            self._allow_overwrite = False
            self.settings = BotSettings().validate()
            return self.settings

    def export_payload(self) -> Dict[str, Any]:
        """In-memory settings over the file. Unknown file keys stay in the result.

        Validates exactly once: BotSettings.to_dict() would validate again,
        so the dict conversion goes through _bot_settings_as_dict() directly
        (same output, same fallback) instead of paying for 2nd/3rd passes.
        """
        self.settings.validate()
        incoming = _bot_settings_as_dict(self.settings)
        existing: Dict[str, Any] = {}
        try:
            if self.settings_path.exists():
                raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
        except Exception as exc:
            logger.debug("Settings merge base unreadable, using in-memory payload: %s", exc)
            existing = {}
        return merge_settings_payload(existing, incoming)

    def save(self, settings: Optional[BotSettings] = None) -> bool:
        try:
            if settings is not None:
                self.settings = settings
                self._allow_overwrite = True
            if not getattr(self, "_allow_overwrite", True) and self.settings_path.exists():
                logger.error(
                    "Refusing to overwrite %s after a failed load.",
                    self.settings_path,
                )
                return False
            # No separate validate() here: export_payload() below validates
            # exactly once (a second pass would only re-clamp already-valid
            # values and re-log nothing).
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            # Source of truth: config/settings.json. settings.enc is an encrypted mirror and is never loaded over this file.
            payload = self.export_payload()
            json_text = json.dumps(payload, indent=4, ensure_ascii=False)
            # F-23-08: atomik yazma (tmp + os.replace) — doğrudan write_text
            # yazma sırasında çökmede yarım (torn) settings.json bırakıyordu;
            # sonraki açılış bozuk-dosya yoluna düşüyordu.
            tmp_path = self.settings_path.with_name(
                f"{self.settings_path.name}.tmp.{os.getpid()}"
            )
            try:
                tmp_path.write_text(json_text, encoding="utf-8")
                os.replace(tmp_path, self.settings_path)
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                raise
            enc_candidate = self.settings_path.parent / "settings.enc"
            if enc_candidate.exists():
                try:
                    from core.opsec_crypto import LocalVaultCrypto

                    salt_candidate = self.settings_path.parent / ".vault_salt"
                    vault = LocalVaultCrypto(
                        vault_path=str(enc_candidate),
                        salt_path=str(salt_candidate),
                    )
                    vault.save_encrypted_config(payload, str(enc_candidate))
                except Exception as enc_exc:
                    logger.debug("Optional encrypted vault sync skipped: %s", enc_exc)
            logger.info("Successfully saved settings to %s", self.settings_path)
            return True
        except Exception as exc:
            logger.error("Error saving settings to %s: %s", self.settings_path, exc)
            return False

    def reset_to_defaults(self) -> BotSettings:
        try:
            self._allow_overwrite = True
            self.settings = BotSettings().validate()
            self.save(self.settings)
            logger.info("Settings have been reset to defaults.")
            return self.settings
        except Exception as exc:
            logger.error("Error resetting settings to defaults: %s", exc)
            self.settings = BotSettings().validate()
            return self.settings

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        try:
            if key is not None:
                key = LEGACY_KEY_ALIASES.get((section, key), key)
            if key is None:
                return getattr(self.settings, section, default)

            if section in ("", "bot", "root", "general"):
                return getattr(self.settings, key, default)

            section_obj = getattr(self.settings, section, None)
            if section_obj is None:
                return default
            if isinstance(section_obj, dict):
                return section_obj.get(key, default)
            return getattr(section_obj, key, default)
        except Exception as exc:
            logger.error("Error getting setting [%s].[%s]: %s", section, key, exc)
            return default

    def set(self, section: str, key: str, value: Any) -> bool:
        try:
            key = LEGACY_KEY_ALIASES.get((section, key), key)
            if section in ("", "bot", "root", "general"):
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                    self.settings.validate()
                    return self.save(self.settings)
                logger.warning("Setting [root].[%s] not found in schema.", key)
                return False

            section_obj = getattr(self.settings, section, None)
            if section_obj is not None and not hasattr(section_obj, "__dataclass_fields__"):
                # `section` itself names a plain top-level attribute
                # (legacy set("version", ...) style call).
                setattr(self.settings, section, value)
                self.settings.validate()
                return self.save(self.settings)

            if section_obj is not None and hasattr(section_obj, key):
                current_val = getattr(section_obj, key)
                if isinstance(current_val, bool):
                    cast_val = _coerce_bool(value, bool(current_val))
                elif isinstance(current_val, int) and not isinstance(current_val, bool):
                    cast_val = int(value)
                elif isinstance(current_val, float):
                    cast_val = float(value)
                elif isinstance(current_val, str):
                    cast_val = str(value)
                elif isinstance(current_val, list):
                    if isinstance(value, (list, tuple, set)):
                        cast_val = [str(x).strip() for x in value if str(x).strip()]
                    elif isinstance(value, str):
                        cast_val = [str(x).strip() for x in value.split(",") if str(x).strip()]
                    else:
                        cast_val = list(value) if value else []
                else:
                    cast_val = value
                setattr(section_obj, key, cast_val)
                if key == "harvests_per_node":
                    clamped_harvests = max(1, min(10, int(cast_val)))
                    self.settings.gathering.harvests_per_node = clamped_harvests
                    self.settings.route.harvests_per_node = clamped_harvests
                self.settings.validate()
                return self.save(self.settings)

            # Never fall through to a top-level attribute: a key that does
            # not exist in the requested section must not silently land in
            # a different section (e.g. set("gathering", "version", ...)).
            logger.warning("Setting [%s].[%s] not found in schema.", section, key)
            return False
        except Exception as exc:
            logger.error("Error setting [%s].[%s] = %s: %s", section, key, value, exc)
            return False


def load_settings(settings_path: Union[str, Path] = "config/settings.json") -> BotSettings:
    """Convenience loader returning validated BotSettings/AppSettings from disk."""
    manager = SettingsManager(settings_path=settings_path)
    return manager.settings.validate()


def save_settings(
    settings: BotSettings,
    settings_path: Union[str, Path] = "config/settings.json",
) -> bool:
    """Convenience saver persisting validated BotSettings/AppSettings to disk."""
    manager = SettingsManager(settings_path=settings_path)
    return manager.save(settings)


def _probe_schema_health() -> bool:
    """Real readiness probe: default settings must validate, the canonical
    hotkey chords must parse to their expected values, and a settings
    payload must survive a merge round-trip."""
    try:
        settings = BotSettings().validate()
        if settings.gathering.harvests_per_node != settings.route.harvests_per_node:
            return False
        if canonical_hotkey_chord(settings.hotkeys.toggle_menu_f9, DEFAULT_MENU_CHORD) != DEFAULT_MENU_CHORD:
            return False
        if canonical_hotkey_chord(settings.hotkeys.record_node_f8, DEFAULT_NODE_CHORD) != DEFAULT_NODE_CHORD:
            return False
        if canonical_hotkey_chord(settings.hotkeys.pause_resume_f10, DEFAULT_PAUSE_CHORD) != DEFAULT_PAUSE_CHORD:
            return False
        merged = merge_settings_payload({"keep_me": 1}, settings.to_dict())
        if merged.get("keep_me") != 1:
            return False
        if merged.get("gathering", {}).get("test_mode") is not settings.gathering.test_mode:
            return False
        return True
    except Exception as exc:
        logger.error("Schema health probe failed: %s", exc)
        return False


def get_schema_validation_summary() -> dict:
    try:
        return {"schema_ready": _probe_schema_health(), "unified_file": "config/settings.json"}
    except Exception as exc:
        logger.error("Error in get_schema_validation_summary: %s", exc)
        return {"schema_ready": False, "unified_file": "config/settings.json"}


def get_schema_health_summary() -> dict:
    return {"module": "schema", "ready": _probe_schema_health()}


def get_schema_ready_flag() -> dict:
    return {"module": "schema", "ready": _probe_schema_health()}


def get_schema_version_info() -> dict:
    # S-0403: ready now reflects the real probe, not a constant True.
    return {"module": "schema", "version": __version__, "ready": _probe_schema_health()}


def get_schema_diagnostic_snapshot() -> dict:
    # S-0403: status/ready now reflect the real probe, not constants.
    ready = _probe_schema_health()
    return {"module": "schema", "status": "ok" if ready else "error", "ready": ready}


def get_schema_capabilities() -> list:
    return ["config_schema", "validation_rules", "diagnostic_snapshot"]


get_config_schema_capabilities = get_schema_capabilities


def validate_config_schema_contract() -> bool:
    return _probe_schema_health()


def get_config_schema_baseline_metrics() -> dict:
    try:
        ready = validate_config_schema_contract()
        return {
            "module": "config_schema",
            "contract_valid": ready,
            "schema_ready": ready,
        }
    except Exception:
        return {
            "module": "config_schema",
            "contract_valid": False,
            "schema_ready": False,
        }


def get_config_schema_readiness_status() -> dict:
    try:
        return {
            "module": "config_schema",
            "ready": validate_config_schema_contract(),
            "metrics": get_config_schema_baseline_metrics(),
        }
    except Exception:
        return {
            "module": "config_schema",
            "ready": False,
            "metrics": {},
        }


validate_schema_contract = validate_config_schema_contract
get_schema_readiness_status = get_config_schema_readiness_status


def get_schema_audit_record() -> dict:
    return {
        "module": "schema",
        "audit_passed": validate_schema_contract(),
        "readiness": get_schema_readiness_status(),
    }


def get_schema_operational_digest() -> dict:
    return {
        "module": "schema",
        "operational": validate_schema_contract(),
        "audit": get_schema_audit_record(),
    }


def get_schema_telemetry_payload() -> dict:
    return {
        "module": "schema",
        "payload": get_schema_operational_digest(),
    }


def get_schema_integrity_seal() -> dict:
    return {
        "module": "schema",
        "sealed": validate_schema_contract(),
        "telemetry": get_schema_telemetry_payload(),
    }


def get_schema_compliance_certificate() -> dict:
    return {
        "module": "schema",
        "certified": validate_schema_contract(),
        "seal": get_schema_integrity_seal(),
    }


def get_schema_readiness_probe() -> dict:
    return {
        "module": "schema",
        "ready": validate_schema_contract(),
        "certificate": get_schema_compliance_certificate(),
    }


def get_schema_audit_summary() -> dict:
    return {
        "module": "schema",
        "audit_passed": validate_schema_contract(),
        "probe": get_schema_readiness_probe(),
    }

