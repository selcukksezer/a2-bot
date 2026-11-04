import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

VALID_GATHER_KEYS = ("f", "e", "auto")


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

    def validate(self) -> "GatheringSettings":
        try:
            self.f_interval_min = max(0.1, min(30.0, float(self.f_interval_min)))
        except (TypeError, ValueError):
            self.f_interval_min = 1.5

        try:
            self.f_interval_max = max(self.f_interval_min, min(60.0, float(self.f_interval_max)))
        except (TypeError, ValueError):
            self.f_interval_max = max(self.f_interval_min, 3.0)

        self.test_mode = bool(self.test_mode)
        self.show_overlay = bool(self.show_overlay)

        try:
            self.harvests_per_node = max(1, min(10, int(self.harvests_per_node)))
        except (TypeError, ValueError):
            self.harvests_per_node = 3

        try:
            self.inter_harvest_delay_min = max(0.05, min(15.0, float(self.inter_harvest_delay_min)))
        except (TypeError, ValueError):
            self.inter_harvest_delay_min = 0.6

        try:
            self.inter_harvest_delay_max = max(
                self.inter_harvest_delay_min,
                min(30.0, float(self.inter_harvest_delay_max)),
            )
        except (TypeError, ValueError):
            self.inter_harvest_delay_max = max(self.inter_harvest_delay_min, 1.4)

        self.retry_on_extraction_fail = bool(self.retry_on_extraction_fail)

        try:
            self.max_retries_per_harvest_stage = max(0, min(10, int(self.max_retries_per_harvest_stage)))
        except (TypeError, ValueError):
            self.max_retries_per_harvest_stage = 2

        raw_key = str(self.gather_key if self.gather_key is not None else "f").strip().lower()
        self.gather_key = raw_key if raw_key in VALID_GATHER_KEYS else "f"
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatheringSettings":
        try:
            if not isinstance(data, dict):
                return cls().validate()
            instance = cls(
                f_interval_min=float(data.get("f_interval_min", 1.5)),
                f_interval_max=float(data.get("f_interval_max", 3.0)),
                test_mode=bool(data.get("test_mode", False)),
                show_overlay=bool(data.get("show_overlay", True)),
                harvests_per_node=int(data.get("harvests_per_node", 3)),
                inter_harvest_delay_min=float(data.get("inter_harvest_delay_min", 0.6)),
                inter_harvest_delay_max=float(data.get("inter_harvest_delay_max", 1.4)),
                retry_on_extraction_fail=bool(data.get("retry_on_extraction_fail", True)),
                max_retries_per_harvest_stage=int(data.get("max_retries_per_harvest_stage", 2)),
                gather_key=str(data.get("gather_key", "f")).strip().lower(),
            )
            return instance.validate()
        except Exception as exc:
            logger.error("Error parsing GatheringSettings from dict: %s", exc)
            return cls().validate()


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
    minimap_auto_filter_check: bool = True
    minimap_cluster_stack_bonus: bool = True
    hybrid_3d_and_minimap_record: bool = True
    auto_record_3d_crests: bool = True

    def validate(self) -> "RouteSettings":
        try:
            self.rest_minutes = max(0, min(240, int(self.rest_minutes)))
        except (TypeError, ValueError):
            self.rest_minutes = 5

        try:
            self.climb_air_max_sec = max(1, min(120, int(self.climb_air_max_sec)))
        except (TypeError, ValueError):
            self.climb_air_max_sec = 10

        self.gathered_join_active_route = bool(self.gathered_join_active_route)
        self.fly_unreachable = bool(self.fly_unreachable)
        self.check_odyle_minimap = bool(self.check_odyle_minimap)

        try:
            self.reclick_node_sec = max(1, min(120, int(self.reclick_node_sec)))
        except (TypeError, ValueError):
            self.reclick_node_sec = 10

        try:
            self.time_to_reach_sec = max(5, min(600, int(self.time_to_reach_sec)))
        except (TypeError, ValueError):
            self.time_to_reach_sec = 60

        self.gather_on_the_way = bool(self.gather_on_the_way)

        try:
            self.wait_min_after_failed = max(0, min(240, int(self.wait_min_after_failed)))
        except (TypeError, ValueError):
            self.wait_min_after_failed = 5

        try:
            self.max_failed_in_row = max(1, min(100, int(self.max_failed_in_row)))
        except (TypeError, ValueError):
            self.max_failed_in_row = 10

        try:
            self.remove_node_after_failures = max(0, min(100, int(self.remove_node_after_failures)))
        except (TypeError, ValueError):
            self.remove_node_after_failures = 0

        try:
            self.pause_after_no_gather_min = max(0, min(240, int(self.pause_after_no_gather_min)))
        except (TypeError, ValueError):
            self.pause_after_no_gather_min = 15

        self.fight_back = bool(self.fight_back)

        try:
            self.fight_stop_gather_sec = max(0, min(60, int(self.fight_stop_gather_sec)))
        except (TypeError, ValueError):
            self.fight_stop_gather_sec = 3

        try:
            self.harvests_per_node = max(1, min(10, int(self.harvests_per_node)))
        except (TypeError, ValueError):
            self.harvests_per_node = 3

        self.minimap_auto_record = bool(self.minimap_auto_record)
        self.minimap_ignore_monsters = bool(self.minimap_ignore_monsters)
        self.minimap_auto_filter_check = bool(self.minimap_auto_filter_check)
        self.minimap_cluster_stack_bonus = bool(self.minimap_cluster_stack_bonus)
        self.hybrid_3d_and_minimap_record = bool(self.hybrid_3d_and_minimap_record)
        self.auto_record_3d_crests = bool(self.auto_record_3d_crests)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteSettings":
        try:
            if not isinstance(data, dict):
                return cls().validate()
            instance = cls(
                rest_minutes=int(data.get("rest_minutes", 5)),
                climb_air_max_sec=int(data.get("climb_air_max_sec", 10)),
                gathered_join_active_route=bool(data.get("gathered_join_active_route", True)),
                fly_unreachable=bool(data.get("fly_unreachable", True)),
                check_odyle_minimap=bool(data.get("check_odyle_minimap", True)),
                reclick_node_sec=int(data.get("reclick_node_sec", 10)),
                time_to_reach_sec=int(data.get("time_to_reach_sec", 60)),
                gather_on_the_way=bool(data.get("gather_on_the_way", False)),
                wait_min_after_failed=int(data.get("wait_min_after_failed", 5)),
                max_failed_in_row=int(data.get("max_failed_in_row", 10)),
                remove_node_after_failures=int(data.get("remove_node_after_failures", 0)),
                pause_after_no_gather_min=int(data.get("pause_after_no_gather_min", 15)),
                fight_back=bool(data.get("fight_back", True)),
                fight_stop_gather_sec=int(data.get("fight_stop_gather_sec", 3)),
                harvests_per_node=int(data.get("harvests_per_node", 3)),
                minimap_auto_record=bool(data.get("minimap_auto_record", True)),
                minimap_ignore_monsters=bool(data.get("minimap_ignore_monsters", True)),
                minimap_auto_filter_check=bool(data.get("minimap_auto_filter_check", True)),
                minimap_cluster_stack_bonus=bool(data.get("minimap_cluster_stack_bonus", True)),
                hybrid_3d_and_minimap_record=bool(data.get("hybrid_3d_and_minimap_record", True)),
                auto_record_3d_crests=bool(data.get("auto_record_3d_crests", True)),
            )
            return instance.validate()
        except Exception as exc:
            logger.error("Error parsing RouteSettings from dict: %s", exc)
            return cls().validate()


@dataclass
class BreakSettings:
    break_every_min: int = 0
    break_every_variance: int = 1
    break_lasts_min: int = 10
    break_lasts_variance: int = 1

    def validate(self) -> "BreakSettings":
        try:
            self.break_every_min = max(0, min(1440, int(self.break_every_min)))
        except (TypeError, ValueError):
            self.break_every_min = 0

        try:
            self.break_every_variance = max(0, min(120, int(self.break_every_variance)))
        except (TypeError, ValueError):
            self.break_every_variance = 1

        try:
            self.break_lasts_min = max(1, min(240, int(self.break_lasts_min)))
        except (TypeError, ValueError):
            self.break_lasts_min = 10

        try:
            self.break_lasts_variance = max(0, min(120, int(self.break_lasts_variance)))
        except (TypeError, ValueError):
            self.break_lasts_variance = 1

        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BreakSettings":
        try:
            if not isinstance(data, dict):
                return cls().validate()
            instance = cls(
                break_every_min=int(data.get("break_every_min", 0)),
                break_every_variance=int(data.get("break_every_variance", 1)),
                break_lasts_min=int(data.get("break_lasts_min", 10)),
                break_lasts_variance=int(data.get("break_lasts_variance", 1)),
            )
            return instance.validate()
        except Exception as exc:
            logger.error("Error parsing BreakSettings from dict: %s", exc)
            return cls().validate()


@dataclass
class HotkeysSettings:
    toggle_menu_f9: str = "f9"
    record_node_f8: str = "f8"
    emergency_exit: str = "esc"

    def validate(self) -> "HotkeysSettings":
        self.toggle_menu_f9 = str(self.toggle_menu_f9 or "f9").strip().lower() or "f9"
        self.record_node_f8 = str(self.record_node_f8 or "f8").strip().lower() or "f8"
        self.emergency_exit = str(self.emergency_exit or "esc").strip().lower() or "esc"
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HotkeysSettings":
        if not isinstance(data, dict):
            return cls().validate()
        return cls(
            toggle_menu_f9=str(data.get("toggle_menu_f9", "f9")),
            record_node_f8=str(data.get("record_node_f8", "f8")),
            emergency_exit=str(data.get("emergency_exit", "esc")),
        ).validate()


@dataclass
class DisplaySettings:
    ui_proportion: str = "Medium"
    monitor_idx: int = 1

    def validate(self) -> "DisplaySettings":
        self.ui_proportion = str(self.ui_proportion or "Medium").strip() or "Medium"
        try:
            self.monitor_idx = max(1, min(8, int(self.monitor_idx)))
        except (TypeError, ValueError):
            self.monitor_idx = 1
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisplaySettings":
        if not isinstance(data, dict):
            return cls().validate()
        return cls(
            ui_proportion=str(data.get("ui_proportion", "Medium")),
            monitor_idx=int(data.get("monitor_idx", 1)),
        ).validate()


@dataclass
class BotSettings:
    gathering: GatheringSettings = field(default_factory=GatheringSettings)
    route: RouteSettings = field(default_factory=RouteSettings)
    breaks: BreakSettings = field(default_factory=BreakSettings)
    hotkeys: HotkeysSettings = field(default_factory=HotkeysSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    active_region: str = "Altgard"
    active_route: str = "Dredgion Crash Site"
    version: str = "v2.0"

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

        # Keep harvests_per_node synchronized between gathering and route
        if self.gathering.harvests_per_node != self.route.harvests_per_node:
            if self.route.harvests_per_node != 3 and self.gathering.harvests_per_node == 3:
                self.gathering.harvests_per_node = self.route.harvests_per_node
            else:
                self.route.harvests_per_node = self.gathering.harvests_per_node

        self.active_region = str(self.active_region or "Altgard").strip() or "Altgard"
        self.active_route = (
            str(self.active_route or "Dredgion Crash Site").strip() or "Dredgion Crash Site"
        )
        self.version = str(self.version or "v1.6.15").strip() or "v1.6.15"
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        try:
            return asdict(self)
        except Exception as exc:
            logger.error("Error converting BotSettings to dict: %s", exc)
            return {
                "gathering": asdict(self.gathering),
                "route": asdict(self.route),
                "breaks": asdict(self.breaks),
                "active_region": self.active_region,
                "active_route": self.active_route,
                "version": self.version,
            }

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

            instance = cls(
                gathering=GatheringSettings.from_dict(gathering_data),
                route=RouteSettings.from_dict(route_data),
                breaks=BreakSettings.from_dict(breaks_data),
                hotkeys=HotkeysSettings.from_dict(data.get("hotkeys", {})),
                display=DisplaySettings.from_dict(data.get("display", {})),
                active_region=str(data.get("active_region", "Altgard")),
                active_route=str(data.get("active_route", "Dredgion Crash Site")),
                version=str(data.get("version", "v2.0")),
            )
            return instance.validate()
        except Exception as exc:
            logger.error("Error parsing BotSettings from dict: %s", exc)
            return cls().validate()


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
            self.settings = BotSettings().validate()

    def load(self) -> BotSettings:
        try:
            if not self.settings_path.exists():
                logger.warning(
                    "Settings file %s does not exist during load(). Creating defaults.",
                    self.settings_path,
                )
                return self.reset_to_defaults()

            raw_text = self.settings_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            self.settings = BotSettings.from_dict(data)
            logger.info("Successfully loaded settings from %s", self.settings_path)
            return self.settings
        except Exception as exc:
            logger.error("Error loading settings from %s: %s", self.settings_path, exc)
            self.settings = BotSettings().validate()
            return self.settings

    def save(self, settings: Optional[BotSettings] = None) -> bool:
        try:
            if settings is not None:
                self.settings = settings
            self.settings.validate()
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.settings.to_dict()
            json_text = json.dumps(payload, indent=4, ensure_ascii=False)
            self.settings_path.write_text(json_text, encoding="utf-8")
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
            if (
                section in ("", "bot", "root", "general")
                or hasattr(self.settings, section)
                and not hasattr(getattr(self.settings, section), "__dataclass_fields__")
            ):
                target_attr = (
                    section
                    if hasattr(self.settings, section)
                    and section not in ("gathering", "route", "breaks")
                    else key
                )
                if hasattr(self.settings, target_attr):
                    setattr(self.settings, target_attr, value)
                    self.settings.validate()
                    return self.save(self.settings)

            section_obj = getattr(self.settings, section, None)
            if section_obj is not None and hasattr(section_obj, key):
                current_val = getattr(section_obj, key)
                if isinstance(current_val, bool):
                    cast_val = bool(value)
                elif isinstance(current_val, int) and not isinstance(current_val, bool):
                    cast_val = int(value)
                elif isinstance(current_val, float):
                    cast_val = float(value)
                elif isinstance(current_val, str):
                    cast_val = str(value)
                else:
                    cast_val = value
                setattr(section_obj, key, cast_val)
                if key == "harvests_per_node":
                    clamped_harvests = max(1, min(10, int(cast_val)))
                    self.settings.gathering.harvests_per_node = clamped_harvests
                    self.settings.route.harvests_per_node = clamped_harvests
                self.settings.validate()
                return self.save(self.settings)

            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                self.settings.validate()
                return self.save(self.settings)

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


def get_schema_validation_summary() -> dict:
    try:
        return {"schema_ready": True, "unified_file": "config/settings.json"}
    except Exception as exc:
        logger.error("Error in get_schema_validation_summary: %s", exc)
        return {"schema_ready": False, "unified_file": "config/settings.json"}
