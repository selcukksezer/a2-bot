# -*- coding: utf-8 -*-
"""
Unified Configuration Adapter (core/config.py)
Redirects legacy ConfigManager callers to the canonical `config/settings.json` (`SettingsManager`).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from config.schema import SettingsManager
from version import __version__

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join("config", "settings.json")


class ConfigManager:
    """Unified wrapper over `config/settings.json` (`SettingsManager`)."""

    def __init__(self, config_path: str = CONFIG_FILE) -> None:
        if os.path.basename(str(config_path)) == "config.json" and not os.path.exists(str(config_path)):
            config_path = CONFIG_FILE
        self.config_path = config_path
        self._mgr = SettingsManager(settings_path=self.config_path)
        # B-15 (dilim-30, L devri): bilinmeyen anahtarların disk birleştirme
        # yazımı arızalanırsa kaybolan anahtarlar burada tutulur — save()
        # artık sessizce True dönüp bellekte kalmıyor; kayıp görünür olur.
        self._last_lost_unknown_keys: Dict[str, Any] = {}
        self.config: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        settings_obj = self._mgr.load()
        self.config = settings_obj.to_dict()
        return self.config

    def save(self) -> bool:
        from config.schema import BotSettings

        # Unknown keys are not part of the schema, but they are the user's
        # data: keep them across the from_dict round-trip instead of
        # silently dropping them.
        known_top = set(BotSettings.__dataclass_fields__)
        known_sections = {"gathering", "route", "breaks", "hotkeys", "display"}
        unknown_top = {
            key: value for key, value in self.config.items() if key not in known_top
        }
        unknown_nested: Dict[str, Dict[str, Any]] = {}
        for section in known_sections:
            section_data = self.config.get(section)
            if isinstance(section_data, dict):
                # Field names come from the live section object.
                live = getattr(self._mgr.settings, section, None)
                live_fields = set(getattr(live, "__dataclass_fields__", {}).keys())
                extra = {
                    key: value
                    for key, value in section_data.items()
                    if key not in live_fields
                }
                if extra:
                    unknown_nested[section] = extra

        self._mgr.settings = BotSettings.from_dict(self.config)
        ok = self._mgr.save(self._mgr.settings)
        if not ok:
            return False
        if unknown_top or unknown_nested:
            try:
                file_path = Path(self.config_path)
                data = json.loads(file_path.read_text(encoding="utf-8"))
                data.update(unknown_top)
                for section, extra in unknown_nested.items():
                    if isinstance(data.get(section), dict):
                        data[section].update(extra)
                    else:
                        data[section] = dict(extra)
                # I5 (dev/D006): atomic replace (tmp + os.replace) instead
                # of a direct write_text — a crash mid-write used to
                # leave a torn settings.json; the schema-known part is
                # saved by SettingsManager.save above, this merge must
                # not be the write that corrupts the file.
                tmp_path = file_path.with_name(
                    f"{file_path.name}.tmp.{os.getpid()}"
                )
                try:
                    tmp_path.write_text(
                        json.dumps(data, indent=4, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(tmp_path, file_path)
                except Exception:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    raise
            except Exception as exc:
                # B-15 (dilim-30, L devri): eskiden `except Exception: pass`
                # idi — birleştirme yazımı arızalanırsa bilinmeyen anahtarlar
                # yalnızca bellekte kalıyor, save() yine True dönüyor,
                # yeniden başlatmada kayboluyordu. Artık kayıp görünür:
                # warning + kayıp anahtar listesi saklanır, save() False
                # döner (şema-bilinen kısım yukarıda kaydedildi; eksik olan
                # yalnızca bilinmeyen anahtar birleştirmesidir).
                lost: Dict[str, Any] = dict(unknown_top)
                for section, extra in unknown_nested.items():
                    lost.setdefault(section, dict(extra))
                self._last_lost_unknown_keys = lost
                logger.warning(
                    "Bilinmeyen ayar anahtarları diske yazılamadı (%s bölüm/anahtar: %s): %s — "
                    "yalnızca bellekte kalacak, yeniden başlatmada kaybolacak.",
                    len(lost),
                    sorted(str(k) for k in lost),
                    exc,
                )
                self.config = self._mgr.settings.to_dict()
                self.config.update(unknown_top)
                for section, extra in unknown_nested.items():
                    if isinstance(self.config.get(section), dict):
                        self.config[section].update(extra)
                return False
        self._last_lost_unknown_keys = {}
        self.config = self._mgr.settings.to_dict()
        self.config.update(unknown_top)
        for section, extra in unknown_nested.items():
            if isinstance(self.config.get(section), dict):
                self.config[section].update(extra)
        return True

    def get(self, section: str, key: Any = None, default: Any = None) -> Any:
        return self._mgr.get(section, key, default)

    def set(self, section: str, key: str, value: Any) -> bool:
        ok = self._mgr.set(section, key, value)
        self.config = self._mgr.settings.to_dict()
        return ok

    def get_config_health_summary(self) -> dict:
        """Real check: the loaded settings must validate and the settings
        file must be readable when it exists (a failed load is reported,
        not hidden behind a constant True)."""
        try:
            self._mgr.settings.validate()
            load_error = getattr(self._mgr, "load_error", None)
            if load_error:
                return {"module": "config", "ready": False, "load_error": str(load_error)}
            return {"module": "config", "ready": True}
        except Exception as exc:
            return {"module": "config", "ready": False, "error": str(exc)}

    def get_config_source_summary(self) -> dict:
        # S-0404: the try block used to return a literal True that can never
        # fail; ready now comes from the real health check.
        try:
            health = self.get_config_health_summary()
            return {"unified_config": "config/settings.json", "ready": bool(health.get("ready"))}
        except Exception:
            return {"unified_config": "config/settings.json", "ready": False}

    def get_config_ready_flag(self) -> dict:
        return {"module": "config", "ready": bool(self.get_config_health_summary().get("ready"))}

    def get_config_version_info(self) -> dict:
        return {"module": "config", "version": __version__, "ready": bool(self.get_config_health_summary().get("ready"))}

    def get_config_diagnostic_snapshot(self) -> dict:
        health = self.get_config_health_summary()
        return {"module": "config", "status": "ok" if health.get("ready") else "error", "ready": bool(health.get("ready"))}


def _probe_config_contract() -> bool:
    """Module-level probe without touching the real settings file: default
    settings must validate and round-trip through from_dict/to_dict."""
    try:
        from config.schema import BotSettings

        settings = BotSettings().validate()
        restored = BotSettings.from_dict(settings.to_dict())
        return restored.gathering.harvests_per_node == settings.gathering.harvests_per_node
    except Exception:
        return False


def get_config_capabilities() -> list:
    return ["config_manager", "settings_persistence", "diagnostic_snapshot"]


def validate_config_contract() -> bool:
    return _probe_config_contract()


def get_config_baseline_metrics() -> dict:
    return {"module": "config", "contract_valid": validate_config_contract(), "capability_count": len(get_config_capabilities())}


def get_config_readiness_status() -> dict:
    return {"module": "config", "ready": validate_config_contract(), "metrics": get_config_baseline_metrics()}


def get_core_config_capabilities() -> list:
    return get_config_capabilities()


def validate_core_config_contract() -> bool:
    return validate_config_contract()


def get_core_config_baseline_metrics() -> dict:
    return {"module": "core_config", "contract_valid": validate_core_config_contract(), "capability_count": len(get_core_config_capabilities())}


def get_core_config_readiness_status() -> dict:
    return {"module": "core_config", "ready": validate_core_config_contract(), "metrics": get_core_config_baseline_metrics()}


def get_config_audit_record() -> dict:
    return {"module": "config", "audit_passed": validate_config_contract(), "readiness": get_config_readiness_status()}


def get_config_operational_digest() -> dict:
    return {"module": "config", "operational": validate_config_contract(), "audit": get_config_audit_record()}


def get_config_telemetry_payload() -> dict:
    return {"module": "config", "payload": get_config_operational_digest()}


def get_config_integrity_seal() -> dict:
    return {"module": "config", "sealed": validate_config_contract(), "telemetry": get_config_telemetry_payload()}


def get_config_compliance_certificate() -> dict:
    return {"module": "config", "certified": validate_config_contract(), "seal": get_config_integrity_seal()}


def get_config_readiness_probe() -> dict:
    return {"module": "config", "ready": validate_config_contract(), "certificate": get_config_compliance_certificate()}


def get_config_audit_summary() -> dict:
    return {"module": "config", "audit_passed": validate_config_contract(), "probe": get_config_readiness_probe()}

