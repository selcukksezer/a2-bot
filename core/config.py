# -*- coding: utf-8 -*-
"""
Unified Configuration Adapter (core/config.py)
Redirects legacy ConfigManager callers to the canonical `config/settings.json` (`SettingsManager`).
"""

import json
import os
from typing import Any, Dict

from config.schema import SettingsManager

CONFIG_FILE = os.path.join("config", "settings.json")


class ConfigManager:
    """Unified wrapper over `config/settings.json` (`SettingsManager`)."""

    def __init__(self, config_path: str = CONFIG_FILE) -> None:
        if os.path.basename(str(config_path)) == "config.json" and not os.path.exists(str(config_path)):
            config_path = CONFIG_FILE
        self.config_path = config_path
        self._mgr = SettingsManager(settings_path=self.config_path)
        self.config: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        settings_obj = self._mgr.load()
        self.config = settings_obj.to_dict()
        return self.config

    def save(self) -> bool:
        from config.schema import BotSettings

        self._mgr.settings = BotSettings.from_dict(self.config)
        return self._mgr.save(self._mgr.settings)

    def get(self, section: str, key: Any = None, default: Any = None) -> Any:
        return self._mgr.get(section, key, default)

    def set(self, section: str, key: str, value: Any) -> bool:
        ok = self._mgr.set(section, key, value)
        self.config = self._mgr.settings.to_dict()
        return ok

    def get_config_health_summary(self) -> dict:
        return {"module": "config", "ready": True}

    def get_config_source_summary(self) -> dict:
        try:
            return {"unified_config": "config/settings.json", "ready": True}
        except Exception:
            return {"unified_config": "config/settings.json", "ready": False}

