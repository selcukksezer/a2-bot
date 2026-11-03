# -*- coding: utf-8 -*-
"""
Global Configuration & Settings Manager
Persists bot configuration, hotkeys, combat settings, and UI preferences.
"""

import os
import json

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "gathering": {
        "min_f_interval": 1.5,
        "max_f_interval": 3.0,
        "test_mode": False,
        "show_overlay": True
    },
    "route": {
        "rest_after_visit_min": 5,
        "gathered_nodes_join_route": True,
        "time_to_reach_node_sec": 60,
        "reclick_interval_sec": 10,
        "exploration_gather": True,
        "max_failures_before_wait": 10,
        "pause_after_idle_min": 15,
        "fight_back_when_attacked": True
    },
    "breaks": {
        "enabled": False,
        "interval_min": 60,
        "interval_jitter_min": 10,
        "duration_min": 10,
        "duration_jitter_min": 2
    },
    "hotkeys": {
        "toggle_menu_f9": "f9",
        "record_node_f8": "f8",
        "emergency_exit": "esc"
    },
    "display": {
        "ui_proportion": "Medium",
        "monitor_idx": 1
    }
}

class ConfigManager:
    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.config = self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # merge with defaults for any missing keys
                    cfg = DEFAULT_CONFIG.copy()
                    for k, v in data.items():
                        if isinstance(v, dict) and k in cfg:
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
                    return cfg
            except Exception as e:
                print(f"[ConfigManager] Ayar okuma hatası: {e}")
        return DEFAULT_CONFIG.copy()

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save()
