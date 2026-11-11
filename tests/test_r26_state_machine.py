# -*- coding: utf-8 -*-
"""R26 (core/state_machine.py) düzeltme testleri.

- S-0102: _init_auto_cartographer / _init_explorer_controller hatası
  _optional_subsystem_errors'a kaydedilir (sağlık özetinde görünür).
- S-0103: savaş HUD'ı bilinmeyen HP / hazır skill için sahte "%0" / "1" yazmaz.
- F-0573: oto-iksir kapalıyken iksir atılmaz (worker özniteliği ve config).
- F-0122: mana_brain.decide() çıktısı apply_to_bus üzerinden tüketilir.
- F-0103: HumanTimingCoordinator.tick() anti-AFK adımına bağlıdır.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.human_timing import HumanTimingCoordinator, OrganicAntiAFK
from core.mana_phase_combat import ManaPhase, ManaPhaseCombat, ManaPhaseDecision
from core.state_machine import BotMode, BotState, BotWorker


class _RecordingDriver:
    """Recording input driver: real press/key_up/key_down surface, no hardware."""

    def __init__(self):
        self.pressed = []
        self.released = []
        self.holds = []
        self.clicks = []
        self.anti_afk_calls = 0
        self._mouse_buttons_down = set()
        self._active_bot_keys = set()

    def keyboard_inject_trusted(self) -> bool:
        return True

    def is_keyboard_trusted(self) -> bool:
        return True

    def press_key(self, key, **kwargs) -> bool:
        self.pressed.append(key)
        return True

    def send_key(self, key, **kwargs) -> bool:
        self.pressed.append(key)
        return True

    def key_down(self, key, **kwargs) -> bool:
        self.holds.append(key)
        self._active_bot_keys.add(str(key).lower())
        return True

    def key_up(self, key, **kwargs) -> bool:
        self.released.append(key)
        self._active_bot_keys.discard(str(key).lower())
        return True

    def release_key(self, key, **kwargs) -> bool:
        return self.key_up(key)

    def human_click(self, button="left", **kwargs) -> bool:
        self.clicks.append(button)
        return True

    def anti_afk_step(self) -> None:
        self.anti_afk_calls += 1

    def _hw_mouse_down(self, button) -> None:
        self._mouse_buttons_down.add(button)

    def _hw_mouse_up(self, button) -> None:
        self._mouse_buttons_down.discard(button)


def _create_lightweight_worker(
    mode: str = BotMode.COMBAT_MODE.value,
    state: str = BotState.FIGHTING.value,
    config=None,
) -> BotWorker:
    driver = _RecordingDriver()
    worker = BotWorker(
        config=config if config is not None else {"test_mode": False},
        vision=MagicMock(),
        input_driver=driver,
        route_manager=MagicMock(),
        combat_recovery=MagicMock(),
        navigator=None,
        recorder=None,
    )
    worker.driver = driver
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker._emit_log = MagicMock()
    return worker


class _RecordingBus:
    """Fake action bus: records cast_skill / hold calls, no hardware."""

    def __init__(self):
        self.cast = []
        self.left_holds = []
        self.basic_holds = []

    def cast_skill(self, key, frame=None):
        self.cast.append(str(key))
        return {"ok": True, "method": "key", "slot": str(key)}

    def press_key(self, key, **kwargs):
        self.cast.append(str(key))
        return True

    def hold_left_attack(self, on: bool) -> None:
        self.left_holds.append(bool(on))

    def hold_basic_attack(self, on: bool) -> None:
        self.basic_holds.append(bool(on))


# --- S-0102 -----------------------------------------------------------------

def test_init_auto_cartographer_failure_is_recorded(monkeypatch):
    sm = _create_lightweight_worker()

    def _boom(*args, **kwargs):
        raise RuntimeError("cartographer boom")

    monkeypatch.setattr("core.auto_cartographer.AutoCartographer", _boom)
    assert sm._init_auto_cartographer() is None
    errors = sm.__dict__.get("_optional_subsystem_errors", {})
    assert errors.get("core.auto_cartographer.AutoCartographer") == "cartographer boom"
    assert "cartographer boom" in sm.get_state_machine_health_summary()["optional_subsystem_errors"].values()


def test_init_explorer_controller_failure_is_recorded(monkeypatch):
    sm = _create_lightweight_worker()

    def _boom(*args, **kwargs):
        raise RuntimeError("explorer boom")

    monkeypatch.setattr("core.explorer_mode.ExplorerModeController", _boom)
    assert sm._init_explorer_controller() is None
    errors = sm.__dict__.get("_optional_subsystem_errors", {})
    assert errors.get("core.explorer_mode.ExplorerModeController") == "explorer boom"
    assert "explorer boom" in sm.get_state_machine_health_summary()["optional_subsystem_errors"].values()


# --- S-0103 -----------------------------------------------------------------

def test_combat_hud_without_cognition_shows_no_fake_values():
    sm = _create_lightweight_worker()
    sm.game_learner = None
    payload = sm._emit_hud_snapshot()
    assert "Hazır Skiller: 1" not in payload["sub"]
    assert "Canavar Canı: %0" not in payload["sub"]
    assert "Canavar Canı: ?" in payload["sub"]
    assert "Hazır Skiller: —" in payload["sub"]
    assert payload["col2_val"] == "HEDEF YOK"
    assert payload["col3_val"] == "—"


def test_combat_hud_with_cognition_shows_real_values():
    sm = _create_lightweight_worker()
    sm.game_learner = SimpleNamespace(
        total_learned_frames=5,
        get_live_cognition_snapshot=lambda: {
            "monster_hp_percent": 42,
            "ready_skills": ["2", "3"],
            "total_learned_frames": 5,
            "monster_targeted": True,
        }
    )
    payload = sm._emit_hud_snapshot()
    assert "Canavar Canı: %42" in payload["sub"]
    assert "Hazır Skiller: 2, 3" in payload["sub"]
    assert payload["col2_val"] == "%42"
    assert payload["col3_val"] == "2,3"


# --- F-0573 -----------------------------------------------------------------

def test_auto_potions_skipped_when_disabled_via_worker_flag():
    sm = _create_lightweight_worker()
    sm.__dict__["_last_hp_potion_ts"] = 0.0
    sm.__dict__["_last_mp_potion_ts"] = 0.0
    sm.mana_phase_combat = ManaPhaseCombat(hp_potion_key="F1", hp_potion_cd_sec=5.0)
    sm.set_auto_consumables_enabled(False)
    assert sm._is_auto_consumables_enabled() is False
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.50, player_mp_ratio=0.90)
    assert triggered is False
    assert sm.input_driver.pressed == []


def test_auto_potions_skipped_when_disabled_via_config():
    sm = _create_lightweight_worker(
        config={"test_mode": False, "gathering": {"auto_consumables": False}}
    )
    sm.__dict__["_last_hp_potion_ts"] = 0.0
    sm.__dict__["_last_mp_potion_ts"] = 0.0
    sm.mana_phase_combat = ManaPhaseCombat(hp_potion_key="F1", hp_potion_cd_sec=5.0)
    assert sm._is_auto_consumables_enabled() is False
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.50, player_mp_ratio=0.90)
    assert triggered is False
    assert sm.input_driver.pressed == []


def test_auto_potions_still_trigger_when_enabled():
    sm = _create_lightweight_worker()
    sm.__dict__["_last_hp_potion_ts"] = 0.0
    sm.__dict__["_last_mp_potion_ts"] = 0.0
    sm.mana_phase_combat = ManaPhaseCombat(hp_potion_key="F1", hp_potion_cd_sec=5.0)
    assert sm._is_auto_consumables_enabled() is True
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.50, player_mp_ratio=0.90)
    assert triggered is True
    assert "F1" in sm.input_driver.pressed


# --- F-0122 -----------------------------------------------------------------

def _decision(**overrides):
    base = {
        "phase": ManaPhase.BURST,
        "hold_left": False,
        "hold_right": True,
        "skill_keys": ["2", "3"],
        "potion_keys": ["F1"],
    }
    base.update(overrides)
    return ManaPhaseDecision(**base)


def test_mana_decision_consumed_via_bus():
    sm = _create_lightweight_worker()
    sm.mana_phase_combat = ManaPhaseCombat()
    bus = _RecordingBus()
    sm.action_bus = bus
    assert sm._consume_mana_decision_via_bus(_decision(), bus) is True
    assert bus.cast == ["F1", "2", "3"] or set(["2", "3"]).issubset(set(bus.cast))
    assert "2" in bus.cast and "3" in bus.cast


def test_mana_decision_potion_keys_stripped_when_consumables_disabled():
    sm = _create_lightweight_worker()
    sm.mana_phase_combat = ManaPhaseCombat()
    sm.set_auto_consumables_enabled(False)
    bus = _RecordingBus()
    sm.action_bus = bus
    assert sm._consume_mana_decision_via_bus(_decision(), bus) is True
    assert "F1" not in bus.cast
    assert "2" in bus.cast and "3" in bus.cast


def test_mana_decision_falls_back_without_bus_or_skills():
    sm = _create_lightweight_worker()
    sm.mana_phase_combat = ManaPhaseCombat()
    assert sm._consume_mana_decision_via_bus(_decision(), None) is False
    assert sm._consume_mana_decision_via_bus(_decision(skill_keys=[]), _RecordingBus()) is False
    assert sm._consume_mana_decision_via_bus(None, _RecordingBus()) is False


# --- F-0103 -----------------------------------------------------------------

def _fake_coordinator(result):
    return SimpleNamespace(tick=lambda now=None, input_ctrl=None: result)


def test_anti_afk_step_uses_coordinator_waiting_cadence():
    sm = _create_lightweight_worker()
    sm.__dict__["_human_timing_coordinator"] = _fake_coordinator(
        {"macro_break": False, "micro_pause": False,
         "anti_afk": {"action": "waiting", "success": True, "executed": False,
                      "next_interval_sec": 120.0}}
    )
    sm._perform_anti_afk_step()
    assert sm.input_driver.anti_afk_calls == 0


def test_anti_afk_step_uses_coordinator_executed_action():
    sm = _create_lightweight_worker()
    sm.__dict__["_human_timing_coordinator"] = _fake_coordinator(
        {"macro_break": False, "micro_pause": False,
         "anti_afk": {"action": "camera_windmouse_pan", "success": True, "executed": True}}
    )
    sm._perform_anti_afk_step()
    assert sm.input_driver.anti_afk_calls == 0


def test_anti_afk_step_falls_back_to_legacy_when_coordinator_pending():
    sm = _create_lightweight_worker()
    sm.__dict__["_human_timing_coordinator"] = _fake_coordinator(
        {"macro_break": False, "micro_pause": False,
         "anti_afk": {"action": "pending", "success": False, "executed": False,
                      "error": "no input_ctrl supplied to tick()"}}
    )
    sm._perform_anti_afk_step()
    assert sm.input_driver.anti_afk_calls == 1


def test_anti_afk_step_falls_back_when_no_coordinator():
    sm = _create_lightweight_worker()
    sm._get_human_timing_coordinator = lambda: None
    sm._perform_anti_afk_step()
    assert sm.input_driver.anti_afk_calls == 1


def test_human_timing_coordinator_is_built_once_and_reuses_subsystems():
    sm = _create_lightweight_worker()
    coord = sm._get_human_timing_coordinator()
    assert isinstance(coord, HumanTimingCoordinator)
    assert sm._get_human_timing_coordinator() is coord
    if sm.organic_anti_afk is not None:
        assert coord.anti_afk is sm.organic_anti_afk
    else:
        assert isinstance(coord.anti_afk, OrganicAntiAFK)
