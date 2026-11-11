# -*- coding: utf-8 -*-
"""R15: GameActionBus click-result honesty (F-1868) + single trust probe (F-1880).

F-1868: the real driver's human_click/click_map return (applied_x, applied_y,
hold_sec) on EVERY path — abort and exception included — so _click_px must
validate the tuple instead of counting any non-False value as success.
F-1880: cast_skill probes keyboard trust once and passes it through to
_key_fallback instead of re-running ensure_serial_hid + the trusted check.
"""

import pytest

from core.game_action_bus import GameActionBus


class _ClickDriver:
    """Fake driver returning a configurable human_click value."""

    def __init__(self, clicked, aborted=False):
        self._clicked = clicked
        self._aborted = aborted

    def focus_game_window(self):
        return None

    def human_click(self, x, y, button="left"):
        return self._clicked

    def motion_aborted(self):
        return self._aborted

    def _hw_mouse_up(self, button="left"):
        return True


def _bus_with_click(clicked, aborted=False):
    return GameActionBus(input_driver=_ClickDriver(clicked, aborted))


@pytest.mark.parametrize(
    "clicked,aborted,expected",
    [
        (True, False, True),  # bool-returning driver/port
        (False, False, False),
        (None, False, False),  # no signal is NOT success (F-1868)
        ((101, 99, 0.05), False, True),  # jittered target, positive hold
        ([101, 99, 0.05], False, True),
        ((500, 500, 0.05), False, False),  # exception path: real cursor pos
        ((100, 100, 0.0), False, False),  # zero hold: nothing was pressed
        ((100, 100), False, False),  # malformed tuple
        (("ok",), False, False),
        (0.05, False, False),  # bare number carries no click signal
        ((101, 99, 0.05), True, False),  # aborted mid-way despite good tuple
    ],
)
def test_click_px_validates_driver_return(clicked, aborted, expected):
    bus = _bus_with_click(clicked, aborted)
    assert bus._click_px(100, 100) is expected


class _KeyDriver:
    """Fake driver for the trusted hardware key path."""

    def __init__(self):
        self.ensure_calls = 0
        self.trusted_calls = 0
        self.down_calls = []
        self.up_calls = []

    def ensure_serial_hid(self, force=False):
        self.ensure_calls += 1

    def keyboard_inject_trusted(self):
        self.trusted_calls += 1
        return True

    def release_modifier_keys(self):
        return None

    def focus_game_window(self):
        return None

    def _hw_key_down(self, key, extended=False):
        self.down_calls.append(key)
        return True

    def _hw_key_up(self, key, extended=False):
        self.up_calls.append(key)
        return True

    def press_key(self, key, **kwargs):
        return 0.1

    def _hw_mouse_up(self, button="left"):
        return True


def test_key_fallback_with_passed_trust_does_not_reprobe():
    """F-1880: trusted=... given -> _keyboard_trusted() is not called again."""
    driver = _KeyDriver()
    bus = GameActionBus(input_driver=driver, mouse_look_mode=True)
    calls = {"n": 0}
    real_probe = bus._keyboard_trusted

    def counting_probe():
        calls["n"] += 1
        return real_probe()

    bus._keyboard_trusted = counting_probe
    assert bus._key_fallback("2", trusted=True) is True
    assert calls["n"] == 0
    assert driver.ensure_calls == 0
    assert driver.trusted_calls == 0


def test_key_fallback_without_trust_probes_once():
    """Default path still probes exactly once (single source of truth)."""
    driver = _KeyDriver()
    bus = GameActionBus(input_driver=driver, mouse_look_mode=True)
    assert bus._key_fallback("2") is True
    assert driver.ensure_calls == 1
    assert driver.trusted_calls == 1


def test_cast_skill_probes_trust_once_for_whole_cast():
    """F-1880: one cast_skill -> one trust probe, not one per _key_fallback."""
    driver = _KeyDriver()
    bus = GameActionBus(input_driver=driver, mouse_look_mode=True)
    calls = {"n": 0}
    real_probe = bus._keyboard_trusted

    def counting_probe():
        calls["n"] += 1
        return real_probe()

    bus._keyboard_trusted = counting_probe
    res = bus.cast_skill("2")
    assert res["ok"] is True
    assert res["method"] == "key"
    assert calls["n"] == 1
    assert driver.down_calls == ["2"]
    assert driver.up_calls == ["2"]
