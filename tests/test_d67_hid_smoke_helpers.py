# -*- coding: utf-8 -*-
"""D67: HID smoke yardımcıları için testler (tools/hid_com_util,
tools/rootless_hid_bridge).

Kapsar:
- tap_keys: iki smoke betiğindeki tekrar eden key-tap döngüsünün tek
  paylaşılan yardımcısı (başarı / başarısız tuş / fırlatan adaptör /
  normalizasyon / hold alt sınırı).
- pick_best_port: Pico tek-aday seçiminin iki dalda tekrar eden
  mantığının tek yardımcıya indirgenmesinin davranış kilidi.
- LocalKeyInjector._inject: key_down/key_up tekrarının tek metoda
  indirgenmesinin davranış kilidi (sahte Interception köprüsüyle).
- _dispatch: UDP/TCP dinleyicilerindeki tekrar eden satır-dağıtım
  mantığının tek yardımcıya indirgenmesinin kilidi.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hid_com_util import ComPortInfo, pick_best_port, tap_keys
from tools.rootless_hid_bridge import LocalKeyInjector, _dispatch, parse_line


# ---------------------------------------------------------------- tap_keys

class FakeHid:
    """key_down/key_up çağrılarını kaydeden sahte adaptör."""

    def __init__(self, fail_down=(), fail_up=(), raise_down=()):
        self.calls: list = []
        self._fail_down = set(fail_down)
        self._fail_up = set(fail_up)
        self._raise_down = set(raise_down)

    def key_down(self, key):
        self.calls.append(("down", key))
        if key in self._raise_down:
            raise RuntimeError(f"boom down {key}")
        return key not in self._fail_down

    def key_up(self, key):
        self.calls.append(("up", key))
        return key not in self._fail_up


def test_tap_keys_success_records_down_up_order():
    hid = FakeHid()
    failed = tap_keys(hid, ["tab", "1"], hold_ms=20, gap_sec=0)
    assert failed == []
    assert hid.calls == [("down", "tab"), ("up", "tab"), ("down", "1"), ("up", "1")]


def test_tap_keys_normalizes_and_skips_blanks():
    hid = FakeHid()
    failed = tap_keys(hid, [" TAB ", "", "  ", "F"], hold_ms=20, gap_sec=0)
    assert failed == []
    assert [c[1] for c in hid.calls] == ["tab", "tab", "f", "f"]


def test_tap_keys_failed_key_reported_and_loop_continues():
    hid = FakeHid(fail_down={"2"})
    failed = tap_keys(hid, ["1", "2", "3"], hold_ms=20, gap_sec=0)
    assert failed == ["2"]
    assert [c[1] for c in hid.calls] == ["1", "1", "2", "2", "3", "3"]


def test_tap_keys_raising_adapter_does_not_abort_run():
    # Eski COM smoke döngüsü fırlatan adaptörde çökerdi; paylaşılan
    # yardımcı tuşu başarısız sayıp kalan tuşlara devam eder.
    hid = FakeHid(raise_down={"2"})
    failed = tap_keys(hid, ["1", "2", "3"], hold_ms=20, gap_sec=0)
    assert failed == ["2"]
    assert ("down", "3") in hid.calls


def test_tap_keys_hold_clamped_to_20ms():
    hid = FakeHid()
    start = time.monotonic()
    tap_keys(hid, ["tab", "1"], hold_ms=1, gap_sec=0)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04  # 2 tuş x 20 ms alt sınır


def test_tap_keys_empty_keys_returns_empty():
    hid = FakeHid()
    assert tap_keys(hid, [], hold_ms=20, gap_sec=0) == []
    assert hid.calls == []


# ---------------------------------------------------------- pick_best_port

def _ports(*specs):
    return [ComPortInfo(device=d, vid=v) for d, v in specs]


def test_pick_best_port_single_pico_among_many():
    ports = _ports(("COM3", 0x0403), ("COM5", 0x2E8A), ("COM6", 0x2341))
    assert pick_best_port(ports) == "COM5"


def test_pick_best_port_two_picos_is_ambiguous():
    ports = _ports(("COM5", 0x2E8A), ("COM7", 0x2E8A))
    assert pick_best_port(ports) == ""


def test_pick_best_port_single_candidate():
    assert pick_best_port(_ports(("COM5", 0x2E8A))) == "COM5"


def test_pick_best_port_no_candidate():
    assert pick_best_port(_ports(("COM3", 0x0403))) == ""


def test_pick_best_port_prefer_new_single_pico():
    ports = _ports(("COM5", 0x2E8A), ("COM7", 0x2E8A))
    assert pick_best_port(ports, prefer_new=["COM7"]) == "COM7"


def test_pick_best_port_prefer_new_two_picos_is_ambiguous():
    ports = _ports(("COM5", 0x2E8A), ("COM7", 0x2E8A))
    assert pick_best_port(ports, prefer_new=["COM5", "COM7"]) == ""


# ------------------------------------------------------- LocalKeyInjector

class FakeBridge:
    def __init__(self):
        self.available = True
        self.dll_path = "fake.dll"
        self.calls: list = []

    def send_key(self, scan, is_down, extended=False):
        self.calls.append((scan, is_down, extended))
        return True

    def close(self):
        pass


def _fake_deps(monkeypatch, bridge):
    import tools.rootless_hid_bridge as mod

    from core.input_driver import EXTENDED_KEYS, HARDWARE_SCANCODES

    monkeypatch.setattr(
        mod,
        "_load_injectors",
        lambda: {
            "ctypes": None,
            "scancodes": HARDWARE_SCANCODES,
            "extended": EXTENDED_KEYS,
            "InterceptionBridge": lambda: bridge,
            "KEYEVENTF_EXTENDEDKEY": 0,
            "KEYEVENTF_KEYUP": 0,
            "KEYEVENTF_SCANCODE": 0,
            "INPUT_KEYBOARD": 0,
            "InnerInputUnion": None,
            "InputStruct": None,
            "KeyBdInput": None,
        },
    )


def test_injector_down_up_use_single_inject_path(monkeypatch):
    bridge = FakeBridge()
    _fake_deps(monkeypatch, bridge)
    inj = LocalKeyInjector(allow_soft=False)
    assert inj.backend == "interception"
    assert inj.key_down("tab") is True
    assert inj.key_up("tab") is True
    assert bridge.calls == [(15, True, False), (15, False, False)]
    assert (inj.inject_ok, inj.inject_fail) == (2, 0)


def test_injector_extended_key_flagged(monkeypatch):
    bridge = FakeBridge()
    _fake_deps(monkeypatch, bridge)
    inj = LocalKeyInjector(allow_soft=False)
    assert inj.key_down("right") is True
    assert bridge.calls == [(205, True, True)]


def test_injector_unknown_key_never_reaches_bridge(monkeypatch):
    bridge = FakeBridge()
    _fake_deps(monkeypatch, bridge)
    inj = LocalKeyInjector(allow_soft=False)
    assert inj.key_down("no-such-key") is False
    assert inj.key_up("") is False
    assert bridge.calls == []
    assert inj.inject_fail == 2


def test_injector_key_pulse_retries_failed_release(monkeypatch):
    class FailingReleaseBridge(FakeBridge):
        def send_key(self, scan, is_down, extended=False):
            self.calls.append((scan, is_down, extended))
            return is_down  # down ok, up hep başarısız

    bridge = FailingReleaseBridge()
    _fake_deps(monkeypatch, bridge)
    inj = LocalKeyInjector(allow_soft=False)
    assert inj.key_pulse("tab", hold_ms=20) is False
    ups = [c for c in bridge.calls if c[1] is False]
    assert len(ups) == 3  # stuck-key guard: 3 deneme


def test_injector_no_backend_counts_failure():
    inj = LocalKeyInjector(allow_soft=False)
    assert inj.backend == "none"
    assert inj.key_down("tab") is False
    assert inj.inject_fail == 1


# ---------------------------------------------------------------- _dispatch

class RaisingInjector:
    backend = "interception"

    def key_down(self, name):
        raise RuntimeError("boom")


class OkInjector:
    backend = "interception"

    def __init__(self):
        self.calls: list = []

    def key_down(self, name):
        self.calls.append(("down", name))
        return True

    def key_up(self, name):
        self.calls.append(("up", name))
        return True

    def key_pulse(self, name, hold_ms=80):
        self.calls.append(("pulse", name, hold_ms))
        return True


def test_dispatch_ping_returns_pong():
    assert _dispatch(OkInjector(), "PING") == "PONG"


def test_dispatch_raising_injector_does_not_raise():
    assert _dispatch(RaisingInjector(), "KEYDOWN tab") is None


def test_dispatch_invalid_line_returns_err():
    assert _dispatch(OkInjector(), "KEY tab xx").startswith("ERR invalid line")


def test_dispatch_unknown_line_returns_none():
    assert _dispatch(OkInjector(), "HELLO") is None
    assert _dispatch(OkInjector(), "") is None


def test_parse_line_key_hold_bounds():
    assert parse_line("KEY tab 100") == ("KEY", ("tab", 100))
    assert parse_line("KEY tab 0")[0] == "INVALID"
    assert parse_line("KEY tab 60001")[0] == "INVALID"
    assert parse_line("KEY tab")[0] == "KEY"
