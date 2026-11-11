# -*- coding: utf-8 -*-
"""R68: route_runner ready-flag / contract probes + hotkey unregister (S-2702, S-2705, F-1097).

Linux CI has no `keyboard`/`winsound`, so both are stubbed in sys.modules
before importing modules.route_runner (same pattern as test_settings_window.py).
RouteRunner collaborators are faked via monkeypatch — no screen capture, no
input injection, no disk route loading beyond the fake.
"""

import sys
import types

import pytest

# --- Stub Windows-only / missing modules before import -----------------------
_winsound_stub = types.ModuleType("winsound")
_winsound_stub.Beep = lambda *args, **kwargs: None
sys.modules.setdefault("winsound", _winsound_stub)

_keyboard_calls = {"add": [], "remove": [], "unhook_all": 0}
_keyboard_fail_on_remove = set()


def _kb_add_hotkey(name, callback, suppress=False):
    _keyboard_calls["add"].append(name)


def _kb_remove_hotkey(name):
    _keyboard_calls["remove"].append(name)
    if name in _keyboard_fail_on_remove:
        raise RuntimeError(f"cannot remove {name}")


def _kb_unhook_all():
    _keyboard_calls["unhook_all"] += 1


_keyboard_stub = types.ModuleType("keyboard")
_keyboard_stub.add_hotkey = _kb_add_hotkey
_keyboard_stub.remove_hotkey = _kb_remove_hotkey
_keyboard_stub.unhook_all = _kb_unhook_all
_keyboard_stub.is_pressed = lambda *args, **kwargs: False
sys.modules.setdefault("keyboard", _keyboard_stub)

import modules.route_runner as rr  # noqa: E402

# R68 stub bağlama notu (v2 tam-süit koşusunda kırılma nedeni): `keyboard`
# gerçek paketi Linux VM'de yok; `test_auto_gather_ready_flag_r69.py` ve
# v2-yeni `test_d78_dev_improvements.py` kendi no-op `keyboard` stub'larını
# `sys.modules`'a R68'den önce kuruyor (koleksiyon sırası) ve
# `modules.route_runner`'ı ilk onlar import ediyor. R68'in yukarıdaki
# `setdefault`'u o durumda hiç etki etmez; `rr.keyboard` onların no-op
# stub'una bağlı kalır ve `_unregister_hotkeys` çağrıları bu dosyanın
# `_keyboard_calls` kaydına hiç düşmez (`remove == []`). Üretim kodu
# (`_unregister_hotkeys`, modules/route_runner.py:95-105) v1'den beri
# değişmedi ve doğru: her ismi tek tek `remove_hotkey` ile kaldırır,
# `unhook_all` hiç çağırmaz, listeyi temizler. Bu yüzden test edilen
# modülün `keyboard` globali, kaydı tutan bu stub'a açıkça bağlanır —
# beklenti/assertion değişikliği YOK, yalnızca gözlem bağlaması.
rr.keyboard = _keyboard_stub


# --- Fakes -------------------------------------------------------------------
class FakeRoutes:
    def __init__(self, regions_ok=True):
        self._regions_ok = regions_ok
        self.active_region = "Heiron"
        self.base_dir = "routes"

    def load_all_regions(self):
        if self._regions_ok:
            return {"regions": {"Heiron": {"nodes": []}}, "active_region": "Heiron"}
        return {"regions": {}, "active_region": "Heiron"}

    def get_next_ready_node(self, *args, **kwargs):
        return None

    def get_route_nodes(self, *args, **kwargs):
        return []


class FakeVision:
    def __init__(self, ready=True):
        self._ready = ready

    def _is_engine_ready(self):
        return self._ready

    def check_gather_prompt(self, *args, **kwargs):
        return (False, 0.0, None)

    def is_map_open(self, *args, **kwargs):
        return False


class FakeDriver:
    def __init__(self, trusted=True):
        self.keyboard_inject_trusted = trusted
        self.active_backend_name = "Interception" if trusted else "ctypes.SendInput"

    def send_gather_key(self, *args, **kwargs):
        return 0.0


class FakeNavigator:
    def __init__(self, *args, **kwargs):
        pass


class FakeRecorder:
    def __init__(self, *args, **kwargs):
        pass


def _make_runner(monkeypatch, regions_ok=True, vision_ok=True, driver_ok=True,
                 settings=None):
    monkeypatch.setattr(rr, "RouteManager", lambda *a, **k: FakeRoutes(regions_ok))
    monkeypatch.setattr(rr, "Vision", lambda *a, **k: FakeVision(vision_ok))
    monkeypatch.setattr(rr, "InputDriver", lambda *a, **k: FakeDriver(driver_ok))
    monkeypatch.setattr(rr, "MapNavigator", FakeNavigator)
    monkeypatch.setattr(rr, "NodeRecorder", FakeRecorder)
    if settings is not None:
        monkeypatch.setattr(rr, "_load_route_settings", lambda: settings)
    return rr.RouteRunner()


@pytest.fixture(autouse=True)
def _reset_keyboard_calls():
    _keyboard_calls["add"].clear()
    _keyboard_calls["remove"].clear()
    _keyboard_calls["unhook_all"] = 0
    _keyboard_fail_on_remove.clear()
    yield
    _keyboard_fail_on_remove.clear()


# --- F-1097: settings are read (second half of the finding) ------------------
def test_settings_rest_and_walk_timeout_read(monkeypatch):
    runner = _make_runner(monkeypatch, settings={"rest_minutes": 7, "time_to_reach_sec": 45})
    assert runner.rest_duration_sec == pytest.approx(7 * 60.0)
    assert runner.walk_timeout_sec == pytest.approx(45.0)


def test_explicit_args_win_over_settings(monkeypatch):
    runner = _make_runner(
        monkeypatch, settings={"rest_minutes": 7, "time_to_reach_sec": 45}
    )
    explicit = rr.RouteRunner(rest_duration_sec=123, walk_timeout_sec=9)
    assert explicit.rest_duration_sec == 123
    assert explicit.walk_timeout_sec == 9
    assert runner.rest_duration_sec == pytest.approx(420.0)


# --- S-2702: instance ready flag probes real component state -----------------
def test_ready_flag_true_when_all_components_ok(monkeypatch):
    runner = _make_runner(monkeypatch)
    flag = runner.get_route_runner_ready_flag()
    assert flag["module"] == "route_runner"
    assert flag["ready"] is True
    assert flag["routes_ok"] is True
    assert flag["vision_ok"] is True
    assert flag["driver_ok"] is True
    assert flag["driver_backend"] == "Interception"
    assert flag["routes"]["regions_loaded"] == 1


def test_ready_flag_false_when_no_regions_loaded(monkeypatch):
    runner = _make_runner(monkeypatch, regions_ok=False)
    flag = runner.get_route_runner_ready_flag()
    assert flag["ready"] is False
    assert flag["routes_ok"] is False
    assert flag["vision_ok"] is True  # other components still reported
    assert flag["driver_ok"] is True


def test_ready_flag_false_when_vision_not_ready(monkeypatch):
    runner = _make_runner(monkeypatch, vision_ok=False)
    flag = runner.get_route_runner_ready_flag()
    assert flag["ready"] is False
    assert flag["vision_ok"] is False
    assert flag["routes_ok"] is True


def test_ready_flag_false_when_driver_untrusted(monkeypatch):
    runner = _make_runner(monkeypatch, driver_ok=False)
    flag = runner.get_route_runner_ready_flag()
    assert flag["ready"] is False
    assert flag["driver_ok"] is False
    assert flag["driver_backend"] == "ctypes.SendInput"


def test_version_info_and_diagnostic_snapshot_follow_probe(monkeypatch):
    ok_runner = _make_runner(monkeypatch)
    assert ok_runner.get_route_runner_version_info()["ready"] is True
    assert ok_runner.get_route_runner_diagnostic_snapshot()["status"] == "ok"

    bad_runner = _make_runner(monkeypatch, driver_ok=False)
    assert bad_runner.get_route_runner_version_info()["ready"] is False
    snap = bad_runner.get_route_runner_diagnostic_snapshot()
    assert snap["status"] == "degraded"
    assert snap["ready"] is False
    assert snap["driver_ok"] is False


def test_ready_flag_survives_probe_exception(monkeypatch):
    runner = _make_runner(monkeypatch)

    def _boom():
        raise RuntimeError("disk gone")

    runner.routes.load_all_regions = _boom
    flag = runner.get_route_runner_ready_flag()
    assert flag["ready"] is False
    assert flag["routes_ok"] is False


# --- S-2702: module-level contract is a real check, not a tautology ---------
def test_validate_contract_true_with_real_classes():
    assert rr.validate_route_runner_contract() is True
    metrics = rr.get_route_runner_baseline_metrics()
    assert metrics["contract_valid"] is True
    assert metrics["runner_ready"] is True


def test_validate_contract_false_when_vision_method_missing(monkeypatch):
    class NoPromptVision:
        pass

    monkeypatch.setattr(rr, "Vision", NoPromptVision)
    assert rr.validate_route_runner_contract() is False


def test_validate_contract_false_when_route_manager_method_missing(monkeypatch):
    class NoLoadRoutes:
        def get_next_ready_node(self):
            return None

        def get_route_nodes(self):
            return []

    monkeypatch.setattr(rr, "RouteManager", NoLoadRoutes)
    assert rr.validate_route_runner_contract() is False


def test_validate_contract_false_when_settings_not_a_dict(monkeypatch):
    monkeypatch.setattr(rr, "_load_route_settings", lambda: "not-a-dict")
    assert rr.validate_route_runner_contract() is False


def test_validate_contract_false_when_capabilities_empty(monkeypatch):
    monkeypatch.setattr(rr, "get_route_runner_capabilities", lambda: [])
    assert rr.validate_route_runner_contract() is False


# --- S-2705: unregister never calls keyboard.unhook_all() --------------------
def test_unregister_hotkeys_never_calls_unhook_all_even_on_error(monkeypatch):
    runner = _make_runner(monkeypatch)
    runner._hotkey_names = ["ctrl+alt+f8", "ctrl+alt+f9"]
    _keyboard_fail_on_remove.add("ctrl+alt+f8")

    runner._unregister_hotkeys()

    assert _keyboard_calls["unhook_all"] == 0
    # Both hotkeys were still attempted despite the first one failing.
    assert _keyboard_calls["remove"] == ["ctrl+alt+f8", "ctrl+alt+f9"]
    assert runner._hotkey_names == []


def test_unregister_hotkeys_removes_all_on_success(monkeypatch):
    runner = _make_runner(monkeypatch)
    runner._hotkey_names = ["ctrl+alt+f8", "ctrl+alt+f9"]

    runner._unregister_hotkeys()

    assert _keyboard_calls["unhook_all"] == 0
    assert _keyboard_calls["remove"] == ["ctrl+alt+f8", "ctrl+alt+f9"]
    assert runner._hotkey_names == []
