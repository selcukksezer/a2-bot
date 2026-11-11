# -*- coding: utf-8 -*-
"""D81 (main.py) DEV improvements — regression + behaviour-preservation tests.

main.py is a Windows/PyQt6 entry point: on this Linux VM `winsound`, `PyQt6`
and several `core.*`/`ui.*` modules cannot be imported. This file therefore
stubs every module that fails a real import (try-import first, stub only on
failure) and then imports `main` to test the pure-logic helpers and the
orchestrator methods with fake collaborators (no QApplication, no bot boot).

Run:  python3 -m pytest test_main_dev_d81.py -v
"""
import enum
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

WT_ROOT = os.path.dirname(os.path.abspath(__file__))
if WT_ROOT not in sys.path:
    sys.path.insert(0, WT_ROOT)


# ---------------------------------------------------------------------------
# Stubs (only for modules that fail a real import on this machine)
# ---------------------------------------------------------------------------
class _FakeSignal:
    """Minimal Qt-signal stand-in: connect/disconnect/emit with recording."""

    def __init__(self):
        self.connected = []
        self.emitted = []

    def connect(self, slot):
        self.connected.append(slot)

    def disconnect(self, slot=None):
        if slot is None:
            if not self.connected:
                raise RuntimeError("no connections")
            self.connected.clear()
            return
        if slot not in self.connected:
            raise RuntimeError("not connected")
        self.connected.remove(slot)

    def emit(self, *args):
        self.emitted.append(args)


def _pyqt_signal(*_a, **_k):
    return _FakeSignal()


def _pyqt_slot(*_a, **_k):
    def deco(fn):
        return fn

    return deco


class _QtNamespace(types.SimpleNamespace):
    pass


def _make_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod
    return mod


def _dummy_class(name):
    return type(name, (), {"__init__": lambda self, *a, **k: None})


class _BotModeStub(enum.Enum):
    ROUTE_MODE = "route"
    COMBAT_MODE = "combat"
    MANUAL_RECORD = "manual_record"
    AUTO_RECORD = "auto_record"
    EXPLORER_MODE = "explorer"
    ROUTE = "route"
    COMBAT = "combat"
    EXPLORER = "explorer"


def _try_import(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def install_stubs():
    """Stub Windows-only / Qt-only modules that fail to import here."""
    if not _try_import("winsound"):
        _make_module("winsound", Beep=lambda *a, **k: None)

    if not _try_import("PyQt6"):
        pyqt6 = _make_module("PyQt6")
        qtcore = _make_module(
            "PyQt6.QtCore",
            QObject=type("QObject", (), {"__init__": lambda self, *a, **k: None}),
            Qt=_QtNamespace(
                GlobalColor=_QtNamespace(transparent=0),
                PenStyle=_QtNamespace(NoPen=0),
                ApplicationAttribute=_QtNamespace(AA_ShareOpenGLContexts=0),
            ),
            QTimer=type("QTimer", (), {"singleShot": staticmethod(lambda *a, **k: None)}),
            pyqtSignal=_pyqt_signal,
            pyqtSlot=_pyqt_slot,
        )
        qtgui = _make_module(
            "PyQt6.QtGui",
            QAction=_dummy_class("QAction"),
            QColor=_dummy_class("QColor"),
            QIcon=_dummy_class("QIcon"),
            QPainter=_dummy_class("QPainter"),
            QPixmap=_dummy_class("QPixmap"),
        )
        qtwidgets = _make_module(
            "PyQt6.QtWidgets",
            QApplication=_dummy_class("QApplication"),
            QMenu=_dummy_class("QMenu"),
            QMessageBox=_dummy_class("QMessageBox"),
            QSystemTrayIcon=_dummy_class("QSystemTrayIcon"),
        )
        pyqt6.QtCore = qtcore
        pyqt6.QtGui = qtgui
        pyqt6.QtWidgets = qtwidgets

    # config / core / ui / dataset_factory: try real import, stub on failure.
    needed = {
        "config.schema": ["SettingsManager"],
        "core.behavioral_guard": ["BehavioralStealthGuard"],
        "core.capture": ["ScreenCapture"],
        "core.combat_recovery": ["CombatRecovery"],
        "core.depth_glide_navigator": ["GlideWaypointController", "MonocularDepthObstacleAvoidance"],
        "core.dpi_focus_guard": ["DPIScalingSynchronizer", "ForegroundFocusGuard", "RouteBackupManager", "SelfHealingTemplateTuner"],
        "core.dry_run_simulator": ["DryRunReplaySimulator"],
        "core.economy_broker": ["AutoCraftingMorphQueue", "SessionProfitSimulator", "TradeBrokerPriceScanner"],
        "core.explorer_mode": ["ExplorerModeController"],
        "core.human_timing": ["HumanTimingEngine", "OrganicAntiAFK"],
        "core.incident_recorder": ["IncidentRecorder", "MobileTokenAuthenticator"],
        "core.input_driver": ["InputController"],
        "core.material_scanner": ["MaterialScanner"],
        "core.minimap_scanner": ["MinimapScanner"],
        "core.multi_char_scheduler": ["AltCharacterManager", "AutomationTaskScheduler", "PetAndBuffManager"],
        "core.neural_audio_perception": ["AudioLoopbackFFTAnalyzer", "ONNXYolo3DDetector"],
        "core.ocr_distance": ["AdaptivePreprocessor", "GameHUDReader"],
        "core.opsec_crypto": ["LocalVaultCrypto", "NativeWin32HotkeyManager", "PanicAlertSystem"],
        "core.patch_normalizer": ["HighResROICropper", "MinimapOrientationGuard", "MotionBlurGate", "SobelClahePatchNormalizer", "WatershedClusterSeparator"],
        "core.path_optimizer": ["RouteOptimizer"],
        "core.questlog_importer": ["QuestlogImporter"],
        "core.route_manager": ["RouteManager"],
        "core.sequential_router": ["SequentialRoutePlanner"],
        "core.skill_bar_reader": ["MicroCameraLookAround", "SkillBarReader"],
        "core.state_machine": ["BotMode", "BotWorker"],
        "core.stealth_capture": ["AdaptiveFramePacer", "DXGIDesktopCapture", "StealthOverlayManager"],
        "core.telemetry": ["TelemetryEngine"],
        "core.uac_elevator": ["ensure_admin_elevation", "is_admin"],
        "core.vision": ["VisionEngine"],
        "core.vision_filters": ["DynamicInteractionKeyReader", "FeatherNegativeFilter", "MonsterCircularityFilter", "NameplateHSVIsolator", "Pyramid3DCrestDetector"],
        "core.watchdog": ["BotWatchdog", "SupportDiagnostics"],
        "core.windmouse_humanizer": ["BiometricClickGenerator", "PhysicalInputGuard", "WindMouseEngine"],
        "ui.app_shell": ["GatherionShell"],
        "ui.analytics_dialog": ["AnalyticsDialog"],
        "ui.calibration_wizard": ["CalibrationWizard"],
        "ui.cartography_radar": ["CartographyRadarDialog"],
        "ui.f9_menu": ["F9Menu"],
        "ui.guide_dialog": ["GuideDialog"],
        "ui.map_viewer": ["MapViewerWindow"],
        "ui.overlay_hud": ["OverlayHUD", "OverlaySignals"],
        "ui.settings_window": ["SettingsWindow"],
        "ui.mobile_connect_dialog": ["MobileConnectDialog"],
        "ui.yolo_factory_dialog": ["YOLOFactoryDialog"],
        "core.mobile_tunnel": ["FreeCloudTunnelManager"],
        "core.mobile_server": ["MobileRemoteServer"],
        "dataset_factory.exemplar_memory": ["get_exemplar_memory"],
    }
    for mod_name, attrs in needed.items():
        if mod_name in sys.modules:
            continue
        if _try_import(mod_name):
            continue
        parent, _, _ = mod_name.rpartition(".")
        if parent and parent not in sys.modules:
            _make_module(parent)
        stub_attrs = {}
        for attr in attrs:
            if attr == "BotMode":
                stub_attrs[attr] = _BotModeStub
            elif attr in ("ensure_admin_elevation", "is_admin"):
                stub_attrs[attr] = (lambda *a, **k: True) if attr == "is_admin" else (lambda *a, **k: False)
            elif attr == "get_exemplar_memory":
                stub_attrs[attr] = lambda *a, **k: None
            else:
                stub_attrs[attr] = _dummy_class(attr)
        _make_module(mod_name, **stub_attrs)
    if "config.schema" in sys.modules and not hasattr(sys.modules["config.schema"], "format_hotkey_label"):
        sys.modules["config.schema"].format_hotkey_label = lambda chord: str(chord)


install_stubs()

import main  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _make_orchestrator(**attrs):
    orch = main.GatherionOrchestrator.__new__(main.GatherionOrchestrator)
    orch._shutdown_done = False
    for key, val in attrs.items():
        setattr(orch, key, val)
    return orch


class _FakeWorker:
    def __init__(self):
        self.calls = []
        self.is_paused = False
        self.mode = None
        self._snapshot_calls = 0

    def set_mode(self, mode):
        self.calls.append(("set_mode", mode))

    def resume(self):
        self.calls.append(("resume",))

    def pause(self, reason=""):
        self.calls.append(("pause", reason))

    def _emit_hud_snapshot(self):
        self._snapshot_calls += 1

    def stop(self):
        self.calls.append(("stop",))

    def isRunning(self):
        return False

    def wait(self, ms):
        return True

    def requestInterruption(self):
        self.calls.append(("requestInterruption",))

    def terminate(self):
        self.calls.append(("terminate",))


class _FakeWatchdog:
    def __init__(self):
        self.calls = []

    def set_paused(self, paused, state=None):
        self.calls.append((paused, state))

    def stop(self):
        self.calls.append(("stop",))


class _FakeOverlay:
    def __init__(self):
        self.logs = []

    def add_log(self, msg):
        self.logs.append(str(msg))


class _FakeView:
    def __init__(self, visible=True):
        self.visible = visible
        self.reloads = 0
        self.table_loads = 0

    def isVisible(self):
        return self.visible

    def reload_map_and_nodes(self):
        self.reloads += 1

    def load_nodes_table(self):
        self.table_loads += 1


class _FakeDetector:
    def __init__(self, load_result=True, raise_on_load=False, has_loader=True):
        self.model_path = None
        self._load_result = load_result
        self._raise = raise_on_load
        self._has_loader = has_loader
        self.load_calls = 0

    def _load_model_if_available(self):
        self.load_calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return self._load_result


class _NoLoaderDetector:
    def __init__(self):
        self.model_path = None


class _FakeClosable:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.calls = []

    def close(self):
        self.calls.append("close")
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")

    def stop(self):
        self.calls.append("stop")
        if self.fail:
            raise RuntimeError(f"{self.name} stop failed")

    def save_session_report(self):
        self.calls.append("save_session_report")
        if self.fail:
            raise RuntimeError("telemetry save failed")

    def hide(self):
        self.calls.append("hide")
        if self.fail:
            raise RuntimeError("tray hide failed")


# ---------------------------------------------------------------------------
# 1. _safe_reconnect
# ---------------------------------------------------------------------------
class TestSafeReconnect:
    def test_disconnects_before_connect_and_returns_true(self):
        sig = _FakeSignal()
        slot = object()
        sig.connect(slot)
        order = []
        orig_dis, orig_con = sig.disconnect, sig.connect
        sig.disconnect = lambda s=None: (order.append("disconnect"), orig_dis(s))
        sig.connect = lambda s: (order.append("connect"), orig_con(s))
        assert main._safe_reconnect(sig, slot) is True
        assert order == ["disconnect", "connect"]
        assert sig.connected == [slot]

    def test_disconnect_failure_is_swallowed(self):
        sig = _FakeSignal()
        slot = object()
        sig.disconnect = mock.Mock(side_effect=RuntimeError("not connected"))
        sig.connect = mock.Mock()
        assert main._safe_reconnect(sig, slot) is True
        sig.connect.assert_called_once_with(slot)

    def test_connect_failure_returns_false_without_raising(self):
        sig = _FakeSignal()
        sig.connect = mock.Mock(side_effect=RuntimeError("gone"))
        assert main._safe_reconnect(sig, object()) is False

    def test_signal_without_disconnect_still_connects(self):
        sig = SimpleNamespace(connect=mock.Mock())
        assert main._safe_reconnect(sig, object()) is True

    def test_non_callable_connect_returns_false(self):
        sig = SimpleNamespace(disconnect=lambda s: None, connect=None)
        assert main._safe_reconnect(sig, object()) is False


# ---------------------------------------------------------------------------
# 2. Menu mode dispatch table
# ---------------------------------------------------------------------------
class TestMenuModeDispatch:
    @pytest.fixture
    def orch(self):
        worker = _FakeWorker()
        watchdog = _FakeWatchdog()
        return _make_orchestrator(
            worker=worker, watchdog=watchdog, overlay=_FakeOverlay(),
            f9_menu=None, settings=None, route_mgr=SimpleNamespace(),
        ), worker, watchdog

    @pytest.mark.parametrize(
        "action, mode_name, state",
        [
            ("resume", None, "WALKING"),
            ("auto_record", "AUTO_RECORD", "WALKING"),
            ("manual_record", "MANUAL_RECORD", "WAITING"),
            ("explorer", "EXPLORER_MODE", "WALKING"),
            ("explorer_mode", "EXPLORER_MODE", "WALKING"),
            ("combat", "COMBAT_MODE", "FIGHTING"),
            ("combat_mode", "COMBAT_MODE", "FIGHTING"),
            ("savas", "COMBAT_MODE", "FIGHTING"),
            ("savaş", "COMBAT_MODE", "FIGHTING"),
            ("fight", "COMBAT_MODE", "FIGHTING"),
            ("route", "ROUTE_MODE", "WALKING"),
            ("route_mode", "ROUTE_MODE", "WALKING"),
            ("gather", "ROUTE_MODE", "WALKING"),
            ("topla", "ROUTE_MODE", "WALKING"),
        ],
    )
    def test_mode_actions(self, orch, action, mode_name, state):
        o, worker, watchdog = orch
        o.handle_menu_action(action)
        assert watchdog.calls == [(False, state)]
        if mode_name is None:
            assert worker.calls == [("resume",)]
        else:
            expected = getattr(main.BotMode, mode_name).value
            assert worker.calls == [("set_mode", expected), ("resume",)]

    def test_unknown_action_warns_and_does_not_resume(self, orch):
        o, worker, watchdog = orch
        o.handle_menu_action("definitely_not_a_real_action")
        assert worker.calls == []
        assert watchdog.calls == []
        assert any("Bilinmeyen" in log for log in o.overlay.logs)


# ---------------------------------------------------------------------------
# 3. Map-view refresh dedup
# ---------------------------------------------------------------------------
class TestRefreshMapViews:
    @pytest.fixture
    def orch(self):
        worker = _FakeWorker()
        return _make_orchestrator(
            worker=worker, overlay=_FakeOverlay(),
            radar_win=_FakeView(visible=True),
            settings_win=_FakeView(visible=True),
            map_viewer_win=None,
        ), worker

    def test_canvas_added_refreshes_visible_windows_once(self, orch):
        o, worker = orch
        o._on_canvas_node_added()
        assert worker._snapshot_calls == 1
        assert o.radar_win.reloads == 1
        assert o.settings_win.table_loads == 1

    def test_canvas_moved_matches_added(self, orch):
        o, worker = orch
        o._on_canvas_node_moved()
        assert worker._snapshot_calls == 1
        assert o.radar_win.reloads == 1
        assert o.settings_win.table_loads == 1

    def test_canvas_handlers_refresh_even_when_hidden(self, orch):
        # Original behaviour: no isVisible() check in these four handlers —
        # hidden windows are reloaded too. The refactor must preserve that.
        o, worker = orch
        o.radar_win.visible = False
        o.settings_win.visible = False
        o._on_canvas_node_added()
        assert worker._snapshot_calls == 1
        assert o.radar_win.reloads == 1
        assert o.settings_win.table_loads == 1

    def test_map_node_toggled_refreshes_even_when_hidden(self, orch):
        o, worker = orch
        o.radar_win.visible = False
        o.settings_win.visible = False
        o._on_map_node_toggled(7)
        assert worker._snapshot_calls == 1
        assert o.radar_win.reloads == 1
        assert o.settings_win.table_loads == 1
        assert any("#7" in log for log in o.overlay.logs)

    def test_map_node_deleted_refreshes(self, orch):
        o, worker = orch
        o._on_map_node_deleted(3)
        assert worker._snapshot_calls == 1
        assert o.radar_win.reloads == 1
        assert any("#3" in log for log in o.overlay.logs)

    def test_snapshot_failure_does_not_raise_and_skips_reload(self, orch):
        o, worker = orch
        worker._emit_hud_snapshot = mock.Mock(side_effect=RuntimeError("hud down"))
        o._on_canvas_node_added()  # must not raise
        assert o.radar_win.reloads == 0


# ---------------------------------------------------------------------------
# 4. Detector reload dedup
# ---------------------------------------------------------------------------
class TestReloadDetector:
    def test_success_sets_path_and_returns_true(self):
        det = _FakeDetector(load_result=True)
        assert main._reload_detector(det, "/x/best.onnx", "det") is True
        assert det.model_path == "/x/best.onnx"
        assert det.load_calls == 1

    def test_load_false_returns_false(self):
        det = _FakeDetector(load_result=False)
        assert main._reload_detector(det, "/x/best.onnx", "det") is False

    def test_load_exception_returns_false(self):
        det = _FakeDetector(raise_on_load=True)
        assert main._reload_detector(det, "/x/best.onnx", "det") is False

    def test_missing_loader_returns_false(self):
        det = _NoLoaderDetector()
        assert main._reload_detector(det, "/x/best.onnx", "det") is False
        assert det.model_path == "/x/best.onnx"

    def test_none_detector_returns_false(self):
        assert main._reload_detector(None, "/x/best.onnx", "det") is False


class TestReloadAiModel:
    @pytest.fixture
    def orch(self):
        worker = _FakeWorker()
        worker.yolo_detector = _FakeDetector(load_result=False)
        worker.state_machine = None
        return _make_orchestrator(
            yolo_detector=_FakeDetector(load_result=True),
            worker=worker, overlay=_FakeOverlay(),
        )

    def test_reload_reports_success_when_one_detector_loads(self, orch):
        assert orch.reload_ai_model(model_path=os.path.abspath(__file__)) is True
        assert any("canlı güncellendi" in log for log in orch.overlay.logs)

    def test_reload_reports_failure_when_no_detector_loads(self, orch):
        orch.yolo_detector = _FakeDetector(load_result=False)
        assert orch.reload_ai_model(model_path=os.path.abspath(__file__)) is False
        assert any("güncellenemedi" in log for log in orch.overlay.logs)

    def test_reload_returns_false_when_no_model_file(self, orch):
        # No explicit path + an empty models dir -> nothing to reload.
        # NOTE: /tmp is full on this VM, so the empty dir lives under ~/workspace.
        empty_models = os.path.join(os.path.expanduser("~"), "workspace",
                                    "dev-d81-testdata", "empty_models")
        os.makedirs(empty_models, exist_ok=True)
        with mock.patch.object(main, "MODELS_DIR", empty_models):
            assert orch.reload_ai_model() is False


# ---------------------------------------------------------------------------
# 5. Shutdown close dedup
# ---------------------------------------------------------------------------
class TestStopQuietly:
    def test_ok_returns_true(self):
        obj = _FakeClosable("x")
        assert main._stop_quietly(obj, "close", "X close") is True
        assert obj.calls == ["close"]

    def test_failing_close_returns_false_without_raising(self):
        obj = _FakeClosable("x", fail=True)
        assert main._stop_quietly(obj, "close", "X close") is False

    def test_none_target_returns_false(self):
        assert main._stop_quietly(None, "close", "X close") is False

    def test_missing_method_returns_false(self):
        assert main._stop_quietly(object(), "close", "X close") is False


class TestShutdown:
    def test_all_close_attempts_run_despite_failures(self):
        worker = _FakeWorker()
        app = SimpleNamespace(quit=mock.Mock())
        orch = _make_orchestrator(
            watchdog=_FakeClosable("watchdog", fail=True),
            worker=worker,
            capture=_FakeClosable("capture", fail=True),
            dxg_capture=_FakeClosable("dxg"),
            input_ctrl=_FakeClosable("input"),
            mobile_server=_FakeClosable("mobile", fail=True),
            tunnel_mgr=_FakeClosable("tunnel"),
            telemetry=_FakeClosable("telemetry", fail=True),
            tray=_FakeClosable("tray"),
            app=app, shell=None, settings_win=None,
        )
        orch.shutdown()
        app.quit.assert_called_once()
        for target in (orch.watchdog, orch.capture, orch.dxg_capture,
                       orch.input_ctrl, orch.mobile_server, orch.tunnel_mgr,
                       orch.telemetry, orch.tray):
            assert target.calls, target.name  # every close/stop was attempted
        assert ("stop",) in worker.calls

    def test_second_shutdown_is_noop(self):
        app = SimpleNamespace(quit=mock.Mock())
        orch = _make_orchestrator(
            watchdog=_FakeClosable("watchdog"), worker=_FakeWorker(),
            capture=_FakeClosable("capture"), dxg_capture=_FakeClosable("dxg"),
            input_ctrl=_FakeClosable("input"), mobile_server=None, tunnel_mgr=None,
            telemetry=_FakeClosable("telemetry"), tray=None,
            app=app, shell=None, settings_win=None,
        )
        orch.shutdown()
        orch.shutdown()
        app.quit.assert_called_once()


# ---------------------------------------------------------------------------
# 6. LAN IP TTL cache
# ---------------------------------------------------------------------------
class TestLanIpCache:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        main._LAN_IP_CACHE["ip"] = ""
        main._LAN_IP_CACHE["ts"] = 0.0
        yield
        main._LAN_IP_CACHE["ip"] = ""
        main._LAN_IP_CACHE["ts"] = 0.0

    def test_repeated_calls_open_one_socket(self):
        constructions = {"n": 0}
        real_socket = main.socket.socket

        class CountingSocket(real_socket):
            def __init__(self, *a, **k):
                constructions["n"] += 1
                super().__init__(*a, **k)

        with mock.patch.object(main.socket, "socket", CountingSocket):
            first = main.get_local_lan_ip()
            for _ in range(19):
                assert main.get_local_lan_ip() == first
        assert constructions["n"] == 1

    def test_expired_cache_probes_again(self):
        with mock.patch.object(main.socket, "socket") as mock_sock_cls:
            mock_sock = mock_sock_cls.return_value.__enter__.return_value
            mock_sock.getsockname.return_value = ("10.0.0.5", 0)
            first = main.get_local_lan_ip()
            assert first == "10.0.0.5"
            calls_after_first = mock_sock_cls.call_count
            main._LAN_IP_CACHE["ts"] = 0.0  # force expiry
            second = main.get_local_lan_ip()
            assert second == "10.0.0.5"
            assert mock_sock_cls.call_count == calls_after_first + 1

    def test_failed_probe_falls_back_and_caches(self):
        with mock.patch.object(main.socket, "socket", side_effect=OSError("no net")), \
                mock.patch.object(main.socket, "gethostbyname", return_value="192.168.1.2") as gh:
            assert main.get_local_lan_ip() == "192.168.1.2"
            assert main.get_local_lan_ip() == "192.168.1.2"
            assert gh.call_count == 1


# ---------------------------------------------------------------------------
# 7. No redundant winsound import inside handlers
# ---------------------------------------------------------------------------
def test_single_winsound_import():
    src = open(main.__file__, encoding="utf-8").read()
    assert src.count("import winsound") == 1
