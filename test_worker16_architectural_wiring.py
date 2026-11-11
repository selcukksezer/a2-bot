# -*- coding: utf-8 -*-
"""
Unit and integration tests for Worker 16: Architectural Wiring & Signal-Slot Binding in main.py.
Verifies:
1. Lazy Page Dynamic Wiring (_on_shell_page_created, shell.page_created, _hook_shell_page_creation).
2. Template updated reload chain (_on_template_updated reloads vision, minimap, exemplars).
3. Settings Window route_changed connection and synchronization (_on_settings_route_changed).
4. Safe trigger_ai_training_and_open_dialog with lazy page retrieval.
5. Model & Patch Path Unification (reload_ai_model, anchored BASE_DIR constants).
"""

import os
import sys
from unittest.mock import patch
import pytest

# Headless Linux (CI/VM without $DISPLAY): run Qt offscreen instead of failing
# to connect to a display. Leaves real displays and Windows/macOS alone.
if os.environ.get("QT_QPA_PLATFORM") is None and os.name == "posix" and not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget

# Ensure QApplication exists for Qt signals/widgets
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

import main
from main import (
    BASE_DIR,
    CONFIG_DIR,
    SETTINGS_JSON_PATH,
    CALIBRATED_ROIS_PATH,
    DATASET_DIR,
    LEARNED_EXEMPLARS_PATH,
    MODELS_DIR,
    TEMPLATES_DIR,
    ROUTES_DIR,
)


class MockShell(QObject):
    page_created = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.pages = {}
        self._opened_tools = []

    def _mount(self, key: str, widget):
        """Minimal stand-in for GatherionShell._mount: store + return the widget."""
        self.pages[key] = widget
        return widget

    def open_tool(self, key: str):
        self._opened_tools.append(key)

    def page(self, key: str):
        return self.pages.get(key)

    def get_page_if_created(self, key: str):
        return self.pages.get(key)


class MockCalibrationWidget(QObject):
    template_updated = pyqtSignal(str)


class MockSettingsWidget(QObject):
    settings_saved = pyqtSignal()
    quit_requested = pyqtSignal()
    route_changed = pyqtSignal(str, str)


class MockMapViewerWidget(QObject):
    node_toggled = pyqtSignal(int)
    node_deleted = pyqtSignal(int)
    route_changed = pyqtSignal(str, str)
    region_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._region_name = ""
        self._route_name = ""
        self.reloaded = False
        self.updated = False
        self.canvas = None

    def reload_map_and_nodes(self):
        self.reloaded = True

    def update(self):
        self.updated = True


class MockRadarWidget(QObject):
    def __init__(self):
        super().__init__()
        self.reloaded = False
        self.updated = False

    def reload_map_and_nodes(self):
        self.reloaded = True

    def update(self):
        self.updated = True


class MockYoloWidget(QObject):
    model_published = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.training_clicked = False

    def on_train_rtx_clicked(self):
        self.training_clicked = True


class _StubRouteManager:
    """Real-attribute stand-in for the route manager (no MagicMock).

    A wrong attribute name (e.g. ``activeRegion``) raises AttributeError
    instead of silently creating a new mock attribute, so the wiring test
    actually proves which attributes production code writes.
    """

    def __init__(self):
        self.active_region = ""
        self.active_route = ""


class _StubSettings:
    def __init__(self):
        self.active_region = ""
        self.active_route = ""


class _StubSettingsManager:
    def __init__(self):
        self.saved = []

    def save(self, settings):
        self.saved.append(settings)


class _StubF9Menu:
    """Records set_active_state calls instead of swallowing them."""

    def __init__(self):
        self.set_active_state_calls = []
        self.set_active_route_calls = []

    def set_active_state(self, **kwargs):
        self.set_active_state_calls.append(kwargs)

    def set_active_route(self, region, route):
        self.set_active_route_calls.append((region, route))


class _StubWorker:
    def __init__(self):
        self.mode = "ROUTE"
        self.auto_cartography_enabled = True
        self.minimap_scanner = None
        self.dataset_factory = None
        self.yolo_detector = _StubDetector()
        self.hud_snapshot_calls = 0

    def _emit_hud_snapshot(self):
        self.hud_snapshot_calls += 1


class _StubDetector:
    def __init__(self):
        self.model_path = None
        self.load_calls = 0

    def _load_model_if_available(self):
        self.load_calls += 1
        # D4/F-1137: the real loader returns bool; a stub returning None
        # makes reload_ai_model honestly report failure.
        return True


class _RecordingVision:
    def __init__(self):
        self.load_templates_calls = 0

    def load_templates(self):
        self.load_templates_calls += 1


class _RecordingMinimapScanner:
    def __init__(self):
        self.load_calls = 0

    def _load_optional_templates(self):
        self.load_calls += 1


class _RecordingMaterialScanner:
    def __init__(self):
        self.reload_calls = 0

    def reload_templates(self):
        self.reload_calls += 1


class _RecordingOverlay:
    def __init__(self):
        self.logs = []

    def add_log(self, message):
        self.logs.append(message)


class FakeOrchestrator(QObject):
    def __init__(self):
        super().__init__()
        self.overlay = _RecordingOverlay()
        self.radar_win = None
        self.settings_win = None
        self.calibration_win = None
        self.map_viewer_win = None
        self.yolo_factory_win = None
        self.analytics_win = None
        self.guide_win = None
        self.mobile_connect_win = None
        self.route_mgr = _StubRouteManager()
        self.settings = _StubSettings()
        self.settings_mgr = _StubSettingsManager()
        self.f9_menu = _StubF9Menu()
        self.worker = _StubWorker()
        self.vision = _RecordingVision()
        self.minimap_scanner = _RecordingMinimapScanner()
        self.material_scanner = _RecordingMaterialScanner()
        self.yolo_detector = _StubDetector()
        self.shell = None

    _on_shell_page_created = main.GatherionOrchestrator._on_shell_page_created
    _connect_map_viewer_signals = main.GatherionOrchestrator._connect_map_viewer_signals
    _connect_yolo_factory_signals = main.GatherionOrchestrator._connect_yolo_factory_signals
    _on_template_updated = main.GatherionOrchestrator._on_template_updated
    _on_settings_route_changed = main.GatherionOrchestrator._on_settings_route_changed
    # R22: _on_shell_page_created "settings" dali bu uc slotu da baglar
    # (main.py:1316 settings_saved, :1318 quit_requested,
    # :1322 selective_gather_changed). Fake, odunc alinan handler'in
    # eristigi her metodu yansitmak zorunda; eksik olunca AttributeError
    # verir (v2 FAILED). Yalnizca baglanirlar, bu testte cagrilmazlar.
    _on_settings_saved = main.GatherionOrchestrator._on_settings_saved
    shutdown = main.GatherionOrchestrator.shutdown
    handle_selective_gather_changed = main.GatherionOrchestrator.handle_selective_gather_changed
    _on_map_node_toggled = main.GatherionOrchestrator._on_map_node_toggled
    _on_map_node_deleted = main.GatherionOrchestrator._on_map_node_deleted
    _on_canvas_node_added = main.GatherionOrchestrator._on_canvas_node_added
    _on_canvas_node_moved = main.GatherionOrchestrator._on_canvas_node_moved
    _hook_shell_page_creation = main.GatherionOrchestrator._hook_shell_page_creation
    reload_ai_model = main.GatherionOrchestrator.reload_ai_model
    trigger_ai_training_and_open_dialog = main.GatherionOrchestrator.trigger_ai_training_and_open_dialog


def test_path_anchoring_constants():
    """Verify all project-wide path constants are strictly anchored to BASE_DIR."""
    assert os.path.isabs(BASE_DIR)
    assert CONFIG_DIR == os.path.join(BASE_DIR, "config")
    assert SETTINGS_JSON_PATH == os.path.join(CONFIG_DIR, "settings.json")
    assert CALIBRATED_ROIS_PATH == os.path.join(CONFIG_DIR, "calibrated_rois.json")
    assert DATASET_DIR == os.path.join(BASE_DIR, "dataset")
    assert LEARNED_EXEMPLARS_PATH == os.path.join(DATASET_DIR, "learned_exemplars.json")
    assert MODELS_DIR == os.path.join(BASE_DIR, "models")
    assert TEMPLATES_DIR == os.path.join(BASE_DIR, "templates")
    assert ROUTES_DIR == os.path.join(BASE_DIR, "routes")


def test_lazy_page_dynamic_wiring_calibration():
    """Verify when calibration page is materialized, template_updated reloads all consumers."""
    orchestrator = FakeOrchestrator()
    calib_widget = MockCalibrationWidget()

    orchestrator._on_shell_page_created("calibration", calib_widget)
    assert orchestrator.calibration_win is calib_widget

    # Trigger template_updated
    calib_widget.template_updated.emit("new_template.png")
    assert orchestrator.vision.load_templates_calls == 1
    assert orchestrator.minimap_scanner.load_calls == 1
    assert orchestrator.material_scanner.reload_calls == 1
    assert len(orchestrator.overlay.logs) == 1
    assert "new_template.png" in orchestrator.overlay.logs[0]


def test_lazy_page_dynamic_wiring_map_viewer():
    """Verify when map page is materialized, region/node change handlers are wired."""
    orchestrator = FakeOrchestrator()
    map_widget = MockMapViewerWidget()

    orchestrator._on_shell_page_created("map", map_widget)
    assert orchestrator.map_viewer_win is map_widget

    # Test node_toggled emission
    map_widget.node_toggled.emit(42)
    assert "Harita: Nokta #42 durumu güncellendi" in orchestrator.overlay.logs

    # Test node_deleted emission
    map_widget.node_deleted.emit(42)
    assert "Harita: Nokta #42 silindi" in orchestrator.overlay.logs


def test_settings_route_changed_sync():
    """Verify settings_win.route_changed synchronizes f9_menu, overlay, map_viewer_win, and radar_win."""
    orchestrator = FakeOrchestrator()
    map_win = MockMapViewerWidget()
    radar_win = MockRadarWidget()
    orchestrator.map_viewer_win = map_win
    orchestrator.radar_win = radar_win

    settings_widget = MockSettingsWidget()
    orchestrator._on_shell_page_created("settings", settings_widget)
    assert orchestrator.settings_win is settings_widget

    # Emit route_changed
    settings_widget.route_changed.emit("Verteron", "Verteron Route 1")

    # Verify route_mgr & settings updated
    assert orchestrator.route_mgr.active_region == "Verteron"
    assert orchestrator.route_mgr.active_route == "Verteron Route 1"
    assert orchestrator.settings.active_region == "Verteron"
    assert orchestrator.settings.active_route == "Verteron Route 1"

    # Verify map viewer updated and reloaded
    assert map_win._region_name == "Verteron"
    assert map_win._route_name == "Verteron Route 1"
    assert map_win.reloaded is True
    assert map_win.updated is True

    # Verify radar updated and reloaded
    assert radar_win.reloaded is True
    assert radar_win.updated is True

    # Worker must have emitted a HUD snapshot as part of the sync
    assert orchestrator.worker.hud_snapshot_calls == 1

    # Verify f9_menu updated with the real region/route values (not just "called")
    assert len(orchestrator.f9_menu.set_active_state_calls) == 1
    f9_kwargs = orchestrator.f9_menu.set_active_state_calls[0]
    assert f9_kwargs["active_region"] == "Verteron"
    assert f9_kwargs["active_route"] == "Verteron Route 1"

    # Settings manager must have persisted the same settings object
    assert orchestrator.settings_mgr.saved == [orchestrator.settings]


def test_trigger_ai_training_safe_retrieval():
    """Verify trigger_ai_training_and_open_dialog retrieves lazy page without crashing if initially None."""
    orchestrator = FakeOrchestrator()
    orchestrator.shell = MockShell()
    orchestrator._show_shell_tab = lambda key: True  # UI tab switch shim

    yolo_widget = MockYoloWidget()
    orchestrator.shell.pages["yolo"] = yolo_widget
    orchestrator.yolo_factory_win = None

    # Run trigger_ai_training_and_open_dialog
    orchestrator.trigger_ai_training_and_open_dialog()

    # It should have retrieved yolo_widget from shell and clicked on_train_rtx_clicked
    assert orchestrator.yolo_factory_win is yolo_widget
    assert yolo_widget.training_clicked is True


def test_reload_ai_model_mechanism(tmp_path):
    """Verify reload_ai_model hot-reloads a REAL ONNX model in yolo_detector and self.worker.

    F-1442: the old test wrote fake bytes (b"ONNX{...}") and claimed the model
    was reloaded although no ONNX parser ever saw the file. This test copies
    the repo's genuine yolov8n_gatherion.onnx (PyTorch-exported ModelProto)
    and reloads that copy, so the reload path handles a real model file.
    """
    import shutil
    from pathlib import Path as _Path

    real_model = _Path(main.MODELS_DIR) / "yolov8n_gatherion.onnx"
    assert real_model.is_file(), "repo models/yolov8n_gatherion.onnx is missing"
    header = real_model.read_bytes()[:64]
    assert header[:1] == b"\x08" and b"pytorch" in header, "not a genuine ONNX ModelProto"

    fake_dir = tmp_path / "models"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_copy = fake_dir / "best.onnx"
    shutil.copy(real_model, real_copy)

    orchestrator = FakeOrchestrator()

    with patch.object(main, "MODELS_DIR", str(fake_dir)):
        res = orchestrator.reload_ai_model(str(real_copy))
        assert res is True
        assert orchestrator.yolo_detector.model_path == os.path.abspath(str(real_copy))
        assert orchestrator.yolo_detector.load_calls == 1
        # The worker's own detector must be reloaded too, with the same path.
        assert orchestrator.worker.yolo_detector.model_path == os.path.abspath(str(real_copy))
        assert orchestrator.worker.yolo_detector.load_calls == 1
        assert len(orchestrator.overlay.logs) == 1


def test_shell_page_created_signal_dynamic_hooking():
    """Verify _hook_shell_page_creation itself wires dynamic page creation.

    F-1336: the old test connected ``shell.page_created`` to the handler by
    hand and then emitted it, so it tested its own wiring, not the hook.
    _hook_shell_page_creation does NOT connect page_created at all — it wraps
    shell._mount and shell.open_tool. This test therefore exercises only the
    hook: after hooking, mounting / opening a page through the shell must
    invoke _on_shell_page_created with no manual signal connection.
    """
    orchestrator = FakeOrchestrator()
    shell = MockShell()
    orchestrator.shell = shell

    # Hook shell page creation — no manual .connect() anywhere in this test.
    orchestrator._hook_shell_page_creation(shell)
    assert getattr(shell._mount, "_gatherion_hooked", False) is True
    assert getattr(shell.open_tool, "_gatherion_hooked", False) is True

    # Path 1: _mount creates the calibration page -> handler runs.
    calib = MockCalibrationWidget()
    shell._mount("calibration", calib)
    assert orchestrator.calibration_win is calib

    # Path 2: open_tool materializes a page already registered in shell.pages.
    stats_widget = QWidget()
    shell.pages["stats"] = stats_widget
    shell.open_tool("stats")
    assert orchestrator.analytics_win is stats_widget

    # The hook must be idempotent: hooking twice must not double-wrap/fire twice.
    orchestrator._hook_shell_page_creation(shell)
    calib2 = MockCalibrationWidget()
    shell._mount("calibration", calib2)
    assert orchestrator.calibration_win is calib2
