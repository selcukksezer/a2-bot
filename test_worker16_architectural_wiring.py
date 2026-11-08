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
from unittest.mock import MagicMock, patch
import pytest

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


class FakeOrchestrator(QObject):
    def __init__(self):
        super().__init__()
        self.overlay = MagicMock()
        self.radar_win = None
        self.settings_win = None
        self.calibration_win = None
        self.map_viewer_win = None
        self.yolo_factory_win = None
        self.analytics_win = None
        self.guide_win = None
        self.mobile_connect_win = None
        self.route_mgr = MagicMock()
        self.settings = MagicMock()
        self.settings_mgr = MagicMock()
        self.f9_menu = MagicMock()
        self.worker = MagicMock()
        self.worker.mode = "ROUTE"
        self.vision = MagicMock()
        self.minimap_scanner = MagicMock()
        self.material_scanner = MagicMock()
        self.yolo_detector = MagicMock()
        self.shell = None

    _on_shell_page_created = main.GatherionOrchestrator._on_shell_page_created
    _connect_map_viewer_signals = main.GatherionOrchestrator._connect_map_viewer_signals
    _connect_yolo_factory_signals = main.GatherionOrchestrator._connect_yolo_factory_signals
    _on_template_updated = main.GatherionOrchestrator._on_template_updated
    _on_settings_route_changed = main.GatherionOrchestrator._on_settings_route_changed
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
    orchestrator.vision.load_templates.assert_called_once()
    orchestrator.minimap_scanner._load_optional_templates.assert_called_once()
    orchestrator.overlay.add_log.assert_called_once()


def test_lazy_page_dynamic_wiring_map_viewer():
    """Verify when map page is materialized, region/node change handlers are wired."""
    orchestrator = FakeOrchestrator()
    map_widget = MockMapViewerWidget()

    orchestrator._on_shell_page_created("map", map_widget)
    assert orchestrator.map_viewer_win is map_widget

    # Test node_toggled emission
    map_widget.node_toggled.emit(42)
    orchestrator.overlay.add_log.assert_called_with("Harita: Nokta #42 durumu güncellendi")

    # Test node_deleted emission
    map_widget.node_deleted.emit(42)
    orchestrator.overlay.add_log.assert_called_with("Harita: Nokta #42 silindi")


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

    # Verify f9_menu updated
    orchestrator.f9_menu.set_active_state.assert_called_once()


def test_trigger_ai_training_safe_retrieval():
    """Verify trigger_ai_training_and_open_dialog retrieves lazy page without crashing if initially None."""
    orchestrator = FakeOrchestrator()
    orchestrator.shell = MockShell()
    orchestrator._show_shell_tab = MagicMock(return_value=True)

    yolo_widget = MockYoloWidget()
    orchestrator.shell.pages["yolo"] = yolo_widget
    orchestrator.yolo_factory_win = None

    # Run trigger_ai_training_and_open_dialog
    orchestrator.trigger_ai_training_and_open_dialog()

    # It should have retrieved yolo_widget from shell and clicked on_train_rtx_clicked
    assert orchestrator.yolo_factory_win is yolo_widget
    assert yolo_widget.training_clicked is True


def test_reload_ai_model_mechanism(tmp_path):
    """Verify reload_ai_model hot-reloads model in yolo_detector and self.worker."""
    fake_model = tmp_path / "models" / "best.onnx"
    fake_model.parent.mkdir(parents=True, exist_ok=True)
    fake_model.write_bytes(b"ONNX{fake_model_bytes}")

    orchestrator = FakeOrchestrator()
    orchestrator.worker.yolo_detector = MagicMock()

    with patch.object(main, "MODELS_DIR", str(fake_model.parent)):
        res = orchestrator.reload_ai_model(str(fake_model))
        assert res is True
        assert orchestrator.yolo_detector.model_path == os.path.abspath(str(fake_model))
        orchestrator.yolo_detector._load_model_if_available.assert_called_once()
        orchestrator.overlay.add_log.assert_called_once()


def test_shell_page_created_signal_dynamic_hooking():
    """Verify shell.page_created signal dynamically invokes _on_shell_page_created."""
    orchestrator = FakeOrchestrator()
    shell = MockShell()
    orchestrator.shell = shell

    # Hook shell page creation
    orchestrator._hook_shell_page_creation(shell)
    shell.page_created.connect(orchestrator._on_shell_page_created)

    # Emit dynamic creation of calibration and stats
    calib = MockCalibrationWidget()
    shell.page_created.emit("calibration", calib)
    assert orchestrator.calibration_win is calib

    stats_widget = QWidget()
    shell.page_created.emit("stats", stats_widget)
    assert orchestrator.analytics_win is stats_widget
