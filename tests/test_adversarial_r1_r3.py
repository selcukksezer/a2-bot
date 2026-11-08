# -*- coding: utf-8 -*-
"""
tests/test_adversarial_r1_r3.py

Adversarial Stress Test and Vulnerability Harness for:
- Requirement R1: Live Data Binding (ui/app_shell.py, ui/settings_window.py, ui/analytics_dialog.py, core/system_monitor.py)
- Requirement R3: YOLO Preview Annotations (ui/yolo_panel.py)

Designed by Teamwork Challenger Live 2 to empirically verify boundary conditions,
null injections, extreme values, out-of-bounds coordinates, thread deadlocks,
and failure modes.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication

# Ensure QApplication is initialized for headless/GUI testing
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from core.system_monitor import (
    SystemMonitor,
    _query_capture_metrics,
    _query_cpu_metrics,
    _query_gpu_metrics,
)
from ui.analytics_dialog import AnalyticsDialog
from ui.app_shell import _HomeMinimapCanvas, _HomePage, GatherionShell
from ui.settings_window import SettingsWindow
from ui.yolo_panel import (
    CLASS_COLORS,
    CLASS_ID_TO_NAME,
    DEFAULT_BOX_COLOR,
    YOLOPreviewCanvasWidget,
    YOLOPreviewDialog,
    draw_annotation_overlay,
    normalize_annotation,
    render_annotated_pixmap,
)


# ==============================================================================
# SECTION 1: REQUIREMENT R3 — YOLO PREVIEW ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestYOLOPreviewAdversarial:
    """Stress tests and boundary condition verification for YOLO preview annotations."""

    def test_yolo_preview_stress_1000_boxes(self):
        """Stress: Pass 1,000+ bounding boxes simultaneously. Verify high-throughput rendering."""
        pixmap = QPixmap(1920, 1080)
        pixmap.fill(QColor("#000000"))

        # Generate 1,500 diverse bounding boxes
        boxes: List[Dict[str, Any]] = []
        for i in range(1500):
            boxes.append({
                "cx": (i % 50) / 50.0,
                "cy": ((i // 50) % 30) / 30.0,
                "w": 0.05,
                "h": 0.04,
                "class_id": i % len(CLASS_ID_TO_NAME),
                "confidence": 0.50 + (i % 50) * 0.01,
                "trusted": (i % 2 == 0),
            })

        t0 = time.time()
        annotated = render_annotated_pixmap(pixmap, boxes, show_labels=True)
        elapsed = time.time() - t0

        assert annotated is not None
        assert not annotated.isNull()
        assert annotated.width() == 1920
        assert annotated.height() == 1080
        # 1,500 boxes should render within reasonable time (< 2.0 seconds)
        assert elapsed < 2.0, f"Rendering 1,500 boxes took too long: {elapsed:.2f}s"

    def test_yolo_preview_out_of_bounds_coordinates(self):
        """Boundary: Pass negative coordinates, excessive dimensions, and inverted spans."""
        pixmap = QPixmap(800, 600)
        pixmap.fill(QColor("#11151c"))

        adversarial_boxes = [
            # Heavily negative cx, cy
            {"cx": -5.0, "cy": -10.0, "w": 0.2, "h": 0.2, "class_name": "herb_aria", "confidence": 0.95},
            # Oversized width and height (> 10.0, where 1.0 is full image)
            {"cx": 0.5, "cy": 0.5, "w": 15.0, "h": 25.0, "class_name": "ore_orichalcum", "confidence": 0.88},
            # Center outside positive boundaries
            {"cx": 10.0, "cy": 12.0, "w": 0.1, "h": 0.1, "class_name": "monster_3d", "confidence": 0.70},
            # Zero and negative dimensions
            {"cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.0, "class_name": "gem_crystal", "confidence": 0.99},
            {"cx": 0.3, "cy": 0.3, "w": -0.5, "h": -0.5, "class_name": "tree_log", "confidence": 0.60},
        ]

        # Must not crash, raise exceptions, or return null pixmap
        out = render_annotated_pixmap(pixmap, adversarial_boxes, show_labels=True)
        assert not out.isNull()

        # Canvas widget rendering check
        canvas = YOLOPreviewCanvasWidget()
        canvas.resize(600, 400)
        canvas.set_content(pixmap, adversarial_boxes)
        # Render onto a test pixmap to trigger paintEvent
        test_surface = QPixmap(600, 400)
        painter = QPainter(test_surface)
        try:
            canvas.render(painter)
        finally:
            painter.end()

    def test_yolo_preview_malformed_annotations_resilience(self):
        """Malformed Data: Verify normalize_annotation handles empty, corrupted, and unexpected structures."""
        # Empty dict
        norm1 = normalize_annotation({})
        assert norm1["cx"] == 0.5
        assert norm1["confidence"] == 0.90
        assert norm1["class_name"] == "target"

        # None input
        norm2 = normalize_annotation(None)
        assert norm2["cx"] == 0.5
        assert norm2["class_name"] == "target"

        # Primitive string input
        norm3 = normalize_annotation("corrupted_string_item")
        assert norm3["cx"] == 0.5

        # Unknown class id
        norm4 = normalize_annotation({"class_id": 999999})
        assert norm4["class_name"] == "class_999999"

        # Corrupted non-numeric values
        norm5 = normalize_annotation({"cx": "not_a_float", "confidence": "invalid_conf"})
        assert norm5["cx"] == 0.5
        assert norm5["confidence"] == 0.90

        # Tuple formats: short, YOLO standard, and extended
        norm_t1 = normalize_annotation([0.1, 0.2, 0.3, 0.4])
        assert norm_t1["cx"] == 0.1
        assert norm_t1["w"] == 0.3

        norm_t2 = normalize_annotation((0, 0.2, 0.3, 0.4, 0.5, 0.88))
        assert norm_t2["class_name"] == CLASS_ID_TO_NAME[0]
        assert norm_t2["confidence"] == 0.88

        # Corrupted tuple elements vulnerability
        bad_tuple_caught = False
        try:
            normalize_annotation((0, "bad", "bad", "bad", "bad", "bad"))
        except (ValueError, TypeError):
            bad_tuple_caught = True

        assert not bad_tuple_caught, "VULNERABILITY CONFIRMED: normalize_annotation crashes on tuple with non-float coordinates"

    def test_yolo_preview_nan_and_inf_vulnerability(self):
        """Vulnerability Hunt: Inject NaN and Inf values into confidence and coordinates.
        Demonstrates whether round(nan) / round(inf) crashes draw_annotation_overlay.
        """
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor("#000000"))

        nan_box = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": float("nan")}
        inf_box = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": float("inf")}
        neg_inf_box = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": float("-inf")}

        # Test NaN confidence
        nan_caught = False
        try:
            render_annotated_pixmap(pixmap, [nan_box], show_labels=True)
        except (ValueError, TypeError, OverflowError):
            nan_caught = True

        # Test Inf confidence
        inf_caught = False
        try:
            render_annotated_pixmap(pixmap, [inf_box], show_labels=True)
        except (ValueError, TypeError, OverflowError):
            inf_caught = True

        # Test -Inf confidence
        neg_inf_caught = False
        try:
            render_annotated_pixmap(pixmap, [neg_inf_box], show_labels=True)
        except (ValueError, TypeError, OverflowError):
            neg_inf_caught = True

        # EMPIRICAL ASSERTION:
        # If any of these raise an uncaught exception, we flag this as a critical failure mode!
        assert not nan_caught, "VULNERABILITY CONFIRMED: NaN confidence crashes render_annotated_pixmap with unhandled ValueError"
        assert not inf_caught, "VULNERABILITY CONFIRMED: Inf confidence crashes render_annotated_pixmap with unhandled OverflowError"
        assert not neg_inf_caught, "VULNERABILITY CONFIRMED: -Inf confidence crashes render_annotated_pixmap with unhandled OverflowError"

    def test_yolo_preview_rapid_dialog_lifecycle(self):
        """Lifecycle Stress: Rapidly create, show, and close 50 YOLOPreviewDialog instances."""
        pixmap = QPixmap(320, 240)
        pixmap.fill(QColor("#1e293b"))
        sample_boxes = [
            {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "class_name": "herb_aria", "confidence": 0.92},
            {"cx": 0.3, "cy": 0.4, "w": 0.1, "h": 0.1, "class_name": "monster_3d", "confidence": 0.45, "trusted": False},
        ]

        for i in range(50):
            dlg = YOLOPreviewDialog(pixmap, sample_boxes, title=f"Test Lightbox {i}", meta="640x640 · 30ms")
            dlg.show()
            dlg.chk_annotations.setChecked(False)
            dlg.chk_annotations.setChecked(True)
            dlg.accept()
            dlg.deleteLater()

        _app.processEvents()


# ==============================================================================
# SECTION 2: REQUIREMENT R1 — SYSTEM MONITOR HARDWARE FAILURES
# ==============================================================================

class TestSystemMonitorAdversarial:
    """Stress tests simulating hardware monitor exceptions, crashes, and null returns."""

    def test_nvml_complete_crash_simulation(self):
        """Simulate NVML initialization and query failures."""
        SystemMonitor.reset_cache()

        # 1. Simulate pynvml import failure / uninstalled
        with patch.dict(sys.modules, {"pynvml": None}):
            with patch("core.system_monitor._NVML_INITIALIZED", None):
                gpu = _query_gpu_metrics()
                assert gpu["available"] is False
                assert gpu["pct"] == 0
                assert gpu["temp_str"] == "—°C"
                assert gpu["label_str"] == "0% · —°C · 0.0/0.0 GB"

        # 2. Simulate pynvml raising exceptions during queries
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML driver communication failed")
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            with patch("core.system_monitor._NVML_INITIALIZED", None):
                gpu = _query_gpu_metrics()
                assert gpu["available"] is False
                assert gpu["pct"] == 0

    def test_psutil_complete_crash_simulation(self):
        """Simulate psutil throwing exceptions and returning empty/None CPU frequencies."""
        # 1. psutil cpu_percent raising exception
        with patch("psutil.cpu_percent", side_effect=PermissionError("WMI permission denied")):
            cpu = _query_cpu_metrics()
            assert cpu["pct"] == 0
            assert cpu["freq_ghz"] == 2.4
            assert cpu["label_str"] == "0% · 2.4 GHz"

        # 2. psutil cpu_freq returning None
        with patch("psutil.cpu_freq", return_value=None):
            with patch("psutil.cpu_count", return_value=None):
                cpu = _query_cpu_metrics()
                assert cpu["cores"] == 4
                assert cpu["freq_ghz"] == 2.4

    def test_capture_metrics_null_and_malformed_vulnerability(self):
        """Verify _query_capture_metrics handling of None, 0.0, non-float strings, and inf."""
        # 1. None capture instance
        cap1 = _query_capture_metrics(None)
        assert cap1["fps"] == 0.0
        assert cap1["latency_ms"] == 0.0
        assert cap1["pct"] == 0
        assert cap1["label_str"] == "0 FPS · 0.0 ms"

        # 2. Capture instance with 0.0 fps
        mock_zero = MagicMock()
        mock_zero.last_fps = 0.0
        cap2 = _query_capture_metrics(mock_zero)
        assert cap2["fps"] == 0.0
        assert cap2["label_str"] == "0 FPS · 0.0 ms"

        # 3. Capture instance with negative fps
        mock_neg = MagicMock()
        mock_neg.last_fps = -15.0
        cap3 = _query_capture_metrics(mock_neg)
        assert cap3["fps"] == -15.0
        assert cap3["label_str"] == "0 FPS · 0.0 ms"

        # 4. Vulnerability Check: last_fps is a corrupt string (e.g. "not_a_number")
        mock_bad_str = MagicMock()
        mock_bad_str.last_fps = "not_a_number"
        bad_str_caught = False
        try:
            _query_capture_metrics(mock_bad_str)
        except ValueError:
            bad_str_caught = True

        assert not bad_str_caught, "VULNERABILITY CONFIRMED: _query_capture_metrics raises unhandled ValueError on non-numeric last_fps"

        # 5. Vulnerability Check: last_fps is float('inf')
        mock_inf = MagicMock()
        mock_inf.last_fps = float("inf")
        inf_caught = False
        try:
            _query_capture_metrics(mock_inf)
        except OverflowError:
            inf_caught = True

        assert not inf_caught, "VULNERABILITY CONFIRMED: _query_capture_metrics raises unhandled OverflowError on float('inf') last_fps"


# ==============================================================================
# SECTION 3: REQUIREMENT R1 — APP SHELL LIVE DATA BINDING ADVERSARIAL TESTS
# ==============================================================================

class TestAppShellLiveDataAdversarial:
    """Stress tests and null injections for ui/app_shell.py."""

    @pytest.fixture
    def home_page(self):
        page = _HomePage(
            on_open=lambda x: None,
            on_action=lambda x: None,
            on_chord=lambda x: None,
        )
        page.resize(1024, 768)
        return page

    def test_app_shell_update_system_metrics_null_injections(self, home_page):
        """Null Injection: Test update_system_metrics when sys_snapshot is None or contains None values."""
        # 1. sys_snapshot is None
        none_snap_caught = False
        try:
            home_page.update_system_metrics(None)
        except (AttributeError, TypeError):
            none_snap_caught = True

        assert not none_snap_caught, "VULNERABILITY CONFIRMED: update_system_metrics crashes on sys_snapshot=None"

        # 2. sys_snapshot has None sub-dictionaries
        none_subdict_caught = False
        try:
            home_page.update_system_metrics({"gpu": None, "cpu": None, "capture": None})
        except (AttributeError, TypeError):
            none_subdict_caught = True

        assert not none_subdict_caught, "VULNERABILITY CONFIRMED: update_system_metrics crashes on None gpu/cpu/capture subdicts"

        # 3. sys_snapshot has None pct values
        none_pct_caught = False
        try:
            home_page.update_system_metrics({
                "gpu": {"name": "GPU", "label_str": "0%", "pct": None},
                "cpu": {"name": "CPU", "label_str": "0%", "pct": None},
                "capture": {"label_str": "0 FPS", "pct": None},
            })
        except (AttributeError, TypeError):
            none_pct_caught = True

        assert not none_pct_caught, "VULNERABILITY CONFIRMED: update_system_metrics crashes on int(None) for pct fields"

    def test_app_shell_minimap_corrupt_nodes_and_rendering(self, home_page):
        """Corrupt Route Data: Pass malformed node objects and verify canvas paintEvent doesn't crash."""
        corrupt_nodes = [
            None,
            {"id": "bad_id", "norm_x": "bad_x", "norm_y": None},
            {"id": None, "norm_x": 0.5, "norm_y": 0.5},
            {"id": 42, "norm_x": float("nan"), "norm_y": float("inf")},
        ]

        # 1. update_route_steps should accept corrupt nodes without raising
        update_caught = False
        try:
            home_page.update_route_steps("Altgard", "TestRoute", corrupt_nodes, active_idx=0)
        except Exception:
            update_caught = True

        assert not update_caught, "update_route_steps raised an unhandled exception on corrupt nodes"

        # 2. Canvas paintEvent must render without crashing on corrupt node coordinates
        render_caught = False
        try:
            home_page.minimap_canvas.paintEvent(None)
        except (ValueError, TypeError):
            render_caught = True

        assert not render_caught, "VULNERABILITY CONFIRMED: _HomeMinimapCanvas.paintEvent crashes on float('bad_x') conversion"

    def test_app_shell_update_detections_malformed_and_nulls(self, home_page):
        """Malformed YOLO Detections: Pass corrupted conf, missing fields, and null footer_stats."""
        # 1. Non-numeric conf string
        bad_conf_caught = False
        try:
            home_page.update_detections([{"name": "Herb", "conf": "bad_conf_string"}], None)
        except (ValueError, TypeError):
            bad_conf_caught = True

        assert not bad_conf_caught, "VULNERABILITY CONFIRMED: update_detections crashes on float('bad_conf_string')"

        # 2. footer_stats with None values
        none_footer_caught = False
        try:
            home_page.update_detections([], {"sessions": None, "best_rate": None, "avg_confidence": None})
        except (ValueError, TypeError):
            none_footer_caught = True

        assert not none_footer_caught, "VULNERABILITY CONFIRMED: update_detections crashes on int(None) for footer stats"

    def test_app_shell_topbar_perf_idle_and_exception_fallback(self):
        """Verify _update_topbar_perf gracefully recovers from exceptions and displays idle indicators."""
        # Mock GatherionShell top_perf_label
        shell = MagicMock(spec=GatherionShell)
        from PyQt6.QtWidgets import QLabel
        lbl = QLabel()
        shell.top_perf_label = lbl
        shell._worker = None

        with patch("core.system_monitor.SystemMonitor.get_snapshot", side_effect=RuntimeError("Monitor exploded")):
            GatherionShell._update_topbar_perf(shell, running=False)
            text = lbl.text()
            assert "0 FPS" in text or "FPS <b style=\"color:#eef1f5;\">0</b>" in text
            assert "BEKLEMEDE" in text
            assert "0ms" in text


# ==============================================================================
# SECTION 4: REQUIREMENT R1 — SETTINGS WINDOW ADVERSARIAL TESTS
# ==============================================================================

class TestSettingsWindowLiveDataAdversarial:
    """Stress tests and null injections for ui/settings_window.py."""

    @pytest.fixture
    def settings_window(self):
        # Create SettingsWindow with minimal mock config
        mock_cfg = MagicMock()
        mock_cfg.rest_minutes = 15
        mock_cfg.auto_record = False
        win = SettingsWindow(settings=mock_cfg)
        win.resize(1100, 700)
        return win

    def test_settings_window_header_perf_null_and_exception_resilience(self, settings_window):
        """Verify _update_header_perf handles None/empty SystemMonitor snapshots gracefully."""
        # 1. Snapshot returns empty dict
        with patch("core.system_monitor.SystemMonitor.get_snapshot", return_value={}):
            settings_window._update_header_perf()
            assert "0" in settings_window.chip_fps.text()
            assert "0%" in settings_window.chip_gpu.text()

        # 2. SystemMonitor raises exception
        with patch("core.system_monitor.SystemMonitor.get_snapshot", side_effect=Exception("System monitor down")):
            # Should not raise
            settings_window._update_header_perf()

    def test_settings_window_live_node_statuses_null_user_data(self, settings_window):
        """Verify _update_live_node_statuses handles rows with None UserRole timestamps."""
        # Populate table with 5 dummy rows
        settings_window.table.setRowCount(5)
        for r in range(5):
            from PyQt6.QtWidgets import QTableWidgetItem
            item = QTableWidgetItem("—")
            item.setData(Qt.ItemDataRole.UserRole, None)  # Inject None as timestamp
            settings_window.table.setItem(r, 3, item)

        # Should not throw TypeError on float(None)
        settings_window._update_live_node_statuses()


# ==============================================================================
# SECTION 5: REQUIREMENT R1 — ANALYTICS DIALOG ADVERSARIAL TESTS
# ==============================================================================

class TestAnalyticsDialogLiveDataAdversarial:
    """Stress tests and null injections for ui/analytics_dialog.py."""

    @pytest.fixture
    def analytics_dialog(self):
        dlg = AnalyticsDialog()
        dlg.resize(1200, 800)
        return dlg

    def test_analytics_dialog_refresh_metrics_null_telemetry_returns(self, analytics_dialog):
        """Null Injection: TelemetryEngine returns None, empty dict, or dict with None values."""
        # 1. get_live_metrics returns None
        analytics_dialog.telemetry.get_live_metrics = MagicMock(return_value=None)
        uncaught_1 = False
        try:
            analytics_dialog.refresh_metrics()
        except Exception:
            uncaught_1 = True
        assert not uncaught_1, "refresh_metrics let an exception escape when get_live_metrics returns None"

        # 2. get_live_metrics returns dict with None fields
        analytics_dialog.telemetry.get_live_metrics = MagicMock(return_value={
            "session_duration_sec": None,
            "combat_count": None,
            "failed_total": None,
            "gathered_total": None,
            "gathers_per_hour": None,
            "success_rate_pct": None,
            "avg_walk_sec": None,
            "avg_match_pct": None,
        })
        uncaught_2 = False
        try:
            analytics_dialog.refresh_metrics()
        except Exception:
            uncaught_2 = True
        assert not uncaught_2, "refresh_metrics let an exception escape when numeric metrics are None"

    def test_analytics_dialog_corrupt_detailed_breakdown_rows(self, analytics_dialog):
        """Corrupt Data: detailed_breakdown contains None items and non-numeric fields."""
        corrupt_breakdown = [
            None,
            {"resource_type": None, "region": None, "route": None, "count": "not_an_int"},
            {"resource_type": "Aria", "region": "Altgard", "route": "R1", "count": 10, "avg_walk_sec": "corrupt_float"},
        ]
        analytics_dialog.telemetry.get_live_metrics = MagicMock(return_value={
            "session_duration_sec": 120.0,
            "detailed_breakdown": corrupt_breakdown,
        })

        uncaught = False
        try:
            analytics_dialog.refresh_metrics()
        except Exception:
            uncaught = True
        assert not uncaught, "refresh_metrics let an exception escape when detailed_breakdown rows are corrupt"

    def test_analytics_dialog_initial_clean_idle_state(self, analytics_dialog):
        """Acceptance Criteria: Verify default clean idle indicators before any live data arrives."""
        assert analytics_dialog.card_gathered.value_lbl.text() == "0"
        assert analytics_dialog.card_rate.value_lbl.text() == "0.0"
        assert analytics_dialog.card_success.value_lbl.text() == "%100.0"
        assert analytics_dialog.card_walk.value_lbl.text() == "0.00sn"
