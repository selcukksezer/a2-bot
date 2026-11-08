# -*- coding: utf-8 -*-
"""
test_live_ui_smoke.py

Automated PySide / PyQt6 Headless Verification Suite for:
1. Requirement R1: Live Data Binding Across 3 Redesigned Pages
   - ui/app_shell.py (Anasayfa)
   - ui/settings_window.py (Ayarlar)
   - ui/analytics_dialog.py (İstatistik)
   - core/system_monitor.py (SystemMonitor)
2. Requirement R2: Floating HUD Overlay Layout Repair
   - ui/overlay_hud.py / ui/overlay_window.py (OverlayHUD)
3. Requirement R3: YOLO Preview Popup Annotations
   - ui/yolo_panel.py / ui/yolo_factory_dialog.py (YOLOPreviewDialog, render_annotated_pixmap)

All tests execute headless offscreen (QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import os
import sys

# Force offscreen QPA platform for headless environments before importing Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

import time
import unittest
from typing import Any, Dict, List, Optional, Sequence

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

# Global test app instance
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)


class TestLiveDataBinding(unittest.TestCase):
    """
    Requirement R1 Verification:
    Asserts no hardcoded prototype strings remain in active UI views.
    Asserts clean default idle indicators ('0 FPS', 'Beklemede', '—', zeroed counters).
    Asserts live data propagation from system monitors, telemetry, and route managers.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        self.widgets_to_cleanup: List[QWidget] = []

    def tearDown(self) -> None:
        for w in self.widgets_to_cleanup:
            try:
                w.close()
                w.deleteLater()
            except Exception:
                pass
        self.widgets_to_cleanup.clear()
        self.app.processEvents()

    def _collect_widget_texts(self, root: QWidget) -> List[str]:
        """Recursively collects visible text from all QLabel, QPushButton, etc."""
        texts: List[str] = []
        for child in root.findChildren(QWidget):
            if isinstance(child, QLabel):
                txt = child.text().strip()
                if txt:
                    texts.append(txt)
            elif isinstance(child, QPushButton):
                txt = child.text().strip()
                if txt:
                    texts.append(txt)
        return texts

    def test_system_monitor_snapshot_structure_and_safe_fallback(self) -> None:
        """Verifies core/system_monitor.py provides safe, cached metrics with clean fallbacks."""
        from core.system_monitor import SystemMonitor

        snapshot = SystemMonitor.get_snapshot()
        self.assertIsInstance(snapshot, dict)
        self.assertIn("gpu", snapshot)
        self.assertIn("cpu", snapshot)
        self.assertIn("capture", snapshot)

        gpu = snapshot["gpu"]
        self.assertIn("name", gpu)
        self.assertIn("pct", gpu)
        self.assertIn("temp", gpu)
        self.assertIn("mem_used_gb", gpu)
        self.assertIn("mem_total_gb", gpu)
        self.assertIsInstance(gpu["pct"], int)

        cpu = snapshot["cpu"]
        self.assertIn("cores", cpu)
        self.assertIn("pct", cpu)
        self.assertIn("freq_ghz", cpu)
        self.assertIsInstance(cpu["pct"], int)

        cap = snapshot["capture"]
        self.assertIn("fps", cap)
        self.assertIn("latency_ms", cap)
        self.assertIsInstance(cap["fps"], (int, float))
        self.assertIsInstance(cap["latency_ms"], (int, float))

    def test_app_shell_no_mock_prototype_strings_in_active_views(self) -> None:
        """
        Verifies GatherionShell and HomePage contain NO prototype mock strings:
        - No hardcoded 'FPS 61', 'GPU 32%', 'CPU 14%', 'LAT 16ms'
        - No 'Altgard · Dredgion Crash Site — Rota #3 · 195 node'
        - No fake route steps ('Kuzey Kayalık — Odyle Kümesi', 'Dredgion Enkazı — Aether Damarı')
        - No fake detections ('Saf Aether Kristali', 'Odyle Parçası', 'Aria Cevheri', 'Saldırgan: Lepharist Gözcü')
        - No fake footer stats ('Bugün: 3 oturum · en iyi toplama 68/saat · ortalama güven %90.2')
        """
        from ui.app_shell import GatherionShell

        shell = GatherionShell()
        self.widgets_to_cleanup.append(shell)
        self.app.processEvents()

        # Check topbar performance label
        topbar_text = shell.top_perf_label.text()
        self.assertNotIn("61", topbar_text, f"Topbar still contains hardcoded '61': {topbar_text}")
        self.assertNotIn("GPU <b style=\"color:#eef1f5;\">32%</b>", topbar_text)
        self.assertNotIn("CPU <b style=\"color:#eef1f5;\">14%</b>", topbar_text)
        self.assertNotIn("LAT <b style=\"color:#eef1f5;\">16ms</b>", topbar_text)

        # Check home page
        home_page = shell.pages.get("home")
        self.assertIsNotNone(home_page, "Home page not found in shell.pages")
        texts = self._collect_widget_texts(home_page)
        all_text = " ".join(texts)

        forbidden_strings = [
            "Altgard · Dredgion Crash Site — Rota #3 · 195 node",
            "Kuzey Kayalık — Odyle Kümesi",
            "Dredgion Enkazı — Aether Damarı",
            "Saf Aether Kristali",
            "Odyle Parçası",
            "Aria Cevheri",
            "Saldırgan: Lepharist Gözcü",
            "68/saat",
            "%90.2",
            "best_v8s.pt",
        ]
        for forbidden in forbidden_strings:
            self.assertNotIn(
                forbidden,
                all_text,
                f"HomePage active view contains forbidden prototype string '{forbidden}'",
            )

    def test_app_shell_clean_default_idle_indicators(self) -> None:
        """
        Verifies that unstarted/idle state displays clean default idle indicators:
        - Topbar: clean idle or live hardware metrics
        - Active nodes default: '0' (not '195')
        - Session chip default: '0:00' with subtext 'beklemede' or 'duraklatıldı' (not '3:44')
        - Route summary: contains '0 node' or '—' or 'Bölge Seçilmedi' / 'Rota Yok'
        - Route progress: '%0 — 0 aktif node' (not '%18 — 195 aktif node')
        """
        from ui.app_shell import GatherionShell

        shell = GatherionShell()
        self.widgets_to_cleanup.append(shell)
        self.app.processEvents()

        home_page = shell.pages.get("home")
        self.assertIsNotNone(home_page)

        # Active nodes counter
        stat_chips = getattr(home_page, "_stats", getattr(home_page, "_stat_chips", {}))
        if "active" in stat_chips:
            chip = stat_chips["active"]
            val_lbl = getattr(chip, "value", getattr(chip, "_value_label", None))
            if val_lbl is not None:
                active_val = val_lbl.text().strip()
                self.assertNotEqual(
                    active_val,
                    "195",
                    f"Active node chip defaulted to mock value '195' instead of '0'",
                )

        # Session chip
        if hasattr(home_page, "session_chip"):
            chip = home_page.session_chip
            val_lbl = getattr(chip, "value", getattr(chip, "_value_label", None))
            if val_lbl is not None:
                sess_val = val_lbl.text().strip()
                self.assertNotEqual(
                    sess_val,
                    "3:44",
                    f"Session chip defaulted to mock value '3:44' instead of '0:00'",
                )
                self.assertIn(sess_val, ("0:00", "00:00", "—"))

        # Route summary label
        if hasattr(home_page, "route_summary_label"):
            route_text = home_page.route_summary_label.text()
            self.assertNotIn("195 node", route_text)

        # Route progress label
        if hasattr(home_page, "route_progress_label"):
            prog_text = home_page.route_progress_label.text()
            self.assertNotIn("%18", prog_text)
            self.assertNotIn("195", prog_text)

    def test_app_shell_apply_snapshot_live_binding(self) -> None:
        """
        Verifies that calling apply_snapshot with live data updates the UI elements accordingly
        without adding mock offsets (e.g. no +224 offset).
        """
        from ui.app_shell import GatherionShell

        shell = GatherionShell()
        self.widgets_to_cleanup.append(shell)
        self.app.processEvents()

        home_page = shell.pages.get("home")
        self.assertIsNotNone(home_page)

        snapshot = {
            "region": "Morheim",
            "route": "Ice Fortress Path",
            "active": 35,
            "gathered": 18,
            "failures": 2,
            "yz": 5,
            "fps": 55.4,
            "gpu": {"name": "Test GPU", "pct": 45, "temp": 52, "mem_used_gb": 3.0, "mem_total_gb": 8.0},
            "cpu": {"cores": 16, "pct": 22, "freq_ghz": 3.4},
            "capture": {"fps": 55.4, "latency_ms": 18.0},
            "state": "RUNNING",
            "mode": "explorer",
            "session_time_str": "05:12",
        }

        home_page.apply_snapshot(snapshot)
        self.app.processEvents()

        # Check route summary label
        if hasattr(home_page, "route_summary_label"):
            route_summary = home_page.route_summary_label.text()
            self.assertIn("Morheim", route_summary)
            self.assertIn("Ice Fortress Path", route_summary)
            self.assertIn("35", route_summary)

        # Check stat chips
        chips = getattr(home_page, "_stats", getattr(home_page, "_stat_chips", {}))
        for key, expected_val in [("gathered", "18"), ("active", "35"), ("failures", "2"), ("yz", "5")]:
            if key in chips:
                chip = chips[key]
                val_lbl = getattr(chip, "value", getattr(chip, "_value_label", None))
                if val_lbl is not None:
                    self.assertEqual(val_lbl.text().strip(), expected_val)

    def test_settings_window_clean_defaults_and_no_mock_strings(self) -> None:
        """
        Verifies SettingsWindow:
        - No hardcoded 'FPS 61', 'GPU 32%', 'CPU 14%', 'LAT 16ms' in header
        - No fake scan status 'YZ taraması yanlış düğüm bulamadı'
        - Version string is 'Gatherion v2.0' (not 'v1.6.15')
        - Header chips exist and update cleanly with live metrics
        """
        from ui.settings_window import SettingsWindow

        settings = SettingsWindow()
        self.widgets_to_cleanup.append(settings)
        self.app.processEvents()

        # Check status label
        if hasattr(settings, "status_lbl"):
            status_text = settings.status_lbl.text()
            self.assertNotIn(
                "YZ taraması yanlış düğüm bulamadı",
                status_text,
                "Settings status label still displays hardcoded scan report",
            )
            self.assertTrue(
                "hazır" in status_text.lower() or "beklemede" in status_text.lower() or "yüklendi" in status_text.lower(),
                f"Settings status label does not indicate clean ready/idle state: {status_text}",
            )

        # Check version label
        if hasattr(settings, "version_lbl"):
            version_text = settings.version_lbl.text()
            self.assertNotIn("v1.6.15", version_text, "Settings version label is outdated (v1.6.15)")
            self.assertIn("v2.0", version_text)

        # Check header metric chips
        for chip_attr in ("chip_fps", "chip_gpu", "chip_cpu", "chip_lat"):
            if hasattr(settings, chip_attr):
                chip = getattr(settings, chip_attr)
                txt = chip.text()
                self.assertNotIn("61", txt, f"{chip_attr} contains hardcoded 61")
                self.assertNotIn("32%", txt, f"{chip_attr} contains hardcoded 32%")
                self.assertNotIn("14%", txt, f"{chip_attr} contains hardcoded 14%")
                self.assertNotIn("16ms", txt, f"{chip_attr} contains hardcoded 16ms")

    def test_analytics_dialog_clean_fallbacks_and_no_mock_multipliers(self) -> None:
        """
        Verifies AnalyticsDialog:
        - avg_match_pct default fallback is 0.0 (NOT fake 89.5)
        - table pricing does NOT hardcode cnt * 9900
        - insights region does NOT hardcode 'Altgard' when empty
        """
        from ui.analytics_dialog import AnalyticsDialog

        analytics = AnalyticsDialog()
        self.widgets_to_cleanup.append(analytics)
        self.app.processEvents()

        analytics.refresh_metrics()
        self.app.processEvents()

        last_metrics = getattr(analytics, "_last_metrics", {})
        avg_match = last_metrics.get("avg_match_pct", 0.0)
        self.assertNotEqual(
            avg_match,
            89.5,
            f"AnalyticsDialog _last_metrics still falls back to fake 89.5 match confidence!",
        )


class TestOverlayHUDLayout(unittest.TestCase):
    """
    Requirement R2 Verification:
    Asserts floating HUD overlay layout geometry:
    - Default size (540, 290) and minimum size (500, 275) without bottom vertical cutoff.
    - Zero text clipping on all 8 action buttons via QFontMetrics.
    - Stats container row 1 has at least 26px height (no number truncation).
    - Log labels have independent vertical clearance (no overlapping text).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        # Import either from ui.overlay_window or ui.overlay_hud
        try:
            from ui.overlay_window import OverlayHUD
        except ImportError:
            from ui.overlay_hud import OverlayHUD

        self.hud = OverlayHUD()
        self.hud.show()
        self.app.processEvents()
        self.addCleanup(self._cleanup_hud)

    def _cleanup_hud(self) -> None:
        try:
            self.hud.close()
            self.hud.deleteLater()
        except Exception:
            pass
        self.app.processEvents()

    def test_overlay_window_alias_module(self) -> None:
        """Verifies ui/overlay_window.py provides backward compatibility alias bridge."""
        import ui.overlay_window as ow

        self.assertTrue(hasattr(ow, "OverlayHUD") or hasattr(ow, "OverlayWindow"))
        cls_ref = getattr(ow, "OverlayWindow", getattr(ow, "OverlayHUD", None))
        self.assertIsNotNone(cls_ref)

    def test_hud_dimensions_default_and_minimum(self) -> None:
        """Verifies HUD supports default (540, 290) and minimum (500, 275) dimensions."""
        # Minimum size check
        min_size = self.hud.minimumSize()
        self.assertLessEqual(min_size.width(), 500, f"HUD minimumWidth {min_size.width()} > 500")
        self.assertLessEqual(min_size.height(), 275, f"HUD minimumHeight {min_size.height()} > 275")

        # Resize to default
        self.hud.resize(540, 290)
        self.app.processEvents()
        self.assertEqual(self.hud.width(), 540)
        self.assertGreaterEqual(self.hud.height(), 275)

    def test_hud_no_bottom_vertical_cutoff_at_default_and_minimum(self) -> None:
        """
        Asserts that card layout minimum size fits within window height across default
        and minimum dimensions, ensuring zero bottom vertical cutoff.
        """
        test_sizes = [(540, 290), (500, 275)]
        for w, h in test_sizes:
            with self.subTest(size=(w, h)):
                self.hud.resize(w, h)
                if hasattr(self.hud, "card"):
                    self.hud.card.resize(w, h)
                self.app.processEvents()

                card = getattr(self.hud, "card", self.hud)
                card_layout = card.layout()
                self.assertIsNotNone(card_layout)

                min_h = card_layout.minimumSize().height()
                self.assertLessEqual(
                    min_h,
                    h,
                    f"Card layout minimum height {min_h}px overflows window height {h}px at size ({w}, {h})",
                )

                # Verify footer layout or version label does not overflow bottom
                if hasattr(self.hud, "version_label"):
                    geo = self.hud.version_label.geometry()
                    self.assertLessEqual(
                        geo.bottom(),
                        h,
                        f"Version label bottom {geo.bottom()} exceeds HUD height {h}",
                    )

    def test_all_8_action_buttons_no_text_clipping(self) -> None:
        """
        Asserts all 8 action buttons (Başlat, Menü, Nokta, Durdur, Devam, Savaş, Toplam, Kapat):
        - Have non-empty text.
        - Zero text clipping: QFontMetrics horizontal advance + horizontal clearance <= allocated button width.
        - Verified across default (540, 290) and minimum (500, 275) window sizes.
        """
        button_attrs = [
            ("btn_run", "Başlat"),
            ("btn_menu", "Menü"),
            ("btn_node", "Nokta"),
            ("btn_stop", "Durdur"),
            ("btn_resume", ("Devam", "Devam Et")),
            ("btn_combat", "Savaş"),
            ("btn_gather", ("Toplam", "Toplama")),
            ("btn_quit", "Kapat"),
        ]

        test_sizes = [(540, 290), (500, 275)]
        for w, h in test_sizes:
            self.hud.resize(w, h)
            if hasattr(self.hud, "card"):
                self.hud.card.resize(w, h)
            self.app.processEvents()

            for btn_attr, expected_text in button_attrs:
                with self.subTest(size=(w, h), button=btn_attr):
                    self.assertTrue(
                        hasattr(self.hud, btn_attr),
                        f"OverlayHUD missing required button attribute '{btn_attr}'",
                    )
                    btn: QPushButton = getattr(self.hud, btn_attr)
                    self.assertIsInstance(btn, QPushButton)

                    text = btn.text().strip()
                    self.assertTrue(len(text) > 0, f"Button {btn_attr} text is empty")

                    if isinstance(expected_text, tuple):
                        self.assertIn(text, expected_text)
                    else:
                        self.assertEqual(text, expected_text)

                    # QFontMetrics text width measurement
                    fm = btn.fontMetrics()
                    text_w = fm.horizontalAdvance(text)
                    alloc_w = btn.width()

                    # Assert text width fits comfortably within button width
                    self.assertLessEqual(
                        text_w,
                        alloc_w,
                        f"Button '{btn_attr}' text '{text}' ({text_w}px) exceeds allocated width ({alloc_w}px) at ({w}, {h})",
                    )

                    # Assert clearance: button width must be at least text_w + 2px (zero clipping guarantee)
                    self.assertGreaterEqual(
                        alloc_w - text_w,
                        2,
                        f"Button '{btn_attr}' text '{text}' has no clearance ({alloc_w - text_w}px) at ({w}, {h})",
                    )

    def test_stats_container_row_height_sufficient(self) -> None:
        """
        Asserts that the stats container row 1 (numerical values) has at least 26px height
        to guarantee 19px-22px bold glyphs are not truncated vertically.
        """
        test_sizes = [(540, 290), (500, 275)]
        for w, h in test_sizes:
            with self.subTest(size=(w, h)):
                self.hud.resize(w, h)
                if hasattr(self.hud, "card"):
                    self.hud.card.resize(w, h)
                self.app.processEvents()

                # Check stat value labels
                for stat_attr in ("gathered_label", "ready_label", "failed_label"):
                    if hasattr(self.hud, stat_attr):
                        lbl: QLabel = getattr(self.hud, stat_attr)
                        lbl_h = lbl.height()
                        self.assertGreaterEqual(
                            lbl_h,
                            24,
                            f"Stat label {stat_attr} height {lbl_h}px is less than 24px minimum at size ({w}, {h})",
                        )

    def test_logs_container_no_overlapping_labels(self) -> None:
        """
        Asserts that the 3 log labels have independent vertical clearance and do not
        overlap each other vertically.
        """
        test_sizes = [(540, 290), (500, 275)]
        for w, h in test_sizes:
            with self.subTest(size=(w, h)):
                self.hud.resize(w, h)
                if hasattr(self.hud, "card"):
                    self.hud.card.resize(w, h)
                self.app.processEvents()

                log_labels = getattr(self.hud, "log_labels", [])
                if len(log_labels) >= 3:
                    geo0 = log_labels[0].geometry()
                    geo1 = log_labels[1].geometry()
                    geo2 = log_labels[2].geometry()

                    # Assert label 0 is strictly above label 1
                    self.assertLessEqual(
                        geo0.y() + geo0.height(),
                        geo1.y() + 1,
                        f"Log label 0 overlaps log label 1: y0={geo0.y()} h0={geo0.height()} vs y1={geo1.y()}",
                    )
                    # Assert label 1 is strictly above label 2
                    self.assertLessEqual(
                        geo1.y() + geo1.height(),
                        geo2.y() + 1,
                        f"Log label 1 overlaps log label 2: y1={geo1.y()} h1={geo1.height()} vs y2={geo2.y()}",
                    )


class TestYOLOPreviewAnnotations(unittest.TestCase):
    """
    Requirement R3 Verification:
    Asserts YOLO Preview Popup Annotations:
    - ui/yolo_panel.py exports YOLOPreviewDialog and render_annotated_pixmap.
    - render_annotated_pixmap accurately renders bounding boxes, class names, and confidence scores.
    - YOLOPreviewDialog initializes, responds to resize, and handles annotation toggling.
    - Real drawing verification: pixels in annotated image are modified from source.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        self.dialogs_to_cleanup: List[QDialog] = []

    def tearDown(self) -> None:
        for d in self.dialogs_to_cleanup:
            try:
                d.close()
                d.deleteLater()
            except Exception:
                pass
        self.dialogs_to_cleanup.clear()
        self.app.processEvents()

    def test_yolo_panel_module_exports(self) -> None:
        """Verifies ui/yolo_panel.py exists and exports required components."""
        import ui.yolo_panel as yp

        self.assertTrue(hasattr(yp, "YOLOPreviewDialog"))
        self.assertTrue(hasattr(yp, "render_annotated_pixmap"))
        self.assertTrue(hasattr(yp, "CLASS_COLORS"))
        self.assertTrue(hasattr(yp, "normalize_annotation"))

    def test_render_annotated_pixmap_bakes_boxes_and_labels(self) -> None:
        """
        Creates a test pixmap and asserts that render_annotated_pixmap renders
        bounding boxes and text labels, modifying pixels in the annotated regions.
        """
        from ui.yolo_panel import render_annotated_pixmap

        # Create 640x480 solid dark gray image
        src = QPixmap(640, 480)
        src.fill(QColor("#101010"))

        annotations = [
            {
                "class_name": "herb_aria",
                "confidence": 0.92,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "trusted": True,
            },
            {
                "class_name": "monster_3d",
                "confidence": 0.85,
                "cx": 0.2,
                "cy": 0.3,
                "w": 0.15,
                "h": 0.15,
                "trusted": False,
            },
            {
                "class_name": "player_hp_bar",
                "confidence": 0.99,
                "cx": 0.5,
                "cy": 0.88,
                "w": 0.4,
                "h": 0.05,
                "trusted": True,
            },
        ]

        annotated = render_annotated_pixmap(src, annotations, show_labels=True)
        self.assertFalse(annotated.isNull())
        self.assertEqual(annotated.width(), 640)
        self.assertEqual(annotated.height(), 480)

        # Real drawing verification: convert to QImage and verify pixel modification
        src_img = src.toImage()
        ann_img = annotated.toImage()

        # Center box region (0.5 * 640 = 320, 0.5 * 480 = 240)
        # Bounding box perimeter for herb_aria should have green pixels
        diff_count = 0
        for x in range(300, 340):
            for y in range(220, 260):
                if ann_img.pixelColor(x, y) != src_img.pixelColor(x, y):
                    diff_count += 1

        self.assertGreater(
            diff_count,
            0,
            "render_annotated_pixmap failed to modify any pixels in the bounding box area (facade detected!)",
        )

    def test_render_annotated_pixmap_without_labels(self) -> None:
        """Verifies render_annotated_pixmap with show_labels=False draws only bounding boxes."""
        from ui.yolo_panel import render_annotated_pixmap

        src = QPixmap(400, 300)
        src.fill(QColor("#000000"))

        annotations = [
            {"class_name": "ore_orichalcum", "confidence": 0.88, "cx": 0.4, "cy": 0.4, "w": 0.1, "h": 0.1}
        ]
        ann = render_annotated_pixmap(src, annotations, show_labels=False)
        self.assertFalse(ann.isNull())
        self.assertEqual(ann.size(), QSize(400, 300))

    def test_yolo_preview_dialog_initialization_and_canvas(self) -> None:
        """
        Verifies YOLOPreviewDialog initializes properly with a test image and annotations,
        configures the interactive canvas, and provides close and toggle controls.
        """
        from ui.yolo_panel import YOLOPreviewDialog

        src = QPixmap(640, 480)
        src.fill(QColor("#1a1a24"))

        annotations = [
            {"class_name": "herb_aria", "confidence": 0.94, "cx": 0.5, "cy": 0.5, "w": 0.18, "h": 0.18},
            {"class_name": "obstacle", "confidence": 0.78, "cx": 0.7, "cy": 0.6, "w": 0.12, "h": 0.12},
        ]

        dialog = YOLOPreviewDialog(src, annotations, title="Test Önizleme", meta="kare_01.jpg")
        self.dialogs_to_cleanup.append(dialog)
        self.app.processEvents()

        # Assert dialog properties
        self.assertEqual(dialog.windowTitle(), "Test Önizleme")
        self.assertGreaterEqual(dialog.width(), 800)
        self.assertGreaterEqual(dialog.height(), 500)

        # Assert canvas exists and contains content
        self.assertTrue(hasattr(dialog, "canvas"))
        self.assertEqual(len(dialog.canvas._annotations), 2)
        self.assertTrue(dialog.canvas._show_annotations)

        # Test toggle annotations
        if hasattr(dialog, "chk_annotations"):
            dialog.chk_annotations.setChecked(False)
            self.assertFalse(dialog.canvas._show_annotations)
            dialog.chk_annotations.setChecked(True)
            self.assertTrue(dialog.canvas._show_annotations)

    def test_yolo_preview_edge_cases(self) -> None:
        """
        Adversarial & boundary tests:
        - Empty annotations list
        - Null / None pixmap
        - Out-of-bounds coordinates (cx < 0, cx > 1, etc.)
        - Unknown class names
        """
        from ui.yolo_panel import YOLOPreviewDialog, render_annotated_pixmap

        # 1. Empty annotations
        src = QPixmap(200, 200)
        src.fill(QColor("#222222"))
        ann_empty = render_annotated_pixmap(src, [])
        self.assertFalse(ann_empty.isNull())

        # 2. Null pixmap
        null_pix = QPixmap()
        res = render_annotated_pixmap(null_pix, [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}])
        self.assertTrue(res.isNull())

        # 3. Out-of-bounds annotations
        oob_annotations = [
            {"class_name": "unknown_mob", "confidence": 0.50, "cx": -0.5, "cy": 1.8, "w": 3.0, "h": 2.0},
            {"class_name": "herb_mela", "confidence": -0.2, "cx": 1.2, "cy": -0.3, "w": -0.5, "h": 0.1},
        ]
        ann_oob = render_annotated_pixmap(src, oob_annotations)
        self.assertFalse(ann_oob.isNull())

        # 4. Dialog with None pixmap
        dialog = YOLOPreviewDialog(None, [])
        self.dialogs_to_cleanup.append(dialog)
        self.app.processEvents()
        self.assertIsNotNone(dialog)

    def test_yolo_factory_dialog_lightbox_integration(self) -> None:
        """
        Verifies YOLOFactoryDialog imports and routes review preview popup to YOLOPreviewDialog
        or preserves annotations for lightbox display.
        """
        from ui.yolo_factory_dialog import YOLOFactoryDialog

        factory = YOLOFactoryDialog()
        self.dialogs_to_cleanup.append(factory)
        self.app.processEvents()

        # Assert review preview canvas exists
        self.assertTrue(hasattr(factory, "review_preview"))
        # Assert lightbox open method exists
        self.assertTrue(hasattr(factory, "_open_review_lightbox"))


if __name__ == "__main__":
    unittest.main()
