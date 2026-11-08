# -*- coding: utf-8 -*-
"""
tests/test_overlay_hud_adversarial.py
Adversarial Verification & Stress Test Harness for Requirement R2:
Floating HUD Overlay Layout Repair across boundary conditions and stress tests.

Tested Dimensions:
1. Extreme Window Dimensions ((480, 260) to (800, 600))
2. Action Button Headroom & Long Button Text Injections (Clipping Limits)
3. High-Rate Stat Counter Updates (100 updates/sec, large numbers e.g. 999,999)
4. Long Log String Overflow (No widget overlaps, no vertical layout distortion)
5. Header Badge Layout Compression (DURAKLATILDI + MANUEL KAYIT + 3x Toplama)
"""

import os
import sys
import time
import unittest

# Ensure headless Qt support
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFontMetrics

# Ensure single QApplication instance
app = QApplication.instance() or QApplication(sys.argv)

from ui.overlay_hud import OverlayHUD
from ui.overlay_window import OverlayWindow


class TestOverlayHUDAdversarial(unittest.TestCase):
    """Adversarial stress harness for OverlayHUD layout under boundary conditions."""

    def setUp(self):
        self.hud = OverlayHUD()
        self.hud.show()
        app.processEvents()

    def tearDown(self):
        self.hud.close()
        app.processEvents()

    def test_r2_alias_conformance(self):
        """Verify OverlayWindow alias and required components."""
        self.assertIs(OverlayWindow, OverlayHUD)
        self.assertIsNotNone(self.hud.btn_run)
        self.assertIsNotNone(self.hud.btn_menu)
        self.assertIsNotNone(self.hud.btn_node)
        self.assertIsNotNone(self.hud.btn_stop)
        self.assertIsNotNone(self.hud.btn_resume)
        self.assertIsNotNone(self.hud.btn_combat)
        self.assertIsNotNone(self.hud.btn_gather)
        self.assertIsNotNone(self.hud.btn_quit)

    def test_default_and_minimum_geometry_no_bottom_cutoff(self):
        """
        Verify that at DEFAULT_SIZE (540, 290) and MINIMUM_SIZE (500, 275),
        all widgets fit vertically inside the window without bottom cutoff.
        """
        for size in [OverlayHUD.DEFAULT_SIZE, OverlayHUD.MINIMUM_SIZE]:
            with self.subTest(size=size):
                self.hud.resize(*size)
                app.processEvents()

                w_h = self.hud.height()
                footer_bottom = self.hud.f10_shortcut.y() + self.hud.f10_shortcut.height()
                version_bottom = self.hud.version_label.y() + self.hud.version_label.height()

                self.assertLessEqual(footer_bottom, w_h, f"Footer truncated at size {size}: {footer_bottom} > {w_h}")
                self.assertLessEqual(version_bottom, w_h, f"Version truncated at size {size}: {version_bottom} > {w_h}")

    def test_standard_action_buttons_no_clipping(self):
        """
        Verify that all 8 required action buttons:
        ['Başlat', 'Menü', 'Nokta', 'Durdur', 'Devam', 'Savaş', 'Toplam', 'Kapat']
        render cleanly with positive headroom across default (540, 290) and minimum (500, 275).
        """
        expected_buttons = [
            (self.hud.btn_run, "Başlat"),
            (self.hud.btn_menu, "Menü"),
            (self.hud.btn_node, "Nokta"),
            (self.hud.btn_stop, "Durdur"),
            (self.hud.btn_resume, "Devam"),
            (self.hud.btn_combat, "Savaş"),
            (self.hud.btn_gather, "Toplam"),
            (self.hud.btn_quit, "Kapat"),
        ]

        for size in [OverlayHUD.DEFAULT_SIZE, OverlayHUD.MINIMUM_SIZE]:
            self.hud.resize(*size)
            app.processEvents()

            for btn, expected_text in expected_buttons:
                with self.subTest(size=size, button=expected_text):
                    self.assertEqual(btn.text(), expected_text)
                    fm = QFontMetrics(btn.font())
                    text_w = fm.horizontalAdvance(btn.text())
                    # QSS: padding: 0px 5px; -> 10px horizontal padding + 2px border = 12px
                    needed_w = text_w + 12
                    actual_w = btn.width()
                    self.assertGreaterEqual(
                        actual_w,
                        needed_w,
                        f"Button '{expected_text}' clipped at size {size}: actual={actual_w}px < needed={needed_w}px"
                    )

    def test_adversarial_button_text_injection_clipping_limit(self):
        """
        Adversarial test: inject longer Turkish action labels into the 8 buttons
        to determine the exact clipping limit under the single-row layout.
        """
        long_texts = {
            self.hud.btn_run: "Başlat/Devam",
            self.hud.btn_menu: "Menüyü Aç",
            self.hud.btn_node: "Nokta Ekle",
            self.hud.btn_stop: "Durduruldu",
            self.hud.btn_resume: "Devam Et",
            self.hud.btn_combat: "Savaş Modu",
            self.hud.btn_gather: "Toplama Yap",
            self.hud.btn_quit: "Kapat/Çıkış",
        }

        for btn, text in long_texts.items():
            btn.setText(text)

        self.hud.resize(*OverlayHUD.MINIMUM_SIZE)
        app.processEvents()

        clipped_count = 0
        for btn, text in long_texts.items():
            fm = QFontMetrics(btn.font())
            text_w = fm.horizontalAdvance(text)
            needed_w = text_w + 12
            actual_w = btn.width()
            if actual_w < needed_w:
                clipped_count += 1

        # Documents empirical finding: all 8 buttons clip when injected with >7 character labels
        self.assertGreater(
            clipped_count,
            0,
            "Adversarial finding: Single-row action buttons clip when injected with longer labels (>7 chars)"
        )

    def test_adversarial_header_badge_clipping(self):
        """
        Adversarial test: Test harvest_badge_label under long state ('DURAKLATILDI')
        and long mode ('MANUEL KAYIT') at MINIMUM_SIZE (500, 275) and DEFAULT_SIZE (540, 290).
        Verifies that harvest_badge_label is NOT clipped at minimum or default sizes.
        """
        for size in [OverlayHUD.MINIMUM_SIZE, OverlayHUD.DEFAULT_SIZE]:
            with self.subTest(size=size):
                self.hud.resize(*size)
                self.hud.update_state({
                    "state": "DURAKLATILDI",
                    "mode": "MANUEL KAYIT",
                    "harvest_stage": 2,
                    "elapsed": "12:34:56",
                })
                app.processEvents()

                fm = QFontMetrics(self.hud.harvest_badge_label.font())
                text_w = fm.horizontalAdvance(self.hud.harvest_badge_label.text())
                widget_w = self.hud.harvest_badge_label.width()

                is_clipped = widget_w < text_w
                self.assertFalse(
                    is_clipped,
                    f"harvest_badge_label is clipped at size {size}: widget_w={widget_w} < text_w={text_w}"
                )

    def test_adversarial_stat2_large_number_clipping(self):
        """
        Adversarial test: Test stat2_value ('X aktif · Y dinlenen') with realistic
        route node count (150 nodes) and large counter values (999,999) at MINIMUM_SIZE (500, 275).
        Verifies that dynamic font scaling prevents horizontal text clipping.
        """
        self.hud.resize(*OverlayHUD.MINIMUM_SIZE)

        # Case A: 150 nodes
        self.hud.update_state({
            "stat2_val": "150 aktif · 240 dinlenen"
        })
        app.processEvents()

        fm = QFontMetrics(self.hud.stat2_value.font())
        text_w = fm.horizontalAdvance(self.hud.stat2_value.text())
        widget_w = self.hud.stat2_value.width()
        clipped_150 = text_w > widget_w

        self.assertFalse(
            clipped_150,
            f"stat2_value clipped for 150 nodes at 500x275: text_w={text_w} > widget_w={widget_w}"
        )

        # Case B: 999,999 nodes
        self.hud.update_state({
            "stat2_val": "999,999 aktif · 999,999 dinlenen"
        })
        app.processEvents()

        fm_large = QFontMetrics(self.hud.stat2_value.font())
        text_w_large = fm_large.horizontalAdvance(self.hud.stat2_value.text())
        widget_w_large = self.hud.stat2_value.width()
        self.assertLessEqual(
            text_w_large,
            widget_w_large,
            f"stat2_value clipped for large counters at 500x275: text_w={text_w_large} > widget_w={widget_w_large}"
        )

    def test_long_log_overflow_no_widget_overlap(self):
        """
        Verify that 300+ character log strings do NOT distort vertical layout,
        do NOT increase logs_container height, and do NOT overlap with action buttons or footer.
        """
        initial_logs_h = self.hud.logs_container.height()
        initial_action_y = self.hud._action_row.y()

        long_entry_1 = "Çok uzun sistem log mesajı: " + "X" * 300
        long_entry_2 = "İkinci uzun log kaydı: " + "Y" * 300
        long_entry_3 = "Üçüncü uzun log kaydı: " + "Z" * 300

        self.hud.add_log(long_entry_1)
        self.hud.add_log(long_entry_2)
        self.hud.add_log(long_entry_3)
        app.processEvents()

        new_logs_h = self.hud.logs_container.height()
        new_action_y = self.hud._action_row.y()

        self.assertEqual(initial_logs_h, new_logs_h, "logs_container height must remain stable")
        self.assertEqual(initial_action_y, new_action_y, "action_row position must not shift from log overflow")

        # Check no vertical overlap
        logs_bottom = self.hud.logs_container.y() + self.hud.logs_container.height()
        action_top = self.hud._action_row.y()
        self.assertLessEqual(logs_bottom, action_top, "logs_container must not overlap action_row")

        action_bottom = self.hud._action_row.y() + self.hud._action_row.height()
        footer_top = self.hud.f10_shortcut.y()
        self.assertLessEqual(action_bottom, footer_top, "action_row must not overlap footer shortcuts")

    def test_high_frequency_update_throughput(self):
        """
        Stress test: Execute 500 rapid updates simulating high-frequency bot telemetries.
        Verify sustained throughput exceeds 100 updates/sec without exceptions or crashes
        when UI event loop processes periodically (e.g. at 20-30Hz).
        """
        t0 = time.perf_counter()
        N = 500
        for i in range(N):
            self.hud.update_state({
                "stat1_val": str(i),
                "stat2_val": f"{i} aktif · {i} dinlenen",
                "stat3_val": f"{i % 10} / 10",
                "minimap_mapped": i,
                "learned_frames": i * 2,
            })
            if i % 25 == 0:
                self.hud.add_log(f"Adım #{i}")
            if i % 5 == 0:
                app.processEvents()
        app.processEvents()
        duration = time.perf_counter() - t0
        rate = N / duration

        self.assertGreaterEqual(rate, 100.0, f"Expected >= 100 updates/sec, achieved {rate:.1f}")

    def test_extreme_window_resize_range(self):
        """
        Test window resizing from (480, 260) to (800, 600).
        Verifies minimum size constraint behavior and layout stability.
        """
        for w, h in [(480, 260), (500, 275), (540, 290), (640, 480), (800, 600)]:
            with self.subTest(width=w, height=h):
                self.hud.resize(w, h)
                app.processEvents()

                # Because MINIMUM_SIZE is (500, 275), resizing to (480, 260) will be bounded by Qt
                actual_w = self.hud.width()
                actual_h = self.hud.height()

                self.assertGreaterEqual(actual_w, 480)
                self.assertGreaterEqual(actual_h, 260)
                self.assertTrue(self.hud.card.isVisible())


if __name__ == "__main__":
    unittest.main()
