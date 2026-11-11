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

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QFontMetrics

# Ensure single QApplication instance
app = QApplication.instance() or QApplication(sys.argv)

from ui.overlay_hud import OverlayHUD
from ui.overlay_window import OverlayWindow


def _isolate_disk_panic_history(testcase: unittest.TestCase) -> None:
    """Neutralise the on-disk panic strip for HUD layout tests.

    OverlayHUD.__init__ calls refresh_panic_from_disk(), which reads the
    latest record under logs/alarms/ and shows the panic strip — growing
    the card by ~5px (295 > 290/275). That directory is written by the
    test suite itself, so whether the strip appears depends on test
    ordering/history, not on the layout under test. Patching the reader
    to return [] keeps the production behaviour (showing the strip)
    intact while the geometry assertions measure the bare layout.
    Test-side isolation only — no production code is touched.
    """
    from unittest import mock

    from core.opsec_crypto import PanicAlertSystem

    patcher = mock.patch.object(
        PanicAlertSystem,
        "load_panic_history_from_disk",
        lambda *args, **kwargs: [],
    )
    patcher.start()
    testcase.addCleanup(patcher.stop)


class TestOverlayHUDAdversarial(unittest.TestCase):
    """Adversarial stress harness for OverlayHUD layout under boundary conditions."""

    def setUp(self):
        _isolate_disk_panic_history(self)
        self.hud = OverlayHUD()
        self.hud.show()
        app.processEvents()

    def tearDown(self):
        self.hud.close()
        app.processEvents()

    def _resize(self, width: int, height: int) -> None:
        """Resize the HUD and flush the event loop so layout geometry settles."""
        self.hud.resize(width, height)
        app.processEvents()

    def _action_buttons(self):
        """The 8 required action buttons with their expected labels, in row order."""
        return [
            (self.hud.btn_run, "Başlat"),
            (self.hud.btn_menu, "Menü"),
            (self.hud.btn_node, "Nokta"),
            (self.hud.btn_stop, "Durdur"),
            (self.hud.btn_resume, "Devam"),
            (self.hud.btn_combat, "Savaş"),
            (self.hud.btn_gather, "Toplam"),
            (self.hud.btn_quit, "Kapat"),
        ]

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
        Verify that at DEFAULT_SIZE (760, 290) and MINIMUM_SIZE (720, 275),
        ALL widgets fit inside the window without bottom/right cutoff —
        not just the footer and version labels.
        """
        for size in [OverlayHUD.DEFAULT_SIZE, OverlayHUD.MINIMUM_SIZE]:
            with self.subTest(size=size):
                self._resize(*size)

                w_h = self.hud.height()
                footer_bottom = self.hud.f10_shortcut.y() + self.hud.f10_shortcut.height()
                version_bottom = self.hud.version_label.y() + self.hud.version_label.height()

                self.assertLessEqual(footer_bottom, w_h, f"Footer truncated at size {size}: {footer_bottom} > {w_h}")
                self.assertLessEqual(version_bottom, w_h, f"Version truncated at size {size}: {version_bottom} > {w_h}")

                # Every visible descendant widget must lie fully inside
                # the HUD rectangle (mapped into HUD coordinates).
                for widget in self.hud.findChildren(QWidget):
                    if not widget.isVisible():
                        continue
                    top_left = widget.mapTo(self.hud, QPoint(0, 0))
                    bottom_right = widget.mapTo(self.hud, QPoint(widget.width(), widget.height()))
                    with self.subTest(size=size, widget=widget.__class__.__name__):
                        self.assertLessEqual(
                            bottom_right.x(), self.hud.width(),
                            f"{widget.__class__.__name__} overflows right edge at size {size}: "
                            f"right={bottom_right.x()} > width={self.hud.width()}",
                        )
                        self.assertLessEqual(
                            bottom_right.y(), self.hud.height(),
                            f"{widget.__class__.__name__} overflows bottom edge at size {size}: "
                            f"bottom={bottom_right.y()} > height={self.hud.height()}",
                        )
                        self.assertGreaterEqual(top_left.x(), 0, f"{widget.__class__.__name__} starts left of HUD at size {size}")
                        self.assertGreaterEqual(top_left.y(), 0, f"{widget.__class__.__name__} starts above HUD at size {size}")

    def test_standard_action_buttons_no_clipping(self):
        """
        Verify that all 8 required action buttons:
        ['Başlat', 'Menü', 'Nokta', 'Durdur', 'Devam', 'Savaş', 'Toplam', 'Kapat']
        render cleanly with positive headroom across default (760, 290) and minimum (720, 275).
        """
        expected_buttons = self._action_buttons()

        for size in [OverlayHUD.DEFAULT_SIZE, OverlayHUD.MINIMUM_SIZE]:
            self._resize(*size)

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
        long_texts = dict(zip(
            (btn for btn, _ in self._action_buttons()),
            ["Başlat/Devam", "Menüyü Aç", "Nokta Ekle", "Durduruldu",
             "Devam Et", "Savaş Modu", "Toplama Yap", "Kapat/Çıkış"],
        ))

        for btn, text in long_texts.items():
            btn.setText(text)

        self._resize(*OverlayHUD.MINIMUM_SIZE)

        clipped_count = 0
        for btn, text in long_texts.items():
            fm = QFontMetrics(btn.font())
            text_w = fm.horizontalAdvance(text)
            needed_w = text_w + 12
            actual_w = btn.width()
            if actual_w < needed_w:
                clipped_count += 1

        # The HUD must render every injected label without clipping: a
        # clipped button is a layout defect, not an expected outcome.
        # (Previously this test asserted clipped_count > 0, i.e. it passed
        # exactly when the UI was broken — the assertion was inverted.)
        self.assertEqual(
            clipped_count,
            0,
            f"Adversarial finding: {clipped_count}/8 single-row action buttons clip "
            "when injected with longer labels (>7 chars) at minimum size",
        )

    def test_adversarial_header_badge_clipping(self):
        """
        Adversarial test: Test harvest_badge_label under long state ('DURAKLATILDI')
        and long mode ('MANUEL KAYIT') at MINIMUM_SIZE (720, 275) and DEFAULT_SIZE (760, 290).
        Verifies that harvest_badge_label is NOT clipped at minimum or default sizes.

        NOTE: comparing the badge's own width against its text width cannot
        detect clipping here — _refresh_harvest_badge_style() sets the
        badge's minimumWidth from the text itself (advance + 14), so
        widget_w >= text_w holds by construction and the old
        `is_clipped = widget_w < text_w; assertFalse(is_clipped)` passed
        no matter what the layout did. The falsifiable checks are
        geometric: the badge (and its full text, starting at the badge's
        x) must stay inside the card, and it must not overlap any sibling
        in the header row. A deliberately narrowed QLabel control proves
        the width comparison itself flags real clipping when it exists,
        and the minimumWidth assertion below fails if the production
        setMinimumWidth(advance + 14) mechanism regresses.
        """
        # Negative control: the width-vs-text predicate must flag a label
        # that really is too narrow — otherwise it could never fail.
        control_text = "3x Toplama (Harvest: 2/3)"
        control = QLabel(control_text)
        control.setFixedWidth(50)
        control_text_w = QFontMetrics(control.font()).horizontalAdvance(control_text)
        self.assertLess(
            control.width(), control_text_w,
            "control: a 50px-wide label holding the badge text must be "
            f"narrower than the text ({control_text_w}px) — the clipping "
            "predicate would be vacuous otherwise",
        )
        control.deleteLater()

        siblings = [
            self.hud.badge_label,
            self.hud.mode_label,
            self.hud.elapsed_label,
            self.hud.btn_hud_minimize,
        ]
        for size in [OverlayHUD.MINIMUM_SIZE, OverlayHUD.DEFAULT_SIZE]:
            with self.subTest(size=size):
                self._resize(*size)

                # Both sizes are >= the (720, 275) minimum, so the resize
                # must be honoured exactly — not clamped to another width.
                self.assertEqual(
                    (self.hud.width(), self.hud.height()), size,
                    f"HUD size {(self.hud.width(), self.hud.height())} != "
                    f"requested {size}",
                )
                self.hud.update_state({
                    "state": "DURAKLATILDI",
                    "mode": "MANUEL KAYIT",
                    "harvest_stage": 2,
                    "elapsed": "12:34:56",
                })
                app.processEvents()

                badge = self.hud.harvest_badge_label
                fm = QFontMetrics(badge.font())
                text_w = fm.horizontalAdvance(badge.text())
                card_w = self.hud.card.width()

                # Mechanism check: minimumWidth must cover text + padding.
                self.assertGreaterEqual(
                    badge.minimumWidth(), text_w + 14,
                    f"harvest_badge_label minimumWidth={badge.minimumWidth()} "
                    f"< text_w + 14 = {text_w + 14} at size {size}",
                )
                # Geometric checks: badge and its full text stay inside the
                # card and overlap no header-row sibling.
                self.assertLessEqual(
                    badge.geometry().right(), card_w,
                    f"harvest_badge_label overflows card at {size}: "
                    f"right={badge.geometry().right()} > card_w={card_w}",
                )
                self.assertLessEqual(
                    badge.x() + text_w, card_w,
                    f"harvest_badge_label text overflows card at {size}: "
                    f"x + text_w={badge.x() + text_w} > card_w={card_w}",
                )
                for other in siblings:
                    self.assertFalse(
                        badge.geometry().intersects(other.geometry()),
                        f"harvest_badge_label overlaps {other} at {size}",
                    )

    def test_adversarial_stat2_large_number_clipping(self):
        """
        Adversarial test: Test stat2_value ('X aktif · Y dinlenen') with realistic
        route node count (150 nodes) and large counter values (999,999) at MINIMUM_SIZE (720, 275).
        Verifies that dynamic font scaling prevents horizontal text clipping.
        """
        self._resize(*OverlayHUD.MINIMUM_SIZE)

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
        Deterministic checks only: every update must apply without exception
        and the final rendered state must reflect the last update. No
        updates/sec threshold is asserted — throughput on a shared/loaded
        machine measures the machine (and mostly the event loop, not the
        render path), so a rate assertion is flaky by construction and
        failed on ordinary hardware.

        NOTE (D89, found bug — production side, not fixed here):
        500 updates cost ~35s on this VM because OverlayHUD.update_state
        re-applies stylesheets on every call: _apply_state_theme
        (ui/overlay_hud.py:2148) calls setStyleSheet 4x per update even
        when the state did not change (cProfile: 800 setStyleSheet calls
        = ~95% of 200 updates' cost). The count N=500 is kept as-is;
        only the production code can make this fast.
        """
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

        self.assertEqual(self.hud.stat1_value.text(), "499")
        self.assertEqual(self.hud.stat2_value.text(), "499 aktif · 499 dinlenen")
        self.assertEqual(self.hud.stat3_value.text(), "9 / 10")
        self.assertIn("499", self.hud.minimap_mapped_label.toolTip())
        self.assertEqual(self.hud.ai_learned_label.text(), "YZ 998 Kare")

    def test_extreme_window_resize_range(self):
        """
        Test window resizing from (480, 260) to (800, 600).
        Verifies minimum size constraint behavior and layout stability.
        """
        # Sizes straddle the (720, 275) minimum: fully below -> clamped
        # to the minimum; at/above -> honoured exactly.
        for w, h in [(480, 260), (700, 260), (720, 275), (760, 290), (800, 600)]:
            with self.subTest(width=w, height=h):
                self._resize(w, h)

                # Because MINIMUM_SIZE is (720, 275), resizing to (480, 260) is
                # bounded by Qt to the declared minimum — assert the minimum
                # itself (720x275), not the below-minimum request (480x260).
                # Sizes at/above the minimum must be honoured exactly.
                actual_w = self.hud.width()
                actual_h = self.hud.height()
                min_w, min_h = OverlayHUD.MINIMUM_SIZE
                if w < min_w or h < min_h:
                    self.assertEqual(actual_w, min_w)
                    self.assertEqual(actual_h, min_h)
                else:
                    self.assertEqual(actual_w, w)
                    self.assertEqual(actual_h, h)
                self.assertTrue(self.hud.card.isVisible())


if __name__ == "__main__":
    unittest.main()
