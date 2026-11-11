# -*- coding: utf-8 -*-
"""Eski yatay HUD durur; ek düğmeler Başlat, Menü, Nokta, Durdur, Devam, Savaş, Toplam, Kapat."""

import unittest

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

from ui.f9_menu import F9Menu
from ui.overlay_hud import OverlayHUD


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


class OverlayHudLayoutTests(unittest.TestCase):
    # Both sizes the HUD geometry is verified at.
    TEST_SIZES = (OverlayHUD.DEFAULT_SIZE, OverlayHUD.MINIMUM_SIZE)

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        _isolate_disk_panic_history(self)
        self.hud = OverlayHUD()
        self.addCleanup(self.hud.deleteLater)

    def _resize_hud(self, w: int, h: int) -> None:
        """Resize the HUD and assert Qt honoured the request exactly.

        Every size used by these tests is >= MINIMUM_SIZE (720, 275), so
        a resize must land exactly. If a future minimum grows past a
        listed size, this fails loudly instead of silently testing a
        different (clamped) width under the requested size's subTest label.
        """
        self.hud.resize(w, h)
        if hasattr(self.hud, "card"):
            self.hud.card.resize(w, h)
        self.hud.show()
        self._app.processEvents()
        self.assertEqual(
            self.hud.width(), w,
            f"HUD width {self.hud.width()} != requested {w} "
            f"(minimum is {OverlayHUD.MINIMUM_SIZE})",
        )
        self.assertEqual(
            self.hud.height(), h,
            f"HUD height {self.hud.height()} != requested {h} "
            f"(minimum is {OverlayHUD.MINIMUM_SIZE})",
        )

    def _show_at(self, width: int, height: int) -> None:
        """Resize the HUD (and its card) to (width, height), show it and flush the event loop.

        Thin alias for _resize_hud: same resize/show/flush, plus the
        exact-size assertion (both sizes in TEST_SIZES are >= MINIMUM_SIZE,
        so the assertion holds for every call site).
        """
        self._resize_hud(width, height)

    def test_old_horizontal_panel_keeps_stats_and_logs(self) -> None:
        self.assertEqual(self.hud.width(), OverlayHUD.DEFAULT_SIZE[0])
        self.assertEqual(self.hud.height(), OverlayHUD.DEFAULT_SIZE[1])
        self.assertIsNotNone(self.hud.stats_container)
        self.assertEqual(len(self.hud.log_labels), 3)
        self.assertEqual(self.hud.badge_label.text(), "BEKLEMEDE")
        self.assertEqual(self.hud.minimap_mapped_label.text(), "Harita")
        self.assertTrue(self.hud.ai_learned_label.text().startswith("YZ"))

    def test_extra_buttons(self) -> None:
        self.assertEqual(self.hud.btn_run.text(), "Başlat")
        self.assertEqual(self.hud.btn_menu.text(), "Menü")
        self.assertEqual(self.hud.btn_node.text(), "Nokta")
        self.assertEqual(self.hud.btn_stop.text(), "Durdur")
        self.assertEqual(self.hud.btn_resume.text(), "Devam")
        self.assertEqual(self.hud.btn_combat.text(), "Savaş")
        self.assertEqual(self.hud.btn_gather.text(), "Toplam")
        self.assertEqual(self.hud.btn_quit.text(), "Kapat")
        self.assertEqual(self.hud.f10_shortcut._description_text, "Durdur")
        self.assertEqual(self.hud.f9_shortcut._description_text, "Menü")
        self.assertEqual(self.hud.f8_shortcut._description_text, "Nokta")

    def test_minimize_toggle_keeps_expand_button(self) -> None:
        self.assertFalse(self.hud.is_minimized)
        self.assertFalse(self.hud.card.isHidden())
        self.assertTrue(self.hud._mini_bar.isHidden())
        self.hud.btn_hud_minimize.click()
        self.assertTrue(self.hud.is_minimized)
        self.assertTrue(self.hud.card.isHidden())
        self.assertFalse(self.hud._mini_bar.isHidden())
        self.assertFalse(self.hud.btn_hud_expand.isHidden())
        self.assertEqual(self.hud.width(), OverlayHUD.MINIMIZED_SIZE[0])
        self.assertEqual(self.hud.height(), OverlayHUD.MINIMIZED_SIZE[1])
        self.hud.btn_hud_expand.click()
        self.assertFalse(self.hud.is_minimized)
        self.assertFalse(self.hud.card.isHidden())
        self.assertTrue(self.hud._mini_bar.isHidden())
        self.assertEqual(self.hud.width(), OverlayHUD.EXPANDED_SIZE[0])
        self.assertEqual(self.hud.height(), OverlayHUD.EXPANDED_SIZE[1])

    def test_buttons_emit(self) -> None:
        seen = []
        self.hud.f10_clicked.connect(lambda: seen.append("f10"))
        self.hud.f9_clicked.connect(lambda: seen.append("f9"))
        self.hud.f8_clicked.connect(lambda: seen.append("f8"))
        self.hud.quit_requested.connect(lambda: seen.append("quit"))
        self.hud.btn_run.click()
        self.hud.btn_menu.click()
        self.hud.btn_node.click()
        self.hud.btn_quit.click()
        self.assertEqual(seen, ["f10", "f9", "f8", "quit"])

        # Real F9 menu (ui/f9_menu.py), not a fake with a hand-rolled
        # _trigger_action: the HUD buttons must drive the menu the user
        # actually sees, and the menu must emit the real action signals.
        menu = F9Menu()
        self.addCleanup(menu.deleteLater)
        menu_actions = []
        menu.action_triggered.connect(menu_actions.append)
        self.hud.bind_f9_menu(menu)
        self.hud.btn_resume.click()
        self.hud.btn_combat.click()
        self.hud.btn_gather.click()
        self.assertEqual(menu_actions, ["resume", "combat", "route"])

    def test_state_snapshot_still_fills_old_rows(self) -> None:
        self.hud.update_state(
            {
                "state": "PAUSED",
                "mode": "explorer",
                "title": "Durum: Duraklatıldı",
                "subtitle": "Watchdog güvenli duraklatma",
                "gathered": 0,
                "region_route": "Altgard - Dredgion",
                "ready_count": 191,
                "resting_count": 0,
                "failed_in_row": 0,
                "learned_frames": 34,
                "minimap_mapped": 12,
            }
        )
        self.assertEqual(self.hud.badge_label.text(), "DURAKLATILDI")
        self.assertEqual(self.hud.mode_label.text(), "OTO KEŞİF")
        self.assertIn("191", self.hud.stat2_value.text())
        self.assertEqual(self.hud.ai_learned_label.text(), "YZ 34 Kare")
        # Merge-fix (O8): D31/F-0586 shows the live mapped count in the label.
        self.assertEqual(self.hud.minimap_mapped_label.text(), "Harita: 12")
        self.assertIn("12", self.hud.minimap_mapped_label.toolTip())

    def test_default_and_minimum_size_bounds(self) -> None:
        """Verifies HUD supports default (760, 290) and minimum (720, 275) geometry."""
        min_w, min_h = OverlayHUD.MINIMUM_SIZE
        # Exact equality: a <= check would also pass if no minimum were
        # defined at all (minimumWidth() == 0), so the minimum must be
        # asserted as the declared value, not as an upper bound.
        self.assertEqual(self.hud.minimumWidth(), min_w)
        self.assertEqual(self.hud.minimumHeight(), min_h)
        self.assertEqual(self.hud.minimumSize(), QSize(min_w, min_h))

        self.hud.resize(*OverlayHUD.DEFAULT_SIZE)
        self.hud.show()
        self._app.processEvents()

        card = getattr(self.hud, "card", self.hud)
        card_layout = card.layout()
        self.assertIsNotNone(card_layout)
        self.assertLessEqual(card_layout.minimumSize().height(), OverlayHUD.DEFAULT_SIZE[1])

    def test_action_buttons_no_text_clipping_across_sizes(self) -> None:
        """Verifies all 8 action buttons have zero text clipping at default and minimum sizes."""
        buttons = [
            self.hud.btn_run,
            self.hud.btn_menu,
            self.hud.btn_node,
            self.hud.btn_stop,
            self.hud.btn_resume,
            self.hud.btn_combat,
            self.hud.btn_gather,
            self.hud.btn_quit,
        ]
        # Both sizes are >= MINIMUM_SIZE (720, 275): (760, 290) is the
        # default, (720, 275) the minimum. The old (540, 290)/(500, 275)
        # pair was below the minimum, so Qt clamped every resize to
        # 720px wide and the subTest labels lied about the tested width.
        # TEST_SIZES == (DEFAULT_SIZE, MINIMUM_SIZE) == this pair.
        test_sizes = self.TEST_SIZES
        for w, h in test_sizes:
            self._resize_hud(w, h)

            for btn in buttons:
                with self.subTest(size=(w, h), button=btn.text()):
                    text = btn.text().strip()
                    self.assertTrue(len(text) > 0)
                    fm = btn.fontMetrics()
                    text_w = fm.horizontalAdvance(text)
                    alloc_w = btn.width()
                    self.assertLessEqual(
                        text_w,
                        alloc_w,
                        f"Button '{text}' text width ({text_w}px) exceeds width ({alloc_w}px) at ({w}, {h})",
                    )
                    self.assertGreaterEqual(
                        alloc_w - text_w,
                        2,
                        f"Button '{text}' has insufficient clearance ({alloc_w - text_w}px) at ({w}, {h})",
                    )

    def test_stats_values_and_logs_layout_geometry(self) -> None:
        """Verifies stats container values height and non-overlapping log lines."""
        test_sizes = self.TEST_SIZES
        for w, h in test_sizes:
            with self.subTest(size=(w, h)):
                self._resize_hud(w, h)

                # Stats values height check
                for stat_lbl in (self.hud.stat1_value, self.hud.stat2_value, self.hud.stat3_value):
                    self.assertGreaterEqual(
                        stat_lbl.height(),
                        24,
                        f"Stat label height {stat_lbl.height()} < 24px at size ({w}, {h})",
                    )

                # Log lines independent clearance check
                labels = self.hud.log_labels
                self.assertEqual(len(labels), 3)
                for i in range(len(labels) - 1):
                    l1, l2 = labels[i], labels[i + 1]
                    self.assertLessEqual(
                        l1.geometry().bottom(),
                        l2.geometry().top() + 1,
                        f"Log label {i} overlaps label {i+1} at size ({w}, {h})",
                    )

    def test_harvest_badge_no_clipping_with_long_state_and_mode(self) -> None:
        """Verifies harvest_badge_label has zero text clipping under long state and mode.

        NOTE: comparing the badge's own width against its text width cannot
        detect clipping here — update_state() sets the badge's minimumWidth
        from the text itself (advance + 14), so widget_w >= text_w is true
        by construction. The falsifiable checks are geometric: the badge
        (and its full text, starting at the badge's x) must stay inside the
        card, and it must not overlap any sibling in the header row.
        """
        siblings = [
            self.hud.badge_label,
            self.hud.mode_label,
            self.hud.elapsed_label,
            self.hud.btn_hud_minimize,
        ]
        test_sizes = self.TEST_SIZES
        for w, h in test_sizes:
            with self.subTest(size=(w, h)):
                self._resize_hud(w, h)
                self.hud.update_state({
                    "state": "DURAKLATILDI",
                    "mode": "MANUEL KAYIT",
                    "harvest_stage": 2,
                    "elapsed": "12:34:56",
                })
                self.hud.show()
                self._app.processEvents()

                badge = self.hud.harvest_badge_label
                fm = badge.fontMetrics()
                text_w = fm.horizontalAdvance(badge.text())
                card_w = self.hud.card.width()

                self.assertLessEqual(
                    badge.geometry().right(),
                    card_w,
                    f"harvest_badge_label overflows card at ({w}, {h}): "
                    f"right={badge.geometry().right()} > card_w={card_w}",
                )
                self.assertLessEqual(
                    badge.x() + text_w,
                    card_w,
                    f"harvest_badge_label text overflows card at ({w}, {h}): "
                    f"x + text_w={badge.x() + text_w} > card_w={card_w}",
                )
                for other in siblings:
                    self.assertFalse(
                        badge.geometry().intersects(other.geometry()),
                        f"harvest_badge_label overlaps {other} at ({w}, {h})",
                    )

    def test_stat2_value_dynamic_font_scaling_no_clipping(self) -> None:
        """Verifies stat2_value dynamically scales down font and never clips on long/large counters."""
        test_cases = [
            ("150 aktif · 240 dinlenen", 14),
            ("999,999 aktif · 999,999 dinlenen", 13),
        ]
        test_sizes = self.TEST_SIZES
        for w, h in test_sizes:
            self._resize_hud(w, h)
            for text_val, max_expected_font_px in test_cases:
                with self.subTest(size=(w, h), text=text_val):
                    self.hud.update_state({"stat2_val": text_val})
                    self._app.processEvents()

                    fm = self.hud.stat2_value.fontMetrics()
                    text_w = fm.horizontalAdvance(self.hud.stat2_value.text())
                    widget_w = self.hud.stat2_value.width()
                    font_px = self.hud.stat2_value.font().pixelSize()

                    self.assertGreaterEqual(
                        widget_w,
                        text_w,
                        f"stat2_value clipped for '{text_val}' at ({w}, {h}): widget_w={widget_w} < text_w={text_w}",
                    )
                    self.assertLessEqual(
                        font_px,
                        max_expected_font_px,
                        f"stat2_value font size {font_px}px did not scale down below {max_expected_font_px}px",
                    )


if __name__ == "__main__":
    unittest.main()
