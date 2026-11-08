# -*- coding: utf-8 -*-
"""Ctrl+Alt+F9 rows are short Turkish buttons. Long sentences stay in tooltips."""

import inspect
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.f9_menu import F9Menu

EXPECTED = (
    ("resume", "Devam Et"),
    ("combat", "Savaş"),
    ("route", "Toplama"),
    ("auto_record", "Oto-Kayıt"),
    ("manual_record", "Konum Ekle"),
    ("explorer", "Keşif"),
    ("minimap_auto_record", "Minimap"),
    ("active_route", "Rota"),
    ("radar", "Radar"),
    ("analytics", "İstatistik"),
    ("calibrate", "Kalibrasyon"),
    ("yolo_factory", "Eğit"),
    ("settings", "Ayarlar"),
    ("open_log", "Günlük"),
    ("guide", "Kılavuz"),
    ("copy_support", "Tanılama"),
    ("exit", "Çıkış"),
)

_LONG_MARKERS = (
    "Canavarları Tab ile seçer",
    "Toplama Modu (Gathering)",
    "Savaşma Modu (Combat):",
)


class F9MenuLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_visible_labels_are_short_and_sentences_are_tooltips(self) -> None:
        menu = F9Menu(active_mode="route", harvests_per_node=3)
        self.addCleanup(menu.deleteLater)

        visible = [(action_id, row.label_text) for action_id, row in menu._rows.items()]
        self.assertEqual(visible, list(EXPECTED))
        for _action_id, label in visible:
            for marker in _LONG_MARKERS:
                self.assertNotIn(marker, label)
            self.assertNotIn(":", label)

        combat = menu._rows["combat"]
        route = menu._rows["route"]
        self.assertIn("Canavarları Tab ile seçer", combat.toolTip())
        self.assertIn("Toplama Modu (Gathering)", route.toolTip())
        self.assertEqual(combat.label_text, "Savaş")
        self.assertEqual(route.label_text, "Toplama")
        for row in menu._rows.values():
            self.assertFalse(getattr(row, "badge_key", ""))

        self.assertEqual(menu.harvest_badge_label.text(), "3x")
        self.assertEqual(menu._menu_chord_label, "Ctrl+Alt+F9")
        self.assertEqual(menu._node_chord_label, "Ctrl+Alt+F8")
        self.assertEqual(menu._pause_chord_label, "Ctrl+Alt+F10")

        menu.set_active_state(active_region="Altgard", active_route="All", paused=True)
        self.assertEqual(menu.header_subtitle_label.text(), "Duraklatıldı · Altgard · All")
        menu.set_active_state(paused=False)
        self.assertEqual(menu.header_subtitle_label.text(), "Çalışıyor · Altgard · All")

        menu.set_active_mode("combat")
        menu.toggle_minimap_auto_record()
        menu.apply_hotkey_labels("Ctrl+Alt+F9", "Ctrl+Alt+F8", "Ctrl+Alt+F10")
        self.assertEqual(
            [(action_id, row.label_text) for action_id, row in menu._rows.items()],
            list(EXPECTED),
        )
        self.assertIn("Ctrl+Alt+F8", menu._rows["manual_record"].toolTip())
        self.assertEqual(menu._rows["manual_record"].label_text, "Konum Ekle")
        self.assertTrue(menu._rows["combat"].radio_checked)

        init_src = inspect.getsource(F9Menu._init_ui)
        refresh_src = inspect.getsource(F9Menu._refresh_labels)
        detail_src = inspect.getsource(F9Menu._detail_for)
        key_src = inspect.getsource(F9Menu.keyPressEvent)
        for marker in _LONG_MARKERS:
            self.assertNotIn(marker, init_src)
            self.assertNotIn(marker, refresh_src)
            self.assertIn(marker, detail_src)
        self.assertNotIn("badge_key", init_src)
        self.assertIn("SHORT_LABELS", init_src)
        self.assertIn("SHORT_LABELS", refresh_src)
        self.assertIn('self._trigger_action("combat")', key_src)
        self.assertIn('self._trigger_action("resume")', key_src)
        self.assertIn("Key_C", key_src)


if __name__ == "__main__":
    unittest.main()
