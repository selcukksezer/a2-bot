# -*- coding: utf-8 -*-
"""Ctrl+Alt+F9 rows are short Turkish buttons. Long sentences stay in tooltips.

Labels are asserted through rendered behaviour (visible row widgets found
via Qt's public findChildren API, the public window title, and tooltips)
and against the public F9Menu.SHORT_LABELS mapping — never by reading
production source text (inspect.getsource) or private menu attributes.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

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


def _visible_rows(menu):
    """Action rows as rendered widgets, via Qt's public findChildren API.

    Rows are direct children of the menu carrying the row widget's public
    ``action_id``/``label_text`` attributes, in creation order.
    """
    return [
        w
        for w in menu.findChildren(QWidget)
        if w.parent() is menu and hasattr(w, "action_id") and hasattr(w, "label_text")
    ]


def _row(menu, action_id):
    for row in _visible_rows(menu):
        if row.action_id == action_id:
            return row
    raise AssertionError(f"row not rendered: {action_id}")


class F9MenuLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_visible_labels_are_short_and_sentences_are_tooltips(self) -> None:
        menu = F9Menu(active_mode="route", harvests_per_node=3)
        self.addCleanup(menu.deleteLater)

        # Rendered labels come from the public SHORT_LABELS mapping.
        self.assertEqual(F9Menu.SHORT_LABELS, dict(EXPECTED))
        visible = [(row.action_id, row.label_text) for row in _visible_rows(menu)]
        self.assertEqual(visible, list(EXPECTED))
        for _action_id, label in visible:
            for marker in _LONG_MARKERS:
                self.assertNotIn(marker, label)
            self.assertNotIn(":", label)

        combat = _row(menu, "combat")
        route = _row(menu, "route")
        self.assertIn("Canavarları Tab ile seçer", combat.toolTip())
        self.assertIn("Toplama Modu (Gathering)", route.toolTip())
        self.assertEqual(combat.label_text, "Savaş")
        self.assertEqual(route.label_text, "Toplama")
        for row in _visible_rows(menu):
            self.assertFalse(getattr(row, "badge_key", ""))

        self.assertEqual(menu.harvest_badge_label.text(), "3x")
        # Hotkey chords are asserted on public surfaces, not private attrs:
        # window title / header title carry the menu chord, the manual-record
        # tooltip carries the node chord.
        self.assertEqual(menu.windowTitle(), "Ctrl+Alt+F9 Hızlı Kontrol Menüsü")
        self.assertEqual(menu.header_title_label.text(), "Ctrl+Alt+F9 Hızlı Kontrol Menüsü")
        self.assertIn("Ctrl+Alt+F8", _row(menu, "manual_record").toolTip())

        menu.set_active_state(active_region="Altgard", active_route="All", paused=True)
        self.assertEqual(menu.header_subtitle_label.text(), "Duraklatıldı · Altgard · All")
        menu.set_active_state(paused=False)
        self.assertEqual(menu.header_subtitle_label.text(), "Çalışıyor · Altgard · All")

        menu.set_active_mode("combat")
        menu.toggle_minimap_auto_record()
        menu.apply_hotkey_labels("Ctrl+Alt+F9", "Ctrl+Alt+F8", "Ctrl+Alt+F10")
        self.assertEqual(
            [(row.action_id, row.label_text) for row in _visible_rows(menu)],
            list(EXPECTED),
        )
        self.assertIn("Ctrl+Alt+F8", _row(menu, "manual_record").toolTip())
        self.assertEqual(_row(menu, "manual_record").label_text, "Konum Ekle")
        self.assertTrue(_row(menu, "combat").radio_checked)
        self.assertEqual(menu.windowTitle(), "Ctrl+Alt+F9 Hızlı Kontrol Menüsü")

    def test_key_shortcuts_drive_actions(self) -> None:
        """Single-key shortcuts trigger the matching action (no source inspection).

        C -> combat mode, M -> route mode, E -> explorer mode, R -> resume.
        """
        menu = F9Menu(active_mode="route", harvests_per_node=3)
        self.addCleanup(menu.deleteLater)
        menu.show()
        menu.activateWindow()
        QApplication.processEvents()
        self.assertTrue(menu.isActiveWindow())

        triggered = []
        menu.action_triggered.connect(lambda action_id: triggered.append(action_id))

        QTest.keyClick(menu, Qt.Key.Key_C)
        self.assertEqual(menu.get_active_mode(), "combat")
        self.assertIn("combat", triggered)

        QTest.keyClick(menu, Qt.Key.Key_M)
        self.assertEqual(menu.get_active_mode(), "route")

        QTest.keyClick(menu, Qt.Key.Key_E)
        self.assertEqual(menu.get_active_mode(), "explorer")

        QTest.keyClick(menu, Qt.Key.Key_R)
        self.assertIn("resume", triggered)

        # Long explanations stay in tooltips even after keyboard use.
        self.assertIn("Canavarları Tab ile seçer", _row(menu, "combat").toolTip())


if __name__ == "__main__":
    unittest.main()
