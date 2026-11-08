# -*- coding: utf-8 -*-
"""OverlayHUD.nativeEvent must tolerate a Qt result pointer of None."""

import unittest

from PyQt6.QtWidgets import QApplication, QWidget

from ui.overlay_hud import OverlayHUD


class _Probe(OverlayHUD):
    """Skip HUD construction; only the nativeEvent return path is under test."""

    def __init__(self) -> None:
        QWidget.__init__(self)


class NativeEventNoneResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_none_result_pointer_does_not_raise(self) -> None:
        probe = _Probe()
        original = QWidget.nativeEvent

        def none_result(_self, _event_type, _message):
            return False, None

        def handled_result(_self, _event_type, _message):
            return True, 7

        try:
            QWidget.nativeEvent = none_result
            self.assertEqual(probe.nativeEvent(b"", 0), (False, 0))
            QWidget.nativeEvent = lambda *_args: None
            self.assertEqual(probe.nativeEvent(b"", 0), (False, 0))
            QWidget.nativeEvent = handled_result
            self.assertEqual(probe.nativeEvent(b"", 0), (True, 7))
        finally:
            QWidget.nativeEvent = original
            probe.deleteLater()


if __name__ == "__main__":
    unittest.main()
