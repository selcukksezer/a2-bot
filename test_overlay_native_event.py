# -*- coding: utf-8 -*-
"""OverlayHUD.nativeEvent and RadarProjectionOverlay.nativeEvent must tolerate a Qt result pointer of None."""

import unittest

from PyQt6.QtWidgets import QApplication, QWidget

from ui.overlay_hud import OverlayHUD, RadarProjectionOverlay


class NativeEventNoneResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _assert_none_result_tolerant(self, widget) -> None:
        """Drive `widget.nativeEvent` through the three Qt result-pointer shapes.

        QWidget.nativeEvent is monkeypatched in turn to return
        (False, None), None, and (True, 7); the override must normalise
        all three to a (bool, int) pair instead of raising, and must pass
        a handled result through unchanged. The original is restored
        afterwards so later tests see the real QWidget.nativeEvent.
        """
        original = QWidget.nativeEvent

        def none_result(_self, _event_type, _message):
            return False, None

        def handled_result(_self, _event_type, _message):
            return True, 7

        try:
            QWidget.nativeEvent = none_result
            self.assertEqual(widget.nativeEvent(b"", 0), (False, 0))
            QWidget.nativeEvent = lambda *_args: None
            self.assertEqual(widget.nativeEvent(b"", 0), (False, 0))
            QWidget.nativeEvent = handled_result
            self.assertEqual(widget.nativeEvent(b"", 0), (True, 7))
        finally:
            QWidget.nativeEvent = original

    def test_none_result_pointer_does_not_raise(self) -> None:
        # Real HUD construction (not a probe that skips __init__): the
        # OverlayHUD.nativeEvent path runs against the fully built widget.
        hud = OverlayHUD()
        self.addCleanup(hud.deleteLater)
        self._assert_none_result_tolerant(hud)

    def test_radar_projection_overlay_none_result_pointer_does_not_raise(self) -> None:
        # Second nativeEvent in ui/overlay_hud.py: RadarProjectionOverlay has
        # its own override (WM_MOUSEACTIVATE / MA_NOACTIVATE on win32, then
        # the same None-tolerant super() fallback). It was never exercised
        # because the old probe only subclassed OverlayHUD.
        overlay = RadarProjectionOverlay()
        self.addCleanup(overlay.deleteLater)
        self._assert_none_result_tolerant(overlay)


if __name__ == "__main__":
    unittest.main()
