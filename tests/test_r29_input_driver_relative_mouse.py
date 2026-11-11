# -*- coding: utf-8 -*-
"""
R29 — F-0452 + F-1693 (Karar 4) doğrulama testleri.

- F-1693: InputController.mouse_move_relative Interception tabanlı göreli
  fare hareketidir; donanım yoksa False döner, yumuşak SendInput'a asla
  düşmez. InterceptionBridge.send_mouse_relative gerçek Interception
  mouse stroke'unu (state=0, flags=MOVE_RELATIVE=0, x/y=delta) gönderir.
- F-0452: yalnızca yazılan _input_refused / _last_logged_test_mode /
  _emitted_move_points artık get_input_controller_diagnostic_snapshot
  probundan okunur (silinmedi, proba bağlandı).
"""

import ctypes
import unittest
from unittest.mock import MagicMock, patch

from core.input_driver import (
    InputController,
    InterceptionBridge,
    InterceptionMouseStroke,
)


def _make_controller(*, test_mode=False, interception=None):
    ctrl = InputController(test_mode=test_mode, use_faker_input=False, use_interception=True)
    ctrl._faker = None
    ctrl._serial_hid = None
    ctrl._interception = interception
    ctrl.ensure_serial_hid = MagicMock(return_value=False)
    return ctrl


class TestR29MouseMoveRelative(unittest.TestCase):
    def test_no_hardware_returns_false_and_never_soft(self):
        """Donanım yoksa False; soft fare yolları hiç çağrılmaz."""
        ctrl = _make_controller(interception=None)
        ctrl._sendinput_mouse_move = MagicMock()
        ctrl._sendinput_mouse_button = MagicMock()
        with patch.object(ctrl, "nudge_camera", wraps=ctrl.nudge_camera) as mock_nudge:
            self.assertFalse(ctrl.mouse_move_relative(12, -7))
            mock_nudge.assert_not_called()
        ctrl._sendinput_mouse_move.assert_not_called()
        ctrl._sendinput_mouse_button.assert_not_called()

    def test_mock_interception_success(self):
        """Interception köprüsü başarılıyken True ve (dx, dy) iletilir."""
        mock_bridge = MagicMock()
        mock_bridge.available = True
        mock_bridge.send_mouse_relative.return_value = True
        ctrl = _make_controller(interception=mock_bridge)
        self.assertTrue(ctrl.mouse_move_relative(12, -7))
        mock_bridge.send_mouse_relative.assert_called_once_with(12, -7)

    def test_mock_interception_failure_propagates(self):
        """Köprü False dönerse denetleyici de False döner (başarı taklidi yok)."""
        mock_bridge = MagicMock()
        mock_bridge.available = True
        mock_bridge.send_mouse_relative.return_value = False
        ctrl = _make_controller(interception=mock_bridge)
        self.assertFalse(ctrl.mouse_move_relative(5, 5))

    def test_test_mode_simulates_without_bridge(self):
        """test_mode'da fiziksel gönderim yapılmadan True döner."""
        mock_bridge = MagicMock()
        mock_bridge.available = True
        ctrl = _make_controller(test_mode=True, interception=mock_bridge)
        self.assertTrue(ctrl.mouse_move_relative(9, 9))
        mock_bridge.send_mouse_relative.assert_not_called()

    def test_zero_delta_is_noop(self):
        """(0, 0) köprüye hiç gitmeden True döner."""
        mock_bridge = MagicMock()
        mock_bridge.available = True
        ctrl = _make_controller(interception=mock_bridge)
        self.assertTrue(ctrl.mouse_move_relative(0, 0))
        mock_bridge.send_mouse_relative.assert_not_called()

    def test_real_bridge_without_driver_returns_false(self):
        """Gerçek köprü (sürücü/DLL yok) her zaman False döner."""
        bridge = InterceptionBridge()
        self.assertFalse(bridge.available)
        self.assertFalse(bridge.send_mouse_relative(5, 5))
        self.assertEqual(bridge._mouse_device, 11)

    def test_mouse_stroke_layout(self):
        """InterceptionMouseStroke C düzeni: state/flags/rolling/x/y/info."""
        self.assertEqual(ctypes.sizeof(InterceptionMouseStroke), 20)
        stroke = InterceptionMouseStroke(0, 0, 0, 12, -7, 0)
        self.assertEqual((stroke.state, stroke.flags, stroke.rolling), (0, 0, 0))
        self.assertEqual((stroke.x, stroke.y, stroke.information), (12, -7, 0))

    def test_bridge_send_builds_relative_stroke(self):
        """Köprü interception_send'e göreli stroke gönderir (state=0, flags=0)."""
        seen = {}

        def fake_send(ctx, device, stroke_ptr, nstroke):
            stroke = ctypes.cast(stroke_ptr, ctypes.POINTER(InterceptionMouseStroke)).contents
            seen["device"] = device
            seen["stroke"] = (stroke.state, stroke.flags, stroke.rolling, stroke.x, stroke.y)
            seen["nstroke"] = nstroke
            return 1

        bridge = InterceptionBridge.__new__(InterceptionBridge)
        bridge.available = True
        bridge._dll = MagicMock()
        bridge._dll.interception_send.side_effect = fake_send
        bridge._ctx = object()
        bridge._mouse_device = 11
        self.assertTrue(bridge.send_mouse_relative(3, -4))
        self.assertEqual(seen["device"], 11)
        self.assertEqual(seen["stroke"], (0, 0, 0, 3, -4))
        self.assertEqual(seen["nstroke"], 1)

    def test_bridge_send_broadcasts_to_next_mouse_device(self):
        """Tercih edilen fare reddederse diğer fare slotları (11..20) denenir."""
        calls = []

        def fake_send(ctx, device, stroke_ptr, nstroke):
            calls.append(device)
            return 1 if device == 13 else 0

        bridge = InterceptionBridge.__new__(InterceptionBridge)
        bridge.available = True
        bridge._dll = MagicMock()
        bridge._dll.interception_send.side_effect = fake_send
        bridge._ctx = object()
        bridge._mouse_device = 11
        self.assertTrue(bridge.send_mouse_relative(2, 2))
        self.assertEqual(calls, [11, 12, 13])
        self.assertEqual(bridge._mouse_device, 13)


class TestR29F0452DiagnosticProbe(unittest.TestCase):
    def test_snapshot_exposes_write_only_fields(self):
        """Üç alan da tanılama anlık görüntüsünden okunur."""
        ctrl = InputController(test_mode=True)
        snap = ctrl.get_input_controller_diagnostic_snapshot()
        self.assertIn("input_refused", snap)
        self.assertIn("last_logged_test_mode", snap)
        self.assertIn("emitted_move_points", snap)
        self.assertFalse(snap["input_refused"])
        self.assertTrue(snap["last_logged_test_mode"])
        self.assertEqual(snap["emitted_move_points"], [])

    def test_hold_refusal_visible_in_snapshot(self):
        """Reddedilen hareket tutma girişimi snapshot'ta görünür, resume temizler."""
        ctrl = _make_controller(interception=None)
        self.assertFalse(ctrl.hold_movement_key(""))
        self.assertTrue(ctrl.get_input_controller_diagnostic_snapshot()["input_refused"])
        ctrl.resume_motion()
        self.assertFalse(ctrl.get_input_controller_diagnostic_snapshot()["input_refused"])

    def test_smooth_move_populates_emitted_points(self):
        """smooth_move_to'nun ürettiği Bezier noktaları snapshot'ta görünür."""
        ctrl = InputController(test_mode=True)
        self.assertEqual(ctrl.smooth_move_to(100, 50), (100, 50))
        points = ctrl.get_input_controller_diagnostic_snapshot()["emitted_move_points"]
        self.assertGreater(len(points), 0)
        self.assertEqual(points[-1], (100, 50))

    def test_set_test_mode_updates_last_logged(self):
        """set_test_mode değişimi _last_logged_test_mode'a yansır."""
        ctrl = InputController(test_mode=True)
        self.assertTrue(ctrl.set_test_mode(False))
        self.assertFalse(ctrl.get_input_controller_diagnostic_snapshot()["last_logged_test_mode"])


if __name__ == "__main__":
    unittest.main()
