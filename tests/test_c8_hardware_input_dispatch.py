# -*- coding: utf-8 -*-
"""
Agent C8 Verification Suite:
Hardware-Trusted Input Dispatch, Opsec Key Timing & Anti-Stuck Key Release Guard.
Tests:
1. Interception/HID single-fire dispatch (no soft double injection).
2. [OPSEC FALLBACK] warning logging when hardware dispatch is unavailable or fails.
3. Gaussian jitter timing and GCD alignment.
4. Anti-stuck key release guard on state transitions (abort, route stop, manual trigger).
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from core.game_action_bus import GameActionBus
from core.input_driver import InputController
from core.skill_bar_reader import SkillBarReader


class TestC8HardwareTrustedInputDispatch(unittest.TestCase):
    """Test suite for Agent C8 hardware dispatch and opsec timing requirements."""

    def test_interception_single_fire_key_down_and_up(self):
        """Interception success must fire exactly once and NEVER call soft fallbacks."""
        ctrl = InputController.__new__(InputController)
        ctrl.test_mode = False
        ctrl._sending_virtual_key = False
        ctrl._last_virtual_key_ts = 0.0
        ctrl._serial_hid = None
        ctrl.ensure_serial_hid = MagicMock(return_value=True)

        mock_interception = MagicMock()
        mock_interception.available = True
        mock_interception.send_key.return_value = True
        ctrl._interception = mock_interception
        ctrl._faker = None

        ctrl._sendinput_key = MagicMock()
        ctrl._keybd_event_key = MagicMock()
        ctrl._post_key_to_game = MagicMock()

        warn_logs = []
        with patch("core.input_driver.logger.warning", side_effect=lambda msg, *args: warn_logs.append(msg % args if args else msg)):
            # Key down
            down_ok = ctrl.key_down("q")
            self.assertTrue(down_ok)
            mock_interception.send_key.assert_called_once_with(16, True, extended=False)  # 'q' scan code is 0x10 = 16
            ctrl._sendinput_key.assert_not_called()
            ctrl._keybd_event_key.assert_not_called()
            ctrl._post_key_to_game.assert_not_called()

            # Key up
            mock_interception.send_key.reset_mock()
            up_ok = ctrl.key_up("q")
            self.assertTrue(up_ok)
            mock_interception.send_key.assert_called_once_with(16, False, extended=False)
            ctrl._sendinput_key.assert_not_called()
            ctrl._keybd_event_key.assert_not_called()

            # No OPSEC FALLBACK logged
            opsec_warnings = [m for m in warn_logs if "OPSEC FALLBACK" in m]
            self.assertEqual(len(opsec_warnings), 0, f"Unexpected OPSEC FALLBACK logged: {opsec_warnings}")

    def test_opsec_fallback_logging_when_hardware_absent(self):
        """When hardware is absent, [OPSEC FALLBACK] must be clearly emitted in logs."""
        ctrl = InputController.__new__(InputController)
        ctrl.test_mode = False
        ctrl._sending_virtual_key = False
        ctrl._last_virtual_key_ts = 0.0
        ctrl._serial_hid = None
        ctrl.ensure_serial_hid = MagicMock(return_value=False)
        ctrl._interception = None
        ctrl._faker = None

        ctrl._sendinput_key = MagicMock(return_value=True)
        ctrl._keybd_event_key = MagicMock(return_value=True)
        ctrl._post_key_to_game = MagicMock(return_value=True)

        warn_logs = []
        with patch("core.input_driver.logger.warning", side_effect=lambda msg, *args: warn_logs.append(msg % args if args else msg)):
            ctrl.key_down("1")
            ctrl.key_up("1")

            opsec_warnings = [m for m in warn_logs if "OPSEC FALLBACK" in m]
            self.assertGreaterEqual(len(opsec_warnings), 2, f"Expected OPSEC FALLBACK for down & up, got: {opsec_warnings}")
            self.assertTrue(any("'1' tuşu için yumuşak" in m for m in opsec_warnings))

    def test_anti_stuck_key_release_guard_clears_active_and_suspect_keys(self):
        """release_all_held_keys must release all active keys and critical suspects."""
        ctrl = InputController(test_mode=True)
        ctrl._active_bot_keys.update({"1", "w", "alt"})
        ctrl._held_movement.add("w")

        released_count = ctrl.release_all_held_keys(reason="test_transition")
        self.assertGreaterEqual(released_count, 15)  # cleared suspects
        self.assertEqual(len(ctrl._active_bot_keys), 0)
        self.assertEqual(len(ctrl._held_movement), 0)
        self.assertFalse(ctrl._bot_pressing)

    def test_set_route_active_deactivation_triggers_anti_stuck_guard(self):
        """Transitioning route_active from True to False triggers anti-stuck guard."""
        ctrl = InputController(test_mode=True)
        ctrl.route_active = True
        ctrl._active_bot_keys.add("w")
        ctrl._held_movement.add("w")

        with patch.object(ctrl, "release_all_held_keys", wraps=ctrl.release_all_held_keys) as mock_release:
            ctrl.set_route_active(False)
            mock_release.assert_called_once_with(reason="route_deactivated")
            self.assertEqual(len(ctrl._active_bot_keys), 0)

    def test_game_action_bus_trusted_dispatch_and_gcd_alignment(self):
        """GameActionBus with trusted driver uses hardware key and calculates GCD alignment."""
        driver = InputController.__new__(InputController)
        driver.test_mode = False
        driver._sending_virtual_key = False
        driver._last_virtual_key_ts = 0.0
        driver._serial_hid = None
        driver.ensure_serial_hid = MagicMock(return_value=True)

        mock_interception = MagicMock()
        mock_interception.available = True
        mock_interception.send_key.return_value = True
        driver._interception = mock_interception
        driver._faker = None

        emitted_logs = []
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=SkillBarReader(),
            on_log=lambda msg: emitted_logs.append(msg),
            mouse_look_mode=True,
        )

        res = bus.cast_skill("2")
        self.assertTrue(res["ok"])
        self.assertEqual(res["method"], "key")
        self.assertTrue(res["trusted"])
        self.assertGreater(res["burst_duration_sec"], 0.0)
        self.assertGreater(res["gcd_align_sec"], 0.0)

        # Single-fire check: Interception was called
        mock_interception.send_key.assert_called()

        # GCD align helper
        gcd_wait = bus.align_to_gcd(0.040, gcd_sec=0.12)
        self.assertGreaterEqual(gcd_wait, 0.020)
        self.assertLessEqual(gcd_wait, 0.12)

    def test_game_action_bus_opsec_fallback_logged_when_untrusted(self):
        """GameActionBus logs [OPSEC FALLBACK] when keyboard inject is not trusted."""
        driver = InputController.__new__(InputController)
        driver.test_mode = False
        driver._sending_virtual_key = False
        driver._last_virtual_key_ts = 0.0
        driver._serial_hid = None
        driver.ensure_serial_hid = MagicMock(return_value=False)
        driver._interception = None
        driver._faker = None

        emitted_logs = []
        warn_logs = []
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=SkillBarReader(),
            on_log=lambda msg: emitted_logs.append(msg),
            mouse_look_mode=True,
        )

        with patch("core.game_action_bus.logger.warning", side_effect=lambda msg, *args: warn_logs.append(msg % args if args else msg)):
            res = bus.cast_skill("1")
            # In mouse_look_mode without slot coords, untrusted key skips
            opsec_warnings = [m for m in warn_logs if "OPSEC FALLBACK" in m]
            self.assertGreaterEqual(len(opsec_warnings), 1, f"Expected OPSEC FALLBACK in warnings: {warn_logs}")
            opsec_emits = [m for m in emitted_logs if "OPSEC FALLBACK" in m]
            self.assertGreaterEqual(len(opsec_emits), 1, f"Expected OPSEC FALLBACK in emits: {emitted_logs}")

    def test_game_action_bus_anti_stuck_guard(self):
        """GameActionBus anti_stuck_guard clears held attacks, exclusive lock, and driver keys."""
        driver = InputController(test_mode=True)
        driver._active_bot_keys.add("w")
        driver._active_bot_keys.add("2")

        released = []
        bus = GameActionBus(
            input_driver=driver,
            release_basic_attack=lambda: released.append("basic_attack_released"),
            mouse_look_mode=True,
        )
        bus._left_held = True
        bus._exclusive_depth = 2
        bus._exclusive_reason = "skill:1"

        bus.anti_stuck_guard(reason="combat_to_route")

        self.assertFalse(bus._left_held)
        self.assertEqual(bus._exclusive_depth, 0)
        self.assertEqual(bus._exclusive_reason, "")
        self.assertEqual(len(driver._active_bot_keys), 0)


if __name__ == "__main__":
    unittest.main()
