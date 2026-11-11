# -*- coding: utf-8 -*-
"""
Agent C8 Verification Suite:
Hardware-Trusted Input Dispatch, Opsec Key Timing & Anti-Stuck Key Release Guard.
1. Interception/HID single-fire dispatch (no soft double injection).
2. No-trusted-path refusal: key is NOT sent, [HARDWARE KEY GEREKLİ] is logged
   (soft SendInput never runs — AION2 drops it, so faking success is worse).
3. Gaussian jitter timing and GCD alignment.
4. Anti-stuck key release guard on state transitions (abort, route stop, manual trigger).
5. GameActionBus trusted dispatch (exact down/up pulses) and GCD alignment.
6. GameActionBus [OPSEC FALLBACK] logging when keyboard inject is untrusted.
"""

import logging
import unittest
from unittest.mock import MagicMock, call, patch

from core.game_action_bus import GameActionBus
from core.input_driver import InputController
from core.skill_bar_reader import SkillBarReader


def _make_controller(*, interception=None, faker=None, serial_hid=None,
                     ensure_serial_hid=False):
    """Build a real InputController (constructor runs — F-1353: no __new__ bypass)
    with the hardware backends stubbed. Soft fallbacks are spies so the tests
    can prove they are never called."""
    ctrl = InputController(test_mode=False, use_faker_input=False, use_interception=True)
    ctrl._faker = faker
    ctrl._serial_hid = serial_hid
    ctrl._interception = interception
    ctrl.ensure_serial_hid = MagicMock(return_value=ensure_serial_hid)
    ctrl._sendinput_key = MagicMock()
    ctrl._keybd_event_key = MagicMock()
    ctrl._post_key_to_game = MagicMock()
    return ctrl


class TestC8HardwareTrustedInputDispatch(unittest.TestCase):
    """Test suite for Agent C8 hardware dispatch and opsec timing requirements."""

    def test_interception_single_fire_key_down_and_up(self):
        """Interception success must fire exactly once and NEVER call soft fallbacks."""
        mock_interception = MagicMock()
        mock_interception.available = True
        mock_interception.send_key.return_value = True
        ctrl = _make_controller(interception=mock_interception, ensure_serial_hid=True)

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

    def test_no_trusted_path_refuses_key_and_logs_hardware_required(self):
        """No trusted path: the key must be REFUSED (never soft-sent as if it worked).

        Production refuses the key and logs a throttled [HARDWARE KEY GEREKLİ]
        error — and, by design, a short [OPSEC FALLBACK] warning PER refused
        key call (core/input_driver.py _log_hw_key_untrusted_once docstring:
        "the short OPSEC FALLBACK warning is emitted per call"). This test
        calls key_down('1') + key_up('1'), so exactly 2 warnings are expected
        (S-6601: the old "0 warnings" expectation contradicted production).
        """
        ctrl = _make_controller(interception=None, faker=None, serial_hid=None,
                                ensure_serial_hid=False)

        error_logs = []
        warn_logs = []
        with patch("core.input_driver.logger.error", side_effect=lambda msg, *args: error_logs.append(msg % args if args else msg)), \
             patch("core.input_driver.logger.warning", side_effect=lambda msg, *args: warn_logs.append(msg % args if args else msg)):
            down_ok = ctrl.key_down("1")
            ctrl.key_up("1")

        self.assertFalse(down_ok, "no trusted path: key_down must refuse, not fake success")
        hw_errors = [m for m in error_logs if "HARDWARE KEY GEREKLİ" in m]
        self.assertGreaterEqual(len(hw_errors), 1, f"Expected [HARDWARE KEY GEREKLİ] error, got: {error_logs}")
        self.assertTrue(any("'1'" in m and "gönderilmedi" in m for m in hw_errors),
                        f"Error must name the refused key: {hw_errors}")
        # One [OPSEC FALLBACK] warning per refused key call by design:
        # key_down('1') ("basma") + key_up('1') ("bırakma") = 2 warnings.
        opsec_warnings = [m for m in warn_logs if "OPSEC FALLBACK" in m]
        self.assertEqual(len(opsec_warnings), 2, f"Expected 2 OPSEC FALLBACK warnings (one per refused key), got: {opsec_warnings}")
        self.assertTrue(all("'1'" in m for m in opsec_warnings),
                        f"OPSEC FALLBACK warnings must name the refused key: {opsec_warnings}")
        ctrl._sendinput_key.assert_not_called()
        ctrl._keybd_event_key.assert_not_called()
        ctrl._post_key_to_game.assert_not_called()

    def test_anti_stuck_key_release_guard_clears_active_and_suspect_keys(self):
        """release_all_held_keys must release all active keys and critical suspects.

        2-kare kuralı değil, üretim sözleşmesi değişimi: release_all_held_keys
        artık yalnızca botun tuttuğu tuşları bırakır (core/input_driver.py
        release_all_held_keys docstring: eski koşulsuz 23 tuşluk salvo
        kaldırıldı — botun basmadığı tuşa sentetik UP gönderilmez).
        test_mode'da sayı = len(_held_movement ∪ _active_bot_keys) =
        len({"1", "w", "alt"}) = 3 ("w" iki kümede de var, bir kez sayılır).
        """
        ctrl = InputController(test_mode=True)
        ctrl._active_bot_keys.update({"1", "w", "alt"})
        ctrl._held_movement.add("w")

        released_count = ctrl.release_all_held_keys(reason="test_transition")
        self.assertEqual(released_count, 3)  # only bot-held keys: {"1", "w", "alt"}
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
        mock_interception = MagicMock()
        mock_interception.available = True
        mock_interception.send_key.return_value = True
        driver = _make_controller(interception=mock_interception, ensure_serial_hid=True)

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

        # Exact dispatch sequence for '2' (scan code 0x03): cast_skill ->
        # _key_fallback calls release_modifier_keys first, but production
        # only releases modifiers the BOT actually holds
        # (core/input_driver.py:3674 release_modifier_keys iterates
        # _active_bot_keys — no modifier is held here, so ZERO modifier
        # releases are sent; the old 10-call unconditional salvo no longer
        # exists). Then the skill key fires exactly one down + one up
        # pulse (F-1565).
        self.assertEqual(
            mock_interception.send_key.call_args_list,
            [
                call(3, True, extended=False),  # '2' down (scan code 0x03)
                call(3, False, extended=False),  # '2' up
            ],
        )
        driver._sendinput_key.assert_not_called()
        driver._keybd_event_key.assert_not_called()
        driver._post_key_to_game.assert_not_called()

        # GCD align helper
        gcd_wait = bus.align_to_gcd(0.040, gcd_sec=0.12)
        self.assertGreaterEqual(gcd_wait, 0.020)
        self.assertLessEqual(gcd_wait, 0.12)

    def test_game_action_bus_opsec_fallback_logged_when_untrusted(self):
        """GameActionBus logs [OPSEC FALLBACK] when keyboard inject is not trusted."""
        driver = _make_controller(interception=None, faker=None, serial_hid=None,
                                  ensure_serial_hid=False)

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
        """GameActionBus anti_stuck_guard clears held attacks, exclusive lock, and driver keys.

        NOTE (S-6602): the guard can only clear _left_held when the driver
        actually releases 'r'. A test_mode driver can never do that — its
        key_up always returns False, and hold_left_attack deliberately keeps
        _left_held in that case ("Key may still be held — keep _left_held so
        the next release attempt retries"). So this test uses a trusted
        Interception driver (same _make_controller pattern as the other bus
        tests), where key_up('r') succeeds and the guard clears the flag.
        """
        mock_interception = MagicMock()
        mock_interception.available = True
        mock_interception.send_key.return_value = True
        driver = _make_controller(interception=mock_interception, ensure_serial_hid=True)
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
