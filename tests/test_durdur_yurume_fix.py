# -*- coding: utf-8 -*-
"""
GÖREV 3 (fix/durdur-yurume) — bot durdurulunca oyunda yürüyememe.

Sözleşme: release/abort yolları YALNIZCA botun gerçekten tuttuğu tuşları
bırakır (_held_movement ∪ _active_bot_keys). Botun basmadığı bir tuş için
ASLA sentetik key-UP gönderilmez — aksi halde kullanıcının fiziksel W'su
oyunda bırakılmış görünür ve karakter durur. Stop sonrası giriş mandalı
(_input_stopped) kuruluyken hiçbir sentetik girdi çıkmaz; resume_motion()
yeniden kurar.

NOT: Bu dosya yalnızca yazıldı, KOŞULMADI (görev kuralı: pytest yok).
Koşacak ajan: `pytest tests/test_durdur_yurume_fix.py -v`.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.input_driver import EXTENDED_KEYS, HARDWARE_SCANCODES, InputController


def _make_controller():
    """Gerçek InputController; donanım backend'leri stub, soft yollar casus.

    Interception mock'u başarılı gönderim yapar; _hw_key_up/_hw_key_down
    böylece Interception üzerinden gider ve her gönderim send_key
    çağrısından sayılabilir. Soft yollar hiçbir zaman çağrılmamalı.
    """
    mock_interception = MagicMock()
    mock_interception.available = True
    mock_interception.send_key.return_value = True
    ctrl = InputController(test_mode=False, use_faker_input=False, use_interception=True)
    ctrl._faker = None
    ctrl._serial_hid = None
    ctrl._interception = mock_interception
    ctrl.ensure_serial_hid = MagicMock(return_value=False)
    ctrl._sendinput_key = MagicMock()
    ctrl._keybd_event_key = MagicMock()
    ctrl._post_key_to_game = MagicMock()
    return ctrl, mock_interception


# (scancode, extended) -> tek kanonik isim. HARDWARE_SCANCODES bir
# scancode'u birden çok isimle paylaşır (örn. 0x38: "alt", "left alt",
# "right alt"); eski yardımcı her eşleşen ismi ekleyip tek bir UP'u
# üç isimmiş gibi sayıyordu. İlk görülen isim kanoniktir ("alt" önce
# gelir); "right alt"/"right ctrl" extended=True ile ayrışır.
_CANONICAL_BY_SCANCODE = {}
for _name, _code in HARDWARE_SCANCODES.items():
    _key = (_code, _name in EXTENDED_KEYS)
    _CANONICAL_BY_SCANCODE.setdefault(_key, _name)


def _key_ups_sent(mock_interception):
    """Interception send_key çağrılarından key-UP gönderilen tuş adları.

    Her (scancode, extended) çifti tek kanonik isme çevrilir; böylece
    tek bir alt-UP {"alt"} olur, {"alt", "left alt", "right alt"} değil.
    """
    sent = set()
    for call in mock_interception.send_key.call_args_list:
        args, kwargs = call
        scancode, is_down = args[0], args[1]
        if len(args) > 2:
            extended = bool(args[2])
        else:
            extended = bool(kwargs.get("extended", False))
        if is_down is False:
            name = _CANONICAL_BY_SCANCODE.get((scancode, extended))
            if name is not None:
                sent.add(name)
    return sent


class TestDurdurYurumeFix(unittest.TestCase):
    def test_release_empty_sends_zero_key_ups(self):
        """Tutulan küme boşken release_all_held_keys sıfır key-UP gönderir."""
        ctrl, mock = _make_controller()
        self.assertEqual(len(ctrl._held_movement), 0)
        self.assertEqual(len(ctrl._active_bot_keys), 0)

        released = ctrl.release_all_held_keys(reason="route_deactivated")

        self.assertEqual(released, 0)
        mock.send_key.assert_not_called()
        ctrl._sendinput_key.assert_not_called()
        ctrl._keybd_event_key.assert_not_called()

    def test_release_only_bot_held_keys(self):
        """Botun tuttuğu tuş bırakılır; tutmadığı tuşa UP gitmez."""
        ctrl, mock = _make_controller()
        ctrl._held_movement.add("w")
        ctrl._active_bot_keys.update({"w", "2"})

        released = ctrl.release_all_held_keys(reason="test_transition")

        self.assertEqual(released, 2)
        self.assertEqual(_key_ups_sent(mock), {"w", "2"})
        # Eski 23 tuşluk salvodan hiçbir iz yok:
        for never in ("a", "s", "d", "q", "e", "space", "tab", "alt",
                      "ctrl", "shift", "1", "0"):
            self.assertNotIn(
                never, _key_ups_sent(mock),
                f"botun tutmadığı '{never}' için key-UP gönderilmemeli",
            )
        self.assertEqual(len(ctrl._active_bot_keys), 0)
        self.assertEqual(len(ctrl._held_movement), 0)
        ctrl._sendinput_key.assert_not_called()
        ctrl._keybd_event_key.assert_not_called()

    def test_release_all_movement_keys_only_bot_held(self):
        """release_all_movement_keys yalnız botun tuttuğu hareket tuşunu bırakır."""
        ctrl, mock = _make_controller()
        ctrl._held_movement.add("w")

        ctrl.release_all_movement_keys()

        self.assertEqual(_key_ups_sent(mock), {"w"})

    def test_release_modifier_keys_only_bot_held(self):
        """release_modifier_keys yalnız botun tuttuğu modifier'ı bırakır."""
        ctrl, mock = _make_controller()
        ctrl._active_bot_keys.add("alt")

        ctrl.release_modifier_keys()

        self.assertEqual(_key_ups_sent(mock), {"alt"})
        ctrl._sendinput_key.assert_not_called()
        ctrl._keybd_event_key.assert_not_called()

    def test_release_mouse_buttons_only_bot_held(self):
        """_release_mouse_buttons yalnız botun tuttuğu fare tuşunu bırakır."""
        ctrl, _mock = _make_controller()
        ctrl._mouse_buttons_down.add("right")
        with patch.object(ctrl, "_hw_mouse_up") as mock_up:
            ctrl._release_mouse_buttons()
        mock_up.assert_called_once_with("right")

    def test_stop_latch_silences_abort_and_releases(self):
        """Stop sonrası F10 abort'u hiçbir şey göndermez; resume yeniden kurar."""
        ctrl, mock = _make_controller()
        ctrl._held_movement.add("w")
        ctrl._active_bot_keys.add("w")

        # mark_input_stopped: önce botun tuttuğu W bırakılır, sonra mandal kurulur.
        ctrl.mark_input_stopped()
        self.assertTrue(ctrl.input_stopped())
        self.assertEqual(_key_ups_sent(mock), {"w"})
        mock.send_key.reset_mock()

        # Stop sonrası abort / guard yolları sessizdir.
        ctrl.abort_motion()
        self.assertEqual(ctrl.release_all_held_keys(reason="route_deactivated"), 0)
        ctrl.release_all_movement_keys()
        ctrl.release_modifier_keys()
        mock.send_key.assert_not_called()
        ctrl._sendinput_key.assert_not_called()
        ctrl._keybd_event_key.assert_not_called()

        # F10 akoru: on_f10 callback'i çalışır ama abort gönderim yapmaz.
        fired = []
        ctrl.hotkey_listener.on_f10 = lambda: fired.append("f10")
        ctrl.hotkey_listener._async_invoke = False  # callback senkron çalışsın
        ctrl.hotkey_listener._invoke_chord("f10")
        self.assertEqual(fired, ["f10"])
        mock.send_key.assert_not_called()

        # press_key kapısı da mandal kalkana dek kapalıdır.
        self.assertEqual(ctrl.press_key("2"), 0.0)
        mock.send_key.assert_not_called()

        # resume yeniden kurar: mandal kalkar, bot tuşu tekrar bırakılabilir.
        ctrl.resume_motion()
        self.assertFalse(ctrl.input_stopped())
        self.assertFalse(ctrl.motion_aborted())
        ctrl._held_movement.add("w")
        ctrl._active_bot_keys.add("w")
        self.assertEqual(ctrl.release_all_held_keys(reason="resume_check"), 1)
        self.assertEqual(_key_ups_sent(mock), {"w"})

    def test_press_key_gated_while_motion_aborted(self):
        """_motion_abort kuruluysa press_key 0.0 döner, gönderim yapmaz."""
        ctrl, mock = _make_controller()
        ctrl.abort_motion()
        self.assertTrue(ctrl.motion_aborted())
        self.assertEqual(ctrl.press_key("2"), 0.0)
        mock.send_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
