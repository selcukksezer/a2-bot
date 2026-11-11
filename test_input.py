# -*- coding: utf-8 -*-
"""
Quick Game Input Test Script
Allows testing if the game client receives keystrokes without anti-cheat blocks.

Manual use (needs the real AION 2 window in focus; human observes the game):
    python test_input.py

Automated use (no game, no window): the pytest tests below drive
``InputDriver(test_mode=True)`` and assert on the returned hold times, so a
refused/aborted key press (0.0) fails the test instead of printing success.
"""

import sys
import time
import unittest
from unittest import mock

from core.input_driver import InputDriver


def run_manual_input_check(driver=None):
    """Send M (map) and F (gather) to the game and report what really happened.

    Returns 0 only when every key press was accepted by the driver
    (hold time > 0); any refused press is reported and returns 1.
    This script can NOT verify what the game shows on screen — the human
    running it must confirm the map opened/closed in game.
    """
    print("=" * 60)
    print("AION 2 - INPUT VE TUŞ İLETİM TESTİ")
    print("=" * 60)
    print("Bu script DirectInput kullanarak oyuna 'M' (Harita) ve 'F' tuşlarını gönderecek.")
    print("\nLÜTFEN 5 SANİYE İÇİNDE AION 2 PENCERESİNE GEÇİŞ YAPIN (ALT+TAB)!")
    print("=" * 60)

    for i in range(5, 0, -1):
        print(f"Başlıyor... {i} saniye", end="\r")
        time.sleep(1.0)
    print("\nTest başladı!\n")

    driver = driver if driver is not None else InputDriver()

    # 1. Test: M tuşu (Harita Açma)
    print("[1/2] 'M' tuşuna basılıyor (Harita açılmalı)...")
    map_open_hold = driver.toggle_map()
    time.sleep(1.5)

    print("[1/2] 'M' tuşuna tekrar basılıyor (Harita kapanmalı)...")
    map_close_hold = driver.toggle_map()
    time.sleep(1.0)

    # 2. Test: F tuşu (Etkileşim)
    print("[2/2] 'F' tuşuna basılıyor (Toplama tuşu)...")
    gather_hold = driver.send_gather_key()
    time.sleep(0.5)

    results = {
        "map_open_hold_s": map_open_hold,
        "map_close_hold_s": map_close_hold,
        "gather_hold_s": gather_hold,
    }
    failed = [name for name, hold in results.items() if not hold or hold <= 0.0]

    print("\n" + "=" * 60)
    for name, hold in results.items():
        print(f"{name}: {hold}")
    if failed:
        print("TEST BAŞARISIZ — sürücü şu tuşları göndermeyi reddetti: " + ", ".join(failed))
        print("Oyunda hiçbir tepki olmaması beklenir; FakerInput/HID yolunu kontrol edin.")
        print("=" * 60)
        return 1
    print("TEST TAMAMLANDI — sürücü tüm tuşları kabul etti (hold > 0).")
    print("Ekrandaki doğrulama sizde: harita açılıp kapandıysa PyDirectInput sorunsuz çalışıyor;")
    print("oyunda hiçbir tepki olmadıysa FakerInput sanal HID sürücüsünü devreye alın.")
    print("=" * 60)
    return 0


def main():
    return run_manual_input_check()


class ManualInputCheckTests(unittest.TestCase):
    """Automated checks of the same driver calls the manual script uses.

    InputDriver(test_mode=True) sends nothing to the OS; press_key returns
    the sampled hold time, and 0.0 when the press is refused — so these
    assertions fail for real when the driver refuses a key.
    """

    def _driver(self):
        return InputDriver(test_mode=True, use_faker_input=False, use_interception=False)

    def test_toggle_map_accepted_in_test_mode(self):
        driver = self._driver()
        hold = driver.toggle_map()
        self.assertGreater(hold, 0.0, "toggle_map ('M') was refused by the driver")

    def test_send_gather_key_accepted_in_test_mode(self):
        driver = self._driver()
        hold = driver.send_gather_key()
        self.assertGreater(hold, 0.0, "send_gather_key ('F') was refused by the driver")

    def test_press_key_hold_within_configured_range(self):
        driver = self._driver()
        hold = driver.press_key("m", 0.04, 0.09)
        self.assertGreaterEqual(hold, 0.04)
        self.assertLessEqual(hold, 0.09)

    def test_aborted_motion_refuses_key_press(self):
        """Negative control: a refused press returns 0.0, so the suite can fail."""
        driver = self._driver()
        driver.abort_motion()
        try:
            self.assertEqual(driver.press_key("m"), 0.0)
            self.assertEqual(driver.toggle_map(), 0.0)
        finally:
            driver.resume_motion()

    def test_manual_check_reports_refused_keys(self):
        """run_manual_input_check returns non-zero instead of unconditional success."""
        driver = self._driver()
        driver.abort_motion()
        try:
            with mock.patch("test_input.time.sleep", return_value=None):
                self.assertEqual(run_manual_input_check(driver=driver), 1)
        finally:
            driver.resume_motion()


if __name__ == "__main__":
    sys.exit(main())
