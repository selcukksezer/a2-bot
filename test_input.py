# -*- coding: utf-8 -*-
"""
Quick Game Input Test Script
Allows testing if the game client receives keystrokes without anti-cheat blocks.
"""

import time
import sys
from core.input_driver import InputDriver

def main():
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

    driver = InputDriver()

    # 1. Test: M tuşu (Harita Açma)
    print("[1/2] 'M' tuşuna basılıyor (Harita açılmalı)...")
    driver.toggle_map()
    time.sleep(1.5)

    print("[1/2] 'M' tuşuna tekrar basılıyor (Harita kapanmalı)...")
    driver.toggle_map()
    time.sleep(1.0)

    # 2. Test: F tuşu (Etkileşim)
    print("[2/2] 'F' tuşuna basılıyor (Toplama tuşu)...")
    driver.send_gather_key()
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("TEST TAMAMLANDI!")
    print("- Eğer oyun içinde Harita (M) açılıp kapandıysa: PyDirectInput sorunsuz çalışıyor!")
    print("- Eğer oyunda hiçbir tepki olmadıysa: FakerInput sanal HID sürücüsünü devreye alacağız.")
    print("=" * 60)

if __name__ == "__main__":
    main()


def get_test_input_ready_flag() -> dict:
    return {"module": "test_input", "ready": True}


def get_test_input_version_info() -> dict:
    return {"module": "test_input", "version": "2.0.0", "ready": True}

