# -*- coding: utf-8 -*-
"""
Auto-Gather Module (Gatherion Auto-Record Style)
Detects 'F Gather' prompt in real time and automatically gathers resources.
"""

import time
import random
import winsound
import keyboard
from core.vision import Vision
from core.input_driver import InputDriver

class AutoGatherBot:
    def __init__(self, match_threshold=0.75, min_f_interval=1.5, max_f_interval=2.8):
        self.vision = Vision()
        self.driver = InputDriver()
        self.match_threshold = match_threshold
        self.min_f_interval = min_f_interval
        self.max_f_interval = max_f_interval

        self.running = False
        self.paused = False
        self.gathered_count = 0
        self.f_pressed_count = 0

    def start(self):
        self.running = True
        self.paused = False
        print("\n" + "=" * 60)
        print("AION 2 AUTO-GATHER (AUTO-RECORD) MODU BAŞLATILDI")
        print("Kontroller: [F9] Duraklat / Devam Et | [ESC] Çıkış")
        print("Karakterinle kaynakların yanına git, bot otomatik toplayacaktır.")
        print("=" * 60 + "\n")

        # Setup hotkeys
        keyboard.add_hotkey('f9', self.toggle_pause)

        last_f_time = 0
        is_gathering = False
        consecutive_not_found = 0

        try:
            while self.running:
                if keyboard.is_pressed('esc'):
                    print("\n[Bot] ESC basıldı, çıkış yapılıyor...")
                    break

                if self.paused:
                    time.sleep(0.2)
                    continue

                # Check for 'F Gather' on screen
                detected, match_pct, loc = self.vision.check_gather_prompt(
                    threshold=self.match_threshold,
                    use_roi=True
                )

                now = time.time()

                if detected:
                    consecutive_not_found = 0
                    if not is_gathering:
                        is_gathering = True
                        print(f"\n[GATHERING] Kaynak tespit edildi! (Eşleşme: %{match_pct})")
                        try:
                            winsound.Beep(1200, 150) # Gentle prompt beep
                        except:
                            pass

                    # Press F at humanized intervals
                    f_interval = random.uniform(self.min_f_interval, self.max_f_interval)
                    if now - last_f_time >= f_interval:
                        self.driver.send_gather_key()
                        self.f_pressed_count += 1
                        last_f_time = now
                        print(f"[{time.strftime('%H:%M:%S')}] F basıldı (#{self.f_pressed_count}) | Eşleşme: %{match_pct}")

                else:
                    consecutive_not_found += 1
                    # If prompt missing for 3 consecutive checks (~0.5s), finish gathering
                    if is_gathering and consecutive_not_found > 3:
                        is_gathering = False
                        self.gathered_count += 1
                        print(f"[{time.strftime('%H:%M:%S')}] Toplama tamamlandı! (Toplam Toplanan: {self.gathered_count})")
                        time.sleep(random.uniform(0.8, 1.5))

                # Sleep a tiny amount to keep CPU near 0-1%
                time.sleep(0.12)

        finally:
            self.running = False
            keyboard.unhook_all()
            print("\nBot durduruldu.")

    def toggle_pause(self):
        self.paused = not self.paused
        state = "DURAKLATILDI" if self.paused else "DEVAM EDİYOR"
        print(f"\n>>> [F9] BOT {state} <<<")
        try:
            freq = 800 if self.paused else 1500
            winsound.Beep(freq, 150)
        except:
            pass

    def get_auto_gather_ready_flag(self) -> dict:
        return {"module": "auto_gather", "ready": True}

    def get_auto_gather_version_info(self) -> dict:
        return {"module": "auto_gather", "version": "2.0.0", "ready": True}

    def get_auto_gather_diagnostic_snapshot(self) -> dict:
        return {"module": "auto_gather", "status": "ok", "ready": True}

if __name__ == "__main__":
    bot = AutoGatherBot()
    bot.start()


def get_auto_gather_capabilities() -> list:
    return ["auto_gather", "proximity_harvesting", "diagnostic_snapshot"]

