# -*- coding: utf-8 -*-
"""
Input Driver Module
Handles humanized keyboard and mouse input.
Uses DirectInput scan codes with randomized delays and micro-variations.
"""

import time
import random
import pydirectinput

# Enable fail-safe and default pauses
pydirectinput.PAUSE = 0.02
pydirectinput.FAILSAFE = False

class InputDriver:
    def __init__(self, use_faker_input=False):
        self.use_faker_input = use_faker_input

    def press_key(self, key: str, min_hold=0.08, max_hold=0.18):
        """
        Presses a key with a natural human duration.
        e.g., 'f', 'm', 'f8', 'f9', 'enter'
        """
        hold_time = random.uniform(min_hold, max_hold)
        pydirectinput.keyDown(key)
        time.sleep(hold_time)
        pydirectinput.keyUp(key)

    def human_click(self, x=None, y=None, button='left', min_hold=0.06, max_hold=0.14):
        """
        Moves the mouse with natural offset and clicks.
        """
        if x is not None and y is not None:
            # Human jitter (+- 1 to 2 pixels)
            target_x = x + random.randint(-1, 1)
            target_y = y + random.randint(-1, 1)
            pydirectinput.moveTo(target_x, target_y)
            time.sleep(random.uniform(0.05, 0.12))

        hold_time = random.uniform(min_hold, max_hold)
        pydirectinput.mouseDown(button=button)
        time.sleep(hold_time)
        pydirectinput.mouseUp(button=button)

    def send_gather_key(self):
        """Simulates pressing the 'F' gather key."""
        self.press_key('f')

    def toggle_map(self):
        """Simulates pressing 'M' to open/close the world map."""
        self.press_key('m')
