# -*- coding: utf-8 -*-
"""
Vision Module
Handles ultra-fast screen capture via mss and OpenCV template matching.
Optimized for high-speed detection (30-60 FPS) with low CPU usage.
"""

import os
import cv2
import numpy as np
import mss

class Vision:
    def __init__(self, template_path="assets/f_gather_template.png", monitor_idx=1):
        self.sct = mss.mss()
        # Primary monitor (index 1 in mss is primary screen, 0 is virtual all-in-one)
        self.monitor = self.sct.monitors[monitor_idx]
        self.screen_w = self.monitor["width"]
        self.screen_h = self.monitor["height"]
        
        self.template_path = template_path
        self.gather_template = None
        self.load_template()

        # ROI for 'F Gather' prompt: Typically in the bottom-right quadrant of screen
        # e.g., X from 60% to 95%, Y from 65% to 95%
        self.prompt_roi = {
            "top": self.monitor["top"] + int(self.screen_h * 0.65),
            "left": self.monitor["left"] + int(self.screen_w * 0.60),
            "width": int(self.screen_w * 0.38),
            "height": int(self.screen_h * 0.32)
        }

    def load_template(self):
        if os.path.exists(self.template_path):
            self.gather_template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            print(f"[Vision] Loaded template '{self.template_path}' shape: {self.gather_template.shape}")
        else:
            print(f"[Vision] Warning: Template '{self.template_path}' not found!")

    def grab_screen(self, region=None, grayscale=True):
        """Grabs full screen or specified region."""
        target = region if region else self.monitor
        raw = self.sct.grab(target)
        img = np.array(raw)
        if grayscale:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def check_gather_prompt(self, threshold=0.75, use_roi=True):
        """
        Scans for the 'F Gather' prompt.
        Returns: (detected: bool, match_percentage: int, location: tuple)
        """
        if self.gather_template is None:
            return False, 0, None

        search_area = self.prompt_roi if use_roi else self.monitor
        screen_gray = self.grab_screen(region=search_area, grayscale=True)

        res = cv2.matchTemplate(screen_gray, self.gather_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        match_pct = int(round(max_val * 100))
        if max_val >= threshold:
            # Calculate absolute screen coordinates of prompt center
            th, tw = self.gather_template.shape
            abs_x = search_area["left"] + max_loc[0] + tw // 2
            abs_y = search_area["top"] + max_loc[1] + th // 2
            return True, match_pct, (abs_x, abs_y)
            
        return False, match_pct, None
