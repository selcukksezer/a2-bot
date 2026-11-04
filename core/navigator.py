# -*- coding: utf-8 -*-
"""
Map Navigator Module
Opens the map, matches the node target coordinate/patch, clicks to move, and monitors travel.
"""

import time
import random
import os
import cv2
import numpy as np
from core.input_driver import InputDriver
from core.vision import Vision
from core.route_manager import RouteManager

class MapNavigator:
    def __init__(self, route_manager: RouteManager, vision: Vision, driver: InputDriver):
        self.routes = route_manager
        self.vision = vision
        self.driver = driver

    def is_map_open(self, threshold=0.80):
        """
        Detects if the in-game world map (M) is currently visible on screen.
        """
        # In AION 2, the map title/header or center area has distinct non-world properties
        # We can check center contrast or match against map header
        screen_bgr = self.vision.grab_screen(grayscale=False)
        hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        
        # World in Aion 2 has high saturation foliage. Map is parchment/sepia (low saturation)
        h, w, _ = screen_bgr.shape
        center_sat = np.mean(hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7), 1])
        # Saturation is typically < 40 when map covers the screen
        return center_sat < 45

    def open_map(self, max_wait=2.0):
        """Opens map with M key and waits until visible."""
        if not self.is_map_open():
            self.driver.toggle_map()
            start = time.time()
            while time.time() - start < max_wait:
                time.sleep(0.1)
                if self.is_map_open():
                    time.sleep(0.15) # Wait for animation
                    return True
        return self.is_map_open()

    def close_map(self, max_wait=2.0):
        """Closes map with M key and waits until closed."""
        if self.is_map_open():
            self.driver.toggle_map()
            start = time.time()
            while time.time() - start < max_wait:
                time.sleep(0.1)
                if not self.is_map_open():
                    time.sleep(0.1)
                    return True
        return not self.is_map_open()

    def get_navigator_status_summary(self) -> dict:
        """Returns navigator status summary safely."""
        try:
            return {"navigator_ready": True}
        except Exception:
            return {"navigator_ready": False}

    def navigate_to_node(self, node: dict, timeout_sec=60):
        """
        Executes navigation to a specified node:
        1. Opens map (M)
        2. Locates target point (direct coordinate or patch matchTemplate)
        3. Clicks target
        4. Closes map (M)
        5. Returns status
        """
        print(f"\n[Navigator] '{node['name']}' konumuna rota başlatılıyor...")

        # 1. Open Map
        if not self.open_map():
            print("[Navigator] Hata: Harita açılamadı!")
            return False

        # 2. Determine Click Target
        target_x = node.get("map_x")
        target_y = node.get("map_y")

        # Optional: refine coordinate via patch matching if patch exists
        patch_file = node.get("patch_file")
        if patch_file:
            patch_path = os.path.join(self.routes.patch_dir, patch_file)
            if os.path.exists(patch_path):
                patch_gray = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                map_gray = self.vision.grab_screen(grayscale=True)
                res = cv2.matchTemplate(map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= 0.70:
                    ph, pw = patch_gray.shape
                    target_x = max_loc[0] + pw // 2
                    target_y = max_loc[1] + ph // 2
                    print(f"[Navigator] Harita şablon eşleşmesi: %{int(max_val*100)} -> ({target_x}, {target_y})")

        # 3. Click target on map
        time.sleep(random.uniform(0.15, 0.3))
        self.driver.human_click(target_x, target_y, button='left')
        time.sleep(random.uniform(0.2, 0.4))

        # 4. Close Map
        self.close_map()
        print(f"[Navigator] Hedefe tıklandı, yürüyüş başladı (Zaman aşımı: {timeout_sec}s)")
        return True
