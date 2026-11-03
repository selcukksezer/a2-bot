# -*- coding: utf-8 -*-
"""
Main Application Entry Point (Gatherion A2 Pixel Bot)
Binds UI (Overlay, F9 Menu, Settings) with Core Engine (Vision, Navigation, Auto-Gather, Combat).
"""

import sys
import time
import threading
import keyboard
import winsound
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, pyqtSlot

from core.config import ConfigManager
from core.route_manager import RouteManager
from core.vision import Vision
from core.input_driver import InputDriver
from core.navigator import MapNavigator
from core.combat import CombatSystem
from modules.recorder import NodeRecorder

from ui.overlay import GameOverlay, OverlaySignals
from ui.menu import F9Menu
from ui.settings_dialog import SettingsDialog
from ui.map_viewer import MapViewerDialog

class BotController:
    def __init__(self, overlay: GameOverlay, signals: OverlaySignals):
        self.overlay = overlay
        self.signals = signals

        self.cfg = ConfigManager()
        self.routes = RouteManager()
        self.vision = Vision()
        self.driver = InputDriver()
        self.navigator = MapNavigator(self.routes, self.vision, self.driver)
        self.recorder = NodeRecorder(self.routes, self.vision, self.driver, self.navigator)
        self.combat = CombatSystem(self.vision, self.driver)

        # State variables
        self.current_mode = "route" # 'route', 'auto_record', 'manual_record'
        self.is_running = True
        self.is_paused = False

        self.gathered_count = 0
        self.f_pressed_count = 0
        self.active_node = None

        # Start worker thread
        self.thread = threading.Thread(target=self.bot_loop, daemon=True)
        self.thread.start()

    def bot_loop(self):
        """Main autonomous bot state machine loop."""
        walk_start_time = 0
        last_f_press = 0
        gather_lost = 0
        state = "PAUSED"

        while self.is_running:
            if self.is_paused:
                self.signals.update_signal.emit({
                    "state": "PAUSED",
                    "title": "Bot Paused",
                    "sub": "Press F9 to resume",
                    "gathered": self.gathered_count,
                    "f_pressed": self.f_pressed_count,
                    "match_pct": 0
                })
                time.sleep(0.2)
                continue

            # 1. Combat Check (if enabled)
            if self.cfg.get("route", "fight_back_when_attacked", True):
                if self.combat.check_under_attack():
                    self.signals.update_signal.emit({
                        "state": "COMBAT",
                        "title": "Fighting enemy...",
                        "sub": "Attacking target",
                        "gathered": self.gathered_count,
                        "f_pressed": self.f_pressed_count,
                        "match_pct": 0
                    })
                    self.combat.fight_back_step()
                    time.sleep(0.3)
                    continue

            # -------------------------------------------------------------
            # MODE: AUTO-RECORD (User walks, bot presses F)
            # -------------------------------------------------------------
            if self.current_mode == "auto_record":
                detected, match_pct, _ = self.vision.check_gather_prompt(threshold=0.75, use_roi=True)
                now = time.time()
                if detected:
                    self.signals.update_signal.emit({
                        "state": "GATHERING",
                        "title": "Auto Gathering",
                        "sub": f"Prompt recognized at {match_pct}%",
                        "gathered": self.gathered_count,
                        "f_pressed": self.f_pressed_count,
                        "match_pct": match_pct
                    })
                    if now - last_f_press >= self.cfg.get("gathering", "min_f_interval", 1.5):
                        if not self.cfg.get("gathering", "test_mode", False):
                            self.driver.send_gather_key()
                        self.f_pressed_count += 1
                        last_f_press = now
                else:
                    self.signals.update_signal.emit({
                        "state": "WALKING",
                        "title": "Auto Record Active",
                        "sub": "Walk towards nodes...",
                        "gathered": self.gathered_count,
                        "f_pressed": self.f_pressed_count,
                        "match_pct": match_pct
                    })
                time.sleep(0.12)

            # -------------------------------------------------------------
            # MODE: ROUTE (Bot walks to nodes and gathers)
            # -------------------------------------------------------------
            elif self.current_mode == "route":
                # Check for gathering prompt
                detected, match_pct, _ = self.vision.check_gather_prompt(threshold=0.75, use_roi=True)
                now = time.time()

                if detected:
                    gather_lost = 0
                    self.signals.update_signal.emit({
                        "state": "GATHERING",
                        "title": "Gathering",
                        "sub": f"Prompt recognized at {match_pct}%",
                        "gathered": self.gathered_count,
                        "f_pressed": self.f_pressed_count,
                        "match_pct": match_pct
                    })
                    min_int = self.cfg.get("gathering", "min_f_interval", 1.5)
                    if now - last_f_press >= min_int:
                        if not self.cfg.get("gathering", "test_mode", False):
                            self.driver.send_gather_key()
                        self.f_pressed_count += 1
                        last_f_press = now

                else:
                    if state == "GATHERING":
                        gather_lost += 1
                        if gather_lost > 4:
                            self.gathered_count += 1
                            if self.active_node:
                                self.routes.mark_node_visited(self.active_node["id"])
                            self.signals.log_signal.emit(f"Gathered {self.active_node['name'] if self.active_node else 'node'}")
                            self.active_node = None
                            state = "NAVIGATING"

                    # Ready to pick next node
                    if self.active_node is None:
                        rest_sec = self.cfg.get("route", "rest_after_visit_min", 5) * 60
                        ready_node = self.routes.get_next_ready_node(rest_duration_sec=rest_sec)
                        if ready_node:
                            self.active_node = ready_node
                            self.signals.update_signal.emit({
                                "state": "WALKING",
                                "title": f"Navigating to {ready_node['name']}",
                                "sub": "Opening map...",
                                "gathered": self.gathered_count,
                                "f_pressed": self.f_pressed_count,
                                "match_pct": 0
                            })
                            if not self.cfg.get("gathering", "test_mode", False):
                                self.navigator.navigate_to_node(ready_node)
                            walk_start_time = time.time()
                            state = "WALKING"
                        else:
                            self.signals.update_signal.emit({
                                "state": "IDLE",
                                "title": "Resting...",
                                "sub": "Waiting for nodes to refresh",
                                "gathered": self.gathered_count,
                                "f_pressed": self.f_pressed_count,
                                "match_pct": 0
                            })
                            time.sleep(5.0)
                    else:
                        # Walking timeout check
                        timeout = self.cfg.get("route", "time_to_reach_node_sec", 60)
                        if time.time() - walk_start_time > timeout:
                            self.signals.log_signal.emit(f"{self.active_node['name']} not reached in {timeout}s: skipping")
                            self.routes.mark_node_failed(self.active_node["id"])
                            self.active_node = None
                            time.sleep(1.0)
                        else:
                            remaining = int(timeout - (time.time() - walk_start_time))
                            self.signals.update_signal.emit({
                                "state": "WALKING",
                                "title": f"Walking to {self.active_node['name']}",
                                "sub": f"Timeout in {remaining}s",
                                "gathered": self.gathered_count,
                                "f_pressed": self.f_pressed_count,
                                "match_pct": match_pct
                            })

                time.sleep(0.12)

    def trigger_f8_record(self):
        node = self.recorder.record_current_node()
        if node:
            self.signals.log_signal.emit(f"New {node['name']} recorded")

    def toggle_f9(self):
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        self.signals.log_signal.emit(f"Bot {status}")
        try:
            winsound.Beep(800 if self.is_paused else 1400, 150)
        except:
            pass

def main():
    app = QApplication(sys.argv)

    # 1. Initialize Signals & Overlay
    signals = OverlaySignals()
    overlay = GameOverlay()
    signals.update_signal.connect(overlay.update_state)
    signals.log_signal.connect(overlay.add_log)
    overlay.show()

    # 2. Initialize Controller
    controller = BotController(overlay, signals)

    # 3. Context Menu
    f9_menu = F9Menu()
    settings_dlg = None

    def handle_menu_selection(action):
        nonlocal settings_dlg
        if action == "resume":
            controller.is_paused = False
        elif action == "auto_record":
            controller.current_mode = "auto_record"
            controller.is_paused = False
            overlay.add_log("Mode: Auto Record")
        elif action == "manual_record":
            controller.current_mode = "manual_record"
            controller.is_paused = False
            overlay.add_log("Mode: Manual Record")
        elif action == "route":
            controller.current_mode = "route"
            controller.is_paused = False
            overlay.add_log("Mode: Route Mode")
        elif action == "settings":
            settings_dlg = SettingsDialog(controller.cfg, controller.routes)
            settings_dlg.show()
        elif action == "exit":
            controller.is_running = False
            app.quit()

    f9_menu.mode_selected.connect(handle_menu_selection)

    # Global Hotkeys Hook via QTimer event loop bridge
    def on_f9_press():
        f9_menu.popup_at_cursor()

    keyboard.add_hotkey('f9', on_f9_press)
    keyboard.add_hotkey('f8', controller.trigger_f8_record)

    print("\n" + "=" * 60)
    print("GATHERION A2 BOT BAŞARIYLA BAŞLATILDI!")
    print("Oyun üstü HUD sol üstte açıldı.")
    print("[F9]  -> Farenin olduğu yerde menüyü açar")
    print("[F8]  -> Kaynağı anında haritaya kaydeder")
    print("=" * 60 + "\n")

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
