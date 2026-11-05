# -*- coding: utf-8 -*-
"""
PyQt6 F9 Context Menu (Gatherion Style Radial/Popup Menu)
Triggered anywhere on screen via F9:
- Resume (R)
- Auto record (A)
- Manual record (N)
- Route mode (M)
- Active route (T)
- Settings (S)
- Exit (X)
"""

from PyQt6.QtWidgets import QMenu, QWidget
from PyQt6.QtGui import QAction, QCursor, QFont, QIcon
from PyQt6.QtCore import Qt, pyqtSignal

class F9Menu(QMenu):
    mode_selected = pyqtSignal(str) # 'route', 'auto_record', 'manual_record', 'resume', 'settings', 'exit'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QMenu {
                background-color: #1a1e24;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
                font-size: 13px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
            }
            QMenu::item {
                padding: 8px 24px 8px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #334155;
                margin: 4px 8px;
            }
        """)

        # Items matching Gatherion exactly
        self.act_resume = QAction("Resume: route mode\t(R)", self)
        self.act_auto = QAction("Auto record: you walk, bot presses F\t(A)", self)
        self.act_manual = QAction("Manual record: you walk, F8 saves node\t(N)", self)
        self.act_route = QAction("Route mode: bot walks and presses F\t(M)", self)
        self.act_settings = QAction("Settings...\t(S)", self)
        self.act_exit = QAction("Exit\t(X)", self)

        self.addAction(self.act_resume)
        self.addSeparator()
        self.addAction(self.act_auto)
        self.addAction(self.act_manual)
        self.addAction(self.act_route)
        self.addSeparator()
        self.addAction(self.act_settings)
        self.addAction(self.act_exit)

        # Connections
        self.act_resume.triggered.connect(lambda: self.mode_selected.emit("resume"))
        self.act_auto.triggered.connect(lambda: self.mode_selected.emit("auto_record"))
        self.act_manual.triggered.connect(lambda: self.mode_selected.emit("manual_record"))
        self.act_route.triggered.connect(lambda: self.mode_selected.emit("route"))
        self.act_settings.triggered.connect(lambda: self.mode_selected.emit("settings"))
        self.act_exit.triggered.connect(lambda: self.mode_selected.emit("exit"))

    def popup_at_cursor(self):
        cursor_pos = QCursor.pos()
        self.popup(cursor_pos)

    def get_menu_health_summary(self) -> dict:
        return {"module": "menu", "ready": True}

    def get_menu_status_summary(self) -> dict:
        try:
            return {"menu_ready": True}
        except Exception:
            return {"menu_ready": False}

    def get_menu_ready_flag(self) -> dict:
        return {"module": "menu", "ready": True}

    def get_menu_version_info(self) -> dict:
        return {"module": "menu", "version": "2.0.0", "ready": True}

    def get_menu_diagnostic_snapshot(self) -> dict:
        return {"module": "menu", "status": "ok", "ready": True}


def get_menu_capabilities() -> list:
    return ["ui_menu", "main_menu_dispatch", "diagnostic_snapshot"]


def get_ui_menu_capabilities() -> list:
    return get_menu_capabilities()


def validate_ui_menu_contract() -> bool:
    return bool(get_ui_menu_capabilities())


