# -*- coding: utf-8 -*-
"""
Full Interactive Map Viewer Dialog
Displays the captured game region map with numbered node circles:
- Green circles: Active route nodes
- Orange circles: Resting / cooldown nodes
- Cyan dot: Player / Target node
"""

import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
from PyQt6.QtCore import Qt, QPoint
from core.route_manager import RouteManager

class MapViewerDialog(QDialog):
    def __init__(self, route_manager: RouteManager, map_image_path="video_analysis/map_at_25s.jpg", parent=None):
        super().__init__(parent)
        self.routes = route_manager
        self.map_path = map_image_path
        self.setWindowTitle("Gatherion - Route Map Viewer")
        self.resize(1000, 650)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.map_lbl = QLabel()
        self.map_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.map_lbl)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        self.render_map()

    def render_map(self):
        if os.path.exists(self.map_path):
            pixmap = QPixmap(self.map_path)
        else:
            pixmap = QPixmap(1280, 720)
            pixmap.fill(QColor(30, 34, 42))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw nodes
        nodes = self.routes.get_route_nodes()
        for n in nodes:
            x = n.get("map_x", 0)
            y = n.get("map_y", 0)
            node_id = str(n.get("id", ""))

            # Green circle with black border
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.setBrush(QBrush(QColor(74, 222, 128))) # Green
            painter.drawEllipse(QPoint(x, y), 8, 8)

            # Node number text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            painter.drawText(x + 10, y + 4, node_id)

        painter.end()
        self.map_lbl.setPixmap(pixmap)
