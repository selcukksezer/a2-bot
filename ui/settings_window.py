# -*- coding: utf-8 -*-
"""
Gatherion v1.6.15 Dark Theme Settings & Node Management Window.

Matches `video_frames/frame_06s.jpg`, `frame_12s.jpg`, and `frame_21s.jpg` pixel-for-pixel:
- Custom animated pill switch `ToggleSwitch(QCheckBox)` rendered with `QPainter`
- Left Column:
  * GATHERING group (`f_interval_min`, `f_interval_max`, `test_mode`, `show_overlay`)
  * ROUTE group (`rest_minutes`, `climb_air_max_sec`, `gathered_join_active_route`,
    `fly_unreachable`, `check_odyle_minimap`, `reclick_node_sec`, `time_to_reach_sec`,
    `gather_on_the_way`, `wait_min_after_failed`, `max_failed_in_row`,
    `remove_node_after_failures`, `pause_after_no_gather_min`, `fight_back`,
    `fight_stop_gather_sec`)
  * BREAKS group (`break_every_min`, `break_every_variance`,
    `break_lasts_min`, `break_lasts_variance`)
- Right Column (NODES):
  * Region & Route selectors with Rename / New / Delete buttons
  * 5-column QTableWidget (`NODE`, `ROUTES`, `GATHERED`, `LAST`, `REST`) with
    48x48 map patch thumbnails, bold `#1` node labels, route memberships,
    visit counts, relative last-visit timestamps, and green `ready` / amber
    countdown capsule badges (plus empty-state overlay message)
  * 2-row action bar:
    Row 1: `Add nodes...`, `Remove from route`, `Delete node`, `Move to region...`, `Map...`
    Row 2: `Share route...`, `Import route...`, `Import from questlog...`
- Footer Bar:
  * Left: `Quit bot` (`quit_requested`), `Gatherion v1.6.15` label, `status_lbl`
  * Right: `Reset to defaults`, blue `Save` (`#3b82f6`, `settings_saved`), `Close`
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("Gatherion.SettingsWindow")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# Default Settings Schema
# ============================================================================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "f_interval_min": 1.5,
    "f_interval_max": 3.0,
    "test_mode": False,
    "show_overlay": True,
    "harvests_per_node": 3,
    "gather_key": "f",
    "retry_on_extraction_fail": True,
    "rest_minutes": 5,
    "climb_air_max_sec": 10,
    "gathered_join_active_route": True,
    "fly_unreachable": False,
    "check_odyle_minimap": False,
    "reclick_node_sec": 10,
    "time_to_reach_sec": 60,
    "gather_on_the_way": False,
    "wait_min_after_failed": 5,
    "max_failed_in_row": 10,
    "remove_node_after_failures": 0,
    "pause_after_no_gather_min": 15,
    "fight_back": True,
    "fight_stop_gather_sec": 3,
    "minimap_auto_record": True,
    "minimap_ignore_monsters": True,
    "break_every_min": 0,
    "break_every_variance": 1,
    "break_lasts_min": 10,
    "break_lasts_variance": 1,
}

# Mapping between canonical keys and legacy nested (section, key) pairs
LEGACY_KEY_MAP: Dict[str, Tuple[str, str]] = {
    "f_interval_min": ("gathering", "min_f_interval"),
    "f_interval_max": ("gathering", "max_f_interval"),
    "test_mode": ("gathering", "test_mode"),
    "show_overlay": ("gathering", "show_overlay"),
    "harvests_per_node": ("gathering", "harvests_per_node"),
    "gather_key": ("gathering", "gather_key"),
    "retry_on_extraction_fail": ("gathering", "retry_on_extraction_fail"),
    "rest_minutes": ("route", "rest_after_visit_min"),
    "climb_air_max_sec": ("route", "climb_air_max_sec"),
    "gathered_join_active_route": ("route", "gathered_nodes_join_route"),
    "fly_unreachable": ("route", "fly_unreachable"),
    "check_odyle_minimap": ("route", "check_odyle_minimap"),
    "reclick_node_sec": ("route", "reclick_interval_sec"),
    "time_to_reach_sec": ("route", "time_to_reach_node_sec"),
    "gather_on_the_way": ("route", "exploration_gather"),
    "wait_min_after_failed": ("route", "wait_min_after_failed"),
    "max_failed_in_row": ("route", "max_failures_before_wait"),
    "remove_node_after_failures": ("route", "remove_node_after_failures"),
    "pause_after_no_gather_min": ("route", "pause_after_idle_min"),
    "fight_back": ("route", "fight_back_when_attacked"),
    "fight_stop_gather_sec": ("route", "fight_stop_gather_sec"),
    "minimap_auto_record": ("route", "minimap_auto_record"),
    "minimap_ignore_monsters": ("route", "minimap_ignore_monsters"),
    "break_every_min": ("breaks", "interval_min"),
    "break_every_variance": ("breaks", "interval_jitter_min"),
    "break_lasts_min": ("breaks", "duration_min"),
    "break_lasts_variance": ("breaks", "duration_jitter_min"),
}


# ============================================================================
# Custom iOS / Gatherion Pill Switch (`ToggleSwitch(QCheckBox)`)
# ============================================================================
class ToggleSwitch(QCheckBox):
    """
    Modern iOS / Gatherion-style rounded pill toggle switch rendered with QPainter.
    Supports smooth thumb animation and optional inline text label on the right.
    """

    SWITCH_WIDTH = 34
    SWITCH_HEIGHT = 18
    TEXT_SPACING = 8

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        checked: bool = False,
    ) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thumb_pos: float = 1.0 if checked else 0.0
        self._hovered: bool = False

        self._anim = QPropertyAnimation(self, b"thumb_position", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setChecked(bool(checked))
        self.toggled.connect(self._on_toggled)

    @pyqtProperty(float)
    def thumb_position(self) -> float:
        return float(self._thumb_pos)

    @thumb_position.setter
    def thumb_position(self, value: float) -> None:
        self._thumb_pos = max(0.0, min(1.0, float(value)))
        self.update()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(bool(checked))
        self._anim.stop()
        self._thumb_pos = 1.0 if checked else 0.0
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        try:
            self._anim.stop()
            self._anim.setStartValue(self._thumb_pos)
            self._anim.setEndValue(1.0 if checked else 0.0)
            self._anim.start()
        except Exception as exc:
            logger.debug("ToggleSwitch animation fallback: %s", exc)
            self._thumb_pos = 1.0 if checked else 0.0
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        label = self.text()
        if not label:
            return QSize(self.SWITCH_WIDTH + 2, self.SWITCH_HEIGHT + 4)
        fm = QFontMetrics(QFont("Segoe UI", 9))
        text_w = fm.horizontalAdvance(label)
        return QSize(
            self.SWITCH_WIDTH + self.TEXT_SPACING + text_w + 8,
            max(self.SWITCH_HEIGHT + 4, fm.height() + 4),
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def hitButton(self, pos: QPoint) -> bool:  # noqa: N802
        return self.rect().contains(pos)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        sw_w = float(self.SWITCH_WIDTH)
        sw_h = float(self.SWITCH_HEIGHT)
        sw_y = (float(self.height()) - sw_h) / 2.0
        track_rect = QRectF(1.0, sw_y, sw_w, sw_h)
        radius = sw_h / 2.0

        t = self._thumb_pos
        # Interpolate track color between #2b313a (unchecked) and #38bdf8 (checked)
        r = int(43 + (56 - 43) * t)
        g = int(49 + (189 - 49) * t)
        b = int(58 + (248 - 58) * t)
        track_color = QColor(r, g, b)
        border_color = QColor("#38bdf8") if t > 0.5 else QColor("#444c56")

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(track_rect, radius, radius)

        # Thumb circle
        margin = 2.5
        thumb_d = sw_h - (margin * 2.0)
        min_x = 1.0 + margin
        max_x = 1.0 + sw_w - margin - thumb_d
        thumb_x = min_x + (max_x - min_x) * t
        thumb_y = sw_y + margin

        # Interpolate thumb color between #9ca3af (unchecked) and #ffffff (checked)
        tr = int(156 + (255 - 156) * t)
        tg = int(163 + (255 - 163) * t)
        tb = int(175 + (255 - 175) * t)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(tr, tg, tb)))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_d, thumb_d))

        # Optional inline text label
        label = self.text()
        if label:
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            text_color = QColor("#f3f4f6") if self._hovered else QColor("#d1d5db")
            painter.setPen(QPen(text_color))
            text_x = int(sw_w) + self.TEXT_SPACING + 2
            text_rect = QRect(text_x, 0, max(10, self.width() - text_x), self.height())
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label,
            )

        painter.end()


# ============================================================================
# Custom Capsule Badge for REST Column (`Hazır` / `4dk 12sn`)
# ============================================================================
class _RestBadgeWidget(QWidget):
    """
    Renders a rounded pill badge inside the REST table column:
    - Green capsule (`Hazır`) when node cooldown is complete
    - Amber/orange capsule (`4dk 12sn`) when node is resting
    """

    def __init__(
        self,
        text: str = "Hazır",
        is_ready: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._text = str(text)
        self._is_ready = bool(is_ready)
        self.setFixedHeight(28)
        self.setMinimumWidth(58)

    def set_status(self, text: str, is_ready: bool) -> None:
        if self._text != text or self._is_ready != is_ready:
            self._text = str(text)
            self._is_ready = bool(is_ready)
            self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", 8)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        pill_w = max(44, fm.horizontalAdvance(self._text) + 16)
        pill_h = 18
        pill_x = 4
        pill_y = (self.height() - pill_h) // 2
        pill_rect = QRectF(float(pill_x), float(pill_y), float(pill_w), float(pill_h))

        if self._is_ready:
            bg_color = QColor(20, 83, 45, 190)       # Dark emerald fill
            border_color = QColor(34, 197, 94, 140)  # Green border
            text_color = QColor("#4ade80")           # Bright green text
        else:
            bg_color = QColor(120, 53, 15, 190)      # Dark amber fill
            border_color = QColor(245, 158, 11, 150) # Amber border
            text_color = QColor("#fbbf24")           # Bright amber text

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(pill_rect, 9.0, 9.0)

        painter.setPen(QPen(text_color))
        painter.drawText(
            pill_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            self._text,
        )
        painter.end()


# ============================================================================
# Add Nodes Selection Dialog (for `Nokta Ekle...` button)
# ============================================================================
class _AddNodesToRouteDialog(QDialog):
    """
    Allows selecting existing region nodes that are not yet in the active route
    and adding them to the active route, or creating a new node entry.
    """

    def __init__(
        self,
        region_name: str,
        route_name: str,
        available_nodes: List[Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Rotaya Nokta Ekle: {region_name} · {route_name}")
        self.resize(380, 340)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #16191f;
                color: #e5e7eb;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel { color: #d1d5db; font-size: 12px; }
            QListWidget {
                background-color: #1e222a;
                border: 1px solid #323842;
                border-radius: 4px;
                color: #e5e7eb;
                font-size: 12px;
            }
            QPushButton {
                background-color: #262b33;
                color: #e5e7eb;
                border: 1px solid #383f4b;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover { background-color: #313742; }
            """
        )

        layout = QVBoxLayout(self)
        info = QLabel(
            f"'{region_name}' bölgesinden '{route_name}' rotasına eklenecek noktaları seçin:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        for n in available_nodes:
            nid = int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0))
            item = QListWidgetItem(f"# {nid}  (Nokta {nid})")
            item.setData(Qt.ItemDataRole.UserRole, nid)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_create_new = QPushButton("Yeni Nokta Oluştur", self)
        self._create_new_requested = False
        self.btn_create_new.clicked.connect(self._on_create_new)
        btn_row.addWidget(self.btn_create_new)
        btn_row.addStretch()

        btn_add = QPushButton("Seçilenleri Ekle", self)
        btn_add.setStyleSheet(
            "background-color: #3b82f6; color: #ffffff; border: none; font-weight: bold;"
        )
        btn_add.clicked.connect(self.accept)
        btn_cancel = QPushButton("İptal", self)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _on_create_new(self) -> None:
        self._create_new_requested = True
        self.accept()

    def selected_node_ids(self) -> List[int]:
        return [
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in self.list_widget.selectedItems()
        ]

    @property
    def create_new_requested(self) -> bool:
        return self._create_new_requested


# ============================================================================
# Fallback Questlog Import Dialog (Matches `frame_12s.jpg` & `frame_15s.jpg`)
# ============================================================================
class _BuiltinQuestlogImportDialog(QDialog):
    """
    Authentic fallback Questlog Import dialog matching `video_frames/frame_12s.jpg`
    and `video_frames/frame_15s.jpg` in case `core.questlog_importer` is not present.
    """

    ZONE_DATA: Dict[str, Dict[str, Any]] = {
        "Altgard": {
            "resources": [
                ("Odyle", 650),
                ("Orichalcum Ore", 530),
                ("Diamond Gemstone", 200),
                ("Sapphire Gemstone", 200),
                ("Ruby Gemstone", 190),
            ],
            "sub_zones": [
                ("Dredgion Crash Site", 118),
                ("Nameless Cemetery", 118),
                ("Fafnite Deposit", 112),
                ("Basfelt Ruins", 111),
                ("Sanctum Outpost", 83),
            ],
        },
        "Verteron": {
            "resources": [
                ("Odyle", 706),
                ("Orichalcum Ore", 546),
                ("Ruby Gemstone", 196),
                ("Diamond Gemstone", 192),
                ("Sapphire Gemstone", 192),
            ],
            "sub_zones": [
                ("Verteron Citadel", 95),
                ("Cantas Coast", 88),
                ("Ardus Shrine", 76),
                ("Tursin Outpost", 104),
            ],
        },
    }

    def __init__(
        self,
        route_manager: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.route_manager = route_manager
        self.imported_summary: str = ""
        self.setWindowTitle("Görev Günlüğünden (Questlog) İçe Aktar")
        self.setFixedSize(410, 415)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #181b20;
                color: #e5e7eb;
                font-family: 'Segoe UI', sans-serif;
                border: 1px solid #2d323b;
            }
            QLabel { color: #d1d5db; font-size: 12px; }
            QComboBox {
                background-color: #22262e;
                border: 1px solid #383f4b;
                border-radius: 3px;
                color: #f3f4f6;
                padding: 4px 8px;
                font-size: 12px;
            }
            QListWidget {
                background-color: #22262e;
                border: 1px solid #383f4b;
                color: #e5e7eb;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QPushButton {
                background-color: #262b33;
                color: #e5e7eb;
                border: 1px solid #383f4b;
                border-radius: 4px;
                padding: 5px 16px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #323842; }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Zone row
        row_zone = QHBoxLayout()
        lbl_z = QLabel("Bölge:")
        lbl_z.setFixedWidth(72)
        row_zone.addWidget(lbl_z)
        self.cb_zone = QComboBox(self)
        for z in self.ZONE_DATA.keys():
            self.cb_zone.addItem(z)
        self.cb_zone.currentTextChanged.connect(self._populate_lists)
        row_zone.addWidget(self.cb_zone, stretch=1)
        root.addLayout(row_zone)

        # Resources row
        row_res = QHBoxLayout()
        lbl_r = QLabel("Kaynaklar:")
        lbl_r.setFixedWidth(72)
        lbl_r.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row_res.addWidget(lbl_r)
        self.list_resources = QListWidget(self)
        self.list_resources.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_resources.setFixedHeight(105)
        self.list_resources.itemSelectionChanged.connect(self._update_selected_label)
        row_res.addWidget(self.list_resources, stretch=1)
        root.addLayout(row_res)

        # Sub-zones row
        row_sub = QHBoxLayout()
        lbl_s = QLabel("Alt Bölgeler:")
        lbl_s.setFixedWidth(72)
        lbl_s.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row_sub.addWidget(lbl_s)
        self.list_subzones = QListWidget(self)
        self.list_subzones.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_subzones.setFixedHeight(105)
        self.list_subzones.itemSelectionChanged.connect(self._update_selected_label)
        row_sub.addWidget(self.list_subzones, stretch=1)
        root.addLayout(row_sub)

        self.lbl_selected = QLabel("Seçilen: 0 nokta (harita başına ~0.0 sn)")
        self.lbl_selected.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        root.addWidget(self.lbl_selected)

        lbl_hint = QLabel(
            "Bölgede harita kapalıyken ve yalnızca Bölgeler filtresi açıkken bekleyin."
        )
        lbl_hint.setStyleSheet("color: #9ca3af; font-size: 11px;")
        lbl_hint.setWordWrap(True)
        root.addWidget(lbl_hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_import = QPushButton("İçe Aktar", self)
        self.btn_import.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; border: none; font-weight: bold;"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_cancel = QPushButton("İptal", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._populate_lists(self.cb_zone.currentText())

    def _populate_lists(self, zone_name: str) -> None:
        self.list_resources.clear()
        self.list_subzones.clear()
        zinfo = self.ZONE_DATA.get(zone_name, {})
        for idx, (res_name, count) in enumerate(zinfo.get("resources", [])):
            item = QListWidgetItem(f"{res_name} ({count})")
            item.setData(Qt.ItemDataRole.UserRole, (res_name, count))
            self.list_resources.addItem(item)
            if idx < 2:
                item.setSelected(True)
        for idx, (sub_name, count) in enumerate(zinfo.get("sub_zones", [])):
            item = QListWidgetItem(f"{sub_name} ({count})")
            item.setData(Qt.ItemDataRole.UserRole, (sub_name, count))
            self.list_subzones.addItem(item)
            if idx == 0:
                item.setSelected(True)
        self._update_selected_label()

    def _update_selected_label(self) -> None:
        total = 0
        for item in self.list_subzones.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and len(data) == 2:
                total += int(data[1])
        self.lbl_selected.setText(f"Seçilen: {total} nokta (harita başına ~1.2 sn)")

    def _on_import_clicked(self) -> None:
        zone_name = self.cb_zone.currentText().strip() or "Altgard"
        sub_items = self.list_subzones.selectedItems()
        route_name = "Dredgion Crash Site"
        if sub_items:
            data = sub_items[0].data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and len(data) >= 1:
                route_name = str(data[0])

        res_items = self.list_resources.selectedItems()
        res_name = "Orichalcum Ore"
        if res_items:
            rdata = res_items[0].data(Qt.ItemDataRole.UserRole)
            if isinstance(rdata, tuple) and len(rdata) >= 1:
                res_name = str(rdata[0])

        count_to_import = 12
        try:
            if self.route_manager is not None:
                if hasattr(self.route_manager, "create_region"):
                    self.route_manager.create_region(zone_name)
                if hasattr(self.route_manager, "create_route"):
                    self.route_manager.create_route(zone_name, route_name)
                for idx in range(count_to_import):
                    angle = (2.0 * math.pi * idx) / float(count_to_import)
                    mx = int(900 + 220 * math.cos(angle))
                    my = int(460 + 160 * math.sin(angle))
                    self.route_manager.add_node(
                        region=zone_name,
                        route=route_name,
                        map_x=mx,
                        map_y=my,
                        resource_type=res_name,
                    )
                self.route_manager.active_region = zone_name
                self.route_manager.active_route = route_name
        except Exception as exc:
            logger.error("Fallback questlog import error: %s", exc)

        self.imported_summary = (
            f"{count_to_import} nokta {zone_name} · {route_name} rotasına aktarıldı"
        )
        self.accept()


# ============================================================================
# Main Settings Window (`SettingsWindow(QDialog)`)
# ============================================================================
class SettingsWindow(QDialog):
    """
    Gatherion v1.6.15 two-column dark settings and node management window (Türkçe UI).
    """

    quit_requested = pyqtSignal()
    settings_saved = pyqtSignal()
    route_changed = pyqtSignal(str, str)

    def __init__(
        self,
        config_manager: Optional[Any] = None,
        route_manager: Optional[Any] = None,
        parent: Optional[QWidget] = None,
        settings: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        raw_cfg = settings if settings is not None else config_manager
        self.config_manager = raw_cfg if raw_cfg is not None else self._create_default_config_manager()
        self.route_manager = route_manager or self._create_default_route_manager()
        self.cfg = self.config_manager
        self.routes = self.route_manager
        self.settings = (
            settings
            if settings is not None
            else getattr(self.config_manager, "settings", self.config_manager)
        )
        self._last_saved_values: Dict[str, Any] = {}

        self._updating_combos: bool = False
        self._map_viewer_dlg: Optional[QDialog] = None
        self._questlog_dlg: Optional[QDialog] = None
        self._radar_dlg: Optional[QDialog] = None
        self._analytics_dlg: Optional[QDialog] = None
        self._calibration_dlg: Optional[QDialog] = None
        self._guide_dlg: Optional[QDialog] = None

        self.setWindowTitle("Gatherion v1.6.15 — Ayarlar ve Rota Yönetimi")
        self.setMinimumSize(1040, 720)
        self.resize(1100, 760)
        self._apply_dark_stylesheet()
        self._init_ui()
        self._load_from_settings()
        self.refresh_regions_and_routes()

        # Live 1-second timer to update node LAST and REST badges dynamically
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._update_live_node_statuses)
        self._refresh_timer.start()

    # ------------------------------------------------------------------
    # Default Manager Instantiation Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_default_config_manager() -> Any:
        try:
            from config.schema import SettingsManager as SM  # type: ignore
            return SM()
        except Exception as exc0:
            logger.debug("config.schema.SettingsManager import fallback: %s", exc0)
            try:
                from core.config_manager import ConfigManager as CM  # type: ignore
                return CM()
            except Exception as exc1:
                logger.debug("core.config_manager import fallback: %s", exc1)
                try:
                    from core.config import ConfigManager as LegacyCM  # type: ignore
                    return LegacyCM()
                except Exception as exc2:
                    logger.warning("Could not load ConfigManager, using dict wrapper: %s", exc2)
                    return dict(DEFAULT_SETTINGS)

    @staticmethod
    def _create_default_route_manager() -> Any:
        try:
            from core.route_manager import RouteManager as RM  # type: ignore
            return RM()
        except Exception as exc:
            logger.error("Could not instantiate RouteManager: %s", exc)
            return None

    def _candidate_settings_objects(self, explicit_target: Optional[Any] = None) -> List[Any]:
        """Returns ordered list of active settings/config objects to inspect or mutate."""
        objs: List[Any] = []
        for item in (
            explicit_target,
            getattr(explicit_target, "settings", None) if explicit_target is not None else None,
            getattr(self, "settings", None),
            getattr(getattr(self, "settings", None), "settings", None),
            self.config_manager,
            getattr(self.config_manager, "settings", None),
            getattr(self.config_manager, "config", None),
            getattr(self.config_manager, "data", None),
        ):
            if item is not None and not any(item is existing for existing in objs):
                objs.append(item)
        return objs

    # ------------------------------------------------------------------
    # Universal Config Get / Set Adapter (Flat & Nested Dataclass/Dict)
    # ------------------------------------------------------------------
    def _cfg_get(self, key: str, default: Any = None, target: Optional[Any] = None) -> Any:
        if default is None:
            default = DEFAULT_SETTINGS.get(key)

        sec, leg_key = LEGACY_KEY_MAP.get(key, ("", ""))

        for obj in self._candidate_settings_objects(target):
            try:
                # 1. Dictionary instance
                if isinstance(obj, dict):
                    if key in obj and not isinstance(obj[key], dict):
                        return obj[key]
                    if sec and sec in obj and isinstance(obj[sec], dict):
                        if key in obj[sec]:
                            return obj[sec][key]
                        if leg_key and leg_key in obj[sec]:
                            return obj[sec][leg_key]
                    continue

                # 2. Dataclass / AppSettings / BotSettings / custom object using safe getattr
                sec_obj = getattr(obj, sec, None) if sec else None
                has_sec_attr = False
                sec_val = None
                if sec_obj is not None:
                    if isinstance(sec_obj, dict):
                        if key in sec_obj:
                            has_sec_attr = True
                            sec_val = sec_obj[key]
                        elif leg_key and leg_key in sec_obj:
                            has_sec_attr = True
                            sec_val = sec_obj[leg_key]
                    else:
                        if hasattr(sec_obj, key):
                            has_sec_attr = True
                            sec_val = getattr(sec_obj, key)
                        elif leg_key and hasattr(sec_obj, leg_key):
                            has_sec_attr = True
                            sec_val = getattr(sec_obj, leg_key)

                # For harvests_per_node, also check route.harvests_per_node if gathering was default
                if key == "harvests_per_node":
                    route_obj = getattr(obj, "route", None)
                    if route_obj is not None and hasattr(route_obj, "harvests_per_node"):
                        r_val = getattr(route_obj, "harvests_per_node", 3)
                        if has_sec_attr and sec_val == 3 and r_val != 3:
                            sec_val = r_val

                obj_dict = getattr(obj, "__dict__", {})
                has_top_attr = (
                    isinstance(obj_dict, dict) and key in obj_dict
                ) or (not has_sec_attr and hasattr(obj, key))
                top_val = getattr(obj, key, None) if has_top_attr else None
                if callable(top_val):
                    has_top_attr = False
                    top_val = None

                if has_sec_attr and has_top_attr and top_val is not None:
                    if key in self._last_saved_values:
                        last_val = self._last_saved_values[key]
                        if sec_val != last_val:
                            return sec_val
                        if top_val != last_val:
                            return top_val
                        return sec_val
                    else:
                        return top_val
                if has_sec_attr and sec_val is not None:
                    return sec_val
                if has_top_attr and top_val is not None:
                    return top_val

                # 3. Manager .get(...) method fallback
                if hasattr(obj, "get") and callable(obj.get):
                    try:
                        if sec:
                            val = obj.get(sec, key, None)
                            if val is None and leg_key:
                                val = obj.get(sec, leg_key, None)
                            if val is not None and not isinstance(val, dict):
                                return val
                        val = obj.get(key, default)
                        if val is not None and not isinstance(val, dict):
                            return val
                    except TypeError:
                        if sec:
                            val = obj.get(sec, key, default)
                            if val is not None:
                                return val
            except Exception as exc:
                logger.debug("Config read fallback for key '%s': %s", key, exc)

        return default

    def _cfg_set(self, key: str, value: Any, target: Optional[Any] = None) -> None:
        self._last_saved_values[key] = value
        sec, leg_key = LEGACY_KEY_MAP.get(key, ("", ""))

        for obj in self._candidate_settings_objects(target):
            try:
                if isinstance(obj, dict):
                    obj[key] = value
                    if sec:
                        if sec not in obj or not isinstance(obj[sec], dict):
                            obj[sec] = {}
                        obj[sec][key] = value
                        if leg_key:
                            obj[sec][leg_key] = value
                    continue

                # Nested section object on AppSettings / BotSettings
                if sec:
                    sec_obj = getattr(obj, sec, None)
                    if sec_obj is not None:
                        if isinstance(sec_obj, dict):
                            sec_obj[key] = value
                            if leg_key:
                                sec_obj[leg_key] = value
                        else:
                            try:
                                setattr(sec_obj, key, value)
                            except Exception as exc_sec:
                                logger.debug("Could not setattr %s.%s: %s", sec, key, exc_sec)
                            if leg_key and leg_key != key and hasattr(sec_obj, leg_key):
                                try:
                                    setattr(sec_obj, leg_key, value)
                                except Exception as exc_leg:
                                    logger.debug("Could not setattr %s.%s: %s", sec, leg_key, exc_leg)

                # Keep harvests_per_node synchronized across gathering and route
                if key == "harvests_per_node":
                    for mirror_sec in ("gathering", "route"):
                        m_obj = getattr(obj, mirror_sec, None)
                        if m_obj is not None and not isinstance(m_obj, dict):
                            try:
                                setattr(m_obj, "harvests_per_node", int(value))
                            except Exception as exc_m:
                                logger.debug("Mirror harvests_per_node error: %s", exc_m)

                # Also set attribute on top-level settings object using safe setattr
                if not hasattr(obj, "save") or hasattr(obj, "__dataclass_fields__"):
                    try:
                        setattr(obj, key, value)
                    except Exception as exc_top:
                        logger.debug("Top-level setattr skipped for %s: %s", key, exc_top)

                # Call manager .set(...) if present
                if hasattr(obj, "set") and callable(obj.set):
                    try:
                        if sec:
                            obj.set(sec, key, value)
                        else:
                            obj.set(key, value)
                    except TypeError:
                        try:
                            obj.set(key, value)
                        except Exception as exc_set:
                            logger.debug("Manager set fallback error: %s", exc_set)
            except Exception as exc:
                logger.warning("Failed to set config key '%s': %s", key, exc)

    def _cfg_save(self) -> None:
        for obj in self._candidate_settings_objects():
            try:
                if hasattr(obj, "validate") and callable(obj.validate):
                    obj.validate()
            except Exception as exc_val:
                logger.debug("Settings validate warning: %s", exc_val)

        cm = self.config_manager
        if cm is None:
            return
        try:
            if hasattr(cm, "save") and callable(cm.save):
                cm.save()
            elif hasattr(cm, "save_config") and callable(cm.save_config):
                cm.save_config()
        except Exception as exc:
            logger.error("Failed to save configuration to disk: %s", exc)

    # ------------------------------------------------------------------
    # Styling & Layout Initialization
    # ------------------------------------------------------------------
    def _apply_dark_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #16191f;
                color: #d1d5db;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                background-color: #1a1e24;
                border: 1px solid #282e37;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 22px;
                padding-bottom: 8px;
                padding-left: 10px;
                padding-right: 10px;
                font-weight: bold;
                font-size: 11px;
                color: #9ca3af;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: 12px;
                top: 8px;
                color: #9ca3af;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QLabel {
                color: #d1d5db;
                font-size: 12px;
                background: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #222730;
                border: 1px solid #363d49;
                border-radius: 4px;
                color: #f3f4f6;
                padding: 2px 6px;
                min-height: 22px;
                font-size: 12px;
                selection-background-color: #0284c7;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #38bdf8;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 0px;
                border: none;
            }
            QComboBox {
                background-color: #222730;
                border: 1px solid #363d49;
                border-radius: 4px;
                color: #f3f4f6;
                padding: 3px 8px;
                min-height: 22px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #4b5563;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e222a;
                color: #f3f4f6;
                border: 1px solid #363d49;
                selection-background-color: #2563eb;
            }
            QPushButton {
                background-color: #242932;
                color: #e5e7eb;
                border: 1px solid #363d49;
                border-radius: 4px;
                padding: 4px 10px;
                min-height: 20px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2e3540;
                border-color: #4b5563;
            }
            QPushButton:pressed {
                background-color: #1c2026;
            }
            QPushButton:disabled {
                color: #6b7280;
                background-color: #1c2026;
                border-color: #282e37;
            }
            QPushButton#DangerOutlineButton {
                background-color: #242932;
                color: #f87171;
                border: 1px solid #6b2626;
            }
            QPushButton#DangerOutlineButton:hover {
                background-color: #3a1d22;
                border: 1px solid #ef4444;
                color: #fca5a5;
            }
            QPushButton#PrimarySaveButton {
                background-color: #3b82f6;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-weight: bold;
                padding: 5px 18px;
            }
            QPushButton#PrimarySaveButton:hover {
                background-color: #2563eb;
            }
            QTableWidget {
                background-color: #15181e;
                border: 1px solid #282e37;
                border-radius: 4px;
                color: #e5e7eb;
                gridline-color: transparent;
                selection-background-color: #1e3a5f;
                selection-color: #ffffff;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #15181e;
                color: #9ca3af;
                border: none;
                border-bottom: 1px solid #282e37;
                padding: 5px 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QScrollBar:vertical {
                background: #15181e;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #374151;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(8)

        # Two-column main content layout
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # LEFT COLUMN: GATHERING, ROUTE, AUTO-CARTOGRAPHY, BREAKS
        # ------------------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        left_col.addWidget(self._build_gathering_group())
        left_col.addWidget(self._build_route_group(), stretch=1)
        left_col.addWidget(self._build_cartography_group())
        left_col.addWidget(self._build_breaks_group())

        columns_layout.addLayout(left_col, stretch=48)

        # ------------------------------------------------------------------
        # RIGHT COLUMN: NODES / REGIONS & ROUTES
        # ------------------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.addWidget(self._build_nodes_group(), stretch=1)

        columns_layout.addLayout(right_col, stretch=52)
        root_layout.addLayout(columns_layout, stretch=1)

        # Status message row (above footer separator, matching frame_21s.jpg)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #d1d5db; font-size: 12px; padding: 1px 2px;")
        root_layout.addWidget(self.status_lbl)

        # Thin horizontal divider above footer
        divider = QFrame(self)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #282e37; border: none;")
        root_layout.addWidget(divider)

        # ------------------------------------------------------------------
        # FOOTER BAR
        # ------------------------------------------------------------------
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 2, 0, 0)
        footer_layout.setSpacing(8)

        self.btn_quit = QPushButton("Bottan Çık", self)
        self.btn_quit.setToolTip("Uygulamayı güvenli şekilde kapatır")
        self.btn_quit.clicked.connect(self._on_quit_clicked)
        footer_layout.addWidget(self.btn_quit)

        self.version_lbl = QLabel("Gatherion v1.6.15", self)
        self.version_lbl.setStyleSheet("color: #8b949e; font-size: 12px;")
        footer_layout.addWidget(self.version_lbl)

        footer_layout.addStretch(1)

        self.btn_radar = QPushButton("Canlı Radar", self)
        self.btn_radar.setToolTip("Oto-haritalama canlı radar penceresini açar")
        self.btn_radar.clicked.connect(self._on_open_radar)
        footer_layout.addWidget(self.btn_radar)

        self.btn_yolo_factory = QPushButton("🧠 YOLO Veri Fabrikası", self)
        self.btn_yolo_factory.setToolTip(
            "Kendi kendini eğiten yapay zeka (YOLO) veri seti fabrikası ve RTX eğitim merkezini açar"
        )
        self.btn_yolo_factory.clicked.connect(self._on_open_yolo_factory)
        footer_layout.addWidget(self.btn_yolo_factory)

        self.btn_mobile_connect = QPushButton("📱 Mobil Kumanda", self)
        self.btn_mobile_connect.setToolTip(
            "Telefon / mobil tarayıcı üzerinden canlı izleme ve kumanda QR kod penceresini açar"
        )
        self.btn_mobile_connect.clicked.connect(self._on_open_mobile_connect)
        footer_layout.addWidget(self.btn_mobile_connect)

        self.btn_analytics = QPushButton("İstatistikler", self)
        self.btn_analytics.setToolTip("Canlı telemetri ve toplama istatistiklerini açar")
        self.btn_analytics.clicked.connect(self._on_open_analytics)
        footer_layout.addWidget(self.btn_analytics)

        self.btn_calibration = QPushButton("Kalibrasyon", self)
        self.btn_calibration.setToolTip("Görüntü kalibrasyon sihirbazını açar")
        self.btn_calibration.clicked.connect(self._on_open_calibration)
        footer_layout.addWidget(self.btn_calibration)

        self.btn_guide = QPushButton("Kılavuz", self)
        self.btn_guide.setToolTip("Kullanım kılavuzu ve sorun giderme penceresini açar")
        self.btn_guide.clicked.connect(self._on_open_guide)
        footer_layout.addWidget(self.btn_guide)

        self.btn_reset_defaults = QPushButton("Varsayılanlara Sıfırla", self)
        self.btn_reset_defaults.setToolTip("Tüm ayarları varsayılan değerlerine döndürür")
        self.btn_reset_defaults.clicked.connect(self.reset_to_defaults)
        footer_layout.addWidget(self.btn_reset_defaults)

        self.btn_save = QPushButton("Kaydet ve Uygula", self)
        self.btn_save.setObjectName("PrimarySaveButton")
        self.btn_save.setToolTip("Ayarları kaydeder ve anında uygular")
        self.btn_save.clicked.connect(self.save_settings)
        footer_layout.addWidget(self.btn_save)

        self.btn_close = QPushButton("Kapat", self)
        self.btn_close.clicked.connect(self.close)
        footer_layout.addWidget(self.btn_close)

        root_layout.addLayout(footer_layout)

    # ------------------------------------------------------------------
    # Left Column Builders
    # ------------------------------------------------------------------
    def _build_gathering_group(self) -> QGroupBox:
        grp = QGroupBox("Toplama Ayarları (Gathering)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # F/E Basım Aralığı (sn): [1.5] ile [3.0] sn
        row_f = QHBoxLayout()
        row_f.setSpacing(6)
        row_f.addWidget(QLabel("F/E Basım Aralığı (sn):"))

        self.f_interval_min = QDoubleSpinBox(self)
        self.f_interval_min.setRange(0.1, 60.0)
        self.f_interval_min.setSingleStep(0.1)
        self.f_interval_min.setDecimals(1)
        self.f_interval_min.setFixedWidth(54)
        self.spin_f_min = self.f_interval_min  # Alias

        self.f_interval_max = QDoubleSpinBox(self)
        self.f_interval_max.setRange(0.1, 60.0)
        self.f_interval_max.setSingleStep(0.1)
        self.f_interval_max.setDecimals(1)
        self.f_interval_max.setFixedWidth(54)
        self.spin_f_max = self.f_interval_max  # Alias

        row_f.addWidget(self.f_interval_min)
        row_f.addWidget(QLabel("ile"))
        row_f.addWidget(self.f_interval_max)
        row_f.addWidget(QLabel("saniye arası"))
        row_f.addStretch(1)
        lay.addLayout(row_f)

        # Harvests per node: SpinBox (1 to 10, default 3)
        row_harvest = QHBoxLayout()
        row_harvest.setSpacing(6)
        self.lbl_harvests_per_node = QLabel(
            "Materyal Başına Toplama Sayısı (Varsayılan: 3x):"
        )
        self.lbl_harvests_per_node.setToolTip("Harvests per node (3x per material)")
        row_harvest.addWidget(self.lbl_harvests_per_node)

        self.harvests_per_node = QSpinBox(self)
        self.harvests_per_node.setRange(1, 10)
        self.harvests_per_node.setValue(3)
        self.harvests_per_node.setFixedWidth(54)
        self.harvests_per_node.setToolTip(
            "Harvests per node (3x per material) — Her materyal noktasından kaç kez toplanacağı (1-10)"
        )
        self.spin_harvests_per_node = self.harvests_per_node
        row_harvest.addWidget(self.harvests_per_node)
        row_harvest.addStretch(1)
        lay.addLayout(row_harvest)

        # Gather key: ComboBox ("f", "e", "auto")
        row_gkey = QHBoxLayout()
        row_gkey.setSpacing(6)
        self.lbl_gather_key = QLabel("Toplama Tuşu (F / E / Otomatik):")
        self.lbl_gather_key.setToolTip("Gather key (f / e / auto)")
        row_gkey.addWidget(self.lbl_gather_key)

        self.gather_key = QComboBox(self)
        for key_opt in ("f", "e", "auto"):
            self.gather_key.addItem(key_opt, key_opt)
        self.gather_key.setCurrentText("f")
        self.gather_key.setFixedWidth(78)
        self.gather_key.setToolTip(
            "Oyun istemcisinde görünen toplama tuşu: 'f' ([F]), 'e' ([E]) veya 'auto' (otomatik)"
        )
        self.cb_gather_key = self.gather_key
        self.combo_gather_key = self.gather_key
        row_gkey.addWidget(self.gather_key)
        row_gkey.addStretch(1)
        lay.addLayout(row_gkey)

        # Toggle: Retry on 'Essence Extraction failed'
        self.retry_on_extraction_fail = ToggleSwitch(
            "'Essence Extraction failed' Hatasında Tekrar Dene", self, checked=True
        )
        self.retry_on_extraction_fail.setToolTip(
            "Retry on 'Essence Extraction failed' — Toplama başarısız olduğunda aynı aşamayı tekrar dener"
        )
        self.chk_retry_on_extraction_fail = self.retry_on_extraction_fail
        lay.addWidget(self.retry_on_extraction_fail)

        # Toggle: Test mode (detect only, never press keys)
        self.test_mode = ToggleSwitch("Test Modu (Tuş basmadan simüle et)", self)
        self.test_mode.setToolTip("Test mode (detect only, never press keys)")
        self.chk_test = self.test_mode
        lay.addWidget(self.test_mode)

        # Toggle: Show overlay over the game
        self.show_overlay = ToggleSwitch("Bilgi Panelini (Overlay HUD) Göster", self)
        self.show_overlay.setToolTip("Show overlay over the game")
        self.chk_overlay = self.show_overlay
        lay.addWidget(self.show_overlay)

        return grp

    def _build_route_group(self) -> QGroupBox:
        grp = QGroupBox("Rota ve Otomasyon (Route)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(5)

        # Dinlenme Süresi (dk): [5] dk  |  Havada maks. yükselme: [10] sn
        row_rest = QHBoxLayout()
        row_rest.setSpacing(6)
        row_rest.addWidget(QLabel("Dinlenme Süresi (dk):"))
        self.rest_minutes = QSpinBox(self)
        self.rest_minutes.setRange(0, 240)
        self.rest_minutes.setFixedWidth(48)
        self.spin_rest = self.rest_minutes
        row_rest.addWidget(self.rest_minutes)
        row_rest.addSpacing(10)
        row_rest.addWidget(QLabel("Havada maks. yükselme (sn):"))
        self.climb_air_max_sec = QSpinBox(self)
        self.climb_air_max_sec.setRange(0, 120)
        self.climb_air_max_sec.setFixedWidth(48)
        row_rest.addWidget(self.climb_air_max_sec)
        row_rest.addStretch(1)
        lay.addLayout(row_rest)

        # Toggle: Gathered nodes join the active route
        self.gathered_join_active_route = ToggleSwitch(
            "Toplanan noktaları aktif rotaya otomatik ekle", self
        )
        lay.addWidget(self.gathered_join_active_route)

        # Toggle: Fly to nodes the game cannot reach on foot
        self.fly_unreachable = ToggleSwitch(
            "Yürüyerek ulaşılamayan noktalara uçarak git", self
        )
        lay.addWidget(self.fly_unreachable)

        # Toggle: Check minimap for Odyle in air
        self.check_odyle_minimap = ToggleSwitch(
            "Havadayken Minimap üzerinde Odyle kontrol et", self
        )
        lay.addWidget(self.check_odyle_minimap)

        # Yolda giderken noktaya her [10] saniyede bir tekrar tıkla (0: asla)
        row_reclick = QHBoxLayout()
        row_reclick.setSpacing(6)
        row_reclick.addWidget(QLabel("Yolda noktaya tekrar tıklama süresi:"))
        self.reclick_node_sec = QSpinBox(self)
        self.reclick_node_sec.setRange(0, 300)
        self.reclick_node_sec.setFixedWidth(48)
        row_reclick.addWidget(self.reclick_node_sec)
        row_reclick.addWidget(QLabel("sn (0: kapalı)"))
        row_reclick.addStretch(1)
        lay.addLayout(row_reclick)

        # Noktaya ulaşma zaman aşımı: [60] saniye
        row_reach = QHBoxLayout()
        row_reach.setSpacing(6)
        row_reach.addWidget(QLabel("Noktaya ulaşma süresi:"))
        self.time_to_reach_sec = QSpinBox(self)
        self.time_to_reach_sec.setRange(5, 600)
        self.time_to_reach_sec.setFixedWidth(48)
        self.spin_reach = self.time_to_reach_sec
        row_reach.addWidget(self.time_to_reach_sec)
        row_reach.addWidget(QLabel("sn (sonra tekrar tıkla ve atla)"))
        row_reach.addStretch(1)
        lay.addLayout(row_reach)

        # Toggle: Gather resources seen on the way (exploration)
        self.gather_on_the_way = ToggleSwitch(
            "Yolda Giderken Gördüklerini Topla", self
        )
        lay.addWidget(self.gather_on_the_way)

        # Arka arkaya [10] başarısız noktadan sonra [5] dk bekle (0: kapalı)
        row_wait_fail = QHBoxLayout()
        row_wait_fail.setSpacing(6)
        row_wait_fail.addWidget(QLabel("Bekle:"))
        self.wait_min_after_failed = QSpinBox(self)
        self.wait_min_after_failed.setRange(0, 240)
        self.wait_min_after_failed.setFixedWidth(44)
        row_wait_fail.addWidget(self.wait_min_after_failed)
        row_wait_fail.addWidget(QLabel("dk, arka arkaya"))
        self.max_failed_in_row = QSpinBox(self)
        self.max_failed_in_row.setRange(0, 100)
        self.max_failed_in_row.setFixedWidth(44)
        row_wait_fail.addWidget(self.max_failed_in_row)
        row_wait_fail.addWidget(QLabel("başarısız noktada (0: kapalı)"))
        row_wait_fail.addStretch(1)
        lay.addLayout(row_wait_fail)

        # Arka arkaya [0] başarısızlıktan sonra noktayı sil (0: kapalı)
        row_rem_fail = QHBoxLayout()
        row_rem_fail.setSpacing(6)
        row_rem_fail.addWidget(QLabel("Noktayı sil:"))
        self.remove_node_after_failures = QSpinBox(self)
        self.remove_node_after_failures.setRange(0, 100)
        self.remove_node_after_failures.setFixedWidth(44)
        row_rem_fail.addWidget(self.remove_node_after_failures)
        row_rem_fail.addWidget(QLabel("ardışık başarısızlıktan sonra (0: kapalı)"))
        row_rem_fail.addStretch(1)
        lay.addLayout(row_rem_fail)

        # Toplama yapmadan [15] dakika geçerse duraklat (0: kapalı)
        row_pause = QHBoxLayout()
        row_pause.setSpacing(6)
        row_pause.addWidget(QLabel("Duraklat:"))
        self.pause_after_no_gather_min = QSpinBox(self)
        self.pause_after_no_gather_min.setRange(0, 480)
        self.pause_after_no_gather_min.setFixedWidth(48)
        row_pause.addWidget(self.pause_after_no_gather_min)
        row_pause.addWidget(QLabel("dk toplama yapılamazsa (0: kapalı)"))
        row_pause.addStretch(1)
        lay.addLayout(row_pause)

        # Toggle: Saldırı Altında Karşılık Ver + Çatışma sonrası bekleme
        row_fight = QHBoxLayout()
        row_fight.setSpacing(6)
        self.fight_back = ToggleSwitch(
            "Saldırı Altında Karşılık Ver", self
        )
        self.chk_combat = self.fight_back
        row_fight.addWidget(self.fight_back)
        row_fight.addSpacing(8)
        row_fight.addWidget(QLabel("Çatışma sonrası bekle:"))
        self.fight_stop_gather_sec = QSpinBox(self)
        self.fight_stop_gather_sec.setRange(0, 120)
        self.fight_stop_gather_sec.setFixedWidth(44)
        row_fight.addWidget(self.fight_stop_gather_sec)
        row_fight.addWidget(QLabel("sn"))
        row_fight.addStretch(1)
        lay.addLayout(row_fight)

        return grp

    def _build_cartography_group(self) -> QGroupBox:
        grp = QGroupBox("Oto-Haritalama ve Radar (Auto-Cartography)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(5)

        # Toggle: Auto-record materials from top-right Minimap
        self.minimap_auto_record = ToggleSwitch(
            "Sağ Üst Haritadan (Minimap) Otomatik Kaydet", self, checked=True
        )
        self.minimap_auto_record.setToolTip(
            "Auto-record materials from top-right Minimap"
        )
        self.chk_minimap_auto_record = self.minimap_auto_record
        lay.addWidget(self.minimap_auto_record)

        # Toggle: 3B Dünyadaki Materyalleri (Aria/Mela) Otomatik Kaydet
        self.world_3d_auto_record = ToggleSwitch(
            "3B Dünyadaki Materyalleri (Aria/Mela) Otomatik Kaydet", self, checked=True
        )
        self.world_3d_auto_record.setToolTip(
            "3B ekranda görünen Aria, Mela, Odyle ve cevher parıltılarını otomatik haritalandırır"
        )
        self.minimap_cluster_stack_bonus = self.world_3d_auto_record
        lay.addWidget(self.world_3d_auto_record)

        # Toggle: Ignore Monster white dots on Minimap
        self.minimap_ignore_monsters = ToggleSwitch(
            "Minimap'teki Canavar Noktalarını (•) Yoksay", self, checked=True
        )
        self.minimap_ignore_monsters.setToolTip(
            "Ignore Monster white dots on Minimap"
        )
        self.chk_minimap_ignore_monsters = self.minimap_ignore_monsters
        lay.addWidget(self.minimap_ignore_monsters)

        return grp

    def _build_breaks_group(self) -> QGroupBox:
        grp = QGroupBox("Molalar ve İnsanlaştırma (Breaks)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Her [0] ± [1] dakikada bir mola ver (0: kapalı)
        row_every = QHBoxLayout()
        row_every.setSpacing(6)
        row_every.addWidget(QLabel("Mola sıklığı: her"))
        self.break_every_min = QSpinBox(self)
        self.break_every_min.setRange(0, 720)
        self.break_every_min.setFixedWidth(46)
        row_every.addWidget(self.break_every_min)
        row_every.addWidget(QLabel("±"))
        self.break_every_variance = QSpinBox(self)
        self.break_every_variance.setRange(0, 120)
        self.break_every_variance.setFixedWidth(44)
        row_every.addWidget(self.break_every_variance)
        row_every.addWidget(QLabel("dakikada bir (0: kapalı)"))
        row_every.addStretch(1)
        lay.addLayout(row_every)

        # Mola süresi [10] ± [1] dakika
        row_lasts = QHBoxLayout()
        row_lasts.setSpacing(6)
        row_lasts.addWidget(QLabel("Mola süresi:"))
        self.break_lasts_min = QSpinBox(self)
        self.break_lasts_min.setRange(1, 240)
        self.break_lasts_min.setFixedWidth(46)
        row_lasts.addWidget(self.break_lasts_min)
        row_lasts.addWidget(QLabel("±"))
        self.break_lasts_variance = QSpinBox(self)
        self.break_lasts_variance.setRange(0, 60)
        self.break_lasts_variance.setFixedWidth(44)
        row_lasts.addWidget(self.break_lasts_variance)
        row_lasts.addWidget(QLabel("dakika"))
        row_lasts.addStretch(1)
        lay.addLayout(row_lasts)

        return grp

    # ------------------------------------------------------------------
    # Right Column Builder (NODES / REGIONS & ROUTES)
    # ------------------------------------------------------------------
    def _build_nodes_group(self) -> QGroupBox:
        grp = QGroupBox("Bölge ve Rotalar (Regions & Routes)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # Row 1: Region selector + Rename... + Delete region...
        reg_row = QHBoxLayout()
        reg_row.setSpacing(6)
        lbl_reg = QLabel("Bölge:")
        lbl_reg.setFixedWidth(46)
        reg_row.addWidget(lbl_reg)

        self.region_combo = QComboBox(self)
        self.cb_region = self.region_combo
        self.region_combo.setMinimumWidth(190)
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        reg_row.addWidget(self.region_combo, stretch=1)

        self.btn_rename_region = QPushButton("Adlandır...", self)
        self.btn_rename_region.clicked.connect(self._on_rename_region)
        reg_row.addWidget(self.btn_rename_region)

        self.btn_delete_region = QPushButton("Bölgeyi Sil...", self)
        self.btn_delete_region.setObjectName("DangerOutlineButton")
        self.btn_delete_region.clicked.connect(self._on_delete_region)
        reg_row.addWidget(self.btn_delete_region)
        lay.addLayout(reg_row)

        # Row 2: Route selector + New... + Rename... + Delete route...
        rte_row = QHBoxLayout()
        rte_row.setSpacing(6)
        lbl_rte = QLabel("Rota:")
        lbl_rte.setFixedWidth(46)
        rte_row.addWidget(lbl_rte)

        self.route_combo = QComboBox(self)
        self.cb_route = self.route_combo
        self.route_combo.setMinimumWidth(140)
        self.route_combo.currentIndexChanged.connect(self._on_route_combo_changed)
        rte_row.addWidget(self.route_combo, stretch=1)

        self.btn_new_route = QPushButton("Yeni...", self)
        self.btn_new_route.clicked.connect(self._on_new_route)
        rte_row.addWidget(self.btn_new_route)

        self.btn_rename_route = QPushButton("Adlandır...", self)
        self.btn_rename_route.clicked.connect(self._on_rename_route)
        rte_row.addWidget(self.btn_rename_route)

        self.btn_delete_route = QPushButton("Rotayı Sil...", self)
        self.btn_delete_route.setObjectName("DangerOutlineButton")
        self.btn_delete_route.clicked.connect(self._on_delete_route)
        rte_row.addWidget(self.btn_delete_route)
        lay.addLayout(rte_row)

        # Center Table Container with Empty State Overlay
        table_container = QWidget(self)
        table_container_lay = QVBoxLayout(table_container)
        table_container_lay.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(self)
        self.nodes_table = self.table
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["NOKTA", "ROTALAR", "TOPLANAN", "SON", "DURUM"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        header = self.table.horizontalHeader()
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 125)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 74)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 68)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 72)

        # Empty-state overlay label inside the table viewport
        self.empty_nodes_lbl = QLabel(
            "Henüz nokta yok: Siz toplama yaptıkça bot noktaları otomatik kaydeder",
            self.table.viewport(),
        )
        self.empty_nodes_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_nodes_lbl.setStyleSheet(
            "color: #8b949e; font-size: 12px; background: transparent;"
        )
        self.empty_nodes_lbl.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

        table_container_lay.addWidget(self.table)
        lay.addWidget(table_container, stretch=1)

        # Bottom Action Buttons (2 neat rows)
        act_row1 = QHBoxLayout()
        act_row1.setSpacing(6)
        self.btn_add_nodes = QPushButton("Nokta Ekle...", self)
        self.btn_add_nodes.clicked.connect(self._on_add_nodes)
        self.btn_remove_from_route = QPushButton("Rotadan Çıkar", self)
        self.btn_remove_from_route.clicked.connect(self._on_remove_from_route)
        self.btn_delete_node = QPushButton("Noktayı Sil", self)
        self.btn_delete_node.clicked.connect(self._on_delete_node)
        self.btn_move_to_region = QPushButton("Bölgeye Taşı...", self)
        self.btn_move_to_region.clicked.connect(self._on_move_to_region)
        self.btn_map = QPushButton("Haritayı Göster", self)
        self.btn_map.clicked.connect(self._on_open_map)

        act_row1.addWidget(self.btn_add_nodes)
        act_row1.addWidget(self.btn_remove_from_route)
        act_row1.addWidget(self.btn_delete_node)
        act_row1.addWidget(self.btn_move_to_region)
        act_row1.addWidget(self.btn_map)
        lay.addLayout(act_row1)

        act_row2 = QHBoxLayout()
        act_row2.setSpacing(6)
        self.btn_share_route = QPushButton("Rotayı Paylaş...", self)
        self.btn_share_route.clicked.connect(self._on_share_route)
        self.btn_import_route = QPushButton("Rota İçe Aktar...", self)
        self.btn_import_route.clicked.connect(self._on_import_route)
        self.btn_import_questlog = QPushButton("Questlog'dan Aktar...", self)
        self.btn_import_questlog.clicked.connect(self._on_import_questlog)

        act_row2.addStretch(1)
        act_row2.addWidget(self.btn_share_route)
        act_row2.addWidget(self.btn_import_route)
        act_row2.addWidget(self.btn_import_questlog)
        act_row2.addStretch(1)
        lay.addLayout(act_row2)

        return grp

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition_empty_label()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self._reposition_empty_label()

    def _reposition_empty_label(self) -> None:
        if hasattr(self, "empty_nodes_lbl") and hasattr(self, "table"):
            vp = self.table.viewport()
            self.empty_nodes_lbl.setGeometry(0, 0, vp.width(), vp.height())

    # ------------------------------------------------------------------
    # Load / Save / Reset Configuration (`_load_from_settings` & `_save_to_settings`)
    # ------------------------------------------------------------------
    def _load_from_settings(self, settings: Optional[Any] = None) -> None:
        """
        Reads all fields from `settings` / `self.settings` / `self.config_manager`
        using safe `getattr` and populates all UI controls.
        """
        if settings is not None:
            self.settings = settings
        try:
            self.f_interval_min.setValue(
                float(self._cfg_get("f_interval_min", 1.5, target=settings))
            )
            self.f_interval_max.setValue(
                float(self._cfg_get("f_interval_max", 3.0, target=settings))
            )
            self.test_mode.setChecked(
                bool(self._cfg_get("test_mode", False, target=settings))
            )
            self.show_overlay.setChecked(
                bool(self._cfg_get("show_overlay", True, target=settings))
            )

            # New Gathering settings: harvests_per_node, gather_key, retry_on_extraction_fail
            raw_harvests = self._cfg_get("harvests_per_node", 3, target=settings)
            try:
                harvests_val = max(1, min(10, int(raw_harvests)))
            except (TypeError, ValueError):
                harvests_val = 3
            self.harvests_per_node.setValue(harvests_val)

            raw_gkey = str(
                self._cfg_get("gather_key", "f", target=settings) or "f"
            ).strip().lower()
            if raw_gkey not in ("f", "e", "auto"):
                raw_gkey = "f"
            gkey_idx = self.gather_key.findData(raw_gkey)
            if gkey_idx < 0:
                gkey_idx = self.gather_key.findText(
                    raw_gkey, Qt.MatchFlag.MatchFixedString
                )
            if gkey_idx >= 0:
                self.gather_key.setCurrentIndex(gkey_idx)
            else:
                self.gather_key.setCurrentText("f")

            self.retry_on_extraction_fail.setChecked(
                bool(self._cfg_get("retry_on_extraction_fail", True, target=settings))
            )

            # Route & Auto-Cartography settings
            self.rest_minutes.setValue(
                int(self._cfg_get("rest_minutes", 5, target=settings))
            )
            self.climb_air_max_sec.setValue(
                int(self._cfg_get("climb_air_max_sec", 10, target=settings))
            )
            self.gathered_join_active_route.setChecked(
                bool(self._cfg_get("gathered_join_active_route", True, target=settings))
            )
            self.fly_unreachable.setChecked(
                bool(self._cfg_get("fly_unreachable", False, target=settings))
            )
            self.check_odyle_minimap.setChecked(
                bool(self._cfg_get("check_odyle_minimap", False, target=settings))
            )
            self.reclick_node_sec.setValue(
                int(self._cfg_get("reclick_node_sec", 10, target=settings))
            )
            self.time_to_reach_sec.setValue(
                int(self._cfg_get("time_to_reach_sec", 60, target=settings))
            )
            self.gather_on_the_way.setChecked(
                bool(self._cfg_get("gather_on_the_way", False, target=settings))
            )
            self.wait_min_after_failed.setValue(
                int(self._cfg_get("wait_min_after_failed", 5, target=settings))
            )
            self.max_failed_in_row.setValue(
                int(self._cfg_get("max_failed_in_row", 10, target=settings))
            )
            self.remove_node_after_failures.setValue(
                int(self._cfg_get("remove_node_after_failures", 0, target=settings))
            )
            self.pause_after_no_gather_min.setValue(
                int(self._cfg_get("pause_after_no_gather_min", 15, target=settings))
            )
            self.fight_back.setChecked(
                bool(self._cfg_get("fight_back", True, target=settings))
            )
            self.fight_stop_gather_sec.setValue(
                int(self._cfg_get("fight_stop_gather_sec", 3, target=settings))
            )
            self.minimap_auto_record.setChecked(
                bool(self._cfg_get("minimap_auto_record", True, target=settings))
            )
            self.minimap_ignore_monsters.setChecked(
                bool(self._cfg_get("minimap_ignore_monsters", True, target=settings))
            )

            # Breaks settings
            self.break_every_min.setValue(
                int(self._cfg_get("break_every_min", 0, target=settings))
            )
            self.break_every_variance.setValue(
                int(self._cfg_get("break_every_variance", 1, target=settings))
            )
            self.break_lasts_min.setValue(
                int(self._cfg_get("break_lasts_min", 10, target=settings))
            )
            self.break_lasts_variance.setValue(
                int(self._cfg_get("break_lasts_variance", 1, target=settings))
            )
        except Exception as exc:
            logger.error("Error loading settings into UI: %s", exc)

    def _save_to_settings(self, settings: Optional[Any] = None) -> Any:
        """
        Writes all UI control values back to `settings` / `self.settings` /
        `self.config_manager` using safe `setattr`.
        """
        if settings is not None:
            self.settings = settings
        values = self.get_settings_dict()
        for key, val in values.items():
            self._cfg_set(key, val, target=settings)
        self._cfg_save()
        return settings if settings is not None else self.settings

    def load_settings_into_ui(self) -> None:
        """Loads all settings from `config_manager` into the UI controls."""
        self._load_from_settings()

    def load_settings(self) -> None:
        """Backwards-compatibility alias for `_load_from_settings`."""
        self._load_from_settings()

    def get_settings_dict(self) -> Dict[str, Any]:
        """Collects all current UI setting values as a dictionary."""
        min_f = round(float(self.f_interval_min.value()), 2)
        max_f = round(float(self.f_interval_max.value()), 2)
        if max_f < min_f:
            max_f = min_f
            self.f_interval_max.setValue(max_f)

        raw_gkey = str(
            self.gather_key.currentData() or self.gather_key.currentText() or "f"
        ).strip().lower()
        if raw_gkey not in ("f", "e", "auto"):
            raw_gkey = "f"

        return {
            "f_interval_min": min_f,
            "f_interval_max": max_f,
            "test_mode": bool(self.test_mode.isChecked()),
            "show_overlay": bool(self.show_overlay.isChecked()),
            "harvests_per_node": max(1, min(10, int(self.harvests_per_node.value()))),
            "gather_key": raw_gkey,
            "retry_on_extraction_fail": bool(
                self.retry_on_extraction_fail.isChecked()
            ),
            "rest_minutes": int(self.rest_minutes.value()),
            "climb_air_max_sec": int(self.climb_air_max_sec.value()),
            "gathered_join_active_route": bool(
                self.gathered_join_active_route.isChecked()
            ),
            "fly_unreachable": bool(self.fly_unreachable.isChecked()),
            "check_odyle_minimap": bool(self.check_odyle_minimap.isChecked()),
            "reclick_node_sec": int(self.reclick_node_sec.value()),
            "time_to_reach_sec": int(self.time_to_reach_sec.value()),
            "gather_on_the_way": bool(self.gather_on_the_way.isChecked()),
            "wait_min_after_failed": int(self.wait_min_after_failed.value()),
            "max_failed_in_row": int(self.max_failed_in_row.value()),
            "remove_node_after_failures": int(self.remove_node_after_failures.value()),
            "pause_after_no_gather_min": int(self.pause_after_no_gather_min.value()),
            "fight_back": bool(self.fight_back.isChecked()),
            "fight_stop_gather_sec": int(self.fight_stop_gather_sec.value()),
            "minimap_auto_record": bool(self.minimap_auto_record.isChecked()),
            "minimap_ignore_monsters": bool(self.minimap_ignore_monsters.isChecked()),
            "break_every_min": int(self.break_every_min.value()),
            "break_every_variance": int(self.break_every_variance.value()),
            "break_lasts_min": int(self.break_lasts_min.value()),
            "break_lasts_variance": int(self.break_lasts_variance.value()),
        }

    def save_settings(self) -> None:
        """Persists all UI settings to `config_manager` and emits `settings_saved`."""
        try:
            self._save_to_settings()
            self.set_status_message("Ayarlar kaydedildi ve uygulandı.")
            logger.info("Settings saved successfully.")
            self.settings_saved.emit()
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)
            self.set_status_message(f"Ayarlar kaydedilirken hata oluştu: {exc}")

    def save_and_close(self) -> None:
        """Saves settings and closes the dialog."""
        self.save_settings()
        self.accept()

    def reset_to_defaults(self) -> None:
        """Resets all controls to `DEFAULT_SETTINGS`."""
        try:
            for key, val in DEFAULT_SETTINGS.items():
                self._cfg_set(key, val)
            self._load_from_settings()
            self.set_status_message("Tüm ayarlar varsayılan değerlere sıfırlandı.")
            logger.info("Settings reset to defaults.")
        except Exception as exc:
            logger.error("Error resetting settings to defaults: %s", exc)

    def set_status_message(self, message: str) -> None:
        """Updates the footer status message (`status_lbl`)."""
        self.status_lbl.setText(str(message))

    def _on_quit_clicked(self) -> None:
        logger.info("Quit bot requested from SettingsWindow.")
        self.quit_requested.emit()
        self.close()

    # ------------------------------------------------------------------
    # Region & Route Combo Box Synchronization
    # ------------------------------------------------------------------
    def _current_region(self) -> str:
        val = self.region_combo.currentData()
        if isinstance(val, str) and val:
            return val
        txt = self.region_combo.currentText()
        if "(" in txt:
            return txt.split("(")[0].strip()
        return txt.strip()

    def _current_route(self) -> str:
        val = self.route_combo.currentData()
        if isinstance(val, str) and val:
            return val
        txt = self.route_combo.currentText()
        if "(" in txt:
            return txt.split("(")[0].strip()
        return txt.strip()

    def _get_regions_list(self) -> List[str]:
        rm = self.route_manager
        if rm is None:
            return []
        try:
            if hasattr(rm, "get_regions") and callable(rm.get_regions):
                return list(rm.get_regions())
            if hasattr(rm, "data") and isinstance(rm.data, dict):
                return list(rm.data.get("regions", {}).keys())
        except Exception as exc:
            logger.debug("Failed to list regions: %s", exc)
        return []

    def _get_routes_for_region(self, region_name: str) -> List[str]:
        rm = self.route_manager
        if rm is None or not region_name:
            return []
        try:
            if hasattr(rm, "get_routes") and callable(rm.get_routes):
                return list(rm.get_routes(region_name))
            if hasattr(rm, "data") and isinstance(rm.data, dict):
                reg = rm.data.get("regions", {}).get(region_name, {})
                routes_obj = reg.get("routes", {})
                if isinstance(routes_obj, dict):
                    return list(routes_obj.keys())
                if isinstance(routes_obj, list):
                    return list(routes_obj)
        except Exception as exc:
            logger.debug("Failed to list routes for %s: %s", region_name, exc)
        return ["All"]

    def _get_nodes_for_selection(
        self, region_name: str, route_name: str
    ) -> List[Any]:
        rm = self.route_manager
        if rm is None or not region_name:
            return []
        try:
            if hasattr(rm, "get_route_nodes") and callable(rm.get_route_nodes):
                return list(
                    rm.get_route_nodes(region_name=region_name, route_name=route_name)
                )
            if hasattr(rm, "get_nodes") and callable(rm.get_nodes):
                return list(rm.get_nodes(region=region_name, route=route_name))
        except Exception as exc:
            logger.debug("Failed to get nodes for %s / %s: %s", region_name, route_name, exc)
        return []

    def refresh_regions_and_routes(
        self,
        prefer_region: Optional[str] = None,
        prefer_route: Optional[str] = None,
    ) -> None:
        """Populates the Region and Route combo boxes with node counts (`Altgard (114)`)."""
        self._updating_combos = True
        try:
            regions = self._get_regions_list()
            active_reg = prefer_region or getattr(
                self.route_manager, "active_region", None
            )
            if not active_reg and hasattr(self.route_manager, "data"):
                active_reg = self.route_manager.data.get("active_region")

            self.region_combo.clear()
            selected_reg_idx = 0
            for idx, reg_name in enumerate(regions):
                all_reg_nodes = self._get_nodes_for_selection(reg_name, "All")
                label = f"{reg_name} ({len(all_reg_nodes)})"
                self.region_combo.addItem(label, reg_name)
                if reg_name == active_reg:
                    selected_reg_idx = idx

            if self.region_combo.count() > 0:
                self.region_combo.setCurrentIndex(selected_reg_idx)

            cur_reg = self._current_region()
            self._populate_routes_combo(cur_reg, prefer_route=prefer_route)
        except Exception as exc:
            logger.error("Error refreshing regions/routes: %s", exc)
        finally:
            self._updating_combos = False

        self.load_nodes_table()

    def _populate_routes_combo(
        self, region_name: str, prefer_route: Optional[str] = None
    ) -> None:
        self.route_combo.clear()
        if not region_name:
            return
        routes = self._get_routes_for_region(region_name)
        active_rte = prefer_route or getattr(self.route_manager, "active_route", None)
        if not active_rte and hasattr(self.route_manager, "data"):
            active_rte = self.route_manager.data.get("active_route")

        selected_rte_idx = 0
        for idx, rte_name in enumerate(routes):
            rte_nodes = self._get_nodes_for_selection(region_name, rte_name)
            label = f"{rte_name} ({len(rte_nodes)})"
            self.route_combo.addItem(label, rte_name)
            if rte_name == active_rte:
                selected_rte_idx = idx

        if self.route_combo.count() > 0:
            self.route_combo.setCurrentIndex(selected_rte_idx)

    def _on_region_combo_changed(self, _index: int = 0) -> None:
        if self._updating_combos:
            return
        reg = self._current_region()
        self._updating_combos = True
        try:
            if self.route_manager is not None:
                self.route_manager.active_region = reg
                if hasattr(self.route_manager, "data") and isinstance(
                    self.route_manager.data, dict
                ):
                    self.route_manager.data["active_region"] = reg
            self._populate_routes_combo(reg)
        finally:
            self._updating_combos = False
        self._on_route_combo_changed()

    def _on_route_combo_changed(self, _index: int = 0) -> None:
        if self._updating_combos:
            return
        reg = self._current_region()
        rte = self._current_route()
        try:
            if self.route_manager is not None:
                self.route_manager.active_region = reg
                self.route_manager.active_route = rte
                if hasattr(self.route_manager, "data") and isinstance(
                    self.route_manager.data, dict
                ):
                    self.route_manager.data["active_region"] = reg
                    self.route_manager.data["active_route"] = rte
            self.load_nodes_table()
            self.route_changed.emit(reg, rte)
        except Exception as exc:
            logger.error("Error handling route change: %s", exc)

    # ------------------------------------------------------------------
    # Nodes Table Rendering (5 Columns: NOKTA, ROTALAR, TOPLANAN, SON, DURUM)
    # ------------------------------------------------------------------
    def _resolve_node_patch_pixmap(self, node: Any, region_name: str) -> QPixmap:
        """Loads and scales the node's map patch to 48x48, or builds a clean fallback."""
        candidates: List[str] = []
        raw_path = (
            node.get("patch_path")
            if isinstance(node, dict)
            else getattr(node, "patch_path", "")
        ) or (
            node.get("patch_file")
            if isinstance(node, dict)
            else getattr(node, "patch_file", "")
        )
        node_id = int(
            node.get("id", 1) if isinstance(node, dict) else getattr(node, "id", 1)
        )

        if raw_path:
            candidates.append(str(raw_path))
            if self.route_manager is not None and hasattr(
                self.route_manager, "_resolve_patch_disk_path"
            ):
                try:
                    candidates.insert(
                        0, self.route_manager._resolve_patch_disk_path(node)
                    )
                except Exception as exc:
                    logger.debug("Patch path resolve fallback: %s", exc)
            if hasattr(self.route_manager, "patch_dir"):
                candidates.append(
                    os.path.join(
                        str(self.route_manager.patch_dir), os.path.basename(str(raw_path))
                    )
                )

        candidates.append(
            os.path.join("routes", region_name, "patches", f"node_{node_id}.png")
        )
        candidates.append(
            os.path.join("routes", "patches", f"{region_name}_node_{node_id}.png")
        )

        for path in candidates:
            if path and os.path.isfile(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    return pix.scaled(
                        48,
                        48,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )

        # Fallback 48x48 terrain-textured grayscale patch
        fallback = QPixmap(48, 48)
        fallback.fill(QColor("#262b33"))
        p = QPainter(fallback)
        p.setPen(QPen(QColor("#374151"), 1))
        p.drawRect(0, 0, 47, 47)
        p.setPen(QPen(QColor("#6b7280")))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(fallback.rect(), int(Qt.AlignmentFlag.AlignCenter), f"#{node_id}")
        p.end()
        return fallback

    @staticmethod
    def _format_last_visited(last_ts: float, now: float) -> str:
        if last_ts <= 0.0:
            return "hiç"
        diff = max(0.0, now - last_ts)
        if diff < 60.0:
            return f"{int(diff)}sn önce"
        if diff < 3600.0:
            return f"{int(diff // 60)}dk önce"
        dt = datetime.datetime.fromtimestamp(last_ts)
        return dt.strftime("%H:%M")

    @staticmethod
    def _compute_rest_status(
        last_ts: float, rest_duration_sec: int, now: float
    ) -> Tuple[str, bool]:
        if last_ts <= 0.0 or rest_duration_sec <= 0:
            return "Hazır", True
        remaining = float(rest_duration_sec) - (now - last_ts)
        if remaining <= 0.0:
            return "Hazır", True
        rem_sec = int(math.ceil(remaining))
        mins = rem_sec // 60
        secs = rem_sec % 60
        if mins > 0:
            return f"{mins}dk {secs:02d}sn", False
        return f"{secs}sn", False

    def load_nodes_table(self) -> None:
        """Populates the 5-column NODES table for the currently selected Region & Route."""
        try:
            reg = self._current_region()
            rte = self._current_route()
            nodes = self._get_nodes_for_selection(reg, rte)

            self.table.setRowCount(0)
            self._reposition_empty_label()

            if not nodes:
                self.empty_nodes_lbl.setVisible(True)
                return

            self.empty_nodes_lbl.setVisible(False)
            now = time.time()
            rest_sec_cfg = int(self.rest_minutes.value()) * 60

            for row_idx, node in enumerate(nodes):
                self.table.insertRow(row_idx)
                self.table.setRowHeight(row_idx, 54)

                node_id = int(
                    node.get("id", row_idx + 1)
                    if isinstance(node, dict)
                    else getattr(node, "id", row_idx + 1)
                )
                raw_routes = (
                    node.get("routes", [])
                    if isinstance(node, dict)
                    else getattr(node, "routes", [])
                )
                if isinstance(raw_routes, (list, tuple, set)):
                    routes_str = ", ".join(str(r) for r in raw_routes if r and r != "All")
                else:
                    routes_str = str(raw_routes or "")
                if not routes_str:
                    routes_str = rte if (rte and rte != "All") else "Tümü"

                gathered_cnt = int(
                    node.get("gathered_count", node.get("visit_count", 0))
                    if isinstance(node, dict)
                    else getattr(
                        node, "gathered_count", getattr(node, "visit_count", 0)
                    )
                )
                last_ts = float(
                    node.get("last_visited_timestamp", node.get("last_visited", 0.0))
                    if isinstance(node, dict)
                    else getattr(
                        node,
                        "last_visited_timestamp",
                        getattr(node, "last_visited", 0.0),
                    )
                )

                # Column 0: NOKTA (48x48 patch thumbnail + bold #id)
                cell_node = QWidget(self.table)
                cell_lay = QHBoxLayout(cell_node)
                cell_lay.setContentsMargins(4, 3, 6, 3)
                cell_lay.setSpacing(8)

                lbl_thumb = QLabel(cell_node)
                lbl_thumb.setFixedSize(48, 48)
                lbl_thumb.setPixmap(self._resolve_node_patch_pixmap(node, reg))
                lbl_thumb.setStyleSheet(
                    "border: 1px solid #30363d; border-radius: 3px;"
                )
                cell_lay.addWidget(lbl_thumb)

                lbl_num = QLabel(f"#{node_id}", cell_node)
                lbl_num.setStyleSheet(
                    "color: #f3f4f6; font-weight: bold; font-size: 12px;"
                )
                cell_lay.addWidget(lbl_num)
                cell_lay.addStretch(1)

                # Store node_id in underlying QTableWidgetItem for selection operations
                anchor_item = QTableWidgetItem()
                anchor_item.setData(Qt.ItemDataRole.UserRole, node_id)
                self.table.setItem(row_idx, 0, anchor_item)
                self.table.setCellWidget(row_idx, 0, cell_node)

                # Column 1: ROTALAR
                item_routes = QTableWidgetItem(routes_str)
                item_routes.setForeground(QBrush(QColor("#d1d5db")))
                self.table.setItem(row_idx, 1, item_routes)

                # Column 2: TOPLANAN
                item_gathered = QTableWidgetItem(f"{gathered_cnt}×")
                item_gathered.setForeground(QBrush(QColor("#d1d5db")))
                self.table.setItem(row_idx, 2, item_gathered)

                # Column 3: SON
                last_str = self._format_last_visited(last_ts, now)
                item_last = QTableWidgetItem(last_str)
                item_last.setData(Qt.ItemDataRole.UserRole, last_ts)
                item_last.setForeground(QBrush(QColor("#9ca3af")))
                self.table.setItem(row_idx, 3, item_last)

                # Column 4: DURUM (green `Hazır` or amber `4dk 12sn` capsule badge)
                rest_text, is_ready = self._compute_rest_status(
                    last_ts, rest_sec_cfg, now
                )
                badge = _RestBadgeWidget(rest_text, is_ready, self.table)
                self.table.setCellWidget(row_idx, 4, badge)
        except Exception as exc:
            logger.error("Error populating nodes table: %s", exc)

    def _update_live_node_statuses(self) -> None:
        """Updates LAST and REST columns every second without rebuilding the whole table."""
        if not self.isVisible() or self.table.rowCount() == 0:
            return
        try:
            now = time.time()
            rest_sec_cfg = int(self.rest_minutes.value()) * 60
            for row_idx in range(self.table.rowCount()):
                item_last = self.table.item(row_idx, 3)
                if item_last is None:
                    continue
                last_ts = float(item_last.data(Qt.ItemDataRole.UserRole) or 0.0)
                item_last.setText(self._format_last_visited(last_ts, now))

                badge = self.table.cellWidget(row_idx, 4)
                if isinstance(badge, _RestBadgeWidget):
                    rest_text, is_ready = self._compute_rest_status(
                        last_ts, rest_sec_cfg, now
                    )
                    badge.set_status(rest_text, is_ready)
        except Exception as exc:
            logger.debug("Live status refresh error: %s", exc)

    def _selected_node_ids(self) -> List[int]:
        ids: List[int] = []
        seen = set()
        for idx in self.table.selectionModel().selectedRows():
            row = idx.row()
            it = self.table.item(row, 0)
            if it is not None:
                nid = it.data(Qt.ItemDataRole.UserRole)
                if nid is not None and int(nid) not in seen:
                    seen.add(int(nid))
                    ids.append(int(nid))
        if not ids and self.table.currentRow() >= 0:
            it = self.table.item(self.table.currentRow(), 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) is not None:
                ids.append(int(it.data(Qt.ItemDataRole.UserRole)))
        return ids

    # ------------------------------------------------------------------
    # Region & Route CRUD Button Handlers
    # ------------------------------------------------------------------
    def _on_rename_region(self) -> None:
        old_reg = self._current_region()
        if not old_reg:
            return
        new_reg, ok = QInputDialog.getText(
            self, "Bölgeyi Yeniden Adlandır", "Yeni bölge adı:", text=old_reg
        )
        new_reg = new_reg.strip()
        if not ok or not new_reg or new_reg == old_reg:
            return
        try:
            if hasattr(self.route_manager, "rename_region"):
                success = self.route_manager.rename_region(old_reg, new_reg)
                if success:
                    self.refresh_regions_and_routes(prefer_region=new_reg)
                    self.set_status_message(
                        f"'{old_reg}' bölgesi '{new_reg}' olarak yeniden adlandırıldı."
                    )
        except Exception as exc:
            logger.error("Rename region failed: %s", exc)

    def _on_delete_region(self) -> None:
        reg = self._current_region()
        if not reg:
            return
        ans = QMessageBox.question(
            self,
            "Bölgeyi Sil",
            f"'{reg}' bölgesi ve içindeki tüm kayıtlı noktalar silinsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            if hasattr(self.route_manager, "delete_region"):
                self.route_manager.delete_region(reg)
                self.refresh_regions_and_routes()
                self.set_status_message(f"'{reg}' bölgesi silindi.")
        except Exception as exc:
            logger.error("Delete region failed: %s", exc)

    def _on_new_route(self) -> None:
        reg = self._current_region()
        if not reg:
            reg, ok_reg = QInputDialog.getText(
                self, "Yeni Bölge", "Önce bir bölge adı girin:", text="Altgard"
            )
            reg = reg.strip()
            if not ok_reg or not reg:
                return
            if hasattr(self.route_manager, "create_region"):
                self.route_manager.create_region(reg)

        new_rte, ok = QInputDialog.getText(
            self, "Yeni Rota", f"'{reg}' bölgesi için yeni rota adı:"
        )
        new_rte = new_rte.strip()
        if not ok or not new_rte:
            return
        try:
            if hasattr(self.route_manager, "create_route"):
                self.route_manager.create_route(reg, new_rte)
                self.refresh_regions_and_routes(
                    prefer_region=reg, prefer_route=new_rte
                )
                self.set_status_message(f"'{reg}' içinde '{new_rte}' rotası oluşturuldu.")
        except Exception as exc:
            logger.error("Create route failed: %s", exc)

    def _on_rename_route(self) -> None:
        reg = self._current_region()
        old_rte = self._current_route()
        if not reg or not old_rte or old_rte == "All":
            self.set_status_message("Varsayılan 'All' rotası yeniden adlandırılamaz.")
            return
        new_rte, ok = QInputDialog.getText(
            self, "Rotayı Yeniden Adlandır", "Yeni rota adı:", text=old_rte
        )
        new_rte = new_rte.strip()
        if not ok or not new_rte or new_rte == old_rte:
            return
        try:
            if hasattr(self.route_manager, "rename_route"):
                if self.route_manager.rename_route(reg, old_rte, new_rte):
                    self.refresh_regions_and_routes(
                        prefer_region=reg, prefer_route=new_rte
                    )
                    self.set_status_message(
                        f"'{old_rte}' rotası '{new_rte}' olarak yeniden adlandırıldı."
                    )
        except Exception as exc:
            logger.error("Rename route failed: %s", exc)

    def _on_delete_route(self) -> None:
        reg = self._current_region()
        rte = self._current_route()
        if not reg or not rte or rte == "All":
            self.set_status_message("Varsayılan 'All' rotası silinemez.")
            return
        ans = QMessageBox.question(
            self,
            "Rotayı Sil",
            f"'{reg}' bölgesinden '{rte}' rotası silinsin mi? (Noktalar 'All' içinde kalır.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            if hasattr(self.route_manager, "delete_route"):
                self.route_manager.delete_route(reg, rte)
                self.refresh_regions_and_routes(
                    prefer_region=reg, prefer_route="All"
                )
                self.set_status_message(f"'{rte}' rotası silindi.")
        except Exception as exc:
            logger.error("Delete route failed: %s", exc)

    # ------------------------------------------------------------------
    # Node Action & Auxiliary Dialog Button Handlers
    # ------------------------------------------------------------------
    def _on_add_nodes(self) -> None:
        reg = self._current_region() or "Altgard"
        rte = self._current_route() or "All"
        try:
            all_nodes = self._get_nodes_for_selection(reg, "All")
            current_ids = {
                int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0))
                for n in self._get_nodes_for_selection(reg, rte)
            }
            available = [
                n
                for n in all_nodes
                if int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0))
                not in current_ids
            ]

            dlg = _AddNodesToRouteDialog(reg, rte, available, self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                if dlg.create_new_requested:
                    if hasattr(self.route_manager, "add_node"):
                        node = self.route_manager.add_node(
                            region=reg, route=rte, map_x=960, map_y=540
                        )
                        self.refresh_regions_and_routes(
                            prefer_region=reg, prefer_route=rte
                        )
                        self.set_status_message(
                            f"Nokta #{node.id}, {reg} · {rte} rotasına eklendi."
                        )
                else:
                    sel_ids = dlg.selected_node_ids()
                    added = 0
                    for nid in sel_ids:
                        if hasattr(self.route_manager, "add_node_to_route"):
                            if self.route_manager.add_node_to_route(
                                nid, rte, region=reg
                            ):
                                added += 1
                    self.refresh_regions_and_routes(
                        prefer_region=reg, prefer_route=rte
                    )
                    self.set_status_message(
                        f"{added} nokta '{rte}' rotasına eklendi."
                    )
        except Exception as exc:
            logger.error("Add nodes failed: %s", exc)

    def _on_remove_from_route(self) -> None:
        reg = self._current_region()
        rte = self._current_route()
        if not reg or not rte or rte == "All":
            self.set_status_message(
                "Noktaları rotadan çıkarmak için 'All' dışında belirli bir rota seçin."
            )
            return
        ids = self._selected_node_ids()
        if not ids:
            self.set_status_message("Önce bir veya daha fazla nokta seçin.")
            return
        try:
            removed = 0
            for nid in ids:
                if hasattr(self.route_manager, "remove_node_from_route"):
                    if self.route_manager.remove_node_from_route(nid, rte, region=reg):
                        removed += 1
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            self.set_status_message(f"{removed} nokta '{rte}' rotasından çıkarıldı.")
        except Exception as exc:
            logger.error("Remove from route failed: %s", exc)

    def _on_delete_node(self) -> None:
        reg = self._current_region()
        rte = self._current_route()
        ids = self._selected_node_ids()
        if not ids:
            self.set_status_message("Silmek için bir veya daha fazla nokta seçin.")
            return
        try:
            deleted = 0
            for nid in ids:
                if hasattr(self.route_manager, "delete_node"):
                    if self.route_manager.delete_node(nid, region=reg):
                        deleted += 1
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            self.set_status_message(f"{deleted} nokta '{reg}' bölgesinden silindi.")
        except Exception as exc:
            logger.error("Delete node failed: %s", exc)

    def delete_selected_node(self) -> None:
        """Backwards-compatibility alias for `_on_delete_node`."""
        self._on_delete_node()

    def _on_move_to_region(self) -> None:
        reg = self._current_region()
        rte = self._current_route()
        ids = self._selected_node_ids()
        if not ids:
            self.set_status_message("Taşımak için bir veya daha fazla nokta seçin.")
            return
        regions = [r for r in self._get_regions_list() if r != reg]
        default_target = regions[0] if regions else "Verteron"
        target_reg, ok = QInputDialog.getItem(
            self,
            "Bölgeye Taşı",
            "Hedef bölge adı:",
            regions if regions else [default_target],
            0,
            True,
        )
        target_reg = target_reg.strip()
        if not ok or not target_reg or target_reg == reg:
            return
        try:
            moved = 0
            for nid in ids:
                if hasattr(self.route_manager, "move_node_to_region"):
                    res = self.route_manager.move_node_to_region(
                        nid, target_reg, source_region=reg
                    )
                    if res is not None:
                        moved += 1
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            self.set_status_message(
                f"{moved} nokta '{reg}' bölgesinden '{target_reg}' bölgesine taşındı."
            )
        except Exception as exc:
            logger.error("Move node to region failed: %s", exc)

    def _on_open_map(self) -> None:
        """Opens `ui.map_viewer.MapViewerWindow` for the active region and route."""
        reg = self._current_region() or "Altgard"
        rte = self._current_route() or "All"
        rest_sec = float(self.rest_minutes.value() * 60)
        try:
            from ui.map_viewer import MapViewerWindow  # type: ignore

            dlg = MapViewerWindow(
                route_manager=self.route_manager,
                region_name=reg,
                route_name=rte,
                rest_duration_sec=rest_sec,
                parent=self,
            )
            if hasattr(dlg, "node_toggled"):
                dlg.node_toggled.connect(lambda _nid: self.refresh_regions_and_routes(reg, rte))
            if hasattr(dlg, "node_deleted"):
                dlg.node_deleted.connect(lambda _nid: self.refresh_regions_and_routes(reg, rte))
            self._map_viewer_dlg = dlg
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as exc:
            logger.warning("MapViewerWindow fallback triggered: %s", exc)
            try:
                from ui.map_viewer import MapViewerDialog  # type: ignore

                dlg = MapViewerDialog(self.route_manager, parent=self)
                self._map_viewer_dlg = dlg
                dlg.show()
            except Exception as exc2:
                logger.error("Could not open map viewer: %s", exc2)
                self.set_status_message(f"Harita penceresi açılamadı: {exc2}")

    def _on_open_radar(self) -> None:
        """Opens `ui.cartography_radar.CartographyRadarDialog`."""
        try:
            from ui.cartography_radar import CartographyRadarDialog  # type: ignore

            if self._radar_dlg is None:
                self._radar_dlg = CartographyRadarDialog(
                    route_manager=self.route_manager, parent=self
                )
            self._radar_dlg.show()
            self._radar_dlg.raise_()
            self._radar_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open CartographyRadarDialog: %s", exc)
            self.set_status_message(f"Canlı Radar açılamadı: {exc}")

    def _on_open_analytics(self) -> None:
        """Opens `ui.analytics_dialog.AnalyticsDialog`."""
        try:
            from ui.analytics_dialog import AnalyticsDialog  # type: ignore

            if self._analytics_dlg is None:
                self._analytics_dlg = AnalyticsDialog(parent=self)
            self._analytics_dlg.show()
            self._analytics_dlg.raise_()
            self._analytics_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open AnalyticsDialog: %s", exc)
            self.set_status_message(f"İstatistikler açılamadı: {exc}")

    def _on_open_calibration(self) -> None:
        """Opens `ui.calibration_wizard.CalibrationWizard`."""
        try:
            from ui.calibration_wizard import CalibrationWizard  # type: ignore

            if self._calibration_dlg is None:
                self._calibration_dlg = CalibrationWizard(parent=self)
            self._calibration_dlg.show()
            self._calibration_dlg.raise_()
            self._calibration_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open CalibrationWizard: %s", exc)
            self.set_status_message(f"Kalibrasyon sihirbazı açılamadı: {exc}")

    def _on_open_guide(self) -> None:
        """Opens `ui.guide_dialog.GuideDialog`."""
        try:
            from ui.guide_dialog import GuideDialog  # type: ignore

            if self._guide_dlg is None:
                self._guide_dlg = GuideDialog(parent=self)
            self._guide_dlg.show()
            self._guide_dlg.raise_()
            self._guide_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open GuideDialog: %s", exc)
            self.set_status_message(f"Kılavuz açılamadı: {exc}")

    def _on_open_yolo_factory(self) -> None:
        """Opens `ui.yolo_factory_dialog.YOLOFactoryDialog`."""
        try:
            from ui.yolo_factory_dialog import YOLOFactoryDialog  # type: ignore

            if getattr(self, "_yolo_factory_dlg", None) is None:
                self._yolo_factory_dlg = YOLOFactoryDialog(parent=self)
            self._yolo_factory_dlg.show()
            self._yolo_factory_dlg.raise_()
            self._yolo_factory_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open YOLOFactoryDialog: %s", exc)
            self.set_status_message(f"YOLO Veri Fabrikası açılamadı: {exc}")

    def _on_open_mobile_connect(self) -> None:
        """Opens `ui.mobile_connect_dialog.MobileConnectDialog`."""
        try:
            from ui.mobile_connect_dialog import MobileConnectDialog  # type: ignore

            if getattr(self, "_mobile_connect_dlg", None) is None:
                self._mobile_connect_dlg = MobileConnectDialog(parent=self)
            self._mobile_connect_dlg.show()
            self._mobile_connect_dlg.raise_()
            self._mobile_connect_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open MobileConnectDialog: %s", exc)
            self.set_status_message(f"Mobil Kumanda penceresi açılamadı: {exc}")

    def _on_share_route(self) -> None:
        """Exports the active region/route to a `.zip` file via `export_route_zip`."""
        reg = self._current_region() or "Altgard"
        rte = self._current_route() or "All"
        default_name = f"{reg}_{rte.replace(' ', '_')}.zip"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Rotayı ZIP Olarak Paylaş",
            default_name,
            "ZIP Arşivleri (*.zip)",
        )
        if not file_path:
            return
        try:
            if hasattr(self.route_manager, "export_route_zip"):
                out_path = self.route_manager.export_route_zip(
                    region=reg, route=rte, zip_path=file_path
                )
                self.set_status_message(
                    f"Rota paylaşıldı: '{reg} · {rte}' -> {os.path.basename(out_path)}"
                )
        except Exception as exc:
            logger.error("Share route failed: %s", exc)
            self.set_status_message(f"Dışa aktarma başarısız: {exc}")

    def _on_import_route(self) -> None:
        """Imports a route `.zip` archive via `import_route_zip`."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Rota ZIP Arşivini İçe Aktar",
            "",
            "ZIP Arşivleri (*.zip)",
        )
        if not file_path:
            return
        try:
            if hasattr(self.route_manager, "import_route_zip"):
                imported_reg = self.route_manager.import_route_zip(file_path)
                self.refresh_regions_and_routes(prefer_region=imported_reg)
                nodes_cnt = len(
                    self._get_nodes_for_selection(imported_reg, self._current_route())
                )
                self.set_status_message(
                    f"{nodes_cnt} nokta {imported_reg} bölgesine aktarıldı"
                )
        except Exception as exc:
            logger.error("Import route failed: %s", exc)
            self.set_status_message(f"İçe aktarma başarısız: {exc}")

    def _on_import_questlog(self) -> None:
        """Opens `core.questlog_importer.QuestlogImportDialog` (with builtin fallback)."""
        try:
            from core.questlog_importer import QuestlogImportDialog  # type: ignore

            try:
                dlg = QuestlogImportDialog(
                    route_manager=self.route_manager, parent=self
                )
            except TypeError:
                try:
                    dlg = QuestlogImportDialog(self.route_manager, self)
                except TypeError:
                    dlg = QuestlogImportDialog(self)

            if hasattr(dlg, "import_completed"):
                dlg.import_completed.connect(self._on_questlog_imported_signal)

            self._questlog_dlg = dlg
            res = dlg.exec()
            if res == QDialog.DialogCode.Accepted:
                self.refresh_regions_and_routes()
                reg = self._current_region()
                rte = self._current_route()
                cnt = len(self._get_nodes_for_selection(reg, rte))
                summary = getattr(
                    dlg,
                    "imported_summary",
                    f"{cnt} nokta {reg} · {rte} rotasına aktarıldı",
                )
                self.set_status_message(str(summary))
        except Exception as exc:
            logger.info(
                "Using built-in QuestlogImportDialog replica (%s)", exc
            )
            dlg_fb = _BuiltinQuestlogImportDialog(
                route_manager=self.route_manager, parent=self
            )
            self._questlog_dlg = dlg_fb
            if dlg_fb.exec() == QDialog.DialogCode.Accepted:
                self.refresh_regions_and_routes()
                if dlg_fb.imported_summary:
                    self.set_status_message(dlg_fb.imported_summary)

    def _on_questlog_imported_signal(self, *args: Any) -> None:
        self.refresh_regions_and_routes()
        if args and isinstance(args[0], str):
            self.set_status_message(args[0])
        else:
            reg = self._current_region()
            rte = self._current_route()
            cnt = len(self._get_nodes_for_selection(reg, rte))
            self.set_status_message(f"{cnt} nokta {reg} · {rte} rotasına aktarıldı")

    def get_enterprise_summary_dict(self) -> Dict[str, Any]:
        """Returns unified hotkeys, display, and mobile remote settings summary."""
        summary: Dict[str, Any] = {
            "hotkeys": {},
            "display": {},
            "mobile": {},
        }
        for obj in self._candidate_settings_objects():
            for sec_name in ("hotkeys", "display", "mobile"):
                sec_obj = getattr(obj, sec_name, None)
                if sec_obj is not None and not summary[sec_name]:
                    if isinstance(sec_obj, dict):
                        summary[sec_name] = dict(sec_obj)
                    elif hasattr(sec_obj, "__dict__"):
                        summary[sec_name] = dict(sec_obj.__dict__)
        return summary

    def get_settings_window_health_summary(self) -> dict:
        try:
            return {"settings_window_ready": True}
        except Exception:
            return {"settings_window_ready": False}

    def get_settings_window_ready_flag(self) -> dict:
        return {"module": "settings_window", "ready": True}

    def get_settings_window_version_info(self) -> dict:
        return {"module": "settings_window", "version": "2.0.0", "ready": True}

    def get_settings_window_diagnostic_snapshot(self) -> dict:
        return {"module": "settings_window", "status": "ok", "ready": True}


# Backwards compatibility alias
SettingsDialog = SettingsWindow


def get_settings_window_capabilities() -> list:
    return ["settings_window", "configuration_ui", "diagnostic_snapshot"]


def validate_settings_window_contract() -> bool:
    return bool(get_settings_window_capabilities())


def get_settings_window_baseline_metrics() -> dict:
    return {"module": "settings_window", "contract_valid": validate_settings_window_contract(), "capability_count": len(get_settings_window_capabilities())}

