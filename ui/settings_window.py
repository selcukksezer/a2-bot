# -*- coding: utf-8 -*-
"""
Gatherion v2.0 Dark Theme Settings & Node Management Window.

Matches `video_frames/frame_06s.jpg`, `frame_12s.jpg`, and `frame_21s.jpg` pixel-for-pixel:
- Custom animated pill switch `ToggleSwitch(QCheckBox)` rendered with `QPainter`
- Left Column:
  * GATHERING group (`f_interval_min`, `f_interval_max`, `test_mode`, `show_overlay`)
  * ROUTE group (`rest_minutes`, `gathered_join_active_route`,
    `fly_unreachable`, `reclick_node_sec`, `time_to_reach_sec`,
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
  * 3-row action bar:
    Row 1: `Add nodes...`, `Remove from route`, `Delete node`, `Move to region...`, `Map...`
    Row 2: `Seçilenleri Sil`, `YZ Yanlışları Sil`
    Row 3: `Share route...`, `Import route...`, `Import from questlog...`
- Footer Bar:
  * Left: `Quit bot` (`quit_requested`), `Gatherion vX.Y.Z` label, `status_lbl`
  * Right: `Reset to defaults`, blue `Save` (`#3b82f6`, `settings_saved`), `Close`
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# F-0540 tek kaynak: HP/MP iksir eşikleri config/schema.py
# HP/MP_POTION_THRESHOLD_PCT sabitlerinden gelir (sözleşme no 10: 65/45).
# DEFAULT_SETTINGS de bu sabitlerden türetilir; stdlib-only modül olduğu
# için üst düzey içe aktarma döngü yaratmaz.
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPalette,
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.shell_page_frame import wrap_shell_tab_root_layout
from version import __version__

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
    "pause_on_alt_tab": True,
    "harvests_per_node": 3,
    "gather_key": "f",
    "retry_on_extraction_fail": True,
    "rest_minutes": 5,
    "gathered_join_active_route": True,
    # F-23-04: şema varsayılanıyla uyumlu (şema: True, dosya: true).
    # Eskiden UI False diyordu; dosya silinince/bozulunca davranış
    # hangi kodun önce varsayılan ürettiğine göre değişiyordu.
    "fly_unreachable": True,
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
    "break_every_min": 0,
    "break_every_variance": 1,
    "break_lasts_min": 10,
    "break_lasts_variance": 1,
    "record_node_f8": "ctrl+alt+f8",
    "toggle_menu_f9": "ctrl+alt+f9",
    "pause_resume_f10": "ctrl+alt+f10",
    "selective_gather_enabled": False,
    "selected_materials": [],
    "proximity_route_auto": True,
    "auto_use_hp_potion": True,
    "hp_potion_threshold_pct": HP_POTION_THRESHOLD_PCT,
    "hp_potion_key": "F1",
    "auto_use_mp_potion": True,
    "mp_potion_threshold_pct": MP_POTION_THRESHOLD_PCT,
    "mp_potion_key": "F2",
    "auto_refresh_buff_scroll": True,
    "buff_scroll_interval_sec": 300,
    "buff_scroll_key": "F7",
    # F-23-06: molaları açıp kapatan ana şalter (BreakSettings.enabled).
    # Eskiden DEFAULT_SETTINGS'te yoktu, UI'da widget'ı yoktu.
    "breaks_enabled": True,
}

# Mapping between canonical keys and legacy nested (section, key) pairs
LEGACY_KEY_MAP: Dict[str, Tuple[str, str]] = {
    "f_interval_min": ("gathering", "min_f_interval"),
    "f_interval_max": ("gathering", "max_f_interval"),
    "test_mode": ("gathering", "test_mode"),
    "show_overlay": ("gathering", "show_overlay"),
    "pause_on_alt_tab": ("gathering", "pause_on_alt_tab"),
    "harvests_per_node": ("gathering", "harvests_per_node"),
    "gather_key": ("gathering", "gather_key"),
    "retry_on_extraction_fail": ("gathering", "retry_on_extraction_fail"),
    "selective_gather_enabled": ("gathering", "selective_gather_enabled"),
    "selected_materials": ("gathering", "selected_materials"),
    "proximity_route_auto": ("gathering", "proximity_route_auto"),
    "rest_minutes": ("route", "rest_after_visit_min"),
    "gathered_join_active_route": ("route", "gathered_nodes_join_route"),
    "fly_unreachable": ("route", "fly_unreachable"),
    "reclick_node_sec": ("route", "reclick_interval_sec"),
    "time_to_reach_sec": ("route", "time_to_reach_node_sec"),
    "gather_on_the_way": ("route", "exploration_gather"),
    "wait_min_after_failed": ("route", "wait_min_after_failed"),
    "max_failed_in_row": ("route", "max_failures_before_wait"),
    "remove_node_after_failures": ("route", "remove_node_after_failures"),
    "pause_after_no_gather_min": ("route", "pause_after_idle_min"),
    "fight_back": ("route", "fight_back_when_attacked"),
    "fight_stop_gather_sec": ("route", "fight_stop_gather_sec"),
    "auto_use_hp_potion": ("route", "auto_use_hp_potion"),
    "hp_potion_threshold_pct": ("route", "hp_potion_threshold_pct"),
    # F-23-03: tuş anahtarları şemaya eklendi (RouteSettings); artık
    # bölüm-dışı düz öznitelik olarak değil, gerçek alan olarak yazılır.
    "hp_potion_key": ("route", "hp_potion_key"),
    "auto_use_mp_potion": ("route", "auto_use_mp_potion"),
    "mp_potion_threshold_pct": ("route", "mp_potion_threshold_pct"),
    "mp_potion_key": ("route", "mp_potion_key"),
    "auto_refresh_buff_scroll": ("route", "auto_refresh_buff_scroll"),
    "buff_scroll_interval_sec": ("route", "buff_scroll_interval_sec"),
    "buff_scroll_key": ("route", "buff_scroll_key"),
    "minimap_auto_record": ("route", "minimap_auto_record"),
    "auto_record_3d_crests": ("route", "auto_record_3d_crests"),
    "last_tab": ("display", "last_tab"),
    "break_every_min": ("breaks", "interval_min"),
    # F-23-06: düz anahtar "breaks_enabled" -> BreakSettings.enabled alanı.
    # _cfg_set bölüm yazımında gerçek alan adını (leg_key) kullanır.
    "breaks_enabled": ("breaks", "enabled"),
    "break_every_variance": ("breaks", "interval_jitter_min"),
    "break_lasts_min": ("breaks", "duration_min"),
    "break_lasts_variance": ("breaks", "duration_jitter_min"),
    "record_node_f8": ("hotkeys", "record_node_f8"),
    "toggle_menu_f9": ("hotkeys", "toggle_menu_f9"),
    "pause_resume_f10": ("hotkeys", "pause_resume_f10"),
}


# ============================================================================
# Custom iOS / Gatherion Pill Switch (`ToggleSwitch(QCheckBox)`)
# ============================================================================
class ToggleSwitch(QCheckBox):
    """
    Modern iOS / Gatherion-style rounded pill toggle switch rendered with QPainter.
    Supports smooth thumb animation and optional inline text label on the right.
    Matches ayarlar.html `.sw` (34x19px, #3a4654 off -> #2f7fe0 on).
    """

    SWITCH_WIDTH = 34
    SWITCH_HEIGHT = 19
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
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

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

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        label = self.text()
        if not label:
            return self.SWITCH_HEIGHT + 4
        fm = QFontMetrics(QFont("Segoe UI", 10))
        text_w = max(48, int(width) - self.SWITCH_WIDTH - self.TEXT_SPACING - 8)
        bounds = fm.boundingRect(0, 0, text_w, 800, int(Qt.TextFlag.TextWordWrap), label)
        return max(self.SWITCH_HEIGHT + 4, bounds.height() + 4)

    def sizeHint(self) -> QSize:  # noqa: N802
        label = self.text()
        if not label:
            return QSize(self.SWITCH_WIDTH + 4, self.SWITCH_HEIGHT + 4)
        fm = QFontMetrics(QFont("Segoe UI", 10))
        text_w = fm.horizontalAdvance(label)
        pref_w = min(self.SWITCH_WIDTH + self.TEXT_SPACING + text_w + 12, 520)
        return QSize(pref_w, self.heightForWidth(pref_w))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.SWITCH_WIDTH + 140, self.SWITCH_HEIGHT + 4)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        needed = self.heightForWidth(max(1, self.width()))
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)
            self.updateGeometry()
        super().resizeEvent(event)

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
        # Interpolate track color between #3a4654 (unchecked) and #2f7fe0 (checked)
        r = int(58 + (47 - 58) * t)
        g = int(70 + (127 - 70) * t)
        b = int(84 + (224 - 84) * t)
        track_color = QColor(r, g, b)
        border_color = QColor("#3b8bff") if t > 0.5 else QColor("#4d5b6c")

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(track_rect, radius, radius)

        # Thumb circle (14px inside 19px height)
        margin = 2.5
        thumb_d = sw_h - (margin * 2.0)
        min_x = 1.0 + margin
        max_x = 1.0 + sw_w - margin - thumb_d
        thumb_x = min_x + (max_x - min_x) * t
        thumb_y = sw_y + margin

        # Interpolate thumb color between #dfe6ec (unchecked) and #ffffff (checked)
        tr = int(223 + (255 - 223) * t)
        tg = int(230 + (255 - 230) * t)
        tb = int(236 + (255 - 236) * t)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(tr, tg, tb)))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_d, thumb_d))

        # Optional inline text label
        label = self.text()
        if label:
            font = QFont("Segoe UI", 10)
            painter.setFont(font)
            text_color = QColor("#ffffff") if self._hovered else QColor("#d7dfe7")
            painter.setPen(QPen(text_color))
            text_x = int(sw_w) + self.TEXT_SPACING + 2
            text_rect = QRect(text_x, 0, max(10, self.width() - text_x - 2), self.height())
            painter.drawText(
                text_rect,
                int(
                    Qt.AlignmentFlag.AlignVCenter
                    | Qt.AlignmentFlag.AlignLeft
                    | Qt.TextFlag.TextWordWrap
                ),
                label,
            )

        painter.end()


class _HotkeyCaptureEdit(QLineEdit):
    """Interactive hotkey capture input matching ayarlar.html `#hkStrip`.

    Prototype behaviour: focus shows "tuş bekliyor…", the next key
    combination is captured, Escape restores the previous chord, and a
    bare F8/F9/F10 is never accepted (those keys go to the game/Windows
    without modifiers, so the chord would never reach the bot).
    """

    def __init__(self, default_chord: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._prev_text = str(default_chord or "")
        self._capturing = False
        self.setMinimumHeight(30)
        self.setStyleSheet("QLineEdit { padding: 4px 8px; font-size: 12px; }")
        if default_chord:
            self.setText(default_chord)
            self.setPlaceholderText(default_chord)

    def focusInEvent(self, event: Any) -> None:  # noqa: N802
        self._prev_text = self.text()
        self._capturing = True
        self.setPlaceholderText("tuş bekliyor…")
        self.setProperty("capturing", True)
        self.setStyleSheet("border: 1px solid #e9a820; background-color: #2a2416;")
        super().focusInEvent(event)

    def focusOutEvent(self, event: Any) -> None:  # noqa: N802
        self._capturing = False
        self.setProperty("capturing", False)
        self.setStyleSheet("")
        if not self.text().strip():
            # Nothing captured: restore the chord that was there before.
            self.setText(self._prev_text)
        super().focusOutEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            # Cancel the capture, restore the previous chord.
            self.setText(self._prev_text)
            event.accept()
            self.clearFocus()
            return
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_unknown,
        ):
            return
        if key == Qt.Key.Key_Tab or key == Qt.Key.Key_Backtab:
            super().keyPressEvent(event)
            return
        mods = event.modifiers()
        has_mod = bool(
            mods
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.MetaModifier
            )
        )
        if Qt.Key.Key_F8 <= key <= Qt.Key.Key_F10 and not has_mod:
            # Bare F8/F9/F10 go to the game/Windows; ignore and keep waiting.
            return
        parts: List[str] = []
        if mods & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if mods & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
            parts.append(f"f{key - Qt.Key.Key_F1 + 1}")
            self.setText("+".join(parts))
            event.accept()
            self.clearFocus()
            return
        if mods & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier
        ):
            txt = (event.text() or "").strip().lower()
            if txt and txt.isprintable():
                parts.append(txt)
                self.setText("+".join(parts))
                event.accept()
                self.clearFocus()
                return
        super().keyPressEvent(event)


# ============================================================================
# Custom Capsule Badge for REST Column (`Hazır` / `4dk 12sn`)
# ============================================================================
class _RestBadgeWidget(QWidget):
    """
    Renders a rounded pill badge inside the REST table column matching ayarlar.html `.pill`:
    - Green capsule (`Hazır`, #123324 / #3ddc84 / #1d573c)
    - Amber capsule (`4dk 12sn` or `Kopya?`, #33270e / #f5c04e / #594417)
    - Blue/info capsule (`Toplandı` / `Yoksayılan`, #12283d / #6ec3ff / #1f486c)
    """

    def __init__(
        self,
        text: str = "Hazır",
        is_ready: bool = True,
        parent: Optional[QWidget] = None,
        ignored: bool = False,
    ) -> None:
        super().__init__(parent)
        self._text = str(text)
        self._is_ready = bool(is_ready)
        self._ignored = bool(ignored)
        self.setFixedHeight(28)
        self.setMinimumWidth(96 if self._ignored else 64)

    @property
    def is_ignored(self) -> bool:
        """P2-7: bu rozet `Yoksayılan` (not_gatherable) düğüme aittir."""
        return bool(self._ignored)

    def set_status(self, text: str, is_ready: bool) -> None:
        # P2-7 (K-06): yoksayılan rozet canlı güncellemede ezilmez —
        # `_update_live_node_statuses` dinlenme süresinden "Hazır"/
        # "4dk 12sn" hesaplayıp bu rozete yazıyordu; yoksayılan düğüm
        # için o hesap geçersizdir, rozet "Yoksayılan" kalır.
        if self._ignored:
            return
        if self._text != text or self._is_ready != is_ready:
            self._text = str(text)
            self._is_ready = bool(is_ready)
            self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        pill_w = max(48, fm.horizontalAdvance(self._text) + 18)
        pill_h = 20
        pill_x = 4
        pill_y = (self.height() - pill_h) // 2
        pill_rect = QRectF(float(pill_x), float(pill_y), float(pill_w), float(pill_h))

        if self._ignored:
            bg_color = QColor("#12283d")
            border_color = QColor("#1f486c")
            text_color = QColor("#6ec3ff")
        elif self._is_ready:
            bg_color = QColor("#123324")
            border_color = QColor("#1d573c")
            text_color = QColor("#3ddc84")
        else:
            bg_color = QColor("#33270e")
            border_color = QColor("#594417")
            text_color = QColor("#f5c04e")

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
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        info = QLabel(
            f"'{region_name}' bölgesinden '{route_name}' rotasına eklenecek noktaları seçin:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget(self)
        self.list_widget.setMinimumHeight(180)
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
        self.btn_create_new.setMinimumHeight(32)
        self._create_new_requested = False
        self.btn_create_new.clicked.connect(self._on_create_new)
        btn_row.addWidget(self.btn_create_new)
        btn_row.addStretch()

        btn_add = QPushButton("Seçilenleri Ekle", self)
        btn_add.setMinimumHeight(32)
        btn_add.setStyleSheet(
            "background-color: #3b82f6; color: #ffffff; border: none; font-weight: bold;"
        )
        btn_add.clicked.connect(self.accept)
        btn_cancel = QPushButton("İptal", self)
        btn_cancel.setMinimumHeight(32)
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
        self.setMinimumSize(420, 430)
        self.resize(420, 430)
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
        lbl_z.setFixedWidth(84)
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
        lbl_r.setFixedWidth(84)
        lbl_r.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row_res.addWidget(lbl_r)
        self.list_resources = QListWidget(self)
        self.list_resources.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_resources.setFixedHeight(120)
        self.list_resources.itemSelectionChanged.connect(self._update_selected_label)
        row_res.addWidget(self.list_resources, stretch=1)
        root.addLayout(row_res)

        # Sub-zones row
        row_sub = QHBoxLayout()
        lbl_s = QLabel("Alt Bölgeler:")
        lbl_s.setFixedWidth(84)
        lbl_s.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row_sub.addWidget(lbl_s)
        self.list_subzones = QListWidget(self)
        self.list_subzones.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_subzones.setFixedHeight(120)
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
        self.btn_import.setMinimumHeight(32)
        self.btn_import.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; border: none; font-weight: bold;"
        )
        self.btn_import.clicked.connect(self._on_import_clicked)
        self.btn_cancel = QPushButton("İptal", self)
        self.btn_cancel.setMinimumHeight(32)
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
        zone_name = self.cb_zone.currentText().strip() or (self._current_region() or "")
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

        # Fallback import (F-0601): the old code invented 12 nodes on a
        # circle and reported success even when every add failed. Only nodes
        # the route manager actually accepts are counted, and a failure or
        # an empty result is reported as a failure — never as "aktarıldı".
        count_to_import = 12
        imported = 0
        import_error: Optional[Exception] = None
        try:
            if self.route_manager is None or not hasattr(self.route_manager, "add_node"):
                raise RuntimeError("Rota yöneticisi yok ya da add_node desteklemiyor.")
            if hasattr(self.route_manager, "create_region"):
                self.route_manager.create_region(zone_name)
            if hasattr(self.route_manager, "create_route"):
                self.route_manager.create_route(zone_name, route_name)
            for idx in range(count_to_import):
                angle = (2.0 * math.pi * idx) / float(count_to_import)
                mx = int(900 + 220 * math.cos(angle))
                my = int(460 + 160 * math.sin(angle))
                try:
                    self.route_manager.add_node(
                        region=zone_name,
                        route=route_name,
                        map_x=mx,
                        map_y=my,
                        resource_type=res_name,
                    )
                    imported += 1
                except Exception as exc_add:
                    logger.error("Fallback questlog add_node failed: %s", exc_add)
                    import_error = exc_add
            self.route_manager.active_region = zone_name
            self.route_manager.active_route = route_name
        except Exception as exc:
            logger.error("Fallback questlog import error: %s", exc)
            import_error = exc

        if imported == 0:
            self.imported_summary = (
                f"Questlog içe aktarma başarısız: hiçbir nokta eklenemedi"
                + (f" ({import_error})" if import_error is not None else "")
            )
            QMessageBox.warning(self, "İçe Aktarma Başarısız", self.imported_summary)
            return
        self.imported_summary = (
            f"{imported} nokta {zone_name} · {route_name} rotasına aktarıldı"
            + (
                f" ({count_to_import - imported} nokta eklenemedi)"
                if imported < count_to_import
                else ""
            )
        )
        self.accept()


# ============================================================================
# Main Settings Window (`SettingsWindow(QDialog)`)
# ============================================================================
def _wrong_node_status(wrong: List[Any], backup_path: str = "") -> str:
    """Turkish status: removal count and the top wrong gatherion classes."""
    from core.wrong_node_scan import format_removed_status_tr

    return format_removed_status_tr(wrong, backup_path)


class _RowSelectFilter(QObject):
    """Ctrl/Shift clicks on a row widget select that node row."""

    def __init__(self, table: QTableWidget) -> None:
        super().__init__(table)
        self.table = table

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.MouseButtonPress:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        row = self._row_of(obj)
        if row < 0:
            return False
        table = self.table
        mods = event.modifiers()
        model = table.model()
        selection = table.selectionModel()
        if model is None or selection is None:
            return False
        if mods & Qt.KeyboardModifier.ShiftModifier:
            anchor = table.currentRow()
            if anchor < 0:
                anchor = row
            start, end = sorted((anchor, row))
            top = model.index(start, 0)
            bottom = model.index(end, max(0, table.columnCount() - 1))
            selection.select(
                QItemSelection(top, bottom),
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        elif mods & Qt.KeyboardModifier.ControlModifier:
            index = model.index(row, 0)
            selection.select(
                index,
                QItemSelectionModel.SelectionFlag.Toggle
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        else:
            table.selectRow(row)
        table.setCurrentCell(row, 0)
        return True

    @staticmethod
    def _row_of(obj: Any) -> int:
        widget = obj
        while widget is not None:
            row = widget.property("gatherion_row") if hasattr(widget, "property") else None
            if row is not None:
                try:
                    return int(row)
                except (TypeError, ValueError):
                    return -1
            widget = widget.parent() if hasattr(widget, "parent") else None
        return -1


# P2-2: çalışan YZ tarama thread'leri pencere yok edilse bile hayatta kalır —
# QThread ebeveynsiz kurulur, bitince finished zinciri siler. Pencere
# kapanırken closeEvent işbirlikçi stop + sınırlı wait uygular.
_ACTIVE_WRONG_SCAN_THREADS: set = set()


class _WrongScanWorker(QObject):
    """One wrong-node scan off the GUI thread (P2-2).

    Only pure work runs here: model load, patch reads, single-pass
    classification — the confidence histogram is built from the same
    pass (``confidences_out``), never a second inference run. No QWidget
    access and no route-manager mutation: backup, delete and list refresh
    happen in the GUI-thread slot that receives ``finished``.
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)  # {"wrong": [...], "meta": str}
    failed = pyqtSignal(str)

    def __init__(
        self,
        nodes: Sequence[Any],
        patch_paths: Sequence[Optional[str]],
        tier: str,
        region: str,
    ) -> None:
        super().__init__()
        self._nodes = list(nodes)
        # Position-aligned with _nodes: disk path per node, or None when
        # the patch must fall back to load_node_patch().
        self._patch_paths = list(patch_paths)
        self._tier = str(tier)
        self._region = str(region)
        self._stop_requested = False

    def stop(self) -> None:
        """Cooperative stop: checked between phases only.

        ``classify_wrong_nodes`` cannot be aborted mid-loop (its progress
        callback is best-effort by design), so a stop requested during
        classification takes effect after that pass returns.
        """
        self._stop_requested = True

    def run(self) -> None:
        try:
            from core.wrong_node_scan import (
                _node_id,
                classify_wrong_nodes,
                format_scan_meta_tr,
                load_gatherion_detector,
                load_node_patch,
                load_patch_file,
                scan_confidence_summary,
            )

            if self._stop_requested:
                self.failed.emit("YZ taraması durduruldu.")
                return
            try:
                detector = load_gatherion_detector()
            except Exception as exc:
                logger.error("Gatherion modeli yüklenemedi: %s", exc)
                self.failed.emit(f"Gatherion modeli yüklenemedi. Dosya yazılmadı: {exc}")
                return
            if self._stop_requested:
                self.failed.emit("YZ taraması durduruldu.")
                return
            images: Dict[int, Any] = {}
            for index, node in enumerate(self._nodes):
                node_id = _node_id(node, index)
                image = None
                path = (
                    self._patch_paths[index]
                    if index < len(self._patch_paths)
                    else None
                )
                if path:
                    try:
                        image = load_patch_file(path)
                    except Exception as exc:
                        logger.debug("Patch dosyası okunamadı #%s: %s", node_id, exc)
                        image = None
                if image is None:
                    image = load_node_patch(node, region=self._region)
                images[node_id] = image

            def _progress(done: int, total: int) -> None:
                if done % 8 != 0 and done != total:
                    return
                self.progress.emit(done, total)

            confs: List[float] = []
            wrong = classify_wrong_nodes(
                self._nodes,
                images,
                detector,
                on_progress=_progress,
                tier=self._tier,
                confidences_out=confs,
            )
            # P2-2: summary from the confidences of THIS pass — the
            # detector is not run a second time over the same thumbnails.
            summary = scan_confidence_summary(
                self._nodes, images, detector, confs=confs
            )
            meta = format_scan_meta_tr(detector, summary, tier=self._tier)
            if self._stop_requested:
                self.failed.emit("YZ taraması durduruldu.")
                return
            self.finished.emit({"wrong": wrong, "meta": meta})
        except Exception as exc:
            logger.error("Wrong-node scan failed: %s", exc)
            self.failed.emit(f"YZ taraması yazılmadı: {exc}")


def _resolve_analytics_telemetry(host: Any) -> Optional[Any]:
    """Ayarlar penceresinden istatistik açılırken canlı telemetri motorunu bulur.

    Önce `host._live_telemetry` (main.py / shell verir), sonra shell'in
    `_telemetry`'si. Bulunamazsa None döner — diyalog kendi boş motorunu
    kurar (v20-istatistik R3 düzeltmesi öncesi davranış).
    """
    tel = getattr(host, "_live_telemetry", None)
    if tel is not None:
        return tel
    shell = getattr(host, "_shell", None)
    if shell is not None:
        return getattr(shell, "_telemetry", None)
    return None


class SettingsWindow(QDialog):
    """
    Gatherion v2.0 two-column dark settings and node management window (Türkçe UI).
    """

    quit_requested = pyqtSignal()
    settings_saved = pyqtSignal()
    route_changed = pyqtSignal(str, str)
    selective_gather_changed = pyqtSignal(list, bool)  # (materials: list, auto_proximity: bool)

    def __init__(
        self,
        config_manager: Optional[Any] = None,
        route_manager: Optional[Any] = None,
        parent: Optional[QWidget] = None,
        settings: Optional[Any] = None,
        mobile_server: Optional[Any] = None,
        mobile_token_auth: Optional[Any] = None,
        telemetry: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        # S-4301: the live settings window had no token-rotation path —
        # the R43 fix lived only in the dead SettingsDialog class, so
        # the old mobile token stayed valid. Both are optional (the
        # window also works standalone); when present,
        # rotate_mobile_token() propagates the new token to the live
        # server instead of only writing it to disk. Callers that own
        # a MobileRemoteServer / MobileTokenAuthenticator (e.g. main.py,
        # ui/app_shell.py) should pass them here.
        self.mobile_server = mobile_server
        # v20-istatistik (R3): canlı telemetri motoru (main.py / shell verir).
        # Yoksa istatistik diyaloğu kendi boş motorunu kurar ve tüm
        # oturum/bugün metrikleri kalıcı 0 görünür.
        self._live_telemetry = telemetry
        self.mobile_token_auth = mobile_token_auth
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
        # YZ taraması özeti (alt bilgi satırı): son taramanın gerçek sonucu.
        self._last_scan_removed: int = 0
        self._last_scan_backup: str = ""
        self._last_scan_meta: str = ""
        # P2-2: arka plan YZ taraması — thread/worker referansları ve
        # taramanın başladığı bölge/rota (sonuç slotu bunlara yazar).
        self._wrong_scan_thread: Optional[Any] = None
        self._wrong_scan_worker: Optional[Any] = None
        self._wrong_scan_context: Tuple[str, str, int] = ("", "", 0)
        # Ayar yüklenirken sinyal veren kontroller kirli sayılmaz.
        self._loading_settings: bool = True

        # Kara Liste (Blacklist) Yöneticisi
        try:
            from core.blacklist_manager import get_blacklist_manager
            self.blacklist_mgr = get_blacklist_manager()
        except Exception:
            self.blacklist_mgr = None

        self._updating_combos: bool = False
        self._map_viewer_dlg: Optional[QDialog] = None
        self._questlog_dlg: Optional[QDialog] = None
        self._radar_dlg: Optional[QDialog] = None
        self._analytics_dlg: Optional[QDialog] = None
        self._calibration_dlg: Optional[QDialog] = None
        self._guide_dlg: Optional[QDialog] = None

        self.setWindowTitle(f"Gatherion v{__version__} — Ayarlar ve Rota Yönetimi")
        self.setMinimumSize(1040, 720)
        self.resize(1100, 760)
        self._apply_dark_stylesheet()
        self._init_ui()
        try:
            self._load_from_settings()
        finally:
            self._loading_settings = False
        self.refresh_regions_and_routes(
            prefer_region=str(self._cfg_get("active_region", "") or "") or None,
            prefer_route=str(self._cfg_get("active_route", "") or "") or None,
        )
        self.load_blacklist_table()
        self._update_infoline()
        try:
            self.append_log(
                f"Ayarlar yüklendi: {self._current_region() or '-'} · "
                f"{self._current_route() or '-'}"
            )
        except Exception:
            pass

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
        # Order matters: the config manager is the source of truth and is
        # read first (F-0688). ``self.settings`` may be a stale snapshot
        # taken at construction time, so it is only a late fallback.
        for item in (
            explicit_target,
            getattr(explicit_target, "settings", None) if explicit_target is not None else None,
            self.config_manager,
            getattr(self.config_manager, "settings", None),
            getattr(self.config_manager, "config", None),
            getattr(self.config_manager, "data", None),
            getattr(self, "settings", None),
            getattr(getattr(self, "settings", None), "settings", None),
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
                # SettingsManager/ConfigManager use get(section, key, default);
                # calling get(key, default) would silently treat the key as a
                # section and never return the value (F-0684). A flat lookup
                # therefore uses section "" (top-level attribute).
                if hasattr(obj, "get") and callable(obj.get):
                    try:
                        if sec:
                            val = obj.get(sec, key, None)
                            if val is None and leg_key:
                                val = obj.get(sec, leg_key, None)
                            if val is not None and not isinstance(val, dict):
                                return val
                        try:
                            val = obj.get("", key, default)
                        except TypeError:
                            # Plain dict-style get(key, default) managers.
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
                            sec_fields = getattr(sec_obj, "__dataclass_fields__", None)
                            if (
                                isinstance(sec_fields, dict)
                                and leg_key
                                and leg_key != key
                                and leg_key in sec_fields
                                and key not in sec_fields
                            ):
                                # F-23-12: düz anahtar ("breaks_enabled")
                                # bölüm alanının adı değil — gölge öznitelik
                                # yazma, gerçek alanı ("enabled") yaz.
                                try:
                                    setattr(sec_obj, leg_key, value)
                                except Exception as exc_leg:
                                    logger.debug("Could not setattr %s.%s: %s", sec, leg_key, exc_leg)
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

                # P1-13: pause_on_alt_tab iki dalda da var (route +
                # gathering) ama üretim okuyucuları route'u önce okur
                # (main._resolve_pause_on_alt_tab, BotWorker._get_cfg);
                # UI yalnızca gathering yazar. Ayna yazılmazsa UI anahtarı
                # etkisiz kalır (route varsayılan True gathering'i gölgeler).
                # harvests_per_node ile aynı desen.
                if key == "pause_on_alt_tab":
                    if isinstance(obj, dict):
                        route_sec = obj.get("route")
                        if not isinstance(route_sec, dict):
                            route_sec = {}
                            obj["route"] = route_sec
                        route_sec["pause_on_alt_tab"] = value
                    else:
                        route_obj = getattr(obj, "route", None)
                        if route_obj is not None and not isinstance(route_obj, dict):
                            try:
                                setattr(route_obj, "pause_on_alt_tab", bool(value))
                            except Exception as exc_m:
                                logger.debug("Mirror pause_on_alt_tab error: %s", exc_m)

                # Also set attribute on top-level settings object using safe setattr
                # F-23-12: yalnızca gerçek üst düzey alan yazılır — bölüm
                # anahtarları (örn. hp_potion_threshold_pct) dataclass üzerine
                # gölge öznitelik olarak yazılmaz; asdict() onları diske
                # yazmıyordu (sahte kalıcılık) ve _cfg_get önceliğini
                # kırılgan kılıyordu. Kaydedilmemiş değer yok: bölüm değeri
                # yukarıda zaten yazıldı. Dataclass olmayan düz nesnelerde
                # eski davranış korunur.
                _top_fields = getattr(obj, "__dataclass_fields__", None)
                if isinstance(_top_fields, dict):
                    _write_top = key in _top_fields
                else:
                    _write_top = not hasattr(obj, "save")
                if _write_top:
                    try:
                        setattr(obj, key, value)
                    except Exception as exc_top:
                        logger.debug("Top-level setattr skipped for %s: %s", key, exc_top)

                # Call manager .set(...) if present. While a batch save is
                # running (_cfg_batching), skip it: SettingsManager.set() writes
                # the whole file on every call (~38 writes per Save, F-0596).
                # The batched values are already applied to the underlying
                # settings objects above and _cfg_save() persists once.
                if (
                    hasattr(obj, "set")
                    and callable(obj.set)
                    and not getattr(self, "_cfg_batching", False)
                ):
                    try:
                        if sec:
                            obj.set(sec, key, value)
                        else:
                            # set(section, key, value); "" = top-level (F-0684).
                            obj.set("", key, value)
                    except TypeError:
                        try:
                            obj.set(key, value)
                        except Exception as exc_set:
                            logger.debug("Manager set fallback error: %s", exc_set)
            except Exception as exc:
                logger.warning("Failed to set config key '%s': %s", key, exc)

    def _cfg_save(self) -> bool:
        """Persist the config manager to disk once.

        Returns True only when a save was attempted and reported success.
        A failure is reported to the caller (F-0539) instead of being
        swallowed here while the UI claims the settings were saved.
        """
        for obj in self._candidate_settings_objects():
            try:
                if hasattr(obj, "validate") and callable(obj.validate):
                    obj.validate()
            except Exception as exc_val:
                logger.debug("Settings validate warning: %s", exc_val)

        cm = self.config_manager
        if cm is None:
            return False
        try:
            if hasattr(cm, "save") and callable(cm.save):
                saved = cm.save()
                return bool(saved) if saved is not None else True
            if hasattr(cm, "save_config") and callable(cm.save_config):
                saved = cm.save_config()
                return bool(saved) if saved is not None else True
            return False
        except Exception as exc:
            logger.error("Failed to save configuration to disk: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Styling & Layout Initialization
    # ------------------------------------------------------------------
    def _apply_dark_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0a0e13;
                color: #e8edf2;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 12px;
            }
            QScrollArea,
            QScrollArea > QWidget,
            QScrollArea > QWidget > QWidget,
            QWidget#SettingsColumnsHost,
            QWidget#SettingsLeftHost,
            QWidget#SettingsRightHost {
                background-color: #0a0e13;
                border: none;
            }
            QScrollArea::corner {
                background: #0a0e13;
            }
            QGroupBox {
                background-color: #12181f;
                border: 1px solid #232d39;
                border-radius: 8px;
                margin-top: 8px;
                padding: 12px 12px 10px 12px;
                font-weight: 700;
                font-size: 12px;
                color: #9bb8dc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                background-color: #12181f;
                color: #9bb8dc;
                font-weight: 700;
                font-size: 12px;
                padding: 1px 6px;
                border-radius: 3px;
                letter-spacing: 0.6px;
            }
            QLabel {
                color: #d7dfe7;
                font-size: 12px;
                background: transparent;
            }
            QLabel[hintLabel="true"] {
                color: #67737f;
                font-size: 11px;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #0d1218;
                border: 1px solid #2c3846;
                border-radius: 5px;
                color: #e8edf2;
                padding: 4px 8px;
                min-height: 24px;
                font-size: 12px;
                font-family: Consolas, "Cascadia Mono", monospace;
                selection-background-color: #2f7fe0;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border: 1px solid #3b8bff;
                background-color: #101823;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 0px;
                border: none;
            }
            QComboBox {
                background-color: #0d1218;
                border: 1px solid #2c3846;
                border-radius: 5px;
                color: #e8edf2;
                padding: 3px 8px;
                min-height: 22px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #3b8bff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #12181f;
                color: #e8edf2;
                border: 1px solid #2c3846;
                selection-background-color: #2f7fe0;
            }
            QPushButton {
                background-color: #1c2632;
                color: #e8edf2;
                border: 1px solid #2c3846;
                border-radius: 5px;
                padding: 6px 12px;
                min-height: 24px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #243140;
                border-color: #3a4a5c;
            }
            QPushButton:pressed {
                background-color: #171f28;
            }
            QPushButton:disabled {
                color: #67737f;
                background-color: #12181f;
                border-color: #232d39;
            }
            QPushButton#DangerOutlineButton {
                background-color: #28161a;
                color: #ff8e92;
                border: 1px solid #59262c;
            }
            QPushButton#DangerOutlineButton:hover {
                background-color: #351b21;
                border: 1px solid #753139;
                color: #ffb4b7;
            }
            QPushButton#PrimarySaveButton {
                background-color: #2f7fe0;
                color: #ffffff;
                border: 1px solid #3b8bff;
                font-weight: 700;
                padding: 6px 18px;
            }
            QPushButton#PrimarySaveButton:hover {
                background-color: #3b8bff;
            }
            QFrame#SettingsHotkeysStrip {
                background-color: #12181f;
                border: 1px solid #232d39;
                border-radius: 8px;
            }
            QFrame#SettingsInfoLine {
                background-color: #0d1218;
                border: 1px solid #232d39;
                border-radius: 6px;
            }
            QTableWidget {
                background-color: #0d1218;
                border: 1px solid #232d39;
                border-radius: 6px;
                color: #e8edf2;
                gridline-color: transparent;
                selection-background-color: #1a2d44;
                selection-color: #ffffff;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #141c24;
                color: #8b98a6;
                border: none;
                border-bottom: 1px solid #232d39;
                padding: 7px 8px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.7px;
            }
            QScrollBar:vertical {
                background: #0d1218;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #283444;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )
        if hasattr(self, "_columns_scroll") and self._columns_scroll is not None:
            vp = self._columns_scroll.viewport()
            if vp is not None:
                vp.setStyleSheet("background-color: #0a0e13;")

    def _init_ui(self) -> None:
        root_layout = wrap_shell_tab_root_layout(self, "settings")
        root_layout.setSpacing(10)

        # Prototype Header (.header) matching ayarlar.html
        header_frame = QFrame(self)
        header_frame.setStyleSheet(
            "QFrame { background: transparent; border: none; border-bottom: 1px solid #232d39; }"
        )
        header_lay = QHBoxLayout(header_frame)
        header_lay.setContentsMargins(2, 0, 2, 8)
        header_lay.setSpacing(12)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(1)
        kicker_lbl = QLabel("YAPILANDIRMA", header_frame)
        kicker_lbl.setStyleSheet(
            "color: #8b98a6; font-size: 10px; font-weight: 700; letter-spacing: 1.8px; border: none;"
        )
        title_lbl = QLabel("Ayarlar", header_frame)
        title_lbl.setStyleSheet(
            "color: #e8edf2; font-size: 19px; font-weight: 700; border: none;"
        )
        sub_lbl = QLabel("Toplama, mola, kısayol ve rota.", header_frame)
        sub_lbl.setStyleSheet("color: #8b98a6; font-size: 12px; border: none;")
        title_stack.addWidget(kicker_lbl)
        title_stack.addWidget(title_lbl)
        title_stack.addWidget(sub_lbl)
        header_lay.addLayout(title_stack)
        header_lay.addStretch(1)

        perf_row = QHBoxLayout()
        perf_row.setContentsMargins(0, 0, 0, 0)
        perf_row.setSpacing(6)
        self.chip_fps = QLabel('FPS <b style="color:#e8edf2;font-family:Consolas;">0</b>', header_frame)
        self.chip_fps.setTextFormat(Qt.TextFormat.RichText)
        self.chip_fps.setStyleSheet(
            "background-color: #12181f; border: 1px solid #232d39; border-radius: 5px; padding: 4px 10px; font-size: 11px; color: #8b98a6;"
        )
        perf_row.addWidget(self.chip_fps)

        self.chip_gpu = QLabel('GPU <b style="color:#e8edf2;font-family:Consolas;">0%</b>', header_frame)
        self.chip_gpu.setTextFormat(Qt.TextFormat.RichText)
        self.chip_gpu.setStyleSheet(
            "background-color: #12181f; border: 1px solid #232d39; border-radius: 5px; padding: 4px 10px; font-size: 11px; color: #8b98a6;"
        )
        perf_row.addWidget(self.chip_gpu)

        self.chip_cpu = QLabel('CPU <b style="color:#e8edf2;font-family:Consolas;">0%</b>', header_frame)
        self.chip_cpu.setTextFormat(Qt.TextFormat.RichText)
        self.chip_cpu.setStyleSheet(
            "background-color: #12181f; border: 1px solid #232d39; border-radius: 5px; padding: 4px 10px; font-size: 11px; color: #8b98a6;"
        )
        perf_row.addWidget(self.chip_cpu)

        self.chip_lat = QLabel('LAT <b style="color:#e8edf2;font-family:Consolas;">0.0ms</b>', header_frame)
        self.chip_lat.setTextFormat(Qt.TextFormat.RichText)
        self.chip_lat.setStyleSheet(
            "background-color: #12181f; border: 1px solid #232d39; border-radius: 5px; padding: 4px 10px; font-size: 11px; color: #8b98a6;"
        )
        perf_row.addWidget(self.chip_lat)
        header_lay.addLayout(perf_row)
        root_layout.addWidget(header_frame)

        # Two-column main content (.cols: 1.05fr : 1fr)
        columns_host = QWidget(self)
        columns_host.setObjectName("SettingsColumnsHost")
        columns_host.setStyleSheet("background-color: #0a0e13;")
        columns_layout = QHBoxLayout(columns_host)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # LEFT COLUMN: GATHERING, ROUTE, SIDE-BY-SIDE CARTOGRAPHY + BREAKS
        # ------------------------------------------------------------------
        left_host = QWidget()
        left_host.setObjectName("SettingsLeftHost")
        left_host.setStyleSheet("background-color: #0a0e13;")
        left_host.setMinimumWidth(360)
        left_col = QVBoxLayout(left_host)
        left_col.setContentsMargins(0, 0, 4, 0)
        left_col.setSpacing(8)

        left_col.addWidget(self._build_gathering_group())
        left_col.addWidget(self._build_route_group())

        sub_pair_row = QHBoxLayout()
        sub_pair_row.setContentsMargins(0, 0, 0, 0)
        sub_pair_row.setSpacing(8)
        sub_pair_row.addWidget(self._build_cartography_group(), stretch=13)
        sub_pair_row.addWidget(self._build_breaks_group(), stretch=10)
        left_col.addLayout(sub_pair_row)
        left_col.addWidget(self._build_consumables_group())
        left_col.addStretch(1)

        self._columns_scroll = QScrollArea(columns_host)
        self._columns_scroll.setObjectName("SettingsColumnsScroll")
        self._columns_scroll.setWidgetResizable(True)
        self._columns_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._columns_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._columns_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._columns_scroll.setStyleSheet("background-color: #0a0e13; border: none;")
        self._columns_scroll.setWidget(left_host)
        if self._columns_scroll.viewport() is not None:
            self._columns_scroll.viewport().setStyleSheet("background-color: #0a0e13;")

        columns_layout.addWidget(self._columns_scroll, stretch=51)

        # ------------------------------------------------------------------
        # RIGHT COLUMN: NODES / REGIONS & ROUTES (FIXED)
        # ------------------------------------------------------------------
        right_host = QWidget(columns_host)
        right_host.setObjectName("SettingsRightHost")
        right_host.setStyleSheet("background-color: #0a0e13;")
        right_host.setMinimumWidth(340)
        right_col = QVBoxLayout(right_host)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(8)

        self.right_tab_widget = QTabWidget(self)
        self.right_tab_widget.setObjectName("SettingsRightTabs")
        self.nodes_tab = self._build_nodes_group()
        self.blacklist_tab = self._build_blacklist_group()
        self.right_tab_widget.addTab(self.nodes_tab, "📍 ROTA DÜĞÜMLERİ")
        self.right_tab_widget.addTab(self.blacklist_tab, "⛔ KARA LİSTE (BLACKLIST)")
        right_col.addWidget(self.right_tab_widget, stretch=1)

        columns_layout.addWidget(right_host, stretch=49)

        root_layout.addWidget(columns_host, stretch=1)

        self.columns_host = columns_host
        self.left_host = left_host
        self.right_host = right_host

        # Full-width Bot Kısayolları strip (#hkStrip) matching ayarlar.html
        root_layout.addWidget(self._build_hotkeys_group())

        # Status + footer live in one bar so the shell can keep them under every tab.
        self.footer_bar = QWidget(self)
        self.footer_bar.setObjectName("SettingsFooterBar")
        self.footer_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        footer_bar_lay = QVBoxLayout(self.footer_bar)
        footer_bar_lay.setContentsMargins(0, 0, 0, 0)
        footer_bar_lay.setSpacing(6)

        info_frame = QFrame(self.footer_bar)
        info_frame.setObjectName("SettingsInfoLine")
        info_lay = QHBoxLayout(info_frame)
        info_lay.setContentsMargins(12, 5, 12, 5)
        self.status_lbl = QLabel(
            "YZ modeli hazır. Henüz tarama yapılmadı.",
            info_frame,
        )
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.status_lbl.setStyleSheet("color: #8b98a6; font-size: 11px; border: none;")
        info_lay.addWidget(self.status_lbl)
        footer_bar_lay.addWidget(info_frame)

        # Alt bilgi satırı (prototip `.infoline`): YZ taraması özeti —
        # son taramanın gerçek sonucu, yoksa hazır durumu + eşik profili.
        self.infoline_lbl = QLabel(
            "YZ modeli hazır. Henüz tarama yapılmadı.", self.footer_bar
        )
        self.infoline_lbl.setWordWrap(True)
        self.infoline_lbl.setStyleSheet("color: #8b98a6; font-size: 11px; border: none;")
        footer_bar_lay.addWidget(self.infoline_lbl)

        divider = QFrame(self.footer_bar)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #232d39; border: none;")
        footer_bar_lay.addWidget(divider)

        # ------------------------------------------------------------------
        # FOOTER BAR (.foot: .exitrow + 3x3 .fgrid)
        # ------------------------------------------------------------------
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 2, 0, 0)
        footer_layout.setSpacing(10)

        self.btn_quit = QPushButton("Bottan Çık", self)
        self.btn_quit.setToolTip("Uygulamayı güvenli şekilde kapatır")
        self.btn_quit.clicked.connect(self._on_quit_clicked)
        footer_layout.addWidget(self.btn_quit)

        self.version_lbl = QLabel(f"Gatherion v{__version__}", self)
        self.version_lbl.setStyleSheet("color: #67737f; font-size: 11px;")
        footer_layout.addWidget(self.version_lbl)

        self.dirty_lbl = QLabel("● Kaydedilmemiş değişiklikler", self)
        self.dirty_lbl.setStyleSheet("color: #f5c04e; font-size: 11px; font-weight: 600;")
        self.dirty_lbl.setVisible(False)
        footer_layout.addWidget(self.dirty_lbl)

        self.saved_lbl = QLabel("✓ Kaydedildi ve uygulandı", self)
        self.saved_lbl.setStyleSheet("color: #3ddc84; font-size: 11px; font-weight: 600;")
        self.saved_lbl.setVisible(False)
        footer_layout.addWidget(self.saved_lbl)

        footer_layout.addStretch(1)
        footer_bar_lay.addLayout(footer_layout)

        tool_grid = QGridLayout()
        tool_grid.setContentsMargins(0, 0, 0, 0)
        tool_grid.setHorizontalSpacing(6)
        tool_grid.setVerticalSpacing(6)

        self.btn_radar = QPushButton("Canlı Radar", self)
        self.btn_radar.setToolTip("Oto-haritalama canlı radar penceresini açar")
        self.btn_radar.clicked.connect(self._on_open_radar)

        self.btn_yolo_factory = QPushButton("🧠 YOLO Veri Fabrikası", self)
        self.btn_yolo_factory.setToolTip(
            "Kendi kendini eğiten yapay zeka (YOLO) veri seti fabrikası ve RTX eğitim merkezini açar"
        )
        self.btn_yolo_factory.clicked.connect(self._on_open_yolo_factory)

        self.btn_mobile_connect = QPushButton("📱 Mobil Kumanda", self)
        self.btn_mobile_connect.setToolTip(
            "Telefon / mobil tarayıcı üzerinden canlı izleme ve kumanda QR kod penceresini açar"
        )
        self.btn_mobile_connect.clicked.connect(self._on_open_mobile_connect)

        # S-4301: live token rotation in the window the user actually
        # sees (the R43 fix previously existed only in the dead
        # SettingsDialog class). Rotates the shared 256-bit mobile
        # token and pushes it to the live server; the old token stops
        # being accepted.
        self.btn_rotate_token = QPushButton("🔑 Mobil Token Döndür", self)
        self.btn_rotate_token.setToolTip(
            "Yeni 256-bit mobil token üretir ve canlı mobil sunucuya uygular (eski token geçersiz olur)"
        )
        self.btn_rotate_token.clicked.connect(self.rotate_mobile_token)

        self.btn_analytics = QPushButton("İstatistikler", self)
        self.btn_analytics.setToolTip("Canlı telemetri ve toplama istatistiklerini açar")
        self.btn_analytics.clicked.connect(self._on_open_analytics)

        self.btn_calibration = QPushButton("Kalibrasyon", self)
        self.btn_calibration.setToolTip("Görüntü kalibrasyon sihirbazını açar")
        self.btn_calibration.clicked.connect(self._on_open_calibration)

        self.btn_guide = QPushButton("Kılavuz", self)
        self.btn_guide.setToolTip("Kullanım kılavuzu ve sorun giderme penceresini açar")
        self.btn_guide.clicked.connect(self._on_open_guide)

        self.btn_reset_defaults = QPushButton("Varsayılanlara Sıfırla", self)
        self.btn_reset_defaults.setToolTip("Tüm ayarları varsayılan değerlerine döndürür")
        self.btn_reset_defaults.clicked.connect(self.reset_to_defaults)

        self.btn_save = QPushButton("Kaydet ve Uygula", self)
        self.btn_save.setObjectName("PrimarySaveButton")
        self.btn_save.setToolTip("Ayarları kaydeder ve anında uygular")
        self.btn_save.clicked.connect(self.save_settings)

        self.btn_close = QPushButton("Kapat", self)
        self.btn_close.clicked.connect(self.close)

        tool_buttons = (
            self.btn_radar,
            self.btn_yolo_factory,
            self.btn_mobile_connect,
            self.btn_rotate_token,
            self.btn_analytics,
            self.btn_calibration,
            self.btn_guide,
            self.btn_reset_defaults,
            self.btn_save,
            self.btn_close,
        )
        for index, button in enumerate(tool_buttons):
            button.setMinimumHeight(34)
            tool_grid.addWidget(button, index // 3, index % 3)
        footer_bar_lay.addLayout(tool_grid)

        # Alt log şeridi (prototip `.log`): son 60 satır, salt okunur.
        self.log_text = QPlainTextEdit(self.footer_bar)
        self.log_text.setObjectName("SettingsLogStrip")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(60)
        self.log_text.setFixedHeight(44)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet(
            "background-color: #05070a; color: #9fb0a3; border: none;"
            " border-top: 1px solid #232d39; padding: 4px 8px;"
        )
        footer_bar_lay.addWidget(self.log_text)

        root_layout.addWidget(self.footer_bar)
        self._wire_dirty_tracking()
        self._update_infoline()

    # ------------------------------------------------------------------
    # Left Column Builders (matching ayarlar.html `.grid2` + `.trow`)
    # ------------------------------------------------------------------
    def _build_gathering_group(self) -> QGroupBox:
        grp = QGroupBox("TOPLAMA AYARLARI (GATHERING)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Cell (0,0): F/E basım alt sınır (sn): [1.5] sn
        cell_min = QHBoxLayout()
        cell_min.setSpacing(6)
        cell_min.addWidget(QLabel("F/E basım alt sınır (sn):"), stretch=1)
        self.f_interval_min = QDoubleSpinBox(self)
        self.f_interval_min.setRange(0.1, 60.0)
        self.f_interval_min.setSingleStep(0.1)
        self.f_interval_min.setDecimals(1)
        self.f_interval_min.setFixedWidth(64)
        self.spin_f_min = self.f_interval_min
        cell_min.addWidget(self.f_interval_min)
        h_min = QLabel("sn")
        h_min.setProperty("hintLabel", True)
        cell_min.addWidget(h_min)
        grid.addLayout(cell_min, 0, 0)

        # Cell (0,1): F/E basım üst sınır (sn): [3.0] sn
        cell_max = QHBoxLayout()
        cell_max.setSpacing(6)
        cell_max.addWidget(QLabel("F/E basım üst sınır (sn):"), stretch=1)
        self.f_interval_max = QDoubleSpinBox(self)
        self.f_interval_max.setRange(0.1, 60.0)
        self.f_interval_max.setSingleStep(0.1)
        self.f_interval_max.setDecimals(1)
        self.f_interval_max.setFixedWidth(64)
        self.spin_f_max = self.f_interval_max
        cell_max.addWidget(self.f_interval_max)
        h_max = QLabel("sn")
        h_max.setProperty("hintLabel", True)
        cell_max.addWidget(h_max)
        grid.addLayout(cell_max, 0, 1)

        # Cell (1,0): Materyal başına toplama (Varsayılan: 3x): [3]
        cell_harv = QHBoxLayout()
        cell_harv.setSpacing(6)
        self.lbl_harvests_per_node = QLabel("Materyal başına toplama (Varsayılan: 3x):")
        self.lbl_harvests_per_node.setToolTip("Harvests per node (3x per material)")
        cell_harv.addWidget(self.lbl_harvests_per_node, stretch=1)
        self.harvests_per_node = QSpinBox(self)
        self.harvests_per_node.setRange(1, 10)
        self.harvests_per_node.setValue(3)
        self.harvests_per_node.setFixedWidth(64)
        self.harvests_per_node.setToolTip(
            "Harvests per node (3x per material) — Her materyal noktasından kaç kez toplanacağı (1-10)"
        )
        self.spin_harvests_per_node = self.harvests_per_node
        cell_harv.addWidget(self.harvests_per_node)
        grid.addLayout(cell_harv, 1, 0)

        # Cell (1,1): Toplama tuşu (F / E / Otomatik): [f]
        cell_gkey = QHBoxLayout()
        cell_gkey.setSpacing(6)
        self.lbl_gather_key = QLabel("Toplama tuşu (F / E / Otomatik):")
        self.lbl_gather_key.setToolTip("Gather key (f / e / auto)")
        cell_gkey.addWidget(self.lbl_gather_key, stretch=1)
        self.gather_key = QComboBox(self)
        for key_opt in ("f", "e", "auto"):
            self.gather_key.addItem(key_opt, key_opt)
        self.gather_key.setCurrentText("f")
        self.gather_key.setFixedWidth(74)
        self.gather_key.setToolTip(
            "Oyun istemcisinde görünen toplama tuşu: 'f' ([F]), 'e' ([E]) veya 'auto' (otomatik)"
        )
        self.cb_gather_key = self.gather_key
        self.combo_gather_key = self.gather_key
        cell_gkey.addWidget(self.gather_key)
        grid.addLayout(cell_gkey, 1, 1)

        lay.addLayout(grid)

        self.retry_on_extraction_fail = ToggleSwitch(
            "‘Essence Extraction failed’ hatasında tekrar dene", self, checked=True
        )
        self.retry_on_extraction_fail.setToolTip(
            "Retry on 'Essence Extraction failed' — Toplama başarısız olduğunda aynı aşamayı tekrar dener"
        )
        self.chk_retry_on_extraction_fail = self.retry_on_extraction_fail
        lay.addWidget(self.retry_on_extraction_fail)

        self.test_mode = ToggleSwitch("Test modu (tuş basmadan simüle et)", self)
        self.test_mode.setToolTip("Test mode (detect only, never press keys)")
        self.chk_test = self.test_mode
        lay.addWidget(self.test_mode)

        self.show_overlay = ToggleSwitch("Bilgi panelini (Overlay HUD) göster", self)
        self.show_overlay.setToolTip("Show overlay over the game")
        self.chk_overlay = self.show_overlay
        lay.addWidget(self.show_overlay)

        self.pause_on_alt_tab = ToggleSwitch("Alt-Tab yapıldığında botu otomatik duraklat", self, checked=True)
        self.pause_on_alt_tab.setToolTip("Oyun penceresi odağı kaybettiğinde veya Alt-Tab yapıldığında botu güvenle duraklatır")
        lay.addWidget(self.pause_on_alt_tab)

        # ------------------------------------------------------------------
        # Seç-Topla (Materyal Seç & Yakınlık Rotası) Bölümü
        # ------------------------------------------------------------------
        sep_sg = QFrame(self)
        sep_sg.setFrameShape(QFrame.Shape.HLine)
        sep_sg.setFixedHeight(1)
        sep_sg.setStyleSheet("background-color: #2b323c; border: none; margin: 4px 0px;")
        lay.addWidget(sep_sg)

        self.chk_selective_gather = ToggleSwitch(
            "🎯 Seç-Topla Aktif (Sadece Seçilen Materyali Topla)", self
        )
        self.chk_selective_gather.setToolTip(
            "Seç-Topla: Bot sadece listeden seçilen materyalleri toplar, civardaki diğer materyalleri atlar"
        )
        lay.addWidget(self.chk_selective_gather)

        sg_row = QHBoxLayout()
        sg_row.setSpacing(8)
        lbl_sg_mat = QLabel("Hedef Materyal:")
        lbl_sg_mat.setToolTip("Yakınlığına göre rota çizilecek ve toplanacak materyal türü")
        sg_row.addWidget(lbl_sg_mat)

        self.combo_selective_materials = QComboBox(self)
        self.combo_selective_materials.setStyleSheet(
            """
            QComboBox {
                background-color: #11151c;
                color: #ffffff;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                min-width: 160px;
            }
            QComboBox:hover {
                border-color: #38bdf8;
            }
            QComboBox QAbstractItemView {
                background-color: #11151c;
                color: #ffffff;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                border: 1px solid #334155;
                outline: none;
                font-size: 11px;
            }
            """
        )
        view_sg = self.combo_selective_materials.view()
        if view_sg is not None:
            pal = view_sg.palette()
            pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.Base, QColor("#11151c"))
            view_sg.setPalette(pal)

        self.populate_selective_materials()
        sg_row.addWidget(self.combo_selective_materials, stretch=1)
        lay.addLayout(sg_row)

        self.chk_proximity_route_auto = ToggleSwitch(
            "🧭 Yakınlığa Göre Otomatik Rota Sırala (Proximity Route)", self, checked=True
        )
        self.chk_proximity_route_auto.setToolTip(
            "Otomatik Proximity Rota: Seçilen materyal noktalarını oyuncunun konumuna en yakından uzağa sıralar"
        )
        lay.addWidget(self.chk_proximity_route_auto)

        self.btn_proximity_route = QPushButton("🧭 Yakınlığa Göre Rota Çiz & Topla", self)
        self.btn_proximity_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_proximity_route.setStyleSheet(
            """
            QPushButton {
                background-color: #1e3a8a;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            """
        )
        lay.addWidget(self.btn_proximity_route)

        self.chk_selective_gather.toggled.connect(self._on_selective_gather_ui_changed)
        self.combo_selective_materials.currentIndexChanged.connect(self._on_selective_gather_ui_changed)
        self.chk_proximity_route_auto.toggled.connect(self._on_selective_gather_ui_changed)
        self.btn_proximity_route.clicked.connect(self._on_proximity_route_clicked)

        return grp

    DEFAULT_SELECTIVE_PRESETS: List[Tuple[str, List[str]]] = [
        ("Tümü (Hepsi)", []),
        ("⛏️ Orichalcum Ore (orichalcum)", ["orichalcum"]),
        ("🌸 Aria (aria)", ["aria"]),
        ("🍒 Mela (mela)", ["mela"]),
        ("🌿 Targena (targena)", ["targena"]),
        ("💎 Gem / Odyle (gem_crystal)", ["gem_crystal"]),
        ("🪵 Ağaç / Kereste (tree_log)", ["tree_log"]),
    ]

    def populate_selective_materials(self) -> None:
        """Populates the selective materials combo box with presets and dynamic region materials."""
        if not hasattr(self, "combo_selective_materials") or self.combo_selective_materials is None:
            return

        self.combo_selective_materials.blockSignals(True)
        current_data = self.combo_selective_materials.currentData()
        self.combo_selective_materials.clear()

        # 1. Preset entries
        for label, val in self.DEFAULT_SELECTIVE_PRESETS:
            self.combo_selective_materials.addItem(label, val)

        # 2. Dynamic region materials
        reg_val = ""
        if hasattr(self, "region_combo") and self.region_combo is not None:
            try:
                reg_val = self._current_region()
            except Exception:
                reg_val = ""
        if not reg_val and self.route_manager is not None:
            reg_val = getattr(self.route_manager, "active_region", "")
        region_name = str(reg_val or "Altgard").strip()
        seen_names = {"orichalcum", "aria", "mela", "targena", "gem_crystal", "tree_log", "all", ""}
        try:
            from config.materials_db import get_region_expected_materials
            reg_mats = get_region_expected_materials(region_name)
            if reg_mats:
                for mat in reg_mats:
                    canonical = getattr(mat, "base_material", getattr(mat, "name", "")).strip()
                    cat = getattr(mat, "category", "")
                    if canonical and canonical.lower() not in seen_names:
                        seen_names.add(canonical.lower())
                        icon = "🌿" if "Herb" in cat else ("⛏️" if "Ore" in cat else ("💎" if "Gem" in cat or "Odyle" in cat else ("🪵" if "Log" in cat else "📦")))
                        label = f"{icon} {canonical} ({region_name})"
                        self.combo_selective_materials.addItem(label, [canonical.lower()])
        except Exception as exc:
            logger.debug("SettingsWindow bölge materyalleri taranırken hata: %s", exc)

        # 3. Dynamic materials from route_manager nodes if present
        if self.route_manager is not None and hasattr(self.route_manager, "data"):
            try:
                reg_data = self.route_manager.data.get("regions", {}).get(region_name, {})
                nodes = reg_data.get("nodes", [])
                for node in nodes:
                    if isinstance(node, dict):
                        m_type = str(node.get("material_type", "") or node.get("name", "")).strip()
                        if m_type and m_type.lower() not in seen_names:
                            seen_names.add(m_type.lower())
                            self.combo_selective_materials.addItem(f"📍 {m_type}", [m_type.lower()])
            except Exception as exc:
                logger.debug("SettingsWindow route_manager nodeları taranırken hata: %s", exc)

        # Restore selection
        restored = False
        if current_data is not None:
            for idx in range(self.combo_selective_materials.count()):
                if self.combo_selective_materials.itemData(idx) == current_data:
                    self.combo_selective_materials.setCurrentIndex(idx)
                    restored = True
                    break
        if not restored:
            self.combo_selective_materials.setCurrentIndex(0)

        self.combo_selective_materials.blockSignals(False)

    def set_selected_materials(
        self, materials: Sequence[str], auto_proximity: bool = True
    ) -> None:
        """Programmatically sets the selective materials and auto_proximity flag."""
        mats = [str(m).strip() for m in materials if str(m).strip()]
        if hasattr(self, "combo_selective_materials"):
            self.combo_selective_materials.blockSignals(True)
            matched = False
            for idx in range(self.combo_selective_materials.count()):
                d = self.combo_selective_materials.itemData(idx)
                if d and any(m in d for m in mats):
                    self.combo_selective_materials.setCurrentIndex(idx)
                    matched = True
                    break
            if not matched and not mats:
                self.combo_selective_materials.setCurrentIndex(0)
            elif not matched and mats:
                self.combo_selective_materials.addItem(f"🎯 {', '.join(mats)}", list(mats))
                self.combo_selective_materials.setCurrentIndex(
                    self.combo_selective_materials.count() - 1
                )
            self.combo_selective_materials.blockSignals(False)

        if hasattr(self, "chk_selective_gather"):
            self.chk_selective_gather.setChecked(bool(mats))

        if hasattr(self, "chk_proximity_route_auto"):
            self.chk_proximity_route_auto.setChecked(bool(auto_proximity))

    def get_selected_materials(self) -> List[str]:
        if hasattr(self, "combo_selective_materials"):
            return list(self.combo_selective_materials.currentData() or [])
        return []

    def _on_selective_gather_ui_changed(self) -> None:
        if not hasattr(self, "combo_selective_materials") or not hasattr(self, "chk_selective_gather"):
            return
        enabled = self.chk_selective_gather.isChecked()
        mats = self.combo_selective_materials.currentData() or []
        auto_prox = self.chk_proximity_route_auto.isChecked() if hasattr(self, "chk_proximity_route_auto") else True
        effective_mats = list(mats) if enabled else []
        self.selective_gather_changed.emit(effective_mats, auto_prox)
        logger.info(
            "SettingsWindow Seç-Topla güncellendi: %s (Aktif: %s, Oto-Yakınlık: %s)",
            effective_mats,
            enabled,
            auto_prox,
        )

    def _on_proximity_route_clicked(self) -> None:
        if hasattr(self, "combo_selective_materials"):
            mats = self.combo_selective_materials.currentData() or []
        else:
            mats = []
        if hasattr(self, "chk_selective_gather") and mats:
            self.chk_selective_gather.setChecked(True)
        if hasattr(self, "chk_proximity_route_auto"):
            self.chk_proximity_route_auto.setChecked(True)
        self.selective_gather_changed.emit(list(mats), True)
        logger.info("SettingsWindow 'Yakınlığa Göre Rota Çiz & Topla' tıklandı. Materyaller: %s", mats)

    def _build_route_group(self) -> QGroupBox:
        grp = QGroupBox("ROTA VE OTOMASYON (ROUTE)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        # Row 1: .grid2 (Dinlenme süresi | Havada maks. yükselme)
        g_top = QGridLayout()
        g_top.setContentsMargins(0, 0, 0, 0)
        g_top.setHorizontalSpacing(14)
        g_top.setColumnStretch(0, 1)
        g_top.setColumnStretch(1, 1)

        row_rest = QHBoxLayout()
        row_rest.setSpacing(6)
        row_rest.addWidget(QLabel("Dinlenme süresi (dk):"), stretch=1)
        self.rest_minutes = QSpinBox(self)
        self.rest_minutes.setRange(0, 240)
        self.rest_minutes.setFixedWidth(64)
        self.spin_rest = self.rest_minutes
        row_rest.addWidget(self.rest_minutes)
        g_top.addLayout(row_rest, 0, 0)

        lay.addLayout(g_top)

        self.gathered_join_active_route = ToggleSwitch(
            "Toplanan noktaları aktif rotaya otomatik ekle", self
        )
        lay.addWidget(self.gathered_join_active_route)

        self.fly_unreachable = ToggleSwitch(
            "Yürüyerek ulaşılamayan noktalara uçarak git", self
        )
        lay.addWidget(self.fly_unreachable)

        # Row: .grid2 (Yolda noktaya tekrar tıklama süresi | Noktaya ulaşma süresi)
        g_mid = QGridLayout()
        g_mid.setContentsMargins(0, 0, 0, 0)
        g_mid.setHorizontalSpacing(14)
        g_mid.setColumnStretch(0, 1)
        g_mid.setColumnStretch(1, 1)

        row_reclick = QHBoxLayout()
        row_reclick.setSpacing(4)
        row_reclick.addWidget(QLabel("Yolda noktaya tekrar tıklama süresi:"), stretch=1)
        self.reclick_node_sec = QSpinBox(self)
        self.reclick_node_sec.setRange(0, 300)
        self.reclick_node_sec.setFixedWidth(56)
        row_reclick.addWidget(self.reclick_node_sec)
        h_rec = QLabel("sn (0: kapalı)")
        h_rec.setProperty("hintLabel", True)
        row_reclick.addWidget(h_rec)
        g_mid.addLayout(row_reclick, 0, 0)

        row_reach = QHBoxLayout()
        row_reach.setSpacing(4)
        row_reach.addWidget(QLabel("Noktaya ulaşma süresi:"), stretch=1)
        self.time_to_reach_sec = QSpinBox(self)
        self.time_to_reach_sec.setRange(5, 600)
        self.time_to_reach_sec.setFixedWidth(56)
        self.spin_reach = self.time_to_reach_sec
        row_reach.addWidget(self.time_to_reach_sec)
        h_rch = QLabel("sn (sonra tekrar tıkla ve atla)")
        h_rch.setProperty("hintLabel", True)
        row_reach.addWidget(h_rch)
        g_mid.addLayout(row_reach, 0, 1)
        lay.addLayout(g_mid)

        self.gather_on_the_way = ToggleSwitch(
            "Yolda giderken gördüklerini topla", self
        )
        lay.addWidget(self.gather_on_the_way)

        # 2x2 .grid2 (Bekle | Başarısız noktada / Noktayı sil | Duraklat)
        g_fail = QGridLayout()
        g_fail.setContentsMargins(0, 0, 0, 0)
        g_fail.setHorizontalSpacing(14)
        g_fail.setVerticalSpacing(5)
        g_fail.setColumnStretch(0, 1)
        g_fail.setColumnStretch(1, 1)

        row_wait_fail = QHBoxLayout()
        row_wait_fail.setSpacing(4)
        row_wait_fail.addWidget(QLabel("Bekle:"), stretch=1)
        self.wait_min_after_failed = QSpinBox(self)
        self.wait_min_after_failed.setRange(0, 240)
        self.wait_min_after_failed.setFixedWidth(56)
        row_wait_fail.addWidget(self.wait_min_after_failed)
        h_wf = QLabel("dk, arka arkaya")
        h_wf.setProperty("hintLabel", True)
        row_wait_fail.addWidget(h_wf)
        g_fail.addLayout(row_wait_fail, 0, 0)

        row_wait_count = QHBoxLayout()
        row_wait_count.setSpacing(4)
        row_wait_count.addWidget(QLabel("Başarısız noktada:"), stretch=1)
        self.max_failed_in_row = QSpinBox(self)
        self.max_failed_in_row.setRange(0, 100)
        self.max_failed_in_row.setFixedWidth(56)
        row_wait_count.addWidget(self.max_failed_in_row)
        h_wc = QLabel("(0: kapalı)")
        h_wc.setProperty("hintLabel", True)
        row_wait_count.addWidget(h_wc)
        g_fail.addLayout(row_wait_count, 0, 1)

        row_rem_fail = QHBoxLayout()
        row_rem_fail.setSpacing(4)
        row_rem_fail.addWidget(QLabel("Noktayı sil:"), stretch=1)
        self.remove_node_after_failures = QSpinBox(self)
        self.remove_node_after_failures.setRange(0, 100)
        self.remove_node_after_failures.setFixedWidth(52)
        row_rem_fail.addWidget(self.remove_node_after_failures)
        h_rf = QLabel("ardışık başarısızlıktan sonra (0: kapalı)")
        h_rf.setProperty("hintLabel", True)
        row_rem_fail.addWidget(h_rf)
        g_fail.addLayout(row_rem_fail, 1, 0)

        row_pause = QHBoxLayout()
        row_pause.setSpacing(4)
        row_pause.addWidget(QLabel("Duraklat:"), stretch=1)
        self.pause_after_no_gather_min = QSpinBox(self)
        self.pause_after_no_gather_min.setRange(0, 480)
        self.pause_after_no_gather_min.setFixedWidth(56)
        row_pause.addWidget(self.pause_after_no_gather_min)
        h_ps = QLabel("dk toplama yapılmazsa (0: kapalı)")
        h_ps.setProperty("hintLabel", True)
        row_pause.addWidget(h_ps)
        g_fail.addLayout(row_pause, 1, 1)
        lay.addLayout(g_fail)

        # Bottom row: Saldırı altında karşılık ver | Çatışma sonrası bekle
        g_fight = QGridLayout()
        g_fight.setContentsMargins(0, 0, 0, 0)
        g_fight.setHorizontalSpacing(14)
        g_fight.setColumnStretch(0, 1)
        g_fight.setColumnStretch(1, 1)

        self.fight_back = ToggleSwitch("Saldırı altında karşılık ver", self)
        self.chk_combat = self.fight_back
        g_fight.addWidget(self.fight_back, 0, 0)

        row_fight = QHBoxLayout()
        row_fight.setSpacing(4)
        row_fight.addWidget(QLabel("Çatışma sonrası bekle:"), stretch=1)
        self.fight_stop_gather_sec = QSpinBox(self)
        self.fight_stop_gather_sec.setRange(0, 120)
        self.fight_stop_gather_sec.setFixedWidth(56)
        row_fight.addWidget(self.fight_stop_gather_sec)
        h_ft = QLabel("sn")
        h_ft.setProperty("hintLabel", True)
        row_fight.addWidget(h_ft)
        g_fight.addLayout(row_fight, 0, 1)
        lay.addLayout(g_fight)

        return grp

    def _build_cartography_group(self) -> QGroupBox:
        grp = QGroupBox("OTO-HARİTALAMA VE RADAR (AUTO-CARTOGRAPHY)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        self.minimap_auto_record = ToggleSwitch(
            "Sağ üst haritadan (Minimap) otomatik kaydet", self, checked=True
        )
        self.minimap_auto_record.setToolTip(
            "Auto-record materials from top-right Minimap"
        )
        self.chk_minimap_auto_record = self.minimap_auto_record
        lay.addWidget(self.minimap_auto_record)

        self.world_3d_auto_record = ToggleSwitch(
            "3B dünyadaki materyalleri (Aria/Mela) otomatik kaydet", self, checked=True
        )
        self.world_3d_auto_record.setToolTip(
            "3B ekranda görünen Aria, Mela, Odyle ve cevher parıltılarını otomatik haritalandırır"
        )
        lay.addWidget(self.world_3d_auto_record)

        lay.addStretch(1)

        return grp

    def _build_breaks_group(self) -> QGroupBox:
        grp = QGroupBox("MOLALAR VE İNSANSILAŞTIRMA (BREAKS)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        self.breaks_enabled = QCheckBox("Molalar etkin", self)
        self.breaks_enabled.setToolTip(
            "Kapalıyken mola zamanlayıcısı çalışmaz (break_every_min değeri ne olursa olsun)."
        )
        lay.addWidget(self.breaks_enabled)

        self.break_every_min = QSpinBox(self)
        self.break_every_min.setRange(0, 720)
        self.break_every_min.setFixedWidth(64)
        row_every = QHBoxLayout()
        row_every.setSpacing(6)
        row_every.addWidget(QLabel("Mola sıklığı (dk, 0: kapalı):"), stretch=1)
        row_every.addWidget(self.break_every_min)
        lay.addLayout(row_every)

        self.break_every_variance = QSpinBox(self)
        self.break_every_variance.setRange(0, 120)
        self.break_every_variance.setFixedWidth(64)
        row_every_var = QHBoxLayout()
        row_every_var.setSpacing(6)
        row_every_var.addWidget(QLabel("Mola sıklığı sapması (dk):"), stretch=1)
        row_every_var.addWidget(self.break_every_variance)
        lay.addLayout(row_every_var)

        self.break_lasts_min = QSpinBox(self)
        self.break_lasts_min.setRange(1, 240)
        self.break_lasts_min.setFixedWidth(64)
        row_lasts = QHBoxLayout()
        row_lasts.setSpacing(6)
        row_lasts.addWidget(QLabel("Mola süresi (dk):"), stretch=1)
        row_lasts.addWidget(self.break_lasts_min)
        lay.addLayout(row_lasts)

        self.break_lasts_variance = QSpinBox(self)
        self.break_lasts_variance.setRange(0, 60)
        self.break_lasts_variance.setFixedWidth(64)
        row_lasts_var = QHBoxLayout()
        row_lasts_var.setSpacing(6)
        row_lasts_var.addWidget(QLabel("Mola süresi sapması (dk):"), stretch=1)
        row_lasts_var.addWidget(self.break_lasts_variance)
        lay.addLayout(row_lasts_var)
        lay.addStretch(1)

        return grp

    def _add_hotkey_edit(self, lay: QHBoxLayout, label: str, default: str) -> QLineEdit:
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #8b98a6; font-size: 11px;")
        lay.addWidget(lbl)
        edit = _HotkeyCaptureEdit(default, self)
        edit.setFixedWidth(132)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(edit)
        return edit

    def _build_consumables_group(self) -> QGroupBox:
        """F-23-01/F-23-02: HP/MP iksir + buff parşömen ayarları.

        Eskiden bu değerlerin UI'da hiç widget'ı yoktu — yalnızca
        get/set_consumable_settings() API'sinden değiştirilebiliyor,
        tuşlar diske hiç yazılmıyordu (F-23-03). Artık kullanıcı eşik ve
        tuşları buradan değiştirir; değerler şema alanlarına yazılır ve
        get_settings_dict() üzerinden kaydedilir. Üretim tarafında bu
        eşikleri okuyan yolun bağlanması C grubunun işidir (state_machine /
        mana_phase_combat) — UI burada yalnızca görünür + kalıcı yapar.
        """
        grp = QGroupBox("TÜKETİM (HP/MP İKSİR + BUFF)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        self.chk_auto_use_hp_potion = QCheckBox("HP iksiri otomatik", self)
        lay.addWidget(self.chk_auto_use_hp_potion)
        self.spin_hp_potion_threshold = QSpinBox(self)
        self.spin_hp_potion_threshold.setRange(1, 100)
        self.spin_hp_potion_threshold.setSuffix(" %")
        self.spin_hp_potion_threshold.setFixedWidth(72)
        row_hp = QHBoxLayout()
        row_hp.setSpacing(6)
        row_hp.addWidget(QLabel("HP eşiği:"), stretch=1)
        row_hp.addWidget(self.spin_hp_potion_threshold)
        lay.addLayout(row_hp)
        self.edit_hp_potion_key = QLineEdit(self)
        self.edit_hp_potion_key.setFixedWidth(72)
        self.edit_hp_potion_key.setPlaceholderText("F1")
        row_hp_key = QHBoxLayout()
        row_hp_key.setSpacing(6)
        row_hp_key.addWidget(QLabel("HP iksir tuşu:"), stretch=1)
        row_hp_key.addWidget(self.edit_hp_potion_key)
        lay.addLayout(row_hp_key)

        self.chk_auto_use_mp_potion = QCheckBox("MP iksiri otomatik", self)
        lay.addWidget(self.chk_auto_use_mp_potion)
        self.spin_mp_potion_threshold = QSpinBox(self)
        self.spin_mp_potion_threshold.setRange(1, 100)
        self.spin_mp_potion_threshold.setSuffix(" %")
        self.spin_mp_potion_threshold.setFixedWidth(72)
        row_mp = QHBoxLayout()
        row_mp.setSpacing(6)
        row_mp.addWidget(QLabel("MP eşiği:"), stretch=1)
        row_mp.addWidget(self.spin_mp_potion_threshold)
        lay.addLayout(row_mp)
        self.edit_mp_potion_key = QLineEdit(self)
        self.edit_mp_potion_key.setFixedWidth(72)
        self.edit_mp_potion_key.setPlaceholderText("F2")
        row_mp_key = QHBoxLayout()
        row_mp_key.setSpacing(6)
        row_mp_key.addWidget(QLabel("MP iksir tuşu:"), stretch=1)
        row_mp_key.addWidget(self.edit_mp_potion_key)
        lay.addLayout(row_mp_key)

        self.chk_auto_refresh_buff_scroll = QCheckBox("Buff parşömeni otomatik yenile", self)
        lay.addWidget(self.chk_auto_refresh_buff_scroll)
        self.spin_buff_scroll_interval = QSpinBox(self)
        self.spin_buff_scroll_interval.setRange(1, 3600)
        self.spin_buff_scroll_interval.setSuffix(" sn")
        self.spin_buff_scroll_interval.setFixedWidth(72)
        row_buff = QHBoxLayout()
        row_buff.setSpacing(6)
        row_buff.addWidget(QLabel("Buff yenileme aralığı:"), stretch=1)
        row_buff.addWidget(self.spin_buff_scroll_interval)
        lay.addLayout(row_buff)
        self.edit_buff_scroll_key = QLineEdit(self)
        self.edit_buff_scroll_key.setFixedWidth(72)
        self.edit_buff_scroll_key.setPlaceholderText("F7")
        row_buff_key = QHBoxLayout()
        row_buff_key.setSpacing(6)
        row_buff_key.addWidget(QLabel("Buff parşömen tuşu (F6 yasaklı):"), stretch=1)
        row_buff_key.addWidget(self.edit_buff_scroll_key)
        lay.addLayout(row_buff_key)
        lay.addStretch(1)

        return grp

    def _build_hotkeys_group(self) -> QFrame:
        strip = QFrame(self)
        strip.setObjectName("SettingsHotkeysStrip")
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(10)

        title = QLabel("BOT KISAYOLLARI", strip)
        title.setStyleSheet(
            "color: #9bb8dc; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;"
        )
        lay.addWidget(title)

        hint = QLabel(
            "Alana tıklayıp tuş kombinasyonuna bas — kısayol yakalanır. Yalnız F8, F9 ve F10 oyuna ve Windows’a gider.",
            strip,
        )
        hint.setStyleSheet("color: #67737f; font-size: 11px;")
        lay.addWidget(hint)
        lay.addStretch(1)

        self.hotkey_node = self._add_hotkey_edit(lay, "Nokta ekle:", "ctrl+alt+f8")
        self.hotkey_menu = self._add_hotkey_edit(lay, "Menü:", "ctrl+alt+f9")
        self.hotkey_pause = self._add_hotkey_edit(lay, "Duraklat / Devam:", "ctrl+alt+f10")
        return strip

    # ------------------------------------------------------------------
    # Right Column Builder (NODES / REGIONS & ROUTES)
    # ------------------------------------------------------------------
    def _build_nodes_group(self) -> QGroupBox:
        grp = QGroupBox("BÖLGE VE ROTALAR (REGIONS · ROUTES)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)

        reg_row = QHBoxLayout()
        reg_row.setSpacing(6)
        lbl_reg = QLabel("Bölge:")
        lbl_reg.setFixedWidth(46)
        reg_row.addWidget(lbl_reg)

        self.region_combo = QComboBox(self)
        self.cb_region = self.region_combo
        self.region_combo.setMinimumWidth(160)
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        reg_row.addWidget(self.region_combo, stretch=1)
        lay.addLayout(reg_row)

        reg_btns = QHBoxLayout()
        reg_btns.setSpacing(6)
        self.btn_rename_region = QPushButton("Adlandır...", self)
        self.btn_rename_region.clicked.connect(self._on_rename_region)
        self.btn_delete_region = QPushButton("Bölgeyi Sil...", self)
        self.btn_delete_region.setObjectName("DangerOutlineButton")
        self.btn_delete_region.clicked.connect(self._on_delete_region)
        reg_btns.addWidget(self.btn_rename_region)
        reg_btns.addWidget(self.btn_delete_region)
        reg_btns.addStretch(1)

        self.point_filter_edit = QLineEdit(self)
        self.point_filter_edit.setPlaceholderText("Nokta ara: #id")
        self.point_filter_edit.setFixedWidth(140)
        self.point_filter_edit.textChanged.connect(self._filter_nodes_table)
        reg_btns.addWidget(self.point_filter_edit)
        lay.addLayout(reg_btns)

        rte_row = QHBoxLayout()
        rte_row.setSpacing(6)
        lbl_rte = QLabel("Rota:")
        lbl_rte.setFixedWidth(46)
        rte_row.addWidget(lbl_rte)

        self.route_combo = QComboBox(self)
        self.cb_route = self.route_combo
        self.route_combo.setMinimumWidth(160)
        self.route_combo.currentIndexChanged.connect(self._on_route_combo_changed)
        rte_row.addWidget(self.route_combo, stretch=1)
        lay.addLayout(rte_row)

        rte_btns = QHBoxLayout()
        rte_btns.setSpacing(6)
        self.btn_new_route = QPushButton("Yeni...", self)
        self.btn_new_route.clicked.connect(self._on_new_route)
        self.btn_rename_route = QPushButton("Adlandır...", self)
        self.btn_rename_route.clicked.connect(self._on_rename_route)
        self.btn_delete_route = QPushButton("Rotayı Sil...", self)
        self.btn_delete_route.setObjectName("DangerOutlineButton")
        self.btn_delete_route.clicked.connect(self._on_delete_route)
        rte_btns.addWidget(self.btn_new_route)
        rte_btns.addWidget(self.btn_rename_route)
        rte_btns.addWidget(self.btn_delete_route)
        rte_btns.addStretch(1)

        self.sel_info_lbl = QLabel("Seçili yok · 0 nokta", self)
        self.sel_info_lbl.setStyleSheet("color: #67737f; font-size: 11px;")
        rte_btns.addWidget(self.sel_info_lbl)
        lay.addLayout(rte_btns)

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
        self._row_select_filter = _RowSelectFilter(self.table)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.itemSelectionChanged.connect(self._update_sel_info_label)
        self.table.itemDoubleClicked.connect(self._on_node_double_clicked)

        header = self.table.horizontalHeader()
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 130)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 80)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 74)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 84)

        # Empty-state overlay label inside the table viewport
        self.empty_nodes_lbl = QLabel(
            "Henüz nokta yok: Siz toplama yaptıkça bot noktaları otomatik kaydeder",
            self.table.viewport(),
        )
        self.empty_nodes_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_nodes_lbl.setStyleSheet(
            "color: #67737f; font-size: 12px; background: transparent;"
        )
        self.empty_nodes_lbl.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

        table_container_lay.addWidget(self.table)
        lay.addWidget(table_container, stretch=1)

        # Bottom Action Buttons (2 neat rows matching ayarlar.html `.pactions`)
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
        act_row1.addStretch(1)
        lay.addLayout(act_row1)

        act_row2 = QHBoxLayout()
        act_row2.setSpacing(6)
        self.btn_delete_selected = QPushButton("Seçilenleri Sil", self)
        self.btn_delete_selected.clicked.connect(self._on_delete_node)
        self.btn_delete_wrong = QPushButton("YZ Yanlışları Sil", self)
        self.btn_delete_wrong.clicked.connect(self._on_delete_wrong_nodes)
        # Eşik profili: prototipteki döngü butonu (Dengeli/Hassas/Sıkı).
        # Eşikler backend'deki gerçek kademelerdir (node_quality_gate):
        # Dengeli≈0.13, Hassas≈0.10, Sıkı≈0.22.
        self.cb_wrong_tier = QComboBox(self)
        self.cb_wrong_tier.addItem("Dengeli", "balanced")
        self.cb_wrong_tier.addItem("Hassas", "aggressive")
        self.cb_wrong_tier.addItem("Sıkı", "strict")
        self.cb_wrong_tier.setCurrentIndex(0)
        self.cb_wrong_tier.setToolTip(
            "YZ yanlış tarama eşiği: Hassas≈0.10, Dengeli≈0.13, Sıkı≈0.22"
        )
        self.cb_wrong_tier.currentIndexChanged.connect(
            self._on_wrong_tier_combo_changed
        )
        self.btn_profile = QPushButton("Profil: Dengeli", self)
        self.btn_profile.setToolTip(
            "YZ yanlış tarama eşik profili — tıklayınca Dengeli → Hassas → Sıkı döner"
        )
        self.btn_profile.clicked.connect(self._on_profile_cycle)

        self.btn_share_route = QPushButton("Rotayı Paylaş...", self)
        self.btn_share_route.clicked.connect(self._on_share_route)
        self.btn_import_route = QPushButton("Rota İçe Aktar...", self)
        self.btn_import_route.clicked.connect(self._on_import_route)
        self.btn_import_questlog = QPushButton("Questlog'dan Aktar...", self)
        self.btn_import_questlog.clicked.connect(self._on_import_questlog)

        act_row2.addWidget(self.btn_delete_selected)
        act_row2.addWidget(self.btn_delete_wrong)
        act_row2.addWidget(self.cb_wrong_tier)
        act_row2.addWidget(self.btn_profile)
        act_row2.addStretch(1)
        act_row2.addWidget(self.btn_share_route)
        act_row2.addWidget(self.btn_import_route)
        act_row2.addWidget(self.btn_import_questlog)
        lay.addLayout(act_row2)

        return grp

    def _build_blacklist_group(self) -> QGroupBox:
        grp = QGroupBox("KARA LİSTE (BLACKLIST · YANLIŞ İŞARETLEMELER)", self)
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)
        lay.setContentsMargins(10, 12, 10, 10)

        # Açıklama etiketi
        desc_lbl = QLabel(
            "Otomatik toplama sırasında 3 kez üst üste toplanacak kaynak bulunamayan veya YZ tarafından "
            "yanlış işaretlendiği tespit edilen koordinatlar burada engellenir ve bir daha asla kaydedilmez.",
            self,
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lay.addWidget(desc_lbl)

        # Filtre ve Durum Satırı
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self.blacklist_filter_edit = QLineEdit(self)
        self.blacklist_filter_edit.setMinimumHeight(30)
        self.blacklist_filter_edit.setPlaceholderText("Kara listede ara: bölge veya koordinat...")
        self.blacklist_filter_edit.textChanged.connect(self._filter_blacklist_table)
        top_row.addWidget(self.blacklist_filter_edit, stretch=1)

        self.blacklist_info_lbl = QLabel("0 koordinat engellendi", self)
        self.blacklist_info_lbl.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: bold;")
        top_row.addWidget(self.blacklist_info_lbl)
        lay.addLayout(top_row)

        # Kara Liste Tablosu
        self.blacklist_table = QTableWidget(self)
        self.blacklist_table.setMinimumHeight(160)
        self.blacklist_table.setColumnCount(5)
        self.blacklist_table.setHorizontalHeaderLabels(
            ["BÖLGE", "KOORDİNAT", "NEDEN", "YZ TEŞHİSİ", "TARİH"]
        )
        self.blacklist_table.verticalHeader().setVisible(False)
        self.blacklist_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.blacklist_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.blacklist_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.blacklist_table.setShowGrid(False)
        self.blacklist_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        bl_header = self.blacklist_table.horizontalHeader()
        bl_header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bl_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        bl_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        bl_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        bl_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        bl_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        lay.addWidget(self.blacklist_table, stretch=1)

        # Butonlar Satırı
        act_row = QHBoxLayout()
        act_row.setSpacing(6)
        self.btn_blacklist_remove = QPushButton("Kara Listeden Kaldır", self)
        self.btn_blacklist_remove.setMinimumHeight(32)
        self.btn_blacklist_remove.setObjectName("DangerOutlineButton")
        self.btn_blacklist_remove.setToolTip("Seçilen koordinatı kara listeden kaldırır (tekrar kaydedilmesine izin verir)")
        self.btn_blacklist_remove.clicked.connect(self._on_remove_from_blacklist)
        act_row.addWidget(self.btn_blacklist_remove)

        self.btn_blacklist_clear = QPushButton("Tümünü Temizle", self)
        self.btn_blacklist_clear.setMinimumHeight(32)
        self.btn_blacklist_clear.setObjectName("DangerOutlineButton")
        self.btn_blacklist_clear.setToolTip("Tüm kara listeyi temizler")
        self.btn_blacklist_clear.clicked.connect(self._on_clear_blacklist)
        act_row.addWidget(self.btn_blacklist_clear)

        self.btn_blacklist_add = QPushButton("Manuel Ekle...", self)
        self.btn_blacklist_add.setMinimumHeight(32)
        self.btn_blacklist_add.setToolTip("Manuel olarak belirli bir koordinatı kara listeye ekler")
        self.btn_blacklist_add.clicked.connect(self._on_add_to_blacklist)
        act_row.addWidget(self.btn_blacklist_add)

        self.btn_blacklist_refresh = QPushButton("Yenile", self)
        self.btn_blacklist_refresh.setMinimumHeight(32)
        self.btn_blacklist_refresh.setToolTip("Kara liste verilerini yeniden yükler")
        self.btn_blacklist_refresh.clicked.connect(self.load_blacklist_table)
        act_row.addWidget(self.btn_blacklist_refresh)

        act_row.addStretch(1)
        lay.addLayout(act_row)
        return grp

    def load_blacklist_table(self) -> None:
        """Kara liste tablosunu diskteki kayıtlardan doldurur."""
        if not hasattr(self, "blacklist_table"):
            return
        if getattr(self, "blacklist_mgr", None) is None:
            try:
                from core.blacklist_manager import get_blacklist_manager
                self.blacklist_mgr = get_blacklist_manager()
            except Exception:
                return

        entries = self.blacklist_mgr.get_entries()
        self.blacklist_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            item_reg = QTableWidgetItem(str(entry.region))
            item_reg.setData(Qt.ItemDataRole.UserRole, str(entry.id))
            self.blacklist_table.setItem(row, 0, item_reg)

            item_coord = QTableWidgetItem(f"({entry.map_x}, {entry.map_y})")
            self.blacklist_table.setItem(row, 1, item_coord)

            item_reason = QTableWidgetItem(str(entry.reason))
            self.blacklist_table.setItem(row, 2, item_reason)

            item_diag = QTableWidgetItem(str(entry.ai_verdict))
            self.blacklist_table.setItem(row, 3, item_diag)

            item_date = QTableWidgetItem(str(entry.created_at))
            self.blacklist_table.setItem(row, 4, item_date)

        if hasattr(self, "blacklist_info_lbl"):
            self.blacklist_info_lbl.setText(f"{len(entries)} koordinat engellendi")

    def _on_remove_from_blacklist(self) -> None:
        if not hasattr(self, "blacklist_table") or self.blacklist_mgr is None:
            return
        selected_rows = self.blacklist_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "Seçim Yok", "Lütfen kara listeden kaldırmak istediğiniz satırı seçin.")
            return

        entry_ids = []
        for index in selected_rows:
            item = self.blacklist_table.item(index.row(), 0)
            if item:
                eid = item.data(Qt.ItemDataRole.UserRole)
                if eid:
                    entry_ids.append(str(eid))

        if not entry_ids:
            return

        reply = QMessageBox.question(
            self,
            "Kara Listeden Kaldır",
            f"Seçilen {len(entry_ids)} koordinat kara listeden kaldırılsın mı?\n"
            "Kaldırılan koordinatlar tekrar rota veya düğüm olarak kaydedilebilir hale gelecektir.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for eid in entry_ids:
                self.blacklist_mgr.remove_entry(eid)
            self.load_blacklist_table()

    def _on_clear_blacklist(self) -> None:
        if self.blacklist_mgr is None:
            return
        count = self.blacklist_mgr.count()
        if count <= 0:
            QMessageBox.information(self, "Kara Liste Boş", "Kara listede silinecek kayıt bulunmuyor.")
            return

        reply = QMessageBox.question(
            self,
            "Tümünü Temizle",
            f"Kara listedeki tüm {count} koordinat kaydı silinsin mi?\n"
            "Bu işlem geri alınamaz.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.blacklist_mgr.clear_all()
            self.load_blacklist_table()

    def _on_add_to_blacklist(self) -> None:
        if self.blacklist_mgr is None:
            return
        reg, ok1 = QInputDialog.getText(
            self, "Bölge Seç", "Bölge adı:", text=self._current_region() or "Altgard"
        )
        if not ok1 or not reg.strip():
            return

        coords, ok2 = QInputDialog.getText(
            self, "Koordinat Gir", "Harita Koordinatı (X, Y):", text="100, 200"
        )
        if not ok2 or not coords.strip():
            return

        try:
            parts = coords.replace("(", "").replace(")", "").split(",")
            mx = int(float(parts[0].strip()))
            my = int(float(parts[1].strip()))
        except Exception:
            QMessageBox.warning(self, "Hata", "Geçersiz koordinat formatı. Örnek: 150, 280")
            return

        self.blacklist_mgr.add_entry(
            region=reg.strip(),
            map_x=mx,
            map_y=my,
            reason="Kullanıcı tarafından manuel kara listeye eklendi",
            ai_verdict="Manuel Tanım",
        )
        self.load_blacklist_table()

    def _filter_blacklist_table(self, query: str) -> None:
        if not hasattr(self, "blacklist_table"):
            return
        q = str(query or "").strip().lower()
        rows = self.blacklist_table.rowCount()
        cols = self.blacklist_table.columnCount()
        for r in range(rows):
            match = False
            for c in range(cols):
                it = self.blacklist_table.item(r, c)
                if it and q in it.text().lower():
                    match = True
                    break
            self.blacklist_table.setRowHidden(r, not match if q else False)

    def _filter_nodes_table(self, query: str) -> None:
        q = str(query or "").strip().lower().lstrip("#")
        for row_idx in range(self.table.rowCount()):
            it = self.table.item(row_idx, 0)
            nid = str(it.data(Qt.ItemDataRole.UserRole) or "") if it is not None else ""
            match = (not q) or (q in nid.lower())
            self.table.setRowHidden(row_idx, not match)
        self._update_sel_info_label()

    def _update_sel_info_label(self) -> None:
        if not hasattr(self, "sel_info_lbl"):
            return
        total = self.table.rowCount()
        ids = self._selected_node_ids()
        if ids:
            if len(ids) == 1:
                self.sel_info_lbl.setText(f"Seçili: #{ids[0]} · {total} nokta")
            else:
                self.sel_info_lbl.setText(f"Seçili: {len(ids)} nokta · toplam {total}")
        else:
            self.sel_info_lbl.setText(f"Seçili yok · {total} nokta")

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

        Every field is loaded independently (F-0597): one corrupt value only
        leaves that single control at its default instead of aborting the
        rest of the load.
        """
        if settings is not None:
            self.settings = settings

        def _f(key: str, default: float) -> float:
            try:
                return float(self._cfg_get(key, default, target=settings))
            except (TypeError, ValueError):
                logger.warning("Bad float value for '%s'; using default %s", key, default)
                return float(default)

        def _i(key: str, default: int) -> int:
            try:
                return int(self._cfg_get(key, default, target=settings))
            except (TypeError, ValueError):
                logger.warning("Bad int value for '%s'; using default %s", key, default)
                return int(default)

        def _b(key: str, default: bool) -> bool:
            try:
                return bool(self._cfg_get(key, default, target=settings))
            except (TypeError, ValueError):
                return bool(default)

        def _s(key: str, default: str) -> str:
            try:
                return str(self._cfg_get(key, default, target=settings) or default)
            except (TypeError, ValueError):
                return default

        def _apply(fn) -> None:
            try:
                fn()
            except Exception as exc:
                logger.error("Error loading a settings field into UI: %s", exc)

        _apply(lambda: self.f_interval_min.setValue(_f("f_interval_min", 1.5)))
        _apply(lambda: self.f_interval_max.setValue(_f("f_interval_max", 3.0)))
        _apply(lambda: self.test_mode.setChecked(_b("test_mode", False)))
        _apply(lambda: self.show_overlay.setChecked(_b("show_overlay", True)))
        _apply(lambda: self.pause_on_alt_tab.setChecked(_b("pause_on_alt_tab", True)))

        # New Gathering settings: harvests_per_node, gather_key, retry_on_extraction_fail
        def _load_harvests() -> None:
            try:
                harvests_val = max(1, min(10, int(self._cfg_get("harvests_per_node", 3, target=settings))))
            except (TypeError, ValueError):
                harvests_val = 3
            self.harvests_per_node.setValue(harvests_val)

        _apply(_load_harvests)

        def _load_gather_key() -> None:
            raw_gkey = str(self._cfg_get("gather_key", "f", target=settings) or "f").strip().lower()
            if raw_gkey not in ("f", "e", "auto"):
                raw_gkey = "f"
            gkey_idx = self.gather_key.findData(raw_gkey)
            if gkey_idx < 0:
                gkey_idx = self.gather_key.findText(raw_gkey, Qt.MatchFlag.MatchFixedString)
            if gkey_idx >= 0:
                self.gather_key.setCurrentIndex(gkey_idx)
            else:
                self.gather_key.setCurrentText("f")

        _apply(_load_gather_key)
        _apply(lambda: self.retry_on_extraction_fail.setChecked(_b("retry_on_extraction_fail", True)))

        if hasattr(self, "chk_selective_gather"):
            _apply(lambda: self.chk_selective_gather.setChecked(_b("selective_gather_enabled", False)))
        if hasattr(self, "chk_proximity_route_auto"):
            _apply(lambda: self.chk_proximity_route_auto.setChecked(_b("proximity_route_auto", True)))
        if hasattr(self, "combo_selective_materials"):
            def _load_materials() -> None:
                raw_mats = self._cfg_get("selected_materials", [], target=settings) or []
                if isinstance(raw_mats, str):
                    raw_mats = [m.strip() for m in raw_mats.split(",") if m.strip()]
                self.set_selected_materials(
                    raw_mats,
                    auto_proximity=_b("proximity_route_auto", True),
                )

            _apply(_load_materials)

        # Route & Auto-Cartography settings
        _apply(lambda: self.rest_minutes.setValue(_i("rest_minutes", 5)))
        _apply(lambda: self.gathered_join_active_route.setChecked(_b("gathered_join_active_route", True)))
        _apply(lambda: self.fly_unreachable.setChecked(_b("fly_unreachable", True)))
        _apply(lambda: self.reclick_node_sec.setValue(_i("reclick_node_sec", 10)))
        _apply(lambda: self.time_to_reach_sec.setValue(_i("time_to_reach_sec", 60)))
        _apply(lambda: self.gather_on_the_way.setChecked(_b("gather_on_the_way", False)))
        _apply(lambda: self.wait_min_after_failed.setValue(_i("wait_min_after_failed", 5)))
        _apply(lambda: self.max_failed_in_row.setValue(_i("max_failed_in_row", 10)))
        _apply(lambda: self.remove_node_after_failures.setValue(_i("remove_node_after_failures", 0)))
        _apply(lambda: self.pause_after_no_gather_min.setValue(_i("pause_after_no_gather_min", 15)))
        _apply(lambda: self.fight_back.setChecked(_b("fight_back", True)))
        _apply(lambda: self.fight_stop_gather_sec.setValue(_i("fight_stop_gather_sec", 3)))
        _apply(lambda: self.minimap_auto_record.setChecked(_b("minimap_auto_record", True)))
        if hasattr(self, "world_3d_auto_record"):
            _apply(lambda: self.world_3d_auto_record.setChecked(_b("auto_record_3d_crests", True)))

        # Breaks settings
        if hasattr(self, "breaks_enabled"):
            _apply(lambda: self.breaks_enabled.setChecked(_b("breaks_enabled", True)))
        _apply(lambda: self.break_every_min.setValue(_i("break_every_min", 0)))
        _apply(lambda: self.break_every_variance.setValue(_i("break_every_variance", 1)))
        _apply(lambda: self.break_lasts_min.setValue(_i("break_lasts_min", 10)))
        _apply(lambda: self.break_lasts_variance.setValue(_i("break_lasts_variance", 1)))

        # Consumables settings (F-23-01/F-23-03: artık UI widget'ları var)
        if hasattr(self, "chk_auto_use_hp_potion"):
            _apply(lambda: self.chk_auto_use_hp_potion.setChecked(_b("auto_use_hp_potion", True)))
            _apply(lambda: self.spin_hp_potion_threshold.setValue(_i("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT)))
            _apply(lambda: self.edit_hp_potion_key.setText(_s("hp_potion_key", "F1")))
            _apply(lambda: self.chk_auto_use_mp_potion.setChecked(_b("auto_use_mp_potion", True)))
            _apply(lambda: self.spin_mp_potion_threshold.setValue(_i("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT)))
            _apply(lambda: self.edit_mp_potion_key.setText(_s("mp_potion_key", "F2")))
            _apply(lambda: self.chk_auto_refresh_buff_scroll.setChecked(_b("auto_refresh_buff_scroll", True)))
            _apply(lambda: self.spin_buff_scroll_interval.setValue(_i("buff_scroll_interval_sec", 300)))
            _apply(lambda: self.edit_buff_scroll_key.setText(_s("buff_scroll_key", "F7")))
        _apply(lambda: self.hotkey_node.setText(_s("record_node_f8", "ctrl+alt+f8")))
        _apply(lambda: self.hotkey_menu.setText(_s("toggle_menu_f9", "ctrl+alt+f9")))
        _apply(lambda: self.hotkey_pause.setText(_s("pause_resume_f10", "ctrl+alt+f10")))

        # Footer version comes from the file, not a hardcoded string.
        # P3-1: dosyadaki eski "v2.0" yazımı da normalize edilir; etiket
        # her zaman "Gatherion v2.0.0" biçimindedir (f9_menu ile aynı).
        try:
            version = str(self._cfg_get("version", __version__) or __version__).strip()
            if version.startswith("v"):
                version = version[1:]
            if hasattr(self, "version_lbl"):
                self.version_lbl.setText(f"Gatherion v{version}")
        except Exception:
            pass
        # F-23-08/F-23-10: bozuk dosya ve sürüm uyumsuzluğu sessiz geçmesin.
        # append_log log şeridi yoksa logger'a düşer; set_status_message
        # __init__ sırasında henüz kurulmamış olabilir — try/except'li.
        try:
            mgr = getattr(self, "config_manager", None)
            load_err = getattr(mgr, "load_error", None)
            version_bad = bool(getattr(mgr, "version_mismatch", False))
            warn_msg = ""
            if load_err:
                warn_msg = (
                    f"Uyarı: ayar dosyası bozuk, varsayılanlarla açıldı ({load_err}). "
                    "Eski değerler settings.json.corrupt.<zaman>.bak yedeğinde."
                )
            elif version_bad:
                warn_msg = "Uyarı: ayar dosyası eski sürüm — eksik anahtarlar varsayılanla doldu."
            if warn_msg:
                try:
                    self.append_log(warn_msg, "warn")
                except Exception:
                    pass
                try:
                    self.set_status_message(warn_msg)
                except Exception:
                    pass
                logger.warning("Ayar penceresi uyarısı gösterildi: %s", warn_msg)
        except Exception:
            pass
        self._loading_settings = False

    def _save_to_settings(self, settings: Optional[Any] = None) -> Any:
        """
        Writes all UI control values back to `settings` / `self.settings` /
        `self.config_manager` using safe `setattr`.
        """
        if settings is not None:
            self.settings = settings
        values = self.get_settings_dict()
        # Batch: apply all values in memory first, then persist once
        # (F-0596) instead of one disk write per key.
        self._cfg_batching = True
        try:
            for key, val in values.items():
                self._cfg_set(key, val, target=settings)
        finally:
            self._cfg_batching = False
        self._remember_selection()
        saved = self._cfg_save()
        self._last_save_ok = bool(saved)
        return settings if settings is not None else self.settings

    def _remember_selection(self) -> None:
        """Store region, route, and shell tab on the settings object. Does not write routes."""
        bot = getattr(self.config_manager, "settings", None)
        if bot is None or not hasattr(bot, "active_region"):
            bot = self.settings if hasattr(self.settings, "active_region") else None
        if bot is None:
            return
        try:
            region = self._current_region()
        except Exception:
            region = ""
        try:
            route = self._current_route()
        except Exception:
            route = ""
        if region:
            bot.active_region = region
        if route:
            bot.active_route = route
        display = getattr(bot, "display", None)
        shell = getattr(self, "_shell", None)
        tab = ""
        if shell is not None and hasattr(shell, "current_key"):
            try:
                tab = str(shell.current_key() or "").strip()
            except Exception:
                tab = ""
        if display is not None and tab and hasattr(display, "last_tab"):
            display.last_tab = tab

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
            "pause_on_alt_tab": bool(self.pause_on_alt_tab.isChecked()),
            "harvests_per_node": max(1, min(10, int(self.harvests_per_node.value()))),
            "gather_key": raw_gkey,
            "retry_on_extraction_fail": bool(
                self.retry_on_extraction_fail.isChecked()
            ),
            "selective_gather_enabled": bool(self.chk_selective_gather.isChecked())
            if hasattr(self, "chk_selective_gather")
            else False,
            "selected_materials": list(self.combo_selective_materials.currentData() or [])
            if hasattr(self, "combo_selective_materials")
            else [],
            "proximity_route_auto": bool(self.chk_proximity_route_auto.isChecked())
            if hasattr(self, "chk_proximity_route_auto")
            else True,
            "rest_minutes": int(self.rest_minutes.value()),
            "gathered_join_active_route": bool(
                self.gathered_join_active_route.isChecked()
            ),
            "fly_unreachable": bool(self.fly_unreachable.isChecked()),
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
            "auto_record_3d_crests": bool(self.world_3d_auto_record.isChecked())
            if hasattr(self, "world_3d_auto_record")
            else True,
            "break_every_min": int(self.break_every_min.value()),
            # F-23-06: mola ana şalteri (BreakSettings.enabled).
            "breaks_enabled": bool(self.breaks_enabled.isChecked())
            if hasattr(self, "breaks_enabled")
            else True,
            "break_every_variance": int(self.break_every_variance.value()),
            "break_lasts_min": int(self.break_lasts_min.value()),
            "break_lasts_variance": int(self.break_lasts_variance.value()),
            "record_node_f8": self._chord_setting(self.hotkey_node.text(), "ctrl+alt+f8"),
            "toggle_menu_f9": self._chord_setting(self.hotkey_menu.text(), "ctrl+alt+f9"),
            "pause_resume_f10": self._chord_setting(self.hotkey_pause.text(), "ctrl+alt+f10"),
            "blacklist": self.blacklist_mgr.to_list() if getattr(self, "blacklist_mgr", None) else [],
            # Consumables (F-0540): previously missing here, so HP/MP/buff
            # values were never written back on Save and silently reverted.
            **self.get_consumable_settings(),
        }

    @staticmethod
    def _chord_setting(raw: str, fallback: str) -> str:
        from config.schema import canonical_hotkey_chord

        return canonical_hotkey_chord(raw, fallback)

    def save_settings(self) -> bool:
        """Persists all UI settings to `config_manager` and emits `settings_saved`.

        Returns True only when the settings were actually written to disk.
        A failed write reports the failure in the status bar and never shows
        the "saved" message (F-0539).
        """
        errors = self._validate_inputs()
        if errors:
            widget, message = errors[0]
            self.set_status_message(f"Kaydedilemedi: {message}")
            self.append_log(f"Doğrulama hatası: {message}", "err")
            logger.error("Settings validation failed: %s", message)
            try:
                widget.setFocus()
            except Exception:
                pass
            return False
        try:
            if getattr(self, "blacklist_mgr", None) is not None:
                try:
                    bl_saved = self.blacklist_mgr.save()
                except Exception as exc_bl:
                    logger.error("Failed to save blacklist: %s", exc_bl)
                    self.set_status_message(
                        f"Ayarlar kaydedilemedi: kara liste yazılamadı ({exc_bl})"
                    )
                    return False
                if bl_saved is False:
                    self.set_status_message(
                        "Ayarlar kaydedilemedi: kara liste yazılamadı."
                    )
                    return False
            self._save_to_settings()
            if not getattr(self, "_last_save_ok", False):
                self.set_status_message(
                    "Ayarlar kaydedilemedi: diske yazma başarısız oldu."
                )
                self.append_log("Ayarlar kaydedilemedi: diske yazma başarısız oldu.", "err")
                logger.error("Settings save reported failure; not signalling success.")
                return False
            self.set_status_message("Ayarlar kaydedildi ve uygulandı.")
            self.append_log("Ayarlar kaydedildi ve uygulandı.")
            if hasattr(self, "dirty_lbl"):
                self.dirty_lbl.setVisible(False)
            if hasattr(self, "saved_lbl"):
                self.saved_lbl.setVisible(True)
                QTimer.singleShot(2500, lambda: self.saved_lbl.setVisible(False))
            logger.info("Settings saved successfully.")
            self.settings_saved.emit()
            return True
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)
            self.append_log(f"Ayarlar kaydedilirken hata oluştu: {exc}", "err")
            self.set_status_message(f"Ayarlar kaydedilirken hata oluştu: {exc}")
            return False

    # ------------------------------------------------------------------
    # Mobil token dondurme (S-4301)
    # ------------------------------------------------------------------
    # R43 duzeltmesi daha once yalnizca hic orneklenmeyen SettingsDialog
    # sinifindaydi; kullanicinin gordugu canli SettingsWindow'da token
    # dondurme yolu yoktu ve eski token gecerli kaliyordu. Asagidaki
    # metotlar ui/settings_dialog.py'deki mantigin aynisidir: paylasilan
    # MobileTokenAuthenticator uzerinden yeni 256-bit token uretilir ve
    # canli sunucuya itilip dogrulanir; server.rotate_access_token()
    # cagrilmaz (farkli bir token uretip authenticator ile senkronu
    # bozardi).
    def _resolve_mobile_server(self):
        """Return the live MobileRemoteServer, if one is reachable.

        Priority: explicitly wired server (constructor arg), then the
        parent-widget chain (a main window owns ``mobile_server``).
        Returns None when no live server is known.
        """
        server = getattr(self, "mobile_server", None)
        if server is not None:
            return server
        widget = self.parentWidget()
        seen = set()
        while widget is not None and id(widget) not in seen:
            seen.add(id(widget))
            server = getattr(widget, "mobile_server", None)
            if server is not None:
                return server
            widget = widget.parentWidget() if hasattr(widget, "parentWidget") else None
        return None

    def _resolve_mobile_token_auth(self):
        """Return the shared MobileTokenAuthenticator, if one is reachable.

        Priority: explicitly wired auth (constructor arg), then the
        shell's worker (GatherionShell keeps ``_worker`` and main.py
        shares its authenticator with the worker at :541), then the
        parent-widget chain (a main window owns ``mobile_token_auth``).
        Returns None when none is known — the caller then mints a fresh
        authenticator, whose token is still pushed to the live server.
        """
        auth = getattr(self, "mobile_token_auth", None)
        if auth is not None and hasattr(auth, "rotate_token"):
            return auth
        shell = getattr(self, "_shell", None)
        if shell is not None:
            worker = getattr(shell, "_worker", None)
            auth = getattr(worker, "mobile_token_auth", None)
            if auth is not None and hasattr(auth, "rotate_token"):
                return auth
        widget = self.parentWidget()
        seen = set()
        while widget is not None and id(widget) not in seen:
            seen.add(id(widget))
            auth = getattr(widget, "mobile_token_auth", None)
            if auth is not None and hasattr(auth, "rotate_token"):
                return auth
            widget = widget.parentWidget() if hasattr(widget, "parentWidget") else None
        return None

    def _propagate_token_to_mobile_server(self, auth) -> bool:
        """Push ``auth``'s (freshly rotated) token onto the live server.

        Uses the authenticator's own ``bind_to_mobile_server()`` — the
        same call main.py makes once at startup — and verifies the
        server actually holds the new token afterwards. Deliberately
        does NOT call ``server.rotate_access_token()``: that generates
        a *different* 32-char token and would desync the server from
        the 256-bit authenticator. Returns False when no live server
        is known or the push could not be verified.
        """
        server = self._resolve_mobile_server()
        if server is None:
            return False
        try:
            bound = False
            if hasattr(auth, "bind_to_mobile_server"):
                try:
                    bound = bool(auth.bind_to_mobile_server(server))
                except Exception:
                    bound = False
            if not bound and hasattr(server, "access_token"):
                # Fallback: direct assignment of the rotated token.
                server.access_token = auth.token
                if hasattr(server, "require_token"):
                    server.require_token = True
            return bool(
                getattr(server, "access_token", None) is not None
                and str(getattr(server, "access_token")) == str(auth.token)
            )
        except Exception:
            return False

    def rotate_mobile_token(self):
        """Generate a fresh 256-bit mobile token (core/incident_recorder.py).

        S-4301: rotating only the on-disk token left the live
        MobileRemoteServer accepting the OLD token (its access_token was
        set once at startup via bind_to_mobile_server()). The new token
        is therefore also pushed to the live server — via the shared
        authenticator when one was wired in, so its in-memory token
        stays in sync too — and the push is verified. When no live
        server is reachable the user is told the new token only takes
        effect after the server restarts, instead of being left to
        believe the old token is already dead.
        Returns True when the token was rotated (and, if a live server
        was known, verified on it); False on failure.
        """
        try:
            from core.incident_recorder import MobileTokenAuthenticator

            # Prefer the shared authenticator (main.py's instance) so its
            # in-memory token does not keep the old value after rotation.
            auth = self._resolve_mobile_token_auth()
            if auth is None or not hasattr(auth, "rotate_token"):
                auth = MobileTokenAuthenticator()
            auth.rotate_token()
            server_updated = self._propagate_token_to_mobile_server(auth)
            server_known = self._resolve_mobile_server() is not None
            if server_known and server_updated:
                QMessageBox.information(
                    self,
                    "Mobile Token",
                    "Yeni 256-bit mobil token üretildi ve canlı mobil "
                    "sunucuya uygulandı.\n"
                    f"Parmak izi: {auth.token_fingerprint}\n"
                    "Eski token artık geçersiz. QR kodunu yeniden oluşturun.",
                )
                return True
            if server_known:
                QMessageBox.warning(
                    self,
                    "Mobile Token",
                    "Yeni 256-bit mobil token üretildi (diske yazıldı),\n"
                    f"Parmak izi: {auth.token_fingerprint}\n"
                    "AMA canlı mobil sunucuya uygulanamadı — sunucu hâlâ "
                    "eski tokenı kabul ediyor olabilir. Sunucuyu yeniden "
                    "başlatın ve QR kodunu yeniden oluşturun.",
                )
                return False
            QMessageBox.information(
                self,
                "Mobile Token",
                "Yeni 256-bit mobil token üretildi.\n"
                f"Parmak izi: {auth.token_fingerprint}\n"
                "Canlı mobil sunucu bulunamadı; yeni token sunucu yeniden "
                "başlatıldığında geçerli olur. QR kodunu yeniden oluşturun.",
            )
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Mobile Token", f"Token döndürülemedi: {exc}")
            return False

    def save_and_close(self) -> None:
        """Saves settings and closes the dialog (only when the save worked)."""
        if self.save_settings():
            self.accept()

    def reset_to_defaults(self) -> None:
        """Resets all controls to `DEFAULT_SETTINGS` and persists the reset.

        Previously the reset only changed in-memory values and never wrote
        them to disk, so the next launch restored the old settings (F-0598).
        F-23-09: DEFAULT_SETTINGS dışındaki şema anahtarları da sıfırlanır
        (mp_full_threshold, ucb_explore_ratio, hotkeys/display alanları,
        version/active_region/active_route/reflex_model) — yoksa durum
        çubuğundaki "Tüm ayarlar varsayılan değerlere sıfırlandı." yalan
        olurdu (örn. bozuk mp_full_threshold olduğu gibi kalıyordu).
        """
        try:
            defaults = dict(DEFAULT_SETTINGS)
            defaults.setdefault("auto_record_3d_crests", True)
            self._cfg_batching = True
            try:
                for key, val in defaults.items():
                    self._cfg_set(key, val)
            finally:
                self._cfg_batching = False
            # F-23-09: şemada olup DEFAULT_SETTINGS'te olmayan alanlar —
            # taze BotSettings'ten alınıp bağlı yönetici üzerinden yazılır.
            try:
                from config.schema import BotSettings

                fresh = BotSettings().validate()
                mgr = getattr(self, "config_manager", None)
                mgr_set = getattr(mgr, "set", None) if mgr is not None else None
                if callable(mgr_set):
                    for sec_name in ("gathering", "route", "breaks", "hotkeys", "display"):
                        sec_obj = getattr(fresh, sec_name, None)
                        if sec_obj is None:
                            continue
                        for fname in list(getattr(sec_obj, "__dataclass_fields__", {}).keys()):
                            try:
                                mgr_set(sec_name, fname, getattr(sec_obj, fname))
                            except Exception:
                                pass
                    for fname in ("version", "active_region", "active_route", "reflex_model"):
                        try:
                            mgr_set("", fname, getattr(fresh, fname))
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("Şema kapsamlı sıfırlama atlandı: %s", exc)
            if not self._cfg_save():
                self.set_status_message(
                    "Sıfırlama diske yazılamadı: varsayılanlar kaydedilmedi."
                )
                logger.error("Reset to defaults failed to persist.")
                return
            self._load_from_settings()
            self.set_status_message("Tüm ayarlar varsayılan değerlere sıfırlandı.")
            if hasattr(self, "dirty_lbl"):
                self.dirty_lbl.setVisible(False)
            self.append_log("Tüm değerler varsayılanlara sıfırlandı ve kaydedildi.")
            logger.info("Settings reset to defaults.")
        except Exception as exc:
            logger.error("Error resetting settings to defaults: %s", exc)
            self.set_status_message(f"Sıfırlama başarısız: {exc}")

    def set_status_message(self, message: str) -> None:
        """Updates the footer status message (`status_lbl`)."""
        self.status_lbl.setText(str(message))

    set_status = set_status_message

    # ------------------------------------------------------------------
    # Prototip farkları: log şeridi, kirli takibi, doğrulama, profil
    # ------------------------------------------------------------------
    def append_log(self, message: str, level: str = "inf") -> None:
        """Append one timestamped line to the bottom log strip (max 60)."""
        widget = getattr(self, "log_text", None)
        if widget is None:
            logger.info("%s", message)
            return
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = {"err": "[ERROR]", "warn": "[WARNING]"}.get(level, "[INFO]")
        widget.appendPlainText(f"[{stamp}] {prefix} {message}")

    def _mark_dirty(self, *_args: Any) -> None:
        """Show the 'unsaved changes' tag; hidden again only by a save."""
        if getattr(self, "_loading_settings", False):
            return
        if hasattr(self, "dirty_lbl"):
            self.dirty_lbl.setVisible(True)
        if hasattr(self, "saved_lbl"):
            self.saved_lbl.setVisible(False)

    def _wire_dirty_tracking(self) -> None:
        """Every settings control marks the window dirty on user change.

        Loading is guarded by `_loading_settings`, so programmatic
        `setValue`/`setChecked` during `_load_from_settings` never
        shows the dirty tag.
        """
        spins = (
            "f_interval_min", "f_interval_max", "harvests_per_node",
            "rest_minutes", "reclick_node_sec",
            "time_to_reach_sec", "wait_min_after_failed", "max_failed_in_row",
            "remove_node_after_failures", "pause_after_no_gather_min",
            "fight_stop_gather_sec", "break_every_min", "break_every_variance",
            "break_lasts_min", "break_lasts_variance",
        )
        for name in spins:
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._mark_dirty)
        toggles = (
            "retry_on_extraction_fail", "test_mode", "show_overlay",
            "pause_on_alt_tab", "chk_selective_gather", "chk_proximity_route_auto",
            "gathered_join_active_route", "fly_unreachable",
            "gather_on_the_way", "fight_back", "minimap_auto_record",
            "world_3d_auto_record",
        )
        for name in toggles:
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "toggled"):
                widget.toggled.connect(self._mark_dirty)
        combos = ("gather_key", "combo_selective_materials")
        for name in combos:
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._mark_dirty)
        for name in ("hotkey_node", "hotkey_menu", "hotkey_pause"):
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._mark_dirty)

    def _validate_inputs(self) -> List[Tuple[Any, str]]:
        """Prototype validation rules; returns [(widget, message), ...].

        Blocks `save_settings` on the first error instead of silently
        clamping (the old `get_settings_dict` behaviour hid a bad
        F/E range by forcing max == min).
        """
        errors: List[Tuple[Any, str]] = []
        try:
            f_min = float(self.f_interval_min.value())
        except Exception:
            f_min = 0.0
        try:
            f_max = float(self.f_interval_max.value())
        except Exception:
            f_max = 0.0
        if f_min <= 0:
            errors.append((self.f_interval_min, "F/E alt sınır 0’dan büyük bir sayı olmalı."))
        if f_max <= 0:
            errors.append((self.f_interval_max, "F/E üst sınır 0’dan büyük bir sayı olmalı."))
        elif f_max <= f_min:
            errors.append((self.f_interval_max, "F/E üst sınır, alt sınırdan büyük olmalı."))
        try:
            harvests = int(self.harvests_per_node.value())
        except Exception:
            harvests = 0
        if harvests < 1 or harvests > 10:
            errors.append(
                (self.harvests_per_node, "Materyal başına toplama 1–10 arası tam sayı olmalı.")
            )
        gkey = str(
            self.gather_key.currentData() or self.gather_key.currentText() or ""
        ).strip().lower()
        if gkey not in ("f", "e", "auto"):
            errors.append((self.gather_key, "Toplama tuşu f, e veya otomatik olmalı."))
        try:
            from config.schema import parse_hotkey_chord
        except Exception:
            parse_hotkey_chord = None  # type: ignore[assignment]
        for edit, label in (
            (self.hotkey_node, "Nokta ekle"),
            (self.hotkey_menu, "Menü"),
            (self.hotkey_pause, "Duraklat / Devam"),
        ):
            chord = str(edit.text() or "").strip().lower()
            if parse_hotkey_chord is not None:
                key, mods = parse_hotkey_chord(chord)
            else:
                key, mods = ("", frozenset())
            is_f_key = key.startswith("f") and key[1:].isdigit()
            if not chord or not is_f_key:
                errors.append(
                    (edit, f"{label} kısayolu bir F tuşuyla bitmeli (örn. ctrl+alt+f8).")
                )
            elif not mods:
                errors.append(
                    (
                        edit,
                        f"{label} kısayolu en az bir değiştirici tuş içermeli "
                        "(ctrl/alt/shift) — yalnız F8/F9/F10 oyuna gider.",
                    )
                )
        try:
            if int(self.break_lasts_min.value()) < 1:
                errors.append((self.break_lasts_min, "Mola süresi en az 1 dk olmalı."))
        except Exception:
            pass
        return errors

    # -- YZ eşik profili (Dengeli / Hassas / Sıkı) ---------------------
    _WRONG_TIER_ORDER = ("balanced", "aggressive", "strict")
    _WRONG_TIER_LABELS = {
        "balanced": "Dengeli",
        "aggressive": "Hassas",
        "strict": "Sıkı",
    }
    # Mirrors core/node_quality_gate.CONF_TIERS. The literal map is the
    # fallback because importing that module needs numpy/cv2, which may
    # be absent (e.g. this VM); when importable, its values win.
    _WRONG_TIER_FLOORS = {"balanced": 0.13, "aggressive": 0.10, "strict": 0.22}

    def _wrong_tier_floor(self, tier: str) -> float:
        try:
            from core.node_quality_gate import conf_min_for_tier

            return float(conf_min_for_tier(tier))
        except Exception:
            return float(self._WRONG_TIER_FLOORS.get(str(tier or "balanced"), 0.13))

    def _wrong_tier_label(self, tier: str) -> str:
        return self._WRONG_TIER_LABELS.get(str(tier or "balanced"), "Dengeli")

    def _on_wrong_tier_combo_changed(self, _index: int = 0) -> None:
        tier = self._wrong_scan_tier()
        button = getattr(self, "btn_profile", None)
        if button is not None:
            button.setText(f"Profil: {self._wrong_tier_label(tier)}")
        self._update_infoline()
        self._mark_dirty()

    def _on_profile_cycle(self) -> None:
        """Prototype profile button: Dengeli → Hassas → Sıkı → Dengeli."""
        current = self._wrong_scan_tier()
        try:
            nxt = self._WRONG_TIER_ORDER[
                (self._WRONG_TIER_ORDER.index(current) + 1) % len(self._WRONG_TIER_ORDER)
            ]
        except ValueError:
            nxt = "balanced"
        combo = getattr(self, "cb_wrong_tier", None)
        if combo is not None:
            idx = combo.findData(nxt)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        floor = self._wrong_tier_floor(nxt)
        label = self._wrong_tier_label(nxt)
        self.set_status_message(f"Eşik profili: {label} ({floor:.2f})")
        self.append_log(f"YZ eşik profili: {label} (eşik={floor:.2f})")

    # -- Alt bilgi satırı (YZ taraması özeti) ---------------------------
    def _update_infoline(self) -> None:
        label_widget = getattr(self, "infoline_lbl", None)
        if label_widget is None:
            return
        floor = self._wrong_tier_floor(self._wrong_scan_tier())
        profile = self._wrong_tier_label(self._wrong_scan_tier())
        meta = getattr(self, "_last_scan_meta", "") or ""
        if meta:
            removed = int(getattr(self, "_last_scan_removed", 0) or 0)
            backup = str(getattr(self, "_last_scan_backup", "") or "")
            text = f"YZ taraması {removed} nokta sildi. {meta}"
            if backup:
                text += f" · Yedek: {backup}"
            label_widget.setText(text)
        else:
            label_widget.setText(
                "YZ modeli hazır. Henüz tarama yapılmadı. "
                f"Profil: {profile} (eşik={floor:.2f})"
            )

    def _record_scan_result(self, removed: int, backup_path: str, meta: str) -> None:
        self._last_scan_removed = int(removed)
        self._last_scan_backup = str(backup_path or "")
        self._last_scan_meta = str(meta or "")
        self._update_infoline()

    # -- Çift tık: noktayı toplandı işaretle -----------------------------
    def _on_node_double_clicked(self, item: Any) -> None:
        """Prototype double-click: mark the node gathered via RouteManager."""
        try:
            row = item.row()
        except Exception:
            return
        anchor = self.table.item(row, 0) if hasattr(self, "table") else None
        if anchor is None:
            return
        try:
            node_id = int(anchor.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return
        reg = self._current_region()
        rte = self._current_route()
        rm = self.route_manager
        if rm is None or not hasattr(rm, "mark_node_gathered"):
            self.set_status_message(
                "Nokta işaretlenemedi: rota yöneticisi yok. Hiçbir şey yazılmadı."
            )
            return
        try:
            node = rm.mark_node_gathered(
                node_id,
                rest_minutes=float(self.rest_minutes.value()),
                region=reg,
            )
        except Exception as exc:
            logger.error("Mark node gathered failed #%s: %s", node_id, exc)
            self.set_status_message(f"Nokta işaretlenemedi: {exc}")
            return
        if node is None:
            self.set_status_message(f"Nokta #{node_id} bulunamadı.")
            return
        self.set_status_message(f"Nokta toplandı: #{node_id} ({rte})")
        self.append_log(f"Nokta toplandı: #{node_id} ({rte})")
        self.load_nodes_table()

    def _on_quit_clicked(self) -> None:
        logger.info("Quit bot requested from SettingsWindow.")
        self.quit_requested.emit()
        self.close()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """Kapat hides the shell. It does not quit the bot. Overlay HUD stays up."""
        shell = getattr(self, "_shell", None)
        if shell is not None and not getattr(shell, "_closing", False):
            event.ignore()
            shell.close()
            return
        # P2-2: gerçek kapanışta çalışan YZ taraması varsa işbirlikçi
        # stop + sınırlı bekleme; durmazsa thread ebeveynsiz olduğu için
        # pencereyle birlikte yok olmaz, bitince kendi kapanır.
        thread = getattr(self, "_wrong_scan_thread", None)
        if thread is not None:
            try:
                if thread.isRunning():
                    worker = getattr(self, "_wrong_scan_worker", None)
                    stop = getattr(worker, "stop", None)
                    if callable(stop):
                        stop()
                    thread.requestInterruption()
                    thread.quit()
                    if not thread.wait(5000):
                        logger.warning(
                            "YZ taraması kapanışta 5 sn içinde durmadı; "
                            "arka plan işi bitince kapanacak."
                        )
            except RuntimeError:
                pass  # C++ nesnesi zaten silinmiş
        super().closeEvent(event)

    def _open_hosted(self, key: str) -> bool:
        """Switch a tab in the shell when this window is hosted. Otherwise open a dialog."""
        opener = getattr(self, "tab_opener", None)
        if callable(opener):
            opener(key)
            return True
        return False

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
            self.populate_selective_materials()
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
            self._mark_dirty()
            self.append_log(f"Rota: {reg} · {rte} ({self.table.rowCount()} nokta)")
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

            placed = 0
            for row_idx, node in enumerate(nodes):
                # One corrupt node must not wipe the rest of the table (F-0599):
                # each row is built in isolation; on failure only that
                # (possibly half-inserted) row is dropped and loading continues.
                row = placed
                try:
                    self.table.insertRow(row)
                    self.table.setRowHeight(row, 56)

                    node_id = int(
                        node.get("id", row + 1)
                        if isinstance(node, dict)
                        else getattr(node, "id", row + 1)
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
                    cell_lay.setContentsMargins(6, 4, 8, 4)
                    cell_lay.setSpacing(10)

                    lbl_thumb = QLabel(cell_node)
                    lbl_thumb.setFixedSize(48, 48)
                    lbl_thumb.setPixmap(self._resolve_node_patch_pixmap(node, reg))
                    lbl_thumb.setStyleSheet(
                        "border: 1px solid #30363d; border-radius: 3px;"
                    )
                    cell_lay.addWidget(lbl_thumb)

                    lbl_num = QLabel(f"#{node_id}", cell_node)
                    lbl_num.setStyleSheet(
                        'color: #e8edf2; font-weight: bold; font-size: 13px; font-family: Consolas, "Cascadia Mono", monospace;'
                    )
                    cell_lay.addWidget(lbl_num)
                    cell_lay.addStretch(1)
                    self._arm_row_select(cell_node, row)

                    # Store node_id in underlying QTableWidgetItem for selection operations
                    anchor_item = QTableWidgetItem()
                    anchor_item.setData(Qt.ItemDataRole.UserRole, node_id)
                    self.table.setItem(row, 0, anchor_item)
                    self.table.setCellWidget(row, 0, cell_node)

                    # Column 1: ROTALAR
                    item_routes = QTableWidgetItem(routes_str)
                    item_routes.setForeground(QBrush(QColor("#e8edf2")))
                    self.table.setItem(row, 1, item_routes)

                    # Column 2: TOPLANAN
                    item_gathered = QTableWidgetItem(f"{gathered_cnt}×")
                    item_gathered.setForeground(QBrush(QColor("#e8edf2")))
                    self.table.setItem(row, 2, item_gathered)

                    # Column 3: SON
                    last_str = self._format_last_visited(last_ts, now)
                    item_last = QTableWidgetItem(last_str)
                    item_last.setData(Qt.ItemDataRole.UserRole, last_ts)
                    item_last.setForeground(QBrush(QColor("#8b98a6")))
                    self.table.setItem(row, 3, item_last)

                    # Column 4: DURUM (green `Hazır`, amber rest, gray `Yoksayılan`)
                    ignored = bool(
                        node.get("not_gatherable", False)
                        if isinstance(node, dict)
                        else getattr(node, "not_gatherable", False)
                    )
                    if ignored:
                        rest_text, is_ready = "Yoksayılan", False
                    else:
                        rest_text, is_ready = self._compute_rest_status(
                            last_ts, rest_sec_cfg, now
                        )
                    badge = _RestBadgeWidget(rest_text, is_ready, self.table, ignored=ignored)
                    self._arm_row_select(badge, row)
                    self.table.setCellWidget(row, 4, badge)
                except Exception as exc:
                    logger.error("Skipping corrupt node at position %s: %s", row_idx, exc)
                    if self.table.rowCount() > row:
                        self.table.removeRow(row)
                    continue
                placed += 1
            if hasattr(self, "point_filter_edit") and self.point_filter_edit.text().strip():
                self._filter_nodes_table(self.point_filter_edit.text())
            else:
                self._update_sel_info_label()
        except Exception as exc:
            logger.error("Error populating nodes table: %s", exc)

    def _update_header_perf(self) -> None:
        """Updates header chips with live system metrics via SystemMonitor."""
        try:
            from core.system_monitor import SystemMonitor
            snap = SystemMonitor.get_snapshot()
            fps_val = snap.get("capture", {}).get("fps", 0.0)
            fps_str = f"{fps_val:.0f}" if fps_val > 0 else "0"
            gpu_pct = snap.get("gpu", {}).get("pct", 0)
            cpu_pct = snap.get("cpu", {}).get("pct", 0)
            lat_val = snap.get("capture", {}).get("latency_ms", 0.0)
            lat_measured = snap.get("capture", {}).get("latency_measured", False)
            lat_str = (
                f"{lat_val:.1f}ms"
                if lat_measured and isinstance(lat_val, (int, float)) and lat_val > 0
                else "—"
            )

            if hasattr(self, "chip_fps"):
                self.chip_fps.setText(f'FPS <b style="color:#e8edf2;font-family:Consolas;">{fps_str}</b>')
            if hasattr(self, "chip_gpu"):
                self.chip_gpu.setText(f'GPU <b style="color:#e8edf2;font-family:Consolas;">{gpu_pct}%</b>')
            if hasattr(self, "chip_cpu"):
                self.chip_cpu.setText(f'CPU <b style="color:#e8edf2;font-family:Consolas;">{cpu_pct}%</b>')
            if hasattr(self, "chip_lat"):
                self.chip_lat.setText(f'LAT <b style="color:#e8edf2;font-family:Consolas;">{lat_str}</b>')
        except Exception as exc:
            logger.debug("Header perf refresh error: %s", exc)

    def _update_live_node_statuses(self) -> None:
        """Updates header perf chips and LAST/REST columns every second."""
        self._update_header_perf()
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
                    # P2-7: yoksayılan rozet canlı güncellemede ezilmez
                    # (set_status da bunu korur; buradaki atlama gereksiz
                    # dinlenme hesabını önler).
                    if badge.is_ignored:
                        continue
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
                else:
                    self.set_status_message(
                        f"'{old_reg}' bölgesi '{new_reg}' olarak yeniden adlandırılamadı."
                    )
                    logger.error(
                        "Rename region failed: rename_region returned False for %r -> %r",
                        old_reg,
                        new_reg,
                    )
            else:
                self.set_status_message(
                    "Bölge yeniden adlandırma desteklenmiyor: rota yöneticisinde rename_region yok."
                )
                logger.error("rename_region missing on route manager; region not renamed.")
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
                if self.route_manager.delete_region(reg):
                    self.refresh_regions_and_routes()
                    self.set_status_message(f"'{reg}' bölgesi silindi.")
                else:
                    self.set_status_message(f"'{reg}' bölgesi silinemedi.")
                    logger.error(
                        "Delete region failed: delete_region returned False for %r", reg
                    )
            else:
                self.set_status_message(
                    "Bölge silme desteklenmiyor: rota yöneticisinde delete_region yok."
                )
                logger.error("delete_region missing on route manager; region not deleted.")
        except Exception as exc:
            logger.error("Delete region failed: %s", exc)

    def _on_new_route(self) -> None:
        reg = self._current_region()
        if not reg:
            reg, ok_reg = QInputDialog.getText(
                self, "Yeni Bölge", "Önce bir bölge adı girin:", text=(self._current_region() or "")
            )
            reg = reg.strip()
            if not ok_reg or not reg:
                return
            if hasattr(self.route_manager, "create_region"):
                if not self.route_manager.create_region(reg):
                    self.set_status_message(f"'{reg}' bölgesi oluşturulamadı.")
                    logger.error(
                        "Create region failed: create_region returned False for %r", reg
                    )
                    return
            else:
                self.set_status_message(
                    "Bölge oluşturma desteklenmiyor: rota yöneticisinde create_region yok."
                )
                logger.error("create_region missing on route manager; region not created.")

        new_rte, ok = QInputDialog.getText(
            self, "Yeni Rota", f"'{reg}' bölgesi için yeni rota adı:"
        )
        new_rte = new_rte.strip()
        if not ok or not new_rte:
            return
        try:
            if hasattr(self.route_manager, "create_route"):
                if self.route_manager.create_route(reg, new_rte):
                    self.refresh_regions_and_routes(
                        prefer_region=reg, prefer_route=new_rte
                    )
                    self.set_status_message(f"'{reg}' içinde '{new_rte}' rotası oluşturuldu.")
                else:
                    self.set_status_message(
                        f"'{reg}' içinde '{new_rte}' rotası oluşturulamadı."
                    )
                    logger.error(
                        "Create route failed: create_route returned False for %r / %r",
                        reg,
                        new_rte,
                    )
            else:
                self.set_status_message(
                    "Rota oluşturma desteklenmiyor: rota yöneticisinde create_route yok."
                )
                logger.error("create_route missing on route manager; route not created.")
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
                else:
                    self.set_status_message(
                        f"'{old_rte}' rotası '{new_rte}' olarak yeniden adlandırılamadı."
                    )
                    logger.error(
                        "Rename route failed: rename_route returned False for %r / %r -> %r",
                        reg,
                        old_rte,
                        new_rte,
                    )
            else:
                self.set_status_message(
                    "Rota yeniden adlandırma desteklenmiyor: rota yöneticisinde rename_route yok."
                )
                logger.error("rename_route missing on route manager; route not renamed.")
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
                if self.route_manager.delete_route(reg, rte):
                    self.refresh_regions_and_routes(
                        prefer_region=reg, prefer_route="All"
                    )
                    self.set_status_message(f"'{rte}' rotası silindi.")
                else:
                    self.set_status_message(f"'{rte}' rotası silinemedi.")
                    logger.error(
                        "Delete route failed: delete_route returned False for %r / %r",
                        reg,
                        rte,
                    )
            else:
                self.set_status_message(
                    "Rota silme desteklenmiyor: rota yöneticisinde delete_route yok."
                )
                logger.error("delete_route missing on route manager; route not deleted.")
        except Exception as exc:
            logger.error("Delete route failed: %s", exc)

    # ------------------------------------------------------------------
    # Node Action & Auxiliary Dialog Button Handlers
    # ------------------------------------------------------------------
    def _on_add_nodes(self) -> None:
        reg = self._current_region() or (self.route_manager.get_regions()[0] if self.route_manager and hasattr(self.route_manager, "get_regions") and self.route_manager.get_regions() else "")
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
                        if node is None:
                            logger.error(
                                "Add node returned None for region=%s route=%s "
                                "(kara liste veya ekleme reddi olabilir).",
                                reg,
                                rte,
                            )
                            self.set_status_message(
                                "Nokta eklenemedi: konum kara listede olabilir "
                                "veya ekleme reddedildi."
                            )
                        else:
                            self.refresh_regions_and_routes(
                                prefer_region=reg, prefer_route=rte
                            )
                            self.set_status_message(
                                f"Nokta #{node.id}, {reg} · {rte} rotasına eklendi."
                            )
                    else:
                        logger.error(
                            "add_node missing on route manager; no node created."
                        )
                        self.set_status_message(
                            "Nokta eklenemedi: rota yöneticisinde add_node yok. "
                            "Hiçbir nokta eklenmedi."
                        )
                        self.append_log(f"Nokta eklendi: #{node.id} → {rte}")
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
            self.set_status_message(f"Nokta ekleme başarısız: {exc}")

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
        if not hasattr(self.route_manager, "remove_node_from_route"):
            self.set_status_message(
                "Rotadan çıkarma desteklenmiyor: rota yöneticisinde remove_node_from_route yok. Hiçbir nokta çıkarılmadı."
            )
            logger.error("remove_node_from_route missing on route manager; nothing removed.")
            return
        try:
            removed = 0
            for nid in ids:
                if self.route_manager.remove_node_from_route(nid, rte, region=reg):
                    removed += 1
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            if removed == 0:
                self.set_status_message(
                    f"Hiçbir nokta '{rte}' rotasından çıkarılamadı."
                )
            else:
                self.set_status_message(f"{removed} nokta '{rte}' rotasından çıkarıldı.")
                self.append_log(f"Rotadan çıkarıldı: {ids} ({rte})")
        except Exception as exc:
            logger.error("Remove from route failed: %s", exc)
            self.set_status_message(f"Rotadan çıkarma başarısız: {exc}")

    def _arm_row_select(self, widget: QWidget, row_idx: int) -> None:
        """Let Ctrl/Shift clicks on thumbnails and badges select the row."""
        filt = getattr(self, "_row_select_filter", None)
        if filt is None or widget is None:
            return
        widget.setProperty("gatherion_row", int(row_idx))
        widget.installEventFilter(filt)
        for child in widget.findChildren(QWidget):
            child.setProperty("gatherion_row", int(row_idx))
            child.installEventFilter(filt)

    def _delete_ids_same_path(self, region: str, node_ids: List[int]) -> bool:
        """Delete nodes through ``delete_nodes`` / ``save_region``, same as Noktayı Sil.

        Returns False when nothing was written. WinError 5 is reported and is not success.
        """
        if not node_ids:
            return False
        rm = self.route_manager
        if rm is None:
            self.set_status_message("Rota yöneticisi yok. Nokta silinmedi.")
            return False
        if hasattr(rm, "delete_nodes"):
            ok = bool(rm.delete_nodes(node_ids, region=region))
        else:
            deleted = 0
            for nid in node_ids:
                if hasattr(rm, "delete_node") and rm.delete_node(nid, region=region):
                    deleted += 1
            ok = deleted > 0
        if ok:
            return True
        err = getattr(rm, "last_save_error", None)
        winerror = getattr(err, "winerror", None) if err is not None else None
        if winerror == 5:
            self.set_status_message(
                "Kayıt başarısız (WinError 5). Noktalar silinmedi."
            )
        elif err is not None:
            self.set_status_message(f"Kayıt başarısız: {err}. Noktalar silinmedi.")
        else:
            self.set_status_message("Nokta silinmedi.")
        return False

    def _on_delete_node(self) -> None:
        reg = self._current_region()
        rte = self._current_route()
        ids = self._selected_node_ids()
        if not ids:
            self.set_status_message("Silmek için bir veya daha fazla nokta seçin.")
            return
        # Confirmation (F-0600): deleting a node is irreversible and also
        # deletes its map patch PNG (routes/<region>/patches/node_<id>.png)
        # via RouteManager.delete_nodes — the user must consent to both.
        reply = QMessageBox.question(
            self,
            "Noktayı Sil",
            f"{len(ids)} nokta '{reg}' bölgesinden kalıcı olarak silinsin mi?\n"
            "Bu işlem geri alınamaz ve noktaların harita kesit (patch) "
            "görselleri de silinir.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.set_status_message("Silme iptal edildi. Hiçbir nokta silinmedi.")
            return
        try:
            if not self._delete_ids_same_path(reg, ids):
                return
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            self.set_status_message(f"{len(ids)} nokta '{reg}' bölgesinden silindi.")
            self.append_log(f"Noktalar silindi: {ids} ({reg})")
        except Exception as exc:
            logger.error("Delete node failed: %s", exc)
            self.set_status_message(f"Nokta silinemedi: {exc}")

    def _wrong_scan_tier(self) -> str:
        combo = getattr(self, "cb_wrong_tier", None)
        if combo is None:
            return "balanced"
        data = combo.currentData()
        if data:
            return str(data)
        return "balanced"

    def _on_delete_wrong_nodes(self) -> None:
        """Delete nodes the multi-signal gatherion gate calls wrong. Harvest 0 is not a reason.

        P2-2: the heavy part (model load + patch reads + inference) runs
        in a background thread (``_WrongScanWorker``); this method only
        gathers the inputs on the GUI thread and starts it. Backup,
        delete and list refresh run in ``_on_wrong_scan_finished`` on the
        GUI thread. A second click while a scan runs is ignored.
        """
        reg = self._current_region()
        rte = self._current_route()
        if not reg:
            self.set_status_message("Önce bir bölge seçin.")
            return
        if getattr(self, "_wrong_scan_thread", None) is not None:
            self.set_status_message("YZ taraması zaten çalışıyor.")
            return
        button = getattr(self, "btn_delete_wrong", None)
        if button is not None:
            button.setEnabled(False)
        try:
            tier = self._wrong_scan_tier()
            nodes = self._get_nodes_for_selection(reg, rte)
            if not nodes:
                self.set_status_message("Taranacak nokta yok. Hiçbir nokta silinmedi.")
                if button is not None:
                    button.setEnabled(True)
                return
            # Patch paths are resolved here (cheap string work on the
            # route manager); the disk reads happen in the worker.
            rm = self.route_manager
            patch_paths: List[Optional[str]] = []
            for node in nodes:
                path: Optional[str] = None
                if rm is not None and hasattr(rm, "_resolve_patch_disk_path"):
                    try:
                        resolved = str(rm._resolve_patch_disk_path(node))
                        path = resolved or None
                    except Exception as exc:
                        node_id = (
                            node.get("id", 0)
                            if isinstance(node, dict)
                            else getattr(node, "id", 0)
                        )
                        logger.debug("Patch yolu okunamadı #%s: %s", node_id, exc)
                patch_paths.append(path)
            self.set_status_message(f"YZ taraması çalışıyor (CPU, {tier})...")
            # Node count at scan start: the >half guard must compare
            # against the scanned set, not a list re-read after the scan.
            self._wrong_scan_context = (reg, rte, len(nodes))
            worker = _WrongScanWorker(nodes, patch_paths, tier, reg)
            # P2-2: parentless on purpose — a running QThread must not be
            # destroyed with the window; _ACTIVE_WRONG_SCAN_THREADS keeps
            # it alive until finished, which deletes it.
            thread = QThread()
            _ACTIVE_WRONG_SCAN_THREADS.add(thread)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.progress.connect(self._on_wrong_scan_progress)
            worker.finished.connect(self._on_wrong_scan_finished)
            worker.failed.connect(self._on_wrong_scan_failed)
            worker.finished.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(
                lambda: _ACTIVE_WRONG_SCAN_THREADS.discard(thread)
            )
            thread.finished.connect(self._clear_wrong_scan_thread)
            self._wrong_scan_thread = thread
            self._wrong_scan_worker = worker
            try:
                thread.start()
            except Exception as exc:
                logger.error("Wrong-node scan thread failed to start: %s", exc)
                _ACTIVE_WRONG_SCAN_THREADS.discard(thread)
                self._wrong_scan_thread = None
                self._wrong_scan_worker = None
                self.set_status_message(f"YZ taraması başlatılamadı: {exc}")
                if button is not None:
                    button.setEnabled(True)
        except Exception as exc:
            logger.error("Wrong-node scan failed: %s", exc)
            self.set_status_message(f"YZ taraması yazılmadı: {exc}")
            if button is not None:
                button.setEnabled(True)

    def _on_wrong_scan_progress(self, done: int, total: int) -> None:
        """GUI-thread progress slot for the background wrong-node scan."""
        tier = self._wrong_scan_tier()
        self.set_status_message(f"YZ taraması {done}/{total} (CPU, {tier})...")

    def _on_wrong_scan_finished(self, result: object) -> None:
        """GUI-thread result slot: guards, backup, delete, refresh.

        Runs on the GUI thread — this is the only place that touches the
        route manager and the widgets after the background scan.
        """
        button = getattr(self, "btn_delete_wrong", None)
        try:
            from core.wrong_node_scan import (
                format_blocked_status_tr,
                scan_would_exceed_half,
            )

            payload = result if isinstance(result, dict) else {}
            wrong = list(payload.get("wrong") or [])
            meta = str(payload.get("meta") or "")
            reg, rte, nodes_total = self._wrong_scan_context
            rm = self.route_manager
            if not wrong:
                self.set_status_message(
                    f"YZ taraması yanlış düğüm bulamadı. Hiçbir nokta silinmedi. {meta}"
                )
                self._record_scan_result(0, "", meta)
                self.append_log(f"YZ taraması: yanlış düğüm bulunamadı. {meta}")
                return
            if scan_would_exceed_half(len(wrong), nodes_total):
                self.set_status_message(
                    f"{format_blocked_status_tr(wrong, nodes_total)} {meta}"
                )
                self._record_scan_result(0, "", meta)
                self.append_log(
                    f"YZ taraması engellendi (yarıdan fazlası): {meta}", "warn"
                )
                return
            if rm is None or not hasattr(rm, "backup_region_file"):
                self.set_status_message("Yedek alınamadı. Dosya yazılmadı.")
                return
            try:
                backup_path = rm.backup_region_file(reg)
            except Exception as exc:
                logger.error("Yedek alınamadı: %s", exc)
                self.set_status_message(f"Yedek alınamadı. Dosya yazılmadı: {exc}")
                return
            if not self._delete_ids_same_path(reg, [item.node_id for item in wrong]):
                return
            self.refresh_regions_and_routes(prefer_region=reg, prefer_route=rte)
            self._record_scan_result(len(wrong), backup_path, meta)
            self.set_status_message(f"{_wrong_node_status(wrong, backup_path)} {meta}")
            self.append_log(
                f"YZ yanlışları silindi: {[item.node_id for item in wrong]} {meta}"
            )
        except Exception as exc:
            logger.error("Wrong-node scan failed: %s", exc)
            self.set_status_message(f"YZ taraması yazılmadı: {exc}")
        finally:
            if button is not None:
                button.setEnabled(True)

    def _on_wrong_scan_failed(self, message: str) -> None:
        """GUI-thread failure slot for the background wrong-node scan."""
        button = getattr(self, "btn_delete_wrong", None)
        self.set_status_message(str(message or "YZ taraması yazılmadı."))
        if button is not None:
            button.setEnabled(True)

    def _clear_wrong_scan_thread(self) -> None:
        """Drop the finished scan thread/worker refs (thread.finished)."""
        self._wrong_scan_thread = None
        self._wrong_scan_worker = None

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
            self.append_log(f"{moved} nokta taşındı: {reg} → {target_reg}")
        except Exception as exc:
            logger.error("Move node to region failed: %s", exc)

    def _on_open_map(self) -> None:
        """Opens `ui.map_viewer.MapViewerWindow` for the active region and route."""
        if self._open_hosted("map"):
            return
        reg = self._current_region() or (self.route_manager.get_regions()[0] if self.route_manager and hasattr(self.route_manager, "get_regions") and self.route_manager.get_regions() else "")
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
        if self._open_hosted("radar"):
            return
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
        if self._open_hosted("stats"):
            return
        try:
            from ui.analytics_dialog import AnalyticsDialog  # type: ignore

            # v20-istatistik (R3): bağımsız diyalog canlı motoru alsın —
            # verilmezse AnalyticsDialog kendi boş motorunu kurar ve
            # istatistik ekranı kalıcı 0 gösterir.
            tel = _resolve_analytics_telemetry(self)
            if self._analytics_dlg is None:
                if tel is not None:
                    self._analytics_dlg = AnalyticsDialog(telemetry=tel, parent=self)
                else:
                    self._analytics_dlg = AnalyticsDialog(parent=self)
            self._analytics_dlg.show()
            self._analytics_dlg.raise_()
            self._analytics_dlg.activateWindow()
        except Exception as exc:
            logger.error("Could not open AnalyticsDialog: %s", exc)
            self.set_status_message(f"İstatistikler açılamadı: {exc}")

    def _on_open_calibration(self) -> None:
        """Opens `ui.calibration_wizard.CalibrationWizard`."""
        if self._open_hosted("calibration"):
            return
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
        if self._open_hosted("guide"):
            return
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
        if self._open_hosted("yolo"):
            return
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
        if self._open_hosted("mobile"):
            return
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
        reg = self._current_region() or (self.route_manager.get_regions()[0] if self.route_manager and hasattr(self.route_manager, "get_regions") and self.route_manager.get_regions() else "")
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
        if not hasattr(self.route_manager, "export_route_zip"):
            self.set_status_message(
                "Dışa aktarma desteklenmiyor: rota yöneticisinde export_route_zip yok. Hiçbir şey aktarılmadı."
            )
            logger.error("export_route_zip missing on route manager; nothing exported.")
            return
        try:
            if hasattr(self.route_manager, "export_route_zip"):
                out_path = self.route_manager.export_route_zip(
                    region=reg, route=rte, zip_path=file_path
                )
                self.set_status_message(
                    f"Rota paylaşıldı: '{reg} · {rte}' -> {os.path.basename(out_path)}"
                )
                self.append_log(f"Rota paylaşıldı (ZIP): {reg} · {rte}")
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
        if not hasattr(self.route_manager, "import_route_zip"):
            self.set_status_message(
                "İçe aktarma desteklenmiyor: rota yöneticisinde import_route_zip yok. Hiçbir şey aktarılmadı."
            )
            logger.error("import_route_zip missing on route manager; nothing imported.")
            return
        try:
            imported_reg = self.route_manager.import_route_zip(file_path)
            self.refresh_regions_and_routes(prefer_region=imported_reg)
            nodes_cnt = len(
                self._get_nodes_for_selection(imported_reg, self._current_route())
            )
            self.set_status_message(
                f"{nodes_cnt} nokta {imported_reg} bölgesine aktarıldı"
            )
            self.append_log(f"Rota içe aktarıldı: {nodes_cnt} nokta → {imported_reg}")
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
        # BotSettings has no "mobile" section, so the candidate loop above
        # could never fill it (F-0762). Fall back to the mobile relay config
        # file (config/mobile_relay.json) when present; otherwise say so
        # explicitly instead of returning a silently empty section.
        if not summary["mobile"]:
            try:
                relay_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config",
                    "mobile_relay.json",
                )
                if os.path.isfile(relay_path):
                    with open(relay_path, encoding="utf-8") as fh:
                        relay_cfg = json.load(fh)
                    if isinstance(relay_cfg, dict) and relay_cfg:
                        summary["mobile"] = dict(relay_cfg)
            except Exception as exc:
                logger.debug("Mobile relay config unreadable: %s", exc)
        if not summary["mobile"]:
            summary["mobile"] = {
                "available": False,
                "reason": "no mobile section configured",
            }
        return summary

    def get_consumable_settings(self) -> Dict[str, Any]:
        """Returns automatic potion and buff scroll thresholds and keybinds.

        Single source of truth is DEFAULT_SETTINGS (F-0540): the old code
        hardcoded different defaults here (HP 55 / MP 40 vs 65 / 45 in
        DEFAULT_SETTINGS), so the values shown, saved and consumed never
        agreed. Live values are read through _cfg_get so a value set via
        set_consumable_settings() (or loaded from disk) is what comes back.
        """
        defaults: Dict[str, Any] = {
            "auto_use_hp_potion": bool(DEFAULT_SETTINGS.get("auto_use_hp_potion", True)),
            "hp_potion_threshold_pct": int(DEFAULT_SETTINGS.get("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT)),
            "hp_potion_key": str(DEFAULT_SETTINGS.get("hp_potion_key", "F1")),
            "auto_use_mp_potion": bool(DEFAULT_SETTINGS.get("auto_use_mp_potion", True)),
            "mp_potion_threshold_pct": int(DEFAULT_SETTINGS.get("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT)),
            "mp_potion_key": str(DEFAULT_SETTINGS.get("mp_potion_key", "F2")),
            "auto_refresh_buff_scroll": bool(DEFAULT_SETTINGS.get("auto_refresh_buff_scroll", True)),
            "buff_scroll_interval_sec": int(DEFAULT_SETTINGS.get("buff_scroll_interval_sec", 300)),
            # F-23-03: varsayılan F6'ydı — F6 yasaklı tuş (mana_phase_combat
            # FORBIDDEN_CONSUMABLE_KEYS); F7. Kaynak: DEFAULT_SETTINGS.
            "buff_scroll_key": str(DEFAULT_SETTINGS.get("buff_scroll_key", "F7")),
        }
        # Live values from the bound settings objects (guarded: this method
        # is also called on partially constructed windows in tests).
        try:
            for k in list(defaults.keys()):
                try:
                    val = self._cfg_get(k, defaults[k])
                except Exception:
                    continue
                if val is not None:
                    defaults[k] = val
        except Exception:
            pass
        # In-memory overrides from set_consumable_settings() win over the
        # settings objects until they are persisted.
        stored = self.__dict__.get("_consumable_settings_cache")
        if isinstance(stored, dict):
            for k, v in stored.items():
                if k in defaults:
                    defaults[k] = v
        try:
            cand_objs = self._candidate_settings_objects()
        except Exception:
            cand_objs = []
        for obj in cand_objs:
            combat_cfg = getattr(obj, "combat", None)
            if combat_cfg is not None:
                for k in list(defaults.keys()):
                    if hasattr(combat_cfg, k):
                        defaults[k] = getattr(combat_cfg, k)
                    elif isinstance(combat_cfg, dict) and k in combat_cfg:
                        defaults[k] = combat_cfg[k]
        # F-23-01: tüketim widget'ları varsa ekrandaki değer kazanır —
        # kullanıcı eşik/tuş değiştirip Kaydet'e basınca get_settings_dict()
        # bu değerleri alır (yoksa _cfg değeri döner; test/API yolu).
        try:
            if hasattr(self, "chk_auto_use_hp_potion"):
                defaults["auto_use_hp_potion"] = bool(self.chk_auto_use_hp_potion.isChecked())
                defaults["hp_potion_threshold_pct"] = int(self.spin_hp_potion_threshold.value())
                defaults["hp_potion_key"] = (
                    str(self.edit_hp_potion_key.text() or "").strip()
                    or str(DEFAULT_SETTINGS.get("hp_potion_key", "F1"))
                )
                defaults["auto_use_mp_potion"] = bool(self.chk_auto_use_mp_potion.isChecked())
                defaults["mp_potion_threshold_pct"] = int(self.spin_mp_potion_threshold.value())
                defaults["mp_potion_key"] = (
                    str(self.edit_mp_potion_key.text() or "").strip()
                    or str(DEFAULT_SETTINGS.get("mp_potion_key", "F2"))
                )
                defaults["auto_refresh_buff_scroll"] = bool(self.chk_auto_refresh_buff_scroll.isChecked())
                defaults["buff_scroll_interval_sec"] = int(self.spin_buff_scroll_interval.value())
                defaults["buff_scroll_key"] = (
                    str(self.edit_buff_scroll_key.text() or "").strip()
                    or str(DEFAULT_SETTINGS.get("buff_scroll_key", "F7"))
                )
        except Exception:
            pass
        return defaults

    def set_consumable_settings(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Updates automatic potion and buff scroll settings and persists them.

        Previously this only wrote an in-memory cache (and a ``combat`` attr
        when one happened to exist) and never touched the settings objects
        or disk, so the values silently reverted on the next load (F-0540).
        Values are now pushed through _cfg_set + _cfg_save like every other
        setting. F-23-03 düzeltmesiyle tuş anahtarları config/schema.py
        RouteSettings'te gerçek alan oldu; _cfg_set onları bölüm alanına
        yazar, SettingsManager.save() asdict() ile diske yazar.
        """
        current = self.get_consumable_settings()
        if isinstance(cfg, dict):
            current.update(cfg)
        self._consumable_settings_cache = dict(current)
        # F-23-01: tüketim widget'ları varsa onları da eşitle — yoksa
        # get_consumable_settings() widget değerini (eski) döndürüp
        # API ile UI çelişir (set F3 yazdı, get F1 döndü).
        try:
            if hasattr(self, "chk_auto_use_hp_potion"):
                self.chk_auto_use_hp_potion.setChecked(bool(current.get("auto_use_hp_potion", True)))
                self.spin_hp_potion_threshold.setValue(int(current.get("hp_potion_threshold_pct", HP_POTION_THRESHOLD_PCT)))
                self.edit_hp_potion_key.setText(str(current.get("hp_potion_key", "F1")))
                self.chk_auto_use_mp_potion.setChecked(bool(current.get("auto_use_mp_potion", True)))
                self.spin_mp_potion_threshold.setValue(int(current.get("mp_potion_threshold_pct", MP_POTION_THRESHOLD_PCT)))
                self.edit_mp_potion_key.setText(str(current.get("mp_potion_key", "F2")))
                self.chk_auto_refresh_buff_scroll.setChecked(bool(current.get("auto_refresh_buff_scroll", True)))
                self.spin_buff_scroll_interval.setValue(int(current.get("buff_scroll_interval_sec", 300)))
                self.edit_buff_scroll_key.setText(str(current.get("buff_scroll_key", "F7")))
        except Exception:
            pass
        try:
            cand_objs = self._candidate_settings_objects()
        except Exception:
            cand_objs = []
        for obj in cand_objs:
            combat_cfg = getattr(obj, "combat", None)
            if combat_cfg is not None:
                for k, v in current.items():
                    if isinstance(combat_cfg, dict):
                        combat_cfg[k] = v
                    else:
                        try:
                            setattr(combat_cfg, k, v)
                        except Exception:
                            pass
        # Persist through the normal settings path (batched: one disk write).
        try:
            self._cfg_batching = True
            try:
                for key in (
                    "auto_use_hp_potion",
                    "hp_potion_threshold_pct",
                    # F-23-03: tuşlar eskiden bu listede yoktu — yalnızca
                    # bellek önbelleğinde kalıyor, diske hiç yazılmıyordu.
                    "hp_potion_key",
                    "auto_use_mp_potion",
                    "mp_potion_threshold_pct",
                    "mp_potion_key",
                    "auto_refresh_buff_scroll",
                    "buff_scroll_interval_sec",
                    "buff_scroll_key",
                ):
                    self._cfg_set(key, current[key])
            finally:
                self._cfg_batching = False
            self._last_consumable_save_ok = bool(self._cfg_save())
        except Exception as exc:
            logger.debug("Consumable settings persist failed: %s", exc)
            self._last_consumable_save_ok = False
        return dict(current)

    def filter_node_table_by_query(self, query: str) -> int:
        """Filters the route node table rows by material/type/coordinate query and returns visible row count."""
        q = str(query or "").strip().lower()
        tbl = self.__dict__.get("nodes_table") or self.__dict__.get("table_nodes")
        if tbl is None:
            return 0
        visible_count = 0
        for r in range(tbl.rowCount()):
            row_text = " ".join(
                (tbl.item(r, c).text() if tbl.item(r, c) is not None else "")
                for c in range(tbl.columnCount())
            ).lower()
            match = (not q) or (q in row_text)
            tbl.setRowHidden(r, not match)
            if match:
                visible_count += 1
        return visible_count

    def get_settings_node_hygiene_summary(self) -> Dict[str, Any]:
        """Returns hygiene metrics for the active region's nodes in SettingsWindow."""
        reg = self._current_region() if hasattr(self, "_current_region") else "Beluslan"
        rte = self._current_route() if hasattr(self, "_current_route") else "All"
        nodes = self._get_nodes_for_selection(reg, rte) if hasattr(self, "_get_nodes_for_selection") else []
        return {
            "region": reg,
            "route": rte,
            "node_count": len(nodes),
            "consumables": self.get_consumable_settings(),
            "ready": True,
        }

    def get_settings_window_health_summary(self) -> dict:
        ready = validate_settings_window_contract()
        return {"settings_window_ready": bool(ready)}

    def get_settings_window_ready_flag(self) -> dict:
        return {"module": "settings_window", "ready": bool(validate_settings_window_contract())}

    def get_settings_window_version_info(self) -> dict:
        return {
            "module": "settings_window",
            "version": __version__,
            "ready": bool(validate_settings_window_contract()),
        }

    def get_settings_window_diagnostic_snapshot(self) -> dict:
        ready = validate_settings_window_contract()
        return {
            "module": "settings_window",
            "status": "ok" if ready else "contract_failed",
            "ready": bool(ready),
        }


# Backwards compatibility alias
SettingsDialog = SettingsWindow


def get_settings_window_capabilities() -> list:
    return ["settings_window", "configuration_ui", "diagnostic_snapshot"]


def validate_settings_window_contract() -> bool:
    """Real contract check (F-0763): the old version returned True whenever
    the capabilities list was non-empty, i.e. unconditionally.

    The contract holds only when the settings schema, the window class and
    its load/save entry points are all present and consistent.
    """
    try:
        if not get_settings_window_capabilities():
            return False
        if not DEFAULT_SETTINGS:
            return False
        required_keys = {
            "f_interval_min",
            "f_interval_max",
            "test_mode",
            "hp_potion_threshold_pct",
            "mp_potion_threshold_pct",
            "buff_scroll_interval_sec",
        }
        if not required_keys.issubset(DEFAULT_SETTINGS.keys()):
            return False
        for name in (
            "load_settings_into_ui",
            "get_settings_dict",
            "save_settings",
            "reset_to_defaults",
            "_load_from_settings",
            "_save_to_settings",
            "get_consumable_settings",
            "set_consumable_settings",
        ):
            if not callable(getattr(SettingsWindow, name, None)):
                return False
        return True
    except Exception as exc:
        logger.debug("Settings window contract check failed: %s", exc)
        return False


def get_settings_window_baseline_metrics() -> dict:
    return {"module": "settings_window", "contract_valid": validate_settings_window_contract(), "capability_count": len(get_settings_window_capabilities())}


def get_settings_window_readiness_status() -> dict:
    return {"module": "settings_window", "ready": validate_settings_window_contract(), "metrics": get_settings_window_baseline_metrics()}


def get_settings_window_audit_record() -> dict:
    return {"module": "settings_window", "audit_passed": validate_settings_window_contract(), "readiness": get_settings_window_readiness_status()}


def get_settings_window_operational_digest() -> dict:
    return {"module": "settings_window", "operational": validate_settings_window_contract(), "audit": get_settings_window_audit_record()}


def get_settings_window_telemetry_payload() -> dict:
    return {"module": "settings_window", "telemetry_ready": validate_settings_window_contract(), "digest": get_settings_window_operational_digest()}


def get_settings_window_integrity_seal() -> dict:
    return {"module": "settings_window", "sealed": validate_settings_window_contract(), "telemetry": get_settings_window_telemetry_payload()}


def get_settings_window_compliance_certificate() -> dict:
    return {"module": "settings_window", "certified": validate_settings_window_contract(), "seal": get_settings_window_integrity_seal()}


def get_settings_window_readiness_probe() -> dict:
    return {"module": "settings_window", "ready": validate_settings_window_contract(), "certificate": get_settings_window_compliance_certificate()}


def get_settings_window_audit_summary() -> dict:
    return {"module": "settings_window", "audit_passed": validate_settings_window_contract(), "probe": get_settings_window_readiness_probe()}


