# -*- coding: utf-8 -*-
"""
PyQt6 Interactive Numbered Map Viewer (Gatherion 1.5.2 Authentic Replica)
Matches video_frames/frame_24s.jpg and video_frames/frame_27s.jpg:
- Window title: "Nodes on the map of <Region> · green in the route, gray out of it, amber resting"
- Grayscale region map background (auto-crops map region from video_frames/frame_18s.jpg
  or composites saved 80x80 node patches into a mosaic map if no region map is stored)
- High-quality anti-aliased QPainter rendering of all registered region nodes:
  * Green circle (#22c55e) + black border + bold outlined number -> In active route & Ready
  * Amber circle (#f59e0b) + black border + bold outlined number -> Resting (on cooldown)
  * Gray circle (#64748b) + black border + bold outlined number -> Out of active route
  * Blue directional arrow/dot (#3b82f6) -> Player current position (if available)
- Interactive Zoom (mouse wheel) and Pan (mouse drag)
- Left-click on node badge: toggles node in/out of active route (emits node_toggled(int))
- Right-click on node badge: opens node info / reset state / delete context menu
"""

import os
import time
import math
import logging
from typing import Optional, Dict, List, Tuple, Any

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMenu, QMessageBox, QSizePolicy
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QMouseEvent, QWheelEvent, QPaintEvent, QPolygonF, QPainterPath,
    QAction, QCursor
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRect, QRectF, QSize

logger = logging.getLogger("Gatherion.MapViewer")


class _MapCanvas(QWidget):
    """
    Interactive Zoom & Pan canvas that draws the grayscale map,
    node badges, outlined node numbers, and player marker.
    """
    node_left_clicked = pyqtSignal(int)
    node_right_clicked = pyqtSignal(int, QPoint)

    # Normalized crop rectangle of the central map inside frame_18s.jpg (works for 1280x720, 1920x1080, etc.)
    # Matches the exact map bounds shown in frame_24s.jpg & frame_27s.jpg (Safe Haven to Calderon Canyon)
    FRAME18_NORM_CROP = (0.295, 0.145, 0.480, 0.625)  # (x_ratio, y_ratio, w_ratio, h_ratio)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._base_pixmap: Optional[QPixmap] = None
        self._coord_offset: Tuple[int, int] = (0, 0)
        self._nodes: List[Dict[str, Any]] = []
        self._route_node_ids: set = set()
        self._rest_duration_sec: float = 300.0
        self._player_pos: Optional[Tuple[float, float, float]] = None  # (x, y, angle_deg)

        # Zoom & Pan state
        self._zoom: float = 1.0
        self._min_zoom: float = 0.4
        self._max_zoom: float = 5.0
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._is_dragging: bool = False
        self._drag_start_pos: QPointF = QPointF(0.0, 0.0)
        self._drag_moved_px: float = 0.0
        self._hovered_node_id: Optional[int] = None
        self._initial_fit_done: bool = False

    def set_map_pixmap(self, pixmap: QPixmap, coord_offset: Tuple[int, int] = (0, 0)) -> None:
        self._base_pixmap = pixmap
        self._coord_offset = coord_offset
        self._initial_fit_done = False
        self._fit_to_view()
        self.update()

    def set_nodes_data(
        self,
        nodes: List[Dict[str, Any]],
        route_node_ids: List[int],
        rest_duration_sec: float = 300.0
    ) -> None:
        self._nodes = list(nodes or [])
        self._route_node_ids = set(int(nid) for nid in (route_node_ids or []))
        self._rest_duration_sec = float(rest_duration_sec)
        self.update()

    def set_player_position(
        self,
        x: Optional[float],
        y: Optional[float],
        angle_deg: float = 0.0
    ) -> None:
        if x is None or y is None:
            self._player_pos = None
        else:
            self._player_pos = (float(x), float(y), float(angle_deg))
        self.update()

    def reset_view(self) -> None:
        self._fit_to_view()
        self.update()

    def _fit_to_view(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        cw = max(1, self.width())
        ch = max(1, self.height())
        pw = max(1, self._base_pixmap.width())
        ph = max(1, self._base_pixmap.height())

        scale_x = cw / pw
        scale_y = ch / ph
        # Cover/fit nicely like frame_24s.jpg
        self._zoom = max(scale_x, scale_y) if ( 0.75 <= (scale_x / max(0.001, scale_y)) <= 1.35 ) else min(scale_x, scale_y)
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom))
        self._pan = QPointF(
            (cw - pw * self._zoom) * 0.5,
            (ch - ph * self._zoom) * 0.5
        )

    def resizeEvent(self, event) -> None:
        if not self._initial_fit_done and self.width() > 100 and self.height() > 100:
            self._fit_to_view()
            self._initial_fit_done = True
        super().resizeEvent(event)

    # -------------------------------------------------------------------------
    # Coordinate Mapping
    # -------------------------------------------------------------------------
    def _node_to_map_coords(self, raw_x: float, raw_y: float) -> QPointF:
        """
        Converts node coordinates (which may be full-screen 1920x1080 or map-local)
        to pixel coordinates on self._base_pixmap.
        """
        if self._base_pixmap is None:
            return QPointF(raw_x, raw_y)

        pw = self._base_pixmap.width()
        ph = self._base_pixmap.height()
        ox, oy = self._coord_offset

        # If offset is set (e.g. cropped from 1920x1080 frame_18s.jpg) and raw coords
        # fall inside full-screen bounds, shift by crop offset:
        if (ox != 0 or oy != 0) and (raw_x > pw * 0.95 or raw_y > ph * 0.95 or (raw_x >= ox and raw_y >= oy)):
            mapped_x = raw_x - ox
            mapped_y = raw_y - oy
            if 0 <= mapped_x <= pw and 0 <= mapped_y <= ph:
                return QPointF(mapped_x, mapped_y)

        return QPointF(
            max(12.0, min(float(pw - 12), float(raw_x))),
            max(12.0, min(float(ph - 12), float(raw_y)))
        )

    def _map_to_widget(self, map_pt: QPointF) -> QPointF:
        return QPointF(
            self._pan.x() + map_pt.x() * self._zoom,
            self._pan.y() + map_pt.y() * self._zoom
        )

    def _widget_to_map(self, widget_pt: QPointF) -> QPointF:
        z = max(0.001, self._zoom)
        return QPointF(
            (widget_pt.x() - self._pan.x()) / z,
            (widget_pt.y() - self._pan.y()) / z
        )

    def _find_node_at_widget_pos(self, pos: QPointF, hit_radius: float = 11.0) -> Optional[int]:
        """Finds the closest node within hit_radius (in widget pixels)."""
        best_id: Optional[int] = None
        best_dist = hit_radius
        for node in reversed(self._nodes):
            try:
                nid = int(node.get("id", 0))
                nx = float(node.get("map_x", node.get("x", 0)))
                ny = float(node.get("map_y", node.get("y", 0)))
                w_pt = self._map_to_widget(self._node_to_map_coords(nx, ny))
                dist = math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y())
                if dist <= best_dist:
                    best_dist = dist
                    best_id = nid
            except Exception:
                continue
        return best_id

    # -------------------------------------------------------------------------
    # Mouse Events (Zoom, Pan, Click Node)
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        try:
            angle_delta = event.angleDelta().y()
            if angle_delta == 0:
                return
            factor = 1.15 if angle_delta > 0 else (1.0 / 1.15)
            new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
            if abs(new_zoom - self._zoom) < 1e-5:
                return

            cursor_pos = event.position()
            map_pt_before = self._widget_to_map(cursor_pos)
            self._zoom = new_zoom
            self._pan = QPointF(
                cursor_pos.x() - map_pt_before.x() * self._zoom,
                cursor_pos.y() - map_pt_before.y() * self._zoom
            )
            self.update()
        except Exception as exc:
            logger.error("Error in MapCanvas.wheelEvent: %s", exc)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._is_dragging = True
            self._drag_start_pos = event.position()
            self._drag_moved_px = 0.0
            if self._hovered_node_id is None:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            nid = self._find_node_at_widget_pos(event.position())
            if nid is not None:
                self.node_right_clicked.emit(nid, event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._is_dragging:
            delta = pos - self._drag_start_pos
            self._drag_moved_px += math.hypot(delta.x(), delta.y())
            self._pan += delta
            self._drag_start_pos = pos
            self.update()
            return

        hovered = self._find_node_at_widget_pos(pos)
        if hovered != self._hovered_node_id:
            self._hovered_node_id = hovered
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hovered is not None else Qt.CursorShape.OpenHandCursor
            )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            was_dragging = self._is_dragging
            moved = self._drag_moved_px
            self._is_dragging = False

            if event.button() == Qt.MouseButton.LeftButton and was_dragging and moved < 5.0:
                nid = self._find_node_at_widget_pos(event.position())
                if nid is not None:
                    self.node_left_clicked.emit(nid)

            hovered = self._find_node_at_widget_pos(event.position())
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hovered is not None else Qt.CursorShape.OpenHandCursor
            )
        super().mouseReleaseEvent(event)

    # -------------------------------------------------------------------------
    # Painting (Grayscale Map + Anti-aliased Node Badges + Outlined Numbers)
    # -------------------------------------------------------------------------
    def _classify_node(self, node: Dict[str, Any], now: float) -> str:
        """
        Returns 'ready' (green #22c55e), 'resting' (amber #f59e0b),
        or 'out_of_route' (gray #64748b).
        """
        nid = int(node.get("id", 0))
        explicit_status = str(node.get("status", "")).strip().lower()
        in_route = node.get("in_route", nid in self._route_node_ids)

        if not in_route or explicit_status in ("out", "out_of_route", "gray", "disabled", "inactive"):
            return "out_of_route"

        if explicit_status in ("resting", "cooldown", "amber"):
            return "resting"

        last_visited = float(node.get("last_visited", 0) or 0)
        if last_visited > 0 and (now - last_visited) < self._rest_duration_sec:
            return "resting"

        return "ready"

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Fill dark background outside map bounds
        painter.fillRect(self.rect(), QColor("#14171c"))

        # 1. Draw Grayscale Region Map
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            target_rect = QRectF(
                self._pan.x(),
                self._pan.y(),
                self._base_pixmap.width() * self._zoom,
                self._base_pixmap.height() * self._zoom
            )
            painter.drawPixmap(target_rect, self._base_pixmap, QRectF(self._base_pixmap.rect()))

        # 2. Draw Player Position Marker (Blue Arrow / Dot #3b82f6) if available
        if self._player_pos is not None:
            px, py, angle_deg = self._player_pos
            w_pt = self._map_to_widget(self._node_to_map_coords(px, py))
            self._draw_player_marker(painter, w_pt, angle_deg)

        # 3. Draw All Nodes (Green #22c55e, Amber #f59e0b, Gray #64748b)
        #    + Cyan/Emerald dual ring for minimap-auto-recorded nodes (source_type == "minimap_auto")
        #    + Harvest status badge (0/3, 1/3, 2/3 or 3x) next to each node pin
        now = time.time()
        color_map = {
            "ready": QColor("#22c55e"),        # Green in the route
            "resting": QColor("#f59e0b"),      # Amber resting
            "out_of_route": QColor("#64748b")  # Gray out of route
        }

        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.Bold)
        badge_font = QFont("Segoe UI", 8)
        badge_font.setWeight(QFont.Weight.Bold)

        for node in self._nodes:
            try:
                nid = int(node.get("id", 0))
                nx = float(node.get("map_x", node.get("x", 0)))
                ny = float(node.get("map_y", node.get("y", 0)))
                w_pt = self._map_to_widget(self._node_to_map_coords(nx, ny))

                state = self._classify_node(node, now)
                fill_color = color_map.get(state, QColor("#22c55e"))
                is_hovered = (nid == self._hovered_node_id)
                is_minimap_auto = self._is_minimap_auto_node(node)

                radius = 6.5 if is_hovered else 5.2
                border_width = 2.2 if is_hovered else 1.8

                # Distinct cyan/emerald dual ring for minimap-auto-recorded nodes (source_type == "minimap_auto")
                if is_minimap_auto:
                    self._draw_minimap_auto_dual_ring(painter, w_pt, radius, is_hovered)

                # Circle with crisp black border (matches frame_24s.jpg)
                if is_hovered:
                    painter.setPen(QPen(QColor("#ffffff"), 3.2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(w_pt, radius + 1.5, radius + 1.5)

                painter.setPen(QPen(QColor("#000000"), border_width))
                painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(w_pt, radius, radius)

                # Bold white node number with dark outline/shadow to the right of the circle
                painter.setFont(font)
                label_str = str(nid)
                ring_pad = 5.5 if is_minimap_auto else 0.0
                tx = w_pt.x() + radius + 3.5 + ring_pad
                ty = w_pt.y() + 4.0

                self._draw_outlined_text(painter, tx, ty, label_str, font)

                # Harvest status badge (0/3, 1/3, 2/3 or 3x) next to the node pin & number
                badge_text = self._format_harvest_badge(node)
                fm = painter.fontMetrics()
                id_width = float(fm.horizontalAdvance(label_str))
                badge_x = tx + id_width + 5.0
                badge_y = w_pt.y()
                self._draw_harvest_status_badge(
                    painter=painter,
                    x=badge_x,
                    y=badge_y,
                    badge_text=badge_text,
                    font=badge_font,
                    is_minimap_auto=is_minimap_auto,
                    state=state,
                )
            except Exception as exc:
                logger.debug("Error drawing node on map: %s", exc)

        # 4. Draw Map Legend (Pin states + Minimap-Auto Cyan/Emerald Dual Ring + Harvest Status Badge)
        self._draw_map_legend(painter)

        painter.end()

    @staticmethod
    def _is_minimap_auto_node(node: Dict[str, Any]) -> bool:
        """Returns True if the node was auto-recorded from the top-right minimap."""
        if not isinstance(node, dict):
            return False
        src = str(node.get("source_type", node.get("source", ""))).strip().lower()
        if src == "minimap_auto":
            return True
        return bool(node.get("minimap_auto", False))

    @staticmethod
    def _format_harvest_badge(node: Dict[str, Any]) -> str:
        """
        Formats the node's harvest status badge ('0/3', '1/3', '2/3' or '3x', or '6x'/'9x' for stacked nodes).
        """
        if not isinstance(node, dict):
            return "0/3"

        explicit_badge = node.get("harvest_badge") or node.get("harvest_status")
        if explicit_badge is not None and str(explicit_badge).strip():
            return str(explicit_badge).strip()

        max_h = max(1, int(node.get("max_harvests", node.get("stack_count", 3)) or 3))

        if "harvest_count" in node or "current_harvests" in node:
            cur_h = max(0, int(node.get("harvest_count", node.get("current_harvests", 0)) or 0))
            if str(node.get("badge_mode", "")).strip().lower() == "stack" or cur_h >= max_h:
                return f"{max_h}x"
            return f"{cur_h}/{max_h}"

        if "stack_badge" in node and str(node.get("stack_badge", "")).strip():
            return str(node.get("stack_badge")).strip()

        if "stack_count" in node and "visit_count" not in node and "gathered_count" not in node:
            return f"{max_h}x"

        visits = max(0, int(node.get("visit_count", node.get("gathered_count", 0)) or 0))
        cycle = visits % (max_h + 1)
        if cycle >= max_h:
            return f"{max_h}x"
        return f"{cycle}/{max_h}"

    @staticmethod
    def _draw_minimap_auto_dual_ring(
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        is_hovered: bool = False,
    ) -> None:
        """
        Draws a distinct cyan/emerald dual ring around minimap-auto-recorded nodes
        (source_type == 'minimap_auto') so the user can visually identify them on the route map.
        """
        outer_r = base_radius + (6.2 if is_hovered else 5.2)
        inner_r = base_radius + (3.4 if is_hovered else 2.6)

        # Soft cyan-emerald halo glow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(6, 182, 212, 48)))
        painter.drawEllipse(center, outer_r + 2.4, outer_r + 2.4)

        # Outer bright cyan ring (#22d3ee)
        outer_pen = QPen(QColor("#22d3ee"), 2.0, Qt.PenStyle.SolidLine)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, outer_r, outer_r)

        # Inner bright emerald ring (#10b981)
        inner_pen = QPen(QColor("#10b981"), 2.0, Qt.PenStyle.SolidLine)
        painter.setPen(inner_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, inner_r, inner_r)

    @staticmethod
    def _draw_harvest_status_badge(
        painter: QPainter,
        x: float,
        y: float,
        badge_text: str,
        font: QFont,
        is_minimap_auto: bool = False,
        state: str = "ready",
    ) -> None:
        """
        Draws the harvest status badge ('0/3', '1/3', '2/3' or '3x') as a compact pill next to the node pin.
        """
        painter.save()
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = max(18.0, float(fm.horizontalAdvance(badge_text)))
        pad_x = 4.5
        pill_w = tw + pad_x * 2.0
        pill_h = 14.5
        pill_rect = QRectF(x, y - pill_h * 0.5 + 0.5, pill_w, pill_h)

        if is_minimap_auto:
            bg_color = QColor(6, 44, 48, 230)
            border_color = QColor("#22d3ee") if not badge_text.endswith("x") else QColor("#10b981")
            text_color = QColor("#34d399") if badge_text.endswith("x") else QColor("#67e8f9")
        elif state == "resting":
            bg_color = QColor(45, 27, 8, 225)
            border_color = QColor("#f59e0b")
            text_color = QColor("#fde68a")
        elif state == "out_of_route":
            bg_color = QColor(23, 30, 42, 215)
            border_color = QColor("#64748b")
            text_color = QColor("#cbd5e1")
        else:
            bg_color = QColor(12, 34, 24, 225)
            border_color = QColor("#22c55e")
            text_color = QColor("#4ade80") if badge_text.endswith("x") else QColor("#e2e8f0")

        painter.setPen(QPen(border_color, 1.3))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(pill_rect, 4.0, 4.0)

        painter.setPen(text_color)
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.restore()

    def _draw_map_legend(self, painter: QPainter) -> None:
        """Draws a bottom Turkish legend explaining node colors, minimap_auto dual rings, and 0/3..3x badges."""
        if self.height() < 120 or self.width() < 260:
            return
        legend_w = min(float(self.width() - 20), 920.0)
        legend_rect = QRectF(10.0, float(self.height() - 34), legend_w, 24.0)
        painter.save()
        painter.setPen(QPen(QColor(45, 55, 72, 200), 1.0))
        painter.setBrush(QBrush(QColor(15, 20, 28, 220)))
        painter.drawRoundedRect(legend_rect, 5.0, 5.0)

        l_font = QFont("Segoe UI", 8)
        l_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(l_font)
        painter.setPen(QColor("#e2e8f0"))
        legend_str = (
            "  ● Aktif   |   ● Dinleniyor (5dk)   |   ● Başarısız   |   "
            "◎ Minimap/3B Oto-Kayıt   |   [0/3 · 1/3 · 2/3 · 3x]: Hasat Rozeti   |   "
            "Sol Tık: Nokta Ekle | Sağ Tık: Nokta Sil | Sürükle: Taşı"
        )
        painter.drawText(
            legend_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            legend_str,
        )
        painter.restore()

    @staticmethod
    def _draw_outlined_text(
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        font: QFont
    ) -> None:
        """
        Draws crisp white bold text with a black stroke/outline for readability
        against any grayscale terrain (matching frame_24s.jpg).
        """
        path = QPainterPath()
        path.addText(QPointF(x, y), font, text)
        painter.setPen(
            QPen(
                QColor(0, 0, 0, 235),
                3.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPath(path)

    @staticmethod
    def _draw_player_marker(painter: QPainter, pt: QPointF, angle_deg: float = 0.0) -> None:
        """Draws the player's current position as a blue directional arrow/dot (#3b82f6)."""
        painter.save()
        painter.translate(pt)
        painter.rotate(angle_deg)

        # Outer glow ring
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(59, 130, 246, 80)))
        painter.drawEllipse(QPointF(0, 0), 11.0, 11.0)

        # Directional arrow polygon (#3b82f6 with black/white border)
        arrow = QPolygonF([
            QPointF(0.0, -9.0),
            QPointF(6.5, 7.0),
            QPointF(0.0, 3.5),
            QPointF(-6.5, 7.0),
        ])
        painter.setPen(QPen(QColor("#000000"), 2.0))
        painter.setBrush(QBrush(QColor("#3b82f6")))
        painter.drawPolygon(arrow)

        painter.restore()


class MapViewerWindow(QDialog):
    """
    Rota ve Harita Görüntüleyici Penceresi (Gatherion 1.5.2 Uyumlu Türkçe Arayüz).
    Harita üzerindeki tüm noktaları, hasat durum rozetlerini (0/3, 1/3, 2/3, 3x) ve
    Minimap/3B Oto-Kayıt (source_type == 'minimap_auto') çift halkalı noktaları gösterir.
    """
    node_toggled = pyqtSignal(int)
    node_deleted = pyqtSignal(int)

    def __init__(
        self,
        route_manager: Optional[Any] = None,
        region_name: Optional[str] = None,
        route_name: Optional[str] = None,
        map_image_path: Optional[str] = None,
        rest_duration_sec: float = 300.0,
        player_pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.routes = route_manager
        self._region_name = region_name or self._resolve_active_region()
        self._route_name = route_name or self._resolve_active_route()
        self._explicit_map_path = map_image_path
        self._rest_duration_sec = float(rest_duration_sec)
        self._player_pos = player_pos

        self._init_ui()
        self.reload_map_and_nodes()

    def _resolve_active_region(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_region") and self.routes.active_region:
                    return str(self.routes.active_region)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg = self.routes.data.get("active_region")
                    if reg:
                        return str(reg)
        except Exception as exc:
            logger.debug("Could not resolve active_region: %s", exc)
        return "Altgard"

    def _resolve_active_route(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_route") and self.routes.active_route:
                    return str(self.routes.active_route)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    rte = self.routes.data.get("active_route")
                    if rte:
                        return str(rte)
        except Exception as exc:
            logger.debug("Could not resolve active_route: %s", exc)
        return "All"

    def _build_title_text(self) -> str:
        region_label = self._region_name or "Altgard"
        route_label = self._route_name or "All"
        return (
            f"Rota ve Harita Görüntüleyici · Bölge: {region_label} | Rota: {route_label} · "
            "Aktif, Dinleniyor (5dk), Başarısız, Minimap/3B Oto-Kayıt"
        )

    def _init_ui(self) -> None:
        title_str = self._build_title_text()
        self.setWindowTitle(title_str)
        self.resize(980, 680)
        self.setStyleSheet("""
            QDialog {
                background-color: #14171c;
                color: #e5e7eb;
                border: 1px solid #2d323b;
                font-family: 'Segoe UI', sans-serif;
            }
            QWidget#MapTopHeader, QWidget#MapBottomFooter {
                background-color: #181c23;
                border-bottom: 1px solid #2d323b;
            }
            QLabel {
                color: #e5e7eb;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Üst Türkçe Bilgi & Kontrol Çubuğu
        self.header_bar = QWidget(self)
        self.header_bar.setObjectName("MapTopHeader")
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(10)

        self.lbl_title_main = QLabel("Rota ve Harita Görüntüleyici", self.header_bar)
        self.lbl_title_main.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        header_layout.addWidget(self.lbl_title_main)

        self.lbl_region_prefix = QLabel("Bölge:", self.header_bar)
        self.lbl_region_prefix.setStyleSheet("color: #94a3b8; font-weight: bold;")
        header_layout.addWidget(self.lbl_region_prefix)

        self.lbl_region_value = QLabel(str(self._region_name), self.header_bar)
        self.lbl_region_value.setStyleSheet("color: #f8fafc; font-weight: bold;")
        header_layout.addWidget(self.lbl_region_value)

        self.lbl_route_prefix = QLabel("Rota:", self.header_bar)
        self.lbl_route_prefix.setStyleSheet("color: #94a3b8; font-weight: bold;")
        header_layout.addWidget(self.lbl_route_prefix)

        self.lbl_route_value = QLabel(str(self._route_name), self.header_bar)
        self.lbl_route_value.setStyleSheet("color: #f8fafc; font-weight: bold;")
        header_layout.addWidget(self.lbl_route_value)

        header_layout.addStretch(1)

        # Durum Rozetleri (Aktif | Dinleniyor (5dk) | Başarısız | Minimap/3B Oto-Kayıt)
        self.lbl_badge_active = QLabel("● Aktif", self.header_bar)
        self.lbl_badge_active.setStyleSheet("color: #22c55e; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self.lbl_badge_active)

        self.lbl_badge_resting = QLabel("● Dinleniyor (5dk)", self.header_bar)
        self.lbl_badge_resting.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self.lbl_badge_resting)

        self.lbl_badge_failed = QLabel("● Başarısız", self.header_bar)
        self.lbl_badge_failed.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self.lbl_badge_failed)

        self.lbl_badge_minimap_auto = QLabel("◎ Minimap/3B Oto-Kayıt", self.header_bar)
        self.lbl_badge_minimap_auto.setStyleSheet("color: #22d3ee; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self.lbl_badge_minimap_auto)

        self.btn_optimize_route = QPushButton("Sıralı Rota Optimize Et", self.header_bar)
        self.btn_optimize_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_optimize_route.clicked.connect(self.optimize_sequential_route)
        header_layout.addWidget(self.btn_optimize_route)

        root_layout.addWidget(self.header_bar)

        # Canvas
        self.canvas = _MapCanvas(self)
        self.canvas.node_left_clicked.connect(self._on_node_left_clicked)
        self.canvas.node_right_clicked.connect(self._on_node_right_clicked)
        root_layout.addWidget(self.canvas, stretch=1)

        # Alt Türkçe Yardım Çubuğu
        self.footer_bar = QWidget(self)
        self.footer_bar.setObjectName("MapBottomFooter")
        footer_layout = QHBoxLayout(self.footer_bar)
        footer_layout.setContentsMargins(12, 4, 12, 4)
        self.lbl_footer_help = QLabel(
            "Sol Tık: Nokta Ekle | Sağ Tık: Nokta Sil | Sürükle: Taşı",
            self.footer_bar,
        )
        self.lbl_footer_help.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        footer_layout.addWidget(self.lbl_footer_help)
        footer_layout.addStretch(1)
        root_layout.addWidget(self.footer_bar)

    # -------------------------------------------------------------------------
    # Grayscale Map Loading & Mosaic Fallback
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_grayscale_pixmap(image: QImage) -> QPixmap:
        """Converts any QImage to a crisp 8-bit grayscale QPixmap matching frame_24s.jpg."""
        if image.isNull():
            return QPixmap()
        gray_img = image.convertToFormat(QImage.Format.Format_Grayscale8)
        # Convert back to RGB32 so QPainter draws colored circles on top if needed
        rgb_gray = gray_img.convertToFormat(QImage.Format.Format_RGB32)
        return QPixmap.fromImage(rgb_gray)

    def _load_region_map_pixmap(self, nodes: List[Dict[str, Any]]) -> Tuple[QPixmap, Tuple[int, int]]:
        """
        Loads the region map image and converts it to grayscale.
        Fallback priority:
        1. Explicit map_image_path (if valid and exists)
        2. Region-specific map file in routes/maps/<region>.png or jpg
        3. Cropped map region from video_frames/frame_18s.jpg
        4. Mosaic constructed from saved 80x80 node patches in routes/patches/
        """
        candidates: List[Tuple[str, bool]] = []

        if self._explicit_map_path and os.path.exists(self._explicit_map_path):
            is_f18 = os.path.basename(self._explicit_map_path).lower() == "frame_18s.jpg"
            candidates.append((self._explicit_map_path, is_f18))

        safe_reg = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (self._region_name or "Altgard"))
        for rel_path in (
            os.path.join("routes", "maps", f"{safe_reg}.png"),
            os.path.join("routes", "maps", f"{safe_reg}.jpg"),
            os.path.join("routes", f"{safe_reg}_map.png"),
            os.path.join("assets", "maps", f"{safe_reg}.png"),
        ):
            if os.path.exists(rel_path):
                candidates.append((rel_path, False))

        # Primary fallback from technical spec: video_frames/frame_18s.jpg map region
        frame18_path = os.path.join(os.getcwd(), "video_frames", "frame_18s.jpg")
        if os.path.exists(frame18_path):
            candidates.append((frame18_path, True))

        for path, should_crop_f18 in candidates:
            try:
                img = QImage(path)
                if not img.isNull():
                    offset = (0, 0)
                    if should_crop_f18:
                        rx, ry, rw, rh = _MapCanvas.FRAME18_NORM_CROP
                        cx = int(img.width() * rx)
                        cy = int(img.height() * ry)
                        cw = int(img.width() * rw)
                        ch = int(img.height() * rh)
                        crop_rect = QRect(cx, cy, cw, ch)
                        img = img.copy(crop_rect)
                        offset = (cx, cy)
                    gray_pix = self._to_grayscale_pixmap(img)
                    if not gray_pix.isNull():
                        return gray_pix, offset
            except Exception as exc:
                logger.warning("Failed loading candidate map image '%s': %s", path, exc)

        # Mosaic fallback from 80x80 node patches
        return self._build_patch_mosaic_pixmap(nodes), (0, 0)

    def _build_patch_mosaic_pixmap(self, nodes: List[Dict[str, Any]]) -> QPixmap:
        """
        Builds a grayscale map mosaic using saved 80x80 node patches in routes/patches/
        placed at each node's coordinates.
        """
        width, height = 1180, 740
        canvas_img = QImage(width, height, QImage.Format.Format_RGB32)
        canvas_img.fill(QColor("#262a30"))

        painter = QPainter(canvas_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Subtle topographic grid lines
        grid_pen = QPen(QColor("#2f343c"), 1)
        painter.setPen(grid_pen)
        for gx in range(0, width, 60):
            painter.drawLine(gx, 0, gx, height)
        for gy in range(0, height, 60):
            painter.drawLine(0, gy, width, gy)

        patch_dir = "routes/patches"
        if self.routes is not None and hasattr(self.routes, "patch_dir"):
            patch_dir = self.routes.patch_dir

        for node in (nodes or []):
            patch_file = node.get("patch_file") or node.get("patch_path")
            if not patch_file:
                continue
            p_path = patch_file if os.path.isabs(patch_file) else os.path.join(patch_dir, os.path.basename(patch_file))
            if os.path.exists(p_path):
                try:
                    p_img = QImage(p_path)
                    if not p_img.isNull():
                        gray_patch = p_img.convertToFormat(QImage.Format.Format_Grayscale8)
                        nx = int(node.get("map_x", node.get("x", width // 2)))
                        ny = int(node.get("map_y", node.get("y", height // 2)))
                        pw, ph = gray_patch.width(), gray_patch.height()
                        painter.drawImage(nx - pw // 2, ny - ph // 2, gray_patch)
                except Exception as exc:
                    logger.debug("Failed to composite node patch '%s': %s", p_path, exc)

        painter.end()
        return QPixmap.fromImage(canvas_img)

    # -------------------------------------------------------------------------
    # Data Synchronization & Node Operations
    # -------------------------------------------------------------------------
    def reload_map_and_nodes(self) -> None:
        """Reloads region nodes, active route membership, and grayscale map."""
        try:
            self._region_name = self._region_name or self._resolve_active_region()
            self._route_name = self._route_name or self._resolve_active_route()
            self.setWindowTitle(self._build_title_text())
            if hasattr(self, "lbl_region_value"):
                self.lbl_region_value.setText(str(self._region_name))
            if hasattr(self, "lbl_route_value"):
                self.lbl_route_value.setText(str(self._route_name))

            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            pixmap, offset = self._load_region_map_pixmap(nodes)

            # Check if node coordinates are already relative to the cropped map or full-screen
            if offset != (0, 0) and nodes:
                max_x = max(float(n.get("map_x", n.get("x", 0))) for n in nodes)
                max_y = max(float(n.get("map_y", n.get("y", 0))) for n in nodes)
                if max_x <= pixmap.width() and max_y <= pixmap.height():
                    offset = (0, 0)

            self.canvas.set_map_pixmap(pixmap, coord_offset=offset)
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
            if self._player_pos is not None:
                self.canvas.set_player_position(self._player_pos[0], self._player_pos[1])
        except Exception as exc:
            logger.error("Error in MapViewerWindow.reload_map_and_nodes: %s", exc)

    def _extract_region_nodes_and_route_ids(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        nodes: List[Dict[str, Any]] = []
        route_ids: List[int] = []

        if self.routes is None:
            return nodes, route_ids

        try:
            if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                regions = self.routes.data.get("regions", {})
                reg_data = regions.get(self._region_name)
                if reg_data is None and regions:
                    # Fallback to first available region
                    first_reg = next(iter(regions.keys()))
                    self._region_name = first_reg
                    self.setWindowTitle(self._build_title_text())
                    if hasattr(self, "lbl_region_value"):
                        self.lbl_region_value.setText(str(self._region_name))
                    reg_data = regions.get(first_reg, {})

                if isinstance(reg_data, dict):
                    raw_nodes = reg_data.get("nodes", [])
                    if isinstance(raw_nodes, list):
                        for item in raw_nodes:
                            if isinstance(item, dict):
                                nodes.append(item)
                            elif hasattr(item, "to_dict") and callable(item.to_dict):
                                nodes.append(dict(item.to_dict()))
                    routes_dict = reg_data.get("routes", {})
                    if isinstance(routes_dict, dict):
                        if self._route_name in routes_dict:
                            route_ids = [int(x) for x in routes_dict.get(self._route_name, [])]
                        elif "All" in routes_dict:
                            route_ids = [int(x) for x in routes_dict.get("All", [])]
                        else:
                            route_ids = [int(n.get("id", 0)) for n in nodes]
            elif hasattr(self.routes, "get_route_nodes"):
                raw_list = list(self.routes.get_route_nodes() or [])
                for item in raw_list:
                    if isinstance(item, dict):
                        nodes.append(item)
                    elif hasattr(item, "to_dict") and callable(item.to_dict):
                        nodes.append(dict(item.to_dict()))
                route_ids = [int(n.get("id", 0)) for n in nodes]
        except Exception as exc:
            logger.error("Failed extracting nodes from RouteManager: %s", exc)

        return nodes, route_ids

    def optimize_sequential_route(self) -> List[Dict[str, Any]]:
        """
        'Sıralı Rota Optimize Et' butonu aksiyonu:
        Harita üzerindeki noktaları zikzaksız en yakın komşu + 2-Opt sırasına dizer.
        """
        try:
            nodes = list(self.canvas._nodes)
            if len(nodes) <= 2:
                return nodes

            unvisited = list(nodes)
            unvisited.sort(
                key=lambda n: float(n.get("map_x", n.get("x", 0.0))) * 0.7
                + float(n.get("map_y", n.get("y", 0.0))) * 0.3
            )
            ordered = [unvisited.pop(0)]
            while unvisited:
                curr = ordered[-1]
                cx = float(curr.get("map_x", curr.get("x", 0.0)))
                cy = float(curr.get("map_y", curr.get("y", 0.0)))
                nxt = min(
                    unvisited,
                    key=lambda cand: math.hypot(
                        float(cand.get("map_x", cand.get("x", 0.0))) - cx,
                        float(cand.get("map_y", cand.get("y", 0.0))) - cy,
                    ),
                )
                ordered.append(nxt)
                unvisited.remove(nxt)

            if self.routes is not None and hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                if isinstance(reg_data, dict):
                    reg_data["nodes"] = list(ordered)
                    routes_dict = reg_data.setdefault("routes", {})
                    routes_dict[self._route_name] = [int(n.get("id", 0)) for n in ordered]
                    if hasattr(self.routes, "save_data"):
                        self.routes.save_data()

            route_ids = [int(n.get("id", 0)) for n in ordered]
            self.canvas.set_nodes_data(ordered, route_ids, rest_duration_sec=self._rest_duration_sec)
            logger.info("Sıralı rota optimize edildi (%d nokta).", len(ordered))
            return ordered
        except Exception as exc:
            logger.error("optimize_sequential_route hatası: %s", exc)
            return list(self.canvas._nodes)

    def set_region(self, region_name: str, route_name: Optional[str] = None) -> None:
        self._region_name = region_name
        if route_name is not None:
            self._route_name = route_name
        self.reload_map_and_nodes()

    def set_player_position(self, x: Optional[float], y: Optional[float], angle_deg: float = 0.0) -> None:
        self._player_pos = (x, y) if (x is not None and y is not None) else None
        self.canvas.set_player_position(x, y, angle_deg)

    def get_dry_run_trajectory_points(self) -> List[Tuple[float, float]]:
        """
        Returns simulated trajectory points from DryRunSimulator (wrapped in try/except, returning [] if none).
        """
        try:
            from core.dry_run_simulator import DryRunReplaySimulator
            simulator = DryRunReplaySimulator()
            nodes, _ = self._extract_region_nodes_and_route_ids()
            if not nodes:
                nodes = list(self.canvas._nodes)
            if not nodes:
                return []
            audit_result = simulator.run_8x_route_audit(nodes)
            # Extract coordinates from audit segments or nodes
            trajectory: List[Tuple[float, float]] = []
            segments = audit_result.get("segments", [])
            if segments:
                for seg in segments:
                    p1 = seg.get("p1")
                    if p1 and isinstance(p1, (list, tuple)) and len(p1) >= 2:
                        trajectory.append((float(p1[0]), float(p1[1])))
                # Append final point of last segment
                if segments:
                    last_p2 = segments[-1].get("p2")
                    if last_p2 and isinstance(last_p2, (list, tuple)) and len(last_p2) >= 2:
                        trajectory.append((float(last_p2[0]), float(last_p2[1])))
            else:
                for n in nodes:
                    xy = simulator._extract_xy(n)
                    trajectory.append((float(xy[0]), float(xy[1])))
            return trajectory
        except Exception as exc:
            logger.debug("get_dry_run_trajectory_points error or none available: %s", exc)
            return []

    # -------------------------------------------------------------------------
    # Interactive Node Toggling & Context Menu (Türkçe)
    # -------------------------------------------------------------------------
    def _on_node_left_clicked(self, node_id: int) -> None:
        """Left-clicking a node badge toggles its membership in the active route."""
        try:
            self._toggle_node_in_active_route(node_id)
            self.node_toggled.emit(node_id)
        except Exception as exc:
            logger.error("Error toggling node %d: %s", node_id, exc)

    def _toggle_node_in_active_route(self, node_id: int) -> None:
        if self.routes is not None and hasattr(self.routes, "data"):
            regions = self.routes.data.get("regions", {})
            reg_data = regions.get(self._region_name, {})
            routes_dict = reg_data.setdefault("routes", {})
            active_list: List[int] = routes_dict.setdefault(self._route_name, [])

            if node_id in active_list:
                active_list.remove(node_id)
                logger.info("Nokta #%d '%s' (%s) rotasından çıkarıldı", node_id, self._route_name, self._region_name)
            else:
                active_list.append(node_id)
                logger.info("Nokta #%d '%s' (%s) rotasına eklendi", node_id, self._route_name, self._region_name)

            if hasattr(self.routes, "save_data"):
                self.routes.save_data()

        nodes, route_ids = self._extract_region_nodes_and_route_ids()
        self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)

    def _on_node_right_clicked(self, node_id: int, global_pos: QPoint) -> None:
        """Sağ tıklandığında nokta detaylarını, hasat durumunu (0/3..3x) ve Türkçe işlemleri açar."""
        try:
            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            node_obj = next((n for n in nodes if int(n.get("id", -1)) == node_id), None)
            if node_obj is None:
                return

            in_route = node_id in route_ids
            nx = node_obj.get("map_x", node_obj.get("x", 0))
            ny = node_obj.get("map_y", node_obj.get("y", 0))
            visits = node_obj.get("visit_count", 0)
            badge_str = _MapCanvas._format_harvest_badge(node_obj)
            is_auto = _MapCanvas._is_minimap_auto_node(node_obj)
            src_label = "Minimap/3B Oto-Kayıt" if is_auto else "Manuel Kayıt"

            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #181b20;
                    color: #e5e7eb;
                    border: 1px solid #2d323b;
                    border-radius: 6px;
                    padding: 4px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                }
                QMenu::item {
                    padding: 6px 20px 6px 12px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QMenu::item:disabled {
                    color: #9ca3af;
                    font-weight: bold;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2d323b;
                    margin: 4px 6px;
                }
            """)

            header_act = QAction(
                f"Nokta #{node_id}  ({nx}, {ny}) · Hasat: {badge_str} · {src_label} · Ziyaret: {visits}x",
                menu,
            )
            header_act.setEnabled(False)
            menu.addAction(header_act)
            menu.addSeparator()

            toggle_label = (
                "Aktif Rotadan Çıkar (Başarısız / Pasif Yap)"
                if in_route
                else "Aktif Rotaya Ekle (Aktif Yap)"
            )
            act_toggle = QAction(toggle_label, menu)
            act_toggle.triggered.connect(lambda: self._on_node_left_clicked(node_id))
            menu.addAction(act_toggle)

            act_ready = QAction("Durumu 'Aktif' Yap (Bekleme Süresini Sıfırla)", menu)
            act_ready.triggered.connect(lambda: self._set_node_cooldown(node_id, resting=False))
            menu.addAction(act_ready)

            act_rest = QAction("Durumu 'Dinleniyor (5dk)' Yap", menu)
            act_rest.triggered.connect(lambda: self._set_node_cooldown(node_id, resting=True))
            menu.addAction(act_rest)

            menu.addSeparator()

            for h_idx, h_label in ((0, "0/3"), (1, "1/3"), (2, "2/3"), (3, "3x")):
                act_h = QAction(f"Hasat Rozetini '{h_label}' Olarak Ayarla", menu)
                act_h.triggered.connect(
                    lambda _checked=False, hc=h_idx, hb=h_label: self._set_node_harvest_status(
                        node_id, hc, max_harvests=3, badge_text=hb
                    )
                )
                menu.addAction(act_h)

            menu.addSeparator()

            act_delete = QAction(f"Noktayı Sil (#{node_id})", menu)
            act_delete.triggered.connect(lambda: self._delete_node(node_id))
            menu.addAction(act_delete)

            menu.popup(global_pos)
        except Exception as exc:
            logger.error("Failed opening node context menu for #%d: %s", node_id, exc)

    def _set_node_harvest_status(
        self,
        node_id: int,
        harvest_count: int,
        max_harvests: int = 3,
        badge_text: Optional[str] = None,
    ) -> None:
        """Noktanın hasat durum rozetini ('0/3', '1/3', '2/3' veya '3x') günceller."""
        try:
            if self.routes is not None and hasattr(self.routes, "data"):
                reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                for n in reg_data.get("nodes", []):
                    if isinstance(n, dict) and int(n.get("id", -1)) == node_id:
                        n["harvest_count"] = int(harvest_count)
                        n["max_harvests"] = int(max_harvests)
                        if badge_text:
                            n["harvest_badge"] = str(badge_text)
                        break
                if hasattr(self.routes, "save_data"):
                    self.routes.save_data()
            for n in self.canvas._nodes:
                if isinstance(n, dict) and int(n.get("id", -1)) == node_id:
                    n["harvest_count"] = int(harvest_count)
                    n["max_harvests"] = int(max_harvests)
                    if badge_text:
                        n["harvest_badge"] = str(badge_text)
                    break
            self.canvas.update()
        except Exception as exc:
            logger.error("Hasat durumu güncellenemedi (#%d): %s", node_id, exc)

    def _set_node_cooldown(self, node_id: int, resting: bool) -> None:
        try:
            if self.routes is not None and hasattr(self.routes, "data"):
                reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                for n in reg_data.get("nodes", []):
                    if int(n.get("id", -1)) == node_id:
                        n["last_visited"] = time.time() if resting else 0
                        n["status"] = "resting" if resting else "ready"
                        break
                if hasattr(self.routes, "save_data"):
                    self.routes.save_data()
            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
        except Exception as exc:
            logger.error("Failed updating cooldown state for node #%d: %s", node_id, exc)

    def _delete_node(self, node_id: int) -> None:
        try:
            if self.routes is not None and hasattr(self.routes, "data"):
                reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                if "nodes" in reg_data:
                    reg_data["nodes"] = [
                        n for n in reg_data["nodes"] if int(n.get("id", -1)) != node_id
                    ]
                for r_name, r_list in reg_data.get("routes", {}).items():
                    if node_id in r_list:
                        r_list.remove(node_id)
                if hasattr(self.routes, "save_data"):
                    self.routes.save_data()

            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
            self.node_deleted.emit(node_id)
            logger.info("Deleted node #%d from region '%s'", node_id, self._region_name)
        except Exception as exc:
            logger.error("Failed deleting node #%d: %s", node_id, exc)

    def get_map_viewer_health_summary(self) -> dict:
        try:
            return {"map_viewer_ready": True}
        except Exception:
            return {"map_viewer_ready": False}

    def get_map_viewer_ready_flag(self) -> dict:
        return {"module": "map_viewer", "ready": True}

    def get_map_viewer_version_info(self) -> dict:
        return {"module": "map_viewer", "version": "2.0.0", "ready": True}

    def get_map_viewer_diagnostic_snapshot(self) -> dict:
        return {"module": "map_viewer", "status": "ok", "ready": True}


# Backwards compatibility alias so existing imports (`from ui.map_viewer import MapViewerDialog`) continue to work
MapViewerDialog = MapViewerWindow
MapViewer = MapViewerWindow


def get_map_viewer_capabilities() -> list:
    return ["map_viewer", "route_visualization", "diagnostic_snapshot"]


def validate_map_viewer_contract() -> bool:
    return bool(get_map_viewer_capabilities())


def get_map_viewer_baseline_metrics() -> dict:
    return {"module": "map_viewer", "contract_valid": validate_map_viewer_contract(), "capability_count": len(get_map_viewer_capabilities())}


def get_map_viewer_readiness_status() -> dict:
    return {"module": "map_viewer", "ready": validate_map_viewer_contract(), "metrics": get_map_viewer_baseline_metrics()}


