# -*- coding: utf-8 -*-
"""
YOLO Preview and Panel Architecture for Gatherion.

Provides:
  - YOLOPreviewDialog: Modal responsive lightbox dialog displaying image frames
    with high-contrast bounding boxes, class names, confidence scores, and pill badges.
  - YOLOPreviewCanvasWidget: Aspect-ratio letterboxed interactive drawing canvas.
  - render_annotated_pixmap: Headless/standalone pixmap annotation baking utility.
  - normalize_annotation: Robust converter for dict, tuple, and YOLO box formats.
  - Clean re-exports of YOLOFactoryDialog, YOLOPreviewCanvas, create_yolo_factory_page.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
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
    QHBoxLayout,
    QLabel,
    QListView,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.shell_theme import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_TIER_COLORS,
    apply_combobox_dark_style,
    apply_combobox_white_text_style,
    style_combobox_dark_popup,
    style_combobox_dark_view as shell_style_combobox_dark_view,
)


def style_combobox_dark_view(combo: Optional[QComboBox]) -> None:
    """Apply the dark combobox style.

    Single source of truth is ``ui.shell_theme.style_combobox_dark_view``
    (imported above as ``shell_style_combobox_dark_view``); this wrapper only
    exists so existing ``ui.yolo_panel.style_combobox_dark_view`` callers keep
    working. Previously this file carried a ~100-line copy that drifted in
    colour from the shell one (F-0778).
    """
    shell_style_combobox_dark_view(combo)


logger = logging.getLogger("Gatherion.YOLOPanel")

# Class ID to Name mapping — canonical 27 classes (0..26), single authority
# dataset_factory/class_taxonomy.py CANONICAL_NAMES (Decision 1 / S-1801).
# Previously this was a stale 25-class map (0..24): 6/7/8/12 carried legacy
# names (world_gather_crest, f_gather_prompt, f_press_key,
# feather_particle_negative) and 25 kinah / 26 player_character were missing.
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "herb_aria",
    1: "herb_mela",
    2: "herb_targena",
    3: "ore_orichalcum",
    4: "gem_crystal",
    5: "tree_log",
    6: "herb_edana",
    7: "herb_lumina",
    8: "ore_obsidian",
    9: "minimap_sprout",
    10: "monster_3d",
    11: "minimap_monster_dot",
    12: "crystal_blood",
    13: "skill_ui_prompt",
    14: "obstacle",
    15: "path_ground",
    16: "player_hp_bar",
    17: "player_mp_bar",
    18: "target_hp_bar",
    19: "skill_bar",
    20: "skill_slot_ready",
    21: "skill_slot_cooldown",
    22: "hp_potion",
    23: "mp_potion",
    24: "buff_scroll",
    25: "kinah",
    26: "player_character",
}

# Confidence tiers — P3-7 (v18): tek kaynak ui.shell_theme (o da
# dataset_factory/confidence_tier_coloring'dan okur; test-pinned:
# tests/test_w27_three_tier_box_colors.py). Orta eşik 0.70'tir — bu modül
# eskiden 0.60 kullanıyordu, web JS (paintBoxes) ile birlikte; o yüzden
# 0.60-0.69 güven bir yerde "şüpheli", fabrika tarafında "düşük" sayılıyordu.
# CONFIDENCE_HIGH_THRESHOLD / CONFIDENCE_MEDIUM_THRESHOLD / CONFIDENCE_TIER_COLORS
# yukarıdaki ui.shell_theme importundan gelir; burada yeniden tanımlanmaz.

# High-contrast color palette categorized by entity type
CLASS_COLORS: Dict[str, str] = {
    # Herbs & Gatherables (Vibrant Greens)
    "herb_aria": "#22c55e",          # Emerald Green
    "herb_mela": "#10b981",          # Teal Green
    "herb_targena": "#14b8a6",       # Aqua Green
    "herb_edana": "#a855f7",         # Violet (pink/purple herb, S-1801)
    "herb_lumina": "#facc15",        # Gold (yellow/golden herb, S-1801)
    # Ores, Gems, Trees (Amber / Magenta / Lime)
    "ore_orichalcum": "#f59e0b",     # Amber Gold
    "ore_obsidian": "#334155",       # Dark Slate (obsidian, S-1801)
    "gem_crystal": "#ec4899",        # Crystal Pink
    "crystal_blood": "#dc2626",      # Blood Red crystal (S-1801)
    "tree_log": "#84cc16",           # Lime Wood
    # Currency & Player (S-1801)
    "kinah": "#fbbf24",              # Kinah Gold
    "player_character": "#38bdf8",   # Sky Blue player
    # Prompts, Crests & Minimap (Sky Blues & Lime)
    # Legacy keys below (world_gather_crest, f_gather_prompt, f_press_key,
    # feather_particle_negative) are no longer canonical class names (S-1801)
    # but are kept so any caller passing the old name still gets a color.
    "world_gather_crest": "#38bdf8", # Sky Blue
    "feather_particle_negative": "#e2e8f0", # Pale feather particle
    "f_gather_prompt": "#06b6d4",    # Cyan
    "f_press_key": "#0284c7",        # Deep Blue
    "minimap_sprout": "#4ade80",     # Bright Lime Sprout
    "minimap_monster_dot": "#f43f5e",# Crimson Dot
    # Combat & Hostile Entities (Red / Orange)
    "monster_3d": "#ef4444",         # Red
    "target_hp_bar": "#f87171",      # Light Red
    "mob_nameplate": "#fb923c",      # Orange
    # HUD Elements & Skills (Violets & Blues)
    "player_hp_bar": "#e11d48",      # Rose Red
    "player_mp_bar": "#3b82f6",      # Sapphire Blue
    "skill_ui_prompt": "#8b5cf6",    # Violet
    "skill_bar": "#a855f7",          # Purple
    "skill_slot_ready": "#22c55e",   # Ready Green
    "skill_slot_cooldown": "#94a3b8",# Cooldown Slate Gray
    "hp_potion": "#fb7185",          # Rose Potion
    "mp_potion": "#60a5fa",          # Mana Potion Blue
    "buff_scroll": "#facc15",        # Gold Buff Scroll
    # Navigation & Ground Paths (Yellow / Aqua)
    "obstacle": "#eab308",           # Amber Yellow
    "path_ground": "#06b6d4",        # Cyan Path
}

DEFAULT_BOX_COLOR: str = "#38bdf8"


def get_annotation_tier_summary(annotations: Sequence[Any]) -> Dict[str, Any]:
    """
    Summarizes annotations into the 3 confidence tiers:
      - confident (>= 0.85): green (#22c55e)
      - suspicious (0.70 .. < 0.85): orange (#fb923c)
      - low (< 0.70): red (#ef4444)

    An annotation with no confidence value at all is never "confident":
    normalize_annotation() reports unknown confidence as 0.0, and the tier
    below additionally treats a missing confidence key as 0.0 (F-0613).
    """
    confident = 0
    suspicious = 0
    low = 0
    for raw in annotations or []:
        norm = normalize_annotation(raw)
        # P2-23 (Y002-K1): geometrisi eksik kutu sayılmaz — şablon
        # geometriyle çizilmediği gibi tier sayacına da girmez.
        if norm.get("valid") is False:
            continue
        conf = float(norm.get("confidence", 0.0) or 0.0)
        if isinstance(raw, dict) and "confidence" not in raw and "conf" not in raw:
            conf = 0.0
        elif isinstance(raw, (list, tuple)) and len(raw) < 6:
            conf = 0.0
        if conf >= CONFIDENCE_HIGH_THRESHOLD and bool(norm.get("trusted", False)):
            confident += 1
        elif conf >= CONFIDENCE_MEDIUM_THRESHOLD:
            suspicious += 1
        else:
            low += 1
    return {
        "total": confident + suspicious + low,
        "confident": confident,
        "suspicious": suspicious,
        "low": low,
        # P3-7: renkler tek kaynaktan (ui.shell_theme.CONFIDENCE_TIER_COLORS).
        "colors": dict(CONFIDENCE_TIER_COLORS),
    }


def normalize_annotation(ann: Any) -> Dict[str, Any]:
    """
    Normalizes varied detection data structures (dict, tuple, list, YOLO vector)
    into a standardized dictionary:
      {
        'cx': float, 'cy': float, 'w': float, 'h': float,
        'class_name': str, 'confidence': float, 'trusted': bool, 'color': str
      }
    """
    if isinstance(ann, dict):
        class_id = ann.get("class_id")
        class_name = ann.get("class_name") or ann.get("key") or ann.get("name")
        if not class_name and isinstance(class_id, int):
            class_name = CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")
        class_name = str(class_name or "target")

        conf = ann.get("confidence", ann.get("conf", 0.0))
        try:
            conf_val = float(conf if conf is not None else 0.0)
            if not math.isfinite(conf_val):
                conf_val = 0.0
            # Clamp to [0, 1] (S-1802): out-of-range values such as 5.0 or
            # -0.5 previously passed through, counted as 'confident' in the
            # tier summary and rendered as "%5" on the overlay badge.
            conf_val = min(1.0, max(0.0, conf_val))
        except (ValueError, TypeError):
            conf_val = 0.0

        trusted = ann.get("trusted")
        # P2-23 (Y002-K1): `trusted` alanı olmayan kutu varsayılan
        # `trusted=False` sayılır. Eskiden alan yoksa `weak` bayrağına
        # bakılıp (yoksa True) varsayılıyordu; bozuk girdi böylece
        # "emin" sayılıyor ve aşağıda uydurma geometriyle görüntünün
        # ortasına %8'lik "target" hayalet kutusu çiziliyordu.
        if trusted is None:
            trusted = False
        else:
            trusted = bool(trusted)

        color = ann.get("color") or CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)

        # P2-23 (Y002-K1): geometrisi eksik kutuya şablon geometri
        # (cx/cy=0.5, w/h=0.08) uydurulmaz — öyle kutu `valid=False`
        # ile işaretlenir; çizilmez, tier sayacına girmez.
        geom_keys = ("cx", "cy", "w", "h")
        has_geometry = all(k in ann and ann.get(k) is not None for k in geom_keys)
        valid = bool(has_geometry)
        try:
            raw_cx = ann.get("cx", 0.5)
            raw_cy = ann.get("cy", 0.5)
            raw_w = ann.get("w", 0.08)
            raw_h = ann.get("h", 0.08)
            cx = float(raw_cx if raw_cx is not None else 0.5)
            cy = float(raw_cy if raw_cy is not None else 0.5)
            w = float(raw_w if raw_w is not None else 0.08)
            h = float(raw_h if raw_h is not None else 0.08)
            if not math.isfinite(cx): cx = 0.5
            if not math.isfinite(cy): cy = 0.5
            if not math.isfinite(w): w = 0.08
            if not math.isfinite(h): h = 0.08
            if not has_geometry or w <= 0 or h <= 0:
                valid = False
        except (ValueError, TypeError):
            cx, cy, w, h = 0.5, 0.5, 0.08, 0.08
            valid = False

        return {
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "class_name": class_name,
            "confidence": conf_val,
            "trusted": trusted,
            "color": str(color),
            "valid": valid,
        }

    if isinstance(ann, (list, tuple)):
        n = len(ann)
        cx, cy, w, h = 0.5, 0.5, 0.08, 0.08
        class_name = "target"
        conf_val = 0.0
        # P2-23 (Y002-K1): alan yoksa trusted=False; geometrisiz
        # tuple (n<4) hayalet kutu olarak çizilmez (valid=False).
        trusted = False
        valid = False

        try:
            if n >= 6 and isinstance(ann[0], int) and isinstance(ann[1], (int, float)):
                # YOLO format: (class_id, cx, cy, w, h, conf)
                cid = int(ann[0])
                class_name = CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
                try:
                    cx = float(ann[1])
                    cy = float(ann[2])
                    w = float(ann[3])
                    h = float(ann[4])
                    conf_val = float(ann[5])
                    valid = True
                except (ValueError, TypeError, IndexError):
                    cx, cy, w, h = 0.5, 0.5, 0.08, 0.08
                    conf_val = 0.0
            elif n >= 4:
                try:
                    cx = float(ann[0])
                    cy = float(ann[1])
                    w = float(ann[2])
                    h = float(ann[3])
                    valid = True
                except (ValueError, TypeError, IndexError):
                    cx, cy, w, h = 0.5, 0.5, 0.08, 0.08
                if n >= 5:
                    raw_c = ann[4]
                    if isinstance(raw_c, int):
                        class_name = CLASS_ID_TO_NAME.get(raw_c, f"class_{raw_c}")
                    else:
                        class_name = str(raw_c)
                if n >= 6:
                    try:
                        conf_val = float(ann[5])
                    except (ValueError, TypeError, IndexError):
                        conf_val = 0.0
                if n >= 7:
                    trusted = bool(ann[6])
        except Exception:
            cx, cy, w, h = 0.5, 0.5, 0.08, 0.08
            conf_val = 0.0
            valid = False

        if not math.isfinite(cx): cx = 0.5
        if not math.isfinite(cy): cy = 0.5
        if not math.isfinite(w): w = 0.08
        if not math.isfinite(h): h = 0.08
        if not math.isfinite(conf_val): conf_val = 0.0
        if w <= 0 or h <= 0:
            valid = False
        # Clamp to [0, 1] (S-1802) — same as the dict path above.
        conf_val = min(1.0, max(0.0, conf_val))

        color = CLASS_COLORS.get(class_name, DEFAULT_BOX_COLOR)
        return {
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "class_name": class_name,
            "confidence": conf_val,
            "trusted": trusted,
            "color": color,
            "valid": valid,
        }

    return {
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.08,
        "h": 0.08,
        "class_name": "target",
        "confidence": 0.0,
        "trusted": False,
        "color": DEFAULT_BOX_COLOR,
        "valid": False,
    }


def draw_annotation_overlay(
    painter: QPainter,
    target_rect: QRectF,
    annotations: Sequence[Any],
    show_labels: bool = True,
    show_close_buttons: bool = False,
    hover_close_idx: int = -1,
    close_hit_rects_out: Optional[List[Tuple[QRectF, Any, int]]] = None,
) -> None:
    """
    Renders high-contrast bounding boxes, translucent fills, and pill badges
    accurately projected onto target_rect with boundary clamping.
    """
    if not annotations:
        return

    dw = target_rect.width()
    dh = target_rect.height()
    if dw <= 0 or dh <= 0:
        return

    # Dynamic badge font proportional to target viewport height
    font_pt = max(10, min(15, int(dh * 0.022)))
    tag_font = QFont("Segoe UI", font_pt, QFont.Weight.Bold)
    close_font = QFont("Segoe UI", max(9, font_pt - 1), QFont.Weight.Black)
    painter.setFont(tag_font)
    fm = QFontMetrics(tag_font)

    for idx, raw in enumerate(annotations):
        ann = normalize_annotation(raw)
        # P2-23 (Y002-K1): geometrisi eksik kutu çizilmez — şablon
        # (0.5/0.5/0.08) "target" hayalet kutu yok; ✕ vuruş alanı da
        # oluşmaz, silme için eşleşecek kutu yoktur.
        if ann.get("valid") is False:
            continue
        try:
            cx = float(ann.get("cx", 0.5))
            cy = float(ann.get("cy", 0.5))
            w = float(ann.get("w", 0.08))
            h = float(ann.get("h", 0.08))
            if not math.isfinite(cx): cx = 0.5
            if not math.isfinite(cy): cy = 0.5
            if not math.isfinite(w): w = 0.08
            if not math.isfinite(h): h = 0.08
        except (ValueError, TypeError):
            cx, cy, w, h = 0.5, 0.5, 0.08, 0.08

        cname = str(ann.get("class_name", "target"))
        try:
            raw_c = ann.get("confidence", 0.0)
            conf = float(raw_c if raw_c is not None else 0.0)
            if not math.isfinite(conf):
                conf = 0.0
            # Defense-in-depth (S-1802): normalize_annotation already clamps
            # to [0, 1], but clamp again here so a hand-built ann dict can
            # never render a "%5" / "%-50" badge.
            conf = min(1.0, max(0.0, conf))
        except (ValueError, TypeError):
            conf = 0.0

        # P2-23 (Y002-K1): `trusted` alanı yoksa False sayılır
        # (normalize_annotation zaten böyle üretir; burası savunma).
        is_trusted = bool(ann.get("trusted", False))
        color_hex = str(ann.get("color") or CLASS_COLORS.get(cname, DEFAULT_BOX_COLOR))

        # Calculate bounding box in pixel coordinates within target_rect
        bx = target_rect.left() + (cx - w * 0.5) * dw
        by = target_rect.top() + (cy - h * 0.5) * dh
        bw_px = max(18.0, w * dw)
        bh_px = max(14.0, h * dh)

        # Clamping to target_rect bounds
        if bx < target_rect.left():
            bw_px = max(6.0, bw_px - (target_rect.left() - bx))
            bx = target_rect.left()
        if by < target_rect.top():
            bh_px = max(6.0, bh_px - (target_rect.top() - by))
            by = target_rect.top()
        if bx + bw_px > target_rect.right():
            bw_px = max(6.0, target_rect.right() - bx)
        if by + bh_px > target_rect.bottom():
            bh_px = max(6.0, target_rect.bottom() - by)

        box_rect = QRectF(bx, by, bw_px, bh_px)
        base_col = QColor(color_hex)

        # 1. Bounding box & interior tint
        fill_col = QColor(base_col)
        fill_col.setAlpha(36)
        pen_width = 2.4 if is_trusted else 1.8
        pen = QPen(base_col, pen_width)
        if not is_trusted:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_col))
        painter.drawRoundedRect(box_rect, 4.0, 4.0)

        # 2. Pill badge with class name and percentage score
        if show_labels:
            painter.setFont(tag_font)
            prefix = "" if is_trusted else "şüpheli "
            if math.isfinite(conf):
                try:
                    # conf is clamped to [0, 1] in normalize_annotation
                    # (S-1802), so this is always a 0..100 percentage.
                    pct = int(round(conf * 100))
                except (ValueError, OverflowError):
                    pct = 0
            else:
                pct = 0
            label_str = f"{prefix}{cname} %{pct}"

            pad_x = 8.0
            pad_y = 4.0
            text_w = float(fm.horizontalAdvance(label_str))
            text_h = float(fm.height())
            close_w = (text_h + 4.0) if show_close_buttons else 0.0
            badge_w = text_w + pad_x * 2.0 + (close_w + 2.0 if show_close_buttons else 0.0)
            badge_h = text_h + pad_y

            # Position badge directly above box, or clamp inside top if box is at upper edge
            if by - badge_h - 2.0 >= target_rect.top():
                ly = by - badge_h - 2.0
            else:
                ly = by + 2.0

            lx = max(target_rect.left(), min(bx, target_rect.right() - badge_w))
            badge_rect = QRectF(lx, ly, badge_w, badge_h)

            # Draw background badge pill
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(base_col))
            painter.drawRoundedRect(badge_rect, 4.0, 4.0)

            # Draw high-contrast text adapted to badge luminance
            lum = 0.299 * base_col.red() + 0.587 * base_col.green() + 0.114 * base_col.blue()
            text_col = QColor("#090d16") if lum > 125 else QColor("#ffffff")
            painter.setPen(text_col)
            if show_close_buttons:
                text_rect = QRectF(lx + pad_x, ly, text_w + 2.0, badge_h)
                painter.drawText(
                    text_rect,
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    label_str,
                )
                close_rect = QRectF(lx + badge_w - close_w - 2.0, ly + 2.0, close_w - 1.0, badge_h - 4.0)
                is_hovered = (hover_close_idx == idx)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#ef4444") if is_hovered else QColor(15, 23, 42, 195)))
                painter.drawRoundedRect(close_rect, 3.0, 3.0)
                painter.setFont(close_font)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")
                if close_hit_rects_out is not None:
                    close_hit_rects_out.append((close_rect.adjusted(-6.0, -6.0, 6.0, 6.0), raw, idx))
            else:
                painter.drawText(
                    badge_rect,
                    int(Qt.AlignmentFlag.AlignCenter),
                    label_str,
                )


def render_annotated_pixmap(
    source: QPixmap,
    annotations: Sequence[Any],
    show_labels: bool = True,
) -> QPixmap:
    """
    Bakes bounding boxes, class names, and confidence scores directly onto a copy
    of the given QPixmap. Returns the annotated QPixmap.
    """
    if source is None or source.isNull():
        return QPixmap()

    out = source.copy()
    if not annotations:
        return out

    painter = QPainter(out)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        rect = QRectF(0.0, 0.0, float(out.width()), float(out.height()))
        draw_annotation_overlay(painter, rect, annotations, show_labels=show_labels)
    finally:
        painter.end()

    return out


class YOLOPreviewCanvasWidget(QWidget):
    """
    Aspect-ratio letterboxed image canvas that dynamically paints the source
    pixmap and overlays normalized YOLO annotations accurately across any widget size.
    Supports clickable '✕' close buttons on each annotation badge pill.
    """

    annotationRemoveRequested = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._src: Optional[QPixmap] = None
        self._annotations: List[Any] = []
        self._show_annotations: bool = True
        self._close_hit_rects: List[Tuple[QRectF, Any, int]] = []
        self._hover_close_idx: int = -1
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #0b0e14;")
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_content(self, pixmap: Optional[QPixmap], annotations: Optional[Sequence[Any]] = None) -> None:
        self._src = pixmap
        self._annotations = list(annotations) if annotations is not None else []
        self._hover_close_idx = -1
        self.update()

    def set_show_annotations(self, show: bool) -> None:
        self._show_annotations = bool(show)
        self.update()

    def hit_test_close_button(self, pos: QPointF) -> Optional[Any]:
        for hit_rect, raw_ann, _idx in reversed(self._close_hit_rects):
            if hit_rect.contains(pos):
                return raw_ann
        return None

    def _hit_test_close_index(self, pos: QPointF) -> int:
        for hit_rect, _raw_ann, idx in reversed(self._close_hit_rects):
            if hit_rect.contains(pos):
                return idx
        return -1

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        idx = self._hit_test_close_index(pos)
        if idx != self._hover_close_idx:
            self._hover_close_idx = idx
            self.setToolTip("Etiketi kaldır (✕)" if idx >= 0 else "")
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        if self._hover_close_idx != -1:
            self._hover_close_idx = -1
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            hit_ann = self.hit_test_close_button(pos)
            if hit_ann is not None:
                self.annotationRemoveRequested.emit(hit_ann)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            vw = float(self.width())
            vh = float(self.height())

            painter.fillRect(self.rect(), QColor("#0b0e14"))
            self._close_hit_rects = []

            if self._src is None or self._src.isNull():
                painter.setPen(QColor("#64748b"))
                painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Medium))
                painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Görüntü yok")
                return

            pw = float(self._src.width())
            ph = float(self._src.height())
            if pw <= 0 or ph <= 0:
                return

            scale = min(vw / pw, vh / ph)
            dw = pw * scale
            dh = ph * scale
            dx = (vw - dw) * 0.5
            dy = (vh - dh) * 0.5
            target_rect = QRectF(dx, dy, dw, dh)

            scaled_pix = self._src.scaled(
                int(dw),
                int(dh),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(int(dx), int(dy), scaled_pix)

            # Outer subtle border around image
            painter.setPen(QPen(QColor("#222d3d"), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target_rect)

            # Overlay bounding boxes and badges with clickable '✕' close buttons
            if self._show_annotations and self._annotations:
                draw_annotation_overlay(
                    painter,
                    target_rect,
                    self._annotations,
                    show_labels=True,
                    show_close_buttons=True,
                    hover_close_idx=self._hover_close_idx,
                    close_hit_rects_out=self._close_hit_rects,
                )
        finally:
            painter.end()


class YOLOPreviewDialog(QDialog):
    """
    Modal responsive lightbox dialog for inspecting YOLO frames and detections.
    Renders overlaid bounding boxes, class names, and confidence scores with
    support for keyboard shortcuts (ESC to close, V to toggle annotations)
    and one-click '✕' removal on any annotation label pill.
    """

    def __init__(
        self,
        source_pixmap: Optional[QPixmap],
        annotations: Optional[Sequence[Any]] = None,
        title: str = "YOLO Önizleme",
        meta: str = "",
        parent: Optional[QWidget] = None,
        on_remove_annotation: Optional[Callable[[Any], None]] = None,
    ) -> None:
        super().__init__(parent)
        self._on_remove_annotation = on_remove_annotation
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(960, 600)
        self.setStyleSheet(
            "QDialog {"
            " background-color: #0b0e14;"
            " color: #f2f3f5;"
            " border: 1px solid #1e2530;"
            "}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # Header bar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(12)

        meta_html = f" &nbsp;·&nbsp; <span style='color:#94a3b8; font-weight:normal;'>{meta}</span>" if meta else ""
        self.title_lbl = QLabel(f"<b>{title}</b>{meta_html}")
        self.title_lbl.setStyleSheet("color: #f8fafc; font-size: 15px;")
        top_bar.addWidget(self.title_lbl)
        top_bar.addStretch(1)

        # Class / category filter dropdown
        self._all_annotations: List[Any] = list(annotations) if annotations is not None else []
        self._filter_combo = QComboBox(self)
        self._filter_combo.setObjectName("YOLOFilterCombo")
        self._filter_combo.addItem("Tüm Tespitler (Hepsi)", "all")
        self._filter_combo.addItem("🟢 Sadece Emin Olunanlar", "trusted")
        self._filter_combo.addItem("🟠 Sadece Şüpheliler", "weak")

        # Discover unique classes present in current annotations
        # (P2-23: geometrisi eksik kutu çizilmez, sınıf listesine de girmez)
        seen_classes: set[str] = set()
        for raw in self._all_annotations:
            norm = normalize_annotation(raw)
            if norm.get("valid") is False:
                continue
            cname = norm.get("class_name")
            if cname and cname not in seen_classes:
                seen_classes.add(cname)
                self._filter_combo.addItem(f"🎯 {cname}", f"class:{cname}")

        self._filter_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._filter_combo.setToolTip("Tespitleri filtrele")
        style_combobox_dark_view(self._filter_combo)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        top_bar.addWidget(self._filter_combo)

        self.chk_annotations = QCheckBox("Etiketleri Göster [V]")
        self.chk_annotations.setChecked(True)
        self.chk_annotations.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_annotations.setStyleSheet(
            "QCheckBox { color: #A8B4C4; font-size: 13px; font-weight: 600; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #334155; background: #1e293b; }"
            "QCheckBox::indicator:checked { background: #38bdf8; border-color: #38bdf8; }"
        )
        self.chk_annotations.toggled.connect(self._on_toggle_annotations)
        top_bar.addWidget(self.chk_annotations)

        btn_close = QPushButton("Kapat (ESC)")
        btn_close.setFixedHeight(28)
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(
            "QPushButton {"
            " background: #1e2530; color: #f1f5f9; border: 1px solid #334155;"
            " border-radius: 6px; padding: 6px 16px; font-size: 13px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #334155; border-color: #475569; }"
            "QPushButton:pressed { background: #0f172a; }"
        )
        btn_close.clicked.connect(self.accept)
        top_bar.addWidget(btn_close)
        root.addLayout(top_bar)

        # Interactive canvas
        self.canvas = YOLOPreviewCanvasWidget(self)
        self.canvas.set_content(source_pixmap, annotations or [])
        self.canvas.annotationRemoveRequested.connect(self._on_canvas_remove_annotation)
        root.addWidget(self.canvas, stretch=1)

        # Footer info
        bottom_bar = QHBoxLayout()
        tip = QLabel("Etiket üzerindeki [✕] ile kutuyu sil &nbsp;·&nbsp; Kapatmak için ESC &nbsp;·&nbsp; [V] gizle/göster")
        tip.setStyleSheet("color: #7E8DA1; font-size: 12px;")
        tip.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bottom_bar.addWidget(tip)
        bottom_bar.addStretch(1)

        # P2-23: yalnızca çizilebilir (geometrili) kutular sayılır.
        box_count = sum(
            1
            for a in (annotations or [])
            if normalize_annotation(a).get("valid") is not False
        )
        self.count_lbl = QLabel(f"{box_count} tespit")
        self.count_lbl.setStyleSheet("color: #A8B4C4; font-size: 12px; font-weight: 600;")
        bottom_bar.addWidget(self.count_lbl)
        root.addLayout(bottom_bar)

    def _rebuild_filter_combo(self) -> None:
        """Rebuild the class filter items from the current annotations (F-0779).

        Previously the class entries were only built in __init__, so after an
        annotation was deleted its class stayed in the dropdown (stale filter
        matching nothing). Preserves the current selection when it still exists.
        """
        previous = self._filter_combo.currentData() if hasattr(self, "_filter_combo") else "all"
        self._filter_combo.blockSignals(True)
        try:
            self._filter_combo.clear()
            self._filter_combo.addItem("Tüm Tespitler (Hepsi)", "all")
            self._filter_combo.addItem("🟢 Sadece Emin Olunanlar", "trusted")
            self._filter_combo.addItem("🟠 Sadece Şüpheliler", "weak")
            seen_classes: set[str] = set()
            for raw in self._all_annotations:
                norm = normalize_annotation(raw)
                if norm.get("valid") is False:
                    continue
                cname = norm.get("class_name")
                if cname and cname not in seen_classes:
                    seen_classes.add(cname)
                    self._filter_combo.addItem(f"🎯 {cname}", f"class:{cname}")
            idx = self._filter_combo.findData(previous)
            self._filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._filter_combo.blockSignals(False)

    def _on_canvas_remove_annotation(self, target_ann: Any) -> None:
        # P2-23 (Y004-K1): silme önce diskte yapılır, UI sonra güncellenir.
        # Eskiden kutu önce `self._all_annotations`'tan (UI) çıkarılıp
        # canvas yeniden çiziliyor, callback hatası `logger.debug`'a
        # yutuluyordu — diskte duran kutu UI'da yok oluyordu (geri alma
        # yok). Callback (`YOLOFactoryDialog._reject_box`) bool döner:
        # True = diskten silindi, False = korumalı / bulunamadı /
        # diske kaydedilemedi. False/hatada kutu UI'da bırakılır ve
        # hata kullanıcıya gösterilir (yalnızca loga değil).
        removed = False
        if callable(self._on_remove_annotation):
            try:
                removed = bool(self._on_remove_annotation(target_ann))
            except Exception as exc:
                logger.warning("on_remove_annotation callback error: %s", exc)
                removed = False
        else:
            # Geri çağırma yok: silinecek disk yok, yerel kaldırma işlemin kendisi.
            removed = True
        if not removed:
            QMessageBox.warning(
                self,
                "Kutu silinemedi",
                "Kutu diskten silinemedi — diskteki etiket değişmedi, "
                "kutu önizlemede bırakıldı.",
            )
            return
        self._all_annotations = [a for a in self._all_annotations if a is not target_ann]
        self._rebuild_filter_combo()
        self._on_filter_changed()

    def _on_toggle_annotations(self, checked: bool) -> None:
        self.canvas.set_show_annotations(checked)

    def _on_filter_changed(self, index: int = 0) -> None:
        data = self._filter_combo.currentData() if hasattr(self, "_filter_combo") else "all"
        # P2-23: geometrisi eksik kutu hiçbir filtrede görünmez (çizilmez).
        source = [
            a for a in self._all_annotations
            if normalize_annotation(a).get("valid") is not False
        ]
        if not data or data == "all":
            filtered = list(source)
        elif data == "trusted":
            filtered = [a for a in source if normalize_annotation(a).get("trusted")]
        elif data == "weak":
            filtered = [a for a in source if not normalize_annotation(a).get("trusted")]
        elif str(data).startswith("class:"):
            target_class = str(data)[len("class:"):]
            filtered = [a for a in source if normalize_annotation(a).get("class_name") == target_class]
        else:
            filtered = list(source)

        self.canvas.set_content(self.canvas._src, filtered)
        if hasattr(self, "count_lbl"):
            self.count_lbl.setText(f"{len(filtered)} tespit")

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.accept()
            return
        if event.key() == Qt.Key.Key_V:
            self.chk_annotations.setChecked(not self.chk_annotations.isChecked())
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        super().mousePressEvent(event)


def _harden_factory_dialog_combos(factory_cls: Any) -> None:
    if not hasattr(factory_cls, "_yolo_combos_hardened"):
        orig_theme = getattr(factory_cls, "_apply_dark_studio_theme", None)
        if orig_theme:
            def _wrapped_theme(self: Any, *args: Any, **kwargs: Any) -> Any:
                res = orig_theme(self, *args, **kwargs)
                for cb in self.findChildren(QComboBox):
                    apply_combobox_dark_style(cb)
                    style_combobox_dark_view(cb)
                return res
            factory_cls._apply_dark_studio_theme = _wrapped_theme

        orig_build_ui = getattr(factory_cls, "_build_ui", None)
        if orig_build_ui:
            def _wrapped_build_ui(self: Any, *args: Any, **kwargs: Any) -> Any:
                res = orig_build_ui(self, *args, **kwargs)
                for cb in self.findChildren(QComboBox):
                    apply_combobox_dark_style(cb)
                    style_combobox_dark_view(cb)
                return res
            factory_cls._build_ui = _wrapped_build_ui

        orig_create_row = getattr(factory_cls, "_create_review_det_row", None)
        if orig_create_row:
            def _wrapped_create_row(self: Any, *args: Any, **kwargs: Any) -> Any:
                row = orig_create_row(self, *args, **kwargs)
                if row is not None:
                    for cb in row.findChildren(QComboBox):
                        apply_combobox_dark_style(cb)
                        style_combobox_dark_view(cb)
                return row
            factory_cls._create_review_det_row = _wrapped_create_row

        orig_show = getattr(factory_cls, "showEvent", None)
        def _wrapped_show(self: Any, event: Any) -> None:
            if orig_show:
                orig_show(self, event)
            for cb in self.findChildren(QComboBox):
                apply_combobox_dark_style(cb)
                style_combobox_dark_view(cb)
        factory_cls.showEvent = _wrapped_show
        factory_cls._yolo_combos_hardened = True


# Re-exports for clean modular imports.
# __all__ is built incrementally: a failed re-export must NOT stay listed
# (previously the except: pass swallowed the failure while __all__ still
# advertised the name — F-0702).
__all__ = [
    "CLASS_COLORS",
    "CLASS_ID_TO_NAME",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "DEFAULT_BOX_COLOR",
    "YOLOPreviewCanvasWidget",
    "YOLOPreviewDialog",
    "draw_annotation_overlay",
    "get_annotation_tier_summary",
    "normalize_annotation",
    "render_annotated_pixmap",
    "apply_combobox_dark_style",
    "apply_combobox_white_text_style",
    "style_combobox_dark_view",
]

try:
    from ui.yolo_factory_dialog import YOLOFactoryDialog, YOLOPreviewCanvas
    _harden_factory_dialog_combos(YOLOFactoryDialog)
    __all__ += ["YOLOFactoryDialog", "YOLOPreviewCanvas"]
except ImportError as exc:
    logger.debug("YOLOFactoryDialog re-export unavailable: %s", exc)

try:
    from ui.yolo_pipeline_web import create_yolo_factory_page
    __all__ += ["create_yolo_factory_page"]
except ImportError as exc:
    logger.debug("create_yolo_factory_page re-export unavailable: %s", exc)

