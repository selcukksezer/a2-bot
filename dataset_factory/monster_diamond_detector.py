# -*- coding: utf-8 -*-
"""dataset_factory/monster_diamond_detector.py
==============================================
3D Monster Nameplate, Turquoise Diamond Tip (`< Yellow-Maned Zait >`), and
Hostile Red Agro Badge Detector (`class_id = 10: monster_3d`).

Detects yellow/olive monster nameplates in both normal and horizontally flipped
(`hflip`) frames that feature:
1. Two flanking turquoise/cyan diamond tips (`< ... >`, e.g. `< Yellow-Maned Zait >`),
2. A red agro badge or hostile monster HP bar directly below the nameplate, or
3. High-contrast yellow/olive monster glyph strokes with a dark outline shadow.

Strictly excludes the center-screen player character zone (`0.42..0.58 X, 0.42..0.58 Y`)
so player overlays, self-buffs, or character accessories are never misclassified as monsters.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dataset_factory.class_taxonomy import CANONICAL_LABELS_TR, CANONICAL_NAMES, format_yolo_txt_line

logger = logging.getLogger(__name__)

MONSTER_CLASS_ID: int = 10
MONSTER_CLASS_NAME: str = CANONICAL_NAMES.get(10, "monster_3d")
MONSTER_CLASS_LABEL_TR: str = CANONICAL_LABELS_TR.get(10, "3B Canavar")


class DiamondTipMonsterDetector:
    """Detects 3D monsters via yellow/olive nameplates, flanking turquoise diamond tips
    (`< Yellow-Maned Zait >`), and red agro badges on both normal and `hflip` frames,
    while strictly excluding the center player character (`0.42..0.58 X, 0.42..0.58 Y`).
    """

    PLAYER_EXCLUSION_X: Tuple[float, float] = (0.42, 0.58)
    PLAYER_EXCLUSION_Y: Tuple[float, float] = (0.42, 0.58)

    def __init__(
        self,
        yellow_hue_range: Tuple[int, int] = (18, 35),
        yellow_min_sat: int = 80,
        yellow_min_val: int = 105,
        cyan_hue_range: Tuple[int, int] = (75, 108),
        cyan_min_sat: int = 65,
        cyan_min_val: int = 115,
        player_exclusion_x: Tuple[float, float] = (0.42, 0.58),
        player_exclusion_y: Tuple[float, float] = (0.42, 0.58),
        require_badge_or_tips: bool = False,
        iou_dedup_threshold: float = 0.45,
    ) -> None:
        self.yellow_hue_range: Tuple[int, int] = (int(yellow_hue_range[0]), int(yellow_hue_range[1]))
        self.yellow_min_sat: int = int(yellow_min_sat)
        self.yellow_min_val: int = int(yellow_min_val)
        self.cyan_hue_range: Tuple[int, int] = (int(cyan_hue_range[0]), int(cyan_hue_range[1]))
        self.cyan_min_sat: int = int(cyan_min_sat)
        self.cyan_min_val: int = int(cyan_min_val)
        self.player_exclusion_x: Tuple[float, float] = (
            float(player_exclusion_x[0]),
            float(player_exclusion_x[1]),
        )
        self.player_exclusion_y: Tuple[float, float] = (
            float(player_exclusion_y[0]),
            float(player_exclusion_y[1]),
        )
        self.require_badge_or_tips: bool = bool(require_badge_or_tips)
        self.iou_dedup_threshold: float = float(iou_dedup_threshold)

    # ------------------------------------------------------------------
    # Coordinate, Masking & Exclusion Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_bgr_frame(frame: Union[np.ndarray, str, os.PathLike]) -> Optional[np.ndarray]:
        """Convert path or ndarray input into a valid 3-channel BGR uint8 image."""
        if frame is None:
            return None
        if isinstance(frame, (str, os.PathLike, Path)):
            path_str = str(frame)
            if not os.path.isfile(path_str):
                return None
            raw = np.fromfile(path_str, dtype=np.uint8)
            if raw.size == 0:
                return None
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    def is_in_player_exclusion_zone(self, cx_norm: float, cy_norm: float) -> bool:
        """Return True if normalized coordinates fall inside the center player exclusion box
        (`0.42..0.58 X, 0.42..0.58 Y`).
        """
        px1, px2 = self.player_exclusion_x
        py1, py2 = self.player_exclusion_y
        return bool((px1 <= float(cx_norm) <= px2) and (py1 <= float(cy_norm) <= py2))

    def _apply_viewport_and_player_masks(
        self,
        mask: np.ndarray,
        img_w: int,
        img_h: int,
        is_hflip: bool = False,
    ) -> None:
        """Mask out HUD regions and the center player character (`0.42..0.58 X, 0.42..0.58 Y`)
        in-place, accounting for both normal and horizontally flipped (`hflip`) frames.
        """
        # Top HUD & bottom skill/chat HUD exclusion
        mask[: int(img_h * 0.10), :] = 0
        mask[int(img_h * 0.80) :, :] = 0

        # Center player character exclusion (0.42..0.58 X, 0.42..0.58 Y)
        px1 = int(round(img_w * self.player_exclusion_x[0]))
        px2 = int(round(img_w * self.player_exclusion_x[1]))
        py1 = int(round(img_h * self.player_exclusion_y[0]))
        py2 = int(round(img_h * self.player_exclusion_y[1]))
        mask[py1:py2, px1:px2] = 0

        # Minimap and quest tracker / chat box exclusions (mirrored when hflip=True)
        if not is_hflip:
            # Normal frame: minimap on top-right, quest tracker on right, chat on bottom-left
            mask[:, int(img_w * 0.855) :] = 0
            mask[: int(img_h * 0.30), int(img_w * 0.72) :] = 0
            mask[int(img_h * 0.66) :, : int(img_w * 0.25)] = 0
        else:
            # Horizontally flipped frame: minimap on top-left, quest tracker on left, chat on bottom-right
            mask[:, : int(img_w * (1.0 - 0.855))] = 0
            mask[: int(img_h * 0.30), : int(img_w * (1.0 - 0.72))] = 0
            mask[int(img_h * 0.66) :, int(img_w * (1.0 - 0.25)) :] = 0

    @staticmethod
    def _pixel_xywh_to_yolo_norm(
        x: float,
        y: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float, float, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Clamp pixel bounding box `(x, y, w, h)` to `(img_w, img_h)` and compute normalized
        `(cx_norm, cy_norm, w_norm, h_norm)` in `[0.000000, 1.000000]`.
        """
        fw = float(max(1, img_w))
        fh = float(max(1, img_h))

        x1 = float(np.clip(x, 0.0, fw - 1.0))
        y1 = float(np.clip(y, 0.0, fh - 1.0))
        x2 = float(np.clip(x + max(1.0, w), x1 + 1.0, fw))
        y2 = float(np.clip(y + max(1.0, h), y1 + 1.0, fh))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + 0.5 * bw
        cy = y1 + 0.5 * bh

        cx_norm = float(np.clip(cx / fw, 0.0, 1.0))
        cy_norm = float(np.clip(cy / fh, 0.0, 1.0))
        w_norm = float(np.clip(bw / fw, 0.0, 1.0))
        h_norm = float(np.clip(bh / fh, 0.0, 1.0))

        nx1 = float(np.clip(cx_norm - 0.5 * w_norm, 0.0, 1.0))
        ny1 = float(np.clip(cy_norm - 0.5 * h_norm, 0.0, 1.0))
        nx2 = float(np.clip(cx_norm + 0.5 * w_norm, 0.0, 1.0))
        ny2 = float(np.clip(cy_norm + 0.5 * h_norm, 0.0, 1.0))

        w_norm = round(float(np.clip(nx2 - nx1, 0.0005, 1.0)), 6)
        h_norm = round(float(np.clip(ny2 - ny1, 0.0005, 1.0)), 6)
        cx_norm = round(float(np.clip(nx1 + 0.5 * w_norm, 0.0, 1.0)), 6)
        cy_norm = round(float(np.clip(ny1 + 0.5 * h_norm, 0.0, 1.0)), 6)

        px_xywh = (int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh)))
        px_xyxy = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        return cx_norm, cy_norm, w_norm, h_norm, px_xywh, px_xyxy

    @staticmethod
    def _compute_iou_xyxy(
        box_a: Sequence[Union[int, float]],
        box_b: Sequence[Union[int, float]],
    ) -> float:
        """Compute Intersection-over-Union (IoU) between two `(x1, y1, x2, y2)` boxes."""
        ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return float(inter / (area_a + area_b - inter + 1e-6))

    def _deduplicate_detections(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Deduplicate overlapping monster detections via Non-Maximum Suppression (NMS)."""
        if not detections:
            return []
        thresh = float(self.iou_dedup_threshold if iou_threshold is None else iou_threshold)
        ordered = sorted(
            detections,
            key=lambda d: (
                bool(d.get("has_diamond_tips", False)),
                bool(d.get("has_agro_badge", False)),
                float(d.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        kept: List[Dict[str, Any]] = []
        for cand in ordered:
            c_xyxy = cand["bbox_xyxy"]
            c_cx = 0.5 * (c_xyxy[0] + c_xyxy[2])
            c_cy = 0.5 * (c_xyxy[1] + c_xyxy[3])

            duplicate = False
            for existing in kept:
                e_xyxy = existing["bbox_xyxy"]
                if self._compute_iou_xyxy(c_xyxy, e_xyxy) >= thresh:
                    duplicate = True
                    break
                e_cx = 0.5 * (e_xyxy[0] + e_xyxy[2])
                e_cy = 0.5 * (e_xyxy[1] + e_xyxy[3])
                min_dim = max(10.0, min(c_xyxy[2] - c_xyxy[0], c_xyxy[3] - c_xyxy[1]) * 0.60)
                if math.hypot(c_cx - e_cx, c_cy - e_cy) <= min_dim:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    # ------------------------------------------------------------------
    # Diamond Tip (`< ... >`) & Agro Badge Inspection Helpers
    # ------------------------------------------------------------------
    def _check_flanking_diamond_tips(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[bool, int, int]:
        """Check for flanking turquoise/cyan diamond bracket tips (`<` on left, `>` on right)
        around a candidate yellow/olive monster nameplate `(x, y, cw, ch)`.

        Because both left and right flanking windows are evaluated symmetrically, this detector
        is naturally invariant to horizontal flips (`hflip`).
        """
        img_h, img_w = hsv.shape[:2]
        pad_outer = max(22, int(round(56 * scale_x)))
        pad_inner = max(6, int(round(10 * scale_x)))
        pad_y = max(4, int(round(8 * scale_y)))

        lx1 = max(0, x - pad_outer)
        lx2 = min(img_w, x + pad_inner)
        rx1 = max(0, x + cw - pad_inner)
        rx2 = min(img_w, x + cw + pad_outer)
        ty1 = max(0, y - pad_y)
        ty2 = min(img_h, y + ch + pad_y)

        c_low = np.array([self.cyan_hue_range[0], self.cyan_min_sat, self.cyan_min_val], dtype=np.uint8)
        c_high = np.array([self.cyan_hue_range[1], 255, 255], dtype=np.uint8)

        left_roi = hsv[ty1:ty2, lx1:lx2]
        right_roi = hsv[ty1:ty2, rx1:rx2]

        left_px = int(np.count_nonzero(cv2.inRange(left_roi, c_low, c_high))) if left_roi.size > 0 else 0
        right_px = int(np.count_nonzero(cv2.inRange(right_roi, c_low, c_high))) if right_roi.size > 0 else 0

        min_tip_px = max(3, int(round(7 * scale_x * scale_y)))
        has_both_tips = bool(left_px >= min_tip_px and right_px >= min_tip_px)
        return has_both_tips, left_px, right_px

    @staticmethod
    def _check_red_agro_badge_or_bar(
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[bool, int]:
        """Check for a red agro badge (e.g. red diamond '4' badge) or hostile red HP bar
        directly below the nameplate `(x, y, cw, ch)`.
        """
        img_h, img_w = hsv.shape[:2]
        cx_label = x + cw // 2

        # Search window below the nameplate center for agro badge or HP bar
        half_span_x = max(18, min(cw // 2 + int(round(12 * scale_x)), int(round(48 * scale_x))))
        bx1 = max(0, cx_label - half_span_x)
        bx2 = min(img_w, cx_label + half_span_x)
        by1 = max(0, min(img_h, y + max(0, ch - int(round(2 * scale_y)))))
        by2 = min(img_h, y + ch + max(24, int(round(52 * scale_y))))

        below_hsv = hsv[by1:by2, bx1:bx2]
        if below_hsv.size == 0:
            return False, 0

        red1 = cv2.inRange(below_hsv, np.array([0, 125, 90], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
        red2 = cv2.inRange(below_hsv, np.array([168, 125, 90], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_px = int(np.count_nonzero(red1 | red2))

        min_red_px = max(18, int(round(40 * scale_x * scale_y)))
        return bool(red_px >= min_red_px), red_px

    # ------------------------------------------------------------------
    # Single-Orientation Internal Pass
    # ------------------------------------------------------------------
    def _detect_single_pass(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        existing_positive_boxes: Sequence[Sequence[Union[int, float]]],
        is_hflip: bool = False,
        include_untargeted: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run yellow/olive monster nameplate + turquoise diamond tip + red agro badge detection
        on a single orientation (`frame_bgr`).
        """
        detections: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 320 or img_h < 200:
            return detections

        scale_x = max(0.45, img_w / 2000.0)
        scale_y = max(0.45, img_h / 1124.0)

        # 1. Top-hat morphological filter to isolate crisp character strokes (symmetric square kernel)
        th_sz = max(5, int(round(7 * scale_x)) | 1)
        tophat = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (th_sz, th_sz)),
        )

        # 2. Yellow / olive monster nameplate glyph stroke mask
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        yellow_stroke = (
            (h_ch >= self.yellow_hue_range[0])
            & (h_ch <= self.yellow_hue_range[1])
            & (s_ch >= self.yellow_min_sat)
            & (v_ch >= self.yellow_min_val)
            & (tophat >= 8)
        )
        ym_u8 = (yellow_stroke.astype(np.uint8)) * 255

        # Apply viewport and center player character exclusion (0.42..0.58 X, 0.42..0.58 Y)
        self._apply_viewport_and_player_masks(ym_u8, img_w=img_w, img_h=img_h, is_hflip=is_hflip)

        # 3. Horizontal morphological closing to group character glyphs into nameplate bars
        kw = max(9, int(round(16 * scale_x)))
        kh = max(3, int(round(4 * scale_y)))
        closed = cv2.morphologyEx(
            ym_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
        )
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = max(56, int(round(105 * scale_x)))
        max_w = max(160, int(round(270 * scale_x)))
        min_h = max(5, int(round(6 * scale_y)))
        max_h = max(20, int(round(30 * scale_y)))

        for cnt in cnts:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if not (min_w <= cw <= max_w and min_h <= ch <= max_h):
                continue

            aspect = float(cw) / float(max(1, ch))
            if aspect < 4.0:
                continue

            # Check if the nameplate center itself falls inside the player exclusion zone
            np_cx_norm = (x + 0.5 * cw) / float(img_w)
            np_cy_norm = (y + 0.5 * ch) / float(img_h)
            if self.is_in_player_exclusion_zone(np_cx_norm, np_cy_norm):
                continue

            sub_ym = ym_u8[y : y + ch, x : x + cw]
            stroke_px = int(np.count_nonzero(sub_ym))
            min_stroke = max(28, int(round(85 * scale_x * scale_y)))

            # Check flanking turquoise diamond tips (< ... >) and red agro badge
            has_diamond_tips, left_cyan_px, right_cyan_px = self._check_flanking_diamond_tips(
                hsv=hsv,
                x=x,
                y=y,
                cw=cw,
                ch=ch,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            has_red_agro, red_agro_px = self._check_red_agro_badge_or_bar(
                hsv=hsv,
                x=x,
                y=y,
                cw=cw,
                ch=ch,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            has_agro_badge = bool(has_diamond_tips or has_red_agro)

            if self.require_badge_or_tips and not has_agro_badge:
                continue
            if not include_untargeted and not has_agro_badge:
                continue

            num_labels, _ = cv2.connectedComponents(sub_ym)
            if not has_agro_badge and (stroke_px < min_stroke or num_labels < 6):
                continue
            if has_agro_badge and (stroke_px < max(14, min_stroke // 3) or num_labels < 3):
                continue

            # Verify dark character outline halo around yellow strokes
            dilated_stroke = cv2.dilate(sub_ym, np.ones((3, 3), dtype=np.uint8))
            border_ring = dilated_stroke - sub_ym
            ring_gray = gray[y : y + ch, x : x + cw][border_ring > 0]
            if ring_gray.size == 0:
                continue
            dark_outline_ratio = float(np.mean(ring_gray < 122))
            if not has_agro_badge and dark_outline_ratio < 0.20:
                continue

            # Expand bounding box to cover the monster nameplate + diamond tips + 3D body below
            if has_agro_badge:
                final_x = max(0, x - int(round(24 * scale_x)))
                final_y = max(0, y - int(round(8 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(48 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(105 * scale_y)))
                conf = 0.96 if (has_diamond_tips and has_red_agro) else 0.95
                src = (
                    "diamond_tip_monster_detector"
                    if has_diamond_tips
                    else "agro_badge_monster_detector"
                )
            else:
                final_x = max(0, x - int(round(6 * scale_x)))
                final_y = max(0, y - int(round(4 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(12 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(8 * scale_y)))
                conf = round(min(0.92, 0.78 + min(0.14, stroke_px / 900.0)), 4)
                src = "yellow_nameplate_monster_detector"

            cx_n, cy_n, w_n, h_n, px_xywh, px_xyxy = self._pixel_xywh_to_yolo_norm(
                final_x, final_y, final_w, final_h, img_w, img_h
            )

            # Enforce center player character exclusion on the final bounding box center too
            if self.is_in_player_exclusion_zone(cx_n, cy_n):
                continue

            if any(self._compute_iou_xyxy(px_xyxy, pos_box) >= 0.25 for pos_box in existing_positive_boxes):
                continue

            is_trusted = bool(has_agro_badge and conf >= 0.85)
            detections.append(
                {
                    "class_id": MONSTER_CLASS_ID,
                    "class_name": MONSTER_CLASS_NAME,
                    "class_name_tr": MONSTER_CLASS_LABEL_TR,
                    "material_name": MONSTER_CLASS_NAME,
                    "cx": cx_n,
                    "cy": cy_n,
                    "w": w_n,
                    "h": h_n,
                    "width": w_n,
                    "height": h_n,
                    "bbox_norm": (cx_n, cy_n, w_n, h_n),
                    "yolo_bbox": [cx_n, cy_n, w_n, h_n],
                    "bbox": px_xywh,
                    "bbox_xywh": px_xywh,
                    "bbox_xyxy": list(px_xyxy),
                    "nameplate_bbox_xywh": (int(x), int(y), int(cw), int(ch)),
                    "nameplate_bbox_xyxy": [int(x), int(y), int(x + cw), int(y + ch)],
                    "confidence": round(float(conf), 4),
                    "source": src,
                    "has_diamond_tips": bool(has_diamond_tips),
                    "has_agro_badge": bool(has_agro_badge),
                    "has_red_agro": bool(has_red_agro),
                    "left_diamond_px": int(left_cyan_px),
                    "right_diamond_px": int(right_cyan_px),
                    "red_agro_px": int(red_agro_px),
                    "stroke_px": int(stroke_px),
                    "dark_outline_ratio": round(float(dark_outline_ratio), 4),
                    "is_hflip": bool(is_hflip),
                    "is_negative": True,
                    "hand_label": False,
                    "trusted": is_trusted,
                    "weak": not is_trusted,
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Public Detection API
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: Union[np.ndarray, str, os.PathLike],
        hsv: Optional[np.ndarray] = None,
        gray: Optional[np.ndarray] = None,
        existing_positive_boxes: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        is_hflip: Optional[bool] = None,
        stem: str = "",
        include_untargeted: bool = True,
        verify_with_hflip: bool = False,
    ) -> List[Dict[str, Any]]:
        """Detect yellow/olive monster nameplates with flanking turquoise diamond tips
        (`< Yellow-Maned Zait >`) or red agro badges on normal and `hflip` frames.

        Args:
            frame: BGR image ndarray or file path.
            hsv: Optional precomputed HSV image matching `frame`.
            gray: Optional precomputed grayscale image matching `frame`.
            existing_positive_boxes: Optional list of `(x1, y1, x2, y2)` positive boxes to avoid overlapping.
            is_hflip: Optional explicit flag indicating `frame` is horizontally flipped.
                      If None, auto-inferred from `stem` or path containing `'hflip'`.
            stem: Optional filename stem for auto-detecting `_hflip`.
            include_untargeted: Whether to also include untargeted yellow/olive monster nameplates.
            verify_with_hflip: If True, also runs detection on `cv2.flip(frame, 1)`, un-flips any
                               recovered detections back to `frame`'s coordinate system, and merges via NMS.

        Returns:
            List of standardized monster (`class_id = 10: monster_3d`) annotation dictionaries.
        """
        frame_bgr = self._coerce_bgr_frame(frame)
        if frame_bgr is None:
            return []

        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 320 or img_h < 200:
            return []

        if is_hflip is None:
            path_hint = str(frame) if isinstance(frame, (str, os.PathLike, Path)) else ""
            combined_hint = f"{stem} {path_hint}".lower()
            inferred_hflip = "hflip" in combined_hint or "flip_h" in combined_hint
        else:
            inferred_hflip = bool(is_hflip)

        if hsv is None or hsv.shape[:2] != (img_h, img_w):
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if gray is None or gray.shape[:2] != (img_h, img_w):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        pos_boxes: List[Sequence[Union[int, float]]] = list(existing_positive_boxes or [])

        primary_dets = self._detect_single_pass(
            frame_bgr=frame_bgr,
            hsv=hsv,
            gray=gray,
            existing_positive_boxes=pos_boxes,
            is_hflip=inferred_hflip,
            include_untargeted=include_untargeted,
        )

        if verify_with_hflip:
            flipped_bgr = cv2.flip(frame_bgr, 1)
            flipped_hsv = cv2.flip(hsv, 1)
            flipped_gray = cv2.flip(gray, 1)
            flipped_pos_boxes = [
                (img_w - float(b[2]), float(b[1]), img_w - float(b[0]), float(b[3]))
                for b in pos_boxes
            ]
            mirror_dets = self._detect_single_pass(
                frame_bgr=flipped_bgr,
                hsv=flipped_hsv,
                gray=flipped_gray,
                existing_positive_boxes=flipped_pos_boxes,
                is_hflip=not inferred_hflip,
                include_untargeted=include_untargeted,
            )
            for mdet in mirror_dets:
                unflipped = self.unflip_detection(mdet, img_w=img_w, img_h=img_h)
                if not self.is_in_player_exclusion_zone(unflipped["cx"], unflipped["cy"]):
                    primary_dets.append(unflipped)

        return self._deduplicate_detections(primary_dets)

    # Aliases for seamless integration across dataset_factory modules
    scan_monsters = detect
    detect_monsters = detect

    def unflip_detection(
        self,
        detection: Dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """Mirror a detection horizontally (`cx -> 1.0 - cx`) between normal and `hflip` coordinate frames."""
        bx1, by1, bx2, by2 = detection["bbox_xyxy"]
        new_x1 = max(0, img_w - int(bx2))
        new_x2 = min(img_w, img_w - int(bx1))
        new_w = max(1, new_x2 - new_x1)
        new_h = max(1, int(by2) - int(by1))

        cx_n, cy_n, w_n, h_n, px_xywh, px_xyxy = self._pixel_xywh_to_yolo_norm(
            new_x1, by1, new_w, new_h, img_w, img_h
        )
        updated = dict(detection)
        updated.update(
            {
                "cx": cx_n,
                "cy": cy_n,
                "w": w_n,
                "h": h_n,
                "width": w_n,
                "height": h_n,
                "bbox_norm": (cx_n, cy_n, w_n, h_n),
                "yolo_bbox": [cx_n, cy_n, w_n, h_n],
                "bbox": px_xywh,
                "bbox_xywh": px_xywh,
                "bbox_xyxy": list(px_xyxy),
                "left_diamond_px": int(detection.get("right_diamond_px", 0)),
                "right_diamond_px": int(detection.get("left_diamond_px", 0)),
            }
        )
        if "nameplate_bbox_xyxy" in detection:
            nx1, ny1, nx2, ny2 = detection["nameplate_bbox_xyxy"]
            unx1 = max(0, img_w - int(nx2))
            unx2 = min(img_w, img_w - int(nx1))
            updated["nameplate_bbox_xyxy"] = [unx1, int(ny1), unx2, int(ny2)]
            updated["nameplate_bbox_xywh"] = (unx1, int(ny1), max(1, unx2 - unx1), max(1, int(ny2) - int(ny1)))
        return updated

    def inspect_nameplate_roi(
        self,
        roi_bgr: Union[np.ndarray, str, os.PathLike],
    ) -> Dict[str, Any]:
        """Inspect a cropped nameplate ROI (normal or `hflip`) for yellow/olive monster text,
        flanking turquoise diamond tips (`< ... >`), and red agro badge pixels.
        """
        bgr = self._coerce_bgr_frame(roi_bgr)
        if bgr is None or bgr.size == 0:
            return {
                "is_monster_nameplate": False,
                "has_diamond_tips": False,
                "has_agro_badge": False,
                "left_diamond_px": 0,
                "right_diamond_px": 0,
                "yellow_stroke_px": 0,
                "red_agro_px": 0,
                "confidence": 0.0,
            }

        rh, rw = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        th = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        yellow_mask = (
            (hsv[:, :, 0] >= self.yellow_hue_range[0])
            & (hsv[:, :, 0] <= self.yellow_hue_range[1])
            & (hsv[:, :, 1] >= self.yellow_min_sat)
            & (hsv[:, :, 2] >= self.yellow_min_val)
            & (th >= 8)
        )
        yellow_px = int(np.count_nonzero(yellow_mask))

        c_low = np.array([self.cyan_hue_range[0], self.cyan_min_sat, self.cyan_min_val], dtype=np.uint8)
        c_high = np.array([self.cyan_hue_range[1], 255, 255], dtype=np.uint8)
        cyan_mask = cv2.inRange(hsv, c_low, c_high)

        left_band = cyan_mask[:, : max(1, int(rw * 0.25))]
        right_band = cyan_mask[:, min(rw - 1, int(rw * 0.75)) :]
        left_px = int(np.count_nonzero(left_band))
        right_px = int(np.count_nonzero(right_band))
        has_tips = bool(left_px >= 3 and right_px >= 3)

        red1 = cv2.inRange(hsv, np.array([0, 125, 90], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
        red2 = cv2.inRange(hsv, np.array([168, 125, 90], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_px = int(np.count_nonzero(red1 | red2))
        has_agro = bool(has_tips or red_px >= 15)

        is_monster = bool(yellow_px >= 20 and (has_agro or yellow_px >= 35))
        conf = 0.95 if (is_monster and has_agro) else (0.85 if is_monster else 0.0)
        return {
            "is_monster_nameplate": is_monster,
            "has_diamond_tips": has_tips,
            "has_agro_badge": has_agro,
            "left_diamond_px": left_px,
            "right_diamond_px": right_px,
            "yellow_stroke_px": yellow_px,
            "red_agro_px": red_px,
            "confidence": round(float(conf), 4),
        }

    @staticmethod
    def format_yolo_lines(detections: Sequence[Dict[str, Any]]) -> List[str]:
        """Format monster detections into canonical YOLO `.txt` lines (`10 cx cy w h`)."""
        lines: List[str] = []
        for det in detections:
            line = format_yolo_txt_line(
                class_id=det.get("class_id", MONSTER_CLASS_ID),
                cx=det.get("cx", 0.0),
                cy=det.get("cy", 0.0),
                w=det.get("w", 0.0),
                h=det.get("h", 0.0),
                class_name=det.get("class_name", MONSTER_CLASS_NAME),
            )
            if line is not None:
                lines.append(line)
        return lines


_DEFAULT_DETECTOR: Optional[DiamondTipMonsterDetector] = None


def get_monster_diamond_detector() -> DiamondTipMonsterDetector:
    """Return a shared singleton instance of `DiamondTipMonsterDetector`."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = DiamondTipMonsterDetector()
    return _DEFAULT_DETECTOR


def detect_diamond_tip_monsters(
    frame: Union[np.ndarray, str, os.PathLike],
    hsv: Optional[np.ndarray] = None,
    gray: Optional[np.ndarray] = None,
    existing_positive_boxes: Optional[Sequence[Sequence[Union[int, float]]]] = None,
    is_hflip: Optional[bool] = None,
    stem: str = "",
    include_untargeted: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function to detect monsters using the default `DiamondTipMonsterDetector`."""
    return get_monster_diamond_detector().detect(
        frame=frame,
        hsv=hsv,
        gray=gray,
        existing_positive_boxes=existing_positive_boxes,
        is_hflip=is_hflip,
        stem=stem,
        include_untargeted=include_untargeted,
    )


__all__ = [
    "MONSTER_CLASS_ID",
    "MONSTER_CLASS_NAME",
    "MONSTER_CLASS_LABEL_TR",
    "DiamondTipMonsterDetector",
    "get_monster_diamond_detector",
    "detect_diamond_tip_monsters",
]
