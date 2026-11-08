# -*- coding: utf-8 -*-
"""dataset_factory/scroll_potion_classifier.py
==============================================
AION 2 Bottom HUD Consumable & Skill Slot Classifier (`HUDConsumableClassifier`).

Separates:
- Center Character Level Emblem (`0.44 <= norm_cx <= 0.56`, `0.85 <= norm_cy <= 0.95`)
- Upper Consumable Row (`0.865 <= norm_cy <= 0.938`):
  * `buff_scroll` (Parşömen / Güçlendirme Parşömeni)
  * `hp_potion` (Can İksiri)
  * `mp_potion` (Mana / Sarı İksir)
  * Empty slots (`F8`, etc. — rejected / unlabeled)
  * Never labeled as `skill_slot` or `skill_ui_prompt`
- Lower Combat Skill Row (`norm_cy >= 0.939`):
  * `skill_slot` (Yetenek Yuvası)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# ==============================================================================
# HUD Geometry & Class Mapping Constants
# ==============================================================================

UPPER_CONSUMABLE_ROW_Y_MIN: float = 0.865
UPPER_CONSUMABLE_ROW_Y_MAX: float = 0.938
LOWER_SKILL_ROW_Y_MIN: float = 0.939

CENTER_EMBLEM_X_MIN: float = 0.44
CENTER_EMBLEM_X_MAX: float = 0.56
CENTER_EMBLEM_Y_MIN: float = 0.85
CENTER_EMBLEM_Y_MAX: float = 0.95

# YOLO Auto-Labeler class IDs (compatible with dataset_factory/yolo_auto_labeler.py)
YOLO_HUD_CLASS_IDS: Dict[str, int] = {
    "hp_potion": 18,
    "mp_potion": 19,
    "skill_slot": 21,
    "buff_scroll": 22,
}

# Canonical Taxonomy class IDs (compatible with dataset_factory/class_taxonomy.py)
CANONICAL_HUD_CLASS_IDS: Dict[str, int] = {
    "skill_slot": 20,
    "hp_potion": 22,
    "mp_potion": 23,
    "buff_scroll": 24,
}

HUD_CLASS_LABELS_TR: Dict[str, str] = {
    "buff_scroll": "Parşömen (Buff Scroll)",
    "hp_potion": "Can İksiri (HP Potion)",
    "mp_potion": "Mana/Sarı İksir (MP Potion)",
    "skill_slot": "Yetenek Yuvası (Skill Slot)",
    "center_emblem": "Merkez Seviye Arması",
    "empty_slot": "Boş Yuva (F8)",
    "unclassified": "Sınıflandırılmamış",
}

HUD_SOURCE_TAGS: Dict[str, str] = {
    "buff_scroll": "hud_buff_scroll",
    "hp_potion": "hud_hp_potion",
    "mp_potion": "hud_mp_potion",
    "skill_slot": "hud_skill_slot",
}


@dataclass(frozen=True)
class SlotClassificationResult:
    """Structured result for a single HUD slot candidate."""

    category: str
    class_id: Optional[int]
    canonical_class_id: Optional[int]
    label_tr: str
    confidence: float
    should_label: bool
    row_type: str
    is_empty: bool = False
    is_center_emblem: bool = False
    norm_cx: float = 0.0
    norm_cy: float = 0.0
    source: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "class_id": self.class_id,
            "canonical_class_id": self.canonical_class_id,
            "label_tr": self.label_tr,
            "confidence": self.confidence,
            "should_label": self.should_label,
            "row_type": self.row_type,
            "is_empty": self.is_empty,
            "is_center_emblem": self.is_center_emblem,
            "norm_cx": round(float(self.norm_cx), 6),
            "norm_cy": round(float(self.norm_cy), 6),
            "source": self.source,
            "metrics": dict(self.metrics),
        }


class HUDConsumableClassifier:
    """
    AION 2 Bottom HUD Classifier for separating the upper consumable row
    (`0.865 <= norm_cy <= 0.938`), the lower combat skill row (`norm_cy >= 0.939`),
    the center level emblem (`0.44 <= norm_cx <= 0.56`), and empty quickbar slots (`F8`).

    Classifies valid HUD slots into:
    - `buff_scroll` (Parşömen)
    - `hp_potion` (Can İksiri)
    - `mp_potion` (Mana/Sarı İksir)
    - `skill_slot` (Yetenek Yuvası)
    """

    def __init__(
        self,
        upper_row_y_range: Tuple[float, float] = (UPPER_CONSUMABLE_ROW_Y_MIN, UPPER_CONSUMABLE_ROW_Y_MAX),
        lower_row_y_min: float = LOWER_SKILL_ROW_Y_MIN,
        center_emblem_x_range: Tuple[float, float] = (CENTER_EMBLEM_X_MIN, CENTER_EMBLEM_X_MAX),
        center_emblem_y_range: Tuple[float, float] = (CENTER_EMBLEM_Y_MIN, CENTER_EMBLEM_Y_MAX),
        empty_sat_mean_max: float = 28.0,
        empty_val_std_max: float = 16.0,
        use_canonical_ids: bool = False,
    ) -> None:
        self.upper_row_y_min = float(upper_row_y_range[0])
        self.upper_row_y_max = float(upper_row_y_range[1])
        self.lower_row_y_min = float(lower_row_y_min)
        self.center_emblem_x_min = float(center_emblem_x_range[0])
        self.center_emblem_x_max = float(center_emblem_x_range[1])
        self.center_emblem_y_min = float(center_emblem_y_range[0])
        self.center_emblem_y_max = float(center_emblem_y_range[1])
        self.empty_sat_mean_max = float(empty_sat_mean_max)
        self.empty_val_std_max = float(empty_val_std_max)
        self.use_canonical_ids = bool(use_canonical_ids)

    # ------------------------------------------------------------------
    # Spatial Row & Emblem Guards
    # ------------------------------------------------------------------
    def is_center_level_emblem(self, norm_cx: float, norm_cy: float) -> bool:
        """Return True if normalized slot center falls inside the center character level emblem (`0.44..0.56 X`)."""
        return (
            self.center_emblem_x_min <= float(norm_cx) <= self.center_emblem_x_max
            and self.center_emblem_y_min <= float(norm_cy) <= self.center_emblem_y_max
        )

    def is_upper_consumable_row(self, norm_cy: float) -> bool:
        """Return True if normalized Y center is in the upper consumable row (`0.865 <= norm_cy <= 0.938`)."""
        return self.upper_row_y_min <= float(norm_cy) <= self.upper_row_y_max

    def is_lower_skill_row(self, norm_cy: float) -> bool:
        """Return True if normalized Y center is in the lower skill row (`norm_cy >= 0.939`)."""
        return float(norm_cy) >= self.lower_row_y_min

    # ------------------------------------------------------------------
    # Empty Slot & HSV Chromatic Analysis
    # ------------------------------------------------------------------
    def is_empty_slot(
        self,
        slot_hsv: np.ndarray,
        slot_edges: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect empty HUD quickbar slots (e.g., empty `F8` slot with dark/flat background).
        """
        if slot_hsv is None or slot_hsv.size == 0:
            return True, {"sat_mean": 0.0, "val_std": 0.0, "val_mean": 0.0}

        h, w = slot_hsv.shape[:2]
        # Trim 15% border so outer slot frame bevel/hotkey text does not inflate variance
        pad_x = max(1, int(round(w * 0.15))) if w >= 8 else 0
        pad_y = max(1, int(round(h * 0.15))) if h >= 8 else 0
        inner_hsv = slot_hsv[pad_y : h - pad_y, pad_x : w - pad_x]
        if inner_hsv.size == 0:
            inner_hsv = slot_hsv

        sat_mean = float(np.mean(inner_hsv[:, :, 1]))
        val_std = float(np.std(inner_hsv[:, :, 2]))
        val_mean = float(np.mean(inner_hsv[:, :, 2]))

        full_sat_mean = float(np.mean(slot_hsv[:, :, 1]))
        full_val_std = float(np.std(slot_hsv[:, :, 2]))

        is_low_chroma = (
            (sat_mean < self.empty_sat_mean_max and val_std < self.empty_val_std_max)
            or (full_sat_mean < self.empty_sat_mean_max and full_val_std < self.empty_val_std_max)
            or (sat_mean < 18.0 and val_mean < 52.0)
        )

        metrics = {
            "sat_mean": round(sat_mean, 3),
            "val_std": round(val_std, 3),
            "val_mean": round(val_mean, 3),
        }

        if slot_edges is not None and slot_edges.size > 0:
            inner_edges = slot_edges[pad_y : h - pad_y, pad_x : w - pad_x]
            if inner_edges.size == 0:
                inner_edges = slot_edges
            edge_ratio = float(np.count_nonzero(inner_edges)) / float(max(1, inner_edges.size))
            metrics["inner_edge_ratio"] = round(edge_ratio, 4)
            if edge_ratio < 0.025 and sat_mean < 32.0:
                return True, metrics

        return bool(is_low_chroma), metrics

    def compute_consumable_pixel_counts(
        self,
        slot_hsv: np.ndarray,
    ) -> Dict[str, int]:
        """
        Compute HSV chromatic pixel counts for:
        - `hp_red_px`: Red potion liquid (`H in [0, 12] U [165, 180], S >= 105, V >= 75`)
        - `mp_blue_px`: Blue mana potion liquid (`H in [95, 135], S >= 85, V >= 75`)
        - `mp_yellow_px`: Yellow mana/serum potion liquid (`H in [20, 35], S >= 95, V >= 95`)
        - `mp_blue_yellow_px`: Combined blue or yellow potion pixels
        - `scroll_px`: Parchment buff scroll (`H in [12, 28], S in [30, 175], V in [85, 235]`)
        """
        if slot_hsv is None or slot_hsv.size == 0:
            return {
                "hp_red_px": 0,
                "mp_blue_px": 0,
                "mp_yellow_px": 0,
                "mp_blue_yellow_px": 0,
                "scroll_px": 0,
                "slot_area": 0,
            }

        # HP Potion (red liquid)
        r1 = cv2.inRange(slot_hsv, np.array([0, 105, 75], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        r2 = cv2.inRange(slot_hsv, np.array([165, 105, 75], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        hp_red_px = int(np.count_nonzero(r1 | r2))

        # MP Potion (blue or yellow liquid)
        b_mask = cv2.inRange(slot_hsv, np.array([95, 85, 75], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
        y_mask = cv2.inRange(slot_hsv, np.array([20, 95, 95], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
        mp_blue_px = int(np.count_nonzero(b_mask))
        mp_yellow_px = int(np.count_nonzero(y_mask))
        mp_blue_yellow_px = int(np.count_nonzero(b_mask | y_mask))

        # Buff Scroll (rolled parchment icon: tan/golden/cream)
        scroll_mask = cv2.inRange(
            slot_hsv,
            np.array([12, 30, 85], dtype=np.uint8),
            np.array([28, 175, 235], dtype=np.uint8),
        )
        scroll_px = int(np.count_nonzero(scroll_mask))

        slot_area = int(slot_hsv.shape[0] * slot_hsv.shape[1])
        return {
            "hp_red_px": hp_red_px,
            "mp_blue_px": mp_blue_px,
            "mp_yellow_px": mp_yellow_px,
            "mp_blue_yellow_px": mp_blue_yellow_px,
            "scroll_px": scroll_px,
            "slot_area": slot_area,
        }

    # ------------------------------------------------------------------
    # Core Slot Classification
    # ------------------------------------------------------------------
    def _resolve_class_id(self, category: str) -> Optional[int]:
        if self.use_canonical_ids:
            return CANONICAL_HUD_CLASS_IDS.get(category)
        return YOLO_HUD_CLASS_IDS.get(category)

    def classify_slot(
        self,
        slot_bgr_or_hsv: np.ndarray,
        norm_cx: float,
        norm_cy: float,
        *,
        is_hsv: bool = False,
        slot_edges: Optional[np.ndarray] = None,
        scale_x: float = 1.0,
    ) -> SlotClassificationResult:
        """
        Classify a single HUD slot crop given its normalized center (`norm_cx`, `norm_cy`).

        Rules:
        1. Center level emblem (`0.44 <= norm_cx <= 0.56` and `0.85 <= norm_cy <= 0.95`) -> excluded (`center_emblem`).
        2. Upper consumable row (`0.865 <= norm_cy <= 0.938`):
           - Empty slot (`F8`, low saturation & low value variance) -> excluded (`empty_slot`).
           - Classifies into `hp_potion`, `mp_potion`, or `buff_scroll`.
           - Never classifies upper row slots as `skill_slot`.
        3. Lower skill row (`norm_cy >= 0.939`):
           - Empty slot -> excluded (`empty_slot`).
           - Classifies active combat skill slots as `skill_slot`.
        """
        norm_cx_f = float(norm_cx)
        norm_cy_f = float(norm_cy)

        # 1. Center character level emblem guard (0.44..0.56 X)
        if self.is_center_level_emblem(norm_cx_f, norm_cy_f):
            return SlotClassificationResult(
                category="center_emblem",
                class_id=None,
                canonical_class_id=None,
                label_tr=HUD_CLASS_LABELS_TR["center_emblem"],
                confidence=0.0,
                should_label=False,
                row_type="center_emblem",
                is_empty=False,
                is_center_emblem=True,
                norm_cx=norm_cx_f,
                norm_cy=norm_cy_f,
                source="center_emblem_filter",
            )

        if slot_bgr_or_hsv is None or slot_bgr_or_hsv.size == 0:
            return SlotClassificationResult(
                category="empty_slot",
                class_id=None,
                canonical_class_id=None,
                label_tr=HUD_CLASS_LABELS_TR["empty_slot"],
                confidence=0.0,
                should_label=False,
                row_type="unknown",
                is_empty=True,
                is_center_emblem=False,
                norm_cx=norm_cx_f,
                norm_cy=norm_cy_f,
                source="empty_slot_filter",
            )

        if is_hsv:
            slot_hsv = slot_bgr_or_hsv
        else:
            slot_hsv = cv2.cvtColor(slot_bgr_or_hsv, cv2.COLOR_BGR2HSV)

        empty_flag, empty_metrics = self.is_empty_slot(slot_hsv, slot_edges=slot_edges)

        # 2. Upper Consumable Row (0.865 <= norm_cy <= 0.938)
        if self.is_upper_consumable_row(norm_cy_f):
            if empty_flag:
                return SlotClassificationResult(
                    category="empty_slot",
                    class_id=None,
                    canonical_class_id=None,
                    label_tr=HUD_CLASS_LABELS_TR["empty_slot"],
                    confidence=0.0,
                    should_label=False,
                    row_type="upper_consumable_row",
                    is_empty=True,
                    is_center_emblem=False,
                    norm_cx=norm_cx_f,
                    norm_cy=norm_cy_f,
                    source="empty_slot_filter",
                    metrics=empty_metrics,
                )

            counts = self.compute_consumable_pixel_counts(slot_hsv)
            slot_area = max(1, counts["slot_area"])
            hp_red_px = counts["hp_red_px"]
            mp_blue_yellow_px = counts["mp_blue_yellow_px"]
            scroll_px = counts["scroll_px"]

            min_colored = max(8, int(0.07 * slot_area))
            min_scroll = max(10, int(0.09 * slot_area))

            metrics = {
                **empty_metrics,
                "hp_red_ratio": round(hp_red_px / float(slot_area), 4),
                "mp_blue_yellow_ratio": round(mp_blue_yellow_px / float(slot_area), 4),
                "scroll_ratio": round(scroll_px / float(slot_area), 4),
            }

            if hp_red_px >= min_colored and hp_red_px > mp_blue_yellow_px and hp_red_px > scroll_px:
                category = "hp_potion"
                conf = 0.70
            elif mp_blue_yellow_px >= min_colored and mp_blue_yellow_px > scroll_px:
                category = "mp_potion"
                conf = 0.70
            elif scroll_px >= min_scroll:
                category = "buff_scroll"
                conf = 0.68
            else:
                return SlotClassificationResult(
                    category="unclassified",
                    class_id=None,
                    canonical_class_id=None,
                    label_tr=HUD_CLASS_LABELS_TR["unclassified"],
                    confidence=0.0,
                    should_label=False,
                    row_type="upper_consumable_row",
                    is_empty=False,
                    is_center_emblem=False,
                    norm_cx=norm_cx_f,
                    norm_cy=norm_cy_f,
                    source="unclassified_upper_slot",
                    metrics=metrics,
                )

            return SlotClassificationResult(
                category=category,
                class_id=self._resolve_class_id(category),
                canonical_class_id=CANONICAL_HUD_CLASS_IDS[category],
                label_tr=HUD_CLASS_LABELS_TR[category],
                confidence=conf,
                should_label=True,
                row_type="upper_consumable_row",
                is_empty=False,
                is_center_emblem=False,
                norm_cx=norm_cx_f,
                norm_cy=norm_cy_f,
                source=HUD_SOURCE_TAGS[category],
                metrics=metrics,
            )

        # 3. Lower Combat Skills Row (norm_cy >= 0.939)
        if self.is_lower_skill_row(norm_cy_f):
            if empty_flag:
                return SlotClassificationResult(
                    category="empty_slot",
                    class_id=None,
                    canonical_class_id=None,
                    label_tr=HUD_CLASS_LABELS_TR["empty_slot"],
                    confidence=0.0,
                    should_label=False,
                    row_type="lower_skill_row",
                    is_empty=True,
                    is_center_emblem=False,
                    norm_cx=norm_cx_f,
                    norm_cy=norm_cy_f,
                    source="empty_slot_filter",
                    metrics=empty_metrics,
                )

            min_edges = max(10, int(15 * float(scale_x)))
            if slot_edges is not None and slot_edges.size > 0:
                edge_cnt = int(np.count_nonzero(slot_edges))
            else:
                v_chan = slot_hsv[:, :, 2]
                edges_local = cv2.Canny(v_chan, 50, 150)
                edge_cnt = int(np.count_nonzero(edges_local))

            metrics = {**empty_metrics, "edge_count": float(edge_cnt)}
            if edge_cnt < min_edges:
                return SlotClassificationResult(
                    category="empty_slot",
                    class_id=None,
                    canonical_class_id=None,
                    label_tr=HUD_CLASS_LABELS_TR["empty_slot"],
                    confidence=0.0,
                    should_label=False,
                    row_type="lower_skill_row",
                    is_empty=True,
                    is_center_emblem=False,
                    norm_cx=norm_cx_f,
                    norm_cy=norm_cy_f,
                    source="low_edge_skill_slot_filter",
                    metrics=metrics,
                )

            category = "skill_slot"
            return SlotClassificationResult(
                category=category,
                class_id=self._resolve_class_id(category),
                canonical_class_id=CANONICAL_HUD_CLASS_IDS[category],
                label_tr=HUD_CLASS_LABELS_TR[category],
                confidence=0.68,
                should_label=True,
                row_type="lower_skill_row",
                is_empty=False,
                is_center_emblem=False,
                norm_cx=norm_cx_f,
                norm_cy=norm_cy_f,
                source=HUD_SOURCE_TAGS[category],
                metrics=metrics,
            )

        # Outside both upper consumable row and lower skill row
        return SlotClassificationResult(
            category="unclassified",
            class_id=None,
            canonical_class_id=None,
            label_tr=HUD_CLASS_LABELS_TR["unclassified"],
            confidence=0.0,
            should_label=False,
            row_type="out_of_hud_rows",
            is_empty=empty_flag,
            is_center_emblem=False,
            norm_cx=norm_cx_f,
            norm_cy=norm_cy_f,
            source="out_of_hud_rows",
            metrics=empty_metrics,
        )

    def classify_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox_xywh: Tuple[int, int, int, int],
        *,
        hsv: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
    ) -> SlotClassificationResult:
        """Classify a bounding box `(x, y, w, h)` in pixel coordinates on `frame_bgr`."""
        if frame_bgr is None or frame_bgr.size == 0:
            return SlotClassificationResult(
                category="empty_slot",
                class_id=None,
                canonical_class_id=None,
                label_tr=HUD_CLASS_LABELS_TR["empty_slot"],
                confidence=0.0,
                should_label=False,
                row_type="unknown",
                is_empty=True,
            )

        img_h, img_w = frame_bgr.shape[:2]
        x, y, w, h = [int(round(float(v))) for v in bbox_xywh]
        x1 = max(0, min(img_w - 1, x))
        y1 = max(0, min(img_h - 1, y))
        x2 = max(x1 + 1, min(img_w, x + max(1, w)))
        y2 = max(y1 + 1, min(img_h, y + max(1, h)))

        norm_cx = (x1 + x2) * 0.5 / float(max(1, img_w))
        norm_cy = (y1 + y2) * 0.5 / float(max(1, img_h))
        scale_x = float(img_w) / 2000.0

        slot_edges = edges[y1:y2, x1:x2] if edges is not None and edges.size > 0 else None
        if hsv is not None and hsv.size > 0:
            slot_hsv = hsv[y1:y2, x1:x2]
            return self.classify_slot(
                slot_hsv,
                norm_cx=norm_cx,
                norm_cy=norm_cy,
                is_hsv=True,
                slot_edges=slot_edges,
                scale_x=scale_x,
            )

        slot_bgr = frame_bgr[y1:y2, x1:x2]
        return self.classify_slot(
            slot_bgr,
            norm_cx=norm_cx,
            norm_cy=norm_cy,
            is_hsv=False,
            slot_edges=slot_edges,
            scale_x=scale_x,
        )

    def scan_hud_slots(
        self,
        frame_bgr: np.ndarray,
        *,
        hsv: Optional[np.ndarray] = None,
        gray: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        max_slots: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Scan the bottom HUD region of a full AION 2 frame and return classified
        consumable (`buff_scroll`, `hp_potion`, `mp_potion`) and `skill_slot` annotations.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 500 or img_h < 300:
            return []

        if hsv is None:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        scale_x = float(img_w) / 2000.0
        y1, y2 = int(img_h * 0.80), int(img_h * 0.98)
        x1, x2 = int(img_w * 0.18), int(img_w * 0.82)
        roi_w = x2 - x1
        skill_roi_x1 = x1 + int(roi_w * 0.20)
        skill_roi_x2 = x2

        skill_roi_gray = gray[y1:y2, skill_roi_x1:skill_roi_x2]
        if skill_roi_gray.size == 0:
            return []

        if edges is not None and edges.shape[:2] == (img_h, img_w):
            roi_edges = edges[y1:y2, skill_roi_x1:skill_roi_x2]
        else:
            roi_edges = cv2.Canny(skill_roi_gray, 50, 150)

        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dil = cv2.dilate(roi_edges, sq_kernel)
        cnts, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_slot_sz = max(18, int(22 * scale_x))
        max_slot_sz = max(40, int(60 * scale_x))
        min_edge_px = max(10, int(15 * scale_x))

        valid_slots: List[Tuple[int, int, int, int]] = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if min_slot_sz <= bw <= max_slot_sz and min_slot_sz <= bh <= max_slot_sz:
                aspect = float(bw) / float(max(1, bh))
                if 0.78 <= aspect <= 1.28:
                    slot_edges = roi_edges[by : by + bh, bx : bx + bw]
                    if int(np.count_nonzero(slot_edges)) >= min_edge_px:
                        valid_slots.append((bx, by, bw, bh))

        results: List[Dict[str, Any]] = []
        for bx, by, bw, bh in valid_slots[:max_slots]:
            abs_x = skill_roi_x1 + bx
            abs_y = y1 + by
            norm_cx = (abs_x + bw * 0.5) / float(img_w)
            norm_cy = (abs_y + bh * 0.5) / float(img_h)
            slot_hsv = hsv[abs_y : abs_y + bh, abs_x : abs_x + bw]
            slot_edges = roi_edges[by : by + bh, bx : bx + bw]

            res = self.classify_slot(
                slot_hsv,
                norm_cx=norm_cx,
                norm_cy=norm_cy,
                is_hsv=True,
                slot_edges=slot_edges,
                scale_x=scale_x,
            )
            if not res.should_label or res.class_id is None:
                continue

            results.append(
                {
                    "class_id": int(res.class_id),
                    "canonical_class_id": int(res.canonical_class_id) if res.canonical_class_id is not None else int(res.class_id),
                    "class_name": res.category,
                    "class_name_tr": res.label_tr,
                    "material_name": res.category,
                    "bbox": (abs_x, abs_y, bw, bh),
                    "bbox_xyxy": (abs_x, abs_y, abs_x + bw, abs_y + bh),
                    "cx": round(norm_cx, 6),
                    "cy": round(norm_cy, 6),
                    "w": round(float(bw) / float(img_w), 6),
                    "h": round(float(bh) / float(img_h), 6),
                    "confidence": float(res.confidence),
                    "source": res.source,
                    "row_type": res.row_type,
                    "trusted": False,
                    "weak": True,
                }
            )

        return results


__all__ = [
    "UPPER_CONSUMABLE_ROW_Y_MIN",
    "UPPER_CONSUMABLE_ROW_Y_MAX",
    "LOWER_SKILL_ROW_Y_MIN",
    "CENTER_EMBLEM_X_MIN",
    "CENTER_EMBLEM_X_MAX",
    "CENTER_EMBLEM_Y_MIN",
    "CENTER_EMBLEM_Y_MAX",
    "YOLO_HUD_CLASS_IDS",
    "CANONICAL_HUD_CLASS_IDS",
    "HUD_CLASS_LABELS_TR",
    "HUD_SOURCE_TAGS",
    "SlotClassificationResult",
    "HUDConsumableClassifier",
]
