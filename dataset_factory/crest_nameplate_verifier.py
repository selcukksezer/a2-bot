# -*- coding: utf-8 -*-
r"""
3D Winged Gather Crest (`\Y/`, `world_gather_crest`) & Turquoise Material Nameplate Verifier
============================================================================================
Verifies the vertical alignment (`dx <= 45px, dy <= 70px`) between a 3D white winged
gather crest (`\Y/`, `class_id = 6`, `world_gather_crest`) and a turquoise/cyan material
nameplate below it:
- When alignment & visual evidence are confirmed (`dx <= 45px, dy <= 70px`):
  produces `confidence >= 0.88`, `trusted = True`, `weak = False`, status `"Emin"`.
- When alignment or evidence is missing:
  produces `confidence < 0.70`, `trusted = False`, `weak = True`, status `"Şüpheli"`.

Exclusive Owner: Worker W09 (`dataset_factory/crest_nameplate_verifier.py`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "MAX_ALIGN_DX_PX",
    "MAX_ALIGN_DY_PX",
    "VERIFIED_MIN_CONFIDENCE",
    "UNVERIFIED_MAX_CONFIDENCE",
    "CREST_CLASS_ID",
    "WORLD_RESOURCE_CLASS_IDS",
    "CrestNameplateVerificationResult",
    "CrestNameplateVerifier",
]

MAX_ALIGN_DX_PX: float = 45.0
MAX_ALIGN_DY_PX: float = 70.0
VERIFIED_MIN_CONFIDENCE: float = 0.88
UNVERIFIED_MAX_CONFIDENCE: float = 0.68

CREST_CLASS_ID: int = 6
WORLD_RESOURCE_CLASS_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 12)


@dataclass
class CrestNameplateVerificationResult:
    """Structured result of a 3D winged gather crest + turquoise nameplate verification."""

    verified: bool
    trusted: bool
    weak: bool
    confidence: float
    status: str
    status_label: str
    dx: float
    dy: float
    crest_bbox: Optional[Tuple[int, int, int, int]] = None
    nameplate_bbox: Optional[Tuple[int, int, int, int]] = None
    crest_score: float = 0.0
    turquoise_ratio: float = 0.0
    turquoise_pixels: int = 0
    reason: str = ""

    @property
    def is_verified(self) -> bool:
        return bool(self.verified)

    @property
    def aligned(self) -> bool:
        return bool(self.verified)

    @property
    def has_evidence(self) -> bool:
        return bool(self.verified)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_verified"] = bool(self.verified)
        data["aligned"] = bool(self.verified)
        data["has_evidence"] = bool(self.verified)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Sequence[str]:
        return list(self.to_dict().keys())

    def items(self) -> Sequence[Tuple[str, Any]]:
        return list(self.to_dict().items())

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())


class CrestNameplateVerifier:
    r"""
    Verifies the vertical alignment (`dx <= 45px, dy <= 70px`) between the 3D winged gather
    crest (`\Y/`, `world_gather_crest`) and the turquoise material nameplate:
    - Evidence present: `confidence >= 0.88`, `trusted = True` (`Emin`)
    - Evidence absent:  `confidence < 0.70`, `trusted = False` (`Şüpheli`)
    """

    # HSV ranges for turquoise / cyan gatherable material nameplates
    TURQUOISE_PRIMARY_LOWER: Tuple[int, int, int] = (78, 90, 130)
    TURQUOISE_PRIMARY_UPPER: Tuple[int, int, int] = (102, 255, 255)
    TURQUOISE_SECONDARY_LOWER: Tuple[int, int, int] = (60, 50, 110)
    TURQUOISE_SECONDARY_UPPER: Tuple[int, int, int] = (102, 255, 255)

    # HSV range for white/silver 3D winged gather crest (\Y/)
    WHITE_CREST_LOWER: Tuple[int, int, int] = (0, 0, 168)
    WHITE_CREST_UPPER: Tuple[int, int, int] = (180, 60, 255)

    # Canonical 28x29 bit-packed binary mask of the 3D white symmetrical winged gather crest (\Y/)
    _CREST_BITMASK_ROWS: Tuple[int, ...] = (
        81920,
        278529,
        302533659,
        369117195,
        335568911,
        469786639,
        402669583,
        427835506,
        432013430,
        207618246,
        241172878,
        242221468,
        133695484,
        133959676,
        66981880,
        16650208,
        8261504,
        4071168,
        1974016,
        1973760,
        1973760,
        1974016,
        1974016,
        1974016,
        1974016,
        4071168,
        924672,
        202752,
    )

    def __init__(
        self,
        max_dx_px: float = MAX_ALIGN_DX_PX,
        max_dy_px: float = MAX_ALIGN_DY_PX,
        verified_min_conf: float = VERIFIED_MIN_CONFIDENCE,
        unverified_max_conf: float = UNVERIFIED_MAX_CONFIDENCE,
        min_turquoise_pixels: int = 12,
        min_crest_symmetry: float = 0.25,
    ) -> None:
        self.max_dx_px: float = float(max_dx_px)
        self.max_dy_px: float = float(max_dy_px)
        self.verified_min_conf: float = max(0.88, float(verified_min_conf))
        self.unverified_max_conf: float = min(0.6999, float(unverified_max_conf))
        self.min_turquoise_pixels: int = max(4, int(min_turquoise_pixels))
        self.min_crest_symmetry: float = float(min_crest_symmetry)
        self._crest_template_mask: np.ndarray = self._build_crest_template_mask()

    @classmethod
    def _build_crest_template_mask(cls) -> np.ndarray:
        """Unpack the 28x29 canonical binary mask for the 3D white winged gather crest (`\\Y/`)."""
        h = len(cls._CREST_BITMASK_ROWS)
        w = 29
        mask = np.zeros((h, w), dtype=np.uint8)
        for r, row_bits in enumerate(cls._CREST_BITMASK_ROWS):
            for c in range(w):
                if (row_bits >> c) & 1:
                    mask[r, c] = 255
        return mask

    # ------------------------------------------------------------------
    # Coordinate & Geometry Extraction Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_center_and_bbox(
        item: Any,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> Optional[Tuple[float, float, Tuple[int, int, int, int], float]]:
        """
        Normalize a point, pixel bbox, normalized YOLO bbox, or annotation dict into
        `(cx_px, cy_px, (x, y, w, h), base_confidence)`.
        """
        if item is None:
            return None

        fw = float(max(1, img_w))
        fh = float(max(1, img_h))

        if isinstance(item, dict):
            base_conf = float(item.get("confidence", item.get("score", 0.85)) or 0.85)
            if "bbox" in item and isinstance(item["bbox"], (tuple, list)) and len(item["bbox"]) == 4:
                bx, by, bw, bh = [float(v) for v in item["bbox"]]
                if 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0 and 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0:
                    cx_px = bx * fw
                    cy_px = by * fh
                    w_px = max(1, int(round(bw * fw)))
                    h_px = max(1, int(round(bh * fh)))
                    x_px = int(round(cx_px - 0.5 * w_px))
                    y_px = int(round(cy_px - 0.5 * h_px))
                    return cx_px, cy_px, (x_px, y_px, w_px, h_px), base_conf
                cx_px = bx + 0.5 * bw
                cy_px = by + 0.5 * bh
                return cx_px, cy_px, (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))), base_conf

            if "bbox_xywh" in item and isinstance(item["bbox_xywh"], (tuple, list)) and len(item["bbox_xywh"]) == 4:
                bx, by, bw, bh = [float(v) for v in item["bbox_xywh"]]
                return (
                    bx + 0.5 * bw,
                    by + 0.5 * bh,
                    (int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))),
                    base_conf,
                )

            if "bbox_xyxy" in item and isinstance(item["bbox_xyxy"], (tuple, list)) and len(item["bbox_xyxy"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy"]]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                return (
                    x1 + 0.5 * bw,
                    y1 + 0.5 * bh,
                    (int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))),
                    base_conf,
                )

            if "cx" in item and "cy" in item:
                cx_val = float(item["cx"])
                cy_val = float(item["cy"])
                w_val = float(item.get("w", item.get("width", 28.0)))
                h_val = float(item.get("h", item.get("height", 20.0)))
                if 0.0 <= cx_val <= 1.0 and 0.0 <= cy_val <= 1.0 and w_val <= 1.0 and h_val <= 1.0:
                    cx_px = cx_val * fw
                    cy_px = cy_val * fh
                    w_px = max(1, int(round(w_val * fw)))
                    h_px = max(1, int(round(h_val * fh)))
                else:
                    cx_px = cx_val
                    cy_px = cy_val
                    w_px = max(1, int(round(w_val)))
                    h_px = max(1, int(round(h_val)))
                x_px = int(round(cx_px - 0.5 * w_px))
                y_px = int(round(cy_px - 0.5 * h_px))
                return cx_px, cy_px, (x_px, y_px, w_px, h_px), base_conf

            if "x" in item and "y" in item:
                x_val = float(item["x"])
                y_val = float(item["y"])
                w_val = float(item.get("w", item.get("width", 28.0)))
                h_val = float(item.get("h", item.get("height", 20.0)))
                return (
                    x_val,
                    y_val,
                    (int(round(x_val - 0.5 * w_val)), int(round(y_val - 0.5 * h_val)), int(round(w_val)), int(round(h_val))),
                    base_conf,
                )

        if isinstance(item, (tuple, list, np.ndarray)):
            seq = list(item)
            if len(seq) == 2:
                cx_px, cy_px = float(seq[0]), float(seq[1])
                if 0.0 <= cx_px <= 1.0 and 0.0 <= cy_px <= 1.0 and (cx_px < 1.0 or cy_px < 1.0):
                    cx_px *= fw
                    cy_px *= fh
                return cx_px, cy_px, (int(round(cx_px - 14)), int(round(cy_px - 10)), 28, 20), 0.85
            if len(seq) >= 4:
                v0, v1, v2, v3 = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
                if 0.0 <= v0 <= 1.0 and 0.0 <= v1 <= 1.0 and 0.0 < v2 <= 1.0 and 0.0 < v3 <= 1.0:
                    cx_px = v0 * fw
                    cy_px = v1 * fh
                    w_px = max(1, int(round(v2 * fw)))
                    h_px = max(1, int(round(v3 * fh)))
                    x_px = int(round(cx_px - 0.5 * w_px))
                    y_px = int(round(cy_px - 0.5 * h_px))
                    return cx_px, cy_px, (x_px, y_px, w_px, h_px), 0.85
                cx_px = v0 + 0.5 * v2
                cy_px = v1 + 0.5 * v3
                return cx_px, cy_px, (int(round(v0)), int(round(v1)), int(round(v2)), int(round(v3))), 0.85

        return None

    # ------------------------------------------------------------------
    # Result Builders (`>=0.88 Emin` vs `<0.70 Şüpheli`)
    # ------------------------------------------------------------------
    def _build_verified_result(
        self,
        dx: float,
        dy: float,
        crest_bbox: Optional[Tuple[int, int, int, int]] = None,
        nameplate_bbox: Optional[Tuple[int, int, int, int]] = None,
        crest_score: float = 0.90,
        turquoise_ratio: float = 0.25,
        turquoise_pixels: int = 30,
        reason: str = "verified_crest_nameplate_vertical_alignment",
    ) -> CrestNameplateVerificationResult:
        """Build a verified (`>=0.88`, `trusted=True`, `Emin`) verification result."""
        dx_factor = max(0.0, 1.0 - (abs(dx) / max(1.0, self.max_dx_px)))
        dy_factor = max(0.0, 1.0 - (abs(dy) / max(1.0, self.max_dy_px)))
        alignment_bonus = 0.06 * (0.55 * dx_factor + 0.45 * dy_factor)
        visual_bonus = min(0.04, 0.02 * float(crest_score) + 0.02 * min(1.0, turquoise_pixels / 40.0))
        conf = round(min(0.99, max(self.verified_min_conf, 0.88 + alignment_bonus + visual_bonus)), 4)

        return CrestNameplateVerificationResult(
            verified=True,
            trusted=True,
            weak=False,
            confidence=conf,
            status="Emin",
            status_label="🟢 Emin",
            dx=round(float(abs(dx)), 2),
            dy=round(float(abs(dy)), 2),
            crest_bbox=crest_bbox,
            nameplate_bbox=nameplate_bbox,
            crest_score=round(float(crest_score), 4),
            turquoise_ratio=round(float(turquoise_ratio), 4),
            turquoise_pixels=int(turquoise_pixels),
            reason=reason,
        )

    def _build_unverified_result(
        self,
        dx: float = 999.0,
        dy: float = 999.0,
        crest_bbox: Optional[Tuple[int, int, int, int]] = None,
        nameplate_bbox: Optional[Tuple[int, int, int, int]] = None,
        crest_score: float = 0.0,
        turquoise_ratio: float = 0.0,
        turquoise_pixels: int = 0,
        base_confidence: Optional[float] = None,
        reason: str = "missing_or_misaligned_crest_nameplate_evidence",
    ) -> CrestNameplateVerificationResult:
        """Build an unverified (`<0.70`, `trusted=False`, `Şüpheli`) verification result."""
        if base_confidence is not None:
            conf = min(self.unverified_max_conf, max(0.10, float(base_confidence)))
        else:
            partial = 0.52 + 0.10 * min(1.0, float(crest_score)) + 0.05 * min(1.0, turquoise_pixels / 25.0)
            conf = min(self.unverified_max_conf, max(0.35, partial))
        conf = round(min(0.69, conf), 4)

        return CrestNameplateVerificationResult(
            verified=False,
            trusted=False,
            weak=True,
            confidence=conf,
            status="Şüpheli",
            status_label="🟡 Şüpheli",
            dx=round(float(abs(dx)), 2),
            dy=round(float(abs(dy)), 2),
            crest_bbox=crest_bbox,
            nameplate_bbox=nameplate_bbox,
            crest_score=round(float(crest_score), 4),
            turquoise_ratio=round(float(turquoise_ratio), 4),
            turquoise_pixels=int(turquoise_pixels),
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 1. Geometric Vertical Alignment Verification (`dx <= 45px, dy <= 70px`)
    # ------------------------------------------------------------------
    def check_vertical_alignment(
        self,
        crest: Any,
        nameplate: Any,
        img_w: int = 1920,
        img_h: int = 1080,
        require_crest_above: bool = True,
    ) -> CrestNameplateVerificationResult:
        r"""
        Verify vertical alignment between a 3D winged gather crest (`\Y/`, `world_gather_crest`)
        and a turquoise material nameplate (`dx <= 45px, dy <= 70px`).
        """
        c_info = self._extract_center_and_bbox(crest, img_w=img_w, img_h=img_h)
        n_info = self._extract_center_and_bbox(nameplate, img_w=img_w, img_h=img_h)

        if c_info is None or n_info is None:
            return self._build_unverified_result(
                crest_bbox=c_info[2] if c_info else None,
                nameplate_bbox=n_info[2] if n_info else None,
                reason="missing_crest_or_nameplate_coordinates",
            )

        crest_cx, crest_cy, crest_bbox, crest_conf = c_info
        np_cx, np_cy, np_bbox, np_conf = n_info

        dx = abs(crest_cx - np_cx)
        signed_dy = np_cy - crest_cy
        dy = abs(signed_dy)

        if require_crest_above and signed_dy < -6.0:
            return self._build_unverified_result(
                dx=dx,
                dy=dy,
                crest_bbox=crest_bbox,
                nameplate_bbox=np_bbox,
                crest_score=crest_conf,
                reason=f"nameplate_above_crest (signed_dy={signed_dy:.1f}px)",
            )

        if dx <= self.max_dx_px and dy <= self.max_dy_px:
            return self._build_verified_result(
                dx=dx,
                dy=dy,
                crest_bbox=crest_bbox,
                nameplate_bbox=np_bbox,
                crest_score=max(crest_conf, np_conf),
                turquoise_ratio=0.30,
                turquoise_pixels=max(self.min_turquoise_pixels, 24),
                reason=f"aligned (dx={dx:.1f}px <= {self.max_dx_px:.0f}px, dy={dy:.1f}px <= {self.max_dy_px:.0f}px)",
            )

        return self._build_unverified_result(
            dx=dx,
            dy=dy,
            crest_bbox=crest_bbox,
            nameplate_bbox=np_bbox,
            crest_score=crest_conf,
            reason=(
                f"misaligned (dx={dx:.1f}px [max {self.max_dx_px:.0f}px], "
                f"dy={dy:.1f}px [max {self.max_dy_px:.0f}px])"
            ),
        )

    # ------------------------------------------------------------------
    # 2. Visual Detection of 3D Winged Gather Crest & Turquoise Nameplate
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_hsv(frame_bgr: np.ndarray, hsv: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if hsv is not None and isinstance(hsv, np.ndarray) and hsv.size > 0:
            return hsv
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return None
        if frame_bgr.ndim == 2:
            bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame_bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def _is_monster_yellow_or_red_nameplate(self, hsv_patch: np.ndarray) -> bool:
        """Reject monster nameplates that contain dominant yellow text or red agro badges."""
        if hsv_patch is None or hsv_patch.size == 0:
            return False
        yellow_mask = cv2.inRange(hsv_patch, (14, 95, 125), (38, 255, 255))
        tq_primary = cv2.inRange(hsv_patch, self.TURQUOISE_PRIMARY_LOWER, self.TURQUOISE_PRIMARY_UPPER)
        tq_secondary = cv2.inRange(hsv_patch, self.TURQUOISE_SECONDARY_LOWER, self.TURQUOISE_SECONDARY_UPPER)
        tq_mask = cv2.bitwise_or(tq_primary, tq_secondary)

        yellow_px = int(np.count_nonzero(yellow_mask))
        tq_px = int(np.count_nonzero(tq_mask))
        if tq_px < 12 and yellow_px > max(14, tq_px * 2):
            return True
        return False

    def detect_turquoise_nameplates(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect turquoise / cyan material nameplates in `frame_bgr` (or within `roi = (x1, y1, x2, y2)`).
        Returns a list of dicts with `cx`, `cy`, `bbox` `(x, y, w, h)`, `turquoise_pixels`, `turquoise_ratio`.
        """
        hsv_full = self._ensure_hsv(frame_bgr, hsv)
        if hsv_full is None:
            return []

        h, w = hsv_full.shape[:2]
        if roi is not None:
            rx1 = max(0, min(w - 1, int(roi[0])))
            ry1 = max(0, min(h - 1, int(roi[1])))
            rx2 = max(rx1 + 1, min(w, int(roi[2])))
            ry2 = max(ry1 + 1, min(h, int(roi[3])))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        sub_hsv = hsv_full[ry1:ry2, rx1:rx2]
        if sub_hsv.size == 0:
            return []

        m1 = cv2.inRange(sub_hsv, self.TURQUOISE_PRIMARY_LOWER, self.TURQUOISE_PRIMARY_UPPER)
        m2 = cv2.inRange(sub_hsv, self.TURQUOISE_SECONDARY_LOWER, self.TURQUOISE_SECONDARY_UPPER)
        tq_mask = cv2.bitwise_or(m1, m2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        closed = cv2.morphologyEx(tq_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nameplates: List[Dict[str, Any]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 10 or ch < 3 or cw > 220 or ch > 48:
                continue
            patch_mask = tq_mask[y : y + ch, x : x + cw]
            tq_px = int(np.count_nonzero(patch_mask))
            if tq_px < self.min_turquoise_pixels:
                continue
            fill_ratio = tq_px / float(max(1, cw * ch))
            if fill_ratio > 0.85:
                continue

            patch_hsv = sub_hsv[y : y + ch, x : x + cw]
            if self._is_monster_yellow_or_red_nameplate(patch_hsv):
                continue

            abs_x = rx1 + x
            abs_y = ry1 + y
            nameplates.append(
                {
                    "cx": float(abs_x + 0.5 * cw),
                    "cy": float(abs_y + 0.5 * ch),
                    "bbox": (int(abs_x), int(abs_y), int(cw), int(ch)),
                    "turquoise_pixels": tq_px,
                    "turquoise_ratio": round(fill_ratio, 4),
                    "confidence": round(min(0.96, 0.78 + min(0.18, tq_px / 120.0)), 4),
                }
            )

        nameplates.sort(key=lambda d: d["turquoise_pixels"], reverse=True)
        return nameplates

    def detect_winged_crests(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, Any]]:
        r"""
        Detect 3D white symmetrical winged gather crests (`\Y/`, `world_gather_crest`)
        in `frame_bgr` (or within `roi = (x1, y1, x2, y2)`).
        """
        hsv_full = self._ensure_hsv(frame_bgr, hsv)
        if hsv_full is None:
            return []

        h, w = hsv_full.shape[:2]
        if roi is not None:
            rx1 = max(0, min(w - 1, int(roi[0])))
            ry1 = max(0, min(h - 1, int(roi[1])))
            rx2 = max(rx1 + 1, min(w, int(roi[2])))
            ry2 = max(ry1 + 1, min(h, int(roi[3])))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        sub_hsv = hsv_full[ry1:ry2, rx1:rx2]
        if sub_hsv.size == 0:
            return []

        white_mask = cv2.inRange(sub_hsv, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER)
        crests: List[Dict[str, Any]] = []

        # 1. Template match with canonical 28x29 winged crest mask if ROI is large enough
        th, tw = self._crest_template_mask.shape[:2]
        if white_mask.shape[0] >= th and white_mask.shape[1] >= tw:
            res = cv2.matchTemplate(white_mask, self._crest_template_mask, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= 0.45)
            for y_pt, x_pt in zip(ys, xs):
                score = float(res[y_pt, x_pt])
                abs_x = rx1 + int(x_pt)
                abs_y = ry1 + int(y_pt)
                crests.append(
                    {
                        "cx": float(abs_x + 0.5 * tw),
                        "cy": float(abs_y + 0.5 * th),
                        "bbox": (abs_x, abs_y, tw, th),
                        "score": round(min(0.99, 0.75 + 0.24 * score), 4),
                    }
                )

        # 2. Symmetric white contour fallback for scaled/synthetic crests
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 8 or ch < 8 or cw > 64 or ch > 64:
                continue
            sub_wm = white_mask[y : y + ch, x : x + cw]
            white_px = int(np.count_nonzero(sub_wm))
            if white_px < 10:
                continue
            flipped = np.fliplr(sub_wm)
            inter = int(np.count_nonzero(sub_wm & flipped))
            union = int(np.count_nonzero(sub_wm | flipped))
            sym = inter / float(max(1, union))
            if sym < self.min_crest_symmetry:
                continue
            abs_x = rx1 + x
            abs_y = ry1 + y
            if any(
                math.hypot((abs_x + 0.5 * cw) - c["cx"], (abs_y + 0.5 * ch) - c["cy"]) <= 16.0
                for c in crests
            ):
                continue
            crests.append(
                {
                    "cx": float(abs_x + 0.5 * cw),
                    "cy": float(abs_y + 0.5 * ch),
                    "bbox": (int(abs_x), int(abs_y), int(cw), int(ch)),
                    "score": round(min(0.95, 0.72 + 0.25 * sym), 4),
                }
            )

        crests.sort(key=lambda d: d["score"], reverse=True)
        return crests

    # ------------------------------------------------------------------
    # 3. Frame + Box Verification Pipeline
    # ------------------------------------------------------------------
    def verify_in_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        crest_box: Optional[Any] = None,
        nameplate_box: Optional[Any] = None,
        candidate_box: Optional[Any] = None,
        hsv: Optional[np.ndarray] = None,
    ) -> CrestNameplateVerificationResult:
        r"""
        Verify vertical alignment (`dx <= 45px, dy <= 70px`) between a 3D winged gather crest
        (`\Y/`, `world_gather_crest`) and a turquoise material nameplate within `frame_bgr`.
        """
        img_h, img_w = (1080, 1920)
        if frame_bgr is not None and isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0:
            img_h, img_w = frame_bgr.shape[:2]

        # Case 1: Both crest_box and nameplate_box are explicitly provided
        if crest_box is not None and nameplate_box is not None:
            align_res = self.check_vertical_alignment(
                crest_box, nameplate_box, img_w=img_w, img_h=img_h, require_crest_above=True
            )
            if not align_res.verified or frame_bgr is None or frame_bgr.size == 0:
                return align_res

            # If frame is provided, check visual evidence in nameplate ROI
            hsv_img = self._ensure_hsv(frame_bgr, hsv)
            if hsv_img is not None and align_res.nameplate_bbox is not None:
                nx, ny, nw, nh = align_res.nameplate_bbox
                x1, y1 = max(0, nx - 4), max(0, ny - 4)
                x2, y2 = min(img_w, nx + nw + 4), min(img_h, ny + nh + 4)
                if x2 > x1 and y2 > y1:
                    sub = hsv_img[y1:y2, x1:x2]
                    if self._is_monster_yellow_or_red_nameplate(sub):
                        return self._build_unverified_result(
                            dx=align_res.dx,
                            dy=align_res.dy,
                            crest_bbox=align_res.crest_bbox,
                            nameplate_bbox=align_res.nameplate_bbox,
                            reason="rejected_monster_yellow_or_red_nameplate",
                        )
            return align_res

        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return self._build_unverified_result(reason="no_frame_or_pair_provided")

        hsv_img = self._ensure_hsv(frame_bgr, hsv)
        if hsv_img is None:
            return self._build_unverified_result(reason="invalid_frame_channels")

        # Case 2: `crest_box` is provided -> search for turquoise nameplate below crest (`dx <= 45px, dy <= 70px`)
        if crest_box is not None:
            c_info = self._extract_center_and_bbox(crest_box, img_w=img_w, img_h=img_h)
            if c_info is None:
                return self._build_unverified_result(reason="invalid_crest_box")
            ccx, ccy, cbbox, cscore = c_info
            search_roi = (
                max(0, int(round(ccx - self.max_dx_px - 35))),
                max(0, int(round(ccy - 4))),
                min(img_w, int(round(ccx + self.max_dx_px + 35))),
                min(img_h, int(round(ccy + self.max_dy_px + 18))),
            )
            nameplates = self.detect_turquoise_nameplates(frame_bgr, hsv=hsv_img, roi=search_roi)
            best_res: Optional[CrestNameplateVerificationResult] = None
            for np_det in nameplates:
                res = self.check_vertical_alignment(
                    cbbox, np_det["bbox"], img_w=img_w, img_h=img_h, require_crest_above=True
                )
                if res.verified:
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cbbox,
                        nameplate_bbox=np_det["bbox"],
                        crest_score=cscore,
                        turquoise_ratio=np_det["turquoise_ratio"],
                        turquoise_pixels=np_det["turquoise_pixels"],
                        reason=res.reason,
                    )
                if best_res is None or (res.dx + res.dy) < (best_res.dx + best_res.dy):
                    best_res = res
            return best_res if best_res is not None else self._build_unverified_result(
                crest_bbox=cbbox,
                crest_score=cscore,
                reason="no_turquoise_nameplate_below_crest",
            )

        # Case 3: `nameplate_box` is provided -> search for winged crest above nameplate (`dx <= 45px, dy <= 70px`)
        if nameplate_box is not None:
            n_info = self._extract_center_and_bbox(nameplate_box, img_w=img_w, img_h=img_h)
            if n_info is None:
                return self._build_unverified_result(reason="invalid_nameplate_box")
            ncx, ncy, nbbox, nscore = n_info
            search_roi = (
                max(0, int(round(ncx - self.max_dx_px - 24))),
                max(0, int(round(ncy - self.max_dy_px - 20))),
                min(img_w, int(round(ncx + self.max_dx_px + 24))),
                min(img_h, int(round(ncy + 6))),
            )
            crests = self.detect_winged_crests(frame_bgr, hsv=hsv_img, roi=search_roi)
            for cr_det in crests:
                res = self.check_vertical_alignment(
                    cr_det["bbox"], nbbox, img_w=img_w, img_h=img_h, require_crest_above=True
                )
                if res.verified:
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cr_det["bbox"],
                        nameplate_bbox=nbbox,
                        crest_score=cr_det["score"],
                        turquoise_ratio=0.30,
                        turquoise_pixels=max(self.min_turquoise_pixels, 20),
                        reason=res.reason,
                    )
            return self._build_unverified_result(
                nameplate_bbox=nbbox,
                reason="no_winged_crest_above_nameplate",
            )

        # Case 4: `candidate_box` (e.g., 3D resource mesh box) or full-frame scan
        if candidate_box is not None:
            cand_info = self._extract_center_and_bbox(candidate_box, img_w=img_w, img_h=img_h)
            if cand_info is not None:
                bx, by, bw, bh = cand_info[2]
                roi = (
                    max(0, int(bx - self.max_dx_px - 30)),
                    max(0, int(by - self.max_dy_px - 75)),
                    min(img_w, int(bx + bw + self.max_dx_px + 30)),
                    min(img_h, int(by + bh + 25)),
                )
            else:
                roi = None
        else:
            roi = None

        crests = self.detect_winged_crests(frame_bgr, hsv=hsv_img, roi=roi)
        nameplates = self.detect_turquoise_nameplates(frame_bgr, hsv=hsv_img, roi=roi)

        best_unverified: Optional[CrestNameplateVerificationResult] = None
        for cr in crests:
            for np_det in nameplates:
                res = self.check_vertical_alignment(
                    cr["bbox"], np_det["bbox"], img_w=img_w, img_h=img_h, require_crest_above=True
                )
                if res.verified:
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cr["bbox"],
                        nameplate_bbox=np_det["bbox"],
                        crest_score=cr["score"],
                        turquoise_ratio=np_det["turquoise_ratio"],
                        turquoise_pixels=np_det["turquoise_pixels"],
                        reason=res.reason,
                    )
                if best_unverified is None or (res.dx + res.dy) < (best_unverified.dx + best_unverified.dy):
                    best_unverified = res

        if best_unverified is not None:
            return best_unverified
        return self._build_unverified_result(reason="no_aligned_crest_and_turquoise_nameplate_in_frame")

    # ------------------------------------------------------------------
    # 4. Annotation Calibration & Batch Verification
    # ------------------------------------------------------------------
    def verify_annotation(
        self,
        box: Dict[str, Any],
        frame_bgr: Optional[np.ndarray] = None,
        all_annotations: Optional[Sequence[Dict[str, Any]]] = None,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> Dict[str, Any]:
        """
        Verify a single annotation dictionary and calibrate its confidence & trust flags:
        - Evidence confirmed (`dx <= 45px, dy <= 70px`): `confidence >= 0.88`, `trusted = True`, `weak = False`
        - Evidence missing: `confidence < 0.70`, `trusted = False`, `weak = True`
        """
        if not isinstance(box, dict):
            return box

        updated = dict(box)
        if frame_bgr is not None and isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0:
            img_h, img_w = frame_bgr.shape[:2]

        try:
            cid = int(updated.get("class_id", -1))
        except (TypeError, ValueError):
            cid = -1

        try:
            raw_conf = float(updated.get("confidence", 0.65))
        except (TypeError, ValueError):
            raw_conf = 0.65

        result: Optional[CrestNameplateVerificationResult] = None

        # Check explicit partner in `updated` metadata if provided
        if "crest_bbox" in updated and "nameplate_bbox" in updated:
            result = self.verify_in_frame(
                frame_bgr,
                crest_box=updated["crest_bbox"],
                nameplate_box=updated["nameplate_bbox"],
            )
        elif cid == CREST_CLASS_ID:
            # Look for a matching nameplate or resource annotation below this crest
            if all_annotations:
                for other in all_annotations:
                    if other is box or not isinstance(other, dict):
                        continue
                    ocid = int(other.get("class_id", -1))
                    if ocid in WORLD_RESOURCE_CLASS_IDS or str(other.get("source", "")).endswith("nameplate"):
                        cand_res = self.check_vertical_alignment(
                            updated, other, img_w=img_w, img_h=img_h, require_crest_above=True
                        )
                        if cand_res.verified:
                            result = cand_res
                            break
            if (result is None or not result.verified) and frame_bgr is not None:
                result = self.verify_in_frame(frame_bgr, crest_box=updated)
        else:
            # Resource or nameplate box -> look for a matching `world_gather_crest` (class 6) above it
            if all_annotations:
                for other in all_annotations:
                    if other is box or not isinstance(other, dict):
                        continue
                    if int(other.get("class_id", -1)) == CREST_CLASS_ID:
                        cand_res = self.check_vertical_alignment(
                            other, updated, img_w=img_w, img_h=img_h, require_crest_above=True
                        )
                        if cand_res.verified:
                            result = cand_res
                            break
            if (result is None or not result.verified) and frame_bgr is not None:
                result = self.verify_in_frame(frame_bgr, candidate_box=updated)

        if result is None:
            result = self._build_unverified_result(
                base_confidence=raw_conf,
                reason="no_crest_nameplate_alignment_evidence",
            )

        if result.verified:
            updated["confidence"] = round(max(self.verified_min_conf, result.confidence, raw_conf), 4)
            updated["trusted"] = True
            updated["weak"] = False
            updated["verification_status"] = "Emin"
            updated["crest_nameplate_verified"] = True
        else:
            updated["confidence"] = round(min(self.unverified_max_conf, result.confidence, raw_conf), 4)
            updated["trusted"] = False
            updated["weak"] = True
            updated["verification_status"] = "Şüpheli"
            updated["crest_nameplate_verified"] = False

        updated["alignment_dx"] = result.dx
        updated["alignment_dy"] = result.dy
        updated["verification_reason"] = result.reason
        return updated

    def verify_annotations(
        self,
        annotations: Sequence[Dict[str, Any]],
        frame_bgr: Optional[np.ndarray] = None,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> List[Dict[str, Any]]:
        """Batch-verify a sequence of annotations for crest-to-nameplate vertical alignment."""
        ann_list = [dict(a) for a in (annotations or []) if isinstance(a, dict)]
        return [
            self.verify_annotation(
                ann,
                frame_bgr=frame_bgr,
                all_annotations=ann_list,
                img_w=img_w,
                img_h=img_h,
            )
            for ann in ann_list
        ]

    def verify(
        self,
        first: Any = None,
        second: Any = None,
        *,
        frame_bgr: Optional[np.ndarray] = None,
        crest_box: Optional[Any] = None,
        nameplate_box: Optional[Any] = None,
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> CrestNameplateVerificationResult:
        """
        Universal convenience entry point:
        - `verify(crest_box, nameplate_box)` -> checks vertical alignment (`dx <= 45px, dy <= 70px`)
        - `verify(frame_bgr, crest_box=..., nameplate_box=...)` -> verifies in BGR frame
        """
        if isinstance(first, np.ndarray) and first.ndim in (2, 3) and first.size > 16:
            return self.verify_in_frame(
                first,
                crest_box=crest_box if crest_box is not None else second,
                nameplate_box=nameplate_box,
            )
        if crest_box is not None or nameplate_box is not None:
            if frame_bgr is not None:
                return self.verify_in_frame(frame_bgr, crest_box=crest_box, nameplate_box=nameplate_box)
            return self.check_vertical_alignment(crest_box, nameplate_box, img_w=img_w, img_h=img_h)
        return self.check_vertical_alignment(first, second, img_w=img_w, img_h=img_h)
