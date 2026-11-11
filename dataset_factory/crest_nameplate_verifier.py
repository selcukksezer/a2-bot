# -*- coding: utf-8 -*-
r"""
3D Winged Gather Crest (`\Y/`, `world_gather_crest`) & Turquoise Material Nameplate Verifier
============================================================================================
Verifies the vertical alignment (`dx <= 45px, dy <= 70px`) between a 3D white winged
gather crest (`\Y/`, `class_id = 13`, `world_gather_crest`) and a turquoise/cyan material
nameplate below it:
- When alignment & visual evidence are confirmed (`dx <= 45px, dy <= 70px`):
  produces `confidence >= 0.88`, `trusted = True`, `weak = False`, status `"Emin"`.
- When alignment or evidence is missing:
  produces `confidence < 0.70`, `trusted = False`, `weak = True`, status `"Şüpheli"`.

Exclusive Owner: Worker W09 (`dataset_factory/crest_nameplate_verifier.py`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields as _dataclass_fields
import math
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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

# A1: the crest is canonical class 13 (skill_ui_prompt). The old value 6
# was the LEGACY scheme's crest id; canonically 6 is herb_edana, so the
# verifier used to treat an Edana plant annotation as a crest. Verified
# against dataset/data.yaml (13: skill_ui_prompt) and
# class_taxonomy.LEGACY_YOLO_CLASS_TO_CANONICAL (6 -> 13).
CREST_CLASS_ID: int = 13
# Canonical 3D material partners of a crest: every 3D material class
# (0..8, 12). The old tuple omitted 6/7/8 because in the legacy scheme
# they were UI icons/prompts, not materials — canonically 6/7/8 ARE
# materials (herb_edana / herb_lumina / ore_obsidian), so a real Lumina
# or Obsidian node can now be a crest partner at all.
WORLD_RESOURCE_CLASS_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)


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
    # F-0787: geometry alignment alone (dx/dy within tolerance) — True
    # even when no turquoise evidence was measured, in which case the
    # result is NOT verified (Şüpheli, confidence < 0.70).
    geometry_aligned: bool = False

    @property
    def is_verified(self) -> bool:
        return bool(self.verified)

    @property
    def aligned(self) -> bool:
        # F-0787: "aligned" is geometry only — it does NOT mean verified.
        # Verification additionally requires measured turquoise evidence.
        return bool(self.geometry_aligned)

    @property
    def has_evidence(self) -> bool:
        return bool(self.verified)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # D45 public dict view: 14 fields + 3 derived keys (17 total).
        # `geometry_aligned` (F-0787, R48) is internal bookkeeping whose
        # public alias is the derived `aligned` key (S-4801, below); it
        # stays readable as the `.geometry_aligned` attribute but is not
        # a dict key, so keys()/to_dict()/items() stay in agreement.
        for name in self._DICT_HIDDEN_FIELDS:
            data.pop(name, None)
        data["is_verified"] = bool(self.verified)
        # S-4801: "aligned" is geometry only (F-0787), matching the
        # .aligned property — it must NOT mirror .verified, or a
        # geometry-only aligned pair reports True via .aligned but
        # False via res['aligned'] / to_dict()['aligned'].
        data["aligned"] = bool(self.geometry_aligned)
        data["has_evidence"] = bool(self.verified)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    # Dict-like key enumeration without building a full value dict: the old
    # `keys()` / `__contains__` / `__iter__` each ran `asdict(self)` (a full
    # copy of all fields + 3 derived keys) just to answer "which keys
    # exist". Key membership/order never depends on values, so it is answered
    # from the dataclass field map (`__dataclass_fields__`, a plain dict)
    # directly. The public key set is the dataclass fields minus
    # `_DICT_HIDDEN_FIELDS`, plus `_DERIVED_KEYS` (14 + 3 = 17 keys).
    # `to_dict()` / `__getitem__` / `get` / `items` still go through
    # `asdict` for values, with the hidden fields removed afterwards.
    _DERIVED_KEYS: ClassVar[Tuple[str, ...]] = ("is_verified", "aligned", "has_evidence")

    # Dataclass fields hidden from the public dict view (see to_dict):
    # `geometry_aligned` is exposed under the derived `aligned` key.
    _DICT_HIDDEN_FIELDS: ClassVar[Tuple[str, ...]] = ("geometry_aligned",)

    def keys(self) -> Sequence[str]:
        return [
            f.name
            for f in _dataclass_fields(self)
            if f.name not in self._DICT_HIDDEN_FIELDS
        ] + list(self._DERIVED_KEYS)

    def items(self) -> Sequence[Tuple[str, Any]]:
        return list(self.to_dict().items())

    def __contains__(self, key: object) -> bool:
        if key in self._DERIVED_KEYS:
            return True
        return any(
            f.name == key
            for f in _dataclass_fields(self)
            if f.name not in self._DICT_HIDDEN_FIELDS
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


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
        # S-2401: every float() conversion below is unguarded — one
        # corrupt annotation used to raise ValueError and kill the
        # whole batch. A bad coordinate now yields None (callers
        # turn it into an unverified result) instead of raising.
        try:

            fw = float(max(1, img_w))
            fh = float(max(1, img_h))

            if isinstance(item, dict):
                # F-0920: a confidence of exactly 0.0 is a real measurement, not
                # a missing value — `or 0.85` used to inflate it to 0.85.
                # A8: a MISSING confidence is unmeasured, never 0.85 — the
                # old 0.85 default fabricated "measured" confidence for
                # confidence-less inputs, and it flowed into crest_score.
                raw_conf = item.get("confidence", item.get("score", 0.0))
                if raw_conf is None:
                    raw_conf = 0.0
                try:
                    base_conf = float(raw_conf)
                except (TypeError, ValueError):
                    base_conf = 0.0
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
                    has_size = any(k in item for k in ("w", "width", "h", "height"))
                    w_val = float(item.get("w", item.get("width", 28.0)))
                    h_val = float(item.get("h", item.get("height", 20.0)))
                    # A29: a centre in [0, 1] without an explicit size is a
                    # NORMALIZED centre (same contract as the "bbox" branch and
                    # the 2-tuple branch). Previously it fell through to the
                    # pixel branch because the default w/h (28/20) are > 1.0,
                    # so {"cx": 0.5, "cy": 0.5} resolved to pixel (0.5, 0.5)
                    # instead of (960, 540) on a 1920x1080 frame.
                    if (
                        0.0 <= cx_val <= 1.0
                        and 0.0 <= cy_val <= 1.0
                        and (not has_size or (w_val <= 1.0 and h_val <= 1.0))
                    ):
                        cx_px = cx_val * fw
                        cy_px = cy_val * fh
                        w_px = max(1, int(round(w_val * fw))) if has_size else 28
                        h_px = max(1, int(round(h_val * fh))) if has_size else 20
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
                    has_size = any(k in item for k in ("w", "width", "h", "height"))
                    w_val = float(item.get("w", item.get("width", 28.0)))
                    h_val = float(item.get("h", item.get("height", 20.0)))
                    # A29: same normalized-centre rule as the "cx"/"cy" branch
                    # above — {"x": 0.5, "y": 0.5} with no explicit size is a
                    # normalized centre, not pixel (0.5, 0.5).
                    if (
                        0.0 <= x_val <= 1.0
                        and 0.0 <= y_val <= 1.0
                        and (not has_size or (w_val <= 1.0 and h_val <= 1.0))
                    ):
                        x_val = x_val * fw
                        y_val = y_val * fh
                        w_val = w_val * fw if has_size else 28.0
                        h_val = h_val * fh if has_size else 20.0
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
                    # A8: a bare coordinate tuple carries no confidence —
                    # unmeasured confidence is 0.0, never a fabricated 0.85.
                    return cx_px, cy_px, (int(round(cx_px - 14)), int(round(cy_px - 10)), 28, 20), 0.0
                if len(seq) >= 4:
                    v0, v1, v2, v3 = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
                    if 0.0 <= v0 <= 1.0 and 0.0 <= v1 <= 1.0 and 0.0 < v2 <= 1.0 and 0.0 < v3 <= 1.0:
                        cx_px = v0 * fw
                        cy_px = v1 * fh
                        w_px = max(1, int(round(v2 * fw)))
                        h_px = max(1, int(round(v3 * fh)))
                        x_px = int(round(cx_px - 0.5 * w_px))
                        y_px = int(round(cy_px - 0.5 * h_px))
                        return cx_px, cy_px, (x_px, y_px, w_px, h_px), 0.0
                    cx_px = v0 + 0.5 * v2
                    cy_px = v1 + 0.5 * v3
                    return cx_px, cy_px, (int(round(v0)), int(round(v1)), int(round(v2)), int(round(v3))), 0.0

            return None
        except (TypeError, ValueError, OverflowError):
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
        # A3: NO fabricated evidence defaults — the old crest_score=0.90 /
        # turquoise_ratio=0.25 / turquoise_pixels=30 let a bare builder call
        # mint an Emin with measured-looking evidence. Every internal caller
        # passes measured values; a caller that omits them gets zeros.
        crest_score: float = 0.0,
        turquoise_ratio: float = 0.0,
        turquoise_pixels: int = 0,
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
            geometry_aligned=True,
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
        geometry_aligned: bool = False,
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
            geometry_aligned=geometry_aligned,
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
        frame_bgr: Optional[np.ndarray] = None,
        hsv: Optional[np.ndarray] = None,
    ) -> CrestNameplateVerificationResult:
        r"""
        Verify vertical alignment between a 3D winged gather crest (`\Y/`, `world_gather_crest`)
        and a turquoise material nameplate (`dx <= 45px, dy <= 70px`).

        F-0787 (single verification path): geometry alone is NEVER
        verification, and a dict-carried ``turquoise_pixels`` /
        ``turquoise_ratio`` is NEVER evidence either — the old
        ``_extract_turquoise_evidence`` trusted numbers any caller wrote
        into a dict, so a fabricated ``{"turquoise_pixels": 999, ...}``
        minted an Emin without a single measured pixel (A3). Verification
        now requires MEASURED evidence from a frame: pass ``frame_bgr``
        (or a precomputed ``hsv``) and the turquoise nameplate pixels and
        the white crest pixels are counted in their ROIs
        (``turquoise_pixels >= min_turquoise_pixels`` AND
        ``white_crest_pixels >= 10`` — the same bar ``verify_in_frame``
        Case 1 applies). Without a frame there is nothing to measure, so
        an aligned pair returns ``verified=False`` / ``Şüpheli`` with
        ``geometry_aligned=True`` and honest ``turquoise_pixels=0``.
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
            # F-0787/A3: geometry alone is NOT verification. Verified/Emin
            # requires evidence MEASURED from a frame in both ROIs — the
            # turquoise nameplate pixels AND the white crest pixels (A4:
            # the crest itself must be seen, not just claimed by a box).
            # Dict-carried turquoise numbers are ignored entirely.
            hsv_img = self._ensure_hsv(frame_bgr, hsv) if (
                hsv is not None or frame_bgr is not None
            ) else None
            if hsv_img is not None:
                tq_px, tq_ratio, white_px = self._measure_roi_visual_evidence(
                    hsv_img, crest_bbox, np_bbox
                )
                if tq_px >= self.min_turquoise_pixels and white_px >= 10:
                    return self._build_verified_result(
                        dx=dx,
                        dy=dy,
                        crest_bbox=crest_bbox,
                        nameplate_bbox=np_bbox,
                        crest_score=max(crest_conf, np_conf),
                        turquoise_ratio=tq_ratio,
                        turquoise_pixels=tq_px,
                        reason=(
                            f"aligned (dx={dx:.1f}px <= {self.max_dx_px:.0f}px, "
                            f"dy={dy:.1f}px <= {self.max_dy_px:.0f}px) with measured "
                            f"turquoise evidence ({tq_px}px) and white crest "
                            f"evidence ({white_px}px)"
                        ),
                    )
            return self._build_unverified_result(
                dx=dx,
                dy=dy,
                crest_bbox=crest_bbox,
                nameplate_bbox=np_bbox,
                crest_score=max(crest_conf, np_conf),
                turquoise_ratio=0.0,
                turquoise_pixels=0,
                geometry_aligned=True,
                reason=(
                    f"aligned_geometry_only_no_turquoise_evidence "
                    f"(dx={dx:.1f}px <= {self.max_dx_px:.0f}px, "
                    f"dy={dy:.1f}px <= {self.max_dy_px:.0f}px; geometry alone "
                    "is not verification — no frame and no measured turquoise)"
                ),
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
        # F-0921: red agro badges were never checked — red wraps the hue
        # circle, so both ends (0-10 and 170-180) must be masked.
        red_mask_lo = cv2.inRange(hsv_patch, (0, 95, 125), (10, 255, 255))
        red_mask_hi = cv2.inRange(hsv_patch, (170, 95, 125), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_lo, red_mask_hi)
        tq_primary = cv2.inRange(hsv_patch, self.TURQUOISE_PRIMARY_LOWER, self.TURQUOISE_PRIMARY_UPPER)
        tq_secondary = cv2.inRange(hsv_patch, self.TURQUOISE_SECONDARY_LOWER, self.TURQUOISE_SECONDARY_UPPER)
        tq_mask = cv2.bitwise_or(tq_primary, tq_secondary)

        yellow_px = int(np.count_nonzero(yellow_mask))
        red_px = int(np.count_nonzero(red_mask))
        tq_px = int(np.count_nonzero(tq_mask))
        if tq_px < 12 and yellow_px > max(14, tq_px * 2):
            return True
        if tq_px < 12 and red_px > max(14, tq_px * 2):
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

    @staticmethod
    def _template_search_rects(
        white_contours: Sequence[np.ndarray],
        mask_w: int,
        mask_h: int,
        th: int,
        tw: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Bounding rects that can contain a `>= 0.45` template-match window.

        A `TM_CCOEFF_NORMED` window scores exactly 0.0 when it contains no
        white pixels (a constant patch has a zero numerator), so every window
        that could reach the 0.45 threshold must contain at least one white
        pixel. Each rect is a white contour's bbox expanded by one template
        size on every side: any window holding a pixel of that contour lies
        fully inside the expanded rect (a window's top-left is at most
        `tw - 1` / `th - 1` px left/above the pixel it contains). Matching
        only inside these rects therefore gives identical scores to a
        full-mask `matchTemplate` while skipping the white-free area that
        dominates a 1080p frame. Rects are merged when they overlap so shared
        white areas are matched once; if the mask is too fragmented (> 64
        rects) a single union rect bounds the work instead.
        """
        if not white_contours:
            return []
        rects: List[Tuple[int, int, int, int]] = []
        for cnt in white_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(0, x - (tw - 1))
            y1 = max(0, y - (th - 1))
            x2 = min(mask_w, x + w + (tw - 1))
            y2 = min(mask_h, y + h + (th - 1))
            if x2 - x1 >= tw and y2 - y1 >= th:
                rects.append((x1, y1, x2 - x1, y2 - y1))
        if not rects:
            return []
        if len(rects) > 64:
            ux1 = min(r[0] for r in rects)
            uy1 = min(r[1] for r in rects)
            ux2 = max(r[0] + r[2] for r in rects)
            uy2 = max(r[1] + r[3] for r in rects)
            return [(ux1, uy1, ux2 - ux1, uy2 - uy1)]
        rects.sort(key=lambda r: (r[1], r[0]))
        merged: List[List[int]] = [list(rects[0])]
        for r in rects[1:]:
            rx1, ry1, rw, rh = r
            rx2, ry2 = rx1 + rw, ry1 + rh
            placed = False
            for m in merged:
                mx1, my1 = m[0], m[1]
                mx2, my2 = mx1 + m[2], my1 + m[3]
                if rx1 <= mx2 and rx2 >= mx1 and ry1 <= my2 and ry2 >= my1:
                    nx1, ny1 = min(mx1, rx1), min(my1, ry1)
                    nx2, ny2 = max(mx2, rx2), max(my2, ry2)
                    m[0], m[1], m[2], m[3] = nx1, ny1, nx2 - nx1, ny2 - ny1
                    placed = True
                    break
            if not placed:
                merged.append([rx1, ry1, rw, rh])
        return [tuple(m) for m in merged]

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
        # A29: no white pixels -> neither the template match (which needs
        # white crest pixels to correlate against) nor the contour fallback
        # can find a crest. Skipping the full-frame matchTemplate here
        # saves ~385 ms (median, 1920x1080) on white-free frames with
        # identical output (verified by probe: empty/black frames return []
        # with and without this guard).
        if not np.any(white_mask):
            return []
        crests: List[Dict[str, Any]] = []

        # White contours are computed once and shared: they bound the
        # template-match search (step 1) and feed the symmetry fallback (step 2).
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1. Template match with canonical 28x29 winged crest mask if ROI is large enough
        th, tw = self._crest_template_mask.shape[:2]
        if white_mask.shape[0] >= th and white_mask.shape[1] >= tw:
            # A full-mask matchTemplate costs ~0.5s on a 1080p frame, yet a
            # crest window must contain white pixels to score >= 0.45 (a
            # constant patch scores exactly 0.0 under TM_CCOEFF_NORMED).
            # Matching only inside the expanded white-contour rects
            # (_template_search_rects) yields identical scores while skipping
            # the (usually vast) white-free area. Candidates from all rects
            # are NMS'd together exactly as before (F-0922).
            candidates: List[Tuple[float, int, int]] = []
            for (sx, sy, sw, sh) in self._template_search_rects(
                contours, white_mask.shape[1], white_mask.shape[0], th, tw
            ):
                crop = white_mask[sy : sy + sh, sx : sx + sw]
                if crop.shape[0] < th or crop.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(crop, self._crest_template_mask, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= 0.45)
                candidates.extend(
                    (float(res[y_pt, x_pt]), int(x_pt) + sx, int(y_pt) + sy)
                    for y_pt, x_pt in zip(ys, xs)
                )
            candidates.sort(reverse=True)
            accepted_centers: List[Tuple[float, float]] = []
            for score, x_pt, y_pt in candidates:
                cand_cx = rx1 + x_pt + 0.5 * tw
                cand_cy = ry1 + y_pt + 0.5 * th
                if any(
                    math.hypot(cand_cx - ax, cand_cy - ay) <= 16.0
                    for ax, ay in accepted_centers
                ):
                    continue
                accepted_centers.append((cand_cx, cand_cy))
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
    def _measure_roi_visual_evidence(
        self,
        hsv_img: np.ndarray,
        crest_bbox: Optional[Tuple[int, int, int, int]],
        nameplate_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, float, int]:
        """Measure REAL visual evidence inside the given ROIs.

        Returns ``(turquoise_pixels, turquoise_ratio, white_crest_pixels)``
        measured from the frame — never a hardcoded constant (F-0787/F-0834).
        """
        tq_px = 0
        tq_ratio = 0.0
        white_px = 0
        h, w = hsv_img.shape[:2]
        if nameplate_bbox is not None:
            nx, ny, nw, nh = (int(v) for v in nameplate_bbox)
            x1, y1 = max(0, nx - 4), max(0, ny - 4)
            x2, y2 = min(w, nx + nw + 4), min(h, ny + nh + 4)
            if x2 > x1 and y2 > y1:
                sub = hsv_img[y1:y2, x1:x2]
                m1 = cv2.inRange(sub, self.TURQUOISE_PRIMARY_LOWER, self.TURQUOISE_PRIMARY_UPPER)
                m2 = cv2.inRange(sub, self.TURQUOISE_SECONDARY_LOWER, self.TURQUOISE_SECONDARY_UPPER)
                tq_px = int(np.count_nonzero(cv2.bitwise_or(m1, m2)))
                tq_ratio = round(tq_px / float(max(1, (x2 - x1) * (y2 - y1))), 4)
        if crest_bbox is not None:
            cx, cy, cw, ch = (int(v) for v in crest_bbox)
            x1, y1 = max(0, cx), max(0, cy)
            x2, y2 = min(w, cx + cw), min(h, cy + ch)
            if x2 > x1 and y2 > y1:
                sub = hsv_img[y1:y2, x1:x2]
                white_mask = cv2.inRange(sub, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER)
                white_px = int(np.count_nonzero(white_mask))
        return tq_px, tq_ratio, white_px

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
            # A3: the frame is passed so the alignment check measures
            # turquoise/white evidence from pixels (single verification
            # path) instead of trusting dict-carried numbers.
            align_res = self.check_vertical_alignment(
                crest_box, nameplate_box, img_w=img_w, img_h=img_h, require_crest_above=True,
                frame_bgr=frame_bgr, hsv=hsv,
            )
            # F-0787: a merely geometry-aligned pair (verified=False,
            # geometry_aligned=True) must NOT return early as Emin — with
            # a frame it proceeds to real ROI measurement below; without
            # a frame it is returned as-is (Şüpheli). Only a misaligned
            # pair returns early here.
            if not (align_res.verified or align_res.geometry_aligned):
                return align_res
            # F-0834: with no frame there is nothing visual to measure, so
            # only the geometry-only alignment result can be returned.
            if frame_bgr is None or frame_bgr.size == 0:
                return align_res

            # With a frame, geometry alone is NOT enough: the crest and the
            # turquoise nameplate must both be visually present in their ROIs.
            hsv_img = self._ensure_hsv(frame_bgr, hsv)
            if hsv_img is None:
                return self._build_unverified_result(
                    dx=align_res.dx,
                    dy=align_res.dy,
                    crest_bbox=align_res.crest_bbox,
                    nameplate_bbox=align_res.nameplate_bbox,
                    reason="invalid_frame_channels",
                )
            if align_res.nameplate_bbox is not None:
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
            tq_px, tq_ratio, white_px = self._measure_roi_visual_evidence(
                hsv_img, align_res.crest_bbox, align_res.nameplate_bbox
            )
            if tq_px < self.min_turquoise_pixels or white_px < 10:
                return self._build_unverified_result(
                    dx=align_res.dx,
                    dy=align_res.dy,
                    crest_bbox=align_res.crest_bbox,
                    nameplate_bbox=align_res.nameplate_bbox,
                    crest_score=align_res.crest_score,
                    turquoise_ratio=tq_ratio,
                    turquoise_pixels=tq_px,
                    reason=(
                        "insufficient_visual_evidence_in_frame "
                        f"(turquoise_pixels={tq_px}, white_crest_pixels={white_px})"
                    ),
                )
            return self._build_verified_result(
                dx=align_res.dx,
                dy=align_res.dy,
                crest_bbox=align_res.crest_bbox,
                nameplate_bbox=align_res.nameplate_bbox,
                crest_score=align_res.crest_score,
                turquoise_ratio=tq_ratio,
                turquoise_pixels=tq_px,
                # A29: the geometry-only reason from check_vertical_alignment
                # ("turquoise not measured — no frame supplied") is wrong
                # here — a frame WAS supplied and turquoise/white were
                # measured in the ROIs. Report the measured evidence.
                reason=(
                    "verified_in_frame "
                    f"(turquoise_pixels={tq_px}, white_crest_pixels={white_px})"
                ),
            )

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
                # A3/A4: the frame is passed so the alignment check measures
                # turquoise AND white-crest evidence from pixels — the crest
                # box itself must be seen, not just claimed by a box.
                res = self.check_vertical_alignment(
                    cbbox, np_det, img_w=img_w, img_h=img_h, require_crest_above=True,
                    frame_bgr=frame_bgr, hsv=hsv_img,
                )
                if res.verified:
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cbbox,
                        nameplate_bbox=np_det["bbox"],
                        crest_score=cscore,
                        turquoise_ratio=res.turquoise_ratio,
                        turquoise_pixels=res.turquoise_pixels,
                        # A29: turquoise WAS measured by
                        # check_vertical_alignment here; do not repeat
                        # the geometry-only "not measured" reason.
                        reason=(
                            "verified_crest_with_detected_nameplate "
                            f"(turquoise_pixels={res.turquoise_pixels})"
                        ),
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
            aligned_but_no_turquoise = False
            for cr_det in crests:
                # A3: the frame is passed so the alignment check measures
                # evidence from pixels; an aligned pair without measured
                # turquoise is NOT verified here — the ROI below is measured
                # before any Emin.
                res = self.check_vertical_alignment(
                    cr_det, nbbox, img_w=img_w, img_h=img_h, require_crest_above=True,
                    frame_bgr=frame_bgr, hsv=hsv_img,
                )
                if res.verified or res.geometry_aligned:
                    # F-0834: the crest was found visually, but the nameplate
                    # box itself must also contain measured turquoise — the
                    # old hardcoded 0.30 / 20px "evidence" is not evidence.
                    tq_px, tq_ratio, _white_px = self._measure_roi_visual_evidence(
                        hsv_img, cr_det["bbox"], nbbox
                    )
                    if tq_px < self.min_turquoise_pixels:
                        aligned_but_no_turquoise = True
                        continue
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cr_det["bbox"],
                        nameplate_bbox=nbbox,
                        crest_score=cr_det["score"],
                        turquoise_ratio=tq_ratio,
                        turquoise_pixels=tq_px,
                        # A29: turquoise WAS measured in the nameplate ROI
                        # here; do not repeat the geometry-only reason.
                        reason=(
                            "verified_nameplate_with_detected_crest "
                            f"(turquoise_pixels={tq_px})"
                        ),
                    )
            if aligned_but_no_turquoise:
                return self._build_unverified_result(
                    nameplate_bbox=nbbox,
                    reason="aligned_crest_but_nameplate_has_no_turquoise_evidence",
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
                # A3: the frame is passed so the alignment check measures
                # turquoise/white evidence from pixels (single path).
                res = self.check_vertical_alignment(
                    cr, np_det, img_w=img_w, img_h=img_h, require_crest_above=True,
                    frame_bgr=frame_bgr, hsv=hsv_img,
                )
                if res.verified:
                    return self._build_verified_result(
                        dx=res.dx,
                        dy=res.dy,
                        crest_bbox=cr["bbox"],
                        nameplate_bbox=np_det["bbox"],
                        crest_score=cr["score"],
                        turquoise_ratio=res.turquoise_ratio,
                        turquoise_pixels=res.turquoise_pixels,
                        # A29: both crest and nameplate were detected
                        # visually here; do not repeat the geometry-only
                        # "turquoise not measured" reason.
                        reason=(
                            "verified_crest_nameplate_in_frame "
                            f"(turquoise_pixels={res.turquoise_pixels})"
                        ),
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
        hsv: Optional[np.ndarray] = None,
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
        # S-4802: a corrupt confidence (>1, inf, nan) must never flow
        # into the calibrated value — clamp to a finite [0, 1] range
        # here so both the verified and unverified branches below stay
        # bounded.
        if not math.isfinite(raw_conf):
            raw_conf = 0.65
        else:
            raw_conf = min(1.0, max(0.0, raw_conf))

        result: Optional[CrestNameplateVerificationResult] = None

        # Check explicit partner in `updated` metadata if provided
        if "crest_bbox" in updated and "nameplate_bbox" in updated:
            result = self.verify_in_frame(
                frame_bgr,
                crest_box=updated["crest_bbox"],
                nameplate_box=updated["nameplate_bbox"],
                hsv=hsv,
            )
        elif cid == CREST_CLASS_ID:
            # Look for a matching nameplate or resource annotation below this crest
            if all_annotations:
                for other in all_annotations:
                    if other is box or not isinstance(other, dict):
                        continue
                    # F-0835: a corrupt class_id must skip only this
                    # annotation, never crash the whole batch.
                    try:
                        ocid = int(other.get("class_id", -1))
                    except (TypeError, ValueError):
                        continue
                    if ocid in WORLD_RESOURCE_CLASS_IDS or str(other.get("source", "")).endswith("nameplate"):
                        cand_res = self.check_vertical_alignment(
                            updated, other, img_w=img_w, img_h=img_h, require_crest_above=True
                        )
                        if cand_res.verified:
                            result = cand_res
                            break
            if (result is None or not result.verified) and frame_bgr is not None:
                result = self.verify_in_frame(frame_bgr, crest_box=updated, hsv=hsv)
        else:
            # Resource or nameplate box -> look for a matching `world_gather_crest` (class 13) above it
            if all_annotations:
                for other in all_annotations:
                    if other is box or not isinstance(other, dict):
                        continue
                    # F-0835: see above — corrupt class_id skips this entry.
                    try:
                        other_cid = int(other.get("class_id", -1))
                    except (TypeError, ValueError):
                        continue
                    if other_cid == CREST_CLASS_ID:
                        cand_res = self.check_vertical_alignment(
                            other, updated, img_w=img_w, img_h=img_h, require_crest_above=True
                        )
                        if cand_res.verified:
                            result = cand_res
                            break
            if (result is None or not result.verified) and frame_bgr is not None:
                result = self.verify_in_frame(frame_bgr, candidate_box=updated, hsv=hsv)

        if result is None:
            result = self._build_unverified_result(
                base_confidence=raw_conf,
                reason="no_crest_nameplate_alignment_evidence",
            )

        if result.verified:
            # S-4802 / A29: cap at the same 0.99 ceiling _build_verified_result
            # applies, so a raw confidence can never push the calibrated
            # value above 1 (defense in depth alongside the parse clamp).
            # The module's own verified builder never exceeds 0.99, but
            # max(..., raw_conf) used to pass a caller's raw_conf of 1.0
            # straight through (probe: confidence 1.0).
            updated["confidence"] = round(min(0.99, max(self.verified_min_conf, result.confidence, raw_conf)), 4)
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
        hsv: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-verify a sequence of annotations for crest-to-nameplate vertical alignment.

        The BGR->HSV conversion of a 1080p frame costs tens of ms, and every
        per-annotation `verify_in_frame` call used to pay it again. The HSV
        image is therefore converted once here and shared by all annotations
        (a caller-supplied `hsv` is used as-is). Conversion is deterministic,
        so shared HSV yields identical results to per-call conversion.
        """
        ann_list = [dict(a) for a in (annotations or []) if isinstance(a, dict)]
        hsv_img = self._ensure_hsv(frame_bgr, hsv) if frame_bgr is not None else None
        return [
            self.verify_annotation(
                ann,
                frame_bgr=frame_bgr,
                all_annotations=ann_list,
                img_w=img_w,
                img_h=img_h,
                hsv=hsv_img,
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
