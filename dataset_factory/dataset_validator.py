# -*- coding: utf-8 -*-
"""Dataset Quality Assurance (QA), Validation, Human Confirmation Gate,
and Stale Model Tracking for Gatherion-Local YOLO Training Pipeline.

Provides rigorous checks for:
1. Overlapping / duplicate boxes (IoU > 0.8) on same class
2. Zero-area boxes (w <= 0 or h <= 0)
3. Out-of-bounds normalized coordinates (< 0.0 or > 1.0)
4. Human confirmation gate: enforces pending == 0 before training
5. Stale model marking and tracking
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger("gatherion.dataset_validator")

# Default thresholds
DEFAULT_IOU_OVERLAP_THRESHOLD: float = 0.80
DEFAULT_MIN_BOX_DIM: float = 1e-5
DEFAULT_MIN_BOX_AREA: float = 1e-6

# L17 Box Size Bounds & Guardrails (Micro noise & massive screen-filling bounds)
MIN_BOX_DIM_BOUND: float = 0.004   # Micro noise threshold: w < 0.004 or h < 0.004
MAX_BOX_DIM_BOUND: float = 0.65    # Massive box threshold: w > 0.65 or h > 0.65
MIN_BOX_AREA_BOUND: float = 1.6e-5 # Minimum area: 0.004 * 0.004

STALE_MANIFEST_NAME = "stale_manifest.json"
PIPELINE_STATE_NAME = "pipeline_ui_state.json"


@dataclass
class ValidationIssue:
    issue_type: str  # 'duplicate_overlap', 'zero_area', 'out_of_bounds', 'untrusted_label', 'syntax_error'
    severity: str    # 'error', 'warning', 'info'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    box_index: Optional[int] = None
    box: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    valid: bool = True
    total_files_scanned: int = 0
    total_boxes_scanned: int = 0
    error_count: int = 0
    warning_count: int = 0
    duplicate_overlap_count: int = 0
    zero_area_count: int = 0
    size_bounds_count: int = 0
    out_of_bounds_count: int = 0
    untrusted_count: int = 0
    pending_review_count: int = 0
    sync_mismatch_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.error_count += 1
            self.valid = False
        elif issue.severity == "warning":
            self.warning_count += 1

        if issue.issue_type == "duplicate_overlap":
            self.duplicate_overlap_count += 1
        elif issue.issue_type == "zero_area":
            self.zero_area_count += 1
        elif issue.issue_type in ("size_bounds", "micro_noise", "massive_box"):
            self.size_bounds_count += 1
        elif issue.issue_type == "out_of_bounds":
            self.out_of_bounds_count += 1
        elif issue.issue_type == "untrusted_label":
            self.untrusted_count += 1
        elif issue.issue_type == "pending_review":
            self.pending_review_count += 1
        elif issue.issue_type == "sync_mismatch":
            self.sync_mismatch_count += 1

    def finalize(self) -> None:
        if self.error_count == 0:
            self.valid = True
            self.summary = (
                f"QA Geçerli: {self.total_files_scanned} dosya, "
                f"{self.total_boxes_scanned} kutu tarandı. Hata yok ({self.warning_count} uyarı)."
            )
        else:
            self.valid = False
            reasons = []
            if self.duplicate_overlap_count > 0:
                reasons.append(f"{self.duplicate_overlap_count} örtüşen kutu (IoU>0.8)")
            if self.zero_area_count > 0:
                reasons.append(f"{self.zero_area_count} sıfır alanlı kutu")
            if self.size_bounds_count > 0:
                reasons.append(f"{self.size_bounds_count} geçersiz boyutlu kutu (w/h < 0.004 veya > 0.65)")
            if self.out_of_bounds_count > 0:
                reasons.append(f"{self.out_of_bounds_count} sınır dışı kutu")
            if self.untrusted_count > 0:
                reasons.append(f"{self.untrusted_count} onaysız etiket")
            if self.pending_review_count > 0:
                reasons.append(f"{self.pending_review_count} onay bekleyen örnek")
            if self.sync_mismatch_count > 0:
                reasons.append(f"{self.sync_mismatch_count} JSON/TXT senkron uyuşmazlığı")
            detail = ", ".join(reasons) if reasons else f"{self.error_count} hata"
            self.summary = f"QA Başarısız: {detail} tespit edildi."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "total_files_scanned": self.total_files_scanned,
            "total_boxes_scanned": self.total_boxes_scanned,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "duplicate_overlap_count": self.duplicate_overlap_count,
            "zero_area_count": self.zero_area_count,
            "size_bounds_count": self.size_bounds_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "untrusted_count": self.untrusted_count,
            "pending_review_count": self.pending_review_count,
            "sync_mismatch_count": self.sync_mismatch_count,
            "summary": self.summary,
            "issues": [it.to_dict() for it in self.issues],
        }


# ==============================================================================
# 1. Geometry & Coordinate Calculations
# ==============================================================================

def box_xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert YOLO center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)."""
    hw = float(w) * 0.5
    hh = float(h) * 0.5
    return float(cx) - hw, float(cy) - hh, float(cx) + hw, float(cy) + hh


def box_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert corner format (x1, y1, x2, y2) to YOLO center format (cx, cy, w, h)."""
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    cx = float(x1) + w * 0.5
    cy = float(y1) + h * 0.5
    return cx, cy, w, h


def compute_box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
    is_xywh: bool = True,
) -> float:
    """Compute Intersection over Union (IoU) between two boxes.
    
    If is_xywh=True, inputs are (cx, cy, w, h); otherwise (x1, y1, x2, y2).
    """
    if is_xywh:
        ax1, ay1, ax2, ay2 = box_xywh_to_xyxy(*box1)
        bx1, by1, bx2, by2 = box_xywh_to_xyxy(*box2)
    else:
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih
    if inter_area <= 1e-12:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 1e-12:
        return 0.0

    return inter_area / union_area


# ==============================================================================
# 2. QA Check Functions (Zero-Area, Out-of-Bounds, Duplicate/Overlap)
# ==============================================================================

def check_zero_area(
    w: float,
    h: float,
    min_dim: float = DEFAULT_MIN_BOX_DIM,
    min_area: float = DEFAULT_MIN_BOX_AREA,
) -> Tuple[bool, str]:
    """Check if box has zero or sub-minimal area.
    
    Returns (is_invalid, error_message).
    """
    if not math.isfinite(w) or not math.isfinite(h):
        return True, f"Kutu boyutları geçersiz sayı (w={w}, h={h})"
    if w <= 0.0 or h <= 0.0:
        return True, f"Sıfır veya negatif boyut (w={w:.6f}, h={h:.6f})"
    if w < min_dim or h < min_dim:
        return True, f"Boyut asgari sınırın altında (w={w:.6f} < {min_dim} veya h={h:.6f} < {min_dim})"
    area = w * h
    if area < min_area:
        return True, f"Alan asgari sınırın altında (alan={area:.8f} < {min_area})"
    return False, ""


def check_out_of_bounds(
    cx: float,
    cy: float,
    w: float,
    h: float,
    tolerance: float = 1e-4,
) -> Tuple[bool, List[str]]:
    """Check if normalized coordinates are shifted or out of [0.0, 1.0] bounds.
    
    Returns (is_invalid, list_of_error_descriptions).
    """
    errors: List[str] = []
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
        return True, ["Koordinatlar geçerli sonlu sayı değil (NaN/Inf)"]

    # Check center bounds
    if cx < -tolerance or cx > (1.0 + tolerance):
        errors.append(f"Merkez X sınır dışı: cx={cx:.6f} (beklenen: 0.0 - 1.0)")
    if cy < -tolerance or cy > (1.0 + tolerance):
        errors.append(f"Merkez Y sınır dışı: cy={cy:.6f} (beklenen: 0.0 - 1.0)")

    # Check corners (extents)
    x1, y1, x2, y2 = box_xywh_to_xyxy(cx, cy, w, h)
    if x1 < -tolerance:
        errors.append(f"Sol kenar sınır dışı: x1={x1:.6f} < 0.0")
    if y1 < -tolerance:
        errors.append(f"Üst kenar sınır dışı: y1={y1:.6f} < 0.0")
    if x2 > (1.0 + tolerance):
        errors.append(f"Sağ kenar sınır dışı: x2={x2:.6f} > 1.0")
    if y2 > (1.0 + tolerance):
        errors.append(f"Alt kenar sınır dışı: y2={y2:.6f} > 1.0")

    # Check unnormalized pixel coordinates (e.g. w > 1.0 or h > 1.0)
    if w > (1.0 + tolerance) or h > (1.0 + tolerance):
        errors.append(f"Normalize edilmemiş piksel boyutu şüphesi: w={w:.2f}, h={h:.2f}")

    return len(errors) > 0, errors


class BoxValidationResult(NamedTuple):
    """Result returned by validate_label_box.
    
    Acts as a 2-tuple (is_valid, message) for tuple unpacking:
        valid, reason = validate_label_box(...)
    And behaves truthy/falsy in boolean contexts:
        if not validate_label_box(...):
    """
    is_valid: bool
    message: str

    def __bool__(self) -> bool:
        return bool(self.is_valid)

    @property
    def reason(self) -> str:
        return self.message


def check_box_size_bounds(
    w: float,
    h: float,
    min_dim: float = MIN_BOX_DIM_BOUND,
    max_dim: float = MAX_BOX_DIM_BOUND,
) -> Tuple[bool, str]:
    """Check if normalized box dimensions violate micro noise or massive box bounds.
    
    L17 Guardrail:
    - Micro noise boxes: w < 0.004 or h < 0.004
    - Massive screen-filling erroneous boxes: w > 0.65 or h > 0.65
    
    Returns (is_invalid, error_message).
    """
    if not (math.isfinite(w) and math.isfinite(h)):
        return True, f"Kutu boyutları geçersiz sayı (w={w}, h={h})"
    if w <= 0.0 or h <= 0.0:
        return True, f"Sıfır veya negatif boyut (w={w:.6f}, h={h:.6f})"
    if w < min_dim or h < min_dim:
        return True, f"Mikro gürültü kutusu: w={w:.6f} < {min_dim} veya h={h:.6f} < {min_dim}"
    if w > max_dim or h > max_dim:
        return True, f"Aşırı büyük ekranı kaplayan kutu: w={w:.6f} > {max_dim} veya h={h:.6f} > {max_dim}"
    return False, ""


def validate_label_box(
    box_or_cx: Union[Dict[str, Any], Sequence[float], float],
    cy: Optional[float] = None,
    w: Optional[float] = None,
    h: Optional[float] = None,
    class_id: Optional[int] = None,
    *,
    min_dim: float = MIN_BOX_DIM_BOUND,
    max_dim: float = MAX_BOX_DIM_BOUND,
    strict_bounds: bool = True,
    tolerance: float = 1e-4,
) -> BoxValidationResult:
    """Validate a bounding box against L17 size bounds and coordinate rules.
    
    Rejects:
    1. Micro noise boxes: w < 0.004 or h < 0.004
    2. Massive screen-filling erroneous boxes: w > 0.65 or h > 0.65
    3. Zero-area or negative dimension boxes (w <= 0.0 or h <= 0.0)
    4. Out-of-bounds coordinates (cx < 0.0, cx > 1.0, cy < 0.0, cy > 1.0)
    5. Coordinates extending outside [0.0, 1.0] when strict_bounds=True (x1 < 0, x2 > 1, y1 < 0, y2 > 1)
    
    Accepts:
    - Dict with 'cx', 'cy', 'w', 'h' (or 'x_center', 'y_center', 'width', 'height') and optional 'class_id'
    - 4-tuple or 5-tuple (cx, cy, w, h) or (class_id, cx, cy, w, h)
    - Discrete arguments: validate_label_box(cx, cy, w, h, class_id)
    
    Returns:
        BoxValidationResult(is_valid: bool, message: str)
    """
    if isinstance(box_or_cx, dict):
        raw = box_or_cx
        try:
            val_cx = float(raw.get("cx", raw.get("x_center", 0.0)))
            val_cy = float(raw.get("cy", raw.get("y_center", 0.0)))
            val_w = float(raw.get("w", raw.get("width", 0.0)))
            val_h = float(raw.get("h", raw.get("height", 0.0)))
            if class_id is None:
                raw_cid = raw.get("class_id", raw.get("cls"))
                if raw_cid is not None:
                    try:
                        class_id = int(raw_cid)
                    except (ValueError, TypeError):
                        pass
        except (ValueError, TypeError) as err:
            return BoxValidationResult(False, f"Invalid box dictionary numeric values: {err}")
    elif isinstance(box_or_cx, (list, tuple)):
        seq = box_or_cx
        if len(seq) == 4:
            try:
                val_cx, val_cy, val_w, val_h = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
            except (ValueError, TypeError) as err:
                return BoxValidationResult(False, f"Invalid sequence numbers: {err}")
        elif len(seq) >= 5:
            if class_id is None:
                try:
                    class_id = int(seq[0])
                    val_cx, val_cy, val_w, val_h = float(seq[1]), float(seq[2]), float(seq[3]), float(seq[4])
                except (ValueError, TypeError):
                    try:
                        val_cx, val_cy, val_w, val_h = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
                    except (ValueError, TypeError) as err:
                        return BoxValidationResult(False, f"Invalid sequence numbers: {err}")
            else:
                try:
                    val_cx, val_cy, val_w, val_h = float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])
                except (ValueError, TypeError) as err:
                    return BoxValidationResult(False, f"Invalid sequence numbers: {err}")
        else:
            return BoxValidationResult(False, f"Sequence must have at least 4 elements, got {len(seq)}")
    else:
        if cy is None or w is None or h is None:
            return BoxValidationResult(False, "Missing required coordinate arguments (cy, w, h)")
        try:
            val_cx = float(box_or_cx)
            val_cy = float(cy)
            val_w = float(w)
            val_h = float(h)
        except (ValueError, TypeError) as err:
            return BoxValidationResult(False, f"Non-numeric coordinates provided: {err}")

    # 1. Finite numbers check
    if not (math.isfinite(val_cx) and math.isfinite(val_cy) and math.isfinite(val_w) and math.isfinite(val_h)):
        return BoxValidationResult(False, f"Non-finite coordinates: cx={val_cx}, cy={val_cy}, w={val_w}, h={val_h}")

    # 2. Zero-area / negative dimension check
    if val_w <= 0.0 or val_h <= 0.0:
        return BoxValidationResult(False, f"Zero or negative dimension: w={val_w:.6f}, h={val_h:.6f}")
    if (val_w * val_h) <= 0.0:
        return BoxValidationResult(False, f"Zero box area: area={(val_w * val_h):.8f}")

    # 3. Micro noise check (w < min_dim or h < min_dim, default 0.004)
    if val_w < min_dim or val_h < min_dim:
        return BoxValidationResult(
            False,
            f"Micro noise box rejected: w={val_w:.6f} < {min_dim} or h={val_h:.6f} < {min_dim}"
        )

    # 4. Massive screen-filling erroneous box check (w > max_dim or h > max_dim, default 0.65)
    if val_w > max_dim or val_h > max_dim:
        return BoxValidationResult(
            False,
            f"Massive screen-filling box rejected: w={val_w:.6f} > {max_dim} or h={val_h:.6f} > {max_dim}"
        )

    # 5. Out-of-bounds center coordinates check (0.0 to 1.0)
    if val_cx < -tolerance or val_cx > (1.0 + tolerance):
        return BoxValidationResult(False, f"Center X out of bounds: cx={val_cx:.6f} not in [0.0, 1.0]")
    if val_cy < -tolerance or val_cy > (1.0 + tolerance):
        return BoxValidationResult(False, f"Center Y out of bounds: cy={val_cy:.6f} not in [0.0, 1.0]")

    # 6. Box boundary check
    x1, y1, x2, y2 = box_xywh_to_xyxy(val_cx, val_cy, val_w, val_h)
    if strict_bounds:
        if x1 < -tolerance or y1 < -tolerance or x2 > (1.0 + tolerance) or y2 > (1.0 + tolerance):
            return BoxValidationResult(
                False,
                f"Box boundaries extend out of [0.0, 1.0]: x1={x1:.6f}, y1={y1:.6f}, x2={x2:.6f}, y2={y2:.6f}"
            )

    # 7. Optional class_id check
    if class_id is not None and class_id < 0:
        return BoxValidationResult(False, f"Invalid negative class_id: {class_id}")

    return BoxValidationResult(True, "")


def fix_out_of_bounds(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    min_dim: float = DEFAULT_MIN_BOX_DIM,
) -> Optional[Tuple[float, float, float, float]]:
    """Attempt to sanitize or clip shifted/out-of-bounds coordinates to [0.0, 1.0].
    
    If unnormalized pixel values are detected and image dimensions are provided,
    normalizes them. Then clips x1, y1, x2, y2 into [0.0, 1.0].
    Returns (new_cx, new_cy, new_w, new_h) or None if box cannot be salvaged.
    """
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
        return None

    # Handle unnormalized pixel coordinates if image dimensions provided
    if image_width and image_height and (cx > 1.0 or cy > 1.0 or w > 1.0 or h > 1.0):
        if image_width > 0 and image_height > 0:
            cx = cx / float(image_width)
            cy = cy / float(image_height)
            w = w / float(image_width)
            h = h / float(image_height)

    x1, y1, x2, y2 = box_xywh_to_xyxy(cx, cy, w, h)

    # If completely outside [0.0, 1.0], cannot be clipped into a valid box
    if x2 <= 0.0 or x1 >= 1.0 or y2 <= 0.0 or y1 >= 1.0:
        return None

    # Clip corners to [0.0, 1.0]
    clip_x1 = max(0.0, min(1.0, x1))
    clip_y1 = max(0.0, min(1.0, y1))
    clip_x2 = max(0.0, min(1.0, x2))
    clip_y2 = max(0.0, min(1.0, y2))

    new_w = clip_x2 - clip_x1
    new_h = clip_y2 - clip_y1
    if new_w < min_dim or new_h < min_dim:
        return None

    new_cx = clip_x1 + new_w * 0.5
    new_cy = clip_y1 + new_h * 0.5
    return round(new_cx, 6), round(new_cy, 6), round(new_w, 6), round(new_h, 6)


def detect_overlapping_boxes(
    boxes: Sequence[Dict[str, Any]],
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    same_class_only: bool = True,
) -> List[Tuple[int, int, float, str]]:
    """Detect pairs of overlapping boxes with IoU > iou_threshold.
    
    Returns list of tuples: (index_i, index_j, iou_score, details_str).
    """
    overlaps: List[Tuple[int, int, float, str]] = []
    n = len(boxes)
    for i in range(n):
        box_i = boxes[i]
        cid_i = int(box_i.get("class_id", -1))
        coords_i = (
            float(box_i.get("cx", 0.0)),
            float(box_i.get("cy", 0.0)),
            float(box_i.get("w", 0.0)),
            float(box_i.get("h", 0.0)),
        )
        for j in range(i + 1, n):
            box_j = boxes[j]
            cid_j = int(box_j.get("class_id", -1))
            if same_class_only and cid_i != cid_j:
                continue

            coords_j = (
                float(box_j.get("cx", 0.0)),
                float(box_j.get("cy", 0.0)),
                float(box_j.get("w", 0.0)),
                float(box_j.get("h", 0.0)),
            )
            iou = compute_box_iou(coords_i, coords_j, is_xywh=True)
            if iou > iou_threshold:
                desc = (
                    f"Kutu #{i} ve #{j} aynı sınıfta ({cid_i}) yüksek örtüşme "
                    f"(IoU={iou:.3f} > {iou_threshold})"
                )
                overlaps.append((i, j, iou, desc))
    return overlaps


# ==============================================================================
# 3. Box Sanitizer & Filter (Drop zero-area, size bounds, fix bounds, deduplicate IoU>0.8)
# ==============================================================================

def sanitize_boxes(
    boxes: Sequence[Dict[str, Any]],
    *,
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    min_dim: float = MIN_BOX_DIM_BOUND,
    max_dim: float = MAX_BOX_DIM_BOUND,
    auto_fix_coords: bool = True,
    drop_zero_area: bool = True,
    drop_invalid_size: bool = True,
    file_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    """Sanitize bounding boxes by:
    1. Rejecting zero-area boxes (w <= 0 or h <= 0)
    2. Rejecting micro noise boxes (w < 0.004 or h < 0.004) and massive boxes (w > 0.65 or h > 0.65)
    3. Rejecting or fixing shifted/out-of-bounds coordinates
    4. Resolving duplicate/overlapping boxes (IoU > 0.8) on same class.
    
    Returns (cleaned_boxes, issues_encountered).
    """
    issues: List[ValidationIssue] = []
    valid_candidates: List[Dict[str, Any]] = []

    for idx, raw in enumerate(boxes or []):
        if not isinstance(raw, dict):
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Kutu #{idx} sözlük biçiminde değil",
                    file_path=file_path,
                    box_index=idx,
                )
            )
            continue

        try:
            cid = int(raw.get("class_id", raw.get("cls", -1)))
            cx = float(raw.get("cx", raw.get("x_center", 0.0)))
            cy = float(raw.get("cy", raw.get("y_center", 0.0)))
            w = float(raw.get("w", raw.get("width", 0.0)))
            h = float(raw.get("h", raw.get("height", 0.0)))
        except (TypeError, ValueError) as err:
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Kutu #{idx} koordinat parse hatası: {err}",
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                )
            )
            continue

        # 1. Zero-area check
        is_zero, zero_msg = check_zero_area(w, h)
        if is_zero:
            issues.append(
                ValidationIssue(
                    issue_type="zero_area",
                    severity="error",
                    message=f"Kutu #{idx} reddedildi: {zero_msg}",
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                    details={"w": w, "h": h},
                )
            )
            if drop_zero_area:
                continue

        # 2. L17 Box size bounds check (micro noise & massive screen-filling box)
        is_bad_size, size_msg = check_box_size_bounds(w, h, min_dim=min_dim, max_dim=max_dim)
        if is_bad_size and not is_zero:
            itype = "micro_noise" if (w < min_dim or h < min_dim) else "massive_box"
            issues.append(
                ValidationIssue(
                    issue_type=itype,
                    severity="error",
                    message=f"Kutu #{idx} reddedildi: {size_msg}",
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                    details={"w": w, "h": h, "min_dim": min_dim, "max_dim": max_dim},
                )
            )
            if drop_invalid_size:
                continue

        # 3. Out-of-bounds check
        is_oob, oob_msgs = check_out_of_bounds(cx, cy, w, h)
        if is_oob:
            msg = f"Kutu #{idx} sınır dışı koordinat: " + "; ".join(oob_msgs)
            if auto_fix_coords:
                fixed = fix_out_of_bounds(cx, cy, w, h, min_dim=min_dim)
                if fixed is not None:
                    fcx, fcy, fw, fh = fixed
                    if fw < min_dim or fh < min_dim or fw > max_dim or fh > max_dim:
                        issues.append(
                            ValidationIssue(
                                issue_type="out_of_bounds",
                                severity="error",
                                message=f"{msg} -> Kırpıldıktan sonra boyut sınırlarını ihlal etti (fw={fw:.6f}, fh={fh:.6f}), reddedildi.",
                                file_path=file_path,
                                box_index=idx,
                                box=raw,
                            )
                        )
                        continue
                    issues.append(
                        ValidationIssue(
                            issue_type="out_of_bounds",
                            severity="warning",
                            message=f"{msg} -> Düzeltildi ([0,1] kırpıldı: {fcx}, {fcy}, {fw}, {fh})",
                            file_path=file_path,
                            box_index=idx,
                            box=raw,
                            details={"original": (cx, cy, w, h), "fixed": fixed},
                        )
                    )
                    cx, cy, w, h = fcx, fcy, fw, fh
                else:
                    issues.append(
                        ValidationIssue(
                            issue_type="out_of_bounds",
                            severity="error",
                            message=f"{msg} -> Düzeltilemedi (tamamen sınır dışı), reddedildi.",
                            file_path=file_path,
                            box_index=idx,
                            box=raw,
                        )
                    )
                    continue
            else:
                issues.append(
                    ValidationIssue(
                        issue_type="out_of_bounds",
                        severity="error",
                        message=msg,
                        file_path=file_path,
                        box_index=idx,
                        box=raw,
                    )
                )
                continue

        # 4. Out-of-bounds center coordinate hard check
        if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
            issues.append(
                ValidationIssue(
                    issue_type="out_of_bounds",
                    severity="error",
                    message=f"Kutu #{idx} merkez koordinatı sınır dışı: cx={cx:.6f}, cy={cy:.6f}",
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                )
            )
            continue

        clean_box = dict(raw)
        clean_box.update({
            "class_id": cid,
            "cx": round(cx, 6),
            "cy": round(cy, 6),
            "w": round(w, 6),
            "h": round(h, 6),
        })
        valid_candidates.append(clean_box)

    # 3. Deduplicate overlapping boxes (IoU > 0.8) on same class
    # Group by class_id and keep highest confidence or hand-label
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for b in valid_candidates:
        by_class.setdefault(int(b["class_id"]), []).append(b)

    deduped_boxes: List[Dict[str, Any]] = []
    for cid, cand_list in by_class.items():
        # Sort so trusted / hand_label comes first, then confidence, then area
        def _sort_key(item: Dict[str, Any]) -> Tuple[int, float, float]:
            is_hand = 1 if (item.get("hand_label") is True or item.get("human_approved") is True) else 0
            is_trust = 1 if item.get("trusted") is True else 0
            conf = float(item.get("confidence", item.get("conf", 0.0)))
            area = float(item.get("w", 0.0)) * float(item.get("h", 0.0))
            return (is_hand * 2 + is_trust, conf, area)

        sorted_cands = sorted(cand_list, key=_sort_key, reverse=True)
        kept_class_boxes: List[Dict[str, Any]] = []

        for cand in sorted_cands:
            cand_coords = (cand["cx"], cand["cy"], cand["w"], cand["h"])
            is_duplicate = False
            for kept in kept_class_boxes:
                kept_coords = (kept["cx"], kept["cy"], kept["w"], kept["h"])
                iou = compute_box_iou(cand_coords, kept_coords, is_xywh=True)
                if iou > iou_threshold:
                    is_duplicate = True
                    issues.append(
                        ValidationIssue(
                            issue_type="duplicate_overlap",
                            severity="warning",
                            message=(
                                f"Sınıf {cid} için örtüşen/kopya kutu elendi "
                                f"(IoU={iou:.3f} > {iou_threshold}). "
                                f"Elenen: {cand_coords}, Korunan: {kept_coords}"
                            ),
                            file_path=file_path,
                            box=cand,
                            details={"iou": round(iou, 4), "kept_box": kept},
                        )
                    )
                    break
            if not is_duplicate:
                kept_class_boxes.append(cand)

        deduped_boxes.extend(kept_class_boxes)

    return deduped_boxes, issues


def sanitize_dataset_boxes(
    boxes: Sequence[Dict[str, Any]],
    *,
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    min_dim: float = MIN_BOX_DIM_BOUND,
    max_dim: float = MAX_BOX_DIM_BOUND,
    auto_fix_coords: bool = True,
    drop_zero_area: bool = True,
    drop_invalid_size: bool = True,
    file_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    """Sanitize dataset bounding boxes enforcing L17 size bounds and guardrails.
    
    Rejects micro noise (w < 0.004 or h < 0.004), massive screen-filling boxes
    (w > 0.65 or h > 0.65), zero-area boxes, out-of-bounds coords, and resolves
    duplicate/overlapping boxes (IoU > 0.8).
    
    Returns (cleaned_boxes, issues_encountered).
    """
    return sanitize_boxes(
        boxes,
        iou_threshold=iou_threshold,
        min_dim=min_dim,
        max_dim=max_dim,
        auto_fix_coords=auto_fix_coords,
        drop_zero_area=drop_zero_area,
        drop_invalid_size=drop_invalid_size,
        file_path=file_path,
    )


# ==============================================================================
# 4. File-Level & Dataset-Level Validation
# ==============================================================================

def validate_yolo_file(
    file_path: Union[str, Path],
    *,
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    strict_bounds: bool = True,
    min_dim: float = MIN_BOX_DIM_BOUND,
    max_dim: float = MAX_BOX_DIM_BOUND,
) -> Tuple[List[Dict[str, Any]], List[ValidationIssue]]:
    """Validate a YOLO format .txt file.
    
    Checks each line for valid 5-tuple, zero-area, L17 box size bounds,
    out-of-bounds coords, and intra-file duplicate overlaps.
    """
    path = Path(file_path)
    issues: List[ValidationIssue] = []
    boxes: List[Dict[str, Any]] = []

    if not path.is_file():
        issues.append(
            ValidationIssue(
                issue_type="file_not_found",
                severity="error",
                message=f"Etiket dosyası bulunamadı: {path}",
                file_path=str(path),
            )
        )
        return [], issues

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        issues.append(
            ValidationIssue(
                issue_type="io_error",
                severity="error",
                message=f"Dosya okuma hatası: {exc}",
                file_path=str(path),
            )
        )
        return [], issues

    lines = content.splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Satır {line_no}: 5 eleman bekleniyordu, {len(parts)} bulundu ({line})",
                    file_path=str(path),
                    line_number=line_no,
                )
            )
            continue

        try:
            cid = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as err:
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Satır {line_no}: Sayısal değer parse hatası ({err})",
                    file_path=str(path),
                    line_number=line_no,
                )
            )
            continue

        # Zero-area check
        is_zero, zero_msg = check_zero_area(w, h)
        if is_zero:
            issues.append(
                ValidationIssue(
                    issue_type="zero_area",
                    severity="error",
                    message=f"Satır {line_no}: {zero_msg}",
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid, "w": w, "h": h},
                )
            )

        # L17 Box size bounds check (micro noise or massive box)
        is_bad_size, size_msg = check_box_size_bounds(w, h, min_dim=min_dim, max_dim=max_dim)
        if is_bad_size and not is_zero:
            itype = "micro_noise" if (w < min_dim or h < min_dim) else "massive_box"
            issues.append(
                ValidationIssue(
                    issue_type=itype,
                    severity="error",
                    message=f"Satır {line_no}: {size_msg}",
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid, "w": w, "h": h, "min_dim": min_dim, "max_dim": max_dim},
                )
            )

        # Out-of-bounds check
        is_oob, oob_msgs = check_out_of_bounds(cx, cy, w, h)
        if is_oob:
            issues.append(
                ValidationIssue(
                    issue_type="out_of_bounds",
                    severity="error" if strict_bounds else "warning",
                    message=f"Satır {line_no}: " + "; ".join(oob_msgs),
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h},
                )
            )

        boxes.append({
            "class_id": cid,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "line_no": line_no,
        })

    # Check duplicate / overlapping boxes in file
    overlaps = detect_overlapping_boxes(boxes, iou_threshold=iou_threshold)
    for idx_i, idx_j, iou, desc in overlaps:
        issues.append(
            ValidationIssue(
                issue_type="duplicate_overlap",
                severity="error",
                message=f"{desc} (satır {boxes[idx_i].get('line_no')} vs {boxes[idx_j].get('line_no')})",
                file_path=str(path),
                details={"iou": round(iou, 4), "box_i": boxes[idx_i], "box_j": boxes[idx_j]},
            )
        )

    return boxes, issues


def count_pending_review_items(dataset_dir: Union[str, Path]) -> int:
    """Count how many review items in dataset/review/ are currently pending.
    
    A review JSON is pending if:
    - It has weak boxes and is not explicitly marked done ("ok" / "no" / human_approved)
    - Or payload has `done: None` with weak boxes
    - Or payload has `requeued: True` and trusted is empty.
    """
    root = Path(dataset_dir)
    review_dir = root / "review"
    if not review_dir.is_dir():
        return 0

    pending_count = 0
    for split in ("train", "val"):
        s_dir = review_dir / split
        if not s_dir.is_dir():
            continue
        try:
            json_files = list(s_dir.glob("*.json"))
        except OSError:
            continue
        for p in json_files:
            if p.name == "trusted_seed.json":
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            done_val = data.get("done")
            if done_val in ("ok", "no", True):
                continue
            weak = data.get("weak") or []
            trusted = data.get("trusted") or []
            requeued = bool(data.get("requeued"))
            if weak:
                pending_count += 1
            elif requeued and not trusted:
                pending_count += 1

    return pending_count


def validate_dataset(
    dataset_dir: Union[str, Path],
    *,
    split: Optional[str] = "train",
    check_pending: bool = True,
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    max_issues_to_record: int = 100,
) -> ValidationReport:
    """Comprehensive Dataset QA audit.
    
    Scans dataset labels and review queues, verifying:
    1. Pending review count (human confirmation gate)
    2. Zero-area boxes
    3. Shifted/out-of-bounds normalized coordinates
    4. Duplicate/overlapping boxes (IoU > 0.8) on same class.
    """
    root = Path(dataset_dir)
    report = ValidationReport()

    # 1. Human Confirmation Gate Check
    if check_pending:
        pending = count_pending_review_items(root)
        if pending > 0:
            report.add_issue(
                ValidationIssue(
                    issue_type="pending_review",
                    severity="error",
                    message=(
                        f"İnsan Onay Kapısı: {pending} adet incelenmemiş/onay bekleyen "
                        "örnek var. Eğitim öncesinde insan onayı zorunludur."
                    ),
                    details={"pending_count": pending},
                )
            )

    splits_to_check = [split] if split in ("train", "val") else ["train", "val"]
    labels_root = root / "labels"

    for s in splits_to_check:
        split_dir = labels_root / s
        if not split_dir.is_dir():
            continue
        try:
            files = sorted(split_dir.glob("*.txt"))
        except OSError:
            continue

        for fpath in files:
            report.total_files_scanned += 1
            boxes, file_issues = validate_yolo_file(fpath, iou_threshold=iou_threshold)
            report.total_boxes_scanned += len(boxes)

            for issue in file_issues:
                if len(report.issues) < max_issues_to_record:
                    report.add_issue(issue)
                else:
                    # Still bump counts
                    if issue.severity == "error":
                        report.error_count += 1
                        report.valid = False
                    elif issue.severity == "warning":
                        report.warning_count += 1

                    if issue.issue_type == "duplicate_overlap":
                        report.duplicate_overlap_count += 1
                    elif issue.issue_type == "zero_area":
                        report.zero_area_count += 1
                    elif issue.issue_type == "out_of_bounds":
                        report.out_of_bounds_count += 1

    report.finalize()
    return report


def is_dataset_ready_for_training(dataset_dir: Union[str, Path]) -> Tuple[bool, str]:
    """Contract: dataset_factory/label_trust.py ↔ ui/yolo_factory_dialog.py.
    
    Enforces human review completion (pending == 0) and runs QA validation.
    Returns (ready: bool, message: str).
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        return False, f"Veri seti dizini bulunamadı: {root}"

    # 1. Enforce human confirmation gate
    pending = count_pending_review_items(root)
    if pending > 0:
        return (
            False,
            f"Eğitim engellendi: {pending} adet onay bekleyen (pending) etiket var. "
            "Adım 2 (İnsan Onay Kapısı) tamamlanmadan eğitim başlatılamaz.",
        )

    # 2. Check if train labels exist
    train_labels = root / "labels" / "train"
    if not train_labels.is_dir():
        return False, "Eğitim engellendi: 'dataset/labels/train' dizini bulunamadı."

    txt_files = list(train_labels.glob("*.txt"))
    if not txt_files:
        return False, "Eğitim engellendi: Eğitim veri setinde hiç etiket (.txt) bulunamadı."

    non_empty = sum(1 for p in txt_files if p.read_text(encoding="utf-8", errors="ignore").strip())
    if non_empty == 0:
        return False, "Eğitim engellendi: Eğitim etiketlerinin tümü boş. Onaylı etiket bulunmuyor."

    # 3. QA Validation
    report = validate_dataset(root, split="train", check_pending=False)
    if not report.valid:
        return False, f"Eğitim engellendi: QA doğrulaması başarısız ({report.summary})."

    return True, f"Veri seti eğitime hazır ({non_empty} onaylı örnek, QA kontrolleri başarılı)."


# ==============================================================================
# 5. Stale Model Tracking
# ==============================================================================

def mark_model_stale(
    reason: str,
    dataset_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Contract: dataset_factory/label_trust.py ↔ ui/yolo_factory_dialog.py.
    
    Sets stale status in metadata and pipeline state whenever dataset labels
    are modified, approved, or deleted.
    """
    now_str = datetime.now().isoformat()
    clean_reason = str(reason or "Labels updated/approved").strip()

    # 1. Update pipeline_ui_state.json if dataset_dir exists
    ds_root = Path(dataset_dir) if dataset_dir else Path("dataset")
    state_file = ds_root / PIPELINE_STATE_NAME
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["model_stale"] = True
            state["stale_reason"] = clean_reason
            state["stale_at"] = now_str
            state["trained"] = False
            state["validated"] = False
            state["published"] = False
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to update pipeline_ui_state: %s", exc)

    # 2. Write stale manifest in models directory
    md_root = Path(models_dir) if models_dir else Path("models")
    if md_root.is_dir():
        stale_file = md_root / STALE_MANIFEST_NAME
        try:
            payload = {
                "model_stale": True,
                "reason": clean_reason,
                "marked_at": now_str,
            }
            stale_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write stale manifest: %s", exc)

    logger.info("Model marked stale: %s", clean_reason)


def is_model_stale(
    dataset_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Check if model state is currently marked stale.
    
    Returns (is_stale: bool, reason: str).
    """
    md_root = Path(models_dir) if models_dir else Path("models")
    stale_file = md_root / STALE_MANIFEST_NAME
    if stale_file.is_file():
        try:
            data = json.loads(stale_file.read_text(encoding="utf-8"))
            if data.get("model_stale") is True:
                return True, str(data.get("reason", "Model işaretlendi: stale"))
        except Exception:
            pass

    ds_root = Path(dataset_dir) if dataset_dir else Path("dataset")
    state_file = ds_root / PIPELINE_STATE_NAME
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("model_stale") is True:
                return True, str(state.get("stale_reason", "Labels modified"))
        except Exception:
            pass

    return False, "Model güncel"


def clear_model_stale(
    dataset_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Clear stale flag after a new successful training run."""
    md_root = Path(models_dir) if models_dir else Path("models")
    stale_file = md_root / STALE_MANIFEST_NAME
    if stale_file.is_file():
        try:
            stale_file.unlink()
        except OSError:
            pass

    ds_root = Path(dataset_dir) if dataset_dir else Path("dataset")
    state_file = ds_root / PIPELINE_STATE_NAME
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["model_stale"] = False
            state.pop("stale_reason", None)
            state.pop("stale_at", None)
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# ==============================================================================
# 6. Review JSON <-> Labels TXT Atomic Synchronization (L16 Guardrail)
# ==============================================================================

def atomic_write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Atomically write text to target path using a temporary file in the same directory and os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to target path."""
    content = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, content, encoding="utf-8")


def verify_and_sync_review_txt_pair(
    json_path: Union[str, Path],
    txt_path: Union[str, Path],
    auto_repair: bool = True,
) -> bool:
    """Verify consistency between review JSON and labels TXT pair and optionally auto-repair.
    
    L16 Guardrail:
    - Checks box count and class consistency between review/{split}/{stem}.json and labels/{split}/{stem}.txt.
    - Uses atomic write (temp file write in same folder + os.replace) to guarantee zero corruption / partial writes.
    - Handles:
      1. Fully synchronized pairs (returns True, no writes).
      2. Orphan TXT (missing JSON -> reconstructed atomically from TXT).
      3. Orphan JSON (missing TXT -> reconstructed atomically from JSON).
      4. Corrupted JSON (syntax error -> salvaged from TXT if valid).
      5. Corrupted TXT (invalid syntax -> reconstructed from JSON if valid).
      6. Mismatched counts or classes -> synced atomically according to review ground truth.
    
    Returns:
        bool: True if verified or successfully repaired to a consistent state; False otherwise.
    """
    json_p = Path(json_path)
    txt_p = Path(txt_path)

    # 1. Parse JSON if exists
    json_exists = json_p.is_file()
    json_valid = False
    json_data: Optional[Dict[str, Any]] = None
    json_boxes: List[Dict[str, Any]] = []

    if json_exists:
        try:
            content = json_p.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                loaded = json.loads(content)
                if isinstance(loaded, dict):
                    json_data = loaded
                    json_valid = True
        except Exception as exc:
            logger.warning("Corrupt JSON in %s: %s", json_p, exc)
            json_valid = False

    if json_valid and json_data is not None:
        done_val = json_data.get("done")
        trusted_list = [b for b in json_data.get("trusted", []) if isinstance(b, dict)]
        weak_list = [b for b in json_data.get("weak", []) if isinstance(b, dict)]

        if done_val in ("ok", True):
            raw_cands = list(trusted_list) + list(weak_list)
        elif done_val in ("no", "rejected", False):
            raw_cands = []
        elif "trusted" in json_data:
            raw_cands = list(trusted_list)
        elif "boxes" in json_data:
            raw_cands = [b for b in json_data.get("boxes", []) if isinstance(b, dict)]
        elif "annotations" in json_data:
            raw_cands = [b for b in json_data.get("annotations", []) if isinstance(b, dict)]
        else:
            raw_cands = []

        # Sanitize with L17 bounds
        sanitized_cands, _ = sanitize_boxes(raw_cands, drop_invalid_size=True, auto_fix_coords=True)
        json_boxes = sanitized_cands

    # 2. Parse TXT if exists
    txt_exists = txt_p.is_file()
    txt_valid = False
    txt_boxes: List[Dict[str, Any]] = []

    if txt_exists:
        try:
            txt_content = txt_p.read_text(encoding="utf-8", errors="ignore")
            lines = txt_content.splitlines()
            parsed_boxes: List[Dict[str, Any]] = []
            parse_ok = True
            for raw_line in lines:
                s = raw_line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) >= 5:
                    try:
                        cid = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        b_dict = {"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h}
                        res = validate_label_box(b_dict)
                        if res.is_valid:
                            parsed_boxes.append({
                                "class_id": cid,
                                "cx": round(cx, 6),
                                "cy": round(cy, 6),
                                "w": round(w, 6),
                                "h": round(h, 6),
                            })
                        else:
                            parse_ok = False
                    except (ValueError, TypeError):
                        parse_ok = False
                else:
                    parse_ok = False

            txt_valid = parse_ok
            txt_boxes = parsed_boxes
        except Exception as exc:
            logger.warning("Corrupt or unreadable TXT in %s: %s", txt_p, exc)
            txt_valid = False

    # 3. Check consistency
    is_synced = False
    if json_exists and txt_exists and json_valid and txt_valid:
        # Check box count
        if len(json_boxes) == len(txt_boxes):
            # Check classes
            json_classes = sorted(int(b["class_id"]) for b in json_boxes)
            txt_classes = sorted(int(b["class_id"]) for b in txt_boxes)
            if json_classes == txt_classes:
                # Check coordinates match
                matched_all = True
                remaining_txt = list(txt_boxes)
                for jb in json_boxes:
                    match_found = False
                    for idx_t, tb in enumerate(remaining_txt):
                        if (
                            int(jb["class_id"]) == int(tb["class_id"])
                            and abs(float(jb["cx"]) - float(tb["cx"])) < 1e-3
                            and abs(float(jb["cy"]) - float(tb["cy"])) < 1e-3
                            and abs(float(jb["w"]) - float(tb["w"])) < 1e-3
                            and abs(float(jb["h"]) - float(tb["h"])) < 1e-3
                        ):
                            remaining_txt.pop(idx_t)
                            match_found = True
                            break
                    if not match_found:
                        matched_all = False
                        break
                if matched_all:
                    is_synced = True

    if is_synced:
        return True

    # If not in sync and auto_repair is disabled, return False
    if not auto_repair:
        return False

    # 4. Auto-Repair using Atomic Writes
    # Case A: JSON is valid, TXT is missing, corrupt, or inconsistent with JSON
    if json_valid:
        lines_out = [
            f"{int(b['class_id'])} {float(b['cx']):.6f} {float(b['cy']):.6f} {float(b['w']):.6f} {float(b['h']):.6f}"
            for b in json_boxes
        ]
        txt_str = ("\n".join(lines_out) + "\n") if lines_out else ""
        atomic_write_text(txt_p, txt_str)

        # Also sanitize json_p if it contained dirty/unfiltered boxes
        if json_data is not None and "trusted" in json_data:
            if len(json_data.get("trusted", [])) != len(json_boxes):
                json_data["trusted"] = json_boxes
                atomic_write_json(json_p, json_data)

        logger.info("Atomic sync: repaired %s from %s (%d boxes)", txt_p.name, json_p.name, len(json_boxes))
        return True

    # Case B: JSON is missing or corrupt, but TXT is valid
    if txt_exists and txt_valid:
        stem = txt_p.stem
        parent_name = txt_p.parent.name
        split = parent_name if parent_name in ("train", "val", "test") else "train"
        repaired_payload = {
            "stem": stem,
            "split": split,
            "trusted": txt_boxes,
            "weak": [],
            "done": "ok" if txt_boxes else None,
            "auto_repaired": True,
        }
        atomic_write_json(json_p, repaired_payload)
        logger.info("Atomic sync: repaired %s from %s (%d boxes)", json_p.name, txt_p.name, len(txt_boxes))
        return True

    # Case C: Neither can be salvaged
    logger.error("Atomic sync failed: both %s and %s are missing or irreparably corrupt", json_p, txt_p)
    return False


def sync_all_review_txt_pairs(
    dataset_dir: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "val"),
    auto_repair: bool = True,
) -> Dict[str, Any]:
    """Scan and verify/synchronize all review JSON and label TXT pairs across dataset splits."""
    root = Path(dataset_dir)
    stats: Dict[str, int] = {
        "total_pairs": 0,
        "already_synced": 0,
        "auto_repaired": 0,
        "failed": 0,
        "orphan_txt_fixed": 0,
        "orphan_json_fixed": 0,
    }
    review_dir = root / "review"
    labels_dir = root / "labels"

    all_stems_by_split: Dict[str, Set[str]] = {}
    for split in splits:
        stems: Set[str] = set()
        s_rev = review_dir / split
        if s_rev.is_dir():
            for p in s_rev.glob("*.json"):
                if p.name != "trusted_seed.json":
                    stems.add(p.stem)
        s_lbl = labels_dir / split
        if s_lbl.is_dir():
            for p in s_lbl.glob("*.txt"):
                stems.add(p.stem)
        all_stems_by_split[split] = stems

    for split, stems in all_stems_by_split.items():
        for stem in sorted(stems):
            stats["total_pairs"] += 1
            j_path = review_dir / split / f"{stem}.json"
            t_path = labels_dir / split / f"{stem}.txt"

            had_json = j_path.is_file()
            had_txt = t_path.is_file()

            # First test without repair to count already_synced
            was_synced = verify_and_sync_review_txt_pair(j_path, t_path, auto_repair=False)
            if was_synced:
                stats["already_synced"] += 1
                continue

            if not auto_repair:
                stats["failed"] += 1
                continue

            # Attempt repair
            ok = verify_and_sync_review_txt_pair(j_path, t_path, auto_repair=True)
            if ok:
                stats["auto_repaired"] += 1
                if had_txt and not had_json:
                    stats["orphan_txt_fixed"] += 1
                elif had_json and not had_txt:
                    stats["orphan_json_fixed"] += 1
            else:
                stats["failed"] += 1

    return stats


# ==============================================================================
# 7. Comprehensive Self-Tests
# ==============================================================================

def run_self_tests() -> None:
    """Run comprehensive verification for Dataset QA, Human Gate, and Stale Tracking."""
    import tempfile

    print("Running Dataset QA and Validation self-tests...")

    # --- Test 1: Zero-area checks ---
    is_zero, _ = check_zero_area(0.0, 0.5)
    assert is_zero, "Expected zero width to fail"
    is_neg, _ = check_zero_area(0.2, -0.05)
    assert is_neg, "Expected negative height to fail"
    is_valid, _ = check_zero_area(0.1, 0.2)
    assert not is_valid, "Expected valid w/h to pass zero area check"

    # --- Test 2: Out-of-bounds checks and fixes ---
    is_oob1, msgs1 = check_out_of_bounds(-0.1, 0.5, 0.2, 0.2)
    assert is_oob1, "Expected negative cx to be out of bounds"
    is_oob2, msgs2 = check_out_of_bounds(1.2, 0.5, 0.2, 0.2)
    assert is_oob2, "Expected cx > 1.0 to be out of bounds"
    is_oob3, msgs3 = check_out_of_bounds(0.95, 0.5, 0.2, 0.2)
    assert is_oob3, "Expected corner x2 = 1.05 to be out of bounds"

    # Fix out-of-bounds corner
    fixed = fix_out_of_bounds(0.95, 0.5, 0.2, 0.2)
    assert fixed is not None, "Expected corner out-of-bounds to be fixable"
    fcx, fcy, fw, fh = fixed
    fx1, fy1, fx2, fy2 = box_xywh_to_xyxy(fcx, fcy, fw, fh)
    assert fx1 >= 0.0 and fx2 <= 1.0, f"Expected clamped corners in [0, 1], got x1={fx1}, x2={fx2}"

    # Pixel coordinate normalization
    fixed_px = fix_out_of_bounds(640.0, 360.0, 100.0, 50.0, image_width=1280.0, image_height=720.0)
    assert fixed_px is not None
    assert abs(fixed_px[0] - 0.5) < 1e-4 and abs(fixed_px[1] - 0.5) < 1e-4

    # Completely out of bounds cannot be salvaged
    assert fix_out_of_bounds(2.5, 2.5, 0.1, 0.1) is None

    # --- Test 3: Duplicate and Overlap checks (IoU > 0.8) ---
    b1 = (0.5, 0.5, 0.2, 0.2)
    b2 = (0.505, 0.505, 0.2, 0.2)  # High overlap
    b3 = (0.8, 0.8, 0.1, 0.1)      # No overlap
    iou_high = compute_box_iou(b1, b2, is_xywh=True)
    assert iou_high > 0.80, f"Expected IoU > 0.80, got {iou_high}"
    iou_low = compute_box_iou(b1, b3, is_xywh=True)
    assert iou_low == 0.0, f"Expected 0 IoU, got {iou_low}"

    test_boxes = [
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.95, "hand_label": True},
        {"class_id": 0, "cx": 0.505, "cy": 0.505, "w": 0.2, "h": 0.2, "confidence": 0.80},  # duplicate
        {"class_id": 1, "cx": 0.505, "cy": 0.505, "w": 0.2, "h": 0.2, "confidence": 0.85},  # different class
        {"class_id": 2, "cx": 0.1, "cy": 0.1, "w": 0.0, "h": 0.2},  # zero area
        {"class_id": 3, "cx": 0.95, "cy": 0.2, "w": 0.2, "h": 0.2},  # corner oob
    ]
    clean, issues = sanitize_boxes(test_boxes, iou_threshold=0.80)
    # Zero area must be dropped
    assert not any(b["class_id"] == 2 for b in clean), "Zero area box should be dropped"
    # Duplicate class 0 must be deduplicated (keeping hand_label/conf=0.95)
    c0_boxes = [b for b in clean if b["class_id"] == 0]
    assert len(c0_boxes) == 1, f"Expected 1 class 0 box after deduplication, got {len(c0_boxes)}"
    assert c0_boxes[0].get("hand_label") is True
    # Different class box must survive
    assert any(b["class_id"] == 1 for b in clean)
    # Corner oob must be fixed to [0, 1]
    c3_boxes = [b for b in clean if b["class_id"] == 3]
    assert len(c3_boxes) == 1
    assert c3_boxes[0]["cx"] + c3_boxes[0]["w"] * 0.5 <= 1.000001

    # --- Test 4: Human Confirmation Gate & Stale Model Marking ---
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp = Path(tmp_dir_str)
        ds_dir = tmp / "dataset"
        models_dir = tmp / "models"
        ds_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create pipeline UI state file
        (ds_dir / PIPELINE_STATE_NAME).write_text(
            json.dumps({"trained": True, "validated": True, "published": True, "model_stale": False}),
            encoding="utf-8",
        )

        # Create review directory with pending weak box
        rev_train = ds_dir / "review" / "train"
        rev_train.mkdir(parents=True)
        sample_json = rev_train / "sample_01.json"
        sample_json.write_text(
            json.dumps({
                "stem": "sample_01",
                "split": "train",
                "trusted": [],
                "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.7}],
                "done": None,
            }),
            encoding="utf-8",
        )

        # Labels directory
        labels_train = ds_dir / "labels" / "train"
        labels_train.mkdir(parents=True)
        (labels_train / "sample_01.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        # Human gate check: pending > 0 must block training
        pending = count_pending_review_items(ds_dir)
        assert pending == 1, f"Expected 1 pending item, got {pending}"
        ready, reason = is_dataset_ready_for_training(ds_dir)
        assert not ready, "Expected training to be blocked by pending items"
        assert "onay bekleyen" in reason.lower() or "pending" in reason.lower()

        # Approve the sample: weak becomes empty, trusted has the box
        sample_json.write_text(
            json.dumps({
                "stem": "sample_01",
                "split": "train",
                "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.7, "hand_label": True}],
                "weak": [],
                "done": "ok",
            }),
            encoding="utf-8",
        )
        assert count_pending_review_items(ds_dir) == 0
        ready2, reason2 = is_dataset_ready_for_training(ds_dir)
        assert ready2, f"Expected dataset to be ready after approval, got: {reason2}"

        # Stale model marking check
        mark_model_stale("Etiket onaylandı: sample_01", dataset_dir=ds_dir, models_dir=models_dir)
        is_stale, s_reason = is_model_stale(dataset_dir=ds_dir, models_dir=models_dir)
        assert is_stale is True, "Expected model to be marked stale"
        assert "sample_01" in s_reason

        # State check in pipeline UI state
        state_data = json.loads((ds_dir / PIPELINE_STATE_NAME).read_text(encoding="utf-8"))
        assert state_data.get("model_stale") is True
        assert state_data.get("trained") is False
        assert state_data.get("validated") is False
        assert state_data.get("published") is False

        # Clear stale
        clear_model_stale(dataset_dir=ds_dir, models_dir=models_dir)
        assert is_model_stale(dataset_dir=ds_dir, models_dir=models_dir)[0] is False

        # --- Test 5: L17 Box Size Bounds & validate_label_box ---
        # Micro noise (< 0.004)
        v_micro = validate_label_box(0.5, 0.5, 0.003, 0.05)
        assert not v_micro, "Micro width (<0.004) must be rejected"
        assert "micro noise" in v_micro.reason.lower() or "mikro" in v_micro.reason.lower()
        is_val, reason = validate_label_box({"cx": 0.5, "cy": 0.5, "w": 0.05, "h": 0.002})
        assert not is_val, "Micro height (<0.004) must be rejected"

        # Massive screen-filling box (> 0.65)
        v_massive = validate_label_box(0.5, 0.5, 0.70, 0.50)
        assert not v_massive, "Massive width (>0.65) must be rejected"
        assert "massive" in v_massive.reason.lower() or "aşırı büyük" in v_massive.reason.lower()
        v_massive_h = validate_label_box(0.5, 0.5, 0.40, 0.66)
        assert not v_massive_h, "Massive height (>0.65) must be rejected"

        # Zero-area and negative dimensions
        assert not validate_label_box(0.5, 0.5, 0.0, 0.1), "Zero width must be rejected"
        assert not validate_label_box(0.5, 0.5, 0.1, -0.05), "Negative height must be rejected"

        # Out-of-bounds coordinates
        assert not validate_label_box(-0.05, 0.5, 0.1, 0.1), "Negative cx must be rejected"
        assert not validate_label_box(1.05, 0.5, 0.1, 0.1), "cx > 1.0 must be rejected"
        assert not validate_label_box(0.5, -0.05, 0.1, 0.1), "Negative cy must be rejected"
        assert not validate_label_box(0.5, 1.05, 0.1, 0.1), "cy > 1.0 must be rejected"
        assert not validate_label_box(0.98, 0.5, 0.1, 0.1, strict_bounds=True), "Boundary x2 > 1.0 must be rejected"

        # Valid boxes
        v_valid = validate_label_box(0.5, 0.5, 0.1, 0.1, class_id=0)
        assert v_valid, f"Expected valid box to pass, got: {v_valid.reason}"
        assert v_valid.is_valid is True
        assert v_valid.reason == ""

        # Tuple unpacking
        valid_flag, reason_msg = validate_label_box({"x_center": 0.3, "y_center": 0.4, "width": 0.2, "height": 0.2})
        assert valid_flag is True
        assert reason_msg == ""

        # Boundary threshold test: exact 0.004 and 0.65
        assert validate_label_box(0.5, 0.5, 0.004, 0.004), "Exact 0.004 must be accepted"
        assert validate_label_box(0.5, 0.5, 0.65, 0.65), "Exact 0.65 must be accepted"

        # --- Test 6: sanitize_dataset_boxes L17 filtering ---
        raw_test_boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.9},    # Valid
            {"class_id": 1, "cx": 0.2, "cy": 0.2, "w": 0.002, "h": 0.05},                     # Micro noise
            {"class_id": 2, "cx": 0.4, "cy": 0.4, "w": 0.80, "h": 0.50},                      # Massive
            {"class_id": 3, "cx": 0.1, "cy": 0.1, "w": 0.0, "h": 0.1},                        # Zero area
            {"class_id": 4, "cx": 2.0, "cy": 2.0, "w": 0.1, "h": 0.1},                        # Completely OOB
        ]
        sanitized_res, issues_res = sanitize_dataset_boxes(raw_test_boxes)
        assert len(sanitized_res) == 1, f"Expected exactly 1 surviving box, got {len(sanitized_res)}"
        assert sanitized_res[0]["class_id"] == 0
        issue_types = [iss.issue_type for iss in issues_res]
        assert "micro_noise" in issue_types or "size_bounds" in issue_types
        assert "massive_box" in issue_types or "size_bounds" in issue_types
        assert "zero_area" in issue_types

        # --- Test 7: L16 Review JSON <-> Labels TXT Atomic Synchronization ---
        sync_tmp = tmp / "sync_test"
        sync_tmp.mkdir(parents=True, exist_ok=True)
        lbl_dir = sync_tmp / "labels" / "train"
        rev_dir = sync_tmp / "review" / "train"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        rev_dir.mkdir(parents=True, exist_ok=True)

        j_file = rev_dir / "sync_sample.json"
        t_file = lbl_dir / "sync_sample.txt"

        # Case 7.1: Synchronized pair
        j_file.write_text(
            json.dumps({
                "stem": "sync_sample",
                "split": "train",
                "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
                "weak": [],
                "done": "ok",
            }),
            encoding="utf-8",
        )
        t_file.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
        assert verify_and_sync_review_txt_pair(j_file, t_file, auto_repair=False) is True

        # Case 7.2: Orphan TXT (missing JSON) -> Auto-repair
        orphan_t = lbl_dir / "orphan_t.txt"
        orphan_j = rev_dir / "orphan_t.json"
        orphan_t.write_text("1 0.400000 0.400000 0.200000 0.200000\n", encoding="utf-8")
        assert verify_and_sync_review_txt_pair(orphan_j, orphan_t, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(orphan_j, orphan_t, auto_repair=True) is True
        assert orphan_j.is_file(), "Orphan TXT should result in created JSON"
        loaded_j = json.loads(orphan_j.read_text(encoding="utf-8"))
        assert len(loaded_j.get("trusted", [])) == 1
        assert loaded_j["trusted"][0]["class_id"] == 1

        # Case 7.3: Orphan JSON (missing TXT) -> Auto-repair
        orphan2_j = rev_dir / "orphan_j.json"
        orphan2_t = lbl_dir / "orphan_j.txt"
        orphan2_j.write_text(
            json.dumps({
                "stem": "orphan_j",
                "split": "train",
                "trusted": [{"class_id": 3, "cx": 0.3, "cy": 0.3, "w": 0.15, "h": 0.15}],
            }),
            encoding="utf-8",
        )
        assert verify_and_sync_review_txt_pair(orphan2_j, orphan2_t, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(orphan2_j, orphan2_t, auto_repair=True) is True
        assert orphan2_t.is_file(), "Orphan JSON should result in created TXT"
        assert "3 0.300000 0.300000 0.150000 0.150000" in orphan2_t.read_text(encoding="utf-8")

        # Case 7.4: Corrupt JSON syntax -> Auto-repair from TXT
        corrupt_j = rev_dir / "corrupt_j.json"
        corrupt_t = lbl_dir / "corrupt_j.txt"
        corrupt_j.write_text("{INCOMPLETE JSON SYNTAX ...", encoding="utf-8")
        corrupt_t.write_text("2 0.600000 0.600000 0.120000 0.120000\n", encoding="utf-8")
        assert verify_and_sync_review_txt_pair(corrupt_j, corrupt_t, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(corrupt_j, corrupt_t, auto_repair=True) is True
        recovered = json.loads(corrupt_j.read_text(encoding="utf-8"))
        assert recovered["trusted"][0]["class_id"] == 2

        # Case 7.5: Out-of-sync count / bad TXT -> Auto-repair from JSON
        bad_txt = lbl_dir / "bad_txt.txt"
        valid_json = rev_dir / "bad_txt.json"
        valid_json.write_text(
            json.dumps({
                "stem": "bad_txt",
                "split": "train",
                "trusted": [
                    {"class_id": 5, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
                    {"class_id": 6, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1},
                ],
            }),
            encoding="utf-8",
        )
        bad_txt.write_text("5 0.5 0.5 0.1 0.1\n999 malformed noise\n", encoding="utf-8")
        assert verify_and_sync_review_txt_pair(valid_json, bad_txt, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(valid_json, bad_txt, auto_repair=True) is True
        fixed_lines = [ln.strip() for ln in bad_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(fixed_lines) == 2, f"Expected 2 lines in repaired TXT, got {len(fixed_lines)}"
        assert fixed_lines[0].startswith("5 ")
        assert fixed_lines[1].startswith("6 ")

        # --- Test 8: sync_all_review_txt_pairs sweep ---
        stats = sync_all_review_txt_pairs(sync_tmp, auto_repair=True)
        assert stats["total_pairs"] >= 4
        assert stats["failed"] == 0

    print("ALL DATASET QA, HUMAN GATE, STALE MODEL, ATOMIC SYNC, AND BOX BOUND TESTS PASSED!")


__all__ = [
    "DEFAULT_IOU_OVERLAP_THRESHOLD",
    "DEFAULT_MIN_BOX_DIM",
    "DEFAULT_MIN_BOX_AREA",
    "MIN_BOX_DIM_BOUND",
    "MAX_BOX_DIM_BOUND",
    "MIN_BOX_AREA_BOUND",
    "STALE_MANIFEST_NAME",
    "PIPELINE_STATE_NAME",
    "BoxValidationResult",
    "ValidationIssue",
    "ValidationReport",
    "box_xywh_to_xyxy",
    "box_xyxy_to_xywh",
    "compute_box_iou",
    "check_zero_area",
    "check_box_size_bounds",
    "check_out_of_bounds",
    "fix_out_of_bounds",
    "validate_label_box",
    "detect_overlapping_boxes",
    "sanitize_boxes",
    "sanitize_dataset_boxes",
    "atomic_write_text",
    "atomic_write_json",
    "verify_and_sync_review_txt_pair",
    "sync_all_review_txt_pairs",
    "validate_yolo_file",
    "count_pending_review_items",
    "validate_dataset",
    "is_dataset_ready_for_training",
    "mark_model_stale",
    "is_model_stale",
    "clear_model_stale",
    "run_self_tests",
]


if __name__ == "__main__":
    run_self_tests()

