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
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

try:  # P1-13 (K-33): tek "ayni kutu" tanimi box_match.py'dedir.
    from dataset_factory.box_match import same_box
except ImportError:  # standalone calistirma: python dataset_validator.py
    from box_match import same_box  # type: ignore[no-redef]

logger = logging.getLogger("gatherion.dataset_validator")

# Default thresholds
DEFAULT_IOU_OVERLAP_THRESHOLD: float = 0.80

# L17 Box Size Bounds & Guardrails (Micro noise & massive screen-filling bounds)
MIN_BOX_DIM_BOUND: float = 0.004   # Micro noise threshold: w < 0.004 or h < 0.004
MAX_BOX_DIM_BOUND: float = 0.65    # Massive box threshold: w > 0.65 or h > 0.65
MIN_BOX_AREA_BOUND: float = 1.6e-5 # Minimum area: 0.004 * 0.004

# Zero-area / sub-minimal defaults follow the L17 bounds (F-0939): a box smaller
# than 0.004 in either dimension (or smaller than 0.004*0.004 in area) is micro
# noise, so check_zero_area must not pass it on its own. This also keeps
# fix_out_of_bounds' default min_dim consistent with MIN_BOX_DIM_BOUND (F-1044).
DEFAULT_MIN_BOX_DIM: float = MIN_BOX_DIM_BOUND
DEFAULT_MIN_BOX_AREA: float = MIN_BOX_AREA_BOUND

# Canonical YOLO class ids for this pipeline (dataset_factory/class_taxonomy.py:
# CANONICAL_NAMES covers 0..26). validate_label_box / validate_yolo_file reject
# ids outside this range (F-0941 / F-0851).
MAX_CANONICAL_CLASS_ID: int = 26

STALE_MANIFEST_NAME = "stale_manifest.json"
PIPELINE_STATE_NAME = "pipeline_ui_state.json"

# Yayın/doğrulama kabul eşiği (mAP50) — TEK KAYNAK (v11 tazelik kapısı).
# Dialog (ui/yolo_factory_dialog.py) ve motor (dataset_factory/
# rtx_trainer_engine.py) aynı eşiği okur: 0.32'lik yarım eğitimi
# engeller, ulaşılabilir eğitimi geçirir. Eşik değişikliği Selçuk kararıdır.
MIN_PUBLISH_MAP50 = 0.50


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
        self._tally_issue(issue)

    def _tally_issue(self, issue: ValidationIssue) -> None:
        """Bump severity/type counters for one issue WITHOUT appending it.

        Shared by add_issue() and by the capped path in validate_dataset(),
        so the counters can never drift apart again (previously the cap
        branch duplicated this whole chain by hand).
        """
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
    """Convert corner format (x1, y1, x2, y2) to YOLO center format (cx, cy, w, h).

    Reversed corners (x2 < x1 or y2 < y1) raise ValueError instead of being
    silently clamped to a zero-size box (F-1041).
    """
    if float(x2) < float(x1) or float(y2) < float(y1):
        raise ValueError(f"Reversed box corners: ({x1}, {y1}) -> ({x2}, {y2})")
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
    if w * h < MIN_BOX_AREA_BOUND:
        return True, f"Alan asgari sınırın altında (alan={w * h:.8f} < {MIN_BOX_AREA_BOUND})"
    return False, ""


def _strict_class_id(raw: Any) -> Optional[int]:
    """Parse a class id strictly -- never silently truncate (S-6301/S-6302).

    Only accepts genuine integers: int (bool hariç), integral float
    (1.0 -> 1), or an integer-token string ("1", "+3"). Returns the int,
    or None when the value is not a valid integer class id.

    The old int(raw) truncated 1.9 -> 1 (wrong class in clean data) and
    int("1.9") raised ValueError that callers swallowed, so the 0..26
    range check (F-0941) never ran for those boxes.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if math.isfinite(raw) and raw.is_integer():
            return int(raw)
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return None
    return None


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
    # S-6301: an explicitly passed class_id must also be a genuine integer;
    # a float 1.9 passed here used to sail through the range check below
    # after int() truncation (or skip it entirely when unparseable).
    if class_id is not None:
        parsed_param_cid = _strict_class_id(class_id)
        if parsed_param_cid is None:
            return BoxValidationResult(
                False,
                f"Invalid class_id: {class_id!r} (expected an integer 0..{MAX_CANONICAL_CLASS_ID})",
            )
        class_id = parsed_param_cid

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
                    # S-6301: strict parse -- int(1.9) truncated to 1 and
                    # int("1.9") raised ValueError that was swallowed here,
                    # leaving class_id None so step 7 never ran. A
                    # non-integer class_id is now rejected outright.
                    parsed_cid = _strict_class_id(raw_cid)
                    if parsed_cid is None:
                        return BoxValidationResult(
                            False,
                            f"Invalid class_id: {raw_cid!r} (expected an integer 0..{MAX_CANONICAL_CLASS_ID})",
                        )
                    class_id = parsed_cid
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
                # S-6301: strict parse -- int(1.9) truncated to 1. If seq[0]
                # is not a genuine integer class id, fall through to the
                # coordinate reading below (a 5-tuple of pure coordinates);
                # a non-integral seq[0] then fails the bounds checks there.
                parsed_seq_cid = _strict_class_id(seq[0])
                if parsed_seq_cid is not None:
                    try:
                        class_id = parsed_seq_cid
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
                # Explicit class_id + 5-tuple means (class_id, cx, cy, w, h);
                # reading seq[0..3] as coordinates would shift everything (F-0940).
                # S-6301: strict parse -- int(1.9) used to truncate silently.
                seq_cid = _strict_class_id(seq[0])
                if seq_cid is None:
                    return BoxValidationResult(False, f"Invalid class_id in sequence: {seq[0]!r}")
                if seq_cid != class_id:
                    return BoxValidationResult(
                        False,
                        f"Conflicting class_id: argument={class_id}, sequence[0]={seq_cid}",
                    )
                try:
                    val_cx, val_cy, val_w, val_h = float(seq[1]), float(seq[2]), float(seq[3]), float(seq[4])
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

    # 2. Zero-area / negative dimension check (w <= 0 or h <= 0 already implies
    # area <= 0, so no separate area branch is reachable here -- F-1043).
    if val_w <= 0.0 or val_h <= 0.0:
        return BoxValidationResult(False, f"Zero or negative dimension: w={val_w:.6f}, h={val_h:.6f}")

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

    # 7. Optional class_id check (canonical taxonomy is 0..26, F-0941)
    if class_id is not None and (class_id < 0 or class_id > MAX_CANONICAL_CLASS_ID):
        return BoxValidationResult(
            False,
            f"Invalid class_id: {class_id} (expected 0..{MAX_CANONICAL_CLASS_ID})",
        )

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

    # P1-7 (K-20): karisik birimli kutu — her koordinat kendi birimine
    # gore normalize edilir. Eski toplu kural (herhangi biri >1.0 ise
    # HEPSINI bol) norm bir koordinati da bolup kutuyu kokeneye
    # cokeriyordu (orn. {"cx": 0.5, "w": 100} -> cx ~= 0.00026).
    # Piksel oldugu kendi degerinden belli olan (>1.0) koordinat yalnizca
    # kendi ekseninin goruntu boyutuna bolunur; norm koordinata dokunulmaz.
    if image_width and image_height and image_width > 0 and image_height > 0:
        if cx > 1.0:
            cx = cx / float(image_width)
        if cy > 1.0:
            cy = cy / float(image_height)
        if w > 1.0:
            w = w / float(image_width)
        if h > 1.0:
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


def _parse_box_for_overlap(box: Any) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
    """Parse one box for overlap detection; None if malformed (F-0849)."""
    if not isinstance(box, dict):
        return None
    # S-6302: strict class_id parse -- int(1.9) truncated to 1, so boxes of
    # different classes looked like the same class in overlap detection.
    # A non-integer class_id makes the box malformed: skip it (None).
    cid = _strict_class_id(box.get("class_id", -1))
    if cid is None:
        return None
    try:
        coords = (
            float(box.get("cx", 0.0)),
            float(box.get("cy", 0.0)),
            float(box.get("w", 0.0)),
            float(box.get("h", 0.0)),
        )
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    return cid, coords


def detect_overlapping_boxes(
    boxes: Sequence[Dict[str, Any]],
    iou_threshold: float = DEFAULT_IOU_OVERLAP_THRESHOLD,
    same_class_only: bool = True,
) -> List[Tuple[int, int, float, str]]:
    """Detect pairs of overlapping boxes with IoU > iou_threshold.

    Malformed boxes (non-numeric class_id / coordinates, NaN/Inf) are skipped
    instead of crashing the whole check (F-0849).

    Fast path, bit-identical results to the old O(n^2) all-pairs scan (same
    pairs, same (i, j) order, same IoU values, same descriptions):
    1. Boxes are grouped by class first (same_class_only, the default):
       different-class pairs can never be a hit, so they are never even
       visited -- the old scan visited every pair to check the class.
    2. Within a group a sweep in x1 order stops the inner scan as soon as
       the next box starts at or after the current box's x2, and a y-interval
       check skips the IoU entirely. Two boxes whose x- or y-intervals do
       not intersect have IoU 0, so no hit can be missed.
    3. Corners are precomputed once per box and the IoU is evaluated inline
       from them, instead of re-converting xywh -> xyxy on every pair
       (compute_box_iou does that conversion per call).

    Returns list of tuples: (index_i, index_j, iou_score, details_str).
    """
    overlaps: List[Tuple[int, int, float, str]] = []
    parsed: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
    for idx, box in enumerate(boxes):
        entry = _parse_box_for_overlap(box)
        if entry is not None:
            parsed.append((idx, entry[0], entry[1]))
    if not parsed:
        return overlaps

    # Group by class (single group when same_class_only is False).
    groups: Dict[int, List[Tuple[int, int, Tuple[float, float, float, float]]]] = {}
    for idx, cid, coords in parsed:
        groups.setdefault(cid if same_class_only else 0, []).append((idx, cid, coords))

    hits: List[Tuple[int, int, int, float]] = []  # (lo, hi, lo_class, iou)
    for group in groups.values():
        # (x1, y1, x2, y2, area, orig_idx, class_id), sorted by x1.
        sweep = sorted(
            (
                coords[0] - coords[2] * 0.5,
                coords[1] - coords[3] * 0.5,
                coords[0] + coords[2] * 0.5,
                coords[1] + coords[3] * 0.5,
                max(0.0, coords[2]) * max(0.0, coords[3]),
                idx,
                cid,
            )
            for idx, cid, coords in group
        )
        for a in range(len(sweep)):
            x1_a, y1_a, x2_a, y2_a, area_a, i, cid_i = sweep[a]
            for b in range(a + 1, len(sweep)):
                x1_b, y1_b, x2_b, y2_b, area_b, j, cid_j = sweep[b]
                if x1_b >= x2_a:
                    # Sorted by x1: every later box starts even further
                    # right, so none of them can overlap box i in x.
                    break
                if y1_b >= y2_a or y2_b <= y1_a:
                    continue
                # Inline IoU from precomputed corners (same arithmetic as
                # compute_box_iou on xyxy input, so values match exactly).
                iw = (x2_a if x2_a < x2_b else x2_b) - (x1_a if x1_a > x1_b else x1_b)
                ih = (y2_a if y2_a < y2_b else y2_b) - (y1_a if y1_a > y1_b else y1_b)
                # compute_box_iou clamps iw/ih to >= 0: a box with negative
                # w AND h passes the interval guards with iw < 0 and ih < 0,
                # so an unclamped iw*ih is positive and can false-hit.
                if iw <= 0.0 or ih <= 0.0:
                    continue
                inter = iw * ih
                if inter <= 1e-12:
                    continue
                union = area_a + area_b - inter
                if union <= 1e-12:
                    continue
                iou = inter / union
                if iou > iou_threshold:
                    # Description names the class of the lower input index,
                    # exactly like the old scan did.
                    lo, hi, lo_cid = (i, j, cid_i) if i < j else (j, i, cid_j)
                    hits.append((lo, hi, lo_cid, iou))
    # Restore the original all-pairs (i, j) order: callers (validate_yolo_file)
    # append these as issues in list order.
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    for lo, hi, lo_cid, iou in hits:
        overlaps.append((
            lo,
            hi,
            iou,
            f"Kutu #{lo} ve #{hi} aynı sınıfta ({lo_cid}) yüksek örtüşme "
            f"(IoU={iou:.3f} > {iou_threshold})",
        ))
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

        # S-6302: strict class_id parse -- int(1.9) truncated to 1, so a
        # box with class_id 1.9 entered clean data mislabelled as class 1
        # (and, via verify_and_sync_review_txt_pair, into training TXT).
        # A non-integer class_id is a syntax error: the box is dropped.
        cid = _strict_class_id(raw.get("class_id", raw.get("cls", -1)))
        if cid is None:
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=(
                        f"Kutu #{idx} geçersiz class_id: "
                        f"{raw.get('class_id', raw.get('cls', -1))!r} "
                        "(tam sayı bekleniyordu), reddedildi"
                    ),
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                )
            )
            continue
        try:
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

        # 1. Zero-area / micro-noise check. With the L17 defaults, check_zero_area
        # also catches sub-minimal (micro noise) boxes. Those are reported as
        # micro_noise (not zero_area) so size-bounds counters stay accurate, and
        # a zero-area box is ALWAYS dropped: drop_zero_area=False must not let
        # an invalid box into the cleaned output (F-0942). The flag is kept for
        # API compatibility but no longer keeps invalid boxes.
        is_zero, zero_msg = check_zero_area(w, h)
        if is_zero:
            itype_zero = "zero_area" if (w <= 0.0 or h <= 0.0 or not (math.isfinite(w) and math.isfinite(h))) else "micro_noise"
            issues.append(
                ValidationIssue(
                    issue_type=itype_zero,
                    severity="error",
                    message=f"Kutu #{idx} reddedildi: {zero_msg}",
                    file_path=file_path,
                    box_index=idx,
                    box=raw,
                    details={"w": w, "h": h},
                )
            )
            continue

        # 2. L17 Box size bounds check (micro noise & massive screen-filling box)
        is_bad_size, size_msg = check_box_size_bounds(w, h, min_dim=min_dim, max_dim=max_dim)
        if is_bad_size:
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
            # NOTE (A23): an error-severity size violation must never survive
            # into the cleaned output. drop_invalid_size=False used to keep
            # massive boxes (w/h > max_dim) despite the error issue -- the
            # same self-contradiction F-0942 fixed for drop_zero_area. The
            # flag is kept for API compatibility but no longer keeps
            # invalid boxes (no production caller passes False).
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

        # NOTE (F-1045): the old step-4 centre hard check was unreachable -- step 3
        # above already rejects unfixable OOB centres and clips fixable ones into
        # [0, 1], so any box reaching here has its centre in bounds (up to the
        # 1e-4 tolerance check_out_of_bounds applies).
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

    # Original input order, restored at the end so dedup output does not get
    # regrouped by class (F-1046).
    _input_order = {id(b): pos for pos, b in enumerate(valid_candidates)}

    deduped_boxes: List[Dict[str, Any]] = []
    for cid, cand_list in by_class.items():
        # Sort so trusted / hand_label comes first, then confidence, then area.
        # Confidence may be non-numeric (e.g. a string) -- never crash the
        # sort, fall back to 0.0 (F-0850).
        def _sort_key(item: Dict[str, Any]) -> Tuple[int, float, float]:
            is_hand = 1 if (item.get("hand_label") is True or item.get("human_approved") is True) else 0
            is_trust = 1 if item.get("trusted") is True else 0
            try:
                conf = float(item.get("confidence", item.get("conf", 0.0)))
            except (ValueError, TypeError):
                conf = 0.0
            if not math.isfinite(conf):
                conf = 0.0
            try:
                area = float(item.get("w", 0.0)) * float(item.get("h", 0.0))
            except (ValueError, TypeError):
                area = 0.0
            if not math.isfinite(area):
                area = 0.0
            return (is_hand * 2 + is_trust, conf, area)

        sorted_cands = sorted(cand_list, key=_sort_key, reverse=True)
        kept_class_boxes: List[Dict[str, Any]] = []

        for cand in sorted_cands:
            cand_coords = (cand["cx"], cand["cy"], cand["w"], cand["h"])
            cand_x1 = cand_coords[0] - cand_coords[2] * 0.5
            cand_x2 = cand_coords[0] + cand_coords[2] * 0.5
            cand_y1 = cand_coords[1] - cand_coords[3] * 0.5
            cand_y2 = cand_coords[1] + cand_coords[3] * 0.5
            is_duplicate = False
            for kept in kept_class_boxes:
                kept_coords = (kept["cx"], kept["cy"], kept["w"], kept["h"])
                # Interval prune (kept order is preserved, so the first
                # duplicate found is still the same one): boxes whose
                # x- or y-intervals do not intersect have IoU 0 and can
                # never be a duplicate -- skip the IoU computation.
                # (A23: same disjoint-interval pre-check; measured ~3x
                # faster at n=500, see ai-results/A23.json.)
                if (
                    kept_coords[0] + kept_coords[2] * 0.5 <= cand_x1
                    or kept_coords[0] - kept_coords[2] * 0.5 >= cand_x2
                    or kept_coords[1] + kept_coords[3] * 0.5 <= cand_y1
                    or kept_coords[1] - kept_coords[3] * 0.5 >= cand_y2
                ):
                    continue
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

    deduped_boxes.sort(key=lambda b: _input_order.get(id(b), 0))
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
        # S-6303: exactly 5 tokens -- the old `< 5` check silently accepted
        # 6+ token lines ("0 0.5 0.5 0.1 0.1 garbage") and swallowed the
        # extras, although the error message says "5 eleman bekleniyordu".
        if len(parts) != 5:
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

        # A23: extra tokens past the 5 YOLO columns used to be silently
        # ignored (e.g. a trailing confidence or garbage column passed QA
        # with no issue at all). Warn -- the box is still read from the
        # first 5 columns, so this never flips a file to invalid.
        if len(parts) > 5:
            issues.append(
                ValidationIssue(
                    issue_type="extra_tokens",
                    severity="warning",
                    message=(
                        f"Satır {line_no}: 5 eleman bekleniyordu, {len(parts)} "
                        f"bulundu; fazladan tokenlar yok sayıldı ({' '.join(parts[5:])})"
                    ),
                    file_path=str(path),
                    line_number=line_no,
                )
            )

        # Class id must be a plain integer token in the canonical range -- the
        # old int(float(...)) silently truncated "1.9" to class 1 and no class
        # was ever validated (F-0851).
        try:
            if not re.fullmatch(r"[+-]?\d+", parts[0].strip()):
                raise ValueError(f"class id is not an integer token: {parts[0]!r}")
            cid = int(parts[0])
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

        if cid < 0 or cid > MAX_CANONICAL_CLASS_ID:
            issues.append(
                ValidationIssue(
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Satır {line_no}: Geçersiz sınıf kimliği {cid} (beklenen 0..{MAX_CANONICAL_CLASS_ID})",
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid},
                )
            )
            continue

        # A line that fails any error-severity check is NOT added to the
        # returned boxes (F-0852); previously it was appended unconditionally.
        line_has_error = False

        # Zero-area / micro-noise check (sub-minimal sizes surface here as
        # micro_noise so size_bounds counters stay accurate -- see F-0939).
        is_zero, zero_msg = check_zero_area(w, h)
        if is_zero:
            itype_zero = "zero_area" if (w <= 0.0 or h <= 0.0 or not (math.isfinite(w) and math.isfinite(h))) else "micro_noise"
            issues.append(
                ValidationIssue(
                    issue_type=itype_zero,
                    severity="error",
                    message=f"Satır {line_no}: {zero_msg}",
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid, "w": w, "h": h},
                )
            )
            line_has_error = True

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
            line_has_error = True

        # Out-of-bounds check
        is_oob, oob_msgs = check_out_of_bounds(cx, cy, w, h)
        if is_oob:
            oob_severity = "error" if strict_bounds else "warning"
            issues.append(
                ValidationIssue(
                    issue_type="out_of_bounds",
                    severity=oob_severity,
                    message=f"Satır {line_no}: " + "; ".join(oob_msgs),
                    file_path=str(path),
                    line_number=line_no,
                    details={"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h},
                )
            )
            if oob_severity == "error":
                line_has_error = True

        if line_has_error:
            continue

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
    - It has weak boxes and is not explicitly decided
    - Or payload has `done: None` with weak boxes
    - Or payload has `requeued: True` and trusted is empty
    - Or it carries trusted boxes but was never human-approved (done is None
      and no human_approved flag) -- e.g. a JSON reconstructed from an orphan
      TXT (F-0800): it must not sail through the gate as "approved"
    - Or it is corrupt / unreadable / not a JSON object: fail closed, the
      gate must not open on a file whose decision cannot be proven (F-0801)

    Decided (not pending): done in ("ok", "no", "rejected", True, False).
    done=False / "rejected" is a decision, exactly as the sync code treats
    it -- the gate must not count a rejected item as pending (F-0853).
    A human_approved=True item with no weak boxes is approved even when
    done is None.
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
                # Corrupt / unreadable review JSON: cannot prove approval.
                pending_count += 1
                continue
            if not isinstance(data, dict):
                pending_count += 1
                continue
            done_val = data.get("done")
            if done_val in ("ok", "no", "rejected", True, False):
                continue
            weak = data.get("weak") or []
            trusted = data.get("trusted") or []
            requeued = bool(data.get("requeued"))
            if weak:
                pending_count += 1
            elif requeued and not trusted:
                pending_count += 1
            elif trusted and data.get("human_approved") is not True:
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

    # "test" is a real split -- scanning train+val for it silently was wrong
    # (F-1039). An unknown split name fails loudly instead of silently
    # falling back to train+val.
    if split is None:
        splits_to_check = ["train", "val"]
    elif split in ("train", "val", "test"):
        splits_to_check = [split]
    else:
        raise ValueError(f"Bilinmeyen split: {split!r} (beklenen: 'train', 'val', 'test' veya None)")
    labels_root = root / "labels"

    for s in splits_to_check:
        split_dir = labels_root / s
        if not split_dir.is_dir():
            # A missing split is an error, not a silent skip: otherwise an
            # empty dataset reports as "valid" (F-0798).
            report.add_issue(
                ValidationIssue(
                    issue_type="missing_split",
                    severity="error",
                    message=f"Split dizini bulunamadı: {split_dir}",
                    file_path=str(split_dir),
                )
            )
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
                    # Still bump counts via the same tally add_issue() uses,
                    # so the capped path can never drift from the uncapped one
                    # (the old code duplicated the whole counter chain here
                    # by hand -- see F-0935 for the size_bounds miss it had).
                    report._tally_issue(issue)

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

    # 3. QA Validation -- train AND val (when a val split exists). The old code
    # only checked train and never checked the val split or JSON<->TXT sync
    # (F-0936).
    report = validate_dataset(root, split="train", check_pending=False)
    if not report.valid:
        return False, f"Eğitim engellendi: QA doğrulaması başarısız ({report.summary})."

    val_labels = root / "labels" / "val"
    if val_labels.is_dir():
        val_report = validate_dataset(root, split="val", check_pending=False)
        if not val_report.valid:
            return False, f"Eğitim engellendi: val split QA doğrulaması başarısız ({val_report.summary})."

    # JSON<->TXT sync is checked only when a review queue exists at all:
    # a dataset with labels but no review/ directory at all is outside the
    # review workflow (its pending gate is already 0) and must not be
    # blocked for lacking review JSONs for every label.
    if (root / "review").is_dir():
        sync_stats = sync_all_review_txt_pairs(root, auto_repair=False)
        if sync_stats.get("failed", 0) > 0:
            return (
                False,
                f"Eğitim engellendi: {sync_stats['failed']} review JSON/TXT çifti senkron değil. "
                "Senkronizasyon tamamlanmadan eğitim başlatılamaz.",
            )

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
    wrote_anything = False

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
            wrote_anything = True
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
            wrote_anything = True
        except Exception as exc:
            logger.warning("Failed to write stale manifest: %s", exc)

    # Only claim success when something was actually written (F-0847):
    # with no state file and no models dir the old code still logged
    # "Model marked stale" although nothing was marked.
    if wrote_anything:
        logger.info("Model marked stale: %s", clean_reason)
    else:
        logger.warning(
            "Model stale işaretlenemedi (hiçbir hedef yazılamadı): %s", clean_reason
        )


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
            if not isinstance(data, dict):
                raise ValueError("stale manifest is not a JSON object")
            if data.get("model_stale") is True:
                return True, str(data.get("reason", "Model işaretlendi: stale"))
        except Exception as exc:
            # A corrupt stale manifest must NOT read as "model güncel" --
            # fail closed: treat it as stale (F-0848).
            logger.warning("Bozuk stale manifest (%s): %s", stale_file, exc)
            return True, f"Stale manifest bozuk, model stale sayılıyor: {stale_file}"

    ds_root = Path(dataset_dir) if dataset_dir else Path("dataset")
    state_file = ds_root / PIPELINE_STATE_NAME
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("pipeline state is not a JSON object")
            if state.get("model_stale") is True:
                return True, str(state.get("stale_reason", "Labels modified"))
        except Exception as exc:
            logger.warning("Bozuk pipeline state (%s): %s", state_file, exc)
            return True, f"Pipeline state bozuk, model stale sayılıyor: {state_file}"

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
        # fsync the temp file before replace, and the directory after it, so
        # a crash/power loss cannot leave a torn or vanished file (F-1040).
        with open(tmp_path, "w", encoding=encoding) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
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


def _backup_before_repair(path: Union[str, Path]) -> Optional[Path]:
    """Write a single-generation ``.bak`` copy before a repair overwrite.

    P0-1 (K-10): auto-repair writes are destructive by design (a TXT can
    be emptied, a JSON rebuilt), so the pre-repair file is kept as
    ``foo.txt`` -> ``foo.txt.bak`` — same convention as
    ``dataset_factory.label_trust.write_backup`` (P0-2). Best-effort:
    if the backup cannot be written, a warning is logged and the repair
    still proceeds (the write itself stays atomic via tmp+rename).
    Returns the backup path, or None when there was nothing to back up
    (file missing) or the copy failed.
    """
    p = Path(path)
    if not p.is_file():
        return None
    bak = p.with_name(p.name + ".bak")
    try:
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(p), str(bak))
    except OSError as exc:
        logger.warning("Yedek yazılamadı (%s -> %s): %s", p, bak, exc)
        return None
    return bak


def _extract_json_array_at(text: str, bracket_idx: int) -> Optional[str]:
    """Return the balanced ``[...]`` substring starting at ``bracket_idx``.

    String/escape aware, so brackets inside quoted strings do not end
    the scan. Returns None when the array never closes (truncated text).
    """
    depth = 0
    in_str = False
    escaped = False
    for i in range(bracket_idx, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[bracket_idx:i + 1]
    return None


def _salvage_corrupt_review_json(raw_text: str) -> Dict[str, Any]:
    """Best-effort salvage of a corrupt review JSON's weak list + meta.

    P0-1 (K-10): a corrupt JSON used to be rebuilt from its TXT with
    ``weak=[]`` and ``done=None`` — the approval flags and the whole
    weak list were silently deleted. This recovers what the raw text
    still carries, without ever inventing data:

    - ``"weak"`` array: balanced-bracket scan + ``json.loads``; only
      dict entries survive (sanitization happens at the call site).
    - ``"human_approved"`` / ``"approved"`` / ``"done"`` flags: last
      occurrence wins (same rule as
      ``atomic_review_sync.extract_salvaged_flags``).
    - Other top-level scalar fields (e.g. ``source``, ``notes``):
      depth-1 scan of the outermost object; nested objects/arrays
      (``trusted``/``boxes``/...) are NOT recovered here — ``trusted``
      comes from the valid TXT in Case B.

    Returns a dict with only the keys actually found (empty when the
    text carries nothing salvageable).
    """
    salvaged: Dict[str, Any] = {}
    if not raw_text:
        return salvaged

    # --- Flags (last occurrence wins) ---
    for key in ("human_approved", "approved"):
        matches = list(
            re.finditer(
                r'"' + key + r'"\s*:\s*(true|false)',
                raw_text,
                flags=re.IGNORECASE,
            )
        )
        if matches:
            salvaged[key] = matches[-1].group(1).lower() == "true"
    done_matches = list(
        re.finditer(
            r'"done"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null)',
            raw_text,
            flags=re.IGNORECASE,
        )
    )
    if done_matches:
        try:
            salvaged["done"] = json.loads(done_matches[-1].group(1))
        except (ValueError, TypeError):
            pass

    # --- "weak" array (balanced-bracket scan, then json.loads) ---
    weak_match = re.search(r'"weak"\s*:', raw_text)
    if weak_match:
        bracket_idx = raw_text.find("[", weak_match.end())
        if bracket_idx != -1:
            raw_array = _extract_json_array_at(raw_text, bracket_idx)
            if raw_array is not None:
                try:
                    parsed = json.loads(raw_array)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, list):
                    salvaged["weak"] = parsed

    # --- Other top-level scalar meta (depth-1 scan of the outermost
    # object). Nested keys (e.g. "class_id" inside a trusted box) are
    # never lifted to the top level: the walk tracks brace/bracket
    # depth and only reads "key": value pairs at depth 1; nested
    # objects/arrays are skipped via their parsed extent. ---
    _handled = {
        "stem", "split", "trusted", "weak", "boxes", "annotations",
        "done", "human_approved", "approved", "auto_repaired",
    }
    decoder = json.JSONDecoder()
    first_brace = raw_text.find("{")
    if first_brace != -1:
        depth = 0
        i = first_brace
        n = len(raw_text)
        in_str = False
        escaped = False
        while i < n:
            ch = raw_text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                if depth == 1:
                    # Top-level "key": parse the key, then the value.
                    try:
                        key, key_end = decoder.raw_decode(raw_text[i:])
                    except (ValueError, TypeError):
                        break
                    j = i + key_end
                    while j < n and raw_text[j] in " \t\r\n":
                        j += 1
                    if j >= n or raw_text[j] != ":":
                        i = j if j < n else n
                        continue
                    j += 1
                    while j < n and raw_text[j] in " \t\r\n":
                        j += 1
                    try:
                        value, val_end = decoder.raw_decode(raw_text[j:])
                    except (ValueError, TypeError):
                        break
                    if (
                        isinstance(key, str)
                        and key not in _handled
                        and not isinstance(value, (dict, list))
                    ):
                        salvaged.setdefault(key, value)
                    i = j + val_end
                    continue
                in_str = True
                i += 1
                continue
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth <= 0:
                    break
            i += 1
    return salvaged


def verify_and_sync_review_txt_pair(
    json_path: Union[str, Path],
    txt_path: Union[str, Path],
    auto_repair: bool = False,
) -> bool:
    """Verify consistency between review JSON and labels TXT pair and optionally auto-repair.

    P0-1 (K-10): ``auto_repair`` defaults to False — a silent repair
    could empty the TXT of a still-pending frame (its JSON holds no
    trusted boxes yet) or rebuild a corrupt JSON from the TXT while
    deleting its approval flags and weak list. Callers that really
    want a repair must opt in explicitly; every repair write is then
    backed up (``.bak`` before overwrite) on top of the atomic
    tmp+rename write.
    
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
            # done="ok" approves the trusted boxes only. Weak (unapproved)
            # boxes must never reach the training TXT, and must never be
            # promoted into "trusted" (F-0799).
            raw_cands = list(trusted_list)
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
                # S-6303: exactly 5 tokens -- the old `>= 5` silently
                # accepted extra-token lines and parsed only the first 5.
                if len(parts) == 5:
                    try:
                        # Class id must be a plain integer token, same as
                        # validate_yolo_file (F-0851). The old int(float(...))
                        # silently truncated "1.9" to class 1, so a corrupt
                        # line passed validate_label_box and could be
                        # double-counted in the sync (S-2101).
                        if not re.fullmatch(r"[+-]?\d+", parts[0].strip()):
                            raise ValueError(f"class id is not an integer token: {parts[0]!r}")
                        cid = int(parts[0])
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
        _backup_before_repair(txt_p)
        atomic_write_text(txt_p, txt_str)

        # Also sanitize json_p if it contained dirty/unfiltered boxes.
        # A rejected item (done="no"/"rejected"/False) is EXCLUDED here: its
        # json_boxes is empty by decision, not by sanitization, and writing
        # trusted=[] would permanently delete the reviewed boxes from the
        # JSON record (F-0937). The TXT is still emptied above -- a rejected
        # frame trains on zero boxes -- but the JSON keeps its boxes.
        if json_data is not None and "trusted" in json_data and done_val not in ("no", "rejected", False):
            if len(json_data.get("trusted", [])) != len(json_boxes):
                json_data["trusted"] = json_boxes
                # Boxes now in "trusted" must not linger in "weak" as well
                # (F-0938): drop weak entries duplicated into trusted.
                weak_list = json_data.get("weak")
                if isinstance(weak_list, list) and weak_list:
                    # P1-13 (K-33): weak/trusted cakisma temizligi tek tanim
                    # ``same_box()`` (IoU >= 0.45 + merkez <= 0.05) kullanir;
                    # eski yerel ``_same_box`` (koordinat basina 1e-6) kaldirildi.
                    # Eksik geometrili weak girdi eslesmez, listede kalir.
                    json_data["weak"] = [
                        b
                        for b in weak_list
                        if not any(same_box(b, jb) for jb in json_boxes)
                    ]
                _backup_before_repair(json_p)
                atomic_write_json(json_p, json_data)

        logger.info("Atomic sync: repaired %s from %s (%d boxes)", txt_p.name, json_p.name, len(json_boxes))
        return True

    # Case B: JSON is missing or corrupt, but TXT is valid
    if txt_exists and txt_valid:
        stem = txt_p.stem
        parent_name = txt_p.parent.name
        split = parent_name if parent_name in ("train", "val", "test") else "train"
        # P0-1 (K-10): a corrupt JSON used to be rebuilt as
        # trusted=txt_boxes / weak=[] / done=None — its approval flags
        # and weak list were silently deleted. Salvage what the raw
        # text still carries (weak list, flags, scalar meta) and keep
        # it in the rebuilt payload instead.
        salvaged: Dict[str, Any] = {}
        salvaged_weak: List[Dict[str, Any]] = []
        if json_exists:
            try:
                raw_corrupt = json_p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                raw_corrupt = ""
            salvaged = _salvage_corrupt_review_json(raw_corrupt)
            raw_weak = salvaged.pop("weak", [])
            if isinstance(raw_weak, list):
                weak_cands = [b for b in raw_weak if isinstance(b, dict)]
                salvaged_weak, _ = sanitize_boxes(
                    weak_cands, drop_invalid_size=True, auto_fix_coords=True
                )
                # Boxes now in "trusted" must not linger in "weak" as
                # well (F-0938, same rule as Case A).
                salvaged_weak = [
                    b
                    for b in salvaged_weak
                    if not any(same_box(b, tb) for tb in txt_boxes)
                ]
        repaired_payload: Dict[str, Any] = {
            "stem": stem,
            "split": split,
            "trusted": txt_boxes,
            "weak": salvaged_weak,
            # done=None unless the corrupt JSON actually carried an
            # approval state: nobody approved these boxes yet (F-0800),
            # but a salvaged done/human_approved must not be deleted.
            "done": salvaged.get("done"),
            "auto_repaired": True,
        }
        if "human_approved" in salvaged:
            repaired_payload["human_approved"] = salvaged["human_approved"]
        if "approved" in salvaged:
            repaired_payload["approved"] = salvaged["approved"]
        for _meta_key, _meta_val in salvaged.items():
            if _meta_key not in repaired_payload:
                repaired_payload[_meta_key] = _meta_val
        _backup_before_repair(json_p)
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
    auto_repair: bool = False,
) -> Dict[str, Any]:
    """Scan and verify/synchronize all review JSON and label TXT pairs across dataset splits.

    P0-1 (K-10): ``auto_repair`` defaults to False (verify-only sweep).
    """
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

    # P1-7 (K-20): mixed-unit box — only the pixel coordinate is divided;
    # the normalized cx must survive untouched.
    fixed_mixed = fix_out_of_bounds(0.5, 360.0, 100.0, 0.05, image_width=1280.0, image_height=720.0)
    assert fixed_mixed is not None
    assert abs(fixed_mixed[0] - 0.5) < 1e-4 and abs(fixed_mixed[1] - 0.5) < 1e-4, (
        f"Mixed-unit box collapsed: {fixed_mixed}"
    )

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

        # --- Test 9: sweep-line overlap detection == brute-force oracle ---
        import random as _random

        def _brute_overlaps(boxes, thr=0.80, same_class_only=True):
            out = []
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if same_class_only and boxes[a]["class_id"] != boxes[b]["class_id"]:
                        continue
                    iou = compute_box_iou(
                        (boxes[a]["cx"], boxes[a]["cy"], boxes[a]["w"], boxes[a]["h"]),
                        (boxes[b]["cx"], boxes[b]["cy"], boxes[b]["w"], boxes[b]["h"]),
                    )
                    if iou > thr:
                        out.append((a, b, iou))
            return out

        rng = _random.Random(20261108)
        for label, n, spread in (("sparse", 300, 1.0), ("dense", 120, 0.12)):
            rnd_boxes = [
                {
                    "class_id": rng.randrange(3),
                    "cx": 0.5 + rng.uniform(-spread / 2, spread / 2),
                    "cy": 0.5 + rng.uniform(-spread / 2, spread / 2),
                    "w": rng.uniform(0.02, 0.12),
                    "h": rng.uniform(0.02, 0.12),
                }
                for _ in range(n)
            ]
            for same_only in (True, False):
                got = detect_overlapping_boxes(rnd_boxes, same_class_only=same_only)
                exp = _brute_overlaps(rnd_boxes, same_class_only=same_only)
                assert [(i, j) for i, j, _, _ in got] == [(a, b) for a, b, _ in exp], (
                    f"sweep pairs != brute force ({label}, same_class_only={same_only})"
                )
                for (_, _, giou, _), (_, _, eiou) in zip(got, exp):
                    assert abs(giou - eiou) < 1e-9
                # Malformed boxes are skipped, never crash the sweep.
                got2 = detect_overlapping_boxes(
                    rnd_boxes + [{"class_id": "x", "cx": float("nan"), "cy": 0.5, "w": 0.1, "h": 0.1}],
                    same_class_only=same_only,
                )
                assert [(i, j) for i, j, _, _ in got2] == [(a, b) for a, b, _ in exp]

        # --- Test 10: sanitize dedup -- no kept pair overlaps, nothing lost ---
        dedup_boxes = [
            {
                "class_id": rng.randrange(3),
                # In-bounds by construction (cx/cy in [0.1, 0.9], w/h <= 0.12):
                # sanitize must keep coordinates verbatim, so "kept" can be
                # matched against the input by exact coordinates below.
                "cx": rng.uniform(0.1, 0.9),
                "cy": rng.uniform(0.1, 0.9),
                "w": rng.uniform(0.02, 0.12),
                "h": rng.uniform(0.02, 0.12),
                "confidence": rng.random(),
            }
            for _ in range(300)
        ]
        # Plant exact duplicates that must collapse to one kept box.
        dedup_boxes.append({"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.99, "hand_label": True})
        dedup_boxes.append({"class_id": 1, "cx": 0.505, "cy": 0.505, "w": 0.1, "h": 0.1, "confidence": 0.5})
        kept, dedup_issues = sanitize_boxes(dedup_boxes)
        assert any(
            b["class_id"] == 1 and b["cx"] == 0.5 and b["cy"] == 0.5 and b.get("hand_label") is True
            for b in kept
        ), "hand-labelled planted box must survive dedup"
        assert not any(
            b["class_id"] == 1 and b["cx"] == 0.505 and b["cy"] == 0.505 for b in kept
        ), "lower-confidence duplicate must be dropped"
        for a in range(len(kept)):
            for b in range(a + 1, len(kept)):
                if kept[a]["class_id"] == kept[b]["class_id"]:
                    assert compute_box_iou(
                        (kept[a]["cx"], kept[a]["cy"], kept[a]["w"], kept[a]["h"]),
                        (kept[b]["cx"], kept[b]["cy"], kept[b]["w"], kept[b]["h"]),
                    ) <= 0.80, "kept pair still overlaps above threshold"
        # Every input box is either kept or covered by a kept duplicate.
        kept_keys = {(b["class_id"], b["cx"], b["cy"], b["w"], b["h"]) for b in kept}
        for src in dedup_boxes:
            key = (src["class_id"], src["cx"], src["cy"], src["w"], src["h"])
            if key in kept_keys:
                continue
            assert any(
                k["class_id"] == src["class_id"]
                and compute_box_iou(
                    (src["cx"], src["cy"], src["w"], src["h"]),
                    (k["cx"], k["cy"], k["w"], k["h"]),
                ) > 0.80
                for k in kept
            ), "dropped box has no kept duplicate covering it"
        # Deterministic: same input -> same output.
        kept_again, _ = sanitize_boxes(dedup_boxes)
        assert kept_again == kept

        # --- Test 11: capped report tallies == uncapped report tallies ---
        cap_tmp = tmp / "cap_test"
        (cap_tmp / "labels" / "train").mkdir(parents=True)
        # 8 files x 3 zero-area lines = 24 errors; cap at 5 recorded issues.
        for f in range(8):
            (cap_tmp / "labels" / "train" / f"f{f:02d}.txt").write_text(
                "0 0.5 0.5 0.0 0.1\n0 0.5 0.5 0.1 0.0\n0 0.5 0.5 0.002 0.1\n",
                encoding="utf-8",
            )
        capped = validate_dataset(cap_tmp, split="train", check_pending=False, max_issues_to_record=5)
        uncapped = validate_dataset(cap_tmp, split="train", check_pending=False, max_issues_to_record=1000)
        assert len(capped.issues) == 5, f"expected 5 recorded issues, got {len(capped.issues)}"
        for field in (
            "error_count", "warning_count", "zero_area_count",
            "size_bounds_count", "total_files_scanned", "total_boxes_scanned",
        ):
            assert getattr(capped, field) == getattr(uncapped, field), (
                f"capped {field}={getattr(capped, field)} != uncapped {getattr(uncapped, field)}"
            )
        assert capped.valid is False and uncapped.valid is False

    print("ALL DATASET QA, HUMAN GATE, STALE MODEL, ATOMIC SYNC, AND BOX BOUND TESTS PASSED!")


__all__ = [
    "DEFAULT_IOU_OVERLAP_THRESHOLD",
    "DEFAULT_MIN_BOX_DIM",
    "DEFAULT_MIN_BOX_AREA",
    "MIN_BOX_DIM_BOUND",
    "MAX_BOX_DIM_BOUND",
    "MIN_BOX_AREA_BOUND",
    "MAX_CANONICAL_CLASS_ID",
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

