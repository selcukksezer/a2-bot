# -*- coding: utf-8 -*-
"""
Pre-Training Quality & Hygiene Gate (`dataset_factory/dataset_hygiene_gate.py`).
==============================================================================
Provides a high-speed, sub-second pre-flight gate that validates and cleans
YOLO training datasets (`dataset/labels/{train,val}/*.txt`,
`dataset/review/{train,val}/*.json`, and `dataset/images/{train,val}/*`)
before YOLO training begins.

Enforces 4 strict zero-defect pre-training invariants:
1. 0 invalid/unknown `class_id` entries. Training ids are the canonical
   taxonomy 0..28 (`dataset_factory/class_taxonomy.py::ALL_CANONICAL_CLASSES`).
2. 0 unapproved/unverified boxes in training and validation sets (strictly enforces human approval
   or high-confidence verified trust). A missing review JSON does not make
   a well-formed exactly-5-field TXT line unverified: that line is a hand
   label, matching the freeze / approved-pool rule.
3. 0 zero-area, unnormalized, or corrupted boxes.
4. 0 empty/corrupted labels with mismatched images (orphaned labels/images).

Auto-cleans/purges toxic, out-of-bounds, unverified heuristic, and corrupted
annotations in-place, synchronizes disk `.txt` and review `.json` files,
and returns a structured status report:
    {"clean": bool, "purged_boxes": int, "fixed_files": int, "details": list}

Exclusive Owner: Worker L20 (`dataset_factory/dataset_hygiene_gate.py`)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    MATERIAL_3D_CLASSES,
    MINIMAP_SYMBOL_CLASSES,
    canonical_names,
    is_valid_canonical_class_id,
)

logger = logging.getLogger("DatasetFactory.HygieneGate")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] [HygieneGate] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ==============================================================================
# Canonical Taxonomy & Domain Definitions
# ==============================================================================

# Canonical training-set class IDs: the full taxonomy 0..28.
# Pinned to dataset_factory/class_taxonomy.py::ALL_CANONICAL_CLASSES so the
# set cannot silently drift. Consumables 22..24, kinah 25, player_character
# 26, and the F4 generics 27..28 are training classes.
CANONICAL_CLASS_IDS: frozenset[int] = ALL_CANONICAL_CLASSES

# Human-readable names for every taxonomy class (diagnostics/messages).
# P1-1: read from the authority (dataset/data.yaml via
# dataset_factory.class_taxonomy.canonical_names) — NOT a second copy of
# the table. Any data.yaml rename flows here automatically.
CANONICAL_CLASS_NAMES: Dict[int, str] = canonical_names()

# Heuristic sources that cannot enter labels/train/*.txt without human approval
UNVERIFIED_HEURISTIC_SOURCES: frozenset[str] = frozenset({
    "chromatic_salient_3d_resource",
    "visual_ground_path_segmenter",
    "visual_obstacle_segmenter",
    "hud_skill_slot",
    "hud_player_hp_bar",
    "hud_player_mp_bar",
    "dataset_requeue",
    "legacy_auto",
    "center_anchor_fallback",
    "gather_material_hint",
    "omni_synthetic",
    "omni_auto",
    "unverified_auto",
})

# Verified ground truth sources allowed in training if confidence >= threshold
VERIFIED_GROUND_TRUTH_SOURCES: frozenset[str] = frozenset({
    "hand_label",
    "human_approved",
    "approved",
    "calibration_wizard",
    "vision_3d_crest",
    "material_scanner_minimap_sprout",
    "minimap_sprout_template",
    "vision_f_gather_prompt",
    "vision_e_gather_icon",
    "vision_floating_icon",
    "f_gather_prompt",
    "e_gather_icon",
    "template_match",
    "exemplar_match",
    "repaired_by_exemplar",
    "positive_exemplar_lock",
})

# Geometry thresholds
MIN_BOX_DIM: float = 0.002
MIN_BOX_AREA: float = 1e-6
MAX_BOX_DIM: float = 0.98
DEFAULT_EMIN_THRESHOLD: float = 0.85
# Demoted boxes are capped below the train floor (label_trust
# TRAIN_CONF_FLOOR = 0.45): a demote must really demote — the old 0.68
# cap still cleared the floor, so a demoted box re-entered the train txt
# through the learn weak path (sweep E-02).
DEMOTED_CONF_CAP: float = 0.40

# Minimap HUD bounds (normalized right-top corner)
MINIMAP_HUD_XYXY: Tuple[float, float, float, float] = (0.80, 0.00, 1.00, 0.35)
# 3D gatherable world materials — P1-1: the taxonomy set itself
# (dataset_factory/class_taxonomy.py::MATERIAL_3D_CLASSES), not a copy.
# Other files used {0..6} or {0..5, 12}; the taxonomy set is authoritative.
WORLD_RESOURCE_CLASS_IDS: frozenset[int] = MATERIAL_3D_CLASSES
# Minimap symbols allowed inside the minimap HUD — P1-1: the taxonomy
# set itself (dataset_factory/class_taxonomy.py::MINIMAP_SYMBOL_CLASSES).
# (7/8 are world materials — herb_lumina / ore_obsidian — not minimap symbols.)
MINIMAP_ALLOWED_CLASS_IDS: frozenset[int] = MINIMAP_SYMBOL_CLASSES

SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


# ==============================================================================
# Helper Verification & Geometry Functions
# ==============================================================================

def is_canonical_class(class_id: Any) -> bool:
    """Check if class_id is a strict int in the canonical taxonomy 0..28.

    Membership is ``ALL_CANONICAL_CLASSES`` (the exported id set in
    ``dataset_factory/class_taxonomy.py``). Strict: only real ints count.
    ``bool`` is rejected (``int(True) == 1`` would otherwise smuggle
    ``True`` in as class 1) and ``float``/``str`` values are rejected
    even when integral (``7.0``/``"7"`` are malformed class ids, not
    canonical ones) (F-1038).
    """
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        return False
    return class_id in CANONICAL_CLASS_IDS


def is_taxonomy_class(class_id: Any) -> bool:
    """Check if class_id is a real taxonomy class (authority, today 0..28).

    P1-1: the range comes from the authority (``dataset/data.yaml`` via
    ``dataset_factory.class_taxonomy.is_valid_canonical_class_id``), which
    also accepts digit-only strings. :func:`is_canonical_class` is the
    stricter int-only check over the same id set.
    """
    return is_valid_canonical_class_id(class_id)


def _non_training_class_reason(cid: Any) -> str:
    """Message fragment for a box purged by the canonical-id check."""
    hi = max(CANONICAL_CLASS_IDS) if CANONICAL_CLASS_IDS else -1
    if is_taxonomy_class(cid) and not is_canonical_class(cid):
        return (
            f"Class id {cid!r} is not a strict canonical integer "
            f"(training ids are 0..{hi})"
        )
    return f"Non-canonical class_id {cid}"


def is_box_in_minimap_hud(cx: float, cy: float, xyxy: Tuple[float, float, float, float] = MINIMAP_HUD_XYXY) -> bool:
    """Check if box center lies within the upper-right minimap HUD region."""
    x1, y1, x2, y2 = xyxy
    return float(x1) <= float(cx) <= float(x2) and float(y1) <= float(cy) <= float(y2)


def _box_score(box: Dict[str, Any]) -> float:
    """Best-effort confidence/score of a review box (0.0 when absent)."""
    for key in ("confidence", "score"):
        try:
            return float(box.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return 0.0


# Tolerance used when matching a label-txt line to a trusted review box
# (class + centre + size must all agree within this, F-0932).
_LABEL_MATCH_TOLERANCE: float = 0.05


def _build_trusted_box_index(
    trusted_boxes: Sequence[Dict[str, Any]],
) -> Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, Any], float, float, float, float]]]:
    """Bucket index over trusted review boxes for label-line matching.

    Keyed by ``(class_id, floor(cx / tol), floor(cy / tol))`` so a lookup
    only inspects the 3x3 neighbourhood of the query cell instead of
    scanning every trusted box per label line (O(n*m) -> ~O(n)).
    A box within tolerance (< tol on both centre axes) always lands in
    the same or an adjacent cell, because the cell width equals the
    tolerance. Malformed entries are skipped, mirroring the old linear
    scan's ``try/except: continue``. Each entry keeps its original list
    position so first-match-wins order is preserved.
    """
    index: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, Any], float, float, float, float]]] = {}
    for order, t_box in enumerate(trusted_boxes):
        try:
            cid = int(t_box.get("class_id", -1))
            cx = float(t_box.get("cx", 0))
            cy = float(t_box.get("cy", 0))
            w = float(t_box.get("w", 0))
            h = float(t_box.get("h", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        key = (cid, math.floor(cx / _LABEL_MATCH_TOLERANCE), math.floor(cy / _LABEL_MATCH_TOLERANCE))
        index.setdefault(key, []).append((order, t_box, cx, cy, w, h))
    return index


def _find_matching_trusted_box(
    index: Dict[Tuple[int, int, int], List[Tuple[int, Dict[str, Any], float, float, float, float]]],
    cid: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> Optional[Dict[str, Any]]:
    """First (lowest original order) trusted box matching class + centre + size.

    Same match rule as the old linear scan: ``|d_cx|, |d_cy|, |d_w|, |d_h|``
    all strictly below ``_LABEL_MATCH_TOLERANCE``. Returns ``None`` when
    nothing matches.
    """
    tol = _LABEL_MATCH_TOLERANCE
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
        return None
    cell_x = math.floor(cx / tol)
    cell_y = math.floor(cy / tol)
    best: Optional[Tuple[int, Dict[str, Any]]] = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for order, t_box, tcx, tcy, tw, th in index.get((cid, cell_x + dx, cell_y + dy), ()):
                if (
                    abs(tcx - cx) < tol
                    and abs(tcy - cy) < tol
                    and abs(tw - w) < tol
                    and abs(th - h) < tol
                ):
                    if best is None or order < best[0]:
                        best = (order, t_box)
    return best[1] if best is not None else None


def _validate_review_box_shape(
    box: Any, split: str, stem: str, list_name: str
) -> Tuple[str, str, bool]:
    """Shared shape/canonical/geometry check for trusted and weak review boxes.

    The trusted and weak sanitization loops ran the same ~30-line block
    twice; this helper is the single copy. It mutates ``box`` in place
    when clamping applies. Returns ``(outcome, detail, clamped_changed)``
    where outcome is ``"ok"``, ``"purged_corrupted"`` or
    ``"purged_invalid_cid"``, ``detail`` is the ``[{split}:{stem}] ...``
    message (``""`` when the old code appended no detail, e.g. non-dict /
    unparsable boxes), and ``clamped_changed`` reports whether clamping
    moved a coordinate (the caller must mark the file modified then).
    """
    if not isinstance(box, dict):
        return "purged_corrupted", "", False
    cid = box.get("class_id")
    if not is_canonical_class(cid):
        return "purged_invalid_cid", f"[{split}:{stem}] Review {list_name} purged {_non_training_class_reason(cid)}", False
    try:
        cx = float(box.get("cx", 0))
        cy = float(box.get("cy", 0))
        w = float(box.get("w", 0))
        h = float(box.get("h", 0))
    except (TypeError, ValueError):
        return "purged_corrupted", "", False
    valid_geom, clamped, geom_reason = validate_and_clamp_box(cx, cy, w, h)
    if not valid_geom:
        return "purged_corrupted", f"[{split}:{stem}] Review {list_name} purged invalid geom ({geom_reason})", False
    if clamped:
        cx_c, cy_c, w_c, h_c = clamped
        if abs(cx_c - cx) > 1e-5 or abs(cy_c - cy) > 1e-5 or abs(w_c - w) > 1e-5 or abs(h_c - h) > 1e-5:
            box["cx"], box["cy"], box["w"], box["h"] = cx_c, cy_c, w_c, h_c
            return "ok", "", True
    return "ok", "", False


def is_hand_or_human_approved(box: Dict[str, Any], emin_threshold: float = DEFAULT_EMIN_THRESHOLD) -> bool:
    """Return True if box has explicit human approval or hand-labeled origin.

    Seed-prompt hints (canonical class 13 ``skill_ui_prompt`` + ``f``/``e``
    key) are NOT bare approval: they only count when the box also carries
    a confidence/score at or above ``emin_threshold`` (F-0844). A key/source
    substring match with no score never approves a box. (Sweep Y1: the old
    7/8 mapping was wrong — 7 is ``herb_lumina``, 8 is ``ore_obsidian``,
    materials, not prompts.)
    """
    if not isinstance(box, dict):
        return False
    if box.get("hand_label") is True or box.get("human_approved") is True or box.get("approved") is True:
        return True
    if box.get("done") in ("ok", True):
        return True
    source = str(box.get("source", "")).strip().lower()
    if any(tok in source for tok in ("hand", "manual", "human")):
        return True
    # Seed prompts — approval only with a high-confidence score.
    try:
        cid = int(box.get("class_id"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    key = str(box.get("gather_key", box.get("key", ""))).strip().lower()
    if cid == 13 and (
        key in ("f", "e")
        or "f_gather" in source
        or "e_gather" in source
        or "gather_prompt" in source
        or "f_template" in source
        or "e_template" in source
        or "e_floating" in source
        or "floating" in source
    ):
        return _box_score(box) >= float(emin_threshold)
    return False


def is_stem_incompatible_with_class(stem: str, class_id: int, source: str = "") -> bool:
    """
    Detect heuristic contamination on known capture stems:
    e.g. boot_*_aria_* or gather_* stems containing path_ground (15),
    skill_ui_prompt (13), player_hp_bar (16..17), or obstacle (14)
    injected by indiscriminate full-frame heuristic segmenters.
    """
    s = str(stem or "").lower()
    src = str(source or "").lower()

    # If it's a known resource recording stem
    is_resource_stem = any(tok in s for tok in ("boot_", "gather_", "_aria_", "_mela_", "_targena_", "_orichalcum_"))
    if is_resource_stem:
        # Heuristic ground path or skill bar on resource harvesting captures is toxic clutter
        if class_id in (13, 14, 15, 16, 17, 18, 19, 20, 21):
            if any(uh in src for uh in ("visual_ground_path_segmenter", "visual_obstacle_segmenter", "hud_skill_slot", "hud_player_hp_bar", "hud_player_mp_bar")):
                return True
    return False


def validate_and_clamp_box(
    cx: float, cy: float, w: float, h: float
) -> Tuple[bool, Optional[Tuple[float, float, float, float]], str]:
    """
    Validates box geometry:
    - Rejects NaN/Inf
    - Rejects zero-area (w <= 0 or h <= 0 or w < MIN_BOX_DIM or h < MIN_BOX_DIM or w*h < MIN_BOX_AREA)
    - Rejects extreme full-screen artifacts (w >= MAX_BOX_DIM and h >= MAX_BOX_DIM)
    - Fixes minor float out-of-bounds rounding [-0.02, 1.02] by clamping
    - Rejects wild out-of-bounds or pixel coordinates
    """
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
        return False, None, "Coordinates contain NaN or Inf"

    if w <= 0.0 or h <= 0.0 or w < MIN_BOX_DIM or h < MIN_BOX_DIM:
        return False, None, f"Zero or sub-minimal dimension (w={w:.6f}, h={h:.6f})"

    area = w * h
    if area < MIN_BOX_AREA:
        return False, None, f"Zero or sub-minimal area ({area:.8f} < {MIN_BOX_AREA})"

    if w >= MAX_BOX_DIM and h >= MAX_BOX_DIM:
        return False, None, f"Spurious full-screen box (w={w:.4f}, h={h:.4f})"

    # Check for unnormalized pixel coords (e.g. w > 1.2 or cx > 1.2)
    if cx > 1.2 or cy > 1.2 or w > 1.2 or h > 1.2 or cx < -0.2 or cy < -0.2:
        return False, None, f"Wildly out-of-bounds or pixel coordinates (cx={cx}, cy={cy}, w={w}, h={h})"

    # Clamp slightly shifted coordinates within [0.0001, 0.9999]
    cx_clamped = min(max(0.0001, cx), 0.9999)
    cy_clamped = min(max(0.0001, cy), 0.9999)
    w_clamped = min(max(MIN_BOX_DIM, w), 0.9999)
    h_clamped = min(max(MIN_BOX_DIM, h), 0.9999)

    return True, (cx_clamped, cy_clamped, w_clamped, h_clamped), ""


def is_training_box_verified(
    box: Dict[str, Any],
    stem: str = "",
    emin_threshold: float = DEFAULT_EMIN_THRESHOLD,
    *,
    jsonless_hand_label: bool = False,
) -> Tuple[bool, str]:
    """
    Evaluates whether a box is authorized to appear in training labels:
    - 0 unapproved/unverified boxes in training set (strictly enforces human
      approval or high-confidence verified trust).

    ``jsonless_hand_label`` is set by the label walker only when the review
    JSON is missing and the TXT line is an exactly-5-field box. That is a
    hand label (freeze / approved_pool). It must be kept when the class id
    is canonical and the geometry is valid, and must not be purged as
    ``unverified_txt``.
    """
    if not isinstance(box, dict):
        return False, "Box is not a dictionary"

    # 1. Canonical class FIRST (S-2304): human approval must never admit a
    # non-canonical class. Previously the approval check came first, so a
    # box with class_id=99 and human_approved=True was admitted as
    # "human_approved" even though it can never belong in training labels.
    cid = box.get("class_id")
    if not is_canonical_class(cid):
        return False, _non_training_class_reason(cid)
    cid = int(cid)

    # 1b. JSON-less exactly-5-field TXT line: hand label, not unverified_txt.
    # Geometry is re-checked here so a direct caller cannot use the flag to
    # admit NaN, zero-area, or out-of-bounds boxes.
    if jsonless_hand_label:
        try:
            cx = float(box["cx"])
            cy = float(box["cy"])
            w = float(box["w"])
            h = float(box["h"])
        except (KeyError, TypeError, ValueError):
            return False, "JSON-less TXT line is not a well-formed box"
        valid_geom, _, geom_reason = validate_and_clamp_box(cx, cy, w, h)
        if not valid_geom:
            return False, geom_reason
        return True, "hand_label_txt"

    # 2. Human approval or hand label always approved (canonical classes only)
    if is_hand_or_human_approved(box, emin_threshold=emin_threshold):
        return True, "human_approved"

    # 3. Check weak flag or untrusted flag
    if box.get("weak") is True or box.get("trusted") is False:
        return False, "Marked weak/untrusted in review metadata"

    # 4. Check confidence
    try:
        conf = float(box.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    if conf < float(emin_threshold):
        return False, f"Confidence {conf:.3f} below emin threshold {emin_threshold:.2f}"

    source = str(box.get("source", "")).strip().lower()

    # 5. Check unverified heuristic sources
    if any(uh in source for uh in UNVERIFIED_HEURISTIC_SOURCES):
        return False, f"Unverified heuristic source '{source}' without human approval"

    if source.startswith("omni_") or box.get("omni_synthetic") is True:
        return False, "Synthetic omni box without human approval"

    # 6. Check stem contamination
    if is_stem_incompatible_with_class(stem, cid, source):
        return False, f"Incompatible class {cid} ({CANONICAL_CLASS_NAMES.get(cid)}) on stem '{stem}'"

    # 7. Check minimap HUD false positive: only minimap symbols belong in
    # the minimap HUD. A 3D world resource centred there is a false
    # positive; minimap symbols (MINIMAP_ALLOWED_CLASS_IDS) and genuine HUD
    # elements (skill/vitals bars) are expected there and pass.
    try:
        cx = float(box.get("cx", 0.5))
        cy = float(box.get("cy", 0.5))
        if is_box_in_minimap_hud(cx, cy):
            if cid in MINIMAP_ALLOWED_CLASS_IDS:
                pass
            elif cid in WORLD_RESOURCE_CLASS_IDS:
                return False, f"World resource class {cid} falsely detected in minimap HUD zone"
    except (TypeError, ValueError):
        pass

    # 8. Must have trusted=True, confidence >= threshold, AND a verified
    # ground-truth source. trusted + high confidence alone never suffices —
    # otherwise the VERIFIED_GROUND_TRUTH_SOURCES allowlist is dead code
    # and any source passes (F-0794).
    if box.get("trusted") is True and conf >= float(emin_threshold):
        if source in VERIFIED_GROUND_TRUTH_SOURCES:
            return True, "verified_high_confidence"
        return False, f"Source '{source}' is not in the verified ground-truth allowlist"

    return False, "Lacks verified trust evidence"


# ==============================================================================
# Safe JSON File Handler with Auto-Repair
# ==============================================================================

def _load_review_json_safe(path_str: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Loads review JSON, with automatic recovery/repair for corrupted/malformed
    JSON files (such as duplicated braces or concatenated trailing blocks).
    Returns (data, was_repaired).
    """
    try:
        with open(path_str, "rb") as f:
            raw_bytes = f.read()
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path_str, exc)
        return None, False

    if not raw_bytes:
        return None, False

    try:
        data = json.loads(raw_bytes)
        if isinstance(data, dict):
            return data, False
    except Exception:
        pass

    content = raw_bytes.decode("utf-8", errors="ignore").strip()

    # Fast path: the C JSON decoder parses the first complete object and
    # ignores trailing garbage (duplicated braces, concatenated blocks)
    # instead of the O(n) Python char-by-char scan below. It is only tried
    # when the text before the first '{' holds no '"' or '\\' — with such
    # a prefix the char scanner's string/escape tracking could disagree
    # about where the first object starts, so those cases still use the
    # scanner. Any decode failure falls back to the scanner unchanged.
    first_brace = content.find("{")
    if first_brace != -1:
        prefix = content[:first_brace]
        if '"' not in prefix and "\\" not in prefix:
            try:
                data, _end = json.JSONDecoder().raw_decode(content[first_brace:])
                if isinstance(data, dict):
                    logger.info("Successfully repaired malformed JSON in %s", os.path.basename(path_str))
                    return data, True
            except Exception:
                pass

    # Recovery: Extract first balanced outermost JSON object '{ ... }'
    brace_depth = 0
    start_idx = -1
    end_idx = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if brace_depth == 0:
                start_idx = idx
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx != -1:
                end_idx = idx + 1
                break

    if start_idx != -1 and end_idx != -1:
        extracted = content[start_idx:end_idx]
        try:
            data = json.loads(extracted)
            if isinstance(data, dict):
                logger.info("Successfully repaired malformed JSON in %s", os.path.basename(path_str))
                return data, True
        except Exception:
            pass

    return None, False


def _file_sha256(path_str: str) -> str:
    """Dosyanın SHA-256 özeti; okunamazsa boş döner."""
    try:
        digest = hashlib.sha256()
        with open(path_str, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _atomic_copy_file(src: str, dst: str) -> bool:
    """P2-6 (K-05/K-19): tmp + flush + fsync + os.replace ile atomik kopya.

    Kopya boyut + SHA-256 ile doğrulanır; başarısızlıkta hedefe
    dokunulmaz ve tmp temizlenir.
    """
    tmp_path = f"{dst}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with open(src, "rb") as f_src, open(tmp_path, "wb") as f_dst:
            for chunk in iter(lambda: f_src.read(1024 * 1024), b""):
                f_dst.write(chunk)
            f_dst.flush()
            try:
                os.fsync(f_dst.fileno())
            except OSError:
                pass
        os.replace(tmp_path, dst)
        try:
            if os.path.getsize(src) != os.path.getsize(dst):
                return False
        except OSError:
            return False
        src_sig = _file_sha256(src)
        return bool(src_sig) and src_sig == _file_sha256(dst)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def _backup_file_before_destructive_write(path_str: str) -> Optional[str]:
    """P2-6 (K-05): yıkıcı yazım/silme ÖNCESİ dosyanın zaman damgalı,
    atomik, doğrulanmış yedeğini aynı dizine alır.

    Yedek adı ``<orijinal>.hygiene-bak-<YYYYMMDD_HHMMSS_ffffff>`` —
    ``.json``/``.txt`` ile bitmez, gate taramalarının
    (``endswith(".json")`` / ``endswith(".txt")``) yeniden okumasına
    girmez. Döner: yedek yolu; kaynak yoksa None (yedeklenecek bir
    şey yok, hata değil); kopya doğrulanamazsa None (çağıran yıkıcı
    adıma GİRMEZ — fail-closed).
    """
    if not os.path.isfile(path_str):
        return None
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    dst = f"{path_str}.hygiene-bak-{ts}"
    if _atomic_copy_file(path_str, dst):
        return dst
    logger.warning("Yedek alınamadı, yıkıcı işlem atlanıyor: %s", path_str)
    return None


def _atomic_write_json(path_str: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON payload with UTF-8 encoding.

    P2-6: tmp artık fsync'lenir — eskiden fsync yoktu, çökme anında
    replace edilmiş yarım/0 bayt dosya kalıcı kayıp demekti.
    """
    temp_path = f"{path_str}.tmp_{os.getpid()}_{time.time_ns()}"
    payload_str = json.dumps(data, indent=2, ensure_ascii=False)
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(payload_str)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(temp_path, path_str)


def _atomic_write_lines(path_str: str, lines: Sequence[str]) -> None:
    """Atomically write text lines to path with UTF-8 encoding.

    P2-6: tmp artık fsync'lenir (yukarıdaki gerekçe aynen geçerli).
    """
    temp_path = f"{path_str}.tmp_{os.getpid()}_{time.time_ns()}"
    content = ("\n".join(lines) + "\n") if lines else ""
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(temp_path, path_str)


# ==============================================================================
# High-Speed Per-Split Worker
# ==============================================================================

def _process_single_review_file(
    args: Tuple[str, str, str, bool, float, bool]
) -> Dict[str, Any]:
    """Worker function for parallel review JSON validation and sanitization."""
    path_str, stem, split, auto_clean, emin_threshold, enforce_approval = args
    details: List[str] = []
    purged_boxes = 0
    fixed_files = 0
    invalid_cids = 0
    corrupted_boxes = 0
    unapproved_boxes = 0
    demoted_boxes = 0
    unresolved_issues = 0
    backups_created = 0

    data, was_repaired = _load_review_json_safe(path_str)
    if data is None:
        if auto_clean:
            # P2-6 (K-05): kurtarılamaz dosya artık yedeksiz silinmiyor.
            # Yedek alınamazsa silme yapılmaz — dosya yerinde kalır ve
            # sorun "çözülmemiş" sayılır (gate clean=False döner).
            backup_path = _backup_file_before_destructive_write(path_str)
            if backup_path is None:
                unresolved_issues += 1
                details.append(f"[{split}:{stem}] Unrecoverable corrupt JSON kept (backup FAILED, not deleted)")
            else:
                backups_created += 1
                try:
                    os.unlink(path_str)
                    fixed_files += 1
                    details.append(f"[{split}:{stem}] Deleted unrecoverable corrupt JSON file (backed up to {os.path.basename(backup_path)})")
                except OSError as exc:
                    unresolved_issues += 1
                    details.append(f"[{split}:{stem}] Could not delete unrecoverable corrupt JSON file: {exc}")
        else:
            unresolved_issues += 1
            details.append(f"[{split}:{stem}] Unrecoverable corrupt JSON file (dry-run, kept)")
        return {
            "stem": stem,
            "data": None,
            "purged_boxes": 0,
            "fixed_files": fixed_files,
            "invalid_cids": 0,
            "corrupted_boxes": 0,
            "unapproved_boxes": 0,
            "demoted_boxes": 0,
            "unresolved_issues": unresolved_issues,
            "backups_created": backups_created,
            "details": details,
        }

    modified = was_repaired
    trusted_boxes = data.get("trusted", [])
    weak_boxes = data.get("weak", [])
    if not isinstance(trusted_boxes, list):
        trusted_boxes = []
        modified = True
    if not isinstance(weak_boxes, list):
        weak_boxes = []
        modified = True

    new_trusted: List[Dict[str, Any]] = []
    new_weak: List[Dict[str, Any]] = []

    # Sanitize trusted boxes (shape/canonical/geometry via the shared
    # helper; approval/demotion is trusted-list-only logic below).
    for b in trusted_boxes:
        outcome, detail, clamped_changed = _validate_review_box_shape(b, split, stem, "trusted")
        if outcome != "ok":
            modified = True
            purged_boxes += 1
            if outcome == "purged_invalid_cid":
                invalid_cids += 1
            else:
                corrupted_boxes += 1
            if detail:
                details.append(detail)
            continue
        if clamped_changed:
            modified = True

        cid = int(b["class_id"])

        # Check approval on EVERY split (train and val alike): an unverified
        # box is toxic wherever it sits, not only in train (F-0845).
        # Weak-list boxes are quarantine by definition and are not re-checked
        # here — they never reach training labels.
        if enforce_approval:
            verified, reason = is_training_box_verified(b, stem=stem, emin_threshold=emin_threshold)
            if not verified:
                modified = True
                unapproved_boxes += 1
                details.append(f"[{split}:{stem}] Review trusted demoted unapproved box (cid={cid}): {reason}")
                # Demote to weak list with capped confidence. A demoted box
                # is NOT purged — it still exists in the dataset (weak list)
                # — so it must not inflate purged_boxes (F-0931). The cap
                # sits below the train floor (sweep E-02): a demoted box
                # must never re-enter the train txt through a weak path.
                demoted_boxes += 1
                b["trusted"] = False
                b["weak"] = True
                try:
                    conf = float(b.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                b["confidence"] = min(conf, DEMOTED_CONF_CAP)
                new_weak.append(b)
                continue

        new_trusted.append(b)

    # Sanitize weak boxes (same shared shape check as trusted; no
    # approval/demotion — weak boxes never reach training labels).
    for b in weak_boxes:
        outcome, detail, clamped_changed = _validate_review_box_shape(b, split, stem, "weak")
        if outcome != "ok":
            modified = True
            purged_boxes += 1
            if outcome == "purged_invalid_cid":
                invalid_cids += 1
            else:
                corrupted_boxes += 1
            if detail:
                details.append(detail)
            continue
        if clamped_changed:
            modified = True

        new_weak.append(b)

    data["trusted"] = new_trusted
    data["weak"] = new_weak

    if modified and auto_clean:
        # P2-6: kutu silen/budayan her yeniden yazım ÖNCESİ yedek —
        # eşik yok, silme sayısı ne olursa olsun (yarıdan az silmede
        # de yedek alınır). Yedek alınamazsa yazım yapılmaz.
        backup_path = _backup_file_before_destructive_write(path_str)
        if backup_path is None:
            unresolved_issues += 1
            details.append(f"[{split}:{stem}] Sanitized review JSON NOT written (backup FAILED)")
        else:
            backups_created += 1
            try:
                _atomic_write_json(path_str, data)
                fixed_files += 1
            except OSError as exc:
                unresolved_issues += 1
                details.append(f"[{split}:{stem}] Failed to write sanitized review JSON: {exc}")

    return {
        "stem": stem,
        "data": data,
        "purged_boxes": purged_boxes,
        "fixed_files": fixed_files,
        "invalid_cids": invalid_cids,
        "corrupted_boxes": corrupted_boxes,
        "unapproved_boxes": unapproved_boxes,
        "demoted_boxes": demoted_boxes,
        "unresolved_issues": unresolved_issues,
        "backups_created": backups_created,
        "details": details,
    }


def _process_single_label_file(
    args: Tuple[str, str, str, bool, float, bool, Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Worker function for parallel label .txt validation and sanitization."""
    path_str, stem, split, auto_clean, emin_threshold, enforce_approval, rev_data = args
    details: List[str] = []
    purged_boxes = 0
    fixed_files = 0
    invalid_cids = 0
    corrupted_boxes = 0
    unapproved_boxes = 0
    empty_label_files = 0
    unresolved_issues = 0
    backups_created = 0

    try:
        with open(path_str, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
    except Exception as exc:
        unresolved_issues += 1
        details.append(f"[{split}:{stem}] Read error: {exc}")
        return {
            "purged_boxes": 0,
            "fixed_files": 0,
            "invalid_cids": 0,
            "corrupted_boxes": 0,
            "unapproved_boxes": 0,
            "empty_label_files": 0,
            "unresolved_issues": unresolved_issues,
            "backups_created": backups_created,
            "details": details,
        }

    lines = raw_content.splitlines()

    # An existing label file with no content lines at all is flagged
    # (informational only — a legit background frame may have no boxes).
    # Previously such a file passed silently with no trace (F-0934).
    if not any(line.strip() for line in lines):
        empty_label_files += 1
        details.append(f"[{split}:{stem}] Empty label file (no boxes)")

    clean_lines: List[str] = []
    modified = False

    trusted_boxes = list(rev_data.get("trusted", [])) if rev_data else []
    # Bucket index over trusted boxes: the old code rescanned the whole
    # trusted list for every label line (O(lines x trusted)); the index
    # answers each lookup from the 3x3 neighbourhood of one cell (~O(1)).
    trusted_index = _build_trusted_box_index(trusted_boxes) if enforce_approval and trusted_boxes else {}

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            details.append(f"[{split}:{stem}] Line {line_idx} syntax corruption ({len(parts)} tokens)")
            purged_boxes += 1
            corrupted_boxes += 1
            modified = True
            continue

        try:
            # S-2306: never truncate a fractional class_id. '7.9' used to
            # become 7 via int(float(...)) — a corrupt token silently
            # turning into a legitimate class. Only integral values pass:
            # '7.0' is tolerated (== 7), '7.9'/'nan'/'inf' are corrupt.
            cid_float = float(parts[0])
            if not cid_float.is_integer():
                raise ValueError(f"fractional class_id {parts[0]!r}")
            cid = int(cid_float)
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as exc:
            details.append(f"[{split}:{stem}] Line {line_idx} non-numeric coordinate error ({exc})")
            purged_boxes += 1
            corrupted_boxes += 1
            modified = True
            continue

        # Check 1: Canonical class_id (taxonomy 0..28)
        if not is_canonical_class(cid):
            hi = max(CANONICAL_CLASS_IDS) if CANONICAL_CLASS_IDS else -1
            details.append(
                f"[{split}:{stem}] Invalid {_non_training_class_reason(cid)} (allowed: 0..{hi})"
            )
            purged_boxes += 1
            invalid_cids += 1
            modified = True
            continue

        # Check 3: Geometry validation and clamping
        valid_geom, clamped, geom_reason = validate_and_clamp_box(cx, cy, w, h)
        if not valid_geom:
            details.append(f"[{split}:{stem}] Geometry rejection (cid={cid}): {geom_reason}")
            purged_boxes += 1
            corrupted_boxes += 1
            modified = True
            continue

        cx_c, cy_c, w_c, h_c = clamped
        if abs(cx_c - cx) > 1e-5 or abs(cy_c - cy) > 1e-5 or abs(w_c - w) > 1e-5 or abs(h_c - h) > 1e-5:
            modified = True

        # Check 2: Unapproved / unverified box — enforced on EVERY split
        # (train and val alike, F-0845), not only train.
        if enforce_approval:
            # Freeze / approved_pool: no review JSON + an exactly-5-field
            # line is a hand label. A 6+ field line is model output and
            # stays unverified. A present review JSON decides the frame;
            # a non-matching line is not promoted to a hand label.
            jsonless_hand_label = rev_data is None and len(parts) == 5
            if jsonless_hand_label:
                box_rep = {
                    "class_id": cid,
                    "cx": cx_c,
                    "cy": cy_c,
                    "w": w_c,
                    "h": h_c,
                    "hand_label": True,
                    "source": "hand_label_txt",
                }
            else:
                # Match on class + centre AND size: centre-only matching pairs
                # a txt line with a different-sized review box (F-0932).
                # Indexed lookup: same rule as the old linear scan (tolerance
                # strictly below _LABEL_MATCH_TOLERANCE on all four axes,
                # first match in review order wins).
                matched_trusted = _find_matching_trusted_box(trusted_index, cid, cx, cy, w, h) if rev_data else None

                box_rep = matched_trusted if matched_trusted else {
                    "class_id": cid,
                    "cx": cx_c,
                    "cy": cy_c,
                    "w": w_c,
                    "h": h_c,
                    "trusted": True if matched_trusted else False,
                    "confidence": 0.50 if not matched_trusted else float(matched_trusted.get("confidence", 0.50)),
                    "source": "unverified_txt" if not matched_trusted else matched_trusted.get("source", ""),
                }

            verified, reason = is_training_box_verified(
                box_rep,
                stem=stem,
                emin_threshold=emin_threshold,
                jsonless_hand_label=jsonless_hand_label,
            )
            if not verified:
                details.append(f"[{split}:{stem}] Purged unverified/unapproved box (cid={cid}): {reason}")
                purged_boxes += 1
                unapproved_boxes += 1
                modified = True
                continue

        clean_lines.append(f"{cid} {cx_c:.6f} {cy_c:.6f} {w_c:.6f} {h_c:.6f}")

    if modified and auto_clean:
        # P2-6: kutu silen/budayan her yeniden yazım ÖNCESİ yedek —
        # eşik yok, silme sayısı ne olursa olsun. Yedek alınamazsa
        # yazım yapılmaz, sorun "çözülmemiş" sayılır.
        backup_path = _backup_file_before_destructive_write(path_str)
        if backup_path is None:
            unresolved_issues += 1
            details.append(f"[{split}:{stem}] Sanitized label txt NOT written (backup FAILED)")
        else:
            backups_created += 1
            try:
                _atomic_write_lines(path_str, clean_lines)
                fixed_files += 1
            except OSError as exc:
                unresolved_issues += 1
                details.append(f"[{split}:{stem}] Failed to write sanitized label txt: {exc}")

    return {
        "purged_boxes": purged_boxes,
        "fixed_files": fixed_files,
        "invalid_cids": invalid_cids,
        "corrupted_boxes": corrupted_boxes,
        "unapproved_boxes": unapproved_boxes,
        "empty_label_files": empty_label_files,
        "unresolved_issues": unresolved_issues,
        "backups_created": backups_created,
        "details": details,
    }


# ==============================================================================
# Public Hygiene Gate APIs
# ==============================================================================

def pre_training_dataset_hygiene_gate(
    dataset_dir: Union[str, Path] = "dataset",
    *,
    auto_clean: bool = True,
    emin_threshold: float = DEFAULT_EMIN_THRESHOLD,
    splits: Sequence[str] = ("train", "val"),
    enforce_training_approval: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sub-second Pre-Training Dataset Quality & Hygiene Gate.

    Fast scan and purge across:
      - `dataset/labels/{train,val}/*.txt`
      - `dataset/review/{train,val}/*.json`
      - `dataset/images/{train,val}/*`

    Enforces 4 strict zero-defect pre-training invariants:
      1. 0 invalid/unknown `class_id` entries (canonical taxonomy 0..28).
      2. 0 unapproved/unverified boxes in training and validation sets.
      3. 0 zero-area, unnormalized, or corrupted boxes.
      4. 0 empty labels with mismatched/missing images.

    Auto-cleans/purges toxic and out-of-bounds annotations in-place.

    Returns:
      dict: {"clean": bool, "purged_boxes": int, "fixed_files": int, "details": list}
      ``clean`` is COMPUTED, never a constant (F-0797): it is True only when
      no defect remains after this run — in dry-run mode (``auto_clean=False``)
      that means no defect was found; in auto-clean mode it means every
      found defect was actually fixed (any failed fix makes the gate FAIL).
    """
    t0 = time.perf_counter()
    root = Path(dataset_dir).resolve()
    details: List[str] = []

    purged_boxes_total = 0
    fixed_files_total = 0
    scanned_labels_total = 0
    scanned_reviews_total = 0
    scanned_images_total = 0

    invalid_class_ids_purged = 0
    unapproved_boxes_purged = 0
    corrupted_boxes_purged = 0
    mismatched_files_fixed = 0
    mismatched_files_detected = 0
    demoted_boxes_total = 0
    empty_label_files_total = 0
    unresolved_issues_total = 0
    backups_created_total = 0

    if not root.is_dir():
        msg = f"Dataset root directory does not exist: {root}"
        logger.warning(msg)
        return {
            "clean": False,
            "purged_boxes": 0,
            "fixed_files": 0,
            "details": [msg],
            "error": "directory_not_found",
        }

    labels_root = root / "labels"
    review_root = root / "review"
    images_root = root / "images"

    n_workers = max_workers or min(32, max(4, (os.cpu_count() or 4) * 2))

    for split in splits:
        split_lbl_dir = labels_root / split
        split_rev_dir = review_root / split
        split_img_dir = images_root / split

        # 1. Map available image stems
        image_stem_to_file: Dict[str, str] = {}
        if split_img_dir.is_dir():
            try:
                for entry in os.scandir(split_img_dir):
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in SUPPORTED_IMAGE_EXTS:
                            stem = os.path.splitext(entry.name)[0]
                            image_stem_to_file[stem] = entry.path
                            scanned_images_total += 1
            except OSError as exc:
                details.append(f"Error scanning images in {split}: {exc}")

        # 2. Process Review JSONs
        review_cache: Dict[str, Dict[str, Any]] = {}
        if split_rev_dir.is_dir():
            review_tasks: List[Tuple[str, str, str, bool, float, bool]] = []
            try:
                for entry in os.scandir(split_rev_dir):
                    if entry.is_file() and entry.name.endswith(".json") and entry.name != "trusted_seed.json":
                        stem = os.path.splitext(entry.name)[0]
                        scanned_reviews_total += 1

                        # Check 4: Orphaned review JSON without image.
                        # "Fixed" is only counted when the file was actually
                        # deleted (auto_clean); a dry-run only DETECTS it
                        # (F-0933).
                        if stem not in image_stem_to_file:
                            mismatched_files_detected += 1
                            if auto_clean:
                                # P2-6: yetim review JSON yedeksiz
                                # silinmiyor — yedek alınamazsa dosya
                                # yerinde kalır, sorun çözülmemiş sayılır.
                                backup_path = _backup_file_before_destructive_write(entry.path)
                                if backup_path is None:
                                    unresolved_issues_total += 1
                                    details.append(f"[{split}] Orphaned review JSON kept (backup FAILED): {entry.name}")
                                else:
                                    backups_created_total += 1
                                    try:
                                        os.unlink(entry.path)
                                        mismatched_files_fixed += 1
                                        fixed_files_total += 1
                                        details.append(f"[{split}] Purged orphaned review JSON with missing image: {entry.name} (backed up to {os.path.basename(backup_path)})")
                                    except OSError as exc:
                                        unresolved_issues_total += 1
                                        details.append(f"[{split}] Could not purge orphaned review JSON {entry.name}: {exc}")
                            else:
                                details.append(f"[{split}] Orphaned review JSON with missing image (dry-run, kept): {entry.name}")
                            continue

                        review_tasks.append(
                            (entry.path, stem, split, auto_clean, float(emin_threshold), enforce_training_approval)
                        )
            except OSError as exc:
                details.append(f"Error scanning review directory in {split}: {exc}")

            if review_tasks:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    results = list(pool.map(_process_single_review_file, review_tasks))

                for res in results:
                    stem = res["stem"]
                    if res["data"] is not None:
                        review_cache[stem] = res["data"]
                    purged_boxes_total += res["purged_boxes"]
                    fixed_files_total += res["fixed_files"]
                    invalid_class_ids_purged += res["invalid_cids"]
                    corrupted_boxes_purged += res["corrupted_boxes"]
                    unapproved_boxes_purged += res["unapproved_boxes"]
                    demoted_boxes_total += res.get("demoted_boxes", 0)
                    unresolved_issues_total += res.get("unresolved_issues", 0)
                    backups_created_total += res.get("backups_created", 0)
                    details.extend(res["details"])

        # 3. Process Label TXT files
        if split_lbl_dir.is_dir():
            label_tasks: List[Tuple[str, str, str, bool, float, bool, Optional[Dict[str, Any]]]] = []
            try:
                for entry in os.scandir(split_lbl_dir):
                    if entry.is_file() and entry.name.endswith(".txt"):
                        stem = os.path.splitext(entry.name)[0]
                        scanned_labels_total += 1

                        # Check 4: Orphaned label file without image.
                        # Counted as fixed only when actually deleted (F-0933).
                        if stem not in image_stem_to_file:
                            mismatched_files_detected += 1
                            if auto_clean:
                                # P2-6: yetim label txt yedeksiz
                                # silinmiyor (gerekçe review JSON ile aynı).
                                backup_path = _backup_file_before_destructive_write(entry.path)
                                if backup_path is None:
                                    unresolved_issues_total += 1
                                    details.append(f"[{split}] Orphaned label file kept (backup FAILED): {entry.name}")
                                else:
                                    backups_created_total += 1
                                    try:
                                        os.unlink(entry.path)
                                        mismatched_files_fixed += 1
                                        fixed_files_total += 1
                                        details.append(f"[{split}] Orphaned label file without image (purged): {entry.name} (backed up to {os.path.basename(backup_path)})")
                                    except OSError as exc:
                                        unresolved_issues_total += 1
                                        details.append(f"[{split}] Could not purge orphaned label file {entry.name}: {exc}")
                            else:
                                details.append(f"[{split}] Orphaned label file without image (dry-run, kept): {entry.name}")
                            continue

                        rev_data = review_cache.get(stem)
                        label_tasks.append(
                            (
                                entry.path,
                                stem,
                                split,
                                auto_clean,
                                float(emin_threshold),
                                enforce_training_approval,
                                rev_data,
                            )
                        )
            except OSError as exc:
                details.append(f"Error scanning labels directory in {split}: {exc}")

            if label_tasks:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    lbl_results = list(pool.map(_process_single_label_file, label_tasks))

                for res in lbl_results:
                    purged_boxes_total += res["purged_boxes"]
                    fixed_files_total += res["fixed_files"]
                    invalid_class_ids_purged += res["invalid_cids"]
                    corrupted_boxes_purged += res["corrupted_boxes"]
                    unapproved_boxes_purged += res["unapproved_boxes"]
                    empty_label_files_total += res.get("empty_label_files", 0)
                    unresolved_issues_total += res.get("unresolved_issues", 0)
                    backups_created_total += res.get("backups_created", 0)
                    details.extend(res["details"])

        # 4. Check 4 reverse: Images that have missing label files
        if split_lbl_dir.is_dir() and split_img_dir.is_dir():
            for stem, img_path_str in image_stem_to_file.items():
                lbl_path_str = os.path.join(str(split_lbl_dir), f"{stem}.txt")
                if not os.path.isfile(lbl_path_str):
                    mismatched_files_detected += 1
                    if auto_clean:
                        try:
                            _atomic_write_lines(lbl_path_str, [])
                            mismatched_files_fixed += 1
                            fixed_files_total += 1
                            details.append(f"[{split}] Missing label file for image (created empty): {os.path.basename(img_path_str)}")
                        except OSError as exc:
                            unresolved_issues_total += 1
                            details.append(f"[{split}] Could not create missing label file for image {os.path.basename(img_path_str)}: {exc}")
                    else:
                        details.append(f"[{split}] Missing label file for image (dry-run, not created): {os.path.basename(img_path_str)}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # clean is COMPUTED (F-0797), never a constant True:
    # - dry-run: clean only when no defect was found at all (nothing was
    #   fixed, so any found defect still stands);
    # - auto-clean: clean when every found defect was actually fixed, i.e.
    #   no fix failed (purged/demoted/deleted/created defects are gone).
    defects_found = purged_boxes_total + demoted_boxes_total + mismatched_files_detected
    if auto_clean:
        is_clean = unresolved_issues_total == 0
    else:
        is_clean = defects_found == 0 and unresolved_issues_total == 0

    summary = (
        f"Pre-Training Hygiene Gate Completed in {elapsed_ms:.1f}ms: "
        f"Scanned {scanned_labels_total} labels, {scanned_reviews_total} reviews, {scanned_images_total} images. "
        f"Purged {purged_boxes_total} toxic/corrupted boxes, demoted {demoted_boxes_total} boxes to weak, "
        f"fixed {fixed_files_total} files, {unresolved_issues_total} unresolved issue(s)."
    )
    logger.info(summary)

    return {
        "clean": is_clean,
        "purged_boxes": purged_boxes_total,
        "fixed_files": fixed_files_total,
        "details": details,
        # Extended metadata
        "scanned_labels": scanned_labels_total,
        "scanned_reviews": scanned_reviews_total,
        "scanned_images": scanned_images_total,
        "invalid_class_ids_purged": invalid_class_ids_purged,
        "unapproved_boxes_purged": unapproved_boxes_purged,
        "corrupted_boxes_purged": corrupted_boxes_purged,
        "demoted_boxes": demoted_boxes_total,
        "empty_label_files": empty_label_files_total,
        "mismatched_files_fixed": mismatched_files_fixed,
        "mismatched_files_detected": mismatched_files_detected,
        "unresolved_issues": unresolved_issues_total,
        "backups_created": backups_created_total,
        "duration_ms": round(elapsed_ms, 2),
        "status": "PASSED" if is_clean else "FAILED",
        "summary": summary,
    }


def audit_dataset_hygiene(
    dataset_dir: Union[str, Path] = "dataset",
    emin_threshold: float = DEFAULT_EMIN_THRESHOLD,
    splits: Sequence[str] = ("train", "val"),
) -> Dict[str, Any]:
    """Dry-run non-destructive audit of dataset hygiene."""
    return pre_training_dataset_hygiene_gate(
        dataset_dir=dataset_dir,
        auto_clean=False,
        emin_threshold=emin_threshold,
        splits=splits,
    )


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    print(f"Running Pre-Training Quality & Hygiene Gate on: {target_dir}")
    report = pre_training_dataset_hygiene_gate(target_dir, auto_clean=True)
    print("\n--- HYGIENE GATE REPORT ---")
    print(f"Clean: {report['clean']}")
    print(f"Purged Boxes: {report['purged_boxes']}")
    print(f"Fixed Files: {report['fixed_files']}")
    print(f"Duration: {report.get('duration_ms', 0):.2f} ms")
    print(f"Invalid Class IDs: {report.get('invalid_class_ids_purged', 0)}")
    print(f"Unapproved Boxes: {report.get('unapproved_boxes_purged', 0)}")
    print(f"Corrupted Boxes: {report.get('corrupted_boxes_purged', 0)}")
    print(f"Demoted to Weak: {report.get('demoted_boxes', 0)}")
    print(f"Unresolved Issues: {report.get('unresolved_issues', 0)}")
    print(f"Mismatched Files: {report.get('mismatched_files_fixed', 0)}")
    print(f"Total Details: {len(report['details'])}")
    if report["details"]:
        print("Sample details (first 10):")
        for d in report["details"][:10]:
            print(f" - {d}")
    print("---------------------------")
