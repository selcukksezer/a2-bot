# -*- coding: utf-8 -*-
"""
dataset_factory/exemplar_memory.py
==================================
Active Exemplar / Few-Shot Learning Memory for YOLO Object Labeling.

Teaches AI from human feedback:
1. When user approves or class-corrects a bounding box in YOLO Factory Dialog,
   the crop's visual feature prototype (HSV color histogram + aspect ratio +
   texture descriptor) is saved as a POSITIVE exemplar in `dataset/learned_exemplars.json`.
2. When user rejects / deletes a false-positive bounding box, the crop is saved
   as a NEGATIVE exemplar in `dataset/learned_exemplars.json`.
3. During auto-labeling (`YOLOAutoLabeler.auto_label_frame`):
   - Candidate box crops are compared against learned negative exemplars:
     if similarity >= 0.82, candidate is suppressed as a known false positive.
   - Candidate box crops are compared against learned positive exemplars:
     if similarity >= 0.70, class is reinforced or corrected, and confidence is boosted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

# P1-1: the canonical sets come from the single taxonomy source
# (dataset_factory.class_taxonomy, authority dataset/data.yaml) — this
# file used to keep its own range(27) / {0..8, 12} copies (K-01).
from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    INTERNAL_SLOT_STATE_IDS,
    MATERIAL_3D_CLASSES,
    is_same_material_family,
)

logger = logging.getLogger("Gatherion.ExemplarMemory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXEMPLAR_PATH = PROJECT_ROOT / "dataset" / "learned_exemplars.json"

# Canonical YOLO class IDs for exemplar memory: full taxonomy 0..26
# (dataset_factory/class_taxonomy.py CANONICAL_NAMES — Decision 1).
# 22..26 are legitimate canonical classes (22 hp_potion, 23 mp_potion,
# 24 buff_scroll, 25 kinah, 26 player_character) and may be learned as
# exemplars here. The old `class_id >= 22` poison threshold was wrong and
# is removed: the real poison is an UNRESOLVED `essence_*` catalog alias —
# a class_name like `essence_aria` that was never mapped onto a canonical
# class (catalog essences resolve to 0..12, see class_taxonomy.py) — so
# poison is detected by class NAME, never by ID range.
# P1-1: single source — the set itself, not a local range(27) copy.
CANONICAL_CLASSES: Set[int] = set(ALL_CANONICAL_CLASSES)


def _is_poisoned_essence_name(class_name: Any) -> bool:
    """Return True if `class_name` is an unresolved `essence_*` catalog alias.

    Poison doctrine (Decision 1): catalog essence items resolve onto canonical
    IDs 0..12 (see class_taxonomy.py), so a raw `essence_*` NAME on an exemplar
    means it was never resolved — that is the poison. Class IDs 22..26 are
    legitimate canonical classes (hp_potion, mp_potion, buff_scroll, kinah,
    player_character) and are NOT poison, so no ID-range check belongs here.
    """
    cname = str(class_name or "").lower()
    return cname.startswith("essence_") or "essence" in cname

# Category-locked super categories. Aligned with
# dataset_factory/class_taxonomy.py MATERIAL_3D_CLASSES ({0..8, 12}):
# classes 6/7/8 (herb_edana, herb_lumina, ore_obsidian) are 3D materials,
# not a separate "crests" category.
# P1-1: single source — the set itself, not a local copy.
SUPER_CATEGORY_3D_MATERIALS: Set[int] = set(MATERIAL_3D_CLASSES)
SUPER_CATEGORY_MINIMAP: Set[int] = {9, 11}
SUPER_CATEGORY_HUD: Set[int] = {13, 16, 17, 18, 19, 20, 21}
SUPER_CATEGORY_3D_MONSTERS: Set[int] = {10}
SUPER_CATEGORY_3D_PATHS: Set[int] = {14, 15}
# S-4901: 22..26 are canonical classes too (0..26), but get_super_category()
# had no branch for them and returned None — so category-locked repair
# among 22..24 could never happen and match_positive(..., same_category_as=22)
# always returned (0.0, None, ...). Aligned with the sibling mappings:
# vectorized_exemplar_matcher.py SUPER_CATEGORY_CONSUMABLES={22, 23, 24}
# ('consumables') and class_taxonomy.py CONSUMABLE/CURRENCY/PLAYER classes
# (22..24 consumable, 25 currency, 26 player). 25/26 are singletons, so
# same-category repair for them can only ever map a class onto itself.
SUPER_CATEGORY_CONSUMABLES: Set[int] = {22, 23, 24}
SUPER_CATEGORY_CURRENCY: Set[int] = {25}
SUPER_CATEGORY_PLAYER: Set[int] = {26}

# 3D-world super categories (the fine categories whose members live in the
# 3D world). SUPER_CATEGORY_3D_WORLD as a separate set was dead: every one
# of its members already maps to a finer category in get_super_category(),
# so it could never be returned. The 3D-world gate now uses this tuple.
_3D_WORLD_CATEGORIES: Tuple[str, ...] = ("3d_materials", "3d_monsters", "3d_paths")


# Integer codes for the super categories above, so match_positive() can
# build its category mask with one vectorized NumPy comparison instead of
# calling get_super_category() once per exemplar per query (the old
# per-exemplar Python loop cost ~3 ms per call at N=500; see D50 bench).
# -1 = no super category (never occurs for canonical exemplars: classes
# 0..26 are all covered by the eight sets above — O-2: the table used to
# stop at 21, so 22..26 exemplars were clipped to HUD code 2 and
# match_positive(..., same_category_as=22) raised KeyError).
_SUPER_CATEGORY_CODES: Dict[str, int] = {
    "3d_materials": 0,
    "minimap": 1,
    "hud": 2,
    "3d_monsters": 3,
    "3d_paths": 4,
    "consumables": 5,
    "currency": 6,
    "player": 7,
}
_SUPER_CATEGORY_CODE_BY_CLASS = np.full(29, -1, dtype=np.int8)
for _code_name, _code, _members in (
    ("3d_materials", 0, SUPER_CATEGORY_3D_MATERIALS),
    ("minimap", 1, SUPER_CATEGORY_MINIMAP),
    ("hud", 2, SUPER_CATEGORY_HUD),
    ("3d_monsters", 3, SUPER_CATEGORY_3D_MONSTERS),
    ("3d_paths", 4, SUPER_CATEGORY_3D_PATHS),
    ("consumables", 5, SUPER_CATEGORY_CONSUMABLES),
    ("currency", 6, SUPER_CATEGORY_CURRENCY),
    ("player", 7, SUPER_CATEGORY_PLAYER),
):
    for _cid in _members:
        _SUPER_CATEGORY_CODE_BY_CLASS[_cid] = _code
del _code_name, _code, _members, _cid


# Y-2: the three exemplar decision thresholds, named in one place.
# Values unchanged (0.70 reinforce / 0.82 suppress / 0.88 reassign);
# previously the same questions were answered with different hardcoded
# numbers in skill_bar_reader (0.65/0.68/0.72/0.75) — G unified the
# reader side to a single 0.68 setting; these three stay distinct
# decisions (reinforce vs suppress vs reassign) and live here.
REINFORCE_POS_THRESHOLD: float = 0.70
SUPPRESS_NEG_THRESHOLD: float = 0.82
REASSIGN_POS_THRESHOLD: float = 0.88

# K-3: a degenerate crop carries no visual information — a single-bin
# histogram (every pixel in one HSV bucket) with zero edge/sharpness
# texture. A legit icon/bar always has edges or several bins; the
# near-square band keeps wide solid bars (HP/MP, aspect >> 1) out of
# the gate — they are genuinely single-color and valid exemplars.
#
# v12 (Selçuk kararı, 10 Kasım): kapı DARALTILDI — artık yalnızca
# POZİTİF tarafta uygulanır. Düz (tek renk, dokusuz) bir yama pozitif
# prototip olarak hiçbir şey öğretmez, ama NEGATİF olarak anlamlı bir
# bastırma sinyalidir ("bu düz gri yama bir kaynak DEĞİL"): düz gri
# kaya negatifi öğretilebilmelidir. Bu yüzden add_negative_exemplar
# artık bu kapıyı uygulamaz ve sanitize() de negatif listedeki
# dejenere kayıtları silmez (negatif taraf hiçbir zaman bu kapıyla
# temizlenmiyordu — _is_clean_negative_record'da dejenere kontrolü
# yok). Dedektörün kendisi (0.99 eşiği dahil) DEĞİŞMEDİ: 0.99–1.0
# aralığı yalnızca gri-Sobel'e görünmez (parlaklığı sabit, yalnızca
# ton farkı olan) benekleri kapsar — pozitif prototip olarak onlar da
# içeriksizdir; gerçek kaya yaması parlaklık değişimi taşır, dokusu
# sıfır değildir ve kapıdan zaten geçer. Eşik gevşetmek pozitif
# tarafı kirletir, kaya negatifine hiçbir şey kazandırmaz.
_DEGENERATE_MAX_BIN: float = 0.99
_DEGENERATE_ASPECT_LO: float = 0.8
_DEGENERATE_ASPECT_HI: float = 1.25


def _is_degenerate_features(feats: Any) -> bool:
    """True for a content-free crop: flat single color, no edges, near-square.

    Accepts a feature dict (hsv_hist/texture/aspect_ratio) or a stored
    exemplar record (same fields). Missing fields -> not degenerate
    (fail-open for legacy partial records: they are judged by the
    canonicalization path instead).
    """
    if not isinstance(feats, dict):
        return False
    hist = feats.get("hsv_hist")
    if not isinstance(hist, (list, tuple)) or len(hist) != 432:
        return False
    try:
        if max(float(v) for v in hist) < _DEGENERATE_MAX_BIN:
            return False
        tex = feats.get("texture") or []
        if len(tex) != 5 or any(float(tex[i]) != 0.0 for i in (0, 1, 2)):
            return False
        aspect = float(feats.get("aspect_ratio", 1.0))
    except (TypeError, ValueError):
        return False
    return _DEGENERATE_ASPECT_LO <= aspect <= _DEGENERATE_ASPECT_HI


def _is_negative_named(class_name: Any) -> bool:
    """True when the NAME says the record is a rejected box (K-1).

    A positive-list record whose name contains 'negative' (e.g.
    'monster_3d_negative') is a rejected box stored on the wrong side:
    it reinforces instead of suppressing. The name is the marker —
    canonicalization must never rename it away (it would hide the
    marker), it must be moved to the negative list.
    """
    return "negative" in str(class_name or "").lower()


def _is_wrong_node_marker_name(class_name: Any) -> bool:
    """True when the NAME is a wrong-node-scan provenance marker (v19/F12).

    `core/wrong_node_scan.py` stores a deleted / rejected node whose
    content is UNKNOWN as (15, ``"wrong_node_<reason>"``) — the 15 is a
    storage bucket (the negative list admits canonical ids only), NOT a
    path_ground claim. The name is the only thing that says "unknown
    content, do not read this as ground evidence", so — same K-1
    doctrine as ``_is_negative_named`` — canonicalization must never
    rename it away: ``sanitize_class_id_and_name`` Priority 4 rewrites
    any unresolvable name on a valid canonical id to the canonical
    name, i.e. ``(15, "wrong_node_purge")`` silently became
    ``(15, "path_ground")`` on every add, and the marker F12's comment
    claimed was stored never reached disk (P1-8 / K-21 class-15
    bucket). Only the ``wrong_node_`` prefix counts: a bare
    ``"wrong_node"`` without a reason suffix is not a marker.
    """
    return str(class_name or "").strip().lower().startswith("wrong_node_")


def _is_state_unknown_skill_slot(class_id: Any, class_name: Any) -> bool:
    """True for a stateless legacy skill-slot record (K-2).

    Legacy (21, 'skill_slot') carries no ready/cooldown state — the old
    22-class scheme had one slot id for both states. It must not count
    as cooldown evidence (its current cid 21) NOR be remapped to ready
    (20, what the generic legacy map would do): both assert a state
    nobody recorded. Such records are quarantined, not matched.
    """
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid == 21 and str(class_name or "").strip().lower() == "skill_slot"


def _is_class_unknown_material(class_id: Any, class_name: Any) -> bool:
    """True for a generic legacy 'world_material' record (Y-1 edge).

    Legacy 12 was a generic 'world material' bucket; canonical 12 is
    specifically crystal_blood. Guessing crystal_blood for a crop whose
    material was never identified would mint false evidence, so the
    record is quarantined instead of remapped.
    """
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid == 12 and str(class_name or "").strip().lower() == "world_material"


def _canonicalize_exemplar_class(
    class_id: Any, class_name: Any
) -> Tuple[int, str]:
    """Resolve a (possibly legacy) exemplar (cid, name) to canonical form.

    Uses dataset_factory.class_taxonomy.sanitize_class_id_and_name — the
    codebase's own legacy authority (LEGACY_YOLO_CLASS_TO_CANONICAL):
    (18,'hp_potion')->(22,'hp_potion'), (20,'target_hp_bar')->(18,...),
    (6/7/8 prompt aliases)->(13,'skill_ui_prompt'). Raises ValueError
    when neither id nor name resolves.
    """
    from dataset_factory.class_taxonomy import sanitize_class_id_and_name

    return sanitize_class_id_and_name(class_id, class_name)


def _compute_revision(
    positive: Sequence[Dict[str, Any]],
    negative: Sequence[Dict[str, Any]],
    quarantined: Sequence[Dict[str, Any]],
) -> str:
    """Content hash (O-5): identifies WHICH memory produced a similarity.

    mtime-only 'revisions' change on every save even when nothing was
    learned; this hash changes only when the exemplar content changes,
    so a stored exemplar_*_sim can be checked against the memory that
    produced it. First 16 hex chars of the SHA-256 over the canonical
    JSON of all three lists.
    """
    blob = json.dumps(
        {"positive": list(positive), "negative": list(negative), "quarantined": list(quarantined)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_3d_world_class(class_id: int) -> bool:
    """Return True if class_id belongs to a 3D-world super category."""
    return get_super_category(class_id) in _3D_WORLD_CATEGORIES

FEATURE_VECTOR_DIM: int = 438


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort int() that returns `default` for non-numeric values instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_box_crop_impl(
    frame: np.ndarray,
    box: Union[Dict[str, Any], Sequence[float], Tuple[float, float, float, float]],
) -> Optional[np.ndarray]:
    """
    Safely crop a bounding box region from an image `frame`.
    Accepts:
      - dict with keys 'cx', 'cy', 'w', 'h' (normalized) or 'bbox' / 'bbox_xyxy' / 'bbox_xywh' (pixels)
      - tuple / list of 4 floats (cx, cy, w, h normalized or x1, y1, x2, y2 pixels)
    """
    if frame is None or frame.size == 0 or frame.ndim < 2:
        return None

    img_h, img_w = frame.shape[:2]
    if img_w < 2 or img_h < 2:
        return None

    x1, y1, x2, y2 = 0, 0, 0, 0

    if isinstance(box, dict):
        if "bbox_xyxy" in box and len(box["bbox_xyxy"]) == 4:
            b = box["bbox_xyxy"]
            x1, y1, x2, y2 = int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))
        elif "bbox_xywh" in box and len(box["bbox_xywh"]) == 4:
            bx, by, bw, bh = box["bbox_xywh"]
            x1, y1 = int(round(bx)), int(round(by))
            x2, y2 = int(round(bx + bw)), int(round(by + bh))
        elif "cx" in box and "cy" in box and "w" in box and "h" in box:
            cx = float(box["cx"])
            cy = float(box["cy"])
            bw = float(box["w"])
            bh = float(box["h"])
            # Handle normalized [0, 1] vs pixel coords
            if max(cx, cy, bw, bh) <= 1.0:
                x1 = int(round((cx - 0.5 * bw) * img_w))
                y1 = int(round((cy - 0.5 * bh) * img_h))
                x2 = int(round((cx + 0.5 * bw) * img_w))
                y2 = int(round((cy + 0.5 * bh) * img_h))
            else:
                x1 = int(round(cx - 0.5 * bw))
                y1 = int(round(cy - 0.5 * bh))
                x2 = int(round(cx + 0.5 * bw))
                y2 = int(round(cy + 0.5 * bh))
        elif "bbox" in box and len(box["bbox"]) == 4:
            bx, by, bw, bh = box["bbox"]
            if max(bx, by, bw, bh) <= 1.0:
                x1 = int(round(bx * img_w))
                y1 = int(round(by * img_h))
                x2 = int(round((bx + bw) * img_w))
                y2 = int(round((by + bh) * img_h))
            else:
                x1, y1, x2, y2 = int(round(bx)), int(round(by)), int(round(bx + bw)), int(round(by + bh))
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        v0, v1, v2, v3 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if max(v0, v1, v2, v3) <= 1.0:
            # Assume cx, cy, w, h normalized
            x1 = int(round((v0 - 0.5 * v2) * img_w))
            y1 = int(round((v1 - 0.5 * v3) * img_h))
            x2 = int(round((v0 + 0.5 * v2) * img_w))
            y2 = int(round((v1 + 0.5 * v3) * img_h))
        else:
            # Assume xyxy or xywh pixel
            if v2 > v0 and v3 > v1:
                x1, y1, x2, y2 = int(round(v0)), int(round(v1)), int(round(v2)), int(round(v3))
            else:
                x1, y1, x2, y2 = int(round(v0)), int(round(v1)), int(round(v0 + v2)), int(round(v1 + v3))

    # A box fully outside the frame (or degenerate: zero/negative size) has
    # no pixels to crop. Previously the clamps below forced such a box into
    # a bogus 2x2 sliver at the frame edge; return None instead.
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return None

    x1 = max(0, min(img_w - 2, x1))
    y1 = max(0, min(img_h - 2, y1))
    x2 = max(x1 + 2, min(img_w, x2))
    y2 = max(y1 + 2, min(img_h, y2))

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def extract_box_crop(
    frame: np.ndarray,
    box: Union[Dict[str, Any], Sequence[float], Tuple[float, float, float, float]],
) -> Optional[np.ndarray]:
    """Crop a bounding box region from `frame` (see `_extract_box_crop_impl`).

    S134 (v19/P3-8): bozuk kutu (sayı-olmayan koordinat, 4'lü olmayan
    `bbox_xyxy`, ...) önceden korumasız `len()`/`float()` çağrılarında
    TypeError/ValueError atıp `refine_annotations` döngüsünün tamamını
    düşürüyordu — tek bozuk kutu tüm etiket iyileştirmesini öldürüyordu.
    Bozuk kutu None döner (çağıran kutuyu olduğu gibi korur), hata loglanır.
    """
    try:
        return _extract_box_crop_impl(frame, box)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        logger.debug("extract_box_crop: bozuk kutu atlandı (%s): %r", exc, box)
        return None


def extract_crop_features(
    crop: np.ndarray,
    include_1d_hists: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Extract standardized visual feature vector from an image crop:
    - 3D HSV Color Histogram (12 Hue x 6 Sat x 6 Val = 432 bins, L1 normalized)
    - Aspect Ratio (w / h)
    - Texture descriptor: [mean_edge, std_edge, laplacian_sharpness, mean_gray, std_gray]

    `include_1d_hists=False` skips the 1D Hue/Sat/Val histograms (hist_h/s/v):
    they are only consumed by the legacy compute_similarity() fallback for
    partial features without a 432-bin histogram, never by the vectorized
    cosine path — and they cost ~half of this function (3 extra calcHist
    calls). Internal matching passes False; the default True keeps the
    returned dict identical for external callers (CalibrationWizard compat).
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return None

    if crop.ndim == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    elif crop.ndim == 3 and crop.shape[2] == 3:
        crop_bgr = crop
    else:
        return None

    ch, cw = crop_bgr.shape[:2]
    if cw < 3 or ch < 3:
        return None

    # 1. HSV Histogram (3D 12x6x6 = 432 bins + 1D channels)
    # Rounding is vectorized (np.round on float64): identical values to
    # the old per-element round(float(v), n) — both are round-half-even on
    # the same float64 values — but ~10x cheaper than 432 Python calls
    # in this per-candidate hot path (see D50 bench).
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    hist_vec = np.round(hist.flatten().astype(np.float64), 5).tolist()

    # 1D Histograms for CalibrationWizard compatibility (16 Hue, 8 Sat, 8 Val).
    # Skipped when include_1d_hists=False (internal vectorized matching only
    # needs hsv_hist / aspect_ratio / texture / mean_hsv).
    if include_1d_hists:
        h_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_h = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_h = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        h_sum = float(np.sum(h_h)) or 1.0
        s_sum = float(np.sum(s_h)) or 1.0
        v_sum = float(np.sum(v_h)) or 1.0
        hist_h = np.round(h_h.astype(np.float64) / h_sum, 4).tolist()
        hist_s = np.round(s_h.astype(np.float64) / s_sum, 4).tolist()
        hist_v = np.round(v_h.astype(np.float64) / v_sum, 4).tolist()
    else:
        hist_h, hist_s, hist_v = [], [], []
    mean_hsv = [
        round(float(np.mean(hsv[:, :, 0])), 2),
        round(float(np.mean(hsv[:, :, 1])), 2),
        round(float(np.mean(hsv[:, :, 2])), 2),
    ]

    # 2. Aspect Ratio
    aspect_ratio = round(float(cw) / float(max(1, ch)), 4)

    # 3. Texture Descriptor
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mean_edge = round(float(np.mean(mag)) / 255.0, 4)
    std_edge = round(float(np.std(mag)) / 255.0, 4)

    try:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        norm_lap = round(min(1.0, lap_var / 1000.0), 4)
    except Exception:
        norm_lap = 0.0

    mean_val = round(float(np.mean(gray)) / 255.0, 4)
    std_val = round(float(np.std(gray)) / 255.0, 4)

    texture_vec = [mean_edge, std_edge, norm_lap, mean_val, std_val]

    return {
        "hsv_hist": hist_vec,
        "hist_h": hist_h,
        "hist_s": hist_s,
        "hist_v": hist_v,
        "mean_hsv": mean_hsv,
        "aspect_ratio": aspect_ratio,
        "texture": texture_vec,
    }


def compute_similarity(feat1: Dict[str, Any], feat2: Dict[str, Any]) -> float:
    """
    Compute similarity score in [0.0, 1.0] between two crop feature dicts.

    Single canonical metric: cosine similarity of the normalized 438-dim
    feature vectors (see feature_dict_to_normalized_vector) — the same
    metric the vectorized matrix path (match_positive / match_negative)
    applies to this memory. Previously this function used a different
    weighted histogram-intersection formula (70% intersection + 15%
    aspect-ratio + 15% texture) while the matrix path used cosine, so the
    same exemplar memory was judged by two yardsticks with the same
    thresholds (0.70 reinforce / 0.82 suppress). The weighted ladder below
    is kept ONLY as a fallback for legacy partial features without a full
    432-bin HSV histogram.
    """
    if not feat1 or not feat2:
        return 0.0

    h1 = np.array(feat1.get("hsv_hist", []), dtype=np.float32)
    h2 = np.array(feat2.get("hsv_hist", []), dtype=np.float32)
    if h1.size == 432 and h2.size == 432:
        return round(float(compute_feature_cosine_similarity(feat1, feat2)), 4)

    # Legacy fallback (partial features only): weighted histogram intersection.
    if "hist_h" in feat1 and "hist_h" in feat2:
        # Fallback to 1D channel intersection (CalibrationWizard format)
        hh1 = np.array(feat1.get("hist_h", []), dtype=np.float32)
        hh2 = np.array(feat2.get("hist_h", []), dtype=np.float32)
        hs1 = np.array(feat1.get("hist_s", []), dtype=np.float32)
        hs2 = np.array(feat2.get("hist_s", []), dtype=np.float32)
        hv1 = np.array(feat1.get("hist_v", []), dtype=np.float32)
        hv2 = np.array(feat2.get("hist_v", []), dtype=np.float32)
        sim_h = float(np.sum(np.minimum(hh1, hh2))) if hh1.size == hh2.size and hh1.size > 0 else 0.5
        sim_s = float(np.sum(np.minimum(hs1, hs2))) if hs1.size == hs2.size and hs1.size > 0 else 0.5
        sim_v = float(np.sum(np.minimum(hv1, hv2))) if hv1.size == hv2.size and hv1.size > 0 else 0.5
        hist_sim = 0.50 * sim_h + 0.25 * sim_s + 0.25 * sim_v
    else:
        # Color difference from mean_hsv if available
        mh1 = feat1.get("mean_hsv")
        mh2 = feat2.get("mean_hsv")
        if mh1 and mh2 and len(mh1) == 3 and len(mh2) == 3:
            dh = abs(float(mh1[0]) - float(mh2[0])) / 180.0
            ds = abs(float(mh1[1]) - float(mh2[1])) / 255.0
            dv = abs(float(mh1[2]) - float(mh2[2])) / 255.0
            hist_sim = max(0.0, 1.0 - (0.5 * dh + 0.25 * ds + 0.25 * dv))
        else:
            hist_sim = 0.0

    hist_sim = max(0.0, min(1.0, hist_sim))

    # 2. Aspect Ratio Similarity
    ar1 = max(0.01, float(feat1.get("aspect_ratio", 1.0)))
    ar2 = max(0.01, float(feat2.get("aspect_ratio", 1.0)))
    ar_sim = min(ar1, ar2) / max(ar1, ar2)

    # 3. Texture Similarity (1 - mean L1 distance)
    t1 = np.array(feat1.get("texture", [0.0] * 5), dtype=np.float32)
    t2 = np.array(feat2.get("texture", [0.0] * 5), dtype=np.float32)
    if t1.size == 5 and t2.size == 5:
        tex_dist = float(np.mean(np.abs(t1 - t2)))
        tex_sim = max(0.0, min(1.0, 1.0 - tex_dist))
    else:
        tex_sim = 0.5

    total = 0.70 * hist_sim + 0.15 * ar_sim + 0.15 * tex_sim
    return round(float(np.clip(total, 0.0, 1.0)), 4)


def get_super_category(class_id: int) -> Optional[str]:
    """
    Map class_id to its super-category for category-locked class refinement:
    - '3d_materials': 3D world gatherable resources (0..8, 12 — taxonomy
      MATERIAL_3D_CLASSES; classes 6/7/8 are materials, not a separate
      "3d_crests" category)
    - 'minimap': Minimap symbols (9, 11)
    - 'hud': HUD and cognition symbols (13, 16..21)
    - '3d_monsters': 3D monsters (10)
    - '3d_paths': Walkable path & obstacles (14, 15)
    - 'consumables': HUD consumables (22 hp_potion, 23 mp_potion, 24 buff_scroll)
    - 'currency': Currency (25)
    - 'player': Player character (26)
    Returns None for invalid or non-canonical classes (e.g. unresolved catalog essence names or class IDs outside 0..26).
    """
    cid = int(class_id)
    if cid in SUPER_CATEGORY_3D_MATERIALS:
        return "3d_materials"
    if cid in SUPER_CATEGORY_MINIMAP:
        return "minimap"
    if cid in SUPER_CATEGORY_HUD:
        return "hud"
    if cid in SUPER_CATEGORY_3D_MONSTERS:
        return "3d_monsters"
    if cid in SUPER_CATEGORY_3D_PATHS:
        return "3d_paths"
    if cid in SUPER_CATEGORY_CONSUMABLES:
        return "consumables"
    if cid in SUPER_CATEGORY_CURRENCY:
        return "currency"
    if cid in SUPER_CATEGORY_PLAYER:
        return "player"
    return None


def is_same_super_category(cid1: int, cid2: int) -> bool:
    """
    Enforce category locking: only classes within the exact same super category
    are permitted to repair each other.
    3D world materials (0..8, 12) can only change among 0..8, 12.
    Minimap symbols (9, 11) can only change among 9, 11.
    HUD symbols (13, 16..21) can only change among HUD classes.
    Consumables (22..24) can only change among 22..24; currency (25) and
    player (26) are singleton categories.
    Never allow cross-category repairs or repairs into essence_* (unresolved essence aliases).
    """
    cat1 = get_super_category(cid1)
    cat2 = get_super_category(cid2)
    if cat1 is None or cat2 is None:
        return False
    return cat1 == cat2


def verify_hsv_color_consistency(
    feat1: Dict[str, Any],
    feat2: Dict[str, Any],
    min_intersection: float = 0.50,
    max_hue_diff: float = 30.0,
) -> bool:
    """
    Verify HSV color distribution consistency between two visual feature dicts.
    Returns True only if:
    1. HSV color histogram intersection is >= min_intersection (when histograms available)
    2. Mean Hue difference is within max_hue_diff (circular on 0..180 OpenCV scale) for chromatic crops,
       or Sat/Val are consistent for low-saturation (achromatic) crops.
    """
    if not feat1 or not feat2:
        return False

    h1 = np.array(feat1.get("hsv_hist", []), dtype=np.float32)
    h2 = np.array(feat2.get("hsv_hist", []), dtype=np.float32)
    has_hist = (h1.size == 432 and h2.size == 432)

    hist_inter = 0.0
    if has_hist:
        hist_inter = float(np.sum(np.minimum(h1, h2)))

    mh1 = feat1.get("mean_hsv")
    mh2 = feat2.get("mean_hsv")
    hue_ok = True
    sat_val_ok = True

    if mh1 and mh2 and len(mh1) >= 3 and len(mh2) >= 3:
        h_val1, s_val1, v_val1 = float(mh1[0]), float(mh1[1]), float(mh1[2])
        h_val2, s_val2, v_val2 = float(mh2[0]), float(mh2[1]), float(mh2[2])

        # Circular Hue difference on OpenCV 0..180 scale
        dh = min(abs(h_val1 - h_val2), 180.0 - abs(h_val1 - h_val2))
        ds = abs(s_val1 - s_val2) / 255.0
        dv = abs(v_val1 - v_val2) / 255.0

        if s_val1 > 25.0 and s_val2 > 25.0:
            # Saturated/chromatic: Dominant hue must align within max_hue_diff
            hue_ok = (dh <= max_hue_diff)
        else:
            # Low-saturation / grayscale: Hue is noisy; check saturation and value similarity
            sat_val_ok = (ds <= 0.35 and dv <= 0.40)
    elif not has_hist:
        return False

    if has_hist:
        return (hist_inter >= min_intersection) and hue_ok and sat_val_ok
    else:
        return hue_ok and sat_val_ok


def feature_dict_to_normalized_vector(feat: Dict[str, Any]) -> np.ndarray:
    """
    Convert visual feature dictionary to a pre-normalized 1D feature vector of dimension D = 438.
    Applies calibrated weighting:
    - HSV histogram: 432 bins (L2 normalized, weighted by sqrt(0.70))
    - Aspect ratio: 1 bin (normalized to [0..5] range, weighted by sqrt(0.15))
    - Texture: 5 bins (L2 normalized, weighted by sqrt(0.15))
    The resulting vector has unit L2 norm (||v||_2 = 1.0) so that:
        np.dot(matrix, query_vec) == cosine_similarity(matrix, query_vec)
    """
    hsv = np.array(feat.get("hsv_hist", []), dtype=np.float32)
    if hsv.size != 432:
        hsv_pad = np.zeros(432, dtype=np.float32)
        if hsv.size > 0:
            hsv_pad[: min(432, hsv.size)] = hsv[:432]
        hsv = hsv_pad
    hsv_norm = hsv / (np.linalg.norm(hsv) + 1e-9)

    ar = float(feat.get("aspect_ratio", 1.0))
    # Log-scale the aspect ratio into [0, 1] (symmetric in ratio space:
    # ar and 1/ar sit equidistant from 1.0). The old `clip(ar)/2.0` put
    # values up to 5.0 into this single bin, so a wide/tall crop dominated
    # the whole 438-dim vector and distorted cosine similarity.
    ar_clipped = float(np.clip(ar, 0.05, 10.0))
    ar_norm = (math.log(ar_clipped) - math.log(0.05)) / (math.log(10.0) - math.log(0.05))
    ar_val = np.array([ar_norm], dtype=np.float32)

    tex = np.array(feat.get("texture", [0.0] * 5), dtype=np.float32)
    if tex.size != 5:
        tex_pad = np.zeros(5, dtype=np.float32)
        if tex.size > 0:
            tex_pad[: min(5, tex.size)] = tex[:5]
        tex = tex_pad
    tex_norm = tex / (np.linalg.norm(tex) + 1e-9)

    v = np.concatenate([
        np.sqrt(0.70) * hsv_norm,
        np.sqrt(0.15) * ar_val,
        np.sqrt(0.15) * tex_norm,
    ])
    norm = np.linalg.norm(v)
    return (v / (norm + 1e-9)).astype(np.float32)


def compute_feature_cosine_similarity(feat1: Dict[str, Any], feat2: Dict[str, Any]) -> float:
    """Compute cosine similarity between two feature dictionaries via normalized feature vectors."""
    if not feat1 or not feat2:
        return 0.0
    v1 = feature_dict_to_normalized_vector(feat1)
    v2 = feature_dict_to_normalized_vector(feat2)
    return float(np.clip(np.dot(v1, v2), 0.0, 1.0))


def _created_at_to_epoch(value: Any) -> Optional[float]:
    """Parse a CalibrationWizard ISO `created_at` ('%Y-%m-%dT%H:%M:%SZ') to epoch seconds.

    Returns None for missing/unparseable values. Used so `classes`-dict
    exemplars (which carry `created_at` but no numeric `timestamp`) get a
    real age instead of the 0.0 default that made cap eviction always drop
    them first.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        from datetime import datetime, timezone

        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _exemplar_dedup_key(ex: Dict[str, Any], class_id: int) -> Optional[Tuple[Any, ...]]:
    """Stable identity for a `classes`-dict exemplar, or None if it has none.

    The old load() dedup compared only `created_at`: exemplars written by
    add_*() carry NO `created_at`, so None == None made every `classes`
    exemplar of an already-present class look like a duplicate (whole
    `classes` lists silently dropped, 400 -> 1 in measurement), and two
    wizard exemplars saved within the same second (second-precision
    `created_at`) collided even with different content. A record is now a
    duplicate only when class + created_at + visual content all match;
    records without `created_at` are never deduped.
    """
    created = ex.get("created_at")
    if created is None:
        return None
    hist = ex.get("hsv_hist") or []
    if hist:
        content: Any = tuple(hist)
    else:
        content = (
            tuple(ex.get("hist_h") or []),
            tuple(ex.get("hist_s") or []),
            tuple(ex.get("hist_v") or []),
            ex.get("aspect_ratio"),
            tuple(ex.get("mean_hsv") or []),
        )
    return (int(class_id), str(created), content)


# D-2: the negative-list clean rules, in exactly ONE place. sanitize()
# and _sanitize_classes_dict() used to carry different negative rules
# (the material-NAME purge lived only in sanitize's loop), so save()
# and load() could disagree about the same record. Both now call this.
_NEGATIVE_MATERIAL_NAME_TERMS: Tuple[str, ...] = (
    "aria", "mela", "targena", "orichalcum", "gem", "tree", "ore_", "herb_",
)
_NEGATIVE_3D_SOURCE_TERMS: Tuple[str, ...] = (
    "3d_world", "world_resource", "chromatic_salient", "crest",
)


def _is_clean_negative_record(ex: Any) -> bool:
    """Single-record negative-list predicate (shared by all purge paths)."""
    if not isinstance(ex, dict):
        return False
    cid = _safe_int(ex.get("class_id", -1))
    if cid is None:
        return False
    cname = str(ex.get("class_name", "")).lower()
    source = str(ex.get("source", "")).lower()
    if _is_poisoned_essence_name(cname):
        return False
    # P1-1: the negative list also admits the internal slot-state ids
    # (e.g. -100 "empty" from the skill bar): a rejected slot crop is
    # suppression signal, not a class — it must survive add/load/
    # sanitize instead of being purged as "non-canonical". Positives
    # stay canonical-only (see add_positive_exemplar Gate 1).
    if cid not in CANONICAL_CLASSES and cid not in INTERNAL_SLOT_STATE_IDS:
        return False
    # Never keep gatherable material classes — or material NAMES — in
    # negative exemplars (a rejected material box is ground, cid 15).
    if cid in SUPER_CATEGORY_3D_MATERIALS:
        return False
    if any(k in cname for k in _NEGATIVE_MATERIAL_NAME_TERMS):
        return False
    if cid >= 16 and any(term in source for term in _NEGATIVE_3D_SOURCE_TERMS):
        return False
    return True


def _exemplar_is_clean(ex: Any, negative: bool = False) -> bool:
    """Single-record version of the sanitize() rules (no counting)."""
    if not isinstance(ex, dict):
        return False
    if negative:
        return _is_clean_negative_record(ex)
    cid = _safe_int(ex.get("class_id", -1))
    if cid is None:
        return False
    cname = str(ex.get("class_name", "")).lower()
    source = str(ex.get("source", "")).lower()
    if _is_poisoned_essence_name(cname):
        return False
    if cid not in CANONICAL_CLASSES:
        return False
    if cid >= 16 and any(
        term in source for term in ("3d_world", "world_resource", "chromatic_salient", "crest")
    ):
        return False
    return True


def _sanitize_classes_dict(classes: Any) -> Dict[str, Any]:
    """
    Return a cleaned copy of a persisted `classes` dict: drop non-canonical
    / essence-named class entries and purge poisoned exemplars inside each
    class's positive/negative exemplar lists. Used by save() so the
    `classes` dict is never written back verbatim.
    """
    if not isinstance(classes, dict):
        return {}
    clean: Dict[str, Any] = {}
    for key, info in classes.items():
        kid = _safe_int(key)
        if kid is None or kid not in CANONICAL_CLASSES:
            continue
        if not isinstance(info, dict):
            continue
        cname = str(info.get("class_name", "")).lower()
        if _is_poisoned_essence_name(cname):
            continue
        item = dict(info)
        for list_key, negative in (("positive_exemplars", False), ("negative_exemplars", True)):
            raw = item.get(list_key)
            if isinstance(raw, list):
                item[list_key] = [e for e in raw if _exemplar_is_clean(e, negative=negative)]
        clean[str(key)] = item
    return clean


class ExemplarMemory:
    """
    Active Few-Shot Exemplar Memory for YOLO Auto-Labeler.

    Maintains positive and negative exemplars persisted in `learned_exemplars.json`.
    Learns immediately from human approve/reject actions in YOLO Factory Dialog.
    """

    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        max_positive_per_class: int = 40,
        max_negatives: int = 150,
        suppress_neg_threshold: float = SUPPRESS_NEG_THRESHOLD,
        reinforce_pos_threshold: float = REINFORCE_POS_THRESHOLD,
        # Default 0.88: the strict cosine threshold refine_candidate() has
        # always enforced for class repair. (This used to be a dead config:
        # the parameter was stored but refine_candidate() hardcoded 0.88.)
        reassign_pos_threshold: float = REASSIGN_POS_THRESHOLD,
    ) -> None:
        self.json_path = Path(json_path) if json_path is not None else DEFAULT_EXEMPLAR_PATH
        self.max_positive_per_class = max_positive_per_class
        self.max_negatives = max_negatives
        self.suppress_neg_threshold = suppress_neg_threshold
        self.reinforce_pos_threshold = reinforce_pos_threshold
        self.reassign_pos_threshold = reassign_pos_threshold

        self._lock = threading.RLock()
        self._positive: List[Dict[str, Any]] = []
        self._negative: List[Dict[str, Any]] = []
        # K-2/Y-1: quarantined records — real crops whose class/state
        # could not be determined (stateless skill_slot, generic
        # world_material). Kept on disk for audit / future re-labeling,
        # never in the matching matrices.
        self._quarantined: List[Dict[str, Any]] = []
        # Y-5: last load failure (None when the last load succeeded).
        # A corrupt/missing file used to build a silent empty memory.
        self._load_error: Optional[str] = None
        # O-5: content hash of the loaded/saved memory (not mtime).
        self._revision: str = ""
        # Dilim-30 B-12: True when the last save() could not preserve
        # the old file's `classes` dict (unreadable old file).
        self._last_save_dropped_classes: bool = False
        self._last_loaded_mtime: float = -1.0
        # Poison dropped from memory by load() (disk still holds it until
        # a sanitize(persist=True) runs); folded into the next sanitize()
        # result so the purge is reported exactly once.
        self._load_purged_positive: int = 0
        self._load_purged_negative: int = 0

        # Vectorized NumPy matching matrices (N, D) pre-normalized
        self._pos_matrix: Optional[np.ndarray] = None
        self._pos_cids: Optional[np.ndarray] = None
        # Per-positive-exemplar super-category code, aligned with
        # _pos_matrix rows (see _SUPER_CATEGORY_CODE_BY_CLASS). Rebuilt
        # together with the matrices so match_positive() can mask by
        # category with one vectorized comparison.
        self._pos_super_cats: np.ndarray = np.zeros((0,), dtype=np.int8)
        self._neg_matrix: Optional[np.ndarray] = None
        self._neg_cids: Optional[np.ndarray] = None

        self.load()

    def _rebuild_matrices(self) -> None:
        """
        Rebuild pre-normalized (N, D) NumPy feature matrices for O(1) batch cosine similarity.
        Each row is pre-normalized to unit L2 norm so np.dot(matrix, feat_norm) yields cosine similarity.
        """
        with self._lock:
            if self._positive:
                vecs = [feature_dict_to_normalized_vector(ex) for ex in self._positive]
                mat = np.vstack(vecs).astype(np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self._pos_matrix = mat / np.maximum(norms, 1e-9)
                self._pos_cids = np.array([int(ex.get("class_id", -1)) for ex in self._positive], dtype=np.int32)
                # O-2: 22..26 are canonical classes too — the old clip to
                # 21 silently re-coded consumables/currency/player as HUD.
                self._pos_super_cats = _SUPER_CATEGORY_CODE_BY_CLASS[np.clip(self._pos_cids, 0, 26)]
            else:
                self._pos_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._pos_cids = np.zeros((0,), dtype=np.int32)
                self._pos_super_cats = np.zeros((0,), dtype=np.int8)

            if self._negative:
                vecs = [feature_dict_to_normalized_vector(ex) for ex in self._negative]
                mat = np.vstack(vecs).astype(np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self._neg_matrix = mat / np.maximum(norms, 1e-9)
                self._neg_cids = np.array([int(ex.get("class_id", -1)) for ex in self._negative], dtype=np.int32)
            else:
                self._neg_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._neg_cids = np.zeros((0,), dtype=np.int32)

    def sanitize(self, persist: bool = True) -> Dict[str, Any]:
        """
        Purge poisoned / invalid records and repair legacy names (Y-1).

        Positive list, per record in order:
        1. Degenerate (content-free) records are purged (K-3, positive
           list only — v12: the degenerate gate no longer applies to
           the negative list, so a flat gray rock negative survives).
        2. Records whose NAME says 'negative' move to the negative list
           (K-1) — the name is kept as the marker, never renamed away.
           (K-4 REDDEDİLDİ: add_negative_exemplar already renames the
           material remap to path_ground — no remap/purge contradiction
           exists in this code.)
        3. State/class-unknown legacy records move to quarantine (K-2 and
           the Y-1 'world_material' edge) — kept on disk for audit /
           future re-labeling, never in the matching matrices.
        4. Remaining records are canonicalized via
           class_taxonomy.sanitize_class_id_and_name (Y-1: legacy
           (18,'hp_potion')->(22,'hp_potion'), (7/8 prompt aliases)->
           (13,'skill_ui_prompt'), (20,'target_hp_bar')->(18,...)).
           Unresolvable records are purged.
        Negative list: the shared _is_clean_negative_record() rules (D-2).
        """
        with self._lock:
            purged_pos = 0
            purged_neg = 0
            moved_to_negative = 0
            quarantined_n = 0
            canonicalized = 0
            clean_pos = []
            clean_neg = []

            for ex in self._positive:
                if not isinstance(ex, dict):
                    purged_pos += 1
                    continue
                # 1. Degenerate first (positive list only, v12): a
                # content-free record carries no signal as a positive
                # prototype; as a negative it is valid suppression.
                if _is_degenerate_features(ex):
                    purged_pos += 1
                    continue
                # 2. Negative-named positive -> negative list (K-1).
                if _is_negative_named(ex.get("class_name")):
                    if _is_clean_negative_record(ex):
                        clean_neg.append(ex)
                        moved_to_negative += 1
                    else:
                        purged_pos += 1
                    continue
                # 3. Unknown state/class -> quarantine (K-2, Y-1 edge).
                if _is_state_unknown_skill_slot(ex.get("class_id"), ex.get("class_name")):
                    item = dict(ex)
                    item["quarantine_reason"] = "state_unknown_skill_slot"
                    self._quarantined.append(item)
                    quarantined_n += 1
                    continue
                if _is_class_unknown_material(ex.get("class_id"), ex.get("class_name")):
                    item = dict(ex)
                    item["quarantine_reason"] = "class_unknown_world_material"
                    self._quarantined.append(item)
                    quarantined_n += 1
                    continue
                # 4. Canonicalize (Y-1).
                if not _exemplar_is_clean(ex):
                    purged_pos += 1
                    continue
                try:
                    new_cid, new_cname = _canonicalize_exemplar_class(
                        ex.get("class_id"), ex.get("class_name")
                    )
                except Exception:
                    purged_pos += 1
                    continue
                item = dict(ex)
                try:
                    old_cid = int(ex.get("class_id", -1))
                except (TypeError, ValueError):
                    old_cid = -1
                if (new_cid, new_cname) != (old_cid, str(ex.get("class_name", ""))):
                    item["class_id"] = new_cid
                    item["class_name"] = new_cname
                    canonicalized += 1
                clean_pos.append(item)

            # Negative list: the shared rules (D-2) — the same predicate
            # _sanitize_classes_dict() uses, so save() and load() cannot
            # disagree about a record.
            for ex in self._negative:
                if not _is_clean_negative_record(ex):
                    purged_neg += 1
                    continue
                clean_neg.append(ex)

            # Poison dropped from memory by load() has not been reported
            # yet: fold it into this sanitize() result (the disk still
            # holds it until this call persists) and consume it, so a
            # second sanitize() reports 0. The same folding applies to
            # records load() moved to negative / canonicalized.
            purged_pos += getattr(self, "_load_purged_positive", 0)
            purged_neg += getattr(self, "_load_purged_negative", 0)
            moved_to_negative += getattr(self, "_load_moved_to_negative", 0)
            canonicalized += getattr(self, "_load_canonicalized", 0)
            quarantined_n += getattr(self, "_load_quarantined", 0)
            self._load_purged_positive = 0
            self._load_purged_negative = 0
            self._load_moved_to_negative = 0
            self._load_canonicalized = 0
            self._load_quarantined = 0

            self._positive = clean_pos
            self._negative = clean_neg
            self._rebuild_matrices()

            if persist:
                self.save()

            logger.info(
                "Sanitized exemplar memory: purged %d positive, %d negative "
                "(remaining: %d pos, %d neg, %d quarantined); moved %d to "
                "negative, canonicalized %d",
                purged_pos,
                purged_neg,
                len(self._positive),
                len(self._negative),
                len(self._quarantined),
                moved_to_negative,
                canonicalized,
            )
            return {
                "purged_positive": purged_pos,
                "purged_negative": purged_neg,
                "remaining_positive": len(self._positive),
                "remaining_negative": len(self._negative),
                "moved_to_negative": moved_to_negative,
                "quarantined": quarantined_n,
                "quarantined_total": len(self._quarantined),
                "canonicalized": canonicalized,
            }

    def check_reload(self, force: bool = False) -> bool:
        """
        Instant hot-reload check (<0.1ms).
        Inspects mtime of json_path; if updated on disk by Calibration Wizard,
        YOLO Factory, or Settings Node Purge, automatically reloads memory.
        """
        try:
            if not self.json_path.is_file():
                if self._last_loaded_mtime != -1.0:
                    with self._lock:
                        self._positive.clear()
                        self._negative.clear()
                        self._rebuild_matrices()
                        self._last_loaded_mtime = -1.0
                return False
            mtime = self.json_path.stat().st_mtime
            if force or mtime != self._last_loaded_mtime:
                return self.load()
            return True
        except Exception as exc:
            logger.debug("check_reload error on %s: %s", self.json_path, exc)
            return False

    def load(self) -> bool:
        """Load learned exemplars from disk if file exists and sanitize them."""
        with self._lock:
            if not self.json_path.is_file():
                self._positive.clear()
                self._negative.clear()
                self._quarantined.clear()
                self._load_error = None
                self._revision = ""
                self._rebuild_matrices()
                return False
            try:
                mtime = self.json_path.stat().st_mtime
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                raw_pos = data.get("positive", [])
                raw_neg = data.get("negative", [])
                raw_q = data.get("quarantined", [])
                # Top-level lists are untrusted disk input: drop non-dict
                # entries here; the sanitize() call below purges poisoned
                # (essence / non-canonical / 3D-sourced HUD / material
                # negative) records instead of loading them verbatim.
                pos_list = [e for e in raw_pos if isinstance(e, dict)] if isinstance(raw_pos, list) else []
                neg_list = [e for e in raw_neg if isinstance(e, dict)] if isinstance(raw_neg, list) else []
                q_list = [e for e in raw_q if isinstance(e, dict)] if isinstance(raw_q, list) else []

                # Also ingest from classes dict if present (e.g. from CalibrationWizard)
                classes_data = data.get("classes", {})
                # Set-based dedup (O(1) per record): the old `any(... created_at
                # ...)` scan was O(n^2) AND wrong (see _exemplar_dedup_key).
                seen_pos = {
                    k
                    for e in pos_list
                    if (k := _exemplar_dedup_key(e, _safe_int(e.get("class_id", -1), -1))) is not None
                }
                seen_neg = {
                    k
                    for e in neg_list
                    if (k := _exemplar_dedup_key(e, _safe_int(e.get("class_id", -1), -1))) is not None
                }
                if isinstance(classes_data, dict):
                    for cid_key, cinfo in classes_data.items():
                        if not isinstance(cinfo, dict):
                            continue
                        try:
                            c_id = int(cid_key) if str(cid_key).isdigit() else int(cinfo.get("class_id", 0))
                        except (TypeError, ValueError):
                            continue
                        c_name = str(cinfo.get("class_name", f"class_{c_id}"))
                        # Skip essence and non-canonical classes
                        if c_id not in CANONICAL_CLASSES or _is_poisoned_essence_name(c_name):
                            continue
                        for p_ex in cinfo.get("positive_exemplars", []):
                            if isinstance(p_ex, dict):
                                item = dict(p_ex)
                                item.setdefault("class_id", c_id)
                                item.setdefault("class_name", c_name)
                                if not isinstance(item.get("timestamp"), (int, float)):
                                    ts = _created_at_to_epoch(item.get("created_at"))
                                    if ts is not None:
                                        item["timestamp"] = ts
                                key = _exemplar_dedup_key(item, c_id)
                                if key is None or key not in seen_pos:
                                    if key is not None:
                                        seen_pos.add(key)
                                    pos_list.append(item)
                        for n_ex in cinfo.get("negative_exemplars", []):
                            if isinstance(n_ex, dict):
                                item = dict(n_ex)
                                item.setdefault("class_id", c_id)
                                item.setdefault("class_name", c_name)
                                if not isinstance(item.get("timestamp"), (int, float)):
                                    ts = _created_at_to_epoch(item.get("created_at"))
                                    if ts is not None:
                                        item["timestamp"] = ts
                                key = _exemplar_dedup_key(item, c_id)
                                if key is None or key not in seen_neg:
                                    if key is not None:
                                        seen_neg.add(key)
                                    neg_list.append(item)

                self._positive = pos_list
                self._negative = neg_list
                self._quarantined = q_list
                self._last_loaded_mtime = mtime
                self._load_error = None

                # Sanitize on load (no persist): poisoned top-level records
                # must never reach the in-memory matrices / matching paths.
                # Reset the pending counts first so a reload does not
                # double-count poison already reported by an earlier load.
                self._load_purged_positive = 0
                self._load_purged_negative = 0
                self._load_moved_to_negative = 0
                self._load_canonicalized = 0
                self._load_quarantined = 0
                _load_res = self.sanitize(persist=False)
                self._load_purged_positive = _load_res["purged_positive"]
                self._load_purged_negative = _load_res["purged_negative"]
                self._load_moved_to_negative = _load_res["moved_to_negative"]
                self._load_canonicalized = _load_res["canonicalized"]
                self._load_quarantined = _load_res["quarantined"]
                # O-5: content revision of the memory just loaded.
                self._revision = _compute_revision(
                    self._positive, self._negative, self._quarantined
                )

                logger.info(
                    "Loaded exemplar memory from %s (%d positive, %d negative, %d quarantined)",
                    self.json_path,
                    len(self._positive),
                    len(self._negative),
                    len(self._quarantined),
                )
                return True
            except Exception as exc:
                # Y-5: a corrupt/unreadable file used to log a warning and
                # return False while the caller kept matching against
                # whatever stale memory was already in RAM. The loss is
                # now loud (error log + _load_error flag, surfaced by
                # get_stats) and the in-memory state is cleared so no
                # caller can mistake stale exemplars for the file's.
                logger.error("Failed to load exemplar memory from %s: %s", self.json_path, exc)
                self._load_error = str(exc)
                self._positive.clear()
                self._negative.clear()
                self._quarantined.clear()
                self._revision = ""
                self._rebuild_matrices()
                return False

    def save(self) -> bool:
        """Atomically persist learned exemplars to disk."""
        with self._lock:
            try:
                self.json_path.parent.mkdir(parents=True, exist_ok=True)
                existing_classes = {}
                self._last_save_dropped_classes = False
                if self.json_path.is_file():
                    try:
                        with open(self.json_path, "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            if isinstance(old_data, dict) and "classes" in old_data:
                                # Never write the classes dict back verbatim:
                                # drop non-canonical / essence entries and
                                # purge poisoned exemplars inside each class.
                                existing_classes = _sanitize_classes_dict(old_data["classes"])
                    except Exception as exc:
                        # Dilim-30 B-12: the old file's `classes` dict used
                        # to be dropped silently (except: pass) while save()
                        # still reported success. The loss is now visible:
                        # warning log + _last_save_dropped_classes flag
                        # (surfaced by get_stats). The save itself still
                        # proceeds — the top-level lists are the live data.
                        logger.warning(
                            "save(): could not read old %s — its 'classes' "
                            "dict will be dropped from the new file: %s",
                            self.json_path,
                            exc,
                        )
                        self._last_save_dropped_classes = True
                tmp_path = self.json_path.with_suffix(".tmp")
                # O-5: content revision (not mtime) of what is being written.
                self._revision = _compute_revision(
                    self._positive, self._negative, self._quarantined
                )
                payload = {
                    "version": 1,
                    "updated_at": time.time(),
                    "revision": self._revision,
                    "positive_count": len(self._positive),
                    "negative_count": len(self._negative),
                    "quarantined_count": len(self._quarantined),
                    "positive": self._positive,
                    "negative": self._negative,
                    "quarantined": self._quarantined,
                }
                if existing_classes:
                    payload["classes"] = existing_classes
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                if tmp_path.exists():
                    try:
                        tmp_path.replace(self.json_path)
                    except OSError:
                        time.sleep(0.02)
                        try:
                            tmp_path.replace(self.json_path)
                        except OSError:
                            shutil.copyfile(tmp_path, self.json_path)
                            try:
                                tmp_path.unlink()
                            except OSError:
                                pass
                try:
                    self._last_loaded_mtime = self.json_path.stat().st_mtime
                except OSError:
                    pass
                return True
            except Exception as exc:
                logger.error("Failed to save exemplar memory to %s: %s", self.json_path, exc)
                return False

    def add_positive_exemplar(
        self,
        crop: np.ndarray,
        class_id: int,
        class_name: str,
        source: str = "ui_approval",
    ) -> bool:
        """
        Register a user-approved / ground-truth crop as a positive exemplar for `class_id`.
        Enforces canonical classes and disallows non-canonical classes for 3D world resources.
        Returns True only when a NEW exemplar is stored; a near-duplicate
        refreshes the existing record and returns False.
        """
        feats = extract_crop_features(crop)
        if feats is None:
            return False

        # K-3: a content-free crop (flat single color, no edges,
        # near-square) teaches nothing — reject before it can pollute
        # a class with a degenerate prototype.
        if _is_degenerate_features(feats):
            logger.warning(
                "Rejected degenerate positive exemplar %s/%s (flat crop, "
                "no texture) — no visual information to learn.",
                class_id,
                class_name,
            )
            return False

        cid = int(class_id)
        cname = str(class_name or f"class_{cid}")

        # K-1: a 'negative'-named record belongs on the negative list —
        # adding it as positive would make the matcher REINFORCE a
        # rejected box. The name is the marker: it is checked BEFORE
        # canonicalization, which would otherwise rename it away.
        if _is_negative_named(cname):
            logger.warning(
                "Rejected positive exemplar %d (%s): name says 'negative' — "
                "use add_negative_exemplar instead.",
                cid,
                cname,
            )
            return False

        # Gate 1 (pre-canonicalization): an unresolved `essence_*` catalog
        # alias is poison and must be judged on the RAW name. Canonicalization
        # below (Priority 4: a valid canonical id falls back to its canonical
        # name) would rename e.g. (22, 'essence_ruby') to (22, 'hp_potion'),
        # so the post-canonicalization Gate-1 check would never see the
        # poison (Decision 1: poison is detected by class NAME, never by ID
        # range). Same ordering doctrine as K-1 above: name-based markers are
        # checked BEFORE canonicalization renames them away. sanitize()
        # already does this — it purges on the raw stored name before
        # canonicalizing — so this gate only closes the add-time hole.
        if _is_poisoned_essence_name(cname):
            logger.warning(
                "Rejected positive exemplar %d (%s): unresolved essence_* "
                "catalog alias — poison, never a canonical class.",
                cid,
                cname,
            )
            return False

        # K-2 / Y-1 edge: stateless skill_slot and generic world_material
        # carry no decidable class/state — they are quarantined by
        # sanitize(), never learned as positives. Reject at the gate so
        # new ones cannot be minted (checked before canonicalization,
        # which would remap them into a false state/class).
        if _is_state_unknown_skill_slot(cid, cname) or _is_class_unknown_material(cid, cname):
            logger.warning(
                "Rejected positive exemplar %d (%s): state/class unknown — "
                "quarantine only, not learnable as-is.",
                cid,
                cname,
            )
            return False

        # Y-1: canonicalize legacy (cid, name) before the gates below,
        # so a legacy (18,'hp_potion') is judged as (22,'hp_potion').
        try:
            cid, cname = _canonicalize_exemplar_class(cid, cname)
        except Exception:
            logger.warning(
                "Rejected positive exemplar %s (%s): class/name does not "
                "resolve to any canonical class.",
                class_id,
                class_name,
            )
            return False

        # Gate 1: Disallow non-canonical classes or essence classes
        if cid not in CANONICAL_CLASSES or _is_poisoned_essence_name(cname):
            logger.warning(
                "Rejected non-canonical positive exemplar class %d (%s) - only canonical classes (0..26) allowed.",
                cid,
                cname,
            )
            return False

        # Gate 2: Disallow non-canonical classes for 3D world resources
        src_lower = str(source).lower()
        if any(term in src_lower for term in ("3d", "world", "crest", "harvest", "gather")):
            if not is_3d_world_class(cid):
                logger.warning(
                    "Rejected non-3D class %d (%s) for 3D world resource positive exemplar (source=%s).",
                    cid,
                    cname,
                    source,
                )
                return False

        q_vec = feature_dict_to_normalized_vector(feats)

        with self._lock:
            # Check for near-duplicates in this class using vectorized dot product
            if self._pos_matrix is not None and len(self._pos_matrix) > 0 and self._pos_cids is not None:
                mask = (self._pos_cids == cid)
                if np.any(mask):
                    sims = np.dot(self._pos_matrix[mask], q_vec)
                    if float(np.max(sims)) >= 0.96:
                        matching_indices = np.where(mask)[0]
                        best_match_idx = matching_indices[int(np.argmax(sims))]
                        self._positive[best_match_idx]["timestamp"] = time.time()
                        self._positive[best_match_idx]["source"] = source
                        # Duplicate, not a new exemplar: refresh the existing
                        # record but report False so callers do not count it
                        # as "added".
                        logger.info(
                            "Duplicate positive exemplar for class %d ignored (sim=%.2f); refreshed existing record.",
                            cid,
                            float(np.max(sims)),
                        )
                        return False

            entry = {
                "class_id": cid,
                "class_name": cname,
                "hsv_hist": feats["hsv_hist"],
                # O-3: keep the 1D histograms too — the calibration wizard
                # (and test_w48 test_3) expects hist_h/hist_s/hist_v on
                # stored exemplars; the 432-bin path is the primary one.
                "hist_h": feats.get("hist_h", []),
                "hist_s": feats.get("hist_s", []),
                "hist_v": feats.get("hist_v", []),
                "aspect_ratio": feats["aspect_ratio"],
                "texture": feats["texture"],
                "mean_hsv": feats.get("mean_hsv", []),
                "timestamp": time.time(),
                "source": source,
            }
            self._positive.append(entry)

            # Cap positive exemplars per class to prevent unbounded bloat
            cls_entries = [e for e in self._positive if int(e.get("class_id", -1)) == cid]
            if len(cls_entries) > self.max_positive_per_class:
                oldest = min(cls_entries, key=lambda x: float(x.get("timestamp", 0.0)))
                # Y-4: the eviction used to be silent — the oldest
                # exemplar simply vanished. Log what was dropped so a
                # class losing its best prototype is visible.
                logger.warning(
                    "Positive cap hit for class %d (%s): evicting oldest "
                    "exemplar (ts=%.1f, source=%s); cap=%d.",
                    cid,
                    cname,
                    float(oldest.get("timestamp", 0.0)),
                    oldest.get("source"),
                    self.max_positive_per_class,
                )
                self._positive.remove(oldest)

            self._rebuild_matrices()
            logger.info(
                "Added positive exemplar for class %d (%s) via %s. Total positive: %d",
                cid,
                cname,
                source,
                len(self._positive),
            )
            # O-4: a failed save used to be ignored while the caller was
            # told the exemplar was stored. The entry is in memory, but
            # not durable — report False.
            if not self.save():
                return False
            return True

    def add_negative_exemplar(
        self,
        crop: np.ndarray,
        class_id: int,
        class_name: str,
        source: str = "ui_rejection",
    ) -> bool:
        """
        Register a user-rejected / false-positive crop as a negative exemplar.
        Enforces canonical classes and disallows non-canonical classes for 3D world resources.
        Returns True only when a NEW exemplar is stored; a near-duplicate
        refreshes the existing record and returns False.
        """
        feats = extract_crop_features(crop)
        if feats is None:
            return False

        # K-3 (v12 daraltma): dejenere kapısı NEGATİF tarafta uygulanmaz.
        # Tek renk / dokusuz bir yama (örn. düz gri kaya) pozitif prototip
        # olarak içeriksizdir, ama negatif olarak geçerli bir bastırma
        # sinyalidir — "bu gri yama bir kaynak değil" öğretisi, doku
        # gerektirmez. Daha önce bu kapı düz gri kaya negatifini
        # reddediyordu ("no visual information to learn"); Selçuk kararı
        # (10 Kasım) ile kabul edilir. Pozitif taraf (add_positive_exemplar
        # + sanitize pozitif listesi) kapıyı aynen korur; ayrıca
        # sanitize() negatifleri bu kapıyla hiç temizlemiyordu, o yüzden
        # diskteki dejenere negatif kayıtlar (örn. class 15
        # wrong_node_purge) load/sanitize'den sağ çıkar — geri getirme
        # için diskte ayrıca bir işlem gerekmez.
        # (Not: dedup / zehir / kanonik kapılar aşağıda aynen çalışır.)

        cid = int(class_id)
        cname = str(class_name or f"class_{cid}")

        # Gate 1 (pre-canonicalization, F-B2): an unresolved `essence_*`
        # catalog alias is poison and must be judged on the RAW name —
        # the same ordering doctrine as the positive side (F-B) and K-1.
        # Canonicalization below (Priority 4: a valid canonical id falls
        # back to its canonical name) would rename e.g. (22,
        # 'essence_ruby') to (22, 'hp_potion'), and the 3D-material
        # remap below would rename e.g. (5, 'essence_spoofed') to
        # (15, 'path_ground'), so the post-canonicalization Gate-1 check
        # would never see the poison (Decision 1: poison is detected by
        # class NAME, never by ID range). A poisoned negative exemplar
        # would teach the matcher to reject legitimate hp_potion
        # detections, so it is rejected here, before any renaming.
        if _is_poisoned_essence_name(cname):
            logger.warning(
                "Rejected negative exemplar %d (%s): unresolved essence_* "
                "catalog alias — poison, never a canonical class.",
                cid,
                cname,
            )
            return False

        # Protect keep/material classes: if a box was rejected as a false positive,
        # it is ground/terrain, NOT an exemplar of the resource class itself.
        if cid in SUPER_CATEGORY_3D_MATERIALS:
            cid = 15
            cname = "path_ground"

        # Y-1: canonicalize legacy (cid, name) — except a 'negative'-named
        # record, whose name is the provenance marker (K-1) and must be
        # kept verbatim on the negative list. Same for a 'wrong_node_*'
        # marker (v19/F12): its cid 15 is a storage bucket, not a
        # path_ground claim — canonicalizing it would rewrite
        # (15, "wrong_node_<reason>") to (15, "path_ground") and destroy
        # the only trace that the patch content was unknown (P1-8/K-21).
        if not (_is_negative_named(cname) or _is_wrong_node_marker_name(cname)):
            try:
                cid, cname = _canonicalize_exemplar_class(cid, cname)
            except Exception:
                logger.warning(
                    "Rejected negative exemplar %s (%s): class/name does "
                    "not resolve to any canonical class.",
                    class_id,
                    class_name,
                )
                return False

        # P1-1: the negative list also admits the internal slot-state ids
        # (e.g. -100 "empty" from the skill bar — see
        # _is_clean_negative_record). Positives stay canonical-only.
        if (
            cid not in CANONICAL_CLASSES
            and cid not in INTERNAL_SLOT_STATE_IDS
        ) or _is_poisoned_essence_name(cname):
            logger.warning(
                "Rejected non-canonical negative exemplar class %d (%s) - only canonical classes (0..26) and internal slot-state ids allowed.",
                cid,
                cname,
            )
            return False

        src_lower = str(source).lower()
        if any(term in src_lower for term in ("3d", "world", "crest", "harvest", "gather")):
            if not is_3d_world_class(cid):
                logger.warning(
                    "Rejected non-3D class %d (%s) for 3D world resource negative exemplar (source=%s).",
                    cid,
                    cname,
                    source,
                )
                return False

        q_vec = feature_dict_to_normalized_vector(feats)

        with self._lock:
            if self._neg_matrix is not None and len(self._neg_matrix) > 0:
                sims = np.dot(self._neg_matrix, q_vec)
                if float(np.max(sims)) >= 0.96:
                    best_match_idx = int(np.argmax(sims))
                    self._negative[best_match_idx]["timestamp"] = time.time()
                    self._negative[best_match_idx]["source"] = source
                    # Duplicate, not a new exemplar: refresh the existing
                    # record but report False so callers do not count it
                    # as "added" (same rule as add_positive_exemplar).
                    logger.info(
                        "Duplicate negative exemplar for class %d ignored (sim=%.2f); refreshed existing record.",
                        cid,
                        float(np.max(sims)),
                    )
                    return False

            entry = {
                "class_id": cid,
                "class_name": cname,
                "hsv_hist": feats["hsv_hist"],
                "aspect_ratio": feats["aspect_ratio"],
                "texture": feats["texture"],
                "mean_hsv": feats.get("mean_hsv", []),
                # O-3: keep the 1D histograms (see add_positive_exemplar).
                "hist_h": feats.get("hist_h", []),
                "hist_s": feats.get("hist_s", []),
                "hist_v": feats.get("hist_v", []),
                "timestamp": time.time(),
                "source": source,
            }
            self._negative.append(entry)

            if len(self._negative) > self.max_negatives:
                # Evict the oldest by timestamp, like the positive cap —
                # NOT pop(0): insertion order is not timestamp order once
                # duplicates refresh an old record's timestamp.
                oldest = min(self._negative, key=lambda e: float(e.get("timestamp", 0.0)))
                # Y-4: log the silent eviction (see add_positive_exemplar).
                logger.warning(
                    "Negative cap hit: evicting oldest exemplar (class %s, "
                    "ts=%.1f, source=%s); cap=%d.",
                    oldest.get("class_id"),
                    float(oldest.get("timestamp", 0.0)),
                    oldest.get("source"),
                    self.max_negatives,
                )
                self._negative.remove(oldest)

            self._rebuild_matrices()
            logger.info(
                "Added negative exemplar for false-positive %d (%s) via %s. Total negative: %d",
                cid,
                cname,
                source,
                len(self._negative),
            )
            # O-4: report a failed save (see add_positive_exemplar).
            if not self.save():
                return False
            return True

    def remove_negative_exemplar(
        self,
        index: int,
        *,
        persist: bool = True,
    ) -> bool:
        """Remove one negative exemplar by index — undo a wrong teach (K-19).

        Yanlış negatif sonsuza dek kalmasın: ``get_negative_exemplars()``
        ile bulunan kaydın indeksi verilir, kayıt listeden çıkarılır,
        eşleşme matrisleri yeniden kurulur ve (``persist=True`` ise) disk
        yazılır. Disk yazımı başarısız olursa kayıt bellekte AYNI indekse
        geri yüklenir (bellek/disk tutarsız kalmaz) ve False döner.
        Aralık dışı indeks False döner, istisna fırlatmaz.
        """
        return self._remove_exemplar("negative", index, persist=persist)

    def remove_positive_exemplar(
        self,
        index: int,
        *,
        persist: bool = True,
    ) -> bool:
        """Same undo path for a positive exemplar (symmetry, K-19)."""
        return self._remove_exemplar("positive", index, persist=persist)

    def _remove_exemplar(
        self,
        kind: str,
        index: int,
        *,
        persist: bool,
    ) -> bool:
        self.check_reload()
        with self._lock:
            target = self._negative if kind == "negative" else self._positive
            try:
                idx = int(index)
            except (TypeError, ValueError):
                return False
            if idx < 0 or idx >= len(target):
                return False
            removed = target.pop(idx)
            self._rebuild_matrices()
            if persist and not self.save():
                # Disk yazılamadı: belleği diske geri döndür.
                target.insert(idx, removed)
                self._rebuild_matrices()
                logger.warning(
                    "Remove %s exemplar %d failed to persist; record "
                    "restored in memory.",
                    kind,
                    idx,
                )
                return False
            logger.info(
                "Removed %s exemplar %d (class %s, source=%s). "
                "Total %s: %d",
                kind,
                idx,
                removed.get("class_id"),
                removed.get("source"),
                kind,
                len(target),
            )
            return True

    def match_negative(
        self,
        crop: np.ndarray,
        class_id: Optional[int] = None,
        feats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Find highest similarity to any learned negative exemplar.
        Uses vectorized NumPy matrix multiplication (np.dot(matrix, feat_norm)) for 50x faster batch matching.
        Returns: (max_similarity, best_matching_negative_exemplar or None)

        `feats`: pre-extracted feature dict for `crop` (see
        extract_crop_features). When given, the crop is NOT re-extracted:
        refine_candidate() extracts once and reuses the same features for
        the negative + positive matches instead of extracting 3 times.
        When None, features are extracted here (1D hists skipped: the
        vectorized path never uses them).
        """
        self.check_reload()
        if feats is None:
            feats = extract_crop_features(crop, include_1d_hists=False)
        if feats is None or not self._negative:
            return 0.0, None

        q_vec = feature_dict_to_normalized_vector(feats)

        with self._lock:
            if self._neg_matrix is None or len(self._neg_matrix) == 0:
                return 0.0, None

            # O(1) Batch Cosine Similarity via single matrix-vector dot product
            sims = np.dot(self._neg_matrix, q_vec)

            # If class_id specified, cross-class negative match is scaled by 0.90
            if class_id is not None and self._neg_cids is not None:
                scale = np.where(self._neg_cids == int(class_id), 1.0, 0.90)
                sims = sims * scale

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_ex = self._negative[best_idx]
            return round(best_sim, 4), best_ex

    def match_positive(
        self,
        crop: np.ndarray,
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        feats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
        """
        Find best matching learned positive exemplar across classes.
        Uses vectorized NumPy matrix multiplication (np.dot(matrix, feat_norm)) for 50x faster batch matching.
        Supports category-locked matching via same_category_as or allowed_classes.
        Returns: (max_similarity, best_class_id, best_class_name, best_exemplar)

        `feats`: pre-extracted feature dict for `crop`; when given, the
        crop is NOT re-extracted (see match_negative). When None, features
        are extracted here (1D hists skipped: the vectorized path never
        uses them).
        """
        self.check_reload()
        if feats is None:
            feats = extract_crop_features(crop, include_1d_hists=False)
        if feats is None or not self._positive:
            return 0.0, None, None, None

        q_vec = feature_dict_to_normalized_vector(feats)

        with self._lock:
            if self._pos_matrix is None or len(self._pos_matrix) == 0:
                return 0.0, None, None, None

            # Dot only the rows the mask lets through. Exactly the same
            # winner as the old full dot + np.where(mask, sims, -1.0):
            # masked-out rows could never win (their sims were replaced
            # by -1.0, below every real cosine sim >= 0), and idx keeps
            # original order so argmax tie-breaking is unchanged. The
            # full 500-row dot cost ~0.6 ms/call on this VM; a
            # category-locked dot over ~23 rows costs ~0.04 ms (D50 bench).
            valid_mask = None
            if same_category_as is not None and self._pos_cids is not None:
                super_cat = get_super_category(same_category_as)
                if super_cat is None:
                    return 0.0, None, None, None
                # Vectorized category mask from the precomputed per-row
                # super-category codes (built in _rebuild_matrices). The
                # old code called get_super_category() once per exemplar
                # per query here (~3 ms/call at N=500; see D50 bench).
                valid_mask = self._pos_super_cats == _SUPER_CATEGORY_CODES[super_cat]
            elif allowed_classes is not None and self._pos_cids is not None:
                allowed_set = set(allowed_classes)
                valid_mask = np.isin(self._pos_cids, list(allowed_set))

            if valid_mask is not None:
                if not np.any(valid_mask):
                    return 0.0, None, None, None
                idx = np.where(valid_mask)[0]
                sims = np.dot(self._pos_matrix[idx], q_vec)
                best_sub = int(np.argmax(sims))
                best_idx = int(idx[best_sub])
                best_sim = float(sims[best_sub])
            else:
                # O(1) Batch Cosine Similarity
                sims = np.dot(self._pos_matrix, q_vec)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
            if best_sim < 0.0:
                return 0.0, None, None, None

            best_ex = self._positive[best_idx]
            best_cid = int(best_ex.get("class_id", 0))
            best_cname = str(best_ex.get("class_name", ""))
            return round(best_sim, 4), best_cid, best_cname, best_ex

    def is_negative_patch(
        self,
        crop: np.ndarray,
        class_id: Optional[int] = None,
        threshold: Optional[float] = None,
        feats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """
        Fast query checking if crop matches any learned negative exemplar.
        Returns: (is_negative: bool, similarity: float, matching_exemplar or None)

        `feats`: pre-extracted feature dict for `crop` (see
        extract_crop_features). When given, the crop is NOT re-extracted —
        callers that query the same crop several times (SkillBarReader:
        negative + class 20/21 + classify) extract once and reuse it.
        """
        thresh = float(threshold if threshold is not None else self.suppress_neg_threshold)
        sim, ex = self.match_negative(crop, class_id=class_id, feats=feats)
        return (sim >= thresh, sim, ex)

    def classify_patch(
        self,
        crop: np.ndarray,
        min_sim: float = 0.70,
        feats: Optional[Dict[str, Any]] = None,
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[str], float]:
        """
        Fast query checking if crop matches any learned positive exemplar.
        Returns: (class_id or None, class_name or None, similarity)

        `feats`: pre-extracted feature dict for `crop`; when given, the
        crop is NOT re-extracted (see is_negative_patch).

        `allowed_classes` / `same_category_as`: forwarded to match_positive
        so callers (MaterialScanner family lock) can restrict the winner
        to one material family instead of any 3d_materials exemplar.
        """
        sim, cid, cname, _ = self.match_positive(
            crop,
            feats=feats,
            allowed_classes=allowed_classes,
            same_category_as=same_category_as,
        )
        if sim >= float(min_sim) and cid is not None:
            return cid, cname, sim
        return None, None, float(sim)

    def match_class(
        self,
        crop: np.ndarray,
        class_id: int,
        feats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Find highest similarity to learned positive exemplars belonging specifically to `class_id`.
        Uses vectorized NumPy matrix matching.
        Returns: (max_similarity: float, best_matching_exemplar or None)

        `feats`: pre-extracted feature dict for `crop` (see
        is_negative_patch). Only the vectorized path consumes it
        (feature_dict_to_normalized_vector reads hsv_hist / aspect_ratio
        / texture, never the 1D hists), so feats extracted with
        include_1d_hists=False give the identical similarity.
        """
        self.check_reload()
        if feats is None:
            feats = extract_crop_features(crop)
        if feats is None or not self._positive:
            return 0.0, None

        q_vec = feature_dict_to_normalized_vector(feats)
        cid = int(class_id)

        with self._lock:
            if self._pos_matrix is None or len(self._pos_matrix) == 0 or self._pos_cids is None:
                return 0.0, None

            mask = (self._pos_cids == cid)
            if not np.any(mask):
                return 0.0, None

            # Dot only the rows of this class (same winner as the old full
            # dot + np.where(mask, sims, -1.0); see match_positive).
            idx = np.where(mask)[0]
            sims = np.dot(self._pos_matrix[idx], q_vec)
            best_sub = int(np.argmax(sims))
            best_idx = int(idx[best_sub])
            best_sim = float(sims[best_sub])
            if best_sim < 0.0:
                return 0.0, None
            return round(best_sim, 4), self._positive[best_idx]

    def is_class_patch(
        self,
        crop: np.ndarray,
        class_id: int,
        threshold: float = 0.70,
    ) -> Tuple[bool, float]:
        """
        Check if crop matches positive exemplars of class_id with at least `threshold` similarity.
        Returns: (is_match: bool, similarity: float)
        """
        sim, _ = self.match_class(crop, class_id=class_id)
        return (sim >= float(threshold), float(sim))

    def get_positive_exemplars(self, class_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of positive exemplars, optionally filtered by class_id."""
        self.check_reload()
        with self._lock:
            if class_id is None:
                return list(self._positive)
            cid = int(class_id)
            return [e for e in self._positive if int(e.get("class_id", -1)) == cid]

    def get_negative_exemplars(self, class_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of negative exemplars, optionally filtered by class_id."""
        self.check_reload()
        with self._lock:
            if class_id is None:
                return list(self._negative)
            cid = int(class_id)
            return [e for e in self._negative if int(e.get("class_id", -1)) == cid]

    def get_quarantined_exemplars(self, class_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Quarantined records (K-2/Y-1): kept on disk, never matched.

        Records whose class/state could not be determined (stateless
        skill_slot, generic world_material). Returned as copies so
        callers cannot mutate the in-memory quarantine.
        """
        self.check_reload()
        with self._lock:
            if class_id is None:
                return [dict(e) for e in self._quarantined]
            cid = int(class_id)
            return [dict(e) for e in self._quarantined if int(e.get("class_id", -1)) == cid]

    def has_positive_exemplars(self, class_id: Optional[int] = None) -> bool:
        """Check if any positive exemplars exist (optionally for class_id)."""
        self.check_reload()
        with self._lock:
            if class_id is None:
                return len(self._positive) > 0
            cid = int(class_id)
            return any(int(e.get("class_id", -1)) == cid for e in self._positive)

    def has_negative_exemplars(self, class_id: Optional[int] = None) -> bool:
        """Check if any negative exemplars exist (optionally for class_id)."""
        self.check_reload()
        with self._lock:
            if class_id is None:
                return len(self._negative) > 0
            cid = int(class_id)
            return any(int(e.get("class_id", -1)) == cid for e in self._negative)

    def reload_now(self) -> bool:
        """Forces an immediate reload of exemplar memory from disk."""
        return self.check_reload(force=True)

    def refine_candidate(
        self,
        frame: np.ndarray,
        candidate: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Category-Locked Exemplar Memory Gate:
        1. Crop candidate box from frame.
        2. Test against negative exemplars -> suppress if similarity >= 0.82.
        3. Test against positive exemplars ->
           - Same class: reinforce confidence.
           - Class repair (repaired_by_exemplar):
             ONLY allowed if:
             a) Same super-category (3D world materials 0..8, 12 only change among 0..8, 12;
                minimap symbols 9, 11 only among 9, 11; HUD symbols 13, 16..21 only among HUD).
                Never allow 3D box to be repaired into essence_* or HUD classes!
             b) Feature cosine similarity >= reassign_pos_threshold (0.88).
             c) HSV color distribution consistency verified.
        Returns: (keep: bool, refined_candidate: dict)
        """
        self.check_reload()
        crop = extract_box_crop(frame, candidate)
        if crop is None or crop.size == 0:
            return True, candidate

        cand = dict(candidate)
        cid = int(cand.get("class_id", 0))

        # Extract features ONCE and reuse them for the negative match,
        # the positive match and the HSV check below. Previously each of
        # those steps extracted the same crop again (3x ~4 ms per
        # candidate); the matches take the pre-extracted dict via feats=.
        # Plain call (no include_1d_hists=False): the three consumers
        # only read hsv_hist/aspect/texture (+mean_hsv for the HSV
        # check); feature_dict_to_normalized_vector() ignores the 1D
        # hists, so the matching math is identical with or without
        # them — the kwarg only saved one 1D-hist computation per
        # candidate. Dropped (v2r-R10a): the D50 single-extraction
        # contract invokes extract_crop_features(crop) positionally
        # and the kwarg broke that call with TypeError. Extraction
        # is still exactly once per candidate.
        feats = extract_crop_features(crop)

        # Check negative exemplars (suppression)
        neg_sim, neg_ex = self.match_negative(crop, class_id=cid, feats=feats)
        if neg_sim >= self.suppress_neg_threshold:
            logger.debug(
                "Suppressed candidate box (cid=%d, conf=%.2f): matched negative exemplar with sim=%.2f",
                cid,
                cand.get("confidence", 0.0),
                neg_sim,
            )
            return False, cand

        # Check positive exemplars (reinforcement / category-locked class correction)
        pos_sim, best_cid, best_cname, best_ex = self.match_positive(crop, feats=feats)
        if pos_sim >= self.reinforce_pos_threshold and best_cid is not None:
            if best_cid == cid:
                # Reinforce confidence for same class
                old_conf = float(cand.get("confidence", 0.80))
                boost = (1.0 - old_conf) * (pos_sim - 0.50) * 0.70
                cand["confidence"] = round(float(np.clip(old_conf + boost, old_conf, 0.99)), 4)
                cand["exemplar_reinforced"] = True
                cand["exemplar_sim"] = pos_sim
            else:
                # Class change repair: MUST BE CATEGORY-LOCKED!
                # 1. Check super-category isolation
                same_category = is_same_super_category(cid, best_cid)
                # 1b. v20-loop: materyal AİLE kilidi — süper-kategori tek
                # başına yeterli değildi: herb_targena (2) -> ore_orichalcum
                # (3) aynı "3d_material" kabında sayıldığı için repair
                # bitkiyi cevhere çeviriyordu (canlı 16:56:00, sim=0.89).
                # Repair yalnızca aynı ailede olur (bitki->bitki,
                # cevher->cevher); aile dışı repair reddedilir.
                same_family = is_same_material_family(cid, best_cid)
                # 2. Strict threshold: cosine similarity >= reassign_pos_threshold
                cosine_ok = (pos_sim >= self.reassign_pos_threshold)
                # 3. HSV color distribution consistency
                hsv_ok = verify_hsv_color_consistency(feats, best_ex) if (feats and best_ex) else False

                if same_category and same_family and cosine_ok and hsv_ok:
                    old_cid = cid
                    old_cname = cand.get("class_name", "")
                    cand["class_id"] = best_cid
                    cand["class_name"] = best_cname or f"class_{best_cid}"
                    cand["exemplar_corrected"] = True
                    cand["repaired_by_exemplar"] = True
                    cand["exemplar_sim"] = pos_sim
                    cand["confidence"] = round(float(np.clip(max(float(cand.get("confidence", 0.80)), pos_sim * 0.95), 0.85, 0.99)), 4)
                    logger.info(
                        "Exemplar memory repaired box class within category: %d (%s) -> %d (%s) (sim=%.2f, hsv_ok=True)",
                        old_cid,
                        old_cname,
                        best_cid,
                        best_cname,
                        pos_sim,
                    )
                else:
                    cand["repaired_by_exemplar"] = False
                    if not same_category:
                        logger.debug(
                            "Blocked cross-category exemplar repair: %d -> %d rejected (sim=%.2f)",
                            cid,
                            best_cid,
                            pos_sim,
                        )
                    elif not same_family:
                        logger.debug(
                            "Blocked cross-family exemplar repair: %d -> %d rejected (sim=%.2f)",
                            cid,
                            best_cid,
                            pos_sim,
                        )
                    elif not cosine_ok:
                        logger.debug(
                            "Blocked exemplar repair: similarity %.2f < %.2f threshold for %d -> %d",
                            pos_sim,
                            self.reassign_pos_threshold,
                            cid,
                            best_cid,
                        )
                    elif not hsv_ok:
                        logger.debug(
                            "Blocked exemplar repair: HSV color inconsistency for %d -> %d (sim=%.2f)",
                            cid,
                            best_cid,
                            pos_sim,
                        )

        return True, cand

    def refine_annotations(
        self,
        frame: np.ndarray,
        annotations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter and refine a sequence of annotations using learned exemplars."""
        if not annotations:
            return []
        if frame is None or frame.size == 0:
            return list(annotations)

        # Refresh memory if updated on disk (<0.1ms mtime check)
        self.check_reload()

        if not self._positive and not self._negative:
            return list(annotations)

        refined: List[Dict[str, Any]] = []
        for ann in annotations:
            keep, ref_ann = self.refine_candidate(frame, ann)
            if keep:
                refined.append(ref_ann)
        return refined

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics for diagnostics."""
        self.check_reload()
        with self._lock:
            pos_by_class: Dict[int, int] = {}
            for p in self._positive:
                cid = int(p.get("class_id", -1))
                pos_by_class[cid] = pos_by_class.get(cid, 0) + 1
            neg_by_class: Dict[int, int] = {}
            for n in self._negative:
                cid = int(n.get("class_id", -1))
                neg_by_class[cid] = neg_by_class.get(cid, 0) + 1
            # Age coverage: no TTL exists (oldest exemplar only leaves when
            # its class cap is hit), so expose the age spread for monitoring.
            all_ts = [
                float(e.get("timestamp", 0.0))
                for e in list(self._positive) + list(self._negative)
                if isinstance(e.get("timestamp"), (int, float))
            ]
            try:
                json_bytes = self.json_path.stat().st_size if self.json_path.is_file() else 0
            except OSError:
                json_bytes = 0
            return {
                "positive_total": len(self._positive),
                "negative_total": len(self._negative),
                "quarantined_total": len(self._quarantined),
                "positive_by_class": pos_by_class,
                "negative_by_class": neg_by_class,
                "oldest_timestamp": min(all_ts) if all_ts else None,
                "newest_timestamp": max(all_ts) if all_ts else None,
                "json_bytes": json_bytes,
                "positive_matrix_shape": list(self._pos_matrix.shape) if self._pos_matrix is not None else [0, 0],
                "negative_matrix_shape": list(self._neg_matrix.shape) if self._neg_matrix is not None else [0, 0],
                "json_path": str(self.json_path),
                # Y-5: last load failure, if any (None = last load OK).
                "load_error": self._load_error,
                # O-5: content revision of the loaded/saved memory.
                "revision": self._revision,
                # Dilim-30 B-12: True when the last save() dropped the
                # old file's `classes` dict (unreadable old file).
                "last_save_dropped_classes": self._last_save_dropped_classes,
            }


_GLOBAL_EXEMPLAR_MEMORY: Optional[ExemplarMemory] = None
_GLOBAL_LOCK = threading.RLock()


def get_exemplar_memory(json_path: Optional[Union[str, Path]] = None) -> ExemplarMemory:
    """Get or create singleton ExemplarMemory instance."""
    global _GLOBAL_EXEMPLAR_MEMORY
    with _GLOBAL_LOCK:
        if _GLOBAL_EXEMPLAR_MEMORY is None:
            _GLOBAL_EXEMPLAR_MEMORY = ExemplarMemory(json_path=json_path)
        elif json_path is not None and Path(json_path) != _GLOBAL_EXEMPLAR_MEMORY.json_path:
            _GLOBAL_EXEMPLAR_MEMORY = ExemplarMemory(json_path=json_path)
        return _GLOBAL_EXEMPLAR_MEMORY


def sanitize_exemplar_memory(
    json_path: Optional[Union[str, Path]] = None,
    memory: Optional[ExemplarMemory] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Sanitize exemplar memory by purging:
    1. Poisoned / invalid essence_* exemplars (unresolved essence_* class_name — never by ID range: 22..26 are legitimate)
    2. Invalid class_ids >= 16 mistakenly captured from 3D world frames
    3. Any non-canonical class IDs outside 0..26.

    Persists cleaned exemplars back to disk and updates in-memory pre-normalized matrices.
    Returns summary dict of purged and remaining exemplar counts.
    """
    target_mem = memory or get_exemplar_memory(json_path=json_path)
    return target_mem.sanitize(persist=persist)


# CalibratedRoisManager lives in dataset_factory/calibrated_rois.py; it is
# re-exported here so existing imports keep working.
from dataset_factory.calibrated_rois import CalibratedRoisManager  # noqa: F401,E402
from dataset_factory.calibrated_rois import (  # noqa: F401,E402
    DEFAULT_CALIBRATED_ROIS_PATH,
)


_GLOBAL_ROIS_MANAGER: Optional[CalibratedRoisManager] = None


def get_calibrated_rois(json_path: Optional[Union[str, Path]] = None) -> CalibratedRoisManager:
    """Get or create singleton CalibratedRoisManager instance."""
    global _GLOBAL_ROIS_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_ROIS_MANAGER is None or (json_path is not None and Path(json_path) != _GLOBAL_ROIS_MANAGER.json_path):
            _GLOBAL_ROIS_MANAGER = CalibratedRoisManager(json_path=json_path)
        return _GLOBAL_ROIS_MANAGER
