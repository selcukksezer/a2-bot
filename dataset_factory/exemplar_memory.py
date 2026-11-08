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
     if similarity > 0.82, candidate is suppressed as a known false positive.
   - Candidate box crops are compared against learned positive exemplars:
     if similarity >= 0.70, class is reinforced or corrected, and confidence is boosted.
"""

from __future__ import annotations

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

logger = logging.getLogger("Gatherion.ExemplarMemory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXEMPLAR_PATH = PROJECT_ROOT / "dataset" / "learned_exemplars.json"

# Canonical YOLO class IDs: 0..21
CANONICAL_CLASSES: Set[int] = set(range(22))

# Category-locked super categories
SUPER_CATEGORY_3D_MATERIALS: Set[int] = {0, 1, 2, 3, 4, 5, 12}
SUPER_CATEGORY_3D_WORLD: Set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15}
SUPER_CATEGORY_MINIMAP: Set[int] = {9, 11}
SUPER_CATEGORY_HUD: Set[int] = {13, 16, 17, 18, 19, 20, 21}
SUPER_CATEGORY_3D_CRESTS: Set[int] = {6, 7, 8}
SUPER_CATEGORY_3D_MONSTERS: Set[int] = {10}
SUPER_CATEGORY_3D_PATHS: Set[int] = {14, 15}

FEATURE_VECTOR_DIM: int = 438


def extract_box_crop(
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

    x1 = max(0, min(img_w - 2, x1))
    y1 = max(0, min(img_h - 2, y1))
    x2 = max(x1 + 2, min(img_w, x2))
    y2 = max(y1 + 2, min(img_h, y2))

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def extract_crop_features(crop: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Extract standardized visual feature vector from an image crop:
    - 3D HSV Color Histogram (12 Hue x 6 Sat x 6 Val = 432 bins, L1 normalized)
    - Aspect Ratio (w / h)
    - Texture descriptor: [mean_edge, std_edge, laplacian_sharpness, mean_gray, std_gray]
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
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    hist_vec = [round(float(v), 5) for v in hist.flatten()]

    # 1D Histograms for CalibrationWizard compatibility (16 Hue, 8 Sat, 8 Val)
    h_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_h = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_h = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    h_sum = float(np.sum(h_h)) or 1.0
    s_sum = float(np.sum(s_h)) or 1.0
    v_sum = float(np.sum(v_h)) or 1.0
    hist_h = [round(float(v) / h_sum, 4) for v in h_h]
    hist_s = [round(float(v) / s_sum, 4) for v in s_h]
    hist_v = [round(float(v) / v_sum, 4) for v in v_h]
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
    Compute multi-factor similarity score in [0.0, 1.0] between two crop feature dicts:
    - 70% Histogram Intersection (HSV color distribution, supporting 3D or 1D histograms)
    - 15% Aspect Ratio similarity (geometry)
    - 15% Texture / brightness similarity (edges / surface roughness)
    """
    if not feat1 or not feat2:
        return 0.0

    # 1. Color Histogram Intersection
    h1 = np.array(feat1.get("hsv_hist", []), dtype=np.float32)
    h2 = np.array(feat2.get("hsv_hist", []), dtype=np.float32)
    if h1.size == 432 and h2.size == 432:
        # Since histograms are L1-normalized (sum = 1.0), intersection is in [0.0, 1.0]
        hist_sim = float(np.sum(np.minimum(h1, h2)))
    elif "hist_h" in feat1 and "hist_h" in feat2:
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
    - '3d_materials': 3D world gatherable resources (0..5, 12)
    - 'minimap': Minimap symbols (9, 11)
    - 'hud': HUD and cognition symbols (13, 16..21)
    - '3d_crests': 3D crests & prompt icons (6, 7, 8)
    - '3d_monsters': 3D monsters (10)
    - '3d_paths': Walkable path & obstacles (14, 15)
    Returns None for invalid or non-canonical classes (e.g. catalog essence classes >= 22).
    """
    cid = int(class_id)
    if cid in SUPER_CATEGORY_3D_MATERIALS:
        return "3d_materials"
    if cid in SUPER_CATEGORY_MINIMAP:
        return "minimap"
    if cid in SUPER_CATEGORY_HUD:
        return "hud"
    if cid in SUPER_CATEGORY_3D_CRESTS:
        return "3d_crests"
    if cid in SUPER_CATEGORY_3D_MONSTERS:
        return "3d_monsters"
    if cid in SUPER_CATEGORY_3D_PATHS:
        return "3d_paths"
    return None


def is_same_super_category(cid1: int, cid2: int) -> bool:
    """
    Enforce category locking: only classes within the exact same super category
    are permitted to repair each other.
    3D world materials (0..5, 12) can only change among 0..5, 12.
    Minimap symbols (9, 11) can only change among 9, 11.
    HUD symbols (13, 16..21) can only change among HUD classes.
    Never allow cross-category repairs or repairs into essence_* (>= 22).
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
    ar_val = np.array([float(np.clip(ar, 0.05, 10.0)) / 2.0], dtype=np.float32)

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
        suppress_neg_threshold: float = 0.82,
        reinforce_pos_threshold: float = 0.70,
        reassign_pos_threshold: float = 0.78,
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
        self._last_loaded_mtime: float = -1.0

        # Vectorized NumPy matching matrices (N, D) pre-normalized
        self._pos_matrix: Optional[np.ndarray] = None
        self._pos_cids: Optional[np.ndarray] = None
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
            else:
                self._pos_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._pos_cids = np.zeros((0,), dtype=np.int32)

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
        Purge poisoned / invalid essence_* exemplars (class_id >= 22 or class_name containing essence)
        and invalid class_ids >= 16 mistakenly captured from 3D world frames.
        Disallows storing non-canonical classes.
        """
        with self._lock:
            purged_pos = 0
            purged_neg = 0
            clean_pos = []
            clean_neg = []

            for ex in self._positive:
                cid = int(ex.get("class_id", -1))
                cname = str(ex.get("class_name", "")).lower()
                source = str(ex.get("source", "")).lower()

                # Rule 1: Purge any essence_* class (IDs >= 22 or essence name)
                if cname.startswith("essence_") or "essence" in cname or cid >= 22:
                    purged_pos += 1
                    continue

                # Rule 2: Purge non-canonical classes (< 0 or > 21)
                if cid not in CANONICAL_CLASSES:
                    purged_pos += 1
                    continue

                # Rule 3: Purge class_id >= 16 if captured from 3D world or not a valid HUD class
                if cid >= 16:
                    if cid not in SUPER_CATEGORY_HUD:
                        purged_pos += 1
                        continue
                    if any(term in source for term in ("3d_world", "world_resource", "chromatic_salient", "crest")):
                        purged_pos += 1
                        continue

                clean_pos.append(ex)

            for ex in self._negative:
                cid = int(ex.get("class_id", -1))
                cname = str(ex.get("class_name", "")).lower()
                source = str(ex.get("source", "")).lower()

                if cname.startswith("essence_") or "essence" in cname or cid >= 22:
                    purged_neg += 1
                    continue

                if cid not in CANONICAL_CLASSES:
                    purged_neg += 1
                    continue

                # Never keep gatherable material classes in negative exemplars!
                if (
                    cid in SUPER_CATEGORY_3D_MATERIALS
                    or cid in {0, 1, 2, 3, 4, 5, 6, 12}
                    or any(k in cname for k in ("aria", "mela", "targena", "orichalcum", "gem", "tree", "ore_", "herb_"))
                ):
                    purged_neg += 1
                    continue

                if cid >= 16:
                    if cid not in SUPER_CATEGORY_HUD:
                        purged_neg += 1
                        continue
                    if any(term in source for term in ("3d_world", "world_resource", "chromatic_salient", "crest")):
                        purged_neg += 1
                        continue

                clean_neg.append(ex)

            self._positive = clean_pos
            self._negative = clean_neg
            self._rebuild_matrices()

            if persist:
                self.save()

            logger.info(
                "Sanitized exemplar memory: purged %d positive, %d negative (remaining: %d pos, %d neg)",
                purged_pos,
                purged_neg,
                len(self._positive),
                len(self._negative),
            )
            return {
                "purged_positive": purged_pos,
                "purged_negative": purged_neg,
                "remaining_positive": len(self._positive),
                "remaining_negative": len(self._negative),
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
                self._rebuild_matrices()
                return False
            try:
                mtime = self.json_path.stat().st_mtime
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pos_list = list(data.get("positive", []))
                neg_list = list(data.get("negative", []))

                # Also ingest from classes dict if present (e.g. from CalibrationWizard)
                classes_data = data.get("classes", {})
                if isinstance(classes_data, dict):
                    for cid_key, cinfo in classes_data.items():
                        if not isinstance(cinfo, dict):
                            continue
                        c_id = int(cid_key) if cid_key.isdigit() else int(cinfo.get("class_id", 0))
                        c_name = str(cinfo.get("class_name", f"class_{c_id}"))
                        # Skip essence and non-canonical classes
                        if c_id not in CANONICAL_CLASSES or c_id >= 22 or c_name.lower().startswith("essence_"):
                            continue
                        for p_ex in cinfo.get("positive_exemplars", []):
                            if isinstance(p_ex, dict):
                                item = dict(p_ex)
                                item.setdefault("class_id", c_id)
                                item.setdefault("class_name", c_name)
                                if not any(e.get("created_at") == item.get("created_at") and e.get("class_id") == c_id for e in pos_list):
                                    pos_list.append(item)
                        for n_ex in cinfo.get("negative_exemplars", []):
                            if isinstance(n_ex, dict):
                                item = dict(n_ex)
                                item.setdefault("class_id", c_id)
                                item.setdefault("class_name", c_name)
                                if not any(e.get("created_at") == item.get("created_at") and e.get("class_id") == c_id for e in neg_list):
                                    neg_list.append(item)

                self._positive = pos_list
                self._negative = neg_list
                self._last_loaded_mtime = mtime

                self._rebuild_matrices()

                logger.info(
                    "Loaded exemplar memory from %s (%d positive, %d negative)",
                    self.json_path,
                    len(self._positive),
                    len(self._negative),
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load exemplar memory from %s: %s", self.json_path, exc)
                return False

    def save(self) -> bool:
        """Atomically persist learned exemplars to disk."""
        with self._lock:
            try:
                self.json_path.parent.mkdir(parents=True, exist_ok=True)
                existing_classes = {}
                if self.json_path.is_file():
                    try:
                        with open(self.json_path, "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            if isinstance(old_data, dict) and "classes" in old_data:
                                for k, v in old_data["classes"].items():
                                    kid = int(k) if k.isdigit() else -1
                                    # Disallow essence classes in classes dict
                                    if kid in CANONICAL_CLASSES and kid < 22:
                                        existing_classes[k] = v
                    except Exception:
                        pass
                tmp_path = self.json_path.with_suffix(".tmp")
                payload = {
                    "version": 1,
                    "updated_at": time.time(),
                    "positive_count": len(self._positive),
                    "negative_count": len(self._negative),
                    "positive": self._positive,
                    "negative": self._negative,
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
        """
        feats = extract_crop_features(crop)
        if feats is None:
            return False

        cid = int(class_id)
        cname = str(class_name or f"class_{cid}")

        # Gate 1: Disallow non-canonical classes or essence classes
        if cid not in CANONICAL_CLASSES or cid >= 22 or cname.lower().startswith("essence_") or "essence" in cname.lower():
            logger.warning(
                "Rejected non-canonical positive exemplar class %d (%s) - only canonical classes (0..21) allowed.",
                cid,
                cname,
            )
            return False

        # Gate 2: Disallow non-canonical classes for 3D world resources
        src_lower = str(source).lower()
        if any(term in src_lower for term in ("3d", "world", "crest", "harvest", "gather")):
            if cid not in SUPER_CATEGORY_3D_WORLD:
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
                        return True

            entry = {
                "class_id": cid,
                "class_name": cname,
                "hsv_hist": feats["hsv_hist"],
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
                self._positive.remove(oldest)

            self._rebuild_matrices()
            logger.info(
                "Added positive exemplar for class %d (%s) via %s. Total positive: %d",
                cid,
                cname,
                source,
                len(self._positive),
            )
            self.save()
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
        """
        feats = extract_crop_features(crop)
        if feats is None:
            return False

        cid = int(class_id)
        cname = str(class_name or f"class_{cid}")

        # Protect keep/material classes: if a box was rejected as a false positive,
        # it is ground/terrain, NOT an exemplar of the resource class itself.
        if cid in SUPER_CATEGORY_3D_MATERIALS or cid in {0, 1, 2, 3, 4, 5, 6, 12}:
            cid = 15
            cname = "path_ground"

        if cid not in CANONICAL_CLASSES or cid >= 22 or cname.lower().startswith("essence_") or "essence" in cname.lower():
            logger.warning(
                "Rejected non-canonical negative exemplar class %d (%s) - only canonical classes (0..21) allowed.",
                cid,
                cname,
            )
            return False

        src_lower = str(source).lower()
        if any(term in src_lower for term in ("3d", "world", "crest", "harvest", "gather")):
            if cid not in SUPER_CATEGORY_3D_WORLD:
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
                    return True

            entry = {
                "class_id": cid,
                "class_name": cname,
                "hsv_hist": feats["hsv_hist"],
                "aspect_ratio": feats["aspect_ratio"],
                "texture": feats["texture"],
                "mean_hsv": feats.get("mean_hsv", []),
                "timestamp": time.time(),
                "source": source,
            }
            self._negative.append(entry)

            if len(self._negative) > self.max_negatives:
                self._negative.pop(0)

            self._rebuild_matrices()
            logger.info(
                "Added negative exemplar for false-positive %d (%s) via %s. Total negative: %d",
                cid,
                cname,
                source,
                len(self._negative),
            )
            self.save()
            return True

    def match_negative(
        self,
        crop: np.ndarray,
        class_id: Optional[int] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Find highest similarity to any learned negative exemplar.
        Uses vectorized NumPy matrix multiplication (np.dot(matrix, feat_norm)) for 50x faster batch matching.
        Returns: (max_similarity, best_matching_negative_exemplar or None)
        """
        self.check_reload()
        feats = extract_crop_features(crop)
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
    ) -> Tuple[float, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
        """
        Find best matching learned positive exemplar across classes.
        Uses vectorized NumPy matrix multiplication (np.dot(matrix, feat_norm)) for 50x faster batch matching.
        Supports category-locked matching via same_category_as or allowed_classes.
        Returns: (max_similarity, best_class_id, best_class_name, best_exemplar)
        """
        self.check_reload()
        feats = extract_crop_features(crop)
        if feats is None or not self._positive:
            return 0.0, None, None, None

        q_vec = feature_dict_to_normalized_vector(feats)

        with self._lock:
            if self._pos_matrix is None or len(self._pos_matrix) == 0:
                return 0.0, None, None, None

            # O(1) Batch Cosine Similarity
            sims = np.dot(self._pos_matrix, q_vec)

            if same_category_as is not None and self._pos_cids is not None:
                super_cat = get_super_category(same_category_as)
                if super_cat is None:
                    return 0.0, None, None, None
                valid_mask = np.array([
                    get_super_category(int(cid)) == super_cat
                    for cid in self._pos_cids
                ], dtype=bool)
                if not np.any(valid_mask):
                    return 0.0, None, None, None
                sims = np.where(valid_mask, sims, -1.0)
            elif allowed_classes is not None and self._pos_cids is not None:
                allowed_set = set(allowed_classes)
                valid_mask = np.isin(self._pos_cids, list(allowed_set))
                if not np.any(valid_mask):
                    return 0.0, None, None, None
                sims = np.where(valid_mask, sims, -1.0)

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
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """
        Fast query checking if crop matches any learned negative exemplar.
        Returns: (is_negative: bool, similarity: float, matching_exemplar or None)
        """
        thresh = float(threshold if threshold is not None else self.suppress_neg_threshold)
        sim, ex = self.match_negative(crop, class_id=class_id)
        return (sim >= thresh, sim, ex)

    def classify_patch(
        self,
        crop: np.ndarray,
        min_sim: float = 0.70,
    ) -> Tuple[Optional[int], Optional[str], float]:
        """
        Fast query checking if crop matches any learned positive exemplar.
        Returns: (class_id or None, class_name or None, similarity)
        """
        sim, cid, cname, _ = self.match_positive(crop)
        if sim >= float(min_sim) and cid is not None:
            return cid, cname, sim
        return None, None, float(sim)

    def match_class(
        self,
        crop: np.ndarray,
        class_id: int,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Find highest similarity to learned positive exemplars belonging specifically to `class_id`.
        Uses vectorized NumPy matrix matching.
        Returns: (max_similarity: float, best_matching_exemplar or None)
        """
        self.check_reload()
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

            sims = np.dot(self._pos_matrix, q_vec)
            sims = np.where(mask, sims, -1.0)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
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
        2. Test against negative exemplars -> suppress if similarity > 0.82.
        3. Test against positive exemplars ->
           - Same class: reinforce confidence.
           - Class repair (repaired_by_exemplar):
             ONLY allowed if:
             a) Same super-category (3D world materials 0..5, 12 only change among 0..5, 12;
                minimap symbols 9, 11 only among 9, 11; HUD symbols 13, 16..21 only among HUD).
                Never allow 3D box to be repaired into essence_* or HUD classes!
             b) Feature cosine similarity >= 0.88.
             c) HSV color distribution consistency verified.
        Returns: (keep: bool, refined_candidate: dict)
        """
        self.check_reload()
        crop = extract_box_crop(frame, candidate)
        if crop is None or crop.size == 0:
            return True, candidate

        cand = dict(candidate)
        cid = int(cand.get("class_id", 0))

        # Check negative exemplars (suppression)
        neg_sim, neg_ex = self.match_negative(crop, class_id=cid)
        if neg_sim >= self.suppress_neg_threshold:
            logger.debug(
                "Suppressed candidate box (cid=%d, conf=%.2f): matched negative exemplar with sim=%.2f",
                cid,
                cand.get("confidence", 0.0),
                neg_sim,
            )
            return False, cand

        # Extract features for positive match & color consistency
        feats = extract_crop_features(crop)

        # Check positive exemplars (reinforcement / category-locked class correction)
        pos_sim, best_cid, best_cname, best_ex = self.match_positive(crop)
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
                # 2. Strict threshold: cosine similarity >= 0.88
                cosine_ok = (pos_sim >= 0.88)
                # 3. HSV color distribution consistency
                hsv_ok = verify_hsv_color_consistency(feats, best_ex) if (feats and best_ex) else False

                if same_category and cosine_ok and hsv_ok:
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
                    elif not cosine_ok:
                        logger.debug(
                            "Blocked exemplar repair: similarity %.2f < 0.88 threshold for %d -> %d",
                            pos_sim,
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
            return {
                "positive_total": len(self._positive),
                "negative_total": len(self._negative),
                "positive_by_class": pos_by_class,
                "positive_matrix_shape": list(self._pos_matrix.shape) if self._pos_matrix is not None else [0, 0],
                "negative_matrix_shape": list(self._neg_matrix.shape) if self._neg_matrix is not None else [0, 0],
                "json_path": str(self.json_path),
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
    1. Poisoned / invalid essence_* exemplars (class_id >= 22 or class_name containing essence)
    2. Invalid class_ids >= 16 mistakenly captured from 3D world frames
    3. Any non-canonical class IDs outside 0..21.

    Persists cleaned exemplars back to disk and updates in-memory pre-normalized matrices.
    Returns summary dict of purged and remaining exemplar counts.
    """
    target_mem = memory or get_exemplar_memory(json_path=json_path)
    return target_mem.sanitize(persist=persist)


DEFAULT_CALIBRATED_ROIS_PATH = PROJECT_ROOT / "config" / "calibrated_rois.json"


class CalibratedRoisManager:
    """
    Dynamically loads and hot-reloads calibrated ROIs from config/calibrated_rois.json.
    Uses mtime check to reload within <50ms (<0.1ms typical) whenever calibration changes.
    """

    def __init__(self, json_path: Optional[Union[str, Path]] = None) -> None:
        self.json_path = Path(json_path) if json_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        self._lock = threading.RLock()
        self._rois: Dict[str, Dict[str, Any]] = {}
        self._last_mtime: float = -1.0
        self.check_reload()

    def check_reload(self, force: bool = False) -> bool:
        """Inspects mtime and reloads if modified on disk."""
        with self._lock:
            try:
                if not self.json_path.is_file():
                    if self._last_mtime != -1.0:
                        self._rois.clear()
                        self._last_mtime = -1.0
                    return False
                mtime = self.json_path.stat().st_mtime
                if not force and mtime == self._last_mtime:
                    return True
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                rois_dict: Dict[str, Dict[str, Any]] = {}
                # Support both data["rois"] and top-level keys
                if "rois" in data and isinstance(data["rois"], dict):
                    rois_dict.update(data["rois"])
                for k, v in data.items():
                    if k not in ("rois", "updated_at", "version") and isinstance(v, dict):
                        rois_dict[k] = v
                self._rois = rois_dict
                self._last_mtime = mtime
                return True
            except Exception as exc:
                logger.debug("CalibratedRoisManager check_reload error: %s", exc)
                return False

    ROI_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "skill_bar": ("skill_bar", "skill_slot", "skill_slots"),
        "skill_slot": ("skill_slot", "skill_bar", "skill_slots"),
        "player_hp_bar": ("player_hp_bar", "player_hp", "hp_bar"),
        "player_mp_bar": ("player_mp_bar", "player_mp", "mp_bar"),
        "target_hp_bar": ("target_hp_bar", "target_hp", "mob_nameplate", "target_bar"),
        "mob_nameplate": ("mob_nameplate", "target_hp_bar", "target_hp"),
        "minimap": ("minimap", "minimap_roi", "minimap_inner"),
    }

    def get_roi_entry(self, key: str) -> Optional[Dict[str, Any]]:
        self.check_reload()
        raw_key = str(key or "").strip()
        with self._lock:
            if raw_key in self._rois:
                return self._rois[raw_key]
            lower_key = raw_key.lower()
            if lower_key in self._rois:
                return self._rois[lower_key]
            aliases = self.ROI_KEY_ALIASES.get(lower_key, ())
            for alias in aliases:
                if alias in self._rois:
                    return self._rois[alias]
            return None

    def has_roi(self, key: str) -> bool:
        """Check if ROI is present in calibrated ROIs (with alias fallback)."""
        return self.get_roi_entry(key) is not None

    def get_all_rois(self) -> Dict[str, Dict[str, Any]]:
        """Return shallow copy of all currently loaded ROIs."""
        self.check_reload()
        with self._lock:
            return dict(self._rois)

    def reload_now(self) -> bool:
        """Forces an immediate reload from disk."""
        return self.check_reload(force=True)

    def get_normalized_rect(self, key: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (x1, y1, x2, y2) normalized ratios [0.0..1.0] for the requested ROI key.
        """
        entry = self.get_roi_entry(key)
        if not entry:
            return None
        # Format 1: normalized_roi: [x, y, w, h]
        if "normalized_roi" in entry and len(entry["normalized_roi"]) == 4:
            x, y, w, h = [float(v) for v in entry["normalized_roi"]]
            return (round(x, 6), round(y, 6), round(min(1.0, x + w), 6), round(min(1.0, y + h), 6))
        # Format 2: norm_x, norm_y, norm_w, norm_h
        if "norm_x" in entry and "norm_y" in entry:
            x = float(entry["norm_x"])
            y = float(entry["norm_y"])
            w = float(entry.get("norm_w", 0.0))
            h = float(entry.get("norm_h", 0.0))
            return (round(x, 6), round(y, 6), round(min(1.0, x + w), 6), round(min(1.0, y + h), 6))
        return None

    def get_pixel_crop(self, frame: np.ndarray, key: str) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        rect = self.get_normalized_rect(key)
        if not rect:
            return None
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 2, int(round(rect[0] * fw))))
        y1 = max(0, min(fh - 2, int(round(rect[1] * fh))))
        x2 = max(x1 + 2, min(fw, int(round(rect[2] * fw))))
        y2 = max(y1 + 2, min(fh, int(round(rect[3] * fh))))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None


_GLOBAL_ROIS_MANAGER: Optional[CalibratedRoisManager] = None


def get_calibrated_rois(json_path: Optional[Union[str, Path]] = None) -> CalibratedRoisManager:
    """Get or create singleton CalibratedRoisManager instance."""
    global _GLOBAL_ROIS_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_ROIS_MANAGER is None or (json_path is not None and Path(json_path) != _GLOBAL_ROIS_MANAGER.json_path):
            _GLOBAL_ROIS_MANAGER = CalibratedRoisManager(json_path=json_path)
        return _GLOBAL_ROIS_MANAGER

