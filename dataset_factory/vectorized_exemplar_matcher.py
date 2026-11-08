# -*- coding: utf-8 -*-
"""
dataset_factory/vectorized_exemplar_matcher.py
==============================================
O(1) Vectorized Positive/Negative Exemplar Matcher for 438-Dimensional
L2-Normalized Visual Feature Vectors with Super-Category Masking.

Architecture & Guarantees:
1. 438-Dimensional L2-Normalized Feature Space:
   - 432 bins: 3D HSV Histogram (12 Hue x 6 Sat x 6 Val), L2-normalized, weighted by sqrt(0.70)
   - 1 bin:    Aspect Ratio (clamped [0.05, 10.0] / 2.0), weighted by sqrt(0.15)
   - 5 bins:   Texture Descriptor [mean_edge, std_edge, laplacian, mean_gray, std_gray],
               L2-normalized, weighted by sqrt(0.15)
   - Combined vector is L2-normalized to unit length (||v||_2 == 1.0) so that
     a single `np.dot(matrix, query)` directly computes exact cosine similarity in [-1.0, 1.0].
2. O(1) Vectorized Matrix Multiplication:
   - Pre-allocates contiguous `np.float32` matrices of shape `(N, 438)` for both
     positive and negative exemplars.
   - Pre-caches integer `class_id` arrays `(N,)` and integer `super_category_id` arrays `(N,)`
     so super-category locking (`self._pos_super_cats == target_cat_id`) executes in pure C/NumPy
     without Python-level loops.
3. Strict Super-Category Isolation:
   - Prevents cross-category class pollution (e.g., 3D world materials 0..5, 12 can never
     match or repair into HUD classes 13, 16..21 or minimap classes 9, 11).
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

FEATURE_VECTOR_DIM: int = 438
HSV_HIST_BINS: int = 432
ASPECT_RATIO_BINS: int = 1
TEXTURE_BINS: int = 5

WEIGHT_HSV: float = 0.70
WEIGHT_ASPECT_RATIO: float = 0.15
WEIGHT_TEXTURE: float = 0.15

# Canonical YOLO class IDs (0..21)
CANONICAL_CLASSES: Set[int] = set(range(22))

# Super-Category Sets
SUPER_CATEGORY_3D_MATERIALS: Set[int] = {0, 1, 2, 3, 4, 5, 12}
SUPER_CATEGORY_MINIMAP: Set[int] = {9, 11}
SUPER_CATEGORY_HUD: Set[int] = {13, 16, 17, 18, 19, 20, 21}
SUPER_CATEGORY_3D_CRESTS: Set[int] = {6, 7, 8}
SUPER_CATEGORY_3D_MONSTERS: Set[int] = {10}
SUPER_CATEGORY_3D_PATHS: Set[int] = {14, 15}
SUPER_CATEGORY_CONSUMABLES: Set[int] = {22, 23, 24}

# Integer codes for O(1) vectorized NumPy super-category masking
SUPER_CAT_ID_NONE: int = -1
SUPER_CAT_ID_3D_MATERIALS: int = 1
SUPER_CAT_ID_MINIMAP: int = 2
SUPER_CAT_ID_HUD: int = 3
SUPER_CAT_ID_3D_CRESTS: int = 4
SUPER_CAT_ID_3D_MONSTERS: int = 5
SUPER_CAT_ID_3D_PATHS: int = 6
SUPER_CAT_ID_CONSUMABLES: int = 7

SUPER_CATEGORY_NAME_TO_ID: Dict[str, int] = {
    "3d_materials": SUPER_CAT_ID_3D_MATERIALS,
    "minimap": SUPER_CAT_ID_MINIMAP,
    "hud": SUPER_CAT_ID_HUD,
    "3d_crests": SUPER_CAT_ID_3D_CRESTS,
    "3d_monsters": SUPER_CAT_ID_3D_MONSTERS,
    "3d_paths": SUPER_CAT_ID_3D_PATHS,
    "consumables": SUPER_CAT_ID_CONSUMABLES,
}

SUPER_CATEGORY_ID_TO_NAME: Dict[int, str] = {
    v: k for k, v in SUPER_CATEGORY_NAME_TO_ID.items()
}


def get_super_category(class_id: int) -> Optional[str]:
    """
    Map canonical class_id to its super-category name:
    - '3d_materials': 3D world gatherable resources (0..5, 12)
    - 'minimap': Minimap symbols (9, 11)
    - 'hud': HUD and cognition symbols (13, 16..21)
    - '3d_crests': 3D crests & prompt icons (6, 7, 8)
    - '3d_monsters': 3D monsters (10)
    - '3d_paths': Walkable path & obstacles (14, 15)
    Returns None for non-canonical classes.
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


def get_super_category_id(class_id: int) -> int:
    """Return integer super-category code for O(1) vectorized array comparisons."""
    cat_name = get_super_category(class_id)
    if cat_name is None:
        return SUPER_CAT_ID_NONE
    return SUPER_CATEGORY_NAME_TO_ID.get(cat_name, SUPER_CAT_ID_NONE)


def is_same_super_category(cid1: int, cid2: int) -> bool:
    """Return True if and only if cid1 and cid2 belong to the same valid super-category."""
    cat1 = get_super_category_id(cid1)
    cat2 = get_super_category_id(cid2)
    return cat1 != SUPER_CAT_ID_NONE and cat1 == cat2


def normalize_l2_vector(vec: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    Ensure a 1D vector of length 438 is float32 and L2-normalized to unit length.
    Pads or truncates safely if input length differs from 438.
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.size != FEATURE_VECTOR_DIM:
        padded = np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
        if arr.size > 0:
            n = min(FEATURE_VECTOR_DIM, arr.size)
            padded[:n] = arr[:n]
        arr = padded
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9 or not np.isfinite(norm):
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def feature_dict_to_normalized_vector(feat: Dict[str, Any]) -> np.ndarray:
    """
    Convert a visual feature dictionary or pre-computed vector dict into a
    438-dimensional L2-normalized float32 vector (`||v||_2 = 1.0`).

    If `feat` already contains `"feature_vector"` or `"vector"` of length 438,
    it is directly L2-normalized and returned. Otherwise builds the 438-D vector from:
    - `hsv_hist`: 432 bins (L2-normalized, weighted by sqrt(0.70))
    - `aspect_ratio`: 1 bin (clamped [0.05, 10.0] / 2.0, weighted by sqrt(0.15))
    - `texture`: 5 bins (L2-normalized, weighted by sqrt(0.15))
    """
    if not feat:
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)

    if "feature_vector" in feat and feat["feature_vector"] is not None:
        return normalize_l2_vector(feat["feature_vector"])
    if "vector" in feat and feat["vector"] is not None:
        return normalize_l2_vector(feat["vector"])

    hsv = np.asarray(feat.get("hsv_hist", []), dtype=np.float32).ravel()
    if hsv.size != HSV_HIST_BINS:
        hsv_pad = np.zeros(HSV_HIST_BINS, dtype=np.float32)
        if hsv.size > 0:
            hsv_pad[: min(HSV_HIST_BINS, hsv.size)] = hsv[:HSV_HIST_BINS]
        hsv = hsv_pad
    hsv_norm = hsv / (float(np.linalg.norm(hsv)) + 1e-9)

    ar = float(feat.get("aspect_ratio", 1.0))
    ar_val = np.array([float(np.clip(ar, 0.05, 10.0)) / 2.0], dtype=np.float32)

    tex = np.asarray(feat.get("texture", [0.0] * TEXTURE_BINS), dtype=np.float32).ravel()
    if tex.size != TEXTURE_BINS:
        tex_pad = np.zeros(TEXTURE_BINS, dtype=np.float32)
        if tex.size > 0:
            tex_pad[: min(TEXTURE_BINS, tex.size)] = tex[:TEXTURE_BINS]
        tex = tex_pad
    tex_norm = tex / (float(np.linalg.norm(tex)) + 1e-9)

    combined = np.concatenate(
        [
            np.sqrt(WEIGHT_HSV, dtype=np.float32) * hsv_norm,
            np.sqrt(WEIGHT_ASPECT_RATIO, dtype=np.float32) * ar_val,
            np.sqrt(WEIGHT_TEXTURE, dtype=np.float32) * tex_norm,
        ]
    )
    norm = float(np.linalg.norm(combined))
    if norm <= 1e-9 or not np.isfinite(norm):
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
    return (combined / norm).astype(np.float32)


def coerce_query_vector(query: Union[np.ndarray, Dict[str, Any], Sequence[float]]) -> np.ndarray:
    """Convert a query (feature dict, 1D array, or sequence) to a 438-D L2-normalized float32 vector."""
    if isinstance(query, dict):
        return feature_dict_to_normalized_vector(query)
    return normalize_l2_vector(query)


@dataclass(frozen=True)
class ExemplarMatchResult:
    """Structured result of an O(1) vectorized exemplar match."""

    matched: bool
    similarity: float
    class_id: Optional[int]
    class_name: Optional[str]
    super_category: Optional[str]
    exemplar_index: int
    exemplar: Optional[Dict[str, Any]]


class VectorizedExemplarMatcher:
    """
    O(1) Vectorized Positive/Negative Exemplar Matcher operating on 438-dimensional
    L2-normalized feature vectors via a single `np.dot(matrix, query)` call and
    vectorized super-category boolean masking.
    """

    def __init__(
        self,
        positive_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        negative_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        suppress_neg_threshold: float = 0.82,
        reinforce_pos_threshold: float = 0.70,
        repair_pos_threshold: float = 0.88,
        cross_class_neg_scale: float = 0.90,
    ) -> None:
        self.suppress_neg_threshold = float(suppress_neg_threshold)
        self.reinforce_pos_threshold = float(reinforce_pos_threshold)
        self.repair_pos_threshold = float(repair_pos_threshold)
        self.cross_class_neg_scale = float(cross_class_neg_scale)

        self._lock = threading.RLock()
        self._positive: List[Dict[str, Any]] = []
        self._negative: List[Dict[str, Any]] = []

        self._pos_matrix: np.ndarray = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
        self._pos_cids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._pos_super_cats: np.ndarray = np.zeros((0,), dtype=np.int32)

        self._neg_matrix: np.ndarray = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
        self._neg_cids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._neg_super_cats: np.ndarray = np.zeros((0,), dtype=np.int32)

        if positive_exemplars or negative_exemplars:
            self.rebuild_matrices(
                positive_exemplars=positive_exemplars or [],
                negative_exemplars=negative_exemplars or [],
            )

    @property
    def feature_dim(self) -> int:
        """Return feature vector dimensionality (438)."""
        return FEATURE_VECTOR_DIM

    @property
    def positive_count(self) -> int:
        """Return number of indexed positive exemplars."""
        with self._lock:
            return int(self._pos_matrix.shape[0])

    @property
    def negative_count(self) -> int:
        """Return number of indexed negative exemplars."""
        with self._lock:
            return int(self._neg_matrix.shape[0])

    @property
    def positive_matrix(self) -> np.ndarray:
        """Return read-only view of the (N_pos, 438) L2-normalized positive matrix."""
        with self._lock:
            view = self._pos_matrix.view()
            view.flags.writeable = False
            return view

    @property
    def negative_matrix(self) -> np.ndarray:
        """Return read-only view of the (N_neg, 438) L2-normalized negative matrix."""
        with self._lock:
            view = self._neg_matrix.view()
            view.flags.writeable = False
            return view

    def rebuild_matrices(
        self,
        positive_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        negative_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Build contiguous `(N, 438)` L2-normalized float32 matrices and pre-computed
        class/super-category index arrays for O(1) matching.
        """
        with self._lock:
            if positive_exemplars is not None:
                self._positive = [dict(ex) for ex in positive_exemplars if isinstance(ex, dict)]
            if negative_exemplars is not None:
                self._negative = [dict(ex) for ex in negative_exemplars if isinstance(ex, dict)]

            if self._positive:
                pos_vecs = [feature_dict_to_normalized_vector(ex) for ex in self._positive]
                pos_mat = np.vstack(pos_vecs).astype(np.float32, copy=False)
                pos_norms = np.linalg.norm(pos_mat, axis=1, keepdims=True)
                self._pos_matrix = np.ascontiguousarray(pos_mat / np.maximum(pos_norms, 1e-9), dtype=np.float32)
                self._pos_cids = np.array(
                    [int(ex.get("class_id", -1)) for ex in self._positive],
                    dtype=np.int32,
                )
                self._pos_super_cats = np.array(
                    [get_super_category_id(int(ex.get("class_id", -1))) for ex in self._positive],
                    dtype=np.int32,
                )
            else:
                self._pos_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._pos_cids = np.zeros((0,), dtype=np.int32)
                self._pos_super_cats = np.zeros((0,), dtype=np.int32)

            if self._negative:
                neg_vecs = [feature_dict_to_normalized_vector(ex) for ex in self._negative]
                neg_mat = np.vstack(neg_vecs).astype(np.float32, copy=False)
                neg_norms = np.linalg.norm(neg_mat, axis=1, keepdims=True)
                self._neg_matrix = np.ascontiguousarray(neg_mat / np.maximum(neg_norms, 1e-9), dtype=np.float32)
                self._neg_cids = np.array(
                    [int(ex.get("class_id", -1)) for ex in self._negative],
                    dtype=np.int32,
                )
                self._neg_super_cats = np.array(
                    [get_super_category_id(int(ex.get("class_id", -1))) for ex in self._negative],
                    dtype=np.int32,
                )
            else:
                self._neg_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._neg_cids = np.zeros((0,), dtype=np.int32)
                self._neg_super_cats = np.zeros((0,), dtype=np.int32)

    def build_super_category_mask(
        self,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
        allowed_classes: Optional[Collection[int]] = None,
        for_negative: bool = False,
    ) -> np.ndarray:
        """
        Compute a boolean mask of shape `(N,)` in O(1) using vectorized NumPy comparisons
        over pre-cached super-category IDs and class IDs.
        """
        with self._lock:
            cids = self._neg_cids if for_negative else self._pos_cids
            scats = self._neg_super_cats if for_negative else self._pos_super_cats
            n = cids.shape[0]
            if n == 0:
                return np.zeros((0,), dtype=bool)

            mask = np.ones((n,), dtype=bool)

            if same_category_as is not None:
                target_cat_id = get_super_category_id(int(same_category_as))
                if target_cat_id == SUPER_CAT_ID_NONE:
                    return np.zeros((n,), dtype=bool)
                mask &= (scats == target_cat_id)

            if super_category is not None:
                cat_key = str(super_category).strip().lower()
                target_cat_id = SUPER_CATEGORY_NAME_TO_ID.get(cat_key, SUPER_CAT_ID_NONE)
                if target_cat_id == SUPER_CAT_ID_NONE:
                    return np.zeros((n,), dtype=bool)
                mask &= (scats == target_cat_id)

            if allowed_classes is not None:
                allowed_arr = np.fromiter((int(c) for c in allowed_classes), dtype=np.int32)
                if allowed_arr.size == 0:
                    return np.zeros((n,), dtype=bool)
                mask &= np.isin(cids, allowed_arr)

            return mask

    def match_positive(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
    ) -> Tuple[float, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
        """
        O(1) Vectorized positive exemplar match via single `np.dot(matrix, query)` and
        super-category masking.

        Returns:
            `(max_similarity, best_class_id, best_class_name, best_exemplar)`
        """
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None, None, None

        with self._lock:
            if self._pos_matrix.shape[0] == 0:
                return 0.0, None, None, None

            # Single matrix-vector multiplication: (N, 438) @ (438,) -> (N,)
            sims = np.dot(self._pos_matrix, q_vec)

            if same_category_as is not None or super_category is not None or allowed_classes is not None:
                valid_mask = self.build_super_category_mask(
                    same_category_as=same_category_as,
                    super_category=super_category,
                    allowed_classes=allowed_classes,
                    for_negative=False,
                )
                if not np.any(valid_mask):
                    return 0.0, None, None, None
                sims = np.where(valid_mask, sims, -1.0)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < 0.0:
                return 0.0, None, None, None

            best_sim = float(np.clip(best_sim, 0.0, 1.0))
            best_ex = self._positive[best_idx]
            best_cid = int(best_ex.get("class_id", 0))
            best_cname = str(best_ex.get("class_name", f"class_{best_cid}"))
            return round(best_sim, 4), best_cid, best_cname, best_ex

    def match_positive_detailed(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> ExemplarMatchResult:
        """Return a structured `ExemplarMatchResult` for a positive exemplar query."""
        threshold = float(self.reinforce_pos_threshold if min_similarity is None else min_similarity)
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

        with self._lock:
            if self._pos_matrix.shape[0] == 0:
                return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

            sims = np.dot(self._pos_matrix, q_vec)

            if same_category_as is not None or super_category is not None or allowed_classes is not None:
                valid_mask = self.build_super_category_mask(
                    same_category_as=same_category_as,
                    super_category=super_category,
                    allowed_classes=allowed_classes,
                    for_negative=False,
                )
                if not np.any(valid_mask):
                    return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)
                sims = np.where(valid_mask, sims, -1.0)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < 0.0:
                return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            best_ex = self._positive[best_idx]
            best_cid = int(best_ex.get("class_id", 0))
            best_cname = str(best_ex.get("class_name", f"class_{best_cid}"))
            best_scat = get_super_category(best_cid)
            return ExemplarMatchResult(
                matched=(best_sim >= threshold),
                similarity=best_sim,
                class_id=best_cid,
                class_name=best_cname,
                super_category=best_scat,
                exemplar_index=best_idx,
                exemplar=best_ex,
            )

    def match_negative(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        class_id: Optional[int] = None,
        same_category_only: bool = False,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        O(1) Vectorized negative exemplar match via single `np.dot(matrix, query)`.
        If `class_id` is given, cross-class negative matches are scaled by `cross_class_neg_scale` (0.90).
        If `same_category_only` is True, masks out negatives from unrelated super-categories.

        Returns:
            `(max_similarity, best_matching_negative_exemplar)`
        """
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None

        with self._lock:
            if self._neg_matrix.shape[0] == 0:
                return 0.0, None

            # Single matrix-vector multiplication: (N, 438) @ (438,) -> (N,)
            sims = np.dot(self._neg_matrix, q_vec)

            if class_id is not None:
                cid = int(class_id)
                if same_category_only:
                    valid_mask = self.build_super_category_mask(
                        same_category_as=cid,
                        for_negative=True,
                    )
                    if not np.any(valid_mask):
                        return 0.0, None
                    sims = np.where(valid_mask, sims, -1.0)
                scale = np.where(self._neg_cids == cid, 1.0, self.cross_class_neg_scale)
                sims = np.where(sims >= 0.0, sims * scale, -1.0)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < 0.0:
                return 0.0, None

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            return best_sim, self._negative[best_idx]

    def match_class(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        class_id: int,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        O(1) Vectorized match against positive exemplars belonging specifically to `class_id`.
        """
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None

        cid = int(class_id)
        with self._lock:
            if self._pos_matrix.shape[0] == 0:
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

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            return best_sim, self._positive[best_idx]

    def match_batch(
        self,
        queries: Sequence[Union[np.ndarray, Dict[str, Any], Sequence[float]]],
        candidate_class_ids: Optional[Sequence[int]] = None,
        enforce_super_category: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch vectorized positive & negative matching for `M` queries simultaneously.
        Computes `(N_pos, 438) @ (438, M)` and `(N_neg, 438) @ (438, M)` in two BLAS calls.
        """
        if not queries:
            return []

        q_vecs = [coerce_query_vector(q) for q in queries]
        # Shape: (438, M)
        query_mat_t = np.column_stack(q_vecs).astype(np.float32, copy=False)
        m_count = len(q_vecs)

        with self._lock:
            has_pos = self._pos_matrix.shape[0] > 0
            has_neg = self._neg_matrix.shape[0] > 0

            # Single matrix-matrix multiply for all M queries: (N, 438) @ (438, M) -> (N, M)
            pos_sims_mat = (
                np.dot(self._pos_matrix, query_mat_t)
                if has_pos
                else np.zeros((0, m_count), dtype=np.float32)
            )
            neg_sims_mat = (
                np.dot(self._neg_matrix, query_mat_t)
                if has_neg
                else np.zeros((0, m_count), dtype=np.float32)
            )

            results: List[Dict[str, Any]] = []
            for j in range(m_count):
                cand_cid = (
                    int(candidate_class_ids[j])
                    if candidate_class_ids is not None and j < len(candidate_class_ids)
                    else None
                )

                # Negative evaluation for query j
                neg_sim = 0.0
                neg_ex: Optional[Dict[str, Any]] = None
                if has_neg:
                    col_neg = neg_sims_mat[:, j]
                    if cand_cid is not None:
                        scale = np.where(self._neg_cids == cand_cid, 1.0, self.cross_class_neg_scale)
                        col_neg = col_neg * scale
                    neg_idx = int(np.argmax(col_neg))
                    neg_sim = round(float(np.clip(col_neg[neg_idx], 0.0, 1.0)), 4)
                    neg_ex = self._negative[neg_idx]

                # Positive evaluation for query j
                pos_sim = 0.0
                pos_cid: Optional[int] = None
                pos_cname: Optional[str] = None
                pos_ex: Optional[Dict[str, Any]] = None
                if has_pos:
                    col_pos = pos_sims_mat[:, j]
                    if enforce_super_category and cand_cid is not None:
                        target_cat = get_super_category_id(cand_cid)
                        if target_cat == SUPER_CAT_ID_NONE:
                            col_pos = np.full_like(col_pos, -1.0)
                        else:
                            col_pos = np.where(self._pos_super_cats == target_cat, col_pos, -1.0)
                    pos_idx = int(np.argmax(col_pos))
                    raw_best = float(col_pos[pos_idx])
                    if raw_best >= 0.0:
                        pos_sim = round(float(np.clip(raw_best, 0.0, 1.0)), 4)
                        pos_ex = self._positive[pos_idx]
                        pos_cid = int(pos_ex.get("class_id", 0))
                        pos_cname = str(pos_ex.get("class_name", f"class_{pos_cid}"))

                results.append(
                    {
                        "query_index": j,
                        "candidate_class_id": cand_cid,
                        "is_suppressed": neg_sim >= self.suppress_neg_threshold,
                        "negative_similarity": neg_sim,
                        "negative_exemplar": neg_ex,
                        "positive_similarity": pos_sim,
                        "positive_class_id": pos_cid,
                        "positive_class_name": pos_cname,
                        "positive_exemplar": pos_ex,
                    }
                )
            return results

    def add_positive_exemplar(
        self,
        exemplar: Dict[str, Any],
        dedup_threshold: float = 0.96,
        max_per_class: int = 40,
    ) -> bool:
        """
        Add a positive exemplar dictionary and rebuild the pre-normalized matrix.
        Rejects non-canonical classes (`class_id` outside `0..21`).
        """
        if not isinstance(exemplar, dict):
            return False
        cid = int(exemplar.get("class_id", -1))
        cname = str(exemplar.get("class_name", f"class_{cid}"))
        if cid not in CANONICAL_CLASSES or "essence" in cname.lower():
            return False

        q_vec = feature_dict_to_normalized_vector(exemplar)
        if not np.any(q_vec):
            return False

        with self._lock:
            if self._pos_matrix.shape[0] > 0:
                mask = (self._pos_cids == cid)
                if np.any(mask):
                    sims = np.dot(self._pos_matrix, q_vec)
                    sims = np.where(mask, sims, -1.0)
                    best_idx = int(np.argmax(sims))
                    if float(sims[best_idx]) >= dedup_threshold:
                        self._positive[best_idx].update(exemplar)
                        return True

            self._positive.append(dict(exemplar))
            cls_entries = [e for e in self._positive if int(e.get("class_id", -1)) == cid]
            if len(cls_entries) > max_per_class:
                oldest = min(cls_entries, key=lambda x: float(x.get("timestamp", 0.0)))
                self._positive.remove(oldest)

            self.rebuild_matrices()
            return True

    def add_negative_exemplar(
        self,
        exemplar: Dict[str, Any],
        dedup_threshold: float = 0.96,
        max_negatives: int = 150,
    ) -> bool:
        """
        Add a negative exemplar dictionary and rebuild the pre-normalized matrix.
        """
        if not isinstance(exemplar, dict):
            return False
        cid = int(exemplar.get("class_id", -1))
        cname = str(exemplar.get("class_name", f"class_{cid}"))
        if cid not in CANONICAL_CLASSES or "essence" in cname.lower():
            return False

        q_vec = feature_dict_to_normalized_vector(exemplar)
        if not np.any(q_vec):
            return False

        with self._lock:
            if self._neg_matrix.shape[0] > 0:
                sims = np.dot(self._neg_matrix, q_vec)
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= dedup_threshold:
                    self._negative[best_idx].update(exemplar)
                    return True

            self._negative.append(dict(exemplar))
            if len(self._negative) > max_negatives:
                self._negative.pop(0)

            self.rebuild_matrices()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics for the vectorized matcher."""
        with self._lock:
            return {
                "feature_dim": FEATURE_VECTOR_DIM,
                "positive_count": int(self._pos_matrix.shape[0]),
                "negative_count": int(self._neg_matrix.shape[0]),
                "positive_matrix_shape": list(self._pos_matrix.shape),
                "negative_matrix_shape": list(self._neg_matrix.shape),
                "suppress_neg_threshold": self.suppress_neg_threshold,
                "reinforce_pos_threshold": self.reinforce_pos_threshold,
                "repair_pos_threshold": self.repair_pos_threshold,
            }


__all__ = [
    "FEATURE_VECTOR_DIM",
    "HSV_HIST_BINS",
    "ASPECT_RATIO_BINS",
    "TEXTURE_BINS",
    "CANONICAL_CLASSES",
    "SUPER_CATEGORY_3D_MATERIALS",
    "SUPER_CATEGORY_MINIMAP",
    "SUPER_CATEGORY_HUD",
    "SUPER_CATEGORY_3D_CRESTS",
    "SUPER_CATEGORY_3D_MONSTERS",
    "SUPER_CATEGORY_3D_PATHS",
    "ExemplarMatchResult",
    "VectorizedExemplarMatcher",
    "get_super_category",
    "get_super_category_id",
    "is_same_super_category",
    "normalize_l2_vector",
    "feature_dict_to_normalized_vector",
    "coerce_query_vector",
]
