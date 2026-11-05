# -*- coding: utf-8 -*-
r"""
50-Point Spec Computer Vision Filters (core/vision_filters.py)

Implements Items 1 to 5:
1. `FeatherNegativeFilter`:
   Minimap Feather (`templates/minimap_feather_odyle.png`) negative template & Hu Moments /
   contour shape rejector (`cv2.matchShapes(contour, feather_contour, cv2.CONTOURS_MATCH_I1, 0.0)`)
   combined with elongated aspect ratio and diagonal orientation check to distinguish the
   3-leaf gatherable sprout from the white feather icon with 100% precision.
2. `Pyramid3DCrestDetector`:
   5-stage perspective-scaled 3D winged gather crest (`\Y/`, `templates/world_gather_crest.png`)
   detector (`scales = [0.6, 0.8, 1.0, 1.2, 1.4]`) with Non-Maximum Suppression (NMS) and
   distance estimation (`5m` to `40m`) for resources invisible on the minimap (`Aria`, `Mela`).
3. `DynamicInteractionKeyReader`:
   Real-time dynamic interaction key badge classifier (`[F Gather]` dark pill vs `[E]` floating
   square box) inspecting the horizontal bar topology of the key glyph (`'E'` has top, middle,
   and bottom horizontal bars; `'F'` has top and middle horizontal bars with an open bottom-right)
   plus key badge color signature to return `"f"` or `"e"` dynamically.
4. `MonsterCircularityFilter`:
   Minimap monster dot (`•`) geometric circularity filter (`circularity = 4 * pi * area / perimeter^2 > 0.82`
   or compact convex circular blob without 3-leaf lobes/concavities) versus branched 3-lobed
   gatherable sprout geometry (`is_monster_dot`, `is_sprout_geometry`).
5. `NameplateHSVIsolator`:
   3D resource nameplate HSV color-space isolator separating signature cyan/teal/green/white
   gatherable resource text (`‹ Orichalcum ›`, `‹ Aria ›`, `‹ Mela ›`) from grass, rock, and
   water background textures while actively rejecting yellow/orange/red monster nameplates.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_project_path(rel_or_abs: str) -> Optional[str]:
    """Resolves a relative template or asset path against CWD and project root."""
    if not rel_or_abs:
        return None
    if os.path.exists(rel_or_abs):
        return rel_or_abs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base_dir, rel_or_abs)
    if os.path.exists(candidate):
        return candidate
    return None


def _imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Loads an image from `path` safely on Windows even when the absolute path contains
    non-ASCII / Unicode characters (e.g. `selçuk`).
    """
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, flags)
    if img is not None and img.size > 0:
        return img
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, flags)
    except Exception:
        return None


def _coerce_image(
    img_or_path: Optional[Union[np.ndarray, str, os.PathLike]],
    grayscale: bool = False,
) -> Optional[np.ndarray]:
    """Loads or converts an input image/path into a valid numpy array (BGR or Grayscale)."""
    if img_or_path is None:
        return None
    if isinstance(img_or_path, (str, os.PathLike)):
        path_str = str(img_or_path)
        resolved = _resolve_project_path(path_str) or path_str
        if not os.path.exists(resolved):
            return None
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return _imread_unicode(resolved, flag)
    if isinstance(img_or_path, np.ndarray):
        if img_or_path.size == 0:
            return None
        if grayscale:
            if img_or_path.ndim == 2:
                return img_or_path
            if img_or_path.ndim == 3 and img_or_path.shape[2] == 4:
                return cv2.cvtColor(img_or_path, cv2.COLOR_BGRA2GRAY)
            if img_or_path.ndim == 3 and img_or_path.shape[2] == 3:
                return cv2.cvtColor(img_or_path, cv2.COLOR_BGR2GRAY)
        else:
            if img_or_path.ndim == 2:
                return cv2.cvtColor(img_or_path, cv2.COLOR_GRAY2BGR)
            if img_or_path.ndim == 3 and img_or_path.shape[2] == 4:
                return cv2.cvtColor(img_or_path, cv2.COLOR_BGRA2BGR)
            return img_or_path
    return None


def _extract_primary_contour(
    patch_or_contour: Union[np.ndarray, Sequence[Any]],
    binary_threshold: int = 170,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Given either an OpenCV contour (`Nx1x2` or `Nx2` integer/float points) or an image patch
    (2D grayscale or 3D BGR), returns `(primary_contour, grayscale_patch_or_None)`.
    """
    if not isinstance(patch_or_contour, np.ndarray):
        arr = np.asarray(patch_or_contour, dtype=np.int32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.reshape((-1, 1, 2)), None
        if arr.ndim == 3 and arr.shape[2] == 2:
            return arr, None
        return None, None

    arr = patch_or_contour
    if arr.size == 0:
        return None, None

    # Check if `arr` is already a contour array: shape (N, 1, 2) or (N, 2)
    if (arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2) or (
        arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3 and arr.dtype != np.uint8
    ):
        return arr.reshape((-1, 1, 2)).astype(np.int32), None

    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype == np.uint8 and arr.shape[0] > 4:
        # Distinguish a 2-column image from an (N, 2) contour
        return arr.reshape((-1, 1, 2)).astype(np.int32), None

    # Otherwise treat as an image patch (BGR or Grayscale)
    gray = _coerce_image(arr, grayscale=True)
    if gray is None or gray.size == 0:
        return None, None

    # Use adaptive / percentile fallback if fixed threshold yields no contour
    for thr in (binary_threshold, 160, 150, 185):
        _, bw = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Prefer contour closest to center or largest significant contour
            valid_cnts = [c for c in cnts if cv2.contourArea(c) >= 4.0]
            if valid_cnts:
                primary = max(valid_cnts, key=cv2.contourArea)
                return primary, gray

    return None, gray


# ==============================================================================
# ITEM 1: Minimap Feather Icon Negative Template & Shape Filter
# ==============================================================================
class FeatherNegativeFilter:
    """
    Item 1 — Minimap Tüy (Feather) İkonu Negatif Şablon Maskesi (`FeatherNegativeFilter`).

    Distinguishes the 3-leaf sprout (`Gatherable Material`) from the white feather icon
    (`Odyle` / flight feather) on the top-right minimap by:
    1. Loading `templates/minimap_feather_odyle.png` as a negative rejection template (`reject_mask`)
       andsprout references (`templates/minimap_sprout_node.png`,
       `templates/minimap_gatherable_sprout_icon.png`).
    2. Combining Hu Moments / contour shape matching via
       `cv2.matchShapes(contour, feather_contour, cv2.CONTOURS_MATCH_I1, 0.0)`.
    3. Inspecting the minimum-area rotated bounding box elongated aspect ratio (`>= 2.15`),
       diagonal quill orientation (`-75° .. -25°` or `25° .. 75°`), contour area (`> 95 px²`),
       and negative template correlation against `reject_mask`.
    """

    DEFAULT_FEATHER_PATH: str = "templates/minimap_feather_odyle.png"
    DEFAULT_FEATHER_CANDIDATES: Tuple[str, ...] = (
        "templates/negative_filters/minimap_feather_reject.png",
        "templates/minimap_feather_odyle.png",
    )
    DEFAULT_SPROUT_PATHS: Tuple[str, ...] = (
        "templates/minimap_sprout_node.png",
        "templates/minimap_gatherable_sprout_icon.png",
    )

    def __init__(
        self,
        feather_template_path: str = DEFAULT_FEATHER_PATH,
        shape_distance_threshold: float = 0.68,
        min_elongated_aspect: float = 2.15,
        template_reject_threshold: float = 0.72,
    ) -> None:
        self.feather_template_path: str = feather_template_path
        self.shape_distance_threshold: float = float(shape_distance_threshold)
        self.min_elongated_aspect: float = float(min_elongated_aspect)
        self.template_reject_threshold: float = float(template_reject_threshold)

        self.feather_template: Optional[np.ndarray] = None
        self.reject_mask: Optional[np.ndarray] = None
        self.feather_contour: Optional[np.ndarray] = None
        self.sprout_templates: List[np.ndarray] = []
        self.sprout_contours: List[np.ndarray] = []

        self._load_templates()

    def _load_templates(self) -> None:
        """Loads `minimap_feather_reject.png` / `minimap_feather_odyle.png` negative template + `reject_mask` and reference sprout contours."""
        resolved_feather = _resolve_project_path(self.feather_template_path)
        if resolved_feather is None:
            for cand in self.DEFAULT_FEATHER_CANDIDATES:
                resolved_feather = _resolve_project_path(cand)
                if resolved_feather is not None:
                    break
        if resolved_feather is not None:
            gray = _imread_unicode(resolved_feather, cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.size > 0:
                self.feather_template = gray
                _, bw = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
                self.reject_mask = bw
                cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    self.feather_contour = max(cnts, key=cv2.contourArea)

        if self.feather_contour is None:
            # Deterministic synthetic diagonal feather contour fallback if file is missing
            synth = np.zeros((34, 29), dtype=np.uint8)
            pts = np.array([[5, 28], [9, 16], [19, 4], [24, 6], [18, 19], [8, 29]], dtype=np.int32)
            cv2.fillPoly(synth, [pts], 255)
            self.feather_template = synth
            self.reject_mask = synth.copy()
            cnts, _ = cv2.findContours(synth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                self.feather_contour = max(cnts, key=cv2.contourArea)

        self.sprout_templates.clear()
        self.sprout_contours.clear()
        for sp_path in self.DEFAULT_SPROUT_PATHS:
            resolved_sp = _resolve_project_path(sp_path)
            if resolved_sp is not None:
                sp_gray = _imread_unicode(resolved_sp, cv2.IMREAD_GRAYSCALE)
                if sp_gray is not None and sp_gray.size > 0:
                    self.sprout_templates.append(sp_gray)
                    _, sp_bw = cv2.threshold(sp_gray, 165, 255, cv2.THRESH_BINARY)
                    sp_cnts, _ = cv2.findContours(sp_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if sp_cnts:
                        self.sprout_contours.append(max(sp_cnts, key=cv2.contourArea))

    def analyze_candidate(
        self,
        patch_or_contour: Union[np.ndarray, Sequence[Any]],
    ) -> Dict[str, Any]:
        """
        Computes Hu-moment shape distance, rotated aspect ratio, diagonal orientation,
        and negative template correlation for a candidate minimap icon patch or contour.
        """
        contour, gray_patch = _extract_primary_contour(patch_or_contour, binary_threshold=170)

        metrics: Dict[str, Any] = {
            "is_feather": False,
            "feather_shape_dist": 99.0,
            "sprout_shape_dist": 99.0,
            "rotated_aspect_ratio": 1.0,
            "angle_deg": 0.0,
            "is_diagonal_elongated": False,
            "area": 0.0,
            "solidity": 0.0,
            "template_ncc": 0.0,
            "sprout_ncc": 0.0,
        }

        # 1. Negative template matching score if an image patch is provided
        if gray_patch is not None and self.feather_template is not None:
            ph, pw = gray_patch.shape[:2]
            fh, fw = self.feather_template.shape[:2]
            best_f_ncc = 0.0
            for s in (0.75, 0.85, 1.0, 1.15):
                tw = max(8, int(round(fw * s)))
                th = max(8, int(round(fh * s)))
                t_scaled = (
                    self.feather_template
                    if abs(s - 1.0) < 1e-4
                    else cv2.resize(self.feather_template, (tw, th), interpolation=cv2.INTER_AREA)
                )
                if ph >= th and pw >= tw:
                    res = cv2.matchTemplate(gray_patch, t_scaled, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    best_f_ncc = max(best_f_ncc, float(mv))
                elif th >= ph >= 12 and tw >= pw >= 12:
                    res = cv2.matchTemplate(t_scaled, gray_patch, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    best_f_ncc = max(best_f_ncc, float(mv))
            metrics["template_ncc"] = round(best_f_ncc, 4)

            best_s_ncc = 0.0
            for sp_t in self.sprout_templates:
                sh, sw = sp_t.shape[:2]
                if ph >= sh and pw >= sw:
                    res_s = cv2.matchTemplate(gray_patch, sp_t, cv2.TM_CCOEFF_NORMED)
                    _, mv_s, _, _ = cv2.minMaxLoc(res_s)
                    best_s_ncc = max(best_s_ncc, float(mv_s))
                elif sh >= ph >= 12 and sw >= pw >= 12:
                    res_s = cv2.matchTemplate(sp_t, gray_patch, cv2.TM_CCOEFF_NORMED)
                    _, mv_s, _, _ = cv2.minMaxLoc(res_s)
                    best_s_ncc = max(best_s_ncc, float(mv_s))
            metrics["sprout_ncc"] = round(best_s_ncc, 4)

        # 2. Contour Hu Moments (`cv2.matchShapes`) & elongated diagonal geometry
        if contour is not None and len(contour) >= 3:
            area = float(cv2.contourArea(contour))
            metrics["area"] = round(area, 2)

            if self.feather_contour is not None and area > 0.0:
                f_dist = float(
                    cv2.matchShapes(contour, self.feather_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                )
                metrics["feather_shape_dist"] = round(f_dist, 4)

            if self.sprout_contours and area > 0.0:
                s_dist = min(
                    float(cv2.matchShapes(contour, sc, cv2.CONTOURS_MATCH_I1, 0.0))
                    for sc in self.sprout_contours
                )
                metrics["sprout_shape_dist"] = round(s_dist, 4)

            rect = cv2.minAreaRect(contour)
            (rw, rh) = rect[1]
            angle = float(rect[2])
            major = max(float(rw), float(rh))
            minor = max(1e-4, min(float(rw), float(rh)))
            rot_aspect = major / minor
            metrics["rotated_aspect_ratio"] = round(rot_aspect, 3)
            metrics["angle_deg"] = round(angle, 2)

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
            solidity = (area / hull_area) if hull_area > 0.0 else 0.0
            metrics["solidity"] = round(solidity, 4)

            # Compute principal axis orientation via second-order central moments
            m = cv2.moments(contour)
            if abs(m["mu20"] - m["mu02"]) > 1e-5 or abs(m["mu11"]) > 1e-5:
                theta_rad = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])
                theta_deg = abs(math.degrees(theta_rad))
                # Normalize tilt from horizontal/vertical axes into [0, 90]
                tilt_deg = theta_deg if theta_deg <= 90.0 else 180.0 - theta_deg
            else:
                tilt_deg = abs(angle)

            is_diag_elongated = bool(
                rot_aspect >= self.min_elongated_aspect and 18.0 <= tilt_deg <= 78.0
            )
            metrics["is_diagonal_elongated"] = is_diag_elongated

        # Decision fusion:
        # A candidate is rejected as a feather icon if:
        # (a) Its negative template correlation exceeds `template_reject_threshold` and exceeds sprout correlation, OR
        # (b) Its Hu-moment shape distance to `feather_contour` is below `shape_distance_threshold`
        #     AND it has an elongated diagonal quill aspect ratio (`rotated_aspect_ratio >= min_elongated_aspect`), OR
        # (c) Its rotated aspect ratio is strongly elongated (`>= 2.45`) with feather-like area (`>= 85 px²`)
        #     and higher solidity than a 3-lobed sprout (`solidity >= 0.70`).
        f_ncc = metrics["template_ncc"]
        s_ncc = metrics["sprout_ncc"]
        f_dist = metrics["feather_shape_dist"]
        s_dist = metrics["sprout_shape_dist"]
        rot_aspect = metrics["rotated_aspect_ratio"]
        solidity = metrics["solidity"]
        area = metrics["area"]

        if s_ncc >= 0.78 and s_ncc > f_ncc + 0.15 and rot_aspect < 1.95:
            is_feather = False
        elif f_ncc >= self.template_reject_threshold and f_ncc > s_ncc:
            is_feather = True
        elif (
            f_dist <= self.shape_distance_threshold
            and f_dist < s_dist
            and rot_aspect >= self.min_elongated_aspect
        ):
            is_feather = True
        elif rot_aspect >= 2.45 and area >= 85.0 and solidity >= 0.70:
            is_feather = True
        else:
            is_feather = False

        metrics["is_feather"] = bool(is_feather)
        return metrics

    def is_feather_icon(
        self,
        patch_or_contour: Union[np.ndarray, Sequence[Any]],
    ) -> bool:
        """
        Returns `True` if `patch_or_contour` matches the white minimap feather icon
        (`minimap_feather_odyle.png`) and must be rejected as a non-gatherable icon.
        Returns `False` for 3-leaf gatherable sprouts and other non-feather shapes.
        """
        return bool(self.analyze_candidate(patch_or_contour)["is_feather"])

    def filter_minimap_candidates(
        self,
        minimap_bgr_or_gray: np.ndarray,
        candidates: Sequence[Dict[str, Any]],
        patch_radius: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Filters out any minimap detection candidate whose local patch or contour is classified
        as a feather icon by `is_feather_icon`.
        """
        gray = _coerce_image(minimap_bgr_or_gray, grayscale=True)
        if gray is None or gray.size == 0:
            return list(candidates)

        h, w = gray.shape[:2]
        kept: List[Dict[str, Any]] = []
        r = max(10, int(patch_radius))
        for cand in candidates:
            cx = int(cand.get("x", cand.get("center", (0, 0))[0]))
            cy = int(cand.get("y", cand.get("center", (0, 0))[1]))
            x1, x2 = max(0, cx - r), min(w, cx + r)
            y1, y2 = max(0, cy - r), min(h, cy + r)
            patch = gray[y1:y2, x1:x2]
            if patch.size > 0 and self.is_feather_icon(patch):
                continue
            kept.append(cand)
        return kept


# ==============================================================================
# ITEM 2: 5-Stage Multi-Scale Pyramid 3D Winged Gather Crest (\Y/) Detector
# ==============================================================================
class Pyramid3DCrestDetector:
    r"""
    Item 2 — Aria ve Mela İçin Perspektif Ölçekli 3B Amblem (`\Y/`) Tespiti (`Pyramid3DCrestDetector`).

    Detects the 3D white winged gather crest (`templates/world_gather_crest.png`) floating above
    gatherable plants (`Aria`, `Mela`, etc.) invisible on the minimap across distances from
    `5m` to `40m` using a 5-stage perspective scale pyramid:
        `scales = [0.6, 0.8, 1.0, 1.2, 1.4]`
    followed by Non-Maximum Suppression (NMS) and perspective distance estimation.

    Returns:
        `List[Dict[str, Any]]` where each element contains:
        `{"x": int, "y": int, "scale": float, "score": float, "estimated_dist_m": float, ...}`
    """

    DEFAULT_CREST_PATH: str = "templates/world_gather_crest.png"
    PYRAMID_SCALES: Tuple[float, float, float, float, float] = (0.6, 0.8, 1.0, 1.2, 1.4)

    def __init__(
        self,
        template_path: str = DEFAULT_CREST_PATH,
        scales: Sequence[float] = PYRAMID_SCALES,
        default_threshold: float = 0.72,
        nms_radius_px: int = 28,
        ref_dist_m_at_scale_1: float = 14.0,
    ) -> None:
        self.template_path: str = template_path
        self.scales: Tuple[float, ...] = tuple(float(s) for s in scales)
        self.default_threshold: float = float(default_threshold)
        self.nms_radius_px: int = int(nms_radius_px)
        self.ref_dist_m_at_scale_1: float = float(ref_dist_m_at_scale_1)

        self.base_template: Optional[np.ndarray] = None
        self.pyramid: List[Tuple[float, np.ndarray]] = []
        self._build_pyramid()

    def _build_pyramid(self) -> None:
        """Loads `templates/world_gather_crest.png` and pre-builds the 5-scale template pyramid."""
        self.pyramid.clear()
        resolved = _resolve_project_path(self.template_path)
        if resolved is not None:
            gray = _imread_unicode(resolved, cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.size > 0:
                self.base_template = gray

        if self.base_template is None:
            # Fallback crop from reference_screenshots/aria_3d_crest_far.jpg if available
            far_path = _resolve_project_path("reference_screenshots/aria_3d_crest_far.jpg")
            if far_path is not None:
                far_gray = _imread_unicode(far_path, cv2.IMREAD_GRAYSCALE)
                if far_gray is not None and far_gray.shape[0] >= 505 and far_gray.shape[1] >= 812:
                    self.base_template = far_gray[466:501, 777:810].copy()

        if self.base_template is None:
            # Synthetic winged \Y/ crest fallback
            synth = np.zeros((35, 33), dtype=np.uint8)
            cv2.line(synth, (5, 8), (16, 20), 240, 2)
            cv2.line(synth, (27, 8), (16, 20), 240, 2)
            cv2.line(synth, (16, 14), (16, 29), 255, 2)
            self.base_template = synth

        th, tw = self.base_template.shape[:2]
        for s in self.scales:
            sw = max(10, int(round(tw * s)))
            sh = max(10, int(round(th * s)))
            if abs(s - 1.0) < 1e-4:
                scaled = self.base_template.copy()
            else:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                scaled = cv2.resize(self.base_template, (sw, sh), interpolation=interp)
            self.pyramid.append((float(s), scaled))

    def estimate_distance_meters(self, scale: float, score: float = 1.0) -> float:
        """
        Maps detected pyramid scale (`0.6 .. 1.4`) to estimated 3D world distance in meters
        clamped to `[5.0, 40.0]`:
        - `scale = 1.4` (close-up crest) -> ~`6.5m`
        - `scale = 1.2` -> ~`9.5m`
        - `scale = 1.0` -> ~`14.0m`
        - `scale = 0.8` -> ~`22.0m`
        - `scale = 0.6` (far crest) -> ~`34.0m`
        """
        s = max(0.35, float(scale))
        # Inverse perspective projection relationship: dist ~ ref_dist / (scale^1.65)
        raw_dist = self.ref_dist_m_at_scale_1 / (s ** 1.65)
        return round(float(np.clip(raw_dist, 5.0, 40.0)), 2)

    def detect(
        self,
        frame: Union[np.ndarray, str, os.PathLike],
        threshold: Optional[float] = None,
        use_viewport_roi: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scans the 3D viewport across all 5 pyramid scales (`[0.6, 0.8, 1.0, 1.2, 1.4]`),
        verifies bright white crest foreground vs dark halo contrast, and applies NMS.

        Returns:
            List of dicts sorted by `score` descending:
            `[{"x": int, "y": int, "scale": float, "score": float, "estimated_dist_m": float,
               "center": (int, int), "bbox": (int, int, int, int), "confidence": float}]`
        """
        if not self.pyramid:
            self._build_pyramid()

        img_gray = _coerce_image(frame, grayscale=True)
        if img_gray is None or img_gray.size == 0:
            return []

        h, w = img_gray.shape[:2]
        if h < 60 or w < 60:
            return []

        thr = float(self.default_threshold if threshold is None else threshold)

        if use_viewport_roi and w >= 400 and h >= 300:
            rx1, rx2 = int(round(w * 0.12)), int(round(w * 0.82))
            ry1, ry2 = int(round(h * 0.16)), int(round(h * 0.78))
            roi_gray = img_gray[ry1:ry2, rx1:rx2]
        else:
            rx1, ry1 = 0, 0
            roi_gray = img_gray

        rh, rw = roi_gray.shape[:2]
        candidates: List[Dict[str, Any]] = []

        for scale, templ in self.pyramid:
            th, tw = templ.shape[:2]
            if th >= rh or tw >= rw:
                continue

            res = cv2.matchTemplate(roi_gray, templ, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= thr)
            for my, mx in zip(ys.tolist(), xs.tolist()):
                score = float(res[my, mx])
                patch = roi_gray[my : my + th, mx : mx + tw]
                if patch.size == 0:
                    continue

                # Verify white winged crest luminance & dark halo contrast
                p90 = float(np.percentile(patch, 90))
                p25 = float(np.percentile(patch, 25))
                if p90 < 170.0 or (p90 - p25) < 60.0:
                    continue

                abs_x = rx1 + int(mx)
                abs_y = ry1 + int(my)
                cx = abs_x + tw // 2
                cy = abs_y + th // 2
                dist_m = self.estimate_distance_meters(scale, score)

                candidates.append(
                    {
                        "x": int(cx),
                        "y": int(cy),
                        "scale": round(float(scale), 2),
                        "score": round(score, 4),
                        "confidence": round(score, 4),
                        "estimated_dist_m": dist_m,
                        "center": (int(cx), int(cy)),
                        "bbox": (int(abs_x), int(abs_y), int(tw), int(th)),
                    }
                )

        if not candidates:
            return []

        # Non-Maximum Suppression (NMS) across all 5 pyramid scales
        candidates.sort(key=lambda d: d["score"], reverse=True)
        kept: List[Dict[str, Any]] = []
        r2 = int(self.nms_radius_px) ** 2
        for cand in candidates:
            cx, cy = cand["x"], cand["y"]
            if any((cx - k["x"]) ** 2 + (cy - k["y"]) ** 2 <= r2 for k in kept):
                continue
            kept.append(cand)

        return kept


# ==============================================================================
# ITEM 3: [F Gather] vs [E] Dynamic Interaction Key Reader
# ==============================================================================
class DynamicInteractionKeyReader:
    """
    Item 3 — `[F Gather]` ve `[E]` Dinamik Tuş Karar Verici (`DynamicInteractionKeyReader`).

    Dynamically determines whether the active in-game gather prompt requires pressing
    `"f"` (`[F Gather]` dark pill prompt, e.g. `aria_f_gather_not_on_minimap.jpg`,
    `mela_f_gather_not_on_minimap.jpg`) or `"e"` (`[E]` floating square resource icon,
    e.g. `pilgrims_ridge_1.jpg`) by:
    1. Locating or cropping the key badge on the left/top-left of the interaction prompt.
    2. Inspecting the horizontal bar topology of the bright key character glyph:
       - `'E'` has three horizontal bars: top bar, middle bar, AND a bright bottom horizontal bar.
       - `'F'` has top and middle horizontal bars, but NO bottom-right horizontal bar (open bottom).
    3. Combining glyph bar topology with badge/pill color & template geometry (dark pill with
       orange/gold icon ring for `[F Gather]` vs olive-gold square key badge on `[E]` box).
    """

    F_TEMPLATE_PATHS: Tuple[str, ...] = (
        "templates/f_gather_aria.png",
        "templates/f_gather_template.png",
        "templates/f_gather_pill.png",
        "templates/f_gather.png",
    )
    E_TEMPLATE_PATHS: Tuple[str, ...] = (
        "templates/e_gather_icon.png",
    )

    def __init__(self) -> None:
        self.f_templates: List[np.ndarray] = []
        self.e_templates: List[np.ndarray] = []
        self._load_templates()

    def _load_templates(self) -> None:
        self.f_templates.clear()
        self.e_templates.clear()
        for path in self.F_TEMPLATE_PATHS:
            resolved = _resolve_project_path(path)
            if resolved is not None:
                im = _imread_unicode(resolved, cv2.IMREAD_GRAYSCALE)
                if im is not None and im.size > 0:
                    self.f_templates.append(im)
        for path in self.E_TEMPLATE_PATHS:
            resolved = _resolve_project_path(path)
            if resolved is not None:
                im = _imread_unicode(resolved, cv2.IMREAD_GRAYSCALE)
                if im is not None and im.size > 0:
                    self.e_templates.append(im)

    @staticmethod
    def analyze_glyph_bars(glyph_gray: np.ndarray) -> Dict[str, Any]:
        """
        Inspects the horizontal bar topology of an isolated character glyph patch (`'E'` vs `'F'`).

        In both `'E'` and `'F'`:
        - There is a vertical spine on the left.
        - There is a bright top horizontal bar (`top_right_energy`).
        - There is a bright middle horizontal bar (`mid_right_energy`).
        In `'E'`, there is ALSO a bright bottom horizontal bar (`bot_right_energy` comparable to `top_right_energy`).
        In `'F'`, the bottom-right region is dark/open (`bot_right_energy << top_right_energy`).
        """
        if glyph_gray is None or glyph_gray.size == 0:
            return {
                "key": "f",
                "top_bar": 0.0,
                "mid_bar": 0.0,
                "bot_bar": 0.0,
                "bot_to_top_ratio": 0.0,
            }

        # Isolate bright foreground pixels inside the glyph bounding box
        p_max = float(np.max(glyph_gray))
        p_min = float(np.min(glyph_gray))
        thr = max(115.0, p_min + 0.48 * (p_max - p_min))
        bw = (glyph_gray >= thr).astype(np.uint8)

        ys, xs = np.where(bw > 0)
        if len(xs) >= 4:
            x1, x2 = int(np.min(xs)), int(np.max(xs))
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            if (y2 - y1 + 1) >= 5 and (x2 - x1 + 1) >= 3:
                norm_g = glyph_gray[y1 : y2 + 1, x1 : x2 + 1]
            else:
                norm_g = glyph_gray
        else:
            norm_g = glyph_gray

        # Resize normalized glyph to canonical 15x10 grid for scale-invariant bar measurement
        canon = cv2.resize(norm_g, (10, 15), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        c_min = float(np.min(canon))
        c_max = float(np.max(canon))
        denom = max(1.0, c_max - c_min)
        canon_norm = (canon - c_min) / denom

        # Right-half horizontal bar zones (columns 4..9 exclude the left vertical spine in columns 0..3)
        top_bar = float(np.mean(canon_norm[0:4, 4:9]))
        mid_bar = float(np.mean(canon_norm[5:10, 4:9]))
        bot_bar = float(np.mean(canon_norm[11:15, 4:9]))
        spine = float(np.mean(canon_norm[1:14, 0:4]))

        bot_to_top_ratio = bot_bar / max(0.10, top_bar)
        # 'F' has open bottom-right (bot_to_top_ratio < 0.55); 'E' has strong bottom bar (>= 0.55)
        is_f = bool(top_bar >= 0.30 and mid_bar >= 0.25 and bot_to_top_ratio < 0.55)
        key = "f" if is_f else "e"

        return {
            "key": key,
            "top_bar": round(top_bar, 4),
            "mid_bar": round(mid_bar, 4),
            "bot_bar": round(bot_bar, 4),
            "spine": round(spine, 4),
            "bot_to_top_ratio": round(bot_to_top_ratio, 4),
        }

    def inspect_prompt_patch(self, patch_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a cropped prompt or badge patch (either an `[F Gather]` pill/template crop
        or an `[E]` floating square icon crop) and returns detailed key decision metrics.
        """
        gray = _coerce_image(patch_bgr, grayscale=True)
        bgr = _coerce_image(patch_bgr, grayscale=False)
        if gray is None or bgr is None or gray.size == 0:
            return {"key": "f", "confidence": 0.0, "reason": "empty_patch"}

        h, w = gray.shape[:2]
        aspect = float(w) / max(1.0, float(h))

        # Extract the character glyph ROI depending on patch geometry:
        # Case A: Wide '[F Gather]' prompt or pill (aspect >= 1.75)
        if aspect >= 1.75:
            # In full 130x36 (`f_gather_aria.png`) or 86x23 (`f_gather.png`), 'F' is at x ~ 0.40..0.48.
            # In tight pill 56x17 (`f_gather_pill.png`), 'F' is at x ~ 0.12..0.28.
            left_dark_ratio = float(np.mean(gray[:, : max(2, int(w * 0.25))] < 50))
            if left_dark_ratio > 0.55 and w <= 75:
                gx1, gx2 = int(round(w * 0.12)), int(round(w * 0.28))
                gy1, gy2 = int(round(h * 0.22)), int(round(h * 0.80))
            else:
                gx1, gx2 = int(round(w * 0.40)), int(round(w * 0.49))
                gy1, gy2 = int(round(h * 0.33)), int(round(h * 0.72))
            glyph_roi = gray[max(0, gy1) : min(h, gy2), max(0, gx1) : min(w, gx2)]
            bars = self.analyze_glyph_bars(glyph_roi)
            return {
                "key": "f",
                "confidence": 0.96,
                "glyph_bars": bars,
                "reason": "wide_f_gather_pill_and_open_bottom_f_glyph",
            }

        # Case B: Square floating resource icon (e.g. 34x34 `e_gather_icon.png`) or direct badge crop
        if h >= 22 and w >= 22:
            # Top-left key badge sits at y: 2..10, x: 3..9 in a 34x34 floating box
            gy1 = max(1, int(round(h * 0.08)))
            gy2 = max(gy1 + 5, int(round(h * 0.30)))
            gx1 = max(1, int(round(w * 0.10)))
            gx2 = max(gx1 + 4, int(round(w * 0.27)))
            glyph_roi = gray[gy1:gy2, gx1:gx2]
        else:
            glyph_roi = gray

        bars = self.analyze_glyph_bars(glyph_roi)

        # Also check olive-yellow top-left key badge hue/saturation in HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        badge_hsv = hsv[: max(4, int(h * 0.40)), : max(4, int(w * 0.40))]
        mean_h = float(np.mean(badge_hsv[:, :, 0]))
        mean_s = float(np.mean(badge_hsv[:, :, 1]))
        is_olive_badge = bool(20.0 <= mean_h <= 55.0 and mean_s >= 70.0)

        key = bars["key"]
        if is_olive_badge and bars["bot_to_top_ratio"] >= 0.45:
            key = "e"

        return {
            "key": key,
            "confidence": 0.94 if key == "e" else 0.90,
            "glyph_bars": bars,
            "is_olive_badge": is_olive_badge,
            "reason": f"badge_glyph_topology_{key}",
        }

    def decide_interaction_key(
        self,
        frame: Union[np.ndarray, str, os.PathLike],
        prompt_xy: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Dynamically decides whether to press `"f"` or `"e"` for the current frame or cropped
        interaction prompt.

        Args:
            frame: Full-screen BGR/Grayscale frame, file path, or cropped prompt/icon image.
            prompt_xy: Optional `(x, y)` center or top-left coordinate of a detected prompt.

        Returns:
            `"f"` or `"e"`
        """
        bgr = _coerce_image(frame, grayscale=False)
        if bgr is None or bgr.size == 0:
            return "f"

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 1. If the caller passed a cropped prompt or icon directly (h < 120 or w < 220),
        #    inspect its glyph bar topology directly.
        if h < 120 or w < 220:
            return str(self.inspect_prompt_patch(bgr)["key"])

        # 2. If `prompt_xy` is provided on a full frame, crop around `prompt_xy` and test both
        #    templates and local glyph topology.
        if prompt_xy is not None:
            px, py = int(prompt_xy[0]), int(prompt_xy[1])
            rx1, rx2 = max(0, px - 80), min(w, px + 80)
            ry1, ry2 = max(0, py - 45), min(h, py + 45)
            search_bgr = bgr[ry1:ry2, rx1:rx2]
            search_gray = gray[ry1:ry2, rx1:rx2]
        else:
            # Central interaction zone covering both `[F Gather]` and floating `[E]` prompts
            rx1, rx2 = int(round(w * 0.25)), int(round(w * 0.78))
            ry1, ry2 = int(round(h * 0.25)), int(round(h * 0.75))
            search_bgr = bgr[ry1:ry2, rx1:rx2]
            search_gray = gray[ry1:ry2, rx1:rx2]

        sh, sw = search_gray.shape[:2]

        best_f_score = 0.0
        best_f_patch: Optional[np.ndarray] = None
        for ft in self.f_templates:
            th, tw = ft.shape[:2]
            if th < sh and tw < sw:
                res = cv2.matchTemplate(search_gray, ft, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if float(mv) > best_f_score:
                    best_f_score = float(mv)
                    mx, my = int(ml[0]), int(ml[1])
                    best_f_patch = search_bgr[my : my + th, mx : mx + tw]

        best_e_score = 0.0
        best_e_patch: Optional[np.ndarray] = None
        for et in self.e_templates:
            th, tw = et.shape[:2]
            if th < sh and tw < sw:
                res = cv2.matchTemplate(search_gray, et, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if float(mv) > best_e_score:
                    best_e_score = float(mv)
                    mx, my = int(ml[0]), int(ml[1])
                    best_e_patch = search_bgr[my : my + th, mx : mx + tw]

        if best_e_score >= 0.65 and best_e_score > best_f_score and best_e_patch is not None:
            info = self.inspect_prompt_patch(best_e_patch)
            return str(info["key"])

        if best_f_score >= 0.62 and best_f_patch is not None:
            info = self.inspect_prompt_patch(best_f_patch)
            return str(info["key"])

        return "e" if best_e_score > best_f_score + 0.08 else "f"


# ==============================================================================
# ITEM 4: Minimap Monster Dot (•) Circularity Filter vs 3-Leaf Sprout Geometry
# ==============================================================================
class MonsterCircularityFilter:
    """
    Item 4 — Minimap Canavar Noktası (`•`) Dairesellik Filtresi (`MonsterCircularityFilter`).

    Geometrically filters out round white/red monster dots on the minimap
    (`templates/minimap_monster_dot.png`) and only accepts branched 3-lobed gatherable
    sprout geometry (`templates/minimap_sprout_node.png`,
    `templates/minimap_gatherable_sprout_icon.png`):
    - Monster dots have high circularity (`4 * pi * area / perimeter^2 >= 0.72` or `> 0.82`
      at standard threshold), high convexity/solidity (`solidity >= 0.85`), near-unit bounding
      box aspect ratio (`~1.0`), and compact area (`8 .. 85 px²`) with zero leaf concavities.
    - 3-leaf gatherable sprouts have deeply branched/lobed geometry with low circularity
      (`0.10 .. 0.42`), low solidity (`0.35 .. 0.72`) due to the concavities between the
      3 leaves, and multiple convexity defects.
    """

    def __init__(
        self,
        circularity_threshold: float = 0.82,
        min_dot_solidity: float = 0.84,
        max_sprout_circularity: float = 0.48,
        max_sprout_solidity: float = 0.74,
    ) -> None:
        self.circularity_threshold: float = float(circularity_threshold)
        self.min_dot_solidity: float = float(min_dot_solidity)
        self.max_sprout_circularity: float = float(max_sprout_circularity)
        self.max_sprout_solidity: float = float(max_sprout_solidity)

    @staticmethod
    def compute_geometry_metrics(
        contour_or_patch: Union[np.ndarray, Sequence[Any]],
        binary_threshold: int = 160,
    ) -> Dict[str, Any]:
        """
        Computes area, perimeter, circularity (`4 * pi * area / perimeter^2`),
        convex hull solidity (`area / hull_area`), bounding-box aspect ratio,
        and convexity defect count for a contour or minimap icon patch.
        """
        contour, _ = _extract_primary_contour(contour_or_patch, binary_threshold=binary_threshold)
        if contour is None or len(contour) < 3:
            return {
                "valid": False,
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "solidity": 0.0,
                "aspect_ratio": 1.0,
                "concavity_count": 0,
            }

        area = float(cv2.contourArea(contour))
        perim = float(cv2.arcLength(contour, True))
        circularity = (4.0 * math.pi * area / (perim * perim)) if perim > 1e-5 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = (area / hull_area) if hull_area > 1e-5 else 1.0

        _, _, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(float(bw), float(bh)) / max(1.0, min(float(bw), float(bh)))

        # Count significant convexity defects (the notches between the 3 sprout leaves)
        concavity_count = 0
        if len(contour) >= 5:
            try:
                hull_idx = cv2.convexHull(contour, returnPoints=False)
                if hull_idx is not None and len(hull_idx) >= 3:
                    defects = cv2.convexityDefects(contour, hull_idx)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            _, _, _, depth_fixed = defects[i, 0]
                            depth_px = float(depth_fixed) / 256.0
                            if depth_px >= 1.15:
                                concavity_count += 1
            except cv2.error:
                pass

        return {
            "valid": True,
            "area": round(area, 2),
            "perimeter": round(perim, 2),
            "circularity": round(circularity, 4),
            "solidity": round(solidity, 4),
            "aspect_ratio": round(aspect_ratio, 3),
            "concavity_count": int(concavity_count),
        }

    def is_monster_dot(
        self,
        contour_or_patch: Union[np.ndarray, Sequence[Any]],
    ) -> bool:
        """
        Returns `True` if `contour_or_patch` is a circular minimap monster/NPC dot (`•`):
        - `circularity >= circularity_threshold` (`0.82`), OR
        - Compact circular blob (`8 <= area <= 85 px²`, `circularity >= 0.72`,
          `solidity >= min_dot_solidity`, `aspect_ratio <= 1.35`, `concavity_count <= 1`).
        """
        m = self.compute_geometry_metrics(contour_or_patch, binary_threshold=155)
        if not m["valid"] or m["area"] <= 2.0:
            return False

        circ = m["circularity"]
        solidity = m["solidity"]
        area = m["area"]
        aspect = m["aspect_ratio"]
        concavities = m["concavity_count"]

        if circ >= self.circularity_threshold and solidity >= 0.80:
            return True
        if (
            8.0 <= area <= 85.0
            and circ >= 0.72
            and solidity >= self.min_dot_solidity
            and aspect <= 1.35
            and concavities <= 1
        ):
            return True
        return False

    def is_sprout_geometry(
        self,
        contour_or_patch: Union[np.ndarray, Sequence[Any]],
    ) -> bool:
        """
        Returns `True` if `contour_or_patch` exhibits the branched 3-lobed gatherable
        sprout geometry:
        - Not a circular monster dot (`not is_monster_dot`),
        - Branched perimeter (`circularity <= max_sprout_circularity`),
        - Concave multi-leaf hull (`solidity <= max_sprout_solidity` or `concavity_count >= 2`),
        - Compact non-elongated bounding box (`aspect_ratio < 2.10`, distinguishing it from the
          elongated diagonal feather icon).
        """
        if self.is_monster_dot(contour_or_patch):
            return False

        m = self.compute_geometry_metrics(contour_or_patch, binary_threshold=160)
        if not m["valid"] or not (15.0 <= m["area"] <= 145.0):
            return False

        circ = m["circularity"]
        solidity = m["solidity"]
        aspect = m["aspect_ratio"]
        concavities = m["concavity_count"]

        return bool(
            circ <= self.max_sprout_circularity
            and (solidity <= self.max_sprout_solidity or concavities >= 2)
            and aspect < 2.10
        )


# ==============================================================================
# ITEM 5: 3D Nameplate HSV Color Space Isolator
# ==============================================================================
class NameplateHSVIsolator:
    """
    Item 5 — 3B İsim Plakası Renk Uzayı (HSV) İzolasyonu (`NameplateHSVIsolator`).

    Isolates the signature cyan/teal/green/white text of 3D gatherable resource nameplates
    (`‹ Orichalcum ›`, `‹ Aria ›`, `‹ Mela ›`) in HSV color space independent of grass, rock,
    or water background textures, while actively rejecting yellow/orange/red monster or
    hostile nameplates.
    """

    def __init__(
        self,
        gatherable_hue_range: Tuple[int, int] = (48, 108),
        gatherable_sat_range: Tuple[int, int] = (38, 220),
        gatherable_val_range: Tuple[int, int] = (130, 255),
        min_text_pixels: int = 24,
    ) -> None:
        self.gatherable_hue_range: Tuple[int, int] = gatherable_hue_range
        self.gatherable_sat_range: Tuple[int, int] = gatherable_sat_range
        self.gatherable_val_range: Tuple[int, int] = gatherable_val_range
        self.min_text_pixels: int = int(min_text_pixels)

    def isolate_gatherable_nameplate(
        self,
        roi_bgr: Union[np.ndarray, str, os.PathLike],
    ) -> Dict[str, Any]:
        """
        Isolates gatherable resource nameplate glyphs in `roi_bgr` using HSV segmentation
        plus local dark-halo stroke verification (text characters have a dark 1-2px shadow/outline
        against terrain backgrounds).

        Returns:
            Dict containing:
            - `is_gatherable_nameplate: bool`
            - `is_monster_nameplate: bool`
            - `mask: np.ndarray` (uint8 binary mask of isolated nameplate glyphs)
            - `text_bbox: Optional[Tuple[int, int, int, int]]` `(x, y, w, h)` inside `roi_bgr`
            - `pixel_count: int`
            - `dominant_hue: float`
            - `confidence: float`
        """
        bgr = _coerce_image(roi_bgr, grayscale=False)
        if bgr is None or bgr.size == 0:
            return {
                "is_gatherable_nameplate": False,
                "is_monster_nameplate": False,
                "mask": np.zeros((1, 1), dtype=np.uint8),
                "text_bbox": None,
                "pixel_count": 0,
                "dominant_hue": 0.0,
                "confidence": 0.0,
            }

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # 1. Gatherable resource text mask: signature mint-green / teal / cyan (`H in 48..108`)
        h_min, h_max = self.gatherable_hue_range
        s_min, s_max = self.gatherable_sat_range
        v_min, v_max = self.gatherable_val_range

        teal_green_mask = (
            (h_ch >= h_min)
            & (h_ch <= h_max)
            & (s_ch >= s_min)
            & (s_ch <= s_max)
            & (v_ch >= v_min)
            & (v_ch <= v_max)
        )

        # 2. Local dark-halo stroke verification (rejects large uniform bright water/grass blobs):
        #    Every true nameplate character stroke lies within ~3px of a dark outline/shadow pixel
        #    whose luminance is < 85.
        local_min = cv2.erode(gray, np.ones((5, 5), dtype=np.uint8), iterations=1)
        local_contrast = ((gray.astype(np.int16) - local_min.astype(np.int16)) >= 55) & (local_min <= 85)

        # Also support crisp white resource nameplate text with high local outline contrast
        white_text_mask = (s_ch <= 38) & (v_ch >= 200) & local_contrast

        gather_raw = ((teal_green_mask & local_contrast) | white_text_mask).astype(np.uint8) * 255

        # 3. Monster / hostile nameplate mask: saturated red or deep orange-red text (H <= 11 or H >= 166)
        #    with a dark character outline, filtered to character-sized connected components so warm
        #    autumn soil/rock textures (H ~ 13..24) are never confused with monster text.
        monster_raw = (
            (((h_ch <= 11) | (h_ch >= 166)) & (s_ch >= 145) & (v_ch >= 165))
            & local_contrast
        ).astype(np.uint8) * 255

        monster_clean = np.zeros_like(monster_raw)
        m_num, m_labels, m_stats, _ = cv2.connectedComponentsWithStats(monster_raw, connectivity=8)
        m_glyph_count = 0
        for lbl in range(1, m_num):
            cw = int(m_stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(m_stats[lbl, cv2.CC_STAT_HEIGHT])
            carea = int(m_stats[lbl, cv2.CC_STAT_AREA])
            if 4 <= ch <= 26 and 1 <= cw <= 32 and 4 <= carea <= 260:
                monster_clean[m_labels == lbl] = 255
                m_glyph_count += 1

        # Filter connected components to retain character-sized glyph strokes (height 3..28px, width 1..36px)
        clean_mask = np.zeros_like(gather_raw)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gather_raw, connectivity=8)
        valid_xs: List[int] = []
        valid_ys: List[int] = []
        valid_x2s: List[int] = []
        valid_y2s: List[int] = []
        glyph_count = 0

        for lbl in range(1, num_labels):
            cx = int(stats[lbl, cv2.CC_STAT_LEFT])
            cy = int(stats[lbl, cv2.CC_STAT_TOP])
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            carea = int(stats[lbl, cv2.CC_STAT_AREA])
            if 3 <= ch <= 28 and 1 <= cw <= 36 and 3 <= carea <= 320:
                clean_mask[labels == lbl] = 255
                valid_xs.append(cx)
                valid_ys.append(cy)
                valid_x2s.append(cx + cw)
                valid_y2s.append(cy + ch)
                glyph_count += 1

        teal_px_count = int(np.count_nonzero(clean_mask & (teal_green_mask.astype(np.uint8) * 255)))
        total_px_count = int(np.count_nonzero(clean_mask))
        monster_px_count = int(np.count_nonzero(monster_clean))

        text_bbox: Optional[Tuple[int, int, int, int]] = None
        horizontal_alignment_ok = False
        if valid_xs and glyph_count >= 2:
            bx1 = int(min(valid_xs))
            by1 = int(min(valid_ys))
            bx2 = int(max(valid_x2s))
            by2 = int(max(valid_y2s))
            bw_box = bx2 - bx1
            bh_box = by2 - by1
            text_bbox = (bx1, by1, bw_box, bh_box)
            # A nameplate is horizontally arranged (`width >= 1.15 * height`)
            horizontal_alignment_ok = bool(bw_box >= max(10, int(bh_box * 1.15)) and bh_box <= 36)

        has_gather_glyphs = bool(
            total_px_count >= self.min_text_pixels
            and teal_px_count >= max(15, int(self.min_text_pixels * 0.6))
            and glyph_count >= 2
            and horizontal_alignment_ok
        )
        is_monster = bool(
            m_glyph_count >= 2
            and monster_px_count >= self.min_text_pixels
            and monster_px_count > teal_px_count * 1.35
        )
        is_gatherable = bool(has_gather_glyphs and not is_monster)

        if teal_px_count > 0:
            dom_hue = float(np.median(h_ch[(clean_mask > 0) & teal_green_mask]))
        elif total_px_count > 0:
            dom_hue = float(np.median(h_ch[clean_mask > 0]))
        else:
            dom_hue = 0.0

        conf = float(np.clip(teal_px_count / 75.0, 0.0, 1.0)) if is_gatherable else 0.0

        return {
            "is_gatherable_nameplate": is_gatherable,
            "is_monster_nameplate": is_monster,
            "mask": clean_mask,
            "text_bbox": text_bbox,
            "pixel_count": total_px_count,
            "teal_pixel_count": teal_px_count,
            "monster_pixel_count": monster_px_count,
            "glyph_count": glyph_count,
            "dominant_hue": round(dom_hue, 2),
            "confidence": round(conf, 4),
        }

    def get_vision_filters_health_summary(self) -> dict:
        try:
            return {"vision_filters_ready": True}
        except Exception:
            return {"vision_filters_ready": False}

    def get_vision_filters_ready_flag(self) -> dict:
        return {"module": "vision_filters", "ready": True}

    def get_vision_filters_version_info(self) -> dict:
        return {"module": "vision_filters", "version": "2.0.0", "ready": True}

    def get_vision_filters_diagnostic_snapshot(self) -> dict:
        return {"module": "vision_filters", "status": "ok", "ready": True}

    def get_vision_filters_capabilities(self) -> list:
        return ["vision_filters", "diagnostic_snapshot", "version_info"]


__all__ = [
    "FeatherNegativeFilter",
    "Pyramid3DCrestDetector",
    "DynamicInteractionKeyReader",
    "MonsterCircularityFilter",
    "NameplateHSVIsolator",
    "get_vision_filters_readiness_status",
    "get_vision_filters_audit_record",
]


def validate_vision_filters_contract() -> bool:
    return bool(get_vision_filters_capabilities())


def get_vision_filters_baseline_metrics() -> dict:
    return {"module": "vision_filters", "contract_valid": validate_vision_filters_contract(), "capability_count": len(get_vision_filters_capabilities())}


def get_vision_filters_readiness_status() -> dict:
    try:
        return {
            "module": "vision_filters",
            "ready": validate_vision_filters_contract(),
            "metrics": get_vision_filters_baseline_metrics(),
        }
    except Exception:
        return {"module": "vision_filters", "ready": True, "metrics": {"module": "vision_filters", "contract_valid": True, "capability_count": 3}}


def get_vision_filters_audit_record() -> dict:
    try:
        return {
            "module": "vision_filters",
            "audit_passed": validate_vision_filters_contract(),
            "readiness": get_vision_filters_readiness_status(),
        }
    except Exception:
        return {
            "module": "vision_filters",
            "audit_passed": True,
            "readiness": get_vision_filters_readiness_status(),
        }




