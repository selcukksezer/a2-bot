"""
Minimap Sprout vs. Monster Red Dot Discriminator.

Strictly discriminates between:
1. 3-leaf green/white resource sprout icons (`class_id=9`, `minimap_resource_sprout`)
2. Circular red/white monster dots (`class_id=11`, `minimap_monster_red_dot`, `circularity >= 0.65`)

while masking the circular minimap area and strictly excluding the central
player arrow zone (`r <= 14px` from minimap center).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# Shared 3x3 morphology kernel: morphologyEx never mutates its kernel, so
# one module-level array replaces a fresh np.ones((3, 3)) per discriminate()
# call.
_MORPH_KERNEL_3X3 = np.ones((3, 3), dtype=np.uint8)


CLASS_ID_MINIMAP_SPROUT: int = 9
CLASS_ID_MINIMAP_MONSTER: int = 11

CLASS_NAME_MINIMAP_SPROUT: str = "minimap_resource_sprout"
CLASS_NAME_MINIMAP_MONSTER: str = "minimap_monster_red_dot"


@dataclass(frozen=True)
class MinimapIconCandidate:
    """Represents a discriminated minimap icon (sprout or monster dot)."""

    class_id: int
    class_name: str
    center_xy: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]
    confidence: float
    circularity: float
    distance_from_center_px: float
    green_ratio: float = 0.0
    red_ratio: float = 0.0
    white_border_ratio: float = 0.0
    lobe_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bbox_xywh
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "center_xy": (int(self.center_xy[0]), int(self.center_xy[1])),
            "bbox": [int(x), int(y), int(w), int(h)],
            "bbox_xywh": (int(x), int(y), int(w), int(h)),
            "bbox_xyxy": (int(x), int(y), int(x + w), int(y + h)),
            "confidence": round(float(self.confidence), 4),
            "circularity": round(float(self.circularity), 4),
            "distance_from_center_px": round(float(self.distance_from_center_px), 3),
            "green_ratio": round(float(self.green_ratio), 4),
            "red_ratio": round(float(self.red_ratio), 4),
            "white_border_ratio": round(float(self.white_border_ratio), 4),
            "lobe_count": int(self.lobe_count),
        }

    def to_yolo_line(self, image_width: int, image_height: int) -> str:
        """Format candidate as a normalized YOLO annotation line."""
        w_img = max(1, int(image_width))
        h_img = max(1, int(image_height))
        x, y, w, h = self.bbox_xywh
        cx = max(0.0, min(1.0, (x + w * 0.5) / w_img))
        cy = max(0.0, min(1.0, (y + h * 0.5) / h_img))
        nw = max(0.001, min(1.0, w / w_img))
        nh = max(0.001, min(1.0, h / h_img))
        return f"{int(self.class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


@dataclass
class DiscriminationResult:
    """Structured output of a minimap frame discrimination pass."""

    sprouts: List[MinimapIconCandidate] = field(default_factory=list)
    monsters: List[MinimapIconCandidate] = field(default_factory=list)
    excluded_center_arrow_count: int = 0
    rejected_non_matching_count: int = 0
    minimap_center_xy: Tuple[int, int] = (0, 0)
    minimap_radius_px: float = 0.0
    player_arrow_exclusion_radius_px: float = 14.0

    @property
    def all_detections(self) -> List[MinimapIconCandidate]:
        return list(self.sprouts) + list(self.monsters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sprouts": [s.to_dict() for s in self.sprouts],
            "monsters": [m.to_dict() for m in self.monsters],
            "all_detections": [d.to_dict() for d in self.all_detections],
            "sprout_count": len(self.sprouts),
            "monster_count": len(self.monsters),
            "excluded_center_arrow_count": int(self.excluded_center_arrow_count),
            "rejected_non_matching_count": int(self.rejected_non_matching_count),
            "minimap_center_xy": (
                int(self.minimap_center_xy[0]),
                int(self.minimap_center_xy[1]),
            ),
            "minimap_radius_px": float(self.minimap_radius_px),
            "player_arrow_exclusion_radius_px": float(
                self.player_arrow_exclusion_radius_px
            ),
        }


class MinimapSproutMonsterDiscriminator:
    """
    Discriminates 3-leaf green/white sprouts (`class_id=9`) from circular
    red/white monster dots (`class_id=11`, `circularity >= 0.65`) inside the
    circular minimap while strictly excluding the central player arrow (`r <= 14px`).
    """

    SPROUT_CLASS_ID: int = CLASS_ID_MINIMAP_SPROUT
    MONSTER_CLASS_ID: int = CLASS_ID_MINIMAP_MONSTER

    # Last-geometry annular-mask cache, shared across instances (class level):
    # batch labelling calls discriminate() on frame after frame with identical
    # geometry, and rebuilding the full-frame mask each time is pure waste.
    # The cached array is READ-ONLY for callers (discriminate() only passes it
    # as a bitwise_and source, which never mutates it); the public
    # build_annular_minimap_mask() still returns a fresh array per call so
    # external mutation can never poison the cache. Single-entry on purpose:
    # one mask is bounded memory, and alternating geometries just rebuild.
    _annular_mask_cache_key: Optional[Tuple[int, int, int, int, float, int]] = None
    _annular_mask_cache: Optional[np.ndarray] = None
    _annular_mask_cache_lock = threading.Lock()

    @staticmethod
    def _draw_annular_mask(
        h: int,
        w: int,
        cx: int,
        cy: int,
        outer_r: float,
        inner_r: int,
    ) -> np.ndarray:
        """Draw the annular mask (minimap disc minus central player-arrow disc)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if outer_r > 0:
            cv2.circle(mask, (cx, cy), int(round(outer_r)), 255, thickness=-1)
        cv2.circle(mask, (cx, cy), int(inner_r), 0, thickness=-1)
        return mask

    @classmethod
    def _cached_annular_mask(
        cls,
        h: int,
        w: int,
        cx: int,
        cy: int,
        outer_r: float,
        inner_r: int,
    ) -> np.ndarray:
        """Return the annular mask for this geometry, reusing the last one built.

        Internal use only: the returned array is shared and must NOT be
        mutated by the caller.
        """
        key = (int(h), int(w), int(cx), int(cy), float(outer_r), int(inner_r))
        with cls._annular_mask_cache_lock:
            if cls._annular_mask_cache is not None and cls._annular_mask_cache_key == key:
                return cls._annular_mask_cache
            mask = cls._draw_annular_mask(h, w, cx, cy, outer_r, inner_r)
            cls._annular_mask_cache_key = key
            cls._annular_mask_cache = mask
            return mask

    def __init__(
        self,
        player_arrow_exclusion_radius_px: float = 14.0,
        min_monster_circularity: float = 0.65,
        minimap_radius_ratio: float = 0.47,
        sprout_hsv_lower: Tuple[int, int, int] = (32, 70, 95),
        sprout_hsv_upper: Tuple[int, int, int] = (92, 255, 255),
        monster_red_lower1: Tuple[int, int, int] = (0, 85, 95),
        monster_red_upper1: Tuple[int, int, int] = (12, 255, 255),
        monster_red_lower2: Tuple[int, int, int] = (166, 85, 95),
        monster_red_upper2: Tuple[int, int, int] = (180, 255, 255),
        white_hsv_lower: Tuple[int, int, int] = (0, 0, 175),
        white_hsv_upper: Tuple[int, int, int] = (180, 75, 255),
        min_sprout_area_px: int = 8,
        max_sprout_area_px: int = 260,
        min_monster_area_px: int = 6,
        max_monster_area_px: int = 180,
        require_white_outline: bool = True,
    ) -> None:
        if not math.isfinite(player_arrow_exclusion_radius_px) or player_arrow_exclusion_radius_px < 0.0:
            raise ValueError(
                "player_arrow_exclusion_radius_px must be a finite value >= 0.0, "
                f"got {player_arrow_exclusion_radius_px!r}"
            )
        if not math.isfinite(min_monster_circularity) or not 0.0 <= min_monster_circularity <= 1.0:
            raise ValueError(
                "min_monster_circularity must be a finite value in [0.0, 1.0], "
                f"got {min_monster_circularity!r}"
            )
        # Caller configuration is honoured as given: no silent upward rounding.
        self.player_arrow_exclusion_radius_px = float(player_arrow_exclusion_radius_px)
        self.min_monster_circularity = float(min_monster_circularity)
        self.minimap_radius_ratio = float(minimap_radius_ratio)

        self.sprout_hsv_lower = np.array(sprout_hsv_lower, dtype=np.uint8)
        self.sprout_hsv_upper = np.array(sprout_hsv_upper, dtype=np.uint8)

        self.monster_red_lower1 = np.array(monster_red_lower1, dtype=np.uint8)
        self.monster_red_upper1 = np.array(monster_red_upper1, dtype=np.uint8)
        self.monster_red_lower2 = np.array(monster_red_lower2, dtype=np.uint8)
        self.monster_red_upper2 = np.array(monster_red_upper2, dtype=np.uint8)

        self.white_hsv_lower = np.array(white_hsv_lower, dtype=np.uint8)
        self.white_hsv_upper = np.array(white_hsv_upper, dtype=np.uint8)

        self.min_sprout_area_px = int(min_sprout_area_px)
        self.max_sprout_area_px = int(max_sprout_area_px)
        self.min_monster_area_px = int(min_monster_area_px)
        self.max_monster_area_px = int(max_monster_area_px)
        self.require_white_outline = bool(require_white_outline)

    @staticmethod
    def resolve_minimap_geometry(
        frame_shape: Sequence[int],
        center_xy: Optional[Tuple[int, int]] = None,
        minimap_radius_px: Optional[float] = None,
        radius_ratio: float = 0.47,
    ) -> Tuple[Tuple[int, int], float]:
        """Return `(center_xy, minimap_radius_px)` for a minimap ROI."""
        h = int(frame_shape[0]) if len(frame_shape) >= 2 else 0
        w = int(frame_shape[1]) if len(frame_shape) >= 2 else 0
        cx = int(center_xy[0]) if center_xy is not None else w // 2
        cy = int(center_xy[1]) if center_xy is not None else h // 2
        if minimap_radius_px is not None and minimap_radius_px > 0:
            radius = float(minimap_radius_px)
        else:
            radius = float(min(w, h)) * float(radius_ratio)
        return (cx, cy), radius

    def is_inside_player_arrow_zone(
        self,
        point_xy: Tuple[float, float],
        minimap_center_xy: Tuple[float, float],
    ) -> bool:
        """
        Return True if `point_xy` lies within the central player arrow exclusion
        zone (`r <= 14px` or configured `player_arrow_exclusion_radius_px`).
        """
        dx = float(point_xy[0]) - float(minimap_center_xy[0])
        dy = float(point_xy[1]) - float(minimap_center_xy[1])
        dist = math.hypot(dx, dy)
        return dist <= self.player_arrow_exclusion_radius_px

    def is_within_minimap_annulus(
        self,
        point_xy: Tuple[float, float],
        minimap_center_xy: Tuple[float, float],
        minimap_radius_px: float,
    ) -> bool:
        """
        Return True iff `point_xy` is strictly outside the central player arrow
        (`r > 14px`) and inside the circular minimap area (`r <= minimap_radius_px`).
        """
        dx = float(point_xy[0]) - float(minimap_center_xy[0])
        dy = float(point_xy[1]) - float(minimap_center_xy[1])
        dist = math.hypot(dx, dy)
        return (
            dist > self.player_arrow_exclusion_radius_px
            and dist <= float(minimap_radius_px)
        )

    def build_annular_minimap_mask(
        self,
        frame_shape: Sequence[int],
        center_xy: Optional[Tuple[int, int]] = None,
        minimap_radius_px: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a uint8 binary mask (0 or 255) that enables pixels inside the
        minimap circle and masks out the central player arrow (`r <= 14px`).
        """
        h = int(frame_shape[0])
        w = int(frame_shape[1])
        (cx, cy), outer_r = self.resolve_minimap_geometry(
            frame_shape,
            center_xy=center_xy,
            minimap_radius_px=minimap_radius_px,
            radius_ratio=self.minimap_radius_ratio,
        )
        mask = self._draw_annular_mask(
            h,
            w,
            cx,
            cy,
            outer_r,
            int(math.ceil(self.player_arrow_exclusion_radius_px)),
        )
        return mask

    @staticmethod
    def compute_circularity(contour: np.ndarray) -> float:
        """
        Compute contour circularity clamped to $[0.0, 1.0]$.

        Base quantity is the isoperimetric ratio $4 \\pi A / P^2$, but that
        ratio alone cannot separate a square from a circle at these sizes:
        a filled square already scores $\\pi/4 \\approx 0.785$ while a real
        rasterized circle ($r=4$) scores $\\approx 0.766$ — the square would
        pass the $0.65$ monster threshold and the circle would score lower
        than the square. The old small-blob stabilization made it worse by
        taking ``max(raw, compact_est)``, lifting a $6\\times6$ square to
        $0.798$.

        The ratio is therefore penalized by the contour's fill inside its
        minimum enclosing circle. A filled square fills exactly
        $2/\\pi \\approx 0.637$ of its enclosing circle whatever its size,
        while rasterized circles of $r \\ge 3$ fill $\\ge \\approx 0.70$;
        ellipses/triangles/plus-shapes fill far less. A fill shortfall below
        $0.70$ subtracts $3\\times$ the shortfall, so squares land at
        $\\approx 0.60$ (below the $0.65$ gate) while genuine circles keep
        their raw ratio ($r=3$: $0.674$, $r=4$: $0.766$, $r=8$: $0.800$).
        """
        if contour is None or len(contour) < 3:
            return 0.0
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            # Zero-area contour (a line / degenerate rasterization) has no
            # circular shape at all: it must never score as a circular dot.
            return 0.0

        raw_circ = (4.0 * math.pi * area) / (perimeter * perimeter)

        # Enclosing-circle fill penalty (see docstring): squares and other
        # angular blobs fill distinctly less of their enclosing circle than
        # a real disc does, and must not pass the 0.65 circularity gate on
        # the isoperimetric ratio alone.
        (_, _), enc_r = cv2.minEnclosingCircle(contour)
        if enc_r > 0.0:
            circle_fill = min(1.0, area / (math.pi * enc_r * enc_r))
            if circle_fill < 0.70:
                raw_circ -= 3.0 * (0.70 - circle_fill)

        return float(max(0.0, min(1.0, raw_circ)))

    @staticmethod
    def estimate_sprout_lobe_count(
        contour: np.ndarray, binary_patch: Optional[np.ndarray] = None
    ) -> int:
        """
        Estimate leaf/lobe count for a candidate sprout contour.

        Real measurement (no constant fallthrough):
        1. Convexity defects — each deep concavity between leaves is one gap,
           so the number of deep defects (> 1.0 px) is the lobe count.
           A convex contour (no defects at all) is a single lobe.
        2. Radial-peak fallback — only when defects exist but none is deep:
           count local radial maxima from the centroid.
        Returns 1 when no lobe structure can be measured.
        """
        if contour is None or len(contour) < 5:
            return 1

        # 1. Convexity defects (handles both (n, 4) and (n, 1, 4) layouts).
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = None
            if hull is not None and len(hull) >= 3:
                defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            defects = None
        if defects is None:
            # Convex contour: no concavities, hence a single lobe.
            return 1
        deep_defects = 0
        try:
            flat = np.asarray(defects).reshape(-1, 4)
            for row in flat:
                if float(row[3]) / 256.0 > 1.0:
                    deep_defects += 1
        except (ValueError, IndexError):
            deep_defects = 0
        if deep_defects > 0:
            return int(min(5, deep_defects))

        # 2. Radial-peak fallback for near-convex shapes with shallow defects.
        pts = contour.reshape(-1, 2).astype(np.float32)
        centroid = np.mean(pts, axis=0)
        radii = np.hypot(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1])
        if len(radii) >= 6:
            mean_r = float(np.mean(radii))
            peaks = 0
            n = len(radii)
            for idx in range(n):
                prev_r = radii[(idx - 1) % n]
                curr_r = radii[idx]
                next_r = radii[(idx + 1) % n]
                if curr_r >= prev_r and curr_r > next_r and curr_r >= mean_r * 0.95:
                    peaks += 1
            if peaks >= 3:
                return min(5, peaks)

        return 1

    def _compute_local_ratios(
        self,
        green_mask: np.ndarray,
        red_mask: np.ndarray,
        hsv: np.ndarray,
        bbox_xywh: Tuple[int, int, int, int],
        pad: int = 2,
    ) -> Tuple[float, float, float]:
        """Return `(green_ratio, red_ratio, white_border_ratio)` around `bbox_xywh`.

        The white ratio is measured lazily from the small padded ROI of `hsv`
        instead of a full-frame white inRange: inRange is pixelwise, so
        counting white pixels inside the ROI gives byte-identical counts to
        slicing a full-frame white mask, while saving one full-frame inRange
        per discriminate() call.
        """
        h, w = green_mask.shape[:2]
        x, y, bw, bh = bbox_xywh
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        total_px = max(1, (x1 - x0) * (y1 - y0))
        core_px = max(1, bw * bh)

        green_core = np.count_nonzero(green_mask[y : y + bh, x : x + bw])
        red_core = np.count_nonzero(red_mask[y : y + bh, x : x + bw])
        white_neighborhood = np.count_nonzero(
            cv2.inRange(hsv[y0:y1, x0:x1], self.white_hsv_lower, self.white_hsv_upper)
        )

        return (
            float(green_core) / float(core_px),
            float(red_core) / float(core_px),
            float(white_neighborhood) / float(total_px),
        )

    def discriminate(
        self,
        minimap_bgr: np.ndarray,
        center_xy: Optional[Tuple[int, int]] = None,
        minimap_radius_px: Optional[float] = None,
    ) -> DiscriminationResult:
        """
        Run full minimap sprout (`class_id=9`) vs. monster red dot (`class_id=11`)
        discrimination on a BGR minimap image.

        - Excludes any candidate whose center is within `r <= 14px` of the minimap center.
        - Classifies 3-leaf green/white sprout icons as `class_id=9`.
        - Classifies circular (`circularity >= 0.65`) red/white monster dots as `class_id=11`.
        """
        if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
            return DiscriminationResult(
                player_arrow_exclusion_radius_px=self.player_arrow_exclusion_radius_px
            )

        if minimap_bgr.ndim == 2:
            bgr = cv2.cvtColor(minimap_bgr, cv2.COLOR_GRAY2BGR)
        elif minimap_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGRA2BGR)
        else:
            bgr = minimap_bgr

        (cx, cy), outer_r = self.resolve_minimap_geometry(
            bgr.shape,
            center_xy=center_xy,
            minimap_radius_px=minimap_radius_px,
            radius_ratio=self.minimap_radius_ratio,
        )

        result = DiscriminationResult(
            minimap_center_xy=(cx, cy),
            minimap_radius_px=outer_r,
            player_arrow_exclusion_radius_px=self.player_arrow_exclusion_radius_px,
        )

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build color masks across full minimap first so we can count center-arrow exclusions accurately
        green_mask = cv2.inRange(hsv, self.sprout_hsv_lower, self.sprout_hsv_upper)
        red_mask1 = cv2.inRange(hsv, self.monster_red_lower1, self.monster_red_upper1)
        red_mask2 = cv2.inRange(hsv, self.monster_red_lower2, self.monster_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # NOTE: no full-frame white mask here — _compute_local_ratios measures
        # the white ratio lazily from each candidate's small HSV ROI (identical
        # counts, one fewer full-frame inRange per call).

        green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL_3X3)
        red_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL_3X3)

        # Apply the annular minimap mask: only pixels inside the minimap
        # circle and outside the central player-arrow zone can form
        # candidates. (Per-candidate center checks below remain as a second
        # line of defence for contours straddling the mask boundary.)
        # The mask is cached per geometry: batch labelling reuses the same
        # frame size/center/radius for every frame. The cached array is only
        # read here (bitwise_and source), never mutated.
        annular_mask = self._cached_annular_mask(
            bgr.shape[0],
            bgr.shape[1],
            cx,
            cy,
            outer_r,
            int(math.ceil(self.player_arrow_exclusion_radius_px)),
        )
        green_clean = cv2.bitwise_and(green_clean, annular_mask)
        red_clean = cv2.bitwise_and(red_clean, annular_mask)

        # 1. Process Sprout Candidates (class_id = 9)
        green_contours, _ = cv2.findContours(
            green_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in green_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = max(float(cv2.contourArea(cnt)), float(np.count_nonzero(green_clean[y : y + h, x : x + w])))
            if area < self.min_sprout_area_px or area > self.max_sprout_area_px:
                result.rejected_non_matching_count += 1
                continue

            cand_cx = x + w * 0.5
            cand_cy = y + h * 0.5
            dist = math.hypot(cand_cx - cx, cand_cy - cy)

            # Strictly exclude central player arrow (r <= 14px)
            if dist <= self.player_arrow_exclusion_radius_px:
                result.excluded_center_arrow_count += 1
                continue
            if dist > outer_r:
                result.rejected_non_matching_count += 1
                continue

            green_ratio, red_ratio, white_ratio = self._compute_local_ratios(
                green_mask, red_mask, hsv, (x, y, w, h), pad=2
            )

            # Must be predominantly green (not red) and satisfy white highlight/outline if required
            if green_ratio <= red_ratio or green_ratio < 0.15:
                result.rejected_non_matching_count += 1
                continue
            if self.require_white_outline and white_ratio < 0.02:
                result.rejected_non_matching_count += 1
                continue

            circ = self.compute_circularity(cnt)
            patch = green_clean[y : y + h, x : x + w]
            lobes = self.estimate_sprout_lobe_count(cnt, patch)

            # Shape gate: a sprout is a compact 3-leaf icon, not any green
            # blotch. Reject elongated streaks and blobs without 3 lobes.
            aspect = (min(w, h) / max(w, h)) if w > 0 and h > 0 else 0.0
            if aspect < 0.4 or lobes < 3:
                result.rejected_non_matching_count += 1
                continue

            # Calibrated confidence: the weakest candidate that passes every
            # gate must not already score ~0.7. Base 0.35, full marks only
            # for strong green + white outline + 3 lobes.
            conf = min(
                0.99,
                0.35
                + 0.25 * min(1.0, green_ratio / 0.50)
                + 0.10 * min(1.0, white_ratio / 0.15)
                + 0.10 * (1.0 if lobes >= 3 else 0.0),
            )

            result.sprouts.append(
                MinimapIconCandidate(
                    class_id=CLASS_ID_MINIMAP_SPROUT,
                    class_name=CLASS_NAME_MINIMAP_SPROUT,
                    center_xy=(int(round(cand_cx)), int(round(cand_cy))),
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    confidence=conf,
                    circularity=circ,
                    distance_from_center_px=dist,
                    green_ratio=green_ratio,
                    red_ratio=red_ratio,
                    white_border_ratio=white_ratio,
                    lobe_count=lobes,
                )
            )

        # 2. Process Monster Red Dot Candidates (class_id = 11, circularity >= 0.65)
        red_contours, _ = cv2.findContours(
            red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in red_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = max(float(cv2.contourArea(cnt)), float(np.count_nonzero(red_clean[y : y + h, x : x + w])))
            if area < self.min_monster_area_px or area > self.max_monster_area_px:
                result.rejected_non_matching_count += 1
                continue

            cand_cx = x + w * 0.5
            cand_cy = y + h * 0.5
            dist = math.hypot(cand_cx - cx, cand_cy - cy)

            # Strictly exclude central player arrow (r <= 14px)
            if dist <= self.player_arrow_exclusion_radius_px:
                result.excluded_center_arrow_count += 1
                continue
            if dist > outer_r:
                result.rejected_non_matching_count += 1
                continue

            circ = self.compute_circularity(cnt)
            if circ < self.min_monster_circularity:
                result.rejected_non_matching_count += 1
                continue

            green_ratio, red_ratio, white_ratio = self._compute_local_ratios(
                green_mask, red_mask, hsv, (x, y, w, h), pad=2
            )

            if red_ratio <= green_ratio or red_ratio < 0.18:
                result.rejected_non_matching_count += 1
                continue
            if self.require_white_outline and white_ratio < 0.02:
                result.rejected_non_matching_count += 1
                continue

            # Ensure no overlap collision with an already accepted higher-green sprout
            collision = False
            for sp in result.sprouts:
                if math.hypot(cand_cx - sp.center_xy[0], cand_cy - sp.center_xy[1]) <= 4.0:
                    if sp.green_ratio >= red_ratio:
                        collision = True
                        break
            if collision:
                result.rejected_non_matching_count += 1
                continue

            # Calibrated confidence: the weakest candidate that passes every
            # gate must not already score ~0.7.
            conf = min(
                0.99,
                0.40
                + 0.25 * min(1.0, (circ - self.min_monster_circularity) / 0.35 + 0.5)
                + 0.15 * min(1.0, red_ratio / 0.50)
                + 0.10 * min(1.0, white_ratio / 0.15),
            )

            result.monsters.append(
                MinimapIconCandidate(
                    class_id=CLASS_ID_MINIMAP_MONSTER,
                    class_name=CLASS_NAME_MINIMAP_MONSTER,
                    center_xy=(int(round(cand_cx)), int(round(cand_cy))),
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    confidence=conf,
                    circularity=circ,
                    distance_from_center_px=dist,
                    green_ratio=green_ratio,
                    red_ratio=red_ratio,
                    white_border_ratio=white_ratio,
                    lobe_count=1,
                )
            )

        return result

    def classify_patch(
        self,
        patch_bgr: np.ndarray,
        candidate_center_xy: Optional[Tuple[float, float]] = None,
        minimap_center_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[int]:
        """
        Classify a single cropped icon patch into:
        - `9` (`CLASS_ID_MINIMAP_SPROUT`)
        - `11` (`CLASS_ID_MINIMAP_MONSTER`, when `circularity >= 0.65`)
        - `None` if inside the central player arrow exclusion zone (`r <= 14px`) or invalid.
        """
        if (
            candidate_center_xy is not None
            and minimap_center_xy is not None
            and self.is_inside_player_arrow_zone(candidate_center_xy, minimap_center_xy)
        ):
            return None

        if patch_bgr is None or not isinstance(patch_bgr, np.ndarray) or patch_bgr.size == 0:
            return None

        if patch_bgr.ndim == 2:
            bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
        elif patch_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_BGRA2BGR)
        else:
            bgr = patch_bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.sprout_hsv_lower, self.sprout_hsv_upper)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.monster_red_lower1, self.monster_red_upper1),
            cv2.inRange(hsv, self.monster_red_lower2, self.monster_red_upper2),
        )
        white_mask = cv2.inRange(hsv, self.white_hsv_lower, self.white_hsv_upper)

        green_px = int(np.count_nonzero(green_mask))
        red_px = int(np.count_nonzero(red_mask))
        white_px = int(np.count_nonzero(white_mask))

        if self.require_white_outline and white_px == 0:
            return None

        if green_px >= self.min_sprout_area_px and green_px > red_px:
            # Shape gate (same strictness as discriminate() /
            # discriminate_patch): a sprout is a compact 3-leaf icon, not any
            # green blotch. A plain green square has one lobe and must never
            # be accepted here — the production call site
            # (core/minimap_scanner.py) marks anything this returns 9 for as
            # three_leaf_verified, so an unshaped accept would certify a
            # square as a verified 3-leaf sprout.
            contours, _ = cv2.findContours(
                green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                aspect = (min(w, h) / max(w, h)) if w > 0 and h > 0 else 0.0
                lobes = self.estimate_sprout_lobe_count(largest)
                if aspect >= 0.4 and lobes >= 3:
                    return CLASS_ID_MINIMAP_SPROUT
            return None

        if red_px >= self.min_monster_area_px and red_px > green_px:
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                circ = self.compute_circularity(largest)
                if circ >= self.min_monster_circularity:
                    return CLASS_ID_MINIMAP_MONSTER

        return None

    def discriminate_patch(
        self,
        patch_bgr: np.ndarray,
        ncc_score: Optional[float] = None,
        chromatic_density: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify a single cropped icon patch as a 3-leaf sprout or a monster dot.

        Contract expected by the production call site
        (`core/minimap_scanner.py::scan_minimap_with_strict_discrimination`):
        returns a dict with ``label`` (``"minimap_sprout"`` /
        ``"minimap_monster_dot"`` / ``None``), ``is_valid_sprout``,
        ``reason``, ``class_id``, ``confidence``, ``lobe_count``,
        ``three_leaf_verified`` and ``circularity``.

        A patch is a valid sprout only when it is predominantly sprout-coloured
        (green/white/yellow, no hostile-red contamination), carries a white
        outline when required, and its largest contour really measures 3 lobes.
        ``ncc_score`` / ``chromatic_density`` are reported (and the supplied
        chromatic density is honoured as the density gate when given); they are
        not silently reinterpreted.
        """
        base: Dict[str, Any] = {
            "label": None,
            "is_valid_sprout": False,
            "reason": "empty_patch",
            "class_id": None,
            "confidence": 0.0,
            "lobe_count": 0,
            "three_leaf_verified": False,
            "circularity": 0.0,
            "ncc_score": ncc_score,
            "chromatic_density": chromatic_density,
        }
        if patch_bgr is None or not isinstance(patch_bgr, np.ndarray) or patch_bgr.size == 0:
            return dict(base)

        if patch_bgr.ndim == 2:
            bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
        elif patch_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_BGRA2BGR)
        else:
            bgr = patch_bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.monster_red_lower1, self.monster_red_upper1),
            cv2.inRange(hsv, self.monster_red_lower2, self.monster_red_upper2),
        )
        white_mask = cv2.inRange(hsv, self.white_hsv_lower, self.white_hsv_upper)
        # Scanner-compatible sprout chromaticity (cf. MinimapScanner
        #._check_sprout_chromatic_density): green S>=35, white/silver, yellow.
        green_m = (
            (hsv[:, :, 0] >= 32)
            & (hsv[:, :, 0] <= 85)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 70)
        )
        white_m = (hsv[:, :, 2] >= 170) & (hsv[:, :, 1] <= 70)
        yellow_m = (
            (hsv[:, :, 0] >= 16)
            & (hsv[:, :, 0] <= 35)
            & (hsv[:, :, 1] >= 50)
            & (hsv[:, :, 2] >= 85)
        )
        sprout_mask = ((green_m | white_m | yellow_m).astype(np.uint8)) * 255

        red_px = int(np.count_nonzero(red_mask))
        sprout_px = int(np.count_nonzero(sprout_mask))
        white_px = int(np.count_nonzero(white_mask))
        total_px = max(1, int(bgr.shape[0]) * int(bgr.shape[1]))

        # 1. Red monster dot: red-dominant and circular.
        if red_px >= self.min_monster_area_px and red_px > sprout_px:
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            circ = 0.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                circ = self.compute_circularity(largest)
            if circ >= self.min_monster_circularity:
                # Calibrated confidence, same structure as discriminate():
                # base 0.40 so the weakest candidate that passes every gate
                # does not already score ~0.78 (the old 0.55 + 0.35*circ
                # gave 0.7775 at the 0.65 threshold and 0.8293 to a square).
                red_ratio = red_px / float(total_px)
                white_ratio = white_px / float(total_px)
                conf = min(
                    0.99,
                    0.40
                    + 0.25 * min(1.0, (circ - self.min_monster_circularity) / 0.35 + 0.5)
                    + 0.15 * min(1.0, red_ratio / 0.50)
                    + 0.10 * min(1.0, white_ratio / 0.15),
                )
                out = dict(base)
                out.update(
                    {
                        "label": "minimap_monster_dot",
                        "is_valid_sprout": False,
                        "reason": "monster_dot",
                        "class_id": CLASS_ID_MINIMAP_MONSTER,
                        "confidence": round(float(conf), 4),
                        "circularity": round(float(circ), 4),
                    }
                )
                return out
            out = dict(base)
            out.update(
                {
                    "label": None,
                    "is_valid_sprout": False,
                    "reason": "red_non_circular",
                    "circularity": round(float(circ), 4),
                }
            )
            return out

        # 2. Hostile-red contamination can never be a sprout.
        if red_px >= 4:
            out = dict(base)
            out.update({"reason": "red_contamination"})
            return out

        # 3. Sprout chromatic density gate (0.06 mirrors the scanner check).
        density = float(chromatic_density) if chromatic_density is not None else float(sprout_px) / float(total_px)
        if sprout_px < self.min_sprout_area_px or density < 0.06:
            out = dict(base)
            out.update(
                {
                    "reason": "low_chromatic_density",
                    "chromatic_density": round(float(density), 4),
                }
            )
            return out

        if self.require_white_outline and white_px == 0:
            out = dict(base)
            out.update({"reason": "missing_white_outline"})
            return out

        # 4. Real 3-leaf shape check on the largest sprout contour.
        contours, _ = cv2.findContours(
            sprout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        lobes = 0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            lobes = self.estimate_sprout_lobe_count(largest, sprout_mask)
        three_leaf = lobes >= 3
        if not three_leaf:
            out = dict(base)
            out.update(
                {
                    "reason": f"lobe_count_{lobes}_not_three_leaf",
                    "lobe_count": int(lobes),
                    "three_leaf_verified": False,
                    "chromatic_density": round(float(density), 4),
                }
            )
            return out

        conf = min(
            0.99,
            0.50
            + 0.25 * min(1.0, density / 0.30)
            + 0.15 * min(1.0, white_px / max(1, sprout_px)),
        )
        out = dict(base)
        out.update(
            {
                "label": "minimap_sprout",
                "is_valid_sprout": True,
                "reason": "ok",
                "class_id": CLASS_ID_MINIMAP_SPROUT,
                "confidence": round(float(conf), 4),
                "lobe_count": int(lobes),
                "three_leaf_verified": True,
                "chromatic_density": round(float(density), 4),
            }
        )
        return out

    def to_yolo_annotations(
        self,
        minimap_bgr: np.ndarray,
        center_xy: Optional[Tuple[int, int]] = None,
        minimap_radius_px: Optional[float] = None,
    ) -> List[str]:
        """Run discrimination and return normalized YOLO annotation lines."""
        if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
            return []
        h, w = minimap_bgr.shape[:2]
        disc = self.discriminate(
            minimap_bgr, center_xy=center_xy, minimap_radius_px=minimap_radius_px
        )
        return [cand.to_yolo_line(w, h) for cand in disc.all_detections]


def _get_sprout_monster_discriminator() -> MinimapSproutMonsterDiscriminator:
    """Factory used by the production strict-discrimination path."""
    return MinimapSproutMonsterDiscriminator()


__all__ = [
    "CLASS_ID_MINIMAP_SPROUT",
    "CLASS_ID_MINIMAP_MONSTER",
    "CLASS_NAME_MINIMAP_SPROUT",
    "CLASS_NAME_MINIMAP_MONSTER",
    "MinimapIconCandidate",
    "DiscriminationResult",
    "MinimapSproutMonsterDiscriminator",
    "_get_sprout_monster_discriminator",
]
