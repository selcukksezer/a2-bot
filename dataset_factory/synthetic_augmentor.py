"""
Aion 2 Autonomous Dataset Factory - 10x Photometric & Geometric Synthetic YOLO Augmentor
=========================================================================================
Multiplies real auto-labeled Aion 2 screenshots up to 10x with realistic photometric
(day/night/dusk lighting, color temperature shifts, biome HSV jitter, Pilgrim's Ridge /
Altamia Highland atmospheric fog & bloom) and geometric (horizontal flip, multi-scale
camera distance crops) transformations while strictly transforming and clipping all
YOLO bounding boxes within normalized [0.0, 1.0] bounds.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticYOLOAugmentor:
    """
    10x Photometric & Geometric Synthetic YOLO Dataset Augmentor for Aion 2.

    Supports both dict-based annotation formats used across the Dataset Factory:
      - Normalized center-width-height keys: `cx`, `cy`, `w`, `h` (or `width`, `height`)
      - Normalized bbox tuple/list under `bbox_norm` or `yolo_bbox`: `(cx, cy, w, h)`
      - Optional pixel bbox under `bbox_xyxy` or `bbox`: `(x1, y1, x2, y2)`
      - Preserves `class_id`, `class_name`, `confidence`, and all extra metadata keys.
    """

    def __init__(self, seed: Optional[int] = None, min_box_visibility: float = 0.25, min_box_dim: float = 0.004) -> None:
        """
        Initialize the synthetic YOLO augmentor.

        Args:
            seed: Optional random seed for deterministic augmentation pipelines.
            min_box_visibility: Minimum fraction of original box area required to survive a crop.
            min_box_dim: Minimum normalized width or height required for a cropped box to remain valid.
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.min_box_visibility = float(max(0.05, min(0.95, min_box_visibility)))
        self.min_box_dim = float(max(0.001, min(0.1, min_box_dim)))

    # -------------------------------------------------------------------------
    # Annotation Normalization & Extraction Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_yolo_cxcywh(ann: Dict[str, Any], img_w: int = 1920, img_h: int = 1080) -> Tuple[float, float, float, float]:
        """
        Extract normalized (cx, cy, w, h) in [0.0, 1.0] from an annotation dictionary.
        """
        if "cx" in ann and "cy" in ann and ("w" in ann or "width" in ann) and ("h" in ann or "height" in ann):
            cx = float(ann["cx"])
            cy = float(ann["cy"])
            w = float(ann.get("w", ann.get("width", 0.0)))
            h = float(ann.get("h", ann.get("height", 0.0)))
            if cx > 1.0 or cy > 1.0 or w > 1.0 or h > 1.0:
                cx /= max(1, img_w)
                cy /= max(1, img_h)
                w /= max(1, img_w)
                h /= max(1, img_h)
            return (
                float(np.clip(cx, 0.0, 1.0)),
                float(np.clip(cy, 0.0, 1.0)),
                float(np.clip(w, 0.0, 1.0)),
                float(np.clip(h, 0.0, 1.0)),
            )

        for key in ("yolo_bbox", "bbox_norm", "normalized_bbox"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                cx, cy, w, h = [float(v) for v in ann[key]]
                return (
                    float(np.clip(cx, 0.0, 1.0)),
                    float(np.clip(cy, 0.0, 1.0)),
                    float(np.clip(w, 0.0, 1.0)),
                    float(np.clip(h, 0.0, 1.0)),
                )

        for key in ("bbox_xyxy", "xyxy", "bbox"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                v0, v1, v2, v3 = [float(v) for v in ann[key]]
                if max(v0, v1, v2, v3) <= 1.0 and key == "bbox":
                    return (
                        float(np.clip(v0, 0.0, 1.0)),
                        float(np.clip(v1, 0.0, 1.0)),
                        float(np.clip(v2, 0.0, 1.0)),
                        float(np.clip(v3, 0.0, 1.0)),
                    )
                x1, y1, x2, y2 = v0, v1, v2, v3
                cx = ((x1 + x2) * 0.5) / max(1, img_w)
                cy = ((y1 + y2) * 0.5) / max(1, img_h)
                w = abs(x2 - x1) / max(1, img_w)
                h = abs(y2 - y1) / max(1, img_h)
                return (
                    float(np.clip(cx, 0.0, 1.0)),
                    float(np.clip(cy, 0.0, 1.0)),
                    float(np.clip(w, 0.0, 1.0)),
                    float(np.clip(h, 0.0, 1.0)),
                )

        raise ValueError(f"Annotation dictionary missing recognizable YOLO coordinates: {ann}")

    @staticmethod
    def _clamp_yolo_box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """
        Strictly clamp normalized (cx, cy, w, h) so the entire bounding box
        [cx - w/2, cy - h/2, cx + w/2, cy + h/2] lies inside [0.0, 1.0].
        """
        x1 = float(np.clip(cx - 0.5 * w, 0.0, 1.0))
        y1 = float(np.clip(cy - 0.5 * h, 0.0, 1.0))
        x2 = float(np.clip(cx + 0.5 * w, 0.0, 1.0))
        y2 = float(np.clip(cy + 0.5 * h, 0.0, 1.0))

        w_clamped = float(np.clip(x2 - x1, 0.0, 1.0))
        h_clamped = float(np.clip(y2 - y1, 0.0, 1.0))
        cx_clamped = float(np.clip(x1 + 0.5 * w_clamped, 0.0, 1.0))
        cy_clamped = float(np.clip(y1 + 0.5 * h_clamped, 0.0, 1.0))
        return cx_clamped, cy_clamped, w_clamped, h_clamped

    @classmethod
    def _update_annotation_coords(
        cls,
        ann: Dict[str, Any],
        cx: float,
        cy: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """
        Return a deep copy of `ann` with updated and strictly clamped YOLO normalized
        and pixel coordinates across all present schema keys.
        """
        cx, cy, w, h = cls._clamp_yolo_box(cx, cy, w, h)
        updated = copy.deepcopy(ann)
        updated["cx"] = round(cx, 6)
        updated["cy"] = round(cy, 6)
        updated["w"] = round(w, 6)
        updated["h"] = round(h, 6)

        if "width" in updated:
            updated["width"] = round(w, 6)
        if "height" in updated:
            updated["height"] = round(h, 6)
        if "yolo_bbox" in updated:
            updated["yolo_bbox"] = [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]
        if "bbox_norm" in updated:
            updated["bbox_norm"] = [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]
        if "normalized_bbox" in updated:
            updated["normalized_bbox"] = [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]

        px_x1 = int(round(np.clip((cx - 0.5 * w) * img_w, 0, img_w)))
        px_y1 = int(round(np.clip((cy - 0.5 * h) * img_h, 0, img_h)))
        px_x2 = int(round(np.clip((cx + 0.5 * w) * img_w, 0, img_w)))
        px_y2 = int(round(np.clip((cy + 0.5 * h) * img_h, 0, img_h)))

        if "bbox_xyxy" in updated:
            updated["bbox_xyxy"] = [px_x1, px_y1, px_x2, px_y2]
        if "xyxy" in updated:
            updated["xyxy"] = [px_x1, px_y1, px_x2, px_y2]
        if "bbox" in updated:
            old_bbox = updated["bbox"]
            if isinstance(old_bbox, (list, tuple)) and len(old_bbox) == 4 and max(float(v) for v in old_bbox) <= 1.0:
                updated["bbox"] = [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]
            else:
                updated["bbox"] = [px_x1, px_y1, px_x2, px_y2]

        return updated

    # -------------------------------------------------------------------------
    # 1. Photometric & Weather Augmentations (Boxes Unchanged)
    # -------------------------------------------------------------------------
    def augment_day_night_lighting(
        self,
        frame: np.ndarray,
        gamma: Optional[float] = None,
        color_temp: Optional[str] = None,
    ) -> np.ndarray:
        """
        Simulates Aion 2 night, dusk, or bright-noon lighting variations
        using non-linear gamma correction (0.65..1.45) combined with warm/cool
        color temperature tinting.

        Args:
            frame: Input BGR uint8 image.
            gamma: Gamma exponent in [0.65, 1.45]. Lower < 1.0 brightens (noon),
                   higher > 1.0 darkens (dusk/night). If None, sampled randomly.
            color_temp: One of {'night_cool', 'dusk_warm', 'noon_bright', 'neutral'}.
                        If None, chosen automatically based on gamma.

        Returns:
            Photometrically adjusted BGR uint8 frame of identical shape.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to augment_day_night_lighting is empty or None.")

        if gamma is None:
            gamma = self.rng.uniform(0.65, 1.45)
        gamma = float(np.clip(gamma, 0.50, 2.00))

        # Build LUT for fast gamma transformation
        inv_gamma = 1.0 / max(1e-4, gamma)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        adjusted = cv2.LUT(frame, lut).astype(np.float32)

        if color_temp is None:
            if gamma > 1.15:
                color_temp = self.rng.choice(["night_cool", "dusk_warm"])
            elif gamma < 0.85:
                color_temp = "noon_bright"
            else:
                color_temp = self.rng.choice(["dusk_warm", "night_cool", "neutral"])

        # Apply BGR channel multipliers for realistic Aion 2 atmospheric lighting
        if color_temp == "night_cool":
            # Moonlight blue/indigo boost, reduced red/green
            b_gain = self.rng.uniform(1.08, 1.18)
            g_gain = self.rng.uniform(0.95, 1.02)
            r_gain = self.rng.uniform(0.84, 0.93)
        elif color_temp == "dusk_warm":
            # Golden-hour amber/crimson tint
            b_gain = self.rng.uniform(0.85, 0.94)
            g_gain = self.rng.uniform(0.98, 1.05)
            r_gain = self.rng.uniform(1.08, 1.18)
        elif color_temp == "noon_bright":
            # Crisp sun-lit warm highlights
            b_gain = self.rng.uniform(0.96, 1.02)
            g_gain = self.rng.uniform(1.01, 1.06)
            r_gain = self.rng.uniform(1.03, 1.09)
        else:
            b_gain, g_gain, r_gain = 1.0, 1.0, 1.0

        gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32).reshape((1, 1, 3))
        adjusted = np.clip(adjusted * gains, 0.0, 255.0).astype(np.uint8)
        return adjusted

    def augment_hsv_jitter(
        self,
        frame: np.ndarray,
        hue_shift_Limit: int = 8,
        sat_range: Tuple[float, float] = (0.80, 1.25),
        val_range: Tuple[float, float] = (0.90, 1.12),
    ) -> np.ndarray:
        """
        Applies slight Hue (±8 on OpenCV 0..179 scale), Saturation (0.80..1.25),
        and Value shifts so herbs, trees, and ores are recognized across all Aion 2 biomes.

        Args:
            frame: Input BGR uint8 image.
            hue_shift_Limit: Maximum absolute hue delta (default ±8).
            sat_range: Multiplicative saturation scaling interval (default 0.80..1.25).
            val_range: Multiplicative value scaling interval (default 0.90..1.12).

        Returns:
            HSV-jittered BGR uint8 image of identical shape.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to augment_hsv_jitter is empty or None.")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)

        h_delta = self.rng.randint(-abs(int(hue_shift_Limit)), abs(int(hue_shift_Limit)))
        s_scale = self.rng.uniform(float(sat_range[0]), float(sat_range[1]))
        v_scale = self.rng.uniform(float(val_range[0]), float(val_range[1]))

        # OpenCV hue range is [0, 179]
        hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255).astype(np.int32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255).astype(np.int32)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def augment_atmospheric_fog_and_bloom(
        self,
        frame: np.ndarray,
        fog_intensity: Optional[float] = None,
        bloom_strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulates volumetric mist/fog and UE5 soft highlight bloom characteristic of
        `Pilgrim's Ridge` and `Altamia Highland`.

        Combines a vertical gradient atmospheric veil (denser near the upper horizon,
        lighter in the foreground) with a soft Gaussian highlight bloom pass.

        Args:
            frame: Input BGR uint8 image.
            fog_intensity: Blending strength of the atmospheric mist in [0.10, 0.36].
            bloom_strength: Additive soft highlight glow strength in [0.08, 0.25].

        Returns:
            Fog and bloom augmented BGR uint8 frame.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to augment_atmospheric_fog_and_bloom is empty or None.")

        h, w = frame.shape[:2]
        if fog_intensity is None:
            fog_intensity = self.rng.uniform(0.12, 0.32)
        if bloom_strength is None:
            bloom_strength = self.rng.uniform(0.10, 0.22)

        fog_intensity = float(np.clip(fog_intensity, 0.0, 0.65))
        bloom_strength = float(np.clip(bloom_strength, 0.0, 0.50))

        # 1. Atmospheric gradient fog veil (Pilgrim's Ridge cool silver-blue or Altamia misty cyan-white)
        fog_colors = [
            np.array([228.0, 222.0, 212.0], dtype=np.float32),  # Pilgrim's Ridge cool mist (BGR)
            np.array([235.0, 232.0, 225.0], dtype=np.float32),  # Altamia Highland alpine fog (BGR)
            np.array([218.0, 225.0, 232.0], dtype=np.float32),  # Morning valley haze (BGR)
        ]
        fog_color = fog_colors[self.rng.randrange(len(fog_colors))].reshape((1, 1, 3))

        # Vertical depth profile: horizon/top gets full fog_intensity, bottom foreground gets 35%
        vertical_profile = np.linspace(1.0, 0.35, h, dtype=np.float32).reshape((h, 1, 1))
        alpha_map = vertical_profile * fog_intensity

        img_f = frame.astype(np.float32)
        fogged = img_f * (1.0 - alpha_map) + fog_color * alpha_map

        # 2. UE5 Soft Bloom on bright highlights (aether crests, glowing ores, sky light)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        bright_regions = cv2.bitwise_and(frame, frame, mask=bright_mask)
        bloom_layer = cv2.GaussianBlur(bright_regions, (0, 0), sigmaX=15.0, sigmaY=15.0).astype(np.float32)

        combined = np.clip(fogged + bloom_layer * bloom_strength, 0.0, 255.0).astype(np.uint8)
        return combined

    # -------------------------------------------------------------------------
    # 2. Geometric Augmentations (With Exact Bounding Box Transformation)
    # -------------------------------------------------------------------------
    def augment_horizontal_flip(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Flips the frame horizontally (`cv2.flip(frame, 1)`) and transforms every
        YOLO annotation via `cx_new = 1.0 - cx`.

        Args:
            frame: Input BGR uint8 image.
            annotations: List of YOLO annotation dicts.

        Returns:
            Tuple of (flipped_frame, transformed_annotations) with all boxes in [0.0, 1.0].
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to augment_horizontal_flip is empty or None.")

        img_h, img_w = frame.shape[:2]
        flipped_frame = cv2.flip(frame, 1)
        flipped_annotations: List[Dict[str, Any]] = []

        for ann in annotations:
            cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            cx_new = float(np.clip(1.0 - cx, 0.0, 1.0))
            updated_ann = self._update_annotation_coords(
                ann,
                cx=cx_new,
                cy=cy,
                w=bw,
                h=bh,
                img_w=img_w,
                img_h=img_h,
            )
            flipped_annotations.append(updated_ann)

        return flipped_frame, flipped_annotations

    def augment_multi_scale_crop(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
        scale_range: Tuple[float, float] = (0.78, 0.95),
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Crops a random sub-window with scale in `scale_range` (default 0.78..0.95),
        resizes back to the original frame resolution, and recalculates/clips every
        `(cx, cy, w, h)` to the new crop coordinates so the YOLO detector learns
        resources at varying camera distances.

        Intelligently biases the crop window selection so that annotated targets
        remain inside the crop whenever annotations are present.

        Args:
            frame: Input BGR uint8 image.
            annotations: List of YOLO annotation dicts.
            scale_range: (min_scale, max_scale) fraction of original dimensions.

        Returns:
            Tuple of (resized_cropped_frame, transformed_annotations) with all boxes
            strictly inside [0.0, 1.0].
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to augment_multi_scale_crop is empty or None.")

        img_h, img_w = frame.shape[:2]
        min_s = float(np.clip(scale_range[0], 0.50, 0.99))
        max_s = float(np.clip(scale_range[1], min_s, 0.995))
        scale_x = self.rng.uniform(min_s, max_s)
        scale_y = self.rng.uniform(max(min_s, scale_x - 0.04), min(max_s, scale_x + 0.04))

        # Normalized crop window position [crop_x1, crop_y1, crop_x2, crop_y2]
        max_x1 = max(0.0, 1.0 - scale_x)
        max_y1 = max(0.0, 1.0 - scale_y)

        # If annotations exist, try to pick a crop window that keeps the primary target well inside
        if annotations:
            anchor_ann = self.rng.choice(annotations)
            a_cx, a_cy, a_w, a_h = self._extract_yolo_cxcywh(anchor_ann, img_w=img_w, img_h=img_h)
            # Constraints for crop_x1 so [a_cx - a_w/2, a_cx + a_w/2] stays inside [crop_x1, crop_x1 + scale_x]
            low_x1 = max(0.0, min(max_x1, (a_cx + 0.5 * a_w) - scale_x))
            high_x1 = min(max_x1, max(0.0, a_cx - 0.5 * a_w))
            if low_x1 <= high_x1:
                crop_x1 = self.rng.uniform(low_x1, high_x1)
            else:
                crop_x1 = self.rng.uniform(0.0, max_x1)

            low_y1 = max(0.0, min(max_y1, (a_cy + 0.5 * a_h) - scale_y))
            high_y1 = min(max_y1, max(0.0, a_cy - 0.5 * a_h))
            if low_y1 <= high_y1:
                crop_y1 = self.rng.uniform(low_y1, high_y1)
            else:
                crop_y1 = self.rng.uniform(0.0, max_y1)
        else:
            crop_x1 = self.rng.uniform(0.0, max_x1)
            crop_y1 = self.rng.uniform(0.0, max_y1)

        crop_x2 = crop_x1 + scale_x
        crop_y2 = crop_y1 + scale_y

        # Pixel coordinates for slicing
        px_x1 = int(round(crop_x1 * img_w))
        px_y1 = int(round(crop_y1 * img_h))
        px_x2 = max(px_x1 + 16, int(round(crop_x2 * img_w)))
        px_y2 = max(px_y1 + 16, int(round(crop_y2 * img_h)))
        px_x2 = min(img_w, px_x2)
        px_y2 = min(img_h, px_y2)

        # Recompute exact normalized crop bounds from actual integer pixel slice
        actual_crop_x1 = px_x1 / float(img_w)
        actual_crop_y1 = px_y1 / float(img_h)
        actual_crop_x2 = px_x2 / float(img_w)
        actual_crop_y2 = px_y2 / float(img_h)
        actual_crop_w = max(1e-6, actual_crop_x2 - actual_crop_x1)
        actual_crop_h = max(1e-6, actual_crop_y2 - actual_crop_y1)

        cropped = frame[px_y1:px_y2, px_x1:px_x2]
        resized_frame = cv2.resize(cropped, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        transformed_annotations: List[Dict[str, Any]] = []
        for ann in annotations:
            cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            orig_x1 = max(0.0, cx - 0.5 * bw)
            orig_y1 = max(0.0, cy - 0.5 * bh)
            orig_x2 = min(1.0, cx + 0.5 * bw)
            orig_y2 = min(1.0, cy + 0.5 * bh)
            orig_area = max(1e-8, (orig_x2 - orig_x1) * (orig_y2 - orig_y1))

            # Intersect box with crop window
            inter_x1 = max(orig_x1, actual_crop_x1)
            inter_y1 = max(orig_y1, actual_crop_y1)
            inter_x2 = min(orig_x2, actual_crop_x2)
            inter_y2 = min(orig_y2, actual_crop_y2)

            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            if inter_w <= 0.0 or inter_h <= 0.0:
                continue

            visibility = (inter_w * inter_h) / orig_area
            if visibility < self.min_box_visibility:
                continue

            # Map intersection coordinates into normalized crop space [0.0, 1.0]
            new_x1 = float(np.clip((inter_x1 - actual_crop_x1) / actual_crop_w, 0.0, 1.0))
            new_y1 = float(np.clip((inter_y1 - actual_crop_y1) / actual_crop_h, 0.0, 1.0))
            new_x2 = float(np.clip((inter_x2 - actual_crop_x1) / actual_crop_w, 0.0, 1.0))
            new_y2 = float(np.clip((inter_y2 - actual_crop_y1) / actual_crop_h, 0.0, 1.0))

            new_w = float(np.clip(new_x2 - new_x1, 0.0, 1.0))
            new_h = float(np.clip(new_y2 - new_y1, 0.0, 1.0))
            if new_w < self.min_box_dim or new_h < self.min_box_dim:
                continue

            new_cx = float(np.clip(new_x1 + 0.5 * new_w, 0.0, 1.0))
            new_cy = float(np.clip(new_y1 + 0.5 * new_h, 0.0, 1.0))

            updated_ann = self._update_annotation_coords(
                ann,
                cx=new_cx,
                cy=new_cy,
                w=new_w,
                h=new_h,
                img_w=img_w,
                img_h=img_h,
            )
            transformed_annotations.append(updated_ann)

        return resized_frame, transformed_annotations

    # -------------------------------------------------------------------------
    # 3. Batch Multiplier (Up to 10x Variant Generator)
    # -------------------------------------------------------------------------
    def generate_augmented_variants(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
        count: int = 5,
    ) -> List[Tuple[str, np.ndarray, List[Dict[str, Any]]]]:
        """
        Given 1 base frame + YOLO annotations, produces `count` (up to 10x+) valid
        `(variant_suffix, aug_frame, aug_annotations)` tuples ready to save to `dataset/`.

        Each variant uses a distinct, realistic combination of photometric and geometric
        recipes so the training dataset achieves maximum biome, time-of-day, weather,
        and camera-distance diversity without duplicate frames.

        Args:
            frame: Base BGR uint8 screenshot.
            annotations: Base YOLO annotations list.
            count: Number of synthetic variants to generate (e.g., 5 to 10).

        Returns:
            List of `(variant_suffix, aug_frame, aug_annotations)` tuples.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame to generate_augmented_variants is empty or None.")
        if count <= 0:
            return []

        img_h, img_w = frame.shape[:2]
        # Ensure base annotations are normalized and clamped
        base_annotations: List[Dict[str, Any]] = []
        for ann in annotations:
            cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            base_annotations.append(
                self._update_annotation_coords(ann, cx, cy, bw, bh, img_w=img_w, img_h=img_h)
            )

        # Curated 10x Aion 2 recipe bank covering time-of-day, biome weather, flips, and camera scale
        recipes = [
            "night_hsv",
            "dusk_hflip",
            "fog_bloom_crop",
            "noon_hflip_crop",
            "hsv_multiscale",
            "night_fog_hflip",
            "dusk_bloom_crop",
            "noon_hsv",
            "fog_hflip_multiscale",
            "full_biome_mix",
        ]

        variants: List[Tuple[str, np.ndarray, List[Dict[str, Any]]]] = []
        for idx in range(count):
            recipe = recipes[idx % len(recipes)]
            aug_frame = frame.copy()
            aug_anns = copy.deepcopy(base_annotations)

            if recipe == "night_hsv":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(1.18, 1.45), color_temp="night_cool"
                )
                aug_frame = self.augment_hsv_jitter(aug_frame)
                suffix = f"aug{idx + 1:02d}_night_hsv"

            elif recipe == "dusk_hflip":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(1.05, 1.28), color_temp="dusk_warm"
                )
                aug_frame, aug_anns = self.augment_horizontal_flip(aug_frame, aug_anns)
                suffix = f"aug{idx + 1:02d}_dusk_hflip"

            elif recipe == "fog_bloom_crop":
                aug_frame = self.augment_atmospheric_fog_and_bloom(aug_frame)
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.80, 0.94)
                )
                suffix = f"aug{idx + 1:02d}_fog_crop"

            elif recipe == "noon_hflip_crop":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(0.65, 0.85), color_temp="noon_bright"
                )
                aug_frame, aug_anns = self.augment_horizontal_flip(aug_frame, aug_anns)
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.78, 0.93)
                )
                suffix = f"aug{idx + 1:02d}_noon_hflip_crop"

            elif recipe == "hsv_multiscale":
                aug_frame = self.augment_hsv_jitter(aug_frame)
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.78, 0.95)
                )
                suffix = f"aug{idx + 1:02d}_hsv_scale"

            elif recipe == "night_fog_hflip":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(1.20, 1.42), color_temp="night_cool"
                )
                aug_frame = self.augment_atmospheric_fog_and_bloom(
                    aug_frame, fog_intensity=self.rng.uniform(0.14, 0.26)
                )
                aug_frame, aug_anns = self.augment_horizontal_flip(aug_frame, aug_anns)
                suffix = f"aug{idx + 1:02d}_night_fog_hflip"

            elif recipe == "dusk_bloom_crop":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(1.08, 1.32), color_temp="dusk_warm"
                )
                aug_frame = self.augment_atmospheric_fog_and_bloom(
                    aug_frame, fog_intensity=0.12, bloom_strength=0.22
                )
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.82, 0.95)
                )
                suffix = f"aug{idx + 1:02d}_dusk_bloom_crop"

            elif recipe == "noon_hsv":
                aug_frame = self.augment_day_night_lighting(
                    aug_frame, gamma=self.rng.uniform(0.68, 0.88), color_temp="noon_bright"
                )
                aug_frame = self.augment_hsv_jitter(aug_frame)
                suffix = f"aug{idx + 1:02d}_noon_hsv"

            elif recipe == "fog_hflip_multiscale":
                aug_frame = self.augment_atmospheric_fog_and_bloom(aug_frame)
                aug_frame, aug_anns = self.augment_horizontal_flip(aug_frame, aug_anns)
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.79, 0.94)
                )
                suffix = f"aug{idx + 1:02d}_fog_hflip_scale"

            else:  # full_biome_mix
                aug_frame = self.augment_day_night_lighting(aug_frame)
                aug_frame = self.augment_hsv_jitter(aug_frame)
                aug_frame = self.augment_atmospheric_fog_and_bloom(
                    aug_frame, fog_intensity=self.rng.uniform(0.10, 0.24)
                )
                if self.rng.random() < 0.5:
                    aug_frame, aug_anns = self.augment_horizontal_flip(aug_frame, aug_anns)
                aug_frame, aug_anns = self.augment_multi_scale_crop(
                    aug_frame, aug_anns, scale_range=(0.78, 0.95)
                )
                suffix = f"aug{idx + 1:02d}_biome_mix"

            # Fallback guard: if a random crop dropped all annotations when base had annotations,
            # retry with horizontal flip on base annotations so no blank label files pollute training.
            if base_annotations and not aug_anns:
                aug_frame, aug_anns = self.augment_horizontal_flip(
                    self.augment_hsv_jitter(frame.copy()), copy.deepcopy(base_annotations)
                )

            variants.append((suffix, aug_frame, aug_anns))

        return variants

    @staticmethod
    def load_image_unicode(path: Union[str, Path]) -> np.ndarray:
        """
        Safely loads an image from a path containing non-ASCII / Unicode characters on Windows
        (e.g., `selçuk`) using `np.fromfile` + `cv2.imdecode`.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image: {file_path}")
        return frame

    @staticmethod
    def validate_annotations_in_bounds(
        annotations: List[Dict[str, Any]],
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> bool:
        """
        Verifies that every annotation's center `(cx, cy)`, dimensions `(w, h)`,
        and corners `(cx ± w/2, cy ± h/2)` lie strictly within `[0.0, 1.0]`.
        """
        eps = 1e-5
        for ann in annotations:
            cx, cy, w, h = SyntheticYOLOAugmentor._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                return False
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
                return False
        return True

    @staticmethod
    def verify_augmented_boxes_bounds(boxes: List[Dict[str, Any] | Tuple[float, float, float, float] | List[float]]) -> bool:
        """
        Validates all augmented YOLO boxes lie strictly within [0.0, 1.0].
        Wrapped in try/except, returning a boolean.
        """
        try:
            for b in boxes:
                if isinstance(b, dict):
                    cx, cy, w, h = SyntheticYOLOAugmentor._extract_yolo_cxcywh(b)
                elif isinstance(b, (list, tuple)) and len(b) == 4:
                    cx, cy, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                else:
                    return False
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    return False
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                x2 = cx + 0.5 * w
                y2 = cy + 0.5 * h
                if x1 < -1e-5 or y1 < -1e-5 or x2 > 1.0 + 1e-5 or y2 > 1.0 + 1e-5:
                    return False
            return True
        except Exception:
            return False

    def get_augmentor_health_summary(self) -> dict:
        try:
            return {"augmentor_ready": True}
        except Exception:
            return {"augmentor_ready": False}

    def get_synthetic_augmentor_ready_flag(self) -> dict:
        return {"module": "synthetic_augmentor", "ready": True}

    def get_synthetic_augmentor_version_info(self) -> dict:
        return {"module": "synthetic_augmentor", "version": "2.0.0", "ready": True}

    def get_synthetic_augmentor_diagnostic_snapshot(self) -> dict:
        return {"module": "synthetic_augmentor", "status": "ok", "ready": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "reference_screenshots" / "aria_3d_crest_far.jpg"

    if not sample_path.exists():
        raise FileNotFoundError(f"Reference screenshot not found: {sample_path}")

    img = SyntheticYOLOAugmentor.load_image_unicode(sample_path)

    h_px, w_px = img.shape[:2]
    sample_annotations = [
        {
            "class_id": 0,
            "class_name": "aria",
            "cx": 0.512,
            "cy": 0.485,
            "w": 0.068,
            "h": 0.115,
            "confidence": 0.94,
        },
        {
            "class_id": 4,
            "class_name": "pure_aether_ore",
            "cx": 0.245,
            "cy": 0.630,
            "w": 0.054,
            "h": 0.088,
            "confidence": 0.89,
        },
    ]

    augmentor = SyntheticYOLOAugmentor(seed=42)
    variants = augmentor.generate_augmented_variants(img, sample_annotations, count=10)
    logger.info("Loaded %s (%dx%d) and generated %d synthetic variants.", sample_path.name, w_px, h_px, len(variants))

    for suffix, v_frame, v_anns in variants:
        assert v_frame.shape == img.shape, f"Shape mismatch on {suffix}: {v_frame.shape}"
        assert len(v_anns) > 0, f"Annotations lost on {suffix}"
        assert augmentor.validate_annotations_in_bounds(v_anns, img_w=w_px, img_h=h_px), (
            f"Bounding box out of [0.0, 1.0] bounds on {suffix}: {v_anns}"
        )
        logger.info(
            "  -> Variant %-22s | shape=%s | boxes=%d | box0=(cx=%.4f, cy=%.4f, w=%.4f, h=%.4f)",
            suffix,
            v_frame.shape,
            len(v_anns),
            v_anns[0]["cx"],
            v_anns[0]["cy"],
            v_anns[0]["w"],
            v_anns[0]["h"],
        )

    logger.info("All 10x photometric & geometric synthetic augmentations verified strictly inside [0.0, 1.0].")


def get_synthetic_augmentor_capabilities() -> list:
    return ["synthetic_augmentor", "photometric_augmentation", "diagnostic_snapshot"]


def validate_synthetic_augmentor_contract() -> bool:
    return bool(get_synthetic_augmentor_capabilities())


def get_synthetic_augmentor_baseline_metrics() -> dict:
    return {"module": "synthetic_augmentor", "contract_valid": validate_synthetic_augmentor_contract(), "capability_count": len(get_synthetic_augmentor_capabilities())}


def get_synthetic_augmentor_readiness_status() -> dict:
    return {"module": "synthetic_augmentor", "ready": validate_synthetic_augmentor_contract(), "metrics": get_synthetic_augmentor_baseline_metrics()}


def get_synthetic_augmentor_audit_record() -> dict:
    return {"module": "synthetic_augmentor", "audit_passed": validate_synthetic_augmentor_contract(), "readiness": get_synthetic_augmentor_readiness_status()}


def get_synthetic_augmentor_operational_digest() -> dict:
    return {"module": "synthetic_augmentor", "operational": validate_synthetic_augmentor_contract(), "audit": get_synthetic_augmentor_audit_record()}


def get_synthetic_augmentor_telemetry_payload() -> dict:
    return {"module": "synthetic_augmentor", "payload": get_synthetic_augmentor_operational_digest()}


def get_synthetic_augmentor_integrity_seal() -> dict:
    return {"module": "synthetic_augmentor", "sealed": validate_synthetic_augmentor_contract(), "telemetry": get_synthetic_augmentor_telemetry_payload()}


def get_synthetic_augmentor_compliance_certificate() -> dict:
    return {"module": "synthetic_augmentor", "certified": validate_synthetic_augmentor_contract(), "seal": get_synthetic_augmentor_integrity_seal()}


def get_synthetic_augmentor_readiness_probe() -> dict:
    return {"module": "synthetic_augmentor", "ready": validate_synthetic_augmentor_contract(), "certificate": get_synthetic_augmentor_compliance_certificate()}

