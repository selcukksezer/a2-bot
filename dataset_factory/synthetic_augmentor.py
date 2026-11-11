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
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from version import __version__

from dataset_factory.class_taxonomy import resolve_canonical_max_class_id

logger = logging.getLogger(__name__)


# F7: a synthetic variant is derived evidence, never a fresh detection — its
# confidence is capped at this value (and the original kept under
# `original_confidence`). The cap is a discount, not a measurement; it sits
# below the 0.85 "trusted" boundary so no synthetic copy can pass a trust
# gate on its original's confidence.
SYNTHETIC_CONFIDENCE_CAP: float = 0.80


def _clamp(value: float, low: float, high: float) -> float:
    """Scalar clamp without numpy dispatch overhead.

    ``np.clip`` on a Python float goes through full ufunc machinery
    (D41 bench, best-of-3, 200 calls: ~9.9 ms vs ~0.10 ms for this
    helper -- ~100x). Plain comparisons are identical for every
    ordered value, including NaN (unordered: falls through and is
    returned unchanged, exactly like ``np.clip``).
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


def _validate_normalized_box(
    cx: float, cy: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    """Validate a normalized (cx, cy, w, h) box; raise ValueError if broken.

    F6: same reject policy as ``class_taxonomy.format_yolo_txt_line`` —
    out-of-range / NaN / infinite / degenerate coordinates are REJECTED,
    never silently clamped into a plausible-looking box. A box whose centre
    sits outside the frame or whose size is zero carries no information;
    clamping it would move it onto a different object and the 10x multiplier
    would then stamp that wrong box onto ten training variants. Pixel inputs
    must already be normalized by the caller (per-coordinate unit detection
    in ``_extract_yolo_cxcywh``); this helper only checks the normalized
    result, with no tolerance beyond exact [0, 1] bounds.
    """
    if not (
        math.isfinite(cx)
        and math.isfinite(cy)
        and math.isfinite(w)
        and math.isfinite(h)
    ):
        raise ValueError(
            f"Non-finite YOLO box coordinates: cx={cx!r}, cy={cy!r}, w={w!r}, h={h!r}"
        )
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(f"YOLO box centre out of [0, 1]: cx={cx!r}, cy={cy!r}")
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0) or w <= 1e-6 or h <= 1e-6:
        raise ValueError(f"YOLO box size out of (0, 1] or degenerate: w={w!r}, h={h!r}")
    return float(cx), float(cy), float(w), float(h)


class SyntheticYOLOAugmentor:
    """
    10x Photometric & Geometric Synthetic YOLO Dataset Augmentor for Aion 2.

    Supports both dict-based annotation formats used across the Dataset Factory:
      - Normalized center-width-height keys: `cx`, `cy`, `w`, `h` (or `width`, `height`)
      - Normalized bbox tuple/list under `bbox_norm` or `yolo_bbox`: `(cx, cy, w, h)`
      - Optional pixel bbox under `bbox_xyxy` or `bbox`: `(x1, y1, x2, y2)`
      - Preserves `class_id`, `class_name`, and all extra metadata keys.
        `confidence` is NOT preserved as-is: a synthetic copy is discounted
        to at most ``SYNTHETIC_CONFIDENCE_CAP`` (F7) and the original is kept
        under `original_confidence`.
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
        self.min_box_visibility = float(max(0.05, min(0.95, min_box_visibility)))
        self.min_box_dim = float(max(0.001, min(0.1, min_box_dim)))

    # -------------------------------------------------------------------------
    # Annotation Normalization & Extraction Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_yolo_cxcywh(ann: Dict[str, Any], img_w: int = 1920, img_h: int = 1080) -> Tuple[float, float, float, float]:
        """
        Extract normalized (cx, cy, w, h) in [0.0, 1.0] from an annotation dictionary.

        P1-7 (K-20): `img_w`/`img_h` varsayılanı 1920x1080 son-çare
        yedektir (çağıran gerçek boyutu geçirmelidir — üretim yolları
        geçirir); boyutsuz çağrıda piksel kutu bu boyuta göre normalize
        edilir, uydurma boyuta göre değil.

        F6: broken coordinates raise ValueError instead of being silently
        clamped into a plausible-looking box (same reject policy as
        ``class_taxonomy.format_yolo_txt_line``). Callers that process a
        batch (``generate_augmented_variants``, flip, crop) catch the error
        per annotation and skip that one box — a broken box never aborts
        the batch and never enters training multiplied 10x. Per-coordinate
        pixel normalization is kept (a pixel cx such as 960 is a legitimate
        input and is divided by the image size); a value that still lands
        outside [0, 1] after normalization, is negative, NaN/infinite, or
        degenerate (w/h <= 0) is rejected here.
        """
        if "cx" in ann and "cy" in ann and ("w" in ann or "width" in ann) and ("h" in ann or "height" in ann):
            cx = float(ann["cx"])
            cy = float(ann["cy"])
            w = float(ann.get("w", ann.get("width", 0.0)))
            h = float(ann.get("h", ann.get("height", 0.0)))
            # Per-coordinate unit detection: a pixel cx (e.g. 960) must not
            # drag a normalized cy/w/h (e.g. 0.5) down with it. The old
            # all-or-nothing check divided EVERY coordinate by the image
            # size as soon as ANY one exceeded 1.0, so a mixed-unit dict
            # {"cx": 0.5, "w": 100} collapsed cx to 0.5/1920 ~= 0.00026
            # while still passing the [0, 1] bounds check -- a garbage box
            # that silently entered training. Each coordinate is now
            # normalized only if it is itself out of [0, 1] range.
            if cx > 1.0:
                cx /= max(1, img_w)
            if cy > 1.0:
                cy /= max(1, img_h)
            if w > 1.0:
                w /= max(1, img_w)
            if h > 1.0:
                h /= max(1, img_h)
            return _validate_normalized_box(cx, cy, w, h)

        for key in ("yolo_bbox", "bbox_norm", "normalized_bbox"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                cx, cy, w, h = [float(v) for v in ann[key]]
                # F6: no unit detection on this branch — a bbox_norm tuple
                # claims normalized coordinates, so an out-of-range value
                # (e.g. cx=5.0) is corrupt input, not pixels; reject it
                # instead of clamping it onto the frame edge.
                return _validate_normalized_box(cx, cy, w, h)

        for key in ("bbox_xyxy", "xyxy"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                v0, v1, v2, v3 = [float(v) for v in ann[key]]
                # P1-7 (K-20): koordinat bazli birim tespiti — toplu
                # max() kurali karisik birimli xyxy'de norm koordinati da
                # bolup kutuyu kokeneye cokeriyordu. Her koordinat yalnizca
                # kendisi >1.0 ise kendi eksenine bolunur.
                x1 = v0 / max(1, img_w) if v0 > 1.0 else v0
                y1 = v1 / max(1, img_h) if v1 > 1.0 else v1
                x2 = v2 / max(1, img_w) if v2 > 1.0 else v2
                y2 = v3 / max(1, img_h) if v3 > 1.0 else v3
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                return _validate_normalized_box(cx, cy, w, h)

        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
            # Repo convention (same as DatasetCurator._extract_dict_cxcywh):
            # normalized "bbox" is (cx, cy, w, h); pixel "bbox" is
            # (x, y, w, h) -- NOT xyxy (D37/F-0901: the old code read a
            # normalized bbox as cxcywh but a pixel bbox as xyxy).
            b0, b1, b2, b3 = [float(v) for v in ann["bbox"]]
            # P1-7 (K-20): koordinat bazli birim tespiti (toplu max()
            # kurali karisik birimli bbox'ta norm koordinati da boluyordu).
            nb0 = b0 / max(1, img_w) if b0 > 1.0 else b0
            nb1 = b1 / max(1, img_h) if b1 > 1.0 else b1
            nb2 = b2 / max(1, img_w) if b2 > 1.0 else b2
            nb3 = b3 / max(1, img_h) if b3 > 1.0 else b3
            if max(b0, b1, b2, b3) <= 1.0:
                return _validate_normalized_box(b0, b1, b2, b3)
            cx = nb0 + 0.5 * nb2
            cy = nb1 + 0.5 * nb3
            w = nb2
            h = nb3
            return _validate_normalized_box(cx, cy, w, h)

        raise ValueError(f"Annotation dictionary missing recognizable YOLO coordinates: {ann}")

    @staticmethod
    def _clamp_yolo_box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """
        Strictly clamp normalized (cx, cy, w, h) so the entire bounding box
        [cx - w/2, cy - h/2, cx + w/2, cy + h/2] lies inside [0.0, 1.0].

        F6 note: this is GEOMETRIC clipping of transform outputs, not input
        repair — a crop/flip may legitimately push a valid box partly past
        the frame edge, and clipping it there is correct geometry (pinned by
        test_clamp_yolo_box_values). Broken INPUT coordinates are rejected
        earlier, in ``_extract_yolo_cxcywh`` via ``_validate_normalized_box``,
        so a corrupt box never reaches this helper to be silently "fixed".
        """
        x1 = _clamp(cx - 0.5 * w, 0.0, 1.0)
        y1 = _clamp(cy - 0.5 * h, 0.0, 1.0)
        x2 = _clamp(cx + 0.5 * w, 0.0, 1.0)
        y2 = _clamp(cy + 0.5 * h, 0.0, 1.0)

        w_clamped = _clamp(x2 - x1, 0.0, 1.0)
        h_clamped = _clamp(y2 - y1, 0.0, 1.0)
        cx_clamped = _clamp(x1 + 0.5 * w_clamped, 0.0, 1.0)
        cy_clamped = _clamp(y1 + 0.5 * h_clamped, 0.0, 1.0)
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

        D37/F-0899: the copy is marked as synthetic -- `synthetic: True`,
        `source: "synthetic_augmentor"` (the original source is preserved
        under `original_source`) -- so a synthetic variant can never pass
        as an original detection with its original source/confidence.
        F7: the confidence is DISCOUNTED to at most
        SYNTHETIC_CONFIDENCE_CAP and the original is kept under
        `original_confidence`, so the variant cannot pass a trust gate on
        its original's confidence.
        """
        cx, cy, w, h = cls._clamp_yolo_box(cx, cy, w, h)
        updated = copy.deepcopy(ann)
        updated["cx"] = round(cx, 6)
        updated["cy"] = round(cy, 6)
        updated["w"] = round(w, 6)
        updated["h"] = round(h, 6)
        updated["synthetic"] = True
        if updated.get("source") not in (None, "", "synthetic_augmentor"):
            updated["original_source"] = updated.get("source")
        updated["source"] = "synthetic_augmentor"
        # F7: discount the confidence (see module constant). `min()` keeps
        # this idempotent when a variant is re-augmented, and
        # `original_confidence` is written only once so the first (real)
        # value is never overwritten by an already-discounted one.
        for _conf_key in ("confidence", "conf"):
            if _conf_key in updated:
                try:
                    _orig_conf = float(updated[_conf_key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(_orig_conf):
                    if "original_confidence" not in updated:
                        updated["original_confidence"] = round(_orig_conf, 4)
                    updated[_conf_key] = round(
                        min(_orig_conf, SYNTHETIC_CONFIDENCE_CAP), 4
                    )

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

        px_x1 = int(round(_clamp((cx - 0.5 * w) * img_w, 0, img_w)))
        px_y1 = int(round(_clamp((cy - 0.5 * h) * img_h, 0, img_h)))
        px_x2 = int(round(_clamp((cx + 0.5 * w) * img_w, 0, img_w)))
        px_y2 = int(round(_clamp((cy + 0.5 * h) * img_h, 0, img_h)))

        if "bbox_xyxy" in updated:
            updated["bbox_xyxy"] = [px_x1, px_y1, px_x2, px_y2]
        if "xyxy" in updated:
            updated["xyxy"] = [px_x1, px_y1, px_x2, px_y2]
        if "bbox" in updated:
            old_bbox = updated["bbox"]
            if isinstance(old_bbox, (list, tuple)) and len(old_bbox) == 4 and max(float(v) for v in old_bbox) <= 1.0:
                updated["bbox"] = [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)]
            else:
                # Pixel "bbox" follows the repo (x, y, w, h) convention.
                updated["bbox"] = [px_x1, px_y1, px_x2 - px_x1, px_y2 - px_y1]

        return updated

    @staticmethod
    def _validate_annotation_class_id(ann: Dict[str, Any]) -> None:
        """Reject annotations whose class_id is not a canonical id (F8).

        The ceiling comes from the same single source as the hygiene gate
        (``dataset_factory.class_taxonomy.resolve_canonical_max_class_id``:
        ``dataset/data.yaml`` ``nc - 1``, fallback 26) — the old hardcoded
        ``0 <= cid <= 26`` literal is gone, so gate and augmentor can no
        longer disagree about the range (P0-3, K-01/K-10). Annotations
        without a class_id are geometry-only and pass through untouched.
        Legacy 7/8 pass this range check — they are valid canonical ids;
        telling a legacy id apart from a canonical one needs scheme info
        the augmentor does not have (see F2).
        """
        if "class_id" not in ann:
            return
        raw = ann.get("class_id")
        if isinstance(raw, bool) or raw is None:
            raise ValueError(f"Invalid class_id in augmentor input: {raw!r}")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"Invalid class_id in augmentor input: {raw!r}")
            cid = int(raw)
        else:
            try:
                cid = int(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise ValueError(
                    f"Unparseable class_id in augmentor input: {raw!r}"
                ) from None
        max_class_id = resolve_canonical_max_class_id()
        if not 0 <= cid <= max_class_id:
            raise ValueError(
                f"class_id {cid} outside canonical 0..{max_class_id} "
                f"(dataset/data.yaml nc={max_class_id + 1})"
            )

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
        gamma = _clamp(gamma, 0.50, 2.00)

        # Build LUT for fast gamma transformation (kept exactly as before:
        # the list-comp values are the reference gamma table).
        inv_gamma = 1.0 / max(1e-4, gamma)
        lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)

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

        # Fold the per-channel gains into the LUT instead of promoting the
        # whole frame to float32 and broadcasting a (1,1,3) multiply over
        # it: the old path did LUT (uint8 truncation) -> float32 -> *gain
        # -> clip -> uint8 (truncation). A folded LUT precomputes exactly
        # that same truncate(clip(lut_base * gain)) per channel, so the
        # output is bit-identical while the 1920x1080 float32 multiply
        # disappears (paired measurement, same frame/run: 0.162s -> 0.033s
        # per call; D41 bench, best-of-3: ~110 ms -> ~24 ms @1080p).
        # 3-channel frames take the single-LUT path; non-3-channel frames
        # keep the original float32 path.
        if frame.ndim == 3 and frame.shape[2] == 3:
            gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32).reshape((1, 3))
            lut_bgr = (
                np.clip(lut.astype(np.float32)[:, None] * gains, 0.0, 255.0)
                .astype(np.uint8)
                .reshape(256, 1, 3)
            )
            return cv2.LUT(frame, lut_bgr)

        adjusted = cv2.LUT(frame, lut).astype(np.float32)
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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_delta = self.rng.randint(-abs(int(hue_shift_Limit)), abs(int(hue_shift_Limit)))
        s_scale = self.rng.uniform(float(sat_range[0]), float(sat_range[1]))
        v_scale = self.rng.uniform(float(val_range[0]), float(val_range[1]))

        # OpenCV hue range is [0, 179]. The three per-channel maps are
        # baked into 256-entry LUTs instead of promoting the whole frame
        # to int32: the arithmetic is identical (float64 inputs 0..255,
        # same clip, same truncation), so the output is bit-identical
        # (D41 bench, best-of-3: ~157 ms -> ~40 ms @1080p).
        idx = np.arange(256, dtype=np.float64)
        hue_lut = ((idx + h_delta) % 180).astype(np.uint8)
        sat_lut = np.clip(idx * s_scale, 0, 255).astype(np.uint8)
        val_lut = np.clip(idx * v_scale, 0, 255).astype(np.uint8)
        lut = np.stack([hue_lut, sat_lut, val_lut], axis=1).reshape(256, 1, 3)
        return cv2.cvtColor(cv2.LUT(hsv, lut), cv2.COLOR_HSV2BGR)

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

        fog_intensity = _clamp(fog_intensity, 0.0, 0.65)
        bloom_strength = _clamp(bloom_strength, 0.0, 0.50)

        # 1. Atmospheric gradient fog veil (Pilgrim's Ridge cool silver-blue or Altamia misty cyan-white)
        fog_colors = [
            np.array([228.0, 222.0, 212.0], dtype=np.float32),  # Pilgrim's Ridge cool mist (BGR)
            np.array([235.0, 232.0, 225.0], dtype=np.float32),  # Altamia Highland alpine fog (BGR)
            np.array([218.0, 225.0, 232.0], dtype=np.float32),  # Morning valley haze (BGR)
        ]
        fog_color = fog_colors[self.rng.randrange(len(fog_colors))].reshape(3)

        # Vertical depth profile: horizon/top gets full fog_intensity, bottom foreground gets 35%
        alpha_col = (np.linspace(1.0, 0.35, h, dtype=np.float32) * fog_intensity).reshape(h, 1)
        alpha_hw = np.tile(alpha_col, (1, w))
        one_minus = 1.0 - alpha_hw

        # Blend per channel with cv2 instead of broadcasting a (h,1,1)
        # alpha over a float32 copy of the whole frame: the same IEEE
        # float32 multiply/add per pixel, so the result is bit-identical,
        # without the broadcast temporaries (D41 bench, best-of-3, fog blend:
        # ~1524 ms -> ~939 ms @1080p).
        planes = cv2.split(frame.astype(np.float32))
        fogged_planes = []
        for c in range(3):
            blended = cv2.multiply(planes[c], one_minus)
            blended = cv2.add(blended, float(fog_color[c]) * alpha_hw)
            fogged_planes.append(blended)
        fogged = cv2.merge(fogged_planes)

        # 2. UE5 Soft Bloom on bright highlights (aether crests, glowing ores, sky light)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        bright_regions = cv2.bitwise_and(frame, frame, mask=bright_mask)
        # Blur at half resolution (sigma halved to keep the same relative
        # radius) and upscale: ~5x faster on the blur, bloom layer
        # differs by <=2 LSB (mean <0.2), i.e. <=1 LSB in the final frame
        # after the bloom_strength (<=0.5) multiply.
        half_w, half_h = max(1, w // 2), max(1, h // 2)
        bloom_small = cv2.resize(bright_regions, (half_w, half_h), interpolation=cv2.INTER_LINEAR)
        bloom_small = cv2.GaussianBlur(bloom_small, (0, 0), sigmaX=7.5, sigmaY=7.5)
        bloom_layer = cv2.resize(bloom_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

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
            # D37/F-1010: one broken annotation skips itself, it never
            # aborts the whole flip.
            try:
                cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping broken annotation in horizontal flip: %s", exc)
                continue
            cx_new = _clamp(1.0 - cx, 0.0, 1.0)
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
        min_s = _clamp(scale_range[0], 0.50, 0.99)
        max_s = _clamp(scale_range[1], min_s, 0.995)
        scale_x = self.rng.uniform(min_s, max_s)
        scale_y = self.rng.uniform(max(min_s, scale_x - 0.04), min(max_s, scale_x + 0.04))

        # Normalized crop window position [crop_x1, crop_y1, crop_x2, crop_y2]
        max_x1 = max(0.0, 1.0 - scale_x)
        max_y1 = max(0.0, 1.0 - scale_y)

        # If annotations exist, try to pick a crop window that keeps the primary target well inside
        if annotations:
            # F11: the anchor extraction used to be unprotected — one broken
            # anchor aborted the whole crop (and, in generate(), the whole
            # variant). A broken anchor now falls back to a random window;
            # the per-annotation loop below still skips broken boxes one by
            # one. (In generate() the annotations were already normalized,
            # so this only bites direct callers — but it must not bite them.)
            anchor_ok = False
            try:
                anchor_ann = self.rng.choice(annotations)
                a_cx, a_cy, a_w, a_h = self._extract_yolo_cxcywh(anchor_ann, img_w=img_w, img_h=img_h)
                anchor_ok = True
            except (TypeError, ValueError) as exc:
                logger.debug("Broken anchor annotation in multi-scale crop, using random window: %s", exc)
            if anchor_ok:
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
            # D37/F-1010: one broken annotation skips itself, it never
            # aborts the whole crop.
            try:
                cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping broken annotation in multi-scale crop: %s", exc)
                continue
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
            new_x1 = _clamp((inter_x1 - actual_crop_x1) / actual_crop_w, 0.0, 1.0)
            new_y1 = _clamp((inter_y1 - actual_crop_y1) / actual_crop_h, 0.0, 1.0)
            new_x2 = _clamp((inter_x2 - actual_crop_x1) / actual_crop_w, 0.0, 1.0)
            new_y2 = _clamp((inter_y2 - actual_crop_y1) / actual_crop_h, 0.0, 1.0)

            new_w = _clamp(new_x2 - new_x1, 0.0, 1.0)
            new_h = _clamp(new_y2 - new_y1, 0.0, 1.0)
            if new_w < self.min_box_dim or new_h < self.min_box_dim:
                continue

            new_cx = _clamp(new_x1 + 0.5 * new_w, 0.0, 1.0)
            new_cy = _clamp(new_y1 + 0.5 * new_h, 0.0, 1.0)

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
        # D37/F-1012: empty input produces no variants at all -- the old
        # code returned `count` variants with 0 boxes and silent empty
        # label files.
        if not annotations:
            logger.warning("generate_augmented_variants called with no annotations; returning [].")
            return []

        img_h, img_w = frame.shape[:2]
        # Ensure base annotations are normalized and clamped. A single
        # broken annotation must not abort the whole 10x batch (D37/F-1010).
        base_annotations: List[Dict[str, Any]] = []
        for ann in annotations:
            try:
                cx, cy, bw, bh = self._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
                # F8: a wrong class must not be multiplied 10x.
                self._validate_annotation_class_id(ann)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping broken annotation in augmentor input: %s", exc)
                continue
            base_annotations.append(
                self._update_annotation_coords(ann, cx, cy, bw, bh, img_w=img_w, img_h=img_h)
            )
        if not base_annotations:
            raise ValueError("All input annotations are invalid; no variants generated.")

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
            # D37/F-1011: the fallback regenerates a DIFFERENT image (flip, no
            # crop), so it must not keep the crop recipe's suffix.
            if base_annotations and not aug_anns:
                aug_frame, aug_anns = self.augment_horizontal_flip(
                    self.augment_hsv_jitter(frame.copy()), copy.deepcopy(base_annotations)
                )
                suffix = f"aug{idx + 1:02d}_hflip_fallback"

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

        F10: returns False (never raises) on broken input — the sibling
        ``verify_augmented_boxes_bounds`` already behaves this way, and
        callers expect a boolean. ``_extract_yolo_cxcywh`` raises
        ValueError on out-of-range/NaN/degenerate boxes (F6); that error is
        caught here and reported as False.
        """
        eps = 1e-5
        for ann in annotations:
            try:
                cx, cy, w, h = SyntheticYOLOAugmentor._extract_yolo_cxcywh(ann, img_w=img_w, img_h=img_h)
            except (TypeError, ValueError):
                return False
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
    def verify_augmented_boxes_bounds(
        boxes: List[Dict[str, Any] | Tuple[float, float, float, float] | List[float]],
        img_w: int = 1920,
        img_h: int = 1080,
    ) -> bool:
        """
        Validates all augmented YOLO boxes lie strictly within [0.0, 1.0].
        Wrapped in try/except, returning a boolean.

        P1-7 (K-20): piksel birimli dict kutular gercek goruntu boyutuyla
        normalize edilir; `img_w`/`img_h` verilmezse 1920x1080 son-care
        varsayilan olarak kalir (cagiran gercek boyutu gecirmelidir).
        """
        try:
            for b in boxes:
                if isinstance(b, dict):
                    cx, cy, w, h = SyntheticYOLOAugmentor._extract_yolo_cxcywh(b, img_w=img_w, img_h=img_h)
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

    def _probe_augmentation(self) -> Dict[str, Any]:
        """
        Run a real mini-augmentation on a synthetic frame and verify the
        output (D37/F-0900): one variant, same shape, boxes in bounds and
        marked synthetic. Returns {"ok": bool, "detail": str}.
        """
        try:
            probe_frame = np.zeros((64, 64, 3), dtype=np.uint8)
            probe_frame[:, :] = (40, 120, 200)
            probe_anns = [
                {
                    "class_id": 0,
                    "class_name": "herb_aria",
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.2,
                    "h": 0.2,
                    "confidence": 0.9,
                }
            ]
            variants = self.generate_augmented_variants(probe_frame, probe_anns, count=1)
            if len(variants) != 1:
                return {"ok": False, "detail": "no variant produced"}
            _suffix, out_frame, out_anns = variants[0]
            if out_frame.shape != probe_frame.shape:
                return {"ok": False, "detail": "variant shape mismatch"}
            if not out_anns or not self.validate_annotations_in_bounds(
                out_anns, img_w=64, img_h=64
            ):
                return {"ok": False, "detail": "variant annotations invalid/out of bounds"}
            if not out_anns[0].get("synthetic"):
                return {"ok": False, "detail": "variant not marked synthetic"}
            return {"ok": True, "detail": "1 variant, shape and boxes verified"}
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    def get_augmentor_health_summary(self) -> dict:
        # D37/F-0900: health is the real probe, not a constant True.
        probe = self._probe_augmentation()
        return {"augmentor_ready": bool(probe["ok"]), "probe": probe}

    def get_synthetic_augmentor_ready_flag(self) -> dict:
        # D37/F-0900: readiness is the real probe, not a constant True.
        probe = self._probe_augmentation()
        return {"module": "synthetic_augmentor", "ready": bool(probe["ok"]), "probe": probe}

    def get_synthetic_augmentor_version_info(self) -> dict:
        # D37/F-0900 pattern: readiness is the real probe, not a constant
        # True -- the old code returned "ready": True unconditionally.
        probe = self._probe_augmentation()
        return {
            "module": "synthetic_augmentor",
            "version": __version__,
            "ready": bool(probe["ok"]),
            "probe": probe,
        }

    def get_synthetic_augmentor_diagnostic_snapshot(self) -> dict:
        # D37/F-0900: the snapshot reports the real probe.
        probe = self._probe_augmentation()
        return {
            "module": "synthetic_augmentor",
            "status": "ok" if probe["ok"] else "degraded",
            "ready": bool(probe["ok"]),
            "probe": probe,
        }


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
    # D37/F-0900 pattern: the old `bool(capabilities)` was a tautology --
    # a non-empty static list is always True. The contract now holds only
    # if a real augmentation probe succeeds.
    return bool(_run_synthetic_augmentor_probe()["ok"])


def _run_synthetic_augmentor_probe() -> dict:
    return SyntheticYOLOAugmentor(seed=42)._probe_augmentation()


def get_synthetic_augmentor_readiness_status() -> dict:
    # D37/F-0900: readiness is the real probe, not a constant True.
    probe = _run_synthetic_augmentor_probe()
    return {"module": "synthetic_augmentor", "ready": bool(probe["ok"]), "probe": probe}


def get_synthetic_augmentor_readiness_probe() -> dict:
    # D37/F-0900: the probe runs a real augmentation and verifies it.
    probe = _run_synthetic_augmentor_probe()
    return {"module": "synthetic_augmentor", "ready": bool(probe["ok"]), "probe": probe}


def get_synthetic_augmentor_audit_summary() -> dict:
    probe = _run_synthetic_augmentor_probe()
    return {"module": "synthetic_augmentor", "audit_passed": bool(probe["ok"]), "probe": probe}


