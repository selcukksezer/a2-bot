"""Six Material Discriminator module for dataset factory.

Distinguishes the 6 core harvestable materials:
- herb_aria
- herb_mela
- herb_targena
- ore_orichalcum
- gem_crystal
- wood_tree

from each other and from environmental background clutter (grass_noise, rock_noise)
using HSV color histograms, Canny edge density, specular brightness, and aspect ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


MATERIAL_CLASSES: Tuple[str, ...] = (
    "herb_aria",
    "herb_mela",
    "herb_targena",
    "ore_orichalcum",
    "gem_crystal",
    "wood_tree",
)

NOISE_CLASSES: Tuple[str, ...] = (
    "grass_noise",
    "rock_noise",
)

ALL_CLASSES: Tuple[str, ...] = MATERIAL_CLASSES + NOISE_CLASSES


@dataclass(frozen=True)
class MaterialFeatureVector:
    """Extracted visual features for material discrimination."""

    hsv_histogram: np.ndarray
    dominant_hue: float
    mean_saturation: float
    mean_value: float
    canny_edge_density: float
    specular_brightness_ratio: float
    aspect_ratio: float


@dataclass(frozen=True)
class MaterialProfile:
    """Reference feature profile for a target material or noise category."""

    name: str
    is_noise: bool
    hue_range: Tuple[float, float]
    saturation_range: Tuple[float, float]
    value_range: Tuple[float, float]
    edge_density_range: Tuple[float, float]
    specular_range: Tuple[float, float]
    aspect_ratio_range: Tuple[float, float]
    reference_histogram: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class DiscriminationResult:
    """Result of evaluating a candidate crop/region with SixMaterialDiscriminator."""

    predicted_class: str
    is_valid_material: bool
    is_noise: bool
    confidence: float
    class_scores: Dict[str, float]
    features: MaterialFeatureVector


class SixMaterialDiscriminator:
    """Discriminates 6 core materials from each other and from grass/rock noise."""

    MATERIALS: Tuple[str, ...] = MATERIAL_CLASSES
    NOISE_LABELS: Tuple[str, ...] = NOISE_CLASSES

    def __init__(
        self,
        h_bins: int = 18,
        s_bins: int = 8,
        v_bins: int = 8,
        canny_low: int = 60,
        canny_high: int = 160,
        specular_v_threshold: int = 220,
        specular_s_max: int = 90,
        min_confidence: float = 0.45,
        noise_margin: float = 0.05,
    ) -> None:
        self.h_bins = int(h_bins)
        self.s_bins = int(s_bins)
        self.v_bins = int(v_bins)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.specular_v_threshold = int(specular_v_threshold)
        self.specular_s_max = int(specular_s_max)
        self.min_confidence = float(min_confidence)
        self.noise_margin = float(noise_margin)

        self.profiles: Dict[str, MaterialProfile] = self._build_default_profiles()

    def _create_synthetic_reference_hist(
        self,
        hue_range: Tuple[float, float],
        sat_range: Tuple[float, float],
        val_range: Tuple[float, float],
    ) -> np.ndarray:
        """Builds a normalized reference 1D concatenated HSV histogram for a profile."""
        h_hist = np.zeros(self.h_bins, dtype=np.float32)
        s_hist = np.zeros(self.s_bins, dtype=np.float32)
        v_hist = np.zeros(self.v_bins, dtype=np.float32)

        for i in range(self.h_bins):
            bin_center = (i + 0.5) * (180.0 / self.h_bins)
            h_min, h_max = hue_range
            if h_min <= bin_center <= h_max:
                h_hist[i] = 1.0
            else:
                dist = min(abs(bin_center - h_min), abs(bin_center - h_max))
                h_hist[i] = float(np.exp(-0.5 * (dist / 12.0) ** 2))

        for i in range(self.s_bins):
            bin_center = (i + 0.5) * (256.0 / self.s_bins)
            s_min, s_max = sat_range
            if s_min <= bin_center <= s_max:
                s_hist[i] = 1.0
            else:
                dist = min(abs(bin_center - s_min), abs(bin_center - s_max))
                s_hist[i] = float(np.exp(-0.5 * (dist / 25.0) ** 2))

        for i in range(self.v_bins):
            bin_center = (i + 0.5) * (256.0 / self.v_bins)
            v_min, v_max = val_range
            if v_min <= bin_center <= v_max:
                v_hist[i] = 1.0
            else:
                dist = min(abs(bin_center - v_min), abs(bin_center - v_max))
                v_hist[i] = float(np.exp(-0.5 * (dist / 25.0) ** 2))

        h_hist /= max(float(np.sum(h_hist)), 1e-6)
        s_hist /= max(float(np.sum(s_hist)), 1e-6)
        v_hist /= max(float(np.sum(v_hist)), 1e-6)

        combined = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(combined))
        return combined / max(norm, 1e-6)

    def _build_default_profiles(self) -> Dict[str, MaterialProfile]:
        """Defines the 6 core material profiles and 2 environmental noise profiles."""
        raw_specs = {
            # Bright/luminous cyan-white-greenish floral herb
            "herb_aria": {
                "is_noise": False,
                "hue": (70.0, 105.0),
                "sat": (60.0, 175.0),
                "val": (140.0, 245.0),
                "edge": (0.08, 0.24),
                "spec": (0.04, 0.22),
                "aspect": (0.65, 1.45),
            },
            # Golden-yellow / warm orange-yellow medicinal herb
            "herb_mela": {
                "is_noise": False,
                "hue": (16.0, 36.0),
                "sat": (110.0, 245.0),
                "val": (135.0, 245.0),
                "edge": (0.09, 0.26),
                "spec": (0.03, 0.18),
                "aspect": (0.60, 1.40),
            },
            # Deep crimson / magenta-violet arcane herb
            "herb_targena": {
                "is_noise": False,
                "hue": (135.0, 175.0),
                "sat": (95.0, 240.0),
                "val": (100.0, 225.0),
                "edge": (0.10, 0.28),
                "spec": (0.02, 0.16),
                "aspect": (0.55, 1.35),
            },
            # Metallic copper-gold / reddish-bronze ore vein with strong edges & metallic glint
            "ore_orichalcum": {
                "is_noise": False,
                "hue": (6.0, 24.0),
                "sat": (90.0, 215.0),
                "val": (110.0, 235.0),
                "edge": (0.14, 0.38),
                "spec": (0.08, 0.35),
                "aspect": (0.75, 1.65),
            },
            # High-specular azure/violet/prismatic crystal cluster with sharp facets
            "gem_crystal": {
                "is_noise": False,
                "hue": (95.0, 145.0),
                "sat": (80.0, 230.0),
                "val": (160.0, 255.0),
                "edge": (0.12, 0.34),
                "spec": (0.15, 0.55),
                "aspect": (0.45, 1.15),
            },
            # Tall vertical trunk + canopy: brown/earthy bark with low specular and tall aspect
            "wood_tree": {
                "is_noise": False,
                "hue": (10.0, 55.0),
                "sat": (45.0, 160.0),
                "val": (55.0, 165.0),
                "edge": (0.11, 0.32),
                "spec": (0.00, 0.05),
                "aspect": (0.20, 0.72),
            },
            # Environmental grass clutter: mid-green, repetitive fine blades, very low specular
            "grass_noise": {
                "is_noise": True,
                "hue": (38.0, 68.0),
                "sat": (50.0, 165.0),
                "val": (50.0, 145.0),
                "edge": (0.22, 0.55),
                "spec": (0.00, 0.02),
                "aspect": (0.80, 2.80),
            },
            # Environmental rock clutter: desaturated gray, low specular, wide/flat aspect
            "rock_noise": {
                "is_noise": True,
                "hue": (0.0, 180.0),
                "sat": (0.0, 42.0),
                "val": (45.0, 145.0),
                "edge": (0.03, 0.16),
                "spec": (0.00, 0.03),
                "aspect": (1.10, 3.00),
            },
        }

        profiles: Dict[str, MaterialProfile] = {}
        for name, spec in raw_specs.items():
            ref_hist = self._create_synthetic_reference_hist(
                spec["hue"], spec["sat"], spec["val"]
            )
            profiles[name] = MaterialProfile(
                name=name,
                is_noise=bool(spec["is_noise"]),
                hue_range=spec["hue"],
                saturation_range=spec["sat"],
                value_range=spec["val"],
                edge_density_range=spec["edge"],
                specular_range=spec["spec"],
                aspect_ratio_range=spec["aspect"],
                reference_histogram=ref_hist,
            )
        return profiles

    def extract_features(
        self,
        bgr_image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> MaterialFeatureVector:
        """Extracts HSV histogram, Canny edge density, specular brightness, and aspect ratio.

        Args:
            bgr_image: Input image in BGR uint8 format (H, W, 3).
            bbox: Optional (x, y, w, h) bounding box to crop and compute aspect ratio from.
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input bgr_image must be a non-empty numpy array.")

        if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError("Input bgr_image must have shape (H, W, 3).")

        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            img_h, img_w = bgr_image.shape[:2]
            x0 = max(0, min(x, img_w - 1))
            y0 = max(0, min(y, img_h - 1))
            x1 = max(x0 + 1, min(x + max(w, 1), img_w))
            y1 = max(y0 + 1, min(y + max(h, 1), img_h))
            crop = bgr_image[y0:y1, x0:x1]
            aspect_ratio = float(max(w, 1)) / float(max(h, 1))
        else:
            crop = bgr_image
            h, w = crop.shape[:2]
            aspect_ratio = float(max(w, 1)) / float(max(h, 1))

        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        h_hist = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256]).flatten()

        h_hist = h_hist / max(float(np.sum(h_hist)), 1e-6)
        s_hist = s_hist / max(float(np.sum(s_hist)), 1e-6)
        v_hist = v_hist / max(float(np.sum(v_hist)), 1e-6)

        combined_hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist_norm = float(np.linalg.norm(combined_hist))
        combined_hist = combined_hist / max(hist_norm, 1e-6)

        dominant_bin = int(np.argmax(h_hist))
        dominant_hue = float((dominant_bin + 0.5) * (180.0 / self.h_bins))
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        mean_value = float(np.mean(hsv[:, :, 2]))

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        canny_edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        bright_core = (v_channel >= self.specular_v_threshold) & (
            s_channel <= self.specular_s_max
        )
        intense_glint = v_channel >= min(255, self.specular_v_threshold + 20)
        specular_mask = bright_core | intense_glint
        specular_brightness_ratio = float(np.count_nonzero(specular_mask)) / float(
            v_channel.size
        )

        return MaterialFeatureVector(
            hsv_histogram=combined_hist,
            dominant_hue=dominant_hue,
            mean_saturation=mean_saturation,
            mean_value=mean_value,
            canny_edge_density=canny_edge_density,
            specular_brightness_ratio=specular_brightness_ratio,
            aspect_ratio=aspect_ratio,
        )

    @staticmethod
    def _range_score(value: float, bounds: Tuple[float, float], sigma: float) -> float:
        """Computes a smooth [0, 1] score for how well a scalar falls within bounds."""
        low, high = bounds
        if low <= value <= high:
            return 1.0
        dist = min(abs(value - low), abs(value - high))
        return float(np.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2))

    def _score_profile(
        self, features: MaterialFeatureVector, profile: MaterialProfile
    ) -> float:
        """Computes a weighted similarity score in [0, 1] against a MaterialProfile."""
        hist_sim = float(
            np.clip(np.dot(features.hsv_histogram, profile.reference_histogram), 0.0, 1.0)
        )

        hue_score = self._range_score(features.dominant_hue, profile.hue_range, sigma=14.0)
        sat_score = self._range_score(
            features.mean_saturation, profile.saturation_range, sigma=30.0
        )
        val_score = self._range_score(features.mean_value, profile.value_range, sigma=30.0)
        color_score = 0.45 * hist_sim + 0.25 * hue_score + 0.15 * sat_score + 0.15 * val_score

        edge_score = self._range_score(
            features.canny_edge_density, profile.edge_density_range, sigma=0.07
        )
        spec_score = self._range_score(
            features.specular_brightness_ratio, profile.specular_range, sigma=0.06
        )
        aspect_score = self._range_score(
            features.aspect_ratio, profile.aspect_ratio_range, sigma=0.35
        )

        total_score = (
            0.40 * color_score
            + 0.20 * edge_score
            + 0.25 * spec_score
            + 0.15 * aspect_score
        )
        return float(np.clip(total_score, 0.0, 1.0))

    def discriminate(
        self,
        bgr_image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> DiscriminationResult:
        """Classifies an image or region into one of the 6 materials or background noise."""
        features = self.extract_features(bgr_image, bbox=bbox)
        scores: Dict[str, float] = {}
        for name, profile in self.profiles.items():
            scores[name] = self._score_profile(features, profile)

        best_material = max(MATERIAL_CLASSES, key=lambda k: scores[k])
        best_material_score = scores[best_material]

        best_noise = max(NOISE_CLASSES, key=lambda k: scores[k])
        best_noise_score = scores[best_noise]

        if (
            best_noise_score + self.noise_margin >= best_material_score
            or best_material_score < self.min_confidence
        ):
            predicted_class = best_noise
            is_noise = True
            is_valid_material = False
            confidence = float(best_noise_score)
        else:
            predicted_class = best_material
            is_noise = False
            is_valid_material = True
            confidence = float(best_material_score)

        return DiscriminationResult(
            predicted_class=predicted_class,
            is_valid_material=is_valid_material,
            is_noise=is_noise,
            confidence=confidence,
            class_scores=scores,
            features=features,
        )

    def filter_detections(
        self,
        bgr_image: np.ndarray,
        detections: List[Dict[str, Union[int, float, str, Tuple[int, int, int, int]]]],
    ) -> List[Dict[str, Union[int, float, str, Tuple[int, int, int, int]]]]:
        """Filters a list of candidate bounding boxes, keeping only valid 6-material detections.

        Each detection dict is expected to contain a 'bbox' key with (x, y, w, h).
        """
        kept: List[Dict[str, Union[int, float, str, Tuple[int, int, int, int]]]] = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                continue
            bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            res = self.discriminate(bgr_image, bbox=bbox_tuple)
            if res.is_valid_material:
                enriched = dict(det)
                enriched["material_class"] = res.predicted_class
                enriched["material_confidence"] = res.confidence
                enriched["is_noise"] = False
                kept.append(enriched)
        return kept
