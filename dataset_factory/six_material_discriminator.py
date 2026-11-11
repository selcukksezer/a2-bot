"""Six Material Discriminator module for dataset factory.

Distinguishes the 6 core harvestable materials -- the production set from
core/material_scanner.py::MAIN_MATERIALS_6 (F-1009: this module used to
carry a different six -- herb_mela/herb_targena/gem_crystal/tree_log --
which no production flow uses):
- herb_aria
- herb_edana
- herb_lumina
- ore_orichalcum
- ore_obsidian
- crystal_blood

from each other and from environmental background clutter (grass_noise, rock_noise)
using HSV color histograms, Canny edge density, specular brightness, and aspect ratio.

Inputs matching no profile above `min_confidence` are reported as "unknown"
instead of being forced into the least-bad class.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
import sys
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np


logger = logging.getLogger(__name__)

__all__ = [
    "SixMaterialDiscriminator",
    "MaterialFeatureVector",
    "MaterialProfile",
    "DiscriminationResult",
    "MATERIAL_CLASSES",
    "NOISE_CLASSES",
    "ALL_CLASSES",
    "LEGACY_MATERIAL_ALIASES",
    "UNKNOWN_CLASS",
    "main",
]

# F-1009: the six must match the production set in
# core/material_scanner.py::MAIN_MATERIALS_6 -- the single source of truth
# for the main materials. Keep this tuple in sync with it.
MATERIAL_CLASSES: Tuple[str, ...] = (
    "herb_aria",
    "herb_edana",
    "herb_lumina",
    "ore_orichalcum",
    "ore_obsidian",
    "crystal_blood",
)

# The previous six (herb_mela, herb_targena, gem_crystal, tree_log) are
# distinct CORE_22 classes (see dataset_factory/class_taxonomy.py) that
# other flows still use (e.g. dataset_factory/active_learning_loop.py);
# they are NOT aliases of the production six, so there is no legacy
# remap here. The old "wood_tree" -> "tree_log" alias died with them:
# fit_from_samples now rejects names outside ALL_CLASSES.
LEGACY_MATERIAL_ALIASES: Dict[str, str] = {}

# Predicted when no material/noise profile scores above `min_confidence`:
# the input matches nothing known, so it must not be forced into a class.
UNKNOWN_CLASS: str = "unknown"

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

        # Fail fast on nonsense parameters instead of silently building
        # degenerate profiles (e.g. 0 bins -> empty histograms, a 1.0+
        # min_confidence -> every input "unknown", an inverted Canny
        # range -> no edges at all). Bounds are the domains the code
        # actually supports: bins >= 1, probabilities in [0, 1], 8-bit
        # thresholds in [0, 255], canny_low <= canny_high.
        if self.h_bins < 1 or self.s_bins < 1 or self.v_bins < 1:
            raise ValueError(
                f"h_bins/s_bins/v_bins must each be >= 1, got "
                f"({self.h_bins}, {self.s_bins}, {self.v_bins})"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if not 0.0 <= self.noise_margin <= 1.0:
            raise ValueError(
                f"noise_margin must be in [0.0, 1.0], got {self.noise_margin}"
            )
        if not 0 <= self.canny_low <= 255:
            raise ValueError(f"canny_low must be in [0, 255], got {self.canny_low}")
        if not 0 <= self.canny_high <= 255:
            raise ValueError(f"canny_high must be in [0, 255], got {self.canny_high}")
        if self.canny_low > self.canny_high:
            raise ValueError(
                f"canny_low ({self.canny_low}) must not exceed "
                f"canny_high ({self.canny_high})"
            )
        if not 0 <= self.specular_v_threshold <= 255:
            raise ValueError(
                f"specular_v_threshold must be in [0, 255], got {self.specular_v_threshold}"
            )
        if not 0 <= self.specular_s_max <= 255:
            raise ValueError(
                f"specular_s_max must be in [0, 255], got {self.specular_s_max}"
            )

        self.profiles: Dict[str, MaterialProfile] = self._build_default_profiles()

    @staticmethod
    def _channel_hist(
        n_bins: int, span: float, value_range: Tuple[float, float], sigma: float
    ) -> np.ndarray:
        """One normalized channel of a synthetic reference histogram.

        Vectorized form of the old per-bin Python loop: a bin scores 1.0
        when its center falls inside the range, else a Gaussian falloff by
        distance to the nearest bound. Same float64 arithmetic in the same
        order, so the result is bit-identical to the loop version.
        """
        centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * (span / n_bins)
        lo, hi = value_range
        in_range = (centers >= lo) & (centers <= hi)
        dist = np.minimum(np.abs(centers - lo), np.abs(centers - hi))
        values = np.where(in_range, 1.0, np.exp(-0.5 * (dist / sigma) ** 2))
        hist = values.astype(np.float32)
        hist /= max(float(np.sum(hist)), 1e-6)
        return hist

    def _create_synthetic_reference_hist(
        self,
        hue_range: Tuple[float, float],
        sat_range: Tuple[float, float],
        val_range: Tuple[float, float],
    ) -> np.ndarray:
        """Builds a normalized reference 1D concatenated HSV histogram for a profile."""
        h_hist = self._channel_hist(self.h_bins, 180.0, hue_range, 12.0)
        s_hist = self._channel_hist(self.s_bins, 256.0, sat_range, 25.0)
        v_hist = self._channel_hist(self.v_bins, 256.0, val_range, 25.0)

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
            # Purple/pink petals (MAIN_MATERIALS_6: herb_edana,
            # "purple/pink petals"; class_taxonomy color #a855f7).
            # Synthetic prior: recalibrate via fit_from_samples.
            "herb_edana": {
                "is_noise": False,
                "hue": (140.0, 170.0),
                "sat": (90.0, 240.0),
                "val": (110.0, 240.0),
                "edge": (0.09, 0.26),
                "spec": (0.02, 0.16),
                "aspect": (0.60, 1.40),
            },
            # Golden/yellow glowing herb (MAIN_MATERIALS_6: herb_lumina,
            # "golden/yellow glow"; class_taxonomy color #eab308).
            # Synthetic prior: recalibrate via fit_from_samples.
            "herb_lumina": {
                "is_noise": False,
                "hue": (20.0, 38.0),
                "sat": (110.0, 245.0),
                "val": (170.0, 255.0),
                "edge": (0.08, 0.24),
                "spec": (0.10, 0.35),
                "aspect": (0.60, 1.40),
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
            # Dark violet/black glassy rock vein (MAIN_MATERIALS_6:
            # ore_obsidian, "dark violet/black glassy rock vein";
            # class_taxonomy color #6366f1). Dark values, glassy glint.
            # Synthetic prior: recalibrate via fit_from_samples.
            "ore_obsidian": {
                "is_noise": False,
                "hue": (115.0, 140.0),
                "sat": (40.0, 150.0),
                "val": (30.0, 120.0),
                "edge": (0.14, 0.38),
                "spec": (0.08, 0.35),
                "aspect": (0.75, 1.65),
            },
            # Crimson red blood crystal (MAIN_MATERIALS_6: crystal_blood,
            # "crimson red blood crystal"; class_taxonomy color #991b1b).
            # Red sits at hue ~0 in OpenCV HSV (no wrap handling in the
            # range scorer, so the band starts at 0). High specular:
            # crystal facets. Synthetic prior: recalibrate via
            # fit_from_samples.
            "crystal_blood": {
                "is_noise": False,
                "hue": (0.0, 10.0),
                "sat": (120.0, 255.0),
                "val": (80.0, 200.0),
                "edge": (0.12, 0.34),
                "spec": (0.15, 0.55),
                "aspect": (0.45, 1.15),
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

    def fit_from_samples(
        self,
        samples: Iterable[
            Tuple[np.ndarray, str, Optional[Tuple[int, int, int, int]]]
        ],
    ) -> Dict[str, int]:
        """Recalibrate reference profiles from real labeled crops.

        The built-in profiles are synthetic priors (Gaussian blobs around
        hand-picked HSV ranges), not learned from real data. Call this with
        labeled crops ``(bgr_image, class_name, bbox?)`` to replace each
        covered class's reference histogram with the mean histogram of its
        real samples and to widen/narrow its scalar ranges to the observed
        min/max (expanded by 10% on each side, floored by the synthetic
        range so a tiny sample cannot collapse a profile to a point).

        Args:
            samples: ``(bgr_image, class_name, bbox_or_None)`` triples.
                ``class_name`` must be one of ``ALL_CLASSES``.

        Returns:
            Mapping of class name -> number of samples used for that class.

        Raises:
            ValueError: if a sample names an unknown class or no usable
                sample is provided for any class.
        """
        grouped: Dict[str, List[MaterialFeatureVector]] = {}
        for item in samples:
            if len(item) == 2:
                bgr_image, class_name = item  # type: ignore[misc]
                bbox = None
            else:
                bgr_image, class_name, bbox = item
            canonical = LEGACY_MATERIAL_ALIASES.get(str(class_name), str(class_name))
            if canonical not in self.profiles:
                raise ValueError(f"Unknown material class for fitting: {class_name!r}")
            grouped.setdefault(canonical, []).append(
                self.extract_features(bgr_image, bbox=bbox)
            )

        if not grouped:
            raise ValueError("fit_from_samples requires at least one labeled sample.")

        fitted: Dict[str, int] = {}
        for name, features in grouped.items():
            old = self.profiles[name]
            mean_hist = np.mean(
                np.stack([f.hsv_histogram for f in features]), axis=0
            ).astype(np.float32)
            hist_norm = float(np.linalg.norm(mean_hist))
            mean_hist = mean_hist / max(hist_norm, 1e-6)

            def _observed_range(
                values: List[float], old_range: Tuple[float, float]
            ) -> Tuple[float, float]:
                lo, hi = min(values), max(values)
                span = max(hi - lo, 1e-6)
                lo = min(old_range[0], lo - 0.10 * span)
                hi = max(old_range[1], hi + 0.10 * span)
                return (float(lo), float(hi))

            self.profiles[name] = MaterialProfile(
                name=name,
                is_noise=old.is_noise,
                hue_range=_observed_range(
                    [f.dominant_hue for f in features], old.hue_range
                ),
                saturation_range=_observed_range(
                    [f.mean_saturation for f in features], old.saturation_range
                ),
                value_range=_observed_range(
                    [f.mean_value for f in features], old.value_range
                ),
                edge_density_range=_observed_range(
                    [f.canny_edge_density for f in features], old.edge_density_range
                ),
                specular_range=_observed_range(
                    [f.specular_brightness_ratio for f in features],
                    old.specular_range,
                ),
                aspect_ratio_range=_observed_range(
                    [f.aspect_ratio for f in features], old.aspect_ratio_range
                ),
                reference_histogram=mean_hist,
            )
            fitted[name] = len(features)
        return fitted

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
            # S-5402: aspect ratio must describe the pixels actually
            # analyzed (the clipped crop), not the requested bbox -- a box
            # extending past the image edge used to keep the requested w/h.
            aspect_ratio = float(x1 - x0) / float(y1 - y0)
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
        """Classifies an image or region into one of the 6 materials, background
        noise, or "unknown" when no profile scores above `min_confidence`."""
        features = self.extract_features(bgr_image, bbox=bbox)
        scores: Dict[str, float] = {}
        for name, profile in self.profiles.items():
            scores[name] = self._score_profile(features, profile)

        best_material = max(MATERIAL_CLASSES, key=lambda k: scores[k])
        best_material_score = scores[best_material]

        best_noise = max(NOISE_CLASSES, key=lambda k: scores[k])
        best_noise_score = scores[best_noise]

        if max(best_material_score, best_noise_score) < self.min_confidence:
            # Nothing matches well enough: report "unknown" instead of forcing
            # the input into the least-bad material/noise class.
            return DiscriminationResult(
                predicted_class=UNKNOWN_CLASS,
                is_valid_material=False,
                is_noise=False,
                confidence=float(max(best_material_score, best_noise_score)),
                class_scores=scores,
                features=features,
            )

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
        for idx, det in enumerate(detections):
            bbox = det.get("bbox")
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                logger.warning(
                    "filter_detections: skipping detection #%d with malformed "
                    "bbox %r (expected (x, y, w, h))",
                    idx,
                    bbox,
                )
                continue
            try:
                bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            except (TypeError, ValueError):
                logger.warning(
                    "filter_detections: skipping detection #%d with "
                    "non-numeric bbox %r",
                    idx,
                    bbox,
                )
                continue
            res = self.discriminate(bgr_image, bbox=bbox_tuple)
            if res.is_valid_material:
                enriched = dict(det)
                enriched["material_class"] = res.predicted_class
                enriched["material_confidence"] = res.confidence
                enriched["is_noise"] = False
                kept.append(enriched)
        return kept


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
# F-1009: this module had no production entry point at all — no caller, no CLI,
# only its own file referenced it. This CLI makes it directly invocable by an
# operator (same pattern as dataset_factory/clean_dataset.py::main). The
# taxonomy is now aligned with production MAIN_MATERIALS_6 (aria/edana/
# lumina/orichalcum/obsidian/crystal_blood). Full pipeline wiring (e.g. a
# secondary visual vote inside core/material_scanner.py::
# discriminate_3d_material) is still cross-slice (R07).
def main(argv: Optional[List[str]] = None) -> int:
    """Classify one image (or a bbox crop of it) with SixMaterialDiscriminator.

    Exit codes: 0 = valid material found, 1 = noise/unknown (no material),
    2 = usage error (unreadable image / malformed --bbox).
    """
    parser = argparse.ArgumentParser(
        description="Classify an image crop into one of the 6 core materials "
        "(herb_aria, herb_edana, herb_lumina, ore_orichalcum, ore_obsidian, "
        "crystal_blood) or grass/rock noise / unknown."
    )
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument(
        "--bbox",
        default=None,
        help="Optional crop as x,y,w,h (default: whole image).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Override the default min_confidence (0.45).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the result line (exit code still reflects the outcome).",
    )
    args = parser.parse_args(argv)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: cannot read image: {args.image}", file=sys.stderr)
        return 2

    bbox: Optional[Tuple[int, int, int, int]] = None
    if args.bbox is not None:
        try:
            parts = [int(v) for v in args.bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            bbox = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            print("error: --bbox must be x,y,w,h", file=sys.stderr)
            return 2

    discriminator = (
        SixMaterialDiscriminator(min_confidence=args.min_confidence)
        if args.min_confidence is not None
        else SixMaterialDiscriminator()
    )
    result = discriminator.discriminate(image, bbox=bbox)
    if not args.quiet:
        print(
            f"predicted_class={result.predicted_class} "
            f"is_valid_material={result.is_valid_material} "
            f"is_noise={result.is_noise} "
            f"confidence={result.confidence:.3f}"
        )
    return 0 if result.is_valid_material else 1


if __name__ == "__main__":
    sys.exit(main())
