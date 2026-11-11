"""3D Game World & Minimap Automatic Ore/Gem/Material Scanner for Aion 2.

Automatically detects gatherable materials (Aria, Mela, Orichalcum Ore,
Diamond/Sapphire/Ruby Gemstones, etc.) directly from the 3D game viewport
(white symmetrical winged gather crests above cyan/green labels, diamond-tipped
resource nameplates, and floating square [E]/[F] gatherable material icons),
the interactive "F Gather" prompt, and the top-right minimap (white 3-leaf
sprout icons) while rejecting monster nameplates, monster dots, and feathers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from version import __version__

# P1-7 (K-20): minimap metre/piksel ölçeğinin tek kaynağı.
from core.corridor_navigator import MINIMAP_METERS_PER_PX

logger = logging.getLogger(__name__)


@dataclass
class MaterialDiscriminationResult:
    """Result of multi-layered material discrimination and authenticity verification."""

    is_authentic_node: bool
    material_type: str
    class_name: str
    category: str
    calibrated_confidence: float
    # Canonical class id of the chosen material (-1 = not assigned / unknown).
    class_id: int = -1
    rejection_reason: str = ""
    color_histogram_score: float = 0.0
    edge_density: float = 0.0
    laplacian_variance: float = 0.0
    specular_crest_score: float = 0.0
    nameplate_width_score: float = 0.0
    color_ratios: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_authentic_node": self.is_authentic_node,
            "material_type": self.material_type,
            "class_name": self.class_name,
            "category": self.category,
            "calibrated_confidence": self.calibrated_confidence,
            "class_id": self.class_id,
            "rejection_reason": self.rejection_reason,
            "color_histogram_score": self.color_histogram_score,
            "edge_density": self.edge_density,
            "laplacian_variance": self.laplacian_variance,
            "specular_crest_score": self.specular_crest_score,
            "nameplate_width_score": self.nameplate_width_score,
            "color_ratios": dict(self.color_ratios),
        }


# Canonical specification of the 6 main game materials
MAIN_MATERIALS_6: dict[str, dict[str, Any]] = {
    "herb_aria": {
        "canonical_name": "Aria",
        "alt_names": ["Aria", "Herb Aria", "Fresh Aria", "herb_aria"],
        "name_tr": "Aria (Mavi/Beyaz Çiçek)",
        "category": "Herbs",
        "description": "blue/white petals",
        "class_id": 0,
        "nameplate_width_range": (35, 80),
        "typical_aspect_ratio": (2.2, 4.6),
        "rarity_tier": 4,
    },
    "herb_edana": {
        "canonical_name": "Herb Edana",
        "alt_names": ["Edana", "Herb Edana", "Fresh Edana", "herb_edana"],
        "name_tr": "Edana (Mor/Pembe Çiçek)",
        "category": "Herbs",
        "description": "purple/pink petals",
        "class_id": 1,
        "nameplate_width_range": (45, 95),
        "typical_aspect_ratio": (3.0, 5.2),
        "rarity_tier": 3,
    },
    "herb_lumina": {
        "canonical_name": "Herb Lumina",
        "alt_names": ["Lumina", "Herb Lumina", "Fresh Lumina", "herb_lumina"],
        "name_tr": "Lumina (Altın/Sarı Parıltı)",
        "category": "Herbs",
        "description": "golden/yellow glow",
        "class_id": 2,
        "nameplate_width_range": (45, 100),
        "typical_aspect_ratio": (3.0, 5.4),
        "rarity_tier": 4,
    },
    "ore_orichalcum": {
        "canonical_name": "Orichalcum Ore",
        "alt_names": ["Orichalcum Ore", "Orichalcum", "ore_orichalcum"],
        "name_tr": "Orichalcum Cevheri (Bakır/Turuncu Damar)",
        "category": "Ores & Metals",
        "description": "copper/orange-brown rock vein",
        "class_id": 3,
        "nameplate_width_range": (75, 160),
        "typical_aspect_ratio": (4.0, 8.5),
        "rarity_tier": 4,
    },
    "ore_obsidian": {
        "canonical_name": "Obsidian Ore",
        "alt_names": ["Obsidian Ore", "Obsidian", "ore_obsidian", "Obsidyen Cevheri"],
        "name_tr": "Obsidyen Cevheri (Koyu Mor/Siyah Kristal)",
        "category": "Ores & Metals",
        "description": "dark violet/black glassy rock vein",
        # Canonical class id (dataset_factory/class_taxonomy.py: ore_obsidian=8).
        # Was 3, colliding with ore_orichalcum=3 so both resolved to Orichalcum.
        "class_id": 8,
        "nameplate_width_range": (70, 150),
        "typical_aspect_ratio": (3.8, 8.0),
        "rarity_tier": 4,
    },
    "crystal_blood": {
        "canonical_name": "Blood Crystal",
        "alt_names": ["Blood Crystal", "Crystal Blood", "crystal_blood", "Kan Kristali"],
        "name_tr": "Kan Kristali (Kırmızı Kristal)",
        "category": "Gemstones",
        "description": "crimson red blood crystal",
        "class_id": 4,
        "nameplate_width_range": (65, 150),
        "typical_aspect_ratio": (3.5, 7.5),
        "rarity_tier": 5,
    },
}


def ensure_aion2_materials_catalog_specs() -> None:
    """Ensure all 6 main materials have canonical MaterialSpec entries and aliases in AION2_MATERIALS_CATALOG."""
    try:
        from config.materials_db import AION2_MATERIALS_CATALOG, MaterialSpec

        specs_to_register = [
            MaterialSpec(
                name="Herb Edana",
                name_tr="Edana (Mor/Pembe Çiçek)",
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade="Standard",
                base_material="Edana",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=("Western Altamia Highland", "Verteron", "Poeta", "Eltnen"),
                hsv_ranges=(((135, 55, 75), (168, 255, 255)), ((168, 40, 130), (178, 200, 255))),
                secondary_hsv_ranges=(),
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                in_air=False,
                gather_keys=("f", "e"),
                visual_description="Purple and pink petaled herb with magenta flowers.",
            ),
            MaterialSpec(
                name="Herb Lumina",
                name_tr="Lumina (Altın/Sarı Parıltı)",
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade="Standard",
                base_material="Lumina",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=("Western Altamia Highland", "Verteron", "Altgard", "Pilgrim's Ridge"),
                hsv_ranges=(((20, 85, 130), (38, 255, 255)),),
                secondary_hsv_ranges=(),
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                in_air=False,
                gather_keys=("f", "e"),
                visual_description="Golden-yellow glowing herb with radiant luminous petals.",
            ),
            MaterialSpec(
                name="Obsidian Ore",
                name_tr="Obsidyen Cevheri",
                category="Ores & Metals",
                category_tr="Madenler / Cevherler",
                profession="Blacksmithing / Armorsmithing",
                grade="Standard",
                base_material="Obsidian Ore",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=("Morheim", "Beluslan", "Inggison", "Gelkmaros"),
                hsv_ranges=(((122, 35, 15), (155, 195, 110)), ((0, 0, 12), (180, 65, 65))),
                secondary_hsv_ranges=(),
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                in_air=False,
                gather_keys=("f", "e"),
                visual_description="Dark violet and black glassy obsidian mineral vein with specular facet glints.",
            ),
            MaterialSpec(
                name="Blood Crystal",
                name_tr="Kan Kristali",
                category="Gemstones",
                category_tr="Değerli Taşlar / Gemler",
                profession="Handicrafting",
                grade="Standard",
                base_material="Blood Crystal",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=("Morheim", "Beluslan", "Inggison", "Dredgion Crash Site"),
                hsv_ranges=(((0, 125, 75), (10, 255, 255)), ((168, 125, 75), (180, 255, 255))),
                secondary_hsv_ranges=(),
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                in_air=False,
                gather_keys=("f", "e"),
                visual_description="Crimson red blood crystal with glowing carmine crystal spires and zero green foliage.",
            ),
        ]

        by_key = getattr(AION2_MATERIALS_CATALOG, "_by_key", None)
        for spec in specs_to_register:
            if spec.name not in AION2_MATERIALS_CATALOG:
                AION2_MATERIALS_CATALOG.append(spec)
            if by_key is not None:
                by_key[spec.name.lower()] = spec
                if spec.name_tr:
                    by_key[spec.name_tr.lower()] = spec
                by_key[spec.base_material.lower()] = spec

        alias_tokens = {
            "herb_aria": "Aria",
            "aria": "Aria",
            "herb aria": "Aria",
            "fresh aria": "Fresh Aria",
            "herb_edana": "Herb Edana",
            "edana": "Herb Edana",
            "herb edana": "Herb Edana",
            "herb_lumina": "Herb Lumina",
            "lumina": "Herb Lumina",
            "herb lumina": "Herb Lumina",
            "ore_orichalcum": "Orichalcum Ore",
            "orichalcum": "Orichalcum Ore",
            "orichalcum ore": "Orichalcum Ore",
            "ore_obsidian": "Obsidian Ore",
            "obsidian": "Obsidian Ore",
            "obsidian ore": "Obsidian Ore",
            "obsidyen": "Obsidian Ore",
            "crystal_blood": "Blood Crystal",
            "blood crystal": "Blood Crystal",
            "blood": "Blood Crystal",
            "kan kristali": "Blood Crystal",
        }
        if by_key is not None:
            for token, canon in alias_tokens.items():
                target_spec = AION2_MATERIALS_CATALOG.get(canon)
                if target_spec is not None:
                    by_key[token] = target_spec

    except Exception as exc:
        logger.debug("ensure_aion2_materials_catalog_specs error: %s", exc)


# Align catalog specifications on module import
ensure_aion2_materials_catalog_specs()


def compute_edge_density_and_texture(gray: np.ndarray) -> tuple[float, float, float]:
    """Compute Canny edge density and Laplacian variance to differentiate genuine 3D objects from flat terrain/noise."""
    if gray is None or gray.size == 0:
        return 0.0, 0.0, 0.10
    h, w = gray.shape[:2]
    total_px = float(max(1, h * w))
    edges = cv2.Canny(gray, 35, 110)
    edge_density = float(np.count_nonzero(edges)) / total_px
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if edge_density < 0.025 and lap_var < 18.0:
        # Featureless flat ground / sky
        edge_score = 0.10
    elif edge_density > 0.65 and float(np.std(gray)) < 20.0:
        # High-frequency uniform texture noise
        edge_score = 0.20
    else:
        # Authentic 3D geometry contours and crystalline/petaled edges
        density_factor = min(1.0, max(0.0, (edge_density - 0.03) / 0.22))
        lap_factor = min(1.0, max(0.0, (lap_var - 20.0) / 180.0))
        edge_score = round(min(0.98, 0.45 + 0.30 * density_factor + 0.23 * lap_factor), 3)

    return edge_density, lap_var, edge_score


def compute_specular_crest_score(
    hsv_crop: np.ndarray,
    crest_patch_bgr: Optional[np.ndarray] = None,
    has_specular_crest: Optional[bool] = None,
) -> float:
    """Evaluate white winged gather crest presence and crystalline facet specular highlights."""
    if has_specular_crest is True:
        return 0.95
    if hsv_crop is None or hsv_crop.size == 0:
        return 0.15

    h, w = hsv_crop.shape[:2]
    total_px = float(max(1, h * w))

    # 1. If crest_patch_bgr is explicitly provided:
    if crest_patch_bgr is not None and crest_patch_bgr.size > 0:
        crest_hsv = cv2.cvtColor(crest_patch_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(crest_hsv, (0, 0, 168), (180, 50, 255))
        w_px = float(np.count_nonzero(white_mask))
        if w_px >= 10:
            flipped = np.fliplr(white_mask)
            inter = float(np.count_nonzero(white_mask & flipped))
            union = float(np.count_nonzero(white_mask | flipped))
            sym = inter / max(1.0, union)
            if sym >= 0.25:
                return round(min(0.99, 0.75 + 0.24 * sym), 3)

    # 2. Check upper portion of node crop for white winged gather crest
    upper_h = max(4, int(round(h * 0.35)))
    upper_hsv = hsv_crop[:upper_h, :]
    white_mask = cv2.inRange(upper_hsv, (0, 0, 168), (180, 50, 255))
    w_px = float(np.count_nonzero(white_mask))
    if w_px >= 8:
        flipped = np.fliplr(white_mask)
        inter = float(np.count_nonzero(white_mask & flipped))
        union = float(np.count_nonzero(white_mask | flipped))
        sym = inter / max(1.0, union)
        if sym >= 0.22:
            return round(min(0.96, 0.65 + 0.30 * sym), 3)

    # 3. Check for crystalline facet specular reflections across the mesh
    v_chan = hsv_crop[:, :, 2]
    s_chan = hsv_crop[:, :, 1]
    specular_mask = (v_chan >= 205) & (s_chan <= 45)
    spec_px = float(np.count_nonzero(specular_mask))
    spec_ratio = spec_px / total_px
    if spec_ratio >= 0.015:
        spec_contrast = float(np.std(v_chan[specular_mask])) if spec_px > 4 else 0.0
        return round(min(0.85, 0.40 + spec_ratio * 6.0 + (spec_contrast / 50.0) * 0.15), 3)

    return 0.18


def compute_nameplate_width_score(
    nameplate_width_px: int,
    target_width_range: tuple[int, int],
    ref_scale: float = 1.0,
) -> float:
    """Evaluate nameplate text width against canonical material label geometry."""
    if nameplate_width_px <= 0:
        # No nameplate detected: zero evidence, not a neutral 0.50 that
        # inflates the total score by 0.15 * 0.50 = 0.075.
        return 0.0
    scale = max(0.4, float(ref_scale))
    scaled_w = float(nameplate_width_px) / scale
    min_w, max_w = target_width_range
    if min_w <= scaled_w <= max_w:
        return 0.95
    elif (min_w - 15) <= scaled_w <= (max_w + 25):
        return 0.72
    elif scaled_w < 15 or scaled_w > 260:
        return 0.15
    return 0.45


def compute_color_histogram_scores(hsv: np.ndarray) -> dict[str, float]:
    """Compute multi-channel chromatic peak distribution and color histogram scores across materials and background."""
    if hsv is None or hsv.size == 0:
        return {}
    h, w = hsv.shape[:2]
    total_px = float(max(1, h * w))

    # 1. herb_aria: blue/white petals + violet-blue
    mask_blue = cv2.inRange(hsv, (88, 30, 85), (130, 220, 255))
    mask_white = cv2.inRange(hsv, (0, 0, 175), (180, 45, 255))
    mask_violet_blue = cv2.inRange(hsv, (125, 45, 85), (145, 220, 255))
    aria_px = (
        float(np.count_nonzero(mask_blue))
        + 0.6 * float(np.count_nonzero(mask_white))
        + float(np.count_nonzero(mask_violet_blue))
    )
    aria_score = min(1.0, aria_px / total_px)

    # 2. herb_edana: purple/pink petals (magenta / pink)
    mask_purple = cv2.inRange(hsv, (148, 50, 75), (168, 255, 255))
    mask_pink = cv2.inRange(hsv, (168, 40, 130), (178, 200, 255))
    edana_px = float(np.count_nonzero(mask_purple)) + float(np.count_nonzero(mask_pink))
    edana_score = min(1.0, edana_px / total_px)

    # 3. herb_lumina: golden/yellow glow
    mask_yellow = cv2.inRange(hsv, (20, 85, 130), (38, 255, 255))
    mask_glow = cv2.inRange(hsv, (22, 110, 190), (36, 255, 255))
    lumina_px = float(np.count_nonzero(mask_yellow)) + 0.5 * float(np.count_nonzero(mask_glow))
    lumina_score = min(1.0, lumina_px / total_px)

    # 4. ore_orichalcum: copper/orange-brown rock vein + dark rock matrix + oxidized teal
    mask_copper = cv2.inRange(hsv, (7, 70, 60), (26, 255, 245))
    mask_dark_rock = cv2.inRange(hsv, (0, 0, 20), (180, 65, 95))
    mask_teal = cv2.inRange(hsv, (72, 45, 65), (98, 210, 225))
    c_px = float(np.count_nonzero(mask_copper))
    r_px = float(np.count_nonzero(mask_dark_rock))
    t_px = float(np.count_nonzero(mask_teal))
    orichalcum_px = (c_px * 1.2 + 0.4 * r_px + 0.3 * t_px) if (c_px > 0 or t_px > 0) else 0.0
    orichalcum_score = min(1.0, orichalcum_px / total_px)

    # 5. ore_obsidian: dark violet/black glassy rock vein
    mask_dark_violet = cv2.inRange(hsv, (122, 35, 15), (155, 195, 110))
    mask_glassy = cv2.inRange(hsv, (0, 0, 12), (180, 65, 65))
    mask_specular = cv2.inRange(hsv, (0, 0, 195), (180, 45, 255))
    dv_px = float(np.count_nonzero(mask_dark_violet))
    gl_px = float(np.count_nonzero(mask_glassy))
    sp_px = float(np.count_nonzero(mask_specular))
    obsidian_px = (dv_px * 1.2 + 0.5 * gl_px + 0.6 * sp_px) if (dv_px > 0) else 0.0
    obsidian_score = min(1.0, obsidian_px / total_px)

    # 6. crystal_blood: crimson red blood crystal
    mask_red1 = cv2.inRange(hsv, (0, 125, 75), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (168, 125, 75), (180, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 60, 50), (82, 240, 220))
    # Counted once and reused below (blood green veto, mela, targena);
    # previously the same mask was counted three times.
    green_px = float(np.count_nonzero(mask_green))
    green_ratio = green_px / total_px
    if green_ratio > 0.08:
        blood_score = 0.0
    else:
        blood_px = float(np.count_nonzero(mask_red1)) + float(np.count_nonzero(mask_red2))
        blood_score = min(1.0, blood_px / total_px)

    # Additional catalog classes: Mela, Targena
    mask_mela_red = cv2.inRange(hsv, (0, 110, 85), (12, 255, 255)) | cv2.inRange(hsv, (168, 110, 85), (180, 255, 255))
    mela_red_px = float(np.count_nonzero(mask_mela_red))
    mela_score = min(1.0, (mela_red_px * 1.2 + green_px * 0.8) / total_px) if (mela_red_px > 0 and green_px > 0) else 0.0

    mask_targena = cv2.inRange(hsv, (26, 110, 110), (38, 255, 255))
    targena_px = float(np.count_nonzero(mask_targena))
    targena_score = min(1.0, (targena_px * 1.1 + green_px * 0.5) / total_px) if targena_px > 0 else 0.0

    # Background terrain ratios
    mask_dirt = cv2.inRange(hsv, (12, 15, 35), (28, 65, 125))
    dirt_terrain_ratio = float(np.count_nonzero(mask_dirt)) / total_px

    mask_gray_rock = cv2.inRange(hsv, (0, 0, 35), (180, 28, 135))
    barren_gray_rock_ratio = float(np.count_nonzero(mask_gray_rock)) / total_px

    mask_dense_bush = cv2.inRange(hsv, (35, 60, 40), (85, 250, 190))
    dense_bush_ratio = float(np.count_nonzero(mask_dense_bush)) / total_px

    return {
        "herb_aria": aria_score,
        "herb_edana": edana_score,
        "herb_lumina": lumina_score,
        "ore_orichalcum": orichalcum_score,
        "ore_obsidian": obsidian_score,
        "crystal_blood": blood_score,
        "mela": mela_score,
        "targena": targena_score,
        "dirt_terrain_ratio": dirt_terrain_ratio,
        "barren_gray_rock_ratio": barren_gray_rock_ratio,
        "dense_bush_ratio": dense_bush_ratio,
    }


def discriminate_3d_material(
    patch_bgr: np.ndarray,
    hsv: Optional[np.ndarray] = None,
    nameplate_width_px: int = 0,
    has_specular_crest: Optional[bool] = None,
    crest_patch_bgr: Optional[np.ndarray] = None,
    exemplar_memory: Optional[Any] = None,
    aspect_ratio: Optional[float] = None,
    ref_scale: float = 1.0,
    precomputed_exemplar: Optional[PrecomputedExemplarVerdict] = None,
) -> MaterialDiscriminationResult:
    """Multi-layered discrimination between the 6 main game materials and background terrain.

    Separates:
    - herb_aria: blue/white petals
    - herb_edana: purple/pink petals
    - herb_lumina: golden/yellow glow
    - ore_orichalcum: copper/orange-brown rock vein
    - ore_obsidian: dark violet/black glassy rock vein
    - crystal_blood: crimson red blood crystal
    from background ground/dirt/bushes/rocks using Color Histogram + Edge Density + Specular Crest + Nameplate Text Width.
    """
    if (
        patch_bgr is None
        or not isinstance(patch_bgr, np.ndarray)
        or patch_bgr.size < 16
        or patch_bgr.ndim != 3
        or patch_bgr.shape[2] < 3
    ):
        return MaterialDiscriminationResult(
            is_authentic_node=False,
            material_type="",
            class_name="",
            category="",
            calibrated_confidence=0.0,
            rejection_reason="invalid_crop",
        )

    if hsv is None:
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    # 0. ExemplarMemory Positive Match (Immunity Shield)
    mem = exemplar_memory
    if mem is None:
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
        except Exception:
            pass

    has_pos_exemplar = False
    pos_cid = None
    pos_cname = ""
    pos_sim = 0.0
    if precomputed_exemplar is not None and precomputed_exemplar.positive is not None:
        # Same call the caller already made on this exact crop
        # (classify_patch with min_sim=0.70); reuse it instead of paying
        # for the same feature extraction a second time.
        pos_cid, pos_cname, pos_sim = precomputed_exemplar.positive
        if pos_cid is not None and float(pos_sim) >= 0.70:
            has_pos_exemplar = True
    elif mem is not None and hasattr(mem, "classify_patch"):
        try:
            pos_cid, pos_cname, pos_sim = mem.classify_patch(patch_bgr, min_sim=0.70)
            if pos_cid is not None and float(pos_sim) >= 0.70:
                has_pos_exemplar = True
        except Exception:
            pass

    # 1. Check ExemplarMemory negative match (learned false patches).
    # A positive exemplar match does NOT exempt a patch from this gate.
    _neg_verdict: Optional[tuple[bool, float, Optional[dict[str, Any]]]] = None
    if precomputed_exemplar is not None and precomputed_exemplar.negative is not None:
        _neg_verdict = precomputed_exemplar.negative
    elif mem is not None and hasattr(mem, "is_negative_patch"):
        try:
            _neg_verdict = mem.is_negative_patch(patch_bgr)
        except Exception:
            _neg_verdict = None
    if _neg_verdict is not None:
        is_neg, neg_sim, neg_ex = _neg_verdict
        if is_neg:
            cname = neg_ex.get("class_name", "negative") if neg_ex else "negative"
            return MaterialDiscriminationResult(
                is_authentic_node=False,
                material_type="Negative Terrain",
                class_name="negative",
                category="",
                calibrated_confidence=round(max(0.05, 1.0 - float(neg_sim)), 3),
                rejection_reason=f"learned_negative:{cname}",
            )

    # 1b. Check node quality gate for barren rock / empty ground.
    # A positive exemplar match does NOT exempt a patch from this gate.
    _fake_verdict: Optional[tuple[bool, float, str]] = None
    if precomputed_exemplar is not None and precomputed_exemplar.fake_gate is not None:
        _fake_verdict = precomputed_exemplar.fake_gate
    else:
        try:
            from core.node_quality_gate import is_fake_gray_rock_or_ground
            _fake_verdict = is_fake_gray_rock_or_ground(patch_bgr, exemplar_memory=mem)
        except Exception:
            _fake_verdict = None
    if _fake_verdict is not None:
        is_fake, fake_conf, fake_reason = _fake_verdict
        if is_fake:
            return MaterialDiscriminationResult(
                is_authentic_node=False,
                material_type="Fake Rock/Ground",
                class_name="negative",
                category="",
                calibrated_confidence=round(max(0.05, 1.0 - float(fake_conf)), 3),
                rejection_reason=f"fake_terrain:{fake_reason}",
            )

    # 2. Edge Density & Texture Variance
    edge_density, lap_var, edge_score = compute_edge_density_and_texture(gray)

    # Featureless flat ground/sky check (no exemplar immunity: applies to
    # every patch). It runs after the colour histogram below so a patch
    # with strong material colour evidence is not called "featureless".
    std_v = float(np.std(hsv[:, :, 2]))
    std_s = float(np.std(hsv[:, :, 1]))

    # 3. Specular Crest & Facet Highlight
    specular_score = compute_specular_crest_score(
        hsv, crest_patch_bgr=crest_patch_bgr, has_specular_crest=has_specular_crest
    )

    # 4. Color Histogram & Chromatic Scores
    color_scores = compute_color_histogram_scores(hsv)

    _flat_mat_ratio = max(
        color_scores.get("herb_aria", 0),
        color_scores.get("herb_edana", 0),
        color_scores.get("herb_lumina", 0),
        color_scores.get("ore_orichalcum", 0),
        color_scores.get("ore_obsidian", 0),
        color_scores.get("crystal_blood", 0),
    )
    # A textureless patch is "featureless terrain" UNLESS it carries vivid
    # material colour (strong material ratio AND high mean saturation):
    # a solid purple Aria / solid green herb patch is colour evidence, while
    # a muted flat brown/grey patch (dirt, S ~ 90) is terrain.
    _mean_saturation = float(np.mean(hsv[:, :, 1]))
    _vivid_material = _flat_mat_ratio >= 0.05 and _mean_saturation >= 150.0
    if (
        not has_specular_crest
        and not _vivid_material
        and std_v < 6.5
        and std_s < 6.0
        and lap_var < 20.0
    ):
        return MaterialDiscriminationResult(
            is_authentic_node=False,
            material_type="Flat Terrain",
            class_name="negative",
            category="",
            calibrated_confidence=0.08,
            rejection_reason="flat_featureless_terrain",
            edge_density=edge_density,
            laplacian_variance=lap_var,
        )

    # 5. Background Terrain Rejection
    max_mat_ratio = max(
        color_scores.get("herb_aria", 0),
        color_scores.get("herb_edana", 0),
        color_scores.get("herb_lumina", 0),
        color_scores.get("ore_orichalcum", 0),
        color_scores.get("ore_obsidian", 0),
        color_scores.get("crystal_blood", 0),
    )

    if specular_score < 0.35:
        if color_scores.get("barren_gray_rock_ratio", 0) >= 0.65 and max_mat_ratio < 0.05:
            return MaterialDiscriminationResult(
                is_authentic_node=False,
                material_type="Barren Rock",
                class_name="negative",
                category="",
                calibrated_confidence=0.12,
                rejection_reason="barren_gray_rock",
                edge_density=edge_density,
                laplacian_variance=lap_var,
                specular_crest_score=specular_score,
                color_ratios=color_scores,
            )
        if color_scores.get("dense_bush_ratio", 0) >= 0.70 and max_mat_ratio < 0.04:
            return MaterialDiscriminationResult(
                is_authentic_node=False,
                material_type="Dense Bush",
                class_name="negative",
                category="",
                calibrated_confidence=0.12,
                rejection_reason="dense_bush",
                edge_density=edge_density,
                laplacian_variance=lap_var,
                specular_crest_score=specular_score,
                color_ratios=color_scores,
            )
        if color_scores.get("dirt_terrain_ratio", 0) >= 0.70 and max_mat_ratio < 0.05:
            return MaterialDiscriminationResult(
                is_authentic_node=False,
                material_type="Dirt Ground",
                class_name="negative",
                category="",
                calibrated_confidence=0.10,
                rejection_reason="dirt_ground",
                edge_density=edge_density,
                laplacian_variance=lap_var,
                specular_crest_score=specular_score,
                color_ratios=color_scores,
            )

    # 6. Multi-Layer Discrimination Among 6 Materials
    best_mat_key = "ore_orichalcum"
    best_mat_score = -1.0
    best_color_score = 0.0
    best_nameplate_score = 0.0

    for mat_key, mat_info in MAIN_MATERIALS_6.items():
        c_score = color_scores.get(mat_key, 0.0)
        np_score = compute_nameplate_width_score(
            nameplate_width_px, mat_info["nameplate_width_range"], ref_scale=ref_scale
        )
        total_score = 0.40 * c_score + 0.25 * edge_score + 0.20 * specular_score + 0.15 * np_score
        # Aspect-ratio evidence: reward a matching nameplate shape, penalise
        # a wildly different one (was an unused parameter).
        if aspect_ratio is not None:
            ar_lo, ar_hi = mat_info["typical_aspect_ratio"]
            if ar_lo <= aspect_ratio <= ar_hi:
                total_score += 0.05
            elif aspect_ratio < ar_lo * 0.6 or aspect_ratio > ar_hi * 1.6:
                total_score -= 0.05
        if total_score > best_mat_score:
            best_mat_score = total_score
            best_mat_key = mat_key
            best_color_score = c_score
            best_nameplate_score = np_score

    # Check Mela and Targena special cases
    if color_scores.get("mela", 0.0) > best_color_score + 0.20:
        best_mat_key = "herb_mela"
        best_mat_score = 0.40 * color_scores["mela"] + 0.25 * edge_score + 0.20 * specular_score + 0.15 * 0.75
        chosen_name = "Mela"
        chosen_cat = "Cooking & Fibers"
        chosen_cid = 1
    elif color_scores.get("targena", 0.0) > best_color_score + 0.20:
        best_mat_key = "herb_targena"
        best_mat_score = 0.40 * color_scores["targena"] + 0.25 * edge_score + 0.20 * specular_score + 0.15 * 0.75
        chosen_name = "Targena"
        chosen_cat = "Herbs"
        chosen_cid = 2
    else:
        mat_meta = MAIN_MATERIALS_6[best_mat_key]
        chosen_name = mat_meta["canonical_name"]
        chosen_cat = mat_meta["category"]
        chosen_cid = mat_meta["class_id"]

    # 7. ExemplarMemory positive reinforcement
    if has_pos_exemplar:
        best_mat_score = min(0.99, max(best_mat_score, float(pos_sim)) + 0.08)
        cn = str(pos_cname or "").lower()
        matched_pos = False
        for mk, mv in MAIN_MATERIALS_6.items():
            if any(alias.lower() in cn for alias in mv["alt_names"]):
                chosen_name = mv["canonical_name"]
                chosen_cat = mv["category"]
                best_mat_key = mk
                matched_pos = True
                break
        if not matched_pos:
            if pos_cid == 4 or any(g in cn for g in ("gem", "crystal", "sapphire", "ruby", "diamond")):
                chosen_name = "Diamond Gemstone"
                chosen_cat = "Gemstones"
                best_mat_key = "gem_crystal"
            elif pos_cid == 5 or any(t in cn for t in ("tree", "log", "wood", "yggdrasil")):
                chosen_name = "Yggdrasil Log"
                chosen_cat = "Logs & Trees"
                best_mat_key = "tree_log"
            elif pos_cid == 1 or "mela" in cn:
                chosen_name = "Mela"
                chosen_cat = "Cooking & Fibers"
                best_mat_key = "herb_mela"
            elif pos_cid == 2 or "targena" in cn:
                chosen_name = "Targena"
                chosen_cat = "Herbs"
                best_mat_key = "herb_targena"

    # 8. Calibration of Confidence. A positive exemplar may confirm the
    # measured score but never floors confidence at 0.85 on its own.
    if has_pos_exemplar:
        calibrated_conf = min(0.99, max(best_mat_score, float(pos_sim)))
    elif specular_score >= 0.75:
        calibrated_conf = max(best_mat_score, 0.86)
    else:
        calibrated_conf = best_mat_score

    if best_color_score < 0.04 and specular_score < 0.30 and edge_score < 0.30:
        return MaterialDiscriminationResult(
            is_authentic_node=False,
            material_type=chosen_name,
            class_name=best_mat_key,
            category=chosen_cat,
            calibrated_confidence=round(min(0.20, max(0.05, calibrated_conf * 0.25)), 3),
            rejection_reason="insufficient_material_evidence",
            class_id=chosen_cid,
            color_histogram_score=round(best_color_score, 3),
            edge_density=edge_density,
            laplacian_variance=lap_var,
            specular_crest_score=specular_score,
            nameplate_width_score=best_nameplate_score,
            color_ratios=color_scores,
        )

    # Weak evidence keeps a low confidence; the old 0.50 floor reported
    # barely-evidenced patches as real nodes at >= 0.50 confidence.
    calibrated_conf = round(float(np.clip(calibrated_conf, 0.05, 0.98)), 3)

    return MaterialDiscriminationResult(
        is_authentic_node=True,
        material_type=chosen_name,
        class_name=best_mat_key,
        category=chosen_cat,
        calibrated_confidence=calibrated_conf,
        rejection_reason="",
        class_id=chosen_cid,
        color_histogram_score=round(best_color_score, 3),
        edge_density=edge_density,
        laplacian_variance=lap_var,
        specular_crest_score=specular_score,
        nameplate_width_score=best_nameplate_score,
        color_ratios=color_scores,
    )


def _measured_max(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """max() over measured values only; None when neither side measured."""
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


def _measured_min(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """min() over measured values only; None when neither side measured."""
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None


@dataclass
class DetectedMaterial:
    """Represents a gatherable resource node detected on screen or minimap."""

    material_type: str
    source: str
    screen_x: int
    screen_y: int
    bbox: tuple[int, int, int, int]
    rel_map_dx: float
    rel_map_dy: float
    estimated_distance_m: float
    confidence: float
    in_air: bool = False
    # NOTE: no scanner measures these. None = unmeasured / unknown; never
    # fabricate stack_count=1 / harvests_remaining=3. Callers must treat
    # None as "unknown" (priority assumes a fresh node, is_depleted is False).
    stack_count: Optional[int] = None
    harvests_remaining: Optional[int] = None
    priority_score: float = 0.0
    rarity_tier: int = 1

    @property
    def is_depleted(self) -> bool:
        """True only when a measured harvest count reached zero.

        An unmeasured (None) count is unknown, not depleted.
        """
        return self.harvests_remaining is not None and self.harvests_remaining <= 0


@dataclass(frozen=True)
class PrecomputedExemplarVerdict:
    """Exemplar verdicts already measured for one crop, handed to
    discriminate_3d_material so it does not pay for the same feature
    extraction twice.

    A caller that already ran is_negative_patch / classify_patch /
    is_fake_gray_rock_or_ground on the exact same crop with the exact same
    memory (same arguments, same defaults) passes the measured results
    here. Each field is a plain measured result; a None field means "not
    precomputed — measure it inside discriminate_3d_material" (the old
    behaviour). No verdict is ever fabricated: only results returned by
    the real exemplar calls may be stored here.
    """

    positive: Optional[tuple[Optional[int], Optional[str], float]] = None
    negative: Optional[tuple[bool, float, Optional[dict[str, Any]]]] = None
    fake_gate: Optional[tuple[bool, float, str]] = None


# Precompiled patterns for normalize_material_name (same patterns as the
# inline re.sub calls used before; compiled once at import, not per call).
_EMOJI_ASTRAL_RE = re.compile(r"[\U00010000-\U0010ffff]")
_SYMBOL_RE = re.compile(r"[\u2600-\u27ff]")
_VARIATION_SELECTOR_RE = re.compile(r"[\ufe00-\ufe0f]")
_SEPARATOR_RE = re.compile(r"[_\-/\\]+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_TURKISH_MAP = str.maketrans("çğıöşü", "cgiosu")

# Canonical alias groups for match_material_names. Built once at import;
# previously rebuilt (dict + set literals) on every single call.
_MATERIAL_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "orichalcum": ("orichalcum", "ore orichalcum", "orichalcum ore", "orichalcum cevheri"),
    "aria": ("aria", "herb aria", "fresh aria", "taze aria"),
    "edana": ("edana", "herb edana", "fresh edana", "taze edana"),
    "lumina": ("lumina", "herb lumina", "fresh lumina", "taze lumina"),
    "obsidian": ("obsidian", "ore obsidian", "obsidian ore", "obsidyen", "obsidyen cevheri"),
    "blood_crystal": (
        "blood crystal",
        "crystal blood",
        "blood",
        "kan kristali",
        "kan kristal",
        "blood gemstone",
    ),
    "mela": ("mela", "herb mela", "fresh mela", "taze mela"),
    "targena": ("targena", "herb targena", "fresh targena", "taze targena"),
    "gem": (
        "gem",
        "gemstone",
        "gem crystal",
        "diamond gemstone",
        "diamond",
        "sapphire",
        "ruby",
        "emerald",
        "amethyst",
        "topaz",
        "odyle",
        "aether",
        "gem odyle",
    ),
    "tree": (
        "tree",
        "log",
        "tree log",
        "yggdrasil",
        "yggdrasil log",
        "agac",
        "kereste",
        "odun",
        "wood",
        "agac kereste",
    ),
}

# Words shared by many different materials; they must not match two names
# on their own ("crystal" alone must not match Blood Crystal to Sapphire).
_MATERIAL_NAME_GENERIC_WORDS: frozenset[str] = frozenset(
    {"ore", "herb", "fresh", "taze", "mor", "kristal", "bitki", "cevheri", "crystal", "gem", "gemstone"}
)


def normalize_material_name(name: str) -> str:
    """Normalize a material name by stripping emojis, punctuation, Turkish characters, and whitespace."""
    if not name:
        return ""
    text = str(name)
    # Remove emojis and special Unicode symbols (e.g. ⛏️, 🌸, 🍒, 🌿, 💎, 🪵, 🎯, 🧭, etc.)
    text = _EMOJI_ASTRAL_RE.sub("", text)
    text = _SYMBOL_RE.sub("", text)
    text = _VARIATION_SELECTOR_RE.sub("", text)
    text = text.lower().strip()
    # Normalize Turkish characters for robust matching
    text = text.translate(_TURKISH_MAP)
    # Normalize underscores, dashes, slashes to single spaces
    text = _SEPARATOR_RE.sub(" ", text)
    # Remove parentheses, brackets and non-word characters except spaces
    text = _PUNCT_RE.sub(" ", text)
    # Collapse multiple whitespaces
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# Canonical material-name alias groups for match_material_names.
# Module-level (was: rebuilt as a fresh dict literal on every call).
_MATERIAL_NAME_ALIASES: dict[str, list[str]] = {
    "orichalcum": ["orichalcum", "ore orichalcum", "orichalcum ore", "orichalcum cevheri"],
    "aria": ["aria", "herb aria", "fresh aria", "taze aria"],
    "edana": ["edana", "herb edana", "fresh edana", "taze edana"],
    "lumina": ["lumina", "herb lumina", "fresh lumina", "taze lumina"],
    "obsidian": ["obsidian", "ore obsidian", "obsidian ore", "obsidyen", "obsidyen cevheri"],
    "blood_crystal": [
        "blood crystal",
        "crystal blood",
        "blood",
        "kan kristali",
        "kan kristal",
        "blood gemstone",
    ],
    "mela": ["mela", "herb mela", "fresh mela", "taze mela"],
    "targena": ["targena", "herb targena", "fresh targena", "taze targena"],
    "gem": [
        "gem",
        "gemstone",
        "gem crystal",
        "diamond gemstone",
        "diamond",
        "sapphire",
        "ruby",
        "emerald",
        "amethyst",
        "topaz",
        "odyle",
        "aether",
        "gem odyle",
    ],
    "tree": [
        "tree",
        "log",
        "tree log",
        "yggdrasil",
        "yggdrasil log",
        "agac",
        "kereste",
        "odun",
        "wood",
        "agac kereste",
    ],
}


def match_material_names(target: str, candidate: str) -> bool:
    """Check if a candidate material name matches a target selected material."""
    t_raw = str(target or "").strip()
    c_raw = str(candidate or "").strip()
    if not t_raw or not c_raw:
        return False

    t = normalize_material_name(t_raw)
    c = normalize_material_name(c_raw)
    if not t or not c:
        return False

    # 1. Exact match after normalization
    if t == c:
        return True

    # 2. Canonical alias dictionary (module-level _MATERIAL_NAME_ALIASES)
    for group_key, group_items in _MATERIAL_NAME_ALIASES.items():
        t_match = any(item == t or item in t or t in item for item in group_items)
        c_match = any(item == c or item in c or c in item for item in group_items)
        if t_match and c_match:
            return True

    # 3. Word token intersection
    t_words = set(t.split())
    c_words = set(c.split())
    # "crystal"/"kristal"/"gem"/"gemstone" are shared by many different
    # materials (Blood Crystal vs Sapphire/Ruby/Diamond) and must not match
    # two names on their own.
    distinct_t = t_words - _MATERIAL_NAME_GENERIC_WORDS
    distinct_c = c_words - _MATERIAL_NAME_GENERIC_WORDS
    if distinct_t and distinct_c and (distinct_t & distinct_c):
        return True

    # 4. Substring containment
    if len(t) >= 4 and len(c) >= 4:
        if t in c or c in t:
            return True

    # 5. Catalog lookup from materials_db if available
    try:
        from config.materials_db import get_material_spec

        spec_t = get_material_spec(t_raw)
        spec_c = get_material_spec(c_raw)
        if spec_t and spec_c and spec_t.name.lower() == spec_c.name.lower():
            return True
    except Exception:
        pass

    return False


class MaterialScanner:
    """Scans game frames for 3D world material crests/nameplates, gather prompts, floating [E]/[F] boxes, and minimap sprout icons."""

    def set_selected_materials(self, materials: Sequence[str]) -> None:
        """Set the list of selected materials for selective gathering mode."""
        if not materials:
            self._selected_materials = []
            return
        filtered: list[str] = []
        for m in materials:
            if not m:
                continue
            s = str(m).strip()
            norm = normalize_material_name(s)
            if norm in ("tumu", "all", "*", "hepsi", "tum", "none"):
                continue
            filtered.append(s)
        self._selected_materials = filtered

    def get_selected_materials(self) -> List[str]:
        """Return the list of currently selected materials."""
        return list(getattr(self, "_selected_materials", []))

    def has_selected_materials(self) -> bool:
        """Return True if selective gathering filter is active."""
        return bool(getattr(self, "_selected_materials", None))

    def is_material_selected(self, mat: Any) -> bool:
        """Return True if the given material matches any of the selected materials."""
        selected = getattr(self, "_selected_materials", [])
        if not selected:
            return False

        mat_str = getattr(mat, "material_type", None) or getattr(mat, "name", None) or str(mat or "")
        for target in selected:
            if match_material_names(target, mat_str):
                return True
        return False

    def _probe_hybrid_scoring(self) -> bool:
        """Functional probe (S-0301): fuse one template + one YOLO candidate
        at the same screen position; real fusion must yield a `hybrid_3d`
        detection with confidence boosted above both inputs."""
        try:
            tpl = DetectedMaterial(
                material_type="Orichalcum Ore",
                source="3d_world",
                screen_x=640,
                screen_y=400,
                bbox=(620, 380, 40, 40),
                rel_map_dx=2.5,
                rel_map_dy=5.0,
                estimated_distance_m=5.6,
                confidence=0.88,
                harvests_remaining=3,
            )
            yolo = DetectedMaterial(
                material_type="ore_vein",
                source="3d_world",
                screen_x=640,
                screen_y=400,
                bbox=(620, 380, 40, 40),
                rel_map_dx=2.5,
                rel_map_dy=5.0,
                estimated_distance_m=5.6,
                confidence=0.82,
                harvests_remaining=3,
            )
            fused = self._fuse_hybrid_detections([tpl], [yolo])
            return any(
                getattr(f, "source", None) == "hybrid_3d"
                and float(getattr(f, "confidence", 0.0)) > 0.88
                for f in fused
            )
        except Exception:
            return False

    def _probe_prioritization(self) -> bool:
        """Functional probe (S-0301): rank one synthetic candidate; real
        prioritization must return it with a score in [0.0, 1.0]."""
        try:
            cand = DetectedMaterial(
                material_type="Orichalcum Ore",
                source="3d_world",
                screen_x=640,
                screen_y=400,
                bbox=(620, 380, 40, 40),
                rel_map_dx=2.5,
                rel_map_dy=5.0,
                estimated_distance_m=5.6,
                confidence=0.88,
                harvests_remaining=3,
            )
            ranked = self.prioritize_candidates([cand])
            if len(ranked) != 1:
                return False
            score = float(ranked[0].priority_score)
            return 0.0 <= score <= 1.0 and int(ranked[0].rarity_tier) >= 1
        except Exception:
            return False

    def get_material_scanner_health_summary(self) -> dict:
        # S-0301: every flag is probed for real — the contract validator
        # checks the taxonomy/entry points, the fusion and prioritization
        # probes execute the real code paths on a synthetic candidate.
        try:
            detector = getattr(self, "yolo_detector", None)
            return {
                "material_scanner_ready": bool(validate_material_scanner_contract()),
                "yolo_detector_ready": bool(detector is not None),
                "hybrid_scoring_ready": bool(self._probe_hybrid_scoring()),
                "prioritization_ready": bool(self._probe_prioritization()),
            }
        except Exception:
            return {"material_scanner_ready": False}

    def get_minimap_roi(self) -> tuple[float, float, float, float]:
        """Returns (x1, y1, x2, y2) normalized ratios, dynamically reloaded from calibrated_rois.json if present."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            mgr = get_calibrated_rois()
            rect = mgr.get_normalized_rect("minimap")
            if not rect:
                rect = mgr.get_normalized_rect("minimap_radar")
            if rect and len(rect) == 4:
                self.MINIMAP_INNER_X1_RATIO = float(rect[0])
                self.MINIMAP_INNER_Y1_RATIO = float(rect[1])
                self.MINIMAP_INNER_X2_RATIO = float(rect[2])
                self.MINIMAP_INNER_Y2_RATIO = float(rect[3])
                return rect
        except Exception:
            pass
        return (
            self.MINIMAP_INNER_X1_RATIO,
            self.MINIMAP_INNER_Y1_RATIO,
            self.MINIMAP_INNER_X2_RATIO,
            self.MINIMAP_INNER_Y2_RATIO,
        )

    def _refine_candidates_with_exemplars(
        self,
        frame_bgr: np.ndarray,
        candidates: list[DetectedMaterial],
        exemplar_memory: Optional[Any] = None,
    ) -> list[DetectedMaterial]:
        """
        Passes candidate material detections through ExemplarMemory:
        - Drops candidates that match learned negative exemplars (empty rocks, terrain, UI chrome)
        - Elevates confidence and refines material class for positive exemplars (Aria, Mela, Targena, Orichalcum, Crystals, Trees)
        """
        if not candidates or frame_bgr is None or frame_bgr.size == 0:
            return candidates

        mem = exemplar_memory
        if mem is None:
            try:
                from dataset_factory.exemplar_memory import get_exemplar_memory
                mem = get_exemplar_memory()
            except Exception as exc:
                logger.debug("ExemplarMemory not available in MaterialScanner: %s", exc)
                return candidates

        h, w = frame_bgr.shape[:2]
        refined: list[DetectedMaterial] = []

        for mat in candidates:
            # Determine crop bounding box
            if hasattr(mat, "bbox") and len(mat.bbox) == 4 and mat.bbox[2] > 2 and mat.bbox[3] > 2:
                bx, by, bw, bh = mat.bbox
            else:
                bx, by, bw, bh = mat.screen_x - 24, mat.screen_y - 24, 48, 48

            x1 = max(0, min(w - 4, int(bx)))
            y1 = max(0, min(h - 4, int(by)))
            x2 = max(x1 + 4, min(w, int(bx + bw)))
            y2 = max(y1 + 4, min(h, int(by + bh)))

            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                refined.append(mat)
                continue

            # 1. Negative Exemplar Check (learned negative exemplars)
            is_neg = False
            neg_sim = 0.0
            neg_ex = None
            if hasattr(mem, "is_negative_patch"):
                try:
                    is_neg, neg_sim, neg_ex = mem.is_negative_patch(crop)
                except Exception as exc_neg:
                    logger.debug("is_negative_patch check error: %s", exc_neg)

            if is_neg:
                cname = neg_ex.get("class_name", "negative") if neg_ex else "negative"
                logger.info(
                    "MaterialScanner filtered candidate '%s' @ (%d, %d): matched learned negative exemplar '%s' (sim=%.2f)",
                    mat.material_type,
                    mat.screen_x,
                    mat.screen_y,
                    cname,
                    neg_sim,
                )
                continue

            # 1b. Empty terrain / barren gray rock check
            _fake_gate: Optional[tuple[bool, float, str]] = None
            try:
                from core.node_quality_gate import is_fake_gray_rock_or_ground
                is_fake, fake_conf, fake_reason = is_fake_gray_rock_or_ground(crop, exemplar_memory=mem)
                _fake_gate = (bool(is_fake), float(fake_conf), str(fake_reason))
                if is_fake:
                    logger.info(
                        "MaterialScanner filtered candidate '%s' @ (%d, %d): matched empty terrain/gray rock '%s' (conf=%.2f)",
                        mat.material_type,
                        mat.screen_x,
                        mat.screen_y,
                        fake_reason,
                        fake_conf,
                    )
                    continue
            except Exception as exc_fake:
                logger.debug("is_fake_gray_rock_or_ground check error: %s", exc_fake)
                _fake_gate = None

            # Candidate family id (name → class) locks positive refine to the
            # same material family (herbs/ores/gems/trees). Without this lock,
            # classify_patch could remap Targena → Orichalcum/Mela via a
            # strong cross-family exemplar (live sim 0.71–0.91).
            cand_family_cid = self._resolve_material_family_cid(mat.material_type)
            family_allowed = self._family_allowed_classes(cand_family_cid)

            # 1b2. Learned positive identity: a patch that strongly matches a
            # taught positive exemplar (sim >= 0.70) is direct learned
            # evidence for that node, so it may skip the discrimination
            # drop below (step 2 still refines its class/confidence from
            # the same match). The classifier itself has no exemplar shield
            # anymore; this rescue lives only in the exemplar gate, whose
            # job is exactly to apply learned identities. A negative
            # exemplar match (step 1) still wins over a positive one.
            # The verdict is measured ONCE here and reused by step 1c
            # (via PrecomputedExemplarVerdict) and step 2 below, instead
            # of running the same feature extraction three times per crop.
            # Only if this call raised do we retry in step 2, preserving
            # the old fallback behaviour.
            pos_match = False
            _pos_result: Optional[tuple[Optional[int], Optional[str], float]] = None
            try:
                if hasattr(mem, "classify_patch"):
                    if family_allowed is not None:
                        _pos_result = mem.classify_patch(
                            crop, min_sim=0.70, allowed_classes=family_allowed
                        )
                    else:
                        _pos_result = mem.classify_patch(crop, min_sim=0.70)
                    _pcid, _pcname, _psim = _pos_result
                    pos_match = _pcid is not None and float(_psim) >= 0.70
            except Exception:
                pos_match = False
                _pos_result = None

            # 1c. Multi-layered 3D material discrimination & background terrain rejection
            disc_res = None
            if mat.source in {"3d_world", "3d_yolo", "3d_floating_icon"}:
                disc_res = discriminate_3d_material(
                    crop,
                    exemplar_memory=mem,
                    precomputed_exemplar=PrecomputedExemplarVerdict(
                        positive=_pos_result,
                        negative=(is_neg, neg_sim, neg_ex),
                        fake_gate=_fake_gate,
                    ),
                )
                if not disc_res.is_authentic_node and not pos_match:
                    logger.info(
                        "MaterialScanner filtered candidate '%s' @ (%d, %d): rejected non-matching background terrain '%s'",
                        mat.material_type,
                        mat.screen_x,
                        mat.screen_y,
                        disc_res.rejection_reason,
                    )
                    continue

                if str(mat.material_type).strip().lower() in self._GENERIC_3D_NAMES and disc_res.material_type:
                    mat.material_type = disc_res.material_type
                    # Discrimination may have named the node; refresh family lock.
                    cand_family_cid = self._resolve_material_family_cid(mat.material_type)
                    if cand_family_cid is None and int(getattr(disc_res, "class_id", -1) or -1) >= 0:
                        cand_family_cid = int(disc_res.class_id)
                    family_allowed = self._family_allowed_classes(cand_family_cid)

                if disc_res.calibrated_confidence > 0:
                    mat.confidence = round(
                        float(np.clip(max(float(mat.confidence), float(disc_res.calibrated_confidence)), 0.05, 0.99)),
                        3,
                    )

            if cand_family_cid is None and disc_res is not None:
                try:
                    _dcid = int(getattr(disc_res, "class_id", -1) or -1)
                except (TypeError, ValueError):
                    _dcid = -1
                if _dcid >= 0:
                    cand_family_cid = _dcid
                    family_allowed = self._family_allowed_classes(cand_family_cid)

            # 2. Positive Exemplar Check (Aria, Edana, Lumina, Orichalcum, Obsidian, Blood Crystal, etc.)
            # Reuses the step-1b2 verdict measured above (no second
            # classify_patch call on the same crop); retries only if 1b2
            # raised and left no verdict, preserving the old fallback.
            if hasattr(mem, "classify_patch"):
                try:
                    if _pos_result is not None:
                        pos_cid, pos_cname, pos_sim = _pos_result
                    elif family_allowed is not None:
                        pos_cid, pos_cname, pos_sim = mem.classify_patch(
                            crop, min_sim=0.70, allowed_classes=family_allowed
                        )
                    else:
                        pos_cid, pos_cname, pos_sim = mem.classify_patch(crop, min_sim=0.70)

                    # Hard family gate even if classify_patch was unrestricted
                    # (generic / unknown candidate name): never remap across families.
                    if (
                        pos_cid is not None
                        and cand_family_cid is not None
                        and not self._is_same_material_family(cand_family_cid, pos_cid)
                    ):
                        logger.info(
                            "MaterialScanner blocked cross-family exemplar refine "
                            "'%s' (cid=%s) <- exemplar cid=%s/%s (sim=%.2f)",
                            mat.material_type,
                            cand_family_cid,
                            pos_cid,
                            pos_cname,
                            pos_sim,
                        )
                        pos_cid, pos_cname = None, None

                    if pos_cid is not None or pos_cname:
                        # Elevate / boost confidence score
                        old_conf = float(mat.confidence)
                        boost = (1.0 - old_conf) * (float(pos_sim) - 0.50) * 0.70
                        mat.confidence = round(float(np.clip(max(old_conf + boost, float(pos_sim)), old_conf, 0.99)), 4)
                        # Refine material class if known
                        new_mat_name = None
                        if pos_cid is not None and pos_cid in self._EXEMPLAR_CLASS_ID_TO_MATERIAL:
                            new_mat_name = self._EXEMPLAR_CLASS_ID_TO_MATERIAL[pos_cid]
                        elif pos_cname:
                            cn = str(pos_cname).lower()
                            if "aria" in cn:
                                new_mat_name = "Aria"
                            elif "edana" in cn:
                                new_mat_name = "Herb Edana"
                            elif "lumina" in cn:
                                new_mat_name = "Herb Lumina"
                            elif "mela" in cn:
                                new_mat_name = "Mela"
                            elif "targena" in cn:
                                new_mat_name = "Targena"
                            elif "orichalcum" in cn:
                                new_mat_name = "Orichalcum Ore"
                            elif "obsidian" in cn:
                                new_mat_name = "Obsidian Ore"
                            elif "blood" in cn:
                                new_mat_name = "Blood Crystal"
                            elif any(g in cn for g in ("gem", "crystal", "sapphire", "ruby", "diamond")):
                                new_mat_name = "Diamond Gemstone"
                            elif any(t in cn for t in ("tree", "log", "wood", "yggdrasil")):
                                new_mat_name = "Yggdrasil Log"

                        if new_mat_name and mat.material_type != new_mat_name:
                            new_cid = self._resolve_material_family_cid(new_mat_name)
                            if (
                                cand_family_cid is not None
                                and new_cid is not None
                                and not self._is_same_material_family(cand_family_cid, new_cid)
                            ):
                                logger.info(
                                    "MaterialScanner blocked cross-family name refine "
                                    "'%s' -> '%s' (cand_cid=%s, new_cid=%s, sim=%.2f)",
                                    mat.material_type,
                                    new_mat_name,
                                    cand_family_cid,
                                    new_cid,
                                    pos_sim,
                                )
                            else:
                                logger.info(
                                    "MaterialScanner refined material '%s' -> '%s' via learned positive exemplar (cid=%s, sim=%.2f)",
                                    mat.material_type,
                                    new_mat_name,
                                    pos_cid,
                                    pos_sim,
                                )
                                mat.material_type = new_mat_name
                except Exception as exc_pos:
                    logger.debug("classify_patch check error in MaterialScanner: %s", exc_pos)

            refined.append(mat)

        return refined

    # Exemplar class id -> canonical material name (used by the refine
    # gate to translate a positive exemplar match). Class-level: it was
    # rebuilt as a fresh dict on every _refine_candidates_with_exemplars
    # call; contents never change.
    _EXEMPLAR_CLASS_ID_TO_MATERIAL: dict[int, str] = {
        0: "Aria",
        1: "Mela",
        2: "Targena",
        3: "Orichalcum Ore",
        4: "Diamond Gemstone",
        8: "Obsidian Ore",
        5: "Yggdrasil Log",
        6: "3D Crest",
        9: "Sprout",
        12: "World Resource",
    }

    @classmethod
    def _resolve_material_family_cid(cls, material_type: Any) -> Optional[int]:
        """Map a material display/name string to a family class id (0..8,12).

        Used by the refine family lock so 'Targena' never remaps to an ore
        exemplar. Returns None for unknown/generic names (no family lock).
        """
        if material_type is None:
            return None
        raw = str(material_type).strip()
        if not raw:
            return None
        low = raw.lower()
        # Exact reverse of the refine map first.
        for cid, name in cls._EXEMPLAR_CLASS_ID_TO_MATERIAL.items():
            if low == str(name).lower():
                return int(cid)
        # Substring aliases (order matters: more specific first).
        if "targena" in low:
            return 2
        if "mela" in low:
            return 1
        if "aria" in low:
            return 0
        if "edana" in low or "lumina" in low:
            # Non-canonical herb names still lock to the herb family (0..2,6,7).
            return 0
        if "orichalcum" in low:
            return 3
        if "obsidian" in low or "obsidyen" in low:
            return 8
        if any(g in low for g in ("gem", "sapphire", "ruby", "diamond")) or "blood" in low:
            return 4
        if any(t in low for t in ("tree", "log", "wood", "yggdrasil")):
            return 5
        if "crest" in low:
            return 6
        return None

    @staticmethod
    def _family_allowed_classes(cand_cid: Optional[int]) -> Optional[frozenset]:
        """Return the material-family class set for cand_cid, or None if unlocked."""
        if cand_cid is None:
            return None
        try:
            from dataset_factory.class_taxonomy import MATERIAL_FAMILIES, MATERIAL_3D_CLASSES
        except Exception:
            return None
        try:
            cid = int(cand_cid)
        except (TypeError, ValueError):
            return None
        if cid not in MATERIAL_3D_CLASSES:
            return None
        for family in MATERIAL_FAMILIES:
            if cid in family:
                return frozenset(family)
        return None

    @staticmethod
    def _is_same_material_family(a: Any, b: Any) -> bool:
        try:
            from dataset_factory.class_taxonomy import is_same_material_family
        except Exception:
            return True
        return bool(is_same_material_family(a, b))

    # Generic (unresolved) material names, per use site. Membership is
    # checked per candidate / per fused pair, so these live at class level
    # instead of being rebuilt as fresh set literals each time.
    _GENERIC_3D_NAMES: frozenset[str] = frozenset(
        {
            "3d crest",
            "3d_crest",
            "gatherable_node",
            "world resource",
            "world_resource",
            "raw_node",
            "unknown",
            "3d_world",
            "resource",
        }
    )
    _GENERIC_FUSE_NAMES: frozenset[str] = frozenset(
        {
            "3d crest",
            "3d_crest",
            "crest",
            "gatherable_node",
            "raw_node",
            "world resource",
            "world_resource",
            "ore_vein",
            "herb_flora",
            "unknown",
        }
    )
    _GENERIC_SELECTIVE_NAMES: frozenset[str] = frozenset(
        {
            "gatherable_node",
            "world resource",
            "3d crest",
            "gatherable material",
            "sprout",
            "3d_world",
        }
    )

    # YOLO detector output -> canonical material name. Rebuilt on every
    # scan_3d_yolo_materials call (i.e. every frame) before; contents are
    # fixed, so they live at class level now.
    _YOLO_CLASS_NAME_MAP: dict[str, str] = {
        "herb_aria": "Aria",
        "aria": "Aria",
        "herb_edana": "Herb Edana",
        "edana": "Herb Edana",
        "herb_lumina": "Herb Lumina",
        "lumina": "Herb Lumina",
        "herb_mela": "Mela",
        "mela": "Mela",
        "herb_targena": "Targena",
        "targena": "Targena",
        "herb_flora": "Aria",
        "ore_orichalcum": "Orichalcum Ore",
        "orichalcum": "Orichalcum Ore",
        "ore_vein": "Orichalcum Ore",
        "ore_obsidian": "Obsidian Ore",
        "obsidian": "Obsidian Ore",
        "crystal_blood": "Blood Crystal",
        "blood_crystal": "Blood Crystal",
        "blood": "Blood Crystal",
        "gem_crystal": "Diamond Gemstone",
        "gem": "Diamond Gemstone",
        "tree_log": "Yggdrasil Log",
        "tree": "Yggdrasil Log",
        "world_resource_mesh": "Orichalcum Ore",
        "gatherable_node": "gatherable_node",
    }

    # Coarse YOLO class names mapped to canonical class ids
    # (-1 = no specific material id; generic bucket only).
    _COARSE_YOLO_CLASSES: dict[str, int] = {
        "herb_flora": 0,
        "ore_vein": 3,
        "world_resource_mesh": 12,
        "gatherable_node": -1,
        "resource": -1,
        "ore": 3,
        "herb": 0,
        "gem": 4,
        "tree": 5,
    }

    _YOLO_NON_RESOURCE_KEYWORDS: frozenset[str] = frozenset(
        {
            "monster",
            "threat",
            "approaching_monster",
            "hostile",
            "mob",
            "agro",
            "player",
            "npc",
            "party",
            "ally",
            "target_hp",
            "skill",
            "path",
        }
    )

    # Reference resolution for normalized coordinate and perspective calculations
    REF_WIDTH: int = 1280
    REF_HEIGHT: int = 720

    # Player anchor position in 3D screen space (normalized)
    PLAYER_ANCHOR_X_RATIO: float = 0.50
    PLAYER_ANCHOR_Y_RATIO: float = 0.54

    # Minimap normalized ROI bounds (x: 0.855..0.992, y: 0.065..0.255)
    MINIMAP_X1_RATIO: float = 0.855
    MINIMAP_Y1_RATIO: float = 0.065
    MINIMAP_X2_RATIO: float = 0.992
    MINIMAP_Y2_RATIO: float = 0.255

    # Inner minimap radar canvas bounds (excluding header text bar and outer frame)
    MINIMAP_INNER_X1_RATIO: float = 0.858
    MINIMAP_INNER_Y1_RATIO: float = 0.088
    MINIMAP_INNER_X2_RATIO: float = 0.990
    MINIMAP_INNER_Y2_RATIO: float = 0.246

    # Primary HSV range for cyan/teal diamond-tipped resource labels (< Orichalcum >)
    CYAN_PRIMARY_LOWER: tuple[int, int, int] = (78, 90, 130)
    CYAN_PRIMARY_UPPER: tuple[int, int, int] = (102, 255, 255)

    # Secondary HSV range for un-targeted teal/green resource text in 3D world (Aria, Mela)
    CYAN_SECONDARY_LOWER: tuple[int, int, int] = (60, 50, 110)
    CYAN_SECONDARY_UPPER: tuple[int, int, int] = (102, 255, 255)

    # White/silver resource crest icon HSV range above 3D nameplates
    WHITE_CREST_LOWER: tuple[int, int, int] = (0, 0, 168)
    WHITE_CREST_UPPER: tuple[int, int, int] = (180, 60, 255)

    # Canonical 28x29 bit-packed binary mask of the 3D white symmetrical winged gather crest
    _CREST_BITMASK_ROWS: Sequence[int] = (
        81920,
        278529,
        302533659,
        369117195,
        335568911,
        469786639,
        402669583,
        427835506,
        432013430,
        207618246,
        241172878,
        242221468,
        133695484,
        133959676,
        66981880,
        16650208,
        8261504,
        4071168,
        1974016,
        1973760,
        1973760,
        1974016,
        1974016,
        1974016,
        1974016,
        4071168,
        924672,
        202752,
    )

    # Canonical 12x12 V-channel template for the full-screen minimap 3-leaf white sprout icon
    _SPROUT_12X12_V: Sequence[Sequence[int]] = (
        (193, 185, 195, 127, 178, 131, 146, 193, 196, 195, 203, 202),
        (186, 193, 165, 129, 233, 200, 115, 177, 201, 199, 208, 204),
        (176, 186, 120, 179, 230, 239, 137, 138, 194, 200, 205, 204),
        (182, 159, 124, 227, 215, 226, 212, 110, 176, 199, 201, 204),
        (191, 119, 162, 255, 199, 219, 255, 134, 135, 186, 198, 208),
        (161, 105, 205, 255, 174, 204, 255, 192, 110, 166, 194, 214),
        (113, 137, 255, 254, 169, 181, 255, 248, 135, 146, 192, 219),
        (85, 123, 237, 255, 143, 181, 255, 225, 104, 125, 213, 220),
        (127, 131, 174, 241, 146, 177, 235, 148, 132, 139, 128, 147),
        (245, 237, 206, 187, 116, 141, 215, 210, 240, 232, 166, 108),
        (206, 255, 242, 139, 63, 68, 179, 255, 255, 174, 149, 196),
        (111, 152, 146, 127, 139, 121, 113, 116, 135, 145, 205, 209),
    )

    # Canonical 17x17 V-channel template for the closeup minimap 3-leaf white sprout icon
    _SPROUT_17X17_V: Sequence[Sequence[int]] = (
        (89, 88, 87, 82, 88, 62, 40, 59, 83, 87, 89, 89, 90, 87, 82, 78, 79),
        (87, 89, 89, 79, 86, 64, 124, 94, 71, 86, 83, 90, 89, 85, 80, 81, 83),
        (95, 93, 83, 89, 68, 92, 232, 161, 55, 85, 83, 82, 83, 83, 81, 82, 83),
        (88, 87, 83, 85, 35, 158, 233, 202, 82, 75, 95, 87, 87, 87, 86, 84, 81),
        (81, 85, 94, 60, 86, 241, 211, 229, 152, 58, 88, 94, 91, 89, 89, 85, 80),
        (91, 91, 90, 45, 185, 246, 215, 228, 203, 85, 75, 92, 89, 87, 86, 86, 82),
        (99, 87, 61, 94, 237, 236, 186, 215, 239, 147, 40, 88, 91, 92, 89, 87, 85),
        (90, 78, 57, 182, 255, 255, 165, 217, 255, 200, 36, 75, 89, 97, 92, 86, 85),
        (77, 77, 85, 229, 255, 234, 155, 196, 255, 249, 122, 51, 76, 92, 88, 83, 84),
        (83, 48, 176, 249, 255, 232, 145, 204, 245, 255, 221, 54, 78, 90, 86, 81, 82),
        (72, 58, 166, 247, 255, 248, 116, 192, 255, 246, 214, 66, 41, 88, 95, 80, 93),
        (55, 39, 94, 185, 245, 236, 116, 173, 255, 206, 94, 30, 70, 73, 81, 90, 82),
        (104, 147, 161, 122, 183, 234, 89, 179, 210, 132, 183, 140, 98, 80, 50, 37, 62),
        (255, 248, 255, 228, 97, 175, 73, 141, 137, 215, 247, 255, 248, 208, 104, 48, 81),
        (154, 253, 254, 243, 220, 82, 47, 89, 202, 245, 254, 239, 177, 86, 70, 88, 76),
        (66, 133, 184, 144, 81, 43, 47, 46, 81, 127, 198, 144, 71, 80, 87, 82, 91),
        (80, 53, 63, 81, 88, 103, 96, 94, 88, 64, 50, 67, 80, 87, 90, 89, 83),
    )

    def __init__(
        self,
        perspective_scale_x: float = 0.085,
        perspective_scale_y: float = 0.115,
        meters_per_map_pixel: float = MINIMAP_METERS_PER_PX,
        templates_dir: str = "templates",
        yolo_detector: Optional[Any] = None,
    ) -> None:
        """Initialize the material scanner with perspective projection parameters and templates.

        Args:
            perspective_scale_x: Horizontal conversion scale from 3D screen offset to map pixels.
            perspective_scale_y: Vertical conversion scale from 3D screen offset to map pixels.
            meters_per_map_pixel: Conversion factor from relative MINIMAP pixels to
                estimated meters. P1-7 (K-20): tek kaynak
                `core.corridor_navigator.MINIMAP_METERS_PER_PX` (0.55); eski
                0.65 varsayılanı kaldırıldı. Değer canlı ölçümle kalibre
                edilmedi — karar gerekçesi corridor_navigator.py başlığında.
            templates_dir: Directory containing optional PNG templates.
            yolo_detector: Optional YOLO 3D detector instance.
        """
        self.perspective_scale_x = float(perspective_scale_x)
        self.perspective_scale_y = float(perspective_scale_y)
        self.meters_per_map_pixel = float(meters_per_map_pixel)
        self.templates_dir = templates_dir
        self.yolo_detector = yolo_detector
        self._yolo_detector_attempted: bool = False
        # S068 (v19/P3-8): başarısız başlatma kalıcı kilitleniyordu
        # (bir kez True olunca bir daha denenmiyordu). Son başarısız
        # deneme zamanı tutulur; 60 sn sonra yeniden deneme yapılır
        # (her taramada ağır model yükü denenmez).
        self._yolo_detector_failed_at: float = 0.0

        self._crest_template_mask = self._build_crest_template_mask()
        self._sprout_12_tpl = np.array(self._SPROUT_12X12_V, dtype=np.uint8)
        self._sprout_17_tpl = np.array(self._SPROUT_17X17_V, dtype=np.uint8)

        self._disk_crest_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "world_gather_crest.png"
        )
        self._disk_e_box_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "e_gather_icon.png"
        )
        self._disk_sprout_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "minimap_sprout.png"
        )
        self._selected_materials: list[str] = []
        self._price_scanner: Optional[Any] = None

    def _build_crest_template_mask(self) -> np.ndarray:
        """Unpack the 28x29 canonical binary mask for the 3D white winged gather crest."""
        h = len(self._CREST_BITMASK_ROWS)
        w = 29
        mask = np.zeros((h, w), dtype=np.uint8)
        for r, row_bits in enumerate(self._CREST_BITMASK_ROWS):
            for c in range(w):
                if (row_bits >> c) & 1:
                    mask[r, c] = 255
        return mask

    def _load_optional_gray_template(self, filename: str) -> Optional[np.ndarray]:
        """Load an optional grayscale template from the templates directory if present."""
        path = os.path.join(self.templates_dir, filename)
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                return img
        return None

    def estimate_material_kinah_value(self, material_name: str, count: int = 1) -> int:
        """Estimate the Kinah value for a given material and count by querying EconomyBroker.

        Returns 0 when no price is known for the material: a fabricated
        12000-kinah default used to be returned (and multiplied) here.
        The price scanner is created once and reused instead of once per call.
        """
        try:
            if self._price_scanner is None:
                from core.economy_broker import TradeBrokerPriceScanner
                self._price_scanner = TradeBrokerPriceScanner(auto_save=False)
            unit_price = self._price_scanner.get_price(material_name, default=0)
            return int(unit_price * max(1, int(count)))
        except Exception as exc:
            logger.warning("Failed to estimate material kinah value for %s: %s", material_name, exc)
            return 0

    def get_yolo_detector(self) -> Optional[Any]:
        """Return the current YOLO detector or lazily instantiate ONNXYolo3DDetector.

        S068 (v19/P3-8): başlatma hatası kalıcı değildi artık — başarısız
        denemeden 60 sn sonra yeniden denenir (model dosyası sonradan
        gelmiş / geçici yükleme hatası olabilir).
        """
        if self.yolo_detector is not None:
            return self.yolo_detector
        if not self._yolo_detector_attempted or (
            self._yolo_detector_failed_at
            and time.monotonic() - self._yolo_detector_failed_at >= 60.0
        ):
            self._yolo_detector_attempted = True
            try:
                from core.neural_audio_perception import ONNXYolo3DDetector
                self.yolo_detector = ONNXYolo3DDetector()
                self._yolo_detector_failed_at = 0.0
            except Exception as exc:
                self._yolo_detector_failed_at = time.monotonic()
                logger.debug("Could not initialize default ONNXYolo3DDetector: %s", exc)
        return self.yolo_detector

    def set_yolo_detector(self, detector: Any) -> None:
        """Set or update the active YOLO 3D detector instance."""
        self.yolo_detector = detector
        self._yolo_detector_attempted = True

    def scan_3d_yolo_materials(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect 3D ore veins, gatherable herbs, crystals, and resource meshes via YOLO detector."""
        detections: list[DetectedMaterial] = []
        detector = self.get_yolo_detector()
        if detector is None or frame_bgr is None or frame_bgr.size == 0:
            return detections

        try:
            h, w = frame_bgr.shape[:2]
            if w < 120 or h < 90:
                return detections

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            raw_dets = []
            if hasattr(detector, "detect_3d_objects"):
                raw_dets = detector.detect_3d_objects(frame_bgr)
            elif hasattr(detector, "detect"):
                raw_dets = detector.detect(frame_bgr)
            elif hasattr(detector, "predict"):
                raw_dets = detector.predict(frame_bgr)
            elif callable(detector):
                raw_dets = detector(frame_bgr)

            if not raw_dets:
                return detections

            # Lookup tables are class-level (_YOLO_CLASS_NAME_MAP etc.);
            # they were rebuilt as fresh dicts on every frame before.
            for det in raw_dets:
                cname = str(det.get("class_name", "") or det.get("label", "")).strip().lower()
                if any(kw in cname for kw in self._YOLO_NON_RESOURCE_KEYWORDS):
                    continue

                conf = float(det.get("confidence", 0.5) or 0.5)
                if conf < 0.25:
                    continue

                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    center = det.get("center")
                    if center:
                        bx = max(0, int(center[0] - 25))
                        by = max(0, int(center[1] - 25))
                        bw, bh = 50, 50
                    else:
                        continue
                else:
                    bx, by, bw, bh = [int(v) for v in bbox]

                cx = bx + bw // 2
                cy = by + bh // 2
                if cx >= int(w * 0.84) and cy <= int(h * 0.30):
                    continue
                if cy >= int(h * 0.88):
                    continue

                x1 = max(0, min(w - 2, bx))
                y1 = max(0, min(h - 2, by))
                x2 = max(x1 + 2, min(w, bx + bw))
                y2 = max(y1 + 2, min(h, by + bh))
                crop = frame_bgr[y1:y2, x1:x2]

                mapped_name = self._YOLO_CLASS_NAME_MAP.get(cname, "")
                in_air = bool(cy < int(h * 0.22))

                # Multi-layer discrimination verification & background terrain rejection
                disc_res = discriminate_3d_material(crop, ref_scale=w / float(self.REF_WIDTH))

                if disc_res.is_authentic_node:
                    if cname in self._COARSE_YOLO_CLASSES or not mapped_name:
                        mapped_name = disc_res.material_type
                    conf = max(conf, disc_res.calibrated_confidence)
                else:
                    # Discrimination said "terrain, not a node". A high YOLO
                    # confidence alone must NOT override that rejection
                    # (previously conf >= 0.65 silently kept the detection).
                    # F-1327 (R19): the rescue below is only attempted for a
                    # strong YOLO detection (confidence >= 0.65). A
                    # low-confidence detection on a featureless crop must
                    # not be rescued by the patch classifier alone: the
                    # uniform-gray crop classifies as Orichalcum Ore at
                    # ~0.87 on dark-rock evidence, so without this gate
                    # every low-confidence false positive would pass.
                    if conf < 0.65:
                        continue
                    # Only a strong INDEPENDENT patch classification (>= 0.70)
                    # rescues it, and a specific YOLO-mapped name is kept
                    # over the patch classifier's name.
                    rescued = False
                    try:
                        from config.materials_db import classify_3d_material_patch
                        patch_info = classify_3d_material_patch(crop)
                        if patch_info and patch_info.get("confidence", 0) >= 0.70:
                            p_name = patch_info.get("name")
                            if p_name and patch_info.get("color_ratios"):
                                if cname in self._COARSE_YOLO_CLASSES or not mapped_name:
                                    mapped_name = p_name
                                rescued = True
                    except Exception:
                        pass
                    if not rescued:
                        continue  # Reject non-matching background ground/dirt/bushes/rocks

                if not mapped_name:
                    # Never fabricate a material: an unclassifiable node is
                    # reported as Unknown, not as Orichalcum Ore.
                    mapped_name = "Unknown"

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(cx, cy, w, h)

                detections.append(
                    DetectedMaterial(
                        material_type=mapped_name,
                        source="3d_yolo",
                        screen_x=int(cx),
                        screen_y=int(cy),
                        bbox=(int(bx), int(by), int(bw), int(bh)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=round(conf, 3),
                        in_air=in_air,
                    )
                )

        except Exception as exc:
            logger.error("Error in scan_3d_yolo_materials: %s", exc, exc_info=True)

        return detections

    def _fuse_hybrid_detections(
        self,
        template_candidates: list[DetectedMaterial],
        yolo_candidates: list[DetectedMaterial],
    ) -> list[DetectedMaterial]:
        """
        Fuses template-based 3D detections with YOLO 3D detections.
        - When a YOLO detection overlaps or sits directly below a 3D winged crest / nameplate,
          they are merged into a verified `hybrid_3d` detection with boosted confidence
          and refined bounding box & material classification.
        - Preserves standalone template detections and standalone high-confidence YOLO detections.
        """
        if not yolo_candidates:
            return template_candidates
        if not template_candidates:
            return yolo_candidates

        fused: list[DetectedMaterial] = []
        matched_yolo_indices: set[int] = set()
        matched_tpl_indices: set[int] = set()

        for t_idx, tpl in enumerate(template_candidates):
            if tpl.source not in {"3d_world", "3d_floating_icon", "gather_prompt"}:
                continue

            tx, ty = tpl.screen_x, tpl.screen_y
            tb_x, tb_y, tb_w, tb_h = tpl.bbox

            best_y_idx: Optional[int] = None
            best_dist: float = 9999.0

            for y_idx, yolo in enumerate(yolo_candidates):
                if y_idx in matched_yolo_indices:
                    continue

                yx, yy = yolo.screen_x, yolo.screen_y
                yb_x, yb_y, yb_w, yb_h = yolo.bbox

                d_center = math.hypot(tx - yx, ty - yy)
                h_overlap = max(0, min(tb_x + tb_w, yb_x + yb_w) - max(tb_x, yb_x))
                v_dist = abs(ty - yy)

                is_match = False
                if d_center <= max(45.0, tb_w * 0.75):
                    is_match = True
                elif h_overlap >= min(tb_w, yb_w) * 0.35 and v_dist <= max(90.0, yb_h * 1.5):
                    is_match = True

                if is_match and d_center < best_dist:
                    best_dist = d_center
                    best_y_idx = y_idx

            if best_y_idx is not None:
                yolo = yolo_candidates[best_y_idx]
                matched_tpl_indices.add(t_idx)
                matched_yolo_indices.add(best_y_idx)
                yb_x, yb_y, yb_w, yb_h = yolo.bbox

                min_x = min(tb_x, yb_x)
                min_y = min(tb_y, yb_y)
                max_x = max(tb_x + tb_w, yb_x + yb_w)
                max_y = max(tb_y + tb_h, yb_y + yb_h)
                fused_bbox = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

                fused_conf = round(min(0.99, max(tpl.confidence, yolo.confidence) + 0.08), 3)

                tpl_is_generic = str(tpl.material_type or "").strip().lower() in self._GENERIC_FUSE_NAMES
                yolo_is_generic = str(yolo.material_type or "").strip().lower() in self._GENERIC_FUSE_NAMES
                if not tpl_is_generic and yolo_is_generic:
                    mat_type = tpl.material_type
                elif not yolo_is_generic:
                    mat_type = yolo.material_type
                else:
                    mat_type = tpl.material_type

                fused.append(
                    DetectedMaterial(
                        material_type=mat_type,
                        source="hybrid_3d",
                        screen_x=int((tx + yx) / 2),
                        screen_y=int(max(ty, yy)),
                        bbox=fused_bbox,
                        rel_map_dx=round((tpl.rel_map_dx + yolo.rel_map_dx) / 2.0, 2),
                        rel_map_dy=round((tpl.rel_map_dy + yolo.rel_map_dy) / 2.0, 2),
                        estimated_distance_m=round(min(tpl.estimated_distance_m, yolo.estimated_distance_m), 2),
                        confidence=fused_conf,
                        in_air=tpl.in_air or yolo.in_air,
                        stack_count=_measured_max(tpl.stack_count, yolo.stack_count),
                        harvests_remaining=_measured_min(
                            tpl.harvests_remaining, yolo.harvests_remaining
                        ),
                    )
                )

        for t_idx, tpl in enumerate(template_candidates):
            if t_idx not in matched_tpl_indices:
                fused.append(tpl)

        for y_idx, yolo in enumerate(yolo_candidates):
            if y_idx not in matched_yolo_indices:
                fused.append(yolo)

        return fused

    def compute_candidate_priority(
        self,
        candidate: DetectedMaterial,
        current_player_pos: Optional[tuple[float, float]] = None,
    ) -> float:
        """
        Computes the priority score [0.0..1.0] for a detected harvesting candidate node based on:
        1. Proximity: Distance in meters (interaction range <=2.5m / 'F Gather' prompt gets max urgency).
        2. Rarity: Material tier & trade value (Gemstones > High-value Ores/Aria > Herbs/Trees > Common).
        3. Harvest Count Remaining: Fresh 3/3 nodes prioritized; depleted (0/3) nodes heavily penalized.
        4. Detection Confidence & Sensor Verification (hybrid YOLO + template boost).
        """
        if candidate is None:
            return 0.0

        _harvests_raw = getattr(candidate, "harvests_remaining", None)
        # Unmeasured harvest count: assume a fresh node for ranking only;
        # the stored value stays None (unknown), it is not fabricated.
        harvests = 3 if _harvests_raw is None else int(_harvests_raw)
        if harvests <= 0:
            candidate.priority_score = round(0.01 * float(candidate.confidence), 4)
            candidate.rarity_tier = 1
            return candidate.priority_score

        # 1. Proximity Factor [0.05..1.0]
        dist_m = float(candidate.estimated_distance_m)
        if candidate.source == "gather_prompt":
            proximity_score = 1.0
        elif dist_m <= 2.5:
            proximity_score = 0.95
        elif dist_m <= 15.0:
            proximity_score = 0.95 - ((dist_m - 2.5) / 12.5) * 0.35
        elif dist_m <= 40.0:
            proximity_score = 0.60 - ((dist_m - 15.0) / 25.0) * 0.35
        else:
            proximity_score = max(0.05, 0.25 / (1.0 + 0.04 * (dist_m - 40.0)))

        # 2. Rarity Factor & Tier [0.15..1.0]
        mat_name = str(candidate.material_type or "").strip()
        mat_lower = mat_name.lower()

        rarity_tier = 1
        base_rarity = 0.35
        grade_bonus = 0.0

        try:
            from config.materials_db import get_material_spec
            spec = get_material_spec(mat_name)
            if spec is not None:
                cat = spec.category
                if cat == "Gemstones":
                    rarity_tier = 5
                    base_rarity = 0.92
                elif cat == "Odyle & Aether":
                    rarity_tier = 5
                    base_rarity = 0.88
                elif cat == "Ores & Metals":
                    if any(ro in mat_lower for ro in ("orichalcum", "obsidian", "adamantium", "platinum", "gold", "titanium")):
                        rarity_tier = 4
                        base_rarity = 0.82
                    elif any(co in mat_lower for co in ("iron", "copper")):
                        rarity_tier = 2
                        base_rarity = 0.50
                    else:
                        rarity_tier = 3
                        base_rarity = 0.68
                elif cat == "Herbs":
                    if any(rh in mat_lower for rh in ("aria", "lumina", "targena", "azpha", "calendula")):
                        rarity_tier = 4
                        base_rarity = 0.78
                    elif "edana" in mat_lower:
                        rarity_tier = 3
                        base_rarity = 0.70
                    else:
                        rarity_tier = 3
                        base_rarity = 0.62
                elif cat == "Logs & Trees":
                    if any(rt in mat_lower for rt in ("yggdrasil", "asvata")):
                        rarity_tier = 3
                        base_rarity = 0.68
                    else:
                        rarity_tier = 2
                        base_rarity = 0.50
                elif cat == "Cooking & Fibers":
                    if "mela" in mat_lower:
                        rarity_tier = 2
                        base_rarity = 0.55
                    else:
                        rarity_tier = 2
                        base_rarity = 0.45

                if spec.grade == "Radiant":
                    grade_bonus = 0.15
                elif spec.grade == "Pure":
                    grade_bonus = 0.10
                elif spec.grade == "Fine":
                    grade_bonus = 0.05
                elif spec.grade == "Fresh":
                    grade_bonus = -0.05
        except Exception:
            pass

        if rarity_tier == 1:
            if any(k in mat_lower for k in ("gem", "diamond", "ruby", "sapphire", "crystal", "blood", "emerald", "topaz", "odyle")):
                rarity_tier = 5
                base_rarity = 0.90
            elif any(k in mat_lower for k in ("orichalcum", "obsidian", "aria", "lumina", "targena", "adamantium")):
                rarity_tier = 4
                base_rarity = 0.80
            elif any(k in mat_lower for k in ("edana", "ore", "herb", "yggdrasil", "asvata")):
                rarity_tier = 3
                base_rarity = 0.65
            elif any(k in mat_lower for k in ("mela", "log", "tree", "plant")):
                rarity_tier = 2
                base_rarity = 0.52
            else:
                rarity_tier = 1
                base_rarity = 0.35

        kinah_bonus = 0.0
        try:
            kinah_val = self.estimate_material_kinah_value(mat_name, 1)
            if kinah_val > 20000:
                kinah_bonus = min(0.12, (kinah_val - 12000) / 200000.0)
        except Exception:
            pass

        rarity_score = min(1.0, max(0.15, base_rarity + grade_bonus + kinah_bonus))

        # 3. Harvest Count Remaining Factor [0.0..1.0]
        if harvests >= 3:
            harvest_score = 1.0
        elif harvests == 2:
            harvest_score = 0.72
        elif harvests == 1:
            harvest_score = 0.42
        else:
            harvest_score = 0.0

        # 4. Confidence & Sensor Verification Bonus
        conf = min(1.0, max(0.0, float(candidate.confidence)))
        sensor_bonus = 0.0
        if candidate.source == "hybrid_3d":
            sensor_bonus += 0.08
        elif candidate.source == "gather_prompt":
            sensor_bonus += 0.12
        elif candidate.source == "3d_world":
            sensor_bonus += 0.04

        conf_factor = min(1.0, conf + sensor_bonus)

        priority = (
            0.35 * proximity_score
            + 0.30 * rarity_score
            + 0.20 * harvest_score
            + 0.15 * conf_factor
        )

        candidate.priority_score = round(float(priority), 4)
        candidate.rarity_tier = int(rarity_tier)
        return candidate.priority_score

    def prioritize_candidates(
        self,
        candidates: list[DetectedMaterial],
        current_player_pos: Optional[tuple[float, float]] = None,
        filter_depleted: bool = False,
    ) -> list[DetectedMaterial]:
        """
        Calculates priority score for each candidate and returns the list sorted descending.

        Args:
            candidates: Input list of DetectedMaterial instances.
            current_player_pos: Optional (x, y) map coordinates of player for fine-grained distance.
            filter_depleted: If True, completely excludes nodes with harvests_remaining <= 0.

        Returns:
            Sorted list of DetectedMaterial instances with priority_score and rarity_tier set.
        """
        if not candidates:
            return []

        scored: list[DetectedMaterial] = []
        for c in candidates:
            _hr = getattr(c, "harvests_remaining", None)
            if filter_depleted and _hr is not None and _hr <= 0:
                continue
            self.compute_candidate_priority(c, current_player_pos=current_player_pos)
            scored.append(c)

        scored.sort(key=lambda item: item.priority_score, reverse=True)
        return scored

    def get_top_candidate(
        self,
        candidates: list[DetectedMaterial],
        current_player_pos: Optional[tuple[float, float]] = None,
        filter_depleted: bool = True,
    ) -> Optional[DetectedMaterial]:
        """Returns the highest priority harvesting candidate, or None if empty."""
        prioritized = self.prioritize_candidates(
            candidates, current_player_pos=current_player_pos, filter_depleted=filter_depleted
        )
        return prioritized[0] if prioritized else None

    @staticmethod
    def _dedupe_detections(
        detections: list["DetectedMaterial"], frame_w: int = 1280
    ) -> list["DetectedMaterial"]:
        """Drop duplicate reports of the same physical node from different scanners.

        Two reports whose centres lie within ~48 px (at 1280 px reference
        width, scaled with the frame) are the same node; the
        highest-confidence one wins.
        """
        if len(detections) < 2:
            return list(detections)
        radius = 48.0 * max(0.25, frame_w / 1280.0)
        ordered = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
        kept: list["DetectedMaterial"] = []
        for det in ordered:
            duplicate = False
            for other in kept:
                dx = float(det.screen_x) - float(other.screen_x)
                dy = float(det.screen_y) - float(other.screen_y)
                if dx * dx + dy * dy <= radius * radius:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def scan_frame_for_materials(
        self,
        frame_bgr: np.ndarray,
        exemplar_memory: Optional[Any] = None,
    ) -> list[DetectedMaterial]:
        """Scan a single BGR game frame for all visible materials across 3D world and minimap.

        Executes:
        1. 3D world white winged gather crest + cyan/green nameplate scanning (Aria, Mela, Orichalcum)
           with strict monster nameplate rejection (Yellow-Maned Zaif, Highland Odyle Sparkie).
        2. 3D world floating square [E]/[F] gatherable material icon scanning.
        3. Interactive "F Gather" button prompt scanning near the character.
        4. 3D YOLO object detections (ore veins, flora, crystals, prompts).
        5. Multi-modal hybrid fusion (YOLO + Template + Prompts).
        6. Top-right minimap white 3-leaf sprout icon scanning (rejecting monster dots and feathers).
        7. ExemplarMemory gate: drops negative exemplars, refines positive classes & boosts confidence.
        8. Candidate prioritization: ranks targets by proximity, rarity tier, and harvests remaining.

        Args:
            frame_bgr: Input image in BGR format (numpy ndarray).
            exemplar_memory: Optional ExemplarMemory instance (defaults to singleton).

        Returns:
            List of DetectedMaterial objects sorted by priority score descending.
        """
        try:
            if (
                frame_bgr is None
                or not isinstance(frame_bgr, np.ndarray)
                or frame_bgr.size == 0
                or frame_bgr.ndim != 3
                or frame_bgr.shape[2] < 3
            ):
                logger.warning("Invalid frame provided to scan_frame_for_materials.")
                return []

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, w = frame_bgr.shape[:2]

            # Only a narrow crop (w < 500) is treated as a minimap closeup.
            # A 960x540 frame is a real game frame and must get the full 3D scan.
            if w < 500:
                minimap_nodes = self.scan_minimap_materials(frame_bgr, hsv)
                minimap_nodes = self._refine_candidates_with_exemplars(
                    frame_bgr, minimap_nodes, exemplar_memory=exemplar_memory
                )
                prioritized = self.prioritize_candidates(minimap_nodes)
                if self.has_selected_materials():
                    prioritized = [c for c in prioritized if self.is_material_selected(c.material_type)]
                return prioritized

            # Reject full-screen modal settings/import dialogs so UI elements are never misclassified
            if self._is_fullscreen_modal_open(frame_bgr, hsv):
                logger.debug("Full-screen modal/settings dialog detected; skipping material scan.")
                return []

            results: list[DetectedMaterial] = []

            # 1. Scan 3D world white winged gather crests and cyan/teal resource nameplates
            world_materials = self.scan_3d_world_nameplates(frame_bgr, hsv)
            results.extend(world_materials)

            # 2. Scan floating square [E] / [F] gatherable material icons in the 3D world
            floating_boxes = self._scan_floating_gather_boxes(frame_bgr, hsv)
            results.extend(floating_boxes)

            # 3. Scan "F Gather" interactive prompt near the player
            gather_prompts = self.scan_gather_prompt(frame_bgr, hsv, world_materials)
            results.extend(gather_prompts)

            # 4. Scan 3D YOLO object detections (ore veins, flora, crystals, prompts)
            yolo_materials = self.scan_3d_yolo_materials(frame_bgr, hsv)

            # 5. Fuse hybrid 3D detections (YOLO + Template + Prompts)
            template_3d = list(results)
            results = self._fuse_hybrid_detections(template_3d, yolo_materials)

            # 6. Scan top-right minimap for 3-leaf sprout gatherable material icons
            minimap_nodes = self.scan_minimap_materials(frame_bgr, hsv)
            results.extend(minimap_nodes)

            # 6b. Cross-source dedup: the same physical node was reported by
            # several scanners (world / floating box / gather prompt / YOLO /
            # minimap). Keep only the highest-confidence report per location.
            results = self._dedupe_detections(results, frame_w=w)

            # 7. Pass candidates through ExemplarMemory gate:
            # - Drop candidates that match learned negative exemplars (empty rocks, terrain, UI chrome)
            # - Elevate confidence score and refine material class for learned positive exemplars
            results = self._refine_candidates_with_exemplars(
                frame_bgr, results, exemplar_memory=exemplar_memory
            )

            # 8. Prioritize candidates by proximity, rarity, and harvest count remaining
            results = self.prioritize_candidates(results)

            # 9. Selective gather filter: if selected materials are active, strictly filter candidates
            if self.has_selected_materials():
                filtered: list[DetectedMaterial] = []
                for c in results:
                    mtype = getattr(c, "material_type", "")
                    if self.is_material_selected(mtype):
                        filtered.append(c)
                    elif str(mtype).lower() in self._GENERIC_SELECTIVE_NAMES:
                        # Attempt patch classification if frame_bgr is available
                        if frame_bgr is not None and frame_bgr.size > 0:
                            try:
                                from config.materials_db import classify_3d_material_patch
                                bx, by, bw, bh = (
                                    c.bbox
                                    if hasattr(c, "bbox") and len(c.bbox) == 4 and c.bbox[2] > 2
                                    else (c.screen_x - 24, c.screen_y - 24, 48, 48)
                                )
                                h_f, w_f = frame_bgr.shape[:2]
                                x1, y1 = max(0, int(bx)), max(0, int(by))
                                x2, y2 = min(w_f, int(bx + bw)), min(h_f, int(by + bh))
                                if x2 > x1 + 4 and y2 > y1 + 4:
                                    p_crop = frame_bgr[y1:y2, x1:x2]
                                    p_info = classify_3d_material_patch(p_crop)
                                    if p_info and p_info.get("confidence", 0) >= 0.65:
                                        p_name = p_info.get("name")
                                        if p_name and self.is_material_selected(p_name):
                                            c.material_type = p_name
                                            filtered.append(c)
                            except Exception:
                                pass
                results = filtered

            return results

        except Exception as exc:
            logger.error("Error in scan_frame_for_materials: %s", exc, exc_info=True)
            return []

    def _scan_3d_gather_crests(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect white symmetrical winged gather crests floating above cyan/green labels (Aria, Mela).

        Rejects yellow/olive monster labels (e.g. Highland Odyle Sparkie, Yellow-Maned Zaif)
        and player character nameplates.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_world".
        """
        detections: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return detections

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            if w < 600 or h < 360:
                return detections

            scale_x = w / 2000.0
            scale_y = h / 1124.0

            # Restrict search to 3D world viewport
            vx1 = int(w * 0.08)
            vy1 = int(h * 0.12)
            vx2 = int(w * 0.82)
            vy2 = int(h * 0.78)

            roi_hsv = hsv[vy1:vy2, vx1:vx2]
            if roi_hsv.size == 0:
                return detections

            white_mask = cv2.inRange(
                roi_hsv, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER
            )

            # Multi-scale template matching using canonical winged crest mask (and disk template if present)
            candidate_boxes: list[tuple[float, int, int, int, int]] = []
            scales = [1.0]
            if abs(scale_x - 1.0) > 0.08:
                scales.append(max(0.55, min(1.35, scale_x)))

            for s in scales:
                tw = max(16, int(round(self._crest_template_mask.shape[1] * s)))
                th = max(16, int(round(self._crest_template_mask.shape[0] * s)))
                if tw >= white_mask.shape[1] or th >= white_mask.shape[0]:
                    continue
                tpl = (
                    self._crest_template_mask
                    if (tw == self._crest_template_mask.shape[1] and th == self._crest_template_mask.shape[0])
                    else cv2.resize(self._crest_template_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                )
                res = cv2.matchTemplate(white_mask, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= 0.58)
                for y_pt, x_pt in zip(ys, xs):
                    candidate_boxes.append(
                        (float(res[y_pt, x_pt]), int(x_pt), int(y_pt), int(tw), int(th))
                    )

            if self._disk_crest_tpl is not None:
                d_th, d_tw = self._disk_crest_tpl.shape[:2]
                if d_tw < roi_hsv.shape[1] and d_th < roi_hsv.shape[0]:
                    res_d = cv2.matchTemplate(
                        roi_hsv[:, :, 2], self._disk_crest_tpl, cv2.TM_CCOEFF_NORMED
                    )
                    ys_d, xs_d = np.where(res_d >= 0.62)
                    for y_pt, x_pt in zip(ys_d, xs_d):
                        candidate_boxes.append(
                            (float(res_d[y_pt, x_pt]), int(x_pt), int(y_pt), int(d_tw), int(d_th))
                        )

            candidate_boxes.sort(key=lambda item: item[0], reverse=True)
            kept_crests: list[tuple[float, int, int, int, int]] = []
            for score, cx_b, cy_b, cw, ch in candidate_boxes:
                if any(
                    abs(cx_b - kx) <= max(16, cw // 2) and abs(cy_b - ky) <= max(16, ch // 2)
                    for _, kx, ky, _, _ in kept_crests
                ):
                    continue
                kept_crests.append((score, cx_b, cy_b, cw, ch))

            for score, cx_b, cy_b, cw, ch in kept_crests:
                sub_wm = white_mask[cy_b : cy_b + ch, cx_b : cx_b + cw]
                white_px = int(np.count_nonzero(sub_wm))
                if white_px < max(60, int(round(110 * scale_x * scale_y))):
                    continue

                # Verify horizontal left-right mirror symmetry of the winged crest
                flipped = np.fliplr(sub_wm)
                inter = int(np.count_nonzero(sub_wm & flipped))
                union = int(np.count_nonzero(sub_wm | flipped))
                sym_ratio = inter / float(max(1, union))
                if sym_ratio < 0.42:
                    continue

                # Inspect the label region directly below the winged crest for cyan/green text (Aria, Mela)
                tx1 = max(0, cx_b - int(round(12 * scale_x)))
                tx2 = min(roi_hsv.shape[1], cx_b + cw + int(round(12 * scale_x)))
                ty1 = max(0, cy_b + ch)
                ty2 = min(roi_hsv.shape[0], cy_b + ch + max(18, int(round(32 * scale_y))))
                if ty2 <= ty1 or tx2 <= tx1:
                    continue

                label_hsv = roi_hsv[ty1:ty2, tx1:tx2]
                cg_mask = cv2.inRange(
                    label_hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
                )
                cg_px = int(np.count_nonzero(cg_mask))
                min_cg_px = max(25, int(round(50 * scale_x * scale_y)))
                if cg_px < min_cg_px:
                    continue

                abs_x = vx1 + cx_b
                abs_y = vy1 + cy_b
                abs_label_y2 = vy1 + ty2

                # Reject if this is a monster nameplate (yellow text + red agro badge)
                if self._is_monster_nameplate(
                    hsv,
                    abs_x,
                    vy1 + ty1,
                    cw,
                    ty2 - ty1,
                    w / float(self.REF_WIDTH),
                    h / float(self.REF_HEIGHT),
                ):
                    continue

                material_type, in_air = self._classify_3d_material(
                    frame_bgr,
                    hsv,
                    abs_x,
                    vy1 + ty1,
                    cw,
                    ty2 - ty1,
                )

                # v20 (MADDE 8): boş materyal adı = mesh learned-negative
                # reddi (canavar/oyuncu) — tespit üretilmez.
                if not material_type:
                    continue

                node_screen_x = int(abs_x + cw // 2)
                node_screen_y = int(
                    min(h - 1, abs_label_y2 + int(round(24 * (h / float(self.REF_HEIGHT)))))
                )
                full_bbox = (
                    int(vx1 + min(cx_b, tx1)),
                    int(abs_y),
                    int(max(cw, tx2 - tx1)),
                    int(abs_label_y2 - abs_y),
                )

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(
                    node_screen_x, node_screen_y, w, h
                )
                conf = round(min(0.99, 0.15 + 0.70 * score + 0.05 * sym_ratio), 3)

                detections.append(
                    DetectedMaterial(
                        material_type=material_type,
                        source="3d_world",
                        screen_x=node_screen_x,
                        screen_y=node_screen_y,
                        bbox=full_bbox,
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=in_air,
                    )
                )

            return detections

        except Exception as exc:
            logger.error("Error in _scan_3d_gather_crests: %s", exc, exc_info=True)
            return detections

    def _is_monster_nameplate(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
        scale_x: float,
        scale_y: float,
    ) -> bool:
        """Check whether a 3D world nameplate belongs to a monster (e.g. Yellow-Maned Zaif, Highland Odyle Sparkie).

        Monsters exhibit:
        1. Yellow/orange/olive text between the diamond bracket tips (H=14..38, S=90..255, V=120..255)
           with little or no cyan/green text inside the inner text span, OR
        2. A red/orange square agro/rank badge ('4') and level number directly below the health bar.
        """
        h, w = hsv.shape[:2]

        # 1. Inspect inner text region (excluding left/right 18% where cyan diamond tips sit)
        pad_x = max(2, int(round(cw * 0.18)))
        ix1 = max(0, x + pad_x)
        ix2 = min(w, x + cw - pad_x)
        iy1 = max(0, y - 2)
        iy2 = min(h, y + ch + 4)

        if ix2 > ix1 and iy2 > iy1:
            inner_hsv = hsv[iy1:iy2, ix1:ix2]
            yellow_text_mask = cv2.inRange(inner_hsv, (14, 95, 125), (38, 255, 255))
            cyan_text_mask = cv2.inRange(
                inner_hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
            )
            yellow_px = int(np.count_nonzero(yellow_text_mask))
            cyan_inner_px = int(np.count_nonzero(cyan_text_mask))
            # If the label contains substantial cyan/green gatherable text, it is NOT a monster nameplate
            if cyan_inner_px >= 18 and yellow_px < cyan_inner_px * 2:
                return False
            if cyan_inner_px < 18 and yellow_px > max(18, cyan_inner_px * 2):
                return True

        # 2. Inspect region directly below the nameplate for a red/orange agro/rank square badge ('4')
        bx1 = max(0, x + cw // 4)
        bx2 = min(w, x + (cw * 3) // 4)
        by1 = min(h, y + ch)
        by2 = min(h, y + ch + max(24, int(round(42 * scale_y))))
        if bx2 > bx1 and by2 > by1:
            below_hsv = hsv[by1:by2, bx1:bx2]
            red1 = cv2.inRange(below_hsv, (0, 110, 80), (12, 255, 255))
            red2 = cv2.inRange(below_hsv, (168, 110, 80), (180, 255, 255))
            red_badge_px = int(np.count_nonzero(red1 | red2))
            if red_badge_px >= max(16, int(round(20 * scale_x * scale_y))):
                return True

        return False

    def scan_3d_world_nameplates(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect 3D world gatherable materials via winged gather crests and cyan/teal diamond-tipped nameplates.

        Strictly filters out monster nameplates (such as '‹ Yellow-Maned Zaif ›' and
        'Highland Odyle Sparkie').

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_world".
        """
        detections: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return detections

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # First run dedicated 3D white symmetrical winged gather crest + cyan/green label scanner (Aria, Mela)
            crest_detections = self._scan_3d_gather_crests(frame_bgr, hsv)
            detections.extend(crest_detections)

            h, w = frame_bgr.shape[:2]
            scale_x = w / float(self.REF_WIDTH)
            scale_y = h / float(self.REF_HEIGHT)

            mask_primary = cv2.inRange(
                hsv, self.CYAN_PRIMARY_LOWER, self.CYAN_PRIMARY_UPPER
            )
            mask_secondary = cv2.inRange(
                hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
            )
            cyan_mask = cv2.bitwise_or(mask_primary, mask_secondary)

            # Restrict to 3D viewport (exclude top HUD bar, bottom skill bar, side panels)
            top_cut = int(h * 0.10)
            bottom_cut = int(h * 0.80)
            left_cut = int(w * 0.08)
            right_cut = int(w * 0.82)

            cyan_mask[:top_cut, :] = 0
            cyan_mask[bottom_cut:, :] = 0
            cyan_mask[:, :left_cut] = 0
            cyan_mask[:, right_cut:] = 0

            kernel_w = max(8, int(round(14 * scale_x)))
            kernel_h = max(3, int(round(4 * scale_y)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
            closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_w = max(16, int(round(18 * scale_x)))
            max_w = max(90, int(round(145 * scale_x)))
            min_h = max(4, int(round(5 * scale_y)))
            max_h = max(20, int(round(30 * scale_y)))

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < min_w or cw > max_w or ch < min_h or ch > max_h:
                    continue

                # Skip if already covered by a 3D winged crest detection
                cx = x + cw // 2
                cy = y + ch // 2
                if any(
                    abs(cx - (d.bbox[0] + d.bbox[2] // 2)) <= max(28, cw)
                    and abs(cy - (d.bbox[1] + d.bbox[3] // 2)) <= max(36, ch * 3)
                    for d in detections
                ):
                    continue

                roi_mask = cyan_mask[y : y + ch, x : x + cw]
                cyan_px = int(np.count_nonzero(roi_mask))
                min_px = max(12, int(round(15 * scale_x * scale_y)))
                fill_ratio = cyan_px / float(max(1, cw * ch))

                # Text + diamond tips have moderate fill ratio, never a solid UI rectangle
                if cyan_px < min_px or fill_ratio > 0.65:
                    continue

                hue_values = hsv[y : y + ch, x : x + cw, 0][roi_mask > 0]
                if hue_values.size == 0:
                    continue
                mean_hue = float(np.mean(hue_values))

                # Filter out pure blue UI selection highlights (which have H >= 98)
                if mean_hue > 97.5:
                    continue

                # Reject monster nameplates (e.g. ‹ Yellow-Maned Zaif › with yellow text / red agro badge)
                if self._is_monster_nameplate(hsv, x, y, cw, ch, scale_x, scale_y):
                    continue

                # Verify the bright white/silver crest icon directly above the nameplate
                crest_y1 = max(0, y - int(round(38 * scale_y)))
                crest_y2 = max(0, y - int(round(2 * scale_y)))
                crest_half_w = max(14, int(round(22 * scale_x)))
                crest_x1 = max(0, cx - crest_half_w)
                crest_x2 = min(w, cx + crest_half_w)

                if crest_y2 <= crest_y1 or crest_x2 <= crest_x1:
                    continue

                crest_hsv = hsv[crest_y1:crest_y2, crest_x1:crest_x2]
                crest_mask = cv2.inRange(
                    crest_hsv, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER
                )
                white_px = int(np.count_nonzero(crest_mask))
                min_white_px = max(10, int(round(14 * scale_x * scale_y)))
                max_white_px = max(140, int(round(280 * scale_x * scale_y)))

                if not (min_white_px <= white_px <= max_white_px):
                    continue

                # Check horizontal symmetry of the white crest above the nameplate
                flipped_crest = np.fliplr(crest_mask)
                c_inter = int(np.count_nonzero(crest_mask & flipped_crest))
                c_union = int(np.count_nonzero(crest_mask | flipped_crest))
                if (c_inter / float(max(1, c_union))) < 0.25:
                    continue

                # Check if diamond bracket tips (< Orichalcum >) are illuminated (active target)
                has_diamond_tips = cw >= int(round(80 * scale_x)) and cyan_px >= int(
                    round(140 * scale_x * scale_y)
                )

                material_type, in_air = self._classify_3d_material(
                    frame_bgr, hsv, x, y, cw, ch
                )

                # v20 (MADDE 8): boş materyal adı = mesh learned-negative
                # reddi (canavar/oyuncu) — tespit üretilmez.
                if not material_type:
                    continue

                # Target node position on the ground right below the nameplate
                node_screen_x = int(cx)
                node_screen_y = int(min(h - 1, y + ch + int(round(28 * scale_y))))
                full_bbox = (
                    int(min(x, crest_x1)),
                    int(crest_y1),
                    int(max(cw, crest_x2 - crest_x1)),
                    int((y + ch) - crest_y1),
                )

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(
                    node_screen_x, node_screen_y, w, h
                )

                base_conf = 0.88 if has_diamond_tips else 0.80
                crest_bonus = min(0.08, (white_px / float(max_white_px)) * 0.10)
                cyan_bonus = min(0.04, (cyan_px / 250.0) * 0.04)
                confidence = round(min(0.99, base_conf + crest_bonus + cyan_bonus), 3)

                detections.append(
                    DetectedMaterial(
                        material_type=material_type,
                        source="3d_world",
                        screen_x=node_screen_x,
                        screen_y=node_screen_y,
                        bbox=full_bbox,
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=confidence,
                        in_air=in_air,
                    )
                )

            return detections

        except Exception as exc:
            logger.error("Error in scan_3d_world_nameplates: %s", exc, exc_info=True)
            return detections

    def _scan_floating_gather_boxes(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect floating square [E] / [F] gatherable material boxes (36x36 to 44x44 px) in the 3D world.

        Detects the square metallic/olive border frame with strong 4-side Sobel edges,
        top-left white [E]/[F] keybind badge on a dark key cap, and glowing resource graphic inside.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_floating_icon" and material_type="gatherable_node".
        """
        boxes: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return boxes

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            if w < 600 or h < 360:
                return boxes

            x1 = int(w * 0.08)
            y1 = int(h * 0.12)
            x2 = int(w * 0.82)
            y2 = int(h * 0.78)

            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                return boxes

            v_chan = roi_hsv[:, :, 2]
            s_chan = roi_hsv[:, :, 1]

            # 1. E/Q icons on screen are combat combo skill prompts, not world gather nodes.
            # Avoid confusing combat combo buttons with gathering targets.
            raw_candidates: list[tuple[float, int, int, int, int]] = []

            # 2. Structural + HSV + Sobel detection for the square metallic/olive frame + top-left E/F badge
            gx = np.abs(cv2.Sobel(v_chan, cv2.CV_32F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(v_chan, cv2.CV_32F, 0, 1, ksize=3))

            # Top-left keybind letter (E or F) has bright white pixels (V>=175, S<=48) adjacent to dark key cap (V<=52)
            key_white = ((v_chan >= 175) & (s_chan <= 48)).astype(np.uint8) * 255
            key_dark = (v_chan <= 52).astype(np.uint8) * 255
            badge_seed = cv2.bitwise_and(
                cv2.dilate(key_white, np.ones((3, 3), np.uint8)),
                cv2.dilate(key_dark, np.ones((3, 3), np.uint8)),
            )
            contours, _ = cv2.findContours(
                badge_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            scaled_sz = max(28, min(46, int(round(36 * (w / 2000.0)))))
            candidate_sizes = sorted({scaled_sz, 36})

            for cnt in contours:
                kx, ky, kw, kh = cv2.boundingRect(cnt)
                if not (2 <= kw <= 16 and 3 <= kh <= 16):
                    continue

                best_match: Optional[tuple[float, int, int, int, int]] = None
                for box_sz in candidate_sizes:
                    for dy in range(-8, -1):
                        for dx in range(-8, -1):
                            bx = kx + dx
                            by = ky + dy
                            bw = box_sz
                            bh = box_sz
                            if (
                                bx < 0
                                or by < 0
                                or bx + bw >= roi_hsv.shape[1]
                                or by + bh >= roi_hsv.shape[0]
                            ):
                                continue

                            top_gy = float(np.mean(gy[by : by + 3, bx : bx + bw]))
                            bot_gy = float(
                                np.mean(gy[by + bh - 3 : by + bh, bx : bx + bw])
                            )
                            left_gx = float(np.mean(gx[by : by + bh, bx : bx + 3]))
                            right_gx = float(
                                np.mean(gx[by : by + bh, bx + bw - 3 : bx + bw])
                            )
                            min_grad = min(top_gy, bot_gy, left_gx, right_gx)
                            if min_grad < 120.0:
                                continue

                            b_top = roi_hsv[by : by + 3, bx : bx + bw]
                            b_bot = roi_hsv[by + bh - 3 : by + bh, bx : bx + bw]
                            b_left = roi_hsv[by : by + bh, bx : bx + 3]
                            b_right = roi_hsv[by : by + bh, bx + bw - 3 : bx + bw]

                            o_top = float(
                                np.mean(
                                    (b_top[:, :, 0] >= 35)
                                    & (b_top[:, :, 0] <= 68)
                                    & (b_top[:, :, 1] >= 75)
                                    & (b_top[:, :, 2] >= 85)
                                )
                            )
                            o_bot = float(
                                np.mean(
                                    (b_bot[:, :, 0] >= 35)
                                    & (b_bot[:, :, 0] <= 68)
                                    & (b_bot[:, :, 1] >= 75)
                                    & (b_bot[:, :, 2] >= 85)
                                )
                            )
                            o_left = float(
                                np.mean(
                                    (b_left[:, :, 0] >= 35)
                                    & (b_left[:, :, 0] <= 68)
                                    & (b_left[:, :, 1] >= 75)
                                    & (b_left[:, :, 2] >= 85)
                                )
                            )
                            o_right = float(
                                np.mean(
                                    (b_right[:, :, 0] >= 35)
                                    & (b_right[:, :, 0] <= 68)
                                    & (b_right[:, :, 1] >= 75)
                                    & (b_right[:, :, 2] >= 85)
                                )
                            )
                            o_min = min(o_top, o_bot, o_left, o_right)
                            if o_min < 0.65:
                                continue

                            # Verify interior glowing graphic has high brightness contrast
                            inner_v = v_chan[
                                by + 4 : by + bh - 4, bx + 4 : bx + bw - 4
                            ]
                            if inner_v.size == 0 or float(np.std(inner_v)) < 32.0:
                                continue

                            conf = min(
                                0.96,
                                0.82
                                + min(0.08, (min_grad - 120.0) / 1200.0)
                                + min(0.06, (o_min - 0.65) * 0.25),
                            )
                            if best_match is None or conf > best_match[0]:
                                best_match = (
                                    conf,
                                    int(x1 + bx),
                                    int(y1 + by),
                                    int(bw),
                                    int(bh),
                                )

                if best_match is not None:
                    raw_candidates.append(best_match)

            raw_candidates.sort(key=lambda item: item[0], reverse=True)
            for conf, abs_x, abs_y, bw, bh in raw_candidates:
                cx = abs_x + bw // 2
                cy = abs_y + bh // 2
                if any(
                    abs(cx - d.screen_x) <= max(24, bw)
                    and abs(cy - d.screen_y) <= max(24, bh)
                    for d in boxes
                ):
                    continue

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(cx, cy, w, h)
                boxes.append(
                    DetectedMaterial(
                        material_type="gatherable_node",
                        source="3d_floating_icon",
                        screen_x=int(cx),
                        screen_y=int(cy),
                        bbox=(int(abs_x), int(abs_y), int(bw), int(bh)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=round(float(conf), 3),
                        in_air=False,
                    )
                )

            return boxes

        except Exception as exc:
            logger.error("Error in _scan_floating_gather_boxes: %s", exc, exc_info=True)
            return boxes

    def scan_gather_prompt(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
        world_detections: Optional[list[DetectedMaterial]] = None,
    ) -> list[DetectedMaterial]:
        """Detect the interactive 'F Gather' button prompt near the character center.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.
            world_detections: Optional 3D world detections to link material type.

        Returns:
            List of DetectedMaterial instances with source="gather_prompt".
        """
        prompts: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return prompts

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            scale_x = w / float(self.REF_WIDTH)
            scale_y = h / float(self.REF_HEIGHT)

            # "F Gather" appears slightly right/below character center
            rx1 = int(w * 0.46)
            ry1 = int(h * 0.50)
            rx2 = int(w * 0.66)
            ry2 = int(h * 0.66)

            roi_hsv = hsv[ry1:ry2, rx1:rx2]
            if roi_hsv.size == 0:
                return prompts

            # Support both full brightness ("F Gather" active) and semi-faded ("F Gather" transition)
            white_mask = cv2.inRange(roi_hsv, (0, 0, 140), (180, 48, 255))
            kw = max(10, int(round(18 * scale_x)))
            kh = max(4, int(round(6 * scale_y)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_w = max(42, int(round(56 * scale_x)))
            max_w = max(70, int(round(98 * scale_x)))
            min_h = max(8, int(round(11 * scale_y)))
            max_h = max(20, int(round(30 * scale_y)))
            icon_split = max(16, int(round(24 * scale_x)))

            for cnt in contours:
                fx, fy, fcw, fch = cv2.boundingRect(cnt)
                if fcw < min_w or fcw > max_w or fch < min_h or fch > max_h:
                    continue

                sub_mask = white_mask[fy : fy + fch, fx : fx + fcw]
                total_white = int(np.count_nonzero(sub_mask))
                left_icon_white = int(np.count_nonzero(sub_mask[:, :icon_split]))
                right_text_white = int(np.count_nonzero(sub_mask[:, icon_split:]))

                if left_icon_white < int(round(28 * scale_x * scale_y)):
                    continue
                if right_text_white < int(round(28 * scale_x * scale_y)):
                    continue

                # Verify dark translucent pill background behind "Gather" text
                pill_v = roi_hsv[fy : fy + fch, fx + icon_split : fx + fcw, 2]
                dark_pixels = pill_v[pill_v < 135]
                if dark_pixels.size == 0:
                    continue
                dark_mean = float(np.mean(dark_pixels))
                if not (20.0 <= dark_mean <= 78.0):
                    continue

                abs_x = rx1 + fx
                abs_y = ry1 + fy
                cx = abs_x + fcw // 2
                cy = abs_y + fch // 2

                # Inherit material_type only from a NEARBY 3D nameplate.
                # Without a close neighbour the type is Unknown (never a
                # default Orichalcum Ore inherited from an arbitrarily far node).
                mat_type = "Unknown"
                in_air = False
                if world_detections:
                    nearest = min(
                        world_detections,
                        key=lambda d: math.hypot(d.screen_x - cx, d.screen_y - cy),
                    )
                    nearest_dist = math.hypot(nearest.screen_x - cx, nearest.screen_y - cy)
                    max_inherit_px = 200.0 * (w / float(self.REF_WIDTH))
                    if nearest_dist <= max_inherit_px:
                        mat_type = nearest.material_type
                        in_air = nearest.in_air

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(cx, cy, w, h)
                # When "F Gather" is visible, the player is within interaction range (~1.5m - 2.5m)
                dist_m = round(min(dist_m, 2.2), 2)

                # Confidence from the RELATIVE white fraction of the prompt
                # box, not from an absolute pixel count (which grows with box size).
                white_ratio = total_white / max(1, fcw * fch)
                conf = 0.94 if white_ratio >= 0.20 else 0.86
                prompts.append(
                    DetectedMaterial(
                        material_type=mat_type,
                        source="gather_prompt",
                        screen_x=int(cx),
                        screen_y=int(cy),
                        bbox=(int(abs_x), int(abs_y), int(fcw), int(fch)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=in_air,
                    )
                )

            return prompts

        except Exception as exc:
            logger.error("Error in scan_gather_prompt: %s", exc, exc_info=True)
            return prompts

    def scan_minimap_materials(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Scan the top-right minimap (x: 0.855..0.992, y: 0.065..0.255) for white 3-leaf sprout icons.

        Strictly detects ONLY the white 3-leaf sprout icon ('Gatherable Material') and
        rejects plain circular white monster dots ('•'), white feathers, blue wings,
        player arrows, and minimap UI text/chrome.

        Args:
            frame_bgr: BGR game frame (full screen or minimap closeup crop).
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="minimap".
        """
        nodes: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return nodes

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]

            # Support both full-screen frames (w >= 1000) and minimap closeup crops (w < 1000)
            if w >= 1000:
                rx1, ry1, rx2, ry2 = self.get_minimap_roi()
                mx1 = int(round(w * rx1))
                my1 = int(round(h * ry1))
                mx2 = int(round(w * rx2))
                my2 = int(round(h * ry2))
                templates = [self._sprout_12_tpl]
                if abs(w - 2000) > 150:
                    s_sz = max(9, min(16, int(round(12 * (w / 2000.0)))))
                    if s_sz != 12:
                        templates.append(
                            cv2.resize(
                                self._sprout_12_tpl,
                                (s_sz, s_sz),
                                interpolation=cv2.INTER_AREA,
                            )
                        )
            elif w > 500:
                # ROI as fractions of the frame (old fixed pixels assumed
                # 1920x1080); scales to any resolution.
                mx1, my1 = int(round(w * 0.187)), int(round(h * 0.100))
                mx2, my2 = min(w, int(round(w * 0.359))), min(h, int(round(h * 0.305)))
                templates = [self._sprout_17_tpl]
            else:
                # Small closeup crop: fractions of the frame (old fixed pixels
                # assumed ~960x540).
                mx1, my1 = int(round(w * 0.008)), int(round(h * 0.126))
                mx2, my2 = min(w, int(round(w * 0.353))), min(h, int(round(h * 0.535)))
                templates = [self._sprout_17_tpl]

            if self._disk_sprout_tpl is not None:
                templates.append(self._disk_sprout_tpl)

            mm_hsv = hsv[my1:my2, mx1:mx2]
            if mm_hsv.size == 0:
                return nodes

            # Ensure minimap is actually visible and not occluded by a dark menu/overlay
            mean_v = float(np.mean(mm_hsv[:, :, 2]))
            mean_s = float(np.mean(mm_hsv[:, :, 1]))
            if mean_v < 70.0 or mean_s < 18.0:
                return nodes

            mm_h, mm_w = mm_hsv.shape[:2]
            center_x = mm_w / 2.0
            center_y = mm_h / 2.0
            mm_v = mm_hsv[:, :, 2]

            # Valid radar canvas mask (exclude top-right zoom glass, bottom-right clock/1km, and center player arrow)
            valid_mask = np.ones((mm_h, mm_w), dtype=bool)
            valid_mask[: int(mm_h * 0.18), int(mm_w * 0.80) :] = False
            valid_mask[int(mm_h * 0.78) :, int(mm_w * 0.65) :] = False

            raw_matches: list[tuple[float, int, int, int, int]] = []
            for tpl in templates:
                th, tw = tpl.shape[:2]
                if tw >= mm_w or th >= mm_h:
                    continue
                res = cv2.matchTemplate(mm_v, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= 0.68)
                for y_pt, x_pt in zip(ys, xs):
                    raw_matches.append(
                        (float(res[y_pt, x_pt]), int(x_pt), int(y_pt), int(tw), int(th))
                    )

            raw_matches.sort(key=lambda item: item[0], reverse=True)
            for score, nx, ny, nw, nh in raw_matches:
                node_cx = nx + nw / 2.0
                node_cy = ny + nh / 2.0

                if not valid_mask[min(mm_h - 1, int(node_cy)), min(mm_w - 1, int(node_cx))]:
                    continue

                # Exclude center player arrow
                if (
                    abs(node_cx - center_x) < mm_w * 0.06
                    and abs(node_cy - center_y) < mm_h * 0.06
                ):
                    continue

                # Avoid duplicate detections of the same sprout icon
                abs_cx = int(round(mx1 + node_cx))
                abs_cy = int(round(my1 + node_cy))
                if any(
                    abs(abs_cx - d.screen_x) <= max(10, nw)
                    and abs(abs_cy - d.screen_y) <= max(10, nh)
                    for d in nodes
                ):
                    continue

                # Verify sprout is white/silver (low saturation on bright pixels) with dark border contrast
                sub_hsv = mm_hsv[ny : ny + nh, nx : nx + nw]
                sub_v = sub_hsv[:, :, 2]
                sub_s = sub_hsv[:, :, 1]
                white_bright_mask = (sub_v > 175) & (sub_s < 58)
                bright_px = int(np.count_nonzero(white_bright_mask))
                if bright_px < 6:
                    continue

                contrast = float(np.max(sub_v)) - float(np.min(sub_v))
                if contrast < 85.0:
                    continue

                # Reject plain circular white monster dots ('•'): monster dots have top width ≈ bottom width
                # and no central vertical vein split, whereas the 3-leaf sprout widens at the bottom
                top_half_bright = int(np.count_nonzero(white_bright_mask[: nh // 2, :]))
                bot_half_bright = int(np.count_nonzero(white_bright_mask[nh // 2 :, :]))
                if bot_half_bright < top_half_bright:
                    continue

                rel_dx = round(float(node_cx - center_x), 2)
                rel_dy = round(float(node_cy - center_y), 2)
                dist_m = round(
                    math.hypot(rel_dx, rel_dy) * self.meters_per_map_pixel, 2
                )
                conf = round(min(0.96, 0.78 + 0.18 * score), 3)

                nodes.append(
                    DetectedMaterial(
                        material_type="Gatherable Material",
                        source="minimap",
                        screen_x=abs_cx,
                        screen_y=abs_cy,
                        bbox=(int(mx1 + nx), int(my1 + ny), int(nw), int(nh)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=False,
                    )
                )

            return nodes

        except Exception as exc:
            logger.error("Error in scan_minimap_materials: %s", exc, exc_info=True)
            return nodes

    def scan_minimap_nodes(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Backward-compatible alias for scan_minimap_materials."""
        return self.scan_minimap_materials(frame_bgr, hsv)

    def _project_3d_to_map_offset(
        self, screen_x: int, screen_y: int, width: int, height: int
    ) -> tuple[float, float, float]:
        """Convert 3D screen coordinates into relative map pixel offsets (dx, dy) and distance (m).

        Uses the player anchor at (W * 0.50, H * 0.54) and applies perspective foreshortening
        so objects higher up toward the horizon correspond to larger map distances.
        """
        player_x = width * self.PLAYER_ANCHOR_X_RATIO
        player_y = height * self.PLAYER_ANCHOR_Y_RATIO

        raw_dx = float(screen_x) - player_x
        raw_dy = float(screen_y) - player_y

        # Normalize to reference resolution (1280x720)
        norm_dx = raw_dx * (self.REF_WIDTH / float(max(1, width)))
        norm_dy = raw_dy * (self.REF_HEIGHT / float(max(1, height)))

        # Perspective depth multiplier increases as screen_y moves toward the upper horizon
        vertical_ratio = max(0.25, min(1.8, (player_y / float(max(1, screen_y))) ** 1.15))

        rel_map_dx = round(norm_dx * self.perspective_scale_x * vertical_ratio, 2)
        rel_map_dy = round(norm_dy * self.perspective_scale_y * vertical_ratio, 2)

        map_pixel_dist = math.hypot(rel_map_dx, rel_map_dy)
        estimated_distance_m = round(
            max(0.5, map_pixel_dist * self.meters_per_map_pixel), 2
        )
        return rel_map_dx, rel_map_dy, estimated_distance_m

    def discriminate_material(
        self,
        patch_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
        nameplate_width_px: int = 0,
        has_specular_crest: Optional[bool] = None,
        crest_patch_bgr: Optional[np.ndarray] = None,
        exemplar_memory: Optional[Any] = None,
        precomputed_exemplar: Optional[PrecomputedExemplarVerdict] = None,
    ) -> MaterialDiscriminationResult:
        """Expose multi-layered 3D material discrimination through MaterialScanner."""
        return discriminate_3d_material(
            patch_bgr,
            hsv=hsv,
            nameplate_width_px=nameplate_width_px,
            has_specular_crest=has_specular_crest,
            crest_patch_bgr=crest_patch_bgr,
            exemplar_memory=exemplar_memory,
            ref_scale=1.0,
            precomputed_exemplar=precomputed_exemplar,
        )

    def _classify_3d_material(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
    ) -> tuple[str, bool]:
        """Classify the 3D resource type across Aion 2's 6 Essence Extraction categories from visual context.

        Categories supported:
        - Herbs: Aria (blue/white petals), Herb Edana (purple/pink petals), Herb Lumina (golden/yellow glow), Targena / Calendula
        - Cooking & Flora: Mela (red-berry/flower bush with green leaves)
        - Gemstones: Blood Crystal (crimson red crystal), Sapphire Gemstone (blue crystal), Ruby Gemstone, Diamond Gemstone
        - Ores: Orichalcum Ore (copper/orange-brown rock vein), Obsidian Ore (dark violet/black glassy rock vein)
        - Logs & Trees: Yggdrasil Log / Asvata Log (woody brown tree trunk)
        - Odyle: Odyle (3D world energy orb, never minimap feathers)
        """
        h, w = frame_bgr.shape[:2]
        in_air = bool(y < int(h * 0.22))

        # Inspect the 3D mesh region directly below the nameplate for crystal/plant/ore/gem/log color signatures
        mesh_y1 = min(h - 1, y + ch + 4)
        mesh_y2 = min(h, y + ch + int(round(75 * (h / float(self.REF_HEIGHT)))))
        mesh_x1 = max(0, x - 20)
        mesh_x2 = min(w, x + cw + 20)

        if mesh_y2 > mesh_y1 and mesh_x2 > mesh_x1:
            mesh_crop = frame_bgr[mesh_y1:mesh_y2, mesh_x1:mesh_x2]
            mesh_hsv = hsv[mesh_y1:mesh_y2, mesh_x1:mesh_x2]

            # Primary: multi-layer discrimination (Color Histogram + Edge Density + Specular + Nameplate Width)
            # has_specular_crest is deliberately NOT passed as True here:
            # asserting it made compute_specular_crest_score return 0.95
            # without measuring anything. Let it measure the mesh crop.
            disc_res = discriminate_3d_material(
                mesh_crop,
                hsv=mesh_hsv,
                nameplate_width_px=cw,
                ref_scale=w / float(self.REF_WIDTH),
            )
            if disc_res.is_authentic_node and disc_res.material_type:
                return disc_res.material_type, in_air

            # v20 (MADDE 8 — Targena yanlış algı): learned-negative kararı
            # kesindir. İsim etiketinin altındaki mesh, öğrenilmiş negatif
            # örneğe uydu (monster_3d_negative / player_character,
            # sim >= 0.82 — discriminate_3d_material adım 1). Aşağıdaki
            # saf-renk yedek kurallar bu yamayı yeniden etiketleyemez:
            # bir mob gövdesi sarı+yeşil Targena kuralını (kural 4)
            # kolayca geçer ve mob "Targena" diye haritalanırdı.
            # Boş materyal adı = reddedildi; çağıranlar tespiti atlar.
            if (disc_res.rejection_reason or "").startswith("learned_negative:"):
                logger.info(
                    "MaterialScanner _classify_3d_material: mesh %s reddi, renk yedeği yok",
                    disc_res.rejection_reason,
                )
                return "", in_air

            # 1. Purple/violet crystal foliage signature for Aria
            aria_purple_mask = cv2.inRange(mesh_hsv, (125, 65, 70), (164, 255, 255))
            aria_purple_px = int(np.count_nonzero(aria_purple_mask))
            if aria_purple_px >= 120:
                return "Aria", in_air

            # 2. Red/coral berry/flower + green leaf bush signature for Mela
            mela_red_mask = cv2.inRange(
                mesh_hsv, (0, 110, 90), (10, 255, 255)
            ) | cv2.inRange(mesh_hsv, (168, 110, 90), (180, 255, 255))
            mela_green_mask = cv2.inRange(mesh_hsv, (36, 65, 50), (75, 255, 220))
            red_px = int(np.count_nonzero(mela_red_mask))
            green_px = int(np.count_nonzero(mela_green_mask))
            if red_px >= 120 and green_px >= 40:
                return "Mela", in_air

            # 3. Gemstones: Sapphire (blue crystal), Ruby (red crystal on rock), Diamond (bright white crystal)
            sapphire_mask = cv2.inRange(mesh_hsv, (105, 130, 130), (130, 255, 255))
            ruby_mask = cv2.inRange(mesh_hsv, (165, 120, 120), (180, 255, 255))
            diamond_mask = cv2.inRange(mesh_hsv, (0, 0, 215), (180, 35, 255))
            sapphire_px = int(np.count_nonzero(sapphire_mask))
            ruby_px = int(np.count_nonzero(ruby_mask))
            diamond_px = int(np.count_nonzero(diamond_mask))

            if sapphire_px >= 25:
                return "Sapphire Gemstone", in_air
            if ruby_px >= 25 and green_px < 20:
                return "Ruby Gemstone", in_air
            if diamond_px >= 65 and green_px < 20:
                return "Diamond Gemstone", in_air

            # 4. Golden-yellow herb with green foliage (Targena / Calendula)
            yellow_herb_mask = cv2.inRange(mesh_hsv, (20, 120, 130), (34, 255, 255))
            if int(np.count_nonzero(yellow_herb_mask)) >= 80 and green_px >= 40:
                return "Targena", in_air

            # 5. Warm copper/amber/orange glow on dark rock for Orichalcum Ore veins
            warm_ore_mask = cv2.inRange(mesh_hsv, (5, 70, 60), (26, 255, 255))
            dark_rock_mask = cv2.inRange(mesh_hsv, (0, 0, 20), (180, 65, 95))
            warm_px = int(np.count_nonzero(warm_ore_mask))
            rock_px = int(np.count_nonzero(dark_rock_mask))
            if warm_px >= 12 and rock_px >= 30:
                return "Orichalcum Ore", in_air

            # 6. Woody brown trunk signature for Yggdrasil Log / Asvata Log
            wood_brown_mask = cv2.inRange(mesh_hsv, (10, 65, 45), (22, 180, 145))
            if int(np.count_nonzero(wood_brown_mask)) >= 140 and rock_px < 25:
                return "Yggdrasil Log", in_air

            if warm_px >= 12:
                return "Orichalcum Ore", in_air

        if in_air:
            return "Odyle", True

        # No colour/geometry evidence matched any material: report Unknown
        # instead of fabricating Orichalcum Ore.
        return "Unknown", in_air

    def _is_fullscreen_modal_open(
        self, frame_bgr: np.ndarray, hsv: np.ndarray
    ) -> bool:
        """Detect whether a large flat dark UI window (e.g. Gatherion settings/import) covers the screen."""
        h, w = frame_bgr.shape[:2]
        # Check central viewport area where settings dialogs sit
        cx1, cy1 = int(w * 0.15), int(h * 0.12)
        cx2, cy2 = int(w * 0.85), int(h * 0.85)
        center_hsv = hsv[cy1:cy2, cx1:cx2]

        # Flat dark slate UI background has very low saturation (S < 35) and narrow value band (18 <= V <= 48)
        slate_mask = cv2.inRange(center_hsv, (0, 0, 18), (180, 35, 48))
        slate_ratio = float(np.count_nonzero(slate_mask)) / float(
            max(1, center_hsv.shape[0] * center_hsv.shape[1])
        )
        # A merely dark scene (cave / ore region) can exceed 0.52 slate
        # coverage; only a near-total slate centre is a fullscreen modal.
        return slate_ratio > 0.80

    def get_material_scanner_ready_flag(self) -> dict:
        # S-0301: bound to the real contract validator, not a constant True.
        return {"module": "material_scanner", "ready": bool(validate_material_scanner_contract())}

    def get_material_scanner_version_info(self) -> dict:
        return {"module": "material_scanner", "version": __version__, "ready": bool(validate_material_scanner_contract())}

    def get_material_scanner_diagnostic_snapshot(self) -> dict:
        _ok = bool(validate_material_scanner_contract())
        return {"module": "material_scanner", "status": "ok" if _ok else "degraded", "ready": _ok}


def get_material_scanner_capabilities() -> list:
    return ["material_scanner", "resource_detection", "diagnostic_snapshot"]


def validate_material_scanner_contract() -> bool:
    """Real contract checks (not a tautology): taxonomy ids unique, every
    main material has a canonical name and a nameplate range, the public
    discrimination/classification entry points exist, and capabilities are
    non-empty. Returns False if any check fails.
    """
    try:
        class_ids = [info["class_id"] for info in MAIN_MATERIALS_6.values()]
        if len(set(class_ids)) != len(class_ids):
            return False
        for key, info in MAIN_MATERIALS_6.items():
            if not key or not info.get("canonical_name"):
                return False
            lo, hi = info["nameplate_width_range"]
            if not (0 < lo <= hi):
                return False
            lo_ar, hi_ar = info["typical_aspect_ratio"]
            if not (0 < lo_ar <= hi_ar):
                return False
        if not callable(discriminate_3d_material):
            return False
        if not callable(compute_nameplate_width_score):
            return False
        if not callable(compute_specular_crest_score):
            return False
        if not get_material_scanner_capabilities():
            return False
        return True
    except Exception:
        return False


def get_material_scanner_baseline_metrics() -> dict:
    return {"module": "material_scanner", "contract_valid": validate_material_scanner_contract(), "capability_count": len(get_material_scanner_capabilities())}


def get_material_scanner_readiness_status() -> dict:
    return {"module": "material_scanner", "ready": validate_material_scanner_contract(), "metrics": get_material_scanner_baseline_metrics()}


def get_material_scanner_audit_record() -> dict:
    return {"module": "material_scanner", "audit_passed": validate_material_scanner_contract(), "readiness": get_material_scanner_readiness_status()}


def get_material_scanner_operational_digest() -> dict:
    return {"module": "material_scanner", "operational": validate_material_scanner_contract(), "audit": get_material_scanner_audit_record()}


def get_material_scanner_telemetry_payload() -> dict:
    return {"module": "material_scanner", "payload": get_material_scanner_operational_digest()}


def get_material_scanner_integrity_seal() -> dict:
    return {"module": "material_scanner", "sealed": validate_material_scanner_contract(), "telemetry": get_material_scanner_telemetry_payload()}


def get_material_scanner_compliance_certificate() -> dict:
    return {"module": "material_scanner", "certified": validate_material_scanner_contract(), "seal": get_material_scanner_integrity_seal()}


def get_material_scanner_readiness_probe() -> dict:
    return {"module": "material_scanner", "ready": validate_material_scanner_contract(), "certificate": get_material_scanner_compliance_certificate()}


def get_material_scanner_audit_summary() -> dict:
    return {"module": "material_scanner", "audit_passed": validate_material_scanner_contract(), "probe": get_material_scanner_readiness_probe()}

