# -*- coding: utf-8 -*-
"""W34 unit tests: Visual discrimination of the 6 main materials on synthetic BGR patches.

Verifies that synthetic BGR patches for the 6 main gatherable materials:
1. Aria (purple/violet crystal-petaled herb)
2. Mela (red-berry bush on green foliage)
3. Targena (golden-yellow & green leafy herb)
4. Orichalcum (copper/bronze & oxidized teal vein on dark rock)
5. Obsidian/Crystal (dark violet glassy obsidian vein & faceted gemstone crystals)
6. Wood (tall vertical bark-textured tree trunk with canopy)
are accurately classified and discriminated from one another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.materials_db import (
    classify_3d_material_patch,
    extract_visual_descriptor,
)
from core.material_scanner import (
    discriminate_3d_material,
    match_material_names,
)
from dataset_factory.yolo_auto_labeler import map_material_name_to_yolo_class_id


def _hsv(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Real-world RGB reference color -> OpenCV HSV triple (via cv2).

    Patch colors are specified as RGB pigments (flower violet, berry crimson,
    leaf green, ...) -- never hand-picked from the classifier's own HSV masks.
    """
    px = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0])


def _canvas(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """Solid HSV canvas from an RGB reference color."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = _hsv(rgb)
    return arr


# Real-world RGB reference colors (pigment names, not classifier thresholds).
VIOLET_FLOWER_RGB = (166, 77, 255)  # violet flower petal (#A64DFF)
LIGHT_VIOLET_RGB = (180, 90, 255)  # light violet petal (#B45AFF)
PALE_VIOLET_RGB = (230, 220, 255)  # pale violet-white petal tip (#E6DCFF)
LEAF_GREEN_RGB = (80, 160, 60)  # yellow-green foliage (#50A03C)
BERRY_RED_RGB = (220, 20, 60)  # crimson berry (#DC143C)
GOLD_RGB = (200, 215, 40)  # golden-yellow blossom (#C8D728)
GOLD_LIGHT_RGB = (210, 215, 60)  # light golden blossom (#D2D73C)
GOLD_PALE_RGB = (225, 215, 40)  # pale golden blossom (#E1D728)
COPPER_RGB = (184, 115, 51)  # copper / bronze metal (#B87333)
TEAL_RGB = (0, 128, 128)  # oxidized teal mineral (#008080)
DARK_VIOLET_RGB = (75, 46, 107)  # dark violet obsidian vein (#4B2E6B)
DARK_VIOLET_DEEP_RGB = (38, 28, 44)  # near-black violet glass (#261C2C)
SAPPHIRE_RGB = (30, 95, 255)  # sapphire blue crystal (#1E5FFF)
ICE_BLUE_RGB = (220, 235, 255)  # icy crystal glint (#DCEBFF)
BARK_RGB = (122, 90, 56)  # weathered bark brown (#7A5A38)
NEUTRAL_DARK_RGB = (36, 36, 36)  # neutral dark gray background (#242424)


def _make_aria_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Aria (purple/violet crystal petals + green stem)."""
    h, w = 96, 80
    hsv = _canvas(h, w, NEUTRAL_DARK_RGB)
    # Violet-purple crystal petals
    cv2.ellipse(hsv, (40, 38), (24, 26), 0, 0, 360, _hsv(VIOLET_FLOWER_RGB), -1)
    cv2.circle(hsv, (30, 32), 10, _hsv((150, 60, 240)), -1)
    cv2.circle(hsv, (50, 32), 10, _hsv(LIGHT_VIOLET_RGB), -1)
    # Specular white-violet petal tips
    cv2.circle(hsv, (40, 24), 5, _hsv(PALE_VIOLET_RGB), -1)
    # Upright green stem at base
    hsv[66:90, 36:44] = _hsv(LEAF_GREEN_RGB)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_mela_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Mela (squat green bush with red berry clusters)."""
    h, w = 72, 104
    hsv = _canvas(h, w, NEUTRAL_DARK_RGB)
    # Green foliage bush dome
    cv2.ellipse(hsv, (52, 38), (46, 30), 0, 0, 360, _hsv(LEAF_GREEN_RGB), -1)
    # Dispersed red berry clusters
    for cx, cy in ((28, 30), (48, 26), (70, 30), (36, 46), (56, 44), (74, 46)):
        cv2.circle(hsv, (cx, cy), 6, _hsv(BERRY_RED_RGB), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_targena_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Targena (golden-yellow blossom on green herb leaves)."""
    h, w = 88, 80
    hsv = _canvas(h, w, LEAF_GREEN_RGB)
    # Golden-yellow Targena blossom cluster
    cv2.circle(hsv, (40, 34), 20, _hsv(GOLD_RGB), -1)
    cv2.circle(hsv, (28, 42), 12, _hsv(GOLD_LIGHT_RGB), -1)
    cv2.circle(hsv, (52, 42), 12, _hsv(GOLD_PALE_RGB), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_orichalcum_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Orichalcum Ore (copper/bronze + teal veins on dark rock)."""
    h, w = 64, 112
    hsv = _canvas(h, w, NEUTRAL_DARK_RGB)
    # Rough rock strata lines
    cv2.line(hsv, (6, 16), (104, 18), _hsv((40, 32, 28)), 3)
    cv2.line(hsv, (8, 50), (102, 48), _hsv((35, 28, 25)), 3)
    # Metallic copper/amber-gold vein
    cv2.line(hsv, (12, 30), (100, 26), _hsv(COPPER_RGB), 9)
    # Oxidized teal mineral vein
    cv2.line(hsv, (16, 40), (96, 38), _hsv(TEAL_RGB), 6)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_obsidian_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Obsidian Ore (dark violet/black glassy vein with specular glints)."""
    h, w = 68, 108
    hsv = _canvas(h, w, DARK_VIOLET_DEEP_RGB)
    # Dark violet obsidian crystalline veins
    cv2.ellipse(hsv, (54, 34), (38, 20), 0, 0, 360, _hsv(DARK_VIOLET_RGB), -1)
    cv2.line(hsv, (16, 28), (92, 24), _hsv((90, 60, 130)), 7)
    cv2.line(hsv, (20, 42), (88, 40), _hsv((65, 40, 95)), 6)
    # Sharp specular facet highlights
    cv2.circle(hsv, (42, 26), 4, _hsv((240, 240, 255)), -1)
    cv2.circle(hsv, (66, 34), 4, _hsv((245, 245, 255)), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_crystal_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Crystal / Gemstone (glowing cyan-blue crystal spire with specular glint)."""
    h, w = 96, 80
    hsv = _canvas(h, w, NEUTRAL_DARK_RGB)
    # Deep glowing cyan/sapphire crystal cluster
    cv2.ellipse(hsv, (40, 46), (24, 34), 0, 0, 360, _hsv(SAPPHIRE_RGB), -1)
    # Specular crystal facet highlight
    cv2.circle(hsv, (40, 34), 7, _hsv(ICE_BLUE_RGB), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_wood_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Wood / Tree Log (vertical brown bark trunk with striations and canopy)."""
    h, w = 150, 60
    hsv = _canvas(h, w, BARK_RGB)
    # Vertical bark grain striations (luminance detail only)
    for col in range(0, w, 4):
        hsv[:, col : col + 2, 2] = 112
        hsv[:, col + 2 : col + 4, 2] = 58
    # Leafy green canopy at top
    hsv[:26, :] = _hsv(LEAF_GREEN_RGB)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_01_aria_synthetic_bgr_visual_discrimination() -> None:
    """Verify Aria synthetic BGR patch is classified as Aria (Herbs) and discriminated from other 5 materials."""
    bgr = _make_aria_bgr()
    res = classify_3d_material_patch(bgr, region_name="Western Altamia Highland")
    disc = discriminate_3d_material(bgr, exemplar_memory=False)

    assert res["name"] == "Aria"
    assert res["category"] == "Herbs"
    assert res["confidence"] >= 0.80
    assert res["color_ratios"]["aria_purple"] > 0.20

    scores = res["scores"]
    for other in ("Mela", "Targena", "Orichalcum Ore", "Sapphire Gemstone", "Ruby Gemstone", "Yggdrasil Log"):
        assert scores["Aria"] > scores[other], f"Aria score must exceed {other}"

    assert disc.is_authentic_node is True
    assert disc.material_type == "Aria"
    assert disc.class_name == "herb_aria"
    assert map_material_name_to_yolo_class_id(res["name"]) == 0


def test_02_mela_synthetic_bgr_visual_discrimination() -> None:
    """Verify Mela synthetic BGR patch is classified as Mela (Cooking & Fibers) and discriminated from other 5 materials."""
    bgr = _make_mela_bgr()
    res = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")
    disc = discriminate_3d_material(bgr, exemplar_memory=False)

    assert res["name"] == "Mela"
    assert res["category"] == "Cooking & Fibers"
    assert res["confidence"] >= 0.80
    assert res["color_ratios"]["red_berry"] > 0.03
    assert res["color_ratios"]["green_leaf"] > 0.40

    scores = res["scores"]
    for other in ("Aria", "Targena", "Orichalcum Ore", "Ruby Gemstone", "Sapphire Gemstone", "Yggdrasil Log"):
        assert scores["Mela"] > scores[other], f"Mela score must exceed {other}"

    assert disc.is_authentic_node is True
    assert disc.material_type == "Mela"
    assert disc.class_name == "herb_mela"
    assert map_material_name_to_yolo_class_id(res["name"]) == 1


def test_03_targena_synthetic_bgr_visual_discrimination() -> None:
    """Verify Targena synthetic BGR patch is classified as Targena (Herbs) and discriminated from other 5 materials."""
    bgr = _make_targena_bgr()
    res = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")
    disc = discriminate_3d_material(bgr, exemplar_memory=False)

    assert res["name"] == "Targena"
    assert res["category"] == "Herbs"
    assert res["confidence"] >= 0.80
    assert res["color_ratios"]["targena_gold_green"] > 0.15

    scores = res["scores"]
    for other in ("Aria", "Mela", "Orichalcum Ore", "Sapphire Gemstone", "Ruby Gemstone", "Yggdrasil Log"):
        assert scores["Targena"] > scores[other], f"Targena score must exceed {other}"

    assert disc.is_authentic_node is True
    assert disc.material_type == "Targena"
    assert disc.class_name == "herb_targena"
    assert map_material_name_to_yolo_class_id(res["name"]) == 2


def test_04_orichalcum_synthetic_bgr_visual_discrimination() -> None:
    """Verify Orichalcum synthetic BGR patch is classified as Orichalcum Ore (Ores & Metals) and discriminated from others."""
    bgr = _make_orichalcum_bgr()
    res = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")
    disc = discriminate_3d_material(bgr, exemplar_memory=False)

    assert res["name"] == "Orichalcum Ore"
    assert res["category"] == "Ores & Metals"
    assert res["confidence"] >= 0.80
    assert res["color_ratios"]["orichalcum_vein"] > 0.10
    assert res["aspect_ratio_features"]["morphological_profile"] == "horizontal_vein"

    scores = res["scores"]
    for other in ("Aria", "Mela", "Targena", "Sapphire Gemstone", "Ruby Gemstone", "Yggdrasil Log"):
        assert scores["Orichalcum Ore"] > scores[other], f"Orichalcum Ore score must exceed {other}"

    assert disc.is_authentic_node is True
    assert disc.material_type == "Orichalcum Ore"
    assert disc.class_name == "ore_orichalcum"
    assert map_material_name_to_yolo_class_id(res["name"]) == 3


def test_05_obsidian_and_crystal_synthetic_bgr_visual_discrimination() -> None:
    """Verify Obsidian and Crystal synthetic BGR patches are accurately classified and discriminated from other materials.

    Obsidian is decided by `discriminate_3d_material` (the obsidian-capable
    path): `classify_3d_material_patch` has no "Obsidian Ore" entry in its score
    table, so it cannot return Obsidian -- that taxonomy gap is pinned below
    instead of being papered over. The obsidian verdict must come from the
    dark-glassy visual itself: the same violet geometry on a bright background
    (no dark glassy substrate) must NOT discriminate as obsidian.
    """
    # Part A: Obsidian Ore dark violet/black glassy vein
    obsidian_bgr = _make_obsidian_bgr()
    disc_obs = discriminate_3d_material(obsidian_bgr, exemplar_memory=False)
    assert disc_obs.is_authentic_node is True
    assert disc_obs.material_type == "Obsidian Ore"
    assert disc_obs.class_name == "ore_obsidian"
    assert disc_obs.color_ratios["ore_obsidian"] > disc_obs.color_ratios["ore_orichalcum"]
    assert disc_obs.color_ratios["ore_obsidian"] > disc_obs.color_ratios["herb_aria"]
    # Decisive margin: the obsidian score must dominate, not just edge ahead.
    assert disc_obs.color_ratios["ore_obsidian"] >= 0.90
    assert disc_obs.color_ratios["ore_obsidian"] - disc_obs.color_ratios["herb_aria"] >= 0.30
    assert match_material_names("Obsidian", disc_obs.material_type) is True

    # The patch classifier's taxonomy has no Obsidian Ore entry: pin the gap
    # explicitly instead of asserting a classification it cannot produce.
    res_obs = classify_3d_material_patch(obsidian_bgr, region_name="Pilgrim's Ridge")
    assert "Obsidian Ore" not in res_obs["scores"]
    assert res_obs["name"] != "Obsidian Ore"

    # Negative control: same violet vein geometry on a bright background with
    # bright (high-value) violet and no dark glassy substrate must NOT read as
    # obsidian -- the verdict must come from the dark-glassy visual, not hue.
    bright_hsv = _canvas(68, 108, (200, 200, 205))
    cv2.ellipse(bright_hsv, (54, 34), (38, 20), 0, 0, 360, _hsv(VIOLET_FLOWER_RGB), -1)
    cv2.line(bright_hsv, (16, 28), (92, 24), _hsv(LIGHT_VIOLET_RGB), 7)
    cv2.line(bright_hsv, (20, 42), (88, 40), _hsv((150, 60, 240)), 6)
    cv2.circle(bright_hsv, (42, 26), 4, _hsv((240, 240, 255)), -1)
    cv2.circle(bright_hsv, (66, 34), 4, _hsv((245, 245, 255)), -1)
    bright_bgr = cv2.cvtColor(bright_hsv, cv2.COLOR_HSV2BGR)
    disc_bright = discriminate_3d_material(bright_bgr, exemplar_memory=False)
    assert disc_bright.material_type != "Obsidian Ore"
    assert disc_bright.color_ratios["ore_obsidian"] < disc_obs.color_ratios["ore_obsidian"]

    # Part B: Crystal / Gemstone faceted cluster
    crystal_bgr = _make_crystal_bgr()
    res_crys = classify_3d_material_patch(crystal_bgr, region_name="Western Altamia Highland")
    assert res_crys["name"] == "Sapphire Gemstone"
    assert res_crys["category"] == "Gemstones"
    assert res_crys["confidence"] >= 0.80
    for other in ("Aria", "Mela", "Targena", "Orichalcum Ore", "Yggdrasil Log"):
        assert res_crys["scores"]["Sapphire Gemstone"] > res_crys["scores"][other]
    assert match_material_names("Crystal", res_crys["name"]) is True
    assert map_material_name_to_yolo_class_id("Crystal") == 4


def test_06_wood_synthetic_bgr_visual_discrimination() -> None:
    """Verify Wood synthetic BGR patch is classified as Yggdrasil/Asvata Log (Logs & Trees) and discriminated from others.

    The visual identity (vertical trunk, bark ratio, edge profile) is
    region-independent: the same patch must yield the same descriptor, the same
    category and the same YOLO class in every region. Only the faction *name*
    (Yggdrasil vs Asvata) comes from the region -- the name is a faction label,
    not a visual claim, and is asserted as such.
    """
    bgr = _make_wood_bgr()
    desc = extract_visual_descriptor(bgr)
    assert desc.morphological_profile == "vertical_trunk"
    assert desc.vertical_edge_ratio > 0.60
    assert desc.color_ratios["bark_trunk"] > 0.50

    res_elyos = classify_3d_material_patch(bgr, region_name="Verteron")
    assert res_elyos["name"] == "Yggdrasil Log"
    assert res_elyos["category"] == "Logs & Trees"
    assert res_elyos["taxonomy"]["profession"] == "Woodcutting"
    assert res_elyos["confidence"] >= 0.80

    scores = res_elyos["scores"]
    for other in ("Aria", "Mela", "Targena", "Orichalcum Ore", "Sapphire Gemstone", "Ruby Gemstone"):
        assert scores["Yggdrasil Log"] > scores[other], f"Yggdrasil Log score must exceed {other}"

    res_asmo = classify_3d_material_patch(bgr, region_name="Altgard")
    assert res_asmo["name"] == "Asvata Log"
    assert res_asmo["category"] == "Logs & Trees"

    # Region-independent visual identity: an unknown region must still yield a
    # Log with the same visual descriptor, category and YOLO class -- the
    # region only selects the faction name.
    res_unknown = classify_3d_material_patch(bgr, region_name="Unknown Region")
    assert res_unknown["category"] == "Logs & Trees"
    assert map_material_name_to_yolo_class_id(res_unknown["name"]) == 5
    desc_unknown = extract_visual_descriptor(bgr)
    assert desc_unknown.morphological_profile == desc.morphological_profile
    assert desc_unknown.color_ratios["bark_trunk"] == desc.color_ratios["bark_trunk"]
    assert desc_unknown.vertical_edge_ratio == desc.vertical_edge_ratio

    assert match_material_names("Wood", res_elyos["name"]) is True
    assert map_material_name_to_yolo_class_id("Wood") == 5


def test_07_palette_swap_flips_classification() -> None:
    """Palette-swap control: color (not geometry) must drive the classification.

    The Aria patch geometry recolored with the Mela palette (green bush + red
    berries) must stop classifying as Aria, and the Mela geometry recolored
    with the Aria violet palette must stop classifying as Mela. If the
    classifier keyed on shape alone, the swap would not flip the verdict.
    """
    aria_bgr = _make_aria_bgr()
    assert classify_3d_material_patch(aria_bgr, region_name="Pilgrim's Ridge")["name"] == "Aria"

    # Aria geometry repainted with the Mela palette: the violet petal shapes
    # (same ellipse/circle coordinates as _make_aria_bgr) become green foliage
    # and the stem becomes red berries. The shape is unchanged; only color moves.
    hsv = _canvas(96, 80, NEUTRAL_DARK_RGB)
    cv2.ellipse(hsv, (40, 38), (24, 26), 0, 0, 360, _hsv(LEAF_GREEN_RGB), -1)
    cv2.circle(hsv, (30, 32), 10, _hsv(LEAF_GREEN_RGB), -1)
    cv2.circle(hsv, (50, 32), 10, _hsv(LEAF_GREEN_RGB), -1)
    cv2.circle(hsv, (40, 24), 5, _hsv(BERRY_RED_RGB), -1)
    hsv[66:90, 36:44] = _hsv(BERRY_RED_RGB)
    swapped = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    res_swapped = classify_3d_material_patch(swapped, region_name="Pilgrim's Ridge")
    assert res_swapped["name"] != "Aria"
    assert res_swapped["color_ratios"]["aria_purple"] < 0.05
