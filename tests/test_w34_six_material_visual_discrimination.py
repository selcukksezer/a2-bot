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


def _make_aria_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Aria (purple/violet crystal petals + green stem)."""
    h, w = 96, 80
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark neutral background
    hsv[:, :, 0] = 15
    hsv[:, :, 1] = 25
    hsv[:, :, 2] = 45
    # Violet-purple crystal petals (Hue 138..145, high saturation & value)
    cv2.ellipse(hsv, (40, 38), (24, 26), 0, 0, 360, (140, 200, 210), -1)
    cv2.circle(hsv, (30, 32), 10, (135, 185, 225), -1)
    cv2.circle(hsv, (50, 32), 10, (144, 210, 220), -1)
    # Specular white-violet petal tips
    cv2.circle(hsv, (40, 24), 5, (138, 30, 245), -1)
    # Upright green stem at base
    hsv[66:90, 36:44] = (52, 165, 145)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_mela_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Mela (squat green bush with red berry clusters)."""
    h, w = 72, 104
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark soil/rock surround so green foliage is a bounded bush dome (< 70% of patch)
    hsv[:, :] = (15, 30, 42)
    # Green foliage bush dome (Hue 55, Sat 175, Val 155)
    cv2.ellipse(hsv, (52, 38), (46, 30), 0, 0, 360, (55, 175, 155), -1)
    # Dispersed red berry clusters (Hue 5, Sat 225, Val 220)
    for cx, cy in ((28, 30), (48, 26), (70, 30), (36, 46), (56, 44), (74, 46)):
        cv2.circle(hsv, (cx, cy), 6, (5, 225, 220), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_targena_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Targena (golden-yellow blossom on green herb leaves)."""
    h, w = 88, 80
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Green herb foliage base
    hsv[:, :] = (54, 175, 150)
    # Golden-yellow Targena blossom cluster (Hue 30..34, Sat 195, Val 185)
    cv2.circle(hsv, (40, 34), 20, (32, 195, 185), -1)
    cv2.circle(hsv, (28, 42), 12, (30, 185, 180), -1)
    cv2.circle(hsv, (52, 42), 12, (34, 200, 185), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_orichalcum_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Orichalcum Ore (copper/bronze + teal veins on dark rock)."""
    h, w = 64, 112
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark rock substrate
    hsv[:, :] = (12, 25, 48)
    # Rough rock strata lines
    cv2.line(hsv, (6, 16), (104, 18), (10, 30, 68), 3)
    cv2.line(hsv, (8, 50), (102, 48), (10, 35, 35), 3)
    # Metallic copper/amber-gold vein
    cv2.line(hsv, (12, 30), (100, 26), (15, 195, 190), 9)
    # Oxidized teal mineral vein
    cv2.line(hsv, (16, 40), (96, 38), (85, 165, 175), 6)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_obsidian_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Obsidian Ore (dark violet/black glassy vein with specular glints)."""
    h, w = 68, 108
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Glassy black obsidian substrate
    hsv[:, :] = (135, 45, 32)
    # Dark violet obsidian crystalline veins (Hue 132..148, Sat 95..160, Val 55..95)
    cv2.ellipse(hsv, (54, 34), (38, 20), 0, 0, 360, (138, 130, 78), -1)
    cv2.line(hsv, (16, 28), (92, 24), (142, 145, 88), 7)
    cv2.line(hsv, (20, 42), (88, 40), (132, 120, 72), 6)
    # Sharp specular facet highlights
    cv2.circle(hsv, (42, 26), 4, (135, 20, 235), -1)
    cv2.circle(hsv, (66, 34), 4, (135, 15, 245), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_crystal_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Crystal / Gemstone (glowing cyan-blue crystal spire with specular glint)."""
    h, w = 96, 80
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark rock base
    hsv[:, :] = (0, 15, 42)
    # Deep glowing cyan/sapphire crystal cluster (Hue 108, Sat 220, Val 225)
    cv2.ellipse(hsv, (40, 46), (24, 34), 0, 0, 360, (108, 220, 225), -1)
    # Specular crystal facet highlight
    cv2.circle(hsv, (40, 34), 7, (108, 25, 252), -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_wood_bgr() -> np.ndarray:
    """Create a synthetic BGR patch for Wood / Tree Log (vertical brown bark trunk with striations and canopy)."""
    h, w = 150, 60
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Brown bark trunk (Hue 15, Sat 120, Val 85)
    hsv[:, :] = (15, 120, 85)
    # Vertical bark grain striations
    for col in range(0, w, 4):
        hsv[:, col : col + 2, 2] = 112
        hsv[:, col + 2 : col + 4, 2] = 58
    # Leafy green canopy at top
    hsv[:26, :] = (50, 170, 135)
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
    """Verify Obsidian and Crystal synthetic BGR patches are accurately classified and discriminated from other materials."""
    # Part A: Obsidian Ore dark violet/black glassy vein
    obsidian_bgr = _make_obsidian_bgr()
    disc_obs = discriminate_3d_material(obsidian_bgr, exemplar_memory=False)
    assert disc_obs.is_authentic_node is True
    assert disc_obs.material_type == "Obsidian Ore"
    assert disc_obs.class_name == "ore_obsidian"
    assert disc_obs.color_ratios["ore_obsidian"] > disc_obs.color_ratios["ore_orichalcum"]
    assert disc_obs.color_ratios["ore_obsidian"] > disc_obs.color_ratios["herb_aria"]
    assert match_material_names("Obsidian", disc_obs.material_type) is True

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
    """Verify Wood synthetic BGR patch is classified as Yggdrasil/Asvata Log (Logs & Trees) and discriminated from others."""
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

    assert match_material_names("Wood", res_elyos["name"]) is True
    assert map_material_name_to_yolo_class_id("Wood") == 5
