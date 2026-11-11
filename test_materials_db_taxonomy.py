# -*- coding: utf-8 -*-
"""Comprehensive verification of Aion 2 real-time 3D visual feature descriptors and exact taxonomy mapping."""

import unittest
import numpy as np
import cv2

from config.materials_db import (
    classify_3d_material_patch,
    compute_aspect_ratio_features,
    compute_3d_crest_texture,
    compute_hsv_color_distribution,
    extract_visual_descriptor,
    is_rejected_non_material,
)


class MaterialsDbTaxonomyTests(unittest.TestCase):
    """Test suite verifying genuine real-time visual feature descriptors and exact taxonomy mapping."""

    def test_empty_and_invalid_inputs_safe_default(self):
        """Empty or invalid images must safely return the documented fallback without crashing.

        F-1329 (stale-test fix, R19): the old assertion pinned the
        pre-D52 behaviour — invalid input returned the "Orichalcum Ore"
        placeholder at 0.50. That behaviour was itself the [Kritik]
        audit finding F-1084 ("invalid input is Orichalcum Ore at 0.50"),
        fixed by D52 (aa19e92, in both v1 and v2): invalid input is
        "Unknown" with confidence 0.0 and never a fabricated material.
        Callers gate on confidence and on the `unknown` flag. The test
        now asserts the corrected safe default: no crash, no region
        match, no color evidence, fallback confidence 0.0 — below every
        scanner acceptance gate.
        """
        for invalid in (None, np.array([]), np.zeros((0, 0, 3), dtype=np.uint8)):
            result = classify_3d_material_patch(invalid)
            # No visual evidence at all: explicitly NOT a classification.
            self.assertEqual(result["name"], "Unknown")
            self.assertEqual(result["category"], "Unknown")
            self.assertEqual(result["max_harvests"], 0)
            self.assertTrue(result["unknown"])
            # ...but it is explicitly flagged as NOT a real classification:
            # no region match, no color evidence, fallback confidence only.
            self.assertFalse(result["region_matched"])
            self.assertEqual(result["confidence"], 0.0)
            self.assertLess(result["confidence"], 0.65)
            self.assertEqual(result["color_ratios"], {})
            self.assertIn("visual_descriptor", result)
            self.assertIn("texture_features", result)
            self.assertIn("aspect_ratio_features", result)

    def test_hsv_color_distribution_extraction(self):
        """HSV color distribution extraction computes normalized histograms and accurate color masks."""
        # Create an image that is 60% purple (Hue ~140 in OpenCV 0..180 scale, Sat 200, Val 200)
        hsv_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_patch[:60, :, 0] = 140  # Hue 140 -> Purple/violet (bin 12)
        hsv_patch[:60, :, 1] = 200
        hsv_patch[:60, :, 2] = 200
        # 40% background rock
        hsv_patch[60:, :, 0] = 15
        hsv_patch[60:, :, 1] = 30
        hsv_patch[60:, :, 2] = 50

        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        data = compute_hsv_color_distribution(bgr_patch)

        self.assertEqual(len(data["hue_histogram"]), 16)
        self.assertAlmostEqual(sum(data["hue_histogram"]), 1.0, places=3)
        self.assertGreater(data["color_ratios"]["aria_purple"], 0.55)
        self.assertAlmostEqual(data["color_ratios"]["aria_purple"], 0.60, delta=0.05)

    def test_3d_crest_texture_directional_gradients(self):
        """Texture extraction computes directional gradients: vertical stripes produce high vertical edge ratio."""
        # Create image with strong vertical stripes (bark fissures)
        h, w = 100, 50
        vert_stripes = np.zeros((h, w, 3), dtype=np.uint8)
        for col in range(0, w, 4):
            vert_stripes[:, col : col + 2] = (180, 180, 180)
            vert_stripes[:, col + 2 : col + 4] = (40, 40, 40)

        tex = compute_3d_crest_texture(vert_stripes)
        self.assertGreater(tex["vertical_edge_ratio"], 0.60)
        self.assertLess(tex["horizontal_edge_ratio"], 0.40)
        self.assertGreater(tex["texture_contrast"], 0.20)

    def test_3d_crest_symmetry(self):
        """Symmetric winged crest pattern produces high crest symmetry score."""
        h, w = 40, 80
        crest_patch = np.zeros((h, w, 3), dtype=np.uint8)
        # Create bilaterally symmetric V-chevron wings
        for y in range(h):
            dx = int(y * 0.8)
            crest_patch[y, max(0, 40 - dx) : max(0, 40 - dx + 3)] = (240, 240, 240)
            crest_patch[y, min(w - 1, 40 + dx - 3) : min(w, 40 + dx)] = (240, 240, 240)

        tex = compute_3d_crest_texture(crest_patch, crest_bgr=crest_patch)
        self.assertGreater(tex["crest_symmetry_score"], 0.70)
        self.assertGreater(tex["crest_brightness"], 0.05)

    def test_aspect_ratio_features_profiles(self):
        """Aspect ratio feature extractor maps geometric dimensions to proper morphological profiles.

        F-1420: the patches below carry REAL visual content for the claimed
        morphology (bark grain, ore vein, foliage + berries) instead of
        content-free zero arrays, so the profile claim is grounded in a
        realistic instance of each shape. The profile itself is derived
        from the patch dimensions (see the explicit-bbox check below).
        """
        # Tall vertical trunk: brown bark with vertical grain + green canopy
        trunk_hsv = np.zeros((150, 50, 3), dtype=np.uint8)
        trunk_hsv[:, :, 0] = 15
        trunk_hsv[:, :, 1] = 120
        trunk_hsv[:, :, 2] = 80
        for c in range(0, 50, 4):
            trunk_hsv[:, c : c + 2, 2] = 110
            trunk_hsv[:, c + 2 : c + 4, 2] = 60
        trunk_hsv[:25, :, 0] = 50
        trunk_hsv[:25, :, 1] = 170
        trunk_hsv[:25, :, 2] = 130
        ar_trunk = compute_aspect_ratio_features(cv2.cvtColor(trunk_hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(ar_trunk["morphological_profile"], "vertical_trunk")
        self.assertGreater(ar_trunk["height_width_ratio"], 2.0)

        # Horizontal ore vein: bronze/teal vein on dark rock outcrop
        vein_hsv = np.zeros((50, 100, 3), dtype=np.uint8)
        vein_hsv[:, :, 0] = 0
        vein_hsv[:, :, 1] = 15
        vein_hsv[:, :, 2] = 45
        cv2.line(vein_hsv, (10, 25), (90, 22), (15, 180, 180), 8)
        cv2.line(vein_hsv, (15, 35), (85, 33), (85, 160, 170), 5)
        ar_vein = compute_aspect_ratio_features(cv2.cvtColor(vein_hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(ar_vein["morphological_profile"], "horizontal_vein")
        self.assertGreater(ar_vein["aspect_ratio"], 1.5)

        # Squat bush: green foliage with dispersed red berries
        bush_hsv = np.zeros((60, 80, 3), dtype=np.uint8)
        bush_hsv[:, :, 0] = 55
        bush_hsv[:, :, 1] = 170
        bush_hsv[:, :, 2] = 150
        for cy, cx in ((20, 25), (35, 60), (45, 30)):
            cv2.circle(bush_hsv, (cx, cy), 5, (5, 220, 220), -1)
        ar_bush = compute_aspect_ratio_features(cv2.cvtColor(bush_hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(ar_bush["morphological_profile"], "squat_bush")

        # Explicit bbox aspect ratio takes precedence
        ar_custom = compute_aspect_ratio_features(np.zeros((50, 50, 3), dtype=np.uint8), bbox_aspect_ratio=2.5)
        self.assertAlmostEqual(ar_custom["aspect_ratio"], 2.5, places=2)
        self.assertEqual(ar_custom["morphological_profile"], "horizontal_vein")

    def test_taxonomy_aria_herb_exact_mapping(self):
        """Aria purple crystal petals with herb proportions map to Aria (Herbs, Alchemy)."""
        h, w = 90, 80
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Violet crystal flower (Hue 145, Sat 200, Val 200)
        hsv[:60, 20:60, 0] = 145
        hsv[:60, 20:60, 1] = 200
        hsv[:60, 20:60, 2] = 200
        # Green stem at bottom
        hsv[60:, 35:45, 0] = 50
        hsv[60:, 35:45, 1] = 160
        hsv[60:, 35:45, 2] = 140

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Western Altamia Highland")

        self.assertEqual(result["name"], "Aria")
        self.assertEqual(result["category"], "Herbs")
        self.assertEqual(result["taxonomy"]["profession"], "Alchemy")
        self.assertEqual(result["max_harvests"], 3)
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertTrue(result["region_matched"])

    def test_taxonomy_mela_bush_exact_mapping(self):
        """Mela red berries on green foliage bush with squat aspect ratio maps to Mela (Cooking & Fibers)."""
        h, w = 70, 100
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Green foliage bush (Hue 55, Sat 170, Val 150)
        hsv[:, :, 0] = 55
        hsv[:, :, 1] = 170
        hsv[:, :, 2] = 150
        # Dispersed red berry clusters
        for cy, cx in ((20, 25), (35, 60), (45, 30), (25, 75)):
            cv2.circle(hsv, (cx, cy), 5, (5, 220, 220), -1)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Mela")
        self.assertEqual(result["category"], "Cooking & Fibers")
        self.assertEqual(result["taxonomy"]["profession"], "Cooking & Tailoring")
        self.assertGreaterEqual(result["confidence"], 0.80)

    def test_taxonomy_ruby_vs_mela_disambiguation(self):
        """Ruby crystal spire (crimson, high specularity, zero foliage, vertical spire) must NEVER be confused with Mela."""
        h, w = 120, 60
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Dark rock background
        hsv[:, :, 0] = 0
        hsv[:, :, 1] = 20
        hsv[:, :, 2] = 40
        # Crimson crystal spire (Hue 5, Sat 230, Val 230)
        cv2.ellipse(hsv, (30, 60), (15, 50), 0, 0, 360, (5, 230, 230), -1)
        # Specular facet highlight (Val 250, Sat 40)
        cv2.circle(hsv, (30, 40), 6, (5, 30, 250), -1)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Ruby Gemstone")
        self.assertEqual(result["category"], "Gemstones")
        self.assertEqual(result["taxonomy"]["profession"], "Handicrafting")
        self.assertNotEqual(result["name"], "Mela")

    def test_taxonomy_sapphire_gemstone_mapping(self):
        """Deep glowing blue/cyan crystal with specular highlights maps to Sapphire Gemstone (Gemstones)."""
        h, w = 90, 80
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Rock background
        hsv[:, :, 2] = 45
        # Deep cyan/blue crystal cluster (Hue 110, Sat 220, Val 220)
        cv2.ellipse(hsv, (40, 45), (25, 35), 0, 0, 360, (110, 220, 220), -1)
        # Specular glint
        cv2.circle(hsv, (40, 35), 7, (110, 30, 252), -1)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Western Altamia Highland")

        self.assertEqual(result["name"], "Sapphire Gemstone")
        self.assertEqual(result["category"], "Gemstones")
        self.assertEqual(result["taxonomy"]["profession"], "Handicrafting")

    def test_taxonomy_diamond_gemstone_mapping(self):
        """Brilliant white/ice crystal cluster maps to Diamond Gemstone."""
        h, w = 90, 80
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 2] = 40
        # White/ice crystal (Hue 100, low/med Sat 30, high Val 240)
        cv2.ellipse(hsv, (40, 45), (25, 35), 0, 0, 360, (100, 30, 240), -1)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Diamond Gemstone")
        self.assertEqual(result["category"], "Gemstones")

    def test_taxonomy_tree_logs_vertical_trunk_mapping(self):
        """Tree trunk with vertical bark texture and tall aspect ratio maps to Yggdrasil/Asvata Log (Logs & Trees)."""
        h, w = 150, 60  # Tall vertical aspect ratio
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Brown bark trunk (Hue 15, Sat 120, Val 80)
        hsv[:, :, 0] = 15
        hsv[:, :, 1] = 120
        hsv[:, :, 2] = 80
        # Vertical bark grain variations
        for c in range(0, w, 4):
            hsv[:, c : c + 2, 2] = 110
            hsv[:, c + 2 : c + 4, 2] = 60
        # Green canopy foliage at the very top
        hsv[:25, :, 0] = 50
        hsv[:25, :, 1] = 170
        hsv[:25, :, 2] = 130

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Elyos region -> Yggdrasil Log
        res_elyos = classify_3d_material_patch(bgr, region_name="Verteron")
        self.assertEqual(res_elyos["name"], "Yggdrasil Log")
        self.assertEqual(res_elyos["category"], "Logs & Trees")
        self.assertEqual(res_elyos["taxonomy"]["profession"], "Woodcutting")

        # Asmodian region -> Asvata Log
        res_asmo = classify_3d_material_patch(bgr, region_name="Altgard")
        self.assertEqual(res_asmo["name"], "Asvata Log")
        self.assertEqual(res_asmo["category"], "Logs & Trees")

    def test_taxonomy_orichalcum_ore_vein_mapping(self):
        """Metallic copper/bronze/teal vein on dark rock outcrop maps to Orichalcum Ore (Ores & Metals)."""
        h, w = 60, 110  # Horizontal outcrop aspect ratio
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Dark rock substrate
        hsv[:, :, 0] = 0
        hsv[:, :, 1] = 15
        hsv[:, :, 2] = 45
        # Bronze/amber metallic rock vein
        cv2.line(hsv, (10, 30), (100, 25), (15, 180, 180), 8)
        # Oxidized teal vein
        cv2.line(hsv, (15, 38), (95, 36), (85, 160, 170), 5)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Orichalcum Ore")
        self.assertEqual(result["category"], "Ores & Metals")
        self.assertEqual(result["taxonomy"]["profession"], "Blacksmithing / Armorsmithing")

    def test_taxonomy_targena_gold_green_herb_mapping(self):
        """Golden-yellow and green herb patch maps to Targena (Herbs, Alchemy)."""
        h, w = 80, 80
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Golden-yellow flower (Hue 32, Sat 190, Val 200)
        cv2.circle(hsv, (40, 35), 20, (32, 190, 200), -1)
        # Green foliage (Hue 55, Sat 180, Val 160)
        hsv[50:, :, 0] = 55
        hsv[50:, :, 1] = 180
        hsv[50:, :, 2] = 160

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Targena")
        self.assertEqual(result["category"], "Herbs")
        self.assertEqual(result["taxonomy"]["profession"], "Alchemy")

    def test_taxonomy_odyle_floating_orb_mapping(self):
        """Floating teal/green glowing energy orb maps to Odyle (Odyle & Aether)."""
        h, w = 80, 80
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # Radial glowing teal orb (Hue 75, Sat 190, Val 220)
        cv2.circle(hsv, (40, 40), 28, (75, 190, 220), -1)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = classify_3d_material_patch(bgr, region_name="Pilgrim's Ridge")

        self.assertEqual(result["name"], "Odyle")
        self.assertEqual(result["category"], "Odyle & Aether")
        self.assertEqual(result["taxonomy"]["profession"], "Flight / Ground Extraction")

    def test_non_material_rejection_keywords(self):
        """Strictly reject minimap feathers, monsters, NPCs, and quests from being treated as gatherables."""
        for token in (
            "white feather",
            "minimap feather",
            "tüy",
            "yellow-maned zaif",
            "highland odyle sparkie",
            "monster dot",
            "functional npc",
            "quest giver",
        ):
            self.assertTrue(is_rejected_non_material(token), f"Expected rejection for '{token}'")

        # Legitimate gatherable catalog materials must never be rejected
        for valid in ("Aria", "Mela", "Targena", "Orichalcum Ore", "Yggdrasil Log", "Sapphire Gemstone"):
            self.assertFalse(is_rejected_non_material(valid), f"Expected acceptance for '{valid}'")

    def test_zero_cheating_dynamic_behavior(self):
        """Zero synthetic cheating: perturbing image colors dynamically and mathematically alters descriptors and predictions."""
        # Baseline Aria
        h, w = 80, 80
        base_hsv = np.full((h, w, 3), (145, 200, 200), dtype=np.uint8)
        bgr_aria = cv2.cvtColor(base_hsv, cv2.COLOR_HSV2BGR)
        desc_aria = extract_visual_descriptor(bgr_aria)
        self.assertGreater(desc_aria.color_ratios["aria_purple"], 0.90)

        # Perturb to a real Mela bush: green foliage + red berry clusters
        # (same recipe as test_taxonomy_mela_bush_exact_mapping).
        # F-1543 (stale-test fix, R19): the old uniform-green patch carried
        # no Mela evidence at all (no red berries) — under the D52
        # calibrated confidence (F-1082, aa19e92, no 0.74 floor) it maps
        # to Mela at only 0.681, because Targena/Azpha share the green
        # evidence. The "verified: confidence 0.945" note was measured
        # under the old floored formula. A genuine Mela patch has berries:
        # red_berry ~0.049 triggers the +0.44 berry bonus and the
        # calibrated confidence is ~0.94.
        pert_hsv = np.full((h, w, 3), (55, 180, 180), dtype=np.uint8)
        for cy, cx in ((20, 25), (35, 60), (45, 30), (25, 70)):
            cv2.circle(pert_hsv, (cx, cy), 5, (5, 220, 220), -1)
        bgr_green = cv2.cvtColor(pert_hsv, cv2.COLOR_HSV2BGR)
        desc_green = extract_visual_descriptor(bgr_green)
        self.assertEqual(desc_green.color_ratios["aria_purple"], 0.0)
        self.assertGreater(desc_green.color_ratios["green_leaf"], 0.90)

        # Dynamic mapping shifts from Aria to green herb
        res_aria = classify_3d_material_patch(bgr_aria)
        res_green = classify_3d_material_patch(bgr_green)
        self.assertEqual(res_aria["name"], "Aria")
        # F-1543: assert WHAT the green patch maps to, not merely what it
        # is not — a genuine Mela bush (green foliage + red berries)
        # classifies as Mela with calibrated confidence ~0.94, so a silent
        # remap to any other material would now fail instead of passing on
        # a bare inequality.
        self.assertNotEqual(res_green["name"], "Aria")
        self.assertEqual(res_green["name"], "Mela")
        self.assertGreaterEqual(res_green["confidence"], 0.80)


if __name__ == "__main__":
    unittest.main()
