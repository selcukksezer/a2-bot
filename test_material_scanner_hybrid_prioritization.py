# -*- coding: utf-8 -*-
"""
test_material_scanner_hybrid_prioritization.py
=============================================
Verification test suite for Agent G2:
Hybrid 3D Material Scanner & Node Prioritization.

Validates:
1. DetectedMaterial dataclass extensions: priority_score, rarity_tier, is_depleted.
2. MaterialScanner 3D YOLO scanner integration: scan_3d_yolo_materials, get_yolo_detector, set_yolo_detector.
3. Multi-modal hybrid fusion (_fuse_hybrid_detections): fuses template + YOLO detections with confidence boost.
4. Candidate prioritization by proximity, rarity, and harvest count remaining:
   - Proximity: near nodes & active interaction prompts receive highest distance weight.
   - Rarity: Gemstones (Tier 5) > Rare Ores/Flora (Tier 4) > Herbs/Trees (Tier 3) > Common (Tier 2) > Generic (Tier 1).
   - Harvests Remaining: fresh 3/3 nodes prioritized over 1/3, and 0/3 (depleted) nodes penalized.
5. ExemplarMemory gating & hot-reload integration:
   - Rejection of learned negative patches (rocks/terrain/clutter).
   - Refinement of learned positive patches (boosted confidence & catalog classification).
6. Full end-to-end scan_frame_for_materials execution and ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from core.material_scanner import DetectedMaterial, MaterialScanner
from dataset_factory.exemplar_memory import (
    ExemplarMemory,
    extract_crop_features,
    get_exemplar_memory,
)


class MockYoloDetector:
    """Mock YOLO 3D detector providing configurable object detections for tests."""

    def __init__(self, detections: list[dict] | None = None) -> None:
        self.detections = detections or []

    def detect_3d_objects(self, frame: np.ndarray) -> list[dict]:
        return list(self.detections)


class TestMaterialScannerHybridPrioritization(unittest.TestCase):
    """Test suite for Hybrid 3D Material Scanner & Node Prioritization."""

    def setUp(self) -> None:
        self.scanner = MaterialScanner()

    def test_01_detected_material_dataclass_fields(self) -> None:
        """Verify DetectedMaterial dataclass fields: priority_score, rarity_tier, is_depleted."""
        mat = DetectedMaterial(
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
        self.assertEqual(mat.priority_score, 0.0)
        self.assertEqual(mat.rarity_tier, 1)
        self.assertFalse(mat.is_depleted)

        depleted = DetectedMaterial(
            material_type="Aria",
            source="3d_world",
            screen_x=500,
            screen_y=350,
            bbox=(480, 330, 40, 40),
            rel_map_dx=1.0,
            rel_map_dy=2.0,
            estimated_distance_m=3.0,
            confidence=0.85,
            harvests_remaining=0,
        )
        self.assertTrue(depleted.is_depleted)

    def test_02_yolo_detector_get_and_set(self) -> None:
        """Verify YOLO detector get, set, and health summary."""
        mock = MockYoloDetector()
        self.scanner.set_yolo_detector(mock)
        self.assertIs(self.scanner.get_yolo_detector(), mock)

        health = self.scanner.get_material_scanner_health_summary()
        self.assertTrue(health.get("material_scanner_ready"))
        self.assertTrue(health.get("yolo_detector_ready"))
        self.assertTrue(health.get("hybrid_scoring_ready"))
        self.assertTrue(health.get("prioritization_ready"))

    def test_03_scan_3d_yolo_materials_filters_monsters_and_hud(self) -> None:
        """Verify that scan_3d_yolo_materials accepts resources and filters monsters/HUD."""
        mock = MockYoloDetector(
            detections=[
                # Valid ore vein in 3D viewport
                {
                    "class_name": "ore_vein",
                    "confidence": 0.82,
                    "bbox": (500, 350, 60, 50),
                    "center": (530, 375),
                },
                # Valid herb flora in 3D viewport
                {
                    "class_name": "herb_aria",
                    "confidence": 0.89,
                    "bbox": (300, 300, 40, 40),
                    "center": (320, 320),
                },
                # Approaching monster (must be filtered out)
                {
                    "class_name": "approaching_monster",
                    "confidence": 0.91,
                    "bbox": (700, 320, 80, 90),
                    "center": (740, 365),
                },
                # Hostile mob threat (must be filtered out)
                {
                    "class_name": "threat_hostile",
                    "confidence": 0.95,
                    "bbox": (600, 250, 70, 70),
                    "center": (635, 285),
                },
                # Detection in bottom skill bar (must be filtered out)
                {
                    "class_name": "herb_flora",
                    "confidence": 0.80,
                    "bbox": (600, 680, 50, 30),
                    "center": (625, 695),
                },
            ]
        )
        self.scanner.set_yolo_detector(mock)
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        dets = self.scanner.scan_3d_yolo_materials(frame)

        # Only the 2 valid 3D resource detections should be returned
        self.assertEqual(len(dets), 2)
        classes = {d.material_type for d in dets}
        self.assertIn("Aria", classes)
        self.assertTrue(any("Ore" in c or c == "Orichalcum Ore" for c in classes))
        for d in dets:
            self.assertEqual(d.source, "3d_yolo")
            self.assertGreaterEqual(d.confidence, 0.80)

    def test_04_hybrid_fusion_merges_overlapping_yolo_and_template(self) -> None:
        """Verify _fuse_hybrid_detections merges template and YOLO detections with confidence boost."""
        # Template detection of a gather crest at (600, 360)
        tpl_crest = DetectedMaterial(
            material_type="3D Crest",
            source="3d_world",
            screen_x=600,
            screen_y=360,
            bbox=(585, 345, 30, 30),
            rel_map_dx=2.0,
            rel_map_dy=4.0,
            estimated_distance_m=4.5,
            confidence=0.84,
        )
        # Standalone template detection at (200, 300)
        tpl_standalone = DetectedMaterial(
            material_type="Mela",
            source="3d_world",
            screen_x=200,
            screen_y=300,
            bbox=(180, 280, 40, 40),
            rel_map_dx=-8.0,
            rel_map_dy=2.0,
            estimated_distance_m=8.2,
            confidence=0.81,
        )
        # YOLO detection directly on the resource mesh below the crest at (602, 380)
        yolo_ore = DetectedMaterial(
            material_type="Orichalcum Ore",
            source="3d_yolo",
            screen_x=602,
            screen_y=385,
            bbox=(575, 365, 55, 45),
            rel_map_dx=2.1,
            rel_map_dy=4.3,
            estimated_distance_m=4.6,
            confidence=0.86,
        )
        # Standalone YOLO detection at (950, 420)
        yolo_standalone = DetectedMaterial(
            material_type="Aria",
            source="3d_yolo",
            screen_x=950,
            screen_y=420,
            bbox=(930, 400, 40, 40),
            rel_map_dx=10.0,
            rel_map_dy=6.0,
            estimated_distance_m=11.6,
            confidence=0.79,
        )

        fused = self.scanner._fuse_hybrid_detections(
            [tpl_crest, tpl_standalone],
            [yolo_ore, yolo_standalone],
        )

        # Expected: 1 hybrid fused node + 1 standalone template + 1 standalone YOLO = 3 total
        self.assertEqual(len(fused), 3)

        hybrid_nodes = [d for d in fused if d.source == "hybrid_3d"]
        self.assertEqual(len(hybrid_nodes), 1)
        h_node = hybrid_nodes[0]
        # Conf must be boosted above max(0.84, 0.86) -> 0.86 + 0.08 = 0.94
        self.assertAlmostEqual(h_node.confidence, 0.94, places=2)
        # Material type refined from generic '3D Crest' to specific 'Orichalcum Ore'
        self.assertEqual(h_node.material_type, "Orichalcum Ore")
        # Encompassing bounding box
        self.assertLessEqual(h_node.bbox[0], 575)
        self.assertGreaterEqual(h_node.bbox[0] + h_node.bbox[2], 615)

    def test_05_candidate_prioritization_by_proximity(self) -> None:
        """Verify closer nodes get higher priority when material type and harvest count are identical."""
        near_node = DetectedMaterial(
            material_type="Aria",
            source="3d_world",
            screen_x=600,
            screen_y=450,
            bbox=(580, 430, 40, 40),
            rel_map_dx=0.5,
            rel_map_dy=1.0,
            estimated_distance_m=2.0,  # 2m away
            confidence=0.85,
            harvests_remaining=3,
        )
        far_node = DetectedMaterial(
            material_type="Aria",
            source="3d_world",
            screen_x=620,
            screen_y=280,
            bbox=(600, 260, 40, 40),
            rel_map_dx=5.0,
            rel_map_dy=25.0,
            estimated_distance_m=25.0,  # 25m away
            confidence=0.85,
            harvests_remaining=3,
        )

        prioritized = self.scanner.prioritize_candidates([far_node, near_node])
        self.assertEqual(len(prioritized), 2)
        self.assertIs(prioritized[0], near_node)
        self.assertGreater(near_node.priority_score, far_node.priority_score)

    def test_06_candidate_prioritization_by_rarity(self) -> None:
        """Verify rarer materials (Gemstones, Orichalcum) prioritize over common (Mela) at same distance."""
        gem_node = DetectedMaterial(
            material_type="Diamond Gemstone",  # Tier 5
            source="3d_world",
            screen_x=600,
            screen_y=400,
            bbox=(580, 380, 40, 40),
            rel_map_dx=2.0,
            rel_map_dy=5.0,
            estimated_distance_m=6.0,
            confidence=0.85,
            harvests_remaining=3,
        )
        ore_node = DetectedMaterial(
            material_type="Orichalcum Ore",  # Tier 4
            source="3d_world",
            screen_x=620,
            screen_y=400,
            bbox=(600, 380, 40, 40),
            rel_map_dx=2.1,
            rel_map_dy=5.0,
            estimated_distance_m=6.0,
            confidence=0.85,
            harvests_remaining=3,
        )
        herb_node = DetectedMaterial(
            material_type="Mela",  # Tier 2
            source="3d_world",
            screen_x=580,
            screen_y=400,
            bbox=(560, 380, 40, 40),
            rel_map_dx=1.9,
            rel_map_dy=5.0,
            estimated_distance_m=6.0,
            confidence=0.85,
            harvests_remaining=3,
        )

        prioritized = self.scanner.prioritize_candidates([herb_node, ore_node, gem_node])
        self.assertEqual(len(prioritized), 3)
        self.assertEqual(prioritized[0].material_type, "Diamond Gemstone")
        self.assertEqual(prioritized[1].material_type, "Orichalcum Ore")
        self.assertEqual(prioritized[2].material_type, "Mela")
        self.assertEqual(prioritized[0].rarity_tier, 5)
        self.assertEqual(prioritized[1].rarity_tier, 4)
        self.assertEqual(prioritized[2].rarity_tier, 2)

    def test_07_candidate_prioritization_by_harvest_count_and_depletion(self) -> None:
        """Verify fresh 3/3 nodes prioritize over 1/3 nodes and 0/3 depleted nodes are demoted."""
        fresh_node = DetectedMaterial(
            material_type="Orichalcum Ore",
            source="3d_world",
            screen_x=600,
            screen_y=400,
            bbox=(580, 380, 40, 40),
            rel_map_dx=2.0,
            rel_map_dy=4.0,
            estimated_distance_m=5.0,
            confidence=0.85,
            harvests_remaining=3,
        )
        partial_node = DetectedMaterial(
            material_type="Orichalcum Ore",
            source="3d_world",
            screen_x=600,
            screen_y=400,
            bbox=(580, 380, 40, 40),
            rel_map_dx=2.0,
            rel_map_dy=4.0,
            estimated_distance_m=5.0,
            confidence=0.85,
            harvests_remaining=1,
        )
        depleted_node = DetectedMaterial(
            material_type="Orichalcum Ore",
            source="3d_world",
            screen_x=600,
            screen_y=400,
            bbox=(580, 380, 40, 40),
            rel_map_dx=2.0,
            rel_map_dy=4.0,
            estimated_distance_m=5.0,
            confidence=0.85,
            harvests_remaining=0,
        )

        prioritized = self.scanner.prioritize_candidates([partial_node, depleted_node, fresh_node])
        self.assertEqual(len(prioritized), 3)
        self.assertIs(prioritized[0], fresh_node)
        self.assertIs(prioritized[1], partial_node)
        self.assertIs(prioritized[2], depleted_node)
        self.assertLess(depleted_node.priority_score, 0.05)

        # Verify filter_depleted excludes 0/3 completely
        filtered = self.scanner.prioritize_candidates(
            [partial_node, depleted_node, fresh_node], filter_depleted=True
        )
        self.assertEqual(len(filtered), 2)
        self.assertNotIn(depleted_node, filtered)

    def test_08_active_gather_prompt_receives_maximum_urgency(self) -> None:
        """Verify an interactive 'F Gather' prompt receives top priority for immediate harvesting."""
        prompt = DetectedMaterial(
            material_type="Orichalcum Ore",
            source="gather_prompt",
            screen_x=640,
            screen_y=420,
            bbox=(610, 400, 60, 30),
            rel_map_dx=0.2,
            rel_map_dy=1.5,
            estimated_distance_m=1.8,
            confidence=0.92,
            harvests_remaining=3,
        )
        far_gem = DetectedMaterial(
            material_type="Diamond Gemstone",
            source="3d_world",
            screen_x=800,
            screen_y=280,
            bbox=(780, 260, 40, 40),
            rel_map_dx=15.0,
            rel_map_dy=20.0,
            estimated_distance_m=25.0,
            confidence=0.90,
            harvests_remaining=3,
        )

        top = self.scanner.get_top_candidate([far_gem, prompt])
        self.assertIsNotNone(top)
        self.assertIs(top, prompt)
        self.assertGreaterEqual(prompt.priority_score, 0.85)

    def test_09_exemplar_memory_negative_drop_and_positive_refine(self) -> None:
        """Verify ExemplarMemory rejects learned negative patches and refines positive patches."""
        mem = get_exemplar_memory()

        # 1. Teach a false gray rock patch as negative
        rock_patch = np.full((40, 40, 3), 118, dtype=np.uint8)
        mem.add_negative_exemplar(rock_patch, class_id=14, class_name="false_rock_ground")

        # 2. Teach Aria purple flower patch as positive
        aria_patch = np.zeros((40, 40, 3), dtype=np.uint8)
        aria_patch[:, :] = (175, 45, 165)  # purple BGR
        mem.add_positive_exemplar(aria_patch, class_id=0, class_name="herb_aria")

        frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
        # Place rock at (300, 300)
        frame[280:320, 280:320] = rock_patch
        # Place Aria at (600, 400)
        frame[380:420, 580:620] = aria_patch

        cand_rock = DetectedMaterial(
            material_type="Unknown_Rock",
            source="3d_world",
            screen_x=300,
            screen_y=300,
            bbox=(280, 280, 40, 40),
            rel_map_dx=0.0,
            rel_map_dy=0.0,
            estimated_distance_m=10.0,
            confidence=0.70,
        )
        cand_aria = DetectedMaterial(
            material_type="Raw_Flower",
            source="3d_yolo",
            screen_x=600,
            screen_y=400,
            bbox=(580, 380, 40, 40),
            rel_map_dx=0.0,
            rel_map_dy=0.0,
            estimated_distance_m=6.0,
            confidence=0.65,
        )

        refined = self.scanner._refine_candidates_with_exemplars(frame, [cand_rock, cand_aria])

        # Negative rock must be completely eliminated
        self.assertEqual(len(refined), 1)
        # Aria must have class refined to Aria and confidence elevated
        self.assertEqual(refined[0].material_type, "Aria")
        self.assertGreaterEqual(refined[0].confidence, 0.85)

    def test_10_end_to_end_scan_frame_for_materials_hybrid_pipeline(self) -> None:
        """Verify full scan_frame_for_materials integrates YOLO, template, ExemplarMemory, and prioritization."""
        # Setup mock YOLO detector with a 3D ore detection
        mock_yolo = MockYoloDetector(
            detections=[
                {
                    "class_name": "ore_orichalcum",
                    "confidence": 0.88,
                    "bbox": (580, 380, 60, 45),
                    "center": (610, 402),
                }
            ]
        )
        self.scanner.set_yolo_detector(mock_yolo)

        frame = np.full((720, 1280, 3), 70, dtype=np.uint8)
        # Add realistic dark rock + copper/amber ore vein texture to the node bbox
        frame[380:425, 580:640] = (40, 40, 40)
        frame[390:415, 590:630] = (30, 135, 215)
        results = self.scanner.scan_frame_for_materials(frame)

        # Detections should be present, prioritized, and non-empty
        self.assertGreaterEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top.material_type, "Orichalcum Ore")
        self.assertGreater(top.priority_score, 0.0)
        self.assertGreaterEqual(top.rarity_tier, 4)

        # Verify results are sorted descending by priority_score
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].priority_score, results[i + 1].priority_score)


if __name__ == "__main__":
    unittest.main()
