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

from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.material_scanner import DetectedMaterial, MaterialScanner
from dataset_factory.exemplar_memory import ExemplarMemory


def _textured_patch(base_bgr, shape, seed=0, amp=12):
    """Deterministic textured patch: base colour + seeded luminance noise.

    K-3 (dataset_factory/exemplar_memory.py) rejects flat single-colour
    crops as degenerate ("flat crop, no texture"), so exemplar fixtures
    must carry real texture. The noise is luminance-only (the same offset
    on all three channels), so hue is preserved and a grey patch stays
    grey through any BGR->grey conversion on the way to the matcher.
    """
    rng = np.random.default_rng(seed)
    base = np.full(shape, base_bgr, dtype=np.int16)
    noise = rng.integers(-amp, amp + 1, size=(shape[0], shape[1], 1), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


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
        # NOTE (F-1419): rarity_tier == 1 here is only the dataclass default
        # for a freshly constructed, unprioritized node — it is NOT the
        # catalog tier of Orichalcum. Prioritization assigns the real tier:
        self.assertEqual(mat.rarity_tier, 1)
        self.scanner.prioritize_candidates([mat])
        self.assertEqual(mat.rarity_tier, 4)  # Ores & Metals / Orichalcum
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
                # F-1327: low-confidence detections on a featureless (uniform
                # gray) crop must be rejected by the visual-discrimination
                # gate: a non-authentic crop is only accepted when
                # confidence >= 0.65, so these must NOT bypass it.
                {
                    "class_name": "ore_vein",
                    "confidence": 0.50,
                    "bbox": (100, 100, 60, 50),
                    "center": (130, 125),
                },
                {
                    "class_name": "herb_aria",
                    "confidence": 0.60,
                    "bbox": (100, 500, 40, 40),
                    "center": (120, 520),
                },
                {
                    "class_name": "ore_vein",
                    "confidence": 0.20,
                    "bbox": (900, 100, 60, 50),
                    "center": (930, 125),
                },
            ]
        )
        self.scanner.set_yolo_detector(mock)
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        dets = self.scanner.scan_3d_yolo_materials(frame)

        # Only the 2 valid 3D resource detections should be returned
        self.assertEqual(len(dets), 2)
        # The low-confidence, visually unconfirmed detections were rejected
        # by the discrimination gate, not by class/position filtering.
        rejected_centers = {(130, 125), (120, 520), (930, 125)}
        self.assertFalse(
            rejected_centers & {(d.screen_x, d.screen_y) for d in dets}
        )
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
        # F-1542: assert the fusion BOOST relationally, not a hardcoded
        # constant: the fused confidence must exceed both inputs (0.84,
        # 0.86) and stay within the valid [0, 0.99] range. Pinning the exact
        # +0.08 increment (0.94) would lock the test to an implementation
        # constant instead of the fusion behaviour.
        self.assertGreater(h_node.confidence, max(0.84, 0.86))
        self.assertLessEqual(h_node.confidence, 0.99)
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
        """Verify ExemplarMemory rejects learned negative patches and refines positive patches.

        F-1298: exemplar teaching MUST NOT touch the global singleton or the
        real learned-data file (``dataset/learned_exemplars.json``). Every
        memory used here is an isolated ``ExemplarMemory`` backed by a
        temporary file, passed explicitly to ``_refine_candidates_with_exemplars``.

        F-1328: the exemplar effect is isolated by DIFFERENTIAL runs that
        differ in exactly one exemplar. A bare gray-rock candidate is dropped
        by the fake-rock / terrain gates even with an empty memory, so its
        disappearance alone proves nothing about the negative exemplar.
        Instead the negative gate is proven with a candidate that survives
        all other gates (the Aria patch, kept alive by its positive
        exemplar) and is dropped only when the SAME patch is additionally
        taught as a negative exemplar.
        """
        repo_learned = Path(__file__).resolve().parent / "dataset" / "learned_exemplars.json"
        before = repo_learned.read_bytes() if repo_learned.exists() else None

        # Teach a false gray rock patch as negative (kept for the realistic
        # frame setup; its drop is NOT asserted as an exemplar effect).
        # Textured — a flat patch is rejected by the K-3 degenerate gate.
        rock_patch = _textured_patch(118, (40, 40, 3), seed=9)
        # Teach Aria purple flower patch as positive
        aria_patch = _textured_patch((175, 45, 165), (40, 40, 3), seed=10)

        frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
        # Place rock at (300, 300)
        frame[280:320, 280:320] = rock_patch
        # Place Aria at (600, 400)
        frame[380:420, 580:620] = aria_patch

        def make_candidates() -> list[DetectedMaterial]:
            return [
                DetectedMaterial(
                    material_type="Unknown_Rock",
                    source="3d_world",
                    screen_x=300,
                    screen_y=300,
                    bbox=(280, 280, 40, 40),
                    rel_map_dx=0.0,
                    rel_map_dy=0.0,
                    estimated_distance_m=10.0,
                    confidence=0.70,
                ),
                DetectedMaterial(
                    material_type="Raw_Flower",
                    source="3d_yolo",
                    screen_x=600,
                    screen_y=400,
                    bbox=(580, 380, 40, 40),
                    rel_map_dx=0.0,
                    rel_map_dy=0.0,
                    estimated_distance_m=6.0,
                    confidence=0.65,
                ),
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Control 1 — empty isolated memory: without any exemplar the
            # Aria candidate is NOT refined to Aria (it is dropped by the
            # other gates), so a later Aria result must come from the
            # exemplar, not from the frame alone.
            empty_mem = ExemplarMemory(json_path=Path(tmpdir) / "empty.json")
            control = self.scanner._refine_candidates_with_exemplars(
                frame, make_candidates(), exemplar_memory=empty_mem
            )
            self.assertFalse(
                any(d.material_type == "Aria" for d in control),
                "Aria must not appear without a positive exemplar",
            )

            # Positive isolation — memory holding ONLY the Aria positive
            # exemplar: the Aria candidate survives, is reclassified to
            # Aria and its confidence is elevated.
            pos_mem = ExemplarMemory(json_path=Path(tmpdir) / "pos.json")
            pos_mem.add_positive_exemplar(aria_patch, class_id=0, class_name="herb_aria")
            refined = self.scanner._refine_candidates_with_exemplars(
                frame, make_candidates(), exemplar_memory=pos_mem
            )
            self.assertEqual(len(refined), 1)
            self.assertEqual(refined[0].material_type, "Aria")
            self.assertGreaterEqual(refined[0].confidence, 0.85)

            # Negative isolation — identical memory PLUS the same Aria patch
            # taught as a negative exemplar (class_id 14). The candidate
            # that survived every other gate in the run above is now
            # dropped; the ONLY difference between the runs is the negative
            # exemplar, so the drop is attributable to it alone.
            neg_mem = ExemplarMemory(json_path=Path(tmpdir) / "neg.json")
            neg_mem.add_positive_exemplar(aria_patch, class_id=0, class_name="herb_aria")
            neg_mem.add_negative_exemplar(aria_patch, class_id=14, class_name="false_rock_ground")
            is_neg, neg_sim, _ = neg_mem.is_negative_patch(aria_patch)
            self.assertTrue(is_neg, f"taught patch must match negative exemplar (sim={neg_sim})")
            dropped = self.scanner._refine_candidates_with_exemplars(
                frame, make_candidates(), exemplar_memory=neg_mem
            )
            self.assertEqual(
                dropped, [], "candidate must be dropped by the negative exemplar"
            )

            # The classic gray-rock negative exemplar is still teachable in
            # the isolated memory (no error, persisted only to the temp file).
            rock_mem = ExemplarMemory(json_path=Path(tmpdir) / "rock.json")
            rock_mem.add_negative_exemplar(rock_patch, class_id=14, class_name="false_rock_ground")
            self.assertTrue(rock_mem.is_negative_patch(rock_patch)[0])

        # F-1298 guard: the real learned-data file must be byte-identical.
        after = repo_learned.read_bytes() if repo_learned.exists() else None
        self.assertEqual(after, before, "test must not modify dataset/learned_exemplars.json")

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
        # F-1299: pass an ISOLATED, empty exemplar memory. The repo's
        # committed learned exemplars (dataset/learned_exemplars.json)
        # false-match this synthetic amber ore crop as herb_aria
        # (classify_patch sim ~0.79) and reclassify the YOLO Orichalcum
        # detection to Aria. The e2e pipeline itself is correct — YOLO
        # alone returns Orichalcum Ore at 0.88 — so the test must not
        # depend on whatever real learned data happens to be on disk.
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_mem = ExemplarMemory(json_path=Path(tmpdir) / "empty.json")
            results = self.scanner.scan_frame_for_materials(frame, exemplar_memory=empty_mem)

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
