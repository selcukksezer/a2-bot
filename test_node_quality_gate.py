# -*- coding: utf-8 -*-
"""Unit tests for the shared cartography / cleanup quality gate."""

import unittest

import numpy as np

from core.node_quality_gate import (
    HARD_WRONG_CLASS_IDS,
    KEEP_CLASS_IDS,
    conf_min_for_tier,
    evaluate_node_candidate,
    is_hard_wrong_class,
    is_keep_class,
    wrong_class_id,
)


class NodeQualityGateTests(unittest.TestCase):
    def test_tiers(self):
        self.assertAlmostEqual(conf_min_for_tier("aggressive"), 0.10)
        self.assertAlmostEqual(conf_min_for_tier("balanced"), 0.13)
        self.assertAlmostEqual(conf_min_for_tier("strict"), 0.22)
        self.assertAlmostEqual(conf_min_for_tier("nope"), 0.13)

    def test_keep_and_hard_wrong_sets(self):
        self.assertTrue(is_keep_class(0))
        self.assertTrue(is_keep_class(12))
        self.assertTrue(is_keep_class(22))
        self.assertTrue(is_keep_class(134))
        self.assertFalse(is_keep_class(10))
        self.assertTrue(is_hard_wrong_class(10))
        self.assertTrue(is_hard_wrong_class(7))
        self.assertFalse(is_hard_wrong_class(9))
        self.assertIn(10, HARD_WRONG_CLASS_IDS)
        self.assertIn(0, KEEP_CLASS_IDS)

    def test_wrong_class_id_floor(self):
        self.assertEqual(wrong_class_id(10, 0.13, 16), 10)
        self.assertIsNone(wrong_class_id(10, 0.12, 16))
        self.assertIsNone(wrong_class_id(9, 0.99, 16))
        self.assertIsNone(wrong_class_id(3, 0.99, 16))

    def test_evaluate_rejects_hard_wrong_model(self):
        patch = np.full((80, 80, 3), 90, dtype=np.uint8)

        class Det:
            num_classes = 16

            def top_class(self, _image):
                return 10, 0.2

        verdict = evaluate_node_candidate(
            patch,
            source_type="minimap_auto",
            resource_type="Gatherable Material",
            detector=Det(),
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(verdict.reject)
        self.assertIn("model", verdict.reasons)

    def test_evaluate_keeps_material_class(self):
        patch = np.full((80, 80, 3), 90, dtype=np.uint8)

        class Det:
            num_classes = 16

            def top_class(self, _image):
                return 3, 0.9

        verdict = evaluate_node_candidate(
            patch,
            source_type="minimap_auto",
            resource_type="Orichalcum Ore",
            detector=Det(),
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(verdict.accept)
        self.assertEqual(verdict.reasons, [])

    def test_evaluate_rejects_scroll(self):
        image = np.full((80, 80), 20, dtype=np.uint8)
        image[10:15, 8:72] = 220
        image[17:52, 6:74] = 160
        image[24:36, 28:40] = 30
        verdict = evaluate_node_candidate(
            image,
            source_type="minimap_auto",
            check_feather=False,
            check_monster=False,
            detector=None,
        )
        self.assertTrue(verdict.reject)
        self.assertIn("scroll", verdict.reasons)

    def test_contradiction_resource_vs_monster(self):
        patch = np.full((80, 80, 3), 40, dtype=np.uint8)
        verdict = evaluate_node_candidate(
            patch,
            source_type="3d_auto",
            resource_type="Gatherable Material",
            detector=None,
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
            model_verdict=(11, 0.25),
        )
        self.assertTrue(verdict.reject)
        self.assertTrue(
            "model" in verdict.reasons or "contradiction" in verdict.reasons
        )

    def test_verdict_is_wrong_property(self):
        patch = np.full((80, 80, 3), 90, dtype=np.uint8)

        class FakeDet:
            num_classes = 16

            def top_class(self, _img):
                return 10, 0.5

        bad = evaluate_node_candidate(
            patch,
            detector=FakeDet(),
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(bad.is_wrong)
        self.assertTrue(bad.reject)

        good = evaluate_node_candidate(
            patch,
            detector=None,
            check_feather=False,
            check_monster=False,
            check_exemplars=False,
            check_terrain=False,
        )
        self.assertFalse(good.is_wrong)
        self.assertTrue(good.accept)

    def test_negative_exemplar_rejection(self):
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mem = ExemplarMemory(json_path=tmp_path)
            neg_crop = np.full((80, 80, 3), 75, dtype=np.uint8)
            cv2_noise = np.random.default_rng(10).normal(0.0, 5.0, (80, 80, 3)).astype(np.float32)
            neg_crop = np.clip(neg_crop.astype(np.float32) + cv2_noise, 0, 255).astype(np.uint8)

            ok = mem.add_negative_exemplar(neg_crop, class_id=10, class_name="Canavar")
            self.assertTrue(ok)

            test_patch = neg_crop.copy()
            verdict = evaluate_node_candidate(
                test_patch,
                exemplar_memory=mem,
                check_feather=False,
                check_monster=False,
            )
            self.assertTrue(verdict.is_wrong)
            self.assertTrue(verdict.reject)
            self.assertIn("negative_exemplar", verdict.reasons)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_fake_gray_rock_ground_detection(self):
        import cv2
        rng = np.random.default_rng(4)
        small = 168.0 + rng.normal(0.0, 6.0, (8, 8))
        ground = cv2.resize(small.astype(np.float32), (80, 80), interpolation=cv2.INTER_CUBIC)
        ground_patch = np.clip(ground, 0, 255).astype(np.uint8)

        from core.node_quality_gate import is_fake_gray_rock_or_ground
        is_fake, conf, reason = is_fake_gray_rock_or_ground(ground_patch)
        self.assertTrue(is_fake)
        self.assertGreaterEqual(conf, 0.80)

        verdict = evaluate_node_candidate(
            ground_patch,
            resource_type="Orichalcum Ore",
            detector=None,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(verdict.is_wrong)
        self.assertTrue(verdict.reject)
        self.assertIn("terrain", verdict.reasons)

    def test_node_quality_gate_class_and_evaluation(self):
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.node_quality_gate import NodeQualityGate, get_node_quality_gate

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mem = ExemplarMemory(json_path=tmp_path)
            gate = NodeQualityGate(exemplar_memory=mem, dedup_radius_px=22.0)
            self.assertIsNotNone(gate)
            self.assertEqual(gate.dedup_radius_px, 22.0)

            # Singleton check
            gate_inst = NodeQualityGate.get_instance(exemplar_memory=mem)
            self.assertIs(gate_inst, NodeQualityGate.get_instance())

            get_gate = get_node_quality_gate(exemplar_memory=mem)
            self.assertIsNotNone(get_gate)

            # Test evaluation: learn negative patch
            neg_patch = np.full((80, 80, 3), 70, dtype=np.uint8)
            mem.add_negative_exemplar(neg_patch, class_id=14, class_name="obstacle")

            v_neg = gate.evaluate(neg_patch, source_type="3d_auto", resource_type="Raw_Node")
            self.assertTrue(v_neg.is_wrong)
            self.assertTrue(v_neg.reject)
            self.assertIn("negative_exemplar", v_neg.reasons)
            self.assertGreater(gate.total_rejected, 0)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_node_quality_gate_positive_exemplar_classification(self):
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.node_quality_gate import NodeQualityGate

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mem = ExemplarMemory(json_path=tmp_path)
            gate = NodeQualityGate(exemplar_memory=mem)

            # Learn positive Aria herb (greenish-cyan)
            aria_crop = np.zeros((80, 80, 3), dtype=np.uint8)
            aria_crop[:, :] = (120, 210, 40)
            mem.add_positive_exemplar(aria_crop, class_id=0, class_name="herb_aria")

            v = gate.evaluate(aria_crop, source_type="3d_auto", resource_type="Unknown_3D")
            self.assertTrue(v.accept)
            self.assertTrue(v.exemplar_matched)
            self.assertEqual(v.verified_resource_type, "Aria")
            self.assertGreaterEqual(v.confidence, 0.85)
            self.assertGreater(gate.total_accepted, 0)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_deduplicate_candidate_pins_merges_close_detections(self):
        from core.node_quality_gate import deduplicate_candidate_pins

        candidates = [
            {"id": 1, "map_x": 100.0, "map_y": 200.0, "confidence": 0.70, "resource_type": "Unknown", "sightings_count": 1},
            {"id": 2, "map_x": 105.0, "map_y": 203.0, "confidence": 0.85, "resource_type": "Orichalcum", "sightings_count": 2},
            {"id": 3, "map_x": 102.0, "map_y": 201.0, "confidence": 0.60, "resource_type": "Raw_Node", "sightings_count": 1},
            # Far candidate (distance > 22 px)
            {"id": 4, "map_x": 180.0, "map_y": 300.0, "confidence": 0.90, "resource_type": "Mela", "sightings_count": 1},
        ]

        merged = deduplicate_candidate_pins(candidates, dedup_radius_px=22.0)
        self.assertEqual(len(merged), 2)

        # Cluster of 1, 2, 3 should be merged into 1 pin
        cluster_pin = next(p for p in merged if p["map_x"] < 150.0)
        self.assertAlmostEqual(cluster_pin["map_x"], 102.7, delta=1.0)
        self.assertAlmostEqual(cluster_pin["map_y"], 201.6, delta=1.0)
        self.assertEqual(cluster_pin["resource_type"], "Orichalcum")
        self.assertEqual(cluster_pin["sightings_count"], 4)
        self.assertGreaterEqual(cluster_pin["confidence"], 0.85)

        # Far pin remains separate
        far_pin = next(p for p in merged if p["map_x"] > 150.0)
        self.assertEqual(far_pin["id"], 4)
        self.assertEqual(far_pin["resource_type"], "Mela")

    def test_deduplicate_candidate_pins_filters_against_existing_nodes(self):
        from core.node_quality_gate import deduplicate_candidate_pins

        existing_nodes = [
            {"id": 10, "map_x": 500.0, "map_y": 500.0, "resource_type": "Aria"},
        ]
        candidates = [
            {"id": 1, "map_x": 506.0, "map_y": 508.0, "confidence": 0.80, "resource_type": "Aria"},  # within 10 px of existing
            {"id": 2, "map_x": 600.0, "map_y": 600.0, "confidence": 0.88, "resource_type": "Gem"},   # far away
        ]

        merged = deduplicate_candidate_pins(candidates, existing_nodes=existing_nodes, dedup_radius_px=22.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], 2)
        self.assertEqual(merged[0]["resource_type"], "Gem")

    def test_filter_and_deduplicate_autonomous_pipeline(self):
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.node_quality_gate import NodeQualityGate

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mem = ExemplarMemory(json_path=tmp_path)
            gate = NodeQualityGate(exemplar_memory=mem, dedup_radius_px=22.0)

            # Learn positive Orichalcum
            ore_crop = np.zeros((80, 80, 3), dtype=np.uint8)
            ore_crop[:, :] = (180, 200, 40)
            mem.add_positive_exemplar(ore_crop, class_id=3, class_name="ore_orichalcum")

            # Learn negative terrain
            dirt_crop = np.zeros((80, 80, 3), dtype=np.uint8)
            dirt_crop[:, :] = (80, 85, 90)
            mem.add_negative_exemplar(dirt_crop, class_id=15, class_name="path_ground")

            # Candidate 1 & 2: nearby Orichalcum detections (5 px apart)
            # Candidate 3: false dirt patch (should be gated out by ExemplarMemory)
            candidates = [
                {"id": 1, "map_x": 200.0, "map_y": 200.0, "patch_img": ore_crop, "confidence": 0.65, "resource_type": "Unknown"},
                {"id": 2, "map_x": 204.0, "map_y": 202.0, "patch_img": ore_crop.copy(), "confidence": 0.75, "resource_type": "Raw_Node"},
                {"id": 3, "map_x": 400.0, "map_y": 400.0, "patch_img": dirt_crop, "confidence": 0.80, "resource_type": "Ground"},
            ]

            results = gate.filter_and_deduplicate(candidates)
            # Candidate 3 is rejected; candidates 1 & 2 are merged into 1 pin
            self.assertEqual(len(results), 1)
            pin = results[0]
            self.assertEqual(pin["resource_type"], "Orichalcum")
            self.assertAlmostEqual(pin["map_x"], 202.2, delta=1.0)
            self.assertAlmostEqual(pin["map_y"], 201.1, delta=1.0)
            self.assertGreaterEqual(pin["confidence"], 0.85)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()


