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


class _MeanBasedDetector:
    """Input-sensitive stand-in detector (F-1546).

    The old stubs returned a constant class no matter what patch they were
    given, so the gate tests could pass without the detector ever looking at
    the input. This one derives its verdict from the patch itself (dark
    patches read as the hard-wrong class 10, bright patches as keep class 3)
    and records every image it was asked about.
    """

    num_classes = 16

    def __init__(self) -> None:
        self.seen: list = []

    def top_class(self, image):
        self.seen.append(np.asarray(image).copy())
        mean = float(np.mean(image))
        return (10, 0.9) if mean < 50 else (3, 0.9)


class NodeQualityGateTests(unittest.TestCase):
    def test_tiers(self):
        self.assertAlmostEqual(conf_min_for_tier("aggressive"), 0.10)
        self.assertAlmostEqual(conf_min_for_tier("balanced"), 0.13)
        self.assertAlmostEqual(conf_min_for_tier("strict"), 0.22)
        self.assertAlmostEqual(conf_min_for_tier("nope"), 0.13)

    def test_keep_and_hard_wrong_sets(self):
        self.assertTrue(is_keep_class(0))
        self.assertTrue(is_keep_class(12))
        # D20: the blanket range(22, 200) keep was removed — every id >= 22
        # counted as "keep", including unknown ids like 134. Keep is now
        # the explicit material set {0..6, 12}; 22 (hp_potion) and 134
        # are not keep classes.
        self.assertFalse(is_keep_class(22))
        self.assertFalse(is_keep_class(134))
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
        # Dark patch -> the input-sensitive detector reads hard-wrong class 10.
        patch = np.full((80, 80, 3), 20, dtype=np.uint8)

        det = _MeanBasedDetector()

        verdict = evaluate_node_candidate(
            patch,
            source_type="minimap_auto",
            resource_type="Gatherable Material",
            detector=det,
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(verdict.reject)
        self.assertIn("model", verdict.reasons)
        # The detector really looked at this patch (F-1546).
        self.assertEqual(len(det.seen), 1)
        self.assertTrue(np.array_equal(det.seen[0], patch))

    def test_evaluate_keeps_material_class(self):
        # Bright patch -> the input-sensitive detector reads keep class 3.
        patch = np.full((80, 80, 3), 200, dtype=np.uint8)

        det = _MeanBasedDetector()

        verdict = evaluate_node_candidate(
            patch,
            source_type="minimap_auto",
            resource_type="Orichalcum Ore",
            detector=det,
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(verdict.accept)
        self.assertEqual(verdict.reasons, [])
        # Verdict.class_id carries the hard-wrong id (None on accept); the
        # detector's own class is raw_class_id.
        self.assertEqual(verdict.raw_class_id, 3)
        self.assertAlmostEqual(verdict.confidence, 0.9)
        self.assertEqual(len(det.seen), 1)
        self.assertTrue(np.array_equal(det.seen[0], patch))

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
        # Dark patch -> input-sensitive detector reads hard-wrong class 10.
        patch = np.full((80, 80, 3), 20, dtype=np.uint8)
        det = _MeanBasedDetector()

        bad = evaluate_node_candidate(
            patch,
            detector=det,
            check_feather=False,
            check_monster=False,
        )
        self.assertTrue(bad.is_wrong)
        self.assertTrue(bad.reject)
        self.assertEqual(len(det.seen), 1)

        # "Good" verdict must come from a passing check, not from the
        # default-accept path: bright patch + a consulted detector, with all
        # real checks enabled (F-1423). The old test disabled every check and
        # asserted the default accept.
        bright = np.full((80, 80, 3), 200, dtype=np.uint8)
        det2 = _MeanBasedDetector()
        good = evaluate_node_candidate(
            bright,
            source_type="minimap_auto",
            resource_type="Orichalcum Ore",
            detector=det2,
            conf_min=0.13,
        )
        self.assertFalse(good.is_wrong)
        self.assertTrue(good.accept)
        self.assertEqual(good.reasons, [])
        self.assertEqual(good.raw_class_id, 3)
        self.assertAlmostEqual(good.confidence, 0.9)
        # The detector was actually consulted on this exact patch.
        self.assertEqual(len(det2.seen), 1)
        self.assertTrue(np.array_equal(det2.seen[0], bright))

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

    def test_wrong_node_marker_negative_is_not_a_terrain_prototype(self):
        # v19/F12 (P1-8 class-15 kovası): a (15, "wrong_node_*") negative
        # is UNKNOWN content under the class-15 storage bucket — it must
        # not be harvested as a terrain prototype. Step-1 negative
        # matching still suppresses near-duplicates (that is the point of
        # the record); only the "exemplar_terrain" reason path must not
        # fire for unknown junk.
        #
        # Discriminator: candidate class 10 vs stored class 15 scales
        # the step-1 match by 0.90 (1.00 -> 0.90 < 0.95 gate), so step 1
        # cannot fire; step 2 compares unscaled, so before the fix the
        # harvested prototype condemned the patch as "exemplar_terrain".
        # The patch is saturated blue + textured, so step 3 (flat gray)
        # and the canonical gray prototypes cannot fire either.
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.node_quality_gate import is_fake_gray_rock_or_ground

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mem = ExemplarMemory(json_path=tmp_path)
            patch = _textured_patch((200, 40, 30), (80, 80, 3), seed=7)

            ok = mem.add_negative_exemplar(
                patch,
                class_id=15,
                class_name="wrong_node_purge",
                source="node_deletion",
            )
            self.assertTrue(ok)
            # Marker survives the add (v19/F12): not rewritten to path_ground.
            _sim, matched = mem.match_negative(patch)
            self.assertIsNotNone(matched)
            self.assertEqual(matched["class_id"], 15)
            self.assertEqual(matched["class_name"], "wrong_node_purge")

            is_fake, _conf, reason = is_fake_gray_rock_or_ground(
                patch, exemplar_memory=mem, candidate_class_id=10
            )
            self.assertFalse(is_fake)
            self.assertEqual(reason, "")
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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

            # Singleton check: the old test only asserted identity and ignored
            # that the singleton carries process-wide state (counters and the
            # last exemplar memory) across tests (F-1545). Reset it first so
            # this test observes fresh state, then prove the shared state is
            # visible, and that a reset yields a fresh instance.
            previous_instance = NodeQualityGate._instance
            NodeQualityGate._instance = None
            try:
                gate_inst = NodeQualityGate.get_instance(exemplar_memory=mem)
                self.assertIs(gate_inst, NodeQualityGate.get_instance())
                # Fresh singleton: no evaluations accumulated yet.
                self.assertEqual(gate_inst.total_evaluated, 0)
                self.assertEqual(gate_inst.total_accepted, 0)
                self.assertEqual(gate_inst.total_rejected, 0)

                # State is shared process-wide: an evaluation through one
                # reference is visible through the other.
                neg_patch = _textured_patch(70, (80, 80, 3), seed=7)
                mem.add_negative_exemplar(neg_patch, class_id=14, class_name="obstacle")
                v_neg = gate_inst.evaluate(neg_patch, source_type="3d_auto", resource_type="Raw_Node")
                self.assertTrue(v_neg.is_wrong)
                self.assertTrue(v_neg.reject)
                self.assertIn("negative_exemplar", v_neg.reasons)
                self.assertGreater(gate_inst.total_rejected, 0)
                again = NodeQualityGate.get_instance()
                self.assertIs(again, gate_inst)
                self.assertEqual(again.total_rejected, gate_inst.total_rejected)
                self.assertEqual(again.total_evaluated, gate_inst.total_evaluated)

                # Resetting the singleton yields a genuinely fresh instance
                # (zeroed counters), proving the leak above is resettable
                # and not hidden by the test.
                NodeQualityGate._instance = None
                fresh = NodeQualityGate.get_instance(exemplar_memory=mem)
                self.assertIsNot(fresh, gate_inst)
                self.assertEqual(fresh.total_evaluated, 0)
                self.assertEqual(fresh.total_rejected, 0)
            finally:
                NodeQualityGate._instance = previous_instance

            get_gate = get_node_quality_gate(exemplar_memory=mem)
            self.assertIsNotNone(get_gate)
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

            # Learn positive Aria herb (greenish-cyan; textured — a flat
            # patch is rejected by the K-3 degenerate-crop gate)
            aria_crop = _textured_patch((120, 210, 40), (80, 80, 3), seed=8)
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

            # Learn positive Orichalcum (textured — see _textured_patch)
            ore_crop = _textured_patch((180, 200, 40), (80, 80, 3), seed=6)
            mem.add_positive_exemplar(ore_crop, class_id=3, class_name="ore_orichalcum")

            # Learn negative terrain
            dirt_crop = _textured_patch((80, 85, 90), (80, 80, 3), seed=7)
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


    def test_deduplicate_grid_and_running_sums_match_pre_change(self):
        """D21/I2: grid existing-filter + running-sum clustering must return
        byte-identical pins to the pre-change O(C x E) / rebuild-the-mean
        implementation (copied verbatim below as the reference)."""
        import copy
        import math
        import random

        from core.node_quality_gate import (
            SAME_POINT_PX,
            _get_candidate_attr,
            _get_candidate_coords,
            _get_candidate_image,
            _safe_conf,
            _safe_count,
            _set_candidate_attr,
            deduplicate_candidate_pins,
        )

        def _reference(candidates, existing_nodes=None, dedup_radius_px=SAME_POINT_PX):
            # Verbatim copy of deduplicate_candidate_pins as of e9794df.
            if not candidates:
                return []
            radius = max(1.0, _safe_conf(dedup_radius_px, SAME_POINT_PX))
            filtered_cands = []
            if existing_nodes:
                existing_coords = [_get_candidate_coords(e) for e in existing_nodes]
                for c in candidates:
                    cx, cy = _get_candidate_coords(c)
                    is_dup = False
                    for ex, ey in existing_coords:
                        if math.hypot(cx - ex, cy - ey) <= radius:
                            is_dup = True
                            break
                    if not is_dup:
                        filtered_cands.append(c)
            else:
                filtered_cands = list(candidates)
            if not filtered_cands:
                return []
            clusters = []
            for cand in filtered_cands:
                cx, cy = _get_candidate_coords(cand)
                matched_cluster = None
                for cl in clusters:
                    cl_xs = [_get_candidate_coords(item)[0] for item in cl]
                    cl_ys = [_get_candidate_coords(item)[1] for item in cl]
                    mean_x = sum(cl_xs) / len(cl_xs)
                    mean_y = sum(cl_ys) / len(cl_ys)
                    if math.hypot(cx - mean_x, cy - mean_y) <= radius:
                        matched_cluster = cl
                        break
                if matched_cluster is not None:
                    matched_cluster.append(cand)
                else:
                    clusters.append([cand])
            merged_pins = []
            for cl in clusters:
                if len(cl) == 1:
                    merged_pins.append(cl[0])
                    continue
                cl.sort(key=lambda item: _safe_conf(_get_candidate_attr(item, "confidence", 0.0), 0.0), reverse=True)
                best_item = cl[0]
                total_w = 0.0
                sum_x = 0.0
                sum_y = 0.0
                for item in cl:
                    w = max(0.1, _safe_conf(_get_candidate_attr(item, "confidence", 1.0), 1.0))
                    ix, iy = _get_candidate_coords(item)
                    sum_x += ix * w
                    sum_y += iy * w
                    total_w += w
                avg_x = round(sum_x / total_w, 2)
                avg_y = round(sum_y / total_w, 2)
                total_sightings = sum(_safe_count(_get_candidate_attr(item, "sightings_count", 1), 1) for item in cl)
                max_stack = max(_safe_count(_get_candidate_attr(item, "stack_count", 1), 1) for item in cl)
                best_conf = _safe_conf(_get_candidate_attr(best_item, "confidence", 0.7), 0.7)
                merged_conf = min(0.99, round(best_conf + 0.04 * (len(cl) - 1), 3))
                best_rtype = _get_candidate_attr(best_item, "resource_type", "")
                for item in cl:
                    rtype = str(_get_candidate_attr(item, "resource_type", "") or "")
                    if rtype and rtype.lower() not in ("resource", "raw_node", "unknown", "unknown_3d", "sprout"):
                        best_rtype = rtype
                        break
                best_img = None
                for item in cl:
                    img = _get_candidate_image(item)
                    if img is not None:
                        best_img = img
                        break
                if isinstance(best_item, dict):
                    merged_pin = dict(best_item)
                    merged_pin["map_x"] = avg_x
                    merged_pin["map_y"] = avg_y
                    merged_pin["confidence"] = merged_conf
                    merged_pin["sightings_count"] = total_sightings
                    merged_pin["stack_count"] = max_stack
                    merged_pin["resource_type"] = best_rtype
                    if best_img is not None:
                        merged_pin["patch_img"] = best_img
                    merged_pins.append(merged_pin)
                else:
                    _set_candidate_attr(best_item, "map_x", avg_x)
                    _set_candidate_attr(best_item, "map_y", avg_y)
                    _set_candidate_attr(best_item, "confidence", merged_conf)
                    _set_candidate_attr(best_item, "sightings_count", total_sightings)
                    _set_candidate_attr(best_item, "stack_count", max_stack)
                    _set_candidate_attr(best_item, "resource_type", best_rtype)
                    if best_img is not None and hasattr(best_item, "patch_img"):
                        _set_candidate_attr(best_item, "patch_img", best_img)
                    merged_pins.append(best_item)
            return merged_pins

        def _signature(pins):
            return [
                (
                    p.get("id"),
                    p.get("map_x"),
                    p.get("map_y"),
                    p.get("confidence"),
                    p.get("sightings_count"),
                    p.get("stack_count"),
                    p.get("resource_type"),
                )
                for p in pins
            ]

        rng = random.Random(20261108)
        candidates = []
        for i in range(300):
            candidates.append(
                {
                    "id": i,
                    "map_x": rng.uniform(0, 2000),
                    "map_y": rng.uniform(0, 2000),
                    "confidence": rng.uniform(0.5, 0.95),
                    "sightings_count": rng.randint(1, 3),
                    "stack_count": rng.randint(1, 2),
                    "resource_type": rng.choice(["Aria", "Mela", "Unknown", "Raw_Node"]),
                }
            )
        # Dense clusters force multi-way merges and running-sum averaging.
        for j, (bx, by) in enumerate([(500.0, 500.0), (1500.0, 200.0), (100.0, 1800.0)]):
            for k in range(12):
                candidates.append(
                    {
                        "id": 1000 + j * 20 + k,
                        "map_x": bx + rng.uniform(-8, 8),
                        "map_y": by + rng.uniform(-8, 8),
                        "confidence": rng.uniform(0.5, 0.95),
                        "sightings_count": 1,
                        "stack_count": 1,
                        "resource_type": "Orichalcum",
                    }
                )
        # Cell-border boundary: exactly 18 px apart across the 18 px grid line.
        candidates.append({"id": 2000, "map_x": 17.9, "map_y": 50.0, "confidence": 0.8,
                           "sightings_count": 1, "stack_count": 1, "resource_type": "Aria"})
        # Coordless candidate clusters at (0, 0) exactly like before the change.
        candidates.append({"id": 2001, "confidence": 0.8,
                           "sightings_count": 1, "stack_count": 1, "resource_type": "Aria"})
        candidates.append({"id": 2002, "map_x": 3.0, "map_y": 4.0, "confidence": 0.7,
                           "sightings_count": 1, "stack_count": 1, "resource_type": "Mela"})

        existing = []
        for i in range(150):
            existing.append(
                {"id": 5000 + i, "map_x": rng.uniform(0, 2000),
                 "map_y": rng.uniform(0, 2000), "resource_type": "Aria"}
            )
        existing.append({"id": 6000, "map_x": 35.9, "map_y": 50.0, "resource_type": "Aria"})
        existing.append({"id": 6001, "map_x": 100.0, "map_y": 100.0, "resource_type": "Aria"})
        # Coordless existing node: pre-change behaviour reads it as (0, 0),
        # so the (5, 5) candidate below is filtered; locked in, not endorsed.
        existing.append({"id": 6002, "resource_type": "Aria"})
        candidates.append({"id": 2003, "map_x": 5.0, "map_y": 5.0, "confidence": 0.8,
                           "sightings_count": 1, "stack_count": 1, "resource_type": "Gem"})
        candidates.append({"id": 2004, "map_x": 100.0, "map_y": 118.0001, "confidence": 0.8,
                           "sightings_count": 1, "stack_count": 1, "resource_type": "Gem"})

        got18 = None
        for radius in (SAME_POINT_PX, 22.0):
            got = deduplicate_candidate_pins(
                copy.deepcopy(candidates), existing_nodes=copy.deepcopy(existing),
                dedup_radius_px=radius,
            )
            want = _reference(
                copy.deepcopy(candidates), existing_nodes=copy.deepcopy(existing),
                dedup_radius_px=radius,
            )
            self.assertEqual(_signature(got), _signature(want), f"radius={radius}")
            if radius == SAME_POINT_PX:
                got18 = got
        # Without existing nodes, too.
        got = deduplicate_candidate_pins(copy.deepcopy(candidates))
        want = _reference(copy.deepcopy(candidates))
        self.assertEqual(_signature(got), _signature(want))
        # Radius 18: the boundary candidate (17.9, 50) is filtered by
        # existing (35.9, 50), exactly 18 px away across the grid line.
        self.assertNotIn(2000, [p.get("id") for p in got18])
        # The (5, 5) candidate is filtered by the coordless existing node.
        self.assertNotIn(2003, [p.get("id") for p in got18])
        # (100, 118.0001) is just over 18 px from existing (100, 100): kept.
        self.assertIn(2004, [p.get("id") for p in got18])


if __name__ == "__main__":
    unittest.main()


