# -*- coding: utf-8 -*-
"""Synthetic wrong-node scan. Does not read or write routes/Altgard/region_data.json."""

import glob
import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import cv2
import numpy as np

from core.node_quality_gate import is_hard_wrong_class, is_keep_class
from core.route_manager import RouteManager
from core.vision_filters import is_parchment_scroll_patch
from core.wrong_node_scan import (
    CLASS_LABELS_TR,
    MODEL_CONF_MIN,
    SAME_POINT_PX,
    classify_wrong_nodes,
    format_blocked_status_tr,
    format_removed_status_tr,
    load_gatherion_detector,
    names_are_gatherion_prefix,
    names_match_dataset,
    read_dataset_class_names,
    scan_would_exceed_half,
    wrong_class_id,
    _top_class_from_output,
)


def _scroll_patch() -> np.ndarray:
    """80x80 parchment that the existing scroll test accepts."""
    image = np.full((80, 80), 20, dtype=np.uint8)
    image[10:15, 8:72] = 220
    image[17:52, 6:74] = 160
    image[24:36, 28:40] = 30
    return image


def _ore_patch(seed: int) -> np.ndarray:
    """Smooth textured rock. Not a scroll, not flat ground, not UI text."""
    rng = np.random.default_rng(seed)
    small = rng.normal(120.0, 38.0, (16, 16)).astype(np.float32)
    ore = cv2.resize(small, (80, 80), interpolation=cv2.INTER_CUBIC)
    return np.clip(ore, 30, 175).astype(np.uint8)


def _ground_patch() -> np.ndarray:
    rng = np.random.default_rng(4)
    small = 168.0 + rng.normal(0.0, 6.0, (8, 8))
    ground = cv2.resize(small.astype(np.float32), (80, 80), interpolation=cv2.INTER_CUBIC)
    return np.clip(ground, 0, 255).astype(np.uint8)


def _node(node_id, x, y, seq, gathered=0, not_gatherable=False, resource="Orichalcum Ore"):
    return {
        "id": node_id,
        "map_x": x,
        "map_y": y,
        "seq_index": seq,
        "gathered_count": gathered,
        "last_visited_timestamp": 0.0,
        "last_label": "hiç",
        "not_gatherable": not_gatherable,
        "resource_type": resource,
    }


def _fixed(verdict, num_classes=16):
    class _Detector:
        def __init__(self) -> None:
            self.num_classes = num_classes

        def top_class(self, _image):
            return verdict

    return _Detector()


@contextmanager
def _temp_exemplar_json():
    """Empty ExemplarMemory backing file, always removed afterwards.

    Replaces the repeated NamedTemporaryFile + try/finally + os.remove
    blocks (3 occurrences)."""
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@contextmanager
def _temp_dir():
    """Scratch directory, always removed afterwards.

    Replaces the repeated mkdtemp blocks (5 occurrences) — three of which
    never cleaned up, leaking a directory under the system temp dir per run.
    """
    tmp = tempfile.mkdtemp()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class WrongNodeScanTests(unittest.TestCase):
    def _duplicate_radius_case(self, dx, detector, expected_ids):
        """Two nodes `dx` px apart on the x axis; assert which ids are
        flagged. Shared by the 18 px / 4 px / 36 px radius tests so the
        node setup is written once; each test keeps its own assertions."""
        ore = _ore_patch(9)
        nodes = [
            _node(11, dx, 0, seq=2, gathered=0),
            _node(10, 0, 0, seq=1, gathered=0),
        ]
        wrong = classify_wrong_nodes(
            nodes, {10: ore, 11: ore.copy()}, detector=detector
        )
        self.assertEqual([item.node_id for item in wrong], expected_ids)
        return wrong

    def test_fake_detector_gates_on_class_and_confidence(self):
        image = _ore_patch(1)
        node = _node(1, 0, 0, seq=1, gathered=0)
        removed = classify_wrong_nodes([node], {1: image}, detector=_fixed((10, 0.8)))
        self.assertEqual([item.node_id for item in removed], [1])
        self.assertEqual(removed[0].class_id, 10)
        self.assertEqual(removed[0].primary_reason, "model")

        kept_mela = classify_wrong_nodes([node], {1: image}, detector=_fixed((1, 0.9)))
        self.assertEqual(kept_mela, [])

        # Below calibrated floor (~0.13) keeps the node.
        kept_low = classify_wrong_nodes([node], {1: image}, detector=_fixed((10, 0.05)))
        self.assertEqual(kept_low, [])

        # Real patch-scale confidence now deletes hard-wrong classes.
        removed_calibrated = classify_wrong_nodes(
            [node], {1: image}, detector=_fixed((10, 0.15))
        )
        self.assertEqual([item.node_id for item in removed_calibrated], [1])

        kept_silent = classify_wrong_nodes([node], {1: image}, detector=_fixed(None))
        self.assertEqual(kept_silent, [])

        self.assertEqual(node["gathered_count"], 0)
        self.assertEqual(node["last_label"], "hiç")

    def test_real_onnx_detector_verdict_flows_into_classification(self):
        """F-1447: classification must also run against the REAL production
        detector (GatherionOnnxDetector over models/yolov8n_gatherion.onnx),
        not only the _fixed() fake — the fake can never catch model-path
        regressions (verdict parsing, num_classes gating, floor logic)."""
        try:
            detector = load_gatherion_detector()
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            self.skipTest(f"gerçek gatherion ONNX modeli yüklenemedi: {exc}")

        # The production detector matches the shipped data.yaml taxonomy.
        root = os.path.dirname(os.path.abspath(__file__))
        dataset_names = read_dataset_class_names(os.path.join(root, "dataset", "data.yaml"))
        self.assertTrue(detector.matches_dataset(dataset_names))
        self.assertEqual(detector.num_classes, len(dataset_names))

        ore = _ore_patch(1)
        verdict = detector.top_class(ore)
        self.assertIsNotNone(verdict, "real detector must return a verdict on a patch")
        class_id, confidence = verdict
        self.assertIsInstance(class_id, int)
        self.assertGreaterEqual(class_id, 0)
        self.assertLess(class_id, detector.num_classes)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        # Deterministic on the same patch.
        self.assertEqual(detector.top_class(ore), verdict)

        # The real verdict on this synthetic patch is far below the
        # calibrated floor, so it can never be a "model" delete reason.
        self.assertLess(confidence, MODEL_CONF_MIN)
        self.assertIsNone(wrong_class_id(class_id, confidence, detector.num_classes))

        # End-to-end: the real verdict flows through classify_wrong_nodes.
        # Whatever the CV/exemplar gates do to this patch, the model path
        # itself must not delete the node on a below-floor verdict.
        node = _node(1, 0, 0, seq=1, gathered=0)
        wrong = classify_wrong_nodes([node], {1: ore}, detector=detector)
        for item in wrong:
            self.assertNotIn("model", item.reasons)

        # Same guarantee for a ground patch (terrain gate may fire, model must not).
        ground = _ground_patch()
        wrong_ground = classify_wrong_nodes(
            [_node(99, 50, 50, seq=1, resource="Orichalcum Ore")],
            {99: ground},
            detector=detector,
        )
        for item in wrong_ground:
            self.assertNotIn("model", item.reasons)

    def test_threshold_and_unknown_classes(self):
        self.assertLessEqual(MODEL_CONF_MIN, 0.15)
        self.assertIsNone(wrong_class_id(7, MODEL_CONF_MIN, 16))
        self.assertIsNone(wrong_class_id(1, 0.99, 16))
        self.assertIsNone(wrong_class_id(12, 0.99, 16))
        self.assertEqual(wrong_class_id(15, 0.8, 16), 15)
        self.assertIsNone(wrong_class_id(15, 0.8, 12))
        self.assertIsNone(wrong_class_id(None, None, 16))
        # Class 9 alone is not a hard-wrong delete reason.
        self.assertIsNone(wrong_class_id(9, 0.9, 16))
        # v19/F12 — 22..26 LİMBO (v17 sözleşmesi, kilitli): hp_potion,
        # mp_potion, buff_scroll, kinah, player_character ne keep ne
        # hard-wrong kümesinde; wrong_class_id hepsi için None döner.
        # (Eski "Essences stay keep" yorumu yanlıştır — 22 keep
        # değildir, hiçbir kümede olmadığı için None döner.)
        # 22..26'yı hard-wrong/keep yapmak Selçuk kararıdır (§7-A'nın
        # 28 çatışmasında yok); bu kilit, taksonomi türetmesinin
        # sessizce genişletmesini engeller (v18/P1-1 regresyonu).
        for cid_consumable in (22, 23, 24):
            self.assertEqual(wrong_class_id(cid_consumable, 0.9, 135), cid_consumable)
            self.assertTrue(is_hard_wrong_class(cid_consumable))
            self.assertFalse(is_keep_class(cid_consumable))
        for cid_limbo in (25, 26):
            self.assertIsNone(wrong_class_id(cid_limbo, 0.9, 135))
            self.assertFalse(is_keep_class(cid_limbo))
            self.assertFalse(is_hard_wrong_class(cid_limbo))
        for cid_keep_mat in (7, 8):
            self.assertIsNone(wrong_class_id(cid_keep_mat, 0.9, 135))
            self.assertTrue(is_keep_class(cid_keep_mat))
            self.assertFalse(is_hard_wrong_class(cid_keep_mat))

    def test_scroll_and_duplicate_radius(self):
        scroll = _scroll_patch()
        self.assertTrue(is_parchment_scroll_patch(scroll))
        older = _ore_patch(1)
        ground = _ground_patch()
        nodes = [
            _node(1, 0, 0, seq=1, gathered=5),
            _node(2, 4, 0, seq=2, gathered=0),
            _node(3, 36, 0, seq=3, gathered=0),
            _node(4, 400, 400, seq=4, gathered=0, resource="Mela"),
            _node(5, 800, 800, seq=5, gathered=0),
            _node(6, 900, 100, seq=6, gathered=0, not_gatherable=True),
            _node(7, 200, 500, seq=7, gathered=0),
        ]
        images = {
            1: older,
            2: older.copy(),
            3: older.copy(),
            4: _ore_patch(3),
            5: scroll,
            6: older.copy(),
            7: ground,
        }
        wrong = {
            item.node_id: item
            for item in classify_wrong_nodes(nodes, images, detector=_fixed((1, 0.9)))
        }
        self.assertEqual(set(wrong), {2, 5})
        self.assertIn("duplicate", wrong[2].reasons)
        self.assertEqual(wrong[5].primary_reason, "scroll")
        self.assertNotIn(1, wrong)
        self.assertNotIn(3, wrong)
        self.assertNotIn(4, wrong)
        self.assertNotIn(6, wrong)
        self.assertNotIn(7, wrong)

    def test_eighteen_px_drops_newer_only(self):
        wrong = self._duplicate_radius_case(
            SAME_POINT_PX, detector=_fixed((3, 0.95)), expected_ids=[11]
        )
        self.assertEqual(wrong[0].primary_reason, "duplicate")

    def test_exact_four_px_drops_newer_only(self):
        wrong = self._duplicate_radius_case(
            4, detector=_fixed((3, 0.95)), expected_ids=[11]
        )
        self.assertEqual(wrong[0].primary_reason, "duplicate")

    def test_thirty_six_px_copy_is_kept(self):
        self._duplicate_radius_case(36, detector=_fixed(None), expected_ids=[])

    def test_output_argmax_is_top_class(self):
        raw = np.zeros((1, 16, 3), dtype=np.float32)
        raw[0, 4 + 10, 1] = 0.8
        raw[0, 4 + 1, 2] = 0.3
        # Centers near letterbox middle so weighting does not flip the winner.
        raw[0, 0, 1] = 320.0
        raw[0, 1, 1] = 320.0
        raw[0, 0, 2] = 320.0
        raw[0, 1, 2] = 320.0
        class_id, confidence = _top_class_from_output(raw, 12)
        self.assertEqual(class_id, 10)
        self.assertAlmostEqual(confidence, 0.8)

    def test_dataset_name_match_rejects_partial_and_coco(self):
        root = os.path.dirname(os.path.abspath(__file__))
        names = read_dataset_class_names(os.path.join(root, "dataset", "data.yaml"))
        self.assertGreaterEqual(len(names), 22)
        self.assertEqual(names[0], "herb_aria")
        self.assertEqual(names[10], "monster_3d")  # canonical: dataset/data.yaml:17, CANONICAL_NAMES[10]
        self.assertEqual(names[15], "path_ground")
        self.assertEqual(names[16], "player_hp_bar")
        # NOTE (S-3101): kanonik 27 sınıflı taksonomide essence_* sınıfı yoktur
        # (dataset/data.yaml, CANONICAL_NAMES) — bayat essence beklentisi yerine
        # data.yaml'da gerçekten olan sınıflar spot-check edilir.
        self.assertIn("crystal_blood", names)
        self.assertIn("buff_scroll", names)
        self.assertTrue(names_match_dataset(names, names))
        self.assertFalse(names_match_dataset(names[:12], names))
        self.assertTrue(names_are_gatherion_prefix(names[:12], names))
        self.assertTrue(names_are_gatherion_prefix(names[:16], names))
        self.assertFalse(names_are_gatherion_prefix(["person", "bicycle"], names))
        self.assertIn(10, CLASS_LABELS_TR)

    def test_status_lists_top_wrong_classes(self):
        from core.wrong_node_scan import WrongNode

        wrong = [
            WrongNode(1, ["model"], class_id=10, confidence=0.8),
            WrongNode(2, ["model"], class_id=10, confidence=0.9),
            WrongNode(3, ["model"], class_id=7, confidence=0.5),
        ]
        text = format_removed_status_tr(wrong, backup_path="routes/Altgard/region_data.backup.json")
        self.assertIn("3 nokta sildi", text)
        self.assertIn("Canavar 2", text)
        self.assertIn("F Gather 1", text)
        self.assertIn("region_data.backup.json", text)
        blocked = format_blocked_status_tr(wrong, total=4)
        self.assertIn("3/4", blocked)
        self.assertIn("Dosya yazılmadı", blocked)
        self.assertIn("Canavar 2", blocked)

    def test_half_guard(self):
        self.assertFalse(scan_would_exceed_half(1, 3))
        self.assertTrue(scan_would_exceed_half(2, 3))
        self.assertFalse(scan_would_exceed_half(0, 0))

    def test_tier_aggressive_deletes_lower_conf(self):
        image = _ore_patch(2)
        node = _node(1, 0, 0, seq=1)
        kept_strict = classify_wrong_nodes(
            [node], {1: image}, detector=_fixed((10, 0.12)), tier="strict"
        )
        self.assertEqual(kept_strict, [])
        removed_agg = classify_wrong_nodes(
            [node], {1: image}, detector=_fixed((10, 0.12)), tier="aggressive"
        )
        self.assertEqual([item.node_id for item in removed_agg], [1])

    def test_winerror_5_does_not_save(self):
        with _temp_dir() as tmp:
            manager = RouteManager(base_dir=tmp)
            self.assertTrue(manager.create_region("Altgard"))
            node = manager.add_node(region="Altgard", route="Dredgion", map_x=10, map_y=10)
            with open(
                os.path.join(tmp, "Altgard", "region_data.json"), encoding="utf-8"
            ) as handle:
                before = handle.read()
            denied = OSError("denied")
            denied.winerror = 5
            with patch("core.route_manager.os.replace", side_effect=denied):
                ok = manager.delete_nodes([node.id], region="Altgard")
            self.assertFalse(ok)
            self.assertEqual(getattr(manager.last_save_error, "winerror", None), 5)
            self.assertIsNotNone(manager.get_node(node.id, region="Altgard"))
            with open(
                os.path.join(tmp, "Altgard", "region_data.json"), encoding="utf-8"
            ) as handle:
                after = handle.read()
            self.assertEqual(before, after)
            empty_ok = manager.delete_nodes([], region="Altgard")
            self.assertTrue(empty_ok)

    def test_wrong_node_is_wrong_attribute(self):
        from core.wrong_node_scan import WrongNode

        wn = WrongNode(101, ["model"], class_id=10, confidence=0.85)
        self.assertTrue(wn.is_wrong)
        self.assertEqual(wn.node_id, 101)

    def test_fake_gray_rock_ground_identified_as_wrong(self):
        ground = _ground_patch()
        node = _node(99, 50, 50, seq=1, resource="Orichalcum Ore")
        wrong = classify_wrong_nodes([node], {99: ground}, detector=None)
        self.assertEqual(len(wrong), 1)
        self.assertEqual(wrong[0].node_id, 99)
        self.assertTrue(wrong[0].is_wrong)
        self.assertIn("terrain", wrong[0].reasons)

    def test_learned_negative_exemplar_purges_node(self):
        from dataset_factory.exemplar_memory import ExemplarMemory

        with _temp_exemplar_json() as tmp_path:
            mem = ExemplarMemory(json_path=tmp_path)
            patch = _ore_patch(42)
            mem.add_negative_exemplar(patch, class_id=14, class_name="obstacle")

            node = _node(55, 100, 100, seq=1)
            wrong = classify_wrong_nodes(
                [node],
                {55: patch.copy()},
                detector=_fixed(None),
                exemplar_memory=mem,
            )
            self.assertEqual(len(wrong), 1)
            self.assertEqual(wrong[0].node_id, 55)
            self.assertTrue(wrong[0].is_wrong)
            self.assertIn("negative_exemplar", wrong[0].reasons)

    def test_active_learning_feedback_loop_saves_to_exemplar_memory(self):
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import WrongNode, save_wrong_nodes_as_negative_exemplars

        with _temp_exemplar_json() as tmp_path:
            mem = ExemplarMemory(json_path=tmp_path)
            patch = _ground_patch()
            wrong_item = WrongNode(77, ["terrain"], class_id=15)

            saved = save_wrong_nodes_as_negative_exemplars(
                [wrong_item],
                images_by_id={77: patch},
                exemplar_memory=mem,
            )
            self.assertEqual(saved, 1)

            # Reload memory and verify persistence
            reloaded = ExemplarMemory(json_path=tmp_path)
            self.assertEqual(len(reloaded._negative), 1)
            self.assertEqual(reloaded._negative[0]["class_id"], 15)
            self.assertEqual(reloaded._negative[0]["source"], "wrong_node_purge")

            # Verify that reloaded memory now matches the patch as negative
            sim, matched = reloaded.match_negative(patch)
            self.assertGreaterEqual(sim, 0.85)
            self.assertIsNotNone(matched)

    def test_safety_backup_on_bulk_purge_exceeds_half(self):
        from core.wrong_node_scan import scan_would_exceed_half

        with _temp_dir() as tmp:
            reg_dir = os.path.join(tmp, "Altgard")
            os.makedirs(reg_dir, exist_ok=True)
            reg_json = os.path.join(reg_dir, "region_data.json")
            backup_json = os.path.join(reg_dir, "region_data.backup.json")

            sample_data = {"region": "Altgard", "nodes": [{"id": i} for i in range(1, 5)]}
            with open(reg_json, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            # F-1792 (D20, 53c3afe): scan_would_exceed_half artık SAF bir
            # kontroldür — yedek yazmaz. Yedek, silmeyi gerçekten yapan
            # purge_wrong_nodes tarafından tam bir kez alınır (F-1801;
            # bkz. test_purge_wrong_nodes_full_flow). Bu test saf kontrol
            # sözleşmesini kilitler: 3/4 (> %50) True döner ama yedek
            # dosyası oluşmaz (eski yan-etki beklentisi bayattı).
            exceeds = scan_would_exceed_half(3, 4, region="Altgard", base_dir=tmp)
            self.assertTrue(exceeds)
            self.assertFalse(os.path.isfile(backup_json))

    def test_purge_wrong_nodes_full_flow(self):
        from core.route_manager import RouteManager
        from core.wrong_node_scan import WrongNode, purge_wrong_nodes
        from dataset_factory.exemplar_memory import ExemplarMemory

        with _temp_dir() as tmp:
            rm = RouteManager(base_dir=tmp)
            rm.create_region("Morheim")
            n1 = rm.add_node(region="Morheim", route="Main", map_x=10, map_y=10)
            n2 = rm.add_node(region="Morheim", route="Main", map_x=20, map_y=20)
            n3 = rm.add_node(region="Morheim", route="Main", map_x=30, map_y=30)

            mem_path = os.path.join(tmp, "learned_exemplars.json")
            mem = ExemplarMemory(json_path=mem_path)

            ground = _ground_patch()
            wrong_nodes = [
                WrongNode(n1.id, ["terrain"], class_id=15),
                WrongNode(n2.id, ["terrain"], class_id=15),
            ]
            images = {n1.id: ground, n2.id: ground.copy()}

            # 2 out of 3 (> 50%): should create backup and active learning negative exemplars
            ok, msg, count = purge_wrong_nodes(
                region="Morheim",
                wrong_nodes=wrong_nodes,
                route_manager=rm,
                images_by_id=images,
                exemplar_memory=mem,
                base_dir=tmp,
            )
            self.assertTrue(ok)
            self.assertEqual(count, 2)
            self.assertIn("2 nokta sildi", msg)

            # Verify nodes deleted from route manager
            self.assertIsNone(rm.get_node(n1.id, region="Morheim"))
            self.assertIsNone(rm.get_node(n2.id, region="Morheim"))
            self.assertIsNotNone(rm.get_node(n3.id, region="Morheim"))

            # Verify automatic safety backup created
            backup_file = os.path.join(tmp, "Morheim", "region_data.backup.json")
            self.assertTrue(os.path.isfile(backup_file))

            # Verify active learning negative exemplars persisted
            self.assertTrue(os.path.isfile(mem_path))
            with open(mem_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertGreaterEqual(len(data.get("negative", [])), 1)

    def test_register_deleted_nodes_as_negative_exemplars_direct_api(self):
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import register_deleted_nodes_as_negative_exemplars

        with _temp_exemplar_json() as tmp_path:
            mem = ExemplarMemory(json_path=tmp_path)
            ground = _ground_patch()
            saved = register_deleted_nodes_as_negative_exemplars(
                [99],
                images_by_id={99: ground},
                exemplar_memory=mem,
                source="user_deleted_node",
            )
            self.assertEqual(saved, 1)

            # Verify memory matches the deleted node crop as negative
            sim, matched = mem.match_negative(ground)
            self.assertGreaterEqual(sim, 0.85)
            self.assertIsNotNone(matched)
            self.assertEqual(matched["source"], "user_deleted_node")

    def test_register_deleted_node_keeps_wrong_node_marker_name(self):
        # v19/F12 (P1-8 class-15 kovası): a bare deleted node is stored
        # under the class-15 BUCKET, but its name must stay
        # "wrong_node_purge" — an unknown-content marker, NOT a
        # path_ground claim. Before the fix add_negative_exemplar
        # canonicalized it to (15, "path_ground") via sanitize
        # Priority 4, so the marker never reached disk.
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import register_deleted_nodes_as_negative_exemplars

        with _temp_exemplar_json() as tmp_path:
            mem = ExemplarMemory(json_path=tmp_path)
            ground = _ground_patch()
            saved = register_deleted_nodes_as_negative_exemplars(
                [99],
                images_by_id={99: ground},
                exemplar_memory=mem,
                source="user_deleted_node",
            )
            self.assertEqual(saved, 1)

            sim, matched = mem.match_negative(ground)
            self.assertGreaterEqual(sim, 0.85)
            self.assertIsNotNone(matched)
            self.assertEqual(matched["class_id"], 15)
            self.assertEqual(matched["class_name"], "wrong_node_purge")
            self.assertEqual(matched["source"], "user_deleted_node")

    def test_register_deleted_nodes_with_disk_patch_loading(self):
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import register_deleted_nodes_as_negative_exemplars

        with _temp_dir() as tmp:
            reg = "TestReg"
            reg_dir = os.path.join(tmp, "routes", reg, "patches")
            os.makedirs(reg_dir, exist_ok=True)
            patch_path = os.path.join(reg_dir, "node_105.png")

            ground = _ground_patch()
            cv2.imwrite(patch_path, ground)

            mem_path = os.path.join(tmp, "learned_exemplars.json")
            mem = ExemplarMemory(json_path=mem_path)

            with patch("core.wrong_node_scan._project_root", return_value=tmp):
                saved = register_deleted_nodes_as_negative_exemplars(
                    [{"id": 105, "class_id": 15, "reason": "terrain"}],
                    region=reg,
                    images_by_id=None,  # No images passed, must load from disk
                    exemplar_memory=mem,
                )
            self.assertEqual(saved, 1)

            sim, matched = mem.match_negative(ground)
            self.assertGreaterEqual(sim, 0.85)
            self.assertIsNotNone(matched)

    def test_backup_region_data_generates_both_backup_and_timestamped_file(self):
        from core.wrong_node_scan import backup_region_data

        with _temp_dir() as tmp:
            reg_dir = os.path.join(tmp, "Altgard")
            os.makedirs(reg_dir, exist_ok=True)
            reg_file = os.path.join(reg_dir, "region_data.json")
            sample = {"region": "Altgard", "nodes": [{"id": 1}]}
            with open(reg_file, "w", encoding="utf-8") as f:
                json.dump(sample, f)

            backup_file = backup_region_data("Altgard", base_dir=tmp)
            self.assertIsNotNone(backup_file)
            self.assertTrue(os.path.isfile(backup_file))

            # Verify timestamped backup exists
            ts_backups = glob.glob(os.path.join(reg_dir, "region_data.backup_*.json"))
            self.assertGreaterEqual(len(ts_backups), 1)


    def test_duplicate_grid_matches_brute_force(self):
        """D21/I1: the grid duplicate check must agree with the old O(n^2)
        pairwise scan, including exact-boundary, negative-coordinate and
        coordless cases. No images / no detector, so "duplicate" is the
        only reason classify_wrong_nodes can produce here."""
        import math
        import random

        from core.wrong_node_scan import SAME_POINT_PX

        crafted = [
            (1000, 0.0, 0.0),
            (1001, 18.0, 0.0),       # exactly SAME_POINT_PX from 1000 -> dup
            (1002, 36.0, 0.0),       # exactly 18 px from dropped 1001 -> dup
            (1003, 100.0, 100.0),
            (1004, 100.0, 118.0001),  # just over the radius -> kept
            (1005, -100.1, -100.1),
            (1006, -118.1, -100.1),  # negative coords, exactly 18 px -> dup
            (1007, 17.9, 50.0),
            (1008, 35.9, 50.0),      # straddles the 18 px cell border, 18.0 -> dup
        ]
        nodes = [_node(nid, x, y, seq=nid) for nid, x, y in crafted]
        rng = random.Random(20261108)
        for i in range(400):
            nodes.append(
                _node(2000 + i, rng.uniform(-500, 500), rng.uniform(-500, 500), seq=2000 + i)
            )
        # Coordless nodes must never count as (0, 0) duplicates (F-1794).
        nodes.append({"id": 3000, "seq_index": 3000, "resource_type": "Aria"})
        nodes.append({"id": 3001, "seq_index": 3001, "resource_type": "Aria"})
        nodes.append(_node(3002, 0.0, 0.0, seq=3002))

        def _brute_force_duplicates(ns):
            ordered = sorted(
                enumerate(ns),
                key=lambda p: (int(p[1].get("seq_index", 0) or 0), int(p[1].get("id", 0) or 0), p[0]),
            )
            seen = []
            dropped = set()
            for _, nd in ordered:
                if nd.get("map_x") is not None and nd.get("map_y") is not None:
                    x, y = float(nd["map_x"]), float(nd["map_y"])
                    if any(math.hypot(x - ox, y - oy) <= SAME_POINT_PX for ox, oy in seen):
                        dropped.add(int(nd["id"]))
                    seen.append((x, y))
            return dropped

        expected = _brute_force_duplicates(nodes)
        wrong = classify_wrong_nodes(nodes)
        got = {item.node_id for item in wrong}
        self.assertEqual(got, expected)
        self.assertEqual(
            {1000, 1003, 1004, 1005, 1007} & got, set(),
            "boundary/kept nodes must not be dropped",
        )
        for item in wrong:
            self.assertIn("duplicate", item.reasons)
            self.assertEqual(item.primary_reason, "duplicate")
        self.assertIn(1001, got)
        self.assertIn(1002, got)
        self.assertIn(1006, got)
        self.assertIn(1008, got)
        self.assertNotIn(3000, got)
        self.assertNotIn(3001, got)

    def test_failing_progress_callback_does_not_abort_scan(self):
        """D21/I3: a progress callback that raises must not abort the scan
        or change its result; the failure is logged and scanning continues."""
        nodes = [
            _node(1, 0.0, 0.0, seq=1),
            _node(2, 5.0, 0.0, seq=2),  # duplicate of node 1
            _node(3, 500.0, 500.0, seq=3),
        ]
        calls = []

        def _boom(step, total):
            calls.append((step, total))
            raise RuntimeError("UI callback boom")

        wrong = classify_wrong_nodes(nodes, on_progress=_boom)
        self.assertTrue(calls, "callback must actually have been attempted")
        self.assertEqual([item.node_id for item in wrong], [2])
        self.assertEqual(wrong[0].primary_reason, "duplicate")
        # Same result as a scan without any callback.
        self.assertEqual(
            [item.node_id for item in classify_wrong_nodes(nodes)],
            [2],
        )


if __name__ == "__main__":
    unittest.main()


