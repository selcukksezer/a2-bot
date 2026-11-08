# -*- coding: utf-8 -*-
"""Synthetic wrong-node scan. Does not read or write routes/Altgard/region_data.json."""

import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.route_manager import RouteManager
from core.vision_filters import is_parchment_scroll_patch
from core.wrong_node_scan import (
    CLASS_LABELS_TR,
    MODEL_CONF_MIN,
    SAME_POINT_PX,
    classify_wrong_nodes,
    format_blocked_status_tr,
    format_removed_status_tr,
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


class WrongNodeScanTests(unittest.TestCase):
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

    def test_threshold_and_unknown_classes(self):
        self.assertLessEqual(MODEL_CONF_MIN, 0.15)
        self.assertEqual(wrong_class_id(7, MODEL_CONF_MIN, 16), 7)
        self.assertIsNone(wrong_class_id(7, MODEL_CONF_MIN - 0.001, 16))
        self.assertIsNone(wrong_class_id(1, 0.99, 16))
        self.assertIsNone(wrong_class_id(12, 0.99, 16))
        self.assertEqual(wrong_class_id(15, 0.8, 16), 15)
        self.assertIsNone(wrong_class_id(15, 0.8, 12))
        self.assertIsNone(wrong_class_id(None, None, 16))
        # Class 9 alone is not a hard-wrong delete reason.
        self.assertIsNone(wrong_class_id(9, 0.9, 16))
        # Essences stay keep.
        self.assertIsNone(wrong_class_id(22, 0.9, 135))

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
        ore = _ore_patch(9)
        nodes = [
            _node(11, SAME_POINT_PX, 0, seq=2, gathered=0),
            _node(10, 0, 0, seq=1, gathered=0),
        ]
        wrong = classify_wrong_nodes(
            nodes, {10: ore, 11: ore.copy()}, detector=_fixed((3, 0.95))
        )
        self.assertEqual([item.node_id for item in wrong], [11])
        self.assertEqual(wrong[0].primary_reason, "duplicate")

    def test_exact_four_px_drops_newer_only(self):
        ore = _ore_patch(9)
        nodes = [
            _node(11, 4, 0, seq=2, gathered=0),
            _node(10, 0, 0, seq=1, gathered=0),
        ]
        wrong = classify_wrong_nodes(nodes, {10: ore, 11: ore.copy()}, detector=_fixed((3, 0.95)))
        self.assertEqual([item.node_id for item in wrong], [11])
        self.assertEqual(wrong[0].primary_reason, "duplicate")

    def test_thirty_six_px_copy_is_kept(self):
        ore = _ore_patch(9)
        nodes = [
            _node(11, 36, 0, seq=2, gathered=0),
            _node(10, 0, 0, seq=1, gathered=0),
        ]
        wrong = classify_wrong_nodes(nodes, {10: ore, 11: ore.copy()}, detector=_fixed(None))
        self.assertEqual(wrong, [])

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
        self.assertEqual(names[10], "monster_3d_negative")
        self.assertEqual(names[15], "path_ground")
        self.assertEqual(names[16], "player_hp_bar")
        self.assertIn("essence_odyle", names)
        self.assertIn("essence_azpha", names)
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
        tmp = tempfile.mkdtemp()
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
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_active_learning_feedback_loop_saves_to_exemplar_memory(self):
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import WrongNode, save_wrong_nodes_as_negative_exemplars

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_safety_backup_on_bulk_purge_exceeds_half(self):
        import json
        import tempfile
        from core.wrong_node_scan import scan_would_exceed_half

        tmp = tempfile.mkdtemp()
        reg_dir = os.path.join(tmp, "Altgard")
        os.makedirs(reg_dir, exist_ok=True)
        reg_json = os.path.join(reg_dir, "region_data.json")
        backup_json = os.path.join(reg_dir, "region_data.backup.json")

        sample_data = {"region": "Altgard", "nodes": [{"id": i} for i in range(1, 5)]}
        with open(reg_json, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # 3 out of 4 (> 50%) triggers automatic backup
        exceeds = scan_would_exceed_half(3, 4, region="Altgard", base_dir=tmp)
        self.assertTrue(exceeds)
        self.assertTrue(os.path.isfile(backup_json))

        with open(backup_json, "r", encoding="utf-8") as f:
            backed_data = json.load(f)
        self.assertEqual(backed_data, sample_data)

    def test_purge_wrong_nodes_full_flow(self):
        import json
        import tempfile
        from core.route_manager import RouteManager
        from core.wrong_node_scan import WrongNode, purge_wrong_nodes
        from dataset_factory.exemplar_memory import ExemplarMemory

        tmp = tempfile.mkdtemp()
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
        import tempfile
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import register_deleted_nodes_as_negative_exemplars

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
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
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_register_deleted_nodes_with_disk_patch_loading(self):
        import tempfile
        import cv2
        from dataset_factory.exemplar_memory import ExemplarMemory
        from core.wrong_node_scan import register_deleted_nodes_as_negative_exemplars

        tmp = tempfile.mkdtemp()
        try:
            reg = "TestReg"
            reg_dir = os.path.join(tmp, "routes", reg, "patches")
            os.makedirs(reg_dir, exist_ok=True)
            patch_path = os.path.join(reg_dir, "node_105.png")

            ground = _ground_patch()
            cv2.imwrite(patch_path, ground)

            mem_path = os.path.join(tmp, "learned_exemplars.json")
            mem = ExemplarMemory(json_path=mem_path)

            from unittest.mock import patch
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
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_backup_region_data_generates_both_backup_and_timestamped_file(self):
        import glob
        import json
        import tempfile
        from core.wrong_node_scan import backup_region_data

        tmp = tempfile.mkdtemp()
        try:
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
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


