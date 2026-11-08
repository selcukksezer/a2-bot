# -*- coding: utf-8 -*-
"""learn_* omni layout boxes must land in train txt as weak (Adım 2 culls)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.omni_game_learner import layout_boxes_for_learn_category
from dataset_factory.label_trust import (
    LEARN_AUTO_CONF,
    TRAIN_CONF_FLOOR,
    commit_split_labels,
    is_learn_stem,
)


class LearnAutoLabelTests(unittest.TestCase):
    def test_layout_helpers(self):
        self.assertTrue(is_learn_stem("learn_monsters_1_2"))
        self.assertFalse(is_learn_stem("cap_123"))
        monsters = layout_boxes_for_learn_category("monsters", has_monster_target=True)
        self.assertGreaterEqual(len(monsters), 2)
        obstacles = layout_boxes_for_learn_category("obstacles", steer_deg=20.0)
        self.assertEqual(int(obstacles[0]["class_id"]), 14)
        self.assertLess(float(obstacles[0]["box"][0]), 0.50)
        self.assertGreaterEqual(float(LEARN_AUTO_CONF), float(TRAIN_CONF_FLOOR))

    def test_commit_writes_learn_weak_to_train_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = "learn_skills_999_1"
            label_path = root / "labels" / "train" / f"{stem}.txt"
            boxes = layout_boxes_for_learn_category("skills")
            anns = []
            for item in boxes:
                cx, cy, bw, bh = item["box"]
                anns.append(
                    {
                        "class_id": int(item["class_id"]),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(bw),
                        "h": float(bh),
                        "confidence": float(LEARN_AUTO_CONF),
                        "source": "omni_skills",
                        "trusted": False,
                        "weak": True,
                        "omni_synthetic": True,
                        "learn_auto": True,
                    }
                )
            info = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem,
                label_path=label_path,
                annotations=anns,
            )
            self.assertGreater(info["train_box_count"], 0)
            self.assertGreater(info["weak_box_count"], 0)
            self.assertEqual(info["trusted"], [])
            text = label_path.read_text(encoding="utf-8").strip()
            self.assertTrue(text)
            self.assertIn("13 ", text.splitlines()[0])
            review = json.loads(
                (root / "review" / "train" / f"{stem}.json").read_text(encoding="utf-8")
            )
            self.assertTrue(review.get("weak"))
            self.assertTrue(all(b.get("weak") for b in review["weak"]))
            self.assertTrue(all(not b.get("trusted") for b in review["weak"]))

    def test_obstacles_no_longer_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = "learn_obstacles_1_1"
            label_path = root / "labels" / "train" / f"{stem}.txt"
            item = layout_boxes_for_learn_category("obstacles")[0]
            cx, cy, bw, bh = item["box"]
            info = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem,
                label_path=label_path,
                annotations=[
                    {
                        "class_id": 14,
                        "cx": cx,
                        "cy": cy,
                        "w": bw,
                        "h": bh,
                        "confidence": float(LEARN_AUTO_CONF),
                        "source": "omni_obstacles",
                        "trusted": False,
                        "weak": True,
                        "omni_synthetic": True,
                        "learn_auto": True,
                    }
                ],
            )
            self.assertEqual(info["train_box_count"], 1)
            self.assertTrue(label_path.read_text(encoding="utf-8").strip().startswith("14 "))


if __name__ == "__main__":
    unittest.main()
