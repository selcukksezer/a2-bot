# -*- coding: utf-8 -*-
"""learn_* omni layout boxes stay weak in review and enter the train txt
ONLY with an explicit per-box human approval (sweep E-06)."""

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
from tools.bootstrap_learn_labels import _layout_annotations


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

    def test_layout_annotations_production_transform(self):
        """The production layout->annotation transform (tools/bootstrap_learn_labels)
        must produce the weak/untrusted learn_auto annotations itself — the test
        must not hand-build them, or the transform goes untested."""
        layout = layout_boxes_for_learn_category("skills")
        anns = _layout_annotations("skills", layout, conf=float(LEARN_AUTO_CONF))
        self.assertEqual(len(anns), len(layout))
        for ann, item in zip(anns, layout):
            self.assertEqual(ann["class_id"], int(item["class_id"]))
            self.assertEqual(
                (ann["cx"], ann["cy"], ann["w"], ann["h"]),
                tuple(float(v) for v in item["box"]),
            )
            self.assertEqual(ann["confidence"], float(LEARN_AUTO_CONF))
            self.assertEqual(ann["source"], "omni_skills")
            self.assertFalse(ann["trusted"])
            self.assertTrue(ann["weak"])
            self.assertTrue(ann["omni_synthetic"])
            self.assertTrue(ann["learn_auto"])

    def test_commit_keeps_unapproved_learn_weak_out_of_train_txt(self):
        """E-06: unapproved weak learn_auto boxes stay in review only — they
        no longer enter the train txt. The same boxes WITH an explicit
        per-box human approval ARE emitted (E-06's only door in)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = "learn_skills_999_1"
            label_path = root / "labels" / "train" / f"{stem}.txt"
            anns = _layout_annotations(
                "skills",
                layout_boxes_for_learn_category("skills"),
                conf=float(LEARN_AUTO_CONF),
            )
            info = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem,
                label_path=label_path,
                annotations=anns,
            )
            self.assertEqual(info["train_box_count"], 0)
            self.assertFalse(info["learn_auto_in_train"])
            self.assertGreater(info["weak_box_count"], 0)
            self.assertEqual(info["trusted"], [])
            text = label_path.read_text(encoding="utf-8").strip()
            self.assertEqual(text, "")
            review = json.loads(
                (root / "review" / "train" / f"{stem}.json").read_text(encoding="utf-8")
            )
            self.assertTrue(review.get("weak"))
            self.assertTrue(all(b.get("weak") for b in review["weak"]))
            self.assertTrue(all(not b.get("trusted") for b in review["weak"]))

            # Same boxes, explicitly human-approved per box: now emitted.
            stem2 = "learn_skills_999_2"
            label_path2 = root / "labels" / "train" / f"{stem2}.txt"
            info2 = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem2,
                label_path=label_path2,
                annotations=anns,
                human_approved=True,
            )
            self.assertGreater(info2["train_box_count"], 0)
            text2 = label_path2.read_text(encoding="utf-8").strip()
            self.assertTrue(text2)
            self.assertIn("13 ", text2.splitlines()[0])

    def test_obstacles_unapproved_learn_weak_stays_out_of_train_txt(self):
        """E-06: the unapproved obstacle layout box stays in review only;
        with an explicit per-box human approval it IS emitted to the txt."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = "learn_obstacles_1_1"
            label_path = root / "labels" / "train" / f"{stem}.txt"
            anns = _layout_annotations(
                "obstacles",
                layout_boxes_for_learn_category("obstacles"),
                conf=float(LEARN_AUTO_CONF),
            )
            self.assertEqual(len(anns), 1)
            info = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem,
                label_path=label_path,
                annotations=anns,
            )
            self.assertEqual(info["train_box_count"], 0)
            self.assertEqual(label_path.read_text(encoding="utf-8").strip(), "")
            stem2 = "learn_obstacles_1_2"
            label_path2 = root / "labels" / "train" / f"{stem2}.txt"
            info2 = commit_split_labels(
                dataset_dir=root,
                split="train",
                stem=stem2,
                label_path=label_path2,
                annotations=anns,
                human_approved=True,
            )
            self.assertEqual(info2["train_box_count"], 1)
            self.assertTrue(label_path2.read_text(encoding="utf-8").strip().startswith("14 "))


if __name__ == "__main__":
    unittest.main()
