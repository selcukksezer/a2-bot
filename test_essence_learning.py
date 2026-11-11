# -*- coding: utf-8 -*-
"""Essence classes and cognition photos stay in the train labels."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from core.omni_game_learner import OmniGameLearner
from core.reflex_layer import state_from_boxes
from dataset_factory.class_taxonomy import (
    alias_map,
    class_id_for_material_text,
    dataset_class_names,
)
from dataset_factory.yolo_auto_labeler import map_material_name_to_yolo_class_id


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Redirect CWD-relative production writes into a throwaway dir.

    The learning path calls ``mark_model_stale``, which writes
    ``models/stale_manifest.json`` relative to the current working
    directory, so a run from the repo root rewrote a repo file. All
    dataset paths in this file are absolute (or patched to tmp), so
    changing CWD is safe here.
    """
    monkeypatch.chdir(tmp_path)


class _Actor:
    def __init__(self, moving=False):
        self.moving = moving
        self.state = "WALKING"
        self.mode = "route"

    def _walking_is_real(self):
        return self.moving


def _game_like_frame(seed, size=120):
    """Random-noise frame that passes the GameFrameGate (>= 60x60, high variance).

    The old 24x24/32x32 near-black frames were rejected as ``frame_too_small``,
    so no learning sample was ever evaluated. Distinct seeds keep the gate's
    perceptual-dHash cache from treating frames of different tests as duplicates.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


class EssenceLearningTests(unittest.TestCase):
    def test_named_essences_map_to_canonical_classes(self):
        """Catalog essences map onto the canonical 0..26 taxonomy (F-1292/F-1293).

        The old assertions (``azpha >= 22``, ``names[18] == "hp_potion"``,
        ``names[21] == "skill_slot"``) contradict the production taxonomy:
        catalog essences are folded into canonical classes by design
        (``class_taxonomy``: never assign IDs >= 25 to catalog essences).
        Azpha is a Targena variant (-> 2, herb_targena); Sapphire Gemstone
        maps to gem_crystal (-> 4). F4 (3e0c679): Odyle has NO canonical
        class any more — the old Odyle -> 4 mapping asserted a specific
        material identity ("odyle is not a gem"), so Odyle resolves to
        None and lands in UNMAPPED_CATALOG_MATERIALS. The distinct essence
        names stay resolvable through the alias map instead of collapsing away.
        """
        names = dataset_class_names()
        aria = class_id_for_material_text("Aria")
        azpha = class_id_for_material_text("Azpha")
        odyle = class_id_for_material_text("Odyle")
        sapphire = class_id_for_material_text("Sapphire Gemstone")
        self.assertEqual(aria, 0)
        self.assertEqual(names[aria], "herb_aria")
        self.assertEqual(azpha, 2)
        self.assertEqual(names[azpha], "herb_targena")
        self.assertIsNone(odyle)
        self.assertEqual(sapphire, 4)
        self.assertEqual(names[sapphire], "gem_crystal")
        self.assertNotEqual(azpha, odyle)
        self.assertNotEqual(odyle, sapphire)
        aliases = alias_map()
        self.assertEqual(aliases["azpha"], 2)
        self.assertNotIn("odyle", aliases)
        self.assertEqual(aliases["sapphire gemstone"], 4)
        self.assertEqual(map_material_name_to_yolo_class_id("Azpha"), azpha)
        # P1-14 (F4): the yolo_auto_labeler lost its "odyle" gem-keyword
        # fallback too — odyle is not a gem, so it is unmapped and falls
        # through to the generic world-material bucket (12) with the
        # default_id=3 contract, exactly like any other unknown name.
        self.assertEqual(map_material_name_to_yolo_class_id("Odyle"), 12)
        self.assertEqual(map_material_name_to_yolo_class_id("Aria"), 0)
        self.assertEqual(names[16], "player_hp_bar")
        self.assertEqual(names[18], "target_hp_bar")
        self.assertEqual(names[21], "skill_slot_cooldown")
        self.assertEqual(names[15], "path_ground")

    def test_mix_does_not_skip_path_obstacle_and_vital_labels(self):
        """The real observe_mix runs (no stub) and independent labels still save (F-1322)."""
        actor = _Actor(moving=True)
        learner = OmniGameLearner(bot_worker=actor, save_screenshots=True, min_sample_interval_sec=0.0)

        # 1. Real observe_mix: the yolo box is fused into a sample (weak box:
        # only trusted F/E prompts or hand labels reach train labels).
        # F-0345 (stale-test fix, R19): the old assertTrue(result["saved"])
        # was written by D72 against the pre-D02 observe_mix, which counted
        # a weak-only sample as saved. D02 (449d1c0, in both v1 and v2)
        # fixed that [Orta] finding: a sample counts as saved only when at
        # least one trusted box reached train. A weak-only yolo mix
        # therefore correctly reports saved=False / no_trusted_boxes —
        # the mix itself still runs for real (no stub) and the box is
        # fused (annotations [0], weak_box_count 1).
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            with mock.patch("core.omni_game_learner.LEARNING_DIR", Path(tmp) / "learning"):
                result = learner.observe_mix(
                    _game_like_frame(seed=11),
                    {"yolo": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]},
                    dataset_dir=dataset_dir,
                )
        self.assertFalse(result["saved"])
        self.assertEqual(result["reason"], "no_trusted_boxes")
        self.assertEqual([b["class_id"] for b in result["annotations"]], [0])
        self.assertEqual(result["weak_box_count"], 1)

        # 2. The mix must not suppress the independent path / obstacle /
        # skill / vitality labels. _evaluate is called directly: going through
        # observe_frame would re-check the same frame in the GameFrameGate and
        # reject it as a duplicate dHash (production issue, cross-slice D2:
        # core/omni_game_learner.py + dataset_factory/game_frame_gate.py).
        saved = []
        learner._save_learning_sample = lambda **kwargs: saved.append(kwargs.get("category"))
        learner._evaluate_and_save_learning_samples(
            frame=_game_like_frame(seed=12),
            now=time.time(),
            current_state="WALKING",
            current_mode="route",
            monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
            skill_info={"ready_skills": ["1"], "cooldown_skills": {}},
            obstacle_info={"obstacle_detected": True, "obstacle_type": "rock", "recommended_steer_deg": 12.0},
            path_info={"is_walking": True, "current_state": "WALKING", "current_mode": "route"},
            vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
        )
        for category in ("paths", "obstacles", "skills", "vitality"):
            self.assertIn(category, saved)

    def test_still_actor_still_skips_path_and_obstacle(self):
        actor = _Actor(moving=False)
        learner = OmniGameLearner(bot_worker=actor, save_screenshots=True, min_sample_interval_sec=0.0)
        saved = []
        learner._save_learning_sample = lambda **kwargs: saved.append(kwargs.get("category"))
        # NOTE: the old 24x24 near-black frame was rejected by the
        # GameFrameGate as frame_too_small, so nothing was saved at all.
        learner._evaluate_and_save_learning_samples(
            frame=_game_like_frame(seed=21),
            now=time.time(),
            current_state="WAITING",
            current_mode="route",
            monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
            skill_info={"ready_skills": ["1"], "cooldown_skills": {}},
            obstacle_info={"obstacle_detected": True, "obstacle_type": "rock", "recommended_steer_deg": 12.0},
            path_info={"is_walking": False, "current_state": "WAITING", "current_mode": "route"},
            vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
        )
        self.assertNotIn("obstacles", saved)
        self.assertNotIn("paths", saved)
        self.assertIn("skills", saved)

    def test_reflex_treats_essence_class_as_gather(self):
        odyle = class_id_for_material_text("Odyle")
        self.assertEqual(state_from_boxes([{"class_id": odyle, "class_name": "essence_odyle"}]), "Topluyor")
        self.assertEqual(state_from_boxes([{"class_id": 14, "class_name": "obstacle"}]), "Arıyor")
        # F-1533: id 18 is target_hp_bar in the canonical taxonomy; hp_potion
        # is 22. The old box mixed the two and was inconsistent.
        self.assertEqual(state_from_boxes([{"class_id": 22, "class_name": "hp_potion"}]), "Arıyor")


if __name__ == "__main__":
    unittest.main()
