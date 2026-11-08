# -*- coding: utf-8 -*-
"""Essence classes and cognition photos stay in the train labels."""

import time
import unittest

import numpy as np

from core.omni_game_learner import OmniGameLearner
from core.reflex_layer import state_from_boxes
from dataset_factory.class_taxonomy import class_id_for_material_text, dataset_class_names
from dataset_factory.yolo_auto_labeler import map_material_name_to_yolo_class_id


class _Actor:
    def __init__(self, moving=False):
        self.moving = moving
        self.state = "WALKING"
        self.mode = "route"

    def _walking_is_real(self):
        return self.moving


class EssenceLearningTests(unittest.TestCase):
    def test_named_essences_are_not_collapsed(self):
        names = dataset_class_names()
        aria = class_id_for_material_text("Aria")
        azpha = class_id_for_material_text("Azpha")
        odyle = class_id_for_material_text("Odyle")
        sapphire = class_id_for_material_text("Sapphire Gemstone")
        self.assertEqual(aria, 0)
        self.assertEqual(names[aria], "herb_aria")
        self.assertGreaterEqual(azpha, 22)
        self.assertGreaterEqual(odyle, 22)
        self.assertGreaterEqual(sapphire, 22)
        self.assertNotEqual(azpha, 2)
        self.assertNotEqual(odyle, 4)
        self.assertEqual(map_material_name_to_yolo_class_id("Azpha"), azpha)
        self.assertEqual(map_material_name_to_yolo_class_id("Odyle"), odyle)
        self.assertEqual(map_material_name_to_yolo_class_id("Aria"), 0)
        self.assertEqual(names[16], "player_hp_bar")
        self.assertEqual(names[18], "hp_potion")
        self.assertEqual(names[21], "skill_slot")
        self.assertEqual(names[15], "path_ground")

    def test_mix_does_not_skip_path_obstacle_and_vital_labels(self):
        actor = _Actor(moving=True)
        learner = OmniGameLearner(bot_worker=actor, save_screenshots=True, min_sample_interval_sec=0.0)
        saved = []

        def _capture(**kwargs):
            saved.append(kwargs.get("category"))

        learner._save_learning_sample = _capture
        learner.observe_mix = lambda *args, **kwargs: {"saved": False}
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[2, 2] = (8, 9, 10)
        learner.observe_frame(
            frame,
            current_state="WALKING",
            current_mode="route",
            mix_sources={"yolo": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]},
        )
        self.assertIn("paths", saved)
        self.assertIn("skills", saved)
        self.assertIn("vitality", saved)

    def test_still_actor_still_skips_path_and_obstacle(self):
        actor = _Actor(moving=False)
        learner = OmniGameLearner(bot_worker=actor, save_screenshots=True, min_sample_interval_sec=0.0)
        saved = []
        learner._save_learning_sample = lambda **kwargs: saved.append(kwargs.get("category"))
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[0, 0] = (3, 2, 1)
        learner._evaluate_and_save_learning_samples(
            frame=frame,
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
        self.assertEqual(state_from_boxes([{"class_id": 18, "class_name": "hp_potion"}]), "Arıyor")


if __name__ == "__main__":
    unittest.main()
