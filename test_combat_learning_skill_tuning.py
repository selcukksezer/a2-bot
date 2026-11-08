# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Event-Driven Combat Learning & Skill Priority Tuning
Exclusive Owner: Agent C6 (core/omni_game_learner.py)
"""

from collections import deque
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from core.omni_game_learner import OmniGameLearner, layout_boxes_for_learn_category


class MockSkillBarReader:
    def __init__(self):
        self.DEFAULT_PRIORITY_WEIGHTS = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0, "Q": 1.0, "E": 1.0, "R": 1.0}
        self.last_preferred_order = None

    def select_best_ready_skill(self, frame=None, preferred_order=None, record_use=True):
        self.last_preferred_order = preferred_order
        if preferred_order:
            return preferred_order[0]
        return "1"


class MockBotWorker:
    def __init__(self):
        self.skill_bar_monitor = MockSkillBarReader()
        self.mode = "combat"
        self.state = "FIGHTING"
        self._route_combat_confirmed = True


class TestEventDrivenCombatLearning(unittest.TestCase):
    def setUp(self):
        self.learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
        self.learner.reset_combat_learning_metrics()

    def test_skill_key_normalization(self):
        self.assertEqual(self.learner._normalize_skill_key("q"), "Q")
        self.assertEqual(self.learner._normalize_skill_key("Q"), "Q")
        self.assertEqual(self.learner._normalize_skill_key("e"), "E")
        self.assertEqual(self.learner._normalize_skill_key("r"), "R")
        self.assertEqual(self.learner._normalize_skill_key("1"), "1")
        self.assertEqual(self.learner._normalize_skill_key("5"), "5")
        self.assertEqual(self.learner._normalize_skill_key(""), "")

    def test_game_frame_gate_protection_rejects_invalid_frames(self):
        # 1. None frame
        ok1, r1 = self.learner.check_game_frame(None)
        self.assertFalse(ok1)
        self.assertIn("invalid", r1)

        # 2. Empty array
        ok2, r2 = self.learner.check_game_frame(np.array([], dtype=np.uint8))
        self.assertFalse(ok2)

        # 3. Tiny array
        ok3, r3 = self.learner.check_game_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(ok3)

        # 4. Solid black / flat color canvas
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok4, r4 = self.learner.check_game_frame(black)
        self.assertFalse(ok4)

        # 5. observe_frame rejects invalid frame and does not update combat state
        initial_hp = self.learner.current_monster_hp_ratio
        snap = self.learner.observe_frame(black)
        self.assertEqual(self.learner.current_monster_hp_ratio, initial_hp)
        self.assertIn("skill_dps_weights", snap)

    def test_record_skill_cast_and_damage(self):
        # Record casts
        self.learner.record_skill_cast("R", hp_ratio=1.0)
        self.learner.record_skill_cast("2", hp_ratio=0.85)

        stats = self.learner.get_skill_stats()
        self.assertEqual(stats["R"]["casts"], 1)
        self.assertEqual(stats["2"]["casts"], 1)
        self.assertEqual(stats["1"]["casts"], 0)

        # Record explicit damage
        self.learner.record_skill_damage("R", hp_drop=0.25)
        self.learner.record_skill_damage("2", hp_drop=0.10)

        stats_after = self.learner.get_skill_stats()
        self.assertEqual(stats_after["R"]["hits"], 1)
        self.assertAlmostEqual(stats_after["R"]["total_hp_drop"], 0.25, places=3)
        self.assertAlmostEqual(stats_after["R"]["avg_hp_drop"], 0.25, places=3)
        self.assertAlmostEqual(stats_after["2"]["avg_hp_drop"], 0.10, places=3)

    def test_on_target_hp_update_auto_attributes_damage(self):
        # Start encounter
        self.learner.on_encounter_start(initial_hp=1.0)
        self.assertTrue(self.learner._in_encounter)

        # Cast heavy skill R
        t0 = time.time()
        self.learner.record_skill_cast("R", hp_ratio=1.0, timestamp=t0)

        # Monster HP drops by 20% within impact window
        measured_drop = self.learner.on_target_hp_update(0.80, timestamp=t0 + 0.35)
        self.assertAlmostEqual(measured_drop, 0.20, places=3)

        # Verify R was credited with the hit and damage
        stats = self.learner.get_skill_stats()
        self.assertEqual(stats["R"]["hits"], 1)
        self.assertAlmostEqual(stats["R"]["avg_hp_drop"], 0.20, places=3)

    def test_dynamic_dps_tuning_fast_vs_slow_encounters(self):
        # Scenario A: Fast encounter where Skill R deals massive damage
        self.learner.on_encounter_start(initial_hp=1.0)
        t_start = 1000.0
        self.learner._encounter_start_ts = t_start

        # Cast skill R and 2
        self.learner.record_skill_cast("R", hp_ratio=1.0, timestamp=t_start + 0.1)
        self.learner.record_skill_damage("R", hp_drop=0.60)

        self.learner.record_skill_cast("2", hp_ratio=0.40, timestamp=t_start + 0.8)
        self.learner.record_skill_damage("2", hp_drop=0.40)

        # Finish encounter in 3.5s (fast kill!)
        result = self.learner.on_encounter_end(killed=True, timestamp=t_start + 3.5)
        self.assertEqual(result["speed_rating"], "fast")
        self.assertEqual(self.learner.fast_encounters, 1)

        weights = self.learner.get_skill_dps_weights()
        # High impact skills in fast encounter must have increased weight > 1.0
        self.assertGreater(weights["R"], 1.00)
        self.assertGreater(weights["2"], 1.00)

        # Priority list should rank R and 2 above uncast skills
        prio = self.learner.get_skills_by_priority()
        self.assertIn(prio[0], ("R", "2"))

        # Scenario B: Slow encounter where skill 1 was spammed with zero damage
        t_start2 = 2000.0
        self.learner.on_encounter_start(initial_hp=1.0, timestamp=t_start2)
        for i in range(10):
            self.learner.record_skill_cast("1", hp_ratio=1.0, timestamp=t_start2 + i)
            # no damage dealt

        result_slow = self.learner.on_encounter_end(killed=False, timestamp=t_start2 + 25.0)
        self.assertEqual(result_slow["speed_rating"], "slow")
        self.assertEqual(self.learner.slow_encounters, 1)

        weights_slow = self.learner.get_skill_dps_weights()
        # Skill 1 (zero hits, slow encounter) should have decayed weight
        self.assertLess(weights_slow["1"], 1.00)

    def test_select_best_skill_uses_dynamic_weights(self):
        # Configure ready skills
        self.learner.ready_skills = ["1", "2", "R", "E"]
        self.learner.cooldown_skills = {}

        # Artificially set learned weight for Skill 2 high
        with self.learner._combat_learning_lock:
            self.learner._skill_metrics["2"]["weight"] = 2.45
            self.learner._skill_metrics["R"]["weight"] = 1.10
            self.learner._skill_metrics["1"]["weight"] = 0.50

        # select_best_skill should choose '2' despite 'R' being ahead in default static order
        best = self.learner.select_best_skill()
        self.assertEqual(best, "2")

        # combo_sequence should place '2' first
        combo = self.learner.get_combo_sequence()
        self.assertEqual(combo[0], "2")

    def test_elimination_of_fake_synthetic_boxes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            frame = np.full((120, 160, 3), 100, dtype=np.uint8)

            # 1. When boxes is None, it must NOT fabricate synthetic (0.5, 0.5, 0.2, 0.2)
            self.learner._save_learning_sample(
                frame=frame,
                category="monsters",
                annotations_meta={"test": True},
                boxes=None,
            )

            # Check json files created in game_learning
            gl_dir = tmp_path / "dataset" / "game_learning" / "monsters"
            # Instead check learner's directory
            from core.omni_game_learner import LEARNING_DIR
            jsons = list((LEARNING_DIR / "monsters").glob("learn_monsters_*.json"))
            self.assertTrue(len(jsons) > 0)
            latest_json = sorted(jsons, key=lambda p: p.stat().st_mtime)[-1]
            data = json.loads(latest_json.read_text(encoding="utf-8"))

            # When boxes=None was passed, boxes list must be empty, no fake synthetic coords!
            self.assertEqual(data["boxes"], [])
            self.assertIsNone(data["box_norm"])
            self.assertIsNone(data["class_id"])

            # 2. When genuine boxes are passed, omni_synthetic must be False
            real_boxes = [
                {
                    "class_id": 10,
                    "box": (0.45, 0.55, 0.12, 0.22),
                    "confidence": 0.88,
                    "source": "visual_monster_detector",
                    "trusted": True,
                    "weak": False,
                }
            ]
            self.learner._save_learning_sample(
                frame=frame,
                category="monsters",
                annotations_meta={"test": True},
                boxes=real_boxes,
            )
            latest_json2 = sorted(
                (LEARNING_DIR / "monsters").glob("learn_monsters_*.json"),
                key=lambda p: p.stat().st_mtime,
            )[-1]
            data2 = json.loads(latest_json2.read_text(encoding="utf-8"))
            self.assertEqual(len(data2["boxes"]), 1)
            self.assertEqual(data2["boxes"][0]["class_id"], 10)

    def test_save_and_load_learned_weights_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_weights.json"

            # Set custom weight and stats
            with self.learner._combat_learning_lock:
                self.learner._skill_metrics["E"]["weight"] = 1.95
                self.learner._skill_metrics["E"]["casts"] = 15
                self.learner._skill_metrics["E"]["hits"] = 12
                self.learner._skill_metrics["E"]["total_hp_drop"] = 3.60
                self.learner._skill_metrics["E"]["avg_hp_drop"] = 0.30
                self.learner.total_encounters = 7
                self.learner.fast_encounters = 5

            ok_save = self.learner.save_learned_weights(path=file_path)
            self.assertTrue(ok_save)
            self.assertTrue(file_path.is_file())

            # Load into fresh learner
            learner2 = OmniGameLearner(bot_worker=None, save_screenshots=False)
            ok_load = learner2.load_learned_weights(path=file_path)
            self.assertTrue(ok_load)

            weights2 = learner2.get_skill_dps_weights()
            self.assertAlmostEqual(weights2["E"], 1.95, places=2)
            self.assertEqual(learner2.total_encounters, 7)
            self.assertEqual(learner2.fast_encounters, 5)

    def test_propagation_to_bot_worker_and_reader(self):
        mock_bot = MockBotWorker()
        learner = OmniGameLearner(bot_worker=mock_bot, save_screenshots=False)

        with learner._combat_learning_lock:
            learner._skill_metrics["Q"]["weight"] = 2.15

        learner._propagate_learned_weights()
        self.assertAlmostEqual(mock_bot.skill_bar_monitor.DEFAULT_PRIORITY_WEIGHTS["Q"], 2.15, places=2)
        self.assertAlmostEqual(mock_bot.skill_bar_monitor.DEFAULT_PRIORITY_WEIGHTS["q"], 2.15, places=2)


if __name__ == "__main__":
    unittest.main()
