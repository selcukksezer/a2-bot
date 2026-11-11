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
from unittest.mock import patch

import numpy as np

import core.omni_game_learner as _ogl
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
        # DEV-D088: on_encounter_end() -> save_learned_weights() with no
        # path writes the REAL repo config/learned_skill_weights.json
        # (updated_at + tuned weights) — every run dirtied a tracked file.
        # Redirect default-path saves of THIS learner to a tmp dir; the
        # real method still runs, explicit-path saves (see
        # test_save_and_load_learned_weights_persistence) pass through.
        self._weights_tmpdir = tempfile.TemporaryDirectory()
        real_save = self.learner.save_learned_weights
        tmp_path = Path(self._weights_tmpdir.name) / "learned_skill_weights.json"

        def _save_redirect(path=None, _real=real_save, _tmp=tmp_path):
            return _real(path=_tmp if path is None else path)

        self.learner.save_learned_weights = _save_redirect

    def tearDown(self):
        self._weights_tmpdir.cleanup()

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
        # F-1290: _save_learning_sample must NEVER write into the real repo
        # dataset. All learner output dirs are redirected to a tmp tree, and
        # the test asserts on the exact files it created (no stale-file pass).
        from core.omni_game_learner import LEARNING_DIR
        from dataset_factory import label_trust as _lt

        real_before = set((LEARNING_DIR / "monsters").glob("learn_monsters_*.json"))
        stale_manifest = Path("models") / "stale_manifest.json"
        stale_before = stale_manifest.read_bytes() if stale_manifest.is_file() else None
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tmp_dataset = tmp_path / "dataset"
            tmp_learning = tmp_dataset / "game_learning"
            (tmp_learning / "monsters").mkdir(parents=True)
            (tmp_dataset / "images" / "train").mkdir(parents=True)
            (tmp_dataset / "labels" / "train").mkdir(parents=True)
            frame = np.full((120, 160, 3), 100, dtype=np.uint8)

            with (
                patch.object(_ogl, "LEARNING_DIR", tmp_learning),
                patch.object(_ogl, "DATASET_DIR", tmp_dataset),
                patch.object(_ogl, "TRAIN_IMAGES_DIR", tmp_dataset / "images" / "train"),
                patch.object(_ogl, "TRAIN_LABELS_DIR", tmp_dataset / "labels" / "train"),
                # DEV-D088: commit_split_labels() -> mark_model_stale()
                # defaults models_dir to CWD-relative Path("models"), so a
                # tmp-redirected save still rewrote the REAL repo
                # models/stale_manifest.json. Record the calls instead: the
                # invalidation must still happen, but only against tmp.
                patch.object(_lt, "mark_model_stale") as mock_stale,
            ):
                # 1. When boxes is None, it must NOT fabricate synthetic (0.5, 0.5, 0.2, 0.2)
                # — and no empty record is written at all (B-04/B-10):
                # no file, no counters, False returned.
                saved_none = self.learner._save_learning_sample(
                    frame=frame,
                    category="monsters",
                    annotations_meta={"test": True},
                    boxes=None,
                )
                self.assertIs(saved_none, False)

                # Exactly zero sample files: nothing fabricated, nothing written.
                jsons = list((tmp_learning / "monsters").glob("learn_monsters_*.json"))
                self.assertEqual(len(jsons), 0)

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
                jsons2 = list((tmp_learning / "monsters").glob("learn_monsters_*.json"))
                self.assertEqual(len(jsons2), 1)
                new_json = [p for p in jsons2 if p not in jsons]
                self.assertEqual(len(new_json), 1)
                data2 = json.loads(new_json[0].read_text(encoding="utf-8"))
                self.assertEqual(len(data2["boxes"]), 1)
                self.assertEqual(data2["boxes"][0]["class_id"], 10)

        # Real repo dataset untouched by either call.
        real_after = set((LEARNING_DIR / "monsters").glob("learn_monsters_*.json"))
        self.assertEqual(real_after, real_before)

        # DEV-D088: the real stale manifest must be byte-identical too, and
        # the (single) successful save must still have requested model
        # invalidation — against the tmp dataset, never the repo. The
        # rejected empty save (boxes=None) invalidates nothing.
        stale_after = stale_manifest.read_bytes() if stale_manifest.is_file() else None
        self.assertEqual(stale_after, stale_before)
        self.assertEqual(mock_stale.call_count, 1)
        for call in mock_stale.call_args_list:
            self.assertEqual(call.kwargs.get("dataset_dir"), tmp_dataset)

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
