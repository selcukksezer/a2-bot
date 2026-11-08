# -*- coding: utf-8 -*-
"""
tests/test_smart_consumables_and_scrolls.py
===========================================
Dedicated test suite for SmartConsumableManager and dynamic potion & buff scroll usage:
1. Upper Consumable Row (y: 0.870..0.938, keybinds F1..F8, X, B) vs Lower Combat Skill Row.
2. Dynamic slot learning via pixel color hue density:
   - hp_potion: Red liquid hue density.
   - mp_potion: Blue/yellow liquid hue density.
   - buff_scroll: Beige/gold parchment scroll icon.
   - empty: Dark low-contrast slot.
3. ExemplarMemory live learning & rejection feedback loop.
4. Cooldown tracking (10s) and buff expiration timers for learned scrolls.
5. ManaPhaseCombat integration:
   - check_emergency_recovery: HP < 65% (or emergency < 40%), MP < 45%, expired buff scrolls.
   - decide_combat_action: returns enriched ManaPhaseDecision with emergency flag, potion_keys, buff_keys.
   - apply_to_bus: dispatches potion and buff keys to action_bus.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.skill_bar_reader import (
    SkillBarReader,
    SmartConsumableManager,
    _ConsumableBarRoiDescriptor,
)
from core.mana_phase_combat import ManaPhaseCombat, ManaPhaseDecision
from dataset_factory.exemplar_memory import ExemplarMemory


def make_red_potion_crop(size=(32, 32)) -> np.ndarray:
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (20, 25, 210)  # BGR: high Red
    return crop


def make_blue_potion_crop(size=(32, 32)) -> np.ndarray:
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (210, 110, 20)  # BGR: high Blue
    return crop


def make_scroll_crop(size=(32, 32)) -> np.ndarray:
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (130, 190, 220)  # BGR: Beige/Gold
    return crop


def make_empty_slot_crop(size=(32, 32)) -> np.ndarray:
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (25, 25, 25)  # BGR: dark grey/black
    return crop


class TestSmartConsumableManagerSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.calibrated_rois_path = self.temp_path / "calibrated_rois.json"
        self.exemplar_memory = ExemplarMemory(json_path=self.temp_path / "learned_exemplars.json")
        self.manager = SmartConsumableManager(
            calibrated_rois_path=self.calibrated_rois_path,
            exemplar_memory=self.exemplar_memory,
        )

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_upper_consumable_roi_descriptor_defaults(self):
        roi = self.manager.current_consumable_bar_roi
        self.assertAlmostEqual(roi[0], 0.345, places=3)
        self.assertAlmostEqual(roi[1], 0.870, places=3)
        self.assertAlmostEqual(roi[2], 0.655, places=3)
        self.assertAlmostEqual(roi[3], 0.938, places=3)
        self.assertLess(roi[3], 0.940)

    def test_dynamic_reload_from_calibrated_rois(self):
        custom_data = {
            "rois": {
                "consumable_bar": {
                    "key": "consumable_bar",
                    "normalized_roi": [0.330, 0.865, 0.670, 0.930],
                }
            }
        }
        with open(self.calibrated_rois_path, "w", encoding="utf-8") as f:
            json.dump(custom_data, f)

        roi = self.manager.current_consumable_bar_roi
        self.assertAlmostEqual(roi[0], 0.330, places=3)
        self.assertAlmostEqual(roi[1], 0.865, places=3)

    def test_pixel_color_hue_classification(self):
        hp_crop = make_red_potion_crop()
        diag = self.manager.classify_slot_crop(hp_crop)
        self.assertEqual(diag["item_type"], "hp_potion")
        self.assertGreaterEqual(diag["confidence"], 0.60)

        mp_crop = make_blue_potion_crop()
        diag = self.manager.classify_slot_crop(mp_crop)
        self.assertEqual(diag["item_type"], "mp_potion")
        self.assertGreaterEqual(diag["confidence"], 0.60)

        scroll_crop = make_scroll_crop()
        diag = self.manager.classify_slot_crop(scroll_crop)
        self.assertEqual(diag["item_type"], "buff_scroll")
        self.assertGreaterEqual(diag["confidence"], 0.55)

        empty_crop = make_empty_slot_crop()
        diag = self.manager.classify_slot_crop(empty_crop)
        self.assertEqual(diag["item_type"], "empty")

    def test_identify_consumable_slots_and_inspect(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = self.manager.current_consumable_bar_roi
        y1, y2 = int(roi[1] * 1080), int(roi[3] * 1080)
        x1, x2 = int(roi[0] * 1920), int(roi[2] * 1920)
        slot_w = (x2 - x1) // 10

        # Slot 0 (F1) -> HP potion (red)
        frame[y1:y2, x1 : x1 + slot_w] = (20, 25, 210)
        # Slot 1 (F2) -> MP potion (blue)
        frame[y1:y2, x1 + slot_w : x1 + 2 * slot_w] = (210, 110, 20)
        # Slot 4 (F5) -> Buff scroll (beige/gold)
        frame[y1:y2, x1 + 4 * slot_w : x1 + 5 * slot_w] = (130, 190, 220)

        slots = self.manager.identify_consumable_slots(frame)
        self.assertEqual(len(slots), 10)
        self.assertEqual(slots["F1"]["item_type"], "hp_potion")
        self.assertEqual(slots["F2"]["item_type"], "mp_potion")
        self.assertEqual(slots["F5"]["item_type"], "buff_scroll")

    def test_exemplar_memory_live_learning_and_rejection(self):
        custom_scroll = np.zeros((32, 32, 3), dtype=np.uint8)
        custom_scroll[:, :] = (120, 185, 215)

        learned = self.manager.learn_slot_assignment("F7", "buff_scroll", custom_scroll)
        self.assertTrue(learned)
        self.assertEqual(self.manager.get_slot_assignment("F7"), "buff_scroll")

        rejected = self.manager.reject_slot_assignment("F7", custom_scroll)
        self.assertTrue(rejected)
        diag = self.manager.classify_slot_crop(custom_scroll)
        self.assertNotEqual(diag["item_type"], "buff_scroll")

    def test_cooldown_and_buff_expiration_lifecycle(self):
        t0 = 1000.0
        self.manager.learn_slot_assignment("F1", "hp_potion")
        self.assertTrue(self.manager.is_slot_ready("F1", now=t0))

        self.manager.mark_used("F1", now=t0)
        self.assertFalse(self.manager.is_slot_ready("F1", now=t0 + 5.0))
        self.assertTrue(self.manager.is_slot_ready("F1", now=t0 + 11.0))

        self.manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=60.0)
        self.manager.mark_buff_applied("F5", duration_sec=60.0, now=t0)

        self.assertTrue(self.manager.is_buff_active("F5", now=t0 + 30.0))
        self.assertFalse(self.manager.is_buff_expired("F5", now=t0 + 30.0))

        self.assertFalse(self.manager.is_buff_active("F5", now=t0 + 65.0))
        self.assertTrue(self.manager.is_buff_expired("F5", now=t0 + 65.0))

        refresh = self.manager.check_and_refresh_scroll_buffs(now=t0 + 65.0)
        self.assertIn("F5", refresh)

    def test_decide_consumable_action(self):
        for s in self.manager.CONSUMABLE_SLOTS:
            self.manager.learn_slot_assignment(s, "empty")
        self.manager.learn_slot_assignment("F1", "hp_potion")
        self.manager.learn_slot_assignment("F2", "mp_potion")
        self.manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=50.0)
        t0 = 2000.0
        self.manager.mark_buff_applied("F5", duration_sec=50.0, now=t0)

        # 1. Healthy vitals and active buff -> no action
        res = self.manager.decide_consumable_action(player_hp_pct=0.90, player_mp_pct=0.85, now=t0 + 10.0)
        self.assertEqual(res["keys_to_press"], [])

        # 2. Low HP (< 0.65) -> trigger F1
        res = self.manager.decide_consumable_action(player_hp_pct=0.60, player_mp_pct=0.85, now=t0 + 10.0)
        self.assertEqual(res["hp_potion_key"], "F1")
        self.assertIn("F1", res["keys_to_press"])

        # 3. Low MP (< 0.45) -> trigger F2
        res = self.manager.decide_consumable_action(player_hp_pct=0.80, player_mp_pct=0.40, now=t0 + 10.0)
        self.assertEqual(res["mp_potion_key"], "F2")
        self.assertIn("F2", res["keys_to_press"])

        # 4. Expired buff scroll at t0 + 55 -> trigger F5
        res = self.manager.decide_consumable_action(player_hp_pct=0.80, player_mp_pct=0.80, now=t0 + 55.0)
        self.assertIn("F5", res["buff_keys"])
        self.assertIn("F5", res["keys_to_press"])

    def test_skill_bar_reader_integration(self):
        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        self.assertIsInstance(reader.consumable_manager, SmartConsumableManager)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        slots = reader.inspect_consumables(frame)
        self.assertEqual(len(slots), 10)


class TestManaPhaseCombatSmartConsumablesSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.calibrated_rois_path = self.temp_path / "calibrated_rois.json"
        self.consumable_manager = SmartConsumableManager(calibrated_rois_path=self.calibrated_rois_path)
        self.consumable_manager.learn_slot_assignment("F1", "hp_potion")
        self.consumable_manager.learn_slot_assignment("F2", "mp_potion")
        self.consumable_manager.learn_slot_assignment("F5", "buff_scroll")
        self.combat = ManaPhaseCombat(consumable_manager=self.consumable_manager)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_hp_below_65_pct_triggers_learned_potion_key(self):
        rec = self.combat.check_emergency_recovery(player_hp=0.60, player_mp=1.0, now=100.0)
        self.assertTrue(rec["hp_potion_needed"])
        self.assertTrue(rec["hp_potion_triggered"])
        self.assertEqual(rec["hp_potion_key"], "F1")
        self.assertFalse(rec["is_emergency"])
        self.assertIn("F1", rec["keys_to_press"])

    def test_hp_below_40_pct_triggers_emergency_flag(self):
        rec = self.combat.check_emergency_recovery(player_hp=0.35, player_mp=1.0, now=100.0)
        self.assertTrue(rec["is_emergency"])
        self.assertTrue(rec["hp_potion_needed"])
        self.assertEqual(rec["hp_potion_key"], "F1")

    def test_mp_below_45_pct_triggers_learned_mp_key(self):
        rec = self.combat.check_emergency_recovery(player_hp=0.90, player_mp=0.42, now=100.0)
        self.assertTrue(rec["mp_potion_needed"])
        self.assertTrue(rec["mp_potion_triggered"])
        self.assertEqual(rec["mp_potion_key"], "F2")
        self.assertIn("F2", rec["keys_to_press"])

    def test_expired_buff_scroll_triggers_reapply_key(self):
        t0 = 500.0
        self.consumable_manager.mark_buff_applied("F5", duration_sec=30.0, now=t0)
        rec = self.combat.check_emergency_recovery(player_hp=0.95, player_mp=0.95, now=t0 + 35.0)
        self.assertTrue(rec["buff_expired"])
        self.assertIn("F5", rec["buff_keys"])
        self.assertIn("F5", rec["keys_to_press"])

    def test_decide_combat_action_unified_output(self):
        t0 = 800.0
        decision = self.combat.decide_combat_action(
            player_hp=0.30,
            player_mp=0.80,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
            now=t0,
        )
        self.assertTrue(decision.emergency)
        self.assertEqual(decision.action_type, "emergency")
        self.assertIn("F1", decision.potion_keys)
        self.assertIn("F1", decision.all_keys_to_press)
        self.assertEqual(decision.all_keys_to_press[0], "F1")

    def test_trigger_emergency_recovery_dispatch(self):
        dispatched_keys = []

        def mock_send(k):
            dispatched_keys.append(k)

        rec = self.combat.trigger_emergency_recovery(
            player_hp=0.35,
            player_mp=0.40,
            send_key_fn=mock_send,
            now=900.0,
        )
        self.assertIn("F1", dispatched_keys)
        self.assertIn("F2", dispatched_keys)
        self.assertEqual(rec["hp_potion_key"], "F1")
        self.assertEqual(rec["mp_potion_key"], "F2")

    def test_apply_to_bus_dispatches_buff_keys(self):
        class MockBus:
            def __init__(self):
                self.casts = []
            def cast_skill(self, k):
                self.casts.append(k)
            def hold_basic_attack(self, state):
                pass

        bus = MockBus()
        decision = ManaPhaseDecision(
            phase=self.combat.phase,
            hold_left=False,
            hold_right=True,
            skill_keys=["1", "2"],
            potion_keys=["F1"],
            buff_keys=["F5", "F6"],
        )
        self.combat.apply_to_bus(decision, action_bus=bus)
        self.assertIn("F1", bus.casts)
        self.assertIn("F5", bus.casts)
        self.assertIn("F6", bus.casts)


if __name__ == "__main__":
    unittest.main()
