# -*- coding: utf-8 -*-
"""
test_smart_consumable_manager.py
================================
Comprehensive tests for SmartConsumableManager and ManaPhaseCombat integration:
- Upper Consumable Row vs Lower Combat Skill Row distinction.
- Dynamic slot learning (pixel color hue density: HP red, MP blue/yellow, Buff parchment beige/gold, Empty).
- Live exemplar feedback loop (learn_slot_assignment, reject_slot_assignment).
- Slot readiness, cooldown lockout (10s), and buff expiration timers.
- ManaPhaseCombat check_emergency_recovery thresholds (HP < 65%, emergency < 40%, MP < 45%, expired scrolls).
- ManaPhaseCombat decide_combat_action unified output and apply_to_bus dispatching.
"""

import json
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

from core.skill_bar_reader import (
    SkillBarReader,
    SmartConsumableManager,
    _ConsumableBarRoiDescriptor,
)
from core.mana_phase_combat import ManaPhaseCombat, ManaPhaseDecision
from dataset_factory.exemplar_memory import ExemplarMemory


def make_red_potion_crop(size=(32, 32)) -> np.ndarray:
    """Creates a synthetic crop dominated by red liquid."""
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (20, 25, 210)  # BGR: high Red
    return crop


def make_blue_potion_crop(size=(32, 32)) -> np.ndarray:
    """Creates a synthetic crop dominated by blue liquid."""
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (210, 110, 20)  # BGR: high Blue
    return crop


def make_scroll_crop(size=(32, 32)) -> np.ndarray:
    """Creates a synthetic crop dominated by beige/gold parchment scroll."""
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (130, 190, 220)  # BGR: Beige/Gold (H ~ 20-35, S ~ 80-180, V ~ 150-240)
    return crop


def make_empty_slot_crop(size=(32, 32)) -> np.ndarray:
    """Creates a dark empty slot crop."""
    crop = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop[:, :] = (25, 25, 25)  # BGR: dark grey/black
    return crop


class TestSmartConsumableManager(unittest.TestCase):
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
        """Upper consumable row must be y: [0.870..0.938], distinct from lower combat row (y >= 0.940)."""
        roi = self.manager.current_consumable_bar_roi
        self.assertAlmostEqual(roi[0], 0.345, places=3)
        self.assertAlmostEqual(roi[1], 0.870, places=3)
        self.assertAlmostEqual(roi[2], 0.655, places=3)
        self.assertAlmostEqual(roi[3], 0.938, places=3)
        # Upper consumable row y2 (0.938) is strictly above lower combat skill row (y >= 0.940)
        self.assertLess(roi[3], 0.940)

    def test_dynamic_reload_from_calibrated_rois(self):
        """Manager reloads consumable_bar ROI dynamically from JSON on mtime change."""
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

        # Re-read ROI
        roi = self.manager.current_consumable_bar_roi
        self.assertAlmostEqual(roi[0], 0.330, places=3)
        self.assertAlmostEqual(roi[1], 0.865, places=3)

    def test_pixel_color_hue_classification(self):
        """Synthetic crops are accurately classified as hp_potion, mp_potion, buff_scroll, or empty."""
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
        """Synthetic frame inspect extracts all 10 consumable slots and assigns types."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Fill consumable bar region
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
        """learn_slot_assignment and reject_slot_assignment teach consumable recognition."""
        custom_scroll = np.zeros((32, 32, 3), dtype=np.uint8)
        custom_scroll[:, :] = (120, 185, 215)

        # 1. Learn assignment for F7 as buff_scroll
        learned = self.manager.learn_slot_assignment("F7", "buff_scroll", custom_scroll)
        self.assertTrue(learned)
        self.assertEqual(self.manager.get_slot_assignment("F7"), "buff_scroll")

        # 2. Rejection marks negative exemplar
        rejected = self.manager.reject_slot_assignment("F7", custom_scroll)
        self.assertTrue(rejected)
        # Slot reassessed, rejected crop is suppressed
        diag = self.manager.classify_slot_crop(custom_scroll)
        self.assertNotEqual(diag["item_type"], "buff_scroll")

    def test_cooldown_and_buff_expiration_lifecycle(self):
        """Tracks slot cooldown (10s) and scroll buff duration (e.g. 60s) correctly."""
        t0 = 1000.0
        self.manager.learn_slot_assignment("F1", "hp_potion")

        # Initially ready
        self.assertTrue(self.manager.is_slot_ready("F1", now=t0))

        # Use HP potion
        self.manager.mark_used("F1", now=t0)
        self.assertFalse(self.manager.is_slot_ready("F1", now=t0 + 5.0))
        self.assertTrue(self.manager.is_slot_ready("F1", now=t0 + 11.0))

        # Buff scroll lifecycle
        self.manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=60.0)
        self.manager.mark_buff_applied("F5", duration_sec=60.0, now=t0)

        # Active at t0 + 30
        self.assertTrue(self.manager.is_buff_active("F5", now=t0 + 30.0))
        self.assertFalse(self.manager.is_buff_expired("F5", now=t0 + 30.0))

        # Expired at t0 + 65
        self.assertFalse(self.manager.is_buff_active("F5", now=t0 + 65.0))
        self.assertTrue(self.manager.is_buff_expired("F5", now=t0 + 65.0))

        # check_and_refresh_scroll_buffs returns expired scroll
        refresh = self.manager.check_and_refresh_scroll_buffs(now=t0 + 65.0)
        self.assertIn("F5", refresh)

    def test_decide_consumable_action(self):
        """decide_consumable_action returns appropriate keys based on hp, mp, and buff expiration."""
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
        """SkillBarReader exposes consumable_manager and inspect_consumables."""
        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        self.assertIsInstance(reader.consumable_manager, SmartConsumableManager)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        slots = reader.inspect_consumables(frame)
        self.assertEqual(len(slots), 10)


class TestManaPhaseCombatSmartConsumables(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.calibrated_rois_path = self.temp_path / "calibrated_rois.json"
        self.consumable_manager = SmartConsumableManager(calibrated_rois_path=self.calibrated_rois_path)
        # Pre-assign learned slots
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
        """Player HP < 65% triggers learned HP potion key ('F1') instead of legacy '7'."""
        rec = self.combat.check_emergency_recovery(player_hp=0.60, player_mp=1.0, now=100.0)
        self.assertTrue(rec["hp_potion_needed"])
        self.assertTrue(rec["hp_potion_triggered"])
        self.assertEqual(rec["hp_potion_key"], "F1")
        self.assertFalse(rec["is_emergency"])
        self.assertIn("F1", rec["keys_to_press"])

    def test_hp_below_40_pct_triggers_emergency_flag(self):
        """Player HP < 40% flags is_emergency=True and triggers learned HP key."""
        rec = self.combat.check_emergency_recovery(player_hp=0.35, player_mp=1.0, now=100.0)
        self.assertTrue(rec["is_emergency"])
        self.assertTrue(rec["hp_potion_needed"])
        self.assertEqual(rec["hp_potion_key"], "F1")

    def test_mp_below_45_pct_triggers_learned_mp_key(self):
        """Player MP < 45% triggers learned MP potion key ('F2') instead of legacy '8'."""
        rec = self.combat.check_emergency_recovery(player_hp=0.90, player_mp=0.42, now=100.0)
        self.assertTrue(rec["mp_potion_needed"])
        self.assertTrue(rec["mp_potion_triggered"])
        self.assertEqual(rec["mp_potion_key"], "F2")
        self.assertIn("F2", rec["keys_to_press"])

    def test_expired_buff_scroll_triggers_reapply_key(self):
        """Expired buff scroll automatically triggers learned scroll key ('F5')."""
        t0 = 500.0
        self.consumable_manager.mark_buff_applied("F5", duration_sec=30.0, now=t0)

        # Check at t0 + 35 (expired)
        rec = self.combat.check_emergency_recovery(player_hp=0.95, player_mp=0.95, now=t0 + 35.0)
        self.assertTrue(rec["buff_expired"])
        self.assertIn("F5", rec["buff_keys"])
        self.assertIn("F5", rec["keys_to_press"])

    def test_decide_combat_action_unified_output(self):
        """decide_combat_action enriches ManaPhaseDecision with emergency, potion_keys, buff_keys, all_keys_to_press."""
        t0 = 800.0
        # Trigger emergency HP < 40%
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
        # Emergency HP potion key must be pressed first in all_keys_to_press!
        self.assertEqual(decision.all_keys_to_press[0], "F1")

    def test_trigger_emergency_recovery_dispatch(self):
        """trigger_emergency_recovery sends keys via send_key_fn or action_bus."""
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
        """apply_to_bus dispatches decision.buff_keys through action_bus."""
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
