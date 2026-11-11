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

def make_noisy_potion_crop(base_bgr, seed=0, size=64, noise=18):
    """Realistic slot crop: base colour + per-pixel sensor noise, dark slot rim,
    small bright highlight -- like a potion bottle icon, not a flat colour swatch."""
    rng = np.random.default_rng(seed)
    crop = np.zeros((size, size, 3), dtype=np.int16)
    crop[:, :] = base_bgr
    crop += rng.integers(-noise, noise + 1, size=crop.shape)
    crop[:3, :] = 25
    crop[-3:, :] = 25
    crop[:, :3] = 25
    crop[:, -3:] = 25
    crop[10:16, 10:16] = 240
    return np.clip(crop, 0, 255).astype(np.uint8)


def make_gradient_crop(top_bgr, bottom_bgr, size=64):
    """Vertical shading gradient between two shades of the same hue."""
    t = np.linspace(0, 1, size)[:, None, None]
    top = np.array(top_bgr, dtype=float)
    bottom = np.array(bottom_bgr, dtype=float)
    grad = top * (1 - t) + bottom * t
    return np.repeat(grad, size, axis=1).astype(np.uint8)


def make_partial_fill_crop(base_bgr, fill_frac=0.6, size=64):
    """Bottle/icon occupying only part of the slot; the rest is dark slot background."""
    crop = np.full((size, size, 3), 25, dtype=np.uint8)
    crop[:, : int(size * fill_frac)] = base_bgr
    return crop


def paint_consumable_frame(manager, slot_colors, width=960, height=540):
    """Paints a half-resolution (960x540) frame with the given per-slot BGR colours.

    slot_colors maps slot key (e.g. "F1") to a BGR tuple; unlisted slots stay
    black (empty). Returns the frame; the manager's ROI locates the slots.
    Half resolution is enough: the ROI is relative, so slot geometry and
    every assertion are identical to a 1920x1080 frame at ~1/4 the pixels.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    roi = manager.current_consumable_bar_roi
    y1, y2 = int(roi[1] * height), int(roi[3] * height)
    x1, x2 = int(roi[0] * width), int(roi[2] * width)
    slot_w = (x2 - x1) // len(manager.CONSUMABLE_SLOTS)
    for idx, key in enumerate(manager.CONSUMABLE_SLOTS):
        if key in slot_colors:
            frame[y1:y2, x1 + idx * slot_w: x1 + (idx + 1) * slot_w] = slot_colors[key]
    return frame


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
        """Upper consumable row must be y: [0.870..0.938], distinct from lower combat row (y >= 0.940).

        F-1552: the old test only re-typed the production literals, so it passed
        even if the ROI pointed at the wrong screen area. The checks below are
        derived from the ROI itself and from production constants, and the ROI
        must actually locate painted slots in a frame.
        """
        roi = self.manager.current_consumable_bar_roi
        # Single source of truth: the instance ROI must match the production default.
        self.assertEqual(roi, SmartConsumableManager.DEFAULT_CONSUMABLE_ROW_ROI)
        # Geometry sanity, derived from the ROI itself.
        x1, y1, x2, y2 = roi
        for value in roi:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        # Upper consumable row y2 is strictly above lower combat skill row (y >= 0.940).
        self.assertLess(y2, 0.940)
        # The row must be large enough for 10 real slots at 1080p.
        slot_w_px = (x2 - x1) * 1920 / len(self.manager.CONSUMABLE_SLOTS)
        row_h_px = (y2 - y1) * 1080
        self.assertGreaterEqual(slot_w_px, 32.0)
        self.assertGreaterEqual(row_h_px, 32.0)
        # Behaviour: the ROI must actually locate painted slots in a frame.
        frame = paint_consumable_frame(
            self.manager,
            {"F1": (20, 25, 210), "F2": (210, 110, 20), "F5": (130, 190, 220)},
        )
        slots = self.manager.identify_consumable_slots(frame)
        self.assertEqual(slots["F1"]["item_type"], "hp_potion")
        self.assertEqual(slots["F2"]["item_type"], "mp_potion")
        self.assertEqual(slots["F5"]["item_type"], "buff_scroll")

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

    def test_pixel_color_hue_classification_realistic_crops(self):
        """Noisy, shaded, gradient and partially-filled crops classify like the flat ones.

        F-1489/F-1551: the old test only used solid-colour squares, which never
        exercise the density thresholds the way a real slot crop does.
        """
        cases = [
            (make_noisy_potion_crop((20, 25, 210), seed=1), "hp_potion", 0.60),
            (make_noisy_potion_crop((210, 110, 20), seed=2), "mp_potion", 0.60),
            (make_noisy_potion_crop((130, 190, 220), seed=3), "buff_scroll", 0.55),
            # Yellow MP potion: exercises the yellow_density branch, not just blue.
            (make_noisy_potion_crop((0, 200, 220), seed=4), "mp_potion", 0.60),
            (make_gradient_crop((10, 15, 170), (40, 50, 235)), "hp_potion", 0.60),
            (make_gradient_crop((110, 170, 200), (150, 210, 240)), "buff_scroll", 0.55),
            # Bottle/icon occupying only part of the slot still classifies.
            (make_partial_fill_crop((20, 25, 210), fill_frac=0.6), "hp_potion", 0.60),
            (make_partial_fill_crop((210, 110, 20), fill_frac=0.6), "mp_potion", 0.60),
        ]
        for crop, expected, min_conf in cases:
            with self.subTest(expected=expected):
                diag = self.manager.classify_slot_crop(crop)
                self.assertEqual(diag["item_type"], expected)
                self.assertGreaterEqual(diag["confidence"], min_conf)

    def test_pixel_color_hue_classification_negative_controls(self):
        """Ambiguous / low-signal crops must NOT classify as a potion or scroll."""
        grey = np.full((64, 64, 3), 128, dtype=np.uint8)
        self.assertEqual(self.manager.classify_slot_crop(grey)["item_type"], "empty")
        green = np.full((64, 64, 3), (40, 200, 60), dtype=np.uint8)
        self.assertEqual(self.manager.classify_slot_crop(green)["item_type"], "empty")
        sparse_red = make_partial_fill_crop((20, 25, 210), fill_frac=0.05)
        diag = self.manager.classify_slot_crop(sparse_red)
        self.assertNotEqual(diag["item_type"], "hp_potion")
        self.assertLess(diag["red_density"], 0.12)

    def test_identify_consumable_slots_and_inspect(self):
        """Synthetic frame inspect extracts all 10 consumable slots and assigns types."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        # Fill consumable bar region
        roi = self.manager.current_consumable_bar_roi
        y1, y2 = int(roi[1] * 540), int(roi[3] * 540)
        x1, x2 = int(roi[0] * 960), int(roi[2] * 960)
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
        """SkillBarReader exposes consumable_manager and inspect_consumables.

        F-1436: the old test only asserted len(slots) == 10. This version paints
        distinct items into the frame and checks every slot's type, readiness,
        diagnostics, the learned assignments, and that slot centres land in the ROI.
        """
        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        self.assertIsInstance(reader.consumable_manager, SmartConsumableManager)
        self.assertIs(reader.consumable_manager, reader.get_consumable_manager())
        frame = paint_consumable_frame(
            reader.consumable_manager,
            {"F1": (20, 25, 210), "F2": (210, 110, 20), "F5": (130, 190, 220)},
        )
        slots = reader.inspect_consumables(frame)
        self.assertEqual(list(slots), list(SmartConsumableManager.CONSUMABLE_SLOTS))
        self.assertEqual(slots["F1"]["item_type"], "hp_potion")
        self.assertEqual(slots["F2"]["item_type"], "mp_potion")
        self.assertEqual(slots["F5"]["item_type"], "buff_scroll")
        self.assertTrue(slots["F1"]["ready"])
        self.assertTrue(slots["F2"]["ready"])
        self.assertTrue(slots["F5"]["ready"])
        # Unpainted slots detect as empty, but item_type falls back to the
        # production seeded defaults (DEFAULT_SLOT_ASSIGNMENTS), not "empty".
        for key in ("F3", "F4", "F6", "F7", "F8", "X", "B"):
            self.assertEqual(slots[key]["detected_type"], "empty")
            self.assertEqual(
                slots[key]["item_type"],
                SmartConsumableManager.DEFAULT_SLOT_ASSIGNMENTS[key],
            )
            self.assertFalse(slots[key]["ready"])
        for info in slots.values():
            self.assertIn("confidence", info)
            self.assertIn("state", info)
            self.assertGreaterEqual(info["confidence"], 0.0)
            self.assertLessEqual(info["confidence"], 1.0)
        # Visual inspection feeds the learned assignments (no manual seeding):
        # with seed_defaults=False nothing is pre-assigned, so any assignment
        # must have come from the painted frame.
        unseeded = SmartConsumableManager(
            calibrated_rois_path=self.calibrated_rois_path, seed_defaults=False
        )
        self.assertIsNone(unseeded.get_slot_assignment("F1"))
        unseeded.identify_consumable_slots(frame)
        self.assertEqual(unseeded.get_slot_assignment("F1"), "hp_potion")
        self.assertEqual(unseeded.get_slot_assignment("F2"), "mp_potion")
        self.assertEqual(unseeded.get_slot_assignment("F5"), "buff_scroll")
        # Slot centres land inside the consumable row.
        roi = reader.consumable_manager.current_consumable_bar_roi
        cx, cy = reader.consumable_manager.slot_center_px(frame, "F1", jitter_px=0)
        fh, fw = frame.shape[:2]
        self.assertGreaterEqual(cx, int(roi[0] * fw))
        self.assertLess(cx, int(roi[2] * fw))
        self.assertGreaterEqual(cy, int(roi[1] * fh))
        self.assertLess(cy, int(roi[3] * fh))


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
    def test_apply_to_bus_dispatches_production_decision(self):
        """apply_to_bus dispatches the keys of a decision built by decide_combat_action.

        F-1488: the old test hand-built a ManaPhaseDecision, so it never proved
        the real pipeline's decision reaches the bus. Here the decision comes
        from decide_combat_action (emergency HP + low MP) and the bus must
        receive exactly its potion keys then its buff keys then its combat
        skill keys, in order, plus the hold state.

        F-0122: apply_to_bus consumes the WHOLE decision -- potion keys,
        buff keys AND combat skill keys (the old code silently dropped
        skill_keys, which is the production failure F-0122 fixed). So the
        bus must receive the skill keys too; asserting their absence here
        would contradict the F-0122 contract
        (tests/test_mana_phase_combat_apply_to_bus_f0122.py, same
        production decision inputs, casts == potion + buff + skill).
        """
        class MockBus:
            def __init__(self):
                self.casts = []
                self.holds = []
            def cast_skill(self, k):
                self.casts.append(k)
            def hold_basic_attack(self, state):
                self.holds.append(state)

        bus = MockBus()
        decision = self.combat.decide_combat_action(
            player_hp=0.30,
            player_mp=0.40,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
            now=900.0,
        )
        self.assertTrue(decision.emergency)
        self.assertEqual(decision.action_type, "emergency")
        self.assertIn("F1", decision.potion_keys)
        self.assertIn("F2", decision.potion_keys)
        self.assertIn("F5", decision.buff_keys)
        self.assertEqual(decision.all_keys_to_press[0], "F1")
        # Guard against a vacuous pass: the decision must actually carry
        # combat skill keys, otherwise the order assertion below proves
        # nothing about the skill leg of the dispatch.
        self.assertTrue(decision.skill_keys)

        self.combat.apply_to_bus(decision, action_bus=bus)
        self.assertEqual(
            bus.casts,
            list(decision.potion_keys) + list(decision.buff_keys) + list(decision.skill_keys),
        )
        self.assertEqual(bus.holds, [decision.hold_right])
        for skill_key in decision.skill_keys:
            self.assertIn(skill_key, bus.casts)

    def test_emergency_recovery_uses_visually_learned_slots(self):
        """Emergency recovery keys come from slots learned from a painted frame.

        F-1580: the old decision tests seeded every slot assignment by hand with
        learn_slot_assignment, so they never exercised the visual pipeline.
        This test never calls learn_slot_assignment: the assignments come from
        identify_consumable_slots on a painted frame.
        """
        # seed_defaults=False: no pre-seeded assignments, so every assignment
        # below can only come from the painted frame.
        manager = SmartConsumableManager(
            calibrated_rois_path=self.calibrated_rois_path, seed_defaults=False
        )
        self.assertIsNone(manager.get_slot_assignment("F1"))
        frame = paint_consumable_frame(
            manager,
            {"F1": (20, 25, 210), "F2": (210, 110, 20), "F5": (130, 190, 220)},
        )
        manager.identify_consumable_slots(frame)
        self.assertEqual(manager.get_slot_assignment("F1"), "hp_potion")
        self.assertEqual(manager.get_slot_assignment("F2"), "mp_potion")
        self.assertEqual(manager.get_slot_assignment("F5"), "buff_scroll")

        combat = ManaPhaseCombat(consumable_manager=manager)
        rec = combat.check_emergency_recovery(player_hp=0.60, player_mp=1.0, now=100.0)
        self.assertTrue(rec["hp_potion_needed"])
        self.assertEqual(rec["hp_potion_key"], "F1")
        rec = combat.check_emergency_recovery(player_hp=0.90, player_mp=0.42, now=100.0)
        self.assertTrue(rec["mp_potion_needed"])
        self.assertEqual(rec["mp_potion_key"], "F2")
        # Expired scroll re-applies from the visually learned F5 slot.
        manager.mark_buff_applied("F5", duration_sec=30.0, now=500.0)
        rec = combat.check_emergency_recovery(player_hp=0.95, player_mp=0.95, now=535.0)
        self.assertTrue(rec["buff_expired"])
        self.assertIn("F5", rec["buff_keys"])


if __name__ == "__main__":
    unittest.main()
