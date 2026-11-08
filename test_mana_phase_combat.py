# -*- coding: utf-8 -*-
"""ManaPhaseCombat regen/burst gate tests."""

import time
import unittest

from core.mana_phase_combat import ManaPhase, ManaPhaseCombat


class ManaPhaseCombatTests(unittest.TestCase):
    def test_should_cast_respects_ms_gap_not_legacy_1_5s(self):
        brain = ManaPhaseCombat()
        self.assertTrue(brain.should_cast_skill(min_gap=0.05))
        self.assertFalse(brain.should_cast_skill(min_gap=0.05))
        time.sleep(0.06)
        self.assertTrue(brain.should_cast_skill(min_gap=0.05))
        gap = brain.gauss_skill_gap()
        self.assertLessEqual(gap, 0.25)
        self.assertGreaterEqual(gap, 0.03)

    def test_regen_when_mp_low(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        d = brain.decide(player_mp=0.40, player_hp=1.0, ready_skills=["1", "2"])
        self.assertEqual(d.phase, ManaPhase.REGEN)
        self.assertTrue(d.hold_left)
        self.assertFalse(d.hold_right)

    def test_burst_when_mp_full(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        brain.decide(player_mp=0.40)  # enter regen
        d = brain.decide(player_mp=1.0, player_hp=1.0, ready_skills=["R", "E"])
        self.assertEqual(d.phase, ManaPhase.BURST)
        self.assertFalse(d.hold_left)
        self.assertTrue(d.hold_right)

    def test_panic_prefers_defense(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95, hp_panic_threshold=0.5)
        brain.set_lexicon({"1": "defense_heal", "2": "attack", "r": "attack"})
        brain.decide(player_mp=1.0)
        d = brain.decide(
            player_mp=1.0,
            player_hp=0.3,
            ready_skills=["1", "2", "R"],
            fallback_keys=["1", "2", "R"],
        )
        self.assertEqual(d.phase, ManaPhase.BURST)
        self.assertEqual(d.skill_keys[0], "1")

    def test_lexicon_filters_regen_skills(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        brain.set_lexicon({"1": "attack_low_mana", "2": "burst", "q": "scroll_buff"})
        d = brain.decide(player_mp=0.2, ready_skills=["1", "2", "Q"])
        self.assertEqual(d.phase, ManaPhase.REGEN)
        self.assertIn("1", d.skill_keys)

    def test_thresholds_configuration_and_clamping(self):
        brain = ManaPhaseCombat()
        self.assertEqual(brain.hp_potion_threshold, 0.50)
        self.assertEqual(brain.mp_potion_threshold, 0.30)
        self.assertEqual(brain.hp_potion_key, "7")
        self.assertEqual(brain.mp_potion_key, "8")
        self.assertEqual(brain.kiting_hp_threshold, 0.50)
        self.assertEqual(brain.kiting_min_dist, 3.5)

        brain.set_thresholds(
            hp_potion_threshold=0.65,
            mp_potion_threshold=0.40,
            hp_potion_key="F1",
            mp_potion_key="F2",
            kiting_hp_threshold=0.45,
            kiting_min_dist=4.0,
            kiting_optimal_dist=9.0,
            kiting_max_dist=18.0,
            kiting_cooldown_sec=2.0,
        )
        self.assertEqual(brain.hp_potion_threshold, 0.65)
        self.assertEqual(brain.mp_potion_threshold, 0.40)
        self.assertEqual(brain.hp_potion_key, "F1")
        self.assertEqual(brain.mp_potion_key, "F2")
        self.assertEqual(brain.kiting_hp_threshold, 0.45)
        self.assertEqual(brain.kiting_min_dist, 4.0)
        self.assertEqual(brain.kiting_optimal_dist, 9.0)
        self.assertEqual(brain.kiting_max_dist, 18.0)
        self.assertEqual(brain.kiting_cooldown_sec, 2.0)

    def test_auto_potion_hp_trigger_below_50_pct(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        # HP = 0.45 (< 50%) -> trigger HP potion key '7'
        res = brain.check_auto_potions(player_hp=0.45, player_mp=0.80, now=100.0)
        self.assertTrue(res.hp_potion_triggered)
        self.assertFalse(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["7"])
        self.assertIn("HP 45.0% < 50%", res.reason)

        # Above threshold (HP = 0.55) -> do not trigger
        res2 = brain.check_auto_potions(player_hp=0.55, player_mp=0.80, now=200.0)
        self.assertFalse(res2.hp_potion_triggered)
        self.assertEqual(res2.keys_to_press, [])

    def test_auto_potion_mp_trigger_below_30_pct(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        # MP = 0.25 (< 30%) -> trigger MP potion key '8'
        res = brain.check_auto_potions(player_hp=0.80, player_mp=0.25, now=100.0)
        self.assertFalse(res.hp_potion_triggered)
        self.assertTrue(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["8"])
        self.assertIn("MP 25.0% < 30%", res.reason)

    def test_auto_potion_both_hp_and_mp_simultaneous(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        # Both HP < 50% and MP < 30%
        res = brain.check_auto_potions(player_hp=0.40, player_mp=0.20, now=100.0)
        self.assertTrue(res.hp_potion_triggered)
        self.assertTrue(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["7", "8"])

    def test_auto_potion_cooldown_enforcement(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, hp_potion_cd_sec=10.0)
        # 1st trigger at t=100.0 -> OK
        res1 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=100.0)
        self.assertTrue(res1.hp_potion_triggered)

        # 2nd call at t=105.0 (< 10s cooldown) -> blocked by cooldown
        res2 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=105.0)
        self.assertFalse(res2.hp_potion_triggered)
        self.assertEqual(res2.keys_to_press, [])

        # 3rd call at t=110.1 (>= 10s cooldown) -> OK
        res3 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=110.1)
        self.assertTrue(res3.hp_potion_triggered)

    def test_trigger_potions_dispatches_keys(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        dispatched_keys = []

        class MockBus:
            def cast_skill(self, key):
                dispatched_keys.append(f"cast_{key}")

        mock_bus = MockBus()
        res = brain.trigger_potions(player_hp=0.35, player_mp=0.20, action_bus=mock_bus, now=100.0)
        self.assertEqual(res.keys_to_press, ["7", "8"])
        self.assertEqual(dispatched_keys, ["cast_7", "cast_8"])

    def test_kiting_emergency_panic_hp(self):
        brain = ManaPhaseCombat(kiting_hp_threshold=0.50)
        # HP = 0.40 (< 50%) -> Emergency kiting backpedal / evasive strafe
        decision = brain.evaluate_kiting(player_hp=0.40, player_mp=1.0, now=100.0)
        self.assertTrue(decision.should_kite)
        self.assertEqual(decision.action, "BACKPEDAL")
        self.assertIn("s", decision.movement_keys)
        self.assertIn("emergency_hp_kite", decision.reason)

    def test_kiting_critical_hp_panic_retreat_jump(self):
        brain = ManaPhaseCombat(kiting_hp_threshold=0.50)
        # Critical HP = 0.20 (< 25%) -> RETREAT_JUMP
        decision = brain.evaluate_kiting(player_hp=0.20, player_mp=1.0, now=100.0)
        self.assertTrue(decision.should_kite)
        self.assertEqual(decision.action, "RETREAT_JUMP")
        self.assertEqual(decision.movement_keys, ["s", "space"])
        self.assertIn("critical_hp_panic", decision.reason)

    def test_kiting_distance_adjustment_too_close(self):
        brain = ManaPhaseCombat(kiting_min_dist=3.5)
        # Target distance = 2.5m (< 3.5m min) -> BACKPEDAL
        decision = brain.evaluate_kiting(
            player_hp=1.0,
            player_mp=1.0,
            target_distance=2.5,
            now=100.0,
        )
        self.assertTrue(decision.should_kite)
        self.assertEqual(decision.action, "BACKPEDAL")
        self.assertEqual(decision.movement_keys, ["s"])
        self.assertIn("too_close_dist", decision.reason)

    def test_kiting_distance_adjustment_too_far(self):
        brain = ManaPhaseCombat(kiting_max_dist=15.0)
        # Target distance = 18.0m (> 15.0m max) -> CLOSE_GAP
        decision = brain.evaluate_kiting(
            player_hp=1.0,
            player_mp=1.0,
            target_distance=18.0,
            now=100.0,
        )
        self.assertTrue(decision.should_kite)
        self.assertEqual(decision.action, "CLOSE_GAP")
        self.assertEqual(decision.movement_keys, ["w"])
        self.assertIn("out_of_range_dist", decision.reason)

    def test_kiting_mana_conservation_strafe(self):
        brain = ManaPhaseCombat(mp_potion_threshold=0.30)
        # MP = 0.25 (< 30%) with normal HP and distance -> STRAFE kiting
        decision = brain.evaluate_kiting(
            player_hp=1.0,
            player_mp=0.25,
            target_distance=8.0,
            now=100.0,
        )
        self.assertTrue(decision.should_kite)
        self.assertIn(decision.action, ("STRAFE_LEFT", "STRAFE_RIGHT"))
        self.assertIn(decision.movement_keys[0], ("a", "d"))
        self.assertIn("mana_conserve_strafe", decision.reason)

    def test_kiting_cooldown_guard(self):
        brain = ManaPhaseCombat(kiting_cooldown_sec=1.5)
        # 1st kite at t=100.0 -> OK
        d1 = brain.evaluate_kiting(player_hp=0.40, player_mp=1.0, now=100.0)
        self.assertTrue(d1.should_kite)

        # 2nd call at t=100.8 (< 1.5s cooldown) -> blocked by cooldown
        d2 = brain.evaluate_kiting(player_hp=0.40, player_mp=1.0, now=100.8)
        self.assertFalse(d2.should_kite)
        self.assertEqual(d2.action, "NONE")
        self.assertEqual(d2.reason, "kiting_cooldown")

    def test_estimate_target_distance_from_bbox(self):
        import numpy as np

        # Close monster: bbox height = 240px on 1000px height frame
        dist_close = ManaPhaseCombat.estimate_target_distance_from_bbox((100, 100, 80, 240), (1000, 1920))
        self.assertIsNotNone(dist_close)
        self.assertLessEqual(dist_close, 5.0)

        # Far monster: bbox height = 60px on 1000px height frame
        dist_far = ManaPhaseCombat.estimate_target_distance_from_bbox((100, 100, 40, 60), (1000, 1920))
        self.assertIsNotNone(dist_far)
        self.assertGreater(dist_far, 12.0)

    def test_read_vitals_from_frame_synthetic(self):
        import cv2
        import numpy as np

        brain = ManaPhaseCombat()
        # Create a synthetic 1000x2000 BGR frame
        frame = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # Fill HP bar ROI (x: 690..970, y: 880..915) halfway with bright red (BGR: [30, 30, 220])
        # In HSV: Hue ~ 0 or 175, Sat ~ 220, Val ~ 220
        frame[880:915, 690:830] = [30, 30, 220]

        # Fill MP bar ROI (x: 1030..1310, y: 880..915) 30% with bright blue (BGR: [220, 120, 20])
        # In HSV: Hue ~ 105, Sat ~ 230, Val ~ 220
        frame[880:915, 1030:1114] = [220, 120, 20]

        vitals = brain.read_vitals_from_frame(frame)
        self.assertIn("player_hp", vitals)
        self.assertIn("player_mp", vitals)
        self.assertIn("hp_potion_needed", vitals)
        self.assertIn("mp_potion_needed", vitals)
        self.assertAlmostEqual(vitals["player_hp"], 0.50, delta=0.05)
        self.assertAlmostEqual(vitals["player_mp"], 0.30, delta=0.05)

    def test_decide_from_frame_with_kiting_and_potions(self):
        import numpy as np

        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
        # Empty frame -> vitals will be 0.0 -> critical HP & low MP
        decision = brain.decide_from_frame(
            frame,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2"],
            target_distance=2.0,
            now=100.0,
        )
        self.assertIsInstance(decision.potion_keys, list)
        self.assertIn("7", decision.potion_keys)
        self.assertIn("8", decision.potion_keys)
        self.assertIsNotNone(decision.kiting)
        self.assertTrue(decision.kiting.should_kite)

    def test_apply_to_bus_dispatches_potions_and_kiting(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        cast_events = []
        kite_events = []

        class MockActionBus:
            def hold_basic_attack(self, val):
                pass

            def hold_left_attack(self, val):
                pass

            def cast_skill(self, key):
                cast_events.append(key)

            def execute_kiting_step(self, keys, duration):
                kite_events.append((list(keys), duration))

        mock_bus = MockActionBus()
        # Decision with both HP/MP potions and kiting
        decision = brain.decide(
            player_mp=0.20,
            player_hp=0.35,
            target_distance=2.0,
            now=100.0,
        )
        brain.apply_to_bus(decision, action_bus=mock_bus)

        self.assertIn("7", cast_events)
        self.assertIn("8", cast_events)
        self.assertTrue(len(kite_events) > 0)
        self.assertEqual(kite_events[0][0], ["s", "a"])


if __name__ == "__main__":
    unittest.main()
