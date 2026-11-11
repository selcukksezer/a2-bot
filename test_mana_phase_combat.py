# -*- coding: utf-8 -*-
"""ManaPhaseCombat regen/burst gate tests."""

import unittest

import cv2
import numpy as np

from core.mana_phase_combat import ManaPhase, ManaPhaseCombat


def _potion_brain():
    """Shared fixture: default-threshold brain (HP 50% / MP 30%).

    DEV-D088 dedup — the same constructor call was repeated 9x.
    Potion keys are the D03 F1/F2 defaults (see test_thresholds_...)."""
    return ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)


class ManaPhaseCombatTests(unittest.TestCase):
    def test_should_cast_respects_ms_gap_not_legacy_1_5s(self):
        # Deterministic gate via injected `now` — no real sleep: wall-clock
        # timing would make this test flaky and slow.
        brain = ManaPhaseCombat()
        self.assertTrue(brain.should_cast_skill(min_gap=0.05, now=1000.0))
        self.assertFalse(brain.should_cast_skill(min_gap=0.05, now=1000.04))
        self.assertTrue(brain.should_cast_skill(min_gap=0.05, now=1000.051))
        # Same gate with the default burst interval (0.08s).
        self.assertFalse(brain.should_cast_skill(now=1000.10))
        self.assertTrue(brain.should_cast_skill(now=1000.131))

    def test_skill_gap_is_deterministic_no_hidden_gaussian(self):
        # Replaces test_gauss_skill_gap_has_real_spread_not_just_clamp_bounds
        # (F-1326): gauss_skill_gap was deliberately removed by D03 (gauss
        # removal, F-0457) — the burst gate is deterministic sub-100ms timing
        # (should_cast_skill docstring: "NO hidden Gaussian delay"), so the
        # old distribution test crashed with AttributeError. Assert the
        # replacement behaviour instead: exact millisecond boundaries and
        # identical outcomes regardless of the random seed.
        import random

        brain = ManaPhaseCombat()
        self.assertFalse(hasattr(brain, "gauss_skill_gap"))

        # Exact boundary at the default 0.08s burst interval (pure probe,
        # no side effects): 79ms after the last cast is gated, 80ms passes.
        self.assertTrue(brain.is_cast_gate_ready(now=1000.0))
        self.assertTrue(brain.should_cast_skill(now=1000.0))
        self.assertFalse(brain.is_cast_gate_ready(now=1000.079))
        self.assertTrue(brain.is_cast_gate_ready(now=1000.08))
        self.assertFalse(brain.should_cast_skill(now=1000.079))
        self.assertTrue(brain.should_cast_skill(now=1000.08))

        # No hidden randomness: the same injected-now sequence yields the
        # same outcomes under different random seeds.
        def run_sequence(seed):
            random.seed(seed)
            b = ManaPhaseCombat()
            return [
                b.should_cast_skill(now=t)
                for t in (2000.0, 2000.04, 2000.08, 2000.12, 2000.16, 2000.24)
            ]

        self.assertEqual(run_sequence(1), run_sequence(9999))
        self.assertEqual(run_sequence(1), [True, False, True, False, True, True])

    def test_burst_gate_is_deterministic_no_hidden_gaussian(self):
        # DEV-D088: gauss_skill_gap was deliberately REMOVED (D03, F-0457)
        # and replaced by a deterministic ms gate; the old test called a
        # method that no longer exists. This retargets the test to the
        # replacement contract with the same intent — no hidden jitter:
        # (1) two fresh brains fed the same fake-clock sequence must agree
        # on every gate decision regardless of random.seed, and (2) the
        # side-effect-free probe is_cast_gate_ready() must not consume the
        # gate (only should_cast_skill does).
        import random

        def gate_sequence(seed):
            # Same FIXED fake-clock sequence both times; only the global
            # random state differs. If the gate hid any randomness, the two
            # runs could disagree — they must not.
            random.seed(seed)
            brain = ManaPhaseCombat()
            t = 1000.0
            out = []
            for i in range(500):
                out.append(brain.should_cast_skill(now=t))
                t += 0.04 + 0.04 * ((i * 37) % 10) / 10.0
            return out

        self.assertEqual(gate_sequence(1234), gate_sequence(999))
        self.assertEqual(gate_sequence(1234), gate_sequence(1234))

        brain = ManaPhaseCombat()
        self.assertTrue(brain.is_cast_gate_ready(now=1000.0))
        self.assertTrue(brain.is_cast_gate_ready(now=1000.0))  # probe: no consume
        self.assertTrue(brain.should_cast_skill(now=1000.0))  # consumes
        self.assertFalse(brain.is_cast_gate_ready(now=1000.04))
        self.assertFalse(brain.should_cast_skill(now=1000.04))
        self.assertTrue(brain.is_cast_gate_ready(now=1000.08))
        self.assertTrue(brain.should_cast_skill(now=1000.08))

    def test_regen_when_mp_low(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        d = brain.decide(player_mp=0.40, player_hp=1.0, ready_skills=["1", "2"])
        self.assertEqual(d.phase, ManaPhase.REGEN)
        # Basic attack here is NOT a mouse hold: hold_left is the R-key basic
        # attack (left-click attack = R) and hold_right the T-key basic attack
        # (right-click attack = T) — see test_basic_attack_holds_map_to_rt_keys_via_bus.
        self.assertTrue(d.hold_left)
        self.assertFalse(d.hold_right)

    def test_burst_when_mp_full(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        brain.decide(player_mp=0.40)  # enter regen
        d = brain.decide(player_mp=1.0, player_hp=1.0, ready_skills=["R", "E", "T"])
        self.assertEqual(d.phase, ManaPhase.BURST)
        self.assertFalse(d.hold_left)
        self.assertTrue(d.hold_right)
        # Both basic-attack keys rotate: R (left) and T (right) must both
        # appear in the burst rotation, not R alone. A single 3-key slice
        # is only a window of the rotation (the regen-enter decide above
        # already advanced it), so collect keys over successive bursts.
        self.assertIn("t", d.skill_keys)
        rotated = list(d.skill_keys)
        for i in range(1, 3):
            nxt = brain.decide(
                player_mp=1.0, player_hp=1.0, ready_skills=["R", "E", "T"], now=100.0 + i
            )
            rotated.extend(nxt.skill_keys)
        self.assertIn("r", rotated)
        self.assertIn("t", rotated)

    def test_panic_prefers_defense(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95, hp_panic_threshold=0.5)
        brain.set_lexicon({"1": "defense_heal", "2": "attack", "r": "attack"})
        brain.decide(player_mp=1.0)
        d = brain.decide(
            player_mp=1.0,
            player_hp=0.3,
            ready_skills=["1", "2", "R", "T"],
            fallback_keys=["1", "2", "R", "T"],
        )
        self.assertEqual(d.phase, ManaPhase.BURST)
        self.assertEqual(d.skill_keys[0], "1")

    def test_lexicon_filters_regen_skills(self):
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        brain.set_lexicon({"1": "attack_low_mana", "2": "burst", "q": "scroll_buff"})
        d = brain.decide(player_mp=0.2, ready_skills=["1", "2", "Q"])
        self.assertEqual(d.phase, ManaPhase.REGEN)
        self.assertIn("1", d.skill_keys)
        # The burst skill "2" and the scroll-buff "Q" must be EXCLUDED from
        # the regen rotation — asserting only "1" is present is not enough.
        self.assertNotIn("2", d.skill_keys)
        self.assertNotIn("q", d.skill_keys)
        self.assertEqual(d.skill_keys, ["1"])

    def test_thresholds_configuration_and_clamping(self):
        # DEV-D088: potion defaults are F1/F2, not "7"/"8" — D03 changed
        # them deliberately (F-0293, commit 726d2ee); these tests were
        # asserting the pre-D03 defaults. Assertion strength unchanged.
        # P1-2 (F-0540): potion defaults are the single source 65/45 from
        # config/schema.py (HP/MP_POTION_THRESHOLD_PCT), not 50/30.
        brain = ManaPhaseCombat()
        self.assertEqual(brain.hp_potion_threshold, 0.65)
        self.assertEqual(brain.mp_potion_threshold, 0.45)
        self.assertEqual(brain.hp_potion_key, "F1")
        self.assertEqual(brain.mp_potion_key, "F2")
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

    def test_thresholds_clamp_out_of_range_values(self):
        # The test above only sets in-range values, so clamping was never
        # exercised. Out-of-range values must be clamped to the documented
        # bounds in set_thresholds (order matters: optimal depends on min,
        # max depends on optimal).
        brain = ManaPhaseCombat()
        brain.set_thresholds(
            mp_full_threshold=0.1,  # floor 0.5
            hp_panic_threshold=0.01,  # floor 0.1
            hp_potion_threshold=2.0,  # ceiling 0.95
            mp_potion_threshold=-1.0,  # floor 0.05
            kiting_hp_threshold=0.0,  # floor 0.1
            hp_potion_cd_sec=0.1,  # floor 0.5
            mp_potion_cd_sec=0.0,  # floor 0.5
            kiting_min_dist=0.2,  # floor 1.0
            kiting_optimal_dist=1.1,  # floor min + 0.5 = 1.5
            kiting_max_dist=1.6,  # floor optimal + 1.0 = 2.5
            kiting_cooldown_sec=0.01,  # floor 0.2
            smart_hp_potion_threshold=3.0,  # ceiling 0.95
            smart_mp_potion_threshold=0.0,  # floor 0.05
            emergency_hp_threshold=9.0,  # ceiling 0.95
        )
        self.assertEqual(brain.mp_full_threshold, 0.5)
        self.assertEqual(brain.hp_panic_threshold, 0.1)
        self.assertEqual(brain.hp_potion_threshold, 0.95)
        self.assertEqual(brain.mp_potion_threshold, 0.05)
        self.assertEqual(brain.kiting_hp_threshold, 0.1)
        self.assertEqual(brain.hp_potion_cd_sec, 0.5)
        self.assertEqual(brain.mp_potion_cd_sec, 0.5)
        self.assertEqual(brain.kiting_min_dist, 1.0)
        self.assertEqual(brain.kiting_optimal_dist, 1.5)
        self.assertEqual(brain.kiting_max_dist, 2.5)
        self.assertEqual(brain.kiting_cooldown_sec, 0.2)
        self.assertEqual(brain.smart_hp_potion_threshold, 0.95)
        self.assertEqual(brain.smart_mp_potion_threshold, 0.05)
        self.assertEqual(brain.emergency_hp_threshold, 0.95)

        # Upper bounds on the other side.
        brain.set_thresholds(mp_full_threshold=2.0, hp_panic_threshold=5.0)
        self.assertEqual(brain.mp_full_threshold, 1.0)
        self.assertEqual(brain.hp_panic_threshold, 0.9)

    def test_auto_potion_hp_trigger_below_50_pct(self):
        brain = _potion_brain()
        # HP = 0.45 (< 50%) -> trigger HP potion key 'F1'
        res = brain.check_auto_potions(player_hp=0.45, player_mp=0.80, now=100.0)
        self.assertTrue(res.hp_potion_triggered)
        self.assertFalse(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["F1"])
        self.assertIn("HP 45.0% < 50%", res.reason)

        # Above threshold (HP = 0.55) -> do not trigger
        res2 = brain.check_auto_potions(player_hp=0.55, player_mp=0.80, now=200.0)
        self.assertFalse(res2.hp_potion_triggered)
        self.assertEqual(res2.keys_to_press, [])

    def test_auto_potion_mp_trigger_below_30_pct(self):
        brain = _potion_brain()
        # MP = 0.25 (< 30%) -> trigger MP potion key 'F2'
        res = brain.check_auto_potions(player_hp=0.80, player_mp=0.25, now=100.0)
        self.assertFalse(res.hp_potion_triggered)
        self.assertTrue(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["F2"])
        self.assertIn("MP 25.0% < 30%", res.reason)

    def test_auto_potion_both_hp_and_mp_simultaneous(self):
        brain = _potion_brain()
        # Both HP < 50% and MP < 30%
        res = brain.check_auto_potions(player_hp=0.40, player_mp=0.20, now=100.0)
        self.assertTrue(res.hp_potion_triggered)
        self.assertTrue(res.mp_potion_triggered)
        self.assertEqual(res.keys_to_press, ["F1", "F2"])

    def test_auto_potion_cooldown_enforcement(self):
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, hp_potion_cd_sec=10.0)
        # 1st trigger at t=100.0 -> OK
        res1 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=100.0)
        self.assertTrue(res1.hp_potion_triggered)

        # P0-6: karar tek başına damgalamaz — gönderim onayı olmadan
        # yeniden karar da tetikler (cooldown yalnızca basım sonrası başlar).
        res_unconfirmed = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=105.0)
        self.assertTrue(res_unconfirmed.hp_potion_triggered)

        # t=100.0 basımı onaylanınca cooldown başlar.
        brain.mark_potions_sent(
            [res1.hp_potion_key], hp_key=res1.hp_potion_key, now=100.0
        )

        # 2nd call at t=105.0 (< 10s cooldown) -> blocked by cooldown
        res2 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=105.0)
        self.assertFalse(res2.hp_potion_triggered)
        self.assertEqual(res2.keys_to_press, [])

        # 3rd call at t=110.1 (>= 10s cooldown) -> OK
        res3 = brain.check_auto_potions(player_hp=0.30, player_mp=1.0, now=110.1)
        self.assertTrue(res3.hp_potion_triggered)

    def test_trigger_potions_dispatches_keys(self):
        brain = _potion_brain()
        dispatched_keys = []

        class MockBus:
            def cast_skill(self, key):
                dispatched_keys.append(f"cast_{key}")

        mock_bus = MockBus()
        res = brain.trigger_potions(player_hp=0.35, player_mp=0.20, action_bus=mock_bus, now=100.0)
        self.assertEqual(res.keys_to_press, ["F1", "F2"])
        self.assertEqual(dispatched_keys, ["cast_F1", "cast_F2"])

    def test_trigger_potions_press_key_fallback_bus(self):
        # A bus without cast_skill must still receive the potion keys via
        # press_key — the key fallback path, not just the mocked cast_skill.
        brain = _potion_brain()
        pressed_keys = []

        class PressKeyOnlyBus:
            def press_key(self, key):
                pressed_keys.append(key)

        res = brain.trigger_potions(
            player_hp=0.35, player_mp=0.20, action_bus=PressKeyOnlyBus(), now=100.0
        )
        self.assertEqual(res.keys_to_press, ["F1", "F2"])
        self.assertEqual(pressed_keys, ["F1", "F2"])

    def test_trigger_potions_send_key_fn_callback(self):
        brain = _potion_brain()
        sent_keys = []
        res = brain.trigger_potions(
            player_hp=0.35, player_mp=0.20, send_key_fn=sent_keys.append, now=100.0
        )
        self.assertEqual(res.keys_to_press, ["F1", "F2"])
        self.assertEqual(sent_keys, ["F1", "F2"])

    def test_trigger_potions_bus_without_dispatch_methods_no_crash(self):
        # A bus exposing neither cast_skill nor press_key must not raise;
        # the result still reports the keys that should have been pressed.
        brain = _potion_brain()

        class BareBus:
            pass

        res = brain.trigger_potions(
            player_hp=0.35, player_mp=0.20, action_bus=BareBus(), now=100.0
        )
        self.assertEqual(res.keys_to_press, ["F1", "F2"])

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
        # Close monster: bbox height = 240px on 1000px height frame
        dist_close = ManaPhaseCombat.estimate_target_distance_from_bbox((100, 100, 80, 240), (1000, 1920))
        self.assertIsNotNone(dist_close)
        self.assertLessEqual(dist_close, 5.0)

        # Far monster: bbox height = 60px on 1000px height frame
        dist_far = ManaPhaseCombat.estimate_target_distance_from_bbox((100, 100, 40, 60), (1000, 1920))
        self.assertIsNotNone(dist_far)
        self.assertGreater(dist_far, 12.0)

    def test_read_vitals_from_frame_synthetic(self):
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
        brain = _potion_brain()
        # Paint REAL critical vitals instead of a black frame: a black /
        # unreadable frame must never stand in for "critical HP" — the old
        # test locked that in as expected behaviour. HP bar 20% red,
        # MP bar 20% blue (same ROIs as test_read_vitals_from_frame_synthetic).
        frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
        frame[880:915, 690:746] = [30, 30, 220]
        frame[880:915, 1030:1086] = [220, 120, 20]
        decision = brain.decide_from_frame(
            frame,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2"],
            target_distance=2.0,
            now=100.0,
        )
        self.assertAlmostEqual(decision.player_hp, 0.20, delta=0.05)
        self.assertAlmostEqual(decision.player_mp, 0.20, delta=0.05)
        self.assertIsInstance(decision.potion_keys, list)
        self.assertIn("F1", decision.potion_keys)
        self.assertIn("F2", decision.potion_keys)
        self.assertIsNotNone(decision.kiting)
        self.assertTrue(decision.kiting.should_kite)
        self.assertEqual(decision.kiting.action, "RETREAT_JUMP")

    def test_apply_to_bus_dispatches_potions_and_kiting(self):
        brain = _potion_brain()
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
        self.assertEqual(decision.skill_keys, ["1", "2"])
        brain.apply_to_bus(decision, action_bus=mock_bus)

        self.assertIn("F1", cast_events)
        self.assertIn("F2", cast_events)
        self.assertTrue(len(kite_events) > 0)
        self.assertEqual(kite_events[0][0], ["s", "a"])
        # The rotation must actually reach the bus: every skill key the
        # decision computed has to be dispatched via cast_skill. Previously
        # this test only checked potions, so a dead skill rotation went
        # unnoticed. NOTE: apply_to_bus (core/mana_phase_combat.py, D3)
        # does not dispatch decision.skill_keys yet — this assertion fails
        # until that production fix lands.
        for key in decision.skill_keys:
            self.assertIn(key, cast_events)

    def test_basic_attack_holds_map_to_rt_keys_via_bus(self):
        # hold_left / hold_right are NOT mouse holds: at the bus boundary
        # they are the R-key (left basic attack) and T-key (right basic
        # attack) holds — GameActionBus.hold_left_attack presses "r" and
        # hold_basic_attack is the T-key basic attack. Verify the decision
        # drives exactly those bus calls in each phase.
        brain = ManaPhaseCombat(mp_full_threshold=0.95)
        hold_calls = []

        class RecordingBus:
            def hold_basic_attack(self, val):
                hold_calls.append(("basic_attack_T", val))

            def hold_left_attack(self, val):
                hold_calls.append(("left_attack_R", val))

            def cast_skill(self, key):
                pass

            def execute_kiting_step(self, keys, duration):
                pass

        bus = RecordingBus()
        regen = brain.decide(player_mp=0.20, player_hp=1.0, now=100.0)
        brain.apply_to_bus(regen, action_bus=bus)
        self.assertIn(("left_attack_R", True), hold_calls)
        self.assertIn(("basic_attack_T", False), hold_calls)

        hold_calls.clear()
        burst = brain.decide(player_mp=1.0, player_hp=1.0, now=200.0)
        brain.apply_to_bus(burst, action_bus=bus)
        self.assertIn(("basic_attack_T", True), hold_calls)
        self.assertIn(("left_attack_R", False), hold_calls)


if __name__ == "__main__":
    unittest.main()
