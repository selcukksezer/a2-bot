"""F-0122: mana_brain.decide() output must reach the bus via apply_to_bus.

Regression tests for the R30 fix in core/mana_phase_combat.py:
- apply_to_bus dispatches skill_keys, potion_keys, buff_keys, holds AND
  kiting (the old code dropped skill_keys, and kiting was silently dropped
  whenever the bus exposed no execute_kiting_step / press_key).
- decide_and_apply() decides AND applies atomically, so a caller can no
  longer decide and then only read decision.phase (the production failure).
- apply_to_bus returns a dispatch summary and warns when kiting is dropped.

Every test builds a fresh ManaPhaseCombat: potion/kiting cooldown stamps
are per-instance, so sharing one would make later tests flaky.
"""

import unittest

from core.mana_phase_combat import ManaPhaseCombat, ManaPhaseDecision


class _CastBus:
    """Fake action_bus with the cast_skill / hold / kiting-step surface."""

    def __init__(self):
        self.casts = []
        self.holds = []
        self.kite_steps = []

    def cast_skill(self, key):
        self.casts.append(key)

    def hold_basic_attack(self, state):
        self.holds.append(state)

    def execute_kiting_step(self, keys, duration_sec):
        self.kite_steps.append((list(keys), duration_sec))


class _PressKeyBus:
    """Fake action_bus with only press_key (no cast_skill, no kiting step)."""

    def __init__(self):
        self.pressed = []

    def press_key(self, key):
        self.pressed.append(key)


def _emergency_decision(combat):
    """Emergency-HP + low-MP production decision: potions, buffs, skills,
    and an active kiting maneuver (HP 0.30 < 0.50 kiting threshold)."""
    return combat.decide_combat_action(
        player_hp=0.30,
        player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=900.0,
    )


class TestApplyToBusConsumesWholeDecision(unittest.TestCase):
    def test_apply_to_bus_dispatches_skills_potions_buffs_holds_and_kiting(self):
        combat = ManaPhaseCombat()
        decision = _emergency_decision(combat)
        self.assertTrue(decision.emergency)
        self.assertTrue(decision.skill_keys)
        self.assertTrue(decision.potion_keys)
        self.assertTrue(decision.buff_keys)
        self.assertTrue(decision.kiting.should_kite)

        bus = _CastBus()
        summary = combat.apply_to_bus(decision, action_bus=bus)

        self.assertEqual(
            bus.casts,
            list(decision.potion_keys)
            + list(decision.buff_keys)
            + list(decision.skill_keys),
        )
        self.assertEqual(bus.holds, [decision.hold_right])
        self.assertEqual(len(bus.kite_steps), 1)
        self.assertEqual(bus.kite_steps[0][0], list(decision.kiting.movement_keys))
        self.assertEqual(summary["potion_keys"], list(decision.potion_keys))
        self.assertEqual(summary["buff_keys"], list(decision.buff_keys))
        self.assertEqual(summary["skill_keys"], list(decision.skill_keys))
        self.assertTrue(summary["holds_applied"])
        self.assertTrue(summary["kiting_applied"])

    def test_apply_to_bus_kiting_via_press_key_fallback(self):
        combat = ManaPhaseCombat()
        decision = _emergency_decision(combat)
        self.assertTrue(decision.kiting.should_kite)

        bus = _PressKeyBus()
        summary = combat.apply_to_bus(decision, action_bus=bus)

        for key in decision.potion_keys + decision.buff_keys + decision.skill_keys:
            self.assertIn(key, bus.pressed)
        for key in decision.kiting.movement_keys:
            self.assertIn(key, bus.pressed)
        self.assertTrue(summary["kiting_applied"])

    def test_apply_to_bus_kiting_dropped_without_route_warns_and_reports_false(self):
        combat = ManaPhaseCombat()
        decision = _emergency_decision(combat)
        self.assertTrue(decision.kiting.should_kite)

        class CastOnlyBus:
            def __init__(self):
                self.casts = []

            def cast_skill(self, key):
                self.casts.append(key)

            def hold_basic_attack(self, state):
                pass

        bus = CastOnlyBus()
        with self.assertLogs("Gatherion.ManaPhaseCombat", level="WARNING") as logs:
            summary = combat.apply_to_bus(decision, action_bus=bus)
        self.assertFalse(summary["kiting_applied"])
        self.assertTrue(any("COMBAT_KITE dropped" in line for line in logs.output))
        # Skills/potions/buffs still flow even when kiting has no route.
        self.assertEqual(summary["skill_keys"], list(decision.skill_keys))

    def test_apply_to_bus_summary_empty_decision(self):
        combat = ManaPhaseCombat()
        decision = ManaPhaseDecision(
            phase=combat.phase,
            hold_left=False,
            hold_right=False,
            skill_keys=[],
            potion_keys=[],
            buff_keys=[],
        )
        summary = combat.apply_to_bus(decision, action_bus=_CastBus())
        self.assertEqual(summary["potion_keys"], [])
        self.assertEqual(summary["buff_keys"], [])
        self.assertEqual(summary["skill_keys"], [])
        self.assertFalse(summary["kiting_applied"])


class TestDecideAndApply(unittest.TestCase):
    def test_decide_and_apply_end_to_end(self):
        combat = ManaPhaseCombat()
        bus = _CastBus()
        decision = combat.decide_and_apply(
            bus,
            player_hp=0.30,
            player_mp=0.40,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
            now=900.0,
        )
        self.assertTrue(decision.emergency)
        self.assertEqual(
            bus.casts,
            list(decision.potion_keys)
            + list(decision.buff_keys)
            + list(decision.skill_keys),
        )
        self.assertEqual(len(bus.kite_steps), 1)
        self.assertEqual(bus.kite_steps[0][0], list(decision.kiting.movement_keys))

    def test_decide_and_apply_without_bus_is_fail_safe(self):
        combat = ManaPhaseCombat()
        decision = combat.decide_and_apply(
            None,
            player_hp=0.30,
            player_mp=0.40,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
            now=900.0,
        )
        self.assertTrue(decision.emergency)
        self.assertTrue(decision.skill_keys)
        self.assertTrue(decision.potion_keys)

    def test_decide_and_apply_kiting_via_send_movement_fn(self):
        combat = ManaPhaseCombat()
        moved = []

        def send_movement(keys, duration_sec):
            moved.append((list(keys), duration_sec))

        class CastOnlyBus:
            def __init__(self):
                self.casts = []

            def cast_skill(self, key):
                self.casts.append(key)

            def hold_basic_attack(self, state):
                pass

        bus = CastOnlyBus()
        decision = combat.decide_and_apply(
            bus,
            player_hp=0.30,
            player_mp=0.40,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
            now=900.0,
            send_movement_fn=send_movement,
        )
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0][0], list(decision.kiting.movement_keys))


if __name__ == "__main__":
    unittest.main()
