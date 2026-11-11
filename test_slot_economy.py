# -*- coding: utf-8 -*-
"""SlotEconomy KPH + UCB tests."""

import os
import tempfile
import unittest
from unittest import mock

from core.rl_reward_log import ObstacleMemory, compute_reward, RlRewardLog
from core.slot_economy import SlotEconomy
from core.hardware_input_port import SoftInputAdapter, SerialHidAdapter, build_default_input_port


class SlotEconomyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "economy.sqlite")
        self.eco = SlotEconomy(db_path=self.db, explore_ratio=0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_kph_formula(self):
        score = self.eco.compute_slot_score(1000, 500, 100, 0.5)
        self.assertAlmostEqual(score, 2800.0, places=1)

    def test_record_and_ucb(self):
        self.eco.enter_slot("Slot_A_000_000", cx=10, cy=20, region="A")
        self.eco.record_drop(kina=2000, item_name="Aria", slot_id="Slot_A_000_000")
        self.eco.enter_slot("Slot_B_001_001", cx=110, cy=120, region="B")
        self.eco.record_drop(kina=100, slot_id="Slot_B_001_001")
        pick = self.eco.pick_slot_ucb()
        self.assertIsNotNone(pick)
        self.assertEqual(pick.slot_id, "Slot_A_000_000")

    def test_ucb_explore_branch_picks_least_visited(self):
        """F-1550: the UCB exploration branch runs deterministically.

        Note the constructor clamps explore_ratio into [0.05, 0.5], so the
        old setUp value 0.0 actually meant 0.05 (a 5% explore chance, never
        exercised deterministically). Forcing random.random pins each branch:
        0.0 -> explore (least-visited slot), 0.99 -> exploit (max UCB/KPH).
        """
        self.assertEqual(self.eco.explore_ratio, 0.05)
        eco = SlotEconomy(db_path=self.db, explore_ratio=0.5)
        eco.enter_slot("Slot_A_000_000", cx=10, cy=20, region="A")
        eco.enter_slot("Slot_A_000_000", cx=10, cy=20, region="A")
        eco.record_drop(kina=2000, item_name="Aria", slot_id="Slot_A_000_000")
        eco.enter_slot("Slot_B_001_001", cx=110, cy=120, region="B")
        eco.record_drop(kina=100, slot_id="Slot_B_001_001")
        visits = {s["slot_id"]: s["visits"] for s in eco.list_slots()}
        # S265 (v19/P3-8) reversed the O8 pin: visits is the UCB pull
        # count, so only enter_slot increments it — a drop is a reward
        # observation inside a pull, not a new pull. (The O8 comment
        # cited D16/F-1936, but F-1936's substantive fix is the KPH-scaled
        # UCB bonus, which still stands; its visits+1 side effect
        # inflated n_i for drop-rich slots and let a never-entered slot
        # look pulled once.)
        self.assertEqual(visits["Slot_A_000_000"], 2)
        self.assertEqual(visits["Slot_B_001_001"], 1)

        with mock.patch("random.random", return_value=0.0):
            pick = eco.pick_slot_ucb()
        self.assertIsNotNone(pick)
        self.assertTrue(pick.explore)
        self.assertEqual(pick.reason, "explore_least_visits")
        self.assertEqual(pick.slot_id, "Slot_B_001_001")

        with mock.patch("random.random", return_value=0.99):
            pick = eco.pick_slot_ucb()
        self.assertIsNotNone(pick)
        self.assertFalse(pick.explore)
        self.assertEqual(pick.slot_id, "Slot_A_000_000")

    def test_obstacle_merge(self):
        self.eco.record_obstacle(100, 200, reason="stuck")
        self.eco.record_obstacle(105, 202, reason="stuck")
        near = self.eco.obstacles_near(100, 200)
        self.assertGreaterEqual(len(near), 1)
        self.assertGreaterEqual(int(near[0]["hits"]), 2)


class RewardAndPortTests(unittest.TestCase):
    def test_compute_reward(self):
        r = compute_reward(kina=100, damage_dealt=50, damage_taken=10, wasted_mana=1, stuck_time=2)
        self.assertGreater(r, 0)

    def test_rl_reward_log_writes(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            db = os.path.join(tmp.name, "eco.sqlite")
            jl = os.path.join(tmp.name, "r.jsonl")
            log = RlRewardLog(db_path=db, jsonl_path=jl)
            s = log.log_tick(kina=50, damage_dealt=10)
            self.assertTrue(os.path.isfile(jl))
            self.assertGreater(s.reward, 0)
            summary = log.episode_summary()
            self.assertEqual(summary["samples"], 1)
        finally:
            tmp.cleanup()

    def test_obstacle_memory_stuck(self):
        mem = ObstacleMemory(slot_economy=None)
        mem.stuck_threshold_sec = 0.0
        self.assertIsNone(mem.note_position(1, 1, moving_cmd=True))
        stuck = mem.note_position(1, 1, moving_cmd=True)
        self.assertIsNotNone(stuck)

    def test_soft_input_adapter(self):
        class D:
            def __init__(self):
                self.keys = []
                self.btns = []

            def press_key(self, key, **kwargs):
                self.keys.append(key)

            def _hw_mouse_down(self, b):
                self.btns.append(("down", b))

            def _hw_mouse_up(self, b):
                self.btns.append(("up", b))

        d = D()
        port = SoftInputAdapter(d)
        self.assertTrue(port.send_key("1"))
        self.assertTrue(port.mouse_button("left", True))
        self.assertTrue(port.mouse_button("left", False))
        self.assertEqual(d.keys, ["1"])
        hid = SerialHidAdapter(port_name="")
        self.assertFalse(hid.available)
        self.assertIsInstance(build_default_input_port(d), SoftInputAdapter)


if __name__ == "__main__":
    unittest.main()
