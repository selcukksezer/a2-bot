# -*- coding: utf-8 -*-
"""PerceptionHub single-frame snapshot tests."""

import unittest

import numpy as np

from core.perception_hub import FrameCognition, PerceptionHub, parse_chat_text_line


class _FakeHUD:
    def read_player_hp_mp_bars(self, frame):
        return {
            "hp_pct": 80.0,
            "mp_pct": 55.0,
            "has_target": True,
            "target_pct": 40.0,
            "is_monster_target": True,
            "has_secondary_target": False,
        }

    def parse_chat_feedback_fast(self, frame):
        return {}


class _FakeSkillBar:
    def inspect_skill_bar(self, frame):
        return {
            "1": {"ready": True},
            "2": {"ready": False},
            "Q": {"ready": True},
        }

    def select_best_ready_skill(self, frame):
        return "1"


class PerceptionHubTests(unittest.TestCase):
    def test_observe_fuses_vitals_and_skills(self):
        hub = PerceptionHub(
            hud_reader=_FakeHUD(),
            skill_bar_reader=_FakeSkillBar(),
            min_interval_sec=0.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cog = hub.observe(frame, force=True)
        self.assertIsInstance(cog, FrameCognition)
        self.assertAlmostEqual(cog.player_hp, 0.8, places=2)
        self.assertAlmostEqual(cog.player_mp, 0.55, places=2)
        self.assertTrue(cog.has_target)
        self.assertIn("1", cog.ready_skills)
        d = cog.as_dict()
        self.assertEqual(d["player_hp_percent"], 80)
        self.assertEqual(d["player_mp_percent"], 55)

    def test_throttle_returns_cached(self):
        hub = PerceptionHub(hud_reader=_FakeHUD(), min_interval_sec=10.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        a = hub.observe(frame, force=True)
        b = hub.observe(frame, force=False)
        self.assertIs(a, b)

    def test_parse_chat_text_line(self):
        events = parse_chat_text_line("You obtained 1,450 Kina")
        self.assertTrue(any(e.get("type") == "kina" and e.get("amount") == 1450 for e in events))
        events2 = parse_chat_text_line("You acquired Sapphire Gemstone")
        self.assertTrue(any(e.get("type") == "item" for e in events2))
        events3 = parse_chat_text_line("Yellow Zaif defeated")
        self.assertTrue(any(e.get("type") == "defeat" for e in events3))


if __name__ == "__main__":
    unittest.main()
