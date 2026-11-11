# -*- coding: utf-8 -*-
"""PerceptionHub single-frame snapshot tests."""

import unittest

import numpy as np

from core.perception_hub import FrameCognition, PerceptionHub, parse_chat_text_line


class _FakeHUD:
    def __init__(self, hp_pct=80.0, mp_pct=55.0, has_target=True, target_pct=40.0):
        self._vitals = {
            "hp_pct": hp_pct,
            "mp_pct": mp_pct,
            "has_target": has_target,
            "target_pct": target_pct,
            "is_monster_target": has_target,
            "has_secondary_target": False,
        }

    def read_player_hp_mp_bars(self, frame):
        return dict(self._vitals)

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
        self.assertAlmostEqual(cog.monster_hp, 0.4, places=2)
        self.assertIn("1", cog.ready_skills)
        d = cog.as_dict()
        self.assertEqual(d["player_hp_percent"], 80)
        self.assertEqual(d["player_mp_percent"], 55)

    def test_observe_follows_changed_vitals_not_fixed_values(self):
        # Fusion must track the HUD input: different vitals -> different
        # cognition. A fixed 80/55 answer would pass the test above alone.
        hub = PerceptionHub(
            hud_reader=_FakeHUD(hp_pct=30.0, mp_pct=10.0, has_target=False, target_pct=0.0),
            skill_bar_reader=_FakeSkillBar(),
            min_interval_sec=0.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cog = hub.observe(frame, force=True)
        self.assertAlmostEqual(cog.player_hp, 0.3, places=2)
        self.assertAlmostEqual(cog.player_mp, 0.1, places=2)
        self.assertFalse(cog.has_target)
        self.assertAlmostEqual(cog.monster_hp, 0.0, places=2)
        self.assertEqual(cog.as_dict()["player_hp_percent"], 30)

    def test_observe_with_real_hud_reader_on_painted_bars(self):
        # Real perception, no fake: paint the bottom-center HP (red) and MP
        # (cyan) bars the GameHUDReader actually scans (HP x 0.345..0.478,
        # MP x 0.520..0.652, y 0.885..0.900) and check the hub fuses the
        # measured fill ratios instead of a canned 80/55.
        from core.ocr_distance import GameHUDReader

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # HP inner track starts at x=0.365*w (468); fill its left half red.
        frame[637:648, 468:540] = (0, 0, 255)
        # MP inner track x=0.520*w (666), width 146 px; fill left ~75% cyan.
        frame[637:648, 666:775] = (255, 255, 0)

        reader = GameHUDReader()
        vitals = reader.read_player_hp_mp_bars(frame)
        self.assertAlmostEqual(vitals["hp_pct"], 50.0, delta=6.0)
        self.assertAlmostEqual(vitals["mp_pct"], 75.0, delta=6.0)
        self.assertFalse(vitals["has_target"])

        hub = PerceptionHub(hud_reader=reader, min_interval_sec=0.0)
        cog = hub.observe(frame, force=True)
        self.assertAlmostEqual(cog.player_hp, 0.5, delta=0.06)
        self.assertAlmostEqual(cog.player_mp, 0.75, delta=0.06)
        self.assertFalse(cog.has_target)

    def test_observe_with_real_hud_reader_on_zero_frame(self):
        # A blank frame has no bars: the real reader must report
        # unknown, never the old fake's 80/55 — and never a fake 0.0
        # either (v20 madde 6: okunamayan bar None + hp_valid False'dur,
        # "ölü" değil; 0.0 yalnızca yuvası görünen boş bar içindir).
        from core.ocr_distance import GameHUDReader

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        reader = GameHUDReader()
        vitals = reader.read_player_hp_mp_bars(frame)
        self.assertIsNone(vitals["hp_pct"])
        self.assertIsNone(vitals["mp_pct"])
        self.assertFalse(vitals["hp_valid"])
        self.assertFalse(vitals["mp_valid"])
        self.assertFalse(vitals["has_target"])
        # NOTE (F-1914, D08 commit 386a00e ile düzeltildi): eski `_fill_vitals`
        # `float(vitals.get("hp_pct", 100.0) or 100.0)` yapıyor, gerçek %0
        # okumasını %100 sayıyordu; test o hatayı 1.0 bekleyerek pinliyordu.
        # Düzeltme sonrası 0 geçerli değerdir (`core/perception_hub.py`
        # `_apply_hp_mp`, satır 193-200: `is not None` kontrolü, `or`
        # yok) — sıfır karede hub artık 0.0 döndürür. v20'de bu 0.0
        # "bilinmiyor" varsayılanıdır (hp_valid False); okunamayan bar
        # hub'a None olarak yayılır, "ölü" sayılmaz.
        hub = PerceptionHub(hud_reader=reader, min_interval_sec=0.0)
        cog = hub.observe(frame, force=True)
        self.assertAlmostEqual(cog.player_hp, 0.0, places=2)
        self.assertAlmostEqual(cog.player_mp, 0.0, places=2)
        self.assertFalse(cog.has_target)
        self.assertFalse(cog.hp_valid)
        self.assertFalse(cog.mp_valid)

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
