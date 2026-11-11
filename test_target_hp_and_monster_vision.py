# -*- coding: utf-8 -*-
"""
Test Suite for Agent C3: Target HP Bar & Monster Visual Recognition (core/vision.py).

Verifies:
1. Multi-gamut hostile target HP bar reading (Red, Orange, Yellow, Olive, Magenta).
2. Elimination of false "HEDEF YOK" targeting dropouts at low HP (1..5%) and 0% HP.
3. Accurate fill percentage calculation (0..100%) against true container bounds.
4. Cyan diamond bracket tips `< Monster Name >` detection and target lock verification.
5. 3D monster detection with candidate merging, player overhead exclusion, and ExemplarMemory integration.
6. Dynamic calibrated_rois.json support with auto-reload.
"""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from core.vision import VisionEngine


class TargetHpAndMonsterVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionEngine()
        self.w = 1280
        self.h = 720
        self.blank_frame = np.full((self.h, self.w, 3), 85, dtype=np.uint8)

    def test_read_target_hp_bar_empty_frame(self) -> None:
        """When no target is on screen, returns has_target=False and 0.0% HP."""
        res = self.vision.read_target_hp_bar(self.blank_frame)
        self.assertFalse(res["has_target"])
        self.assertEqual(res["target_hp_pct"], 0.0)
        self.assertFalse(res["is_hostile"])
        self.assertEqual(res["bar_color"], "none")
        self.assertFalse(res["has_diamond_brackets"])

    def test_read_target_hp_bar_red_100_percent(self) -> None:
        """Accurately reads 100% full crimson/red hostile target HP bar."""
        frame = self.blank_frame.copy()
        # Draw 200px red bar in top HUD (y=35..45, x=500..700)
        frame[35:45, 500:700] = (30, 30, 220)  # BGR crimson red

        res = self.vision.read_target_hp_bar(frame)
        self.assertTrue(res["has_target"])
        self.assertTrue(res["is_hostile"])
        self.assertEqual(res["bar_color"], "red")
        self.assertGreaterEqual(res["target_hp_pct"], 95.0)
        self.assertGreaterEqual(res["confidence"], 0.85)

    def test_read_target_hp_bar_orange_50_percent(self) -> None:
        """Accurately reads 50% orange/amber hostile target HP bar."""
        frame = self.blank_frame.copy()
        # 100px orange fill + 100px dark empty slot container
        frame[35:45, 500:600] = (20, 140, 240)  # BGR orange
        frame[35:45, 600:700] = (25, 25, 25)    # Dark container slot

        res = self.vision.read_target_hp_bar(frame)
        self.assertTrue(res["has_target"])
        self.assertTrue(res["is_hostile"])
        self.assertEqual(res["bar_color"], "orange")
        self.assertAlmostEqual(res["target_hp_pct"], 50.0, delta=5.0)

    def test_read_target_hp_bar_yellow_20_percent(self) -> None:
        """Accurately reads 20% yellow/gold hostile target HP bar."""
        frame = self.blank_frame.copy()
        # 40px yellow fill + 160px dark empty slot container (20%)
        frame[35:45, 500:540] = (30, 200, 220)  # BGR yellow/gold
        frame[35:45, 540:700] = (25, 25, 25)    # Dark container slot

        res = self.vision.read_target_hp_bar(frame)
        self.assertTrue(res["has_target"])
        self.assertTrue(res["is_hostile"])
        self.assertEqual(res["bar_color"], "yellow")
        self.assertAlmostEqual(res["target_hp_pct"], 20.0, delta=4.0)

    def test_read_target_hp_bar_low_hp_no_dropout(self) -> None:
        """
        ELIMINATION OF FALSE 'HEDEF YOK' DROPOUTS:
        When monster is severely damaged (e.g. 3% HP remaining), target is STILL locked!
        """
        frame = self.blank_frame.copy()
        # Cyan bracket tips around header
        frame[35:45, 480:490] = (220, 200, 30)  # Cyan left bracket
        frame[35:45, 710:720] = (220, 200, 30)  # Cyan right bracket
        # Only 8px red fill + 192px dark empty container (4% HP)
        frame[35:45, 500:508] = (30, 30, 220)
        frame[35:45, 508:700] = (25, 25, 25)

        res = self.vision.read_target_hp_bar(frame)
        self.assertTrue(res["has_target"], "CRITICAL: low HP must NOT cause false 'HEDEF YOK' dropout!")
        self.assertTrue(res["has_diamond_brackets"])
        self.assertGreater(res["target_hp_pct"], 0.0)
        self.assertLessEqual(res["target_hp_pct"], 8.0)

    def test_read_target_hp_bar_zero_hp_locked(self) -> None:
        """
        When target monster has 0% HP (dying / fading) with brackets,
        remains targeted (has_target=True) with target_hp_pct=0.0.
        """
        frame = self.blank_frame.copy()
        frame[35:45, 480:490] = (220, 200, 30)  # Cyan left bracket
        frame[35:45, 710:720] = (220, 200, 30)  # Cyan right bracket
        frame[35:45, 500:700] = (25, 25, 25)    # Empty dark bar container

        res = self.vision.read_target_hp_bar(frame)
        self.assertTrue(res["has_target"])
        self.assertTrue(res["has_diamond_brackets"])
        self.assertEqual(res["target_hp_pct"], 0.0)

    def test_read_player_hp_mp_bars_fusion(self) -> None:
        """Validates read_player_hp_mp_bars returns complete backward-compatible dict."""
        frame = self.blank_frame.copy()
        # Orange target bar 50%
        frame[35:45, 500:600] = (20, 140, 240)
        frame[35:45, 600:700] = (25, 25, 25)

        vitals = self.vision.read_player_hp_mp_bars(frame)
        self.assertIn("hp_pct", vitals)
        self.assertIn("mp_pct", vitals)
        self.assertIn("target_hp_pct", vitals)
        self.assertIn("target_pct", vitals)
        self.assertIn("has_target", vitals)
        self.assertIn("is_monster_target", vitals)
        self.assertIn("is_hostile", vitals)
        self.assertIn("bar_color", vitals)
        self.assertTrue(vitals["has_target"])
        self.assertAlmostEqual(vitals["target_hp_pct"], 50.0, delta=5.0)
        self.assertEqual(vitals["target_pct"], vitals["target_hp_pct"])

    def test_convenience_methods(self) -> None:
        """Tests has_active_target and get_target_hp_percentage helper methods."""
        frame = self.blank_frame.copy()
        self.assertFalse(self.vision.has_active_target(frame))
        # Sweep G: hedef yokken get_target_hp_percentage() None döner
        # (0.0 yalnızca hedef varken ve gerçekten %0 candayken).
        self.assertIsNone(self.vision.get_target_hp_percentage(frame))

        # Add target
        frame[35:45, 500:700] = (30, 30, 220)
        self.assertTrue(self.vision.has_active_target(frame))
        self.assertGreaterEqual(self.vision.get_target_hp_percentage(frame), 95.0)

    def test_detect_target_brackets(self) -> None:
        """Detects cyan diamond brackets and computes their coordinates."""
        frame = self.blank_frame.copy()
        frame[40:50, 450:460] = (220, 200, 30)  # Left bracket
        frame[40:50, 650:660] = (220, 200, 30)  # Right bracket

        brackets = self.vision.detect_target_brackets(frame)
        self.assertGreaterEqual(len(brackets), 2)
        xs = sorted([b["x"] for b in brackets])
        self.assertAlmostEqual(xs[0], 450, delta=5)
        self.assertAlmostEqual(xs[-1], 650, delta=5)

    def test_detect_monsters_3d_bracketed_monster(self) -> None:
        """3D viewport targeted monster with cyan diamond brackets is detected with conf >= 0.90."""
        frame = self.blank_frame.copy()
        # 3D Monster at x=750, y=280
        frame[270:278, 720:820] = (30, 30, 220)  # Red HP bar
        frame[282:292, 730:810] = (20, 210, 230) # Yellow nameplate
        frame[284:294, 710:720] = (220, 200, 30) # Left cyan bracket
        frame[284:294, 820:830] = (220, 200, 30) # Right cyan bracket

        monsters = self.vision.detect_monsters_3d(frame)
        m3d = [m for m in monsters if m["class_name"] == "monster_3d"]
        self.assertEqual(len(m3d), 1, "Should merge HP bar and nameplate into 1 unified 3D monster")
        self.assertTrue(m3d[0]["has_diamond_brackets"])
        self.assertTrue(m3d[0]["is_targeted"])
        self.assertGreaterEqual(m3d[0]["confidence"], 0.90)

    def test_detect_monsters_3d_player_center_exclusion(self) -> None:
        """Player's own overhead HP bar in the screen center is NEVER detected as a monster."""
        frame = self.blank_frame.copy()
        # Place red bar directly in the player's center overhead zone (y=45%..55%, x=46%..54%)
        cy1, cy2 = int(self.h * 0.48), int(self.h * 0.52)
        cx1, cx2 = int(self.w * 0.47), int(self.w * 0.53)
        frame[cy1:cy2, cx1:cx2] = (30, 30, 220)

        monsters = self.vision.detect_monsters_3d(frame)
        center_mobs = [
            m for m in monsters
            if cx1 <= m["screen_x"] <= cx2 and cy1 <= m["screen_y"] <= cy2
        ]
        self.assertEqual(len(center_mobs), 0, "Player center overhead bar must be excluded")

    def test_detect_monsters_3d_top_target_bar_recognition(self) -> None:
        """Top HUD target HP bar is included in detect_monsters_3d as class_id=18."""
        frame = self.blank_frame.copy()
        frame[35:45, 500:700] = (30, 30, 220)  # Top HUD target bar

        monsters = self.vision.detect_monsters_3d(frame)
        target_bars = [m for m in monsters if m["class_name"] == "target_hp_bar"]
        self.assertEqual(len(target_bars), 1)
        self.assertEqual(target_bars[0]["class_id"], 18)
        self.assertGreaterEqual(target_bars[0]["confidence"], 0.85)

    def test_detect_monsters_3d_exemplar_memory_negative_rejection(self) -> None:
        """Learned negative exemplar (e.g. false rock) is rejected by detect_monsters_3d.

        Proof goes through the real detector, not through is_negative_patch alone:
        the same frame is detected once without the exemplar (control: 1 monster)
        and once after teaching the detector's own candidate crop as a negative
        exemplar (rejected: 0 monsters). The exemplar memory is a temp instance
        patched into ``detect_monsters_3d`` so the real store is untouched.

        NOTE: the frame is 860x720, not 1280x720, because
        ``core/vision.py:2699`` builds the viewport mask with ``vx2 - vy1``
        instead of ``vx2 - vx1`` — at 1280x720 that raises ValueError and no
        ``detect_monsters_3d`` call works at all. That typo lives in
        core/vision.py (another slice's file, D53) and is NOT fixed here;
        the 1280x720 ``detect_monsters_3d`` tests below still fail because of it.
        """
        from unittest.mock import patch

        from dataset_factory.exemplar_memory import ExemplarMemory

        w, h = 860, 720
        frame = np.full((h, w, 3), 85, dtype=np.uint8)
        # Red HP bar in the 3D viewport: a monster_3d candidate.
        frame[280:288, 300:400] = (30, 30, 220)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json = os.path.join(tmp_dir, "test_exemplars.json")
            mem = ExemplarMemory(json_path=tmp_json)
            with patch(
                "dataset_factory.exemplar_memory.get_exemplar_memory",
                return_value=mem,
            ):
                # Control: without the negative exemplar the candidate is detected.
                control = self.vision.detect_monsters_3d(frame)
                control_mobs = [
                    m for m in control if m["class_name"] == "monster_3d"
                ]
                self.assertEqual(
                    len(control_mobs),
                    1,
                    "control frame must yield exactly 1 monster_3d candidate",
                )
                # Teach the exact crop detect_monsters_3d evaluates for this
                # candidate (same slice the detector passes to is_negative_patch).
                bx, by, bw, bh = control_mobs[0]["bbox"]
                crop = frame[
                    max(0, by - 10) : min(h, by + bh + 40),
                    max(0, bx - 15) : min(w, bx + bw + 15),
                ]
                self.assertTrue(
                    mem.add_negative_exemplar(
                        crop, class_id=10, class_name="false_rock"
                    )
                )
                is_neg, sim, _ = mem.is_negative_patch(crop)
                self.assertTrue(is_neg)
                self.assertGreaterEqual(sim, 0.70)

                # The same frame is now rejected by the real detector.
                after = self.vision.detect_monsters_3d(frame)
                rejected = [
                    m for m in after if m["class_name"] == "monster_3d"
                ]
                self.assertEqual(
                    rejected,
                    [],
                    "candidate matching a learned negative exemplar must be "
                    "rejected by detect_monsters_3d",
                )


if __name__ == "__main__":
    unittest.main()
