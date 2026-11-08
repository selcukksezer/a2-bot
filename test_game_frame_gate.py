# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for İş Paketi D:
- Intelligent Game Frame Gate (dataset_factory/game_frame_gate.py)
- Factory Controller Ingestion Hardening (dataset_factory/factory_controller.py)
- Desktop & Non-Game Screenshot Purging (dataset_factory/clean_dataset.py)
"""

import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dataset_factory.clean_dataset import purge_non_game_screenshots
from dataset_factory.factory_controller import AutoLabelingDatasetFactory
from dataset_factory.game_frame_gate import (
    GameFrameGate,
    check_process_and_focus,
    check_visual_hud,
    check_visual_novelty,
    clear_frame_gate_cache,
    get_default_frame_gate,
    is_valid_game_frame,
)


class TestGameFrameGateSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real_game_sample_path = Path("dataset/images/train/boot_478467_000_aria_3d_crest_far.jpg")
        if cls.real_game_sample_path.is_file():
            raw = np.fromfile(str(cls.real_game_sample_path), dtype=np.uint8)
            cls.real_game_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        else:
            cls.real_game_img = None

    def setUp(self) -> None:
        self.gate = GameFrameGate(max_hash_cache=64, min_dhash_distance=6)
        clear_frame_gate_cache()

    # ==========================================================================
    # 1. Kapı 1: Process & Focus Guard Tests
    # ==========================================================================
    def test_gate1_rejects_when_game_not_running(self) -> None:
        self.gate.set_simulated_game_state(is_running=False)
        ok, reason = self.gate.check_gate1_process_and_focus()
        self.assertFalse(ok)
        self.assertIn("game_not_running", reason)

    def test_gate1_rejects_windows_desktop(self) -> None:
        self.gate.set_simulated_game_state(is_running=True, foreground_title="Program Manager", foreground_pid=100)
        ok, reason = self.gate.check_gate1_process_and_focus()
        self.assertFalse(ok)
        self.assertIn("desktop_focused", reason)

        self.gate.set_simulated_game_state(is_running=True, foreground_title="Desktop", foreground_pid=101)
        ok2, reason2 = self.gate.check_gate1_process_and_focus()
        self.assertFalse(ok2)
        self.assertIn("desktop_focused", reason2)

    def test_gate1_rejects_gatherion_ui(self) -> None:
        self.gate.set_simulated_game_state(is_running=True, foreground_title="Gatherion Autonomous Bot v2.4", foreground_pid=9999)
        ok, reason = self.gate.check_gate1_process_and_focus()
        self.assertFalse(ok)
        self.assertIn("gatherion_ui_focused", reason)

        self.gate.set_simulated_game_state(is_running=True, foreground_title="Overlay HUD", foreground_pid=os.getpid())
        ok2, reason2 = self.gate.check_gate1_process_and_focus()
        self.assertFalse(ok2)
        self.assertIn("gatherion_ui_focused", reason2)

    def test_gate1_rejects_blocked_apps(self) -> None:
        for blocked_title in ["Google Chrome - New Tab", "Visual Studio Code", "File Explorer", "Task Manager"]:
            self.gate.set_simulated_game_state(is_running=True, foreground_title=blocked_title, foreground_pid=5000)
            ok, reason = self.gate.check_gate1_process_and_focus()
            self.assertFalse(ok, f"Should reject {blocked_title}")
            self.assertTrue("blocked_foreground" in reason or "window_not_focused" in reason)

    def test_gate1_accepts_verified_game_window(self) -> None:
        self.gate.set_simulated_game_state(is_running=True, foreground_title="AION2 Client 64-bit", foreground_pid=8888)
        ok, reason = self.gate.check_gate1_process_and_focus()
        self.assertTrue(ok)
        self.assertIn("process_and_focus_ok", reason)

    def test_master_gate_bypasses_gate1_when_require_foreground_false(self) -> None:
        self.gate.set_simulated_game_state(is_running=False)
        if self.real_game_img is not None:
            # Game is not running, but require_foreground=False allows offline analysis
            ok, reason = self.gate.is_valid_game_frame(self.real_game_img, require_foreground=False)
            self.assertTrue(ok)
            self.assertIn("accepted_hud_evidence", reason)

    # ==========================================================================
    # 2. Kapı 2: Visual HUD Evidence Tests
    # ==========================================================================
    def test_gate2_rejects_black_frame(self) -> None:
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(black)
        self.assertFalse(ok)
        self.assertIn("black_frame", reason)

    def test_gate2_rejects_solid_color_frame(self) -> None:
        solid = np.full((720, 1280, 3), 120, dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(solid)
        self.assertFalse(ok)
        self.assertIn("solid_color", reason)

    def test_gate2_rejects_pure_white_canvas(self) -> None:
        white = np.full((720, 1280, 3), 255, dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(white)
        self.assertFalse(ok)
        self.assertTrue("solid_color" in reason or "blank_white" in reason)

    def test_gate2_rejects_empty_or_invalid_arrays(self) -> None:
        ok1, _ = self.gate.check_gate2_visual_hud(None)  # type: ignore
        self.assertFalse(ok1)
        ok2, _ = self.gate.check_gate2_visual_hud(np.array([], dtype=np.uint8))
        self.assertFalse(ok2)
        ok3, _ = self.gate.check_gate2_visual_hud(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(ok3)

    def test_gate2_rejects_synthetic_desktop_canvas(self) -> None:
        # Wallpaper + bottom taskbar
        desktop = np.full((1080, 1920, 3), (180, 120, 40), dtype=np.uint8)
        desktop[1035:, :] = (25, 25, 25)  # dark taskbar strip
        ok, reason = self.gate.check_gate2_visual_hud(desktop)
        self.assertFalse(ok)
        self.assertTrue("no_hud_evidence" in reason or "desktop" in reason)

    def test_gate2_accepts_player_hp_mp_bar(self) -> None:
        # Create a non-flat frame with a realistic player HP bar in bottom-center
        h, w = 1080, 1920
        grad = np.tile(np.linspace(20, 80, w, dtype=np.uint8), (h, 1))
        frame = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        # Draw red HP bar: y=920..936, x=750..950 (width=200, height=16)
        cv2.rectangle(frame, (750, 920), (950, 936), (20, 20, 220), -1)
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("player_red_hp_bar", reason)

    def test_gate2_accepts_minimap_compass_arrow(self) -> None:
        # Create a frame with a bright arrow in center of top-right minimap ROI
        h, w = 1080, 1920
        grad = np.tile(np.linspace(20, 80, w, dtype=np.uint8), (h, 1))
        frame = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        # Radar center is at (1650, 195)
        pts = np.array([[1650, 185], [1640, 205], [1660, 205]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (250, 250, 250))
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("accepted_hud_evidence", reason)

    def test_gate2_accepts_skill_bar_slots(self) -> None:
        frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
        # Draw 4 aligned skill slot boxes at y=980
        for i in range(4):
            x = 600 + i * 60
            cv2.rectangle(frame, (x, 980), (x + 40, 1020), (200, 200, 200), 2)
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("skill_bar_slots", reason)

    def test_gate2_accepts_real_dataset_screenshot(self) -> None:
        if self.real_game_img is not None:
            ok, reason = self.gate.check_gate2_visual_hud(self.real_game_img)
            self.assertTrue(ok)
            self.assertIn("accepted_hud_evidence", reason)

    # ==========================================================================
    # 3. Kapı 3: Visual Novelty & dHash Tests
    # ==========================================================================
    def test_gate3_dhash_calculation_and_hamming_distance(self) -> None:
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw horizontal ramp
        for x in range(100):
            img2[:, x] = x * 2

        h1 = GameFrameGate.compute_dhash(img1)
        h2 = GameFrameGate.compute_dhash(img2)
        self.assertIsInstance(h1, int)
        self.assertIsInstance(h2, int)
        dist = GameFrameGate.hamming_distance(h1, h2)
        self.assertGreaterEqual(dist, 0)
        self.assertEqual(GameFrameGate.hamming_distance(h1, h1), 0)

    def test_gate3_rejects_identical_and_duplicate_frames(self) -> None:
        if self.real_game_img is None:
            return
        self.gate.clear_hash_cache()
        ok1, _ = self.gate.check_gate3_visual_novelty(self.real_game_img, update_cache=True)
        self.assertTrue(ok1)
        self.assertEqual(self.gate.hash_cache_size, 1)

        # Same frame -> distance 0 < 6 -> rejected
        ok2, r2 = self.gate.check_gate3_visual_novelty(self.real_game_img, update_cache=True)
        self.assertFalse(ok2)
        self.assertIn("duplicate_frame", r2)
        self.assertIn("0 < 6", r2)

    def test_gate3_accepts_distinct_frames(self) -> None:
        self.gate.clear_hash_cache()
        img_a = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_a, (50, 50), 30, (255, 255, 255), -1)

        img_b = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_b, (150, 150), 40, (255, 255, 255), -1)

        ok_a, _ = self.gate.check_gate3_visual_novelty(img_a, update_cache=True)
        self.assertTrue(ok_a)
        ok_b, _ = self.gate.check_gate3_visual_novelty(img_b, update_cache=True)
        self.assertTrue(ok_b)
        self.assertEqual(self.gate.hash_cache_size, 2)

    # ==========================================================================
    # 4. Factory Controller Ingestion Hardening Tests
    # ==========================================================================
    def test_factory_controller_drops_black_live_capture(self) -> None:
        fc = AutoLabelingDatasetFactory(project_root=".")
        black = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Live ingest rejected
        res_live = fc.ingest_live_frame(black)
        self.assertFalse(res_live["ingested"])
        self.assertIn("game_frame_gate_rejected", res_live["reason"])

        # Gather success rejected
        res_gather = fc.on_gather_success(current_frame=black)
        self.assertFalse(res_gather["saved"])
        self.assertIn("game_frame_gate_rejected", res_gather["reason"])

    # ==========================================================================
    # 5. Clean Dataset Purging Tests
    # ==========================================================================
    def test_clean_dataset_purges_invalid_desktop_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            train_img = td / "images" / "train"
            train_lbl = td / "labels" / "train"
            val_img = td / "images" / "val"
            val_lbl = td / "labels" / "val"
            gl_dir = td / "game_learning" / "monsters"

            for d in (train_img, train_lbl, val_img, val_lbl, gl_dir):
                d.mkdir(parents=True)

            # Valid game frame
            if self.real_game_img is not None:
                cv2.imwrite(str(train_img / "valid_aria.jpg"), self.real_game_img)
                (train_lbl / "valid_aria.txt").write_text("0 0.5 0.5 0.1 0.1\n")

            # Invalid desktop screenshot
            desktop = np.full((1080, 1920, 3), (180, 120, 40), dtype=np.uint8)
            desktop[1035:, :] = (25, 25, 25)
            cv2.imwrite(str(val_img / "desktop_capture.jpg"), desktop)
            (val_lbl / "desktop_capture.txt").write_text("10 0.2 0.2 0.1 0.1\n")

            # Invalid black screenshot in game_learning
            black = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(gl_dir / "black_bad.jpg"), black)
            (gl_dir / "black_bad.json").write_text('{"state": "error"}\n')

            # 1. Dry run
            dry_res = purge_non_game_screenshots(dataset_dir=td, dry_run=True, verbose=False)
            self.assertEqual(dry_res["invalid_purged"], 2)
            self.assertTrue((val_img / "desktop_capture.jpg").exists())

            # 2. Real remove
            remove_res = purge_non_game_screenshots(dataset_dir=td, action="remove", dry_run=False, verbose=False)
            self.assertEqual(remove_res["invalid_purged"], 2)
            self.assertFalse((val_img / "desktop_capture.jpg").exists())
            self.assertFalse((val_lbl / "desktop_capture.txt").exists())
            self.assertFalse((gl_dir / "black_bad.jpg").exists())
            self.assertFalse((gl_dir / "black_bad.json").exists())

            if self.real_game_img is not None:
                self.assertTrue((train_img / "valid_aria.jpg").exists())
                self.assertTrue((train_lbl / "valid_aria.txt").exists())

    def test_clean_dataset_quarantine_action_and_preview_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            train_img = td / "images" / "train"
            train_lbl = td / "labels" / "train"
            previews_dir = td / "previews"
            q_dir = td / "quarantine_target"

            for d in (train_img, train_lbl, previews_dir):
                d.mkdir(parents=True)

            # Invalid black screenshot with label and preview
            black = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(train_img / "bad_black.jpg"), black)
            (train_lbl / "bad_black.txt").write_text("1 0.5 0.5 0.1 0.1\n")
            (previews_dir / "bad_black.jpg").write_text("fake_preview\n")

            res_move = purge_non_game_screenshots(
                dataset_dir=td,
                action="move",
                quarantine_dir=q_dir,
                dry_run=False,
                verbose=False,
            )
            self.assertEqual(res_move["invalid_purged"], 1)
            self.assertFalse((train_img / "bad_black.jpg").exists())
            self.assertFalse((train_lbl / "bad_black.txt").exists())
            self.assertFalse((previews_dir / "bad_black.jpg").exists())

            # Verified moved into quarantine_target
            self.assertTrue((q_dir / "images" / "train" / "bad_black.jpg").exists())
            self.assertTrue((q_dir / "labels" / "train" / "bad_black.txt").exists())
            self.assertTrue((q_dir / "previews" / "bad_black.jpg").exists())

    def test_gate3_lru_cache_bounded_size(self) -> None:
        small_gate = GameFrameGate(max_hash_cache=5, min_dhash_distance=1)
        for i in range(12):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[i % 50, :] = 255
            small_gate.check_gate3_visual_novelty(img, update_cache=True)
        self.assertLessEqual(small_gate.hash_cache_size, 5)

    def test_module_level_convenience_functions(self) -> None:
        clear_frame_gate_cache()
        self.assertIsNotNone(get_default_frame_gate())
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        ok_hud, r_hud = check_visual_hud(black)
        self.assertFalse(ok_hud)
        self.assertIn("black_frame", r_hud)

        # check_process_and_focus on current workstation
        ok_proc, r_proc = check_process_and_focus()
        self.assertIsInstance(ok_proc, bool)
        self.assertIsInstance(r_proc, str)

        # check_visual_novelty
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        img[20:40, 20:40] = 200
        ok_nov1, _ = check_visual_novelty(img, update_cache=True)
        self.assertTrue(ok_nov1)
        ok_nov2, r_nov2 = check_visual_novelty(img, update_cache=True)
        self.assertFalse(ok_nov2)
        self.assertIn("duplicate_frame", r_nov2)


if __name__ == "__main__":
    unittest.main()
