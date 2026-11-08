# -*- coding: utf-8 -*-
"""
test_skill_bar_calibrated_exemplar.py
======================================
Tests for Agent C2:
- Dynamic reload of SKILL_BAR_ROI from config/calibrated_rois.json with mtime check.
- Integration of ExemplarMemory (classes 20: skill_slot_ready and 21: skill_slot_cooldown)
  into slot ready/cooldown detection and confidence scoring.
- Cooldown lockout tracking and backward compatibility.
"""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import cv2
import numpy as np

from core.skill_bar_reader import SkillBarReader, _SkillBarRoiDescriptor
from dataset_factory.exemplar_memory import ExemplarMemory


class TestSkillBarCalibratedExemplar(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.calibrated_rois_path = self.temp_path / "calibrated_rois.json"
        self.exemplar_json_path = self.temp_path / "learned_exemplars.json"

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_default_skill_bar_roi_when_file_missing(self):
        """When calibrated_rois.json does not exist, defaults to (0.355, 0.905, 0.655, 0.955)."""
        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        self.assertEqual(reader.current_skill_bar_roi, reader.DEFAULT_SKILL_BAR_ROI)
        self.assertEqual(reader.SKILL_BAR_ROI, reader.DEFAULT_SKILL_BAR_ROI)
        self.assertEqual(reader.SKILL_BAR_ROI[1], 0.905)

    def test_dynamic_reload_on_mtime_change(self):
        """When config/calibrated_rois.json is modified on disk, reader automatically updates ROI."""
        # 1. Create initial calibrated ROIs with custom skill_bar
        initial_data = {
            "rois": {
                "skill_bar": {
                    "key": "skill_bar",
                    "normalized_roi": [0.300, 0.880, 0.380, 0.080],
                }
            }
        }
        with open(self.calibrated_rois_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        roi1 = reader.current_skill_bar_roi
        # Expected: x1=0.300, y1=0.880, x2=0.680, y2=0.960
        self.assertAlmostEqual(roi1[0], 0.300, places=3)
        self.assertAlmostEqual(roi1[1], 0.880, places=3)
        self.assertAlmostEqual(roi1[2], 0.680, places=3)
        self.assertAlmostEqual(roi1[3], 0.960, places=3)
        self.assertEqual(reader.SKILL_BAR_ROI, roi1)

        # 2. Modify calibrated_rois.json on disk with new coordinates
        time.sleep(0.05)  # Ensure mtime resolution
        updated_data = {
            "rois": {
                "skill_bar": {
                    "key": "skill_bar",
                    "normalized_roi": [0.320, 0.890, 0.340, 0.070],
                }
            }
        }
        with open(self.calibrated_rois_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f)
        # Touch mtime to guarantee timestamp advance on fast filesystems
        now_ts = time.time() + 1.0
        os.utime(self.calibrated_rois_path, (now_ts, now_ts))

        # 3. Query without recreating reader — dynamic mtime check must reload
        roi2 = reader.current_skill_bar_roi
        self.assertAlmostEqual(roi2[0], 0.320, places=3)
        self.assertAlmostEqual(roi2[1], 0.890, places=3)
        self.assertAlmostEqual(roi2[2], 0.660, places=3)
        self.assertAlmostEqual(roi2[3], 0.960, places=3)
        self.assertEqual(reader.SKILL_BAR_ROI, roi2)

    def test_single_slot_does_not_squash_full_bar(self):
        """If calibrated_rois.json only contains single-slot width (e.g. 0.0375), reader preserves full bar."""
        slot_only_data = {
            "rois": {
                "skill_slot": {
                    "key": "skill_slot",
                    "normalized_roi": [0.4675, 0.905, 0.0375, 0.050],
                }
            }
        }
        with open(self.calibrated_rois_path, "w", encoding="utf-8") as f:
            json.dump(slot_only_data, f)

        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        roi = reader.current_skill_bar_roi
        # Width must be >= 0.20 (full bar), not 0.0375
        self.assertGreaterEqual(roi[2] - roi[0], 0.20)
        self.assertEqual(roi, reader.DEFAULT_SKILL_BAR_ROI)

    def test_exemplar_memory_ready_boosts_slot(self):
        """When an exemplar for class 20 (skill_slot_ready) matches, slot is marked ready with high confidence."""
        mem = ExemplarMemory(json_path=self.exemplar_json_path)

        # Create a bright/saturated green-gold skill icon exemplar
        exemplar_crop = np.zeros((48, 48, 3), dtype=np.uint8)
        exemplar_crop[:, :] = (30, 200, 210)  # Golden BGR
        cv2.circle(exemplar_crop, (24, 24), 16, (20, 240, 255), -1)

        mem.add_positive_exemplar(exemplar_crop, class_id=20, class_name="skill_slot_ready")

        # Create test frame with matching icon in slot '1' (x1..x2)
        reader = SkillBarReader(
            calibrated_rois_path=self.calibrated_rois_path,
            exemplar_memory=mem,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = reader.current_skill_bar_roi
        fh, fw = frame.shape[:2]
        x1 = int(roi[0] * fw)
        y1 = int(roi[1] * fh)
        x2 = int(roi[2] * fw)
        y2 = int(roi[3] * fh)
        slot_w = (x2 - x1) // 8

        # Draw matching golden icon in slot '1' (first slot)
        resized_icon = cv2.resize(exemplar_crop, (slot_w, y2 - y1))
        frame[y1:y2, x1 : x1 + slot_w] = resized_icon

        results = reader.inspect_skill_bar(frame)
        self.assertIn("1", results)
        slot1 = results["1"]

        self.assertTrue(slot1["ready"])
        self.assertTrue(slot1["off_cooldown"])
        self.assertEqual(slot1["exemplar_state"], "ready")
        self.assertGreaterEqual(slot1["exemplar_ready_sim"], 0.70)
        self.assertGreaterEqual(slot1["score"], 0.75)

    def test_exemplar_memory_cooldown_suppresses_slot(self):
        """When an exemplar for class 21 (skill_slot_cooldown) matches, slot is marked not ready with low confidence."""
        mem = ExemplarMemory(json_path=self.exemplar_json_path)

        # Create a darkened cooldown sweep icon exemplar (dark grey/blue)
        cooldown_crop = np.zeros((48, 48, 3), dtype=np.uint8)
        cooldown_crop[:, :] = (40, 35, 30)  # Dark BGR
        cv2.circle(cooldown_crop, (24, 24), 16, (55, 50, 45), -1)

        mem.add_positive_exemplar(cooldown_crop, class_id=21, class_name="skill_slot_cooldown")

        reader = SkillBarReader(
            calibrated_rois_path=self.calibrated_rois_path,
            exemplar_memory=mem,
            min_brightness_ready=30.0,  # Lower threshold so only exemplar suppresses it
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = reader.current_skill_bar_roi
        fh, fw = frame.shape[:2]
        x1 = int(roi[0] * fw)
        y1 = int(roi[1] * fh)
        x2 = int(roi[2] * fw)
        y2 = int(roi[3] * fh)
        slot_w = (x2 - x1) // 8

        # Draw cooldown icon in slot '2' (second slot)
        resized_cd = cv2.resize(cooldown_crop, (slot_w, y2 - y1))
        frame[y1:y2, x1 + slot_w : x1 + 2 * slot_w] = resized_cd

        results = reader.inspect_skill_bar(frame)
        self.assertIn("2", results)
        slot2 = results["2"]

        self.assertFalse(slot2["ready"])
        self.assertFalse(slot2["off_cooldown"])
        self.assertEqual(slot2["exemplar_state"], "cooldown")
        self.assertGreaterEqual(slot2["exemplar_cooldown_sim"], 0.70)
        self.assertLessEqual(slot2["score"], 0.20)

    def test_negative_exemplar_drops_invalid_slot(self):
        """When an empty/corrupted slot matches a negative exemplar, it is suppressed as not ready."""
        mem = ExemplarMemory(json_path=self.exemplar_json_path)

        corrupt_crop = np.zeros((48, 48, 3), dtype=np.uint8)
        corrupt_crop[:, :] = (128, 128, 128)  # Flat gray noise
        mem.add_negative_exemplar(corrupt_crop, class_id=20, class_name="false_slot_rejection")

        reader = SkillBarReader(
            calibrated_rois_path=self.calibrated_rois_path,
            exemplar_memory=mem,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = reader.current_skill_bar_roi
        fh, fw = frame.shape[:2]
        x1 = int(roi[0] * fw)
        y1 = int(roi[1] * fh)
        x2 = int(roi[2] * fw)
        y2 = int(roi[3] * fh)
        slot_w = (x2 - x1) // 8

        frame[y1:y2, x1 : x1 + slot_w] = cv2.resize(corrupt_crop, (slot_w, y2 - y1))

        results = reader.inspect_skill_bar(frame)
        self.assertIn("1", results)
        slot1 = results["1"]
        self.assertFalse(slot1["ready"])
        self.assertEqual(slot1["exemplar_state"], "negative")

    def test_cooldown_lockout_tracking(self):
        """mark_slot_pressed applies millisecond lockout and tracks remaining time."""
        reader = SkillBarReader(
            calibrated_rois_path=self.calibrated_rois_path,
            cooldown_lockout_sec=0.20,
        )
        # Create textured active frame (with brightness, saturation, and contrast)
        rng = np.random.RandomState(42)
        frame = rng.randint(60, 220, (1080, 1920, 3), dtype=np.uint8)
        res_before = reader.inspect_skill_bar(frame)
        self.assertTrue(res_before["1"]["ready"])
        self.assertEqual(res_before["1"]["lockout_remaining_sec"], 0.0)

        # Mark slot '1' pressed
        reader.mark_slot_pressed("1")
        res_locked = reader.inspect_skill_bar(frame)
        self.assertFalse(res_locked["1"]["ready"])
        self.assertTrue(res_locked["1"]["recently_pressed"])
        self.assertGreater(res_locked["1"]["lockout_remaining_sec"], 0.0)

        # Wait for lockout expiration
        time.sleep(0.22)
        res_after = reader.inspect_skill_bar(frame)
        self.assertTrue(res_after["1"]["ready"])
        self.assertFalse(res_after["1"]["recently_pressed"])
        self.assertEqual(res_after["1"]["lockout_remaining_sec"], 0.0)

    def test_backward_compatibility_descriptor_and_slot_center(self):
        """SKILL_BAR_ROI tuple indexing, slot_center_px, and diagnostics snapshot remain backward-compatible."""
        reader = SkillBarReader(calibrated_rois_path=self.calibrated_rois_path)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 1. Tuple indexing on instance and class
        self.assertIsInstance(reader.SKILL_BAR_ROI, tuple)
        self.assertEqual(len(reader.SKILL_BAR_ROI), 4)
        self.assertIsInstance(SkillBarReader.SKILL_BAR_ROI, tuple)

        # 2. slot_center_px inside ROI band
        p1 = reader.slot_center_px(frame, "1", jitter_px=0)
        pr = reader.slot_center_px(frame, "R", jitter_px=0)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(pr)
        self.assertLess(p1[0], pr[0])
        self.assertGreaterEqual(p1[1], int(1080 * reader.SKILL_BAR_ROI[1]))
        self.assertLessEqual(p1[1], int(1080 * reader.SKILL_BAR_ROI[3]))

        # 3. Diagnostic snapshot
        snapshot = reader.get_skill_bar_reader_diagnostic_snapshot()
        self.assertEqual(snapshot["module"], "skill_bar_reader")
        self.assertEqual(snapshot["status"], "ok")
        self.assertIn("current_roi", snapshot)
        self.assertIn("exemplar_memory_active", snapshot)


if __name__ == "__main__":
    unittest.main()
