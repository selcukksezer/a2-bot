# -*- coding: utf-8 -*-
"""Tests for extract_video.py (D73: F-1188, F-1189, F-1190, F-1251)."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import extract_video


def _make_video(path, seconds=9.5, fps=10.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48)
    )
    assert writer.isOpened()
    for i in range(int(seconds * fps)):
        writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    writer.release()


def _make_noise_video(path, seconds=10.0, fps=10.0, seed=7):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    assert writer.isOpened()
    for _ in range(int(seconds * fps)):
        writer.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()


def _total_jpg_bytes(out_dir):
    return sum(p.stat().st_size for p in Path(out_dir).glob("*.jpg"))


class ExtractVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.out_dir = self.tmp_path / "frames"

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_has_no_side_effects(self):
        # F-1190: importing must not open a video or create video_frames.
        repo_root = str(Path(extract_video.__file__).resolve().parent)
        env = dict(os.environ, PYTHONPATH=repo_root)
        proc = subprocess.run(
            [sys.executable, "-c", "import extract_video"],
            cwd=self.tmp.name, capture_output=True, text=True, timeout=60, env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse((self.tmp_path / "video_frames").exists())
        self.assertFalse((self.tmp_path / "frame_00s.jpg").exists())

    def test_unopenable_video_raises_instead_of_zero_frame_success(self):
        # F-1188: a missing/unopenable video must fail, not "succeed" with 0 frames.
        with self.assertRaises(RuntimeError):
            extract_video.extract_frames(
                str(self.tmp_path / "nope.mp4"), str(self.out_dir)
            )
        self.assertEqual(extract_video.main(
            ["--video", str(self.tmp_path / "nope.mp4"),
             "--out-dir", str(self.out_dir)]), 2)

    def test_final_partial_interval_is_extracted(self):
        # F-1251: a 9.5 s video must yield frames at 0, 3, 6 and 9 s
        # (the last partial 3 s slice used to be skipped).
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=9.5, fps=10.0)
        count = extract_video.extract_frames(str(video), str(self.out_dir))
        self.assertEqual(count, 4)
        for sec in (0, 3, 6, 9):
            self.assertTrue((self.out_dir / f"frame_{sec:02d}s.jpg").exists())

    def test_main_returns_zero_on_success(self):
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=6.0, fps=10.0)
        self.assertEqual(extract_video.main(
            ["--video", str(video), "--out-dir", str(self.out_dir)]), 0)
        self.assertEqual(len(list(self.out_dir.glob("frame_*.jpg"))), 2)

    def test_readiness_helpers_derive_from_real_probe(self):
        # F-1189: readiness helpers must not unconditionally return True.
        self.assertTrue(extract_video.get_extract_video_ready_flag()["ready"])
        self.assertTrue(extract_video.validate_extract_video_contract())
        self.assertEqual(
            extract_video.get_extract_video_diagnostic_snapshot()["status"], "ok")
        with mock.patch.object(
            extract_video.cv2, "VideoCapture", side_effect=RuntimeError("cv2 down")
        ):
            self.assertFalse(extract_video.get_extract_video_ready_flag()["ready"])
            self.assertFalse(extract_video.validate_extract_video_contract())
            self.assertEqual(
                extract_video.get_extract_video_diagnostic_snapshot()["status"],
                "unavailable",
            )
            self.assertFalse(
                extract_video.get_extract_video_readiness_status()["ready"])
            self.assertFalse(
                extract_video.get_extract_video_audit_summary()["audit_passed"])


    # --- D80 improvements: fractional interval, time window, JPEG quality ---

    def test_float_interval_extracts_subsecond_frames(self):
        # D80: interval 0.5 on a 9.5 s video yields 19 frames (0, 0.5, ... 9.0);
        # before, range() raised TypeError for a float step.
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=9.5, fps=10.0)
        count = extract_video.extract_frames(str(video), str(self.out_dir), 0.5)
        self.assertEqual(count, 19)
        names = sorted(p.name for p in self.out_dir.glob("frame_*.jpg"))
        self.assertEqual(len(names), 19)
        self.assertEqual(names, sorted(names))  # fixed-width names sort correctly
        self.assertTrue((self.out_dir / "frame_000.00s.jpg").exists())
        self.assertTrue((self.out_dir / "frame_000.50s.jpg").exists())
        self.assertTrue((self.out_dir / "frame_009.00s.jpg").exists())

    def test_float_interval_via_cli(self):
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=9.5, fps=10.0)
        self.assertEqual(extract_video.main(
            ["--video", str(video), "--out-dir", str(self.out_dir),
             "--interval", "0.5"]), 0)
        self.assertEqual(len(list(self.out_dir.glob("frame_*.jpg"))), 19)

    def test_time_window_extracts_only_segment(self):
        # D80: a 30 s video, window [10, 20) at interval 3 yields only
        # frames 10, 13, 16, 19 (before: whole video only, 10 frames).
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=30.0, fps=10.0)
        count = extract_video.extract_frames(
            str(video), str(self.out_dir), 3, start_sec=10, end_sec=20)
        self.assertEqual(count, 4)
        for sec in (10, 13, 16, 19):
            self.assertTrue((self.out_dir / f"frame_{sec:02d}s.jpg").exists())
        self.assertFalse((self.out_dir / "frame_07s.jpg").exists())
        self.assertFalse((self.out_dir / "frame_22s.jpg").exists())

    def test_time_window_validation_and_clamping(self):
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=30.0, fps=10.0)
        with self.assertRaises(ValueError):
            extract_video.extract_frames(str(video), str(self.out_dir), 3, start_sec=-1)
        with self.assertRaises(ValueError):
            extract_video.extract_frames(
                str(video), str(self.out_dir), 3, start_sec=10, end_sec=10)
        with self.assertRaises(ValueError):
            extract_video.extract_frames(
                str(video), str(self.out_dir), 3, start_sec=20, end_sec=10)
        # end past the video end is clamped: window [27, 999) -> only 27 s.
        count = extract_video.extract_frames(
            str(video), str(self.out_dir), 3, start_sec=27, end_sec=999)
        self.assertEqual(count, 1)
        self.assertTrue((self.out_dir / "frame_27s.jpg").exists())
        # CLI rejects a negative start with exit code 2.
        self.assertEqual(extract_video.main(
            ["--video", str(video), "--out-dir", str(self.out_dir),
             "--start", "-1"]), 2)

    def test_jpeg_quality_controls_size_and_default_is_unchanged(self):
        # D80: quality 50 output is much smaller than quality 95 on noisy
        # frames; explicit quality=95 is byte-identical to the default
        # (OpenCV default is 95), so the old default behaviour is preserved.
        video = self.tmp_path / "noise.mp4"
        _make_noise_video(video, seconds=10.0, fps=10.0)
        out95 = self.tmp_path / "q95"
        out50 = self.tmp_path / "q50"
        outdef = self.tmp_path / "qdef"
        self.assertEqual(extract_video.extract_frames(str(video), str(out95), 1, quality=95), 10)
        self.assertEqual(extract_video.extract_frames(str(video), str(out50), 1, quality=50), 10)
        self.assertEqual(extract_video.extract_frames(str(video), str(outdef), 1), 10)
        size95 = _total_jpg_bytes(out95)
        size50 = _total_jpg_bytes(out50)
        sizedef = _total_jpg_bytes(outdef)
        self.assertLess(size50, size95)
        self.assertEqual(sizedef, size95)
        with self.assertRaises(ValueError):
            extract_video.extract_frames(str(video), str(self.out_dir), 1, quality=101)
        with self.assertRaises(ValueError):
            extract_video.extract_frames(str(video), str(self.out_dir), 1, quality=-1)
        self.assertEqual(extract_video.main(
            ["--video", str(video), "--out-dir", str(self.out_dir),
             "--quality", "101"]), 2)

    def test_integer_interval_naming_unchanged(self):
        # D80: whole-second runs keep the historical frame_03s.jpg names.
        video = self.tmp_path / "route.mp4"
        _make_video(video, seconds=9.5, fps=10.0)
        self.assertEqual(
            extract_video.extract_frames(str(video), str(self.out_dir), 3), 4)
        self.assertEqual(
            sorted(p.name for p in self.out_dir.glob("frame_*.jpg")),
            ["frame_00s.jpg", "frame_03s.jpg", "frame_06s.jpg", "frame_09s.jpg"])


if __name__ == "__main__":
    unittest.main()
