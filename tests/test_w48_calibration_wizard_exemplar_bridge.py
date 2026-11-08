# -*- coding: utf-8 -*-
"""
tests/test_w48_calibration_wizard_exemplar_bridge.py
====================================================
Worker W48 Verification Suite:
Validates that ROIs and templates saved by CalibrationWizard (`ui.calibration_wizard`)
are immediately persisted and hot-reloaded through `config/calibrated_rois.json`
(`CalibratedRoisManager` / `SkillBarReader`) and `dataset/learned_exemplars.json`
(`ExemplarMemory`) without restarting the process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

from core.skill_bar_reader import SkillBarReader
from dataset_factory.exemplar_memory import (
    CalibratedRoisManager,
    ExemplarMemory,
    compute_similarity,
    extract_crop_features,
)
from ui.calibration_wizard import (
    CalibrationWizard,
    save_calibrated_roi,
    save_positive_exemplar,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def w48_sandbox() -> Dict[str, Any]:
    temp_dir = tempfile.mkdtemp(prefix="test_w48_calib_bridge_")
    base_dir = Path(temp_dir)
    rois_path = base_dir / "config" / "calibrated_rois.json"
    exemplars_path = base_dir / "dataset" / "learned_exemplars.json"
    yield {
        "base_dir": base_dir,
        "rois_path": rois_path,
        "exemplars_path": exemplars_path,
    }
    shutil.rmtree(temp_dir, ignore_errors=True)


def _touch_future_mtime(path: Path, delta_sec: float = 1.5) -> None:
    if path.exists():
        future = time.time() + delta_sec
        os.utime(path, (future, future))


def test_1_calibration_wizard_save_cropped_region_persists_roi_and_exemplar_instantly(
    qapp: QApplication, w48_sandbox: Dict[str, Any]
) -> None:
    """
    Test 1: Verify that selecting an ROI in CalibrationWizard and calling
    `save_cropped_region_as_template()` writes the template PNG, updates
    `config/calibrated_rois.json`, and persists the positive exemplar so both
    `CalibratedRoisManager` and `ExemplarMemory` read it immediately.
    """
    base_dir: Path = w48_sandbox["base_dir"]
    rois_path: Path = w48_sandbox["rois_path"]
    exemplars_path: Path = w48_sandbox["exemplars_path"]

    wizard = CalibrationWizard(base_dir=base_dir)
    try:
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        # Draw a vibrant purple-green Aria herb patch at (384, 540, 96, 96)
        roi_x, roi_y, roi_w, roi_h = 384, 540, 96, 96
        patch = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        patch[:, :] = (180, 60, 210)
        cv2.circle(patch, (48, 48), 28, (60, 220, 100), -1)
        frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = patch

        wizard.canvas.set_image(frame)
        target_label = "Kaynak: Aria Bitkisi (herb_aria) Kes"
        wizard.combo_target.setCurrentText(target_label)
        wizard.canvas.set_roi_image_rect(QRect(roi_x, roi_y, roi_w, roi_h))

        saved_tpl_path = wizard.save_cropped_region_as_template()
        assert saved_tpl_path is not None
        assert Path(saved_tpl_path).is_file()
        assert Path(saved_tpl_path).name == "herb_aria_template.png"

        # Archive copy must exist under templates/calibrated/node_aria/
        calib_arch_dir = base_dir / "templates" / "calibrated" / "node_aria"
        assert calib_arch_dir.is_dir()
        assert len(list(calib_arch_dir.glob("crop_*.png"))) == 1

        # Immediately read via CalibratedRoisManager
        rois_mgr = CalibratedRoisManager(json_path=rois_path)
        assert rois_mgr.has_roi("node_aria") is True
        entry = rois_mgr.get_roi_entry("node_aria")
        assert entry is not None
        assert entry["yolo_class_id"] == 0
        assert entry["class_name"] == "herb_aria"
        assert entry["bbox_px"] == [roi_x, roi_y, roi_w, roi_h]

        norm_rect = rois_mgr.get_normalized_rect("node_aria")
        assert norm_rect is not None
        assert norm_rect[0] == pytest.approx(384 / 1920.0, abs=1e-4)
        assert norm_rect[1] == pytest.approx(540 / 1080.0, abs=1e-4)
        assert norm_rect[2] == pytest.approx((384 + 96) / 1920.0, abs=1e-4)
        assert norm_rect[3] == pytest.approx((540 + 96) / 1080.0, abs=1e-4)

        # Immediately read via ExemplarMemory
        mem = ExemplarMemory(json_path=exemplars_path)
        assert mem.has_positive_exemplars(class_id=0) is True
        pos_list = mem.get_positive_exemplars(class_id=0)
        assert len(pos_list) >= 1
        assert pos_list[0]["class_name"] == "herb_aria"
        assert pos_list[0]["source"] == "calibration_wizard"
    finally:
        wizard.close()


def test_2_calibrated_rois_hot_reload_and_alias_resolution_across_consumers(
    w48_sandbox: Dict[str, Any]
) -> None:
    """
    Test 2: Verify that pre-initialized `CalibratedRoisManager` and `SkillBarReader`
    instances immediately detect disk writes from `save_calibrated_roi`, hot-reload
    normalized coordinates, resolve ROI aliases, and extract accurate pixel crops.
    """
    base_dir: Path = w48_sandbox["base_dir"]
    rois_path: Path = w48_sandbox["rois_path"]

    rois_mgr = CalibratedRoisManager(json_path=rois_path)
    skill_reader = SkillBarReader(calibrated_rois_path=rois_path)

    assert rois_mgr.has_roi("target_hp_bar") is False
    assert skill_reader.current_skill_bar_roi == SkillBarReader.DEFAULT_SKILL_BAR_ROI

    # Save target_hp_bar and full-width skill_bar via calibration wizard bridge
    ok_hp = save_calibrated_roi(
        base_dir=base_dir,
        key="target_hp_bar",
        target_name="Hedef Canavar Can (HP) Barı Kes",
        norm_x=0.380,
        norm_y=0.045,
        norm_w=0.240,
        norm_h=0.030,
        bbox_px=(730, 49, 460, 32),
        resolution=(1920, 1080),
        yolo_class_id=20,
        class_name="target_hp_bar",
    )
    ok_sb = save_calibrated_roi(
        base_dir=base_dir,
        key="skill_bar",
        target_name="Yetenek / Skill Bar Yuvası Kes",
        norm_x=0.310,
        norm_y=0.890,
        norm_w=0.360,
        norm_h=0.065,
        bbox_px=(595, 961, 691, 70),
        resolution=(1920, 1080),
        yolo_class_id=13,
        class_name="skill_ui_prompt",
    )
    assert ok_hp is True and ok_sb is True
    _touch_future_mtime(rois_path)

    # Without recreating rois_mgr, verify instant reload and alias lookup ('mob_nameplate' -> 'target_hp_bar')
    assert rois_mgr.has_roi("target_hp_bar") is True
    assert rois_mgr.has_roi("mob_nameplate") is True
    rect_hp = rois_mgr.get_normalized_rect("mob_nameplate")
    assert rect_hp == pytest.approx((0.380, 0.045, 0.620, 0.075), abs=1e-4)

    # Verify pixel crop extraction matches the calibrated ROI region
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[int(0.045 * 1080) : int(0.075 * 1080), int(0.380 * 1920) : int(0.620 * 1920)] = (0, 0, 250)
    crop = rois_mgr.get_pixel_crop(frame, "target_hp_bar")
    assert crop is not None and crop.size > 0
    assert float(np.mean(crop[:, :, 2])) > 240.0

    # Without recreating SkillBarReader, verify dynamic mtime reload of SKILL_BAR_ROI
    updated_sb_roi = skill_reader.current_skill_bar_roi
    assert updated_sb_roi[0] == pytest.approx(0.310, abs=1e-3)
    assert updated_sb_roi[1] == pytest.approx(0.890, abs=1e-3)
    assert updated_sb_roi[2] == pytest.approx(0.670, abs=1e-3)
    assert updated_sb_roi[3] == pytest.approx(0.955, abs=1e-3)


def test_3_exemplar_memory_hot_reload_and_similarity_matching_from_wizard(
    w48_sandbox: Dict[str, Any]
) -> None:
    """
    Test 3: Verify that an already-running `ExemplarMemory` instance hot-reloads
    new exemplars written by CalibrationWizard's `save_positive_exemplar()` in < 50ms
    and computes high similarity against the saved CalibrationWizard histogram record.
    """
    base_dir: Path = w48_sandbox["base_dir"]
    exemplars_path: Path = w48_sandbox["exemplars_path"]

    mem = ExemplarMemory(json_path=exemplars_path)
    assert mem.has_positive_exemplars(class_id=3) is False

    # Create a distinctive golden-copper Orichalcum ore crop (class_id=3)
    ore_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    ore_crop[:, :] = (25, 140, 235)
    cv2.rectangle(ore_crop, (14, 14), (50, 50), (40, 195, 255), -1)

    saved = save_positive_exemplar(
        crop=ore_crop,
        class_id=3,
        class_name="ore_orichalcum",
        base_dir=base_dir,
    )
    assert saved is True
    _touch_future_mtime(exemplars_path)

    t0 = time.perf_counter()
    reloaded = mem.check_reload()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert reloaded is True
    assert elapsed_ms < 50.0, f"ExemplarMemory hot-reload took {elapsed_ms:.2f}ms (expected < 50ms)"
    assert mem.has_positive_exemplars(class_id=3) is True

    stats = mem.get_stats()
    assert stats["positive_by_class"].get(3, 0) >= 1

    wizard_ex = mem.get_positive_exemplars(class_id=3)[0]
    assert "hist_h" in wizard_ex and "hist_s" in wizard_ex and "hist_v" in wizard_ex

    # Verify multi-factor similarity between live crop features and CalibrationWizard exemplar
    live_feats = extract_crop_features(ore_crop)
    assert live_feats is not None
    sim = compute_similarity(live_feats, wizard_ex)
    assert sim >= 0.85, f"Expected high similarity >= 0.85 for identical crop, got {sim:.4f}"


def test_4_wizard_template_matching_and_registered_dataset_hand_label_bridge(
    qapp: QApplication, w48_sandbox: Dict[str, Any]
) -> None:
    """
    Test 4: Verify that saving an interaction prompt template via `CalibrationWizard`
    enables immediate `test_template_matching()` verification on the canvas while
    simultaneously populating `calibrated_rois.json`, `ExemplarMemory`, and
    trusted hand-labels in `dataset/labels/train` and `dataset/review/train`.
    """
    base_dir: Path = w48_sandbox["base_dir"]
    rois_path: Path = w48_sandbox["rois_path"]
    exemplars_path: Path = w48_sandbox["exemplars_path"]

    wizard = CalibrationWizard(base_dir=base_dir)
    try:
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
        # Draw a distinctive F/E gather pill at (600, 380, 84, 34)
        bx, by, bw, bh = 600, 380, 84, 34
        pill = np.full((bh, bw, 3), (25, 28, 35), dtype=np.uint8)
        cv2.rectangle(pill, (2, 2), (bw - 3, bh - 3), (0, 215, 255), 2)
        cv2.circle(pill, (18, 17), 9, (240, 240, 240), -1)
        cv2.rectangle(pill, (34, 10), (74, 24), (220, 230, 240), -1)
        frame[by : by + bh, bx : bx + bw] = pill

        wizard.canvas.set_image(frame)
        wizard.combo_target.setCurrentText("F/E Toplama Butonu Şablonu Kes")
        wizard.canvas.set_roi_image_rect(QRect(bx, by, bw, bh))
        wizard.slider_threshold.setValue(80)

        out_path = wizard.save_cropped_region_as_template()
        assert out_path is not None and Path(out_path).exists()

        # Run live template matching test inside the wizard
        match_res = wizard.test_template_matching()
        assert match_res["found"] is True
        assert match_res["score"] >= 0.95
        assert match_res["bbox"] == (bx, by, bw, bh)

        # Verify ROI & ExemplarMemory bridge
        rois_mgr = CalibratedRoisManager(json_path=rois_path)
        assert rois_mgr.has_roi("gather_prompt") is True
        assert rois_mgr.get_roi_entry("gather_prompt")["yolo_class_id"] == 7

        mem = ExemplarMemory(json_path=exemplars_path)
        assert mem.has_positive_exemplars(class_id=7) is True

        # Verify trusted hand-label registration in dataset/
        labels_dir = base_dir / "dataset" / "labels" / "train"
        review_dir = base_dir / "dataset" / "review" / "train"
        lbl_files = list(labels_dir.glob("calib_f_gather_prompt_*.txt"))
        rev_files = list(review_dir.glob("calib_f_gather_prompt_*.json"))
        assert len(lbl_files) == 1
        assert len(rev_files) == 1

        rev_payload = json.loads(rev_files[0].read_text(encoding="utf-8"))
        assert rev_payload["hand_label"] is True
        assert len(rev_payload["trusted"]) == 1
        assert rev_payload["trusted"][0]["class_id"] == 7
        assert rev_payload["trusted"][0]["trusted"] is True
    finally:
        wizard.close()


def test_5_auto_calibrate_loaded_frame_end_to_end_rois_and_exemplars_sync(
    qapp: QApplication, w48_sandbox: Dict[str, Any]
) -> None:
    """
    Test 5: Verify that `CalibrationWizard.auto_calibrate_loaded_frame()` automatically
    detects HUD bars in a loaded game frame, saves their templates, and immediately
    synchronizes both `CalibratedRoisManager` and `ExemplarMemory` on disk.
    """
    base_dir: Path = w48_sandbox["base_dir"]
    rois_path: Path = w48_sandbox["rois_path"]
    exemplars_path: Path = w48_sandbox["exemplars_path"]

    # Pre-initialize consumers before running auto-calibration to test live hot-reload
    rois_mgr = CalibratedRoisManager(json_path=rois_path)
    mem = ExemplarMemory(json_path=exemplars_path)

    wizard = CalibrationWizard(base_dir=base_dir)
    try:
        fh, fw = 720, 1280
        frame = np.full((fh, fw, 3), 35, dtype=np.uint8)

        # 1. Draw Player HP Bar (red) in bottom HUD band (x: 0.345..0.485, y: 0.880..0.910)
        hp_x1, hp_x2 = int(round(fw * 0.3450)), int(round(fw * 0.4850))
        hp_y1, hp_y2 = int(round(fh * 0.8800)), int(round(fh * 0.9100))
        frame[hp_y1:hp_y2, hp_x1:hp_x2] = (20, 25, 220)

        # 2. Draw Player MP Bar (blue) in bottom HUD band (x: 0.515..0.640, y: 0.880..0.910)
        mp_x1, mp_x2 = int(round(fw * 0.5150)), int(round(fw * 0.6400))
        mp_y1, mp_y2 = int(round(fh * 0.8800)), int(round(fh * 0.9100))
        frame[mp_y1:mp_y2, mp_x1:mp_x2] = (225, 90, 20)

        # 3. Draw Target HP Bar (bright red horizontal bar) in top HUD band (x: 0.42..0.58, y: 0.05..0.065)
        thp_x1, thp_x2 = int(round(fw * 0.42)), int(round(fw * 0.58))
        thp_y1, thp_y2 = int(round(fh * 0.05)), int(round(fh * 0.065))
        frame[thp_y1:thp_y2, thp_x1:thp_x2] = (20, 20, 230)

        wizard.canvas.set_image(frame)
        wizard.slider_threshold.setValue(65)

        results = wizard.auto_calibrate_loaded_frame()
        saved_targets = {r["target"]: r for r in results if r.get("saved")}

        assert "Oyuncu Can (HP) Barı Kes" in saved_targets
        assert "Oyuncu Mana (MP) Barı Kes" in saved_targets
        assert "Hedef Canavar Can (HP) Barı Kes" in saved_targets

        _touch_future_mtime(rois_path)
        _touch_future_mtime(exemplars_path)

        # Verify immediate hot-reload in CalibratedRoisManager
        assert rois_mgr.has_roi("player_hp_bar") is True
        assert rois_mgr.has_roi("player_mp_bar") is True
        assert rois_mgr.has_roi("target_hp_bar") is True

        # Verify immediate hot-reload in ExemplarMemory (classes 16, 17, 20)
        assert mem.has_positive_exemplars(class_id=16) is True
        assert mem.has_positive_exemplars(class_id=17) is True
        assert mem.has_positive_exemplars(class_id=20) is True
    finally:
        wizard.close()
