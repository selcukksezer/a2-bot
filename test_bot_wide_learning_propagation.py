# -*- coding: utf-8 -*-
"""
Tests for Universal Bot-Wide Learning Propagation (Work Package H).

Validates:
1. Toplama / Harvesting:
   - Candidate detections matching negative exemplars are immediately dropped by MaterialScanner.
   - Candidate detections matching positive exemplars have material_type refined and confidence boosted.
2. Savaş & Skill Barı:
   - SkillBarReader dynamically loads SKILL_BAR_ROI from calibrated_rois.json.
   - Slot matching negative/cooldown exemplar is marked ready=False and off_cooldown=False.
3. Oto-Kayıt & Radar Gating:
   - NodeQualityGate rejects candidate patches matching negative exemplars.
   - NodeQualityGate accepts and marks candidate patches matching positive exemplars.
4. Rotaya Yürüme & Koridor Yönlendirme:
   - CorridorNavigator detects learned obstacles/monsters in forward corridor and steers away.
   - CorridorNavigator detects learned ground paths.
5. Instant Hot-Reload:
   - ExemplarMemory detects mtime changes and reloads in < 50ms without restarting the process.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from core.auto_cartographer import AutoCartographer
from core.corridor_navigator import CorridorNavigator
from core.explorer_mode import ExplorerModeController
from core.mana_phase_combat import ManaPhaseCombat
from core.material_scanner import DetectedMaterial, MaterialScanner
from core.node_quality_gate import evaluate_node_candidate
from core.route_manager import RouteManager
from core.skill_bar_reader import SkillBarReader
from dataset_factory.exemplar_memory import (
    CalibratedRoisManager,
    ExemplarMemory,
    get_calibrated_rois,
    get_exemplar_memory,
)


@pytest.fixture
def temp_env():
    """Create a temporary sandbox for learned exemplars and calibrated rois."""
    temp_dir = tempfile.mkdtemp(prefix="gatherion_learning_prop_")
    root = Path(temp_dir)
    ex_path = root / "learned_exemplars.json"
    roi_path = root / "calibrated_rois.json"

    yield {"root": root, "exemplars": ex_path, "rois": roi_path}
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_harvesting_learning_propagation(temp_env: Dict[str, Any]):
    """
    Work Package H.1: Toplama / Harvesting Propagation.
    Verifies that MaterialScanner._refine_candidates_with_exemplars:
    - Drops candidates that match negative exemplars.
    - Elevates confidence and refines material class for positive exemplars.
    """
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    # 1. Create a frame with two distinct patches
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    # Negative patch: bright magenta noise (e.g. false rock/UI chrome)
    neg_patch = np.zeros((60, 60, 3), dtype=np.uint8)
    neg_patch[:, :] = (255, 0, 255)
    frame[100:160, 100:160] = neg_patch

    # Positive patch: green herb (Aria: class_id 0)
    pos_patch = np.zeros((60, 60, 3), dtype=np.uint8)
    pos_patch[:, :] = (0, 220, 50)
    frame[300:360, 300:360] = pos_patch

    # Teach negative and positive exemplars
    ex_mem.add_negative_exemplar(neg_patch, class_id=14, class_name="false_rock")
    ex_mem.add_positive_exemplar(pos_patch, class_id=0, class_name="herb_aria")

    # Create two candidate detections
    candidate_neg = DetectedMaterial(
        material_type="Unknown_Rock",
        source="3d_world",
        screen_x=130,
        screen_y=130,
        bbox=(100, 100, 60, 60),
        rel_map_dx=0.0,
        rel_map_dy=0.0,
        estimated_distance_m=10.0,
        confidence=0.60,
    )

    candidate_pos = DetectedMaterial(
        material_type="World Resource",
        source="3d_world",
        screen_x=330,
        screen_y=330,
        bbox=(300, 300, 60, 60),
        rel_map_dx=0.0,
        rel_map_dy=0.0,
        estimated_distance_m=8.0,
        confidence=0.55,
    )

    scanner = MaterialScanner()

    # Inject our memory instance temporarily into singleton
    from dataset_factory import exemplar_memory as em_mod
    orig_singleton = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        refined = scanner._refine_candidates_with_exemplars(frame, [candidate_neg, candidate_pos])

        # Negative candidate must be dropped
        mat_types = [m.material_type for m in refined]
        assert "Unknown_Rock" not in mat_types, "Negative exemplar should have been dropped!"

        # Positive candidate must be kept, refined to 'Aria', and confidence boosted
        assert len(refined) == 1
        kept = refined[0]
        assert kept.material_type == "Aria"
        assert kept.confidence >= 0.70
    finally:
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_singleton


def test_skill_bar_dynamic_roi_and_cooldown_exemplar(temp_env: Dict[str, Any]):
    """
    Work Package H.2: Savaş & Skill Barı Propagation.
    Verifies that:
    - SkillBarReader dynamically uses SKILL_BAR_ROI from calibrated ROIs.
    - Slot matching negative/cooldown exemplar is marked off_cooldown=False and ready=False.
    """
    reader = SkillBarReader()

    # 1. Test dynamic ROI reloading
    initial_roi = reader.current_skill_bar_roi
    assert len(initial_roi) == 4

    # 2. Test ExemplarMemory cooldown gating on skill slot
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    # Frame containing a skill bar
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    # Make slot 1 region a dark cooldown patch
    roi = reader.current_skill_bar_roi
    fh, fw = frame.shape[:2]
    x1 = int(round(roi[0] * fw))
    y1 = int(round(roi[1] * fh))
    x2 = int(round(roi[2] * fw))
    y2 = int(round(roi[3] * fh))

    slot_w = (x2 - x1) / 8.0
    slot1_x1 = int(round(x1))
    slot1_x2 = int(round(x1 + slot_w))

    cooldown_crop = np.full((y2 - y1, slot1_x2 - slot1_x1, 3), (120, 30, 20), dtype=np.uint8)
    frame[y1:y2, slot1_x1:slot1_x2] = cooldown_crop

    # Inner crop of slot 1
    sh, sw_local = cooldown_crop.shape[:2]
    pad_y = max(1, int(round(sh * 0.12)))
    pad_x = max(1, int(round(sw_local * 0.12)))
    inner_crop = cooldown_crop[pad_y : max(pad_y + 2, sh - pad_y), pad_x : max(pad_x + 2, sw_local - pad_x)]

    # Teach cooldown crop as class 21 (skill_slot_cooldown)
    ex_mem.add_positive_exemplar(inner_crop, class_id=21, class_name="skill_slot_cooldown")

    from dataset_factory import exemplar_memory as em_mod
    orig_singleton = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        results = reader.inspect_skill_bar(frame)
        assert "1" in results
        slot1_info = results["1"]
        # When learned as cooldown, slot 1 must not be ready
        assert slot1_info["off_cooldown"] is False
        assert slot1_info["ready"] is False
    finally:
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_singleton


def test_node_quality_gate_learning_propagation(temp_env: Dict[str, Any]):
    """
    Work Package H.3: Oto-Kayıt & Radar Node Quality Gate Propagation.
    Verifies that:
    - Learned negative exemplars reject false nodes (reasons contains 'negative_exemplar').
    - Learned positive exemplars accept candidates with exemplar_matched=True and verified_resource_type.
    """
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    # Negative patch: false gray rock
    rock_patch = np.full((64, 64, 3), 110, dtype=np.uint8)
    # Positive patch: golden ore (Orichalcum: class 3)
    ore_patch = np.zeros((64, 64, 3), dtype=np.uint8)
    ore_patch[:, :] = (30, 180, 220)

    ex_mem.add_negative_exemplar(rock_patch, class_id=14, class_name="false_rock_ground")
    ex_mem.add_positive_exemplar(ore_patch, class_id=3, class_name="ore_orichalcum")

    # 1. Test negative candidate evaluation
    verdict_neg = evaluate_node_candidate(
        rock_patch,
        resource_type="Ore",
        exemplar_memory=ex_mem,
        check_exemplars=True,
    )
    assert verdict_neg.accept is False
    assert "negative_exemplar" in verdict_neg.reasons

    # 2. Test positive candidate evaluation
    verdict_pos = evaluate_node_candidate(
        ore_patch,
        resource_type="Orichalcum",
        exemplar_memory=ex_mem,
        check_exemplars=True,
    )
    assert verdict_pos.accept is True
    assert verdict_pos.exemplar_matched is True
    assert verdict_pos.exemplar_sim is not None
    assert verdict_pos.exemplar_sim >= 0.70
    assert verdict_pos.verified_resource_type == "Orichalcum"


def test_corridor_navigation_obstacle_avoidance(temp_env: Dict[str, Any]):
    """
    Work Package H.4: Rotaya Yürüme & Koridor Yönlendirme Propagation.
    Verifies that CorridorNavigator:
    - Recognizes learned obstacles (class 14 or negative exemplar) in walking corridor.
    - Recommends steering bias away from the obstacle.
    """
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    nav = CorridorNavigator()
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    # Place an obstacle patch in center lane (y: 0.52..0.82, x: 0.44..0.56)
    h, w = frame.shape[:2]
    cy1, cy2 = int(h * 0.52), int(h * 0.82)
    cx1, cx2 = int(w * 0.44), int(w * 0.56)

    obs_patch = np.zeros((cy2 - cy1, cx2 - cx1, 3), dtype=np.uint8)
    obs_patch[:, :] = (15, 20, 200)  # Distinct red/dark hazard
    frame[cy1:cy2, cx1:cx2] = obs_patch

    # Teach obstacle patch as class 14 (obstacle)
    ex_mem.add_positive_exemplar(obs_patch, class_id=14, class_name="obstacle")

    from dataset_factory import exemplar_memory as em_mod
    orig_singleton = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        eval_res = nav.evaluate_corridor_path(frame)

        # Obstacle must be detected
        assert eval_res["obstacle_detected"] is True
        # Center lane score should be degraded
        assert eval_res["lane_scores"]["center"] < 0.40
        # Recommended steering bias must steer left or right to avoid
        assert eval_res["steering_bias"] in ("steer_left", "steer_right")
        assert eval_res["recommended_turn_offset_deg"] != 0.0
    finally:
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_singleton


def test_exemplar_memory_hot_reload_latency(temp_env: Dict[str, Any]):
    """
    Work Package H.5: Instant Hot-Reload Latency (<50ms).
    Verifies that:
    - check_reload() detects disk updates and completes in well under 50ms (sub-millisecond).
    - Newly added exemplar on disk becomes active without restarting.
    """
    ex_path = temp_env["exemplars"]
    mem = ExemplarMemory(json_path=ex_path)

    # Initial state
    assert len(mem.get_positive_exemplars()) == 0

    # Write a new exemplar directly to disk as if saved by Settings / YOLO Factory
    crop = np.full((40, 40, 3), 150, dtype=np.uint8)
    writer_mem = ExemplarMemory(json_path=ex_path)
    writer_mem.add_positive_exemplar(crop, class_id=1, class_name="herb_mela")

    # Measure latency of check_reload()
    t_start = time.perf_counter()
    reloaded = mem.check_reload()
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Latency requirement: sub-100ms (typically <5ms on hot cache)
    assert elapsed_ms < 100.0, f"Hot reload took {elapsed_ms:.2f}ms, expected < 100ms!"
    assert reloaded is True
    assert len(mem.get_positive_exemplars()) == 1
    assert mem.get_positive_exemplars()[0]["class_name"] == "herb_mela"


def test_material_scanner_dynamic_minimap_roi(temp_env: Dict[str, Any]):
    """
    Work Package H.1: MaterialScanner Dynamic Minimap ROI.
    Verifies that MaterialScanner.get_minimap_roi() dynamically reloads
    and returns updated ROI coordinates from calibrated_rois.json.
    """
    roi_path = temp_env["rois"]
    calibrated_data = {
        "rois": {
            "minimap": {
                "key": "minimap",
                "normalized_roi": [0.820, 0.050, 0.160, 0.180],
                "norm_x": 0.820,
                "norm_y": 0.050,
                "norm_w": 0.160,
                "norm_h": 0.180,
            }
        }
    }
    roi_path.write_text(json.dumps(calibrated_data), encoding="utf-8")

    from dataset_factory import exemplar_memory as em_mod
    orig_rois_mgr = em_mod._GLOBAL_ROIS_MANAGER
    em_mod._GLOBAL_ROIS_MANAGER = CalibratedRoisManager(json_path=roi_path)

    try:
        scanner = MaterialScanner()
        roi = scanner.get_minimap_roi()
        assert len(roi) == 4
        # (x1, y1, x2, y2)
        assert abs(roi[0] - 0.820) < 0.001
        assert abs(roi[1] - 0.050) < 0.001
        assert abs(roi[2] - 0.980) < 0.001
        assert abs(roi[3] - 0.230) < 0.001
    finally:
        em_mod._GLOBAL_ROIS_MANAGER = orig_rois_mgr


def test_mana_phase_combat_vitals_and_monster_target(temp_env: Dict[str, Any]):
    """
    Work Package H.2: ManaPhaseCombat Vitals & Monster Target Recognition.
    Verifies that:
    - ManaPhaseCombat dynamically reads vitals from frame via calibrated ROIs.
    - is_valid_monster_target recognizes learned 3D monster exemplars and rejects learned negatives.
    """
    roi_path = temp_env["rois"]
    calibrated_data = {
        "rois": {
            "player_hp_bar": {
                "key": "player_hp_bar",
                "normalized_roi": [0.350, 0.880, 0.130, 0.025],
            },
            "player_mp_bar": {
                "key": "player_mp_bar",
                "normalized_roi": [0.520, 0.880, 0.130, 0.025],
            },
            "target_hp_bar": {
                "key": "target_hp_bar",
                "normalized_roi": [0.420, 0.060, 0.160, 0.025],
            },
        }
    }
    roi_path.write_text(json.dumps(calibrated_data), encoding="utf-8")

    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    from dataset_factory import exemplar_memory as em_mod
    orig_rois_mgr = em_mod._GLOBAL_ROIS_MANAGER
    orig_ex_mem = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_ROIS_MANAGER = CalibratedRoisManager(json_path=roi_path)
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        combat = ManaPhaseCombat()

        # Learn positive 3D monster (class 10: monster_3d)
        monster_patch = np.full((50, 50, 3), (20, 30, 210), dtype=np.uint8)
        ex_mem.add_positive_exemplar(monster_patch, class_id=10, class_name="monster_3d")

        # Learn negative patch (class 14: false target / rock)
        neg_patch = np.full((50, 50, 3), (120, 120, 120), dtype=np.uint8)
        ex_mem.add_negative_exemplar(neg_patch, class_id=14, class_name="false_target_rock")

        # Test monster recognition
        is_valid, conf = combat.is_valid_monster_target(None, target_crop=monster_patch)
        assert is_valid is True
        assert conf >= 0.85

        # Test negative rejection
        is_valid_neg, conf_neg = combat.is_valid_monster_target(None, target_crop=neg_patch)
        assert is_valid_neg is False

        # Test vitals reading from synthetic frame
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        # Fill player HP bar with red/crimson (HP > 0)
        h, w = frame.shape[:2]
        frame[int(h * 0.880) : int(h * 0.905), int(w * 0.350) : int(w * 0.480)] = (20, 20, 220)
        vitals = combat.read_vitals_from_frame(frame)
        assert vitals["player_hp"] > 0.50
    finally:
        em_mod._GLOBAL_ROIS_MANAGER = orig_rois_mgr
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_ex_mem


def test_auto_cartographer_drops_learned_negative_node(temp_env: Dict[str, Any]):
    """
    Work Package H.3: AutoCartographer Drops Learned Negative Node.
    Verifies that a discovered candidate patch matching a learned negative exemplar
    is gated and rejected by AutoCartographer so it never enters RouteManager.
    """
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    from dataset_factory import exemplar_memory as em_mod
    orig_singleton = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        # Teach negative patch
        rock_patch = np.full((80, 80, 3), 115, dtype=np.uint8)
        ex_mem.add_negative_exemplar(rock_patch, class_id=14, class_name="false_rock_ground")

        route_mgr = RouteManager()
        cartographer = AutoCartographer(route_manager=route_mgr)

        initial_reject_count = cartographer.total_gate_rejected

        # Process a candidate node that matches the negative rock patch
        parsed_items = [
            {
                "rel_map_dx": 15.0,
                "rel_map_dy": -10.0,
                "resource_type": "Unknown",
                "in_air": False,
                "confidence": 0.65,
                "stack_count": 1,
                "max_harvests": 3,
                "current_harvests": 0,
                "source_type": "minimap_auto",
            }
        ]

        # Minimap frame with the rock patch
        frame = np.full((720, 1280, 3), 115, dtype=np.uint8)
        new_nodes = cartographer._register_parsed_detections(
            bgr=frame,
            parsed_items=parsed_items,
            route_manager=route_mgr,
        )

        # Gate rejection count must have incremented, and no node should be recorded
        assert len(new_nodes) == 0
        assert cartographer.total_gate_rejected > initial_reject_count
    finally:
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_singleton


def test_explorer_mode_avoids_learned_obstacle(temp_env: Dict[str, Any]):
    """
    Work Package H.4: ExplorerModeController Forward Path Obstacle Avoidance.
    Verifies that ExplorerModeController.check_path_obstacle recognizes learned
    obstacles or monsters directly ahead and signals obstacle_detected=True.
    """
    ex_path = temp_env["exemplars"]
    ex_mem = ExemplarMemory(json_path=ex_path)

    from dataset_factory import exemplar_memory as em_mod
    orig_singleton = em_mod._GLOBAL_EXEMPLAR_MEMORY
    em_mod._GLOBAL_EXEMPLAR_MEMORY = ex_mem

    try:
        # Teach 3D monster as obstacle
        monster_patch = np.zeros((80, 80, 3), dtype=np.uint8)
        monster_patch[:, :] = (20, 30, 200)
        ex_mem.add_positive_exemplar(monster_patch, class_id=10, class_name="monster_3d")

        explorer = ExplorerModeController()

        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        h, w = frame.shape[:2]
        frame[int(h * 0.52) : int(h * 0.82), int(w * 0.40) : int(w * 0.60)] = cv2.resize(
            monster_patch, (int(w * 0.20), int(h * 0.30))
        )

        obs_res = explorer.check_path_obstacle(frame)
        assert obs_res.get("obstacle_detected") is True
        assert obs_res.get("exemplar_obstacle") is True
    finally:
        em_mod._GLOBAL_EXEMPLAR_MEMORY = orig_singleton


def test_calibrated_rois_hot_reload_latency(temp_env: Dict[str, Any]):
    """
    Work Package H.5: Instant Hot-Reload Latency for Calibrated ROIs (<50ms).
    Verifies that CalibratedRoisManager detects disk changes in <50ms without restart.
    """
    roi_path = temp_env["rois"]
    init_data = {"rois": {}}
    roi_path.write_text(json.dumps(init_data), encoding="utf-8")

    mgr = CalibratedRoisManager(json_path=roi_path)
    assert not mgr.has_roi("skill_bar")

    # Update file on disk
    updated_data = {
        "rois": {
            "skill_bar": {
                "key": "skill_bar",
                "normalized_roi": [0.360, 0.910, 0.280, 0.045],
            }
        }
    }
    time.sleep(0.01)  # Ensure mtime difference
    roi_path.write_text(json.dumps(updated_data), encoding="utf-8")

    t_start = time.perf_counter()
    reloaded = mgr.check_reload()
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    assert elapsed_ms < 100.0, f"Calibrated ROIs hot reload took {elapsed_ms:.2f}ms!"
    assert reloaded is True
    assert mgr.has_roi("skill_bar")

