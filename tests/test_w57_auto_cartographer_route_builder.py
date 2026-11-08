# -*- coding: utf-8 -*-
"""
W57 Unit Tests: AutoCartographer Autonomous Route Builder,
22px Minimap Sprout (9) Spatial Deduplication, and Monster Dot (11) Exclusion.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.auto_cartographer import (
    DEFAULT_DEDUP_RADIUS_PX,
    AutoCartographer,
)
from core.node_quality_gate import (
    HARD_WRONG_CLASS_IDS,
    SPROUT_CLASS_ID,
    GateVerdict,
    evaluate_node_candidate,
    is_hard_wrong_class,
)
from core.route_manager import RouteManager


def _make_game_frame_with_minimap(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a synthetic 1280x720 game frame with a valid non-occluded minimap ROI."""
    rng = np.random.default_rng(57)
    frame = np.full((height, width, 3), (68, 78, 72), dtype=np.uint8)
    noise = rng.integers(-22, 22, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 25, 220).astype(np.uint8)
    return frame


def _make_clean_sprout_patch_80() -> np.ndarray:
    """Create an 80x80 textured minimap patch with a 3-leaf sprout pattern that passes CV & terrain gates."""
    rng = np.random.default_rng(109)
    patch = np.full((80, 80, 3), (65, 95, 70), dtype=np.uint8)
    noise = rng.integers(-25, 25, (80, 80, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 20, 230).astype(np.uint8)
    # Draw a 3-leaf non-circular sprout shape in the center
    cv2.rectangle(patch, (34, 28), (46, 52), (25, 30, 25), -1)
    patch[39:42, 39:42] = (240, 245, 240)  # stem core
    patch[33:37, 39:42] = (245, 250, 245)  # top leaf
    patch[39:42, 34:38] = (245, 250, 245)  # left leaf
    patch[39:42, 43:47] = (245, 250, 245)  # right leaf
    return patch


def _draw_3_leaf_sprout_on_minimap(
    minimap_bgr: np.ndarray,
    top_left_x: int,
    top_left_y: int,
) -> Tuple[float, float]:
    """
    Draw a 3-leaf white sprout icon (11x11 bounding box, ~41 white pixels, fill_ratio ~0.34,
    dark outer contour min_v <= 85, contrast >= 115) onto a minimap ROI.
    Returns expected icon center (icx, icy) in minimap coordinates.
    """
    # Dark outer contour box (ensures min_v <= 85 and high contrast)
    cv2.rectangle(
        minimap_bgr,
        (top_left_x - 2, top_left_y - 2),
        (top_left_x + 12, top_left_y + 12),
        (30, 35, 30),
        -1,
    )
    pattern = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    ys, xs = np.where(pattern == 1)
    for dy, dx in zip(ys, xs):
        minimap_bgr[top_left_y + int(dy), top_left_x + int(dx)] = (245, 245, 245)
    return float(top_left_x) + float( np.mean(xs)), float(top_left_y) + float(np.mean(ys))


def _draw_circular_monster_dot_on_minimap(
    minimap_bgr: np.ndarray,
    cx: int,
    cy: int,
    radius: int = 3,
    hostile_red: bool = False,
) -> None:
    """Draw a compact circular monster dot (class 11: white '•' or hostile red dot) on the minimap."""
    cv2.circle(minimap_bgr, (cx, cy), radius + 2, (28, 30, 28), -1)
    color = (20, 20, 245) if hostile_red else (245, 245, 245)
    cv2.circle(minimap_bgr, (cx, cy), radius, color, -1)


def test_minimap_sprout_class_9_recorded_and_connected_to_autonomous_route_graph(tmp_path: Path) -> None:
    """
    1. Verify that AutoCartographer in autonomous exploration mode accepts minimap sprout
    detections (class 9 / SPROUT_CLASS_ID), uses a 22.0px default deduplication radius,
    records the sprout into RouteManager, and links it to the active RouteGraph waypoint.
    """
    assert SPROUT_CLASS_ID == 9
    assert not is_hard_wrong_class(SPROUT_CLASS_ID)

    rm = RouteManager(base_dir=str(tmp_path), active_region="Verteron")
    rm.create_region("Verteron")
    cart = AutoCartographer(
        route_manager=rm,
        dedup_radius_px=DEFAULT_DEDUP_RADIUS_PX,
        initial_map_x=500.0,
        initial_map_y=400.0,
    )
    assert cart.dedup_radius_px == pytest.approx(22.0)

    cart.enable_auto_cartography(True)
    cart.set_position(500.0, 400.0)

    patch_80 = _make_clean_sprout_patch_80()
    # Evaluate with model verdict returning class 9 (minimap sprout)
    verdict = evaluate_node_candidate(
        patch_80,
        source_type="minimap_auto",
        resource_type="Gatherable Material",
        model_verdict=(9, 0.91),
        check_exemplars=False,
        check_terrain=False,
    )
    assert verdict.accept is True
    assert verdict.raw_class_id == 9

    frame = _make_game_frame_with_minimap()
    sprout_det = {
        "rel_map_dx": 18.0,
        "rel_map_dy": -12.0,
        "resource_type": "Gatherable Material",
        "confidence": 0.91,
        "stack_count": 1,
        "source_type": "minimap_auto",
        "class_id": 9,
    }

    with patch.object(cart, "extract_clean_minimap_patch", return_value=patch_80), patch(
        "core.auto_cartographer.evaluate_node_candidate", return_value=verdict
    ):
        discovered = cart.process_frame_discoveries(
            frame,
            detected_materials=[sprout_det],
            route_manager=rm,
            region="Verteron",
        )

    assert len(discovered) == 1
    node = discovered[0]
    assert node["id"] == 1
    assert node["source_type"] == "minimap_auto"
    assert node["map_x"] == pytest.approx(518, abs=1)
    assert node["map_y"] == pytest.approx(388, abs=1)
    assert node["max_harvests"] == 3

    rg = cart.get_route_graph("Verteron")
    assert len(rg.waypoints) == 1
    wp = rg.waypoints[1]
    assert 1 in wp.connected_node_ids


def test_minimap_sprouts_deduplicated_within_22px_radius_across_roaming_frames(tmp_path: Path) -> None:
    """
    2. Verify that while roaming, minimap sprout (9) detections within the 22px radius
    (both in same-frame clusters and across sequential movement frames) are deduplicated
    into a single node with Kalman/EMA coordinate refinement, while a sprout > 22px away
    creates a second distinct node.
    """
    rm = RouteManager(base_dir=str(tmp_path), active_region="Verteron")
    rm.create_region("Verteron")
    cart = AutoCartographer(
        route_manager=rm,
        dedup_radius_px=22.0,
        initial_map_x=600.0,
        initial_map_y=500.0,
    )
    cart.enable_auto_cartography(True)
    cart.set_position(600.0, 500.0)

    patch_80 = _make_clean_sprout_patch_80()
    accept_gate = GateVerdict(accept=True, raw_class_id=9, confidence=0.90)
    frame = _make_game_frame_with_minimap()

    with patch.object(cart, "extract_clean_minimap_patch", return_value=patch_80), patch(
        "core.auto_cartographer.evaluate_node_candidate", return_value=accept_gate
    ):
        # Frame 1: Two simultaneous sprout (9) detections 12px apart -> clustered into 1 node near (620, 510)
        batch_1 = [
            {
                "exact_map_x": 620.0,
                "exact_map_y": 510.0,
                "resource_type": "Gatherable Material",
                "confidence": 0.90,
                "source_type": "minimap_auto",
                "class_id": 9,
            },
            {
                "exact_map_x": 628.0,
                "exact_map_y": 518.0,  # dist = hypot(8, 8) = 11.31px <= 22.0px
                "resource_type": "Gatherable Material",
                "confidence": 0.85,
                "source_type": "minimap_auto",
                "class_id": 9,
            },
        ]
        created_f1 = cart.process_frame_discoveries(frame, detected_materials=batch_1, region="Verteron")
        assert len(created_f1) == 1

        # Frame 2: Player moves to (615, 505) and sees the same sprout at (634, 519),
        # which is ~13.0px from the tracked node (<= 22.0px radius) -> must deduplicate!
        cart.set_position(615.0, 505.0)
        obs_within_22px = [
            {
                "exact_map_x": 634.0,
                "exact_map_y": 519.0,
                "resource_type": "Gatherable Material",
                "confidence": 0.92,
                "source_type": "minimap_auto",
                "class_id": 9,
            }
        ]
        created_f2 = cart.process_frame_discoveries(frame, detected_materials=obs_within_22px, region="Verteron")
        assert created_f2 == [], "Sprout within 22px radius must be deduplicated, not added as a new node"

        tracked_after_f2 = cart.get_tracked_nodes(region="Verteron")
        assert len(tracked_after_f2) == 1
        assert tracked_after_f2[0]["sightings_count"] == 2

        # Frame 3: Player walks further to (650, 505) and discovers a new sprout at (710, 560),
        # > 22.0px away in map coords and > 36px in relative offset -> must be recorded as node #2!
        cart.set_position(650.0, 505.0)
        obs_beyond_22px = [
            {
                "exact_map_x": 710.0,
                "exact_map_y": 560.0,
                "resource_type": "Gatherable Material",
                "confidence": 0.93,
                "source_type": "minimap_auto",
                "class_id": 9,
            }
        ]
        created_f3 = cart.process_frame_discoveries(frame, detected_materials=obs_beyond_22px, region="Verteron")
        assert len(created_f3) == 1
        assert created_f3[0]["id"] == 2

    rm_nodes = rm.get_nodes(region="Verteron")
    assert len(rm_nodes) == 2


def test_minimap_monster_dot_class_11_rejected_by_quality_gate_and_never_added_to_route(tmp_path: Path) -> None:
    """
    3. Verify that minimap monster dot detections (class 11 in HARD_WRONG_CLASS_IDS or
    circular monster dot patches) are rejected by the quality gate and never added to
    RouteManager, _tracked_nodes, or RouteGraph.
    """
    assert 11 in HARD_WRONG_CLASS_IDS
    assert is_hard_wrong_class(11) is True

    rm = RouteManager(base_dir=str(tmp_path), active_region="Altgard")
    rm.create_region("Altgard")
    cart = AutoCartographer(
        route_manager=rm,
        dedup_radius_px=22.0,
        initial_map_x=400.0,
        initial_map_y=300.0,
    )
    cart.enable_auto_cartography(True)
    cart.set_position(400.0, 300.0)

    patch_80 = _make_clean_sprout_patch_80()
    # Evaluate candidate when detector classifies it as class 11 (Minimap Monster)
    monster_verdict = evaluate_node_candidate(
        patch_80,
        source_type="minimap_auto",
        resource_type="Gatherable Material",
        model_verdict=(11, 0.88),
        check_exemplars=False,
        check_terrain=False,
    )
    assert monster_verdict.reject is True
    assert monster_verdict.class_id == 11
    assert "model" in monster_verdict.reasons

    frame = _make_game_frame_with_minimap()
    monster_det = {
        "rel_map_dx": 25.0,
        "rel_map_dy": 15.0,
        "resource_type": "Gatherable Material",
        "confidence": 0.88,
        "source_type": "minimap_auto",
        "class_id": 11,
    }

    with patch.object(cart, "extract_clean_minimap_patch", return_value=patch_80), patch(
        "core.auto_cartographer.evaluate_node_candidate", return_value=monster_verdict
    ):
        discovered = cart.process_frame_discoveries(
            frame,
            detected_materials=[monster_det],
            route_manager=rm,
            region="Altgard",
        )
        recorded_direct = cart.record_node(
            patch=patch_80,
            map_x=425.0,
            map_y=315.0,
            source_type="minimap_auto",
            route_manager=rm,
            region="Altgard",
        )

    assert discovered == []
    assert recorded_direct is None
    assert cart.total_gate_rejected >= 2
    assert cart.get_tracked_nodes(region="Altgard") == []
    assert rm.get_nodes(region="Altgard") == []
    assert cart.get_route_graph("Altgard").waypoints[1].connected_node_ids == []


def test_minimap_cv_detector_accepts_3_leaf_sprout_and_filters_out_circular_monster_dots(tmp_path: Path) -> None:
    """
    4. Verify that _detect_minimap_gatherables and record_from_minimap on a live minimap ROI
    detect only the 3-leaf white sprout icon (class 9 geometry) while ignoring circular
    white and red monster dots (class 11 geometry).
    """
    rm = RouteManager(base_dir=str(tmp_path), active_region="PilgrimsRidge")
    rm.create_region("PilgrimsRidge")
    cart = AutoCartographer(
        route_manager=rm,
        dedup_radius_px=22.0,
        initial_map_x=800.0,
        initial_map_y=600.0,
    )
    cart.enable_auto_cartography(True)
    cart.set_position(800.0, 600.0)

    frame = _make_game_frame_with_minimap(1280, 720)
    x1, y1, x2, y2 = cart._get_minimap_pixel_rect(1280, 720)
    minimap_roi = frame[y1:y2, x1:x2]

    # Draw 1 valid 3-leaf sprout at (42, 58) inside minimap ROI
    sprout_icx, sprout_icy = _draw_3_leaf_sprout_on_minimap(minimap_roi, top_left_x=42, top_left_y=58)
    # Draw circular white & red monster dots (class 11) at other valid minimap positions
    _draw_circular_monster_dot_on_minimap(minimap_roi, cx=125, cy=64, radius=3, hostile_red=False)
    _draw_circular_monster_dot_on_minimap(minimap_roi, cx=55, cy=98, radius=4, hostile_red=True)

    raw_detections = cart._detect_minimap_gatherables(frame)
    assert len(raw_detections) == 1, f"Expected only the 3-leaf sprout to be detected, got: {raw_detections}"
    det = raw_detections[0]
    assert det["source_type"] == "minimap_auto"
    assert det["resource_type"] == "Gatherable Material"
    assert math.hypot(det["minimap_x"] - sprout_icx, det["minimap_y"] - sprout_icy) <= 2.0

    patch_80 = _make_clean_sprout_patch_80()
    accept_gate = GateVerdict(accept=True, raw_class_id=9, confidence=0.89)
    with patch.object(cart, "extract_clean_minimap_patch", return_value=patch_80), patch(
        "core.auto_cartographer.evaluate_node_candidate", return_value=accept_gate
    ):
        recorded = cart.record_from_minimap(
            frame,
            route_manager=rm,
            region="PilgrimsRidge",
            update_odometry=False,
        )

    assert len(recorded) == 1
    assert len(rm.get_nodes(region="PilgrimsRidge")) == 1


def test_autonomous_roaming_mixed_sprout_9_and_monster_11_stream_builds_clean_deduplicated_route(
    tmp_path: Path,
) -> None:
    """
    5. End-to-end autonomous roaming test: as AutoCartographer walks a multi-waypoint path,
    it receives repeated minimap sprout (9) sightings within 22px, monster dots (11), and
    a second stacked sprout (9) further along the route. Verify that the final route graph
    and persisted region_data.json contain only the 2 deduplicated sprout nodes and 0 monsters.
    """
    rm = RouteManager(base_dir=str(tmp_path), active_region="Heiron")
    rm.create_region("Heiron")
    cart = AutoCartographer(
        route_manager=rm,
        dedup_radius_px=22.0,
        initial_map_x=300.0,
        initial_map_y=300.0,
    )
    cart.enable_auto_cartography(True)

    patch_80 = _make_clean_sprout_patch_80()
    frame = _make_game_frame_with_minimap()

    def _mock_gate(patch_img, **kwargs):
        res_type = str(kwargs.get("resource_type", ""))
        if "monster" in res_type.lower():
            return GateVerdict(accept=False, reasons=["model", "monster"], class_id=11, raw_class_id=11, confidence=0.92)
        return GateVerdict(accept=True, raw_class_id=9, confidence=0.91)

    with patch.object(cart, "extract_clean_minimap_patch", return_value=patch_80), patch(
        "core.auto_cartographer.evaluate_node_candidate", side_effect=_mock_gate
    ):
        # Step 1: Start at (300, 300) -> Waypoint #1 created; see Sprout #1 at (305, 310) and a Monster (11) at (340, 290)
        cart.set_position(300.0, 300.0)
        cart.process_frame_discoveries(
            frame,
            detected_materials=[
                {
                    "exact_map_x": 305.0,
                    "exact_map_y": 310.0,
                    "resource_type": "Gatherable Material",
                    "confidence": 0.88,
                    "source_type": "minimap_auto",
                    "class_id": 9,
                },
                {
                    "exact_map_x": 340.0,
                    "exact_map_y": 290.0,
                    "resource_type": "minimap_monster_11",
                    "confidence": 0.92,
                    "source_type": "minimap_auto",
                    "class_id": 11,
                },
            ],
            region="Heiron",
        )

        # Step 2: Walk 35px to (335, 300) -> Waypoint #2 created; see Sprout #1 again at (312, 312)
        # Distance from (305, 310) is hypot(7, 2) = 7.28px <= 22.0px -> deduplicated!
        cart.set_position(335.0, 300.0)
        cart.process_frame_discoveries(
            frame,
            detected_materials=[
                {
                    "exact_map_x": 312.0,
                    "exact_map_y": 312.0,
                    "resource_type": "Gatherable Material",
                    "confidence": 0.93,
                    "source_type": "minimap_auto",
                    "class_id": 9,
                }
            ],
            region="Heiron",
        )

        # Step 3: Walk 40px to (375, 300) -> Waypoint #3 created; see Sprout #2 (2x stack) at (420, 260)
        # Distance from Sprout #1 is > 100px (> 22.0px) -> recorded as Node #2!
        cart.set_position(375.0, 300.0)
        cart.process_frame_discoveries(
            frame,
            detected_materials=[
                {
                    "exact_map_x": 420.0,
                    "exact_map_y": 260.0,
                    "resource_type": "Gatherable Material",
                    "confidence": 0.94,
                    "stack_count": 2,
                    "source_type": "minimap_auto",
                    "class_id": 9,
                }
            ],
            region="Heiron",
        )

        cart.persist_to_region_data("Heiron", base_dir=str(tmp_path))

    tracked = cart.get_tracked_nodes(region="Heiron")
    assert len(tracked) == 2
    assert tracked[0]["id"] == 1
    assert tracked[0]["sightings_count"] == 2
    assert tracked[1]["id"] == 2
    assert tracked[1]["stack_count"] == 2
    assert tracked[1]["max_harvests"] == 6
    assert cart.total_gate_rejected == 1

    seq = cart.get_navigation_sequence("Heiron")
    waypoint_items = [item for item in seq if item["target_type"] == "waypoint"]
    node_items = [item for item in seq if item["target_type"] == "node"]
    assert len(waypoint_items) == 3
    assert [n["node_id"] for n in node_items] == [1, 2]

    region_json = os.path.join(str(tmp_path), "Heiron", "region_data.json")
    assert os.path.isfile(region_json)
    with open(region_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved["nodes"]) == 2
    assert saved["route_graph"]["waypoint_count"] == 3
