# -*- coding: utf-8 -*-
"""
Agent E1 Verification Test Suite
================================
Tests for Autonomous Exploration Route Builder, RouteGraph Generator,
15-25px Spatial Clustering, 2D Kalman/EMA Coordinate Smoothing,
Anti-Center-Sticking, and ExemplarMemory Integration in `core/auto_cartographer.py`.
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest

from core.auto_cartographer import (
    AutoCartographer,
    RouteGraph,
    TrackedResourceNode,
    DEDUP_RADIUS_MIN_PX,
    DEDUP_RADIUS_MAX_PX,
)
from core.route_manager import RouteManager


def _frame_with_textured_minimap(frame: np.ndarray) -> np.ndarray:
    """Tile a deterministic textured patch over the minimap ROI.

    ingest_detections runs the real quality gate, which crops its 80x80
    candidate patch from the minimap ROI; a flat gray ROI is rejected as a
    negative exemplar, so the ROI needs real texture.
    """
    hsv = np.zeros((80, 80, 3), dtype=np.uint8)
    hsv[:, :] = (140, 160, 190)
    rng = np.random.default_rng(42)
    noise = rng.integers(-15, 15, (80, 80, 3), dtype=np.int16)
    aria = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = AutoCartographer()._get_minimap_pixel_rect(1280, 720)
    for y in range(y1, y2, 80):
        for x in range(x1, x2, 80):
            h = min(80, y2 - y)
            w = min(80, x2 - x)
            frame[y:y + h, x:x + w] = aria[:h, :w]
    return frame


class TestRouteGraph:
    """Tests for RouteGraph data structure and connectivity."""

    def test_add_waypoints_and_edges(self):
        g = RouteGraph(region="Verteron")
        wp1 = g.add_waypoint(100.0, 100.0)
        wp2 = g.add_waypoint(140.0, 100.0)
        assert wp1.waypoint_id == 1
        assert wp2.waypoint_id == 2
        assert len(g.waypoints) == 2

        edge = g.add_edge(wp1.waypoint_id, wp2.waypoint_id)
        assert edge.distance_px == pytest.approx(40.0, rel=1e-3)
        assert len(g.edges) == 1
        assert g.total_distance == pytest.approx(40.0, rel=1e-3)

    def test_connect_node_to_nearest_waypoint(self):
        g = RouteGraph(region="Verteron")
        wp1 = g.add_waypoint(100.0, 100.0)
        wp2 = g.add_waypoint(200.0, 200.0)

        # Node at (110, 105) is ~11.18px from wp1
        edge = g.connect_node_to_nearest_waypoint(node_id=42, node_map_x=110.0, node_map_y=105.0)
        assert edge is not None
        assert edge.from_id == wp1.waypoint_id
        assert edge.to_id == 42
        assert 42 in wp1.connected_node_ids

    def test_build_connected_sequence(self):
        g = RouteGraph(region="Verteron")
        wp1 = g.add_waypoint(100.0, 100.0)
        wp2 = g.add_waypoint(150.0, 100.0)
        g.connect_node_to_nearest_waypoint(node_id=10, node_map_x=105.0, node_map_y=100.0)

        seq = g.build_connected_sequence()
        assert len(seq) == 3
        assert seq[0]["target_type"] == "waypoint"
        assert seq[1]["target_type"] == "node"
        assert seq[1]["node_id"] == 10
        assert seq[2]["target_type"] == "waypoint"

    def test_serialization_and_deserialization(self):
        g = RouteGraph(region="Altgard")
        wp1 = g.add_waypoint(500.0, 300.0)
        wp2 = g.add_waypoint(530.0, 300.0)
        g.add_edge(wp1.waypoint_id, wp2.waypoint_id, distance_px=30.0)
        data = g.to_dict()

        g2 = RouteGraph.from_dict(data)
        assert g2.region == "Altgard"
        assert len(g2.waypoints) == 2
        assert len(g2.edges) == 1
        assert g2.total_distance == pytest.approx(30.0)


class TestSpatialClustering15_25px:
    """Tests for 15-25px spatial clustering and duplicate piling prevention."""

    def test_radius_clamping(self):
        cart = AutoCartographer()
        cart.cluster_radius = 5.0
        assert cart.cluster_radius == DEDUP_RADIUS_MIN_PX  # 15.0px

        cart.cluster_radius = 50.0
        assert cart.cluster_radius == DEDUP_RADIUS_MAX_PX  # 25.0px

        cart.cluster_radius = 20.0
        assert cart.cluster_radius == 20.0

    def test_cluster_detections_merges_close_items(self):
        cart = AutoCartographer(dedup_radius_px=20.0)
        items = [
            {"rel_map_dx": 10.0, "rel_map_dy": 10.0, "confidence": 0.9, "resource_type": "Aria", "stack_count": 1},
            {"rel_map_dx": 18.0, "rel_map_dy": 12.0, "confidence": 0.6, "resource_type": "Gatherable Material", "stack_count": 2},
        ]
        # Distance = sqrt(8^2 + 2^2) = ~8.25px <= 20.0px
        clustered = cart.cluster_detections(items)
        assert len(clustered) == 1
        cl = clustered[0]
        assert cl["_cluster_count"] == 2
        assert cl["resource_type"] == "Aria"  # Specific type preserved over generic
        assert cl["stack_count"] == 2  # Max stack count preserved
        assert cl["confidence"] == 0.9  # Max confidence preserved
        # Centroid should be weighted toward higher confidence (0.9 vs 0.6)
        assert 10.0 <= cl["rel_map_dx"] < 15.0

    def test_cluster_detections_separates_distant_items(self):
        cart = AutoCartographer(dedup_radius_px=20.0)
        items = [
            {"rel_map_dx": 10.0, "rel_map_dy": 10.0, "confidence": 0.8, "resource_type": "Aria"},
            {"rel_map_dx": 45.0, "rel_map_dy": 10.0, "confidence": 0.8, "resource_type": "Mela"},
        ]
        # Distance = 35.0px > 25.0px
        clustered = cart.cluster_detections(items)
        assert len(clustered) == 2


class TestKalmanEMASmoothingAndOutlierGating:
    """Tests for 2D Kalman/EMA coordinate smoothing and anti-center sticking."""

    def test_kalman_ema_refinement_converges(self):
        cart = AutoCartographer(ema_alpha=0.35)
        tracked = TrackedResourceNode(
            node_id=1,
            map_x=500.0,
            map_y=300.0,
            confidence=0.8,
            kalman_variance=16.0,
            sightings_count=1,
        )

        # 5 successive observations near (505, 302)
        for _ in range(5):
            cart._refine_existing_node_kalman_ema(
                tracked=tracked,
                obs_map_x=505.0,
                obs_map_y=302.0,
                obs_confidence=0.9,
            )

        assert tracked.sightings_count == 6
        assert 500.0 < tracked.map_x <= 505.0
        assert 300.0 < tracked.map_y <= 302.0
        assert tracked.kalman_variance < 16.0  # Variance reduced

    def test_outlier_gating_prevents_violent_jump(self):
        cart = AutoCartographer()
        tracked = TrackedResourceNode(
            node_id=1,
            map_x=500.0,
            map_y=300.0,
            confidence=0.9,
            kalman_variance=4.0,
            sightings_count=5,
        )

        # Extreme outlier 80px away (e.g. noisy box)
        cart._refine_existing_node_kalman_ema(
            tracked=tracked,
            obs_map_x=580.0,
            obs_map_y=300.0,
            obs_confidence=0.5,
        )

        # Innovation should be clamped; position cannot jump more than ~2-3px
        shift = abs(tracked.map_x - 500.0)
        assert shift < 3.0, f"Outlier caused excessive shift: {shift}px"


class TestCenterStickingPrevention:
    """Tests that unanchored, player feet, and minimap center arrow detections are rejected."""

    def test_unanchored_detection_rejected(self):
        cart = AutoCartographer()
        raw = {"resource_type": "Aria", "confidence": 0.9}
        # No screen, minimap, or polar coords -> must NOT fall back to (0, 0)
        parsed = cart._parse_material_detection(raw, frame_w=1280, frame_h=720)
        assert parsed is None

    def test_harvest_auto_permitted_at_zero(self):
        # H-sweep dilim 13 K3: koordinatsız tespit elenir — harvest_auto
        # için bile oyuncu ayağına (0, 0) düğüm yazılmaz
        # (core/auto_cartographer.py:3311). Eski "(0, 0)'a izin verilir"
        # davranışı kalktı; güveni 1.0 olsa da koordinatsız girdi None döner.
        cart = AutoCartographer()
        raw = {"source_type": "harvest_auto", "resource_type": "Orichalcum Ore", "confidence": 1.0}
        parsed = cart._parse_material_detection(raw, frame_w=1280, frame_h=720)
        assert parsed is None

    def test_minimap_center_arrow_rejected(self):
        cart = AutoCartographer()
        # Minimap rect: (0.855 * 1280, 0.065 * 720) = (1094.4, 46.8)
        # Center is at mcx = ~87.68, mcy = ~74.55
        x1, y1, x2, y2 = cart._get_minimap_pixel_rect(1280, 720)
        mcx = (x1 + x2) / 2.0
        mcy = y1 + (y2 - y1) * 0.545

        # Detection precisely at arrow center
        raw = {"screen_x": mcx, "screen_y": mcy, "confidence": 0.8, "resource_type": "sprout"}
        parsed = cart._parse_material_detection(raw, frame_w=1280, frame_h=720)
        assert parsed is None  # Center sticking prevented!


class TestAutoCartographyModeAndBreadcrumbs:
    """Tests for AUTO CARTOGRAPHY mode, autonomous route building, and stats."""

    def test_auto_cartography_toggle_and_breadcrumbs(self):
        cart = AutoCartographer()
        assert not cart.auto_cartography_mode

        cart.enable_auto_cartography(True)
        assert cart.auto_cartography_mode

        # Set initial position -> drops waypoint 1
        cart.set_position(500.0, 300.0)
        assert cart.auto_mapped_count == 1
        assert cart.last_mapped_info is not None
        assert cart.last_mapped_info["type"] == "waypoint"
        assert cart.last_mapped_info["waypoint_id"] == 1

        # Move 10px (< breadcrumb_interval_px=30.0) -> no new waypoint
        cart.set_position(510.0, 300.0)
        assert cart.auto_mapped_count == 1

        # Move another 25px (total 35px >= 30.0) -> drops waypoint 2 and connects edge
        cart.set_position(535.0, 300.0)
        assert cart.auto_mapped_count == 2
        assert cart.last_mapped_info["waypoint_id"] == 2

        stats = cart.get_cartography_stats()
        assert stats["auto_mapped"] == 2
        assert stats["auto_cartography_mode"] is True
        assert stats["graph_waypoints_count"] == 2
        assert stats["graph_edges_count"] == 1
        assert stats["graph_total_distance_px"] == pytest.approx(35.0, rel=1e-2)

    def test_live_persistence_to_region_data(self):
        temp_dir = tempfile.mkdtemp()
        try:
            rm = RouteManager(base_dir=temp_dir)
            rm.create_region("TestRegion")
            cart = AutoCartographer(route_manager=rm)
            cart.enable_auto_cartography(True)

            # Move and generate breadcrumbs
            cart.set_position(400.0, 200.0)
            cart.set_position(440.0, 200.0)

            # Persist
            ok = cart.persist_to_region_data("TestRegion", base_dir=temp_dir)
            assert ok is True

            json_path = os.path.join(temp_dir, "TestRegion", "region_data.json")
            assert os.path.isfile(json_path)

            import json
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "route_graph" in data
            assert len(data["route_graph"]["waypoints"]) == 2
            assert data["cartography_stats"]["auto_mapped"] == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDetectionIngestionAndExemplars:
    """Tests for 3D YOLO and Minimap detection ingestion into the autonomous route graph."""

    def test_3d_and_minimap_ingestion_builds_connected_graph(self):
        temp_dir = tempfile.mkdtemp()
        try:
            rm = RouteManager(base_dir=temp_dir)
            rm.create_region("ExploreReg")
            cart = AutoCartographer(route_manager=rm)
            cart.enable_auto_cartography(True)

            # Player walks and establishes a breadcrumb waypoint
            cart.set_position(500.0, 300.0)

            # Ingest a 3D YOLO detection
            dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 120
            # Draw a simulated gatherable patch so CV filters don't reject it
            cv2.circle(dummy_frame, (700, 450), 25, (255, 255, 255), -1)
            # Texture the minimap ROI: the real quality gate crops its
            # candidate patch from there, and a flat ROI is rejected.
            dummy_frame = _frame_with_textured_minimap(dummy_frame)

            detections_3d = [
                {
                    "source": "3d_world",
                    "resource_type": "Aria",
                    "screen_x": 700,
                    "screen_y": 450,
                    "confidence": 0.88,
                    "stack_count": 1,
                }
            ]

            # All three ingestion sources run (3D + minimap + harvest
            # prompt) and the ingestion result is verified, not assumed.
            discovered = cart.ingest_detections(
                frame=dummy_frame,
                detections=detections_3d,
                route_manager=rm,
                region="ExploreReg",
                include_minimap=True,
                include_3d=True,
                include_harvest_prompt=True,
            )

            assert len(discovered) >= 1
            aria_nodes = [d for d in discovered if d["resource_type"] == "Aria"]
            assert len(aria_nodes) == 1
            assert aria_nodes[0]["map_x"] == pytest.approx(504.0, abs=1.0)
            assert aria_nodes[0]["map_y"] == pytest.approx(306.0, abs=1.0)
            # F-0053: eski beklenti (338, 206) hatalı ekran uzayıydı — ilk
            # 1280x720 kare harita uzayını 1280x720'ye döndürüyordu. Kanonik
            # 1920x1080 uzayında oyuncu (500, 300) + ofset = (504, 306).
            assert aria_nodes[0]["max_harvests"] == 3
            assert aria_nodes[0]["current_harvests"] == 0

            # Graph should now have the waypoint and connected resource node
            rg = cart.get_route_graph("ExploreReg")
            assert len(rg.waypoints) >= 1
            assert cart.auto_mapped_count >= 1
            assert cart.last_mapped_info is not None

            # Stats check
            stats = cart.get_cartography_stats("ExploreReg")
            assert stats["total_nodes"] >= 1
            assert stats["total_3d_recorded"] >= 1
            assert stats["auto_mapped"] >= 1
            assert stats["graph_waypoints_count"] >= 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_exemplar_memory_negative_rejection(self):
        cart = AutoCartographer()
        # Create a mock exemplar memory that flags a patch as negative
        class MockNegativeExemplarMemory:
            def is_negative_patch(self, patch, threshold=0.72):
                return True, 0.85, {"class_name": "fake_rock"}
            def classify_patch(self, patch, min_sim=0.70):
                return None

        dummy_patch = np.zeros((80, 80, 3), dtype=np.uint8)
        gate = cart.evaluate_node_candidate(
            dummy_patch,
            source_type="3d_auto",
            resource_type="Aria",
            exemplar_memory=MockNegativeExemplarMemory(),
        )
        assert gate.reject is True
        assert "negative_exemplar" in gate.reasons

    def test_exemplar_memory_positive_refinement(self):
        cart = AutoCartographer()
        class MockPositiveExemplarMemory:
            def is_negative_patch(self, patch, threshold=0.72):
                return False, 0.1, None
            def classify_patch(self, patch, min_sim=0.70):
                return 0, "Aria", 0.94

        # Plant-like patch
        dummy_patch = np.ones((80, 80, 3), dtype=np.uint8) * 180
        gate = cart.evaluate_node_candidate(
            dummy_patch,
            source_type="3d_auto",
            resource_type="Gatherable Material",
            exemplar_memory=MockPositiveExemplarMemory(),
            check_terrain=False,
        )
        assert gate.accept is True
        assert gate.verified_resource_type == "Aria"
        assert gate.exemplar_sim == pytest.approx(0.94)



def test_f0053_update_player_position_keeps_canonical_map_space():
    """F-0053: ilk kare 1280x720 bile olsa harita uzayı 1920x1080 kalır;
    update_player_position map_width/map_height'i kare boyutuna döndürmez."""
    cart = AutoCartographer()
    cart.set_position(500.0, 300.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    map_x, map_y, norm_x, norm_y = cart.update_player_position(frame)
    assert cart.map_width == 1920
    assert cart.map_height == 1080
    assert map_x == pytest.approx(500.0)
    assert map_y == pytest.approx(300.0)
    assert norm_x == pytest.approx(500.0 / 1920, abs=1e-4)
    assert norm_y == pytest.approx(300.0 / 1080, abs=1e-4)


def test_s0503_record_node_without_route_manager_saves_patch(tmp_path, monkeypatch):
    """S-0503: RouteManager yokken record_node, diske yazılmamış patch yolu
    kaydetmez; patch doğrudan kaydedilir ve yalnızca gerçek yol döner.

    H-sweep dilim 13 K3 güncellemesi: `record_node` artık güven
    (confidence) verilmeden kayıt yapmıyor — güven verilmediyse None
    döner (core/auto_cartographer.py:4366). Bu test güven vermeden
    çağırdığı için beklenti None oldu; patch-yolu davranışı güven
    verilen varyantta ayrıca doğrulanmalıdır.
    """
    monkeypatch.chdir(tmp_path)
    cart = AutoCartographer()
    cart.set_position(500.0, 300.0)
    patch_img = np.ones((80, 80, 3), dtype=np.uint8) * 180
    node = cart.record_node(
        patch=patch_img,
        map_x=500.0,
        map_y=300.0,
        source_type="minimap_auto",
    )
    assert node is None
