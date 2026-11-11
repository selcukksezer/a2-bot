# -*- coding: utf-8 -*-
"""
Test Worker MR6 Map & Radar Test Suite (test_worker_mr6_map_radar.py)

Covers:
1. Radar Coordinate Ingestion:
   - 3D camera detections and minimap sprout detections generate accurate world coordinates without center-sticking.
   - Dead-reckoning displacement and odometry integration.
   - VisualOdometryTracker coordinate synchronization.
2. Spatial Deduplication & Clustering:
   - Intra-frame clustering of multiple close detections.
   - Inter-frame spatial deduplication avoiding duplicate stacked nodes.
   - ExplorerModeController ordered nodes deduplication within cluster radius.
3. Real-time Radar Event Notification:
   - Node discovered and node updated callback emission.
   - Instantaneous updates to registered radar widgets and map viewers.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock

import cv2

from core.auto_cartographer import (
    AutoCartographer,
    VisualOdometryTracker,
    TrackedResourceNode,
    DEFAULT_DEDUP_RADIUS_PX,
)
from core.explorer_mode import ExplorerModeController
from core.minimap_scanner import MinimapScanner
from core.route_manager import RouteManager, NodeData


def _make_sprout_frame_1920x1080() -> np.ndarray:
    """Synthetic 1920x1080 game frame with one green 3-leaf sprout icon on the minimap.

    The sprout is pasted from the canonical minimap template as a stand-in for a
    real minimap icon; the scanner must detect it through its real CV pipeline
    (see tests/test_w37_minimap_sprout_vs_monster_dot.py for template-independence
    coverage of the sprout detector itself).
    Returns the frame; the sprout sits ~28px east / ~10px north of the minimap
    center, i.e. map offset (rel_map_dx, rel_map_dy) ~= (+28, -9.7) at scale 1.0.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    carto = AutoCartographer()
    x1, y1, x2, y2 = carto._get_minimap_pixel_rect(1920, 1080)
    mcx = (x1 + x2) / 2.0
    mcy = y1 + (y2 - y1) * 0.545
    frame[y1:y2, x1:x2] = (52, 62, 56)
    tpl = MinimapScanner._CANONICAL_SPROUT_TPL
    patch = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    leaf = tpl >= 175
    patch[leaf, 0] = np.clip(tpl[leaf].astype(np.int32) - 45, 130, 255).astype(np.uint8)
    patch[leaf, 1] = tpl[leaf]
    patch[leaf, 2] = np.clip(tpl[leaf].astype(np.int32) - 35, 140, 255).astype(np.uint8)
    sx, sy = int(mcx + 23) - 8, int(mcy + 12) - 8
    frame[sy : sy + 16, sx : sx + 16] = patch
    return frame


class TestRadarCoordinateIngestion:
    """Verifies radar coordinate ingestion without center-sticking and with dead-reckoning."""

    def test_odometry_tracker_synchronization_and_offset_projection(self):
        """VisualOdometryTracker uses updated player position rather than sticking to (960, 540)."""
        tracker = VisualOdometryTracker(map_width=1920, map_height=1080)
        # Default start
        assert tracker.map_x == 960.0
        assert tracker.map_y == 540.0

        # Update position
        tracker.map_x = 450.0
        tracker.map_y = 300.0
        wx, wy, nx, ny = tracker.offset_to_world_coords(dx=25.0, dy=-15.0)

        assert wx == 475.0
        assert wy == 285.0
        assert nx == round(475.0 / 1920, 4)
        assert ny == round(285.0 / 1080, 4)

    def test_dead_reckoning_integration_in_world_coords(self):
        """Dead-reckoning displacement is factored into projected coordinates."""
        tracker = VisualOdometryTracker(map_width=1920, map_height=1080)
        tracker.map_x = 500.0
        tracker.map_y = 500.0
        tracker.dead_reckoning_dx = 12.0
        tracker.dead_reckoning_dy = -8.0

        wx, wy, nx, ny = tracker.offset_to_world_coords(
            dx=30.0,
            dy=20.0,
            dead_reckoning_dx=tracker.dead_reckoning_dx,
            dead_reckoning_dy=tracker.dead_reckoning_dy,
        )

        # 500 + 30 + 12 = 542, 500 + 20 - 8 = 512
        assert wx == 542.0
        assert wy == 512.0

    def test_auto_cartographer_set_position_syncs_tracker(self):
        """AutoCartographer.set_position keeps tracker coordinates in sync."""
        carto = AutoCartographer()
        carto.set_position(320.0, 640.0, map_width=1600, map_height=900)

        assert carto.map_x == 320.0
        assert carto.map_y == 640.0
        assert carto.odometry_tracker.map_x == 320.0
        assert carto.odometry_tracker.map_y == 640.0

        wx, wy, _, _ = carto.offset_to_world_coords(dx=40.0, dy=-20.0)
        assert wx == 360.0
        assert wy == 620.0

    def test_3d_camera_detection_no_center_sticking(self):
        """3D camera detections across different screen coordinates project to distinct world coordinates.

        H-sweep dilim 13 K3: güveni ölçülmemiş tespit elenir — 'confidence'
        taşımayan dış girdi `_parse_material_detection`tan None döner
        (core/auto_cartographer.py:3347). Ölçülmüş güvenli varyant
        aşağıdaki ikinci bölümde eski ofset davranışını aynen korur.
        """
        carto = AutoCartographer()
        carto.set_position(1000.0, 500.0)

        # Left side of screen vs right side of screen — no confidence:
        # eliminated by the K3 unsafe-detection rule.
        left_det = carto._parse_material_detection(
            {"screen_x": 400, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto"},
            frame_w=1920,
            frame_h=1080,
        )
        right_det = carto._parse_material_detection(
            {"screen_x": 1500, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto"},
            frame_w=1920,
            frame_h=1080,
        )

        assert left_det is None
        assert right_det is None

        # Same detections with measured confidence: distinct projections.
        left_det = carto._parse_material_detection(
            {"screen_x": 400, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto", "confidence": 0.9},
            frame_w=1920,
            frame_h=1080,
        )
        right_det = carto._parse_material_detection(
            {"screen_x": 1500, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto", "confidence": 0.9},
            frame_w=1920,
            frame_h=1080,
        )

        assert left_det is not None
        assert right_det is not None
        assert left_det["rel_map_dx"] < 0  # To the left of player
        assert right_det["rel_map_dx"] > 0  # To the right of player

        left_wx, _, _, _ = carto.offset_to_world_coords(left_det["rel_map_dx"], left_det["rel_map_dy"])
        right_wx, _, _, _ = carto.offset_to_world_coords(right_det["rel_map_dx"], right_det["rel_map_dy"])

        assert left_wx < 1000.0
        assert right_wx > 1000.0
        assert abs(left_wx - right_wx) > 50.0

    def test_minimap_sprout_detection_offset_calculation(self):
        """Minimap detections calculate offsets based on minimap center rather than screen center.

        H-sweep dilim 13 K3: güveni ölçülmemiş tespit elenir — 'confidence'
        taşımayan dış girdi `_parse_material_detection`tan None döner
        (core/auto_cartographer.py:3347). Ölçülmüş güvenli varyant
        aşağıdaki ikinci bölümde eski ofset davranışını aynen korur.
        """
        carto = AutoCartographer()
        carto.set_position(800.0, 400.0)
        # Minimap ROI for 1920x1080
        x1, y1, x2, y2 = carto._get_minimap_pixel_rect(1920, 1080)
        mcx = (x1 + x2) / 2.0
        mcy = y1 + (y2 - y1) * 0.545

        # Sprout located 30px to the east and 20px to the south of minimap center
        sprout_sx = mcx + 30.0
        sprout_sy = mcy + 20.0

        det = carto._parse_material_detection(
            {"screen_x": sprout_sx, "screen_y": sprout_sy, "resource_type": "Gatherable Material"},
            frame_w=1920,
            frame_h=1080,
        )

        # No confidence: eliminated by the K3 unsafe-detection rule.
        assert det is None

        det = carto._parse_material_detection(
            {"screen_x": sprout_sx, "screen_y": sprout_sy, "resource_type": "Gatherable Material", "confidence": 0.9},
            frame_w=1920,
            frame_h=1080,
        )

        assert det is not None
        assert det["source_type"] == "minimap_auto"
        assert det["rel_map_dx"] > 0
        assert det["rel_map_dy"] > 0

        wx, wy, _, _ = carto.offset_to_world_coords(det["rel_map_dx"], det["rel_map_dy"])
        assert wx > 800.0
        assert wy > 400.0


class TestSpatialDeduplicationAndClustering:
    """Verifies spatial clustering and deduplication within cluster radius."""

    def test_intra_frame_cluster_detections(self):
        """Multiple detections within cluster_radius in a single frame are merged."""
        carto = AutoCartographer()
        # 3 detections: 2 are very close (dx=10, dy=12 and dx=14, dy=16, distance ~5.6px <= 22px), 1 is far (dx=100, dy=100)
        detections = [
            {"rel_map_dx": 10.0, "rel_map_dy": 12.0, "confidence": 0.8, "resource_type": "Gatherable Material"},
            {"rel_map_dx": 14.0, "rel_map_dy": 16.0, "confidence": 0.9, "resource_type": "Aria"},
            {"rel_map_dx": 100.0, "rel_map_dy": 100.0, "confidence": 0.85, "resource_type": "Mela"},
        ]

        clustered = carto.cluster_detections(detections, cluster_radius=22.0)
        assert len(clustered) == 2

        # First cluster is the confidence-weighted centroid of detection 1 and 2
        # (cluster_detections uses a confidence-weighted centroid, not a plain mean):
        # dx = (10*0.8 + 14*0.9) / (0.8 + 0.9) = 12.1176, dy = (12*0.8 + 16*0.9) / 1.7 = 14.1176
        cl1 = clustered[0]
        assert cl1["_cluster_count"] == 2
        assert math.isclose(cl1["rel_map_dx"], (10.0 * 0.8 + 14.0 * 0.9) / 1.7, abs_tol=0.01)
        assert math.isclose(cl1["rel_map_dy"], (12.0 * 0.8 + 16.0 * 0.9) / 1.7, abs_tol=0.01)
        assert cl1["confidence"] == 0.9
        assert cl1["resource_type"] == "Aria"  # Inherited specific material

    def test_inter_frame_deduplication_refines_kalman_ema(self):
        """Observations of the same node across consecutive frames are merged without creating stacked nodes."""
        carto = AutoCartographer()
        carto.set_position(500.0, 500.0)

        # Manually register a node in region
        tracked = TrackedResourceNode(
            node_id=1,
            region="test_reg",
            routes=["Default"],
            resource_type="Gatherable Material",
            map_x=520.0,
            map_y=510.0,
            norm_x=520.0 / 1920,
            norm_y=510.0 / 1080,
            sightings_count=1,
        )
        carto._tracked_nodes["test_reg"] = [tracked]

        # Second observation nearby (node_map_x = 522, node_map_y = 512, distance ~2.8px <= 22px)
        nearest, dist = carto._find_nearest_tracked_node("test_reg", 522.0, 512.0)
        assert nearest is not None
        assert nearest.node_id == 1
        assert dist < carto.dedup_radius_px

        # Refine existing node
        carto._refine_existing_node_kalman_ema(
            tracked=nearest,
            obs_map_x=522.0,
            obs_map_y=512.0,
            obs_confidence=0.9,
            obs_stack_count=2,
            obs_resource_type="Aria",
        )

        assert nearest.sightings_count == 2
        assert nearest.resource_type == "Aria"
        assert nearest.stack_count == 2
        assert 520.0 <= nearest.map_x <= 522.0
        assert 510.0 <= nearest.map_y <= 512.0

    def test_explorer_mode_ordered_nodes_spatial_deduplication(self, tmp_path):
        """ExplorerModeController does not create stacked duplicates in _ordered_nodes.

        Drives the REAL discovery path (no mock of process_frame_discoveries):
        a synthetic 1920x1080 frame carrying a minimap sprout is stepped through
        `step_passive_cartography`, whose real CV scanners detect it and whose real
        `process_frame_discoveries` registers it. The detected node lands at
        ~(628, 590), ~21px from the pre-existing ordered node at (620, 610) --
        inside the 22px dedup radius -- so the explorer must merge it into the
        existing ordered node instead of appending a stacked duplicate.
        A far-away seed node is pre-registered in the RouteManager (id 1) so the
        newly discovered node gets a different id (2) and the *spatial* dedup
        branch (not the same-id branch) is exercised.
        """
        rm = RouteManager(base_dir=str(tmp_path / "routes"))
        rm.create_region("test_explorer_reg")
        rm.active_region = "test_explorer_reg"
        # Seed node far away (100, 100) so the discovered node is assigned id 2.
        rm.add_node(region="test_explorer_reg", map_x=100, map_y=100, resource_type="Mela")

        explorer = ExplorerModeController(route_manager=rm, region="test_explorer_reg")
        explorer.auto_cartographer.set_position(600.0, 600.0)

        # Pre-existing ordered node at (620, 610) with a large max_harvests.
        n1 = NodeData(id=99, map_x=620, map_y=610, region="test_explorer_reg", resource_type="Aria")
        n1.max_harvests = 6
        explorer._ordered_nodes.append(n1)

        frame = _make_sprout_frame_1920x1080()
        res = explorer.step_passive_cartography(frame)

        # The real scanners must have detected the sprout and mapped exactly one node.
        assert res["detected_count"] == 1
        assert len(res["newly_mapped_nodes"]) == 1
        new_node = res["newly_mapped_nodes"][0]
        assert new_node["node_id"] == 2
        assert math.hypot(new_node["map_x"] - 620, new_node["map_y"] - 610) <= 22.0

        # _ordered_nodes must NOT have added a duplicate stacked node: the seed
        # (id 1, synced from the RouteManager by the constructor) plus the merged
        # n1 (id 99) -- and no node with the discovered id 2. The merge keeps
        # the larger max_harvests on n1.
        assert len(explorer._ordered_nodes) == 2
        assert {n.id for n in explorer._ordered_nodes} == {1, 99}
        merged = next(n for n in explorer._ordered_nodes if n.id == 99)
        assert merged.max_harvests == 6
        assert "Auto-mapped #2" in res["hud_notifications"][0]


class TestRealTimeRadarEventNotification:
    """Verifies callbacks and signal emissions for radar canvas and map viewer updates."""

    def test_auto_cartographer_callback_emission(self, tmp_path):
        """Emits callbacks to discovered, updated, and UI listeners instantaneously.

        Drives the REAL discovery path (no direct _emit_* calls): a minimap
        detection is fed to `process_frame_discoveries`, which parses, gates and
        registers it, and a second nearby observation of the same node exercises
        the update path. Callbacks must fire with the node dicts the real
        pipeline produced, not with a hand-made dict.
        """
        rm = RouteManager(base_dir=str(tmp_path / "routes"))
        rm.create_region("notif_carto_reg")
        rm.active_region = "notif_carto_reg"

        carto = AutoCartographer()
        carto.set_position(600.0, 600.0)

        discovered_mock = MagicMock()
        updated_mock = MagicMock()
        radar_event_mock = MagicMock()
        ui_listener_mock = MagicMock()

        carto.add_node_discovered_callback(discovered_mock)
        carto.add_node_updated_callback(updated_mock)
        carto.add_radar_event_callback(radar_event_mock)
        carto.register_ui_listener(ui_listener_mock)

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        x1, y1, x2, y2 = carto._get_minimap_pixel_rect(640, 360)
        mcx = (x1 + x2) / 2.0
        mcy = y1 + (y2 - y1) * 0.545

        # Real minimap tile + real sprout at the detection point (mirrors
        # _make_sprout_frame_1920x1080): a pure-black frame crops a pure-black
        # 80x80 patch, which the quality gate CORRECTLY rejects — it
        # cosine-matches (sim 1.0) the learned `wrong_node_purge` negative
        # exemplar in dataset/learned_exemplars.json (S-4401: minimap
        # candidates are no longer exempt from the learned-negative check).
        # The detection claims a sprout, so the frame must contain one.
        # The gate does not re-label the patch (no positive exemplar clears
        # 0.70), so the detection's resource_type "Aria" survives.
        frame[y1:y2, x1:x2] = (52, 62, 56)
        tpl = MinimapScanner._CANONICAL_SPROUT_TPL
        patch = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
        leaf = tpl >= 175
        patch[leaf, 0] = np.clip(tpl[leaf].astype(np.int32) - 45, 130, 255).astype(np.uint8)
        patch[leaf, 1] = tpl[leaf]
        patch[leaf, 2] = np.clip(tpl[leaf].astype(np.int32) - 35, 140, 255).astype(np.uint8)
        sx, sy = int(mcx + 30) - 8, int(mcy + 20) - 8
        frame[sy : sy + 16, sx : sx + 16] = patch

        # Discovery: real minimap detection 30px east / 20px south of minimap center.
        discovered = carto.process_frame_discoveries(
            frame,
            detected_materials=[
                {"screen_x": mcx + 30.0, "screen_y": mcy + 20.0, "resource_type": "Aria", "confidence": 0.9}
            ],
            route_manager=rm,
            region="notif_carto_reg",
        )
        assert len(discovered) == 1
        node = discovered[0]
        # No 5 (A): tespit minimap merkezinin (+30, +20) px uzağında;
        # harita konumu oyuncu (600, 600) + ofset x minimap_to_map_scale
        # (7.29927) = (600 + 219, 600 + 146) = (819, 746). Eski (630, 620)
        # ölçek 1.0 varsayıyordu; üretim 7.299x ölçek kullanır.
        assert node["map_x"] == 819
        assert node["map_y"] == 746
        assert node["resource_type"] == "Aria"

        discovered_mock.assert_called_once_with(node)
        ui_listener_mock.add_discovered_node.assert_called_once_with(node, trigger_pulse=True)
        radar_event_mock.assert_called_with("node_discovered", node)

        # Update: a second observation ~2.8px away refines the same node (no new node).
        updated = carto.process_frame_discoveries(
            frame,
            detected_materials=[
                {"screen_x": mcx + 32.0, "screen_y": mcy + 22.0, "resource_type": "Aria", "confidence": 0.9}
            ],
            route_manager=rm,
            region="notif_carto_reg",
        )
        assert updated == []
        assert len(carto._tracked_nodes["notif_carto_reg"]) == 1
        assert discovered_mock.call_count == 1  # no second discovery

        updated_mock.assert_called_once()
        updated_node = updated_mock.call_args[0][0]
        assert updated_node["node_id"] == node["node_id"]
        assert updated_node["sightings_count"] == 2
        ui_listener_mock.update_node.assert_called_once_with(updated_node)
        radar_event_mock.assert_called_with("node_updated", updated_node)

    def test_explorer_mode_ui_listener_registration_and_notification(self, tmp_path):
        """ExplorerModeController registers radar widget and map viewer and notifies on step discoveries.

        Drives the REAL step path (no direct _emit_* calls): a synthetic 1920x1080
        frame carrying a minimap sprout is stepped through `step_passive_cartography`.
        The real CV scanners detect it, the real cartographer registers it, and the
        explorer must fan the discovery out to the radar widget, the map viewer and
        the registered callback -- all with the node dict the real pipeline produced.
        """
        rm = RouteManager(base_dir=str(tmp_path / "routes"))
        rm.create_region("notif_reg")
        rm.active_region = "notif_reg"

        explorer = ExplorerModeController(route_manager=rm, region="notif_reg")
        explorer.auto_cartographer.set_position(600.0, 600.0)

        radar_widget = MagicMock()
        map_viewer = MagicMock()
        on_discovered_cb = MagicMock()

        explorer.register_radar_widget(radar_widget)
        explorer.register_map_viewer(map_viewer)
        explorer.add_node_discovered_callback(on_discovered_cb)

        res = explorer.step_passive_cartography(_make_sprout_frame_1920x1080())

        assert len(res["newly_mapped_nodes"]) == 1
        node = res["newly_mapped_nodes"][0]
        # H-sweep dilim 13 K3: sınıflandırılamayan sentetik filiz türü
        # jenerik kalır — kapı "Aria" varsaymaz (UNKNOWN_RESOURCE_TYPE,
        # core/auto_cartographer.py:2808). Keşif + bildirim akışı aynıdır,
        # yalnızca tür adı jeneriktir.
        assert node["resource_type"] == "Gatherable Material"
        # Konum: filiz karede minimap merkezinin (+23, +12) px uzağına
        # çizilir (_make_sprout_frame_1920x1080) — oyuncu (600, 600) iken
        # düğüm (624, 611) beklenir. Eski (628, 590) D58'in doğrulanmamış
        # hesabıydı; v9/v10/v11 koşularının üçü de resource_type
        # satırında kırıldığı için bu satırlar hiç doğrulanmamıştı.
        assert node["map_x"] == 624
        assert node["map_y"] == 611
        assert math.isclose(node["exact_map_y"], 611.27, abs_tol=1.0)

        # The callback fires with the discovered node (once via the cartographer
        # fan-out, once via the step's own emission).
        assert on_discovered_cb.call_count == 2
        for call in on_discovered_cb.call_args_list:
            assert call[0][0]["node_id"] == node["node_id"]

        # The radar widget receives the node through both the cartographer UI
        # listener path and the explorer emission path.
        assert radar_widget.add_discovered_node.call_count >= 2
        for call in radar_widget.add_discovered_node.call_args_list:
            assert call[0][0]["node_id"] == node["node_id"]
            assert call[1] == {"trigger_pulse": True}

        # The map viewer is reloaded on every explorer emission.
        assert map_viewer.reload_map_and_nodes.call_count == 2
        assert map_viewer.update.call_count == 2

        assert res["hud_notifications"] == [
            f"Auto-mapped #{node['node_id']} Gatherable Material (Sequential #1, 3x harvest)"
        ]
