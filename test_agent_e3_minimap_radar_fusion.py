# -*- coding: utf-8 -*-
"""
Agent E3 Verification Suite:
Minimap Sprout/Monster Recognition, 0.075 Player Clearance & Live Radar Fusion.
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from core.minimap_scanner import MinimapScanner
from ui.cartography_radar import (
    _GATHER_PLAYER_CLEARANCE,
    RadarCanvas,
    classify_minimap_crop_bgr,
    minimap_material_is_recordable,
    minimap_norm_is_hud_chrome,
    resolve_canonical_region,
)


class TestMinimapSproutAndMonsterSeparation:
    """Requirement: Separate 3-leaf harvest sprouts from hostile red dots on minimap."""

    def test_class_constants_and_ids(self) -> None:
        """MinimapScanner defines correct class IDs for sprouts (9) and monster dots (11)."""
        scanner = MinimapScanner()
        assert scanner.CLASS_ID_SPROUT == 9
        assert scanner.CLASS_ID_MONSTER_DOT == 11
        assert scanner.CLASS_NAME_SPROUT == "minimap_sprout"
        assert scanner.CLASS_NAME_HOSTILE_DOT == "minimap_monster_negative"
        assert scanner.CLASS_NAME_MONSTER_DOT == "monster_dot"
        assert scanner.PLAYER_CLEARANCE_NORM == 0.075

    def test_sprout_vs_hostile_dot_discrimination(self) -> None:
        """
        Synthesize a minimap with both a genuine 3-leaf sprout and a hostile red dot.
        Scanner must separate them cleanly with mutually exclusive classifications.
        """
        scanner = MinimapScanner()
        h, w = 257, 332
        crop = np.full((h, w, 3), (45, 60, 72), dtype=np.uint8)

        # Center player
        cx = int(round(w * 0.502))
        cy = int(round(h * 0.545))
        cv2.circle(crop, (cx, cy), 4, (255, 255, 255), -1)

        # 1. Genuine 3-leaf sprout at (cx + 50, cy - 30) (dx=50, dy=-30, dist ~ 58px > 0.075*257)
        sprout_patch = cv2.cvtColor(MinimapScanner._CANONICAL_SPROUT_TPL, cv2.COLOR_GRAY2BGR)
        crop[cy - 38 : cy - 22, cx + 42 : cx + 58] = sprout_patch

        # 2. Hostile red dot at (cx - 60, cy + 20) (red BGR: B=20, G=20, R=240)
        cv2.circle(crop, (cx - 60, cy + 20), 5, (20, 20, 240), -1)

        # 3. White neutral monster dot at (cx - 40, cy - 40)
        cv2.circle(crop, (cx - 40, cy - 40), 4, (245, 245, 245), -1)

        results = scanner.scan_minimap_crop(crop)
        sprouts = results["gatherable_nodes"]
        hostiles = results["hostile_dots"]
        monsters = results["monster_dots"]

        # Verify sprout detection
        assert len(sprouts) >= 1
        sprout = sprouts[0]
        assert sprout["kind"] == "sprout_material"
        assert sprout["class_id"] == 9
        assert sprout["class_name"] == "minimap_sprout"
        assert sprout["is_gatherable"] is True
        assert sprout["is_hostile"] is False
        assert sprout["harvests_available"] == 3

        # Verify hostile red dot detection
        assert len(hostiles) >= 1
        hostile = hostiles[0]
        assert hostile["kind"] == "hostile_dot"
        assert hostile["class_id"] == 11
        assert hostile["class_name"] == "minimap_monster_negative"
        assert hostile["is_hostile"] is True
        assert hostile["is_gatherable"] is False

        # Verify mutual exclusivity: Sprout position does not appear in hostiles
        sprout_pos = sprout["pos"]
        assert all(
            math.hypot(sprout_pos[0] - h["pos"][0], sprout_pos[1] - h["pos"][1]) > 10
            for h in hostiles
        )

    def test_classify_minimap_targets_api(self) -> None:
        """classify_minimap_targets returns structured dictionary with separated classes."""
        scanner = MinimapScanner()
        h, w = 257, 332
        crop = np.full((h, w, 3), (40, 55, 68), dtype=np.uint8)

        cx = int(round(w * 0.502))
        cy = int(round(h * 0.545))

        # Paste sprout
        sprout_patch = cv2.cvtColor(MinimapScanner._CANONICAL_SPROUT_TPL, cv2.COLOR_GRAY2BGR)
        crop[cy - 38 : cy - 22, cx + 42 : cx + 58] = sprout_patch

        # Red dot
        cv2.circle(crop, (cx - 50, cy + 30), 4, (15, 15, 245), -1)

        targets = scanner.classify_minimap_targets(crop)
        assert "sprouts" in targets
        assert "hostile_dots" in targets
        assert "monster_dots" in targets
        assert "all_targets" in targets
        assert targets["player_clearance_norm"] == 0.075
        assert len(targets["sprouts"]) >= 1
        assert len(targets["hostile_dots"]) >= 1
        assert all(s["class_id"] == 9 for s in targets["sprouts"])
        assert all(h["class_id"] == 11 and h["is_hostile"] for h in targets["hostile_dots"])


class TestPlayerClearanceNorm:
    """Requirement: Apply 0.075 player clearance."""

    def test_clearance_filtering_in_scanner(self) -> None:
        """
        Detections within 0.075 clearance radius of player center are dropped,
        while detections outside clearance are preserved.
        """
        scanner = MinimapScanner()
        h, w = 257, 332
        crop = np.full((h, w, 3), (45, 60, 72), dtype=np.uint8)

        cx = int(round(w * 0.502))
        cy = int(round(h * 0.545))
        clearance_px = min(w, h) * 0.075  # ~19.275 px

        # 1. Hostile red dot INSIDE clearance (dist = 10px <= 19.275px)
        cv2.circle(crop, (cx + 8, cy + 6), 4, (20, 20, 240), -1)

        # 2. Hostile red dot OUTSIDE clearance (dist = 40px > 19.275px)
        cv2.circle(crop, (cx + 35, cy + 20), 4, (20, 20, 240), -1)

        # 3. Monster dot INSIDE clearance (dist = 12px <= 19.275px)
        cv2.circle(crop, (cx - 8, cy - 8), 4, (245, 245, 245), -1)

        # 4. Monster dot OUTSIDE clearance (dist = 45px > 19.275px)
        cv2.circle(crop, (cx - 35, cy - 30), 4, (245, 245, 245), -1)

        results = scanner.scan_minimap_crop(crop)
        hostiles = results["hostile_dots"]
        monsters = results["monster_dots"]

        # Only the outside hostile dot should survive
        assert all(h["dist_px"] > clearance_px for h in hostiles)

        # Only the outside monster dot should survive
        assert all(m["dist_px"] > clearance_px for m in monsters)

    def test_radar_clearance_constants(self) -> None:
        """_GATHER_PLAYER_CLEARANCE constant in cartography_radar is exactly 0.075."""
        assert _GATHER_PLAYER_CLEARANCE == 0.075

    def test_classify_minimap_crop_bgr_clearance(self) -> None:
        """classify_minimap_crop_bgr applies 0.075 player clearance to both sprouts and monsters."""
        h, w = 257, 332
        crop = np.full((h, w, 3), (45, 60, 72), dtype=np.uint8)
        cx = int(round(w * 0.502))
        cy = int(round(h * 0.545))

        # Red dot inside clearance
        cv2.circle(crop, (cx + 5, cy + 5), 4, (20, 20, 240), -1)
        # Red dot outside clearance
        cv2.circle(crop, (cx + 50, cy + 20), 4, (20, 20, 240), -1)

        materials, monsters = classify_minimap_crop_bgr(crop)
        # Ensure inside red dot was filtered by clearance
        gather_clearance_px = 0.075 * min(w, h)
        for mob in monsters:
            mx = mob["norm_x"] * w
            my = mob["norm_y"] * h
            dist = math.hypot(mx - cx, my - cy)
            assert dist > gather_clearance_px, f"Monster at dist {dist} inside clearance {gather_clearance_px}"


class TestSubregionRecognition:
    """Requirement: Auto-identify region/subregion (Altgard / Dredgion Crash Site)."""

    def test_subregion_canonical_aliases(self) -> None:
        """Altgard subregions map to canonical 'Altgard' region."""
        assert resolve_canonical_region("Dredgion Crash Site") == "Altgard"
        assert resolve_canonical_region("Safe Haven") == "Altgard"
        assert resolve_canonical_region("Calderon Canyon") == "Altgard"
        assert resolve_canonical_region("Silent Hill") == "Altgard"
        assert resolve_canonical_region("Altgard Fortress") == "Altgard"
        assert resolve_canonical_region("Moslan Forest") == "Altgard"
        assert resolve_canonical_region("Altgard") == "Altgard"

    def test_identify_subregion_from_minimap_title(self) -> None:
        """identify_subregion correctly recognizes 'Dredgion Crash Site' and 'Safe Haven' from title text."""
        scanner = MinimapScanner()
        h, w = 257, 332

        # 1. Test Dredgion Crash Site
        img_dredgion = np.full((h, w, 3), (40, 50, 60), dtype=np.uint8)
        cv2.putText(
            img_dredgion,
            "Dredgion Crash Site",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        res_d = scanner.identify_subregion(img_dredgion)
        assert res_d["subregion"] == "Dredgion Crash Site"
        assert res_d["canonical_region"] == "Altgard"
        assert res_d["confidence"] > 0.70

        # 2. Test Safe Haven
        img_haven = np.full((h, w, 3), (40, 50, 60), dtype=np.uint8)
        cv2.putText(
            img_haven,
            "Safe Haven",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        res_h = scanner.identify_subregion(img_haven)
        assert res_h["subregion"] == "Safe Haven"
        assert res_h["canonical_region"] == "Altgard"

    def test_identify_subregion_spatial_lookup(self) -> None:
        """Player map coordinates within Dredgion Crash Site bounding box identify the subregion."""
        scanner = MinimapScanner()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Coordinates inside Dredgion Crash Site bbox (center is around 655, 455)
        res = scanner.identify_subregion(img, player_map_pos=(650.0, 450.0))
        assert res["canonical_region"] == "Altgard"
        assert "Dredgion" in res["subregion"]
        assert res["source"] == "spatial_bbox"


class Test3DCamera70HFOVPerspectiveProjection:
    """Requirement: Project 3D camera boxes onto radar canvas via 70° HFOV perspective."""

    def test_camera_hfov_constant(self) -> None:
        """RadarCanvas initializes with 70° horizontal FOV."""
        canvas = RadarCanvas()
        assert canvas._camera_hfov_deg == 70.0

    def test_pinhole_perspective_projection_trigonometry(self) -> None:
        """
        Verify exact pinhole trigonometric perspective projection:
        focal_px = (frame_w / 2) / tan(35°)
        angle = atan2(dx, focal_px)
        """
        canvas = RadarCanvas()
        canvas.set_player_position(500.0, 500.0, angle_deg=0.0)  # North

        fw, fh = 1280, 720
        cx = fw * 0.5  # 640
        focal_px = cx / math.tan(math.radians(35.0))  # ~914.015

        # Test object at screen quarter (sx = 960 -> dx = +320)
        expected_angle = math.degrees(math.atan2(320.0, focal_px))  # ~19.29°
        boxes = [
            {"bbox": [920, 360, 1000, 440], "class_name": "herb_aria", "confidence": 0.90}
        ]
        projections = canvas.project_3d_yolo_boxes(boxes, frame_w=fw, frame_h=fh, heading_deg=0.0)
        assert len(projections) == 1
        p = projections[0]
        assert abs(p["bearing_deg"] - expected_angle) < 0.2
        assert p["camera_hfov_deg"] == 70.0

        # Test object at screen edge (sx = 1280 -> dx = +640)
        # Perspective projection at the very edge must equal half HFOV = 35.0°
        boxes_edge = [
            {"bbox": [1260, 360, 1300, 440], "class_name": "ore_orichalcum", "confidence": 0.88}
        ]
        proj_edge = canvas.project_3d_yolo_boxes(boxes_edge, frame_w=fw, frame_h=fh, heading_deg=0.0)
        assert len(proj_edge) == 1
        assert abs(proj_edge[0]["bearing_deg"] - 35.0) < 0.2

    def test_close_range_player_clearance_in_3d_projection(self) -> None:
        """
        3D detections within close-range player clearance (0.075 * 45m = 3.375m)
        with low confidence or false obstacle/enemy are rejected.
        """
        canvas = RadarCanvas()
        canvas.set_player_position(500.0, 500.0, angle_deg=0.0)

        fw, fh = 1280, 720
        # Box right at feet center (very close distance < 3.375m) with low confidence
        boxes = [
            {"bbox": [620, 680, 660, 715], "class_name": "obstacle", "confidence": 0.50},
            {"bbox": [620, 320, 660, 380], "class_name": "herb_mela", "confidence": 0.92},
        ]
        projections = canvas.project_3d_yolo_boxes(boxes, frame_w=fw, frame_h=fh, heading_deg=0.0)
        # The obstacle right on player's feet should be dropped
        assert len(projections) == 1
        assert projections[0]["class_name"] == "herb_mela"

    def test_live_radar_fusion_with_3d_and_minimap(self) -> None:
        """
        RadarCanvas updates live frame with both 3D YOLO projections and
        minimap sprout/monster detections simultaneously.
        """
        canvas = RadarCanvas()
        canvas.set_player_position(400.0, 400.0, angle_deg=90.0)  # Facing East

        # 3D boxes
        boxes = [
            {"bbox": [640, 360, 700, 420], "class_name": "gem_crystal", "confidence": 0.95}
        ]
        # Minimap detections
        mm_dets = [
            {
                "rel_map_dx": 20.0,
                "rel_map_dy": -15.0,
                "kind": "sprout_material",
                "class_id": 9,
                "is_gatherable": True,
                "resource_type": "Aria Filizi",
            },
            {
                "rel_map_dx": -25.0,
                "rel_map_dy": 30.0,
                "kind": "hostile_dot",
                "class_id": 11,
                "is_hostile": True,
                "is_gatherable": False,
                "resource_type": "Düşman Canavar",
            },
        ]

        proj_3d = canvas.project_3d_yolo_boxes(boxes, frame_w=1280, frame_h=720, heading_deg=90.0)
        proj_mm = canvas.project_minimap_detections(mm_dets)

        assert len(proj_3d) == 1
        assert proj_3d[0]["category"] == "gem"
        assert abs(proj_3d[0]["bearing_deg"] - 90.0) < 3.0  # Center object faces East

        assert len(proj_mm) == 2
        sprout_det = [d for d in proj_mm if d["class_id"] == 9][0]
        hostile_det = [d for d in proj_mm if d["class_id"] == 11][0]

        assert sprout_det["category"] == "sprout"
        assert sprout_det["color_hex"] == "#22d3ee"
        assert sprout_det["is_gatherable"] is True

        assert hostile_det["category"] == "enemy"
        assert hostile_det["color_hex"] == "#ef4444"
        assert hostile_det["is_hostile"] is True

    def test_subregion_hud_title_update(self) -> None:
        """RadarCanvas HUD overlay reflects the current subregion and canonical region."""
        canvas = RadarCanvas()
        canvas.set_current_subregion("Dredgion Crash Site", "Altgard")
        assert canvas.get_current_subregion() == "Dredgion Crash Site"
        assert canvas.get_current_canonical_region() == "Altgard"
