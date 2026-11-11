# -*- coding: utf-8 -*-
"""D27 regression tests for core/questlog_importer.py fixes.

Covers: F-1710 (synthetic coords skipped), F-1715 (per-zone map, no Altgard
reuse), F-1712 (no round-robin resource fabrication), F-1711 (new-API nodes
in full-map x18/y18 space), F-1716 (failed import must not accept),
F-1726 (unknown zone returns []), F-1725 (stricter live-map detection +
live path works with fast_mode), F-1728 (canvas bounds), F-1724 (already
there counted once), F-1729 (failed patch write skips node), F-1730
(legacy id = max+1), F-1737 (no contrast normalization), F-1733
(progress denominator excludes skipped nodes), F-1735 (empty selection
warns instead of silent return).
"""
import math
import os

import cv2
import numpy as np
import pytest

from core.questlog_importer import (
    DREDGION_CRASH_SITE_24S_COORDS,
    QuestlogImporter,
    QuestlogImportDialog,
    _convert_24s_to_18s_coords,
)
from core.route_manager import NodeData


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture(scope="module")
def importer():
    # Module scope: QuestlogImporter is effectively stateless across tests.
    # Its only mutable state is _cached_ref_maps (additive instance cache;
    # get_reference_map_image returns a .copy(), so callers cannot poison
    # it) and self.db is module-level read-only data. Sharing one importer
    # avoids re-loading the 1280x720 reference map (and its clean-canvas
    # variant) from disk in each of the ~7 import tests below.
    return QuestlogImporter()


@pytest.fixture(scope="module")
def textured_map_frame():
    """One 720x1280 textured (low-saturation, high-variance) map frame.

    Shared by the live-map detection test and the live-capture test:
    _looks_like_game_map and _capture_or_load_map only read the frame
    (cv2.cvtColor returns a new array), so sharing cannot leak state
    between the two tests. Built once instead of twice.
    Returns (gray, bgr) pair; RandomState(0) texture passes detection.
    """
    rng = np.random.RandomState(0)
    gray = rng.randint(60, 140, (720, 1280), dtype=np.uint8)
    return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class FakeNewRouteManager:
    """Minimal new-API (_regions/NodeData) route manager fake."""

    def __init__(self, base_dir):
        self.base_dir = str(base_dir)
        self._regions = {}
        self.default_rest_sec = 300
        self.active_region = None
        self.active_route = None
        self.saved = []

    def create_region(self, zone):
        self._regions[zone] = {"nodes": [], "routes": []}
        return True

    def _next_node_id(self, region):
        nodes = self._regions[region]["nodes"]
        return (max(n.id for n in nodes) + 1) if nodes else 1

    def _get_region_patches_dir(self, region):
        return os.path.join(self.base_dir, "routes", region, "patches")

    def _get_patch_rel_path(self, region, node_id):
        return f"routes/{region}/patches/node_{node_id}.png"

    def save_region(self, region):
        self.saved.append(region)
        return True


class FakeLegacyRouteManager:
    """Minimal legacy data-dict route manager fake."""

    def __init__(self, base_dir):
        self.base_dir = str(base_dir)
        self.patch_dir = os.path.join(str(base_dir), "routes", "patches")
        self.data = {"regions": {}}
        self.saved = False

    def save_data(self):
        self.saved = True
        return True


def _seed_new_node(rm, zone, map_x, map_y, node_id=1):
    rm._regions.setdefault(zone, {"nodes": [], "routes": []})
    node = NodeData(id=node_id, region=zone, routes=["Dredgion Crash Site"],
                    map_x=map_x, map_y=map_y, resource_type="Orichalcum Ore")
    rm._regions[zone]["nodes"].append(node)
    return node


# --- F-1726 ---
def test_unknown_zone_returns_empty_not_altgard(importer):
    assert importer.get_resources("NoSuchZone") == []
    assert importer.get_sub_zones("NoSuchZone") == []
    assert importer.get_resources("Altgard")


def test_only_dredgion_has_verified_coords(importer):
    verified = [(z, sz["name"]) for z, zd in importer.db.items()
                for sz in zd["sub_zones"] if sz.get("verified_coords")]
    assert verified == [("Altgard", "Dredgion Crash Site")]
    total = sum(len(zd["sub_zones"]) for zd in importer.db.values())
    assert total == 35


# --- F-1715 ---
def test_reference_map_other_zone_raises(importer):
    with pytest.raises(ValueError):
        importer.get_reference_map_image("Verteron")


def test_reference_map_unknown_zone_raises(importer):
    with pytest.raises(ValueError):
        importer.get_reference_map_image("NoSuchZone")


def test_reference_map_altgard_loads(importer):
    img = importer.get_reference_map_image("Altgard")
    assert img is not None and img.shape[0] > 0 and img.shape[2] == 3


# --- F-1728 ---
def test_validate_bounds_uses_canvas(importer):
    nodes = [
        {"map_x": 100, "map_y": 100},      # valid
        {"map_x": 1280, "map_y": 720},     # valid (edge)
        {"map_x": 1281, "map_y": 100},     # beyond canvas
        {"map_x": 100, "map_y": 721},      # beyond canvas
        {"map_x": -5, "map_y": 100},       # negative
        {"map_x": 0, "map_y": 0},          # valid (edge)
        {"map_x": float("nan"), "map_y": 100},
        {"map_x": "abc", "map_y": 100},
        (500, 500),                        # valid tuple
        (9999, 9999),                      # invalid tuple
    ]
    assert importer.validate_imported_nodes_bounds(nodes) == 4


# --- F-1737 ---
def test_crop_patch_preserves_texture(importer):
    gray = np.full((720, 1280), 100, dtype=np.uint8)
    patch = QuestlogImporter.crop_80x80_gray_patch(gray, 640, 360)
    assert patch.shape == (80, 80)
    assert np.all(patch == 100)  # no normalize stretch to 25..235


# --- F-1725 ---
def test_live_map_detection_rejects_uniform_and_colourful(importer, textured_map_frame):
    uniform = np.full((720, 1280, 3), 100, dtype=np.uint8)
    assert QuestlogImporter._looks_like_game_map(uniform) is False
    colourful = np.zeros((720, 1280, 3), dtype=np.uint8)
    colourful[:, :, 0] = 30
    colourful[:, :, 1] = 200  # high saturation green screen
    colourful[:, :, 2] = 30
    assert QuestlogImporter._looks_like_game_map(colourful) is False
    _, textured_bgr = textured_map_frame
    assert QuestlogImporter._looks_like_game_map(textured_bgr) is True


class _FakeVision:
    def __init__(self, frame):
        self.frame = frame

    def grab_screen(self, grayscale=False):
        return self.frame


def test_live_capture_runs_even_with_fast_mode(importer, textured_map_frame):
    gray, frame = textured_map_frame
    got = importer._capture_or_load_map("Altgard", vision_engine=_FakeVision(frame),
                                        fast_mode=True)
    assert got.shape == (720, 1280)
    assert np.array_equal(got, gray)


# --- F-1710 / F-1711 / F-1712 / F-1724 / F-1733 (new API) ---
def test_import_dredgion_new_api(importer, tmp_path):
    rm = FakeNewRouteManager(tmp_path)
    progress = []
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle", "Orichalcum Ore"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm,
        progress_callback=lambda c, t, s: progress.append((c, t)),
        fast_mode=True,
    )
    assert result["success"] is True
    assert result["imported_count"] == 114
    assert result["already_there_count"] == 4  # metadata only, counted once
    assert result["skipped_sub_zones"] == []
    assert result["failed_patch_writes"] == 0
    assert len(rm._regions["Altgard"]["nodes"]) == 114
    expected18 = _convert_24s_to_18s_coords(DREDGION_CRASH_SITE_24S_COORDS)
    first = rm._regions["Altgard"]["nodes"][0]
    # F-1711: stored in full-map (x18, y18) space, not zoomed (x24, y24).
    assert (first.map_x, first.map_y) == expected18[0]
    assert first.norm_x == round(expected18[0][0] / 1280.0, 4)
    assert first.norm_y == round(expected18[0][1] / 720.0, 4)
    # F-1712: no fabricated round-robin resource assignment.
    assert {n.resource_type for n in rm._regions["Altgard"]["nodes"]} == {"unknown"}
    assert all(n.in_air is False for n in rm._regions["Altgard"]["nodes"])
    # F-1733: progress denominator is the nodes actually to import.
    assert progress[0] == (0, 100)  # "Looking at the map" step
    assert progress[-1] == (114, 114)
    assert all(t == 114 for _, t in progress[1:])
    assert os.path.isfile(os.path.join(rm._get_region_patches_dir("Altgard"), "node_1.png"))


def test_import_skips_synthetic_subzone(importer, tmp_path):
    rm = FakeNewRouteManager(tmp_path)
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Nameless Cemetery"], route_manager=rm, fast_mode=True)
    assert result["success"] is False
    assert result["imported_count"] == 0
    assert result["skipped_sub_zones"] == ["Nameless Cemetery"]


def test_import_mixed_verified_and_synthetic(importer, tmp_path):
    rm = FakeNewRouteManager(tmp_path)
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site", "Nameless Cemetery"],
        route_manager=rm, fast_mode=True)
    assert result["success"] is True
    assert result["imported_count"] == 114
    assert result["skipped_sub_zones"] == ["Nameless Cemetery"]
    assert result["routes"] == ["Dredgion Crash Site"]


def test_import_unknown_zone_fails(importer, tmp_path):
    rm = FakeNewRouteManager(tmp_path)
    result = importer.import_sub_zones(
        zone="NoSuchZone", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm, fast_mode=True)
    assert result["success"] is False
    assert result["imported_count"] == 0


def test_already_there_not_double_counted(importer, tmp_path):
    rm = FakeNewRouteManager(tmp_path)
    rm.create_region("Altgard")
    x18, y18 = _convert_24s_to_18s_coords(DREDGION_CRASH_SITE_24S_COORDS)[0]
    _seed_new_node(rm, "Altgard", x18, y18, node_id=7)
    progress = []
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm,
        progress_callback=lambda c, t, s: progress.append((c, t)), fast_mode=True)
    # 4 excluded by slice + 1 duplicate found in route manager = 5, not 9.
    assert result["already_there_count"] == 5
    assert result["imported_count"] == 113
    assert progress[-1] == (113, 113)


# --- F-1729 ---
def test_prefixed_x24_duplicate_still_detected(importer, tmp_path):
    """Nodes written before the F-1711 fix stored zoomed (x24, y24) coords;
    they must still be recognised as already there."""
    rm = FakeNewRouteManager(tmp_path)
    rm.create_region("Altgard")
    x24, y24 = DREDGION_CRASH_SITE_24S_COORDS[0]
    _seed_new_node(rm, "Altgard", x24, y24, node_id=3)
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm, fast_mode=True)
    assert result["already_there_count"] == 5  # 4 sliced out + 1 pre-fix duplicate
    assert result["imported_count"] == 113


def test_failed_patch_write_skips_node(importer, tmp_path, monkeypatch):
    import core.questlog_importer as qi
    monkeypatch.setattr(qi, "_imwrite_unicode", lambda path, img: False)
    rm = FakeNewRouteManager(tmp_path)
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm, fast_mode=True)
    assert result["success"] is False
    assert result["imported_count"] == 0
    assert result["failed_patch_writes"] == 114
    assert rm._regions["Altgard"]["nodes"] == []


# --- F-1730 (legacy) ---
def test_legacy_import_id_is_max_plus_one(importer, tmp_path):
    rm = FakeLegacyRouteManager(tmp_path)
    rm.data["regions"]["Altgard"] = {
        "routes": {"All": [2, 5, 9], "Dredgion Crash Site": [2, 5, 9]},
        "nodes": [
            {"id": 2, "map_x": 1, "map_y": 1},
            {"id": 5, "map_x": 2, "map_y": 2},
            {"id": 9, "map_x": 3, "map_y": 3},
        ],
    }
    result = importer.import_sub_zones(
        zone="Altgard", selected_resources=["Odyle"],
        selected_sub_zones=["Dredgion Crash Site"], route_manager=rm, fast_mode=True)
    assert result["success"] is True
    assert result["imported_count"] == 114
    new_ids = [n["id"] for n in result["nodes"]]
    assert min(new_ids) == 10  # max(2,5,9)+1, not len(3)+1 == 4
    assert len(set(new_ids)) == 114
    assert all(n["resource_type"] == "unknown" for n in result["nodes"])


# --- F-1716 / F-1735 (dialog) ---
def test_execute_import_failure_does_not_accept(qapp, importer, tmp_path, monkeypatch):
    import core.questlog_importer as qi
    monkeypatch.setattr(qi.QMessageBox, "critical", lambda *a, **k: None)
    rm = FakeLegacyRouteManager(tmp_path)
    dlg = QuestlogImportDialog(route_manager=rm, importer=importer,
                               ask_confirm_dialog=False, fast_mode=True)
    result = dlg.execute_import(zone="NoSuchZone", resources=["Odyle"],
                                sub_zones=["Dredgion Crash Site"])
    assert result["success"] is False
    from PyQt6.QtWidgets import QDialog
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert dlg.btn_import.isEnabled()  # controls re-enabled, dialog stays open


def test_empty_selection_warns_instead_of_silent_return(qapp, importer, tmp_path, monkeypatch):
    import core.questlog_importer as qi
    warned = []
    monkeypatch.setattr(qi.QMessageBox, "warning",
                        lambda *a, **k: warned.append(True))
    rm = FakeLegacyRouteManager(tmp_path)
    dlg = QuestlogImportDialog(route_manager=rm, importer=importer,
                               ask_confirm_dialog=False, fast_mode=True)
    dlg.list_resources.clearSelection()
    dlg.list_sub_zones.clearSelection()
    dlg._on_import_clicked()
    assert warned, "empty selection must warn, not silently return"
    assert dlg.last_import_result is None
