# -*- coding: utf-8 -*-
"""
Comprehensive Verification Suite for All 10 Bot Screens / Tabs, Overlays,
and 70-Agent Labeling & Precision Architecture.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Headless Linux (CI/VM without $DISPLAY): run Qt offscreen instead of failing
# to connect to a display. Leaves real displays and Windows/macOS alone.
if os.environ.get("QT_QPA_PLATFORM") is None and os.name == "posix" and not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import ui.analytics_dialog as analytics_mod
import ui.app_shell as app_shell_mod
import ui.calibration_wizard as cal_mod
import ui.cartography_radar as radar_mod
import ui.f9_menu as f9_mod
import ui.guide_dialog as guide_mod
import ui.map_viewer as map_mod
import ui.mobile_connect_dialog as mobile_mod
import ui.overlay_hud as hud_mod
import ui.settings_window as settings_mod
import ui.shell_page_frame as page_frame_mod
import ui.shell_theme as theme_mod
import ui.yolo_factory_dialog as yolo_factory_mod
import ui.yolo_panel as yolo_panel_mod
import ui.yolo_pipeline_web as yolo_web_mod


_TAB_KEYS = (
    "home", "radar", "map", "yolo", "calibration",
    "stats", "mobile", "guide", "logs", "settings",
)


def test_tab01_home_and_app_shell_health_and_switching():
    """Tab 1 (Anasayfa) & GatherionShell: 10-tab health snapshot and programmatic tab switching."""
    # Distinct page object per tab so the snapshot must actually report each
    # mounted page (a shared ``object()`` per tab cannot tell tabs apart).
    pages = {}
    for key in _TAB_KEYS:
        pages[key] = type(f"Page_{key}", (), {})()
    shell = app_shell_mod.GatherionShell.__new__(app_shell_mod.GatherionShell)
    shell.pages = pages
    shell._active_tab_key = "home"
    snap = shell.get_all_tabs_health_snapshot()
    assert snap["total_tabs"] == 10
    assert snap["all_ready"] is True
    assert set(snap["tabs"].keys()) == set(_TAB_KEYS)
    for key in _TAB_KEYS:
        assert snap["tabs"][key]["mounted"] is True
        assert snap["tabs"][key]["class_name"] == f"Page_{key}"
    assert snap["active_tab"] == "home"

    assert shell.switch_to_tab("yolo") is True
    assert shell._active_tab_key == "yolo"
    assert shell.get_all_tabs_health_snapshot()["active_tab"] == "yolo"
    # An invalid switch must fail AND leave the active tab untouched.
    assert shell.switch_to_tab("nonexistent_tab") is False
    assert shell._active_tab_key == "yolo"
    assert shell.switch_to_tab("  ") is False
    assert shell._active_tab_key == "yolo"

    # A missing page must show up as unmounted instead of silently "ready".
    shell2 = app_shell_mod.GatherionShell.__new__(app_shell_mod.GatherionShell)
    shell2.pages = {"home": object()}
    shell2._active_tab_key = "home"
    snap2 = shell2.get_all_tabs_health_snapshot()
    assert snap2["total_tabs"] == 10
    assert snap2["tabs"]["home"]["mounted"] is True
    for key in _TAB_KEYS:
        if key != "home":
            assert snap2["tabs"][key]["mounted"] is False
            assert snap2["tabs"][key]["class_name"] == "Unmounted"


def test_tab02_radar_diagnostics_and_route_optimization():
    """Tab 2 (Radar): CartographyRadarDialog route diagnostics and nearest-neighbor proximity ordering."""
    dlg = radar_mod.CartographyRadarDialog.__new__(radar_mod.CartographyRadarDialog)

    class _DummyCanvas:
        def __init__(self):
            self._ordered_nodes = [
                {"id": 1, "map_x": 100, "map_y": 100, "source_type": "minimap_auto"},
                {"id": 2, "map_x": 500, "map_y": 500, "source_type": "manual"},
                {"id": 3, "map_x": 120, "map_y": 110, "source_type": "minimap_auto"},
            ]
            self._material_filter = "herb_aria"

        def set_sequential_nodes(self, nodes):
            self._ordered_nodes = list(nodes)

    dlg.canvas = _DummyCanvas()
    diag = dlg.get_radar_route_diagnostics()
    assert diag["total_nodes"] == 3
    assert diag["minimap_auto_nodes"] == 2
    assert diag["material_filter"] == "herb_aria"
    assert diag["radar_ready"] is True

    # Diagnostics must follow the canvas content, not a fixed answer.
    dlg.canvas._ordered_nodes = dlg.canvas._ordered_nodes[:1]
    diag2 = dlg.get_radar_route_diagnostics()
    assert diag2["total_nodes"] == 1
    assert diag2["minimap_auto_nodes"] == 1

    # 6-node case: the greedy nearest-neighbour order differs from both the
    # input order and from a plain distance-from-player sort, so the exact
    # order pins the algorithm instead of a 3-node coincidence.
    # Player (0,0); A(10,0) B(11,10) C(100,0) D(101,10) E(50,50) F(51,60).
    # NN walk: A -> B -> E -> F -> D -> C.
    nodes = [
        {"id": 3, "map_x": 100, "map_y": 0, "source_type": "manual"},
        {"id": 6, "map_x": 51, "map_y": 60, "source_type": "minimap_auto"},
        {"id": 1, "map_x": 10, "map_y": 0, "source_type": "minimap_auto"},
        {"id": 5, "map_x": 50, "map_y": 50, "source_type": "minimap_auto"},
        {"id": 4, "map_x": 101, "map_y": 10, "source_type": "manual"},
        {"id": 2, "map_x": 11, "map_y": 10, "source_type": "minimap_auto"},
    ]
    dlg.canvas._ordered_nodes = list(nodes)
    ordered = dlg.optimize_visible_radar_route(player_xy=(0.0, 0.0))
    assert [n["id"] for n in ordered] == [1, 2, 5, 6, 4, 3]
    # [O3]: radar assigns 1-based seq_index (len(ordered)+1, D34) — same
    # convention as RouteManager.build_proximity_material_route (D25/F-1647).
    assert [n["seq_index"] for n in ordered] == [1, 2, 3, 4, 5, 6]
    # Greedy property: every step must go to the nearest still-unvisited node.
    remaining = {n["id"]: n for n in nodes}
    cx, cy = 0.0, 0.0
    for step in ordered:
        nearest = min(
            remaining.values(),
            key=lambda n: math.hypot(n["map_x"] - cx, n["map_y"] - cy),
        )
        assert step["id"] == nearest["id"]
        remaining.pop(step["id"])
        cx, cy = step["map_x"], step["map_y"]
    # Total route length must match the walked path exactly.
    total = sum(
        math.hypot(b["map_x"] - a["map_x"], b["map_y"] - a["map_y"])
        for a, b in zip(ordered, ordered[1:])
    )
    assert total == pytest.approx(
        math.hypot(1, 10) + math.hypot(39, 40) + math.hypot(1, 10)
        + math.hypot(50, 50) + math.hypot(1, 10)
    )
    # The canvas must receive the reordered nodes (side effect, not just return).
    assert [n["id"] for n in dlg.canvas._ordered_nodes] == [1, 2, 5, 6, 4, 3]


def test_tab03_map_viewer_material_filter_and_proximity_route():
    """Tab 3 (Harita): MapViewerDialog selective material filtering, proximity resequencing, and route summary."""
    dlg = map_mod.MapViewerDialog.__new__(map_mod.MapViewerDialog)
    dlg._current_region = "Beluslan"
    dlg._current_route = "All"
    dlg._selected_material_filter = "Aria"
    dlg._current_nodes = [
        {"id": 10, "x": 300, "y": 300, "type": "Orichalcum"},
        {"id": 11, "x": 110, "y": 110, "type": "Aria"},
        {"id": 12, "x": 90, "y": 90, "type": "Aria"},
    ]

    aria_nodes = dlg.filter_nodes_by_selected_material("Aria")
    assert {n["id"] for n in aria_nodes} == {11, 12}
    assert all(n["type"] == "Aria" for n in aria_nodes)
    # Filtering must actually exclude the non-matching node, and a different
    # material must select a different set.
    assert {n["id"] for n in dlg.filter_nodes_by_selected_material("Orichalcum")} == {10}
    # Restore the Aria filter: the resequence below must run on the Aria pool.
    dlg._selected_material_filter = "Aria"

    # Exact proximity order from (85,85): node 12 (90,90) is nearer than 11.
    reseq = dlg.resequence_active_route_by_proximity(player_pos=(85.0, 85.0))
    assert reseq == [12, 11]
    # From the opposite corner the order must flip.
    reseq2 = dlg.resequence_active_route_by_proximity(player_pos=(300.0, 300.0))
    assert reseq2 == [11, 12]
    summary = dlg.get_map_viewer_route_summary()
    assert summary["region"] == "Beluslan"
    assert summary["total_nodes"] == 3
    assert summary["filtered_nodes"] == 2
    assert summary["selected_material"] == "Aria"
    assert summary["ready"] is True


def test_tab04_yolo_panel_and_web_pipeline_consumables_and_tiers():
    """Tab 4 (YOLO): 25-class taxonomy (hp_potion=22, mp_potion=23, buff_scroll=24) & 3-tier color coding."""
    assert yolo_panel_mod.CLASS_ID_TO_NAME[22] == "hp_potion"
    assert yolo_panel_mod.CLASS_ID_TO_NAME[23] == "mp_potion"
    assert yolo_panel_mod.CLASS_ID_TO_NAME[24] == "buff_scroll"

    summary = yolo_panel_mod.get_annotation_tier_summary([
        # P2-23 (Y002-K1): geometrisi eksik kutu valid=False sayılır,
        # tier sayacına girmez — o yüzden fikstürler gerçek geometri taşır.
        {"class_name": "herb_aria", "confidence": 0.91, "trusted": True, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        {"class_name": "buff_scroll", "confidence": 0.76, "trusted": False, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        {"class_name": "path_ground", "confidence": 0.52, "trusted": False, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
    ])
    assert summary["confident"] == 1
    assert summary["suspicious"] == 1
    assert summary["low"] == 1
    assert summary["colors"] == {
        "confident": "#22c55e",
        "suspicious": "#fb923c",
        "low": "#ef4444",
    }

    web_state = yolo_web_mod.get_web_pipeline_extended_state()
    assert web_state["canonical_class_count"] >= 25
    assert yolo_web_mod.confidence_to_hex_color(0.90) == "#22c55e"
    assert yolo_web_mod.confidence_to_hex_color(0.75) == "#fb923c"
    assert yolo_web_mod.confidence_to_hex_color(0.55) == "#ef4444"


def test_tab05_calibration_wizard_consumable_hud_and_completeness():
    """Tab 5 (Kalibrasyon): Auto-calibration of upper HUD consumable slots and completeness report."""
    wiz = cal_mod.CalibrationWizard.__new__(cal_mod.CalibrationWizard)
    wiz.templates_dir = Path("templates")
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake_frame[:, :] = (40, 60, 180)
    res = wiz.auto_calibrate_consumables_and_hud(frame_bgr=fake_frame)
    assert res["calibrated"] is True
    assert res["resolution"] == (1280, 720)
    # 9 consumable slots (F1..F8 + X/B row) with in-frame bboxes, 1-based indices.
    assert res["consumable_count"] == 9
    assert len(res["consumable_slots"]) == 9
    for idx, slot in enumerate(res["consumable_slots"], start=1):
        assert slot["slot_index"] == idx
        bx, by, bw, bh = slot["bbox"]
        assert 0 <= bx and 0 <= by and bx + bw <= 1280 and by + bh <= 720
        assert bw > 0 and bh > 0
        assert isinstance(slot["class_name"], str) and slot["class_name"]
        assert 0.0 <= slot["confidence"] <= 1.0
    # A different resolution must be reported, not echoed from the first call.
    # F-0523 [O3]: the flat-colour 640x360 frame classifies zero slots, so
    # honest calibration reports calibrated=False (no_classified_slots);
    # the old True came from the fabricated slot filler D30 removed.
    small = np.zeros((360, 640, 3), dtype=np.uint8)
    res_small = wiz.auto_calibrate_consumables_and_hud(frame_bgr=small)
    assert res_small["calibrated"] is False
    assert res_small["resolution"] == (640, 360)
    assert res_small["consumable_count"] == 0
    assert res_small["reason"] == "no_classified_slots"
    # No frame at all must NOT calibrate (negative case, was never exercised).
    assert wiz.auto_calibrate_consumables_and_hud(frame_bgr=None)["calibrated"] is False
    assert wiz.auto_calibrate_consumables_and_hud(frame_bgr=None)["reason"] == "no_frame"
    assert wiz.auto_calibrate_consumables_and_hud(
        frame_bgr=np.zeros((0, 0, 3), dtype=np.uint8)
    )["calibrated"] is False

    comp = wiz.get_calibration_completeness_report()
    assert comp["total_targets"] > 0
    # F-0556 [O3]: honest readiness — 15 of the 25 canonical templates are
    # missing from the repo, so the wizard is not ready.
    assert comp["ready"] is False
    assert comp["missing_count"] == 15
    assert comp["completeness_pct"] == pytest.approx(40.0)

    # Completeness must reflect the real template dir: an empty dir scores 0%.
    with tempfile.TemporaryDirectory() as tmpdir:
        wiz.templates_dir = Path(tmpdir)
        empty = wiz.get_calibration_completeness_report()
        assert empty["total_targets"] == comp["total_targets"]
        assert empty["calibrated_count"] == 0
        assert empty["missing_count"] == empty["total_targets"]
        assert empty["completeness_pct"] == 0.0
        assert len(empty["missing_targets"]) > 0
        # Adding one real template file must move the count by exactly one:
        # completeness counts unique template FILES (D30/F-0525), so the
        # targets sharing that filename are covered together by the one file.
        # [O3]: the old `sharing` expectation counted TEMPLATE_TARGETS
        # entries, but the report counts unique files by design.
        targets = getattr(wiz, "TEMPLATE_TARGETS", {}) or {}
        first_name = next(iter(targets.values())).get("filename")
        assert first_name
        sharing = sum(
            1 for meta in targets.values() if meta.get("filename") == first_name
        )
        assert sharing >= 1
        (Path(tmpdir) / first_name).write_bytes(b"fake-template")
        one = wiz.get_calibration_completeness_report()
        assert one["calibrated_count"] == 1
        assert one["missing_count"] == one["total_targets"] - 1
        assert one["completeness_pct"] == pytest.approx(
            round(100.0 * 1 / one["total_targets"], 1)
        )


def test_tab06_stats_analytics_combat_and_consumables():
    """Tab 6 (İstatistik): Combat, potion/scroll usage, and selective gather analytics export."""
    dlg = analytics_mod.AnalyticsDialog.__new__(analytics_mod.AnalyticsDialog)
    dlg._active_range = "today"
    dlg._last_metrics = {
        "combat_count": 7,
        "gathered_total": 42,
        "failed_total": 2,
        "hp_potions_used": 5,
        "mp_potions_used": 8,
        "buff_scrolls_used": 3,
        "selective_gather_hits": 39,
        "success_rate_pct": 95.4,
    }
    dlg._last_rows = [{"resource_type": "herb_aria", "count": 42}]
    ca = dlg.get_combat_and_consumable_analytics()
    assert ca["combat_count"] == 7
    assert ca["hp_potions_used"] == 5
    assert ca["mp_potions_used"] == 8
    assert ca["buff_scrolls_used"] == 3
    assert ca["gathered_total"] == 42
    assert ca["selective_gather_hits"] == 39
    assert ca["success_rate_pct"] == 95.4
    snap = dlg.export_analytics_snapshot_dict()
    assert snap["ready"] is True
    assert snap["rows_count"] == 1
    assert snap["active_range"] == "today"
    assert snap["combat_and_consumables"] == ca

    # The analytics must read the metrics, not echo a fixed answer: a second,
    # different metrics set must produce different output.
    dlg._last_metrics = {
        "combat_count": 0,
        "gathered_total": 10,
        "failed_total": 1,
        "hp_potions_used": 0,
        "mp_potions_used": 2,
        "buff_scrolls_used": 0,
        "success_rate_pct": 50.0,
    }
    dlg._last_rows = [
        {"resource_type": "herb_aria", "count": 6},
        {"resource_type": "ore_orichalcum", "count": 4},
    ]
    ca2 = dlg.get_combat_and_consumable_analytics()
    assert ca2["combat_count"] == 0
    assert ca2["mp_potions_used"] == 2
    assert ca2["success_rate_pct"] == 50.0
    # F-0548: selective_gather_hits is never silently equated with
    # gathered_total — absent means unknown (None), not a fabricated 10.
    assert ca2["selective_gather_hits"] is None
    snap2 = dlg.export_analytics_snapshot_dict()
    assert snap2["rows_count"] == 2
    assert snap2["combat_and_consumables"] == ca2

    # Empty metrics must yield zeros for the counted totals and None for
    # the keys telemetry does not track (F-0548: unknown is None, never a
    # fabricated 0; success rate unknown, never a fabricated 100.0).
    dlg._last_metrics = {}
    dlg._last_rows = []
    ca3 = dlg.get_combat_and_consumable_analytics()
    assert ca3 == {
        "combat_count": 0,
        "gathered_total": 0,
        "failed_total": 0,
        "hp_potions_used": None,
        "mp_potions_used": None,
        "buff_scrolls_used": None,
        "selective_gather_hits": None,
        "success_rate_pct": None,
    }
    assert dlg.export_analytics_snapshot_dict()["rows_count"] == 0


def test_tab07_mobile_connect_diagnostics_and_ntfy_payload():
    """Tab 7 (Mobil): Mobile connection diagnostics and ntfy test payload helper."""
    dlg = mobile_mod.MobileConnectDialog.__new__(mobile_mod.MobileConnectDialog)
    diag = dlg.get_mobile_connection_diagnostics()
    assert isinstance(diag, dict)
    assert diag["lan"]["url"].startswith("http://")
    assert isinstance(diag["lan"]["ip"], str) and diag["lan"]["ip"]
    assert isinstance(diag["lan"]["port"], int) and diag["lan"]["port"] > 0
    assert diag["ntfy"]["url"].startswith("https://ntfy.sh/")
    assert diag["ntfy"]["topic"]
    assert set(diag["remote_commands"].keys()) == {
        "pause_resume", "emergency_stop", "toggle_pick_collect",
    }
    # D09 / Y-7 (dilim 22): remote commands are honestly gated — pause_resume
    # and emergency_stop are enabled only while the ntfy relay listener is
    # actually running (a bare __new__ dialog has no tunnel_mgr, so both are
    # False here), and toggle_pick_collect has no server counterpart at all,
    # so it is always unsupported, with a reason.
    assert diag["remote_commands"]["pause_resume"]["enabled"] is False
    assert diag["remote_commands"]["pause_resume"]["supported"] is False
    assert diag["remote_commands"]["emergency_stop"]["enabled"] is False
    assert diag["remote_commands"]["emergency_stop"]["supported"] is False
    assert diag["remote_commands"]["toggle_pick_collect"]["enabled"] is False
    assert diag["remote_commands"]["toggle_pick_collect"]["supported"] is False
    assert diag["remote_commands"]["toggle_pick_collect"]["reason"]
    assert diag["remote_commands_ready"] is False
    for cmd in diag["remote_commands"].values():
        assert cmd["endpoint"].startswith("/api/command/")

    # With a running ntfy relay listener the two real commands flip to
    # enabled — toggle_pick_collect stays unsupported (Y-7).
    class _FakeRelay:
        def get_status(self):
            return {"listener_running": True}

    class _FakeTunnelMgr:
        def __init__(self):
            self.ntfy_relay = _FakeRelay()

    dlg.tunnel_mgr = _FakeTunnelMgr()
    diag_live = dlg.get_mobile_connection_diagnostics()
    assert diag_live["remote_commands"]["pause_resume"]["enabled"] is True
    assert diag_live["remote_commands"]["pause_resume"]["supported"] is True
    assert diag_live["remote_commands"]["emergency_stop"]["enabled"] is True
    assert diag_live["remote_commands"]["emergency_stop"]["supported"] is True
    assert diag_live["remote_commands"]["toggle_pick_collect"]["enabled"] is False
    assert diag_live["remote_commands"]["toggle_pick_collect"]["supported"] is False
    assert diag_live["remote_commands_ready"] is True
    assert diag["remote_command_labels"] == [
        "Duraklat/Devam", "Acil Durdur", "Seç-Topla değiştir",
    ]
    assert isinstance(diag["qr_verification"]["valid"], bool)

    # ntfy must be exercised WITHOUT a real network call: patch urlopen.
    # 1) Offline (urlopen raises) -> simulated fallback, no live delivery.
    with mock.patch.object(
        mobile_mod.urllib.request, "urlopen", side_effect=OSError("offline")
    ) as mocked:
        res = dlg.send_test_ntfy_payload("Test Bildirimi")
        mocked.assert_called_once()
        req = mocked.call_args.args[0]
        assert req.data == "Test Bildirimi".encode("utf-8")
        assert req.full_url.startswith("https://ntfy.sh/")
        # D09: offline outcome is honestly "failed_offline" (status reports
        # the outcome); "simulated_fallback" is the delivery MODE, not the
        # status — the old status expectation predates D09.
        assert res["status"] == "failed_offline"
        assert res["mode"] == "simulated_fallback"
        assert res["simulated"] is True
        assert res["fallback"] is True
        assert res["message"] == "Test Bildirimi"
        assert res["channel"] == "ntfy.sh"
        assert res["error"]
    # 2) Live 200 -> delivered, same message echoed in the payload record.
    fake_resp = mock.MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False
    with mock.patch.object(
        mobile_mod.urllib.request, "urlopen", return_value=fake_resp
    ) as mocked:
        res2 = dlg.send_test_ntfy_payload("Canli Test")
        mocked.assert_called_once()
        req2 = mocked.call_args.args[0]
        assert req2.data == "Canli Test".encode("utf-8")
        assert req2.get_method() == "POST"
        assert res2["status"] == "delivered"
        assert res2["mode"] == "live_ntfy"
        assert res2["simulated"] is False
        assert res2["http_status"] == 200
        assert res2["message"] == "Canli Test"


def test_tab08_guide_dialog_interactive_summary_and_readiness():
    """Tab 8 (Kılavuz): Interactive feature guide summary and approved/exemplar readiness check."""
    dlg = guide_mod.GuideDialog.__new__(guide_mod.GuideDialog)
    dlg.base_dir = Path(".")
    info = dlg.get_interactive_feature_guide_summary()
    assert info["module_count"] >= 12
    assert len(info["modules"]) == info["module_count"]
    for mod in info["modules"]:
        assert isinstance(mod.get("module"), str) and mod["module"]
        assert isinstance(mod.get("summary"), str) and mod["summary"]
    assert info["approved_pool_supported"] is True
    assert info["manual_box_drawing_supported"] is True
    assert "buff_scroll (24)" in info["consumable_classes"]
    assert info["confidence_tiers"]["confident_green"].startswith(">= 85%")

    # Readiness must reflect the real filesystem, not a hardcoded True:
    # an empty base dir has neither the approved pool nor the exemplar file.
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dlg.base_dir = base
        missing = dlg.check_labeling_and_exemplar_readiness()
        assert missing["approved_pool_dir"] == str(base / "dataset" / "approved")
        assert missing["approved_pool_exists"] is False
        assert missing["exemplar_file_exists"] is False
        # Creating both must flip the flags.
        (base / "dataset" / "approved").mkdir(parents=True)
        (base / "config").mkdir(parents=True)
        (base / "config" / "learned_exemplars.json").write_text("{}", encoding="utf-8")
        present = dlg.check_labeling_and_exemplar_readiness()
        assert present["approved_pool_exists"] is True
        assert present["exemplar_file_exists"] is True


def test_tab09_logs_export_filtered_log():
    """Tab 9 (Log): _LogPage filtered log export to disk."""
    log_page = app_shell_mod._LogPage.__new__(app_shell_mod._LogPage)
    log_page._records = [
        ("INFO", "12:00:01 [INFO] Herb Aria gathered"),
        ("WARN", "12:00:05 [WARN] Suspicious box filtered"),
    ]
    log_page._active_filter = "ALL"
    expected = (
        "12:00:01 [INFO] Herb Aria gathered\n"
        "12:00:05 [WARN] Suspicious box filtered"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "exported_bot.log")
        res = log_page.export_filtered_log(out_file)
        # Both the file and the return value must carry the exact log lines;
        # either one alone (or a bare bool) is not a passing export.
        assert os.path.exists(out_file)
        assert Path(out_file).read_text(encoding="utf-8") == expected
        assert isinstance(res, str)
        assert res == expected
        assert "Herb Aria gathered" in res
        assert "Suspicious box filtered" in res
    # An empty record list must export an empty log, not stale content.
    log_page._records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "exported_empty.log")
        res2 = log_page.export_filtered_log(out_file)
        assert res2 == ""
        assert Path(out_file).read_text(encoding="utf-8") == ""


def test_tab10_settings_window_consumable_settings_and_hygiene():
    """Tab 10 (Ayarlar): Automatic potion/scroll thresholds, keybinds, and node hygiene summary."""
    win = settings_mod.SettingsWindow.__new__(settings_mod.SettingsWindow)
    win.settings = None
    win.config = None
    defaults = win.get_consumable_settings()
    assert defaults["auto_use_hp_potion"] is True
    assert defaults["hp_potion_threshold_pct"] == 65  # no10/F-0540: DEFAULT_SETTINGS tek kaynak (65); 55 eski hardcode
    assert defaults["auto_refresh_buff_scroll"] is True

    updated = win.set_consumable_settings({"hp_potion_threshold_pct": 65, "buff_scroll_key": "F7"})
    assert updated["hp_potion_threshold_pct"] == 65
    assert updated["buff_scroll_key"] == "F7"


def test_overlays_f9_menu_and_overlay_hud_integration():
    """Overlays (F9Menu & OverlayHUD): Quick control summary, auto-consumables toggle, and HUD telemetry."""
    f9 = f9_mod.F9Menu.__new__(f9_mod.F9Menu)
    f9._selected_materials = ["herb_aria", "ore_orichalcum"]
    f9._active_mode = "route"
    f9.set_auto_consumables_enabled(True)
    f9_sum = f9.get_f9_quick_control_summary()
    assert isinstance(f9_sum, dict)
    assert f9_sum["auto_consumables_enabled"] is True
    assert f9_sum["auto_pot_scroll_enabled"] is True
    assert f9_sum["selected_materials"] == ["herb_aria", "ore_orichalcum"]
    assert f9_sum["selected_materials_count"] == 2
    assert f9_sum["active_mode"] == "route"
    # Toggling off must be reflected in the summary (not a fixed True).
    f9.set_auto_consumables_enabled(False)
    assert f9.is_auto_consumables_enabled() is False
    f9_sum2 = f9.get_f9_quick_control_summary()
    assert f9_sum2["auto_consumables_enabled"] is False
    assert f9_sum2["auto_pot_scroll_enabled"] is False
    assert f9_sum2["selected_materials_count"] == 2
    f9.set_auto_consumables_enabled(True)

    hud = hud_mod.OverlayHUD.__new__(hud_mod.OverlayHUD)
    payload = hud.update_consumable_and_selective_status(
        hp_pct=82.0,
        mp_pct=64.0,
        active_buffs=["Hız Parşömeni"],
        selected_material="herb_aria",
    )
    assert payload["hp_pct"] == 82.0
    assert payload["mp_pct"] == 64.0
    assert payload["active_scrolls"] == 1
    assert payload["selective_target"] == "herb_aria"
    assert payload["selective_active"] is True
    assert payload["hp_potion"] == "READY"
    assert payload["buff_scroll"] == "ACTIVE (1)"
    assert payload["badge_text"] == "HP 82% · MP 64% · S:1"
    hud_snap = hud.get_overlay_telemetry_snapshot()
    assert isinstance(hud_snap, dict)
    assert hud_snap["hp_pct"] == 82.0
    assert hud_snap["mp_pct"] == 64.0
    assert hud_snap["active_scrolls"] == 1
    assert hud_snap["selective_target"] == "herb_aria"
    assert hud_snap["hp_potion"] == "READY"
    assert hud_snap["buff_scroll"] == "ACTIVE (1)"
    # Low HP must flip the potion state to TRIGGERED in both payload and snapshot.
    low = hud.update_consumable_and_selective_status(
        hp_pct=30.0, mp_pct=20.0, active_buffs=[], selected_material=""
    )
    assert low["hp_potion"] == "TRIGGERED"
    assert low["mp_potion"] == "TRIGGERED"
    assert low["buff_scroll"] == "READY"
    assert low["selective_active"] is False
    low_snap = hud.get_overlay_telemetry_snapshot()
    assert low_snap["hp_pct"] == 30.0
    assert low_snap["hp_potion"] == "TRIGGERED"


def test_shell_theme_and_page_frame_badges():
    """Shared Shell Theme & Page Frame: 3-tier confidence QSS and status badge helper."""
    qss_green = theme_mod.get_confidence_badge_qss(0.92, trusted=True)
    qss_orange = theme_mod.get_confidence_badge_qss(0.75, trusted=False)
    qss_red = theme_mod.get_confidence_badge_qss(0.50, trusted=False)
    assert "#22c55e" in qss_green and "#ffffff" in qss_green
    assert "#fb923c" in qss_orange and "#ffffff" in qss_orange
    assert "#ef4444" in qss_red and "#ffffff" in qss_red

    meta = page_frame_mod.set_page_status_badge(None, "ONAYLI HAVUZ", tier="confident")
    assert meta["color"] == "#22c55e"
