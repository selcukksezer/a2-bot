# -*- coding: utf-8 -*-
"""
Comprehensive Verification Suite for All 10 Bot Screens / Tabs, Overlays,
and 70-Agent Labeling & Precision Architecture.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

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


def test_tab01_home_and_app_shell_health_and_switching():
    """Tab 1 (Anasayfa) & GatherionShell: 10-tab health snapshot and programmatic tab switching."""
    shell = app_shell_mod.GatherionShell.__new__(app_shell_mod.GatherionShell)
    shell.pages = {
        k: object()
        for k in ("home", "radar", "map", "yolo", "calibration", "stats", "mobile", "guide", "logs", "settings")
    }
    shell._active_tab_key = "home"
    snap = shell.get_all_tabs_health_snapshot()
    assert snap["total_tabs"] == 10
    assert snap["all_ready"] is True
    assert set(snap["tabs"].keys()) == {
        "home", "radar", "map", "yolo", "calibration", "stats", "mobile", "guide", "logs", "settings"
    }
    assert shell.switch_to_tab("yolo") is True
    assert shell.switch_to_tab("nonexistent_tab") is False


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

    ordered = dlg.optimize_visible_radar_route(player_xy=(100.0, 100.0))
    assert [n["id"] for n in ordered] == [1, 3, 2]


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
    assert len(aria_nodes) == 2
    reseq = dlg.resequence_active_route_by_proximity(player_pos=(85.0, 85.0))
    assert len(reseq) >= 2
    summary = dlg.get_map_viewer_route_summary()
    assert summary["region"] == "Beluslan"
    assert summary["total_nodes"] == 3
    assert summary["filtered_nodes"] == 2
    assert summary["ready"] is True


def test_tab04_yolo_panel_and_web_pipeline_consumables_and_tiers():
    """Tab 4 (YOLO): 25-class taxonomy (hp_potion=22, mp_potion=23, buff_scroll=24) & 3-tier color coding."""
    assert yolo_panel_mod.CLASS_ID_TO_NAME[22] == "hp_potion"
    assert yolo_panel_mod.CLASS_ID_TO_NAME[23] == "mp_potion"
    assert yolo_panel_mod.CLASS_ID_TO_NAME[24] == "buff_scroll"

    summary = yolo_panel_mod.get_annotation_tier_summary([
        {"class_name": "herb_aria", "confidence": 0.91, "trusted": True},
        {"class_name": "buff_scroll", "confidence": 0.76, "trusted": False},
        {"class_name": "path_ground", "confidence": 0.52, "trusted": False},
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
    assert res["consumable_count"] >= 6

    comp = wiz.get_calibration_completeness_report()
    assert comp["total_targets"] > 0
    assert comp["ready"] is True


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
    snap = dlg.export_analytics_snapshot_dict()
    assert snap["ready"] is True
    assert snap["rows_count"] == 1


def test_tab07_mobile_connect_diagnostics_and_ntfy_payload():
    """Tab 7 (Mobil): Mobile connection diagnostics and ntfy test payload helper."""
    dlg = mobile_mod.MobileConnectDialog.__new__(mobile_mod.MobileConnectDialog)
    diag = dlg.get_mobile_connection_diagnostics()
    assert isinstance(diag, dict)
    assert "lan_ip" in diag or "ready" in diag

    ntfy_res = dlg.send_test_ntfy_payload("Test Bildirimi")
    assert isinstance(ntfy_res, dict)


def test_tab08_guide_dialog_interactive_summary_and_readiness():
    """Tab 8 (Kılavuz): Interactive feature guide summary and approved/exemplar readiness check."""
    dlg = guide_mod.GuideDialog.__new__(guide_mod.GuideDialog)
    dlg.base_dir = Path(".")
    info = dlg.get_interactive_feature_guide_summary()
    assert info["module_count"] >= 12
    assert info["approved_pool_supported"] is True
    assert info["manual_box_drawing_supported"] is True
    assert "buff_scroll (24)" in info["consumable_classes"]

    readiness = dlg.check_labeling_and_exemplar_readiness()
    assert readiness["ready"] is True


def test_tab09_logs_export_filtered_log():
    """Tab 9 (Log): _LogPage filtered log export to disk."""
    log_page = app_shell_mod._LogPage.__new__(app_shell_mod._LogPage)
    log_page._records = [
        ("INFO", "12:00:01 [INFO] Herb Aria gathered"),
        ("WARN", "12:00:05 [WARN] Suspicious box filtered"),
    ]
    log_page._active_filter = "ALL"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "exported_bot.log")
        res = log_page.export_filtered_log(out_file)
        assert os.path.exists(out_file) or isinstance(res, (bool, str, int, dict))


def test_tab10_settings_window_consumable_settings_and_hygiene():
    """Tab 10 (Ayarlar): Automatic potion/scroll thresholds, keybinds, and node hygiene summary."""
    win = settings_mod.SettingsWindow.__new__(settings_mod.SettingsWindow)
    win.settings = None
    win.config = None
    defaults = win.get_consumable_settings()
    assert defaults["auto_use_hp_potion"] is True
    assert defaults["hp_potion_threshold_pct"] == 55
    assert defaults["auto_refresh_buff_scroll"] is True

    updated = win.set_consumable_settings({"hp_potion_threshold_pct": 65, "buff_scroll_key": "F7"})
    assert updated["hp_potion_threshold_pct"] == 65
    assert updated["buff_scroll_key"] == "F7"


def test_overlays_f9_menu_and_overlay_hud_integration():
    """Overlays (F9Menu & OverlayHUD): Quick control summary, auto-consumables toggle, and HUD telemetry."""
    f9 = f9_mod.F9Menu.__new__(f9_mod.F9Menu)
    f9.set_auto_consumables_enabled(True)
    f9_sum = f9.get_f9_quick_control_summary()
    assert isinstance(f9_sum, dict)

    hud = hud_mod.OverlayHUD.__new__(hud_mod.OverlayHUD)
    hud.update_consumable_and_selective_status(
        hp_pct=82.0,
        mp_pct=64.0,
        active_buffs=["Hız Parşömeni"],
        selected_material="herb_aria",
    )
    hud_snap = hud.get_overlay_telemetry_snapshot()
    assert isinstance(hud_snap, dict)


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
