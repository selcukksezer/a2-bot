# -*- coding: utf-8 -*-
"""PyQt6 SettingsWindow (Gatherion Ayarlar) prototype-conformance tests.

Covers the W3 conversion of
`your_files/gatherion-ayarlar-prototipi/gatherion-ayarlar-prototipi.html`
into `ui/settings_window.py`:

- every prototype form control exists and is bound to a real settings key
- Save & Apply really writes config/settings.json and emits settings_saved
  (the running bot picks that signal up in main._on_settings_saved)
- prototype validation (F/E alt < üst, hotkey ends with an F key and has
  a modifier) blocks a save instead of silently clamping
- dirty tag, profile cycle button (Dengeli/Hassas/Sıkı), double-click
  marks a node gathered via RouteManager, hotkey capture rules,
  infoline + bottom log strip

Runs offscreen; the real RouteManager cannot load on this VM (no cv2),
so a small fake with the same interface is used. The real
SettingsManager (config/schema.py) is used against a temp settings.json.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

from config.schema import SettingsManager
from ui import settings_window as settings_mod
from ui.settings_window import SettingsWindow, _HotkeyCaptureEdit


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeRouteManager:
    """Minimal RouteManager-shaped double (dict nodes, no disk writes)."""

    def __init__(self):
        self.active_region = "Altgard"
        self.active_route = "Dredgion Crash Site"
        self.data = {"active_region": "Altgard", "active_route": "Dredgion Crash Site"}
        self._nodes = {
            ("Altgard", "Dredgion Crash Site"): [
                {
                    "id": 245,
                    "routes": ["Dredgion Crash Site"],
                    "gathered_count": 0,
                    "visit_count": 0,
                    "last_visited": 0.0,
                    "not_gatherable": False,
                },
                {
                    "id": 40,
                    "routes": ["Dredgion Crash Site"],
                    "gathered_count": 2,
                    "visit_count": 2,
                    "last_visited": 0.0,
                    "not_gatherable": False,
                },
            ],
            ("Altgard", "Altgard Fortress Loop"): [
                {
                    "id": 12,
                    "routes": ["Altgard Fortress Loop"],
                    "gathered_count": 0,
                    "visit_count": 0,
                    "last_visited": 0.0,
                    "not_gatherable": False,
                }
            ],
            ("Verteron", "Verteron Citadel"): [
                {
                    "id": 7,
                    "routes": ["Verteron Citadel"],
                    "gathered_count": 0,
                    "visit_count": 0,
                    "last_visited": 0.0,
                    "not_gatherable": False,
                }
            ],
        }
        self.marked = []

    def get_regions(self):
        return ["Altgard", "Verteron"]

    def get_routes(self, region=None):
        if region == "Verteron":
            return ["Verteron Citadel"]
        return ["Dredgion Crash Site", "Altgard Fortress Loop"]

    def get_route_nodes(self, region_name=None, route_name=None):
        if route_name in (None, "All"):
            out = []
            for (reg, _rte), nodes in self._nodes.items():
                if reg == region_name:
                    out.extend(nodes)
            return list(out)
        return list(self._nodes.get((region_name, route_name), []))

    def get_nodes(self, region=None, route=None):
        return self.get_route_nodes(region_name=region, route_name=route)

    def mark_node_gathered(self, node_id, rest_minutes=5.0, force_deplete=False, region=None):
        for nodes in self._nodes.values():
            for node in nodes:
                if node["id"] == int(node_id):
                    node["gathered_count"] += 1
                    self.marked.append(int(node_id))
                    return node
        return None

    def delete_nodes(self, node_ids, region=None):
        removed = False
        for nodes in self._nodes.values():
            before = len(nodes)
            nodes[:] = [n for n in nodes if n["id"] not in set(node_ids)]
            removed = removed or len(nodes) != before
        return removed

    def delete_node(self, node_id, region=None):
        return self.delete_nodes([node_id], region=region)

    def remove_node_from_route(self, node_id, route_name, region=None):
        nodes = self._nodes.get((region, route_name), [])
        before = len(nodes)
        nodes[:] = [n for n in nodes if n["id"] != int(node_id)]
        return len(nodes) != before

    def add_node_to_route(self, node_id, route_name, region=None):
        nodes = self._nodes.setdefault((region, route_name), [])
        if any(n["id"] == int(node_id) for n in nodes):
            return False
        nodes.append(
            {
                "id": int(node_id),
                "routes": [route_name],
                "gathered_count": 0,
                "visit_count": 0,
                "last_visited": 0.0,
                "not_gatherable": False,
            }
        )
        return True

    def move_node_to_region(self, node_id, target_region, source_region=None):
        for (reg, rte), nodes in list(self._nodes.items()):
            for node in list(nodes):
                if node["id"] == int(node_id):
                    nodes.remove(node)
                    target_routes = self.get_routes(target_region)
                    first = target_routes[0] if target_routes else "All"
                    node["routes"] = [first]
                    self._nodes.setdefault((target_region, first), []).append(node)
                    return node
        return None


class FakeBlacklistManager:
    """Stand-in for core.blacklist_manager (its real save() writes the
    repo's config/settings.json + config/blacklist.json)."""

    def get_entries(self):
        return []

    def to_list(self):
        return []

    def save(self):
        return True

    def count(self):
        return 0

    def remove_entry(self, _eid):
        return None

    def clear_all(self):
        return None

    def add_entry(self, **_kwargs):
        return None


# Fixed settings seed (W2, fix/v4-pyqt6-W2) — the fixture used to copy the
# repo's config/settings.json, but that file is rewritten by earlier tests
# in a full run, so the copy inherited their region/route and all 5
# region/route-dependent tests below broke.
#
# (W3, fix/v5-W3-misc) — the old "(rowCount 2 -> 1)" gloss on that
# sentence was only partly right: 4 of the 5 break on rowCount 2 -> 1
# (smoke, double_click, remove_from_route, delete_node). The 5th,
# region_switch, breaks on the dirty flag instead — with the region
# already Verteron, switching to Verteron is a no-op and nothing marks
# the window dirty. test_point_filter is NOT one of the 5: under
# pollution it passes vacuously (the single Verteron row is hidden by
# the "#40" filter, and the row-1 assertion is vacuous because no row
# 1 exists), so it proves nothing about the pollution.
#
# Pollution chain, verified statically (no test run):
# 1. test_worker9_appshell_radar_hud.py:225 — shared shell built with the
#    DEFAULT SettingsManager, i.e. the repo's config/settings.json
#    (ui/app_shell.py:2099-2107 -> _settings factory at :2490-2504 ->
#    ui/settings_window.py:518 SettingsWindow(config_manager=None) ->
#    _create_default_config_manager() at :1067-1072 -> config/schema.py:650
#    SettingsManager(settings_path="config/settings.json")).
# 2. Its tests run in unittest alphabetical order, so the final shell state
#    is region=Verteron (test_region_dropdown_synchronization_to_map_and_
#    radar, after test_canonical_subregion_resolution set Altgard) and
#    route=IronLoop (test_route_dropdown_synchronization). Both propagate
#    into the shell's settings page combos (ui/app_shell.py:3094-3140
#    _propagate_region_selection and :3166-3210 _propagate_route_selection).
# 3. W5's guard (ui/app_shell.py:3801-3805, persist_settings no-ops when
#    the shell was never shown) does NOT hold here: the tab-switching test
#    calls shell.open_tool("map"/"radar"/"home"), and open_tool() calls
#    self.show() (ui/app_shell.py:2732), so showEvent sets _ever_shown=True
#    (:3814-3817). tearDownClass close() -> closeEvent (:3954/:3959) ->
#    persist_settings() -> settings page save_settings()
#    (ui/settings_window.py:3159) -> _save_to_settings (:3026) ->
#    _remember_selection (:3046-3065) writes bot.active_region=Verteron /
#    bot.active_route=IronLoop -> _cfg_save (:1296) -> SettingsManager.save()
#    (config/schema.py:734-752, export_payload includes active_region /
#    active_route at :517-518) into the repo's config/settings.json.
#    test_map_and_radar_suite.py:750-780 is a second instance of the same
#    hole (open_tool in every setUp, close in tearDownClass); its final
#    region/route happens to be the harmless default, so worker9 is the
#    effective polluter — but ANY future shown shell re-breaks a repo-copy
#    fixture, which is why the fixture is immunised instead of patching
#    one polluting test. No assertion was touched; the seed below is the
#    clean HEAD content of config/settings.json, verbatim.
#
#    MAINTENANCE (W3, fix/v5-W3-misc): the seed is a verbatim snapshot of
#    the HEAD defaults — if the HEAD config/settings.json defaults change
#    (new/renamed keys, different region/route), this seed silently goes
#    stale and the tests below keep passing against the OLD defaults.
#    When changing the defaults, re-copy the HEAD file content here.
_SEED_SETTINGS_JSON = """{
    "gathering": {
        "f_interval_min": 1.5,
        "f_interval_max": 3.0,
        "test_mode": false,
        "show_overlay": true,
        "harvests_per_node": 3,
        "inter_harvest_delay_min": 0.6,
        "inter_harvest_delay_max": 1.4,
        "retry_on_extraction_fail": true,
        "max_retries_per_harvest_stage": 2,
        "gather_key": "f",
        "selective_gather_enabled": false,
        "selected_materials": [],
        "proximity_route_auto": true,
        "pause_on_alt_tab": true,
        "auto_consumables": true
    },
    "route": {
        "rest_minutes": 5,
        "climb_air_max_sec": 10,
        "gathered_join_active_route": true,
        "fly_unreachable": true,
        "check_odyle_minimap": true,
        "reclick_node_sec": 10,
        "time_to_reach_sec": 58,
        "gather_on_the_way": false,
        "wait_min_after_failed": 2,
        "max_failed_in_row": 10,
        "remove_node_after_failures": 0,
        "pause_after_no_gather_min": 15,
        "fight_back": true,
        "fight_stop_gather_sec": 0,
        "harvests_per_node": 3,
        "minimap_auto_record": true,
        "minimap_ignore_monsters": true,
        "hybrid_3d_and_minimap_record": true,
        "auto_record_3d_crests": true,
        "mp_full_threshold": 0.95,
        "ucb_explore_ratio": 0.2,
        "skill_discovery_enabled": true,
        "pause_on_alt_tab": true,
        "auto_use_hp_potion": true,
        "hp_potion_threshold_pct": 65,
        "auto_use_mp_potion": true,
        "mp_potion_threshold_pct": 45,
        "auto_refresh_buff_scroll": true,
        "buff_scroll_interval_sec": 300
    },
    "breaks": {
        "break_every_min": 0,
        "break_every_variance": 1,
        "break_lasts_min": 10,
        "break_lasts_variance": 1,
        "enabled": true
    },
    "hotkeys": {
        "toggle_menu_f9": "ctrl+alt+f9",
        "record_node_f8": "ctrl+alt+f8",
        "pause_resume_f10": "ctrl+alt+f10"
    },
    "display": {
        "monitor_idx": 1,
        "last_tab": "home"
    },
    "active_region": "Altgard",
    "active_route": "Dredgion Crash Site",
    "version": "2.0.0",
    "reflex_model": "yolov8",
    "blacklist": []
}"""


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """SettingsWindow with a real SettingsManager on a temp settings.json."""
    monkeypatch.chdir(ROOT)
    # Isolate the blacklist manager: the real one syncs into the repo's
    # config/settings.json + config/blacklist.json on every save.
    monkeypatch.setattr(
        "core.blacklist_manager.get_blacklist_manager",
        lambda *a, **k: FakeBlacklistManager(),
    )
    settings_path = tmp_path / "settings.json"
    # Fixed seed, NOT a copy of the repo file (W2, fix/v4-pyqt6-W2):
    # the repo's config/settings.json is rewritten by earlier tests in a
    # full run (see _SEED_SETTINGS_JSON note above), so copying it
    # inherited their region/route. The seed below is the clean HEAD
    # content of config/settings.json, verbatim.
    settings_path.write_text(_SEED_SETTINGS_JSON, encoding="utf-8")
    mgr = SettingsManager(settings_path=settings_path)
    rm = FakeRouteManager()
    win = SettingsWindow(config_manager=mgr, route_manager=rm)
    win.show()
    qapp.processEvents()
    yield win, mgr, rm, settings_path
    win.close()


def _read_settings(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_smoke_instantiates_and_loads_real_regions(window):
    win, _mgr, _rm, _path = window
    assert win.f_interval_min.value() == pytest.approx(1.5)
    assert win.f_interval_max.value() == pytest.approx(3.0)
    assert win.gather_key.currentText() == "f"
    assert win.harvests_per_node.value() == 3
    assert win._current_region() == "Altgard"
    assert win._current_route() == "Dredgion Crash Site"
    assert win.table.rowCount() == 2
    assert win.version_lbl.text() == "Gatherion v2.0.0"
    assert "Dengeli" in win.infoline_lbl.text()
    assert "0.13" in win.infoline_lbl.text()
    assert "Ayarlar yüklendi" in win.log_text.toPlainText()
    # Prototype table columns: thumb+#id, routes, gathered, last, status.
    headers = [
        win.table.horizontalHeaderItem(c).text() for c in range(win.table.columnCount())
    ]
    assert headers == ["NOKTA", "ROTALAR", "TOPLANAN", "SON", "DURUM"]


def test_all_prototype_controls_exist_and_are_wired(window):
    win, _mgr, _rm, _path = window
    # Left form — every prototype field.
    # v11 (a39d624): climb_air_max_sec, check_odyle_minimap and
    # minimap_ignore_monsters widgets were removed from the settings
    # window — the schema keys still exist, the controls do not.
    for name in (
        "f_interval_min", "f_interval_max", "harvests_per_node", "gather_key",
        "retry_on_extraction_fail", "test_mode", "show_overlay",
        "rest_minutes", "gathered_join_active_route",
        "fly_unreachable", "reclick_node_sec",
        "time_to_reach_sec", "gather_on_the_way", "wait_min_after_failed",
        "max_failed_in_row", "remove_node_after_failures",
        "pause_after_no_gather_min", "fight_back", "fight_stop_gather_sec",
        "minimap_auto_record", "world_3d_auto_record",
        "break_every_min", "break_every_variance", "break_lasts_min",
        "break_lasts_variance",
    ):
        assert hasattr(win, name), f"missing control {name}"
    # Hotkey strip (capture edits).
    assert isinstance(win.hotkey_node, _HotkeyCaptureEdit)
    assert win.hotkey_node.text() == "ctrl+alt+f8"
    assert win.hotkey_menu.text() == "ctrl+alt+f9"
    assert win.hotkey_pause.text() == "ctrl+alt+f10"
    # Right panel actions.
    for name in (
        "btn_rename_region", "btn_delete_region", "btn_new_route",
        "btn_rename_route", "btn_delete_route", "btn_add_nodes",
        "btn_remove_from_route", "btn_delete_node", "btn_move_to_region",
        "btn_map", "btn_delete_selected", "btn_delete_wrong", "btn_profile",
        "btn_share_route", "btn_import_route", "btn_import_questlog",
        "point_filter_edit",
    ):
        assert hasattr(win, name), f"missing button {name}"
        assert getattr(win, name).isEnabled()
    # Footer bar.
    for name in (
        "btn_quit", "btn_radar", "btn_yolo_factory", "btn_mobile_connect",
        "btn_analytics", "btn_calibration", "btn_guide",
        "btn_reset_defaults", "btn_save", "btn_close",
        "dirty_lbl", "saved_lbl", "status_lbl", "infoline_lbl", "log_text",
    ):
        assert hasattr(win, name), f"missing footer control {name}"
    # No-selection action reports it instead of crashing.
    win._on_delete_node.__self__  # sanity: handler exists
    win.table.clearSelection()
    win._on_delete_node()
    assert "seçin" in win.status_lbl.text()


def test_save_writes_settings_json_and_emits_saved(window, qapp):
    win, mgr, _rm, path = window
    emitted = []
    win.settings_saved.connect(lambda: emitted.append(True))

    win.f_interval_min.setValue(2.0)
    win.f_interval_max.setValue(4.5)
    win.harvests_per_node.setValue(5)
    win.rest_minutes.setValue(7)
    win.break_every_min.setValue(45)
    win.break_lasts_min.setValue(12)
    win.gather_key.setCurrentIndex(win.gather_key.findData("e"))
    win.test_mode.setChecked(True)
    win.hotkey_pause.setText("ctrl+alt+f12")
    qapp.processEvents()

    assert win.save_settings() is True
    assert emitted == [True]
    data = _read_settings(path)
    assert data["gathering"]["f_interval_min"] == pytest.approx(2.0)
    assert data["gathering"]["f_interval_max"] == pytest.approx(4.5)
    assert data["gathering"]["harvests_per_node"] == 5
    assert data["gathering"]["gather_key"] == "e"
    assert data["gathering"]["test_mode"] is True
    assert data["route"]["rest_minutes"] == 7
    assert data["breaks"]["break_every_min"] == 45
    assert data["breaks"]["break_lasts_min"] == 12
    assert data["hotkeys"]["pause_resume_f10"] == "ctrl+alt+f12"
    # Reload through a fresh manager: the file really holds the values.
    assert SettingsManager(settings_path=path).settings.gathering.f_interval_min == pytest.approx(2.0)
    assert "kaydedildi" in win.status_lbl.text().lower()
    assert "kaydedildi ve uygulandı" in win.log_text.toPlainText()


def test_validation_blocks_bad_f_range_without_writing(window):
    win, _mgr, _rm, path = window
    before = _read_settings(path)
    win.f_interval_min.setValue(3.0)
    win.f_interval_max.setValue(1.5)
    assert win.save_settings() is False
    assert "üst sınır" in win.status_lbl.text()
    assert _read_settings(path) == before
    assert win.dirty_lbl.isVisible()  # still dirty: nothing was saved
    assert "Doğrulama hatası" in win.log_text.toPlainText()


def test_validation_blocks_non_f_hotkey(window):
    win, _mgr, _rm, path = window
    before = _read_settings(path)
    win.hotkey_node.setText("ctrl+alt+g")
    assert win.save_settings() is False
    assert "F tuşuyla" in win.status_lbl.text()
    assert _read_settings(path) == before


def test_dirty_tracking_show_and_clear_on_save(window, qapp):
    win, _mgr, _rm, _path = window
    assert not win.dirty_lbl.isVisible()
    win.rest_minutes.setValue(9)
    qapp.processEvents()
    assert win.dirty_lbl.isVisible()
    assert not win.saved_lbl.isVisible()
    assert win.save_settings() is True
    assert not win.dirty_lbl.isVisible()


def test_profile_button_cycles_dengeli_hassas_siki(window, qapp):
    win, _mgr, _rm, _path = window
    assert win.btn_profile.text() == "Profil: Dengeli"
    assert win._wrong_scan_tier() == "balanced"

    win.btn_profile.click()
    qapp.processEvents()
    assert win.btn_profile.text() == "Profil: Hassas"
    assert win._wrong_scan_tier() == "aggressive"
    assert win.cb_wrong_tier.currentData() == "aggressive"
    assert "0.10" in win.infoline_lbl.text()

    win.btn_profile.click()
    assert win.btn_profile.text() == "Profil: Sıkı"
    assert win._wrong_scan_tier() == "strict"
    assert "0.22" in win.infoline_lbl.text()

    win.btn_profile.click()
    assert win.btn_profile.text() == "Profil: Dengeli"
    assert win._wrong_scan_tier() == "balanced"


def test_double_click_marks_node_gathered(window, qapp):
    win, _mgr, rm, _path = window
    assert win.table.rowCount() == 2
    item = win.table.item(0, 0)
    win._on_node_double_clicked(item)
    qapp.processEvents()
    assert rm.marked == [245]
    assert rm._nodes[("Altgard", "Dredgion Crash Site")][0]["gathered_count"] == 1
    assert "toplandı" in win.status_lbl.text().lower()
    assert "#245" in win.status_lbl.text()
    assert win.table.rowCount() == 2  # table reloaded, same nodes


def test_hotkey_capture_rejects_bare_f8_and_escape_restores():
    edit = _HotkeyCaptureEdit("ctrl+alt+f8")
    edit.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))

    bare_f8 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F8, Qt.KeyboardModifier.NoModifier)
    edit.keyPressEvent(bare_f8)
    assert edit.text() == "ctrl+alt+f8"  # bare F8/F9/F10 never captured

    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    f9 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F9, mods)
    edit.keyPressEvent(f9)
    assert edit.text() == "ctrl+alt+f9"

    # Escape restores the chord that was there when capture started.
    edit.focusInEvent(QFocusEvent(QEvent.Type.FocusIn))
    edit.setText("ctrl+alt+f7")
    esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    edit.keyPressEvent(esc)
    assert edit.text() == "ctrl+alt+f9"


def test_remove_from_route_uses_route_manager(window, qapp):
    win, _mgr, rm, _path = window
    win.table.selectRow(1)
    qapp.processEvents()
    assert win._selected_node_ids() == [40]
    win._on_remove_from_route()
    qapp.processEvents()
    assert [n["id"] for n in rm._nodes[("Altgard", "Dredgion Crash Site")]] == [245]
    assert win.table.rowCount() == 1


def test_delete_node_uses_route_manager(window, qapp, monkeypatch):
    win, _mgr, rm, _path = window
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    win.table.selectRow(0)
    qapp.processEvents()
    win._on_delete_node()
    qapp.processEvents()
    assert [n["id"] for n in rm._nodes[("Altgard", "Dredgion Crash Site")]] == [40]
    assert win.table.rowCount() == 1


def test_region_switch_reloads_nodes_and_marks_dirty(window, qapp):
    win, _mgr, _rm, _path = window
    assert not win.dirty_lbl.isVisible()
    idx = win.region_combo.findData("Verteron")
    win.region_combo.setCurrentIndex(idx)
    qapp.processEvents()
    assert win._current_region() == "Verteron"
    assert win._current_route() == "Verteron Citadel"
    assert win.table.rowCount() == 1
    assert win.dirty_lbl.isVisible()


def test_point_filter_hides_non_matching_rows(window, qapp):
    win, _mgr, _rm, _path = window
    win.point_filter_edit.setText("#40")
    qapp.processEvents()
    assert win.table.isRowHidden(0) is True
    assert win.table.isRowHidden(1) is False
    win.point_filter_edit.setText("")
    qapp.processEvents()
    assert win.table.isRowHidden(0) is False


def test_log_strip_appends_and_caps_lines(window):
    win, _mgr, _rm, _path = window
    before = win.log_text.toPlainText()
    win.append_log("test satırı", "inf")
    assert "test satırı" in win.log_text.toPlainText()
    assert win.log_text.toPlainText().startswith(before.splitlines()[0][:20])
    assert win.log_text.maximumBlockCount() == 60
