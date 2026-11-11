# -*- coding: utf-8 -*-
"""
Tests for SettingsWindow and Worker S3 Selective Gather Orchestration:
- GatheringConfig schema additions (selective_gather_enabled, selected_materials, proximity_route_auto).
- SettingsWindow UI controls for "🎯 Seç-Topla" and proximity route button.
- F9Menu UI controls and signal emission.
- main.py handle_selective_gather_changed wiring across BotWorker, RouteManager, MapViewer, Radar, and OverlayHUD.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)


# Repo files the orchestrator test's collaborators rewrite as a side effect
# (BotWorker() rewrites dataset/data.yaml + dataset/manifest_stats.json; a
# default-region RouteManager re-saves routes/Verteron/*). The test snapshots
# and restores them so a test run never dirties the repo.
_REPO_SIDE_EFFECT_FILES = (
    "dataset/data.yaml",
    "dataset/manifest_stats.json",
    "routes/Verteron/region_data.json",
    "routes/Verteron/region_data.backup.json",
)


def _snapshot_repo_files():
    snap = {}
    for rel in _REPO_SIDE_EFFECT_FILES:
        p = Path(rel)
        snap[rel] = p.read_bytes() if p.is_file() else None
    return snap


def _restore_repo_files(snapshot):
    for rel, data in snapshot.items():
        p = Path(rel)
        if data is None:
            if p.exists():
                p.unlink()
        else:
            p.write_bytes(data)

from config.schema import GatheringConfig, GatheringSettings, SettingsManager
from ui.f9_menu import F9Menu
from ui.settings_window import SettingsWindow


class TestGatheringConfigSchema(unittest.TestCase):
    def test_schema_defaults(self):
        cfg = GatheringConfig()
        self.assertFalse(cfg.selective_gather_enabled)
        self.assertEqual(cfg.selected_materials, [])
        self.assertTrue(cfg.proximity_route_auto)

    def test_schema_validation(self):
        cfg = GatheringConfig(
            selective_gather_enabled=1,  # type: ignore
            selected_materials=["  orichalcum  ", "", "aria"],  # type: ignore
            proximity_route_auto=0,  # type: ignore
        )
        cfg.validate()
        self.assertTrue(cfg.selective_gather_enabled)
        self.assertEqual(cfg.selected_materials, ["orichalcum", "aria"])
        self.assertFalse(cfg.proximity_route_auto)

    def test_schema_from_dict(self):
        data = {
            "selective_gather_enabled": True,
            "selected_materials": ["mela", "targena"],
            "proximity_route_auto": False,
        }
        cfg = GatheringSettings.from_dict(data)
        self.assertTrue(cfg.selective_gather_enabled)
        self.assertEqual(cfg.selected_materials, ["mela", "targena"])
        self.assertFalse(cfg.proximity_route_auto)

    def test_gathering_config_alias(self):
        self.assertIs(GatheringConfig, GatheringSettings)


class TestF9MenuSelectiveGather(unittest.TestCase):
    def setUp(self):
        self.menu = F9Menu()

    def test_controls_exist(self):
        self.assertTrue(hasattr(self.menu, "selective_gather_frame"))
        self.assertTrue(hasattr(self.menu, "chk_selective_gather"))
        self.assertTrue(hasattr(self.menu, "combo_selective_materials"))
        self.assertTrue(hasattr(self.menu, "btn_proximity_route"))
        self.assertTrue(hasattr(self.menu, "selective_gather_changed"))

    def test_preset_items_present(self):
        combo = self.menu.combo_selective_materials
        texts = [combo.itemText(i) for i in range(combo.count())]
        self.assertTrue(any("Tümü" in t for t in texts))
        self.assertTrue(any("Orichalcum" in t for t in texts))
        self.assertTrue(any("Aria" in t for t in texts))
        self.assertTrue(any("Mela" in t for t in texts))
        self.assertTrue(any("Targena" in t for t in texts))
        self.assertTrue(any("Gem" in t or "Odyle" in t for t in texts))
        self.assertTrue(any("Kereste" in t or "Ağaç" in t for t in texts))

    def test_selection_emits_signal(self):
        emitted = []
        self.menu.selective_gather_changed.connect(lambda m, p: emitted.append((m, p)))
        # Merge-fix (O8): D33/F-0646 stopped the material combo from
        # auto-enabling selective gather (browsing the list used to turn it
        # on). Enable it explicitly first, then select a material.
        self.menu.chk_selective_gather.setChecked(True)
        # Select Orichalcum (index 1)
        self.menu.combo_selective_materials.setCurrentIndex(1)
        self.assertGreater(len(emitted), 0)
        materials, auto_prox = emitted[-1]
        self.assertEqual(materials, ["orichalcum"])
        self.assertTrue(auto_prox)
        self.assertTrue(self.menu.chk_selective_gather.isChecked())

    def test_proximity_button_click(self):
        emitted = []
        self.menu.selective_gather_changed.connect(lambda m, p: emitted.append((m, p)))
        self.menu.btn_proximity_route.click()
        self.assertGreater(len(emitted), 0)
        _, auto_prox = emitted[-1]
        self.assertTrue(auto_prox)


class TestSettingsWindowSelectiveGather(unittest.TestCase):
    # One shared window: constructing a SettingsWindow costs ~3.5 s
    # (measured 3.3–6.2 s per construction, 3 runs) and the four tests
    # below only read controls or drive them through their own listeners.
    # setUp() resets the mutable UI state so every test starts from the
    # same baseline a fresh window would have.
    @classmethod
    def setUpClass(cls):
        cls.win = SettingsWindow()

    @classmethod
    def tearDownClass(cls):
        cls.win.deleteLater()
        cls.win = None

    def setUp(self):
        self.win = self.__class__.win
        self.win.chk_selective_gather.setChecked(False)
        self.win.chk_proximity_route_auto.setChecked(True)
        self.win.combo_selective_materials.setCurrentIndex(0)

    def test_controls_exist(self):
        self.assertTrue(hasattr(self.win, "chk_selective_gather"))
        self.assertTrue(hasattr(self.win, "combo_selective_materials"))
        self.assertTrue(hasattr(self.win, "chk_proximity_route_auto"))
        self.assertTrue(hasattr(self.win, "btn_proximity_route"))
        self.assertTrue(hasattr(self.win, "selective_gather_changed"))

    def test_get_settings_dict_includes_fields(self):
        d = self.win.get_settings_dict()
        self.assertIn("selective_gather_enabled", d)
        self.assertIn("selected_materials", d)
        self.assertIn("proximity_route_auto", d)

    def test_signal_emission(self):
        """F-1433: the signal must come from real UI interaction, not from
        calling the handler directly."""
        emitted = []
        self.win.selective_gather_changed.connect(lambda m, p: emitted.append((m, p)))
        # Set the state programmatically, then drive the real controls:
        # combo index change -> currentIndexChanged -> handler -> signal.
        self.win.set_selected_materials(["orichalcum"], auto_proximity=True)
        self.assertEqual(self.win.get_selected_materials(), ["orichalcum"])
        self.assertTrue(self.win.chk_selective_gather.isChecked())
        self.win.combo_selective_materials.setCurrentIndex(0)  # "Tümü"
        self.win.combo_selective_materials.setCurrentIndex(1)  # Orichalcum
        self.assertGreaterEqual(len(emitted), 2)
        self.assertEqual(emitted[-2], ([], True))
        materials, auto_prox = emitted[-1]
        self.assertEqual(materials, ["orichalcum"])
        self.assertTrue(auto_prox)

    def test_btn_proximity_route_click(self):
        emitted = []
        self.win.selective_gather_changed.connect(lambda m, p: emitted.append((m, p)))
        self.win.btn_proximity_route.click()
        self.assertGreater(len(emitted), 0)
        _, auto_prox = emitted[-1]
        self.assertTrue(auto_prox)


class TestMainOrchestratorSelectiveGatherWiring(unittest.TestCase):
    def _assert_repo_files_unchanged(self, snapshot):
        for rel, data in snapshot.items():
            p = Path(rel)
            if data is None:
                self.assertFalse(p.exists(), f"test run created repo file: {rel}")
            else:
                self.assertEqual(p.read_bytes(), data, f"test run dirtied repo file: {rel}")

    def test_handle_selective_gather_changed(self):
        """F-1434: wiring is verified against the real collaborators, not
        MagicMock stand-ins. State is asserted on the real objects.

        NOTE: main.py imports winsound unconditionally (line 83), which does
        not exist on Linux; the test-only sys.modules shim below works around
        it. Making that import platform-safe is a production fix for D4.
        """
        # Hermeticity: BotWorker() etc. rewrite repo files as a side effect
        # (see _REPO_SIDE_EFFECT_FILES). Snapshot + restore them so the
        # test run leaves `git status` clean, and assert that it did.
        # NOTE: cleanups run LIFO, so the assert is registered FIRST and
        # the restore second — restore runs before the assert.
        repo_snapshot = _snapshot_repo_files()
        self.addCleanup(self._assert_repo_files_unchanged, repo_snapshot)
        self.addCleanup(_restore_repo_files, repo_snapshot)

        winsound_stub = types.ModuleType("winsound")
        winsound_stub.Beep = lambda *args, **kwargs: None
        winsound_patcher = mock.patch.dict(sys.modules, {"winsound": winsound_stub})
        winsound_patcher.start()
        self.addCleanup(winsound_patcher.stop)

        from main import GatherionOrchestrator

        from core.route_manager import RouteManager
        from core.state_machine import BotWorker
        from ui.cartography_radar import CartographyRadarDialog
        from ui.map_viewer import MapViewerWindow
        from ui.overlay_hud import OverlayHUD

        with tempfile.TemporaryDirectory() as tmp:
            # Real RouteManager on a throwaway copy of the Altgard region so
            # the persisted proximity route never touches the repo.
            tmp_routes = Path(tmp) / "routes"
            shutil.copytree(Path("routes") / "Altgard", tmp_routes / "Altgard")
            route_mgr = RouteManager(base_dir=str(tmp_routes))
            route_mgr.active_region = "Altgard"

            # Create dummy orchestrator without launching threads
            orchestrator = GatherionOrchestrator.__new__(GatherionOrchestrator)
            orchestrator.settings = GatheringConfig()
            orchestrator.settings_mgr = SettingsManager(
                settings_path=str(Path(tmp) / "settings.json")
            )
            vault_calls = []
            orchestrator._sync_encrypted_vault = lambda: vault_calls.append(1)

            worker = BotWorker()
            map_win = MapViewerWindow(route_manager=route_mgr)
            radar_win = CartographyRadarDialog()
            overlay = OverlayHUD()
            f9_menu = F9Menu()
            settings_win = SettingsWindow()

            orchestrator.worker = worker
            orchestrator.bot_worker = worker
            orchestrator.routes = route_mgr
            orchestrator.route_mgr = route_mgr
            orchestrator.overlay_hud = overlay
            orchestrator.overlay = overlay
            orchestrator.map_viewer_win = map_win
            orchestrator.radar_win = radar_win
            orchestrator.f9_menu = f9_menu
            orchestrator.settings_win = settings_win

            # Call handler
            orchestrator.handle_selective_gather_changed(["orichalcum"], auto_proximity=True)

            # 1. Real BotWorker updated
            self.assertEqual(worker.selected_materials, ["orichalcum"])
            self.assertEqual(worker.selected_gather_materials, ["orichalcum"])
            self.assertTrue(worker.selective_gather_enabled)

            # 2. Real RouteManager built a proximity route for orichalcum
            # (nodes tagged + persisted into the throwaway region file).
            tagged = [
                n
                for n in route_mgr.get_nodes("Altgard")
                if "proximity_ore_orichalcum" in list(getattr(n, "routes", []) or [])
            ]
            self.assertGreater(len(tagged), 0)
            persisted = (tmp_routes / "Altgard" / "region_data.json").read_text(encoding="utf-8")
            self.assertIn("proximity_ore_orichalcum", persisted)

            # 3. Real MapViewerWindow filter set
            self.assertEqual(map_win.get_material_filter(), "orichalcum")

            # 4. Real Radar filter set
            self.assertEqual(radar_win.get_material_filter(), "orichalcum")

            # 5. Real OverlayHUD updated
            self.assertTrue(
                any("Seç-Topla: orichalcum" in line for line in overlay.logs)
            )

            # 6. Real F9Menu and SettingsWindow kept in sync
            self.assertEqual(f9_menu.get_selected_materials(), ["orichalcum"])
            self.assertEqual(settings_win.get_selected_materials(), ["orichalcum"])


if __name__ == "__main__":
    unittest.main()
