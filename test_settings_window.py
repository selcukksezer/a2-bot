# -*- coding: utf-8 -*-
"""
Tests for SettingsWindow and Worker S3 Selective Gather Orchestration:
- GatheringConfig schema additions (selective_gather_enabled, selected_materials, proximity_route_auto).
- SettingsWindow UI controls for "🎯 Seç-Topla" and proximity route button.
- F9Menu UI controls and signal emission.
- main.py handle_selective_gather_changed wiring across BotWorker, RouteManager, MapViewer, Radar, and OverlayHUD.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

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
    def setUp(self):
        self.win = SettingsWindow()

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
        emitted = []
        self.win.selective_gather_changed.connect(lambda m, p: emitted.append((m, p)))
        self.win.set_selected_materials(["orichalcum"], auto_proximity=True)
        self.win._on_selective_gather_ui_changed()
        self.assertGreater(len(emitted), 0)
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
    def test_handle_selective_gather_changed(self):
        from main import GatherionOrchestrator

        # Create dummy orchestrator without launching threads
        orchestrator = GatherionOrchestrator.__new__(GatherionOrchestrator)
        orchestrator.settings = GatheringConfig()
        orchestrator.settings_mgr = MagicMock()
        orchestrator._sync_encrypted_vault = MagicMock()

        dummy_worker = MagicMock()
        dummy_routes = MagicMock()
        dummy_overlay = MagicMock()
        dummy_map_win = MagicMock()
        dummy_radar_win = MagicMock()

        orchestrator.worker = dummy_worker
        orchestrator.bot_worker = dummy_worker
        orchestrator.routes = dummy_routes
        orchestrator.route_mgr = dummy_routes
        orchestrator.overlay_hud = dummy_overlay
        orchestrator.overlay = dummy_overlay
        orchestrator.map_viewer_win = dummy_map_win
        orchestrator.radar_win = dummy_radar_win
        orchestrator.f9_menu = MagicMock()
        orchestrator.settings_win = MagicMock()

        # Call handler
        orchestrator.handle_selective_gather_changed(["orichalcum"], auto_proximity=True)

        # 1. BotWorker updated
        dummy_worker.set_selected_gather_materials.assert_called_with(["orichalcum"])
        self.assertEqual(dummy_worker.selected_materials, ["orichalcum"])
        self.assertTrue(dummy_worker.selective_gather_enabled)

        # 2. Route manager build_proximity_material_route called
        dummy_routes.build_proximity_material_route.assert_called_once()
        call_kwargs = dummy_routes.build_proximity_material_route.call_args[1]
        self.assertEqual(call_kwargs["target_materials"], ["orichalcum"])

        # 3. MapViewerWindow filter set
        dummy_map_win.set_material_filter.assert_called_with(["orichalcum"])

        # 4. Radar filter set
        dummy_radar_win.set_material_filter.assert_called_with(["orichalcum"])

        # 5. OverlayHUD updated
        dummy_overlay.add_log.assert_called()
        dummy_overlay.update_state.assert_called()


if __name__ == "__main__":
    unittest.main()
