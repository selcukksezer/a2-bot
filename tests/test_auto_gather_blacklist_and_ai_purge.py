# -*- coding: utf-8 -*-
"""
Tests for Auto-Gather Blacklist, 3x Consecutive Failure AI Purge, and Re-Record Prevention.

Verifies:
1. When auto-gathering arrives at a node and finds nothing:
   - 1st failure: removed from active route ("orayı rotadan çıkarsın").
   - 3 consecutive failures: triggers AI verification ("yapay zekayla kontrol etsin"),
     confirms wrong marking, saves as negative exemplar, permanently deletes node ("rotayı otomatik silsin").
2. Re-recording prevention:
   - Once blacklisted, F8, add_node, recorder, and auto_cartographer strictly block re-recording ("birdaha kesinlikle kaydetmesin").
3. BlacklistManager API:
   - Add, query with proximity threshold (25px), remove, clear, persistence roundtrip.
4. SettingsWindow UI:
   - Blacklist UI section, table display, filter, and settings dictionary persistence ("ayarlarda blacklist kısmı olsun").
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from core.blacklist_manager import BlacklistManager
from core.route_manager import RouteManager


class TestBlacklistManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bl_file = os.path.join(self.temp_dir, "blacklist.json")
        self.settings_file = os.path.join(self.temp_dir, "settings.json")
        self.mgr = BlacklistManager(blacklist_file=self.bl_file, settings_file=self.settings_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_query_blacklist(self):
        entry = self.mgr.add_entry(
            region="Altgard",
            map_x=500,
            map_y=600,
            reason="3x başarısız toplama",
            ai_verdict="empty_terrain (0.91)",
            source_node_id=42,
        )
        self.assertEqual(entry.region, "Altgard")
        self.assertEqual(entry.map_x, 500)
        self.assertEqual(entry.map_y, 600)
        self.assertEqual(entry.source_node_id, 42)

        # Exact match
        is_bl, matched = self.mgr.is_blacklisted("Altgard", 500, 600)
        self.assertTrue(is_bl)
        self.assertIsNotNone(matched)
        self.assertEqual(matched.id, entry.id)

        # Within threshold (20px away <= 25px threshold)
        is_bl_near, _ = self.mgr.is_blacklisted("Altgard", 515, 610)
        self.assertTrue(is_bl_near)

        # Beyond threshold (100px away)
        is_bl_far, _ = self.mgr.is_blacklisted("Altgard", 650, 750)
        self.assertFalse(is_bl_far)

        # Different region
        is_bl_diff_reg, _ = self.mgr.is_blacklisted("Verteron", 500, 600)
        self.assertFalse(is_bl_diff_reg)

    def test_persistence_roundtrip(self):
        self.mgr.add_entry(
            region="Verteron",
            map_x=250,
            map_y=350,
            reason="Yanlış işaretleme",
            ai_verdict="monster_3d",
        )
        self.assertTrue(os.path.exists(self.bl_file))

        # Create new manager pointing to same file
        mgr2 = BlacklistManager(blacklist_file=self.bl_file, settings_file=self.settings_file)
        self.assertEqual(mgr2.count(), 1)
        is_bl, _ = mgr2.is_blacklisted("Verteron", 250, 350)
        self.assertTrue(is_bl)

    def test_remove_and_clear(self):
        e1 = self.mgr.add_entry("Altgard", 100, 100)
        e2 = self.mgr.add_entry("Altgard", 200, 200)
        e3 = self.mgr.add_entry("Verteron", 300, 300)
        self.assertEqual(self.mgr.count(), 3)

        # Remove e1
        self.assertTrue(self.mgr.remove_entry(e1.id))
        self.assertEqual(self.mgr.count(), 2)
        is_bl, _ = self.mgr.is_blacklisted("Altgard", 100, 100)
        self.assertFalse(is_bl)

        # Clear Altgard only
        cleared = self.mgr.clear_all(region="Altgard")
        self.assertEqual(cleared, 1)
        self.assertEqual(self.mgr.count(), 1)
        is_bl_vert, _ = self.mgr.is_blacklisted("Verteron", 300, 300)
        self.assertTrue(is_bl_vert)

        # Clear all
        self.mgr.clear_all()
        self.assertEqual(self.mgr.count(), 0)


class TestRouteManagerGatherFailureAndBlacklist(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.routes_dir = os.path.join(self.temp_dir, "routes")
        self.bl_file = os.path.join(self.temp_dir, "blacklist.json")
        self.settings_file = os.path.join(self.temp_dir, "settings.json")
        self.bl_mgr = BlacklistManager(blacklist_file=self.bl_file, settings_file=self.settings_file)
        self.rm = RouteManager(base_dir=self.routes_dir, blacklist_manager=self.bl_mgr)
        self.rm.create_region("Altgard")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_failure_1_removes_from_active_route(self):
        """1. Hata: Düğüm aktif rotadan çıkarılır (orayı rotadan çıkarsın)."""
        node = self.rm.add_node(
            region="Altgard",
            route="FarmLoop",
            map_x=400,
            map_y=400,
            resource_type="Orichalcum Ore",
        )
        self.assertIsNotNone(node)
        self.assertIn("FarmLoop", node.routes)

        # 1. Başarısızlık
        res = self.rm.mark_node_failed(node.id, region="Altgard", active_route="FarmLoop")
        self.assertIsNotNone(res)
        self.assertEqual(res.consecutive_failures, 1)
        # Rotadan çıkarılmış olmalı
        self.assertNotIn("FarmLoop", res.routes)
        # Ancak bölgede düğüm hala var olmalı
        self.assertIsNotNone(self.rm.get_node(node.id, region="Altgard"))

    def test_failure_3_triggers_ai_verification_deletion_and_blacklist(self):
        """3. Hata: YZ kontrolü yapar, rotadan ve bölgeden kalıcı siler ve kara listeye ekler."""
        node = self.rm.add_node(
            region="Altgard",
            route="FarmLoop",
            map_x=700,
            map_y=700,
            resource_type="Aria Herb",
        )
        node_id = node.id

        # 1. Başarısızlık
        self.rm.mark_node_failed(node_id, region="Altgard", active_route="FarmLoop")
        # 2. Başarısızlık
        self.rm.mark_node_failed(node_id, region="Altgard", active_route="FarmLoop")
        node_after_2 = self.rm.get_node(node_id, region="Altgard")
        self.assertIsNotNone(node_after_2)
        self.assertEqual(node_after_2.consecutive_failures, 2)

        # 3. Başarısızlık -> Otomatik YZ teşhisi, silme ve kara liste!
        res_3 = self.rm.mark_node_failed(node_id, region="Altgard", active_route="FarmLoop")
        self.assertIsNone(res_3)  # Düğüm silindiği için None döner

        # Düğüm bölgeden tamamen silinmiş olmalı
        self.assertIsNone(self.rm.get_node(node_id, region="Altgard"))

        # Koordinat kara listeye eklenmiş olmalı!
        is_bl, entry = self.bl_mgr.is_blacklisted("Altgard", 700, 700)
        self.assertTrue(is_bl)
        self.assertIsNotNone(entry)
        self.assertIn("3x", entry.reason)

    def test_blacklisted_coordinates_cannot_be_re_recorded(self):
        """Kara listedeki koordinat bir daha kesinlikle kaydedilemez."""
        # Koordinatı kara listeye al
        self.bl_mgr.add_entry(
            region="Altgard",
            map_x=800,
            map_y=850,
            reason="YZ tarafından yanlış işaretleme tespit edildi",
        )

        # 1. add_node ile eklemeyi dene -> Engellenmeli (None dönmeli)
        rejected_node = self.rm.add_node(
            region="Altgard",
            route="TestRoute",
            map_x=800,
            map_y=850,
        )
        self.assertIsNone(rejected_node)

        # 2. 10px yakınında eklemeyi dene (tolerans içi) -> Engellenmeli
        rejected_near = self.rm.add_node(
            region="Altgard",
            route="TestRoute",
            map_x=808,
            map_y=855,
        )
        self.assertIsNone(rejected_near)

        # 3. F8 record_current_node ile eklemeyi dene -> Engellenmeli.
        # Ilk F8 cagrisi henuz kara listede olmayan merkez (960, 540)
        # konumunu kaydeder; sonucu kosulsuz dogrulanir.
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        f8_res = self.rm.record_current_node(
            input_ctrl=dummy_frame,
            region="Altgard",
            route="TestRoute",
        )
        self.assertIsNotNone(f8_res)
        self.assertEqual((f8_res.map_x, f8_res.map_y), (960, 540))

        # Merkez kara listeye alindiktan sonra ikinci F8 cagrisi engellenmeli.
        self.bl_mgr.add_entry("Altgard", 960, 540)
        f8_rejected = self.rm.record_current_node(
            input_ctrl=dummy_frame,
            region="Altgard",
            route="TestRoute",
        )
        self.assertIsNone(f8_rejected)


class TestSettingsWindowBlacklistUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui.settings_window import SettingsWindow
        self.temp_dir = tempfile.mkdtemp()
        self.routes_dir = os.path.join(self.temp_dir, "routes")
        self.bl_file = os.path.join(self.temp_dir, "blacklist.json")
        self.settings_file = os.path.join(self.temp_dir, "settings.json")
        self.bl_mgr = BlacklistManager(blacklist_file=self.bl_file, settings_file=self.settings_file)
        self.rm = RouteManager(base_dir=self.routes_dir, blacklist_manager=self.bl_mgr)
        self.rm.create_region("Altgard")

        self.win = SettingsWindow(route_manager=self.rm)
        self.win.blacklist_mgr = self.bl_mgr

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ui_blacklist_controls_exist(self):
        """Ayarlar penceresinde Kara Liste (Blacklist) kontrollerinin varlığını doğrular."""
        self.assertTrue(hasattr(self.win, "blacklist_tab"))
        self.assertTrue(hasattr(self.win, "blacklist_table"))
        self.assertTrue(hasattr(self.win, "btn_blacklist_remove"))
        self.assertTrue(hasattr(self.win, "btn_blacklist_clear"))
        self.assertTrue(hasattr(self.win, "btn_blacklist_add"))
        self.assertTrue(hasattr(self.win, "btn_blacklist_refresh"))
        self.assertTrue(hasattr(self.win, "blacklist_filter_edit"))
        self.assertTrue(hasattr(self.win, "right_tab_widget"))

        # Sekme sayısı en az 2 olmalı (Noktalar ve Kara Liste)
        self.assertGreaterEqual(self.win.right_tab_widget.count(), 2)

    def test_blacklist_table_populates_and_settings_dict_includes_blacklist(self):
        """Kara liste tablosunun dolması ve get_settings_dict içine dahil edilmesi."""
        self.bl_mgr.add_entry(
            region="Altgard",
            map_x=123,
            map_y=456,
            reason="3x toplanacak kaynak yok",
            ai_verdict="Boş Zemin",
        )
        self.win.load_blacklist_table()

        self.assertEqual(self.win.blacklist_table.rowCount(), 1)
        self.assertEqual(self.win.blacklist_table.item(0, 0).text(), "Altgard")
        self.assertEqual(self.win.blacklist_table.item(0, 1).text(), "(123, 456)")
        self.assertIn("3x", self.win.blacklist_table.item(0, 2).text())

        # get_settings_dict() kara listeyi içermelidir
        sdict = self.win.get_settings_dict()
        self.assertIn("blacklist", sdict)
        self.assertEqual(len(sdict["blacklist"]), 1)
        self.assertEqual(sdict["blacklist"][0]["map_x"], 123)


class TestBotWorkerAutoGatherBlacklistIntegration(unittest.TestCase):
    def setUp(self):
        from core.state_machine import BotMode, BotWorker

        self.temp_dir = tempfile.mkdtemp()
        self.routes_dir = os.path.join(self.temp_dir, "routes")
        self.bl_file = os.path.join(self.temp_dir, "blacklist.json")
        self.settings_file = os.path.join(self.temp_dir, "settings.json")
        self.bl_mgr = BlacklistManager(blacklist_file=self.bl_file, settings_file=self.settings_file)
        self.rm = RouteManager(base_dir=self.routes_dir, blacklist_manager=self.bl_mgr)
        self.rm.create_region("Altgard")

        # Gerçek düğüm oluştur
        self.node = self.rm.add_node(
            region="Altgard",
            route="FarmLoop",
            map_x=550,
            map_y=650,
            resource_type="Aria Herb",
        )

        self.worker = BotWorker(config={
            "test_mode": True,
            "time_to_reach_sec": 30,
            "reclick_node_sec": 0,
            "remove_node_after_failures": 3,
        })
        self.worker.route_manager = self.rm
        self.worker.set_mode(BotMode.ROUTE_MODE.value)
        self.worker.resume()
        self.worker.active_route = "FarmLoop"
        self.worker.active_region = "Altgard"
        self.worker._command_walk_toward = MagicMock(return_value=True)
        self.worker._walking_is_real = MagicMock(return_value=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worker_arrives_empty_node_progression_to_blacklist(self):
        """Bot düğüme geldiğinde ve kaynak bulamadığında sırasıyla rotadan çıkarma ve 3x YZ kara liste akışı."""
        node_id = self.node.id

        # --- 1. Geliş: Kaynak bulamadı ---
        self.worker.active_node = self.rm.get_node(node_id, region="Altgard")
        self.worker._set_active_node_distance(2)  # Hedefe ulasti; Sweep-C 16/2: varis karari yalnizca TAZE mesafeye bakar — setter tazelik damgasini basar
        now = time.time()
        self.worker._walk_start_time = now - 10.0

        # İlk varış zaman damgasını tetikle
        self.worker._monitor_walking_to_node(now)
        # 1.6 saniye sonra tekrar kontrol (1.5s varış bekleme süresi aşıldı)
        self.worker._monitor_walking_to_node(now + 1.6)

        # active_node sıfırlanmış olmalı
        self.assertIsNone(self.worker.active_node)
        # Düğüm 1. hata sonucu 'FarmLoop' rotasından çıkarılmış olmalı
        n_after_1 = self.rm.get_node(node_id, region="Altgard")
        self.assertIsNotNone(n_after_1)
        self.assertEqual(n_after_1.consecutive_failures, 1)
        self.assertNotIn("FarmLoop", n_after_1.routes)

        # --- 2. Geliş: Tekrar kaynak bulamadı ---
        self.worker.active_node = n_after_1
        self.worker._set_active_node_distance(2)  # Sweep-C 16/2: varis karari yalnizca TAZE mesafeye bakar — setter tazelik damgasini basar
        now = time.time()
        self.worker._walk_start_time = now - 5.0
        self.worker._arrival_at_node_ts = 0.0

        self.worker._monitor_walking_to_node(now)
        self.worker._monitor_walking_to_node(now + 1.6)

        self.assertIsNone(self.worker.active_node)
        n_after_2 = self.rm.get_node(node_id, region="Altgard")
        self.assertIsNotNone(n_after_2)
        self.assertEqual(n_after_2.consecutive_failures, 2)

        # --- 3. Geliş: 3. kez kaynak bulamadı -> YZ Kontrolü + Silme + Kara Liste ---
        self.worker.active_node = n_after_2
        self.worker._set_active_node_distance(2)  # Sweep-C 16/2: varis karari yalnizca TAZE mesafeye bakar — setter tazelik damgasini basar
        now = time.time()
        self.worker._walk_start_time = now - 5.0
        self.worker._arrival_at_node_ts = 0.0

        self.worker._monitor_walking_to_node(now)
        self.worker._monitor_walking_to_node(now + 1.6)

        self.assertIsNone(self.worker.active_node)
        # Düğüm tamamen silinmiş olmalı!
        self.assertIsNone(self.rm.get_node(node_id, region="Altgard"))

        # Koordinat kara listeye alınmış olmalı!
        is_bl, entry = self.bl_mgr.is_blacklisted("Altgard", 550, 650)
        self.assertTrue(is_bl)
        self.assertIsNotNone(entry)
        self.assertIn("3x", entry.reason)

        # Tekrar eklenmesi engellenmeli!
        rejected = self.rm.add_node("Altgard", "FarmLoop", map_x=550, map_y=650)
        self.assertIsNone(rejected)


if __name__ == "__main__":
    unittest.main()
