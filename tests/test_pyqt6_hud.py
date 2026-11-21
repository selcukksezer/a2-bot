# -*- coding: utf-8 -*-
"""
tests/test_pyqt6_hud.py — W5 (HUD) prototip uyumu testleri.

Kapsam (gatherion-hud-prototipi.html -> ui/overlay_hud.py):
- State machine'in gerçek mod rozet adları tanınır ("TOPLAMA (ROUTE)",
  "SAVAŞ (COMBAT)"); gerçekten bilinmeyen mod uyarılıp yoksayılır.
- Prototip durumları: DİNLENİYOR (RESTING), ÇALIŞIYOR (RUNNING).
- Rota ilerlemesi / hedef / sistem satırları canlı kaynaktan beslenir;
  kaynak yokken "—" gösterilir, sahte sayı üretilmez.
- Kontrol düğmeleri prototipteki gibi etkin/pasif olur; mod düğmeleri
  aktif modu vurgular; Harita düğmesi F9 menüsüne gider.
- HUD AYARLARI paneli: Mini harita + Tespit kutuları canlı bağlı;
  canlı kaynağı olmayan üç anahtar devre dışıdır (sahte anahtar yok).
- Sahte/demo veri yok: taze HUD'da rota/hedef/sistem "—" gösterir.

Çalıştırma: QT_QPA_PLATFORM=offscreen <venv>/bin/python -m pytest tests/test_pyqt6_hud.py -v
"""

import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from ui.overlay_hud import OverlayHUD, validate_overlay_hud_contract


class _FakeNode:
    def __init__(self, nid, name):
        self.id = nid
        self.name = name


class _FakeRouteManager:
    active_region = "Altgard"
    active_route = "All"
    data = {"active_region": "Altgard", "active_route": "All"}

    def get_route_nodes(self):
        return [_FakeNode(i, "ore_%d" % i) for i in (11, 12, 13, 14, 15)]


class _FakeWorker:
    def __init__(self, node=None, distance_m=None):
        self.active_node = node
        self.active_node_distance_m = distance_m


class TestPyQt6HudPrototype(unittest.TestCase):
    def setUp(self):
        self.hud = OverlayHUD()
        self.hud.show()
        app.processEvents()

    def tearDown(self):
        self.hud.close()
        app.processEvents()

    # -- Kurulum / sözleşme -------------------------------------------------

    def test_hud_ready_and_contract(self):
        self.assertTrue(self.hud._hud_is_ready())
        self.assertTrue(validate_overlay_hud_contract())
        for name in (
            "update_route_progress",
            "update_target",
            "update_system_status",
            "set_detection_boxes_visible",
            "set_minimap_visible",
            "reset_position",
            "bind_system_monitor",
            "bind_live_sources",
        ):
            self.assertTrue(callable(getattr(OverlayHUD, name, None)), name)

    def test_fresh_hud_shows_dashes_not_fake_numbers(self):
        """Kaynak yokken rota/hedef/sistem boş/'—' gösterir, sayı uydurmaz."""
        self.assertIn("—", self.hud.route_label.text())
        self.assertEqual(self.hud.target_label.text(), "Hedef: —")
        self.assertEqual(self.hud.system_label.text(), "FPS — · GPU — · CPU —")
        self.assertEqual(self.hud.route_progress_bar.value(), 0)

    # -- Mod adları (state machine MODE_BADGE_MAP) ---------------------------

    def test_real_mode_badge_names_recognized(self):
        self.hud.set_mode("TOPLAMA (ROUTE)")
        self.assertEqual(self.hud.mode_label.text(), "TOPLAMA (ROUTE)")
        self.assertEqual(self.hud._current_mode_key, "gather")
        self.hud.set_mode("SAVAŞ (COMBAT)")
        self.assertEqual(self.hud.mode_label.text(), "SAVAŞ (COMBAT)")
        self.assertEqual(self.hud._current_mode_key, "combat")
        # Prototip combat chip'i "SAVAŞ (ROUTE)" yazar; gerçek rozete çevrilir.
        self.hud.set_mode("SAVAŞ (ROUTE)")
        self.assertEqual(self.hud.mode_label.text(), "SAVAŞ (COMBAT)")
        for raw, expected in (
            ("route", "ROTA MODU"),
            ("combat", "SAVAŞ (COMBAT)"),
            ("MANUAL RECORD", "MANUEL KAYIT"),
            ("AUTO RECORD", "OTO KAYIT"),
            ("AUTO CARTOGRAPHY", "OTO KEŞİF"),
        ):
            with self.subTest(raw=raw):
                self.hud.set_mode(raw)
                self.assertEqual(self.hud.mode_label.text(), expected)

    def test_unknown_mode_ignored_and_kept(self):
        self.hud.set_mode("TOPLAMA (ROUTE)")
        self.hud.set_mode("BOGUS_MODE_XYZ")
        self.assertEqual(self.hud.mode_label.text(), "TOPLAMA (ROUTE)")

    def test_worker_snapshot_mode_badge_flows_through(self):
        """Worker'ın _emit_hud_snapshot gönderdiği mode_badge olduğu gibi alınır."""
        self.hud.update_state({"state": "GATHERING", "mode_badge": "TOPLAMA (ROUTE)"})
        self.assertEqual(self.hud.mode_label.text(), "TOPLAMA (ROUTE)")
        self.assertEqual(self.hud.badge_label.text(), "TOPLANIYOR")
        self.hud.update_state({"state": "FIGHTING", "mode_badge": "SAVAŞ (COMBAT)"})
        self.assertEqual(self.hud.mode_label.text(), "SAVAŞ (COMBAT)")
        self.assertEqual(self.hud.badge_label.text(), "SAVAŞTA")

    # -- Durum rozetleri ------------------------------------------------------

    def test_prototype_states(self):
        self.hud.update_state({"state": "RESTING"})
        self.assertEqual(self.hud.badge_label.text(), "DİNLENİYOR")
        self.hud.update_state({"state": "RUNNING"})
        self.assertEqual(self.hud.badge_label.text(), "ÇALIŞIYOR")
        self.hud.update_state({"state": "PAUSED"})
        self.assertEqual(self.hud.badge_label.text(), "DURAKLATILDI")

    def test_worker_snapshot_columns_stay_live(self):
        self.hud.update_state({
            "state": "GATHERING", "mode_badge": "TOPLAMA (ROUTE)",
            "col1_title": "GATHERED", "col1_val": "7",
            "col2_title": "ALTGARD · ALL", "col2_val": "5 ready · 2 resting",
            "col3_title": "FAILED IN A ROW", "col3_val": "1 / 10",
            "learned_frames": 1234, "harvest_stage": 2,
            "hp_pct": 82.0, "mp_pct": 64.0,
        })
        app.processEvents()
        self.assertEqual(self.hud.stat1_value.text(), "7")
        self.assertEqual(self.hud.stat2_value.text(), "5 aktif · 2 dinlenen")
        self.assertEqual(self.hud.stat3_value.text(), "1 / 10")
        self.assertEqual(self.hud.ai_learned_label.text(), "YZ 1234 Kare")
        self.assertEqual(self.hud.harvest_badge_label.text(), "3x Toplama (Harvest: 2/3)")
        self.assertEqual(self.hud.consumable_badge_label.text(), "HP 82% · MP 64% · S:0")

    # -- Rota ilerlemesi / hedef (canlı kaynak) ------------------------------

    def test_route_progress_from_live_sources(self):
        worker = _FakeWorker(node=_FakeNode(13, "ore_orichalcum"), distance_m=120)
        self.hud.bind_live_sources(route_manager=_FakeRouteManager(), worker=worker)
        app.processEvents()
        # 13, listedeki 3. düğüm (11,12,13,14,15) -> Nokta 3/5; tur sayacı yok -> "—"
        self.assertEqual(
            self.hud.route_label.text(), "Rota: Altgard · Nokta 3/5 · Tur —"
        )
        self.assertEqual(self.hud.route_progress_bar.maximum(), 5)
        self.assertEqual(self.hud.route_progress_bar.value(), 3)
        self.assertEqual(
            self.hud.target_label.text(), "Hedef: Nokta 13 · ore_orichalcum · 120 m"
        )

    def test_route_progress_explicit_keys_without_sources(self):
        self.hud.update_state({
            "state": "WALKING",
            "route_current": 2, "route_total": 12, "route_loops": 1,
            "zone": "Dredgion",
        })
        self.assertEqual(
            self.hud.route_label.text(), "Rota: Dredgion · Nokta 2/12 · Tur 1"
        )
        self.assertEqual(self.hud.route_progress_bar.value(), 2)

    def test_target_explicit_keys(self):
        self.hud.update_target(node_id=4, distance_m=87, material="herb_aria")
        self.assertEqual(
            self.hud.target_label.text(), "Hedef: Nokta 4 · herb_aria · 87 m"
        )

    # -- Sistem durumu --------------------------------------------------------

    def test_system_status_explicit_values(self):
        self.hud.update_system_status(
            fps=29.5, gpu_pct=61, cpu_pct=23,
            gpu_name="GPU · RTX 3070", gpu_temp_c=58,
            cpu_name="CPU · 8 Çekirdek", cpu_freq_ghz=3.4,
        )
        self.assertEqual(self.hud.system_label.text(), "FPS 30 · GPU %61 · CPU %23")
        self.assertIn("RTX 3070", self.hud.system_label.toolTip())

    def test_system_refresh_without_sources_keeps_dashes(self):
        """Kaynaksız yenileme çökmez; ölçümü olmayan alan '—' kalır.

        FPS (capture bağlı değil) ve GPU (bu VM'de NVML yok) her zaman '—'.
        CPU: psutil yoksa '—', varsa gerçek ölçüm ('%N' — boşta %0 da
        gerçek bir ölçüm). 2026-11-08 17:18 UTC'de a2-bot-venv'e psutil
        kurulduğu için katı '—' beklentisi ortama bağımlı hale geldi;
        test artık iki durumu da kabul eder (birleştirme kararı).
        """
        import re

        self.hud._refresh_system_status()
        app.processEvents()
        self.assertRegex(
            self.hud.system_label.text(),
            r"^FPS — · GPU — · CPU (—|%\d+)$",
        )

    # -- Tespitler / radar -----------------------------------------------------

    def test_detections_feed_radar_blips(self):
        self.hud.update_state({"state": "WALKING", "detections": [
            {"distance_m": 12, "bearing_deg": 45, "kind": "gather"},
            {"distance_m": 30, "bearing_deg": 180, "kind": "monster"},
        ]})
        self.assertEqual(len(self.hud._radar_projection.widget.blips), 2)

    # -- Kontroller -----------------------------------------------------------

    def test_control_buttons_follow_state(self):
        # Başlangıç: beklemede -> Başlat açık, Durdur kapalı, Devam kapalı.
        self.hud.update_state({"state": "WAITING"})
        self.assertTrue(self.hud.btn_run.isEnabled())
        self.assertFalse(self.hud.btn_stop.isEnabled())
        self.assertFalse(self.hud.btn_resume.isEnabled())
        # Çalışırken: Başlat kapalı, Durdur açık.
        self.hud.update_state({"state": "GATHERING"})
        self.assertFalse(self.hud.btn_run.isEnabled())
        self.assertTrue(self.hud.btn_stop.isEnabled())
        # Başlamış + duraklatılmış: Devam açık.
        self.hud.update_state({"state": "PAUSED"})
        self.assertTrue(self.hud.btn_resume.isEnabled())

    def test_mode_buttons_highlight_active_mode(self):
        self.hud.set_mode("TOPLAMA (ROUTE)")
        self.assertIn("43d68a", self.hud.btn_gather.styleSheet())
        self.assertEqual(self.hud.btn_combat.styleSheet(), "")
        self.hud.set_mode("SAVAŞ (COMBAT)")
        self.assertIn("ff6a5e", self.hud.btn_combat.styleSheet())
        self.assertEqual(self.hud.btn_gather.styleSheet(), "")

    def test_f9_f10_f8_and_quit_signals_preserved(self):
        fired = []
        self.hud.f9_clicked.connect(lambda: fired.append("f9"))
        self.hud.f10_clicked.connect(lambda: fired.append("f10"))
        self.hud.f8_clicked.connect(lambda: fired.append("f8"))
        self.hud.quit_requested.connect(lambda: fired.append("quit"))
        self.hud.btn_menu.click()
        self.hud.btn_run.click()
        self.hud.btn_node.click()
        self.hud.btn_quit.click()
        app.processEvents()
        self.assertEqual(fired, ["f9", "f10", "f8", "quit"])

    def test_map_button_dispatches_to_f9_menu_action(self):
        """Harita düğmesi F9 menüsüne 'map' eylemini gönderir; menü bağlı
        değilse f9_clicked yedeğine düşer (sessizce yutulmaz)."""
        fired = []
        self.hud.f9_clicked.connect(lambda: fired.append("f9"))
        self.hud.btn_map.click()
        app.processEvents()
        self.assertEqual(fired, ["f9"])

    def test_minimize_restore_still_works(self):
        self.hud.toggle_minimized()
        app.processEvents()
        self.assertTrue(self.hud.is_minimized)
        self.assertTrue(self.hud._mini_bar.isVisible())
        self.hud.toggle_minimized()
        app.processEvents()
        self.assertFalse(self.hud.is_minimized)

    # -- HUD ayarları paneli ---------------------------------------------------

    def test_settings_gear_emits_settings_clicked(self):
        """High #5: overlay dişli opens main Ayarlar (signal), does not
        auto-toggle the in-HUD panel or resume the bot."""
        fired = []
        self.hud.settings_clicked.connect(lambda: fired.append("settings"))
        self.assertFalse(self.hud.settings_panel.isVisible())
        self.hud.btn_hud_settings.click()
        app.processEvents()
        self.assertEqual(fired, ["settings"])
        self.assertFalse(self.hud.settings_panel.isVisible())
        # Panel can still be opened programmatically for HUD-local toggles.
        self.hud.toggle_settings_panel()
        app.processEvents()
        self.assertTrue(self.hud.settings_panel.isVisible())

    def test_settings_live_toggles(self):
        self.hud.chk_detection_boxes.setChecked(False)
        app.processEvents()
        self.assertFalse(self.hud._detection_boxes_visible)
        self.assertFalse(self.hud._radar_projection.widget.show_blips)
        self.hud.chk_detection_boxes.setChecked(True)
        app.processEvents()
        self.assertTrue(self.hud._radar_projection.widget.show_blips)
        self.hud.chk_minimap.setChecked(False)
        app.processEvents()
        self.assertFalse(self.hud._minimap_visible)
        self.assertFalse(self.hud._radar_projection.isVisible())
        self.hud.chk_minimap.setChecked(True)
        app.processEvents()
        self.assertTrue(self.hud._minimap_visible)

    def test_settings_unbound_toggles_are_disabled_not_fake(self):
        """Otomatik rota / Düşük CAN kaçışı / Sesli uyarı: HUD kapsamında
        canlı kaynak yok -> devre dışı; tıklanabilir sahte anahtar değil."""
        self.assertFalse(self.hud.chk_auto_route.isEnabled())
        self.assertFalse(self.hud.chk_low_hp_escape.isEnabled())
        self.assertFalse(self.hud.chk_sound_alert.isEnabled())

    def test_reset_position_and_clear_log(self):
        self.hud.move(500, 400)
        self.hud.reset_position()
        self.assertEqual((self.hud.x(), self.hud.y()), (30, 30))
        self.hud.add_log("deneme kaydı")
        self.assertTrue(self.hud.logs)
        self.hud.btn_clear_log.click()
        app.processEvents()
        self.assertEqual(self.hud.logs, [])
        self.assertEqual(self.hud.log_label_1.text(), "")

    def test_telemetry_snapshot_has_prototype_rows(self):
        snap = self.hud.get_overlay_telemetry_snapshot()
        for key in (
            "route_current", "route_total", "route_loops", "route_zone",
            "target_node_id", "target_distance_m", "target_material",
            "system_status", "mode_key", "state_key",
            "detection_boxes_visible", "minimap_visible",
        ):
            self.assertIn(key, snap)


if __name__ == "__main__":
    unittest.main()
