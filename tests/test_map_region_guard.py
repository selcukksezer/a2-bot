# -*- coding: utf-8 -*-
"""Görev 4 (harita-istatistik) bölge koruması testleri — KOŞTURULMADI (pytest yasak).

Kapsar:
- RouteManager canlı bölge tespiti + uyumsuzluk engeli (aktif rota okuması
  ve yanlış bölgeye düğüm yazımı).
- Bölgeye duyarlı harita görseli seçimi (radar + map_viewer): Verteron'da
  Altgard görseli asla aday olmaz; görseli olmayan bölge boş liste döner.
- AutoCartographer dünya haritası (M) ORB bölge eşleşmesi (sentetik kırpma).
- BotWorker._rehydrate_saved_nodes_count last_saved_str rehidrasyonu.
"""

import json
import os

import cv2
import numpy as np
import pytest

from core.auto_cartographer import MINIMAP_NORM_RECT, AutoCartographer
from core.route_manager import RouteManager

# Dosya tests/ altında: repo kökü bir üst dizindir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# RouteManager: canlı bölge tespiti & uyumsuzluk koruması
# ----------------------------------------------------------------------
class TestRegionMismatchGuard:
    def _manager(self, tmp_path):
        rm = RouteManager(base_dir=str(tmp_path))
        rm.active_region = "Altgard"
        rm.active_route = "All"
        return rm

    def test_low_confidence_detection_ignored(self, tmp_path):
        rm = self._manager(tmp_path)
        assert rm.note_detected_region("Verteron", confidence=0.10, source="spatial_bbox") is None
        assert rm.get_detected_region() is None
        assert rm.get_region_mismatch() is None

    def test_unknown_region_detection_ignored(self, tmp_path):
        rm = self._manager(tmp_path)
        assert rm.note_detected_region("FooLand", confidence=0.95, source="spatial_bbox") is None
        assert rm.get_region_mismatch() is None

    def test_subregion_name_resolves_to_canonical(self, tmp_path):
        rm = self._manager(tmp_path)
        assert rm.note_detected_region("Dredgion Crash Site", confidence=0.95, source="spatial_bbox") == "Altgard"
        assert rm.get_region_mismatch() is None  # Altgard aktif, Altgard tespit

    def test_mismatch_blocks_active_route_read(self, tmp_path):
        rm = self._manager(tmp_path)
        node = rm.add_node(map_x=100, map_y=100, region_name="Altgard")
        assert node is not None
        assert rm.note_detected_region("Verteron", confidence=0.95, source="spatial_bbox") == "Verteron"
        mismatch = rm.get_region_mismatch()
        assert mismatch is not None
        assert mismatch["active_region"] == "Altgard"
        assert mismatch["detected_region"] == "Verteron"
        assert rm.is_route_blocked_by_region_mismatch() is True
        # Aktif-rotayı çalıştıran (argümansız) okuma engellenir...
        assert rm.get_route_nodes() == []
        # ...ama açıkça bölge isteyen okuma etkilenmez.
        assert len(rm.get_route_nodes(region_name="Altgard")) == 1

    def test_mismatch_clears_when_active_region_fixed(self, tmp_path):
        rm = self._manager(tmp_path)
        rm.note_detected_region("Verteron", confidence=0.95, source="spatial_bbox")
        assert rm.is_route_blocked_by_region_mismatch() is True
        rm.active_region = "Verteron"
        assert rm.get_region_mismatch() is None
        assert rm.is_route_blocked_by_region_mismatch() is False

    def test_add_node_refused_into_mismatched_region(self, tmp_path):
        rm = self._manager(tmp_path)
        rm.note_detected_region("Verteron", confidence=0.95, source="spatial_bbox")
        # Verteron'dayken Altgard'a yazılmaz...
        assert rm.add_node(map_x=100, map_y=100, region_name="Altgard") is None
        # ...ama tespit edilen bölgenin kendisine yazılabilir.
        node = rm.add_node(map_x=100, map_y=100, region_name="Verteron")
        assert node is not None
        assert node.region == "Verteron"


# ----------------------------------------------------------------------
# Bölgeye duyarlı harita görseli seçimi (radar + map_viewer)
# ----------------------------------------------------------------------
class TestRegionMapSelection:
    def test_verteron_map_resolves_to_verteron_image(self):
        from ui.cartography_radar import hd_zone_map_path, region_map_candidates

        candidates = region_map_candidates("Verteron")
        assert len(candidates) == 1
        assert candidates[0].endswith("verteron.png")
        assert os.path.isfile(candidates[0])
        assert hd_zone_map_path("Verteron").endswith("verteron.png")

    def test_altgard_default_preserved(self):
        from ui.cartography_radar import hd_zone_map_path, region_map_candidates

        assert region_map_candidates("Altgard")[0].endswith("altgard.png")
        assert region_map_candidates(None)[0].endswith("altgard.png")
        assert hd_zone_map_path().endswith("altgard.png")

    def test_region_without_map_returns_no_candidates(self):
        from ui.cartography_radar import is_altgard_map_region, region_map_candidates

        assert region_map_candidates("Heiron") == []
        assert is_altgard_map_region("Verteron") is False
        assert is_altgard_map_region("Altgard") is True

    def test_map_viewer_matches_radar(self):
        from ui import map_viewer
        from ui.cartography_radar import hd_zone_map_path as radar_path

        assert map_viewer.hd_zone_map_path("Verteron") == radar_path("Verteron")
        assert map_viewer.hd_zone_map_path() == radar_path()
        assert map_viewer.region_map_candidates("Heiron") == []


# ----------------------------------------------------------------------
# AutoCartographer: dünya haritası ORB bölge eşleşmesi (sentetik kırpma)
# ----------------------------------------------------------------------
def _minimal_cartographer():
    ac = AutoCartographer.__new__(AutoCartographer)
    ac.route_manager = None
    ac._region_scanner = None
    ac._detected_region = None
    ac._last_region_detect_ts = 0.0
    ac._last_world_map_region_ts = 0.0
    ac._world_map_orb_cache = None
    ac.minimap_norm_rect = MINIMAP_NORM_RECT
    return ac


def _center_crop(path, w=1280, h=720):
    img = cv2.imread(path)
    assert img is not None, path
    ih, iw = img.shape[:2]
    return img[ih // 2 - h // 2 : ih // 2 + h // 2, iw // 2 - w // 2 : iw // 2 + w // 2]


class TestWorldMapRegionIdentification:
    def test_verteron_crop_identifies_verteron(self):
        ac = _minimal_cartographer()
        crop = _center_crop(os.path.join(REPO_ROOT, "assets", "maps", "verteron.png"))
        res = ac.identify_region_from_world_map(crop)
        assert res is not None
        assert res["canonical_region"] == "Verteron"
        assert res["source"] == "world_map_orb"

    def test_altgard_crop_identifies_altgard(self):
        ac = _minimal_cartographer()
        crop = _center_crop(os.path.join(REPO_ROOT, "assets", "maps", "altgard.png"))
        res = ac.identify_region_from_world_map(crop)
        assert res is not None
        assert res["canonical_region"] == "Altgard"

    def test_blank_frame_identifies_nothing(self):
        ac = _minimal_cartographer()
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert ac.identify_region_from_world_map(blank) is None
        assert ac.identify_region_from_world_map(None) is None

    def test_detect_region_from_frame_rejects_blank(self):
        ac = _minimal_cartographer()
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert ac.detect_region_from_frame(None) is None
        # Boş kare: başlık eşleşmesi yok -> spatial_bbox değil -> None.
        assert ac.detect_region_from_frame(blank, player_map_pos=(0.0, 0.0)) is None


# ----------------------------------------------------------------------
# BotWorker: last_saved_str rehidrasyonu
# ----------------------------------------------------------------------
class TestLastSavedRehydration:
    def _worker_with_nodes(self, tmp_path):
        reg_dir = tmp_path / "Altgard"
        reg_dir.mkdir()
        nodes = [
            {"id": 1, "region": "Altgard", "routes": ["All"], "map_x": 10, "map_y": 10},
            {"id": 2, "region": "Altgard", "routes": ["All"], "map_x": 20, "map_y": 20},
            {"id": 5, "region": "Altgard", "routes": ["All"], "map_x": 30, "map_y": 30},
        ]
        (reg_dir / "region_data.json").write_text(
            json.dumps({"region": "Altgard", "routes": {"All": [1, 2, 5]}, "nodes": nodes}),
            encoding="utf-8",
        )
        from core.state_machine import BotWorker

        rm = RouteManager(base_dir=str(tmp_path))
        rm.active_region = "Altgard"
        rm.active_route = "All"
        sm = BotWorker.__new__(BotWorker)
        sm.__dict__["route_manager"] = rm
        sm.__dict__["saved_nodes_count"] = 0
        sm.__dict__["last_saved_str"] = "None"
        return sm

    def test_rehydrates_count_and_last_saved(self, tmp_path):
        sm = self._worker_with_nodes(tmp_path)
        sm._rehydrate_saved_nodes_count()
        assert sm.saved_nodes_count == 3
        assert sm.last_saved_str == "Node 5"

    def test_existing_last_saved_is_preserved(self, tmp_path):
        sm = self._worker_with_nodes(tmp_path)
        sm.__dict__["last_saved_str"] = "Node 2"
        sm._rehydrate_saved_nodes_count()
        assert sm.last_saved_str == "Node 2"


# ----------------------------------------------------------------------
# Rota modu hook'u (v10 5b): _step_route_mode pasif kartografiyi
# gather_while_exploring=False ile çağırır (rota modunun kendi toplama
# motoru vardır; kartografi tick'i yalnızca konum/bölge besler).
# ----------------------------------------------------------------------
class TestRouteModePassiveCartographyHook:
    def test_step_route_mode_calls_passive_cartography_without_gather(
        self, monkeypatch
    ):
        import threading
        import time

        from core.state_machine import BotWorker

        sm = BotWorker.__new__(BotWorker)
        sm.__dict__.update(
            # is_paused property'si self._lock/_paused okur; __new__ ile
            # kurulan iskelette Qt __init__'i çalışmadığı için bu ikisi
            # __dict__'te yoksa RuntimeError verir.
            _lock=threading.Lock(),
            _paused=False,
            _state="WALKING",
            _failure_wait_until=0.0,
            _last_gather_time=time.time(),
            active_node=None,
            _is_gathering_active=False,
        )
        calls = []

        def _spy(gather_while_exploring=False):
            calls.append(gather_while_exploring)

        monkeypatch.setattr(sm, "_check_game_focus", lambda: True)
        monkeypatch.setattr(sm, "_should_enter_break", lambda now: False)
        monkeypatch.setattr(sm, "_handle_combat_and_death", lambda: False)
        monkeypatch.setattr(sm, "_handle_blocking_game_window", lambda: False)
        monkeypatch.setattr(
            sm,
            "_get_cfg",
            lambda key, default=None: {
                "pause_after_no_gather_min": 15,
                "gather_on_the_way": False,
            }.get(key, default),
        )
        monkeypatch.setattr(sm, "_select_and_navigate_next_node", lambda: None)
        monkeypatch.setattr(sm, "_run_passive_auto_cartography", _spy)

        sm._step_route_mode()

        assert calls == [False]
