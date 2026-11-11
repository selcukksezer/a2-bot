# -*- coding: utf-8 -*-
"""D08 (economy_broker) gelistirme dilimi testleri.

Kilitlenen iyilestirmeler:
  I1 - get_price buyuk/kucuk harf duyarsiz arama O(n) -> O(1) dizin
  I2 - select_most_profitable_route'ta inspect.signature rota basina 1 kez
       degil, secim basina 1 kez cagrilir
  I3 - scan_broker_window: _pooled_std_bgr (np.std ile ayni deger, daha hizli)
       + onceden hesaplanmis HSV imza dizileri

Bu testler davranis esitligini kilitler; sure olcumu bench betigindedir
(~/workspace/bench-d08.py) cunku paylasimli VM'de zamanlama gurultuludur.
"""

import json
import os

import cv2
import numpy as np
import pytest

import core.economy_broker as eb
from core.economy_broker import (
    BROKER_ICON_HSV_SIGNATURE_ARRAYS,
    BROKER_ICON_HSV_SIGNATURES,
    TradeBrokerPriceScanner,
    _pooled_std_bgr,
)


def _scanner(tmp_path, auto_save=False):
    return TradeBrokerPriceScanner(
        price_db_path=str(tmp_path / "prices.json"), auto_save=auto_save
    )


# ---------------------------------------------------------------- I1
class TestGetPriceLowerIndex:
    def test_exact_and_case_variant_and_unknown(self, tmp_path):
        sc = _scanner(tmp_path)
        assert sc.get_price("Orichalcum Ore") == 14500
        assert sc.get_price("oRiChAlCuM oRe") == 14500
        assert sc.get_price("ORICHALCUM ORE") == 14500
        assert sc.get_price("  Aria  ") == 9800
        assert sc.get_price("bilinmeyen xyz", 999) == 999
        assert sc.get_price("", 777) == 777

    def test_index_stays_in_sync_after_update_price(self, tmp_path):
        sc = _scanner(tmp_path)
        sc.update_price("Aria", 12345, save=False)
        assert sc.get_price("aria") == 12345
        assert sc.get_price("ARIA") == 12345
        # Yeni materyal de dizine girer.
        sc.update_price("Yeni Materyal", 4242, save=False)
        assert sc.get_price("yeni materyal") == 4242
        assert sc.get_price("YENI MATERYAL") == 4242

    def test_case_variant_update_keeps_first_key_wins(self, tmp_path):
        """Farkli yazimli anahtar eklenince eski dogrusal tarama 'ilk
        eslesen anahtar kazanir' davranisi korunur."""
        sc = _scanner(tmp_path)
        assert sc.get_price("ARIA") == 9800  # varsayilan "Aria"
        sc.update_price("aria", 1111, save=False)
        assert sc.get_price("aria") == 1111  # tam eslesme yeni anahtari bulur
        assert sc.get_price("ARIA") == 9800  # duyarsiz aramada ilk anahtar kazanir

    def test_load_from_json_with_duplicate_case_keys(self, tmp_path):
        db = tmp_path / "prices.json"
        db.write_text(
            json.dumps(
                {"prices": {"Aria": 9800, "ARIA": 5555, "Orichalcum Ore": 14500}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        sc = TradeBrokerPriceScanner(price_db_path=str(db), auto_save=False)
        assert sc.get_price("ARIA") == 5555  # tam eslesme
        assert sc.get_price("aRiA") == 9800  # duyarsiz: dosyada ilk gelen kazanir
        assert sc.get_price("orichalcum ore") == 14500

    def test_load_rebuilds_index(self, tmp_path):
        db = tmp_path / "prices.json"
        db.write_text(
            json.dumps({"prices": {"Orichalcum Ore": 77777}}, ensure_ascii=False),
            encoding="utf-8",
        )
        sc = TradeBrokerPriceScanner(price_db_path=str(db), auto_save=False)
        assert sc.get_price("ORICHALCUM ORE") == 77777
        # Dosyada olmayan materyal varsayilan tablodan gelir.
        assert sc.get_price("aria") == 9800


# ---------------------------------------------------------------- I2
class _FakeRouteManager:
    def __init__(self, n_routes=5, style="kw"):
        self.routes = [f"Rota-{i:02d}" for i in range(n_routes)]
        self.active_region = None
        self.active_route = None
        self.meters_per_pixel = 0.28
        self.style = style

    def get_regions(self):
        return ["Verteron"]

    def get_routes(self, region):
        return list(self.routes)

    def get_route_nodes(self, *args, **kwargs):
        if kwargs:
            region_name = kwargs.get("region_name", kwargs.get("region"))
            route_name = kwargs.get("route_name", kwargs.get("route"))
        else:  # positional
            region_name, route_name = args[0], args[1]
        assert region_name == "Verteron"
        assert route_name in self.routes
        return [
            {
                "resource_type": "Orichalcum Ore",
                "max_harvests": 3,
                "in_air": False,
                "map_x": 100.0 + i * 37.0,
                "map_y": 200.0 + i * 11.0,
            }
            for i in range(10)
        ]


class _KwRouteManager(_FakeRouteManager):
    def get_route_nodes(self, region_name, route_name):  # noqa: D102
        return _FakeRouteManager.get_route_nodes(
            self, region_name=region_name, route_name=route_name
        )


class _RegionRouteManager(_FakeRouteManager):
    def get_route_nodes(self, region, route):  # noqa: D102
        return _FakeRouteManager.get_route_nodes(
            self, region=region, route=route
        )


class _PositionalRouteManager(_FakeRouteManager):
    def get_route_nodes(self, a, b):  # noqa: D102
        return _FakeRouteManager.get_route_nodes(self, a, b)


@pytest.mark.parametrize("manager_cls", [_KwRouteManager, _RegionRouteManager, _PositionalRouteManager])
def test_select_route_inspects_signature_once(tmp_path, monkeypatch, manager_cls):
    sc = _scanner(tmp_path)
    mgr = manager_cls(n_routes=5)
    calls = {"n": 0}
    orig_sig = eb.inspect.signature

    def counting_sig(obj):
        calls["n"] += 1
        return orig_sig(obj)

    monkeypatch.setattr(eb.inspect, "signature", counting_sig)
    res = sc.select_most_profitable_route(mgr, activate=True)
    assert calls["n"] == 1, f"inspect.signature {calls['n']} kez cagrildi, 1 bekleniyordu"
    assert res["best_region"] == "Verteron"
    assert res["best_route"] == "Rota-00"
    assert res["expected_kinah_per_hour"] > 0
    assert res["activated"] is True
    assert mgr.active_region == "Verteron"
    assert mgr.active_route == "Rota-00"


def test_select_route_signature_once_for_many_routes(tmp_path, monkeypatch):
    sc = _scanner(tmp_path)
    mgr = _KwRouteManager(n_routes=40)
    calls = {"n": 0}
    orig_sig = eb.inspect.signature

    def counting_sig(obj):
        calls["n"] += 1
        return orig_sig(obj)

    monkeypatch.setattr(eb.inspect, "signature", counting_sig)
    res = sc.select_most_profitable_route(mgr, activate=False)
    assert calls["n"] == 1
    assert len(res["route_rankings"]) == 40


# ---------------------------------------------------------------- I3
class TestPooledStd:
    def test_matches_np_std_on_random_rois(self):
        rng = np.random.default_rng(11)
        for shape in [(734, 1229, 3), (100, 200, 3), (24, 24, 3), (480, 640, 3)]:
            arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
            assert abs(_pooled_std_bgr(arr) - float(np.std(arr))) < 1e-9

    def test_flat_frame_is_zero(self):
        flat = np.full((734, 1229, 3), 128, dtype=np.uint8)
        assert _pooled_std_bgr(flat) == 0.0
        assert float(np.std(flat)) == 0.0

    def test_precomputed_signature_arrays_match_source(self):
        assert len(BROKER_ICON_HSV_SIGNATURE_ARRAYS) == len(BROKER_ICON_HSV_SIGNATURES)
        for (name_a, lo_a, hi_a), (name_b, lo_b, hi_b) in zip(
            BROKER_ICON_HSV_SIGNATURE_ARRAYS, BROKER_ICON_HSV_SIGNATURES
        ):
            assert name_a == name_b
            assert lo_a.dtype == np.uint8 and hi_a.dtype == np.uint8
            assert lo_a.tolist() == list(lo_b)
            assert hi_a.tolist() == list(hi_b)


class TestScanBrokerWindow:
    def _noise_frame(self, seed=7):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)

    def test_flat_frame_rejected_prices_unchanged(self, tmp_path):
        sc = _scanner(tmp_path)
        before = dict(sc.prices)
        flat = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        out = sc.scan_broker_window(flat)
        assert out == before
        assert sc.last_scan_detected == []

    def test_detects_icon_patch_in_row_zero(self, tmp_path):
        sc = _scanner(tmp_path)
        frame = self._noise_frame()
        # Elmas (Diamond) HSV imzasi (88..108, 25..120, 210..255) icinde bir renk.
        hsv_px = np.array([[[98, 80, 230]]], dtype=np.uint8)
        bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
        # 0. satir ikon sutunu: ROI (384..1613, 173..907), satir 0 y 103..170,
        # ikon x %4..%16 -> cercevede x 433..581, y 276..343.
        frame[276:343, 433:581] = bgr_px
        out = sc.scan_broker_window(frame)
        assert "Diamond Gemstone" in sc.last_scan_detected
        assert out == dict(sc.prices)  # OCR yok: fiyatlar degismez

    def test_grayscale_and_bgra_frames_do_not_crash(self, tmp_path):
        sc = _scanner(tmp_path)
        rng = np.random.default_rng(3)
        gray = rng.integers(0, 256, size=(1080, 1920), dtype=np.uint8)
        assert isinstance(sc.scan_broker_window(gray), dict)
        bgra = rng.integers(0, 256, size=(1080, 1920, 4), dtype=np.uint8)
        assert isinstance(sc.scan_broker_window(bgra), dict)
        assert sc.scan_broker_window(None) == dict(sc.prices)
