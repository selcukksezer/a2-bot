# -*- coding: utf-8 -*-
"""
D09: grid tabanlı tespit tekilleştirme (`_dedupe_detections_by_proximity`)
için eşdeğerlik + davranış testleri.

Kural: yeni yardımcı, eski O(n²) doğrusal taramayla AYNI sonucu üretmek
zorunda (sıra, ilk-eşleşme kuralı, birleştirme geri çağrısı). Testler eski
mantığı bağımsız bir referans uygulamayla karşılaştırır.
"""
import copy
import math
import random

import numpy as np
import pytest

from core.explorer_mode import (
    _MORPH_CLOSE_KERNEL_3X3,
    _dedupe_detections_by_proximity,
    ExplorerModeController,
)
from core.route_manager import RouteManager


# --- Eski (D09 öncesi) mantığın bağımsız referans kopyası -------------------
def _naive_dedup(detections, radius, on_duplicate=None, base=None):
    kept = [dict(d) for d in (base or [])]
    for det in detections:
        det = dict(det)
        dup = None
        for ex in kept:
            if (
                math.hypot(
                    int(det.get("screen_x", 0)) - int(ex.get("screen_x", 0)),
                    int(det.get("screen_y", 0)) - int(ex.get("screen_y", 0)),
                )
                <= radius
            ):
                dup = ex
                break
        if dup is not None:
            if on_duplicate is not None:
                on_duplicate(dup, det)
            continue
        kept.append(det)
    return kept


def _make_det(x, y, conf=0.5, mat="Gatherable Material", **extra):
    d = {
        "material_type": mat,
        "source": "3d_world",
        "screen_x": int(x),
        "screen_y": int(y),
        "bbox": (int(x), int(y), 10, 10),
        "rel_map_dx": 0.0,
        "rel_map_dy": 0.0,
        "estimated_distance_m": 5.0,
        "confidence": float(conf),
        "in_air": False,
        "stack_count": 1,
        "max_harvests": 3,
    }
    d.update(extra)
    return d


def _keep_higher_confidence(existing, det):
    if float(det["confidence"]) > float(existing.get("confidence", 0)):
        existing.update(det)


def _random_dets(n, seed, spread_x=1920, spread_y=1080):
    rng = random.Random(seed)
    return [
        _make_det(
            rng.randint(0, spread_x),
            rng.randint(0, spread_y),
            conf=round(rng.random(), 3),
        )
        for _ in range(n)
    ]


class TestGridDedupEquivalence:
    @pytest.mark.parametrize("radius", [9.0, 28.0, 32.0, 36.0])
    @pytest.mark.parametrize("seed", [1, 7, 42])
    @pytest.mark.parametrize("n", [0, 1, 25, 120])
    def test_matches_naive_randomized(self, radius, seed, n):
        dets = _random_dets(n, seed)
        expected = _naive_dedup(dets, radius, _keep_higher_confidence)
        actual = _dedupe_detections_by_proximity(
            copy.deepcopy(dets), radius, _keep_higher_confidence
        )
        assert actual == expected

    @pytest.mark.parametrize("radius", [9.0, 28.0, 36.0])
    def test_matches_naive_with_base_not_self_deduped(self, radius):
        # Taban öğeler kendi aralarında elenmez (3D/minimap birleştirme
        # noktalarındaki eski davranış); yalnızca yeni öğeler tabana karşı
        # kontrol edilir.
        base = [_make_det(100, 100, 0.9), _make_det(105, 102, 0.8)]
        new = [_make_det(103, 101, 0.5), _make_det(500, 500, 0.7)]
        expected = _naive_dedup(new, radius, _keep_higher_confidence, base=base)
        actual = _dedupe_detections_by_proximity(
            copy.deepcopy(new), radius, _keep_higher_confidence, base=copy.deepcopy(base)
        )
        assert actual == expected

    def test_matches_naive_when_merge_moves_coordinates(self):
        # Birleştirme korunan öğenin koordinatını taşırsa (existing.update),
        # sonraki tespitler YENİ konuma göre değerlendirilmeli — ızgara
        # hücresi güncellenir.
        dets = [
            _make_det(100, 100, 0.3),
            _make_det(110, 100, 0.9),  # ilk öğeyi (140, 100)'e taşır
            _make_det(145, 100, 0.5),  # yeni konuma 5 px, eskiye 45 px
            _make_det(300, 300, 0.5),
        ]

        def _move_merge(existing, det):
            if float(det["confidence"]) > float(existing.get("confidence", 0)):
                existing.update(det)
                existing["screen_x"] = 140
                existing["screen_y"] = 100

        expected = _naive_dedup(dets, 28.0, _move_merge)
        actual = _dedupe_detections_by_proximity(copy.deepcopy(dets), 28.0, _move_merge)
        assert actual == expected
        assert len(actual) == 2  # (140,100) birleşimi + (300,300)

    def test_first_match_in_insertion_order_wins(self):
        # Yarıçap içinde iki korunan öğe varsa İLK eklenen kazanır.
        base = [_make_det(100, 100, 0.5, mat="A"), _make_det(110, 100, 0.5, mat="B")]
        new = [_make_det(105, 100, 0.9, mat="C")]

        def _upgrade(existing, det):
            existing["material_type"] = det["material_type"]

        actual = _dedupe_detections_by_proximity(
            copy.deepcopy(new), 28.0, _upgrade, base=copy.deepcopy(base)
        )
        assert [d["material_type"] for d in actual] == ["C", "B"]

    def test_radius_boundary_inclusive(self):
        # (28,0) tam sınırda -> birleşir ve korunan öğe (28,0)'a taşınır;
        # (57,0) taşınmış öğeden 29 px uzakta -> ayrı kalır.
        dets = [_make_det(0, 0, 0.5), _make_det(28, 0, 0.9), _make_det(57, 0, 0.9)]
        actual = _dedupe_detections_by_proximity(
            copy.deepcopy(dets), 28.0, _keep_higher_confidence
        )
        assert len(actual) == 2
        assert actual[0]["confidence"] == 0.9
        assert (actual[0]["screen_x"], actual[0]["screen_y"]) == (28, 0)
        assert (actual[1]["screen_x"], actual[1]["screen_y"]) == (57, 0)

    def test_negative_coordinates(self):
        dets = [_make_det(-50, -50, 0.5), _make_det(-55, -50, 0.9), _make_det(500, 500, 0.5)]
        expected = _naive_dedup(dets, 28.0, _keep_higher_confidence)
        actual = _dedupe_detections_by_proximity(
            copy.deepcopy(dets), 28.0, _keep_higher_confidence
        )
        assert actual == expected


class TestYzDedupEndToEnd:
    def test_scan_yz_dedupes_overlapping_boxes_keep_higher_confidence(self):
        from core.reflex_layer import _gather_ids

        gather_id = sorted(_gather_ids())[0]

        class _FakeYolo:
            def detect(self, bgr):
                return [
                    # İki çakışan kutu (merkezler 10 px arayla): düşük güven önce.
                    {"class_id": gather_id, "class_name": "sapphire",
                     "x1": 100, "y1": 100, "x2": 120, "y2": 120, "confidence": 0.5},
                    {"class_id": gather_id, "class_name": "sapphire",
                     "x1": 110, "y1": 100, "x2": 130, "y2": 120, "confidence": 0.9},
                    # Uzak kutu: ayrı kalır.
                    {"class_id": gather_id, "class_name": "sapphire",
                     "x1": 800, "y1": 600, "x2": 820, "y2": 620, "confidence": 0.4},
                ]

        ctrl = ExplorerModeController(
            route_manager=RouteManager(), test_mode=True, yolo_detector=_FakeYolo()
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dets = ctrl._scan_yz_gather_detections(frame)
        assert len(dets) == 2
        merged = [d for d in dets if d["screen_x"] in (115, 120)]
        assert len(merged) == 1
        assert merged[0]["confidence"] == 0.9


class TestMinimapSharedKernel:
    def test_morph_kernel_is_single_shared_constant(self):
        # İki morphologyEx çağrısı da aynı modül düzeyi çekirdeği kullanır;
        # her karede yeni np.ones ayrılmaz.
        assert _MORPH_CLOSE_KERNEL_3X3.shape == (3, 3)
        assert _MORPH_CLOSE_KERNEL_3X3.dtype == np.uint8

    def test_minimap_icons_internal_still_detects_bright_blob(self):
        # I2 değişikliği (iç import kaldırma + paylaşılan çekirdek) tarama
        # sonucunu değiştirmez: parlak leke + yeşil rozet bulunur.
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[70:275, 1640:1905] = (52, 62, 56)
        frame[150:166, 1750:1766] = (235, 240, 245)
        frame[140:152, 1740:1752] = (40, 200, 60)  # yeşil rozet benzeri
        ctrl = ExplorerModeController(route_manager=RouteManager(), test_mode=True)
        dets = ctrl._scan_minimap_icons_internal(frame)
        assert len(dets) >= 1
        assert all(d["source"] == "minimap" for d in dets)
