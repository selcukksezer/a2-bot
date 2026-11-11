# -*- coding: utf-8 -*-
"""D15: MinimapScanner iyilestirme kilit testleri.

Kapsar:
1. `_masked_ncc` yerel-istatistik paylasimi (`_masked_ncc_local_std`):
   onceden hesaplanmis loc_std ile taze hesaplama birebir aynidir.
2. `scan_minimap_crop` NCC asamasi: ayni maskeyi paylasan sablonlar
   istatistigi paylasir (filter2D cagri sayisi duser) ve NCC yalnizca
   gecerli bbox + pad kirpiminda calisir (islenen piksel duser);
   tarama sonuclari referans snapshot ile birebir aynidir.
3. Yön (heading) döndürme önbelleği: ikinci çağrı warpAffine yapmaz,
   sonuç aynıdır; şablon değişince önbellek sıfırlanır.
4. Sözleşme zinciri (`get_minimap_scanner_*`): her üst düzey çağrı
   `validate_minimap_scanner_contract`'u tam 1 kez çalıştırır ve
   döndürülen iç içe sözlükler beklenen yapıyla birebir aynıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.minimap_scanner as ms_mod
from core.minimap_scanner import MinimapScanner


def _paste_sprout(crop_bgr: np.ndarray, top_left_x: int, top_left_y: int) -> None:
    """Kanonik filiz sablonunu yesil/beyaz renkli yapiştir (test_w37 ile ayni tarif)."""
    tpl = MinimapScanner._CANONICAL_SPROUT_TPL
    patch_bgr = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    leaf_pixels = tpl >= 175
    patch_bgr[leaf_pixels, 0] = np.clip(tpl[leaf_pixels].astype(np.int32) - 45, 130, 255).astype(np.uint8)
    patch_bgr[leaf_pixels, 1] = tpl[leaf_pixels]
    patch_bgr[leaf_pixels, 2] = np.clip(tpl[leaf_pixels].astype(np.int32) - 35, 140, 255).astype(np.uint8)
    crop_bgr[top_left_y : top_left_y + 16, top_left_x : top_left_x + 16] = patch_bgr


# ----------------------------------------------------------------------
# 1. _masked_ncc yerel-istatistik paylaşımı
# ----------------------------------------------------------------------
def test_masked_ncc_loc_std_reuse_is_exact() -> None:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 255, size=(257, 332), dtype=np.uint8)
    for tpl, mask in (
        (MinimapScanner._CANONICAL_SPROUT_TPL, MinimapScanner._CANONICAL_SPROUT_MASK),
        (MinimapScanner._CANONICAL_SPROUT_TPL_ALT, MinimapScanner._CANONICAL_SPROUT_MASK),
    ):
        for mult in (0.95, 1.00, 1.05):
            ks = max(9, int(round(16.0 * mult)))
            t_scaled = cv2.resize(tpl, (ks, ks), interpolation=cv2.INTER_AREA)
            m_scaled = (
                cv2.resize(mask.astype(np.float32), (ks, ks), interpolation=cv2.INTER_LINEAR) > 0.40
            ).astype(np.uint8)
            fresh = MinimapScanner._masked_ncc(gray, t_scaled, m_scaled)
            loc_std = MinimapScanner._masked_ncc_local_std(gray, m_scaled)
            assert loc_std is not None
            reused = MinimapScanner._masked_ncc(gray, t_scaled, m_scaled, loc_std=loc_std)
            assert np.array_equal(fresh, reused), f"loc_std reuse differs (mult={mult})"


def test_masked_ncc_local_std_none_for_tiny_mask() -> None:
    gray = np.full((40, 40), 100, dtype=np.uint8)
    tiny_mask = np.zeros((16, 16), dtype=np.uint8)
    tiny_mask[0, 0] = 1
    assert MinimapScanner._masked_ncc_local_std(gray, tiny_mask) is None
    out = MinimapScanner._masked_ncc(gray, np.zeros((16, 16), dtype=np.uint8), tiny_mask)
    assert out.shape == gray.shape and float(np.max(out)) == 0.0


# ----------------------------------------------------------------------
# 2. Tarama NCC işi azaldı + sonuçlar referansla birebir aynı
# ----------------------------------------------------------------------
def test_scan_ncc_shares_stats_and_crops_to_bbox(monkeypatch, tmp_path) -> None:
    """Yalnızca 2 kanonik şablon varken (boş templates_dir) bir tarama:
    - filter2D: ölçek başına 4 çağrı (paylaşımlı istatistik 2 + corr 2)
      x 3 ölçek = 12 çağrı (önce: 18),
    - işlenen piksel tam-kare hacminden küçük (bbox+pad kırpımı)."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))
    assert scanner._external_sprout_templates == []
    crop = np.full((257, 332, 3), (52, 62, 56), dtype=np.uint8)
    _paste_sprout(crop, 200, 100)
    _paste_sprout(crop, 60, 40)

    work = {"calls": 0, "pixels": 0}
    real_filter2d = cv2.filter2D

    def counting(src, ddepth, kernel, **kw):
        work["calls"] += 1
        work["pixels"] += int(src.shape[0]) * int(src.shape[1])
        return real_filter2d(src, ddepth, kernel, **kw)

    monkeypatch.setattr(cv2, "filter2D", counting)
    res = scanner.scan_minimap_crop(crop)
    assert work["calls"] == 12, f"beklenen 12 filter2D, gelen {work['calls']}"
    full_pixels = 257 * 332 * 18  # önceki: 18 çağrı x tam kare
    assert work["pixels"] < full_pixels, f"piksel hacmi düşmedi: {work['pixels']} vs {full_pixels}"
    assert len(res["gatherable_nodes"]) == 2


def test_scan_results_match_reference_snapshots() -> None:
    """D15 öncesi snapshot ile birebir aynı tarama sonuçları
    (gerçek ekran görüntüsü + sentetik 2 filiz + boş kesit)."""
    scanner = MinimapScanner()
    shot_path = _PROJECT_ROOT / "reference_screenshots" / "minimap_sprout_closeup.jpg"
    if not shot_path.is_file():
        pytest.skip("reference screenshot yok")
    shot = cv2.imread(str(shot_path))
    assert shot is not None
    res = scanner.scan_minimap_crop(shot)
    nodes = res["gatherable_nodes"]
    # NOT (W37 düzeltmesi): bu karede iki gerçek filiz var; tarayıcı artık
    # kanonik şablonun kendi 16px boyutunu da denediği için ikisini de
    # buluyor. Önce yalnızca (292, 176) 0.8845 ile geçiyor, (126, 266)
    # 0.8468'de kalıyordu. Her iki filiz de yerel ölçekte NCC 1.0 verir;
    # düğümler dist_px sırasına göre dizilir (101.26 < 119.29).
    assert [(n["pos"], n["ncc_score"], n["confidence"], n["chromatic_density"]) for n in nodes] == [
        ((126, 266), 1.0, 1.0, 0.762),
        ((292, 176), 1.0, 1.0, 0.463),
    ]
    assert [(n["pos"], n["area"]) for n in res["monster_dots"]] == [((238, 206), 30.0), ((258, 188), 31.0)]
    assert res["hostile_dots"] == []
    assert res["player_center"] == (172.71, 176.16)

    crop = np.full((257, 332, 3), (52, 62, 56), dtype=np.uint8)
    _paste_sprout(crop, 200, 100)
    _paste_sprout(crop, 60, 40)
    res2 = scanner.scan_minimap_crop(crop)
    assert [(n["pos"], n["ncc_score"], n["confidence"], n["chromatic_density"]) for n in res2["gatherable_nodes"]] == [
        ((208, 108), 0.9985, 0.998, 0.263),
        ((68, 48), 0.9985, 0.998, 0.263),
    ]

    empty = np.full((257, 332, 3), (52, 62, 56), dtype=np.uint8)
    res3 = scanner.scan_minimap_crop(empty)
    assert res3["gatherable_nodes"] == [] and res3["monster_dots"] == [] and res3["hostile_dots"] == []


# ----------------------------------------------------------------------
# 3. Heading döndürme önbelleği
# ----------------------------------------------------------------------
def _roi_with_rotated_real_arrow(scanner: MinimapScanner, angle_deg: float) -> np.ndarray:
    tpl = scanner._player_arrow_template
    assert tpl is not None
    th, tw = tpl.shape[:2]
    roi = np.full((48, 48), 40, dtype=np.uint8)
    m = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(tpl, m, (tw, th))
    y0, x0 = 24 - th // 2, 24 - tw // 2
    roi[y0 : y0 + th, x0 : x0 + tw] = np.maximum(roi[y0 : y0 + th, x0 : x0 + tw], rot)
    return roi


def test_heading_rotation_cache_reuses_warps(monkeypatch) -> None:
    scanner = MinimapScanner()
    assert scanner._player_arrow_template is not None
    roi = _roi_with_rotated_real_arrow(scanner, 0)

    warps = {"n": 0}
    real_warp = cv2.warpAffine

    def counting(*args, **kwargs):
        warps["n"] += 1
        return real_warp(*args, **kwargs)

    first = scanner._detect_heading_via_template(roi)
    assert first is not None and abs(first - 282.0) < 0.01
    assert len(scanner._heading_rot_cache) == 44  # 36 kaba + 8 ince ayar
    roi90 = _roi_with_rotated_real_arrow(scanner, 90)

    monkeypatch.setattr(cv2, "warpAffine", counting)
    warps["n"] = 0
    second = scanner._detect_heading_via_template(roi)
    assert second == first
    assert warps["n"] == 0, f"sıcak önbellek warpAffine yapmamalı, gelen {warps['n']}"

    # Göreli dönüş korunur: +90 döndürme başlığı -90 kaydırır.
    h90 = scanner._detect_heading_via_template(roi90)
    assert h90 is not None
    assert abs((first - h90) % 360.0 - 90.0) < 0.01


def test_heading_cache_invalidates_on_template_swap(monkeypatch) -> None:
    scanner = MinimapScanner()
    roi = _roi_with_rotated_real_arrow(scanner, 0)
    assert scanner._detect_heading_via_template(roi) is not None
    assert len(scanner._heading_rot_cache) == 44

    # Şablonu değiştir: önbellek sıfırlanmalı, warpAffine yeniden çalışmalı.
    tri = np.zeros((16, 16), dtype=np.uint8)
    cv2.fillPoly(tri, [np.array([[8, 1], [13, 13], [3, 13]], dtype=np.int32)], 255)
    scanner._player_arrow_template = tri
    scanner._player_arrow_mask = (tri > 0).astype(np.uint8) * 255

    warps = {"n": 0}
    real_warp = cv2.warpAffine

    def counting(*args, **kwargs):
        warps["n"] += 1
        return real_warp(*args, **kwargs)

    monkeypatch.setattr(cv2, "warpAffine", counting)
    tri_roi = np.full((36, 36), 40, dtype=np.uint8)
    m = cv2.getRotationMatrix2D((8, 8), -90, 1.0)
    tri_roi[10:26, 10:26] = np.maximum(tri_roi[10:26, 10:26], real_warp(tri, m, (16, 16)))
    warps["n"] = 0
    heading = scanner._detect_heading_via_template(tri_roi)
    assert heading is not None and abs(heading - 226.0) < 0.01
    assert warps["n"] > 0, "şablon değişince döndürmeler yeniden hesaplanmalı"


# ----------------------------------------------------------------------
# 4. Sözleşme zinciri tek-geçişli
# ----------------------------------------------------------------------
def test_contract_chain_validates_once_per_call(monkeypatch) -> None:
    calls = {"n": 0}
    real_validate = ms_mod.validate_minimap_scanner_contract

    def counting():
        calls["n"] += 1
        return real_validate()

    monkeypatch.setattr(ms_mod, "validate_minimap_scanner_contract", counting)
    for fn in (
        ms_mod.get_minimap_scanner_audit_summary,
        ms_mod.get_minimap_scanner_readiness_probe,
        ms_mod.get_minimap_scanner_compliance_certificate,
        ms_mod.get_minimap_scanner_integrity_seal,
        ms_mod.get_minimap_scanner_telemetry_payload,
        ms_mod.get_minimap_scanner_operational_digest,
        ms_mod.get_minimap_scanner_audit_record,
        ms_mod.get_minimap_scanner_readiness_status,
        ms_mod.get_minimap_scanner_baseline_metrics,
    ):
        calls["n"] = 0
        fn()
        assert calls["n"] == 1, f"{fn.__name__}: beklenen 1 doğrulama, gelen {calls['n']}"


def test_contract_chain_output_structure() -> None:
    expected = {
        "module": "minimap_scanner",
        "audit_passed": True,
        "probe": {
            "module": "minimap_scanner",
            "ready": True,
            "certificate": {
                "module": "minimap_scanner",
                "certified": True,
                "seal": {
                    "module": "minimap_scanner",
                    "sealed": True,
                    "telemetry": {
                        "module": "minimap_scanner",
                        "payload": {
                            "module": "minimap_scanner",
                            "operational": True,
                            "audit": {
                                "module": "minimap_scanner",
                                "audit_passed": True,
                                "readiness": {
                                    "module": "minimap_scanner",
                                    "ready": True,
                                    "metrics": {
                                        "module": "minimap_scanner",
                                        "contract_valid": True,
                                        "capability_count": 3,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    assert ms_mod.get_minimap_scanner_audit_summary() == expected
    assert ms_mod.get_minimap_scanner_readiness_status() == {
        "module": "minimap_scanner",
        "ready": True,
        "metrics": {
            "module": "minimap_scanner",
            "contract_valid": True,
            "capability_count": 3,
        },
    }
