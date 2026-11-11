# -*- coding: utf-8 -*-
"""
v20 MADDE 8 — Targena yanlış algı regresyon testleri (KOŞULMADI — yazıldı, koşma yok).

Kök neden: canlı oto-haritalama yolu (`ExplorerModeController.step_passive_cartography`
-> `_scan_3d_world_hybrid` -> `MaterialScanner.scan_3d_world_nameplates`) toplama
yolunun (`scan_frame_for_materials`) kullandığı exemplar kapısını
(`_refine_candidates_with_exemplars`: öğrenilmiş negatif örnek, sim >= 0.82)
hiç çalıştırmıyordu. Üstüne `_classify_3d_material`, `discriminate_3d_material`
mesh'i `learned_negative:monster_3d_negative` diye reddetse bile saf-renk yedek
kurallara düşüp mob gövdesini sarı+yeşil Targena kuralıyla (kural 4) "Targena"
diye etiketliyordu. Sonuç: Karnif / Two-Faced Muses / Sylphen gibi moblar
oto-haritalanan "Targena" düğümleri (#288/#289/#290) oluyordu.

Bu dosya üç kilidi doğrular:
1. `_classify_3d_material`, learned-negative reddinden sonra renk yedeğine
   düşmez (boş ad döner) — tarayıcı içi sızıntı kapalı.
2. `_refine_candidates_with_exemplars`, `monster_3d_negative` eşleşen Targena
   adayını düşürür — filtre kararı.
3. `_scan_3d_world_hybrid`, isim etiketi tespitlerini kartografa vermeden önce
   aynı exemplar kapısından geçirir — filtre kararına saygı (oto-haritalama
   yolundaki baypas kapalı).
"""

import numpy as np
import cv2

from core.explorer_mode import ExplorerModeController
from core.material_scanner import (
    DetectedMaterial,
    MaterialDiscriminationResult,
    MaterialScanner,
)
from core.route_manager import RouteManager


def _yellow_green_mesh_frame() -> np.ndarray:
    """Sarı (H 20-34) + yeşil yapraklı mesh: Targena renk kuralı 4'ü geçirir."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Mesh bölgesi: isim etiketi (x=900, y=400, cw=120, ch=20) varsayımıyla
    # y=424..499, x=880..1040 arasını sarı+yeşil doldur.
    mesh = np.zeros((75, 160, 3), dtype=np.uint8)
    mesh[:, :, 0] = 30    # B
    mesh[:, :, 1] = 200   # G
    mesh[:, :, 2] = 220   # R -> sarımsı (HSV H ~25)
    frame[424:499, 880:1040] = mesh
    # Yeşil yaprak şeridi (H ~60).
    frame[460:499, 880:1040] = (40, 180, 60)
    return frame


def _negative_verdict() -> MaterialDiscriminationResult:
    return MaterialDiscriminationResult(
        is_authentic_node=False,
        material_type="Negative Terrain",
        class_name="negative",
        category="",
        calibrated_confidence=0.12,
        rejection_reason="learned_negative:monster_3d_negative",
    )


class _MonsterNegativeMemory:
    """Sahte ExemplarMemory: her yamayı monster_3d_negative (sim 0.88) sayar."""

    def is_negative_patch(self, crop, class_id=None, threshold=None, feats=None):
        return (True, 0.88, {"class_name": "monster_3d_negative", "class_id": 10})

    def classify_patch(self, crop, min_sim=0.70, feats=None):
        return (None, None, 0.0)


class _CleanMemory:
    """Sahte ExemplarMemory: negatif yok; Targena pozitif örneği (sim 0.75)."""

    def is_negative_patch(self, crop, class_id=None, threshold=None, feats=None):
        return (False, 0.0, None)

    def classify_patch(self, crop, min_sim=0.70, feats=None):
        return (2, "targena", 0.75)


class TestClassify3dMaterialLearnedNegative:
    def test_learned_negative_mesh_never_falls_back_to_targena(self, monkeypatch):
        """discriminate learned-negative dediyse renk yedeği Targena vermez."""
        scanner = MaterialScanner()
        monkeypatch.setattr(
            "core.material_scanner.discriminate_3d_material",
            lambda *a, **k: _negative_verdict(),
        )
        frame = _yellow_green_mesh_frame()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        name, _in_air = scanner._classify_3d_material(frame, hsv, 900, 400, 120, 20)
        # Boş ad = reddedildi; çağıranlar (nameplate/crest tarayıcıları)
        # bu tespiti hiç üretmez.
        assert name == ""


class TestRefineGateDropsMonsterTargena:
    def test_monster_negative_targena_candidate_dropped(self):
        """Exemplar kapısı monster_3d_negative eşleşen Targena adayını düşürür."""
        scanner = MaterialScanner()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        det = DetectedMaterial(
            material_type="Targena",
            source="3d_world",
            screen_x=960,
            screen_y=540,
            bbox=(900, 400, 120, 120),
            rel_map_dx=10.0,
            rel_map_dy=5.0,
            estimated_distance_m=8.0,
            confidence=0.88,
        )
        kept = scanner._refine_candidates_with_exemplars(
            frame, [det], exemplar_memory=_MonsterNegativeMemory()
        )
        assert kept == []

    def test_clean_targena_candidate_kept(self, monkeypatch):
        """Negatif eşleşme yoksa (pozitif Targena örneği var) aday kalır."""
        scanner = MaterialScanner()
        # Kontrol deterministik olsun: sahte-zemin ve diskriminasyon
        # dalları gerçek dışı reddetmesin.
        monkeypatch.setattr(
            "core.node_quality_gate.is_fake_gray_rock_or_ground",
            lambda *a, **k: (False, 0.0, ""),
        )
        monkeypatch.setattr(
            "core.material_scanner.discriminate_3d_material",
            lambda *a, **k: MaterialDiscriminationResult(
                is_authentic_node=True,
                material_type="Targena",
                class_name="herb_targena",
                category="Herbs",
                calibrated_confidence=0.9,
                class_id=2,
            ),
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        det = DetectedMaterial(
            material_type="Targena",
            source="3d_world",
            screen_x=960,
            screen_y=540,
            bbox=(900, 400, 120, 120),
            rel_map_dx=10.0,
            rel_map_dy=5.0,
            estimated_distance_m=8.0,
            confidence=0.88,
        )
        kept = scanner._refine_candidates_with_exemplars(
            frame, [det], exemplar_memory=_CleanMemory()
        )
        assert len(kept) == 1
        assert kept[0].material_type == "Targena"


class TestHybridScanRespectsExemplarGate:
    def test_monster_targena_never_reaches_cartography(self, tmp_path, monkeypatch):
        """_scan_3d_world_hybrid, kapının elediği Targena'yı kartografa vermez.

        İsim etiketi tarayıcısı bir Targena mobu üretir; exemplar kapısı
        (sahte: canavar diye eler) onu düşürür; hibrit tarama çıktısında
        Targena kalmaz — yani process_frame_discoveries / auto-map yoluna
        canavar düğümü girmez.
        """
        rm = RouteManager(base_dir=str(tmp_path / "routes"))
        rm.create_region("test_v20_targena")
        rm.active_region = "test_v20_targena"
        explorer = ExplorerModeController(route_manager=rm, region="test_v20_targena")

        mob_targena = DetectedMaterial(
            material_type="Targena",
            source="3d_world",
            screen_x=960,
            screen_y=500,
            bbox=(900, 400, 120, 120),
            rel_map_dx=10.0,
            rel_map_dy=5.0,
            estimated_distance_m=8.0,
            confidence=0.88,
        )
        monkeypatch.setattr(
            explorer.material_scanner,
            "scan_3d_world_nameplates",
            lambda bgr, hsv=None: [mob_targena],
        )
        gate_calls = []

        def _fake_gate(frame_bgr, candidates, exemplar_memory=None):
            gate_calls.append(list(candidates))
            return []  # kapı: canavar, hepsini ele

        monkeypatch.setattr(
            explorer.material_scanner,
            "_refine_candidates_with_exemplars",
            _fake_gate,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = explorer._scan_3d_world_hybrid(frame)

        assert len(gate_calls) == 1
        assert gate_calls[0] == [mob_targena]
        assert not any(
            d.get("material_type") == "Targena" for d in out
        ), "canavar Targena hibrit taramadan kartografa sızdı"
