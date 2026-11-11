# -*- coding: utf-8 -*-
"""
v20/empty-slots regression tests (MADDE 1: BOŞ SLOT 6/7/8'E BASMA).

Canlı v19 kanıtı (2026-11-11 15:11:28): SKILL KEY '6','8' basıldı, oysa
6/7/8 slotları BOŞTU ve knobs keys=1,2,3,4,q,e,r,t idi. İki delik vardı:

1. Okuyucu deliği (core/skill_bar_reader.py): boş slot, tüm-slot
   parlaklık/doygunluk/kontrast/mavi eşiklerini geçebiliyor (çerçeve
   parlaması / yumuşak gradyan) ve exemplar-hazır baypası (class 20/13)
   taban eşikleri hiç aramadan ready=True yazabiliyordu. :780'deki
   "empty slots read dark and are never ready" iddiası canlıda çürüdü.
2. Karar deliği (core/mana_phase_combat.py decide()): hazır listesi
   havuza aynen alınıyordu (active_pool = ready + fallback); burst yolu
   _send_configured_combat_skill filtrelerken mana-kararı→bus yolu
   filtrelemiyordu.

Bu testler KOŞULMADI (görev kuralı: pytest yasak) — yazıldı, PC/CI
doğrulaması bekliyor.
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

from core.mana_phase_combat import DEFAULT_COMBAT_SKILL_KEYS, ManaPhaseCombat
from core.skill_bar_reader import SkillBarReader, slot_bounds_fractions
from dataset_factory.exemplar_memory import ExemplarMemory


def _slot_window(frame: np.ndarray, reader: SkillBarReader, slot_key: str):
    """(y1, y2, sx1, sx2) — inspect_skill_bar() ile aynı Voronoi kenarlar."""
    fh, fw = frame.shape[:2]
    roi = reader.current_skill_bar_roi
    x1 = int(round(roi[0] * fw))
    y1 = int(round(roi[1] * fh))
    x2 = int(round(roi[2] * fw))
    y2 = int(round(roi[3] * fh))
    idx = list(reader.SKILL_SLOTS).index(slot_key)
    edges = slot_bounds_fractions()
    bw = float(x2 - x1)
    sx1 = x1 + int(round(edges[idx] * bw))
    sx2 = x1 + int(round(edges[idx + 1] * bw))
    return y1, y2, sx1, sx2


def _draw_ready_icon(frame: np.ndarray, reader: SkillBarReader, slot_key: str, seed: int = 42) -> None:
    """Parlak + doygun + gürültülü/detaylı hazır simge (test_w51 ile aynı reçete)."""
    y1, y2, sx1, sx2 = _slot_window(frame, reader, slot_key)
    sh = max(2, y2 - y1)
    sw = max(2, sx2 - sx1)
    idx = list(reader.SKILL_SLOTS).index(slot_key)
    rng = np.random.default_rng(seed + idx * 17)
    hsv = np.zeros((sh, sw, 3), dtype=np.uint8)
    hsv[:, :, 0] = 95
    hsv[:, :, 1] = 200
    grad = np.linspace(80, 245, sw, dtype=np.float32)
    noise = rng.normal(0.0, 25.0, (sh, sw))
    hsv[:, :, 2] = np.clip(np.tile(grad, (sh, 1)) + noise, 55, 255).astype(np.uint8)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.rectangle(patch, (3, 3), (sw - 4, sh - 4), (255, 210, 40), 2)
    frame[y1:y2, sx1:sx2] = patch


def _draw_glow_empty_slot(frame: np.ndarray, reader: SkillBarReader, slot_key: str) -> None:
    """CANLI DELİĞİN SENTEZİ: simgesiz ama parlak slot.

    Yumuşak mavi-mor gradyan (V 40→160, S 70): tüm-slot taban eşiklerini
    GEÇER (parlaklık ~100≥50, doygunluk 70≥45, kontrast ~27≥22, mavi ~1.0)
    ama merkezde simge kenarı yoktur (Sobel p95 ~19 < 40) ve canlı simge
    pikseli yoktur (S p95 70 < 90). Düzeltmesiz kodda ready=True çıkar.
    """
    y1, y2, sx1, sx2 = _slot_window(frame, reader, slot_key)
    sh = max(2, y2 - y1)
    sw = max(2, sx2 - sx1)
    hsv = np.zeros((sh, sw, 3), dtype=np.uint8)
    hsv[:, :, 0] = 100
    hsv[:, :, 1] = 70
    grad = np.linspace(40, 160, sw, dtype=np.float32)
    hsv[:, :, 2] = np.tile(grad, (sh, 1)).astype(np.uint8)
    frame[y1:y2, sx1:sx2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_dark_slot(frame: np.ndarray, reader: SkillBarReader, slot_key: str) -> None:
    """Düz koyu slot (beklemede/boş — eski 'koyu okunur' durumu)."""
    y1, y2, sx1, sx2 = _slot_window(frame, reader, slot_key)
    frame[y1:y2, sx1:sx2] = np.full((max(2, y2 - y1), max(2, sx2 - sx1), 3), 22, dtype=np.uint8)


def _reader(tmp_path: Path) -> SkillBarReader:
    mem = ExemplarMemory(json_path=tmp_path / "empty_exemplars.json")
    return SkillBarReader(
        calibrated_rois_path=tmp_path / "missing_rois.json",
        exemplar_memory=mem,
        cooldown_lockout_sec=0.15,
    )


def test_glow_empty_slots_never_ready(tmp_path: Path) -> None:
    """Parlak gradyanlı boş 6/7/8: empty=True, ready=False; seçim '2'dir."""
    reader = _reader(tmp_path)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_ready_icon(frame, reader, "2")
    for slot in ("6", "7", "8"):
        _draw_glow_empty_slot(frame, reader, slot)
    for slot in ("1", "3", "4", "5", "Q", "E", "R", "T"):
        _draw_dark_slot(frame, reader, slot)

    states = reader.inspect_skill_bar(frame)
    for slot in ("6", "7", "8"):
        assert states[slot]["empty"] is True, f"slot {slot} boş sayılmadı: {states[slot]}"
        assert states[slot]["ready"] is False, f"boş slot {slot} hazır sayıldı: {states[slot]}"
    assert states["2"]["ready"] is True, f"dolu slot 2 hazır sayılmadı: {states['2']}"
    assert states["2"]["empty"] is False

    assert "6" not in reader.get_ready_skills(frame)
    assert "7" not in reader.get_ready_skills(frame)
    assert "8" not in reader.get_ready_skills(frame)

    # '2' seçilir; kilitlenince sıradaki seçim "" olur — asla 6/7/8 değil.
    assert reader.select_best_ready_skill(frame, record_use=True) == "2"
    assert reader.select_best_ready_skill(frame, record_use=True) == ""


def test_flat_dark_empty_slot_is_empty(tmp_path: Path) -> None:
    """Düz koyu boş slot da empty=True / ready=False (eski durum korunur)."""
    reader = _reader(tmp_path)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_ready_icon(frame, reader, "3")
    for slot in reader.SKILL_SLOTS:
        if slot != "3":
            _draw_dark_slot(frame, reader, slot)

    states = reader.inspect_skill_bar(frame)
    assert states["6"]["empty"] is True
    assert states["6"]["ready"] is False
    assert states["3"]["ready"] is True
    assert states["3"]["empty"] is False


def test_mana_decide_drops_ready_keys_outside_configured_pool() -> None:
    """decide(): vizyon ready'si yapılandırılmış tuş kümesini aşamaz.

    knobs keys=1,2,3,4,q,e,r,t iken ready=['6','8','2'] gelse bile
    skill_keys içinde '6'/'8' olamaz (BURST ve REGEN fazında da).
    """
    combat = ManaPhaseCombat()
    combat.reset_rotation()

    burst = combat.decide(player_mp=1.0, player_hp=1.0, ready_skills=["6", "8", "2"], now=1000.0)
    assert "6" not in burst.skill_keys
    assert "8" not in burst.skill_keys
    assert set(burst.skill_keys) <= set(DEFAULT_COMBAT_SKILL_KEYS)

    regen = combat.decide(player_mp=0.2, player_hp=1.0, ready_skills=["6", "8"], now=2000.0)
    assert "6" not in regen.skill_keys
    assert "8" not in regen.skill_keys
    assert set(regen.skill_keys) <= set(DEFAULT_COMBAT_SKILL_KEYS)
