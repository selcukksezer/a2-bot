# -*- coding: utf-8 -*-
"""R12 düzeltme testleri (F-0130 + F-1807).

F-0130: minimap filizindeki stack/harvest sayıları artık rozetten ölçülüyor
  (``_measure_stack_badge``); rozet yoksa stack_count=1 ölçülmüş değerdir.
F-1807: modül seviyesi ``select_natural_next_node`` sarmalayıcısı ``frame_size``
  alıp sınıf metoduna geçiriyor; dönüşüm artık gerçek kare boyutuyla yapılıyor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.behavioral_guard as bg
from core.behavioral_guard import BehavioralStealthGuard
from core.minimap_scanner import MinimapScanner

# Rozet yeşili: HSV H~75 (68..90 aralığında), S=255, V=200.
# (Mavi kanat reddi H>=82'de başlar; H~75 rozet o filtrenin dışındadır.)
BADGE_BGR = (100, 200, 0)


def _paste_tinted_sprout(crop_bgr: np.ndarray, top_left_x: int, top_left_y: int):
    """Kanonik şablondan yeşil-beyaz tonlu 16x16 filiz yapıştırır; merkez döndürür."""
    tpl = MinimapScanner._CANONICAL_SPROUT_TPL
    patch_bgr = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    leaf_pixels = tpl >= 175
    patch_bgr[leaf_pixels, 0] = np.clip(tpl[leaf_pixels].astype(np.int32) - 45, 130, 255).astype(np.uint8)
    patch_bgr[leaf_pixels, 1] = tpl[leaf_pixels]
    patch_bgr[leaf_pixels, 2] = np.clip(tpl[leaf_pixels].astype(np.int32) - 35, 140, 255).astype(np.uint8)
    crop_bgr[top_left_y : top_left_y + 16, top_left_x : top_left_x + 16] = patch_bgr
    return top_left_x + 8, top_left_y + 8


def _make_crop() -> np.ndarray:
    return np.full((257, 332, 3), (52, 62, 56), dtype=np.uint8)


def _scan_one_sprout(crop: np.ndarray):
    scanner = MinimapScanner()
    result = scanner.scan_minimap_crop(crop)
    gatherables = result["gatherable_nodes"]
    assert len(gatherables) == 1, f"Expected exactly 1 sprout, got: {gatherables}"
    return gatherables[0]


# ----------------------------------------------------------------------
# F-0130: rozet ölçümü
# ----------------------------------------------------------------------

def test_f0130_no_badge_stack_is_measured_one(tmp_path):
    """Rozet yoksa stack_count=1 ama artık ÖLÇÜLMÜŞ (measured=True), harvests=3."""
    crop = _make_crop()
    _paste_tinted_sprout(crop, 200, 80)
    sprout = _scan_one_sprout(crop)
    assert sprout["stack_count"] == 1
    assert sprout["stack_count_measured"] is True
    assert sprout["harvests_available"] == 3
    assert sprout["harvests_available_measured"] is True


def test_f0130_big_badge_measures_stack_three(tmp_path):
    """Filiz yanındaki büyük yeşil rozet (8x10) stack_count=3, harvests=9 ölçer."""
    crop = _make_crop()
    cx, cy = _paste_tinted_sprout(crop, 200, 80)
    cv2.rectangle(crop, (cx + 8, cy - 5), (cx + 15, cy + 4), BADGE_BGR, -1)
    sprout = _scan_one_sprout(crop)
    assert sprout["stack_count"] == 3
    assert sprout["stack_count_measured"] is True
    assert sprout["harvests_available"] == 9
    assert sprout["harvests_available_measured"] is True


def test_f0130_small_badge_measures_stack_two(tmp_path):
    """Küçük yeşil rozet (5x4 = 20px) stack_count=2, harvests=6 ölçer."""
    crop = _make_crop()
    cx, cy = _paste_tinted_sprout(crop, 200, 80)
    cv2.rectangle(crop, (cx + 8, cy - 2), (cx + 12, cy + 1), BADGE_BGR, -1)
    sprout = _scan_one_sprout(crop)
    assert sprout["stack_count"] == 2
    assert sprout["stack_count_measured"] is True
    assert sprout["harvests_available"] == 6
    assert sprout["harvests_available_measured"] is True


def test_f0130_measure_stack_badge_unit():
    """_measure_stack_badge: rozet yoksa (1, 0); kompakt rozet pikseli sayılır."""
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    assert MinimapScanner._measure_stack_badge(hsv, 50, 50, 1.0) == (1, 0)
    # 8x10 rozet bloğu (HSV H=75, S=255, V=200) ikon merkezinin sağına.
    hsv[45:55, 58:66] = (75, 255, 200)
    stack, badge_px = MinimapScanner._measure_stack_badge(hsv, 50, 50, 1.0)
    assert stack == 3
    assert badge_px == 80


def test_f0130_scattered_green_is_not_a_badge():
    """Dağınık yeşil pikseller (F-0055) rozet sayılmaz: stack 1 kalır."""
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(35, 65, size=(2, 40))
    hsv[ys, xs] = (75, 255, 200)
    stack, _ = MinimapScanner._measure_stack_badge(hsv, 50, 50, 1.0)
    assert stack == 1


# ----------------------------------------------------------------------
# F-1807: modül sarmalayıcısı frame_size geçiriyor
# ----------------------------------------------------------------------

def _guard_with_zero_rng() -> BehavioralStealthGuard:
    """Seçim çekişini 0.0'a sabitleyen guard: her zaman etkin mesafede ilk aday seçilir."""
    guard = BehavioralStealthGuard(seed=0)
    guard._rng = SimpleNamespace(random=lambda: 0.0)
    return guard


def test_f1807_module_wrapper_passes_frame_size(monkeypatch):
    """Aynı norm-koordinatlı düğümlerde kare boyu değişince en yakın aday değişir.

    A norm (0.95, 0.05), B norm (0.05, 0.95), oyuncu (0, 0):
    - 1920x1080 -> B yakın; 800x1200 -> A yakın (en-boy oranı dönüşümü değiştirir).
    Sarmalayıcı frame_size'i geçirmeseydi iki çağrı da 1920x1080 varsayımıyla
    aynı düğümü döndürürdü.
    """
    guard = _guard_with_zero_rng()
    monkeypatch.setattr(bg, "_get_default_guard", lambda: guard)
    node_a = {"id": 1, "norm_x": 0.95, "norm_y": 0.05}
    node_b = {"id": 2, "norm_x": 0.05, "norm_y": 0.95}
    ready = [node_a, node_b]
    seq = [node_a, node_b]

    wide = bg.select_natural_next_node(ready, seq, player_pos=(0.0, 0.0), frame_size=(1920.0, 1080.0))
    assert wide is node_b
    tall = bg.select_natural_next_node(ready, seq, player_pos=(0.0, 0.0), frame_size=(800.0, 1200.0))
    assert tall is node_a


def test_f1807_module_wrapper_default_is_1920x1080(monkeypatch):
    """frame_size verilmezse varsayılan 1920x1080 dönüşümü korunur (geriye uyumluluk)."""
    guard = _guard_with_zero_rng()
    monkeypatch.setattr(bg, "_get_default_guard", lambda: guard)
    node_a = {"id": 1, "norm_x": 0.95, "norm_y": 0.05}
    node_b = {"id": 2, "norm_x": 0.05, "norm_y": 0.95}
    chosen = bg.select_natural_next_node([node_a, node_b], [node_a, node_b], player_pos=(0.0, 0.0))
    assert chosen is node_b
