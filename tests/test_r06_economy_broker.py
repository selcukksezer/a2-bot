# -*- coding: utf-8 -*-
"""
R06 — core/economy_broker.py dilim testleri
===========================================
- S-0702: scan_broker_window tarama sayaci + price_history kaydi
  (F-0014 durustluk siniri korunur: fiyat olculmez/degistirilmez).
- S-0701: reset_session raw_inventory + pending_queue temizligi.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

from core.economy_broker import (  # noqa: E402
    AutoCraftingMorphQueue,
    SessionProfitSimulator,
    TradeBrokerPriceScanner,
)


def _make_scanner(tmp_path):
    return TradeBrokerPriceScanner(price_db_path=str(tmp_path / "market_prices.json"))


def _history_total(scanner):
    return sum(len(v) for v in scanner.price_history.values())


# ---------------------------------------------------------------------------
# S-0702
# ---------------------------------------------------------------------------

def test_scan_broker_window_counts_scan_and_writes_history(tmp_path):
    """Referans karede tarama: scan_count artar, history buyur, fiyatlar degismez."""
    ref_path = os.path.join(PROJECT_ROOT, "reference_screenshots", "pilgrims_ridge_1.jpg")
    if not os.path.isfile(ref_path):
        pytest.skip("referans ekran goruntusu yok: %s" % ref_path)
    frame = cv2.imdecode(np.fromfile(ref_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None and frame.size > 0

    scanner = _make_scanner(tmp_path)
    prices_before = dict(scanner.prices)
    assert scanner.scan_count == 0
    assert _history_total(scanner) == 0

    scanner.scan_broker_window(frame)

    assert scanner.scan_count == 1
    assert _history_total(scanner) > 0
    assert "__scan__" in scanner.price_history
    assert scanner.price_history["__scan__"][-1]["scan_count"] == 1
    # F-0014: fiyatlar olculmedi/de gistirilmedi.
    assert dict(scanner.prices) == prices_before


def test_scan_broker_window_invalid_frame_does_not_count(tmp_path):
    """Gecersiz/bos karede tarama yapilmis sayilmaz: sayac artmaz, kayit yazilmaz."""
    scanner = _make_scanner(tmp_path)
    scanner.scan_broker_window(None)
    scanner.scan_broker_window(np.zeros((120, 200, 3), dtype=np.uint8))
    assert scanner.scan_count == 0
    assert _history_total(scanner) == 0


def test_scan_broker_window_detected_material_history_entry(tmp_path):
    """Tespit edilen materyal icin onbellek fiyatli, olculmemis history kaydi duser."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(360, 640, 3)).astype(np.uint8)
    # Aria HSV imzasi (35,85,115)-(55,225,225) icinde kalan renk.
    aria_bgr = cv2.cvtColor(
        np.uint8([[[45, 150, 180]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    # ROI (0.20,0.16)-(0.84,0.84): x1=128, y1=58; 1. satir ikon alani
    # broker_roi icinde satir 34..56, sutun 16..66 -> mutlak y 92..114, x 144..194.
    frame[92:114, 144:194] = aria_bgr

    scanner = _make_scanner(tmp_path)
    prices_before = dict(scanner.prices)
    scanner.scan_broker_window(frame)

    assert scanner.scan_count == 1
    assert "Aria" in scanner.last_scan_detected
    assert "Aria" in scanner.price_history
    entry = scanner.price_history["Aria"][-1]
    assert entry["price"] == prices_before["Aria"]
    assert entry["source"] == "cache"
    assert entry["measured"] is False
    assert dict(scanner.prices) == prices_before


# ---------------------------------------------------------------------------
# S-0701
# ---------------------------------------------------------------------------

def test_reset_session_clears_raw_inventory_and_pending_queue(tmp_path):
    """Bulgu probu: 25x Orichalcum Ore sonrasi reset tum oturum durumunu sifirlar."""
    scanner = _make_scanner(tmp_path)
    morph_q = AutoCraftingMorphQueue(default_threshold=20, price_scanner=scanner)
    sim = SessionProfitSimulator(price_scanner=scanner, crafting_queue=morph_q)

    morph_q.record_gathered_material("Orichalcum Ore", count=25)
    sim.record_gather("Pure Aether", count=12)
    sim.record_gather("Aria", count=18)
    summary_before = sim.get_session_economy_summary(elapsed_hours=0.5)
    assert summary_before.get("net_kinah_earned", 0) > 0
    assert morph_q.raw_inventory.get("Orichalcum Ore", 0) == 25
    assert len(morph_q.pending_queue) >= 1

    sim.reset_session()

    assert sim.gathered_counts == {}
    assert morph_q.raw_inventory == {}
    assert morph_q.pending_queue == []
    assert morph_q.crafting_history == []
    assert morph_q.crafted_inventory == {}
    summary_after = sim.get_session_economy_summary(elapsed_hours=0.5)
    assert summary_after.get("net_kinah_earned", 0) == 0
    assert summary_after.get("total_items_gathered", 0) == 0
