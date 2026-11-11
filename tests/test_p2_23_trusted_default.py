# -*- coding: utf-8 -*-
"""
tests/test_p2_23_trusted_default.py

P2-23 (Y002-K1) hedefli testleri — `trusted` varsayılanı ve geometrisiz kutu:

- `trusted` alanı olmayan kutu varsayılan `trusted=False` sayılır
  (eskiden `weak` bayrağına bakılıp yoksa True varsayılıyordu).
- Geometrisi eksik kutu `valid=False` ile işaretlenir; tier sayacına
  girmez, çizilmez (şablon 0.5/0.5/0.08 "target" hayalet kutu yok).
- Yüksek güvenli ama `trusted` alanı olmayan kutu "confident"
  sayılmaz (0.85 eşiğini geçse bile trusted=False ise confident değil).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.yolo_panel import get_annotation_tier_summary, normalize_annotation


def _box(**overrides):
    base = {
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "class_id": 0,
        "class_name": "monster",
        "confidence": 0.95,
    }
    base.update(overrides)
    return base


def test_normalize_missing_trusted_defaults_false():
    """`trusted` alanı yoksa False — `weak` yokluğu artık True yapmaz."""
    norm = normalize_annotation(_box())
    assert norm["trusted"] is False
    assert norm["valid"] is True


def test_normalize_missing_trusted_with_weak_false_still_false():
    """`weak=False` olsa bile `trusted` alanı yoksa False sayılır."""
    norm = normalize_annotation(_box(weak=False))
    assert norm["trusted"] is False


def test_normalize_explicit_trusted_true_kept():
    """Açık `trusted=True` aynen korunur."""
    norm = normalize_annotation(_box(trusted=True))
    assert norm["trusted"] is True
    assert norm["valid"] is True


def test_normalize_missing_geometry_is_invalid():
    """Geometrisi eksik kutu valid=False — şablon geometri uydurulmaz."""
    box = _box(trusted=True)
    del box["cx"]
    norm = normalize_annotation(box)
    assert norm["valid"] is False
    box2 = _box(cx=None, trusted=True)
    assert normalize_annotation(box2)["valid"] is False
    box3 = _box(w=0.0, trusted=True)
    assert normalize_annotation(box3)["valid"] is False


def test_normalize_short_tuple_is_invalid_and_untrusted():
    """4'ten kısa tuple hayalet kutu değildir: valid=False, trusted=False."""
    norm = normalize_annotation((0.5, 0.5))
    assert norm["valid"] is False
    assert norm["trusted"] is False


def test_normalize_full_tuple_is_valid_and_untrusted():
    """Tam tuple (sınıf+geometri+conf) valid=True ama trusted=False."""
    norm = normalize_annotation((0, 0.5, 0.5, 0.2, 0.2, 0.95))
    assert norm["valid"] is True
    assert norm["trusted"] is False


def test_tier_summary_untrusted_high_conf_is_not_confident():
    """Yüksek güvenli ama trusted'sız kutu confident sayılmaz."""
    summary = get_annotation_tier_summary([_box(confidence=0.95)])
    assert summary["confident"] == 0
    # 0.95 >= 0.60 olduğu için suspicious'a düşer.
    assert summary["suspicious"] == 1
    assert summary["low"] == 0


def test_tier_summary_trusted_high_conf_is_confident():
    """Yüksek güvenli + trusted kutu confident sayılır."""
    summary = get_annotation_tier_summary([_box(confidence=0.95, trusted=True)])
    assert summary["confident"] == 1
    assert summary["suspicious"] == 0
    assert summary["low"] == 0


def test_tier_summary_geometryless_box_not_counted():
    """Geometrisi eksik kutu hiçbir tier'a girmez."""
    box = _box(confidence=0.95, trusted=True)
    del box["w"]
    summary = get_annotation_tier_summary([box])
    assert summary["confident"] == 0
    assert summary["suspicious"] == 0
    assert summary["low"] == 0
