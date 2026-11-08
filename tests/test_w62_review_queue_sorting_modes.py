# -*- coding: utf-8 -*-
"""W62: ReviewQueueSorter sıralama ve onaylıları gizleme modları birim testleri."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.review_queue_sorter import ReviewQueueSorter, SortOrder


def test_sort_by_confidence_high_to_low() -> None:
    """Güven skoru yüksekten düşüğe (conf_desc) sıralama modunu doğrular."""
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    items = [
        {"frame_id": "frame_mid", "confidence": 0.72},
        {"frame_id": "frame_high", "boxes": [{"conf": 0.90}, {"conf": 0.96}]},  # avg: 0.93
        {"frame_id": "frame_low", "conf": 0.41},
        {"frame_id": "frame_tie_b", "confidence": 0.85},
        {"frame_id": "frame_tie_a", "confidence": 0.85},
    ]

    result = sorter.sort_queue(items, sort_by=SortOrder.CONF_DESC.value, only_pending=False)
    ordered_ids = [item["frame_id"] for item in result]

    assert ordered_ids == [
        "frame_high",
        "frame_tie_a",
        "frame_tie_b",
        "frame_mid",
        "frame_low",
    ]


def test_sort_by_confidence_low_to_high() -> None:
    """Güven skoru düşükten yükseğe (conf_asc) sıralama modunu (min güven bazlı) doğrular."""
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    items = [
        {"frame_id": "frame_high", "boxes": [{"conf": 0.88}, {"conf": 0.92}]},  # min: 0.88
        {"frame_id": "frame_lowest_box", "boxes": [{"conf": 0.91}, {"conf": 0.25}]},  # min: 0.25
        {"frame_id": "frame_mid", "min_conf": 0.54},
        {"frame_id": "frame_low", "confidence": 0.38},
    ]

    result = sorter.sort_queue(items, sort_by=SortOrder.CONF_ASC.value, only_pending=False)
    ordered_ids = [item["frame_id"] for item in result]

    assert ordered_ids == [
        "frame_lowest_box",
        "frame_low",
        "frame_mid",
        "frame_high",
    ]


def test_sort_by_suspicious_weak_count_desc() -> None:
    """Şüpheli/zayıf kutu sayısına göre azalan (weak_count_desc) sıralamayı ve eşitlik bozmayı doğrular."""
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    items = [
        {
            "frame_id": "frame_one_weak",
            "boxes": [{"conf": 0.45}, {"conf": 0.85}],  # 1 şüpheli, min=0.45
        },
        {
            "frame_id": "frame_three_weak",
            "boxes": [
                {"conf": 0.30},
                {"conf": 0.55},
                {"conf": 0.80, "suspicious": True},
            ],  # 3 şüpheli
        },
        {
            "frame_id": "frame_two_weak_higher_min",
            "boxes": [{"conf": 0.50}, {"conf": 0.58}],  # 2 şüpheli, min=0.50
        },
        {
            "frame_id": "frame_two_weak_lower_min",
            "boxes": [{"conf": 0.20}, {"conf": 0.55}],  # 2 şüpheli, min=0.20
        },
        {
            "frame_id": "frame_zero_weak",
            "boxes": [{"conf": 0.90}, {"conf": 0.75}],  # 0 şüpheli
        },
    ]

    result = sorter.sort_queue(items, sort_by=SortOrder.WEAK_COUNT_DESC.value, only_pending=False)
    ordered_ids = [item["frame_id"] for item in result]

    assert ordered_ids == [
        "frame_three_weak",
        "frame_two_weak_lower_min",
        "frame_two_weak_higher_min",
        "frame_one_weak",
        "frame_zero_weak",
    ]


def test_sort_by_suspicious_weak_count_asc() -> None:
    """Şüpheli/zayıf kutu sayısına göre artan (weak_count_asc) sıralamayı doğrular."""
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    items = [
        {"frame_id": "frame_3_suspicious", "suspicious_count": 3, "confidence": 0.40},
        {"frame_id": "frame_0_low_conf", "weak_count": 0, "confidence": 0.70},
        {"frame_id": "frame_0_high_conf", "weak_count": 0, "confidence": 0.95},
        {"frame_id": "frame_1_suspicious", "weak_count": 1, "confidence": 0.55},
    ]

    result = sorter.sort_queue(items, sort_by=SortOrder.WEAK_COUNT_ASC.value, only_pending=False)
    ordered_ids = [item["frame_id"] for item in result]

    # Eşit şüpheli sayısında yüksek güvenli olan önce gelir (-extract_confidence)
    assert ordered_ids == [
        "frame_0_high_conf",
        "frame_0_low_conf",
        "frame_1_suspicious",
        "frame_3_suspicious",
    ]


def test_hide_approved_items_in_pending_queue() -> None:
    """Onaylanmış karelerin (human_approved=True, done='ok', metadata) gizlendiğini doğrular."""
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    items = [
        {"frame_id": "pending_1", "weak_count": 2, "confidence": 0.45, "human_approved": False},
        {"frame_id": "approved_direct", "weak_count": 5, "confidence": 0.20, "human_approved": True},
        {"frame_id": "approved_done_ok", "weak_count": 4, "confidence": 0.30, "done": "OK"},
        {"frame_id": "pending_2", "weak_count": 3, "confidence": 0.35},
        {
            "frame_id": "approved_meta",
            "weak_count": 6,
            "confidence": 0.10,
            "metadata": {"human_approved": True},
        },
    ]

    # Varsayılan only_pending=True ile onaylılar gizlenmeli ve kalanlar şüpheli sayısına göre sıralanmalı
    pending_sorted = sorter.sort_queue(items, sort_by=SortOrder.WEAK_COUNT_DESC.value, only_pending=True)
    assert [item["frame_id"] for item in pending_sorted] == ["pending_2", "pending_1"]

    # only_pending=False verildiğinde onaylılar da dahil edilmeli
    all_sorted = sorter.sort_queue(items, sort_by=SortOrder.WEAK_COUNT_DESC.value, only_pending=False)
    assert len(all_sorted) == 5

    # Özet istatistik kontrolü
    summary = sorter.summarize_queue(items)
    assert summary == {
        "total_frames": 5,
        "pending_frames": 2,
        "approved_frames": 3,
        "pending_weak_boxes": 5,
    }
