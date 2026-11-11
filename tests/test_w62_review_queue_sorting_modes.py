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


@pytest.fixture
def sorter() -> ReviewQueueSorter:
    """Shared sorter with the suite-wide weak threshold (0.60).

    The sorter is stateless apart from its threshold, so one instance per
    test is enough — the per-test construction it replaces was pure noise.
    """
    return ReviewQueueSorter(weak_conf_threshold=0.60)


def test_sort_by_confidence_high_to_low(sorter: ReviewQueueSorter) -> None:
    """Güven skoru yüksekten düşüğe (conf_desc) sıralama modunu doğrular."""
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


def test_sort_by_confidence_low_to_high(sorter: ReviewQueueSorter) -> None:
    """Güven skoru düşükten yükseğe (conf_asc) sıralama modunu (min güven bazlı) doğrular."""
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


def test_sort_by_suspicious_weak_count_desc(sorter: ReviewQueueSorter) -> None:
    """Şüpheli/zayıf kutu sayısına göre azalan (weak_count_desc) sıralamayı ve eşitlik bozmayı doğrular."""
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


def test_sort_by_suspicious_weak_count_asc(sorter: ReviewQueueSorter) -> None:
    """Şüpheli/zayıf kutu sayısına göre artan (weak_count_asc) sıralamayı doğrular."""
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


def test_hide_approved_items_in_pending_queue(sorter: ReviewQueueSorter) -> None:
    """Onaylanmış karelerin (human_approved=True, done='ok', metadata) gizlendiğini doğrular."""
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


def test_conf_desc_uses_average_while_conf_asc_uses_minimum(sorter: ReviewQueueSorter) -> None:
    """CONF_DESC ortalama, CONF_ASC minimum güven kullanır — asimetri kanıtlanır.

    frame_x kutuları [0.95, 0.55]: ortalama 0.75, minimum 0.55.
    frame_y kutuları [0.70, 0.70]: ortalama 0.70, minimum 0.70.
    CONF_DESC (ortalama): z(0.99) > x(0.75) > y(0.70).
    CONF_ASC (minimum): x(0.55) < y(0.70) < z(0.99).
    CONF_ASC ortalama kullansaydı sıra y, x, z olurdu — bu test onu yakalar.
    """
    items = [
        {"frame_id": "frame_x", "boxes": [{"conf": 0.95}, {"conf": 0.55}]},
        {"frame_id": "frame_y", "boxes": [{"conf": 0.70}, {"conf": 0.70}]},
        {"frame_id": "frame_z", "boxes": [{"conf": 0.99}, {"conf": 0.99}]},
    ]

    # Doğrudan çıkarma: aynı karede ortalama != minimum.
    assert sorter.extract_confidence(items[0]) == pytest.approx(0.75)
    assert sorter.extract_confidence(items[0], mode="min") == pytest.approx(0.55)
    assert sorter.extract_confidence(items[0], mode="avg") == pytest.approx(0.75)

    desc = sorter.sort_queue(items, sort_by=SortOrder.CONF_DESC.value, only_pending=False)
    assert [item["frame_id"] for item in desc] == ["frame_z", "frame_x", "frame_y"]

    asc = sorter.sort_queue(items, sort_by=SortOrder.CONF_ASC.value, only_pending=False)
    assert [item["frame_id"] for item in asc] == ["frame_x", "frame_y", "frame_z"]

    # Asimetri: CONF_ASC sırası, CONF_DESC sırasının tersi değildir.
    assert [item["frame_id"] for item in asc] != list(reversed([item["frame_id"] for item in desc]))


def test_sort_by_date_desc_and_asc_unknown_date_sorts_last(sorter: ReviewQueueSorter) -> None:
    """DATE_DESC/DATE_ASC: epoch timestamp'larla sıralar; tarihi çözülemeyen
    kare (tarih yok) HER İKİ yönde de en sona düşer (epoch 0 değildir).

    Bu modlar daha önce hiç test edilmiyordu — sort_queue'un date dalı
    kapsama dışıydı.
    """
    items = [
        {"frame_id": "frame_new", "timestamp": 300},
        {"frame_id": "frame_old", "timestamp": 100},
        {"frame_id": "frame_nodate", "weak_count": 0},
        {"frame_id": "frame_mid", "timestamp": 200},
    ]

    desc = sorter.sort_queue(items, sort_by=SortOrder.DATE_DESC.value, only_pending=False)
    assert [item["frame_id"] for item in desc] == [
        "frame_new",
        "frame_mid",
        "frame_old",
        "frame_nodate",
    ]

    asc = sorter.sort_queue(items, sort_by=SortOrder.DATE_ASC.value, only_pending=False)
    assert [item["frame_id"] for item in asc] == [
        "frame_old",
        "frame_mid",
        "frame_new",
        "frame_nodate",
    ]


def test_sort_by_box_count_desc_and_asc_and_total_boxes_alias(sorter: ReviewQueueSorter) -> None:
    """BOX_COUNT_DESC/ASC: kutu sayısına göre sıralar; TOTAL_BOXES_DESC
    box_count_desc ile aynı sırayı verir (alias). Bu modlar da daha önce
    hiç test edilmiyordu."""
    items = [
        {"frame_id": "frame_one", "boxes": [{"conf": 0.5}]},
        {"frame_id": "frame_three", "boxes": [{"conf": 0.9}, {"conf": 0.8}, {"conf": 0.7}]},
        {"frame_id": "frame_zero", "boxes": []},
        {"frame_id": "frame_two", "boxes": [{"conf": 0.6}, {"conf": 0.6}]},
    ]

    desc = sorter.sort_queue(items, sort_by=SortOrder.BOX_COUNT_DESC.value, only_pending=False)
    assert [item["frame_id"] for item in desc] == [
        "frame_three",
        "frame_two",
        "frame_one",
        "frame_zero",
    ]

    asc = sorter.sort_queue(items, sort_by=SortOrder.BOX_COUNT_ASC.value, only_pending=False)
    assert [item["frame_id"] for item in asc] == [
        "frame_zero",
        "frame_one",
        "frame_two",
        "frame_three",
    ]

    total_desc = sorter.sort_queue(
        items, sort_by=SortOrder.TOTAL_BOXES_DESC.value, only_pending=False
    )
    assert [item["frame_id"] for item in total_desc] == [
        "frame_three",
        "frame_two",
        "frame_one",
        "frame_zero",
    ]


def test_empty_queue_sorts_to_empty_and_summarizes_to_zeros(sorter: ReviewQueueSorter) -> None:
    """Boş kuyruk: sort_queue [] döner, summarize_queue sıfırlarla döner."""
    assert sorter.sort_queue([], sort_by=SortOrder.CONF_DESC.value, only_pending=False) == []
    assert sorter.sort_queue([], only_pending=True) == []
    assert sorter.summarize_queue([]) == {
        "total_frames": 0,
        "pending_frames": 0,
        "approved_frames": 0,
        "pending_weak_boxes": 0,
    }


def test_unsupported_sort_mode_raises_value_error(sorter: ReviewQueueSorter) -> None:
    """Desteklenmeyen sıralama modu sessizce varsayılana düşmez — ValueError
    fırlatır ve mesaj desteklenen modları listeler."""
    items = [{"frame_id": "frame_a", "confidence": 0.5}]
    with pytest.raises(ValueError, match="Desteklenmeyen sıralama modu"):
        sorter.sort_queue(items, sort_by="bogus_mode", only_pending=False)


def test_extract_confidence_and_counts_without_any_confidence_data(
    sorter: ReviewQueueSorter,
) -> None:
    """Hiç güven verisi olmayan kare: extract_confidence her modda 0.0,
    extract_weak_count 0, extract_total_boxes 0, extract_timestamp inf
    (çözülemeyen tarih epoch 0 değil, sıralamada en sona düşer)."""
    item = {"frame_id": "frame_bare"}
    assert sorter.extract_confidence(item) == 0.0
    assert sorter.extract_confidence(item, mode="min") == 0.0
    assert sorter.extract_confidence(item, mode="max") == 0.0
    assert sorter.extract_weak_count(item) == 0
    assert sorter.extract_total_boxes(item) == 0
    assert sorter.extract_timestamp(item) == float("inf")
    assert sorter.extract_timestamp({"frame_id": "x", "timestamp": "not-a-date"}) == float("inf")


def test_is_approved_accepts_case_insensitive_done_and_string_true(
    sorter: ReviewQueueSorter,
) -> None:
    """is_approved: done='Ok' (karışık büyüklük), done=True ve
    human_approved='true' (string) da onay sayılır — filter_pending'in
    bu varyantları daha önce kilitli değildi."""
    assert sorter.is_approved({"frame_id": "a", "done": "Ok"}) is True
    assert sorter.is_approved({"frame_id": "b", "done": True}) is True
    assert sorter.is_approved({"frame_id": "c", "human_approved": "true"}) is True
    assert sorter.is_approved({"frame_id": "d", "done": "pending"}) is False
    assert sorter.is_approved({"frame_id": "e"}) is False


def test_default_sort_mode_is_weak_count_desc(sorter: ReviewQueueSorter) -> None:
    """sort_by verilmezse varsayılan weak_count_desc çalışır: zayıf kutusu
    olan önce gelir; eşitlikte min güveni düşük olan önce gelir."""
    items = [
        {"frame_id": "frame_clean_high", "boxes": [{"conf": 0.70}]},
        {"frame_id": "frame_weak", "boxes": [{"conf": 0.50}]},
        {"frame_id": "frame_clean_low", "boxes": []},
    ]
    result = sorter.sort_queue(items, only_pending=False)
    assert [item["frame_id"] for item in result] == [
        "frame_weak",
        "frame_clean_low",
        "frame_clean_high",
    ]
