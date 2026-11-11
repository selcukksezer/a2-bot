# -*- coding: utf-8 -*-
"""D57: review-time hot-path improvements — behaviour locks.

Locks the behaviour of the three D57 modules against the e9794df baseline
while the implementations underneath get faster:

* ``dataset_factory/frame_time_machine.py`` — lookbacks resolved with
  ``bisect`` over a parallel timestamp index instead of a full O(n) scan.
* ``dataset_factory/confidence_tier_coloring.py`` — per-tier constant style
  fields precomputed once at import (``_TIER_STYLES``) instead of rebuilt
  (incl. hex parsing) on every ``get_box_visual_style`` call.
* ``dataset_factory/review_queue_sorter.py`` — one box scan per frame for
  multi-aggregate modes (``_frame_metrics``), dict fast paths instead of a
  per-key ``_get_field`` call, and lean single-aggregate extractors.

NOTE (conflict RESOLVED by fix operation R20b, fix/v2r-R20b):
``test_conf_desc_uses_average_while_conf_asc_uses_minimum`` in
``tests/test_w62_review_queue_sorting_modes.py`` locks CONF_DESC to the
AVERAGE, but commit 23a2b14 (D51) changed CONF_DESC to the MINIMUM
("symmetric with conf_asc"). The fix operation decided for the w62
contract — CONF_DESC ranks by AVERAGE, CONF_ASC by MINIMUM (asymmetric)
— and restored it in ``dataset_factory/review_queue_sorter.py``; the
``test_d57_conf_desc_ranks_by_average`` lock below was updated with that
decision, as this NOTE prescribed. The conflict was recorded in
``dev-results/D57.json`` as a found bug.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.confidence_tier_coloring import get_box_visual_style
from dataset_factory.frame_time_machine import PreGatherTimeMachineBuffer
from dataset_factory.review_queue_sorter import ReviewQueueSorter


# ---------------------------------------------------------------------------
# confidence_tier_coloring: precomputed templates, same output
# ---------------------------------------------------------------------------

def test_tier_templates_return_identical_style_per_tier() -> None:
    """Each tier keeps its exact colours/widths/badge text (w27 values)."""
    cases = [
        (
            # "trusted" (w27 sözleşmesi): "is_confident" girdisi üretimde
            # hiç okunmadı — D40 kendini-yükseltme engeli; kutu güvenilir
            # olduğunu kendisi iddia ederek yeşile çıkamaz.
            {"class_name": "herb_aria", "confidence": 0.97, "trusted": True},
            {"tier": "confident", "line_color": "#22c55e", "color_rgb": (34, 197, 94),
             "color_bgr": (94, 197, 34), "line_style": "solid", "pen_style": "SolidLine",
             "line_width": 2.2, "fill_alpha": 40, "badge_text": "✓ herb_aria %97",
             "status_text": "Emin · %97 güven"},
        ),
        (
            {"class_name": "ore_orichalcum", "confidence": 0.75},
            {"tier": "suspicious", "line_color": "#fb923c", "color_rgb": (251, 146, 60),
             "color_bgr": (60, 146, 251), "line_style": "dashed", "pen_style": "DashLine",
             "line_width": 1.8, "fill_alpha": 30, "badge_text": "🟠 şüpheli ore_orichalcum %75",
             "status_text": "Şüpheli · %75 güven"},
        ),
        (
            {"class_name": "herb_mela", "confidence": 0.42},
            {"tier": "low_confidence", "line_color": "#ef4444", "color_rgb": (239, 68, 68),
             "color_bgr": (68, 68, 239), "line_style": "dashed", "pen_style": "DashLine",
             "line_width": 1.8, "fill_alpha": 30, "badge_text": "şüpheli herb_mela %42",
             "status_text": "Düşük Güven (Şüpheli) · %42 güven"},
        ),
    ]
    for ann, expected in cases:
        style = get_box_visual_style(dict(ann))
        for key, value in expected.items():
            assert style[key] == value, (ann, key, style[key], value)
        assert style["conf_style"].startswith("background:")
        assert style["class_name"] == ann["class_name"]


def test_style_dict_is_a_fresh_copy_per_call() -> None:
    """Mutating a returned style must not corrupt the precomputed template."""
    # "trusted" (w27 sözleşmesi) — bkz. test_tier_templates... notu.
    ann = {"class_name": "herb_aria", "confidence": 0.97, "trusted": True}
    first = get_box_visual_style(dict(ann))
    first["line_color"] = "#000000"
    first["color_rgb"] = (0, 0, 0)
    second = get_box_visual_style(dict(ann))
    assert second["line_color"] == "#22c55e"
    assert second["color_rgb"] == (34, 197, 94)


# ---------------------------------------------------------------------------
# frame_time_machine: bisect lookups match the old linear scan exactly
# ---------------------------------------------------------------------------

def _make_buffer(n: int = 40, start: float = 1000.0, step: float = 0.2):
    buf = PreGatherTimeMachineBuffer(
        max_duration_sec=3600.0, min_push_interval_sec=0.0, max_capacity=64
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(n):
        assert buf.push_frame(frame, timestamp=start + i * step, force=True) is not None
    return buf


def _linear_frame_before(buf, target_ts):
    """The pre-D57 linear scan, kept here as the reference implementation."""
    snaps = list(buf._buffer)
    if not snaps:
        return None
    if target_ts < snaps[0].timestamp - 1e-6:
        return None
    latest = snaps[-1]
    if target_ts >= latest.timestamp:
        return latest
    best, best_diff = snaps[0], abs(snaps[0].timestamp - target_ts)
    for snap in snaps:
        diff = abs(snap.timestamp - target_ts)
        if diff <= best_diff:
            best_diff, best = diff, snap
    return best


def test_get_frame_before_matches_linear_scan() -> None:
    buf = _make_buffer()
    ref_ts = 1000.0 + 39 * 0.2
    for look in (0.0, 0.1, 0.5, 2.0, 2.5, 3.7, 7.8, 7.9, 8.0, 100.0):
        got = buf.get_frame_before(seconds_ago=look, reference_ts=ref_ts)
        want = _linear_frame_before(buf, ref_ts - look)
        assert (got.timestamp if got else None) == (want.timestamp if want else None)


def test_tie_resolves_to_later_snapshot() -> None:
    buf = _make_buffer()
    # NOTE (fix v5-W2, bulgu #5): eski yorum "hedef frame 10 ile 11'in tam
    # ortası" diyordu ama aritmetik öyle değil: hedef =
    # (1000.0 + 10*0.2 + 0.1) - 0.1 = 1002.0, yani TAM frame 10'un
    # timestamp'i — ortada tie yok. Referans linear scan da (_linear_frame_before:
    # frame 10 için diff 0.0) frame 10'u döndürür; eski frame 11 beklentisi
    # (1002.2) scan ile çelişiyordu. Doğru beklenti frame 10 (1002.0).
    # Tie->later davranışı zaten test_get_frame_before_matches_linear_scan
    # ile kilitli (look=0.1, ref 1007.8 -> hedef 1007.7, frame 38/39'un tam
    # ortası -> later kazanır ve geçer).
    got = buf.get_frame_before(seconds_ago=0.1, reference_ts=1000.0 + 10 * 0.2 + 0.1)
    assert got is not None and got.timestamp == pytest.approx(1000.0 + 10 * 0.2)


def test_multi_horizon_and_window_match_brute_force() -> None:
    buf = _make_buffer()
    ref_ts = 1000.0 + 39 * 0.2
    for horizons in ((2.5, 2.0, 1.2), (0.0, 0.0, 2.0), (7.8, 7.8, 0.2)):
        got = [s.timestamp for s in buf.get_multi_horizon_frames(horizons_sec=horizons, reference_ts=ref_ts)]
        seen: set[float] = set()
        want = []
        for h in horizons:
            snap = _linear_frame_before(buf, ref_ts - max(0.0, h))
            if snap is not None and snap.timestamp not in seen:
                seen.add(snap.timestamp)
                want.append(snap.timestamp)
        assert got == want
    for start_ago, end_ago in ((3.0, 1.0), (8.0, 0.0), (0.5, 0.5)):
        got = [s.timestamp for s in buf.get_window(start_ago, end_ago, reference_ts=ref_ts)]
        t_start, t_end = ref_ts - max(start_ago, end_ago), ref_ts - min(start_ago, end_ago)
        want = [s.timestamp for s in buf._buffer if (t_start - 1e-6) <= s.timestamp <= (t_end + 1e-6)]
        assert got == want


def test_empty_and_cleared_buffer_returns_none() -> None:
    buf = PreGatherTimeMachineBuffer(max_duration_sec=60.0, min_push_interval_sec=0.0)
    assert buf.get_frame_before(seconds_ago=2.0, reference_ts=1000.0) is None
    assert buf.get_multi_horizon_frames(reference_ts=1000.0) == []
    buf2 = _make_buffer(5)
    buf2.clear()
    assert buf2.get_frame_before(seconds_ago=1.0, reference_ts=1000.0) is None


def test_eviction_keeps_timestamp_index_in_sync() -> None:
    """Pushing past capacity/duration must not desync _timestamps from _buffer."""
    buf = PreGatherTimeMachineBuffer(max_duration_sec=2.0, min_push_interval_sec=0.0, max_capacity=8)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(20):
        buf.push_frame(frame, timestamp=2000.0 + i * 0.2, force=True)
    assert len(buf._timestamps) == len(buf._buffer)
    assert buf._timestamps == [s.timestamp for s in buf._buffer]
    latest = buf.get_frame_before(seconds_ago=0.0, reference_ts=2000.0 + 19 * 0.2)
    assert latest is not None and latest.timestamp == pytest.approx(2000.0 + 19 * 0.2)


# ---------------------------------------------------------------------------
# review_queue_sorter: extractors + sort orders
# ---------------------------------------------------------------------------

def _sorter_items():
    rng = random.Random(7)
    items = []
    for i in range(12):
        boxes = [{"conf": round(rng.uniform(0.1, 0.99), 3)} for _ in range(rng.randint(0, 5))]
        it = {"id": f"f{i:02d}", "boxes": boxes, "created_at": f"2026-11-0{1 + i % 8}T10:00:00"}
        if i % 3 == 0:
            it["weak_count"] = i
        if i % 4 == 0:
            it["confidence"] = round(rng.uniform(0.2, 0.95), 3)
        if i % 5 == 0:
            it["approved"] = True
        items.append(it)
    items.append({"id": "f12", "boxes": [{"conf": 0.2, "is_weak": True}, {"conf": 0.9, "status": "weak"}],
                  "created_at": "2026-11-02T09:00:00"})
    items.append({"id": "f13", "boxes": [], "created_at": "2026-11-03T09:00:00"})
    return items


def _old_sort(items, sorter, mode):
    """Pre-D57 sort keys, reimplemented here as the reference."""
    if mode in ("date_desc", "date", "newest"):
        return sorted(items, key=lambda x: (-sorter.extract_timestamp(x), sorter._get_tie_breaker(x)))
    if mode in ("date_asc", "oldest"):
        return sorted(items, key=lambda x: (sorter.extract_timestamp(x), sorter._get_tie_breaker(x)))
    if mode == "conf_desc":
        # R20b (d42db73) restored the w62 contract: CONF_DESC ranks by
        # AVERAGE, CONF_ASC by MINIMUM (asymmetric). The pre-D57 reference
        # used "min" here (D51 symmetry) and was stale after R20b changed
        # sort_queue to "avg" without updating this reference — that stale
        # "min" is what broke only the conf_desc parametrisation.
        return sorted(items, key=lambda x: (-sorter.extract_confidence(x, mode="avg"), sorter._get_tie_breaker(x)))
    if mode == "conf_asc":
        return sorted(items, key=lambda x: (sorter.extract_confidence(x, mode="min"), sorter._get_tie_breaker(x)))
    if mode in ("weak_count_desc", "suspicious_desc", "weak_desc"):
        return sorted(items, key=lambda x: (-sorter.extract_weak_count(x),
                                            sorter.extract_confidence(x, mode="min"),
                                            sorter._get_tie_breaker(x)))
    if mode in ("weak_count_asc", "suspicious_asc", "weak_asc"):
        return sorted(items, key=lambda x: (sorter.extract_weak_count(x),
                                            -sorter.extract_confidence(x),
                                            sorter._get_tie_breaker(x)))
    if mode in ("box_count_desc", "total_boxes_desc", "boxes_desc", "total_boxes", "box_count"):
        return sorted(items, key=lambda x: (-sorter.extract_total_boxes(x), sorter._get_tie_breaker(x)))
    if mode in ("box_count_asc", "total_boxes_asc", "boxes_asc"):
        return sorted(items, key=lambda x: (sorter.extract_total_boxes(x), sorter._get_tie_breaker(x)))
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", [
    "date_desc", "date_asc", "newest", "oldest", "conf_desc", "conf_asc",
    "weak_count_desc", "weak_count_asc", "suspicious_desc", "suspicious_asc",
    "box_count_desc", "box_count_asc", "total_boxes", "boxes_desc",
])
def test_all_sort_modes_match_old_keys(mode: str) -> None:
    sorter = ReviewQueueSorter()
    items = _sorter_items()
    got = [x["id"] for x in sorter.sort_queue(items, sort_by=mode, only_pending=False)]
    want = [x["id"] for x in _old_sort(items, sorter, mode)]
    assert got == want


def test_unknown_sort_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ReviewQueueSorter().sort_queue(_sorter_items(), sort_by="nope", only_pending=False)


def test_extractors_direct_field_precedence() -> None:
    sorter = ReviewQueueSorter()
    # min order: min_conf wins; avg/max order: confidence wins over min_conf.
    item = {"min_conf": 0.11, "max_conf": 0.99, "avg_conf": 0.5, "boxes": [{"conf": 0.7}]}
    assert sorter.extract_confidence(item, mode="min") == 0.11
    assert sorter.extract_confidence(item, mode="avg") == 0.5
    assert sorter.extract_confidence(item, mode="max") == 0.5
    # Box-derived values when no direct field exists; avg is bit-identical
    # to sum()/len() (no running-sum float drift).
    item2 = {"boxes": [{"confidence": 0.33}, {"score": 0.88}]}
    assert sorter.extract_confidence(item2, mode="min") == 0.33
    assert sorter.extract_confidence(item2, mode="max") == 0.88
    assert sorter.extract_confidence(item2, mode="avg") == (0.33 + 0.88) / 2
    # Non-numeric first key falls through to the next key.
    item3 = {"boxes": [{"conf": "high"}, {"confidence": 0.25}]}
    assert sorter.extract_confidence(item3, mode="avg") == 0.25
    # Empty / no boxes -> 0.0, never a crash.
    assert sorter.extract_confidence({"boxes": []}, mode="avg") == 0.0
    assert sorter.extract_confidence({}, mode="min") == 0.0


def test_extract_weak_count_flags_status_and_threshold() -> None:
    sorter = ReviewQueueSorter(weak_conf_threshold=0.60)
    assert sorter.extract_weak_count({"boxes": [{"conf": 0.2, "is_weak": True}, {"conf": 0.9, "status": "weak"}]}) == 2
    assert sorter.extract_weak_count({"boxes": [{"conf": 0.59}, {"conf": 0.60}, {"conf": 0.9}]}) == 1
    assert sorter.extract_weak_count({"boxes": [{"conf": 0.5, "suspicious": True}]}) == 1
    assert sorter.extract_weak_count({"weak_count": 4, "boxes": [{"conf": 0.1}]}) == 4
    assert sorter.extract_weak_count({"boxes": [{"conf": 0.5, "tier": "review"}]}) == 1


def test_extract_total_boxes_direct_field_and_len() -> None:
    sorter = ReviewQueueSorter()
    assert sorter.extract_total_boxes({"total_boxes": 7, "boxes": [{"conf": 0.5}]}) == 7
    assert sorter.extract_total_boxes({"count": 3, "boxes": []}) == 3
    assert sorter.extract_total_boxes({"boxes": [{"conf": 0.1}, {"conf": 0.2}]}) == 2
    assert sorter.extract_total_boxes({"boxes": [], "detections": [{"conf": 0.1}]}) == 1


def test_object_style_items_still_work() -> None:
    sorter = ReviewQueueSorter()
    item = SimpleNamespace(
        boxes=[SimpleNamespace(conf=0.2), SimpleNamespace(conf=0.9, status="weak")],
        frame_id="obj1",
    )
    assert sorter.extract_weak_count(item) == 2
    assert sorter.extract_confidence(item, mode="min") == 0.2
    assert sorter.extract_total_boxes(item) == 2


def test_d57_conf_desc_ranks_by_average() -> None:
    """Restored contract (fix operation R20b): CONF_DESC ranks by AVERAGE,
    CONF_ASC by MINIMUM — asymmetric. D51 (23a2b14) had made CONF_DESC
    rank by minimum, symmetric with CONF_ASC; that conflicted with the
    w62 lock (test_conf_desc_uses_average_while_conf_asc_uses_minimum)
    and was resolved here in favour of the w62 contract. See the module
    docstring."""
    sorter = ReviewQueueSorter()
    items = [
        {"frame_id": "frame_x", "boxes": [{"conf": 0.95}, {"conf": 0.55}]},
        {"frame_id": "frame_y", "boxes": [{"conf": 0.70}, {"conf": 0.70}]},
        {"frame_id": "frame_z", "boxes": [{"conf": 0.99}, {"conf": 0.99}]},
    ]
    desc = sorter.sort_queue(items, sort_by="conf_desc", only_pending=False)
    assert [i["frame_id"] for i in desc] == ["frame_z", "frame_x", "frame_y"]
    asc = sorter.sort_queue(items, sort_by="conf_asc", only_pending=False)
    assert [i["frame_id"] for i in asc] == ["frame_x", "frame_y", "frame_z"]


def test_summarize_queue_single_pass_matches_old_counts() -> None:
    sorter = ReviewQueueSorter()
    items = [
        {"frame_id": "pending_1", "weak_count": 2, "confidence": 0.45, "human_approved": False},
        {"frame_id": "approved_direct", "weak_count": 5, "confidence": 0.20, "human_approved": True},
        {"frame_id": "approved_done_ok", "weak_count": 4, "confidence": 0.30, "done": "OK"},
        {"frame_id": "pending_2", "weak_count": 3, "confidence": 0.35},
    ]
    assert sorter.summarize_queue(items) == {
        "total_frames": 4,
        "pending_frames": 2,
        "approved_frames": 2,
        "pending_weak_boxes": 5,
    }
