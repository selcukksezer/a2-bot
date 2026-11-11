# -*- coding: utf-8 -*-
"""D31 DEV improvements: tests locking the new behaviour.

Covers:
- get_live_metrics single-pass aggregation == old three-pass formulas
- _compute_hourly_buckets index assignment == old per-event bucket scan
  (incl. exact-boundary, window-end and out-of-window events)
- InventoryFullnessEstimator template cache + boolean-mask classification
- YieldHeatmapTracker shared cluster helper behaviour
- Real diagnostic/health snapshots + exposed sample counter
- Module-level sleep_gaussian_jitter delegating to the engine method
"""
import time
from datetime import datetime

import numpy as np
import pytest

from core.human_timing import (
    DynamicBreakScheduler,
    HumanTimingEngine,
    sleep_gaussian_jitter as module_sleep_gaussian_jitter,
)
from core.telemetry import (
    InventoryFullnessEstimator,
    TelemetryEngine,
    YieldHeatmapTracker,
)
import core.telemetry as telemetry_mod


# --- helpers ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_lifetime_disk_writes(monkeypatch):
    """record_gather writes lifetime stats to config/lifetime_stats.json;
    stub it so tests never touch the worktree."""
    class _Stub:
        def record_material(self, *a, **k):
            return None

        def record_monster_kill(self, *a, **k):
            return None

    import core.lifetime_stats as _ls
    monkeypatch.setattr(_ls, "get_lifetime_stats", lambda: _Stub())


def _seeded_engine(n_gathers=120, n_failures=7, n_combats=3):
    eng = TelemetryEngine(bucket_span_sec=300, num_buckets=12)
    now = time.time()
    eng.session_start_ts = now - 3600.0  # elapsed >> 2*span -> span stays 300
    for i in range(n_gathers):
        eng.record_gather(
            node_id=i % 9,
            region="R%d" % (i % 3),
            route="Route%d" % (i % 2),
            resource_type="Ore%d" % (i % 2),
            walk_duration_sec=1.0 + (i % 5) * 0.5,
            match_pct=70 + (i % 31),
        )
    for i in range(n_failures):
        eng.record_failure(node_id=i, reason="timeout" if i % 2 == 0 else "combat")
    for i in range(n_combats):
        eng.record_combat(duration_sec=5.0 + i, outcome="CLEARED")
    return eng, now


def _old_live_aggregates(eng):
    """Reference: the pre-D31 three-pass formulas, recomputed from events."""
    gathered = len(eng.gather_events)
    if gathered > 0:
        total_walk = sum(float(e.get("walk_duration_sec", 0.0)) for e in eng.gather_events)
        avg_walk = round(total_walk / gathered, 2)
        avg_match = round(
            sum(int(e.get("match_pct", 0)) for e in eng.gather_events) / gathered, 1
        )
    else:
        avg_walk, avg_match = 0.0, 0.0
    breakdown = {}
    for g in eng.gather_events:
        res = g.get("resource_type", "Orichalcum Ore")
        reg = g.get("region", "Unknown")
        rte = g.get("route", "Default Route")
        key = "%s||%s||%s" % (res, reg, rte)
        if key not in breakdown:
            breakdown[key] = {"count": 0, "total_walk_sec": 0.0, "total_match_pct": 0}
        breakdown[key]["count"] += 1
        breakdown[key]["total_walk_sec"] += float(g.get("walk_duration_sec", 0.0))
        breakdown[key]["total_match_pct"] += int(g.get("match_pct", 80))
    return avg_walk, avg_match, breakdown


# --- I1: get_live_metrics single pass --------------------------------------

def test_live_metrics_single_pass_matches_old_formulas():
    eng, _ = _seeded_engine()
    metrics = eng.get_live_metrics()
    avg_walk, avg_match, breakdown = _old_live_aggregates(eng)
    assert metrics["gathered_total"] == 120
    assert metrics["failed_total"] == 7
    assert metrics["combat_count"] == 3
    assert metrics["avg_walk_sec"] == avg_walk
    assert metrics["avg_match_pct"] == avg_match
    assert metrics["success_rate_pct"] == round(120 / 127 * 100.0, 2)
    rows = {(r["resource_type"], r["region"], r["route"]): r
            for r in metrics["detailed_breakdown"]}
    assert len(rows) == len(breakdown)
    for key, old in breakdown.items():
        res, reg, rte = key.split("||")
        row = rows[(res, reg, rte)]
        assert row["count"] == old["count"]
        assert row["avg_walk_sec"] == round(old["total_walk_sec"] / old["count"], 2)
        assert row["avg_match_pct"] == round(old["total_match_pct"] / old["count"], 1)
        assert row["share_pct"] == round(old["count"] / 120 * 100.0, 1)
    # Rows still sorted by count desc.
    counts = [r["count"] for r in metrics["detailed_breakdown"]]
    assert counts == sorted(counts, reverse=True)


def test_live_metrics_empty_session_unchanged():
    eng = TelemetryEngine()
    m = eng.get_live_metrics()
    assert m["gathered_total"] == 0
    assert m["avg_walk_sec"] == 0.0
    assert m["avg_match_pct"] == 0.0
    assert m["success_rate_pct"] is None
    assert m["detailed_breakdown"] == []
    assert m["hourly_buckets"] and all(b["count"] == 0 for b in m["hourly_buckets"])


# --- I1: hourly buckets ------------------------------------------------------

def _old_buckets(eng, now):
    """Reference: the pre-D31 per-event bucket scan."""
    span = float(eng.bucket_span_sec)
    elapsed = max(0.001, now - eng.session_start_ts)
    eff = span if elapsed >= span * 2 else max(5.0, elapsed / float(eng.num_buckets))
    buckets = []
    for i in range(eng.num_buckets):
        start_t = now - (eng.num_buckets - i) * eff
        end_t = start_t + eff
        buckets.append({"index": i, "start_ts": start_t, "end_ts": end_t,
                        "count": 0, "failures": 0, "combats": 0})
    for ev in eng.gather_events:
        ts = ev.get("timestamp", 0.0)
        for b in buckets:
            if b["start_ts"] <= ts <= b["end_ts"]:
                b["count"] += 1
                break
    for fv in eng.failure_events:
        ts = fv.get("timestamp", 0.0)
        for b in buckets:
            if b["start_ts"] <= ts <= b["end_ts"]:
                b["failures"] += 1
                break
    for cv in eng.combat_events:
        ts = cv.get("timestamp", 0.0)
        for b in buckets:
            if b["start_ts"] <= ts <= b["end_ts"]:
                b["combats"] += 1
                break
    return buckets, eff


def test_hourly_buckets_match_old_scan_including_boundaries():
    eng = TelemetryEngine(bucket_span_sec=300, num_buckets=12)
    now = time.time()
    eng.session_start_ts = now - 3600.0
    window_start = now - 12 * 300.0
    # Events on every shared boundary, mid-bucket, before/after the window,
    # and exactly at the window end (old loop counts it in the last bucket).
    ts_list = [window_start + k * 300.0 for k in range(13)]
    ts_list += [window_start + k * 300.0 + 137.5 for k in range(12)]
    ts_list += [window_start - 1.0, now + 60.0, window_start, now]
    for i, ts in enumerate(ts_list):
        eng.gather_events.append({"timestamp": ts, "node_id": i})
        if i % 5 == 0:
            eng.failure_events.append({"timestamp": ts, "node_id": i})
        if i % 7 == 0:
            eng.combat_events.append({"timestamp": ts, "duration_sec": 1.0})
    new = eng._compute_hourly_buckets(now)
    old, eff = _old_buckets(eng, now)
    assert len(new) == 12
    for nb, ob in zip(new, old):
        assert nb["start_ts"] == ob["start_ts"]
        assert nb["end_ts"] == ob["end_ts"]
        assert nb["count"] == ob["count"]
        assert nb["failures"] == ob["failures"]
        assert nb["combats"] == ob["combats"]
        assert nb["rate_per_hour"] == round((ob["count"] / eff) * 3600.0, 2)
    # Boundary events land in the earlier bucket (first-match rule): bucket 0
    # holds window_start (x2: boundary + duplicate), boundary k=1 and mid k=0;
    # bucket 11 holds `now` (x2: boundary + duplicate) and mid k=11.
    assert new[0]["count"] == 4
    assert new[11]["count"] == 3
    # Out-of-window events are skipped entirely.
    assert sum(b["count"] for b in new) == len(ts_list) - 2


# --- I2: inventory template cache + glyph classification --------------------

def test_digit_templates_compiled_once_and_shared():
    a = InventoryFullnessEstimator()
    b = InventoryFullnessEstimator()
    assert a._compiled_templates is b._compiled_templates
    assert a._compiled_template_masks is b._compiled_template_masks
    assert set(a._compiled_templates) == set("0123456789")
    # Mutating one instance's view would affect the other (shared read-only
    # cache) — the classify path only reads it.


def test_classify_glyph_identifies_every_template_digit():
    est = InventoryFullnessEstimator()
    for digit, rows in InventoryFullnessEstimator._HUD_DIGIT_TEMPLATES.items():
        glyph = np.array(
            [[255 if ch == "#" else 0 for ch in row] for row in rows],
            dtype=np.uint8,
        )
        got, conf = est._classify_glyph(glyph)
        assert got == digit
        assert conf == pytest.approx(1.0)


def test_classify_glyph_rejects_garbage_below_threshold_via_masks():
    est = InventoryFullnessEstimator()
    rng = np.random.RandomState(0)
    glyph = (rng.rand(8, 5) * 255).astype(np.uint8)
    got, conf = est._classify_glyph(glyph)
    assert got in "0123456789"
    assert 0.0 <= conf <= 1.0


# --- I3: heatmap cluster helper ----------------------------------------------

def test_heatmap_yield_clusters_merge_move_and_score():
    trk = YieldHeatmapTracker()
    first = trk.record_node_yield_event(7, 0.10, 0.20, success=True)
    assert first["total_events"] == 1 and first["success_count"] == 1
    second = trk.record_node_yield_event(9, 0.12, 0.20, success=True)
    # Same cluster (0.02 < radius 0.025), centroid moves, node_id kept (F-1761).
    assert second["total_events"] == 2
    assert second["node_id"] == 7
    assert second["x"] == pytest.approx(0.11)
    assert second["raw_score"] == pytest.approx(2.0)
    failed = trk.record_node_yield_event(9, 0.11, 0.20, success=False)
    assert failed["fail_count"] == 1
    assert failed["raw_score"] == pytest.approx(1.75)  # failure lowers score
    far = trk.record_node_yield_event(3, 0.50, 0.50, success=True)
    assert far["total_events"] == 1
    assert len(trk.get_heatmap_points()) == 2


def test_heatmap_coord_spaces_never_merge():
    trk = YieldHeatmapTracker()
    trk.record_node_yield_event(1, 500.0, 300.0, success=True)   # pixel space
    trk.record_node_yield_event(2, 0.5, 0.3, success=True)       # normalized
    pts = trk.get_heatmap_points()
    assert len(pts) == 2
    assert {p["coord_space"] for p in pts} == {"pixel", "normalized"}


def test_heatmap_ambush_counts_and_cap(monkeypatch):
    monkeypatch.setattr(telemetry_mod, "_MAX_HEATMAP_CLUSTERS", 3)
    trk = YieldHeatmapTracker()
    for i in range(4):
        trk.record_combat_ambush_event(0.1 + i * 0.2, 0.1)
    pts = trk.get_heatmap_points()
    assert len(pts) == 3  # oldest cluster dropped past the cap
    assert all(p["zone_type"] == "AMBUSH" for p in pts)
    assert sum(p["ambush_count"] for p in pts) == 3


# --- I4: real snapshots -------------------------------------------------------

def test_sample_counter_exposed_and_snapshot_is_live():
    eng = HumanTimingEngine(seed=1, fatigue_enabled=False)
    assert eng.total_samples_generated == 0
    for _ in range(5):
        eng.sample_ex_gaussian(0.2, 0.02, 0.04, 0.18, 0.38)
    eng.sample_log_normal(0.06, 0.22, 0.042, 0.095)
    assert eng.total_samples_generated == 6
    snap = eng.get_human_timing_diagnostic_snapshot()
    assert snap["module"] == "human_timing"
    assert snap["total_samples_generated"] == 6
    assert 0.98 <= snap["fatigue_multiplier"] <= 1.08
    assert snap["session_elapsed_sec"] >= 0.0


def test_scheduler_health_summary_reflects_real_state():
    sched = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=7))
    h = sched.get_human_timing_health_summary()
    assert h["enabled"] is True and h["human_timing_ready"] is True
    assert h["is_on_break"] is False
    assert h["total_macro_breaks"] == 0 and h["total_micro_pauses"] == 0
    assert h["remaining_macro_sec"] > 0
    # Force a macro break, then health must show it.
    t0 = time.monotonic()
    sched._next_macro_interval_sec = 1.0
    due, dur = sched.should_take_macro_break(now=t0 + 5.0)
    assert due and 180.0 <= dur <= 540.0
    h2 = sched.get_human_timing_health_summary()
    assert h2["total_macro_breaks"] == 1
    # Disabled scheduler reports not-ready.
    off = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=7), enabled=False)
    assert off.get_human_timing_health_summary()["human_timing_ready"] is False


def test_telemetry_health_summary_reflects_real_state():
    eng, _ = _seeded_engine(n_gathers=3, n_failures=2, n_combats=1)
    h = eng.get_telemetry_health_summary()
    assert h["telemetry_ready"] is True
    assert h["gathered_total"] == 3
    assert h["failed_total"] == 2
    assert h["combat_count"] == 1
    assert h["alert_history_len"] == 0
    assert h["session_elapsed_sec"] >= 0.0
    assert h["inventory_fill_pct"] == pytest.approx(
        eng.inventory_estimator.get_status()["fill_pct"]
    )
    fresh = TelemetryEngine()
    hf = fresh.get_telemetry_health_summary()
    assert hf["gathered_total"] == 0 and hf["failed_total"] == 0


# --- I5: module sleep delegates to the engine -------------------------------

def test_module_sleep_gaussian_jitter_delegates(monkeypatch):
    import core.human_timing as ht

    seen = []

    def _fake_sleep(sec):
        seen.append(sec)

    monkeypatch.setattr(ht.time, "sleep", _fake_sleep)
    delay = module_sleep_gaussian_jitter(0.001, std_sec=0.0001, min_sec=0.001, max_sec=0.002)
    assert len(seen) == 1
    assert seen[0] == pytest.approx(delay)
    assert 0.001 <= delay <= 0.002
    # The engine method is the single implementation: same call shape.
    delay2 = ht.get_human_timing_engine().sleep_gaussian_jitter(
        0.001, std_sec=0.0001, min_sec=0.001, max_sec=0.002
    )
    assert 0.001 <= delay2 <= 0.002
