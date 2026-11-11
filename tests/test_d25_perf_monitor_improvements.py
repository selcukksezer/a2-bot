# -*- coding: utf-8 -*-
"""D25 (perf-monitor) improvements: locks for the AABB pre-filter, the
render-audit freshness fix, the snapshot shallow copy, the GPU fallback
dedup, and the shared PROCESSENTRY32W struct.

No Windows-only behaviour is asserted as passed here: the throttle tests
run the Linux guards (os.name != "nt") only.
"""

from __future__ import annotations

import copy
import os
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import game_gpu_throttle as ggt  # noqa: E402
from core import system_monitor as sm  # noqa: E402
from core.dry_run_simulator import DryRunReplaySimulator  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _brute_force_intersections(sim, nodes):
    """Independent reference: every non-adjacent pair through the exact
    predicate, no AABB filter. Mirrors the audit's pair rules (skip
    adjacent pairs and the 0 / last wrap-around pair)."""
    coords, valid, _skipped = sim._valid_coords(nodes)
    hits = []
    n = len(coords)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2:
                continue
            hit, point = sim._segments_intersect(coords[i], coords[i + 1], coords[j], coords[j + 1])
            if hit and point is not None:
                hits.append(((i, i + 1), (j, j + 1), point))
    return hits


def _make_route(n, seed=7):
    rng = random.Random(seed)
    x = y = 0.0
    nodes = []
    for i in range(n):
        x += rng.uniform(-60, 60)
        y += rng.uniform(-60, 60)
        nodes.append({"id": f"node_{i}", "x": round(x, 1), "y": round(y, 1), "type": "resource"})
    return nodes


# ----------------------------------------------------------------------
# I1 — AABB pre-filter in run_8x_route_audit (identical results, far fewer
# exact segment tests)
# ----------------------------------------------------------------------

def test_d25_bowtie_crossing_still_detected():
    # 5 nodes: the crossing pair (seg 0 vs seg 2) is not the skipped
    # (0, last) wrap-around pair, so the audit must report it.
    sim = DryRunReplaySimulator()
    nodes = [
        {"id": "a", "x": 0, "y": 0},
        {"id": "b", "x": 100, "y": 100},
        {"id": "c", "x": 100, "y": 0},
        {"id": "d", "x": 0, "y": 100},
        {"id": "e", "x": 50, "y": 150},
    ]
    rep = sim.run_8x_route_audit(nodes)
    assert rep["self_intersection_count"] == 1
    assert rep["self_intersections"][0]["intersection_xy"] == (50.0, 50.0)


def test_d25_no_crossing_route_stays_empty():
    sim = DryRunReplaySimulator()
    nodes = [{"id": f"n{i}", "x": i * 100, "y": (i % 2) * 10} for i in range(30)]
    rep = sim.run_8x_route_audit(nodes)
    assert rep["self_intersection_count"] == 0
    assert rep["zigzag_count"] == 0


@pytest.mark.parametrize("n,seed", [(12, 1), (30, 2), (80, 3), (150, 4)])
def test_d25_audit_matches_brute_force_reference(n, seed):
    """Filtered audit must report exactly the brute-force intersections
    (same pairs, same points) on random routes."""
    sim = DryRunReplaySimulator()
    nodes = _make_route(n, seed)
    rep = sim.run_8x_route_audit(nodes)
    ref = _brute_force_intersections(sim, nodes)
    assert rep["self_intersection_count"] == len(ref)
    got = sorted((tuple(a["seg_a"]), tuple(a["seg_b"])) for a in rep["self_intersections"])
    want = sorted((a, b) for a, b, _p in ref)
    assert got == want


def test_d25_aabb_filter_slashes_exact_tests():
    """400-node seeded route: 79,002 exact _segments_intersect calls
    before the filter, 557 after (measured 2026-11-08). Same 201
    crossings / 79 zigzags. Threshold leaves a wide margin."""
    sim = DryRunReplaySimulator()
    nodes = _make_route(400)
    calls = {"n": 0}
    orig = DryRunReplaySimulator._segments_intersect

    def counting(cls, *args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    with patch.object(DryRunReplaySimulator, "_segments_intersect", classmethod(counting)):
        rep = sim.run_8x_route_audit(nodes)
    assert rep["self_intersection_count"] == 201
    assert rep["zigzag_count"] == 79
    assert calls["n"] < 5000, f"exact-test calls not filtered enough: {calls['n']}"
    assert calls["n"] < 79002 // 10


def test_d25_bbox_touching_segments_still_go_to_exact_test():
    """Segments whose bboxes merely touch must not be skipped by the
    filter — the exact predicate decides (here: collinear overlap on a
    shared edge is NOT a strict crossing, so 0 hits, but the exact test
    must have run)."""
    sim = DryRunReplaySimulator()
    nodes = [
        {"id": "a", "x": 0, "y": 0},
        {"id": "b", "x": 100, "y": 0},
        {"id": "c", "x": 100, "y": 0},
        {"id": "d", "x": 200, "y": 0},
        {"id": "e", "x": 200, "y": 100},
    ]
    calls = {"n": 0}
    orig = DryRunReplaySimulator._segments_intersect

    def counting(cls, *args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    with patch.object(DryRunReplaySimulator, "_segments_intersect", classmethod(counting)):
        rep = sim.run_8x_route_audit(nodes)
    assert rep["self_intersection_count"] == 0
    assert calls["n"] >= 1  # touching bboxes were not filtered away


# ----------------------------------------------------------------------
# I4 — render_simulation_frame no longer re-audits on every frame when the
# route has coordinate-less nodes
# ----------------------------------------------------------------------

def test_d25_render_reuses_audit_when_route_has_skipped_nodes():
    sim = DryRunReplaySimulator()
    nodes = _make_route(200)
    for i in range(0, 200, 20):
        nodes.insert(i, {"id": f"bad_{i}", "type": "resource"})
    calls = {"n": 0}
    orig = sim.run_8x_route_audit

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    sim.run_8x_route_audit = counting  # noqa: B038 - instance-level spy
    frame1 = sim.render_simulation_frame(nodes)
    assert calls["n"] == 1
    frame2 = sim.render_simulation_frame(nodes)
    assert calls["n"] == 1, "second render must reuse the audit, not re-run it"
    assert frame1.shape == (720, 960, 3) and frame1.dtype == np.uint8
    assert frame2.shape == frame1.shape
    assert sim._last_audit_report["skipped_node_count"] == 10
    assert sim._last_audit_report["node_count"] == 200


# ----------------------------------------------------------------------
# I2 — SystemMonitor: shallow per-section copy (was deepcopy) + GPU
# fallback dedup
# ----------------------------------------------------------------------

def test_d25_snapshot_mutation_does_not_poison_cache():
    """Mutating a returned snapshot (nested section + top level) must not
    change what the next get_snapshot() returns (F-0396)."""
    sm.SystemMonitor.reset_cache()
    snap1 = sm.SystemMonitor.get_snapshot()
    orig_gpu_pct = snap1["gpu"]["pct"]
    orig_cpu_pct = snap1["cpu"]["pct"]
    orig_ts = snap1["timestamp"]
    snap1["gpu"]["pct"] = 999
    snap1["cpu"]["pct"] = 999
    snap1["timestamp"] = 0.0
    snap1["gpu"]["injected"] = True
    snap2 = sm.SystemMonitor.get_snapshot()
    assert snap2["gpu"]["pct"] == orig_gpu_pct
    assert snap2["cpu"]["pct"] == orig_cpu_pct
    assert snap2["timestamp"] == orig_ts
    assert "injected" not in snap2["gpu"]
    sm.SystemMonitor.reset_cache()


def test_d25_snapshot_sections_are_flat_scalars():
    """Locks the precondition _copy_snapshot relies on: every section is a
    flat dict of scalars, so a per-section dict() copy isolates exactly
    like deepcopy did. If a nested value appears, this test fails first."""
    sm.SystemMonitor.reset_cache()
    snap = sm.SystemMonitor.get_snapshot()
    for section in ("gpu", "cpu", "capture"):
        assert isinstance(snap[section], dict)
        for value in snap[section].values():
            assert value is None or isinstance(value, (str, int, float, bool)), (section, value)
    assert isinstance(snap["timestamp"], float)
    sm.SystemMonitor.reset_cache()


def test_d25_copy_snapshot_matches_deepcopy_and_isolates():
    sample = {"gpu": {"pct": 42, "available": True}, "cpu": {"pct": 7}, "timestamp": 1.5}
    copied = sm._copy_snapshot(sample)
    assert copied == copy.deepcopy(sample)
    copied["gpu"]["pct"] = 999
    assert sample["gpu"]["pct"] == 42


def test_d25_gpu_unavailable_returns_fresh_identical_dicts():
    a = sm._gpu_unavailable()
    b = sm._gpu_unavailable()
    assert a == b
    assert a["available"] is False and a["pct"] == 0 and a["temp"] == 0
    assert a["label_str"] == "0% · —°C · 0.0/0.0 GB"
    a["pct"] = 99
    assert b["pct"] == 0  # no shared mutable fallback


def test_d25_snapshot_returns_copies_not_cache_handle():
    sm.SystemMonitor.reset_cache()
    s1 = sm.SystemMonitor.get_snapshot()
    s2 = sm.SystemMonitor.get_snapshot()
    assert s1 == s2 and s1 is not s2 and s1["gpu"] is not s2["gpu"]
    sm.SystemMonitor.reset_cache()


# ----------------------------------------------------------------------
# I3 — game_gpu_throttle: one shared _PROCESSENTRY32W struct
# ----------------------------------------------------------------------

def test_d25_process_entry_struct_defined_once_with_expected_fields():
    fields = [name for name, _t in ggt._PROCESSENTRY32W._fields_]
    assert fields == [
        "dwSize", "cntUsage", "th32ProcessID", "th32DefaultHeapID",
        "th32ModuleID", "cntThreads", "th32ParentProcessID",
        "pcPriClassBase", "dwFlags", "szExeFile",
    ]
    assert ggt._PROCESSENTRY32W().dwSize == 0
    import inspect

    src = inspect.getsource(ggt)
    assert src.count("class _PROCESSENTRY32W") == 1
    assert "class PROCESSENTRY32W" not in src
    assert src.count("_PROCESSENTRY32W") >= 5  # def + both users


def test_d25_snapshot_helpers_return_empty_off_windows():
    if os.name != "nt":
        assert ggt._snapshot_pid_to_exe_map() == {}
        assert ggt._pids_from_process_names() == []
        assert ggt.find_aion_game_processes() == []
        assert ggt.is_game_running() is False
    else:
        pytest.skip("windows-only (Linux VM)")
