# -*- coding: utf-8 -*-
"""D10 (focus-glide) improvement tests.

Locks in:
  D10-1  estimate_depth_map: two (5,9) blurs fused into one via blur linearity,
         and analyze_forward_corridor: three np.percentile calls fused into one.
         Golden depth maps + corridor outputs (recorded pre-change) must match
         within float32 rounding noise.
  D10-2  RouteBackupManager: daemon tick must not re-parse + re-hash an unchanged
         route file every 5 s (stat-keyed fast path); corruption must still be
         detected and auto-restored because any write changes (size, mtime_ns).
  D10-3  validate_depth_glide_navigator_contract is a real probe now (was a
         tautology: bool(non-empty capability list)); class ready flags report
         real state instead of unconditional True.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.depth_glide_navigator as dgn
from core.depth_glide_navigator import (
    GlideWaypointController,
    MonocularDepthObstacleAvoidance,
)
from core.dpi_focus_guard import RouteBackupManager

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _build_frames():
    """Same deterministic frames the golden files were recorded with (seed 1234)."""
    rng = np.random.default_rng(1234)
    h, w = 720, 1280
    base = np.linspace(200, 60, h, dtype=np.float32)[:, None, None] * np.ones((1, w, 3), dtype=np.float32)
    frames = {}
    frames["gradient"] = np.clip(base + rng.normal(0, 8, (h, w, 3)), 0, 255).astype(np.uint8)
    frames["noise"] = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    frames["dark"] = np.full((h, w, 3), 18, dtype=np.uint8)
    frames["bright"] = np.full((h, w, 3), 235, dtype=np.uint8)
    frames["gray2d"] = np.clip(np.linspace(0, 255, h)[:, None] * np.ones((1, w)), 0, 255).astype(np.uint8)
    return frames


# ---------------------------------------------------------------- D10-1: depth
def test_depth_maps_match_pre_change_goldens():
    goldens = np.load(FIXTURES / "d10_depth_goldens.npz")
    nav = MonocularDepthObstacleAvoidance()
    for name, frame in _build_frames().items():
        depth = nav.estimate_depth_map(frame)
        assert depth.shape == (90, 160)
        assert depth.dtype == np.float32
        assert float(np.min(depth)) >= 0.0 and float(np.max(depth)) <= 1.0
        diff = float(np.max(np.abs(depth - goldens[name])))
        assert diff < 1e-5, f"{name}: max abs diff vs golden {diff}"


def test_corridor_outputs_match_pre_change_goldens():
    with open(FIXTURES / "d10_depth_goldens.json") as fh:
        goldens = json.load(fh)
    nav = MonocularDepthObstacleAvoidance()
    # Same instance + same order as golden recording: steer uses EMA state.
    for name, frame in _build_frames().items():
        got = nav.analyze_forward_corridor(frame)
        want = goldens[name]
        assert got["recommended_steer_deg"] == want["recommended_steer_deg"], name
        assert got["obstacle_detected"] == want["obstacle_detected"], name
        assert got["obstacle_type"] == want["obstacle_type"], name
        assert got["cliff_drop_detected"] == want["cliff_drop_detected"], name
        for key in ("clearance_left", "clearance_center", "clearance_right", "lower_ground_depth"):
            assert abs(got[key] - want[key]) < 1e-3, (name, key)


def test_blur_fusion_matches_two_blur_reference():
    """blur(a) - 0.6*blur(b) == blur(a - 0.6*b): the identity D10-1 relies on."""
    rng = np.random.default_rng(7)
    luma = rng.random((90, 160)).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    old = np.clip((cv2.blur(np.abs(gx), (5, 9)) - 0.60 * cv2.blur(np.abs(gy), (5, 9))) * 1.6, 0.0, 1.0)
    new = np.clip(cv2.blur(np.abs(gx) - 0.60 * np.abs(gy), (5, 9)) * 1.6, 0.0, 1.0)
    assert float(np.max(np.abs(old - new))) < 1e-6


def test_percentile_fusion_matches_separate_calls():
    rng = np.random.default_rng(2)
    sector = rng.random((45, 42)).astype(np.float32)
    p15, p22, p90 = np.percentile(sector, [15, 22, 90])
    assert abs(float(p90) - float(np.percentile(sector, 90))) < 1e-7
    assert abs(float(p15) - float(np.percentile(sector, 15))) < 1e-7
    assert abs(float(p22) - float(np.percentile(sector, 22))) < 1e-7


def test_estimate_depth_map_fail_closed_and_deterministic():
    nav = MonocularDepthObstacleAvoidance()
    assert np.all(nav.estimate_depth_map(None) == 0.0)
    assert np.all(nav.estimate_depth_map(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0)
    frame = _build_frames()["gradient"]
    a = nav.estimate_depth_map(frame)
    b = nav.estimate_depth_map(frame)
    # Bit-exactness is not guaranteed under OpenCV multithreading; lock
    # determinism within float32 rounding noise instead.
    assert float(np.max(np.abs(a - b))) < 1e-6


# ------------------------------------------------------- D10-2: backup fast path
def _make_region(base: Path, size_nodes: int = 20000) -> Path:
    region_dir = base / "routes" / "testregion"
    region_dir.mkdir(parents=True)
    data = {"region": "testregion",
            "nodes": [{"id": i, "x": float(i), "y": 1.0, "pad": "x" * 200} for i in range(size_nodes)]}
    live = region_dir / "region_data.json"
    live.write_text(json.dumps(data), encoding="utf-8")
    return live


def test_backup_tick_skips_parse_and_hash_when_unchanged(tmp_path, monkeypatch):
    live = _make_region(tmp_path)
    mgr = RouteBackupManager(base_dir=str(tmp_path), backup_interval_sec=600.0)
    assert mgr.backup_region("testregion", force=True) is not None

    calls = {"sig": 0, "valid": 0}
    orig_sig = RouteBackupManager._compute_file_signature
    orig_valid = RouteBackupManager.is_valid_route_json

    def counting_sig(fp):
        calls["sig"] += 1
        return orig_sig(fp)

    def counting_valid(fp):
        calls["valid"] += 1
        return orig_valid(fp)

    monkeypatch.setattr(RouteBackupManager, "_compute_file_signature", staticmethod(counting_sig))
    monkeypatch.setattr(RouteBackupManager, "is_valid_route_json", staticmethod(counting_valid))
    for _ in range(5):
        assert mgr.backup_all_regions(force=False) == {"testregion": None}
    assert calls == {"sig": 0, "valid": 0}


def test_backup_unchanged_after_interval_returns_latest_without_new_file(tmp_path):
    _make_region(tmp_path)
    # D10 race: varsayılan auto_start_daemon=True arka plan tick'i,
    # 0.0'a çekilen _last_backup_ts'yi test çağrısından önce yenileyip
    # backup_region'ın None döndürmesine yol açabiliyor (süit yükünde
    # bir kez görüldü). Bu test daemon davranışını değil, değişmemiş
    # dosyanın mevcut yedeği döndürmesini kilitler (satır 163 emsali).
    mgr = RouteBackupManager(base_dir=str(tmp_path), backup_interval_sec=600.0,
                             auto_start_daemon=False)
    first = mgr.backup_region("testregion", force=True)
    assert first is not None
    mgr._last_backup_ts_by_region["testregion"] = 0.0  # simulate expired interval
    assert mgr.backup_region("testregion", force=False) == first
    assert len(mgr.list_backups_for_region("testregion")) == 1


def test_backup_changed_file_creates_new_backup(tmp_path):
    live = _make_region(tmp_path)
    # fix v5-W2, bulgu #3 (d10 flaky): daemon KAPALI kurulur. Varsayılan
    # auto_start_daemon=True ile daemon thread'i testin write_text'i
    # sırasında (truncate+write atomik değil) canlı dosyayı boş okuyup
    # "bozuk" sanarak restore_latest_backup ile eski yedeği üstüne
    # yazıyordu; sonraki backup_region değişikliği görmeyip mevcut
    # yedeği döndürüyordu (len==1). Bu test daemon davranışını değil,
    # değişmiş dosyanın yeni yedek üretmesini kilitler.
    mgr = RouteBackupManager(base_dir=str(tmp_path), backup_interval_sec=600.0,
                             auto_start_daemon=False)
    mgr.backup_region("testregion", force=True)
    live.write_text(json.dumps({"region": "testregion", "nodes": [{"id": 1}]}), encoding="utf-8")
    mgr._last_backup_ts_by_region["testregion"] = 0.0
    new_backup = mgr.backup_region("testregion", force=False)
    assert new_backup is not None
    assert len(mgr.list_backups_for_region("testregion")) == 2
    assert json.loads(new_backup.read_text(encoding="utf-8"))["nodes"] == [{"id": 1}]


def test_backup_corrupted_file_still_auto_restores(tmp_path):
    live = _make_region(tmp_path)
    mgr = RouteBackupManager(base_dir=str(tmp_path), backup_interval_sec=600.0)
    mgr.backup_region("testregion", force=True)
    good = live.read_text(encoding="utf-8")
    live.write_text("{corrupted json", encoding="utf-8")
    mgr._last_backup_ts_by_region["testregion"] = 0.0
    assert mgr.backup_region("testregion", force=False) is None  # refuses corrupt backup
    assert live.read_text(encoding="utf-8") == good  # auto-restored from backup
    ok, data, _msg = mgr.ensure_region_integrity_or_restore("testregion")
    assert ok and len(data["nodes"]) == 20000


# ------------------------------------------------------- D10-3: real readiness
def test_validate_depth_glide_contract_is_real_probe(monkeypatch):
    assert dgn.validate_depth_glide_navigator_contract() is True
    assert dgn.get_depth_glide_navigator_readiness_status()["ready"] is True

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no estimator")

    monkeypatch.setattr(dgn, "MonocularDepthObstacleAvoidance", _Boom)
    assert dgn.validate_depth_glide_navigator_contract() is False


def test_depth_ready_flag_reports_real_state():
    nav = MonocularDepthObstacleAvoidance()
    flag = nav.get_depth_glide_navigator_ready_flag()
    assert flag["ready"] is True
    assert flag["backend"] == "geometric_perspective_sobel_realtime"
    assert flag["last_inference_ms"] >= 0.0
    assert nav.get_depth_glide_navigator_version_info()["ready"] is True


def test_glide_ready_flag_rejects_misconfigured_thresholds():
    ok = GlideWaypointController()
    assert ok.get_depth_glide_navigator_ready_flag()["ready"] is True
    bad = GlideWaypointController(min_horizontal_dist=500.0, max_horizontal_dist=100.0)
    flag = bad.get_depth_glide_navigator_ready_flag()
    assert flag["ready"] is False
    assert flag["status"] == "misconfigured"
    assert bad.get_depth_glide_navigator_diagnostic_snapshot()["status"] == "misconfigured"
    neg = GlideWaypointController(min_elevation_drop=-5.0)
    assert neg.get_depth_glide_navigator_ready_flag()["ready"] is False
