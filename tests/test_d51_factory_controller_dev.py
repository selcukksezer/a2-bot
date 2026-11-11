# -*- coding: utf-8 -*-
"""
tests/test_d51_factory_controller_dev.py
=========================================
DEV slice D51 (dataset_factory/factory_controller.py) improvements:

1. `_find_latest_preview_on_disk` — cached + single `os.scandir` pass
   instead of a full `iterdir` + `is_file()` + sort on every status poll.
   Cache is keyed on the previews-dir mtime and bounded by a 10 s TTL.
2. `on_gather_success` — `time_machine.get_latest_frame()` is looked up
   exactly once when `reference_ts` is None (was: twice — condition + value).
3. Readiness/audit/seal/certificate chain — each public wrapper calls
   `validate_factory_controller_contract()` exactly once and threads the
   single verdict through private builders (was: 8 calls for
   `get_factory_controller_audit_summary`, one per nesting level, with the
   levels theoretically able to disagree).
4. Dead `img_h, img_w = target_frame.shape[:2]` removed (never read).

Public names, signatures and returned payloads are unchanged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

import dataset_factory.factory_controller as fc
from dataset_factory.factory_controller import AutoLabelingDatasetFactory
from dataset_factory.quality_dedup_gate import QualityGateDecision


# ---------------------------------------------------------------- helpers

def _bare_factory(previews_dir: Path) -> AutoLabelingDatasetFactory:
    """Instance with only the attributes `_find_latest_preview_on_disk` needs."""
    inst = object.__new__(AutoLabelingDatasetFactory)
    inst.previews_dir = previews_dir
    inst._latest_preview_cache = None
    return inst


def _seed_previews(d: Path, n: int = 50) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    base = time.time() - n
    out = []
    for i in range(n):
        p = d / f"preview_{i:04d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 16)
        os.utime(p, (base + i, base + i))
        out.append(p)
    # Non-image files must be ignored by the scan.
    (d / "notes.txt").write_text("not a preview")
    (d / "stale.png.bak").write_text("not a preview")
    return out


# ------------------------------------------------------- 1. preview scan

def test_latest_preview_returns_newest_and_ignores_non_images(tmp_path):
    d = tmp_path / "previews"
    files = _seed_previews(d, 50)
    inst = _bare_factory(d)
    assert inst._find_latest_preview_on_disk() == files[-1]


def test_latest_preview_empty_dir_returns_none_and_caches_miss(tmp_path):
    d = tmp_path / "previews"
    d.mkdir()
    inst = _bare_factory(d)
    assert inst._find_latest_preview_on_disk() is None
    assert inst._latest_preview_cache is not None
    assert inst._latest_preview_cache[0] is None
    # A cached miss must not rescan: with the dir removed from under it the
    # call still returns the cached None only while the mtime matches; the
    # cache entry itself proves the miss was recorded.
    assert inst._find_latest_preview_on_disk() is None


def test_latest_preview_missing_dir_returns_none(tmp_path):
    inst = _bare_factory(tmp_path / "no-such-dir")
    assert inst._find_latest_preview_on_disk() is None


def test_latest_preview_new_file_invalidates_cache(tmp_path):
    d = tmp_path / "previews"
    files = _seed_previews(d, 10)
    inst = _bare_factory(d)
    assert inst._find_latest_preview_on_disk() == files[-1]
    # Creating a newer preview changes the directory mtime -> rescan.
    time.sleep(0.02)
    newest = d / "preview_9999.jpg"
    newest.write_bytes(b"\xff\xd8\xff\xe0" + b"y" * 16)
    os.utime(newest, (time.time() + 5, time.time() + 5))
    assert inst._find_latest_preview_on_disk() == newest


def test_latest_preview_in_place_rewrite_seen_after_ttl(tmp_path):
    """In-place overwrite of the latest file does not change the dir mtime,
    so the answer is stale until the TTL expires — then it is re-read."""
    d = tmp_path / "previews"
    files = _seed_previews(d, 5)
    inst = _bare_factory(d)
    assert inst._find_latest_preview_on_disk() == files[-1]
    # Rewrite the latest file in place with a much older mtime.
    os.utime(files[-1], (time.time() - 1000, time.time() - 1000))
    # Still cached (dir mtime unchanged, TTL not expired).
    assert inst._find_latest_preview_on_disk() == files[-1]
    # Force TTL expiry and confirm the scan picks up the new reality.
    cached_path, cached_mtime, _ts = inst._latest_preview_cache
    inst._latest_preview_cache = (
        cached_path,
        cached_mtime,
        time.monotonic() - (fc.AutoLabelingDatasetFactory._PREVIEW_SCAN_CACHE_TTL_SEC + 1),
    )
    assert inst._find_latest_preview_on_disk() == files[-2]


def test_latest_preview_cache_ttl_is_bounded():
    assert 0 < fc.AutoLabelingDatasetFactory._PREVIEW_SCAN_CACHE_TTL_SEC <= 60


# ------------------------------------------- 2. get_latest_frame dedup

def _gather_factory(tmp_path: Path) -> AutoLabelingDatasetFactory:
    root = tmp_path / "proj"
    factory = AutoLabelingDatasetFactory(
        project_root=root, dataset_dir=root / "dataset", models_dir=root / "models"
    )
    factory.auto_labeler = SimpleNamespace(
        auto_label_frame=lambda **k: [],
        draw_annotated_preview=lambda f, a: f,
    )
    factory.curator = SimpleNamespace(
        auto_label_frame=lambda f, **_k: [],
        save_labeled_sample=lambda **k: {"preview_path": ""},
        rebuild_manifest_stats=lambda: {},
    )
    factory.quality_gate = SimpleNamespace(
        evaluate_frame=lambda **k: QualityGateDecision(
            accepted=True, reason="stub", sharpness=100.0, dhash=0,
            min_hamming_seen=99, split="train", validated_annotations=[],
        )
    )
    factory.active_learning = SimpleNamespace(
        compute_sample_priority=lambda a: 1.0,
        increment_class_count=lambda *a: None,
    )
    factory.auto_augment_on_gather = False
    factory.synthetic_augment_10x = False
    return factory


def test_on_gather_success_calls_get_latest_frame_once(tmp_path, monkeypatch):
    factory = _gather_factory(tmp_path)
    monkeypatch.setattr(fc, "is_valid_game_frame", lambda *a, **k: (True, "stub-ok"))
    frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    now = time.time()
    for i in range(5):
        assert factory.time_machine.push_frame(
            frame=frame, timestamp=now - (4 - i) * 0.8,
            player_pos=(100.0, 200.0), region_name="Test", force=True,
        ) is not None

    orig = factory.time_machine.get_latest_frame
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return orig()

    factory.time_machine.get_latest_frame = counting
    out = factory.on_gather_success(
        material_name="Aria", harvest_stage=1, force_save=True
    )
    assert calls["n"] == 1
    assert out.get("saved") is True
    assert out.get("used_time_machine") is True
    assert out.get("actual_lookback_sec", 0) > 0


def test_on_gather_success_with_reference_ts_never_calls_get_latest_frame(
    tmp_path, monkeypatch
):
    factory = _gather_factory(tmp_path)
    monkeypatch.setattr(fc, "is_valid_game_frame", lambda *a, **k: (True, "stub-ok"))
    frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    now = time.time()
    for i in range(5):
        factory.time_machine.push_frame(
            frame=frame, timestamp=now - (4 - i) * 0.8,
            player_pos=(100.0, 200.0), region_name="Test", force=True,
        )

    def boom():
        raise AssertionError("get_latest_frame must not be called with reference_ts")

    factory.time_machine.get_latest_frame = boom
    out = factory.on_gather_success(
        material_name="Aria", harvest_stage=1, reference_ts=now, force_save=True
    )
    assert out.get("saved") is True
    assert out.get("actual_lookback_sec", -1) == pytest.approx(2.0, abs=0.6)


# ------------------------------------------- 3. contract chain, one call

_CHAIN_WRAPPERS = [
    "get_factory_controller_baseline_metrics",
    "get_factory_controller_readiness_status",
    "get_factory_controller_audit_record",
    "get_factory_controller_operational_digest",
    "get_factory_controller_telemetry_payload",
    "get_factory_controller_integrity_seal",
    "get_factory_controller_compliance_certificate",
    "get_factory_controller_readiness_probe",
    "get_factory_controller_audit_summary",
]


@pytest.mark.parametrize("name", _CHAIN_WRAPPERS)
def test_chain_wrapper_validates_exactly_once(monkeypatch, name):
    calls = {"n": 0}
    orig = fc.validate_factory_controller_contract

    def counting():
        calls["n"] += 1
        return orig()

    monkeypatch.setattr(fc, "validate_factory_controller_contract", counting)
    getattr(fc, name)()
    assert calls["n"] == 1, f"{name} validated {calls['n']} times"


def test_chain_audit_summary_payload_shape_unchanged():
    out = fc.get_factory_controller_audit_summary()
    assert out["module"] == "factory_controller"
    assert out["audit_passed"] == fc.validate_factory_controller_contract()
    probe = out["probe"]
    assert probe["ready"] == out["audit_passed"]
    cert = probe["certificate"]
    assert cert["certified"] == out["audit_passed"]
    seal = cert["seal"]
    assert seal["sealed"] == out["audit_passed"]
    telemetry = seal["telemetry"]
    assert telemetry["payload"]["operational"] == out["audit_passed"]
    digest = telemetry["payload"]
    assert digest["audit"]["audit_passed"] == out["audit_passed"]
    assert digest["audit"]["readiness"]["ready"] == out["audit_passed"]
    metrics = digest["audit"]["readiness"]["metrics"]
    assert metrics["contract_valid"] == out["audit_passed"]
    assert metrics["capability_count"] == len(fc.get_factory_controller_capabilities())


def test_chain_nested_levels_agree_when_contract_is_false(monkeypatch):
    """A single verdict is threaded through: every nested level reports the
    same (False) value instead of each level re-validating."""
    monkeypatch.setattr(fc, "validate_factory_controller_contract", lambda: False)
    out = fc.get_factory_controller_audit_summary()
    assert out["audit_passed"] is False
    assert out["probe"]["ready"] is False
    assert out["probe"]["certificate"]["certified"] is False
    assert out["probe"]["certificate"]["seal"]["sealed"] is False
    assert out["probe"]["certificate"]["seal"]["telemetry"]["payload"]["operational"] is False
