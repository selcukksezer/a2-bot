"""D46 (dataset-utils) dev-improvement tests.

Locks in the three improvements the previous D46 agent left dirty:
1. contract-chain functions evaluate validate_*_contract() exactly once
   (private builders thread the result down) with unchanged return values;
2. CalibratedRoisManager.get_normalized_rect() returns None for malformed
   entries instead of raising or returning inverted/degenerate rects;
3. clean_dataset._validate_images() parallel validation returns the same
   ordered results as the sequential path, and require_novelty=True never
   uses the thread pool (novelty decisions must stay scheduling-independent).
"""

import json

import cv2
import numpy as np
import pytest

import dataset_factory as pkg
import dataset_factory.active_learning as al
import dataset_factory.calibrated_rois as cr
from dataset_factory import clean_dataset as cd


# ---------------------------------------------------------------------------
# 1. Contract chains: single evaluation, unchanged values
# ---------------------------------------------------------------------------
CHAIN_FUNCS_PKG = [
    "get_dataset_factory_package_baseline_metrics",
    "get_dataset_factory_package_readiness_status",
    "get_dataset_factory_package_audit_record",
    "get_dataset_factory_init_operational_digest",
    "get_dataset_factory_init_telemetry_payload",
    "get_dataset_factory_package_telemetry_payload",
    "get_dataset_factory_package_integrity_seal",
    "get_dataset_factory_package_compliance_certificate",
    "get_dataset_factory_pkg_readiness_probe",
    "get_dataset_factory_pkg_audit_summary",
    "get_dataset_factory_init_audit_record",
    "get_dataset_factory_init_audit_summary",
    "get_dataset_factory_init_heartbeat_snapshot",
]

CHAIN_FUNCS_AL = [
    "get_active_learning_baseline_metrics",
    "get_active_learning_readiness_status",
    "get_active_learning_audit_record",
    "get_active_learning_operational_digest",
    "get_active_learning_telemetry_payload",
    "get_active_learning_integrity_seal",
    "get_active_learning_compliance_certificate",
    "get_active_learning_readiness_probe",
    "get_active_learning_audit_summary",
]


@pytest.mark.parametrize("mod,validate_name,funcs", [
    (pkg, "validate_dataset_factory_package_contract", CHAIN_FUNCS_PKG),
    (al, "validate_active_learning_contract", CHAIN_FUNCS_AL),
])
def test_contract_chain_single_evaluation_and_unchanged_values(mod, validate_name, funcs, monkeypatch):
    orig = getattr(mod, validate_name)
    expected = {name: getattr(mod, name)() for name in funcs}
    for name in funcs:
        calls = {"n": 0}

        def counting(*a, _orig=orig, _c=calls, **k):
            _c["n"] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(mod, validate_name, counting)
        try:
            got = getattr(mod, name)()
        finally:
            monkeypatch.undo()
        assert got == expected[name], f"{name} return value changed"
        assert calls["n"] == 1, f"{name} evaluated contract {calls['n']}x, expected 1x"


# ---------------------------------------------------------------------------
# 2. Calibrated ROIs: malformed entries -> None, valid entries unchanged
# ---------------------------------------------------------------------------
ROI_ENTRIES = {
    "valid_fmt1": {"normalized_roi": [0.1, 0.2, 0.3, 0.4]},
    "valid_fmt2": {"norm_x": 0.1, "norm_y": 0.2, "norm_w": 0.3, "norm_h": 0.4},
    "neg_x": {"normalized_roi": [-0.05, 0.2, 0.3, 0.4]},
    "overflow": {"normalized_roi": [0.8, 0.8, 0.5, 0.5]},
    "zero_w": {"normalized_roi": [0.1, 0.2, 0.0, 0.4]},
    "neg_h": {"normalized_roi": [0.1, 0.2, 0.3, -0.1]},
    "nan": {"normalized_roi": [0.1, 0.2, float("nan"), 0.4]},
    "inf": {"norm_x": 0.1, "norm_y": 0.2, "norm_w": float("inf"), "norm_h": 0.4},
    "str_val": {"normalized_roi": ["a", 0.2, 0.3, 0.4]},
    "none_val": {"normalized_roi": None},
    "outside": {"normalized_roi": [1.5, 1.5, 0.3, 0.4]},
    "short": {"normalized_roi": [0.1, 0.2]},
    "empty": {},
}


@pytest.fixture
def rois_mgr(tmp_path):
    path = tmp_path / "rois.json"
    path.write_text(json.dumps(ROI_ENTRIES))
    return cr.CalibratedRoisManager(path)


def test_roi_valid_entries_unchanged(rois_mgr):
    assert rois_mgr.get_normalized_rect("valid_fmt1") == (0.1, 0.2, 0.4, 0.6)
    assert rois_mgr.get_normalized_rect("valid_fmt2") == (0.1, 0.2, 0.4, 0.6)


def test_roi_partially_out_of_range_clamped_not_dropped(rois_mgr):
    assert rois_mgr.get_normalized_rect("neg_x") == (0.0, 0.2, 0.25, 0.6)
    assert rois_mgr.get_normalized_rect("overflow") == (0.8, 0.8, 1.0, 1.0)


@pytest.mark.parametrize("key", [
    "zero_w", "neg_h", "nan", "inf", "str_val", "none_val", "outside",
    "short", "empty", "missing_key",
])
def test_roi_malformed_entries_return_none(rois_mgr, key):
    assert rois_mgr.get_normalized_rect(key) is None


def test_roi_malformed_entry_pixel_crop_returns_none(rois_mgr):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert rois_mgr.get_pixel_crop(frame, "nan") is None
    assert rois_mgr.get_pixel_crop(frame, "outside") is None
    crop = rois_mgr.get_pixel_crop(frame, "valid_fmt1")
    assert crop is not None and crop.shape[:2] == (40, 60)


# ---------------------------------------------------------------------------
# 3. Parallel validation: same ordered results; novelty stays sequential
# ---------------------------------------------------------------------------
def _make_images(tmp_path, n=6, shape=(180, 320)):
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(7)
    files = []
    for i in range(n):
        frame = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
        p = d / f"frame_{i}.png"
        cv2.imwrite(str(p), frame)
        files.append(p)
    files.sort()
    return files


def test_validate_images_parallel_matches_sequential(tmp_path):
    files = _make_images(tmp_path)
    seq = cd._validate_images(files, require_foreground=False, require_novelty=False, max_workers=1)
    par = cd._validate_images(files, require_foreground=False, require_novelty=False, max_workers=4)
    assert par == seq
    assert len(par) == len(files)


def test_validate_images_novelty_never_uses_thread_pool(tmp_path, monkeypatch):
    files = _make_images(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("ThreadPoolExecutor must not be used when require_novelty=True")

    monkeypatch.setattr(cd, "ThreadPoolExecutor", _boom)
    res = cd._validate_images(files, require_foreground=False, require_novelty=True, max_workers=4)
    assert len(res) == len(files)


def test_purge_dry_run_parallel_matches_sequential(tmp_path):
    root = tmp_path / "ds"
    train = root / "images" / "train"
    train.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(4):
        cv2.imwrite(str(train / f"f{i}.png"), rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    (train / "bad.png").write_bytes(b"not an image")

    def _summary(workers):
        s = cd.purge_non_game_screenshots(
            dataset_dir=str(root), action="move", require_foreground=False,
            require_novelty=False, dry_run=True, verbose=False, max_workers=workers,
        )
        s.pop("elapsed_sec", None)
        return s

    assert _summary(4) == _summary(1)
