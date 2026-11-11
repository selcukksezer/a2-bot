# -*- coding: utf-8 -*-
"""
D58 tests for dataset_factory/rtx_trainer_engine.py.

Covers the two D58 improvements:
1. Vectorized `_evaluate_prototype_validation` — must match the original
   per-sample loop (kept here as an independent reference) within 1e-4,
   and run in < 1 s for N=2000/C=10 (baseline: ~4.4 s min per call).
2. `predict_roi` detector-JSON cache + single-pass distances — same
   outputs as a manual softmax computation, JSON parsed once per file
   version, reloaded when the detector file changes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_factory.rtx_trainer_engine import RTXTrainerEngine


# ----------------------------------------------------------------------
# Reference implementation: the ORIGINAL per-sample loop, kept verbatim
# in semantics so the vectorized version can be proven equivalent —
# EXCEPT for the deliberate F3 (dilim 05) behaviour changes, which the
# reference mirrors: the predicted box is the prior only (GT-blend
# removed, `blend_box = 0.0`), a missing bbox prior falls back to the
# neutral default prior (0.12 x 0.12) instead of the GT box, and the
# result carries `"metrics_kind": "opencv_prototype_prior"`.
# ----------------------------------------------------------------------
def _reference_eval(X_val_norm, y_val, bboxes_val, centroids, feature_weights,
                    class_exemplars, bbox_priors, unique_classes,
                    progress_ratio, avg_box_err):
    n_val = int(X_val_norm.shape[0])
    if n_val == 0:
        return {"box_loss": 0.0, "cls_loss": 0.0, "precision": 0.0,
                "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}
    correct = 0
    total_margin_loss = 0.0
    iou_scores = []
    predicted, truths = [], []
    for i in range(n_val):
        feat = X_val_norm[i]
        true_cid = int(y_val[i])
        gt_bbox = bboxes_val[i]
        best_cid = unique_classes[0]
        best_dist = float("inf")
        true_dist = float("inf")
        for cid in unique_classes:
            diff = feat - centroids[cid]
            centroid_dist = float(np.sqrt(max(
                0.0, float(np.mean(feature_weights[cid] * np.square(diff))))))
            exemplars = class_exemplars.get(cid)
            if exemplars is not None and exemplars.shape[0] > 0:
                ex_diffs = exemplars - feat[None, :]
                ex_dists = np.sqrt(np.maximum(
                    0.0, np.mean(feature_weights[cid][None, :] * np.square(ex_diffs), axis=1)))
                min_ex_dist = float(np.min(ex_dists))
                dist = (1.0 - 0.65 * progress_ratio) * centroid_dist + (
                    0.65 * progress_ratio) * min_ex_dist
            else:
                dist = centroid_dist
            if dist < best_dist:
                best_dist = dist
                best_cid = cid
            if cid == true_cid:
                true_dist = dist
        predicted.append(best_cid)
        truths.append(true_cid)
        if best_cid == true_cid:
            correct += 1
        total_margin_loss += min(2.0, true_dist * 0.12)
        # F3: predicted box is the prior only — the old 0.88 GT-blend
        # (`blend_box = 0.88 * progress_ratio`) is gone (`blend_box = 0.0`),
        # and a missing prior uses the neutral default (0.12 x 0.12),
        # never the GT box (which used to guarantee IoU=1).
        prior_wh = bbox_priors.get(
            best_cid, np.array([0.5, 0.5, 0.12, 0.12], dtype=np.float32)
        )[2:4]
        pred_w = float(prior_wh[0])
        pred_h = float(prior_wh[1])
        inter_w = min(float(gt_bbox[2]), pred_w)
        inter_h = min(float(gt_bbox[3]), pred_h)
        inter_area = max(0.0, inter_w) * max(0.0, inter_h)
        union_area = float(gt_bbox[2] * gt_bbox[3]) + (pred_w * pred_h) - inter_area + 1e-6
        iou_scores.append(float(inter_area / union_area))
    tp = {cid: 0 for cid in unique_classes}
    fp = {cid: 0 for cid in unique_classes}
    fn = {cid: 0 for cid in unique_classes}
    for pred_cid, true_cid in zip(predicted, truths):
        if pred_cid == true_cid:
            tp[true_cid] += 1
        else:
            fp[pred_cid] += 1
            fn[true_cid] += 1
    precisions = [(tp[c] / (tp[c] + fp[c])) if (tp[c] + fp[c]) > 0 else 0.0
                  for c in unique_classes]
    recalls = [(tp[c] / (tp[c] + fn[c])) if (tp[c] + fn[c]) > 0 else 0.0
               for c in unique_classes]
    correct_flags = [int(pr == tr) for pr, tr in zip(predicted, truths)]
    map50 = float(np.mean([1.0 if (cf and iou >= 0.50) else 0.0
                           for cf, iou in zip(correct_flags, iou_scores)]))
    thresholds = [0.50 + 0.05 * k for k in range(10)]
    map50_95 = float(np.mean([
        float(np.mean([1.0 if (cf and iou >= thr) else 0.0
                       for cf, iou in zip(correct_flags, iou_scores)]))
        for thr in thresholds]))
    return {
        "box_loss": max(0.0, float(avg_box_err)),
        "cls_loss": max(0.0, total_margin_loss / float(n_val)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "mAP50": map50,
        "mAP50_95": map50_95,
        # F3: the production result tags these as prototype metrics.
        "metrics_kind": "opencv_prototype_prior",
    }


def _make_eval_data(seed=7, n_val=120, n_classes=6, feat_dim=56, n_exemplars=24):
    rng = np.random.default_rng(seed)
    classes = list(range(n_classes))
    X = rng.normal(0, 1, size=(n_val, feat_dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_val).astype(np.int32)
    bboxes = rng.uniform(0.05, 0.3, size=(n_val, 4)).astype(np.float32)
    centroids = {c: rng.normal(0, 1, size=feat_dim).astype(np.float32) for c in classes}
    weights = {c: rng.uniform(0.5, 1.5, size=feat_dim).astype(np.float32) for c in classes}
    exemplars = {c: rng.normal(0, 1, size=(n_exemplars, feat_dim)).astype(np.float32)
                 for c in classes}
    priors = {c: np.array([0.5, 0.5, 0.12, 0.12], dtype=np.float32) for c in classes}
    return X, y, bboxes, centroids, weights, exemplars, priors, classes


@pytest.fixture
def engine(tmp_path: Path) -> RTXTrainerEngine:
    return RTXTrainerEngine(project_root=str(tmp_path), models_dir=str(tmp_path / "models"))


@pytest.mark.parametrize("progress_ratio", [0.0, 0.25, 0.5, 1.0])
def test_eval_matches_reference_loop(engine, progress_ratio):
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data()
    got = engine._evaluate_prototype_validation(
        X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
        feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
        unique_classes=classes, progress_ratio=progress_ratio, avg_box_err=0.02)
    want = _reference_eval(X, y, bboxes, centroids, weights, exemplars,
                           priors, classes, progress_ratio, 0.02)
    assert set(got) == set(want)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-4), key


def test_eval_matches_reference_without_exemplars(engine):
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data()
    exemplars = {c: (None if c % 2 == 0 else np.zeros((0, 56), dtype=np.float32))
                 for c in classes}
    got = engine._evaluate_prototype_validation(
        X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
        feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
        unique_classes=classes, progress_ratio=0.5, avg_box_err=0.03)
    want = _reference_eval(X, y, bboxes, centroids, weights, exemplars,
                           priors, classes, 0.5, 0.03)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-4), key


def test_eval_missing_bbox_prior_falls_back_to_gt(engine):
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data()
    priors = {c: p for c, p in priors.items() if c != classes[-1]}
    got = engine._evaluate_prototype_validation(
        X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
        feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
        unique_classes=classes, progress_ratio=0.5, avg_box_err=0.02)
    want = _reference_eval(X, y, bboxes, centroids, weights, exemplars,
                           priors, classes, 0.5, 0.02)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-4), key


def test_eval_single_class(engine):
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data(
        n_classes=1, n_val=25)
    got = engine._evaluate_prototype_validation(
        X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
        feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
        unique_classes=classes, progress_ratio=0.5, avg_box_err=0.01)
    want = _reference_eval(X, y, bboxes, centroids, weights, exemplars,
                           priors, classes, 0.5, 0.01)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-4), key


def test_eval_empty_val_returns_zeros(engine):
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data(n_val=0)
    got = engine._evaluate_prototype_validation(
        X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
        feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
        unique_classes=classes, progress_ratio=0.5, avg_box_err=0.02)
    assert got == {"box_loss": 0.0, "cls_loss": 0.0, "precision": 0.0,
                   "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}


def test_min_weighted_exemplar_dist_matches_direct(engine):
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(50, 56)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=56).astype(np.float32)
    ex = rng.normal(0, 1, size=(24, 56)).astype(np.float32)
    got = engine._min_weighted_exemplar_dist(X, w, ex)
    diffs = ex[None, :, :] - X[:, None, :]
    want = np.sqrt(np.maximum(0.0, np.mean(w[None, None, :] * diffs ** 2, axis=2))).min(axis=1)
    assert np.allclose(got, want, atol=1e-5)


def test_eval_speed_regression_guard(engine):
    """Vectorized eval must stay far below the old ~4.4 s/call at N=2000/C=10."""
    X, y, bboxes, centroids, weights, exemplars, priors, classes = _make_eval_data(
        n_val=2000, n_classes=10)
    kwargs = dict(X_val_norm=X, y_val=y, bboxes_val=bboxes, centroids=centroids,
                  feature_weights=weights, class_exemplars=exemplars, bbox_priors=priors,
                  unique_classes=classes, progress_ratio=0.5, avg_box_err=0.02)
    engine._evaluate_prototype_validation(**kwargs)  # warm-up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        engine._evaluate_prototype_validation(**kwargs)
        times.append(time.perf_counter() - t0)
    assert min(times) < 1.0, f"eval too slow: {times}"


# ----------------------------------------------------------------------
# predict_roi: cache + single-pass distances
# ----------------------------------------------------------------------
def _write_detector(engine, centroids, weights, feat_dim=56):
    det = {
        "feature_mean": [0.0] * feat_dim,
        "feature_std": [1.0] * feat_dim,
        "classes": {
            str(c): {"class_id": c, "class_name": f"class_{c}",
                     "centroid": centroids[c].tolist(),
                     "feature_weights": weights[c].tolist()}
            for c in centroids
        },
    }
    engine.custom_detector_path.parent.mkdir(parents=True, exist_ok=True)
    engine.custom_detector_path.write_text(json.dumps(det), encoding="utf-8")
    return det


def _manual_predict(engine, roi):
    det = json.loads(engine.custom_detector_path.read_text(encoding="utf-8"))
    feat_mean = np.array(det["feature_mean"], dtype=np.float32)
    feat_std = np.array(det["feature_std"], dtype=np.float32)
    feat_vec, _ = engine.extract_roi_features(roi)
    feat_norm = (feat_vec - feat_mean) / (feat_std + 1e-5)
    dists = []
    for _cid, cinfo in det["classes"].items():
        centroid = np.array(cinfo["centroid"], dtype=np.float32)
        weights = np.array(cinfo["feature_weights"], dtype=np.float32)
        diff = feat_norm - centroid
        dists.append(float(np.sqrt(max(0.0, float(np.mean(weights * np.square(diff)))))))
    neg = np.array([-d for d in dists], dtype=np.float64)
    neg = neg - float(np.max(neg))
    soft = np.exp(neg)
    best = int(np.argmin(dists))
    return best, float(soft[best] / float(np.sum(soft))), dists[best]


def test_predict_roi_matches_manual_softmax(engine):
    rng = np.random.default_rng(9)
    centroids = {c: rng.normal(0, 1, size=56).astype(np.float32) for c in range(10)}
    weights = {c: np.ones(56, dtype=np.float32) for c in range(10)}
    _write_detector(engine, centroids, weights)
    roi = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    got = engine.predict_roi(roi)
    best, conf, dist = _manual_predict(engine, roi)
    assert got["class_id"] == best
    assert got["confidence"] == pytest.approx(conf, abs=1e-4)
    assert got["distance"] == pytest.approx(dist, abs=1e-4)


def test_predict_roi_caches_detector_json(engine, monkeypatch):
    rng = np.random.default_rng(9)
    centroids = {c: rng.normal(0, 1, size=56).astype(np.float32) for c in range(10)}
    weights = {c: np.ones(56, dtype=np.float32) for c in range(10)}
    _write_detector(engine, centroids, weights)
    roi = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    import dataset_factory.rtx_trainer_engine as mod
    calls = {"n": 0}
    real_loads = mod.json.loads

    def counting(text, *a, **k):
        calls["n"] += 1
        return real_loads(text, *a, **k)

    monkeypatch.setattr(mod.json, "loads", counting)
    first = engine.predict_roi(roi)
    second = engine.predict_roi(roi)
    assert first == second
    assert calls["n"] == 1, "detector JSON must be parsed once while the file is unchanged"

    # Retrain (rewrite the file, bump mtime): next call must reload.
    new_centroids = {c: rng.normal(0, 1, size=56).astype(np.float32) for c in range(10)}
    _write_detector(engine, new_centroids, weights)
    st = engine.custom_detector_path.stat()
    os.utime(engine.custom_detector_path, ns=(st.st_atime_ns + 10_000_000,
                                              st.st_mtime_ns + 10_000_000))
    third = engine.predict_roi(roi)
    assert calls["n"] == 2, "detector JSON must be re-parsed after the file changes"
    best, conf, dist = _manual_predict(engine, roi)
    assert third["class_id"] == best
    assert third["confidence"] == pytest.approx(conf, abs=1e-4)


def test_predict_roi_untrained_when_no_detector(engine):
    roi = np.zeros((32, 32, 3), dtype=np.uint8)
    got = engine.predict_roi(roi)
    assert got["class_id"] == -1
    assert got["class_name"] == "untrained"
    assert got["confidence"] == 0.0


# ----------------------------------------------------------------------
# Mode B end-to-end (vectorized eval inside the real pipeline)
# ----------------------------------------------------------------------
def test_mode_b_end_to_end_small(engine, tmp_path):
    rng = np.random.default_rng(11)
    ds = tmp_path / "dataset"
    for split, n in (("train", 8), ("val", 4)):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
            cv2.imwrite(str(ds / "images" / split / f"img_{i:03d}.jpg"), img)
            lines = [f"{int(rng.integers(0, 3))} 0.5 0.5 0.2 0.2" for _ in range(3)]
            (ds / "labels" / split / f"img_{i:03d}.txt").write_text("\n".join(lines))
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 3\n"
        "names: {0: herb_aria, 1: herb_mela, 2: herb_targena}\n")
    cfg = engine.build_rtx_training_config(epochs=2, imgsz=640, batch=16)
    res = engine._run_mode_b_opencv_prototype(
        yaml_path=yaml_path, epochs=2, imgsz=640, batch=16, train_cfg=cfg,
        env_info={"gpu_name": "CPU", "ultralytics_installed": False},
        bat_path=tmp_path / "train_rtx_yolo.bat", progress_callback=None)
    assert res["state"] == "completed"
    assert res["train_samples"] == 24
    assert res["val_samples"] == 12
    assert 0.0 <= res["mAP50"] <= 1.0
    assert 0.0 <= res["mAP50_95"] <= 1.0
    assert Path(res["artifacts"]["custom_detector_json"]).is_file()
