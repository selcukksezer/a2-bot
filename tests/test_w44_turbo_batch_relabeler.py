# -*- coding: utf-8 -*-
"""
Unit tests for W44 Turbo Batch Relabeler (`DatasetCurator.relabel_dataset_batch`).
Verifies multi-threaded parallel execution, thread-local worker isolation,
incremental caching rules, and `trusted` vs `weak` box separation across
`review/{split}/*.json` and `labels/{split}/*.txt`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List
from unittest.mock import patch

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.dataset_curator import DatasetCurator, imwrite_unicode


def _create_dummy_image(path: Path, width: int = 320, height: int = 240, color: tuple = (60, 120, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (120, 120), (200, 220, 50), -1)
    assert imwrite_unicode(path, img) is True
    return path


def test_relabel_dataset_batch_parallel_execution_and_thread_local_workers(tmp_path: Path) -> None:
    """
    1. Verifies that `relabel_dataset_batch` executes across multiple worker threads
       in parallel, instantiates thread-local YOLOAutoLabeler objects per worker thread,
       and emits accurate progress_callback telemetry.
    """
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)

    img_paths: List[Path] = []
    for i in range(8):
        split_dir = curator.images_train_dir if i < 6 else curator.images_val_dir
        p = _create_dummy_image(split_dir / f"sample_node_{i:02d}.jpg")
        img_paths.append(p)

    thread_ids: set[int] = set()
    labeler_ids_by_thread: Dict[int, set[int]] = {}
    lock = threading.Lock()
    progress_events: List[Dict[str, Any]] = []
    # Deterministic parallelism proof (D89/F-2062): a 2-party barrier trips only
    # when two worker threads overlap inside auto_label_frame. The old
    # time.sleep(0.04) merely hoped the pool would schedule concurrently; the
    # barrier either rendezvouses (parallelism proven) or times out (failure).
    rendezvous = threading.Barrier(2)
    barrier_failures: List[str] = []

    class DummyParallelLabeler:
        def __init__(self) -> None:
            self.instance_id = id(self)

        def auto_label_frame(
            self, frame: np.ndarray, fast_mode: bool = True, source_path: Any = None, full_scan: bool = False
        ) -> List[Dict[str, Any]]:
            tid = threading.get_ident()
            with lock:
                thread_ids.add(tid)
                labeler_ids_by_thread.setdefault(tid, set()).add(self.instance_id)
            try:
                rendezvous.wait(timeout=10.0)
            except threading.BrokenBarrierError:
                with lock:
                    barrier_failures.append(f"thread {tid} timed out at barrier")
            return [
                {
                    "class_id": 0,
                    "class_name": "herb_aria",
                    "cx": 0.50,
                    "cy": 0.50,
                    "w": 0.20,
                    "h": 0.20,
                    "confidence": 0.93,
                    "source": "yolo_detector",
                    # E-07: high confidence alone never trusts; this fixture
                    # needs a txt-written box, so it carries an explicit
                    # human decision (the test's subject is parallelism).
                    "hand_label": True,
                }
            ]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DummyParallelLabeler):
        res = curator.relabel_dataset_batch(
            max_workers=4,
            incremental=False,
            force_all=True,
            progress_callback=lambda payload: progress_events.append(dict(payload)),
        )

    assert res["total"] == 8
    assert res["processed"] == 8
    assert res["labeled"] == 8
    assert res["skipped"] == 0
    assert res["total_boxes"] == 8
    assert res["fps"] > 0.0

    # Multi-threading check: multiple worker threads participated, and at least
    # two of them overlapped inside auto_label_frame (barrier rendezvous).
    assert not barrier_failures, f"Worker threads never overlapped: {barrier_failures}"
    assert len(thread_ids) >= 2, f"Expected multi-threaded execution, got threads={thread_ids}"
    # Thread-local check: each thread reused a single YOLOAutoLabeler instance
    for tid, inst_set in labeler_ids_by_thread.items():
        assert len(inst_set) == 1, f"Thread {tid} created multiple labeler instances: {inst_set}"

    assert len(progress_events) == 8
    assert {ev["processed"] for ev in progress_events} == set(range(1, 9))
    assert all(ev["status"] == "labeled" and ev["total"] == 8 for ev in progress_events)


def test_relabel_dataset_batch_trusted_vs_weak_separation_and_sidecar_output(tmp_path: Path) -> None:
    """
    2. Verifies that `relabel_dataset_batch` separates `trusted` boxes (hand_label /
       human_approved) from `weak` boxes
       (UNVERIFIED_HEURISTIC_SOURCES capped <= 0.68 or low-conf boxes),
       writing both to `review/{split}/{stem}.json` and ONLY trusted boxes to `labels/{split}/{stem}.txt`.
       E-07: high confidence alone never trusts; Y1: the F/E seed is cid 13.
       P1-12 (K-34): an F/E gather seed ALONE is no longer a human decision —
       the v16 rule ("seed -> trusted, conf boosted >= 0.90") was the K-34 bug
       itself (template trace is not evidence). A seed without an explicit
       per-box human approval calibrates weak (conf kept, never inflated) and
       is blocked from the txt by the K1 gate in `train_txt_boxes`; the only
       door in is `human_approved`/`hand_label` (see test 5's approved omni box).
    """
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)

    img_path = _create_dummy_image(curator.images_train_dir / "field_capture_01.jpg")

    class DummyTrustSplitLabeler:
        def auto_label_frame(
            self, frame: np.ndarray, fast_mode: bool = True, source_path: Any = None, full_scan: bool = False
        ) -> List[Dict[str, Any]]:
            return [
                # 1. F-gather prompt seed (cid 13, Y1) WITHOUT an explicit
                # per-box human approval -> weak=True (P1-12/K-34: a seed
                # alone is not a human decision; conf is KEPT at 0.72, never
                # boosted to 0.90, and the K1 gate keeps it out of the txt).
                {
                    "class_id": 13,
                    "class_name": "skill_ui_prompt",
                    "cx": 0.55,
                    "cy": 0.48,
                    "w": 0.12,
                    "h": 0.08,
                    "confidence": 0.72,
                    "source": "f_gather_template",
                },
                # 2. Human-approved high-confidence detection -> trusted=True
                # (E-07: conf >= 0.85 alone no longer trusts)
                {
                    "class_id": 0,
                    "class_name": "herb_aria",
                    "cx": 0.25,
                    "cy": 0.60,
                    "w": 0.18,
                    "h": 0.22,
                    "confidence": 0.91,
                    "source": "yolo_model_v2",
                    "human_approved": True,
                },
                # 3. Unverified heuristic source -> capped at <= 0.68, trusted=False, weak=True
                {
                    "class_id": 3,
                    "class_name": "ore_orichalcum",
                    "cx": 0.75,
                    "cy": 0.70,
                    "w": 0.16,
                    "h": 0.18,
                    "confidence": 0.96,
                    "source": "chromatic_salient_3d_resource",
                },
                # 4. Low confidence detection (< emin_threshold 0.85) -> trusted=False, weak=True
                {
                    "class_id": 4,
                    "class_name": "gem_crystal",
                    "cx": 0.15,
                    "cy": 0.30,
                    "w": 0.10,
                    "h": 0.12,
                    "confidence": 0.62,
                    "source": "yolo_model_v2",
                },
            ]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DummyTrustSplitLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[img_path],
            max_workers=2,
            emin_threshold=0.85,
            conf_floor=0.45,
        )

    assert res["labeled"] == 1
    # O9: total_boxes counts boxes actually written to the txt (1 trusted),
    # not the raw calibrated box count (4). P1-12 (K-34): the unapproved
    # F/E seed (cid 13) is weak and never reaches the txt.
    assert res["total_boxes"] == 1

    review_path = dataset_dir / "review" / "train" / "field_capture_01.json"
    label_path = dataset_dir / "labels" / "train" / "field_capture_01.txt"
    assert review_path.is_file()
    assert label_path.is_file()

    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    trusted_boxes = review_data["trusted"]
    weak_boxes = review_data["weak"]

    assert len(trusted_boxes) == 1
    assert len(weak_boxes) == 3

    trusted_cids = {b["class_id"] for b in trusted_boxes}
    weak_cids = {b["class_id"] for b in weak_boxes}
    assert trusted_cids == {0}
    assert weak_cids == {3, 4, 13}

    assert all(b["trusted"] is True and b["weak"] is False for b in trusted_boxes)
    assert all(b["trusted"] is False and b["weak"] is True for b in weak_boxes)

    # Unverified heuristic source must be capped at <= 0.68
    heuristic_box = next(b for b in weak_boxes if b["class_id"] == 3)
    assert heuristic_box["confidence"] <= 0.68

    # P1-12 (K-32/K-34, never inflate): the unapproved seed keeps its
    # measured confidence (0.72) — it is weak, not boosted to 0.90.
    seed_box = next(b for b in weak_boxes if b["class_id"] == 13)
    assert seed_box["confidence"] == 0.72

    # Train .txt file must contain ONLY the 1 trusted box (class 0)
    txt_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(txt_lines) == 1
    emitted_cids = {int(ln.split()[0]) for ln in txt_lines}
    assert emitted_cids == {0}


def test_relabel_dataset_batch_stem_mismatch_and_exemplar_trust_calibration(tmp_path: Path) -> None:
    """
    3. Verifies stem-material mismatch penalty and exemplar match trust rules during
       batch relabeling:
       - On an `aria_cluster_01.jpg` stem, a mismatched material class (`mela`, cid=1)
         is penalized to <= 0.45 and marked `weak=True` (`trusted=False`).
       - A matching `herb_aria` (cid=0) with `exemplar_match=True` AND an explicit
         human approval is marked `trusted=True` and emitted to
         `labels/val/aria_cluster_01.txt` (E-07: exemplar_match alone no
         longer trusts — calibrate is the single trust path).
    """
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)

    val_img = _create_dummy_image(curator.images_val_dir / "aria_cluster_01.jpg")

    class DummyStemLabeler:
        def auto_label_frame(
            self, frame: np.ndarray, fast_mode: bool = True, source_path: Any = None, full_scan: bool = False
        ) -> List[Dict[str, Any]]:
            return [
                # Matching material + exemplar_match=True + explicit human
                # approval -> trusted (exemplar_match alone does not trust)
                {
                    "class_id": 0,
                    "class_name": "herb_aria",
                    "cx": 0.40,
                    "cy": 0.50,
                    "w": 0.15,
                    "h": 0.15,
                    "confidence": 0.89,
                    "exemplar_match": True,
                    "human_approved": True,
                    "source": "exemplar_memory",
                },
                # Mismatched material (class 1 mela on aria stem) despite high raw conf -> weak (<= 0.45)
                {
                    "class_id": 1,
                    "class_name": "herb_mela",
                    "cx": 0.70,
                    "cy": 0.55,
                    "w": 0.15,
                    "h": 0.15,
                    "confidence": 0.94,
                    "source": "yolo_detector",
                },
            ]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DummyStemLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[val_img],
            max_workers=2,
            emin_threshold=0.85,
            conf_floor=0.45,
        )

    assert res["labeled"] == 1
    review_path = dataset_dir / "review" / "val" / "aria_cluster_01.json"
    label_path = dataset_dir / "labels" / "val" / "aria_cluster_01.txt"

    rev = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(rev["trusted"]) == 1
    assert len(rev["weak"]) == 1

    assert rev["trusted"][0]["class_id"] == 0
    assert rev["trusted"][0]["trusted"] is True

    mismatched = rev["weak"][0]
    assert mismatched["class_id"] == 1
    assert mismatched["trusted"] is False
    assert mismatched["weak"] is True
    assert mismatched["confidence"] <= 0.45

    txt_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(txt_lines) == 1
    assert int(txt_lines[0].split()[0]) == 0


def test_relabel_dataset_batch_incremental_cache_skips_valid_and_reprocesses_zero_box(tmp_path: Path) -> None:
    """
    4. Verifies incremental `.label_cache.json` behavior:
       - Unchanged images with > 0 boxes are skipped on incremental runs (`status == "skipped"`).
       - Images with 0 boxes are NEVER skipped on incremental runs and are re-labeled.
       - `force_all=True` overrides the cache and relabels all images.
    """
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)

    img_with_box = _create_dummy_image(curator.images_train_dir / "node_with_box.jpg")
    img_empty = _create_dummy_image(curator.images_train_dir / "node_empty.jpg")

    call_counts: Dict[str, int] = {"node_with_box": 0, "node_empty": 0}
    lock = threading.Lock()

    class DummyCacheLabeler:
        def auto_label_frame(
            self, frame: np.ndarray, fast_mode: bool = True, source_path: Any = None, full_scan: bool = False
        ) -> List[Dict[str, Any]]:
            stem = Path(source_path).stem if source_path else ""
            with lock:
                call_counts[stem] = call_counts.get(stem, 0) + 1
            if stem == "node_with_box":
                return [
                    {
                        "class_id": 0,
                        "class_name": "herb_aria",
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.2,
                        "h": 0.2,
                        "confidence": 0.92,
                        "source": "hand_seed",
                        "hand_label": True,
                    }
                ]
            return []

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DummyCacheLabeler):
        # First pass: both images are processed and labeled
        res1 = curator.relabel_dataset_batch(incremental=True, force_all=False, max_workers=2)
        assert res1["total"] == 2
        assert res1["labeled"] == 2
        assert res1["skipped"] == 0
        assert call_counts["node_with_box"] == 1
        assert call_counts["node_empty"] == 1

        # Second pass with incremental=True: node_with_box is skipped, node_empty (0 boxes) is re-processed
        res2 = curator.relabel_dataset_batch(incremental=True, force_all=False, max_workers=2)
        assert res2["total"] == 2
        assert res2["skipped"] == 1
        assert res2["labeled"] == 1
        assert call_counts["node_with_box"] == 1
        assert call_counts["node_empty"] == 2

        # Third pass with force_all=True: both are re-processed regardless of cache
        res3 = curator.relabel_dataset_batch(incremental=True, force_all=True, max_workers=2)
        assert res3["total"] == 2
        assert res3["skipped"] == 0
        assert res3["labeled"] == 2
        assert call_counts["node_with_box"] == 2
        assert call_counts["node_empty"] == 3


def test_relabel_dataset_batch_learn_stem_no_longer_emits_unapproved_weak_omni_boxes(tmp_path: Path) -> None:
    """
    5. Verifies `trusted`/`weak` handling for `learn_*` stems vs normal stems
       under sweep E-06 (the old learn-stem exception is gone):
       - `omni_*` synthetic boxes are calibrated as `weak` (`trusted=False`, `weak=True`, conf <= 0.68).
       - An UNAPPROVED weak omni box reaches NEITHER `labels/train/learn_skills_01.txt`
         NOR `labels/train/normal_capture_01.txt` — it stays only in the
         review sidecars. "Adım 2 culls mistakes" was never enforced by code,
         so unapproved weak boxes trained the model.
       - The same omni box WITH an explicit per-box human approval is trusted
         and IS emitted to the learn train .txt (E-06's only door in).
       - A weak box from a non-learn source below the confidence floor must
         NOT leak into train .txt even on a learn_* stem.
    """
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)

    learn_img = _create_dummy_image(curator.images_train_dir / "learn_skills_01.jpg")
    normal_img = _create_dummy_image(curator.images_train_dir / "normal_capture_01.jpg")
    learn_img_approved = _create_dummy_image(curator.images_train_dir / "learn_skills_02.jpg")

    class DummyOmniLabeler:
        def auto_label_frame(
            self, frame: np.ndarray, fast_mode: bool = True, source_path: Any = None, full_scan: bool = False
        ) -> List[Dict[str, Any]]:
            stem = Path(source_path).stem if source_path else ""
            omni_box = {
                "class_id": 13,
                "class_name": "skill_ui_prompt",
                "cx": 0.50,
                "cy": 0.85,
                "w": 0.08,
                "h": 0.08,
                "confidence": 0.65,
                "source": "omni_skill_sweep",
            }
            if stem == "learn_skills_02":
                # Explicit per-box human approval: the E-06 door in.
                omni_box["human_approved"] = True
            return [
                omni_box,
                # Weak box from a NON-learn source, below the floor: must never
                # reach train .txt, not even on a learn_* stem.
                {
                    "class_id": 4,
                    "class_name": "gem_crystal",
                    "cx": 0.15,
                    "cy": 0.30,
                    "w": 0.10,
                    "h": 0.12,
                    "confidence": 0.30,
                    "source": "yolo_model_v2",
                },
            ]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DummyOmniLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[learn_img, normal_img, learn_img_approved],
            max_workers=2,
            emin_threshold=0.85,
            conf_floor=0.45,
        )

    assert res["labeled"] == 3

    learn_rev = json.loads((dataset_dir / "review" / "train" / "learn_skills_01.json").read_text(encoding="utf-8"))
    normal_rev = json.loads((dataset_dir / "review" / "train" / "normal_capture_01.json").read_text(encoding="utf-8"))
    approved_rev = json.loads((dataset_dir / "review" / "train" / "learn_skills_02.json").read_text(encoding="utf-8"))

    # In both unapproved review sidecars, omni_skill_sweep is untrusted (weak)
    assert len(learn_rev["trusted"]) == 0 and len(learn_rev["weak"]) == 2
    assert len(normal_rev["trusted"]) == 0 and len(normal_rev["weak"]) == 2
    assert all(b["trusted"] is False and b["weak"] is True for b in learn_rev["weak"])
    assert all(b["trusted"] is False and b["weak"] is True for b in normal_rev["weak"])
    assert {b["class_id"] for b in learn_rev["weak"]} == {4, 13}
    assert {b["class_id"] for b in normal_rev["weak"]} == {4, 13}

    # E-06: the unapproved weak omni box is emitted to NEITHER train .txt.
    learn_txt = (dataset_dir / "labels" / "train" / "learn_skills_01.txt").read_text(encoding="utf-8").strip()
    assert learn_txt == ""
    normal_txt = (dataset_dir / "labels" / "train" / "normal_capture_01.txt").read_text(encoding="utf-8").strip()
    assert normal_txt == ""

    # ...but the human-approved omni box IS trusted and emitted.
    assert len(approved_rev["trusted"]) == 1 and len(approved_rev["weak"]) == 1
    assert approved_rev["trusted"][0]["class_id"] == 13
    approved_txt = (dataset_dir / "labels" / "train" / "learn_skills_02.txt").read_text(encoding="utf-8").strip()
    approved_lines = [ln for ln in approved_txt.splitlines() if ln.strip()]
    assert len(approved_lines) == 1
    assert int(approved_lines[0].split()[0]) == 13
    # The non-learn weak box (class 4, conf 0.30 < floor) must NOT leak in.
    assert {int(ln.split()[0]) for ln in approved_lines} == {13}
