# -*- coding: utf-8 -*-
"""D69 (label-tools) improvement tests.

Covers:
- I1: calibrate_single_box with a precomputed stem_ctx is identical to the
  plain per-box call (recalibrate_dataset_trust.stem_context).
- I2: grouped _dedupe_and_nms is identical to the old flat O(N^2) scan.
- I3: _find_learning_json with a caller-built index is authoritative — no
  per-miss folder rescan; index-less callers still get the fallback.
- I4: relabel_weak_only builds YOLOAutoLabeler lazily — zero constructions
  when there is no work, exactly one when there is.

NOTE (2026-11-08 21:12 Selçuk rule): coding agents do not run pytest while
250 agents are active — these tests are written for the dedicated test
agents to run.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory.label_trust import annotation_confidence, slot_iou
from tools.bootstrap_learn_labels import _find_learning_json
from tools.recalibrate_dataset_trust import (
    _dedupe_and_nms,
    calibrate_single_box,
    stem_context,
)


# ---------------------------------------------------------------- I1
_STEMS = [
    "boot_478467_000_aria_3d_crest",
    "learn_paths_1793975849145_424",
    "learn_skills_1794032002457_474",
    "learn_vitality_1794032002457_9",
    "boot_123_targena_gather",
    "learn_essence_9_aethertapping",
    "boot_999_orichalcum_gather",
    "unknown_stem",
]


def _sample_boxes():
    boxes = []
    for i in range(120):
        boxes.append(
            {
                "class_id": i % 30,
                "cx": 0.1 + (i % 8) * 0.1,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.5 + (i % 5) * 0.1,
                "source": (
                    "chromatic_salient_3d_resource"
                    if i % 3 == 0
                    else "vision_3d_crest"
                    if i % 3 == 1
                    else "legacy_auto"
                ),
                "trusted": bool(i % 2),
                "hand_label": i % 17 == 0,
            }
        )
    return boxes


def test_stem_ctx_matches_plain_calibrate():
    """Precomputed stem context must not change any calibration outcome."""
    for stem in _STEMS:
        ctx = stem_context(stem)
        for box in _sample_boxes():
            plain_box, plain_action = calibrate_single_box(box, stem=stem)
            ctx_box, ctx_action = calibrate_single_box(
                box, stem=stem, stem_ctx=ctx
            )
            assert ctx_action == plain_action
            assert ctx_box == plain_box


def test_stem_ctx_matches_plain_calibrate_purge_off():
    """Same equivalence with purge_synthetic=False."""
    for stem in _STEMS:
        ctx = stem_context(stem)
        for box in _sample_boxes():
            plain_box, plain_action = calibrate_single_box(
                box, stem=stem, purge_synthetic=False
            )
            ctx_box, ctx_action = calibrate_single_box(
                box, stem=stem, purge_synthetic=False, stem_ctx=ctx
            )
            assert ctx_action == plain_action
            assert ctx_box == plain_box


# ---------------------------------------------------------------- I2
def _flat_nms_reference(boxes, iou_thresh=0.65):
    """The pre-D69 flat O(N^2) implementation, kept as a test oracle."""
    def _rank(b):
        is_t = 1.0 if (b.get("trusted") is True or b.get("hand_label") is True) else 0.0
        return (is_t, annotation_confidence(b))

    sorted_boxes = sorted(boxes, key=_rank, reverse=True)
    kept = []
    dropped = 0
    for b in sorted_boxes:
        suppress = False
        for k in kept:
            if int(b.get("class_id", -1)) == int(k.get("class_id", -1)):
                if slot_iou(b, k) > iou_thresh:
                    suppress = True
                    dropped += 1
                    break
        if not suppress:
            kept.append(b)
    return kept, dropped


def _random_boxes(n, seed):
    rng = random.Random(seed)
    return [
        {
            "class_id": rng.randrange(12),
            "cx": rng.random(),
            "cy": rng.random(),
            "w": 0.3,
            "h": 0.3,
            "confidence": rng.random(),
            "trusted": rng.randrange(5) == 0,
            "hand_label": False,
        }
        for _ in range(n)
    ]


@pytest.mark.parametrize("n,seed", [(0, 1), (1, 2), (50, 3), (600, 7), (300, 8)])
def test_grouped_nms_matches_flat_reference(n, seed):
    boxes = _random_boxes(n, seed)
    kept, dropped = _dedupe_and_nms(boxes)
    ref_kept, ref_dropped = _flat_nms_reference(boxes)
    assert dropped == ref_dropped
    assert kept == ref_kept


def test_grouped_nms_hand_computed():
    """Same-class overlap suppresses; cross-class overlap does not; trusted wins."""
    boxes = [
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.9, "trusted": True, "hand_label": False},
        {"class_id": 0, "cx": 0.52, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.8, "trusted": False, "hand_label": False},
        {"class_id": 1, "cx": 0.52, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.7, "trusted": False, "hand_label": False},
    ]
    kept, dropped = _dedupe_and_nms(boxes)
    assert dropped == 1
    assert [b["class_id"] for b in kept] == [0, 1]
    assert kept[0]["confidence"] == 0.9


def test_grouped_nms_malformed_class_id_kept():
    """A non-numeric class_id must not crash NMS (old flat scan raised)."""
    boxes = [
        {"class_id": "abc", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.9, "trusted": False, "hand_label": False},
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.8, "trusted": False, "hand_label": False},
    ]
    kept, dropped = _dedupe_and_nms(boxes)
    assert dropped == 0
    assert len(kept) == 2


# ---------------------------------------------------------------- I3
def test_find_learning_json_index_is_authoritative(tmp_path):
    """With an index, a miss stays a miss — no fallback folder rescan."""
    root = tmp_path / "game_learning"
    sub = root / "monsters"
    sub.mkdir(parents=True)
    (sub / "learn_monsters_hidden.json").write_text("{}", encoding="utf-8")
    # Index does NOT contain the stem, even though the file exists on disk.
    assert _find_learning_json(root, "learn_monsters_hidden", "monsters", {}) is None
    # Index-less callers still get the one-level fallback.
    found = _find_learning_json(root, "learn_monsters_hidden", "monsters", None)
    assert found == sub / "learn_monsters_hidden.json"
    # An index hit is returned as-is.
    index = {"learn_monsters_hidden": sub / "learn_monsters_hidden.json"}
    assert (
        _find_learning_json(root, "learn_monsters_hidden", "monsters", index)
        == sub / "learn_monsters_hidden.json"
    )


# ---------------------------------------------------------------- I4
class _CountingLabeler:
    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def auto_label_frame(self, img):
        return []

    def label_frame(self, img):
        return []


def _write_image(path: Path):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    assert cv2.imwrite(str(path), img)


def test_relabel_weak_only_no_work_builds_no_labeler(tmp_path, monkeypatch):
    """Empty dataset (no review dirs): YOLOAutoLabeler is never constructed."""
    import dataset_factory.yolo_auto_labeler as yal_mod
    from tools.relabel_weak_only import main

    _CountingLabeler.instances = 0
    monkeypatch.setattr(yal_mod, "YOLOAutoLabeler", _CountingLabeler)
    monkeypatch.setattr(
        sys, "argv", ["relabel_weak_only.py", "--dataset", str(tmp_path)]
    )
    assert main() == 0
    assert _CountingLabeler.instances == 0


def test_relabel_weak_only_labeler_built_once(tmp_path, monkeypatch):
    """Two weak-only frames: the labeler is constructed exactly once."""
    import dataset_factory.yolo_auto_labeler as yal_mod
    from tools.relabel_weak_only import main

    for split in ("train",):
        (tmp_path / "review" / split).mkdir(parents=True)
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        for stem in ("frame_a", "frame_b"):
            payload = {
                "stem": stem,
                "split": split,
                "trusted": [],
                "weak": [
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
                     "confidence": 0.6, "trusted": False, "weak": True,
                     "source": "legacy_auto"}
                ],
            }
            (tmp_path / "review" / split / f"{stem}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            _write_image(tmp_path / "images" / split / f"{stem}.jpg")

    _CountingLabeler.instances = 0
    monkeypatch.setattr(yal_mod, "YOLOAutoLabeler", _CountingLabeler)
    monkeypatch.setattr(
        sys, "argv", ["relabel_weak_only.py", "--dataset", str(tmp_path)]
    )
    assert main() == 0
    assert _CountingLabeler.instances == 1
