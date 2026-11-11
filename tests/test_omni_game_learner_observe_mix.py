# -*- coding: utf-8 -*-
"""
Unit tests for `OmniGameLearner.observe_mix` train-image write gate (S-0101).

S-0101: `cv2.imencode` returning False, or `write_bytes` raising, used to be
swallowed by a bare `logger.debug`, after which the sample was unconditionally
counted (`total_learned_frames += 1`) and reported as saved. A sample now
counts as saved only when the train image was actually written.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.omni_game_learner import OmniGameLearner


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _install_mix_mocks(monkeypatch) -> None:
    import dataset_factory.multi_mix as multi_mix

    monkeypatch.setattr(
        multi_mix,
        "fuse_sources",
        lambda sources: {"annotations": [{"class_id": 1, "box": (0.5, 0.5, 0.1, 0.1)}]},
    )
    monkeypatch.setattr(
        multi_mix,
        "commit_mix_sample",
        lambda **kwargs: {
            "train_box_count": 2,
            "weak_box_count": 0,
            "label_path": "labels/train/mix_1.txt",
        },
    )


def _learner() -> OmniGameLearner:
    return OmniGameLearner(bot_worker=None, save_screenshots=False)


def test_train_image_write_success_counts_sample(monkeypatch, tmp_path):
    _install_mix_mocks(monkeypatch)
    learner = _learner()
    result = learner.observe_mix(
        _frame(), {"a": {}}, force=True, dataset_dir=tmp_path
    )
    assert result["saved"] is True
    assert learner.total_learned_frames == 1
    assert list((tmp_path / "images" / "train").glob("*.jpg"))


def test_train_image_encode_failure_not_counted(monkeypatch, tmp_path):
    _install_mix_mocks(monkeypatch)
    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, None))
    learner = _learner()
    result = learner.observe_mix(
        _frame(), {"a": {}}, force=True, dataset_dir=tmp_path
    )
    assert result["saved"] is False
    assert result["reason"] == "train_image_write_failed"
    assert learner.total_learned_frames == 0
    assert not list((tmp_path / "images" / "train").glob("*.jpg"))


def test_train_image_write_bytes_failure_not_counted(monkeypatch, tmp_path):
    _install_mix_mocks(monkeypatch)

    def _boom(self, data):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    learner = _learner()
    result = learner.observe_mix(
        _frame(), {"a": {}}, force=True, dataset_dir=tmp_path
    )
    assert result["saved"] is False
    assert result["reason"] == "train_image_write_failed"
    assert learner.total_learned_frames == 0
