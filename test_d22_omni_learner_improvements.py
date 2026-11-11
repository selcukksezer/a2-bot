# -*- coding: utf-8 -*-
"""D22: OmniGameLearner improvements — locks for
1. per-frame YOLO detection cache (5 category calls -> 1 inference),
2. single GameFrameGate evaluation per observe_frame (the stateful
   novelty cache used to reject the frame as its own duplicate, so no
   learning sample was ever saved via observe_frame),
3. fast vectorized dHash + bounded oldest-first dedup eviction.
"""
import time
import timeit
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import core.omni_game_learner as ogl
from core.omni_game_learner import OmniGameLearner


def _noise_frame(seed=7, shape=(360, 640, 3)):
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


class _FakeLabeler:
    def __init__(self):
        self.calls = 0

    def auto_label_frame(self, frame, **_kwargs):
        self.calls += 1
        return [
            {"class_id": 10, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
             "confidence": 0.9}
        ]


class _FakeWorker:
    def _walking_is_real(self):
        return True


def _evaluate_all_categories(learner, frame, now, save_stub=None):
    if save_stub is not None:
        learner._save_learning_sample = save_stub
    learner._evaluate_and_save_learning_samples(
        frame, now, current_state="WALKING", current_mode="route",
        monster_info={"has_target": True, "hp_percent": 80, "monster_died": False},
        skill_info={"ready_skills": ["1"], "cooldown_skills": {}, "best_skill": "1"},
        obstacle_info={"obstacle_detected": True, "obstacle_type": "rock",
                       "recommended_steer_deg": 10.0},
        path_info={"is_walking": True, "current_state": "WALKING",
                   "current_mode": "route"},
        vitality_info={"player_hp_percent": 40, "player_mp_percent": 30,
                       "hp_pot_used": True, "mp_pot_used": False},
    )


# --- 1. Detection cache -------------------------------------------------------

def test_detect_real_boxes_cached_per_frame():
    learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
    fake = _FakeLabeler()
    learner._yolo_labeler = fake
    frame = _noise_frame()
    key = OmniGameLearner._compute_dhash(frame)
    assert key is not None
    first = learner._detect_real_boxes(frame, "monsters", cache_key=key)
    for cat in ("skills", "obstacles", "paths", "vitality"):
        assert learner._detect_real_boxes(frame, cat, cache_key=key) == first
    assert fake.calls == 1
    # A different frame (different key) triggers a fresh inference.
    other = _noise_frame(seed=99)
    other_key = OmniGameLearner._compute_dhash(other)
    assert other_key != key
    learner._detect_real_boxes(other, "monsters", cache_key=other_key)
    assert fake.calls == 2


def test_detect_real_boxes_uncached_without_key():
    learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
    fake = _FakeLabeler()
    learner._yolo_labeler = fake
    frame = _noise_frame()
    learner._detect_real_boxes(frame, "monsters")
    learner._detect_real_boxes(frame, "skills")
    assert fake.calls == 2


def test_detect_cache_returns_copy():
    learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
    fake = _FakeLabeler()
    learner._yolo_labeler = fake
    frame = _noise_frame()
    key = OmniGameLearner._compute_dhash(frame)
    boxes = learner._detect_real_boxes(frame, "monsters", cache_key=key)
    boxes.append({"class_id": 99, "box": (0, 0, 1, 1)})
    assert len(learner._detect_real_boxes(frame, "skills", cache_key=key)) == 1
    assert fake.calls == 1


def test_evaluate_runs_one_inference_for_all_categories(tmp_path):
    learner = OmniGameLearner(bot_worker=_FakeWorker(), save_screenshots=True,
                              min_sample_interval_sec=0.0)
    fake = _FakeLabeler()
    learner._yolo_labeler = fake
    learner._depth_avoider = mock.Mock()
    learner._depth_avoider.analyze_forward_corridor.return_value = {
        "obstacle_detected": True, "obstacle_type": "rock",
        "recommended_steer_deg": 10.0}
    saved = []
    with mock.patch.object(ogl, "LEARNING_DIR", tmp_path / "learning"), \
        mock.patch.object(ogl, "TRAIN_IMAGES_DIR", tmp_path / "train-img"), \
        mock.patch.object(ogl, "TRAIN_LABELS_DIR", tmp_path / "train-lbl"), \
        mock.patch.object(ogl, "DATASET_DIR", tmp_path / "dataset"), \
        mock.patch.object(learner, "check_game_frame",
                          return_value=(True, "test-ok")):
        _evaluate_all_categories(
            learner, _noise_frame(), time.time(),
            save_stub=lambda **kw: saved.append(kw.get("category")) or True)
    assert fake.calls == 1
    assert len(saved) == 5  # monsters, skills, obstacles, paths, vitality


# --- 2. Single gate evaluation -----------------------------------------------

def test_observe_frame_evaluates_gate_once(tmp_path):
    learner = OmniGameLearner(bot_worker=None, save_screenshots=True)
    calls = {"n": 0}

    def counting_gate(*a, **k):
        calls["n"] += 1
        return True, "test-ok"

    with mock.patch.object(ogl, "LEARNING_DIR", tmp_path / "learning"), \
        mock.patch.object(ogl, "TRAIN_IMAGES_DIR", tmp_path / "train-img"), \
        mock.patch.object(ogl, "TRAIN_LABELS_DIR", tmp_path / "train-lbl"), \
        mock.patch.object(ogl, "DATASET_DIR", tmp_path / "dataset"), \
        mock.patch.object(learner, "check_game_frame", counting_gate):
        learner.observe_frame(_noise_frame())
    assert calls["n"] == 1


def test_observe_frame_samples_not_self_rejected(tmp_path):
    """Real gate: call 1 accepts the frame and caches its dHash; a second
    evaluation of the same frame rejects it as a duplicate (hamming 0).
    observe_frame must therefore save samples instead of dropping them."""
    learner = OmniGameLearner(bot_worker=_FakeWorker(), save_screenshots=True,
                              min_sample_interval_sec=0.0)
    fake = _FakeLabeler()
    learner._yolo_labeler = fake
    learner._depth_avoider = mock.Mock()
    learner._depth_avoider.analyze_forward_corridor.return_value = {
        "obstacle_detected": True, "obstacle_type": "rock",
        "recommended_steer_deg": 10.0}
    saved = []
    frame = _noise_frame(seed=7)
    assert learner.check_game_frame(frame)[0] is True  # sanity: gate accepts it
    # Fresh gate state for the observe_frame run under test.
    learner._frame_gate = None
    with mock.patch.object(ogl, "LEARNING_DIR", tmp_path / "learning"), \
        mock.patch.object(ogl, "TRAIN_IMAGES_DIR", tmp_path / "train-img"), \
        mock.patch.object(ogl, "TRAIN_LABELS_DIR", tmp_path / "train-lbl"), \
        mock.patch.object(ogl, "DATASET_DIR", tmp_path / "dataset"):
        learner._save_learning_sample = (
            lambda **kw: saved.append(kw.get("category")) or True)
        learner.observe_frame(frame)
    assert len(saved) >= 1, f"expected >=1 saved sample, got {saved}"
    assert learner._frame_gate.stats["rejected_gate3_duplicate_dhash"] == 0


def test_evaluate_gate_ok_param_backward_compatible(tmp_path):
    learner = OmniGameLearner(bot_worker=None, save_screenshots=True)
    saved = []
    learner._save_learning_sample = lambda **kw: saved.append(kw.get("category")) or True
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    # No gate_ok: old behaviour — the real gate rejects the black frame.
    learner._evaluate_and_save_learning_samples(
        black, time.time(), current_state="WALKING", current_mode="route",
        monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
        skill_info={"ready_skills": [], "cooldown_skills": {}},
        obstacle_info={"obstacle_detected": False, "obstacle_type": "none",
                       "recommended_steer_deg": 0.0},
        path_info={"is_walking": False, "current_state": "WALKING",
                   "current_mode": "route"},
        vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
    )
    assert saved == []
    # gate_ok=False drops without consulting the gate at all.
    with mock.patch.object(learner, "check_game_frame",
                           side_effect=AssertionError("gate must not run")):
        learner._evaluate_and_save_learning_samples(
            _noise_frame(), time.time(), current_state="WALKING",
            current_mode="route",
            monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
            skill_info={"ready_skills": [], "cooldown_skills": {}},
            obstacle_info={"obstacle_detected": False, "obstacle_type": "none",
                           "recommended_steer_deg": 0.0},
            path_info={"is_walking": False, "current_state": "WALKING",
                       "current_mode": "route"},
            vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
            gate_ok=False,
        )
    assert saved == []


# --- 3. Fast dHash + bounded dedup -------------------------------------------

def test_dhash_contract_preserved():
    frame = _noise_frame()
    h1 = OmniGameLearner._compute_dhash(frame)
    h2 = OmniGameLearner._compute_dhash(frame)
    assert h1 is not None and h1 == h2  # identical frame -> identical hash
    noisy = frame.copy()
    noisy[100, 100] = np.clip(noisy[100, 100].astype(int) + 3, 0, 255)
    assert OmniGameLearner._compute_dhash(noisy) == h1  # near-duplicate
    hashes = {OmniGameLearner._compute_dhash(_noise_frame(seed=s))
              for s in range(200)}
    assert len(hashes) >= 195  # distinct frames -> (almost always) distinct
    assert OmniGameLearner._compute_dhash(np.array([], dtype=np.uint8)) is None


def test_dhash_is_fast():
    frame = _noise_frame()
    # Old implementation: ~6-7 ms/call on this VM (200 calls > 1.2 s).
    elapsed = timeit.timeit(lambda: OmniGameLearner._compute_dhash(frame),
                            number=200)
    assert elapsed < 1.0, f"dhash 200 calls took {elapsed:.2f}s"


def test_dedup_bounded_oldest_first_eviction():
    learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
    for i in range(601):
        learner._remember_dhash(i)
    assert len(learner._seen_dhashes) == 600
    assert 0 not in learner._seen_dhashes  # oldest evicted
    assert 600 in learner._seen_dhashes   # newest retained
    assert 1 in learner._seen_dhashes     # only the single oldest left
    # Re-adding an existing hash does not grow the history.
    learner._remember_dhash(600)
    assert len(learner._seen_dhashes) == 600
    # Plain-set .clear() (used by existing tests) still works.
    learner._seen_dhashes.clear()
    assert len(learner._seen_dhashes) == 0
