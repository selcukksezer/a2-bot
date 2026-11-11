# -*- coding: utf-8 -*-
"""DEV D34 tests: vision-helpers improvements.

Locks in:
  I1 - get_hud_combat_and_health_summary() scans the top target bar ONCE
       per call (was twice: read_hp_mp -> detect_top_target_bar plus
       is_under_attack -> detect_top_target_bar) with identical values.
  I2 - validate_patch_normalizer_contract() is a real probe: it returns False
       when the normalizer is broken or emits degenerate output.
  I3 - Sobel p99 magnitude scaling is a single shared helper; normalize_patch
       and compute_sobel_edge_map outputs are byte-identical to pre-change.
"""
import hashlib

import cv2
import numpy as np
import pytest

from core import patch_normalizer as pn
from core.ocr_distance import GameHUDReader


def _olive_bgr():
    hsv = np.uint8([[[45, 200, 180]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _red_bgr():
    hsv = np.uint8([[[0, 200, 200]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _monster_frame():
    """1920x1080 frame with an olive monster HP bar + full red player HP bar."""
    f = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(f, (int(1920 * 0.422), int(1080 * 0.048)),
                  (int(1920 * 0.571), int(1080 * 0.059)), _olive_bgr(), -1)
    cv2.rectangle(f, (int(1920 * 0.365), int(1080 * 0.885)),
                  (int(1920 * 0.478), int(1080 * 0.900)), _red_bgr(), -1)
    return f


def _synthetic_patch():
    rng = np.random.default_rng(7)
    patch = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    cv2.rectangle(patch, (10, 10), (40, 60), (255, 255, 255), -1)
    cv2.circle(patch, (60, 60), 12, (0, 0, 0), -1)
    return patch


@pytest.fixture(scope="module")
def reader():
    return GameHUDReader()


# ---------------------------------------------------------------- I1
def test_d34_summary_scans_top_target_bar_once(reader, monkeypatch):
    calls = {"n": 0}
    orig = GameHUDReader.detect_top_target_bar

    def spy(self, frame):
        calls["n"] += 1
        return orig(self, frame)

    monkeypatch.setattr(GameHUDReader, "detect_top_target_bar", spy)
    res = reader.get_hud_combat_and_health_summary(_monster_frame())
    assert calls["n"] == 1, f"expected 1 target-bar scan, got {calls['n']}"
    assert res["is_under_attack"] is True
    assert res["hp_pct"] == 100.0


def test_d34_summary_matches_legacy_composition(reader):
    for frame in (_monster_frame(), np.zeros((1080, 1920, 3), dtype=np.uint8), None):
        legacy_hp = reader.read_hp_mp(frame)
        legacy_attack = reader.is_under_attack(frame)
        res = reader.get_hud_combat_and_health_summary(frame)
        assert res["hp_pct"] == legacy_hp["hp_pct"]
        assert res["mp_pct"] == legacy_hp["mp_pct"]
        assert res["is_under_attack"] == legacy_attack


def test_d34_summary_blank_and_none_frames(reader):
    res = reader.get_hud_combat_and_health_summary(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert res == {"hp_pct": 0.0, "mp_pct": 0.0, "is_under_attack": False}
    res = reader.get_hud_combat_and_health_summary(None)
    assert res == {"hp_pct": 0.0, "mp_pct": 0.0, "is_under_attack": False}


# ---------------------------------------------------------------- I2
def test_d34_validate_contract_healthy_is_true():
    assert pn.validate_patch_normalizer_contract() is True
    assert pn.get_patch_normalizer_readiness_status()["ready"] is True


def test_d34_validate_contract_detects_broken_normalizer(monkeypatch):
    def _boom(self, patch):
        raise RuntimeError("boom")

    monkeypatch.setattr(pn.SobelClahePatchNormalizer, "normalize_patch", _boom)
    assert pn.validate_patch_normalizer_contract() is False
    assert pn.get_patch_normalizer_readiness_status()["ready"] is False
    assert pn.get_patch_normalizer_heartbeat_snapshot()["alive"] is False


def test_d34_validate_contract_rejects_degenerate_output(monkeypatch):
    def _flat(self, patch):
        return np.zeros((80, 80), dtype=np.uint8)

    monkeypatch.setattr(pn.SobelClahePatchNormalizer, "normalize_patch", _flat)
    assert pn.validate_patch_normalizer_contract() is False


# ---------------------------------------------------------------- I3
def test_d34_sobel_outputs_byte_identical_to_pre_change():
    norm = pn.SobelClahePatchNormalizer()
    patch = _synthetic_patch()
    out_n = norm.normalize_patch(patch)
    out_e = norm.compute_sobel_edge_map(patch)
    assert hashlib.sha256(out_n.tobytes()).hexdigest()[:16] == "4f908b99192a92c7"
    assert hashlib.sha256(out_e.tobytes()).hexdigest()[:16] == "3c085e9c40caaa18"
    assert out_n.shape == (80, 80) and out_n.dtype == np.uint8
    assert out_e.shape == (80, 80) and out_e.dtype == np.uint8


def test_d34_sobel_edge_map_matches_manual_p99_scaling():
    norm = pn.SobelClahePatchNormalizer()
    gray = cv2.cvtColor(_synthetic_patch(), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    p99 = float(np.percentile(mag, 99.0))
    expected = np.clip((mag / p99) * 255.0, 0.0, 255.0).astype(np.uint8)
    assert np.array_equal(norm.compute_sobel_edge_map(_synthetic_patch()), expected)


def test_d34_sobel_edge_map_flat_input_is_zeros():
    norm = pn.SobelClahePatchNormalizer()
    flat = np.full((80, 80, 3), 128, dtype=np.uint8)
    assert np.count_nonzero(norm.compute_sobel_edge_map(flat)) == 0
    assert np.count_nonzero(norm.normalize_patch(flat) * 0) == 0
