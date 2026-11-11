# -*- coding: utf-8 -*-
"""D33 regression tests for core/vision.py performance work.

Locks in the behaviour-preserving optimisations:
1. `is_map_open` runs the cheap chrome check before the header template
   pyramid and reuses cached resized header templates (no `cv2.resize`
   on repeat calls). Verdicts are tri-state (P1-3/K-07): `True` (map open),
   `False` (gameplay HUD proves the map is closed), `None` (no evidence —
   a no-evidence frame must NOT be reported as "map closed", which was
   fail-open: the bot kept gathering on a loading/dark/menu frame).
2. `detect_gather_prompt` passes its already-known map verdict to
   `detect_floating_resource_icon` instead of re-running `is_map_open`
   on the same frame (exactly one map check per call).
3. `verify_crest_alignment_proof` no-proof exits return the same dict as
   the shared `_unverified_crest_proof` helper.
4. `detect_3d_gather_crests` resolves the exemplar-memory singleton once
   per call, not once per candidate.

Frames are small synthetic 640x360 BGR images (fast matchTemplate) unless
noted; no live screen capture is involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.vision import MAP_HEADER_SCALES, VisionEngine  # noqa: E402


@pytest.fixture(scope="module")
def eng() -> VisionEngine:
    return VisionEngine()


def _noise_frame(seed: int = 3, h: int = 360, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Uniform 96..160 grey noise: no red HP / blue MP fill, no dark slot,
    # and canvas value not > left value + 28, so the map chrome fails.
    return rng.integers(96, 161, size=(h, w, 3), dtype=np.uint8)


def _map_open_frame(h: int = 360, w: int = 640) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Bright canvas (V~180) over x 0.22..0.90, y 0.16..0.78.
    frame[int(h * 0.16):int(h * 0.78), int(w * 0.22):int(w * 0.90)] = (80, 180, 120)
    # Dark left list (V=60, S=0), y 0.18..0.72, x 0.012..0.16.
    frame[int(h * 0.18):int(h * 0.72), int(w * 0.012):int(w * 0.16)] = (60, 60, 60)
    # ~1% bright "text" pixels inside the left list.
    rng = np.random.default_rng(7)
    ly1, ly2 = int(h * 0.18), int(h * 0.72)
    lx1, lx2 = int(w * 0.012), int(w * 0.16)
    n = (ly2 - ly1) * (lx2 - lx1) // 100
    frame[rng.integers(ly1, ly2, size=n), rng.integers(lx1, lx2, size=n)] = (255, 255, 255)
    # Top strip darker than canvas by > 35.
    frame[int(h * 0.012):int(h * 0.10), int(w * 0.20):int(w * 0.80)] = (40, 40, 40)
    # Top-right close X bright.
    frame[int(h * 0.012):int(h * 0.070), int(w * 0.955):int(w * 0.995)] = (255, 255, 255)
    # HP/MP slots mid-grey: no fill, no dark slot.
    frame[int(h * 0.87):int(h * 0.92), int(w * 0.36):int(w * 0.47)] = (100, 100, 100)
    frame[int(h * 0.87):int(h * 0.92), int(w * 0.52):int(w * 0.63)] = (100, 100, 100)
    return frame


def _gameplay_frame(h: int = 360, w: int = 640) -> np.ndarray:
    frame = _noise_frame(seed=5, h=h, w=w)
    frame[int(h * 0.87):int(h * 0.92), int(w * 0.36):int(w * 0.47)] = (0, 0, 210)
    frame[int(h * 0.87):int(h * 0.92), int(w * 0.52):int(w * 0.63)] = (210, 160, 0)
    return frame


def test_is_map_open_verdicts_tri_state(eng: VisionEngine) -> None:
    assert eng.is_map_open(_map_open_frame()) is True
    # P1-3 (K-07): kanıtsız karede fallthrough False -> None. Gürültü
    # karesinde ne harita-açık ne oyun-ekranı (HP/MP barı) kanıtı var;
    # "harita kapalı" demek fail-open'dı (core/vision.py:1909-1916).
    assert eng.is_map_open(_noise_frame()) is None
    # Oyun karesi: HP+MP barı birlikte haritanın kapalı olduğunun kanıtıdır.
    assert eng.is_map_open(_gameplay_frame()) is False


def test_is_map_open_header_path_still_works(eng: VisionEngine) -> None:
    """Header template matching still fires when chrome fails (reorder proof)."""
    tmpl = eng.map_header_legacy_template
    assert tmpl is not None and tmpl.shape[1] >= 60
    frame = _noise_frame(seed=11)
    th, tw = tmpl.shape[:2]
    # Paste the legacy header (as BGR) into the top strip, scale 1.0.
    y0, x0 = 5, 200
    frame[y0:y0 + th, x0:x0 + tw] = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
    assert eng.is_map_open(frame) is True


def test_map_header_scales_cached_no_resize_on_repeat(eng: VisionEngine, monkeypatch) -> None:
    tmpl = eng.map_header_legacy_template
    assert tmpl is not None
    first = eng._get_map_header_scales(tmpl)
    second = eng._get_map_header_scales(tmpl)
    assert first is second
    assert [s for s, _ in first] == list(MAP_HEADER_SCALES)
    assert len(first) == len(MAP_HEADER_SCALES)
    # Every cached pyramid entry matches the resize formula of the old loop.
    th, tw = tmpl.shape[:2]
    for s, resized in first:
        assert resized.shape == (max(6, int(round(th * s))), max(10, int(round(tw * s))))

    calls = {"n": 0}
    real_resize = cv2.resize

    def counting_resize(src, dsize, *args, **kwargs):
        calls["n"] += 1
        return real_resize(src, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", counting_resize)
    # Warm the cache, then a repeat map check must not resize anything.
    eng.is_map_open(_noise_frame())
    calls["n"] = 0
    eng.is_map_open(_noise_frame(seed=12))
    assert calls["n"] == 0


def test_detect_gather_prompt_runs_map_check_once(eng: VisionEngine, monkeypatch) -> None:
    calls = {"n": 0}
    real = eng.is_map_open

    def counting(frame=None):
        calls["n"] += 1
        return real(frame)

    monkeypatch.setattr(eng, "is_map_open", counting)
    found, conf, bbox = eng.detect_gather_prompt(frame=_noise_frame())
    assert (found, conf, bbox) == (False, 0, None) or found is False
    assert calls["n"] == 1


def test_floating_icon_map_open_param_skips_check(eng: VisionEngine, monkeypatch) -> None:
    calls = {"n": 0}
    real = eng.is_map_open

    def counting(frame=None):
        calls["n"] += 1
        return real(frame)

    monkeypatch.setattr(eng, "is_map_open", counting)
    # Precomputed verdict: no map check at all.
    res = eng.detect_floating_resource_icon(frame=_noise_frame(), map_open=False)
    assert calls["n"] == 0
    assert res[0] is False
    res_open = eng.detect_floating_resource_icon(frame=_noise_frame(), map_open=True)
    assert calls["n"] == 0
    assert res_open == (False, 0.0, None, eng.last_detected_gather_key)
    # Default (None) still runs the check exactly once.
    eng.detect_floating_resource_icon(frame=_noise_frame())
    assert calls["n"] == 1


def test_detect_gather_prompt_rejects_map_open_frame(eng: VisionEngine) -> None:
    assert eng.detect_gather_prompt(frame=_map_open_frame()) == (False, 0, None)


def test_unverified_crest_proof_matches_helper(eng: VisionEngine) -> None:
    for raw in (0.74, 0.85, 1.0, 0.50):
        got = eng.verify_crest_alignment_proof((100, 100), frame_bgr=None, raw_score=raw)
        assert got == eng._unverified_crest_proof(raw)
        assert got["has_proof"] is False
        assert got["trusted"] is False and got["weak"] is True
        assert 0.50 <= got["confidence"] <= 0.68
    # No nameplate / prompt below the crest on a blank frame either.
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    got = eng.verify_crest_alignment_proof((320, 100), frame_bgr=blank, raw_score=0.85)
    assert got == eng._unverified_crest_proof(0.85)


def test_crest_exemplar_memory_resolved_once(eng: VisionEngine, monkeypatch) -> None:
    import dataset_factory.exemplar_memory as em

    calls = {"n": 0}

    class _PermissiveMemory:
        """Stub memory: the exact pasted template crop is rejected as a
        negative patch by the real exemplar memory (class 15
        'wrong_node_purge', sim 0.87 on this synthetic crop), so no
        candidate would ever reach the proof step. The property under
        test is the call count, not the memory's verdict, so a
        permissive stub keeps candidates alive past verification."""

        def is_negative_patch(self, crop):
            return (False, 0.0, {})

        def classify_patch(self, crop):
            return (0, "", 0.0)

    def counting(*args, **kwargs):
        calls["n"] += 1
        return _PermissiveMemory()

    monkeypatch.setattr(em, "get_exemplar_memory", counting)
    tmpl = eng.gather_crest_template
    assert tmpl is not None
    frame = _noise_frame(seed=21)
    th, tw = tmpl.shape[:2]
    # Paste the exact crest template at 5 well-separated ROI positions.
    for (x0, y0) in ((100, 80), (300, 80), (500, 80), (150, 200), (400, 200)):
        frame[y0:y0 + th, x0:x0 + tw] = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
    found = eng.detect_3d_gather_crests(frame=frame)
    assert len(found) >= 2  # several pasted crests survive NMS + proof
    assert calls["n"] == 1
