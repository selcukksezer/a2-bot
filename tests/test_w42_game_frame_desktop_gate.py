# -*- coding: utf-8 -*-
"""
W42 Unit Tests for GameFrameGate (`dataset_factory/game_frame_gate.py`).
Verifies rejection of:
1. Windows Desktop screenshots & foreground focus
2. Web Browser screenshots & foreground focus
3. Black screen frames
4. Duplicate / spam frames with perceptual dHash Hamming distance < 6
5. Acceptance of distinct valid game frames vs non-HUD frames
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_factory.game_frame_gate import GameFrameGate


class _Win32ForegroundStub:
    """Win32 `GetForegroundWindow`/`GetWindowTextW` biçiminde (hwnd, pid, title)
    döndüren odak koruyucusu — `set_simulated_game_state` kullanmadan Gate 1'in
    gerçek (hwnd, pid, title) protokolünü tükettiğini kanıtlar."""

    def __init__(self, title: str, pid: int = 424242) -> None:
        self._title = title
        self._pid = pid

    def query_foreground_window_info(self):
        return (1234, self._pid, self._title)

    def is_game_window_focused(self, allow_gatherion_ui: bool = True):
        title_lower = self._title.lower().strip()
        if title_lower in ("program manager", "desktop", "masaüstü"):
            return False, f"Masaüstü odakta: '{self._title}' (PID={self._pid})"
        if "aion" in title_lower:
            return True, f"{self._title} (PID={self._pid})"
        return False, f"Unrecognized foreground window: '{self._title}' (PID={self._pid})"


def _build_valid_game_frame(seed: int = 42, width: int = 1280, height: int = 720) -> np.ndarray:
    """Builds a synthetic 720p gameplay frame with natural 3D texture and a valid player red HP bar.

    NOTE: the RNG + sin/cos meshgrid build costs ~1.7 s per frame, and the
    suite builds the same seeds repeatedly (seed 10 in three tests, seed 20
    in two). The build is therefore cached; see _make_valid_game_frame.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(35, 165, size=(height, width, 3), dtype=np.uint8)

    # Add broad spatial gradients so 9x8 dHash depends on seed
    yy, xx = np.meshgrid(
        np.linspace(0, 1, height, dtype=np.float32),
        np.linspace(0, 1, width, dtype=np.float32),
        indexing="ij",
    )
    pattern = ((np.sin(xx * (seed % 7 + 2) * np.pi) + np.cos(yy * (seed % 5 + 2) * np.pi)) * 35.0).astype(
        np.int16
    )
    frame = np.clip(frame.astype(np.int16) + pattern[:, :, None], 15, 235).astype(np.uint8)

    # Draw bottom-center player red HP bar inside [0.80..0.98]h, [0.25..0.75]w
    bar_y1 = int(height * 0.88)
    bar_y2 = bar_y1 + 12
    bar_x1 = int(width * 0.42)
    bar_x2 = bar_x1 + 140
    # Pure bright red in BGR -> HSV ~(0, 255, 220)
    frame[bar_y1:bar_y2, bar_x1:bar_x2] = (0, 0, 220)
    return frame


@lru_cache(maxsize=16)
def _cached_valid_game_frame(seed: int = 42, width: int = 1280, height: int = 720) -> np.ndarray:
    return _build_valid_game_frame(seed=seed, width=width, height=height)


def _make_valid_game_frame(seed: int = 42, width: int = 1280, height: int = 720) -> np.ndarray:
    """Cached build: same (seed, size) returns an equal frame without
    rebuilding (~1.7 s saved per repeat). The copy matters: callers mutate
    frames in place (duplicate-frame perturbation, taskbar strip), and the
    cached base must never leak a mutation into another test."""
    return _cached_valid_game_frame(seed=seed, width=width, height=height).copy()


def _make_smooth_wallpaper_canvas(width: int = 1280, height: int = 720) -> np.ndarray:
    """Builds a smooth non-game desktop/browser background without HUD elements or dense Canny noise."""
    yy, xx = np.meshgrid(
        np.linspace(55, 135, height, dtype=np.uint8),
        np.linspace(65, 145, width, dtype=np.uint8),
        indexing="ij",
    )
    return np.stack(
        [xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)],
        axis=-1,
    )


def test_gate_rejects_desktop_focus_and_taskbar_canvas() -> None:
    """1. Verifies GameFrameGate rejects Windows desktop focus (Gate 1) and desktop taskbar screenshot (Gate 2)."""
    gate = GameFrameGate()

    # Gate 1: Desktop focus rejection when Program Manager / Desktop is in foreground
    gate.set_simulated_game_state(is_running=True, foreground_title="Program Manager", foreground_pid=424242)
    valid_frame = _make_valid_game_frame(seed=10)
    ok_focus, reason_focus = gate.is_valid_game_frame(valid_frame, require_foreground=True)
    assert ok_focus is False
    assert "desktop_focused" in reason_focus
    assert gate.stats["rejected_gate1_process_or_focus"] == 1

    # Gate 2: Desktop canvas screenshot with Windows taskbar at bottom (y >= 0.955, dark solid strip)
    desktop_frame = _make_smooth_wallpaper_canvas(1280, 720)
    desktop_frame[int(720 * 0.955) :, :] = 25  # Solid dark taskbar strip (std=0 < 10, mean=25 < 55)

    ok_canvas, reason_canvas = gate.is_valid_game_frame(
        desktop_frame,
        require_foreground=False,
        require_novelty=False,
    )
    assert ok_canvas is False
    assert "windows_taskbar_detected" in reason_canvas
    assert gate.stats["rejected_gate2_visual_hud"] == 1


def test_gate_rejects_browser_focus_and_tab_bar_canvas() -> None:
    """2. Verifies GameFrameGate rejects browser focus (Gate 1) and browser tab-bar screenshot (Gate 2)."""
    gate = GameFrameGate(templates_dir=PROJECT_ROOT / "__nonexistent_sub__" / "templates")

    # Gate 1: Browser window title in foreground
    gate.set_simulated_game_state(
        is_running=True,
        foreground_title="New Tab - Google Chrome",
        foreground_pid=424242,
    )
    valid_frame = _make_valid_game_frame(seed=20)
    ok_focus, reason_focus = gate.is_valid_game_frame(valid_frame, require_foreground=True)
    assert ok_focus is False
    assert "blocked_foreground_window" in reason_focus

    # Gate 2: Browser canvas with bright uniform top tab/address bar strip (y < 0.06, mean > 210, std < 8)
    # and smooth vertical ramp below so no step-edge Canny ring forms in the top-right ROI
    top_h = int(720 * 0.06)
    ramp = np.linspace(235, 140, 720 - top_h, dtype=np.uint8)[:, None]
    body = np.repeat(ramp, 1280, axis=1)
    top_strip = np.full((top_h, 1280), 235, dtype=np.uint8)
    gray_browser = np.vstack([top_strip, body])
    browser_frame = cv2.cvtColor(gray_browser, cv2.COLOR_GRAY2BGR)

    ok_canvas, reason_canvas = gate.is_valid_game_frame(
        browser_frame,
        require_foreground=False,
        require_novelty=False,
    )
    assert ok_canvas is False
    assert "browser_tab_bar_detected" in reason_canvas


def test_gate_rejects_black_screen_frame() -> None:
    """3. Verifies GameFrameGate rejects pure black and near-black screens (mean intensity < 8.0)."""
    gate = GameFrameGate()
    gate.set_simulated_game_state(is_running=True, foreground_title="AION 2", foreground_pid=424242)

    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ok_black, reason_black = gate.is_valid_game_frame(black_frame, require_foreground=True)
    assert ok_black is False
    assert "black_frame" in reason_black

    near_black_frame = np.full((720, 1280, 3), 5, dtype=np.uint8)
    ok_near, reason_near = gate.is_valid_game_frame(near_black_frame, require_foreground=True)
    assert ok_near is False
    assert "black_frame" in reason_near
    assert gate.stats["rejected_gate2_visual_hud"] == 2


def test_gate_rejects_duplicate_frames_with_dhash_under_6() -> None:
    """4. Verifies GameFrameGate rejects identical or near-identical frames with perceptual dHash < 6."""
    gate = GameFrameGate(min_dhash_distance=6)
    gate.set_simulated_game_state(is_running=True, foreground_title="AION 2", foreground_pid=424242)

    frame1 = _make_valid_game_frame(seed=77)
    ok1, reason1 = gate.is_valid_game_frame(frame1, require_foreground=True, require_novelty=True)
    assert ok1 is True
    assert "valid_game_frame" in reason1
    assert gate.hash_cache_size == 1

    # Exact duplicate (dHash distance == 0 < 6)
    frame_dup = frame1.copy()
    # Perturb a few pixels so arrays are not byte-identical but dHash distance remains < 6
    frame_dup[100:104, 100:104] = np.clip(frame_dup[100:104, 100:104].astype(np.int16) + 2, 0, 255).astype(
        np.uint8
    )
    dist = gate.hamming_distance(gate.compute_dhash(frame1), gate.compute_dhash(frame_dup))
    assert dist < 6

    ok2, reason2 = gate.is_valid_game_frame(frame_dup, require_foreground=True, require_novelty=True)
    assert ok2 is False
    assert "duplicate_frame" in reason2
    assert gate.stats["rejected_gate3_duplicate_dhash"] == 1


def test_gate_accepts_distinct_valid_game_frames_and_rejects_no_hud() -> None:
    """5. Verifies GameFrameGate accepts visually distinct game frames (dHash >= 6) and rejects HUD-less noise."""
    gate = GameFrameGate(min_dhash_distance=6)
    gate.set_simulated_game_state(is_running=True, foreground_title="AION 2", foreground_pid=424242)

    frame_a = _make_valid_game_frame(seed=1)
    frame_b = cv2.flip(_make_valid_game_frame(seed=5), 1)
    assert gate.hamming_distance(gate.compute_dhash(frame_a), gate.compute_dhash(frame_b)) >= 6

    ok_a, _ = gate.is_valid_game_frame(frame_a, require_foreground=True, require_novelty=True)
    ok_b, _ = gate.is_valid_game_frame(frame_b, require_foreground=True, require_novelty=True)
    assert ok_a is True
    assert ok_b is True
    assert gate.stats["accepted"] == 2

    # Non-HUD mid-tone gradient frame without any AION UI elements
    # (same smooth-wallpaper builder as the desktop-canvas test; the
    # inline meshgrid here used to duplicate it with near-identical ranges).
    no_hud_frame = _make_smooth_wallpaper_canvas(1280, 720)
    ok_no_hud, reason_no_hud = gate.is_valid_game_frame(no_hud_frame, require_foreground=True, require_novelty=True)
    assert ok_no_hud is False
    assert "no_hud_evidence" in reason_no_hud


def test_frame_cache_returns_independent_copies() -> None:
    """The _make_valid_game_frame cache must never leak an in-place
    mutation from one test into another: two builds of the same seed are
    equal but independent arrays."""
    first = _make_valid_game_frame(seed=10)
    second = _make_valid_game_frame(seed=10)
    assert np.array_equal(first, second)
    assert first is not second
    first[0, 0] = (255, 255, 255)
    third = _make_valid_game_frame(seed=10)
    assert np.array_equal(third, second)
    assert not np.array_equal(third, first)
    """Gate 1 `set_simulated_game_state` olmadan da doğrulanır: gerçek foreground
    sorgusu ve enjekte edilen oyun-durumu fonksiyonu üzerinden reddetme/kabul."""
    if os.name == "nt":
        pytest.skip("Gerçek Win32 foreground sorgusu yalnızca Windows'ta anlamlıdır")

    # (a) Gerçek ForegroundFocusGuard, simülasyon yok: Linux'ta Win32 olmadığı için
    # query_foreground_window_info() (0, 0, "") döner ve Gate 1 foreground_empty
    # ile reddeder — simüle edilmiş "Program Manager" başlığı kullanılmaz.
    gate = GameFrameGate(is_game_running_fn=lambda: True)
    assert gate.focus_guard.query_foreground_window_info() == (0, 0, "")
    ok, reason = gate.check_gate1_process_and_focus()
    assert ok is False
    assert "foreground_empty" in reason

    valid_frame = _make_valid_game_frame(seed=10)
    ok_frame, reason_frame = gate.is_valid_game_frame(valid_frame, require_foreground=True)
    assert ok_frame is False
    assert "foreground_empty" in reason_frame
    assert gate.stats["rejected_gate1_process_or_focus"] == 1

    # (b) Oyun çalışmıyorken (enjekte edilen fonksiyon) Gate 1 game_not_running döner.
    stopped_gate = GameFrameGate(is_game_running_fn=lambda: False)
    ok_stopped, reason_stopped = stopped_gate.check_gate1_process_and_focus()
    assert ok_stopped is False
    assert "game_not_running" in reason_stopped


def test_gate1_consumes_win32_shaped_foreground_report_without_simulation() -> None:
    """Gate 1, Win32'nin döndürdüğü (hwnd, pid, title) raporunu simülasyon
    yardımcısı olmadan tüketir: masaüstü başlığı reddedilir, AION 2 başlığı
    geçerli kareyi kabul ettirir."""
    # Masaüstü odakta: Gate 1 desktop_focused ile reddeder.
    desktop_gate = GameFrameGate(
        focus_guard=_Win32ForegroundStub("Program Manager"),
        is_game_running_fn=lambda: True,
    )
    ok_desktop, reason_desktop = desktop_gate.is_valid_game_frame(
        _make_valid_game_frame(seed=10), require_foreground=True
    )
    assert ok_desktop is False
    assert "desktop_focused" in reason_desktop
    assert desktop_gate.stats["rejected_gate1_process_or_focus"] == 1

    # Tarayıcı odakta: Gate 1 blocked_foreground_window ile reddeder.
    browser_gate = GameFrameGate(
        focus_guard=_Win32ForegroundStub("New Tab - Google Chrome"),
        is_game_running_fn=lambda: True,
    )
    ok_browser, reason_browser = browser_gate.is_valid_game_frame(
        _make_valid_game_frame(seed=20), require_foreground=True
    )
    assert ok_browser is False
    assert "blocked_foreground_window" in reason_browser

    # Oyun odakta: Gate 1 geçer, geçerli kare kabul edilir.
    game_gate = GameFrameGate(
        focus_guard=_Win32ForegroundStub("AION 2"),
        is_game_running_fn=lambda: True,
    )
    ok_game, reason_game = game_gate.is_valid_game_frame(
        _make_valid_game_frame(seed=30), require_foreground=True, require_novelty=True
    )
    assert ok_game is True
    assert "valid_game_frame" in reason_game
    assert game_gate.stats["accepted"] == 1
