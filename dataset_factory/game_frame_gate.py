# -*- coding: utf-8 -*-
"""
Intelligent Game Frame Gate (`dataset_factory/game_frame_gate.py`).
==================================================================
Multi-stage capture guard for AION 2 data ingestion and dataset sanitization:

1. Kapı 1 (Process & Focus Guard):
   - Verifies verified AION 2 game process is active via `is_game_running()`.
   - Verifies active foreground window via `ForegroundFocusGuard.is_game_window_focused()`.
   - Immediately rejects non-game foreground contexts: Windows desktop (`Program Manager`,
     `Desktop`), `explorer.exe`, `chrome.exe`, `code.exe`, `cursor.exe`, `msedge.exe`,
     terminal shells, or Gatherion UI itself.
   - Bypassed when `require_foreground=False` (used for offline dataset validation & purging).

2. Kapı 2 (Visual HUD Evidence):
   - Inspects frame for genuine AION 2 UI features:
     a) Minimap radar circle / circular boundary and white/cyan player compass arrow.
     b) Player vitality HP (red) and MP (blue) action frames in bottom-center HUD.
     c) Overhead character HP or target entity HP bar.
     d) Skill bar action grid / slot bounding boxes.
     e) In-game interactive prompts (F Gather, 3D Winged Gather Crest, E icon).
     f) Natural 3D game entity texture for training crops.
   - Rejects blank black screens, solid/flat colors, desktop icon canvases, or browser windows.

3. Kapı 3 (Visual Novelty / dHash Gate):
   - Computes 64-bit gradient perceptual difference hash (dHash).
   - Maintains an in-memory LRU hash cache with LRU eviction.
   - Rejects frames with Hamming distance `dHash < 6` to eliminate duplicate/spam captures.

Exclusive Owner: Worker 15 (`dataset_factory/game_frame_gate.py`)
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
from ctypes import wintypes
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from core.dpi_focus_guard import ForegroundFocusGuard
from core.game_gpu_throttle import is_game_running

logger = logging.getLogger("DatasetFactory.GameFrameGate")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] [GameFrameGate] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ==============================================================================
# Blocked Foreground Executables and Window Substrings
# ==============================================================================
BLOCKED_FOREGROUND_EXECUTABLES: Set[str] = {
    "explorer.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "opera_gx.exe",
    "code.exe",
    "cursor.exe",
    "pycharm.exe",
    "pycharm64.exe",
    "idea.exe",
    "idea64.exe",
    "devenv.exe",
    "sublime_text.exe",
    "notepad.exe",
    "notepad++.exe",
    "taskmgr.exe",
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "windowsterminal.exe",
    "wt.exe",
    "conhost.exe",
    "discord.exe",
    "telegram.exe",
    "slack.exe",
    "spotify.exe",
}

BLOCKED_WINDOW_TITLES: Sequence[str] = (
    "program manager",
    "workerw",
    "progman",
    "desktop",
    "masaüstü",
    "gatherion",
    "google chrome",
    "microsoft edge",
    "visual studio code",
    "visual studio",
    "cursor",
    "file explorer",
    "dosya gezgini",
    "task manager",
    "görev yöneticisi",
)

# D52/I4: blocked-title patterns are compiled ONCE at import, not rebuilt
# (re.escape + re.compile + cache lookup) on every Gate-1 call. Same
# word-boundary semantics as the old per-call re.search (F-1054).
_BLOCKED_TITLE_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(r"\b" + re.escape(title) + r"\b") for title in BLOCKED_WINDOW_TITLES
)

# D52/I4: own PID cached once; the old per-call os.getpid() is the same
# value for the life of the process.
_OWN_PID: int = os.getpid()

_MASK64 = (1 << 64) - 1


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized population count of a uint64 array (bit counts per element)."""
    return np.unpackbits(
        np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8), axis=1
    ).sum(axis=1)


def _get_process_name_for_pid(pid: int) -> str:
    """Best-effort process executable name lookup for a given PID on Windows."""
    if pid <= 0 or os.name != "nt":
        return ""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        return str(proc.name() or "").casefold()
    except Exception:
        pass

    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if h_proc:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h_proc, 0, buf, ctypes.byref(size)):
                kernel32.CloseHandle(h_proc)
                return os.path.basename(buf.value).casefold()
            kernel32.CloseHandle(h_proc)
    except Exception:
        pass
    return ""


# ==============================================================================
# Core GameFrameGate Class
# ==============================================================================
class GameFrameGate:
    """
    Three-gate intelligent frame validator for game ingestion and dataset cleaning.
    """

    DEFAULT_TEMPLATES = (
        ("map_header", "templates/map_header.png"),
        ("map_header_legacy", "templates/legacy/map_header_video.png"),
        ("world_gather_crest", "templates/world_gather_crest.png"),
        ("world_gather_crest_near", "templates/world_gather_crest_near.png"),
        ("f_gather_pill", "templates/f_gather_pill.png"),
        ("f_gather_pill_video", "templates/legacy/f_gather_pill_video.png"),
        ("f_gather", "templates/f_gather.png"),
        ("e_gather_icon", "templates/e_gather_icon.png"),
        ("sprout_node", "templates/minimap_sprout_node.png"),
        ("sprout_icon", "templates/minimap_gatherable_sprout_icon.png"),
        ("filter_funnel", "templates/minimap_filter_funnel.png"),
    )

    def __init__(
        self,
        max_hash_cache: int = 1024,
        min_dhash_distance: int = 6,
        focus_guard: Optional[ForegroundFocusGuard] = None,
        is_game_running_fn: Optional[Callable[[], bool]] = None,
        templates_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.max_hash_cache: int = max(32, int(max_hash_cache))
        self.min_dhash_distance: int = max(1, int(min_dhash_distance))
        self.focus_guard: ForegroundFocusGuard = focus_guard or ForegroundFocusGuard()
        self.is_game_running_fn: Callable[[], bool] = is_game_running_fn or is_game_running

        # F-1053: gerçek LRU — erişim sırası tazelenir (OrderedDict);
        # eskiden deque (FIFO) idi, isabet sırayı tazelemiyordu.
        self._hash_cache: OrderedDict = OrderedDict()
        self._simulated_game_running: Optional[bool] = None

        self.stats: Dict[str, int] = {
            "total_evaluated": 0,
            "accepted": 0,
            "rejected_gate1_process_or_focus": 0,
            "rejected_gate2_visual_hud": 0,
            "rejected_gate3_duplicate_dhash": 0,
        }

        # Resolve templates
        self._templates_dir: Path = (
            Path(templates_dir) if templates_dir else Path(__file__).resolve().parent.parent / "templates"
        )
        self._templates: Dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Pre-load grayscale UI templates for fast cross-correlation.

        F-0863: yol templates_dir.parent ile kuruluyordu; özel
        templates_dir verilince yanlış dizine bakılırdı. Şablonlar
        _templates_dir altındadır ("templates/" öneki atılır).
        F-0953: yükleme hatası artık sessiz değil; bozuk dosya
        uyarılır, hiç şablon yüklenemediyse dedektörlerin ölü olduğu
        açıkça loglanır.
        """
        failed: List[str] = []
        for key, rel_path in self.DEFAULT_TEMPLATES:
            rel = rel_path[len("templates/"):] if rel_path.startswith("templates/") else rel_path
            full_path = self._templates_dir / rel
            if not full_path.is_file():
                logger.debug("Optional template %s not found: %s", key, full_path)
                failed.append(key)
                continue
            try:
                raw = np.fromfile(str(full_path), dtype=np.uint8)
                if raw.size > 0:
                    tpl = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
                    if tpl is not None and tpl.size > 0:
                        self._templates[key] = tpl
                        continue
                failed.append(key)
            except Exception as exc:
                logger.warning("Template %s failed to load (%s): %s", key, full_path, exc)
                failed.append(key)
        if failed:
            logger.debug("GameFrameGate: %d templates unavailable: %s", len(failed), ", ".join(failed))
        if not self._templates:
            logger.warning(
                "GameFrameGate: no templates loaded; prompt/minimap-template detectors are dead"
            )

    def set_simulated_game_state(
        self,
        is_running: Optional[bool] = None,
        foreground_title: Optional[str] = None,
        foreground_pid: Optional[int] = None,
    ) -> None:
        """Helper for unit tests to emulate focus transitions and game state."""
        self._simulated_game_running = is_running
        if foreground_title is not None or foreground_pid is not None:
            self.focus_guard.set_simulated_foreground_window(foreground_title, foreground_pid)

    def clear_simulated_game_state(self) -> None:
        """Resets simulated game state (F-0954): the simulated running
        flag and the focus guard simulation are cleared, so the
        simulation cannot leak into the global singleton / production."""
        self._simulated_game_running = None
        self.focus_guard.clear_simulated_foreground_window()

    def clear_hash_cache(self) -> None:
        """Empties the perceptual LRU dHash cache."""
        self._hash_cache.clear()

    @property
    def hash_cache_size(self) -> int:
        return len(self._hash_cache)

    # --------------------------------------------------------------------------
    # dHash & Perceptual Distance Algorithms
    # --------------------------------------------------------------------------
    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                return frame[:, :, 0]
            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            if frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # F-0864: 2/5 kanallı kare cv2 hatasıyla çökertirdi; açık
            # ValueError verilir, çağıran False döner.
            raise ValueError(
                f"Unsupported channel count: {frame.shape[2]} (expected 1, 3 or 4)"
            )
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    @staticmethod
    def compute_dhash(frame: np.ndarray) -> int:
        """
        Computes 64-bit gradient difference hash (dHash).
        Resizes grayscale frame to 9x8 and computes adjacent column differences.
        """
        gray = GameFrameGate._to_grayscale(frame)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        # A14: bit-packing via np.packbits is bit-identical to the
        # 64-step Python loop (verified on 300 random frames, 0
        # mismatches) and ~20x faster on the packing step.
        return int.from_bytes(np.packbits(diff.flatten()).tobytes(), "big")

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Bitwise Hamming distance between two 64-bit perceptual hashes."""
        xor_val = (int(h1) ^ int(h2)) & ((1 << 64) - 1)
        return int(xor_val.bit_count())

    # --------------------------------------------------------------------------
    # Kapı 1: Process & Focus Guard
    # --------------------------------------------------------------------------
    def check_gate1_process_and_focus(self) -> Tuple[bool, str]:
        """
        Kapı 1: Checks process running status and active foreground window.
        Rejects Windows desktop, explorer, chrome, code, Gatherion UI, or non-focused game.
        """
        running = (
            self._simulated_game_running
            if self._simulated_game_running is not None
            else bool(self.is_game_running_fn())
        )
        if not running:
            return False, "game_not_running: AION 2 game process is not running"

        hwnd, pid, title = self.focus_guard.query_foreground_window_info()
        title_lower = (title or "").strip().lower()

        # Reject empty / lockscreen / desktop window
        if hwnd == 0 and not title_lower:
            return False, "foreground_empty: No active foreground window (Desktop or lock screen)"

        for desk_kw in ("program manager", "workerw", "progman", "desktop", "masaüstü"):
            if title_lower == desk_kw or title_lower.startswith(desk_kw):
                return False, f"desktop_focused: Windows desktop is active foreground ('{title or desk_kw}')"

        # Check process executable name
        proc_name = _get_process_name_for_pid(pid)
        if proc_name in BLOCKED_FOREGROUND_EXECUTABLES:
            return (
                False,
                f"blocked_foreground_process: Active window belongs to {proc_name} ('{title}', PID={pid})",
            )

        # Check Gatherion UI itself
        if "gatherion" in title_lower or (pid > 0 and pid == _OWN_PID):
            return False, f"gatherion_ui_focused: Foreground window is Gatherion UI ('{title}', PID={pid})"

        # Check blocked window title markers (word-boundary match, F-1054:
        # alt-dize eşleşmesi "decode" içindeki "code" gibi yanlış
        # pozitif üretirdi). Patterns are precompiled at module level.
        for pattern in _BLOCKED_TITLE_PATTERNS:
            if pattern.search(title_lower):
                return (
                    False,
                    f"blocked_foreground_window: Active window title matches blocked non-game app: '{title}'",
                )

        focused, info = self.focus_guard.is_game_window_focused()
        if not focused:
            return False, f"window_not_focused: {info}"

        return True, f"process_and_focus_ok: '{title}' (PID={pid})"

    # --------------------------------------------------------------------------
    # Kapı 2: Visual HUD Evidence
    # --------------------------------------------------------------------------
    def _detect_minimap_compass(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Detects circular minimap radar border or player compass arrow in top-right."""
        h, w = frame_bgr.shape[:2]
        # Standard top-right minimap ROI
        x1, x2 = int(w * 0.72), int(w * 0.998)
        y1, y2 = int(h * 0.01), int(h * 0.35)
        if x2 <= x1 or y2 <= y1:
            return False, ""
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return False, ""

        rh, rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. Player directional arrow in center region of minimap
        sub = hsv[int(rh * 0.30) : int(rh * 0.72), int(rw * 0.30) : int(rw * 0.72)]
        if sub.size > 0:
            arrow_mask = cv2.inRange(sub, (0, 0, 185), (180, 60, 255))
            num, _, stats, _ = cv2.connectedComponentsWithStats(arrow_mask)
            for idx in range(1, num):
                _, _, bw, bh, area = stats[idx]
                # F-0955: eşikler çok gevşekti (8 px leke bile ok
                # sayılırdı). Ok kompakt, üçgene yakın bir parlak
                # bileşendir: alan/boyut alt sınırı ve doluluk oranı
                # istenir.
                if 20 <= area <= 600 and 5 <= bw <= 45 and 5 <= bh <= 45:
                    fill = float(area) / max(1, int(bw) * int(bh))
                    if 0.20 <= fill <= 0.95:
                        return True, "minimap_compass_arrow"

        # 2. Minimap circular boundary via circular edge distribution
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        center_x, center_y = rw * 0.52, rh * 0.54
        expected_r = min(rw, rh) * 0.38
        edge_pts = np.argwhere(edges > 0)
        if len(edge_pts) >= 40:
            dists = np.hypot(edge_pts[:, 1] - center_x, edge_pts[:, 0] - center_y)
            circle_pts = np.count_nonzero(np.abs(dists - expected_r) <= (expected_r * 0.16))
            if circle_pts >= 28:
                return True, "minimap_circular_ring"

        # 3. Minimap templates (sprout icon, header pin, funnel)
        for tname in ("map_header", "map_header_legacy", "sprout_node", "sprout_icon", "filter_funnel"):
            tpl = self._templates.get(tname)
            if tpl is not None and tpl.shape[0] <= gray_roi.shape[0] and tpl.shape[1] <= gray_roi.shape[1]:
                res = cv2.matchTemplate(gray_roi, tpl, cv2.TM_CCOEFF_NORMED)
                if cv2.minMaxLoc(res)[1] >= 0.52:
                    return True, f"minimap_template_{tname}"

        return False, ""

    def _detect_player_hp_mp_frame(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Detects player vitality bar (red HP / blue MP) or overhead HP bar."""
        h, w = frame_bgr.shape[:2]

        # 1. Bottom-center HUD vitality bar
        y1, y2 = int(h * 0.80), int(h * 0.98)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        hp_roi = frame_bgr[y1:y2, x1:x2]
        if hp_roi.size > 0:
            hsv = cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV)
            max_bar_h = min(42, max(4, int(h * 0.045)))
            max_area = int(hp_roi.shape[0] * hp_roi.shape[1] * 0.25)
            # Red HP mask
            red_mask = cv2.inRange(hsv, (0, 75, 60), (15, 255, 255)) | cv2.inRange(
                hsv, (165, 75, 60), (180, 255, 255)
            )
            num_r, _, stats_r, _ = cv2.connectedComponentsWithStats(red_mask)
            for idx in range(1, num_r):
                _, _, bw, bh, area = stats_r[idx]
                if bw >= 20 and 3 <= bh <= max_bar_h and (bw / max(1, bh)) >= 2.5 and area < max_area:
                    return True, "player_red_hp_bar"

            # Blue MP mask
            blue_mask = cv2.inRange(hsv, (85, 75, 60), (125, 255, 255))
            num_b, _, stats_b, _ = cv2.connectedComponentsWithStats(blue_mask)
            for idx in range(1, num_b):
                _, _, bw, bh, area = stats_b[idx]
                if bw >= 20 and 3 <= bh <= max_bar_h and (bw / max(1, bh)) >= 2.5 and area < max_area:
                    return True, "player_blue_mp_bar"

        # 2. Character overhead HP bar (center screen)
        char_roi = frame_bgr[int(h * 0.35) : int(h * 0.65), int(w * 0.40) : int(w * 0.60)]
        if char_roi.size > 0:
            hsv_c = cv2.cvtColor(char_roi, cv2.COLOR_BGR2HSV)
            red_c = cv2.inRange(hsv_c, (0, 75, 60), (15, 255, 255)) | cv2.inRange(
                hsv_c, (165, 75, 60), (180, 255, 255)
            )
            num_c, _, stats_c, _ = cv2.connectedComponentsWithStats(red_c)
            for idx in range(1, num_c):
                _, _, bw, bh, _ = stats_c[idx]
                if bw >= 15 and 2 <= bh <= min(25, int(h * 0.03)) and (bw / max(1, bh)) >= 2.5:
                    return True, "overhead_character_hp_bar"

        # 3. Target HP bar (top-center)
        tgt_roi = frame_bgr[int(h * 0.03) : int(h * 0.20), int(w * 0.30) : int(w * 0.70)]
        if tgt_roi.size > 0:
            hsv_t = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2HSV)
            red_t = cv2.inRange(hsv_t, (0, 75, 60), (15, 255, 255)) | cv2.inRange(
                hsv_t, (165, 75, 60), (180, 255, 255)
            )
            num_t, _, stats_t, _ = cv2.connectedComponentsWithStats(red_t)
            for idx in range(1, num_t):
                _, _, bw, bh, _ = stats_t[idx]
                if bw >= 20 and 3 <= bh <= min(35, int(h * 0.04)) and (bw / max(1, bh)) >= 2.5:
                    return True, "target_entity_hp_bar"

        return False, ""

    def _detect_skill_bar(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Detects action skill bar with regular slot boxes at screen bottom."""
        h, w = frame_bgr.shape[:2]
        skill_roi = frame_bgr[int(h * 0.82) : int(h * 0.99), int(w * 0.20) : int(w * 0.80)]
        if skill_roi.size == 0:
            return False, ""
        gray = cv2.cvtColor(skill_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 110)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slot_boxes = 0
        for cnt in contours:
            _, _, bw, bh = cv2.boundingRect(cnt)
            if 15 <= bw <= 85 and 15 <= bh <= 85 and 0.65 <= (bw / max(1, bh)) <= 1.5:
                slot_boxes += 1
        if slot_boxes >= 2:
            return True, f"skill_bar_slots_{slot_boxes}"
        return False, ""

    def _detect_game_prompts_or_crests(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Template cross-correlation for 3D gather crests or F Gather prompts."""
        gray = self._to_grayscale(frame_bgr)
        for key in (
            "world_gather_crest",
            "world_gather_crest_near",
            "f_gather_pill",
            "f_gather_pill_video",
            "f_gather",
            "e_gather_icon",
        ):
            tpl = self._templates.get(key)
            if tpl is not None and tpl.shape[0] <= gray.shape[0] and tpl.shape[1] <= gray.shape[1]:
                res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                score = float(cv2.minMaxLoc(res)[1])
                if score >= 0.44:
                    return True, f"prompt_match_{key}_{score:.2f}"
        return False, ""

    def _detect_desktop_or_browser_canvas(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Detects unmistakable non-game desktop taskbar or browser tabs."""
        h, w = frame_bgr.shape[:2]
        gray = self._to_grayscale(frame_bgr)

        # Check Windows Taskbar: solid bar spanning full width at bottom (y >= 0.955)
        tb_roi = gray[int(h * 0.955) :, :]
        if tb_roi.size > 0:
            tb_std = float(np.std(tb_roi))
            tb_mean = float(np.mean(tb_roi))
            # Standard Windows taskbar: dark solid strip across entire bottom.
            # A14: the strip is int(h*0.045) rows, so a fixed ">= 24 rows"
            # floor disabled this veto below ~534 px height (16 rows at
            # 360p) and a noise frame with a taskbar passed Gate 2.
            # 8 rows is enough to tell a solid strip from texture.
            if tb_std < 10.0 and tb_mean < 55.0 and tb_roi.shape[0] >= 8:
                return True, f"windows_taskbar_detected (std={tb_std:.1f}, mean={tb_mean:.1f})"

        # Check Browser URL bar & tabs: solid light strip at top
        top_roi = gray[: int(h * 0.06), :]
        if top_roi.size > 0:
            top_std = float(np.std(top_roi))
            top_mean = float(np.mean(top_roi))
            # A14: same resolution floor as the taskbar check (21 rows
            # at 360p < 24 disabled the browser veto at low resolution).
            if top_std < 8.0 and top_mean > 210.0 and top_roi.shape[0] >= 8:
                return True, f"browser_tab_bar_detected (std={top_std:.1f}, mean={top_mean:.1f})"

        return False, ""

    def check_gate2_visual_hud(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Kapı 2: Verifies frame contains genuine AION game UI features.
        Rejects solid colors, black frames, desktop icons, or browser canvases.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, "invalid_frame: Frame is None or empty"
        if frame.ndim not in (2, 3):
            return False, f"invalid_shape: Frame dimension {frame.ndim} is not supported"
        # F-0864: kanal sayısı denetlenir; 2/5 kanallı kare cv2
        # hatasıyla çökertirdi.
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            return False, f"invalid_channels: Frame has {frame.shape[2]} channels (expected 1, 3 or 4)"

        h, w = frame.shape[:2]
        if h < 32 or w < 32:
            return False, f"invalid_resolution: Frame resolution ({h}x{w}) is too small"

        # Coerce to BGR for visual checks
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 1:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        gray = self._to_grayscale(frame_bgr)
        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))

        # 1. Flat & blank canvas rejections
        if mean_val < 8.0:
            return False, f"black_frame: Frame intensity too dark ({mean_val:.2f} < 8.0)"
        if std_val < 3.0:
            return False, f"solid_color: Frame has near-zero standard deviation ({std_val:.2f} < 3.0)"
        if mean_val > 250.0 and std_val < 8.0:
            return False, f"blank_white_canvas: Frame is pure white canvas ({mean_val:.2f} > 250.0)"

        # 2. Masaüstü/tarayıcı kanıtı HUD kanıtından ÖNCE veto eder
        # (F-0956): eskiden yalnızca HUD kanıtı yoksa çalışırdı; masaüstü
        # karesi HUD benzeri bir lekeyle geçebilirdi.
        # F-0865: küçük crop'lar artık HUD kanıtı olmadan kabul
        # edilmez; önceki "accepted_crop_texture" erken kabulü kaldırıldı.
        is_desk, desk_reason = self._detect_desktop_or_browser_canvas(frame_bgr)
        if is_desk:
            return False, f"desktop_or_browser_canvas: {desk_reason}"

        # 3. Gather evidence from genuine game HUD features
        evidence: List[str] = []

        ok_mm, ev_mm = self._detect_minimap_compass(frame_bgr)
        if ok_mm:
            evidence.append(ev_mm)

        ok_hp, ev_hp = self._detect_player_hp_mp_frame(frame_bgr)
        if ok_hp:
            evidence.append(ev_hp)

        ok_sb, ev_sb = self._detect_skill_bar(frame_bgr)
        if ok_sb:
            evidence.append(ev_sb)

        ok_pr, ev_pr = self._detect_game_prompts_or_crests(frame_bgr)
        if ok_pr:
            evidence.append(ev_pr)

        if evidence:
            return True, f"accepted_hud_evidence: {', '.join(evidence)}"

        return (
            False,
            "no_hud_evidence: Frame lacks genuine Aion game UI features (minimap, compass arrow, HP frame, skill bar)",
        )

    # --------------------------------------------------------------------------
    # Kapı 3: Visual Novelty / dHash Gate
    # --------------------------------------------------------------------------
    def check_gate3_visual_novelty(
        self,
        frame: np.ndarray,
        update_cache: bool = True,
    ) -> Tuple[bool, str]:
        """
        Kapı 3: Computes 64-bit dHash and checks LRU cache.
        Rejects frames with perceptual Hamming distance dHash < 6.
        Invalid frames return False instead of crashing (F-0866).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, "invalid_frame: Frame is None or empty"
        try:
            dhash = self.compute_dhash(frame)
        except Exception as exc:
            return False, f"invalid_frame: dHash could not be computed ({exc})"

        if self._hash_cache:
            # D52/I3: vectorized Hamming scan over the whole LRU cache
            # (one XOR + popcount) instead of a Python-level
            # hamming_distance() call per cached hash. np.argmin returns
            # the FIRST minimum, matching the old strictly-less scan, so
            # the LRU touch below moves the same entry to the end.
            # Keys are dHash values (< 2**64); both sides are masked to
            # 64 bits exactly like hamming_distance() does.
            keys = list(self._hash_cache)
            cached = np.fromiter(
                (int(h) & _MASK64 for h in keys),
                dtype=np.uint64,
                count=len(keys),
            )
            dists = _popcount_u64(cached ^ np.uint64(dhash & _MASK64))
            idx = int(np.argmin(dists))
            min_dist = int(dists[idx])
            closest = keys[idx]
            if min_dist < self.min_dhash_distance:
                # F-1053: gerçek LRU — isabet eden hash en sona alınır.
                if closest is not None:
                    self._hash_cache.move_to_end(closest)
                return (
                    False,
                    f"duplicate_frame: Perceptual dHash distance {min_dist} < {self.min_dhash_distance} (duplicate/spam)",
                )
        else:
            min_dist = 64

        if update_cache:
            self._hash_cache[dhash] = None
            while len(self._hash_cache) > self.max_hash_cache:
                self._hash_cache.popitem(last=False)

        return True, f"visual_novelty_ok: dhash={dhash:#018x}, min_hamming={min_dist}"

    # --------------------------------------------------------------------------
    # Master Gate Method: is_valid_game_frame
    # --------------------------------------------------------------------------
    def is_valid_game_frame(
        self,
        frame: np.ndarray,
        require_foreground: bool = True,
        require_novelty: bool = True,
    ) -> Tuple[bool, str]:
        """
        Master Game Frame Gate:
        - Kapı 1: Process & focus guard (if require_foreground=True).
        - Kapı 2: Visual HUD evidence verification.
        - Kapı 3: Perceptual dHash novelty check (if require_novelty=True).

        Returns:
            `(is_valid, reason)`
        """
        self.stats["total_evaluated"] += 1

        # Kapı 1
        if require_foreground:
            ok1, reason1 = self.check_gate1_process_and_focus()
            if not ok1:
                self.stats["rejected_gate1_process_or_focus"] += 1
                return False, reason1

        # Kapı 2
        ok2, reason2 = self.check_gate2_visual_hud(frame)
        if not ok2:
            self.stats["rejected_gate2_visual_hud"] += 1
            return False, reason2

        # Kapı 3
        if require_novelty:
            ok3, reason3 = self.check_gate3_visual_novelty(frame, update_cache=True)
            if not ok3:
                self.stats["rejected_gate3_duplicate_dhash"] += 1
                return False, reason3

        self.stats["accepted"] += 1
        return True, f"valid_game_frame: {reason2}"


# ==============================================================================
# Global Default Gate & Module-Level Convenience Functions
# ==============================================================================
_GLOBAL_FRAME_GATE = GameFrameGate()


def get_default_frame_gate() -> GameFrameGate:
    """Returns the singleton GameFrameGate instance."""
    return _GLOBAL_FRAME_GATE


def is_valid_game_frame(
    frame: np.ndarray,
    require_foreground: bool = True,
    require_novelty: bool = True,
    gate: Optional[GameFrameGate] = None,
) -> Tuple[bool, str]:
    """
    Convenience function routing directly to the default `GameFrameGate.is_valid_game_frame()`.

    F-0957: varsayılan tekil kapının dHash önbelleği tüm çağıranlar
    arasında paylaşılır; yalıtım isteyen çağıran kendi kapı örneğini
    `gate` ile verir (örn. OmniGameLearner kendi örneğini kullanır).
    """
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    return use_gate.is_valid_game_frame(
        frame=frame,
        require_foreground=require_foreground,
        require_novelty=require_novelty,
    )


def check_visual_hud(frame: np.ndarray, gate: Optional[GameFrameGate] = None) -> Tuple[bool, str]:
    """Inspects only Kapı 2 (Visual HUD Evidence)."""
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    return use_gate.check_gate2_visual_hud(frame)


def check_process_and_focus(gate: Optional[GameFrameGate] = None) -> Tuple[bool, str]:
    """Inspects only Kapı 1 (Process & Focus Guard)."""
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    return use_gate.check_gate1_process_and_focus()


def check_visual_novelty(
    frame: np.ndarray,
    update_cache: bool = True,
    gate: Optional[GameFrameGate] = None,
) -> Tuple[bool, str]:
    """Inspects only Kapı 3 (Perceptual Novelty / dHash)."""
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    return use_gate.check_gate3_visual_novelty(frame, update_cache=update_cache)


def clear_frame_gate_cache(gate: Optional[GameFrameGate] = None) -> None:
    """Clears the default GameFrameGate perceptual LRU cache."""
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    use_gate.clear_hash_cache()


def reset_simulated_game_state(gate: Optional[GameFrameGate] = None) -> None:
    """Clears simulated game state on the default (or given) gate (F-0954)."""
    use_gate = gate if gate is not None else _GLOBAL_FRAME_GATE
    use_gate.clear_simulated_game_state()
