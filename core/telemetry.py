import csv
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# F-1755/F-1744: bounded histories (no unbounded list growth).
_MAX_GATHER_EVENTS = 5000
_MAX_FAILURE_EVENTS = 2000
_MAX_COMBAT_EVENTS = 2000
_MAX_ALERT_HISTORY = 500
_MAX_HEATMAP_CLUSTERS = 500
_MAX_HISTORY_ENTRIES = 200
# F-1745: minimum template-match confidence for a bag-counter glyph.
_MIN_GLYPH_CONFIDENCE = 0.75
# F-1763: repeat suppression window for identical alerts.
_ALERT_DEDUP_SEC = 30.0

logger = logging.getLogger("TelemetryEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Telemetry] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class YieldHeatmapTracker:
    """
    Item 42 — Oturum Verim ve Tehlike Isı Haritası (YieldHeatmapTracker).

    Tracks spatial yield events (`record_node_yield_event(node_id, x, y, success=True)`)
    and monster ambush events (`record_combat_ambush_event(x, y)`), aggregating nearby
    coordinates into spatial clusters and computing normalized intensity (`0.0..1.0`)
    for high-yield emerald zones (`#10B981`, `zone_type="YIELD"`) vs monster-ambush
    red zones (`#EF4444`, `zone_type="AMBUSH"`) so `MapViewerWindow` and Mobile Radar
    can visualize the live heatmap.
    """

    EMERALD_HEX: str = "#10B981"
    AMBUSH_RED_HEX: str = "#EF4444"

    def __init__(self, cluster_radius: float = 0.025) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.cluster_radius: float = max(0.001, float(cluster_radius))
        self._yield_clusters: List[Dict[str, Any]] = []
        self._ambush_clusters: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clears all recorded yield and ambush heatmap events."""
        with self._lock:
            self._yield_clusters.clear()
            self._ambush_clusters.clear()

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    @staticmethod
    def _coord_space(x: float, y: float) -> str:
        # F-1761: normalized [0..1] and pixel coordinates must never share a
        # cluster — distances across the two spaces are meaningless.
        return "pixel" if (abs(x) > 2.0 or abs(y) > 2.0) else "normalized"

    def _effective_radius(self, x: float, y: float) -> float:
        # If pixel coordinates (> 2.0) are passed instead of normalized [0..1], scale cluster radius
        if abs(x) > 2.0 or abs(y) > 2.0:
            return max(24.0, self.cluster_radius * 1000.0)
        return self.cluster_radius

    def _match_or_create_cluster(
        self,
        clusters: List[Dict[str, Any]],
        fx: float,
        fy: float,
        radius: float,
        space: str,
        new_cluster: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Finds the first cluster within `radius` (same coord space) or
        appends `new_cluster` (dropping the oldest past the cap) and returns it.

        Shared by the yield and ambush recorders — both used to duplicate
        this scan + cap + creation block (~20 lines each).
        """
        for cluster in clusters:
            if cluster.get("coord_space", "normalized") != space:
                continue
            if self._distance(fx, fy, cluster["x"], cluster["y"]) <= radius:
                return cluster
        clusters.append(new_cluster)
        if len(clusters) > _MAX_HEATMAP_CLUSTERS:
            clusters.pop(0)
        return new_cluster

    @staticmethod
    def _move_centroid(cluster: Dict[str, Any], fx: float, fy: float) -> None:
        """Running-mean centroid update (n = events so far, before this one)."""
        n = cluster["total_events"]
        cluster["x"] = round((cluster["x"] * n + fx) / (n + 1), 6)
        cluster["y"] = round((cluster["y"] * n + fy) / (n + 1), 6)

    def record_node_yield_event(
        self,
        node_id: int,
        x: float,
        y: float,
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        Records a resource node harvest attempt at `(x, y)` and updates the spatial
        emerald yield cluster intensity.
        """
        fx = float(x)
        fy = float(y)
        nid = int(node_id)
        now = time.time()
        radius = self._effective_radius(fx, fy)
        space = self._coord_space(fx, fy)

        with self._lock:
            matched = self._match_or_create_cluster(
                self._yield_clusters,
                fx,
                fy,
                radius,
                space,
                {
                    "node_id": nid,
                    "x": fx,
                    "y": fy,
                    "coord_space": space,
                    "zone_type": "YIELD",
                    "color_hex": self.EMERALD_HEX,
                    "success_count": 0,
                    "fail_count": 0,
                    "total_events": 0,
                    "raw_score": 0.0,
                    "last_ts": now,
                },
            )

            self._move_centroid(matched, fx, fy)
            # F-1761: keep the first node_id of a cluster; the latest event
            # must not overwrite the cluster's identity.
            matched["total_events"] += 1
            matched["last_ts"] = now
            if success:
                matched["success_count"] += 1
                matched["raw_score"] += 1.0
            else:
                matched["fail_count"] += 1
                # F-1742: a failed gather must LOWER the yield score, never raise it.
                matched["raw_score"] = max(0.0, matched["raw_score"] - 0.25)

            return dict(matched)

    def record_combat_ambush_event(self, x: float, y: float) -> Dict[str, Any]:
        """
        Records a monster ambush / combat interruption at `(x, y)` and updates the
        spatial danger red-zone cluster intensity.
        """
        fx = float(x)
        fy = float(y)
        now = time.time()
        radius = self._effective_radius(fx, fy)
        space = self._coord_space(fx, fy)

        with self._lock:
            matched = self._match_or_create_cluster(
                self._ambush_clusters,
                fx,
                fy,
                radius,
                space,
                {
                    "node_id": -1,
                    "x": fx,
                    "y": fy,
                    "coord_space": space,
                    "zone_type": "AMBUSH",
                    "color_hex": self.AMBUSH_RED_HEX,
                    "ambush_count": 0,
                    "total_events": 0,
                    "raw_score": 0.0,
                    "last_ts": now,
                },
            )

            self._move_centroid(matched, fx, fy)
            matched["ambush_count"] += 1
            matched["total_events"] += 1
            matched["raw_score"] += 1.0
            matched["last_ts"] = now
            return dict(matched)

    def get_heatmap_points(self) -> List[Dict[str, Any]]:
        """
        Returns all recorded yield (emerald) and ambush (red) heatmap points with
        normalized intensity in `[0.0 .. 1.0]`.
        """
        with self._lock:
            points: List[Dict[str, Any]] = []

            if self._yield_clusters:
                # F-1761: normalize against at least 3.0 so one lone event does
                # not render at full intensity.
                max_yield_score = max(max(c["raw_score"] for c in self._yield_clusters), 3.0)
                for c in self._yield_clusters:
                    norm_intensity = float(np.clip(c["raw_score"] / max_yield_score, 0.10, 1.0))
                    points.append(
                        {
                            "node_id": c["node_id"],
                            "x": c["x"],
                            "y": c["y"],
                            "coord_space": c.get("coord_space", "normalized"),
                            "zone_type": "YIELD",
                            "category": "emerald_yield",
                            "color_hex": self.EMERALD_HEX,
                            "intensity": round(norm_intensity, 4),
                            "success_count": c["success_count"],
                            "fail_count": c["fail_count"],
                            "count": c["total_events"],
                            "last_ts": c["last_ts"],
                        }
                    )

            if self._ambush_clusters:
                max_ambush_score = max(max(c["raw_score"] for c in self._ambush_clusters), 3.0)
                for c in self._ambush_clusters:
                    norm_intensity = float(np.clip(c["raw_score"] / max_ambush_score, 0.15, 1.0))
                    points.append(
                        {
                            "node_id": c["node_id"],
                            "x": c["x"],
                            "y": c["y"],
                            "coord_space": c.get("coord_space", "normalized"),
                            "zone_type": "AMBUSH",
                            "category": "monster_ambush",
                            "color_hex": self.AMBUSH_RED_HEX,
                            "intensity": round(norm_intensity, 4),
                            "ambush_count": c["ambush_count"],
                            "count": c["total_events"],
                            "last_ts": c["last_ts"],
                        }
                    )

            return points


class InventoryFullnessEstimator:
    """
    Item 43 — Otomatik Envanter Doluluk Sayacı (InventoryFullnessEstimator).

    Tracks every gathered material and 3x stage (`record_gathered_material(...)`) AND
    reads the top-right bag slot counter badge (`61/75` at `x: 0.895..0.925, y: 0.030..0.055`
    seen in `pilgrims_ridge_1.jpg`). Estimates remaining free inventory slots and fires a
    `95% Doluluk Uyarısı` (`inventory_near_full: bool`, `used_slots: int`, `max_slots: int`,
    `fill_pct: float`) before the bag overflows.
    """

    BAG_BADGE_ROI: Tuple[float, float, float, float] = (0.895, 0.030, 0.925, 0.055)
    NEAR_FULL_THRESHOLD_PCT: float = 95.0

    # Canonical 5x8 bitmap templates extracted from the game HUD font (`61/75` at top-right)
    _HUD_DIGIT_TEMPLATES: Dict[str, List[str]] = {
        "0": [
            ".###.",
            "##.##",
            "##.##",
            "##.##",
            "##.##",
            "##.##",
            "##.##",
            ".###.",
        ],
        "1": [
            "..##.",
            ".###.",
            "...#.",
            "...#.",
            "...#.",
            "...#.",
            "...#.",
            "...#.",
        ],
        "2": [
            ".###.",
            "#..##",
            "....#",
            "...#.",
            "..#..",
            ".#...",
            "#....",
            "#####",
        ],
        "3": [
            "####.",
            "....#",
            "...#.",
            ".###.",
            "....#",
            "....#",
            "#..##",
            ".###.",
        ],
        "4": [
            "...#.",
            "..##.",
            ".#.#.",
            "#..#.",
            "#####",
            "...#.",
            "...#.",
            "...#.",
        ],
        "5": [
            "####.",
            "##...",
            "##...",
            ".###.",
            "...##",
            "....#",
            "#..##",
            "####.",
        ],
        "6": [
            "...#.",
            "..#..",
            ".##..",
            ".####",
            "##..#",
            "##..#",
            ".#..#",
            "..###",
        ],
        "7": [
            "#####",
            "...#.",
            "..##.",
            "..#..",
            "..#..",
            ".##..",
            ".#...",
            "##...",
        ],
        "8": [
            ".###.",
            "##.##",
            "##.##",
            ".###.",
            "##.##",
            "##.##",
            "##.##",
            ".###.",
        ],
        "9": [
            ".###.",
            "##.##",
            "##.##",
            ".####",
            "...##",
            "..##.",
            ".##..",
            "##...",
        ],
    }

    # Class-level cache: the 5x8 digit templates are identical for every
    # instance, so compiling 10 numpy arrays per __init__ was pure waste.
    # Filled lazily once (guarded by _templates_lock) and shared read-only.
    _compiled_templates_cache: Optional[Dict[str, np.ndarray]] = None
    _compiled_template_masks_cache: Optional[Dict[str, np.ndarray]] = None
    _templates_lock = threading.Lock()

    def __init__(
        self,
        default_used_slots: int = 0,
        default_max_slots: int = 75,
        near_full_threshold_pct: float = 95.0,
        stack_size: int = 20,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.used_slots: int = max(0, int(default_used_slots))
        self.max_slots: int = max(1, int(default_max_slots))
        self.near_full_threshold_pct: float = float(np.clip(near_full_threshold_pct, 50.0, 100.0))
        # F-1752: stack size is a configurable parameter, not a hidden constant.
        self.stack_size: int = max(1, int(stack_size))
        self.total_materials_gathered: int = 0
        self.total_stages_completed: int = 0
        self._unique_resources_seen: set = set()
        self._resource_stack_counts: Dict[str, int] = defaultdict(int)
        self._hud_baseline_slots: Optional[int] = None
        self._gathers_since_hud_sync: int = 0
        self._near_full_warning_fired: bool = False
        with InventoryFullnessEstimator._templates_lock:
            if InventoryFullnessEstimator._compiled_templates_cache is None:
                compiled = self._compile_digit_templates()
                InventoryFullnessEstimator._compiled_templates_cache = compiled
                # Precomputed boolean masks: _classify_glyph used to evaluate
                # (tmpl > 127) for every template on every glyph call.
                InventoryFullnessEstimator._compiled_template_masks_cache = {
                    digit: (tmpl > 127) for digit, tmpl in compiled.items()
                }
        self._compiled_templates: Dict[str, np.ndarray] = (
            InventoryFullnessEstimator._compiled_templates_cache
        )
        self._compiled_template_masks: Dict[str, np.ndarray] = (
            InventoryFullnessEstimator._compiled_template_masks_cache
        )

    @classmethod
    def _compile_digit_templates(cls) -> Dict[str, np.ndarray]:
        compiled: Dict[str, np.ndarray] = {}
        for digit, rows in cls._HUD_DIGIT_TEMPLATES.items():
            arr = np.array(
                [[255 if ch == "#" else 0 for ch in row] for row in rows],
                dtype=np.uint8,
            )
            compiled[digit] = arr
        return compiled

    def reset(self, used_slots: int = 0, max_slots: int = 75) -> None:
        """Resets the inventory fullness estimator for a new session."""
        with self._lock:
            self.used_slots = max(0, int(used_slots))
            self.max_slots = max(1, int(max_slots))
            self.total_materials_gathered = 0
            self.total_stages_completed = 0
            self._unique_resources_seen.clear()
            self._resource_stack_counts.clear()
            self._hud_baseline_slots = None
            self._gathers_since_hud_sync = 0
            self._near_full_warning_fired = False

    def _classify_glyph(self, glyph_bw: np.ndarray) -> Tuple[str, float]:
        """Matches a segmented binary digit glyph against the 5x8 HUD digit templates.

        The glyph's boolean mask is computed once here (it used to be
        re-evaluated inside the per-template loop) and compared against the
        precomputed template masks — identical scores, ~10 fewer array
        conversions per glyph.

        v19-smoke: dar glifler (`'1'`, 3 px geniş) 5 px'e GERİLEREK
        yeniden boyutlandırılmamalı — INTER_NEAREST gerilmesi üst
        bayrağı tüm genişliğe yayar ve `'1'` 0.625 güvenle
        `_MIN_GLYPH_CONFIDENCE` (0.75) altında kalıp tüm okumayı
        reddettirir (smoke M42-43: `61/75` -> None). 5 px'ten dar
        glif ortaya ortalanarak dolgulanır (şekil korunur, `'1'` -> 1.0);
        5+ px glifler eski yoldan aynen devam eder.
        """
        if glyph_bw.shape[1] < 5:
            padded = np.zeros((glyph_bw.shape[0], 5), dtype=glyph_bw.dtype)
            x0 = (5 - int(glyph_bw.shape[1])) // 2
            padded[:, x0 : x0 + int(glyph_bw.shape[1])] = glyph_bw
            glyph_bw = padded
        resized = cv2.resize(glyph_bw, (5, 8), interpolation=cv2.INTER_NEAREST)
        glyph_bool = resized > 127
        best_digit = "0"
        best_score = -1.0
        for digit, mask in self._compiled_template_masks.items():
            matches = float(np.mean(glyph_bool == mask))
            if matches > best_score:
                best_score = matches
                best_digit = digit
        return best_digit, best_score

    def read_bag_counter_from_frame(
        self,
        frame: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int]]:
        """
        Inspects the top-right bag slot counter badge (`x: 0.895..0.925, y: 0.030..0.055`,
        e.g. `61/75` in `pilgrims_ridge_1.jpg`) and parses `(used_slots, max_slots)`.
        Updates internal inventory fullness state when a valid `used/max` reading is found.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        try:
            fh, fw = frame.shape[:2]
            x1 = int(round(self.BAG_BADGE_ROI[0] * fw))
            y1 = int(round(self.BAG_BADGE_ROI[1] * fh))
            x2 = int(round(self.BAG_BADGE_ROI[2] * fw))
            y2 = int(round(self.BAG_BADGE_ROI[3] * fh))

            x1 = max(0, min(x1, fw - 2))
            y1 = max(0, min(y1, fh - 2))
            x2 = max(x1 + 1, min(x2, fw))
            y2 = max(y1 + 1, min(y2, fh))

            badge = frame[y1:y2, x1:x2]
            if badge.size == 0:
                return None

            # F-1745: accept BGRA frames too (4 channels previously crashed
            # cvtColor); grayscale input passes through unchanged.
            if badge.ndim == 3 and badge.shape[2] == 4:
                gray = cv2.cvtColor(badge, cv2.COLOR_BGRA2GRAY)
            elif badge.ndim == 3:
                gray = cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY)
            else:
                gray = badge
            bh, bw_w = gray.shape[:2]

            # The numeric text "61/75" sits in the lower-middle band of the badge (y: 0.38..0.78)
            band_y1 = max(0, int(round(0.38 * bh)))
            band_y2 = max(band_y1 + 4, int(round(0.78 * bh)))
            digit_band = gray[band_y1:band_y2, :]

            _, binary = cv2.threshold(digit_band, 155, 255, cv2.THRESH_BINARY)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            glyphs: List[Tuple[int, int, int, int, int]] = []
            for label_idx in range(1, num_labels):
                gx, gy, gw, gh, area = (int(v) for v in stats[label_idx])
                # Filter out tiny icon sparkles on the left (`x < 0.25 * bw_w`) and keep character-height components
                if gx < int(0.25 * bw_w):
                    continue
                if gh >= 5 and 2 <= gw <= 14 and area >= 8:
                    glyphs.append((gx, gy, gw, gh, area))

            glyphs.sort(key=lambda item: item[0])
            if len(glyphs) < 3:
                return None

            # Identify the slash '/' separator: typically the tallest or middle diagonal component
            slash_idx = -1
            max_h = max(g[3] for g in glyphs)
            for idx, (gx, gy, gw, gh, area) in enumerate(glyphs):
                if 1 <= idx <= len(glyphs) - 2 and gh == max_h and gh >= 9:
                    slash_idx = idx
                    break

            if slash_idx == -1 and len(glyphs) >= 5:
                slash_idx = len(glyphs) // 2

            if slash_idx <= 0 or slash_idx >= len(glyphs) - 1:
                return None

            left_glyphs = glyphs[:slash_idx]
            right_glyphs = glyphs[slash_idx + 1 :]

            # F-1745: reject low-confidence glyph matches instead of accepting
            # any best-match digit.
            used_digits: List[str] = []
            for gx, gy, gw, gh, _ in left_glyphs:
                crop = binary[gy : gy + gh, gx : gx + gw]
                d, conf = self._classify_glyph(crop)
                if conf < _MIN_GLYPH_CONFIDENCE:
                    return None
                used_digits.append(d)

            max_digits: List[str] = []
            for gx, gy, gw, gh, _ in right_glyphs:
                crop = binary[gy : gy + gh, gx : gx + gw]
                d, conf = self._classify_glyph(crop)
                if conf < _MIN_GLYPH_CONFIDENCE:
                    return None
                max_digits.append(d)

            if not used_digits or not max_digits:
                return None

            parsed_used = int("".join(used_digits))
            parsed_max = int("".join(max_digits))
            if parsed_max <= 0 or parsed_used < 0 or parsed_used > parsed_max * 2:
                return None

            with self._lock:
                self._hud_baseline_slots = parsed_used
                self._gathers_since_hud_sync = 0
                self.used_slots = parsed_used
                self.max_slots = parsed_max
                fill_pct = (float(self.used_slots) / float(self.max_slots)) * 100.0
                if fill_pct >= self.near_full_threshold_pct:
                    self._near_full_warning_fired = True

            return parsed_used, parsed_max
        except Exception as exc:
            # sweep-29 Y-5: gerçek okuma hatası debug'da görünmüyordu
            # (çağrı gather başına en fazla bir kez yapılır, spam riski yok).
            logger.warning("InventoryFullnessEstimator.read_bag_counter_from_frame error: %s", exc)
            return None

    def record_gathered_material(
        self,
        # P1-6 (K-06): türsüz materyal artık "Orichalcum Ore" diye
        # uydurulmaz — state_machine'deki O1 kuralıyla aynı: bilinmeyen
        # tür "Unknown" sayılır, ayrı kovada toplanır.
        resource_type: str = "Unknown",
        stages_completed: int = 3,
        items_per_stage: int = 1,
        frame: Optional[np.ndarray] = None,
        stack_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Records a gathered resource node (including its 3x gathering stages) and updates
        the estimated inventory slot usage. If `frame` is provided, also synchronizes
        against the live HUD bag badge (`61/75`).
        """
        if frame is not None:
            self.read_bag_counter_from_frame(frame)

        clean_resource = str(resource_type or "Unknown").strip() or "Unknown"
        stages = max(1, int(stages_completed))
        qty = max(1, stages * max(1, int(items_per_stage)))

        with self._lock:
            self.total_materials_gathered += qty
            self.total_stages_completed += stages
            prev_count = self._resource_stack_counts[clean_resource]
            self._resource_stack_counts[clean_resource] = prev_count + qty
            self._gathers_since_hud_sync += 1

            # New resource type or full stack rollover (every `stack_size`
            # units; F-1752: the 20-stack was a hidden hard-coded assumption).
            eff_stack = max(1, int(stack_size)) if stack_size else self.stack_size
            if clean_resource not in self._unique_resources_seen:
                self._unique_resources_seen.add(clean_resource)
                self.used_slots = min(self.max_slots, self.used_slots + 1)
            else:
                prev_stacks = max(1, (prev_count + eff_stack - 1) // eff_stack)
                new_stacks = max(1, (self._resource_stack_counts[clean_resource] + eff_stack - 1) // eff_stack)
                if new_stacks > prev_stacks:
                    self.used_slots = min(self.max_slots, self.used_slots + (new_stacks - prev_stacks))

            return self.get_status(frame=None)

    def set_bag_slots(self, used_slots: int, max_slots: Optional[int] = None) -> Dict[str, Any]:
        """Explicitly sets the current `used_slots` and optional `max_slots` capacity."""
        with self._lock:
            if max_slots is not None and int(max_slots) > 0:
                self.max_slots = int(max_slots)
            self.used_slots = max(0, min(int(used_slots), self.max_slots))
            return self.get_status(frame=None)

    def get_status(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Returns the current inventory fullness status dictionary including:
        `inventory_near_full: bool`, `used_slots: int`, `max_slots: int`,
        `free_slots: int`, `fill_pct: float`, and `warning_message: str`.
        """
        if frame is not None:
            self.read_bag_counter_from_frame(frame)

        with self._lock:
            max_s = max(1, self.max_slots)
            used_s = max(0, min(self.used_slots, max_s))
            free_s = max(0, max_s - used_s)
            fill_pct = round((float(used_s) / float(max_s)) * 100.0, 2)
            near_full = bool(fill_pct >= self.near_full_threshold_pct)

            if near_full:
                self._near_full_warning_fired = True
                # F-1762: the text must show the configured threshold, not a
                # hard-coded "95%".
                warning_msg = (
                    f"%{self.near_full_threshold_pct:.0f} Doluluk Uyarısı: Çanta neredeyse "
                    f"dolu ({used_s}/{max_s} slot - %{fill_pct:.1f})!"
                )
            else:
                warning_msg = "OK"

            return {
                "inventory_near_full": near_full,
                "used_slots": used_s,
                "max_slots": max_s,
                "free_slots": free_s,
                "fill_pct": fill_pct,
                "total_materials_gathered": self.total_materials_gathered,
                "total_stages_completed": self.total_stages_completed,
                "warning_message": warning_msg,
                "near_full_threshold_pct": self.near_full_threshold_pct,
                # F-1762: these fields were written but never read anywhere.
                "near_full_warning_fired": self._near_full_warning_fired,
                "hud_baseline_slots": self._hud_baseline_slots,
                "gathers_since_hud_sync": self._gathers_since_hud_sync,
            }
class TelemetryEngine:
    """
    Canlı Telemetri, Oturum Analitiği, Saatlik Verim ve Bildirim/Webhook Motoru.
    Oturum boyunca toplanan kaynakları, zaman damgalarını, yürüyüş sürelerini,
    başarısız node denemelerini, savaş sayısını ve mola sürelerini gerçek zamanlı izler.
    Ayrıca Item 42 (`YieldHeatmapTracker`) ve Item 43 (`InventoryFullnessEstimator`)
    modüllerini entegre eder.
    """

    ALERT_PATTERNS = {
        "DEATH": [(440, 250), (392, 250), (349, 450)],
        "COMBAT_TIMEOUT": [(880, 180), (660, 180), (880, 250)],
        "NO_GATHER_IDLE": [(523, 200), (523, 200)],
        "DAILY_LIMIT": [(587, 150), (784, 150), (987, 300)],
        "INVENTORY_NEAR_FULL": [(659, 180), (784, 180), (1046, 300)],
        "DEFAULT": [(750, 200)],
    }

    def __init__(
        self,
        bucket_span_sec: int = 300,
        num_buckets: int = 12,
        items_per_gather: int = 3,
        history_path: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.bucket_span_sec = max(10, int(bucket_span_sec))
        self.num_buckets = max(4, int(num_buckets))
        self.session_start_ts: float = time.time()
        # F-1754: durations/pauses use a monotonic clock (wall-clock jumps
        # must not corrupt session timing); session_start_ts stays epoch for
        # display/report timestamps.
        self._session_start_mono: float = time.monotonic()
        self.total_pause_sec: float = 0.0
        self._pause_start_ts: Optional[float] = None
        self._pause_start_mono: Optional[float] = None
        # F-1752/F-1753: items yielded per gather node (3 stages x 1 item).
        # Single source for the inventory estimator, the profit simulator
        # and the lifetime counter, so the three can no longer disagree 3x.
        self.items_per_gather: int = max(1, int(items_per_gather))
        self._total_items_gathered: int = 0
        # P1-6 (K-06): kümülatif oturum sayaçları — kaplı olay
        # listelerinden (gather_events 5000, failure_events 2000,
        # combat_events 2000) ayrı tutulur. Kaplama (cap) eskiden
        # `len(gather_events)` ile okunan gathered_total'ı 5000'de
        # takarken kapsız `by_resource` büyümeye devam ediyordu;
        # artık toplamlar bu sayaçlardan okunur, listeler yalnızca
        # son-olay penceresi (ortalama/dilim/kırılım) içindir.
        self._gathered_total: int = 0
        self._failed_total: int = 0
        self._combat_total: int = 0
        # F-1763: last accepted alert per (type, message), monotonic.
        self._last_alert_mono: Dict[Tuple[str, str], float] = {}
        # sweep-29 Y-4: oturum telemetrisi yalnızca RAM'de duruyordu; çökmede
        # / zorla kapatmada kayboluyordu. Periyodik otomatik kayıt
        # (varsayılan 10 dk; save_session_report ile aynı dosyalar).
        self._autosave_interval_sec: float = 600.0
        self._last_autosave_mono: float = time.monotonic()

        self.gather_events: List[Dict[str, Any]] = []
        self.failure_events: List[Dict[str, Any]] = []
        self.combat_events: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []

        self.by_resource: Dict[str, int] = defaultdict(int)
        self.by_route: Dict[str, int] = defaultdict(int)
        self.by_region: Dict[str, int] = defaultdict(int)
        self.failure_reasons: Dict[str, int] = defaultdict(int)

        # Item 42 & Item 43 trackers
        self.heatmap_tracker: YieldHeatmapTracker = YieldHeatmapTracker()
        self.inventory_estimator: InventoryFullnessEstimator = InventoryFullnessEstimator()

        # Lazy-loaded SessionProfitSimulator from core.economy_broker
        self._profit_simulator: Optional[Any] = None
        try:
            from core.economy_broker import SessionProfitSimulator
            self._profit_simulator = SessionProfitSimulator()
        except Exception as _exc:
            # sweep-29 Y-5: gerçek yükleme hatası debug'da görünmüyordu.
            logger.warning("SessionProfitSimulator lazy-load edilemedi: %s", _exc)

        # v20-istatistik (R2): geçmiş oturum olayları — YALNIZCA history_path
        # açıkça verilirse yüklenir (üretim motoru main.py'den alır).
        # save_session_report logs/telemetry_history.json dosyasına yazar
        # ama dosyayı hiçbir yer okumuyordu: "Bugün" her restart'ta
        # sıfırlanıyordu. Geçmiş olaylar AYRI listelerde tutulur; canlı
        # oturum listeleri ve kümülatif sayaçlar asla kirletilmez, bu
        # yüzden "Oturum" aralığı ve boş-durum testleri etkilenmez.
        self.history_gather_events: List[Dict[str, Any]] = []
        self.history_failure_events: List[Dict[str, Any]] = []
        self.history_combat_events: List[Dict[str, Any]] = []
        if history_path:
            self.load_history_events(history_path)

    def load_history_events(
        self, history_path: str = "logs/telemetry_history.json"
    ) -> Dict[str, int]:
        """`save_session_report` ile yazılan geçmiş oturum olaylarını geri yükler.

        Son 7 gün içindeki gather/failure/combat olayları
        `history_*_events` listelerine konur (en yeniden eskiye, kaplı).
        Aynı olay birden çok snapshot'ta geçebilir (otomatik kayıt her
        10 dk'da bir yazar) — (zaman, düğüm, tür/sonuç) anahtarıyla
        tekilleştirilir. Best-effort: dosya yoksa/bozuksa boş döner,
        asla exception fırlatmaz.
        """
        counts = {"gather": 0, "failure": 0, "combat": 0}
        try:
            if not history_path or not os.path.exists(history_path):
                return counts
            with open(history_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                loaded = [loaded]
            if not isinstance(loaded, list):
                return counts
            cutoff = time.time() - 7 * 86400.0

            def _ts(ev: Any) -> float:
                try:
                    return float(ev.get("timestamp", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            seen_g, seen_f, seen_c = set(), set(), set()
            gathers, failures, combats = [], [], []
            for entry in reversed(loaded):
                if not isinstance(entry, dict):
                    continue
                for ev in entry.get("recent_gathers") or []:
                    if not isinstance(ev, dict) or _ts(ev) <= cutoff:
                        continue
                    key = (round(_ts(ev), 3), ev.get("node_id"), str(ev.get("resource_type", "")))
                    if key in seen_g:
                        continue
                    seen_g.add(key)
                    gathers.append(ev)
                for ev in entry.get("recent_failures") or []:
                    if not isinstance(ev, dict) or _ts(ev) <= cutoff:
                        continue
                    key = (round(_ts(ev), 3), ev.get("node_id"), str(ev.get("reason", "")))
                    if key in seen_f:
                        continue
                    seen_f.add(key)
                    failures.append(ev)
                for ev in entry.get("recent_combats") or []:
                    if not isinstance(ev, dict) or _ts(ev) <= cutoff:
                        continue
                    key = (round(_ts(ev), 3), str(ev.get("outcome", "")), ev.get("duration_sec"))
                    if key in seen_c:
                        continue
                    seen_c.add(key)
                    combats.append(ev)
            by_ts = lambda e: _ts(e)
            gathers.sort(key=by_ts)
            failures.sort(key=by_ts)
            combats.sort(key=by_ts)
            with self._lock:
                self.history_gather_events = gathers[-_MAX_GATHER_EVENTS:]
                self.history_failure_events = failures[-_MAX_FAILURE_EVENTS:]
                self.history_combat_events = combats[-_MAX_COMBAT_EVENTS:]
                counts = {
                    "gather": len(self.history_gather_events),
                    "failure": len(self.history_failure_events),
                    "combat": len(self.history_combat_events),
                }
            if any(counts.values()):
                logger.info("Telemetri geçmişi yüklendi (%s): %s", history_path, counts)
        except Exception as exc:
            logger.warning("Telemetri geçmişi yüklenemedi (%s): %s", history_path, exc)
        return counts

    def _maybe_autosave(self) -> None:
        """Sweep-29 Y-4: oturum telemetrisini periyodik olarak diske yazar.

        save_session_report yalnızca temiz kapanışta çağrılıyordu ve o çağrı
        _stop_quietly içinde hatayı debug'da yutuyordu — çökmede tüm oturum
        kayboluyordu. Artık her gather/savaş/başarısızlık kaydında süre
        kontrol edilir; 10 dk dolduysa rapor yazılır. Yazım hatası warning
        ile görünür (yutulmaz). Best-effort: hiçbir exception fırlatmaz.
        """
        now_mono = time.monotonic()
        if (now_mono - self._last_autosave_mono) < float(self._autosave_interval_sec):
            return
        self._last_autosave_mono = now_mono
        try:
            ok = self.save_session_report()
        except Exception as exc:
            logger.warning("Telemetri otomatik kaydı hata verdi: %s", exc)
            return
        if not ok:
            logger.warning(
                "Telemetri otomatik kaydı başarısız (save_session_report False döndü)."
            )

    def _elapsed_sec(self, now_mono: Optional[float] = None) -> float:
        """F-1754: monotonic session elapsed seconds excluding pauses."""
        now_m = time.monotonic() if now_mono is None else now_mono
        active_pause = (now_m - self._pause_start_mono) if self._pause_start_mono is not None else 0.0
        return max(0.0, (now_m - self._session_start_mono) - (self.total_pause_sec + active_pause))

    def _append_capped(
        self, lst: List[Dict[str, Any]], event: Dict[str, Any], cap: int
    ) -> None:
        """F-1755: append an event, dropping oldest entries past the cap."""
        lst.append(event)
        if len(lst) > cap:
            del lst[: len(lst) - cap]

    def reset_session(self) -> None:
        """Oturum istatistiklerini sıfırlar ve yeni bir oturum başlatır."""
        with self._lock:
            self.session_start_ts = time.time()
            self._session_start_mono = time.monotonic()
            self.total_pause_sec = 0.0
            self._pause_start_ts = None
            self._pause_start_mono = None
            self._total_items_gathered = 0
            self._gathered_total = 0
            self._failed_total = 0
            self._combat_total = 0
            self._last_alert_mono.clear()
            self._last_autosave_mono = time.monotonic()
            self.gather_events.clear()
            self.failure_events.clear()
            self.combat_events.clear()
            self.alert_history.clear()
            self.by_resource.clear()
            self.by_route.clear()
            self.by_region.clear()
            self.failure_reasons.clear()
            self.heatmap_tracker.reset()
            self.inventory_estimator.reset()
            if self._profit_simulator is not None and hasattr(self._profit_simulator, "reset_session"):
                try:
                    self._profit_simulator.reset_session()
                except Exception:
                    pass
            logger.info("Telemetri oturum verileri sıfırlandı.")

    def record_node_yield_event(
        self,
        node_id: int,
        x: float,
        y: float,
        success: bool = True,
    ) -> Dict[str, Any]:
        """Item 42 — Kaynak düğümü verim olayını ısı haritasına kaydeder."""
        return self.heatmap_tracker.record_node_yield_event(
            node_id=node_id,
            x=x,
            y=y,
            success=success,
        )

    def record_combat_ambush_event(self, x: float, y: float) -> Dict[str, Any]:
        """Item 42 — Yaratık pususu / savaş tehlike noktasını ısı haritasına kaydeder."""
        return self.heatmap_tracker.record_combat_ambush_event(x=x, y=y)

    def get_heatmap_points(self) -> List[Dict[str, Any]]:
        """Item 42 — Harita görüntüleyici ve Mobil Radar için ısı haritası noktalarını döndürür."""
        return self.heatmap_tracker.get_heatmap_points()

    def update_inventory_from_frame(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Item 43 — Ekran görüntüsündeki sağ-üst çanta sayacını (`61/75`) okur ve doluluk durumunu döndürür."""
        return self.inventory_estimator.get_status(frame=frame)

    def get_economy_summary(self) -> Dict[str, Any]:
        """
        Returns estimated Kinah profit and hourly Kinah rate from the optional
        SessionProfitSimulator (if available), or computes a fallback summary.
        """
        with self._lock:
            if self._profit_simulator is not None:
                try:
                    if hasattr(self._profit_simulator, "get_session_economy_summary"):
                        eff_hours = max(1.0 / 3600.0, self._elapsed_sec() / 3600.0)
                        return self._profit_simulator.get_session_economy_summary(elapsed_hours=eff_hours)
                except Exception as _exc:
                    logger.debug("SessionProfitSimulator get_session_economy_summary error: %s", _exc)

            # Fallback summary if profit simulator is not loaded or failed.
            # F-1743: no fabricated constants — without price data there is no
            # Kinah estimate at all (the old `gathered * 3 * 14500` invented
            # money and exploded hourly rates early in a session).
            eff_hours = max(1.0 / 3600.0, self._elapsed_sec() / 3600.0)
            return {
                "net_kinah_earned": 0,
                "gross_kinah_earned": 0,
                "kinah_per_hour": 0,
                "elapsed_hours": round(eff_hours, 4),
                "total_items_gathered": int(self._total_items_gathered),
                "formatted_net_kinah": "0",
                "formatted_kinah_per_hour": "0",
                "formatted_summary_tr": "Fiyat verisi yok — Kinah tahmini yapılamadı",
                "material_breakdown": dict(self.by_resource),
                "price_data_available": False,
                "estimated": False,
            }

    def record_pause_start(self) -> None:
        """Mola başlangıcını kaydeder."""
        with self._lock:
            if self._pause_start_mono is None:
                self._pause_start_ts = time.time()
                self._pause_start_mono = time.monotonic()

    def record_pause_end(self) -> None:
        """Mola bitişini kaydeder ve toplam mola süresine ekler."""
        with self._lock:
            if self._pause_start_mono is not None:
                elapsed = max(0.0, time.monotonic() - self._pause_start_mono)
                self.total_pause_sec += elapsed
                self._pause_start_ts = None
                self._pause_start_mono = None

    def record_gather(
        self,
        node_id: int,
        region: str,
        route: str,
        # P1-6 (K-06): türsüz toplama "Orichalcum Ore" varsayılmaz —
        # bilinmeyen tür "Unknown" olarak kaydedilir (O1 kuralı).
        resource_type: str = "Unknown",
        walk_duration_sec: float = 0.0,
        match_pct: int = 80,
        stages_completed: Optional[int] = None,
        items_per_stage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Başarılı bir kaynak toplama işlemini kaydeder.
        F-1752: stage/item counts are explicit parameters (default = this
        engine's items_per_gather), not hidden hard-coded assumptions.
        F-1749: nothing is appended until every step that can fail has run —
        a partial record used to return {} while staying in the list.
        F-1753: lifetime, profit simulator and inventory all count the same
        item quantity (items_per_gather per node), not 1 vs 3.
        """
        try:
            now = time.time()
            clean_region = str(region or "Unknown").strip()
            clean_route = str(route or "Default Route").strip()
            clean_resource = str(resource_type or "Unknown").strip() or "Unknown"
            clean_walk = max(0.0, float(walk_duration_sec))
            clean_match = max(0, min(100, int(match_pct)))
            # Defaults reproduce the historical 3-stages x 1-item node; the
            # values are parameters now, and items_per_gather is their product.
            stages = self.items_per_gather if stages_completed is None else max(1, int(stages_completed))
            per_stage = 1 if items_per_stage is None else max(1, int(items_per_stage))
            items = stages * per_stage

            event = {
                "timestamp": now,
                "iso_time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "node_id": int(node_id),
                "region": clean_region,
                "route": clean_route,
                "resource_type": clean_resource,
                "walk_duration_sec": round(clean_walk, 3),
                "match_pct": clean_match,
                "items_gathered": items,
            }

            # Compute inventory status BEFORE appending, so a failure here
            # leaves no partial record behind (F-1749).
            inv_status = self.inventory_estimator.record_gathered_material(
                resource_type=clean_resource,
                stages_completed=stages,
                items_per_stage=per_stage,
            )
            event["inventory_near_full"] = inv_status["inventory_near_full"]
            event["used_slots"] = inv_status["used_slots"]
            event["max_slots"] = inv_status["max_slots"]
            event["fill_pct"] = inv_status["fill_pct"]

            with self._lock:
                self._append_capped(self.gather_events, event, _MAX_GATHER_EVENTS)
                # P1-6: kümülatif sayaç kaplı listeden bağımsız artar.
                self._gathered_total += 1
                self.by_resource[clean_resource] += 1
                self.by_route[clean_route] += 1
                self.by_region[clean_region] += 1
                self._total_items_gathered += items

                if self._profit_simulator is not None:
                    try:
                        if hasattr(self._profit_simulator, "record_gather"):
                            self._profit_simulator.record_gather(clean_resource, count=items)
                        elif hasattr(self._profit_simulator, "record_gathered_material"):
                            self._profit_simulator.record_gathered_material(clean_resource, count=items)
                    except Exception as _ps_err:
                        # sweep-29 Y-5: gerçek kayıt hatası debug'da görünmüyordu.
                        logger.warning("SessionProfitSimulator record_gather error: %s", _ps_err)

            # Kalıcı lifetime sayaç (oturum reset'inden bağımsız; tek kaynak: telemetry.record_gather)
            try:
                from core.lifetime_stats import get_lifetime_stats

                get_lifetime_stats().record_material(clean_resource, count=items)
            except Exception as _lt_err:
                # sweep-29 Y-5: gerçek kayıt hatası debug'da görünmüyordu.
                logger.warning("lifetime material record skipped: %s", _lt_err)

            logger.info(
                f"Gather kaydedildi: Node #{node_id} | {clean_resource} | "
                f"{clean_region}/{clean_route} | Yürüyüş: {clean_walk:.2f}s | Eşleşme: %{clean_match}"
            )
            # sweep-29 Y-4: periyodik otomatik kayıt (çökmede kayıp olmasın).
            self._maybe_autosave()
            return event
        except Exception as exc:
            logger.error(f"record_gather sırasında hata: {exc}", exc_info=True)
            return {}

    def record_failure(self, node_id: int, reason: str = "timeout") -> Dict[str, Any]:
        """Başarısız bir node toplama/erişim denemesini kaydeder."""
        try:
            now = time.time()
            clean_reason = str(reason or "timeout").strip()
            event = {
                "timestamp": now,
                "iso_time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "node_id": int(node_id),
                "reason": clean_reason,
            }
            with self._lock:
                self._append_capped(self.failure_events, event, _MAX_FAILURE_EVENTS)
                # P1-6: kümülatif sayaç kaplı listeden bağımsız artar.
                self._failed_total += 1
                self.failure_reasons[clean_reason] += 1

            logger.warning(f"Node başarısızlığı kaydedildi: Node #{node_id} | Sebep: {clean_reason}")
            # sweep-29 Y-4: periyodik otomatik kayıt.
            self._maybe_autosave()
            return event
        except Exception as exc:
            logger.error(f"record_failure sırasında hata: {exc}", exc_info=True)
            return {}

    def record_combat(
        self, duration_sec: float, outcome: str = "CLEARED", mob_key: str = ""
    ) -> Dict[str, Any]:
        """Oturum sırasında gerçekleşen bir savaş olayını kaydeder.
        F-1764: the lifetime kill counter is keyed by the actual mob when the
        caller knows it; the combat outcome is never used as a fake mob key.
        """
        try:
            now = time.time()
            clean_duration = max(0.0, float(duration_sec))
            clean_outcome = str(outcome or "CLEARED").strip().upper()
            event = {
                "timestamp": now,
                "iso_time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "duration_sec": round(clean_duration, 3),
                "outcome": clean_outcome,
            }
            with self._lock:
                self._append_capped(self.combat_events, event, _MAX_COMBAT_EVENTS)
                # P1-6: kümülatif sayaç kaplı listeden bağımsız artar.
                self._combat_total += 1

            # CLEARED / WIN tarzı sonuçlarda kalıcı canavar sayacı (debounce store içinde)
            if clean_outcome in ("CLEARED", "WIN", "VICTORY", "KILLED", "SUCCESS"):
                try:
                    from core.lifetime_stats import get_lifetime_stats

                    get_lifetime_stats().record_monster_kill(mob_key=str(mob_key or ""))
                except Exception as _lt_err:
                    # sweep-29 Y-5: gerçek kayıt hatası debug'da görünmüyordu.
                    logger.warning("lifetime combat kill skipped: %s", _lt_err)

            logger.info(f"Savaş kaydedildi: Süre {clean_duration:.2f}s | Sonuç: {clean_outcome}")
            # sweep-29 Y-4: periyodik otomatik kayıt.
            self._maybe_autosave()
            return event
        except Exception as exc:
            logger.error(f"record_combat sırasında hata: {exc}", exc_info=True)
            return {}

    def _compute_hourly_buckets(self, now: float) -> List[Dict[str, Any]]:
        """Son 12 zaman dilimi için grafik ve hız verilerini hesaplar.

        Each event is assigned to its bucket by index arithmetic instead of
        scanning all 12 buckets per event (was O(events x buckets)). The
        assignment preserves the old loop's exact semantics: an event on a
        shared boundary belongs to the earlier bucket (the old loop broke at
        the first `start <= ts <= end` match), an event at the window end
        belongs to the last bucket, and events outside the window are skipped.
        The one-step adjust guards the float rounding of the division.
        """
        buckets: List[Dict[str, Any]] = []
        span = float(self.bucket_span_sec)
        elapsed = max(0.001, now - self.session_start_ts)

        # Eğer oturum yeni başladıysa dinamik dilim genişliğiyle anlamlı dağılım üret
        effective_span = span if elapsed >= span * 2 else max(5.0, elapsed / float(self.num_buckets))
        window_start = now - self.num_buckets * effective_span

        starts: List[float] = []
        ends: List[float] = []
        for i in range(self.num_buckets):
            # Same edge arithmetic as before (now - (num-i)*span) so the
            # returned start_ts/end_ts values are bit-identical to the old code.
            start_t = now - (self.num_buckets - i) * effective_span
            end_t = start_t + effective_span
            starts.append(start_t)
            ends.append(end_t)
            label = datetime.fromtimestamp(end_t).strftime("%H:%M:%S")
            buckets.append(
                {
                    "index": i,
                    "label": label,
                    "start_ts": start_t,
                    "end_ts": end_t,
                    "count": 0,
                    "failures": 0,
                    "combats": 0,
                    "rate_per_hour": 0.0,
                }
            )

        def _bucket_index(ts: float) -> int:
            # Floor division (not int() truncation): a timestamp just before
            # the window must yield -1, not 0 — int(-0.003) is 0.
            idx = int((ts - window_start) // effective_span)
            if idx < 0:
                return -1
            if idx >= self.num_buckets:
                return self.num_buckets - 1 if ts <= ends[-1] else -1
            if idx > 0 and ts <= ends[idx - 1]:
                idx -= 1
            elif ts > ends[idx]:
                idx += 1
                if idx >= self.num_buckets:
                    return -1
            return idx

        for ev in self.gather_events:
            idx = _bucket_index(ev["timestamp"])
            if idx >= 0:
                buckets[idx]["count"] += 1

        for fv in self.failure_events:
            idx = _bucket_index(fv["timestamp"])
            if idx >= 0:
                buckets[idx]["failures"] += 1

        for cv in self.combat_events:
            idx = _bucket_index(cv["timestamp"])
            if idx >= 0:
                buckets[idx]["combats"] += 1

        for b in buckets:
            b["rate_per_hour"] = round((b["count"] / effective_span) * 3600.0, 2)

        return buckets

    def get_live_metrics(self) -> Dict[str, Any]:
        """
        Gerçek zamanlı oturum metriklerini döndürür:
        session_duration_sec, gathered_total, gathers_per_hour, avg_walk_sec,
        success_rate_pct, combat_count, failed_total, hourly_buckets,
        by_resource, by_route.
        """
        try:
            with self._lock:
                now = time.time()
                now_mono = time.monotonic()
                active_pause = (now_mono - self._pause_start_mono) if self._pause_start_mono is not None else 0.0
                raw_duration = max(0.001, now_mono - self._session_start_mono)
                effective_duration = max(0.001, raw_duration - (self.total_pause_sec + active_pause))

                # P1-6 (K-06): toplamlar kaplı listelerin `len()`'inden
                # DEĞİL ayrı kümülatif sayaçlardan okunur — 5000 gather
                # sonrası liste başından atmaya başlasa da toplam takılmaz.
                # Ortalama/pay kırılımları ise tutulan son-olay
                # penceresinden hesaplanır (window_gathered).
                gathered_total = int(self._gathered_total)
                failed_total = int(self._failed_total)
                combat_count = int(self._combat_total)
                window_gathered = len(self.gather_events)
                total_attempts = gathered_total + failed_total

                # Saatlik toplama hızı (en az 1 saniye taban alınarak aşırı sıçramalar dengelenir)
                rate_divisor = max(1.0, effective_duration)
                gathers_per_hour = round((gathered_total / rate_divisor) * 3600.0, 2)

                # Single pass over gather_events: the old code scanned the
                # (up to 5000) events three times — two sum() genexprs plus
                # the breakdown loop, each doing several dict.get() calls per
                # event. One loop with direct key access (record_gather always
                # writes these keys) computes the same totals.
                total_walk = 0.0
                total_match = 0
                resource_breakdown: Dict[str, Dict[str, Any]] = {}
                for g in self.gather_events:
                    res = g["resource_type"]
                    reg = g["region"]
                    rte = g["route"]
                    walk = float(g["walk_duration_sec"])
                    match = int(g["match_pct"])
                    total_walk += walk
                    total_match += match
                    key = f"{res}||{reg}||{rte}"
                    entry = resource_breakdown.get(key)
                    if entry is None:
                        entry = {
                            "resource_type": res,
                            "region": reg,
                            "route": rte,
                            "count": 0,
                            "total_walk_sec": 0.0,
                            "total_match_pct": 0,
                        }
                        resource_breakdown[key] = entry
                    entry["count"] += 1
                    entry["total_walk_sec"] += walk
                    entry["total_match_pct"] += match

                if window_gathered > 0:
                    avg_walk_sec = round(total_walk / window_gathered, 2)
                    avg_match_pct = round(total_match / window_gathered, 1)
                else:
                    avg_walk_sec = 0.0
                    avg_match_pct = 0.0

                if total_attempts > 0:
                    success_rate_pct = round((gathered_total / total_attempts) * 100.0, 2)
                else:
                    # F-1750: no attempts yet means "unknown" (None), never 100%.
                    success_rate_pct = None

                combat_duration_total = round(
                    sum(float(c.get("duration_sec", 0.0)) for c in self.combat_events), 2
                )

                hourly_buckets = self._compute_hourly_buckets(now)

                detailed_rows: List[Dict[str, Any]] = []
                for item in resource_breakdown.values():
                    cnt = max(1, item["count"])
                    detailed_rows.append(
                        {
                            "resource_type": item["resource_type"],
                            "region": item["region"],
                            "route": item["route"],
                            "count": item["count"],
                            # P1-6: pay, kırılımın hesaplandığı tutulan
                            # pencereye bölünür (kümülatif toplama değil).
                            "share_pct": round((item["count"] / max(1, window_gathered)) * 100.0, 1),
                            "avg_walk_sec": round(item["total_walk_sec"] / cnt, 2),
                            "avg_match_pct": round(item["total_match_pct"] / cnt, 1),
                        }
                    )
                detailed_rows.sort(key=lambda x: x["count"], reverse=True)

                inv_status = self.inventory_estimator.get_status(frame=None)
                heatmap_pts = self.heatmap_tracker.get_heatmap_points()

                return {
                    "session_duration_sec": round(raw_duration, 2),
                    "effective_duration_sec": round(effective_duration, 2),
                    "pause_duration_sec": round(self.total_pause_sec + active_pause, 2),
                    "gathered_total": gathered_total,
                    "gathers_per_hour": gathers_per_hour,
                    "avg_walk_sec": avg_walk_sec,
                    "avg_match_pct": avg_match_pct,
                    "success_rate_pct": success_rate_pct,
                    "combat_count": combat_count,
                    "combat_duration_sec": combat_duration_total,
                    "failed_total": failed_total,
                    "hourly_buckets": hourly_buckets,
                    "by_resource": dict(self.by_resource),
                    "by_route": dict(self.by_route),
                    "by_region": dict(self.by_region),
                    "failure_reasons": dict(self.failure_reasons),
                    "detailed_breakdown": detailed_rows,
                    "heatmap_points": heatmap_pts,
                    "inventory_near_full": inv_status["inventory_near_full"],
                    "used_slots": inv_status["used_slots"],
                    "max_slots": inv_status["max_slots"],
                    "free_slots": inv_status["free_slots"],
                    "fill_pct": inv_status["fill_pct"],
                    "inventory_status": inv_status,
                }
        except Exception as exc:
            logger.error(f"get_live_metrics sırasında hata: {exc}", exc_info=True)
            # F-1750: the error path returns the FULL schema (same keys as the
            # success path) and never claims 100% success on failure.
            return {
                "session_duration_sec": 0.0,
                "effective_duration_sec": 0.0,
                "pause_duration_sec": 0.0,
                "gathered_total": 0,
                "gathers_per_hour": 0.0,
                "avg_walk_sec": 0.0,
                "avg_match_pct": 0.0,
                "success_rate_pct": None,
                "combat_count": 0,
                "combat_duration_sec": 0.0,
                "failed_total": 0,
                "hourly_buckets": [],
                "by_resource": {},
                "by_route": {},
                "by_region": {},
                "failure_reasons": {},
                "detailed_breakdown": [],
                "heatmap_points": [],
                "inventory_near_full": False,
                # sweep-29 O-1: hata yolunda artık uydurma çanta yazılmıyor —
                # eskiden max_slots=75 / "OK" dönüyor, hata anında çanta boş
                # ve her şey yolundaymış gibi görünüyordu. Bilinmiyorsa 0
                # ve "ölçülemedi" yazılır; metrics_error=True zaten var.
                "used_slots": 0,
                "max_slots": 0,
                "free_slots": 0,
                "fill_pct": 0.0,
                "inventory_status": {
                    "inventory_near_full": False,
                    "used_slots": 0,
                    "max_slots": 0,
                    "free_slots": 0,
                    "fill_pct": 0.0,
                    "warning_message": "ölçülemedi (metrik hatası)",
                },
                "metrics_error": True,
                "error": str(exc),
            }

    def save_session_report(
        self,
        json_path: str = "logs/telemetry_history.json",
        csv_path: str = "logs/session_stats.csv",
    ) -> bool:
        """Oturum verilerini JSON geçmiş dosyasına ve CSV tablosuna kaydeder.
        F-1744: the JSON history is written atomically (tmp + os.replace),
        capped at _MAX_HISTORY_ENTRIES entries, and guarded by the engine
        lock so concurrent saves cannot interleave or truncate each other.
        S-3201: the CSV append (header check-then-act included) runs under
        the same lock, so two concurrent saves cannot both see the file as
        missing and write a duplicate header row.
        """
        try:
            metrics = self.get_live_metrics()
            now_iso = datetime.now().isoformat(timespec="seconds")

            if json_path:
                json_dir = os.path.dirname(os.path.abspath(json_path))
                if json_dir:
                    os.makedirs(json_dir, exist_ok=True)

                with self._lock:
                    history_data: List[Dict[str, Any]] = []
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                loaded = json.load(jf)
                                if isinstance(loaded, list):
                                    history_data = loaded
                                elif isinstance(loaded, dict):
                                    history_data = [loaded]
                        except Exception as read_err:
                            logger.warning(f"Mevcut JSON geçmişi okunamadı, yeni oluşturuluyor: {read_err}")

                    session_snapshot = {
                        "saved_at": now_iso,
                        "session_start": datetime.fromtimestamp(self.session_start_ts).isoformat(timespec="seconds"),
                        "metrics": metrics,
                        "recent_gathers": list(self.gather_events[-100:]),
                        "recent_failures": list(self.failure_events[-50:]),
                        "recent_combats": list(self.combat_events[-50:]),
                    }
                    history_data.append(session_snapshot)
                    # Cap: keep only the newest entries, history never grows unbounded.
                    if len(history_data) > _MAX_HISTORY_ENTRIES:
                        history_data = history_data[-_MAX_HISTORY_ENTRIES:]

                    tmp_path = f"{json_path}.tmp.{os.getpid()}"
                    try:
                        with open(tmp_path, "w", encoding="utf-8") as jf:
                            json.dump(history_data, jf, ensure_ascii=False, indent=2)
                            jf.flush()
                            try:
                                os.fsync(jf.fileno())
                            except OSError:
                                pass
                        os.replace(tmp_path, json_path)
                    except Exception:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                        raise

            with self._lock:
                if csv_path:
                    csv_dir = os.path.dirname(os.path.abspath(csv_path))
                    if csv_dir:
                        os.makedirs(csv_dir, exist_ok=True)

                    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
                    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf)
                        if not file_exists:
                            writer.writerow(
                                [
                                    "timestamp",
                                    "session_duration_sec",
                                    "gathered_total",
                                    "gathers_per_hour",
                                    "avg_walk_sec",
                                    "success_rate_pct",
                                    "combat_count",
                                    "failed_total",
                                    "top_resource",
                                    "top_route",
                                ]
                            )

                        top_resource = (
                            max(metrics["by_resource"].items(), key=lambda x: x[1])[0]
                            if metrics["by_resource"]
                            else "None"
                        )
                        top_route = (
                            max(metrics["by_route"].items(), key=lambda x: x[1])[0]
                            if metrics["by_route"]
                            else "None"
                        )

                        writer.writerow(
                            [
                                now_iso,
                                metrics["session_duration_sec"],
                                metrics["gathered_total"],
                                metrics["gathers_per_hour"],
                                metrics["avg_walk_sec"],
                                metrics["success_rate_pct"],
                                metrics["combat_count"],
                                metrics["failed_total"],
                                top_resource,
                                top_route,
                            ]
                        )

            logger.info(f"Oturum raporu başarıyla kaydedildi: JSON={json_path}, CSV={csv_path}")
            return True
        except Exception as exc:
            logger.error(f"save_session_report sırasında hata: {exc}", exc_info=True)
            return False

    @staticmethod
    def _play_alert_pattern(pattern: List[Tuple[int, int]]) -> None:
        """F-1763: alert sounds play on a daemon thread — send_alert must
        never block the bot loop on seconds of beeps."""
        if not HAS_WINSOUND:
            return
        try:
            for freq, dur in pattern:
                winsound.Beep(int(freq), int(dur))
        except Exception as sound_err:
            logger.warning("Sesli uyarı çalınamadı: %s", sound_err)

    def _post_webhook_alert(
        self,
        clean_url: str,
        data_bytes: bytes,
        clean_type: str,
        alert_record: Dict[str, Any],
    ) -> None:
        """Background webhook POST for :meth:`send_alert` (P2-2).

        Runs on a daemon thread so the 5 s ``urlopen`` timeout can never
        stall the bot loop. The outcome is NOT swallowed: success is
        logged at info level, any failure (HTTP non-2xx or exception) is
        logged at error level, and the result is written back into the
        ``alert_history`` record (shared dict, under the engine lock).
        """
        status_code: Any = None
        try:
            req = urllib.request.Request(
                clean_url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Gatherion-A2-TelemetryEngine/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                status_code = getattr(resp, "status", 200)
                delivered = 200 <= int(status_code) < 300
            with self._lock:
                alert_record["webhook_sent"] = delivered
            if delivered:
                logger.info("Webhook bildirimi gönderildi [%s]", clean_type)
            else:
                logger.error(
                    "Webhook bildirimi HTTP %s döndürdü [%s]",
                    status_code,
                    clean_type,
                )
        except Exception as web_err:
            with self._lock:
                alert_record["webhook_sent"] = False
            logger.error(
                f"Webhook bildirimi gönderilemedi ({clean_type}): {web_err}"
            )

    def send_alert(
        self,
        event_type: str,
        message: str,
        webhook_url: str = "",
        play_sound: bool = True,
    ) -> bool:
        """
        Kritik olaylarda (DEATH, COMBAT_TIMEOUT, NO_GATHER_IDLE, DAILY_LIMIT)
        winsound ile özel sesli uyarı çalar ve eğer webhook_url tanımlıysa
        standart urllib.request ile JSON payload gönderir.
        F-1751: returns True only if at least one channel actually delivered
        (sound queued on a platform that has it, or webhook 2xx). Nothing
        sent -> False. F-1763: identical alerts within _ALERT_DEDUP_SEC are
        suppressed, and the sound no longer blocks the caller.
        P2-2: the webhook POST is now asynchronous (daemon thread, same as
        the alert sound) — the 5 s urlopen timeout no longer blocks the bot
        loop. Return semantics change accordingly: True now means the sound
        was delivered/queued OR the webhook was queued for sending; the
        actual webhook outcome lands later in the alert_history record
        (``webhook_sent``) and in the log. Nothing sent/queued -> False.
        """
        clean_type = str(event_type or "DEFAULT").strip().upper()
        clean_msg = str(message or "").strip()
        now_iso = datetime.now().isoformat(timespec="seconds")

        alert_record = {
            "timestamp": now_iso,
            "event_type": clean_type,
            "message": clean_msg,
            "webhook_configured": bool(webhook_url and webhook_url.strip()),
            "sound_played": False,
            "sound_queued": False,
            "webhook_sent": False,
            "webhook_queued": False,
            "suppressed_duplicate": False,
        }

        # Repeat protection: same type+message inside the dedup window.
        dedup_key = (clean_type, clean_msg)
        with self._lock:
            last = self._last_alert_mono.get(dedup_key)
            now_mono = time.monotonic()
            if last is not None and (now_mono - last) < _ALERT_DEDUP_SEC:
                alert_record["suppressed_duplicate"] = True
                self._append_capped(self.alert_history, alert_record, _MAX_ALERT_HISTORY)
                logger.info("Uyarı tekrar koruması: atlandı [%s]: %s", clean_type, clean_msg)
                return False
            self._last_alert_mono[dedup_key] = now_mono

        sound_delivered = False
        if play_sound:
            if HAS_WINSOUND:
                pattern = self.ALERT_PATTERNS.get(clean_type, self.ALERT_PATTERNS["DEFAULT"])
                threading.Thread(
                    target=self._play_alert_pattern, args=(pattern,), daemon=True
                ).start()
                alert_record["sound_queued"] = True
                alert_record["sound_played"] = True
                sound_delivered = True
            else:
                logger.warning("Sesli uyarı isteniyor ama winsound yok (%s)", clean_type)

        # P2-2: webhook POST is async (daemon thread, like the alert
        # sound) — building the payload is cheap, the network round-trip
        # (up to 5 s) must not stall the bot loop. The outcome is logged
        # and written back into alert_record by _post_webhook_alert.
        webhook_queued = False
        clean_url = str(webhook_url or "").strip()
        if clean_url:
            try:
                metrics = self.get_live_metrics()
                payload = {
                    "username": "Gatherion A2 Telemetry",
                    "content": f"**[{clean_type}]** {clean_msg}",
                    "event_type": clean_type,
                    "message": clean_msg,
                    "timestamp": now_iso,
                    "metrics_summary": {
                        "gathered_total": metrics.get("gathered_total", 0),
                        "gathers_per_hour": metrics.get("gathers_per_hour", 0.0),
                        "success_rate_pct": metrics.get("success_rate_pct", 0.0),
                        "combat_count": metrics.get("combat_count", 0),
                        "failed_total": metrics.get("failed_total", 0),
                    },
                }
                data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                threading.Thread(
                    target=self._post_webhook_alert,
                    args=(clean_url, data_bytes, clean_type, alert_record),
                    daemon=True,
                ).start()
                alert_record["webhook_queued"] = True
                webhook_queued = True
            except Exception as exc:
                logger.error(
                    f"Webhook bildirimi kuyruğa alınamadı ({clean_type}): {exc}"
                )

        success = bool(sound_delivered or webhook_queued)
        if not success:
            logger.warning(
                "Uyarı hiçbir kanaldan iletilemedi [%s]: %s (ses yok, webhook yok/kuyruğa alınamadı)",
                clean_type,
                clean_msg,
            )

        with self._lock:
            self._append_capped(self.alert_history, alert_record, _MAX_ALERT_HISTORY)

        logger.info(f"Uyarı işlendi [{clean_type}]: {clean_msg} (Durum={success})")
        return success

    def count_recent_failures(self, window_sec: float = 3600.0) -> int:
        """P1-6 (K-06): son `window_sec` içindeki başarısızlık sayısı.

        Tek kaynak: telemetri `failure_events` zaman damgaları. Anasayfa
        "Hata / son 1 saatte" rozeti bu değeri gösterir; oturum toplamı
        (`failed_total`) ya da ardışık hata serisi (`failed_in_row`)
        ile karıştırılmaz.
        """
        try:
            cutoff = time.time() - max(0.0, float(window_sec))
            with self._lock:
                events = list(self.failure_events)
            count = 0
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                try:
                    ts = float(ev.get("timestamp", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if ts >= cutoff:
                    count += 1
            return count
        except Exception:
            return 0

    def get_telemetry_health_summary(self) -> dict:
        """Live health snapshot with real session counters (not a constant).

        S-0801: eskiden sabit True donuyordu (try/except hic fail edemezdi).
        `telemetry_ready` gercek prob sonucudur: canli metrikler
        uretilebilmeli, hata bayragi yok olmali ve alt izleyiciler
        (heatmap + envanter) ayakta olmali. Kalan alanlar canli oturum
        durumundan okunur — bos oturumla aktif oturum ayirt edilebilir.
        """
        try:
            # S-0801 probu: canli metrikler + alt izleyiciler + aralik kontrolleri.
            metrics = self.get_live_metrics()
            ready = bool(
                isinstance(metrics, dict)
                and not metrics.get("metrics_error", False)
                and self.heatmap_tracker is not None
                and self.inventory_estimator is not None
                and float(metrics.get("session_duration_sec", -1.0)) >= 0.0
                and int(metrics.get("max_slots", 0)) > 0
                and 0.0 <= float(metrics.get("fill_pct", -1.0)) <= 100.0
                and isinstance(metrics.get("hourly_buckets"), list)
            )
            max_s = max(1, int(self.inventory_estimator.max_slots))
            fill_pct = round(
                (float(self.inventory_estimator.used_slots) / float(max_s)) * 100.0, 2
            )
            return {
                "telemetry_ready": ready,
                # P1-6: kümülatif sayaçlar (kaplı liste `len()` değil).
                "gathered_total": int(self._gathered_total),
                "failed_total": int(self._failed_total),
                "combat_count": int(self._combat_total),
                "alert_history_len": len(self.alert_history),
                "session_elapsed_sec": round(self._elapsed_sec(), 2),
                "inventory_fill_pct": fill_pct,
            }
        except Exception:
            return {"telemetry_ready": False}

# S-0801: beklenen yetenek kumesi — validate_telemetry_contract() yalnizca
# listenin bos olmamasina degil, beklenen yeteneklerin tamam + gercek motor
# probunun gecmesine bakar (behavioral_guard'daki desen).
_EXPECTED_TELEMETRY_CAPABILITIES = (
    "telemetry",
    "metrics_aggregation",
    "diagnostic_snapshot",
)


def get_telemetry_capabilities() -> list:
    return list(_EXPECTED_TELEMETRY_CAPABILITIES)


def _probe_telemetry_engine() -> bool:
    """Gercek prob: TelemetryEngine kurulup canli metrik uretebilmeli.

    Basarisiz olmasi gereken durumlar: motor kurulamiyor, alt izleyiciler
    eksik, degismezler (invariant) bozuk, ya da get_live_metrics() hata
    bayragi (metrics_error) donuyor / semasi bozuk. Her istisna False'dur.
    """
    try:
        engine = TelemetryEngine()
        if engine.heatmap_tracker is None or engine.inventory_estimator is None:
            return False
        if not (
            engine.bucket_span_sec >= 10
            and engine.num_buckets >= 4
            and engine.items_per_gather >= 1
            and 0.0 < _MIN_GLYPH_CONFIDENCE < 1.0
            and _MAX_GATHER_EVENTS > 0
            and _MAX_FAILURE_EVENTS > 0
            and _MAX_COMBAT_EVENTS > 0
            and _MAX_ALERT_HISTORY > 0
            and engine._elapsed_sec() >= 0.0
        ):
            return False
        metrics = engine.get_live_metrics()
        if not isinstance(metrics, dict) or metrics.get("metrics_error", False):
            return False
        required_keys = (
            "session_duration_sec",
            "gathered_total",
            "gathers_per_hour",
            "success_rate_pct",
            "hourly_buckets",
            "max_slots",
            "fill_pct",
        )
        if any(key not in metrics for key in required_keys):
            return False
        return bool(
            float(metrics["session_duration_sec"]) >= 0.0
            and int(metrics["max_slots"]) > 0
            and 0.0 <= float(metrics["fill_pct"]) <= 100.0
            and isinstance(metrics["hourly_buckets"], list)
        )
    except Exception:
        return False


def validate_telemetry_contract() -> bool:
    # S-0801 (Karar 3: proba baglama): sabit 3 elemanli listenin dolu olmasi
    # artik yetmez — beklenen yeteneklerin tamami + gercek motor probu gerekir.
    capabilities = get_telemetry_capabilities()
    return bool(
        capabilities
        and all(cap in capabilities for cap in _EXPECTED_TELEMETRY_CAPABILITIES)
        and _probe_telemetry_engine()
    )


def _telemetry_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama.

    Audit/readiness katmanlari ``validate_telemetry_contract()``'u tekrar
    cagirmaz; zincirin giris noktasi dogrulamayi bir kez yapar ve sonucu
    ``_contract_ok`` ile alt katmanlara aktarir. ``None`` ise dogrulama
    burada bir kez calisir. Cagrilar arasi onbellek YOKTUR: prob sonucu
    her ust duzey okumada taze hesaplanir (testler validate sonucunun
    cagridan cagriya degisebildigini pinler).
    """
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_telemetry_contract())


def get_telemetry_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "telemetry", "contract_valid": _telemetry_contract_ok(_contract_ok), "capability_count": len(get_telemetry_capabilities())}


def get_telemetry_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "ready": ok,
        "metrics": get_telemetry_baseline_metrics(_contract_ok=ok),
    }


def get_telemetry_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "audit_passed": ok,
        "readiness": get_telemetry_readiness_status(_contract_ok=ok),
    }


def get_telemetry_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "operational": ok,
        "audit": get_telemetry_audit_record(_contract_ok=ok),
    }


def get_telemetry_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "telemetry_ready": ok,
        "digest": get_telemetry_operational_digest(_contract_ok=ok),
    }


def get_telemetry_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "sealed": ok,
        "telemetry": get_telemetry_telemetry_payload(_contract_ok=ok),
    }


def get_telemetry_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "certified": ok,
        "seal": get_telemetry_integrity_seal(_contract_ok=ok),
    }


def get_telemetry_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "ready": ok,
        "certificate": get_telemetry_compliance_certificate(_contract_ok=ok),
    }


def get_telemetry_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _telemetry_contract_ok(_contract_ok)
    return {
        "module": "telemetry",
        "audit_passed": ok,
        "probe": get_telemetry_readiness_probe(_contract_ok=ok),
    }

