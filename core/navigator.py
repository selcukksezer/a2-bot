# -*- coding: utf-8 -*-
"""
Map Navigator Module
Opens the map, matches the node target coordinate/patch, clicks to move, and monitors travel.
"""

import time
import random
import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from core.corridor_navigator import (
    DEFAULT_DEADBAND_DEG,
    MAX_DEADBAND_DEG,
    MIN_DEADBAND_DEG,
    MINIMAP_METERS_PER_PX as PIXEL_TO_METER_DEFAULT,
)
from core.input_driver import InputDriver
from core.vision import Vision
from core.route_manager import RouteManager
# P1-7 (K-20): minimap metre/piksel ölçeğinin tek kaynağı.
from core.corridor_navigator import MINIMAP_METERS_PER_PX
from version import __version__

# P3-3: ölü-bölge (6.0/5.0/8.0) ve minimap ölçeği (0.55) tek tanımı
# `core/corridor_navigator.py`dedir; buradaki isimler yeniden ihraç edilir
# (`from core.navigator import DEFAULT_DEADBAND_DEG, ...` çalışmaya devam eder).
# DEFAULT_WAYPOINT_RADIUS_PX/M tek tanımı da buradadır (P3-3 v19);
# `core/navigation_intel.py` bu iki adı buradan içe aktarır.
DEFAULT_WAYPOINT_RADIUS_PX: float = 20.0
DEFAULT_WAYPOINT_RADIUS_M: float = 3.0


def compute_bearing(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
    """
    Calculates compass bearing from from_pos to to_pos in degrees (0.0° .. 360.0°).
    Coordinates: X grows right (East), Y grows down (South).
    North is 0°, East is 90°, South is 180°, West is 270°.
    Eliminates 180° inversion bug between Cartesian (0° = East) and
    Minimap / Compass arrow heading (0° = North).
    Returns float("nan") when both points coincide: no direction exists,
    so no arbitrary bearing (e.g. North) is invented.
    """
    dx = float(to_pos[0]) - float(from_pos[0])
    dy = float(to_pos[1]) - float(from_pos[1])
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return float("nan")
    angle = math.degrees(math.atan2(dx, -dy))
    return round((angle + 360.0) % 360.0, 2)


def compute_angle_diff(target_bearing: float, player_heading: float) -> float:
    """
    Calculates shortest rotational angle difference in range [-180.0°, +180.0°]
    to turn player_heading towards target_bearing.
    Positive (+) indicates clockwise / right turn.
    Negative (-) indicates counter-clockwise / left turn.
    Formula: (target_bearing - player_heading + 180) % 360 - 180
    """
    diff = (float(target_bearing) - float(player_heading) + 180.0) % 360.0 - 180.0
    return round(diff, 2)


def apply_turn_damping(
    angle_diff: float,
    deadband_deg: float = DEFAULT_DEADBAND_DEG,
    min_turn_sec: float = 0.04,
    max_turn_sec: float = 0.35,
    max_error_deg: float = 180.0,
) -> Dict[str, Any]:
    """
    Applies deadband zone (±5° to ±8°) and smoothly interpolates turn duration
    based on angular error to prevent left-right rapid steering oscillation (yalpalanma).
    """
    clamped_deadband = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))
    abs_err = abs(float(angle_diff))
    in_deadband = bool(abs_err <= clamped_deadband)

    if in_deadband:
        return {
            "angle_diff": round(float(angle_diff), 2),
            "abs_error": round(abs_err, 2),
            "in_deadband": True,
            "turn_needed": False,
            "turn_direction": "straight",
            "turn_duration_sec": 0.0,
            "deadband_deg": clamped_deadband,
        }

    direction = "right" if angle_diff > 0.0 else "left"
    span = max(1.0, float(max_error_deg) - clamped_deadband)
    norm = min(1.0, max(0.0, (abs_err - clamped_deadband) / span))
    smooth_t = norm * norm * (3.0 - 2.0 * norm)
    turn_duration = float(min_turn_sec) + smooth_t * (float(max_turn_sec) - float(min_turn_sec))

    return {
        "angle_diff": round(float(angle_diff), 2),
        "abs_error": round(abs_err, 2),
        "in_deadband": False,
        "turn_needed": True,
        "turn_direction": direction,
        "turn_duration_sec": round(turn_duration, 4),
        "deadband_deg": clamped_deadband,
    }


class TurnDampingController:
    """
    Stateful controller that prevents rapid steering oscillation near the target bearing.
    Uses deadband filtering (±5°..±8°) and smooth turn duration interpolation.
    """
    def __init__(
        self,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        min_turn_sec: float = 0.04,
        max_turn_sec: float = 0.35,
        reversal_cooldown_sec: float = 0.12,
    ):
        self.deadband_deg = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))
        self.min_turn_sec = float(min_turn_sec)
        self.max_turn_sec = float(max_turn_sec)
        self.reversal_cooldown_sec = float(reversal_cooldown_sec)
        self.last_turn_direction: Optional[str] = None
        self.last_turn_ts: float = 0.0

    def calculate_turn(
        self,
        target_bearing: float,
        player_heading: float,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        curr_ts = time.time() if now is None else float(now)
        diff = compute_angle_diff(target_bearing, player_heading)
        res = apply_turn_damping(
            angle_diff=diff,
            deadband_deg=self.deadband_deg,
            min_turn_sec=self.min_turn_sec,
            max_turn_sec=self.max_turn_sec,
        )
        if res["turn_needed"]:
            if (
                self.last_turn_direction is not None
                and self.last_turn_direction != res["turn_direction"]
                and (curr_ts - self.last_turn_ts) < self.reversal_cooldown_sec
            ):
                res["turn_duration_sec"] = round(res["turn_duration_sec"] * 0.65, 4)
                res["reversal_damped"] = True
            else:
                res["reversal_damped"] = False
            self.last_turn_direction = res["turn_direction"]
            self.last_turn_ts = curr_ts
        else:
            res["reversal_damped"] = False
        return res


def normalize_compass_heading(heading: float) -> float:
    """Normalizes any heading angle in degrees to [0.0, 360.0) range."""
    return round((float(heading) % 360.0 + 360.0) % 360.0, 2)


def anti_inversion_compass_heading(
    heading: float,
    reference_bearing: Optional[float] = None,
    threshold_deg: float = 150.0,
) -> float:
    """
    Prevents and corrects 180° compass heading anti-inversion bugs.
    If reference_bearing is provided and angular discrepancy between
    heading and reference exceeds threshold_deg (>150°), inverts heading
    by 180° to align with true forward travel direction.
    """
    norm_h = normalize_compass_heading(heading)
    if reference_bearing is None:
        return norm_h
    diff = abs(compute_angle_diff(reference_bearing, norm_h))
    if diff >= float(threshold_deg):
        corrected = normalize_compass_heading(norm_h + 180.0)
        return corrected
    return norm_h


def cartesian_to_compass(cartesian_deg: float) -> float:
    """
    Converts Cartesian angle (0° = East, counter-clockwise)
    to Compass bearing (0° = North, clockwise).
    Formula: (450.0 - cartesian_deg) % 360.0
    """
    return round((450.0 - float(cartesian_deg)) % 360.0, 2)


def compass_to_cartesian(compass_deg: float) -> float:
    """
    Converts Compass bearing (0° = North, clockwise)
    to Cartesian angle (0° = East, counter-clockwise).
    Formula: (450.0 - compass_deg) % 360.0
    """
    return round((450.0 - float(compass_deg)) % 360.0, 2)


def extract_node_xy(node: Any) -> Tuple[float, float]:
    """
    Safely extracts (x, y) coordinates from dict, object, or tuple.
    Raises ValueError when the node carries no usable coordinates instead of
    silently inventing (0, 0) (which would navigate/click at the map origin).
    Callers that scan many nodes must skip nodes raising ValueError.
    """
    try:
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return float(node[0]), float(node[1])
        if isinstance(node, dict):
            if "map_x" in node or "x" in node:
                mx = node.get("map_x", node.get("x"))
                my = node.get("map_y", node.get("y"))
                if mx is None or my is None:
                    raise ValueError(f"node has incomplete coordinates: {node!r}")
                return float(mx), float(my)
            raise ValueError(f"node has no coordinates (map_x/map_y or x/y): {node!r}")
        if hasattr(node, "map_x") or hasattr(node, "x"):
            mx = getattr(node, "map_x", getattr(node, "x", None))
            my = getattr(node, "map_y", getattr(node, "y", None))
            if mx is None or my is None:
                raise ValueError(f"node has incomplete coordinates: {node!r}")
            return float(mx), float(my)
        raise ValueError(f"node has no coordinates: {node!r}")
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise ValueError(f"node coordinates unparseable: {node!r} ({exc})") from exc


def find_nearest_waypoint(
    player_pos: Tuple[float, float],
    nodes: Sequence[Any],
) -> Tuple[int, Optional[Any], float]:
    """
    Finds the index and node closest to player_pos.
    Returns: (nearest_index, nearest_node, distance_px).
    """
    if not nodes:
        return -1, None, float("inf")
    px, py = float(player_pos[0]), float(player_pos[1])
    best_idx = -1
    # Karesel mesafeyle karşılaştır (sqrt monoton: sıralama aynı); tek sqrt
    # sonda yapılır, düğüm başına hypot kalkar.
    best_dist_sq = float("inf")
    best_node = None
    for idx, node in enumerate(nodes):
        try:
            nx, ny = extract_node_xy(node)
        except ValueError:
            # Coordinate-less node: skip it instead of treating it as (0, 0).
            continue
        dx = nx - px
        dy = ny - py
        d_sq = dx * dx + dy * dy
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_idx = idx
            best_node = node
    return best_idx, best_node, round(math.sqrt(best_dist_sq), 2)


def resume_route_after_interrupt(
    player_pos: Tuple[float, float],
    nodes: Sequence[Any],
    last_index: Optional[int] = None,
    waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
    advance_if_reached: bool = False,
) -> Dict[str, Any]:
    """
    Resumes route following after an F10 pause or combat interruption.
    Finds the nearest waypoint to player_pos.
    If advance_if_reached=True and player is already within waypoint_radius_px,
    advances to the next sequential waypoint in cyclical order.
    `last_index` (the waypoint active before the interrupt) is honoured: when the
    player is still within waypoint_radius_px of that node, resume stays on it
    instead of jumping to the geometrically nearest node.
    """
    if not nodes:
        return {
            "resumed_index": -1,
            "active_node": None,
            "distance_px": 0.0,
            "resumed_from": "empty",
            "cyclical": True,
        }
    n = len(nodes)
    try:
        last_idx = int(last_index) if last_index is not None else -1
    except (TypeError, ValueError):
        last_idx = -1
    if 0 <= last_idx < n:
        try:
            lx, ly = extract_node_xy(nodes[last_idx])
            last_dist = math.hypot(lx - float(player_pos[0]), ly - float(player_pos[1]))
            if last_dist <= float(waypoint_radius_px):
                return {
                    "resumed_index": last_idx,
                    "active_node": nodes[last_idx],
                    "distance_px": round(last_dist, 2),
                    "resumed_from": "last_index",
                    "cyclical": True,
                }
        except ValueError:
            pass
    nearest_idx, nearest_node, dist_px = find_nearest_waypoint(player_pos, nodes)
    if nearest_idx < 0:
        return {
            "resumed_index": -1,
            "active_node": None,
            "distance_px": float("inf"),
            "resumed_from": "no_valid_node",
            "cyclical": True,
        }
    if advance_if_reached and dist_px <= float(waypoint_radius_px) and n > 1:
        target_idx = (nearest_idx + 1) % n
        return {
            "resumed_index": target_idx,
            "active_node": nodes[target_idx],
            "distance_px": dist_px,
            "resumed_from": "advance_from_nearest",
            "cyclical": True,
        }
    return {
        "resumed_index": nearest_idx,
        "active_node": nearest_node,
        "distance_px": dist_px,
        "resumed_from": "nearest",
        "cyclical": True,
    }


class WaypointArrivalDetector:
    """
    Waypoint Arrival Detector:
    Verifies arrival thresholds (<=20px in pixel coordinates or <=3.0m in world distance).
    Provides atomic transition to the next sequential waypoint in cyclical 3+ node routes.
    """

    def __init__(
        self,
        waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
        waypoint_radius_m: float = DEFAULT_WAYPOINT_RADIUS_M,
        pixel_to_meter: float = PIXEL_TO_METER_DEFAULT,
    ) -> None:
        self.waypoint_radius_px: float = float(waypoint_radius_px)
        self.waypoint_radius_m: float = float(waypoint_radius_m)
        self.pixel_to_meter: float = float(pixel_to_meter)

    def is_arrival_reached(
        self,
        player_pos: Tuple[float, float],
        waypoint_pos: Tuple[float, float],
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Determines whether the player has arrived at the target waypoint.
        Arrival criteria:
        1. pixel_distance <= waypoint_radius_px (<=20px), OR
        2. distance_m <= waypoint_radius_m (<=3.0m) — only when an explicitly
           measured `distance_m` is supplied. Deriving metres from pixels with
           the same scale would make this check a strict subset of (1) and
           therefore dead; the derived value is reported as an estimate only.
        """
        px = float(player_pos[0])
        py = float(player_pos[1])
        wx = float(waypoint_pos[0])
        wy = float(waypoint_pos[1])

        computed_px_dist = math.hypot(wx - px, wy - py)
        px_dist = float(pixel_distance) if pixel_distance is not None else computed_px_dist

        if distance_m is not None:
            dist_m = float(distance_m)
            dist_m_measured = True
        else:
            # Estimate only — never used for the metre-threshold decision.
            dist_m = px_dist * self.pixel_to_meter
            dist_m_measured = False

        reached_px = bool(px_dist <= self.waypoint_radius_px)
        reached_m = bool(dist_m_measured and dist_m <= self.waypoint_radius_m)
        arrived = bool(reached_px or reached_m)

        return {
            "arrived": arrived,
            "reached_px": reached_px,
            "reached_m": reached_m,
            "pixel_distance": round(px_dist, 2),
            "distance_m": round(dist_m, 2),
            "distance_m_measured": dist_m_measured,
            "threshold_px": self.waypoint_radius_px,
            "threshold_m": self.waypoint_radius_m,
        }

    def check_and_transition(
        self,
        player_pos: Tuple[float, float],
        waypoint_pos: Tuple[float, float],
        nodes: Sequence[Any],
        current_index: int,
        distance_m: Optional[float] = None,
        allow_cyclical: bool = True,
    ) -> Dict[str, Any]:
        """
        Checks arrival at current waypoint. If arrived, transitions cleanly
        to the next sequential node (cyclically wrapping around for 3+ node routes).
        """
        if not nodes:
            return {
                "arrived": False,
                "transitioned": False,
                "current_index": -1,
                "active_node": None,
                "prev_index": current_index,
                "pixel_distance": 0.0,
                "distance_m": 0.0,
                "reached_px": False,
                "reached_m": False,
            }

        n = len(nodes)
        curr_idx = max(0, min(n - 1, int(current_index)))
        arrival = self.is_arrival_reached(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            distance_m=distance_m,
        )

        if arrival["arrived"]:
            if allow_cyclical:
                next_idx = (curr_idx + 1) % n
            else:
                next_idx = min(n - 1, curr_idx + 1)
            # A single-node route has nowhere to transition to: staying on the
            # same node is not a transition (and not a completed lap).
            transitioned = bool(next_idx != curr_idx)
            return {
                "arrived": True,
                "transitioned": transitioned,
                "current_index": next_idx,
                "active_node": nodes[next_idx],
                "prev_index": curr_idx,
                "pixel_distance": arrival["pixel_distance"],
                "distance_m": arrival["distance_m"],
                "reached_px": arrival["reached_px"],
                "reached_m": arrival["reached_m"],
            }

        return {
            "arrived": False,
            "transitioned": False,
            "current_index": curr_idx,
            "active_node": nodes[curr_idx],
            "prev_index": curr_idx,
            "pixel_distance": arrival["pixel_distance"],
            "distance_m": arrival["distance_m"],
            "reached_px": arrival["reached_px"],
            "reached_m": arrival["reached_m"],
        }


class CyclicalRouteFollower:
    """
    Cyclical 3+ Node Sequential Route Follower with Turn Damping & Anti-Inversion.
    Maintains active sequence index, detects waypoint arrival (<=20px / <=3m),
    advances cyclically through 3+ nodes, applies ±6° turn damping, and provides
    instant nearest-waypoint resume after pause/combat interrupt.
    """

    def __init__(
        self,
        nodes: Optional[Sequence[Any]] = None,
        waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
        waypoint_radius_m: float = DEFAULT_WAYPOINT_RADIUS_M,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        start_index: int = 0,
        pixel_to_meter: float = PIXEL_TO_METER_DEFAULT,
    ) -> None:
        self.nodes: List[Any] = list(nodes) if nodes else []
        self.current_index: int = max(0, int(start_index)) if self.nodes else 0
        self.arrival_detector = WaypointArrivalDetector(
            waypoint_radius_px=waypoint_radius_px,
            waypoint_radius_m=waypoint_radius_m,
            pixel_to_meter=pixel_to_meter,
        )
        self.turn_controller = TurnDampingController(deadband_deg=deadband_deg)
        self.total_completed_laps: int = 0
        self.last_transition_ts: float = 0.0

    def set_nodes(self, nodes: Sequence[Any], reset_index: bool = False) -> None:
        """Sets route nodes and optionally resets current_index to 0."""
        self.nodes = list(nodes)
        if reset_index or not self.nodes:
            self.current_index = 0
        else:
            self.current_index = self.current_index % len(self.nodes)

    def get_current_waypoint(self) -> Optional[Any]:
        """Returns the currently targeted waypoint node."""
        if not self.nodes:
            return None
        return self.nodes[self.current_index % len(self.nodes)]

    def resume_after_interrupt(self, player_pos: Tuple[float, float]) -> Dict[str, Any]:
        """
        Resumes route traversal after pause or combat interrupt by finding
        the nearest waypoint along the cyclical route.
        """
        res = resume_route_after_interrupt(
            player_pos=player_pos,
            nodes=self.nodes,
            last_index=self.current_index,
            waypoint_radius_px=self.arrival_detector.waypoint_radius_px,
        )
        if res["resumed_index"] >= 0:
            self.current_index = res["resumed_index"]
        return res

    def step(
        self,
        player_pos: Tuple[float, float],
        player_heading: float,
        distance_m: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes one navigation cycle:
        1. Checks waypoint arrival (<=20px / <=3m)
        2. Advances sequence index cyclically (e.g. 0 -> 1 -> 2 -> 0)
        3. Computes compass bearing with anti-inversion
        4. Applies ±6° turn damping and reversal smoothing
        Returns comprehensive navigation state.
        """
        if not self.nodes:
            return {
                "active": False,
                "arrived": False,
                "transitioned": False,
                "current_index": -1,
                "active_node": None,
                "turn_needed": False,
                "in_deadband": True,
            }

        n = len(self.nodes)
        curr_node = self.nodes[self.current_index % n]
        try:
            waypoint_xy = extract_node_xy(curr_node)
        except ValueError as exc:
            return {
                "active": False,
                "arrived": False,
                "transitioned": False,
                "current_index": self.current_index,
                "active_node": curr_node,
                "turn_needed": False,
                "in_deadband": True,
                "error": f"current waypoint has no coordinates: {exc}",
            }

        # 1. Arrival & Cyclical Transition Check
        arrival = self.arrival_detector.check_and_transition(
            player_pos=player_pos,
            waypoint_pos=waypoint_xy,
            nodes=self.nodes,
            current_index=self.current_index,
            distance_m=distance_m,
            allow_cyclical=True,
        )

        curr_ts = time.time() if now is None else float(now)
        transitioned = arrival["transitioned"]
        if transitioned:
            prev_idx = self.current_index
            self.current_index = arrival["current_index"]
            self.last_transition_ts = curr_ts
            if prev_idx == n - 1 and self.current_index == 0:
                self.total_completed_laps += 1
            curr_node = self.nodes[self.current_index]
            waypoint_xy = extract_node_xy(curr_node)

        # 2. Bearing & Anti-Inversion Heading
        target_bearing = compute_bearing(player_pos, waypoint_xy)
        if not math.isfinite(target_bearing):
            # Player is exactly on the waypoint: no direction exists, so hold
            # the current heading instead of steering toward an invented one.
            target_bearing = normalize_compass_heading(player_heading)
        cleaned_heading = anti_inversion_compass_heading(
            player_heading, reference_bearing=target_bearing
        )

        # 3. Turn Damping
        turn_data = self.turn_controller.calculate_turn(
            target_bearing=target_bearing,
            player_heading=cleaned_heading,
            now=curr_ts,
        )

        return {
            "active": True,
            "arrived": arrival["arrived"],
            "transitioned": transitioned,
            "current_index": self.current_index,
            "active_node": curr_node,
            "total_nodes": n,
            "target_bearing": target_bearing,
            "player_heading": cleaned_heading,
            "angle_diff": turn_data["angle_diff"],
            "in_deadband": turn_data["in_deadband"],
            "turn_needed": turn_data["turn_needed"],
            "turn_direction": turn_data["turn_direction"],
            "turn_duration_sec": turn_data["turn_duration_sec"],
            "reversal_damped": turn_data["reversal_damped"],
            "pixel_distance": arrival["pixel_distance"],
            "distance_m": arrival["distance_m"],
            "total_completed_laps": self.total_completed_laps,
            "closed_loop": n >= 3,
        }


class MapNavigator:
    def __init__(self, route_manager: RouteManager, vision: Vision, driver: InputDriver):
        self.routes = route_manager
        self.vision = vision
        self.driver = driver
        self.arrival_detector = WaypointArrivalDetector(
            waypoint_radius_px=DEFAULT_WAYPOINT_RADIUS_PX,
            waypoint_radius_m=DEFAULT_WAYPOINT_RADIUS_M,
        )
        self.turn_controller = TurnDampingController(deadband_deg=DEFAULT_DEADBAND_DEG)
        # Cached cyclical follower for follow_cyclical_step(): state (index,
        # laps, turn controller) must survive across calls instead of being
        # rebuilt from scratch on every step.
        self._cyclical_follower: Optional[CyclicalRouteFollower] = None
        self._cyclical_follower_nodes: List[Any] = []

    def is_map_open(self, threshold=0.80) -> Optional[bool]:
        """
        Detects if the in-game world map (M) is currently visible on screen.
        Delegates to VisionEngine.is_map_open when available for multi-zone accuracy.

        Üç durumlu (H-sweep ek; G grubu vision.is_map_open'u üç durumlu
        yaptı): True = açık, False = kapalı, None = bilinmiyor. Vision
        açıkça None döndürürse bool() ile False'a indirilmez, None olduğu
        gibi yayılır — bilinmiyor durumunda kör M basmak açık haritayı
        kapatır (fail-open). Vision motoru yoksa da doğrulanamadığı için
        None döner. Yerel sezgisel yedek yalnızca vision'da is_map_open
        yoksa / çağrı patlarsa devreye girer; vision'ın None'u otorite
        sayılır.
        """
        if self.vision is None:
            return None
        if hasattr(self.vision, "is_map_open"):
            try:
                result = self.vision.is_map_open()
                if result is None:
                    return None
                return bool(result)
            except Exception:
                pass
        try:
            screen_bgr = self.vision.grab_screen(grayscale=False)
        except Exception:
            return False
        if screen_bgr is None or getattr(screen_bgr, "size", 0) == 0:
            return False
        try:
            hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        except Exception:
            return False
        h, w, _ = screen_bgr.shape
        # Fallback heuristic needs two agreeing signals, not saturation alone:
        # a desaturated centre AND a lit map canvas (a dark/loading screen is
        # desaturated too, but it is not the open map).
        center_hsv = hsv[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        center_sat = float(np.mean(center_hsv[:, :, 1]))
        center_val = float(np.mean(center_hsv[:, :, 2]))
        return bool(center_sat < 65 and center_val > 40)

    def open_map(self, max_wait=2.0):
        """Opens map with M key and waits until visible.

        Üç durumlu: durum bilinmiyorsa (None) kör M basılmaz — harita
        zaten açıksa M kapatır. None yayılır; çağıran başarısız sayar
        (örn. navigate_to_node hedefe tıklamaz).
        """
        state = self.is_map_open()
        if state is False:
            self.driver.toggle_map()
            start = time.time()
            while time.time() - start < max_wait:
                time.sleep(0.1)
                if self.is_map_open() is True:
                    time.sleep(0.15) # Wait for animation
                    return True
            return self.is_map_open()
        return state

    def close_map(self, max_wait=2.0):
        """Closes map with M key and waits until closed.

        Üç durumlu: durum bilinmiyorsa (None) kör M basılmaz — harita
        zaten kapalıysa M açar. None yayılır (eskiden yanlışlıkla
        True döner, 'kapalı' sanılırdı).
        """
        state = self.is_map_open()
        if state is True:
            self.driver.toggle_map()
            start = time.time()
            while time.time() - start < max_wait:
                time.sleep(0.1)
                if self.is_map_open() is False:
                    time.sleep(0.1)
                    return True
            final = self.is_map_open()
            return None if final is None else (not final)
        if state is False:
            return True
        return None

    def get_navigator_status_summary(self) -> dict:
        """Returns navigator status summary from real component state."""
        ready = self.routes is not None and self.vision is not None and self.driver is not None
        return {
            "navigator_ready": bool(ready),
            "routes": self.routes is not None,
            "vision": self.vision is not None,
            "driver": self.driver is not None,
        }

    def navigate_to_node(self, node: dict):
        """
        Executes navigation to a specified node:
        1. Opens map (M)
        2. Locates target point (direct coordinate or patch matchTemplate)
        3. Clicks target
        4. Closes map (M)
        5. Returns status
        The map is always closed again (try/finally), even when an intermediate
        step fails. A node without coordinates and without a usable patch match
        is rejected (False) instead of clicking at (None, None) / (0, 0).

        P1-10 (K-14): eski `timeout_sec` parametresi ölüydü — bu metot
        tıklamadan sonra dönüyor, seyahati hiç izlemiyor, zaman aşımını
        hiç uygulamıyordu; parametre yalnızca son print'te geçiyordu.
        Seyahat izleme + zaman aşımı zaten çağıranda yaşıyor
        (`modules/route_runner.py` WALKING durumu: toplanma istemi =
        varış, `walk_timeout_sec` = zaman aşımı) — bu metot içinde
        ikinci bir izleme döngüsü ESC/duraklatma denetimini
        engelleyeceği için parametre kaldırıldı, izleme çağıranda kaldı.
        """
        name = node.get("name", "?") if isinstance(node, dict) else "?"
        print(f"\n[Navigator] '{name}' konumuna rota başlatılıyor...")

        # 1. Open Map
        if not self.open_map():
            print("[Navigator] Hata: Harita açılamadı!")
            return False

        try:
            # 2. Determine Click Target
            target_x = node.get("map_x")
            target_y = node.get("map_y")

            # Optional: refine coordinate via patch matching if patch exists
            patch_file = node.get("patch_file")
            if patch_file:
                patch_path = os.path.join(self.routes.patch_dir, patch_file)
                if not os.path.exists(patch_path):
                    print(f"[Navigator] Uyarı: Harita yaması bulunamadı: {patch_path}; kayıtlı koordinat kullanılıyor.")
                else:
                    patch_gray = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                    if patch_gray is None or patch_gray.size == 0:
                        print(f"[Navigator] Hata: Harita yaması okunamadı: {patch_path}")
                        return False
                    map_gray = self.vision.grab_screen(grayscale=True)
                    res = cv2.matchTemplate(map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.70:
                        ph, pw = patch_gray.shape
                        target_x = max_loc[0] + pw // 2
                        target_y = max_loc[1] + ph // 2
                        print(f"[Navigator] Harita şablon eşleşmesi: %{int(max_val*100)} -> ({target_x}, {target_y})")
                    else:
                        print(
                            f"[Navigator] Uyarı: Yama eşleşmesi zayıf ({float(max_val):.2f} < 0.70); "
                            f"kayıtlı koordinat kullanılıyor: ({target_x}, {target_y})"
                        )

            if target_x is None or target_y is None:
                print("[Navigator] Hata: Düğümde koordinat yok (map_x/map_y) ve yama eşleşmesi yok; tıklama yapılmadı.")
                return False
            try:
                target_x = float(target_x)
                target_y = float(target_y)
            except (TypeError, ValueError):
                print(f"[Navigator] Hata: Düğüm koordinatı sayısal değil: ({target_x!r}, {target_y!r}); tıklama yapılmadı.")
                return False
            if not (math.isfinite(target_x) and math.isfinite(target_y)):
                print("[Navigator] Hata: Düğüm koordinatı sonlu değil; tıklama yapılmadı.")
                return False

            # S215 (v19/P3-8): yalnızca sonluluk denetleniyordu — ekran
            # dışı koordinat (negatif / ekranı aşan) kör tıklamaya
            # gidiyordu. Ekran boyutu vision'dan okunur; çözülemezse
            # denetim atlanır (eski davranış), çözülürse sınır dışı
            # hedef reddedilir (kenara kırpılıp yanlış yere tıklanmaz).
            screen_w = getattr(self.vision, "screen_w", None)
            screen_h = getattr(self.vision, "screen_h", None)
            if not (isinstance(screen_w, (int, float)) and isinstance(screen_h, (int, float)) and screen_w > 0 and screen_h > 0):
                try:
                    _probe = self.vision.grab_screen(grayscale=True)
                    if _probe is not None and getattr(_probe, "size", 0) > 0:
                        screen_h, screen_w = _probe.shape[:2]
                except Exception:
                    screen_w = screen_h = None
            if screen_w is not None and screen_h is not None:
                if not (0 <= target_x < float(screen_w) and 0 <= target_y < float(screen_h)):
                    print(f"[Navigator] Hata: Düğüm koordinatı ekran dışında: ({target_x}, {target_y}) / ekran {int(screen_w)}x{int(screen_h)}; tıklama yapılmadı.")
                    return False

            # 3. Click target on map
            time.sleep(random.uniform(0.15, 0.3))
            self.driver.human_click(target_x, target_y, button='left')
            time.sleep(random.uniform(0.2, 0.4))
        finally:
            # 4. Close Map — always, even when a step above failed.
            self.close_map()

        print("[Navigator] Hedefe tıklandı, yürüyüş başladı (seyahat izleme + zaman aşımı route_runner WALKING durumunda)")
        return True

    def compute_bearing(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> float:
        """Calculates compass bearing (0°=North, 90°=East) from current_pos to target_pos."""
        return compute_bearing(current_pos, target_pos)

    def compute_angle_diff(self, target_bearing: float, player_heading: float) -> float:
        """Calculates shortest rotational angle diff [-180°..+180°] to reach target_bearing."""
        return compute_angle_diff(target_bearing, player_heading)

    def apply_turn_damping(
        self,
        angle_diff: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        min_turn_sec: float = 0.04,
        max_turn_sec: float = 0.35,
    ) -> Dict[str, Any]:
        """Applies deadband zone (±5° to ±8°) and smooth turn duration interpolation."""
        return apply_turn_damping(
            angle_diff=angle_diff,
            deadband_deg=deadband_deg,
            min_turn_sec=min_turn_sec,
            max_turn_sec=max_turn_sec,
        )

    def calculate_turn(
        self,
        player_heading: float,
        target_bearing: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
    ) -> Dict[str, Any]:
        """Calculates damped steering turn command towards target bearing."""
        diff = compute_angle_diff(target_bearing, player_heading)
        return apply_turn_damping(diff, deadband_deg=deadband_deg)

    def steer_step(
        self,
        current_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        player_heading: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
    ) -> Dict[str, Any]:
        """
        Executes one steering calculation step:
        1. Calculates target bearing via compass coords (North=0°, East=90°)
        2. Calculates shortest rotational angle difference
        3. Applies turn damping and deadband (±5° to ±8°)
        """
        bearing = compute_bearing(current_pos, target_pos)
        if not math.isfinite(bearing):
            # Coincident points have no direction; hold the current heading.
            bearing = normalize_compass_heading(player_heading)
        turn_data = self.calculate_turn(
            player_heading=player_heading,
            target_bearing=bearing,
            deadband_deg=deadband_deg,
        )
        return {
            "target_bearing": bearing,
            "player_heading": round(float(player_heading), 2),
            **turn_data,
        }

    def is_arrival_reached(
        self,
        player_pos: Tuple[float, float],
        waypoint_pos: Tuple[float, float],
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Checks if player has arrived at waypoint (<=20px / <=3m)."""
        return self.arrival_detector.is_arrival_reached(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            distance_m=distance_m,
            pixel_distance=pixel_distance,
        )

    def check_and_transition(
        self,
        player_pos: Tuple[float, float],
        waypoint_pos: Tuple[float, float],
        nodes: Sequence[Any],
        current_index: int,
        distance_m: Optional[float] = None,
        allow_cyclical: bool = True,
    ) -> Dict[str, Any]:
        """Checks waypoint arrival and transitions cyclically along sequence."""
        return self.arrival_detector.check_and_transition(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            nodes=nodes,
            current_index=current_index,
            distance_m=distance_m,
            allow_cyclical=allow_cyclical,
        )

    def find_nearest_waypoint(
        self,
        player_pos: Tuple[float, float],
        nodes: Sequence[Any],
    ) -> Tuple[int, Optional[Any], float]:
        """Finds closest waypoint in sequence to player_pos."""
        return find_nearest_waypoint(player_pos, nodes)

    def resume_after_interrupt(
        self,
        player_pos: Tuple[float, float],
        nodes: Sequence[Any],
        last_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resumes route following after F10 pause or combat interrupt."""
        return resume_route_after_interrupt(
            player_pos=player_pos,
            nodes=nodes,
            last_index=last_index,
            waypoint_radius_px=self.arrival_detector.waypoint_radius_px,
        )

    def create_cyclical_follower(
        self,
        nodes: Optional[Sequence[Any]] = None,
        start_index: int = 0,
    ) -> CyclicalRouteFollower:
        """Instantiates a dedicated CyclicalRouteFollower for cyclical 3+ node traversal."""
        return CyclicalRouteFollower(
            nodes=nodes,
            waypoint_radius_px=self.arrival_detector.waypoint_radius_px,
            waypoint_radius_m=self.arrival_detector.waypoint_radius_m,
            deadband_deg=self.turn_controller.deadband_deg,
            start_index=start_index,
        )

    def follow_cyclical_step(
        self,
        player_pos: Tuple[float, float],
        player_heading: float,
        nodes: Sequence[Any],
        current_index: int,
        distance_m: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single step of cyclical route navigation:
        arrival check (<=20px / <=3m) -> sequence transition -> anti-inversion bearing -> ±6° turn damping.
        Reuses one cached CyclicalRouteFollower per node list so lap count,
        current index and turn-damping state survive across calls; the follower
        is only rebuilt when the node list changes (current_index seeds it then).
        """
        nodes_list = list(nodes)
        if (
            self._cyclical_follower is None
            or self._cyclical_follower_nodes != nodes_list
        ):
            self._cyclical_follower = self.create_cyclical_follower(
                nodes=nodes_list, start_index=current_index
            )
            self._cyclical_follower_nodes = nodes_list
        return self._cyclical_follower.step(
            player_pos=player_pos,
            player_heading=player_heading,
            distance_m=distance_m,
            now=now,
        )

    def reset_cyclical_follower(self) -> None:
        """Drops the cached follower so the next follow_cyclical_step() rebuilds it."""
        self._cyclical_follower = None
        self._cyclical_follower_nodes = []

    def _components_ready(self) -> bool:
        """True when all three collaborators the navigator needs are present."""
        return self.routes is not None and self.vision is not None and self.driver is not None

    def get_navigator_health_summary(self) -> dict:
        ready = self._components_ready()
        return {"module": "navigator", "ready": ready, "status": "ok" if ready else "degraded"}

    def get_navigator_ready_flag(self) -> dict:
        return {"module": "navigator", "ready": self._components_ready()}

    def get_navigator_version_info(self) -> dict:
        return {"module": "navigator", "version": __version__, "ready": self._components_ready()}

    def get_navigator_diagnostic_snapshot(self) -> dict:
        ready = self._components_ready()
        return {"module": "navigator", "status": "ok" if ready else "degraded", "ready": ready}


def get_navigator_capabilities() -> list:
    return ["navigator", "waypoint_traversal", "heading_alignment", "turn_damping", "diagnostic_snapshot"]


def validate_navigator_contract() -> bool:
    """
    Real contract check (not an unconditional True): capabilities are present,
    bearing math follows the compass convention, and the arrival detector
    honours both the pixel and the measured-metre thresholds.
    """
    try:
        if not get_navigator_capabilities():
            return False
        if compute_bearing((0.0, 0.0), (0.0, -1.0)) != 0.0:
            return False
        if compute_bearing((0.0, 0.0), (1.0, 0.0)) != 90.0:
            return False
        if math.isfinite(compute_bearing((1.0, 1.0), (1.0, 1.0))):
            return False
        det = WaypointArrivalDetector()
        if not det.is_arrival_reached((100.0, 100.0), (105.0, 100.0))["arrived"]:
            return False
        if not det.is_arrival_reached((0.0, 0.0), (50.0, 50.0), distance_m=2.0)["reached_m"]:
            return False
        if det.is_arrival_reached((0.0, 0.0), (50.0, 50.0), distance_m=9.0)["arrived"]:
            return False
        return True
    except Exception:
        return False


def get_navigator_baseline_metrics() -> dict:
    return {"module": "navigator", "contract_valid": validate_navigator_contract(), "capability_count": len(get_navigator_capabilities())}


def get_navigator_readiness_status() -> dict:
    return {"module": "navigator", "ready": validate_navigator_contract(), "metrics": get_navigator_baseline_metrics()}


def get_navigator_audit_record() -> dict:
    return {"module": "navigator", "audit_passed": validate_navigator_contract(), "readiness": get_navigator_readiness_status()}


def get_navigator_operational_digest() -> dict:
    return {"module": "navigator", "operational": validate_navigator_contract(), "audit": get_navigator_audit_record()}


def get_navigator_telemetry_payload() -> dict:
    return {"module": "navigator", "payload": get_navigator_operational_digest()}


def get_navigator_integrity_seal() -> dict:
    return {"module": "navigator", "sealed": validate_navigator_contract(), "telemetry": get_navigator_telemetry_payload()}


def get_navigator_compliance_certificate() -> dict:
    return {"module": "navigator", "certified": validate_navigator_contract(), "seal": get_navigator_integrity_seal()}


def get_navigator_readiness_probe() -> dict:
    return {"module": "navigator", "ready": validate_navigator_contract(), "certificate": get_navigator_compliance_certificate()}


def get_navigator_audit_summary() -> dict:
    return {"module": "navigator", "audit_passed": validate_navigator_contract(), "probe": get_navigator_readiness_probe()}


