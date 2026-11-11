# -*- coding: utf-8 -*-
"""
Combat & Defense Module (core/combat.py)
AION 2 Gatherion Pixel Bot — Agent C9

Features:
1. Combat Engagement Ranges (Melee vs Ranged vs Hybrid):
   - Configurable engagement profiles (melee, ranged, hybrid)
   - Dynamic target distance / bounding box scale evaluation
   - Range status estimation (in_range, too_far, too_close)
   - Active gap-closing (W) for melee and tactical kiting/backpedaling (S) for ranged
2. Line-of-Sight (LoS) Check Against Obstacles:
   - Geometric ray-casting from player anchor to target bounding box
   - Fast 2D segment-to-box intersection against detected obstacle colliders
   - CV corridor analysis: edge density and barrier obstruction along sightline
   - Empirical damage stagnation detection: flags LoS obstruction when N consecutive
     skill casts produce zero target HP drop despite target frame being active
3. Target Obstruction Recovery:
   - Multi-phase recovery state machine:
     * Phase 1: Lateral strafe (A or D) determined by obstacle centroid offset
     * Phase 2: Diagonal flanking maneuver (A+W or D+W) to round obstacle corners
     * Phase 3: Obstacle clearing hop / jump (Space) over low terrain bumps
     * Phase 4: Backpedal-flank (S then A/D) to break edge clipping
     * Phase 5: Soft-aim / camera re-center and TAB cycle if target remains unreachable
   - Hardware key dispatch via InputDriver with humanized durations
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from version import __version__

# F-0540 tek kaynak: HP/MP iksir eşikleri config/schema.py sabitlerinden
# okunur (sözleşme no 10: HP 65 / MP 45). config.schema stdlib-only'dir.
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT

try:
    from core.input_driver import InputDriver
except ImportError:
    InputDriver = Any  # type: ignore

try:
    from core.vision import Vision
except ImportError:
    Vision = Any  # type: ignore

logger = logging.getLogger("CombatSystem")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# R06: settle delay before the post-cast HP re-read on the first cast after a
# (re)target. Damage takes longer to reach the HP bar than a key press takes
# to register, so 0.30s is deliberately above the largest in-file key-wait
# precedent (0.18s in fight_back_step, 0.08s/0.10s elsewhere) — a round,
# conservative value. It is paid only once per target (first cast), never
# on every step.
POSTCAST_HP_SETTLE_SEC = 0.30


# =============================================================================
# Enums and Data Models
# =============================================================================

class EngagementProfile(str, Enum):
    """Combat class profile defining desired combat range and positioning tactics."""
    MELEE = "melee"      # Gladiator, Templar, Assassin: close quarters, aggressive gap-closing
    RANGED = "ranged"    # Sorcerer, Ranger, Spiritmaster, Cleric: standoff range, kiting
    HYBRID = "hybrid"    # Chanter / adaptive: flexible engagement


class RangeStatus(str, Enum):
    """Evaluation of target distance relative to active profile."""
    IN_RANGE = "in_range"      # Target is within optimal attack range
    TOO_FAR = "too_far"        # Target is beyond effective range (needs gap-close)
    TOO_CLOSE = "too_close"    # Target is inside minimum safety zone (ranged needs kiting)
    UNKNOWN = "unknown"        # Target distance cannot be determined


class ObstructionType(str, Enum):
    """Reason why line of sight is obstructed."""
    NONE = "none"
    GEOMETRIC_OBSTACLE = "geometric_obstacle"  # Bounding box of obstacle intersects sightline
    CORRIDOR_OCCLUSION = "corridor_occlusion"  # High CV edge/barrier density along ray corridor
    DAMAGE_STAGNATION = "damage_stagnation"    # N consecutive casts dealt 0 target damage
    UI_PROMPT = "ui_prompt"                    # In-game error prompt (e.g. "Target obstructed")


class RecoveryTactic(str, Enum):
    """Tactical maneuver chosen to recover line of sight."""
    NONE = "none"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    FLANK_LEFT = "flank_left"
    FLANK_RIGHT = "flank_right"
    JUMP_FORWARD = "jump_forward"
    BACKPEDAL_FLANK = "backpedal_flank"
    CAMERA_RECENTER = "camera_recenter"
    TAB_RESET = "tab_reset"


@dataclass
class CombatPositioningConfig:
    """Tunable parameters for engagement ranges, LoS checks, and recovery."""
    profile: EngagementProfile = EngagementProfile.MELEE
    
    # Melee range thresholds (LEGACY — P1-12 / v18).
    # These normalized bbox-height bounds are NO LONGER consulted by
    # evaluate_target_range(): the bbox branch now converts norm_h to meters
    # with the single pinhole formula (PINHOLE_K_M, see below) and compares
    # against the meter limits in _PROFILE_RANGE_LIMITS, so both branches
    # decide from one formula and one set of limits. The old bounds were
    # provably inconsistent with those limits — implied pinhole constants
    # melee 0.42 / ranged 0.84..0.96 / hybrid 1.75..1.80, i.e. no single
    # formula could have produced them (hybrid alone matched the old 1.8).
    # The fields stay (with their old defaults) only so existing
    # CombatPositioningConfig(...) construction keeps working; a follow-up
    # may remove them together with the tests that pass custom norm_h.
    # Larger bbox height = target is closer.
    melee_min_norm_h: float = 0.11       # Below 11% screen height -> too far for melee
    melee_max_norm_h: float = 0.70       # Above 70% screen height -> right on top of player

    # Ranged range thresholds (LEGACY — see above; no longer consulted)
    ranged_min_norm_h: float = 0.035     # Below 3.5% screen height -> > 25m, too far
    ranged_max_norm_h: float = 0.160     # Above 16% screen height -> < 6m, danger zone (kite)

    # Hybrid thresholds (LEGACY — see above; no longer consulted)
    hybrid_min_norm_h: float = 0.070
    hybrid_max_norm_h: float = 0.450
    
    # Movement pacing (seconds)
    gap_close_step_sec: float = 0.28
    kite_step_sec: float = 0.24
    strafe_step_sec: float = 0.26
    flank_step_sec: float = 0.34
    backpedal_step_sec: float = 0.28
    
    # LoS and Stagnation thresholds
    corridor_width_px: int = 40
    corridor_occlusion_threshold: float = 0.20
    damage_stagnation_threshold: int = 3   # 3 consecutive casts with 0 HP loss triggers LoS recovery
    max_recovery_attempts: int = 5         # After 5 failed recovery steps, cycle tab / reset
    
    # Default camera / anchor offsets
    player_anchor_x_ratio: float = 0.50   # Player character screen center X
    player_anchor_y_ratio: float = 0.72   # Player character screen center Y

    # Minimum interval between skill casts in fight_back_step (seconds).
    # Same value as the previously hard-coded 0.28s GCD; tunable per profile.
    skill_gcd_sec: float = 0.28


@dataclass
class RangeEvaluation:
    status: RangeStatus
    profile: EngagementProfile
    target_norm_h: float = 0.0
    estimated_distance_m: float = 0.0
    recommended_action: str = "none"      # "step_forward", "backpedal", "none"
    reason: str = ""


@dataclass
class LineOfSightResult:
    has_los: bool
    obstruction_type: ObstructionType = ObstructionType.NONE
    obstruction_score: float = 0.0        # 0.0 clear -> 1.0 completely blocked
    obstacle_center: Optional[Tuple[int, int]] = None
    flank_direction: str = "left"         # "left" or "right"
    reason: str = ""


@dataclass
class RecoveryActionResult:
    tactic: RecoveryTactic
    keys_dispatched: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    attempt_number: int = 1
    target_switched: bool = False
    description: str = ""


def _press_key_sent(result: Any) -> bool:
    """True only when a press_key call actually dispatched a key.

    InputDriver.press_key returns the hold duration (> 0.0) on success and
    0.0 when nothing was sent (game focus refused / motion aborted / no
    trusted path). A None return (legacy duck-typed drivers that report
    nothing) is treated as sent; a bool return is honoured as-is. Every
    keys_dispatched / looted / total_attacks claim in this module must go
    through here — never assume a press_key call succeeded (S-1302).
    """
    if result is None:
        return True
    try:
        return float(result) > 0.0
    except (TypeError, ValueError):
        return bool(result)


# =============================================================================
# Geometry & CV Line-of-Sight Helpers
# =============================================================================

def segment_intersects_rect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> bool:
    """
    Determines if line segment p1-p2 intersects axis-aligned bounding box rect=(x, y, w, h).
    Uses Liang-Barsky / Slab clipping test.
    """
    rx, ry, rw, rh = rect
    rx2, ry2 = rx + rw, ry + rh
    x1, y1 = p1
    x2, y2 = p2
    
    # Quick reject if both points are strictly on one side of rectangle
    if max(x1, x2) < rx or min(x1, x2) > rx2:
        return False
    if max(y1, y2) < ry or min(y1, y2) > ry2:
        return False

    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - rx, rx2 - x1, y1 - ry, ry2 - y1]

    u1 = 0.0
    u2 = 1.0

    for i in range(4):
        if abs(p[i]) < 1e-9:
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u1:
                    u1 = t
            else:
                if t < u2:
                    u2 = t
            if u1 > u2:
                return False

    return True


def analyze_corridor_occlusion(
    frame_bgr: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    corridor_width: int = 40,
    threshold: float = 0.20,
) -> Tuple[bool, float, Optional[Tuple[int, int]], str]:
    """
    Analyzes visual occlusion along the corridor strip between player anchor (p1)
    and target (p2) using edge gradient density and dark obstacle contour detection.
    
    Returns:
        (is_obstructed, occlusion_score, obstacle_center, flank_direction)
    """
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return False, 0.0, None, "left"

    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, min(w - 1, int(p1[0]))), max(0, min(h - 1, int(p1[1])))
    x2, y2 = max(0, min(w - 1, int(p2[0]))), max(0, min(h - 1, int(p2[1])))

    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 20:
        return False, 0.0, None, "left"

    # Direction vector & normal
    vx, vy = (x2 - x1) / dist, (y2 - y1) / dist
    nx, ny = -vy, vx
    half_w = corridor_width / 2.0

    # Sample bounding rectangle around corridor to avoid whole-frame overhead
    min_x = max(0, int(min(x1, x2) - half_w - 5))
    max_x = min(w, int(max(x1, x2) + half_w + 5))
    min_y = max(0, int(min(y1, y2) - half_w - 5))
    max_y = min(h, int(max(y1, y2) + half_w + 5))

    if max_x <= min_x or max_y <= min_y:
        return False, 0.0, None, "left"

    sub_img = frame_bgr[min_y:max_y, min_x:max_x]
    sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    
    # Corridor mask in local sub-image coordinates
    mask = np.zeros(sub_gray.shape, dtype=np.uint8)
    poly = np.array([
        [int(x1 + nx * half_w - min_x), int(y1 + ny * half_w - min_y)],
        [int(x2 + nx * half_w - min_x), int(y2 + ny * half_w - min_y)],
        [int(x2 - nx * half_w - min_x), int(y2 - ny * half_w - min_y)],
        [int(x1 - nx * half_w - min_x), int(y1 - ny * half_w - min_y)],
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, poly, 255)

    corridor_pixels = cv2.countNonZero(mask)
    if corridor_pixels < 25:
        return False, 0.0, None, "left"

    # Edge detection inside corridor (excluding endpoints where player/target silhouettes live)
    # Trim 15% from both ends to avoid self/target boundary edges
    t_start = 0.18
    t_end = 0.82
    mid_poly = np.array([
        [int(x1 + vx * dist * t_start + nx * half_w - min_x), int(y1 + vy * dist * t_start + ny * half_w - min_y)],
        [int(x1 + vx * dist * t_end + nx * half_w - min_x), int(y1 + vy * dist * t_end + ny * half_w - min_y)],
        [int(x1 + vx * dist * t_end - nx * half_w - min_x), int(y1 + vy * dist * t_end - ny * half_w - min_y)],
        [int(x1 + vx * dist * t_start - nx * half_w - min_x), int(y1 + vy * dist * t_start - ny * half_w - min_y)],
    ], dtype=np.int32)

    mid_mask = np.zeros(sub_gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mid_mask, mid_poly, 255)
    mid_pixels = max(1, cv2.countNonZero(mid_mask))

    # Only the trimmed middle corridor (mid_mask) is ever counted, so run
    # Canny and the HSV barrier mask on its bounding box (+3 px pad) instead
    # of the whole sub-image. The pad keeps every counted pixel's 3x3
    # neighbourhood identical, so edge/barrier counts are unchanged.
    _PAD = 3
    _mx0 = max(0, int(np.min(mid_poly[:, 0])) - _PAD)
    _my0 = max(0, int(np.min(mid_poly[:, 1])) - _PAD)
    _mx1 = min(sub_gray.shape[1], int(np.max(mid_poly[:, 0])) + _PAD)
    _my1 = min(sub_gray.shape[0], int(np.max(mid_poly[:, 1])) + _PAD)
    if _mx1 > _mx0 and _my1 > _my0:
        gray_work = sub_gray[_my0:_my1, _mx0:_mx1]
        img_work = sub_img[_my0:_my1, _mx0:_mx1]
        mid_work = mid_mask[_my0:_my1, _mx0:_mx1]
        crop_ox, crop_oy = _mx0, _my0
    else:  # Degenerate corridor: fall back to the full sub-image path.
        gray_work, img_work, mid_work = sub_gray, sub_img, mid_mask
        crop_ox, crop_oy = 0, 0

    edges = cv2.Canny(gray_work, 50, 150)
    occlusion_edges = cv2.bitwise_and(edges, edges, mask=mid_work)
    edge_count = cv2.countNonZero(occlusion_edges)
    edge_density = float(edge_count) / float(mid_pixels)

    # Dark impassable rock/wall mask: low brightness and high saturation or high texture contrast
    hsv_sub = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    barrier_mask = (hsv_sub[:, :, 2] < 60) | ((hsv_sub[:, :, 1] < 45) & (hsv_sub[:, :, 2] < 80))
    barrier_corridor = cv2.bitwise_and(barrier_mask.astype(np.uint8) * 255, mid_work)
    barrier_ratio = float(cv2.countNonZero(barrier_corridor)) / float(mid_pixels)

    # An actual physical 3D obstacle (rock, tree, wall) has edge boundaries.
    # Uniform flat regions with zero edges are ambient terrain or night lighting, not blocking obstacles.
    if edge_density < 0.025:
        combined_score = edge_density * 1.2
    else:
        combined_score = min(1.0, (edge_density * 2.5) + (barrier_ratio * 0.9))
    is_obstructed = combined_score >= threshold

    # Determine obstacle center of mass to choose optimal flank direction
    flank_direction = "left"
    obstacle_center = None
    if is_obstructed:
        ys, xs = np.where(occlusion_edges > 0)
        if len(xs) > 0:
            avg_x = float(np.mean(xs)) + crop_ox + min_x
            avg_y = float(np.mean(ys)) + crop_oy + min_y
            obstacle_center = (int(round(avg_x)), int(round(avg_y)))
            
            # Cross product of (p2 - p1) and (obstacle - p1) determines which side obstacle is on
            cross = (x2 - x1) * (avg_y - y1) - (y2 - y1) * (avg_x - x1)
            # If obstacle is to the right of the forward sightline, strafe left; otherwise strafe right
            flank_direction = "left" if cross > 0 else "right"

    return is_obstructed, combined_score, obstacle_center, flank_direction


# =============================================================================
# Core Combat Positioning & Line-of-Sight Classes
# =============================================================================

# =============================================================================
# Per-profile range limits (single source of truth for evaluate_target_range)
# =============================================================================
# TEK FORMÜL (P1-12 / v18): bbox -> mesafe dönüşümü ile mesafe kararı aynı
# formüldür — `est_dist = PINHOLE_K_M / norm_h` — ve her iki dal da aynı
# metre sınırlarına (dist_far_m / dist_close_m) karşı karar verir. Eski kod
# iki ayrı formül kullanıyordu: mesafe dalı bu tablodaki metre sınırlarını,
# bbox dalı ise config'deki norm_h sınırlarını; üstelik tablonun üstündeki
# yorum "sınırlar 1.8 / norm_h dönüşümünü yansıtır" diyordu ama bu yalnızca
# hybrid için doğruydu (melee: 3.8 m x 0.11 = 0.42; ranged: 24 x 0.035 =
# 0.84, 6 x 0.16 = 0.96; hybrid: 25 x 0.07 = 1.75, 4 x 0.45 = 1.80).
#
# PINHOLE_K_M = 0.96, kalibrasyon dayanağı core/mana_phase_combat.py:925:
# "Character model height ~ 1.8m. At 8m distance, norm_h ≈ 0.12"
# (0.96 / 0.12 = 8.0 m) — canlı savaş döngüsünde kullanılan
# ManaPhaseCombat.estimate_target_distance_from_bbox ile aynı sabit ve
# test_mana_phase_combat.py::test_estimate_target_distance_from_bbox ile
# kilitli (norm_h 0.24 -> <= 5 m; norm_h 0.06 -> > 12 m; eski 1.8 sabiti
# bu testi bozardı: 1.8 / 0.24 = 7.5 m). Eski 1.8 değeri kalibresizdi —
# karakter boyu (1.8 m) pinhole sabitiyle karıştırılmış görünüyor.
#
# Metre sınırları tek doğruluk kaynağıdır: oyun-içi menzil sayılarıdır
# (melee ~3.8 m, ranged 6-24 m, hybrid 4-25 m) ve config yorumlarıyla da
# uyumludur (ranged 0.16 -> 0.96/0.16 = 6.0 m tam; 0.035 -> 27.4 m,
# "> 25 m" yorumu yön olarak doğru). k=0.96 ile türetilmiş norm_h
# karşılıkları: melee_min 0.253, ranged 0.040 / 0.160, hybrid 0.038 / 0.240.
#
# Melee has NO too-close distance check: point-blank is still in melee
# range. (Eski bbox dalı norm_h > 0.70 için TOO_CLOSE veriyordu — mesafe
# dalıyla çelişen bu asimetri tek formüle geçişle kalktı.)
# Reason strings are the distance-branch strings, now used by both branches.

PINHOLE_K_M: float = 0.96

@dataclass(frozen=True)
class _ProfileRangeLimits:
    label: str
    dist_far_m: float
    dist_close_m: Optional[float]  # None: distance never reports TOO_CLOSE
    dist_in_range_phrase: str


_PROFILE_RANGE_LIMITS: Dict[EngagementProfile, _ProfileRangeLimits] = {
    EngagementProfile.MELEE: _ProfileRangeLimits(
        label="Melee",
        dist_far_m=3.8,
        dist_close_m=None,
        dist_in_range_phrase="in range",
    ),
    EngagementProfile.RANGED: _ProfileRangeLimits(
        label="Ranged",
        dist_far_m=24.0,
        dist_close_m=6.0,
        dist_in_range_phrase="in optimal standoff zone",
    ),
    EngagementProfile.HYBRID: _ProfileRangeLimits(
        label="Hybrid",
        dist_far_m=25.0,
        dist_close_m=4.0,
        dist_in_range_phrase="in optimal zone",
    ),
}


def _evaluate_by_distance(
    profile: EngagementProfile,
    limits: _ProfileRangeLimits,
    dist: float,
    target_norm_h: float = 0.0,
) -> RangeEvaluation:
    """Single range decision for both branches: compare meters to limits."""
    if dist > limits.dist_far_m:
        return RangeEvaluation(
            status=RangeStatus.TOO_FAR,
            profile=profile,
            target_norm_h=target_norm_h,
            estimated_distance_m=dist,
            recommended_action="step_forward",
            reason=f"{limits.label} target distance {dist:.1f}m > {limits.dist_far_m:.1f}m limit",
        )
    if limits.dist_close_m is not None and dist < limits.dist_close_m:
        return RangeEvaluation(
            status=RangeStatus.TOO_CLOSE,
            profile=profile,
            target_norm_h=target_norm_h,
            estimated_distance_m=dist,
            recommended_action="backpedal",
            reason=f"{limits.label} target distance {dist:.1f}m < {limits.dist_close_m:.1f}m danger zone",
        )
    return RangeEvaluation(
        status=RangeStatus.IN_RANGE,
        profile=profile,
        target_norm_h=target_norm_h,
        estimated_distance_m=dist,
        recommended_action="none",
        reason=f"{limits.label} target distance {dist:.1f}m {limits.dist_in_range_phrase}",
    )


def _driver_can_send(driver: Any) -> bool:
    """True when the driver can dispatch at least one key input.

    Callers must not report (or count) an input action as performed when the
    driver cannot send anything — otherwise the bot logs phantom successes.
    """
    if driver is None:
        return False
    if hasattr(driver, "press_key"):
        return True
    return hasattr(driver, "key_down") and hasattr(driver, "key_up")


class CombatPositioner:
    """
    Calculates engagement ranges, evaluates target distance, and executes
    gap-closing (W) or kiting/backpedaling (S) movement commands.
    """

    def __init__(self, config: Optional[CombatPositioningConfig] = None):
        self.config = config or CombatPositioningConfig()
        self.consecutive_gap_closes = 0
        self.consecutive_kites = 0

    def set_profile(self, profile: Union[str, EngagementProfile]) -> None:
        if isinstance(profile, str):
            profile = EngagementProfile(profile.lower())
        self.config.profile = profile
        logger.info("Combat engagement profile set to: %s", self.config.profile.value)

    def evaluate_target_range(
        self,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        screen_shape: Optional[Tuple[int, int]] = None,
        estimated_distance_m: Optional[float] = None,
    ) -> RangeEvaluation:
        """
        Evaluates whether target is IN_RANGE, TOO_FAR, or TOO_CLOSE.
        """
        profile = self.config.profile
        limits = _PROFILE_RANGE_LIMITS[profile]

        # Case 1: Direct distance in meters provided
        if estimated_distance_m is not None and estimated_distance_m > 0:
            return _evaluate_by_distance(profile, limits, float(estimated_distance_m))

        # Case 2: Target bounding box (x, y, w, h) provided.
        # Same single formula as Case 1: convert norm_h to meters with
        # PINHOLE_K_M (the calibrated constant shared with
        # ManaPhaseCombat.estimate_target_distance_from_bbox) and decide
        # against the same meter limits — no separate norm_h comparison.
        if target_bbox is not None and screen_shape is not None:
            sh, sw = screen_shape[:2]
            if sh > 0:
                bh = target_bbox[3]
                norm_h = float(bh) / float(sh)
                # Perspective pinhole approximation: distance is inversely proportional to norm_h
                est_dist = max(1.0, min(50.0, PINHOLE_K_M / max(0.01, norm_h)))
                return _evaluate_by_distance(profile, limits, est_dist, target_norm_h=norm_h)

        # Fallback when no visual bounding box is available
        return RangeEvaluation(
            status=RangeStatus.UNKNOWN,
            profile=profile,
            recommended_action="none",
            reason="No bounding box or distance metric available",
        )

    def execute_gap_close(
        self,
        driver: Any,
        duration_sec: Optional[float] = None,
    ) -> bool:
        """Executes forward step (W) to close the distance toward target.

        Returns True only when a key was actually dispatched. A driver that
        cannot send input returns False and counts no gap-close.
        """
        if driver is None or not _driver_can_send(driver):
            return False
        dur = float(duration_sec or self.config.gap_close_step_sec)
        try:
            # P1-11 / S-1302: gönderilmeyen basış başarı sayılmaz —
            # press_key 0.0 (veya key_down False) dönerse False döner,
            # sayaç artmaz.
            if hasattr(driver, "press_key"):
                if not _press_key_sent(driver.press_key("w", hold_sec=dur)):
                    logger.debug("execute_gap_close W not sent (press_key 0.0)")
                    return False
            elif hasattr(driver, "key_down") and hasattr(driver, "key_up"):
                if not bool(driver.key_down("w")):
                    logger.debug("execute_gap_close W key_down refused")
                    return False
                time.sleep(dur)
                if not bool(driver.key_up("w")):
                    logger.debug("execute_gap_close W key_up not sent")
            else:
                return False
        except Exception as exc:
            logger.debug("execute_gap_close failed: %s", exc)
            return False
        self.consecutive_gap_closes += 1
        logger.info("Executing gap close step #%d: W for %.2fs", self.consecutive_gap_closes, dur)
        return True

    def execute_kiting(
        self,
        driver: Any,
        duration_sec: Optional[float] = None,
    ) -> bool:
        """Executes backpedaling step (S) to maintain standoff distance for ranged.

        Returns True only when a key was actually dispatched. A driver that
        cannot send input returns False and counts no kite.
        """
        if driver is None or not _driver_can_send(driver):
            return False
        dur = float(duration_sec or self.config.kite_step_sec)
        try:
            # P1-11 / S-1302: gönderilmeyen basış başarı sayılmaz —
            # press_key 0.0 (veya key_down False) dönerse False döner,
            # sayaç artmaz.
            if hasattr(driver, "press_key"):
                if not _press_key_sent(driver.press_key("s", hold_sec=dur)):
                    logger.debug("execute_kiting S not sent (press_key 0.0)")
                    return False
            elif hasattr(driver, "key_down") and hasattr(driver, "key_up"):
                if not bool(driver.key_down("s")):
                    logger.debug("execute_kiting S key_down refused")
                    return False
                time.sleep(dur)
                if not bool(driver.key_up("s")):
                    logger.debug("execute_kiting S key_up not sent")
            else:
                return False
        except Exception as exc:
            logger.debug("execute_kiting failed: %s", exc)
            return False
        self.consecutive_kites += 1
        logger.info("Executing kiting step #%d: S for %.2fs", self.consecutive_kites, dur)
        return True

    def reset(self) -> None:
        self.consecutive_gap_closes = 0
        self.consecutive_kites = 0


class LineOfSightTracker:
    """
    Checks line of sight against obstacles via:
    1. 2D ray intersection with detected obstacle bounding boxes
    2. Computer vision corridor analysis (edge density and rock/wall masks)
    3. Empirical damage stagnation (N casts with 0 HP loss)
    Executes tactical maneuvers (strafe, flank, jump, tab-reset) when obstructed.
    """

    def __init__(self, config: Optional[CombatPositioningConfig] = None):
        self.config = config or CombatPositioningConfig()
        self.consecutive_stagnant_casts = 0
        self.recovery_attempts = 0
        # S-1303: no fabricated 100.0 baseline. None means "no reading yet":
        # the first HP reading after (re)targeting is the baseline, never a
        # verdict — a pre-damaged target must not count as "damage dealt"
        # on a 0-damage first cast.
        self.last_target_hp_pct: Optional[float] = None
        self.last_cast_time = 0.0

    def check_line_of_sight(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        player_pos: Optional[Tuple[int, int]] = None,
        obstacle_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> LineOfSightResult:
        """
        Evaluates full line-of-sight status.
        """
        # 1. Damage stagnation check (empirical cast feedback)
        if self.consecutive_stagnant_casts >= self.config.damage_stagnation_threshold:
            flank_dir = "left" if (self.recovery_attempts % 2 == 0) else "right"
            return LineOfSightResult(
                has_los=False,
                obstruction_type=ObstructionType.DAMAGE_STAGNATION,
                obstruction_score=1.0,
                flank_direction=flank_dir,
                reason=(
                    f"Damage stagnation: {self.consecutive_stagnant_casts} consecutive casts "
                    f"with 0 target HP drop"
                ),
            )

        # Determine player and target positions in screen space.
        # Never invent a 1080x1920 frame: without a frame and without explicit
        # positions there is nothing to anchor the sightline against, so the
        # check is skipped (clear) instead of evaluated on fake coordinates.
        if frame_bgr is None and (player_pos is None or target_pos is None):
            return LineOfSightResult(
                has_los=True,
                obstruction_type=ObstructionType.NONE,
                obstruction_score=0.0,
                reason="Line-of-sight check skipped: no frame available to anchor player/target positions",
            )
        if frame_bgr is not None and frame_bgr.ndim >= 2:
            h, w = frame_bgr.shape[:2]
            p_pos = player_pos or (
                int(w * self.config.player_anchor_x_ratio),
                int(h * self.config.player_anchor_y_ratio),
            )
            t_pos = target_pos or (int(w * 0.50), int(h * 0.38))
        else:
            # Frame absent but both positions given explicitly: use them as-is.
            p_pos = player_pos  # type: ignore[assignment]
            t_pos = target_pos  # type: ignore[assignment]

        # 2. Geometric obstacle bounding boxes intersection check
        if obstacle_bboxes:
            for obs in obstacle_bboxes:
                ox, oy, ow, oh = obs
                obs_center = (ox + ow // 2, oy + oh // 2)
                # Skip obstacles that are coincident with target
                if math.hypot(obs_center[0] - t_pos[0], obs_center[1] - t_pos[1]) < 30:
                    continue
                if segment_intersects_rect(p_pos, t_pos, obs):
                    cross = (t_pos[0] - p_pos[0]) * (obs_center[1] - p_pos[1]) - (t_pos[1] - p_pos[1]) * (obs_center[0] - p_pos[0])
                    # Same sign convention as analyze_corridor_occlusion:
                    # obstacle right of the sightline (cross > 0) -> strafe left.
                    flank_dir = "left" if cross > 0 else "right"
                    return LineOfSightResult(
                        has_los=False,
                        obstruction_type=ObstructionType.GEOMETRIC_OBSTACLE,
                        obstruction_score=0.95,
                        obstacle_center=obs_center,
                        flank_direction=flank_dir,
                        reason=f"Sightline blocked by obstacle bbox {obs}",
                    )

        # 3. Computer Vision corridor occlusion analysis
        if frame_bgr is not None and frame_bgr.size > 0:
            is_obstructed, score, obs_center, flank_dir = analyze_corridor_occlusion(
                frame_bgr,
                p_pos,
                t_pos,
                corridor_width=self.config.corridor_width_px,
                threshold=self.config.corridor_occlusion_threshold,
            )
            if is_obstructed:
                return LineOfSightResult(
                    has_los=False,
                    obstruction_type=ObstructionType.CORRIDOR_OCCLUSION,
                    obstruction_score=score,
                    obstacle_center=obs_center,
                    flank_direction=flank_dir,
                    reason=f"Sightline corridor occluded by barrier (score={score:.2f})",
                )

        return LineOfSightResult(
            has_los=True,
            obstruction_type=ObstructionType.NONE,
            obstruction_score=0.0,
            reason="Line of sight clear",
        )

    def record_cast_outcome(
        self,
        target_hp_pct: float,
        now: Optional[float] = None,
    ) -> bool:
        """
        Records the target's HP percentage after a skill cast.
        If HP decreased, damage was dealt -> resets stagnation counter!
        If HP remained unchanged, increments stagnation counter.
        Returns True if damage was dealt.
        """
        curr_hp = float(target_hp_pct)
        # S-1303: first reading after reset()/__init__ is the baseline only.
        # No damage verdict is possible without a previous reading, so the
        # stagnation counter is left untouched (neither reset nor raised).
        prev_hp = self.last_target_hp_pct
        self.last_target_hp_pct = curr_hp
        self.last_cast_time = now if now is not None else time.time()
        if prev_hp is None:
            return False

        prev_hp = float(prev_hp)

        # If target took damage (at least 0.5% drop)
        if curr_hp < prev_hp - 0.5:
            self.consecutive_stagnant_casts = 0
            self.recovery_attempts = 0
            return True

        self.consecutive_stagnant_casts += 1
        return False

    def recover_line_of_sight(
        self,
        driver: Any,
        los_result: Optional[LineOfSightResult] = None,
        _camera_nudge_fn: Optional[Callable[[int, int], None]] = None,
    ) -> RecoveryActionResult:
        """
        Executes an adaptive recovery tactic to clear obstruction and regain LoS.
        Tactic progression:
        Attempt 1: Lateral strafe (A or D)
        Attempt 2: Diagonal flank (A+W or D+W)
        Attempt 3: Hop / Jump forward (Space + W)
        Attempt 4: Backpedal-flank (S then A/D)
        Attempt 5+: TAB reset / cycle target
        """
        self.recovery_attempts += 1
        flank_dir = "left"
        if los_result is not None and los_result.flank_direction in ("left", "right"):
            flank_dir = los_result.flank_direction
        else:
            flank_dir = "left" if (self.recovery_attempts % 2 == 1) else "right"

        strafe_key = "a" if flank_dir == "left" else "d"

        # Phase 5: Exceeded max attempts -> Tab cycle to next target
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            # Capture the attempt number BEFORE reset() zeroes the counter.
            attempt_number = self.recovery_attempts
            logger.warning(
                "LoS recovery attempts exceeded limit (%d >= %d). Resetting target via Tab/Esc.",
                attempt_number,
                self.config.max_recovery_attempts,
            )
            # S-1101: report only what was actually sent. When the driver
            # has no press_key (or raises), no Tab/Esc went out, so the
            # target was NOT cleared — target_switched must stay False and
            # the description must say so (Phases 1-4 already do this).
            # S-1302: a press_key returning 0.0 sent nothing (focus refused
            # / no trusted path) — that key must not be claimed either.
            dispatched_keys: List[str] = []
            if hasattr(driver, "press_key"):
                try:
                    if _press_key_sent(driver.press_key("escape", hold_sec=0.08)):
                        dispatched_keys.append("escape")
                    time.sleep(0.10)
                    if _press_key_sent(driver.press_key("tab", hold_sec=0.08)):
                        dispatched_keys.append("tab")
                except Exception as exc:
                    logger.debug("LoS recovery Tab-reset dispatch failed: %s", exc)
            self.reset()
            if dispatched_keys == ["escape", "tab"]:
                description = "Max recovery attempts reached; target cleared via Tab/Esc"
            elif dispatched_keys:
                description = (
                    "Max recovery attempts reached; target partially cleared via "
                    f"{'/'.join(dispatched_keys)} (remaining keys NOT dispatched)"
                )
            else:
                description = (
                    "Max recovery attempts reached; Tab/Esc NOT dispatched "
                    "(no driver press_key) - target NOT cleared"
                )
            return RecoveryActionResult(
                tactic=RecoveryTactic.TAB_RESET,
                keys_dispatched=dispatched_keys,
                duration_sec=0.20,
                attempt_number=attempt_number,
                target_switched=bool(dispatched_keys),
                description=description,
            )

        # Phase 1: Pure lateral strafe (A or D)
        if self.recovery_attempts == 1:
            dur = self.config.strafe_step_sec
            logger.info("LoS Recovery #1: Lateral strafe %s for %.2fs", strafe_key.upper(), dur)
            dispatched_keys = []
            if hasattr(driver, "press_key"):
                try:
                    # S-1302: press_key returns 0.0 when nothing was sent —
                    # claim the strafe only when it was actually dispatched.
                    if _press_key_sent(driver.press_key(strafe_key, hold_sec=dur)):
                        dispatched_keys = [strafe_key]
                except Exception as exc:
                    logger.debug("LoS recovery strafe dispatch failed: %s", exc)
            tactic = RecoveryTactic.STRAFE_LEFT if strafe_key == "a" else RecoveryTactic.STRAFE_RIGHT
            return RecoveryActionResult(
                tactic=tactic,
                keys_dispatched=dispatched_keys,
                duration_sec=dur,
                attempt_number=1,
                description=f"Strafe {strafe_key.upper()} to acquire angle around obstacle",
            )

        # Phase 2: Diagonal flank (A+W or D+W)
        if self.recovery_attempts == 2:
            dur = self.config.flank_step_sec
            logger.info("LoS Recovery #2: Diagonal flank %s+W for %.2fs", strafe_key.upper(), dur)
            dispatched_keys = []
            if hasattr(driver, "key_down") and hasattr(driver, "key_up"):
                try:
                    # S-1301: key_down/key_up return False when no trusted
                    # path exists (no soft fallthrough) — claim only keys
                    # that really went down. A key that went down is always
                    # released again, even if its partner failed, so no
                    # key is left stuck.
                    down_strafe = bool(driver.key_down(strafe_key))
                    down_w = bool(driver.key_down("w")) if down_strafe else False
                    if down_strafe and down_w:
                        time.sleep(dur)
                    if down_w and not driver.key_up("w"):
                        logger.debug("LoS recovery flank: key_up('w') not sent")
                    if down_strafe and not driver.key_up(strafe_key):
                        logger.debug("LoS recovery flank: key_up(%r) not sent", strafe_key)
                    if down_strafe:
                        dispatched_keys.append(strafe_key)
                    if down_w:
                        dispatched_keys.append("w")
                except Exception as exc:
                    logger.debug("LoS recovery flank dispatch failed: %s", exc)
            elif hasattr(driver, "press_key"):
                try:
                    # S-1302: same 0.0 rule as Phases 1/3/4 for the fallback.
                    if _press_key_sent(driver.press_key(strafe_key, hold_sec=dur)):
                        dispatched_keys.append(strafe_key)
                    if _press_key_sent(driver.press_key("w", hold_sec=dur * 0.7)):
                        dispatched_keys.append("w")
                except Exception as exc:
                    logger.debug("LoS recovery flank dispatch failed: %s", exc)
            tactic = RecoveryTactic.FLANK_LEFT if strafe_key == "a" else RecoveryTactic.FLANK_RIGHT
            return RecoveryActionResult(
                tactic=tactic,
                keys_dispatched=dispatched_keys,
                duration_sec=dur,
                attempt_number=2,
                description=f"Diagonal flank {strafe_key.upper()}+W rounding obstacle corner",
            )

        # Phase 3: Obstacle clearing hop / jump (Space + W)
        if self.recovery_attempts == 3:
            dur = 0.30
            logger.info("LoS Recovery #3: Jump-forward (Space + W) to clear low barrier")
            dispatched_keys = []
            if hasattr(driver, "press_key"):
                try:
                    # S-1302: claim each key only if its press was sent.
                    if _press_key_sent(driver.press_key("space", hold_sec=0.08)):
                        dispatched_keys.append("space")
                    if _press_key_sent(driver.press_key("w", hold_sec=dur)):
                        dispatched_keys.append("w")
                except Exception as exc:
                    logger.debug("LoS recovery jump dispatch failed: %s", exc)
            return RecoveryActionResult(
                tactic=RecoveryTactic.JUMP_FORWARD,
                keys_dispatched=dispatched_keys,
                duration_sec=dur,
                attempt_number=3,
                description="Jump-forward (Space+W) over obstacle threshold",
            )

        # Phase 4: Backpedal-flank (S then A/D) to clear tight corners
        dur = self.config.backpedal_step_sec
        logger.info("LoS Recovery #4: Backpedal-flank (S then %s)", strafe_key.upper())
        dispatched_keys = []
        if hasattr(driver, "press_key"):
            try:
                # S-1302: claim each key only if its press was sent.
                if _press_key_sent(driver.press_key("s", hold_sec=dur)):
                    dispatched_keys.append("s")
                time.sleep(0.08)
                if _press_key_sent(driver.press_key(strafe_key, hold_sec=dur)):
                    dispatched_keys.append(strafe_key)
            except Exception as exc:
                logger.debug("LoS recovery backpedal dispatch failed: %s", exc)
        return RecoveryActionResult(
            tactic=RecoveryTactic.BACKPEDAL_FLANK,
            keys_dispatched=dispatched_keys,
            duration_sec=dur * 2,
            attempt_number=4,
            description=f"Backpedal S and lateral {strafe_key.upper()} to escape obstacle corner",
        )

    def reset(self) -> None:
        self.consecutive_stagnant_casts = 0
        self.recovery_attempts = 0
        # S-1303: baseline cleared, not fabricated. The next
        # record_cast_outcome() reading becomes the baseline (see above).
        self.last_target_hp_pct = None


# =============================================================================
# Upgraded CombatSystem Class
# =============================================================================

class CombatSystem:
    """
    Combat & Defense System with Integrated Positioning and LoS Recovery.
    Provides complete backward compatibility with existing callers while
    enforcing range awareness, obstacle checking, and genuine recovery actions.
    """

    def __init__(
        self,
        vision: Optional[Vision] = None,
        driver: Optional[InputDriver] = None,
        profile: Union[str, EngagementProfile] = EngagementProfile.MELEE,
        config: Optional[CombatPositioningConfig] = None,
    ):
        self.vision = vision
        self.driver = driver
        self.in_combat = False
        self.last_attack_time = 0.0
        self.was_fighting = False
        self.no_target_streak = 0
        self.last_tab_time = 0.0

        # Positioning and Line-of-Sight components
        self.config = config or CombatPositioningConfig()
        if profile is not None:
            if isinstance(profile, str):
                profile = EngagementProfile(profile.lower())
            self.config.profile = profile

        self.positioner = CombatPositioner(self.config)
        self.los_tracker = LineOfSightTracker(self.config)

        # Diagnostics & telemetry stats
        self.total_attacks = 0
        self.total_gap_closes = 0
        self.total_kites = 0
        self.total_los_recoveries = 0
        self.last_target_hp = 100.0
        self.combat_skill_keys: Tuple[str, ...] = ("1", "2", "3", "4", "q", "e", "r", "t")
        self.last_skill_key: Optional[str] = None
        self.skill_rotation_counts: Dict[str, int] = {k: 0 for k in self.combat_skill_keys}
        self.anti_sticky_2_prevented: int = 0
        # F13 (v18): tek F+Tab loot zinciri — §7-A kararı ("loot tek F",
        # davranış değişmeyecek): ölüm dalı ölüm başına TEK F basar.
        # Sayaç yalnızca GÖNDERİLEN zinciri sayar (S-1302: her iki basış
        # da _press_key_sent'ten geçtiyse); "loot toplandı" iddiası
        # değildir, doğrulama combat_recovery no27 akışına aittir.
        self.loot_tab_chain_count: int = 0
        self.consumable_priority_counts: Dict[str, int] = {
            "hp_potion": 0,
            "mp_potion": 0,
            "scroll": 0,
        }

    # -------------------------------------------------------------------------
    # Profile & Configuration
    # -------------------------------------------------------------------------

    def set_engagement_profile(self, profile: Union[str, EngagementProfile]) -> None:
        """Sets active combat profile (melee, ranged, hybrid)."""
        self.positioner.set_profile(profile)

    def get_engagement_profile(self) -> EngagementProfile:
        return self.config.profile

    def reset_positioning_state(self) -> None:
        """Resets all positioning and LoS recovery counters."""
        self.positioner.reset()
        self.los_tracker.reset()

    # -------------------------------------------------------------------------
    # Attack Detection (Backward Compatible & Enhanced)
    # -------------------------------------------------------------------------

    def check_under_attack(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Detects if player is being attacked or targeting a hostile:
        1. Checks target bar (red/olive health bar at top-center/top-left)
        2. Checks center combat zone for hostile nameplates
        3. Supports calibrated ROIs from vision engine when available
        """
        # If vision engine provides read_player_hp_mp_bars, use it as high-accuracy path
        if self.vision is not None and hasattr(self.vision, "read_player_hp_mp_bars"):
            try:
                vitals = self.vision.read_player_hp_mp_bars(frame)
                if isinstance(vitals, dict) and vitals.get("has_target", False):
                    return True
            except Exception as exc:
                logger.debug("check_under_attack vitals read failed: %s", exc)

        screen_bgr = frame
        if screen_bgr is None and self.vision is not None and hasattr(self.vision, "grab_screen"):
            screen_bgr = self.vision.grab_screen(grayscale=False)

        if screen_bgr is None or not isinstance(screen_bgr, np.ndarray) or screen_bgr.size == 0:
            return False

        h, w = screen_bgr.shape[:2]
        # Top-center enemy target frame zone: y: 2% to 15%, x: 30% to 70%
        target_roi = screen_bgr[int(h * 0.02) : int(h * 0.15), int(w * 0.30) : int(w * 0.70)]
        if target_roi.size == 0:
            return False

        # Red enemy health bar in HSV
        hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
        red_mask1 = (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 12) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 100)
        red_mask2 = (hsv[:, :, 0] >= 168) & (hsv[:, :, 0] <= 180) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 100)
        red_pixels = int(np.sum(red_mask1 | red_mask2))

        # Early exit: the target-bar hit already decides True — the center
        # zone scan below is only a second opinion for frames without a
        # target bar. Skipping it saves a second full-frame HSV conversion
        # (~40% of the frame) on every engaged step.
        if red_pixels > 60:
            return True

        # Center combat zone check for enemy overhead bars
        cz_roi = screen_bgr[int(h * 0.16) : int(h * 0.58), int(w * 0.22) : int(w * 0.78)]
        cz_pixels = 0
        if cz_roi.size > 0:
            hsv_cz = cv2.cvtColor(cz_roi, cv2.COLOR_BGR2HSV)
            cz_mask = (
                ((hsv_cz[:, :, 0] <= 10) | (hsv_cz[:, :, 0] >= 170))
                & (hsv_cz[:, :, 1] >= 135)
                & (hsv_cz[:, :, 2] >= 110)
            )
            cz_pixels = int(np.sum(cz_mask))

        return (red_pixels > 60) or (cz_pixels > 85)

    # -------------------------------------------------------------------------
    # Positioning & LoS Methods
    # -------------------------------------------------------------------------

    def check_engagement_range(
        self,
        frame: Optional[np.ndarray] = None,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        distance_m: Optional[float] = None,
    ) -> RangeEvaluation:
        """Evaluates engagement range for the current target."""
        # S-1102: never invent a (1080, 1920) screen when there is no
        # frame — the F-1785 pattern in LineOfSightTracker. Without a
        # frame there is no anchor for the bbox height, so pass no screen
        # shape and let evaluate_target_range() report UNKNOWN instead of
        # a TOO_CLOSE/IN_RANGE verdict computed against a guessed
        # resolution (wrong on 1440p/720p screens).
        screen_shape = frame.shape[:2] if (frame is not None and frame.ndim >= 2) else None
        return self.positioner.evaluate_target_range(
            target_bbox=target_bbox,
            screen_shape=screen_shape,
            estimated_distance_m=distance_m,
        )

    def check_line_of_sight(
        self,
        frame: Optional[np.ndarray] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        player_pos: Optional[Tuple[int, int]] = None,
        obstacle_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> LineOfSightResult:
        """Evaluates line of sight against obstacles."""
        return self.los_tracker.check_line_of_sight(
            frame_bgr=frame,
            target_pos=target_pos,
            player_pos=player_pos,
            obstacle_bboxes=obstacle_bboxes,
        )

    def recover_obstructed_target(
        self,
        driver: Optional[Any] = None,
        los_result: Optional[LineOfSightResult] = None,
    ) -> RecoveryActionResult:
        """Executes tactical maneuver to regain line of sight."""
        drv = driver if driver is not None else self.driver
        result = self.los_tracker.recover_line_of_sight(driver=drv, los_result=los_result)
        # Count the recovery only when a maneuver key was actually dispatched.
        if result.keys_dispatched:
            self.total_los_recoveries += 1
        return result

    # -------------------------------------------------------------------------
    # Fight Back Step (Integrated Positioning + LoS + Skill Rotation)
    # -------------------------------------------------------------------------

    def fight_back_step(
        self,
        input_ctrl: Optional[Any] = None,
        frame: Optional[np.ndarray] = None,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_hp_pct: Optional[float] = None,
        obstacle_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> bool:
        """
        Executes one step of the combat loop:
        1. Checks if under attack / targeting hostile
        2. When enemy dead: loots (F, once — §7-A "loot tek F"), resets
           positioning/LoS state, tabs next target
        3. When active target:
           a. Evaluates engagement range -> executes gap-close (W) or kiting (S) if needed
           b. Evaluates line of sight -> executes recovery tactic (strafe/flank/jump) if blocked
           c. Casts attack skill rotation and records target HP outcome
        """
        driver = input_ctrl if input_ctrl is not None else self.driver
        now = time.time()
        has_enemy = self.check_under_attack(frame)

        if not has_enemy:
            self.no_target_streak += 1
            if self.was_fighting and self.no_target_streak >= 2:
                # Enemy died -> Loot (F), reset state, Tab to next monster.
                # F13 (v18): v17 davranışı geri getirildi — §7-A kararı
                # ("loot tek F", davranış değişmeyecek). P1-11 (1b48732)
                # bu F'i "no27 akışıyla çift F" gerekçesiyle kaldırmıştı;
                # o gerekçe bu yolda geçersiz: CombatSystem yalnızca
                # combat_recovery import edilemezse fallback'tir
                # (state_machine._init_default_combat_recovery) ve L2
                # akışı streak>=2'de zaten F'i atlar (l3_looted) — yani
                # v17'de çift F yoktu, tek F vardı. Kaldırmak loot'u
                # sıfırladı (combat_10 ×2 regresyon). Dal bir kez
                # çalışır (was_fighting/streak sıfırlanır) — ölüm başına
                # TEK F, kör tekrar yok; "loot toplandı" iddiası
                # yapılmaz, sayaç yalnızca gönderilen zinciri sayar.
                self.was_fighting = False
                self.no_target_streak = 0
                self.in_combat = False
                self.reset_positioning_state()
                looted = False
                if hasattr(driver, "press_key"):
                    try:
                        # S-1302: the loot+Tab chain counts only when BOTH
                        # presses were actually sent (0.0 = nothing sent).
                        f_sent = _press_key_sent(driver.press_key("f", hold_sec=0.10))
                        time.sleep(0.18)
                        tab_sent = _press_key_sent(driver.press_key("tab", hold_sec=0.08))
                        looted = bool(f_sent and tab_sent)
                    except Exception as exc:
                        logger.debug("loot F+Tab dispatch failed: %s", exc)
                if looted:
                    self.loot_tab_chain_count += 1
                return looted

            # If no target, press Tab periodically
            if now - self.last_tab_time >= 1.2:
                self.last_tab_time = now
                if hasattr(driver, "press_key"):
                    driver.press_key("tab", hold_sec=0.08)
            return False

        # Active target confirmed
        self.was_fighting = True
        self.no_target_streak = 0
        self.in_combat = True

        # Read or update target HP if provided or accessible
        curr_target_hp = target_hp_pct
        if curr_target_hp is None and self.vision is not None and hasattr(self.vision, "read_player_hp_mp_bars"):
            try:
                vitals = self.vision.read_player_hp_mp_bars(frame)
                if isinstance(vitals, dict) and vitals.get("has_target", False):
                    curr_target_hp = float(vitals.get("target_hp_pct", 100.0))
            except Exception as exc:
                logger.debug("fight_back_step vitals read failed: %s", exc)

        # NOTE: the stagnation outcome is recorded only after an actual skill
        # cast below — never on gap-close / kiting / LoS-recovery steps, which
        # cast nothing and must not inflate consecutive_stagnant_casts.

        # ---------------------------------------------------------------------
        # Range Check & Repositioning
        # ---------------------------------------------------------------------
        range_eval = self.check_engagement_range(frame=frame, target_bbox=target_bbox)
        if range_eval.status == RangeStatus.TOO_FAR:
            if self.positioner.execute_gap_close(driver):
                self.total_gap_closes += 1
                return True
            return False
        elif range_eval.status == RangeStatus.TOO_CLOSE:
            if self.positioner.execute_kiting(driver):
                self.total_kites += 1
                return True
            return False

        # ---------------------------------------------------------------------
        # Line-of-Sight Check & Recovery
        # ---------------------------------------------------------------------
        t_pos = None
        if target_bbox is not None:
            t_pos = (target_bbox[0] + target_bbox[2] // 2, target_bbox[1] + target_bbox[3] // 2)

        los_result = self.check_line_of_sight(
            frame=frame,
            target_pos=t_pos,
            obstacle_bboxes=obstacle_bboxes,
        )
        if not los_result.has_los:
            recovery = self.recover_obstructed_target(driver=driver, los_result=los_result)
            # True only when the recovery maneuver actually dispatched keys.
            return bool(recovery.keys_dispatched)

        # ---------------------------------------------------------------------
        # Skill Rotation Execution (8-key: 1,2,3,4,q,e,r,t with anti-sticky '2';
        # t = right-click skill, r = left-click attack — no mouse clicks, keys only)
        # ---------------------------------------------------------------------
        # GCD wait: still engaged with the target, so True — but it counts no
        # attack (total_attacks is untouched on this path).
        if now - self.last_attack_time < self.config.skill_gcd_sec:
            return True

        # The rotation pool only dispatches via press_key; a driver without it
        # cannot cast — report failure instead of a phantom attack.
        if driver is None or not hasattr(driver, "press_key"):
            return False

        action_pool: List[Tuple[str, Callable[[], Any]]] = [
            ("t", lambda: driver.press_key("t") if hasattr(driver, "press_key") else None),
            ("1", lambda: driver.press_key("1") if hasattr(driver, "press_key") else None),
            ("2", lambda: driver.press_key("2") if hasattr(driver, "press_key") else None),
            ("3", lambda: driver.press_key("3") if hasattr(driver, "press_key") else None),
            ("4", lambda: driver.press_key("4") if hasattr(driver, "press_key") else None),
            ("q", lambda: driver.press_key("q") if hasattr(driver, "press_key") else None),
            ("e", lambda: driver.press_key("e") if hasattr(driver, "press_key") else None),
            ("r", lambda: driver.press_key("r") if hasattr(driver, "press_key") else None),
        ]
        # F-0230: deterministic round-robin rotation instead of
        # random.choice. The cursor advances over the fixed 8-key pool
        # order (t, 1, 2, 3, 4, q, e, r — t = right-click skill, r =
        # left-click attack, keys only, no mouse clicks), so every key is
        # used exactly once per cycle and the sequence is reproducible.
        # The anti-sticky "2" guard steps the cursor past "2" when the
        # previous cast was "2" instead of rebuilding a shorter pool: a
        # filtered pool would shift the cycle and starve keys (e.g. "3").
        rotation_index = getattr(self, "_skill_rotation_index", 0)
        chosen_label, chosen = action_pool[0]
        for _ in range(len(action_pool)):
            candidate_label, candidate = action_pool[rotation_index % len(action_pool)]
            rotation_index += 1
            if candidate_label == "2" and self.last_skill_key == "2":
                # S-1401: count a prevented sticky "2" only when the cursor
                # actually landed on "2" and stepped past it — never on a
                # mere "last cast was 2" (the old pre-loop increment fired
                # on every cast after a "2", even with no skip happening).
                self.anti_sticky_2_prevented += 1
                continue
            chosen_label, chosen = candidate_label, candidate
            break
        self._skill_rotation_index = rotation_index
        try:
            # S-1302: chosen() returns press_key's result — 0.0 means the
            # cast never went out (focus refused / no trusted path). Count
            # no attack, record no cast outcome, and report failure.
            cast_result = chosen()
        except Exception as exc:
            logger.debug("skill rotation dispatch failed: %s", exc)
            return False
        if not _press_key_sent(cast_result):
            logger.debug("skill rotation cast '%s' not sent (no trusted path/focus)", chosen_label)
            return False
        if chosen_label in self.skill_rotation_counts:
            self.skill_rotation_counts[chosen_label] += 1
            self.last_skill_key = chosen_label
        self.last_attack_time = now
        self.total_attacks += 1
        # Stagnation is judged on a real cast only: compare the HP reading
        # taken for this step against the previous cast's outcome.
        if curr_target_hp is not None:
            # R06: the first record after a (re)target is the baseline.
            # Capture the flag BEFORE recording — record_cast_outcome sets
            # last_target_hp_pct, so the check must precede the call.
            is_first_reading = self.los_tracker.last_target_hp_pct is None
            self.los_tracker.record_cast_outcome(curr_target_hp, now=now)
            self.last_target_hp = curr_target_hp
            if is_first_reading:
                # Post-cast re-read, first cast for this target only: the HP
                # bar read at step start predates this cast, so a stale
                # value can seed a false stagnation streak. Settle, then
                # re-read once with the same pattern as the step-start read
                # above. Vision missing / no reader / exception / no target
                # / no reading -> skip silently, behaviour unchanged.
                post_hp: Optional[float] = None
                if self.vision is not None and hasattr(self.vision, "read_player_hp_mp_bars"):
                    time.sleep(POSTCAST_HP_SETTLE_SEC)
                    try:
                        post_vitals = self.vision.read_player_hp_mp_bars(frame)
                        if isinstance(post_vitals, dict) and post_vitals.get("has_target", False):
                            post_hp = float(post_vitals.get("target_hp_pct", 100.0))
                    except Exception as exc:
                        logger.debug("fight_back_step post-cast vitals read failed: %s", exc)
                if post_hp is not None:
                    self.los_tracker.record_cast_outcome(post_hp, now=time.time())
                    self.last_target_hp = post_hp
        return True

    # -------------------------------------------------------------------------
    # Emergency Consumable Priority & Combat Loop Telemetry
    # -------------------------------------------------------------------------

    def evaluate_combat_consumable_priority(
        self,
        hp_pct: float,
        mp_pct: float,
        scroll_expired: bool = False,
        hp_threshold_pct: Optional[float] = None,
        mp_threshold_pct: Optional[float] = None,
    ) -> Optional[str]:
        """
        Evaluates emergency consumable priority during combat:
        1. Emergency HP potion if HP is below the potion threshold (F-0540
           tek kaynak: HP_POTION_THRESHOLD_PCT = 65, unless overridden).
        2. Emergency MP potion if MP is below the potion threshold (F-0540
           tek kaynak: MP_POTION_THRESHOLD_PCT = 45, unless overridden).
        3. Scroll renewal if scroll_expired is True.
        4. None if all vitals and buffs are healthy.
        Supports both [0.0..1.0] ratio and [0.0..100.0] percentage inputs.
        """
        raw_hp = float(hp_pct)
        raw_mp = float(mp_pct)
        norm_hp = raw_hp * 100.0 if 0.0 < raw_hp <= 1.0 else raw_hp
        norm_mp = raw_mp * 100.0 if 0.0 < raw_mp <= 1.0 else raw_mp
        hp_thresh = float(hp_threshold_pct) if hp_threshold_pct is not None else float(HP_POTION_THRESHOLD_PCT)
        mp_thresh = float(mp_threshold_pct) if mp_threshold_pct is not None else float(MP_POTION_THRESHOLD_PCT)

        if norm_hp < hp_thresh:
            self.consumable_priority_counts["hp_potion"] += 1
            return "hp_potion"
        if norm_mp < mp_thresh:
            self.consumable_priority_counts["mp_potion"] += 1
            return "mp_potion"
        if bool(scroll_expired):
            self.consumable_priority_counts["scroll"] += 1
            return "scroll"
        return None

    def get_combat_loop_telemetry(self) -> dict:
        """
        Returns telemetry statistics for the sequential 'f' (loot) + 'tab'
        (next target) chain on monster death — the two keys are dispatched in
        one step with a 0.18s humanized gap, not instantly — and the 8-key
        ('1','2','3','4','q','e','r','t') anti-sticky '2' skill rotation.
        """
        return {
            "module": "combat",
            "in_combat": self.in_combat,
            "loot_tab_chain": {
                "sequence": ["f", "tab"],
                "loot_key": "f",
                "target_key": "tab",
                "chain_executions": self.loot_tab_chain_count,
                "chain_enabled": True,
                "chain_gap_sec": 0.18,
            },
            "loot_tab_chain_count": self.loot_tab_chain_count,
            "skill_rotation": {
                "keys": list(self.combat_skill_keys),
                "key_count": len(self.combat_skill_keys),
                "anti_sticky_key": "2",
                "anti_sticky_2_enabled": True,
                "anti_sticky_2_prevented": self.anti_sticky_2_prevented,
                "last_skill_key": self.last_skill_key,
                "counts": dict(self.skill_rotation_counts),
            },
            "skill_keys": list(self.combat_skill_keys),
            "anti_sticky_2_enabled": True,
            "anti_sticky_2_prevented": self.anti_sticky_2_prevented,
            "skill_rotation_counts": dict(self.skill_rotation_counts),
            "consumable_priority_counts": dict(self.consumable_priority_counts),
            "total_attacks": self.total_attacks,
            "total_gap_closes": self.total_gap_closes,
            "total_kites": self.total_kites,
            "total_los_recoveries": self.total_los_recoveries,
        }

    # -------------------------------------------------------------------------
    # Diagnostics & Health Summaries (Preserved & Enhanced)
    # -------------------------------------------------------------------------

    def get_combat_detector_summary(self) -> dict:
        try:
            return {
                "combat_detector_ready": True,
                "profile": self.config.profile.value,
                "in_combat": self.in_combat,
                "vision_attached": self.vision is not None,
                "input_ready": _driver_can_send(self.driver),
                "consecutive_stagnant_casts": self.los_tracker.consecutive_stagnant_casts,
                "recovery_attempts": self.los_tracker.recovery_attempts,
            }
        except Exception:
            return {"combat_detector_ready": False}

    def get_combat_health_summary(self) -> dict:
        # "ready" means the module is initialized and its evaluation/detection
        # paths work (they need only a frame, no driver). Whether the bot can
        # actually act is reported separately via driver_attached/input_ready
        # so callers never mistake a driverless instance for a fighting one.
        return {
            "module": "combat",
            "ready": True,
            "profile": self.config.profile.value,
            "driver_attached": self.driver is not None,
            "input_ready": _driver_can_send(self.driver),
            "vision_attached": self.vision is not None,
            "in_combat": self.in_combat,
            "total_attacks": self.total_attacks,
            "total_gap_closes": self.total_gap_closes,
            "total_kites": self.total_kites,
            "total_los_recoveries": self.total_los_recoveries,
        }

    def get_combat_ready_flag(self) -> dict:
        return {
            "module": "combat",
            "ready": True,
            "driver_attached": self.driver is not None,
            "input_ready": _driver_can_send(self.driver),
        }

    def get_combat_version_info(self) -> dict:
        return {
            "module": "combat",
            "version": __version__,
            "ready": True,
            "capabilities": get_combat_capabilities(),
        }

    def get_combat_diagnostic_snapshot(self) -> dict:
        return {
            "module": "combat",
            "status": "ok",
            "ready": True,
            "profile": self.config.profile.value,
            "driver_attached": self.driver is not None,
            "input_ready": _driver_can_send(self.driver),
            "vision_attached": self.vision is not None,
            "in_combat": self.in_combat,
            "positioning": {
                "melee_min_norm_h": self.config.melee_min_norm_h,
                "ranged_min_norm_h": self.config.ranged_min_norm_h,
                "ranged_max_norm_h": self.config.ranged_max_norm_h,
                "stagnation_threshold": self.config.damage_stagnation_threshold,
                "max_recovery_attempts": self.config.max_recovery_attempts,
            },
        }


# =============================================================================
# Module Integrity & Contract Functions
# =============================================================================

def get_combat_capabilities() -> list:
    return [
        "combat",
        "skill_rotation",
        "diagnostic_snapshot",
        "engagement_range_optimization",
        "melee_ranged_profiles",
        "line_of_sight_obstacle_check",
        "geometric_ray_occlusion",
        "damage_stagnation_detection",
        "adaptive_obstruction_recovery",
    ]


def validate_combat_contract() -> bool:
    return bool(get_combat_capabilities())


def get_combat_baseline_metrics() -> dict:
    return {
        "module": "combat",
        "contract_valid": validate_combat_contract(),
        "capability_count": len(get_combat_capabilities()),
    }


def get_combat_readiness_status() -> dict:
    return {"module": "combat", "ready": validate_combat_contract(), "metrics": get_combat_baseline_metrics()}


def get_combat_audit_record() -> dict:
    return {"module": "combat", "audit_passed": validate_combat_contract(), "readiness": get_combat_readiness_status()}


def get_combat_operational_digest() -> dict:
    return {"module": "combat", "operational": validate_combat_contract(), "audit": get_combat_audit_record()}


def get_combat_telemetry_payload() -> dict:
    return {"module": "combat", "payload": get_combat_operational_digest()}


def get_combat_integrity_seal() -> dict:
    return {"module": "combat", "sealed": validate_combat_contract(), "telemetry": get_combat_telemetry_payload()}


def get_combat_compliance_certificate() -> dict:
    return {"module": "combat", "certified": validate_combat_contract(), "seal": get_combat_integrity_seal()}


def get_combat_readiness_probe() -> dict:
    return {"module": "combat", "ready": validate_combat_contract(), "certificate": get_combat_compliance_certificate()}


def get_combat_audit_summary() -> dict:
    return {"module": "combat", "audit_passed": validate_combat_contract(), "probe": get_combat_readiness_probe()}
