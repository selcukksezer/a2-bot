# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Humanized Camera & Movement Smoothing (Agent G8)
Validates:
- Elimination of robotic snap movements via Bezier / quintic minimum jerk trajectories
- Character heading turns and camera pitch/yaw smoothing
- Wrist arc (orthogonal pitch drift) and natural micro-overshoot / settling
- Safe pitch clamping and deadband damping
- Clean RMB drag holding and guaranteed release
- Physical input abort resilience
"""

import math
import random
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

import core.windmouse_humanizer as wm
from core.windmouse_humanizer import (
    CameraMovementSmoother,
    CameraSmoothingConfig,
    PhysicalInputGuard,
    WindMouseEngine,
    compute_bearing_deg,
    generate_heading_turn_trajectory,
    generate_pitch_yaw_trajectory,
    generate_relative_camera_trajectory,
    normalize_angle_diff,
    quintic_ease_in_out,
    smooth_camera_nudge,
    smooth_camera_pitch_yaw,
    smooth_movement_heading_step,
    smooth_turn_heading,
    verify_trajectory_anti_snap,
)


class MockInputController:
    """Mock controller tracking relative moves and mouse button events."""

    def __init__(self):
        self.moves: List[Tuple[int, int]] = []
        self.mouse_events: List[Tuple[str, str]] = []

    def nudge_camera(self, dx: int, dy: int = 0) -> bool:
        self.moves.append((int(dx), int(dy)))
        return True

    def mouse_move_relative(self, dx: int, dy: int = 0) -> bool:
        # F-1301 / Karar 4 (R27 ile birlikte): donanım zincirli göreli hareket
        # yolu (Interception). R27 sonrası _resolve_relative_mouse_setter
        # önce bunu çözer; soft nudge_camera yoluna düşülmez. Mock, donanım
        # sürücüsünü taklit edip hareketi moves'a kaydeder.
        self.moves.append((int(dx), int(dy)))
        return True

    def _hw_mouse_down(self, button: str = "left") -> bool:
        self.mouse_events.append(("down", str(button)))
        return True

    def _hw_mouse_up(self, button: str = "left") -> bool:
        self.mouse_events.append(("up", str(button)))
        return True


# ============================================================================
# 1. MATHEMATICAL INVARIANTS & POLYNOMIAL PROFILE TESTS
# ============================================================================


def test_quintic_ease_in_out_boundary_and_derivatives():
    """Verify quintic polynomial boundary values and smooth acceleration profile."""
    assert quintic_ease_in_out(0.0) == 0.0
    assert quintic_ease_in_out(1.0) == 1.0
    assert quintic_ease_in_out(0.5) == 0.5
    assert quintic_ease_in_out(-0.5) == 0.0  # Clamped
    assert quintic_ease_in_out(1.5) == 1.0  # Clamped

    # Verify zero velocity at endpoints (smooth start and stop)
    eps = 1e-4
    vel_start = (quintic_ease_in_out(eps) - quintic_ease_in_out(0.0)) / eps
    vel_end = (quintic_ease_in_out(1.0) - quintic_ease_in_out(1.0 - eps)) / eps
    assert vel_start < 0.01, f"Start velocity too high: {vel_start}"
    assert vel_end < 0.01, f"End velocity too high: {vel_end}"


def test_normalize_angle_diff():
    """Verify shortest angular difference calculation within [-180°, +180°]."""
    assert normalize_angle_diff(90.0, 0.0) == 90.0  # Right 90°
    assert normalize_angle_diff(270.0, 0.0) == -90.0  # Left 90° (shortest path)
    assert normalize_angle_diff(10.0, 350.0) == 20.0  # Cross 0° clockwise
    assert normalize_angle_diff(350.0, 10.0) == -20.0  # Cross 0° counter-clockwise
    assert normalize_angle_diff(180.0, 0.0) in (180.0, -180.0)
    assert normalize_angle_diff(45.0, 45.0) == 0.0


def test_compute_bearing_deg():
    """Verify compass bearing calculation: North=0°, East=90°, South=180°, West=270°."""
    assert compute_bearing_deg((100, 100), (100, 50)) == 0.0  # Up / North
    assert compute_bearing_deg((100, 100), (150, 100)) == 90.0  # Right / East
    assert compute_bearing_deg((100, 100), (100, 150)) == 180.0  # Down / South
    assert compute_bearing_deg((100, 100), (50, 100)) == 270.0  # Left / West
    assert compute_bearing_deg((100, 100), (100, 100)) == 0.0  # Zero distance


# ============================================================================
# 2. TRAJECTORY GENERATION & EXACT DISPLACEMENT ACCUMULATION
# ============================================================================


@pytest.mark.parametrize(
    "total_dx,total_dy",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (15, -10),
        (-120, 0),
        (240, 45),
        (-360, -80),
        (720, 0),  # 180° turn (720px)
    ],
)
def test_trajectory_exact_sum_matching(total_dx, total_dy):
    """Verify that every generated trajectory sums up to exact requested pixel displacement."""
    smoother = CameraMovementSmoother()
    steps = smoother.generate_relative_camera_trajectory(total_dx, total_dy)

    if total_dx == 0 and total_dy == 0:
        assert steps == []
        return

    accum_x = sum(s[0] for s in steps)
    accum_y = sum(s[1] for s in steps)
    assert accum_x == total_dx, f"X sum mismatch: {accum_x} != {total_dx}"
    assert accum_y == total_dy, f"Y sum mismatch: {accum_y} != {total_dy}"


def test_wrist_arc_creates_transient_pitch_drift_and_returns_to_zero():
    """
    Biometric wrist arc: horizontal turns induce slight vertical wrist curve
    that naturally returns to the baseline.
    """
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(
        enable_wrist_arc=True,
        wrist_arc_ratio=0.04,
        enable_overshoot=False,
        tremor_amplitude_px=0.0,
    )
    # 250px horizontal turn with dy=0
    steps = smoother.generate_relative_camera_trajectory(250, 0, config=cfg)
    assert len(steps) > 10

    # Net vertical displacement must be exactly zero
    net_dy = sum(s[1] for s in steps)
    assert net_dy == 0

    # Intermediate vertical positions must have deviated (wrist arc)
    intermediate_y = []
    curr_y = 0
    for s in steps:
        curr_y += s[1]
        intermediate_y.append(curr_y)

    max_arc = max(abs(y) for y in intermediate_y)
    assert max_arc > 0, "Wrist arc did not create transient vertical displacement"
    assert intermediate_y[-1] == 0, "Wrist arc did not return to 0 baseline"


def test_overshoot_for_large_turns():
    """Turns >= 25° (>= 100px) exhibit natural human micro-overshoot and corrective settling."""
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(
        enable_overshoot=True,
        overshoot_threshold_deg=20.0,
        enable_wrist_arc=False,
        tremor_amplitude_px=0.0,
    )
    # Flaky-by-design deterministik kilidi (W3): üreteç overshoot'u
    # `random.random() < 0.85` kapısıyla açar
    # (core/windmouse_humanizer.py:969-972) — çağrıların ~%15'inde hiç
    # overshoot üretilmez ve aşağıdaki düzeltme assertion'ı o zaman
    # kaçınılmaz olarak düşer. Bu test ise overshoot'un VARLIĞINI
    # deterministik bir özellik olarak pinler; olasılıksal kapıyla
    # çelişir. Üretim davranışı değiştirilmeden (canlı botta %15
    # overshoot'suz varyasyon kasıtlı humanisation'dır) test, çağrıdan
    # hemen önce RNG'yi tohumlar: seed=1'de kapı çekilişi 0.134364
    # (< 0.85) ve ov_ratio 0.036186 -> ov_dx=11px olur; kapıdan sonraki
    # ilk random() çağrısı üreteçteki kapının kendisidir, arada RNG
    # tüketen kod yoktur (CameraMovementSmoother.__init__ deterministik).
    # Assertion'lar aynen korundu, gevşetilmedi.
    random.seed(1)
    steps = smoother.generate_relative_camera_trajectory(300, 0, config=cfg)
    assert len(steps) > 15

    # Check for presence of negative micro-steps near the end (corrective homing)
    late_steps_x = [s[0] for s in steps[-6:]]
    has_corrective_return = any(sx < 0 for sx in late_steps_x)
    assert has_corrective_return, "Expected overshoot corrective settling steps"
    assert sum(s[0] for s in steps) == 300


# ============================================================================
# 3. ANTI-SNAP VERIFICATION & ROBOTIC SNAP ELIMINATION
# ============================================================================


def test_verify_trajectory_anti_snap_passes_for_smooth_trajectory():
    """Generated trajectories must pass anti-snap verification."""
    smoother = CameraMovementSmoother()
    steps = smoother.generate_heading_turn_trajectory(angle_diff_deg=60.0)
    res = smoother.verify_trajectory_anti_snap(steps)

    assert res["is_smooth"] is True
    assert res["is_robotic_snap"] is False
    assert res["step_count"] >= 15
    assert res["has_acceleration"] is True
    assert res["has_deceleration"] is True
    assert res["max_single_step"] <= smoother.config.max_single_step_px


def test_verify_trajectory_anti_snap_catches_robotic_teleport():
    """Single-frame snap movements must be flagged as robotic snap."""
    smoother = CameraMovementSmoother()

    # Robotic single-step 240px jump
    robotic_snap = [(240, 0, 0.008)]
    res = smoother.verify_trajectory_anti_snap(robotic_snap)

    assert res["is_robotic_snap"] is True
    assert res["is_smooth"] is False


def test_no_single_step_exceeds_max_cap_even_on_360_turn():
    """Even for massive turns (360° = 1440px), individual steps remain capped (anti-snap)."""
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(max_single_step_px=24)
    steps = smoother.generate_relative_camera_trajectory(1440, 0, config=cfg)

    step_mags = [math.hypot(s[0], s[1]) for s in steps]
    max_step = max(step_mags)
    # Strict cap: no rounding slack. The quintic peak (~1.875x the mean step)
    # plus overshoot must stay within max_single_step_px; see F-1301/F-1440.
    # Karar 8: cap kalkmaz — üretim (R27, core/windmouse_humanizer.py)
    # 360° dönüşü cap'i aşmayan artımlı adımlara böler; bu test o davranışı
    # pinler (baz e9794df'de üretim düzeltmesi yokken fail verir).
    assert max_step <= cfg.max_single_step_px
    assert sum(s[0] for s in steps) == 1440


# ============================================================================
# 4. HEADING & PITCH/YAW INTEGRATION
# ============================================================================


def test_deadband_filtering_prevents_steering_shudder():
    """Angles below deadband_deg should return no turn steps (preventing jitter)."""
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(deadband_deg=2.5)

    # 1.5° is within 2.5° deadband
    steps = smoother.generate_heading_turn_trajectory(angle_diff_deg=1.5, config=cfg)
    assert steps == []

    # 3.0° is outside deadband -> generates steps
    steps2 = smoother.generate_heading_turn_trajectory(angle_diff_deg=3.0, config=cfg)
    assert len(steps2) > 0


def test_pitch_clamping_prevents_looking_at_sky_or_ground():
    """Pitch angle must be clamped within realistic physiological bounds."""
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(max_pitch_up_deg=40.0, max_pitch_down_deg=50.0)

    # The documented scale is 3.5 px/deg (CameraSmoothingConfig.deg_to_px_pitch
    # default): -40deg -> -140px, +50deg -> +175px. Expectations are literals,
    # not derived from the production constant, so the test cannot pass by
    # construction if the scale changes.
    # Looking up 80° -> clamped to -40°
    steps_up = smoother.generate_pitch_yaw_trajectory(yaw_deg=0.0, pitch_deg=-80.0, config=cfg)
    assert sum(s[1] for s in steps_up) == -140
    # At the clamp limit itself the result must be identical (clamp equivalence)
    steps_up_limit = smoother.generate_pitch_yaw_trajectory(yaw_deg=0.0, pitch_deg=-40.0, config=cfg)
    assert sum(s[1] for s in steps_up_limit) == -140

    # Looking down 90° -> clamped to +50°
    steps_down = smoother.generate_pitch_yaw_trajectory(yaw_deg=0.0, pitch_deg=90.0, config=cfg)
    assert sum(s[1] for s in steps_down) == 175
    steps_down_limit = smoother.generate_pitch_yaw_trajectory(yaw_deg=0.0, pitch_deg=50.0, config=cfg)
    assert sum(s[1] for s in steps_down_limit) == 175


def test_reversal_damping_on_rapid_direction_changes():
    """Rapid left-to-right reversal within reversal_damping_sec applies velocity damping."""
    smoother = CameraMovementSmoother()
    cfg = CameraSmoothingConfig(reversal_damping_sec=0.20, reversal_damping_factor=0.60)

    # Turn right at t=100.0
    d1 = smoother.calculate_turn_damping(angle_diff=25.0, now=100.0, config=cfg)
    assert d1["turn_direction"] == "right"
    assert d1["reversal_damped"] is False

    # Immediate turn left at t=100.05 (within 0.20s window)
    d2 = smoother.calculate_turn_damping(angle_diff=-25.0, now=100.05, config=cfg)
    assert d2["turn_direction"] == "left"
    assert d2["reversal_damped"] is True
    assert d2["damping_multiplier"] == 0.60


# ============================================================================
# 5. LIVE CONTROLLER EXECUTION TESTS
# ============================================================================


def test_smooth_turn_heading_execution():
    """smooth_turn_heading executes relative moves on controller and updates metrics."""
    driver = MockInputController()
    smoother = CameraMovementSmoother()

    ok = smoother.smooth_turn_heading(
        input_ctrl=driver,
        angle_diff_deg=45.0,
        sleep_fn=lambda dt: None,
    )
    assert ok is True
    assert len(driver.moves) >= 20

    sum_x = sum(m[0] for m in driver.moves)
    expected_x = int(round(45.0 * smoother.config.deg_to_px_yaw))
    assert sum_x == expected_x

    metrics = smoother.get_metrics()
    assert metrics["turns_completed"] == 1
    assert metrics["total_yaw_deg"] == 45.0


def test_smooth_turn_heading_rmb_drag_mode():
    """When use_rmb_drag=True, right mouse button is pressed, held, and released."""
    driver = MockInputController()
    smoother = CameraMovementSmoother()

    ok = smoother.smooth_turn_heading(
        input_ctrl=driver,
        angle_diff_deg=-30.0,
        use_rmb_drag=True,
        sleep_fn=lambda dt: None,
    )
    assert ok is True
    assert ("down", "right") in driver.mouse_events
    assert ("up", "right") in driver.mouse_events
    # Ensure correct ordering: down before moves, up after moves
    assert driver.mouse_events[0] == ("down", "right")
    assert driver.mouse_events[-1] == ("up", "right")


def test_abort_callback_stops_motion_and_releases_rmb():
    """If an abort callback triggers, movement stops immediately and RMB is released."""
    driver = MockInputController()
    smoother = CameraMovementSmoother()

    step_counter = [0]

    def trigger_abort():
        step_counter[0] += 1
        return step_counter[0] >= 5

    ok = smoother.smooth_turn_heading(
        input_ctrl=driver,
        angle_diff_deg=90.0,
        use_rmb_drag=True,
        abort_callback=trigger_abort,
        sleep_fn=lambda dt: None,
    )
    assert ok is False
    # Abort must stop the motion EARLY: at least one step ran, but strictly
    # fewer than the full uninterrupted 90-degree trajectory (46 steps), and
    # no more than the 5 steps the abort callback allows before it fires.
    # (No exact count: the count depends on where the abort check sits in the
    # step loop, which is an implementation detail.)
    full_trajectory = smoother.generate_heading_turn_trajectory(angle_diff_deg=90.0)
    assert 0 < len(driver.moves) <= 5 < len(full_trajectory)
    assert driver.mouse_events[-1] == ("up", "right")  # RMB cleaned up
    assert smoother.get_metrics()["aborts_count"] == 1


def test_smooth_movement_heading_step_during_walk():
    """Navigating towards waypoint while walking computes bearing and executes smooth turn."""
    driver = MockInputController()
    smoother = CameraMovementSmoother()

    # Character at (0, 0), moving North (heading=0.0). Target is at (100, 0) -> East (bearing=90.0)
    res = smoother.smooth_movement_heading_step(
        input_ctrl=driver,
        current_pos=(0.0, 0.0),
        next_waypoint=(100.0, 0.0),
        current_heading=0.0,
        sleep_fn=lambda dt: None,
    )
    assert res["turned"] is True
    assert res["target_bearing"] == 90.0
    assert res["angle_diff"] == 90.0
    assert res["estimated_heading"] == 90.0
    assert len(driver.moves) >= 20


def test_smooth_camera_nudge_replaces_robotic_snap():
    """smooth_camera_nudge breaks a raw pixel nudge into smooth Bezier steps."""
    driver = MockInputController()
    smoother = CameraMovementSmoother()

    ok = smoother.smooth_camera_nudge(
        input_ctrl=driver,
        dx=80,
        dy=-40,
        sleep_fn=lambda dt: None,
    )
    assert ok is True
    assert len(driver.moves) >= 10
    assert sum(m[0] for m in driver.moves) == 80
    assert sum(m[1] for m in driver.moves) == -40
    assert smoother.get_metrics()["nudges_completed"] == 1


# ============================================================================
# 6. ENGINE INTEGRATION & CAPABILITY ADVERTISING
# ============================================================================


def test_windmouse_engine_integration_and_capabilities():
    """WindMouseEngine exposes camera smoothing and advertises new capabilities."""
    engine = WindMouseEngine()
    caps = wm.get_windmouse_humanizer_capabilities()
    assert "camera_smoothing" in caps
    assert "bezier_heading_smoother" in caps
    assert "pitch_yaw_humanizer" in caps
    assert "anti_snap_movement" in caps

    profile = engine.get_humanizer_profile_summary()
    assert profile["camera_smoothing_ready"] is True
    assert profile["anti_snap_movement"] is True

    snapshot = engine.get_windmouse_humanizer_diagnostic_snapshot()
    assert snapshot["camera_smoothing"]["camera_smoothing_ready"] is True

    # Delegated methods test
    driver = MockInputController()
    ok = engine.smooth_turn_heading(driver, angle_diff_deg=20.0, sleep_fn=lambda dt: None)
    assert ok is True
    assert len(driver.moves) >= 5


def test_module_level_convenience_functions():
    """Module-level convenience functions route to singleton smoother."""
    driver = MockInputController()
    ok = smooth_turn_heading(driver, angle_diff_deg=25.0, sleep_fn=lambda dt: None)
    assert ok is True
    assert len(driver.moves) >= 5

    traj = generate_heading_turn_trajectory(30.0)
    assert len(traj) > 0
    assert verify_trajectory_anti_snap(traj)["is_smooth"] is True
