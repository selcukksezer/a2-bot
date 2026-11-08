# -*- coding: utf-8 -*-
"""
Agent W54 Verification Suite: Gather Combat Key Lock & 3-Stage Harvest Cycle Lock
Tests:
1. Combat keys (1..9, q, r, tab) and combat skill/attack methods are completely locked during gathering.
2. Entering gathering halts motion and immediately releases any held combat keys and mouse buttons.
3. Three-stage harvest cycle (1/3 -> 2/3 -> 3/3) keeps the bot locked on active_node until 3/3 completes.
4. Route step (_step_route_mode) between stages (casting & inter-harvest wait) does not leave the node or divert to combat.
5. Extraction failure retries the current stage without advancing stage count, leaving the node early, or leaking combat keys.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.state_machine import BotMode, BotState, BotWorker


class MockW54InputDriver:
    """Mock input driver tracking all key presses, releases, and mouse states."""

    def __init__(self, trusted: bool = True):
        self.pressed: List[str] = []
        self.released: List[str] = []
        self.holds: List[str] = []
        self.mouse_events: List[Tuple[str, str]] = []
        self._mouse_buttons_down = set()
        self._active_bot_keys = set()
        self._held_movement = set()
        self._trusted = trusted
        self.motion_released = False
        self.sent_gather_keys = 0

    def keyboard_inject_trusted(self) -> bool:
        return self._trusted

    def is_keyboard_trusted(self) -> bool:
        return self._trusted

    def movement_active(self) -> bool:
        return bool(self._held_movement)

    def press_key(self, key: str, *args, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
        return True

    def send_key(self, key: str) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
        return True

    def send_gather_key(self, *args, **kwargs) -> bool:
        self.sent_gather_keys += 1
        self.pressed.append("f")
        return True

    def key_down(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.holds.append(k)
        self._active_bot_keys.add(k)
        return True

    def key_up(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.released.append(k)
        self._active_bot_keys.discard(k)
        return True

    def release_key(self, key: str) -> bool:
        return self.key_up(key)

    def _hw_key_up(self, key: str) -> bool:
        return self.key_up(key)

    def _hw_mouse_down(self, button: str) -> None:
        self.mouse_events.append(("down", button))
        self._mouse_buttons_down.add(button)

    def _hw_mouse_up(self, button: str) -> None:
        self.mouse_events.append(("up", button))
        self._mouse_buttons_down.discard(button)

    def abort_motion(self) -> None:
        self.motion_released = True
        self._held_movement.clear()
        self._mouse_buttons_down.clear()

    def release_all_movement_keys(self) -> None:
        self.motion_released = True
        self._held_movement.clear()

    def end_path_step(self) -> None:
        self.motion_released = True
        self._held_movement.clear()


class MockW54Vision:
    def __init__(self):
        self.prompt_detected: bool = True
        self.prompt_conf: int = 92
        self.last_detected_gather_key: str = "f"
        self.inventory_open: bool = False
        self.map_open: bool = False

    def detect_gather_prompt(self, *args, **kwargs):
        if self.prompt_detected:
            return True, self.prompt_conf, (100, 100, 50, 20)
        return False, 0, None

    def is_inventory_open(self, *args, **kwargs) -> bool:
        return self.inventory_open

    def is_map_open(self, *args, **kwargs) -> bool:
        return self.map_open

    def is_game_window_blocking(self, *args, **kwargs) -> bool:
        return False

    def grab_screen(self, grayscale: bool = False) -> np.ndarray:
        if grayscale:
            return np.zeros((720, 1280), dtype=np.uint8)
        return np.zeros((720, 1280, 3), dtype=np.uint8)


class MockW54HUDReader:
    def __init__(self):
        self.is_cast_bar_present: bool = False
        self.cast_progress_pct: float = 0.0
        self.chat_extraction_failed: bool = False
        self.chat_inventory_full: bool = False

    def verify_gather_cast_bar(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        return self.is_cast_bar_present, self.cast_progress_pct

    def read_cast_bar_progress(self, frame: np.ndarray) -> float:
        return self.cast_progress_pct / 100.0 if self.is_cast_bar_present else 0.0

    def parse_chat_feedback_fast(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        return {
            "extraction_failed": self.chat_extraction_failed,
            "inventory_full": self.chat_inventory_full,
        }


class MockW54RouteManager:
    def __init__(self):
        self.gathered_nodes: List[Tuple[Any, float]] = []
        self.harvest_stages: List[Tuple[Any, int]] = []
        self.active_region: str = "Altgard"
        self.active_route: str = "W54Route"
        self.next_node_calls: int = 0

    def mark_node_gathered(self, node_id: Any, rest_minutes: float = 5.0):
        self.gathered_nodes.append((node_id, rest_minutes))

    def record_harvest_stage(self, node_id: Any, rest_duration_sec: int = 300):
        self.harvest_stages.append((node_id, rest_duration_sec))

    def get_next_ready_node(self, *args, **kwargs):
        self.next_node_calls += 1
        return {"id": 999, "resource_type": "Aria", "max_harvests": 3}


def create_w54_worker(config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[BotWorker, MockW54InputDriver]:
    config: Dict[str, Any] = {
        "test_mode": False,
        "gather_on_the_way": True,
        "rest_minutes": 5,
        "f_interval_min": 0.01,
        "f_interval_max": 0.02,
        "inter_harvest_delay_min": 0.02,
        "inter_harvest_delay_max": 0.04,
        "post_harvest_prompt_timeout": 1.0,
        "gather_cast_duration_sec": 0.2,
        "max_retries_per_harvest_stage": 2,
        "gather_key": "f",
        "pause_after_no_gather_min": 0,
        "break_every_min": 0,
        "fight_back": True,
        "route": {"fight_back": True},
        "combat_skill_keys": ["1", "2", "3", "4", "5", "q", "e", "r"],
    }
    if config_overrides:
        config.update(config_overrides)

    driver = MockW54InputDriver(trusted=True)
    vision = MockW54Vision()
    hud_reader = MockW54HUDReader()
    route_mgr = MockW54RouteManager()
    combat_rec = MagicMock()
    combat_rec.is_dead.return_value = False
    combat_rec.check_character_dead.return_value = False
    combat_rec.check_under_attack.return_value = (False, 1.0)
    combat_rec._detect_enemy_target_bar.return_value = False
    combat_rec.detect_enemy_target_bar.return_value = False
    combat_rec.in_combat = False

    worker = BotWorker(
        config=config,
        vision=vision,
        input_driver=driver,
        route_manager=route_mgr,
        combat_recovery=combat_rec,
        hud_reader=hud_reader,
    )
    worker._mode = BotMode.ROUTE_MODE.value
    worker._state = BotState.WALKING.value
    worker._paused = False
    worker.action_bus = None
    worker.human_timing = None
    worker.game_learner = None
    worker.perception_hub = None
    worker.reflex = None
    worker.explorer_controller = None
    worker.behavioral_guard = MagicMock()
    worker.behavioral_guard.check_prompt_stuck.return_value = (False, "")
    worker.panic_system = MagicMock()
    worker.dataset_factory = MagicMock()
    worker.telemetry = MagicMock()
    worker.economy_calculator = MagicMock()
    worker._next_f_interval = 0.01

    worker.active_node = {
        "id": 54,
        "resource_type": "Orichalcum Ore",
        "max_harvests": 3,
    }
    return worker, driver


class TestW54GatherThreeStageHarvestLock(unittest.TestCase):
    """5 unit tests verifying combat key lock during gathering and strict 1/3 -> 2/3 -> 3/3 node retention."""

    def test_1_combat_keys_completely_locked_during_gathering(self):
        """Test 1: While gathering is active, all combat keys and combat routines are strictly blocked."""
        worker, driver = create_w54_worker()
        now = time.time()

        # Enter gathering state
        worker._handle_route_gathering(conf=95, now=now)
        self.assertTrue(worker._is_gathering_active)
        self.assertEqual(worker.state, BotState.GATHERING.value)
        driver.pressed.clear()

        # Verify combat authorization gates are completely locked
        self.assertFalse(worker.is_combat_authorized("skill"))
        self.assertFalse(worker.is_combat_authorized("target"))
        self.assertFalse(worker.is_combat_authorized("attack"))

        # Attempt direct combat key presses and skill rotation methods
        for combat_key in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "r", "tab"):
            worker._press_key(combat_key)
        worker._cast_one_combat_skill("1")
        worker._cast_one_combat_skill("q")
        self.assertFalse(worker._send_configured_combat_skill())
        worker._press_combat_tab(soft_aim=True)
        worker._sync_combat_right_button(True)
        worker._sync_combat_left_button(True)

        # Simulate hostile attack during active gathering
        worker.combat_recovery.check_under_attack.return_value = (True, 0.55)
        worker._enemy_target_bar_visible = MagicMock(return_value=True)
        worker._note_player_hp_drop = MagicMock(return_value=True)
        diverted = worker._handle_combat_and_death()
        self.assertFalse(diverted, "Active gathering must not divert to combat")

        # Zero combat keys or mouse attack holds should have leaked
        self.assertEqual(driver.pressed, [], f"Combat keys leaked during gathering: {driver.pressed}")
        self.assertEqual(len(driver._mouse_buttons_down), 0)

    def test_2_entering_gathering_releases_held_combat_inputs_and_halts_motion(self):
        """Test 2: Entering gathering immediately releases any active combat keys/mouse buttons and halts motion."""
        worker, driver = create_w54_worker()
        driver._held_movement.add("w")
        driver._active_bot_keys.update({"1", "3", "q", "tab"})
        driver._mouse_buttons_down.update({"right", "left"})
        worker._mark_right_button(True)

        worker._handle_route_gathering(conf=92, now=time.time())

        self.assertTrue(driver.motion_released)
        self.assertEqual(len(driver._held_movement), 0)
        self.assertEqual(len(driver._active_bot_keys), 0)
        self.assertNotIn("right", driver._mouse_buttons_down)
        self.assertFalse(worker._right_button_is_tracked())
        for k in ("1", "3", "q", "tab"):
            self.assertIn(k, driver.released)

    def test_3_three_stage_harvest_cycle_1_to_2_to_3_does_not_leave_node_early(self):
        """Test 3: 1/3 -> 2/3 -> 3/3 harvest cycle retains active_node through stages 1 and 2, completing only at 3/3."""
        worker, driver = create_w54_worker()
        now = time.time()

        # --- STAGE 1/3 ---
        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 1)
        self.assertTrue(worker._stage_cast_in_progress)
        self.assertEqual(worker._current_node_harvest_stage, 0)

        # Complete Stage 1 cast (progress >= 90%)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=94, now=now + 0.05)

        # Verify Stage 1/3 finished, but node is NOT abandoned
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertEqual(worker.gathered_count, 1)
        self.assertTrue(worker._is_gathering_active)
        self.assertTrue(worker._waiting_for_next_stage_prompt)
        self.assertIsNotNone(worker.active_node, "Must NOT leave node after stage 1/3")
        self.assertEqual(worker.active_node["id"], 54)
        self.assertEqual(len(worker.route_manager.gathered_nodes), 0)

        # --- STAGE 2/3 ---
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.is_cast_bar_present = False
        worker.hud_reader.cast_progress_pct = 0.0

        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 2)
        self.assertTrue(worker._stage_cast_in_progress)

        # Complete Stage 2 cast
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 96.0
        worker._handle_route_gathering(conf=94, now=now + 0.05)

        # Verify Stage 2/3 finished, and node is STILL NOT abandoned
        self.assertEqual(worker._current_node_harvest_stage, 2)
        self.assertEqual(worker.gathered_count, 2)
        self.assertTrue(worker._is_gathering_active)
        self.assertTrue(worker._waiting_for_next_stage_prompt)
        self.assertIsNotNone(worker.active_node, "Must NOT leave node after stage 2/3")
        self.assertEqual(worker.active_node["id"], 54)
        self.assertEqual(len(worker.route_manager.gathered_nodes), 0)

        # --- STAGE 3/3 ---
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.is_cast_bar_present = False
        worker.hud_reader.cast_progress_pct = 0.0

        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 3)
        self.assertTrue(worker._stage_cast_in_progress)

        # Complete Stage 3 cast
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 98.0
        worker._handle_route_gathering(conf=94, now=now + 0.05)

        # Verify full 3/3 completion: node is now marked gathered and cleared
        self.assertEqual(worker._current_node_harvest_stage, 0)
        self.assertEqual(worker.gathered_count, 3)
        self.assertFalse(worker._is_gathering_active)
        self.assertIsNone(worker.active_node, "Active node must be cleared after 3/3 completion")
        self.assertEqual(len(worker.route_manager.harvest_stages), 3)
        self.assertEqual(len(worker.route_manager.gathered_nodes), 1)
        self.assertEqual(worker.route_manager.gathered_nodes[0][0], 54)

    def test_4_step_route_mode_stays_at_node_across_inter_harvest_waits_and_blocks_combat(self):
        """Test 4: _step_route_mode keeps bot at the current node even when prompt flickers off during casting/inter-stage wait."""
        worker, driver = create_w54_worker()
        select_spy = MagicMock()
        worker._select_and_navigate_next_node = select_spy

        for stage_idx in (1, 2, 3):
            # 1) Prompt visible -> presses F and starts stage cast
            worker.vision.prompt_detected = True
            worker.hud_reader.is_cast_bar_present = False
            worker.hud_reader.cast_progress_pct = 0.0
            worker._step_route_mode()

            # Once gathering is active, simulate hostile attack throughout casting and inter-stage wait
            worker.combat_recovery.check_under_attack.return_value = (True, 0.70)
            worker._enemy_target_bar_visible = MagicMock(return_value=True)
            worker._note_player_hp_drop = MagicMock(return_value=True)

            self.assertTrue(worker._is_gathering_active)
            self.assertTrue(worker._stage_cast_in_progress)
            self.assertEqual(worker.active_node["id"], 54)
            self.assertEqual(select_spy.call_count, 0)

            # 2) Prompt disappears while casting at 50% -> must stay at node and not press F again
            worker.vision.prompt_detected = False
            worker.hud_reader.is_cast_bar_present = True
            worker.hud_reader.cast_progress_pct = 50.0
            worker._step_route_mode()
            self.assertTrue(worker._is_gathering_active)
            self.assertEqual(worker.active_node["id"], 54)

            # 3) Cast reaches 95% -> finishes current stage
            worker.hud_reader.cast_progress_pct = 95.0
            worker._step_route_mode()

            if stage_idx < 3:
                # Between stages (1/3 -> 2/3, 2/3 -> 3/3): prompt still hidden before deadline
                self.assertEqual(worker._current_node_harvest_stage, stage_idx)
                self.assertTrue(worker._waiting_for_next_stage_prompt)
                self.assertIsNotNone(worker.active_node)
                worker._step_route_mode()
                self.assertIsNotNone(worker.active_node, f"Node must not be abandoned during wait after stage {stage_idx}/3")
                self.assertEqual(select_spy.call_count, 0)
                # Advance past inter-harvest cooldown so next stage can begin
                worker._inter_harvest_wait_until = time.time() - 0.01

        # After 3/3 completes, active_node is None and zero combat skills were pressed
        self.assertIsNone(worker.active_node)
        self.assertEqual(worker.gathered_count, 3)
        combat_keys = {"1", "2", "3", "4", "5", "q", "r", "tab"}
        leaked = [k for k in driver.pressed if k in combat_keys]
        self.assertEqual(leaked, [])

    def test_5_extraction_failure_retries_stage_without_early_node_departure_or_combat_leak(self):
        """Test 5: Extraction failure on stage 2/3 retries stage 2/3 without leaving the node or unlocking combat keys."""
        worker, driver = create_w54_worker({"max_retries_per_harvest_stage": 2})
        now = time.time()

        # Stage 1/3 succeeds
        worker._handle_route_gathering(conf=90, now=now)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)
        self.assertEqual(worker._current_node_harvest_stage, 1)

        # Start Stage 2/3
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.is_cast_bar_present = False
        worker.hud_reader.cast_progress_pct = 0.0
        worker._handle_route_gathering(conf=90, now=now)
        self.assertTrue(worker._stage_cast_in_progress)

        # Stage 2/3 cast finishes with chat reporting 'extraction_failed'
        worker.hud_reader.chat_extraction_failed = True
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)

        # Must remain at stage 1 (retrying stage 2/3), still gathering on node 54, combat still locked
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertEqual(worker._current_node_stage_retries, 1)
        self.assertEqual(worker.gathered_count, 1)
        self.assertTrue(worker._is_gathering_active)
        self.assertIsNotNone(worker.active_node)
        self.assertEqual(worker.active_node["id"], 54)
        self.assertFalse(worker.is_combat_authorized("skill"))
        worker._press_key("1")
        self.assertNotIn("1", driver.pressed)

        # Retry Stage 2/3 succeeds
        worker.hud_reader.chat_extraction_failed = False
        worker.hud_reader.is_cast_bar_present = False
        worker._handle_route_gathering(conf=90, now=now + 0.1)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.15)
        self.assertEqual(worker._current_node_harvest_stage, 2)
        self.assertIsNotNone(worker.active_node)

        # Stage 3/3 succeeds -> node completes
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.is_cast_bar_present = False
        worker._handle_route_gathering(conf=90, now=now)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)

        self.assertEqual(worker.gathered_count, 3)
        self.assertFalse(worker._is_gathering_active)
        self.assertIsNone(worker.active_node)


if __name__ == "__main__":
    unittest.main()
