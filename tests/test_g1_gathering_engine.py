# -*- coding: utf-8 -*-
"""
Agent G1 Gathering Engine Verification Suite
============================================

Tests for:
1. 3x Harvest Cycle Tracking (1/3 -> 2/3 -> 3/3 and early depletion)
2. Interaction Prompts (F vs E key loop and dynamic dispatch)
3. Zero False F Presses (guarded during casting and inter-harvest delay)
4. Progress Bar Monitoring (cast progress reading via GameHUDReader and HUD updates)
5. Essence Extraction Failure Retry (chat feedback retry without advancing stage)
6. Inventory Full & Stuck Detection (clean pause without looping)
7. Node Completion Transition (cooldown marking, clearing active node, resuming route)
8. Motion Halting on Gather (releasing movement keys when entering gathering)
"""

import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.state_machine import BotMode, BotState, BotWorker


class MockInputDriver:
    def __init__(self):
        self.pressed_keys: List[str] = []
        self.sent_gather_keys: int = 0
        self.motion_released: bool = False

    def press_key(self, key: str, *args, **kwargs):
        self.pressed_keys.append(str(key).lower())

    def send_gather_key(self, *args, **kwargs):
        self.sent_gather_keys += 1
        self.pressed_keys.append("f")

    def release_all_movement_keys(self):
        self.motion_released = True

    def end_path_step(self):
        self.motion_released = True


class MockVision:
    def __init__(self):
        self.prompt_detected: bool = False
        self.prompt_conf: int = 88
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


class MockHUDReader:
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


class MockRouteManager:
    def __init__(self):
        self.gathered_nodes: List[Tuple[Any, float]] = []
        self.harvest_stages: List[Tuple[Any, int]] = []
        self.active_region: str = "Altgard"
        self.active_route: str = "TestRoute"

    def mark_node_gathered(self, node_id: Any, rest_minutes: float = 5.0):
        self.gathered_nodes.append((node_id, rest_minutes))

    def record_harvest_stage(self, node_id: Any, rest_duration_sec: int = 300):
        self.harvest_stages.append((node_id, rest_duration_sec))

    def get_next_ready_node(self, *args, **kwargs):
        return {"id": 101, "resource_type": "Orichalcum Ore", "max_harvests": 3}


def create_test_worker(config_overrides: Optional[Dict[str, Any]] = None) -> BotWorker:
    config: Dict[str, Any] = {
        "test_mode": True,
        "gather_on_the_way": True,
        "rest_minutes": 5,
        "inter_harvest_delay_min": 0.05,
        "inter_harvest_delay_max": 0.1,
        "post_harvest_prompt_timeout": 0.5,
        "gather_cast_duration_sec": 0.2,
        "max_retries_per_harvest_stage": 2,
        "gather_key": "auto",
        "pause_after_no_gather_min": 0,
        "fight_back": False,
    }
    if config_overrides:
        config.update(config_overrides)

    worker = BotWorker(config=config)
    worker.input_driver = MockInputDriver()
    worker.vision = MockVision()
    worker.hud_reader = MockHUDReader()
    worker.route_manager = MockRouteManager()
    worker.behavioral_guard = MagicMock()
    worker.behavioral_guard.check_prompt_stuck.return_value = (False, "")
    worker.panic_system = MagicMock()
    worker.dataset_factory = MagicMock()
    worker.telemetry = MagicMock()
    worker.economy_calculator = MagicMock()

    # Pre-populate sample node
    worker.active_node = {
        "id": 42,
        "resource_type": "Aria",
        "max_harvests": 3,
    }
    return worker


class TestG1GatheringEngine(unittest.TestCase):

    def test_motion_halted_upon_gathering(self):
        """Verify that movement is halted as soon as gathering interaction begins."""
        worker = create_test_worker()
        worker.vision.prompt_detected = True
        halted = []
        worker._halt_motion = lambda: halted.append(True)

        worker._handle_route_gathering(conf=92, now=time.time())
        self.assertTrue(len(halted) > 0, "Motion must be halted when entering gathering")
        self.assertTrue(worker._is_gathering_active)
        self.assertEqual(worker.state, BotState.GATHERING.value)

    def test_dynamic_key_selection_f_vs_e(self):
        """Verify dynamic key selection picks 'e' when [E] prompt is detected, and 'f' otherwise."""
        worker = create_test_worker({"gather_key": "auto"})

        # Case 1: Vision detects 'e'
        worker.vision.last_detected_gather_key = "e"
        worker._last_detected_prompt_key = "e"
        self.assertEqual(worker._get_active_interaction_key(), "e")

        # Case 2: Vision detects 'f'
        worker.vision.last_detected_gather_key = "f"
        worker._last_detected_prompt_key = "f"
        self.assertEqual(worker._get_active_interaction_key(), "f")

        # Case 3: Explicit config overrides prompt detection
        worker.config["gather_key"] = "e"
        self.assertEqual(worker._get_active_interaction_key(), "e")

        worker.config["gather_key"] = "f"
        self.assertEqual(worker._get_active_interaction_key(), "f")

    def test_zero_false_f_presses_during_casting(self):
        """Verify that F/E is NEVER pressed while casting progress is active."""
        worker = create_test_worker()
        worker.vision.prompt_detected = True
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 45.0  # 45% cast filled

        now = time.time()
        # Initial enter -> presses key once to start casting
        worker._handle_route_gathering(conf=90, now=now)
        initial_f_count = worker.f_pressed_count
        self.assertEqual(initial_f_count, 1)
        self.assertTrue(worker._stage_cast_in_progress)

        # Subsequent ticks while casting is at 55%, 70%, 85%
        for progress in (55.0, 70.0, 85.0):
            worker.hud_reader.cast_progress_pct = progress
            worker._handle_route_gathering(conf=90, now=now + 0.1)
            # F must NOT be pressed again while casting!
            self.assertEqual(worker.f_pressed_count, initial_f_count, "False F pressed during cast!")

    def test_progress_bar_completion_advances_stage(self):
        """Verify that cast bar reaching >= 90% completes stage 1 and advances to stage 2."""
        worker = create_test_worker()
        worker.vision.prompt_detected = True
        now = time.time()

        # Start stage 1
        worker._handle_route_gathering(conf=95, now=now)
        self.assertEqual(worker._current_node_harvest_stage, 0)
        self.assertTrue(worker._stage_cast_in_progress)

        # Cast bar reaches 95%
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=95, now=now + 0.1)

        # Stage 1 should be finished! Stage count becomes 1, waiting for next stage prompt
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertEqual(worker.gathered_count, 1)
        self.assertFalse(worker._stage_cast_in_progress)
        self.assertTrue(worker._waiting_for_next_stage_prompt)
        worker.dataset_factory.on_gather_success.assert_called_with(
            material_name="Aria",
            harvest_stage=1,
            max_harvests=3,
        )

    def test_full_3x_harvest_cycle(self):
        """Simulate a full 3-harvest cycle on a single node (1/3 -> 2/3 -> 3/3 -> completion)."""
        worker = create_test_worker({"inter_harvest_delay_min": 0.01, "inter_harvest_delay_max": 0.02})
        worker.vision.prompt_detected = True
        now = time.time()

        # Stage 1: interact -> cast -> finish
        worker._handle_route_gathering(conf=90, now=now)
        self.assertEqual(worker.f_pressed_count, 1)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertTrue(worker._waiting_for_next_stage_prompt)

        # Inter-harvest pause elapses
        now = worker._inter_harvest_wait_until + 0.01
        worker.hud_reader.is_cast_bar_present = False
        worker.hud_reader.cast_progress_pct = 0.0

        # Stage 2: interact -> cast -> finish
        worker._handle_route_gathering(conf=90, now=now)
        self.assertEqual(worker.f_pressed_count, 2)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)
        self.assertEqual(worker._current_node_harvest_stage, 2)
        self.assertTrue(worker._waiting_for_next_stage_prompt)

        # Inter-harvest pause elapses
        now = worker._inter_harvest_wait_until + 0.01
        worker.hud_reader.is_cast_bar_present = False
        worker.hud_reader.cast_progress_pct = 0.0

        # Stage 3: interact -> cast -> finish
        worker._handle_route_gathering(conf=90, now=now)
        self.assertEqual(worker.f_pressed_count, 3)
        worker.hud_reader.is_cast_bar_present = True
        worker.hud_reader.cast_progress_pct = 95.0
        worker._handle_route_gathering(conf=90, now=now + 0.05)

        # Node should now be FULLY completed and depleted!
        self.assertEqual(worker._current_node_harvest_stage, 0)
        self.assertFalse(worker._is_gathering_active)
        self.assertIsNone(worker.active_node, "Active node must be None after 3x cycle completes")
        self.assertEqual(worker.gathered_count, 3)
        self.assertEqual(len(worker.route_manager.gathered_nodes), 1)
        self.assertEqual(worker.route_manager.gathered_nodes[0][0], 42)

    def test_early_node_depletion_timeout(self):
        """Verify that if a node only has 1 harvest and prompt does not reappear, it completes gracefully."""
        worker = create_test_worker({"post_harvest_prompt_timeout": 0.2})
        now = time.time()

        # Harvest stage 1 completes
        worker._handle_route_gathering(conf=90, now=now)
        worker._on_harvest_stage_finished(now)
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertTrue(worker._waiting_for_next_stage_prompt)

        # Time machine / world simulation: prompt does not reappear
        worker.vision.prompt_detected = False

        # Next route tick after deadline:
        worker.resume()
        worker._post_harvest_prompt_deadline = now - 0.01  # Force deadline passed
        worker._step_route_mode()

        # Should have recognized early depletion and closed the node cleanly
        self.assertFalse(worker._is_gathering_active)
        self.assertIsNone(worker.active_node)
        self.assertEqual(len(worker.route_manager.gathered_nodes), 1)

    def test_essence_extraction_failed_retry_preserves_stage(self):
        """Verify that chat feedback 'Essence Extraction failed' retries stage without advancing count."""
        worker = create_test_worker({"max_retries_per_harvest_stage": 2})
        worker.config["test_mode"] = False  # Enable chat feedback check
        now = time.time()

        # Start stage 1
        worker._handle_route_gathering(conf=90, now=now)
        self.assertTrue(worker._stage_cast_in_progress)

        # Chat reports extraction failure
        worker.hud_reader.chat_extraction_failed = True
        worker._on_harvest_stage_finished(now)

        # Stage count must STILL be 0 (did not falsely advance)!
        self.assertEqual(worker._current_node_harvest_stage, 0)
        self.assertEqual(worker._current_node_stage_retries, 1)
        self.assertFalse(worker._stage_cast_in_progress)
        self.assertEqual(worker.gathered_count, 0)

        # Second try succeeds
        worker.hud_reader.chat_extraction_failed = False
        worker._on_harvest_stage_finished(now + 1.0)
        self.assertEqual(worker._current_node_harvest_stage, 1)
        self.assertEqual(worker.gathered_count, 1)

    def test_inventory_full_pauses_bot(self):
        """Verify that when inventory/cube is detected as full, the bot pauses immediately without pressing F."""
        worker = create_test_worker()
        worker.vision.prompt_detected = True
        worker.vision.inventory_open = True  # Inventory modal window detected

        paused = []
        worker.pause = lambda reason: paused.append(reason)

        worker._handle_route_gathering(conf=90, now=time.time())
        self.assertEqual(len(paused), 1, "Bot must pause when inventory is full")
        self.assertEqual(worker.f_pressed_count, 0, "Must not press F when inventory is full")


if __name__ == "__main__":
    unittest.main()
