# -*- coding: utf-8 -*-
"""Focused gates for false combat, obstacle, and map actions. No GPU and no game."""

import hashlib
import json
import logging
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from core.input_driver import InputController
from core.omni_game_learner import OmniGameLearner
from core.state_machine import BotWorker

ROOT = Path(__file__).resolve().parent
REGION_FILE = ROOT / "routes" / "Altgard" / "region_data.json"


def _region_digest():
    data = REGION_FILE.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    nodes = parsed.get("nodes") or parsed.get("route") or []
    if isinstance(parsed, dict) and not nodes:
        for value in parsed.values():
            if isinstance(value, list):
                nodes = value
                break
    ids = []
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and "id" in node:
                ids.append(node["id"])
    return hashlib.sha256(data).hexdigest(), ids


class _Keys:
    def __init__(self, moving=False):
        self.pressed = []
        self.holds = []
        self.moving = moving
        self.path_step = False

    def movement_active(self):
        return bool(self.moving)

    def press_key(self, key, *args, **kwargs):
        self.pressed.append(str(key).lower())
        # B-02: üretim _driver_press_ok fail-closed çalışır — None/False
        # dönen sürücü "gönderilmedi" sayılır. Gerçek InputController
        # press_key float (>0.0 = gönderildi) döner; kayıt sürücüsü de
        # başarıyı bool True ile raporlar (state_machine.py _driver_press_ok
        # docstring). None bırakmak loot F'ini "gönderilemedi" saydırırdı.
        return True

    def hold_movement_key(self, key):
        self.holds.append(str(key).lower())
        self.moving = True
        return True

    def release_all_movement_keys(self):
        self.moving = False

    def end_path_step(self):
        self.moving = False
        self.path_step = False

    def begin_path_step(self):
        self.moving = True
        self.path_step = True

    def motion_aborted(self):
        return False


class _Recovery:
    def __init__(self):
        self.bar = False
        self.under = False
        self.hp = 1.0
        self.fight_calls = 0

    def check_character_dead(self, frame=None):
        return False

    def check_under_attack(self, frame=None):
        return (self.under, self.hp)

    def _detect_enemy_target_bar(self, frame):
        return self.bar

    def fight_back_step(self, input_ctrl=None, frame=None):
        self.fight_calls += 1


class _Routes:
    def __init__(self):
        self.failed = []

    def mark_node_failed(self, node_id, remove_after_failures=0):
        self.failed.append(node_id)


def _worker(moving=False):
    worker = BotWorker.__new__(BotWorker)
    try:
        from PyQt6.QtCore import QThread

        QThread.__init__(worker)
    except Exception:
        pass
    worker._lock = threading.RLock()
    worker._mode = "route"
    worker._state = "WAITING"
    worker._paused = False
    worker._state_enter_time = time.time()
    worker.config = {
        "test_mode": False,
        "fight_back": True,
        "rest_minutes": 5,
        "reclick_node_sec": 10,
        "time_to_reach_sec": 60,
    }
    worker._route_was_fighting = False
    worker._route_fight_had_signal = False
    worker._route_combat_confirmed = False
    worker._route_saw_target_bar = False
    worker._route_saw_hp_drop = False
    worker._route_prev_hp_ratio = None
    worker._route_last_loot_ts = 0.0
    worker._route_bar_since_loot = False
    worker._map_open_backoff_until = 0.0
    worker._map_open_fail_announced = False
    worker._last_nav_block = ""
    worker._still_obstacle_angles = []
    worker._last_obstacle_evade_ts = 0.0
    worker._evasion_sign_hist = []
    worker._evasion_locked_key = None
    worker._evasion_lock_until = 0.0
    worker.active_node = None
    worker._walk_start_time = 0.0
    worker.logs = []
    worker.input_driver = _Keys(moving=moving)
    worker.vision = None
    worker.navigator = None
    worker.route_manager = _Routes()
    worker.combat_recovery = _Recovery()
    worker.behavioral_guard = None
    worker.skill_bar_monitor = None
    worker.game_learner = None
    # BotWorker.__init__ sets this to None when the subsystem is unavailable;
    # _motion_should_stop() reads it directly, so the fixture must provide it.
    worker.focus_monitor = None
    worker._combat_was_targeting = False
    worker._combat_no_target_streak = 0
    worker._combat_last_tab_ts = 0.0
    worker._combat_right_button_down = False
    worker._combat_skill_used_at = {}
    worker._combat_skill_idx = 0
    worker.depth_avoider = None
    worker._last_bot_key_time = 0.0
    worker.msleep = lambda _ms: None
    worker._emit_log = lambda message: worker.logs.append(str(message))
    worker._emit_hud_snapshot = lambda **_kwargs: {}

    def _set_state(new_state):
        if str(new_state).upper() in ("WALKING", "YÜRÜYOR") and not worker._walking_is_real():
            new_state = "WAITING"
        worker._state = new_state

    worker._set_state_internal = _set_state
    worker._grab_frame = lambda grayscale=True: np.zeros((16, 16, 3), dtype=np.uint8)
    return worker


class FalseActionGateTests(unittest.TestCase):
    def test_no_bar_and_no_hp_drop_does_not_counterattack_or_press_f(self):
        worker = _worker()
        recovery = worker.combat_recovery
        recovery.bar = False
        recovery.hp = 1.0
        for i in range(6):
            recovery.under = i % 2 == 0
            self.assertFalse(worker._handle_combat_and_death())
        joined = "\n".join(worker.logs)
        self.assertNotIn("Saldırı altında", joined)
        self.assertNotIn("Karşı saldırı", joined)
        self.assertNotIn("Ganimet", joined)
        self.assertNotIn("Saldırgan", joined)
        self.assertNotIn("f", worker.input_driver.pressed)
        self.assertEqual(recovery.fight_calls, 0)
        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        self.assertNotEqual(worker._state, "FIGHTING")

    def test_stuck_combat_flag_without_signal_does_not_block_route_tick(self):
        worker = _worker()
        recovery = worker.combat_recovery
        recovery.bar = False
        recovery.under = True
        recovery.hp = 1.0
        recovery.in_combat = True
        worker._route_combat_confirmed = True
        worker._route_was_fighting = True
        worker._state = "FIGHTING"
        worker._failure_wait_until = 0.0
        worker._is_gathering_active = False
        worker.config["pause_after_no_gather_min"] = 0
        walked = []
        def _navigate():
            worker._clear_route_combat()
            walked.append("route")
        worker._select_and_navigate_next_node = _navigate

        worker._step_route_mode()

        self.assertEqual(walked, ["route"])
        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        self.assertFalse(recovery.in_combat)
        self.assertNotEqual(worker._state, "FIGHTING")
        self.assertEqual(recovery.fight_calls, 0)
        self.assertNotIn("f", worker.input_driver.pressed)

    def test_bar_seen_then_gone_loots_once(self):
        worker = _worker()
        worker._mode = "manual_record"
        recovery = worker.combat_recovery
        recovery.bar = True
        recovery.under = True
        recovery.hp = 1.0
        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): the bar must be seen on 2
        # consecutive ticks — tick 1 only primes the streak, tick 2
        # confirms the fight.
        # fix/v20r1-combat-gate (madde 4, commit 106bdde): bar TEK BAŞINA
        # artık FIGHTING'e sokmaz — bar penceresinde HP düşüşü veya ToT
        # aggro gerekir (hayalet-savaş: pasif mob barı + sabit HP).
        # Bar + (True, 1.0) sabit HP, madde 4'ün
        # test_two_consecutive_bar_frames_alone_do_not_enter_fighting
        # testinde 10 tick'te bile FIGHTING'e sokmuyor — bu testin eski
        # kurulumu tam o hayalet imzaydı. Gerçek savaş için tick 2'de
        # HP düşüşü verilir (1.0 -> 0.97; test_bar_plus_hp_drop_
        # enters_fighting deseni); loot-tek-F iddiası değişmedi.
        self.assertFalse(worker._handle_combat_and_death())
        recovery.hp = 0.97
        self.assertTrue(worker._handle_combat_and_death())
        self.assertTrue(worker._route_combat_confirmed)
        self.assertNotIn("f", worker.input_driver.pressed)

        recovery.bar = False
        recovery.under = False
        worker._handle_combat_and_death()
        self.assertEqual(worker.input_driver.pressed.count("f"), 1)
        self.assertEqual(sum("Ganimet" in line for line in worker.logs), 1)

        worker._handle_combat_and_death()
        recovery.under = True
        worker._handle_combat_and_death()
        self.assertEqual(worker.input_driver.pressed.count("f"), 1)
        self.assertEqual(sum("Ganimet" in line for line in worker.logs), 1)

    def test_still_character_repeating_angle_does_not_strafe(self):
        worker = _worker(moving=False)
        now = 10_000.0
        samples = (
            ("rock", 18.8),
            ("rock", 20.7),
            ("tree_trunk", 21.4),
            ("tree_trunk", 22.1),
            ("wall", 20.7),
        )
        for index, (kind, angle) in enumerate(samples):
            applied = worker._apply_active_obstacle_evasion(kind, angle, now + index * 3.0)
            self.assertFalse(applied)
        self.assertEqual(worker.input_driver.pressed, [])
        self.assertFalse(any("Engel Kaçınma" in line for line in worker.logs))

    def test_moving_character_close_obstacle_still_strafes(self):
        worker = _worker(moving=True)
        applied = worker._apply_active_obstacle_evasion("rock", 20.7, 20_000.0)
        self.assertTrue(applied)
        self.assertIn("w", worker.input_driver.holds)
        self.assertIn("d", worker.input_driver.holds)
        self.assertIn("space", worker.input_driver.pressed)
        self.assertTrue(any("W+D tutuldu" in line for line in worker.logs))

    def test_alternating_obstacle_angles_keep_one_strafe(self):
        worker = _worker(moving=True)
        now = 30_000.0
        for index, angle in enumerate((21.4, -8.5, 12.1, -10.3, -17.5)):
            worker._last_obstacle_evade_ts = 0.0
            self.assertTrue(
                worker._apply_active_obstacle_evasion("rock", angle, now + index * 2.0)
            )
        strafe_holds = [key for key in worker.input_driver.holds if key in ("a", "d")]
        self.assertGreaterEqual(len(strafe_holds), 3)
        self.assertEqual(len(set(strafe_holds[-3:])), 1)
        self.assertIn("w", worker.input_driver.holds)

    def test_release_one_movement_key_keeps_forward(self):
        driver = InputController(test_mode=True, use_faker_input=False)
        self.assertTrue(driver.hold_movement_key("w"))
        self.assertTrue(driver.hold_movement_key("d"))
        driver.release_movement_key("d")
        self.assertIn("w", driver._held_movement)
        self.assertNotIn("d", driver._held_movement)
        self.assertTrue(driver.movement_active())

    def test_map_open_failure_backs_off_and_keeps_node_40(self):
        before_hash, before_ids = _region_digest()
        self.assertIn(40, before_ids)

        worker = _worker(moving=False)
        opens = []

        def _fail_open():
            opens.append(time.time())
            return False

        worker._ensure_map_open = _fail_open
        worker._get_next_ready_node = lambda _rest: {"id": 40}

        warnings = []

        class _Warn(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = _Warn()
        logger = logging.getLogger("Gatherion.StateMachine")
        logger.addHandler(handler)
        try:
            worker._select_and_navigate_next_node()
            worker._select_and_navigate_next_node()
        finally:
            logger.removeHandler(handler)

        self.assertEqual(len(opens), 1)
        self.assertEqual(sum("Map did not open when navigating to node 40" in msg for msg in warnings), 1)
        self.assertEqual(sum("staying idle" in line for line in worker.logs), 0)
        self.assertEqual(sum("Not walking" in line for line in worker.logs), 0)
        self.assertEqual(worker.route_manager.failed, [])
        self.assertIn("w", worker.input_driver.holds)
        self.assertIsNotNone(worker.active_node)
        self.assertNotEqual(worker._state, "WAITING")
        self.assertTrue(worker._walking_is_real())
        self.assertNotIn("f", worker.input_driver.pressed)

        after_hash, after_ids = _region_digest()
        self.assertEqual(before_hash, after_hash)
        self.assertIn(40, after_ids)

    def test_route_tick_after_failed_map_open_does_not_stay_waiting(self):
        """Route mode, not user-paused: one tick after map open fails must leave Not walking / WAITING with no path step."""
        worker = _worker(moving=False)
        worker._mode = "route"
        worker._paused = False
        worker._state = "WAITING"
        worker._failure_wait_until = 0.0
        worker._last_gather_time = time.time()
        worker._is_gathering_active = False
        worker._gather_lost_frames = 0
        worker.config["pause_after_no_gather_min"] = 0
        worker.config["gather_on_the_way"] = False
        worker.active_node = {"id": 40, "map_x": 100.0}
        worker.active_node_distance_m = 12
        worker._walk_start_time = time.time()
        worker._last_reclick_time = time.time()
        worker._map_open_backoff_until = time.time() + 45.0
        worker._map_open_fail_announced = True
        worker._last_nav_block = "map_open"
        hud = []

        def _capture_hud(**kwargs):
            hud.append(dict(kwargs))
            return {}

        worker._emit_hud_snapshot = _capture_hud
        worker._deny_walk_hud("Map did not open")

        worker._step_route_mode()

        title = str(hud[-1].get("title") or "") if hud else ""
        subtitle = str(hud[-1].get("subtitle") or "") if hud else ""
        path_step = bool(worker.input_driver.path_step)
        stuck = (
            worker._state == "WAITING"
            and title == "Not walking"
            and subtitle == "Map did not open"
            and not path_step
        )
        self.assertFalse(worker.is_paused)
        self.assertEqual(worker.mode, "route")
        self.assertFalse(stuck)
        self.assertNotEqual(worker._state, "WAITING")
        self.assertNotEqual(title, "Not walking")
        self.assertTrue(path_step)
        self.assertTrue(worker._walking_is_real())
        self.assertEqual(worker.route_manager.failed, [])
        self.assertEqual(worker._extract_node_id(worker.active_node), 40)

    def _arm_route_tick(self, worker, vision):
        worker.vision = vision
        worker.navigator = None
        worker.overlay_rect_provider = None
        worker.biometric_clicker = None
        worker.windmouse = None
        worker._mode = "route"
        worker._paused = False
        worker._state = "WAITING"
        worker._failure_wait_until = 0.0
        worker._last_esc_press_time = 0.0
        worker._last_gather_time = time.time()
        worker._is_gathering_active = False
        worker._gather_lost_frames = 0
        worker.config["pause_after_no_gather_min"] = 0
        worker.config["gather_on_the_way"] = False
        worker.config["test_mode"] = False
        worker.config["show_overlay"] = False
        worker._get_next_ready_node = lambda _rest: {
            "id": 40,
            "map_x": 900,
            "map_y": 500,
            "distance_m": 18,
        }

    def test_route_tick_sends_map_key_then_clicks_node(self):
        """TOPLAMA, not paused, map closed: M goes out before any backoff, then the node is clicked."""
        worker = _worker(moving=False)

        class _Map:
            def __init__(self):
                self.open = False

            def is_map_open(self, frame=None):
                return self.open

            def locate_node_on_open_map(self, node, frame=None, **_kwargs):
                return (900, 500, 18, False)

        vision = _Map()
        events = []

        def press_key(key, *args, **kwargs):
            name = str(key).lower()
            events.append(("key", name))
            if name == "m":
                vision.open = not vision.open

        def toggle_map(frame=None):
            press_key("m")

        def click_map(x, y, button="left"):
            events.append(("click", int(x), int(y)))

        worker.input_driver.press_key = press_key
        worker.input_driver.toggle_map = toggle_map
        worker.input_driver.click_map = click_map
        worker.input_driver.focus_game_window = lambda: False
        self._arm_route_tick(worker, vision)

        worker._step_route_mode()

        keys = [item[1] for item in events if item[0] == "key"]
        clicks = [item for item in events if item[0] == "click"]
        self.assertIn("m", keys)
        self.assertEqual(keys[0], "m")
        self.assertTrue(clicks)
        self.assertLess(events.index(("key", "m")), events.index(clicks[0]))
        self.assertEqual(clicks[0], ("click", 900, 500))
        self.assertEqual(worker._map_open_backoff_until, 0.0)
        self.assertFalse(worker.is_paused)
        self.assertEqual(worker.mode, "route")
        self.assertNotEqual(worker._state, "WAITING")
        self.assertIn("w", worker.input_driver.holds)
        self.assertTrue(worker._walking_is_real())
        self.assertNotIn("f", keys)
        self.assertEqual(worker._extract_node_id(worker.active_node), 40)

    def test_failed_map_open_keeps_walking(self):
        """If M still does not open the map, the route tick must not sit in WAITING with movement released."""
        worker = _worker(moving=False)

        class _Closed:
            def is_map_open(self, frame=None):
                return False

        events = []

        def press_key(key, *args, **kwargs):
            events.append(("key", str(key).lower()))

        worker.input_driver.press_key = press_key
        worker.input_driver.toggle_map = lambda frame=None: press_key("m")
        worker.input_driver.focus_game_window = lambda: False
        self._arm_route_tick(worker, _Closed())

        worker._step_route_mode()

        keys = [item[1] for item in events if item[0] == "key"]
        self.assertEqual(keys.count("m"), 1)
        self.assertEqual(keys[0], "m")
        self.assertTrue(worker._map_open_fail_announced)
        self.assertFalse(worker.is_paused)
        self.assertEqual(worker.mode, "route")
        stuck_waiting = worker._state == "WAITING" and not worker._walking_is_real()
        self.assertFalse(stuck_waiting)
        self.assertNotEqual(worker._state, "WAITING")
        self.assertIn("w", worker.input_driver.holds)
        self.assertTrue(worker._walking_is_real())
        self.assertEqual(worker.route_manager.failed, [])
        self.assertEqual(worker._extract_node_id(worker.active_node), 40)
        self.assertNotIn("f", keys)

    def test_1440p_world_map_detected_without_english_header(self):
        """A native 2560x1440 map stays open after the 720p English header crop is gone."""
        import cv2

        from core.vision import VisionEngine

        engine = VisionEngine()
        engine.capture.screen_w = 8000
        engine.capture.screen_h = 8000

        def _load(name):
            return cv2.imdecode(np.fromfile(str(ROOT / "video_frames" / name), dtype=np.uint8), cv2.IMREAD_COLOR)

        world = cv2.resize(_load("frame_18s.jpg"), (2560, 1440))
        world[0 : int(1440 * 0.16), int(2560 * 0.12) : int(2560 * 0.88)] = (28, 28, 28)
        gameplay = cv2.resize(_load("frame_00s.jpg"), (2560, 1440))
        self.assertTrue(engine.is_map_open(world))
        self.assertFalse(engine.is_map_open(gameplay))

    def test_route_walk_tick_holds_movement_key_until_pause(self):
        """Route active and not paused: one walk tick holds W. A pause tick releases it."""
        before_hash, before_ids = _region_digest()
        self.assertIn(40, before_ids)

        worker = _worker(moving=False)
        driver = InputController(use_faker_input=False, test_mode=True)
        driver.set_route_active(True)
        worker.input_driver = driver
        worker._mode = "route"
        worker._paused = False
        worker._state = "WAITING"
        worker._failure_wait_until = 0.0
        worker._last_gather_time = time.time()
        worker._is_gathering_active = False
        worker._gather_lost_frames = 0
        worker.config["pause_after_no_gather_min"] = 0
        worker.config["gather_on_the_way"] = False
        worker.active_node = {"id": 40, "map_x": 100.0}
        worker.active_node_distance_m = 12
        worker._walk_start_time = time.time()
        worker._last_reclick_time = time.time()
        worker._map_open_backoff_until = time.time() + 45.0
        worker._map_open_fail_announced = True
        worker._last_nav_block = "map_open"
        worker._sync_input_driver_state = lambda: driver.set_route_active(
            (not worker._paused) and worker._mode == "route"
        )

        worker._deny_walk_hud("Map did not open")
        self.assertNotIn("w", driver._held_movement)

        worker._step_route_mode()

        self.assertFalse(worker.is_paused)
        self.assertEqual(worker.mode, "route")
        self.assertIn("w", driver._held_movement)
        self.assertTrue(driver.movement_active())
        self.assertNotEqual(worker._state, "WAITING")
        self.assertEqual(worker._extract_node_id(worker.active_node), 40)
        self.assertEqual(worker.route_manager.failed, [])

        worker._paused = True
        worker._step_route_mode()
        self.assertNotIn("w", driver._held_movement)
        self.assertFalse(driver.movement_active())

        driver.resume_motion()
        self.assertTrue(driver.hold_movement_key("w"))
        worker.pause("Paused by user")
        self.assertNotIn("w", driver._held_movement)
        self.assertFalse(driver.movement_active())
        self.assertTrue(driver.motion_aborted())

        driver.resume_motion()
        worker._paused = False
        self.assertTrue(driver.hold_movement_key("w"))
        worker.requestInterruption = lambda: None
        worker.isRunning = lambda: False
        worker.stop()
        self.assertNotIn("w", driver._held_movement)
        self.assertFalse(driver.movement_active())

        after_hash, after_ids = _region_digest()
        self.assertEqual(before_hash, after_hash)
        self.assertIn(40, after_ids)

    def test_selected_target_sends_skill_key_and_idle_does_not(self):
        from core.combat_recovery import CombatRecovery
        from core.skill_bar_reader import SkillBarReader

        known = {str(key).lower() for key in CombatRecovery.COMBAT_SKILL_KEYS}
        known.update(str(key).lower() for key in SkillBarReader.SKILL_SLOTS)

        targeted = _worker()
        targeted._mode = "combat"
        targeted.combat_recovery.bar = True
        targeted._step_combat_mode()
        sent = [key for key in targeted.input_driver.pressed if key in known]
        self.assertTrue(sent, targeted.input_driver.pressed)
        self.assertNotIn("f", targeted.input_driver.pressed)

        idle = _worker()
        idle._mode = "combat"
        idle.combat_recovery.bar = False
        idle.combat_recovery.under = True
        idle.combat_recovery.hp = 1.0
        idle._combat_was_targeting = False
        idle._route_saw_target_bar = False
        idle._step_combat_mode()
        idle_skills = [key for key in idle.input_driver.pressed if key in known]
        self.assertEqual(idle_skills, [])
        self.assertEqual(idle.combat_recovery.fight_calls, 0)

        configured = _worker()
        configured._mode = "combat"
        configured.config["combat_skill_keys"] = ["r", "e"]
        configured.combat_recovery.bar = True
        configured._step_combat_mode()
        # The first two presses are "t": the R/T equivalent of holding the
        # right mouse button, not rotation skills. The rotation itself must
        # use the configured keys, in order, starting with "r".
        rotation = [key for key in configured.input_driver.pressed if key in ("r", "e")]
        self.assertEqual(rotation, ["r", "e"])

        held = _worker()
        held._mode = "combat"
        held.combat_recovery.bar = False
        held.combat_recovery.hp = 1.0
        held._combat_was_targeting = True
        # Target was lost only just now: the 2.5 s target-persistence grace
        # still applies, so skills keep firing instead of the F+TAB death path.
        # (Without a recent timestamp the grace computes 999.0 s and the bot
        # loots with F and Tabs to the next target.)
        held._combat_last_target_ts = time.time()
        held._step_combat_mode()
        held_skills = [key for key in held.input_driver.pressed if key in known]
        self.assertTrue(held_skills, held.input_driver.pressed)

    def test_learner_skips_false_obstacle_and_path_frames(self):
        class _Actor:
            def __init__(self):
                self.moving = False
                self.state = "FIGHTING"
                self.mode = "route"
                self._route_fight_had_signal = False
                self._route_saw_target_bar = False
                self._route_combat_confirmed = False

            def _walking_is_real(self):
                return self.moving

        actor_view = _Actor()
        learner = OmniGameLearner(bot_worker=actor_view, save_screenshots=True, min_sample_interval_sec=0.0)
        saved = []

        def _capture(**kwargs):
            saved.append(kwargs.get("category"))

        learner._save_learning_sample = _capture
        # GameFrameGate rejects frames smaller than 60x60, so the samples
        # must use a textured frame at/above that size (uniform noise passes
        # the gate as a small crop with natural texture). Still and moving
        # frames use different seeds so the gate's dHash-novelty check also
        # accepts the second frame as a new sample.
        still = np.random.default_rng(7).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        still[0, 0] = (1, 2, 3)
        learner._evaluate_and_save_learning_samples(
            frame=still,
            now=time.time(),
            current_state="FIGHTING",
            current_mode="route",
            monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
            skill_info={"ready_skills": ["1"], "cooldown_skills": {}},
            obstacle_info={
                "obstacle_detected": True,
                "obstacle_type": "rock",
                "recommended_steer_deg": 20.7,
            },
            path_info={"is_walking": True, "current_state": "WALKING", "current_mode": "route"},
            vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
        )
        self.assertNotIn("obstacles", saved)
        self.assertNotIn("paths", saved)

        actor_view.moving = True
        learner._seen_dhashes.clear()
        moving = np.random.default_rng(11).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        moving[1, 1] = (4, 5, 6)
        learner._evaluate_and_save_learning_samples(
            frame=moving,
            now=time.time() + 5.0,
            current_state="WALKING",
            current_mode="route",
            monster_info={"has_target": False, "monster_died": False, "hp_percent": 0},
            skill_info={"ready_skills": ["1"], "cooldown_skills": {}},
            obstacle_info={
                "obstacle_detected": True,
                "obstacle_type": "rock",
                "recommended_steer_deg": 20.7,
            },
            path_info={"is_walking": True, "current_state": "WALKING", "current_mode": "route"},
            vitality_info={"player_hp_percent": 100, "player_mp_percent": 100},
        )
        self.assertIn("obstacles", saved)
        self.assertIn("paths", saved)

        learner.current_monster_targeted = True
        learner.last_monster_death_ts = 0.0
        actor_view.state = "WAITING"
        actor_view._route_fight_had_signal = False
        idle_death = learner._analyze_monster_and_health(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertFalse(idle_death["monster_died"])


if __name__ == "__main__":
    unittest.main()
