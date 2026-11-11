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

from core.input_driver import HARDWARE_SCANCODES, InputController
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
        # Y2/Y3: üretim kapısı aşamayı ancak barın ARDIŞIK okumalarda
        # yükseldiği kanıtlanırsa bitirir (_stage_cast_rose). Tek karede
        # %95 görmek yetmez — dolu başlayan statik bir bar da %95
        # görünür. Dizi ayarlıysa verify_gather_cast_bar her çağrıda
        # sıradaki değeri döner (son değer tekrarlanır); ayarlı değilse
        # tek değerli eski davranış aynen sürer.
        self.cast_progress_sequence: Optional[List[float]] = None
        self._cast_seq_idx: int = 0

    def set_cast_bar_rising(self, pcts: List[float]) -> None:
        """Cast barını yükselen bir diziyle simüle et (Y2/Y3 kanıtı)."""
        self.cast_progress_sequence = [float(p) for p in pcts]
        self._cast_seq_idx = 0
        self.is_cast_bar_present = True

    def hide_cast_bar(self) -> None:
        """Cast barını gizle (dizi de sıfırlanır)."""
        self.is_cast_bar_present = False
        self.cast_progress_sequence = None
        self._cast_seq_idx = 0

    def _next_cast_pct(self) -> float:
        if self.cast_progress_sequence:
            idx = min(self._cast_seq_idx, len(self.cast_progress_sequence) - 1)
            pct = self.cast_progress_sequence[idx]
            self._cast_seq_idx += 1
            return float(pct)
        return self.cast_progress_pct

    def verify_gather_cast_bar(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        if not self.is_cast_bar_present:
            return False, 0.0
        return True, self._next_cast_pct()

    def read_cast_bar_progress(self, frame: np.ndarray) -> float:
        if not self.is_cast_bar_present:
            return 0.0
        return self._next_cast_pct() / 100.0

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


class MockW54FocusMonitor:
    """Focus-monitor double: the game window is always in the foreground.

    F-0180/F-0373: _step_route_mode() starts with _check_game_focus(),
    which fail-closes (returns False, tick paused) when focus_monitor
    is None on a real __init__-built worker with test_mode=False — the
    W54 worker has no focus monitor on this VM, so the first
    _step_route_mode() call in test_4 returned early and gathering
    never started (AssertionError: False is not true). The double
    supplies the missing monitor with the real
    ForegroundFocusGuard.is_game_window_focused() signature
    (allow_gatherion_ui kwarg, (focused, reason) tuple) and reports
    the game focused, keeping test_mode=False (live-mode semantics).
    Harness-only: no production code is touched, no assertion is
    relaxed, and only the _step_* entry points consult the monitor,
    so the other tests in this file are unaffected.
    """

    def __init__(self, focused: bool = True):
        self.focused = focused

    def is_game_window_focused(self, allow_gatherion_ui: bool = True):
        if self.focused:
            return True, ""
        return False, "game window not focused (mock)"


def _w54_base_config(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fresh base config for a W54 test worker (new dict every call)."""
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
    return config


def _attach_w54_doubles(
    worker: BotWorker, config_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[BotWorker, MockW54InputDriver]:
    """(Re)attach fresh mock doubles + a fresh config to a worker.

    Mirrors what create_w54_worker does after BotWorker(...) — fresh
    input/vision/HUD/route/combat mocks, fresh MagicMock subsystems,
    running route-mode state — but does NOT rebuild the worker itself,
    so it is the per-test reset used with the class-shared worker (see
    TestW54GatherThreeStageHarvestLock.setUpClass).
    """
    worker.config = _w54_base_config(config_overrides)
    worker.cfg = worker.config
    driver = MockW54InputDriver(trusted=True)
    worker.input_driver = driver
    worker.driver = driver
    worker.vision = MockW54Vision()
    worker.hud_reader = MockW54HUDReader()
    worker.route_manager = MockW54RouteManager()
    worker.routes = worker.route_manager
    worker.route_mgr = worker.route_manager
    # F-0180/F-0373 harness: _step_route_mode() fail-closes when the
    # worker has no focus monitor (see MockW54FocusMonitor). test_mode
    # stays False; production code is untouched.
    worker.focus_monitor = MockW54FocusMonitor()
    combat_rec = MagicMock()
    combat_rec.is_dead.return_value = False
    combat_rec.check_character_dead.return_value = False
    combat_rec.check_under_attack.return_value = (False, 1.0)
    combat_rec._detect_enemy_target_bar.return_value = False
    combat_rec.detect_enemy_target_bar.return_value = False
    combat_rec.in_combat = False
    worker.combat_recovery = combat_rec
    worker.combat = combat_rec

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


def _finish_stage_with_rising_bar(
    worker: BotWorker,
    conf: int,
    now: float,
    pcts: Tuple[float, ...] = (25.0, 60.0, 95.0),
) -> None:
    """Drive _handle_route_gathering until the Y2/Y3 rising-bar gate is satisfied.

    Production only counts a stage when the cast bar is SEEN RISING across
    consecutive reads (_stage_cast_rose) before it reaches >= 90% — a single
    95% frame is not proof (a static full target bar also reads 95%).
    Each _handle_route_gathering call is one observation, so the rising
    sequence is fed one value per call; the last value repeats if read again.
    The cast must already be in progress (F pressed) when this is called.
    """
    worker.hud_reader.set_cast_bar_rising(list(pcts))
    for i in range(len(pcts)):
        worker._handle_route_gathering(conf=conf, now=now + 0.05 * (i + 1))


def create_w54_worker(config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[BotWorker, MockW54InputDriver]:
    # NOTE: BotWorker.__init__ is very expensive on this VM (~2-10s: it
    # constructs ~40 optional subsystems incl. ONNXYolo3DDetector ~4s and
    # AutoLabelingDatasetFactory ~2s, measured 2026-11-08). Building one
    # per test made this file ~24s for 5 tests. The class below builds
    # ONE worker in setUpClass and resets it per test instead; this
    # function stays as the single-construction entry point for that.
    #
    # Construction is also trimmed: the subsystems listed here are
    # overwritten with mocks by _attach_w54_doubles right after
    # construction, so no test can ever observe the real ones — building
    # them is pure waste (AutoLabelingDatasetFactory alone is ~2.1s).
    # Everything else is built for real, because the gather/combat-lock
    # paths under test can reach it.
    from unittest.mock import patch

    _stubbed = {
        ("dataset_factory.factory_controller", "AutoLabelingDatasetFactory"),
        ("core.telemetry", "TelemetryEngine"),
        ("core.behavioral_guard", "BehavioralStealthGuard"),
        ("core.opsec_crypto", "PanicAlertSystem"),
        ("core.economy_broker", "SessionProfitSimulator"),
    }
    real_init = BotWorker._init_optional_subsystem

    def _patched_init(self: Any, module_name: str, class_name: str, *args: Any, **kwargs: Any) -> Any:
        if (module_name, class_name) in _stubbed:
            return MagicMock(name=f"stubbed.{module_name}.{class_name}")
        return real_init(self, module_name, class_name, *args, **kwargs)

    # The mocks are passed into __init__ directly, so construction needs
    # the same doubles _attach_w54_doubles builds per test; build a
    # throwaway set here and let _attach replace it with a fresh set.
    with patch.object(BotWorker, "_init_optional_subsystem", _patched_init):
        worker = BotWorker(
            config=_w54_base_config(config_overrides),
            vision=MockW54Vision(),
            input_driver=MockW54InputDriver(trusted=True),
            route_manager=MockW54RouteManager(),
            combat_recovery=MagicMock(),
            hud_reader=MockW54HUDReader(),
        )
    return _attach_w54_doubles(worker, config_overrides)


def _real_interception_controller_or_none() -> Optional[InputController]:
    """Gerçek InputController kur; Interception köprüsü canlı değilse None dön.

    P2-5/K-22: mock `True` driver yerine gerçek HID yolu. Köprü
    (InterceptionBridge) `available` değilse None döner — çağıran test
    skip eder, sahte True ile geçmez. Kurulum istisnası da None sayılır
    (bu VM'de interception.dll yok; sürücüsüz PC'de de yok).
    """
    try:
        ctrl = InputController(test_mode=False, use_faker_input=False, use_interception=True)
    except Exception:
        return None
    bridge = getattr(ctrl, "_interception", None)
    if bridge is None or not bool(getattr(bridge, "available", False)):
        try:
            ctrl.close()
        except Exception:
            pass
        return None
    return ctrl


class TestW54GatherThreeStageHarvestLock(unittest.TestCase):
    """5 unit tests verifying combat key lock during gathering and strict 1/3 -> 2/3 -> 3/3 node retention."""

    @classmethod
    def setUpClass(cls):
        # One expensive BotWorker construction shared by all 5 tests.
        cls.worker, cls.driver = create_w54_worker()
        # Snapshot of every instance attribute right after construction
        # (state fields, counters, mode/pause flags, locks, subsystems).
        cls._worker_snapshot = dict(cls.worker.__dict__)

    def _fresh_worker(
        self, config_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[BotWorker, MockW54InputDriver]:
        """Return the shared worker reset to post-construction state.

        Restores every instance attribute from the setUpClass snapshot
        (deleting attributes a previous test added, e.g. monkeypatched
        `_select_and_navigate_next_node` / `_enemy_target_bar_visible`),
        then re-attaches fresh mock doubles and a fresh config. The
        result is observationally identical to a freshly constructed
        worker for everything these tests assert (verified: same 4/5
        pass + the same 1 pre-existing failure, ~3x faster).
        """
        worker = self.worker
        for key, value in self._worker_snapshot.items():
            worker.__dict__[key] = value
        for key in list(worker.__dict__):
            if key not in self._worker_snapshot:
                del worker.__dict__[key]
        return _attach_w54_doubles(worker, config_overrides)

    def test_1_combat_keys_completely_locked_during_gathering(self):
        """Test 1: While gathering is active, all combat keys and combat routines are strictly blocked."""
        # fight_stop_gather_sec=0: "toplamayı hiç bölme" demektir.
        # ESKİ BULGU (10 Kasım, hedefli koşuyla doğrulandı): fixture 0
        # YETMİYORDU — core/state_machine.py `int(_get_cfg(...) or 3)`
        # ifadesi açık 0'ı da 3'e çeviriyordu (`0 or 3` == 3).
        # DÜZELTME (F-23-05, commit 7dbecc2 — v17 tabanında var):
        # state_machine.py:4874-4878 artık yalnızca None varsayılana
        # düşer (`3 if _stop_gather_raw is None else int(...)`);
        # _get_cfg açık 0'ı döndürür (None kontrolü), şema
        # _clamp(0, 0, 60, 3) 0'ı korur ve tek okuma noktası burasıdır.
        # Bu testin assertion'ları değiştirilmedi — doğru davranışı
        # (0 = asla savaşa sapma) kilitler. Yeşil olduğu hedefli
        # koşuyla ayrıca doğrulanmalıdır (bu dalda test koşturulmadı).
        worker, driver = self._fresh_worker({"fight_stop_gather_sec": 0})
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

        # Simulate hostile attack during active gathering. F-1516: the attack
        # mocks must NOT be set up here — production short-circuits at the
        # gathering lock (fight_stop_gather_sec <= 0) and returns False before
        # ever consulting check_under_attack / the target bar / HP drop, so
        # any such mock setup would be dead. Prove the short-circuit instead:
        worker.combat_recovery.check_under_attack.reset_mock()
        diverted = worker._handle_combat_and_death()
        self.assertFalse(diverted, "Active gathering must not divert to combat")
        worker.combat_recovery.check_under_attack.assert_not_called()

        # Zero combat keys or mouse attack holds should have leaked
        self.assertEqual(driver.pressed, [], f"Combat keys leaked during gathering: {driver.pressed}")
        self.assertEqual(len(driver._mouse_buttons_down), 0)

    def test_1b_real_interception_dispatch_single_fire(self):
        """P2-5/K-22: combat tuşu ('1') gerçek Interception köprüsünden tek atış gider.

        Mock driver her zaman True döner; bu test gerçek InputController +
        gerçek InterceptionBridge kurar (C8'in MagicMock köprüsünden farkı:
        köprünün `available` bayrağı gerçektir — DLL yüklü + context açık).
        Sürücü yoksa SKIP (sahte geçme yok). Çekirdek `interception_send`
        çağrısı kayda alınıp çalıştırılmaz — fiziksel tuş makineden çıkmaz
        (test, odaklı pencerede/oyunda gerçek '1' basamaz); yönlendirme,
        scancode ve tek-atış kanıtı gerçektir. Yumuşak yedekler hiç
        çağrılmamalı.
        """
        ctrl = _real_interception_controller_or_none()
        if ctrl is None:
            self.skipTest(
                "Interception sürücüsü yok (interception.dll + kernel driver "
                "kurulu değil) — gerçek HID yolu çalıştırılamadı, sahte True sayılmadı."
            )
        bridge = ctrl._interception
        calls: List[Tuple[int, bool, bool]] = []
        orig_send_key = bridge.send_key

        def _record_only(scancode: int, is_down: bool, extended: bool = False) -> bool:
            calls.append((int(scancode), bool(is_down), bool(extended)))
            return True

        try:
            bridge.send_key = _record_only
            ctrl._sendinput_key = MagicMock()
            ctrl._keybd_event_key = MagicMock()
            ctrl._post_key_to_game = MagicMock()
            self.assertIn("Interception", ctrl.active_backend_name)
            self.assertTrue(ctrl.keyboard_inject_trusted)
            self.assertTrue(ctrl.key_down("1"))
            self.assertTrue(ctrl.key_up("1"))
            self.assertEqual(
                calls,
                [
                    (int(HARDWARE_SCANCODES["1"]), True, False),
                    (int(HARDWARE_SCANCODES["1"]), False, False),
                ],
                f"Interception köprüsüne tek atış down/up gitmeliydi: {calls}",
            )
            ctrl._sendinput_key.assert_not_called()
            ctrl._keybd_event_key.assert_not_called()
            ctrl._post_key_to_game.assert_not_called()
        finally:
            bridge.send_key = orig_send_key
            try:
                ctrl.close()
            except Exception:
                pass

    def test_1c_real_interception_driver_combat_lock(self):
        """P2-5/K-22: gathering kilidi GERÇEK Interception sürücüsü takılıyken de tutar.

        Worker'a mock yerine gerçek InputController takılır; combat
        kapıları kilitliyken hiçbir combat tuşu çekirdek yoluna (köprü
        send_key) ulaşmamalı — sıfır çağrı assert edilir. Sürücü yoksa
        SKIP. `_handle_route_gathering` çağrılmaz: gerçek sürücüyle
        gerçek F basardı; gathering durumu doğrudan kurulur. Aynı
        nedenle `_sync_combat_left_button(True)` bu varyantta çağrılmaz
        (bırakma yolu sürücüden gerçek R key-up gönderir); o yol
        mock'lu test_1'de kapsanır.
        """
        ctrl = _real_interception_controller_or_none()
        if ctrl is None:
            self.skipTest(
                "Interception sürücüsü yok (interception.dll + kernel driver "
                "kurulu değil) — gerçek HID yolu çalıştırılamadı, sahte True sayılmadı."
            )
        bridge = ctrl._interception
        calls: List[Tuple[int, bool, bool]] = []
        orig_send_key = bridge.send_key

        def _record_only(scancode: int, is_down: bool, extended: bool = False) -> bool:
            calls.append((int(scancode), bool(is_down), bool(extended)))
            return True

        worker, mock_driver = self._fresh_worker({"fight_stop_gather_sec": 0})
        try:
            bridge.send_key = _record_only
            worker.input_driver = ctrl
            worker.driver = ctrl
            # Gathering durumu doğrudan kurulur (bkz. docstring).
            worker._is_gathering_active = True
            worker._state = BotState.GATHERING.value
            self.assertFalse(worker.is_combat_authorized("skill"))
            self.assertFalse(worker.is_combat_authorized("target"))
            self.assertFalse(worker.is_combat_authorized("attack"))
            for combat_key in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "r", "tab"):
                self.assertFalse(
                    worker._press_key(combat_key),
                    f"combat key '{combat_key}' leaked with real driver",
                )
            worker._cast_one_combat_skill("1")
            worker._cast_one_combat_skill("q")
            self.assertFalse(worker._send_configured_combat_skill())
            worker._press_combat_tab(soft_aim=True)
            worker._sync_combat_right_button(True)
            worker.combat_recovery.check_under_attack.reset_mock()
            diverted = worker._handle_combat_and_death()
            self.assertFalse(diverted, "Active gathering must not divert to combat (real driver)")
            worker.combat_recovery.check_under_attack.assert_not_called()
            self.assertEqual(calls, [], f"Combat keys reached the real HID path: {calls}")
        finally:
            bridge.send_key = orig_send_key
            try:
                ctrl.close()
            except Exception:
                pass
            worker.input_driver = mock_driver
            worker.driver = mock_driver

    def test_2_entering_gathering_releases_held_combat_inputs_and_halts_motion(self):
        """Test 2: Entering gathering immediately releases any active combat keys/mouse buttons and halts motion."""
        worker, driver = self._fresh_worker()
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
        worker, driver = self._fresh_worker()
        now = time.time()

        # --- STAGE 1/3 ---
        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 1)
        self.assertTrue(worker._stage_cast_in_progress)
        self.assertEqual(worker._current_node_harvest_stage, 0)

        # Complete Stage 1 cast: Y2/Y3 — barın YÜKSELDİĞİ kanıtlanmalı.
        # Tek karede %95 yetmez; ardışık okumalarda 25 -> 60 -> 95.
        _finish_stage_with_rising_bar(worker, conf=94, now=now)

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
        worker.hud_reader.hide_cast_bar()

        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 2)
        self.assertTrue(worker._stage_cast_in_progress)

        # Complete Stage 2 cast (bar yükselerek: 25 -> 60 -> 95)
        _finish_stage_with_rising_bar(worker, conf=94, now=now)

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
        worker.hud_reader.hide_cast_bar()

        worker._handle_route_gathering(conf=94, now=now)
        self.assertEqual(worker.f_pressed_count, 3)
        self.assertTrue(worker._stage_cast_in_progress)

        # Complete Stage 3 cast (bar yükselerek: 25 -> 60 -> 95)
        _finish_stage_with_rising_bar(worker, conf=94, now=now)

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
        worker, driver = self._fresh_worker()
        select_spy = MagicMock()
        worker._select_and_navigate_next_node = select_spy

        for stage_idx in (1, 2, 3):
            # 1) Prompt visible -> presses F and starts stage cast
            worker.vision.prompt_detected = True
            worker.hud_reader.hide_cast_bar()
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
            # Y2/Y3: bar %50'de tek başına aşamayı bitirmez; yükselme
            # kanıtı için dizi 50 -> 95 kurulur (2. okuma %95'te biter).
            worker.vision.prompt_detected = False
            worker.hud_reader.set_cast_bar_rising([50.0, 95.0])
            worker._step_route_mode()
            self.assertTrue(worker._is_gathering_active)
            self.assertEqual(worker.active_node["id"], 54)

            # 3) Cast reaches 95% (dizinin 2. okuması: 50 -> 95 yükselişi)
            # -> finishes current stage
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
        worker, driver = self._fresh_worker({"max_retries_per_harvest_stage": 2})
        now = time.time()

        # Stage 1/3 succeeds (bar yükselerek: 25 -> 60 -> 95)
        worker._handle_route_gathering(conf=90, now=now)
        _finish_stage_with_rising_bar(worker, conf=90, now=now)
        self.assertEqual(worker._current_node_harvest_stage, 1)

        # Start Stage 2/3
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.hide_cast_bar()
        worker._handle_route_gathering(conf=90, now=now)
        self.assertTrue(worker._stage_cast_in_progress)

        # Stage 2/3 cast finishes with chat reporting 'extraction_failed'.
        # Y2/Y3: extraction failure ancak bar yükselip %95'e ulaşınca
        # (aşama bitince) okunur — tek karede %95 sayılmaz.
        worker.hud_reader.chat_extraction_failed = True
        _finish_stage_with_rising_bar(worker, conf=90, now=now, pcts=(30.0, 65.0, 95.0))

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

        # Retry Stage 2/3 succeeds (F yeniden basılır, bar yükselerek biter)
        worker.hud_reader.chat_extraction_failed = False
        worker.hud_reader.hide_cast_bar()
        worker._handle_route_gathering(conf=90, now=now + 0.1)
        _finish_stage_with_rising_bar(worker, conf=90, now=now + 0.1, pcts=(30.0, 65.0, 95.0))
        self.assertEqual(worker._current_node_harvest_stage, 2)
        self.assertIsNotNone(worker.active_node)

        # Stage 3/3 succeeds -> node completes
        now = worker._inter_harvest_wait_until + 0.02
        worker.hud_reader.hide_cast_bar()
        worker._handle_route_gathering(conf=90, now=now)
        _finish_stage_with_rising_bar(worker, conf=90, now=now)

        self.assertEqual(worker.gathered_count, 3)
        self.assertFalse(worker._is_gathering_active)
        self.assertIsNone(worker.active_node)


if __name__ == "__main__":
    unittest.main()
