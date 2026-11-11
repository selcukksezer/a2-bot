# -*- coding: utf-8 -*-
"""
Görev 2 — Rota iptal zinciri + keşifte hemen saldırı düzeltmesi.

Kapsanan davranışlar (core/state_machine.py):
1. Düşman barı TEK BAŞINA FIGHTING'e sokmaz — kaç tick sürerse sürsün
   (fix/v20-panic-combat, canlı 14:48 hayalet-savaş: pasif mob barı
   "Fighting enemy" çıkarıyordu). Bar yalnızca bar penceresinde HP
   düşüşü veya ToT aggro da görüldüyse doğrulanır; HP düşüşü tek
   başına yalnızca ardışık tick'lerde sürerse doğrular
   (route_fight_confirm_frames, varsayılan 2).
2. Doğrulanmamış sinyal rota adımını iptal etmez: _handle_combat_and_death
   False döner, route step yürümeye/toplanmaya devam eder.
3. Doğrulanmış savaş rotayı askıya alır (active_node korunur) ve savaş
   doğrulandı-bittiğinde rota kendiliğinden devam eder (loot F + WALKING).
4. Histerezis: bir savaş bayrağı temizlendikten sonra
   route_fight_reentry_cooldown_sec (varsayılan 8 sn) içinde FIGHTING'e
   tekrar girilmez.
5. Keşif modunda öğrenici/kartografi kilidi (current_monster_targeted)
   ve çıplak "saldırı altında" (vignette/boş bar) savaşa çevirmez.
6. FIGHTING tek başına rotayı kapatmaz: _sync_input_driver_state
   ROUTE_MODE + FIGHTING durumunda route_active=True bırakır.

NOT: Bu dosya SADECE test yazar — pytest/bench KOŞTURULMAZ (görev kuralı).
"""

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from core.state_machine import BotMode, BotState, BotWorker


@pytest.fixture(autouse=True)
def _isolated_lifetime_stats(tmp_path):
    """Loot yolu get_lifetime_stats() singleton'una yazar; tmp store'a yönlendir."""
    from core.lifetime_stats import LifetimeStatsStore

    store = LifetimeStatsStore(path=str(tmp_path / "lifetime_stats.json"))
    with patch("core.lifetime_stats.get_lifetime_stats", return_value=store):
        yield


class MockInputDriver:
    """C7 harness'ındaki sürücünün route_active izleyen uzantısı."""

    def __init__(self, trusted: bool = True):
        self.pressed = []
        self.released = []
        self.holds = []
        self.mouse_events = []
        self._mouse_buttons_down = set()
        self._active_bot_keys = set()
        self._held_movement = set()
        self._trusted = trusted
        self._motion_abort_count = 0
        self.route_active = False
        self.route_active_calls = []

    def keyboard_inject_trusted(self) -> bool:
        return self._trusted

    def is_keyboard_trusted(self) -> bool:
        return self._trusted

    def movement_active(self) -> bool:
        return bool(self._held_movement) or getattr(self, "moving", True)

    def set_route_active(self, active: bool) -> None:
        self.route_active = bool(active)
        self.route_active_calls.append(bool(active))

    def press_key(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
        return True

    def send_key(self, key: str) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
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
        self._motion_abort_count += 1
        self._mouse_buttons_down.clear()

    def release_all_movement_keys(self) -> None:
        self._held_movement.clear()

    def clear(self):
        self.pressed.clear()
        self.released.clear()
        self.holds.clear()
        self.mouse_events.clear()


def create_worker(mode=BotMode.ROUTE_MODE.value, state=BotState.WALKING.value, config=None):
    """Gerçek BotWorker.__init__ + mock bağımlılıklar (C7 deseni).

    _route_enemy_bar_visible burada MagicMock recovery üzerinde GERÇEK
    çalıştırılsa truthy Mock döner; o yüzden her test bar sinyalini
    worker._route_enemy_bar_visible stub'ı ile kurar (mevcut testlerin
    _enemy_target_bar_visible stub'ladığı gibi).
    """
    base_config = {
        "test_mode": False,
        "fight_back": True,
        "route": {"fight_back": True},
        "combat_skill_keys": ["1", "2", "3", "4", "q", "e", "r"],
    }
    if config:
        base_config.update(config)

    driver = MockInputDriver(trusted=True)
    combat_recovery = MagicMock()
    combat_recovery.is_dead.return_value = False
    combat_recovery.check_character_dead.return_value = False
    combat_recovery.check_under_attack.return_value = (False, 1.0)
    combat_recovery.in_combat = False

    with patch.object(BotWorker, "_init_optional_subsystem", return_value=None):
        worker = BotWorker(
            config=base_config,
            vision=MagicMock(),
            input_driver=driver,
            route_manager=MagicMock(),
            combat_recovery=combat_recovery,
            navigator=None,
            recorder=None,
        )
    worker.driver = driver
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker._state_enter_time = time.time()
    worker.active_node = {"id": 101, "distance_m": 15}
    worker.logs = []
    worker._emit_log = lambda msg: worker.logs.append(str(msg))
    worker._emit_hud_snapshot = lambda **kwargs: {}
    worker._grab_frame = lambda grayscale=False: np.zeros((720, 1280, 3), dtype=np.uint8)
    worker._route_enemy_bar_visible = MagicMock(return_value=False)

    worker._combat_was_targeting = False
    worker._combat_no_target_streak = 0
    worker._combat_last_tab_ts = 0.0
    worker._combat_right_button_down = False
    worker._combat_skill_used_at = {}
    worker._combat_skill_idx = 0
    worker._last_bot_key_time = 0.0
    worker._is_gathering_active = False

    return worker, driver


class TestRouteFightConfirmation(unittest.TestCase):
    """Tek karelik sinyal FIGHTING'e sokmaz; doğrulanmış savaş rotayı askıya alıp bitince devam ettirir."""

    def test_single_frame_bar_does_not_enter_fighting(self):
        """1 tick bar -> FIGHTING yok, handler False, skill yok, rota sürer."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        res = worker._handle_combat_and_death()

        self.assertFalse(res)
        self.assertEqual(worker._state, BotState.WALKING.value)
        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        self.assertFalse(worker._route_fight_suspended)
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)
        combat_keys = {"1", "2", "3", "4", "5", "q", "e", "r", "tab"}
        self.assertEqual([k for k in driver.pressed if k in combat_keys], [])

    def test_two_consecutive_bar_frames_alone_do_not_enter_fighting(self):
        """fix/v20-panic-combat: bar TEK BAŞINA (2 tick de, 10 tick de)
        FIGHTING'e sokmaz — pasif mob barı hayalet-savaştır."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)

        for _ in range(10):
            res = worker._handle_combat_and_death()
            self.assertFalse(res)
            self.assertEqual(worker._state, BotState.WALKING.value)

        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        self.assertFalse(worker._route_fight_suspended)
        self.assertFalse(worker._route_bar_corroborated)
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)
        combat_keys = {"1", "2", "3", "4", "5", "q", "e", "r", "tab"}
        self.assertEqual([k for k in driver.pressed if k in combat_keys], [])

    def test_bar_plus_hp_drop_enters_fighting(self):
        """Bar + bar penceresinde HP düşüşü -> 2. tick FIGHTING."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)

        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        res = worker._handle_combat_and_death()
        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_combat_confirmed)
        self.assertTrue(worker._route_was_fighting)
        self.assertTrue(worker._route_fight_suspended)
        self.assertTrue(worker._route_bar_corroborated)
        self.assertGreater(worker.combat_recovery.fight_back_step.call_count, 0)

    def test_bar_plus_tot_agro_enters_fighting(self):
        """Bar + ToT aggro (HP sabit, HP okuyucudan bağımsız) -> 2. tick FIGHTING."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)
        worker.combat_recovery.last_target_of_target_agro = True
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)

        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)

        res = worker._handle_combat_and_death()
        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_combat_confirmed)
        self.assertTrue(worker._route_bar_corroborated)

    def test_bare_under_attack_vignette_never_fights(self):
        """Çıplak 'saldırı altında' (vignette, tam can, bar yok) 10 tick'te bile savaşa çevirmez."""
        worker, driver = create_worker()
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        worker._route_enemy_bar_visible = MagicMock(return_value=False)

        for _ in range(10):
            res = worker._handle_combat_and_death()
            self.assertFalse(res)
            self.assertEqual(worker._state, BotState.WALKING.value)

        self.assertFalse(worker._route_combat_confirmed)
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)

    def test_single_hp_drop_does_not_confirm(self):
        """Tek HP düşüşü (1.0 -> 0.97) sonra can sabit -> FIGHTING yok."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=False)

        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertFalse(worker._handle_combat_and_death())

        self.assertFalse(worker._route_combat_confirmed)
        self.assertEqual(worker._state, BotState.WALKING.value)

    def test_consecutive_hp_drops_confirm_fight(self):
        """Ardışık HP düşüşleri (1.0 -> 0.97 -> 0.94, gerçek _note_player_hp_drop) 3. tick'te doğrular."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=False)

        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)
        worker.combat_recovery.check_under_attack.return_value = (True, 0.94)
        res = worker._handle_combat_and_death()

        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_combat_confirmed)

    def test_confirmed_fight_suspends_route_and_resumes_after_end(self):
        """Doğrulanmış savaş active_node'u koruyup askıya alır; sinyal bitince loot F + WALKING, rota devam eder."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        # fix/v20: giriş için bar + HP düşüşü gerekir (bar tek başına yetmez).
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        worker._handle_combat_and_death()
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        res = worker._handle_combat_and_death()
        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_fight_suspended)
        self.assertIsNotNone(worker.active_node)

        # Savaş bitti: bar yok, can sabit.
        worker._route_enemy_bar_visible = MagicMock(return_value=False)
        worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
        worker._handle_combat_and_death()

        self.assertIn("f", driver.pressed)
        self.assertEqual(worker._state, BotState.WALKING.value)
        self.assertIsNotNone(worker.active_node)
        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        self.assertFalse(worker._route_fight_suspended)

    def test_hysteresis_blocks_immediate_reentry(self):
        """Bayrak temizlendikten hemen sonra 2 karelik bar giremez; pencere dolunca girer."""
        worker, driver = create_worker()
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        # fix/v20: giriş için bar + HP düşüşü gerekir (bar tek başına yetmez).
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        worker._handle_combat_and_death()
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertTrue(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.FIGHTING.value)

        # Savaş bitsin (loot + temizle -> histerezis damgası).
        worker._route_enemy_bar_visible = MagicMock(return_value=False)
        worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
        worker._handle_combat_and_death()
        self.assertEqual(worker._state, BotState.WALKING.value)
        self.assertGreater(worker._route_fight_last_cleared_ts, 0.0)

        # Hemen ardından 2 karelik bar: pencere dolmadı -> girilmez.
        worker._route_enemy_bar_visible = MagicMock(return_value=True)
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)
        self.assertFalse(worker._route_combat_confirmed)

        # Pencere dolmuş say (30 sn önce temizlenmiş): bar streak'i zaten
        # 2'de olduğundan sıradaki tick girer (HP düşüşü kanıtıyla).
        worker._route_fight_last_cleared_ts = time.time() - 30.0
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertTrue(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.FIGHTING.value)

    def test_hysteresis_disabled_by_zero_cooldown(self):
        """route_fight_reentry_cooldown_sec=0 iken temizlik sonrası anında tekrar giriş serbesttir."""
        worker, driver = create_worker(config={"route_fight_reentry_cooldown_sec": 0})
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        # fix/v20: giriş için bar + HP düşüşü gerekir (bar tek başına yetmez).
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        worker._handle_combat_and_death()
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertTrue(worker._handle_combat_and_death())

        worker._route_enemy_bar_visible = MagicMock(return_value=False)
        worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
        worker._handle_combat_and_death()
        self.assertEqual(worker._state, BotState.WALKING.value)

        worker._route_enemy_bar_visible = MagicMock(return_value=True)
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertTrue(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.FIGHTING.value)

    def test_confirm_frames_config(self):
        """route_fight_confirm_frames=3 iken 2 tick yetmez, 3. tick girer."""
        worker, driver = create_worker(config={"route_fight_confirm_frames": 3})
        worker._route_enemy_bar_visible = MagicMock(return_value=True)

        # fix/v20: giriş için bar + HP düşüşü gerekir (bar tek başına yetmez).
        worker.combat_recovery.check_under_attack.return_value = (True, 1.0)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.97)
        self.assertFalse(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.WALKING.value)
        worker.combat_recovery.check_under_attack.return_value = (True, 0.94)
        self.assertTrue(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.FIGHTING.value)

    def test_explorer_learner_lock_is_not_a_fight(self):
        """Keşif modunda öğrenici kilidi + geniş bar kilidi varken, gerçek bar yoksa savaşa dönülmez."""
        worker, driver = create_worker(
            mode=BotMode.EXPLORER_MODE.value,
            state=BotState.WALKING.value,
        )
        worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
        worker._route_enemy_bar_visible = MagicMock(return_value=False)
        # Geniş kilit "var" dese bile kapı dar tanımı kullanır:
        worker._enemy_target_bar_visible = MagicMock(return_value=True)
        learner = MagicMock()
        learner.current_monster_targeted = True
        learner.get_live_cognition_snapshot.return_value = {"monster_targeted": True}
        worker.game_learner = learner

        for _ in range(5):
            res = worker._handle_combat_and_death()
            self.assertFalse(res)
            self.assertEqual(worker._state, BotState.WALKING.value)

        self.assertFalse(worker._route_combat_confirmed)
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)

    def test_fighting_alone_keeps_route_active(self):
        """Item 5: FIGHTING tek başına rotayı kapatmaz (route_active True kalır); PAUSED kapatır."""
        worker, driver = create_worker()
        worker._state = BotState.FIGHTING.value
        worker._paused = False

        worker._sync_input_driver_state()
        self.assertTrue(driver.route_active)
        self.assertIn(True, driver.route_active_calls)

        worker._paused = True
        worker._state = BotState.PAUSED.value
        worker._sync_input_driver_state()
        self.assertFalse(driver.route_active)


if __name__ == "__main__":
    unittest.main()
