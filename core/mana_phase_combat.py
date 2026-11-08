# -*- coding: utf-8 -*-
"""
ManaPhaseCombat — MP eşiğine göre Regen (sol tık) / Burst (sağ tık) faz kapısı.

Mevcut sağ-tık + GameActionBus skill rotasyonunu silmez; yalnızca faz seçer.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("Gatherion.ManaPhaseCombat")


class ManaPhase(str, Enum):
    REGEN = "A"
    BURST = "B"


class KitingAction(str, Enum):
    NONE = "NONE"
    BACKPEDAL = "BACKPEDAL"
    STRAFE_LEFT = "STRAFE_LEFT"
    STRAFE_RIGHT = "STRAFE_RIGHT"
    RETREAT_JUMP = "RETREAT_JUMP"
    CLOSE_GAP = "CLOSE_GAP"


@dataclass
class KitingDecision:
    should_kite: bool = False
    action: str = "NONE"
    movement_keys: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    reason: str = ""


@dataclass
class AutoPotionResult:
    hp_potion_triggered: bool = False
    mp_potion_triggered: bool = False
    keys_to_press: List[str] = field(default_factory=list)
    hp_ratio: float = 1.0
    mp_ratio: float = 1.0
    reason: str = ""


# Default 7-key hotbar sequence and sub-100ms burst timing constants
DEFAULT_COMBAT_SKILL_KEYS: Tuple[str, ...] = ("1", "2", "3", "4", "q", "e", "r")
DEFAULT_BURST_INTERVAL_SEC: float = 0.08  # 80ms sub-100ms burst delay
DEFAULT_GCD_DELAY_SEC: float = 0.12       # 120ms global cooldown
DEFAULT_COMBO_CHAINS: Dict[str, List[str]] = {}


@dataclass
class ManaPhaseDecision:
    phase: ManaPhase
    hold_left: bool
    hold_right: bool
    skill_keys: List[str]
    reason: str = ""
    primary_skill: Optional[str] = None
    combo_active: bool = False
    burst_delay_sec: float = DEFAULT_BURST_INTERVAL_SEC
    gcd_delay_sec: float = DEFAULT_GCD_DELAY_SEC
    potion_keys: List[str] = field(default_factory=list)
    kiting: Optional[KitingDecision] = None
    player_hp: float = 1.0
    player_mp: float = 1.0
    buff_keys: List[str] = field(default_factory=list)
    emergency: bool = False
    action_type: str = "combat"
    all_keys_to_press: List[str] = field(default_factory=list)


class ManaPhaseCombat:
    """
    Decide regen vs burst and manage millisecond skill rotation & burst timing.
    
    Features:
    - Sub-100ms burst delays (0.08s default, configurable 0.08s-0.15s, gcd ~0.12s).
    - 7-key round-robin sequence across hotbar ('1', '2', '3', '4', 'q', 'e', 'r').
    - Combo chain support (e.g. 1 -> 2 -> 3, q -> e -> r) with auto-advancement and expiry.
    - Zero sticky '2' key bias: fair cycle rotation and anti-repetition protection.
    - Deterministic millisecond pacing with no hidden Gaussian delays.
    - SmartConsumableManager integration: dynamically tracks upper consumable hotbar (F1..F8, X, B),
      triggering HP potion (<65%, emergency <40%), MP potion (<45%), and re-applying expired buff scrolls.
    """

    DEFAULT_COMBAT_SKILL_KEYS = DEFAULT_COMBAT_SKILL_KEYS
    DEFAULT_BURST_INTERVAL_SEC = DEFAULT_BURST_INTERVAL_SEC
    DEFAULT_GCD_DELAY_SEC = DEFAULT_GCD_DELAY_SEC

    def __init__(
        self,
        mp_full_threshold: float = 0.95,
        hp_panic_threshold: float = 0.50,
        on_log: Optional[Callable[[str], None]] = None,
        burst_skill_interval_sec: float = DEFAULT_BURST_INTERVAL_SEC,
        gcd_delay_sec: float = DEFAULT_GCD_DELAY_SEC,
        combat_skill_keys: Optional[Sequence[str]] = None,
        combo_chains: Optional[Dict[str, Sequence[str]]] = None,
        hp_potion_threshold: float = 0.50,
        mp_potion_threshold: float = 0.30,
        hp_potion_key: str = "7",
        mp_potion_key: str = "8",
        hp_potion_cd_sec: float = 10.0,
        mp_potion_cd_sec: float = 10.0,
        kiting_enabled: bool = True,
        kiting_hp_threshold: float = 0.50,
        kiting_min_dist: float = 3.5,
        kiting_optimal_dist: float = 8.0,
        kiting_max_dist: float = 15.0,
        kiting_cooldown_sec: float = 1.2,
        consumable_manager: Optional[Any] = None,
        exemplar_memory: Optional[Any] = None,
        smart_hp_potion_threshold: float = 0.65,
        smart_mp_potion_threshold: float = 0.45,
        emergency_hp_threshold: float = 0.40,
    ) -> None:
        self.mp_full_threshold = float(mp_full_threshold)
        self.hp_panic_threshold = float(hp_panic_threshold)
        self._on_log = on_log
        self._phase = ManaPhase.BURST
        self._last_log_phase: Optional[ManaPhase] = None

        # Auto-Potion Triggers & Vitality Monitoring (HP < 50%, MP < 30%)
        self.hp_potion_threshold = float(hp_potion_threshold)
        self.mp_potion_threshold = float(mp_potion_threshold)
        self.hp_potion_key = str(hp_potion_key).strip() or "7"
        self.mp_potion_key = str(mp_potion_key).strip() or "8"
        self.hp_potion_cd_sec = float(hp_potion_cd_sec)
        self.mp_potion_cd_sec = float(mp_potion_cd_sec)
        self._last_hp_potion_ts: float = 0.0
        self._last_mp_potion_ts: float = 0.0

        # Smart Consumables & Emergency Recovery (HP < 65%, emergency < 40%, MP < 45%, Buff scrolls)
        self.smart_hp_potion_threshold = float(smart_hp_potion_threshold)
        self.smart_mp_potion_threshold = float(smart_mp_potion_threshold)
        self.emergency_hp_threshold = float(emergency_hp_threshold)
        self.exemplar_memory = exemplar_memory
        if consumable_manager is not None:
            self.consumable_manager = consumable_manager
        else:
            try:
                from core.skill_bar_reader import SmartConsumableManager
                self.consumable_manager = SmartConsumableManager(exemplar_memory=self.exemplar_memory)
            except Exception as exc:
                logger.debug("SmartConsumableManager init fallback: %s", exc)
                self.consumable_manager = None

        # Combat Kiting Distance Adjustments & State
        self.kiting_enabled = bool(kiting_enabled)
        self.kiting_hp_threshold = float(kiting_hp_threshold)
        self.kiting_min_dist = float(kiting_min_dist)
        self.kiting_optimal_dist = float(kiting_optimal_dist)
        self.kiting_max_dist = float(kiting_max_dist)
        self.kiting_cooldown_sec = float(kiting_cooldown_sec)
        self._last_kiting_ts: float = 0.0
        self._strafe_toggle: bool = False

        # Sub-100ms Burst & GCD Timing (Deterministic, no hidden Gaussian delay)
        import os
        env_burst = str(os.environ.get("GATHERION_SKILL_BURST_SEC", "") or "").strip()
        if env_burst:
            try:
                self.burst_skill_interval_sec = max(0.02, min(0.50, float(env_burst)))
            except (TypeError, ValueError):
                self.burst_skill_interval_sec = float(burst_skill_interval_sec)
        else:
            self.burst_skill_interval_sec = float(burst_skill_interval_sec)

        self.gcd_delay_sec = float(gcd_delay_sec)
        self._last_skill_ts = 0.0
        self._last_cast_ts = 0.0
        self._last_cast_key: Optional[str] = None

        # 7-Key Round-Robin Sequence
        raw_keys = combat_skill_keys if combat_skill_keys is not None else self.DEFAULT_COMBAT_SKILL_KEYS
        self.combat_skill_keys: List[str] = [str(k).strip().lower() for k in raw_keys if str(k).strip()]
        if not self.combat_skill_keys:
            self.combat_skill_keys = list(self.DEFAULT_COMBAT_SKILL_KEYS)
        self._rotation_idx: int = 0
        self._key_cast_counts: Dict[str, int] = {k: 0 for k in self.combat_skill_keys}
        self._consecutive_same_key_count: int = 0

        # Combo Chains Support
        self._combo_chains: Dict[str, List[str]] = {}
        chains_src = combo_chains if combo_chains is not None else DEFAULT_COMBO_CHAINS
        for trigger, steps in chains_src.items():
            self._combo_chains[str(trigger).strip().lower()] = [str(s).strip().lower() for s in steps]
        self._active_combo_chain: List[str] = []
        self._combo_chain_step: int = 0
        self._combo_expires_at: float = 0.0
        self._combo_timeout_sec: float = 2.0

        self._lexicon: Dict[str, str] = {}

    def set_lexicon(self, lexicon: Dict[str, str]) -> None:
        self._lexicon = {str(k).lower(): str(v).lower() for k, v in (lexicon or {}).items()}

    def set_thresholds(
        self,
        mp_full_threshold: Optional[float] = None,
        hp_panic_threshold: Optional[float] = None,
        hp_potion_threshold: Optional[float] = None,
        mp_potion_threshold: Optional[float] = None,
        kiting_hp_threshold: Optional[float] = None,
        hp_potion_key: Optional[str] = None,
        mp_potion_key: Optional[str] = None,
        hp_potion_cd_sec: Optional[float] = None,
        mp_potion_cd_sec: Optional[float] = None,
        kiting_enabled: Optional[bool] = None,
        kiting_min_dist: Optional[float] = None,
        kiting_optimal_dist: Optional[float] = None,
        kiting_max_dist: Optional[float] = None,
        kiting_cooldown_sec: Optional[float] = None,
        smart_hp_potion_threshold: Optional[float] = None,
        smart_mp_potion_threshold: Optional[float] = None,
        emergency_hp_threshold: Optional[float] = None,
    ) -> None:
        if mp_full_threshold is not None:
            self.mp_full_threshold = max(0.5, min(1.0, float(mp_full_threshold)))
        if hp_panic_threshold is not None:
            self.hp_panic_threshold = max(0.1, min(0.9, float(hp_panic_threshold)))
        if hp_potion_threshold is not None:
            self.hp_potion_threshold = max(0.05, min(0.95, float(hp_potion_threshold)))
        if mp_potion_threshold is not None:
            self.mp_potion_threshold = max(0.05, min(0.95, float(mp_potion_threshold)))
        if kiting_hp_threshold is not None:
            self.kiting_hp_threshold = max(0.1, min(0.9, float(kiting_hp_threshold)))
        if hp_potion_key is not None:
            self.hp_potion_key = str(hp_potion_key).strip() or self.hp_potion_key
        if mp_potion_key is not None:
            self.mp_potion_key = str(mp_potion_key).strip() or self.mp_potion_key
        if hp_potion_cd_sec is not None:
            self.hp_potion_cd_sec = max(0.5, float(hp_potion_cd_sec))
        if mp_potion_cd_sec is not None:
            self.mp_potion_cd_sec = max(0.5, float(mp_potion_cd_sec))
        if kiting_enabled is not None:
            self.kiting_enabled = bool(kiting_enabled)
        if kiting_min_dist is not None:
            self.kiting_min_dist = max(1.0, float(kiting_min_dist))
        if kiting_optimal_dist is not None:
            self.kiting_optimal_dist = max(self.kiting_min_dist + 0.5, float(kiting_optimal_dist))
        if kiting_max_dist is not None:
            self.kiting_max_dist = max(self.kiting_optimal_dist + 1.0, float(kiting_max_dist))
        if kiting_cooldown_sec is not None:
            self.kiting_cooldown_sec = max(0.2, float(kiting_cooldown_sec))
        if smart_hp_potion_threshold is not None:
            self.smart_hp_potion_threshold = max(0.05, min(0.95, float(smart_hp_potion_threshold)))
        if smart_mp_potion_threshold is not None:
            self.smart_mp_potion_threshold = max(0.05, min(0.95, float(smart_mp_potion_threshold)))
        if emergency_hp_threshold is not None:
            self.emergency_hp_threshold = max(0.05, min(0.95, float(emergency_hp_threshold)))

    def set_consumable_manager(self, manager: Any) -> None:
        """Sets or replaces the active SmartConsumableManager."""
        self.consumable_manager = manager

    def set_combat_skill_keys(self, keys: Sequence[str]) -> None:
        """Update active combat skill keys and re-anchor round-robin rotation without bias."""
        cleaned = [str(k).strip().lower() for k in keys if str(k).strip()]
        if cleaned:
            self.combat_skill_keys = cleaned
            for k in cleaned:
                if k not in self._key_cast_counts:
                    self._key_cast_counts[k] = 0
            self._rotation_idx = self._rotation_idx % len(self.combat_skill_keys)

    def set_burst_delays(
        self,
        burst_interval_sec: Optional[float] = None,
        gcd_delay_sec: Optional[float] = None,
    ) -> None:
        """Set sub-100ms burst interval (e.g. 0.08s / 0.15s) and GCD delay (~0.12s)."""
        if burst_interval_sec is not None:
            self.burst_skill_interval_sec = max(0.02, min(0.50, float(burst_interval_sec)))
        if gcd_delay_sec is not None:
            self.gcd_delay_sec = max(0.04, min(1.0, float(gcd_delay_sec)))

    def register_combo(self, trigger_key: str, chain_keys: Sequence[str]) -> None:
        """Register a combo chain for a trigger skill (e.g. '1' -> ['2', '3'])."""
        trig = str(trigger_key).strip().lower()
        self._combo_chains[trig] = [str(k).strip().lower() for k in chain_keys if str(k).strip()]

    def reset_rotation(self) -> None:
        """Reset rotation index, combo state, and cast counters."""
        self._rotation_idx = 0
        self._last_cast_key = None
        self._consecutive_same_key_count = 0
        self._active_combo_chain.clear()
        self._combo_chain_step = 0
        self._combo_expires_at = 0.0
        for k in self._key_cast_counts:
            self._key_cast_counts[k] = 0

    @property
    def phase(self) -> ManaPhase:
        return self._phase

    def get_next_combo_skill(self, now: Optional[float] = None) -> Optional[str]:
        """Returns the next skill in an active combo chain if unexpired, else None."""
        t = time.time() if now is None else float(now)
        if self._active_combo_chain and t <= self._combo_expires_at:
            if 0 <= self._combo_chain_step < len(self._active_combo_chain):
                return self._active_combo_chain[self._combo_chain_step]
        # Expired or complete
        self._active_combo_chain.clear()
        self._combo_chain_step = 0
        self._combo_expires_at = 0.0
        return None

    def record_skill_cast(self, key: str, now: Optional[float] = None) -> None:
        """
        Record that a skill key was cast. Updates timestamps, GCD, rotation,
        cast counts, and advances combo chains.
        """
        k = str(key).strip().lower()
        t = time.time() if now is None else float(now)
        self._last_cast_ts = t
        self._last_skill_ts = t

        if k == self._last_cast_key:
            self._consecutive_same_key_count += 1
        else:
            self._consecutive_same_key_count = 1
        self._last_cast_key = k

        self._key_cast_counts[k] = self._key_cast_counts.get(k, 0) + 1

        # Combo chain progression
        if self._active_combo_chain and t <= self._combo_expires_at:
            expected = (
                self._active_combo_chain[self._combo_chain_step]
                if self._combo_chain_step < len(self._active_combo_chain)
                else None
            )
            if k == expected:
                self._combo_chain_step += 1
                if self._combo_chain_step >= len(self._active_combo_chain):
                    # Combo finished
                    self._active_combo_chain.clear()
                    self._combo_chain_step = 0
                    self._combo_expires_at = 0.0
                else:
                    # Refresh expiry for next step
                    self._combo_expires_at = t + self._combo_timeout_sec
                return

        # Check if k triggers a new combo chain
        if k in self._combo_chains and self._combo_chains[k]:
            self._active_combo_chain = list(self._combo_chains[k])
            self._combo_chain_step = 0
            self._combo_expires_at = t + self._combo_timeout_sec

    def get_round_robin_burst_slice(
        self,
        count: int = 3,
        pool: Optional[Sequence[str]] = None,
        allow_combo: bool = True,
        now: Optional[float] = None,
    ) -> List[str]:
        """
        Selects `count` skills strictly following round-robin progression across pool
        (defaulting to self.combat_skill_keys: 1, 2, 3, 4, q, e, r).
        
        Guarantees zero sticky '2' key bias:
        - Cycles fairly across all keys in the pool.
        - Prioritizes active combo chain step if valid.
        - Prevents consecutive repetition of any single key unless len(pool) == 1.
        """
        active_pool = [str(k).strip().lower() for k in (pool or self.combat_skill_keys) if str(k).strip()]
        if not active_pool:
            active_pool = list(self.DEFAULT_COMBAT_SKILL_KEYS)

        # De-duplicate pool preserving order
        seen = set()
        dedup_pool = []
        for k in active_pool:
            if k not in seen:
                seen.add(k)
                dedup_pool.append(k)
        active_pool = dedup_pool

        result: List[str] = []
        t = time.time() if now is None else float(now)

        # 1. Combo chain priority
        if allow_combo:
            combo_skill = self.get_next_combo_skill(now=t)
            if combo_skill and combo_skill in active_pool:
                result.append(combo_skill)

        # 2. Fill remaining from round-robin
        needed = max(1, count) - len(result)
        pool_len = len(active_pool)
        if needed > 0 and pool_len > 0:
            start_idx = self._rotation_idx % pool_len
            collected = 0
            offset = 0

            # Scan up to 2 full loops to find non-repeated keys
            while collected < needed and offset < (pool_len * 2):
                candidate = active_pool[(start_idx + offset) % pool_len]
                offset += 1

                # Avoid duplicate within the same burst slice
                if candidate in result:
                    continue

                # Avoid sticky '2' bias: do not repeat last cast key if alternatives exist
                if (
                    candidate == self._last_cast_key
                    and self._consecutive_same_key_count >= 1
                    and pool_len > 1
                    and len(result) == 0
                    and offset < pool_len
                ):
                    continue

                result.append(candidate)
                collected += 1

            # Advance the global rotation index by collected count
            self._rotation_idx = (start_idx + max(1, collected)) % pool_len

        return result

    def check_auto_potions(
        self,
        player_hp: Optional[float] = None,
        player_mp: Optional[float] = None,
        now: Optional[float] = None,
        hp_ratio: Optional[float] = None,
        mp_ratio: Optional[float] = None,
    ) -> AutoPotionResult:
        """
        Dynamically monitors player HP and MP thresholds:
        - HP < hp_potion_threshold (default 0.50 / 50%): trigger HP potion key ('7').
        - MP < mp_potion_threshold (default 0.30 / 30%): trigger MP potion key ('8').
        Respects independent per-potion cooldowns (default 10.0s).
        """
        t = time.time() if now is None else float(now)
        raw_hp = player_hp if player_hp is not None else (hp_ratio if hp_ratio is not None else 1.0)
        raw_mp = player_mp if player_mp is not None else (mp_ratio if mp_ratio is not None else 1.0)
        hp = max(0.0, min(1.0, float(raw_hp)))
        mp = max(0.0, min(1.0, float(raw_mp)))

        hp_trigger = False
        mp_trigger = False
        keys: List[str] = []
        reasons: List[str] = []

        FORBIDDEN = {"x", "f6", "b"}

        if hp < self.hp_potion_threshold:
            if (t - self._last_hp_potion_ts) >= self.hp_potion_cd_sec:
                if str(self.hp_potion_key).lower() not in FORBIDDEN:
                    hp_trigger = True
                    self._last_hp_potion_ts = t
                    keys.append(self.hp_potion_key)
                    reasons.append(f"HP {hp*100:.1f}% < {self.hp_potion_threshold*100:.0f}%")
                    msg = f"AUTO_POTION: HP triggered [{self.hp_potion_key}] (hp={hp:.2f})"
                    logger.info(msg)
                    if callable(self._on_log):
                        try:
                            self._on_log(msg)
                        except Exception:
                            pass

        if mp < self.mp_potion_threshold:
            if (t - self._last_mp_potion_ts) >= self.mp_potion_cd_sec:
                if str(self.mp_potion_key).lower() not in FORBIDDEN:
                    mp_trigger = True
                    self._last_mp_potion_ts = t
                    keys.append(self.mp_potion_key)
                    reasons.append(f"MP {mp*100:.1f}% < {self.mp_potion_threshold*100:.0f}%")
                    msg = f"AUTO_POTION: MP triggered [{self.mp_potion_key}] (mp={mp:.2f})"
                    logger.info(msg)
                    if callable(self._on_log):
                        try:
                            self._on_log(msg)
                        except Exception:
                            pass

        return AutoPotionResult(
            hp_potion_triggered=hp_trigger,
            mp_potion_triggered=mp_trigger,
            keys_to_press=keys,
            hp_ratio=hp,
            mp_ratio=mp,
            reason="; ".join(reasons) if reasons else "vitals_ok",
        )

    def trigger_potions(
        self,
        player_hp: Optional[float] = None,
        player_mp: Optional[float] = None,
        action_bus: Any = None,
        send_key_fn: Optional[Callable[[str], None]] = None,
        now: Optional[float] = None,
        hp_ratio: Optional[float] = None,
        mp_ratio: Optional[float] = None,
    ) -> AutoPotionResult:
        """
        Check and execute auto-potion key dispatch via action_bus or callback.
        """
        res = self.check_auto_potions(
            player_hp=player_hp,
            player_mp=player_mp,
            now=now,
            hp_ratio=hp_ratio,
            mp_ratio=mp_ratio,
        )
        for key in res.keys_to_press:
            if send_key_fn is not None:
                try:
                    send_key_fn(key)
                except Exception as exc:
                    logger.debug("send_key_fn potion error: %s", exc)
            elif action_bus is not None:
                if hasattr(action_bus, "cast_skill"):
                    try:
                        action_bus.cast_skill(key)
                    except Exception as exc:
                        logger.debug("action_bus cast_skill potion error: %s", exc)
                elif hasattr(action_bus, "press_key"):
                    try:
                        action_bus.press_key(key)
                    except Exception as exc:
                        logger.debug("action_bus press_key potion error: %s", exc)
        return res

    def check_emergency_recovery(
        self,
        frame: Optional[np.ndarray] = None,
        player_hp: Optional[float] = None,
        player_mp: Optional[float] = None,
        active_buffs: Optional[Union[Collection[str], Dict[str, Any]]] = None,
        now: Optional[float] = None,
        use_smart_thresholds: bool = True,
    ) -> Dict[str, Any]:
        """
        Item 47 — Akıllı Tüketilebilir ve Acil İyileşme Kontrolü (`check_emergency_recovery`).

        Inspects player vitals and consumable slots:
        - HP < 65% (or emergency < 40%): triggers learned hp_potion key (e.g. 'F1' or 'X').
        - MP < 45%: triggers learned mp_potion key (e.g. 'F2' or 'F4').
        - Any learned buff scroll expired ('buff_expired'): triggers learned scroll key
          (e.g. 'F5', 'F6', 'F7') to re-apply the buff automatically!

        Returns diagnostics dictionary with:
        `is_emergency`, `hp_potion_needed`, `mp_potion_needed`, `buff_expired`,
        `hp_potion_key`, `mp_potion_key`, `buff_keys`, `keys_to_press`, `reason`.
        """
        t = time.time() if now is None else float(now)

        # 1. Read vitals from frame if not explicitly provided
        if frame is not None and (player_hp is None or player_mp is None):
            vitals = self.read_vitals_from_frame(frame)
            if player_hp is None:
                player_hp = vitals.get("player_hp", 1.0)
            if player_mp is None:
                player_mp = vitals.get("player_mp", 1.0)

        # Inspect consumables from frame if frame provided
        if frame is not None and self.consumable_manager is not None:
            self.consumable_manager.inspect_consumables(frame)

        hp = max(0.0, min(1.0, float(player_hp if player_hp is not None else 1.0)))
        mp = max(0.0, min(1.0, float(player_mp if player_mp is not None else 1.0)))

        hp_thresh = self.smart_hp_potion_threshold if use_smart_thresholds else self.hp_potion_threshold
        mp_thresh = self.smart_mp_potion_threshold if use_smart_thresholds else self.mp_potion_threshold
        emergency_thresh = self.emergency_hp_threshold

        is_emergency = hp < emergency_thresh
        hp_potion_needed = hp < hp_thresh
        mp_potion_needed = mp < mp_thresh

        hp_triggered = False
        mp_triggered = False
        buff_expired = False

        chosen_hp_key: Optional[str] = None
        chosen_mp_key: Optional[str] = None
        chosen_buff_keys: List[str] = []
        keys_to_press: List[str] = []
        reasons: List[str] = []

        # 2. HP Potion Check (< 65% or emergency < 40%)
        FORBIDDEN = {"x", "f6", "b"}
        if hp_potion_needed:
            learned_hp = self.consumable_manager.get_best_hp_potion(now=t) if self.consumable_manager is not None else None
            candidate_key = learned_hp or self.hp_potion_key
            if candidate_key and str(candidate_key).lower() in FORBIDDEN:
                candidate_key = None

            cd_ready = (t - self._last_hp_potion_ts) >= self.hp_potion_cd_sec
            if self.consumable_manager is not None and candidate_key and candidate_key in self.consumable_manager.CONSUMABLE_SLOTS:
                cd_ready = cd_ready and self.consumable_manager.is_slot_ready(candidate_key, now=t)

            if cd_ready and candidate_key:
                hp_triggered = True
                chosen_hp_key = candidate_key
                self._last_hp_potion_ts = t
                if self.consumable_manager is not None:
                    self.consumable_manager.mark_used(candidate_key, now=t)
                severity = "EMERGENCY" if is_emergency else "AUTO_POTION"
                reasons.append(f"{severity}: HP {hp*100:.1f}% < {hp_thresh*100:.0f}% -> [{chosen_hp_key}]")
                msg = f"{severity}: HP triggered [{chosen_hp_key}] (hp={hp:.2f})"
                logger.info(msg)
                if callable(self._on_log):
                    try:
                        self._on_log(msg)
                    except Exception:
                        pass

        # 3. MP Potion Check (< 45%)
        if mp_potion_needed:
            learned_mp = self.consumable_manager.get_best_mp_potion(now=t) if self.consumable_manager is not None else None
            candidate_key = learned_mp or self.mp_potion_key
            if candidate_key and str(candidate_key).lower() in FORBIDDEN:
                candidate_key = None

            cd_ready = (t - self._last_mp_potion_ts) >= self.mp_potion_cd_sec
            if self.consumable_manager is not None and candidate_key and candidate_key in self.consumable_manager.CONSUMABLE_SLOTS:
                cd_ready = cd_ready and self.consumable_manager.is_slot_ready(candidate_key, now=t)

            if cd_ready and candidate_key:
                mp_triggered = True
                chosen_mp_key = candidate_key
                self._last_mp_potion_ts = t
                if self.consumable_manager is not None:
                    self.consumable_manager.mark_used(candidate_key, now=t)
                reasons.append(f"AUTO_POTION: MP {mp*100:.1f}% < {mp_thresh*100:.0f}% -> [{chosen_mp_key}]")
                msg = f"AUTO_POTION: MP triggered [{chosen_mp_key}] (mp={mp:.2f})"
                logger.info(msg)
                if callable(self._on_log):
                    try:
                        self._on_log(msg)
                    except Exception:
                        pass

        # 4. Expired Buff Scroll Check (buff_expired -> re-apply automatically)
        if self.consumable_manager is not None:
            expired_scrolls = self.consumable_manager.check_and_refresh_scroll_buffs(active_buffs=active_buffs, now=t)
            expired_scrolls = [s for s in (expired_scrolls or []) if str(s).lower() not in FORBIDDEN]
            if expired_scrolls:
                buff_expired = True
                for skey in expired_scrolls:
                    chosen_buff_keys.append(skey)
                    reasons.append(f"BUFF_REAPPLY: Scroll expired -> [{skey}]")
                    msg = f"BUFF_REAPPLY: Scroll triggered [{skey}]"
                    logger.info(msg)
                    if callable(self._on_log):
                        try:
                            self._on_log(msg)
                        except Exception:
                            pass

        # 5. Build prioritized keys_to_press:
        # Emergency HP potion has absolute highest priority!
        if chosen_hp_key:
            keys_to_press.append(chosen_hp_key)
        if chosen_mp_key:
            keys_to_press.append(chosen_mp_key)
        if chosen_buff_keys:
            keys_to_press.extend(chosen_buff_keys)

        return {
            "is_emergency": is_emergency,
            "hp_potion_needed": hp_potion_needed,
            "mp_potion_needed": mp_potion_needed,
            "hp_potion_triggered": hp_triggered,
            "mp_potion_triggered": mp_triggered,
            "buff_expired": buff_expired,
            "hp_potion_key": chosen_hp_key,
            "mp_potion_key": chosen_mp_key,
            "buff_keys": chosen_buff_keys,
            "expired_scrolls": chosen_buff_keys,
            "keys_to_press": keys_to_press,
            "player_hp": hp,
            "player_mp": mp,
            "reason": "; ".join(reasons) if reasons else "vitals_healthy",
        }

    def decide_combat_action(
        self,
        frame: Optional[np.ndarray] = None,
        player_hp: Optional[float] = None,
        player_mp: Optional[float] = None,
        ready_skills: Optional[Sequence[str]] = None,
        fallback_keys: Optional[Sequence[str]] = None,
        cc_active: bool = False,
        now: Optional[float] = None,
        target_distance: Optional[float] = None,
        target_hp: float = 1.0,
        is_melee_threat: bool = False,
        is_casting: bool = False,
        active_buffs: Optional[Union[Collection[str], Dict[str, Any]]] = None,
    ) -> ManaPhaseDecision:
        """
        Unified combat action decision:
        1. Checks emergency recovery (HP < 65%, emergency < 40%, MP < 45%, expired buff scrolls).
        2. Evaluates combat rotation and mana phase (BURST vs REGEN).
        3. Returns enriched ManaPhaseDecision with potion_keys, buff_keys, emergency flag,
           action_type, and all_keys_to_press.
        """
        t = time.time() if now is None else float(now)

        # Run emergency recovery check
        recovery = self.check_emergency_recovery(
            frame=frame,
            player_hp=player_hp,
            player_mp=player_mp,
            active_buffs=active_buffs,
            now=t,
            use_smart_thresholds=True,
        )

        eff_hp = recovery["player_hp"]
        eff_mp = recovery["player_mp"]

        # Run base combat rotation decision
        base_decision = self.decide(
            player_mp=eff_mp,
            player_hp=eff_hp,
            ready_skills=ready_skills,
            fallback_keys=fallback_keys,
            cc_active=cc_active,
            now=t,
            target_distance=target_distance,
            target_hp=target_hp,
            is_melee_threat=is_melee_threat,
            is_casting=is_casting,
        )

        potion_keys = [k for k in [recovery.get("hp_potion_key"), recovery.get("mp_potion_key")] if k]
        buff_keys = list(recovery.get("buff_keys", []))
        is_emergency = bool(recovery.get("is_emergency", False))

        base_decision.potion_keys = potion_keys
        base_decision.buff_keys = buff_keys
        base_decision.emergency = is_emergency

        if is_emergency:
            base_decision.action_type = "emergency"
        elif potion_keys or buff_keys:
            base_decision.action_type = "consumable_buff"
        else:
            base_decision.action_type = "combat"

        recovery_keys = list(recovery.get("keys_to_press", []))
        base_decision.all_keys_to_press = recovery_keys + base_decision.skill_keys

        if recovery.get("reason") and recovery.get("reason") != "vitals_healthy":
            base_decision.reason = f"{recovery['reason']}; {base_decision.reason}"

        return base_decision

    def trigger_emergency_recovery(
        self,
        frame: Optional[np.ndarray] = None,
        player_hp: Optional[float] = None,
        player_mp: Optional[float] = None,
        action_bus: Any = None,
        send_key_fn: Optional[Callable[[str], None]] = None,
        active_buffs: Optional[Union[Collection[str], Dict[str, Any]]] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes emergency recovery and dispatches triggered potion / buff keys.
        """
        rec = self.check_emergency_recovery(
            frame=frame,
            player_hp=player_hp,
            player_mp=player_mp,
            active_buffs=active_buffs,
            now=now,
        )
        for key in rec.get("keys_to_press", []):
            if send_key_fn is not None:
                try:
                    send_key_fn(key)
                except Exception as exc:
                    logger.debug("trigger_emergency_recovery send_key_fn error: %s", exc)
            elif action_bus is not None:
                if hasattr(action_bus, "cast_skill"):
                    try:
                        action_bus.cast_skill(key)
                    except Exception as exc:
                        logger.debug("trigger_emergency_recovery action_bus cast_skill error: %s", exc)
                elif hasattr(action_bus, "press_key"):
                    try:
                        action_bus.press_key(key)
                    except Exception as exc:
                        logger.debug("trigger_emergency_recovery action_bus press_key error: %s", exc)
        return rec

    @staticmethod
    def estimate_target_distance_from_bbox(
        bbox: Optional[Union[Tuple[int, int, int, int], Sequence[float], Dict[str, Any]]],
        frame_shape: Tuple[int, int],
    ) -> Optional[float]:
        """
        Estimates distance to monster (in meters) from bounding box height.
        bbox: (x, y, w, h) in pixels or normalized [0..1], or dict with 'bbox' / 'h'.
        """
        if bbox is None:
            return None
        bh: Optional[float] = None
        if isinstance(bbox, dict):
            if "h" in bbox:
                bh = float(bbox["h"])
            elif "bbox" in bbox and len(bbox["bbox"]) >= 4:
                bh = float(bbox["bbox"][3])
        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            bh = float(bbox[3])

        if bh is None or bh <= 0:
            return None

        fh = frame_shape[0] if frame_shape and len(frame_shape) >= 1 else 1080
        norm_h = float(bh) if bh <= 1.0 else float(bh) / max(1.0, float(fh))
        if norm_h <= 0.001:
            return 30.0
        # Character model height ~ 1.8m. At 8m distance, norm_h ≈ 0.12.
        dist = round(0.96 / max(0.03, min(0.60, norm_h)), 1)
        return max(1.0, min(35.0, dist))

    def evaluate_kiting(
        self,
        player_hp: float,
        player_mp: float,
        target_distance: Optional[float] = None,
        target_hp: float = 1.0,
        is_melee_threat: bool = False,
        is_casting: bool = False,
        now: Optional[float] = None,
    ) -> KitingDecision:
        """
        Dynamic Combat Kiting distance adjustments:
        - Emergency Vitality Kiting: HP < kiting_hp_threshold (50%) -> retreat jump / backpedal / evasive strafe.
        - Distance-Based Kiting:
            - target_distance < min_dist (3.5m) -> backpedal ('s') to regain spacing.
            - target_distance > max_dist (15.0m) -> close gap ('w') to bring monster into spell range.
        - Mana Conservation Kiting: MP < 30% -> circle-strafe ('a'/'d') to avoid melee swings during regen.
        """
        if not self.kiting_enabled or is_casting:
            return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="disabled_or_casting")

        t = time.time() if now is None else float(now)
        if (t - self._last_kiting_ts) < self.kiting_cooldown_sec:
            return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="kiting_cooldown")

        hp = max(0.0, min(1.0, float(player_hp)))
        mp = max(0.0, min(1.0, float(player_mp)))

        # 1. Emergency Vitality Kiting (HP < 50%)
        if hp < self.kiting_hp_threshold:
            self._last_kiting_ts = t
            if hp < 0.25:
                # Severe panic -> jump retreat
                decision = KitingDecision(
                    should_kite=True,
                    action=KitingAction.RETREAT_JUMP.value,
                    movement_keys=["s", "space"],
                    duration_sec=0.60,
                    reason=f"critical_hp_panic={hp:.2f}<0.25",
                )
            else:
                self._strafe_toggle = not self._strafe_toggle
                strafe_key = "a" if self._strafe_toggle else "d"
                decision = KitingDecision(
                    should_kite=True,
                    action=KitingAction.BACKPEDAL.value,
                    movement_keys=["s", strafe_key],
                    duration_sec=0.45,
                    reason=f"emergency_hp_kite={hp:.2f}<{self.kiting_hp_threshold:.2f}",
                )
            msg = f"COMBAT_KITE: {decision.action} ({decision.reason})"
            logger.info(msg)
            if callable(self._on_log):
                try:
                    self._on_log(msg)
                except Exception:
                    pass
            return decision

        # 2. Distance-based adjustments
        if target_distance is not None:
            dist = float(target_distance)
            if dist < self.kiting_min_dist or (is_melee_threat and dist < 4.5):
                self._last_kiting_ts = t
                decision = KitingDecision(
                    should_kite=True,
                    action=KitingAction.BACKPEDAL.value,
                    movement_keys=["s"],
                    duration_sec=0.35,
                    reason=f"too_close_dist={dist:.1f}<{self.kiting_min_dist:.1f}",
                )
                msg = f"COMBAT_KITE: {decision.action} ({decision.reason})"
                logger.info(msg)
                if callable(self._on_log):
                    try:
                        self._on_log(msg)
                    except Exception:
                        pass
                return decision

            if dist > self.kiting_max_dist:
                self._last_kiting_ts = t
                decision = KitingDecision(
                    should_kite=True,
                    action=KitingAction.CLOSE_GAP.value,
                    movement_keys=["w"],
                    duration_sec=0.30,
                    reason=f"out_of_range_dist={dist:.1f}>{self.kiting_max_dist:.1f}",
                )
                msg = f"COMBAT_KITE: {decision.action} ({decision.reason})"
                logger.info(msg)
                if callable(self._on_log):
                    try:
                        self._on_log(msg)
                    except Exception:
                        pass
                return decision

        # 3. Mana Conservation Evasive Kiting (MP < 30%)
        if mp < self.mp_potion_threshold:
            self._last_kiting_ts = t
            self._strafe_toggle = not self._strafe_toggle
            strafe_key = "a" if self._strafe_toggle else "d"
            decision = KitingDecision(
                should_kite=True,
                action=KitingAction.STRAFE_LEFT.value if strafe_key == "a" else KitingAction.STRAFE_RIGHT.value,
                movement_keys=[strafe_key],
                duration_sec=0.30,
                reason=f"mana_conserve_strafe_mp={mp:.2f}<{self.mp_potion_threshold:.2f}",
            )
            msg = f"COMBAT_KITE: {decision.action} ({decision.reason})"
            logger.info(msg)
            if callable(self._on_log):
                try:
                    self._on_log(msg)
                except Exception:
                    pass
            return decision

        return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="distance_and_vitals_optimal")

    def apply_kiting(
        self,
        decision: KitingDecision,
        action_bus: Any = None,
        send_movement_fn: Optional[Callable[[Sequence[str], float], None]] = None,
    ) -> bool:
        """
        Execute kiting maneuver via action_bus or callback.
        """
        if not decision.should_kite or not decision.movement_keys:
            return False

        if send_movement_fn is not None:
            try:
                send_movement_fn(decision.movement_keys, decision.duration_sec)
                return True
            except Exception as exc:
                logger.debug("send_movement_fn error: %s", exc)

        if action_bus is not None:
            if hasattr(action_bus, "execute_kiting_step"):
                try:
                    action_bus.execute_kiting_step(decision.movement_keys, decision.duration_sec)
                    return True
                except Exception as exc:
                    logger.debug("action_bus execute_kiting_step error: %s", exc)
            elif hasattr(action_bus, "press_key"):
                try:
                    for k in decision.movement_keys:
                        action_bus.press_key(k)
                    return True
                except Exception as exc:
                    logger.debug("action_bus press_key kiting error: %s", exc)

        return False

    def decide(
        self,
        player_mp: float,
        player_hp: float = 1.0,
        ready_skills: Optional[Sequence[str]] = None,
        fallback_keys: Optional[Sequence[str]] = None,
        cc_active: bool = False,
        now: Optional[float] = None,
        target_distance: Optional[float] = None,
        target_hp: float = 1.0,
        is_melee_threat: bool = False,
        is_casting: bool = False,
    ) -> ManaPhaseDecision:
        mp = max(0.0, min(1.0, float(player_mp)))
        hp = max(0.0, min(1.0, float(player_hp)))
        t = time.time() if now is None else float(now)

        ready = [str(k).strip().lower() for k in (ready_skills or []) if str(k).strip()]
        fallback = [str(k).strip().lower() for k in (fallback_keys or self.combat_skill_keys) if str(k).strip()]
        if not fallback:
            fallback = list(self.DEFAULT_COMBAT_SKILL_KEYS)

        # Sticky: once below threshold, stay in regen until full again.
        if self._phase == ManaPhase.REGEN:
            if mp >= 0.999 or mp >= self.mp_full_threshold:
                self._phase = ManaPhase.BURST
        else:
            if mp < self.mp_full_threshold:
                self._phase = ManaPhase.REGEN

        if self._phase == ManaPhase.REGEN:
            # Regen phase: LMB auto-attack, cast low-mana / basic skills in round-robin
            pool = ready if (ready and len(ready) > 1) else fallback
            filtered = self._filter_skills(pool, allow=("attack_low_mana", "basic", "attack"))
            if self._lexicon:
                low = [k for k in filtered if self._lexicon.get(k.lower()) in ("attack_low_mana", "basic")]
                if low:
                    filtered = low
                else:
                    filtered = [
                        k
                        for k in filtered
                        if self._lexicon.get(k.lower(), "attack") not in ("scroll_buff",)
                        and "burst" not in self._lexicon.get(k.lower(), "")
                        and "high" not in self._lexicon.get(k.lower(), "")
                    ]
            if not filtered:
                filtered = fallback[:2]

            burst_keys = self.get_round_robin_burst_slice(count=2, pool=filtered, allow_combo=False, now=t)
            decision = ManaPhaseDecision(
                phase=ManaPhase.REGEN,
                hold_left=True,
                hold_right=False,
                skill_keys=burst_keys[:2],
                reason=f"mp={mp:.2f}<{self.mp_full_threshold:.2f}",
                primary_skill=burst_keys[0] if burst_keys else None,
                combo_active=False,
                burst_delay_sec=self.burst_skill_interval_sec,
                gcd_delay_sec=self.gcd_delay_sec,
            )
        else:
            # Burst phase: RMB attack, full round-robin hotbar rotation across 1,2,3,4,q,e,r
            # Anti-sticky '2' check:
            # If ready_skills only has 1 key (e.g. ['2']), do NOT lock into ['2']!
            # Instead, blend ready with full 7-key round-robin sequence so other skills rotate.
            if ready:
                if len(ready) == 1 and ready[0] == self._last_cast_key and len(fallback) > 1:
                    # Sticky single-key guard: slot already cast, blend in remaining sequence
                    active_pool = ready + [k for k in fallback if k not in ready]
                else:
                    active_pool = ready + [k for k in fallback if k not in ready]
            else:
                active_pool = list(fallback)

            filtered = self._filter_skills(active_pool, allow=("attack", "burst", "chain", "stigma"))
            if not filtered:
                filtered = list(active_pool)

            # Panic interrupt: defense first if low HP or CC
            defense_key: Optional[str] = None
            if hp < self.hp_panic_threshold or cc_active:
                defense_candidates = [
                    k
                    for k in (ready or fallback)
                    if self._lexicon.get(k.lower(), "") in ("defense_heal", "defiance", "heal", "defense")
                ]
                if defense_candidates:
                    defense_key = defense_candidates[0]

            burst_keys = self.get_round_robin_burst_slice(count=3, pool=filtered, allow_combo=True, now=t)
            if defense_key and defense_key not in burst_keys[:1]:
                burst_keys = [defense_key] + [k for k in burst_keys if k != defense_key][:2]

            reason = "cc" if cc_active else (f"panic hp={hp:.2f}" if hp < self.hp_panic_threshold else f"mp={mp:.2f} burst")
            decision = ManaPhaseDecision(
                phase=ManaPhase.BURST,
                hold_left=False,
                hold_right=True,
                skill_keys=burst_keys[:3],
                reason=reason,
                primary_skill=burst_keys[0] if burst_keys else None,
                combo_active=bool(self._active_combo_chain and t <= self._combo_expires_at),
                burst_delay_sec=self.burst_skill_interval_sec,
                gcd_delay_sec=self.gcd_delay_sec,
            )

        # Auto-potion dynamic check (HP < 50%, MP < 30%)
        potion_res = self.check_auto_potions(player_hp=hp, player_mp=mp, now=t)
        decision.potion_keys = list(potion_res.keys_to_press)
        decision.player_hp = hp
        decision.player_mp = mp

        # Combat kiting distance evaluation
        decision.kiting = self.evaluate_kiting(
            player_hp=hp,
            player_mp=mp,
            target_distance=target_distance,
            target_hp=target_hp,
            is_melee_threat=is_melee_threat,
            is_casting=is_casting,
            now=t,
        )

        self._emit_phase(decision)
        return decision

    def _filter_skills(self, keys: Sequence[str], allow: Sequence[str]) -> List[str]:
        if not self._lexicon:
            return list(keys)
        allow_set = {a.lower() for a in allow}
        out = []
        for k in keys:
            tag = self._lexicon.get(str(k).lower(), "attack")
            if tag in allow_set or (tag.startswith("attack") and "attack" in allow_set):
                out.append(str(k))
        return out or list(keys)

    def _emit_phase(self, decision: ManaPhaseDecision) -> None:
        if decision.phase == self._last_log_phase:
            return
        self._last_log_phase = decision.phase
        msg = f"MANA_PHASE {decision.phase.value} ({decision.reason})"
        logger.info(msg)
        if callable(self._on_log):
            try:
                self._on_log(msg)
            except Exception:
                pass

    def should_cast_skill(self, min_gap: Optional[float] = None, now: Optional[float] = None) -> bool:
        """
        Sub-100ms Burst Gate.
        
        Uses deterministic millisecond timing (self.burst_skill_interval_sec, default 0.08s / 0.15s).
        NO hidden Gaussian delay — ensures precise millisecond skill bursts.
        """
        if min_gap is not None:
            gap = max(0.02, min(1.0, float(min_gap)))
        else:
            gap = max(0.02, min(1.0, float(self.burst_skill_interval_sec)))
        t = time.time() if now is None else float(now)
        # 1e-6 epsilon guards against IEEE 754 precision artifacts at millisecond scale
        if (t - self._last_skill_ts) < (gap - 1e-6):
            return False
        self._last_skill_ts = t
        return True

    def is_gcd_ready(self, now: Optional[float] = None) -> bool:
        """Returns True if GCD delay (~0.12s) has elapsed since last skill cast."""
        t = time.time() if now is None else float(now)
        return (t - self._last_cast_ts) >= (self.gcd_delay_sec - 1e-6)

    def gauss_skill_gap(self, mean: float = 0.08, std: float = 0.02) -> float:
        """
        Legacy Gaussian jitter helper.
        Retained for backward compatibility; should_cast_skill() now defaults
        to deterministic sub-100ms timing (0.08s).
        """
        gap = random.gauss(mean, std)
        return max(0.03, min(0.25, gap))

    def apply_to_bus(
        self,
        decision: ManaPhaseDecision,
        action_bus: Any,
        sync_right: Optional[Callable[[bool], None]] = None,
        sync_left: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Apply hold_left / hold_right via bus or callbacks."""
        if sync_right is not None:
            try:
                sync_right(bool(decision.hold_right))
            except Exception as exc:
                logger.debug("sync_right failed: %s", exc)
        if sync_left is not None:
            try:
                sync_left(bool(decision.hold_left))
            except Exception as exc:
                logger.debug("sync_left failed: %s", exc)
        elif action_bus is not None and hasattr(action_bus, "hold_left_attack"):
            try:
                action_bus.hold_left_attack(bool(decision.hold_left))
            except Exception as exc:
                logger.debug("hold_left_attack failed: %s", exc)
        if action_bus is not None and hasattr(action_bus, "hold_basic_attack"):
            # hold_basic_attack historically means right-click basic; only when burst.
            try:
                if decision.hold_right:
                    action_bus.hold_basic_attack(True)
                else:
                    action_bus.hold_basic_attack(False)
            except Exception:
                pass

        # Dispatch auto-potions if triggered
        if getattr(decision, "potion_keys", None):
            for pkey in decision.potion_keys:
                if action_bus is not None:
                    if hasattr(action_bus, "cast_skill"):
                        try:
                            action_bus.cast_skill(pkey)
                        except Exception as exc:
                            logger.debug("action_bus cast_skill potion error: %s", exc)
                    elif hasattr(action_bus, "press_key"):
                        try:
                            action_bus.press_key(pkey)
                        except Exception as exc:
                            logger.debug("action_bus press_key potion error: %s", exc)

        # Dispatch buff scrolls if triggered
        if getattr(decision, "buff_keys", None):
            for bkey in decision.buff_keys:
                if action_bus is not None:
                    if hasattr(action_bus, "cast_skill"):
                        try:
                            action_bus.cast_skill(bkey)
                        except Exception as exc:
                            logger.debug("action_bus cast_skill buff error: %s", exc)
                    elif hasattr(action_bus, "press_key"):
                        try:
                            action_bus.press_key(bkey)
                        except Exception as exc:
                            logger.debug("action_bus press_key buff error: %s", exc)

        # Dispatch kiting movement if active
        kiting = getattr(decision, "kiting", None)
        if kiting and getattr(kiting, "should_kite", False):
            self.apply_kiting(kiting, action_bus=action_bus)

    def get_calibrated_roi_bounds(self) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
        """
        Dynamically returns current normalized ROI bounds for player_hp_bar,
        player_mp_bar, and target_hp_bar from config/calibrated_rois.json (auto-reloading on mtime change).
        """
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            cm = get_calibrated_rois()
            return {
                "player_hp_bar": cm.get_normalized_rect("player_hp_bar") or (0.345, 0.880, 0.485, 0.915),
                "player_mp_bar": cm.get_normalized_rect("player_mp_bar") or (0.515, 0.880, 0.655, 0.915),
                "target_hp_bar": cm.get_normalized_rect("target_hp_bar") or (0.350, 0.030, 0.650, 0.075),
                "skill_bar": cm.get_normalized_rect("skill_bar") or (0.355, 0.905, 0.655, 0.955),
            }
        except Exception:
            return {
                "player_hp_bar": (0.345, 0.880, 0.485, 0.915),
                "player_mp_bar": (0.515, 0.880, 0.655, 0.915),
                "target_hp_bar": (0.350, 0.030, 0.650, 0.075),
                "skill_bar": (0.355, 0.905, 0.655, 0.955),
            }

    def read_vitals_from_frame(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Dynamically reads player HP ratio [0.0..1.0], MP ratio [0.0..1.0], and target HP ratio [0.0..1.0]
        from frame using config/calibrated_rois.json.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "player_hp": 1.0,
                "player_mp": 1.0,
                "target_hp": 0.0,
                "has_target": 0.0,
                "hp_potion_needed": False,
                "mp_potion_needed": False,
                "is_vitality_critical": False,
            }

        try:
            rois = self.get_calibrated_roi_bounds()
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Player HP
            hp_rect = rois.get("player_hp_bar") or (0.345, 0.880, 0.485, 0.915)
            hy1, hy2 = int(h * hp_rect[1]), int(h * hp_rect[3])
            hx1, hx2 = int(w * hp_rect[0]), int(w * hp_rect[2])
            hp_crop = hsv[hy1:hy2, hx1:hx2]
            hp_ratio = 1.0
            if hp_crop.size > 0:
                red_mask = ((hp_crop[:, :, 0] < 12) | (hp_crop[:, :, 0] > 168)) & (hp_crop[:, :, 1] > 110) & (hp_crop[:, :, 2] > 90)
                col_fill = np.mean(red_mask, axis=0) > 0.25
                hp_ratio = round(float(np.sum(col_fill)) / float(max(1, len(col_fill))), 3)

            # Player MP
            mp_rect = rois.get("player_mp_bar") or (0.515, 0.880, 0.655, 0.915)
            my1, my2 = int(h * mp_rect[1]), int(h * mp_rect[3])
            mx1, mx2 = int(w * mp_rect[0]), int(w * mp_rect[2])
            mp_crop = hsv[my1:my2, mx1:mx2]
            mp_ratio = 1.0
            if mp_crop.size > 0:
                blue_mask = (mp_crop[:, :, 0] > 85) & (mp_crop[:, :, 0] < 135) & (mp_crop[:, :, 1] > 100) & (mp_crop[:, :, 2] > 80)
                col_fill_mp = np.mean(blue_mask, axis=0) > 0.25
                mp_ratio = round(float(np.sum(col_fill_mp)) / float(max(1, len(col_fill_mp))), 3)

            # Target HP
            tgt_rect = rois.get("target_hp_bar") or (0.350, 0.030, 0.650, 0.075)
            ty1, ty2 = int(h * tgt_rect[1]), int(h * tgt_rect[3])
            tx1, tx2 = int(w * tgt_rect[0]), int(w * tgt_rect[2])
            tgt_crop = hsv[ty1:ty2, tx1:tx2]
            target_hp = 0.0
            has_target = 0.0
            if tgt_crop.size > 0:
                t_mask = (
                    (((tgt_crop[:, :, 0] < 12) | (tgt_crop[:, :, 0] > 168)) & (tgt_crop[:, :, 1] > 120))
                    | ((tgt_crop[:, :, 0] >= 28) & (tgt_crop[:, :, 0] <= 65) & (tgt_crop[:, :, 1] >= 90))
                ) & (tgt_crop[:, :, 2] >= 90)
                t_fill = np.mean(t_mask, axis=0) > 0.25
                t_ratio = float(np.sum(t_fill)) / float(max(1, len(t_fill)))
                if t_ratio > 0.05:
                    has_target = 1.0
                    target_hp = round(t_ratio, 3)

            hp_potion_needed = hp_ratio < self.hp_potion_threshold
            mp_potion_needed = mp_ratio < self.mp_potion_threshold
            is_critical = hp_ratio < 0.25

            return {
                "player_hp": hp_ratio,
                "player_mp": mp_ratio,
                "target_hp": target_hp,
                "has_target": has_target,
                "hp_potion_needed": hp_potion_needed,
                "mp_potion_needed": mp_potion_needed,
                "is_vitality_critical": is_critical,
            }
        except Exception as exc:
            logger.debug("read_vitals_from_frame failed: %s", exc)
            return {
                "player_hp": 1.0,
                "player_mp": 1.0,
                "target_hp": 0.0,
                "has_target": 0.0,
                "hp_potion_needed": False,
                "mp_potion_needed": False,
                "is_vitality_critical": False,
            }

    def decide_from_frame(
        self,
        frame: Optional[np.ndarray],
        ready_skills: Optional[Sequence[str]] = None,
        fallback_keys: Optional[Sequence[str]] = None,
        cc_active: bool = False,
        now: Optional[float] = None,
        target_distance: Optional[float] = None,
        target_bbox: Optional[Union[Tuple[int, int, int, int], Sequence[float], Dict[str, Any]]] = None,
        is_melee_threat: bool = False,
        is_casting: bool = False,
    ) -> ManaPhaseDecision:
        """Reads vitals from frame using calibrated ROIs and makes combat phase decision."""
        vitals = self.read_vitals_from_frame(frame)
        dist = target_distance
        if dist is None and target_bbox is not None and frame is not None and getattr(frame, "size", 0) > 0:
            dist = self.estimate_target_distance_from_bbox(target_bbox, frame.shape[:2])

        return self.decide(
            player_mp=vitals.get("player_mp", 1.0),
            player_hp=vitals.get("player_hp", 1.0),
            ready_skills=ready_skills,
            fallback_keys=fallback_keys,
            cc_active=cc_active,
            now=now,
            target_distance=dist,
            target_hp=vitals.get("target_hp", 1.0),
            is_melee_threat=is_melee_threat,
            is_casting=is_casting,
        )

    def is_valid_monster_target(
        self,
        frame: Optional[np.ndarray],
        target_crop: Optional[np.ndarray] = None,
        bbox: Optional[Union[Tuple[int, int, int, int], Sequence[float], Dict[str, Any]]] = None,
    ) -> Tuple[bool, float]:
        """
        Uses ExemplarMemory positive and negative exemplars to recognize 3D monsters (10: monster_3d)
        and reject learned false targets (e.g. friendly NPCs, rocks, negative exemplars, or resources).
        Returns: (is_valid: bool, confidence: float)
        """
        crop = target_crop
        if crop is None and frame is not None and bbox is not None:
            try:
                from dataset_factory.exemplar_memory import extract_box_crop
                crop = extract_box_crop(frame, bbox)
            except Exception:
                if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                    h, w = frame.shape[:2]
                    bx, by, bw, bh = [int(v) for v in bbox]
                    crop = frame[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)]

        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return True, 0.70

        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
            # 1. Negative Exemplar Check (rejected false targets, rocks, friendly NPCs)
            is_neg, neg_sim, _ = mem.is_negative_patch(crop, threshold=0.72)
            if is_neg:
                return False, round(max(0.05, 1.0 - float(neg_sim)), 3)

            # 2. Positive Exemplar Check
            pos_cid, pos_cname, pos_sim = mem.classify_patch(crop, min_sim=0.68)
            if pos_cid is not None:
                cname_lower = str(pos_cname or "").lower()
                if pos_cid == 10 or "monster" in cname_lower or "mob" in cname_lower:
                    # Confirmed learned monster
                    return True, round(float(np.clip(max(0.85, pos_sim * 1.05), 0.85, 0.99)), 3)
                elif pos_cid in (0, 1, 2, 3, 4, 5, 14) and pos_sim >= 0.75:
                    # Resource node or obstacle, not a monster!
                    return False, round(max(0.05, 1.0 - float(pos_sim)), 3)

            return True, 0.75
        except Exception as exc:
            logger.debug("is_valid_monster_target exception: %s", exc)
            return True, 0.70

    def filter_monster_targets(
        self,
        frame: Optional[np.ndarray],
        candidates: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Filters and ranks candidate monster targets using ExemplarMemory.
        Returns a list of validated target dictionaries ordered by confidence (descending).
        """
        if not candidates or frame is None or frame.size == 0:
            return []

        results: List[Dict[str, Any]] = []
        for cand in candidates:
            cand_dict = dict(cand) if isinstance(cand, dict) else {"bbox": cand}
            is_valid, conf = self.is_valid_monster_target(frame, bbox=cand_dict)
            if is_valid:
                cand_dict["is_valid_monster"] = True
                cand_dict["monster_confidence"] = conf
                cand_dict["confidence"] = max(float(cand_dict.get("confidence", 0.70)), conf)
                results.append(cand_dict)

        results.sort(key=lambda x: float(x.get("monster_confidence", 0.0)), reverse=True)
        return results

    def is_skill_slot_ready_exemplar(
        self,
        slot_crop: Optional[np.ndarray],
    ) -> Tuple[bool, float, str]:
        """
        Consults ExemplarMemory for skill slot state:
        Returns: (is_ready: bool, confidence: float, state: "ready" | "cooldown" | "negative" | "unknown")
        """
        if slot_crop is None or not isinstance(slot_crop, np.ndarray) or slot_crop.size == 0:
            return True, 0.50, "unknown"

        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
            is_neg, neg_sim, _ = mem.is_negative_patch(slot_crop, threshold=0.72)
            if is_neg:
                return False, float(neg_sim), "negative"

            pos_cid, pos_cname, pos_sim = mem.classify_patch(slot_crop, min_sim=0.68)
            if pos_cid is not None:
                cname = str(pos_cname or "").lower()
                if "cooldown" in cname or pos_cid == 21 or "empty" in cname:
                    return False, float(pos_sim), "cooldown"
                if "ready" in cname or pos_cid in (20, 13) or "skill" in cname:
                    return True, float(pos_sim), "ready"

            return True, 0.50, "unknown"
        except Exception:
            return True, 0.50, "unknown"

