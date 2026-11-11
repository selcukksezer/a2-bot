# -*- coding: utf-8 -*-
"""
ManaPhaseCombat — MP eşiğine göre Regen (sol tık) / Burst (sağ tık) faz kapısı.

Mevcut sağ-tık + GameActionBus skill rotasyonunu silmez; yalnızca faz seçer.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# F-0540 tek kaynak: HP/MP iksir eşikleri config/schema.py sabitlerinden
# okunur (sözleşme no 10: HP 65 / MP 45). Düz ve smart yol aynı değere
# çözülür; 0.50/0.30 kurucu varsayılanları kaldırıldı.
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT
from core.exemplar_threshold import is_negative_exemplar_name, load_exemplar_match_threshold

logger = logging.getLogger("Gatherion.ManaPhaseCombat")


def _load_exemplar_match_threshold(default: float = 0.68) -> float:
    """Tek exemplar-benzerlik eşiği (P3-3: `core.exemplar_threshold` yardımcısına delege).

    Eskiden bu dosyada negatif 0.72, pozitif 0.68, kaynak/engel 0.75 olarak
    üç ayrı değer vardı.
    """
    return load_exemplar_match_threshold(default)


EXEMPLAR_MATCH_THRESHOLD: float = _load_exemplar_match_threshold()


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
    # P0-6: kararın seçtiği tuşlar, türüyle birlikte — gönderenin onayı
    # (mark_potions_sent) hangi damganın yazılacağını bunlardan bilir.
    hp_potion_key: Optional[str] = None
    mp_potion_key: Optional[str] = None


# Default 8-key hotbar sequence and sub-100ms burst timing constants
DEFAULT_COMBAT_SKILL_KEYS: Tuple[str, ...] = ("1", "2", "3", "4", "q", "e", "r", "t")
DEFAULT_BURST_INTERVAL_SEC: float = 0.08  # 80ms sub-100ms burst delay
DEFAULT_GCD_DELAY_SEC: float = 0.12       # 120ms global cooldown
DEFAULT_COMBO_CHAINS: Dict[str, List[str]] = {}

# Slots the bot must never press as a consumable: X / F6 / B are automatic
# client slots (or reserved). Shared by check_auto_potions and
# check_emergency_recovery so the two paths can never drift apart.
FORBIDDEN_CONSUMABLE_KEYS = frozenset({"x", "f6", "b"})


def _send_confirmed(result: Any) -> bool:
    """P0-6 / B-02: gönderim dönüşünü "tuş gerçekten gönderildi mi" kararına çevirir.

    Gönderen açıkça başarısızlık bildirmediği sürece gönderildi sayılır
    (v18 burst kuralı: non-False = gönderildi):
    - ``dict``: ``"ok"`` alanı belirler (``GameActionBus.cast_skill`` —
      ``ok=True`` yalnızca tuş/tıklama gerçekten gittiyse; ``ok=False`` =
      gönderilmedi, örn. yalnızca R temel-saldırı nabzı çıktı, istenen
      tuş basılmadı; ``"ok"`` yoksa da onay yoktur).
    - ``False`` / ``0`` / ``0.0``: gönderilmedi (sürücü/bus açık raporu;
      ``InputController.press_key`` 0.0 = gönderilmedi).
    - ``None`` / diğer: rapor yok — istisna fırlatılmadıysa gönderildi
      sayılır (dönüşsüz ``send_key_fn`` callback'leri ile ``None`` dönen
      bus'lar bu dala girer).
    """
    if result is None:
        return True
    if isinstance(result, dict):
        return bool(result.get("ok"))
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return float(result) > 0.0
    return bool(result)


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
    # P0-6: potion_keys içindeki HP/MP tuşu (hangisi hangisi, gönderenin
    # onayı mark_potions_sent'e türü söyler; yoksa yalnızca slot damgası yazılır).
    hp_potion_key: Optional[str] = None
    mp_potion_key: Optional[str] = None
    buff_keys: List[str] = field(default_factory=list)
    emergency: bool = False
    action_type: str = "combat"
    all_keys_to_press: List[str] = field(default_factory=list)


class ManaPhaseCombat:
    """
    Decide regen vs burst and manage millisecond skill rotation & burst timing.
    
    Features:
    - Sub-100ms burst delays (0.08s default, configurable 0.08s-0.15s, gcd ~0.12s).
    - 8-key round-robin sequence across hotbar ('1', '2', '3', '4', 'q', 'e', 'r', 't').
    - Combo chain support (e.g. 1 -> 2 -> 3, q -> e -> r) with auto-advancement and expiry.
    - Zero sticky '2' key bias: fair cycle rotation and anti-repetition protection.
    - Deterministic millisecond pacing with no hidden Gaussian delays.
    - SmartConsumableManager integration: dynamically tracks upper consumable hotbar (F1..F8, X, B),
      triggering HP potion (<65%, emergency <40%), MP potion (<45%), and re-applying expired buff scrolls.
      Per-potion toggles (route.auto_use_hp/mp_potion), buff auto-refresh
      (route.auto_refresh_buff_scroll) and scroll interval/key
      (route.buff_scroll_interval_sec / buff_scroll_key) are config-driven
      (P1-13); before P1-13 they existed in schema/UI but were never read
      in production.
    """

    DEFAULT_COMBAT_SKILL_KEYS = DEFAULT_COMBAT_SKILL_KEYS
    DEFAULT_BURST_INTERVAL_SEC = DEFAULT_BURST_INTERVAL_SEC
    DEFAULT_GCD_DELAY_SEC = DEFAULT_GCD_DELAY_SEC

    def __init__(
        self,
        mp_full_threshold: float = 0.95,
        hp_panic_threshold: float = 0.50,
        on_log: Optional[Callable[[str], None]] = None,
        burst_skill_interval_sec: Optional[float] = None,
        gcd_delay_sec: float = DEFAULT_GCD_DELAY_SEC,
        combat_skill_keys: Optional[Sequence[str]] = None,
        combo_chains: Optional[Dict[str, Sequence[str]]] = None,
        hp_potion_threshold: Optional[float] = None,
        mp_potion_threshold: Optional[float] = None,
        hp_potion_key: str = "F1",
        mp_potion_key: str = "F2",
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
        smart_hp_potion_threshold: Optional[float] = None,
        smart_mp_potion_threshold: Optional[float] = None,
        emergency_hp_threshold: float = 0.40,
        mp_regen_exit_threshold: Optional[float] = None,
        # P1-13: route.* tüketim anahtarları (şema + UI'da vardı ama
        # üretimde hiç okunmuyordu) — varsayılanlar eski davranışı korur.
        auto_use_hp_potion: bool = True,
        auto_use_mp_potion: bool = True,
        auto_refresh_buff_scroll: bool = True,
        buff_scroll_interval_sec: float = 300.0,
        buff_scroll_key: str = "F7",
    ) -> None:
        self.mp_full_threshold = float(mp_full_threshold)
        # Hysteresis: enter REGEN below mp_full_threshold, leave only near-full.
        # Exit must be >= entry: an exit below the entry threshold flips
        # REGEN->BURST->REGEN every tick at constant MP (measured A/B/A/B
        # at mp=0.60 with exit=0.50, entry=0.95).
        self.mp_regen_exit_threshold = max(
            self.mp_full_threshold,
            0.99 if mp_regen_exit_threshold is None else float(mp_regen_exit_threshold),
        )
        self.hp_panic_threshold = float(hp_panic_threshold)
        self._on_log = on_log
        self._phase = ManaPhase.BURST
        self._last_log_phase: Optional[ManaPhase] = None

        # Auto-Potion Triggers & Vitality Monitoring (F-0540 tek kaynak:
        # HP 65%, MP 45% — None gelirse config/schema.py sabitine çözülür.)
        self.hp_potion_threshold = (
            float(HP_POTION_THRESHOLD_PCT) / 100.0
            if hp_potion_threshold is None
            else float(hp_potion_threshold)
        )
        self.mp_potion_threshold = (
            float(MP_POTION_THRESHOLD_PCT) / 100.0
            if mp_potion_threshold is None
            else float(mp_potion_threshold)
        )
        self.hp_potion_key = str(hp_potion_key).strip() or "F1"
        self.mp_potion_key = str(mp_potion_key).strip() or "F2"
        self.hp_potion_cd_sec = float(hp_potion_cd_sec)
        self.mp_potion_cd_sec = float(mp_potion_cd_sec)
        self._last_hp_potion_ts: float = 0.0
        self._last_mp_potion_ts: float = 0.0
        self._last_recovery_eval_ts: float = 0.0

        # v20 (HP yanlış okuma): tek karelik düşük-HP okumasının iksir /
        # kaçış tetiklemesini engelleyen doğrulama durumu. Kural,
        # state_machine._note_player_hp_drop / _publish_worker_hp_ratio'daki
        # F-0067/F-0376 deseninin aynısıdır (ardışık 2 örnek + sağlıklı
        # taban); taban burada iksir eşiğidir (canlıda HP 65, §7 kararı),
        # 0.5 değil — böylece %60'lık tek karelik yanlış okuma da iksir
        # tetiklemez. Gerçek düşüş kademeli seyrettiği için (önceki örnek
        # de eşik altındadır) en fazla 1 tick gecikir.
        self._hp_confirm_prev: Optional[float] = None
        self._hp_confirm_streak: int = 0
        self._hp_confirm_ts: Optional[float] = None
        self._hp_confirm_hp: Optional[float] = None
        self._hp_confirm_verdict: bool = True

        # Smart Consumables & Emergency Recovery (F-0540 tek kaynak: smart yol
        # düz yolla aynı 65/45 değerine çözülür; ayrı smart sabiti yoktur.)
        self.smart_hp_potion_threshold = (
            float(HP_POTION_THRESHOLD_PCT) / 100.0
            if smart_hp_potion_threshold is None
            else float(smart_hp_potion_threshold)
        )
        self.smart_mp_potion_threshold = (
            float(MP_POTION_THRESHOLD_PCT) / 100.0
            if smart_mp_potion_threshold is None
            else float(smart_mp_potion_threshold)
        )
        self.emergency_hp_threshold = float(emergency_hp_threshold)
        # P1-13: iksir/buff anahtarları (route.*) — ayardan beslenir;
        # set_thresholds ile canlı tazelenir.
        self.auto_use_hp_potion = bool(auto_use_hp_potion)
        self.auto_use_mp_potion = bool(auto_use_mp_potion)
        self.auto_refresh_buff_scroll = bool(auto_refresh_buff_scroll)
        self.buff_scroll_interval_sec = max(1.0, float(buff_scroll_interval_sec or 300.0))
        self.buff_scroll_key = str(buff_scroll_key or "F7").strip().upper() or "F7"
        self.exemplar_memory = exemplar_memory
        if consumable_manager is not None:
            self.consumable_manager = consumable_manager
        else:
            try:
                from core.skill_bar_reader import SmartConsumableManager
                self.consumable_manager = SmartConsumableManager(
                    exemplar_memory=self.exemplar_memory,
                    scroll_duration_sec=self.buff_scroll_interval_sec,
                    preferred_scroll_key=self.buff_scroll_key,
                )
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
        # Precedence: an explicitly passed value wins; the env var only applies
        # when the caller did not pass one (it used to silently override the
        # constructor/config value). The override is logged, never silent.
        import os
        if burst_skill_interval_sec is not None:
            self.burst_skill_interval_sec = float(burst_skill_interval_sec)
        else:
            env_burst = str(os.environ.get("GATHERION_SKILL_BURST_SEC", "") or "").strip()
            if env_burst:
                try:
                    self.burst_skill_interval_sec = max(0.02, min(0.50, float(env_burst)))
                    logger.info("GATHERION_SKILL_BURST_SEC=%s applied (no explicit value)", env_burst)
                except (TypeError, ValueError):
                    self.burst_skill_interval_sec = float(DEFAULT_BURST_INTERVAL_SEC)
            else:
                self.burst_skill_interval_sec = float(DEFAULT_BURST_INTERVAL_SEC)

        self.gcd_delay_sec = float(gcd_delay_sec)
        self._last_skill_ts = 0.0
        self._last_cast_ts = 0.0
        self._last_cast_key: Optional[str] = None

        # 8-Key Round-Robin Sequence
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

    def _log_info(self, msg: str) -> None:
        """Single choke point for info logging + the on_log callback.

        Replaces the logger.info + callable(_on_log) try/except block that
        was copy-pasted at every log site (11 sites); a failing callback
        must never break combat decisions, so the swallow stays here.
        """
        logger.info(msg)
        if callable(self._on_log):
            try:
                self._on_log(msg)
            except Exception:
                pass

    @staticmethod
    def _hsv_roi_crop(
        frame: np.ndarray, rect: Tuple[float, float, float, float], h: int, w: int
    ) -> Optional[np.ndarray]:
        """Crop one ROI from the BGR frame and convert only that crop to HSV.

        The old code converted the whole frame to HSV and then cropped the
        three bars out of it; the bars cover ~1.5% of a 1080p frame, so
        ~98% of the conversion work was thrown away on every tick.
        cvtColor is per-pixel, so cropping first gives identical masks.
        Returns None for an empty crop (caller keeps its default ratio).
        """
        y1, y2 = int(h * rect[1]), int(h * rect[3])
        x1, x2 = int(w * rect[0]), int(w * rect[2])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

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
        mp_regen_exit_threshold: Optional[float] = None,
        # P1-13: iksir/buff anahtarları (route.*).
        auto_use_hp_potion: Optional[bool] = None,
        auto_use_mp_potion: Optional[bool] = None,
        auto_refresh_buff_scroll: Optional[bool] = None,
        buff_scroll_interval_sec: Optional[float] = None,
        buff_scroll_key: Optional[str] = None,
    ) -> None:
        if mp_full_threshold is not None:
            self.mp_full_threshold = max(0.5, min(1.0, float(mp_full_threshold)))
        if mp_regen_exit_threshold is not None:
            # Same invariant as __init__: exit >= entry, else the phase flaps.
            self.mp_regen_exit_threshold = max(
                self.mp_full_threshold, min(1.0, float(mp_regen_exit_threshold))
            )
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
        # P1-13: iksir/buff anahtarları — None "değiştirme" demektir.
        if auto_use_hp_potion is not None:
            self.auto_use_hp_potion = bool(auto_use_hp_potion)
        if auto_use_mp_potion is not None:
            self.auto_use_mp_potion = bool(auto_use_mp_potion)
        if auto_refresh_buff_scroll is not None:
            self.auto_refresh_buff_scroll = bool(auto_refresh_buff_scroll)
        if buff_scroll_interval_sec is not None:
            self.buff_scroll_interval_sec = max(1.0, float(buff_scroll_interval_sec))
            if self.consumable_manager is not None and hasattr(
                self.consumable_manager, "set_scroll_duration_sec"
            ):
                try:
                    self.consumable_manager.set_scroll_duration_sec(self.buff_scroll_interval_sec)
                except Exception:
                    pass
        if buff_scroll_key is not None:
            key = str(buff_scroll_key or "").strip().upper()
            if key:
                self.buff_scroll_key = key
                if self.consumable_manager is not None and hasattr(
                    self.consumable_manager, "set_preferred_scroll_key"
                ):
                    try:
                        self.consumable_manager.set_preferred_scroll_key(key)
                    except Exception:
                        pass

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
        - Never repeats a key within one slice: when the pool is smaller
          than `count` the slice is shorter than `count` (callers must not
          assume a full slice). Padding a 1-key pool to ["1", "1"] made the
          same skill dispatch twice in one tick (D03 added that padding;
          removed: a decision's skill_keys must be duplicate-free).
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

            # Advance the global rotation index by collected count.
            # v20-loop: pool_len <= collected iken ilerleme
            # (start + collected) % pool_len = start oluyordu — [2,q]
            # havuzunda slice hep [2,q] kalıyordu (canlı kanıt). İlerleme
            # 0'a düşerse 1'e zorlanır ki başlangıç noktası dönsün.
            advance = max(1, collected) % pool_len
            if advance == 0:
                advance = 1
            self._rotation_idx = (start_idx + advance) % pool_len

        return result

    def _hp_low_reading_confirmed(self, hp: float, now: Optional[float] = None) -> bool:
        """Tek karelik düşük-HP okumasının gerçek düşüş mü okuma hatası mı
        olduğunu söyler (v20, HP yanlış okuma).

        Canlı v19: dolu HP 0.00/0.11/0.34 okunup F1 + RETREAT_JUMP
        tetikleniyordu. İksir eşiğinin (canlıda 0.65) üstünden altına
        TEK karede düşen okuma, ardışık 2. örnek de eşik altında
        kalmadıkça doğrulanmaz. Kural, state_machine._note_player_hp_drop
        / _publish_worker_hp_ratio'daki F-0067/F-0376 desenidir (ardışık
        2 + sağlıklı taban); yalnızca taban iksir eşiğidir, 0.5 değil —
        böylece %60'lık tek karelik yanlış okuma da iksir tetiklemez.

        - hp >= eşik: sağlıklı, sayaç sıfırlanır, True.
        - İlk örnek (geçmiş yok): eski tek-çağrı davranışı korunur, True
          (mevcut testler tek çağrıda tetikleme bekler).
        - Zaten eşik altında seyrediyorsa (kademeli düşüş): True.
        - Aynı tick içinde aynı `now` ile tekrar çağrılırsa (decide()
          iksir + kite'ı aynı `t` ile sorar) önbellekteki hüküm döner —
          sayaç bir tick'te iki kez ilerlemez. `now` verilmezse her çağrı
          ayrı örnek sayılır.
        """
        hp_f = max(0.0, min(1.0, float(hp)))
        if (
            now is not None
            and self._hp_confirm_ts is not None
            and float(now) == self._hp_confirm_ts
            and self._hp_confirm_hp is not None
            and hp_f == self._hp_confirm_hp
        ):
            return self._hp_confirm_verdict
        baseline = float(
            getattr(self, "smart_hp_potion_threshold", None)
            or self.hp_potion_threshold
        )
        prev = self._hp_confirm_prev
        if hp_f >= baseline:
            self._hp_confirm_streak = 0
            verdict = True
        elif prev is None:
            verdict = True
        elif prev >= baseline:
            self._hp_confirm_streak += 1
            verdict = self._hp_confirm_streak >= 2
        else:
            self._hp_confirm_streak += 1
            verdict = True
        self._hp_confirm_prev = hp_f
        self._hp_confirm_ts = float(now) if now is not None else None
        self._hp_confirm_hp = hp_f
        self._hp_confirm_verdict = bool(verdict)
        return self._hp_confirm_verdict

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
        - HP < hp_potion_threshold (F-0540 tek kaynak, varsayılan 0.65 / 65%): trigger HP potion key ('F1').
        - MP < mp_potion_threshold (F-0540 tek kaynak, varsayılan 0.45 / 45%): trigger MP potion key ('F2').
        Keys match the production fallback and the learned consumable slots.
        Respects independent per-potion cooldowns (default 10.0s).
        HP/MP branches are gated by route.auto_use_hp/mp_potion (P1-13).

        Pure decision (P0-6 / K-13): this method reads the cooldown state
        but never writes a stamp — a chosen key counts as used only after
        the sender actually dispatches it, via ``mark_potions_sent``.
        Deciding without sending must not consume the 10s cooldown (the old
        decision-time stamp suppressed the potion — and, through the shared
        ``_last_hp/mp_potion_ts``, the live state_machine B-02 path too —
        for 10s even when the key never left the bot).
        """
        t = time.time() if now is None else float(now)
        raw_hp = player_hp if player_hp is not None else (hp_ratio if hp_ratio is not None else 1.0)
        raw_mp = player_mp if player_mp is not None else (mp_ratio if mp_ratio is not None else 1.0)
        hp = max(0.0, min(1.0, float(raw_hp)))
        mp = max(0.0, min(1.0, float(raw_mp)))

        # v20: tek karelik düşük-HP okuması iksir tetiklemez — HP
        # bilinmiyorsa (None) 1.0 varsayımı zaten yukarıda yapılır;
        # buradaki kapı yalnızca ölçülen düşük değerin doğrulanmasıdır.
        hp_low_confirmed = self._hp_low_reading_confirmed(hp, now=t)

        hp_trigger = False
        mp_trigger = False
        keys: List[str] = []
        reasons: List[str] = []
        chosen_hp_key: Optional[str] = None
        chosen_mp_key: Optional[str] = None

        if getattr(self, "auto_use_hp_potion", True) and hp < self.hp_potion_threshold and hp_low_confirmed:
            if (t - self._last_hp_potion_ts) >= self.hp_potion_cd_sec:
                if str(self.hp_potion_key).lower() not in FORBIDDEN_CONSUMABLE_KEYS:
                    hp_trigger = True
                    chosen_hp_key = self.hp_potion_key
                    keys.append(self.hp_potion_key)
                    reasons.append(f"HP {hp*100:.1f}% < {self.hp_potion_threshold*100:.0f}%")
                    self._log_info(f"AUTO_POTION: HP triggered [{self.hp_potion_key}] (hp={hp:.2f})")

        if getattr(self, "auto_use_mp_potion", True) and mp < self.mp_potion_threshold:
            if (t - self._last_mp_potion_ts) >= self.mp_potion_cd_sec:
                if str(self.mp_potion_key).lower() not in FORBIDDEN_CONSUMABLE_KEYS:
                    mp_trigger = True
                    chosen_mp_key = self.mp_potion_key
                    keys.append(self.mp_potion_key)
                    reasons.append(f"MP {mp*100:.1f}% < {self.mp_potion_threshold*100:.0f}%")
                    self._log_info(f"AUTO_POTION: MP triggered [{self.mp_potion_key}] (mp={mp:.2f})")

        return AutoPotionResult(
            hp_potion_triggered=hp_trigger,
            mp_potion_triggered=mp_trigger,
            keys_to_press=keys,
            hp_ratio=hp,
            mp_ratio=mp,
            reason="; ".join(reasons) if reasons else "vitals_ok",
            hp_potion_key=chosen_hp_key,
            mp_potion_key=chosen_mp_key,
        )

    def mark_potions_sent(
        self,
        sent_keys: Sequence[str] = (),
        *,
        hp_key: Optional[str] = None,
        mp_key: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[str]:
        """P0-6: yalnızca gerçekten gönderilmiş iksir tuşları için cooldown damgası yazar.

        Karar fonksiyonları (``check_auto_potions`` /
        ``check_emergency_recovery``) saf karardır — damga buraya aittir ve
        yalnızca gönderen ``_send_confirmed`` ile onayı alınmış tuşları
        geçirir. Gönderilmemiş / başarısız tuş damgalanmaz: cooldown
        başlamaz, bir sonraki tick'te yeniden denenir (canlı
        ``state_machine._check_and_trigger_auto_potions`` B-02 kuralıyla aynı).
        - ``hp_key`` / ``mp_key`` verildiyse ilgili ``_last_*_potion_ts``
          damgası yazılır (``consumable_manager`` yokken tek kapı budur).
        - ``sent_keys`` (hp/mp tuşları dahil) ``consumable_manager`` varsa
          onun ``mark_keys_pressed`` yolundan slot cooldown'u alır — v18'de
          eklenen yardımcı, buff_scroll ataması görürse buff uygular.
        Damgalanan tuşları (normalize edilmeden, verildiği haliyle) döner.
        """
        t = time.time() if now is None else float(now)
        stamped: List[str] = []
        for key in list(sent_keys or []):
            if key is None:
                continue
            k = str(key)
            if k and k not in stamped:
                stamped.append(k)
        if hp_key is not None:
            self._last_hp_potion_ts = t
            if str(hp_key) not in stamped:
                stamped.append(str(hp_key))
        if mp_key is not None:
            self._last_mp_potion_ts = t
            if str(mp_key) not in stamped:
                stamped.append(str(mp_key))
        if stamped and self.consumable_manager is not None:
            mark = getattr(self.consumable_manager, "mark_keys_pressed", None)
            if callable(mark):
                try:
                    mark(stamped, now=t)
                except Exception as exc:
                    logger.debug("mark_keys_pressed failed for %s: %s", stamped, exc)
        return stamped

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

        P0-6: cooldown damgası yalnızca gönderimi onaylanmış tuş için yazılır
        (``mark_potions_sent``) — gönderilemeyen tuş (istisna, ``False``/``0``
        dönüşü, ``{"ok": False}`` raporu, gönderim metodu olmayan bus)
        damgalanmaz, bir sonraki çağrıda yeniden denenir.
        """
        res = self.check_auto_potions(
            player_hp=player_hp,
            player_mp=player_mp,
            now=now,
            hp_ratio=hp_ratio,
            mp_ratio=mp_ratio,
        )
        t = time.time() if now is None else float(now)
        for key in res.keys_to_press:
            sent = False
            if send_key_fn is not None:
                try:
                    sent = _send_confirmed(send_key_fn(key))
                except Exception as exc:
                    logger.debug("send_key_fn potion error: %s", exc)
            elif action_bus is not None:
                if hasattr(action_bus, "cast_skill"):
                    try:
                        sent = _send_confirmed(action_bus.cast_skill(key))
                    except Exception as exc:
                        logger.debug("action_bus cast_skill potion error: %s", exc)
                elif hasattr(action_bus, "press_key"):
                    try:
                        sent = _send_confirmed(action_bus.press_key(key))
                    except Exception as exc:
                        logger.debug("action_bus press_key potion error: %s", exc)
            if sent:
                self.mark_potions_sent(
                    [key],
                    hp_key=key if res.hp_potion_key is not None and str(key) == str(res.hp_potion_key) else None,
                    mp_key=key if res.mp_potion_key is not None and str(key) == str(res.mp_potion_key) else None,
                    now=t,
                )
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
          (e.g. 'F5', 'F7'; F6 is an automatic client slot and forbidden)
          to re-apply the buff automatically!
        HP/MP/buff branches are gated by route.auto_use_hp_potion,
        route.auto_use_mp_potion and route.auto_refresh_buff_scroll (P1-13);
        the `*_needed` / `is_emergency` diagnostics still reflect vitals.

        Pure decision (P0-6 / K-13): this method reads the cooldown state
        but never writes a stamp — no ``_last_*_potion_ts`` and no
        ``consumable_manager.mark_used`` here. A chosen key counts as used
        only after the sender actually dispatches it, via
        ``mark_potions_sent`` (``trigger_emergency_recovery`` /
        ``apply_to_bus`` do this). Deciding without sending must not consume
        the 10s cooldown (the old decision-time stamp suppressed the potion
        — and, through the shared ``_last_hp/mp_potion_ts``, the live
        state_machine B-02 path too — for 10s even when the key never left
        the bot). Buff scrolls were already pure here
        (``check_and_refresh_scroll_buffs`` is a query); the buff-applied
        stamp still happens only after a real press, at the send site.

        Returns diagnostics dictionary with:
        `is_emergency`, `hp_potion_needed`, `mp_potion_needed`, `buff_expired`,
        `hp_potion_key`, `mp_potion_key`, `buff_keys`, `keys_to_press`, `reason`.
        """
        t = time.time() if now is None else float(now)
        # Mark this tick as potion-evaluated: decide() called in the same tick
        # (decide_combat_action) must not re-run check_auto_potions against the
        # same cooldown stamps and silently discard its own output.
        self._last_recovery_eval_ts = t

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

        # v20: tek karelik düşük-HP okuması iksir anahtarı seçtirmez.
        # Tanı alanları (is_emergency / hp_potion_needed) ölçümü olduğu
        # gibi yansıtır; yalnızca tuş gönderimi doğrulanmayı bekler.
        hp_low_confirmed = self._hp_low_reading_confirmed(hp, now=t)

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
        # P1-13: route.auto_use_hp_potion kapalıyken iksir basılmaz.
        # v20: tek karelik düşük-HP okuması (doğrulanmamış uçurum)
        # anahtar seçtirmez — bkz. _hp_low_reading_confirmed.
        if hp_potion_needed and getattr(self, "auto_use_hp_potion", True) and hp_low_confirmed:
            learned_hp = self.consumable_manager.get_best_hp_potion(now=t) if self.consumable_manager is not None else None
            candidate_key = learned_hp or self.hp_potion_key
            if candidate_key and str(candidate_key).lower() in FORBIDDEN_CONSUMABLE_KEYS:
                # A learned forbidden slot (e.g. 'X') must not block the
                # configured fallback: retry with hp_potion_key itself.
                fallback_key = self.hp_potion_key
                candidate_key = fallback_key if str(fallback_key).lower() not in FORBIDDEN_CONSUMABLE_KEYS else None

            cd_ready = (t - self._last_hp_potion_ts) >= self.hp_potion_cd_sec
            if self.consumable_manager is not None and candidate_key and candidate_key in self.consumable_manager.CONSUMABLE_SLOTS:
                cd_ready = cd_ready and self.consumable_manager.is_slot_ready(candidate_key, now=t)

            if cd_ready and candidate_key:
                hp_triggered = True
                chosen_hp_key = candidate_key
                severity = "EMERGENCY" if is_emergency else "AUTO_POTION"
                reasons.append(f"{severity}: HP {hp*100:.1f}% < {hp_thresh*100:.0f}% -> [{chosen_hp_key}]")
                self._log_info(f"{severity}: HP triggered [{chosen_hp_key}] (hp={hp:.2f})")

        # 3. MP Potion Check (< 45%)
        # P1-13: route.auto_use_mp_potion kapalıyken iksir basılmaz.
        if mp_potion_needed and getattr(self, "auto_use_mp_potion", True):
            learned_mp = self.consumable_manager.get_best_mp_potion(now=t) if self.consumable_manager is not None else None
            candidate_key = learned_mp or self.mp_potion_key
            if candidate_key and str(candidate_key).lower() in FORBIDDEN_CONSUMABLE_KEYS:
                # Same fallback as HP above: a learned forbidden slot must
                # not block the configured mp_potion_key.
                fallback_key = self.mp_potion_key
                candidate_key = fallback_key if str(fallback_key).lower() not in FORBIDDEN_CONSUMABLE_KEYS else None

            cd_ready = (t - self._last_mp_potion_ts) >= self.mp_potion_cd_sec
            if self.consumable_manager is not None and candidate_key and candidate_key in self.consumable_manager.CONSUMABLE_SLOTS:
                cd_ready = cd_ready and self.consumable_manager.is_slot_ready(candidate_key, now=t)

            if cd_ready and candidate_key:
                mp_triggered = True
                chosen_mp_key = candidate_key
                reasons.append(f"AUTO_POTION: MP {mp*100:.1f}% < {mp_thresh*100:.0f}% -> [{chosen_mp_key}]")
                self._log_info(f"AUTO_POTION: MP triggered [{chosen_mp_key}] (mp={mp:.2f})")

        # 4. Expired Buff Scroll Check (buff_expired -> re-apply automatically)
        # P1-13: route.auto_refresh_buff_scroll kapalıyken parşömen yenilenmez.
        if getattr(self, "auto_refresh_buff_scroll", True) and self.consumable_manager is not None:
            expired_scrolls = self.consumable_manager.check_and_refresh_scroll_buffs(active_buffs=active_buffs, now=t)
            expired_scrolls = [s for s in (expired_scrolls or []) if str(s).lower() not in FORBIDDEN_CONSUMABLE_KEYS]
            if expired_scrolls:
                buff_expired = True
                for skey in expired_scrolls:
                    chosen_buff_keys.append(skey)
                    reasons.append(f"BUFF_REAPPLY: Scroll expired -> [{skey}]")
                    self._log_info(f"BUFF_REAPPLY: Scroll triggered [{skey}]")

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
        base_decision.hp_potion_key = recovery.get("hp_potion_key")
        base_decision.mp_potion_key = recovery.get("mp_potion_key")
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

        P0-6: cooldown/buff stamps happen only for keys whose dispatch was
        confirmed (``_send_confirmed``) — a key that raised, returned
        ``False``/``0`` or ``{"ok": False}`` is not stamped, so the next
        tick retries it instead of consuming its cooldown.
        """
        rec = self.check_emergency_recovery(
            frame=frame,
            player_hp=player_hp,
            player_mp=player_mp,
            active_buffs=active_buffs,
            now=now,
        )
        buff_keys = set(rec.get("buff_keys", []) or [])
        t = time.time() if now is None else float(now)
        dispatched_potions: List[str] = []
        for key in rec.get("keys_to_press", []):
            dispatched = False
            if send_key_fn is not None:
                try:
                    dispatched = _send_confirmed(send_key_fn(key))
                except Exception as exc:
                    logger.debug("trigger_emergency_recovery send_key_fn error: %s", exc)
            elif action_bus is not None:
                if hasattr(action_bus, "cast_skill"):
                    try:
                        dispatched = _send_confirmed(action_bus.cast_skill(key))
                    except Exception as exc:
                        logger.debug("trigger_emergency_recovery action_bus cast_skill error: %s", exc)
                elif hasattr(action_bus, "press_key"):
                    try:
                        dispatched = _send_confirmed(action_bus.press_key(key))
                    except Exception as exc:
                        logger.debug("trigger_emergency_recovery action_bus press_key error: %s", exc)
            # Only a key that was actually dispatched counts as applied: the
            # check function no longer marks buffs by itself (F-0173).
            if dispatched and key in buff_keys and self.consumable_manager is not None:
                try:
                    self.consumable_manager.mark_buff_applied(key, now=t)
                except Exception as exc:
                    logger.debug("mark_buff_applied failed for %s: %s", key, exc)
            if dispatched and key not in buff_keys:
                dispatched_potions.append(key)
        if dispatched_potions:
            self.mark_potions_sent(
                dispatched_potions,
                hp_key=rec.get("hp_potion_key") if rec.get("hp_potion_key") in dispatched_potions else None,
                mp_key=rec.get("mp_potion_key") if rec.get("mp_potion_key") in dispatched_potions else None,
                now=t,
            )
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
        - Mana Conservation Kiting: MP < mp_potion_threshold (F-0540: 45%)
          -> circle-strafe ('a'/'d') to avoid melee swings during regen.
        """
        if not self.kiting_enabled or is_casting:
            return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="disabled_or_casting")

        t = time.time() if now is None else float(now)
        hp = max(0.0, min(1.0, float(player_hp)))
        mp = max(0.0, min(1.0, float(player_mp)))

        # v20: tek karelik düşük-HP okuması kaçış tetiklemez.
        hp_low_confirmed = self._hp_low_reading_confirmed(hp, now=t)

        # 0. Critical-HP panic (HP < 0.25) bypasses the kiting cooldown: a
        # cooldown must never suppress the panic escape itself.
        if hp < 0.25 and hp_low_confirmed:
            self._last_kiting_ts = t
            decision = KitingDecision(
                should_kite=True,
                action=KitingAction.RETREAT_JUMP.value,
                movement_keys=["s", "space"],
                duration_sec=0.60,
                reason=f"critical_hp_panic={hp:.2f}<0.25",
            )
            self._log_info(f"COMBAT_KITE: {decision.action} ({decision.reason})")
            return decision

        if (t - self._last_kiting_ts) < self.kiting_cooldown_sec:
            return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="kiting_cooldown")

        # 1. Emergency Vitality Kiting (HP < 50%; HP < 0.25 panic handled above)
        # v20: tek karelik düşük-HP okuması geri çekilme de tetiklemez.
        if hp < self.kiting_hp_threshold and hp_low_confirmed:
            self._last_kiting_ts = t
            self._strafe_toggle = not self._strafe_toggle
            strafe_key = "a" if self._strafe_toggle else "d"
            decision = KitingDecision(
                should_kite=True,
                action=KitingAction.BACKPEDAL.value,
                movement_keys=["s", strafe_key],
                duration_sec=0.45,
                reason=f"emergency_hp_kite={hp:.2f}<{self.kiting_hp_threshold:.2f}",
            )
            self._log_info(f"COMBAT_KITE: {decision.action} ({decision.reason})")
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
                self._log_info(f"COMBAT_KITE: {decision.action} ({decision.reason})")
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
                self._log_info(f"COMBAT_KITE: {decision.action} ({decision.reason})")
                return decision

        # 3. Mana Conservation Evasive Kiting (MP < mp_potion_threshold,
        # F-0540: 45% — kiting bu eşiği iksir eşiğiyle paylaşır, ayrı sabiti yok).
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
            self._log_info(f"COMBAT_KITE: {decision.action} ({decision.reason})")
            return decision

        return KitingDecision(should_kite=False, action=KitingAction.NONE.value, reason="distance_and_vitals_optimal")

    def _send_movement_via_driver(
        self,
        driver: Any,
        keys: Sequence[str],
        duration_sec: float,
    ) -> bool:
        """Hold movement keys via a driver-like object (S-3001).

        Production route: ``GameActionBus`` exposes no
        ``execute_kiting_step`` / ``press_key`` of its own, but it carries
        ``input_driver`` — the real ``InputDriver`` whose ``key_down`` /
        ``key_up`` / ``press_key`` go through the hardware (Interception /
        HID) path. ``InputDriver.key_down`` has no soft fallthrough
        (returns False when no trusted path exists), so this route never
        silently degrades to soft SendInput.

        Preferred: ``press_key(key, hold_sec=duration)`` (one held press
        per key, return value checked — 0.0 / False means nothing was
        sent). Fallback: ``key_down`` all keys, sleep, ``key_up`` all
        (``finally`` releases every key pressed, so a failure mid-hold
        can never leave a movement key stuck down).
        """
        if driver is None:
            return False
        keys = [str(k) for k in keys if str(k).strip()]
        if not keys:
            return False
        duration = max(0.02, min(2.0, float(duration_sec or 0.0)))

        press = getattr(driver, "press_key", None)
        if callable(press):
            try:
                sent_all = True
                for k in keys:
                    try:
                        sent = press(k, hold_sec=duration)
                    except TypeError:
                        # Legacy (key)-only press_key signature.
                        sent = press(k)
                    if sent is False or sent == 0.0:
                        sent_all = False
                        break
                if sent_all:
                    return True
                logger.debug("driver press_key kiting aborted for %s", keys)
            except Exception as exc:
                logger.debug("driver press_key kiting error: %s", exc)

        down = getattr(driver, "key_down", None)
        up = getattr(driver, "key_up", None)
        if not (callable(down) and callable(up)):
            return False
        pressed: List[str] = []
        try:
            for k in keys:
                if down(k) is False:
                    logger.debug("driver key_down(%s) rejected; kiting not sent", k)
                    return False
                pressed.append(k)
            time.sleep(duration)
            return True
        except Exception as exc:
            logger.debug("driver key_down/key_up kiting error: %s", exc)
            return False
        finally:
            for k in reversed(pressed):
                try:
                    up(k)
                except Exception:
                    pass

    def apply_kiting(
        self,
        decision: KitingDecision,
        action_bus: Any = None,
        send_movement_fn: Optional[Callable[[Sequence[str], float], None]] = None,
    ) -> bool:
        """
        Execute kiting maneuver via action_bus or callback.

        Routes, in order: explicit ``send_movement_fn``; bus
        ``execute_kiting_step``; bus ``press_key`` (hold-aware, return
        checked); the bus's ``input_driver`` / ``input_port`` hardware
        driver (S-3001 — this is the production ``GameActionBus`` route:
        the bus itself has no movement method, but its driver does).
        ``cast_skill`` is deliberately NOT a route: it pulses a skill
        key for ~40ms through the skill-cast path (exclusive lock, Alt+
        click fallback) and cannot hold a movement key for the kiting
        duration — routing WASD through it would fake a kite.
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
                # Hold-aware: movement needs the key HELD for
                # duration_sec, not tapped; a 0.0 / False return means
                # the press was aborted, so fall through to the next
                # route instead of claiming success.
                try:
                    sent_all = True
                    for k in decision.movement_keys:
                        try:
                            sent = action_bus.press_key(k, hold_sec=decision.duration_sec)
                        except TypeError:
                            sent = action_bus.press_key(k)
                        if sent is False or sent == 0.0:
                            sent_all = False
                            break
                    if sent_all:
                        return True
                    logger.debug("action_bus press_key kiting aborted for %s", decision.movement_keys)
                except Exception as exc:
                    logger.debug("action_bus press_key kiting error: %s", exc)
            # S-3001: production GameActionBus has neither of the above,
            # but its input_driver is the hardware keyboard path.
            for driver_attr in ("input_driver", "input_port"):
                driver = getattr(action_bus, driver_attr, None)
                if driver is not None and self._send_movement_via_driver(
                    driver, decision.movement_keys, decision.duration_sec
                ):
                    return True

        # F-0122: kiting was decided but no route could execute it — say so
        # loudly instead of silently dropping the maneuver (the old code only
        # debug-logged inside the except branches, so a bus without
        # execute_kiting_step / press_key swallowed every kite).
        logger.warning(
            "COMBAT_KITE dropped: no movement route (no send_movement_fn, "
            "no action_bus.execute_kiting_step / press_key / input_driver) for %s",
            decision.movement_keys,
        )
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

        # v20/empty-slots: vizyonun hazır listesi yapılandırılmış tuş
        # kümesinin dışına taşamaz. Canlı v19 kanıtı: knobs
        # keys=1,2,3,4,q,e,r,t iken '6'/'8' basıldı — decide() ready'yi
        # havuza aynen alıyordu (active_pool = ready + fallback). Burst
        # yolu _send_configured_combat_skill filtreler ama mana-kararı→bus
        # yolu filtrelemiyordu; bu kesişim iki yolu aynı sözleşmeye
        # bağlar. Ayar 6/7/8'i içeriyorsa fallback onları kapsar, yine basılır.
        _fallback_set = set(fallback)
        ready = [k for k in ready if k in _fallback_set]

        # Hysteresis: once below the entry threshold, stay in regen until
        # (nearly) full again — exiting at the entry threshold caused flapping.
        if self._phase == ManaPhase.REGEN:
            if mp >= self.mp_regen_exit_threshold:
                self._phase = ManaPhase.BURST
        else:
            if mp < self.mp_full_threshold:
                self._phase = ManaPhase.REGEN

        if self._phase == ManaPhase.REGEN:
            # Regen phase: LMB auto-attack, cast low-mana / basic skills in round-robin
            # v20/madde 2: ready tekilse (okuyucu yalnızca '2'yi hazır
            # görüyorsa) havuz o tek tuşa çökmesin — BURST dalındaki gibi
            # ready-öncelikli harman; lexicon filtresi REGEN disiplinini
            # (yalnızca düşük-mana/basic) ayrıca korur.
            pool = (ready + [k for k in fallback if k not in ready]) if ready else fallback
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
                # S145 (v19/P3-8): eski `fallback[:2]` lexicon
                # filtresini baypas ediyordu — REGEN'de (düşük mana)
                # burst/high/scroll_buff skill'i seçilebiliyordu.
                # Yedek de aynı dışlamadan geçirilir; hiçbir şey
                # kalmazsa boş kalır (LMB oto-saldırı sürer).
                if self._lexicon:
                    filtered = [
                        k
                        for k in fallback
                        if self._lexicon.get(k.lower(), "attack") not in ("scroll_buff",)
                        and "burst" not in self._lexicon.get(k.lower(), "")
                        and "high" not in self._lexicon.get(k.lower(), "")
                    ][:2]
                else:
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
            # Anti-sticky '2' guard lives in get_round_robin_burst_slice
            # (it skips the last-cast key / _last_cast_key as the first pick
            # when alternatives exist); blending ready with the fallback pool
            # here is the same in both cases, so there is no separate
            # single-key branch (the old if/else here had byte-identical branches).
            if ready:
                active_pool = ready + [k for k in fallback if k not in ready]
            else:
                active_pool = list(fallback)

            filtered = self._filter_skills(active_pool, allow=("attack", "burst", "chain", "stigma"))
            if not filtered:
                filtered = list(active_pool)
            else:
                # v20-loop: filtre dışı kalanlar (örn. scroll_buff 1,3)
                # atılmaz, kuyruğa alınır — BURST tam hotbar'ı döndürür
                # (karşı saldırı yolundaki gibi). S145 disiplini REGEN'de
                # korunur; BURST'te buff kuyrukta, saldırı skill'i önce.
                filtered = self._order_with_tail(active_pool, filtered)

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

        # Auto-potion dynamic check (F-0540: HP < 65%, MP < 45%). Skipped when
        # check_emergency_recovery already evaluated potions this tick
        # (decide_combat_action path): running both against the same cooldown
        # stamps discarded one path's output.
        if t == self._last_recovery_eval_ts:
            potion_res = AutoPotionResult(hp_ratio=hp, mp_ratio=mp, reason="potion_evaluated_by_recovery")
        else:
            potion_res = self.check_auto_potions(player_hp=hp, player_mp=mp, now=t)
        decision.potion_keys = list(potion_res.keys_to_press)
        # P0-6: tür bilgisi kararla birlikte taşınır ki apply_to_bus
        # gönderim onayı sonrası doğru damgayı yazabilsin.
        decision.hp_potion_key = potion_res.hp_potion_key
        decision.mp_potion_key = potion_res.mp_potion_key
        # Same contract as decide_combat_action: all_keys_to_press carries
        # every key this decision wants pressed (was: always [] here, so a
        # caller reading it from decide() saw no keys at all).
        decision.all_keys_to_press = decision.potion_keys + decision.skill_keys
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
        # v20/madde 2 tanı: her kararda hazır liste + seçilen tuşlar
        # (tek tuşa kilitlenme canlı logdan okunur).
        logger.debug(
            "ROTATION phase=%s ready=%s skill_keys=%s",
            decision.phase.value,
            ready,
            list(decision.skill_keys),
        )
        return decision

    def _filter_skills(self, keys: Sequence[str], allow: Sequence[str]) -> List[str]:
        if not self._lexicon:
            return list(keys)
        allow_set = {a.lower() for a in allow}
        out = []
        for k in keys:
            tag = self._lexicon.get(str(k).lower(), "attack")
            # v20-loop: "unknown" (sınıflandırılmamış) elenmez — sözlükte
            # OLMAYAN tuş varsayılan "attack" sayılırken sözlükte
            # "unknown" yazan tuşun elenmesi tutarsızdı. Canlı lexicon
            # (1,3=scroll_buff; 4,5,e,r=unknown) BURST/REGEN havuzunu
            # [2,q]'ya çökertiyordu: 4,5,e,r hiçbir filtreden geçemiyor,
            # 1,3 scroll_buff diye atılıyordu. "unknown" artık "attack"
            # sayılır — oyunun kendi cooldown'u gerçek kapıdır.
            if tag == "unknown":
                tag = "attack"
            if tag in allow_set or (tag.startswith("attack") and "attack" in allow_set):
                out.append(str(k))
        # No silent fallback to the unfiltered list: callers already fall back
        # explicitly when the filter empties the pool. Returning the full list
        # here made lexicon-based filtering a no-op.
        return out

    def _order_with_tail(
        self, pool: Sequence[str], preferred: Sequence[str]
    ) -> List[str]:
        """Lexicon tercihine göre sırala ama tuş ATMA: tercih edilenler
        önce, kalanlar kuyrukta (orijinal sırayla). Havuzun [2,q] gibi
        2 tuşa çökmesi, round-robin'in de ilerleyememesi demekti
        (ilerleme = collected % pool_len = 0). Kuyruk sayesinde BURST
        tam hotbar'ı (karşı saldırı yolundaki gibi) döndürür; tercih
        sırası korunur — saldırı skill'leri buff'lardan önce gelir."""
        preferred_list = [str(k) for k in preferred]
        seen = set(preferred_list)
        tail = [str(k) for k in pool if str(k) not in seen]
        return preferred_list + tail

    def _emit_phase(self, decision: ManaPhaseDecision) -> None:
        if decision.phase == self._last_log_phase:
            return
        self._last_log_phase = decision.phase
        self._log_info(f"MANA_PHASE {decision.phase.value} ({decision.reason})")

    def is_cast_gate_ready(self, min_gap: Optional[float] = None, now: Optional[float] = None) -> bool:
        """Pure (side-effect-free) burst-gate probe: True when enough time has
        elapsed since the last skill. Use this when only *checking* the gate;
        `should_cast_skill` consumes the gate (see below)."""
        if min_gap is not None:
            gap = max(0.02, min(1.0, float(min_gap)))
        else:
            gap = max(0.02, min(1.0, float(self.burst_skill_interval_sec)))
        t = time.time() if now is None else float(now)
        return (t - self._last_skill_ts) >= (gap - 1e-6)

    def should_cast_skill(self, min_gap: Optional[float] = None, now: Optional[float] = None) -> bool:
        """
        Sub-100ms Burst Gate (consuming).

        Uses deterministic millisecond timing (self.burst_skill_interval_sec, default 0.08s / 0.15s).
        NO hidden Gaussian delay — ensures precise millisecond skill bursts.

        NOTE: a successful check consumes the gate (updates `_last_skill_ts`),
        so call it only when a cast will actually follow. For a mere probe
        use `is_cast_gate_ready()`, which has no side effects.
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

    def apply_to_bus(
        self,
        decision: ManaPhaseDecision,
        action_bus: Any,
        sync_right: Optional[Callable[[bool], None]] = None,
        sync_left: Optional[Callable[[bool], None]] = None,
        send_movement_fn: Optional[Callable[[Sequence[str], float], None]] = None,
    ) -> Dict[str, Any]:
        """Apply hold_left / hold_right via bus or callbacks.

        F-0122: consumes the WHOLE decision — holds, potion keys, buff keys,
        combat skill keys and kiting. Nothing the brain decided is dropped
        on the floor. Returns a summary of what was actually dispatched so
        callers (and tests) can verify consumption instead of assuming it.
        ``send_movement_fn`` is the kiting movement route for buses that
        expose no ``execute_kiting_step`` / ``press_key`` (S-3001: the
        production GameActionBus exposes neither itself, but
        ``apply_kiting`` now also drives its ``input_driver`` hardware
        path, so kiting no longer depends on the caller supplying one).
        """
        dispatched: Dict[str, Any] = {
            "holds_applied": False,
            "potion_keys": [],
            "buff_keys": [],
            "skill_keys": [],
            "kiting_applied": False,
        }
        if sync_right is not None:
            try:
                sync_right(bool(decision.hold_right))
                dispatched["holds_applied"] = True
            except Exception as exc:
                logger.debug("sync_right failed: %s", exc)
        if sync_left is not None:
            try:
                sync_left(bool(decision.hold_left))
                dispatched["holds_applied"] = True
            except Exception as exc:
                logger.debug("sync_left failed: %s", exc)
        elif action_bus is not None and hasattr(action_bus, "hold_left_attack"):
            try:
                action_bus.hold_left_attack(bool(decision.hold_left))
                dispatched["holds_applied"] = True
            except Exception as exc:
                logger.debug("hold_left_attack failed: %s", exc)
        if action_bus is not None and hasattr(action_bus, "hold_basic_attack"):
            # hold_basic_attack is the T-key basic attack hold (right-click
            # attack == T key per the R/T rule; the bus no longer uses the mouse).
            try:
                if decision.hold_right:
                    action_bus.hold_basic_attack(True)
                else:
                    action_bus.hold_basic_attack(False)
                dispatched["holds_applied"] = True
            except Exception:
                pass

        # Dispatch auto-potions if triggered. P0-6: cooldown damgası yalnızca
        # gönderimi onaylanmış tuş için yazılır (mark_potions_sent) — istisna,
        # False/0 dönüşü veya {"ok": False} raporu damgalanmaz, tuş bir
        # sonraki tick'te yeniden denenir. dispatched["potion_keys"] ile
        # damga aynı kuralı kullanır (onaysız tuş hiçbirinde görünmez).
        if getattr(decision, "potion_keys", None):
            hp_key = getattr(decision, "hp_potion_key", None)
            mp_key = getattr(decision, "mp_potion_key", None)
            for pkey in decision.potion_keys:
                sent = False
                if action_bus is not None:
                    if hasattr(action_bus, "cast_skill"):
                        try:
                            sent = _send_confirmed(action_bus.cast_skill(pkey))
                        except Exception as exc:
                            logger.debug("action_bus cast_skill potion error: %s", exc)
                    elif hasattr(action_bus, "press_key"):
                        try:
                            sent = _send_confirmed(action_bus.press_key(pkey))
                        except Exception as exc:
                            logger.debug("action_bus press_key potion error: %s", exc)
                if sent:
                    dispatched["potion_keys"].append(pkey)
                    try:
                        self.mark_potions_sent(
                            [pkey],
                            hp_key=pkey if hp_key is not None and str(pkey) == str(hp_key) else None,
                            mp_key=pkey if mp_key is not None and str(pkey) == str(mp_key) else None,
                            now=time.time(),
                        )
                    except Exception as exc:
                        logger.debug("mark_potions_sent failed for %s: %s", pkey, exc)

        # Dispatch buff scrolls if triggered
        if getattr(decision, "buff_keys", None):
            for bkey in decision.buff_keys:
                buff_sent = False
                if action_bus is not None:
                    if hasattr(action_bus, "cast_skill"):
                        try:
                            buff_sent = _send_confirmed(action_bus.cast_skill(bkey))
                        except Exception as exc:
                            logger.debug("action_bus cast_skill buff error: %s", exc)
                    elif hasattr(action_bus, "press_key"):
                        try:
                            buff_sent = _send_confirmed(action_bus.press_key(bkey))
                        except Exception as exc:
                            logger.debug("action_bus press_key buff error: %s", exc)
                if buff_sent:
                    dispatched["buff_keys"].append(bkey)
                # A scroll key that was actually dispatched counts as applied;
                # without this the scroll would be re-requested every tick.
                if buff_sent and self.consumable_manager is not None:
                    try:
                        self.consumable_manager.mark_buff_applied(bkey, now=time.time())
                    except Exception as exc:
                        logger.debug("mark_buff_applied failed for %s: %s", bkey, exc)

        # Dispatch combat skill keys (was: silently dropped — the decision
        # carried skill_keys but apply_to_bus never pressed them).
        # v20/madde 2: yalnızca onaylı gönderim sayılır ve kaydedilir —
        # eskiden record_skill_cast hiç çağrılmıyordu, o yüzden
        # _last_cast_key hep None kalıyor ve get_round_robin_burst_slice
        # içindeki anti-sticky '2' koruması canlıda hiç devreye
        # girmiyordu.
        if getattr(decision, "skill_keys", None):
            for skey in decision.skill_keys:
                if action_bus is not None:
                    sent = False
                    if hasattr(action_bus, "cast_skill"):
                        try:
                            sent = _send_confirmed(action_bus.cast_skill(skey))
                        except Exception as exc:
                            logger.debug("action_bus cast_skill skill error: %s", exc)
                    elif hasattr(action_bus, "press_key"):
                        try:
                            sent = _send_confirmed(action_bus.press_key(skey))
                        except Exception as exc:
                            logger.debug("action_bus press_key skill error: %s", exc)
                    if sent:
                        dispatched["skill_keys"].append(skey)
                        try:
                            self.record_skill_cast(skey)
                        except Exception as exc:
                            logger.debug("record_skill_cast failed for %s: %s", skey, exc)
        if dispatched["skill_keys"]:
            logger.debug("ROTATION dispatched skill_keys=%s", dispatched["skill_keys"])

        # Dispatch kiting movement if active
        kiting = getattr(decision, "kiting", None)
        if kiting and getattr(kiting, "should_kite", False):
            dispatched["kiting_applied"] = bool(
                self.apply_kiting(kiting, action_bus=action_bus, send_movement_fn=send_movement_fn)
            )

        return dispatched

    def decide_and_apply(
        self,
        action_bus: Any,
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
        sync_right: Optional[Callable[[bool], None]] = None,
        sync_left: Optional[Callable[[bool], None]] = None,
        send_movement_fn: Optional[Callable[[Sequence[str], float], None]] = None,
    ) -> ManaPhaseDecision:
        """Decide the combat action AND apply it to the bus, atomically.

        F-0122: the production combat loop used to call ``decide()`` and then
        only read ``decision.phase`` — ``skill_keys`` / ``potion_keys`` /
        ``kiting`` never reached the bus because ``apply_to_bus`` was never
        called. Callers must use this one-call path (or call ``apply_to_bus``
        themselves immediately after deciding); deciding without applying is
        the exact failure this method removes. Returns the decision so the
        caller can still read ``phase`` / ``reason`` for HUD/logging.
        """
        decision = self.decide_combat_action(
            frame=frame,
            player_hp=player_hp,
            player_mp=player_mp,
            ready_skills=ready_skills,
            fallback_keys=fallback_keys,
            cc_active=cc_active,
            now=now,
            target_distance=target_distance,
            target_hp=target_hp,
            is_melee_threat=is_melee_threat,
            is_casting=is_casting,
            active_buffs=active_buffs,
        )
        self.apply_to_bus(
            decision,
            action_bus,
            sync_right=sync_right,
            sync_left=sync_left,
            send_movement_fn=send_movement_fn,
        )
        return decision

    def get_calibrated_roi_bounds(self) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
        """
        Dynamically returns current normalized ROI bounds for player_hp_bar,
        player_mp_bar, and target_hp_bar from config/calibrated_rois.json (auto-reloading on mtime change).
        """
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            from core.skill_bar_reader import SkillBarReader

            cm = get_calibrated_rois()
            return {
                "player_hp_bar": cm.get_normalized_rect("player_hp_bar") or (0.345, 0.880, 0.485, 0.915),
                "player_mp_bar": cm.get_normalized_rect("player_mp_bar") or (0.515, 0.880, 0.655, 0.915),
                "target_hp_bar": cm.get_normalized_rect("target_hp_bar") or (0.350, 0.030, 0.650, 0.075),
                # Tek dogruluk kaynagi: SkillBarReader.DEFAULT_SKILL_BAR_ROI
                # (G2 12 slot: x2=0.660 T'yi kapsar, y2=0.991 slot altidir).
                "skill_bar": cm.get_normalized_rect("skill_bar") or SkillBarReader.DEFAULT_SKILL_BAR_ROI,
            }
        except Exception:
            from core.skill_bar_reader import SkillBarReader

            return {
                "player_hp_bar": (0.345, 0.880, 0.485, 0.915),
                "player_mp_bar": (0.515, 0.880, 0.655, 0.915),
                "target_hp_bar": (0.350, 0.030, 0.650, 0.075),
                "skill_bar": SkillBarReader.DEFAULT_SKILL_BAR_ROI,
            }

    def read_vitals_from_frame(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Dynamically reads player HP ratio [0.0..1.0], MP ratio [0.0..1.0], and target HP ratio [0.0..1.0]
        from frame using config/calibrated_rois.json.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            # Fail-closed: an unreadable frame must NEVER be reported as full
            # health (the old 1.0/1.0 made the bot believe it was perfectly
            # healthy during an ROI/OCR failure). vitals_valid=False tells
            # callers the numbers are unknown, not measured.
            return {
                "player_hp": 0.0,
                "player_mp": 0.0,
                "target_hp": 0.0,
                "has_target": 0.0,
                "hp_potion_needed": True,
                "mp_potion_needed": True,
                "is_vitality_critical": True,
                "vitals_valid": False,
            }

        try:
            rois = self.get_calibrated_roi_bounds()
            h, w = frame.shape[:2]
            # Player HP — convert only the bar crop, not the whole frame.
            # Crop-first HSV: only the 3 small ROI crops are converted
            # instead of the full frame (measured 39.4ms -> 7.1ms per
            # 1920x1080 call; cvtColor is per-pixel, so cropped results
            # are identical).
            hp_crop = self._hsv_roi_crop(frame, rois.get("player_hp_bar") or (0.345, 0.880, 0.485, 0.915), h, w)
            hp_ratio = 1.0
            if hp_crop is not None:
                red_mask = ((hp_crop[:, :, 0] < 12) | (hp_crop[:, :, 0] > 168)) & (hp_crop[:, :, 1] > 110) & (hp_crop[:, :, 2] > 90)
                col_fill = np.mean(red_mask, axis=0) > 0.25
                hp_ratio = round(float(np.sum(col_fill)) / float(max(1, len(col_fill))), 3)

            # Player MP
            # Player MP — same crop-first HSV helper as HP above.
            mp_crop = self._hsv_roi_crop(frame, rois.get("player_mp_bar") or (0.515, 0.880, 0.655, 0.915), h, w)
            mp_ratio = 1.0
            if mp_crop is not None:
                blue_mask = (mp_crop[:, :, 0] > 85) & (mp_crop[:, :, 0] < 135) & (mp_crop[:, :, 1] > 100) & (mp_crop[:, :, 2] > 80)
                col_fill_mp = np.mean(blue_mask, axis=0) > 0.25
                mp_ratio = round(float(np.sum(col_fill_mp)) / float(max(1, len(col_fill_mp))), 3)

            # Target HP
            # Target HP — same crop-first HSV helper as HP above.
            tgt_crop = self._hsv_roi_crop(frame, rois.get("target_hp_bar") or (0.350, 0.030, 0.650, 0.075), h, w)
            target_hp = 0.0
            has_target = 0.0
            if tgt_crop is not None:
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
            is_critical = hp_ratio < self.hp_panic_threshold

            return {
                "player_hp": hp_ratio,
                "player_mp": mp_ratio,
                "target_hp": target_hp,
                "has_target": has_target,
                "hp_potion_needed": hp_potion_needed,
                "mp_potion_needed": mp_potion_needed,
                "is_vitality_critical": is_critical,
                "vitals_valid": True,
            }
        except Exception as exc:
            logger.debug("read_vitals_from_frame failed: %s", exc)
            return {
                "player_hp": 0.0,
                "player_mp": 0.0,
                "target_hp": 0.0,
                "has_target": 0.0,
                "hp_potion_needed": True,
                "mp_potion_needed": True,
                "is_vitality_critical": True,
                "vitals_valid": False,
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
        """Reads vitals from frame using calibrated ROIs and makes combat phase decision.

        Sweep G notu: bu yöntemin üretim çağırıcısı yok — canlı savaş yolu
        `decide()` çağrısı (`core/state_machine.py:2923`, hub oranlarıyla).
        Kaldırılmadı: `test_mana_phase_combat.py` ve
        `tests/test_w53_combat_hp_mp_vitals_reader.py` bu yöntemi kareden
        karar giriş noktası olarak kullanıyor; silmek o testleri kırar.
        Canlı bağlama (state_machine'ın bu yöntemi kullanması) C grubunun
        dosyasında — koordinatöre bildirildi. Yöntem fail-closed
        (bilinmeyen vitals -> saldırı yok) ve ucuz olduğundan korundu.

        Fail-closed: unknown vitals (no/invalid frame) produce an idle decision
        — no skill keys, no attack holds, no blind potion. With no target and
        no HP panic, offensive output is stripped (hold fire); a panic
        (critical HP) keeps the defensive decision (potion + retreat).
        """
        vitals = self.read_vitals_from_frame(frame)
        if not vitals.get("vitals_valid", False):
            return ManaPhaseDecision(
                phase=ManaPhase.REGEN,
                hold_left=False,
                hold_right=False,
                skill_keys=[],
                reason="vitals_unknown_no_frame",
                primary_skill=None,
                combo_active=False,
                burst_delay_sec=self.burst_skill_interval_sec,
                gcd_delay_sec=self.gcd_delay_sec,
                potion_keys=[],
                player_hp=0.0,
                player_mp=0.0,
            )
        dist = target_distance
        if dist is None and target_bbox is not None and frame is not None and getattr(frame, "size", 0) > 0:
            dist = self.estimate_target_distance_from_bbox(target_bbox, frame.shape[:2])

        decision = self.decide(
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
        if not vitals.get("has_target", 0.0) and decision.player_hp >= self.hp_panic_threshold:
            # No target and not panicking: do not BURST at nothing. Keep
            # potion keys / kiting (defensive), drop the attack itself.
            decision.skill_keys = []
            decision.primary_skill = None
            decision.hold_left = False
            decision.hold_right = False
            decision.phase = ManaPhase.REGEN
            # Keep internal state in sync with the reported phase: the old
            # code left self._phase == BURST, so the very next decide()
            # BURSTed again at the same vitals (measured).
            self._phase = ManaPhase.REGEN
            self._last_log_phase = ManaPhase.REGEN
            decision.reason = f"{decision.reason}; no_target_hold_fire"
            decision.all_keys_to_press = list(decision.potion_keys) + list(decision.buff_keys)
        return decision

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
            # No evidence at all: fail-closed (was: True, 0.70 — every
            # unrecognised NPC / rock / resource counted as a monster).
            return False, 0.0

        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
            # 1. Negative Exemplar Check (rejected false targets, rocks, friendly NPCs)
            is_neg, neg_sim, _ = mem.is_negative_patch(crop, threshold=EXEMPLAR_MATCH_THRESHOLD)
            if is_neg:
                return False, round(max(0.05, 1.0 - float(neg_sim)), 3)

            # 2. Positive Exemplar Check
            pos_cid, pos_cname, pos_sim = mem.classify_patch(crop, min_sim=EXEMPLAR_MATCH_THRESHOLD)
            if pos_cid is not None:
                cname_lower = str(pos_cname or "").lower()
                # Sweep G: pozitif listeye sızmış (10,
                # 'monster_3d_negative') kaydı canavar kanıtı sayılmaz —
                # adı negatif olan eşleşme reddedilmiş hedeftir.
                if is_negative_exemplar_name(pos_cname):
                    return False, round(max(0.05, 1.0 - float(pos_sim)), 3)
                if pos_cid == 10 or "monster" in cname_lower or "mob" in cname_lower:
                    # Confirmed learned monster. S-3002: confidence is the
                    # measured exemplar similarity itself — the old
                    # max(0.85, sim*1.05) floor reported a barely-passing
                    # 0.68 match as 0.85 "confirmed". Strong matches still
                    # score high (held-out monster sim ~0.999 -> 0.99).
                    return True, round(float(np.clip(pos_sim, 0.0, 0.99)), 3)
                elif pos_cid in (0, 1, 2, 3, 4, 5, 14) and pos_sim >= EXEMPLAR_MATCH_THRESHOLD:
                    # Resource node or obstacle, not a monster!
                    return False, round(max(0.05, 1.0 - float(pos_sim)), 3)

            # Classified as nothing recognisable: not a confirmed monster.
            return False, 0.0
        except Exception as exc:
            logger.debug("is_valid_monster_target exception: %s", exc)
            return False, 0.0

    def filter_monster_targets(
        self,
        frame: Optional[np.ndarray],
        candidates: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Filters and ranks candidate monster targets using ExemplarMemory.
        Returns target dictionaries ordered confirmed-monsters-first, then
        unconfirmed fallbacks, each tier by confidence (descending).

        Three tiers (w52 test_5 contract; F-0121 boundary):
        1. Confirmed monster — is_valid_monster_target True: exemplar
           similarity as monster_confidence, is_valid_monster=True.
        2. Rejected — negative-exemplar match or resource/obstacle match:
           dropped (same verdicts as the strict validator).
        3. Unknown — matches no exemplar either way: kept as an explicitly
           UNCONFIRMED fallback (is_valid_monster=False,
           unconfirmed_monster=True), ranked after every confirmed monster,
           monster_confidence = the detector's own candidate confidence.
           Rationale: the detector already proposed the box as a monster
           candidate; a bot that only ever engages positively-recognised
           monsters can never fight a monster type it has not seen before.
           This does NOT reopen F-0121: is_valid_monster_target stays
           fail-closed (unknown -> (False, 0.0)) and no fallback entry is
           ever presented as a validated monster.
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
                # Sweep G: doğrulanmış güven, exemplar benzerliğinin kendisi
                # — eskiden `max(detector_conf, conf)` ile dedektör güveni
                # exemplar güvenini tek yönlü şişiriyordu.
                cand_dict["confidence"] = conf
                results.append(cand_dict)
                continue
            # Invalid per the strict validator: distinguish "rejected"
            # (negative/resource match -> drop) from "unknown" (no match at
            # all -> unconfirmed fallback). Empty crop / lookup error ->
            # drop (fail-closed, same as the validator).
            if self._is_rejected_target_crop(frame, cand_dict):
                continue
            try:
                # Sweep G: güveni bilinmeyen adaya 0.70 uydurulmaz — varsa
                # dedektör güveni, yoksa 0.0 (doğrulanmamış olduğu açık).
                raw_conf = cand_dict.get("confidence")
                detector_conf = float(raw_conf) if raw_conf is not None else 0.0
            except (TypeError, ValueError):
                detector_conf = 0.0
            cand_dict["is_valid_monster"] = False
            cand_dict["unconfirmed_monster"] = True
            cand_dict["monster_confidence"] = detector_conf
            results.append(cand_dict)

        results.sort(
            key=lambda x: (
                bool(x.get("is_valid_monster")),
                float(x.get("monster_confidence", 0.0)),
            ),
            reverse=True,
        )
        return results

    def _is_rejected_target_crop(
        self,
        frame: Optional[np.ndarray],
        cand_dict: Dict[str, Any],
    ) -> bool:
        """
        True when a candidate crop positively matches a negative exemplar
        or a resource/obstacle class — i.e. the strict validator rejected
        it for a REASON, not for lack of evidence. An empty/unreadable crop
        or any lookup error is also a rejection (fail-closed). An unknown
        crop (no exemplar match either way) returns False; the caller keeps
        it as an unconfirmed fallback. Thresholds mirror
        is_valid_monster_target exactly (single EXEMPLAR_MATCH_THRESHOLD).
        """
        try:
            from dataset_factory.exemplar_memory import (
                extract_box_crop,
                get_exemplar_memory,
            )

            try:
                crop = extract_box_crop(frame, cand_dict)
            except Exception:
                crop = None
            if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                return True

            mem = get_exemplar_memory()
            is_neg, _neg_sim, _ = mem.is_negative_patch(crop, threshold=EXEMPLAR_MATCH_THRESHOLD)
            if is_neg:
                return True

            pos_cid, pos_cname, pos_sim = mem.classify_patch(crop, min_sim=EXEMPLAR_MATCH_THRESHOLD)
            if pos_cid is not None:
                cname_lower = str(pos_cname or "").lower()
                # Sweep G: adı negatif olan eşleşme reddedilmiş hedeftir.
                if is_negative_exemplar_name(pos_cname):
                    return True
                if pos_cid == 10 or "monster" in cname_lower or "mob" in cname_lower:
                    # Unreachable with the real validator (it would have
                    # returned valid); treat as not-rejected so the caller
                    # still keeps it, ranked as a fallback.
                    return False
                if pos_cid in (0, 1, 2, 3, 4, 5, 14) and pos_sim >= EXEMPLAR_MATCH_THRESHOLD:
                    return True
            return False
        except Exception as exc:
            logger.debug("_is_rejected_target_crop exception: %s", exc)
            return True

    def is_skill_slot_ready_exemplar(
        self,
        slot_crop: Optional[np.ndarray],
    ) -> Tuple[bool, float, str]:
        """
        Consults ExemplarMemory for skill slot state:
        Returns: (is_ready: bool, confidence: float, state: "ready" | "cooldown" | "negative" | "unknown")

        S-3003: fail-CLOSED. An empty/unclassifiable slot or an
        exception used to return (True, 0.50, "unknown") — unknown was
        treated as ready. Unknown now means NOT ready; callers must
        treat the slot as unusable until it is positively recognised
        as ready. (No production caller exists yet — this is a
        diagnostic helper; the fail-closed default is what makes it
        safe to wire in later.)
        """
        if slot_crop is None or not isinstance(slot_crop, np.ndarray) or slot_crop.size == 0:
            return False, 0.0, "unknown"

        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
            is_neg, neg_sim, _ = mem.is_negative_patch(slot_crop, threshold=EXEMPLAR_MATCH_THRESHOLD)
            if is_neg:
                return False, float(neg_sim), "negative"

            pos_cid, pos_cname, pos_sim = mem.classify_patch(slot_crop, min_sim=EXEMPLAR_MATCH_THRESHOLD)
            if pos_cid is not None:
                cname = str(pos_cname or "").lower()
                # Sweep G: adı negatif olan eşleşme reddedilmiş slottur.
                if is_negative_exemplar_name(pos_cname):
                    return False, float(pos_sim), "negative"
                if "cooldown" in cname or pos_cid == 21 or "empty" in cname:
                    return False, float(pos_sim), "cooldown"
                if "ready" in cname or pos_cid in (20, 13) or "skill" in cname:
                    return True, float(pos_sim), "ready"

            return False, 0.50, "unknown"
        except Exception:
            return False, 0.0, "unknown"

