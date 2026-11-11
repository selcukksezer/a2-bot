# -*- coding: utf-8 -*-
"""
OmniGameLearner & Game Cognition Engine (core/omni_game_learner.py)
==================================================================
Comprehensive Autonomous Game Perception, Continuous Screenshot Capturing &
Active Learning Engine.

Learns the ENTIRE game environment continuously:
1. Walking, Roads & Navigable Paths (Yürümeyi, yolları)
2. Monsters & Monster Health Bars (Monsterları, canavar canlarını)
3. Caves, Obstacles, Walls & Cliffs (Caveleri, engelleri, derinlik haritası)
4. Skills, Cooldowns & Mana Status (Skilleri, skil sürelerini)
5. Comprehensive Dataset Generation & Real-Time Application Reflection:
   - Actively captures and annotates screenshots with deduplication & quality checks
   - Reflects live intelligence into the application (smart skill casting, obstacle steering, target switching)
   - Exposes live telemetry to Overlay HUD and console logs.
"""

from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from dataset_factory.game_frame_gate import is_valid_game_frame as _is_valid_game_frame
except Exception:  # pragma: no cover
    _is_valid_game_frame = None

logger = logging.getLogger("Gatherion.GameLearner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
LEARNING_DIR = DATASET_DIR / "game_learning"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"

# Dedup history bound: _seen_dhashes keeps at most this many frame hashes.
# Eviction is oldest-first (see _remember_dhash); the history is never
# wiped wholesale, so a burst of new frames cannot re-admit old duplicates.
_DEDUP_MAX_HASHES = 600

# Layout conf ≥ TRAIN_CONF_FLOOR so boxes survive purge; still weak + learn_auto.
try:
    from dataset_factory.label_trust import CONF_EMIN as _CONF_EMIN
    from dataset_factory.label_trust import LEARN_AUTO_CONF as _LEARN_AUTO_CONF
except Exception as _learn_conf_exc:  # pragma: no cover - import cycle / early boot
    # B-14: the fallback used to be silent — an import failure hid while
    # the learner kept writing 0.50-confidence boxes as if measured.
    logger.warning(
        "LEARN_AUTO_CONF import failed (%s); using fallback 0.50 — "
        "check dataset_factory/label_trust.py",
        _learn_conf_exc,
    )
    _LEARN_AUTO_CONF = 0.50
    _CONF_EMIN = 0.85


# B-01: category -> canonical class ids whose boxes may be saved under it.
# Disjoint partition (data.yaml 0..26): one detection belongs to exactly
# one learn category, so the same YOLO boxes can no longer be written to
# all five category folders at once. Classes outside every set (world
# materials 0..8/12, minimap symbols 9/11, kinah 25, player 26, ...) have
# no learn category here — their frames are not saved as a category
# sample (they reach training through the mix/general path instead).
_CATEGORY_CLASS_IDS: Dict[str, frozenset] = {
    "monsters": frozenset({10, 18}),  # monster_3d + target_hp_bar
    "skills": frozenset({13, 19, 20, 21, 22, 23, 24}),  # prompt/bar/slots/consumables
    "obstacles": frozenset({14}),  # obstacle
    "paths": frozenset({15}),  # path_ground
    "vitality": frozenset({16, 17}),  # player_hp_bar + player_mp_bar
}

# B-05: the 12-slot hotbar (1-8, Q, E, R, T). Tiebreak order used when
# learned weights are equal (fresh start: every weight is 1.0). Existing
# keys keep their historic order; T (right-click skill) sits just under
# R, 6/7/8 continue the numeric tail after 5.
_STD_SKILL_PRIO: Dict[str, float] = {
    "R": 8, "T": 7.5, "E": 7, "Q": 6,
    "2": 5, "3": 4, "4": 3, "5": 2,
    "6": 1.8, "7": 1.6, "8": 1.4, "1": 1,
}


def _press_ok(result: Any) -> bool:
    """Sürücü dönüşünü 'tuş gönderildi mi' kararına çevirir (P0-6, B-02
    ile aynı kural: InputController.press_key float döner, 0.0 =
    gönderilmedi; None / False / 0.0 = gönderilmedi; dönüşü
    raporlamayan sürücü 'gönderildi' sayılmaz — fail-closed)."""
    if result is None or result is False:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return float(result) > 0.0
    return bool(result)


def _filter_boxes_for_category(
    boxes: Optional[Sequence[Dict[str, Any]]],
    category: str,
) -> List[Dict[str, Any]]:
    """Keep only boxes whose class belongs to `category` (B-01/B-02).

    Boxes without a parseable class_id are dropped, never defaulted to
    monster (10) — see B-09. Returns a new list; callers keep the
    original untouched.
    """
    allowed = _CATEGORY_CLASS_IDS.get(str(category or "").strip().lower())
    if allowed is None:
        return []
    out: List[Dict[str, Any]] = []
    for b in boxes or []:
        if not isinstance(b, dict) or "box" not in b:
            continue
        try:
            cid = int(b.get("class_id"))
        except (TypeError, ValueError):
            continue
        if cid in allowed:
            out.append(b)
    return out


def layout_boxes_for_learn_category(
    category: str,
    *,
    has_monster_target: bool = False,
    pot_hp: bool = False,
    pot_mp: bool = False,
    steer_deg: float = 0.0,
) -> List[Dict[str, Any]]:
    """Default HUD/world layout boxes for learn_* captures (Adım 2 culls)."""
    cat = str(category or "").strip().lower()
    if cat in ("monster", "monsters"):
        boxes: List[Dict[str, Any]] = [
            {"class_id": 20, "box": (0.50, 0.08, 0.35, 0.08)},
        ]
        if has_monster_target:
            boxes.append({"class_id": 10, "box": (0.50, 0.46, 0.18, 0.28)})
        return boxes
    if cat in ("skill", "skills"):
        boxes = [{"class_id": 13, "box": (0.50, 0.955, 0.34, 0.055)}]
        for index in range(8):
            slot_x = 0.36 + index * 0.037
            boxes.append({"class_id": 21, "box": (slot_x, 0.955, 0.032, 0.045)})
        return boxes
    if cat in ("path", "paths"):
        return [{"class_id": 15, "box": (0.50, 0.76, 0.62, 0.30)}]
    if cat in ("obstacle", "obstacles"):
        # Forward corridor mass; bias opposite steer when depth suggests a side hit.
        cx = 0.50
        try:
            steer = float(steer_deg)
        except (TypeError, ValueError):
            steer = 0.0
        if abs(steer) >= 10.0:
            cx = 0.50 - (0.12 if steer > 0.0 else -0.12)
        cx = max(0.22, min(0.78, cx))
        return [{"class_id": 14, "box": (cx, 0.55, 0.30, 0.38)}]
    if cat in ("vitality", "vital", "hp", "hud"):
        boxes = [
            {"class_id": 16, "box": (0.416, 0.895, 0.13, 0.035)},
            {"class_id": 17, "box": (0.584, 0.895, 0.13, 0.035)},
        ]
        if pot_hp:
            boxes.append({"class_id": 18, "box": (0.325, 0.90, 0.04, 0.05)})
        if pot_mp:
            boxes.append({"class_id": 19, "box": (0.675, 0.90, 0.04, 0.05)})
        return boxes
    return []


class OmniGameLearner:
    """
    Kapsamlı Oyun Algısı ve Sürekli Yapay Zeka Öğrenim Motoru.
    Oyunun tamamını (yollar, canavarlar, can barları, engeller, skiller, bekleme süreleri)
    sürekli olarak inceler, ekran görüntülerini kaydeder ve karar mekanizmasına yansıtır.
    """

    def __init__(
        self,
        bot_worker: Optional[Any] = None,
        save_screenshots: bool = True,
        min_sample_interval_sec: float = 10.0,
    ) -> None:
        self._lock = threading.RLock()
        self.bot_worker = bot_worker
        self.save_screenshots = bool(save_screenshots)
        # Disk JPEG spam was dropping combat loop to ~0.3 FPS — keep cognition, throttle writes.
        # B-14: a below-minimum interval used to be silently raised to 4.0 —
        # the caller never learned its configuration was ignored.
        _requested_iv = float(min_sample_interval_sec)
        if _requested_iv < 4.0:
            logger.warning(
                "min_sample_interval_sec %.2f < 4.0 minimum; clamped to 4.0",
                _requested_iv,
            )
        self.min_sample_interval_sec = max(4.0, _requested_iv)

        # Otomatik İksir ve Canlılık Yapılandırması
        self.hp_pot_threshold: float = 0.50
        self.mp_pot_threshold: float = 0.35
        self.pot_cooldown_sec: float = 8.0
        self._last_hp_pot_ts: float = 0.0
        self._last_mp_pot_ts: float = 0.0

        # GameFrameGate: bu learner kendi kapı örneğini kullanır; modül
        # düzeyindeki tekil kapının dHash önbelleği çağıranlar arasında
        # paylaşılmaz (F-0957). Tembel kurulur.
        self._frame_gate = None

        # _detect_real_boxes için önbelleğe alınmış etiketleyici (F-0342):
        # her karede yeni YOLOAutoLabeler() kurulmaz.
        self._yolo_labeler = None
        self._yolo_labeler_failed_ts: float = 0.0

        # Alt klasörleri oluştur
        self._init_storage_directories()

        # İstatistikler ve sayaçlar
        self.total_learned_frames: int = 0
        self.monsters_captured: int = 0
        self.skills_captured: int = 0
        self.paths_captured: int = 0
        self.obstacles_captured: int = 0
        self.vitality_captured: int = 0

        # Son örnekleme zaman damgaları
        self._last_monster_sample_ts: float = 0.0
        self._last_skill_sample_ts: float = 0.0
        self._last_path_sample_ts: float = 0.0
        self._last_obstacle_sample_ts: float = 0.0
        self._last_vitality_sample_ts: float = 0.0
        self._last_general_sample_ts: float = 0.0
        self._last_select_reader_warn_ts: float = 0.0  # dilim-30 B-08 throttle
        self._last_detect_failed_warn_ts: float = 0.0  # B-04 throttle
        self._last_skill_signature: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None  # B-10

        # Canlı durum takibi
        self.current_monster_targeted: bool = False
        self.current_monster_hp_ratio: float = 0.0
        self.prev_monster_hp_ratio: float = 0.0
        self.last_monster_death_ts: float = 0.0
        self._learner_no_target_frames: int = 0

        self.player_hp_ratio: float = 1.0
        self.player_mp_ratio: float = 1.0
        # S-3101: son bilinen degerlerdir; okunabilirlik ayri izlenir —
        # okunamayan bar asla %100 diye raporlanmaz.
        self.player_hp_readable: bool = False
        self.player_mp_readable: bool = False

        self.ready_skills: List[str] = []
        self.cooldown_skills: Dict[str, Any] = {}
        # Cooldown'daki slotun ölçülen parlaklığı (süre DEĞİL; F-0482).
        # cooldown_skills değerleri yalnızca "cooldown'da" bayrağıdır (True).
        self.cooldown_brightness: Dict[str, float] = {}

        self.obstacle_detected: bool = False
        self.obstacle_type: str = "none"
        self.recommended_steer_deg: float = 0.0

        # Event-driven combat learning & dynamic skill DPS metrics
        # B-05: the bar has 12 slots (1-8, Q, E, R, T). The old 8-key list
        # silently dropped every 6/7/8/T cast in record_skill_cast, so T
        # (right-click skill) could never learn a DPS weight.
        self._combat_learning_lock = threading.RLock()
        self.combat_skill_keys: Tuple[str, ...] = (
            "1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T",
        )
        self._skill_metrics: Dict[str, Dict[str, Any]] = {}
        for k in self.combat_skill_keys:
            self._skill_metrics[k] = {
                "casts": 0,
                "hits": 0,
                "total_hp_drop": 0.0,
                "avg_hp_drop": 0.0,
                "max_hp_drop": 0.0,
                "last_cast_ts": 0.0,
                "hp_at_cast": 0.0,
                "weight": 1.000,
            }

        self._recent_cast_queue: deque = deque(maxlen=32)
        self._in_encounter: bool = False
        self._encounter_start_ts: float = 0.0
        self._encounter_start_hp: float = 1.0
        self._encounter_last_hp: float = 1.0
        self._encounter_skills: List[str] = []
        self._encounter_damage_by_skill: Dict[str, float] = {k: 0.0 for k in self.combat_skill_keys}

        self.total_encounters: int = 0
        self.fast_encounters: int = 0
        self.slow_encounters: int = 0
        self.avg_encounter_duration: float = 0.0
        self.last_encounter_duration: float = 0.0
        self.last_encounter_dps: float = 0.0
        self.last_encounter_speed_rating: str = "normal"

        # Load previously saved learned weights
        self.load_learned_weights()

        # Perceptual hash geçmişi (mükerrer kare kaydetmeme koruması).
        # Sınırlı FIFO: _remember_dhash en eski hash'i atar; 600'ü aşınca
        # tüm geçmişi silen clear() davranışı kaldırıldı (D22).
        self._seen_dhashes: set = set()
        self._seen_dhash_order: deque = deque()

        # Kare başına YOLO tespit önbelleği (D22): _evaluate aynı kareyi
        # 5 kategoride (monsters/skills/obstacles/paths/vitality) inceler
        # ama _detect_real_boxes kategori argümanını yok sayıp her seferinde
        # aynı auto_label_frame çıkarımını tekrarlardı. Tek girişli
        # önbellek, aynı dhash anahtarıyla gelen tekrarları 1 çıkarıma indirir.
        self._detect_cache_key: Optional[int] = None
        self._detect_cache_boxes: Optional[List[Dict[str, Any]]] = []

        # Alt sistem referansları (lazy resolver)
        self._skill_bar_reader = None
        self._depth_avoider = None

        logger.info(
            "OmniGameLearner (Kapsamlı Oyun Öğrenim Motoru) başlatıldı. Kayıt Dizini: %s",
            LEARNING_DIR,
        )

    def _init_storage_directories(self) -> None:
        """Öğrenilen kategorilere ait veri dizinlerini hazırlar."""
        try:
            for sub in ("monsters", "skills", "paths", "obstacles", "vitality", "general"):
                (LEARNING_DIR / sub).mkdir(parents=True, exist_ok=True)
            TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            # B-14: directory failure used to be debug-only — every later
            # save then failed silently, per frame, for the whole session.
            logger.warning("Öğrenim dizinleri oluşturulamadı: %s", exc)

    @property
    def skill_bar_reader(self) -> Optional[Any]:
        if self._skill_bar_reader is None and self.bot_worker is not None:
            self._skill_bar_reader = getattr(self.bot_worker, "skill_bar_monitor", None)
        return self._skill_bar_reader

    @property
    def depth_avoider(self) -> Optional[Any]:
        if self._depth_avoider is None and self.bot_worker is not None:
            self._depth_avoider = getattr(self.bot_worker, "depth_avoider", None)
        return self._depth_avoider

    # =========================================================================
    # 0. Event-Driven Combat Learning & Skill Priority Tuning Methods
    # =========================================================================

    @staticmethod
    def _normalize_skill_key(key: Any) -> str:
        """Normalizes skill key to standard hotbar identifier ('1'..'8', 'Q', 'E', 'R', 'T')."""
        s = str(key or "").strip()
        if not s:
            return ""
        upper = s.upper()
        if upper in ("Q", "E", "R", "T"):
            return upper
        if s in ("1", "2", "3", "4", "5", "6", "7", "8"):
            return s
        if upper in ("1", "2", "3", "4", "5", "6", "7", "8"):
            return upper
        return upper

    def _get_frame_gate(self):
        """Learner'a özel GameFrameGate örneği (tembel; F-0957)."""
        if self._frame_gate is None:
            from dataset_factory.game_frame_gate import GameFrameGate

            self._frame_gate = GameFrameGate()
        return self._frame_gate

    def check_game_frame(
        self,
        frame: Optional[np.ndarray],
        require_foreground: bool = False,
    ) -> Tuple[bool, str]:
        """
        GameFrameGate protection: Verifies frame contains genuine game canvas/HUD
        evidence and is not desktop, black screen, flat solid color, or Gatherion UI.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, "invalid_frame_data"

        if frame.ndim != 3 or frame.shape[0] < 60 or frame.shape[1] < 60:
            return False, "frame_too_small"

        if _is_valid_game_frame is not None:
            try:
                gate = self._get_frame_gate()
                ok, reason = gate.is_valid_game_frame(
                    frame, require_foreground=require_foreground
                )
                return bool(ok), str(reason)
            except Exception as exc:
                logger.debug("GameFrameGate evaluation error: %s", exc)

        # GameFrameGate kullanılamıyorsa fail-closed: kapı yokken kareyi
        # oyun karesi saymak masaüstü/tarayıcı karelerini veri setine
        # yazdırır (F-0344). Eski "accepted_fallback" kabulü kaldırıldı.
        return False, "gate_unavailable"

    def record_skill_cast(
        self,
        skill_key: str,
        hp_ratio: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records an event-driven skill cast by key ('1'..'8', 'Q', 'E', 'R', 'T').
        Queues cast for subsequent target HP drop measurement.
        """
        norm_key = self._normalize_skill_key(skill_key)
        if not norm_key or norm_key not in self._skill_metrics:
            return
        now = time.time() if timestamp is None else float(timestamp)
        with self._combat_learning_lock:
            if hp_ratio is not None and float(hp_ratio) > 0.0:
                current_hp = float(hp_ratio)
                self.current_monster_hp_ratio = current_hp
            else:
                current_hp = float(self.current_monster_hp_ratio)
            if current_hp <= 0.0 and self.prev_monster_hp_ratio > 0.0:
                current_hp = float(self.prev_monster_hp_ratio)

            metrics = self._skill_metrics[norm_key]
            metrics["casts"] += 1
            metrics["last_cast_ts"] = now
            metrics["hp_at_cast"] = current_hp

            self._recent_cast_queue.append((norm_key, now, current_hp))

            if self._in_encounter:
                self._encounter_skills.append(norm_key)

            logger.debug(
                "CombatLearning: Skill cast recorded [%s], target_HP=%.2f, total_casts=%d",
                norm_key,
                current_hp,
                metrics["casts"],
            )

    def record_skill_damage(
        self,
        skill_key: str,
        hp_drop: float,
        latency_sec: float = 0.0,
    ) -> None:
        """Registers verified target HP drop attributed to a specific skill key."""
        norm_key = self._normalize_skill_key(skill_key)
        if not norm_key or norm_key not in self._skill_metrics:
            return
        drop = float(max(0.0, hp_drop))
        if drop < 0.003:
            return

        with self._combat_learning_lock:
            metrics = self._skill_metrics[norm_key]
            metrics["hits"] += 1
            metrics["total_hp_drop"] += drop
            metrics["max_hp_drop"] = max(float(metrics["max_hp_drop"]), drop)
            metrics["avg_hp_drop"] = metrics["total_hp_drop"] / max(1, metrics["hits"])

            if self._in_encounter:
                self._encounter_damage_by_skill[norm_key] = (
                    self._encounter_damage_by_skill.get(norm_key, 0.0) + drop
                )

            logger.debug(
                "CombatLearning: Skill [%s] registered HP drop=%.3f (avg=%.3f, hits=%d/%d)",
                norm_key,
                drop,
                metrics["avg_hp_drop"],
                metrics["hits"],
                metrics["casts"],
            )

    def on_target_hp_update(
        self,
        hp_ratio: float,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Updates current target HP and attributes observed HP drop to recent skill cast.
        Returns the observed HP drop (>= 0.0).
        """
        now = time.time() if timestamp is None else float(timestamp)
        with self._combat_learning_lock:
            new_hp = float(np.clip(hp_ratio, 0.0, 1.0))
            prev_hp = float(self.current_monster_hp_ratio)
            self.prev_monster_hp_ratio = prev_hp
            self.current_monster_hp_ratio = new_hp

            if new_hp > 0.0 and not self._in_encounter:
                self.on_encounter_start(initial_hp=new_hp, timestamp=now)

            hp_drop = prev_hp - new_hp
            if hp_drop >= 0.005:
                # Find the skill cast responsible in recent queue
                attributed = None
                for norm_key, cast_ts, hp_at_cast in reversed(self._recent_cast_queue):
                    dt = now - cast_ts
                    if 0.02 <= dt <= 1.60:
                        attributed = norm_key
                        break

                if attributed is not None:
                    self.record_skill_damage(attributed, hp_drop)
                elif self._recent_cast_queue:
                    last_k, last_t, _ = self._recent_cast_queue[-1]
                    if (now - last_t) <= 2.0:
                        self.record_skill_damage(last_k, hp_drop)

                return hp_drop
            return 0.0

    def on_encounter_start(
        self,
        initial_hp: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Starts a new combat encounter tracking session."""
        now = time.time() if timestamp is None else float(timestamp)
        with self._combat_learning_lock:
            self._in_encounter = True
            self._encounter_start_ts = now
            self._encounter_start_hp = float(np.clip(initial_hp, 0.05, 1.0))
            self._encounter_last_hp = self._encounter_start_hp
            self.current_monster_hp_ratio = self._encounter_start_hp
            self._encounter_skills.clear()
            for k in self.combat_skill_keys:
                self._encounter_damage_by_skill[k] = 0.0
            logger.info("⚔️ [CombatLearning] Encounter started (Start HP: %d%%)", int(self._encounter_start_hp * 100))

    def reset_combat_learning_metrics(self) -> None:
        """Resets in-memory combat learning metrics to clean baseline state."""
        with self._combat_learning_lock:
            for k in self.combat_skill_keys:
                self._skill_metrics[k] = {
                    "casts": 0,
                    "hits": 0,
                    "total_hp_drop": 0.0,
                    "avg_hp_drop": 0.0,
                    "max_hp_drop": 0.0,
                    "last_cast_ts": 0.0,
                    "hp_at_cast": 0.0,
                    "weight": 1.000,
                }
            self._recent_cast_queue.clear()
            self._in_encounter = False
            self.total_encounters = 0
            self.fast_encounters = 0
            self.slow_encounters = 0
            self.avg_encounter_duration = 0.0
            self.last_encounter_duration = 0.0
            self.last_encounter_dps = 0.0
            self.last_encounter_speed_rating = "normal"

    def on_encounter_end(
        self,
        killed: bool = True,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Concludes combat encounter, analyzes speed/DPS, and tunes skill weights.
        """
        now = time.time() if timestamp is None else float(timestamp)
        with self._combat_learning_lock:
            if not self._in_encounter:
                # Karşılaşma yokken hayali karşılaşma sayılmaz ve DPS
                # uydurulmaz (F-0162): karşılaşma başlamadan bitirilemez.
                logger.debug(
                    "CombatLearning: on_encounter_end called without an active encounter; ignored"
                )
                return {
                    "duration": 0.0,
                    "killed": bool(killed),
                    "speed_rating": "no_encounter",
                    "dps": 0.0,
                    "weights": self.get_skill_dps_weights(),
                    "damage_by_skill": dict(self._encounter_damage_by_skill),
                }
            duration = max(0.5, now - self._encounter_start_ts)
            self._in_encounter = False

            self.last_encounter_duration = duration
            self.total_encounters += 1
            if self.avg_encounter_duration <= 0.0:
                self.avg_encounter_duration = duration
            else:
                self.avg_encounter_duration = 0.80 * self.avg_encounter_duration + 0.20 * duration

            total_dmg = sum(self._encounter_damage_by_skill.values())
            # S-3104: olculmemis hasar yokken oldurme durumunda DPS
            # baslangic HP'sinden turetilmez — olcum yoksa DPS 0.0'dir.
            encounter_dps = total_dmg / duration if total_dmg > 0.0 else 0.0
            self.last_encounter_dps = encounter_dps

            # Encounter speed rating (reference kill time: ~7.5s)
            if duration <= 6.5 and killed:
                self.fast_encounters += 1
                self.last_encounter_speed_rating = "fast"
            elif duration >= 15.0 or not killed:
                self.slow_encounters += 1
                self.last_encounter_speed_rating = "slow"
            else:
                self.last_encounter_speed_rating = "normal"

            new_weights = self.tune_skill_priority_weights(
                encounter_duration=duration,
                monster_died=killed,
            )

            logger.info(
                "⚔️ [CombatLearning] Encounter finished: duration=%.2fs (%s), DPS=%.3f, kills=%d (fast=%d, slow=%d)",
                duration,
                self.last_encounter_speed_rating,
                encounter_dps,
                self.total_encounters,
                self.fast_encounters,
                self.slow_encounters,
            )

            self._propagate_learned_weights()

            return {
                "duration": duration,
                "killed": killed,
                "speed_rating": self.last_encounter_speed_rating,
                "dps": round(encounter_dps, 4),
                "weights": dict(new_weights),
                "damage_by_skill": dict(self._encounter_damage_by_skill),
            }

    def tune_skill_priority_weights(
        self,
        encounter_duration: Optional[float] = None,
        monster_died: bool = True,
    ) -> Dict[str, float]:
        """
        Dynamically tunes skill DPS priority weights based on:
        1. Observed average HP drop per skill cast (skill impact).
        2. Encounter speed (fast encounters reward the rotation; slow encounters penalize weak skills).
        3. Reliability (hit rate vs casts).
        """
        with self._combat_learning_lock:
            dur = float(
                encounter_duration
                if encounter_duration is not None
                else (self.last_encounter_duration or 7.5)
            )
            ref_duration = 7.5

            active_skills = [
                m for m in self._skill_metrics.values()
                if m["hits"] > 0 and m["avg_hp_drop"] > 0.0
            ]
            # Tek aktif skill varken ortalama onun kendi değeridir ve
            # damage_ratio hep 1.0 çıkar — ayırt edemez (F-0484). En az
            # 2 aktif skill yoksa sabit referans (0.10) kullanılır.
            if len(active_skills) >= 2:
                mean_all_dmg = float(np.mean([m["avg_hp_drop"] for m in active_skills]))
            else:
                mean_all_dmg = 0.10

            if monster_died and dur <= 6.5:
                speed_mult = min(1.80, ref_duration / max(1.5, dur))
            elif dur >= 14.0 or not monster_died:
                # P3-8 (S-130): dur==0 + monster_died=False ZeroDivisionError
                # verirdi (ref_duration / dur).
                speed_mult = max(0.60, ref_duration / max(dur, 0.5))
            else:
                speed_mult = 1.00

            updated_weights: Dict[str, float] = {}
            for key, metrics in self._skill_metrics.items():
                current_w = float(metrics.get("weight", 1.0))
                hits = int(metrics.get("hits", 0))
                casts = int(metrics.get("casts", 0))
                avg_drop = float(metrics.get("avg_hp_drop", 0.0))

                if casts == 0:
                    target_w = current_w
                else:
                    if mean_all_dmg > 0.001 and hits > 0:
                        damage_ratio = avg_drop / mean_all_dmg
                    else:
                        damage_ratio = 0.80

                    hit_rate = float(hits) / max(1.0, float(casts))
                    reliability = 0.50 + 0.50 * hit_rate

                    enc_dmg = float(self._encounter_damage_by_skill.get(key, 0.0))
                    enc_boost = 1.0 + (0.35 * enc_dmg) if enc_dmg > 0.0 else 1.0

                    if speed_mult >= 1.0:
                        target_w = (1.00 * damage_ratio * reliability * enc_boost) * speed_mult
                    else:
                        if damage_ratio < 0.90:
                            target_w = current_w * speed_mult * 0.85
                        else:
                            target_w = (1.00 * damage_ratio * reliability)

                alpha = 0.35
                new_w = (1.0 - alpha) * current_w + alpha * target_w
                new_w = float(np.clip(new_w, 0.20, 3.00))
                new_w = round(new_w, 3)

                metrics["weight"] = new_w
                updated_weights[key] = new_w

            self.save_learned_weights()
            return updated_weights

    def get_skills_by_priority(self) -> List[str]:
        """Returns all 12 combat skill keys ordered by dynamic DPS weight descending."""
        with self._combat_learning_lock:
            std_prio = dict(_STD_SKILL_PRIO)
            skills = list(self.combat_skill_keys)
            skills.sort(
                key=lambda k: (
                    self.get_skill_weight(k),
                    std_prio.get(k.upper(), 0),
                ),
                reverse=True,
            )
            return skills

    def get_skill_weight(self, skill_key: str) -> float:
        """Returns the learned priority weight for a skill key."""
        norm_key = self._normalize_skill_key(skill_key)
        with self._combat_learning_lock:
            return float(self._skill_metrics.get(norm_key, {}).get("weight", 1.0))

    def get_skill_dps_weights(self) -> Dict[str, float]:
        """Returns copy of all learned skill DPS weights."""
        with self._combat_learning_lock:
            return {k: float(m.get("weight", 1.0)) for k, m in self._skill_metrics.items()}

    def get_skill_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns detailed combat diagnostics per skill key."""
        with self._combat_learning_lock:
            return {
                k: {
                    "casts": int(m["casts"]),
                    "hits": int(m["hits"]),
                    "total_hp_drop": round(float(m["total_hp_drop"]), 4),
                    "avg_hp_drop": round(float(m["avg_hp_drop"]), 4),
                    "max_hp_drop": round(float(m["max_hp_drop"]), 4),
                    "weight": round(float(m["weight"]), 3),
                    "last_cast_ts": float(m["last_cast_ts"]),
                }
                for k, m in self._skill_metrics.items()
            }

    def get_encounter_stats(self) -> Dict[str, Any]:
        """Returns encounter speed and kill diagnostics."""
        with self._combat_learning_lock:
            return {
                "total_encounters": self.total_encounters,
                "fast_encounters": self.fast_encounters,
                "slow_encounters": self.slow_encounters,
                "avg_encounter_duration": round(self.avg_encounter_duration, 2),
                "last_encounter_duration": round(self.last_encounter_duration, 2),
                "last_encounter_dps": round(self.last_encounter_dps, 4),
                "last_encounter_speed_rating": self.last_encounter_speed_rating,
                "in_encounter": self._in_encounter,
            }

    def _propagate_learned_weights(self) -> None:
        """Propagates dynamically tuned weights to BotWorker and SkillBarReader.

        DEFAULT_PRIORITY_WEIGHTS sınıf düzeyinde paylaşımlı bir sözlüktür;
        yerinde değiştirmek (F-0485) tüm örneklere sızar. Bu yüzden her
        alıcıya kendi örnek niteliği olarak yeni bir sözlük atanır.
        """
        weights = self.get_skill_dps_weights()
        propagated = dict(weights)
        propagated.update({k.lower(): w for k, w in weights.items()})
        if self.skill_bar_reader is not None:
            try:
                self.skill_bar_reader.DEFAULT_PRIORITY_WEIGHTS = dict(propagated)
            except Exception as exc:
                logger.debug("Could not propagate weights to skill_bar_reader: %s", exc)

        if self.bot_worker is not None:
            sbm = getattr(self.bot_worker, "skill_bar_monitor", None)
            if sbm is not None:
                try:
                    sbm.DEFAULT_PRIORITY_WEIGHTS = dict(propagated)
                except Exception as exc:
                    # Dilim-30 B-08: the second target used to swallow its
                    # failure with a bare `pass` — propagation stopped
                    # halfway and reader/monitor weights silently diverged.
                    # Propagation is best-effort, but the split must be visible.
                    logger.warning(
                        "Could not propagate weights to skill_bar_monitor: %s",
                        exc,
                    )

    def save_learned_weights(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Persists learned skill weights and metrics to config JSON."""
        try:
            target_path = Path(path) if path else PROJECT_ROOT / "config" / "learned_skill_weights.json"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": time.time(),
                "weights": self.get_skill_dps_weights(),
                "skill_stats": self.get_skill_stats(),
                "encounter_stats": self.get_encounter_stats(),
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as exc:
            logger.debug("Failed to save learned skill weights: %s", exc)
            return False

    def load_learned_weights(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Loads learned skill weights from config JSON."""
        try:
            target_path = Path(path) if path else PROJECT_ROOT / "config" / "learned_skill_weights.json"
            if not target_path.is_file():
                return False
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._combat_learning_lock:
                weights = data.get("weights", {})
                for k, w in weights.items():
                    norm = self._normalize_skill_key(k)
                    if norm in self._skill_metrics:
                        self._skill_metrics[norm]["weight"] = float(w)
                skill_stats = data.get("skill_stats", {})
                for k, s in skill_stats.items():
                    norm = self._normalize_skill_key(k)
                    if norm in self._skill_metrics and isinstance(s, dict):
                        self._skill_metrics[norm]["casts"] = int(s.get("casts", 0))
                        self._skill_metrics[norm]["hits"] = int(s.get("hits", 0))
                        self._skill_metrics[norm]["total_hp_drop"] = float(s.get("total_hp_drop", 0.0))
                        self._skill_metrics[norm]["avg_hp_drop"] = float(s.get("avg_hp_drop", 0.0))
                        self._skill_metrics[norm]["max_hp_drop"] = float(s.get("max_hp_drop", 0.0))
                enc_stats = data.get("encounter_stats", {})
                self.total_encounters = int(enc_stats.get("total_encounters", 0))
                self.fast_encounters = int(enc_stats.get("fast_encounters", 0))
                self.slow_encounters = int(enc_stats.get("slow_encounters", 0))
                self.avg_encounter_duration = float(enc_stats.get("avg_encounter_duration", 0.0))
            return True
        except Exception as exc:
            logger.debug("Failed to load learned skill weights: %s", exc)
            return False

    # =========================================================================
    # 1. Ana Gözlem ve Öğrenim Döngüsü (`observe_frame`)
    # =========================================================================

    def observe_frame(
        self,
        frame: Optional[np.ndarray],
        current_state: str = "WALKING",
        current_mode: str = "route",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Her görüntü karesinde çağrılır:
        1. Canavar ve hedef can barı analizi
        2. Skill bar, hazır yetenekler ve bekleme süreleri analizi
        3. Yol, zemin ve hareket analizi
        4. Engel, mağara ve derinlik analizi
        5. Gerekli durumlarda ekran görüntüsü alıp veri setine etiketli kaydetme.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self.get_live_cognition_snapshot()

        # Gate check: Reject non-game screens (desktop, solid color, black, browser, etc.)
        is_valid_frame, gate_reason = self.check_game_frame(frame, require_foreground=False)
        if not is_valid_frame:
            logger.debug("OmniGameLearner: frame rejected by GameFrameGate (%s)", gate_reason)
            return self.get_live_cognition_snapshot()

        now = time.time()
        mix_sources = kwargs.get("mix_sources")

        with self._lock:
            # 1. Canavar & Can Barı Analizi
            # F-0038: _analyze_monster_and_health yeni HP'yi
            # current_monster_hp_ratio'ya yazar; on_target_hp_update aynı
            # değeri "önceki" sanıp düşüşü hep 0.0 ölçer. Analiz öncesi
            # değer saklanır ve on_target_hp_update öncesinde geri
            # yüklenir ki gerçek düşüş ölçülsün.
            prev_monster_hp = self.current_monster_hp_ratio
            monster_info = self._analyze_monster_and_health(frame)

            # Event-driven combat encounter tracking & target HP drop processing
            if monster_info.get("has_target"):
                curr_hp = float(monster_info.get("hp_ratio", 0.0))
                self.current_monster_hp_ratio = prev_monster_hp
                self.on_target_hp_update(curr_hp, timestamp=now)
            elif monster_info.get("monster_died"):
                self.on_encounter_end(killed=True, timestamp=now)
            elif self._in_encounter and not monster_info.get("has_target"):
                self.on_encounter_end(killed=False, timestamp=now)

            # 2. Skill Bar & Süre Analizi
            skill_info = self._analyze_skills_and_cooldowns(frame, now)

            # 3. Oyuncu Can & Mana (Vitality) ve Otomatik İksir Analizi
            vitality_info = self._analyze_player_vitality(frame, now)

            # 4. Engel, Mağara & Derinlik Analizi
            obstacle_info = self._analyze_obstacles_and_depth(frame)

            # 5. Yol & Navigasyon Analizi
            path_info = self._analyze_path_and_walking(frame, current_state, current_mode, now)

            # Mix, toplama kutularını yazar. Yol, engel, skill ve can barı
            # o karışımdan bağımsız olarak ayrıca etiketlenir.
            if mix_sources is not None:
                try:
                    self.observe_mix(frame, mix_sources, now=now)
                except Exception as exc:
                    # F-0345: mix füzyon/kayıt hatası observe_frame'i
                    # yarıda kesmemeli.
                    logger.debug("observe_mix skipped after error: %s", exc)
            self._evaluate_and_save_learning_samples(
                frame,
                now,
                current_state=current_state,
                current_mode=current_mode,
                monster_info=monster_info,
                skill_info=skill_info,
                vitality_info=vitality_info,
                obstacle_info=obstacle_info,
                path_info=path_info,
                # D22 + A13: kapı bu kare için observe_frame başında zaten
                # değerlendirildi; sonucu aktar ki _evaluate aynı kareyi
                # ikinci kez kapıdan geçirmesin — kapı stateful'dır, ikinci
                # denetim aynı kareyi dHash yenilik denetiminde
                # "duplicate_frame" ile reddeder ve hiçbir öğrenim örneği
                # yazılmaz. Doğrudan çağrılarda gate_ok=None kalır (kapı
                # burada çalışır); gate_ok=False kapıya danışmadan düşürür.
                gate_ok=is_valid_frame,
            )

        return self.get_live_cognition_snapshot()

    def observe_mix(
        self,
        frame: Optional[np.ndarray],
        sources: Optional[Dict[str, Any]],
        now: Optional[float] = None,
        stem: Optional[str] = None,
        force: bool = False,
        dataset_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Fuse every source in this observation into one sample.

        Low-confidence boxes stay out of labels/train. The train image is
        written only when a trusted F/E prompt or a hand label is in the mix.
        A sample counts as saved only when at least one trusted box reached
        train (F-0345) and the train image was written (S-0101); if the
        train image cannot be written, the committed label is rolled back
        so no orphan label without an image stays in train (S-3103).
        Fuse/commit errors are reported, never raised.
        """
        try:
            from dataset_factory.multi_mix import commit_mix_sample, fuse_sources

            fused = fuse_sources(sources)
        except Exception as exc:
            logger.debug("observe_mix fuse_sources failed: %s", exc)
            return {"saved": False, "reason": "fuse_error", "error": str(exc)}
        now = time.time() if now is None else float(now)
        if not force and (now - self._last_general_sample_ts) < self.min_sample_interval_sec:
            fused["saved"] = False
            fused["reason"] = "throttled"
            return fused
        if not fused.get("annotations"):
            fused["saved"] = False
            fused["reason"] = "empty"
            return fused

        # B-11: throttle commit'ten ÖNCE tüketilmiyor — yalnızca kayıt
        # gerçekten yapıldıysa ilerler (B-06'nın mix karşılığı).
        sample_stem = stem or f"mix_{time.time_ns()}_{random.randint(100, 999)}"
        mix_jpg: Optional[Path] = None
        if (
            self.save_screenshots
            and frame is not None
            and isinstance(frame, np.ndarray)
            and frame.size
        ):
            try:
                LEARNING_DIR.joinpath("mix").mkdir(parents=True, exist_ok=True)
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    mix_jpg = LEARNING_DIR / "mix" / f"{sample_stem}.jpg"
                    mix_jpg.write_bytes(enc.tobytes())
            except Exception as exc:
                logger.debug("mix frame save skipped: %s", exc)

        root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
        try:
            result = commit_mix_sample(
                dataset_dir=root,
                stem=sample_stem,
                fused=fused,
                split="train",
            )
        except Exception as exc:
            logger.debug("observe_mix commit_mix_sample failed: %s", exc)
            # B-11: commit patladıysa mix jpg yetim kalmaz.
            if mix_jpg is not None:
                try:
                    mix_jpg.unlink(missing_ok=True)
                except OSError:
                    pass
            fused["saved"] = False
            fused["reason"] = "commit_error"
            fused["error"] = str(exc)
            return fused
        trusted_n = int(result.get("train_box_count", 0))
        fused["train_box_count"] = trusted_n
        fused["weak_box_count"] = int(result.get("weak_box_count", 0))
        fused["label_path"] = result.get("label_path") or ""
        fused["stem"] = sample_stem
        if trusted_n <= 0:
            # Güvenilir kutu yoksa train'e hiçbir şey yazılmadı; bunu
            # "kaydedildi" saymak ve sayacı artırmak yanlıştır (F-0345).
            # B-11: mix jpg de silinir — "kaydedilmedi" sayılan karenin
            # dosyası diskte kalmaz.
            if mix_jpg is not None:
                try:
                    mix_jpg.unlink(missing_ok=True)
                except OSError:
                    pass
            fused["saved"] = False
            fused["reason"] = "no_trusted_boxes"
            return fused
        train_image_ok = True
        train_image_error = ""
        if (
            frame is not None
            and isinstance(frame, np.ndarray)
            and frame.size
        ):
            try:
                train_images = root / "images" / "train"
                train_images.mkdir(parents=True, exist_ok=True)
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    (train_images / f"{sample_stem}.jpg").write_bytes(enc.tobytes())
                else:
                    # Kodlama başarısız: görüntü diske yazılmadı (S-0101).
                    train_image_ok = False
                    train_image_error = "train image encode failed"
            except Exception as exc:
                logger.debug("mix train image skipped: %s", exc)
                train_image_ok = False
                train_image_error = str(exc)
        else:
            # S-3103: kare yoksa/boşsa train görüntüsü yazılmadı sayılır —
            # train_image_ok True kalıp saved=True + sayaç artmaz.
            train_image_ok = False
            train_image_error = "no valid train frame"
        if not train_image_ok:
            # Train görüntüsü yazılamadıysa örnek "kaydedildi" sayılmaz;
            # sayaç da artmaz (S-0101; F-0343 sınıfı kusur).
            # S-3103: az önce commit edilen etiket geri alınır — train'de
            # görüntüsü olmayan yetim etiket kalmaz.
            label_path = str(fused.get("label_path") or "")
            if label_path:
                try:
                    Path(label_path).unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("orphan mix label rollback skipped: %s", exc)
                fused["label_path"] = ""
            # B-11: mix jpg de yetim kalmaz.
            if mix_jpg is not None:
                try:
                    mix_jpg.unlink(missing_ok=True)
                except OSError:
                    pass
            fused["saved"] = False
            fused["reason"] = "train_image_write_failed"
            if train_image_error:
                fused["error"] = train_image_error
            return fused
        # B-11: throttle yalnızca burada, kayıt gerçekten tamamlanınca tüketilir.
        self._last_general_sample_ts = now
        self.total_learned_frames += 1
        fused["saved"] = True
        return fused

    def _actor_is_moving(self) -> bool:
        """True while a movement key is held or a path step is active."""
        worker = self.bot_worker
        if worker is None:
            return False
        walker = getattr(worker, "_walking_is_real", None)
        if callable(walker):
            try:
                return bool(walker())
            except Exception:
                return False
        driver = getattr(worker, "input_driver", None)
        active = getattr(driver, "movement_active", None) if driver is not None else None
        if callable(active):
            try:
                return bool(active())
            except Exception:
                return False
        return False

    def _combat_context_for_death(self) -> bool:
        """Bar loss is a death only after a real fight signal, not on an empty idle bar."""
        worker = self.bot_worker
        if worker is None:
            return False
        if getattr(worker, "_route_fight_had_signal", False) or getattr(worker, "_route_saw_target_bar", False):
            return True
        mode = str(getattr(worker, "mode", "") or "").lower()
        state = str(getattr(worker, "state", "") or "").upper()
        if mode == "combat" and self.current_monster_targeted:
            return True
        if state in ("FIGHTING", "COMBAT") and getattr(worker, "_route_combat_confirmed", False):
            return True
        return False

    def _obstacle_sample_allowed(self, obstacle_info: Dict[str, Any]) -> bool:
        """Do not store obstacle frames from a still character or a false combat tick."""
        if not obstacle_info.get("obstacle_detected"):
            return False
        if not self._actor_is_moving():
            return False
        kind = str(obstacle_info.get("obstacle_type") or "").strip().lower()
        return kind not in ("", "none")

    # =========================================================================
    # 2. Canavar ve Hedef Can Barı İncelemesi
    # =========================================================================

    def _analyze_monster_and_health(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Ekranın üst-ortasında kırmızı düşman can barını ve merkez savaş bölgesini analiz eder.
        Düşmanın tam can oranını (%0 - %100) hesaplar.
        """
        h, w = frame.shape[:2]

        # Üst-orta hedef barı bölgesi (Target Frame: y: 2%..15%, x: 30%..70%)
        ta_y1, ta_y2 = int(h * 0.02), int(h * 0.15)
        ta_x1, ta_x2 = int(w * 0.30), int(w * 0.70)
        target_roi = frame[ta_y1:ta_y2, ta_x1:ta_x2]

        hsv_top = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
        # Strict-ish red UI bar (grass orange false-positives were locking combat w/o Tab)
        red_mask = (
            ((hsv_top[:, :, 0] <= 12) | (hsv_top[:, :, 0] >= 168))
            & (hsv_top[:, :, 1] >= 110)
            & (hsv_top[:, :, 2] >= 95)
        )
        red_pixel_count = int(np.sum(red_mask))
        # Require a horizontal bar blob, not scattered grass pixels
        has_bar_shape = False
        if red_pixel_count > 80:
            col_sums = np.sum(red_mask, axis=0)
            active_cols = np.where(col_sums > 2)[0]
            row_sums = np.sum(red_mask, axis=1)
            active_rows = np.where(row_sums > 2)[0]
            if len(active_cols) > 0 and len(active_rows) > 0:
                width_px = float(active_cols[-1] - active_cols[0] + 1)
                height_px = float(active_rows[-1] - active_rows[0] + 1)
                has_bar_shape = width_px >= 40 and height_px <= 28 and (width_px / max(1.0, height_px)) >= 2.5

        # Soft-lock: yerdeki mavi hedef halkası (yüksek eşik)
        ry1, ry2 = int(h * 0.48), int(h * 0.78)
        rx1, rx2 = int(w * 0.32), int(w * 0.68)
        reticle_roi = frame[ry1:ry2, rx1:rx2]
        has_reticle = False
        if reticle_roi.size > 0:
            hsv_r = cv2.cvtColor(reticle_roi, cv2.COLOR_BGR2HSV)
            cyan = (
                (hsv_r[:, :, 0] >= 85)
                & (hsv_r[:, :, 0] <= 115)
                & (hsv_r[:, :, 1] >= 90)
                & (hsv_r[:, :, 2] >= 110)
            )
            has_reticle = int(np.sum(cyan)) >= max(280, int(0.004 * h * w))

        has_target = bool(has_bar_shape or has_reticle)
        hp_ratio = 0.0

        if has_bar_shape:
            col_sums = np.sum(red_mask, axis=0)
            active_cols = np.where(col_sums > 2)[0]
            if len(active_cols) > 0:
                width_px = float(active_cols[-1] - active_cols[0] + 1)
                # F-0039: fill_ratio (dolu sütun / aktif aralık) tanımı gereği
                # bitişik dolu barda ~1.0 çıkar; max(width_ratio,
                # fill_ratio) bu yüzden HP'yi hep ~%100 okur. Doluluk,
                # dolu genişliğin beklenen tam bar genişliğine oranıdır.
                max_expected_bar_w = float(target_roi.shape[1] * 0.55)
                width_ratio = min(1.0, width_px / max(40.0, max_expected_bar_w))
                hp_ratio = min(1.0, max(0.08, width_ratio))
            else:
                # S-3102: bar sekli var ama aktif sutun yoksa uydurma
                # taban (0.35) yazilmaz — son bilinen gecerli deger
                # korunur, yoksa olcum yok sayilir (0.0).
                prev = float(self.current_monster_hp_ratio or 0.0)
                hp_ratio = prev if prev >= 0.08 else 0.0
        elif has_reticle:
            # Soft-lock without top bar: keep last known HP if any;
            # S-3102: bilinmiyorsa uydurma 0.55 placeholder yazilmaz.
            prev = float(self.current_monster_hp_ratio or 0.0)
            hp_ratio = prev if prev >= 0.08 else 0.0
        else:
            hp_ratio = 0.0

        # Canavar öldü mü tespiti. Kısa titreme / hasar yazıları / soft-lock kaybı ölüm değildir.
        if has_target:
            self._learner_no_target_frames = 0
            effective_targeted = True
        else:
            self._learner_no_target_frames = getattr(self, "_learner_no_target_frames", 0) + 1
            # Savaş esnasında devamlılık toleransı: 15 kare (~1.5sn) boyunca hedef düşürülmez
            effective_targeted = bool(self.current_monster_targeted and (self._learner_no_target_frames <= 15))

        # Canavar öldü mü tespiti: sadece hedef kalıcı olarak kaybolduğunda (>= 15 kare) tetiklenir
        monster_died = (
            self.current_monster_targeted
            and not effective_targeted
            and self._learner_no_target_frames > 15
            and self._combat_context_for_death()
            and (time.time() - self.last_monster_death_ts > 2.5)
        )
        if monster_died:
            self.last_monster_death_ts = time.time()
            logger.info("OmniGameLearner: Canavar öldü algılandı (Hedef barı kalıcı olarak kayboldu).")

        self.prev_monster_hp_ratio = self.current_monster_hp_ratio
        self.current_monster_targeted = effective_targeted
        if not effective_targeted:
            self.current_monster_hp_ratio = 0.0
        elif not has_target:
            # Görsel titreme esnasında can barının son bilinen değerini koru (0'a düşürüp sahte ölüm yaratma)
            pass
        else:
            self.current_monster_hp_ratio = hp_ratio

        return {
            "has_target": effective_targeted,
            "hp_ratio": round(self.current_monster_hp_ratio, 3),
            "hp_percent": int(round(self.current_monster_hp_ratio * 100)),
            "monster_died": monster_died,
            "red_pixels": red_pixel_count,
        }

    # =========================================================================
    # 3. Yetenekler (Skills) ve Cooldown Süreleri İncelemesi
    # =========================================================================

    def _analyze_skills_and_cooldowns(self, frame: np.ndarray, now: float) -> Dict[str, Any]:
        """
        Alt skill barındaki 12 yuvayı (1, 2, 3, 4, 5, 6, 7, 8, Q, E, R, T) inceler:
        - Hangi skillerin bekleme süresi bitti (hazır)
        - Hangi skillerin bekleme süresinde (cooldown) veya manası yetersiz olduğunu öğrenir.
        """
        ready_list: List[str] = []
        cd_map: Dict[str, Any] = {}
        cd_brightness: Dict[str, float] = {}

        if self.skill_bar_reader is not None and hasattr(self.skill_bar_reader, "inspect_skill_bar"):
            try:
                slot_states = self.skill_bar_reader.inspect_skill_bar(frame)
                for slot_key, info in slot_states.items():
                    if info.get("ready", False) and info.get("mana_ok", True):
                        ready_list.append(slot_key)
                    else:
                        # F-0482: bu değer süre değil, ölçülen parlaklıktır.
                        # cooldown_skills yalnızca "cooldown'da" bayrağı
                        # taşır (True); parlaklık ayrı sözlükte tutulur.
                        cd_map[slot_key] = True
                        cd_brightness[slot_key] = round(float(info.get("brightness", 0.0)), 1)
            except Exception as exc:
                logger.debug("Skill bar okuma istisnası: %s", exc)

        # Auto-detect casts by detecting transition from ready to cooldown
        new_casts = [
            k for k in self.ready_skills
            if k in cd_map and k not in self.cooldown_skills
        ]
        for cast_k in new_casts:
            self.record_skill_cast(cast_k, timestamp=now)

        # F-0341: okuyucu yokken ya da boş döndüğünde doğrulanmamış '1'
        # uydurulmaz — hazır listesi boş kalır; bot doğrulanmamış
        # skile spam yapmaz.
        self.ready_skills = ready_list
        self.cooldown_skills = cd_map
        self.cooldown_brightness = cd_brightness

        return {
            "ready_skills": ready_list,
            "cooldown_skills": cd_map,
            "cooldown_brightness": cd_brightness,
            "best_skill": self.select_best_skill(frame),
            "combo_sequence": self.get_combo_sequence(frame),
        }

    def select_best_skill(self, frame: Optional[np.ndarray] = None) -> str:
        """En uygun hazır yeteneği seçer (cooldown'da olmayan ve dinamik DPS ağırlığı en yüksek olan)."""
        pref = self.get_skills_by_priority()
        if self.skill_bar_reader is not None and hasattr(self.skill_bar_reader, "select_best_ready_skill"):
            try:
                res = self.skill_bar_reader.select_best_ready_skill(frame, preferred_order=pref)
                if res:
                    return str(res)
            except Exception as exc:
                # Dilim-30 B-08: the reader failure used to fall through
                # with a bare `pass` — the local weight-sorted fallback
                # (which DOES use learned weights) silently replaced the
                # reader's choice. Throttled warning: this runs per frame.
                now_w = time.monotonic()
                if (now_w - float(getattr(self, "_last_select_reader_warn_ts", 0.0))) >= 60.0:
                    self._last_select_reader_warn_ts = now_w
                    logger.warning(
                        "SkillBarReader.select_best_ready_skill failed, "
                        "local weight-sorted fallback used: %s",
                        exc,
                    )

        with self._combat_learning_lock:
            candidates = [
                k for k in self.ready_skills
                if k not in self.cooldown_skills
            ]
            if not candidates:
                # F-0341: doğrulanmış hazır skill yoksa '1' uydurulmaz.
                return ""

            std_prio = dict(_STD_SKILL_PRIO)
            candidates.sort(
                key=lambda k: (
                    self.get_skill_weight(k),
                    std_prio.get(k.upper(), 0),
                ),
                reverse=True,
            )
            return candidates[0]

    def get_combo_sequence(self, frame: Optional[np.ndarray] = None) -> List[str]:
        """Hazır olan ve art arda basılabilecek kombo yetenek dizisini dinamik DPS ağırlık sırasıyla döndürür."""
        with self._combat_learning_lock:
            combo = [
                k for k in self.ready_skills
                if k not in self.cooldown_skills
            ]
            if not combo:
                # F-0341: doğrulanmış hazır skill yoksa kombo boştur.
                return []
            std_prio = dict(_STD_SKILL_PRIO)
            combo.sort(
                key=lambda k: (
                    self.get_skill_weight(k),
                    std_prio.get(k.upper(), 0),
                ),
                reverse=True,
            )
            return combo

    # =========================================================================
    # 4. Oyuncu Can & Mana (Vitality) ve Otomatik İksir Sistemi
    # =========================================================================

    def _analyze_player_vitality(self, frame: np.ndarray, now: float) -> Dict[str, Any]:
        """
        Ekranın altındaki oyuncu HP ve MP barlarını inceler:
        - HP < %50 ise otomatik '7' (HP İksiri) basar.
        - MP < %35 ise otomatik '8' (MP İksiri) basar.
        - Cooldown süresince (8 sn) mükerrer iksir basımını engeller.
        """
        h, w = frame.shape[:2]

        # 1. Oyuncu HP Barı Bölgesi (x: 0.355..0.478, y: 0.885..0.905)
        hp_y1, hp_y2 = int(h * 0.885), int(h * 0.905)
        hp_x1, hp_x2 = int(w * 0.355), int(w * 0.478)
        hp_roi = frame[hp_y1:hp_y2, hp_x1:hp_x2]

        # S-3101: okunamayan bar %100 varsayilmaz — olcum yoksa None
        # kalir; iksir tetikleyicisi ve ornek meta'si uydurma %100
        # uzerinden calismaz.
        hp_ratio: Optional[float] = None
        hp_readable = False
        if hp_roi.size > 0:
            hsv_hp = cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV)
            red_mask = (
                ((hsv_hp[:, :, 0] <= 12) | (hsv_hp[:, :, 0] >= 168))
                & (hsv_hp[:, :, 1] >= 95)
                & (hsv_hp[:, :, 2] >= 85)
            )
            col_counts = np.sum(red_mask, axis=0)
            valid_cols = np.where(col_counts >= max(1, int(hp_roi.shape[0] * 0.20)))[0]
            if len(valid_cols) > 0:
                hp_ratio = float(valid_cols[-1] + 1) / float(hp_roi.shape[1])
                hp_ratio = float(np.clip(hp_ratio, 0.05, 1.0))
                hp_readable = True
            else:
                # Combat recovery üzerinden kontrol et
                cr = getattr(self.bot_worker, "combat_recovery", None) if self.bot_worker else None
                if cr is not None:
                    try:
                        if hasattr(cr, "read_hp_ratio_or_none"):
                            cr_ratio = cr.read_hp_ratio_or_none(frame)
                            if cr_ratio is not None:
                                cr_ratio = float(cr_ratio)
                                if 0.0 < cr_ratio <= 1.0:
                                    hp_ratio = cr_ratio
                                    hp_readable = True
                        elif hasattr(cr, "get_player_hp_ratio"):
                            # get_player_hp_ratio okunamazsa 0.0 doner ve
                            # 0.0 "bilinmiyor" demektir — gecerli olcum degil.
                            cr_ratio = float(cr.get_player_hp_ratio(frame))
                            if 0.0 < cr_ratio <= 1.0:
                                hp_ratio = cr_ratio
                                hp_readable = True
                    except Exception:
                        # Okunamayan HP 1.0 varsayilmaz — bilinmiyor kalir.
                        hp_ratio = None
                        hp_readable = False

        # 2. Oyuncu MP Barı Bölgesi (x: 0.522..0.645, y: 0.885..0.905)
        mp_x1, mp_x2 = int(w * 0.522), int(w * 0.645)
        mp_roi = frame[hp_y1:hp_y2, mp_x1:mp_x2]

        mp_ratio: Optional[float] = None
        mp_readable = False
        if mp_roi.size > 0:
            hsv_mp = cv2.cvtColor(mp_roi, cv2.COLOR_BGR2HSV)
            blue_mask = (
                (hsv_mp[:, :, 0] >= 95)
                & (hsv_mp[:, :, 0] <= 135)
                & (hsv_mp[:, :, 1] >= 90)
                & (hsv_mp[:, :, 2] >= 80)
            )
            col_counts_mp = np.sum(blue_mask, axis=0)
            valid_cols_mp = np.where(col_counts_mp >= max(1, int(mp_roi.shape[0] * 0.20)))[0]
            if len(valid_cols_mp) > 0:
                mp_ratio = float(valid_cols_mp[-1] + 1) / float(mp_roi.shape[1])
                mp_ratio = float(np.clip(mp_ratio, 0.05, 1.0))
                mp_readable = True

        # Yalnizca gercek olcum saklanir; okunamayan kare son bilinen
        # degeri uydurma %100 ile ezmez.
        if hp_readable and hp_ratio is not None:
            self.player_hp_ratio = hp_ratio
        if mp_readable and mp_ratio is not None:
            self.player_mp_ratio = mp_ratio
        self.player_hp_readable = hp_readable
        self.player_mp_readable = mp_readable

        # 3. Otomatik İksir Mantığı (X, F6, B otomatik oyun mekanikleri kesinlikle dışlanır)
        worker = self.bot_worker
        driver = getattr(worker, "input_driver", None) if worker else None
        hp_pot_used = False
        mp_pot_used = False

        if worker is not None and hasattr(worker, "_check_and_trigger_auto_potions"):
            try:
                hp_pot_used = bool(worker._check_and_trigger_auto_potions(hp_ratio, mp_ratio, frame))
            except Exception as w_exc:
                logger.debug("worker._check_and_trigger_auto_potions hatası: %s", w_exc)
        else:
            # S-3101: okunamayan bar uzerinden iksir tetiklenmez.
            # P0-6: cooldown damgası + hp_pot_used yalnızca press_key
            # gerçekten gönderildiyse yazılır. Sürücü yoksa (driver None)
            # veya press başarısızsa (0.0/False/istisna) damga YOK —
            # eskisi gibi basılmamış iksir 8 sn "kullanıldı" sayılmaz.
            if hp_ratio is not None and hp_ratio < self.hp_pot_threshold and (now - self._last_hp_pot_ts) >= self.pot_cooldown_sec:
                hp_sent = False
                if driver and hasattr(driver, "press_key"):
                    try:
                        hp_sent = _press_ok(driver.press_key("7", hold_sec=0.065))
                    except Exception as exc:
                        logger.debug("HP iksiri basma hatası: %s", exc)
                if hp_sent:
                    self._last_hp_pot_ts = now
                    hp_pot_used = True
                    logger.info(
                        "🧪 [YZ Can/İksir Koruyucu] Oyuncu Canı: %%%d (< %%%d) -> HP İksiri (7) basıldı!",
                        int(round(hp_ratio * 100)),
                        int(round(self.hp_pot_threshold * 100)),
                    )
                else:
                    logger.debug("HP iksiri gönderilemedi (sürücü yok/başarısız) — cooldown başlatılmadı.")

            if mp_ratio is not None and mp_ratio < self.mp_pot_threshold and (now - self._last_mp_pot_ts) >= self.pot_cooldown_sec:
                mp_sent = False
                if driver and hasattr(driver, "press_key"):
                    try:
                        mp_sent = _press_ok(driver.press_key("8", hold_sec=0.065))
                    except Exception as exc:
                        logger.debug("Mana iksiri basma hatası: %s", exc)
                if mp_sent:
                    self._last_mp_pot_ts = now
                    mp_pot_used = True
                    logger.info(
                        "💧 [YZ Mana/İksir Koruyucu] Oyuncu Manası: %%%d (< %%%d) -> Mana İksiri (8) basıldı!",
                        int(round(mp_ratio * 100)),
                        int(round(self.mp_pot_threshold * 100)),
                    )
                else:
                    logger.debug("Mana iksiri gönderilemedi (sürücü yok/başarısız) — cooldown başlatılmadı.")

        return {
            # S-3101: okunamayan bar None doner — %100 uydurulmaz.
            "player_hp_ratio": round(hp_ratio, 3) if hp_ratio is not None else None,
            "player_hp_percent": int(round(hp_ratio * 100)) if hp_ratio is not None else None,
            "player_mp_ratio": round(mp_ratio, 3) if mp_ratio is not None else None,
            "player_mp_percent": int(round(mp_ratio * 100)) if mp_ratio is not None else None,
            "hp_readable": hp_readable,
            "mp_readable": mp_readable,
            "hp_pot_used": hp_pot_used,
            "mp_pot_used": mp_pot_used,
        }

    # =========================================================================
    # 4. Derinlik, Mağara ve Engel Algılama İncelemesi
    # =========================================================================

    def _analyze_obstacles_and_depth(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Karakterin önündeki hareket koridorunu (kaya, ağaç, mağara duvarı, uçurum) analiz eder.
        """
        obs_detected = False
        obs_type = "none"
        steer_deg = 0.0

        if self.depth_avoider is not None and hasattr(self.depth_avoider, "analyze_forward_corridor"):
            try:
                corridor = self.depth_avoider.analyze_forward_corridor(frame)
                obs_detected = bool(corridor.get("obstacle_detected", False))
                obs_type = str(corridor.get("obstacle_type", "none"))
                steer_deg = float(corridor.get("recommended_steer_deg", 0.0))
            except Exception as exc:
                logger.debug("Derinlik koridoru analiz istisnası: %s", exc)

        # A still character cannot be blocked. Repeating ~20° labels are a stuck classifier.
        if obs_detected and not self._actor_is_moving():
            obs_detected = False
            obs_type = "none"
            steer_deg = 0.0

        self.obstacle_detected = obs_detected
        self.obstacle_type = obs_type
        self.recommended_steer_deg = steer_deg

        return {
            "obstacle_detected": obs_detected,
            "obstacle_type": obs_type,
            "recommended_steer_deg": steer_deg,
        }

    # =========================================================================
    # 5. Yol ve Yürüme Analizi
    # =========================================================================

    def _analyze_path_and_walking(
        self,
        frame: np.ndarray,
        current_state: str,
        current_mode: str,
        now: float,
    ) -> Dict[str, Any]:
        """Yol zeminini ve navigasyon durumunu analiz eder."""
        # WALKING rozeti yetmez: hareket tuşu veya path adımı yoksa yol karesi yazılmaz.
        is_walking = self._actor_is_moving()
        return {
            "is_walking": is_walking,
            "current_state": current_state,
            "current_mode": current_mode,
        }

    # =========================================================================
    # 6. Akıllı Ekran Görüntüsü Kaydetme ve Veri Seti Üretimi
    # =========================================================================

    def _get_yolo_labeler(self):
        """Paylaşılan YOLOAutoLabeler örneği (F-0342): her karede yeniden
        kurulmaz. Kurulum başarısız olursa 60 sn boyunca tekrar denenmez.

        v20-onnx-reload: bot_worker'daki VisionEngine / MaterialScanner /
        ONNXYolo3DDetector enjekte edilir — aksi halde her labeler kendi
        MaterialScanner'ı üzerinden ayrı CUDA session açıyordu.
        """
        if self._yolo_labeler is not None:
            return self._yolo_labeler
        if (time.time() - self._yolo_labeler_failed_ts) < 60.0:
            return None
        try:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            kwargs: Dict[str, Any] = {}
            worker = self.bot_worker
            if worker is not None:
                ve = getattr(worker, "vision", None)
                ms = getattr(worker, "material_scanner", None)
                det = getattr(worker, "yolo_detector", None)
                if ve is not None:
                    kwargs["vision_engine"] = ve
                if ms is not None:
                    kwargs["material_scanner"] = ms
                if det is not None:
                    kwargs["yolo_detector"] = det
            self._yolo_labeler = YOLOAutoLabeler(**kwargs)
            return self._yolo_labeler
        except Exception as exc:
            self._yolo_labeler_failed_ts = time.time()
            logger.debug("YOLOAutoLabeler init failed, will retry later: %s", exc)
            return None

    def _detect_real_boxes(
        self,
        frame: np.ndarray,
        category: str,
        cache_key: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Detect genuine visual bounding boxes via YOLOAutoLabeler.

        Returns a list (possibly empty: genuinely nothing detected) or
        None when the detector itself is unavailable/failed (B-04) — the
        two cases must never be confused: a failed detector must skip
        the save, an empty detection may still save a frame-only record.

        D22 per-frame cache: the detector ignores `category` — every call on
        the same frame returns the same boxes — so _evaluate passes the
        frame's dHash as `cache_key` and the 5 category calls collapse to a
        single auto_label_frame inference. `cache_key=None` disables the
        cache (old behaviour). A shallow list copy is returned so callers
        cannot mutate the cached entry.
        """
        if cache_key is not None and cache_key == self._detect_cache_key:
            return list(self._detect_cache_boxes) if self._detect_cache_boxes is not None else None
        try:
            labeler = self._get_yolo_labeler()
            if labeler is None:
                if cache_key is not None:
                    self._detect_cache_key = cache_key
                    self._detect_cache_boxes = None
                return None
            # V20-YZ-ETİKET: öğrenim yakalaması karedeki HER ŞEYİ etiketler
            # (full_scan=True). Stem verilmediği için ("") HUD/yol stem
            # kapıları kapalı kalırdı; skills/obstacles/paths/vitality
            # kutuları hiç üretilemez, kare hep "monsters" düşerdi.
            # Çıkan kutular zayıf-cap'lidir; _filter_boxes_for_category
            # bölümler, _save_learning_sample incelemeye yazar.
            anns = labeler.auto_label_frame(frame, full_scan=True)
            boxes: List[Dict[str, Any]] = []
            for a in anns:
                if isinstance(a, dict) and "cx" in a and "cy" in a and "w" in a and "h" in a:
                    # B-09: a box without a class_id is NOT a monster (10)
                    # and a box without a confidence is NOT 0.85-confident
                    # (which would clear the CONF_EMIN trusted bar). Unclassed
                    # boxes are dropped; unmeasured confidence falls back
                    # to the weak learn-auto level (0.50).
                    try:
                        cid = int(a["class_id"])
                    except (KeyError, TypeError, ValueError):
                        logger.debug("Dropping detection without class_id: %r", a)
                        continue
                    boxes.append(
                        {
                            "class_id": cid,
                            "box": (
                                float(a["cx"]),
                                float(a["cy"]),
                                float(a["w"]),
                                float(a["h"]),
                            ),
                            "confidence": float(a.get("confidence", _LEARN_AUTO_CONF)),
                        }
                    )
            if cache_key is not None:
                self._detect_cache_key = cache_key
                self._detect_cache_boxes = list(boxes)
            return boxes
        except Exception as exc:
            logger.debug("Visual detection in learner failed: %s", exc)
            if cache_key is not None:
                self._detect_cache_key = cache_key
                self._detect_cache_boxes = None
            return None

    def _evaluate_and_save_learning_samples(
        self,
        frame: np.ndarray,
        now: float,
        current_state: str,
        current_mode: str,
        monster_info: Dict[str, Any],
        skill_info: Dict[str, Any],
        obstacle_info: Dict[str, Any],
        path_info: Dict[str, Any],
        vitality_info: Optional[Dict[str, Any]] = None,
        gate_ok: Optional[bool] = None,
    ) -> None:
        """
        Kullanıcının talep ettiği üzere sadece gem toplarken değil:
        - Canavarla savaşırken / canavar ölünce / hedef seçildiğinde
        - Skiller basılırken ve bekleme süresine girdiğinde
        - Yolda yürürken ve araziyi tanırken
        - Mağara, kaya, engel veya uçuruma yaklaşıldığında
        otomatik olarak ekran görüntüsü çeker ve veri setine kaydeder.

        gate_ok=True ise kare bu çağrı zincirinde zaten kapıdan geçmiştir;
        ikinci kapı denetimi atlanır (aynı karenin dHash'i önbellekte
        olduğundan "duplicate_frame" ile reddedilirdi). gate_ok=None ise
        kapı burada çalışır (eski davranış); gate_ok=False kapıya
        danışmadan düşürür.
        """
        if not self.save_screenshots:
            return

        # GameFrameGate protection: Drop desktop/non-game frames before saving to training dataset
        # D22 + A13: gate_ok verildiyse (observe_frame az önce aynı kareyi
        # değerlendirdi) kapı ikinci kez çalıştırılmaz. Kapı stateful'dır:
        # Kapı 3 (novelty) kabul edilen karenin dHash'ini önbelleğe yazar;
        # ikinci değerlendirme aynı kareyi hamming 0 < 6 ile "duplicate"
        # diye reddeder ve observe_frame yoluyla HİÇBİR öğrenim karesi
        # kaydedilemezdi. gate_ok=None ise eski davranış korunur (kapı
        # burada çalışır); gate_ok=False kapıya danışmadan düşürür.
        if gate_ok is None:
            gate_ok, gate_reason = self.check_game_frame(frame, require_foreground=False)
            if not gate_ok:
                logger.debug("Learning sample dropped by GameFrameGate: %s", gate_reason)
                return
        if not gate_ok:
            return

        vitality_info = vitality_info or {}
        in_combat = (
            current_mode == "combat"
            or str(current_state or "").upper() in ("FIGHTING", "COMBAT", "ATTACKING")
        )
        # Combat: keep live HUD cognition, write disk samples rarely (was freezing game ~0.3 FPS).
        base_iv = self.min_sample_interval_sec * (5.0 if in_combat else 1.0)

        # Çok bulanık veya hareketsiz tekrarlayan kareleri atla
        # F-0483: dHash hesaplanamazsa None döner; rastgele değerle
        # yanlış dedup yapılmaz, hatalı kare dedup kümesine girmez.
        dhash_val = self._compute_dhash(frame)
        if dhash_val is not None and dhash_val in self._seen_dhashes:
            return

        # A13: _detect_real_boxes kategori argümanını yok sayar — aynı
        # kareye her kategori için ayrı YOLO çıkarımı yapmak aynı kutuları
        # 5 kez üretir. Kare başına tek çıkarım yapılır, sonuç paylaşılır.
        # B-04: _detect_once None dönebilir (etiketleyici arızalı) — bu,
        # "hiç nesne yok" ([]) halinden farklıdır; None ise hiçbir dal
        # kayıt yapmaz.
        _shared_boxes: Optional[List[Dict[str, Any]]] = None
        _shared_computed = False

        def _detect_once() -> Optional[List[Dict[str, Any]]]:
            nonlocal _shared_boxes, _shared_computed
            if not _shared_computed:
                # D22 cache_key ile birleşir: aynı karenin dHash'i
                # _detect_real_boxes iç önbelleğinde de tek çıkarımda
                # birleşir (bu _evaluate çağrısı dışındaki çağrılar için
                # de geçerli).
                _shared_boxes = self._detect_real_boxes(frame, "general", cache_key=dhash_val)
                _shared_computed = True
                if _shared_boxes is None:
                    now_w = time.monotonic()
                    if (now_w - self._last_detect_failed_warn_ts) >= 60.0:
                        self._last_detect_failed_warn_ts = now_w
                        logger.warning(
                            "YOLO etiketleyici yok/arızalı — öğrenim kaydı "
                            "atlandı (etiketleyici 60 sn içinde tekrar denenmez)"
                        )
            return _shared_boxes

        # B-03: dhash yalnızca gerçekten kaydedilmiş kare için hatırlanır;
        # hiçbir dal kayıt üretemediyse kare "görüldü" sayılmaz.
        saved_any = False

        # 1. Canavar & Savaş Öğrenimi Örneklemesi (Gerçek görsel tespit)
        if (
            monster_info.get("has_target")
            or monster_info.get("monster_died")
            or in_combat
        ):
            if now - self._last_monster_sample_ts >= base_iv:
                monster_boxes = _detect_once()
                if monster_boxes is None:
                    # B-04: etiketleyici arızası — kayıt yok, throttle yok.
                    pass
                else:
                    # B-06: throttle yalnızca kayıt gerçekten yapıldıysa
                    # tüketilir; başarısız deneme 5 sn sonra tekrarlanır.
                    saved = self._save_learning_sample(
                        frame=frame,
                        category="monsters",
                        annotations_meta={
                            "monster_targeted": monster_info.get("has_target"),
                            "monster_hp_percent": monster_info.get("hp_percent"),
                            "monster_died": monster_info.get("monster_died"),
                            "state": current_state,
                        },
                        boxes=monster_boxes,
                        dhash=dhash_val,
                    ) is not False
                    self._last_monster_sample_ts = now if saved else now - base_iv + 5.0
                    if saved:
                        # F-0343: sayaç yalnızca kayıt gerçekten yapıldıysa artar.
                        self.monsters_captured += 1
                        saved_any = True

        # 2. Skill barı — savaşta seyrek (disk I/O donması, gerçek tespit)
        skill_iv = max(20.0, base_iv * 2.0) if in_combat else max(8.0, self.min_sample_interval_sec * 2.0)
        if now - self._last_skill_sample_ts >= skill_iv:
            # B-10: skill dalı artık koşulsuz periyodik kayıt yapmıyor —
            # bar durumu (hazır/beklemede imzası) son örnekten beri
            # değişmediyse boşta kare kaydedilmez.
            _ready = skill_info.get("ready_skills") or []
            _cd = skill_info.get("cooldown_skills") or {}
            try:
                skill_sig = (
                    tuple(sorted(str(k) for k in _ready)),
                    tuple(sorted(str(k) for k in _cd)),
                )
            except TypeError:
                skill_sig = (tuple(_ready), tuple(_cd))
            if skill_sig != self._last_skill_signature:
                skill_boxes = _detect_once()
                if skill_boxes is None:
                    pass
                else:
                    saved = self._save_learning_sample(
                        frame=frame,
                        category="skills",
                        annotations_meta={
                            "ready_skills": skill_info.get("ready_skills"),
                            "cooldown_skills": skill_info.get("cooldown_skills"),
                            "best_skill": skill_info.get("best_skill"),
                        },
                        boxes=skill_boxes,
                        dhash=dhash_val,
                    ) is not False
                    self._last_skill_sample_ts = now if saved else now - skill_iv + 5.0
                    if saved:
                        self.skills_captured += 1
                        self._last_skill_signature = skill_sig
                        saved_any = True

        # 3. Engel, Mağara & Derinlik Öğrenimi Örneklemesi (Gerçek tespit)
        if (not in_combat) and self._obstacle_sample_allowed(obstacle_info):
            if now - self._last_obstacle_sample_ts >= self.min_sample_interval_sec:
                obstacle_boxes = _detect_once()
                if obstacle_boxes is None:
                    pass
                else:
                    saved = self._save_learning_sample(
                        frame=frame,
                        category="obstacles",
                        annotations_meta={
                            "obstacle_type": obstacle_info.get("obstacle_type"),
                            "recommended_steer_deg": float(obstacle_info.get("recommended_steer_deg") or 0.0),
                            "layout_auto": False,
                        },
                        boxes=obstacle_boxes,
                        dhash=dhash_val,
                    ) is not False
                    self._last_obstacle_sample_ts = (
                        now if saved else now - self.min_sample_interval_sec + 5.0
                    )
                    if saved:
                        self.obstacles_captured += 1
                        saved_any = True

        # 4. Yol & Yürüme Öğrenimi Örneklemesi (Gerçek tespit)
        if (not in_combat) and path_info.get("is_walking") and self._actor_is_moving():
            path_iv = max(8.0, self.min_sample_interval_sec * 2.0)
            if now - self._last_path_sample_ts >= path_iv:
                path_boxes = _detect_once()
                if path_boxes is None:
                    pass
                else:
                    saved = self._save_learning_sample(
                        frame=frame,
                        category="paths",
                        annotations_meta={
                            "walking": True,
                            "state": current_state,
                            "mode": current_mode,
                        },
                        boxes=path_boxes,
                        dhash=dhash_val,
                    ) is not False
                    self._last_path_sample_ts = now if saved else now - path_iv + 5.0
                    if saved:
                        self.paths_captured += 1
                        saved_any = True

        # 5. Oyuncu Can & Mana (Vitality) (Gerçek tespit)
        # S-3101: okunamayan bar None'dur — `or 100` ile %100'e
        # cevrilmez; dusuk-can kosulu ve ornek meta'si uydurma
        # %100 uzerinden calismaz.
        _hp_pct = vitality_info.get("player_hp_percent")
        hp_pct = int(_hp_pct) if _hp_pct is not None else None
        _mp_pct = vitality_info.get("player_mp_percent")
        mp_pct = int(_mp_pct) if _mp_pct is not None else None
        pot_event = bool(vitality_info.get("hp_pot_used") or vitality_info.get("mp_pot_used"))
        vitality_periodic = (not in_combat) and (now - self._last_vitality_sample_ts) >= 20.0
        if pot_event or (hp_pct is not None and hp_pct < 65) or (mp_pct is not None and mp_pct < 50) or vitality_periodic:
            vit_iv = base_iv if in_combat else self.min_sample_interval_sec
            if now - self._last_vitality_sample_ts >= vit_iv:
                vital_boxes = _detect_once()
                if vital_boxes is None:
                    pass
                else:
                    saved = self._save_learning_sample(
                        frame=frame,
                        category="vitality",
                        annotations_meta={
                            "player_hp_percent": hp_pct,
                            "player_mp_percent": mp_pct,
                            "hp_pot_used": vitality_info.get("hp_pot_used"),
                            "mp_pot_used": vitality_info.get("mp_pot_used"),
                        },
                        boxes=vital_boxes,
                        dhash=dhash_val,
                    ) is not False
                    self._last_vitality_sample_ts = now if saved else now - vit_iv + 5.0
                    if saved:
                        self.vitality_captured += 1
                        saved_any = True

        # B-03: dedup yalnızca kaydedilmiş kareyi "görüldü" sayar.
        if saved_any and dhash_val is not None:
            self._remember_dhash(dhash_val)

    def _save_learning_sample(
        self,
        frame: np.ndarray,
        category: str,
        annotations_meta: Dict[str, Any],
        boxes: Optional[Sequence[Dict[str, Any]]] = None,
        dhash: Optional[int] = None,
    ) -> bool:
        """Kareyi hem kategori dizinine hem de YOLO eğitim klasörüne kaydeder.

        Kayıt başarısını döner (F-0343): False ise çağıran sayaç
        artırmaz. Görüntü kodlanamazsa da False döner.

        B-01/B-02: yalnızca bu kategoriye ait kutular yazılır; üst düzey
        `box_norm`/`class_id` ilk kategori kutusudur (artık en düşük
        class_id'li kutu değil). Kategoriye ait kutu yoksa hiçbir dosya
        yazılmaz ve False döner — boş kayıt sayaç şişirmez (B-04/B-10).
        B-13: ölü `box_norm`/`yolo_class_id` parametreleri kaldırıldı
        (hiçbir çağıran geçmiyordu; varsayılanları şablon kutu ve
        canavar-10 uydurmasıydı).
        """
        # B-07: kısmi kayıtta geri alınacak dosyalar (S-3103'ün bu
        # fonksiyondaki karşılığı — yazılan her dosya izlenir).
        written: List[Path] = []
        try:
            # B-01: kategori filtresi — tek doğruluk kaynağı
            # _filter_boxes_for_category (çağıran zaten filtrelemiş olsa
            # da savunma amaçlı tekrar uygulanır).
            label_boxes = _filter_boxes_for_category(boxes, category)
            if not label_boxes:
                logger.debug(
                    "Öğrenim kaydı atlandı [%s]: kategoriye ait kutu yok",
                    category,
                )
                return False

            ts_ns = time.time_ns()
            rnd = random.randint(100, 999)
            base_name = f"learn_{category}_{ts_ns}_{rnd}"

            # 1. Kategori klasörüne kaydet
            cat_dir = LEARNING_DIR / category
            img_path = cat_dir / f"{base_name}.jpg"
            json_path = cat_dir / f"{base_name}.json"

            # Unicode-safe JPEG kaydı
            success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if not success:
                logger.debug("Öğrenim karesi kodlanamadı [%s]; kayıt atlandı", category)
                return False
            with open(img_path, "wb") as f:
                f.write(enc.tobytes())
            written.append(img_path)

            # Genuine visual boxes only: boxes=None or boxes=[] means frame only (no fake synthetic box)
            first_box = label_boxes[0].get("box")
            try:
                first_class_id = int(label_boxes[0].get("class_id"))
            except (KeyError, TypeError, ValueError):
                first_class_id = None
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": time.time(),
                        "category": category,
                        "metadata": annotations_meta,
                        # B-16: denetim izi — hangi kare, hangi çözünürlük,
                        # hangi dedektör. Aynı karenin 5 kopyası (B-01)
                        # artık dhash ile geriye dönük birleştirilebilir.
                        "dhash": f"{int(dhash):#018x}" if dhash is not None else None,
                        "frame_shape": [int(frame.shape[0]), int(frame.shape[1])]
                        if hasattr(frame, "shape") and len(frame.shape) >= 2
                        else None,
                        "detection_source": "yolo_auto_labeler",
                        "box_norm": list(first_box) if first_box else None,
                        "class_id": first_class_id,
                        "boxes": [
                            {
                                "class_id": int(item["class_id"]),
                                "box": list(item["box"]),
                            }
                            for item in label_boxes
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            written.append(json_path)

            # 2. YOLO Eğitim Veri Setine (images/train & labels/train) ekle
            train_img = TRAIN_IMAGES_DIR / f"{base_name}.jpg"
            train_lbl = TRAIN_LABELS_DIR / f"{base_name}.txt"

            if success:
                with open(train_img, "wb") as f:
                    f.write(enc.tobytes())
                written.append(train_img)

            conf = float(_LEARN_AUTO_CONF)
            annotations = []
            for item in label_boxes:
                raw_box = item.get("box")
                if not raw_box or len(raw_box) != 4:
                    continue
                # B-09: sınıfı olmayan kutu atılır — canavar (10)
                # varsayımı yok.
                try:
                    item_cid = int(item["class_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                cx, cy, bw, bh = raw_box
                box_conf = float(item.get("confidence", conf))
                annotations.append(
                    {
                        "class_id": item_cid,
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(bw),
                        "h": float(bh),
                        "confidence": box_conf,
                        "source": str(item.get("source", f"omni_{category}")),
                        # P1-12 (K-31): "Emin" eşiği tek sabittir (CONF_EMIN);
                        # eski 0.80 barı aynı kutuyu başka yollarda weak yapıyordu.
                        # Bu damga tek başına eğitmez — K1 kapısı (train_txt_boxes)
                        # learn_auto kutuyu insan onayı olmadan txt'ye sokmaz.
                        "trusted": bool(item.get("trusted", box_conf >= _CONF_EMIN)),
                        "weak": bool(item.get("weak", box_conf < _CONF_EMIN)),
                        "omni_synthetic": False,  # Pure genuine visual detection, zero synthetic poisoning
                        "learn_auto": True,
                    }
                )
            from dataset_factory.label_trust import (
                commit_split_labels,
                filter_hud_false_positives,
            )

            annotations = filter_hud_false_positives(
                annotations, stem=base_name, remap_sprout=False
            )
            if str(category).lower() == "paths":
                # Only keep path boxes in the lower world band (not minimap).
                annotations = [
                    a
                    for a in annotations
                    if float(a.get("cy", 0.0)) >= 0.55
                    and float(a.get("cx", 0.5)) < 0.82
                ]

            result = commit_split_labels(
                dataset_dir=DATASET_DIR,
                split="train",
                stem=base_name,
                label_path=train_lbl,
                annotations=annotations,
            )
            # P2-21 (O004-K1 / K-07): train_box_count OKUNUR — commit
            # 0 kutu yazdıysa kare "kaydedildi" sayılmaz: sayaç artmaz,
            # True dönmez. (Eskiden debug logda okunup yutuluyordu;
            # total_learned_frames yine de artıyordu.)
            # F3: geri alma YALNIZCA eğitim çıktısını kapsar (train
            # img/lbl) — LEARNING_DIR kaydı (kategori img+json) Adım 2
            # inceleme kuyruğudur, eğitim iddiası değildir. K1/E-06
            # kapısı gereği onaysız auto kutu zaten weak olarak review
            # sidecar'da kalır; kategori kaydını da silmek insan-onay
            # kapısını anlamsız bırakırdı (onaylanacak örnek kalmaz) ve
            # v17'de öğrenici hiçbir auto örneği biriktiremezdi
            # (train_box_count auto kutuda deterministik 0'dır — K-10
            # kanıtı). Review JSON commit tarafından yazılır ve geri
            # alma listesinde zaten değildir; weak kutu incelemede kalır.
            if int(result.get("train_box_count", 0) or 0) <= 0:
                logger.warning(
                    "Öğrenim karesi eğitime girmedi [%s] #%s: commit 0 kutu yazdı "
                    "(dropped=%s trusted_in=%s trusted_out=%s) — inceleme kaydı "
                    "duruyor, sayaç artmadı",
                    category,
                    base_name,
                    result.get("dropped"),
                    result.get("trusted_in"),
                    result.get("trusted_out"),
                )
                for p in (train_img, train_lbl):
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                return False
            if train_lbl.is_file():
                written.append(train_lbl)
            review_json = DATASET_DIR / "review" / "train" / f"{base_name}.json"
            if review_json.is_file():
                written.append(review_json)
            logger.debug(
                "learn_auto write [%s] stem=%s train_boxes=%s weak=%s sources=%s conf=%.2f",
                category,
                base_name,
                result.get("train_box_count"),
                result.get("weak_box_count"),
                sorted(
                    {
                        str(a.get("source", ""))
                        for a in annotations
                        if isinstance(a, dict)
                    }
                ),
                conf,
            )

            self.total_learned_frames += 1
            logger.info(
                "📸 [YZ Öğrenimi] Yeni kare kaydedildi: [%s] #%d -> %s",
                category.upper(),
                self.total_learned_frames,
                img_path.name,
            )
            return True
        except Exception as exc:
            # B-07: kısmi kayıt geri alınır — "başarısız" sayılan kaydın
            # dosyaları diskte kalmaz (S-3103). Hata artık debug değil,
            # warning düzeyinde: sessiz disk arızası gizlenmez (B-14).
            for p in written:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            logger.warning("Öğrenim karesi kaydetme hatası [%s]: %s", category, exc)
            return False

    def _remember_dhash(self, dhash_val: int) -> None:
        """Records a frame hash in the bounded dedup history.

        Oldest-first eviction at _DEDUP_MAX_HASHES entries; the history is
        never wiped wholesale (the old clear() at >600 forgot every frame
        at once, so duplicates seen before the wipe could be saved again).
        `_seen_dhashes` stays a plain set, so external `.clear()` calls
        (e.g. tests resetting dedup state) keep working; stale order
        entries are skipped harmlessly during eviction.
        """
        if dhash_val in self._seen_dhashes:
            return
        self._seen_dhashes.add(dhash_val)
        self._seen_dhash_order.append(dhash_val)
        while (
            len(self._seen_dhashes) > _DEDUP_MAX_HASHES
            and self._seen_dhash_order
        ):
            oldest = self._seen_dhash_order.popleft()
            self._seen_dhashes.discard(oldest)

    @staticmethod
    def _compute_dhash(image: np.ndarray, hash_size: int = 8) -> Optional[int]:
        """Hızlı perceptual fark karma (dHash) algoritması.

        Hata durumunda rastgele değer DÖNMEZ (F-0483): rastgele hash hem
        yanlış dedup'a hem de tekrar kayda yol açar. None döner; çağıran
        dedup'u atlar.

        D22: INTER_AREA ile 3-kanal tam kare küçültmek çıkarımın en
        pahalı adımıydı (~7 ms/kare, profil: süre %98 resize'da).
        INTER_LINEAR + önce küçült/sonra griye çevir + packbits ile aynı
        dedup sözleşmesi korunur (aynı/neredeyse aynı kare → aynı hash,
        farklı kare → farklı hash) ve hash ~100 kat hızlanır. Hash
        değerleri eski sürümle birebir aynı değildir ama hash'ler
        diske yazılmaz, yalnızca oturum içi dedup'ta kullanılır.
        """
        try:
            resized = cv2.resize(
                image, (hash_size + 1, hash_size), interpolation=cv2.INTER_LINEAR
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            diff = gray[:, 1:] > gray[:, :-1]
            return int.from_bytes(np.packbits(diff.flatten()).tobytes(), "big")
        except Exception:
            return None

    # =========================================================================
    # 7. Canlı Biliş ve Telemetri Özeti
    # =========================================================================

    def get_live_cognition_snapshot(self) -> Dict[str, Any]:
        """Uygulama ve HUD paneline anında yansıtılacak canlı öğrenim ve algı durumu."""
        with self._lock:
            return {
                "total_learned_frames": self.total_learned_frames,
                "monsters_captured": self.monsters_captured,
                "skills_captured": self.skills_captured,
                "paths_captured": self.paths_captured,
                "obstacles_captured": self.obstacles_captured,
                "vitality_captured": self.vitality_captured,
                "monster_targeted": self.current_monster_targeted,
                # B-15: hedef yokken canavar canı %0 değildir — bilinmiyor.
                # Tüketiciler None görünce atlar (state_machine F-0046).
                "monster_hp_ratio": round(self.current_monster_hp_ratio, 2)
                if self.current_monster_targeted
                else None,
                "monster_hp_percent": int(round(self.current_monster_hp_ratio * 100))
                if self.current_monster_targeted
                else None,
                # B-15: okunmamış bar %100 değildir — S-3101 doktrini:
                # okunabilirlik bayrağı ayrı alanda durur, yüzde None olur.
                # Tüketiciler (perception_hub, state_machine) None
                # görünce doldurmaz; bayrağa bakmayan tüketici de artık
                # sahte %100 görmez.
                "player_hp_ratio": round(self.player_hp_ratio, 2)
                if self.player_hp_readable
                else None,
                "player_hp_percent": int(round(self.player_hp_ratio * 100))
                if self.player_hp_readable
                else None,
                "player_mp_ratio": round(self.player_mp_ratio, 2)
                if self.player_mp_readable
                else None,
                "player_mp_percent": int(round(self.player_mp_ratio * 100))
                if self.player_mp_readable
                else None,
                # S-3101: saklanan deger son bilinen olcumdur; okunabilirlik
                # bayragi olmadan %100 taze olcum sanilmaz.
                "player_hp_readable": bool(self.player_hp_readable),
                "player_mp_readable": bool(self.player_mp_readable),
                "ready_skills": list(self.ready_skills),
                "combo_sequence": self.get_combo_sequence(),
                "obstacle_detected": self.obstacle_detected,
                "obstacle_type": self.obstacle_type,
                "recommended_steer_deg": round(self.recommended_steer_deg, 1),
                "skill_dps_weights": self.get_skill_dps_weights(),
                "encounter_speed_rating": self.last_encounter_speed_rating,
                "last_encounter_dps": round(self.last_encounter_dps, 4),
                "total_encounters": self.total_encounters,
                "fast_encounters": self.fast_encounters,
            }
