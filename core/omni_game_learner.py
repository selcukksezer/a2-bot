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

# Layout conf ≥ TRAIN_CONF_FLOOR so boxes survive purge; still weak + learn_auto.
try:
    from dataset_factory.label_trust import LEARN_AUTO_CONF as _LEARN_AUTO_CONF
except Exception:  # pragma: no cover - import cycle / early boot
    _LEARN_AUTO_CONF = 0.50


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
        self.min_sample_interval_sec = max(4.0, float(min_sample_interval_sec))

        # Otomatik İksir ve Canlılık Yapılandırması
        self.hp_pot_threshold: float = 0.50
        self.mp_pot_threshold: float = 0.35
        self.pot_cooldown_sec: float = 8.0
        self._last_hp_pot_ts: float = 0.0
        self._last_mp_pot_ts: float = 0.0

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

        # Canlı durum takibi
        self.current_monster_targeted: bool = False
        self.current_monster_hp_ratio: float = 0.0
        self.prev_monster_hp_ratio: float = 0.0
        self.last_monster_death_ts: float = 0.0
        self._learner_no_target_frames: int = 0

        self.player_hp_ratio: float = 1.0
        self.player_mp_ratio: float = 1.0

        self.ready_skills: List[str] = ["1"]
        self.cooldown_skills: Dict[str, float] = {}

        self.obstacle_detected: bool = False
        self.obstacle_type: str = "none"
        self.recommended_steer_deg: float = 0.0

        # Event-driven combat learning & dynamic skill DPS metrics
        self._combat_learning_lock = threading.RLock()
        self.combat_skill_keys: Tuple[str, ...] = ("1", "2", "3", "4", "5", "Q", "E", "R")
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

        # Perceptual hash geçmişi (mükerrer kare kaydetmeme koruması)
        self._seen_dhashes: set = set()

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
            logger.debug("Öğrenim dizinleri oluşturulurken bilgi: %s", exc)

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
        """Normalizes skill key to standard hotbar identifier ('1'..'5', 'Q', 'E', 'R')."""
        s = str(key or "").strip()
        if not s:
            return ""
        upper = s.upper()
        if upper in ("Q", "E", "R"):
            return upper
        if s in ("1", "2", "3", "4", "5"):
            return s
        if upper in ("1", "2", "3", "4", "5"):
            return upper
        return upper

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
                ok, reason = _is_valid_game_frame(frame, require_foreground=require_foreground)
                return bool(ok), str(reason)
            except Exception as exc:
                logger.debug("GameFrameGate evaluation error: %s", exc)

        # Heuristic fallback if GameFrameGate module is unavailable
        std_val = float(np.std(frame))
        if std_val < 4.0:
            return False, "flat_solid_color"
        return True, "accepted_fallback"

    def record_skill_cast(
        self,
        skill_key: str,
        hp_ratio: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Records an event-driven skill cast by key ('1'..'5', 'Q', 'E', 'R').
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
                duration = max(0.5, now - self.last_monster_death_ts) if self.last_monster_death_ts > 0 else 5.0
            else:
                duration = max(0.5, now - self._encounter_start_ts)
                self._in_encounter = False

            self.last_encounter_duration = duration
            self.total_encounters += 1
            if self.avg_encounter_duration <= 0.0:
                self.avg_encounter_duration = duration
            else:
                self.avg_encounter_duration = 0.80 * self.avg_encounter_duration + 0.20 * duration

            total_dmg = sum(self._encounter_damage_by_skill.values())
            if total_dmg <= 0.0 and killed:
                total_dmg = max(0.10, self._encounter_start_hp)

            encounter_dps = total_dmg / duration
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
            mean_all_dmg = (
                float(np.mean([m["avg_hp_drop"] for m in active_skills]))
                if active_skills
                else 0.10
            )

            if monster_died and dur <= 6.5:
                speed_mult = min(1.80, ref_duration / max(1.5, dur))
            elif dur >= 14.0 or not monster_died:
                speed_mult = max(0.60, ref_duration / dur)
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
        """Returns all 8 combat skill keys ordered by dynamic DPS weight descending."""
        with self._combat_learning_lock:
            std_prio = {"R": 8, "E": 7, "Q": 6, "2": 5, "3": 4, "4": 3, "5": 2, "1": 1}
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
        """Propagates dynamically tuned weights to BotWorker and SkillBarReader."""
        weights = self.get_skill_dps_weights()
        if self.skill_bar_reader is not None:
            if hasattr(self.skill_bar_reader, "DEFAULT_PRIORITY_WEIGHTS"):
                try:
                    for k, w in weights.items():
                        self.skill_bar_reader.DEFAULT_PRIORITY_WEIGHTS[k] = w
                        self.skill_bar_reader.DEFAULT_PRIORITY_WEIGHTS[k.lower()] = w
                except Exception as exc:
                    logger.debug("Could not propagate weights to skill_bar_reader: %s", exc)

        if self.bot_worker is not None:
            sbm = getattr(self.bot_worker, "skill_bar_monitor", None)
            if sbm and hasattr(sbm, "DEFAULT_PRIORITY_WEIGHTS"):
                try:
                    for k, w in weights.items():
                        sbm.DEFAULT_PRIORITY_WEIGHTS[k] = w
                        sbm.DEFAULT_PRIORITY_WEIGHTS[k.lower()] = w
                except Exception:
                    pass

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
            monster_info = self._analyze_monster_and_health(frame)

            # Event-driven combat encounter tracking & target HP drop processing
            if monster_info.get("has_target"):
                curr_hp = float(monster_info.get("hp_ratio", 0.0))
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
                self.observe_mix(frame, mix_sources, now=now)
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
        """
        from dataset_factory.multi_mix import commit_mix_sample, fuse_sources

        fused = fuse_sources(sources)
        now = time.time() if now is None else float(now)
        if not force and (now - self._last_general_sample_ts) < self.min_sample_interval_sec:
            fused["saved"] = False
            fused["reason"] = "throttled"
            return fused
        if not fused.get("annotations"):
            fused["saved"] = False
            fused["reason"] = "empty"
            return fused

        self._last_general_sample_ts = now
        sample_stem = stem or f"mix_{int(now * 1000)}_{random.randint(100, 999)}"
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
                    (LEARNING_DIR / "mix" / f"{sample_stem}.jpg").write_bytes(enc.tobytes())
            except Exception as exc:
                logger.debug("mix frame save skipped: %s", exc)

        root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
        result = commit_mix_sample(
            dataset_dir=root,
            stem=sample_stem,
            fused=fused,
            split="train",
        )
        trusted_n = int(result.get("train_box_count", 0))
        if (
            trusted_n > 0
            and frame is not None
            and isinstance(frame, np.ndarray)
            and frame.size
        ):
            try:
                train_images = root / "images" / "train"
                train_images.mkdir(parents=True, exist_ok=True)
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    (train_images / f"{sample_stem}.jpg").write_bytes(enc.tobytes())
            except Exception as exc:
                logger.debug("mix train image skipped: %s", exc)
        self.total_learned_frames += 1
        fused["saved"] = True
        fused["stem"] = sample_stem
        fused["train_box_count"] = trusted_n
        fused["weak_box_count"] = int(result.get("weak_box_count", 0))
        fused["label_path"] = result.get("label_path") or ""
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
                # Fill ratio vs active bar span (not vs huge ROI) — old formula
                # under-reported HP / flickered to ~0 and cleared combat lock.
                span = max(1.0, width_px)
                filled = float(np.sum(col_sums[active_cols[0] : active_cols[-1] + 1] > 2))
                fill_ratio = filled / span
                max_expected_bar_w = float(target_roi.shape[1] * 0.55)
                width_ratio = min(1.0, width_px / max(40.0, max_expected_bar_w))
                hp_ratio = min(1.0, max(0.08, max(width_ratio, fill_ratio)))
            else:
                hp_ratio = max(0.35, float(self.current_monster_hp_ratio) or 0.35)
        elif has_reticle:
            # Soft-lock without top bar: keep last known HP if any, else mid placeholder.
            prev = float(self.current_monster_hp_ratio or 0.0)
            hp_ratio = prev if prev >= 0.08 else 0.55
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
        Alt skill barındaki 8 ana yuvayı (1, 2, 3, 4, 5, Q, E, R) inceler:
        - Hangi skillerin bekleme süresi bitti (hazır)
        - Hangi skillerin bekleme süresinde (cooldown) veya manası yetersiz olduğunu öğrenir.
        """
        ready_list: List[str] = []
        cd_map: Dict[str, float] = {}

        if self.skill_bar_reader is not None and hasattr(self.skill_bar_reader, "inspect_skill_bar"):
            try:
                slot_states = self.skill_bar_reader.inspect_skill_bar(frame)
                for slot_key, info in slot_states.items():
                    if info.get("ready", False) and info.get("mana_ok", True):
                        ready_list.append(slot_key)
                    else:
                        cd_map[slot_key] = round(float(info.get("brightness", 0.0)), 1)
            except Exception as exc:
                logger.debug("Skill bar okuma istisnası: %s", exc)

        # Auto-detect casts by detecting transition from ready to cooldown
        new_casts = [
            k for k in self.ready_skills
            if k in cd_map and k not in self.cooldown_skills
        ]
        for cast_k in new_casts:
            self.record_skill_cast(cast_k, timestamp=now)

        if not ready_list:
            ready_list = ["1"]  # Güvenli temel vuruş fallback

        self.ready_skills = ready_list
        self.cooldown_skills = cd_map

        return {
            "ready_skills": ready_list,
            "cooldown_skills": cd_map,
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
            except Exception:
                pass

        with self._combat_learning_lock:
            candidates = [
                k for k in self.ready_skills
                if k not in self.cooldown_skills
            ]
            if not candidates:
                return self.ready_skills[0] if self.ready_skills else "1"

            std_prio = {"R": 8, "E": 7, "Q": 6, "2": 5, "3": 4, "4": 3, "5": 2, "1": 1}
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
                return ["1"]
            std_prio = {"R": 8, "E": 7, "Q": 6, "2": 5, "3": 4, "4": 3, "5": 2, "1": 1}
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

        hp_ratio = 1.0
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
            else:
                # Combat recovery üzerinden kontrol et
                cr = getattr(self.bot_worker, "combat_recovery", None) if self.bot_worker else None
                if cr and hasattr(cr, "get_player_hp_ratio"):
                    try:
                        hp_ratio = float(cr.get_player_hp_ratio(frame))
                    except Exception:
                        hp_ratio = 1.0

        # 2. Oyuncu MP Barı Bölgesi (x: 0.522..0.645, y: 0.885..0.905)
        mp_x1, mp_x2 = int(w * 0.522), int(w * 0.645)
        mp_roi = frame[hp_y1:hp_y2, mp_x1:mp_x2]

        mp_ratio = 1.0
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

        self.player_hp_ratio = hp_ratio
        self.player_mp_ratio = mp_ratio

        # 3. Otomatik İksir Mantığı (X, F6, B otomatik oyun mekanikleri kesinlikle dışlanır)
        worker = self.bot_worker
        driver = getattr(worker, "input_driver", None) if worker else None
        hp_pot_used = False
        mp_pot_used = False

        if worker is not None and hasattr(worker, "_check_and_trigger_auto_potions"):
            try:
                hp_pot_used = bool(worker._check_and_trigger_auto_potions(hp_ratio, mp_ratio, frame_bgr))
            except Exception as w_exc:
                logger.debug("worker._check_and_trigger_auto_potions hatası: %s", w_exc)
        else:
            if hp_ratio < self.hp_pot_threshold and (now - self._last_hp_pot_ts) >= self.pot_cooldown_sec:
                self._last_hp_pot_ts = now
                hp_pot_used = True
                if driver and hasattr(driver, "press_key"):
                    try:
                        driver.press_key("7", hold_sec=0.065)
                        logger.info(
                            "🧪 [YZ Can/İksir Koruyucu] Oyuncu Canı: %%%d (< %%%d) -> HP İksiri (7) basıldı!",
                            int(round(hp_ratio * 100)),
                            int(round(self.hp_pot_threshold * 100)),
                        )
                    except Exception as exc:
                        logger.debug("HP iksiri basma hatası: %s", exc)

            if mp_ratio < self.mp_pot_threshold and (now - self._last_mp_pot_ts) >= self.pot_cooldown_sec:
                self._last_mp_pot_ts = now
                mp_pot_used = True
                if driver and hasattr(driver, "press_key"):
                    try:
                        driver.press_key("8", hold_sec=0.065)
                        logger.info(
                            "💧 [YZ Mana/İksir Koruyucu] Oyuncu Manası: %%%d (< %%%d) -> Mana İksiri (8) basıldı!",
                            int(round(mp_ratio * 100)),
                            int(round(self.mp_pot_threshold * 100)),
                        )
                    except Exception as exc:
                        logger.debug("Mana iksiri basma hatası: %s", exc)

        return {
            "player_hp_ratio": round(hp_ratio, 3),
            "player_hp_percent": int(round(hp_ratio * 100)),
            "player_mp_ratio": round(mp_ratio, 3),
            "player_mp_percent": int(round(mp_ratio * 100)),
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

    def _detect_real_boxes(self, frame: np.ndarray, category: str) -> List[Dict[str, Any]]:
        """
        Detect genuine visual bounding boxes via YOLOAutoLabeler.
        Returns empty list [] if no genuine objects are detected; never fabricates
        hardcoded coordinates.
        """
        try:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            labeler = YOLOAutoLabeler()
            anns = labeler.auto_label_frame(frame)
            boxes: List[Dict[str, Any]] = []
            for a in anns:
                if isinstance(a, dict) and "cx" in a and "cy" in a and "w" in a and "h" in a:
                    boxes.append(
                        {
                            "class_id": int(a.get("class_id", 10)),
                            "box": (
                                float(a["cx"]),
                                float(a["cy"]),
                                float(a["w"]),
                                float(a["h"]),
                            ),
                            "confidence": float(a.get("confidence", 0.85)),
                        }
                    )
            return boxes
        except Exception as exc:
            logger.debug("Visual detection in learner failed: %s", exc)
            return []

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
    ) -> None:
        """
        Kullanıcının talep ettiği üzere sadece gem toplarken değil:
        - Canavarla savaşırken / canavar ölünce / hedef seçildiğinde
        - Skiller basılırken ve bekleme süresine girdiğinde
        - Yolda yürürken ve araziyi tanırken
        - Mağara, kaya, engel veya uçuruma yaklaşıldığında
        otomatik olarak ekran görüntüsü çeker ve veri setine kaydeder.
        """
        if not self.save_screenshots:
            return

        # GameFrameGate protection: Drop desktop/non-game frames before saving to training dataset
        is_valid_frame, gate_reason = self.check_game_frame(frame, require_foreground=False)
        if not is_valid_frame:
            logger.debug("Learning sample dropped by GameFrameGate: %s", gate_reason)
            return

        vitality_info = vitality_info or {}
        in_combat = (
            current_mode == "combat"
            or str(current_state or "").upper() in ("FIGHTING", "COMBAT", "ATTACKING")
        )
        # Combat: keep live HUD cognition, write disk samples rarely (was freezing game ~0.3 FPS).
        base_iv = self.min_sample_interval_sec * (5.0 if in_combat else 1.0)

        # Çok bulanık veya hareketsiz tekrarlayan kareleri atla
        dhash_val = self._compute_dhash(frame)
        if dhash_val in self._seen_dhashes:
            return

        # 1. Canavar & Savaş Öğrenimi Örneklemesi (Gerçek görsel tespit)
        if (
            monster_info.get("has_target")
            or monster_info.get("monster_died")
            or in_combat
        ):
            if now - self._last_monster_sample_ts >= base_iv:
                self._last_monster_sample_ts = now
                monster_boxes = self._detect_real_boxes(frame, "monsters")
                self._save_learning_sample(
                    frame=frame,
                    category="monsters",
                    annotations_meta={
                        "monster_targeted": monster_info.get("has_target"),
                        "monster_hp_percent": monster_info.get("hp_percent"),
                        "monster_died": monster_info.get("monster_died"),
                        "state": current_state,
                    },
                    boxes=monster_boxes,
                )
                self.monsters_captured += 1

        # 2. Skill barı — savaşta seyrek (disk I/O donması, gerçek tespit)
        skill_iv = max(20.0, base_iv * 2.0) if in_combat else max(8.0, self.min_sample_interval_sec * 2.0)
        if now - self._last_skill_sample_ts >= skill_iv:
            self._last_skill_sample_ts = now
            skill_boxes = self._detect_real_boxes(frame, "skills")
            self._save_learning_sample(
                frame=frame,
                category="skills",
                annotations_meta={
                    "ready_skills": skill_info.get("ready_skills"),
                    "cooldown_skills": skill_info.get("cooldown_skills"),
                    "best_skill": skill_info.get("best_skill"),
                },
                boxes=skill_boxes,
            )
            self.skills_captured += 1

        # 3. Engel, Mağara & Derinlik Öğrenimi Örneklemesi (Gerçek tespit)
        if (not in_combat) and self._obstacle_sample_allowed(obstacle_info):
            if now - self._last_obstacle_sample_ts >= self.min_sample_interval_sec:
                self._last_obstacle_sample_ts = now
                steer = float(obstacle_info.get("recommended_steer_deg") or 0.0)
                obstacle_boxes = self._detect_real_boxes(frame, "obstacles")
                self._save_learning_sample(
                    frame=frame,
                    category="obstacles",
                    annotations_meta={
                        "obstacle_type": obstacle_info.get("obstacle_type"),
                        "recommended_steer_deg": steer,
                        "layout_auto": False,
                    },
                    boxes=obstacle_boxes,
                )
                self.obstacles_captured += 1

        # 4. Yol & Yürüme Öğrenimi Örneklemesi (Gerçek tespit)
        if (not in_combat) and path_info.get("is_walking") and self._actor_is_moving():
            if now - self._last_path_sample_ts >= max(8.0, self.min_sample_interval_sec * 2.0):
                self._last_path_sample_ts = now
                path_boxes = self._detect_real_boxes(frame, "paths")
                self._save_learning_sample(
                    frame=frame,
                    category="paths",
                    annotations_meta={
                        "walking": True,
                        "state": current_state,
                        "mode": current_mode,
                    },
                    boxes=path_boxes,
                )
                self.paths_captured += 1

        # 5. Oyuncu Can & Mana (Vitality) (Gerçek tespit)
        hp_pct = int(vitality_info.get("player_hp_percent", 100) or 100)
        mp_pct = int(vitality_info.get("player_mp_percent", 100) or 100)
        pot_event = bool(vitality_info.get("hp_pot_used") or vitality_info.get("mp_pot_used"))
        vitality_periodic = (not in_combat) and (now - self._last_vitality_sample_ts) >= 20.0
        if pot_event or hp_pct < 65 or mp_pct < 50 or vitality_periodic:
            if now - self._last_vitality_sample_ts >= (base_iv if in_combat else self.min_sample_interval_sec):
                self._last_vitality_sample_ts = now
                vital_boxes = self._detect_real_boxes(frame, "vitality")
                self._save_learning_sample(
                    frame=frame,
                    category="vitality",
                    annotations_meta={
                        "player_hp_percent": hp_pct,
                        "player_mp_percent": mp_pct,
                        "hp_pot_used": vitality_info.get("hp_pot_used"),
                        "mp_pot_used": vitality_info.get("mp_pot_used"),
                    },
                    boxes=vital_boxes,
                )
                self.vitality_captured += 1

        self._seen_dhashes.add(dhash_val)
        if len(self._seen_dhashes) > 600:
            self._seen_dhashes.clear()

    def _save_learning_sample(
        self,
        frame: np.ndarray,
        category: str,
        annotations_meta: Dict[str, Any],
        yolo_class_id: int = 10,
        box_norm: Tuple[float, float, float, float] = (0.5, 0.5, 0.2, 0.2),
        boxes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Kareyi hem kategori dizinine hem de YOLO eğitim klasörüne kaydeder."""
        try:
            ts_int = int(time.time() * 1000)
            rnd = random.randint(100, 999)
            base_name = f"learn_{category}_{ts_int}_{rnd}"

            # 1. Kategori klasörüne kaydet
            cat_dir = LEARNING_DIR / category
            img_path = cat_dir / f"{base_name}.jpg"
            json_path = cat_dir / f"{base_name}.json"

            # Unicode-safe JPEG kaydı
            success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if success:
                with open(img_path, "wb") as f:
                    f.write(enc.tobytes())

            # Genuine visual boxes only: boxes=None or boxes=[] means frame only (no fake synthetic box)
            if boxes is None:
                label_boxes = []
            else:
                label_boxes = [b for b in boxes if isinstance(b, dict) and "box" in b]

            first_box = label_boxes[0].get("box") if label_boxes else None
            first_class_id = (
                int(label_boxes[0].get("class_id", yolo_class_id))
                if label_boxes
                else None
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": time.time(),
                        "category": category,
                        "metadata": annotations_meta,
                        "box_norm": list(first_box) if first_box else None,
                        "class_id": first_class_id,
                        "boxes": [
                            {
                                "class_id": int(item.get("class_id", yolo_class_id)),
                                "box": list(item["box"]),
                            }
                            for item in label_boxes
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 2. YOLO Eğitim Veri Setine (images/train & labels/train) ekle
            train_img = TRAIN_IMAGES_DIR / f"{base_name}.jpg"
            train_lbl = TRAIN_LABELS_DIR / f"{base_name}.txt"

            if success:
                with open(train_img, "wb") as f:
                    f.write(enc.tobytes())

            conf = float(_LEARN_AUTO_CONF)
            annotations = []
            for item in label_boxes:
                raw_box = item.get("box")
                if not raw_box or len(raw_box) != 4:
                    continue
                cx, cy, bw, bh = raw_box
                box_conf = float(item.get("confidence", conf))
                annotations.append(
                    {
                        "class_id": int(item.get("class_id", yolo_class_id)),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(bw),
                        "h": float(bh),
                        "confidence": box_conf,
                        "source": str(item.get("source", f"omni_{category}")),
                        "trusted": bool(item.get("trusted", box_conf >= 0.80)),
                        "weak": bool(item.get("weak", box_conf < 0.80)),
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
        except Exception as exc:
            logger.debug("Öğrenim karesi kaydetme hatası: %s", exc)

    @staticmethod
    def _compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
        """Hızlı perceptual fark karma (dHash) algoritması."""
        try:
            resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            diff = gray[:, 1:] > gray[:, :-1]
            return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        except Exception:
            return random.randint(0, 10000000)

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
                "monster_hp_ratio": round(self.current_monster_hp_ratio, 2),
                "monster_hp_percent": int(round(self.current_monster_hp_ratio * 100)),
                "player_hp_ratio": round(self.player_hp_ratio, 2),
                "player_hp_percent": int(round(self.player_hp_ratio * 100)),
                "player_mp_ratio": round(self.player_mp_ratio, 2),
                "player_mp_percent": int(round(self.player_mp_ratio * 100)),
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
