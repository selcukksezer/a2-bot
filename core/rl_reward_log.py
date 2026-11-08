# -*- coding: utf-8 -*-
"""
RL reward logger — tick ödül skoru dosya/SQLite'a yazılır; canlı PPO yok.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Gatherion.RlRewardLog")

DB_PATH = os.path.join("dataset", "economy.sqlite")
JSONL_PATH = os.path.join("dataset", "rl_reward.jsonl")


@dataclass
class RewardSample:
    ts: float
    reward: float
    kina: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    wasted_mana: float = 0.0
    stuck_time: float = 0.0
    slot_id: str = ""
    meta: Optional[Dict[str, Any]] = None


def compute_reward(
    kina: float = 0.0,
    damage_dealt: float = 0.0,
    damage_taken: float = 0.0,
    wasted_mana: float = 0.0,
    stuck_time: float = 0.0,
    kina_w: float = 1.0,
    dmg_w: float = 0.01,
    taken_w: float = 0.02,
    mana_w: float = 0.5,
    stuck_w: float = 2.0,
) -> float:
    """
    r = +kina +damage -damage_taken -wasted_mana -stuck_time
    (weights scale raw units into a comparable score)
    """
    return (
        float(kina_w) * float(kina)
        + float(dmg_w) * float(damage_dealt)
        - float(taken_w) * float(damage_taken)
        - float(mana_w) * float(wasted_mana)
        - float(stuck_w) * float(stuck_time)
    )


class RlRewardLog:
    def __init__(
        self,
        db_path: str = DB_PATH,
        jsonl_path: str = JSONL_PATH,
    ) -> None:
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._lock = threading.RLock()
        self._episode_reward = 0.0
        self._samples = 0
        self._ensure()

    def _ensure(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rl_rewards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL,
                        reward REAL,
                        kina REAL,
                        damage_dealt REAL,
                        damage_taken REAL,
                        wasted_mana REAL,
                        stuck_time REAL,
                        slot_id TEXT,
                        meta TEXT
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.debug("rl_rewards schema: %s", exc)

    def log_tick(
        self,
        kina: float = 0.0,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
        wasted_mana: float = 0.0,
        stuck_time: float = 0.0,
        slot_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> RewardSample:
        reward = compute_reward(
            kina=kina,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            wasted_mana=wasted_mana,
            stuck_time=stuck_time,
        )
        sample = RewardSample(
            ts=time.time(),
            reward=reward,
            kina=float(kina),
            damage_dealt=float(damage_dealt),
            damage_taken=float(damage_taken),
            wasted_mana=float(wasted_mana),
            stuck_time=float(stuck_time),
            slot_id=str(slot_id or ""),
            meta=meta,
        )
        with self._lock:
            self._episode_reward += reward
            self._samples += 1
            self._write(sample)
        return sample

    def _write(self, sample: RewardSample) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO rl_rewards(
                        ts, reward, kina, damage_dealt, damage_taken,
                        wasted_mana, stuck_time, slot_id, meta
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sample.ts,
                        sample.reward,
                        sample.kina,
                        sample.damage_dealt,
                        sample.damage_taken,
                        sample.wasted_mana,
                        sample.stuck_time,
                        sample.slot_id,
                        json.dumps(sample.meta or {}, ensure_ascii=False),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("rl_rewards insert: %s", exc)
        try:
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": sample.ts,
                            "reward": sample.reward,
                            "kina": sample.kina,
                            "damage_dealt": sample.damage_dealt,
                            "damage_taken": sample.damage_taken,
                            "wasted_mana": sample.wasted_mana,
                            "stuck_time": sample.stuck_time,
                            "slot_id": sample.slot_id,
                            "meta": sample.meta or {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.debug("rl_reward jsonl: %s", exc)

    def episode_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "episode_reward": self._episode_reward,
                "mean_reward": (self._episode_reward / self._samples) if self._samples else 0.0,
            }

    def reset_episode(self) -> None:
        with self._lock:
            self._episode_reward = 0.0
            self._samples = 0


class ObstacleMemory:
    """Thin wrapper around SlotEconomy.obstacles or standalone table."""

    def __init__(self, slot_economy: Any = None, db_path: str = DB_PATH) -> None:
        self.slot_economy = slot_economy
        self.db_path = db_path
        self._last_pos: Optional[tuple] = None
        self._still_since: Optional[float] = None
        self.stuck_threshold_sec = 3.0

    def note_position(self, cx: float, cy: float, moving_cmd: bool = True) -> Optional[float]:
        """
        If movement was commanded but coords unchanged for stuck_threshold_sec,
        record obstacle and return stuck duration.
        """
        now = time.time()
        pos = (round(float(cx), 1), round(float(cy), 1))
        if not moving_cmd:
            self._last_pos = pos
            self._still_since = None
            return None
        if self._last_pos == pos:
            if self._still_since is None:
                self._still_since = now
            stuck_for = now - self._still_since
            if stuck_for >= self.stuck_threshold_sec:
                if self.slot_economy is not None and hasattr(self.slot_economy, "record_obstacle"):
                    self.slot_economy.record_obstacle(cx, cy, reason="stuck")
                else:
                    self._record_local(cx, cy)
                return stuck_for
            return None
        self._last_pos = pos
        self._still_since = None
        return None

    def _record_local(self, cx: float, cy: float) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS obstacles (
                        obstacle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL, cx REAL, cy REAL, reason TEXT, hits INTEGER DEFAULT 1
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO obstacles(ts, cx, cy, reason, hits) VALUES(?,?,?,?,1)",
                    (time.time(), float(cx), float(cy), "stuck"),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("obstacle local record: %s", exc)
