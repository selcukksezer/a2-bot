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
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

logger = logging.getLogger("Gatherion.RlRewardLog")

# Anchor default paths to the repo root (this file's parent), never to the
# current working directory — otherwise the same code writes to a different
# database depending on where the process was launched from.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(_REPO_ROOT / "dataset" / "economy.sqlite")
JSONL_PATH = str(_REPO_ROOT / "dataset" / "rl_reward.jsonl")


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
    # False when neither the SQLite nor the JSONL sink accepted the sample.
    persisted: bool = True


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
        jsonl_max_bytes: int = 10 * 1024 * 1024,
        jsonl_backups: int = 3,
    ) -> None:
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        # O-6: JSONL rotasyonu (varsayılan 10 MB, 3 yedek .1/.2/.3).
        self.jsonl_max_bytes = max(0, int(jsonl_max_bytes))
        self.jsonl_backups = max(0, int(jsonl_backups))
        self._jsonl_rotations = 0
        self._lock = threading.RLock()
        self._episode_reward = 0.0
        self._samples = 0
        self._write_errors = 0
        # I3 (dev/D006): one shared SQLite connection (WAL + NORMAL, see
        # SlotEconomy) and one persistent JSONL handle instead of a new
        # connect + a file open/close per tick. log_tick used to pay a
        # full commit (fsync) plus an open/close on every tick.
        self._conn: Optional[sqlite3.Connection] = None
        self._jsonl_fh: Optional[TextIO] = None
        self._ensure()

    def _connect(self) -> Optional[sqlite3.Connection]:
        """Shared connection (lazy; None if the DB cannot be opened).

        Called with self._lock held. A failed DB must not kill the
        JSONL sink: callers treat None as "sqlite sink unavailable".
        """
        if self._conn is not None:
            return self._conn
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if mode and str(mode[0]).lower() == "wal":
                    conn.execute("PRAGMA synchronous=NORMAL")
            except Exception as exc:
                logger.debug("rl_reward WAL pragma: %s", exc)
            self._conn = conn
            return conn
        except Exception as exc:
            logger.debug("rl_rewards connect: %s", exc)
            return None

    def close(self) -> None:
        """Flush and close the JSONL handle and the shared connection."""
        with self._lock:
            fh, self._jsonl_fh = self._jsonl_fh, None
            conn, self._conn = self._conn, None
        if fh is not None:
            try:
                fh.flush()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.commit()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self) -> None:  # best effort; never raise at shutdown
        try:
            self.close()
        except Exception:
            pass

    def _ensure(self) -> None:
        try:
            conn = self._connect()
            if conn is None:
                return
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
            # Persist first: a sample whose write failed must not count toward
            # the episode totals nor be returned as a successful sample.
            persisted = self._write(sample)
            sample.persisted = persisted
            if persisted:
                self._episode_reward += reward
                self._samples += 1
            else:
                self._write_errors += 1
                # O-6: yazma hatası sayacı üretimde okunmuyordu (tek okuyucu
                # smoke testi) — eşiklerde görünür özet ver; her başarısız
                # tick warning'i kalır.
                if self._write_errors in (1, 10) or self._write_errors % 100 == 0:
                    logger.error(
                        "rl_reward yazma hatası eşiği: %d başarısız tick (toplam %d örnek)",
                        self._write_errors,
                        self._samples,
                    )
                logger.warning("rl_reward tick not persisted (db+jsonl write failed)")
        return sample

    def _maybe_rotate_jsonl(self) -> None:
        """O-6: JSONL boyut rotasyonu (varsayılan 10 MB, 3 yedek .1/.2/.3).

        self._lock altında çağrılır (log_tick -> _write içinden). Sınır
        0 ise rotasyon kapalıdır. Eski yedek zinciri kaydırılır (.3
        silinir, .2->.3, .1->.2, mevcut->.1), sayaç artar.
        """
        if self.jsonl_max_bytes <= 0 or self.jsonl_backups <= 0:
            return
        try:
            if not os.path.isfile(self.jsonl_path):
                return
            if os.path.getsize(self.jsonl_path) < self.jsonl_max_bytes:
                return
        except OSError:
            return
        try:
            if self._jsonl_fh is not None:
                try:
                    self._jsonl_fh.flush()
                except Exception:
                    pass
                try:
                    self._jsonl_fh.close()
                except Exception:
                    pass
                self._jsonl_fh = None
            oldest = f"{self.jsonl_path}.{self.jsonl_backups}"
            if os.path.isfile(oldest):
                os.remove(oldest)
            for idx in range(self.jsonl_backups - 1, 0, -1):
                src = f"{self.jsonl_path}.{idx}"
                dst = f"{self.jsonl_path}.{idx + 1}"
                if os.path.isfile(src):
                    os.replace(src, dst)
            os.replace(self.jsonl_path, f"{self.jsonl_path}.1")
            self._jsonl_rotations += 1
            logger.info(
                "rl_reward JSONL rotasyonu: %s -> %s.1 (rotasyon #%d)",
                self.jsonl_path,
                self.jsonl_path,
                self._jsonl_rotations,
            )
        except OSError as exc:
            logger.warning("rl_reward JSONL rotasyonu başarısız: %s", exc)

    def _write(self, sample: RewardSample) -> bool:
        """Writes the sample to SQLite and JSONL. True if at least one sink accepted it."""
        persisted = False
        conn = self._connect()
        if conn is not None:
            try:
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
                persisted = True
            except Exception as exc:
                logger.debug("rl_rewards insert: %s", exc)
                # A dead connection (closed DB, unreadable file, locked
                # DB left in a bad state) must not poison every later
                # tick: drop it so the next tick reconnects.
                try:
                    conn.close()
                except Exception:
                    pass
                self._conn = None
        # JSONL sink: one handle kept open for the logger's lifetime,
        # flushed after every sample (same durability as the old
        # open/write/close per tick, without the per-tick open cost).
        # A failed handle is dropped so the next tick retries the open.
        # O-6: yazmadan önce boyut rotasyonu kontrol edilir.
        try:
            self._maybe_rotate_jsonl()
            if self._jsonl_fh is None:
                os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
                self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")
            self._jsonl_fh.write(
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
            self._jsonl_fh.flush()
            persisted = True
        except Exception as exc:
            logger.debug("rl_reward jsonl: %s", exc)
            if self._jsonl_fh is not None:
                try:
                    self._jsonl_fh.close()
                except Exception:
                    pass
                self._jsonl_fh = None
        return persisted

    def episode_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "episode_reward": self._episode_reward,
                "mean_reward": (self._episode_reward / self._samples) if self._samples else 0.0,
                "write_errors": self._write_errors,
            }

    def health_summary(self) -> Dict[str, Any]:
        """O-6: üretimde okunabilir sağlık özeti — örnek/hata sayaçları,
        JSONL rotasyon sayısı ve güncel JSONL boyutu. episode_summary()
        yalnızca smoke testinde okunuyordu; bu özet izleme/HUD için."""
        with self._lock:
            try:
                jsonl_bytes = os.path.getsize(self.jsonl_path) if os.path.isfile(self.jsonl_path) else 0
            except OSError:
                jsonl_bytes = -1
            return {
                "samples": self._samples,
                "episode_reward": self._episode_reward,
                "write_errors": self._write_errors,
                "jsonl_rotations": int(getattr(self, "_jsonl_rotations", 0) or 0),
                "jsonl_bytes": jsonl_bytes,
                "jsonl_max_bytes": int(self.jsonl_max_bytes),
                "jsonl_backups": int(self.jsonl_backups),
            }

    def reset_episode(self) -> None:
        with self._lock:
            self._episode_reward = 0.0
            self._samples = 0
            self._write_errors = 0


class ObstacleMemory:
    """Thin wrapper around SlotEconomy.obstacles or standalone table."""

    def __init__(self, slot_economy: Any = None, db_path: str = DB_PATH) -> None:
        self.slot_economy = slot_economy
        self.db_path = db_path
        self._last_pos: Optional[tuple] = None
        self._still_since: Optional[float] = None
        self.stuck_threshold_sec = 3.0
        # Position for which a stuck obstacle was already recorded; a stuck
        # episode records once instead of once per note_position() call.
        self._stuck_recorded_pos: Optional[tuple] = None

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
            self._stuck_recorded_pos = None
            return None
        if self._last_pos == pos:
            if self._still_since is None:
                self._still_since = now
            stuck_for = now - self._still_since
            if stuck_for >= self.stuck_threshold_sec:
                if self._stuck_recorded_pos != pos:
                    if self.slot_economy is not None and hasattr(self.slot_economy, "record_obstacle"):
                        self.slot_economy.record_obstacle(cx, cy, reason="stuck")
                    else:
                        self._record_local(cx, cy)
                    self._stuck_recorded_pos = pos
                return stuck_for
            return None
        self._last_pos = pos
        self._still_since = None
        self._stuck_recorded_pos = None
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
