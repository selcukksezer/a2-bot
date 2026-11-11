# -*- coding: utf-8 -*-
"""
SlotEconomy — SQLite monsters/drops/slots + KPH + UCB farm-slot policy.

Mevcut economy_broker üzerine eklenir; broker JSON silinmez.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Gatherion.SlotEconomy")

DB_PATH = os.path.join("dataset", "economy.sqlite")


@dataclass
class SlotPick:
    slot_id: str
    cx: float
    cy: float
    kph: float
    explore: bool
    reason: str = ""


class SlotEconomy:
    def __init__(
        self,
        db_path: str = DB_PATH,
        explore_ratio: float = 0.20,
        ucb_c: float = 1.4,
        on_log: Optional[Callable[[str], None]] = None,
        item_value_lookup: Optional[Callable[[str], float]] = None,
        grid_meters: float = 100.0,
    ) -> None:
        self.db_path = db_path
        self.explore_ratio = max(0.05, min(0.5, float(explore_ratio)))
        self.ucb_c = float(ucb_c)
        self._on_log = on_log
        self.item_value_lookup = item_value_lookup
        self.grid_meters = float(grid_meters)
        self._lock = threading.RLock()
        self._session_start = time.time()
        self._current_slot: Optional[str] = None
        self._slot_entered_ts = time.time()
        # I2 (dev/D006): one shared connection instead of a new
        # sqlite3.connect() per call. Every call used to open the file,
        # run 1-2 statements and commit (fsync) — ~170 ms/commit under
        # load on this VM. The shared connection is created with WAL +
        # synchronous=NORMAL (commits no longer fsync the journal on
        # every write; a process crash still loses nothing, only an
        # OS crash / power loss can lose the last transaction — the
        # documented SQLite trade-off, acceptable for farm stats).
        # All DB access runs under self._lock (sqlite3 connection with
        # check_same_thread=False); see _connect().
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def set_explore_ratio(self, ratio: float) -> None:
        """F-23-11: ucb_explore_ratio'yu canlı değiştirir (yeniden başlatmasız).

        main._on_settings_saved -> BotWorker.refresh_consumable_settings
        yolundan çağrılır. Aralık kurucu ile aynıdır (0.05–0.5); kilit
        altında yazılır çünkü seçim (UCB) başka thread'den okunabilir.
        """
        with self._lock:
            self.explore_ratio = max(0.05, min(0.5, float(ratio)))

    def close(self) -> None:
        """Commit pending work and close the shared connection.

        Safe to call repeatedly; instances that are never closed are
        still reclaimed at process exit (the OS closes the fd and
        SQLite recovers the WAL on the next open).
        """
        with self._lock:
            conn, self._conn = self._conn, None
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

    def __enter__(self) -> "SlotEconomy":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _emit(self, msg: str) -> None:
        logger.info(msg)
        if callable(self._on_log):
            try:
                self._on_log(msg)
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        """Return the shared connection, creating it on first use.

        Must be called with self._lock held (or during __init__ before
        the instance escapes). One connection per instance; concurrent
        use from several threads is serialised by the RLock, and
        check_same_thread=False allows that sharing.
        """
        if self._conn is not None:
            return self._conn
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode and str(mode[0]).lower() == "wal":
                conn.execute("PRAGMA synchronous=NORMAL")
        except Exception as exc:
            logger.debug("slot_economy WAL pragma: %s", exc)
        self._conn = conn
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS monsters (
                    mob_key TEXT PRIMARY KEY,
                    name TEXT,
                    avg_ttk REAL DEFAULT 0,
                    samples INTEGER DEFAULT 0,
                    avg_damage_taken REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS drops (
                    drop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    mob_key TEXT,
                    kina REAL DEFAULT 0,
                    item_name TEXT,
                    item_value REAL DEFAULT 0,
                    slot_id TEXT
                );
                CREATE TABLE IF NOT EXISTS slots (
                    slot_id TEXT PRIMARY KEY,
                    region TEXT,
                    cx REAL,
                    cy REAL,
                    density REAL DEFAULT 0,
                    kph REAL DEFAULT 0,
                    visits INTEGER DEFAULT 0,
                    total_kina REAL DEFAULT 0,
                    total_item_value REAL DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    total_hours REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS obstacles (
                    obstacle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    cx REAL,
                    cy REAL,
                    reason TEXT,
                    hits INTEGER DEFAULT 1
                );
                """
            )
            conn.commit()

    def slot_id_from_coords(self, x: float, y: float, region: str = "map") -> str:
        gx = int(math.floor(float(x) / self.grid_meters))
        gy = int(math.floor(float(y) / self.grid_meters))
        safe_region = "".join(c if c.isalnum() else "_" for c in str(region or "map"))
        return f"Slot_{safe_region}_{gx:03d}_{gy:03d}"

    def enter_slot(self, slot_id: str, cx: float = 0.0, cy: float = 0.0, region: str = "") -> None:
        with self._lock:
            self._flush_slot_time_locked()
            self._current_slot = str(slot_id)
            self._slot_entered_ts = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO slots(slot_id, region, cx, cy, visits)
                    VALUES(?,?,?,?,1)
                    ON CONFLICT(slot_id) DO UPDATE SET
                        visits = visits + 1,
                        cx = COALESCE(excluded.cx, cx),
                        cy = COALESCE(excluded.cy, cy),
                        region = COALESCE(NULLIF(excluded.region,''), region)
                    """,
                    (slot_id, region, float(cx), float(cy)),
                )
                conn.commit()

    def _flush_slot_time_locked(self) -> None:
        if not self._current_slot:
            return
        hours = max(0.0, (time.time() - self._slot_entered_ts) / 3600.0)
        if hours <= 0:
            return
        slot_id = self._current_slot
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE slots SET total_hours = total_hours + ? WHERE slot_id = ?",
                    (hours, slot_id),
                )
                self._recompute_kph(conn, slot_id)
                conn.commit()
        except Exception as exc:
            logger.debug("flush slot time: %s", exc)
        self._slot_entered_ts = time.time()

    def record_kill(
        self,
        mob_key: str,
        ttk_sec: float = 0.0,
        damage_taken: float = 0.0,
        name: str = "",
    ) -> None:
        key = str(mob_key or "unknown")
        # I2: DB work under the lock — the connection is now shared.
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT avg_ttk, samples, avg_damage_taken FROM monsters WHERE mob_key=?",
                    (key,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO monsters(mob_key, name, avg_ttk, samples, avg_damage_taken) VALUES(?,?,?,?,?)",
                        (key, name or key, float(ttk_sec), 1, float(damage_taken)),
                    )
                else:
                    n = int(row["samples"] or 0) + 1
                    avg_ttk = ((float(row["avg_ttk"] or 0) * (n - 1)) + float(ttk_sec)) / n
                    avg_dmg = ((float(row["avg_damage_taken"] or 0) * (n - 1)) + float(damage_taken)) / n
                    conn.execute(
                        "UPDATE monsters SET name=?, avg_ttk=?, samples=?, avg_damage_taken=? WHERE mob_key=?",
                        (name or key, avg_ttk, n, avg_dmg, key),
                    )
                conn.commit()
        try:
            from core.lifetime_stats import get_lifetime_stats

            get_lifetime_stats().record_monster_kill(mob_key=key)
        except Exception as exc:
            logger.debug("lifetime kill from slot_economy skipped: %s", exc)

    def record_drop(
        self,
        kina: float = 0.0,
        item_name: str = "",
        mob_key: str = "",
        slot_id: Optional[str] = None,
        pot_scroll_cost: float = 0.0,
    ) -> None:
        sid = slot_id or self._current_slot or "Slot_unknown"
        item_val = 0.0
        if item_name and callable(self.item_value_lookup):
            try:
                item_val = float(self.item_value_lookup(item_name) or 0.0)
            except Exception:
                item_val = 0.0
        elif item_name:
            # F-1922: no price lookup -> unknown value is 0, never a fabricated
            # 1000 Kina that inflates KPH. Unknown items contribute no value
            # until a real price is known.
            item_val = 0.0
            logger.debug("bilinmeyen eşya fiyatı yok, değer 0 sayıldı: %s", item_name)
        try:
            kina_i = int(float(kina) or 0)
        except (TypeError, ValueError):
            kina_i = 0
        if kina_i > 0:
            try:
                from core.lifetime_stats import get_lifetime_stats

                raw = str(mob_key or item_name or "drop")[:64]
                get_lifetime_stats().record_gold(
                    kina_i,
                    fingerprint=f"kina:{kina_i}:{raw}:{time.time_ns()}:{uuid.uuid4().hex[:12]}",
                )
            except Exception as exc:
                logger.debug("lifetime gold from slot_economy skipped: %s", exc)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO drops(ts, mob_key, kina, item_name, item_value, slot_id)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (time.time(), mob_key, float(kina), item_name, item_val, sid),
                )
                conn.execute(
                    """
                    INSERT INTO slots(slot_id, total_kina, total_item_value, total_cost, visits)
                    VALUES(?,?,?,?,0)
                    ON CONFLICT(slot_id) DO UPDATE SET
                        total_kina = total_kina + excluded.total_kina,
                        total_item_value = total_item_value + excluded.total_item_value,
                        total_cost = total_cost + excluded.total_cost
                    """,
                    (sid, float(kina), float(item_val), float(pot_scroll_cost)),
                )
                # S265 (v19/P3-8): drop kaydı eskiden visits'i de +1
                # artırıyordu — visits giriş+drop toplamı oluyor, UCB
                # n_i'si "çekiş sayısı" değil "kanıt miktarı" oluyordu.
                # visits yalnızca slot girişinde artar (yukarıdaki
                # INSERT); drop yalnızca ekonomik toplamları yazar.
                self._recompute_kph(conn, sid)
                conn.commit()

    def _recompute_kph(self, conn: sqlite3.Connection, slot_id: str) -> None:
        row = conn.execute(
            "SELECT total_kina, total_item_value, total_cost, total_hours FROM slots WHERE slot_id=?",
            (slot_id,),
        ).fetchone()
        if row is None:
            return
        hours = max(1.0 / 60.0, float(row["total_hours"] or 0.0))  # floor 1 minute
        net = float(row["total_kina"] or 0) + float(row["total_item_value"] or 0) - float(row["total_cost"] or 0)
        kph = net / hours
        conn.execute("UPDATE slots SET kph=? WHERE slot_id=?", (kph, slot_id))

    def compute_slot_score(
        self,
        sum_kina: float,
        sum_item_value: float,
        pot_scroll_cost: float,
        hours: float,
    ) -> float:
        h = max(1e-6, float(hours))
        return (float(sum_kina) + float(sum_item_value) - float(pot_scroll_cost)) / h

    def list_slots(self) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT slot_id, region, cx, cy, density, kph, visits, total_hours FROM slots"
                ).fetchall()
                return [dict(r) for r in rows]

    def pick_slot_ucb(self, total_pulls: Optional[int] = None) -> Optional[SlotPick]:
        """
        ~explore_ratio chance of underexplored slot; else max UCB / KPH.
        UCB1: kph + c * sqrt(ln(N)/n_i)
        """
        slots = self.list_slots()
        if not slots:
            return None
        n_total = int(total_pulls) if total_pulls is not None else sum(int(s.get("visits") or 0) for s in slots)
        n_total = max(1, n_total)

        # Exploration branch: least visits.
        if random.random() < self.explore_ratio:
            slots_sorted = sorted(slots, key=lambda s: (int(s.get("visits") or 0), -float(s.get("kph") or 0)))
            s = slots_sorted[0]
            pick = SlotPick(
                slot_id=str(s["slot_id"]),
                cx=float(s.get("cx") or 0),
                cy=float(s.get("cy") or 0),
                kph=float(s.get("kph") or 0),
                explore=True,
                reason="explore_least_visits",
            )
            self._emit(f"SLOT_UCB pick={pick.slot_id} explore=1 kph={pick.kph:.1f}")
            return pick

        best: Optional[SlotPick] = None
        best_score = -1e18
        # F-1936: scale the UCB exploration bonus into KPH units; a raw bonus
        # of ~1-2 is invisible next to KPH values in the thousands.
        kph_scale = max(1.0, max(float(s.get("kph") or 0.0) for s in slots))
        for s in slots:
            n_i = max(1, int(s.get("visits") or 0))
            kph = float(s.get("kph") or 0)
            bonus = self.ucb_c * math.sqrt(math.log(n_total + 1.0) / n_i) * kph_scale
            score = kph + bonus
            if score > best_score:
                best_score = score
                best = SlotPick(
                    slot_id=str(s["slot_id"]),
                    cx=float(s.get("cx") or 0),
                    cy=float(s.get("cy") or 0),
                    kph=kph,
                    explore=False,
                    reason=f"ucb={score:.1f}",
                )
        if best:
            self._emit(f"SLOT_UCB pick={best.slot_id} explore=0 kph={best.kph:.1f}")
        return best

    def record_obstacle(self, cx: float, cy: float, reason: str = "stuck") -> None:
        with self._lock:
            with self._connect() as conn:
                # Merge nearby obstacles within ~15 units.
                row = conn.execute(
                    """
                    SELECT obstacle_id, hits FROM obstacles
                    WHERE ABS(cx-?)<15 AND ABS(cy-?)<15
                    ORDER BY hits DESC LIMIT 1
                    """,
                    (float(cx), float(cy)),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE obstacles SET hits = hits + 1, ts=?, reason=? WHERE obstacle_id=?",
                        (time.time(), reason, int(row["obstacle_id"])),
                    )
                else:
                    conn.execute(
                        "INSERT INTO obstacles(ts, cx, cy, reason, hits) VALUES(?,?,?,?,1)",
                        (time.time(), float(cx), float(cy), reason),
                    )
                conn.commit()

    def obstacles_near(self, cx: float, cy: float, radius: float = 40.0) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM obstacles WHERE ABS(cx-?)<=? AND ABS(cy-?)<=?",
                    (float(cx), float(radius), float(cy), float(radius)),
                ).fetchall()
                return [dict(r) for r in rows]

    def suggested_hint(self) -> Optional[Dict[str, Any]]:
        pick = self.pick_slot_ucb()
        if pick is None:
            return None
        return {
            "slot_id": pick.slot_id,
            "cx": pick.cx,
            "cy": pick.cy,
            "kph": pick.kph,
            "explore": pick.explore,
            "reason": pick.reason,
        }

    def estimate_consumable_depletion(
        self,
        hp_pots_used: int,
        mp_pots_used: int,
        scrolls_used: int,
        elapsed_minutes: float,
    ) -> Dict[str, Any]:
        return estimate_consumable_depletion(
            hp_pots_used=hp_pots_used,
            mp_pots_used=mp_pots_used,
            scrolls_used=scrolls_used,
            elapsed_minutes=elapsed_minutes,
        )


def estimate_consumable_depletion(
    hp_pots_used: int,
    mp_pots_used: int,
    scrolls_used: int,
    elapsed_minutes: float,
    hp_pots_initial_stock: Optional[int] = None,
    mp_pots_initial_stock: Optional[int] = None,
    scrolls_initial_stock: Optional[int] = None,
) -> Dict[str, Any]:
    """
    İksir (HP/MP) ve parşömen kullanım hızına göre stok tükenme tahmini yapar.
    F-1941: başlangıç stokları bilinmiyorsa (None) uydurma 100/100/50
    varsayılmaz; kalan/ETA o kalem için None döner.
    """
    hp_used = max(0, int(hp_pots_used or 0))
    mp_used = max(0, int(mp_pots_used or 0))
    sc_used = max(0, int(scrolls_used or 0))
    mins = max(1e-3, float(elapsed_minutes or 0.0))

    hp_rate_per_min = hp_used / mins
    mp_rate_per_min = mp_used / mins
    sc_rate_per_min = sc_used / mins

    def _remaining(stock: Optional[int], used: int) -> Optional[int]:
        if stock is None:
            return None
        return max(0, int(stock) - used)

    hp_remaining = _remaining(hp_pots_initial_stock, hp_used)
    mp_remaining = _remaining(mp_pots_initial_stock, mp_used)
    sc_remaining = _remaining(scrolls_initial_stock, sc_used)

    def _eta_minutes(remaining: Optional[int], rate_per_min: float) -> Optional[float]:
        if remaining is None:
            return None
        if remaining <= 0:
            return 0.0
        if rate_per_min <= 1e-9:
            return None
        return round(remaining / rate_per_min, 2)

    hp_eta_min = _eta_minutes(hp_remaining, hp_rate_per_min)
    mp_eta_min = _eta_minutes(mp_remaining, mp_rate_per_min)
    sc_eta_min = _eta_minutes(sc_remaining, sc_rate_per_min)

    etas = {
        "hp_pots": hp_eta_min,
        "mp_pots": mp_eta_min,
        "scrolls": sc_eta_min,
    }
    finite_etas = {k: v for k, v in etas.items() if v is not None}
    if finite_etas:
        limiting_item = min(finite_etas, key=lambda k: finite_etas[k])
        min_eta_min: Optional[float] = finite_etas[limiting_item]
    elif any(v is None for v in (hp_pots_initial_stock, mp_pots_initial_stock, scrolls_initial_stock)):
        limiting_item = "unknown"
        min_eta_min = None
    else:
        limiting_item = "none"
        min_eta_min = None

    needs_restock = bool(min_eta_min is not None and min_eta_min <= 10.0)

    return {
        "initial_stock_known": {
            "hp_pots": hp_pots_initial_stock is not None,
            "mp_pots": mp_pots_initial_stock is not None,
            "scrolls": scrolls_initial_stock is not None,
        },
        "elapsed_minutes": round(mins, 3),
        "used": {
            "hp_pots": hp_used,
            "mp_pots": mp_used,
            "scrolls": sc_used,
        },
        "remaining_estimate": {
            "hp_pots": hp_remaining,
            "mp_pots": mp_remaining,
            "scrolls": sc_remaining,
        },
        "burn_rate_per_min": {
            "hp_pots": round(hp_rate_per_min, 4),
            "mp_pots": round(mp_rate_per_min, 4),
            "scrolls": round(sc_rate_per_min, 4),
        },
        "burn_rate_per_hour": {
            "hp_pots": round(hp_rate_per_min * 60.0, 2),
            "mp_pots": round(mp_rate_per_min * 60.0, 2),
            "scrolls": round(sc_rate_per_min * 60.0, 2),
        },
        "depletion_eta_minutes": etas,
        "limiting_consumable": limiting_item,
        "min_minutes_to_depletion": min_eta_min,
        "needs_restock": needs_restock,
    }
