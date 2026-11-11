# -*- coding: utf-8 -*-
"""
LifetimeStatsStore — kalıcı (ömür boyu) toplama / altın / canavar sayaçları.

Uygulama restart, bot restart veya oturum değişiminde silinmez.
Yalnızca açık kullanıcı onayıyla sıfırlanır.
Atomik yazım: config/lifetime_stats.json.tmp → os.replace.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("Gatherion.LifetimeStats")

_DEFAULT_REL = os.path.join("config", "lifetime_stats.json")
_KILL_DEBOUNCE_SEC = 1.8
_GOLD_DEDUP_TTL_SEC = 45.0
# I4 (dev/D006): cap on the per-mob debounce table. One entry was kept
# forever per mob_key ever killed, so a long session grew it without
# bound. Entries older than the debounce window can never suppress a
# kill again, so they are pruned once the table passes this size.
_KILL_DEBOUNCE_MAX_KEYS = 1024


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_stats_path() -> str:
    return os.path.join(_project_root(), _DEFAULT_REL)


def _empty_payload() -> Dict[str, Any]:
    return {
        "materials_by_key": {},
        "gold_earned": 0,
        "monsters_killed": 0,
        "last_updated": None,
        "version": 1,
    }


class LifetimeStatsStore:
    """Thread-safe persistent counters for gather / gold / kills."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(path or default_stats_path())
        self._lock = threading.RLock()
        self._materials_by_key: Dict[str, int] = {}
        self._gold_earned: int = 0
        self._monsters_killed: int = 0
        self._last_updated: Optional[str] = None
        self._last_kill_ts: float = 0.0
        # F-0118: debounce is per mob_key, not global — a kill of mob B must
        # not be swallowed because mob A died <1.8s earlier.
        self._last_kill_ts_by_mob: Dict[str, float] = {}
        self._recent_gold_fps: Dict[str, float] = {}
        # F-0119: path of the quarantined corrupt file (if _load found one).
        self._corrupt_backup: Optional[str] = None
        self._load()

    def _quarantine_corrupt_file(self) -> None:
        """F-0119: move a corrupt lifetime file aside instead of silently
        resetting it and overwriting the (possibly recoverable) data."""
        backup = f"{self.path}.corrupt.{int(time.time())}"
        try:
            os.replace(self.path, backup)
            self._corrupt_backup = backup
            logger.error(
                "lifetime_stats dosyası bozuk, yedeğe taşındı (%s -> %s); "
                "sayaçlar boş başlatıldı",
                self.path,
                backup,
            )
        except OSError as exc:
            logger.error("bozuk lifetime_stats dosyası yedeklenemedi (%s): %s", self.path, exc)

    def _load(self) -> None:
        with self._lock:
            if not os.path.isfile(self.path):
                return
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if not isinstance(raw, dict):
                    raise ValueError("lifetime_stats kökü dict değil")
                mats = raw.get("materials_by_key") or {}
                cleaned: Dict[str, int] = {}
                if isinstance(mats, dict):
                    for key, val in mats.items():
                        name = str(key or "").strip()
                        if not name:
                            continue
                        try:
                            qty = int(val)
                        except (TypeError, ValueError):
                            continue
                        if qty > 0:
                            cleaned[name] = cleaned.get(name, 0) + qty
                self._materials_by_key = cleaned
                try:
                    self._gold_earned = max(0, int(raw.get("gold_earned", 0) or 0))
                except (TypeError, ValueError):
                    self._gold_earned = 0
                try:
                    self._monsters_killed = max(0, int(raw.get("monsters_killed", 0) or 0))
                except (TypeError, ValueError):
                    self._monsters_killed = 0
                lu = raw.get("last_updated")
                self._last_updated = str(lu) if lu else None
            except Exception as exc:
                logger.error("lifetime_stats yüklenemedi (%s): %s", self.path, exc)
                self._quarantine_corrupt_file()

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _atomic_write_locked(self) -> None:
        payload = _empty_payload()
        payload.update(
            {
                "materials_by_key": dict(self._materials_by_key),
                "gold_earned": int(self._gold_earned),
                "monsters_killed": int(self._monsters_killed),
                "last_updated": self._last_updated,
                "version": 1,
            }
        )
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_path = f"{self.path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self.path)
        except Exception as exc:
            logger.error("lifetime_stats yazılamadı (%s): %s", self.path, exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            mats = dict(self._materials_by_key)
            total_mats = int(sum(mats.values()))
            return {
                "materials_by_key": mats,
                "materials_total": total_mats,
                "gold_earned": int(self._gold_earned),
                "monsters_killed": int(self._monsters_killed),
                "last_updated": self._last_updated,
                "path": self.path,
                "corrupt_backup": self._corrupt_backup,
            }

    def record_material(self, material_key: str, count: int = 1) -> int:
        clean = str(material_key or "").strip()
        try:
            added = int(count)
        except (TypeError, ValueError):
            return 0
        if not clean or added <= 0:
            return 0
        with self._lock:
            new_total = int(self._materials_by_key.get(clean, 0)) + added
            self._materials_by_key[clean] = new_total
            self._last_updated = self._stamp()
            self._atomic_write_locked()
            return new_total

    def record_gold(self, amount: int, fingerprint: str = "") -> int:
        try:
            added = int(amount)
        except (TypeError, ValueError):
            return int(self._gold_earned)
        if added <= 0:
            return int(self._gold_earned)
        # F-0117: the old default fingerprint (amount + whole second) swallowed
        # distinct drops of the same amount inside one second. A caller-supplied
        # fingerprint is still deduplicated; an absent one must be unique.
        fp = str(fingerprint or "").strip() or (
            f"gold:{added}:{time.time_ns()}:{uuid.uuid4().hex[:12]}"
        )
        with self._lock:
            now = time.time()
            # Expire old fingerprints
            stale = [k for k, ts in self._recent_gold_fps.items() if (now - ts) > _GOLD_DEDUP_TTL_SEC]
            for k in stale:
                self._recent_gold_fps.pop(k, None)
            if fp in self._recent_gold_fps:
                return int(self._gold_earned)
            self._recent_gold_fps[fp] = now
            self._gold_earned = int(self._gold_earned) + added
            self._last_updated = self._stamp()
            self._atomic_write_locked()
            return int(self._gold_earned)

    def record_monster_kill(self, mob_key: str = "", *, force: bool = False) -> bool:
        """Increment kill counter. Debounced (~1.8s) per mob_key to avoid
        double-count across hooks; kills of different mobs are never merged."""
        with self._lock:
            now = time.time()
            key = str(mob_key or "")
            last = self._last_kill_ts_by_mob.get(key, 0.0)
            if not force and (now - last) < _KILL_DEBOUNCE_SEC:
                return False
            self._last_kill_ts_by_mob[key] = now
            self._last_kill_ts = now
            self._monsters_killed = int(self._monsters_killed) + 1
            self._last_updated = self._stamp()
            self._prune_kill_debounce_locked(now)
            self._atomic_write_locked()
            if mob_key:
                logger.debug("lifetime kill #%d (%s)", self._monsters_killed, mob_key)
            return True

    def _prune_kill_debounce_locked(self, now: float) -> None:
        """Bound _last_kill_ts_by_mob: drop entries older than the
        debounce window (they can never suppress a kill again); if the
        table is still over the cap, drop the oldest first. Only the
        entries inside the live 1.8 s window affect behaviour, so
        pruning never changes a debounce decision."""
        if len(self._last_kill_ts_by_mob) <= _KILL_DEBOUNCE_MAX_KEYS:
            return
        stale = [
            k
            for k, ts in self._last_kill_ts_by_mob.items()
            if (now - ts) >= _KILL_DEBOUNCE_SEC
        ]
        for k in stale:
            self._last_kill_ts_by_mob.pop(k, None)
        if len(self._last_kill_ts_by_mob) > _KILL_DEBOUNCE_MAX_KEYS:
            oldest = sorted(
                self._last_kill_ts_by_mob.items(), key=lambda kv: kv[1]
            )
            for k, _ts in oldest[: len(self._last_kill_ts_by_mob) - _KILL_DEBOUNCE_MAX_KEYS]:
                self._last_kill_ts_by_mob.pop(k, None)

    def reset(self, confirm: bool = False) -> None:
        """Wipe all lifetime counters. Only after explicit user confirmation:
        callers must pass confirm=True (the UI asks first); otherwise the
        wipe is refused instead of silently destroying lifetime data."""
        if not confirm:
            raise ValueError(
                "LifetimeStatsStore.reset() requires confirm=True "
                "(explicit user confirmation)"
            )
        with self._lock:
            self._materials_by_key.clear()
            self._gold_earned = 0
            self._monsters_killed = 0
            self._last_updated = self._stamp()
            self._last_kill_ts = 0.0
            self._last_kill_ts_by_mob.clear()
            self._recent_gold_fps.clear()
            self._atomic_write_locked()
            logger.info("Kalıcı lifetime istatistikleri kullanıcı onayıyla sıfırlandı.")


_STORE: Optional[LifetimeStatsStore] = None
_STORE_LOCK = threading.Lock()


def get_lifetime_stats(path: Optional[str] = None) -> LifetimeStatsStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = LifetimeStatsStore(path=path)
        elif path and os.path.abspath(path) != _STORE.path:
            # S304 (v19/P3-8): farklı yolla çağrı eskiden singleton'ı
            # sessizce değiştiriyordu — eski yolun bellekteki durumu
            # kaybolur gibi görünürdü. Davranış aynı (tek store),
            # ama değişim artık uyarıyla görünür.
            logger.warning(
                "get_lifetime_stats: singleton yolu değişiyor: %s -> %s",
                _STORE.path,
                os.path.abspath(path),
            )
            _STORE = LifetimeStatsStore(path=path)
        return _STORE


__all__ = [
    "LifetimeStatsStore",
    "get_lifetime_stats",
    "default_stats_path",
]
