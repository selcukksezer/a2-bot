# -*- coding: utf-8 -*-
"""
SkillDiscovery — quickslot etkisini HP-delta ile sınıflandırır (Zero-Knowledge).

Birincil: cast öncesi/sonrası state farkı.
İkincil: tooltip OCR (opsiyonel; üzerine yazmaz).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("Gatherion.SkillDiscovery")

# R/T mapping (2026-11-08): right-click attack = T, left-click attack = R.
# T is a basic-attack key, not a hotbar icon: it belongs in the probe list
# (DEFAULT_SLOTS) but NOT in COMBAT_ROW_SLOTS (HUD geometry has no T slot).
DEFAULT_SLOTS = ("1", "2", "3", "4", "5", "Q", "E", "R", "T")
CONSUMABLE_ROW_SLOTS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "X", "B")
COMBAT_ROW_SLOTS = ("1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R")
CENTER_EMBLEM_X_RANGE = (0.44, 0.56)
LEXICON_PATH = os.path.join("dataset", "skill_lexicon.json")
DB_PATH = os.path.join("dataset", "economy.sqlite")


def discover_hud_rows_layout(frame_bgr: Any) -> Dict[str, Any]:
    """
    Alt HUD'un üst tüketilebilir satırını (F1..F8, X, B: parşömen ve iksirler) ile
    alt savaş yetenek satırını (1..8, Q, E, R: savaş skilleri) otomatik ayırır ve
    merkezdeki seviye 45 amblemini (0.44 <= X <= 0.56) dışlar.
    """
    height, width = 1080, 1920
    has_frame = False
    if frame_bgr is not None and hasattr(frame_bgr, "shape"):
        try:
            shape = frame_bgr.shape
            if len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
                height = int(shape[0])
                width = int(shape[1])
                has_frame = True
        except Exception:
            has_frame = False

    emblem_x_min, emblem_x_max = CENTER_EMBLEM_X_RANGE
    upper_y_range = (0.875, 0.922)
    lower_y_range = (0.930, 0.978)
    slot_half_w = 0.014

    # Sol kanat (0.28..0.43) ve sağ kanat (0.57..0.72) — merkez amblem [0.44, 0.56] dışlanır.
    upper_left_slots = ("F1", "F2", "F3", "F4", "F5")
    upper_right_slots = ("F6", "F7", "F8", "X", "B")
    lower_left_slots = ("1", "2", "3", "4", "5")
    lower_right_slots = ("6", "7", "8", "Q", "E", "R")

    def _wing_positions(slots: Sequence[str], x_start: float, x_end: float) -> List[tuple[str, float]]:
        n = len(slots)
        if n <= 0:
            return []
        if n == 1:
            return [(str(slots[0]), (x_start + x_end) * 0.5)]
        step = (x_end - x_start) / float(n - 1)
        return [(str(s), x_start + i * step) for i, s in enumerate(slots)]

    def _build_row(
        left_slots: Sequence[str],
        right_slots: Sequence[str],
        y_range: tuple[float, float],
        row_role: str,
    ) -> Dict[str, Dict[str, Any]]:
        y1_n, y2_n = y_range
        cy_n = (y1_n + y2_n) * 0.5
        pairs = _wing_positions(left_slots, 0.295, 0.425) + _wing_positions(right_slots, 0.575, 0.715)
        row_map: Dict[str, Dict[str, Any]] = {}
        for slot_key, cx_n in pairs:
            # Merkez seviye 45 amblemini kesin olarak dışla (0.44 <= X <= 0.56)
            if emblem_x_min <= cx_n <= emblem_x_max:
                continue
            x1_n = max(0.0, cx_n - slot_half_w)
            x2_n = min(1.0, cx_n + slot_half_w)
            if x1_n < emblem_x_max and x2_n > emblem_x_min:
                if cx_n < (emblem_x_min + emblem_x_max) * 0.5:
                    x2_n = min(x2_n, emblem_x_min - 0.001)
                else:
                    x1_n = max(x1_n, emblem_x_max + 0.001)

            px1 = max(0, min(width - 1, int(round(x1_n * width))))
            px2 = max(px1 + 1, min(width, int(round(x2_n * width))))
            py1 = max(0, min(height - 1, int(round(y1_n * height))))
            py2 = max(py1 + 1, min(height, int(round(y2_n * height))))

            mean_luma = 0.0
            # No frame → no measurement: never report the slot as occupied.
            occupied = False
            if has_frame:
                try:
                    patch = frame_bgr[py1:py2, px1:px2]
                    if patch is not None and getattr(patch, "size", 0) > 0:
                        mean_luma = float(patch.mean())
                        occupied = bool(mean_luma > 18.0)
                except Exception:
                    pass

            row_map[slot_key] = {
                "slot": slot_key,
                "role": row_role,
                "wing": "left" if cx_n < emblem_x_min else "right",
                "center_norm": (round(cx_n, 4), round(cy_n, 4)),
                "bbox_norm": (round(x1_n, 4), round(y1_n, 4), round(x2_n, 4), round(y2_n, 4)),
                "bbox_px": (px1, py1, px2, py2),
                "mean_luma": round(mean_luma, 2),
                "occupied": occupied,
            }
        return row_map

    consumable_row = _build_row(
        upper_left_slots,
        upper_right_slots,
        upper_y_range,
        "consumable_scroll_potion",
    )
    combat_row = _build_row(
        lower_left_slots,
        lower_right_slots,
        lower_y_range,
        "combat_skill",
    )

    return {
        "frame_size": (width, height),
        "frame_present": has_frame,
        "excluded_center_emblem": {
            "label": "level_45_emblem",
            "x_range_norm": CENTER_EMBLEM_X_RANGE,
            "y_range_norm": (upper_y_range[0], lower_y_range[1]),
            "x_range_px": (int(round(emblem_x_min * width)), int(round(emblem_x_max * width))),
        },
        "upper_consumable_row": {
            "keys": list(CONSUMABLE_ROW_SLOTS),
            "category": "scrolls_and_potions",
            "y_range_norm": upper_y_range,
            "slots": consumable_row,
        },
        "lower_combat_row": {
            "keys": list(COMBAT_ROW_SLOTS),
            "category": "combat_skills",
            "y_range_norm": lower_y_range,
            "slots": combat_row,
        },
    }


@dataclass
class SkillProbeResult:
    slot: str
    label: str
    delta_monster_hp: float = 0.0
    delta_player_hp: float = 0.0
    delta_buffs: int = 0
    confidence: float = 0.0


@dataclass
class DiscoveryState:
    pending_slots: List[str] = field(default_factory=list)
    classified: Dict[str, str] = field(default_factory=dict)
    active: bool = False
    last_probe_ts: float = 0.0


class SkillDiscovery:
    def __init__(
        self,
        lexicon_path: str = LEXICON_PATH,
        db_path: str = DB_PATH,
        on_log: Optional[Callable[[str], None]] = None,
        probe_gap_sec: float = 2.2,
    ) -> None:
        self.lexicon_path = lexicon_path
        self.db_path = db_path
        self._on_log = on_log
        self.probe_gap_sec = float(probe_gap_sec)
        self._lock = threading.RLock()
        self.state = DiscoveryState()
        self._lexicon: Dict[str, str] = {}
        self._ensure_db()
        self._load_lexicon()

    def _emit(self, msg: str) -> None:
        logger.info(msg)
        if callable(self._on_log):
            try:
                self._on_log(msg)
            except Exception:
                pass

    def _ensure_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS skill_lexicon (
                        slot TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        confidence REAL DEFAULT 0,
                        updated_ts REAL
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.debug("skill_lexicon db init: %s", exc)

    def _load_lexicon(self) -> None:
        loaded: Dict[str, str] = {}
        if os.path.isfile(self.lexicon_path):
            try:
                with open(self.lexicon_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    loaded = {str(k).lower(): str(v).lower() for k, v in data.items()}
            except Exception as exc:
                logger.debug("lexicon json load: %s", exc)
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT slot, label FROM skill_lexicon").fetchall()
                for slot, label in rows:
                    loaded[str(slot).lower()] = str(label).lower()
        except Exception:
            pass
        with self._lock:
            self._lexicon = loaded
            self.state.classified = dict(loaded)

    def lexicon(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._lexicon)

    def start(self, slots: Optional[Sequence[str]] = None) -> None:
        keys = [str(s) for s in (slots or DEFAULT_SLOTS)]
        with self._lock:
            known = set(self._lexicon.keys())
            pending = [k for k in keys if k.lower() not in known]
            self.state.pending_slots = pending
            self.state.active = bool(pending)
        if pending:
            self._emit(f"DISCOVER start pending={pending}")
        else:
            self._emit("DISCOVER skip (lexicon complete)")

    def needs_discovery(self) -> bool:
        with self._lock:
            return self.state.active and bool(self.state.pending_slots)

    def classify_delta(
        self,
        slot: str,
        monster_hp_before: float,
        monster_hp_after: float,
        player_hp_before: float,
        player_hp_after: float,
        buffs_before: int = 0,
        buffs_after: int = 0,
        mana_before: Optional[float] = None,
        mana_after: Optional[float] = None,
    ) -> SkillProbeResult:
        d_m = float(monster_hp_before) - float(monster_hp_after)
        d_p = float(player_hp_after) - float(player_hp_before)
        d_b = int(buffs_after) - int(buffs_before)
        # "Low mana" needs an actual mana measurement — never infer it from
        # low damage alone.
        mana_low = False
        if mana_before is not None and mana_after is not None:
            try:
                mana_low = float(mana_after) <= 0.01
            except (TypeError, ValueError):
                mana_low = False
        label = "unknown"
        conf = 0.3
        if d_b >= 1 and d_m < 0.05:
            label = "scroll_buff"
            conf = 0.7 + min(0.25, 0.1 * d_b)
        elif d_m >= 0.02:
            # Damage was dealt, so this is an attack even if the player also
            # healed (lifesteal) — the old order labelled that defense_heal.
            label = "attack"
            conf = 0.6 + min(0.35, d_m)
        elif d_p >= 0.03:
            label = "defense_heal"
            conf = 0.65 + min(0.3, d_p)
        elif d_m > 0.005:
            # Weak hit with no mana data: plain attack, never "low mana".
            label = "attack"
            conf = 0.45
        elif mana_low:
            label = "attack_low_mana"
            conf = 0.5
        result = SkillProbeResult(
            slot=str(slot),
            label=label,
            delta_monster_hp=d_m,
            delta_player_hp=d_p,
            delta_buffs=d_b,
            confidence=conf,
        )
        self._commit(result)
        self._emit(f"DISCOVER {label} slot={slot} dm={d_m:.3f} dp={d_p:.3f}")
        return result

    def _commit(self, result: SkillProbeResult) -> None:
        key = str(result.slot).lower()
        with self._lock:
            prev = self._lexicon.get(key)
            # Do not overwrite a stronger prior label with unknown.
            if result.label == "unknown" and prev:
                return
            self._lexicon[key] = result.label
            self.state.classified[key] = result.label
            if key in [p.lower() for p in self.state.pending_slots]:
                self.state.pending_slots = [
                    p for p in self.state.pending_slots if p.lower() != key
                ]
            if not self.state.pending_slots:
                self.state.active = False
        self._persist(result)

    def _persist(self, result: SkillProbeResult) -> None:
        try:
            os.makedirs(os.path.dirname(self.lexicon_path) or ".", exist_ok=True)
            with self._lock:
                payload = dict(self._lexicon)
            with open(self.lexicon_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.debug("lexicon json save: %s", exc)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO skill_lexicon(slot, label, confidence, updated_ts)
                    VALUES(?,?,?,?)
                    ON CONFLICT(slot) DO UPDATE SET
                        label=excluded.label,
                        confidence=excluded.confidence,
                        updated_ts=excluded.updated_ts
                    """,
                    (str(result.slot).lower(), result.label, float(result.confidence), time.time()),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("lexicon db save: %s", exc)

    def next_slot(self) -> Optional[str]:
        with self._lock:
            if not self.state.pending_slots:
                return None
            now = time.time()
            if now - self.state.last_probe_ts < self.probe_gap_sec:
                return None
            self.state.last_probe_ts = now
            return self.state.pending_slots[0]

    def confirm_tooltip_hint(self, slot: str, tooltip_text: str) -> Optional[str]:
        """Secondary OCR/tooltip hint — never overwrites a dynamic label."""
        text = str(tooltip_text or "").lower()
        hint = None
        # "attack speed" contains "attack" — check buff phrasing before the
        # generic attack branch, or every attack-speed/physical-attack buff
        # is misclassified as an attack (S-1303 / D24). The later buff branch's
        # "physical attack" key was dead code: "increases physical attack"
        # fell into the attack branch first. Buff context words
        # ("increases"/"scroll"/"buff") distinguish a buff scroll
        # ("Increases physical attack") from an attack skill
        # ("Deals physical attack damage"), which still hits the attack
        # branch below. Note: a bare "physical attack" without buff context
        # must NOT be forced to scroll_buff — it is an attack.
        if "attack speed" in text:
            hint = "scroll_buff"
        elif "physical attack" in text and any(
            w in text for w in ("increases", "increase", "scroll", "buff")
        ):
            hint = "scroll_buff"
        elif any(w in text for w in ("damage", "hasar", "deals", "strike", "attack")):
            hint = "attack"
        elif any(w in text for w in ("restore", "heal", "shield", "removes stun", "defiance")):
            hint = "defense_heal"
        elif any(w in text for w in ("increases", "scroll", "buff")):
            hint = "scroll_buff"
        if hint is None:
            return None
        key = str(slot).lower()
        with self._lock:
            if key in self._lexicon:
                return self._lexicon[key]
            self._lexicon[key] = hint
            self.state.classified[key] = hint
        self._persist(
            SkillProbeResult(slot=slot, label=hint, confidence=0.45)
        )
        self._emit(f"DISCOVER tooltip {hint} slot={slot}")
        return hint

    def skills_by_label(self, label: str) -> List[str]:
        lab = str(label).lower()
        with self._lock:
            return [k for k, v in self._lexicon.items() if v == lab]

    def discover_hud_rows_layout(self, frame_bgr: Any) -> Dict[str, Any]:
        return discover_hud_rows_layout(frame_bgr)
