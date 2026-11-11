# -*- coding: utf-8 -*-
"""
Active Learning & Hard-Example Mining Loop + Passive Shadow Gameplay Recorder.

Pushes the custom Aion 2 YOLO resource detector toward 99.9% accuracy by:
1. Uncertainty & Hard-Example Mining (`evaluate_frame_for_active_learning`):
   - Captures borderline confidence detections (`0.42 <= confidence <= 0.74`)
     ("Model kararsız kaldı, eğitim setine ekle!").
   - Captures false negatives when the user manually presses `F8` or `F`/`E` on a
     resource node that the scanner missed (`false_negative_mining`).
   - Captures hard negatives when `"Essence Extraction failed"` occurs or a
     false-positive monster/feather rejection happens near a resource node
     (`hard_negative_mining`).
2. Class Rarity Prioritizer (`compute_sample_priority`):
   - Assigns priority in `[1.0, 3.0]` to balance all 12 YOLO material classes,
     heavily prioritizing rare/underrepresented classes (e.g. `gem_crystal` with
     30 samples over `herb_aria` with 500 samples) for storage and augmentation.
3. Passive Shadow Recording Mode (`ShadowGameplayRecorder`):
   - `"Pasif Gölge Kayıt Modu"`: Zero key presses or mouse inputs from the bot.
   - Silently observes normal user gameplay, detects manual harvests or `F8`
     triggers, pulls the clean `t - 2.0s` intact-node frame from the
     `PreGatherTimeMachineBuffer`, and auto-labels it in the background.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from config.materials_db import (
    AION2_MATERIALS_CATALOG,
    classify_3d_material_patch,
    is_rejected_non_material,
)
from dataset_factory.frame_time_machine import FrameSnapshot, PreGatherTimeMachineBuffer

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical 12-Class YOLO Material Taxonomy & Alias Mapping
# =============================================================================

# The 12 material classes, in CANONICAL taxonomy order (ids 0..11 of
# dataset_factory.class_taxonomy.CANONICAL_NAMES). The previous list used a
# private 12-class order (herb_targena at 1, herb_azpha at 2, fruit_mela at
# 4, gem_sapphire at 7, ...) that disagreed with the canonical ids, so the
# same class_id meant different materials in different files (F-0781).
CANONICAL_12_YOLO_CLASSES: Tuple[str, str, str, str, str, str, str, str, str, str, str, str] = (
    "herb_aria",
    "herb_mela",
    "herb_targena",
    "ore_orichalcum",
    "gem_crystal",
    "tree_log",
    "herb_edana",
    "herb_lumina",
    "ore_obsidian",
    "minimap_sprout",
    "monster_3d",
    "minimap_monster_dot",
)

CLASS_ID_TO_NAME: Dict[int, str] = {
    idx: name for idx, name in enumerate(CANONICAL_12_YOLO_CLASSES)
}

CLASS_NAME_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(CANONICAL_12_YOLO_CLASSES)
}

MATERIAL_ALIAS_TO_CANONICAL_CLASS: Dict[str, str] = {
    # Canonical names (ids 0..11 of the canonical taxonomy)
    "herb_aria": "herb_aria",
    "herb_mela": "herb_mela",
    "herb_targena": "herb_targena",
    "ore_orichalcum": "ore_orichalcum",
    "gem_crystal": "gem_crystal",
    "tree_log": "tree_log",
    "herb_edana": "herb_edana",
    "herb_lumina": "herb_lumina",
    "ore_obsidian": "ore_obsidian",
    "minimap_sprout": "minimap_sprout",
    "monster_3d": "monster_3d",
    "minimap_monster_dot": "minimap_monster_dot",
    # Legacy material names (old 12-class scheme) -> canonical names
    # P1-14 (F4): "ore_iron_silver" removed — an iron/silver ore is NOT
    # orichalcum (21 ores are not orichalcum); it must stay unmapped.
    "herb_azpha": "herb_targena",
    "herb_calendula": "herb_targena",
    "fruit_mela": "herb_mela",
    "gem_sapphire": "gem_crystal",
    "gem_ruby": "gem_crystal",
    "gem_diamond": "gem_crystal",
    "log_timber": "tree_log",
    # Direct material names from AION2_MATERIALS_CATALOG.
    # NOTE (S-5501): every value here MUST be a canonical class in
    # CANONICAL_12_YOLO_CLASSES. The previous version mapped these direct
    # material names to legacy intermediate keys (herb_azpha, fruit_mela,
    # ore_iron_silver, gem_sapphire, gem_ruby, gem_diamond, log_timber),
    # which are NOT canonical: class_counts then grew a 13th+ key and the
    # health check (set(class_counts) == set(CANONICAL_12_YOLO_CLASSES))
    # failed permanently. Map through the legacy row above instead.
    "aria": "herb_aria",
    "fresh aria": "herb_aria",
    "taze aria": "herb_aria",
    "targena": "herb_targena",
    "fresh targena": "herb_targena",
    "targena otu": "herb_targena",
    "azpha": "herb_targena",
    "fresh azpha": "herb_targena",
    "azpha otu": "herb_targena",
    "calendula": "herb_targena",
    "fresh calendula": "herb_targena",
    "mela": "herb_mela",
    "fresh mela": "herb_mela",
    "taze mela": "herb_mela",
    "orichalcum": "ore_orichalcum",
    "orichalcum ore": "ore_orichalcum",
    "fine orichalcum ore": "ore_orichalcum",
    "pure orichalcum ore": "ore_orichalcum",
    "radiant orichalcum ore": "ore_orichalcum",
    # P1-14 (F4): "iron ore" / "silver ore" / "copper ore" / "gold ore" /
    # "titanium ore" REMOVED — 21 ores are not orichalcum. They resolve
    # to no class now (unknown), never to a false orichalcum.
    "sapphire": "gem_crystal",
    "sapphire gemstone": "gem_crystal",
    "fine sapphire gemstone": "gem_crystal",
    "pure sapphire gemstone": "gem_crystal",
    "radiant sapphire gemstone": "gem_crystal",
    "aquamarine": "gem_crystal",
    "ruby": "gem_crystal",
    "ruby gemstone": "gem_crystal",
    "fine ruby gemstone": "gem_crystal",
    "pure ruby gemstone": "gem_crystal",
    "radiant ruby gemstone": "gem_crystal",
    "garnet": "gem_crystal",
    "diamond": "gem_crystal",
    "diamond gemstone": "gem_crystal",
    "fine diamond gemstone": "gem_crystal",
    "pure diamond gemstone": "gem_crystal",
    "radiant diamond gemstone": "gem_crystal",
    "crystal": "gem_crystal",
    "kristal": "gem_crystal",
    "rose quartz": "gem_crystal",
    "amethyst": "gem_crystal",
    "topaz": "gem_crystal",
    "emerald": "gem_crystal",
    "opal": "gem_crystal",
    "yggdrasil log": "tree_log",
    "asvata log": "tree_log",
    "pine log": "tree_log",
    "oak log": "tree_log",
    "maple log": "tree_log",
    "cedar log": "tree_log",
}

# Fast path for the per-detection non-material rejection scan in
# `ActiveLearningLoop.evaluate_frame_for_active_learning`:
# `is_rejected_non_material` runs ~34 regex searches per label and
# dominated evaluate_frame (~80% of its wall time with 50 detections).
# Every alias key resolving to one of the 9 gatherable material classes
# is verified (tests/test_d38_active_learning_loop.py) to return False
# from that scan, so those labels skip it. Anything else -- including
# the canonical monster classes `monster_3d` / `minimap_monster_dot`,
# which the scan DOES reject -- still goes through the full scan, so
# behaviour is unchanged.
GATHERABLE_CLASS_NAMES: frozenset = frozenset(
    (
        "herb_aria",
        "herb_mela",
        "herb_targena",
        "ore_orichalcum",
        "gem_crystal",
        "tree_log",
        "herb_edana",
        "herb_lumina",
        "ore_obsidian",
    )
)

KNOWN_GATHERABLE_LABELS: frozenset = frozenset(
    key for key, canon in MATERIAL_ALIAS_TO_CANONICAL_CLASS.items() if canon in GATHERABLE_CLASS_NAMES
) | GATHERABLE_CLASS_NAMES


@lru_cache(maxsize=512)
def _is_rejected_non_material_cached(label: str) -> bool:
    """Memoized rejection scan for labels outside the gatherable fast
    path: detector labels repeat identically frame after frame, and the
    scan is pure over its string argument, so a bounded cache turns the
    per-frame ~34-pattern regex scan into a dict lookup."""
    return is_rejected_non_material(label)

EXTRACTION_FAILURE_PATTERNS: Sequence[str] = (
    "essence extraction failed",
    "vitality extraction failed",
    "extraction failed",
    "gathering failed",
    "harvest failed",
    "toplama başarısız",
    "çıkarma başarısız",
    "invalid target",
    "target is too far",
    "cannot gather",
)


def normalize_material_class_key(raw_key: Union[int, str, None]) -> str:
    """
    Resolve an integer class ID, YOLO class name, or Aion 2 material name to a
    canonical class key. Preserves custom class keys if not in default aliases.

    Unknown / missing input returns "" (unknown) — it NEVER fabricates a
    default class: previously None/"" silently became "herb_aria", so a
    class-less (or empty) dict was labelled as an herb_aria node that was
    never seen (S-2001). The same applies to integer ids outside the
    canonical 0..11 range (legacy UI/cognition ids and unknown ids):
    previously they became a fabricated "class_<id>" key, which polluted
    class_counts with non-canonical keys and permanently broke the health
    check (S-5501). Callers must treat "" as "no class evidence".
    """
    if raw_key is None:
        return ""
    if isinstance(raw_key, (int, np.integer)):
        cid = int(raw_key)
        # P1-14 (K-36): NO legacy translation. A bare id is canonical by
        # definition (6 = herb_edana, not a gather crest); ids with no
        # canonical 12-class name are unknown, not "class_<id>".
        return CLASS_ID_TO_NAME.get(cid, "")
    text = str(raw_key).strip()
    if not text:
        return ""
    if text.isdigit():
        cid = int(text)
        # P1-14 (K-36): see above — no legacy translation for digit strings.
        return CLASS_ID_TO_NAME.get(cid, "")
    lowered = text.lower()
    if lowered in MATERIAL_ALIAS_TO_CANONICAL_CLASS:
        return MATERIAL_ALIAS_TO_CANONICAL_CLASS[lowered]

    # Category-level heuristic fallback (canonical names only)
    # P1-14 (F4): bare "berry"/"fruit" is not mela, and any "ore"/"iron"/
    # "silver"/"metal" is not orichalcum — those branches are removed.
    # Generic gemstones -> 4 and logs/trees -> 5 stay: they are the F4
    # approved generic classes, not specific-material claims.
    # P1-8 (K-21): token-based on purpose — substring matching minted
    # wrong classes out of unrelated words ("dialog" contains "log", so
    # it became tree_log; "barbarian" contains "aria", so it became
    # herb_aria). Only whole underscore/space/punctuation-separated
    # tokens count, so "dialog" / "barbarian" fall through to unknown.
    tokens = frozenset(
        "".join(ch if ch.isalnum() else " " for ch in lowered).split()
    )
    if "aria" in tokens:
        return "herb_aria"
    if tokens & {"targena", "azpha", "calendula"}:
        return "herb_targena"
    if "mela" in tokens:
        return "herb_mela"
    if "orichalcum" in tokens:
        return "ore_orichalcum"
    if "edana" in tokens:
        return "herb_edana"
    if "lumina" in tokens:
        return "herb_lumina"
    if "obsidian" in tokens:
        return "ore_obsidian"
    if tokens & {"crystal", "quartz", "amethyst", "gem"}:
        return "gem_crystal"
    if tokens & {"log", "tree", "timber", "wood"}:
        return "tree_log"
    return lowered


def resolve_class_id(raw_key: Union[int, str, None]) -> int:
    """Resolve an annotation class identifier or material name to a canonical train class id.

    P1-14 (K-36): integer ids are canonical as-is — NO legacy translation.
    A bare id is canonical by definition (6 = herb_edana, 18 = target_hp_bar);
    an id outside the canonical 0..26 range is unknown (-1).

    Unknown / missing input returns -1 (unknown class) — it NEVER fabricates
    class 0 ("herb_aria"): previously empty input and unresolvable names
    silently became herb_aria, labelling unseen objects as a real node
    (S-2001). Callers must skip annotations with class_id < 0.
    """
    if isinstance(raw_key, (int, np.integer)) and not isinstance(raw_key, bool):
        cid = int(raw_key)
        if cid < 0 or cid > 26:
            return -1
        return cid
    try:
        from dataset_factory.class_taxonomy import class_id_for_material_text

        specific = class_id_for_material_text(str(raw_key or ""))
    except Exception:
        specific = None
    if specific is not None:
        return int(specific)
    canonical = normalize_material_class_key(raw_key)
    if not canonical:
        return -1
    return CLASS_NAME_TO_ID.get(canonical, -1)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HardExampleRecord:
    """Structured record of a mined hard/uncertain/shadow-recorded training sample."""

    timestamp: float
    mining_types: List[str]
    primary_mining_type: str
    reason: str
    priority: float
    augmentation_multiplier: int
    frame_shape: Sequence[int]
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    uncertain_detections: List[Dict[str, Any]] = field(default_factory=list)
    rejected_detections: List[Dict[str, Any]] = field(default_factory=list)
    is_background_negative: bool = False
    region_name: str = ""
    player_pos: Tuple[float, float] = (0.0, 0.0)
    lookback_sec: float = 0.0
    frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_frame: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": float(self.timestamp),
            "mining_types": list(self.mining_types),
            "primary_mining_type": str(self.primary_mining_type),
            "reason": str(self.reason),
            "priority": float(self.priority),
            "augmentation_multiplier": int(self.augmentation_multiplier),
            "frame_shape": tuple(self.frame_shape),
            "annotations": [dict(a) for a in self.annotations],
            "uncertain_detections": [dict(d) for d in self.uncertain_detections],
            "rejected_detections": [dict(d) for d in self.rejected_detections],
            "is_background_negative": bool(self.is_background_negative),
            "region_name": str(self.region_name),
            "player_pos": (float(self.player_pos[0]), float(self.player_pos[1])),
            "lookback_sec": float(self.lookback_sec),
            "metadata": dict(self.metadata),
        }
        if include_frame and self.frame is not None:
            payload["frame"] = self.frame
        return payload


# =============================================================================
# 1 & 2. Active Learning & Hard-Example Mining Loop + Class Rarity Prioritizer
# =============================================================================


class ActiveLearningLoop:
    """
    Active Learning & Hard-Example Mining Engine for the Aion 2 YOLO Dataset Factory.

    Responsibilities:
    1. Uncertainty & Hard-Example Detection (`evaluate_frame_for_active_learning`):
       - Borderline confidence mining (`0.42 <= confidence <= 0.74`):
         `"Model kararsız kaldı, eğitim setine ekle!"`
       - False negative mining (`manual_f8_pressed` or manual `F` gather when
         detector missed the resource node): `"false_negative_mining"`
       - Hard negative mining (`extraction_failed` or false-positive monster/feather
         rejection near a resource node): `"hard_negative_mining"`
    2. Class Rarity Prioritizer (`compute_sample_priority`):
       - Computes a sample priority score in `[1.0, 3.0]` based on current dataset
         class distribution so rare classes (e.g., `gem_crystal` with 30 samples vs
         `herb_aria` with 500 samples) receive maximum save & augmentation weight.
    """

    UNCERTAINTY_MESSAGE_TR: str = "Model kararsız kaldı, eğitim setine ekle!"
    FALSE_NEGATIVE_MESSAGE_TR: str = (
        "Kullanıcı F8/F ile materyal topladı fakat tarayıcı kaçırdı (False Negative Mining)!"
    )
    HARD_NEGATIVE_MESSAGE_TR: str = (
        "Essence Extraction başarısız veya sahte pozitif canavar/tüy reddi (Hard Negative Mining)!"
    )

    def __init__(
        self,
        uncertainty_min_conf: float = 0.42,
        uncertainty_max_conf: float = 0.74,
        min_priority: float = 1.0,
        max_priority: float = 3.0,
        initial_class_counts: Optional[Dict[Union[int, str], int]] = None,
        max_history: int = 500,
        on_hard_example_callback: Optional[Callable[[HardExampleRecord], None]] = None,
    ) -> None:
        self.uncertainty_min_conf: float = float(np.clip(uncertainty_min_conf, 0.05, 0.90))
        self.uncertainty_max_conf: float = float(
            np.clip(uncertainty_max_conf, self.uncertainty_min_conf + 0.01, 0.98)
        )
        self.min_priority: float = float(max(0.1, min_priority))
        self.max_priority: float = float(max(self.min_priority + 0.1, max_priority))
        self.max_history: int = max(16, int(max_history))
        self.on_hard_example_callback = on_hard_example_callback

        self._lock = threading.RLock()
        self.class_counts: Dict[str, int] = {name: 0 for name in CANONICAL_12_YOLO_CLASSES}
        if initial_class_counts:
            self.update_class_counts(initial_class_counts)

        self._mined_history: Deque[HardExampleRecord] = deque(maxlen=self.max_history)
        self.stats: Dict[str, int] = {
            "frames_evaluated": 0,
            "total_triggered": 0,
            "uncertainty_mining_count": 0,
            "false_negative_mining_count": 0,
            "hard_negative_mining_count": 0,
            "background_negatives_mined": 0,
            # Telemetry for previously silent failure/drop paths (A08): these
            # only count events, they never change a mining decision.
            "callback_errors": 0,
            "sub_threshold_detections": 0,
        }

    @staticmethod
    def _thumbnail_frame(frame: Optional[np.ndarray], max_dim: int = 320) -> Optional[np.ndarray]:
        """Return a downscaled copy of ``frame`` for the mined-history buffer.

        Storing a full-resolution copy per mined record (500-record deque)
        cost gigabytes of RAM; a <=320px thumbnail keeps the visual evidence
        while bounding the buffer to tens of MB. ``frame_shape`` on the record
        still holds the original dimensions.
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        try:
            h, w = int(frame.shape[0]), int(frame.shape[1])
        except Exception:
            return None
        if max(h, w) <= max_dim:
            return frame.copy()
        scale = float(max_dim) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame.copy()

    def get_active_learning_health_summary(self) -> dict:
        """Real health check: uncertainty band, priority range and class-count
        keys must be sane (previously this unconditionally returned True)."""
        try:
            healthy = bool(
                0.0 < self.uncertainty_min_conf < self.uncertainty_max_conf < 1.0
                and 0.0 < self.min_priority <= self.max_priority
                and set(self.class_counts) == set(CANONICAL_12_YOLO_CLASSES)
            )
            return {"active_learning_ready": healthy}
        except Exception:
            return {"active_learning_ready": False}

    # -------------------------------------------------------------------------
    # Class Counts Management & Rarity Prioritizer
    # -------------------------------------------------------------------------

    def update_class_counts(self, counts: Dict[Union[int, str], int], replace: bool = False) -> Dict[str, int]:
        """Update internal class sample counts dictionary in a thread-safe manner."""
        with self._lock:
            if replace:
                self.class_counts = {name: 0 for name in CANONICAL_12_YOLO_CLASSES}
            for raw_k, val in counts.items():
                key = normalize_material_class_key(raw_k)
                if not key:
                    # Unknown class: do not create a "" bucket in the counts.
                    continue
                if key not in CANONICAL_12_YOLO_CLASSES:
                    # S-5501: never let a non-canonical key (legacy name or
                    # custom key from the normalize fallback) into
                    # class_counts — it breaks the health check, which
                    # requires set(class_counts) == set(CANONICAL_12...).
                    continue
                self.class_counts[key] = max(0, int(val))
            return dict(self.class_counts)

    def increment_class_count(self, class_key: Union[int, str], delta: int = 1) -> int:
        """Increment sample count for a specific material class."""
        with self._lock:
            key = normalize_material_class_key(class_key)
            if not key:
                # Unknown class: nothing to count; never fabricate herb_aria.
                return 0
            if key not in CANONICAL_12_YOLO_CLASSES:
                # S-5501: non-canonical keys must not grow class_counts.
                return 0
            self.class_counts[key] = max(0, self.class_counts.get(key, 0) + int(delta))
            return self.class_counts[key]

    @staticmethod
    def _extract_annotation_class_keys(annotations: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
        """Extract canonical class keys from a list of detection/annotation dicts."""
        if not annotations:
            return []
        keys: List[str] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            raw_cls = (
                ann.get("class_name")
                or ann.get("material_name")
                or ann.get("name")
                or ann.get("label")
                or ann.get("category")
            )
            if raw_cls is None and "class_id" in ann:
                raw_cls = ann["class_id"]
            if raw_cls is None and "cls" in ann:
                raw_cls = ann["cls"]
            keys.append(normalize_material_class_key(raw_cls))
        return keys

    def compute_sample_priority(
        self,
        annotations: Optional[Sequence[Dict[str, Any]]],
        class_counts: Optional[Dict[Union[int, str], int]] = None,
    ) -> float:
        """
        Assigns higher priority (`1.0 .. 3.0`) to underrepresented material classes.

        Example:
            If `herb_aria` has 500 samples and `gem_crystal` has 30 samples,
            a `gem_crystal` sample receives a priority near `2.85 - 3.0` whereas
            an abundant `herb_aria` sample receives `~1.0`.

        Args:
            annotations: Sequence of annotation/detection dicts for the candidate frame.
            class_counts: Mapping of class names or class IDs to current sample counts.
                          Falls back to `self.class_counts` if omitted.

        Returns:
            Priority score strictly clamped to `[1.0, 3.0]`.
        """
        with self._lock:
            raw_counts = class_counts if class_counts is not None else self.class_counts
            normalized_counts: Dict[str, int] = {}
            for k, v in (raw_counts or {}).items():
                norm_k = normalize_material_class_key(k)
                if not norm_k:
                    continue
                # S-5501: only the normalized (canonical) key is kept. The
                # old code also stored the raw string key, so a legacy key
                # such as "gem_sapphire" and its canonical "gem_crystal"
                # were counted as two separate classes in the rarity math.
                normalized_counts[norm_k] = normalized_counts.get(norm_k, 0) + max(0, int(v))

        if not normalized_counts:
            return float(np.clip(2.0, self.min_priority, self.max_priority))

        positive_counts = [c for c in normalized_counts.values() if c >= 0]
        max_count = max(positive_counts) if positive_counts else 0
        if max_count <= 0:
            # Dataset is completely empty: every positive class has maximum priority
            return float(self.max_priority if annotations else 1.5)

        if not annotations:
            # Hard-negative / background frame: moderate priority
            return float(np.clip(1.4, self.min_priority, self.max_priority))

        # One shared extraction path: the class key for each annotation
        # comes from _extract_annotation_class_keys (previously this was
        # duplicated inline here and missed the "cls"/"category" keys the
        # helper honours, so e.g. {"cls": 4} was scored as herb_aria).
        # normalized_counts is keyed by canonical name, so the canonical
        # key is the only lookup needed.
        dict_annotations = [ann for ann in annotations if isinstance(ann, dict)]
        class_keys = self._extract_annotation_class_keys(dict_annotations)
        class_priorities: List[float] = []
        for ann, norm_cls in zip(dict_annotations, class_keys):
            if not norm_cls:
                # No class evidence: this annotation must not inflate
                # priority as a "rare unseen class" (S-2001).
                continue
            # HEAD's raw-string fallback is subsumed here: normalized_counts
            # is keyed only by canonical names (S-5501), so the canonical
            # key is the only lookup needed.
            cls_count = normalized_counts.get(norm_cls, 0)

            # Scarcity fraction: 0.0 when cls_count == max_count, 1.0 when cls_count == 0
            scarcity_linear = float(max(0, max_count - cls_count)) / float(max(1, max_count))
            # Non-linear concave boost so severely underrepresented classes (e.g. 30 vs 500)
            # strongly approach 3.0 while well-represented classes stay near 1.0.
            imbalance_factor = math.pow(scarcity_linear, 0.65)
            raw_prio = self.min_priority + (self.max_priority - self.min_priority) * imbalance_factor

            # Extra boost if the annotation itself was borderline confidence
            conf = ann.get("confidence", ann.get("conf"))
            if conf is not None:
                try:
                    c_val = float(conf)
                    if self.uncertainty_min_conf <= c_val <= self.uncertainty_max_conf:
                        raw_prio = min(self.max_priority, raw_prio + 0.15)
                except (TypeError, ValueError):
                    c_val = -1.0

            class_priorities.append(float(np.clip(raw_prio, self.min_priority, self.max_priority)))

        if not class_priorities:
            return float(self.min_priority)

        highest_priority = max(class_priorities)
        # Small bonus if frame contains multiple rare resource nodes
        multi_node_bonus = min(0.10, 0.03 * max(0, len(class_priorities) - 1))
        final_priority = float(
            np.clip(highest_priority + multi_node_bonus, self.min_priority, self.max_priority)
        )
        return round(final_priority, 4)

    # -------------------------------------------------------------------------
    # Uncertainty & Hard-Example Detector
    # -------------------------------------------------------------------------

    def evaluate_frame_for_active_learning(
        self,
        frame: Optional[np.ndarray],
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        extraction_failed: bool = False,
        manual_f8_pressed: bool = False,
        *,
        manual_f_pressed: bool = False,
        monster_rejected: bool = False,
        status_text: str = "",
        class_counts: Optional[Dict[Union[int, str], int]] = None,
        region_name: str = "",
        player_pos: Tuple[float, float] = (0.0, 0.0),
        lookback_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate a gameplay frame and its detector outputs to decide whether to capture
        it for Active Learning & Hard-Example Mining.

        Triggers capture when ANY of the following conditions hold:
        1. `uncertainty_mining`:
           At least one detection has borderline confidence
           (`0.42 <= confidence <= 0.74`) -> `"Model kararsız kaldı, eğitim setine ekle!"`
        2. `false_negative_mining`:
           The user manually pressed `F8` (or `F` on a gatherable node) while the
           detector produced zero confident resource detections.
        3. `hard_negative_mining`:
           `"Essence Extraction failed"` occurred (`extraction_failed=True` or matching
           `status_text`), OR a false-positive monster/feather/NPC rejection occurred
           near a candidate detection.

        Returns:
            Dictionary describing the active-learning decision, mining categories,
            Turkish/English diagnostic reason, computed class-rarity priority (`1.0..3.0`),
            recommended augmentation multiplier, and partitioned detections.
        """
        now_ts = time.time()
        det_list: List[Dict[str, Any]] = [dict(d) for d in (detections or []) if isinstance(d, dict)]

        frame_shape: Sequence[int] = (
            tuple(int(x) for x in frame.shape)
            if isinstance(frame, np.ndarray) and frame.size > 0
            else (0, 0, 0)
        )

        # Check status text for extraction failure strings
        status_lower = str(status_text or "").strip().lower()
        text_indicates_failure = any(pat in status_lower for pat in EXTRACTION_FAILURE_PATTERNS)
        effective_extraction_failed = bool(extraction_failed or text_indicates_failure)

        uncertain_detections: List[Dict[str, Any]] = []
        high_conf_detections: List[Dict[str, Any]] = []
        low_conf_detections: List[Dict[str, Any]] = []
        rejected_detections: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for det in det_list:
            label_candidate = str(
                det.get("class_name")
                or det.get("material_name")
                or det.get("name")
                or det.get("label")
                or ""
            )
            explicitly_rejected = bool(
                det.get("rejected", False)
                or det.get("is_monster", False)
                or det.get("is_feather", False)
                or det.get("false_positive", False)
                # Fast path: a known gatherable label (canonical name or
                # alias key) is verified to never be rejected by the
                # regex scan, so skip the ~34-pattern scan for it.
                or (
                    label_candidate
                    and label_candidate.strip().lower() not in KNOWN_GATHERABLE_LABELS
                    and _is_rejected_non_material_cached(label_candidate)
                )
            )
            if explicitly_rejected:
                rejected_detections.append(det)
                continue

            # Missing/unparseable confidence is UNKNOWN (0.0), never 1.0:
            # previously a confidence-less detection landed in
            # high_conf_detections, suppressing false-negative mining
            # (scanner_missed_node=False) and inflating priority (S-2002).
            raw_conf = det.get("confidence", det.get("conf", det.get("score")))
            try:
                conf = float(raw_conf) if raw_conf is not None else 0.0
            except (TypeError, ValueError):
                conf = 0.0
            conf = float(np.clip(conf, 0.0, 1.0))
            det["confidence"] = conf
            confidences.append(conf)

            if self.uncertainty_min_conf <= conf <= self.uncertainty_max_conf:
                # Uncertainty score peaks at the midpoint (0.58) of [0.42, 0.74]
                midpoint = 0.5 * (self.uncertainty_min_conf + self.uncertainty_max_conf)
                half_span = max(1e-6, 0.5 * (self.uncertainty_max_conf - self.uncertainty_min_conf))
                uncertainty_score = float(np.clip(1.0 - abs(conf - midpoint) / half_span, 0.1, 1.0))
                det_copy = dict(det)
                det_copy["uncertainty_score"] = round(uncertainty_score, 4)
                uncertain_detections.append(det_copy)
            elif conf > self.uncertainty_max_conf:
                high_conf_detections.append(det)
            else:
                low_conf_detections.append(det)

        # 1. Borderline Confidence Trigger
        uncertainty_triggered = len(uncertain_detections) > 0

        # 2. False-Negative Trigger (User manually pressed F8, or pressed F when scanner missed node)
        scanner_missed_node = len(high_conf_detections) == 0
        false_negative_triggered = bool(
            manual_f8_pressed or (manual_f_pressed and scanner_missed_node)
        )

        # 3. Hard-Negative Trigger (Extraction failed or monster/non-material rejection near node)
        hard_negative_triggered = bool(
            effective_extraction_failed or monster_rejected or len(rejected_detections) > 0
        )

        mining_types: List[str] = []
        reasons: List[str] = []

        if uncertainty_triggered:
            mining_types.append("uncertainty_mining")
            min_u = min(d["confidence"] for d in uncertain_detections)
            max_u = max(d["confidence"] for d in uncertain_detections)
            reasons.append(
                f"{self.UNCERTAINTY_MESSAGE_TR} (conf={min_u:.2f}..{max_u:.2f}, n={len(uncertain_detections)})"
            )

        if false_negative_triggered:
            mining_types.append("false_negative_mining")
            trigger_key = "F8" if manual_f8_pressed else "F"
            reasons.append(f"{self.FALSE_NEGATIVE_MESSAGE_TR} [key={trigger_key}]")

        if hard_negative_triggered:
            mining_types.append("hard_negative_mining")
            reasons.append(self.HARD_NEGATIVE_MESSAGE_TR)

        should_capture = len(mining_types) > 0
        if "false_negative_mining" in mining_types:
            primary_mining_type = "false_negative_mining"
        elif "hard_negative_mining" in mining_types:
            primary_mining_type = "hard_negative_mining"
        elif "uncertainty_mining" in mining_types:
            primary_mining_type = "uncertainty_mining"
        else:
            primary_mining_type = "none"

        # Candidate positive annotations for priority computation
        candidate_annotations = high_conf_detections + uncertain_detections
        priority = self.compute_sample_priority(
            annotations=candidate_annotations,
            class_counts=class_counts,
        )

        # Boost priority for explicit user F8 false-negative mining
        if false_negative_triggered:
            priority = float(np.clip(max(priority, 2.4), self.min_priority, self.max_priority))
        # NOTE: no uncertainty-only boost here -- uncertainty_triggered implies
        # candidate_annotations is non-empty (it contains the uncertain
        # detections), so the old "uncertainty and not candidate_annotations"
        # branch could never run.

        augmentation_multiplier = int(np.clip(round(priority * 3.3334), 1, 10))
        is_background_negative = bool(
            hard_negative_triggered and len(high_conf_detections) == 0 and len(uncertain_detections) == 0
        )

        max_uncertainty_score = (
            max(float(d.get("uncertainty_score", 0.0)) for d in uncertain_detections)
            if uncertain_detections
            else 0.0
        )

        reason_str = " | ".join(reasons) if reasons else "Normal yüksek güvenilirlikli kare."

        # P2-3 (K-17): thumbnail (OpenCV resize) + harici callback kilit
        # DISINDA. Eski kod kaydi, thumbnail'i ve callback cagrisini
        # `with self._lock:` icinde yapiyordu; yavas/hatali bir callback
        # tum degerlendirme + istatistik okuyucularini blokluyordu.
        # Kayit once kilitsiz kurulur, kilit yalnizca sayac/ekleme icin
        # kisa tutulur, callback kayit gecmise girdikten sonra kilitsiz
        # cagrilir (callback get_mined_history ile kaydi gorebilir).
        record: Optional[HardExampleRecord] = None
        callback: Optional[Callable[[HardExampleRecord], None]] = None
        if should_capture:
            record = HardExampleRecord(
                timestamp=now_ts,
                mining_types=list(mining_types),
                primary_mining_type=primary_mining_type,
                reason=reason_str,
                priority=priority,
                augmentation_multiplier=augmentation_multiplier,
                frame_shape=frame_shape,
                annotations=[] if is_background_negative else candidate_annotations,
                uncertain_detections=uncertain_detections,
                rejected_detections=rejected_detections,
                is_background_negative=is_background_negative,
                region_name=region_name,
                player_pos=player_pos,
                lookback_sec=lookback_sec,
                frame=self._thumbnail_frame(frame),
            )

        with self._lock:
            self.stats["frames_evaluated"] += 1
            # Detections below the uncertainty band are never mined and were
            # previously invisible in telemetry (A08): count them so a
            # sub-threshold false-negative stream can be noticed at all.
            self.stats["sub_threshold_detections"] += len(low_conf_detections)
            if should_capture:
                self.stats["total_triggered"] += 1
                if uncertainty_triggered:
                    self.stats["uncertainty_mining_count"] += 1
                if false_negative_triggered:
                    self.stats["false_negative_mining_count"] += 1
                if hard_negative_triggered:
                    self.stats["hard_negative_mining_count"] += 1
                if is_background_negative:
                    self.stats["background_negatives_mined"] += 1

                assert record is not None
                self._mined_history.append(record)
                callback = self.on_hard_example_callback

        if callback is not None and record is not None:
            try:
                callback(record)
            except Exception as exc:
                # Previously swallowed with only a log line (A08): the
                # mining decision is unchanged, but the failure is now
                # visible in stats instead of silently vanishing.
                with self._lock:
                    self.stats["callback_errors"] += 1
                logger.warning("on_hard_example_callback raised exception: %s", exc)

        return {
            "should_capture": should_capture,
            "triggered": should_capture,
            "mining_types": mining_types,
            "primary_mining_type": primary_mining_type,
            "uncertainty_mining": uncertainty_triggered,
            "false_negative_mining": false_negative_triggered,
            "hard_negative_mining": hard_negative_triggered,
            "uncertain_detections": uncertain_detections,
            "high_confidence_detections": high_conf_detections,
            "low_confidence_detections": low_conf_detections,
            "rejected_detections": rejected_detections,
            "min_confidence": min(confidences) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
            "max_uncertainty_score": round(max_uncertainty_score, 4),
            "priority": priority,
            "augmentation_multiplier": augmentation_multiplier,
            "is_background_negative": is_background_negative,
            "reason": reason_str,
            "reasons": reasons,
            "timestamp": now_ts,
            "frame_shape": frame_shape,
        }

    def get_mined_history(self, limit: Optional[int] = None) -> List[HardExampleRecord]:
        """Return recent mined hard-example records."""
        with self._lock:
            items = list(self._mined_history)
        if limit is not None and limit > 0:
            return items[-int(limit) :]
        return items

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic telemetry for the active learning loop."""
        with self._lock:
            return {
                **self.stats,
                "buffered_hard_examples": len(self._mined_history),
                "uncertainty_band": (self.uncertainty_min_conf, self.uncertainty_max_conf),
                "priority_range": (self.min_priority, self.max_priority),
                "class_counts": dict(self.class_counts),
            }


# =============================================================================
# 3. Passive Shadow Recording Mode ("Pasif Gölge Kayıt Modu")
# =============================================================================


class ShadowGameplayRecorder:
    """
    Passive Shadow Recording Mode (`"Pasif Gölge Kayıt Modu"`).

    Allows the user to play Aion 2 normally while the bot sends ZERO keyboard or
    mouse inputs (`bot_input_enabled = False`). Silently watches live frames:
    - Pushes every frame into the `PreGatherTimeMachineBuffer`.
    - Detects when the user manually approaches and harvests a resource node
      (via `F` / `E` interaction, `F8` manual capture hotkey, gather completion
      event, or floating interaction UI state transition).
    - Rewinds the Time Machine by `lookback_sec = 2.0s` (plus optional multi-horizon
      offsets `2.5s`, `2.0s`, `1.2s`) to grab the exact frame where the resource
      node (Aria, Mela, Orichalcum, Sapphire, etc.) was still intact in front of
      the character.
    - Auto-labels the retrieved frame(s) in the background (either synchronously
      or on a background worker thread) and scores them with `ActiveLearningLoop`.
    """

    MODE_NAME_TR: str = "Pasif Gölge Kayıt Modu"

    def __init__(
        self,
        time_machine: Optional[PreGatherTimeMachineBuffer] = None,
        active_learning_loop: Optional[ActiveLearningLoop] = None,
        auto_labeler: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = None,
        on_sample_recorded: Optional[Callable[[Dict[str, Any]], None]] = None,
        lookback_sec: float = 2.0,
        multi_horizon_sec: Sequence[float] = (2.5, 2.0, 1.2),
        enable_multi_horizon: bool = False,
        harvest_cooldown_sec: float = 1.25,
        background_worker: bool = True,
        max_recorded_samples: int = 500,
        passive_shadow_enabled: bool = False,
    ) -> None:
        self.time_machine: PreGatherTimeMachineBuffer = (
            time_machine
            if time_machine is not None
            else PreGatherTimeMachineBuffer(max_duration_sec=6.0, min_push_interval_sec=0.15)
        )
        self.active_learning: ActiveLearningLoop = (
            active_learning_loop if active_learning_loop is not None else ActiveLearningLoop()
        )
        self.auto_labeler = auto_labeler
        self.on_sample_recorded = on_sample_recorded

        self.lookback_sec: float = max(0.2, float(lookback_sec))
        self.multi_horizon_sec: Tuple[float, float, float] = (
            float(multi_horizon_sec[0]) if len(multi_horizon_sec) > 0 else 2.5,
            float(multi_horizon_sec[1]) if len(multi_horizon_sec) > 1 else 2.0,
            float(multi_horizon_sec[2]) if len(multi_horizon_sec) > 2 else 1.2,
        )
        self.enable_multi_horizon: bool = bool(enable_multi_horizon)
        self.harvest_cooldown_sec: float = max(0.0, float(harvest_cooldown_sec))
        self.max_recorded_samples: int = max(16, int(max_recorded_samples))

        # Strict passive guarantee: bot never presses keys in Shadow Mode.
        # Born DISABLED unless the caller (factory) explicitly enables passive
        # shadow mode -- previously this defaulted to True even when the
        # factory was constructed with passive_shadow_mode=False.
        self.passive_shadow_enabled: bool = bool(passive_shadow_enabled)
        self.bot_input_enabled: bool = False

        self._lock = threading.RLock()
        self._last_harvest_ts: float = -1e9
        self._last_uncertainty_ts: float = -1e9
        self._recorded_samples: Deque[Dict[str, Any]] = deque(maxlen=self.max_recorded_samples)
        self._task_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=256)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.stats: Dict[str, int] = {
            "frames_observed": 0,
            "manual_harvests_detected": 0,
            "f8_triggers_detected": 0,
            "uncertainty_triggers_detected": 0,
            "time_machine_retrievals": 0,
            "samples_auto_labeled": 0,
            "bot_key_presses_sent": 0,  # Always 0 in Pasif Gölge Kayıt Modu
            # Telemetry for previously silent failure/drop paths (A08): these
            # only count events, they never change a capture decision.
            "callback_errors": 0,
            "worker_job_errors": 0,
            "time_machine_misses": 0,
            "queue_full_sync_fallbacks": 0,
        }

        if background_worker:
            self.start_background_worker()

    # -------------------------------------------------------------------------
    # Strict Zero-Key-Press Safety Guards
    # -------------------------------------------------------------------------

    def can_send_input(self) -> bool:
        """In Passive Shadow Recording Mode, the bot is strictly forbidden from sending input."""
        return False

    def assert_passive_only(self) -> bool:
        """Verify that zero key presses have been emitted by the shadow recorder."""
        if self.bot_input_enabled or self.stats["bot_key_presses_sent"] != 0:
            raise RuntimeError("Passive Shadow Mode violation: bot input must remain strictly disabled!")
        return True

    # -------------------------------------------------------------------------
    # Background Auto-Labeling Worker
    # -------------------------------------------------------------------------

    def start_background_worker(self) -> None:
        """Start the daemon thread that auto-labels retrieved Time Machine frames."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._background_worker_loop,
                name="ShadowGameplayAutoLabelWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_background_worker(self, timeout: float = 2.0) -> bool:
        """Gracefully drain and stop the background auto-labeling worker.

        Returns True when no worker thread is left alive afterwards (A08:
        previously returned None, so a caller could not tell a hung worker
        from a clean stop).
        """
        self._stop_event.set()
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            logger.debug("ShadowGameplayRecorder queue already full during stop signal.")
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.1, float(timeout)))
            return not worker.is_alive()
        return True

    def flush_pending_tasks(self, timeout: float = 2.0) -> int:
        """Process any queued auto-labeling jobs synchronously and wait for queue completion.

        The stop sentinel (``None``) no longer discards the jobs still queued
        behind it: after seeing it, the remaining queued jobs are drained
        first, then flushing stops.
        """
        deadline = time.time() + max(0.05, float(timeout))
        processed = 0
        stop_seen = False
        while time.time() < deadline:
            try:
                job = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if job is None:
                self._task_queue.task_done()
                stop_seen = True
                continue
            try:
                self._process_capture_job(job)
                processed += 1
            finally:
                self._task_queue.task_done()
            if stop_seen:
                # Drain whatever was queued behind the sentinel, then stop.
                continue
        return processed

    def _background_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._task_queue.get(timeout=0.15)
            except queue.Empty:
                continue
            if job is None:
                self._task_queue.task_done()
                break
            try:
                self._process_capture_job(job)
            except Exception as exc:
                # Previously only a log line (A08): the job is still dropped,
                # but the loss is now visible in stats instead of silent.
                with self._lock:
                    self.stats["worker_job_errors"] += 1
                logger.warning("ShadowGameplayRecorder worker error: %s", exc)
            finally:
                self._task_queue.task_done()

    # -------------------------------------------------------------------------
    # Built-in Fallback Auto-Labeler (3D Crest + HSV Patch + Detections)
    # -------------------------------------------------------------------------

    def _auto_label_snapshot(
        self,
        snapshot: FrameSnapshot,
        material_hint: str = "",
        region_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Auto-label a retrieved `t - 2.0s` `FrameSnapshot`.

        Uses:
        1. External `self.auto_labeler` if injected by the caller.
        2. Existing detections cached on the `FrameSnapshot` (converting pixel boxes
           to normalized YOLO `(cx, cy, w, h)`).
        3. Built-in 3D HSV + Center-Interaction Zone Auto-Labeler powered by
           `classify_3d_material_patch` from `config.materials_db` so that even when
           the user harvested a node the detector missed, a clean bounding box and
           class label are generated around the intact resource node in front of the
           character.
        """
        frame = snapshot.frame
        if frame is None or frame.size == 0 or frame.ndim < 2:
            return []

        img_h, img_w = int(frame.shape[0]), int(frame.shape[1])
        effective_region = str(region_name or snapshot.region_name or "")

        # 1. External auto-labeler callback if provided
        if callable(self.auto_labeler):
            try:
                external_anns = self.auto_labeler(
                    frame,
                    material_hint=material_hint,
                    region_name=effective_region,
                    detections=snapshot.detections,
                )
            except TypeError:
                external_anns = self.auto_labeler(frame)
            if external_anns:
                return self._normalize_annotations(external_anns, img_w, img_h, material_hint)

        # 2. Snapshot's existing detections if present
        if snapshot.detections:
            valid_dets = [
                d
                for d in snapshot.detections
                if not is_rejected_non_material(
                    str(d.get("class_name") or d.get("material_name") or d.get("name") or "")
                )
            ]
            if valid_dets:
                normalized = self._normalize_annotations(valid_dets, img_w, img_h, material_hint)
                if normalized:
                    return normalized

        # 3. Built-in 3D Resource Node Localization in the Pre-Gather Front Zone
        # At t - 2.0s before harvest, the intact resource node lies in the central
        # foreground corridor in front of the character ([0.28..0.72]x, [0.30..0.78]y).
        x1_roi = int(round(img_w * 0.28))
        x2_roi = int(round(img_w * 0.72))
        y1_roi = int(round(img_h * 0.30))
        y2_roi = int(round(img_h * 0.78))
        roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        resolved_class_name = normalize_material_class_key(material_hint) if material_hint else ""
        # No fabricated fallback: a box is emitted ONLY when there is real
        # evidence -- a classifier match and/or a salient contour plus a
        # class hint. Previously every failed detection produced an invented
        # herb_aria box at the frame centre with conf 0.88, poisoning the
        # training set with boxes of objects that were never seen.
        resolved_conf: Optional[float] = None
        classifier_matched = False
        best_box_px: Optional[Tuple[int, int, int, int]] = None

        if roi.size > 0 and roi.ndim == 3 and roi.shape[2] >= 3:
            bgr_roi = roi[:, :, :3]
            patch_result = classify_3d_material_patch(
                mesh_bgr=bgr_roi,
                region_name=effective_region or None,
            )
            best_match: Optional[Dict[str, Any]] = None
            if isinstance(patch_result, dict):
                best_match = patch_result
            elif isinstance(patch_result, list) and len(patch_result) > 0:
                best_match = patch_result[0]
            if best_match is not None:
                # The patch classifier ALWAYS returns a best guess (its
                # default is Orichalcum at 0.65 even for a black patch), so a
                # bare dict is not evidence. Only count it when its colour
                # ratios actually show signal in the ROI.
                color_ratios = best_match.get("color_ratios") or {}
                has_color_signal = bool(
                    isinstance(color_ratios, dict)
                    and any(float(v or 0.0) > 0.0 for v in color_ratios.values())
                )
                if has_color_signal:
                    classifier_matched = True
                    if not resolved_class_name:
                        resolved_class_name = normalize_material_class_key(
                            best_match.get("base_material") or best_match.get("name")
                        )
                    try:
                        raw_score = float(best_match.get("confidence", best_match.get("score", 0.0)))
                    except (TypeError, ValueError):
                        raw_score = 0.0
                    # No floor: the classifier's own score is reported as-is.
                    resolved_conf = float(np.clip(raw_score, 0.05, 0.99))

            # Localize the most salient chromatic/bright object contour inside the ROI
            hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
            sat_mask = cv2.inRange(hsv_roi, (0, 65, 65), (180, 255, 255))
            kernel = np.ones((5, 5), dtype=np.uint8)
            sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_area = 0.0
            min_contour_area = max(16.0, float(roi.shape[0] * roi.shape[1]) * 0.008)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area >= min_contour_area and area > best_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    best_area = area
                    best_box_px = (x1_roi + bx, y1_roi + by, bw, bh)

        if not resolved_class_name or (not classifier_matched and best_box_px is None):
            # Nothing was actually detected: no class evidence, or a contour
            # with no class to attach to it. Emit no box at all.
            logger.debug(
                "Shadow fallback found no evidence (classifier=%s, contour=%s); no box emitted",
                classifier_matched,
                best_box_px is not None,
            )
            return []

        if best_box_px is not None:
            px, py, pw, bh = best_box_px
            cx = float(np.clip((px + 0.5 * pw) / max(1, img_w), 0.05, 0.95))
            cy = float(np.clip((py + 0.5 * bh) / max(1, img_h), 0.05, 0.95))
            nw = float(np.clip(pw / max(1, img_w), 0.04, 0.45))
            nh = float(np.clip(bh / max(1, img_h), 0.04, 0.45))
        else:
            # Classifier matched a material but no contour localized it: use
            # the ROI centre as a weak hint, never as a confident box.
            cx, cy, nw, nh = 0.50, 0.56, 0.14, 0.16
            if resolved_conf is None:
                resolved_conf = 0.50

        if resolved_conf is None:
            # Contour found with a material hint but no classifier score.
            resolved_conf = 0.50
        class_id = resolve_class_id(resolved_class_name)

        return [
            {
                "class_id": int(class_id),
                "class_name": str(resolved_class_name),
                "cx": round(cx, 6),
                "cy": round(cy, 6),
                "w": round(nw, 6),
                "h": round(nh, 6),
                "yolo_bbox": (round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)),
                "confidence": round(float(resolved_conf), 4),
                "source": "shadow_time_machine_auto_labeler",
                # Weak whenever the box rests on a hint/contour without a
                # classifier match, or when no contour localized it (centre
                # guess), so downstream counters never treat it as a
                # trusted detection.
                "weak": (not classifier_matched) or (best_box_px is None),
                "trusted": False,
            }
        ]

    @staticmethod
    def _to_normalized(value: float, dim: int) -> float:
        """Interpret one coordinate: >1.0 means pixels (divide by dim), else normalized."""
        return value / max(1, dim) if value > 1.0 else value

    @staticmethod
    def _normalize_annotations(
        raw_annotations: Sequence[Dict[str, Any]],
        img_w: int,
        img_h: int,
        material_hint: str = "",
    ) -> List[Dict[str, Any]]:
        """Ensure every annotation dict has normalized `cx, cy, w, h`, `class_id`, and `class_name`."""
        normalized: List[Dict[str, Any]] = []
        for ann in raw_annotations:
            if not isinstance(ann, dict):
                continue
            raw_cls = (
                ann.get("class_name")
                or ann.get("material_name")
                or ann.get("name")
                or material_hint
                or ann.get("class_id")
            )
            if raw_cls is None or (isinstance(raw_cls, str) and not raw_cls.strip()):
                # No class evidence at all (empty dict / class-less dict):
                # never fabricate herb_aria for an unseen object (S-2001).
                logger.debug("Skipping annotation without class evidence: %r", ann)
                continue
            class_name = normalize_material_class_key(raw_cls)
            if not class_name:
                logger.debug("Skipping annotation with unresolvable class: %r", ann)
                continue
            # S-5502: derive class_id from the NORMALIZED class_name, never
            # from ann's raw class_id. The two resolvers disagreed on legacy
            # material names (e.g. "sapphire" normalized to "gem_sapphire"
            # while resolve_class_id gave 4 = gem_crystal), emitting
            # {"class_id": 4, "class_name": "gem_sapphire"}. A single
            # source of truth makes id and name agree by construction.
            # Annotations carrying only a legacy/non-12 id now normalize to
            # "" above and are skipped instead of being mislabelled.
            class_id = resolve_class_id(class_name)
            if class_id < 0:
                # Class name could not be resolved to a trainable id:
                # do not label it as herb_aria (id 0) (S-2001).
                logger.debug("Skipping annotation with unknown class %r: %r", raw_cls, ann)
                continue

            # Missing confidence is UNKNOWN (0.0), never 0.90 (S-2001):
            # a confidence-less annotation must not pose as a confident box.
            raw_conf = ann.get("confidence", ann.get("conf"))
            try:
                conf_val = float(raw_conf) if raw_conf is not None else 0.0
            except (TypeError, ValueError):
                conf_val = 0.0
            conf_val = float(np.clip(conf_val, 0.0, 1.0))

            cx: Optional[float] = None
            cy: Optional[float] = None
            w: Optional[float] = None
            h: Optional[float] = None

            if "cx" in ann and "cy" in ann and ("w" in ann or "width" in ann) and ("h" in ann or "height" in ann):
                # Each coordinate is judged on its own: a box may legitimately
                # mix a normalized centre (cx=0.5) with a pixel width (w=120).
                # The old code divided ALL four values whenever ANY one
                # exceeded 1.0, destroying such mixed boxes.
                cx = ShadowGameplayRecorder._to_normalized(float(ann["cx"]), img_w)
                cy = ShadowGameplayRecorder._to_normalized(float(ann["cy"]), img_h)
                w = ShadowGameplayRecorder._to_normalized(float(ann.get("w", ann.get("width", 0.1))), img_w)
                h = ShadowGameplayRecorder._to_normalized(float(ann.get("h", ann.get("height", 0.1))), img_h)
            elif "yolo_bbox" in ann and len(ann["yolo_bbox"]) == 4:
                _bx, _by, _bw, _bh = [float(v) for v in ann["yolo_bbox"]]
                cx = ShadowGameplayRecorder._to_normalized(_bx, img_w)
                cy = ShadowGameplayRecorder._to_normalized(_by, img_h)
                w = ShadowGameplayRecorder._to_normalized(_bw, img_w)
                h = ShadowGameplayRecorder._to_normalized(_bh, img_h)
            elif "bbox_xyxy" in ann and len(ann["bbox_xyxy"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in ann["bbox_xyxy"]]
                cx = ((x1 + x2) * 0.5) / max(1, img_w)
                cy = ((y1 + y2) * 0.5) / max(1, img_h)
                w = abs(x2 - x1) / max(1, img_w)
                h = abs(y2 - y1) / max(1, img_h)
            elif "bbox" in ann and len(ann["bbox"]) == 4:
                b0, b1, b2, b3 = [float(v) for v in ann["bbox"]]
                if max(b0, b1, b2, b3) <= 1.0:
                    # Normalize (cx, cy, w, h) — YOLO merkez sözleşmesi
                    # (yolo_bbox dalıyla aynı).
                    cx, cy, w, h = b0, b1, b2, b3
                else:
                    # S161/S179 (v19/P3-8): piksel `bbox` repo
                    # sözleşmesinde her zaman (x, y, w, h) sol-üsttür
                    # (vision.py:1508,2743,2852 — hepsi (x, y, bw, bh)).
                    # Eski kod w > img_w/2 veya h > img_h/2 ise köşe
                    # (x1,y1,x2,y2) sanıyordu; geniş bir tespit kutusu
                    # yanlış merkeze/genişliğe dönüşüyordu. Köşe biçimi
                    # yalnızca açık `bbox_xyxy` anahtarıyla gelir.
                    cx = (b0 + 0.5 * b2) / max(1, img_w)
                    cy = (b1 + 0.5 * b3) / max(1, img_h)
                    w = b2 / max(1, img_w)
                    h = b3 / max(1, img_h)
            elif "x" in ann and "y" in ann:
                px = float(ann["x"])
                py = float(ann["y"])
                pw = float(ann.get("w", ann.get("width", img_w * 0.12)))
                ph = float(ann.get("h", ann.get("height", img_h * 0.14)))
                cx = px / max(1, img_w) if px > 1.0 else px
                cy = py / max(1, img_h) if py > 1.0 else py
                w = pw / max(1, img_w) if pw > 1.0 else pw
                h = ph / max(1, img_h) if ph > 1.0 else ph
            else:
                # No coordinate evidence at all: do NOT invent a centre
                # box (0.50, 0.56, 0.14, 0.16) for an object that was never
                # seen — skip the annotation entirely (S-2001).
                logger.debug("Skipping annotation without coordinates: %r", ann)
                continue

            cx = float(np.clip(cx, 0.01, 0.99))
            cy = float(np.clip(cy, 0.01, 0.99))
            w = float(np.clip(w, 0.01, min(0.98, 2.0 * min(cx, 1.0 - cx))))
            h = float(np.clip(h, 0.01, min(0.98, 2.0 * min(cy, 1.0 - cy))))

            normalized.append(
                {
                    **ann,
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w": round(w, 6),
                    "h": round(h, 6),
                    "yolo_bbox": (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)),
                    "confidence": conf_val,
                }
            )
        return normalized

    # -------------------------------------------------------------------------
    # Main Observation & Time-Machine Capture Pipeline
    # -------------------------------------------------------------------------

    def _process_capture_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-label the retrieved `t - 2.0s` snapshot(s) and register the training sample."""
        snapshot: FrameSnapshot = job["snapshot"]
        trigger_source: str = job["trigger_source"]
        event_ts: float = float(job["event_ts"])
        lookback_sec: float = float(job["lookback_sec"])
        material_hint: str = str(job.get("material_hint", ""))
        region_name: str = str(job.get("region_name") or snapshot.region_name or "")
        evaluation: Dict[str, Any] = job["evaluation"]

        annotations = self._auto_label_snapshot(
            snapshot=snapshot,
            material_hint=material_hint,
            region_name=region_name,
        )

        # Recompute sample priority with the newly generated annotations
        priority = self.active_learning.compute_sample_priority(annotations)
        for ann in annotations:
            self.active_learning.increment_class_count(ann["class_name"], delta=1)

        sample_record: Dict[str, Any] = {
            "mode": self.MODE_NAME_TR,
            "trigger_source": trigger_source,
            "event_timestamp": event_ts,
            "snapshot_timestamp": snapshot.timestamp,
            "actual_lookback_sec": round(max(0.0, event_ts - snapshot.timestamp), 3),
            "target_lookback_sec": lookback_sec,
            "frame": snapshot.frame,
            "frame_shape": snapshot.shape,
            "player_pos": snapshot.player_pos,
            "region_name": region_name,
            "material_hint": material_hint,
            "annotations": annotations,
            "priority": priority,
            "augmentation_multiplier": int(np.clip(round(priority * 3.3334), 1, 10)),
            "active_learning": evaluation,
            "bot_key_presses_sent": 0,
        }

        # Bound retained memory: the long-lived deque keeps a <=320px
        # thumbnail of the frame (same bound as HardExampleRecord, whose
        # full-res copies cost gigabytes at 500 records); the full frame
        # still goes to the on_sample_recorded callback and to the
        # returned record, which are transient -- the factory callback
        # saves it to disk synchronously.
        stored_record = {
            **sample_record,
            "frame": ActiveLearningLoop._thumbnail_frame(snapshot.frame),
        }
        with self._lock:
            self._recorded_samples.append(stored_record)
            self.stats["samples_auto_labeled"] += 1

        if callable(self.on_sample_recorded):
            try:
                self.on_sample_recorded(sample_record)
            except Exception as exc:
                # Previously only a log line (A08): the sample stays recorded,
                # but the downstream persistence failure is now counted.
                with self._lock:
                    self.stats["callback_errors"] += 1
                logger.warning("on_sample_recorded callback failed: %s", exc)

        return sample_record

    def observe_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        player_pos: Tuple[float, float] = (0.0, 0.0),
        detections: Optional[List[Dict[str, Any]]] = None,
        region_name: str = "",
        *,
        user_harvested: bool = False,
        manual_f_pressed: bool = False,
        manual_f8_pressed: bool = False,
        extraction_failed: bool = False,
        material_hint: str = "",
        status_text: str = "",
        sync_label: bool = False,
    ) -> Dict[str, Any]:
        """
        Observe a single live gameplay frame in Passive Shadow Mode (`"Pasif Gölge Kayıt Modu"`).

        1. Never sends any key or mouse inputs (`assert_passive_only()`).
        2. Pushes `frame` into the `PreGatherTimeMachineBuffer`.
        3. Evaluates active learning conditions (`uncertainty_mining`, `false_negative_mining`,
           `hard_negative_mining`) and user harvest triggers (`user_harvested`, `manual_f_pressed`,
           `manual_f8_pressed`).
        4. When a user harvest or F8 trigger occurs, pulls the `t - 2.0s` frame from the
           Time Machine (when the material was still intact in front of the player) and
           auto-labels it in the background (or synchronously if `sync_label=True` or
           background worker is disabled).
        """
        self.assert_passive_only()
        ts = float(timestamp) if timestamp is not None else time.time()

        # 1. Push frame into the 2-Second Pre-Gather Time Machine Ring Buffer
        pushed_snap = self.time_machine.push_frame(
            frame=frame,
            timestamp=ts,
            player_pos=player_pos,
            detections=detections,
            region_name=region_name,
            metadata={"material_hint": material_hint, "mode": self.MODE_NAME_TR},
        )

        with self._lock:
            self.stats["frames_observed"] += 1

        # 2. Evaluate frame through Active Learning & Hard-Example Detector
        evaluation = self.active_learning.evaluate_frame_for_active_learning(
            frame=frame,
            detections=detections,
            extraction_failed=extraction_failed,
            manual_f8_pressed=manual_f8_pressed,
            manual_f_pressed=manual_f_pressed or user_harvested,
            status_text=status_text,
            region_name=region_name,
            player_pos=player_pos,
            lookback_sec=self.lookback_sec,
        )

        # 3. Check if passive shadow capture should rewind the Time Machine
        harvest_event = bool(user_harvested or manual_f_pressed)
        cooldown_elapsed = (ts - self._last_harvest_ts) >= (self.harvest_cooldown_sec - 1e-6)
        # Uncertainty rewinds are rate-limited by their OWN timestamp: an
        # uncertainty trigger used to consume the harvest cooldown, so a
        # borderline frame could block a real harvest rewind right after it.
        # F8 stays cooldown-free (it never touches either timestamp).
        uncertainty_cooldown_elapsed = (ts - self._last_uncertainty_ts) >= (self.harvest_cooldown_sec - 1e-6)

        should_rewind_time_machine = False
        trigger_source = "none"

        if manual_f8_pressed:
            should_rewind_time_machine = True
            trigger_source = "manual_f8_hotkey"
            with self._lock:
                self.stats["f8_triggers_detected"] += 1
        elif harvest_event and cooldown_elapsed:
            should_rewind_time_machine = True
            trigger_source = "passive_shadow_user_harvest"
            with self._lock:
                self._last_harvest_ts = ts
                self.stats["manual_harvests_detected"] += 1
        elif evaluation["uncertainty_mining"] and uncertainty_cooldown_elapsed:
            should_rewind_time_machine = True
            trigger_source = "active_learning_uncertainty"
            with self._lock:
                self._last_uncertainty_ts = ts
                self.stats["uncertainty_triggers_detected"] += 1

        captured_samples: List[Dict[str, Any]] = []
        retrieved_snapshots: List[FrameSnapshot] = []

        if should_rewind_time_machine:
            if harvest_event:
                # Rewind 2.0 seconds to grab the intact node before the user harvested it!
                if self.enable_multi_horizon:
                    retrieved_snapshots = self.time_machine.get_multi_horizon_frames(
                        horizons_sec=self.multi_horizon_sec,
                        reference_ts=ts,
                        deduplicate=True,
                    )
                else:
                    snap_2s = self.time_machine.get_frame_before(
                        seconds_ago=self.lookback_sec,
                        reference_ts=ts,
                    )
                    if snap_2s is not None:
                        retrieved_snapshots = [snap_2s]
            else:
                # For immediate F8 or uncertainty on the current frame, use latest or t-2.0s if requested
                snap_target = (
                    self.time_machine.get_frame_before(seconds_ago=self.lookback_sec, reference_ts=ts)
                    if harvest_event
                    else (pushed_snap or self.time_machine.get_latest_frame())
                )
                if snap_target is not None:
                    retrieved_snapshots = [snap_target]

            with self._lock:
                self.stats["time_machine_retrievals"] += len(retrieved_snapshots)
                # A rewind that retrieves nothing used to leave no trace (A08):
                # count it so an empty/short buffer or a dead time machine is
                # visible instead of silently dropping the harvest sample.
                if should_rewind_time_machine and not retrieved_snapshots:
                    self.stats["time_machine_misses"] += 1

            for snap in retrieved_snapshots:
                job = {
                    "snapshot": snap,
                    "trigger_source": trigger_source,
                    "event_ts": ts,
                    "lookback_sec": self.lookback_sec if harvest_event else 0.0,
                    "material_hint": material_hint,
                    "region_name": region_name or snap.region_name,
                    "evaluation": evaluation,
                }
                use_sync = bool(
                    sync_label
                    or self._worker_thread is None
                    or not self._worker_thread.is_alive()
                )
                if use_sync:
                    sample_rec = self._process_capture_job(job)
                    captured_samples.append(sample_rec)
                else:
                    try:
                        self._task_queue.put_nowait(job)
                    except queue.Full:
                        # Sync fallback still runs the job inline (A08): count
                        # it so queue saturation shows up in telemetry.
                        with self._lock:
                            self.stats["queue_full_sync_fallbacks"] += 1
                        sample_rec = self._process_capture_job(job)
                        captured_samples.append(sample_rec)

        return {
            "mode": self.MODE_NAME_TR,
            "passive_only": True,
            "bot_key_presses_sent": 0,
            "triggered": should_rewind_time_machine or evaluation["should_capture"],
            "trigger_source": trigger_source,
            "time_machine_rewound": len(retrieved_snapshots) > 0,
            "retrieved_snapshot_timestamps": [s.timestamp for s in retrieved_snapshots],
            "captured_samples": captured_samples,
            "active_learning": evaluation,
        }

    def get_recorded_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return auto-labeled samples collected during Passive Shadow Recording Mode."""
        with self._lock:
            samples = list(self._recorded_samples)
        if limit is not None and limit > 0:
            return samples[-int(limit) :]
        return samples

    def get_stats(self) -> Dict[str, Any]:
        """Return full telemetry for Passive Shadow Recording Mode."""
        with self._lock:
            return {
                "mode": self.MODE_NAME_TR,
                "passive_shadow_enabled": self.passive_shadow_enabled,
                "bot_input_enabled": self.bot_input_enabled,
                # A08: health/dead-worker checks need liveness, not just the
                # queue size (a dead worker leaves jobs queued with no error).
                "worker_alive": bool(self._worker_thread is not None and self._worker_thread.is_alive()),
                **self.stats,
                "recorded_samples_buffered": len(self._recorded_samples),
                "pending_queue_size": self._task_queue.qsize(),
                "time_machine": self.time_machine.get_stats(),
                "active_learning": self.active_learning.get_stats(),
            }


__all__ = [
    "CANONICAL_12_YOLO_CLASSES",
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "HardExampleRecord",
    "ActiveLearningLoop",
    "ShadowGameplayRecorder",
    "normalize_material_class_key",
    "resolve_class_id",
]
