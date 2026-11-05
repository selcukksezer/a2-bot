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
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

CANONICAL_12_YOLO_CLASSES: Tuple[str, str, str, str, str, str, str, str, str, str, str, str] = (
    "herb_aria",
    "herb_targena",
    "herb_azpha",
    "herb_calendula",
    "fruit_mela",
    "ore_orichalcum",
    "ore_iron_silver",
    "gem_sapphire",
    "gem_ruby",
    "gem_diamond",
    "gem_crystal",
    "log_timber",
)

CLASS_ID_TO_NAME: Dict[int, str] = {
    idx: name for idx, name in enumerate(CANONICAL_12_YOLO_CLASSES)
}

CLASS_NAME_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(CANONICAL_12_YOLO_CLASSES)
}

MATERIAL_ALIAS_TO_CANONICAL_CLASS: Dict[str, str] = {
    # Canonical names
    "herb_aria": "herb_aria",
    "herb_targena": "herb_targena",
    "herb_azpha": "herb_azpha",
    "herb_calendula": "herb_calendula",
    "fruit_mela": "fruit_mela",
    "ore_orichalcum": "ore_orichalcum",
    "ore_iron_silver": "ore_iron_silver",
    "gem_sapphire": "gem_sapphire",
    "gem_ruby": "gem_ruby",
    "gem_diamond": "gem_diamond",
    "gem_crystal": "gem_crystal",
    "log_timber": "log_timber",
    # Direct material names from AION2_MATERIALS_CATALOG
    "aria": "herb_aria",
    "fresh aria": "herb_aria",
    "taze aria": "herb_aria",
    "targena": "herb_targena",
    "fresh targena": "herb_targena",
    "targena otu": "herb_targena",
    "azpha": "herb_azpha",
    "fresh azpha": "herb_azpha",
    "azpha otu": "herb_azpha",
    "calendula": "herb_calendula",
    "fresh calendula": "herb_calendula",
    "mela": "fruit_mela",
    "fresh mela": "fruit_mela",
    "taze mela": "fruit_mela",
    "orichalcum": "ore_orichalcum",
    "orichalcum ore": "ore_orichalcum",
    "fine orichalcum ore": "ore_orichalcum",
    "pure orichalcum ore": "ore_orichalcum",
    "radiant orichalcum ore": "ore_orichalcum",
    "iron ore": "ore_iron_silver",
    "silver ore": "ore_iron_silver",
    "copper ore": "ore_iron_silver",
    "gold ore": "ore_iron_silver",
    "titanium ore": "ore_iron_silver",
    "sapphire": "gem_sapphire",
    "sapphire gemstone": "gem_sapphire",
    "fine sapphire gemstone": "gem_sapphire",
    "pure sapphire gemstone": "gem_sapphire",
    "radiant sapphire gemstone": "gem_sapphire",
    "aquamarine": "gem_sapphire",
    "ruby": "gem_ruby",
    "ruby gemstone": "gem_ruby",
    "fine ruby gemstone": "gem_ruby",
    "pure ruby gemstone": "gem_ruby",
    "radiant ruby gemstone": "gem_ruby",
    "garnet": "gem_ruby",
    "diamond": "gem_diamond",
    "diamond gemstone": "gem_diamond",
    "fine diamond gemstone": "gem_diamond",
    "pure diamond gemstone": "gem_diamond",
    "radiant diamond gemstone": "gem_diamond",
    "crystal": "gem_crystal",
    "kristal": "gem_crystal",
    "rose quartz": "gem_crystal",
    "amethyst": "gem_crystal",
    "topaz": "gem_crystal",
    "emerald": "gem_crystal",
    "opal": "gem_crystal",
    "yggdrasil log": "log_timber",
    "asvata log": "log_timber",
    "pine log": "log_timber",
    "oak log": "log_timber",
    "maple log": "log_timber",
    "cedar log": "log_timber",
}

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
    """
    if raw_key is None:
        return "herb_aria"
    if isinstance(raw_key, (int, np.integer)):
        cid = int(raw_key)
        return CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
    text = str(raw_key).strip()
    if not text:
        return "herb_aria"
    if text.isdigit():
        cid = int(text)
        return CLASS_ID_TO_NAME.get(cid, f"class_{cid}")
    lowered = text.lower()
    if lowered in MATERIAL_ALIAS_TO_CANONICAL_CLASS:
        return MATERIAL_ALIAS_TO_CANONICAL_CLASS[lowered]

    # Category-level heuristic fallback
    if "aria" in lowered:
        return "herb_aria"
    if "targena" in lowered:
        return "herb_targena"
    if "azpha" in lowered:
        return "herb_azpha"
    if "calendula" in lowered:
        return "herb_calendula"
    if "mela" in lowered or "berry" in lowered or "fruit" in lowered:
        return "fruit_mela"
    if "orichalcum" in lowered:
        return "ore_orichalcum"
    if "ore" in lowered or "iron" in lowered or "silver" in lowered or "metal" in lowered:
        return "ore_iron_silver"
    if "sapphire" in lowered:
        return "gem_sapphire"
    if "ruby" in lowered or "garnet" in lowered:
        return "gem_ruby"
    if "diamond" in lowered:
        return "gem_diamond"
    if "crystal" in lowered or "quartz" in lowered or "amethyst" in lowered or "gem" in lowered:
        return "gem_crystal"
    if "log" in lowered or "tree" in lowered or "timber" in lowered or "wood" in lowered:
        return "log_timber"
    return lowered


def resolve_class_id(raw_key: Union[int, str, None]) -> int:
    """Resolve an annotation class identifier or material name to an integer class ID in 0..11."""
    if isinstance(raw_key, (int, np.integer)):
        return int(np.clip(int(raw_key), 0, len(CANONICAL_12_YOLO_CLASSES) - 1))
    canonical = normalize_material_class_key(raw_key)
    return CLASS_NAME_TO_ID.get(canonical, 0)


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
        }

    def get_active_learning_health_summary(self) -> dict:
        try:
            return {"active_learning_ready": True}
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
                self.class_counts[key] = max(0, int(val))
            return dict(self.class_counts)

    def compute_economic_priority_bonus(self, material_name: str) -> float:
        """Compute economic priority bonus in [0.0, 0.5] based on Aion 2 material market value."""
        try:
            from core.economy_broker import DEFAULT_AION2_MATERIAL_PRICES
            norm = (material_name or "").strip().lower()
            price = 500
            for k, v in DEFAULT_AION2_MATERIAL_PRICES.items():
                if k in norm or norm in k:
                    price = int(v)
                    break
            return round(min(0.5, max(0.0, (price - 400.0) / 8000.0)), 3)
        except Exception:
            return 0.0

    def increment_class_count(self, class_key: Union[int, str], delta: int = 1) -> int:
        """Increment sample count for a specific material class."""
        with self._lock:
            key = normalize_material_class_key(class_key)
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
                normalized_counts[norm_k] = normalized_counts.get(norm_k, 0) + max(0, int(v))
                # Also keep exact string key for direct custom lookups
                if isinstance(k, str):
                    normalized_counts[k.strip().lower()] = max(0, int(v))

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

        class_priorities: List[float] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            raw_cls = (
                ann.get("class_name")
                or ann.get("material_name")
                or ann.get("name")
                or ann.get("label")
            )
            if raw_cls is None and "class_id" in ann:
                raw_cls = ann["class_id"]

            norm_cls = normalize_material_class_key(raw_cls)
            raw_str = str(raw_cls).strip().lower() if raw_cls is not None else norm_cls

            if raw_str in normalized_counts:
                cls_count = normalized_counts[raw_str]
            elif norm_cls in normalized_counts:
                cls_count = normalized_counts[norm_cls]
            else:
                cls_count = 0

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
                or (label_candidate and is_rejected_non_material(label_candidate))
            )
            if explicitly_rejected:
                rejected_detections.append(det)
                continue

            raw_conf = det.get("confidence", det.get("conf", det.get("score", 1.0)))
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 1.0
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
        elif uncertainty_triggered and not candidate_annotations:
            priority = float(np.clip(max(priority, 1.8), self.min_priority, self.max_priority))

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

        with self._lock:
            self.stats["frames_evaluated"] += 1
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
                    frame=frame.copy() if isinstance(frame, np.ndarray) and frame.size > 0 else None,
                )
                self._mined_history.append(record)
                if self.on_hard_example_callback is not None:
                    try:
                        self.on_hard_example_callback(record)
                    except Exception as exc:
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

        # Strict passive guarantee: bot never presses keys in Shadow Mode
        self.passive_shadow_enabled: bool = True
        self.bot_input_enabled: bool = False

        self._lock = threading.RLock()
        self._last_harvest_ts: float = -1e9
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

    def stop_background_worker(self, timeout: float = 2.0) -> None:
        """Gracefully drain and stop the background auto-labeling worker."""
        self._stop_event.set()
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            logger.debug("ShadowGameplayRecorder queue already full during stop signal.")
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.1, float(timeout)))

    def flush_pending_tasks(self, timeout: float = 2.0) -> int:
        """Process any queued auto-labeling jobs synchronously and wait for queue completion."""
        deadline = time.time() + max(0.05, float(timeout))
        processed = 0
        while time.time() < deadline:
            try:
                job = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if job is None:
                self._task_queue.task_done()
                break
            try:
                self._process_capture_job(job)
                processed += 1
            finally:
                self._task_queue.task_done()
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
        resolved_conf = 0.88

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
                if not resolved_class_name:
                    resolved_class_name = normalize_material_class_key(
                        best_match.get("base_material") or best_match.get("name")
                    )
                raw_score = best_match.get("confidence", best_match.get("score", 0.85))
                resolved_conf = float(np.clip(max(0.76, float(raw_score)), 0.50, 0.99))

            # Localize the most salient chromatic/bright object contour inside the ROI
            hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
            sat_mask = cv2.inRange(hsv_roi, (0, 65, 65), (180, 255, 255))
            kernel = np.ones((5, 5), dtype=np.uint8)
            sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_box_px: Optional[Tuple[int, int, int, int]] = None
            best_area = 0.0
            min_contour_area = max(16.0, float(roi.shape[0] * roi.shape[1]) * 0.008)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area >= min_contour_area and area > best_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    best_area = area
                    best_box_px = (x1_roi + bx, y1_roi + by, bw, bh)

            if best_box_px is not None:
                px, py, pw, bh = best_box_px
                cx = float(np.clip((px + 0.5 * pw) / max(1, img_w), 0.05, 0.95))
                cy = float(np.clip((py + 0.5 * bh) / max(1, img_h), 0.05, 0.95))
                nw = float(np.clip(pw / max(1, img_w), 0.04, 0.45))
                nh = float(np.clip(bh / max(1, img_h), 0.04, 0.45))
            else:
                cx, cy, nw, nh = 0.50, 0.56, 0.14, 0.16
        else:
            cx, cy, nw, nh = 0.50, 0.56, 0.14, 0.16

        if not resolved_class_name:
            resolved_class_name = "herb_aria"
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
                "confidence": round(resolved_conf, 4),
                "source": "shadow_time_machine_auto_labeler",
            }
        ]

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
                or ann.get("class_id", 0)
            )
            class_name = normalize_material_class_key(raw_cls)
            class_id = resolve_class_id(ann.get("class_id", class_name))

            cx: Optional[float] = None
            cy: Optional[float] = None
            w: Optional[float] = None
            h: Optional[float] = None

            if "cx" in ann and "cy" in ann and ("w" in ann or "width" in ann) and ("h" in ann or "height" in ann):
                cx = float(ann["cx"])
                cy = float(ann["cy"])
                w = float(ann.get("w", ann.get("width", 0.1)))
                h = float(ann.get("h", ann.get("height", 0.1)))
                if cx > 1.0 or cy > 1.0 or w > 1.0 or h > 1.0:
                    cx /= max(1, img_w)
                    cy /= max(1, img_h)
                    w /= max(1, img_w)
                    h /= max(1, img_h)
            elif "yolo_bbox" in ann and len(ann["yolo_bbox"]) == 4:
                cx, cy, w, h = [float(v) for v in ann["yolo_bbox"]]
            elif "bbox_xyxy" in ann and len(ann["bbox_xyxy"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in ann["bbox_xyxy"]]
                cx = ((x1 + x2) * 0.5) / max(1, img_w)
                cy = ((y1 + y2) * 0.5) / max(1, img_h)
                w = abs(x2 - x1) / max(1, img_w)
                h = abs(y2 - y1) / max(1, img_h)
            elif "bbox" in ann and len(ann["bbox"]) == 4:
                b0, b1, b2, b3 = [float(v) for v in ann["bbox"]]
                if max(b0, b1, b2, b3) <= 1.0:
                    cx, cy, w, h = b0, b1, b2, b3
                else:
                    # Treat as (x, y, w_px, h_px) or (x1, y1, x2, y2)
                    if b2 > b0 and b3 > b1 and (b2 > img_w * 0.5 or b3 > img_h * 0.5):
                        cx = ((b0 + b2) * 0.5) / max(1, img_w)
                        cy = ((b1 + b3) * 0.5) / max(1, img_h)
                        w = abs(b2 - b0) / max(1, img_w)
                        h = abs(b3 - b1) / max(1, img_h)
                    else:
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
                cx, cy, w, h = 0.50, 0.56, 0.14, 0.16

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
                    "confidence": float(ann.get("confidence", ann.get("conf", 0.90))),
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

        with self._lock:
            self._recorded_samples.append(sample_record)
            self.stats["samples_auto_labeled"] += 1

        if callable(self.on_sample_recorded):
            try:
                self.on_sample_recorded(sample_record)
            except Exception as exc:
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
        elif evaluation["uncertainty_mining"] and cooldown_elapsed:
            should_rewind_time_machine = True
            trigger_source = "active_learning_uncertainty"
            with self._lock:
                self._last_harvest_ts = ts
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
