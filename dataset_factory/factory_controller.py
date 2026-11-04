# -*- coding: utf-8 -*-
"""
Unified Auto-Labeling Dataset Factory Controller (`AutoLabelingDatasetFactory`).
================================================================================
Unites all Dataset Factory components into a single, clean, thread-safe API that
hooks directly into `BotWorker` (`core/state_machine.py`) and `YOLOFactoryDialog`
(`ui/yolo_factory_dialog.py`):

1. `PreGatherTimeMachineBuffer` (`dataset_factory/frame_time_machine.py`):
   - Rolling ring buffer of gameplay frames sampled every `0.25s` over `6.0s`
     so that when a gather completes, the exact frame from `2.0 seconds earlier`
     (`t - 2.0s`, when the resource node was still intact in front of the player)
     is retrieved.
2. `YOLOAutoLabeler` (`dataset_factory/yolo_auto_labeler.py`):
   - Multi-class 3D + Minimap automatic bounding-box generator across all 12
     Gatherion YOLO classes (`0..5` 3D resource meshes below crests/icons,
     `6` 3D winged gather crest, `7` F Gather prompt, `8` E/F floating icon,
     `9` minimap 3-leaf sprout, `10` 3D monster negative, `11` minimap monster dot).
3. `DatasetQualityGate` (`dataset_factory/quality_dedup_gate.py`):
   - 64-bit perceptual dHash deduplication, Laplacian variance motion-blur gate,
     YOLO coordinate validation + class-aware NMS, and stratified 85%/15% train/val
     split assignment.
4. `SyntheticYOLOAugmentor` (`dataset_factory/synthetic_augmentor.py`):
   - Up to 10x photometric (day/night/dusk lighting, biome HSV jitter, Pilgrim's
     Ridge / Altamia Highland fog & bloom) and geometric (horizontal flip,
     multi-scale camera crop) synthetic multiplier with strict `[0.0, 1.0]`
     bounding-box transformation.
5. `DatasetCurator` (`dataset_factory/dataset_curator.py`):
   - Manages `dataset/images/train|val`, `dataset/labels/train|val`,
     `dataset/previews/`, `dataset/data.yaml`, and `dataset/manifest_stats.json`
     with Unicode-safe Windows file I/O (`selçuk` path compatible).
6. `RTXTrainerEngine` (`dataset_factory/rtx_trainer_engine.py`):
   - 1-Click RTX 3070 CUDA FP16 YOLO Trainer & ONNX Exporter (`models/yolov8n_gatherion.onnx`
     + `models/gatherion_custom_detector.json` + `train_rtx_yolo.bat`).
7. `ActiveLearningLoop` & `ShadowGameplayRecorder` (`dataset_factory/active_learning_loop.py`):
   - Uncertainty mining (`0.42 <= conf <= 0.74`), false-negative F8 mining,
     hard-negative extraction-failure mining, class-rarity prioritization (`1.0..3.0`),
     and Passive Shadow Recording Mode (`"Pasif Gölge Kayıt Modu"`).

Exclusive Owner: Dataset Factory Sub-Agent 8 (`dataset_factory/factory_controller.py`)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dataset_factory.active_learning_loop import (
    ActiveLearningLoop,
    HardExampleRecord,
    ShadowGameplayRecorder,
)
from dataset_factory.dataset_curator import (
    DatasetCurator,
    imread_unicode,
    imwrite_unicode,
)
from dataset_factory.frame_time_machine import (
    FrameSnapshot,
    PreGatherTimeMachineBuffer,
)
from dataset_factory.quality_dedup_gate import (
    DatasetQualityGate,
    QualityGateDecision,
)
from dataset_factory.rtx_trainer_engine import (
    RTXTrainerEngine,
    probe_gpu_environment,
)
from dataset_factory.synthetic_augmentor import SyntheticYOLOAugmentor
from dataset_factory.yolo_auto_labeler import (
    YOLO_CLASS_COLORS_BGR,
    YOLO_CLASSES,
    YOLO_CLASSES_TR,
    YOLOAutoLabeler,
    map_material_name_to_yolo_class_id,
)

logger = logging.getLogger("DatasetFactory.Controller")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# Unified 12-Class YOLO Schema (Compatible with YOLOAutoLabeler & YOLOFactoryDialog)
# ==============================================================================
FACTORY_12_CLASS_NAMES: Dict[int, str] = {
    0: "herb_aria",
    1: "herb_mela",
    2: "herb_targena",
    3: "ore_orichalcum",
    4: "gem_crystal",
    5: "wood_tree",
    6: "world_gather_crest",
    7: "f_gather_prompt",
    8: "e_gather_icon",
    9: "minimap_sprout",
    10: "monster_3d_negative",
    11: "minimap_monster_negative",
}

FACTORY_12_CLASS_LABELS_TR: Dict[int, str] = {
    0: "0: Aria (Mor Kristal Bitki)",
    1: "1: Mela (Kırmızı Meyveli Çalı)",
    2: "2: Targena / Azpha / Calendula",
    3: "3: Orichalcum / Maden Cevheri",
    4: "4: Gem / Kristal (Sapphire/Ruby/Diamond)",
    5: "5: Ağaç / Kereste (Yggdrasil/Asvata)",
    6: "6: 3B Amblem (Beyaz Kanatlı İkon)",
    7: "7: F Gather (Etkileşim İstem Kutusu)",
    8: "8: E İkonu (Yüzen Kaynak Kutusu)",
    9: "9: Minimap Filiz (3 Yapraklı İkon)",
    10: "10: Canavar (3B Negatif Sınıf)",
    11: "11: Minimap Canavar (Yuvarlak Nokta Negatif)",
}


class AutoLabelingDatasetFactory:
    """
    Unified Auto-Labeling Dataset Factory Controller (`AutoLabelingDatasetFactory`).

    Integrates:
      - `PreGatherTimeMachineBuffer` (rolling 6.0s buffer at 0.25s intervals for `t - 2.0s` lookback)
      - `YOLOAutoLabeler` (12-class 3D + Minimap + UI detector & contour segmenter)
      - `DatasetQualityGate` (64-bit dHash deduplicator + Laplacian sharpness gate + NMS validator)
      - `SyntheticYOLOAugmentor` (up to 10x photometric & geometric YOLO augmentor)
      - `DatasetCurator` (`dataset/` manager, `data.yaml` writer, preview renderer & bootstrap seeder)
      - `RTXTrainerEngine` (1-Click RTX 3070 CUDA FP16 trainer & ONNX exporter)
      - `ActiveLearningLoop` & `ShadowGameplayRecorder` (hard-example mining & passive shadow mode)
    """

    DEFAULT_TARGET_IMAGES: int = 5000
    DEFAULT_LOOKBACK_SEC: float = 2.0
    DEFAULT_PUSH_INTERVAL_SEC: float = 0.25

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        auto_train_threshold: int = DEFAULT_TARGET_IMAGES,
        lookback_sec: float = DEFAULT_LOOKBACK_SEC,
        push_interval_sec: float = DEFAULT_PUSH_INTERVAL_SEC,
        auto_label_on_gather: bool = True,
        auto_augment_on_gather: bool = True,
        augment_variants_per_gather: int = 4,
        passive_shadow_mode: bool = False,
        auto_trigger_rtx_training: bool = False,
        min_sharpness_threshold: float = 18.0,
        min_hamming_distance: int = 3,
        val_ratio: float = 0.15,
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        if dataset_dir is None:
            self.dataset_dir = self.project_root / "dataset"
        else:
            self.dataset_dir = Path(dataset_dir).resolve()

        if models_dir is None:
            self.models_dir = self.project_root / "models"
        else:
            self.models_dir = Path(models_dir).resolve()

        self._lock = threading.RLock()

        # Configurable thresholds & feature toggles
        self.auto_train_threshold: int = max(1, int(auto_train_threshold))
        self.target_photo_goal: int = self.auto_train_threshold
        self.lookback_sec: float = max(0.25, float(lookback_sec))
        self.push_interval_sec: float = max(0.05, float(push_interval_sec))

        self.auto_label_on_gather: bool = bool(auto_label_on_gather)
        self.auto_label_pre_gather: bool = self.auto_label_on_gather
        self.auto_label_enabled: bool = self.auto_label_on_gather

        self.auto_augment_on_gather: bool = bool(auto_augment_on_gather)
        self.synthetic_augment_10x: bool = self.auto_augment_on_gather
        self.augment_10x_enabled: bool = self.auto_augment_on_gather
        self.augment_variants_per_gather: int = int(np.clip(augment_variants_per_gather, 1, 10))

        self.passive_shadow_mode: bool = bool(passive_shadow_mode)
        self.shadow_mode_enabled: bool = self.passive_shadow_mode
        self.auto_trigger_rtx_training: bool = bool(auto_trigger_rtx_training)

        # 1. Pre-Gather Time Machine Ring Buffer (sampled every 0.25s over 6.0s)
        self.time_machine = PreGatherTimeMachineBuffer(
            max_duration_sec=max(6.0, self.lookback_sec * 3.0),
            min_push_interval_sec=self.push_interval_sec,
            copy_frames=True,
            max_capacity=240,
        )

        # 2. Multi-Class 3D + Minimap Automatic YOLO Labeler
        self.auto_labeler = YOLOAutoLabeler()

        # 3. Perceptual dHash Deduplicator & Sharpness Quality Gate
        self.quality_gate = DatasetQualityGate(
            min_hamming_distance=int(min_hamming_distance),
            min_sharpness_threshold=float(min_sharpness_threshold),
            val_ratio=float(val_ratio),
        )

        # 4. 10x Photometric & Geometric Synthetic YOLO Augmentor
        self.augmentor = SyntheticYOLOAugmentor(seed=42)

        # 5. Dataset Curator (`dataset/` structure, `data.yaml`, previews, manifest)
        self.curator = DatasetCurator(
            project_root=self.project_root,
            dataset_dir=self.dataset_dir,
            target_goal=self.auto_train_threshold,
            class_names=dict(FACTORY_12_CLASS_NAMES),
            val_ratio=float(val_ratio),
            auto_labeler=self.auto_labeler,
            augmentor=self.augmentor,
            quality_gate=self.quality_gate,
        )

        # Convenience directory attributes for UI compatibility
        self.train_images_dir: Path = self.curator.images_train_dir
        self.val_images_dir: Path = self.curator.images_val_dir
        self.train_labels_dir: Path = self.curator.labels_train_dir
        self.val_labels_dir: Path = self.curator.labels_val_dir
        self.previews_dir: Path = self.curator.previews_dir
        self.data_yaml_path: Path = self.curator.data_yaml_path

        # 6. 1-Click RTX 3070 CUDA FP16 YOLO Trainer & ONNX Exporter
        self.rtx_trainer = RTXTrainerEngine(
            project_root=self.project_root,
            dataset_dir=self.dataset_dir,
            models_dir=self.models_dir,
        )
        self.trainer = self.rtx_trainer

        # 7. Active Learning Loop & Passive Shadow Gameplay Recorder
        self.active_learning = ActiveLearningLoop(
            uncertainty_min_conf=0.42,
            uncertainty_max_conf=0.74,
            min_priority=1.0,
            max_priority=3.0,
        )
        self.shadow_recorder = ShadowGameplayRecorder(
            time_machine=self.time_machine,
            active_learning_loop=self.active_learning,
            auto_labeler=self._shadow_auto_label_callback,
            on_sample_recorded=self._on_shadow_sample_recorded,
            lookback_sec=self.lookback_sec,
            background_worker=False,
        )

        # Runtime counters and state tracking
        self._gather_events_total: int = 0
        self._gather_samples_accepted: int = 0
        self._gather_samples_rejected: int = 0
        self._synthetic_variants_saved: int = 0
        self._active_learning_saves: int = 0
        self._last_saved_preview_path: str = ""
        self._last_saved_image_path: str = ""
        self._last_gather_result: Dict[str, Any] = {}
        self._auto_train_triggered_count: int = 0
        self._cached_gpu_info: Optional[Dict[str, Any]] = None
        self._cached_gpu_ts: float = 0.0

        # Ensure initial dataset directories, data.yaml, and initial preview exist
        self.curator.ensure_dataset_structure()
        self._sync_initial_state()

    # ==========================================================================
    # Internal Initialization & Synchronization Helpers
    # ==========================================================================
    def _sync_initial_state(self) -> None:
        """Sync preview path and class counts from existing `dataset/` files on disk."""
        with self._lock:
            manifest = self.curator.rebuild_manifest_stats()
            class_id_counts = manifest.get("class_id_counts", {})
            mapped_counts: Dict[Union[int, str], int] = {}
            for cid_str, cnt in class_id_counts.items():
                try:
                    cid = int(cid_str)
                    cname = FACTORY_12_CLASS_NAMES.get(cid, f"class_{cid}")
                    mapped_counts[cname] = int(cnt)
                except ValueError:
                    continue
            if mapped_counts:
                self.active_learning.update_class_counts(mapped_counts)

            latest_preview = self._find_latest_preview_on_disk()
            if latest_preview:
                self._last_saved_preview_path = str(latest_preview)

    def _find_latest_preview_on_disk(self) -> Optional[Path]:
        """Return the most recently modified preview image in `dataset/previews/`."""
        if not self.previews_dir.exists():
            return None
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        previews = [
            p
            for p in self.previews_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        if not previews:
            return None
        previews.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return previews[0]

    def _get_gpu_info_cached(self, max_age_sec: float = 15.0) -> Dict[str, Any]:
        """Return cached GPU probe dictionary to avoid repeated subprocess calls on fast UI polls."""
        now = time.time()
        with self._lock:
            if self._cached_gpu_info is None or (now - self._cached_gpu_ts) > max_age_sec:
                self._cached_gpu_info = probe_gpu_environment()
                self._cached_gpu_ts = now
            return dict(self._cached_gpu_info)

    def set_passive_shadow_mode(self, enabled: bool) -> None:
        """Enable or disable Passive Shadow Recording Mode (`Pasif Gölge Kayıt Modu`)."""
        with self._lock:
            self.passive_shadow_mode = bool(enabled)
            self.shadow_mode_enabled = self.passive_shadow_mode
            self.shadow_recorder.passive_shadow_enabled = self.passive_shadow_mode

    def _shadow_auto_label_callback(
        self,
        frame: np.ndarray,
        material_hint: str = "",
        region_name: str = "",
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Adapter callback used by `ShadowGameplayRecorder` to invoke `YOLOAutoLabeler`."""
        return self.auto_labeler.auto_label_frame(
            frame=frame,
            extra_detections=detections,
            known_material_name=material_hint,
        )

    def _on_shadow_sample_recorded(self, sample_record: Dict[str, Any]) -> None:
        """Persist auto-labeled samples captured by `ShadowGameplayRecorder` into `dataset/`."""
        frame = sample_record.get("frame")
        annotations = sample_record.get("annotations", [])
        if not isinstance(frame, np.ndarray) or frame.size == 0 or not annotations:
            return

        decision = self.quality_gate.evaluate_frame(
            frame=frame,
            annotations=annotations,
            record_on_accept=True,
        )
        if not decision.accepted:
            return

        valid_anns = decision.validated_annotations or annotations
        save_res = self.curator.save_labeled_sample(
            frame=frame,
            annotations=valid_anns,
            split=decision.split,
            prefix="shadow_t2s",
            save_preview=True,
            update_manifest=True,
        )
        with self._lock:
            self._active_learning_saves += 1
            if save_res.get("preview_path"):
                self._last_saved_preview_path = str(save_res["preview_path"])
            if save_res.get("image_path"):
                self._last_saved_image_path = str(save_res["image_path"])

    # ==========================================================================
    # 1. Live Frame Ingestion (`ingest_live_frame`)
    # ==========================================================================
    def ingest_live_frame(
        self,
        frame: np.ndarray,
        player_pos: Tuple[float, float] = (0.0, 0.0),
        detections: Optional[List[Dict[str, Any]]] = None,
        region_name: str = "",
        *,
        timestamp: Optional[float] = None,
        user_harvested: bool = False,
        manual_f_pressed: bool = False,
        manual_f8_pressed: bool = False,
        extraction_failed: bool = False,
        status_text: str = "",
        material_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Called on each bot/vision tick to:
          1. Push the current frame into `PreGatherTimeMachineBuffer` (throttled to every `0.25s`
             unless `force` or explicit event flags are active).
          2. Check `ActiveLearningLoop` (and `ShadowGameplayRecorder`) if passive shadow recording
             (`passive_shadow_mode`) or active-learning event triggers (`manual_f8_pressed`,
             `extraction_failed`, borderline confidence detections) are present.

        Returns:
            Dictionary describing whether the frame was buffered in the Time Machine and
            any Active Learning / Shadow Recording actions taken.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "ingested": False,
                "reason": "empty_or_invalid_frame",
                "buffered_frames": len(self.time_machine),
                "span_seconds": self.time_machine.span_seconds,
            }

        ts = float(timestamp) if timestamp is not None else time.time()

        # 1. If Passive Shadow Mode is enabled, delegate observation to ShadowGameplayRecorder
        #    (which pushes into self.time_machine and evaluates self.active_learning).
        if self.passive_shadow_mode:
            shadow_res = self.shadow_recorder.observe_frame(
                frame=frame,
                timestamp=ts,
                player_pos=player_pos,
                detections=detections,
                region_name=region_name,
                user_harvested=user_harvested,
                manual_f_pressed=manual_f_pressed,
                manual_f8_pressed=manual_f8_pressed,
                extraction_failed=extraction_failed,
                material_hint=material_hint,
                status_text=status_text,
                sync_label=True,
            )
            return {
                "ingested": True,
                "mode": "passive_shadow",
                "buffered_frames": len(self.time_machine),
                "span_seconds": round(self.time_machine.span_seconds, 3),
                "shadow_result": shadow_res,
                "active_learning_triggered": bool(shadow_res.get("triggered", False)),
            }

        # 2. Standard Live Tick Mode: Push frame into PreGatherTimeMachineBuffer (every 0.25s)
        snapshot = self.time_machine.push_frame(
            frame=frame,
            timestamp=ts,
            player_pos=player_pos,
            detections=detections,
            region_name=region_name,
            metadata={"material_hint": material_hint} if material_hint else None,
        )

        # 3. Evaluate ActiveLearningLoop if detections or explicit mining triggers are present
        al_decision: Optional[Dict[str, Any]] = None
        mined_saved = False
        if (
            detections
            or manual_f8_pressed
            or manual_f_pressed
            or extraction_failed
            or status_text
        ):
            al_decision = self.active_learning.evaluate_frame_for_active_learning(
                frame=frame,
                detections=detections,
                extraction_failed=extraction_failed,
                manual_f8_pressed=manual_f8_pressed,
                manual_f_pressed=manual_f_pressed,
                status_text=status_text,
                region_name=region_name,
                player_pos=player_pos,
                lookback_sec=0.0,
            )
            # If user manually pressed F8 to mine a false negative, auto-label & save immediately
            if manual_f8_pressed and al_decision.get("should_capture"):
                anns = self.auto_labeler.auto_label_frame(
                    frame=frame,
                    extra_detections=detections,
                    known_material_name=material_hint,
                )
                if anns:
                    q_dec = self.quality_gate.evaluate_frame(frame, anns, record_on_accept=True)
                    if q_dec.accepted:
                        saved_info = self.curator.save_labeled_sample(
                            frame=frame,
                            annotations=q_dec.validated_annotations or anns,
                            split=q_dec.split,
                            prefix="f8_active_mine",
                            save_preview=True,
                            update_manifest=True,
                        )
                        mined_saved = bool(saved_info.get("saved", False))
                        with self._lock:
                            self._active_learning_saves += 1
                            if saved_info.get("preview_path"):
                                self._last_saved_preview_path = str(saved_info["preview_path"])
                            if saved_info.get("image_path"):
                                self._last_saved_image_path = str(saved_info["image_path"])

        return {
            "ingested": snapshot is not None,
            "throttled": snapshot is None,
            "timestamp": ts,
            "buffered_frames": len(self.time_machine),
            "span_seconds": round(self.time_machine.span_seconds, 3),
            "active_learning": al_decision,
            "mined_sample_saved": mined_saved,
        }

    # ==========================================================================
    # 2. Pre-Gather Time-Machine Auto-Labeling (`on_gather_success`)
    # ==========================================================================
    def on_gather_success(
        self,
        current_frame: Optional[np.ndarray] = None,
        node_info: Optional[Dict[str, Any]] = None,
        material_name: str = "",
        harvest_stage: int = 1,
        *,
        lookback_sec: Optional[float] = None,
        reference_ts: Optional[float] = None,
        force_save: bool = False,
    ) -> Dict[str, Any]:
        """
        THE KEY FEATURE: Called automatically every time the bot (or player) completes a gather!

        Workflow:
          1. Retrieves the **2.0 seconds earlier (`t - 2.0s`) frame** from
             `PreGatherTimeMachineBuffer` (falling back to `current_frame` if the buffer
             was just started or empty).
          2. Runs `YOLOAutoLabeler.auto_label_frame(...)` on that `t - 2.0s` frame,
             incorporating any `node_info` target coordinates and `material_name` hints.
          3. Checks `DatasetQualityGate` (64-bit dHash deduplication + Laplacian sharpness
             check + YOLO box validation & class-aware NMS + deterministic train/val split).
          4. Saves the raw image + YOLO `.txt` label into `dataset/images/train|val` and
             `dataset/labels/train|val` (plus an annotated bounding-box preview in
             `dataset/previews/`).
          5. Generates synthetic augmented variants (`SyntheticYOLOAugmentor`) if
             `auto_augment_on_gather=True` (or `synthetic_augment_10x=True`), scaled by
             `ActiveLearningLoop.compute_sample_priority(...)`.
          6. If `total_images >= auto_train_threshold` (default `5000`), flags
             `auto_train_ready=True` and optionally triggers 1-Click RTX training!
        """
        with self._lock:
            self._gather_events_total += 1
            gather_seq = self._gather_events_total

        if not self.auto_label_on_gather and not force_save:
            return {
                "saved": False,
                "reason": "auto_label_on_gather_disabled",
                "gather_event_index": gather_seq,
            }

        target_lookback = float(lookback_sec) if lookback_sec is not None else self.lookback_sec

        # Step 1: Retrieve the t - 2.0s frame from PreGatherTimeMachineBuffer
        snapshot: Optional[FrameSnapshot] = self.time_machine.get_frame_before(
            seconds_ago=target_lookback,
            reference_ts=reference_ts,
        )

        used_time_machine = False
        actual_lookback_sec = 0.0
        cached_detections: List[Dict[str, Any]] = []
        region_name = ""
        player_pos: Tuple[float, float] = (0.0, 0.0)

        if snapshot is not None and snapshot.frame is not None and snapshot.frame.size > 0:
            target_frame = snapshot.frame
            used_time_machine = True
            ref_clock = (
                float(reference_ts)
                if reference_ts is not None
                else (
                    self.time_machine.get_latest_frame().timestamp
                    if self.time_machine.get_latest_frame() is not None
                    else time.time()
                )
            )
            actual_lookback_sec = max(0.0, ref_clock - snapshot.timestamp)
            cached_detections = list(snapshot.detections or [])
            region_name = str(snapshot.region_name or "")
            player_pos = snapshot.player_pos
        elif current_frame is not None and isinstance(current_frame, np.ndarray) and current_frame.size > 0:
            target_frame = current_frame
            used_time_machine = False
            actual_lookback_sec = 0.0
        else:
            with self._lock:
                self._gather_samples_rejected += 1
            return {
                "saved": False,
                "reason": "no_frame_available_in_buffer_or_argument",
                "gather_event_index": gather_seq,
            }

        # Extract optional hints from node_info
        node_dict = dict(node_info) if isinstance(node_info, dict) else {}
        resolved_material = str(
            material_name
            or node_dict.get("material_name")
            or node_dict.get("material_type")
            or node_dict.get("name")
            or node_dict.get("type")
            or ""
        ).strip()
        if not region_name and node_dict.get("region_name"):
            region_name = str(node_dict["region_name"])

        known_target_xy: Optional[Tuple[int, int]] = None
        if "screen_x" in node_dict and "screen_y" in node_dict:
            try:
                known_target_xy = (int(node_dict["screen_x"]), int(node_dict["screen_y"]))
            except (TypeError, ValueError):
                known_target_xy = None
        elif "target_xy" in node_dict and isinstance(node_dict["target_xy"], (list, tuple)) and len(node_dict["target_xy"]) >= 2:
            known_target_xy = (int(node_dict["target_xy"][0]), int(node_dict["target_xy"][1]))

        # Step 2: Run YOLOAutoLabeler on the t - 2.0s frame
        annotations = self.auto_labeler.auto_label_frame(
            frame=target_frame,
            extra_detections=cached_detections if cached_detections else None,
            known_target_xy=known_target_xy,
            known_material_name=resolved_material,
        )

        # Fallback to DatasetCurator's salient 3D patch detector if frame had no crest/prompt templates
        if not annotations:
            annotations = self.curator.auto_label_frame(target_frame)

        # Ensure if a specific material_name was gathered (e.g. "Aria", "Mela", "Orichalcum"),
        # at least one 3D resource mesh box (0..5) reflects the gathered material class
        if resolved_material:
            expected_cid = map_material_name_to_yolo_class_id(resolved_material, default_id=0)
            has_mesh_box = any(0 <= int(a.get("class_id", -1)) <= 5 for a in annotations)
            if not has_mesh_box and 0 <= expected_cid <= 5:
                img_h, img_w = target_frame.shape[:2]
                annotations.append(
                    {
                        "class_id": int(expected_cid),
                        "class_name": FACTORY_12_CLASS_NAMES.get(expected_cid, "herb_aria"),
                        "cx": 0.50,
                        "cy": 0.55,
                        "w": 0.11,
                        "h": 0.15,
                        "confidence": 0.92,
                        "source": "gather_material_hint",
                    }
                )
            elif has_mesh_box and 0 <= expected_cid <= 5:
                # Align primary 3D mesh class_id with confirmed gathered material name
                for ann in annotations:
                    if 0 <= int(ann.get("class_id", -1)) <= 5 and ann.get("source") in (
                        "hsv_3d_patch_fallback",
                        "center_anchor_fallback",
                    ):
                        ann["class_id"] = int(expected_cid)
                        ann["class_name"] = FACTORY_12_CLASS_NAMES.get(expected_cid, "herb_aria")

        # Step 3: Check DatasetQualityGate (dHash deduplication + Laplacian sharpness check)
        gate_decision: QualityGateDecision = self.quality_gate.evaluate_frame(
            frame=target_frame,
            annotations=annotations,
            record_on_accept=True,
        )

        if not gate_decision.accepted and not force_save:
            with self._lock:
                self._gather_samples_rejected += 1
            return {
                "saved": False,
                "reason": gate_decision.reason,
                "sharpness": round(gate_decision.sharpness, 2),
                "dhash": gate_decision.dhash,
                "min_hamming_seen": gate_decision.min_hamming_seen,
                "used_time_machine": used_time_machine,
                "actual_lookback_sec": round(actual_lookback_sec, 3),
                "gather_event_index": gather_seq,
            }

        validated_anns = gate_decision.validated_annotations or annotations
        for ann in validated_anns:
            cid = int(ann.get("class_id", 0))
            ann["class_name"] = FACTORY_12_CLASS_NAMES.get(
                cid, str(ann.get("class_name", f"class_{cid}"))
            )

        split = gate_decision.split if gate_decision.split in ("train", "val") else "train"

        # Compute Active Learning sample priority (1.0 .. 3.0) based on class rarity
        sample_priority = self.active_learning.compute_sample_priority(validated_anns)
        for ann in validated_anns:
            self.active_learning.increment_class_count(ann.get("class_name", ann.get("class_id", 0)))

        # Step 4: Save the base t - 2.0s raw image + YOLO .txt label + annotated preview
        mat_slug = re.sub(
            r"[^a-z0-9]+",
            "_",
            (resolved_material or "node").lower(),
        ).strip("_") or "node"
        ts_ms = int(time.time() * 1000) % 10000000
        base_stem = f"gather_t2s_{mat_slug}_s{int(harvest_stage)}_{ts_ms}_{gather_seq:04d}"

        base_save_res = self.curator.save_labeled_sample(
            frame=target_frame,
            annotations=validated_anns,
            split=split,
            prefix="gather_t2s",
            save_preview=True,
            sample_name=base_stem,
            update_manifest=False,
        )

        # Also save a richly labeled preview using YOLOAutoLabeler's bilingual renderer
        preview_path_str = str(base_save_res.get("preview_path") or "")
        try:
            annotated_bgr = self.auto_labeler.draw_annotated_preview(target_frame, validated_anns)
            latest_preview_file = self.previews_dir / "latest_t_minus_2s_preview.jpg"
            if imwrite_unicode(latest_preview_file, annotated_bgr, jpeg_quality=93):
                if not preview_path_str:
                    preview_path_str = str(latest_preview_file)
        except Exception as exc:
            logger.debug("Optional latest preview render skipped: %s", exc)

        # Step 5: Generate and save synthetic augmented variants if enabled
        augmented_saved_paths: List[str] = []
        augmented_boxes_count = 0
        do_augment = bool(self.auto_augment_on_gather and self.synthetic_augment_10x)

        if do_augment and validated_anns:
            # Scale variant count by class rarity priority (up to 9 additional variants = 10x total)
            if self.augment_variants_per_gather >= 9:
                variant_count = 9
            else:
                variant_count = int(
                    np.clip(
                        round(self.augment_variants_per_gather * (sample_priority / 2.0)),
                        2,
                        9,
                    )
                )

            variants = self.augmentor.generate_augmented_variants(
                frame=target_frame,
                annotations=validated_anns,
                count=variant_count,
            )
            for v_idx, (v_suffix, v_frame, v_anns) in enumerate(variants):
                if not v_anns:
                    continue
                v_split = DatasetQualityGate.choose_split(
                    gather_seq * 10 + v_idx + 1,
                    val_ratio=self.quality_gate.val_ratio,
                )
                v_stem = f"{base_stem}_{v_suffix}"
                v_res = self.curator.save_labeled_sample(
                    frame=v_frame,
                    annotations=v_anns,
                    split=v_split,
                    prefix="gather_aug",
                    save_preview=False,
                    sample_name=v_stem,
                    jpeg_quality=91,
                    update_manifest=False,
                )
                if v_res.get("saved"):
                    augmented_saved_paths.append(str(v_res["image_path"]))
                    augmented_boxes_count += int(v_res.get("box_count", 0))

        # Rebuild manifest_stats.json once after saving base + variants
        manifest = self.curator.rebuild_manifest_stats()
        total_images_now = int(manifest.get("total_images", 0))
        auto_train_ready = total_images_now >= self.auto_train_threshold

        auto_train_launched = False
        auto_train_info: Optional[Dict[str, Any]] = None
        if auto_train_ready and self.auto_trigger_rtx_training:
            trainer_state = self.rtx_trainer.get_status().get("state", "idle")
            if trainer_state != "running":
                auto_train_info = self.rtx_trainer.start_training_async(
                    data_yaml_path=self.data_yaml_path,
                    epochs=50,
                )
                auto_train_launched = bool(auto_train_info.get("started", False))
                if auto_train_launched:
                    with self._lock:
                        self._auto_train_triggered_count += 1

        result_payload: Dict[str, Any] = {
            "saved": True,
            "gather_event_index": gather_seq,
            "harvest_stage": int(harvest_stage),
            "material_name": resolved_material or "auto_detected",
            "region_name": region_name,
            "player_pos": player_pos,
            "used_time_machine": used_time_machine,
            "target_lookback_sec": target_lookback,
            "actual_lookback_sec": round(actual_lookback_sec, 3),
            "split": split,
            "sharpness": round(gate_decision.sharpness, 2),
            "dhash": gate_decision.dhash,
            "sample_priority": round(sample_priority, 3),
            "image_path": str(base_save_res.get("image_path", "")),
            "label_path": str(base_save_res.get("label_path", "")),
            "preview_path": preview_path_str,
            "box_count": len(validated_anns),
            "annotations": validated_anns,
            "augmented_variants_saved": len(augmented_saved_paths),
            "augmented_image_paths": augmented_saved_paths,
            "total_samples_added_this_gather": 1 + len(augmented_saved_paths),
            "total_boxes_added_this_gather": len(validated_anns) + augmented_boxes_count,
            "total_images": total_images_now,
            "target_images": self.auto_train_threshold,
            "progress_pct": float(manifest.get("progress_pct", 0.0)),
            "auto_train_ready": auto_train_ready,
            "auto_train_launched": auto_train_launched,
            "auto_train_info": auto_train_info,
            "message_tr": (
                f"Toplamadan {actual_lookback_sec:.1f} sn önceki kare otomatik etiketlendi "
                f"({len(validated_anns)} kutu, +{len(augmented_saved_paths)} sentetik varyant). "
                f"İlerleme: {total_images_now} / {self.auto_train_threshold}."
            ),
        }

        with self._lock:
            self._gather_samples_accepted += 1
            self._synthetic_variants_saved += len(augmented_saved_paths)
            if preview_path_str:
                self._last_saved_preview_path = preview_path_str
            if base_save_res.get("image_path"):
                self._last_saved_image_path = str(base_save_res["image_path"])
            self._last_gather_result = result_payload

        return result_payload

    # ==========================================================================
    # 3. Live Factory Status (`get_factory_status`)
    # ==========================================================================
    def get_factory_status(self) -> Dict[str, Any]:
        """
        Returns live statistics in Turkish & structured JSON:
          - `total_images`, `target_images: 5000`, `progress_pct`
          - `train_images`, `val_images`, `total_boxes`
          - `class_counts` (canonical 12 class names -> box count, plus `class_counts_by_id` & `class_counts_tr`)
          - `gpu_info` (hardware probe from `RTXTrainerEngine`)
          - `last_saved_preview_path` (plus `latest_preview_path` alias)
          - `is_training`, `training_metrics`
          - Turkish diagnostic summaries (`status_tr`, `summary_tr`, `time_machine_status_tr`, `gpu_status_tr`)
        """
        with self._lock:
            manifest = self.curator.rebuild_manifest_stats()
            trainer_status = self.rtx_trainer.get_status()
            gpu_info = self._get_gpu_info_cached()
            tm_stats = self.time_machine.get_stats()
            al_stats = self.active_learning.get_stats()

            train_images = int(manifest.get("train_count", 0))
            val_images = int(manifest.get("val_count", 0))
            total_images = int(manifest.get("total_images", train_images + val_images))
            target_images = int(self.auto_train_threshold)
            progress_pct = round(
                min(100.0, (float(total_images) / float(max(1, target_images))) * 100.0),
                2,
            )

            raw_id_counts = manifest.get("class_id_counts", {})
            class_counts: Dict[str, int] = {}
            class_counts_by_id: Dict[int, int] = {}
            class_counts_tr: Dict[str, int] = {}

            for cid in range(12):
                cnt = int(raw_id_counts.get(str(cid), raw_id_counts.get(cid, 0)))
                cname = FACTORY_12_CLASS_NAMES[cid]
                ctr = FACTORY_12_CLASS_LABELS_TR[cid]
                class_counts[cname] = cnt
                class_counts_by_id[cid] = cnt
                class_counts_tr[ctr] = cnt

            total_boxes = max(
                int(manifest.get("total_boxes", 0)),
                sum(class_counts_by_id.values()),
            )

            # Resolve latest preview image path
            preview_path = self._last_saved_preview_path
            if not preview_path or not Path(preview_path).exists():
                disk_preview = self._find_latest_preview_on_disk()
                if disk_preview is not None:
                    preview_path = str(disk_preview)
                    self._last_saved_preview_path = preview_path

            is_training = str(trainer_status.get("state", "idle")).lower() == "running"
            auto_train_ready = total_images >= target_images

            gpu_name = str(gpu_info.get("gpu_name", "NVIDIA RTX GPU"))
            vram_mb = int(gpu_info.get("vram_total_mb", 0))
            fp16_ok = bool(gpu_info.get("fp16_supported", True))

            if is_training:
                curr_ep = int(trainer_status.get("epoch", 0))
                tot_ep = int(trainer_status.get("epochs", 50))
                map50_val = float(trainer_status.get("mAP50", 0.0)) * 100.0
                status_tr = (
                    f"RTX CUDA FP16 Eğitimi Çalışıyor (Epoch {curr_ep}/{tot_ep} | mAP50: %{map50_val:.1f})"
                )
            elif auto_train_ready:
                status_tr = (
                    f"Hedef {target_images} Fotoğrafa Ulaşıldı! Tek Tıkla RTX CUDA FP16 Eğitimine Hazır."
                )
            else:
                status_tr = (
                    f"Otomatik Etiketlenen Veri Seti İlerlemesi: {total_images} / {target_images} Fotoğraf (%{progress_pct:.1f})"
                )

            tm_status_tr = (
                f"Toplamadan {self.lookback_sec:.1f} sn Önceki Kare Yakalayıcı: AKTİF "
                f"({tm_stats['buffered_frames']} kare, {tm_stats['span_seconds']:.1f} sn tampon)"
                if self.auto_label_on_gather
                else "Toplamadan 2.0 sn Önceki Kare Yakalayıcı: DURAKLATILDI"
            )

            gpu_status_tr = (
                f"{gpu_name} ({vram_mb} MB VRAM | {'FP16 Tensor Hazır' if fp16_ok else 'CPU Modu'})"
                if vram_mb > 0
                else f"{gpu_name} (RTX / CUDA FP16 Hazır)"
            )

            return {
                # Required primary keys
                "total_images": total_images,
                "target_images": target_images,
                "progress_pct": progress_pct,
                "train_images": train_images,
                "val_images": val_images,
                "total_boxes": total_boxes,
                "class_counts": class_counts,
                "gpu_info": gpu_info,
                "last_saved_preview_path": preview_path,
                "is_training": is_training,
                "training_metrics": trainer_status,
                # Extended & Turkish keys + UI compatibility aliases
                "total_frames": total_images,
                "train_frames": train_images,
                "val_frames": val_images,
                "train_count": train_images,
                "val_count": val_images,
                "target_photo_goal": target_images,
                "target_goal": target_images,
                "latest_preview_path": preview_path,
                "last_saved_image_path": self._last_saved_image_path,
                "class_counts_by_id": class_counts_by_id,
                "class_counts_tr": class_counts_tr,
                "gpu_environment": gpu_info,
                "status_tr": status_tr,
                "summary_tr": (
                    f"Toplam {total_images}/{target_images} etiketli kare "
                    f"(Train: {train_images}, Val: {val_images}) ve 12 sınıfta {total_boxes} YOLO kutusu."
                ),
                "time_machine_status_tr": tm_status_tr,
                "gpu_status_tr": gpu_status_tr,
                "time_machine_active": self.auto_label_on_gather,
                "auto_label_on_gather": self.auto_label_on_gather,
                "auto_augment_on_gather": self.auto_augment_on_gather,
                "synthetic_augment_10x": self.synthetic_augment_10x,
                "passive_shadow_mode": self.passive_shadow_mode,
                "auto_train_ready": auto_train_ready,
                "gather_events_total": self._gather_events_total,
                "gather_samples_accepted": self._gather_samples_accepted,
                "gather_samples_rejected": self._gather_samples_rejected,
                "synthetic_variants_saved": self._synthetic_variants_saved,
                "active_learning_saves": self._active_learning_saves,
                "time_machine": tm_stats,
                "quality_gate": dict(self.quality_gate.stats),
                "active_learning": al_stats,
                "data_yaml_path": str(self.data_yaml_path),
                "dataset_dir": str(self.dataset_dir),
            }

    # Aliases expected by `YOLOFactoryDialog` and external callers
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Alias for `get_factory_status()` used by `YOLOFactoryDialog`."""
        return self.get_factory_status()

    def get_stats(self) -> Dict[str, Any]:
        """Alias for `get_factory_status()`."""
        return self.get_factory_status()

    def get_summary(self) -> Dict[str, Any]:
        """Alias for `get_factory_status()`."""
        return self.get_factory_status()

    # ==========================================================================
    # 4. 1-Click RTX Training (`run_one_click_rtx_training`)
    # ==========================================================================
    def run_one_click_rtx_training(
        self,
        epochs: int = 50,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        imgsz: int = 640,
        batch: int = 16,
        async_mode: bool = False,
        force_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Launches `RTXTrainerEngine` on `dataset/data.yaml`.

        Args:
            epochs: Number of training epochs (default `50`).
            progress_callback: Optional callback invoked after each epoch with live
                               `epoch`, `mAP50`, `mAP50_95`, `box_loss`, `cls_loss`, `eta_sec`.
            imgsz: Training image resolution (default `640`).
            batch: Batch size tuned for RTX 3070 8GB VRAM (default `16`).
            async_mode: If True, starts training in a background daemon thread via
                        `start_training_async`; if False (default), runs synchronously
                        and returns the completed training result dictionary.
            force_fallback: Force built-in OpenCV/NumPy prototype trainer mode.

        Returns:
            Dictionary containing final training metrics (`mAP50`, `mAP50_95`, `box_loss`,
            `cls_loss`, `precision`, `recall`), exported model paths (`artifacts`), and
            hardware environment details.
        """
        self.curator.ensure_dataset_structure()

        # If dataset is currently empty on disk, automatically bootstrap from reference_screenshots
        manifest = self.curator.get_manifest_stats()
        if int(manifest.get("total_images", 0)) == 0:
            self.bootstrap_from_existing_frames(augment_multiplier=3)

        if async_mode:
            return self.rtx_trainer.start_training_async(
                data_yaml_path=self.data_yaml_path,
                epochs=int(epochs),
                progress_callback=progress_callback,
                imgsz=int(imgsz),
                batch=int(batch),
                force_fallback=bool(force_fallback),
            )

        return self.rtx_trainer.run_training_sync(
            data_yaml_path=self.data_yaml_path,
            epochs=int(epochs),
            progress_callback=progress_callback,
            imgsz=int(imgsz),
            batch=int(batch),
            force_fallback=bool(force_fallback),
        )

    # ==========================================================================
    # 5. Bootstrap & UI Integration Helpers
    # ==========================================================================
    def bootstrap_from_existing_frames(
        self,
        augment_multiplier: Optional[int] = None,
        augment_factor: Optional[int] = None,
        max_source_frames: int = 24,
    ) -> Dict[str, Any]:
        """
        Bootstraps `dataset/` from `reference_screenshots/` and `video_frames/`,
        generating synthetic variants and updating `dataset/manifest_stats.json`.
        """
        if augment_factor is not None:
            factor = max(0, int(augment_factor))
        elif augment_multiplier is not None:
            factor = max(0, int(augment_multiplier) - 1)
        else:
            factor = 3 if self.auto_augment_on_gather else 0

        boot_res = self.curator.bootstrap_from_existing_frames(
            augment_factor=factor,
            save_base_previews=True,
            save_variant_previews=False,
        )
        self._sync_initial_state()
        status = self.get_factory_status()
        status.update(
            {
                "bootstrapped": True,
                "bootstrapped_frames_added": int(boot_res.get("total_samples_saved", 0)),
                "bootstrapped_boxes_added": int(boot_res.get("total_boxes", 0)),
                "bootstrap_summary": boot_res,
            }
        )
        return status

    def render_preview_image(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
        output_name: str = "latest_t_minus_2s_preview.jpg",
    ) -> Path:
        """Render annotated YOLO bounding boxes on `frame` and write to `dataset/previews/`."""
        preview_bgr = self.auto_labeler.draw_annotated_preview(frame, annotations)
        out_path = self.previews_dir / output_name
        imwrite_unicode(out_path, preview_bgr, jpeg_quality=93)
        with self._lock:
            self._last_saved_preview_path = str(out_path)
        return out_path

    def get_factory_gpu_and_onnx_summary(self) -> Dict[str, Any]:
        """Return GPU environment and exported ONNX verification summary."""
        try:
            gpu_info = probe_gpu_environment()
        except Exception as exc:
            gpu_info = {"error": str(exc)}
        try:
            onnx_info = self.trainer.verify_exported_onnx_gpu_ready()
        except Exception as exc:
            onnx_info = {"onnx_ready": False, "error": str(exc)}
        return {
            "gpu_environment": gpu_info,
            "onnx_verification": onnx_info,
        }


__all__ = [
    "FACTORY_12_CLASS_NAMES",
    "FACTORY_12_CLASS_LABELS_TR",
    "AutoLabelingDatasetFactory",
]


if __name__ == "__main__":
    import json

    factory = AutoLabelingDatasetFactory()
    ref_dir = factory.project_root / "reference_screenshots"
    screenshots = sorted(ref_dir.glob("*.jpg")) if ref_dir.exists() else []
    for idx, img_path in enumerate(screenshots[:4]):
        img = imread_unicode(img_path)
        if img is not None:
            factory.ingest_live_frame(
                frame=img,
                player_pos=(100.0 + idx * 5.0, 200.0 + idx * 3.0),
                region_name="Pilgrim's Ridge",
                timestamp=time.time() - (4 - idx) * 0.8,
            )
    gather_out = factory.on_gather_success(
        material_name="Aria",
        harvest_stage=1,
    )
    status_out = factory.get_factory_status()
    print(
        json.dumps(
            {
                "gather_saved": gather_out.get("saved"),
                "box_count": gather_out.get("box_count"),
                "total_images": status_out.get("total_images"),
                "total_boxes": status_out.get("total_boxes"),
                "status_tr": status_out.get("status_tr"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
