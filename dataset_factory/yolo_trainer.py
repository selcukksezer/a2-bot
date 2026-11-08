# -*- coding: utf-8 -*-
"""dataset_factory/yolo_trainer.py
==================================
Aion 2 Autonomous Dataset Factory - YOLO Trainer & Approved Dataset Orchestrator
Division 1 (A08, A09): Selective Training Pipeline ("Sadece Onaylananlarla Eğit").

User Requirement:
"onaylananlar kuyruktan eksilsin birde tekrar tekrar aç kapa yapınca geri gelmesin
onaylananlar ayrı bir yere gitsin eğitim için bekletilsin / onaylananlarla eğit seçeneği olsun hepsini eğitime almasın"

Key Capabilities:
1. Selective Approved Dataset Training:
   - Supports `only_approved: bool = True` in `prepare_and_resolve_training_yaml`,
     `YOLOTrainer`, training worker, and training trigger methods.
   - When `only_approved=True`:
     - Calls `dataset_factory.approved_pool.get_approved_pool(dataset_dir).prepare_approved_training_dataset()`
       and resolves `dataset/approved/data_approved.yaml`.
     - Directs YOLO training strictly to `dataset/approved/data_approved.yaml` instead of
       the entire unvetted dataset (`dataset/data.yaml` / 2400 frames).
     - Logs clear status:
       `[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {count} onaylı kare`
   - When `only_approved=False`:
     - Trains on the full dataset (`dataset/data.yaml`).
     - Logs: `[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {count} kare`.
2. Hardware Acceleration & Engine Integration:
   - Integrates with `RTXTrainerEngine` (RTX 3070 CUDA FP16 mixed precision training + ONNX export).
   - Generates standalone `train_rtx_yolo.bat` targeting the selected dataset YAML.
   - Fallback protection: Gracefully falls back to the built-in OpenCV prototype trainer when
     Ultralytics is not available or requested.
3. Training Worker (`YOLOTrainingWorker` / `RTXTrainingWorker`):
   - Off-GUI background thread with PyQt6 `QThread` and pure Python fallback.
   - Live telemetry emission: `status_text`, `progress_updated`, `training_finished`,
     `training_failed`, `training_deferred`.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import dataset_factory.approved_pool

logger = logging.getLogger("DatasetFactory.YOLOTrainer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Optional PyQt6 Imports with Thread Fallback
# -----------------------------------------------------------------------------
try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal
    _HAS_PYQT = True
except ImportError:
    _HAS_PYQT = False

    class _DummySignal:
        def __init__(self) -> None:
            self._callbacks: List[Callable[..., Any]] = []

        def connect(self, slot: Callable[..., Any]) -> None:
            if callable(slot) and slot not in self._callbacks:
                self._callbacks.append(slot)

        def disconnect(self, slot: Optional[Callable[..., Any]] = None) -> None:
            if slot is None:
                self._callbacks.clear()
            elif slot in self._callbacks:
                self._callbacks.remove(slot)

        def emit(self, *args: Any, **kwargs: Any) -> None:
            for cb in list(self._callbacks):
                try:
                    cb(*args, **kwargs)
                except Exception as ex:
                    logger.warning("Signal callback error: %s", ex)

    class QThread(threading.Thread):  # type: ignore[no-redef]
        def __init__(self, parent: Optional[Any] = None) -> None:
            super().__init__(daemon=True)
            self._parent = parent

        def isRunning(self) -> bool:
            return self.is_alive()

        def wait(self, timeout_ms: int = 15000) -> bool:
            self.join(timeout=max(0.001, timeout_ms / 1000.0))
            return not self.is_alive()

    def pyqtSignal(*_types: Any) -> _DummySignal:  # type: ignore[no-redef]
        return _DummySignal()

# -----------------------------------------------------------------------------
# Approved Pool & RTX Engine Imports
# -----------------------------------------------------------------------------
try:
    from dataset_factory.approved_pool import ApprovedDatasetPool
except Exception as _pool_import_exc:
    ApprovedDatasetPool = None  # type: ignore[assignment,misc]
    logger.debug("ApprovedDatasetPool not directly imported: %s", _pool_import_exc)

try:
    from dataset_factory.rtx_trainer_engine import RTXTrainerEngine, offload_root, rtx_runs_dir_for
except Exception as _rtx_import_exc:
    RTXTrainerEngine = None  # type: ignore[assignment,misc]
    offload_root = None  # type: ignore[assignment]
    rtx_runs_dir_for = None  # type: ignore[assignment]
    logger.debug("RTXTrainerEngine not directly imported: %s", _rtx_import_exc)


# =============================================================================
# Helper Utilities
# =============================================================================

def _resolve_project_root(hint: Optional[Union[str, Path]] = None) -> Path:
    """Resolves project root path, favoring F:/a2 if exists."""
    if hint is not None:
        p = Path(hint).resolve()
        if p.exists():
            return p
    if offload_root is not None:
        preferred = offload_root()
        if preferred is not None and preferred.exists():
            return preferred
    p_a2 = Path("F:/a2")
    if p_a2.is_dir():
        return p_a2
    return Path(__file__).resolve().parent.parent


def _normalize_dataset_and_approved_dir(
    dataset_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path, Path]:
    """
    Normalizes `(project_root, dataset_dir, approved_dir)` from a `dataset_dir`
    or `project_root` argument.
    """
    if dataset_dir is None:
        proj = _resolve_project_root(project_root)
        ds = proj / "dataset"
        return proj, ds, ds / "approved"

    raw = Path(dataset_dir).resolve()
    if raw.name.lower() == "approved":
        ds = raw.parent
        proj = ds.parent if ds.name.lower() == "dataset" else ds
        return proj, ds, raw

    if raw.name.lower() == "dataset":
        return raw.parent, raw, raw / "approved"

    if (raw / "dataset").is_dir() and not ((raw / "images").is_dir() or (raw / "review").is_dir()):
        ds = raw / "dataset"
        return raw, ds, ds / "approved"

    return raw.parent, raw, raw / "approved"


def count_images_in_dir(directory: Path) -> int:
    """Counts valid image files (.jpg, .png, .jpeg) in a directory."""
    if not directory.is_dir():
        return 0
    count = 0
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".png", ".jpeg"):
            count += 1
    return count


def _write_fallback_approved_yaml(approved_dir: Path, approved_yaml_path: Path) -> Path:
    """Writes canonical 22-class `data_approved.yaml` into `approved_dir`."""
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (approved_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (approved_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (approved_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    pool_posix = approved_dir.as_posix()
    lines = [
        "# Ultralytics YOLO Configuration - Approved Dataset Pool (A09 Standalone Fallback)",
        f"path: {pool_posix}",
        "train: images/train",
        "val: images/val",
        "nc: 22",
        "names:",
        "  0: herb_aria",
        "  1: herb_mela",
        "  2: herb_targena",
        "  3: ore_orichalcum",
        "  4: gem_crystal",
        "  5: tree_log",
        "  6: herb_edana",
        "  7: herb_lumina",
        "  8: ore_obsidian",
        "  9: minimap_sprout",
        "  10: monster_3d",
        "  11: minimap_monster_dot",
        "  12: crystal_blood",
        "  13: skill_ui_prompt",
        "  14: obstacle",
        "  15: path_ground",
        "  16: player_hp_bar",
        "  17: player_mp_bar",
        "  18: target_hp_bar",
        "  19: skill_bar",
        "  20: skill_slot_ready",
        "  21: skill_slot_cooldown",
    ]
    content = "\n".join(lines) + "\n"
    approved_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    approved_yaml_path.write_text(content, encoding="utf-8")
    return approved_yaml_path


def prepare_and_resolve_training_yaml(
    dataset_dir: Optional[Union[str, Path]] = None,
    only_approved: bool = True,
) -> Path:
    """
    Prepares and resolves the training dataset YAML file path.

    When `only_approved=True`:
      Calls `dataset_factory.approved_pool.get_approved_pool(dataset_dir).prepare_approved_training_dataset()`
      and returns the path to `dataset/approved/data_approved.yaml`.
    When `only_approved=False`:
      Returns the path to `dataset/data.yaml`.
    """
    proj_root, ds_dir, approved_dir = _normalize_dataset_and_approved_dir(dataset_dir)
    if not only_approved:
        return ds_dir / "data.yaml"

    approved_yaml = approved_dir / "data_approved.yaml"
    nested_ds = ds_dir / "dataset"
    nested_existed = nested_ds.exists()

    pool: Any = None
    prep_result: Any = None
    try:
        pool = dataset_factory.approved_pool.get_approved_pool(dataset_dir)
        prep_result = pool.prepare_approved_training_dataset()
    except Exception as exc:
        logger.warning("prepare_approved_training_dataset via get_approved_pool failed: %s", exc)

    # Reconcile if ApprovedDatasetPool singleton was bound to another path or nested `dataset/dataset`
    try:
        if ApprovedDatasetPool is not None and isinstance(pool, ApprovedDatasetPool):
            pool_root_resolved = Path(getattr(pool, "pool_root", approved_dir)).resolve()
            if pool_root_resolved != approved_dir.resolve():
                if not nested_existed and nested_ds.is_dir() and ds_dir.name.lower() == "dataset":
                    shutil.rmtree(nested_ds, ignore_errors=True)
                scoped_pool = ApprovedDatasetPool(pool_root=approved_dir, project_root=proj_root)
                scoped_pool.sync_from_disk()
                prep_result = scoped_pool.prepare_approved_training_dataset(source_dataset_dir=ds_dir)
                try:
                    dataset_factory.approved_pool._GLOBAL_POOL = scoped_pool
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("Scoped ApprovedDatasetPool reconciliation skipped: %s", exc)

    if isinstance(prep_result, (str, Path)):
        cand = Path(prep_result)
        if cand.suffix.lower() in (".yaml", ".yml") and cand.is_file() and not approved_yaml.is_file():
            approved_yaml = cand
    elif isinstance(prep_result, dict):
        yaml_from_dict = prep_result.get("data_yaml_path") or prep_result.get("yaml_path")
        if yaml_from_dict and not approved_yaml.is_file():
            cand = Path(str(yaml_from_dict))
            if cand.is_file() and cand.name == "data_approved.yaml":
                approved_yaml = cand

    if not approved_yaml.is_file():
        _write_fallback_approved_yaml(approved_dir, approved_yaml)

    return approved_yaml


# =============================================================================
# YOLOTrainer Class
# =============================================================================

class YOLOTrainer:
    """
    YOLO Model Trainer & Approved Dataset Pipeline Orchestrator.
    
    Supports training strictly with human-approved samples (`only_approved=True`)
    via `dataset/approved/data_approved.yaml` or with the full raw dataset
    (`only_approved=False`) via `dataset/data.yaml`.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        engine: Optional[Any] = None,
        approved_pool: Optional[Any] = None,
    ) -> None:
        if dataset_dir is not None and project_root is None:
            proj, ds, appr = _normalize_dataset_and_approved_dir(dataset_dir)
            self.project_root: Path = proj
            self.dataset_dir: Path = ds
            self.approved_dir: Path = appr
        else:
            self.project_root = _resolve_project_root(project_root)
            self.dataset_dir = (
                Path(dataset_dir).resolve() if dataset_dir is not None else self.project_root / "dataset"
            )
            self.approved_dir = self.dataset_dir / "approved"

        self.models_dir: Path = (
            Path(models_dir).resolve() if models_dir is not None else self.project_root / "models"
        )
        self.approved_yaml_path: Path = self.approved_dir / "data_approved.yaml"
        self.default_yaml_path: Path = self.dataset_dir / "data.yaml"

        self._approved_pool: Optional[Any] = approved_pool
        self._engine: Optional[Any] = engine
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Approved Pool & Engine Lazy Getters
    # -------------------------------------------------------------------------

    def get_approved_pool(self) -> Any:
        """Returns or lazily creates an ApprovedDatasetPool instance."""
        with self._lock:
            if self._approved_pool is not None:
                return self._approved_pool
            if ApprovedDatasetPool is not None:
                self._approved_pool = ApprovedDatasetPool(
                    pool_root=self.approved_dir,
                    project_root=self.project_root,
                )
                return self._approved_pool
            return None

    def get_engine(self) -> Any:
        """Returns or lazily creates an RTXTrainerEngine instance."""
        with self._lock:
            if self._engine is not None:
                return self._engine
            if RTXTrainerEngine is not None:
                self._engine = RTXTrainerEngine(
                    project_root=self.project_root,
                    dataset_dir=self.dataset_dir,
                    models_dir=self.models_dir,
                )
                return self._engine
            raise RuntimeError("RTXTrainerEngine is not available in current environment.")

    # -------------------------------------------------------------------------
    # Dataset Sample Counting Helpers
    # -------------------------------------------------------------------------

    def count_approved_samples(self) -> int:
        """
        Calculates the number of approved images in `dataset/approved/`.
        Uses `ApprovedDatasetPool.get_approved_stats()` or inspects the disk directly.
        """
        train_img_dir = self.approved_dir / "images" / "train"
        val_img_dir = self.approved_dir / "images" / "val"
        disk_count = count_images_in_dir(train_img_dir) + count_images_in_dir(val_img_dir)
        if disk_count > 0:
            return disk_count

        pool = self.get_approved_pool()
        if pool is not None:
            try:
                stats = pool.get_approved_stats()
                count = int(stats.get("total_approved_images", 0))
                if count > 0:
                    return count
            except Exception as ex:
                logger.debug("Failed to get approved stats from pool: %s", ex)

        manifest_p = self.approved_dir / "approved_manifest.json"
        if manifest_p.is_file():
            try:
                data = json.loads(manifest_p.read_text(encoding="utf-8"))
                stems = data.get("stems", {})
                if isinstance(stems, dict):
                    return len(stems)
                return int(data.get("total_approved_images", 0))
            except Exception:
                pass

        return 0

    def count_total_dataset_samples(self) -> int:
        """
        Calculates the number of total images in `dataset/images/{train,val}`.
        """
        train_img_dir = self.dataset_dir / "images" / "train"
        val_img_dir = self.dataset_dir / "images" / "val"
        count = count_images_in_dir(train_img_dir) + count_images_in_dir(val_img_dir)
        if count > 0:
            return count

        manifest_stats = self.dataset_dir / "manifest_stats.json"
        if manifest_stats.is_file():
            try:
                data = json.loads(manifest_stats.read_text(encoding="utf-8"))
                return int(data.get("total_frames", data.get("total_images", 0)))
            except Exception:
                pass

        return 0

    # -------------------------------------------------------------------------
    # Core YAML & Dataset Resolution
    # -------------------------------------------------------------------------

    def prepare_and_resolve_training_yaml(
        self,
        dataset_dir: Optional[Union[str, Path, bool]] = None,
        only_approved: bool = True,
    ) -> Path:
        """
        Instance helper around `prepare_and_resolve_training_yaml(dataset_dir, only_approved)`.
        """
        if isinstance(dataset_dir, bool):
            effective_only_approved = dataset_dir
            target_ds_dir = self.dataset_dir
        elif dataset_dir is not None:
            effective_only_approved = bool(only_approved)
            target_ds_dir = Path(dataset_dir).resolve()
        else:
            effective_only_approved = bool(only_approved)
            target_ds_dir = self.dataset_dir

        return prepare_and_resolve_training_yaml(
            dataset_dir=target_ds_dir,
            only_approved=effective_only_approved,
        )

    def resolve_training_data_yaml(
        self,
        only_approved: bool = True,
        custom_yaml: Optional[Union[str, Path]] = None,
        auto_generate: bool = True,
    ) -> Tuple[Path, int]:
        """
        Resolves which dataset YAML file to use for training.

        When `only_approved=True`:
          - Prepares and verifies `dataset/approved/data_approved.yaml` via
            `prepare_and_resolve_training_yaml(self.dataset_dir, only_approved=True)`.
          - Points YOLO training strictly to `dataset/approved/data_approved.yaml`.
          - Logs: `[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {count} onaylı kare`.
          - Returns: `(approved_yaml_path, count)`.

        When `only_approved=False`:
          - Points YOLO training to `custom_yaml` or `dataset/data.yaml`.
          - Logs: `[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {count} kare`.
          - Returns: `(yaml_path, count)`.
        """
        if only_approved:
            if auto_generate or not self.approved_yaml_path.is_file():
                yaml_p = prepare_and_resolve_training_yaml(
                    dataset_dir=self.dataset_dir,
                    only_approved=True,
                )
            else:
                yaml_p = self.approved_yaml_path

            count = self.count_approved_samples()
            if not yaml_p.is_file():
                self._generate_fallback_approved_yaml()
                yaml_p = self.approved_yaml_path

            log_line = f"[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {count} onaylı kare"
            logger.info(log_line)
            print(log_line, flush=True)

            return yaml_p, count
        else:
            if custom_yaml is not None:
                yaml_p = Path(custom_yaml).resolve()
            else:
                yaml_p = prepare_and_resolve_training_yaml(
                    dataset_dir=self.dataset_dir,
                    only_approved=False,
                ).resolve()

            count = self.count_total_dataset_samples()

            log_line = f"[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {count} kare"
            logger.info(log_line)
            print(log_line, flush=True)

            return yaml_p, count

    def _generate_fallback_approved_yaml(self) -> Path:
        """Generates fallback `data_approved.yaml` if ApprovedDatasetPool is unavailable."""
        return _write_fallback_approved_yaml(self.approved_dir, self.approved_yaml_path)

    # -------------------------------------------------------------------------
    # Synchronous Training Execution
    # -------------------------------------------------------------------------

    def run_training_sync(
        self,
        epochs: int = 25,
        only_approved: bool = True,
        data_yaml_path: Optional[Union[str, Path]] = None,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        force_fallback: bool = False,
        allow_opencv_fallback: bool = True,
        use_approved_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes YOLO training synchronously.

        Points strictly to `dataset/approved/data_approved.yaml` when `only_approved=True`
        (or `use_approved_only=True`).
        """
        if use_approved_only is not None:
            only_approved = bool(use_approved_only)
        elif data_yaml_path is not None and Path(data_yaml_path).name.lower() == "data_approved.yaml":
            only_approved = True

        yaml_path, sample_count = self.resolve_training_data_yaml(
            only_approved=only_approved,
            custom_yaml=data_yaml_path,
        )

        status_msg = (
            f"[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {sample_count} onaylı kare"
            if only_approved
            else f"[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {sample_count} kare"
        )
        if callable(status_callback):
            try:
                status_callback(status_msg)
            except Exception:
                pass

        if only_approved and sample_count == 0:
            err_msg = "Onaylanmış eğitim karesi bulunamadı. Lütfen önce Step 2'de kareleri onaylayın."
            logger.warning("[EĞİTİM] %s", err_msg)
            if callable(status_callback):
                try:
                    status_callback(f"[HATA] {err_msg}")
                except Exception:
                    pass
            return {
                "started": False,
                "status": "failed",
                "state": "failed",
                "error": err_msg,
                "reason": err_msg,
                "only_approved": True,
                "use_approved_only": True,
                "approved_count": 0,
                "data_yaml_path": str(yaml_path),
            }

        engine = self.get_engine()
        call_kwargs: Dict[str, Any] = dict(kwargs)
        try:
            sig = inspect.signature(engine.run_training_sync)
            if "use_approved_only" in sig.parameters:
                call_kwargs["use_approved_only"] = only_approved
        except Exception:
            pass

        result = engine.run_training_sync(
            data_yaml_path=yaml_path,
            epochs=epochs,
            progress_callback=progress_callback,
            imgsz=imgsz,
            batch=batch,
            force_fallback=force_fallback,
            model=model,
            allow_opencv_fallback=allow_opencv_fallback,
            **call_kwargs,
        )

        if isinstance(result, dict):
            result["only_approved"] = only_approved
            result["use_approved_only"] = only_approved
            result["sample_count"] = sample_count
            result["selected_yaml"] = str(yaml_path)

        return result

    # -------------------------------------------------------------------------
    # Asynchronous Training Execution
    # -------------------------------------------------------------------------

    def start_training_async(
        self,
        epochs: int = 25,
        only_approved: bool = True,
        data_yaml_path: Optional[Union[str, Path]] = None,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        force_fallback: bool = False,
        allow_opencv_fallback: bool = True,
        use_approved_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Starts training asynchronously on a background thread.
        """
        if use_approved_only is not None:
            only_approved = bool(use_approved_only)
        elif data_yaml_path is not None and Path(data_yaml_path).name.lower() == "data_approved.yaml":
            only_approved = True

        yaml_path, sample_count = self.resolve_training_data_yaml(
            only_approved=only_approved,
            custom_yaml=data_yaml_path,
        )

        status_msg = (
            f"[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {sample_count} onaylı kare"
            if only_approved
            else f"[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {sample_count} kare"
        )
        if callable(status_callback):
            try:
                status_callback(status_msg)
            except Exception:
                pass

        if only_approved and sample_count == 0:
            err_msg = "Onaylanmış eğitim karesi bulunamadı. Lütfen önce Step 2'de kareleri onaylayın."
            logger.warning("[EĞİTİM] %s", err_msg)
            return {
                "started": False,
                "status": "failed",
                "state": "failed",
                "error": err_msg,
                "reason": err_msg,
                "only_approved": True,
                "use_approved_only": True,
                "approved_count": 0,
                "data_yaml_path": str(yaml_path),
            }

        engine = self.get_engine()
        call_kwargs: Dict[str, Any] = dict(kwargs)
        try:
            sig = inspect.signature(engine.start_training_async)
            if "use_approved_only" in sig.parameters:
                call_kwargs["use_approved_only"] = only_approved
        except Exception:
            pass

        result = engine.start_training_async(
            data_yaml_path=yaml_path,
            epochs=epochs,
            progress_callback=progress_callback,
            imgsz=imgsz,
            batch=batch,
            force_fallback=force_fallback,
            model=model,
            allow_opencv_fallback=allow_opencv_fallback,
            **call_kwargs,
        )

        if isinstance(result, dict):
            result["only_approved"] = only_approved
            result["use_approved_only"] = only_approved
            result["sample_count"] = sample_count
            result["selected_yaml"] = str(yaml_path)

        return result

    # -------------------------------------------------------------------------
    # Convenience Wrapper
    # -------------------------------------------------------------------------

    def train(
        self,
        epochs: int = 25,
        only_approved: bool = True,
        async_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unified training entry point (sync or async)."""
        if async_run:
            return self.start_training_async(epochs=epochs, only_approved=only_approved, **kwargs)
        return self.run_training_sync(epochs=epochs, only_approved=only_approved, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """Queries current status from the underlying training engine."""
        try:
            engine = self.get_engine()
            return engine.get_status()
        except Exception:
            return {"state": "idle", "started": False}

    def stop_training(self) -> None:
        """Requests training stop."""
        try:
            engine = self.get_engine()
            engine.stop_training()
        except Exception:
            pass


# =============================================================================
# YOLOTrainingWorker (QThread Worker Supporting only_approved)
# =============================================================================

class YOLOTrainingWorker(QThread):
    """
    QThread worker for YOLO training.
    
    Fully supports `only_approved: bool = True` to train strictly with
    approved samples, updating GUI status with:
      `[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {count} onaylı kare`
    """

    progress_updated = pyqtSignal(dict)
    training_finished = pyqtSignal(dict)
    training_failed = pyqtSignal(str)
    status_text = pyqtSignal(str)
    training_deferred = pyqtSignal(str)

    def __init__(
        self,
        controller: Any = None,
        epochs: int = 25,
        parent: Optional[Any] = None,
        *,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        only_approved: bool = True,
        trainer: Optional[YOLOTrainer] = None,
        project_root: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        allow_opencv_fallback: bool = True,
        force_fallback: bool = False,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.epochs = max(1, int(epochs))
        self.imgsz = max(320, int(imgsz))
        self.batch = max(1, int(batch))
        self.model = str(model or "yolo11n.pt").strip() or "yolo11n.pt"
        self.only_approved = bool(only_approved)
        self.allow_opencv_fallback = bool(allow_opencv_fallback)
        self.force_fallback = bool(force_fallback)
        self._stop_requested: bool = False

        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = YOLOTrainer(
                project_root=project_root,
                dataset_dir=dataset_dir,
            )

    def is_stop_requested(self) -> bool:
        return bool(getattr(self, "_stop_requested", False) or self.isInterruptionRequested())

    def stop(self) -> None:
        """Requests training stop across worker and underlying trainer/controller."""
        self._stop_requested = True
        self.requestInterruption()
        if hasattr(self.trainer, "stop_training"):
            try:
                self.trainer.stop_training()
            except Exception:
                pass
        if hasattr(self.controller, "stop_training"):
            try:
                self.controller.stop_training()
            except Exception:
                pass

    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if self.is_stop_requested():
            self.stop()
        self.progress_updated.emit(dict(payload))

    def run(self) -> None:
        threading.current_thread().name = "GatherionYOLOTrainingWorker"
        try:
            if self.is_stop_requested():
                self.training_failed.emit("Eğitim kullanıcı tarafından durduruldu.")
                return
            # 1. Resolve data yaml and calculate sample count
            yaml_path, sample_count = self.trainer.resolve_training_data_yaml(
                only_approved=self.only_approved
            )

            status_line = (
                f"[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {sample_count} onaylı kare"
                if self.only_approved
                else f"[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {sample_count} kare"
            )
            self.status_text.emit(status_line)

            # 2. Check if approved dataset is empty
            if self.only_approved and sample_count == 0:
                err_msg = "Onaylanmış eğitim karesi bulunamadı. Lütfen önce Step 2'de kareleri onaylayın."
                logger.warning("[EĞİTİM] %s", err_msg)
                self.training_failed.emit(err_msg)
                return

            # 3. Check for game running blocking GPU
            if hasattr(self.controller, "_game_blocks_cuda_training"):
                try:
                    blocked = self.controller._game_blocks_cuda_training()
                    if blocked:
                        self.status_text.emit(f"Eğitim Durumu: {blocked}")
                        self.training_deferred.emit(blocked)
                        return
                except Exception:
                    pass

            # 4. Invoke training
            ctrl_fn = getattr(self.controller, "run_one_click_rtx_training", None)
            if callable(ctrl_fn):
                sig = inspect.signature(ctrl_fn)
                kwargs: Dict[str, Any] = {
                    "epochs": self.epochs,
                    "imgsz": self.imgsz,
                    "batch": self.batch,
                    "model": self.model,
                    "progress_callback": self._emit_progress,
                }
                if "data_yaml_path" in sig.parameters:
                    kwargs["data_yaml_path"] = yaml_path
                if "only_approved" in sig.parameters:
                    kwargs["only_approved"] = self.only_approved
                if "use_approved_only" in sig.parameters:
                    kwargs["use_approved_only"] = self.only_approved
                if "allow_opencv_fallback" in sig.parameters:
                    kwargs["allow_opencv_fallback"] = self.allow_opencv_fallback
                if "force_fallback" in sig.parameters:
                    kwargs["force_fallback"] = self.force_fallback

                result = ctrl_fn(**kwargs)
            else:
                result = self.trainer.run_training_sync(
                    epochs=self.epochs,
                    only_approved=self.only_approved,
                    data_yaml_path=yaml_path,
                    imgsz=self.imgsz,
                    batch=self.batch,
                    model=self.model,
                    progress_callback=self._emit_progress,
                    status_callback=self.status_text.emit,
                    allow_opencv_fallback=self.allow_opencv_fallback,
                    force_fallback=self.force_fallback,
                )

            # 5. Handle training result
            if isinstance(result, dict) and (
                result.get("waiting_for_game_exit") or result.get("reason") == "aion2_running"
            ):
                reason = "Eğitim, oyun kapanana kadar bekliyor."
                self.status_text.emit(f"Eğitim Durumu: {reason}")
                self.training_deferred.emit(reason)
                return

            if isinstance(result, dict) and (
                result.get("stopped")
                or result.get("interrupted")
                or result.get("status") == "stopped"
                or result.get("state") == "stopped"
            ):
                err = str(result.get("reason") or "Eğitim kullanıcı tarafından durduruldu.")
                self.status_text.emit(f"Eğitim Durumu: Durduruldu ({err})")
                self.training_failed.emit(err)
                return

            if isinstance(result, dict) and (
                result.get("started") is False
                or result.get("status") == "failed"
                or result.get("state") in ("failed", "refused", "busy")
            ):
                err = str(result.get("reason") or result.get("error") or "Eğitim başarısız.")
                self.training_failed.emit(err)
                return

            self.training_finished.emit(result if isinstance(result, dict) else {"status": "success"})

        except Exception as exc:
            logger.exception("YOLO training worker error: %s", exc)
            self.training_failed.emit(str(exc))


# Alias for drop-in compatibility with existing codebase
RTXTrainingWorker = YOLOTrainingWorker


# =============================================================================
# Standalone Training Trigger Functions
# =============================================================================

def start_yolo_training(
    only_approved: bool = True,
    epochs: int = 25,
    model: str = "yolo11n.pt",
    imgsz: int = 640,
    batch: int = 16,
    controller: Any = None,
    trainer: Optional[YOLOTrainer] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    async_mode: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    High-level trigger method to launch YOLO training.

    Supports `only_approved: bool = True` to train strictly with approved samples.
    """
    active_trainer = trainer or YOLOTrainer()
    if async_mode:
        return active_trainer.start_training_async(
            epochs=epochs,
            only_approved=only_approved,
            model=model,
            imgsz=imgsz,
            batch=batch,
            progress_callback=progress_callback,
            status_callback=status_callback,
            **kwargs,
        )
    return active_trainer.run_training_sync(
        epochs=epochs,
        only_approved=only_approved,
        model=model,
        imgsz=imgsz,
        batch=batch,
        progress_callback=progress_callback,
        status_callback=status_callback,
        **kwargs,
    )


def start_training_worker(
    controller: Any = None,
    epochs: int = 25,
    model: str = "yolo11n.pt",
    imgsz: int = 640,
    batch: int = 16,
    only_approved: bool = True,
    parent: Optional[Any] = None,
    trainer: Optional[YOLOTrainer] = None,
    **kwargs: Any,
) -> YOLOTrainingWorker:
    """
    Creates and starts a YOLOTrainingWorker thread with `only_approved` support.
    """
    worker = YOLOTrainingWorker(
        controller=controller,
        epochs=epochs,
        parent=parent,
        imgsz=imgsz,
        batch=batch,
        model=model,
        only_approved=only_approved,
        trainer=trainer,
        **kwargs,
    )
    worker.start()
    return worker


def resolve_yolo_training_yaml(
    only_approved: bool = True,
    project_root: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, int]:
    """
    Stand-alone helper to resolve the active training YAML and sample count.
    """
    trainer = YOLOTrainer(project_root=project_root, dataset_dir=dataset_dir)
    return trainer.resolve_training_data_yaml(only_approved=only_approved)
