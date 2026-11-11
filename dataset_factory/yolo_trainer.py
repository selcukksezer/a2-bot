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
            self._interruption_requested = False

        def isRunning(self) -> bool:
            return self.is_alive()

        # F-1971: the PyQt-less fallback must offer the same interruption
        # API the worker relies on (isInterruptionRequested/requestInterruption).
        def isInterruptionRequested(self) -> bool:
            return bool(self._interruption_requested)

        def requestInterruption(self) -> None:
            self._interruption_requested = True

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

def _canonical_class_names() -> Dict[int, str]:
    """Canonical 27-class taxonomy (0..26) for fallback approved yamls."""
    try:
        from dataset_factory.class_taxonomy import CANONICAL_NAMES

        return dict(CANONICAL_NAMES)
    except Exception:
        return {
            0: "herb_aria", 1: "herb_mela", 2: "herb_targena",
            3: "ore_orichalcum", 4: "gem_crystal", 5: "tree_log",
            6: "herb_edana", 7: "herb_lumina", 8: "ore_obsidian",
            9: "minimap_sprout", 10: "monster_3d", 11: "minimap_monster_dot",
            12: "crystal_blood", 13: "skill_ui_prompt", 14: "obstacle",
            15: "path_ground", 16: "player_hp_bar", 17: "player_mp_bar",
            18: "target_hp_bar", 19: "skill_bar", 20: "skill_slot_ready",
            21: "skill_slot_cooldown", 22: "hp_potion", 23: "mp_potion",
            24: "buff_scroll", 25: "kinah", 26: "player_character",
        }


def resolve_yaml_split_dir(yaml_path: Path, parsed: Dict[str, Any], split_key: str) -> Path:
    """Resolve one `train`/`val` entry of a dataset yaml the way Ultralytics does.

    P0-5 (K-28/K-30, F058 — Ultralytics 8.4.209 `check_det_dataset`,
    `data/utils.py:796-811`): a relative `path:` is resolved against the
    TRAINING PROCESS's working directory (CWD), NOT against the yaml's own
    directory. The yaml-dir fallback applies only when the `path:` key is
    ABSENT. The old `_count_images_for_yaml` resolved a relative `path:`
    against the yaml dir, so the counter and the engine looked at different
    directories — the counter said "N kare" while training found 0 images.
    Both now share this one function, so the counted directory IS the
    directory the engine will read.

    All three yaml writers in this repo write an ABSOLUTE `path:` (faz-1
    0d727f6), so in practice the relative branch never fires for our own
    yamls; it exists so a hand-written / foreign yaml is counted exactly
    where Ultralytics would look, and `validate_training_yaml` rejects it
    before training starts.
    """
    base_raw = parsed.get("path")
    if base_raw is None or str(base_raw).strip() == "":
        # No `path:` key: Ultralytics falls back to the yaml's directory.
        base = Path(yaml_path).parent
    else:
        base = Path(str(base_raw))
        if not base.is_absolute():
            # Ultralytics rule: relative `path:` is CWD-relative.
            base = Path.cwd() / base
    rel = parsed.get(split_key)
    if rel is None or str(rel).strip() == "":
        return base
    d = Path(str(rel))
    if d.is_absolute():
        return d
    return base / d


def validate_training_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate a dataset yaml BEFORE training starts (P0-5, K-30).

    Checks, in order: file exists, parses, `path:` present AND absolute,
    `train`/`val` entries present, both split dirs exist (resolved with the
    Ultralytics rule in `resolve_yaml_split_dir` — the same rule the engine
    uses), and at least one LABELED image exists across both splits (images
    without a non-empty .txt are silently dropped by YOLO, so a raw image
    count would lie — see `count_labeled_images`).

    Returns a dict: `ok` (bool), `errors` (list of concrete, user-facing
    Turkish reasons — each names the file/dir that is wrong), `train_count`
    / `val_count` (labeled images per split), `total_labeled`, `base_dir`
    (the resolved `path:` value). Callers must NOT start training when
    `ok` is False; they surface `errors` to the user instead of letting
    the worker thread / Ultralytics fail with a cryptic message.
    """
    yaml_p = Path(yaml_path)
    errors: List[str] = []
    info: Dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "yaml_path": str(yaml_p),
        "base_dir": "",
        "train_count": 0,
        "val_count": 0,
        "total_labeled": 0,
    }
    if not yaml_p.is_file():
        errors.append(f"eğitim yaml'ı yok: {yaml_p}")
        return info
    try:
        from dataset_factory.rtx_trainer_engine import _parse_simple_yaml

        parsed = _parse_simple_yaml(yaml_p)
    except Exception as exc:
        errors.append(f"eğitim yaml'ı okunamadı ({yaml_p}): {exc}")
        return info
    if not isinstance(parsed, dict):
        errors.append(f"eğitim yaml'ı geçersiz (sözlük değil): {yaml_p}")
        return info

    base_raw = parsed.get("path")
    if base_raw is None or str(base_raw).strip() == "":
        errors.append(
            f"yaml'da `path:` yok ({yaml_p}) — Ultralytics train/val'i "
            "çalışma klasörüne (CWD) göre çözer, yaml'ın dizinine göre değil; "
            "`path:` mutlak yol olmalı."
        )
        return info
    base = Path(str(base_raw))
    if not base.is_absolute():
        errors.append(
            f"yaml'daki `path:` göreli ('{base_raw}') — mutlak yol olmalı; "
            "Ultralytics göreli yolu eğitim sürecinin çalışma klasörüne "
            f"(CWD) göre çözer ve yanlış dizini arar. yaml: {yaml_p}"
        )
        return info
    info["base_dir"] = str(base)

    counts: Dict[str, int] = {}
    for key in ("train", "val"):
        rel = parsed.get(key)
        if rel is None or str(rel).strip() == "":
            errors.append(f"yaml'da `{key}:` yok: {yaml_p}")
            counts[key] = 0
            continue
        split_dir = resolve_yaml_split_dir(yaml_p, parsed, key)
        if not split_dir.is_dir():
            errors.append(
                f"`{key}` dizini yok: {split_dir} "
                f"(yaml: {yaml_p} — `path:` {base_raw}, `{key}:` {rel})"
            )
            counts[key] = 0
            continue
        labels_dir = base / "labels" / key
        counts[key] = count_labeled_images(split_dir, labels_dir)
    info["train_count"] = counts.get("train", 0)
    info["val_count"] = counts.get("val", 0)
    info["total_labeled"] = info["train_count"] + info["val_count"]
    if info["total_labeled"] == 0:
        # P0-5 tur 2: 0-etiketli-kare hatası eski B-06 kapısının
        # ("Eğitim verisi boş") sözleşme cümlesiyle başlar — iki kapı
        # aynı koşulu yakalar; dışarıya dönük eski mesaj korunur.
        # Diğer geçersizlik nedenlerinin mesajları aynen kalır.
        errors.append(
            "Eğitim verisi boş: etiketli kare yok (0 örnek) — train/val "
            f"dizinlerinde .txt etiketli görsel bulunamadı (yaml: {yaml_p}). "
            "Önce kare onaylayın/dondurun."
        )
    info["ok"] = not errors
    return info


def _count_images_for_yaml(yaml_path: Path) -> int:
    """Count images in the train/val dirs a dataset yaml points at.

    F-1990: a custom yaml's own data must be counted, not the whole dataset.
    Falls back to 0 when the yaml cannot be parsed (caller falls back).

    P0-5 (K-30): the split dirs are resolved with `resolve_yaml_split_dir`
    — the Ultralytics rule (relative `path:` is CWD-relative) — so this
    counter and the training engine look at the SAME directory. It used to
    resolve a relative `path:` against the yaml's own directory, which is
    NOT what Ultralytics does.
    """
    try:
        from dataset_factory.rtx_trainer_engine import _parse_simple_yaml

        parsed = _parse_simple_yaml(Path(yaml_path))
    except Exception:
        return 0
    total = 0
    for key in ("train", "val"):
        if not parsed.get(key):
            continue
        total += count_images_in_dir(resolve_yaml_split_dir(Path(yaml_path), parsed, key))
    return total


def _resolve_project_root(hint: Optional[Union[str, Path]] = None) -> Path:
    """Resolves project root path.

    F-1994: no hardcoded F:/a2 preference — offload_root() already returns
    F:/a2 when that drive/folder exists; otherwise the repo root is used.
    """
    if hint is not None:
        p = Path(hint).resolve()
        if p.exists():
            return p
    if offload_root is not None:
        preferred = offload_root()
        if preferred is not None and preferred.exists():
            return preferred
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


_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)

_DEFAULT_MODEL = "yolo11n.pt"


def count_images_in_dir(directory: Path) -> int:
    """Counts valid image files (.jpg, .png, .jpeg, .webp, .bmp, .tif/.tiff).

    Uses `os.scandir` + `DirEntry.is_file()` (d_type, no per-file stat) instead
    of `Path.iterdir()` + `Path.is_file()` (one `stat` per entry): ~7x faster
    on a 3500-entry directory (median ~300 ms -> ~42 ms/call, timeit, 5x10
    reps). Suffix matching is identical (`os.path.splitext`, lower-cased).
    """
    if not directory.is_dir():
        return 0
    count = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in _IMAGE_EXTENSIONS:
                    count += 1
    return count


def normalize_training_params(
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    model: str = "yolo11n.pt",
) -> Tuple[int, int, int, str]:
    """Clamps training hyperparameters to sane ranges (single validation point).

    The worker used to clamp epochs/imgsz/batch/model while `run_training_sync`
    / `start_training_async` passed raw values to the engine, so `epochs=0`
    behaved differently per entry point. All three now share this helper.
    """
    return (
        max(1, int(epochs)),
        max(320, int(imgsz)),
        max(1, int(batch)),
        str(model or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL,
    )


def _training_status_message(only_approved: bool, sample_count: int) -> str:
    """Canonical `[EĞİTİM] ...` start line, byte-identical in every caller."""
    if only_approved:
        return (
            "[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: "
            f"{sample_count} onaylı kare"
        )
    return f"[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: {sample_count} kare"


def _empty_approved_result(yaml_path: Path) -> Dict[str, Any]:
    """Failure dict returned when only_approved=True but 0 samples exist."""
    err_msg = "Onaylanmış eğitim karesi bulunamadı. Lütfen önce Step 2'de kareleri onaylayın."
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


def _invalid_yaml_result(
    yaml_path: Path, errors: List[str], only_approved: bool
) -> Dict[str, Any]:
    """Failure dict returned when the training yaml fails pre-flight validation.

    P0-5 (K-30): the worker thread / Ultralytics must never be the first
    thing to discover a bad yaml. The concrete reasons travel in `error` /
    `reason` so the UI can show them verbatim instead of a generic
    "Eğitim başarısız.".
    """
    err_msg = "Eğitim yaml'ı geçersiz: " + "; ".join(errors)
    return {
        "started": False,
        "status": "failed",
        "state": "failed",
        "error": err_msg,
        "reason": err_msg,
        "only_approved": bool(only_approved),
        "use_approved_only": bool(only_approved),
        "approved_count": 0,
        "data_yaml_path": str(yaml_path),
    }


def _engine_supports_kwarg(engine: Any, method_name: str, kwarg: str) -> bool:
    """True when `engine.<method_name>` accepts `kwarg` (best effort)."""
    try:
        return kwarg in inspect.signature(getattr(engine, method_name)).parameters
    except Exception:
        return False


def _annotate_training_result(
    result: Any,
    only_approved: bool,
    sample_count: int,
    yaml_path: Path,
) -> Any:
    """Adds trainer bookkeeping fields to an engine result dict (in place)."""
    if isinstance(result, dict):
        result["only_approved"] = only_approved
        result["use_approved_only"] = only_approved
        result["sample_count"] = sample_count
        result["selected_yaml"] = str(yaml_path)
    return result


def classify_training_result(
    result: Any,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Pure classification of an engine/worker training result.

    Returns `(outcome, message, status_line)`:
    - `("finished", None, None)` — training completed; caller emits the result.
    - `("deferred", reason, "Eğitim Durumu: {reason}")` — game still running.
    - `("failed", err, status_line_or_None)` — caller emits training_failed(err),
      plus status_text(status_line) when one is given (stopped case only).

    Extracted verbatim (same branch order, same strings) from
    `YOLOTrainingWorker.run()` so every branch is unit-testable without
    spinning up a thread.
    """
    if isinstance(result, dict) and (
        result.get("waiting_for_game_exit") or result.get("reason") == "aion2_running"
    ):
        reason = "Eğitim, oyun kapanana kadar bekliyor."
        return "deferred", reason, f"Eğitim Durumu: {reason}"

    if isinstance(result, dict) and (
        result.get("stopped")
        or result.get("interrupted")
        or result.get("status") == "stopped"
        or result.get("state") == "stopped"
    ):
        err = str(result.get("reason") or "Eğitim kullanıcı tarafından durduruldu.")
        return "failed", err, f"Eğitim Durumu: Durduruldu ({err})"

    if isinstance(result, dict) and (
        result.get("started") is False
        or result.get("status") == "failed"
        or result.get("state") in ("failed", "refused", "busy")
    ):
        err = str(result.get("reason") or result.get("error") or "Eğitim başarısız.")
        return "failed", err, None

    # F-1972: a None / empty / content-free result is a failure,
    # never a silent "success".
    if not isinstance(result, dict) or not result:
        return "failed", "Eğitim sonucu boş/belirsiz — eğitim başarısız sayıldı.", None

    if (
        result.get("status") not in ("success", "completed")
        and result.get("state") not in ("completed", "success")
        and result.get("started") is not True
    ):
        err = str(result.get("reason") or result.get("error") or "Eğitim sonucu başarısız sayıldı.")
        return "failed", err, None

    return "finished", None, None


def count_labeled_images(images_dir: Path, labels_dir: Path) -> int:
    """Images with a non-empty same-stem `.txt` label — the ones YOLO will
    actually train on.

    `count_images_in_dir` counts every image, including label-less ones
    YOLO silently drops (dilim 05 F9, dilim 27 B-05). Callers that report
    "N kareyle eğitim" must use this, not the raw image count.
    """
    if not images_dir.is_dir():
        return 0
    count = 0
    for f in images_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in (
            ".jpg",
            ".png",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tif",
            ".tiff",
        ):
            continue
        lbl = labels_dir / f"{f.stem}.txt"
        try:
            if lbl.is_file() and lbl.read_text(encoding="utf-8", errors="ignore").strip():
                count += 1
        except OSError:
            continue
    return count


def count_images_missing_labels(images_dir: Path, labels_dir: Path) -> int:
    """Counts images in `images_dir` with no same-stem `.txt` in `labels_dir`.

    Diagnostic only: YOLO training silently drops label-less images, so
    `count_images_in_dir` overstates the usable sample count when this
    is > 0. Does not change any count; callers only warn on it.
    """
    if not images_dir.is_dir():
        return 0
    label_stems: Set[str] = set()
    if labels_dir.is_dir():
        label_stems = {f.stem for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"}
    missing = 0
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (
            ".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
        ) and f.stem not in label_stems:
            missing += 1
    return missing


def read_stale_manifest(models_dir: Optional[Union[str, Path]]) -> Tuple[bool, Optional[str]]:
    """Reads `models/stale_manifest.json` without raising.

    Returns `(is_stale, reason)`. `(False, None)` when the file is absent,
    malformed, or `model_stale` is not exactly `True` — a missing or broken
    manifest must never block training, it only means "no stale signal".
    """
    if models_dir is None:
        return False, None
    manifest = Path(models_dir) / "stale_manifest.json"
    if not manifest.is_file():
        return False, None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if not isinstance(data, dict) or data.get("model_stale") is not True:
        return False, None
    reason = data.get("reason")
    return True, str(reason) if reason is not None else None


def _write_fallback_approved_yaml(approved_dir: Path, approved_yaml_path: Path) -> Path:
    """Writes canonical 27-class `data_approved.yaml` into `approved_dir`."""
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (approved_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (approved_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (approved_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    # B-09 DÜZELTME (2026-11-10, canlı kanıt): `path:` MUTLAK yol
    # yazar — eski göreli `.` YANLIŞTI. Ultralytics göreli path'i
    # yaml'ın dizinine göre DEĞİL, eğitim sürecinin çalışma klasörüne
    # (CWD, PC'de F:\a2) göre çözer; `.` yazılınca eğitim
    # F:\a2\images\val'i aradı ve düştü. Bu fallback yaml da aynı
    # data_approved.yaml'ı yazdığı için aynı hataya düşerdi.
    # Taşınabilirlik doğruluk karşısında ikincildir; yaml her
    # hazırlıkta yeniden üretilir. Windows uyumu için as_posix.
    names = _canonical_class_names()
    lines = [
        "# Ultralytics YOLO Configuration - Approved Dataset Pool (A09 Standalone Fallback)",
        f"path: {Path(approved_dir).resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for cid in sorted(names.keys()):
        lines.append(f"  {cid}: {names[cid]}")
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
                # F-1974: a pool-path mismatch must NEVER delete user data
                # (the old code removed a whole dataset/dataset tree).
                # Just use a correctly scoped pool and warn.
                logger.warning(
                    "Approved pool singleton points at %s, expected %s; "
                    "using a scoped pool instead (no files deleted).",
                    pool_root_resolved,
                    approved_dir.resolve(),
                )
                scoped_pool = ApprovedDatasetPool(pool_root=approved_dir, project_root=proj_root)
                scoped_pool.sync_from_disk()
                prep_result = scoped_pool.prepare_approved_training_dataset(source_dataset_dir=ds_dir)
                try:
                    dataset_factory.approved_pool._GLOBAL_POOL = scoped_pool
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("Scoped ApprovedDatasetPool reconciliation skipped: %s", exc)

    # P1-09 (K-24): dondurma-sayı kapısı — eğitim, dondurulan setle
    # başlamalı. frozen_manifest.json varsa ve approved_pool kaynaklıysa
    # sayıları havuzla uyuşmalı; uyumsuzluk eğitimi başlatmaz (donmuş set
    # bayatlamıştır — Adım 3'te yeniden dondurmak gerekir). Manifesto
    # yoksa yalnızca uyarılır (dondurmasız akışlar kilitlenmez).
    frozen_check = dataset_factory.approved_pool.check_frozen_manifest_consistency(
        dataset_dir=ds_dir
    )
    if frozen_check.get("checked") and not frozen_check.get("ok"):
        raise RuntimeError(
            frozen_check.get("reason") or "Dondurulan veri seti havuzla uyuşmuyor."
        )
    if not frozen_check.get("checked"):
        logger.warning("[EĞİTİM] %s", frozen_check.get("reason") or "Dondurma karşılaştırması yapılamadı.")

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

        Counts only images with a non-empty label file (dilim 05 F9):
        YOLO silently drops label-less images, so the raw image count
        overstated the usable sample count. Disk is the truth — no
        manifest/pool fallback: a stale manifest must not inflate the
        number the training status line reports.
        """
        train_img_dir = self.approved_dir / "images" / "train"
        val_img_dir = self.approved_dir / "images" / "val"
        train_lbl_dir = self.approved_dir / "labels" / "train"
        val_lbl_dir = self.approved_dir / "labels" / "val"
        return count_labeled_images(train_img_dir, train_lbl_dir) + count_labeled_images(
            val_img_dir, val_lbl_dir
        )

    def count_total_dataset_samples(self) -> int:
        """
        Calculates the number of total images in `dataset/images/{train,val}`.

        Counts only images with a non-empty label file (dilim 27 B-03/B-05).
        Disk is the truth: the old `manifest_stats.json` fallback let a
        stale cache claim frames that no longer exist on disk.
        """
        train_img_dir = self.dataset_dir / "images" / "train"
        val_img_dir = self.dataset_dir / "images" / "val"
        train_lbl_dir = self.dataset_dir / "labels" / "train"
        val_lbl_dir = self.dataset_dir / "labels" / "val"
        return count_labeled_images(train_img_dir, train_lbl_dir) + count_labeled_images(
            val_img_dir, val_lbl_dir
        )

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
            # A02: a custom yaml is silently ignored on this branch today —
            # warn instead of letting the caller believe it was honoured.
            if custom_yaml is not None:
                logger.warning(
                    "custom_yaml verildi (%s) ama only_approved=True; "
                    "özel yaml yok sayılıyor, data_approved.yaml kullanılıyor.",
                    custom_yaml,
                )
            # A02: yolo_trainer never consulted models/stale_manifest.json —
            # warn when training starts while the published model is stale.
            stale, stale_reason = read_stale_manifest(self.models_dir)
            if stale:
                logger.warning(
                    "[EĞİTİM] Model bayat olarak işaretli (models/stale_manifest.json): %s",
                    stale_reason or "sebep belirtilmemiş",
                )
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

            # A02: count counts images even when their .txt label is missing
            # (YOLO drops those silently); warn with the measured number.
            missing_labels = count_images_missing_labels(
                self.approved_dir / "images" / "train", self.approved_dir / "labels" / "train"
            ) + count_images_missing_labels(
                self.approved_dir / "images" / "val", self.approved_dir / "labels" / "val"
            )
            if missing_labels > 0:
                logger.warning(
                    "[EĞİTİM] %d onaylı görselin etiket (.txt) dosyası yok; "
                    "bu kareler eğitimde sessizce atlanır, sayılan %d kare kullanılabilir olandan fazla olabilir.",
                    missing_labels,
                    count,
                )

            log_line = _training_status_message(True, count)
            logger.info(log_line)
            print(log_line, flush=True)

            return yaml_p, count
        else:
            if custom_yaml is not None:
                yaml_p = Path(custom_yaml).resolve()
                # A02: a nonexistent custom yaml was silently forwarded to
                # the engine, which failed later with an unrelated error.
                if not yaml_p.is_file():
                    logger.warning(
                        "custom_yaml dosyası bulunamadı: %s; motor bu yolla çağrılacak ve hata verebilir.",
                        yaml_p,
                    )
            else:
                yaml_p = prepare_and_resolve_training_yaml(
                    dataset_dir=self.dataset_dir,
                    only_approved=False,
                ).resolve()

            # F-1990: when a custom yaml is given, count ITS train/val
            # images instead of reporting the whole dataset total.
            if custom_yaml is not None and Path(custom_yaml).is_file():
                yaml_count = _count_images_for_yaml(Path(custom_yaml))
                count = yaml_count if yaml_count > 0 else self.count_total_dataset_samples()
            else:
                count = self.count_total_dataset_samples()

            log_line = _training_status_message(False, count)
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
        allow_opencv_fallback: bool = False,
        use_approved_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes YOLO training synchronously.

        Points strictly to `dataset/approved/data_approved.yaml` when `only_approved=True`
        (or `use_approved_only=True`).
        """
        only_approved, yaml_path, sample_count, early_result = self._prepare_training_run(
            only_approved=only_approved,
            data_yaml_path=data_yaml_path,
            status_callback=status_callback,
            use_approved_only=use_approved_only,
            emit_error_status=True,
        )
        if early_result is not None:
            return early_result

        epochs, imgsz, batch, model = normalize_training_params(epochs, imgsz, batch, model)

        engine = self.get_engine()
        call_kwargs: Dict[str, Any] = dict(kwargs)
        if _engine_supports_kwarg(engine, "run_training_sync", "use_approved_only"):
            call_kwargs["use_approved_only"] = only_approved

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

        return _annotate_training_result(result, only_approved, sample_count, yaml_path)

    def _prepare_training_run(
        self,
        only_approved: bool,
        data_yaml_path: Optional[Union[str, Path]],
        status_callback: Optional[Callable[[str], None]],
        use_approved_only: Optional[bool],
        emit_error_status: bool,
    ) -> Tuple[bool, Path, int, Optional[Dict[str, Any]]]:
        """Shared prologue of `run_training_sync` / `start_training_async`.

        Normalizes the approved-only flag, resolves the dataset yaml, emits
        the start status line and, when only_approved=True with 0 samples,
        returns the early failure dict as the 4th element (None otherwise).
        `emit_error_status` preserves the historical asymmetry: the sync path
        reports the empty-approved error via status_callback ("[HATA] ..."),
        the async path does not.
        """
        if use_approved_only is not None:
            only_approved = bool(use_approved_only)
        # F-1987: a data_approved.yaml filename no longer silently overrides
        # an explicit only_approved=False — the caller's flag wins; warn.
        if (
            data_yaml_path is not None
            and Path(data_yaml_path).name.lower() == "data_approved.yaml"
            and not only_approved
        ):
            logger.warning(
                "data_approved.yaml verildi ama only_approved=False; "
                "çağrıcının seçimi geçerli sayılıyor."
            )

        yaml_path, sample_count = self.resolve_training_data_yaml(
            only_approved=only_approved,
            custom_yaml=data_yaml_path,
        )

        # P0-5 (K-30): pre-flight yaml validation — path absolute, train/val
        # dirs exist, > 0 labeled frames. This is the trainer-side gate for
        # every entry point (sync, async, worker, .bat): a bad yaml must
        # fail HERE with a concrete reason, not inside the worker thread or
        # inside Ultralytics. The dialog shows the same check before the
        # worker even starts (on_train_rtx_clicked); this is the second
        # line of defence for non-UI callers.
        yaml_report = validate_training_yaml(yaml_path)
        if not yaml_report["ok"]:
            # P0-5 tur 2: only_approved + TEK hata 0-etiketli-kare ise
            # eski boş-havuz sözleşmesi korunur ("Onaylanmış eğitim
            # karesi bulunamadı") — yaml kapısı ile eski kapı aynı
            # koşulu yakalar; dışarıya dönük mesaj değişmez. Başka bir
            # geçersizlik (göreli path, eksik dizin, eksik dosya) varsa
            # yaml mesajı aynen kalır — ayırt edicilik korunur.
            _yaml_errors = list(yaml_report["errors"])
            if (
                only_approved
                and len(_yaml_errors) == 1
                and "etiketli kare yok" in _yaml_errors[0]
            ):
                empty_result = _empty_approved_result(yaml_path)
                logger.warning("[EĞİTİM] %s", empty_result["error"])
                if emit_error_status and callable(status_callback):
                    try:
                        status_callback(f"[HATA] {empty_result['error']}")
                    except Exception:
                        pass
                return only_approved, yaml_path, sample_count, empty_result
            invalid_result = _invalid_yaml_result(
                yaml_path, _yaml_errors, only_approved
            )
            logger.warning("[EĞİTİM] %s", invalid_result["error"])
            if emit_error_status and callable(status_callback):
                try:
                    status_callback(f"[HATA] {invalid_result['error']}")
                except Exception:
                    pass
            return only_approved, yaml_path, sample_count, invalid_result

        status_msg = _training_status_message(only_approved, sample_count)
        if callable(status_callback):
            try:
                status_callback(status_msg)
            except Exception:
                pass

        if only_approved and sample_count == 0:
            err_msg = "Onaylanmış eğitim karesi bulunamadı. Lütfen önce Step 2'de kareleri onaylayın."
            logger.warning("[EĞİTİM] %s", err_msg)
            if emit_error_status and callable(status_callback):
                try:
                    status_callback(f"[HATA] {err_msg}")
                except Exception:
                    pass
            return only_approved, yaml_path, sample_count, _empty_approved_result(yaml_path)

        return only_approved, yaml_path, sample_count, None

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
        allow_opencv_fallback: bool = False,
        use_approved_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Starts training asynchronously on a background thread.
        """
        only_approved, yaml_path, sample_count, early_result = self._prepare_training_run(
            only_approved=only_approved,
            data_yaml_path=data_yaml_path,
            status_callback=status_callback,
            use_approved_only=use_approved_only,
            emit_error_status=False,
        )
        if early_result is not None:
            return early_result

        epochs, imgsz, batch, model = normalize_training_params(epochs, imgsz, batch, model)

        engine = self.get_engine()
        call_kwargs: Dict[str, Any] = dict(kwargs)
        if _engine_supports_kwarg(engine, "start_training_async", "use_approved_only"):
            call_kwargs["use_approved_only"] = only_approved

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

        return _annotate_training_result(result, only_approved, sample_count, yaml_path)

    def stop_training(self) -> None:
        """Requests training stop."""
        try:
            engine = self.get_engine()
            engine.stop_training()
        except Exception as exc:
            # Was a silent `pass`: a failing stop request left no trace.
            logger.debug("stop_training failed (ignored): %s", exc)


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
        allow_opencv_fallback: bool = False,
        force_fallback: bool = False,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.epochs, self.imgsz, self.batch, self.model = normalize_training_params(
            epochs, imgsz, batch, model
        )
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
        # F-1971: defensive — never crash on a thread object without the
        # interruption API (e.g. a plain threading.Thread).
        interruption = False
        if hasattr(self, "isInterruptionRequested"):
            try:
                interruption = bool(self.isInterruptionRequested())
            except Exception:
                interruption = False
        return bool(getattr(self, "_stop_requested", False) or interruption)

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
        # F-1996: when a stop was requested, stop and do NOT emit another
        # progress update afterwards.
        if self.is_stop_requested():
            self.stop()
            return
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

            status_line = _training_status_message(self.only_approved, sample_count)
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

            # 5. Handle training result (pure classifier: same branches/strings).
            outcome, message, status_line = classify_training_result(result)
            if status_line is not None:
                self.status_text.emit(status_line)
            if outcome == "deferred":
                self.training_deferred.emit(message)
            elif outcome == "failed":
                self.training_failed.emit(message)
            else:
                self.training_finished.emit(result)

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
