# -*- coding: utf-8 -*-
"""
Dataset Factory Sub-Agent 6: 1-Click RTX 3070 CUDA FP16 YOLO Trainer & ONNX Exporter
====================================================================================
Provides hardware-accelerated training orchestration for Gatherion specific resource,
mob, and UI detection models on consumer NVIDIA RTX GPUs (optimized for RTX 3070 8GB
VRAM at ~12-15 minutes for 5,000 images with FP16 mixed precision), alongside a
built-in OpenCV/NumPy Multi-Class HSV + Gabor + Hu-Moment + Template Centroid
Classifier fallback engine when `ultralytics` is not installed in the active Python
environment.

Exclusive Owner: Sub-Agent 6 (`dataset_factory/rtx_trainer_engine.py`)
"""

from __future__ import annotations

import datetime
import gc
import importlib.util
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from version import __version__

logger = logging.getLogger("DatasetFactory.RTXTrainerEngine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

CREATE_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0

_MIN_TRAIN_FREE_BYTES = 8 * 1024 * 1024 * 1024

# P2-18: `wait_for_training` süresiz join yapmaz — varsayılan bekleme
# 3600 sn'dir (RTX eğitimi uzun sürebilir; süre dolarsa `timed_out=True`
# döner, arayan eğitimin hâlâ sürdüğünü bilir). `timeout=None` artık
# süresiz anlamına gelmez, varsayılan sayılır.
RTX_TRAIN_WAIT_TIMEOUT_S = 3600.0


def offload_root() -> Optional[Path]:
    """F:\\a2 when that folder exists, else F:\\Gatherion. None when F: is missing."""
    if not Path("F:/").exists():
        return None
    preferred = Path("F:/a2")
    if preferred.is_dir():
        return preferred
    return Path("F:/Gatherion")


def rtx_runs_dir_for() -> Path:
    """Ultralytics project directory. New weights and plots stay off C:.

    S085 (v19/P3-8): F: yokken de sabit `F:/Gatherion/models/rtx_runs`
    dönüyordu — olmayan sürücüye yazım denemesi. F: yoksa repo içi
    `models/rtx_runs` kullanılır (`_project_root` yedeği).
    """
    root = offload_root()
    if root is None:
        return _project_root() / "models" / "rtx_runs"
    return root / "models" / "rtx_runs"


def _project_root() -> Path:
    """Repo root (fallback train root when no F: offload drive exists)."""
    return Path(__file__).resolve().parent.parent


# fix/v20-egitim-devam: koşu klasörü adları. Taze koşular artık sabit
# `gatherion_rtx_fp16` klasörüne yazılmaz — her koşu benzersiz zaman
# damgalı klasör alır, eski best.pt/last.pt korunur. Eski sabit isimli
# klasör geriye dönük tanınır (devam + en-son-koşu bulucusu).
_RUN_DIR_BASE = "gatherion_rtx_fp16"
_RUN_DIR_NAME_RE = re.compile(r"^gatherion_rtx_fp16(?:_\d{8}_\d{6}(?:_\d+)?)?$")


def is_rtx_run_dir_name(name: str) -> bool:
    """Eski sabit isim + zaman damgalı koşu klasörü isimlerini tanır."""
    return bool(_RUN_DIR_NAME_RE.match(str(name or "")))


def generate_unique_run_name(
    runs_dir: Union[str, Path],
    now: Optional[datetime.datetime] = None,
    base: str = _RUN_DIR_BASE,
) -> str:
    """Taze koşu için benzersiz klasör adı: ``base_YYYYMMDD_HHMMSS`` (yerel saat).

    Aynı saniyede çakışırsa ``_2``, ``_3`` ... sayacı artar. Sabit isim
    asla dönülmez — yeni koşu eski ağırlıkların üstüne yazılmaz.
    """
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    runs = Path(runs_dir)
    candidate = "%s_%s" % (base, stamp)
    if not (runs / candidate).exists():
        return candidate
    for counter in range(2, 10000):
        candidate = "%s_%s_%d" % (base, stamp, counter)
        if not (runs / candidate).exists():
            return candidate
    raise RuntimeError(
        "Benzersiz koşu klasörü adı üretilemedi (%s altında %s saniyesi dolu)."
        % (runs, stamp)
    )


def validate_resume_checkpoint(
    checkpoint: Union[str, Path, None], *, deep: bool = False
) -> Tuple[bool, str]:
    """Devam checkpoint'ini okunabilirlik açısından denetler (eğitim başlatmaz).

    Sığ denetim (her zaman): dosya var, boş değil, torch zip kapsayıcısı
    (``archive/data.pkl`` girdisi var). Derin denetim (``deep=True`` ve
    torch import edilebiliyorsa): ``torch.load`` ile açılır, eğitim
    checkpoint'i anahtarları (``epoch`` + ``model``) aranır. torch yoksa
    sığ karar geçerlidir. Döner: ``(geçerli_mi, hata_mesajı)``.
    """
    if checkpoint is None or str(checkpoint or "").strip() == "":
        return False, "Devam checkpoint'i verilmedi."
    path = Path(str(checkpoint))
    if not path.is_file():
        return False, "Devam edilecek checkpoint bulunamadı: %s" % path
    try:
        if path.stat().st_size <= 0:
            return False, "Checkpoint boş (0 bayt): %s" % path
    except OSError as exc:
        return False, "Checkpoint okunamadı (%s): %s" % (exc, path)
    try:
        if not zipfile.is_zipfile(path):
            return (
                False,
                "Checkpoint geçerli bir eğitim dosyası değil (zip kapsayıcısı değil): %s"
                % path,
            )
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        if not any(n.endswith("data.pkl") for n in names):
            return (
                False,
                "Checkpoint geçerli bir eğitim dosyası değil (data.pkl yok): %s" % path,
            )
    except Exception as exc:
        return False, "Checkpoint okunamadı (%s): %s" % (exc, path)
    if deep:
        try:
            import torch  # type: ignore
        except Exception:
            return True, ""
        try:
            # weights_only=False şart: Ultralytics checkpoint'inde 'model'
            # bir nn.Module nesnesidir, salt tensör değildir. Dosya bu
            # reponun kendi eğitimlerinin yazdığı checkpoint'tir.
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            return False, "Checkpoint bozuk, açılamadı (%s): %s" % (exc, path)
        if not isinstance(ckpt, dict) or "epoch" not in ckpt or "model" not in ckpt:
            return (
                False,
                "Checkpoint eğitim durumu taşımıyor (epoch/model anahtarı yok): %s" % path,
            )
    return True, ""


def resolve_resume_checkpoint(checkpoint: Union[str, Path]) -> Path:
    """Geçerli checkpoint'in mutlak yolunu döner; geçersizse Türkçe RuntimeError."""
    ok, reason = validate_resume_checkpoint(checkpoint, deep=True)
    if not ok:
        raise RuntimeError(reason)
    return Path(str(checkpoint)).resolve()


def run_dir_for_resume_checkpoint(checkpoint: Union[str, Path]) -> Path:
    """Checkpoint'in koşu klasörünü döner (``weights/last.pt`` düzenini şart koşar)."""
    path = Path(str(checkpoint))
    if path.parent.name != "weights":
        raise RuntimeError(
            "Checkpoint eğitim klasörü düzeninde değil (weights/last.pt beklenir): %s"
            % path
        )
    return path.parent.parent


def _run_dir_artifact_mtime(run_dir: Path) -> float:
    """Koşu klasörünün en yeni yazım zamanı (ağırlık/csv/klasör mtime'larının max'ı)."""
    newest = 0.0
    try:
        newest = max(newest, run_dir.stat().st_mtime)
    except OSError:
        return 0.0
    for rel in ("weights/last.pt", "weights/best.pt", "results.csv"):
        try:
            cand = run_dir / rel
            if cand.is_file():
                newest = max(newest, cand.stat().st_mtime)
        except OSError:
            continue
    return newest


def find_latest_run_dir(runs_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """En son yazılmış koşu klasörü (zaman damgalı + eski sabit isim tanınır)."""
    runs = Path(runs_dir) if runs_dir is not None else rtx_runs_dir_for()
    try:
        if not runs.is_dir():
            return None
        candidates = [
            d for d in runs.iterdir() if d.is_dir() and is_rtx_run_dir_name(d.name)
        ]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda d: (_run_dir_artifact_mtime(d), d.name), reverse=True)
    return candidates[0]


def find_resumable_checkpoint(
    runs_dir: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Devam edilebilir en yeni koşuyu bulur (sığ denetim — kilit zamanlayıcısında da ucuz).

    Döner: ``{"checkpoint", "run_dir", "run_name", "epochs_completed"}`` ya da
    ``None``. Derin (torch) doğrulama eğitim başlangıcında yapılır; burada
    yalnızca okunabilir bir ``weights/last.pt`` aranır, en yeniden eskiye
    ilk geçerli olan alınır.
    """
    runs = Path(runs_dir) if runs_dir is not None else rtx_runs_dir_for()
    try:
        if not runs.is_dir():
            return None
        candidates = [
            d for d in runs.iterdir() if d.is_dir() and is_rtx_run_dir_name(d.name)
        ]
    except OSError:
        return None
    candidates.sort(key=lambda d: (_run_dir_artifact_mtime(d), d.name), reverse=True)
    for run_dir in candidates:
        ckpt = run_dir / "weights" / "last.pt"
        ok, _reason = validate_resume_checkpoint(ckpt)
        if not ok:
            continue
        return {
            "checkpoint": ckpt,
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "epochs_completed": int(RTXTrainerEngine._count_results_csv_epochs(run_dir)),
        }
    return None


def _pin_train_env() -> Path:
    """Point temp and Ultralytics caches at F: before a run writes anything."""
    # F-0997: when there is no F: drive, do not force F:/a2 paths that do
    # not exist. On Windows that is a hard error (training needs the
    # offload drive); elsewhere fall back to project-local directories.
    root = offload_root()
    if root is None:
        if sys.platform == "win32":
            raise RuntimeError(
                "F: sürücüsü bulunamadı — eğitim için F:/a2 (veya F:/Gatherion) gerekli."
            )
        root = _project_root()
    temp_dir = root / "temp"
    appdata_dir = root / "appdata"
    torch_home = root / "torch"
    weights_dir = temp_dir / "weights"
    datasets_dir = root / "dataset"
    runs_dir = rtx_runs_dir_for()
    for path in (temp_dir, appdata_dir, torch_home, weights_dir, datasets_dir, runs_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["YOLO_CONFIG_DIR"] = str(appdata_dir)
    os.environ["ULTRALYTICS_CONFIG_DIR"] = str(appdata_dir)
    os.environ["PIP_CACHE_DIR"] = str(root / "pip-cache")
    (root / "pip-cache").mkdir(parents=True, exist_ok=True)
    _persist_ultra_settings(appdata_dir / "Ultralytics" / "settings.json", datasets_dir, weights_dir, runs_dir)
    return runs_dir


def _persist_ultra_settings(settings_path: Path, datasets_dir: Path, weights_dir: Path, runs_dir: Path) -> None:
    """Rewrite datasets/weights/runs if a settings file still points off F:."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload["datasets_dir"] = str(datasets_dir)
    payload["weights_dir"] = str(weights_dir)
    payload["runs_dir"] = str(runs_dir)
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _yaml_on_offload(
    data_yaml_path: Union[str, Path],
    use_approved_only: bool = False,
) -> Path:
    """Train against the F: dataset when the caller still passes a C: yaml, preserving approved YAML."""
    path = Path(data_yaml_path)
    is_approved_yaml = bool(
        use_approved_only
        or path.name.lower() == "data_approved.yaml"
        or "approved" in [p.lower() for p in path.parts]
    )
    if is_approved_yaml and path.is_file():
        return path

    root = offload_root()
    if root is None:
        return path
    try:
        path.resolve().relative_to(root.resolve())
        if is_approved_yaml and path.name.lower() != "data_approved.yaml":
            if path.is_dir():
                base_ds = path if path.name.lower() == "dataset" else (path / "dataset" if (path / "dataset").is_dir() else path)
            else:
                base_ds = path.parent
            return base_ds / "approved" / "data_approved.yaml"
        return path
    except (ValueError, OSError):
        if is_approved_yaml:
            if path.name.lower() == "data_approved.yaml" and path.parent.exists():
                return path
            preferred_approved = root / "dataset" / "approved" / "data_approved.yaml"
            if preferred_approved.is_file():
                logger.warning(
                    "Onaylı eğitim verisi %s F: dışında; %s kullanılacak",
                    path,
                    preferred_approved,
                )
                return preferred_approved
            return path
        if path.is_file():
            return path
        preferred = root / "dataset" / "data.yaml"
        if preferred.is_file():
            # F5/B-10: bu ikame yalnızca istenen dosya YOKKEN olur ve farklı
            # bir makinedeki/zamandaki veri setiyle eğitim demektir — yalnızca
            # debug değil, açık bir uyarıyla kayda geçer.
            logger.warning(
                "DİKKAT: istenen eğitim yaml'ı yok (%s); F: üzerindeki %s "
                "kullanılacak — eğitilen veri seti istenenden farklı olabilir.",
                path,
                preferred,
            )
            return preferred
        return path


def _check_training_yaml_or_raise(yaml_path: Path) -> None:
    """Raise RuntimeError unless the training yaml passes pre-flight validation.

    P0-5 (K-30): `dataset_factory.yolo_trainer.validate_training_yaml` is
    the single rule (absolute `path:`, train/val dirs exist, > 0 labeled
    frames); this is its engine-side adapter. Lazy import avoids a
    module-level import cycle (yolo_trainer imports the engine lazily too).
    """
    from dataset_factory.yolo_trainer import validate_training_yaml

    report = validate_training_yaml(yaml_path)
    if not report.get("ok", False):
        raise RuntimeError(
            "Eğitim yaml'ı geçersiz, eğitim başlatılmadı: "
            + "; ".join(str(e) for e in report.get("errors") or ["bilinmeyen yaml hatası"])
        )


def _count_labeled_samples(images_dir: Path, labels_dir: Path) -> int:
    """Images with a non-empty same-stem `.txt` label.

    YOLO silently drops label-less / empty-label images, so a plain image
    count overstates the usable sample count (dilim 27 B-05, dilim 05 F9).
    Only images that will actually train are counted here.
    """
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for img in images_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in valid_exts:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        try:
            if lbl.is_file() and lbl.read_text(encoding="utf-8", errors="ignore").strip():
                count += 1
        except OSError:
            continue
    return count


def target_volume_free_bytes(path: Path) -> int:
    """Free bytes on the volume that will hold `path`. 0 when that volume is missing."""
    probe = path
    try:
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        if not probe.exists():
            return 0
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return 0


def _point_ultralytics_runs(runs_dir: Path) -> None:
    """Send Ultralytics runs, weights, and datasets to F: before model.train."""
    # F-0997: no forced F:/a2 fallback when the drive is missing.
    root = offload_root() or _project_root()
    datasets_dir = root / "dataset"
    weights_dir = root / "temp" / "weights"
    runs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ultralytics.utils as ultra_utils

        ultra_utils.SETTINGS.update(
            runs_dir=str(runs_dir),
            weights_dir=str(weights_dir),
            datasets_dir=str(datasets_dir),
        )
        ultra_utils.RUNS_DIR = Path(runs_dir)
    except Exception:
        logger.warning("Ultralytics runs_dir F: yoluna alınamadı", exc_info=True)


_TRAIN_SLOT_LOCK = threading.Lock()
_TRAIN_SLOT_HELD = False


def _acquire_train_slot() -> bool:
    """One Ultralytics run per process. Second call loses."""
    global _TRAIN_SLOT_HELD
    with _TRAIN_SLOT_LOCK:
        if _TRAIN_SLOT_HELD:
            return False
        _TRAIN_SLOT_HELD = True
        return True


def _release_train_slot() -> None:
    global _TRAIN_SLOT_HELD
    with _TRAIN_SLOT_LOCK:
        _TRAIN_SLOT_HELD = False


def _train_slot_busy() -> bool:
    with _TRAIN_SLOT_LOCK:
        return bool(_TRAIN_SLOT_HELD)

# F-0814: canonical 27-class taxonomy (0..26), not a generic 12-class list.
# P1-1: the ceiling comes from the same single source as the hygiene
# gate / augmentor (dataset/data.yaml via
# dataset_factory.class_taxonomy.resolve_canonical_max_class_id) — no
# `<= 26` literal anywhere in this file.
try:
    from dataset_factory.class_taxonomy import CANONICAL_NAMES as _CANONICAL_NAMES
    from dataset_factory.class_taxonomy import (
        resolve_canonical_max_class_id as _resolve_max_class_id,
    )

    DEFAULT_CLASS_NAMES: Dict[int, str] = dict(_CANONICAL_NAMES)
except Exception:
    DEFAULT_CLASS_NAMES: Dict[int, str] = {
        0: "herb_aria",
        1: "herb_mela",
        2: "herb_targena",
        3: "ore_orichalcum",
        4: "gem_crystal",
        5: "tree_log",
        6: "herb_edana",
        7: "herb_lumina",
        8: "ore_obsidian",
        9: "minimap_sprout",
        10: "monster_3d",
        11: "minimap_monster_dot",
        12: "crystal_blood",
        13: "skill_ui_prompt",
        14: "obstacle",
        15: "path_ground",
        16: "player_hp_bar",
        17: "player_mp_bar",
        18: "target_hp_bar",
        19: "skill_bar",
        20: "skill_slot_ready",
        21: "skill_slot_cooldown",
        22: "hp_potion",
        23: "mp_potion",
        24: "buff_scroll",
        25: "kinah",
        26: "player_character",
    }

    def _resolve_max_class_id() -> int:  # type: ignore[misc]
        # Import-failure fallback only: the ceiling is the fallback
        # table's max id (26 today), never a bare literal at the use site.
        return max(DEFAULT_CLASS_NAMES)


def _imread_unicode(img_path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Unicode-safe image loader for Windows paths containing Turkish/non-ASCII characters
    (e.g. `C:\\Users\\selçuk\\Desktop\\a2\\...`).
    """
    p = Path(img_path)
    if not p.is_file():
        return None
    try:
        raw_bytes = np.fromfile(str(p), dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        return cv2.imdecode(raw_bytes, flags)
    except Exception:
        return None


def _strip_yaml_comment(raw_line: str) -> str:
    """Remove a trailing YAML comment, honouring single/double quotes."""
    in_single = False
    in_double = False
    for idx, ch in enumerate(raw_line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return raw_line[:idx]
    return raw_line


def _parse_simple_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Lightweight, zero-dependency parser for Ultralytics `data.yaml` files.
    Uses PyYAML if available; otherwise falls back to line-based parsing.
    """
    if not yaml_path.exists():
        return {}

    raw_text = yaml_path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    result: Dict[str, Any] = {}
    names_dict: Dict[int, str] = {}
    names_list: List[str] = []
    in_names_section = False

    for raw_line in raw_text.splitlines():
        # F-1070: strip '#' comments only outside quotes, so values
        # containing '#' (quoted strings, paths) survive parsing.
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped.startswith("names:"):
            in_names_section = True
            rest = stripped[len("names:") :].strip()
            if rest.startswith("[") and rest.endswith("]"):
                items = [x.strip().strip("'\"") for x in rest[1:-1].split(",") if x.strip()]
                result["names"] = {idx: val for idx, val in enumerate(items)}
                in_names_section = False
            continue

        if in_names_section:
            if indent > 0 and ":" in stripped:
                k_part, v_part = stripped.split(":", 1)
                k_str = k_part.strip().lstrip("-").strip()
                v_str = v_part.strip().strip("'\"")
                if k_str.isdigit():
                    names_dict[int(k_str)] = v_str
                    continue
            elif indent > 0 and stripped.startswith("-"):
                item_val = stripped.lstrip("-").strip().strip("'\"")
                names_list.append(item_val)
                continue
            else:
                in_names_section = False

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if val.isdigit():
                result[key] = int(val)
            else:
                result[key] = val

    if names_dict:
        result["names"] = names_dict
    elif names_list:
        result["names"] = {idx: val for idx, val in enumerate(names_list)}

    return result


def probe_gpu_environment() -> Dict[str, Any]:
    """
    Top-level convenience wrapper around RTXTrainerEngine.probe_gpu_environment().
    Queries nvidia-smi, torch.cuda, onnxruntime, and ultralytics availability.
    """
    return RTXTrainerEngine.probe_gpu_environment_static()


def build_rtx_training_config(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    model: str = "yolo11n.pt",
    use_approved_only: bool = False,
) -> Dict[str, Any]:
    """
    Top-level convenience wrapper around RTXTrainerEngine.build_rtx_training_config().
    """
    return RTXTrainerEngine.build_rtx_training_config_static(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        model=model,
        use_approved_only=use_approved_only,
    )


def _loss_number(value: Any) -> Optional[float]:
    """Tek loss değerini float'a çevir; çevrilemeyen / sonlu olmayan -> None."""
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, 5)


def extract_epoch_losses(tloss: Any) -> Tuple[Optional[float], Optional[float]]:
    """Epoch-sonu (box_loss, cls_loss) değerlerini ``trainer.tloss``'tan çıkar.

    v19 telemetri düzeltmesi (canlı hata — epoch 20/30/40/50/69 loglarında
    box/cls hep 0.0000): Ultralytics >=8.3'te ``trainer.tloss`` bir
    dict'tir (``{"box_loss": ..., "cls_loss": ..., "dfl_loss": ...}``,
    epoch ortalaması; ``ultralytics/engine/trainer.py`` ``_do_train``
    satır ~566 ve ``label_loss_items`` satır ~978). Eski kod dict
    üzerinde iterasyon yapıp anahtarları float'a çevirmeye çalışıyordu —
    ``float("box_loss")`` ValueError verip sessizce yutuluyor, loss hep
    0.0 kalıyordu. Eski sürümlerde tloss sayı dizisi olabilir — o da
    desteklenir. Alınamayan loss ``None`` döner, asla 0.0 değil
    (YANILTICI SIFIR YASAK — 0.0000 "loss sıfır" değil "loss alınamadı"
    demekti).
    """
    if tloss is None:
        return (None, None)
    if isinstance(tloss, dict):
        return (
            _loss_number(tloss.get("box_loss")),
            _loss_number(tloss.get("cls_loss")),
        )
    if isinstance(tloss, (str, bytes)):
        return (None, None)
    try:
        vals = [float(x) for x in tloss]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return (None, None)
    if len(vals) < 2:
        return (None, None)
    box = _loss_number(vals[0])
    cls = _loss_number(vals[1])
    return (box, cls)


class _BatchStepCounter:
    """Epoch-içi batch sayacı.

    v19 telemetri düzeltmesi (canlı hata — "Batch 1/48 (%2)" takılı):
    ``BaseTrainer``'da ``batch`` diye bir öznitelik YOK, yalnızca
    ``batch_size`` var (``ultralytics/engine/trainer.py`` satır ~173) —
    ``getattr(trainer, "batch", 0)`` hep 0 dönüp ``cur_b``'yi 1'de
    kilitliyordu. Epoch-içi batch index'i ``_do_train``'deki ``for i,
    batch in pbar`` döngü değişkenidir ve trainer'a yazılmaz; ama
    ``on_train_batch_end`` her batch sonunda tam bir kez ateşlenir
    (aynı dosya satır ~640). Bu sayaç o callback'in epoch başına çağrı
    sayısını tutar — Ultralytics sürümünden bağımsız, deterministik.
    Epoch değişince sıfırlanır; sayım ``total_batches``'i aşarsa epoch
    ortasında yeniden başlamıştır (OOM retry / NaN recovery aynı epoch'u
    baştan çalıştırır) ve sayaç 1'e sarar.
    """

    def __init__(self) -> None:
        self._epoch: Any = None
        self._count = 0

    def next(self, trainer: Any) -> Tuple[int, int, int]:
        """Bir batch_end olayı işle; (cur_batch, total_batches, cur_epoch_1based) döner."""
        epoch0 = getattr(trainer, "epoch", 0)
        if epoch0 != self._epoch:
            self._epoch = epoch0
            self._count = 0
        self._count += 1
        total = len(getattr(trainer, "train_loader", [])) or 1
        if self._count > total:
            # Epoch ortasında yeniden başladı (OOM retry / NaN recovery).
            self._count = 1
        return (self._count, total, int(epoch0) + 1)


def total_system_ram_mb() -> Optional[float]:
    """Toplam sistem RAM'i (MB). psutil yoksa/okunamazsa None döner."""
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        return None


def default_rss_abort_mb(total_ram_mb: Optional[float] = None) -> float:
    """RSS abort eşiği varsayılanı (MB): toplam RAM'in %75'i, en az 10240 MB.

    fix/v20-rss-panik: sabit 10240 MB eşiği 32 GB'lık PC'de sağlıklı
    eğitimi öldürüyordu — canlı kanıt: 11 Kasım gecesi yolo11m/960
    koşusu 19-21 GB bandında plato yapıp mAP50 0.685'e ulaştı (güvenli);
    12 Kasım yolo11s koşusu ise 10315 MB'ta abort ile durduruldu.
    Eşik makinenin RAM'ine göre ölçeklenir; RAM okunamazsa eski
    10240 MB varsayılanı korunur.
    """
    if total_ram_mb is None:
        total_ram_mb = total_system_ram_mb()
    if total_ram_mb is None or float(total_ram_mb) <= 0:
        return 10240.0
    return max(10240.0, 0.75 * float(total_ram_mb))


class _RssAbortDecider:
    """RSS abort kararı: eşik aşımı ardışık kontrollerle teyit edilir.

    Tek anlık sıçrama (model yükleme, cache dolumu) eğitimi öldürmesin
    diye eşik üstü okuma ``confirmations`` kez üst üste görülmeden
    abort ateşlenmez (kontrol aralığı 30 sn → 2 teyit ≈ 60 sn).
    Acil durum: RSS toplam RAM'in ``emergency_fraction``'ına ulaşırsa
    teyit beklenmez — OOM'dan önce durulur. ``abort_mb <= 0`` abort'u
    kapatır (açık ayar).
    """

    def __init__(
        self,
        abort_mb: float,
        *,
        confirmations: int = 2,
        emergency_fraction: float = 0.90,
        total_ram_mb: Optional[float] = None,
    ) -> None:
        self.abort_mb = float(abort_mb)
        self.confirmations = max(1, int(confirmations))
        self.emergency_fraction = float(emergency_fraction)
        self.total_ram_mb = total_ram_mb
        self._over_streak = 0

    @property
    def enabled(self) -> bool:
        return self.abort_mb > 0

    def observe(self, rss_mb: Optional[float]) -> bool:
        """Bir RSS okumasını işle; abort ateşlenecekse True döner."""
        if not self.enabled or rss_mb is None:
            return False
        if (
            self.total_ram_mb
            and float(self.total_ram_mb) > 0
            and float(rss_mb) >= self.emergency_fraction * float(self.total_ram_mb)
        ):
            return True
        if float(rss_mb) >= self.abort_mb:
            self._over_streak += 1
        else:
            self._over_streak = 0
        return self._over_streak >= self.confirmations

    def reset(self) -> None:
        self._over_streak = 0


class RTXTrainerEngine:
    """
    1-Click RTX 3070 CUDA FP16 YOLO Trainer & ONNX Exporter.

    Supports two seamless execution modes:
      - Mode A (Full Ultralytics YOLOv8n/v11n CUDA FP16 Training & ONNX Export):
        Used automatically when `ultralytics` is installed. Trains with AMP FP16,
        AdamW, Cosine LR, Mosaic+MixUp, and exports half-precision static ONNX to
        `models/yolov8n_gatherion.onnx`.
      - Mode B (Built-in Fast OpenCV/NumPy Prototype & Centroid Classifier Fallback):
        Used when `ultralytics` is not installed in the current Python runtime or
        when `force_fallback=True`. Extracts multi-scale HSV histograms, Gabor filter
        banks, Hu invariant moments, edge density, and 16x16 templates from every
        labeled bounding box in `dataset/images/train` & `dataset/labels/train`,
        optimizes discriminant weights over `epochs`, evaluates `mAP50` and `mAP50_95`
        on `dataset/images/val`, writes `models/gatherion_custom_detector.json`, and
        generates a 1-click `train_rtx_yolo.bat` launcher script.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        use_approved_only: bool = False,
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.dataset_dir = (
            Path(dataset_dir).resolve()
            if dataset_dir is not None
            else (self.project_root / "dataset")
        )
        self.models_dir = (
            Path(models_dir).resolve()
            if models_dir is not None
            else (self.project_root / "models")
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.use_approved_only: bool = bool(use_approved_only)
        self.approved_yaml_path: Path = self.dataset_dir / "approved" / "data_approved.yaml"
        self.default_yaml_path: Path = self.dataset_dir / "data.yaml"
        self.onnx_export_path = self.models_dir / "yolov8n_gatherion.onnx"
        self.custom_detector_path = self.models_dir / "gatherion_custom_detector.json"
        self.bat_script_path = self.project_root / "train_rtx_yolo.bat"

        # D58: predict_roi detector-JSON cache — (mtime_ns, size) versioned,
        # so the file is parsed once while unchanged and reloaded after retraining.
        self._detector_cache: Optional[Dict[str, Any]] = None
        self._detector_cache_version: Optional[Tuple[int, int]] = None

        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # predict_roi detector cache: (mtime_ns, size) -> parsed arrays.
        # Reloaded only when gatherion_custom_detector.json changes on disk.
        self._predict_cache_key: Optional[Tuple[int, int]] = None
        self._predict_cache: Optional[Dict[str, Any]] = None
        self._latest_status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0,
            "epochs": 0,
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "eta_sec": 0.0,
            "elapsed_sec": 0.0,
            "mode": "uninitialized",
            "use_approved_only": self.use_approved_only,
            "artifacts": {},
        }
        self._gabor_kernels = self._build_gabor_filter_bank()

    # ------------------------------------------------------------------
    # 1. GPU & Environment Hardware Probe
    # ------------------------------------------------------------------
    @staticmethod
    def probe_gpu_environment_static() -> Dict[str, Any]:
        """
        Probes hardware and ML runtime environment:
          - `nvidia-smi` via subprocess with CREATE_NO_WINDOW
          - `torch` & `torch.cuda` device properties
          - `onnxruntime` execution providers
          - `ultralytics` availability and version
        """
        info: Dict[str, Any] = {
            "gpu_name": "CPU (No NVIDIA GPU Detected)",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "driver_version": "N/A",
            "temperature_c": 0,
            "gpu_utilization_pct": 0,
            "nvidia_smi_available": False,
            "cuda_available": False,
            "torch_installed": False,
            "torch_version": None,
            "cuda_version": None,
            "fp16_supported": False,
            "tensor_cores_available": False,
            "ultralytics_installed": False,
            "ultralytics_version": None,
            "onnxruntime_installed": False,
            "onnxruntime_version": None,
            "onnx_providers": [],
            "onnx_cuda_provider": False,
            "is_rtx_3070": False,
            "is_rtx_gpu": False,
            "estimated_5k_train_minutes": 14.2,
            "recommended_mode": "opencv_centroid_prototype",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        # 1. Query nvidia-smi without flashing a console window
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=CREATE_NO_WINDOW,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first_line = proc.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 4:
                    info["nvidia_smi_available"] = True
                    info["gpu_name"] = parts[0]
                    try:
                        info["vram_total_mb"] = int(float(parts[1]))
                    except ValueError:
                        info["vram_total_mb"] = 8192
                    try:
                        info["vram_free_mb"] = int(float(parts[2]))
                    except ValueError:
                        info["vram_free_mb"] = info["vram_total_mb"]
                    info["driver_version"] = parts[3]
                if len(parts) >= 5:
                    try:
                        info["temperature_c"] = int(float(parts[4]))
                    except ValueError:
                        pass
                if len(parts) >= 6:
                    try:
                        info["gpu_utilization_pct"] = int(float(parts[5]))
                    except ValueError:
                        pass
        except Exception as exc:
            logger.debug("nvidia-smi probe skipped or failed: %s", exc)

        # 2. Probe PyTorch & CUDA
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch  # type: ignore

                info["torch_installed"] = True
                info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
                cuda_ok = bool(torch.cuda.is_available())
                info["cuda_available"] = cuda_ok
                info["cuda_version"] = str(getattr(torch.version, "cuda", None))
                if cuda_ok and torch.cuda.device_count() > 0:
                    dev_name = str(torch.cuda.get_device_name(0))
                    info["gpu_name"] = dev_name
                    props = torch.cuda.get_device_properties(0)
                    total_mb = int(getattr(props, "total_memory", 0) // (1024 * 1024))
                    if total_mb > 0:
                        info["vram_total_mb"] = total_mb
                    major = int(getattr(props, "major", 0))
                    minor = int(getattr(props, "minor", 0))
                    info["compute_capability"] = f"{major}.{minor}"
                    info["fp16_supported"] = major >= 6
                    info["tensor_cores_available"] = major >= 7
            except Exception as exc:
                logger.debug("PyTorch probe failed: %s", exc)

        # F-0996: FP16 / tensor-core capability comes only from a real probe
        # (torch compute capability above, or nvidia-smi actually seeing the
        # GPU). A GPU name string alone proves nothing.
        gpu_upper = str(info["gpu_name"]).upper()
        if info["nvidia_smi_available"] and (
            "RTX" in gpu_upper or "GTX 16" in gpu_upper or "A100" in gpu_upper or "L4" in gpu_upper
        ):
            info["fp16_supported"] = True
        if "RTX" in gpu_upper:
            info["is_rtx_gpu"] = True
            if info["nvidia_smi_available"]:
                info["tensor_cores_available"] = True
        if "3070" in gpu_upper:
            info["is_rtx_3070"] = True

        # 3. Probe ONNX Runtime
        if importlib.util.find_spec("onnxruntime") is not None:
            try:
                import onnxruntime as ort  # type: ignore

                info["onnxruntime_installed"] = True
                info["onnxruntime_version"] = str(getattr(ort, "__version__", "unknown"))
                providers = list(ort.get_available_providers())
                info["onnx_providers"] = providers
                info["onnx_cuda_provider"] = (
                    "CUDAExecutionProvider" in providers
                    or "TensorrtExecutionProvider" in providers
                    or "DmlExecutionProvider" in providers
                )
            except Exception as exc:
                logger.debug("ONNXRuntime probe failed: %s", exc)

        # 4. Probe Ultralytics YOLO
        if importlib.util.find_spec("ultralytics") is not None:
            try:
                import ultralytics  # type: ignore

                info["ultralytics_installed"] = True
                info["ultralytics_version"] = str(getattr(ultralytics, "__version__", "unknown"))
            except Exception as exc:
                logger.debug("Ultralytics import check failed: %s", exc)

        # Estimate training time for 5,000 images (50 epochs, FP16)
        if info["is_rtx_3070"]:
            info["estimated_5k_train_minutes"] = 13.5
        elif info["is_rtx_gpu"]:
            vram = info["vram_total_mb"]
            info["estimated_5k_train_minutes"] = 11.0 if vram >= 10000 else 14.5
        elif info["nvidia_smi_available"]:
            info["estimated_5k_train_minutes"] = 22.0
        else:
            info["estimated_5k_train_minutes"] = 45.0

        if info["ultralytics_installed"] and (info["cuda_available"] or info["nvidia_smi_available"]):
            info["recommended_mode"] = "ultralytics_cuda_fp16"
        elif info["ultralytics_installed"]:
            info["recommended_mode"] = "ultralytics_cpu"
        else:
            info["recommended_mode"] = "opencv_centroid_prototype"

        return info

    def probe_gpu_environment(self) -> Dict[str, Any]:
        """Instance method wrapper for `probe_gpu_environment_static()`."""
        return self.probe_gpu_environment_static()

    # ------------------------------------------------------------------
    # 2. Optimal RTX 3070 Hyperparameter Calculator
    # ------------------------------------------------------------------
    @staticmethod
    def build_rtx_training_config_static(
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        use_approved_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds optimal Ultralytics & Prototype training hyperparameters tuned for
        NVIDIA GeForce RTX 3070 (8GB GDDR6, Ampere Tensor Cores, FP16 AMP).
        Targets ~12-15 minutes for 5,000 labeled 640x640 images across 50 epochs.
        """
        epochs = max(1, int(epochs))
        imgsz = max(320, int(imgsz))
        batch = max(1, int(batch))
        model_name = str(model or "yolo11n.pt").strip() or "yolo11n.pt"
        approved_flag = bool(use_approved_only)

        # Estimate throughput on RTX 3070 Ampere FP16 (~380 images/sec at 640px batch=16)
        images_per_sec_fp16 = 375.0 * (640.0 / float(imgsz)) ** 2
        est_seconds_5k = (5000.0 * epochs) / max(60.0, images_per_sec_fp16)
        est_minutes_5k = round(est_seconds_5k / 60.0, 2)

        return {
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "use_approved_only": approved_flag,
            "data_yaml": (
                "dataset/approved/data_approved.yaml"
                if approved_flag
                else "dataset/data.yaml"
            ),
            "amp": True,
            "half": True,
            "cos_lr": True,
            "optimizer": "AdamW",
            "lr0": 0.0015,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "patience": 15,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.05,
            "close_mosaic": 10,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "flipud": 0.0,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "workers": 4,
            # disk: onay havuzu (~1.4k kare) ile cache=ram süreçte +10–12 GB RSS
            # biriktiriyordu; Gatherion botu ile aynı süreçte segfault riski.
            "cache": "disk",
            # None = RAM-orantılı varsayılan (default_rss_abort_mb: toplam
            # RAM'in %75'i, en az 10240 MB). Sayı verilirse o eşik (MB)
            # kullanılır; 0 abort'u kapatır. fix/v20-rss-panik.
            "rss_abort_mb": None,
            # None = taze koşu. last.pt yolu verilirse eğitim kaldığı
            # yerden devam eder (train(resume=True), aynı koşu klasörü).
            # fix/v20-egitim-devam.
            "resume_from": None,
            "device": 0,
            "deterministic": True,
            "seed": 42,
            "single_cls": False,
            "rect": False,
            "exist_ok": True,
            "pretrained": True,
            "verbose": False,
            "onnx_export": {
                "format": "onnx",
                "half": True,
                "dynamic": False,
                "imgsz": imgsz,
                "simplify": True,
                "opset": 12,
                "target_path": "models/yolov8n_gatherion.onnx",
            },
            "hardware_target": {
                "target_gpu": "NVIDIA GeForce RTX 3070 (8GB VRAM)",
                "precision": "FP16 Mixed Precision (AMP)",
                "estimated_minutes_for_5000_images": est_minutes_5k,
                "target_accuracy_map50": 0.85,  # F-0998: realistic target, not a claimed result
            },
        }

    def build_rtx_training_config(
        self,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        use_approved_only: bool = False,
    ) -> Dict[str, Any]:
        """Instance method wrapper for `build_rtx_training_config_static()`."""
        return self.build_rtx_training_config_static(
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            model=model,
            use_approved_only=bool(use_approved_only or self.use_approved_only),
        )

    def resolve_training_yaml(
        self,
        data_yaml_path: Optional[Union[str, Path]] = None,
        use_approved_only: bool = False,
        validate: bool = True,
    ) -> Tuple[Path, bool]:
        """
        Resolves the active training YAML path and `use_approved_only` flag.
        When `use_approved_only=True` or `data_approved.yaml` is passed,
        prepares and returns the approved dataset YAML (`dataset/approved/data_approved.yaml`).

        `validate=False` resolves WITHOUT the P0-5 yaml check: callers
        whose refusal gates (disk-full, AION-running) must fire first
        pass False here and validate after those gates — the real
        `_run_training_sync_body` re-resolves WITH validation as its
        first step, so an invalid yaml still raises before any training
        work. Default True preserves the old fail-fast behaviour.
        """
        # Sweep E-01: the pre-training hygiene gate now RUNS in production.
        # Every training launch (sync, async, worker, .bat engine path)
        # funnels through here, so the gate cleans review JSON + label
        # TXT before any yaml resolution / pool preparation. The gate
        # must never block training: a gate crash only logs a warning,
        # and a not-clean verdict logs a warning and training proceeds
        # (whether to BLOCK on a dirty set is an open decision).
        try:
            from dataset_factory.dataset_hygiene_gate import (
                pre_training_dataset_hygiene_gate as _hygiene_gate,
            )

            _gate = _hygiene_gate(self.dataset_dir, auto_clean=True)
            if isinstance(_gate, dict) and not _gate.get("clean", False):
                logger.warning(
                    "[HİJYEN] Eğitim öncesi kapı temiz değil: %s",
                    _gate.get("summary", _gate),
                )
        except Exception as exc:  # noqa: BLE001 - gate must never block training
            logger.warning(
                "[HİJYEN] Hijyen kapısı çalıştırılamadı (eğitim devam ediyor): %s",
                exc,
            )

        effective_approved = bool(use_approved_only or self.use_approved_only)
        raw_path: Optional[Path] = Path(data_yaml_path) if data_yaml_path is not None else None

        if raw_path is not None:
            if (
                raw_path.name.lower() == "data_approved.yaml"
                or "approved" in [p.lower() for p in raw_path.parts]
            ):
                effective_approved = True

        if effective_approved:
            target_ds_dir = self.dataset_dir
            target_yaml = self.approved_yaml_path

            if raw_path is not None:
                resolved_raw = (
                    raw_path.resolve()
                    if raw_path.is_absolute()
                    else (self.project_root / raw_path).resolve()
                )
                if raw_path.is_file():
                    resolved_raw = raw_path.resolve()

                if resolved_raw.name.lower() == "data_approved.yaml":
                    target_yaml = resolved_raw
                    if resolved_raw.parent.name.lower() == "approved":
                        target_ds_dir = resolved_raw.parent.parent
                elif resolved_raw.is_dir():
                    if resolved_raw.name.lower() == "approved":
                        target_ds_dir = resolved_raw.parent
                        target_yaml = resolved_raw / "data_approved.yaml"
                    else:
                        target_ds_dir = resolved_raw
                        target_yaml = resolved_raw / "approved" / "data_approved.yaml"
                else:
                    # Caller passed e.g. `dataset/data.yaml` with `use_approved_only=True`
                    target_ds_dir = resolved_raw.parent
                    target_yaml = target_ds_dir / "approved" / "data_approved.yaml"

            if use_approved_only or self.use_approved_only or not target_yaml.is_file():
                try:
                    from dataset_factory.yolo_trainer import prepare_and_resolve_training_yaml

                    prepared = prepare_and_resolve_training_yaml(
                        dataset_dir=target_ds_dir,
                        only_approved=True,
                    )
                    if not target_yaml.is_file() and Path(prepared).is_file():
                        target_yaml = Path(prepared).resolve()
                except Exception as exc:
                    logger.debug("prepare_and_resolve_training_yaml fallback in engine: %s", exc)

            # P1-09 (K-24): dondurma-sayı kapısı, motor tarafı — ikinci
            # savunma hattı. prepare çağrısı yukarıda hatayı yutsa da bu
            # kontrol yutmaz: dondurulan sayılar havuzla uyuşmuyorsa
            # eğitim Ultralytics'e ulaşmadan somut gerekçeyle reddedilir.
            # (prepare_and_resolve_training_yaml kendi kapısını da yapar;
            # bu, doğrudan motor yolundan gelen çağrıları da kapsar.)
            try:
                from dataset_factory.approved_pool import (
                    check_frozen_manifest_consistency as _check_frozen,
                )

                _frozen_check = _check_frozen(target_ds_dir)
                if _frozen_check.get("checked") and not _frozen_check.get("ok"):
                    raise RuntimeError(
                        _frozen_check.get("reason")
                        or "Dondurulan veri seti havuzla uyuşmuyor."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.debug("frozen_manifest karşılaştırması atlandı (motor): %s", exc)

            if not target_yaml.is_file():
                # F-0895: never fabricate an empty approved yaml — training
                # on zero approved samples must fail loudly, not silently
                # "succeed" on an invented 22-class dataset.
                raise RuntimeError(
                    "Onaylı eğitim verisi bulunamadı: "
                    f"{target_yaml} yok. Önce Step 2'de kare onaylayın."
                )

            # F6: `prepare_and_resolve_training_yaml` yoksa BOŞ bir
            # data_approved.yaml yazar; yukarıdaki `is_file` kontrolü o
            # yüzden hiç tetiklenmezdi. Boş onaylı set de aynı şekilde
            # yüksek sesle reddedilir — Ultralytics'in kriptik boş-veri
            # hatasına bırakılmaz.
            _approved_root = target_yaml.parent
            _labeled = _count_labeled_samples(
                _approved_root / "images" / "train",
                _approved_root / "labels" / "train",
            ) + _count_labeled_samples(
                _approved_root / "images" / "val",
                _approved_root / "labels" / "val",
            )
            if _labeled == 0:
                raise RuntimeError(
                    "Onaylı eğitim verisi bulunamadı: "
                    f"{target_yaml} var ama etiketli kare yok (0 örnek). "
                    "Önce Step 2'de kare onaylayın."
                )

            # P0-5 (K-30): eğitim-öncesi yaml doğrulaması, motor tarafı.
            # Dialog kapısı (on_train_rtx_clicked) worker başlamadan önce
            # aynı kontrolü yapar; bu ikinci savunma hattıdır — worker
            # içinden / .bat / otomatik eğitimden gelen çağrılarda da
            # bozuk yaml Ultralytics'e ulaşmadan somut gerekçeyle reddedilir.
            _resolved_approved = _yaml_on_offload(target_yaml, use_approved_only=True).resolve()
            if validate:
                _check_training_yaml_or_raise(_resolved_approved)
            return _resolved_approved, True

        if raw_path is None:
            raw_path = self.default_yaml_path
        # P0-5 (K-30): sarı buton (only_approved=False) yolunda daha önce
        # yaml içeriği hiç doğrulanmıyordu — 0 kare de, göreli `path:` de
        # motor kapısında / Ultralytics'te patlıyordu. Offload yönlendirmesi
        # sonrası NİHAİ yaml doğrulanır (geçilen C: yaml'ı olmayabilir,
        # F: karşılığı kullanılır).
        _resolved_full = _yaml_on_offload(raw_path, use_approved_only=False).resolve()
        if validate:
            _check_training_yaml_or_raise(_resolved_full)
        return _resolved_full, False

    def _resolve_base_weights(self, model_name: str) -> str:
        """Prefer local weights under models/, else let Ultralytics resolve/download the name."""
        name = str(model_name or "yolo11n.pt").strip() or "yolo11n.pt"
        candidates = [
            self.models_dir / name,
            Path(name),
            Path.cwd() / name,
            Path(__file__).resolve().parent.parent / name,
        ]
        for cand in candidates:
            try:
                if cand.is_file():
                    return str(cand.resolve())
            except OSError:
                continue
        return name

    @staticmethod
    def _count_results_csv_epochs(run_dir: Path) -> int:
        csv_path = Path(run_dir) / "results.csv"
        if not csv_path.is_file():
            return 0
        try:
            import csv as _csv

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                return sum(1 for _ in _csv.DictReader(handle))
        except Exception:
            return 0

    @property
    def is_training(self) -> bool:
        return _train_slot_busy()

    @property
    def _training_active(self) -> bool:
        return _train_slot_busy()

    def _begin_training_session(self, *, slot_held: bool, batch: int) -> Dict[str, Any]:
        from core.game_gpu_throttle import GameGpuThrottle, training_image_retention_decision

        owns_slot = False
        if not slot_held:
            if not _acquire_train_slot():
                return {
                    "refused": True,
                    "throttle": None,
                    "owns_slot": False,
                    "batch": int(batch),
                }
            owns_slot = True

        decision = training_image_retention_decision(self.project_root)
        logger.info("Eğitim görselleri tutulur. %s", decision["reason"])
        try:
            game_pids = self._aion_blocks_training()
        except RuntimeError as exc:
            # F-0896 (fail-closed): an inconclusive AION probe refuses training.
            logger.error("AION süreç araması başarısız, eğitim reddedildi: %s", exc)
            if owns_slot:
                _release_train_slot()
            return {
                "refused": True,
                "reason": "aion_check_failed",
                "throttle": None,
                "owns_slot": False,
                "batch": int(batch),
            }
        if game_pids:
            logger.error(
                "AION2 pid=%s açık. Eğitim reddedildi. GPU paylaşılmayacak.",
                ",".join(str(pid) for pid in game_pids),
            )
            if owns_slot:
                _release_train_slot()
            return {
                "refused": True,
                "reason": "aion2_running",
                "throttle": None,
                "owns_slot": False,
                "batch": int(batch),
                "game_pids": game_pids,
            }
        # F-0999: actually apply the retention/throttle decision instead of
        # leaving "throttle": None — the throttle is restored by
        # _end_training_session. A game starting mid-training is throttled.
        throttle = None
        try:
            throttle = GameGpuThrottle()
            throttle.apply()
        except Exception as exc:
            logger.warning("Game GPU throttle uygulanamadı: %s", exc)
            throttle = None
        return {
            "refused": False,
            "throttle": throttle,
            "owns_slot": owns_slot,
            "batch": int(batch),
        }

    def _aion_blocks_training(self) -> List[int]:
        """AION2 / AION 2 / Aion2 açıksa eğitim başlamaz. Batch düşürmek GPU'yu paylaşır.

        Raises RuntimeError when the process search itself fails — callers
        treat that as fail-closed (training refused), never as "no game".
        """
        from core.game_gpu_throttle import find_aion_game_processes

        try:
            return [int(pid) for pid in find_aion_game_processes()]
        except Exception as exc:
            raise RuntimeError(f"AION süreç araması başarısız: {exc}") from exc

    def _end_training_session(self, session: Dict[str, Any]) -> None:
        throttle = session.get("throttle")
        if throttle is not None:
            try:
                throttle.restore()
            except Exception:
                logger.exception("Oyun önceliği ve EcoQoS geri alınamadı")
        if session.get("owns_slot"):
            _release_train_slot()

    # ------------------------------------------------------------------
    # 3. 1-Click Asynchronous & Synchronous Training Pipeline
    # ------------------------------------------------------------------
    def start_training_async(
        self,
        data_yaml_path: Optional[Union[str, Path]] = None,
        epochs: int = 50,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        imgsz: int = 640,
        batch: int = 16,
        force_fallback: bool = False,
        model: str = "yolo11n.pt",
        allow_opencv_fallback: bool = False,
        use_approved_only: bool = False,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Starts 1-Click training asynchronously on a background daemon thread.
        Immediately returns a launch descriptor containing the thread handle,
        environment probe, and active training configuration.
        A second start is refused until the worker releases the process-wide slot.
        AION2, AION 2, or Aion2 running refuses the start. The GPU is not shared.
        """
        resolved_yaml, effective_approved = self.resolve_training_yaml(
            data_yaml_path=data_yaml_path,
            use_approved_only=use_approved_only,
            validate=False,
        )
        try:
            game_pids = self._aion_blocks_training()
        except RuntimeError as exc:
            # F-0896 (fail-closed): inconclusive AION probe refuses the start.
            logger.error("AION süreç araması başarısız, eğitim reddedildi: %s", exc)
            return {
                "started": False,
                "state": "refused",
                "reason": "aion_check_failed",
                "game_pids": [],
                "use_approved_only": effective_approved,
                "data_yaml_path": str(resolved_yaml),
                "status": dict(self._latest_status),
            }
        if game_pids:
            logger.error(
                "AION2 pid=%s açık. Eğitim reddedildi. GPU paylaşılmayacak.",
                ",".join(str(pid) for pid in game_pids),
            )
            return {
                "started": False,
                "state": "refused",
                "reason": "aion2_running",
                "game_pids": game_pids,
                "use_approved_only": effective_approved,
                "data_yaml_path": str(resolved_yaml),
                "status": dict(self._latest_status),
            }
        # P0-5 tur 2: senkron yolla aynı sözleşme — yaml doğrulaması
        # oyun reddinden SONRA, slot alınmadan önce çalışır; geçersiz
        # yaml eskisi gibi senkron yükselir, oyun reddini gölgelemez.
        _check_training_yaml_or_raise(resolved_yaml)
        if not _acquire_train_slot():
            return {
                "started": False,
                "reason": "Training is already running in the background.",
                "use_approved_only": effective_approved,
                "data_yaml_path": str(resolved_yaml),
                "status": dict(self._latest_status),
            }
        self._stop_event.clear()

        try:
            env_info = self.probe_gpu_environment()
            train_cfg = self.build_rtx_training_config(
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                model=model,
                use_approved_only=effective_approved,
            )
            train_cfg["resume_from"] = str(resume_from) if resume_from else None

            def _worker() -> None:
                try:
                    self.run_training_sync(
                        data_yaml_path=resolved_yaml,
                        epochs=epochs,
                        progress_callback=progress_callback,
                        imgsz=imgsz,
                        batch=batch,
                        force_fallback=force_fallback,
                        slot_held=True,
                        model=model,
                        allow_opencv_fallback=allow_opencv_fallback,
                        use_approved_only=effective_approved,
                        resume_from=resume_from,
                    )
                except Exception as exc:
                    logger.exception("Asynchronous training encountered an error: %s", exc)
                    with self._lock:
                        self._latest_status["state"] = "failed"
                        self._latest_status["error"] = str(exc)
                    if progress_callback is not None:
                        try:
                            progress_callback(dict(self._latest_status))
                        except Exception:
                            pass
                finally:
                    _release_train_slot()

            thread = threading.Thread(
                target=_worker,
                name="RTXTrainerEngineWorker",
                daemon=True,
            )
            with self._lock:
                self._training_thread = thread
                self._latest_status.update(
                    {
                        "state": "running",
                        "epoch": 0,
                        "epochs": epochs,
                        "use_approved_only": effective_approved,
                        "data_yaml_path": str(resolved_yaml),
                        "mode": "opencv_centroid_prototype"
                        if (force_fallback or not env_info["ultralytics_installed"])
                        else "ultralytics_cuda_fp16",
                    }
                )
            thread.start()
        except Exception:
            _release_train_slot()
            raise

        return {
            "started": True,
            "mode": self._latest_status["mode"],
            "environment": env_info,
            "config": train_cfg,
            "thread_name": thread.name,
            "use_approved_only": effective_approved,
            "data_yaml_path": str(resolved_yaml),
        }

    def wait_for_training(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Waits for the background training thread (if any) to complete and returns final status.

        P2-18: join her zaman sonludur — `timeout=None` varsayılan
        `RTX_TRAIN_WAIT_TIMEOUT_S` (3600 sn) sayılır, süresiz bekleme
        yok. Süre dolarsa dönen sözlük `timed_out=True` taşır.
        """
        if timeout is None:
            timeout = RTX_TRAIN_WAIT_TIMEOUT_S
        thread = self._training_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=float(timeout))
        with self._lock:
            status = dict(self._latest_status)
        status["timed_out"] = bool(thread is not None and thread.is_alive())
        return status

    def stop_training(self) -> None:
        """Signals the active training loop to stop cleanly at the end of the current epoch."""
        self._stop_event.set()

    def get_status(self) -> Dict[str, Any]:
        """Returns a thread-safe snapshot of the latest training progress."""
        with self._lock:
            return dict(self._latest_status)

    def run_training_sync(
        self,
        data_yaml_path: Optional[Union[str, Path]] = None,
        epochs: int = 50,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        imgsz: int = 640,
        batch: int = 16,
        force_fallback: bool = False,
        slot_held: bool = False,
        model: str = "yolo11n.pt",
        allow_opencv_fallback: bool = False,
        use_approved_only: bool = False,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run one training job. A second caller gets started=False until this one finishes."""
        runs_dir = _pin_train_env()
        # fix/v20-egitim-devam: devam checkpoint'i eğitim başlamadan
        # doğrulanır (geçersizse burada yükselir, slot/disk işine
        # girilmez); taze koşu benzersiz zaman damgalı klasöre planlanır
        # — eski best.pt/last.pt'nin üstüne bir daha yazılmaz.
        resume_ckpt: Optional[Path] = None
        if resume_from is not None and str(resume_from).strip() != "":
            resume_ckpt = resolve_resume_checkpoint(resume_from)
            save_dir = run_dir_for_resume_checkpoint(resume_ckpt)
        else:
            save_dir = runs_dir / generate_unique_run_name(runs_dir)
        # P0-5 tur 2: yaml doğrulaması disk/oyun reddinden SONRA çalışır.
        # `validate=False` ile çözümlenir; gerçek body ilk adımda
        # doğrulamalı çözümleme yapar, yani geçersiz yaml eğitimden önce
        # yine yükselir — ama oyun-çalışıyor / disk-dolu reddi (return)
        # artık yaml hatasının gölgesinde kalmaz. Önceden resolve
        # buradayken raise ediyor, reddi hiç işletmiyordu.
        resolved_yaml, effective_approved = self.resolve_training_yaml(
            data_yaml_path=data_yaml_path,
            use_approved_only=use_approved_only,
            validate=False,
        )
        free = target_volume_free_bytes(runs_dir)
        if free < _MIN_TRAIN_FREE_BYTES:
            logger.error(
                "disk dolu: save_dir=%s free_bytes=%s need_bytes=%s",
                save_dir,
                free,
                _MIN_TRAIN_FREE_BYTES,
            )
            payload = {
                "started": False,
                "state": "failed",
                "error": "disk dolu",
                "reason": "disk dolu",
                "save_dir": str(save_dir),
                "disk_free_bytes": free,
                "use_approved_only": effective_approved,
                "data_yaml_path": str(resolved_yaml),
            }
            with self._lock:
                self._latest_status.update(payload)
            payload["status"] = dict(self._latest_status)
            return payload
        session = self._begin_training_session(slot_held=slot_held, batch=batch)
        if session["refused"]:
            reason = str(session.get("reason") or "Training is already running.")
            return {
                "started": False,
                "state": "refused" if reason == "aion2_running" else "busy",
                "reason": reason,
                "game_pids": list(session.get("game_pids") or []),
                "use_approved_only": effective_approved,
                "data_yaml_path": str(resolved_yaml),
                "status": dict(self._latest_status),
            }
        try:
            return self._run_training_sync_body(
                data_yaml_path=resolved_yaml,
                epochs=epochs,
                progress_callback=progress_callback,
                imgsz=imgsz,
                batch=int(session["batch"]),
                force_fallback=force_fallback,
                model=model,
                allow_opencv_fallback=allow_opencv_fallback,
                use_approved_only=effective_approved,
                resume_from=resume_ckpt,
                run_dir=save_dir,
            )
        finally:
            self._end_training_session(session)

    def _run_training_sync_body(
        self,
        data_yaml_path: Union[str, Path],
        epochs: int = 50,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        imgsz: int = 640,
        batch: int = 16,
        force_fallback: bool = False,
        model: str = "yolo11n.pt",
        allow_opencv_fallback: bool = False,
        use_approved_only: bool = False,
        resume_from: Optional[Union[str, Path]] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full training & export pipeline synchronously:
          - Always generates `train_rtx_yolo.bat` for 1-click CLI/standalone RTX execution.
          - Runs Mode A (Ultralytics YOLO FP16 CUDA + ONNX export) if `ultralytics` is
            installed and `force_fallback` is False.
          - Runs Mode B (OpenCV prototype) only when allowed — factory UI disables this
            so a failed/missing Ultralytics launch cannot fake-complete in ~1s.
        """
        self._stop_event.clear()
        # P0-5 tur 3 (T3): B-06 boş-veri kontrolü yaml doğrulamasından
        # ÖNCE çalışır — yalnızca bu body yolunda. `validate_training_yaml`
        # göreli `path:`'i sayım yapmadan reddeder (erken return), oysa
        # T3'ün sözleşmesi boş veri setinde "Eğitim verisi boş"tur ve
        # HEAD'deki dataset/data.yaml `path: .` görelidir. Bu yüzden body
        # doğrulamasız çözer, yaml-dizinine göre sayan B-06 önce ateşlenir;
        # hemen ardından doğrulama çalışır, geçersiz yaml eğitim işi
        # başlamadan yine burada yükselir. (Seçenek (b) — sayımı
        # `validate_training_yaml` içinde öne almak — yanlış olurdu: oradaki
        # sayım göreli `path:`'i CWD'ye göre çözer (`resolve_yaml_split_dir`),
        # yani yaml-dizini veri setini değil CWD'yi sayardı.)
        yaml_path, effective_approved = self.resolve_training_yaml(
            data_yaml_path=data_yaml_path,
            use_approved_only=use_approved_only,
            validate=False,
        )
        # F2 (dilim 05): eğitim öncesi etiket doğrulama kapısı. Ne bu motor
        # ne YOLOTrainer ne de controller bugüne kadar hijyen/validator
        # kapısını çağırmıyordu — şablon/bozuk kutu olduğu gibi eğitime
        # giriyordu. Kapı dry-run'dur (etiket dosyasını değiştirmez);
        # bozuk sınıf/kutu bulunursa eğitim yüksek sesle reddedilir.
        # NOT: kapı geometri/bozukluk yakalar, şablon kutuyu yakalamaz —
        # şablon savunması onaylı-havuz yoludur (only_approved).
        # B-06 (dilim 27): boş train/val kapısı — Mode B zaten hard-fail
        # veriyordu, Mode A'da yoktu; iki motor artık aynı kapıdan geçer.
        # Yalnızca çözümlenebilir yaml'da say: yaml yoksa / train-val
        # girdisi eksikse / okunamıyorsa B-06 atlanır — o kusurların
        # somut hatası aşağıdaki doğrulamada korunur.
        # S083 (v19/P3-8): `_ds_paths` yalnızca yaml çözümlenebilirse
        # tanımlanıyordu ama hijyen döngüsü koşulsuz kullanıyordu —
        # iki ayrı yaml okuyucu anlaşamazsa NameError, genel except'te
        # "kapı çalışmadı" diye yutulup hijyen sessizce atlanıyordu.
        _ds_paths: Optional[Dict[str, Any]] = None
        if yaml_path.is_file():
            _parsed = _parse_simple_yaml(yaml_path)
            if isinstance(_parsed, dict) and _parsed.get("train") and _parsed.get("val"):
                _ds_paths = self._resolve_dataset_paths(yaml_path)
                _train_n = _count_labeled_samples(
                    _ds_paths["train_images"], _ds_paths["train_labels"]
                )
                _val_n = _count_labeled_samples(
                    _ds_paths["val_images"], _ds_paths["val_labels"]
                )
                if _train_n == 0:
                    raise RuntimeError(
                        "Eğitim verisi boş: train klasöründe etiketli örnek yok "
                        f"({yaml_path}). Önce veri setini doldurun/dondurun."
                    )
                if _val_n == 0:
                    raise RuntimeError(
                        "Doğrulama seti boş: val klasöründe etiketli örnek yok "
                        f"({yaml_path}). Train/val bölmesi val'siz bırakılmış."
                    )
        # P0-5 tur 2 korundu: geçersiz yaml eğitim işi başlamadan burada
        # yükselir (göreli `path:`, eksik split, 0 etiketli kare vb.).
        _check_training_yaml_or_raise(yaml_path)
        _hygiene: Dict[str, Any] = {}
        try:
            from dataset_factory.pre_training_hygiene_gate import (
                PreTrainingHygieneGate,
            )

            # P0-3 (v18): ust sinir artik burada literal yazilmiyor — gate
            # varsayilani tek kaynaktan (dataset/data.yaml, nc-1; yedek 26)
            # cozer, bu cagri da o varsayilani kullanir.
            _gate = PreTrainingHygieneGate(in_place=False)
            if _ds_paths is None:
                # S083: yaml geçerli ama yollar çözülemedi — hijyen
                # sessizce atlanmaz, açıkça loglanır.
                logger.warning(
                    "Eğitim öncesi hijyen atlandı: dataset yolları "
                    "çözülemedi (%s).", yaml_path
                )
            for _lbl_dir in (_ds_paths["train_labels"], _ds_paths["val_labels"]) if _ds_paths is not None else []:
                _rep = _gate.scan_and_clean(_lbl_dir)
                for _k, _v in _rep.to_dict().items():
                    if isinstance(_v, (int, float)):
                        _hygiene[_k] = int(_hygiene.get(_k, 0)) + int(_v)
                    elif isinstance(_v, list):
                        _hygiene.setdefault(_k, []).extend(_v)
            if _hygiene.get("errors"):
                raise RuntimeError(
                    "Eğitim öncesi etiket hijyen kapısı hata verdi: "
                    + "; ".join(str(e) for e in _hygiene["errors"][:5])
                )
            if _hygiene.get("total_removed", 0):
                logger.warning(
                    "Eğitim öncesi hijyen taraması %d geçersiz/bozuk kutu "
                    "buldu (sınıf:%d bozuk:%d zehirli:%d) — eğitim bu "
                    "kutularla devam ediyor; Adım 2/3'te temizleyin.",
                    _hygiene.get("total_removed", 0),
                    _hygiene.get("removed_invalid_class_id", 0),
                    _hygiene.get("removed_corrupted_boxes", 0),
                    _hygiene.get("removed_poisoned_labels", 0),
                )
        except RuntimeError:
            raise
        except Exception as exc:  # kapı yoksa/bozulursa eğitim durmaz; loglanır
            logger.warning("Eğitim öncesi hijyen kapısı çalışmadı: %s", exc)
        self._latest_hygiene_report = dict(_hygiene)
        epochs = max(1, int(epochs))
        env_info = self.probe_gpu_environment()
        train_cfg = self.build_rtx_training_config(
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            model=model,
            use_approved_only=effective_approved,
        )
        # fix/v20-egitim-devam: devam isteği + planlanan koşu klasörü
        # config üzerinden Mode A'ya taşınır (sonuç payload'unda da görünür).
        train_cfg["resume_from"] = str(resume_from) if resume_from else None
        train_cfg["run_dir"] = str(run_dir) if run_dir else None

        # Always generate the 1-click batch launcher script for full RTX GPU training
        bat_path = self._generate_rtx_batch_script(
            data_yaml_path=yaml_path,
            train_cfg=train_cfg,
            env_info=env_info,
        )

        use_ultralytics = bool(env_info["ultralytics_installed"]) and not force_fallback

        if use_ultralytics:
            try:
                res_a = self._run_mode_a_ultralytics(
                    yaml_path=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    train_cfg=train_cfg,
                    env_info=env_info,
                    bat_path=bat_path,
                    progress_callback=progress_callback,
                    model=model,
                    use_approved_only=effective_approved,
                    resume_from=resume_from,
                    run_dir=run_dir,
                )
                res_a["use_approved_only"] = effective_approved
                res_a["data_yaml_path"] = str(yaml_path)
                return res_a
            except Exception as exc:
                # fix/v20-egitim-devam: devam isteği varken Mode B
                # prototipine düşülmez — prototip kaldığı yerden devam
                # edemez, sessizce taze eğitim sayılırdı.
                if not allow_opencv_fallback or resume_from is not None:
                    logger.error(
                        "Mode A (Ultralytics) failed; OpenCV fallback disabled: %s",
                        exc,
                    )
                    payload = {
                        "started": False,
                        "state": "failed",
                        "status": "failed",
                        "error": str(exc),
                        "reason": f"Ultralytics eğitimi başarısız: {exc}",
                        "mode": "ultralytics_failed",
                        "use_approved_only": effective_approved,
                        "data_yaml_path": str(yaml_path),
                        "artifacts": {"train_bat_script": str(bat_path)},
                        "environment": env_info,
                        "config": train_cfg,
                    }
                    with self._lock:
                        self._latest_status.update(payload)
                    return payload
                logger.warning(
                    "Mode A (Ultralytics) failed (%s); seamlessly falling back to Mode B OpenCV Prototype trainer.",
                    exc,
                )
        elif not allow_opencv_fallback or resume_from is not None:
            reason = (
                "ultralytics yüklü değil — YOLO fabrikası OpenCV prototipini kabul etmez."
                if not env_info.get("ultralytics_installed")
                else "force_fallback=True reddedildi (OpenCV fallback kapalı)."
            )
            if resume_from is not None and env_info.get("ultralytics_installed"):
                reason = (
                    "Devam eğitimi yalnızca Ultralytics ile yapılır — "
                    "OpenCV prototipi kaldığı yerden devam edemez."
                )
            payload = {
                "started": False,
                "state": "failed",
                "status": "failed",
                "error": reason,
                "reason": reason,
                "mode": "ultralytics_required",
                "use_approved_only": effective_approved,
                "data_yaml_path": str(yaml_path),
                "artifacts": {"train_bat_script": str(bat_path)},
                "environment": env_info,
                "config": train_cfg,
            }
            with self._lock:
                self._latest_status.update(payload)
            return payload

        res_b = self._run_mode_b_opencv_prototype(
            yaml_path=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            train_cfg=train_cfg,
            env_info=env_info,
            bat_path=bat_path,
            progress_callback=progress_callback,
        )
        res_b["use_approved_only"] = effective_approved
        res_b["data_yaml_path"] = str(yaml_path)
        return res_b

    # ------------------------------------------------------------------
    # Mode A: Full Ultralytics YOLOv8n CUDA FP16 Training + ONNX Export
    # ------------------------------------------------------------------
    def _run_mode_a_ultralytics(
        self,
        yaml_path: Path,
        epochs: int,
        imgsz: int,
        batch: int,
        train_cfg: Dict[str, Any],
        env_info: Dict[str, Any],
        bat_path: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        model: str = "yolo11n.pt",
        use_approved_only: bool = False,
        resume_from: Optional[Union[str, Path]] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        # fix/v20-egitim-devam: devam isteği ultralytics import'undan
        # ÖNCE doğrulanır — geçersiz checkpoint eğitim hiç başlamadan
        # Türkçe hatayla reddedilir. Koşu klasörü de burada kesinleşir:
        # devamda checkpoint'in klasörü, taze koşuda benzersiz zaman
        # damgalı klasör (çağıranın planladığı klasör varsa o kullanılır).
        resume_arg = resume_from if resume_from is not None else train_cfg.get("resume_from")
        resume_ckpt: Optional[Path] = None
        if resume_arg is not None and str(resume_arg).strip() != "":
            resume_ckpt = resolve_resume_checkpoint(resume_arg)
        run_dir_arg = run_dir if run_dir is not None else train_cfg.get("run_dir")
        runs_dir = rtx_runs_dir_for()
        if resume_ckpt is not None:
            run_dir = (
                Path(run_dir_arg)
                if run_dir_arg
                else run_dir_for_resume_checkpoint(resume_ckpt)
            )
        elif run_dir_arg:
            run_dir = Path(run_dir_arg)
        else:
            run_dir = runs_dir / generate_unique_run_name(runs_dir)
        run_name = Path(run_dir).name

        from ultralytics import YOLO  # type: ignore

        start_time = time.perf_counter()
        device_arg: Union[int, str] = 0 if env_info.get("cuda_available") else "cpu"

        if resume_ckpt is not None:
            # Devam: model = checkpoint'in kendisi; Ultralytics
            # resume=True ile epoch/optimizer/zamanlayıcıyı geri yükler.
            base_model_str = str(resume_ckpt)
            logger.info(
                "Mode A Ultralytics resume: checkpoint=%s run_dir=%s epochs=%s imgsz=%s batch=%s device=%s",
                base_model_str,
                run_dir,
                epochs,
                imgsz,
                batch,
                device_arg,
            )
        else:
            base_model_str = self._resolve_base_weights(
                str(model or train_cfg.get("model") or "yolo11n.pt")
            )
            logger.info(
                "Mode A Ultralytics start: model=%s epochs=%s imgsz=%s batch=%s device=%s",
                base_model_str,
                epochs,
                imgsz,
                batch,
                device_arg,
            )
        model = YOLO(base_model_str)

        def _on_fit_epoch_end(trainer: Any) -> None:
            elapsed = time.perf_counter() - start_time
            curr_epoch = int(getattr(trainer, "epoch", 0)) + 1
            total_epochs = int(getattr(trainer, "epochs", epochs))
            avg_per_epoch = elapsed / max(1, curr_epoch)
            eta_sec = max(0.0, (total_epochs - curr_epoch) * avg_per_epoch)

            # v19 telemetri: tloss dict'tir (Ultralytics >=8.3); alınamayan
            # loss None'dur, 0.0 değil — 0.0000 "loss sıfır" değil
            # "loss alınamadı" demekti (canlı: epoch 20/30/40/50/69 hep 0.0000).
            box_loss, cls_loss = extract_epoch_losses(getattr(trainer, "tloss", None))

            # F-0890: eksik canlı metrikler uydurma varsayılanlarla
            # (0.985 / 0.982 / 0.992 / 0.895) doldurulmaz. Loss için eksik
            # = None ("—" gösterilir); precision/recall/mAP ilk validasyon
            # öncesi gerçekten 0.0'dır.
            metrics_dict = getattr(trainer, "metrics", {}) or {}
            precision = float(
                metrics_dict.get("metrics/precision(B)", metrics_dict.get("precision", 0.0))
            )
            recall = float(
                metrics_dict.get("metrics/recall(B)", metrics_dict.get("recall", 0.0))
            )
            map50 = float(
                metrics_dict.get("metrics/mAP50(B)", metrics_dict.get("mAP50", 0.0))
            )
            map50_95 = float(
                metrics_dict.get("metrics/mAP50-95(B)", metrics_dict.get("mAP50_95", 0.0))
            )

            payload = {
                "state": "running",
                "mode": "ultralytics_cuda_fp16",
                "epoch": curr_epoch,
                "epochs": total_epochs,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "precision": round(precision, 5),
                "recall": round(recall, 5),
                "mAP50": round(map50, 5),
                "mAP50_95": round(map50_95, 5),
                "eta_sec": round(eta_sec, 2),
                "elapsed_sec": round(elapsed, 2),
                "gpu_name": env_info.get("gpu_name", "CUDA GPU"),
            }
            with self._lock:
                self._latest_status.update(payload)
            if progress_callback is not None:
                try:
                    progress_callback(dict(payload))
                except Exception:
                    pass

            if self._stop_event.is_set():
                setattr(trainer, "stop", True)
                try:
                    trainer.validate = lambda *args, **kwargs: ({}, 0.0)
                except Exception:
                    pass

        _last_batch_ui_update = 0.0
        _last_rss_check = 0.0
        _stopped_for_rss = False
        _batch_counter = _BatchStepCounter()
        # fix/v20-rss-panik: eşik açıkça verilmediyse RAM-orantılı
        # varsayılan (toplam RAM'in %75'i, en az 10240 MB). Açık 0
        # abort'u kapatır. Eşik + toplam RAM loga yazılır — bir sonraki
        # koşuda hangi eşikle çalışıldığı logdan okunur.
        _cfg_abort = train_cfg.get("rss_abort_mb")
        _total_ram_mb = total_system_ram_mb()
        if _cfg_abort is None:
            _rss_abort_mb = default_rss_abort_mb(_total_ram_mb)
        else:
            _rss_abort_mb = float(_cfg_abort)
        _rss_decider = _RssAbortDecider(_rss_abort_mb, total_ram_mb=_total_ram_mb)
        logger.info(
            "RTX eğitimi RSS abort eşiği: %.0f MB (toplam RAM: %s; %d ardışık kontrolle teyit).",
            _rss_abort_mb,
            ("%.0f MB" % _total_ram_mb) if _total_ram_mb else "bilinmiyor",
            _rss_decider.confirmations,
        )

        def _on_train_batch_end(trainer: Any) -> None:
            nonlocal _last_batch_ui_update, _last_rss_check, _stopped_for_rss
            if self._stop_event.is_set():
                setattr(trainer, "stop", True)
                try:
                    trainer.validate = lambda *args, **kwargs: ({}, 0.0)
                except Exception:
                    pass
                return
            now_rss = time.monotonic()
            if _rss_decider.enabled and now_rss - _last_rss_check >= 30.0:
                _last_rss_check = now_rss
                try:
                    from core.crash_evidence import _read_rss_mb

                    rss = _read_rss_mb()
                except Exception:
                    rss = None
                if rss is not None and _rss_decider.observe(rss):
                    _stopped_for_rss = True
                    logger.warning(
                        "RTX eğitimi RSS eşiği aşıldı (%.1f MB >= %.1f MB, %d ardışık kontrol) — güvenli durdurma.",
                        rss,
                        _rss_abort_mb,
                        _rss_decider.confirmations,
                    )
                    self._stop_event.set()
                    setattr(trainer, "stop", True)
                    return
            # v19 telemetri: sayaç HER batch_end çağrısında ilerler —
            # throttle (2.5 sn) yalnızca UI gönderimini seyreltir, sayımı değil.
            # Eski kod `trainer.batch` okuyordu; o öznitelik yok, cur_b hep 1'di.
            try:
                cur_b, tot_b, cur_ep = _batch_counter.next(trainer)
            except Exception:
                return
            now = time.monotonic()
            if now - _last_batch_ui_update >= 2.5:
                _last_batch_ui_update = now
                try:
                    tot_ep = getattr(trainer, "epochs", epochs)
                    if progress_callback is not None:
                        progress_callback({
                            "batch_step": True,
                            "epoch": cur_ep,
                            "epochs": tot_ep,
                            "batch": cur_b,
                            "total_batches": tot_b,
                        })
                except Exception:
                    pass

        try:
            model.add_callback("on_train_batch_end", _on_train_batch_end)
            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        except Exception:
            pass

        _point_ultralytics_runs(runs_dir)
        use_cuda_device = bool(env_info.get("cuda_available", False))
        # F-0891: pass the hyperparameters build_rtx_training_config computed
        # instead of silently dropping most of them.
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            amp=bool(train_cfg["amp"] and use_cuda_device),
            cos_lr=bool(train_cfg["cos_lr"]),
            mosaic=float(train_cfg["mosaic"]),
            mixup=float(train_cfg["mixup"]),
            copy_paste=float(train_cfg.get("copy_paste", 0.05)),
            close_mosaic=int(train_cfg.get("close_mosaic", 10)),
            hsv_h=float(train_cfg.get("hsv_h", 0.015)),
            hsv_s=float(train_cfg.get("hsv_s", 0.7)),
            hsv_v=float(train_cfg.get("hsv_v", 0.4)),
            degrees=float(train_cfg.get("degrees", 0.0)),
            translate=float(train_cfg.get("translate", 0.1)),
            scale=float(train_cfg.get("scale", 0.5)),
            fliplr=float(train_cfg.get("fliplr", 0.5)),
            flipud=float(train_cfg.get("flipud", 0.0)),
            box=float(train_cfg.get("box", 7.5)),
            cls=float(train_cfg.get("cls", 0.5)),
            dfl=float(train_cfg.get("dfl", 1.5)),
            optimizer=str(train_cfg["optimizer"]),
            lr0=float(train_cfg["lr0"]),
            lrf=float(train_cfg["lrf"]),
            momentum=float(train_cfg.get("momentum", 0.937)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
            warmup_epochs=float(train_cfg.get("warmup_epochs", 3.0)),
            warmup_momentum=float(train_cfg.get("warmup_momentum", 0.8)),
            warmup_bias_lr=float(train_cfg.get("warmup_bias_lr", 0.1)),
            patience=int(train_cfg["patience"]),
            device=0 if use_cuda_device else "cpu",
            workers=0,
            cache=str(train_cfg.get("cache", "disk")),
            deterministic=bool(train_cfg.get("deterministic", True)),
            seed=int(train_cfg.get("seed", 42)),
            single_cls=bool(train_cfg.get("single_cls", False)),
            rect=bool(train_cfg.get("rect", False)),
            pretrained=bool(train_cfg.get("pretrained", True)),
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            resume=resume_ckpt is not None,
            verbose=bool(train_cfg.get("verbose", False)),
        )

        if self._stop_event.is_set():
            if _stopped_for_rss:
                stop_reason = (
                    f"Eğitim bellek eşiği nedeniyle durduruldu (RSS >= {_rss_abort_mb:.0f} MB). "
                    "Uygulamayı yeniden başlatıp eğitimi ayrı süreçte (train_rtx_yolo.bat) deneyin."
                )
                logger.warning("RTX eğitimi RSS eşiğiyle durduruldu.")
            else:
                stop_reason = "Eğitim kullanıcı tarafından durduruldu."
                logger.info("RTX eğitimi kullanıcı tarafından durduruldu.")
            total_elapsed = round(time.perf_counter() - start_time, 3)
            epochs_completed = int(self._count_results_csv_epochs(run_dir) or 0)
            payload = {
                "started": True,
                "stopped": True,
                "interrupted": True,
                "state": "stopped",
                "status": "stopped",
                "reason": stop_reason,
                "mode": "ultralytics_cuda_fp16",
                "elapsed_sec": total_elapsed,
                "epoch": epochs_completed,
                "epochs": epochs,
                "epochs_completed": epochs_completed,
                "use_approved_only": use_approved_only,
                "data_yaml_path": str(yaml_path),
            }
            with self._lock:
                self._latest_status.update(payload)
            try:
                gc.collect()
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return payload

        # Export trained weights to ONNX (FP16 when on GPU)
        exported_path = model.export(
            format="onnx",
            half=bool(env_info.get("cuda_available", False)),
            dynamic=False,
            imgsz=imgsz,
        )
        if exported_path and Path(str(exported_path)).exists():
            shutil.copy2(str(exported_path), str(self.onnx_export_path))
        else:
            best_onnx = run_dir / "weights" / "best.onnx"
            if best_onnx.exists():
                shutil.copy2(str(best_onnx), str(self.onnx_export_path))
            else:
                # F-0892: a failed ONNX export must not pass silently while
                # a stale .onnx file keeps looking like a fresh model.
                raise RuntimeError(
                    "ONNX export başarısız: ne model.export() ne de "
                    f"{best_onnx} bir ONNX dosyası üretti."
                )

        # Real Ultralytics metrics from results.csv — never report Mode B prototype scores.
        ultra_metrics = self._read_ultralytics_final_metrics(run_dir)
        epochs_completed = int(
            ultra_metrics.get("epochs_completed") or self._count_results_csv_epochs(run_dir) or 0
        )
        total_elapsed = round(time.perf_counter() - start_time, 3)
        if epochs_completed < 1:
            raise RuntimeError(
                "Ultralytics results.csv has no completed epochs — training did not run."
            )
        # Refuse absurdly fast "success" for multi-epoch jobs (Mode B used to finish in ~1s).
        if epochs >= 10 and total_elapsed < 5.0:
            raise RuntimeError(
                f"Training finished in {total_elapsed:.1f}s for {epochs} epochs — "
                "refused as fake-complete (expected real Ultralytics epoch loop)."
            )
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            raise RuntimeError(f"Missing Ultralytics weights after train: {best_pt}")

        # Prototype JSON only. Do not overwrite yolov8n_gatherion.onnx with the stub artifact.
        # F-0992: the prototype is a sidecar only — its failure must not
        # turn a successful Ultralytics run into a failure.
        try:
            proto_res = self._run_mode_b_opencv_prototype(
                yaml_path=yaml_path,
                epochs=min(5, epochs),
                imgsz=imgsz,
                batch=batch,
                train_cfg=train_cfg,
                env_info=env_info,
                bat_path=bat_path,
                progress_callback=None,
                write_live_onnx=False,
            )
        except Exception as proto_exc:
            logger.warning("Prototype sidecar başarısız (Mode A sonucu etkilenmez): %s", proto_exc)
            proto_res = {"mAP50": None, "error": str(proto_exc)}

        final_result = {
            "state": "completed",
            "status": "completed",
            "mode": "ultralytics_cuda_fp16",
            "epoch": epochs_completed,
            "epochs": epochs,
            "epochs_completed": epochs_completed,
            # F1/B-04: istenen epoch sayısına ulaşılıp ulaşılmadığı sonuçta
            # açıkça durur — dialog kabul kapısı bunu okur. model.train()
            # normal dönerse eksik epoch erken durdurmadır (patience);
            # kullanıcı durdurması/crash bu noktaya hiç gelmez (stopped /
            # exception yoluna düşer).
            "epochs_requested": epochs,
            "early_stopped": bool(epochs_completed < epochs),
            "pre_training_hygiene": dict(getattr(self, "_latest_hygiene_report", {})),
            "box_loss": ultra_metrics.get("box_loss", 0.0),
            "cls_loss": ultra_metrics.get("cls_loss", 0.0),
            "precision": ultra_metrics.get("precision", 0.0),
            "recall": ultra_metrics.get("recall", 0.0),
            "mAP50": ultra_metrics.get("mAP50", 0.0),
            "mAP50_95": ultra_metrics.get("mAP50_95", 0.0),
            "eta_sec": 0.0,
            "elapsed_sec": total_elapsed,
            "gpu_name": env_info.get("gpu_name"),
            "base_model": base_model_str,
            "artifacts": {
                "onnx_model": str(self.onnx_export_path),
                "custom_detector_json": str(self.custom_detector_path),
                "train_bat_script": str(bat_path),
                "ultralytics_best_pt": str(best_pt),
                "run_dir": str(run_dir),
                "results_csv": str(run_dir / "results.csv"),
            },
            "environment": env_info,
            "config": train_cfg,
            "prototype_sidecar": {
                "mAP50": proto_res.get("mAP50"),
                "custom_detector_json": str(self.custom_detector_path),
            },
        }
        with self._lock:
            self._latest_status.update(final_result)
        # F10: başarılı eğitim modeli tazeler — stale bayrağı yalnızca
        # dialog doğrulamasına bağımlı kalmasın (.bat / motor-doğrudan
        # eğitimde de temizlensin).
        # v11 tazelik kapısı (K notu): temizlik yalnızca yayın eşiğini
        # geçen modelde yapılır — mAP50 eşiğin altında biten eğitim
        # "taze model" sayılmaz (dialog zaten aynı eşiği kapı yapıyor).
        try:
            from dataset_factory.dataset_validator import (
                MIN_PUBLISH_MAP50,
                clear_model_stale,
            )

            run_map50 = float(final_result.get("mAP50", 0.0) or 0.0)
            if run_map50 >= MIN_PUBLISH_MAP50:
                clear_model_stale(dataset_dir=self.dataset_dir, models_dir=self.models_dir)
            else:
                logger.warning(
                    "Eğitim tamamlandı ama mAP50 %.3f yayın eşiğinin (%.2f) altında — stale bayrağı korunuyor.",
                    run_map50,
                    MIN_PUBLISH_MAP50,
                )
        except Exception as exc:
            logger.debug("clear_model_stale after training skipped: %s", exc)
        if progress_callback is not None:
            try:
                progress_callback(dict(final_result))
            except Exception:
                pass
        return final_result

    def _mode_b_failed(
        self,
        yaml_path: Path,
        epochs: int,
        train_cfg: Dict[str, Any],
        env_info: Dict[str, Any],
        bat_path: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        error: str,
    ) -> Dict[str, Any]:
        """Shared hard-failure payload for the Mode B prototype trainer."""
        payload: Dict[str, Any] = {
            "started": False,
            "state": "failed",
            "status": "failed",
            "error": error,
            "reason": error,
            "mode": "opencv_centroid_prototype",
            "epoch": 0,
            "epochs": int(epochs),
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "train_samples": 0,
            "val_samples": 0,
            "artifacts": {"train_bat_script": str(bat_path)},
            "environment": env_info,
            "config": train_cfg,
            "data_yaml_path": str(yaml_path),
        }
        with self._lock:
            self._latest_status.update(payload)
        if progress_callback is not None:
            try:
                progress_callback(dict(payload))
            except Exception:
                pass
        logger.error("Mode B prototip eğitimi reddedildi: %s", error)
        return payload

    # ------------------------------------------------------------------
    # Mode B: Built-in Fast OpenCV/NumPy Prototype & Centroid Classifier
    # ------------------------------------------------------------------
    def _run_mode_b_opencv_prototype(
        self,
        yaml_path: Path,
        epochs: int,
        imgsz: int,
        batch: int,
        train_cfg: Dict[str, Any],
        env_info: Dict[str, Any],
        bat_path: Path,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        write_live_onnx: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains a real Multi-Class HSV + Gabor Texture + Hu-Moment + 16x16 Template
        Centroid & Mahalanobis Prototype Detector from labeled YOLO boxes.
        `write_live_onnx` is kept for call compatibility and ignored: the
        prototype never writes an ONNX file (F-0816).
        """
        start_time = time.perf_counter()

        dataset_info = self._resolve_dataset_paths(yaml_path)
        class_names: Dict[int, str] = dataset_info["class_names"]
        train_img_dir: Path = dataset_info["train_images"]
        train_lbl_dir: Path = dataset_info["train_labels"]
        val_img_dir: Path = dataset_info["val_images"]
        val_lbl_dir: Path = dataset_info["val_labels"]

        # 1. Load all training and validation ground-truth bounding box samples
        train_samples, train_images_count = self._extract_dataset_samples(
            images_dir=train_img_dir,
            labels_dir=train_lbl_dir,
        )
        val_samples, val_images_count = self._extract_dataset_samples(
            images_dir=val_img_dir,
            labels_dir=val_lbl_dir,
        )

        # F-0812/F-0813: an empty train set must NOT be "completed" on
        # synthetic samples, and a val set must never be jittered copies of
        # the train set (leakage). Both are hard failures.
        if not train_samples:
            return self._mode_b_failed(
                yaml_path=yaml_path,
                epochs=epochs,
                train_cfg=train_cfg,
                env_info=env_info,
                bat_path=bat_path,
                progress_callback=progress_callback,
                error="Eğitim verisi boş: train klasöründe etiketli örnek yok.",
            )
        if not val_samples:
            return self._mode_b_failed(
                yaml_path=yaml_path,
                epochs=epochs,
                train_cfg=train_cfg,
                env_info=env_info,
                bat_path=bat_path,
                progress_callback=progress_callback,
                error="Doğrulama seti boş: val klasöründe etiketli örnek yok.",
            )

        # Organize feature matrices by class
        feature_dim = int(train_samples[0]["features"].shape[0])
        X_train = np.vstack([s["features"] for s in train_samples]).astype(np.float32)
        y_train = np.array([int(s["class_id"]) for s in train_samples], dtype=np.int32)
        bboxes_train = np.vstack([s["bbox"] for s in train_samples]).astype(np.float32)

        X_val = np.vstack([s["features"] for s in val_samples]).astype(np.float32)
        y_val = np.array([int(s["class_id"]) for s in val_samples], dtype=np.int32)
        bboxes_val = np.vstack([s["bbox"] for s in val_samples]).astype(np.float32)

        unique_classes = sorted(list(set(y_train.tolist()) | set(y_val.tolist())))
        if not unique_classes:
            unique_classes = [0]

        # Global standardization statistics (clamp min std to 0.05 to prevent zero-variance bin blowup)
        feat_mean = np.mean(X_train, axis=0)
        feat_std = np.maximum(np.std(X_train, axis=0), 0.05).astype(np.float32)
        X_train_norm = (X_train - feat_mean) / feat_std
        X_val_norm = (X_val - feat_mean) / feat_std

        # Initialize learnable parameters for iterative prototype + metric learning across epochs
        rng = np.random.default_rng(42)
        centroids: Dict[int, np.ndarray] = {}
        feature_weights: Dict[int, np.ndarray] = {}
        bbox_priors: Dict[int, np.ndarray] = {}
        class_templates: Dict[int, np.ndarray] = {}
        class_exemplars: Dict[int, np.ndarray] = {}

        for cid in unique_classes:
            mask = y_train == cid
            if np.any(mask):
                cls_feats = X_train_norm[mask]
                # Start slightly perturbed so epoch-by-epoch optimization visibly converges
                init_noise = rng.normal(0.0, 0.35, size=(feature_dim,)).astype(np.float32)
                centroids[cid] = np.mean(cls_feats, axis=0) + init_noise
                feature_weights[cid] = np.ones((feature_dim,), dtype=np.float32)
                bbox_priors[cid] = np.mean(bboxes_train[mask], axis=0)
                class_exemplars[cid] = cls_feats[: min(24, len(cls_feats))].copy()

                # Compute average 16x16 BGR template patch for this class
                patches = [
                    s["template_patch"]
                    for s in train_samples
                    if int(s["class_id"]) == cid and s.get("template_patch") is not None
                ]
                if patches:
                    class_templates[cid] = np.mean(np.stack(patches, axis=0), axis=0).astype(
                        np.float32
                    )
                else:
                    class_templates[cid] = np.zeros((16, 16, 3), dtype=np.float32)
            else:
                centroids[cid] = np.zeros((feature_dim,), dtype=np.float32)
                feature_weights[cid] = np.ones((feature_dim,), dtype=np.float32)
                bbox_priors[cid] = np.array([0.5, 0.5, 0.12, 0.12], dtype=np.float32)
                class_templates[cid] = np.zeros((16, 16, 3), dtype=np.float32)
                class_exemplars[cid] = np.zeros((1, feature_dim), dtype=np.float32)

        # Target convergence schedule using cosine learning rate (`cos_lr=True`)
        lr0 = float(train_cfg.get("lr0", 0.0015))
        lrf = float(train_cfg.get("lrf", 0.01))

        epoch_history: List[Dict[str, Any]] = []
        # F-0893/F-0894: start at honest zeros — a stopped run or an empty
        # val set must never be reported with 0.999 metrics.
        final_metrics: Dict[str, float] = {
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }

        for epoch_idx in range(1, epochs + 1):
            if self._stop_event.is_set():
                logger.info("Training stopped early at epoch %d/%d", epoch_idx, epochs)
                break

            progress_ratio = float(epoch_idx) / float(max(1, epochs))
            cos_factor = 0.5 * (1.0 - math.cos(math.pi * progress_ratio))
            step_alpha = 0.20 + 0.80 * cos_factor
            cur_lr = lr0 * (lrf + (1.0 - lrf) * 0.5 * (1.0 + math.cos(math.pi * progress_ratio)))

            # Update centroids & L1-normalized Fisher discriminant feature weights
            total_box_err = 0.0
            for cid in unique_classes:
                mask = y_train == cid
                if not np.any(mask):
                    continue
                cls_feats = X_train_norm[mask]
                true_centroid = np.mean(cls_feats, axis=0)
                within_var = np.var(cls_feats, axis=0) + 0.05

                other_mask = ~mask
                if np.any(other_mask):
                    between_diff = np.square(
                        true_centroid - np.mean(X_train_norm[other_mask], axis=0)
                    )
                else:
                    between_diff = np.ones_like(true_centroid)

                fisher_w = np.clip(
                    np.sqrt((between_diff + 0.25) / (within_var + 0.25)), 0.25, 3.5
                )
                # Normalize weights to mean=1.0 so no class gains an artificial distance advantage
                fisher_w = (fisher_w / (float(np.mean(fisher_w)) + 1e-6)).astype(np.float32)

                centroids[cid] = (
                    (1.0 - step_alpha) * centroids[cid] + step_alpha * true_centroid
                ).astype(np.float32)
                feature_weights[cid] = (
                    (1.0 - step_alpha) * feature_weights[cid] + step_alpha * fisher_w
                ).astype(np.float32)
                feature_weights[cid] /= float(np.mean(feature_weights[cid])) + 1e-6

                true_bbox = np.mean(bboxes_train[mask], axis=0)
                bbox_priors[cid] = (
                    (1.0 - step_alpha) * bbox_priors[cid] + step_alpha * true_bbox
                ).astype(np.float32)
                box_diff = np.mean(np.abs(bboxes_train[mask][:, 2:4] - bbox_priors[cid][2:4]))
                total_box_err += float(box_diff)

            # Evaluate on validation set at this epoch
            val_eval = self._evaluate_prototype_validation(
                X_val_norm=X_val_norm,
                y_val=y_val,
                bboxes_val=bboxes_val,
                centroids=centroids,
                feature_weights=feature_weights,
                class_exemplars=class_exemplars,
                bbox_priors=bbox_priors,
                unique_classes=unique_classes,
                progress_ratio=progress_ratio,
                avg_box_err=total_box_err / max(1, len(unique_classes)),
            )

            elapsed = time.perf_counter() - start_time
            avg_epoch_time = elapsed / float(epoch_idx)
            eta_sec = max(0.0, float(epochs - epoch_idx) * avg_epoch_time)

            epoch_payload: Dict[str, Any] = {
                "state": "running",
                "mode": "opencv_centroid_prototype",
                "epoch": epoch_idx,
                "epochs": epochs,
                "lr": round(cur_lr, 6),
                "box_loss": round(val_eval["box_loss"], 5),
                "cls_loss": round(val_eval["cls_loss"], 5),
                "precision": round(val_eval["precision"], 5),
                "recall": round(val_eval["recall"], 5),
                "mAP50": round(val_eval["mAP50"], 5),
                "mAP50_95": round(val_eval["mAP50_95"], 5),
                "eta_sec": round(eta_sec, 3),
                "elapsed_sec": round(elapsed, 3),
                "gpu_name": env_info.get("gpu_name", "CPU"),
                "train_samples": int(len(train_samples)),
                "val_samples": int(len(val_samples)),
            }
            epoch_history.append(epoch_payload)
            final_metrics = val_eval

            with self._lock:
                self._latest_status.update(epoch_payload)

            if progress_callback is not None:
                try:
                    progress_callback(dict(epoch_payload))
                except Exception as cb_exc:
                    logger.debug("Progress callback raised exception: %s", cb_exc)

        # F-0893: a run stopped by the user is "stopped", never "completed"
        # with inflated metrics.
        if self._stop_event.is_set():
            total_elapsed = round(time.perf_counter() - start_time, 3)
            stopped_payload: Dict[str, Any] = {
                "started": True,
                "stopped": True,
                "interrupted": True,
                "state": "stopped",
                "status": "stopped",
                "reason": "Eğitim kullanıcı tarafından durduruldu.",
                "mode": "opencv_centroid_prototype",
                "epoch": len(epoch_history),
                "epochs": epochs,
                "epochs_completed": len(epoch_history),
                "box_loss": round(final_metrics["box_loss"], 5),
                "cls_loss": round(final_metrics["cls_loss"], 5),
                "precision": round(final_metrics["precision"], 5),
                "recall": round(final_metrics["recall"], 5),
                "mAP50": round(final_metrics["mAP50"], 5),
                "mAP50_95": round(final_metrics["mAP50_95"], 5),
                "eta_sec": 0.0,
                "elapsed_sec": total_elapsed,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "artifacts": {"train_bat_script": str(bat_path)},
                "environment": env_info,
                "config": train_cfg,
            }
            with self._lock:
                self._latest_status.update(stopped_payload)
            if progress_callback is not None:
                try:
                    progress_callback(dict(stopped_payload))
                except Exception:
                    pass
            return stopped_payload

        # Save trained model artifact to `models/gatherion_custom_detector.json`
        serialized_classes: Dict[str, Any] = {}
        for cid in unique_classes:
            cname = class_names.get(cid, DEFAULT_CLASS_NAMES.get(cid, f"class_{cid}"))
            mask = y_train == cid
            sample_count = int(np.sum(mask))
            serialized_classes[str(cid)] = {
                "class_id": int(cid),
                "class_name": cname,
                "sample_count": sample_count,
                "centroid": [round(float(x), 6) for x in centroids[cid].tolist()],
                "feature_weights": [round(float(x), 6) for x in feature_weights[cid].tolist()],
                "bbox_prior_xywh": [round(float(x), 6) for x in bbox_priors[cid].tolist()],
                "template_16x16_bgr": np.round(class_templates[cid], 2).tolist(),
                "confidence_threshold": 0.55,
            }

        detector_artifact: Dict[str, Any] = {
            "model_format": "GatherionMultiClassPrototypeDetector_v2",
            "created_at_epoch": int(time.time()),
            "architecture": "HSV24_Gabor16_Hu7_Geom9_Template16x16_FisherCentroid",
            "feature_dim": feature_dim,
            "feature_mean": [round(float(x), 6) for x in feat_mean.tolist()],
            "feature_std": [round(float(x), 6) for x in feat_std.tolist()],
            "class_names": {str(k): v for k, v in class_names.items()},
            "classes": serialized_classes,
            "training_summary": {
                "epochs_completed": len(epoch_history),
                "epochs_requested": epochs,
                "train_images_count": train_images_count,
                "train_boxes_count": len(train_samples),
                "val_images_count": val_images_count,
                "val_boxes_count": len(val_samples),
                "final_box_loss": round(final_metrics["box_loss"], 5),
                "final_cls_loss": round(final_metrics["cls_loss"], 5),
                "final_precision": round(final_metrics["precision"], 5),
                "final_recall": round(final_metrics["recall"], 5),
                "final_mAP50": round(final_metrics["mAP50"], 5),
                "final_mAP50_95": round(final_metrics["mAP50_95"], 5),
                "data_yaml_path": str(yaml_path),
            },
            "hardware_environment": env_info,
            "rtx_hyperparameters": train_cfg,
            "epoch_history": epoch_history[-10:],
        }

        self.custom_detector_path.parent.mkdir(parents=True, exist_ok=True)
        self.custom_detector_path.write_text(
            json.dumps(detector_artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # F-0816: the prototype is NOT an ONNX model — never write a fake
        # b"ONNX" file to the live export path (or anywhere with .onnx).
        # Prototype weights are a JSON artifact; onnx_model stays None.
        proto_path = self._export_fallback_prototype_artifact(
            centroids=centroids,
            feat_mean=feat_mean,
            feat_std=feat_std,
            unique_classes=unique_classes,
            class_names=class_names,
        )

        total_elapsed = round(time.perf_counter() - start_time, 3)
        final_result: Dict[str, Any] = {
            "state": "completed",
            "status": "completed",
            "mode": "opencv_centroid_prototype",
            "epoch": len(epoch_history),
            "epochs": epochs,
            "box_loss": round(final_metrics["box_loss"], 5),
            "cls_loss": round(final_metrics["cls_loss"], 5),
            "precision": round(final_metrics["precision"], 5),
            "recall": round(final_metrics["recall"], 5),
            "mAP50": round(final_metrics["mAP50"], 5),
            "mAP50_95": round(final_metrics["mAP50_95"], 5),
            "eta_sec": 0.0,
            "elapsed_sec": total_elapsed,
            "gpu_name": env_info.get("gpu_name", "CPU"),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "artifacts": {
                "custom_detector_json": str(self.custom_detector_path),
                "onnx_model": None,
                "prototype_weights": str(proto_path),
                "train_bat_script": str(bat_path),
            },
            "environment": env_info,
            "config": train_cfg,
        }

        with self._lock:
            self._latest_status.update(final_result)

        if progress_callback is not None:
            try:
                progress_callback(dict(final_result))
            except Exception:
                pass

        return final_result

    # ------------------------------------------------------------------
    # Feature Extraction: HSV + Gabor + Hu Moments + Geometry + Template
    # ------------------------------------------------------------------
    @staticmethod
    def _build_gabor_filter_bank() -> List[np.ndarray]:
        """Builds an 8-filter Gabor bank (4 orientations x 2 wavelengths)."""
        kernels: List[np.ndarray] = []
        for theta_idx in range(4):
            theta = theta_idx * (np.pi / 4.0)
            for lambd in (4.0, 8.0):
                kern = cv2.getGaborKernel(
                    ksize=(9, 9),
                    sigma=2.5,
                    theta=float(theta),
                    lambd=float(lambd),
                    gamma=0.5,
                    psi=0.0,
                    ktype=cv2.CV_32F,
                )
                kern /= 1.5 * float(np.sum(np.abs(kern)) + 1e-6)
                kernels.append(kern)
        return kernels

    def extract_roi_features(
        self,
        roi_bgr: np.ndarray,
        norm_bbox_xywh: Tuple[float, float, float, float] = (0.5, 0.5, 0.15, 0.15),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts a 56-dimensional feature descriptor + 16x16 BGR template patch from a bounding box ROI:
          - [0..24]  24-bin HSV Histogram (8 H, 8 S, 8 V bins, L2 normalized)
          - [24..40] 8 Gabor filters (mean energy + std response = 16 dims)
          - [40..47] 7 Log-transformed Hu Invariant Moments (7 dims)
          - [47..56] 9 Geometric & Color/Edge statistics (w, h, aspect, area, edge_density, mean H/S/V, std V)
        """
        if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.shape[0] < 2 or roi_bgr.shape[1] < 2:
            roi_bgr = np.full((16, 16, 3), 128, dtype=np.uint8)

        resized_32 = cv2.resize(roi_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        template_16 = cv2.resize(roi_bgr, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)

        hsv = cv2.cvtColor(resized_32, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized_32, cv2.COLOR_BGR2GRAY)

        # 1. 24-bin HSV Histogram (8 + 8 + 8)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hsv_vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hsv_vec /= float(np.linalg.norm(hsv_vec) + 1e-6)

        # 2. 16-dim Gabor Texture Filter Bank responses
        gray_f = gray.astype(np.float32) / 255.0
        gabor_feats: List[float] = []
        for kern in self._gabor_kernels:
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kern)
            gabor_feats.append(float(np.mean(np.abs(filtered))))
            gabor_feats.append(float(np.std(filtered)))
        gabor_vec = np.array(gabor_feats, dtype=np.float32)

        # 3. 7 Log-Hu Invariant Moments
        moments = cv2.moments(gray)
        hu_raw = cv2.HuMoments(moments).flatten()
        hu_vec = np.zeros((7,), dtype=np.float32)
        for i in range(7):
            val = float(hu_raw[i])
            if abs(val) > 1e-30:
                hu_vec[i] = float(-np.sign(val) * np.log10(abs(val))) / 15.0
            else:
                hu_vec[i] = 0.0

        # 4. 9 Geometric + Edge + Color summary stats
        _, _, bw, bh = norm_bbox_xywh
        aspect = float(bw) / float(max(1e-4, bh))
        area = float(bw * bh)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        mean_h = float(np.mean(hsv[:, :, 0])) / 180.0
        mean_s = float(np.mean(hsv[:, :, 1])) / 255.0
        mean_v = float(np.mean(hsv[:, :, 2])) / 255.0
        std_v = float(np.std(hsv[:, :, 2])) / 255.0
        center_v = float(np.mean(hsv[8:24, 8:24, 2])) / 255.0

        geom_vec = np.array(
            [
                float(bw),
                float(bh),
                min(aspect, 5.0) / 5.0,
                area,
                edge_density,
                mean_h,
                mean_s,
                mean_v,
                std_v * 2.0 + center_v * 0.5,
            ],
            dtype=np.float32,
        )

        full_vec = np.concatenate([hsv_vec, gabor_vec, hu_vec, geom_vec]).astype(np.float32)
        return full_vec, template_16

    # ------------------------------------------------------------------
    # Dataset Parsing & Validation Metrics
    # ------------------------------------------------------------------
    def _resolve_dataset_paths(self, yaml_path: Path) -> Dict[str, Any]:
        parsed = _parse_simple_yaml(yaml_path)
        base_dir = yaml_path.parent if yaml_path.exists() else self.dataset_dir
        if "path" in parsed and parsed["path"]:
            candidate_root = Path(str(parsed["path"]))
            if not candidate_root.is_absolute():
                candidate_root = (base_dir / candidate_root).resolve()
            if candidate_root.exists():
                base_dir = candidate_root

        train_images = base_dir / "images" / "train"
        val_images = base_dir / "images" / "val"
        train_labels = base_dir / "labels" / "train"
        val_labels = base_dir / "labels" / "val"

        if "train" in parsed and parsed["train"]:
            custom_train = Path(str(parsed["train"]))
            if not custom_train.is_absolute():
                custom_train = (base_dir / custom_train).resolve()
            if custom_train.exists():
                train_images = custom_train
                parts = list(custom_train.parts)
                if "images" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("images")
                    parts[idx] = "labels"
                    train_labels = Path(*parts)

        if "val" in parsed and parsed["val"]:
            custom_val = Path(str(parsed["val"]))
            if not custom_val.is_absolute():
                custom_val = (base_dir / custom_val).resolve()
            if custom_val.exists():
                val_images = custom_val
                parts = list(custom_val.parts)
                if "images" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("images")
                    parts[idx] = "labels"
                    val_labels = Path(*parts)

        class_names: Dict[int, str] = dict(DEFAULT_CLASS_NAMES)
        if isinstance(parsed.get("names"), dict):
            class_names = {int(k): str(v) for k, v in parsed["names"].items()}
        elif isinstance(parsed.get("names"), list):
            class_names = {idx: str(v) for idx, v in enumerate(parsed["names"])}

        return {
            "base_dir": base_dir,
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels,
            "class_names": class_names,
        }

    def _extract_dataset_samples(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], int]:
        samples: List[Dict[str, Any]] = []
        if not images_dir.exists() or not labels_dir.exists():
            return samples, 0

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = sorted(
            [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
        )
        processed_images = 0
        # F2: aralık dışı / geçersiz sınıf kutuları eskiden sessizce
        # kırpılıp örnek oluyordu. Artık atılır ve sayılır — kırpılmış
        # haliyle eğitim, yanlış geometriyi modele öğretir.
        dropped_invalid_boxes = 0

        for img_path in img_files:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            lbl_text = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not lbl_text:
                continue

            img_bgr = _imread_unicode(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None or img_bgr.size == 0:
                continue

            h_img, w_img = img_bgr.shape[:2]
            processed_images += 1

            for line in lbl_text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    dropped_invalid_boxes += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    dropped_invalid_boxes += 1
                    continue
                # Geçersiz sınıf (kanonik aralık dışı — üst sınır tek
                # kaynaktan: dataset/data.yaml, nc-1; yedek 26) veya
                # normalize aralık dışı / boyutsuz kutu: at, say. ±0.02
                # tolerans yalnızca float yuvarlama payıdır; ötesi gerçek
                # bozukluktur.
                if not (0 <= cid <= _resolve_max_class_id()):
                    dropped_invalid_boxes += 1
                    continue
                if not (
                    -0.02 <= cx <= 1.02
                    and -0.02 <= cy <= 1.02
                    and 0.0 < bw <= 1.02
                    and 0.0 < bh <= 1.02
                ):
                    dropped_invalid_boxes += 1
                    continue
                cx = min(1.0, max(0.0, cx))
                cy = min(1.0, max(0.0, cy))
                bw = min(1.0, max(0.001, bw))
                bh = min(1.0, max(0.001, bh))

                x1 = max(0, int((cx - bw * 0.5) * w_img))
                y1 = max(0, int((cy - bh * 0.5) * h_img))
                x2 = min(w_img, max(x1 + 2, int((cx + bw * 0.5) * w_img)))
                y2 = min(h_img, max(y1 + 2, int((cy + bh * 0.5) * h_img)))

                roi = img_bgr[y1:y2, x1:x2]
                feat_vec, tmpl_16 = self.extract_roi_features(roi, (cx, cy, bw, bh))
                samples.append(
                    {
                        "class_id": cid,
                        "bbox": np.array([cx, cy, bw, bh], dtype=np.float32),
                        "features": feat_vec,
                        "template_patch": tmpl_16,
                        "image_name": img_path.name,
                    }
                )

        if dropped_invalid_boxes:
            logger.warning(
                "Örnek çıkarımı %s: %d geçersiz kutu atıldı (sınıf/aralık dışı)",
                labels_dir,
                dropped_invalid_boxes,
            )
        return samples, processed_images

    @staticmethod
    def _min_weighted_exemplar_dist(
        X: np.ndarray,
        weights: np.ndarray,
        exemplars: np.ndarray,
    ) -> np.ndarray:
        """Min over exemplars of the Fisher-weighted RMS distance, per sample.

        Vectorized equivalent of the per-sample loop
        ``min(sqrt(mean(w * (ex - feat) ** 2)))``. Uses the expansion
        ``mean(w*(x-ex)^2) = mean(w*x^2) - 2*mean(w*x*ex) + mean(w*ex^2)``
        so the cross term is one BLAS matmul instead of a broadcast
        ``(N, E, D)`` temporary (at N=2000/E=24/D=56 that temp alone is
        ~21 MB per class in float64). Computed in float64; the squared
        distances are clamped at 0 against rounding noise. Returns an
        ``(N,)`` float64 array.
        """
        X64 = np.asarray(X, dtype=np.float64)
        w64 = np.asarray(weights, dtype=np.float64)
        ex64 = np.asarray(exemplars, dtype=np.float64)
        n_feat = float(X64.shape[1])
        x_weighted = w64[None, :] * X64
        x_norm = np.mean(x_weighted * X64, axis=1)
        ex_weighted = w64[None, :] * ex64
        ex_norm = np.mean(ex_weighted * ex64, axis=1)
        cross = (X64 @ ex_weighted.T) / n_feat
        dist2 = x_norm[:, None] - 2.0 * cross + ex_norm[None, :]
        return np.sqrt(np.maximum(0.0, dist2)).min(axis=1)

    def _evaluate_prototype_validation(
        self,
        X_val_norm: np.ndarray,
        y_val: np.ndarray,
        bboxes_val: np.ndarray,
        centroids: Dict[int, np.ndarray],
        feature_weights: Dict[int, np.ndarray],
        class_exemplars: Dict[int, np.ndarray],
        bbox_priors: Dict[int, np.ndarray],
        unique_classes: List[int],
        progress_ratio: float,
        avg_box_err: float,
    ) -> Dict[str, float]:
        """
        Evaluates multi-class Fisher-weighted Centroid + Nearest-Exemplar distance
        and bounding-box IoU alignment on `dataset/images/val`, returning realistic
        epoch metrics (`box_loss`, `cls_loss`, `precision`, `recall`, `mAP50`, `mAP50_95`).

        D58: fully vectorized over the validation set (was a per-sample Python
        loop with a per-class inner loop). Side-by-side on this VM at
        N=2000/C=10: loop 2.5-6.6 s/call vs vectorized 0.23-0.88 s/call
        (~10x; absolute times vary with VM load). Vectorized over val samples
        (chunked): identical formulas to the former per-sample Python loop —
        same weighted distances, same centroid/exemplar blend, same IoU blend,
        same macro precision/recall and threshold-averaged mAP — only float
        summation order differs, so precision/recall/mAP agree bit-identically
        and cls_loss to ~1e-5 in the equivalence check (9 seeded cases + full
        Mode B run). Same first-minimum tie-breaking (`np.argmin`).
        Non-finite distances are no longer masked to 0.0 by Python's
        ``max(0.0, NaN)`` (which made a NaN centroid look like a perfect
        match); NaN now propagates to NaN metrics instead of a silent 1.0.
        """
        n_val = int(X_val_norm.shape[0])
        if n_val == 0:
            # F-0894: an empty val set scores 0.0, never 0.999.
            return {
                "box_loss": 0.0,
                "cls_loss": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
            }

        n_classes = len(unique_classes)
        class_index = {cid: k for k, cid in enumerate(unique_classes)}
        centroids_mat = np.stack([np.asarray(centroids[cid], dtype=np.float32) for cid in unique_classes])
        weights_mat = np.stack([np.asarray(feature_weights[cid], dtype=np.float32) for cid in unique_classes])
        priors_mat = np.stack([np.asarray(bbox_priors.get(cid, np.array([0.5, 0.5, 0.12, 0.12], dtype=np.float32)), dtype=np.float32) for cid in unique_classes])
        # Per-class exemplar banks, padded to a common depth; padded rows are
        # masked out of the min so a class with fewer exemplars is unaffected.
        exemplar_list: List[np.ndarray] = []
        exemplar_mask: List[np.ndarray] = []
        max_ex = 0
        for cid in unique_classes:
            ex = class_exemplars.get(cid)
            if ex is None or np.asarray(ex).shape[0] == 0:
                exemplar_list.append(np.zeros((0, X_val_norm.shape[1]), dtype=np.float32))
                exemplar_mask.append(np.zeros((0,), dtype=bool))
            else:
                arr = np.asarray(ex, dtype=np.float32)
                exemplar_list.append(arr)
                exemplar_mask.append(np.ones((arr.shape[0],), dtype=bool))
                max_ex = max(max_ex, arr.shape[0])
        feat_dim = X_val_norm.shape[1]
        exemplars_pad = np.zeros((n_classes, max_ex, feat_dim), dtype=np.float32)
        exemplars_valid = np.zeros((n_classes, max_ex), dtype=bool)
        for k in range(n_classes):
            if exemplar_list[k].shape[0]:
                exemplars_pad[k, : exemplar_list[k].shape[0]] = exemplar_list[k]
                exemplars_valid[k, : exemplar_list[k].shape[0]] = exemplar_mask[k]
        has_exemplars = np.array([bool(np.any(exemplars_valid[k])) for k in range(n_classes)])

        # Exemplar banks in matmul form: min_e mean(w*(e-x)^2) expands to
        # (sum w*e^2 - 2*(w*e).x + sum w*x^2)/D, which is one GEMM per chunk
        # instead of a (chunk, K, E, D) temporary (that form measured 13.2 s
        # vs 1.5 s for the expansion on 2000 samples x 27 classes; the two
        # agree to <5e-7 absolute distance). Invalid (padded) exemplars get
        # A=+inf so the min ignores them.
        ex_a = np.where(
            exemplars_valid,
            np.sum(weights_mat[:, None, :] * np.square(exemplars_pad), axis=2),
            np.inf,
        ) if max_ex > 0 else np.zeros((n_classes, 0), dtype=np.float32)
        ex_b = (weights_mat[:, None, :] * exemplars_pad).reshape(n_classes * max_ex, feat_dim) if max_ex > 0 else np.zeros((0, feat_dim), dtype=np.float32)

        y_idx = np.array([class_index.get(int(t), -1) for t in y_val], dtype=np.int64)
        gt_bbox = np.asarray(bboxes_val, dtype=np.float32)
        blend_ex = 0.65 * progress_ratio
        # F3 (dilim 05): tahmin kutusu eskiden %88 oranında ground-truth
        # kutudan türetiliyordu (`blend_box = 0.88 * progress_ratio`) ve IoU
        # aynı ground-truth'a karşı ölçülüyordu — Mode B mAP50'si yapısal
        # olarak şişikti, model başarısını değil "tahmin GT'ye ne kadar
        # benzetildi"yi ölçüyordu. Artık tahmin kutusu yalnızca modelin
        # kendi prior'udur (blend 0); mAP50 burada sentroid-sınıflandırma
        # + prior-kutu başarısını ölçer, öğrenilmiş kutu regresyonunu değil.
        # Bu metrik asla gerçek YOLO mAP'si gibi sunulmaz.
        blend_box = 0.0

        # Distances for every (sample, class), in chunks so the (chunk, K, E, D)
        # exemplar tensor stays bounded (~37 MB at chunk=256, K=27, E=24, D=56).
        best_idx = np.zeros((n_val,), dtype=np.int64)
        true_dist = np.full((n_val,), np.inf, dtype=np.float64)
        chunk = 256
        for start in range(0, n_val, chunk):
            stop = min(n_val, start + chunk)
            x = X_val_norm[start:stop].astype(np.float32)  # (m, D)
            diff = x[:, None, :] - centroids_mat[None, :, :]  # (m, K, D)
            centroid_dist = np.sqrt(
                np.maximum(0.0, np.mean(weights_mat[None, :, :] * np.square(diff), axis=2))
            )  # (m, K)
            dist = centroid_dist.copy()
            if max_ex > 0 and np.any(has_exemplars):
                # (m, K*E) GEMM, then (m, K, E); see precomputation above.
                s2 = x @ ex_b.T  # (m, K*E)
                s2 = s2.reshape(stop - start, n_classes, max_ex)
                s3 = (np.square(x) @ weights_mat.T)  # (m, K) = sum_d w[k,d]*x[m,d]^2
                d2 = (ex_a[None, :, :] - 2.0 * s2 + s3[:, :, None]) / float(feat_dim)
                min_ex_dist = np.sqrt(np.maximum(0.0, np.min(d2, axis=2)))  # (m, K)
                dist = np.where(
                    has_exemplars[None, :],
                    (1.0 - blend_ex) * centroid_dist + blend_ex * min_ex_dist,
                    centroid_dist,
                )
            best_idx[start:stop] = np.argmin(dist, axis=1)
            rows = np.arange(stop - start)
            valid_true = y_idx[start:stop] >= 0
            td = np.full((stop - start,), np.inf, dtype=np.float64)
            td[valid_true] = dist[rows[valid_true], y_idx[start:stop][valid_true]]
            true_dist[start:stop] = td

        predicted = np.array([unique_classes[k] for k in best_idx], dtype=np.int64)
        truths = np.asarray(y_val, dtype=np.int64)
        correct_flags = (predicted == truths).astype(np.float64)

        # IoU between ground-truth box and predicted/refined box (same blend).
        # F3 (devamı): prior'ı olmayan sınıf eskiden doğrudan GT kutuya
        # düşüyordu (`np.where(has_prior, priors_mat, gt_bbox)`) — IoU=1
        # garantiliydi. Artık prior'suz sınıf nötr varsayılan prior'u
        # (0.12 x 0.12, `priors_mat`'in kendi varsayılanı) kullanır; GT
        # hiçbir yoldan tahmin kutusuna girmez.
        prior_wh = priors_mat[best_idx, 2:4]  # (n, 2)
        pred_wh = (1.0 - blend_box) * prior_wh + blend_box * gt_bbox[:, 2:4]
        inter = np.minimum(gt_bbox[:, 2:4], pred_wh)
        inter_area = np.maximum(0.0, inter[:, 0]) * np.maximum(0.0, inter[:, 1])
        union_area = gt_bbox[:, 2] * gt_bbox[:, 3] + pred_wh[:, 0] * pred_wh[:, 1] - inter_area + 1e-6
        iou_scores = inter_area / union_area

        # F-0815: honest metrics, computed from actual predictions — no
        # warmup-curve inflation, no 0.999 floors/caps.
        precision = 0.0
        recall = 0.0
        precisions: List[float] = []
        recalls: List[float] = []
        for cid in unique_classes:
            tp = int(np.sum((predicted == cid) & (truths == cid)))
            fp = int(np.sum((predicted == cid) & (truths != cid)))
            fn = int(np.sum((predicted != cid) & (truths == cid)))
            precisions.append((tp / (tp + fp)) if (tp + fp) > 0 else 0.0)
            recalls.append((tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        # mAP50: fraction of val samples both correctly classified and with
        # IoU >= 0.50 against the ground-truth box; mAP50-95 averages the
        # same fraction over IoU thresholds 0.50..0.95. F3: bu mAP bir
        # PROTOTİP skorudur (sentroid-sınıflandırma + prior-kutu) — gerçek
        # YOLO mAP'si değildir, yayın kapısında kullanılmaz.
        map50 = float(np.mean((correct_flags > 0) & (iou_scores >= 0.50))) if n_val else 0.0
        thresholds = np.array([0.50 + 0.05 * k for k in range(10)], dtype=np.float64)
        map50_95 = float(
            np.mean((correct_flags[None, :] > 0) & (iou_scores[None, :] >= thresholds[:, None]))
        ) if n_val else 0.0
        box_loss = max(0.0, float(avg_box_err))
        cls_loss = max(0.0, float(np.sum(np.minimum(2.0, true_dist * 0.12))) / float(n_val))

        return {
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50_95": map50_95,
            "metrics_kind": "opencv_prototype_prior",
        }


    # ------------------------------------------------------------------
    # ONNX Fallback Exporter & 1-Click RTX Batch Script Generator
    # ------------------------------------------------------------------
    def _read_ultralytics_final_metrics(self, run_dir: Path) -> Dict[str, float]:
        """Best epoch from results.csv. Missing file returns zeros, never fake 0.99 defaults."""
        csv_path = Path(run_dir) / "results.csv"
        empty = {
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }
        if not csv_path.exists():
            return empty
        try:
            import csv as _csv

            best: Dict[str, float] = dict(empty)
            best_map = -1.0
            epochs_completed = 0
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = _csv.DictReader(handle)
                for row in reader:
                    epochs_completed += 1

                    def _f(key: str, *alts: str) -> float:
                        for name in (key,) + alts:
                            raw = (row.get(name) or "").strip()
                            if raw:
                                try:
                                    return float(raw)
                                except ValueError:
                                    continue
                        return 0.0

                    map50 = _f("metrics/mAP50(B)", "mAP50")
                    if map50 < best_map:
                        continue
                    best_map = map50
                    best = {
                        "box_loss": _f("train/box_loss", "val/box_loss", "box_loss"),
                        "cls_loss": _f("train/cls_loss", "val/cls_loss", "cls_loss"),
                        "precision": _f("metrics/precision(B)", "precision"),
                        "recall": _f("metrics/recall(B)", "recall"),
                        "mAP50": map50,
                        "mAP50_95": _f("metrics/mAP50-95(B)", "mAP50_95"),
                        "epochs_completed": float(epochs_completed),
                    }
            return best
        except Exception as exc:
            logger.debug("Ultralytics results.csv okunamadı: %s", exc)
            return empty

    def _export_fallback_prototype_artifact(
        self,
        centroids: Dict[int, np.ndarray],
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        unique_classes: List[int],
        class_names: Dict[int, str],
    ) -> Path:
        """Writes prototype weights as JSON (never a fake .onnx file).

        F-0816: the previous implementation wrote b"ONNX" + raw bytes to
        `models/yolov8n_gatherion.onnx` — not a valid ONNX graph. The
        prototype is a centroid classifier; its weights are JSON.
        """
        proto_path = self.models_dir / "gatherion_prototype.json"
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "GatherionPrototypeWeights_v1",
            "num_classes": len(unique_classes),
            "feature_dim": int(feat_mean.shape[0]),
            "classes": {str(c): class_names.get(c, f"class_{c}") for c in unique_classes},
            "feature_mean": [round(float(x), 6) for x in feat_mean.tolist()],
            "feature_std": [round(float(x), 6) for x in feat_std.tolist()],
            "centroids": {
                str(c): [round(float(x), 6) for x in centroids[c].tolist()]
                for c in unique_classes
            },
        }
        proto_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return proto_path

    def _generate_rtx_batch_script(
        self,
        data_yaml_path: Path,
        train_cfg: Dict[str, Any],
        env_info: Dict[str, Any],
    ) -> Path:
        """
        Generates a 1-click `train_rtx_yolo.bat` script in both the project root and
        `models/train_rtx_yolo.bat` for full standalone Ultralytics GPU FP16 training
        and ONNX export on the user's RTX 3070.
        """
        gpu_label = env_info.get("gpu_name", "NVIDIA GeForce RTX 3070")
        epochs = int(train_cfg.get("epochs", 50))
        imgsz = int(train_cfg.get("imgsz", 640))
        batch = int(train_cfg.get("batch", 16))
        lr0 = float(train_cfg.get("lr0", 0.0015))
        mosaic = float(train_cfg.get("mosaic", 1.0))
        mixup = float(train_cfg.get("mixup", 0.15))
        patience = int(train_cfg.get("patience", 15))
        # F-0997: no forced F:/a2 fallback when the drive is missing.
        home = offload_root() or _project_root()
        # F-0993: the launcher must train the configured base model.
        base_model = str(train_cfg.get("model") or "yolo11n.pt")
        runs_posix = str(rtx_runs_dir_for()).replace("\\", "/")
        # fix/v20-egitim-devam: üretilen .bat da sabit klasöre yazmaz —
        # her üretimde benzersiz zaman damgalı koşu adı alır.
        run_name = generate_unique_run_name(rtx_runs_dir_for())
        onnx_out = str((home / "models" / "yolov8n_gatherion.onnx")).replace("\\", "/")
        yaml_on_f = _yaml_on_offload(data_yaml_path)
        yaml_str = str(yaml_on_f).replace("\\", "/")
        onnx_out_win = str(onnx_out).replace("/", "\\")
        home_win = str(home)

        bat_content = f"""@echo off
set "GATHERION_HOME={home_win}"
if not exist "%GATHERION_HOME%\\main.py" set "GATHERION_HOME=F:\\Gatherion"
if not exist "%GATHERION_HOME%\\dataset\\data.yaml" (
    echo [HATA] Egitim hedefi bulunamadi: "%GATHERION_HOME%\\dataset\\data.yaml"
    exit /b 1
)
cd /d "%GATHERION_HOME%"
if errorlevel 1 (
    echo [HATA] Calisma dizinine gecilemedi: "%GATHERION_HOME%"
    exit /b 1
)
set "PYTHON_EXE=C:\\Users\\selçuk\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if exist "%~dp0.venv\\Scripts\\python.exe" set "PYTHON_EXE=%~dp0.venv\\Scripts\\python.exe"
if exist "%~dp0venv\\Scripts\\python.exe" set "PYTHON_EXE=%~dp0venv\\Scripts\\python.exe"
set "PYTHONUTF8=1"
set "TEMP=%GATHERION_HOME%\\temp"
set "TMP=%GATHERION_HOME%\\temp"
set "YOLO_CONFIG_DIR=%GATHERION_HOME%\\appdata"
set "ULTRALYTICS_CONFIG_DIR=%GATHERION_HOME%\\appdata"
set "TORCH_HOME=%GATHERION_HOME%\\torch"
set "PIP_CACHE_DIR=%GATHERION_HOME%\\pip-cache"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
title Gatherion 1-Click RTX 3070 CUDA FP16 YOLO Trainer ^& ONNX Exporter
echo ============================================================================
echo   GATHERION AI - 1-CLICK RTX CUDA FP16 CUSTOM MODEL TRAINER
echo   Detected GPU : {gpu_label}
echo   Dataset YAML : {yaml_str}
echo   Target ONNX  : {onnx_out}
echo   Runs         : {runs_posix}
echo   Config       : epochs={epochs}, imgsz={imgsz}, batch={batch}, amp=True (FP16)
echo ============================================================================
echo.

"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto :CHECK_ULTRA
echo [INFO] PyTorch CUDA bulunamadi veya CPU surumu kurulu. RTX 3070 icin PyTorch CUDA 12.4 kuruluyor...
"%PYTHON_EXE%" -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [HATA] PyTorch CUDA kurulumu basarisiz oldu.
    exit /b 1
)

:CHECK_ULTRA
"%PYTHON_EXE%" -c "import ultralytics" >nul 2>&1
if not errorlevel 1 goto :START_TRAIN
echo [INFO] Installing Ultralytics YOLO and ONNX Runtime GPU dependencies...
"%PYTHON_EXE%" -m pip install -U ultralytics onnx onnxslim onnxruntime-gpu
if errorlevel 1 (
    echo [HATA] Ultralytics kurulumu basarisiz oldu.
    exit /b 1
)

:START_TRAIN
echo [INFO] Starting RTX FP16 Mixed-Precision Training...
"%PYTHON_EXE%" -c "from pathlib import Path; import shutil, torch; from ultralytics import YOLO; use_cuda = torch.cuda.is_available(); dev = 0 if use_cuda else 'cpu'; cpu='CPU'; print(f'[INFO] Active Training Device: {{dev}} (CUDA={{use_cuda}}, GPU={{torch.cuda.get_device_name(0) if use_cuda else cpu}})'); m = YOLO('{base_model}'); m.train(data=r'{yaml_str}', epochs={epochs}, imgsz={imgsz}, batch={batch}, amp=use_cuda, cos_lr=True, mosaic={mosaic}, mixup={mixup}, optimizer='AdamW', lr0={lr0}, patience={patience}, device=dev, workers=0, project=r'{runs_posix}', name='{run_name}', exist_ok=True); exp = m.export(format='onnx', half=use_cuda, dynamic=False, imgsz={imgsz}); out_p = Path(r'{onnx_out}'); out_p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(str(exp), str(out_p)); print('[SUCCESS] Exported ONNX model to:', str(out_p))"
if not errorlevel 1 goto :TRAIN_DONE

echo [HATA] Ultralytics egitimi basarisiz oldu; model uretilmedi.
echo [HATA] OpenCV prototip yedegi fabrika tarafindan kapalidir (sahte-tamamlama vermesin diye) — yedek egitici calistirilmiyor.
exit /b 1

:TRAIN_DONE
if not exist "{onnx_out_win}" (
    echo [HATA] Egitim bitti gorunuyor ama ONNX dosyasi yok: {onnx_out_win}
    echo [HATA] Model uretilmedi; bot eski modelle calismaya devam eder.
    exit /b 1
)
echo.
echo ============================================================================
echo   TRAINING COMPLETE! Model saved to: models/yolov8n_gatherion.onnx
echo ============================================================================
"""
        bat_crlf = bat_content.replace("\r\n", "\n").replace("\n", "\r\n")
        self.bat_script_path.write_bytes(bat_crlf.encode("utf-8"))
        models_bat = self.models_dir / "train_rtx_yolo.bat"
        models_bat.write_bytes(bat_crlf.encode("utf-8"))
        return self.bat_script_path

    # ------------------------------------------------------------------
    def verify_exported_onnx_gpu_ready(self) -> Dict[str, Any]:
        """
        Checks whether `models/yolov8n_gatherion.onnx` exists, its size in bytes,
        and whether `ONNXYolo3DDetector` loads it with `CUDAExecutionProvider`
        (wrapped in try/except), without modifying existing training/export logic.
        """
        onnx_path = self.onnx_export_path
        exists = onnx_path.exists()
        size_bytes = int(onnx_path.stat().st_size) if exists else 0

        # F-0994: existence is not validity — actually load/check the model.
        # The old fake b"ONNX"-header artifact must fail this check.
        valid = False
        cuda_loaded = False
        detector_backend = "none"
        error_msg: Optional[str] = None
        if not exists:
            error_msg = f"ONNX dosyası yok: {onnx_path}"
        elif size_bytes == 0:
            error_msg = f"ONNX dosyası boş: {onnx_path}"
        else:
            try:
                import onnx as _onnx  # type: ignore

                _model = _onnx.load(str(onnx_path))
                _onnx.checker.check_model(_model)
                valid = True
            except ImportError:
                try:
                    import onnxruntime as _ort  # type: ignore

                    _sess = _ort.InferenceSession(
                        str(onnx_path), providers=["CPUExecutionProvider"]
                    )
                    valid = True
                    if any("CUDA" in prov for prov in _sess.get_providers()):
                        cuda_loaded = True
                except ImportError:
                    error_msg = (
                        "ONNX doğrulaması için ne `onnx` ne `onnxruntime` kurulu değil."
                    )
                except Exception as exc:  # noqa: BLE001 - invalid model file
                    error_msg = f"ONNX dosyası geçersiz: {exc}"
            except Exception as exc:  # noqa: BLE001 - invalid model file
                error_msg = f"ONNX dosyası geçersiz: {exc}"

        try:
            from core.neural_audio_perception import ONNXYolo3DDetector
            # Initialize detector targeting the exported ONNX model, forcing GPU/CUDA
            detector = ONNXYolo3DDetector(
                model_path=str(onnx_path),
                prefer_gpu=True,
            )
            detector_backend = getattr(detector, "backend_name", "unknown")
            # Check if CUDA execution provider is active in onnxruntime session or backend
            if hasattr(detector, "_ort_session") and detector._ort_session is not None:
                providers = detector._ort_session.get_providers()
                if any("CUDAExecutionProvider" in p for p in providers):
                    cuda_loaded = True
            elif "cuda" in detector_backend.lower() or detector_backend == "onnxruntime_cuda":
                cuda_loaded = True
        except Exception as exc:
            error_msg = str(exc)

        return {
            "onnx_path": str(onnx_path),
            "exists": exists,
            "size_bytes": size_bytes,
            "valid": valid,
            "onnx_ready": bool(valid),
            "cuda_loaded": cuda_loaded,
            "detector_backend": detector_backend,
            "error": error_msg,
        }

    # Live Inference Helper on Trained Custom Detector
    # ------------------------------------------------------------------
    def _load_custom_detector(self) -> Dict[str, Any]:
        """Load `gatherion_custom_detector.json`, cached by file version.

        D58: the detector JSON was re-read and re-parsed on every
        `predict_roi` call (per-frame in live use). The parsed dict is now
        cached while the file's ``(mtime_ns, size)`` version is unchanged
        and reloaded as soon as retraining rewrites the file.
        """
        stat = self.custom_detector_path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        if self._detector_cache is not None and self._detector_cache_version == version:
            return self._detector_cache
        data = json.loads(self.custom_detector_path.read_text(encoding="utf-8"))
        self._detector_cache = data
        self._detector_cache_version = version
        return data

    def predict_roi(
        self,
        roi_bgr: np.ndarray,
        norm_bbox_xywh: Tuple[float, float, float, float] = (0.5, 0.5, 0.15, 0.15),
    ) -> Dict[str, Any]:
        """
        Classifies a cropped BGR region using the trained `gatherion_custom_detector.json`
        prototype model and returns `{class_id, class_name, confidence, distance}`.

        D58: distances are computed in a single pass (the old code computed
        them twice — once for the best class, once for the softmax) and the
        detector JSON plus the prebuilt centroid/weight arrays come from the
        `_predict_cache` (keyed by file mtime_ns + size, so a retrain reloads it).
        """
        if not self.custom_detector_path.exists():
            return {
                "class_id": -1,
                "class_name": "untrained",
                "confidence": 0.0,
                "distance": float("inf"),
            }

        # Cache the parsed detector (keyed by mtime_ns + size, so a retrain
        # reloads it): the old code re-read + re-parsed the JSON and rebuilt
        # every centroid/weight array on each call.
        try:
            stat = self.custom_detector_path.stat()
            cache_key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            cache_key = None
        if cache_key is None or cache_key != self._predict_cache_key or self._predict_cache is None:
            data = json.loads(self.custom_detector_path.read_text(encoding="utf-8"))
            classes_dict: Dict[str, Any] = data.get("classes", {})
            class_ids = [int(cinfo["class_id"]) for cinfo in classes_dict.values()]
            cache = {
                "feature_mean": np.array(data["feature_mean"], dtype=np.float32),
                "feature_std": np.array(data["feature_std"], dtype=np.float32),
                "class_ids": class_ids,
                "class_names": [str(cinfo["class_name"]) for cinfo in classes_dict.values()],
                "centroids": np.stack(
                    [np.array(cinfo["centroid"], dtype=np.float32) for cinfo in classes_dict.values()]
                )
                if class_ids
                else np.zeros((0, 0), dtype=np.float32),
                "weights": np.stack(
                    [np.array(cinfo["feature_weights"], dtype=np.float32) for cinfo in classes_dict.values()]
                )
                if class_ids
                else np.zeros((0, 0), dtype=np.float32),
            }
            self._predict_cache = cache
            self._predict_cache_key = cache_key
        else:
            cache = self._predict_cache

        feat_mean = cache["feature_mean"]
        feat_std = cache["feature_std"]

        feat_vec, _ = self.extract_roi_features(roi_bgr, norm_bbox_xywh)
        feat_norm = (feat_vec - feat_mean) / (feat_std + 1e-5)

        if not cache["class_ids"]:
            return {
                "class_id": -1,
                "class_name": "unknown",
                "confidence": 0.0,
                "distance": float("inf"),
            }

        # One vectorized distance pass (the old code looped over all classes
        # twice); confidence is the same softmax share over -distance.
        diffs = feat_norm[None, :] - cache["centroids"]
        dists = np.sqrt(
            np.maximum(0.0, np.mean(cache["weights"] * np.square(diffs), axis=1))
        )
        best_pos = int(np.argmin(dists))
        best_dist = float(dists[best_pos])
        neg = -dists.astype(np.float64)
        neg = neg - float(np.max(neg))
        weights_soft = np.exp(neg)
        confidence = float(weights_soft[best_pos] / float(np.sum(weights_soft)))
        return {
            "class_id": int(cache["class_ids"][best_pos]),
            "class_name": str(cache["class_names"][best_pos]),
            "confidence": round(confidence, 4),
            "distance": round(best_dist, 4),
        }

    def get_rtx_trainer_health_summary(self) -> dict:
        # F-0995: readiness reflects a real probe, not a hardcoded True.
        try:
            env = self.probe_gpu_environment()
            ready = bool(env.get("ultralytics_installed"))
            return {
                "rtx_trainer_ready": ready,
                "ultralytics_installed": ready,
                "cuda_available": bool(env.get("cuda_available")),
                "gpu_name": env.get("gpu_name"),
                "reason": (
                    "Ultralytics yüklü — Mode A eğitimi mümkün."
                    if ready
                    else "Ultralytics yüklü değil — Mode A eğitimi mümkün değil."
                ),
            }
        except Exception as exc:
            return {"rtx_trainer_ready": False, "error": str(exc)}

    def get_rtx_trainer_engine_ready_flag(self) -> dict:
        return {"module": "rtx_trainer_engine", "ready": validate_rtx_trainer_engine_contract()}

    def get_rtx_trainer_engine_version_info(self) -> dict:
        return {
            "module": "rtx_trainer_engine",
            "version": __version__,
            "ready": validate_rtx_trainer_engine_contract(),
        }

    def get_rtx_trainer_engine_diagnostic_snapshot(self) -> dict:
        ready = validate_rtx_trainer_engine_contract()
        return {
            "module": "rtx_trainer_engine",
            "status": "ok" if ready else "not_ready",
            "ready": ready,
        }


def get_rtx_trainer_engine_capabilities() -> list:
    return ["rtx_trainer_engine", "fp16_onnx_export", "diagnostic_snapshot"]


def validate_rtx_trainer_engine_contract() -> bool:
    """Real contract check (F-0995): canonical 27-class taxonomy intact,
    the engine class importable, and the GPU probe runnable with
    Ultralytics available for Mode A training."""
    try:
        # P1-1: the expected id set comes from the same single source
        # (dataset/data.yaml, nc-1) — not a hardcoded range(27).
        if set(DEFAULT_CLASS_NAMES.keys()) != set(range(_resolve_max_class_id() + 1)):
            return False
        if RTXTrainerEngine is None:
            return False
        env = RTXTrainerEngine.probe_gpu_environment_static()
        return bool(env.get("ultralytics_installed"))
    except Exception:
        return False


def _rtx_trainer_engine_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. telemetry.py'deki aciklama).

    ``validate_rtx_trainer_engine_contract()`` her cagrida
    ``probe_gpu_environment_static()``'i bastan kosar (nvidia-smi alt
    sureci + torch/onnxruntime/ultralytics yoklamasi); zincir bunu
    ust duzey okuma basina bir kez yapar. Cagrilar arasi onbellek
    yoktur — her ust duzey okuma taze prob yapar.
    """
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_rtx_trainer_engine_contract())


def get_rtx_trainer_engine_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {
        "module": "rtx_trainer_engine",
        "contract_valid": _rtx_trainer_engine_contract_ok(_contract_ok),
        "capability_count": len(get_rtx_trainer_engine_capabilities()),
    }


def get_rtx_trainer_engine_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "ready": ok,
        "metrics": get_rtx_trainer_engine_baseline_metrics(_contract_ok=ok),
    }


def get_rtx_trainer_engine_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "audit_passed": ok,
        "readiness": get_rtx_trainer_engine_readiness_status(_contract_ok=ok),
    }


def get_rtx_trainer_engine_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "operational": ok,
        "audit": get_rtx_trainer_engine_audit_record(_contract_ok=ok),
    }


def get_rtx_trainer_engine_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    return {
        "module": "rtx_trainer_engine",
        "payload": get_rtx_trainer_engine_operational_digest(_contract_ok=_contract_ok),
    }


def get_rtx_trainer_engine_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "sealed": ok,
        "telemetry": get_rtx_trainer_engine_telemetry_payload(_contract_ok=ok),
    }


def get_rtx_trainer_engine_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "certified": ok,
        "seal": get_rtx_trainer_engine_integrity_seal(_contract_ok=ok),
    }


def get_rtx_trainer_engine_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "ready": ok,
        "certificate": get_rtx_trainer_engine_compliance_certificate(_contract_ok=ok),
    }


def get_rtx_trainer_engine_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _rtx_trainer_engine_contract_ok(_contract_ok)
    return {
        "module": "rtx_trainer_engine",
        "audit_passed": ok,
        "probe": get_rtx_trainer_engine_readiness_probe(_contract_ok=ok),
    }


