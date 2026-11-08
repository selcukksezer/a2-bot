# -*- coding: utf-8 -*-
"""Delete route nodes the gatherion detector / multi-signal gate calls wrong.

A node is removed when any of these fire:

- gatherion hard-wrong class (F Gather, E, monster, UI, …) at the tier
  confidence floor (default balanced ≈ 0.13)
- parchment scroll thumbnail
- monster-dot circularity on the patch
- resource_type vs model contradiction
- newer node within SAME_POINT_PX (18) of an older kept node

Class 9 (minimap sprout) alone does not delete: minimap patches are sprout
icons by design. Harvest count is ignored. Inference is one thumbnail at a
time on CPU only.
"""

from __future__ import annotations

import ast
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pathlib import Path
import shutil

import cv2
import numpy as np

from core.node_quality_gate import (
    CLASS_LABELS_TR,
    CONF_TIERS,
    DEFAULT_TIER,
    KEEP_CLASS_IDS,
    MODEL_CONF_MIN,
    SAME_POINT_PX,
    WRONG_CLASS_IDS,
    conf_min_for_tier,
    evaluate_node_candidate,
    get_default_exemplar_memory,
    is_fake_gray_rock_or_ground,
    reason_label_tr,
    wrong_class_id,
)
from core.vision_filters import is_parchment_scroll_patch

try:
    from dataset_factory.exemplar_memory import (
        ExemplarMemory,
        get_exemplar_memory,
    )
except Exception:
    ExemplarMemory = None  # type: ignore
    get_exemplar_memory = None  # type: ignore

logger = logging.getLogger(__name__)

INPUT_SIZE = 640

_YOLO11_GATHERION = os.path.join("models", "yolo11n_gatherion.onnx")
_YOLOV8_GATHERION = os.path.join("models", "yolov8n_gatherion.onnx")
_COCO_BASENAMES = frozenset({"yolo11n.onnx", "rtdetr-l.onnx"})
_CPU_PROVIDER = "CPUExecutionProvider"

_REASON_ORDER = (
    "model",
    "contradiction",
    "monster",
    "scroll",
    "feather",
    "negative_exemplar",
    "terrain",
    "empty_terrain",
    "duplicate",
)

_LAST_SCANNED_REGION: str = ""
_LAST_SCANNED_IMAGES: Dict[int, Optional[np.ndarray]] = {}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _field(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def read_dataset_class_names(path: str) -> List[str]:
    """Class names from ``dataset/data.yaml``, in id order."""
    names: Dict[int, str] = {}
    in_names = False
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.split("#", 1)[0].rstrip()
            if not stripped.strip():
                continue
            if not in_names:
                if stripped.strip() == "names:" or stripped.strip().startswith("names:"):
                    in_names = True
                continue
            if not raw.startswith((" ", "\t")):
                break
            body = stripped.strip()
            if ":" not in body:
                continue
            key_text, value_text = body.split(":", 1)
            names[int(key_text.strip())] = value_text.strip().strip("'\"")
    if not names:
        raise ValueError(f"no class names in {path}")
    count = max(names) + 1
    if sorted(names) != list(range(count)):
        raise ValueError(f"class ids in {path} are not contiguous")
    return [names[index] for index in range(count)]


def names_match_dataset(model_names: Sequence[str], dataset_names: Sequence[str]) -> bool:
    """True when the export has every dataset class, in order."""
    return list(model_names) == list(dataset_names) and len(dataset_names) > 0


def names_are_gatherion_prefix(
    model_names: Sequence[str], dataset_names: Sequence[str]
) -> bool:
    """True when the export's names are a leading slice of the dataset names."""
    names = list(model_names)
    if not names or len(names) > len(dataset_names):
        return False
    if names[0] != dataset_names[0]:
        return False
    return names == list(dataset_names[: len(names)])


def _refuse_coco_path(path: str) -> None:
    base = os.path.basename(path).lower()
    if base in _COCO_BASENAMES:
        raise RuntimeError(f"COCO modeli taranamaz: {base}")


def _parse_metadata_names(session: Any) -> List[str]:
    meta = session.get_modelmeta().custom_metadata_map or {}
    raw = str(meta.get("names", "") or "").strip()
    if not raw:
        return []
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, dict):
        pairs = {int(key): str(value) for key, value in parsed.items()}
        count = max(pairs) + 1 if pairs else 0
        if sorted(pairs) != list(range(count)):
            return []
        return [pairs[index] for index in range(count)]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []


def _output_class_count(session: Any) -> int:
    shape = session.get_outputs()[0].shape
    if len(shape) < 2:
        raise RuntimeError("onnx çıktı şekli sınıf sayısı vermiyor")
    channels = int(shape[1])
    if channels <= 4:
        raise RuntimeError(f"onnx çıktı kanalı yetersiz: {shape}")
    return channels - 4


class GatherionOnnxDetector:
    """One-thumbnail YOLOv8/YOLO11 detect head. CPU only."""

    def __init__(self, model_path: str) -> None:
        _refuse_coco_path(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        import onnxruntime as ort

        with open(model_path, "rb") as handle:
            model_bytes = handle.read()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=[_CPU_PROVIDER],
        )
        providers = list(session.get_providers())
        if providers != [_CPU_PROVIDER]:
            raise RuntimeError(f"CPU zorunlu, sağlayıcılar: {providers}")
        class_names = _parse_metadata_names(session)
        num_classes = _output_class_count(session)
        input_shape = session.get_inputs()[0].shape
        try:
            input_size = int(input_shape[2])
        except (TypeError, ValueError, IndexError):
            input_size = INPUT_SIZE
        self.model_path = os.path.abspath(model_path)
        self.device = _CPU_PROVIDER
        self.providers = providers
        self.class_names = class_names
        self.num_classes = int(num_classes)
        self.names_consistent = len(class_names) == self.num_classes
        self.input_size = input_size if input_size > 0 else INPUT_SIZE
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def matches_dataset(self, dataset_names: Sequence[str]) -> bool:
        """Full gatherion export: class count and names equal ``data.yaml``."""
        return self.names_consistent and names_match_dataset(self.class_names, dataset_names)

    def is_gatherion_subset(self, dataset_names: Sequence[str]) -> bool:
        """Prefix gatherion export (e.g. 16-class) or a full match."""
        if not self.names_consistent:
            return False
        return names_are_gatherion_prefix(self.class_names, dataset_names)

    def top_class(self, image: np.ndarray) -> Optional[Tuple[int, float]]:
        """Best center-weighted class on this thumbnail, or None when empty."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        blob = _thumbnail_blob(image, self.input_size)
        raw = self._session.run(None, {self._input_name: blob})[0]
        return _top_class_from_output(raw, self.num_classes, input_size=self.input_size)


def load_gatherion_detector(root: Optional[str] = None) -> GatherionOnnxDetector:
    """Prefer the longest gatherion-compatible export available on CPU.

    Full ``data.yaml`` match wins. Otherwise the longest gatherion prefix
    among ``yolo11n_gatherion.onnx`` and ``yolov8n_gatherion.onnx`` is used
    (16-class yolo11 beats a shorter v8). COCO exports are never opened.
    """
    base = root or _project_root()
    dataset_names = read_dataset_class_names(os.path.join(base, "dataset", "data.yaml"))
    candidates: List[GatherionOnnxDetector] = []

    for rel in (_YOLO11_GATHERION, _YOLOV8_GATHERION):
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            continue
        try:
            candidate = GatherionOnnxDetector(path)
        except Exception as exc:
            logger.info("%s kullanılmadı: %s", os.path.basename(path), exc)
            continue
        if candidate.matches_dataset(dataset_names):
            logger.info(
                "YZ taraması modeli: %s nc=%s %s",
                candidate.model_path,
                candidate.num_classes,
                candidate.device,
            )
            return candidate
        if candidate.is_gatherion_subset(dataset_names):
            candidates.append(candidate)
        else:
            logger.info(
                "%s gatherion sınıf adları taşımıyor (nc=%s)",
                os.path.basename(path),
                candidate.num_classes,
            )

    if not candidates:
        raise FileNotFoundError(
            f"gatherion ONNX bulunamadı veya uyumsuz: {_YOLO11_GATHERION}, {_YOLOV8_GATHERION}"
        )

    candidates.sort(
        key=lambda det: (det.num_classes, 1 if "yolo11" in det.model_path.lower() else 0),
        reverse=True,
    )
    detector = candidates[0]
    logger.info(
        "YZ taraması modeli (en uzun gatherion öneki): %s nc=%s %s (data.yaml=%s)",
        detector.model_path,
        detector.num_classes,
        detector.device,
        len(dataset_names),
    )
    return detector


def _thumbnail_blob(image: np.ndarray, input_size: int) -> np.ndarray:
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame = np.ascontiguousarray(frame[:, :, :3])
    padded = _letterbox(frame, int(input_size))
    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 255.0,
        size=(int(input_size), int(input_size)),
        swapRB=True,
        crop=False,
    )
    return np.ascontiguousarray(blob, dtype=np.float32)


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / float(width), size / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (size - new_w) / 2.0
    dh = (size - new_h) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[0] != size or padded.shape[1] != size:
        padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)
    return padded


def _top_class_from_output(
    raw: np.ndarray,
    num_classes: int,
    input_size: int = INPUT_SIZE,
) -> Optional[Tuple[int, float]]:
    """Pick the highest class score, preferring anchors near the letterbox center."""
    preds = np.squeeze(raw)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    if preds.ndim != 2 or preds.size == 0:
        return None
    class_axis = 4 + int(num_classes)
    if preds.shape[0] == class_axis and preds.shape[1] != class_axis:
        preds = preds.T
    elif preds.shape[1] != class_axis and preds.shape[0] < preds.shape[1] and preds.shape[0] > 4:
        preds = preds.T
    if preds.shape[1] < 5:
        return None
    class_scores = np.asarray(preds[:, 4 : 4 + int(num_classes)], dtype=np.float32)
    if class_scores.size == 0 or not np.isfinite(class_scores).any():
        return None

    best_per_anchor = np.nanmax(class_scores, axis=1)
    best_cls = np.nanargmax(class_scores, axis=1).astype(np.int32)
    boxes = np.asarray(preds[:, :4], dtype=np.float32)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    center = float(input_size) * 0.5
    dist = np.hypot(cx - center, cy - center)
    weight = np.clip(1.0 - dist / (float(input_size) * 0.55), 0.20, 1.0)
    weighted = best_per_anchor * weight
    if not np.isfinite(weighted).any():
        return None
    idx = int(np.nanargmax(weighted))
    confidence = float(best_per_anchor[idx])
    class_id = int(best_cls[idx])
    if not math.isfinite(confidence):
        return None
    return class_id, confidence


def _detector_num_classes(detector: Any) -> Optional[int]:
    raw = getattr(detector, "num_classes", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _top_from_detector(detector: Any, image: np.ndarray) -> Optional[Tuple[int, float]]:
    try:
        if hasattr(detector, "top_class"):
            verdict = detector.top_class(image)
        else:
            verdict = detector(image)
    except Exception as exc:
        logger.debug("Küçük resim çıkarımı atlandı: %s", exc)
        return None
    if verdict is None:
        return None
    if isinstance(verdict, dict):
        class_id = verdict.get("class_id", verdict.get("class"))
        confidence = verdict.get("confidence", verdict.get("conf"))
    elif isinstance(verdict, (tuple, list)) and len(verdict) >= 2:
        class_id, confidence = verdict[0], verdict[1]
    else:
        return None
    try:
        return int(class_id), float(confidence)
    except (TypeError, ValueError):
        return None


def load_patch_file(path: str) -> Optional[np.ndarray]:
    """Read a patch from disk. Unicode paths are supported."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    except Exception:
        image = None
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    return image


def load_node_patch(node: Any, region: str = "", base_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Load the saved thumbnail for ``node`` with several path fallbacks."""
    raw_path = _field(node, "patch_path", "") or _field(node, "patch_file", "")
    node_id = int(_field(node, "id", 0) or 0)
    node_region = str(_field(node, "region", "") or region or "")
    root = base_dir or _project_root()
    candidates: List[str] = []
    if raw_path:
        candidates.append(str(raw_path))
        if not os.path.isabs(str(raw_path)):
            candidates.append(os.path.basename(str(raw_path)))
            candidates.append(os.path.join(root, str(raw_path)))
            if node_region:
                candidates.append(
                    os.path.join("routes", node_region, "patches", os.path.basename(str(raw_path)))
                )
                candidates.append(
                    os.path.join(root, "routes", node_region, "patches", os.path.basename(str(raw_path)))
                )
                candidates.append(
                    os.path.join("routes", node_region, os.path.basename(str(raw_path)))
                )
                candidates.append(
                    os.path.join(root, "routes", node_region, os.path.basename(str(raw_path)))
                )
    if node_region and node_id:
        candidates.append(
            os.path.join("routes", node_region, "patches", f"node_{node_id}.png")
        )
        candidates.append(
            os.path.join(root, "routes", node_region, "patches", f"node_{node_id}.png")
        )
        candidates.append(
            os.path.join("routes", node_region, f"node_{node_id:02d}.png")
        )
        candidates.append(
            os.path.join(root, "routes", node_region, f"node_{node_id:02d}.png")
        )
        candidates.append(
            os.path.join("routes", node_region, f"node_{node_id}.png")
        )
        candidates.append(
            os.path.join(root, "routes", node_region, f"node_{node_id}.png")
        )
    for path in candidates:
        image = load_patch_file(path)
        if image is not None:
            return image
    return None


def _coords(node: Any) -> Tuple[float, float]:
    try:
        x = float(_field(node, "map_x", 0) or 0)
        y = float(_field(node, "map_y", 0) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return x, y


def _age_key(node: Any, index: int) -> Tuple[int, int, int]:
    try:
        seq = int(_field(node, "seq_index", 0) or 0)
    except (TypeError, ValueError):
        seq = 0
    try:
        node_id = int(_field(node, "id", 0) or 0)
    except (TypeError, ValueError):
        node_id = 0
    return seq, node_id, index


def _node_id(node: Any, fallback: int) -> int:
    try:
        return int(_field(node, "id", fallback) or fallback)
    except (TypeError, ValueError):
        return fallback


class WrongNode:
    """One node the scan would delete, with the tests that failed."""

    def __init__(
        self,
        node_id: int,
        reasons: Sequence[str],
        class_id: Optional[int] = None,
        confidence: Optional[float] = None,
        is_wrong: bool = True,
    ) -> None:
        self.node_id = int(node_id)
        self.reasons = list(reasons)
        self.class_id = None if class_id is None else int(class_id)
        self.confidence = None if confidence is None else float(confidence)
        self.is_wrong = bool(is_wrong)

    @property
    def primary_reason(self) -> str:
        keys = [reason.split(":", 1)[0] for reason in self.reasons]
        for key in _REASON_ORDER:
            if key in keys:
                return key
        return keys[0] if keys else ""


def classify_wrong_nodes(
    nodes: Sequence[Any],
    images_by_id: Optional[Dict[int, Optional[np.ndarray]]] = None,
    detector: Any = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    conf_min: float = MODEL_CONF_MIN,
    tier: str = "",
    exemplar_memory: Any = None,
    check_exemplars: bool = True,
) -> List[WrongNode]:
    """Return nodes the multi-signal gate or an 18 px duplicate rejects.

    ``detector`` is a ``top_class(image) -> (class_id, confidence) | None``
    callable, or any object with that method. ``num_classes`` on the detector
    limits which ids are scored. Missing images skip model/CV image tests but
    can still be duplicates. Harvest count is ignored.
    """
    global _LAST_SCANNED_REGION, _LAST_SCANNED_IMAGES
    _LAST_SCANNED_IMAGES = dict(images_by_id or {})
    for n in nodes:
        r = _field(n, "region", "")
        if r:
            _LAST_SCANNED_REGION = str(r)
            break

    floor = conf_min_for_tier(tier) if tier else float(conf_min)
    indexed = list(enumerate(nodes))
    ordered = sorted(indexed, key=lambda pair: _age_key(pair[1], pair[0]))
    images = images_by_id or {}
    kept: List[Any] = []
    dropped: List[WrongNode] = []
    total = len(ordered)
    num_classes = _detector_num_classes(detector) if detector is not None else None

    for step, (_, node) in enumerate(ordered):
        if on_progress is not None:
            on_progress(step, total)
        node_id = _node_id(node, 0)
        image = images.get(node_id)
        source_type = str(_field(node, "source_type", "") or "")
        resource_type = str(_field(node, "resource_type", "") or "")

        model_verdict = None
        if image is not None and detector is not None:
            model_verdict = _top_from_detector(detector, image)

        gate = evaluate_node_candidate(
            image,
            source_type=source_type,
            resource_type=resource_type,
            detector=None,
            num_classes=num_classes,
            conf_min=floor,
            check_feather=False,
            check_monster=True,
            model_verdict=model_verdict,
            exemplar_memory=exemplar_memory,
            check_exemplars=check_exemplars,
        )
        reasons = list(gate.reasons)
        class_id = gate.class_id
        confidence = gate.confidence

        x, y = _coords(node)
        for older in kept:
            older_x, older_y = _coords(older)
            dist = math.hypot(x - older_x, y - older_y)
            if dist <= SAME_POINT_PX:
                reasons.append("duplicate")
                break

        reasons = list(dict.fromkeys(reasons))
        if reasons:
            dropped.append(
                WrongNode(
                    node_id,
                    reasons,
                    class_id=class_id,
                    confidence=confidence,
                    is_wrong=True,
                )
            )
        else:
            kept.append(node)
    if on_progress is not None:
        on_progress(total, total)
    return dropped


def scan_confidence_summary(
    nodes: Sequence[Any],
    images_by_id: Optional[Dict[int, Optional[np.ndarray]]] = None,
    detector: Any = None,
) -> Dict[str, Any]:
    """Conf histogram helpers for UI status (does not delete)."""
    images = images_by_id or {}
    confs: List[float] = []
    missing = 0
    for node in nodes:
        node_id = _node_id(node, 0)
        image = images.get(node_id)
        if image is None:
            missing += 1
            continue
        if detector is None:
            continue
        verdict = _top_from_detector(detector, image)
        if verdict is None:
            continue
        confs.append(float(verdict[1]))
    confs_sorted = sorted(confs)
    return {
        "scored": len(confs),
        "missing_patches": int(missing),
        "max_conf": max(confs) if confs else 0.0,
        "p50_conf": confs_sorted[len(confs_sorted) // 2] if confs_sorted else 0.0,
        "ge_balanced": sum(1 for c in confs if c >= CONF_TIERS["balanced"]),
        "ge_aggressive": sum(1 for c in confs if c >= CONF_TIERS["aggressive"]),
    }


def backup_region_data(
    region: str,
    base_dir: Optional[str] = None,
) -> Optional[str]:
    """Create safety backup `region_data.backup.json` and timestamped copy for a region."""
    if not region:
        return None
    root = base_dir or os.path.join(_project_root(), "routes")
    reg_dir = os.path.join(root, region)
    src = os.path.join(reg_dir, "region_data.json")
    dst = os.path.join(reg_dir, "region_data.backup.json")
    if not os.path.isfile(src):
        src_rel = os.path.join("routes", region, "region_data.json")
        if os.path.isfile(src_rel):
            src = src_rel
            dst = os.path.join("routes", region, "region_data.backup.json")
        else:
            return None
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

        # Also create timestamped backup so multiple safety snapshots are preserved
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_dst = os.path.join(os.path.dirname(dst), f"region_data.backup_{ts}.json")
        try:
            shutil.copyfile(src, ts_dst)
        except Exception:
            pass

        logger.info("Safety backup created on region %s: %s (and %s)", region, dst, ts_dst)
        return dst
    except Exception as exc:
        logger.warning("Failed to create safety backup for %s: %s", region, exc)
        return None


def scan_would_exceed_half(
    removal_count: int,
    total: int,
    region: Optional[str] = None,
    base_dir: Optional[str] = None,
    auto_backup: bool = True,
) -> bool:
    """True when the scan would delete more than half the nodes.

    If purging would delete > 50% of the nodes in a region and auto_backup
    is True, automatically creates `region_data.backup.json` before applying
    the purge when triggered by the UI button.
    """
    if total <= 0:
        return False
    exceeds = int(removal_count) * 2 > int(total)
    if exceeds and auto_backup:
        target_region = str(region or _LAST_SCANNED_REGION or "").strip()
        if target_region:
            backup_region_data(target_region, base_dir=base_dir)
    return exceeds


def save_wrong_nodes_as_negative_exemplars(
    wrong: Sequence[Any],
    images_by_id: Optional[Dict[int, Optional[np.ndarray]]] = None,
    exemplar_memory: Any = None,
    exemplar_path: Optional[Union[str, Path]] = None,
    source: str = "wrong_node_purge",
    region: str = "",
) -> int:
    """Active Learning Feedback Loop:
    When wrong nodes are purged or deleted, automatically save them as negative
    exemplars into dataset/learned_exemplars.json so the AI never marks them again.

    Accepts:
    - Sequence[WrongNode]
    - Sequence[int] (node IDs)
    - Sequence[dict] / objects (nodes with id, resource_type, etc.)
    """
    if not wrong:
        return 0
    mem = exemplar_memory
    if mem is None and get_exemplar_memory is not None:
        try:
            mem = get_exemplar_memory(json_path=exemplar_path)
        except Exception as exc:
            logger.warning("Failed to acquire exemplar memory for active learning: %s", exc)
            return 0

    if mem is None or not hasattr(mem, "add_negative_exemplar"):
        return 0

    target_region = str(region or _LAST_SCANNED_REGION or "")
    images = images_by_id if images_by_id is not None else _LAST_SCANNED_IMAGES
    saved = 0
    for item in wrong:
        node_id = 0
        class_id = None
        primary_reason = ""

        if isinstance(item, WrongNode):
            node_id = item.node_id
            class_id = item.class_id
            primary_reason = item.primary_reason
        elif isinstance(item, int):
            node_id = item
        elif isinstance(item, dict):
            node_id = int(item.get("id", 0))
            class_id = item.get("class_id")
            primary_reason = str(item.get("reason", item.get("primary_reason", "")))
        else:
            node_id = int(getattr(item, "id", getattr(item, "node_id", 0)) or 0)
            class_id = getattr(item, "class_id", None)
            primary_reason = str(getattr(item, "primary_reason", getattr(item, "reason", "")))

        img = images.get(node_id)
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            # Fallback: attempt to load patch from disk
            if target_region and node_id:
                img = load_node_patch({"id": node_id, "region": target_region}, region=target_region)
            if img is None and node_id:
                for reg_try in (_LAST_SCANNED_REGION, "Altgard", "Verteron", "Morheim"):
                    if reg_try:
                        img = load_node_patch({"id": node_id, "region": reg_try}, region=reg_try)
                        if img is not None:
                            break

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            continue

        cid = class_id if class_id is not None else 15
        if class_id is not None and int(class_id) in CLASS_LABELS_TR:
            cname = CLASS_LABELS_TR[int(class_id)]
        else:
            primary = primary_reason
            if primary in ("terrain", "empty_terrain"):
                cid = 15
                cname = "path_ground"
            elif primary == "monster":
                cid = 11
                cname = "minimap_monster_negative"
            elif primary == "scroll":
                cid = 13
                cname = "skill_ui_prompt"
            elif primary == "feather":
                cid = 13
                cname = "feather_negative"
            else:
                cid = 15
                cname = f"wrong_node_{primary or 'purge'}"

        try:
            ok = mem.add_negative_exemplar(
                img,
                class_id=int(cid),
                class_name=str(cname),
                source=source,
            )
            if ok:
                saved += 1
        except Exception as exc:
            logger.debug("Failed to save negative exemplar for node %s: %s", node_id, exc)

    logger.info(
        "Active learning feedback: saved %d wrong nodes as negative exemplars into ExemplarMemory",
        saved,
    )
    return saved


def register_deleted_nodes_as_negative_exemplars(
    nodes_or_ids: Sequence[Any],
    region: str = "",
    images_by_id: Optional[Dict[int, Optional[np.ndarray]]] = None,
    exemplar_memory: Any = None,
    exemplar_path: Optional[Union[str, Path]] = None,
    source: str = "node_deletion",
) -> int:
    """Public helper to register any list of deleted nodes / IDs as negative exemplars in dataset/learned_exemplars.json."""
    return save_wrong_nodes_as_negative_exemplars(
        nodes_or_ids,
        images_by_id=images_by_id,
        exemplar_memory=exemplar_memory,
        exemplar_path=exemplar_path,
        source=source,
        region=region,
    )


def purge_wrong_nodes(
    region: str,
    wrong_nodes: Sequence[WrongNode],
    route_manager: Any = None,
    images_by_id: Optional[Dict[int, Optional[np.ndarray]]] = None,
    exemplar_memory: Any = None,
    total_nodes_count: Optional[int] = None,
    base_dir: Optional[str] = None,
    auto_backup: bool = True,
    save_negative_exemplars: bool = True,
) -> Tuple[bool, str, int]:
    """Purge wrong nodes with automatic safety backup and active learning loop.

    1. If purging would delete > 50% of nodes in a region (scan_would_exceed_half),
       creates automatic backup `region_data.backup.json` before applying the purge.
    2. Active learning: saves purged wrong nodes into dataset/learned_exemplars.json
       so the AI never marks them again.
    3. Applies node deletion in RouteManager or region_data.json.

    Returns: (success: bool, status_message: str, count: int)
    """
    total = total_nodes_count
    if total is None and route_manager is not None and hasattr(route_manager, "get_region_nodes"):
        try:
            total = len(route_manager.get_region_nodes(region) or [])
        except Exception:
            total = len(wrong_nodes)
    if total is None:
        total = len(wrong_nodes)

    backup_path = ""
    exceeds_half = scan_would_exceed_half(
        len(wrong_nodes),
        total,
        region=region,
        base_dir=base_dir,
        auto_backup=auto_backup,
    )
    if exceeds_half:
        backup_path = backup_region_data(region, base_dir=base_dir) or ""
    elif auto_backup and route_manager is not None and hasattr(route_manager, "backup_region_file"):
        try:
            backup_path = route_manager.backup_region_file(region)
        except Exception:
            backup_path = ""

    if save_negative_exemplars and wrong_nodes:
        save_wrong_nodes_as_negative_exemplars(
            wrong_nodes,
            images_by_id=images_by_id,
            exemplar_memory=exemplar_memory,
            region=region,
        )

    ids_to_delete = [w.node_id for w in wrong_nodes]
    if route_manager is not None and hasattr(route_manager, "delete_nodes"):
        ok = route_manager.delete_nodes(ids_to_delete, region=region)
    else:
        ok = True

    msg = format_removed_status_tr(
        wrong_nodes,
        backup_path=backup_path,
    )
    return ok, msg, len(wrong_nodes)


def class_histogram(wrong: Sequence[WrongNode]) -> Dict[int, int]:
    """Wrong-class counts among nodes the model itself rejected."""
    counts: Dict[int, int] = {}
    for item in wrong:
        if item.class_id is None or int(item.class_id) not in WRONG_CLASS_IDS:
            continue
        cid = int(item.class_id)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _reason_counts(wrong: Sequence[WrongNode]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for item in wrong:
        if item.class_id is not None and int(item.class_id) in WRONG_CLASS_IDS:
            label = CLASS_LABELS_TR.get(int(item.class_id), str(item.class_id))
        else:
            label = reason_label_tr(item.primary_reason, item.class_id)
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked


def format_removed_status_tr(
    wrong: Sequence[WrongNode],
    backup_path: str = "",
) -> str:
    """Turkish status: how many were removed, then the top wrong classes."""
    ranked = _reason_counts(wrong)[:4]
    parts = ", ".join(f"{name} {count}" for name, count in ranked)
    text = f"YZ taraması {len(list(wrong))} nokta sildi. Yanlış sınıflar: {parts}."
    if backup_path:
        text += f" Yedek: {backup_path}"
    return text


def format_blocked_status_tr(wrong: Sequence[WrongNode], total: int) -> str:
    """Turkish status when more than half would be removed and nothing is written."""
    ranked = _reason_counts(wrong)
    parts = ", ".join(f"{name} {count}" for name, count in ranked)
    return (
        f"YZ taraması {len(list(wrong))}/{int(total)} düğümü silerdi "
        f"(yarısından fazla). Dosya yazılmadı. Sınıflar: {parts}."
    )


def format_scan_meta_tr(
    detector: Any,
    summary: Optional[Dict[str, Any]] = None,
    tier: str = DEFAULT_TIER,
) -> str:
    """Short Turkish meta line: model path, nc, missing patches, conf peek."""
    path = str(getattr(detector, "model_path", "") or "")
    base = os.path.basename(path) if path else "?"
    nc = getattr(detector, "num_classes", "?")
    floor = conf_min_for_tier(tier)
    text = f"Model: {base} nc={nc}, eşik={floor:.2f} ({tier})"
    if summary:
        text += (
            f", eksik patch={summary.get('missing_patches', 0)}, "
            f"max conf={float(summary.get('max_conf', 0.0)):.2f}"
        )
    return text


def verify_and_blacklist_node(
    node: Any,
    region: str = "",
    reason: str = "3x ardışık toplama başarısızlığı (YZ yanlış işaretleme doğrulaması)",
    detector: Any = None,
    exemplar_memory: Any = None,
    blacklist_mgr: Any = None,
    base_dir: Optional[str] = None,
) -> Tuple[bool, str, Any]:
    """
    Kullanıcı İsteği:
    3 kez üst üste toplanacak bir şey bulunamayan düğümü yapay zeka (YZ) ile inceler,
    aktif öğrenme negatif hafızasına (learned_exemplars.json) kaydeder ve koordinatlarını
    kara listeye (Blacklist) ekler. Bir daha kesinlikle kaydedilmez.

    Döner:
        (is_verified: bool, ai_verdict: str, blacklist_entry: Any)
    """
    from core.blacklist_manager import get_blacklist_manager

    node_id = int(_field(node, "id", 0) or 0)
    target_region = str(region or _field(node, "region", "") or _LAST_SCANNED_REGION or "")
    map_x = int(_field(node, "map_x", 0) or 0)
    map_y = int(_field(node, "map_y", 0) or 0)
    norm_x = float(_field(node, "norm_x", 0.0) or 0.0)
    norm_y = float(_field(node, "norm_y", 0.0) or 0.0)
    source_type = str(_field(node, "source_type", "") or "")
    resource_type = str(_field(node, "resource_type", "") or "")

    # 1. Düğümün görsel yamasını diskten veya önbellekten yükle
    image = load_node_patch(node, region=target_region, base_dir=base_dir)

    # 2. YZ Dedektörünü temin et (eğer verilmemişse yüklemeyi dene)
    det = detector
    if det is None:
        try:
            det = load_gatherion_detector(root=base_dir)
        except Exception as exc:
            logger.debug("Gatherion detector doğrudan yüklenemedi: %s", exc)

    ai_reasons: List[str] = []
    class_id: Optional[int] = None
    confidence: Optional[float] = None

    if image is not None:
        model_verdict = None
        if det is not None:
            model_verdict = _top_from_detector(det, image)

        num_classes = _detector_num_classes(det) if det is not None else None
        gate = evaluate_node_candidate(
            image,
            source_type=source_type,
            resource_type=resource_type,
            detector=None,
            num_classes=num_classes,
            conf_min=MODEL_CONF_MIN,
            check_feather=True,
            check_monster=True,
            model_verdict=model_verdict,
            exemplar_memory=exemplar_memory,
            check_exemplars=True,
        )
        ai_reasons = list(gate.reasons)
        class_id = gate.class_id
        confidence = gate.confidence

        is_fake, fake_conf, fake_detail = is_fake_gray_rock_or_ground(
            image, exemplar_memory=exemplar_memory, candidate_class_id=class_id
        )
        if is_fake:
            ai_reasons.append(f"fake_terrain:{fake_detail}")

    # Teşhis metnini derle
    if ai_reasons:
        ai_diag = ", ".join(ai_reasons)
        cid_label = CLASS_LABELS_TR.get(int(class_id), str(class_id)) if class_id is not None else "Bilinmiyor"
        conf_str = f"{float(confidence):.2f}" if confidence is not None else "N/A"
        ai_verdict = f"YZ Doğrulandı: {ai_diag} (Sınıf: {cid_label}, Güven: {conf_str})"
    elif image is None:
        ai_verdict = "YZ Görsel Yok: Yama görseli eksik, 3x fiziksel ulaşılamama tespiti"
    else:
        ai_verdict = "YZ Tespiti: Kaynak boş/geçersiz zemin (3x ardışık toplama hatası)"

    # 3. Aktif Öğrenme (Active Learning): Yanlış düğümü negatif hafızaya kaydet
    if image is not None:
        try:
            save_wrong_nodes_as_negative_exemplars(
                [node],
                images_by_id={node_id: image} if node_id else None,
                exemplar_memory=exemplar_memory,
                region=target_region,
                source="consecutive_fail_3x",
            )
        except Exception as exc:
            logger.debug("Negatif örnek kaydı atlandı: %s", exc)

    # 4. Kara Listeye (Blacklist) Ekle
    bl_mgr = blacklist_mgr if blacklist_mgr is not None else get_blacklist_manager()
    bl_entry = bl_mgr.add_entry(
        region=target_region,
        map_x=map_x,
        map_y=map_y,
        norm_x=norm_x,
        norm_y=norm_y,
        reason=reason,
        ai_verdict=ai_verdict,
        source_node_id=node_id,
    )

    logger.warning(
        "🧠 [YZ Yanlış İşaretleme] Düğüm #%d incelendi ve kara listeye alındı. Teşhis: %s",
        node_id,
        ai_verdict,
    )
    return True, ai_verdict, bl_entry
