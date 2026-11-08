# -*- coding: utf-8 -*-
"""One observation, several sources.

A single sample can carry minimap sprouts, 3D/YOLO boxes, an F/E gather
prompt, and the time-machine frame from about two seconds earlier.
Class ids 0-11 stay as they are. Boxes under the trust floor never enter
the train txt. A later, higher-confidence read can replace a weak box when
the same stem is observed again. Trusted boxes are only a real F/E prompt
or a hand label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_factory.label_trust import (
    TRAIN_CONF_FLOOR,
    annotation_confidence,
    commit_split_labels,
    is_trusted_annotation,
)

_CLASS_F = 7
_CLASS_E = 8
_CLASS_SPROUT = 9
_CLASS_MONSTER = 10
_CLASS_MINIMAP_MONSTER = 11


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_id(raw: Dict[str, Any], default: int) -> int:
    if "class_id" in raw and raw.get("class_id") is not None:
        try:
            return int(raw["class_id"])
        except (TypeError, ValueError):
            return int(default)
    return int(default)


def _xywh(raw: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if all(key in raw for key in ("cx", "cy", "w", "h")):
        return (
            _as_float(raw.get("cx"), 0.5),
            _as_float(raw.get("cy"), 0.5),
            max(0.0, _as_float(raw.get("w"), 0.0)),
            max(0.0, _as_float(raw.get("h"), 0.0)),
        )
    bbox = raw.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, w, h = [_as_float(v) for v in bbox[:4]]
        if max(x, y, w, h) > 1.5:
            return None
        return x + w * 0.5, y + h * 0.5, w, h
    return None


def _one_box(raw: Any, default_class: int, source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    xywh = _xywh(raw)
    if xywh is None:
        return None
    cx, cy, w, h = xywh
    cid = _class_id(raw, default_class)
    if cid < 0 or cid > 15:
        return None
    box = {
        "class_id": cid,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "confidence": annotation_confidence(raw),
        "source": str(raw.get("source") or source),
        "trusted": False,
    }
    if raw.get("hand_label") is True:
        box["hand_label"] = True
        box["trusted"] = True
    key = str(raw.get("gather_key", raw.get("key", ""))).strip().lower()
    if key in ("f", "e"):
        box["gather_key"] = key
    if is_trusted_annotation(box) or (
        raw.get("trusted") is True and (raw.get("hand_label") is True or key in ("f", "e"))
    ):
        if is_trusted_annotation({**box, "trusted": True}):
            box["trusted"] = True
    else:
        box["trusted"] = False
    return box


def _extend(
    out: List[Dict[str, Any]],
    items: Any,
    default_class: int,
    source: str,
) -> None:
    if items is None:
        return
    if isinstance(items, dict) and "boxes" in items:
        items = items.get("boxes")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        return
    for raw in items:
        box = _one_box(raw, default_class, source)
        if box is not None:
            out.append(box)


def _prompt_box(prompt: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    if prompt.get("detected") is False and "cx" not in prompt and "bbox" not in prompt:
        return None
    key = str(prompt.get("gather_key", prompt.get("key", "f"))).strip().lower()
    if key not in ("f", "e"):
        key = "f"
    cid = _CLASS_E if key == "e" else _CLASS_F
    real = prompt.get("real_prompt") is True or prompt.get("hand_label") is True
    raw = dict(prompt)
    raw.setdefault("class_id", cid)
    raw.setdefault("cx", raw.get("cx", 0.50))
    raw.setdefault("cy", raw.get("cy", 0.62))
    raw.setdefault("w", raw.get("w", 0.08))
    raw.setdefault("h", raw.get("h", 0.08))
    if real:
        raw["gather_key"] = key
        raw["source"] = "f_gather_prompt" if key == "f" else "e_gather_prompt"
        raw["trusted"] = True
    else:
        raw.pop("gather_key", None)
        raw.pop("key", None)
        raw["source"] = "prompt_candidate"
        raw["trusted"] = False
    return _one_box(raw, cid, raw["source"])


def _time_machine_parts(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    if payload is None:
        return [], None
    age = None
    detections: Any = None
    if isinstance(payload, dict):
        age = payload.get("age_sec", payload.get("time_machine_age_sec"))
        detections = payload.get("detections", payload.get("boxes"))
    else:
        detections = getattr(payload, "detections", None)
        try:
            age = float(getattr(payload, "age_sec", 2.0))
        except (TypeError, ValueError):
            age = 2.0
    boxes: List[Dict[str, Any]] = []
    _extend(boxes, detections, 12, "time_machine")
    for box in boxes:
        box["source"] = "time_machine"
        box["trusted"] = False
        box.pop("gather_key", None)
        box.pop("hand_label", None)
    if age is None and boxes:
        age = 2.0
    try:
        age_f = float(age) if age is not None else None
    except (TypeError, ValueError):
        age_f = 2.0 if boxes else None
    return boxes, age_f


def fuse_sources(sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold every present source into one annotation list. Ids are not renumbered."""
    sources = sources or {}
    boxes: List[Dict[str, Any]] = []
    used: List[str] = []

    world = sources.get("yolo", sources.get("world", sources.get("detections")))
    if world:
        _extend(boxes, world, 12, "yolo3d")
        used.append("yolo")

    minimap = sources.get("minimap", sources.get("sprouts"))
    if minimap:
        _extend(boxes, minimap, _CLASS_SPROUT, "minimap_sprout")
        used.append("minimap")

    monsters = sources.get("minimap_monsters")
    if monsters:
        _extend(boxes, monsters, _CLASS_MINIMAP_MONSTER, "minimap_monster")
        if "minimap" not in used:
            used.append("minimap")

    prompt = _prompt_box(sources.get("gather_prompt", sources.get("prompt")))
    if prompt is not None:
        boxes.append(prompt)
        used.append("gather_prompt")

    tm_boxes, age = _time_machine_parts(sources.get("time_machine"))
    if sources.get("time_machine") is not None:
        boxes.extend(tm_boxes)
        used.append("time_machine")

    return {
        "sources": used,
        "annotations": boxes,
        "time_machine_age_sec": age,
        "floor": TRAIN_CONF_FLOOR,
    }


def commit_mix_sample(
    *,
    dataset_dir: Path,
    stem: str,
    fused: Dict[str, Any],
    split: str = "train",
) -> Dict[str, Any]:
    """Write trusted boxes only. Weak boxes stay in the review sidecar."""
    dataset_dir = Path(dataset_dir)
    split_norm = "val" if str(split).lower() == "val" else "train"
    label_path = dataset_dir / "labels" / split_norm / f"{stem}.txt"
    result = commit_split_labels(
        dataset_dir=dataset_dir,
        split=split_norm,
        stem=stem,
        label_path=label_path,
        annotations=list(fused.get("annotations") or []),
        floor=TRAIN_CONF_FLOOR,
    )
    if int(result.get("train_box_count", 0)) <= 0 and label_path.is_file():
        try:
            label_path.unlink()
        except OSError:
            pass
        result["label_path"] = ""
    result["sources"] = list(fused.get("sources") or [])
    result["time_machine_age_sec"] = fused.get("time_machine_age_sec")
    return result
