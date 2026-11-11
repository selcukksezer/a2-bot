# -*- coding: utf-8 -*-
"""One observation, several sources.

A single sample can carry minimap sprouts, 3D/YOLO boxes, an F/E gather
prompt, and the time-machine frame from about two seconds earlier.
Class ids follow the canonical 0..26 taxonomy
(dataset_factory.class_taxonomy.CANONICAL_NAMES). F/E gather prompts are
class 13 (skill_ui_prompt) — not 7 (herb_lumina) / 8 (ore_obsidian).
Boxes under the trust floor never enter
the train txt. A later, higher-confidence read can replace a weak box when
the same stem is observed again. Trusted boxes are only a real F/E prompt
or a hand label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_factory.class_taxonomy import CANONICAL_NAME_TO_ID
from dataset_factory.label_trust import (
    TRAIN_CONF_FLOOR,
    annotation_confidence,
    commit_split_labels,
    is_trusted_annotation,
)

# Canonical remap (F-0882, Decision 1): F/E gather prompts are UI prompts,
# class 13 (skill_ui_prompt). The legacy 7/8 ids are herb_lumina /
# ore_obsidian in the canonical taxonomy.
_CLASS_F = CANONICAL_NAME_TO_ID["skill_ui_prompt"]
_CLASS_E = CANONICAL_NAME_TO_ID["skill_ui_prompt"]
# Canonical class range is 0..26 (bars 16-18, skills 19-21, potions/scroll
# 22-24, kinah 25, player_character 26); see S-2003.
_MAX_CLASS_ID = max(CANONICAL_NAME_TO_ID.values())
_CLASS_SPROUT = 9
_CLASS_MONSTER = 10
_CLASS_MINIMAP_MONSTER = 11


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_id(raw: Dict[str, Any], default: Optional[int]) -> Optional[int]:
    if "class_id" in raw and raw.get("class_id") is not None:
        try:
            return int(raw["class_id"])
        except (TypeError, ValueError):
            return default
    return default


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


def _one_box(raw: Any, default_class: Optional[int], source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    xywh = _xywh(raw)
    if xywh is None:
        return None
    cx, cy, w, h = xywh
    cid = _class_id(raw, default_class)
    if cid is None or cid < 0 or cid > _MAX_CLASS_ID:
        # No class evidence and no semantic default for this source:
        # drop instead of inventing a class (e.g. crystal_blood/12).
        # Canonical range is 0..26 (S-2003): ids 16..26 (bars, skills,
        # potions/scroll, kinah, player_character) must survive fusion.
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
        # Y1 (tarama): a trusted prompt is cid 13 (skill_ui_prompt) + an
        # F/E trace — never a legacy 7/8 seed. label_trust's
        # _is_hand_or_gather_seed still recognises 7/8 in this branch
        # (E's fix lives in fix/sweep-label-pipeline, label_trust.py is
        # E's file); until that merges, the seed half of
        # is_trusted_annotation() is filtered here, and _real_prompt
        # requires cid 13. Hand labels stay trusted (human approval is
        # cid-independent — the box already carries trusted=True above).
        seed_trusted = is_trusted_annotation(box) and not (
            cid in (7, 8) and box.get("hand_label") is not True
        )
        _real_prompt = (
            cid == 13
            and key in ("f", "e")
            and raw.get("trusted") is True
            and str(box.get("source", "")).endswith("_gather_prompt")
        )
        # D41: single predicate call — is_trusted_annotation() never reads
        # the "trusted" key, so the old second call on
        # {**box, "trusted": True} was identical to the first.
        if seed_trusted or _real_prompt:
            box["trusted"] = True
    else:
        box["trusted"] = False
    return box


def _extend(
    out: List[Dict[str, Any]],
    items: Any,
    default_class: Optional[int],
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
    # No location evidence (no cx/cy, no bbox) -> no box at all. A prompt
    # without coordinates must never become a "trusted" box at a fixed
    # invented position (0.50, 0.62) with 0.0 confidence.
    if "cx" not in prompt and "bbox" not in prompt:
        return None
    if prompt.get("detected") is False:
        return None
    raw_key = prompt.get("gather_key", prompt.get("key"))
    if raw_key is None:
        key = "f"
    else:
        key = str(raw_key).strip().lower()
        if key not in ("f", "e"):
            # Unknown gather key: do not silently remap it to F.
            return None
    cid = _CLASS_E if key == "e" else _CLASS_F
    real = prompt.get("real_prompt") is True or prompt.get("hand_label") is True
    raw = dict(prompt)
    raw.setdefault("class_id", cid)
    raw.setdefault("w", 0.08)
    raw.setdefault("h", 0.08)
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
            age = float(getattr(payload, "age_sec", None)) if getattr(payload, "age_sec", None) is not None else None
        except (TypeError, ValueError):
            age = None
    boxes: List[Dict[str, Any]] = []
    _extend(boxes, detections, None, "time_machine")
    for box in boxes:
        box["source"] = "time_machine"
        box["trusted"] = False
        box.pop("gather_key", None)
        box.pop("hand_label", None)
    try:
        age_f = float(age) if age is not None else None
    except (TypeError, ValueError):
        age_f = None
    return boxes, age_f


def _dedup_exact(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse exact duplicates (same class + same rounded coords).

    ai/A15: the same object arrives twice when yolo and the ~2s-old
    time-machine frame agree. Downstream sanitize_boxes dedups IoU>0.8
    before the train txt, but the fused list itself carried the double
    count into the review sidecar and any per-class tally. Keep the best
    copy: trusted first, then higher confidence. Near-duplicates (IoU
    0.8-1.0, coords differ) are left to sanitize_boxes.
    """
    best: Dict[Tuple[int, float, float, float, float], Dict[str, Any]] = {}
    for box in boxes:
        key = (
            int(box["class_id"]),
            round(float(box["cx"]), 4),
            round(float(box["cy"]), 4),
            round(float(box["w"]), 4),
            round(float(box["h"]), 4),
        )
        cur = best.get(key)
        if cur is None or (bool(box.get("trusted")), float(box.get("confidence", 0.0))) > (
            bool(cur.get("trusted")),
            float(cur.get("confidence", 0.0)),
        ):
            best[key] = box
    order = {id(b): i for i, b in enumerate(boxes)}
    return sorted(best.values(), key=lambda b: order[id(b)])


def fuse_sources(sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold every present source into one annotation list. Ids are not renumbered."""
    sources = sources or {}
    boxes: List[Dict[str, Any]] = []
    used: List[str] = []

    world = sources.get("yolo", sources.get("world", sources.get("detections")))
    if world:
        n0 = len(boxes)
        _extend(boxes, world, None, "yolo3d")
        if len(boxes) > n0:
            used.append("yolo")

    minimap = sources.get("minimap", sources.get("sprouts"))
    if minimap:
        n0 = len(boxes)
        _extend(boxes, minimap, _CLASS_SPROUT, "minimap_sprout")
        if len(boxes) > n0 and "minimap" not in used:
            used.append("minimap")

    monsters = sources.get("minimap_monsters")
    if monsters:
        n0 = len(boxes)
        _extend(boxes, monsters, _CLASS_MINIMAP_MONSTER, "minimap_monster")
        if len(boxes) > n0 and "minimap" not in used:
            used.append("minimap")

    prompt = _prompt_box(sources.get("gather_prompt", sources.get("prompt")))
    if prompt is not None:
        boxes.append(prompt)
        used.append("gather_prompt")

    tm_boxes, age = _time_machine_parts(sources.get("time_machine"))
    if sources.get("time_machine") is not None:
        if tm_boxes:
            boxes.extend(tm_boxes)
            used.append("time_machine")

    boxes = _dedup_exact(boxes)
    class_counts: Dict[int, int] = {}
    for box in boxes:
        cid = int(box["class_id"])
        class_counts[cid] = class_counts.get(cid, 0) + 1

    return {
        "sources": used,
        "annotations": boxes,
        "time_machine_age_sec": age,
        "floor": TRAIN_CONF_FLOOR,
        "class_counts": class_counts,
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
    # P1-16 (K-05): bozuk review JSON'da commit durur — 0 kutu "boş kare"
    # değildir; labels TXT silinmez, sonuç aynen döndürülür.
    if result.get("blocked"):
        result["sources"] = list(fused.get("sources") or [])
        return result
    if int(result.get("train_box_count", 0)) <= 0 and label_path.is_file():
        try:
            label_path.unlink()
        except OSError:
            pass
        result["label_path"] = ""
    result["sources"] = list(fused.get("sources") or [])
    result["time_machine_age_sec"] = fused.get("time_machine_age_sec")
    # ai/A15: per-class mix balance (additive; fused may predate class_counts).
    counts: Dict[int, int] = dict(fused.get("class_counts") or {})
    if not counts:
        for b in fused.get("annotations") or []:
            cid = int(b["class_id"])
            counts[cid] = counts.get(cid, 0) + 1
    result["class_counts"] = counts
    return result
