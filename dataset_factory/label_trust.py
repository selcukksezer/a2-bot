# -*- coding: utf-8 -*-
"""Split trusted gather labels from weak auto boxes.

YOLO train txt files have no confidence. Writing every auto box there teaches
the wrong thing (last run mAP50 ~0.04). Low-confidence and untrusted boxes go
to dataset/review/*.json. A later, more confident observation may replace a
weak box. A box marked trusted is never overwritten.

Trusted seed: an F/E gather template hit, or a hand label (trusted=True).
Without that seed, a higher auto score still stays in the review sidecar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from dataset_factory.dataset_validator import (
    DEFAULT_IOU_OVERLAP_THRESHOLD,
    ValidationIssue,
    ValidationReport,
    check_out_of_bounds,
    check_zero_area,
    clear_model_stale,
    compute_box_iou,
    count_pending_review_items,
    detect_overlapping_boxes,
    fix_out_of_bounds,
    is_dataset_ready_for_training,
    is_model_stale,
    mark_model_stale,
    sanitize_boxes,
    validate_dataset,
    validate_yolo_file,
)

# No settings key for a train-label floor. The 0.42 band is uncertainty mining,
# not permission to enter the train txt. Boxes under this floor stay weak.
TRAIN_CONF_FLOOR: float = 0.45
# Omni learn_* layout boxes stay weak in review but are allowed into train txt
# (user culls wrong ones in Adım 2). Keep conf ≥ floor so purge does not wipe them.
LEARN_AUTO_CONF: float = 0.50
SLOT_IOU: float = 0.30

# Full-frame minimap HUD (sağ-üst). Padded vs mobile_server.MINIMAP_NORM_BOUNDS.
MINIMAP_HUD_XYXY: Tuple[float, float, float, float] = (0.82, 0.02, 1.0, 0.32)
# 0..5 world resources + world_gather_crest — minimap içinde herb_aria sahte pozitif
WORLD_RESOURCE_CLASS_IDS: frozenset = frozenset({0, 1, 2, 3, 4, 5, 6})
# Minimap ROI: filiz/canavar noktası + F/E UI (hflip sonrası sağ üstte kalabilir)
MINIMAP_ALLOWED_CLASS_IDS: frozenset = frozenset({7, 8, 9, 11})
SYNTHETIC_FALLBACK_SOURCES: frozenset = frozenset(
    {"center_anchor_fallback", "gather_material_hint"}
)
MINIMAP_SPROUT_CLASS_ID: int = 9
PATH_GROUND_CLASS_ID: int = 15
OBSTACLE_CLASS_ID: int = 14

UNVERIFIED_HEURISTIC_SOURCES: set = {
    "chromatic_salient_3d_resource",
    "visual_ground_path_segmenter",
    "visual_obstacle_segmenter",
    "hud_skill_slot",
    "hud_player_hp_bar",
    "hud_player_mp_bar",
    "dataset_requeue",
    "legacy_auto",
    "center_anchor_fallback",
    "gather_material_hint",
}

KNOWN_MATERIAL_TOKENS: Tuple[str, ...] = (
    "aria",
    "edana",
    "lumina",
    "orichalcum",
    "obsidian",
    "blood",
    "mela",
    "targena",
    "crystal",
    "tree",
    "log",
    "wood",
    "iron",
    "titanium",
    "adamantium",
    "garnet",
    "amethyst",
    "emerald",
    "ruby",
    "sapphire",
    "diamond",
)

_SEED_NAME = "trusted_seed.json"


def box_center_in_minimap_hud(
    cx: float,
    cy: float,
    xyxy: Tuple[float, float, float, float] = MINIMAP_HUD_XYXY,
) -> bool:
    x1, y1, x2, y2 = xyxy
    return float(x1) <= float(cx) <= float(x2) and float(y1) <= float(cy) <= float(y2)


def is_path_learning_stem(stem: str) -> bool:
    s = str(stem or "").strip().lower()
    return s.startswith("learn_paths_") or "_path_ground" in s or s.startswith("path_")


def is_omni_synthetic_source(source: str) -> bool:
    s = str(source or "").strip().lower()
    return s.startswith("omni_")


def is_learn_stem(stem: str) -> bool:
    """OmniGameLearner capture stems: learn_monsters_*, learn_skills_*, …"""
    return str(stem or "").strip().lower().startswith("learn_")


def is_learn_auto_annotation(ann: Dict[str, Any]) -> bool:
    """Weak layout/auto box from learn_* capture — user culls in Adım 2."""
    if not isinstance(ann, dict):
        return False
    if ann.get("learn_auto") is True:
        return True
    if ann.get("hand_label") is True:
        return False
    return bool(
        ann.get("omni_synthetic") is True
        or is_omni_synthetic_source(str(ann.get("source", "")))
    )


def train_txt_boxes(
    trusted: Sequence[Dict[str, Any]],
    weak: Sequence[Dict[str, Any]],
    *,
    stem: str = "",
    floor: float = TRAIN_CONF_FLOOR,
) -> List[Dict[str, Any]]:
    """Boxes that should appear in labels/{split}/*.txt.

    Trusted (floor/hand) always. For learn_* stems, weak learn/omni auto boxes
    are also emitted so train txt is not empty — Adım 2 culls mistakes.
    """
    floor_f = float(floor)
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(b: Dict[str, Any]) -> None:
        if "cx" not in b or "cy" not in b:
            return
        try:
            key = (
                int(b.get("class_id", -1)),
                round(float(b.get("cx", 0.0)), 6),
                round(float(b.get("cy", 0.0)), 6),
                round(float(b.get("w", 0.0)), 6),
                round(float(b.get("h", 0.0)), 6),
            )
        except (TypeError, ValueError):
            return
        if key in seen:
            return
        seen.add(key)
        out.append(b)

    for b in trusted or []:
        if not isinstance(b, dict):
            continue
        if bool(b.get("hand_label") is True) or annotation_confidence(b) >= floor_f:
            _add(b)

    if is_learn_stem(stem):
        for b in weak or []:
            if not isinstance(b, dict):
                continue
            if is_learn_auto_annotation(b) or annotation_confidence(b) >= floor_f:
                _add(b)

    return out


def filter_hud_false_positives(
    annotations: Sequence[Dict[str, Any]],
    *,
    stem: str = "",
    remap_sprout: bool = False,
) -> List[Dict[str, Any]]:
    """Drop boxes that cannot be true positives on HUD / wrong category stems.

    - Minimap ROI: only class 9/11 allowed (herb/path/skill/obstacle there = poison)
    - learn_paths_*: drop world resources; drop any box centered on minimap
    - Optional remap_sprout: world→class 9 instead of drop (rare)
    """
    out: List[Dict[str, Any]] = []
    path_mode = is_path_learning_stem(stem)
    for raw in annotations or []:
        if not isinstance(raw, dict):
            continue
        ann = dict(raw)
        try:
            cid = int(ann.get("class_id", -1))
        except (TypeError, ValueError):
            cid = -1
        try:
            cx = float(ann.get("cx", 0.5))
            cy = float(ann.get("cy", 0.5))
        except (TypeError, ValueError):
            out.append(ann)
            continue

        in_hud = box_center_in_minimap_hud(cx, cy)

        if path_mode and cid in WORLD_RESOURCE_CLASS_IDS:
            continue
        if path_mode and in_hud:
            # path_ground on minimap player-arrow = classic poison
            continue

        # Player HP/MP bars (16, 17) strictly occupy y: 0.83..0.925 above the bottom action bar.
        # Boxes at cy > 0.925 are on the bottom skill/potion slots and are false positives.
        if cid in (16, 17):
            if cy > 0.925 or cy < 0.82 or cx < 0.20 or cx > 0.80:
                continue

        if in_hud and cid not in MINIMAP_ALLOWED_CLASS_IDS:
            if remap_sprout and cid in WORLD_RESOURCE_CLASS_IDS:
                ann["class_id"] = MINIMAP_SPROUT_CLASS_ID
                ann["class_name"] = "minimap_sprout"
                ann["remapped_from"] = cid
                ann["source"] = str(ann.get("source", "")) + "|hud_remap_sprout"
                out.append(ann)
            continue

        out.append(ann)
    return out


def _is_hand_or_gather_seed(ann: Dict[str, Any]) -> bool:
    """True for hand labels / F/E gather templates — never bulk-demote these."""
    if ann.get("hand_label") is True:
        return True
    source = str(ann.get("source", "")).lower()
    if any(token in source for token in ("hand", "manual", "human")):
        return True
    key = str(ann.get("gather_key", ann.get("key", ""))).strip().lower()
    try:
        cid = int(ann.get("class_id", -1))
    except (TypeError, ValueError):
        cid = -1
    if cid == 7 and (
        key == "f"
        or "f_gather" in source
        or "gather_prompt" in source
        or "f_template" in source
    ):
        return True
    if cid == 8 and (
        key == "e"
        or "e_gather" in source
        or "e_floating" in source
        or "e_template" in source
        or "floating" in source
    ):
        return True
    return False


def _class_name_for_id(class_id: int) -> str:
    try:
        from dataset_factory.class_taxonomy import dataset_class_names

        names = dataset_class_names()
        if int(class_id) in names:
            return str(names[int(class_id)])
    except Exception:
        pass
    return f"class_{int(class_id)}"


def is_material_conflict(stem: str, class_name: str, class_id: int = -1) -> bool:
    """Detect if box class completely conflicts with stem material or is an invalid essence on 3D frame."""
    s_lower = str(stem or "").strip().lower()
    c_lower = str(class_name or "").strip().lower()
    if not s_lower:
        return False

    is_3d = any(k in s_lower for k in ("3d", "crest", "boot", "gather", "ref_"))
    # Essence classes (e.g. essence_yggdrasil_log, class_id >= 24) on any 3D frame
    if ("essence" in c_lower or c_lower.startswith("essence_") or class_id >= 24) and is_3d:
        return True

    # Identify material tokens present in stem
    m_in_stem = {m for m in KNOWN_MATERIAL_TOKENS if m in s_lower}
    if not m_in_stem:
        return False

    # If stem has a known material, an essence box is conflicting
    if "essence" in c_lower or c_lower.startswith("essence_") or class_id >= 24:
        return True

    # Check if box represents a conflicting specific material class
    m_in_box = {m for m in KNOWN_MATERIAL_TOKENS if m in c_lower}
    if class_id == 0:
        m_in_box.add("aria")
    elif class_id == 1:
        m_in_box.add("mela")
    elif class_id == 2:
        m_in_box.add("targena")
    elif class_id == 3:
        m_in_box.add("orichalcum")
    elif class_id == 4:
        m_in_box.add("crystal")
    elif class_id == 5:
        m_in_box.add("tree")
        m_in_box.add("wood")
        m_in_box.add("log")

    if m_in_box and not (m_in_stem & m_in_box):
        return True

    return False


def calibrate_box_confidence_and_trust(
    box: Dict[str, Any],
    stem: str = "",
    emin_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Calibrate box confidence and trust to prevent false >= 0.85 confident / trusted labels.

    1. If box.get("hand_label") or box.get("human_approved") or box.get("approved") or _is_hand_or_gather_seed(box):
       box["trusted"] = True, box["weak"] = False, box["confidence"] = max(0.90, float(box.get("confidence", 0.90))).
    2. If stem contains a known material identifier and the box class is completely conflicting,
       or essence_* on any 3D frame:
       set box["confidence"] = min(float(box.get("confidence", 0.40)), 0.45), box["trusted"] = False, box["weak"] = True.
    3. If source in UNVERIFIED_HEURISTIC_SOURCES and not human approved:
       box["confidence"] = min(float(box.get("confidence", 0.58)), 0.68), box["trusted"] = False, box["weak"] = True.
       Never allow unverified heuristics to be trusted=True or >= 0.85!
    4. Otherwise:
       if confidence >= emin_threshold: box["trusted"] = True, box["weak"] = False
       else: box["trusted"] = False, box["weak"] = True.
    Returns calibrated box.
    """
    if not isinstance(box, dict):
        return box

    # 1. Human approved or seed check
    if (
        box.get("hand_label") is True
        or box.get("human_approved") is True
        or box.get("approved") is True
        or box.get("confirmed") is True
        or _is_hand_or_gather_seed(box)
    ):
        try:
            cur_conf = float(box.get("confidence", 0.90))
        except (TypeError, ValueError):
            cur_conf = 0.90
        box["trusted"] = True
        box["weak"] = False
        box["confidence"] = round(max(0.90, cur_conf), 4)
        return box

    # Extract source, class, stem
    src = str(box.get("source", "")).strip().lower()
    cid = -1
    try:
        cid = int(box.get("class_id", -1))
    except (TypeError, ValueError):
        pass
    cname = str(box.get("class_name") or box.get("key") or _class_name_for_id(cid)).lower()
    stem_str = str(stem or box.get("stem", "")).strip()

    # 2. Known material conflict check
    if is_material_conflict(stem_str, cname, cid):
        try:
            raw_c = float(box.get("confidence", 0.40))
        except (TypeError, ValueError):
            raw_c = 0.40
        box["confidence"] = round(min(raw_c, 0.45), 4)
        box["trusted"] = False
        box["weak"] = True
        return box

    # 3. Unverified heuristic sources check
    is_unverified = (
        src in UNVERIFIED_HEURISTIC_SOURCES
        or any(uhs in src for uhs in UNVERIFIED_HEURISTIC_SOURCES)
        or is_omni_synthetic_source(src)
        or box.get("omni_synthetic") is True
        or box.get("synthetic_fallback") is True
    )
    if is_unverified:
        try:
            raw_c = float(box.get("confidence", 0.58))
        except (TypeError, ValueError):
            raw_c = 0.58
        box["confidence"] = round(min(raw_c, 0.68), 4)
        box["trusted"] = False
        box["weak"] = True
        return box

    # 4. Standard confidence vs emin_threshold check
    try:
        cur_conf = float(box.get("confidence", 0.58))
    except (TypeError, ValueError):
        cur_conf = 0.58
    eth = float(emin_threshold)
    if cur_conf >= eth:
        box["confidence"] = round(cur_conf, 4)
        box["trusted"] = True
        box["weak"] = False
    else:
        box["confidence"] = round(cur_conf, 4)
        box["trusted"] = False
        box["weak"] = True

    return box


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        path.write_text(content, encoding=encoding)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content, encoding="utf-8")


def _rewrite_label_txt(label_path: Path, trusted: Sequence[Dict[str, Any]]) -> None:
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_yolo_line(b) for b in trusted if "cx" in b]
    content = ("\n".join(lines) + "\n") if lines else ""
    _atomic_write_text(label_path, content, encoding="utf-8")


def purge_poisoned_weak_reviews(
    dataset_dir: Union[str, Path],
    *,
    drop_sub_floor: bool = True,
    floor: float = TRAIN_CONF_FLOOR,
    use_ai: bool = False,
    exemplar_memory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Rewrite review JSON: filter weak only. Hand/F/E trusted untouched.

    Drops HUD false positives and (optionally) sub-floor weak boxes.
    If use_ai=True, runs the full 4-layer AI poison cleaner (clean_poison_labels_ai).
    """
    if use_ai:
        return clean_poison_labels_ai(
            dataset_dir,
            exemplar_memory=exemplar_memory,
            floor=floor,
        )

    root = Path(dataset_dir)
    review_root = root / "review"
    touched = 0
    dropped = 0
    emptied = 0
    scanned = 0
    for split in ("train", "val"):
        split_dir = review_root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.json")):
            if path.name == _SEED_NAME:
                continue
            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            weak = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            if not weak:
                continue
            stem = str(payload.get("stem") or path.stem)
            cleaned = filter_hud_false_positives(weak, stem=stem, remap_sprout=False)
            if drop_sub_floor:
                kept: List[Dict[str, Any]] = []
                learn_frame = is_learn_stem(stem)
                for b in cleaned:
                    if annotation_confidence(b) < float(floor) and not _is_hand_or_gather_seed(b):
                        # learn_* / learn_auto weak must survive for Adım 2 cull.
                        if learn_frame or is_learn_auto_annotation(b):
                            kept.append(b)
                        continue
                    kept.append(b)
                cleaned = kept
            dropped_here = len(weak) - len(cleaned)
            if dropped_here <= 0:
                continue
            dropped += dropped_here
            payload["weak"] = cleaned
            _atomic_write_json(path, payload)
            touched += 1
            if not cleaned and not (payload.get("trusted") or []):
                emptied += 1
    if touched > 0 or dropped > 0:
        mark_model_stale(
            reason=f"Zayıf etiketler temizlendi ({dropped} kutu silindi, {touched} dosya)",
            dataset_dir=root,
        )
    return {
        "scanned": scanned,
        "files_rewritten": touched,
        "weak_dropped": dropped,
        "files_weak_empty": emptied,
        "floor": float(floor),
    }


def demote_omni_synthetic_labels(
    dataset_dir: Union[str, Path],
    *,
    keep_as_weak: bool = False,
    floor: float = TRAIN_CONF_FLOOR,
) -> Dict[str, Any]:
    """Strip omni_* synthetic boxes out of train txt.

    Hand/F/E seed trusted boxes stay. Omni boxes are dropped from trusted
    (and optionally kept as weak below floor for UI review). Also runs
    HUD/sub-floor purge on remaining weak. Rewrites matching labels/*.txt.
    """
    root = Path(dataset_dir)
    scanned = 0
    files_rewritten = 0
    demoted_trusted = 0
    dropped_boxes = 0
    weak_purged = 0
    for split in ("train", "val"):
        review_dir = root / "review" / split
        labels_dir = root / "labels" / split
        if not review_dir.is_dir():
            continue
        for path in sorted(review_dir.glob("*.json")):
            if path.name == _SEED_NAME:
                continue
            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            stem = str(payload.get("stem") or path.stem)
            trusted_in = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
            weak_in = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []
            changed = False
            learn_frame = is_learn_stem(stem)
            # learn_* frames: keep omni layout as weak (user culls); still demote from trusted.
            keep_weak_here = bool(keep_as_weak) or learn_frame

            for b in trusted_in:
                src = str(b.get("source", "")).lower()
                if _is_hand_or_gather_seed(b):
                    new_trusted.append(b)
                    continue
                synthetic = (
                    is_omni_synthetic_source(src)
                    or b.get("omni_synthetic") is True
                    or b.get("synthetic_fallback") is True
                    or src in SYNTHETIC_FALLBACK_SOURCES
                )
                if synthetic:
                    demoted_trusted += 1
                    changed = True
                    filtered = filter_hud_false_positives([b], stem=stem, remap_sprout=False)
                    if not filtered:
                        dropped_boxes += 1
                        continue
                    if keep_weak_here:
                        nb = dict(filtered[0])
                        nb["trusted"] = False
                        nb["weak"] = True
                        nb["learn_auto"] = True if learn_frame else nb.get("learn_auto")
                        if learn_frame:
                            nb["confidence"] = max(
                                annotation_confidence(nb), float(LEARN_AUTO_CONF)
                            )
                        else:
                            nb["confidence"] = min(
                                annotation_confidence(nb), float(floor) - 0.05
                            )
                        nb["demoted_from_omni"] = True
                        new_weak.append(nb)
                    else:
                        dropped_boxes += 1
                    continue
                # Trusted but sitting illegally on minimap (world class) → drop
                filtered_t = filter_hud_false_positives([b], stem=stem, remap_sprout=False)
                if not filtered_t:
                    demoted_trusted += 1
                    dropped_boxes += 1
                    changed = True
                    continue
                new_trusted.append(filtered_t[0])

            for b in weak_in:
                src = str(b.get("source", "")).lower()
                filtered = filter_hud_false_positives([b], stem=stem, remap_sprout=False)
                if not filtered:
                    dropped_boxes += 1
                    weak_purged += 1
                    changed = True
                    continue
                bb = filtered[0]
                if (
                    is_omni_synthetic_source(src) or bb.get("omni_synthetic") is True
                ) and not keep_weak_here:
                    dropped_boxes += 1
                    weak_purged += 1
                    changed = True
                    continue
                if (
                    annotation_confidence(bb) < float(floor)
                    and not _is_hand_or_gather_seed(bb)
                    and not learn_frame
                    and not is_learn_auto_annotation(bb)
                ):
                    dropped_boxes += 1
                    weak_purged += 1
                    changed = True
                    continue
                if learn_frame:
                    bb = dict(bb)
                    bb["learn_auto"] = True
                    bb["weak"] = True
                    bb["trusted"] = False
                new_weak.append(bb)
                if bb is not b:
                    changed = True

            if not changed and len(new_trusted) == len(trusted_in) and len(new_weak) == len(weak_in):
                # still rewrite txt if trusted list shrunk via identity
                pass
            if (
                len(new_trusted) == len(trusted_in)
                and len(new_weak) == len(weak_in)
                and not changed
            ):
                continue

            payload["trusted"] = new_trusted
            payload["weak"] = new_weak
            _atomic_write_json(path, payload)
            emit = train_txt_boxes(
                new_trusted, new_weak, stem=stem, floor=float(floor)
            )
            _rewrite_label_txt(labels_dir / f"{stem}.txt", emit)
            files_rewritten += 1

    weak_stats = purge_poisoned_weak_reviews(root, drop_sub_floor=True, floor=floor)
    total_rewritten = files_rewritten + int(weak_stats.get("files_rewritten", 0))
    if total_rewritten > 0 or demoted_trusted > 0:
        mark_model_stale(
            reason=f"Omni sentetik etiketler düşürüldü ({demoted_trusted} kutu, {total_rewritten} dosya)",
            dataset_dir=root,
        )
    return {
        "scanned": scanned,
        "files_rewritten": total_rewritten,
        "demoted_trusted": demoted_trusted,
        "dropped_boxes": dropped_boxes + int(weak_stats.get("weak_dropped", 0)),
        "weak_purged": weak_purged,
        "floor": float(floor),
        "keep_as_weak": bool(keep_as_weak),
    }


def _find_image_path(root: Path, split: str, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        p = root / "images" / split / f"{stem}{ext}"
        if p.is_file():
            return p
        alt_split = "val" if split == "train" else "train"
        p_alt = root / "images" / alt_split / f"{stem}{ext}"
        if p_alt.is_file():
            return p_alt
        p_base = root / "images" / f"{stem}{ext}"
        if p_base.is_file():
            return p_base
    return None


def _nms_dedupe(boxes: List[Dict[str, Any]], iou_thresh: float = 0.65) -> Tuple[List[Dict[str, Any]], int]:
    def _rank(b: Dict[str, Any]) -> Tuple[float, float]:
        is_t = 1.0 if (b.get("trusted") is True or b.get("hand_label") is True) else 0.0
        return (is_t, annotation_confidence(b))

    sorted_boxes = sorted(boxes, key=_rank, reverse=True)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for b in sorted_boxes:
        suppress = False
        for k in kept:
            if int(b.get("class_id", -1)) == int(k.get("class_id", -1)):
                if slot_iou(b, k) > iou_thresh:
                    suppress = True
                    dropped += 1
                    break
        if not suppress:
            kept.append(b)
    return kept, dropped


def clean_poison_labels_ai(
    dataset_dir: Union[str, Path],
    *,
    exemplar_memory: Optional[Any] = None,
    floor: float = TRAIN_CONF_FLOOR,
    emin_threshold: float = 0.85,
    teach_negatives: bool = True,
    dry_run: bool = False,
    frame_map: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    4-Layer Advanced AI Poison Label Cleaner:
    Katman 0: Dokunulmazlık Kalkanı (hand_label, human_approved, template seed, or positive exemplar >= 0.72)
              boxes are NEVER dropped and are marked trusted=True (🟢 Emin).
    Katman 1: YZ Örnek Hafızası & Otomatik Sınıf Onarımı (positive exemplar match >= 0.82 on different class
              repairs class_id/class_name; negative exemplar match >= 0.82 with margin >= 0.15 deletes poison).
    Katman 2: Görsel Boş Zemin / Gökyüzü / HUD Zehir Kontrolü (illegal HUD position or Laplacian variance < 10.0
              flat featureless surface deletes box and optionally teaches negative exemplar).
    Katman 3: Sentetik omni_*, NMS Deduplication, and synchronized Review / Train txt rewriting.
    """
    import cv2
    import numpy as np
    from dataset_factory.exemplar_memory import ExemplarMemory, extract_box_crop

    root = Path(dataset_dir)
    if exemplar_memory is None:
        try:
            exemplar_memory = ExemplarMemory()
        except Exception:
            exemplar_memory = None

    scanned = 0
    files_rewritten = 0
    labels_rewritten = 0
    protected_boxes = 0
    auto_repaired_boxes = 0
    poison_deleted_boxes = 0
    negative_taught_boxes = 0
    negative_match_deleted = 0
    hud_poison_deleted = 0
    flat_texture_deleted = 0
    omni_demoted_boxes = 0
    sub_floor_deleted = 0
    deduped_boxes = 0

    for split in ("train", "val"):
        review_dir = root / "review" / split
        labels_dir = root / "labels" / split
        if not review_dir.is_dir():
            continue

        review_files = sorted(review_dir.glob("*.json"))
        total_files = len(review_files)

        for idx, path in enumerate(review_files):
            if path.name == _SEED_NAME:
                continue
            scanned += 1
            if progress_callback:
                try:
                    progress_callback(scanned, total_files)
                except Exception:
                    pass

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            stem = str(payload.get("stem") or path.stem)
            trusted_in = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
            weak_in = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            all_boxes = list(trusted_in) + list(weak_in)
            if not all_boxes:
                continue

            # Load frame if available
            frame: Optional[np.ndarray] = None
            if frame_map and stem in frame_map:
                frame = frame_map[stem]
            else:
                img_path = _find_image_path(root, split, stem)
                if img_path and img_path.is_file():
                    try:
                        frame = cv2.imread(str(img_path))
                    except Exception:
                        frame = None

            learn_frame = is_learn_stem(stem)
            candidate_kept: List[Dict[str, Any]] = []

            for b in all_boxes:
                b = dict(b)
                conf = annotation_confidence(b)
                cid = int(b.get("class_id", 0))
                cname = str(b.get("class_name", ""))
                src = str(b.get("source", "")).lower()

                crop: Optional[np.ndarray] = None
                if frame is not None:
                    try:
                        crop = extract_box_crop(frame, b)
                    except Exception:
                        crop = None

                # Katman 0: Dokunulmazlık Kalkanı (hand_label / human_approved / seed)
                is_immune = (
                    b.get("hand_label") is True
                    or b.get("human_approved") is True
                    or b.get("approved") is True
                    or _is_hand_or_gather_seed(b)
                )

                if is_immune:
                    protected_boxes += 1
                    b["trusted"] = True
                    b["weak"] = False
                    candidate_kept.append(b)
                    continue

                # Katman 2: Görsel Boş Zemin / Gökyüzü / HUD Zehir Kontrolü
                filtered_hud = filter_hud_false_positives([b], stem=stem, remap_sprout=False)
                if not filtered_hud:
                    poison_deleted_boxes += 1
                    hud_poison_deleted += 1
                    if teach_negatives and exemplar_memory is not None and crop is not None and crop.size > 0:
                        try:
                            exemplar_memory.add_negative_exemplar(
                                crop, cid, cname, source="poison_hud_illegal"
                            )
                            negative_taught_boxes += 1
                        except Exception:
                            pass
                    continue

                lap_var = 50.0
                if crop is not None and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                    try:
                        cgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        lap_var = float(cv2.Laplacian(cgray, cv2.CV_32F).var())
                    except Exception:
                        lap_var = 50.0

                if lap_var < 10.0 and cid != 15:
                    poison_deleted_boxes += 1
                    flat_texture_deleted += 1
                    if teach_negatives and exemplar_memory is not None and crop is not None and crop.size > 0:
                        try:
                            exemplar_memory.add_negative_exemplar(
                                crop, cid, cname, source="poison_flat_texture"
                            )
                            negative_taught_boxes += 1
                        except Exception:
                            pass
                    continue

                # Katman 1: YZ Örnek Hafızası & Negatif Bastırma / Otomatik Onarım
                pos_sim = 0.0
                pos_cid: Optional[int] = None
                pos_cname: Optional[str] = None
                if "exemplar_repair_sim" in b:
                    pos_sim = float(b["exemplar_repair_sim"])
                    pos_cid = int(b.get("exemplar_repair_class_id", cid))
                    pos_cname = str(b.get("exemplar_repair_class_name", f"class_{pos_cid}"))
                elif "exemplar_pos_sim" in b:
                    pos_sim = float(b["exemplar_pos_sim"])
                elif exemplar_memory is not None and crop is not None and crop.size > 0:
                    try:
                        pos_sim, pos_cid, pos_cname, _ = exemplar_memory.match_positive(crop)
                    except Exception:
                        pos_sim = 0.0

                neg_sim = 0.0
                if "exemplar_neg_sim" in b:
                    neg_sim = float(b["exemplar_neg_sim"])
                elif exemplar_memory is not None and crop is not None and crop.size > 0:
                    try:
                        neg_sim, _ = exemplar_memory.match_negative(crop, class_id=cid)
                    except Exception:
                        neg_sim = 0.0

                if neg_sim >= 0.82 and (neg_sim - pos_sim) >= 0.15:
                    poison_deleted_boxes += 1
                    negative_match_deleted += 1
                    continue

                if pos_sim >= 0.82 and pos_cid is not None and pos_cid != cid:
                    b["class_id"] = pos_cid
                    b["class_name"] = pos_cname or f"class_{pos_cid}"
                    b["confidence"] = max(conf, float(pos_sim))
                    b["repaired_by_exemplar"] = True
                    auto_repaired_boxes += 1

                if pos_sim >= 0.72:
                    protected_boxes += 1
                    if conf >= emin_threshold or pos_sim >= 0.85:
                        b["trusted"] = True
                        b["weak"] = False
                    candidate_kept.append(b)
                    continue

                # Katman 3: Omni synthetic & sub-floor filtering
                if (
                    is_omni_synthetic_source(src)
                    or b.get("omni_synthetic") is True
                    or b.get("synthetic_fallback") is True
                    or src in SYNTHETIC_FALLBACK_SOURCES
                ):
                    poison_deleted_boxes += 1
                    omni_demoted_boxes += 1
                    continue

                if conf < float(floor) and not learn_frame and not is_learn_auto_annotation(b):
                    poison_deleted_boxes += 1
                    sub_floor_deleted += 1
                    continue

                # Check if positive exemplar match can repair class even if pos_sim < 0.72 but >= 0.82
                if pos_sim >= 0.82 and pos_cid is not None and pos_cid != cid:
                    b["class_id"] = pos_cid
                    b["class_name"] = pos_cname or f"class_{pos_cid}"
                    b["confidence"] = max(conf, float(pos_sim))
                    b["repaired_by_exemplar"] = True
                    auto_repaired_boxes += 1

                candidate_kept.append(b)

            # NMS Deduplication
            deduped, nms_dropped = _nms_dedupe(candidate_kept, iou_thresh=0.65)
            deduped_boxes += nms_dropped

            # Categorize into trusted (Emin) vs weak (Şüpheli)
            final_trusted: List[Dict[str, Any]] = []
            final_weak: List[Dict[str, Any]] = []
            for b in deduped:
                b = calibrate_box_confidence_and_trust(b, stem=stem, emin_threshold=emin_threshold)
                if b.get("trusted") is True:
                    final_trusted.append(b)
                else:
                    final_weak.append(b)

            if not dry_run:
                payload["trusted"] = final_trusted
                payload["weak"] = final_weak
                _atomic_write_json(path, payload)
                files_rewritten += 1

                emit = train_txt_boxes(
                    final_trusted, final_weak, stem=stem, floor=float(floor)
                )
                labels_dir.mkdir(parents=True, exist_ok=True)
                _rewrite_label_txt(labels_dir / f"{stem}.txt", emit)
                labels_rewritten += 1

    if not dry_run and files_rewritten > 0:
        mark_model_stale(
            reason=f"AI Zehirli Etiket Temizleme ({poison_deleted_boxes} kutu silindi, {files_rewritten} dosya)",
            dataset_dir=root,
        )
        try:
            from dataset_factory.dataset_curator import DatasetCurator
            curator = DatasetCurator(dataset_dir=root)
            curator.rebuild_manifest_stats()
        except Exception:
            pass

    return {
        "ok": True,
        "scanned": scanned,
        "files_rewritten": files_rewritten,
        "labels_rewritten": labels_rewritten,
        "protected_boxes": protected_boxes,
        "auto_repaired_boxes": auto_repaired_boxes,
        "poison_deleted_boxes": poison_deleted_boxes,
        "negative_taught_boxes": negative_taught_boxes,
        "negative_match_deleted": negative_match_deleted,
        "hud_poison_deleted": hud_poison_deleted,
        "flat_texture_deleted": flat_texture_deleted,
        "omni_demoted_boxes": omni_demoted_boxes,
        "sub_floor_deleted": sub_floor_deleted,
        "deduped_boxes": deduped_boxes,
        "floor": float(floor),
        "emin_threshold": float(emin_threshold),
    }

def prune_low_confidence_boxes_in_dataset(
    dataset_dir: Union[str, Path],
    threshold: float = 0.85,
    *,
    min_confidence: Optional[float] = None,
    indices_or_stems: Optional[Sequence[str]] = None,
    record_negative_exemplars: bool = True,
    teach_negative: Optional[bool] = None,
    exemplar_memory: Optional[Any] = None,
    frame_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Box-level pruning: Prunes ONLY individual boxes below threshold from review JSONs and train txt.
    Preserves images and high-confidence boxes, marking kept boxes trusted=True (🟢 Emin).
    Pruned boxes are learned as negative exemplars in ExemplarMemory.
    """
    root = Path(dataset_dir)
    if min_confidence is not None:
        threshold = float(min_confidence)
    if teach_negative is not None:
        record_negative_exemplars = bool(teach_negative)
    if exemplar_memory is None and record_negative_exemplars:
        try:
            from dataset_factory.exemplar_memory import ExemplarMemory
            exemplar_memory = ExemplarMemory()
        except Exception:
            exemplar_memory = None

    from dataset_factory.exemplar_memory import extract_box_crop

    stem_filter = set(indices_or_stems) if indices_or_stems is not None else None
    th = float(threshold)

    scanned = 0
    pruned_boxes = 0
    promoted_boxes = 0
    preserved_frames = 0
    files_rewritten = 0

    for split in ("train", "val"):
        review_dir = root / "review" / split
        labels_dir = root / "labels" / split
        if not review_dir.is_dir():
            continue

        for path in sorted(review_dir.glob("*.json")):
            if path.name == _SEED_NAME:
                continue
            stem = path.stem
            if stem_filter is not None and stem not in stem_filter:
                continue

            scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            trusted_in = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
            weak_in = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            all_boxes = list(trusted_in) + list(weak_in)
            if not all_boxes:
                continue

            frame = None
            if record_negative_exemplars and exemplar_memory is not None:
                if frame_map and stem in frame_map:
                    frame = frame_map[stem]
                else:
                    img_path = _find_image_path(root, split, stem)
                    if img_path and img_path.is_file():
                        try:
                            import cv2
                            frame = cv2.imread(str(img_path))
                        except Exception:
                            frame = None

            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []
            frame_pruned = 0

            for b in all_boxes:
                b = dict(b)
                conf = annotation_confidence(b)
                is_protected = (
                    b.get("hand_label") is True
                    or b.get("human_approved") is True
                    or b.get("approved") is True
                    or _is_hand_or_gather_seed(b)
                )

                if is_protected or conf >= th:
                    b["trusted"] = True
                    b["weak"] = False
                    new_trusted.append(b)
                    promoted_boxes += 1
                else:
                    frame_pruned += 1
                    pruned_boxes += 1
                    if record_negative_exemplars and exemplar_memory is not None and frame is not None:
                        try:
                            crop = extract_box_crop(frame, b)
                            if crop is not None and crop.size > 0:
                                cid = int(b.get("class_id", 0))
                                cname = str(b.get("class_name", ""))
                                exemplar_memory.add_negative_exemplar(
                                    crop, cid, cname, source="bulk_prune_low_conf"
                                )
                        except Exception:
                            pass

            if frame_pruned > 0 or len(new_trusted) != len(trusted_in):
                payload["trusted"] = new_trusted
                payload["weak"] = new_weak
                _atomic_write_json(path, payload)
                emit = train_txt_boxes(new_trusted, new_weak, stem=stem, floor=TRAIN_CONF_FLOOR)
                labels_dir.mkdir(parents=True, exist_ok=True)
                _rewrite_label_txt(labels_dir / f"{stem}.txt", emit)
                files_rewritten += 1

            preserved_frames += 1

    if files_rewritten > 0:
        mark_model_stale(
            reason=f"Kutu düzeyinde budama ({pruned_boxes} kutu silindi, {files_rewritten} dosya)",
            dataset_dir=root,
        )
        try:
            from dataset_factory.dataset_curator import DatasetCurator
            DatasetCurator(dataset_dir=root).rebuild_manifest_stats()
        except Exception:
            pass

    return {
        "ok": True,
        "scanned": scanned,
        "images_checked": scanned,
        "pruned_boxes": pruned_boxes,
        "boxes_pruned": pruned_boxes,
        "promoted_boxes": promoted_boxes,
        "preserved_frames": preserved_frames,
        "files_rewritten": files_rewritten,
        "threshold": th,
    }




def annotation_confidence(ann: Dict[str, Any]) -> float:
    try:
        return float(ann.get("confidence", ann.get("conf", ann.get("score", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def is_trusted_annotation(ann: Dict[str, Any]) -> bool:
    """F/E template or an explicit hand label / human approved label. Omni synthetic boxes are never trusted."""
    if not isinstance(ann, dict):
        return False
    source = str(ann.get("source", "")).lower()
    if is_omni_synthetic_source(source) or ann.get("omni_synthetic") is True:
        return False
    if (
        ann.get("hand_label") is True
        or ann.get("human_approved") is True
        or ann.get("approved") is True
        or ann.get("confirmed") is True
    ):
        return True
    if source in UNVERIFIED_HEURISTIC_SOURCES or any(s in source for s in UNVERIFIED_HEURISTIC_SOURCES):
        return False
    if _is_hand_or_gather_seed(ann):
        return True
    return False


def _box_xyxy(ann: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx = float(ann.get("cx", 0.5))
    cy = float(ann.get("cy", 0.5))
    w = float(ann.get("w", ann.get("width", 0.0)))
    h = float(ann.get("h", ann.get("height", 0.0)))
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def slot_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _box_xyxy(a)
    bx1, by1, bx2, by2 = _box_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def _find_slot(boxes: Sequence[Dict[str, Any]], candidate: Dict[str, Any]) -> Optional[int]:
    best_i: Optional[int] = None
    best = SLOT_IOU
    for i, box in enumerate(boxes):
        score = slot_iou(box, candidate)
        if score >= best:
            best = score
            best_i = i
    return best_i


def _review_path(dataset_dir: Path, split: str, stem: str) -> Path:
    return dataset_dir / "review" / split / f"{stem}.json"


def _seed_path(dataset_dir: Path) -> Path:
    return dataset_dir / "review" / _SEED_NAME


def has_trusted_seed(dataset_dir: Path) -> bool:
    path = _seed_path(dataset_dir)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("has_trusted_seed"))


def _mark_trusted_seed(dataset_dir: Path) -> None:
    path = _seed_path(dataset_dir)
    _atomic_write_json(path, {"has_trusted_seed": True})


def _legacy_weak_from_txt(label_path: Path) -> List[Dict[str, Any]]:
    if not label_path.is_file():
        return []
    weak: List[Dict[str, Any]] = []
    try:
        raw = label_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf_val: Optional[float] = float(parts[5]) if len(parts) >= 6 else None
            entry: Dict[str, Any] = {
                "class_id": cid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "confidence": conf_val if conf_val is not None else 0.0,
                "trusted": False,
                "weak": True,
                "legacy_auto": True,
            }
            if conf_val is not None:
                entry["has_explicit_conf"] = True
            weak.append(entry)
        except ValueError:
            continue
    return weak


def _load_state(
    dataset_dir: Path,
    split: str,
    stem: str,
    label_path: Path,
    *,
    return_meta: bool = False,
) -> Any:
    path = _review_path(dataset_dir, split, stem)
    meta: Dict[str, Any] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for k in ("human_approved", "done", "approved_at"):
                    if k in payload:
                        meta[k] = payload[k]
                is_approved = bool(
                    payload.get("human_approved") is True or payload.get("done") == "ok"
                )
                trusted: List[Dict[str, Any]] = []
                for b in payload.get("trusted", []):
                    if isinstance(b, dict):
                        nb = dict(b)
                        if is_approved and "human_approved" not in nb:
                            nb["human_approved"] = True
                        if "approved_at" in meta and "approved_at" not in nb:
                            nb["approved_at"] = meta["approved_at"]
                        trusted.append(nb)
                weak = [dict(b) for b in payload.get("weak", []) if isinstance(b, dict)]
                if return_meta:
                    return trusted, weak, meta
                return trusted, weak
        except Exception:
            pass
    if return_meta:
        return [], _legacy_weak_from_txt(label_path), meta
    return [], _legacy_weak_from_txt(label_path)


def _yolo_line(ann: Dict[str, Any]) -> str:
    return (
        f"{int(ann['class_id'])} {float(ann['cx']):.6f} {float(ann['cy']):.6f} "
        f"{float(ann['w']):.6f} {float(ann['h']):.6f}"
    )


def merge_label_observation(
    trusted: Sequence[Dict[str, Any]],
    weak: Sequence[Dict[str, Any]],
    observations: Iterable[Dict[str, Any]],
    *,
    has_seed: bool,
    floor: float = TRAIN_CONF_FLOOR,
    require_human_confirmation: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Return updated trusted list, weak list, and whether a trusted seed exists."""
    kept_trusted = [dict(b) for b in trusted]
    kept_weak = [dict(b) for b in weak]
    seed = bool(has_seed) or any(is_trusted_annotation(b) or b.get("trusted") for b in kept_trusted)

    floor_f = float(floor)
    for raw in observations:
        if not isinstance(raw, dict):
            continue
        if "cx" not in raw or "cy" not in raw:
            continue
        new = dict(raw)
        new_conf = annotation_confidence(new)
        new["confidence"] = new_conf
        # Explicit hand labels may be trusted below floor; omni synthetic never trains.
        hand = bool(
            new.get("hand_label") is True
            or new.get("human_approved") is True
            or new.get("approved") is True
            or new.get("confirmed") is True
        )
        src_l = str(new.get("source", "")).lower()
        omni_syn = is_omni_synthetic_source(src_l) or (
            new.get("omni_synthetic") is True
        )
        synthetic_fallback = bool(
            new.get("synthetic_fallback") is True or src_l in SYNTHETIC_FALLBACK_SOURCES
        )
        if synthetic_fallback:
            omni_syn = True  # same promote block as omni layout fakes
        new_is_trusted = is_trusted_annotation(new) and (hand or new_conf >= floor_f)
        if omni_syn:
            new_is_trusted = False
        if not hand and new_conf < floor_f:
            new_is_trusted = False

        if require_human_confirmation:
            # Human confirmation gate: auto observations stay in weak sidecar until human approves.
            may_promote = bool(new_is_trusted)
        else:
            # Un-gated legacy mode
            may_promote = bool(
                new_is_trusted or ((not omni_syn) and seed and new_conf >= floor_f)
            )

        if _find_slot(kept_trusted, new) is not None:
            continue

        weak_i = _find_slot(kept_weak, new)
        if weak_i is not None:
            old = kept_weak[weak_i]
            if old.get("trusted") is True and not is_omni_synthetic_source(
                str(old.get("source", ""))
            ):
                continue
            old_conf = annotation_confidence(old)
            if new_conf <= old_conf and not new_is_trusted:
                continue
            new["replaced_class_id"] = int(old.get("class_id", -1))
            if may_promote:
                new["trusted"] = True
                new["weak"] = False
                kept_trusted.append(new)
                kept_weak.pop(weak_i)
                seed = True
            else:
                new["trusted"] = False
                new["weak"] = True
                kept_weak[weak_i] = new
            continue

        if may_promote:
            new["trusted"] = True
            new["weak"] = False
            kept_trusted.append(new)
            seed = True
        else:
            new["trusted"] = False
            new["weak"] = True
            kept_weak.append(new)

    return kept_trusted, kept_weak, seed


def commit_split_labels(
    *,
    dataset_dir: Path,
    split: str,
    stem: str,
    label_path: Path,
    annotations: Sequence[Dict[str, Any]],
    floor: float = TRAIN_CONF_FLOOR,
    require_human_confirmation: bool = True,
    human_approved: Optional[bool] = None,
    done: Optional[str] = None,
) -> Dict[str, Any]:
    """Write trusted boxes into train/val txt; learn_* weak auto boxes also emit.

    Normal gather frames: only trusted (≥ floor / hand) enter the txt.
    Omni learn_* frames: weak layout boxes stay marked weak in review but are
    also written to the txt so Adım 2 can cull instead of starting from empty.
    """
    dataset_dir = Path(dataset_dir)
    label_path = Path(label_path)
    split_norm = "val" if str(split).lower() == "val" else "train"

    raw_anns = [dict(b) for b in (annotations or []) if isinstance(b, dict)]
    if human_approved is True:
        for b in raw_anns:
            b.setdefault("human_approved", True)

    # QA validation & sanitization: filter zero-area, fix bounds, deduplicate IoU>0.8
    qa_cleaned, _qa_issues = sanitize_boxes(raw_anns, file_path=str(label_path))

    cleaned = filter_hud_false_positives(qa_cleaned, stem=str(stem))
    prev_trusted, prev_weak, prev_meta = _load_state(
        dataset_dir, split_norm, stem, label_path, return_meta=True
    )
    trusted, weak, seed = merge_label_observation(
        prev_trusted,
        prev_weak,
        cleaned,
        has_seed=has_trusted_seed(dataset_dir),
        floor=float(floor),
        require_human_confirmation=require_human_confirmation,
    )
    # Final hard gate: never write sub-floor trusted boxes to train txt (unless hand_label).
    floor_f = float(floor)
    train_trusted = [
        b
        for b in trusted
        if "cx" in b
        and (
            bool(b.get("hand_label") is True)
            or bool(b.get("human_approved") is True)
            or bool(b.get("approved") is True)
            or bool(b.get("confirmed") is True)
            or annotation_confidence(b) >= floor_f
        )
    ]
    demoted = [b for b in trusted if b not in train_trusted]
    if demoted:
        for b in demoted:
            nb = dict(b)
            nb["trusted"] = False
            nb["weak"] = True
            weak.append(nb)
        trusted = train_trusted

    sanitized_trusted, _ = sanitize_boxes(trusted, file_path=str(label_path))
    emit = train_txt_boxes(sanitized_trusted, weak, stem=str(stem), floor=floor_f)
    _rewrite_label_txt(label_path, emit)

    resolved_human_approved = (
        human_approved if human_approved is not None else prev_meta.get("human_approved")
    )
    resolved_done = done if done is not None else prev_meta.get("done")
    resolved_approved_at = prev_meta.get("approved_at")
    if resolved_approved_at is None:
        for b in raw_anns:
            if b.get("approved_at") is not None:
                resolved_approved_at = b.get("approved_at")
                break

    review_payload: Dict[str, Any] = {
        "stem": stem,
        "split": split_norm,
        "floor": float(floor),
        "trusted": trusted,
        "weak": weak,
        "learn_auto_in_train": bool(is_learn_stem(stem) and weak),
    }
    if resolved_human_approved is not None:
        review_payload["human_approved"] = bool(resolved_human_approved)
    if resolved_done is not None:
        review_payload["done"] = str(resolved_done)
    if resolved_approved_at is not None:
        review_payload["approved_at"] = resolved_approved_at

    review = _review_path(dataset_dir, split_norm, stem)
    _atomic_write_json(review, review_payload)
    if seed:
        _mark_trusted_seed(dataset_dir)

    # Invalidate model state on label modification
    mark_model_stale(
        reason=f"Etiketler güncellendi: {stem} ({split_norm})",
        dataset_dir=dataset_dir,
    )

    result: Dict[str, Any] = {
        "trusted": trusted,
        "weak": weak,
        "train_box_count": len(emit),
        "weak_box_count": len(weak),
        "has_trusted_seed": seed,
        "review_path": str(review),
        "label_path": str(label_path),
        "floor": float(floor),
        "learn_auto_in_train": bool(is_learn_stem(stem) and weak),
    }
    if resolved_human_approved is not None:
        result["human_approved"] = bool(resolved_human_approved)
    if resolved_done is not None:
        result["done"] = str(resolved_done)
    if resolved_approved_at is not None:
        result["approved_at"] = resolved_approved_at
    return result


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _class_name_for_id(class_id: int) -> str:
    try:
        from dataset_factory.class_taxonomy import dataset_class_names

        names = dataset_class_names()
        if int(class_id) in names:
            return str(names[int(class_id)])
    except Exception:
        pass
    return f"class_{int(class_id)}"


def _normalize_requeue_box(
    raw: Dict[str, Any],
    *,
    default_conf: float = 0.58,
    emin_threshold: float = 0.85,
    stem: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict) or "cx" not in raw or "cy" not in raw:
        return None
    try:
        cid = int(raw.get("class_id", -1))
        cx = float(raw["cx"])
        cy = float(raw["cy"])
        w = float(raw.get("w", 0.0))
        h = float(raw.get("h", 0.0))
    except (TypeError, ValueError):
        return None
    has_explicit = bool(raw.get("has_explicit_conf") is True)
    conf = annotation_confidence(raw)
    if not has_explicit and conf <= 1e-6:
        conf = float(default_conf)
    name = str(raw.get("class_name") or raw.get("key") or _class_name_for_id(cid))

    raw_src = raw.get("source")
    if raw_src:
        src = str(raw_src).strip().lower()
    else:
        src = "explicit_txt_conf" if has_explicit else "dataset_requeue"

    is_omni_syn = is_omni_synthetic_source(src) or (raw.get("omni_synthetic") is True)
    has_hand = bool(
        raw.get("hand_label") is True
        or raw.get("human_approved") is True
        or raw.get("approved") is True
    )
    has_seed = _is_hand_or_gather_seed(raw)
    is_exemplar = bool(raw.get("exemplar_match") is True or "exemplar" in src)

    # Boxes loaded from .txt without explicit verified proof MUST be classified as trusted = False, weak = True
    if not has_explicit and not has_hand and not has_seed and not is_exemplar:
        is_emin = False
    elif has_hand or has_seed or is_exemplar:
        is_emin = True
    elif has_explicit and conf >= float(emin_threshold) and not is_omni_syn and src not in UNVERIFIED_HEURISTIC_SOURCES:
        is_emin = True
    else:
        is_emin = False

    out = dict(raw)
    out.update(
        {
            "class_id": cid,
            "class_name": name,
            "key": str(raw.get("key") or name),
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "confidence": round(float(conf), 4),
            "trusted": is_emin,
            "weak": not is_emin,
            "source": src,
            "requeued": True,
        }
    )
    return calibrate_box_confidence_and_trust(out, stem=stem, emin_threshold=emin_threshold)


def _dedupe_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        try:
            key = (
                int(b.get("class_id", -1)),
                round(float(b.get("cx", 0.0)), 4),
                round(float(b.get("cy", 0.0)), 4),
                round(float(b.get("w", 0.0)), 4),
                round(float(b.get("h", 0.0)), 4),
            )
        except (TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def requeue_labeled_dataset_for_review(
    dataset_dir: Union[str, Path],
    *,
    default_conf: float = 0.58,
    include_empty: bool = True,
    emin_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Put existing train/val samples back into the Adım 2 review queue.

    Scans ``images/{train,val}``, builds/refills ``review/{split}/{stem}.json`` with:
    - ``trusted``: boxes with explicit confidence >= emin_threshold (0.85), hand_label=True,
      exemplar_match=True, or gather seed templates (Emin).
    - ``weak``: boxes with confidence < emin_threshold (0.58 default for unverified .txt labels -> Şüpheli).
    Label ``.txt`` and images are left intact until user explicitly modifies them.
    Does not load image pixels (meta/json only).
    """
    root = Path(dataset_dir)
    written = 0
    refreshed = 0
    empty_n = 0
    box_n = 0
    trusted_n = 0
    weak_n = 0
    skipped = 0
    conf_f = float(default_conf)
    emin_th = float(emin_threshold)

    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        review_dir = root / "review" / split
        if not images_dir.is_dir():
            continue
        review_dir.mkdir(parents=True, exist_ok=True)
        try:
            image_paths = [
                p
                for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            ]
        except OSError:
            continue
        for img_path in image_paths:
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            review_path = review_dir / f"{stem}.json"

            trusted: List[Dict[str, Any]] = []
            weak_existing: List[Dict[str, Any]] = []
            existing_meta: Dict[str, Any] = {}
            is_already_approved = False
            if review_path.is_file():
                try:
                    payload = json.loads(review_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        for k in ("human_approved", "done", "approved_at"):
                            if k in payload:
                                existing_meta[k] = payload[k]
                        is_already_approved = bool(
                            payload.get("human_approved") is True
                            or str(payload.get("done", "")).lower() == "ok"
                        )
                        trusted = [
                            dict(b) for b in payload.get("trusted", []) if isinstance(b, dict)
                        ]
                        weak_existing = [
                            dict(b) for b in payload.get("weak", []) if isinstance(b, dict)
                        ]
                except Exception:
                    trusted, weak_existing = [], []

            legacy = _legacy_weak_from_txt(label_path)
            if is_already_approved:
                for b in trusted:
                    b.setdefault("human_approved", True)
                for b in legacy:
                    b.setdefault("human_approved", True)
            merged_src = list(weak_existing) + list(trusted) + list(legacy)
            all_boxes: List[Dict[str, Any]] = []
            for raw in merged_src:
                nb = _normalize_requeue_box(
                    raw,
                    default_conf=conf_f,
                    emin_threshold=emin_th,
                    stem=stem,
                )
                if nb is not None:
                    all_boxes.append(nb)
            all_boxes = _dedupe_boxes(all_boxes)

            if not all_boxes and not include_empty and not is_already_approved:
                skipped += 1
                continue

            trusted_boxes = [b for b in all_boxes if b.get("trusted") is True]
            weak_boxes = [b for b in all_boxes if b.get("trusted") is not True]

            existed = review_path.is_file()
            payload_out: Dict[str, Any] = {
                "stem": stem,
                "split": split,
                "floor": TRAIN_CONF_FLOOR,
                "trusted": trusted_boxes,
                "weak": weak_boxes,
                "requeued": True,
                "requeue_source": "labeled_dataset",
            }
            if is_already_approved:
                payload_out["was_previously_approved"] = True
                payload_out["human_approved"] = True
                payload_out["done"] = "ok"
            else:
                payload_out["human_approved"] = False
                payload_out["done"] = None
            if "approved_at" in existing_meta and existing_meta["approved_at"] is not None:
                payload_out["approved_at"] = existing_meta["approved_at"]
            try:
                _atomic_write_json(review_path, payload_out)
            except Exception:
                skipped += 1
                continue
            written += 1
            if existed:
                refreshed += 1
            box_n += len(all_boxes)
            trusted_n += len(trusted_boxes)
            weak_n += len(weak_boxes)
            if not all_boxes:
                empty_n += 1

    if written > 0:
        mark_model_stale(
            reason=f"Veri seti yeniden onay kuyruğuna alındı ({written} örnek)",
            dataset_dir=root,
        )

    return {
        "ok": True,
        "samples": written,
        "refreshed": refreshed,
        "empty_frames": empty_n,
        "boxes": box_n,
        "trusted_boxes": trusted_n,
        "weak_boxes": weak_n,
        "skipped": skipped,
        "default_conf": conf_f,
        "emin_threshold": emin_th,
        "include_empty": bool(include_empty),
    }


def proposals_from_loaded_onnx(frame: Any, model_path: str) -> List[Dict[str, Any]]:
    """Use the gatherion onnx only when the file actually loads. No color fallback."""
    try:
        from core.neural_audio_perception import ONNXYolo3DDetector
    except Exception:
        return []
    path = Path(model_path)
    if not path.is_file():
        return []
    try:
        detector = ONNXYolo3DDetector(model_path=str(path), prefer_gpu=False, conf_threshold=TRAIN_CONF_FLOOR)
    except Exception:
        return []
    if getattr(detector, "_ort_session", None) is None and getattr(detector, "_cv_net", None) is None:
        return []
    try:
        detections = detector.detect_3d_objects(frame)
    except Exception:
        return []
    if frame is None or not hasattr(frame, "shape"):
        return []
    img_h, img_w = int(frame.shape[0]), int(frame.shape[1])
    if img_w <= 0 or img_h <= 0:
        return []
    name_to_id = {
        "ore_vein": 3,
        "herb_flora": 12,
        "approaching_monster": 10,
    }
    proposals: List[Dict[str, Any]] = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        if str(det.get("backend", "")).startswith("fast_3d"):
            continue
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x, y, w, h = [float(v) for v in bbox[:4]]
        cname = str(det.get("class_name", ""))
        proposals.append(
            {
                "class_id": int(name_to_id.get(cname, 12)),
                "cx": (x + w * 0.5) / float(img_w),
                "cy": (y + h * 0.5) / float(img_h),
                "w": w / float(img_w),
                "h": h / float(img_h),
                "confidence": float(det.get("confidence", 0.0)),
                "source": "onnx_duzelt",
                "trusted": False,
            }
        )
    return proposals
