# -*- coding: utf-8 -*-
"""
Aion 2 Autonomous Dataset Factory - Dataset Curator, `data.yaml` Generator & Bootstrap Seeder
==============================================================================================
Manages the on-disk Ultralytics YOLOv8/v11 dataset directory (`dataset/`), generates
`dataset/data.yaml`, persists Unicode-safe image + `.txt` label pairs and annotated
bounding-box previews (`dataset/previews/`), maintains real-time `dataset/manifest_stats.json`
telemetry toward the 100,000-image target goal, and bootstraps the initial training/validation
dataset directly from `reference_screenshots/*.jpg` and `video_frames/*.jpg`.

Exclusive Owner: Dataset Factory Sub-Agent 5 (`dataset_factory/dataset_curator.py`)
"""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import errno
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.materials_db import classify_3d_material_patch, is_rejected_non_material
from core.material_scanner import DetectedMaterial, MaterialScanner

logger = logging.getLogger("DatasetFactory.DatasetCurator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Canonical 27-Class YOLO Taxonomy for Aion 2 Resource Nodes
# =============================================================================
# NOTE (D37/F-0793, F-0843): the historical name `YOLO_12_CLASS_NAMES` is kept
# for backward compatibility, but it does NOT hold 12 classes. The canonical
# taxonomy is 27 classes (0..26), identical to
# `dataset_factory.class_taxonomy.CANONICAL_NAMES`. The literal below is the
# full canonical table -- it is never mutated at import time, so labels stored
# on disk keep their meaning across imports.

TARGET_DATASET_GOAL: int = 100000

# Image extensions counted by rebuild_manifest_stats() -- shared with the
# incremental manifest fingerprint so both count exactly the same files.
MANIFEST_IMAGE_EXTS: frozenset = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

YOLO_12_CLASS_NAMES: Dict[int, str] = {
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

# Canonical alias for the same 27-class table (the `YOLO_12_` name is legacy).
YOLO_CLASS_NAMES: Dict[int, str] = YOLO_12_CLASS_NAMES

CLASS_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in YOLO_12_CLASS_NAMES.items()}

# High-contrast BGR colors for preview bounding boxes per class ID.
# Covers all 27 canonical classes (D37/F-0923); ids 12-26 used to fall back
# to a single default green.
CLASS_PREVIEW_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (235, 110, 215),   # herb_aria: violet/purple crystal
    1: (65, 75, 245),     # herb_mela: crimson berry bush
    2: (40, 225, 245),    # herb_targena: golden-yellow herb
    3: (30, 165, 255),    # ore_orichalcum: bronze/amber vein
    4: (250, 245, 220),   # gem_crystal: ice diamond white
    5: (55, 145, 95),     # tree_log: forest timber green-brown
    6: (200, 90, 180),    # herb_edana: deep magenta herb
    7: (120, 200, 255),   # herb_lumina: pale luminous gold
    8: (60, 60, 70),      # ore_obsidian: near-black volcanic ore
    9: (90, 230, 90),     # minimap_sprout: bright sprout green
    10: (40, 40, 200),    # monster_3d: dark red monster
    11: (30, 30, 160),    # minimap_monster_dot: deep red dot
    12: (150, 40, 180),   # crystal_blood: blood-crystal purple-red
    13: (255, 220, 0),    # skill_ui_prompt: bright cyan-blue
    14: (140, 140, 140),  # obstacle: neutral grey
    15: (180, 200, 120),  # path_ground: pale ground khaki
    16: (0, 60, 220),     # player_hp_bar: red HP bar
    17: (220, 120, 0),    # player_mp_bar: blue MP bar
    18: (0, 90, 200),     # target_hp_bar: orange-red target HP
    19: (255, 180, 60),   # skill_bar: light blue skill bar
    20: (0, 200, 0),      # skill_slot_ready: green ready slot
    21: (90, 90, 90),     # skill_slot_cooldown: dark grey cooldown
    22: (60, 60, 230),    # hp_potion: red potion
    23: (230, 120, 30),   # mp_potion: blue potion
    24: (0, 200, 220),    # buff_scroll: yellow scroll
    25: (0, 215, 255),    # kinah: gold currency
    26: (240, 240, 240),  # player_character: near-white player
}

# NOTE (D37/F-0791): values are CANONICAL 27-class ids directly -- the old
# table held ids from a retired 12-id layout and a legacy translation map
# then remapped them, which systematically mislabelled materials
# (e.g. "herb_calendula" -> targena, "aether_odyle" -> gem_crystal).
# Values here match `dataset_factory.class_taxonomy.class_id_for_material_text`
# wherever that function resolves the alias.
MATERIAL_ALIAS_TO_CLASS_ID: Dict[str, int] = {
    # Canonical names
    "herb_aria": 0,
    "herb_mela": 1,
    "fruit_mela": 1,
    "herb_targena": 2,
    "herb_calendula": 2,
    "herb_azpha": 2,
    "ore_orichalcum": 3,
    "ore_iron_silver": 3,
    "ore_iron_copper": 3,
    "gem_sapphire": 4,
    "gem_ruby": 4,
    "gem_diamond": 4,
    "gem_crystal": 4,
    "log_yggdrasil": 5,
    "log_timber": 5,
    "aether_odyle": 4,
    # Direct material names from AION2_MATERIALS_CATALOG and MaterialScanner
    "aria": 0,
    "fresh aria": 0,
    "taze aria": 0,
    "mela": 1,
    "fresh mela": 1,
    "taze mela": 1,
    "targena": 2,
    "fresh targena": 2,
    "targena otu": 2,
    "calendula": 2,
    "fresh calendula": 2,
    "azpha": 2,
    "fresh azpha": 2,
    "lura": 2,
    "orichalcum": 3,
    "orichalcum ore": 3,
    "fine orichalcum ore": 3,
    "pure orichalcum ore": 3,
    "radiant orichalcum ore": 3,
    "pure_aether_ore": 3,
    "iron ore": 3,
    "copper ore": 3,
    "silver ore": 3,
    "gold ore": 3,
    "titanium ore": 3,
    "mithril ore": 3,
    "adamantium": 3,
    "sapphire": 4,
    "sapphire gemstone": 4,
    "fine sapphire gemstone": 4,
    "pure sapphire gemstone": 4,
    "radiant sapphire gemstone": 4,
    "aquamarine": 4,
    "ruby": 4,
    "ruby gemstone": 4,
    "fine ruby gemstone": 4,
    "pure ruby gemstone": 4,
    "radiant ruby gemstone": 4,
    "garnet": 4,
    "diamond": 4,
    "diamond gemstone": 4,
    "fine diamond gemstone": 4,
    "pure diamond gemstone": 4,
    "radiant diamond gemstone": 4,
    "crystal": 4,
    "rose quartz": 4,
    "amethyst": 4,
    "topaz": 4,
    "emerald": 4,
    "opal": 4,
    "yggdrasil log": 5,
    "fine yggdrasil log": 5,
    "pure yggdrasil log": 5,
    "radiant yggdrasil log": 5,
    "asvata log": 5,
    "pine log": 5,
    "oak log": 5,
    "maple log": 5,
    "cedar log": 5,
    "odyle": 4,
    "aether": 4,
    "gatherable material": 12,
    "gatherable_node": 12,
    "gather_prompt": 13,
    "minimap_sprout": 9,
}


def resolve_yolo_class_id_and_name(raw_value: Union[int, str, None]) -> Tuple[int, str]:
    """
    Resolve an integer class ID, YOLO class key, or Aion 2 material name into
    a canonical `(class_id, class_name)` pair over the 27-class taxonomy (0..26).

    Raises `ValueError` for an integer id outside 0..26 instead of silently
    clipping it onto a different class (D37/F-0790). Unknown text falls back
    to the generic `crystal_blood` (12) world-material class only after the
    taxonomy lookup, the alias table, and the substring rules all miss.
    """
    if raw_value is None:
        return 0, YOLO_12_CLASS_NAMES[0]

    if isinstance(raw_value, (int, np.integer)):
        cid = int(raw_value)
        if cid not in YOLO_12_CLASS_NAMES:
            raise ValueError(f"class id {cid} is outside the canonical 0..26 taxonomy")
        return cid, YOLO_12_CLASS_NAMES[cid]

    text = str(raw_value).strip()
    if not text:
        return 0, YOLO_12_CLASS_NAMES[0]

    if text.isdigit():
        cid = int(text)
        if cid not in YOLO_12_CLASS_NAMES:
            raise ValueError(f"class id {cid} is outside the canonical 0..26 taxonomy")
        return cid, YOLO_12_CLASS_NAMES[cid]

    lowered = text.lower()
    if lowered in CLASS_NAME_TO_ID:
        cid = CLASS_NAME_TO_ID[lowered]
        return cid, YOLO_12_CLASS_NAMES[cid]

    try:
        from dataset_factory.class_taxonomy import class_id_for_material_text

        specific = class_id_for_material_text(lowered)
    except Exception:
        specific = None
    if specific is not None and specific in YOLO_12_CLASS_NAMES:
        return int(specific), YOLO_12_CLASS_NAMES[int(specific)]

    if "gatherable plant" in lowered or "gatherable material" in lowered or "world_material" in lowered:
        return 12, YOLO_12_CLASS_NAMES[12]
    if "skill" in lowered or "cooldown" in lowered or "vitality" in lowered:
        return 13, YOLO_12_CLASS_NAMES[13]
    if "obstacle" in lowered or "engel" in lowered or "cliff" in lowered:
        return 14, YOLO_12_CLASS_NAMES[14]
    if "path_ground" in lowered or "walking path" in lowered:
        return 15, YOLO_12_CLASS_NAMES[15]
    if "sprout" in lowered or "filiz" in lowered:
        return 9, YOLO_12_CLASS_NAMES[9]
    if "minimap_monster" in lowered or "monster_dot" in lowered:
        return 11, YOLO_12_CLASS_NAMES[11]
    if "monster" in lowered or "canavar" in lowered:
        return 10, YOLO_12_CLASS_NAMES[10]
    # Gather crests / [F]/[E] prompts are UI prompts in the 27-class taxonomy
    # (13); the old code returned 6/7/8, which are now herb_edana,
    # herb_lumina and ore_obsidian (D37/F-0791).
    if "crest" in lowered or "amblem" in lowered:
        return 13, YOLO_12_CLASS_NAMES[13]
    if "f_gather" in lowered or "f gather" in lowered:
        return 13, YOLO_12_CLASS_NAMES[13]
    if "e_gather" in lowered or "e_floating" in lowered or "floating" in lowered:
        return 13, YOLO_12_CLASS_NAMES[13]

    # Alias table values are canonical ids already -- no legacy remapping
    # (D37/F-0791: the retired `_LEGACY_MATERIAL_ID_TO_STABLE` translation
    # systematically mislabelled materials, so it was deleted).
    if lowered in MATERIAL_ALIAS_TO_CLASS_ID:
        cid = int(MATERIAL_ALIAS_TO_CLASS_ID[lowered])
        return cid, YOLO_12_CLASS_NAMES[cid]

    if "aria" in lowered:
        return 0, YOLO_12_CLASS_NAMES[0]
    if "mela" in lowered or "berry" in lowered or "fruit" in lowered:
        return 1, YOLO_12_CLASS_NAMES[1]
    if "targena" in lowered or "lura" in lowered or "calendula" in lowered or "azpha" in lowered or "herb" in lowered:
        return 2, YOLO_12_CLASS_NAMES[2]
    if "orichalcum" in lowered or "ore" in lowered or "iron" in lowered or "silver" in lowered or "copper" in lowered or "gold" in lowered:
        return 3, YOLO_12_CLASS_NAMES[3]
    if (
        "sapphire" in lowered
        or "aquamarine" in lowered
        or "ruby" in lowered
        or "garnet" in lowered
        or "diamond" in lowered
        or "crystal" in lowered
        or "quartz" in lowered
        or "amethyst" in lowered
        or "gem" in lowered
        or "odyle" in lowered
        or "aether" in lowered
    ):
        return 4, YOLO_12_CLASS_NAMES[4]
    if "log" in lowered or "tree" in lowered or "yggdrasil" in lowered or "asvata" in lowered or "wood" in lowered:
        return 5, YOLO_12_CLASS_NAMES[5]

    return 12, YOLO_12_CLASS_NAMES[12]


# =============================================================================
# Calibrated Trust Scoring & Unverified Heuristic Source Constants
# =============================================================================

try:
    from dataset_factory.label_trust import UNVERIFIED_HEURISTIC_SOURCES
except (ImportError, AttributeError):
    UNVERIFIED_HEURISTIC_SOURCES = frozenset({
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
    })


def calibrate_box_trust(
    box: Dict[str, Any],
    stem: str = "",
    emin_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Calibrate box confidence score and determine trusted (🟢 Emin) vs weak (🟠 Şüpheli).
    Delegates to dataset_factory.label_trust.calibrate_box_confidence_and_trust if available,
    otherwise strictly applies proof-based immunity, unverified heuristic caps (<= 0.68),
    and stem-mismatch penalties (<= 0.45).
    """
    try:
        from dataset_factory.label_trust import calibrate_box_confidence_and_trust as _lt_cal

        if callable(_lt_cal):
            return _lt_cal(box, stem=stem, emin_threshold=emin_threshold)
    except (ImportError, AttributeError):
        pass

    b = dict(box)
    source = str(b.get("source", "")).strip().lower()
    cid = int(b.get("class_id", -1))
    try:
        conf = float(b.get("confidence", b.get("conf", 0.0)))
    except (TypeError, ValueError):
        conf = 0.0

    # 1. Dokunulmazlık Kalkanı (Immunity Shield):
    # hand_label, human_approved, approved, or F/E gather prompt seed
    is_immune = (
        b.get("hand_label") is True
        or b.get("human_approved") is True
        or b.get("approved") is True
        or b.get("confirmed") is True
        or any(tok in source for tok in ("hand", "manual", "human"))
        or source.startswith("gather_crest")
        # F/E gather prompts are class 13 (skill_ui_prompt) in the 27-class
        # taxonomy; the old code used 7/8, now herb_lumina / ore_obsidian.
        or (cid == 13 and any(tok in source for tok in ("f_gather", "gather_prompt", "f_template")))
        or (cid == 13 and any(tok in source for tok in ("e_gather", "floating", "e_template")))
    )

    if is_immune:
        conf = max(0.90, conf)
        b["confidence"] = round(conf, 4)
        b["trusted"] = True
        b["weak"] = False
        return b

    # 2. Exemplar match (verified by ExemplarMemory)
    if b.get("exemplar_match") is True and conf >= emin_threshold:
        b["confidence"] = round(conf, 4)
        b["trusted"] = True
        b["weak"] = False
        return b

    # 3. UNVERIFIED_HEURISTIC_SOURCES:
    # Never trusted without human approval, strictly capped at <= 0.68
    try:
        from dataset_factory.label_trust import UNVERIFIED_HEURISTIC_SOURCES as _sources
    except (ImportError, AttributeError):
        _sources = UNVERIFIED_HEURISTIC_SOURCES

    is_unverified = (
        source in _sources
        or any(src in source for src in _sources)
        or source.startswith("omni_")
        or b.get("omni_synthetic") is True
    )

    if is_unverified:
        conf = min(conf, 0.68)
        b["confidence"] = round(conf, 4)
        b["trusted"] = False
        b["weak"] = True
        return b

    # 4. Stem-based material mismatch penalty. Material names map to the
    # canonical 27-class ids (D37/F-0842); the old table used the retired
    # 12-id layout (e.g. orichalcum=4, sapphire=6), which no longer matches
    # the taxonomy and mis-penalised correct boxes.
    stem_lower = str(stem or "").lower()
    canonical_materials = {
        0: ("aria",),
        1: ("mela",),
        2: ("targena", "calendula", "azpha", "lura"),
        3: ("orichalcum", "iron", "copper", "silver", "gold"),
        4: ("sapphire", "ruby", "diamond", "crystal", "gem"),
        5: ("tree", "log", "wood", "yggdrasil"),
        6: ("edana",),
        7: ("lumina",),
        8: ("obsidian",),
        9: ("sprout",),
        12: ("crystal_blood", "blood_crystal"),
    }
    stem_mat_cid = None
    for mcid, mnames in canonical_materials.items():
        if any(mname in stem_lower for mname in mnames):
            stem_mat_cid = mcid
            break

    if stem_mat_cid is not None and cid in canonical_materials and cid != stem_mat_cid:
        conf = min(conf, 0.45)
        b["confidence"] = round(conf, 4)
        b["trusted"] = False
        b["weak"] = True
        return b

    # 5. Standard confidence threshold check
    is_emin = conf >= float(emin_threshold)
    b["confidence"] = round(conf, 4)
    b["trusted"] = bool(is_emin)
    b["weak"] = not bool(is_emin)
    return b


def _classify_material_patch(
    patch: Any, min_confidence: float = 0.15
) -> Optional[Tuple[int, str, float]]:
    """
    Call `config.materials_db.classify_3d_material_patch` with its real
    signature (D37/F-0788). It takes `mesh_bgr` (not `bgr_patch`) and
    returns a single dict with `name` / `confidence` (not a list of
    matches with `score` / `top_k` / `min_score`). The old call passed the
    wrong keywords and indexed the returned dict like a list, so every
    call raised and was swallowed -- the classifier never ran.

    Returns `(class_id, class_name, confidence)` or `None` when the patch
    is empty, the call fails, or confidence is below `min_confidence`.
    """
    if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
        return None
    try:
        result = classify_3d_material_patch(
            mesh_bgr=patch,
            region_name="Pilgrim's Ridge",
        )
    except Exception as exc:
        logger.debug("classify_3d_material_patch failed: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    try:
        conf = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if conf < float(min_confidence):
        return None
    name = result.get("name") or result.get("base_material")
    if not name:
        return None
    cid, cname = resolve_yolo_class_id_and_name(str(name))
    return cid, cname, float(np.clip(conf, 0.05, 0.99))


# =============================================================================
# Unicode-Safe OpenCV Image I/O Helpers (Windows `selçuk` Path Compatible)
# =============================================================================

def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Safely read an image file from disk when the path contains non-ASCII / Turkish
    characters (such as `C:\\Users\\selçuk\\Desktop\\a2`) using `np.fromfile` + `cv2.imdecode`.
    Fast-paths ASCII paths directly via OpenCV C++ backend to release Python GIL.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        p_str = str(file_path)
        try:
            p_str.encode("ascii")
            img = cv2.imread(p_str, flags)
            if img is not None:
                return img
        except (UnicodeEncodeError, Exception):
            pass

        raw_bytes = np.fromfile(p_str, dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        return cv2.imdecode(raw_bytes, flags)
    except Exception as exc:
        logger.warning("Failed to decode image %s: %s", file_path, exc)
        return None



def _prepare_image_for_encode(image: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Return a contiguous uint8 array, or a rejection reason.

    None, empty, and 0-size arrays are rejected. Non-contiguous and non-uint8
    numeric arrays are repaired before `cv2.imencode` so a bad buffer is not
    reported as a path `Errno 22`.
    """
    if image is None:
        return None, "image is None"
    if not isinstance(image, np.ndarray):
        return None, f"image type {type(image).__name__} is not a numpy array"
    if image.ndim == 0 or image.size == 0 or any(int(dim) == 0 for dim in image.shape):
        return None, f"image is empty shape={tuple(int(dim) for dim in image.shape)}"

    if image.dtype == np.uint8:
        if image.flags.c_contiguous:
            return image, None
        return np.ascontiguousarray(image), None

    if np.issubdtype(image.dtype, np.floating):
        if not bool(np.isfinite(image).all()):
            return None, f"image dtype {image.dtype} contains non-finite values"
        peak = float(np.max(image))
        floor = float(np.min(image))
        scaled = image * 255.0 if floor >= 0.0 and peak <= 1.0 else image
        return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8)), None

    if np.issubdtype(image.dtype, np.integer):
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8)), None

    return None, f"image dtype {image.dtype} cannot be encoded"


def _write_encoded_bytes(out_path: Path, payload: bytes) -> None:
    """
    Write encoded bytes with `open(path, "wb")`.

    On Windows, truncating a file that is already memory-mapped raises
    `[Errno 22] Invalid argument: '<path>'`. Overwriting in place with
    `r+b` still succeeds in that case. `Errno 28` (disk full) is re-raised.
    """
    try:
        with open(out_path, "wb") as handle:
            written = handle.write(payload)
    except OSError as exc:
        if exc.errno != errno.EINVAL or not out_path.is_file():
            raise
        with open(out_path, "r+b") as handle:
            written = handle.write(payload)
            handle.truncate()
    if written != len(payload):
        raise OSError(
            errno.EIO,
            f"short image write {written} of {len(payload)} bytes",
            str(out_path),
        )


def imwrite_unicode(
    path: Union[str, Path],
    image: np.ndarray,
    jpeg_quality: int = 95,
) -> bool:
    """
    Encode with `cv2.imencode` and write the buffer with Python `open`.

    Does not call `cv2.imwrite` or `Path.write_bytes`. A disk-full `Errno 28`
    is logged and re-raised instead of being returned as success.
    """
    prepared, reject_reason = _prepare_image_for_encode(image)
    if prepared is None:
        logger.error("Rejected image write %s: %s", path, reject_reason)
        return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"

    params: List[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 50, 100))]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    try:
        ok, encoded = cv2.imencode(ext, prepared, params)
    except Exception as exc:
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            logger.error("Failed to write image %s: disk full: %s", out_path, exc)
            raise
        logger.error(
            "Rejected image encode for %s: %s (shape=%s dtype=%s)",
            out_path,
            exc,
            prepared.shape,
            prepared.dtype,
        )
        return False
    if not ok or encoded is None or int(encoded.size) == 0:
        logger.error(
            "Rejected image encode for %s: encoder returned an empty buffer (shape=%s dtype=%s)",
            out_path,
            prepared.shape,
            prepared.dtype,
        )
        return False

    payload = encoded.tobytes()
    try:
        _write_encoded_bytes(out_path, payload)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            logger.error("Failed to write image %s: disk full: %s", out_path, exc)
            raise
        # Windows preview lock (Errno 13 / EACCES): try temp+replace, then skip.
        locked = exc.errno in (
            errno.EACCES,
            getattr(errno, "EPERM", 1),
            13,
        ) or (
            isinstance(getattr(exc, "winerror", None), int)
            and int(exc.winerror) in (5, 32)
        )
        if locked:
            tmp_path = out_path.with_name(out_path.name + ".tmp_write")
            try:
                _write_encoded_bytes(tmp_path, payload)
                try:
                    tmp_path.replace(out_path)
                    return True
                except OSError:
                    logger.warning(
                        "Preview locked, skip write %s (%s)", out_path, exc
                    )
                    try:
                        if tmp_path.is_file():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    return False
            except OSError as retry_exc:
                logger.warning(
                    "Preview write skipped (locked) %s: %s", out_path, retry_exc
                )
                return False
        logger.error("Failed to write image %s: %s", out_path, exc)
        return False


# =============================================================================
# DatasetCurator Implementation
# =============================================================================

def _count_files_with_exts(directory: Path, exts: frozenset) -> Optional[int]:
    """Count regular files in `directory` whose suffix is in `exts` (case-insensitive).

    Missing directory counts as 0 (mirrors rebuild_manifest_stats); any
    other scan error returns None so the caller falls back to a full rebuild.
    """
    try:
        with os.scandir(directory) as it:
            return sum(
                1
                for entry in it
                if entry.is_file()
                and os.path.splitext(entry.name)[1].lower() in exts
            )
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def _labels_dir_fingerprint(directory: Path) -> Optional[Tuple[int, int, int]]:
    """`(count, total_size, max_mtime_ns)` of `*.txt` files in `directory`."""
    count = 0
    total_size = 0
    max_mtime_ns = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".txt"):
                    st = entry.stat()
                    count += 1
                    total_size += st.st_size
                    if st.st_mtime_ns > max_mtime_ns:
                        max_mtime_ns = st.st_mtime_ns
    except FileNotFoundError:
        return (0, 0, 0)
    except OSError:
        return None
    return (count, total_size, max_mtime_ns)


def _is_hand_box(b: Dict[str, Any]) -> bool:
    """El etiketi kutusu mu? (P1-17 / K-06: canlı yol deseni.)

    ``ui/yolo_factory_dialog.py::_relabel_single_item`` ile aynı predikat:
    ``hand_label is True`` veya ``source == "manual_draw"``. TXT
    uzlaştırması (`label_trust._load_state`) 5 alanlı TXT satırlarını
    ``hand_label=True`` + ``source="hand_label_txt"`` ile trusted'a taşıdığı
    için TXT-only el etiketleri de bu predikatla yakalanır.
    """
    return (
        isinstance(b, dict)
        and (b.get("hand_label") is True or b.get("source") == "manual_draw")
    )


def _turbo_box_identity_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Aynı kutu mu? (P1-17 / K-06: canlı yol deseni.)

    ``ui/yolo_factory_dialog.py::_box_identity_match`` ile birebir aynı
    kural: aynı ``class_id`` + ``cx/cy/w/h`` 0.02 tolerans içinde. P1-13
    (K-33) bunu tek ``same_box()`` tanımında birleştirecek; o güne kadar
    iki kopya bilerek aynı tutulur.
    """
    if a is b:
        return True
    try:
        cid_a = int(a.get("class_id", -2))
        cid_b = int(b.get("class_id", -3))
        if cid_a != cid_b:
            return False
    except (TypeError, ValueError):
        pass
    for key in ("cx", "cy", "w", "h"):
        if key not in a or key not in b:
            continue
        try:
            if abs(float(a[key]) - float(b[key])) > 0.02:
                return False
        except (TypeError, ValueError):
            return False
    return True


class DatasetCurator:
    """
    Manages the `dataset/` directory hierarchy, writes Ultralytics `dataset/data.yaml`,
    saves Unicode-safe `.jpg` + `.txt` pairs and bounding-box previews, maintains
    `dataset/manifest_stats.json`, and bootstraps the initial YOLO training & validation
    dataset from `reference_screenshots/*.jpg` and `video_frames/*.jpg`.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        target_goal: int = TARGET_DATASET_GOAL,
        class_names: Optional[Dict[int, str]] = None,
        val_ratio: float = 0.15,
        auto_labeler: Optional[Any] = None,
        augmentor: Optional[Any] = None,
        quality_gate: Optional[Any] = None,
    ) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        if dataset_dir is None:
            self.dataset_dir = self.project_root / "dataset"
        else:
            self.dataset_dir = Path(dataset_dir).resolve()

        self.target_goal: int = max(1, int(target_goal))
        self.class_names: Dict[int, str] = (
            {int(k): str(v) for k, v in class_names.items()}
            if class_names is not None
            else dict(YOLO_12_CLASS_NAMES)
        )
        self.val_ratio: float = float(np.clip(val_ratio, 0.05, 0.40))

        self.images_train_dir: Path = self.dataset_dir / "images" / "train"
        self.images_val_dir: Path = self.dataset_dir / "images" / "val"
        self.labels_train_dir: Path = self.dataset_dir / "labels" / "train"
        self.labels_val_dir: Path = self.dataset_dir / "labels" / "val"
        self.previews_dir: Path = self.dataset_dir / "previews"
        self.data_yaml_path: Path = self.dataset_dir / "data.yaml"
        self.manifest_stats_path: Path = self.dataset_dir / "manifest_stats.json"

        self._lock = threading.RLock()
        self._sample_seq: int = 0
        self._auto_labeler = auto_labeler
        self._augmentor = augmentor
        self._quality_gate = quality_gate
        self._scanner: Optional[MaterialScanner] = None
        # Incremental manifest state (D47): last exact manifest written by
        # rebuild_manifest_stats() plus the on-disk fingerprint it matches.
        # None until the first full rebuild -- the first save then rebuilds
        # once and every later save applies the O(1) delta instead.
        self._manifest_cache: Optional[Dict[str, Any]] = None
        self._manifest_fingerprint: Optional[Tuple[Any, ...]] = None

        self.ensure_dataset_structure()

    # -------------------------------------------------------------------------
    # 1. Directory & `data.yaml` Management
    # -------------------------------------------------------------------------
    def ensure_dataset_structure(self) -> Dict[str, Any]:
        """
        Create the complete Ultralytics YOLO directory structure and write `dataset/data.yaml`:
          - `dataset/images/train`
          - `dataset/images/val`
          - `dataset/labels/train`
          - `dataset/labels/val`
          - `dataset/previews`
          - `dataset/data.yaml`
          - `dataset/manifest_stats.json` (initialized or synced with disk)
        """
        with self._lock:
            self.images_train_dir.mkdir(parents=True, exist_ok=True)
            self.images_val_dir.mkdir(parents=True, exist_ok=True)
            self.labels_train_dir.mkdir(parents=True, exist_ok=True)
            self.labels_val_dir.mkdir(parents=True, exist_ok=True)
            self.previews_dir.mkdir(parents=True, exist_ok=True)

            yaml_content = self._build_data_yaml_text()
            # P0-5 (K-29) — TEK YAZICI KURALI: curator artık `data.yaml`'ı
            # "içerik farklıysa yaz" (eski D2 kuralı) ile EZMEZ. Dialog
            # `_ensure_data_yaml` mutlak yollu yaml'ı zaten yazar; curator
            # her örneklenmede / her `prepare_one_click_dataset` çağrısında
            # aynı dosyayı yeniden yazarsa dialog'un doğrusu ile curator'un
            # kopyası çekişir ve son yazan kazanır — 21:27 hatasının
            # kökeni buydu. Yeni kural: yaml YOKSA oluştur, mevcut
            # yaml'ın `path:`'i göreli / eksik / başka dizinse MUTLAK
            # yola çevirerek onar; `path:` bu dataset'in MUTLAK yoluysa
            # dosyaya DOKUNMA (başlık/nc/names farkı bile ezme sebebi
            # değildir — içerik dialog'un sorumluluğundadır).
            # Gerekçe (seçim): "curator hiç yaml yazmasın" seçilmedi —
            # curator dialogsuz da kullanılır (factory_controller
            # bootstrap, __init__), taze datasette yaml'ı birinin
            # oluşturması gerekir. "Mutlak yola çevir" tek başına da
            # yetmezdi — 0d727f6 zaten mutlak yazıyor, sorun EZME'ydi.
            # Bu kural ikisini birleştirir: oluştur + onar, asla ezme.
            try:
                existing = self.data_yaml_path.read_text(encoding="utf-8")
            except OSError:
                existing = None
            if existing is None or not self._existing_yaml_path_is_current_dataset(existing):
                self.data_yaml_path.write_text(yaml_content, encoding="utf-8")

            if not self.manifest_stats_path.exists():
                self.rebuild_manifest_stats()

            return {
                "dataset_dir": str(self.dataset_dir),
                "images_train_dir": str(self.images_train_dir),
                "images_val_dir": str(self.images_val_dir),
                "labels_train_dir": str(self.labels_train_dir),
                "labels_val_dir": str(self.labels_val_dir),
                "previews_dir": str(self.previews_dir),
                "data_yaml_path": str(self.data_yaml_path),
                "manifest_stats_path": str(self.manifest_stats_path),
                "nc": len(self.class_names),
                "names": dict(self.class_names),
            }

    @staticmethod
    def _yaml_path_value(yaml_text: str) -> Optional[str]:
        """Return the raw value of the top-level `path:` line, if any."""
        match = re.search(r"(?m)^path:\s*(.+?)\s*$", yaml_text or "")
        return match.group(1).strip().strip("'\"") if match else None

    def _existing_yaml_path_is_current_dataset(self, yaml_text: str) -> bool:
        """True iff the existing yaml's `path:` is THIS dataset's ABSOLUTE dir.

        P0-5 (K-29): only then does `ensure_dataset_structure` leave the
        file untouched. A relative `path:` (the old `path: .` yaml), a
        missing `path:` line, or an absolute path pointing elsewhere all
        return False, so the caller repairs the file with an absolute path.
        """
        raw = self._yaml_path_value(yaml_text)
        if raw is None or raw == "":
            return False
        candidate = Path(raw)
        if not candidate.is_absolute():
            return False
        try:
            return candidate.resolve() == self.dataset_dir.resolve()
        except OSError:
            return False

    def _build_data_yaml_text(self) -> str:
        """
        Build standard Ultralytics YOLOv8/v11 `data.yaml` content with forward-slash
        paths, `train: images/train`, `val: images/val`, a dynamic `nc:` equal to
        the number of canonical classes (27), and `names:` from `self.class_names`.
        """
        lines: List[str] = [
            "# Ultralytics YOLOv8 / YOLOv11 Dataset Configuration - Aion 2 Autonomous Dataset Factory",
            # B-09 DÜZELTME (2026-11-10, canlı kanıt): `path:` MUTLAK
            # yol yazar — eski göreli `.` YANLIŞTI. Ultralytics göreli
            # path'i yaml'ın dizinine göre DEĞİL, eğitim sürecinin
            # çalışma klasörüne (CWD, PC'de proje kökü) göre çözer;
            # `.` yazılınca eğitim proje kökü/images/val'i arar ve
            # düşer. Bu yaml da model.train(data=...) ile aynı
            # süreçte, proje kökünden tüketildiği için aynı hataya
            # düşerdi. Taşınabilirlik doğruluk karşısında ikincildir;
            # yaml her kurulumda yeniden üretilir. as_posix ile
            # Windows uyumlu forward-slash yazılır.
            f"path: {self.dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(self.class_names)}",
            "names:",
        ]
        for cid in sorted(self.class_names.keys()):
            lines.append(f"  {int(cid)}: {self.class_names[cid]}")
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------------------
    # Annotation Normalization & Validation
    # -------------------------------------------------------------------------
    def normalize_annotations(
        self,
        annotations: Optional[Sequence[Any]],
        img_w: int,
        img_h: int,
        default_material_hint: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Convert a sequence of heterogeneous annotation dicts, `DetectedMaterial` objects,
        or `(class_id, cx, cy, w, h)` tuples into validated, NMS-deduplicated YOLO
        annotation dicts with normalized `cx, cy, w, h` in `[0.001, 0.999]`.
        """
        if not annotations:
            return []

        img_w = max(1, int(img_w))
        img_h = max(1, int(img_h))
        normalized_list: List[Dict[str, Any]] = []

        for item in annotations:
            try:
                if isinstance(item, DetectedMaterial):
                    # O10: reddedilen-materyal filtresi eskiden yalnızca dict
                    # dalında vardı; tuple/DetectedMaterial yoluyla gelen
                    # aynı kirli girdi engellenmeden etiketleniyordu.
                    if item.material_type and is_rejected_non_material(
                        str(item.material_type)
                    ):
                        continue
                    bx, by, bw_px, bh_px = item.bbox
                    # Expand tight nameplate/crest box downward slightly to cover the 3D resource mesh
                    if item.source == "3d_world":
                        mesh_extra_h = int(round(bh_px * 0.85))
                        pad_x = int(round(bw_px * 0.12))
                        x1_px = max(0, bx - pad_x)
                        y1_px = max(0, by)
                        x2_px = min(img_w, bx + bw_px + pad_x)
                        y2_px = min(img_h, by + bh_px + mesh_extra_h)
                    else:
                        x1_px = max(0, bx)
                        y1_px = max(0, by)
                        x2_px = min(img_w, bx + max(4, bw_px))
                        y2_px = min(img_h, by + max(4, bh_px))

                    cx = ((x1_px + x2_px) * 0.5) / float(img_w)
                    cy = ((y1_px + y2_px) * 0.5) / float(img_h)
                    w = max(4.0, float(x2_px - x1_px)) / float(img_w)
                    h = max(4.0, float(y2_px - y1_px)) / float(img_h)
                    cid, cname = resolve_yolo_class_id_and_name(item.material_type or default_material_hint)
                    conf = float(item.confidence)
                    source = str(item.source)

                elif isinstance(item, (list, tuple)) and len(item) >= 5:
                    if isinstance(item[0], str) and is_rejected_non_material(item[0]):
                        continue
                    cid, cname = resolve_yolo_class_id_and_name(item[0])
                    cx = float(item[1])
                    cy = float(item[2])
                    w = float(item[3])
                    h = float(item[4])
                    conf = float(item[5]) if len(item) >= 6 else 0.0
                    source = "tuple_yolo"

                elif isinstance(item, dict):
                    raw_label = (
                        item.get("class_name")
                        or item.get("material_type")
                        or item.get("material_name")
                        or item.get("name")
                        or item.get("label")
                    )
                    if raw_label and is_rejected_non_material(str(raw_label)):
                        continue

                    if "class_id" in item and item["class_id"] is not None:
                        # D37/F-0790: the class_id field must be numeric --
                        # a non-numeric string here (e.g. "abc") used to
                        # resolve to the generic class 12 instead of being
                        # rejected. Names belong in class_name/material_type.
                        _raw_cid = item["class_id"]
                        if isinstance(_raw_cid, str) and not _raw_cid.strip().isdigit():
                            raise ValueError(f"class_id must be an integer, got {_raw_cid!r}")
                        cid, cname = resolve_yolo_class_id_and_name(_raw_cid)
                        if raw_label:
                            _, alias_name = resolve_yolo_class_id_and_name(raw_label)
                            if cid == 0 and alias_name != YOLO_12_CLASS_NAMES[0]:
                                cid, cname = resolve_yolo_class_id_and_name(raw_label)
                    elif "cls" in item and item["cls"] is not None:
                        _raw_cls = item["cls"]
                        if isinstance(_raw_cls, str) and not _raw_cls.strip().isdigit():
                            raise ValueError(f"cls must be an integer, got {_raw_cls!r}")
                        cid, cname = resolve_yolo_class_id_and_name(_raw_cls)
                    else:
                        cid, cname = resolve_yolo_class_id_and_name(raw_label or default_material_hint)

                    cx, cy, w, h = self._extract_dict_cxcywh(item, img_w=img_w, img_h=img_h)
                    conf = float(item.get("confidence", item.get("conf", item.get("score", 0.0))))
                    source = str(item.get("source", "auto_labeler"))
                else:
                    continue

            except (TypeError, ValueError) as _item_exc:
                # D37/F-0930: one broken confidence/coordinate must not
                # abort the whole normalize_annotations call -- skip it.
                logger.debug("Skipping broken annotation %r: %s", item, _item_exc)
                continue

            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
                continue
            if w <= 0.0 or h <= 0.0:
                continue

            # Clamp box corners strictly inside [0.001, 0.999]
            x1 = float(np.clip(cx - 0.5 * w, 0.001, 0.998))
            y1 = float(np.clip(cy - 0.5 * h, 0.001, 0.998))
            x2 = float(np.clip(cx + 0.5 * w, x1 + 0.002, 0.999))
            y2 = float(np.clip(cy + 0.5 * h, y1 + 0.002, 0.999))

            w_clamped = float(np.clip(x2 - x1, 0.002, 0.998))
            h_clamped = float(np.clip(y2 - y1, 0.002, 0.998))
            cx_clamped = float(np.clip(x1 + 0.5 * w_clamped, 0.002, 0.998))
            cy_clamped = float(np.clip(y1 + 0.5 * h_clamped, 0.002, 0.998))

            if w_clamped * h_clamped < 0.00002:
                continue

            entry = {
                "class_id": int(cid),
                "class_name": self.class_names.get(int(cid), cname),
                "cx": round(cx_clamped, 6),
                "cy": round(cy_clamped, 6),
                "w": round(w_clamped, 6),
                "h": round(h_clamped, 6),
                "confidence": round(float(np.clip(conf, 0.01, 1.0)), 4),
                "source": source,
            }
            if isinstance(item, dict):
                for k in (
                    "hand_label",
                    "human_approved",
                    "approved",
                    "confirmed",
                    "exemplar_match",
                    "exemplar_pos_sim",
                    "exemplar_neg_sim",
                    "exemplar_repair_sim",
                    "exemplar_repair_class_id",
                    "exemplar_repair_class_name",
                    "gather_key",
                    "key",
                    "trusted",
                    "weak",
                ):
                    if k in item:
                        entry[k] = item[k]
            normalized_list.append(entry)

        return self._apply_class_aware_nms(normalized_list, iou_threshold=0.65)

    @staticmethod
    def _extract_dict_cxcywh(ann: Dict[str, Any], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Extract normalized `(cx, cy, w, h)` from various dictionary coordinate schemas."""
        if (
            ("cx" in ann or "x_center" in ann)
            and ("cy" in ann or "y_center" in ann)
            and ("w" in ann or "width" in ann)
            and ("h" in ann or "height" in ann)
        ):
            cx = float(ann.get("cx", ann.get("x_center", 0.5)))
            cy = float(ann.get("cy", ann.get("y_center", 0.5)))
            w = float(ann.get("w", ann.get("width", 0.1)))
            h = float(ann.get("h", ann.get("height", 0.1)))
            if cx > 1.0 or cy > 1.0 or w > 1.0 or h > 1.0:
                cx /= float(img_w)
                cy /= float(img_h)
                w /= float(img_w)
                h /= float(img_h)
            return cx, cy, w, h

        for key in ("yolo_bbox", "bbox_norm", "normalized_bbox"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                vals = [float(v) for v in ann[key]]
                return vals[0], vals[1], vals[2], vals[3]

        for key in ("bbox_xyxy", "xyxy"):
            if key in ann and isinstance(ann[key], (list, tuple)) and len(ann[key]) == 4:
                x1, y1, x2, y2 = [float(v) for v in ann[key]]
                cx = ((x1 + x2) * 0.5) / float(img_w)
                cy = ((y1 + y2) * 0.5) / float(img_h)
                w = abs(x2 - x1) / float(img_w)
                h = abs(y2 - y1) / float(img_h)
                return cx, cy, w, h

        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4:
            b0, b1, b2, b3 = [float(v) for v in ann["bbox"]]
            if max(b0, b1, b2, b3) <= 1.0:
                return b0, b1, b2, b3
            # Pixel (x, y, w, h) from MaterialScanner / OpenCV boundingRect
            cx = (b0 + 0.5 * b2) / float(img_w)
            cy = (b1 + 0.5 * b3) / float(img_h)
            w = b2 / float(img_w)
            h = b3 / float(img_h)
            return cx, cy, w, h

        # D37/F-0792: no coordinates at all -- raise instead of inventing a
        # fake centre box (0.50, 0.54, 0.12, 0.14). Callers skip such items.
        raise ValueError(f"Annotation dictionary has no recognizable YOLO coordinates: {ann}")

    @staticmethod
    def _apply_class_aware_nms(
        annotations: List[Dict[str, Any]],
        iou_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Deduplicate overlapping boxes of the same class using Non-Maximum Suppression.

        Greedy NMS with the exact semantics of the previous O(n^2) Python
        loop: candidates are visited in (confidence desc, area desc) order
        (ties keep input order) and a candidate is dropped when its IoU
        with any already-kept box of the same class is strictly greater
        than ``iou_threshold``. The IoU matrix is built with one NumPy
        broadcast pass (up to 800 boxes per class) instead of ~n^2/2
        Python-level IoU evaluations; larger classes fall back to the
        same greedy order with one vectorized IoU row per kept box.
        """
        if len(annotations) <= 1:
            return annotations

        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            by_class.setdefault(int(ann["class_id"]), []).append(ann)

        kept: List[Dict[str, Any]] = []
        for cid in sorted(by_class.keys()):
            cands = by_class[cid]
            if len(cands) == 1:
                kept.extend(cands)
                continue

            cx = np.array([float(a["cx"]) for a in cands], dtype=np.float64)
            cy = np.array([float(a["cy"]) for a in cands], dtype=np.float64)
            w = np.array([float(a["w"]) for a in cands], dtype=np.float64)
            h = np.array([float(a["h"]) for a in cands], dtype=np.float64)
            conf = np.array(
                [float(a.get("confidence", 0.9)) for a in cands], dtype=np.float64
            )
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            areas = np.maximum((x2 - x1) * (y2 - y1), 1e-9)

            # (confidence desc, area desc); np.lexsort is stable, so full
            # ties keep input order -- same order as the old sorted() call.
            order = np.lexsort((-areas, -conf))
            thr = float(iou_threshold)
            n = len(cands)

            if n <= 800:
                # Full IoU matrix in one broadcast pass; the greedy loop
                # then costs one boolean-vector op per candidate.
                xs1, ys1, xs2, ys2 = x1[order], y1[order], x2[order], y2[order]
                ar = areas[order]
                inter_w = np.maximum(
                    0.0, np.minimum(xs2[:, None], xs2[None, :])
                    - np.maximum(xs1[:, None], xs1[None, :])
                )
                inter_h = np.maximum(
                    0.0, np.minimum(ys2[:, None], ys2[None, :])
                    - np.maximum(ys1[:, None], ys1[None, :])
                )
                inter = inter_w * inter_h
                iou = inter / (ar[:, None] + ar[None, :] - inter + 1e-9)
                suppressed = np.zeros(n, dtype=bool)
                kept_idx: List[int] = []
                for idx in range(n):
                    if suppressed[idx]:
                        continue
                    kept_idx.append(idx)
                    suppressed |= iou[idx] > thr
                kept.extend(cands[int(order[i])] for i in kept_idx)
                continue

            # Large class: same greedy order, one vectorized IoU row per
            # kept box (avoids an n^2 float64 matrix).
            remaining = order
            kept_idx = []
            while remaining.size:
                i = int(remaining[0])
                kept_idx.append(i)
                if remaining.size == 1:
                    break
                rest = remaining[1:]
                ix1 = np.maximum(x1[i], x1[rest])
                iy1 = np.maximum(y1[i], y1[rest])
                ix2 = np.minimum(x2[i], x2[rest])
                iy2 = np.minimum(y2[i], y2[rest])
                inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
                remaining = rest[iou <= thr]
            kept.extend(cands[i] for i in kept_idx)
        return kept

    # -------------------------------------------------------------------------
    # Preview Image Rendering
    # -------------------------------------------------------------------------
    def render_annotated_preview(
        self,
        frame: np.ndarray,
        annotations: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw clean, high-visibility bounding boxes and class/confidence badges onto
        a copy of `frame` for `dataset/previews/<name>_preview.jpg`.
        """
        preview = frame.copy()
        h, w = preview.shape[:2]

        for ann in annotations:
            # D37/F-1036: a raw dict missing cx/cy/w/h (or with non-numeric
            # values) must not crash the whole preview -- skip that box.
            try:
                cid = int(ann.get("class_id", 0))
                cname = str(ann.get("class_name", self.class_names.get(cid, f"class_{cid}")))
                conf = float(ann.get("confidence", 0.95))
                cx = float(ann["cx"])
                cy = float(ann["cy"])
                bw = float(ann["w"])
                bh = float(ann["h"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping unrenderable annotation %r: %s", ann, exc)
                continue

            x1 = max(0, int(round((cx - 0.5 * bw) * w)))
            y1 = max(0, int(round((cy - 0.5 * bh) * h)))
            x2 = min(w - 1, int(round((cx + 0.5 * bw) * w)))
            y2 = min(h - 1, int(round((cy + 0.5 * bh) * h)))

            color = CLASS_PREVIEW_COLORS_BGR.get(cid, (0, 230, 118))
            thickness = max(2, int(round(min(w, h) / 420.0)))
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)

            label_text = f"[{cid}] {cname} {conf:.2f}"
            font_scale = max(0.45, min(0.75, w / 2200.0))
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            bg_y1 = max(0, y1 - th - baseline - 6)
            bg_y2 = max(th + baseline + 6, y1)
            bg_x2 = min(w - 1, x1 + tw + 8)
            cv2.rectangle(preview, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(
                preview,
                label_text,
                (x1 + 4, bg_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

        return preview

    # -------------------------------------------------------------------------
    # 2. Sample Persistence (`save_labeled_sample`)
    # -------------------------------------------------------------------------
    def save_labeled_sample(
        self,
        frame: np.ndarray,
        annotations: Sequence[Any],
        split: str = "train",
        prefix: str = "sample",
        save_preview: bool = True,
        *,
        sample_name: Optional[str] = None,
        jpeg_quality: int = 95,
        update_manifest: bool = True,
        # O4: boş etiketle kayıt varsayılan olarak reddedilir — boş .txt,
        # YOLO eğitiminde "bu karede hiçbir şey yok" negatifi sayılır,
        # oysa gerçekte etiketleyici bulamamış olabilir. Bilinçli negatif
        # örnek için çağıran allow_empty=True geçmelidir.
        allow_empty: bool = False,
    ) -> Dict[str, Any]:
        """
        Save a single labeled sample pair (`dataset/images/<split>/<name>.jpg` and
        `dataset/labels/<split>/<name>.txt`), optionally render a bounding-box preview
        in `dataset/previews/<name>_preview.jpg`, and update `dataset/manifest_stats.json`.

        Args:
            frame: BGR `np.ndarray` image.
            annotations: Sequence of YOLO annotation dicts, `DetectedMaterial` items, or tuples.
            split: Target dataset split (`"train"`, `"val"`, or `"auto"`).
            prefix: Filename prefix when `sample_name` is not explicitly provided.
            save_preview: If True, writes annotated bounding-box preview to `dataset/previews/`.
            sample_name: Optional explicit stem name (without extension) for deterministic naming.
            jpeg_quality: JPEG compression quality (`50..100`, default `95`).
            update_manifest: Whether to immediately refresh `dataset/manifest_stats.json`.

        Returns:
            Dictionary containing saved file paths, box counts, normalized annotations,
            and updated manifest statistics.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("save_labeled_sample received an empty or invalid frame.")

        with self._lock:
            self._sample_seq += 1
            seq_num = self._sample_seq

            split_norm = str(split or "train").strip().lower()
            if split_norm not in ("train", "val"):
                # One deterministic schedule for the whole factory
                # (D37/F-0924): DatasetQualityGate.choose_split (Bresenham),
                # the same schedule bootstrap and the quality gate use.
                split_norm = self._choose_split(seq_num, val_ratio=self.val_ratio)

            img_h, img_w = frame.shape[:2]
            # O3: düşen kutu sayısı artık dönüşte raporlanır — eskiden
            # bozuk kutu sessizce atlanıyor, "kaydedildi" sayısı ile
            # "alınan" sayı arasındaki fark hiçbir yerde görünmüyordu.
            input_box_count = len(annotations) if annotations else 0
            norm_anns = self.normalize_annotations(annotations, img_w=img_w, img_h=img_h)
            dropped_box_count = max(0, input_box_count - len(norm_anns))

            # S-6202: run the normalized annotations through the quality
            # gate so the label flow actually enforces it (_get_quality_gate
            # was dead code: no caller). Degenerate/tiny boxes and
            # same-class duplicates (IoU > 0.65) are dropped before the
            # label file is written. Gate unavailable -> keep annotations
            # unchanged (fail-open; the load failure is logged by
            # _get_quality_gate) so a save is never lost to the gate.
            gate = self._get_quality_gate()
            if gate is not None and norm_anns:
                gated_anns = gate.validate_yolo_annotations(
                    [dict(a) for a in norm_anns]
                )
                if len(gated_anns) != len(norm_anns):
                    dropped_box_count += len(norm_anns) - len(gated_anns)
                    logger.debug(
                        "Quality gate dropped %d of %d annotations",
                        len(norm_anns) - len(gated_anns),
                        len(norm_anns),
                    )
                norm_anns = gated_anns

            if sample_name:
                clean_stem = re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(sample_name).strip())
            else:
                clean_prefix = re.sub(r"[^A-Za-z0-9_\-]", "_", str(prefix or "sample").strip())
                ts_ms = int(time.time() * 1000)
                clean_stem = f"{clean_prefix}_{ts_ms}_{seq_num:05d}"

            # Y1: kalibrasyon eskiden yalnızca toplu yeniden etiketlemede
            # (relabel) çalışıyordu; ana kayıt yolu kalibrasyonsuz
            # akıyordu. Artık her kayıt kalibre edilir: ölçülmemiş
            # sezgisel kutunun güveni cap'lenir, insan onayı dokunulmazlığı
            # korunur (calibrate_box_trust içinde).
            norm_anns = [
                calibrate_box_trust(dict(a), stem=clean_stem) for a in norm_anns
            ]

            # O4: etiketsiz kare boş .txt ile dataset'e girmez.
            if not norm_anns and not allow_empty:
                logger.warning(
                    "save_labeled_sample reddedildi (stem=%s): etiket yok — "
                    "boş .txt YOLO'da sahte negatif sayılır.",
                    clean_stem,
                )
                return {
                    "saved": False,
                    "reason": "empty_annotations",
                    "sample_name": clean_stem,
                    "split": split_norm,
                    "image_path": "",
                    "label_path": "",
                    "preview_path": None,
                    "box_count": 0,
                    "weak_box_count": 0,
                    "input_box_count": input_box_count,
                    "dropped_box_count": dropped_box_count,
                    "annotations": [],
                    "weak_annotations": [],
                    "has_trusted_seed": False,
                    "manifest_stats": self.get_manifest_stats(),
                }

            images_dir = self.images_val_dir if split_norm == "val" else self.images_train_dir
            labels_dir = self.labels_val_dir if split_norm == "val" else self.labels_train_dir

            image_path = images_dir / f"{clean_stem}.jpg"
            label_path = labels_dir / f"{clean_stem}.txt"
            preview_path = self.previews_dir / f"{clean_stem}_preview.jpg"

            # D47: the incremental manifest path only applies to brand-new
            # files. Overwriting a pre-existing stem changes the old label's
            # box counts, so that save always takes the full rebuild.
            allow_incremental = not (image_path.exists() or label_path.exists())

            from dataset_factory.label_trust import commit_split_labels

            # D37/F-0836: commit labels BEFORE writing the image. If the
            # image write then fails, the label/review files are removed
            # again so no orphan label (or orphan image) is left behind.
            trust_info = commit_split_labels(
                dataset_dir=self.dataset_dir,
                split=split_norm,
                stem=clean_stem,
                label_path=label_path,
                annotations=norm_anns,
            )

            ok_img = imwrite_unicode(image_path, frame, jpeg_quality=jpeg_quality)
            if not ok_img:
                for _stale in (label_path, Path(str(trust_info.get("review_path", "")))):
                    try:
                        if str(_stale):
                            _stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise RuntimeError(f"Failed to write dataset image to {image_path}")
            trusted_anns = list(trust_info.get("trusted", []))
            weak_anns = list(trust_info.get("weak", []))

            saved_preview_path: Optional[str] = None
            if save_preview:
                preview_existed = preview_path.exists()
                preview_img = self.render_annotated_preview(frame, norm_anns)
                if imwrite_unicode(preview_path, preview_img, jpeg_quality=90):
                    saved_preview_path = str(preview_path)
                    if preview_existed:
                        allow_incremental = False
                else:
                    # D3: önizleme yazılamazsa eskiden sessizce yok
                    # sayılıyordu; Adım 2 insan denetimi eksik kalıyordu.
                    logger.warning(
                        "save_labeled_sample önizleme yazılamadı (stem=%s): %s",
                        clean_stem,
                        preview_path,
                    )

            # D47: incremental refresh (delta on the cached manifest) when
            # the fingerprint proves nothing else changed on disk; otherwise
            # the exact full rebuild. Bulk callers already pass
            # update_manifest=False and rebuild once at the end.
            manifest = (
                self._refresh_manifest_after_save(
                    split=split_norm,
                    image_path=image_path,
                    label_path=label_path,
                    preview_path=preview_path if saved_preview_path is not None else None,
                    allow_incremental=allow_incremental,
                )
                if update_manifest
                else self.get_manifest_stats()
            )

            # P2-21 (F061-K4): commit 0 kutu yazdıysa bunu logla — kare
            # diske/review'a kaydedildi ama train TXT boş; `saved=True`
            # karenin persist edildiğini söyler, eğitime girdiğini değil.
            if int(trust_info.get("train_box_count", 0) or 0) <= 0:
                logger.warning(
                    "save_labeled_sample: commit 0 kutu yazdı (stem=%s, dropped=%s) — "
                    "kare review'a kaydedildi ama train TXT'ye kutu girmedi",
                    clean_stem,
                    trust_info.get("dropped"),
                )

            return {
                "saved": True,
                "sample_name": clean_stem,
                "split": split_norm,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "preview_path": saved_preview_path,
                "box_count": len(trusted_anns),
                "weak_box_count": len(weak_anns),
                # P2-21 (F061-K4 / K-07): commit'in train TXT'ye yazdığı
                # gerçek kutu sayısı okunur — 0 ise kare review'a
                # kaydedildi ama eğitime 0 kutu girdi; çağıran bunu
                # görür (factory_controller sayacı `saved`'e bakar;
                # 0-kutu kare orada ayrıca sayılmaz — bkz. açık nokta).
                "train_box_count": int(trust_info.get("train_box_count", 0) or 0),
                "commit_dropped": int(trust_info.get("dropped", 0) or 0),
                "commit_trusted_in": int(trust_info.get("trusted_in", 0) or 0),
                "commit_trusted_out": int(trust_info.get("trusted_out", 0) or 0),
                "input_box_count": input_box_count,
                "dropped_box_count": dropped_box_count,
                "annotations": trusted_anns,
                "weak_annotations": weak_anns,
                "has_trusted_seed": bool(trust_info.get("has_trusted_seed")),
                "manifest_stats": manifest,
            }

    # -------------------------------------------------------------------------
    # Manifest & Dataset Telemetry (`manifest_stats.json`)
    # -------------------------------------------------------------------------
    def rebuild_manifest_stats(self) -> Dict[str, Any]:
        """
        Scan `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`,
        `dataset/labels/val`, and `dataset/previews` on disk and write an exact,
        up-to-date `dataset/manifest_stats.json`.
        """
        with self._lock:
            valid_img_exts = MANIFEST_IMAGE_EXTS

            train_images = (
                sorted(
                    [
                        p
                        for p in self.images_train_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in valid_img_exts
                    ]
                )
                if self.images_train_dir.exists()
                else []
            )
            val_images = (
                sorted(
                    [
                        p
                        for p in self.images_val_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in valid_img_exts
                    ]
                )
                if self.images_val_dir.exists()
                else []
            )
            preview_images = (
                [
                    p
                    for p in self.previews_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in valid_img_exts
                ]
                if self.previews_dir.exists()
                else []
            )

            class_counts: Dict[str, int] = {
                cname: 0 for _, cname in sorted(self.class_names.items())
            }
            class_id_counts: Dict[str, int] = {
                str(cid): 0 for cid in sorted(self.class_names.keys())
            }
            total_boxes = 0
            train_boxes = 0
            val_boxes = 0

            for split_tag, lbl_dir in (("train", self.labels_train_dir), ("val", self.labels_val_dir)):
                if not lbl_dir.exists():
                    continue
                for lbl_file in sorted(lbl_dir.glob("*.txt")):
                    try:
                        raw_text = lbl_file.read_text(encoding="utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not raw_text:
                        continue
                    for line in raw_text.splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        # A23: strict integer class token, consistent with
                        # dataset_validator.validate_yolo_file (F-0851).
                        # int(float(...)) silently truncated "1.9" -> 1, so
                        # an invalid line was counted as a real class-1 box
                        # in the manifest while the validator rejected it.
                        if not re.fullmatch(r"[+-]?\d+", parts[0].strip()):
                            continue
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        cname = self.class_names.get(cid, f"class_{cid}")
                        class_counts[cname] = class_counts.get(cname, 0) + 1
                        class_id_counts[str(cid)] = class_id_counts.get(str(cid), 0) + 1
                        total_boxes += 1
                        if split_tag == "train":
                            train_boxes += 1
                        else:
                            val_boxes += 1

            train_count = len(train_images)
            val_count = len(val_images)
            total_images = train_count + val_count
            progress_pct = round(min(100.0, (total_images / float(self.target_goal)) * 100.0), 2)

            stats_payload: Dict[str, Any] = {
                "total_images": total_images,
                "target_goal": self.target_goal,
                "progress_pct": progress_pct,
                "train_count": train_count,
                "val_count": val_count,
                "preview_count": len(preview_images),
                "total_boxes": total_boxes,
                "train_boxes": train_boxes,
                "val_boxes": val_boxes,
                "nc": len(self.class_names),
                "class_counts": class_counts,
                "class_id_counts": class_id_counts,
                "data_yaml_path": str(self.data_yaml_path),
                "dataset_dir": str(self.dataset_dir),
                "updated_at_epoch": round(time.time(), 3),
                "updated_at_iso": datetime.now(timezone.utc).isoformat(),
            }

            self.manifest_stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_stats_path.write_text(
                json.dumps(stats_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # D47: remember the exact manifest + the disk state it matches,
            # so save_labeled_sample() can refresh it incrementally.
            self._manifest_cache = copy.deepcopy(stats_payload)
            self._manifest_fingerprint = self._compute_manifest_fingerprint()
            return stats_payload

    # -------------------------------------------------------------------------
    # Incremental manifest refresh (D47)
    # -------------------------------------------------------------------------
    def _compute_manifest_fingerprint(self) -> Optional[Tuple[Any, ...]]:
        """
        Cheap on-disk fingerprint of exactly the files `rebuild_manifest_stats`
        counts: per-split image counts, per-split label (count, total size,
        max mtime_ns), preview count. `stat` on label files (not content
        reads) is what keeps this ~8x cheaper than a full rebuild, and it
        still catches in-place label edits (size/mtime change) that a
        count-only scan would miss. Returns None when the disk cannot be
        scanned -- callers then fall back to a full rebuild.
        """
        try:
            img_train = _count_files_with_exts(self.images_train_dir, MANIFEST_IMAGE_EXTS)
            img_val = _count_files_with_exts(self.images_val_dir, MANIFEST_IMAGE_EXTS)
            lbl_train = _labels_dir_fingerprint(self.labels_train_dir)
            lbl_val = _labels_dir_fingerprint(self.labels_val_dir)
            prev = _count_files_with_exts(self.previews_dir, MANIFEST_IMAGE_EXTS)
        except OSError:
            return None
        if None in (img_train, img_val, lbl_train, lbl_val, prev):
            return None
        return (img_train, img_val, lbl_train, lbl_val, prev)

    @staticmethod
    def _count_label_file_boxes(label_path: Path) -> Optional[List[int]]:
        """Parse one label txt exactly like rebuild_manifest_stats() does."""
        try:
            raw_text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return None
        cids: List[int] = []
        if not raw_text:
            return cids
        for line in raw_text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Y5: artımlı sayım eskiden `int(float(...))` kullanıyordu —
            # "1.9" → 1 sayılıyordu; tam rebuild katı regex kullanır.
            # İki sayım aynı kuralı kullanır artık.
            if not re.fullmatch(r"[+-]?\d+", parts[0].strip()):
                continue
            try:
                cids.append(int(parts[0]))
            except ValueError:
                continue
        return cids

    def _refresh_manifest_after_save(
        self,
        *,
        split: str,
        image_path: Path,
        label_path: Path,
        preview_path: Optional[Path],
        allow_incremental: bool,
    ) -> Dict[str, Any]:
        """
        Refresh `manifest_stats.json` after one `save_labeled_sample`.

        Fast path: when the cached manifest is known to match disk and the
        fingerprint after this save is exactly the cached fingerprint plus
        this one new image/label/preview, apply the delta (counts read back
        from the label file just written) and rewrite the JSON -- no full
        rescan. Any deviation (no cache, a pre-existing file overwritten,
        an external add/delete/edit, an unscannable directory) falls back
        to the exact `rebuild_manifest_stats()`.

        Exactness note: the fingerprint covers label size+mtime, so an
        out-of-band in-place label edit is detected and triggers a rebuild.
        The residual blind spot is an external process concurrently adding
        AND removing label files whose total size and max mtime net out to
        the same values inside one save window; that combination is not
        detected and the manifest then needs a manual rebuild. The live
        factory writers (this curator, relabel_dataset_batch) all end with
        a full rebuild, which re-syncs the cache.
        """
        with self._lock:
            if (
                not allow_incremental
                or self._manifest_cache is None
                or self._manifest_fingerprint is None
            ):
                return self.rebuild_manifest_stats()

            try:
                image_path.stat()
                label_stat = label_path.stat()
                preview_stat = preview_path.stat() if preview_path is not None else None
            except OSError:
                return self.rebuild_manifest_stats()

            cached = self._manifest_fingerprint
            img_train, img_val, lbl_train, lbl_val, prev = cached
            preview_delta = 1 if preview_stat is not None else 0
            if split == "val":
                expected = (
                    img_train,
                    img_val + 1,
                    lbl_train,
                    (
                        lbl_val[0] + 1,
                        lbl_val[1] + label_stat.st_size,
                        max(lbl_val[2], label_stat.st_mtime_ns),
                    ),
                    prev + preview_delta,
                )
            else:
                expected = (
                    img_train + 1,
                    img_val,
                    (
                        lbl_train[0] + 1,
                        lbl_train[1] + label_stat.st_size,
                        max(lbl_train[2], label_stat.st_mtime_ns),
                    ),
                    lbl_val,
                    prev + preview_delta,
                )

            actual = self._compute_manifest_fingerprint()
            if actual is None or actual != expected:
                return self.rebuild_manifest_stats()

            new_cids = self._count_label_file_boxes(label_path)
            if new_cids is None:
                return self.rebuild_manifest_stats()

            stats = copy.deepcopy(self._manifest_cache)
            if split == "val":
                stats["val_count"] = int(stats.get("val_count", 0)) + 1
            else:
                stats["train_count"] = int(stats.get("train_count", 0)) + 1
            stats["total_images"] = int(stats.get("total_images", 0)) + 1
            for cid in new_cids:
                cname = self.class_names.get(cid, f"class_{cid}")
                stats["class_counts"][cname] = stats["class_counts"].get(cname, 0) + 1
                stats["class_id_counts"][str(cid)] = stats["class_id_counts"].get(str(cid), 0) + 1
                stats["total_boxes"] = int(stats.get("total_boxes", 0)) + 1
                if split == "val":
                    stats["val_boxes"] = int(stats.get("val_boxes", 0)) + 1
                else:
                    stats["train_boxes"] = int(stats.get("train_boxes", 0)) + 1
            if preview_stat is not None:
                stats["preview_count"] = int(stats.get("preview_count", 0)) + 1
            stats["progress_pct"] = round(
                min(100.0, (int(stats["total_images"]) / float(self.target_goal)) * 100.0), 2
            )
            stats["updated_at_epoch"] = round(time.time(), 3)
            stats["updated_at_iso"] = datetime.now(timezone.utc).isoformat()

            self.manifest_stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_stats_path.write_text(
                json.dumps(stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._manifest_cache = copy.deepcopy(stats)
            self._manifest_fingerprint = actual
            return stats

    def get_manifest_stats(self) -> Dict[str, Any]:
        """Return the current dataset manifest statistics, rebuilding if missing."""
        with self._lock:
            if self.manifest_stats_path.is_file():
                try:
                    return json.loads(self.manifest_stats_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return self.rebuild_manifest_stats()

    # -------------------------------------------------------------------------
    # Auto-Labeling Helper (External `YOLOAutoLabeler` + Built-in Multi-Stage Fallback)
    # -------------------------------------------------------------------------
    def _choose_split(self, sample_index: int, val_ratio: Optional[float] = None) -> str:
        """
        Single deterministic train/val schedule for the whole factory
        (D37/F-0924): `DatasetQualityGate.choose_split` (Bresenham), so
        `save_labeled_sample`, `bootstrap_from_existing_frames` and the
        quality gate no longer run three different split calendars.
        """
        ratio = self.val_ratio if val_ratio is None else float(val_ratio)
        try:
            from dataset_factory.quality_dedup_gate import DatasetQualityGate

            return DatasetQualityGate.choose_split(int(sample_index), val_ratio=ratio)
        except Exception:
            # O6/B-12: yedek takvim eskiden sessizce devreye giriyordu
            # (log yok, %14,3 val — ana takvim %15'ten farklı). Artık
            # uyarı bırakır; aynı kare farklı koşularda farklı split'e
            # düşerse sebebi logda görünür.
            logger.warning(
                "DatasetQualityGate yüklenemedi; split yedek takvime "
                "(%%7) düştü — ana Bresenham takviminden farklı oran verir."
            )
            return "val" if (max(0, int(sample_index)) % 7 == 0) else "train"

    def _get_material_scanner(self) -> MaterialScanner:
        if self._scanner is None:
            # D37/F-0925: never resolve templates relative to the CWD; the
            # scanner always uses <project_root>/templates.
            self._scanner = MaterialScanner(
                templates_dir=str(self.project_root / "templates")
            )
        return self._scanner

    def _get_auto_labeler(self) -> Optional[Any]:
        """Lazily load `YOLOAutoLabeler` from `dataset_factory.yolo_auto_labeler` if available."""
        if self._auto_labeler is not None:
            return self._auto_labeler
        try:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            self._auto_labeler = YOLOAutoLabeler()
            return self._auto_labeler
        except Exception as exc:
            logger.warning("Failed to initialize YOLOAutoLabeler: %s", exc)
            return None

    def _get_augmentor(self) -> Optional[Any]:
        """Lazily load `SyntheticYOLOAugmentor` from `dataset_factory.synthetic_augmentor`."""
        if self._augmentor is not None:
            return self._augmentor
        try:
            from dataset_factory.synthetic_augmentor import SyntheticYOLOAugmentor

            self._augmentor = SyntheticYOLOAugmentor(seed=42)
            return self._augmentor
        except Exception as exc:
            # D37/F-0926: a failed lazy load must be visible; silently
            # returning None used to drop the pipeline to a fallback
            # without anyone noticing.
            logger.warning("Failed to initialize SyntheticYOLOAugmentor: %s", exc)
            return None

    def _get_quality_gate(self) -> Optional[Any]:
        """Lazily load `QualityGate` from `dataset_factory.quality_gate`."""
        if self._quality_gate is not None:
            return self._quality_gate
        try:
            from dataset_factory.quality_gate import QualityGate

            self._quality_gate = QualityGate(
                min_hamming_distance=3,
                min_sharpness_threshold=15.0,
                val_ratio=self.val_ratio,
            )
            return self._quality_gate
        except Exception as exc:
            # D37/F-0926: see _get_augmentor -- log, don't swallow.
            logger.warning("Failed to initialize QualityGate: %s", exc)
            return None

    def auto_label_frame(
        self,
        frame: np.ndarray,
        source_path: Optional[Union[str, Path]] = None,
        fast_mode: bool = False,
        full_scan: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Auto-label a gameplay screenshot or video frame into normalized YOLO annotations.

        Pipeline:
          1. If `YOLOAutoLabeler` (`dataset_factory.auto_labeler`) is available, invoke it first.
          2. Run `MaterialScanner.scan_frame_for_materials(frame)` to detect 3D winged gather
             crests, cyan/teal resource nameplates, floating `[E]`/`[F]` gather icons,
             `"F Gather"` interactive prompts, and minimap 3-leaf sprout icons.
          3. Apply filename & visual patch classification (`classify_3d_material_patch`) so
             screenshots like `aria_*.jpg`, `mela_*.jpg`, `pilgrims_ridge_*.jpg`, and
             `video_frames/*.jpg` always produce accurate, high-confidence bounding boxes.

        ``full_scan=True`` Stage C (HUD) / Stage D (yol/engel) stem kapısını
        atlar — YOLOAutoLabeler'a geçirilir. Yalnızca öğrenme / yeniden
        etiketleme bağlamları geçirir; varsayılan False (W29 kilidi korunur).
        """
        if frame is None or frame.size == 0:
            return []

        img_h, img_w = frame.shape[:2]
        stem_lower = Path(source_path).stem.lower() if source_path is not None else ""

        # Hint from filename when available
        material_hint = ""
        if "aria" in stem_lower:
            material_hint = "herb_aria"
        elif "mela" in stem_lower:
            material_hint = "herb_mela"
        elif "targena" in stem_lower:
            material_hint = "herb_targena"
        elif "calendula" in stem_lower:
            material_hint = "herb_calendula"
        elif "sapphire" in stem_lower:
            material_hint = "gem_sapphire"
        elif "ruby" in stem_lower:
            material_hint = "gem_ruby"
        elif "diamond" in stem_lower:
            material_hint = "gem_diamond"
        elif "sprout" in stem_lower or "minimap" in stem_lower:
            # Y2 (dilim 03): eskiden "aether_odyle" (sınıf 4, gem_crystal)
            # yazılıyordu — minimap filizi (doğru sınıf 9, minimap_sprout)
            # sistematik olarak kristal diye etiketleniyordu.
            material_hint = "minimap_sprout"
        elif "orichalcum" in stem_lower:
            material_hint = "ore_orichalcum"

        collected: List[Dict[str, Any]] = []

        # Stage 1: External YOLOAutoLabeler if present
        labeler = self._get_auto_labeler()
        if labeler is not None:
            try:
                if hasattr(labeler, "auto_label_frame"):
                    ext_res = labeler.auto_label_frame(
                        frame,
                        fast_mode=fast_mode,
                        source_path=source_path,
                        full_scan=full_scan,
                    )
                elif hasattr(labeler, "label_frame"):
                    ext_res = labeler.label_frame(frame)
                elif callable(labeler):
                    ext_res = labeler(frame)
                else:
                    ext_res = []

                if isinstance(ext_res, dict):
                    ext_anns = ext_res.get("annotations", ext_res.get("labels", []))
                elif isinstance(ext_res, list):
                    ext_anns = ext_res
                else:
                    ext_anns = []

                if ext_anns:
                    norm_ext = self.normalize_annotations(
                        ext_anns,
                        img_w=img_w,
                        img_h=img_h,
                        default_material_hint=material_hint,
                    )
                    collected.extend(norm_ext)
            except Exception as exc:
                logger.debug("External YOLOAutoLabeler skipped on %s: %s", stem_lower, exc)

        # Stage 2: Built-in MaterialScanner (run only as fallback if Stage 1 found nothing)
        if not collected:
            try:
                scanner = self._get_material_scanner()
                scanner_dets = scanner.scan_frame_for_materials(frame)
                if scanner_dets:
                    # D37/F-0837: the detector's own classification stands.
                    # The old code let a filename hint ("herb_aria" /
                    # "herb_mela") overwrite the detector finding and force
                    # a wrong class; the hint is only a fallback for
                    # annotations that carry no material of their own.
                    for det in scanner_dets:
                        collected.extend(
                            self.normalize_annotations([det], img_w=img_w, img_h=img_h)
                        )
            except Exception as exc:
                logger.debug("MaterialScanner fallback failed on %s: %s", stem_lower, exc)

        # Stage 3: Salient 3D Resource & HSV Patch Classifier Fallback if frame had no scanner hits
        if not collected:
            fallback_anns = self._fallback_salient_resource_boxes(
                frame=frame,
                stem_lower=stem_lower,
                material_hint=material_hint,
            )
            collected.extend(fallback_anns)

        return self._apply_class_aware_nms(collected, iou_threshold=0.65)

    def _fallback_salient_resource_boxes(
        self,
        frame: np.ndarray,
        stem_lower: str = "",
        material_hint: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Localize salient gatherable resource nodes in 3D viewport crops or full frames
        using HSV chromaticity + `classify_3d_material_patch` when text nameplates are
        occluded or cropped (such as `crop_frame_*.jpg` in `video_frames/`).
        """
        img_h, img_w = frame.shape[:2]

        # Focus on the central 3D gameplay zone (avoiding HUD borders on full frames)
        if img_w >= 1000 and img_h >= 600:
            rx1, ry1 = int(img_w * 0.16), int(img_h * 0.18)
            rx2, ry2 = int(img_w * 0.82), int(img_h * 0.78)
        else:
            rx1, ry1 = int(img_w * 0.08), int(img_h * 0.08)
            rx2, ry2 = int(img_w * 0.92), int(img_h * 0.92)

        roi_bgr = frame[ry1:ry2, rx1:rx2]
        if roi_bgr.size == 0:
            roi_bgr = frame
            rx1, ry1 = 0, 0

        # Classify the dominant 3D material signature in the ROI
        resolved_cid: Optional[int] = None
        resolved_cname: str = ""
        resolved_conf: float = 0.86

        if material_hint:
            resolved_cid, resolved_cname = resolve_yolo_class_id_and_name(material_hint)
        else:
            classified = _classify_material_patch(roi_bgr, min_confidence=0.15)
            if classified is not None:
                resolved_cid, resolved_cname, resolved_conf = classified

        # If no verified material is found via hint or classification, do not generate fake boxes
        if resolved_cid is None:
            return []

        # Locate chromatic resource clusters inside `roi_bgr`
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        sat_val_mask = cv2.inRange(hsv, (0, 70, 75), (180, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(sat_val_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rh, rw = roi_bgr.shape[:2]
        min_area = max(36.0, float(rw * rh) * 0.004)
        max_area = float(rw * rh) * 0.55

        scored_boxes: List[Tuple[float, int, int, int, int]] = []
        center_x, center_y = rw * 0.5, rh * 0.54
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if not (min_area <= area <= max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.22 or aspect > 4.2:
                continue
            dist_center = math.hypot((bx + 0.5 * bw) - center_x, (by + 0.5 * bh) - center_y)
            centrality = 1.0 / (1.0 + dist_center / float(max(1, min(rw, rh))))
            score = area * centrality
            scored_boxes.append((score, bx, by, bw, bh))

        scored_boxes.sort(key=lambda item: item[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, (_, bx, by, bw, bh) in enumerate(scored_boxes[:2]):
            abs_x = rx1 + bx
            abs_y = ry1 + by
            sub_patch = frame[abs_y : abs_y + bh, abs_x : abs_x + bw]
            sub_cid, sub_cname = resolved_cid, resolved_cname
            if not material_hint and sub_patch.size > 0:
                classified = _classify_material_patch(sub_patch, min_confidence=0.15)
                if classified is not None:
                    sub_cid, sub_cname = classified[0], classified[1]
                else:
                    continue  # Skip unverified patch

            cx = float(np.clip((abs_x + 0.5 * bw) / float(img_w), 0.05, 0.95))
            cy = float(np.clip((abs_y + 0.5 * bh) / float(img_h), 0.05, 0.95))
            nw = float(np.clip(bw / float(img_w), 0.04, 0.45))
            nh = float(np.clip(bh / float(img_h), 0.04, 0.45))

            results.append(
                {
                    "class_id": int(sub_cid),
                    "class_name": self.class_names.get(int(sub_cid), sub_cname),
                    "cx": round(cx, 6),
                    "cy": round(cy, 6),
                    "w": round(nw, 6),
                    "h": round(nh, 6),
                    # D37/F-0838: HSV chromaticity is an unverified heuristic,
                    # so its boxes are capped at 0.68 (the same cap
                    # calibrate_box_trust applies) instead of being floored
                    # at 0.75-0.86 as if they were verified detections.
                    "confidence": round(min(0.68, resolved_conf - 0.04 * idx), 4),
                    "source": "hsv_3d_patch_fallback",
                }
            )

        if not results:
            return []

        return self.normalize_annotations(results, img_w=img_w, img_h=img_h)

    # -------------------------------------------------------------------------
    # Built-in Augmentation Fallback (Used only if `SyntheticYOLOAugmentor` unavailable)
    # -------------------------------------------------------------------------
    def _generate_variants_with_fallback(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
        augment_factor: int = 3,
    ) -> List[Tuple[str, np.ndarray, List[Dict[str, Any]]]]:
        """
        Generate `augment_factor` photometric & geometric variants using
        `SyntheticYOLOAugmentor` (or built-in deterministic OpenCV transforms).
        """
        if augment_factor <= 0:
            return []

        augmentor = self._get_augmentor()
        if augmentor is not None and hasattr(augmentor, "generate_augmented_variants"):
            try:
                return list(
                    augmentor.generate_augmented_variants(
                        frame=frame,
                        annotations=annotations,
                        count=int(augment_factor),
                    )
                )
            except Exception as exc:
                logger.warning("SyntheticYOLOAugmentor failed (%s); using curator fallback.", exc)

        # Built-in OpenCV fallback variants
        variants: List[Tuple[str, np.ndarray, List[Dict[str, Any]]]] = []
        img_h, img_w = frame.shape[:2]

        for idx in range(int(augment_factor)):
            mode = idx % 3
            if mode == 0:
                # Horizontal flip + warm dusk tint
                aug_img = cv2.flip(frame, 1)
                aug_img = np.clip(
                    aug_img.astype(np.float32) * np.array([0.92, 1.01, 1.10], dtype=np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                aug_anns = [
                    {
                        **a,
                        "cx": round(float(np.clip(1.0 - float(a["cx"]), 0.002, 0.998)), 6),
                    }
                    for a in annotations
                ]
                variants.append((f"aug{idx + 1:02d}_hflip", aug_img, aug_anns))
            elif mode == 1:
                # Cool night gamma & HSV shift
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
                hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255).astype(np.int32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255).astype(np.int32)
                aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                variants.append((f"aug{idx + 1:02d}_night_hsv", aug_img, list(annotations)))
            else:
                # Bright noon highlight + horizontal flip
                hsv = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2HSV).astype(np.int32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255).astype(np.int32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255).astype(np.int32)
                aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                aug_anns = self.normalize_annotations(
                    [
                        {
                            **a,
                            "cx": round(float(np.clip(1.0 - float(a["cx"]), 0.002, 0.998)), 6),
                        }
                        for a in annotations
                    ],
                    img_w=img_w,
                    img_h=img_h,
                )
                variants.append((f"aug{idx + 1:02d}_noon_hflip", aug_img, aug_anns))

        return variants

    # -------------------------------------------------------------------------
    # 3. Bootstrap Seeder (`bootstrap_from_existing_frames`)
    # -------------------------------------------------------------------------
    def bootstrap_from_existing_frames(
        self,
        augment_factor: int = 3,
        save_base_previews: bool = True,
        save_variant_previews: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Scan `reference_screenshots/*` and `video_frames/*` (jpg, jpeg, png, webp, bmp), auto-label every
        valid frame (using `YOLOAutoLabeler` or the built-in `MaterialScanner` + HSV 3D
        patch classifier fallback), generate `augment_factor` synthetic variants per frame,
        and populate `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`,
        `dataset/labels/val`, `dataset/previews`, `dataset/data.yaml`, and
        `dataset/manifest_stats.json`.

        P2-8 (DF.9): `should_stop` verilirse tarama kare/varyant
        sınırında durur; o ana kadar kaydedilen örnekler diskte kalır
        (kısmi sonuç geçerlidir), özet `cancelled: True` taşır.
        İptal yarım örnek yazmaz — kontrol yalnızca sınırda yapılır.

        Args:
            augment_factor: Number of augmented variants to generate per source image (default `3`).
            save_base_previews: Save bounding-box preview images for base screenshots.
            save_variant_previews: Save bounding-box preview images for augmented variants.
            should_stop: Optional zero-arg callable; True dönerse tarama
                bir sonraki kare/varyant sınırında durur.

        Returns:
            Summary dictionary with counts of scanned, base, augmented, train, and val samples,
            total bounding boxes, per-class distribution, and `manifest_stats`.
        """
        # Do not hold `_lock` across image decode. A multi-minute lock stalls BotWorker
        # heartbeats and the Qt refresh that rebuilds the manifest. Each save locks itself.
        self.ensure_dataset_structure()
        augment_factor = max(0, int(augment_factor))

        ref_dir = self.project_root / "reference_screenshots"
        vid_dir = self.project_root / "video_frames"

        # D37/F-0927: scan every supported image extension, not just *.jpg.
        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        source_files: List[Tuple[str, Path]] = []
        for src_tag, src_dir in (("ref", ref_dir), ("vid", vid_dir)):
            if not src_dir.is_dir():
                continue
            for pattern in image_exts:
                for p in sorted(src_dir.glob(pattern)):
                    if p.is_file():
                        source_files.append((src_tag, p))
        source_files.sort(key=lambda item: (item[0], item[1].name))

        base_saved = 0
        augmented_saved = 0
        # O8: eskiden bozuk görüntü ve etiketsiz kare tek sayaca
        # yazılıyordu; etiketleyicinin kör kaldığı kareler bozuk dosya
        # sanılıp gözden kaçıyordu. Artık ayrı sayılır.
        skipped_unreadable = 0
        skipped_unlabeled = 0
        sample_counter = 0
        # P2-8: iptal bayrağı — kare/varyant sınırında kontrol edilir.
        cancelled = False

        def _stop_requested() -> bool:
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception:
                return False

        for _src_idx, (src_tag, img_path) in enumerate(source_files):
            # P2-8: iptal yalnızca kare sınırında — yarım örnek yazılmaz.
            if _stop_requested():
                cancelled = True
                logger.info(
                    "Dataset bootstrap iptal edildi (%d/%d kaynak işlendi).",
                    _src_idx,
                    len(source_files),
                )
                break
            frame = imread_unicode(img_path, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                skipped_unreadable += 1
                continue

            img_h, img_w = frame.shape[:2]
            annotations = self.auto_label_frame(frame, source_path=img_path)
            if not annotations:
                skipped_unlabeled += 1
                continue

            # Deterministic train/val split (~85% train, ~15% val) from
            # the single factory schedule (D37/F-0924).
            base_split = self._choose_split(sample_counter, val_ratio=self.val_ratio)
            sample_counter += 1

            base_stem = f"{src_tag}_{img_path.stem}"
            self.save_labeled_sample(
                frame=frame,
                annotations=annotations,
                split=base_split,
                prefix=src_tag,
                save_preview=save_base_previews,
                sample_name=base_stem,
                update_manifest=False,
            )
            base_saved += 1

            # Generate `augment_factor` synthetic variants
            if augment_factor > 0:
                variants = self._generate_variants_with_fallback(
                    frame=frame,
                    annotations=annotations,
                    augment_factor=augment_factor,
                )
                for suffix, aug_frame, aug_anns in variants:
                    # P2-8: iptal varyant sınırında da kontrol edilir.
                    if _stop_requested():
                        cancelled = True
                        break
                    valid_aug_anns = self.normalize_annotations(
                        aug_anns, img_w=img_w, img_h=img_h
                    )
                    if not valid_aug_anns:
                        continue

                    # K2 (dilim 03): varyant ana kareyle AYNI split'e
                    # gider — eskiden sayaç artırılıp ayrı `_choose_split`
                    # çağrılıyordu; ana kare train'deyken varyantı val'e
                    # düşebiliyor, val train'in kopyasını ölçüyordu.
                    var_split = base_split

                    var_stem = f"{base_stem}_{suffix}"
                    self.save_labeled_sample(
                        frame=aug_frame,
                        annotations=valid_aug_anns,
                        split=var_split,
                        prefix=src_tag,
                        save_preview=save_variant_previews,
                        sample_name=var_stem,
                        update_manifest=False,
                    )
                    augmented_saved += 1

        # Ensure all 27 canonical classes have representative exemplars in both
        # train and val splits so downstream YOLO training has full coverage
        # (seeding itself is disabled -- see _seed_missing_class_exemplars).
        # P2-8: iptalde tohumlama atlanır — kısmi manifest olduğu gibi kalır.
        seeded_exemplars = 0
        if not cancelled:
            seeded_exemplars = self._seed_missing_class_exemplars(
                source_files=source_files,
                sample_counter_start=sample_counter,
            )

        manifest = self.rebuild_manifest_stats()

        total_saved = base_saved + augmented_saved + seeded_exemplars
        classes_covered = sum(
            1 for count in manifest["class_counts"].values() if count > 0
        )
        summary: Dict[str, Any] = {
            # D37/F-0839: "bootstrapped" is only True when at least one sample
            # was actually saved; with zero sources/samples the old code
            # still reported True. Coverage is measured from the manifest
            # (the exemplar seeder is a deliberate no-op stub -- see
            # _seed_missing_class_exemplars -- so it must not be reported
            # as coverage).
            "bootstrapped": bool(total_saved > 0),
            # P2-8: kısmi (iptal edilmiş) tarama açıkça işaretlenir.
            "cancelled": bool(cancelled),
            "interrupted": bool(cancelled),
            "source_images_found": len(source_files),
            "source_images_skipped": skipped_unreadable + skipped_unlabeled,
            "source_images_skipped_unreadable": skipped_unreadable,
            "source_images_skipped_unlabeled": skipped_unlabeled,
            "base_samples_saved": base_saved,
            "augmented_samples_saved": augmented_saved,
            "class_coverage_exemplars_saved": seeded_exemplars,
            "total_samples_saved": total_saved,
            "classes_covered": classes_covered,
            "class_coverage_complete": classes_covered >= len(self.class_names),
            "train_count": manifest["train_count"],
            "val_count": manifest["val_count"],
            "total_images": manifest["total_images"],
            "target_goal": manifest["target_goal"],
            "total_boxes": manifest["total_boxes"],
            "class_counts": manifest["class_counts"],
            "data_yaml_path": str(self.data_yaml_path),
            "manifest_stats_path": str(self.manifest_stats_path),
            "manifest_stats": manifest,
        }
        logger.info(
            "Dataset bootstrap complete: %d total images (train=%d, val=%d), %d bounding boxes across %d classes.",
            manifest["total_images"],
            manifest["train_count"],
            manifest["val_count"],
            manifest["total_boxes"],
            len(self.class_names),
        )
        return summary

    def _seed_missing_class_exemplars(
        self,
        source_files: Sequence[Tuple[str, Path]],
        sample_counter_start: int = 0,
    ) -> int:
        """
        Synthetic center rectangle generation across missing classes is disabled.
        Active learning uses genuine visual detections and human-confirmed ExemplarMemory
        rather than stamped/painted rectangular patches.
        """
        return 0

    def get_curator_health_summary(self) -> dict:
        """
        Return DatasetCurator health and Unicode I/O readiness summary.

        D37/F-0928: this used to return `True` unconditionally. It now
        actually checks: the dataset directories exist, `data.yaml` lists
        the full canonical class table, and a Unicode-named image survives
        an imwrite/imread round-trip.
        """
        checks: Dict[str, bool] = {
            "dataset_dirs": False,
            "data_yaml_classes": False,
            "unicode_io": False,
        }
        try:
            checks["dataset_dirs"] = bool(
                self.images_train_dir.is_dir()
                and self.images_val_dir.is_dir()
                and self.labels_train_dir.is_dir()
                and self.labels_val_dir.is_dir()
            )
            if self.data_yaml_path.is_file():
                yaml_text = self.data_yaml_path.read_text(encoding="utf-8")
                checks["data_yaml_classes"] = all(
                    f"  {cid}: {name}" in yaml_text
                    for cid, name in sorted(self.class_names.items())
                )
            import tempfile

            probe = np.zeros((8, 8, 3), dtype=np.uint8)
            probe[:, :] = (10, 120, 200)
            with tempfile.TemporaryDirectory() as tmp:
                probe_path = Path(tmp) / "sağlık_probesi_çerçeve.jpg"
                if imwrite_unicode(probe_path, probe):
                    back = imread_unicode(probe_path)
                    checks["unicode_io"] = back is not None and back.shape == probe.shape
        except Exception as exc:
            logger.debug("Curator health check failed: %s", exc)
        ready = all(checks.values())
        return {"curator_ready": ready, **checks}


    @property
    def label_cache_path(self) -> Path:
        return self.dataset_dir / ".label_cache.json"

    def load_label_cache(self) -> Dict[str, Any]:
        if self.label_cache_path.is_file():
            try:
                return json.loads(self.label_cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                # D4: bozuk cache eskiden sessizce boş sayılıyordu;
                # tekrarlayan bozulma fark edilmiyordu.
                logger.warning("Bozuk etiket cache'i yoksayıldı (%s): %s", self.label_cache_path, exc)
                return {}
        return {}

    def save_label_cache(self, cache: Dict[str, Any]) -> None:
        try:
            tmp = self.label_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            if tmp.exists():
                tmp.replace(self.label_cache_path)
        except Exception as exc:
            logger.debug("Failed to save label cache: %s", exc)

    def relabel_dataset_batch(
        self,
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        *,
        max_workers: Optional[int] = None,
        incremental: bool = True,
        force_all: bool = False,
        conf_floor: float = 0.45,
        emin_threshold: float = 0.85,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        High-throughput multi-core batch relabeler with incremental caching (.label_cache.json).
        - Skips unchanged images in milliseconds if incremental=True and mtime/size/exemplar match.
        - NEVER skips 0-box images thinking they are done.
        - Skips writing disk previews during bulk batch runs for maximum throughput.
        - Emits live throughput metrics (fps, ETA, processed/total).
        """
        import concurrent.futures
        import os

        if max_workers is None:
            max_workers = min(16, (os.cpu_count() or 4) * 2)

        # 1. Discover target images
        target_paths: List[Path] = []
        if image_paths is not None and len(image_paths) > 0:
            for p in image_paths:
                path_obj = Path(p)
                if path_obj.is_file():
                    target_paths.append(path_obj)
        else:
            for split_dir in (self.images_train_dir, self.images_val_dir):
                if split_dir.is_dir():
                    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                        target_paths.extend(sorted(split_dir.glob(ext)))

        total_images = len(target_paths)
        if total_images == 0:
            return {
                "total": 0,
                "processed": 0,
                "labeled": 0,
                "skipped": 0,
                "errors": 0,
                "error_stems": [],
                "total_boxes": 0,
                "fps": 0.0,
                "elapsed_seconds": 0.0,
            }

        # Pre-create review and label directories once to avoid per-thread mkdir lock contention
        for split_name in ("train", "val"):
            (self.dataset_dir / "review" / split_name).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        cache = self.load_label_cache()
        ex_rev = 0
        ex_file = self.dataset_dir / "learned_exemplars.json"
        if ex_file.is_file():
            try:
                ex_rev = int(ex_file.stat().st_mtime_ns)
            except Exception:
                ex_rev = 0

        # Thread local labeler instance
        _tls = threading.local()

        def _get_worker_labeler():
            if not hasattr(_tls, "labeler") or _tls.labeler is None:
                from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler
                _tls.labeler = YOLOAutoLabeler()
            return _tls.labeler

        from dataset_factory.label_trust import (
            _load_state,
            train_txt_boxes,
            write_review_pair,
        )

        def _process_one(img_p: Path) -> Dict[str, Any]:
            frame = None
            raw_anns = None
            norm_anns = None
            try:
                st = img_p.stat()
                stem = img_p.stem
                # D37/F-1034: the split is decided by the exact split
                # directory the image lives in, not by a "val" membership
                # test over path parts.
                try:
                    is_val = img_p.resolve().parent == self.images_val_dir.resolve()
                except OSError:
                    is_val = img_p.parent.name == "val"
                split = "val" if is_val else "train"

                review_file = self.dataset_dir / "review" / split / f"{stem}.json"
                label_file = self.dataset_dir / "labels" / split / f"{stem}.txt"

                # D37/F-0840: the cache key is the full resolved path, not
                # just the filename -- train/0001.jpg and val/0001.jpg used
                # to collide, and a weak-boxed file was skipped forever.
                cache_key = str(img_p.resolve())
                c_entry = cache.get(cache_key)

                # Incremental skip check: only skip if genuinely has boxes (>0) and matching stat
                if (
                    not force_all
                    and incremental
                    and c_entry is not None
                    and review_file.is_file()
                    and label_file.is_file()
                ):
                    mtime_match = (c_entry.get("mtime_ns") == st.st_mtime_ns)
                    size_match = (c_entry.get("size") == st.st_size)
                    rev_match = (c_entry.get("exemplar_rev") == ex_rev)
                    box_cnt = c_entry.get("box_count", 0)
                    if mtime_match and size_match and rev_match and box_cnt > 0:
                        return {
                            "stem": stem,
                            "status": "skipped",
                            "boxes": box_cnt,
                            "cached": True,
                            "key": cache_key,
                        }

                # P1-17 / K-06: turbo eski review/TXT'yi hiç okumadan
                # eziyordu — el etiketleri sessizce kayboluyordu (iki
                # aşamalı kayıp: review ezilir, sonraki commit TXT'yi
                # de götürür). Önce mevcut durumu oku: _load_state
                # review JSON + TXT'yi uzlaştırır (TXT-only el etiketi
                # trusted'a taşınır, K-01/K-03) ve bozuk review'u
                # meta'da bildirir. Bozuk review fail-closed: içeriği
                # bilinmeyen kare üzerine yazılmaz (P1-16/K-05).
                existing_hand_trusted: List[Dict[str, Any]] = []
                existing_hand_weak: List[Dict[str, Any]] = []
                try:
                    prev_trusted, prev_weak, prev_meta = _load_state(
                        self.dataset_dir, split, stem, label_file,
                        return_meta=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "Turbo: mevcut durum okunamadı, %s atlanıyor: %s",
                        stem, exc,
                    )
                    return {
                        "stem": stem,
                        "status": "error",
                        "boxes": 0,
                        "error": "existing_state_unreadable",
                        "key": cache_key,
                    }
                if prev_meta.get("review_corrupt"):
                    logger.warning(
                        "Turbo: bozuk review JSON, %s üzerine yazılmıyor "
                        "(el etiketi kaybı engellendi).",
                        stem,
                    )
                    return {
                        "stem": stem,
                        "status": "error",
                        "boxes": 0,
                        "error": "review_corrupt",
                        "key": cache_key,
                    }
                for b in prev_trusted:
                    if _is_hand_box(b):
                        existing_hand_trusted.append(dict(b))
                for b in prev_weak:
                    # Weak'teki el kutusu şüpheliye ayrılmış el
                    # etiketidir (demote): trusted'a terfi ettirilmez,
                    # weak'te korunur — kullanıcı kararı korunur.
                    if _is_hand_box(b) or b.get("demoted_by_user") is True:
                        existing_hand_weak.append(dict(b))

                frame = imread_unicode(img_p, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    return {
                        "stem": stem,
                        "status": "error",
                        "boxes": 0,
                        "error": "decode_fail",
                        "key": cache_key,
                    }

                labeler = _get_worker_labeler()
                # V20-YZ-ETİKET: yeniden etiketleme karedeki HER ŞEYİ etiketler
                # (full_scan=True). Stem learn_monsters_* olsa bile HUD (16-24)
                # ve yol/engel (14/15) kutuları üretilir; yoksa relabel yalnızca
                # canavar + materyal + minimap kutusu üretebilirdi. Kutular
                # zayıf-cap'lidir, incelemeye düşer, onaysız TXT'ye girmez.
                raw_anns = labeler.auto_label_frame(
                    frame, fast_mode=True, source_path=img_p, full_scan=True
                )
                norm_anns = self.normalize_annotations(
                    raw_anns, img_w=frame.shape[1], img_h=frame.shape[0]
                )

                trusted_list: List[Dict[str, Any]] = []
                weak_list: List[Dict[str, Any]] = []
                for b in norm_anns:
                    b_cal = calibrate_box_trust(b, stem=stem, emin_threshold=emin_threshold)
                    if b_cal.get("trusted") is True:
                        trusted_list.append(b_cal)
                    else:
                        weak_list.append(b_cal)

                # P1-17 / K-06: canlı yol deseni
                # (ui/yolo_factory_dialog.py::_relabel_single_item):
                # el etiketleri model çıktısıyla değiştirilmez — önce
                # korunur, model kutuları yalnızca çakışmayanlar
                # eklenir. Trusted kökenli el kutusu trusted'ta,
                # weak'teki el kutusu weak'te kalır (terfi yok).
                # Bu birleştirme olmadan 0-kutu bulan model bile
                # review'u boşaltıp el etiketini yok ediyordu
                # (F-0789 yalnızca TXT'yi koruyordu, review'u değil).
                existing_hand_all = existing_hand_trusted + existing_hand_weak
                merged_trusted: List[Dict[str, Any]] = list(existing_hand_trusted)
                for b in trusted_list:
                    if not any(
                        _turbo_box_identity_match(b, h) for h in existing_hand_all
                    ):
                        merged_trusted.append(b)
                merged_weak: List[Dict[str, Any]] = list(existing_hand_weak)
                for b in weak_list:
                    if not any(
                        _turbo_box_identity_match(b, h) for h in existing_hand_all
                    ):
                        merged_weak.append(b)

                # Persist review json + label txt through the single
                # writer (P3-26): the old hand-rolled `write_text` for
                # the JSON was non-atomic, and a JSON-empty/TXT-old
                # divergence was possible (F026-03). The pair writer
                # computes the TXT emit with the canonical rule and
                # writes both files atomically (.tmp + os.replace,
                # .bak before overwrite).
                rev_payload = {
                    "stem": stem,
                    "image_path": str(img_p),
                    "trusted": merged_trusted,
                    "weak": merged_weak,
                    "requeued_at": time.time(),
                }
                emit_boxes = train_txt_boxes(
                    merged_trusted, merged_weak, stem=stem, floor=conf_floor
                )
                # D37/F-0789: when the auto-labeler finds nothing, do NOT
                # blank an existing non-empty label file -- the pair
                # writer refuses an empty emit over a non-empty TXT (P0-1)
                # and reports it via txt_written=False; the previous
                # labels are kept and recorded below.
                pair = write_review_pair(
                    review_path=review_file,
                    label_path=label_file,
                    review_payload=rev_payload,
                    stem=stem,
                    floor=conf_floor,
                    emit_boxes=emit_boxes,
                )
                kept_existing_labels = not pair["txt_written"]

                # O9: özet/cache eskiden normalize edilmiş ham kutu sayısını
                # (`len(norm_anns)`) yazıyordu — eşik-altı/zayıf kutular
                # txt'ye girmese bile "kutu var" sayılıyordu; txt'si boş ama
                # box_count>0 olan dosya sonsuza dek atlanıyordu. Artık
                # txt'ye GERÇEKTEN yazılan kutu sayılır.
                if kept_existing_labels:
                    try:
                        written_boxes = sum(
                            1
                            for _ln in label_file.read_text(
                                encoding="utf-8", errors="ignore"
                            ).splitlines()
                            if _ln.strip()
                        )
                    except OSError:
                        written_boxes = 0
                else:
                    written_boxes = len(emit_boxes)

                return {
                    "stem": stem,
                    "status": "labeled",
                    "boxes": written_boxes,
                    # O9/E-07 ayrımı: "boxes" txt'ye GERÇEKTEN yazılanı
                    # sayar (total_boxes özeti için — W44 pin'i). Cache'in
                    # "box_count"u ise ayrı bir soruyu yanıtlar: "bu kare
                    # etiketlendi mi?" — review JSON'a kalıcı yazılan
                    # (trusted+weak) kutu sayısı. E-07 sonrası relabel
                    # weak-only üretebilir (txt boş kalır) ama kare
                    # etiketlenmiştir: atlama kararı onu saymazsa Work
                    # Package F'in artımlı önbelleği hiç atlamaz. 0-box
                    # koruması aynen durur: labeler hiç kutu bulamadıysa
                    # labeled_boxes=0 olur ve kare asla atlanmaz.
                    # P1-17: merged listeler sayılır — el etiketleri de
                    # review'a kalıcı yazılan kutudur.
                    "labeled_boxes": len(merged_trusted) + len(merged_weak),
                    "trusted": len(merged_trusted),
                    "weak": len(merged_weak),
                    "hand_preserved": len(existing_hand_all),
                    "kept_existing_labels": kept_existing_labels,
                    "mtime_ns": st.st_mtime_ns,
                    "size": st.st_size,
                    "exemplar_rev": ex_rev,
                    "key": cache_key,
                }
            except Exception as exc:
                return {
                    "stem": img_p.stem,
                    "status": "error",
                    "boxes": 0,
                    "error": str(exc),
                    "key": str(img_p.resolve()),
                }
            finally:
                # Explicit cleanup to ensure immediate memory freeing without leaks
                del frame
                del raw_anns
                del norm_anns

        start_time = time.time()
        processed = 0
        labeled_count = 0
        skipped_count = 0
        total_boxes = 0
        # D37/F-0841: per-image errors used to be swallowed -- the summary
        # reported no error count at all. Count them and keep the stems.
        error_count = 0
        error_stems: List[str] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_one, p) for p in target_paths]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                processed += 1
                status = res.get("status")
                boxes = res.get("boxes", 0)
                total_boxes += boxes

                if status == "skipped":
                    skipped_count += 1
                elif status == "error":
                    error_count += 1
                    if len(error_stems) < 20:
                        error_stems.append(str(res.get("stem", "?")))
                elif status == "labeled":
                    labeled_count += 1
                    cache_key = res.get("key")
                    if cache_key:
                        # Cache box_count = review'a yazılan kutu (bkz.
                        # yukarıdaki O9/E-07 ayrımı); "boxes" (txt'ye
                        # yazılan) yalnızca total_boxes özetine girer.
                        cache[cache_key] = {
                            "mtime_ns": res.get("mtime_ns"),
                            "size": res.get("size"),
                            "box_count": res.get("labeled_boxes", boxes),
                            "exemplar_rev": res.get("exemplar_rev"),
                            "updated_at": time.time(),
                        }

                now = time.time()
                elapsed = max(0.001, now - start_time)
                fps = processed / elapsed
                eta_s = (total_images - processed) / max(0.1, fps)

                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "processed": processed,
                                "total": total_images,
                                "fps": round(fps, 1),
                                "eta_s": round(eta_s, 1),
                                "stem": res.get("stem"),
                                "boxes": boxes,
                                "status": status,
                            }
                        )
                    except Exception:
                        pass

        # Calculate pure batch processing duration and fps before disk metadata sync
        batch_duration = max(0.001, time.time() - start_time)
        batch_fps = round(processed / batch_duration, 1)

        # Save cache and manifest
        self.save_label_cache(cache)
        manifest = self.rebuild_manifest_stats()
        elapsed_total = max(0.001, time.time() - start_time)

        return {
            "total": total_images,
            "processed": processed,
            "labeled": labeled_count,
            "skipped": skipped_count,
            "errors": error_count,
            "error_stems": error_stems,
            "total_boxes": total_boxes,
            "fps": batch_fps,
            "elapsed_seconds": round(batch_duration, 2),
            "manifest_stats": manifest,
        }


def relabel_dataset_batch(
    image_paths: Optional[Sequence[Union[str, Path]]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function running batch relabeling via DatasetCurator."""
    curator = DatasetCurator(dataset_dir=dataset_dir) if dataset_dir else DatasetCurator()
    return curator.relabel_dataset_batch(image_paths=image_paths, **kwargs)


__all__ = [
    "TARGET_DATASET_GOAL",
    "YOLO_12_CLASS_NAMES",
    "YOLO_CLASS_NAMES",
    "CLASS_NAME_TO_ID",
    "DatasetCurator",
    "relabel_dataset_batch",
    "resolve_yolo_class_id_and_name",
    "imread_unicode",
    "imwrite_unicode",
    "UNVERIFIED_HEURISTIC_SOURCES",
    "calibrate_box_trust",
]


if __name__ == "__main__":
    curator = DatasetCurator()
    result = curator.bootstrap_from_existing_frames(augment_factor=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))


