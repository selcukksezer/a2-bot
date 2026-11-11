# -*- coding: utf-8 -*-
"""Split trusted gather labels from weak auto boxes.

YOLO train txt files have no confidence. Writing every auto box there teaches
the wrong thing (last run mAP50 ~0.04). Low-confidence and untrusted boxes go
to dataset/review/*.json. A later, more confident observation may replace a
weak box. A box marked trusted is never overwritten.

Trusted seed: a hand label (trusted=True) or an explicit per-box human
approval. An F/E gather template hit is NOT a trusted seed on its own
anymore (P1-12/K-34): template boxes need an explicit human approval to
enter the train txt (K1 gate in train_txt_boxes). Without that seed, a
higher auto score still stays in the review sidecar.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

from dataset_factory.approval_state import (
    box_has_human_provenance,
    box_is_never_requeue,
    is_frame_closed,
)
from dataset_factory.dataset_validator import (
    # is_dataset_ready_for_training is re-exported: ui/yolo_factory_dialog.py
    # imports it from this module, so it must stay importable here.
    is_dataset_ready_for_training,
    mark_model_stale,
    sanitize_boxes,
)
from dataset_factory.box_match import (
    SAME_BOX_IOU,
    box_iou,
    iou_xyxy,
    parse_box,
    same_box_parsed,
)

# TXT-only kare = düzeltilmiş el etiketi (ui/yolo_factory_dialog.py'deki
# HAND_LABEL_TXT_SOURCE ile aynı değer; kuyruk yükleyici TXT kökenli
# kutuyu bu kaynakla damgalar). Requeue damgayı buradan atar (P0-8/K-14).
HAND_LABEL_TXT_SOURCE: str = "hand_label_txt"

# No settings key for a train-label floor. The 0.42 band is uncertainty mining,
# not permission to enter the train txt. Boxes under this floor stay weak.
TRAIN_CONF_FLOOR: float = 0.45
# Omni learn_* layout boxes carry this placeholder confidence in review.
# It is NOT permission to enter the train txt: since sweep E-06 a weak
# learn box enters the txt only with an explicit per-box human approval.
# Keep conf ≥ floor so purge does not wipe them before review.
LEARN_AUTO_CONF: float = 0.50
# P1-13 (K-33): SLOT_IOU 0.30 KALDIRILDI — ayni slot sayilmak icin %30
# ortusme yetiyordu; yan yana iki nesne tek slot sayilip yeni gozlem
# dusebiliyordu (F070). Slot eslesmesi artik tek tanim ``same_box()``
# (IoU >= 0.45 + merkez <= 0.05). Ad yalnizca geriye donuk uyumluluk
# icin kalir (tests/test_dev_d54_label_trust.py ``lt.SLOT_IOU`` okur)
# ve her zaman SAME_BOX_IOU'ya esittir.
SLOT_IOU: float = SAME_BOX_IOU

# ----------------------------------------------------------------------
# P1-12 (K-31/K-32/K-34/K-35): TEK güven sabitleri. Eskiden 0.90
# literali 6 ayrı yerde, 0.85 üç ayrı değerde tekrarlanıyordu; aynı kutu
# yola göre trusted veya weak oluyordu. Artık her eşik/floor/varsayılan
# buradan okunur — yeni bir güven sayısı gerekiyorsa BURAYA eklenir,
# hiçbir dosyada çıplak literal tutulmaz.
# ----------------------------------------------------------------------
# "Emin" (trusted) eşiği: motor tarafı (calibrate / requeue / temizlik
# katmanları / budama) bu değeri okur. UI rozeti ise bulk_range
# kaydırıcısını okur (bkz. ui/yolo_factory_dialog.py::_emin_badge_threshold):
# kaydırıcı varsayılanı CONF_BULK_DEFAULT (0.90) olduğu için rozet "Emin"
# ile toplu onay aynı eşikte buluşur; kaydırıcı oynarsa ikisi birlikte oynar.
CONF_EMIN: float = 0.85
# Toplu onay kaydırıcısı (bulk_range) varsayılanı / kaydırıcı yokken
# kullanılan yedek eşik. Kaydırıcı 0.90'dan aşağı indirilirse filtreler,
# sayaçlar ve toplu onay aynı değeri okur.
CONF_BULK_DEFAULT: float = 0.90
# Ölçülmemiş güvenin requeue'daki yer tutucusu. Gerçek ölçüm DEĞİLDİR:
# her yazılışında ``confidence_fabricated=True`` işaretlenir ve uydurma
# güven tek başına hiçbir emit/onay yolunu geçemez (E-03).
CONF_REQUEUE_DEFAULT: float = 0.58
# Doğrulanmamış heuristik kaynağın calibrate kapağı.
CONF_HEURISTIC_CAP: float = 0.68
# Materyal-çakışmalı kutunun calibrate kapağı (TRAIN_CONF_FLOOR ile
# aynı değer, ayrı anlam: çakışmalı kutu tabanda bile güvenilmez).
CONF_MATERIAL_CAP: float = 0.45

# Full-frame minimap HUD (sağ-üst). Padded vs mobile_server.MINIMAP_NORM_BOUNDS.
MINIMAP_HUD_XYXY: Tuple[float, float, float, float] = (0.82, 0.02, 1.0, 0.32)
# 0..5 world resources + world_gather_crest — minimap içinde herb_aria sahte pozitif
WORLD_RESOURCE_CLASS_IDS: frozenset = frozenset({0, 1, 2, 3, 4, 5, 6})
# Minimap ROI: filiz/canavar noktası + F/E UI (hflip sonrası sağ üstte kalabilir)
# Kanonik taksonomi (dataset_factory/class_taxonomy.py CANONICAL_NAMES):
# 13 = skill_ui_prompt (gerçek F/E gather prompt), 9 = minimap_sprout,
# 11 = minimap_monster_dot. 7 = herb_lumina / 8 = ore_obsidian dünya
# kaynağıdır — minimap'te herb = poison, bu yüzden sette yok.
MINIMAP_ALLOWED_CLASS_IDS: frozenset = frozenset({9, 11, 13})
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
# Pre-built set: is_material_conflict runs per box in the calibrate/clean
# hot paths and used to rebuild this set from the tuple on every call.
KNOWN_MATERIAL_TOKENS_SET: frozenset = frozenset(KNOWN_MATERIAL_TOKENS)

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
    """True for synthetic (invented/derived, not directly observed) sources.

    Covers the omni_* layout/fallback family AND the synthetic augmentor
    (sweep F7): an augmented variant is derived evidence — it carries
    ``synthetic: True`` with its original source under ``original_source``
    (D37/F-0899) and its confidence is capped below the trust boundary
    (F7, ``SYNTHETIC_CONFIDENCE_CAP``). A synthetic source never trusts a
    box on its own (see ``is_trusted_annotation`` / ``merge_label_observation``):
    a variant of a hand-labelled box still needs its own human approval.
    """
    s = str(source or "").strip().lower()
    return s.startswith("omni_") or s.startswith("synthetic_augmentor")


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


def _is_explicitly_human_approved(ann: Dict[str, Any]) -> bool:
    """Per-box explicit human decision: hand-drawn or individually approved.

    Frame/doc-level flags (`done == "ok"`, bare `approved`/`confirmed`)
    do NOT count: bulk approval must never turn an uninspected box into a
    hand-verified one (sweep E-04).

    P3-27: kanonik tanım ``dataset_factory.approval_state.
    box_has_human_provenance``'tadır — bu fonksiyon yalnızca uyumluluk
    için korunan ince bir takma addır (çağrı yerleri değişmez). Kural
    değişirse SADECE kanonik fonksiyon değişir; P1-10'un
    ``approved_pool``'a bıraktığı aynı adlı kopya da birleşmede silinip
    kanonik tanıma bağlanacak.
    """
    return box_has_human_provenance(ann)


def _has_measured_confidence(ann: Dict[str, Any]) -> bool:
    """True when a parseable confidence/conf/score value exists.

    A trusted box with NO measured confidence (key absent, None, or
    unparseable) is not a below-floor box: the synchronizer's S-6103 rule
    ("hand labels carry no score") lets it into the TXT, and P3-26 adopts
    that rule here so commit and sync emit the same boxes. A MEASURED
    confidence — including an explicit 0.0 — still has to clear the floor.
    Fabricated-confidence boxes are filtered before this check runs, so
    this never re-admits an E-03 placeholder.
    """
    if not isinstance(ann, dict):
        return False
    for key in ("confidence", "conf", "score"):
        if key not in ann:
            continue
        try:
            value = float(ann[key])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN is not a measurement
            continue
        return True
    return False


def train_txt_boxes(
    trusted: Sequence[Dict[str, Any]],
    weak: Sequence[Dict[str, Any]],
    *,
    stem: str = "",
    floor: float = TRAIN_CONF_FLOOR,
) -> List[Dict[str, Any]]:
    """Boxes that should appear in labels/{split}/*.txt.

    P3-26: this is the SINGLE canonical emit rule — every JSON+TXT write
    path (commit, purge, clean, prune, demote, persist/sync, relabel)
    computes its TXT emit through this function, so the synchronizer
    (`atomic_review_sync._extract_emitted_boxes_from_json_data`) and
    `commit_split_labels` can no longer disagree (F026-02).

    Trusted (floor/hand) always. For learn_* stems, weak learn/omni auto
    boxes enter the txt ONLY with an explicit per-box human approval:
    "Adım 2 culls mistakes" was never enforced by code, so unapproved
    weak boxes trained the model (sweep E-06). Below-floor weak boxes
    never enter the txt. A box whose confidence was fabricated (no
    measurement) never enters on the number alone, either (sweep E-03).
    A box with NO measured confidence at all (hand labels carry no score,
    S-6103) passes the floor — unknown is not below-floor.
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
        # Fabricated confidence is not a measurement: without an explicit
        # per-box human decision the box stays out of training (E-03).
        if b.get("confidence_fabricated") is True and not _is_explicitly_human_approved(b):
            continue
        # K1: a layout/template box is not evidence even when it sits in
        # the trusted list (stale review file, old stamp). Its confidence
        # (LEARN_AUTO_CONF 0.50) clears the train floor (0.45), so the
        # floor alone cannot keep it out — only an explicit per-box human
        # approval lets a template box into the txt. P1-12 (K-34/L008):
        # F/E şablon (seed) kutular da K1 kapsamındadır — şablon izi
        # (f_template/e_template, gather_prompt) tek başına kanıt
        # değildir; el etiketi zaten açık insan onayı sayıldığı için
        # bu kapıdan geçer.
        if (is_learn_auto_annotation(b) or _is_hand_or_gather_seed(b)) and not _is_explicitly_human_approved(b):
            continue
        if (
            bool(b.get("hand_label") is True)
            or not _has_measured_confidence(b)
            or annotation_confidence(b) >= floor_f
        ):
            _add(b)

    if is_learn_stem(stem):
        for b in weak or []:
            if not isinstance(b, dict):
                continue
            # Learn/omni auto boxes are layout placeholders, not evidence:
            # they enter the txt only with an explicit per-box human
            # approval at/above the floor. Below-floor weak boxes stay in
            # the review sidecar.
            if not _is_explicitly_human_approved(b):
                continue
            if not _has_measured_confidence(b) or annotation_confidence(b) >= floor_f:
                _add(b)

    return out


def filter_hud_false_positives(
    annotations: Sequence[Dict[str, Any]],
    *,
    stem: str = "",
    remap_sprout: bool = False,
) -> List[Dict[str, Any]]:
    """Drop boxes that cannot be true positives on HUD / wrong category stems.

    - Minimap ROI: only class 13 (F/E gather prompt / skill_ui_prompt),
      9 (sprout) and 11 (minimap monster) allowed
      (herb/path/obstacle/skill-bar there = poison)
    - Boxes whose coordinates cannot be parsed are dropped, not passed
      through unfiltered (their position cannot be verified).
    - learn_paths_*: drop world resources; drop any box centered on minimap
    - Optional remap_sprout: world→class 9 instead of drop (rare)

    Drop kararı _hud_drop_reason() verir (K-19: neden kodu ayrı taşınır,
    temizlik öğretimi nedene göre ayırır); bu fonksiyon yalnızca kararı
    uygular + remap yapar. Davranış eski gövdenin aynısıdır.
    """
    out: List[Dict[str, Any]] = []
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
            # Unparseable coordinates: position cannot be verified, drop.
            continue

        in_hud = box_center_in_minimap_hud(cx, cy)

        if _hud_drop_reason(ann, stem=stem, remap_sprout=remap_sprout) is not None:
            continue

        if (
            remap_sprout
            and in_hud
            and cid not in MINIMAP_ALLOWED_CLASS_IDS
            and cid in WORLD_RESOURCE_CLASS_IDS
        ):
            ann["class_id"] = MINIMAP_SPROUT_CLASS_ID
            ann["class_name"] = "minimap_sprout"
            ann["remapped_from"] = cid
            ann["source"] = str(ann.get("source", "")) + "|hud_remap_sprout"

        out.append(ann)
    return out


def _is_hand_or_gather_seed(ann: Dict[str, Any]) -> bool:
    """True for hand labels / F/E gather templates — never bulk-demote these.

    The F/E gather prompt is canonical class 13 (``skill_ui_prompt``).
    The old 7/8 mapping was wrong: 7 is ``herb_lumina`` and 8 is
    ``ore_obsidian`` (materials, not prompts) — a material box carrying
    an F/E trace must NOT seed, and a real class-13 prompt must (sweep
    Y1). No taxonomy change: only the seed predicate is corrected to
    the canonical id.
    """
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
    if cid == 13 and (
        key in ("f", "e")
        or "f_gather" in source
        or "e_gather" in source
        or "gather_prompt" in source
        or "f_template" in source
        or "e_template" in source
        or "e_floating" in source
        or "floating" in source
    ):
        return True
    return False


# ----------------------------------------------------------------------
# K-19: HUD düşürme nedenleri. filter_hud_false_positives eskiden 7
# farklı nedeni tek "poison" hükmüyle düşürüyor, temizlik de hepsini tek
# "poison_hud_illegal" kaynağıyla negatif öğretiyordu — path-stem'deki
# gerçek herb/ore crop'u bile negatif diye yazılıyordu. Neden artık ayrı
# kodla taşınır; öğretim nedene göre ayrılır (bkz.
# HUD_NO_NEGATIVE_TEACH_REASONS).
# ----------------------------------------------------------------------
HUD_DROP_PATH_WORLD_RESOURCE = "path_world_resource"
HUD_DROP_PATH_IN_HUD = "path_in_hud"
HUD_DROP_HP_MP_OUT_OF_BAND = "hp_mp_out_of_band"
HUD_DROP_HUD_DISALLOWED_CLASS = "hud_disallowed_class"
HUD_DROP_HUD_UNVERIFIED_PROMPT = "hud_unverified_prompt_13"
HUD_DROP_UNPARSEABLE_COORDS = "unparseable_coords"
HUD_DROP_INVALID_BOX = "invalid_box"

# Bu nedenlerle düşen kutunun crop'u negatif ÖĞRETİLMEZ: path-stem'deki
# dünya kaynağı gerçek bir herb/ore crop'udur (negatif öğretmek 3D remap
# ile path_ground'a çevirip 0.90 çapraz-sınıf cezasına uğratır);
# çözülemeyen/geçersiz kutunun crop'u ise doğrulanamaz.
HUD_NO_NEGATIVE_TEACH_REASONS: frozenset = frozenset(
    {
        HUD_DROP_PATH_WORLD_RESOURCE,
        HUD_DROP_UNPARSEABLE_COORDS,
        HUD_DROP_INVALID_BOX,
    }
)


def _hud_drop_reason(
    ann: Any,
    *,
    stem: str = "",
    remap_sprout: bool = False,
) -> Optional[str]:
    """filter_hud_false_positives'ın düşürme kararı + nedeni (K-19).

    ``None`` = kutu korunur (ya da ``remap_sprout`` ile yeniden
    sınıflanır); aksi halde düşürme neden kodu döner. Filtre fonksiyonu
    bu yardımcı üzerinden çalışır — karar tek yerde tutulur, neden ile
    hüküm birbirinden ayrılmaz. Karar sırası eski filtrenin aynısıdır.
    """
    if not isinstance(ann, dict):
        return HUD_DROP_INVALID_BOX
    try:
        cid = int(ann.get("class_id", -1))
    except (TypeError, ValueError):
        cid = -1
    try:
        cx = float(ann.get("cx", 0.5))
        cy = float(ann.get("cy", 0.5))
    except (TypeError, ValueError):
        # Unparseable coordinates: position cannot be verified, drop.
        return HUD_DROP_UNPARSEABLE_COORDS

    in_hud = box_center_in_minimap_hud(cx, cy)
    path_mode = is_path_learning_stem(stem)

    if path_mode and cid in WORLD_RESOURCE_CLASS_IDS:
        return HUD_DROP_PATH_WORLD_RESOURCE
    if path_mode and in_hud:
        # path_ground on minimap player-arrow = classic poison
        return HUD_DROP_PATH_IN_HUD

    # Player HP/MP bars (16, 17) strictly occupy y: 0.83..0.925 above the bottom action bar.
    # Boxes at cy > 0.925 are on the bottom skill/potion slots and are false positives.
    if cid in (16, 17):
        if cy > 0.925 or cy < 0.82 or cx < 0.20 or cx > 0.80:
            return HUD_DROP_HP_MP_OUT_OF_BAND

    if in_hud and cid not in MINIMAP_ALLOWED_CLASS_IDS:
        if remap_sprout and cid in WORLD_RESOURCE_CLASS_IDS:
            return None
        return HUD_DROP_HUD_DISALLOWED_CLASS

    # W29_05 / R18a: a 13 (skill_ui_prompt) centred inside the minimap HUD
    # widget is only a genuine F/E gather prompt when it comes from a real
    # in-pipeline detector (vision_*/material_scanner_*/minimap_*). The
    # game never renders a gather prompt underneath the minimap overlay,
    # so an UNVERIFIED caller-supplied box (extra_detections, or any
    # unverified-heuristic source) claiming 13 there is minimap HUD
    # poison injected through extra_detections — drop it. Genuine
    # detector 13s keep the F-1056 allowance above.
    if in_hud and cid == 13:
        src = str(ann.get("source", "")).strip().lower()
        if src == "extra_detections" or any(
            src == s or src.startswith(s) for s in UNVERIFIED_HEURISTIC_SOURCES
        ):
            return HUD_DROP_HUD_UNVERIFIED_PROMPT

    return None


def is_protected_box(ann: Dict[str, Any]) -> bool:
    """Kanonik kutu koruma predikatı (K-13 / P0-7) — TEK tanım.

    Dış budama (`prune_low_confidence_boxes_in_dataset`) ile dialog
    fallback budaması AYNI fonksiyonu çağırır. Eskiden dış yol
    `source` içindeki "hand" yüzünden kutuyu koruyup trusted yaparken
    fallback yalnızca `hand_label`'a bakıyordu — aynı conf 0.0 TXT
    kutu bir yolda korunup öbür yolda siliniyordu. Yeni bir koruma
    koşulu eklenirse BURAYA eklenir; hiçbir yolda kopyası tutulmaz.
    """
    if not isinstance(ann, dict):
        return False
    if ann.get("hand_label") is True:
        return True
    if ann.get("human_approved") is True:
        return True
    if ann.get("approved") is True:
        return True
    return bool(_is_hand_or_gather_seed(ann))


def _class_name_for_id(class_id: int) -> str:
    try:
        from dataset_factory.class_taxonomy import dataset_class_names

        names = dataset_class_names()
        if int(class_id) in names:
            return str(names[int(class_id)])
    except Exception:
        pass
    return f"class_{int(class_id)}"


def _material_tokens(text: str) -> set:
    """Whole-word material tokens of a stem/class name (no substring hits)."""
    return {t for t in re.split(r"[^a-z0-9]+", str(text or "").strip().lower()) if t}


def is_material_conflict(stem: str, class_name: str, class_id: int = -1) -> bool:
    """Detect if box class completely conflicts with stem material or is an invalid essence on 3D frame."""
    s_lower = str(stem or "").strip().lower()
    c_lower = str(class_name or "").strip().lower()
    if not s_lower:
        return False

    is_3d = any(k in s_lower for k in ("3d", "crest", "boot", "gather", "ref_"))
    # Essence is a name, not a class-id range: canonical 24/25/26 are
    # buff_scroll / kinah / player_character, never essence.
    is_essence = "essence" in _material_tokens(c_lower)
    if is_essence and is_3d:
        return True

    # Identify material tokens present in stem (whole-word match only:
    # substring search misfires, e.g. "tree" inside an unrelated word).
    m_in_stem = KNOWN_MATERIAL_TOKENS_SET & _material_tokens(s_lower)
    if not m_in_stem:
        return False

    # If stem has a known material, an essence box is conflicting
    if is_essence:
        return True

    # Check if box represents a conflicting specific material class
    # (mutable copy: class-id fallbacks below add tokens to it).
    m_in_box = set(KNOWN_MATERIAL_TOKENS_SET & _material_tokens(c_lower))
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


def _is_unverified_heuristic(box: Dict[str, Any], src: str) -> bool:
    """True if the box comes from an unverified-heuristic / omni-synthetic source.

    Single definition of the predicate: calibrate used to compute the same
    expression twice (once before the human check, once after it) and run
    is_material_conflict twice as well. The second copies were unreachable
    dead code — a non-human box with a conflict / unverified source already
    returns at the first check, and a human box already returns at the human
    check — yet their conditions were still evaluated on every call.
    """
    return bool(
        src in UNVERIFIED_HEURISTIC_SOURCES
        or any(uhs in src for uhs in UNVERIFIED_HEURISTIC_SOURCES)
        or is_omni_synthetic_source(src)
        or box.get("omni_synthetic") is True
        or box.get("synthetic_fallback") is True
    )


def calibrate_box_confidence_and_trust(
    box: Dict[str, Any],
    stem: str = "",
    emin_threshold: float = CONF_EMIN,
    trust_by_confidence: bool = False,
) -> Dict[str, Any]:
    """Calibrate box confidence and trust to prevent false >= 0.85 confident / trusted labels.

    1. Material conflict / unverified-heuristic / omni-synthetic checks run
       FIRST: a bare "approved"/"confirmed" flag must not bypass them.
       Only an explicit human decision (hand_label / human_approved)
       overrides those checks.
    2. If box.get("hand_label") or box.get("human_approved") (and not
       demoted): box["trusted"] = True, box["weak"] = False, confidence
       is KEPT as measured (annotation_confidence reads
       confidence/conf/score). P1-12 (K-32/F-1212 "never inflate"):
       güven ASLA şişirilmez — eski ``max(0.90, ...)`` kaldırıldı; el
       kutusu 0.40 güvenle de 0.40 kalır, güveni ölçülmemişse 0.0 yazılır
       ve ``confidence_fabricated=True`` işaretlenir. Çıplak
       ``approved``/``confirmed`` ve şablon (seed) izi artık insan
       sayılmaz — bu dala girmez.
    3. If stem contains a known material identifier and the box class is completely conflicting,
       or essence_* on any 3D frame:
       set box["confidence"] = min(measured_conf, CONF_MATERIAL_CAP), box["trusted"] = False, box["weak"] = True.
    4. If source in UNVERIFIED_HEURISTIC_SOURCES and not human approved:
       box["confidence"] = min(measured_conf, CONF_HEURISTIC_CAP), box["trusted"] = False, box["weak"] = True.
       Never allow unverified heuristics to be trusted=True or >= 0.85!
    5. Otherwise: high confidence alone NEVER trusts a box — a model
       score is not a human approval (sweep E-07). A box that only
       clears emin_threshold stays weak, marked
       high_confidence_candidate=True for review. Pass
       trust_by_confidence=True only where an explicit measured-
       confidence contract exists (requeue of already-labeled data).
       W47 (2026-11-10) relaxes E-07 this far and no further: layer
       evidence also trusts — a box the AI poison cleaner kept after
       passing all four layers, or a box kept by pruning, earns trust
       through trust_by_confidence=True when it clears emin_threshold
       (the vetoes above still run first). Outside those opt-in call
       sites, high confidence alone still never trusts.
       A fabricated confidence (confidence_fabricated=True) is never
       evidence either, unless the box carries an explicit per-box
       human decision (sweep E-03).
    Returns calibrated box.
    """
    if not isinstance(box, dict):
        return box

    # Extract source, class, stem first — steps 2/3 must run before step 1.
    src = str(box.get("source", "")).strip().lower()
    cid = -1
    try:
        cid = int(box.get("class_id", -1))
    except (TypeError, ValueError):
        pass
    cname = str(box.get("class_name") or box.get("key") or _class_name_for_id(cid)).lower()
    stem_str = str(stem or box.get("stem", "")).strip()

    # Explicit human decision overrides the heuristic/material caps below;
    # a bare "approved"/"confirmed" flag (stale, bulk, or inherited) does not.
    # Each check runs exactly once: the material-conflict and unverified
    # predicates are evaluated here only, before the human check.
    # P0-8 (K-12): a user demote (``demoted_by_user=True``) is a newer,
    # explicit human decision than the origin flag — it vetoes the
    # hand→trusted inference, so a demoted box stays weak until a human
    # re-approves it (drag / accept clears the flag).
    is_demoted = bool(box.get("demoted_by_user") is True)
    # P3-27: dar per-box insan-onayı okuması tek kanonik yoldan
    # (``approval_state.box_has_human_provenance``); demote vetosu korunur.
    is_human = box_has_human_provenance(box) and not is_demoted
    if not is_human and is_material_conflict(stem_str, cname, cid):
        raw_c = annotation_confidence(box)
        box["confidence"] = round(min(raw_c, CONF_MATERIAL_CAP), 4)
        box["trusted"] = False
        box["weak"] = True
        return box
    if not is_human and _is_unverified_heuristic(box, src):
        raw_c = annotation_confidence(box)
        box["confidence"] = round(min(raw_c, CONF_HEURISTIC_CAP), 4)
        box["trusted"] = False
        box["weak"] = True
        return box

    # A fabricated confidence (no measurement, placeholder 0.58) is not
    # evidence: without an explicit per-box human decision the box stays
    # weak. This is the reader the confidence_fabricated flag never had
    # (sweep E-03). It runs after the vetoes above so a fabricated box
    # from a vetoed source still gets the veto's confidence cap.
    if box.get("confidence_fabricated") is True and not (
        box.get("hand_label") is True or box.get("human_approved") is True
    ):
        box["trusted"] = False
        box["weak"] = True
        return box

    # 1. Human approved check (demoted boxes excluded: P0-8/K-12).
    # TEK insan tanımı: yalnızca hand_label / human_approved. Çıplak
    # approved/confirmed ve şablon (seed) izi insan sayılmaz (P1-12/K-34)
    # — o kutular aşağıdaki standart dala düşer. Güven ŞİŞİRİLMEZ
    # (F-1212 "never inflate"): ölçülmüş güven aynen korunur,
    # ölçülmemiş güven 0.0 yazılır ve uydurma işaretlenir.
    if is_human:
        cur_conf = annotation_confidence(box)
        if not has_measured_confidence(box):
            box["confidence_fabricated"] = True
        box["trusted"] = True
        box["weak"] = False
        box["confidence"] = round(cur_conf, 4)
        # K-15: köken damgası — güvenin insan kararından geldiği kayda
        # geçer (TXT el etiketinin güveni 0.0'dır; damga yoksa 0.90
        # "model skoru" sanılır). Varsa üzerine yazılmaz.
        if box.get("hand_label") is True:
            box.setdefault("confidence_origin", "hand_label")
        elif box.get("human_approved") is True:
            box.setdefault("confidence_origin", "human_approved")
        else:
            box.setdefault("confidence_origin", "gather_seed")
        return box

    # NOTE: the material-conflict and unverified-heuristic caps above
    # already ran (and returned) before the human branch, on the same
    # inputs — a second copy of those checks here could never fire, so
    # it was removed (dead code). Only the standard check remains below.

    # 2. Standard confidence vs emin_threshold check. High confidence
    # alone never trusts: a model score is not a human approval (E-07).
    # P1-12 (K-32): güven confidence/conf/score anahtarlarından okunur
    # (annotation_confidence); ölçülmemiş güven 0.0'dır, 0.58 varsayılanı yok.
    cur_conf = annotation_confidence(box)
    eth = float(emin_threshold)
    if cur_conf >= eth and bool(trust_by_confidence):
        box["confidence"] = round(cur_conf, 4)
        box["trusted"] = True
        box["weak"] = False
        box.pop("high_confidence_candidate", None)
    else:
        box["confidence"] = round(cur_conf, 4)
        box["trusted"] = False
        box["weak"] = True
        if cur_conf >= eth:
            box["high_confidence_candidate"] = True
        else:
            box.pop("high_confidence_candidate", None)

    return box


# ----------------------------------------------------------------------
# P0-2 (K-08): yıkıcı yazımlara tek nesil yedek (.bak).
#
# Yıkıcı yazım (commit / _rewrite_label_txt / purge / temizlik / budama /
# requeue) üzerine yazmadan ÖNCE dosyanın tek nesil .bak kopyasını alır:
# ``foo.txt`` -> ``foo.txt.bak``, ``foo.json`` -> ``foo.json.bak``.
# .bak'ın soneki .bak olduğu için ``*.txt`` / ``*.json`` glob'larına
# girmez; tarama, sayaç ve eğitim hiçbir .bak dosyasını görmez.
#
# Kullanıcı kararı (11 Kasım): REDDET kalıcı silmedir — reddedilen
# kare kuyruğa dönmez, çöp/geri yükleme YOKTUR. Reddet yolları
# (tekli/toplu red, tarama-GC) dosyaları düz ``unlink`` ile siler;
# rejected_stems kaydı kalır (kuyruk filtresi + GC için), ama
# kaldırma/geri yükleme yolu yoktur. .bak güvenlik ağı SADECE
# istem-dışı yıkıcı yazımlar (commit, purge / temizlik / budama /
# requeue) için kalır; reddet bilinçli işlemdir, kapsamaz.
# ----------------------------------------------------------------------

BACKUP_SUFFIX = ".bak"


def backup_path_for(path: Union[str, Path]) -> Path:
    """Tek nesil yedek yolu: ``foo.txt`` -> ``foo.txt.bak``."""
    p = Path(path)
    return p.with_name(p.name + BACKUP_SUFFIX)


def write_backup(path: Union[str, Path]) -> Optional[Path]:
    """Varsa dosyanın .bak kopyasını yaz; yoksa None döner.

    Yedek yazılamazsa (salt-okunur disk vb.) warning loglayıp None
    döner — yazımın kendisi atomik olduğu için yarım dosya oluşmaz,
    ama o yazımın yedeği yoktur.
    """
    p = Path(path)
    if not p.is_file():
        return None
    bak = backup_path_for(p)
    try:
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(p), str(bak))
    except OSError as exc:
        logger.warning("Yedek yazılamadı (%s -> %s): %s", p, bak, exc)
        return None
    return bak


def _backup_before_overwrite(path: Union[str, Path]) -> Optional[Path]:
    return write_backup(path)


def restore_backup(path: Union[str, Path]) -> bool:
    """``.bak`` varsa içeriği dosyaya geri yazar; yoksa False döner."""
    p = Path(path)
    bak = backup_path_for(p)
    if not bak.is_file():
        return False
    try:
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        shutil.copyfile(str(bak), str(tmp_path))
        os.replace(str(tmp_path), str(p))
    except OSError as exc:
        logger.warning("Yedekten geri yükleme başarısız (%s): %s", p, exc)
        return False
    return True


def load_rejected_stems(dataset_dir: Union[str, Path]) -> List[str]:
    """``rejected_stems.json``'daki stem listesi (yoksa boş)."""
    path = Path(dataset_dir) / "rejected_stems.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(s).strip() for s in data if str(s).strip()]


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(str(tmp_path), str(path))
    except Exception:
        # Never fall back to a direct (non-atomic) write: a half-written
        # review/txt file would silently corrupt the dataset. Clean up
        # the temp file and let the caller see the failure.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(path, content, encoding="utf-8")


def _sanitize_stem(raw: Any, fallback: str = "") -> str:
    """Review JSON `stem` alanını dosya adı olarak güvenli hale getir.

    P2-25 (bölüm 8, F077-K2): `stem` payload'dan alınıp
    ``labels/<split>/<stem>.txt`` yoluna konuyordu; ``../../`` içeren
    bir stem `mkdir(parents=True)` ile dizin de oluşturarak dataset
    dışına yazabilirdi. Ayırıcı (`/`, `\\`), `..`, NUL ve boş değer
    kabul edilmez; geçersiz stem'de dosya adı olan `fallback`
    (review dosyasının kendi stem'i) kullanılır.
    """
    candidate = str(raw or "").strip()
    if not candidate:
        candidate = str(fallback or "").strip()
    if (
        not candidate
        or candidate in (".", "..")
        or "/" in candidate
        or "\\" in candidate
        or "\x00" in candidate
        or Path(candidate).name != candidate
    ):
        safe_fallback = str(fallback or "").strip()
        if (
            safe_fallback
            and safe_fallback not in (".", "..")
            and "/" not in safe_fallback
            and "\\" not in safe_fallback
            and "\x00" not in safe_fallback
            and Path(safe_fallback).name == safe_fallback
        ):
            return safe_fallback
        raise ValueError(f"Güvensiz stem reddedildi: {raw!r}")
    return candidate


def _label_txt_path(labels_dir: Path, stem: str) -> Path:
    """``labels/<split>/<stem>.txt`` — yolun labels dizini altında kaldığını doğrular.

    P2-25: sanitize edilmiş stem bile çözümlenmiş yolun (örn. sembolik
    bağ üzerinden) dizin dışına çıkmasını engellemek için `resolve()` +
    `is_relative_to` kontrolü yapılır; dışarı çıkan yol yazılmaz.
    """
    target = (Path(labels_dir) / f"{stem}.txt").resolve()
    base = Path(labels_dir).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Etiket yolu dataset dışında: {target}")
    return target


def _rewrite_label_txt(
    label_path: Path,
    trusted: Sequence[Dict[str, Any]],
    *,
    allow_empty: bool = False,
) -> bool:
    """labels TXT'yi trusted kutulardan yazar; boş emit'i reddeder.

    Emit boşsa dosyayı 0 bayt yazmak yerine yazmayı reddedip warning
    loglar (K-01: boş-yazma, 21:24 kaybının normal akışıydı — veri yok
    etmektense eski TXT kalsın). ``True`` yazıldı, ``False`` reddedildi.
    Gerçekten boşaltılması gereken kare (kullanıcı tüm kutuları sildi)
    için ``allow_empty=True`` geçilmeli — o çağrı noktası P0-1 dışında
    ayrıca ele alınacak (boşaltma + uzlaştırma birlikte diriltme yapar).

    Red yalnızca MEVCUT ve DOLU bir TXT'yi korur: dosya yoksa (veya
    zaten boşsa) kaybolacak veri yoktur — boş emit boş TXT oluşturur
    (fixup P0-1: commit/relabel "eğitim TXT'si boş" sözleşmesi; yoksa
    onaylanmamış learn/relabel karelerinde dosya hiç oluşmaz ve
    ``read_text``/senkronizasyon kırılır).
    """
    label_path = Path(label_path)
    # A box without an explicit, parseable class id must never reach the
    # txt: _yolo_line would otherwise write it as class -1 (sweep E-05 —
    # missing field, template value). Geometry is filtered the same way.
    lines = [
        _yolo_line(b)
        for b in trusted
        if all(k in b for k in ("cx", "cy", "w", "h"))
        and _safe_class_id(b.get("class_id"), -1) >= 0
    ]
    if not lines and not allow_empty:
        try:
            _existing_nonempty = label_path.is_file() and label_path.stat().st_size > 0
        except OSError:
            _existing_nonempty = False
        if _existing_nonempty:
            logger.warning(
                "[KORUMA] %s boş emit ile yazılmadı — eski TXT korundu "
                "(0 bayt yazımı veri kaybıdır, K-01)",
                label_path,
            )
            return False
    if not lines:
        # P2-21 (K-07): emit boşsa ve yazım GERÇEKTEN yapılacaksa
        # (yukarıdaki koruma dalı reddetmediyse) 0 bayt yazım warning
        # ile iz bırakır — 0 bayt TXT sessizce oluşmaz.
        logger.warning(
            "[KORUMA] %s boş emit ile 0 bayt yazılıyor (allow_empty=%s) — "
            "karede TXT'ye giren kutu yok",
            label_path,
            bool(allow_empty),
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = ("\n".join(lines) + "\n") if lines else ""
    # P0-2 (K-08): üzerine yazmadan önce tek nesil .bak.
    _backup_before_overwrite(label_path)
    _atomic_write_text(label_path, content, encoding="utf-8")
    return True


def write_review_pair(
    *,
    review_path: Union[str, Path],
    label_path: Union[str, Path],
    review_payload: Dict[str, Any],
    stem: str = "",
    floor: float = TRAIN_CONF_FLOOR,
    allow_empty: bool = False,
    emit_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """TEK YAZICI (P3-26): review JSON + labels TXT çiftini tek yerden yazar.

    JSON+TXT çift yazımı yapan her yol (commit, purge, temizlik, budama,
    demote, persist/sync, relabel, recalibrate) bu fonksiyondan geçer;
    el yapımı ikinci yazım kalmaz. Emit kuralı tektir:
    ``train_txt_boxes`` — syncer ile commit artık aynı kutuları yazar
    (F026-02). Her iki dosya da atomik yazılır (.tmp + os.replace);
    üzerine yazmadan önce her ikisinin de tek nesil .bak'ı alınır (P0-2);
    TXT uzlaştırması (P0-1) ve boş-emit reddi ``_rewrite_label_txt``'te
    korunur.

    Sıra TXT-önce-JSON-sonra: commit'in tarihsel sırasıdır. TXT yazımı
    boş-emit yüzünden reddedilirse (``allow_empty=False``) JSON yine de
    yazılır ve ``txt_written=False`` döner — bugünkü commit davranışı
    aynen korunur; reddin ardından bayat TXT satırının uzlaştırmada
    trusted diye dirilmesi (K-01/K-02) P0-1'in ayrı takip işidir.
    Bilerek boşaltan yollar (purge / temizlik / budama / demote /
    persist / recalibrate) ``allow_empty=True`` geçer.

    ``emit_boxes``: bu payload için önceden hesaplanmış
    ``train_txt_boxes`` çıktısı. Yalnızca commit_split_labels geçer —
    ``learn_auto_in_train`` payload alanını yazımdan önce hesaplamak
    için emit'e erken ihtiyaç duyar. Verilirse aynen kullanılır;
    verilmezse burada hesaplanır. Başka hiçbir çağrı noktası kendi
    emit'ini hesaplamaz.
    """
    review_path = Path(review_path)
    label_path = Path(label_path)
    if emit_boxes is None:
        trusted = review_payload.get("trusted") or []
        weak = review_payload.get("weak") or []
        emit = train_txt_boxes(trusted, weak, stem=stem, floor=float(floor))
    else:
        emit = list(emit_boxes)
    txt_written = _rewrite_label_txt(label_path, emit, allow_empty=allow_empty)
    # P0-2 (K-08): review JSON üzerine yazılmadan önce .bak.
    _backup_before_overwrite(review_path)
    _atomic_write_json(review_path, review_payload)
    return {
        "txt_written": bool(txt_written),
        "emit_boxes": emit,
        "txt_box_count": len(emit),
        "review_path": str(review_path),
        "label_path": str(label_path),
    }


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
    corrupt_files = 0
    labels_rewritten = 0
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
                # Corrupt review file: previously skipped silently, so a
                # growing pile of unreadable reviews was invisible in the
                # return stats. Count it; the file is left untouched.
                corrupt_files += 1
                continue
            if not isinstance(payload, dict):
                corrupt_files += 1
                continue
            weak = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            if not weak:
                continue
            stem = _sanitize_stem(payload.get("stem"), path.stem)
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
            # P3-26 (F026-01): purge yalnızca JSON yazıyordu — düşürülen
            # weak kutu TXT'de kalıp eğitime giriyordu. Tek yazıcıdan
            # geçir: TXT, kanonik emit kuralıyla yeniden yazılır.
            # allow_empty=True: purge bilerek yıkıcıdır; emit boşsa TXT
            # de boşalır — yoksa bayat TXT satırı uzlaştırmada trusted
            # el etiketi diye dirilir (K-01/K-02).
            pair = write_review_pair(
                review_path=path,
                label_path=root / "labels" / split / f"{stem}.txt",
                review_payload=payload,
                stem=stem,
                floor=float(floor),
                allow_empty=True,
            )
            labels_rewritten += int(bool(pair["txt_written"]))
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
        "labels_rewritten": labels_rewritten,
        "weak_dropped": dropped,
        "files_weak_empty": emptied,
        "corrupt_files": corrupt_files,
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
    corrupt_files = 0
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
                corrupt_files += 1
                continue
            if not isinstance(payload, dict):
                corrupt_files += 1
                continue
            stem = _sanitize_stem(payload.get("stem"), path.stem)
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
                        # Demote must REALLY demote: never raise confidence.
                        # The old learn-frame branch used max(conf, 0.50),
                        # lifting the box above the train floor, so the
                        # "demoted" box re-entered the txt through the
                        # learn weak path (sweep E-02).
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
                # filter_hud_false_positives always returns a copy, so
                # identity ("bb is not b") is always True; compare content.
                if bb != b:
                    changed = True

            if (
                len(new_trusted) == len(trusted_in)
                and len(new_weak) == len(weak_in)
                and not changed
            ):
                continue

            payload["trusted"] = new_trusted
            payload["weak"] = new_weak
            # P3-26: tek yazıcı — JSON+TXT aynı çağrıda, kanonik emit
            # kuralıyla. allow_empty=True: demote bilerek yıkıcıdır
            # (sentetik kutu TXT'den de çıkar); emit boşsa TXT de
            # boşalır — yoksa bayat satır uzlaştırmada trusted diye
            # dirilir (K-01/K-02).
            write_review_pair(
                review_path=path,
                label_path=labels_dir / f"{stem}.txt",
                review_payload=payload,
                stem=stem,
                floor=float(floor),
                allow_empty=True,
            )
            files_rewritten += 1

    weak_stats = purge_poisoned_weak_reviews(root, drop_sub_floor=True, floor=floor)
    # Do not add the purge pass's file count to this pass's: the same review
    # files are rewritten by both passes, so summing double-counts them.
    # The purge count is reported separately below.
    total_rewritten = files_rewritten
    if total_rewritten > 0 or demoted_trusted > 0:
        mark_model_stale(
            reason=f"Omni sentetik etiketler düşürüldü ({demoted_trusted} kutu, {total_rewritten} dosya)",
            dataset_dir=root,
        )
    return {
        "scanned": scanned,
        "files_rewritten": total_rewritten,
        "weak_purge_files_rewritten": int(weak_stats.get("files_rewritten", 0)),
        "demoted_trusted": demoted_trusted,
        "dropped_boxes": dropped_boxes + int(weak_stats.get("weak_dropped", 0)),
        "weak_purged": weak_purged,
        # Do NOT add the purge pass's corrupt count: it re-scans the same
        # files, so the same corrupt file would be counted twice (same
        # reason files_rewritten is not summed above).
        "corrupt_files": corrupt_files,
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


# ----------------------------------------------------------------------
# K-19: düz-zemin (Laplacian varyansı) eşiği sınıfa/büyüklüğe kademeli.
# Eski kural tek eşikti (lap_var < 10.0, tek muafiyet sınıf 15):
# küçük/uzak kutular ile düz gölgeli renderlar sessizce siliniyordu.
# - Düz olması DOĞASI gereği beklenen sınıflar muaf: 15 path_ground
#   (zemin) + 16/17/18 HP/MP/hedef barları (düz dolgulu UI çubukları).
# - Geri kalanlarda eşik kutu alanına (w*h, normalize) göre kademelenir:
#   küçük kutuda birkaç pikselin varyansı anlamsızdır, o yüzden eşik
#   düşer (küçük <0.002 → 3.0, orta <0.01 → 6.0, büyük → 10.0).
# ----------------------------------------------------------------------
FLAT_TEXTURE_EXEMPT_CLASSES: frozenset = frozenset({15, 16, 17, 18})
FLAT_TEXTURE_SMALL_AREA = 0.002
FLAT_TEXTURE_MEDIUM_AREA = 0.01


def _flat_texture_threshold(box: Dict[str, Any]) -> Optional[float]:
    """Düz-zemin silme eşiği (K-19); ``None`` = sınıf muaf, silme yok."""
    try:
        cid = int(box.get("class_id", -1))
    except (TypeError, ValueError):
        cid = -1
    if cid in FLAT_TEXTURE_EXEMPT_CLASSES:
        return None
    try:
        area = float(box.get("w", 0.0)) * float(box.get("h", 0.0))
    except (TypeError, ValueError):
        area = 0.0
    if area < FLAT_TEXTURE_SMALL_AREA:
        return 3.0
    if area < FLAT_TEXTURE_MEDIUM_AREA:
        return 6.0
    return 10.0


def _nms_dedupe(boxes: List[Dict[str, Any]], iou_thresh: float = 0.65) -> Tuple[List[Dict[str, Any]], int]:
    # P1-13 (K-33): 0.65 bir BASTIRMA esigidir, kimlik degil — same_box
    # (0.45 + merkez) ile karistirilmaz; esik aynen kalir. IoU hesabi
    # tek kopyadir (box_match.iou_xyxy; _iou_xyxy ona delege eder).
    # Parse each box ONCE (xyxy, class, trust flag, confidence) and sort
    # indices by rank. The old loop called slot_iou per pair, which
    # re-parsed both dicts (float() + dict .get per field) on every
    # comparison — O(n^2) parses for n boxes. Rank order is unchanged:
    # (trusted, confidence) desc, stable sort keeps input order on ties.
    # Kept boxes are also bucketed by class: suppression only ever fires
    # within one class, so a candidate is compared against same-class kept
    # boxes only instead of the whole kept list. The decision is identical
    # (same rank order, same per-class kept order, same IoU test).
    # K-15: hand_label MUTLAK muaf — el etiketi hiçbir koşulda çakışan
    # model kutu tarafından bastırılmaz (güveni 0.0 olan TXT el etiketi
    # eskiden NMS'te model kutuya yeniliyordu). Muaf kutu kept listesine
    # yine girer, yani kendinden SONRA gelen aynı-sınıf model kutuları
    # bastırmaya devam eder; yalnızca kendisi bastırılamaz.
    parsed: List[Tuple[Dict[str, Any], Tuple[float, float, float, float], int, float, float, bool]] = []
    for b in boxes:
        is_t = 1.0 if (b.get("trusted") is True or b.get("hand_label") is True) else 0.0
        parsed.append(
            (b, _box_xyxy(b), _safe_class_id(b.get("class_id", -1)), is_t, annotation_confidence(b),
             b.get("hand_label") is True)
        )
    order = sorted(range(len(parsed)), key=lambda i: (parsed[i][3], parsed[i][4]), reverse=True)
    kept_idx: List[int] = []
    kept_by_class: Dict[int, List[int]] = {}
    dropped = 0
    for i in order:
        _b, b_xyxy, b_cid, _t, _c, b_hand = parsed[i]
        suppress = False
        if not b_hand:
            for k in kept_by_class.get(b_cid, ()):
                if _iou_xyxy(b_xyxy, parsed[k][1]) > iou_thresh:
                    suppress = True
                    dropped += 1
                    break
        if not suppress:
            kept_idx.append(i)
            kept_by_class.setdefault(b_cid, []).append(i)
    return [parsed[i][0] for i in kept_idx], dropped


def clean_poison_labels_ai(
    dataset_dir: Union[str, Path],
    *,
    exemplar_memory: Optional[Any] = None,
    floor: float = TRAIN_CONF_FLOOR,
    emin_threshold: float = CONF_EMIN,
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
    W47 (2026-11-10): katmanlardan geçip tutulan kutu trusted sayılır
    (E-07 gevşetmesi) — final calibrate trust_by_confidence=True ile
    çalışır; veto'lar ve emin eşiği aynen korunur.
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
    # B-13: okunamayan/bozuk review JSON'u "taranmış" sayılmaz; ayrı sayılır.
    unreadable_files = 0
    unreadable_file_names: list = []
    manifest_error: Optional[str] = None
    files_rewritten = 0
    labels_rewritten = 0
    protected_boxes = 0
    auto_repaired_boxes = 0
    poison_deleted_boxes = 0
    negative_taught_boxes = 0
    negative_match_deleted = 0
    hud_poison_deleted = 0
    hud_drop_reasons: Dict[str, int] = {}
    flat_texture_deleted = 0
    omni_demoted_boxes = 0
    sub_floor_deleted = 0
    deduped_boxes = 0
    boxes_without_visual = 0

    # [YOLO-BULK-ASYNC] İlerleme kümülatiftir (train+val toplamı): eskiden
    # her split kendi sayacını sıfırlıyordu (val başında "1/N"e dönüyordu),
    # overlay "132/1180" sayacı artık tek yönlü ilerler. İmza aynı
    # (opsiyonel kwarg), mevcut çağrılar bozulmaz.
    grand_total = 0
    if progress_callback:
        try:
            for _s in ("train", "val"):
                _d = root / "review" / _s
                if _d.is_dir():
                    grand_total += sum(
                        1 for _f in _d.glob("*.json") if _f.name != _SEED_NAME
                    )
        except Exception:
            grand_total = 0
    progress_offset = 0

    for split in ("train", "val"):
        review_dir = root / "review" / split
        labels_dir = root / "labels" / split
        if not review_dir.is_dir():
            continue

        review_files = sorted(review_dir.glob("*.json"))
        total_files = len(review_files)
        seen_in_split = 0

        for idx, path in enumerate(review_files):
            if path.name == _SEED_NAME:
                continue
            seen_in_split += 1
            if progress_callback:
                try:
                    if grand_total > 0:
                        progress_callback(progress_offset + seen_in_split, grand_total)
                    else:
                        progress_callback(idx + 1, total_files)
                except Exception:
                    pass

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                # B-13: bozuk dosya "taranmış" sayılmaz; atlanır ama sayılır.
                unreadable_files += 1
                if len(unreadable_file_names) < 20:
                    unreadable_file_names.append(path.name)
                logger.warning("Bozuk review JSON atlandı (temizlenmedi): %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                unreadable_files += 1
                if len(unreadable_file_names) < 20:
                    unreadable_file_names.append(path.name)
                logger.warning("Review JSON dict değil, atlandı (temizlenmedi): %s", path)
                continue
            scanned += 1

            stem = _sanitize_stem(payload.get("stem"), path.stem)
            trusted_in = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
            weak_in = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]
            # K-16: temizlik önce TXT'yi okuyup uzlaştırır (P0-1'deki
            # _reconcile_txt_hand_labels) — JSON'da bayraksız kalan el
            # etiketi TXT'den trusted'a taşınır; temizlik onu silemez,
            # 0 bayt TXT üzerine yazamaz. TXT-only kare (JSON yok) bu
            # döngüde görülmez (yalnızca review JSON'ları taranır).
            trusted_in, weak_in = _reconcile_txt_hand_labels(
                trusted_in, weak_in, labels_dir / f"{stem}.txt"
            )
            all_boxes = list(trusted_in) + list(weak_in)
            if not all_boxes:
                continue

            # K-16: kare-bazlı değişim takibi — 0 silmede dosyaya
            # dokunulmaz (aşağıdaki yazım kapısı bu sayaçlara bakar).
            frame_deleted_before = poison_deleted_boxes
            frame_repaired_before = auto_repaired_boxes

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
                cid = _safe_class_id(b.get("class_id", 0), 0)
                cname = str(b.get("class_name", ""))
                src = str(b.get("source", "")).lower()

                crop: Optional[np.ndarray] = None
                if frame is not None:
                    try:
                        crop = extract_box_crop(frame, b)
                    except Exception:
                        crop = None

                # Katman 0: Dokunulmazlık Kalkanı (hand_label / human_approved / seed)
                # P1-12 (K-34): çıplak "approved" boru hattı bayrağıdır —
                # dokunulmazlık vermez, katmanlardan geçer.
                is_immune = (
                    b.get("hand_label") is True
                    or b.get("human_approved") is True
                    or _is_hand_or_gather_seed(b)
                )

                if is_immune:
                    protected_boxes += 1
                    b["trusted"] = True
                    b["weak"] = False
                    # K-15: güven 0.0'da bırakılmaz — kutu köken damgası
                    # taşır. TXT el etiketinin güven sütunu yoktur (0.0);
                    # calibrate güveni 0.90'a çekerken bu damga NEDENİ
                    # kayda geçirir (insan kararı, model skoru değil).
                    if b.get("hand_label") is True:
                        b["confidence_origin"] = "hand_label"
                    elif b.get("human_approved") is True:
                        b["confidence_origin"] = "human_approved"
                    elif _is_hand_or_gather_seed(b):
                        b["confidence_origin"] = "gather_seed"
                    candidate_kept.append(b)
                    continue

                # Katman 1 (kayıtlı kanıt önce): box'a kayıtlı exemplar
                # benzerlikleri (exemplar_*_sim alanları) HUD/flat silmeden
                # ÖNCE çalışır — bunlar kutuya ait kaydedilmiş kanıttır.
                # Canlı hafıza (match_positive) eşleşmesi ise görsel
                # kontrollerden SONRA değerlendirilir: düz gri bir crop,
                # histogramla herhangi bir düz exemplara 1.0 benzer; bu
                # sahte eşleşme flat-zemin zehir kontrolünü yenemez.
                has_field_sim = (
                    "exemplar_repair_sim" in b
                    or "exemplar_pos_sim" in b
                    or "exemplar_neg_sim" in b
                )
                pos_sim = 0.0
                pos_cid: Optional[int] = None
                pos_cname: Optional[str] = None
                neg_sim = 0.0
                if has_field_sim:
                    if "exemplar_repair_sim" in b:
                        pos_sim = _safe_float(b.get("exemplar_repair_sim"), 0.0)
                        pos_cid = _safe_class_id(b.get("exemplar_repair_class_id", cid), cid)
                        pos_cname = str(b.get("exemplar_repair_class_name", f"class_{pos_cid}"))
                    elif "exemplar_pos_sim" in b:
                        pos_sim = _safe_float(b.get("exemplar_pos_sim"), 0.0)
                    if "exemplar_neg_sim" in b:
                        neg_sim = _safe_float(b.get("exemplar_neg_sim"), 0.0)

                is_omni_box = (
                    is_omni_synthetic_source(src)
                    or b.get("omni_synthetic") is True
                    or b.get("synthetic_fallback") is True
                    or src in SYNTHETIC_FALLBACK_SOURCES
                )

                if has_field_sim and neg_sim >= 0.82 and (neg_sim - pos_sim) >= 0.15:
                    poison_deleted_boxes += 1
                    negative_match_deleted += 1
                    continue

                if has_field_sim and pos_sim >= 0.72 and not is_omni_box:
                    # Omni/synthetic boxes never earn exemplar protection;
                    # they fall through to the Layer-3 omni filter below.
                    if pos_sim >= 0.82 and pos_cid is not None and pos_cid not in (cid, -1):
                        b["class_id"] = pos_cid
                        b["class_name"] = pos_cname or f"class_{pos_cid}"
                        b["confidence"] = max(conf, float(pos_sim))
                        b["repaired_by_exemplar"] = True
                        auto_repaired_boxes += 1
                    protected_boxes += 1
                    if conf >= emin_threshold or pos_sim >= CONF_EMIN:
                        b["trusted"] = True
                        b["weak"] = False
                    else:
                        # Do not leave a stale trusted=True on a box that
                        # only earned weak protection.
                        b["trusted"] = False
                        b["weak"] = True
                    candidate_kept.append(b)
                    continue

                # Katman 2: Görsel Boş Zemin / Gökyüzü / HUD Zehir Kontrolü
                # (kayıtlı exemplar koruması olmayan kutular için)
                filtered_hud = filter_hud_false_positives([b], stem=stem, remap_sprout=False)
                if not filtered_hud:
                    poison_deleted_boxes += 1
                    hud_poison_deleted += 1
                    # K-19: öğretim nedene göre ayrılır — 7 farklı neden
                    # tek "poison_hud_illegal" kaynağıyla öğretilmez.
                    hud_reason = _hud_drop_reason(b, stem=stem) or "unknown"
                    hud_drop_reasons[hud_reason] = hud_drop_reasons.get(hud_reason, 0) + 1
                    # Path-stem'deki gerçek herb/ore crop'u
                    # (path_world_resource) ve doğrulanamayan kutu asla
                    # negatif öğretilmez; sayaç yalnızca BAŞARILI
                    # öğretimi sayar (add_negative_exemplar False
                    # dönerse — dedup, kapı reddi, save hatası — artmaz).
                    if (
                        teach_negatives
                        and not dry_run
                        and exemplar_memory is not None
                        and crop is not None
                        and crop.size > 0
                        and hud_reason not in HUD_NO_NEGATIVE_TEACH_REASONS
                    ):
                        try:
                            if exemplar_memory.add_negative_exemplar(
                                crop, cid, cname, source=f"poison_hud_{hud_reason}"
                            ):
                                negative_taught_boxes += 1
                        except Exception:
                            pass
                    continue

                lap_var: Optional[float] = None
                if crop is not None and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                    try:
                        cgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        lap_var = float(cv2.Laplacian(cgray, cv2.CV_32F).var())
                    except Exception:
                        lap_var = None
                if frame is None:
                    # No visual at all: the flat-texture check did NOT run.
                    # Count it explicitly instead of silently passing.
                    boxes_without_visual += 1

                # K-19: düz-zemin eşiği sınıfa/büyüklüğe kademeli
                # (_flat_texture_threshold) — tek muafiyet sınıf 15 değil.
                flat_thresh = _flat_texture_threshold(b)
                if lap_var is not None and flat_thresh is not None and lap_var < flat_thresh:
                    poison_deleted_boxes += 1
                    flat_texture_deleted += 1
                    if teach_negatives and not dry_run and exemplar_memory is not None and crop is not None and crop.size > 0:
                        try:
                            # Sayaç yalnızca BAŞARILI öğretimi sayar
                            # (dedup / kapı reddi / save hatası artırmaz).
                            if exemplar_memory.add_negative_exemplar(
                                crop, cid, cname, source="poison_flat_texture"
                            ):
                                negative_taught_boxes += 1
                        except Exception:
                            pass
                    continue

                # Katman 1 (canlı hafıza): kayıtlı benzerliği olmayan kutular
                # için hafızayı şimdi sorgula — görsel kontrollerden sonra.
                if not has_field_sim:
                    if exemplar_memory is not None and crop is not None and crop.size > 0:
                        try:
                            pos_sim, pos_cid, pos_cname, _ = exemplar_memory.match_positive(crop)
                            pos_sim = _safe_float(pos_sim, 0.0)
                        except Exception:
                            pos_sim = 0.0
                        try:
                            neg_sim, _ = exemplar_memory.match_negative(crop, class_id=cid)
                            neg_sim = _safe_float(neg_sim, 0.0)
                        except Exception:
                            neg_sim = 0.0

                if neg_sim >= 0.82 and (neg_sim - pos_sim) >= 0.15:
                    poison_deleted_boxes += 1
                    negative_match_deleted += 1
                    continue

                # Omni/synthetic boxes never get their class repaired either;
                # they fall through to the Layer-3 omni filter below.
                if not is_omni_box and pos_sim >= 0.82 and pos_cid is not None and pos_cid not in (cid, -1):
                    b["class_id"] = pos_cid
                    b["class_name"] = pos_cname or f"class_{pos_cid}"
                    b["confidence"] = max(conf, float(pos_sim))
                    b["repaired_by_exemplar"] = True
                    auto_repaired_boxes += 1

                if pos_sim >= 0.72 and not is_omni_box:
                    # Omni/synthetic boxes never earn exemplar protection;
                    # they fall through to the Layer-3 omni filter below.
                    protected_boxes += 1
                    if conf >= emin_threshold or pos_sim >= CONF_EMIN:
                        b["trusted"] = True
                        b["weak"] = False
                    else:
                        b["trusted"] = False
                        b["weak"] = True
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

                candidate_kept.append(b)

            # NMS Deduplication
            deduped, nms_dropped = _nms_dedupe(candidate_kept, iou_thresh=0.65)
            deduped_boxes += nms_dropped

            # Categorize into trusted (Emin) vs weak (Şüpheli)
            # W47 (2026-11-10, Selçuk kararı): katman kanıtı güven sayılır.
            # Bu noktaya ulaşan her kutu 4 katmanın tamamından geçmiştir
            # (dokunulmazlık / exemplar / görsel HUD-düz zemin / omni-NMS-
            # taban); o yüzden trust_by_confidence=True opt-in'i burada
            # geçerlidir — conf >= emin olan katman-kept kutu trusted olur.
            # Veto'lar calibrate içinde önce çalışır (materyal çakışması /
            # doğrulanmamış heuristik / uydurma güven olduğu gibi weak
            # kalır) ve emin altı kutu weak kalır; güven SADECE bu çağrı
            # noktasında verilir, calibrate varsayılanı (False) değişmez —
            # başka hiçbir yoldan trusted sızmaz. Kutuya human_approved /
            # hand_label damgası basılmaz (budamadaki kuralın aynısı).
            final_trusted: List[Dict[str, Any]] = []
            final_weak: List[Dict[str, Any]] = []
            for b in deduped:
                b = calibrate_box_confidence_and_trust(
                    b, stem=stem, emin_threshold=emin_threshold, trust_by_confidence=True
                )
                if b.get("trusted") is True:
                    final_trusted.append(b)
                else:
                    final_weak.append(b)

            # K-16: 0 silmede dosyaya dokunulmaz — JSON+TXT yalnızca bu
            # karede gerçekten kutu silindiyse / onarıldıysa / NMS
            # düşürdüyse yeniden yazılır. Uzlaştırma ve kalibrasyon tek
            # başına yazım tetiklemez: koruma bu çalışmada bellekte zaten
            # uygulanır, disk bir sonraki temizlikte yine uzlaştırılır.
            frame_deleted = poison_deleted_boxes - frame_deleted_before
            frame_repaired = auto_repaired_boxes - frame_repaired_before
            frame_changed = (
                frame_deleted > 0 or frame_repaired > 0 or nms_dropped > 0
            )
            if not dry_run and frame_changed:
                payload["trusted"] = final_trusted
                payload["weak"] = final_weak
                # P3-26: tek yazıcı — JSON+TXT aynı çağrıda, kanonik
                # emit kuralıyla. allow_empty=True: temizlik bilerek
                # yıkıcıdır; emit boşsa TXT de boşalır — yoksa silinen
                # zehir satır uzlaştırmada trusted diye dirilir (K-01).
                write_review_pair(
                    review_path=path,
                    label_path=labels_dir / f"{stem}.txt",
                    review_payload=payload,
                    stem=stem,
                    floor=float(floor),
                    allow_empty=True,
                )
                files_rewritten += 1
                labels_rewritten += 1

        progress_offset += seen_in_split

    if not dry_run and files_rewritten > 0:
        mark_model_stale(
            reason=f"AI Zehirli Etiket Temizleme ({poison_deleted_boxes} kutu silindi, {files_rewritten} dosya)",
            dataset_dir=root,
        )
        try:
            from dataset_factory.dataset_curator import DatasetCurator
            curator = DatasetCurator(dataset_dir=root)
            curator.rebuild_manifest_stats()
        except Exception as exc:
            # B-13: manifest yeniden kurma arızası artık ok:True ile
            # gizlenmiyor — istatistikler bayat kalır, çağıran bunu görür.
            manifest_error = f"rebuild_manifest_stats failed: {exc}"
            logger.warning("Zehir temizliği manifest yenileme hatası: %s", exc)

    # P2-21 (F030-K2, F056): katman dökümü yalnızca return dict'te
    # kalmasın — 14 yutulan alan dahil tam özet loga düşer. `ok` /
    # `manifest_error` / `unreadable_files` burada okunur; manifest
    # arızası veya okunamayan dosya varken çağıran "tüm etiketler
    # temizlendi" diyemez (dialog o metni bu alanlara bakarak seçer).
    # Not: files_rewritten "değişen dosya" değil, yeniden yazılan review
    # dosyası sayısıdır (F056-K4) — hiç kutusu değişmeyen dosya da
    # sayılır; semantik testte sabitlendiği için değiştirilmedi.
    if manifest_error is not None or unreadable_files > 0:
        logger.warning(
            "[TEMİZ] zehir temizliği eksik bitti: ok=%s manifest_error=%s "
            "okunamayan=%d %s",
            manifest_error is None,
            manifest_error,
            unreadable_files,
            unreadable_file_names[:20],
        )
    logger.info(
        "[TEMİZ] zehir temizliği özeti: ok=%s taranan=%d okunamayan=%d "
        "dosya=%d etiket_txt=%d korunan=%d onarilan=%d silinen=%d "
        "(negatif_eslesme=%d hud=%d duz_zemin=%d taban_alti=%d omni=%d "
        "nms_tekrar=%d) negatif_ogretilen=%d gorselsiz_kutu=%d",
        manifest_error is None,
        scanned,
        unreadable_files,
        files_rewritten,
        labels_rewritten,
        protected_boxes,
        auto_repaired_boxes,
        poison_deleted_boxes,
        negative_match_deleted,
        hud_poison_deleted,
        flat_texture_deleted,
        sub_floor_deleted,
        omni_demoted_boxes,
        deduped_boxes,
        negative_taught_boxes,
        boxes_without_visual,
    )
    # K-16: katman dökümü loga — 21:16'daki 342 silmenin dökümü logda
    # yoktu; hangi katmanın ne sildiği artık görünür.
    logger.info(
        "Zehir temizliği katman dökümü: taranan=%d silinen=%d "
        "(negatif-eşleşme=%d hud=%d düz-zemin=%d omni=%d taban-altı=%d) "
        "nms=%d onarılan=%d korunan=%d negatif-öğretilen=%d "
        "dosya-yazılan=%d hud-nedenleri=%s",
        scanned,
        poison_deleted_boxes,
        negative_match_deleted,
        hud_poison_deleted,
        flat_texture_deleted,
        omni_demoted_boxes,
        sub_floor_deleted,
        deduped_boxes,
        auto_repaired_boxes,
        protected_boxes,
        negative_taught_boxes,
        files_rewritten,
        dict(hud_drop_reasons),
    )

    return {
        "ok": manifest_error is None,
        "manifest_error": manifest_error,
        "scanned": scanned,
        "unreadable_files": unreadable_files,
        "unreadable_file_names": unreadable_file_names,
        "files_rewritten": files_rewritten,
        "labels_rewritten": labels_rewritten,
        "protected_boxes": protected_boxes,
        "auto_repaired_boxes": auto_repaired_boxes,
        "poison_deleted_boxes": poison_deleted_boxes,
        "negative_taught_boxes": negative_taught_boxes,
        "negative_match_deleted": negative_match_deleted,
        "hud_poison_deleted": hud_poison_deleted,
        "hud_drop_reasons": dict(hud_drop_reasons),
        "flat_texture_deleted": flat_texture_deleted,
        "omni_demoted_boxes": omni_demoted_boxes,
        "sub_floor_deleted": sub_floor_deleted,
        "deduped_boxes": deduped_boxes,
        "boxes_without_visual": boxes_without_visual,
        "floor": float(floor),
        "emin_threshold": float(emin_threshold),
    }

def prune_low_confidence_boxes_in_dataset(
    dataset_dir: Union[str, Path],
    threshold: float = CONF_EMIN,
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

    Trust basis (E-07): pruning itself is the human decision — the user
    picks the threshold, so a box that clears it is kept by an explicit
    measured-confidence contract, not by a model score alone. The
    calibrate call below therefore passes ``trust_by_confidence=True``.
    Calibrate's vetoes still run first: a material-conflict /
    unverified-heuristic / fabricated-confidence box stays weak (kept,
    not pruned) instead of being promoted. Kept boxes are NOT stamped
    ``human_approved`` — a threshold choice is not a per-box inspection,
    so it must not bypass those vetoes on a later calibration.
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
    # B-13: okunamayan/bozuk review JSON'u "taranmış" sayılmaz; ayrı sayılır.
    unreadable_files = 0
    unreadable_file_names: list = []
    manifest_error: Optional[str] = None
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

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                # B-13: bozuk dosya "taranmış" sayılmaz; atlanır ama sayılır.
                unreadable_files += 1
                if len(unreadable_file_names) < 20:
                    unreadable_file_names.append(path.name)
                logger.warning("Bozuk review JSON atlandı (budanmadı): %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                unreadable_files += 1
                if len(unreadable_file_names) < 20:
                    unreadable_file_names.append(path.name)
                logger.warning("Review JSON dict değil, atlandı (budanmadı): %s", path)
                continue
            scanned += 1

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
                # P0-7/K-13: kanonik predikat — dialog fallback budaması
                # da is_protected_box'u çağırır (aynı kutuda zıt karar yok).
                is_protected = is_protected_box(b)

                if is_protected or conf >= th:
                    # Pruning is a human decision (the user chose the
                    # threshold), so the kept box is trusted through the
                    # explicit measured-confidence contract: pass
                    # trust_by_confidence=True (E-07 opt-in). Calibrate's
                    # vetoes still run first — a material-conflict /
                    # unverified-heuristic / fabricated-confidence box
                    # stays weak (kept, not pruned) instead of being
                    # promoted to trusted.
                    cal = calibrate_box_confidence_and_trust(
                        dict(b), stem=stem, emin_threshold=th, trust_by_confidence=True
                    )
                    if cal.get("trusted") is True:
                        new_trusted.append(cal)
                        promoted_boxes += 1
                    else:
                        new_weak.append(cal)
                else:
                    frame_pruned += 1
                    pruned_boxes += 1
                    if record_negative_exemplars and exemplar_memory is not None and frame is not None:
                        try:
                            crop = extract_box_crop(frame, b)
                            if crop is not None and crop.size > 0:
                                cid = _safe_class_id(b.get("class_id", 0), 0)
                                cname = str(b.get("class_name", ""))
                                exemplar_memory.add_negative_exemplar(
                                    crop, cid, cname, source="bulk_prune_low_conf"
                                )
                        except Exception:
                            pass

            # P0-7/K-13 (bölüm 3.2): içerik karşılaştırması — terfi +
            # tenzil aynı karede birbirini götürürse len() farkı 0 olur
            # ama disk değişmelidir. Eski `len(new_trusted) !=
            # len(trusted_in)` koşulu o kareyi "dokunulmadı" sayıp
            # yazmıyordu; dialog da "dış budama dokunmadı" sanıp bayat
            # bellekle fallback çalıştırıyor, kalibrasyon vetosunu geri
            # alıyordu. İçerik aynıysa (yalnızca zaten-trusted kutular
            # sayıldıysa) yazım hâlâ atlanır.
            if frame_pruned > 0 or new_trusted != trusted_in or new_weak != weak_in:
                payload["trusted"] = new_trusted
                payload["weak"] = new_weak
                # P3-26: tek yazıcı — JSON+TXT aynı çağrıda, kanonik
                # emit kuralıyla. Budama kullanıcının bilerek yaptığı
                # yıkıcı işlemdir (eşik kullanıcı seçer): tüm kutular
                # budandıysa TXT de bilerek boşalır — yoksa JSON boş,
                # TXT dolu kalır ve JSON↔TXT senkronu bozulur (w41).
                # allow_empty=True yalnızca bu çağrı noktası içindir.
                write_review_pair(
                    review_path=path,
                    label_path=labels_dir / f"{stem}.txt",
                    review_payload=payload,
                    stem=stem,
                    floor=TRAIN_CONF_FLOOR,
                    allow_empty=True,
                )
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
        except Exception as exc:
            # B-13: manifest yeniden kurma arızası artık ok:True ile
            # gizlenmiyor — istatistikler bayat kalır, çağıran bunu görür.
            manifest_error = f"rebuild_manifest_stats failed: {exc}"
            logger.warning("Budama manifest yenileme hatası: %s", exc)

    return {
        "ok": manifest_error is None,
        "manifest_error": manifest_error,
        "scanned": scanned,
        "images_checked": scanned,
        "unreadable_files": unreadable_files,
        "unreadable_file_names": unreadable_file_names,
        "pruned_boxes": pruned_boxes,
        "boxes_pruned": pruned_boxes,
        "promoted_boxes": promoted_boxes,
        "preserved_frames": preserved_frames,
        "files_rewritten": files_rewritten,
        "threshold": th,
    }




def annotation_confidence(ann: Dict[str, Any]) -> float:
    """First parseable confidence among confidence/conf/score, clamped to [0, 1].

    A present-but-unparseable primary key (None, "high") used to mask a
    valid fallback and return 0.0; out-of-range values (inf, 1.5, -0.3)
    passed through unclamped and could reach JSON as Infinity. Threshold
    decisions are unchanged: every floor/threshold in this module lies
    inside [0, 1], so clamping only repairs the stored value.
    """
    if not isinstance(ann, dict):
        return 0.0
    for key in ("confidence", "conf", "score"):
        if key not in ann:
            continue
        try:
            value = float(ann[key])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN: not a measurement, try the next key
            continue
        return min(1.0, max(0.0, value))
    return 0.0


def has_measured_confidence(ann: Dict[str, Any]) -> bool:
    """True when a parseable confidence/conf/score measurement exists.

    P1-12 (K-32): ölçülmemiş güven (anahtar yok / None / sayı değil /
    NaN) 0.0 sayılmaz — 0.0 ölçüLMÜŞ sıfırdan ayrıdır. Varsayılan
    yazan yollar (requeue yer tutucusu hariç) bu bayrağı kullanır;
    emit yolları ölçülmemiş kutuyu tabana bakmadan eler.
    """
    if not isinstance(ann, dict):
        return False
    for key in ("confidence", "conf", "score"):
        if key not in ann:
            continue
        try:
            value = float(ann[key])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN: not a measurement
            continue
        return True
    return False


def is_trusted_annotation(ann: Dict[str, Any]) -> bool:
    """TEK "insan" tanımı (P1-12/K-34): yalnızca açık kutu-düzeyi insan
    kararı (hand_label / human_approved) güvenir. Çıplak ``approved`` /
    ``confirmed`` boru hattı bayrağıdır, insan kararı değildir (E-04);
    F/E şablon (seed) kutu da tek başına güvenilmez — K1 kapısından
    (train_txt_boxes) geçmesi için açık insan onayı gerekir. Omni
    sentetik kutu asla güvenilmez; doğrulanmamış heuristik kaynak da
    insan onayı olmadan güvenilmez."""
    if not isinstance(ann, dict):
        return False
    source = str(ann.get("source", "")).lower()
    if is_omni_synthetic_source(source) or ann.get("omni_synthetic") is True:
        return False
    if _is_explicitly_human_approved(ann):
        return True
    if source in UNVERIFIED_HEURISTIC_SOURCES or any(s in source for s in UNVERIFIED_HEURISTIC_SOURCES):
        return False
    return False


def _safe_class_id(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))  # e.g. "3.0"
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _box_xyxy(ann: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # Non-numeric coordinates must not crash IoU/NMS: fall back to a
    # zero-area box at the centre (IoU 0 against everything).
    try:
        cx = float(ann.get("cx", 0.5))
        cy = float(ann.get("cy", 0.5))
        w = float(ann.get("w", ann.get("width", 0.0)))
        h = float(ann.get("h", ann.get("height", 0.0)))
    except (TypeError, ValueError):
        return 0.5, 0.5, 0.5, 0.5
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU of two pre-parsed (x1, y1, x2, y2) boxes.

    P1-13 (K-33): tek kopya ``dataset_factory/box_match.py::iou_xyxy``'de
    yasar; bu ad yalnizca geriye donuk uyumluluk icin delege eder
    (``_find_slot`` / ``_nms_dedupe`` ve testler bu adi kullanir).
    Aritmetik degismedi — ``slot_iou`` ile bit-for-bit ayni.
    """
    return iou_xyxy(a, b)


def slot_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    # P1-13 (K-33): tek IoU hesabi box_match.box_iou'dadir.
    return box_iou(a, b)


def _find_slot(boxes: Sequence[Dict[str, Any]], candidate: Dict[str, Any]) -> Optional[int]:
    """Ayni slotun indisi — tek tanim ``same_box()`` (P1-13 / K-33).

    Eski davranis (IoU >= SLOT_IOU 0.30, sinif-disinda merkez sarti
    yok) kaldirildi: artik ayni sinif + IoU >= 0.45 + merkez mesafesi
    <= 0.05 gerekir. Eksik geometrili/sinifsiz aday ya da liste girdisi
    eslesmeden once elenir (``parse_box`` None / ``class_id`` None).
    Birden cok eslesme varsa en yuksek IoU'lusu doner (eski en-iyi
    secim korunur; esitlikte ilk eslesen kazanir).
    """
    cand = parse_box(candidate)
    if cand is None or cand.class_id is None:
        return None
    best_i: Optional[int] = None
    best = 0.0
    for i, box in enumerate(boxes):
        # Class-blind slot matching drops a new observation of a different
        # class at the same spot and immortalises the old trusted box.
        parsed = parse_box(box)
        if parsed is None or parsed.class_id is None:
            continue
        if parsed.class_id != cand.class_id:
            continue
        if not same_box_parsed(parsed, cand):
            continue
        score = iou_xyxy(parsed.xyxy, cand.xyxy)
        if best_i is None or score > best:
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
    except Exception as exc:
        # P1-16 (K-17): okunamayan TXT sessizce boş sayılmaz — çağıran
        # uzlaştırma boş listeyle devam eder ama kayıp logda iz bırakır.
        logger.warning("labels TXT okunamadı (%s): %s — kutular belirsiz", label_path, exc)
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


HAND_LABEL_TXT_SOURCE = "hand_label_txt"


# ----------------------------------------------------------------------
# P1-16 (K-17 / K-05): bozuk dosya karantinası — fail-closed koruma.
# Bozuk review JSON / labels TXT silinmez, üzerine yazılmaz;
# <dataset>/quarantine altına taşınır, kullanıcıya bildirilir.
# ----------------------------------------------------------------------
QUARANTINE_DIR_NAME = "quarantine"


def review_json_is_corrupt(path: Union[str, Path]) -> bool:
    """Review JSON bozuk mu? Yoksa False — yokluk bozukluk değildir.

    Okunamayan / JSON-çözümlenemeyen / dict olmayan payload bozuk
    sayılır: içeriği bilinmeyen dosya "korumasız" sayılmaz (K-17).
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return True
    return not isinstance(payload, dict)


def label_txt_corrupt_lines(path: Union[str, Path]) -> List[int]:
    """Labels TXT'deki bozuk satır numaraları (1-tabanlı).

    Boş satır bozuk sayılmaz. ``[-1]`` dosyanın hiç okunamadığı
    anlamına gelir; bozuk satır yoksa ``[]`` döner. Kural
    ``_legacy_weak_from_txt`` ile aynıdır (en az 5 alan, sayısal
    değerler) — tek bozuk satır diğer kutuları düşürmez (K-17).
    """
    p = Path(path)
    if not p.is_file():
        return []
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return [-1]
    bad: List[int] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            bad.append(lineno)
            continue
        try:
            int(float(parts[0]))
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
            if len(parts) >= 6:
                float(parts[5])
        except ValueError:
            bad.append(lineno)
    return bad


def quarantine_corrupt_file(
    path: Union[str, Path],
    *,
    dataset_dir: Union[str, Path],
    reason: str = "",
) -> Optional[Path]:
    """Bozuk review JSON / labels TXT'yi karantina dizinine taşı.

    Hedef: ``<dataset>/quarantine/<tür>/<split>/<stem>__<zaman>.<uzantı>``
    (tür: ``review`` / ``labels`` / ``misc``). Silmez — bozuk dosya
    kanıt olarak saklanır, kullanıcı eliyle incelenip geri yüklenebilir.
    Taşıma başarısız olursa dosya yerinde bırakılır ve None döner
    (fail-closed: silinmez, üzerine yazılmaz).
    """
    src = Path(path)
    ds = Path(dataset_dir)
    if not src.is_file():
        return None
    kind = (
        src.parent.parent.name
        if src.parent.parent.name in ("review", "labels")
        else "misc"
    )
    split = src.parent.name if src.parent.name in ("train", "val") else "unknown"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_dir = ds / QUARANTINE_DIR_NAME / kind / split
    base = f"{src.stem}__{stamp}"
    dest = dest_dir / f"{base}{src.suffix}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while dest.exists():
            n += 1
            dest = dest_dir / f"{base}__{n}{src.suffix}"
        shutil.move(str(src), str(dest))
    except Exception as exc:
        logger.error(
            "[KARANTİNA] bozuk dosya taşınamadı (%s -> %s): %s", src, dest, exc
        )
        return None
    logger.warning(
        "[KARANTİNA] bozuk dosya karantinaya alındı: %s -> %s%s",
        src,
        dest,
        f" ({reason})" if reason else "",
    )
    return dest


def _reconcile_txt_hand_labels(
    trusted: Sequence[Dict[str, Any]],
    weak: Sequence[Dict[str, Any]],
    label_path: Path,
    *,
    keep_unmatched_model_lines_as_weak: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """labels TXT'deki el etiketlerini review durumuyla uzlaştır (K-01/K-02).

    Tek kanonik TXT uzlaştırması: ``_load_state`` ve dialog taraması
    (``ui/yolo_factory_dialog.py::_scan_review_queue_items``) aynı
    fonksiyonu çağırır — iki alt sistem aynı TXT'ye zıt hüküm vermez.

    Kural: TXT'deki **5 alanlı** kutu (güven sütunu yok = el ile
    düzeltilmiş zip yüklemesi, K-02'nin "5 alanlı TXT satırı") review
    JSON'da bayraksız kalsa bile trusted el etiketi sayılır. 6 alanlı
    (güven skorlu) satır model çıktısıdır (w21, approved_pool ile aynı
    kural) — el etiketi sayılmaz, weak'te kalır/görünmez kalır.
    Kullanıcı açıkça şüpheliye ayırmışsa (kutuda ``hand_label is False``
    VEYA ``demoted_by_user is True`` varsa — P0-8 bu bayrağı tanıtıyor,
    iki yazımı da say) hariç, o karar korunur. TXT'de olmayan kutuya
    dokunulmaz. Bozuk review JSON'da çağrılmaz (çağıran
    ``_load_state`` bozuk dalda erken döner).
    """
    txt_boxes = _legacy_weak_from_txt(label_path)
    if not txt_boxes:
        return list(trusted), list(weak)
    kept_trusted: List[Dict[str, Any]] = [dict(b) for b in trusted]
    kept_weak: List[Dict[str, Any]] = [dict(b) for b in weak]
    for tb in txt_boxes:
        if _find_slot(kept_trusted, tb) is not None:
            continue
        wi = _find_slot(kept_weak, tb)
        if wi is not None:
            existing = kept_weak[wi]
            if (
                existing.get("hand_label") is False
                or existing.get("demoted_by_user") is True
            ):
                continue  # kullanıcı şüpheliye ayırmış — karar korunur
            if tb.get("has_explicit_conf"):
                continue  # 6 alanlı model satırı weak'te kalır
            # P1-12 (K-32): ölçülmüş güvenli weak kutu MODEL kutusudur —
            # 5 alanlı TXT satırı yalnızca yayım (emission) kanıtıdır,
            # el düzenleme kanıtı değil (üretim TXT'si zaten 5 alanlı
            # yazılır; el kutusu ölçülmüş skor taşımaz, S-6103). K-02
            # terfisi yalnızca bayraksız + ölçülmemiş kutuya uygulanır;
            # ölçülmüş kutu weak'te kalır — yoksa 0.30'luk taban-altı
            # zehir el etiketi sayılıp temizlikten kaçar (purge async
            # testi). Açık insan bayrağı (hand_label/human_approved)
            # her zaman terfi eder.
            if has_measured_confidence(existing) and not (
                existing.get("hand_label") is True
                or existing.get("human_approved") is True
            ):
                continue
            kept_weak.pop(wi)
        elif tb.get("has_explicit_conf"):
            # Review'da karşılığı olmayan 6 alanlı satır: model çıktısı,
            # el etiketi değil. Yalnızca review'suz fallback'te
            # (eski davranış: TXT weak görünürdü) weak'te tutulur.
            if keep_unmatched_model_lines_as_weak:
                kept_weak.append(dict(tb))
            continue
        nb = dict(tb)
        nb["hand_label"] = True
        nb["trusted"] = True
        nb["weak"] = False
        nb["source"] = HAND_LABEL_TXT_SOURCE
        nb.pop("legacy_auto", None)
        kept_trusted.append(nb)
    return kept_trusted, kept_weak


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
        except Exception:
            # The review file EXISTS but is corrupt: do not silently swap
            # its trusted list for legacy txt boxes. Return empty state
            # and flag the corruption so callers can see it.
            meta["review_corrupt"] = True
            if return_meta:
                return [], [], meta
            return [], []
        if isinstance(payload, dict):
            for k in ("human_approved", "done", "approved_at"):
                if k in payload:
                    meta[k] = payload[k]
            # P3-27: kuyruk-kapalı okuması tek kanonik yoldan
            # (``approval_state.is_frame_closed``: done=="ok" VEYA açık
            # onay). Bu, insan-onayı iddiası değildir — yalnızca bulk
            # işaret kuralının kapsayıcı okumasıdır; kapılar katı
            # ``is_frame_human_approved`` okur.
            is_approved = is_frame_closed(payload)
            trusted: List[Dict[str, Any]] = []
            for b in payload.get("trusted", []):
                if isinstance(b, dict):
                    nb = dict(b)
                    # Payload-level (frame) approval must NOT stamp per-box
                    # human approval: a frame-level "ok" says nothing about
                    # whether THIS box was inspected (sweep E-04). Record
                    # the bulk provenance instead; boxes keep whatever
                    # per-box flag they already had.
                    if is_approved and nb.get("human_approved") is not False:
                        nb["bulk_approved"] = True
                    if "approved_at" in meta and "approved_at" not in nb:
                        nb["approved_at"] = meta["approved_at"]
                    trusted.append(nb)
            weak = [dict(b) for b in payload.get("weak", []) if isinstance(b, dict)]
            # TXT uzlaştırması (K-01/K-03): el etiketleri review JSON'unda
            # bayraksız kalsa bile TXT'den trusted'a taşınır (şüpheliye
            # ayırma hariç). Bozuk review dalı yukarıda erken döndüğü için
            # buraya bozuk payload gelmez.
            trusted, weak = _reconcile_txt_hand_labels(trusted, weak, label_path)
            if return_meta:
                return trusted, weak, meta
            return trusted, weak
    # No review file at all: TXT boxes are hand labels (zip yüklemesi),
    # not legacy weak auto boxes — uzlaştırma onları trusted'a taşır.
    # 6 alanlı (güven skorlu) satırlar model çıktısıdır (w21): legacy
    # weak olarak görünür kalır.
    trusted, weak = _reconcile_txt_hand_labels(
        [], [], label_path, keep_unmatched_model_lines_as_weak=True
    )
    if return_meta:
        return trusted, weak, meta
    return trusted, weak


def _yolo_line(ann: Dict[str, Any]) -> str:
    # Missing keys must not blow up a whole txt write: fall back to 0/0.5
    # defaults. Callers filter out boxes without cx/cy/w/h first.
    return (
        f"{_safe_class_id(ann.get('class_id', -1), -1)} "
        f"{_safe_float(ann.get('cx', 0.5), 0.5):.6f} "
        f"{_safe_float(ann.get('cy', 0.5), 0.5):.6f} "
        f"{_safe_float(ann.get('w', 0.0), 0.0):.6f} "
        f"{_safe_float(ann.get('h', 0.0), 0.0):.6f}"
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
    # A stale trusted=True flag is not evidence: only an evidence-based
    # annotation (hand label / human approval) seeds. P1-12 (K-34): F/E
    # gather seed tek başına tohum değildir — K1 kapısından geçmesi için
    # açık insan onayı gerekir.
    seed = bool(has_seed) or any(is_trusted_annotation(b) for b in kept_trusted)

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
        # P1-12 (K-34): TEK insan tanımı — çıplak approved/confirmed
        # boru hattı bayrağıdır, el sayılmaz.
        hand = _is_explicitly_human_approved(new)
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
    """Write trusted boxes into train/val txt.

    Normal gather frames: only trusted (≥ floor / hand) enter the txt.
    Omni learn_* frames: weak layout boxes stay marked weak in review and
    enter the txt ONLY with an explicit per-box human approval (sweep
    E-06) — the old "also written so Adım 2 can cull" rule let unapproved
    template boxes train the model.
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
    # P2-21 (K-07): QA'nın elediği kutu sayısı — commit return'ünde tutulur.
    qa_dropped = max(0, len(raw_anns) - len(qa_cleaned))

    cleaned = filter_hud_false_positives(qa_cleaned, stem=str(stem))
    prev_trusted, prev_weak, prev_meta = _load_state(
        dataset_dir, split_norm, stem, label_path, return_meta=True
    )
    # P1-16 (K-05): bozuk review JSON okunur — commit bu karede DURUR.
    # Bozuk dosya karantinaya alınır; TXT boş yazılmaz, review'un üzerine
    # yeni payload yazılmaz, model-stale işaretlenmez. Çağıran
    # "blocked"/"review_corrupt" bayraklarıyla kareyi atlar (0 kutu
    # "boş kare" değildir — örn. multi_mix TXT silmez).
    if prev_meta.get("review_corrupt"):
        review_path = _review_path(dataset_dir, split_norm, stem)
        dest = quarantine_corrupt_file(
            review_path,
            dataset_dir=dataset_dir,
            reason="bozuk review JSON — commit durduruldu (K-05/P1-16)",
        )
        logger.error(
            "[KORUMA] commit DURDURULDU (%s/%s): review JSON bozuk; "
            "TXT yazılmadı, review üzerine yazılmadı%s",
            split_norm,
            stem,
            f" — karantina: {dest}" if dest is not None else " — karantina BAŞARISIZ, dosya yerinde duruyor",
        )
        return {
            "blocked": True,
            "review_corrupt": True,
            "quarantined": str(dest) if dest is not None else None,
            "trusted": [],
            "weak": [],
            "train_box_count": 0,
            "weak_box_count": 0,
            "has_trusted_seed": False,
            "review_path": str(review_path),
            "label_path": str(label_path),
            "floor": float(floor),
            "learn_auto_in_train": False,
        }
    trusted, weak, seed = merge_label_observation(
        prev_trusted,
        prev_weak,
        cleaned,
        has_seed=has_trusted_seed(dataset_dir),
        floor=float(floor),
        require_human_confirmation=require_human_confirmation,
    )
    # P2-21 (K-07): TXT yoluna giren trusted aday sayısı — demote
    # kapılarından ÖNCE ölçülür; dropped bu sayıdan TXT'ye
    # ulaşamayanların toplamıdır (demote edilen kutu review'da weak
    # olarak kalır, datasetten silinmez — "düşürülen" burada yalnızca
    # train TXT yolundan düşen demektir).
    trusted_in = len(trusted)
    # K1: merge never promotes a template box, but a stale trusted entry
    # (e.g. from an old review file, written before the template gate)
    # can survive in the trusted list. Demote it back to weak here so it
    # neither stays trusted in the review JSON nor reaches the txt —
    # unless a human explicitly approved that box. P1-12 (K-34/L008):
    # F/E şablon (seed) kutular da bu kapsamda demote edilir.
    _stale_templates = [
        b
        for b in trusted
        if (is_learn_auto_annotation(b) or _is_hand_or_gather_seed(b))
        and not _is_explicitly_human_approved(b)
    ]
    if _stale_templates:
        for b in _stale_templates:
            nb = dict(b)
            nb["trusted"] = False
            nb["weak"] = True
            weak.append(nb)
        trusted = [b for b in trusted if b not in _stale_templates]
    # Final hard gate: never write sub-floor trusted boxes to train txt (unless hand_label).
    # P1-12 (K-34/F016-06): çıplak approved/confirmed bu kapıyı baypas
    # edemez — yalnızca açık insan onayı (hand_label/human_approved)
    # veya taban üstü ölçülmüş güven geçirir.
    floor_f = float(floor)
    train_trusted = [
        b
        for b in trusted
        if "cx" in b
        and (
            bool(b.get("hand_label") is True)
            or bool(b.get("human_approved") is True)
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
    sanitized_weak, _ = sanitize_boxes(weak, file_path=str(label_path))
    sanitize_dropped = max(
        0, (len(trusted) + len(weak)) - (len(sanitized_trusted) + len(sanitized_weak))
    )
    trusted, weak = sanitized_trusted, sanitized_weak
    trusted_out = len(trusted)
    # P2-21 (K-07): TXT yolundan düşen kutu sayısı — QA + bayat şablon
    # demote + taban-altı demote + ikinci sanitize toplamı.
    dropped = qa_dropped + len(_stale_templates) + len(demoted) + sanitize_dropped
    emit = train_txt_boxes(sanitized_trusted, sanitized_weak, stem=str(stem), floor=floor_f)
    # TXT yazımı aşağıda tek yazıcıda (write_review_pair) yapılır —
    # emit burada yalnızca learn_auto_in_train / sayaçlar için erken
    # hesaplanır; kural tektir, iki ayrı hesap yoktur.

    # "learn_auto_in_train" must describe what actually entered the txt,
    # not what stayed weak in review (sweep E-06: weak boxes no longer
    # enter the txt without per-box human approval).
    _weak_ids = {id(b) for b in sanitized_weak}
    learn_auto_in_train = bool(
        is_learn_stem(stem) and any(id(b) in _weak_ids for b in emit)
    )

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
        "learn_auto_in_train": learn_auto_in_train,
    }
    if resolved_human_approved is not None:
        review_payload["human_approved"] = bool(resolved_human_approved)
    if resolved_done is not None:
        review_payload["done"] = str(resolved_done)
    if resolved_approved_at is not None:
        review_payload["approved_at"] = resolved_approved_at

    review = _review_path(dataset_dir, split_norm, stem)
    # P3-26: tek yazıcı — JSON+TXT aynı çağrıda, kanonik emit kuralıyla.
    # allow_empty verilmez (varsayılan False): kazara boşalan commit
    # TXT'yi reddedip eski TXT'yi korur (P0-1); ret durumu
    # txt_written=False olarak sonuçta döner.
    pair = write_review_pair(
        review_path=review,
        label_path=label_path,
        review_payload=review_payload,
        stem=str(stem),
        floor=floor_f,
        emit_boxes=emit,
    )
    txt_written = pair["txt_written"]
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
        # P2-21 (K-07): commit sayacı — TXT yoluna giren / review'da
        # trusted kalan / TXT yolundan düşen kutu + TXT yazım durumu.
        "trusted_in": trusted_in,
        "trusted_out": trusted_out,
        "dropped": dropped,
        "dropped_qa": qa_dropped,
        "dropped_stale_template": len(_stale_templates),
        "dropped_sub_floor": len(demoted),
        "dropped_sanitize": sanitize_dropped,
        # P3-26: TXT yazım durumu — boş-emit reddi (P0-1) burada görünür.
        "txt_written": bool(txt_written),
        "has_trusted_seed": seed,
        "review_path": str(review),
        "label_path": str(label_path),
        "floor": float(floor),
        "learn_auto_in_train": learn_auto_in_train,
    }
    # P2-21 (K-07): commit artık logsuz değil — her commit tek satırda
    # giren/kalan/düşen/TXT kutu sayısını bırakır.
    logger.info(
        "[COMMIT] %s/%s: trusted_in=%d trusted_out=%d dropped=%d "
        "(qa=%d bayat_sablon=%d taban_alti=%d sanitize=%d) "
        "train_txt=%d txt_yazildi=%s",
        split_norm,
        stem,
        trusted_in,
        trusted_out,
        dropped,
        qa_dropped,
        len(_stale_templates),
        len(demoted),
        sanitize_dropped,
        len(emit),
        bool(txt_written),
    )
    if resolved_human_approved is not None:
        result["human_approved"] = bool(resolved_human_approved)
    if resolved_done is not None:
        result["done"] = str(resolved_done)
    if resolved_approved_at is not None:
        result["approved_at"] = resolved_approved_at
    return result


_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _normalize_requeue_box(
    raw: Dict[str, Any],
    *,
    default_conf: float = CONF_REQUEUE_DEFAULT,
    emin_threshold: float = CONF_EMIN,
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
    conf_fabricated = False
    if not has_explicit and conf <= 1e-6:
        # No measured confidence exists: the default is a placeholder for
        # review, flagged as fabricated — never a real measurement.
        conf = float(default_conf)
        conf_fabricated = True
    name = str(raw.get("class_name") or raw.get("key") or _class_name_for_id(cid))

    raw_src = raw.get("source")
    if raw_src:
        src = str(raw_src).strip().lower()
    else:
        src = "explicit_txt_conf" if has_explicit else "dataset_requeue"

    is_omni_syn = is_omni_synthetic_source(src) or (raw.get("omni_synthetic") is True)
    # P0-8 (K-12): a user demote is a newer, explicit human decision than
    # the origin flag — it vetoes every hand-derived trust inference in
    # this function AND in calibrate_box_confidence_and_trust below, so a
    # demoted box stays weak through requeue until a human re-approves it.
    # P1-12 (K-34): TEK insan tanımı — çıplak "approved" boru hattı
    # bayrağıdır, el/insan sayılmaz.
    is_demoted = bool(raw.get("demoted_by_user") is True)
    is_exemplar = bool(raw.get("exemplar_match") is True or "exemplar" in src) and not is_demoted

    # v18/F15 (§7, kullanıcı kararı 11 Kasım 12:47 — "kural kazansın"):
    # requeue damgası güvenilirlik VERMEZ. Köken (hand_label / tekli
    # human_approved / şablon seed) tek başına Emin yapmaz — kanıtsız
    # kutu requeue'da bile trusted olmaz. Yalnızca ölçülmüş güven
    # (6. txt alanı, >= emin, omni/sezgisel kaynak hariç) veya exemplar
    # eşleşmesi trusted bırakır. Bu, P1-12 docstring'inin söylediğini
    # kodun da yapmasıdır (seed tek başına Emin değildi ama eski kod
    # has_seed'i Emin sayıyordu). Köken bayrakları (hand_label, source,
    # demoted_by_user) dict(raw) ile JSON'a aynen taşınır — yalnızca
    # trusted/weak hükmü değişir. hand/human_approved kutular zaten
    # requeue_labeled_dataset_for_review filtresinde elenir; buradaki
    # kural doğrudan çağrıyı da aynı hükme bağlar.
    if (
        has_explicit
        and conf >= float(emin_threshold)
        and not is_omni_syn
        and src not in UNVERIFIED_HEURISTIC_SOURCES
    ):
        is_emin = True
    else:
        is_emin = False
    if is_demoted:
        # Demote wins over the explicit-confidence contract too: the user
        # said "suspicious", so the box stays weak until re-approved.
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
            "confidence_fabricated": conf_fabricated,
            "has_explicit_conf": has_explicit,
            "trusted": is_emin,
            "weak": not is_emin,
            "source": src,
            "requeued": True,
        }
    )
    # Opt in to confidence-based trust ONLY for boxes whose confidence was
    # explicitly measured (6th txt token): requeue restores already-labeled
    # data, and its documented contract trusts explicit conf >= emin.
    # Anything else gets the strict rule (confidence alone never trusts).
    calibrated = calibrate_box_confidence_and_trust(
        out,
        stem=stem,
        emin_threshold=emin_threshold,
        trust_by_confidence=bool(has_explicit and conf >= float(emin_threshold)),
    )
    exemplar_restored = False
    if is_exemplar and not calibrated.get("trusted"):
        # F-0963: the trailing calibrate must not silently overwrite the
        # exemplar decision above. Restore exemplar trust — unless the
        # box was demoted for a real reason (material conflict, omni
        # synthetic, or an unverified-heuristic source).
        _unverified = (
            src in UNVERIFIED_HEURISTIC_SOURCES
            or any(u in src for u in UNVERIFIED_HEURISTIC_SOURCES)
        )
        if not is_omni_syn and not _unverified and not is_material_conflict(stem, name, cid):
            calibrated["trusted"] = True
            calibrated["weak"] = False
            exemplar_restored = True
    # v18/F15 (§7 "kural kazansın"): kalibrasyonun insan dalı
    # (hand_label/human_approved -> trusted) requeue yolunda geçersizdir
    # — yoksa yukarıda kaldırılan damga calibrate üzerinden geri gelirdi.
    # Kuyruğa düşen kutu güveni yalnızca ölçülmüş güven (>= emin) veya
    # exemplar eşleşmesiyle kazanır; köken tek başına trusted bırakmaz.
    if calibrated.get("trusted") is True:
        explicit_trust = bool(
            has_explicit
            and conf >= float(emin_threshold)
            and not is_omni_syn
            and src not in UNVERIFIED_HEURISTIC_SOURCES
        )
        if not explicit_trust and not exemplar_restored:
            calibrated["trusted"] = False
            calibrated["weak"] = True
    return calibrated


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
    default_conf: float = CONF_REQUEUE_DEFAULT,
    include_empty: bool = True,
    emin_threshold: float = CONF_EMIN,
) -> Dict[str, Any]:
    """Put existing train/val samples back into the Adım 2 review queue.

    Scans ``images/{train,val}``, builds/refills ``review/{split}/{stem}.json`` with:
    - ``trusted``: boxes with explicit confidence >= emin_threshold (CONF_EMIN)
      or exemplar_match=True. P1-12 (K-34): gather seed templates are NOT
      Emin on their own anymore — a seed box needs an explicit per-box
      human approval (or explicit conf >= emin) to be trusted; otherwise
      it stays weak for review (K1 gate).
    - ``weak``: boxes with confidence < emin_threshold (CONF_REQUEUE_DEFAULT
      placeholder for unverified .txt labels, flagged confidence_fabricated -> Şüpheli).
    Label ``.txt`` and images are left intact until user explicitly modifies them.
    Does not load image pixels (meta/json only).

    v18/F15 (§7, kullanıcı kararı 11 Kasım 12:47 — "kural kazansın"):
    - Asla kuyruğa geri dönmez: en az bir kutusu ``hand_label`` veya TEKLİ
      ``human_approved`` taşıyan kare (``approval_state.box_is_never_requeue``)
      requeue'ya GİRMEZ — review JSON'unun üzerine yazılmaz, eski hali
      aynen kalır, ``skipped_protected`` sayılır. Toplu onay (``bulk_approved``)
      bu korumaya girmez; toplu onaylı kare requeue'ya düşebilir.
      Demote edilmiş kutu da girer (demote daha yeni bir insan kararıdır).
    - Kuyruğa düşen kutuda requeue hiçbir trusted damgası vurmaz: kanıtsız
      kutu trusted olmaz; güven yalnızca ölçülmüş güven (>= emin) veya
      exemplar eşleşmesiyle kazanılır (``_normalize_requeue_box``).
    - K-14 damgası (TXT-only kareye ``hand_label``) KORUNUR: TXT-only kutu
      tanım gereği el etiketidir, bu yüzden TXT-only kare de bu filtreye
      takılır ve kuyruğa dönmez. Sonuç: salt-TXT veri setinde "Mevcut seti
      onaya geri al" hiç kare yazmaz — bu, kararın doğrudan sonucudur;
      TXT-only karenin yeniden incelenmesi istenirse K-14 tanı kuralının
      kendisi ayrı kararla gözden geçirilmeli.
    """
    root = Path(dataset_dir)
    written = 0
    refreshed = 0
    empty_n = 0
    box_n = 0
    trusted_n = 0
    weak_n = 0
    skipped = 0
    skipped_protected = 0
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
            had_review_json = review_path.is_file()
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
            if not had_review_json:
                # P0-8 (K-14): TXT-only kare = düzeltilmiş el etiketi
                # (kuyruk yükleyicisiyle aynı kural). TXT'nin kendisi
                # köken taşımaz — damga burada atılmazsa requeue review
                # JSON yazar, kare TXT-only olmaktan çıkar ve el etiketi
                # koruması ölür. v18/F15: damgalı kutu artık hand→trusted
                # olmaz; damga aşağıdaki asla-kuyruğa-dönmez filtresini
                # tetikler — TXT-only kare kuyruğa hiç girmez, eski
                # review JSON'u / TXT aynen kalır.
                for lb in legacy:
                    lb["hand_label"] = True
                    lb["source"] = HAND_LABEL_TXT_SOURCE
            if is_already_approved:
                # Re-review means re-earning approval: do NOT carry the
                # old per-box human_approved into the new review round
                # (sweep E-04). The previous approval stays visible as
                # provenance only, never as trust.
                for b in trusted:
                    b["previously_approved"] = True
                for b in legacy:
                    b["previously_approved"] = True
            # Trusted first: _dedupe_boxes keeps the first copy, so a
            # weak duplicate of the same geometry must not shadow the
            # trusted one.
            merged_src = list(trusted) + list(weak_existing) + list(legacy)
            # v18/F15 (§7 "kural kazansın" + asla-kuyruğa-dönmez):
            # el etiketi / tekli insan onaylı kutu içeren kare requeue'ya
            # girmez. KARE düzeyi atlama — kutu düzeyi eleme yapılmaz:
            # korumalı kutuyu yeni JSON'dan düşürmek, kare onaylanınca
            # TXT emitinden de düşürür (sessiz el-etiketi kaybı). Atlanan
            # karenin review JSON'u / TXT'si aynen kalır; hiçbir şeyin
            # üzerine yazılmaz (.bak bile alınmaz), yalnızca sayılır.
            # Toplu onay (bulk_approved) ve demote edilmiş kutu bu
            # filtreye takılmaz — onlar requeue'ya düşer (bkz.
            # approval_state.box_is_never_requeue).
            if any(box_is_never_requeue(b) for b in merged_src):
                skipped += 1
                skipped_protected += 1
                continue
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
            # Back in the review queue the frame is pending again: the old
            # approval must not re-admit it to the pool (prepare_... only
            # counts human_approved) before a human looks at it a second
            # time (sweep E-04).
            payload_out["human_approved"] = False
            payload_out["done"] = None
            if "approved_at" in existing_meta and existing_meta["approved_at"] is not None:
                payload_out["approved_at"] = existing_meta["approved_at"]
            try:
                # P0-2 (K-08): review JSON üzerine yazılmadan önce .bak.
                _backup_before_overwrite(review_path)
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
        "skipped_protected": skipped_protected,
        "default_conf": conf_f,
        "emin_threshold": emin_th,
        "include_empty": bool(include_empty),
    }


