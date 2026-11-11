# -*- coding: utf-8 -*-
"""Shared multi-signal quality gate for cartography save and wrong-node cleanup.

A candidate is rejected / deleted when any hard signal fires:

- parchment scroll patch
- monster-dot circularity on a minimap-sized crop
- gatherion hard-wrong class (F/E/monster/UI/…) at calibrated confidence
- resource_type vs model contradiction (gatherable label + hard-wrong class)
- feather icon (cartography path)

Class 9 (minimap sprout) alone is NOT a hard reject: minimap_auto patches are
sprout icons by design. Class 9 plus monster geometry still rejects.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.vision_filters import (
    FeatherNegativeFilter,
    MonsterCircularityFilter,
    is_parchment_scroll_patch,
)

# P1-1: keep / hard-wrong / label sets are derived from the single
# taxonomy source (dataset_factory.class_taxonomy, authority
# dataset/data.yaml) — no second copy of the class table lives here.
from dataset_factory.class_taxonomy import (
    CANONICAL_NAME_TO_ID,
    CANONICAL_SHORT_LABELS_TR,
    MATERIAL_3D_CLASSES,
    is_valid_canonical_class_id,
)

try:
    from dataset_factory.exemplar_memory import (
        ExemplarMemory,
        compute_similarity,
        extract_crop_features,
        get_exemplar_memory,
    )
except Exception:
    ExemplarMemory = None  # type: ignore
    compute_similarity = None  # type: ignore
    extract_crop_features = None  # type: ignore
    get_exemplar_memory = None  # type: ignore

logger = logging.getLogger(__name__)

CONF_TIERS = {
    "aggressive": 0.10,
    "balanced": 0.13,
    "strict": 0.22,
}
DEFAULT_TIER = "balanced"
MODEL_CONF_MIN = CONF_TIERS[DEFAULT_TIER]
SAME_POINT_PX = 18.0

# R14a — delete-gate bar for learned-negative condemnation.
# A learned negative exemplar may only DELETE a node when the match is a
# near-duplicate (the same patch seen again). The suite's own calibration
# (tests/test_w49_wrong_node_negative_learning.py) asserts near-duplicate
# variants (flip / 2px shift / +8 brightness / shifted crop / 95% rescale)
# still score >= 0.95, while a merely similar gray patch scores 0.89-0.92
# (measured: synthetic ore patch vs the committed class-15
# "wrong_node_purge" negative in dataset/learned_exemplars.json, sim
# 0.8937-0.9214). At the old 0.82 suppress bar that resemblance deleted
# valid ore nodes even on below-floor model verdicts (test_wrong_node_scan
# kept_low / kept_strict / 36px). The 0.82 bar is untouched for the
# ExemplarMemory API itself (reporting/suppression elsewhere); only the
# delete gate demands the near-duplicate bar. The bar is passed INTO
# is_negative_patch (never post-filtered on the returned sim) so memories
# that report their own verdict keep working (mock in
# tests/test_auto_cartographer_e1.py returns True at sim 0.85).
GATE_NEGATIVE_SIM_MIN = 0.95

# Materials / essences that must never be deleted by model score alone.
# P1-1: derived from the single taxonomy source — KEEP is the 3D
# material set minus {7, 8}. The 7/8 exclusion is a LEGACY-scheme
# holdover (in the old scheme 7/8 were UI icons, not materials);
# canonically 7 is herb_lumina and 8 is ore_obsidian, i.e. gatherable.
# Membership is UNCHANGED here: whether 7/8 become keepers is a
# contract decision (items 23/24/26/29/31, Selçuk) — this only makes
# the exclusion explicit instead of a silent hardcoded set.
KEEP_CLASS_IDS = frozenset(MATERIAL_3D_CLASSES)
# Clear false positives. Class 9 is listed for labels/histograms but needs CV backup.
# F12 (v18/P1-1 regresyonu): v18 bu kümeyi "kanonik olup keeper ya da
# filiz olmayan her sınıf" diye türetip 22..26'yı (hp_potion, mp_potion,
# buff_scroll, kinah, player_character) da hard-wrong yaptı. v17 kümesi
# geri yüklendi.
# v19/F12 — 22..26 LİMBO (bilinçli, kararsız değil eksik değil):
# 22..26 ne KEEP_CLASS_IDS'te ne HARD_WRONG_CLASS_IDS'te; bu yüzden
# wrong_class_id(22..26, ...) None döner ve düğüm bu kanaldan
# silinmez (yalnızca tekrar/terrain/negatif-örnek kanalları eler).
# DİKKAT: v17 testindeki "Essences stay keep" yorumu mekanizmayı
# yanlış anlatır — 22 keep DEĞİLDİR (KEEP = {0,1,2,3,4,5,6,12}),
# None dönmesinin nedeni hiçbir kümede olmamasıdır. 22..26'yı
# hard-wrong yapmak (§7-A'nın 28 çatışmasında YOK — no 26 yalnızca
# super-category eşlemesidir) ya da keep'e almak Selçuk kararıdır;
# taksonomi türetmesiyle sessizce genişletilemez/daraltılamaz.
# Mevcut sözleşme test_wrong_node_scan.py::test_threshold_and_unknown_
# classes'ta kilitlidir (22..26'nın tamamı -> None).
HARD_WRONG_CLASS_IDS = frozenset({10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})
# v20/Selçuk 11.11: 25 kinah, 26 player_character LIMBO — silinmez;
# 26 için referans eşik 0.13 (ileride limbo→aksiyon kararına hazırlık).
LIMBO_CLASS_CONF_MIN = {"balanced": {25: MODEL_CONF_MIN, 26: 0.13}}
SPROUT_CLASS_ID = CANONICAL_NAME_TO_ID["minimap_sprout"]
WRONG_CLASS_IDS = frozenset(HARD_WRONG_CLASS_IDS | {SPROUT_CLASS_ID})

# F12 (v18/P1-1 regresyonu): v18 bu tabloyu kanonik kısa etiketlerle
# değiştirdi (7 "Lumina", 8 "Obsidyen", 12 "Kan Kristali" ...), oysa
# yanlış-düğüm raporu sözleşmesi eski görünen adları bekler
# (test_wrong_node_scan.py::test_status_lists_top_wrong_classes:
# sınıf 7 -> "F Gather"). 0..21 için v17 tablosu geri yüklendi; v17'de
# karşılığı olmayan 22..26 kanonik etiketleriyle korunur.
CLASS_LABELS_TR = {
    0: "Aria",
    1: "Mela",
    2: "Targena",
    3: "Orichalcum",
    4: "Gem",
    5: "Ağaç",
    6: "3B Amblem",
    7: "F Gather",
    8: "E",
    9: "Minimap Filiz",
    10: "Canavar",
    11: "Minimap Canavar",
    12: "dünya malzemesi",
    13: "skill/UI",
    14: "engel",
    15: "yol",
    16: "can barı",
    17: "mana barı",
    18: "can iksiri",
    19: "mana iksiri",
    20: "hedef canı",
    21: "skill yuvası",
    22: CANONICAL_SHORT_LABELS_TR[22],
    23: CANONICAL_SHORT_LABELS_TR[23],
    24: CANONICAL_SHORT_LABELS_TR[24],
    25: CANONICAL_SHORT_LABELS_TR[25],
    26: CANONICAL_SHORT_LABELS_TR[26],
}

_GATHERABLE_TOKENS = (
    "gatherable",
    "material",
    "aria",
    "mela",
    "targena",
    "orichalcum",
    "gem",
    "odyle",
    "sprout",
    "herb",
    "ore",
    "wood",
    "essence",
    "filiz",
    "bitki",
)

_monster_filter: Optional[MonsterCircularityFilter] = None
_feather_filter: Optional[FeatherNegativeFilter] = None
# F-1863: global filter singletons / prototype cache are created under locks.
_FILTER_LOCK = threading.Lock()
_PROTOTYPE_LOCK = threading.Lock()


def conf_min_for_tier(tier: str = DEFAULT_TIER) -> float:
    """Confidence floor for the named scan aggressiveness tier."""
    key = str(tier or DEFAULT_TIER).strip().lower()
    return float(CONF_TIERS.get(key, CONF_TIERS[DEFAULT_TIER]))


def is_keep_class(class_id: int, num_classes: Optional[int] = None) -> bool:
    if num_classes is not None and not (0 <= int(class_id) < int(num_classes)):
        return False
    return int(class_id) in KEEP_CLASS_IDS


def is_hard_wrong_class(class_id: int, num_classes: Optional[int] = None) -> bool:
    if num_classes is not None and not (0 <= int(class_id) < int(num_classes)):
        return False
    return int(class_id) in HARD_WRONG_CLASS_IDS


def wrong_class_id(
    class_id: Optional[int],
    confidence: Optional[float],
    num_classes: Optional[int],
    conf_min: float = MODEL_CONF_MIN,
) -> Optional[int]:
    """Hard-wrong class id when confidence clears the tier floor; else None."""
    if class_id is None or confidence is None:
        return None
    try:
        cid = int(class_id)
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(conf) or conf < float(conf_min):
        return None
    if num_classes is not None and not (0 <= cid < int(num_classes)):
        return None
    if is_keep_class(cid, num_classes):
        return None
    if not is_hard_wrong_class(cid, num_classes):
        return None
    return cid


def _get_monster_filter() -> MonsterCircularityFilter:
    global _monster_filter
    if _monster_filter is None:
        with _FILTER_LOCK:
            if _monster_filter is None:
                _monster_filter = MonsterCircularityFilter()
    return _monster_filter


def _get_feather_filter() -> FeatherNegativeFilter:
    global _feather_filter
    if _feather_filter is None:
        with _FILTER_LOCK:
            if _feather_filter is None:
                _feather_filter = FeatherNegativeFilter()
    return _feather_filter


def looks_gatherable_resource(resource_type: str) -> bool:
    text = str(resource_type or "").strip().lower()
    if not text:
        return False
    return any(tok in text for tok in _GATHERABLE_TOKENS)


def patch_is_monster_dot(image: Optional[np.ndarray]) -> bool:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False
    try:
        return bool(_get_monster_filter().is_monster_dot(image))
    except Exception as exc:
        logger.debug("monster circularity skipped: %s", exc)
        return False


def patch_is_feather(image: Optional[np.ndarray]) -> bool:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False
    try:
        return bool(_get_feather_filter().is_feather_icon(image))
    except Exception as exc:
        logger.debug("feather filter skipped: %s", exc)
        return False


def get_default_exemplar_memory() -> Any:
    """Return default singleton ExemplarMemory instance if available."""
    if get_exemplar_memory is not None:
        try:
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("Failed to get default exemplar memory: %s", exc)
    return None


_CANONICAL_TERRAIN_PROTOTYPES: Optional[List[Dict[str, Any]]] = None


def get_canonical_terrain_prototypes() -> List[Dict[str, Any]]:
    """Standardized visual feature prototypes for empty terrain and barren rock."""
    global _CANONICAL_TERRAIN_PROTOTYPES
    if _CANONICAL_TERRAIN_PROTOTYPES is not None:
        return _CANONICAL_TERRAIN_PROTOTYPES
    with _PROTOTYPE_LOCK:
        if _CANONICAL_TERRAIN_PROTOTYPES is not None:
            return _CANONICAL_TERRAIN_PROTOTYPES
        protos = _build_canonical_terrain_prototypes()
        _CANONICAL_TERRAIN_PROTOTYPES = protos
        return protos


def _build_canonical_terrain_prototypes() -> List[Dict[str, Any]]:
    import cv2
    protos: List[Dict[str, Any]] = []
    if extract_crop_features is None:
        return protos

    # 1. Flat gray ground (uniform terrain without harvest features)
    rng = np.random.default_rng(4)
    small1 = 168.0 + rng.normal(0.0, 6.0, (8, 8))
    g1 = np.clip(cv2.resize(small1.astype(np.float32), (80, 80), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    feat1 = extract_crop_features(g1)
    if feat1:
        feat1["class_name"] = "path_ground"
        feat1["class_id"] = 15
        protos.append(feat1)

    # 2. Barren gray stone / rock surface (low saturation, mid gray)
    gray_stone = np.full((80, 80, 3), 128, dtype=np.uint8)
    noise = rng.normal(0.0, 10.0, (80, 80, 3)).astype(np.float32)
    gray_stone = np.clip(gray_stone.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    feat2 = extract_crop_features(gray_stone)
    if feat2:
        feat2["class_name"] = "gray_rock"
        feat2["class_id"] = 15
        protos.append(feat2)

    # 3. Dark barren ground / rock
    dark_stone = np.full((80, 80, 3), 90, dtype=np.uint8)
    noise2 = rng.normal(0.0, 8.0, (80, 80, 3)).astype(np.float32)
    dark_stone = np.clip(dark_stone.astype(np.float32) + noise2, 0, 255).astype(np.uint8)
    feat3 = extract_crop_features(dark_stone)
    if feat3:
        feat3["class_name"] = "empty_terrain"
        feat3["class_id"] = 14
        protos.append(feat3)

    return protos


def is_fake_gray_rock_or_ground(
    image: Optional[np.ndarray],
    exemplar_memory: Any = None,
    candidate_class_id: Optional[int] = None,
) -> Tuple[bool, float, str]:
    """Test if a node thumbnail is an erroneously added barren gray rock or empty ground.

    Checks against:
    0. Confirmed positive exemplars in ExemplarMemory (Immunity Shield)
    1. Learned negative exemplars in ExemplarMemory (dataset/learned_exemplars.json)
    2. Empty terrain prototypes in ExemplarMemory
    3. Canonical empty terrain prototypes (flat ground, gray rock)
    4. Direct visual desaturation + flat texture analysis (barren stone/dirt)

    Returns: (is_fake: bool, confidence: float, reason_detail: str)
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return False, 0.0, ""

    import cv2
    mem = exemplar_memory if exemplar_memory is not None else get_default_exemplar_memory()

    # 0. Immunity Shield: If candidate matches a confirmed positive harvestable resource, it's not fake rock/ground
    if mem is not None and hasattr(mem, "match_positive"):
        try:
            pos_sim, pos_cid, pos_cname, _ = mem.match_positive(image)
            if pos_cid is not None and pos_sim >= 0.72 and is_keep_class(pos_cid):
                return False, 0.0, ""
        except Exception as exc:
            logger.debug("Positive exemplar shield check error: %s", exc)

    # 1. Check against ExemplarMemory negative exemplars
    # R14a: delete-grade evidence only — a merely similar patch (sim
    # 0.89-0.92) must not condemn a node; see GATE_NEGATIVE_SIM_MIN.
    if mem is not None and hasattr(mem, "match_negative"):
        try:
            neg_sim, neg_ex = mem.match_negative(image, class_id=candidate_class_id)
            thresh = GATE_NEGATIVE_SIM_MIN
            if neg_sim >= thresh and neg_ex is not None:
                cname = str(neg_ex.get("class_name", "negative_exemplar"))
                cid = neg_ex.get("class_id")
                if cid is None or not is_keep_class(cid):
                    return True, float(neg_sim), f"learned_negative:{cname}:{cid}"
        except Exception as exc:
            logger.debug("Negative exemplar match error: %s", exc)

    # 2. Check against empty terrain prototypes in ExemplarMemory
    if extract_crop_features is not None and compute_similarity is not None:
        feats = extract_crop_features(image)
        if feats is not None:
            # Check custom prototypes on exemplar_memory if present
            mem_protos = []
            if mem is not None:
                mem_protos = list(getattr(mem, "empty_terrain_prototypes", []) or [])
                if not mem_protos:
                    mem_protos = list(getattr(mem, "terrain_prototypes", []) or [])
                if not mem_protos and hasattr(mem, "_negative"):
                    for e in getattr(mem, "_negative", []):
                        cn = str(e.get("class_name", "")).lower()
                        cid = int(e.get("class_id", -1))
                        # v19/F12 (P1-8 class-15 kovası): a "wrong_node_*"
                        # negative is UNKNOWN content stored under the
                        # class-15 bucket — it is not ground evidence, so
                        # it must not become a terrain prototype. Only a
                        # record whose NAME claims ground (canonical
                        # "path_ground" or a terrain/ground/rock token)
                        # counts as a terrain prototype; cid 15 alone no
                        # longer suffices. (cid 14 obstacle keeps the old
                        # rule; genuine (15, "path_ground") records are
                        # unaffected — their name claims ground.)
                        _name_claims_ground = any(
                            t in cn
                            for t in ("terrain", "ground", "rock", "zemin", "kaya")
                        ) or cn == "path_ground"
                        if cid == 14 or _name_claims_ground:
                            mem_protos.append(e)

            for proto in mem_protos:
                sim = compute_similarity(feats, proto)
                # R14a: same near-duplicate bar as step 1 — a merely similar
                # learned-terrain patch must not condemn the node.
                if sim >= GATE_NEGATIVE_SIM_MIN:
                    cname = str(proto.get("class_name", "empty_terrain"))
                    return True, float(sim), f"exemplar_terrain:{cname}"

            # Check canonical empty terrain prototypes. F-1846: these
            # prototypes are synthetic noise, so a plain gray square already
            # scores ~0.82 against them. Require a much stricter 0.90 match
            # before synthetic prototypes alone condemn a patch as fake.
            for proto in get_canonical_terrain_prototypes():
                sim = compute_similarity(feats, proto)
                if sim >= 0.90:
                    cname = str(proto.get("class_name", "empty_terrain"))
                    return True, float(sim), f"canonical_terrain:{cname}"

    # 3. Direct visual check for desaturated flat gray rock / dirt ground
    try:
        if image.ndim == 2:
            gray = image
            sat_mean = 0.0
            sat_std = 0.0
        elif image.ndim == 3 and image.shape[2] >= 3:
            bgr = image[:, :, :3]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            sat_mean = float(np.mean(sat))
            sat_std = float(np.std(sat))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            return False, 0.0, ""

        val_std = float(np.std(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # Barren flat ground: almost zero color saturation, flat surface without sharp crest / ore facets
        if sat_mean < 18.0 and sat_std < 10.0 and val_std < 15.0 and lap_var < 50.0:
            return True, 0.85, "flat_gray_ground"
    except Exception as exc:
        logger.debug("Visual terrain check error: %s", exc)

    return False, 0.0, ""


@dataclass
class GateVerdict:
    """Result of evaluating one node thumbnail / candidate."""

    accept: bool
    reasons: List[str] = field(default_factory=list)
    class_id: Optional[int] = None
    confidence: Optional[float] = None
    raw_class_id: Optional[int] = None
    # F-1859: __post_init__ used to overwrite any caller-provided is_wrong.
    # None means "derive from accept"; an explicit True/False is respected.
    is_wrong: Optional[bool] = None
    verified_resource_type: Optional[str] = None
    exemplar_matched: bool = False
    exemplar_sim: Optional[float] = None

    def __post_init__(self) -> None:
        if self.is_wrong is None:
            self.is_wrong = not self.accept

    @property
    def reject(self) -> bool:
        return not self.accept


def evaluate_node_candidate(
    image: Optional[np.ndarray],
    *,
    source_type: str = "",
    resource_type: str = "",
    detector: Any = None,
    num_classes: Optional[int] = None,
    conf_min: float = MODEL_CONF_MIN,
    check_feather: bool = True,
    check_monster: bool = True,
    model_verdict: Optional[Tuple[int, float]] = None,
    exemplar_memory: Any = None,
    check_exemplars: bool = True,
    check_terrain: bool = True,
) -> GateVerdict:
    """Multi-signal accept/reject for one 80x80 (or similar) node patch.

    ``model_verdict`` may be precomputed ``(class_id, confidence)`` to avoid a
    second inference. When omitted and ``detector`` is set, runs ``top_class``.

    P1-3 (K-07): görüntüsüz aday reddedilir (``no_image``) — patch yoksa
    canavar/zemin/örnek sinyallerinin hiçbiri çalışamaz; ``model_verdict``
    varsa sınıf/güven kaydı verdict'te korunur ama kabul verilmez.
    """
    reasons: List[str] = []
    raw_id: Optional[int] = None
    confidence: Optional[float] = None
    hard_wrong: Optional[int] = None
    source = str(source_type or "").strip().lower()
    resource = str(resource_type or "")
    is_minimap_source = source.startswith("minimap")

    # F-1838: a harvest source used to switch off ALL THREE CV signals
    # (monster, terrain, exemplar/negative memory) at once, so a known-bad
    # patch harvested again sailed through the gate. The signals stay on:
    # feather is still minimap-only (enforced below), monster geometry and
    # terrain/exemplar checks run for harvest candidates exactly like any
    # other source. Only the minimap-source terrain exemption remains,
    # because 2D map tiles are not 3D terrain.
    # Minimap is a 2D top-down map tile; 3D terrain/rock checks must never run on 2D minimap tiles.
    if is_minimap_source:
        check_terrain = False

    # Feather icons only exist on the minimap HUD, never in 3D world or radar.
    if not is_minimap_source:
        check_feather = False

    # P1-3 (K-07): görüntüsüz aday reddedilir — patch yoksa canavar/
    # zemin/örnek sinyallerinin hiçbiri çalışamaz, `accept =
    # len(reasons) == 0` boş kanıtla True verirdi (fail-open).
    # model_verdict varsa bile patch olmadan doğrulanamaz: sınıf/güven
    # kaydı aşağıda verdict'te korunur ama kabul verilmez.
    if image is None:
        reasons.append("no_image")

    if image is not None:
        try:
            if is_parchment_scroll_patch(image):
                reasons.append("scroll")
        except Exception as exc:
            # F-1851: a malformed patch must not crash the whole gate.
            logger.debug("parchment scroll check skipped: %s", exc)

    if image is not None and check_monster and patch_is_monster_dot(image):
        reasons.append("monster")

    if image is not None and check_feather and patch_is_feather(image):
        reasons.append("feather")

    verdict = model_verdict
    if verdict is None and detector is not None and image is not None:
        try:
            if hasattr(detector, "top_class"):
                verdict = detector.top_class(image)
            elif callable(detector):
                verdict = detector(image)
        except Exception as exc:
            logger.debug("gate model inference skipped: %s", exc)
            verdict = None

    if isinstance(verdict, dict):
        try:
            raw_id = int(verdict.get("class_id", verdict.get("class")))
            confidence = float(verdict.get("confidence", verdict.get("conf")))
        except (TypeError, ValueError):
            raw_id, confidence = None, None
    elif isinstance(verdict, (tuple, list)) and len(verdict) >= 2:
        try:
            raw_id, confidence = int(verdict[0]), float(verdict[1])
        except (TypeError, ValueError):
            raw_id, confidence = None, None

    nc = num_classes
    if nc is None and detector is not None:
        try:
            nc = int(getattr(detector, "num_classes"))
        except (TypeError, ValueError):
            nc = None

    if raw_id is not None and confidence is not None:
        hard_wrong = wrong_class_id(raw_id, confidence, nc, conf_min=conf_min)
        if hard_wrong is not None:
            reasons.append("model")
        # Sprout class on a non-minimap source with monster geometry already flagged;
        # also flag sprout+monster even if circularity used a different path.
        elif (
            int(raw_id) == SPROUT_CLASS_ID
            and float(confidence) >= float(conf_min)
            and "monster" in reasons
        ):
            reasons.append("model")
            hard_wrong = SPROUT_CLASS_ID

        if (
            hard_wrong is None
            and looks_gatherable_resource(resource)
            and is_hard_wrong_class(int(raw_id), nc)
            and float(confidence) >= float(conf_min)
        ):
            reasons.append("contradiction")
            hard_wrong = int(raw_id)

    # Minimap saves: require sprout-like geometry when CV is available and model is silent.
    if (
        source.startswith("minimap")
        and image is not None
        and "monster" not in reasons
        and "feather" not in reasons
        and hard_wrong is None
    ):
        try:
            if not _get_monster_filter().is_sprout_geometry(image) and patch_is_monster_dot(image):
                reasons.append("monster")
        except Exception as exc:
            # F-1862: was a silent `except Exception: pass`.
            logger.debug("minimap sprout geometry check skipped: %s", exc)

    # Exemplar memory & terrain check: learned negative exemplars and empty terrain prototypes
    # F-1838: harvest candidates are no longer exempt from this block either.
    # S-4401: minimap candidates are no longer exempt from the learned-negative
    # exemplar check either — a known-bad patch must not sail through just
    # because it came from the minimap. Terrain stays off for minimap via
    # check_terrain=False (set above), so only the negative-exemplar branch
    # runs for minimap sources.
    if image is not None and check_exemplars:
        mem = exemplar_memory if exemplar_memory is not None else get_default_exemplar_memory()
        is_high_keep = (
            raw_id is not None
            and is_keep_class(raw_id, nc)
            and confidence is not None
            and confidence >= 0.85
        )

        if mem is not None and hasattr(mem, "is_negative_patch"):
            try:
                # R14a: pass the near-duplicate bar INTO the memory call
                # (see GATE_NEGATIVE_SIM_MIN) — never post-filter the sim.
                is_neg, neg_sim, neg_ex = mem.is_negative_patch(
                    image, threshold=GATE_NEGATIVE_SIM_MIN
                )
                if is_neg:
                    neg_cid = int(neg_ex.get("class_id", 10)) if neg_ex else 10
                    if not is_high_keep or neg_cid == raw_id or (exemplar_memory is not None and neg_sim >= 0.95):
                        if "negative_exemplar" not in reasons:
                            reasons.append("negative_exemplar")
                        if hard_wrong is None and is_valid_canonical_class_id(neg_cid):
                            # P1-1: only a canonical class may become the
                            # verdict class_id — an internal slot-state
                            # negative (e.g. -100 "empty" from the skill
                            # bar) still suppresses via the reason above,
                            # but must never surface as a class id.
                            hard_wrong = neg_cid
                        if confidence is None or neg_sim > confidence:
                            confidence = float(neg_sim)
            except Exception as exc_neg:
                logger.debug("is_negative_patch check error: %s", exc_neg)

        if check_terrain and not is_high_keep:
            is_fake, fake_conf, fake_reason = is_fake_gray_rock_or_ground(image, exemplar_memory=mem, candidate_class_id=raw_id)
            if is_fake:
                if "terrain" not in reasons:
                    reasons.append("terrain")
                if fake_reason.startswith("learned_negative") and "negative_exemplar" not in reasons:
                    reasons.append("negative_exemplar")
                if hard_wrong is None:
                    hard_wrong = 15  # path_ground
                if confidence is None or fake_conf > confidence:
                    confidence = float(fake_conf)

    accept = len(reasons) == 0
    verified_resource_type: Optional[str] = None
    exemplar_matched: bool = False
    exemplar_sim: Optional[float] = None

    if accept and image is not None and check_exemplars:
        mem = exemplar_memory if exemplar_memory is not None else get_default_exemplar_memory()
        if mem is not None and hasattr(mem, "classify_patch"):
            try:
                pos_cid, pos_cname, pos_sim = mem.classify_patch(image, min_sim=0.70)
                if pos_cid is not None or pos_cname:
                    # P1-1: 6 is herb_edana (canonical), not a crest —
                    # the old "3D Crest" here was the legacy-scheme
                    # meaning of id 6. A real crest exemplar is cid 13
                    # and still maps via the "crest" name branch below.
                    CLASS_ID_TO_MAP_CATEGORY = {
                        0: "Aria",
                        1: "Mela",
                        2: "Targena",
                        3: "Orichalcum",
                        4: "Gem",
                        5: "Tree",
                        6: "Edana",
                        9: "Sprout",
                        12: "World Resource",
                    }
                    if pos_cid is not None and pos_cid in CLASS_ID_TO_MAP_CATEGORY:
                        verified_resource_type = CLASS_ID_TO_MAP_CATEGORY[pos_cid]
                    elif pos_cname:
                        cn = str(pos_cname).lower()
                        if "aria" in cn:
                            verified_resource_type = "Aria"
                        elif "mela" in cn:
                            verified_resource_type = "Mela"
                        elif "targena" in cn:
                            verified_resource_type = "Targena"
                        elif "orichalcum" in cn:
                            verified_resource_type = "Orichalcum"
                        elif any(g in cn for g in ("gem", "crystal", "sapphire", "ruby", "diamond")):
                            verified_resource_type = "Gem"
                        elif any(t in cn for t in ("tree", "log", "wood", "yggdrasil")):
                            verified_resource_type = "Tree"
                        elif "crest" in cn:
                            verified_resource_type = "3D Crest"
                        elif "sprout" in cn:
                            verified_resource_type = "Sprout"
                        else:
                            verified_resource_type = None
                    else:
                        verified_resource_type = "Resource"

                    exemplar_matched = True
                    exemplar_sim = float(pos_sim)
                    # F-1845: never invent a 0.70 base confidence when the
                    # model gave none — the exemplar similarity IS the
                    # confidence then. Only boost when a real model
                    # confidence exists to boost from.
                    if confidence is None:
                        confidence = round(float(np.clip(float(pos_sim), 0.0, 0.99)), 4)
                    else:
                        old_conf = float(confidence)
                        boost = (1.0 - old_conf) * (float(pos_sim) - 0.50) * 0.70
                        confidence = round(float(np.clip(max(old_conf + boost, float(pos_sim)), old_conf, 0.99)), 4)
                    if raw_id is None and pos_cid is not None:
                        raw_id = pos_cid
            except Exception as exc_pos:
                logger.debug("classify_patch check error in gate: %s", exc_pos)

    return GateVerdict(
        accept=accept,
        reasons=list(dict.fromkeys(reasons)),
        class_id=hard_wrong,
        confidence=confidence,
        raw_class_id=raw_id,
        is_wrong=not accept,
        verified_resource_type=verified_resource_type,
        exemplar_matched=exemplar_matched,
        exemplar_sim=exemplar_sim,
    )


def reason_label_tr(reason: str, class_id: Optional[int] = None) -> str:
    if reason == "model" and class_id is not None:
        return CLASS_LABELS_TR.get(int(class_id), str(class_id))
    mapping = {
        "scroll": "parşömen",
        "duplicate": "kopya",
        "monster": "canavar noktası",
        "feather": "tüy",
        "contradiction": "sınıf çelişkisi",
        # F-1857: "structure" was mapped here but is never produced as a
        # reason anywhere, so it is dropped instead of implying coverage.
        "model": "model",
        "negative_exemplar": "öğrenilmiş negatif",
        "terrain": "boş zemin/kaya",
        "empty_terrain": "boş zemin",
        "no_image": "görüntü yok",
    }
    return mapping.get(reason, reason or "diğer")


def _get_candidate_attr(cand: Any, attr: str, default: Any = None) -> Any:
    if isinstance(cand, dict):
        return cand.get(attr, default)
    return getattr(cand, attr, default)


def _set_candidate_attr(cand: Any, attr: str, value: Any) -> None:
    if isinstance(cand, dict):
        cand[attr] = value
    else:
        try:
            setattr(cand, attr, value)
        except Exception:
            pass


def _get_candidate_coords(cand: Any) -> Tuple[float, float]:
    for x_key, y_key in (("map_x", "map_y"), ("x", "y"), ("screen_x", "screen_y")):
        x = _get_candidate_attr(cand, x_key, None)
        y = _get_candidate_attr(cand, y_key, None)
        if x is not None and y is not None:
            try:
                return float(x), float(y)
            except (TypeError, ValueError):
                continue
    return 0.0, 0.0


def _get_candidate_image(cand: Any) -> Optional[np.ndarray]:
    import os
    import cv2
    for key in ("patch_img", "patch", "image", "img", "thumbnail", "crop"):
        img = _get_candidate_attr(cand, key, None)
        if img is not None and isinstance(img, np.ndarray) and img.size > 0:
            return img
    patch_path = _get_candidate_attr(cand, "patch_path", None) or _get_candidate_attr(cand, "patch_file", None)
    if patch_path and isinstance(patch_path, str) and os.path.isfile(patch_path):
        try:
            raw = np.fromfile(patch_path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                return img
        except Exception:
            pass
    return None


def _safe_conf(value: Any, default: float = 0.0) -> float:
    """float() for candidate confidence that never raises (F-1852)."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_count(value: Any, default: int = 1) -> int:
    """int() for candidate counters that never raises (F-1852)."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class _SpatialGrid:
    """Uniform-grid index for fixed-radius neighbour lookups.

    Cell size equals the query radius, so any two points within the radius
    sit in the same cell or an adjacent one: checking the ``span =
    ceil(radius / cell)`` neighbourhood around the query cell is exact —
    no in-range point is missed, and the final verdict still comes from
    the exact ``math.hypot`` distance check, so results are identical to
    a brute-force scan while lookups stay near-O(1) instead of O(n).
    """

    def __init__(self, cell: float) -> None:
        self._cell = max(1.0, float(cell))
        self._cells: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def add(self, x: float, y: float) -> None:
        key = (math.floor(x / self._cell), math.floor(y / self._cell))
        self._cells.setdefault(key, []).append((float(x), float(y)))

    def has_neighbour_within(self, x: float, y: float, radius: float) -> bool:
        span = max(1, int(math.ceil(float(radius) / self._cell)))
        cx = math.floor(x / self._cell)
        cy = math.floor(y / self._cell)
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                for ox, oy in self._cells.get((gx, gy), ()):
                    if math.hypot(x - ox, y - oy) <= float(radius):
                        return True
        return False


def deduplicate_candidate_pins(
    candidates: Sequence[Any],
    existing_nodes: Optional[Sequence[Any]] = None,
    # F-1856: was 22.0 while the duplicate rule in wrong_node_scan uses
    # SAME_POINT_PX (18.0) — two different "same point" radii. Aligned.
    dedup_radius_px: float = SAME_POINT_PX,
) -> List[Any]:
    """Spatial deduplication for candidate node detections during autonomous exploration.

    Merges duplicate nearby candidate detections within ``dedup_radius_px`` (default
    SAME_POINT_PX, 18.0) into a single accurate pin:
    - Clusters detections by Euclidean distance on map_x, map_y.
    - Repositions merged pin to confidence-weighted average coordinates.
    - Assigns highest confidence, best thumbnail image, and verified exemplar class.
    - Aggregates sightings_count and stack_count.
    - Discards / ignores candidates that already fall within ``dedup_radius_px``
      of an existing node in ``existing_nodes`` (avoiding duplicates on disk).
    """
    if not candidates:
        return []
    radius = max(1.0, _safe_conf(dedup_radius_px, SAME_POINT_PX))

    # 1. Filter out candidates that are already within dedup_radius_px of existing nodes.
    # Grid index: identical verdicts to the old O(C x E) pairwise scan, near-O(1) each.
    filtered_cands: List[Any] = []
    if existing_nodes:
        existing_grid = _SpatialGrid(radius)
        for e in existing_nodes:
            ex, ey = _get_candidate_coords(e)
            existing_grid.add(ex, ey)
        for c in candidates:
            cx, cy = _get_candidate_coords(c)
            if not existing_grid.has_neighbour_within(cx, cy, radius):
                filtered_cands.append(c)
    else:
        filtered_cands = list(candidates)

    if not filtered_cands:
        return []

    # 2. Cluster candidates that are within dedup_radius_px of each other.
    # Running (sum_x, sum_y, count) per cluster replaces rebuilding the full
    # coordinate lists and re-averaging them for every candidate; the sums
    # accumulate members in insertion order, exactly like the old sum(),
    # so the means — and therefore every cluster decision — are identical.
    clusters: List[List[Any]] = []
    cluster_sums: List[List[float]] = []
    for cand in filtered_cands:
        cx, cy = _get_candidate_coords(cand)
        matched_cluster = None
        matched_sums = None
        for idx, cl in enumerate(clusters):
            sums = cluster_sums[idx]
            mean_x = sums[0] / sums[2]
            mean_y = sums[1] / sums[2]
            if math.hypot(cx - mean_x, cy - mean_y) <= radius:
                matched_cluster = cl
                matched_sums = sums
                break
        if matched_cluster is not None:
            matched_cluster.append(cand)
            matched_sums[0] += cx
            matched_sums[1] += cy
            matched_sums[2] += 1.0
        else:
            clusters.append([cand])
            cluster_sums.append([cx, cy, 1.0])

    # 3. Merge each cluster into a single representative pin
    merged_pins: List[Any] = []
    for cl in clusters:
        if len(cl) == 1:
            merged_pins.append(cl[0])
            continue

        cl.sort(key=lambda item: _safe_conf(_get_candidate_attr(item, "confidence", 0.0), 0.0), reverse=True)
        best_item = cl[0]

        total_w = 0.0
        sum_x = 0.0
        sum_y = 0.0
        for item in cl:
            w = max(0.1, _safe_conf(_get_candidate_attr(item, "confidence", 1.0), 1.0))
            ix, iy = _get_candidate_coords(item)
            sum_x += ix * w
            sum_y += iy * w
            total_w += w
        avg_x = round(sum_x / total_w, 2)
        avg_y = round(sum_y / total_w, 2)

        total_sightings = sum(_safe_count(_get_candidate_attr(item, "sightings_count", 1), 1) for item in cl)
        max_stack = max(_safe_count(_get_candidate_attr(item, "stack_count", 1), 1) for item in cl)
        best_conf = _safe_conf(_get_candidate_attr(best_item, "confidence", 0.7), 0.7)
        merged_conf = min(0.99, round(best_conf + 0.04 * (len(cl) - 1), 3))

        best_rtype = _get_candidate_attr(best_item, "resource_type", "")
        for item in cl:
            rtype = str(_get_candidate_attr(item, "resource_type", "") or "")
            if rtype and rtype.lower() not in ("resource", "raw_node", "unknown", "unknown_3d", "sprout"):
                best_rtype = rtype
                break

        best_img = None
        for item in cl:
            img = _get_candidate_image(item)
            if img is not None:
                best_img = img
                break

        if isinstance(best_item, dict):
            merged_pin = dict(best_item)
            merged_pin["map_x"] = avg_x
            merged_pin["map_y"] = avg_y
            merged_pin["confidence"] = merged_conf
            merged_pin["sightings_count"] = total_sightings
            merged_pin["stack_count"] = max_stack
            merged_pin["resource_type"] = best_rtype
            if best_img is not None:
                merged_pin["patch_img"] = best_img
            merged_pins.append(merged_pin)
        else:
            _set_candidate_attr(best_item, "map_x", avg_x)
            _set_candidate_attr(best_item, "map_y", avg_y)
            _set_candidate_attr(best_item, "confidence", merged_conf)
            _set_candidate_attr(best_item, "sightings_count", total_sightings)
            _set_candidate_attr(best_item, "stack_count", max_stack)
            _set_candidate_attr(best_item, "resource_type", best_rtype)
            if best_img is not None and hasattr(best_item, "patch_img"):
                _set_candidate_attr(best_item, "patch_img", best_img)
            merged_pins.append(best_item)

    return merged_pins


class NodeQualityGate:
    """Multi-signal quality gate, spatial deduplication, and ExemplarMemory filtering."""

    _instance: Optional["NodeQualityGate"] = None

    def __init__(
        self,
        exemplar_memory: Any = None,
        detector: Any = None,
        conf_min: float = MODEL_CONF_MIN,
        # F-1856: default aligned with SAME_POINT_PX (18.0), was 22.0.
        dedup_radius_px: float = SAME_POINT_PX,
    ) -> None:
        self.exemplar_memory = exemplar_memory
        self.detector = detector
        self.conf_min = float(conf_min)
        self.dedup_radius_px = max(1.0, float(dedup_radius_px))
        # F-1863: evaluation counters are mutated under a lock.
        self._lock = threading.Lock()
        self.total_evaluated: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0
        self.total_deduplicated: int = 0

    @classmethod
    def get_instance(
        cls,
        exemplar_memory: Any = None,
        detector: Any = None,
        # F-1843: None means "not passed". The old defaults silently
        # overwrote nothing — a second get_instance(conf_min=0.22) kept
        # using the first call's 0.13. Explicit values now update the
        # existing singleton.
        conf_min: Optional[float] = None,
        dedup_radius_px: Optional[float] = None,
    ) -> "NodeQualityGate":
        if cls._instance is None:
            cls._instance = cls(
                exemplar_memory=exemplar_memory,
                detector=detector,
                conf_min=MODEL_CONF_MIN if conf_min is None else conf_min,
                dedup_radius_px=SAME_POINT_PX if dedup_radius_px is None else dedup_radius_px,
            )
        else:
            if exemplar_memory is not None:
                cls._instance.exemplar_memory = exemplar_memory
            if detector is not None:
                cls._instance.detector = detector
            if conf_min is not None:
                cls._instance.conf_min = float(conf_min)
            if dedup_radius_px is not None:
                cls._instance.dedup_radius_px = max(1.0, float(dedup_radius_px))
        return cls._instance

    def get_exemplar_memory(self) -> Any:
        if self.exemplar_memory is not None:
            return self.exemplar_memory
        return get_default_exemplar_memory()

    def evaluate(
        self,
        image: Optional[np.ndarray],
        *,
        source_type: str = "",
        resource_type: str = "",
        detector: Any = None,
        num_classes: Optional[int] = None,
        conf_min: Optional[float] = None,
        check_feather: bool = True,
        check_monster: bool = True,
        model_verdict: Optional[Tuple[int, float]] = None,
        check_exemplars: bool = True,
        check_terrain: bool = True,
    ) -> GateVerdict:
        with self._lock:
            self.total_evaluated += 1
        floor = self.conf_min if conf_min is None else float(conf_min)
        det = self.detector if detector is None else detector
        mem = self.get_exemplar_memory()

        verdict = evaluate_node_candidate(
            image,
            source_type=source_type,
            resource_type=resource_type,
            detector=det,
            num_classes=num_classes,
            conf_min=floor,
            check_feather=check_feather,
            check_monster=check_monster,
            model_verdict=model_verdict,
            exemplar_memory=mem,
            check_exemplars=check_exemplars,
            check_terrain=check_terrain,
        )
        with self._lock:
            if verdict.accept:
                self.total_accepted += 1
            else:
                self.total_rejected += 1
        return verdict

    def deduplicate(
        self,
        candidates: Sequence[Any],
        existing_nodes: Optional[Sequence[Any]] = None,
        radius_px: Optional[float] = None,
    ) -> List[Any]:
        r = float(radius_px if radius_px is not None else self.dedup_radius_px)
        initial_count = len(candidates)
        merged = deduplicate_candidate_pins(
            candidates,
            existing_nodes=existing_nodes,
            dedup_radius_px=r,
        )
        with self._lock:
            self.total_deduplicated += max(0, initial_count - len(merged))
        return merged

    def filter_and_deduplicate(
        self,
        candidates: Sequence[Any],
        existing_nodes: Optional[Sequence[Any]] = None,
        radius_px: Optional[float] = None,
        source_type: str = "3d_auto",
    ) -> List[Any]:
        """Gating + Deduplication pipeline for autonomous exploration."""
        accepted: List[Any] = []
        for cand in candidates:
            img = _get_candidate_image(cand)
            rtype = _get_candidate_attr(cand, "resource_type", "")
            stype = _get_candidate_attr(cand, "source_type", source_type)
            verdict = self.evaluate(
                img,
                source_type=stype,
                resource_type=rtype,
            )
            if verdict.accept:
                if verdict.verified_resource_type:
                    _set_candidate_attr(cand, "resource_type", verdict.verified_resource_type)
                if verdict.exemplar_sim is not None:
                    old_conf = _get_candidate_attr(cand, "confidence", 0.5)
                    _set_candidate_attr(cand, "confidence", max(float(old_conf or 0.5), float(verdict.exemplar_sim)))
                _set_candidate_attr(cand, "gate_verdict", verdict)
                accepted.append(cand)

        return self.deduplicate(accepted, existing_nodes=existing_nodes, radius_px=radius_px)


_default_gate: Optional[NodeQualityGate] = None


def get_node_quality_gate(
    exemplar_memory: Any = None,
    detector: Any = None,
    # F-1843: same None-means-unset rule as NodeQualityGate.get_instance.
    conf_min: Optional[float] = None,
    dedup_radius_px: Optional[float] = None,
) -> NodeQualityGate:
    """Return default singleton NodeQualityGate instance."""
    global _default_gate
    if _default_gate is None:
        _default_gate = NodeQualityGate(
            exemplar_memory=exemplar_memory,
            detector=detector,
            conf_min=MODEL_CONF_MIN if conf_min is None else conf_min,
            dedup_radius_px=SAME_POINT_PX if dedup_radius_px is None else dedup_radius_px,
        )
    else:
        if exemplar_memory is not None:
            _default_gate.exemplar_memory = exemplar_memory
        if detector is not None:
            _default_gate.detector = detector
        if conf_min is not None:
            _default_gate.conf_min = float(conf_min)
        if dedup_radius_px is not None:
            _default_gate.dedup_radius_px = max(1.0, float(dedup_radius_px))
    return _default_gate


