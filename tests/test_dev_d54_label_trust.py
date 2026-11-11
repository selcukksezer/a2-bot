# -*- coding: utf-8 -*-
"""DEV D54 tests for dataset_factory/label_trust.py improvements.

Locks the behaviour of the D54 refactors:
  1. calibrate_box_confidence_and_trust single-pass dedup (dead second
     material-conflict / unverified checks removed, predicate extracted to
     _is_unverified_heuristic) — exact outcomes for a 20-case matrix.
  2. _find_slot / _nms_dedupe parse-hoisting + class bucketing — outputs
     identical to the pre-refactor logic (reference copies in this file),
     except _find_slot, whose reference now encodes the P1-13 same_box()
     contract (same class + IoU >= 0.45 + centre distance <= 0.05;
     SLOT_IOU 0.30 removed, ties keep the first match).
     P1-15 (K-15): _nms_dedupe hand_label MUTLAK muaf — the reference
     models the exemption, and test_nms_dedupe_hand_label_never_suppressed
     locks it directly.
  3. _iou_xyxy agrees with slot_iou.
  4. purge_poisoned_weak_reviews / demote_omni_synthetic_labels count
     corrupt review files instead of silently skipping them.
"""
import copy
import json
import random
from pathlib import Path

import pytest

from dataset_factory import label_trust as lt

# ---------------------------------------------------------------------------
# 1. calibrate behaviour matrix (expected values taken from the pre-change
#    behaviour snapshot: ~/workspace/a2-d54-scratch/snapshot_before.json)
# ---------------------------------------------------------------------------

_BASE = {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}

_CALIBRATE_CASES = [
    # (box overrides, stem, expected (trusted, weak, confidence))
    # P1-12 (K-32, never inflate): human trusts WITHOUT inflating — 0.5 stays 0.5.
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.5, "hand_label": True}, "gather_aria_3d", (True, False, 0.5)),
    ({"class_id": 1, "class_name": "herb_mela", "confidence": 0.99, "human_approved": True}, "gather_mela_3d", (True, False, 0.99)),
    # bare "approved" must NOT bypass a material conflict
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9, "approved": True}, "gather_mela_3d", (False, True, 0.45)),
    # bare "approved" must NOT bypass an unverified-heuristic source
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9, "approved": True, "source": "legacy_auto"}, "plain", (False, True, 0.68)),
    # P1-12 (K-34): bare "approved" is a pipeline flag, not a human decision — weak.
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9, "approved": True}, "plain", (False, True, 0.9)),
    # material conflict, no human
    ({"class_id": 1, "class_name": "herb_mela", "confidence": 0.9}, "gather_aria_3d", (False, True, 0.45)),
    # unverified heuristic source
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9, "source": "legacy_auto"}, "plain", (False, True, 0.68)),
    # omni synthetic source
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9, "source": "omni_layout"}, "plain", (False, True, 0.68)),
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.9}, "plain", (False, True, 0.9)),
    # E-07: high confidence alone never trusts (high_confidence_candidate=True)
    ({"class_id": 0, "class_name": "herb_aria", "confidence": 0.5}, "plain", (False, True, 0.5)),
    # essence on a 3D frame
    ({"class_id": 3, "class_name": "essence_fire", "confidence": 0.9}, "gather_aria_3d", (False, True, 0.45)),
    # non-numeric confidence + missing class_name (taxonomy fallback)
    # P1-12 (K-32): unmeasured confidence is 0.0, no 0.58 placeholder.
    ({"class_id": 0, "confidence": "abc"}, "plain", (False, True, 0.0)),
    # taxonomy fallback name (herb_aria) conflicts with a mela stem
    ({"class_id": 0, "confidence": 0.9}, "gather_mela_3d", (False, True, 0.45)),
    # explicit human decision beats a material conflict
    ({"class_id": 1, "class_name": "herb_mela", "confidence": 0.9, "hand_label": True}, "gather_aria_3d", (True, False, 0.9)),
    # F/E gather seed — Y1: only cid 13 (skill_ui_prompt) + F/E trace seeds;
    # cid 7 is herb_lumina (a material), so it must NOT seed anymore.
    # P1-12 (K-34): a seed alone is not a human decision — weak, conf kept.
    ({"class_id": 7, "class_name": "f_gather", "confidence": 0.4, "source": "f_gather_prompt", "gather_key": "f"}, "plain", (False, True, 0.4)),
    ({"class_id": 13, "class_name": "skill_ui_prompt", "confidence": 0.4, "source": "f_gather_prompt", "gather_key": "f"}, "plain", (False, True, 0.4)),
    ({"class_id": 5, "class_name": "tree_log", "confidence": 0.7, "source": "hud_skill_slot"}, "plain", (False, True, 0.68)),
    ({"class_id": 2, "class_name": "herb_targena", "confidence": 0.95, "synthetic_fallback": True}, "plain", (False, True, 0.68)),
    # E-07: 0.85 clears emin_threshold but high confidence alone never trusts
    ({"class_id": 4, "class_name": "gem_crystal", "confidence": 0.85}, "plain", (False, True, 0.85)),
    ({"class_id": 4, "class_name": "gem_crystal", "confidence": 0.8499}, "plain", (False, True, 0.8499)),
    ({"class_id": 9, "class_name": "minimap_sprout", "confidence": 0.9, "omni_synthetic": True}, "plain", (False, True, 0.68)),
]


@pytest.mark.parametrize("overrides,stem,expected", _CALIBRATE_CASES)
def test_calibrate_matrix(overrides, stem, expected):
    box = dict(_BASE)
    box.update(overrides)
    out = lt.calibrate_box_confidence_and_trust(copy.deepcopy(box), stem=stem)
    assert (out.get("trusted"), out.get("weak"), out.get("confidence")) == expected


def test_calibrate_human_skips_caps_but_bare_approved_does_not():
    # The single-pass order matters: human (hand/human_approved) skips the
    # caps; a bare approved flag goes through them first.
    human = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=0, class_name="herb_aria", confidence=0.9,
             human_approved=True, source="legacy_auto"),
        stem="plain",
    )
    assert human["trusted"] is True
    bare = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=0, class_name="herb_aria", confidence=0.9,
             approved=True, source="legacy_auto"),
        stem="plain",
    )
    assert bare["trusted"] is False and bare["confidence"] == 0.68


def test_unverified_heuristic_helper_matches_old_expression():
    # _is_unverified_heuristic must be exactly the predicate calibrate used
    # to compute inline twice: member, substring, omni prefix, flags.
    assert lt._is_unverified_heuristic({"source": "legacy_auto"}, "legacy_auto") is True
    assert lt._is_unverified_heuristic({"source": "xx_legacy_auto_yy"}, "xx_legacy_auto_yy") is True
    assert lt._is_unverified_heuristic({"source": "omni_layout"}, "omni_layout") is True
    assert lt._is_unverified_heuristic({"omni_synthetic": True}, "plain") is True
    assert lt._is_unverified_heuristic({"synthetic_fallback": True}, "plain") is True
    assert lt._is_unverified_heuristic({"source": "f_gather_prompt"}, "f_gather_prompt") is False


# ---------------------------------------------------------------------------
# 2. _find_slot / _nms_dedupe equivalence vs the pre-refactor logic
#    (P1-13: _find_slot reference encodes the same_box() contract)
# ---------------------------------------------------------------------------

def test_slot_iou_is_same_box_iou_not_030():
    # P1-13 (K-33): SLOT_IOU 0.30 is gone — the compat alias equals the
    # single same_box() threshold, so a 30%-overlap pair is NOT the same slot.
    assert lt.SLOT_IOU == 0.45
    a = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    # dx=0.09 -> IoU = (0.2-0.09)/(0.2+0.09) ~= 0.379: old SLOT_IOU 0.30
    # would have called this the same slot, same_box() must not.
    b = {"class_id": 0, "cx": 0.59, "cy": 0.5, "w": 0.2, "h": 0.2}
    assert 0.30 <= lt.slot_iou(a, b) < 0.45
    assert lt._find_slot([a], b) is None


def _ref_find_slot(boxes, candidate):
    # P1-13 (K-33): reference mirrors the single same_box() definition —
    # same class + IoU >= 0.45 + centre distance <= 0.05. Unparseable
    # geometry or a missing class never matches; best IoU wins and ties
    # keep the FIRST match. (Old reference: best IoU >= SLOT_IOU 0.30,
    # no centre check, class-blind whenever either side parsed to -1,
    # ties kept the LAST match.)
    from dataset_factory.box_match import iou_xyxy, parse_box, same_box_parsed

    cand = parse_box(candidate)
    if cand is None or cand.class_id is None:
        return None
    best_i = None
    best = 0.0
    for i, box in enumerate(boxes):
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


def _ref_nms_dedupe(boxes, iou_thresh=0.65):
    def _rank(b):
        is_t = 1.0 if (b.get("trusted") is True or b.get("hand_label") is True) else 0.0
        return (is_t, lt.annotation_confidence(b))

    sorted_boxes = sorted(boxes, key=_rank, reverse=True)
    kept = []
    dropped = 0
    for b in sorted_boxes:
        suppress = False
        # K-15 (P1-15): hand_label MUTLAK muaf — el etiketi hiçbir koşulda
        # çakışan model kutu tarafından bastırılmaz (ama kept listesine
        # girer, kendinden sonra gelen aynı-sınıf kutuları bastırır).
        if b.get("hand_label") is not True:
            for k in kept:
                if lt._safe_class_id(b.get("class_id", -1)) == lt._safe_class_id(k.get("class_id", -1)):
                    if lt.slot_iou(b, k) > iou_thresh:
                        suppress = True
                        dropped += 1
                        break
        if not suppress:
            kept.append(b)
    return kept, dropped


def _rand_box(rng):
    box = {
        "class_id": rng.choice([0, 1, 2, 3, 5, 7, 9, "3", 3.0, None, -1, True]),
        "cx": round(rng.random(), 4),
        "cy": round(rng.random(), 4),
        "w": round(rng.uniform(0.02, 0.2), 4),
        "h": round(rng.uniform(0.02, 0.2), 4),
        "confidence": round(rng.uniform(0.3, 0.99), 4),
    }
    if rng.random() < 0.2:
        box["trusted"] = True
    if rng.random() < 0.1:
        box["hand_label"] = True
    if rng.random() < 0.1:
        box["cx"] = "bozuk"  # non-numeric coordinate -> zero-area fallback
    return box


def test_find_slot_matches_reference_on_random_boxes():
    rng = random.Random(123)
    for _ in range(5):
        boxes = [_rand_box(rng) for _ in range(300)]
        for _ in range(30):
            cand = _rand_box(rng)
            assert lt._find_slot(boxes, cand) == _ref_find_slot(boxes, cand)


def test_find_slot_weird_class_ids():
    boxes = [
        {"class_id": "3", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        {"class_id": 3.0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        {"class_id": None, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        {"class_id": True, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
    ]
    for cand_cid in ("3", 3.0, 3, None, True, 1):
        cand = {"class_id": cand_cid, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        assert lt._find_slot(boxes, cand) == _ref_find_slot(boxes, cand)


def test_nms_dedupe_matches_reference_on_random_boxes():
    rng = random.Random(123)
    for trial in range(5):
        boxes = [_rand_box(rng) for _ in range(300)]
        # add near-duplicates so suppression actually fires
        # (only numeric-cx sources: _rand_box sometimes makes cx="bozuk")
        numeric = [b for b in boxes if isinstance(b.get("cx"), (int, float))]
        for _ in range(40):
            src = dict(rng.choice(numeric))
            src["cx"] = round(min(0.99, src["cx"] + 0.005), 4)
            src["confidence"] = round(rng.uniform(0.3, 0.99), 4)
            boxes.append(src)
        kept, dropped = lt._nms_dedupe(list(boxes))
        ref_kept, ref_dropped = _ref_nms_dedupe(list(boxes))
        assert kept == ref_kept
        assert dropped == ref_dropped


def test_nms_dedupe_suppresses_same_class_overlap_only():
    a = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9}
    b = {"class_id": 0, "cx": 0.51, "cy": 0.51, "w": 0.2, "h": 0.2, "confidence": 0.5}
    c = {"class_id": 1, "cx": 0.51, "cy": 0.51, "w": 0.2, "h": 0.2, "confidence": 0.5}
    kept, dropped = lt._nms_dedupe([a, b, c])
    assert kept == [a, c] and dropped == 1


def test_nms_dedupe_hand_label_never_suppressed():
    # K-15 (P1-15): hand_label MUTLAK muaf — güveni 0.0 olan TXT el
    # etiketi, daha yüksek güvenli aynı-sınıf model kutuyla çakışsa bile
    # bastırılmaz. El etiketi kept listesine girer ve kendinden sonra
    # gelen aynı-sınıf model kutuyu bastırmaya devam eder.
    model = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.99}
    hand = {"class_id": 0, "cx": 0.51, "cy": 0.51, "w": 0.2, "h": 0.2, "confidence": 0.0, "hand_label": True}
    kept, dropped = lt._nms_dedupe([model, hand])
    assert kept == [hand]
    assert dropped == 1
    # Çakışmayan el etiketi + model kutu ikisi de korunur.
    far = {"class_id": 0, "cx": 0.9, "cy": 0.9, "w": 0.1, "h": 0.1, "confidence": 0.5, "hand_label": True}
    kept, dropped = lt._nms_dedupe([model, far])
    assert len(kept) == 2 and dropped == 0


# ---------------------------------------------------------------------------
# 3. _iou_xyxy agrees with slot_iou
# ---------------------------------------------------------------------------

def test_iou_xyxy_matches_slot_iou():
    rng = random.Random(7)
    for _ in range(200):
        a = _rand_box(rng)
        b = _rand_box(rng)
        assert lt._iou_xyxy(lt._box_xyxy(a), lt._box_xyxy(b)) == lt.slot_iou(a, b)
    assert lt._iou_xyxy((0.4, 0.4, 0.6, 0.6), (0.4, 0.4, 0.6, 0.6)) == 1.0
    assert lt._iou_xyxy((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# 4. material-conflict whole-word tokens (KNOWN_MATERIAL_TOKENS_SET)
# ---------------------------------------------------------------------------

def test_material_conflict_whole_word_only():
    # "tree" inside "street" must NOT count as a material token.
    assert lt.is_material_conflict("street_3d", "herb_aria", 0) is False
    assert lt.is_material_conflict("gather_tree_3d", "herb_aria", 0) is True
    assert lt.is_material_conflict("gather_tree_3d", "tree_log", 5) is False
    assert lt.is_material_conflict("gather_aria_3d", "essence_fire", 3) is True
    assert lt.KNOWN_MATERIAL_TOKENS_SET == frozenset(lt.KNOWN_MATERIAL_TOKENS)


# ---------------------------------------------------------------------------
# 5. corrupt review files are counted, not silently skipped
# ---------------------------------------------------------------------------

def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_purge_counts_corrupt_files(tmp_path):
    root = tmp_path / "ds"
    good = {
        "stem": "frame1", "trusted": [],
        "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                  "confidence": 0.1, "source": "legacy_auto"}],
    }
    _write(root / "review" / "train" / "frame1.json", good)
    _write(root / "review" / "train" / "bad.json", "{not json")
    res = lt.purge_poisoned_weak_reviews(root, drop_sub_floor=True, floor=0.45)
    assert res["scanned"] == 2
    assert res["corrupt_files"] == 1
    assert res["weak_dropped"] == 1
    # corrupt file left untouched
    assert (root / "review" / "train" / "bad.json").read_text(encoding="utf-8") == "{not json"


def test_demote_counts_corrupt_files_once(tmp_path):
    root = tmp_path / "ds"
    good = {
        "stem": "frame1",
        "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                     "confidence": 0.9, "source": "omni_layout"}],
        "weak": [],
    }
    _write(root / "review" / "train" / "frame1.json", good)
    _write(root / "review" / "train" / "bad.json", "{not json")
    res = lt.demote_omni_synthetic_labels(root, floor=0.45)
    assert res["scanned"] == 2
    # one corrupt file, even though the inner purge pass re-scans it
    assert res["corrupt_files"] == 1
    assert res["demoted_trusted"] == 1


# ---------------------------------------------------------------------------
# 6. K-19 (P1-15): HUD düşürme nedenleri + kademeli düz-zemin eşiği +
#    güven köken damgası
# ---------------------------------------------------------------------------

def test_hud_drop_reason_codes():
    # path-stem'de dünya kaynağı: gerçek herb/ore crop'u — negatif
    # öğretilmez (HUD_NO_NEGATIVE_TEACH_REASONS üyesi).
    assert lt._hud_drop_reason(
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        stem="learn_paths_aria",
    ) == lt.HUD_DROP_PATH_WORLD_RESOURCE
    assert lt.HUD_DROP_PATH_WORLD_RESOURCE in lt.HUD_NO_NEGATIVE_TEACH_REASONS
    # path-stem'de HUD içindeki başka sınıf: klasik poison.
    assert lt._hud_drop_reason(
        {"class_id": 9, "cx": 0.9, "cy": 0.15, "w": 0.1, "h": 0.1},
        stem="learn_paths_aria",
    ) == lt.HUD_DROP_PATH_IN_HUD
    # HP/MP bandı dışı.
    assert lt._hud_drop_reason(
        {"class_id": 16, "cx": 0.5, "cy": 0.95, "w": 0.1, "h": 0.1},
        stem="plain",
    ) == lt.HUD_DROP_HP_MP_OUT_OF_BAND
    # HUD'da yasadışı sınıf.
    assert lt._hud_drop_reason(
        {"class_id": 0, "cx": 0.9, "cy": 0.15, "w": 0.1, "h": 0.1},
        stem="plain",
    ) == lt.HUD_DROP_HUD_DISALLOWED_CLASS
    # HUD'da doğrulanmamış 13 (gather prompt) — yalnızca extra_detections /
    # unverified-heuristic kaynaklarda düşer.
    assert lt._hud_drop_reason(
        {"class_id": 13, "cx": 0.9, "cy": 0.15, "w": 0.1, "h": 0.1, "source": "extra_detections"},
        stem="plain",
    ) == lt.HUD_DROP_HUD_UNVERIFIED_PROMPT
    # Gerçek detektör 13'ü korunur (F-1056 izni).
    assert lt._hud_drop_reason(
        {"class_id": 13, "cx": 0.9, "cy": 0.15, "w": 0.1, "h": 0.1, "source": "yolo_detector"},
        stem="plain",
    ) is None
    # remap_sprout: HUD'daki dünya kaynağı düşmez, yeniden sınıflanır.
    assert lt._hud_drop_reason(
        {"class_id": 0, "cx": 0.9, "cy": 0.15, "w": 0.1, "h": 0.1},
        stem="plain",
        remap_sprout=True,
    ) is None
    # Çözülemeyen koordinat / geçersiz kutu — doğrulanamaz, öğretilmez.
    assert lt._hud_drop_reason(
        {"class_id": 0, "cx": "bozuk", "cy": 0.5}, stem="plain"
    ) == lt.HUD_DROP_UNPARSEABLE_COORDS
    assert lt._hud_drop_reason("bozuk", stem="plain") == lt.HUD_DROP_INVALID_BOX
    assert lt.HUD_DROP_UNPARSEABLE_COORDS in lt.HUD_NO_NEGATIVE_TEACH_REASONS
    assert lt.HUD_DROP_INVALID_BOX in lt.HUD_NO_NEGATIVE_TEACH_REASONS
    # HUD dışı normal kutu korunur.
    assert lt._hud_drop_reason(
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        stem="plain",
    ) is None


def test_flat_texture_threshold_graduated():
    # Düz olması doğası gereği beklenen sınıflar muaf: 15 zemin,
    # 16/17/18 düz dolgulu UI çubukları.
    for cid in (15, 16, 17, 18):
        assert lt._flat_texture_threshold({"class_id": cid, "w": 0.2, "h": 0.2}) is None
    # Geri kalanlarda eşik kutu alanına göre kademelenir.
    assert lt._flat_texture_threshold({"class_id": 4, "w": 0.02, "h": 0.05}) == 3.0
    assert lt._flat_texture_threshold({"class_id": 4, "w": 0.05, "h": 0.1}) == 6.0
    assert lt._flat_texture_threshold({"class_id": 4, "w": 0.2, "h": 0.2}) == 10.0
    # Sınır değerler üst kademeye girer (küçük değil).
    assert lt._flat_texture_threshold({"class_id": 4, "w": 0.04, "h": 0.05}) == 6.0
    assert lt._flat_texture_threshold({"class_id": 4, "w": 0.1, "h": 0.1}) == 10.0


def test_calibrate_stamps_confidence_origin():
    # K-15 (P1-15): insan kararından gelen güvenin KÖKENİ kayda geçer
    # (damga yoksa insan güveni "model skoru" sanılır). Güven ŞİŞİRİLMEZ
    # (P1-12/K-32 never-inflate): 0.5 aynen 0.5 kalır, damga basılır.
    hand = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=0, class_name="herb_aria", confidence=0.5, hand_label=True),
        stem="plain",
    )
    assert hand["trusted"] is True and hand["confidence"] == 0.5
    assert hand["confidence_origin"] == "hand_label"
    approved = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=1, class_name="herb_mela", confidence=0.99, human_approved=True),
        stem="plain",
    )
    assert approved["confidence_origin"] == "human_approved"
    # P1-12/K-34: seed (şablon) insan kararı değildir — trusted değil,
    # güven aynen kalır (0.4), gather_seed damgası basılmaz.
    seed = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=13, class_name="skill_ui_prompt", confidence=0.4,
             source="f_gather_prompt", gather_key="f"),
        stem="plain",
    )
    assert seed["trusted"] is False and seed["confidence"] == 0.4
    assert seed.get("confidence_origin") is None
    # İnsan kararı yoksa damga basılmaz.
    plain = lt.calibrate_box_confidence_and_trust(
        dict(_BASE, class_id=0, class_name="herb_aria", confidence=0.9),
        stem="plain",
    )
    assert plain.get("confidence_origin") is None
