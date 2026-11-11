# -*- coding: utf-8 -*-
"""tests/test_p3_26_single_writer.py
=====================================
P3-26 (tek yazıcı): ``write_review_pair`` + tek emit kuralı testleri.

P3-26 iki şeyi birleştirir:
1. Tek emit kuralı — ``train_txt_boxes``: commit, purge, demote, sync
   artık aynı kutuları yazar (F026-02). Yeni kural (S-6103 ile uyumlu):
   ÖLÇÜLMEMİŞ güven (anahtar yok / None / çözülemez / NaN) floor altı
   değildir — el etiketi skor taşımaz, TXT'ye girer. ÖLÇÜLMÜŞ güven
   (0.0 dahil) floor'u geçmek zorundadır.
2. Tek yazıcı — ``write_review_pair``: review JSON + labels TXT çiftini
   tek yerden yazar (atomik .tmp + os.replace, üzerine yazmadan önce
   her ikisinin de tek nesil .bak'ı, P0-2).

Ayrıca: ``_may_promote_weak_box`` — weak→trusted terfi tek predikattan;
kullanıcı demote'u (``demoted_by_user``) her köken çıkarımını veto eder
(P0-8/K-12).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory import label_trust as lt
from dataset_factory.atomic_review_sync import (
    AtomicReviewTxtSynchronizer,
    _may_promote_weak_box,
)

FLOOR = 0.45


def _box(**kw):
    b = {"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    b.update(kw)
    return b


# --- Tek emit kuralı: ölçülmemiş güven floor altı değildir (S-6103) --------

@pytest.mark.parametrize(
    "box,expected",
    [
        (_box(), True),  # güven anahtarı hiç yok → ölçülmemiş → girer
        (_box(confidence=None), True),  # None = bilinmiyor → girer
        (_box(conf="high"), True),  # çözülemez → ölçülmemiş sayılır
        (_box(score=float("nan")), True),  # NaN ölçüm değildir
        (_box(confidence=0.0), False),  # ÖLÇÜLMÜŞ 0.0 floor altı
        (_box(confidence=0.44), False),
        (_box(confidence=0.45), True),
        (_box(confidence=0.9), True),
        (_box(confidence=0.1, hand_label=True), True),  # el her zaman
        (_box(confidence=0.1, human_approved=True), False),  # onay floor'u aşmaz
        (_box(confidence=0.9, confidence_fabricated=True), False),  # E-03
        (_box(confidence=0.9, confidence_fabricated=True, human_approved=True), True),
    ],
)
def test_p326_train_txt_boxes_unmeasured_confidence_passes_floor(box, expected):
    out = lt.train_txt_boxes([box], [], stem="plain_stem", floor=FLOOR)
    assert (len(out) == 1) is expected


def test_p326_train_txt_boxes_learn_stem_weak_needs_approval_and_floor():
    approved = _box(learn_auto=True, human_approved=True, confidence=0.9, cx=0.10)
    approved_low = _box(learn_auto=True, human_approved=True, confidence=0.1, cx=0.20)
    approved_noconf = _box(learn_auto=True, human_approved=True, cx=0.30)
    unapproved = _box(learn_auto=True, confidence=0.9, cx=0.40)
    out = lt.train_txt_boxes(
        [], [approved, approved_low, approved_noconf, unapproved],
        stem="learn_skills_x", floor=FLOOR,
    )
    assert approved in out
    assert approved_low not in out  # onaylı ama floor altı
    assert approved_noconf in out  # onaylı + ölçülmemiş → girer
    assert unapproved not in out  # onaysız weak asla (E-06)


# --- Tek yazıcı: write_review_pair ------------------------------------------

def _payload(*, trusted, weak=None):
    return {
        "stem": "pair_stem",
        "split": "train",
        "done": "ok",
        "human_approved": True,
        "trusted": list(trusted),
        "weak": list(weak or []),
    }


def test_p326_write_review_pair_writes_json_and_txt_with_canonical_emit(tmp_path):
    review_p = tmp_path / "review" / "pair_stem.json"
    label_p = tmp_path / "labels" / "pair_stem.txt"
    review_p.parent.mkdir(parents=True)
    label_p.parent.mkdir(parents=True)
    review_p.write_text(json.dumps({"old": True}), encoding="utf-8")
    label_p.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    trusted = [_box(), _box(confidence=0.1), _box(confidence=0.9, cx=0.7)]
    payload = _payload(trusted=trusted)
    result = lt.write_review_pair(
        review_path=review_p, label_path=label_p,
        review_payload=payload, stem="pair_stem", floor=FLOOR,
    )
    assert result["txt_written"] is True
    assert result["emit_boxes"] == lt.train_txt_boxes(trusted, [], stem="pair_stem", floor=FLOOR)
    assert result["txt_box_count"] == 2  # ölçülmemiş + 0.9; 0.1 floor altı

    # JSON yazıldı; TXT kanonik emit ile yazıldı.
    assert json.loads(review_p.read_text(encoding="utf-8"))["trusted"] == trusted
    lines = label_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    # .bak: her iki dosyanın da tek nesil yedeği var.
    assert (tmp_path / "review" / "pair_stem.json.bak").is_file()
    assert (tmp_path / "labels" / "pair_stem.txt.bak").is_file()
    assert (tmp_path / "labels" / "pair_stem.txt.bak").read_text(encoding="utf-8") == "0 0.1 0.1 0.1 0.1\n"
    # .tmp artığı yok.
    assert list(tmp_path.rglob("*.tmp")) == []


def test_p326_write_review_pair_empty_emit_refused_but_json_written(tmp_path):
    """allow_empty=False: boş emit TXT'yi reddeder ama JSON yine yazılır
    (tarihsel commit davranışı)."""
    review_p = tmp_path / "review" / "pair_stem.json"
    label_p = tmp_path / "labels" / "pair_stem.txt"
    review_p.parent.mkdir(parents=True)
    label_p.parent.mkdir(parents=True)
    label_p.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    payload = _payload(trusted=[_box(confidence=0.1)])
    result = lt.write_review_pair(
        review_path=review_p, label_path=label_p,
        review_payload=payload, stem="pair_stem", floor=FLOOR, allow_empty=False,
    )
    assert result["txt_written"] is False
    assert result["txt_box_count"] == 0
    assert label_p.read_text(encoding="utf-8") == "3 0.5 0.5 0.2 0.2\n"  # eski TXT korundu
    assert json.loads(review_p.read_text(encoding="utf-8"))["trusted"] == [_box(confidence=0.1)]


def test_p326_write_review_pair_allow_empty_empties_txt(tmp_path):
    """Bilerek boşaltan yollar (purge/demote/temizlik): TXT boşalır."""
    review_p = tmp_path / "review" / "pair_stem.json"
    label_p = tmp_path / "labels" / "pair_stem.txt"
    review_p.parent.mkdir(parents=True)
    label_p.parent.mkdir(parents=True)
    label_p.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    result = lt.write_review_pair(
        review_path=review_p, label_path=label_p,
        review_payload=_payload(trusted=[]), stem="pair_stem",
        floor=FLOOR, allow_empty=True,
    )
    assert result["txt_written"] is True
    assert label_p.read_text(encoding="utf-8") == ""


def test_p326_write_review_pair_uses_given_emit_verbatim(tmp_path):
    """commit_split_labels'ın önceden hesaplanmış emit'i aynen kullanılır."""
    review_p = tmp_path / "review" / "pair_stem.json"
    label_p = tmp_path / "labels" / "pair_stem.txt"
    review_p.parent.mkdir(parents=True)
    label_p.parent.mkdir(parents=True)
    precomputed = [_box(confidence=0.9, cx=0.3)]
    result = lt.write_review_pair(
        review_path=review_p, label_path=label_p,
        review_payload=_payload(trusted=[_box(confidence=0.1)]),
        stem="pair_stem", floor=FLOOR, emit_boxes=precomputed,
    )
    assert result["emit_boxes"] == precomputed
    assert result["txt_box_count"] == 1
    assert "0.3" in label_p.read_text(encoding="utf-8")


# --- Terfi predikatı: demote vetosu ------------------------------------------

@pytest.mark.parametrize(
    "box,expected",
    [
        (_box(hand_label=True), True),
        (_box(human_approved=True), True),
        (_box(hand_label=True, human_approved=True), True),
        (_box(hand_label=True, demoted_by_user=True), False),  # veto (P0-8)
        (_box(human_approved=True, demoted_by_user=True), False),
        (_box(confidence=0.9), False),
        (_box(approved=True), False),
        (None, False),
        ("x", False),
    ],
)
def test_p326_may_promote_weak_box(box, expected):
    assert _may_promote_weak_box(box) is expected


# --- Sync ≡ commit: aynı emit (F026-02) --------------------------------------

def test_p326_sync_emit_equals_commit_emit():
    """Sync'in emit'i ``train_txt_boxes`` çıktısıyla birebir aynıdır:
    demote edilmiş el kutusu weak kalır ve TXT'ye girmez; ölçülmemiş
    güvenli trusted kutu girer."""
    sync = AtomicReviewTxtSynchronizer(floor=FLOOR)
    trusted = [_box(), _box(confidence=0.1), _box(confidence=0.9, cx=0.7)]
    weak = [
        _box(hand_label=True, cx=0.2),  # terfi eder (el kökeni)
        _box(hand_label=True, demoted_by_user=True, cx=0.3),  # veto: weak kalır
    ]
    json_data = {
        "stem": "sync_stem",
        "done": "ok",
        "trusted": [dict(b) for b in trusted],
        "weak": [dict(b) for b in weak],
    }
    sanitized_trusted, sanitized_weak, emit_boxes, _ = (
        sync._extract_emitted_boxes_from_json_data(json_data, stem="sync_stem")
    )
    # Demote edilmiş kutu weak listesinde kaldı.
    assert any(b.get("cx") == 0.3 and b.get("demoted_by_user") is True for b in sanitized_weak)
    # El kutusu terfi etti.
    assert any(b.get("cx") == 0.2 for b in sanitized_trusted)
    # Emit, kanonik kuralın çıktısıyla aynı (sync kendi hesabını yapmaz).
    expected_emit = lt.train_txt_boxes(sanitized_trusted, sanitized_weak, stem="sync_stem", floor=FLOOR)
    assert [ (b.get("class_id"), b.get("cx")) for b in emit_boxes ] == [
        (b.get("class_id"), b.get("cx")) for b in expected_emit
    ]
    assert len(emit_boxes) == 3  # ölçülmemiş + 0.9 + terfi eden el kutusu


def test_p326_sync_rejected_frame_emits_nothing():
    sync = AtomicReviewTxtSynchronizer(floor=FLOOR)
    json_data = {
        "stem": "sync_stem",
        "done": "no",
        "trusted": [_box(confidence=0.9)],
        "weak": [],
    }
    _, _, emit_boxes, _ = sync._extract_emitted_boxes_from_json_data(json_data, stem="sync_stem")
    assert emit_boxes == []
