# -*- coding: utf-8 -*-
"""tests/test_p3_27_approval_state_equivalence.py
==================================================
P3-27 (tek onay modeli): kanonik predikat denklik testleri.

``dataset_factory/approval_state.py`` bayrak okumalarını tek kanonik
tanımda toplar; anlamı DEĞİŞTİRMEZ. Bu testler her kanonik predikatı,
dal ÖNCESİ (v16) kodda yazan eski yazımla birebir karşılaştırır — eski
kural testin içinde bağımsız (kopyala-yapıştır) yazılır, üretim kodundan
import edilmez; böylece kanonik fonksiyonun gelecekte kayması yakalanır.

Eski yazımlar (v16 ağacından, grep ile çıkarıldı):
- kutu kökeni (dar tanım): ``hand_label is True or human_approved is True``
  (``label_trust._is_explicitly_human_approved``; P3-27'de o fonksiyon
  kanonik fonksiyona delege eder, adı uyumluluk için korunur).
- kare onayı (katı kural, F-0828): ``payload.get("human_approved") is True``
  (``approved_pool`` freeze kapısı, ``atomic_review_sync`` top/doc onayı).
- kuyruk-kapalı (kapsayıcı okuma): ``done == "ok" or human_approved is True``
  (dialog ``_load_state`` bulk-işaret kuralı + tarama kuyruk atlaması).
- persist türetimi (K-21, P1-10'a kadar korunur): ``bool(item.get("human_approved")
  or done == "ok")`` (dialog ``_persist_review_item``).

Notlar:
- ``box_has_human_provenance`` demote vetosu TAŞIMAZ: demote edilmiş el
  kutusu köken kanıtını korur (P0-8/K-12 — demote ``hand_label``'ı asla
  False yazmaz); veto çağrı yerinde uygulanır
  (``... and not is_demoted``). Bu test bunu açıkça kilitler.
- ``is_frame_closed`` insan-onayı iddiası DEĞİLDİR; kapılar (havuz /
  dondurma) ``is_frame_human_approved`` okur, bunu okumaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory import approval_state as aps
from dataset_factory import label_trust as lt


# ---------------------------------------------------------------------------
# Eski kurallar — v16 yazımının test-içi bağımsız kopyası.
# ---------------------------------------------------------------------------

def _old_box_provenance(box):
    """v16 ``_is_explicitly_human_approved``: dar tanım."""
    if not isinstance(box, dict):
        return False
    return bool(box.get("hand_label") is True or box.get("human_approved") is True)


def _old_frame_human_approved(payload):
    """v16 kare onayı: yalnızca ``human_approved is True``."""
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("human_approved") is True)


def _old_frame_closed(payload):
    """v16 kuyruk-kapalı: ``done == "ok"`` VEYA açık onay."""
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("done") == "ok" or payload.get("human_approved") is True)


def _old_derive_persist(item):
    """v16 ``_persist_review_item`` türetimi (K-21)."""
    if not isinstance(item, dict):
        return False
    return bool(item.get("human_approved") or item.get("done") == "ok")


# ---------------------------------------------------------------------------
# box_has_human_provenance  ≡  _is_explicitly_human_approved (eski kural)
# ---------------------------------------------------------------------------

BOX_CASES = [
    ({}, False),
    ({"hand_label": True}, True),
    ({"human_approved": True}, True),
    ({"hand_label": True, "human_approved": True}, True),
    ({"hand_label": False}, False),
    ({"human_approved": False}, False),
    ({"hand_label": False, "human_approved": False}, False),
    ({"hand_label": 1}, False),  # `is True` katılığı: 1 sayılmaz
    ({"human_approved": 1}, False),
    ({"hand_label": "True"}, False),
    ({"human_approved": "yes"}, False),
    ({"approved": True}, False),  # çıplak approved sayılmaz (E-04)
    ({"confirmed": True}, False),
    ({"bulk_approved": True}, False),  # toplu onay izi terfi etmez (B-03)
    ({"done": "ok"}, False),  # kare bayrağı kutu kökeni değildir
    ({"hand_label": True, "confidence": 0.1}, True),
    ({"human_approved": True, "confidence_fabricated": True}, True),
    ({"hand_label": True, "demoted_by_user": True}, True),  # veto çağrı yerinde
    ({"human_approved": True, "demoted_by_user": True}, True),
    ({"source": "hand"}, False),  # source metni dar tanımda sayılmaz
    ({"source": "manual"}, False),
    (None, False),
    ("hand_label", False),
    (42, False),
    ([{"hand_label": True}], False),
]


@pytest.mark.parametrize("box,expected", BOX_CASES)
def test_p327_box_provenance_matches_old_rule(box, expected):
    assert aps.box_has_human_provenance(box) is expected
    assert _old_box_provenance(box) is expected


@pytest.mark.parametrize("box,expected", BOX_CASES)
def test_p327_label_trust_alias_still_matches_canonical(box, expected):
    """Uyumluluk takma adı ``_is_explicitly_human_approved`` kanonikle aynı."""
    assert lt._is_explicitly_human_approved(box) is expected
    assert lt._is_explicitly_human_approved(box) is aps.box_has_human_provenance(box)


def test_p327_box_provenance_does_not_veto_demote_itself():
    """Demote vetosu kanonik fonksiyonda değil, çağrı yerindedir.

    El kutusu demote edilse de köken kanıtı durur; terfi kararını veren
    çağrı yeri (``_normalize_box_dict`` / ``_may_promote_weak_box``)
    ``... and not is_demoted`` ile veto eder. Kanonik fonksiyonun
    demote'a bakmaması bilinçli tercihtir — veto eklenirse bu test
    kırılır ve karar yeniden alınır.
    """
    demoted_hand = {"hand_label": True, "demoted_by_user": True}
    assert aps.box_has_human_provenance(demoted_hand) is True
    # Çağrı-yeri biçimi (P3-27 atomic sync yazımı):
    assert (aps.box_has_human_provenance(demoted_hand) and not demoted_hand.get("demoted_by_user") is True) is False


# ---------------------------------------------------------------------------
# is_frame_human_approved  ≡  eski katı kare kuralı (F-0828)
# ---------------------------------------------------------------------------

FRAME_APPROVAL_CASES = [
    ({}, False),
    ({"human_approved": True}, True),
    ({"human_approved": False}, False),
    ({"human_approved": 1}, False),
    ({"human_approved": "True"}, False),
    ({"done": "ok"}, False),  # done tek başına onay değildir (F-0828)
    ({"done": "ok", "human_approved": True}, True),
    ({"done": "ok", "human_approved": False}, False),
    ({"bulk_approved": True}, False),
    ({"approved": True}, False),
    ({"confirmed": True}, False),
    ({"human_approved": True, "bulk_approved": True}, True),
    (None, False),
    ("ok", False),
    (7, False),
]


@pytest.mark.parametrize("payload,expected", FRAME_APPROVAL_CASES)
def test_p327_frame_human_approved_matches_old_rule(payload, expected):
    assert aps.is_frame_human_approved(payload) is expected
    assert _old_frame_human_approved(payload) is expected


# ---------------------------------------------------------------------------
# is_frame_closed  ≡  eski kapsayıcı kuyruk okuması
# ---------------------------------------------------------------------------

FRAME_CLOSED_CASES = [
    ({}, False),
    ({"done": None}, False),
    ({"done": "ok"}, True),
    ({"done": "no"}, False),  # reddet=sil; "no" kapalı sayılmaz
    ({"done": "OK"}, False),
    ({"done": True}, False),
    ({"human_approved": True}, True),
    ({"human_approved": False}, False),
    ({"done": "ok", "human_approved": False}, True),
    ({"done": None, "human_approved": True}, True),
    ({"bulk_approved": True}, False),
    ({"approved": True}, False),
    (None, False),
    ([], False),
]


@pytest.mark.parametrize("payload,expected", FRAME_CLOSED_CASES)
def test_p327_frame_closed_matches_old_rule(payload, expected):
    assert aps.is_frame_closed(payload) is expected
    assert _old_frame_closed(payload) is expected


def test_p327_closed_is_not_approval():
    """``is_frame_closed`` kapı değildir: done="ok" kare kapalı ama
    insan-onaylı değildir; iki okuma karıştırılmamalı."""
    done_only = {"done": "ok"}
    assert aps.is_frame_closed(done_only) is True
    assert aps.is_frame_human_approved(done_only) is False


# ---------------------------------------------------------------------------
# is_frame_done_ok
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"done": "ok"}, True),
        ({"done": None}, False),
        ({}, False),
        ({"done": "no"}, False),
        ({"done": True}, False),
        ({"done": "OK"}, False),
        (None, False),
    ],
)
def test_p327_frame_done_ok(payload, expected):
    assert aps.is_frame_done_ok(payload) is expected


# ---------------------------------------------------------------------------
# derive_persist_human_approved  ≡  eski K-21 türetimi
# ---------------------------------------------------------------------------

DERIVE_CASES = [
    ({}, False),
    ({"done": None}, False),
    ({"done": "ok"}, True),  # K-21: done tek başına diskte onay üretir
    ({"human_approved": True}, True),
    ({"human_approved": False, "done": None}, False),
    ({"human_approved": False, "done": "ok"}, True),
    ({"human_approved": "yes", "done": None}, True),  # truthy sayılır (eski `or`)
    ({"human_approved": 0, "done": "ok"}, True),
    ({"human_approved": 0, "done": None}, False),
    ({"done": "no"}, False),
    (None, False),
]


@pytest.mark.parametrize("item,expected", DERIVE_CASES)
def test_p327_derive_persist_matches_old_rule(item, expected):
    assert aps.derive_persist_human_approved(item) is expected
    assert _old_derive_persist(item) is expected


# ---------------------------------------------------------------------------
# frame_has_provenance_box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"trusted": [{"hand_label": True}]}, True),
        ({"trusted": [{"human_approved": True}]}, True),
        ({"trusted": [{"confidence": 0.9}]}, False),
        ({"trusted": [{"approved": True}]}, False),
        ({"trusted": []}, False),
        ({}, False),
        ({"trusted": "x"}, False),
        ({"trusted": None}, False),
        ({"weak": [{"hand_label": True}]}, False),  # yalnızca trusted sayılır
        ({"trusted": [{"hand_label": False}, {"human_approved": True}]}, True),
        (None, False),
    ],
)
def test_p327_frame_has_provenance_box(payload, expected):
    assert aps.frame_has_provenance_box(payload) is expected


# ---------------------------------------------------------------------------
# Durum makinesi: FRAME_TRANSITIONS / is_allowed_frame_transition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "old,new,expected",
    [
        (None, None, True),
        (None, "ok", True),
        ("ok", None, True),
        ("ok", "ok", True),
        (None, "no", False),  # reddet kalıcı silmedir; "no" hedefi yok
        ("ok", "no", False),
        (None, True, False),
        ("ok", False, False),
        (None, "pending", False),
        ("no", "ok", False),  # makine dışı kaynak
        ("bogus", "ok", False),
        (None, "", False),
        ("ok", "OK", False),
    ],
)
def test_p327_frame_transition_table(old, new, expected):
    assert aps.is_allowed_frame_transition(old, new) is expected


def test_p327_frame_transitions_have_no_no_target():
    assert "no" not in {t for targets in aps.FRAME_TRANSITIONS.values() for t in targets}
    assert set(aps.FRAME_TRANSITIONS.keys()) == {None, "ok"}


# ---------------------------------------------------------------------------
# Yapısal: kanonik modül repo içinden hiçbir şey import etmez (döngüsüz).
# ---------------------------------------------------------------------------

def test_p327_approval_state_imports_nothing_from_repo():
    src = Path(aps.__file__).read_text(encoding="utf-8")
    assert "from dataset_factory" not in src
    assert "import dataset_factory" not in src
    assert "from ui." not in src
    assert "import ui." not in src


def test_p327_flag_table_covers_canonical_flags():
    flags = {row["flag"] for row in aps.APPROVAL_FLAG_TABLE}
    assert {"done", "human_approved", "bulk_approved", "trusted", "hand_label"} <= flags
    for row in aps.APPROVAL_FLAG_TABLE:
        assert row["flag"] and row["level"] and row["meaning"]
