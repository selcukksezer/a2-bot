# -*- coding: utf-8 -*-
"""tests/test_label_trust_txt_hand_labels.py
=============================================
P0-1 (K-01, K-02, K-03): TXT uzlaştırması + commit koruması.

- TXT'deki 5 alanlı kutu, review JSON'da bayraksız kalsa bile trusted el
  etiketi sayılır (``_load_state`` + dialog taraması aynı kanonik
  ``_reconcile_txt_hand_labels`` kuralını kullanır).
- Kullanıcı açıkça şüpheliye ayırmışsa (``hand_label is False`` VEYA
  ``demoted_by_user is True``) o karar korunur, kutu weak'te kalır.
- 6 alanlı (güven skorlu) TXT satırı model çıktısıdır (w21) — el etiketi
  sayılmaz.
- ``commit_split_labels`` TXT kutusunu korur (TXT yeniden yazımında
  kaybolmaz).
- ``_rewrite_label_txt`` MEVCUT DOLU TXT'de boş emit'i reddeder, eski
  TXT'yi korur; dosya yoksa (veya zaten boşsa) boş emit boş TXT
  oluşturur — commit/relabel "eğitim TXT'si boş" sözleşmesi bozulmaz.

Bu testler yalnızca tmp_path altında kendi dataset ağaçlarını kurar;
gerçek dataset dizinine dokunmaz. P0-1 kuralı gereği bu dosyadaki
testler yazıldı ama bu turda KOŞULMADI — test ajanları koşacak.
"""

import json
from pathlib import Path

from dataset_factory.label_trust import (
    HAND_LABEL_TXT_SOURCE,
    _load_state,
    _reconcile_txt_hand_labels,
    _rewrite_label_txt,
    commit_split_labels,
    label_txt_corrupt_lines,
    quarantine_corrupt_file,
    review_json_is_corrupt,
)
from dataset_factory.multi_mix import commit_mix_sample

BOX = {"class_id": 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
TXT_LINE = "6 0.500000 0.500000 0.100000 0.100000\n"


def _setup_frame(root: Path, stem: str = "frame_001", txt_content=None,
                 review_payload=None):
    """labels/train/<stem>.txt + review/train/<stem>.json kurar."""
    lbl_dir = root / "labels" / "train"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if txt_content is not None:
        (lbl_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")
    if review_payload is not None:
        rev_dir = root / "review" / "train"
        rev_dir.mkdir(parents=True, exist_ok=True)
        (rev_dir / f"{stem}.json").write_text(
            json.dumps(review_payload), encoding="utf-8"
        )
    return lbl_dir / f"{stem}.txt"


def test_txt_only_frame_loads_as_trusted_hand_label(tmp_path):
    """Review JSON yok + 5 alanlı TXT: kutu trusted el etiketi yüklenir."""
    lbl_path = _setup_frame(tmp_path, txt_content=TXT_LINE)

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert len(trusted) == 1
    assert weak == []
    box = trusted[0]
    assert box["hand_label"] is True
    assert box["trusted"] is True
    assert box["weak"] is False
    assert box["source"] == HAND_LABEL_TXT_SOURCE
    assert box.get("legacy_auto") is None
    assert box["class_id"] == 6
    assert box["cx"] == 0.5


def test_review_json_unflagged_txt_box_reconciled_to_trusted(tmp_path):
    """Review JSON'da bayraksız weak kutu + aynı TXT satırı: trusted'a taşınır."""
    lbl_path = _setup_frame(
        tmp_path,
        txt_content=TXT_LINE,
        review_payload={
            "stem": "frame_001",
            "split": "train",
            "trusted": [],
            "weak": [dict(BOX)],
        },
    )

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert len(trusted) == 1
    assert weak == []
    assert trusted[0]["hand_label"] is True
    assert trusted[0]["source"] == HAND_LABEL_TXT_SOURCE


def test_reconcile_keeps_user_demoted_box_weak(tmp_path):
    """hand_label is False kutu şüpheliye ayrılmış sayılır, weak'te kalır."""
    lbl_path = _setup_frame(
        tmp_path,
        txt_content=TXT_LINE,
        review_payload={
            "stem": "frame_001",
            "split": "train",
            "trusted": [],
            "weak": [dict(BOX, hand_label=False)],
        },
    )

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert trusted == []
    assert len(weak) == 1
    assert weak[0]["hand_label"] is False


def test_reconcile_keeps_demoted_by_user_box_weak(tmp_path):
    """demoted_by_user is True kutu da şüpheliye ayrılmış sayılır (P0-8 yazımı)."""
    lbl_path = _setup_frame(
        tmp_path,
        txt_content=TXT_LINE,
        review_payload={
            "stem": "frame_001",
            "split": "train",
            "trusted": [],
            "weak": [dict(BOX, demoted_by_user=True)],
        },
    )

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert trusted == []
    assert len(weak) == 1
    assert weak[0]["demoted_by_user"] is True


def test_six_field_model_line_not_promoted_to_trusted(tmp_path):
    """6 alanlı (güven skorlu) TXT satırı model çıktısıdır, el etiketi değildir."""
    lbl_path = _setup_frame(
        tmp_path, txt_content="6 0.500000 0.500000 0.100000 0.100000 0.600000\n"
    )

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert trusted == []
    assert len(weak) == 1
    assert weak[0].get("has_explicit_conf") is True
    # 6 alanlı satırdan üretilen weak dict'te `hand_label` anahtarı yoktur
    # (``_legacy_weak_from_txt`` koymaz — koysaydı ``False`` değeri
    # uzlaştırmayı "kullanıcı demote etti" sanıp bozardı); `.get` ile
    # `None is not True` beklenir.
    assert weak[0].get("hand_label") is not True


def test_reconcile_does_not_touch_txt_absent_box(tmp_path):
    """TXT'de olmayan review kutusuna dokunulmaz (weak'te kalır)."""
    lbl_path = _setup_frame(
        tmp_path,
        txt_content="6 0.500000 0.500000 0.100000 0.100000\n",
        review_payload={
            "stem": "frame_001",
            "split": "train",
            "trusted": [],
            "weak": [dict(BOX, class_id=3, cx=0.2, cy=0.2)],
        },
    )

    trusted, weak = _load_state(tmp_path, "train", "frame_001", lbl_path)

    assert len(trusted) == 1  # yalnızca TXT'deki kutu
    assert trusted[0]["class_id"] == 6
    assert len(weak) == 1  # TXT'de olmayan kutu weak'te kaldı
    assert weak[0]["class_id"] == 3


def test_corrupt_review_json_does_not_reconcile(tmp_path):
    """Bozuk review JSON: uzlaştırma çalışmaz, review_corrupt bayrağı döner."""
    lbl_path = _setup_frame(tmp_path, txt_content=TXT_LINE)
    rev_dir = tmp_path / "review" / "train"
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / "frame_001.json").write_text("{bozuk json", encoding="utf-8")

    trusted, weak, meta = _load_state(
        tmp_path, "train", "frame_001", lbl_path, return_meta=True
    )

    assert trusted == []
    assert weak == []
    assert meta.get("review_corrupt") is True


def test_reconcile_is_idempotent(tmp_path):
    """Uzlaştırma iki kez çalışsa da kutu tek kez trusted'ta olur."""
    lbl_path = _setup_frame(tmp_path, txt_content=TXT_LINE)
    trusted, weak = _reconcile_txt_hand_labels([], [], lbl_path)
    trusted, weak = _reconcile_txt_hand_labels(trusted, weak, lbl_path)

    assert len(trusted) == 1
    assert weak == []


def test_commit_keeps_txt_hand_label(tmp_path):
    """commit TXT kutularını korur: boş gözlemle bile TXT satırı silinmez."""
    lbl_path = _setup_frame(
        tmp_path,
        txt_content=TXT_LINE,
        review_payload={
            "stem": "frame_001",
            "split": "train",
            "trusted": [],
            "weak": [dict(BOX)],
        },
    )

    result = commit_split_labels(
        dataset_dir=tmp_path,
        split="train",
        stem="frame_001",
        label_path=lbl_path,
        annotations=[],
    )

    assert result["train_box_count"] == 1
    assert lbl_path.read_text(encoding="utf-8").strip() == TXT_LINE.strip()
    review = json.loads(
        (tmp_path / "review" / "train" / "frame_001.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(review["trusted"]) == 1
    assert review["trusted"][0]["hand_label"] is True


def test_rewrite_label_txt_refuses_empty_emit(tmp_path):
    """Boş emit: dosya yazılmaz, eski TXT aynen korunur, False döner."""
    lbl_path = tmp_path / "frame_001.txt"
    lbl_path.write_text(TXT_LINE, encoding="utf-8")

    assert _rewrite_label_txt(lbl_path, []) is False
    assert lbl_path.read_text(encoding="utf-8") == TXT_LINE


def test_rewrite_label_txt_allows_empty_only_with_flag(tmp_path):
    """allow_empty=True ile boş emit bilerek yazılabilir, varsayılan reddeder."""
    lbl_path = tmp_path / "frame_001.txt"
    lbl_path.write_text(TXT_LINE, encoding="utf-8")

    assert _rewrite_label_txt(lbl_path, [], allow_empty=True) is True
    assert lbl_path.read_text(encoding="utf-8") == ""


def test_rewrite_label_txt_creates_empty_file_when_missing(tmp_path):
    """Dosya yoksa boş emit boş TXT oluşturur — kaybolacak veri yoktur
    (fixup P0-1: onaylanmamış learn/relabel karesinde dosya hiç
    oluşmazsa ``read_text`` ve JSON↔TXT senkronu kırılır)."""
    lbl_path = tmp_path / "frame_001.txt"
    assert not lbl_path.exists()

    assert _rewrite_label_txt(lbl_path, []) is True
    assert lbl_path.is_file()
    assert lbl_path.read_text(encoding="utf-8") == ""


def test_rewrite_label_txt_empty_existing_file_stays_empty(tmp_path):
    """Zaten boş olan TXT'de boş emit reddedilmez — korunacak veri yoktur."""
    lbl_path = tmp_path / "frame_001.txt"
    lbl_path.write_text("", encoding="utf-8")

    assert _rewrite_label_txt(lbl_path, []) is True
    assert lbl_path.read_text(encoding="utf-8") == ""


def test_rewrite_label_txt_writes_nonempty_emit(tmp_path):
    """Boş olmayan emit normal yazılır, True döner."""
    lbl_path = tmp_path / "frame_001.txt"

    assert _rewrite_label_txt(lbl_path, [dict(BOX)]) is True
    assert lbl_path.read_text(encoding="utf-8").strip() == TXT_LINE.strip()


# ---------------------------------------------------------------------------
# P1-16 (K-17 / K-05): bozuk dosya karantinası — fail-closed koruma.
# ---------------------------------------------------------------------------

def test_review_json_is_corrupt(tmp_path):
    """Okunamayan / JSON-çözümlenemeyen / dict olmayan review JSON bozuktur;
    yokluk bozukluk değildir."""
    missing = tmp_path / "yok.json"
    assert review_json_is_corrupt(missing) is False

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"stem": "x", "trusted": [], "weak": []}), encoding="utf-8")
    assert review_json_is_corrupt(ok) is False

    broken = tmp_path / "bozuk.json"
    broken.write_text("{bozuk json", encoding="utf-8")
    assert review_json_is_corrupt(broken) is True

    not_dict = tmp_path / "liste.json"
    not_dict.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert review_json_is_corrupt(not_dict) is True


def test_label_txt_corrupt_lines(tmp_path):
    """Bozuk satır numaraları (1-tabanlı); boş satır bozuk sayılmaz,
    tek bozuk satır diğer kutuları düşürmez."""
    lbl = tmp_path / "a.txt"
    lbl.write_text(
        "6 0.5 0.5 0.1 0.1\n"
        "3 0.25\n"  # 4 alan — bozuk
        "\n"
        "x 0.5 0.5 0.1 0.1\n"  # sayısal değil — bozuk
        "6 0.5 0.5 0.1 0.1 0.9\n",
        encoding="utf-8",
    )
    assert label_txt_corrupt_lines(lbl) == [2, 4]

    clean = tmp_path / "b.txt"
    clean.write_text("6 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert label_txt_corrupt_lines(clean) == []
    assert label_txt_corrupt_lines(tmp_path / "yok.txt") == []


def test_quarantine_corrupt_file_moves_review_json(tmp_path):
    """Bozuk review JSON silinmez — <dataset>/quarantine altına taşınır,
    kaynak yerinde kalmaz; yok dosya None döner."""
    rev = tmp_path / "review" / "train" / "frame_001.json"
    rev.parent.mkdir(parents=True, exist_ok=True)
    rev.write_text("{bozuk json", encoding="utf-8")

    dest = quarantine_corrupt_file(rev, dataset_dir=tmp_path, reason="test")

    assert dest is not None
    assert dest.is_file()
    assert dest.parent == tmp_path / "quarantine" / "review" / "train"
    assert dest.name.startswith("frame_001__") and dest.suffix == ".json"
    assert not rev.exists()
    assert quarantine_corrupt_file(tmp_path / "yok.json", dataset_dir=tmp_path) is None


def test_commit_blocked_on_corrupt_review_json(tmp_path):
    """Bozuk review JSON'da commit DURUR: TXT yazılmaz/silinmez, review
    üzerine yazılmaz, bozuk dosya karantinaya alınır."""
    lbl_path = _setup_frame(tmp_path, txt_content=TXT_LINE)
    rev_path = tmp_path / "review" / "train" / "frame_001.json"
    rev_path.parent.mkdir(parents=True, exist_ok=True)
    rev_path.write_text("{bozuk json", encoding="utf-8")
    txt_before = lbl_path.read_text(encoding="utf-8")

    result = commit_split_labels(
        dataset_dir=tmp_path,
        split="train",
        stem="frame_001",
        label_path=lbl_path,
        annotations=[dict(BOX)],
    )

    assert result.get("blocked") is True
    assert result.get("review_corrupt") is True
    assert result.get("train_box_count") == 0
    assert result.get("learn_auto_in_train") is False
    # TXT aynen durur, review JSON karantinaya taşındı (üzerine yazılmadı).
    assert lbl_path.read_text(encoding="utf-8") == txt_before
    assert not rev_path.exists()
    quarantined = list((tmp_path / "quarantine" / "review" / "train").glob("frame_001__*.json"))
    assert len(quarantined) == 1
    assert result.get("quarantined") == str(quarantined[0])


def test_commit_mix_sample_blocked_keeps_label_txt(tmp_path):
    """commit_mix_sample: bozuk review'da 0 kutu 'boş kare' değildir —
    labels TXT silinmez, blocked sonucu + kaynaklar aynen döner."""
    lbl_path = _setup_frame(tmp_path, txt_content=TXT_LINE)
    rev_path = tmp_path / "review" / "train" / "frame_001.json"
    rev_path.parent.mkdir(parents=True, exist_ok=True)
    rev_path.write_text("{bozuk json", encoding="utf-8")

    result = commit_mix_sample(
        dataset_dir=tmp_path,
        stem="frame_001",
        fused={"annotations": [dict(BOX)], "sources": ["yolo_detector"]},
        split="train",
    )

    assert result.get("blocked") is True
    assert result.get("review_corrupt") is True
    assert result.get("sources") == ["yolo_detector"]
    assert lbl_path.is_file()
    assert lbl_path.read_text(encoding="utf-8").strip() == TXT_LINE.strip()
    assert not rev_path.exists()
