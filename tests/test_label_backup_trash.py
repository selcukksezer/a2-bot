# -*- coding: utf-8 -*-
"""P0-2 (K-08) yedek (.bak) testleri — reddet kalıcı silme (kullanıcı kararı 11 Kasım).

- commit / _rewrite_label_txt yıkıcı yazımdan önce tek nesil .bak yazar.
- REDDET kalıcı silmedir: reddedilen kare kuyruğa dönmez, çöp/geri
  yükleme YOKTUR — tarama-GC reddedilen kareyi düz unlink ile siler,
  dataset/.trash oluşmaz, rejected_stems kaydı kalır (kuyruk filtresi
  + GC için) ama kaldırma/geri yükleme yolu yoktur.
- .bak dosyaları dataset taramalarına girmez (yalnızca *.txt / *.json
  glob'ları + images/review/labels alt dizinleri taranır; .bak soneki
  hiçbir glob'a uymaz).

Yalnızca gerçek dosya sistemi (tmp_path) kullanılır; pytest KOŞTURULMAZ
(P0 kuralı: test ajanları koşar).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_factory import label_trust
from dataset_factory.label_trust import (
    _rewrite_label_txt,
    commit_split_labels,
    load_rejected_stems,
    restore_backup,
)
from ui.yolo_factory_dialog import YOLOFactoryDialog

STEM = "frame_bak_trash"
OLD_TXT = b"6 0.500000 0.500000 0.100000 0.200000\n"
# FİKSTÜR NOTU: HAND_BOX'un serileşimi OLD_TXT'ten farklı olmalı — cx=0.5
# aynen OLD_TXT baytlarına dönüşüyordu ("6 0.500000 ..."), bu yüzden
# "yeni içerik eskiden farklı" assertion'ı patlıyordu. cx=0.55 ile
# emit "6 0.550000 0.500000 0.100000 0.200000\n" olur.
HAND_BOX = {
    "class_id": 6,
    "class_name": "edana_crest",
    "cx": 0.55,
    "cy": 0.5,
    "w": 0.1,
    "h": 0.2,
    "confidence": 0.95,
    "trusted": True,
    "weak": False,
    "hand_label": True,
}


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for sub in ("review/train", "labels/train", "images/train"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    # CWD bağlı yazımlar (stale manifest vb.) tmp altına gitsin.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _dummy_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


def _frame(dataset: Path, stem: str = STEM, *, with_review: bool = True) -> dict:
    """Görsel + TXT (+ review JSON) içeren kare."""
    img = dataset / "images" / "train" / f"{stem}.png"
    lbl = dataset / "labels" / "train" / f"{stem}.txt"
    rev = dataset / "review" / "train" / f"{stem}.json"
    _dummy_png(img)
    lbl.write_bytes(OLD_TXT)
    if with_review:
        rev.write_text(
            json.dumps(
                {"stem": stem, "split": "train", "trusted": [dict(HAND_BOX)], "weak": []},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {"image": img, "label": lbl, "review": rev}


def _scan(dataset: Path, rejected: set) -> list:
    return YOLOFactoryDialog._scan_review_queue_items(
        dataset, rejected, approved_pool=None
    )


def test_rewrite_label_txt_writes_single_generation_bak(dataset: Path) -> None:
    paths = _frame(dataset, with_review=False)
    lbl = paths["label"]
    _rewrite_label_txt(lbl, [dict(HAND_BOX)])
    bak = lbl.with_name(lbl.name + ".bak")
    assert bak.is_file()
    assert bak.read_bytes() == OLD_TXT  # ilk yazımın .bak'i eski içerik
    first_emit = lbl.read_bytes()
    assert first_emit != OLD_TXT
    # İkinci yazım .bak'i TEK nesil tutar: .bak artık ilk emit olur.
    _rewrite_label_txt(lbl, [dict(HAND_BOX, cx=0.6)])
    assert bak.read_bytes() == first_emit
    assert lbl.read_bytes() != first_emit


def test_commit_writes_bak_for_txt_and_review(dataset: Path) -> None:
    paths = _frame(dataset)
    old_review = json.loads(paths["review"].read_text(encoding="utf-8"))
    commit_split_labels(
        dataset_dir=dataset,
        split="train",
        stem=STEM,
        label_path=paths["label"],
        annotations=[dict(HAND_BOX)],
    )
    txt_bak = paths["label"].with_name(paths["label"].name + ".bak")
    rev_bak = paths["review"].with_name(paths["review"].name + ".bak")
    assert txt_bak.is_file()
    assert txt_bak.read_bytes() == OLD_TXT
    assert rev_bak.is_file()
    assert json.loads(rev_bak.read_text(encoding="utf-8")) == old_review


def test_restore_backup_recovers_previous_content(dataset: Path) -> None:
    paths = _frame(dataset, with_review=False)
    lbl = paths["label"]
    _rewrite_label_txt(lbl, [dict(HAND_BOX)])
    assert lbl.read_bytes() != OLD_TXT
    lbl.write_bytes(b"")  # bozulma / yanlış yazım simülasyonu
    assert restore_backup(lbl) is True
    assert lbl.read_bytes() == OLD_TXT
    # .bak yoksa geri yükleme False döner, dosyaya dokunmaz.
    (lbl.with_name(lbl.name + ".bak")).unlink()
    assert restore_backup(lbl) is False


def test_bak_files_invisible_to_dataset_globs(dataset: Path) -> None:
    lbl_bak = dataset / "labels" / "train" / "frame_x.txt.bak"
    rev_bak = dataset / "review" / "train" / "frame_x.json.bak"
    lbl_bak.write_bytes(OLD_TXT)
    rev_bak.write_text("{}", encoding="utf-8")
    assert list((dataset / "labels" / "train").glob("*.txt")) == []
    assert list((dataset / "review" / "train").glob("*.json")) == []
    assert _scan(dataset, set()) == []


def test_scan_gc_permanently_deletes_rejected_frame(dataset: Path) -> None:
    # Reddedilen, el koruması OLMAYAN kare (review JSON'lu, hand bayraksız).
    # Kullanıcı kararı (11 Kasım): reddet kalıcı silmedir — tarama-GC
    # dosyaları düz unlink ile siler, çöp dizini oluşmaz, geri dönüş yok.
    img = dataset / "images" / "train" / f"{STEM}.png"
    rev = dataset / "review" / "train" / f"{STEM}.json"
    _dummy_png(img)
    rev.write_text(
        json.dumps(
            {
                "stem": STEM,
                "split": "train",
                "trusted": [],
                "weak": [dict(HAND_BOX, trusted=False, weak=True, hand_label=False)],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (dataset / "rejected_stems.json").write_text(
        json.dumps([STEM]), encoding="utf-8"
    )
    # Tarama GC'si reddedilen kareyi kuyruğa almaz ve dosyaları SİLER.
    assert _scan(dataset, {STEM}) == []
    assert not img.exists()
    assert not rev.exists()
    assert not (dataset / ".trash").exists()  # çöp dizini oluşmadı
    # rejected_stems kaydı kalır (kuyruk filtresi + GC için); kaldırma yok.
    assert load_rejected_stems(dataset) == [STEM]
    # İkinci tarama da kareyi vermez, hâlâ çöp yok.
    assert _scan(dataset, set(load_rejected_stems(dataset))) == []
    assert not (dataset / ".trash").exists()


def test_no_trash_or_rejected_restore_api() -> None:
    # Çöp/geri yükleme/kaldırma API'si kalktı — geriye dönmesin.
    for name in (
        "move_to_trash",
        "restore_from_trash",
        "trash_dir_for",
        "remove_rejected_stems",
    ):
        assert not hasattr(label_trust, name), name
    assert not hasattr(YOLOFactoryDialog, "_remove_rejected_stems")
    # .bak API'si + rejected okuma durur:
    for name in ("write_backup", "restore_backup", "backup_path_for", "load_rejected_stems"):
        assert hasattr(label_trust, name), name
