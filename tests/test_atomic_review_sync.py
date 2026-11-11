# -*- coding: utf-8 -*-
"""R62 / F-1980: `dataset_factory.atomic_review_sync` import eden testler.

F-1980: bu modül repo genelinde hiçbir yerden import edilmiyordu
(`dataset_factory/__init__.py` dahil) ve 1295 satırlık public sync API
ölü kod olarak duruyordu. Karar 6 (silme YOK): modül
`dataset_factory/__init__.py`'dan public API olarak dışa aktarılacak
(dışa aktarma R46 dilimindedir) ve bu test modülü import edip public
API'sini çalıştırarak canlı tutar.

Not: `dataset_factory/dataset_validator.py` aynı yardımcıların paralel
bir kopyasını taşır (`atomic_write_text`, `atomic_write_json`,
`verify_and_sync_review_txt_pair`, `sync_all_review_txt_pairs`);
W41 testleri o kopyayı çalıştırır, bu modülü değil.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_factory.atomic_review_sync as ars
from dataset_factory.atomic_review_sync import (
    AtomicReviewTxtSynchronizer,
    SyncPairResult,
    atomic_write_json,
    atomic_write_text,
    extract_salvaged_flags,
    get_atomic_review_sync_capabilities,
    repair_json_content,
    sync_all_review_txt_pairs,
    validate_atomic_review_sync_contract,
    verify_and_sync_review_txt_pair,
)


def _list_tmp_artifacts(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if ".tmp" in p.name]


def _synced_pair(tmp_path: Path, split: str = "train", stem: str = "r62_pair_001"):
    """`write_synchronized_pair` ile senkron bir review JSON + label TXT çifti kurar."""
    syncer = AtomicReviewTxtSynchronizer(dataset_dir=tmp_path, splits=(split,))
    result = syncer.write_synchronized_pair(
        split,
        stem,
        trusted=[
            {
                "class_id": 0,
                "cx": 0.45,
                "cy": 0.55,
                "w": 0.12,
                "h": 0.18,
                "confidence": 0.94,
                "human_approved": True,
            },
            {
                "class_id": 3,
                "cx": 0.25,
                "cy": 0.35,
                "w": 0.10,
                "h": 0.14,
                "confidence": 0.58,
            },
        ],
        weak=[],
        human_approved=True,
        done="ok",
    )
    assert result.synced is True
    json_path = tmp_path / "review" / split / f"{stem}.json"
    txt_path = tmp_path / "labels" / split / f"{stem}.txt"
    assert json_path.is_file()
    assert txt_path.is_file()
    return syncer, json_path, txt_path


def test_r62_01_module_importable_and_all_names_exist() -> None:
    """F-1980: modül import ediliyor ve `__all__`'daki her isim mevcut."""
    assert ars.__name__ == "dataset_factory.atomic_review_sync"
    assert set(ars.__all__) == {
        "AtomicReviewTxtSynchronizer",
        "SyncPairResult",
        "atomic_write_text",
        "atomic_write_json",
        "extract_salvaged_flags",
        "repair_json_content",
        "verify_and_sync_review_txt_pair",
        "sync_all_review_txt_pairs",
        "get_atomic_review_sync_capabilities",
        "validate_atomic_review_sync_contract",
    }
    for name in ars.__all__:
        assert hasattr(ars, name), f"__all__ üyesi eksik: {name}"


def test_r62_02_capabilities_and_contract() -> None:
    """Public capability/contract yardımcıları beklenen yetenekleri döner."""
    caps = get_atomic_review_sync_capabilities()
    assert caps == [
        "atomic_tmp_os_replace",
        "corrupt_json_repair",
        "human_approved_done_preservation",
        "review_txt_bidirectional_sync",
    ]
    assert validate_atomic_review_sync_contract() is True


def test_r62_03_atomic_write_text_and_json_roundtrip_no_tmp_left(tmp_path: Path) -> None:
    """Atomik yazıcılar içeriği yazar ve geride .tmp bırakmaz."""
    txt_path = tmp_path / "labels" / "train" / "r62_atomic.txt"
    returned = atomic_write_text(txt_path, "0 0.450000 0.550000 0.120000 0.180000\n")
    assert returned == txt_path
    assert txt_path.read_text(encoding="utf-8") == "0 0.450000 0.550000 0.120000 0.180000\n"
    assert _list_tmp_artifacts(txt_path.parent) == []

    json_path = tmp_path / "review" / "train" / "r62_atomic.json"
    payload = {"stem": "r62_atomic", "split": "train", "trusted": [], "weak": []}
    atomic_write_json(json_path, payload)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert _list_tmp_artifacts(json_path.parent) == []


def test_r62_04_extract_salvaged_flags_last_occurrence_wins() -> None:
    """Bozuk JSON metninden bayraklar kurtarılır; son tekrar kazanır."""
    raw = '{"human_approved": false, "done": "ok"}{"human_approved": true}'
    flags = extract_salvaged_flags(raw)
    assert flags["human_approved"] is True
    assert flags["done"] == "ok"
    assert extract_salvaged_flags("") == {}


def test_r62_05_repair_json_content_valid_concat_and_garbage() -> None:
    """Onarım: geçerli JSON'a dokunmaz, birleşik JSON'u birleştirir, çöpü reddeder."""
    parsed, repaired = repair_json_content('{"stem": "a", "trusted": []}')
    assert parsed == {"stem": "a", "trusted": []}
    assert repaired is False

    parsed, repaired = repair_json_content(
        '{"stem": "a", "trusted": [{"class_id": 0}]}{"human_approved": true}'
    )
    assert repaired is True
    assert parsed["stem"] == "a"
    assert parsed["trusted"] == [{"class_id": 0}]
    assert parsed["human_approved"] is True

    parsed, repaired = repair_json_content("bu bir json degil {{{")
    assert parsed is None
    assert repaired is False


def test_r62_06_write_synchronized_pair_verifies_and_preserves_flags(tmp_path: Path) -> None:
    """Senkron yazılan çift doğrulanır; human_approved/done korunur, .tmp kalmaz."""
    _, json_path, txt_path = _synced_pair(tmp_path)

    assert verify_and_sync_review_txt_pair(json_path, txt_path, auto_repair=False) is True

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["human_approved"] is True
    assert payload["done"] == "ok"
    assert len(payload["trusted"]) == 2

    txt_lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert txt_lines == [
        "0 0.450000 0.550000 0.120000 0.180000",
        "3 0.250000 0.350000 0.100000 0.140000",
    ]
    assert _list_tmp_artifacts(json_path.parent) == []
    assert _list_tmp_artifacts(txt_path.parent) == []


def test_r62_07_orphan_json_repaired_by_txt_creation(tmp_path: Path) -> None:
    """Yetim JSON (TXT yok): onarımsız False, onarımlı True ve TXT üretilir."""
    json_path = tmp_path / "review" / "train" / "r62_orphan.json"
    atomic_write_json(
        json_path,
        {
            "stem": "r62_orphan",
            "split": "train",
            "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
            "weak": [],
        },
    )
    txt_path = tmp_path / "labels" / "train" / "r62_orphan.txt"

    assert verify_and_sync_review_txt_pair(json_path, txt_path, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(json_path, txt_path, auto_repair=True) is True
    assert txt_path.is_file()
    assert txt_path.read_text(encoding="utf-8").strip() == "1 0.500000 0.500000 0.200000 0.200000"


def test_r62_08_corrupt_json_rebuilt_from_valid_txt(tmp_path: Path) -> None:
    """Bozuk JSON + geçerli TXT: varsayılan (onarım kapalı) False, opt-in onarımlı True ve JSON yeniden kurulur.

    P0-1 (K-10): ``verify_and_sync_review_txt_pair`` varsayılanı
    ``auto_repair=False``'tur — sessiz onarım opt-in'dir. Onarımda
    kurtarılan kutulara belge düzeyi onay damgası BASILMAZ (yetim TXT
    hiçbir onayı kanıtlamaz); belge düzeyi ``human_approved`` üst
    düzeyde korunur. TXT temizliği ``_rewrite_label_txt`` üzerinden
    yapılır, yani üzerine yazılmadan önce tek nesil ``.bak`` alınır.
    """
    _, json_path, txt_path = _synced_pair(tmp_path, stem="r62_corrupt")
    # Bozuk ama bayrak metnini hâlâ taşıyan kesik JSON: onarım bayrakları
    # ham metinden kurtarıp TXT'den kutuları yeniden kurmalı.
    json_path.write_text(
        '{"stem": "r62_corrupt", "split": "train", "human_approved": true, '
        '"done": "ok", "trusted": [',
        encoding="utf-8",
    )
    original_txt_lines = [
        ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]

    # Varsayılan çağrı onarım yapmaz (verify-only): bozuk JSON yerinde kalır.
    assert verify_and_sync_review_txt_pair(json_path, txt_path) is False
    assert verify_and_sync_review_txt_pair(json_path, txt_path, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(json_path, txt_path, auto_repair=True) is True
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["human_approved"] is True
    assert payload["done"] == "ok"
    # Tasarım gereği TXT'den yeniden kurulan kutular WEAK olur, asla
    # trusted olmaz (sync_files Case 2); belge düzeyi onay kutulara
    # damgalanmaz — P0-1 (K-10): damga basılsaydı bir sonraki sync'te
    # `_may_promote_weak_box` bu kutuları trusted'a terfi ettirirdi.
    assert payload["trusted"] == []
    assert len(payload["weak"]) == 2
    assert all(b.get("human_approved") is not True for b in payload["weak"])
    assert payload["auto_repaired"] is True
    # Temizlenen TXT'nin eski içeriği tek nesil .bak'ta durur; TXT
    # boşaltılır (yeniden kurulan kutular weak-only'dir, TXT'ye girmez).
    txt_bak = txt_path.with_name(txt_path.name + ".bak")
    assert txt_bak.is_file()
    assert [ln.strip() for ln in txt_bak.read_text(encoding="utf-8").splitlines() if ln.strip()] == original_txt_lines
    assert txt_path.read_text(encoding="utf-8") == ""


def test_r62_09_sync_all_sweep_counts_synced_pair(tmp_path: Path) -> None:
    """Dizin süpürmesi senkron çifti sayar ve başarısızlık bırakmaz."""
    _synced_pair(tmp_path)
    stats = sync_all_review_txt_pairs(tmp_path, splits=("train",), auto_repair=True)
    assert stats["total_pairs"] == 1
    assert stats["already_synced"] == 1
    assert stats["failed"] == 0


def test_r62_10_sync_pair_result_to_dict_roundtrip(tmp_path: Path) -> None:
    """SyncPairResult.to_dict() alanları eksiksiz döner."""
    syncer, json_path, txt_path = _synced_pair(tmp_path)
    result = syncer.sync_pair("train", "r62_pair_001")
    assert isinstance(result, SyncPairResult)
    assert result.synced is True
    data = result.to_dict()
    assert data["stem"] == "r62_pair_001"
    assert data["split"] == "train"
    assert data["json_path"] == str(json_path)
    assert data["txt_path"] == str(txt_path)
    assert data["synced"] is True
