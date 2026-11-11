# -*- coding: utf-8 -*-
"""tests/test_approved_pool_origin_gate.py
=========================================
P0-6 (K-23): `sync_from_disk` köken kapısı.

Eski davranış: sync, havuz dizinindeki her >=1 geçerli kutulu görüntüyü
review JSON'u / `human_approved`'u / `done`'u / `hand_label`'ı hiç
okumadan onaylı sayıyordu — biçimsel olarak geçerli şablon kutu bile
havuza giriyordu. İlk P0-6 kapısı bunu tersine çevirdi (köken kanıtı
olmayan kare girmez) ama fazla sertti: manifest silinince review-onaylı
`good_one` (r46) ile doğrudan havuz dizinine yazılmış şablon kare
diskte ayırt edilemez — ikisi de havuz görüntüsü + geçerli 5-alanlı
TXT, review JSON yok, kaynak TXT yok. Koordinatör kararı (P0-6
fixup): havuz üyeliği + geçerli etiket TXT'si başlı başına köken
kanıtıdır (`origin="pool"`); şablon-kutu koruması dondurma anının
işidir, sync anının değil. Köken önceliği: review onayı >
el-etiket kaynak TXT'si > havuz üyeliği. Ayrıca 0-kutu / okunamayan
etiket atlamaları artık sessiz geçilmiyor (sayaç + log) ve manifest
toptan ezilmeden önce `.bak` yazılıyor.

Bu testler yalnızca tmp_path altında kendi dataset ağaçlarını kurar;
gerçek dataset dizinine dokunmaz. KOŞTURULMADI (P0 kuralı: testleri
test ajanları koşar) — yalnızca yazıldı.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from dataset_factory.approved_pool import ApprovedDatasetPool


def _valid_jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    """Encode a tiny REAL JPEG (decodable bytes, SOI..EOI) for pool tests."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2 failed to encode test JPEG fixture"
    data = bytes(buf)
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9", "fixture is not a JPEG"
    return data


def _make_pool(tmp_path: Path) -> ApprovedDatasetPool:
    return ApprovedDatasetPool(
        pool_root=tmp_path / "approved", project_root=tmp_path
    )


def _add_pool_frame(pool: ApprovedDatasetPool, stem: str, split: str = "train",
                    txt_content: str = "6 0.500000 0.500000 0.100000 0.100000\n") -> None:
    """Write one frame straight into the pool dirs (as if freeze wrote it)."""
    img_dir = pool.images_train_dir if split == "train" else pool.images_val_dir
    lbl_dir = pool.labels_train_dir if split == "train" else pool.labels_val_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(_valid_jpeg_bytes())
    (lbl_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")


def _add_review_json(tmp_path: Path, stem: str, human_approved: bool) -> None:
    rev_dir = tmp_path / "dataset" / "review"
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / f"{stem}.json").write_text(
        json.dumps({
            "stem": stem,
            "human_approved": human_approved,
            "done": "ok" if human_approved else "pending",
            "split": "train",
            # P1-10 / K-21: the box carries per-box provenance iff the
            # frame was human-approved (frame flag alone admits nothing).
            "trusted": [{"class_id": 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": human_approved}],
        }),
        encoding="utf-8",
    )


def _add_source_txt(tmp_path: Path, stem: str, split: str, txt_content: str) -> None:
    lbl_dir = tmp_path / "dataset" / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")


def test_review_approved_frame_enters_pool_via_sync(tmp_path):
    """Normal onaylı akış bozulmaz: review onayı olan kare sync'te girer."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "rev_ok_001")
    _add_review_json(tmp_path, "rev_ok_001", human_approved=True)

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 1
    assert stats["origin_review_count"] == 1
    assert stats["origin_hand_label_txt_count"] == 0
    assert pool.is_approved("rev_ok_001") is True
    assert pool.get_sample_manifest("rev_ok_001")["origin"] == "review"


def test_review_frame_without_box_provenance_stays_out_of_pool(tmp_path):
    """P1-10 / K-21: kare düzeyi `human_approved=True` tek başına havuza
    sokmaz — trusted kutulardan en az birinde kutu düzeyinde insan onayı
    (`human_approved` / `hand_label`) yoksa kare havuza girmez."""
    src = tmp_path / "dataset"
    img_train = src / "images" / "train"
    review_train = src / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)
    img_file = img_train / "rev_noprov_001.jpg"
    img_train.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(_valid_jpeg_bytes())
    (review_train / "rev_noprov_001.json").write_text(
        json.dumps({
            "stem": "rev_noprov_001",
            "human_approved": True,
            "done": "ok",
            "split": "train",
            "image_path": str(img_file),
            # Kökeni kanıtlı kutu YOK: düz trusted kutu.
            "trusted": [{"class_id": 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        }),
        encoding="utf-8",
    )

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src)

    assert stats["total_approved_images"] == 0
    assert pool.is_approved("rev_noprov_001") is False


def test_template_box_pool_member_admitted_as_pool_origin(tmp_path):
    """Havuz üyesi + geçerli etiket TXT'si köken sayılır: manifest
    silinince onay yoluyla girmiş kare ile şablon kare ayırt edilemez,
    şablon koruması dondurma anındadır — kare `origin="pool"` ile girer."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "template_001")

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 1
    assert pool.is_approved("template_001") is True
    assert pool.get_sample_manifest("template_001")["origin"] == "pool"
    assert stats["origin_pool_count"] == 1
    assert stats["origin_review_count"] == 0
    assert stats["origin_hand_label_txt_count"] == 0
    assert stats["skipped_origin_unknown_count"] == 0
    assert stats["skipped_origin_unknown"] == []


def test_model_line_source_txt_admitted_as_pool_origin(tmp_path):
    """Kaynak TXT yalnızca 6 alanlı model satırı içeriyorsa el etiketi
    sayılmaz, ama havuz üyesi + geçerli havuz TXT'si köken sayılır:
    manifest silinince köken ayırt edilemez, şablon koruması dondurma
    anındadır — kare `origin="pool"` ile girer."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "model_only_001")
    _add_source_txt(
        tmp_path, "model_only_001", "train",
        "6 0.500000 0.500000 0.100000 0.100000 0.920000\n",
    )

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 1
    assert pool.is_approved("model_only_001") is True
    assert pool.get_sample_manifest("model_only_001")["origin"] == "pool"
    assert stats["origin_pool_count"] == 1
    assert stats["origin_hand_label_txt_count"] == 0
    assert stats["skipped_origin_unknown_count"] == 0


def test_txt_hand_label_frame_enters_pool_via_sync(tmp_path):
    """TXT-only el etiketli kare (review JSON yok, kaynak TXT 5 alanlı
    geçerli kutulu) sync'te havuza girer — faz-1 dondurma ile aynı
    predikat."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "txt_only_001")
    _add_source_txt(
        tmp_path, "txt_only_001", "train",
        "6 0.500000 0.500000 0.100000 0.100000\n",
    )

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 1
    assert stats["origin_hand_label_txt_count"] == 1
    assert stats["origin_review_count"] == 0
    assert pool.is_approved("txt_only_001") is True
    assert pool.get_sample_manifest("txt_only_001")["origin"] == "hand_label_txt"


def test_review_not_approved_frame_admitted_as_pool_origin(tmp_path):
    """Review JSON'u var ama insan onayı yoksa TXT yolu kapalıdır, ama
    havuz üyeliği + geçerli etiket başlı başına kökendir: manifest
    silinince köken ayırt edilemez, şablon koruması dondurma anındadır
    — kare `origin="pool"` ile girer."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "rev_no_001")
    _add_review_json(tmp_path, "rev_no_001", human_approved=False)
    _add_source_txt(
        tmp_path, "rev_no_001", "train",
        "6 0.500000 0.500000 0.100000 0.100000\n",
    )

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 1
    assert pool.is_approved("rev_no_001") is True
    assert pool.get_sample_manifest("rev_no_001")["origin"] == "pool"
    assert stats["origin_pool_count"] == 1
    assert stats["origin_hand_label_txt_count"] == 0
    assert stats["skipped_review_not_approved_count"] == 0
    assert stats["skipped_review_not_approved"] == []


def test_zero_box_skip_is_counted(tmp_path):
    """21:24 vakası: 0-bayt TXT'li kare sessiz geçilmez — sayaçta görünür."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "empty_001", txt_content="")

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 0
    assert pool.is_approved("empty_001") is False
    assert stats["skipped_zero_box_count"] == 1
    assert stats["skipped_zero_box"] == ["train/empty_001"]


def test_unreadable_label_txt_skip_is_counted(tmp_path):
    """Okunamayan etiket TXT'li kare sessiz geçilmez — sayaçta görünür."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "unreadable_001")
    (pool.labels_train_dir / "unreadable_001.txt").write_bytes(
        b"\xff\xfe\x00\xffbad-bytes"
    )

    stats = pool.sync_from_disk()

    assert stats["total_approved_images"] == 0
    assert pool.is_approved("unreadable_001") is False
    assert stats["skipped_label_read_error_count"] == 1
    assert stats["skipped_label_read_errors"] == ["train/unreadable_001"]


def test_manifest_bak_written_before_overwrite(tmp_path):
    """Manifest toptan ezilmeden önce `.bak` yazılır ve önceki içeriği taşır."""
    pool = _make_pool(tmp_path)
    _add_pool_frame(pool, "bak_001")
    _add_review_json(tmp_path, "bak_001", human_approved=True)

    stats_first = pool.sync_from_disk()
    assert stats_first["total_approved_images"] == 1
    manifest_path = pool.manifest_path
    bak_path = Path(str(manifest_path) + ".bak")
    before_bytes = manifest_path.read_bytes()

    pool.sync_from_disk()

    assert bak_path.is_file()
    assert bak_path.read_bytes() == before_bytes
    bak_payload = json.loads(bak_path.read_text(encoding="utf-8"))
    assert bak_payload["total_approved_images"] == 1
    assert "bak_001" in bak_payload["stems"]
