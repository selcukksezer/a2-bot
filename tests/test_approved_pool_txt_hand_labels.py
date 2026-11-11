# -*- coding: utf-8 -*-
"""tests/test_approved_pool_txt_hand_labels.py
==============================================
TXT hand-label freeze path for ApprovedDatasetPool.

Tarama (ui/yolo_factory_dialog.py::_scan_review_queue_items) review JSON'u
yok + labels TXT var ise kutuyu hand_label=True / trusted /
source="hand_label_txt" sayar — ama dondurma
(prepare_approved_training_dataset) yalnızca review JSON taradığı için bu
kareler havuza hiç girmiyordu. Kullanıcı kararı (2026-11-10): dondurma TXT
kökenli el etiketli kareleri onaylanmış saysın.

Bu testler yalnızca tmp_path altında kendi dataset ağaçlarını kurar; gerçek
dataset dizinine dokunmaz.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

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


def _add_frame(src_dir: Path, stem: str, split: str = "train",
               txt_content=None, review_payload=None) -> None:
    """Add one frame to a tmp source dataset tree.

    txt_content: YOLO TXT text (None = no labels file at all).
    review_payload: dict written as review/<stem>.json (None = no review JSON).
    """
    img_dir = src_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(_valid_jpeg_bytes())
    if txt_content is not None:
        lbl_dir = src_dir / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (lbl_dir / f"{stem}.txt").write_text(txt_content, encoding="utf-8")
    if review_payload is not None:
        rev_dir = src_dir / "review"
        rev_dir.mkdir(parents=True, exist_ok=True)
        (rev_dir / f"{stem}.json").write_text(
            json.dumps(review_payload), encoding="utf-8"
        )


def _make_pool(tmp_path: Path) -> ApprovedDatasetPool:
    return ApprovedDatasetPool(
        pool_root=tmp_path / "approved", project_root=tmp_path
    )


def test_txt_hand_label_frame_enters_pool(tmp_path):
    """Geçerli kutulu TXT-only kare (review JSON yok) havuza girer."""
    src_dir = tmp_path / "dataset"
    _add_frame(src_dir, "txt_only_001", split="train",
               txt_content="6 0.500000 0.500000 0.100000 0.100000\n")

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 1
    assert stats["txt_hand_label_candidates_added"] == 1
    assert stats["total_approved_images"] == 1
    assert stats["train_count"] == 1
    assert pool.is_approved("txt_only_001") is True
    assert (pool.images_train_dir / "txt_only_001.jpg").is_file()
    lbl = pool.labels_train_dir / "txt_only_001.txt"
    assert lbl.is_file()
    assert lbl.read_text(encoding="utf-8").strip().startswith("6 0.500000 0.500000")
    assert Path(stats["data_yaml_path"]).is_file()


def test_txt_hand_label_val_frame_keeps_val_split(tmp_path):
    """Val dizinindeki TXT-only kare val split'iyle havuza girer."""
    src_dir = tmp_path / "dataset"
    _add_frame(src_dir, "txt_only_val", split="val",
               txt_content="3 0.250000 0.350000 0.050000 0.080000\n")

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 1
    assert stats["txt_hand_label_candidates_added"] == 1
    assert stats["val_count"] == 1
    assert stats["train_count"] == 0
    assert pool.is_approved("txt_only_val") is True
    assert (pool.images_val_dir / "txt_only_val.jpg").is_file()
    assert (pool.labels_val_dir / "txt_only_val.txt").is_file()


def test_unapproved_review_json_frame_not_admitted_via_txt(tmp_path):
    """Onaysız model kutulu review JSON'lu kare TXT'si olsa da girmez."""
    src_dir = tmp_path / "dataset"
    _add_frame(
        src_dir, "model_only_001", split="train",
        txt_content="6 0.500000 0.500000 0.100000 0.100000\n",
        review_payload={
            "stem": "model_only_001",
            "image_path": str(src_dir / "images" / "train" / "model_only_001.jpg"),
            "split": "train",
            "human_approved": False,
            "done": "pending",
            "trusted": [{"class_id": 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        },
    )

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 0
    assert stats["txt_hand_label_candidates_added"] == 0
    assert stats["total_approved_images"] == 0
    assert pool.is_approved("model_only_001") is False


def test_empty_txt_frame_not_admitted(tmp_path):
    """Boş TXT'li kare girmez (W46: boş kare onaylanmaz)."""
    src_dir = tmp_path / "dataset"
    _add_frame(src_dir, "empty_txt_001", split="train", txt_content="")

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 0
    assert stats["txt_hand_label_candidates_added"] == 0
    assert stats["total_approved_images"] == 0
    assert pool.is_approved("empty_txt_001") is False


def test_invalid_only_txt_frame_not_admitted(tmp_path):
    """Yalnızca geçersiz satırlı TXT (sınıf dışı id / bozuk satır) girmez."""
    src_dir = tmp_path / "dataset"
    _add_frame(
        src_dir, "bad_txt_001", split="train",
        txt_content="99 0.500000 0.500000 0.100000 0.100000\ngarbage line\n",
    )

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 0
    assert stats["txt_hand_label_candidates_added"] == 0
    assert stats["total_approved_images"] == 0
    assert pool.is_approved("bad_txt_001") is False


def test_review_approved_frame_counted_once_not_via_txt(tmp_path):
    """Review-onaylı kare TXT'si olsa da review yolundan bir kez sayılır.

    P1-10 / K-21: review onayı kutu düzeyinde köken de ister — trusted
    kutu `"human_approved": True` taşır.
    """
    src_dir = tmp_path / "dataset"
    _add_frame(
        src_dir, "both_001", split="train",
        txt_content="6 0.500000 0.500000 0.100000 0.100000\n",
        review_payload={
            "stem": "both_001",
            "image_path": str(src_dir / "images" / "train" / "both_001.jpg"),
            "split": "train",
            "human_approved": True,
            "done": "ok",
            "trusted": [{"class_id": 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": True}],
        },
    )

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["candidates_found"] == 1
    assert stats["txt_hand_label_candidates_found"] == 0
    assert stats["txt_hand_label_candidates_added"] == 0
    assert stats["total_approved_images"] == 1
    assert pool.is_approved("both_001") is True


def test_only_model_confidence_txt_frame_not_admitted(tmp_path):
    """Yalnızca 6 alanlı (güven skorlu) satırlı TXT-only kare girmez.

    w21 regresyonu: fixture'daki `still_pending` karesinin TXT'si
    `1 0.4 0.4 0.1 0.1 0.60` — 6. alan güven skoru, yani model çıktısı
    formatıdır, el etiketi değildir. Dondurma yalnızca 5 alanlı satırı
    alır; bu kare havuza girmemelidir.
    """
    src_dir = tmp_path / "dataset"
    _add_frame(src_dir, "model_conf_only_001", split="train",
               txt_content="1 0.4 0.4 0.1 0.1 0.60\n")

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 0
    assert stats["txt_hand_label_candidates_added"] == 0
    assert stats["txt_model_lines_skipped"] == 1
    assert stats["total_approved_images"] == 0
    assert pool.is_approved("model_conf_only_001") is False


def test_mixed_txt_frame_admits_only_five_field_box(tmp_path):
    """Karışık TXT'li kare girer ama onaylanan etikette YALNIZCA 5 alanlı kutu vardır.

    Bir 5 alanlı el etiketi satırı + bir 6 alanlı (güven skorlu) model
    satırı: kare aday olur, ancak 6 alanlı satır etikete yazılmaz.
    """
    src_dir = tmp_path / "dataset"
    _add_frame(
        src_dir, "mixed_txt_001", split="train",
        txt_content=(
            "6 0.500000 0.500000 0.100000 0.100000\n"
            "1 0.400000 0.400000 0.100000 0.100000 0.600000\n"
        ),
    )

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)

    assert stats["txt_hand_label_candidates_found"] == 1
    assert stats["txt_hand_label_candidates_added"] == 1
    assert stats["txt_model_lines_skipped"] == 1
    assert stats["total_approved_images"] == 1
    assert pool.is_approved("mixed_txt_001") is True
    lbl = pool.labels_train_dir / "mixed_txt_001.txt"
    assert lbl.is_file()
    lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == ["6 0.500000 0.500000 0.100000 0.100000"]
    assert "0.400000" not in lbl.read_text(encoding="utf-8")


def test_data_approved_yaml_absolute_path_resolves_from_other_cwd(tmp_path, monkeypatch):
    """data_approved.yaml mutlak `path` yazar; train/val başka CWD'den de çözülür.

    Canlı hata (2026-11-10): yaml `path: .` yazıyordu; Ultralytics
    göreli path'i yaml'ın dizinine göre değil eğitim sürecinin CWD'sine
    (PC'de F:\\a2) göre çözdüğü için eğitim F:\\a2\\images\\val'i aradı
    ve "Hata Oluştu" ile düştü. Bu test çözümü tmp_path DIŞINDA bir
    klasöre chdir edip yapar — CWD-bağımsızlığı kanıtlanır.
    """
    src_dir = tmp_path / "dataset"
    _add_frame(src_dir, "abs_path_train", split="train",
               txt_content="6 0.500000 0.500000 0.100000 0.100000\n")
    _add_frame(src_dir, "abs_path_val", split="val",
               txt_content="3 0.250000 0.350000 0.050000 0.080000\n")

    pool = _make_pool(tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dir)
    yaml_path = Path(stats["data_yaml_path"])
    assert yaml_path.is_file()

    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    # Mutlak yol, Windows uyumu için forward-slash (ters bölü yok).
    assert Path(parsed["path"]).is_absolute()
    assert "\\" not in parsed["path"]
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    # `path` yaml'ın bulunduğu onaylı klasörün kendisidir.
    assert Path(parsed["path"]).resolve() == yaml_path.parent.resolve()

    # Ultralytics gibi çöz — ama tmp_path DIŞINDA bir CWD'den.
    other_cwd = tmp_path / "baska_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    yaml_root = Path(parsed["path"])  # mutlak path: CWD'ye bakmaz
    train_dir = (yaml_root / parsed["train"]).resolve()
    val_dir = (yaml_root / parsed["val"]).resolve()
    assert train_dir.is_dir()
    assert val_dir.is_dir()
    assert (train_dir / "abs_path_train.jpg").is_file()
    assert (val_dir / "abs_path_val.jpg").is_file()
