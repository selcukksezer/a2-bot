# -*- coding: utf-8 -*-
"""tests/test_yaml_absolute_path.py
================================
P0-5 (ultra tarama K-28, K-29, K-30): dataset yaml'ları mutlak `path:` yazar,
curator dialog'un yaml'ını ezmez, eğitim öncesi doğrulama bozuk yaml'da
eğitimi engeller, sayaç ile motor (Ultralytics) aynı çözümleme kuralını
kullanır.

Kapsam notu: üçüncü yazıcı `ApprovedDatasetPool` (approved_pool.py) burada
tekrar test edilmez — faz-1 (0d727f6) onun için zaten
`tests/test_approved_pool_txt_hand_labels.py::test_data_approved_yaml_absolute_path_resolves_from_other_cwd`
testini ekledi; bu dosya kalan iki yazıcıyı (curator + yolo_trainer fallback)
ve K-29/K-30 davranışlarını kapsar.

Bu testler yalnızca tmp_path altında kendi dataset ağaçlarını kurar; gerçek
dataset dizinine dokunmaz. Görsel içeriği decode edilmez — sayım yalnızca
dosya varlığı + etiket (.txt) varlığına bakar, o yüzden sahte bayt yeter.
"""

from pathlib import Path

import yaml

from dataset_factory.dataset_curator import DatasetCurator
from dataset_factory.yolo_trainer import (
    _count_images_for_yaml,
    _write_fallback_approved_yaml,
    validate_training_yaml,
)


def _labeled_frame(root: Path, split: str, stem: str) -> None:
    """One image + one non-empty label txt under root/images|labels/<split>."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    (lbl_dir / f"{stem}.txt").write_text("6 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")


def _write_yaml(root: Path, path_value: str, name: str = "data.yaml") -> Path:
    """Minimal dataset yaml with a caller-chosen `path:` value."""
    yaml_path = root / name
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {path_value}",
                "train: images/train",
                "val: images/val",
                "nc: 27",
                "names:",
                "  6: herb_edana",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


# ---------------------------------------------------------------------------
# K-28 — yazıcılar mutlak yol yazar
# ---------------------------------------------------------------------------


def test_curator_yaml_writes_absolute_path(tmp_path):
    ds = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=ds)
    yaml_path = ds / "data.yaml"
    assert yaml_path.is_file()
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert Path(parsed["path"]).is_absolute()
    assert "\\" not in parsed["path"]
    assert Path(parsed["path"]).resolve() == ds.resolve()
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    assert curator.data_yaml_path == yaml_path


def test_fallback_approved_yaml_writes_absolute_path(tmp_path):
    approved_dir = tmp_path / "dataset" / "approved"
    yaml_path = approved_dir / "data_approved.yaml"
    out = _write_fallback_approved_yaml(approved_dir, yaml_path)
    assert out == yaml_path and yaml_path.is_file()
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert Path(parsed["path"]).is_absolute()
    assert "\\" not in parsed["path"]
    assert Path(parsed["path"]).resolve() == approved_dir.resolve()


# ---------------------------------------------------------------------------
# K-29 — curator dialog yaml'ını ezmez; göreli yaml'ı mutlak yola çevirir
# ---------------------------------------------------------------------------


def test_curator_does_not_overwrite_dialog_yaml(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True)
    # Dialog tarzı yaml: mutlak yol, ama curator'unkinden farklı başlık/nc.
    dialog_yaml = (
        f"path: {ds.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 12\n"
        "names:\n"
        "  0: herb_aria\n"
    )
    (ds / "data.yaml").write_text(dialog_yaml, encoding="utf-8")
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=ds)
    # ensure_dataset_structure __init__'te + ayrıca doğrudan çağrıda da ezmemeli.
    curator.ensure_dataset_structure()
    assert (ds / "data.yaml").read_text(encoding="utf-8") == dialog_yaml


def test_curator_repairs_relative_path_yaml(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True)
    (ds / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 27\nnames:\n  6: herb_edana\n",
        encoding="utf-8",
    )
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=ds)
    curator.ensure_dataset_structure()
    parsed = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert Path(parsed["path"]).is_absolute()
    assert Path(parsed["path"]).resolve() == ds.resolve()


def test_curator_creates_yaml_when_missing(tmp_path):
    ds = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=ds)
    assert (ds / "data.yaml").is_file()
    parsed = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert Path(parsed["path"]).is_absolute()


# ---------------------------------------------------------------------------
# K-30 — eğitim öncesi doğrulama
# ---------------------------------------------------------------------------


def test_validate_training_yaml_ok(tmp_path):
    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    _labeled_frame(ds, "val", "b")
    yaml_path = _write_yaml(ds, ds.as_posix())
    report = validate_training_yaml(yaml_path)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["train_count"] == 1
    assert report["val_count"] == 1
    assert report["total_labeled"] == 2


def test_validate_training_yaml_missing_val_blocks(tmp_path):
    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    yaml_path = _write_yaml(ds, ds.as_posix())
    report = validate_training_yaml(yaml_path)
    assert report["ok"] is False
    assert any("val" in e for e in report["errors"])


def test_validate_training_yaml_zero_labeled_blocks(tmp_path):
    ds = tmp_path / "dataset"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "images" / split / "nolabel.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    yaml_path = _write_yaml(ds, ds.as_posix())
    report = validate_training_yaml(yaml_path)
    assert report["ok"] is False
    assert any("0 örnek" in e for e in report["errors"])


def test_validate_training_yaml_relative_path_blocks(tmp_path):
    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    _labeled_frame(ds, "val", "b")
    yaml_path = _write_yaml(ds, ".")
    report = validate_training_yaml(yaml_path)
    assert report["ok"] is False
    assert any("göreli" in e for e in report["errors"])


def test_validate_training_yaml_missing_file_blocks(tmp_path):
    report = validate_training_yaml(tmp_path / "dataset" / "data.yaml")
    assert report["ok"] is False
    assert any("yok" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# K-30 — sayaç ile motor aynı kural: göreli `path:` CWD'ye göre çözülür
# ---------------------------------------------------------------------------


def test_count_images_uses_ultralytics_cwd_rule(tmp_path, monkeypatch):
    """Göreli `path: .` olan yaml'da sayaç CWD'deki images/ altına bakar.

    Ultralytics kuralı (F058): göreli path CWD'ye göre çözülür, yaml'ın
    dizinine göre değil. Yaml-dizininde 1 kare, CWD'de 2 kare (train+val)
    varken sayaç 2 demeli — yaml-dizini kuralı 1 derdi.
    """
    yaml_dir = tmp_path / "yamldir"
    _labeled_frame(yaml_dir, "train", "yaml_side")
    yaml_path = _write_yaml(yaml_dir, ".")
    other_cwd = tmp_path / "baska_cwd"
    _labeled_frame(other_cwd, "train", "cwd_train")
    _labeled_frame(other_cwd, "val", "cwd_val")
    monkeypatch.chdir(other_cwd)
    assert _count_images_for_yaml(yaml_path) == 2


def test_count_images_absolute_path_is_cwd_independent(tmp_path, monkeypatch):
    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    _labeled_frame(ds, "val", "b")
    yaml_path = _write_yaml(ds, ds.as_posix())
    other_cwd = tmp_path / "baska_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    assert _count_images_for_yaml(yaml_path) == 2


# ---------------------------------------------------------------------------
# K-30 — motor tarafı kapı (engine.resolve_training_yaml ikinci hattı)
# ---------------------------------------------------------------------------


def test_engine_yaml_check_raises_on_missing_val(tmp_path):
    import pytest

    from dataset_factory.rtx_trainer_engine import _check_training_yaml_or_raise

    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    yaml_path = _write_yaml(ds, ds.as_posix())
    with pytest.raises(RuntimeError, match="val"):
        _check_training_yaml_or_raise(yaml_path)


def test_engine_yaml_check_passes_on_good_yaml(tmp_path):
    from dataset_factory.rtx_trainer_engine import _check_training_yaml_or_raise

    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    _labeled_frame(ds, "val", "b")
    yaml_path = _write_yaml(ds, ds.as_posix())
    _check_training_yaml_or_raise(yaml_path)  # raises nothing
