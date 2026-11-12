# -*- coding: utf-8 -*-
"""
fix/v20-egitim-devam — regresyon testleri.

Canlı kanıt (Selçuk, PC, 2026-11-12): 11 Kasım gecesi mAP50 0.685'e
ulaşan yolo11m/300ep koşusu durduktan sonra devam ettirilemedi, baştan
başlandı. Kök nedenler (koddan): `resume=False` sabitti, her koşu aynı
klasöre (`models/rtx_runs/gatherion_rtx_fp16`, `exist_ok=True`) yazıyor,
dialogda eğitim-devam düğmesi yoktu.

Kapsanan düzeltmeler (dataset_factory/rtx_trainer_engine.py + dialog):
(a) Taze koşu benzersiz zaman damgalı klasör alır; eski best.pt/last.pt
    üstüne yazılmaz.
(b) `resume_from` (last.pt) verilirse checkpoint doğrulanır, aynı
    klasörde `train(resume=True)` ile devam edilir.
(c) En-son-koşu / devam-edilebilir-checkpoint bulucuları (eski sabit
    isimli klasör de tanınır).
(d) Dialog düğmesinin durum mantığı motordaki yardımcıdadır
    (`find_resumable_checkpoint`) — Qt'suz burada test edilir; düğmenin
    kendisi Qt gerektirir, bu dosyada test edilmez.

Testler headless koşulur; gerçek eğitim/torch import edilmez —
Ultralytics, sahte bir modül ile `sys.modules` üzerinden verilir.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataset_factory.rtx_trainer_engine as engine_mod
from dataset_factory.rtx_trainer_engine import (
    RTXTrainerEngine,
    find_latest_run_dir,
    find_resumable_checkpoint,
    generate_unique_run_name,
    is_rtx_run_dir_name,
    resolve_resume_checkpoint,
    run_dir_for_resume_checkpoint,
    validate_resume_checkpoint,
)

_CSV_HEADER = (
    "epoch,train/box_loss,train/cls_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
)


def _write_checkpoint_zip(path: Path, *, valid: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        if valid:
            zf.writestr("archive/data.pkl", b"fake-pickle-payload")
            zf.writestr("archive/byteorder", b"little")
        else:
            zf.writestr("notes.txt", b"not a checkpoint")
    return path


def _write_results_csv(run_dir: Path, rows: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{i},1.0,1.0,0.5,0.5,0.0,0.0" for i in range(1, rows + 1)]
    csv_path = run_dir / "results.csv"
    csv_path.write_text(_CSV_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _set_mtime(path: Path, epoch: float) -> None:
    os.utime(path, (epoch, epoch))


# --- (a) Benzersiz koşu klasörü adı --------------------------------------------

def test_generate_unique_run_name_format(tmp_path):
    now = datetime.datetime(2026, 11, 12, 21, 59, 3)
    name = generate_unique_run_name(tmp_path, now=now)
    assert name == "gatherion_rtx_fp16_20261112_215903"
    assert name != "gatherion_rtx_fp16"


def test_generate_unique_run_name_collision_counter(tmp_path):
    now = datetime.datetime(2026, 11, 12, 21, 59, 3)
    first = generate_unique_run_name(tmp_path, now=now)
    (tmp_path / first).mkdir()
    second = generate_unique_run_name(tmp_path, now=now)
    assert second == "gatherion_rtx_fp16_20261112_215903_2"
    assert first != second
    (tmp_path / second).mkdir()
    third = generate_unique_run_name(tmp_path, now=now)
    assert third == "gatherion_rtx_fp16_20261112_215903_3"


def test_generate_unique_run_name_never_legacy(tmp_path):
    name = generate_unique_run_name(tmp_path)
    assert name != "gatherion_rtx_fp16"
    assert re.fullmatch(r"gatherion_rtx_fp16_\d{8}_\d{6}", name)


def test_is_rtx_run_dir_name():
    assert is_rtx_run_dir_name("gatherion_rtx_fp16") is True
    assert is_rtx_run_dir_name("gatherion_rtx_fp16_20261112_215903") is True
    assert is_rtx_run_dir_name("gatherion_rtx_fp16_20261112_215903_2") is True
    assert is_rtx_run_dir_name("gatherion_rtx_fp16_2") is False
    assert is_rtx_run_dir_name("other_run") is False
    assert is_rtx_run_dir_name("") is False


# --- (c) En-son-koşu bulucusu ---------------------------------------------------

def test_find_latest_run_dir_prefers_newest(tmp_path):
    legacy = tmp_path / "gatherion_rtx_fp16"
    (legacy / "weights").mkdir(parents=True)
    (legacy / "weights" / "best.pt").write_bytes(b"x")
    old = tmp_path / "gatherion_rtx_fp16_20261110_100000"
    (old / "weights").mkdir(parents=True)
    (old / "weights" / "best.pt").write_bytes(b"x")
    new = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    (new / "weights").mkdir(parents=True)
    (new / "weights" / "best.pt").write_bytes(b"x")
    _set_mtime(legacy / "weights" / "best.pt", 1000.0)
    _set_mtime(legacy, 1000.0)
    _set_mtime(old / "weights" / "best.pt", 2000.0)
    _set_mtime(old, 2000.0)
    _set_mtime(new / "weights" / "best.pt", 3000.0)
    _set_mtime(new, 3000.0)
    assert find_latest_run_dir(tmp_path) == new


def test_find_latest_run_dir_legacy_only(tmp_path):
    legacy = tmp_path / "gatherion_rtx_fp16"
    (legacy / "weights").mkdir(parents=True)
    (legacy / "weights" / "best.pt").write_bytes(b"x")
    assert find_latest_run_dir(tmp_path) == legacy


def test_find_latest_run_dir_empty_or_missing(tmp_path):
    assert find_latest_run_dir(tmp_path) is None
    assert find_latest_run_dir(tmp_path / "yok") is None


def test_find_latest_run_dir_ignores_unrelated_dirs(tmp_path):
    run = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    other = tmp_path / "baska_klasor"
    other.mkdir()
    _set_mtime(run / "weights" / "best.pt", 1000.0)
    _set_mtime(run, 1000.0)
    _set_mtime(other, 9999.0)
    assert find_latest_run_dir(tmp_path) == run


# --- (b) Checkpoint doğrulama ---------------------------------------------------

def test_validate_resume_checkpoint_missing(tmp_path):
    ok, reason = validate_resume_checkpoint(tmp_path / "yok" / "last.pt")
    assert ok is False
    assert "bulunamadı" in reason
    with pytest.raises(RuntimeError, match="bulunamadı"):
        resolve_resume_checkpoint(tmp_path / "yok" / "last.pt")


def test_validate_resume_checkpoint_empty_file(tmp_path):
    empty = tmp_path / "last.pt"
    empty.write_bytes(b"")
    ok, reason = validate_resume_checkpoint(empty)
    assert ok is False
    assert "boş" in reason


def test_validate_resume_checkpoint_not_a_zip(tmp_path):
    text = tmp_path / "last.pt"
    text.write_text("bu bir checkpoint degil", encoding="utf-8")
    ok, reason = validate_resume_checkpoint(text)
    assert ok is False
    assert "zip" in reason


def test_validate_resume_checkpoint_zip_without_data_pkl(tmp_path):
    bad = _write_checkpoint_zip(tmp_path / "last.pt", valid=False)
    ok, reason = validate_resume_checkpoint(bad)
    assert ok is False
    assert "data.pkl" in reason


def test_validate_resume_checkpoint_shallow_valid_zip(tmp_path):
    good = _write_checkpoint_zip(tmp_path / "last.pt", valid=True)
    ok, reason = validate_resume_checkpoint(good)
    assert ok is True, reason


def test_validate_resume_checkpoint_deep_with_real_torch_file(tmp_path):
    torch = pytest.importorskip("torch")
    # Repo kökündeki boş `torch/` dizini (veya başka bir gölge paket)
    # import'u gerçek torch olmadan da "başarılı" gösterebilir; derin
    # test yalnızca gerçek serileştirme varsa koşulur.
    if not hasattr(torch, "save"):
        pytest.skip("gerçek torch yok (torch.save bulunamadı)")
    good = tmp_path / "last.pt"
    torch.save({"epoch": 5, "model": {}, "train_args": {"epochs": 30}}, str(good))
    ok, reason = validate_resume_checkpoint(good, deep=True)
    assert ok is True, reason
    bad = tmp_path / "bad.pt"
    torch.save({"foo": 1}, str(bad))
    ok, _reason = validate_resume_checkpoint(bad, deep=True)
    assert ok is False


def test_run_dir_for_resume_checkpoint_layout(tmp_path):
    run_dir = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    ckpt = run_dir / "weights" / "last.pt"
    assert run_dir_for_resume_checkpoint(ckpt) == run_dir
    stray = tmp_path / "last.pt"
    with pytest.raises(RuntimeError, match="weights/last.pt"):
        run_dir_for_resume_checkpoint(stray)


# --- (d) Dialog düğmesi durum mantığı (Qt'suz yardımcı) ---------------------------

def test_find_resumable_checkpoint_returns_info(tmp_path):
    run_dir = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    ckpt = _write_checkpoint_zip(run_dir / "weights" / "last.pt")
    _write_results_csv(run_dir, 7)
    found = find_resumable_checkpoint(tmp_path)
    assert found is not None
    assert found["checkpoint"] == ckpt
    assert found["run_dir"] == run_dir
    assert found["run_name"] == "gatherion_rtx_fp16_20261112_215903"
    assert found["epochs_completed"] == 7


def test_find_resumable_checkpoint_none_without_last_pt(tmp_path):
    run_dir = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    (run_dir / "weights").mkdir(parents=True)
    _write_results_csv(run_dir, 3)
    assert find_resumable_checkpoint(tmp_path) is None


def test_find_resumable_checkpoint_skips_corrupt_newest(tmp_path):
    older = tmp_path / "gatherion_rtx_fp16_20261110_100000"
    older_ckpt = _write_checkpoint_zip(older / "weights" / "last.pt")
    _write_results_csv(older, 12)
    newer = tmp_path / "gatherion_rtx_fp16_20261112_215903"
    (newer / "weights").mkdir(parents=True)
    (newer / "weights" / "last.pt").write_text("bozuk", encoding="utf-8")
    _write_results_csv(newer, 2)
    _set_mtime(older / "weights" / "last.pt", 1000.0)
    _set_mtime(older, 1000.0)
    _set_mtime(newer / "weights" / "last.pt", 3000.0)
    _set_mtime(newer, 3000.0)
    found = find_resumable_checkpoint(tmp_path)
    assert found is not None
    assert found["checkpoint"] == older_ckpt
    assert found["epochs_completed"] == 12


def test_find_resumable_checkpoint_empty(tmp_path):
    assert find_resumable_checkpoint(tmp_path) is None


def test_build_config_exposes_resume_from():
    cfg = RTXTrainerEngine.build_rtx_training_config_static()
    assert "resume_from" in cfg
    assert cfg["resume_from"] is None


# --- (b)/(iv) Motor: resume çağrısı + taze koşu -----------------------------------

class _FakeYOLO:
    """Ultralytics YOLO yerine geçer; eğitim YAPMAZ, çağrıyı kaydeder."""

    instances = []

    def __init__(self, weights):
        self.weights = weights
        self.train_kwargs = None
        type(self).instances.append(self)

    def add_callback(self, _name, _fn):
        return None

    def train(self, **kwargs):
        self.train_kwargs = dict(kwargs)
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        if not (run_dir / "results.csv").is_file():
            _write_results_csv(run_dir, 3)
        (run_dir / "weights" / "best.pt").write_bytes(b"fake-best")
        return None

    def export(self, **_kwargs):
        return str(_FakeYOLO.export_path)


@pytest.fixture()
def _fake_ultralytics(monkeypatch, tmp_path):
    """Sahte ultralytics modülü + torch import engeli (sığ doğrulama yolu)."""
    _FakeYOLO.instances.clear()
    dummy_onnx = tmp_path / "dummy.onnx"
    dummy_onnx.write_bytes(b"fake-onnx")
    _FakeYOLO.export_path = dummy_onnx
    fake_mod = types.ModuleType("ultralytics")
    fake_mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    # torch import edilirse derin doğrulama devreye girer; sahte zip
    # dosyaları derin denetimden geçemez — engelle ki sığ yol test edilsin.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(engine_mod, "_point_ultralytics_runs", lambda runs_dir: None)
    yield fake_mod


def _make_engine(tmp_path):
    return RTXTrainerEngine(
        project_root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        models_dir=tmp_path / "models",
    )


def test_mode_a_rejects_bad_checkpoint_before_training(tmp_path):
    engine = _make_engine(tmp_path)
    train_cfg = RTXTrainerEngine.build_rtx_training_config_static(
        epochs=5, imgsz=640, batch=16
    )
    train_cfg["resume_from"] = str(tmp_path / "yok" / "last.pt")
    with pytest.raises(RuntimeError, match="bulunamadı"):
        engine._run_mode_a_ultralytics(
            yaml_path=tmp_path / "data.yaml",
            epochs=5,
            imgsz=640,
            batch=16,
            train_cfg=train_cfg,
            env_info={"cuda_available": False, "gpu_name": "test"},
            bat_path=tmp_path / "train.bat",
            progress_callback=None,
        )
    # Eğitim hiç başlamadı: koşu klasörü oluşmadı.
    assert _FakeYOLO.instances == []


def test_mode_a_resume_uses_checkpoint_and_resume_true(
    monkeypatch, tmp_path, _fake_ultralytics
):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "gatherion_rtx_fp16_20261112_215903"
    ckpt = _write_checkpoint_zip(run_dir / "weights" / "last.pt")
    _write_results_csv(run_dir, 30)
    (run_dir / "weights" / "best.pt").write_bytes(b"fake-best")
    monkeypatch.setattr(engine_mod, "rtx_runs_dir_for", lambda: runs_dir)

    engine = _make_engine(tmp_path)
    monkeypatch.setattr(
        engine, "_run_mode_b_opencv_prototype", lambda **kw: {"mAP50": None}
    )
    train_cfg = RTXTrainerEngine.build_rtx_training_config_static(
        epochs=5, imgsz=640, batch=16
    )
    train_cfg["resume_from"] = str(ckpt)
    result = engine._run_mode_a_ultralytics(
        yaml_path=tmp_path / "data.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        train_cfg=train_cfg,
        env_info={"cuda_available": False, "gpu_name": "test"},
        bat_path=tmp_path / "train.bat",
        progress_callback=None,
    )
    assert len(_FakeYOLO.instances) == 1
    fake = _FakeYOLO.instances[0]
    assert fake.weights == str(ckpt)
    assert fake.train_kwargs["resume"] is True
    assert fake.train_kwargs["name"] == "gatherion_rtx_fp16_20261112_215903"
    assert fake.train_kwargs["project"] == str(runs_dir)
    assert result["artifacts"]["run_dir"] == str(run_dir)
    assert result["epochs_completed"] == 30


def test_mode_a_fresh_run_uses_unique_name_and_resume_false(
    monkeypatch, tmp_path, _fake_ultralytics
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(engine_mod, "rtx_runs_dir_for", lambda: runs_dir)

    engine = _make_engine(tmp_path)
    monkeypatch.setattr(
        engine, "_run_mode_b_opencv_prototype", lambda **kw: {"mAP50": None}
    )
    train_cfg = RTXTrainerEngine.build_rtx_training_config_static(
        epochs=5, imgsz=640, batch=16
    )
    assert train_cfg["resume_from"] is None
    result = engine._run_mode_a_ultralytics(
        yaml_path=tmp_path / "data.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        train_cfg=train_cfg,
        env_info={"cuda_available": False, "gpu_name": "test"},
        bat_path=tmp_path / "train.bat",
        progress_callback=None,
    )
    assert len(_FakeYOLO.instances) == 1
    fake = _FakeYOLO.instances[0]
    assert fake.train_kwargs["resume"] is False
    run_name = fake.train_kwargs["name"]
    assert run_name != "gatherion_rtx_fp16"
    assert re.fullmatch(r"gatherion_rtx_fp16_\d{8}_\d{6}", run_name)
    assert Path(result["artifacts"]["run_dir"]).name == run_name
    assert result["epochs_completed"] == 3
