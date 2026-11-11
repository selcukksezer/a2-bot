# -*- coding: utf-8 -*-
"""tests/test_p3_30_test_kapsama.py

P3-30 — Test kapsama: P0 düzeltmelerini kilitleyen 5 test grubu
(ultra tarama bölüm 9 madde 30; K-27 / K-01 / K-11 / K-20 / K-28-30).

Bu dosya, P0 dosyalarında ZATEN kapsanan davranışları tekrarlamaz —
yalnızca her P0 maddesinde açık kalan boşluğu kilitler:

1. Purge worker (K-27): ``_PoisonPurgeWorker`` doğrudan kurulur —
   ``_dataset_dir`` bir ``Path`` alanıdır (metot değil); ``run()``
   temizleyiciye Path'i geçirir, ``finished``/``failed`` sinyali verir.
   Ayrıca purge ve toplu-onay yeniden-giriş bayrakları ayrı ayrı
   kilitlenir (çapraz dışlama v16 kodunda YOKTUR — bkz. açık noktalar).
2. Commit 0-bayt (K-01): ``commit_split_labels`` emit boşken mevcut
   TXT'yi bayt-bayt korur, ``txt_written=False`` döner.
3. ``_reject_box`` guard (K-11): gelen ``det`` canvas/lightbox kopyası
   olup el-etiketi bayrağı taşırken bellek kutusu bayraksızsa bile red
   engellenir (guard ``det``'e de bakar).
4. Kapı sarmalı (K-20): Adım 4 → eğitim tıkı gidiş-dönüş turu —
   yalnızca korumalı bekleyen varken tur sarmala girmez (Adım 4'te
   kalır, worker başlar); korumasız bekleyen varken eğitim tıkı
   Adım 2'ye döner ve worker başlamaz.
5. YAML mutlak yol (K-28/29/30): dialog'un ``_ensure_data_yaml``
   yazıcısı CWD'den bağımsız mutlak yol yazar; çıktısı
   ``validate_training_yaml`` kapısından geçer.

Yalnızca test dosyasıdır — üretim kodu değiştirilmez.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.label_trust import commit_split_labels
from dataset_factory.yolo_trainer import validate_training_yaml
from ui.yolo_factory_dialog import (
    HAND_LABEL_TXT_SOURCE,
    _BuiltinAutoLabelingDatasetFactory,
    _PoisonPurgeWorker,
    YOLOFactoryDialog,
)


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Geçici veri seti dizinli gerçek dialog (Adım 2, kuyruk yüklemesi bitmiş)."""
    import dataset_factory.approved_pool as ap_mod

    prev_pool = ap_mod._GLOBAL_POOL
    test_pool = ap_mod.ApprovedDatasetPool(
        pool_root=tmp_path / "approved", project_root=tmp_path
    )
    ap_mod._GLOBAL_POOL = test_pool

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "labels").mkdir(parents=True, exist_ok=True)
    if hasattr(dlg, "_refresh_timer"):
        dlg._refresh_timer.stop()
    dlg._go_step(2)
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    yield dlg
    if hasattr(dlg, "_refresh_timer"):
        dlg._refresh_timer.stop()
    dlg.close()
    ap_mod._GLOBAL_POOL = prev_pool


def _protected_item(stem: str) -> Dict[str, Any]:
    """TXT kökenli el etiketi kutulu, onay bekleyen (done=None) kare."""
    return {
        "stem": stem,
        "split": "train",
        "weak": [
            {
                "class_id": 3,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "hand_label": True,
                "source": HAND_LABEL_TXT_SOURCE,
            }
        ],
        "trusted": [],
        "done": None,
    }


def _plain_pending_item(stem: str) -> Dict[str, Any]:
    """Korumasız, onay bekleyen (done=None) kare."""
    return {"stem": stem, "split": "train", "weak": [], "trusted": [], "done": None}


def _logged(log_mock, needle: str) -> bool:
    for call in log_mock.call_args_list:
        args = call.args or ()
        if args and needle in str(args[0]):
            return True
    return False


# ============================================================================
# 1. Purge worker'ı kuran test (K-27)
# ============================================================================


def test_p3_30_purge_worker_dataset_dir_is_path_not_callable(tmp_path: Path) -> None:
    """K-27 tekrarı: ``_dataset_dir`` Path alanıdır; ``()`` ile çağrılmaz.

    03068f8'deki hata ``self._dataset_dir()`` idi — Path çağrılınca
    TypeError, her purge ``failed``'a düşüyordu. Kurulum + ``run()``
    bunu yeniden üretemez.
    """
    worker = _PoisonPurgeWorker(dataset_dir=tmp_path, floor=0.90)
    assert isinstance(worker._dataset_dir, Path)
    assert worker._dataset_dir == tmp_path
    assert not callable(worker._dataset_dir)
    assert worker._floor == 0.90


def test_p3_30_purge_worker_run_passes_path_to_cleaner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run()`` temizleyiciyi Path ile çağırır, ``finished`` emit eder."""
    seen: Dict[str, Any] = {}
    finished: List[Dict[str, Any]] = []
    failed: List[str] = []
    progress: List[tuple] = []

    def fake_cleaner(dataset_dir, **kwargs):
        seen["dataset_dir"] = dataset_dir
        seen["floor"] = kwargs.get("floor")
        cb = kwargs.get("progress_callback")
        if cb is not None:
            cb(1, 2)
        return {"removed": 1, "kept": 2}

    monkeypatch.setattr(
        "dataset_factory.label_trust.clean_poison_labels_ai", fake_cleaner
    )
    worker = _PoisonPurgeWorker(dataset_dir=tmp_path, floor=0.90)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.progress.connect(lambda d, t: progress.append((d, t)))

    worker.run()  # aynı thread'de senkron — sinyal anında emit edilir

    assert isinstance(seen.get("dataset_dir"), Path)
    assert seen["dataset_dir"] == tmp_path
    assert seen["floor"] == 0.90
    assert failed == []
    assert len(finished) == 1
    assert finished[0].get("stopped") is False
    assert finished[0].get("removed") == 1
    assert progress == [(1, 2)]


def test_p3_30_purge_worker_run_failure_emits_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temizleyici patlarsa ``run()`` TypeError'a değil ``failed``'a düşer."""
    finished: List[Dict[str, Any]] = []
    failed: List[str] = []

    def boom(dataset_dir, **kwargs):
        raise RuntimeError("temizlik patladı")

    monkeypatch.setattr("dataset_factory.label_trust.clean_poison_labels_ai", boom)
    worker = _PoisonPurgeWorker(dataset_dir=tmp_path, floor=0.90)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert finished == []
    assert failed == ["temizlik patladı"]


def test_p3_30_purge_and_bulk_reentry_flags(
    factory_dialog: YOLOFactoryDialog,
) -> None:
    """Purge sürerken ikinci purge, toplu onay sürerken ikinci onay başlamaz."""
    dlg = factory_dialog

    # Purge bayrağı kuruluysa ikinci çağrı worker kurmadan döner.
    dlg._purge_running = True
    with patch.object(dlg, "_append_log") as mock_log:
        dlg._purge_poisoned_labels()
    assert dlg._purge_worker is None
    assert dlg._purge_thread is None
    assert _logged(mock_log, "zaten sürüyor")
    dlg._purge_running = False

    # Toplu-onay async bayrağı kuruluysa ikinci başlatma False döner.
    dlg._bulk_approve_running = True
    with patch.object(dlg, "_append_log") as mock_log2:
        started = dlg._start_bulk_approve_async([0], 0.90)
    assert started is False
    assert dlg._bulk_approve_worker is None
    assert _logged(mock_log2, "zaten sürüyor")
    dlg._bulk_approve_running = False


# ============================================================================
# 2. Commit 0-bayt testi (K-01)
# ============================================================================


def test_p3_30_commit_empty_emit_preserves_existing_txt(tmp_path: Path) -> None:
    """Emit boşken commit 0 bayt yazmaz: eski TXT aynen durur, ``txt_written`` False."""
    ds = tmp_path
    (ds / "review" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "train").mkdir(parents=True, exist_ok=True)

    label_path = ds / "labels" / "train" / "bos_emit.txt"
    # 6 alanlı satır = güven skorlu model çıktısı (w21 kuralı): el etiketi
    # sayılmaz, uzlaştırmada weak'te kalır — 5 alanlı olsaydı trusted el
    # etiketi sayılıp emit'e girerdi (o senaryo boş emit üretmez).
    txt_before = "3 0.500000 0.500000 0.200000 0.200000 0.60\n"
    label_path.write_text(txt_before, encoding="utf-8")

    # Yeni anotasyon yok: merge yalnızca TXT'den gelen weak model kutuyu
    # görür, onaysız weak TXT'ye girmez → emit boş.
    result = commit_split_labels(
        dataset_dir=ds,
        split="train",
        stem="bos_emit",
        label_path=label_path,
        annotations=[],
    )

    # v16'da commit sonucu `txt_written` anahtarı taşımaz (P2-21 sayacı
    # henüz yok) — kilit disk kanıtıdır: TXT bayt-bayt aynen durur.
    assert result["train_box_count"] == 0
    assert label_path.read_text(encoding="utf-8") == txt_before
    assert label_path.stat().st_size == len(txt_before.encode("utf-8"))
    # Red yalnızca yazımı engeller: .bak (tek nesil yedek) yazılmayan
    # dosya için oluşmaz; review JSON kaydı yine yazılır.
    assert not (ds / "labels" / "train" / "bos_emit.txt.bak").exists()
    review = json.loads(
        (ds / "review" / "train" / "bos_emit.json").read_text(encoding="utf-8")
    )
    assert review["stem"] == "bos_emit"


# ============================================================================
# 3. _reject_box koruma testi (K-11)
# ============================================================================


def test_p3_30_reject_blocked_when_det_copy_carries_hand_flags(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Bellek kutusu bayraksız ama gelen ``det`` (canvas kopyası) el
    etiketi bayrağı taşıyorsa red engellenir — guard ``det``'e de bakar."""
    dlg = factory_dialog
    plain_memory_box = {
        "class_id": 0,
        "class_name": "monster",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.0,
        "conf": 0.0,
        "trusted": True,
        "weak": False,
    }
    item = {
        "stem": "guard_det_flag_frame",
        "split": "train",
        "weak": [],
        "trusted": [dict(plain_memory_box)],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "guard_det_flag_frame.json"),
    }
    (tmp_path / "review" / "train" / "guard_det_flag_frame.json").write_text(
        json.dumps(
            {
                "stem": "guard_det_flag_frame",
                "split": "train",
                "trusted": [dict(plain_memory_box)],
                "weak": [],
                "done": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lbl_file = tmp_path / "labels" / "train" / "guard_det_flag_frame.txt"
    lbl_file.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    txt_before = lbl_file.read_text(encoding="utf-8")
    dlg._review_items = [item]
    dlg._review_index = 0

    # Canvas/lightbox kopyası: aynı geometri + el-etiketi bayrakları.
    det_copy = dict(plain_memory_box)
    det_copy["hand_label"] = True
    det_copy["source"] = HAND_LABEL_TXT_SOURCE

    with (
        patch.object(dlg, "_learn_box_exemplar") as mock_learn,
        patch.object(dlg, "_persist_review_item", return_value=True) as mock_persist,
        patch.object(dlg, "_append_log") as mock_log,
    ):
        dlg._reject_box(det_copy, from_trusted=True)

    assert len(item["trusted"]) == 1
    assert item["trusted"][0].get("class_id") == 0
    assert item.get("done") is None  # kare kuyrukta kalır
    mock_learn.assert_not_called()
    mock_persist.assert_not_called()
    assert _logged(mock_log, "[KORUMA]")
    assert lbl_file.read_text(encoding="utf-8") == txt_before


# ============================================================================
# 4. Kapı sarmal testi (K-20)
# ============================================================================


def _stub_train_start(monkeypatch: pytest.MonkeyPatch, dlg: YOLOFactoryDialog) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        dlg, "_start_training_worker", lambda **kwargs: calls.append(dict(kwargs))
    )
    return calls


def test_p3_30_gate_spiral_protected_only_queue_reaches_training(
    factory_dialog: YOLOFactoryDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sarmal kırıldı: yalnızca korumalı bekleyen varken Adım 4 → eğitim
    tıkı Adım 2'ye dönmez, worker başlar, adım 4'te kalır."""
    dlg = factory_dialog
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_protected_item("el_1"), _protected_item("el_2")]
    assert dlg._pending_count() == 2
    assert dlg._pending_train_blockers() == 0
    calls = _stub_train_start(monkeypatch, dlg)

    dlg._go_step(4)
    assert int(dlg._pipeline.get("step", 0)) == 4

    with patch.object(dlg, "_append_log") as mock_log:
        dlg.on_train_rtx_clicked(only_approved=False)

    assert len(calls) == 1
    assert calls[0].get("only_approved") is False
    assert int(dlg._pipeline.get("step", 0)) == 4  # Adım 2'ye dönmedi
    assert not _logged(mock_log, "[EĞİTİM] RED")


def test_p3_30_gate_spiral_unprotected_pending_bounces_to_step2(
    factory_dialog: YOLOFactoryDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Karşıt yön: korumasız bekleyen varken Adım 4'e varılır ama eğitim
    tıkı Adım 2'ye döner, worker başlamaz — sarmal yalnızca gerçek
    engelde devreye girer."""
    dlg = factory_dialog
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_protected_item("el_1"), _plain_pending_item("bekleyen_1")]
    assert dlg._pending_train_blockers() == 1
    calls = _stub_train_start(monkeypatch, dlg)

    dlg._go_step(4)
    assert int(dlg._pipeline.get("step", 0)) == 4

    with patch.object(dlg, "_append_log") as mock_log:
        dlg.on_train_rtx_clicked(only_approved=False)

    assert calls == []
    assert int(dlg._pipeline.get("step", 0)) == 2
    assert _logged(mock_log, "[EĞİTİM] RED")
    assert _logged(mock_log, "1 bekleyen")


# ============================================================================
# 5. YAML mutlak-yol testi (K-28/29/30)
# ============================================================================


def _labeled_frame(root: Path, split: str, stem: str) -> None:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    (lbl_dir / f"{stem}.txt").write_text(
        "6 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8"
    )


def test_p3_30_dialog_ensure_data_yaml_absolute_and_cwd_independent(
    factory_dialog: YOLOFactoryDialog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialog yazıcısı mutlak yol yazar; CWD değişse de içerik aynı kalır
    ve çıktı eğitim-öncesi yaml kapısından geçer.

    Yazıcı ``YOLOFactoryDialog``'un değil, dialog'un
    ``_get_builtin_helper()`` ile kurduğu
    ``_BuiltinAutoLabelingDatasetFactory``'nin metodudur
    (``ui/yolo_factory_dialog.py:517``, bağlantı ``:3680``) — test
    doğrudan o sınıfı kurar; dialog bağlantısı ayrı kilitlenir.
    """
    dlg = factory_dialog
    ds = tmp_path / "dataset"
    _labeled_frame(ds, "train", "a")
    _labeled_frame(ds, "val", "b")

    # Dialog bağlantısı: helper bu sınıfın örneğidir (disk yazımı olmadan,
    # sınıf stub'lanarak doğrulanır).
    built: List[Any] = []

    class _StubFactory:
        def __init__(self, *a, **k):
            built.append((a, k))

    monkeypatch.setattr(
        "ui.yolo_factory_dialog._BuiltinAutoLabelingDatasetFactory", _StubFactory
    )
    dlg._builtin_helper = None
    helper = dlg._get_builtin_helper()
    assert isinstance(helper, _StubFactory)
    assert len(built) == 1
    # İkinci çağrı önbellekteki örneği döner, yeniden kurmaz.
    assert dlg._get_builtin_helper() is helper
    assert len(built) == 1

    # Gerçek yazıcı: tmp project_root ile kurulur (ağır ctor; yalnızca
    # bu testte kurulur). Modül niteliği stub'lı olsa da dosya başındaki
    # import gerçek sınıfa bağlıdır.
    factory = _BuiltinAutoLabelingDatasetFactory(project_root=tmp_path)
    assert factory.dataset_dir == ds
    assert factory.dataset_dir.is_absolute()
    yaml_path = factory.data_yaml_path
    assert yaml_path == ds / "data.yaml"
    content_first = yaml_path.read_text(encoding="utf-8")

    path_line = next(
        ln for ln in content_first.splitlines() if ln.startswith("path:")
    )
    path_value = path_line.split(":", 1)[1].strip()
    assert Path(path_value).is_absolute()
    assert Path(path_value) == ds
    assert path_value != "."

    # Başka CWD'den tekrar çağrı: içerik değişmez (curator ezmesi yok).
    elsewhere = tmp_path / "baska_cwd"
    elsewhere.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(elsewhere)
    assert factory._ensure_data_yaml() == yaml_path
    assert yaml_path.read_text(encoding="utf-8") == content_first

    monkeypatch.chdir(tmp_path)
    factory._ensure_data_yaml()
    assert yaml_path.read_text(encoding="utf-8") == content_first

    # Yazılan yaml, eğitim-öncesi kapıdan (K-30) geçer.
    report = validate_training_yaml(yaml_path)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["total_labeled"] == 2
