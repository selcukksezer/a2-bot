# -*- coding: utf-8 -*-
"""tests/test_train_gate_delta.py

P0-4 — Eğitim kapısı deltası (ultra tarama K-20) hedefli testleri.

Kapsam (v15 + fix/p0-4-kapi-delta):
1. Dondurma kapısı `_pending_train_blockers` kullanır: el etiketli korumalı
   kareler dondurmayı kilitlemez; yalnızca gerçekten bekleyen kareler engeldir.
2. Eğitim kapısı onaylı (yeşil) modda da pending kontrol eder: yaml varlığı
   tek başına yetmez, engel varken eğitim başlamaz, kullanıcıya neden + sayı
   söylenir.
3. `_update_train_lock` + özet satırı kapı kararıyla tutarlıdır: engel varken
   iki eğitim butonu da kilitli görünür, özet satırı engeli yazar.

NOT: Bu dosyadaki testler KOŞTURULMAZ (P0 kuralı: düzeltme ajanları test
koşmaz, yalnızca py_compile). Testleri test ajanları çalıştırır.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import HAND_LABEL_TXT_SOURCE, YOLOFactoryDialog


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


def _done_item(stem: str) -> Dict[str, Any]:
    """Onaylanmış kare (kapı sayımına girmez)."""
    return {
        "stem": stem,
        "split": "train",
        "weak": [],
        "trusted": [{"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        "done": "ok",
    }


def _set_approved_mode(dlg: YOLOFactoryDialog, enabled: bool) -> None:
    """Onaylı-havuz modunu aç/kapat (var olan tüm ilgili anahtarlar)."""
    for attr in (
        "chk_only_approved",
        "chk_train_only_approved",
        "toggle_approved_only_prepare",
        "toggle_approved_only_train",
        "toggle_approved_only",
    ):
        widget = getattr(dlg, attr, None)
        if widget is not None:
            widget.setChecked(enabled)


def _log_text(dlg: YOLOFactoryDialog) -> str:
    return dlg.training_log_box.toPlainText()


def _stub_ready_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ham-yol dondurma QA kapısını hazır say (kapı sayacı test ediliyor)."""
    monkeypatch.setattr(
        "dataset_factory.label_trust.is_dataset_ready_for_training",
        lambda *args, **kwargs: (True, "ok"),
    )


def _stub_train_start(monkeypatch: pytest.MonkeyPatch, dlg: YOLOFactoryDialog) -> List[Dict[str, Any]]:
    """Gerçek worker başlatmadan eğitim-başlatma denemesini kaydet."""
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        dlg, "_start_training_worker", lambda **kwargs: calls.append(dict(kwargs))
    )
    return calls


# --- Sayaç -------------------------------------------------------------------

def test_pending_train_blockers_excludes_hand_label_protected(
    factory_dialog: YOLOFactoryDialog,
) -> None:
    """Korumalı kareler engel sayılmaz; yalnızca korumasız bekleyenler sayılır."""
    dlg = factory_dialog
    dlg._review_items = [
        _protected_item("el_1"),
        _protected_item("el_2"),
        _plain_pending_item("bekleyen_1"),
        _done_item("onayli_1"),
    ]
    assert dlg._pending_count() == 3  # iki korumalı + bir korumasız bekleyen
    assert dlg._pending_train_blockers() == 1  # yalnızca korumasız bekleyen


# --- Dondurma kapısı ---------------------------------------------------------

def test_freeze_gate_ignores_hand_label_protected_frames(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Yalnızca el etiketli korumalı kareler beklerken dondurma RED vermez."""
    dlg = factory_dialog
    _set_approved_mode(dlg, False)  # ham veri seti yolu
    _stub_ready_qa(monkeypatch)
    dlg._review_items = [_protected_item("el_1"), _protected_item("el_2")]
    assert dlg._pending_train_blockers() == 0

    dlg._freeze_dataset()

    assert dlg._pipeline["frozen"] is True
    assert (tmp_path / "frozen_manifest.json").is_file()
    assert "[DONDURMA] RED" not in _log_text(dlg)


def test_freeze_gate_still_blocks_unprotected_pending(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Korumasız bekleyen kare varken dondurma RED verir, dondurma olmaz."""
    dlg = factory_dialog
    _set_approved_mode(dlg, False)  # ham veri seti yolu
    _stub_ready_qa(monkeypatch)
    dlg._review_items = [_protected_item("el_1"), _plain_pending_item("bekleyen_1")]
    assert dlg._pending_train_blockers() == 1

    dlg._freeze_dataset()

    assert dlg._pipeline["frozen"] is False
    assert not (tmp_path / "frozen_manifest.json").is_file()
    log = _log_text(dlg)
    assert "[DONDURMA] RED" in log
    assert "1 bekleyen" in log


# --- Eğitim kapısı: onaylı mod ----------------------------------------------

def test_approved_mode_train_blocked_while_pending(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Onaylı modda yaml olsa bile bekleyen kare varken eğitim başlamaz."""
    dlg = factory_dialog
    (tmp_path / "approved" / "data_approved.yaml").write_text(
        "path: .\ntrain: approved/train/images\nval: approved/train/images\n",
        encoding="utf-8",
    )
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_plain_pending_item("bekleyen_1")]
    calls = _stub_train_start(monkeypatch, dlg)

    dlg.on_train_rtx_clicked(only_approved=True)

    assert calls == []  # worker başlatılmadı
    assert dlg._training_worker is None
    log = _log_text(dlg)
    assert "[EĞİTİM] RED" in log
    assert "1 bekleyen" in log


def test_approved_mode_train_starts_when_no_blockers(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Onaylı modda engel yokken (yalnızca korumalı bekleyen varken) eğitim başlar."""
    dlg = factory_dialog
    (tmp_path / "approved" / "data_approved.yaml").write_text(
        "path: .\ntrain: approved/train/images\nval: approved/train/images\n",
        encoding="utf-8",
    )
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_protected_item("el_1")]
    calls = _stub_train_start(monkeypatch, dlg)

    dlg.on_train_rtx_clicked(only_approved=True)

    assert len(calls) == 1
    assert calls[0].get("only_approved") is True


def test_yellow_mode_train_starts_when_no_blockers(
    factory_dialog: YOLOFactoryDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sarı modda engel yokken eğitim başlar (yalnızca korumalı bekleyen sayılmaz)."""
    dlg = factory_dialog
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_protected_item("el_1"), _done_item("onayli_1")]
    calls = _stub_train_start(monkeypatch, dlg)

    dlg.on_train_rtx_clicked(only_approved=False)

    assert len(calls) == 1
    assert calls[0].get("only_approved") is False


# --- Kilit + özet satırı tutarlılığı -----------------------------------------

def test_train_lock_and_summary_match_gate(
    factory_dialog: YOLOFactoryDialog,
) -> None:
    """Engel varken iki buton da kilitli + özet engeli yazar; engel kalkınca açılır."""
    dlg = factory_dialog
    dlg._pipeline["frozen"] = True
    dlg._review_items = [_plain_pending_item("bekleyen_1")]

    _set_approved_mode(dlg, False)
    dlg._update_train_lock()
    assert dlg.btn_train_rtx.isEnabled() is False
    assert dlg.btn_train_only_approved.isEnabled() is False
    assert "Sarı buton şu an başlamaz: 1 kare" in dlg.lbl_train_summary.text()

    _set_approved_mode(dlg, True)
    dlg._update_train_lock()
    assert dlg.btn_train_rtx.isEnabled() is False
    assert dlg.btn_train_only_approved.isEnabled() is False
    assert "Yeşil buton şu an başlamaz: 1 kare" in dlg.lbl_train_summary.text()

    # Engel kalkınca (yalnızca korumalı bekleyen kaldı) butonlar açılır.
    dlg._review_items = [_protected_item("el_1")]
    dlg._update_train_lock()
    assert dlg.btn_train_rtx.isEnabled() is True
    assert dlg.btn_train_only_approved.isEnabled() is True
    assert "başlamaz" not in dlg.lbl_train_summary.text()
