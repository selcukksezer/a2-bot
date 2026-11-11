# -*- coding: utf-8 -*-
"""
tests/test_p2_19_bulk_purge_cross_exclusion.py

P2-19 (bölüm 4.2 / F019): toplu onay <-> zehir temizliği (purge)
karşılıklı dışlama.

v16'da yoktu: iki yazıcı aynı review JSON/TXT'yi yazıyordu — onay
worker'ı `_persist`, purge `clean_poison_labels_ai`. P2-19 her iki
giriş kapısına da karşı bayrak kontrolü ekledi:

- `_bulk_approve` / `_bulk_approve_sync`: `_purge_running` True ise
  başlamaz, "[ONAY] Zehir temizliği sürüyor" yazar.
- `_purge_poisoned_labels`: `_bulk_approve_running` True ise başlamaz,
  "[TEMİZ] Toplu onay sürüyor" yazar.

Bu testler P3-30'da yazılamamıştı; burada yazıldı. Yalnızca tests/
değişir — üretim koduna dokunulmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import YOLOFactoryDialog


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Initialized YOLOFactoryDialog in Step 2 with test dataset directory.

    Pattern copied from tests/test_yolo_factory_shortcuts_and_bulk.py.
    """
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dlg._models_dir = lambda: models_dir  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    dlg._go_step(2)
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    return dlg


def _collect_logs(dlg: YOLOFactoryDialog) -> List[str]:
    """Capture _append_log output into a list and return it."""
    logs: List[str] = []
    dlg._append_log = lambda msg: logs.append(str(msg))  # type: ignore[method-assign]
    return logs


def test_bulk_approve_blocked_while_purge_running(factory_dialog: YOLOFactoryDialog) -> None:
    """Purge sürerken toplu onay (giriş kapısı) başlamaz."""
    dlg = factory_dialog
    logs = _collect_logs(dlg)
    dlg._purge_running = True
    try:
        with (
            patch.object(dlg, "_bulk_approve_sync") as sync_spy,
            patch.object(dlg, "_start_bulk_approve_async") as async_spy,
        ):
            dlg._bulk_approve()
        sync_spy.assert_not_called()
        async_spy.assert_not_called()
        assert dlg._bulk_approve_running is False
        assert any("Zehir temizliği sürüyor" in line for line in logs), logs
    finally:
        dlg._purge_running = False


def test_bulk_approve_sync_blocked_while_purge_running(factory_dialog: YOLOFactoryDialog) -> None:
    """Purge sürerken senkron toplu onay gövdesi de çalışmaz.

    `_bulk_approve_sync` giriş kapısı `_purge_running` görürse
    `_bulk_approve_sync_inner` hiç çağrılmaz ve bayrak kurulmaz.
    """
    dlg = factory_dialog
    logs = _collect_logs(dlg)
    dlg._purge_running = True
    try:
        with patch.object(dlg, "_bulk_approve_sync_inner") as inner_spy:
            dlg._bulk_approve_sync([0, 1], 0.90)
        inner_spy.assert_not_called()
        assert dlg._bulk_approve_running is False
        assert any("Zehir temizliği sürüyor" in line for line in logs), logs
    finally:
        dlg._purge_running = False


def test_purge_blocked_while_bulk_approve_running(factory_dialog: YOLOFactoryDialog) -> None:
    """Toplu onay sürerken zehir temizliği başlamaz (ters yön)."""
    dlg = factory_dialog
    logs = _collect_logs(dlg)
    dlg._bulk_approve_running = True
    try:
        dlg._purge_poisoned_labels()
        assert dlg._purge_running is False
        assert dlg._purge_thread is None
        assert dlg._purge_worker is None
        assert any("Toplu onay sürüyor" in line for line in logs), logs
    finally:
        dlg._bulk_approve_running = False


def test_bulk_approve_passes_guard_when_idle(factory_dialog: YOLOFactoryDialog) -> None:
    """Negatif kontrol: bayraklar kapalıyken toplu onay kapıdan geçer.

    Boş kuyrukta kapı geçilir, "Filtrede onaylanacak kare yok" ile
    durur — engel bayrak değil, boş filtre olur.
    """
    dlg = factory_dialog
    logs = _collect_logs(dlg)
    assert dlg._bulk_approve_running is False
    assert dlg._purge_running is False
    dlg._bulk_approve()
    assert dlg._bulk_approve_running is False  # iş bitince bayrak kalkar
    assert any("Filtrede onaylanacak kare yok" in line for line in logs), logs


def test_purge_passes_guard_when_idle(factory_dialog: YOLOFactoryDialog) -> None:
    """Negatif kontrol: bayraklar kapalıyken purge kapıdan geçer.

    Worker mock'lanır — gerçek temizlik çalıştırılmaz; kapı geçilip
    `_purge_running` kurulduğu ve "başladı" logu doğrulandıktan sonra
    thread temizlenir.
    """
    dlg = factory_dialog
    logs = _collect_logs(dlg)
    assert dlg._bulk_approve_running is False
    assert dlg._purge_running is False
    thread = None
    try:
        with patch(
            "ui.yolo_factory_dialog._PoisonPurgeWorker",
            return_value=MagicMock(),
        ):
            dlg._purge_poisoned_labels()
        assert dlg._purge_running is True
        assert any("Zehir temizliği başladı" in line for line in logs), logs
        thread = dlg._purge_thread
        assert thread is not None
    finally:
        dlg._purge_running = False
        dlg._purge_thread = None
        dlg._purge_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
