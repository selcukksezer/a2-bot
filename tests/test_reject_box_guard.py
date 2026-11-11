# -*- coding: utf-8 -*-
"""
tests/test_reject_box_guard.py

P0-3 (K-11) hedefli testleri — `_reject_box` el etiketi guard'ı +
tekli kare reddinde silme başarısı kontrolü (S-4401).

- Korumalı (TXT el etiketi) kutu reddedilemez: bellekten çıkmaz,
  negatif örnek öğretilmez, diske yazılmaz, dosya durur.
- Korumasız kutunun reddi aynen çalışır (karışık karede bile:
  trusted'da el etiketi varken weak model kutusu reddedilebilir).
- Disk kanıtı (TXT-only kare, bayraksız kutu) da reddi engeller.
- Tekli kare reddinde 0 dosya silindiyse kare kuyrukta kalır,
  `reviewed` artmaz, "[RED] diskten silindi" başarı logu basılmaz.
- Kutu silme diske yazılamazsa kutu belleğe geri alınır (S-1502).

Bu dosya yalnızca yazılır — KOŞTURULMAZ (görev kuralı: pytest yok).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Headless QApplication instance
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import HAND_LABEL_TXT_SOURCE, YOLOFactoryDialog


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Fixture providing an initialized YOLOFactoryDialog in Step 2 with temporary dataset layout."""
    import dataset_factory.approved_pool as ap_mod
    from dataset_factory.approved_pool import ApprovedDatasetPool

    prev_pool = ap_mod._GLOBAL_POOL
    test_pool = ApprovedDatasetPool(pool_root=tmp_path / "approved", project_root=tmp_path)
    ap_mod._GLOBAL_POOL = test_pool

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "labels").mkdir(parents=True, exist_ok=True)
    dlg._go_step(2)
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    yield dlg
    ap_mod._GLOBAL_POOL = prev_pool


def _txt_hand_box() -> Dict[str, Any]:
    """TXT kökenli el etiketi kutusu (tarama damgası)."""
    return {
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
        "hand_label": True,
        "source": HAND_LABEL_TXT_SOURCE,
    }


def _plain_weak_box() -> Dict[str, Any]:
    """Korumasız model (weak) kutusu."""
    return {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.4,
        "cy": 0.4,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.6,
        "conf": 0.6,
        "trusted": False,
        "weak": True,
    }


def _plain_trusted_box() -> Dict[str, Any]:
    """Bayraksız (el etiketi olmayan) trusted kutusu."""
    return {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.4,
        "cy": 0.4,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.6,
        "conf": 0.6,
        "trusted": True,
        "weak": False,
    }


def _make_item(
    tmp_path: Path,
    stem: str,
    trusted: List[Dict[str, Any]] | None = None,
    weak: List[Dict[str, Any]] | None = None,
    *,
    write_image: bool = True,
    write_review: bool = True,
    txt_lines: List[str] | None = None,
) -> Dict[str, Any]:
    """Diskte (istenirse) görsel + labels TXT + review JSON oluşturur, item dict döner."""
    trusted_boxes = trusted if trusted is not None else []
    weak_boxes = weak if weak is not None else []

    if write_image:
        import cv2
        import numpy as np

        img_path = tmp_path / "images" / "train" / f"{stem}.png"
        dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_bgr)

    if txt_lines is not None:
        lbl_path = tmp_path / "labels" / "train" / f"{stem}.txt"
        lbl_path.write_text("".join(txt_lines), encoding="utf-8")

    json_path = tmp_path / "review" / "train" / f"{stem}.json"
    if write_review:
        json_path.write_text(
            json.dumps(
                {
                    "stem": stem,
                    "split": "train",
                    "trusted": trusted_boxes,
                    "weak": weak_boxes,
                    "done": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "stem": stem,
        "split": "train",
        "weak": weak_boxes,
        "trusted": trusted_boxes,
        "done": None,
        "path": str(json_path),
    }


def _logged(log_mock: MagicMock, needle: str) -> bool:
    """_append_log mock'unda `needle` içeren çağrı var mı?"""
    for call in log_mock.call_args_list:
        args = call.args or ()
        if args and needle in str(args[0]):
            return True
    return False


# ============================================================================
# P0-3 madde 1: korumalı kutu reddedilemez
# ============================================================================

def test_reject_txt_hand_box_is_blocked(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """TXT el etiketi kutusu reddedilemez: kutu durur, dosya durur, öğretim/persist yok."""
    dlg = factory_dialog
    hand = _txt_hand_box()
    item = _make_item(
        tmp_path, "guard_hand_frame", trusted=[hand], txt_lines=["0 0.5 0.5 0.2 0.2\n"]
    )
    dlg._review_items = [item]
    dlg._review_index = 0

    lbl_file = tmp_path / "labels" / "train" / "guard_hand_frame.txt"
    rev_file = tmp_path / "review" / "train" / "guard_hand_frame.json"
    txt_before = lbl_file.read_text(encoding="utf-8")
    rev_before = rev_file.read_text(encoding="utf-8")

    with (
        patch.object(dlg, "_learn_box_exemplar") as mock_learn,
        patch.object(dlg, "_persist_review_item", return_value=True) as mock_persist,
        patch.object(dlg, "_append_log") as mock_log,
    ):
        # P2-23: _reject_box bool döner — korumalı kutuda False.
        assert dlg._reject_box(dict(hand), from_trusted=True) is False

    # 1. Kutu bellekten çıkmadı
    assert len(item["trusted"]) == 1
    assert item["trusted"][0].get("hand_label") is True
    # 2. Negatif örnek öğretilmedi, diske yazılmadı
    mock_learn.assert_not_called()
    mock_persist.assert_not_called()
    # 3. Diskteki dosya aynen duruyor
    assert lbl_file.read_text(encoding="utf-8") == txt_before
    assert rev_file.read_text(encoding="utf-8") == rev_before
    # 4. Kullanıcıya görünür koruma mesajı basıldı
    assert _logged(mock_log, "[KORUMA]")


def test_reject_box_in_txt_only_frame_blocked_by_disk_evidence(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Bayraksız kutu bile TXT-only karede reddedilemez (disk kanıtı: TXT var, review JSON yok)."""
    dlg = factory_dialog
    plain = _plain_trusted_box()
    # Review JSON YAZILMAZ — yalnızca labels TXT var.
    item = _make_item(
        tmp_path,
        "guard_txt_only_frame",
        trusted=[plain],
        write_review=False,
        txt_lines=["3 0.4 0.4 0.15 0.15\n"],
    )
    dlg._review_items = [item]
    dlg._review_index = 0

    with (
        patch.object(dlg, "_learn_box_exemplar") as mock_learn,
        patch.object(dlg, "_persist_review_item", return_value=True) as mock_persist,
        patch.object(dlg, "_append_log") as mock_log,
    ):
        # P2-23: _reject_box bool döner — disk kanıtıyla engellenen red False.
        assert dlg._reject_box(dict(plain), from_trusted=True) is False

    assert len(item["trusted"]) == 1
    mock_learn.assert_not_called()
    mock_persist.assert_not_called()
    assert _logged(mock_log, "[KORUMA]")
    assert "3 0.4 0.4 0.15 0.15" in (tmp_path / "labels" / "train" / "guard_txt_only_frame.txt").read_text(
        encoding="utf-8"
    )


# ============================================================================
# P0-3 madde 4: korumasız kutuda red aynen çalışır
# ============================================================================

def test_reject_unprotected_weak_box_still_deletes(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Korumasız weak kutu reddedilir: bellekten çıkar, TXT güncellenir."""
    dlg = factory_dialog
    weak_box = _plain_weak_box()
    item = _make_item(
        tmp_path, "guard_weak_frame", weak=[weak_box], txt_lines=["3 0.4 0.4 0.15 0.15\n"]
    )
    dlg._review_items = [item]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)

    with patch.object(dlg, "_learn_box_exemplar"):
        # P2-23: _reject_box bool döner — başarılı silmede True.
        assert dlg._reject_box(item["weak"][0], from_trusted=False) is True

    assert len(item["weak"]) == 0
    content_after = (tmp_path / "labels" / "train" / "guard_weak_frame.txt").read_text(encoding="utf-8")
    assert "3 0.4" not in content_after
    rev_data = json.loads(
        (tmp_path / "review" / "train" / "guard_weak_frame.json").read_text(encoding="utf-8")
    )
    assert len(rev_data.get("weak", [])) == 0


def test_reject_unprotected_weak_box_in_mixed_frame_still_deletes(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Karışık kare: trusted'da el etiketi varken weak model kutusu reddedilebilir.

    Kare-düzeyi guard bunu engellerdi — guard kutu-düzeyidir; el etiketi durur.
    """
    dlg = factory_dialog
    hand = _txt_hand_box()
    weak_box = _plain_weak_box()
    item = _make_item(
        tmp_path,
        "guard_mixed_frame",
        trusted=[hand],
        weak=[weak_box],
        txt_lines=["0 0.5 0.5 0.2 0.2\n", "3 0.4 0.4 0.15 0.15\n"],
    )
    dlg._review_items = [item]
    dlg._review_index = 0

    with patch.object(dlg, "_learn_box_exemplar"):
        # P2-23: _reject_box bool döner — başarılı silmede True.
        assert dlg._reject_box(item["weak"][0], from_trusted=False) is True

    # Weak kutu gitti, el etiketi duruyor
    assert len(item["weak"]) == 0
    assert len(item["trusted"]) == 1
    assert item["trusted"][0].get("hand_label") is True
    assert item["trusted"][0].get("source") == HAND_LABEL_TXT_SOURCE
    content_after = (tmp_path / "labels" / "train" / "guard_mixed_frame.txt").read_text(
        encoding="utf-8"
    )
    assert "0 0.5" in content_after


def test_reject_box_persist_failure_restores_box(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Kutu silme diske yazılamazsa kutu belleğe geri alınır (S-1502) — bellek/disk sapmaz."""
    dlg = factory_dialog
    plain = _plain_trusted_box()
    item = _make_item(tmp_path, "guard_persist_fail_frame", trusted=[plain])
    dlg._review_items = [item]
    dlg._review_index = 0

    with (
        patch.object(dlg, "_learn_box_exemplar"),
        patch.object(dlg, "_persist_review_item", return_value=False),
        patch.object(dlg, "_append_log") as mock_log,
    ):
        # P2-23: _reject_box bool döner — yazım başarısızsa False.
        assert dlg._reject_box(dict(plain), from_trusted=True) is False

    assert len(item["trusted"]) == 1
    assert item["trusted"][0].get("class_id") == 3
    assert _logged(mock_log, "kaydedilemedi")


# ============================================================================
# P0-3 madde 3: tekli kare reddinde silme başarısı kontrolü (S-4401)
# ============================================================================

def test_frame_reject_zero_files_keeps_frame_in_queue(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """0 dosya silindiyse kare kuyrukta kalır, `reviewed` artmaz, başarı logu basılmaz."""
    dlg = factory_dialog
    # Diskte HİÇ dosyası olmayan hayalet kare.
    item = _make_item(
        tmp_path,
        "guard_ghost_frame",
        trusted=[_plain_trusted_box()],
        write_image=False,
        write_review=False,
        txt_lines=None,
    )
    dlg._review_items = [item]
    dlg._review_index = 0
    dlg._pipeline["reviewed"] = 3

    with patch.object(dlg, "_append_log") as mock_log:
        dlg._review_done("no")

    assert len(dlg._review_items) == 1
    assert dlg._review_items[0].get("stem") == "guard_ghost_frame"
    assert not dlg._review_items[0].get("done")
    assert int(dlg._pipeline.get("reviewed", 0)) == 3
    assert not _logged(mock_log, "diskten silindi")
    assert _logged(mock_log, "0 dosya")


def test_frame_reject_success_removes_frame_and_counts_reviewed(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Pozitif kontrol: dosyaları gerçekten silinen kare kuyruktan çıkar, reviewed +1 olur."""
    dlg = factory_dialog
    item_a = _make_item(
        tmp_path,
        "guard_reject_ok_a",
        trusted=[_plain_trusted_box()],
        txt_lines=["3 0.4 0.4 0.15 0.15\n"],
    )
    item_b = _make_item(
        tmp_path,
        "guard_reject_ok_b",
        trusted=[_plain_trusted_box()],
        txt_lines=["3 0.4 0.4 0.15 0.15\n"],
    )
    dlg._review_items = [item_a, item_b]
    dlg._review_index = 0
    dlg._pipeline["reviewed"] = 5

    with patch.object(dlg, "_append_log") as mock_log:
        dlg._review_done("no")

    assert [x.get("stem") for x in dlg._review_items] == ["guard_reject_ok_b"]
    assert int(dlg._pipeline.get("reviewed", 0)) == 6
    assert _logged(mock_log, "[RED]")
    assert _logged(mock_log, "diskten silindi")
    assert not (tmp_path / "images" / "train" / "guard_reject_ok_a.png").exists()
    assert not (tmp_path / "labels" / "train" / "guard_reject_ok_a.txt").exists()
    assert not (tmp_path / "review" / "train" / "guard_reject_ok_a.json").exists()
