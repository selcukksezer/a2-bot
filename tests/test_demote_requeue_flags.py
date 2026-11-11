# -*- coding: utf-8 -*-
"""P0-8 — Demote/requeue bayrak koruması (ultra tarama K-12, K-14).

Kapsam (fix/p0-8-demote-requeue):
- demote `hand_label`'ı değiştirmez, `demoted_by_user: True` yazar ve
  `human_approved`'u temizler;
- demote edilmiş kutu persist'te (`_normalize_box_dict`) trusted'a geri
  yazılmaz (demoted vetosu);
- requeue TXT-only karenin köken bayraklarını (`hand_label`, `source`)
  JSON'a taşır; demote edilmiş kutu requeue'da weak kalır;
- requeue edilmiş + el etiketli kare `suspicious` filtresine girmez;
- aile (aug) silmesi kardeş karenin review JSON'undaki el etiketini
  yok saymaz (`_stem_review_has_hand_boxes`);
- canvas drag yalnızca sürüklenen kutuyu terfi ettirir; kutu içine tek
  tık (sürükleme yok) weak kutuyu el etiketine terfi ettirmez.

NOT: Bu dosya yazıldı ama KOŞTURULMADI (görev kuralı: pytest yok,
yalnızca py_compile). Dialog testleri `__new__` ile hafif bare-instance
kullanır — tam dialog kurulumu (model yükleme) gerekmez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.atomic_review_sync import (
    AtomicReviewTxtSynchronizer,
    _normalize_box_dict,
)
from dataset_factory.label_trust import requeue_labeled_dataset_for_review
from ui.yolo_factory_dialog import HAND_LABEL_TXT_SOURCE, YOLOFactoryDialog, YOLOPreviewCanvas


def _hand_box(**over: Any) -> Dict[str, Any]:
    box: Dict[str, Any] = {
        "class_id": 0,
        "class_name": "herb_aria",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.9,
        "conf": 0.9,
        "trusted": True,
        "weak": False,
        "hand_label": True,
        "human_approved": True,
        "source": HAND_LABEL_TXT_SOURCE,
    }
    box.update(over)
    return box


def _bare_dialog() -> YOLOFactoryDialog:
    """__init__ çalıştırmadan dialog: yalnızca P0-8 metotları + stub'lar."""
    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg._review_items = []
    dlg._review_index = 0
    dlg._queue_filter_mode = "all"
    # P0-8 fixup: _filtered_review_indices, __init__ görmemiş bare
    # QWidget üzerinde hasattr(self, "bulk_range") denediğinde PyQt
    # RuntimeError fırlatır (C++ nesnesi yok) — instance-dict stub'ları
    # aramayı dict'te bitirir, C++ tarafına düşmez. _queue_min_conf de
    # aynı sebeple getattr(..., 0.0) varsayılanına düşemez, dict'te olur.
    dlg.bulk_range = SimpleNamespace(value=lambda: 90)  # type: ignore[attr-defined]
    dlg.review_class_filter = SimpleNamespace(currentData=lambda: -1)  # type: ignore[attr-defined]
    dlg.review_search_edit = SimpleNamespace(text=lambda: "")  # type: ignore[attr-defined]
    dlg._queue_min_conf = 0.0
    dlg._persist_review_item = lambda it: True  # type: ignore[method-assign]
    dlg._mark_model_stale = lambda *a, **k: None  # type: ignore[method-assign]
    dlg._sync_pending_from_queue = lambda: None  # type: ignore[method-assign]
    dlg._render_queue_list = lambda: None  # type: ignore[method-assign]
    dlg._render_review_frame = lambda: None  # type: ignore[method-assign]
    dlg._append_log = lambda *a, **k: None  # type: ignore[method-assign]
    return dlg


def _write_frame(root: Path, split: str, stem: str, txt: str | None = None) -> None:
    img_dir = root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    if txt is not None:
        lbl_dir = root / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (lbl_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")


# --- 1. Demote -----------------------------------------------------------

def test_p08_01_demote_keeps_hand_label_and_writes_demoted_by_user() -> None:
    """Demote kökeni korur: hand_label True kalır, source durur,
    demoted_by_user True yazılır, human_approved temizlenir."""
    dlg = _bare_dialog()
    box = _hand_box()
    dlg._review_items = [
        {"stem": "s1", "split": "train", "trusted": [box], "weak": [],
         "done": "ok", "human_approved": True}
    ]
    dlg._demote_box_to_weak(box)
    it = dlg._review_items[0]
    assert it["trusted"] == []
    assert len(it["weak"]) == 1
    demoted = it["weak"][0]
    assert demoted["hand_label"] is True
    assert demoted["source"] == HAND_LABEL_TXT_SOURCE
    assert demoted["demoted_by_user"] is True
    assert demoted["human_approved"] is False
    assert demoted["trusted"] is False
    assert demoted["weak"] is True
    assert it["done"] is None


def test_p08_02_demoted_box_survives_persist_normalize() -> None:
    """K-12 kanıtı: persist, demote edilmiş kutuyu hand_label yüzünden
    trusted'a geri yazmaz — demoted_by_user vetosu."""
    demoted = _hand_box(trusted=False, weak=True, human_approved=False,
                        demoted_by_user=True)
    nb = _normalize_box_dict(demoted, default_trusted=False, preserve_flags=True)
    assert nb is not None
    assert nb["trusted"] is False
    assert nb["weak"] is True
    assert nb["hand_label"] is True
    assert nb["demoted_by_user"] is True


def test_p08_03_demoted_box_survives_synchronized_pair_write(tmp_path: Path) -> None:
    """write_synchronized_pair round-trip: weak listesindeki demote kutu
    JSON'da weak kalır (trusted'a dönmez)."""
    syncer = AtomicReviewTxtSynchronizer(dataset_dir=tmp_path, splits=("train",))
    demoted = _hand_box(trusted=False, weak=True, human_approved=False,
                        demoted_by_user=True)
    result = syncer.write_synchronized_pair(
        split="train", stem="s1", trusted=[], weak=[demoted],
        human_approved=False, done=None,
    )
    assert bool(getattr(result, "synced", False)) is True
    payload = json.loads((tmp_path / "review" / "train" / "s1.json").read_text(encoding="utf-8"))
    assert len(payload["weak"]) == 1
    assert payload["weak"][0]["trusted"] is False
    assert payload["weak"][0]["hand_label"] is True
    assert payload["weak"][0]["demoted_by_user"] is True


# --- 2. Requeue ----------------------------------------------------------

def test_p08_04_requeue_carries_txt_origin_flags(tmp_path: Path) -> None:
    """TXT-only kare requeue'ya GİRMEZ: K-14 damgası (hand_label) artık
    trusted vermez — korumalı kare kare düzeyinde atlanır, review JSON
    yazılmaz, eski TXT aynen kalır (§7 P0-8 requeue kararı: el etiketi /
    tekli insan onaylı kutu içeren kare asla kuyruğa geri dönmez)."""
    _write_frame(tmp_path, "train", "txt_only", txt="0 0.5 0.5 0.2 0.2\n")
    stats = requeue_labeled_dataset_for_review(tmp_path)
    assert stats["ok"] is True
    assert stats["samples"] == 0
    assert stats["boxes"] == 0
    assert stats["skipped_protected"] == 1
    assert not (tmp_path / "review" / "train" / "txt_only.json").is_file()
    assert (tmp_path / "labels" / "train" / "txt_only.txt").read_text(
        encoding="utf-8"
    ) == "0 0.5 0.5 0.2 0.2\n"


def test_p08_05_requeue_keeps_demoted_box_weak(tmp_path: Path) -> None:
    """Demote edilmiş el kutusu requeue'da weak kalır, bayrakları taşınır."""
    _write_frame(tmp_path, "train", "demoted_frame")
    review_dir = tmp_path / "review" / "train"
    review_dir.mkdir(parents=True, exist_ok=True)
    demoted = _hand_box(cx=0.3, cy=0.3, trusted=False, weak=True,
                        human_approved=False, demoted_by_user=True,
                        confidence=0.58, conf=0.58)
    (review_dir / "demoted_frame.json").write_text(json.dumps(
        {"stem": "demoted_frame", "split": "train", "trusted": [],
         "weak": [demoted], "human_approved": False, "done": None}
    ), encoding="utf-8")
    requeue_labeled_dataset_for_review(tmp_path)
    payload = json.loads(
        (review_dir / "demoted_frame.json").read_text(encoding="utf-8")
    )
    assert len(payload["trusted"]) == 0
    assert len(payload["weak"]) == 1
    box = payload["weak"][0]
    assert box["demoted_by_user"] is True
    assert box["hand_label"] is True
    assert box["trusted"] is False


# --- 3. Suspicious filtresi + aile koruması ------------------------------

def test_p08_06_requeued_hand_frame_excluded_from_suspicious() -> None:
    """Requeue edilmiş + el etiketli kare (trusted boş) suspicious'e
    girmez; normal şüpheli kare girer (K-14)."""
    dlg = _bare_dialog()
    requeued_hand = {
        "stem": "rq_hand", "split": "train", "requeued": True,
        "weak": [_hand_box(trusted=False, weak=True, human_approved=False)],
        "trusted": [], "done": None,
    }
    plain_suspicious = {
        "stem": "plain", "split": "train", "requeued": False,
        "weak": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
                  "confidence": 0.5, "trusted": False, "weak": True}],
        "trusted": [], "done": None,
    }
    dlg._review_items = [requeued_hand, plain_suspicious]
    dlg._queue_filter_mode = "suspicious"
    indices = dlg._filtered_review_indices()
    assert 0 not in indices
    assert 1 in indices


def test_p08_07_family_guard_sees_hand_boxes_in_sibling_review_json(tmp_path: Path) -> None:
    """Aile silmesi: kardeş karenin review JSON'undaki el etiketi,
    TXT-only kontrolünden geçemese de korumayı tetikler."""
    review_dir = tmp_path / "review" / "train"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "fam_aug1.json").write_text(json.dumps(
        {"stem": "fam_aug1", "trusted": [_hand_box()], "weak": []}
    ), encoding="utf-8")
    assert YOLOFactoryDialog._stem_review_has_hand_boxes(tmp_path, "fam_aug1") is True
    (review_dir / "plain.json").write_text(json.dumps(
        {"stem": "plain", "trusted": [], "weak": [{"class_id": 1, "cx": 0.5,
         "cy": 0.5, "w": 0.2, "h": 0.2}]}
    ), encoding="utf-8")
    assert YOLOFactoryDialog._stem_review_has_hand_boxes(tmp_path, "plain") is False


# --- 4. Canvas drag ------------------------------------------------------

def test_p08_08_drag_promotes_only_dragged_box() -> None:
    """Tek kutu sürüklenince yalnızca o kutu trusted olur; diğer weak ve
    trusted kutular üyeliğini korur (D046-K1)."""
    dlg = _bare_dialog()
    dragged = {"class_id": 0, "class_name": "herb_aria", "cx": 0.2, "cy": 0.2,
               "w": 0.2, "h": 0.2, "confidence": 0.5, "trusted": False, "weak": True}
    other_weak = {"class_id": 1, "class_name": "herb_mela", "cx": 0.7, "cy": 0.7,
                  "w": 0.2, "h": 0.2, "confidence": 0.5, "trusted": False, "weak": True}
    trusted = {"class_id": 2, "class_name": "ore_orichalcum", "cx": 0.5, "cy": 0.5,
               "w": 0.2, "h": 0.2, "confidence": 0.9, "trusted": True, "weak": False,
               "hand_label": True}
    it = {"stem": "s1", "split": "train", "trusted": [trusted],
          "weak": [dragged, other_weak], "done": None}
    dlg._review_items = [it]
    dlg.review_preview = SimpleNamespace(
        _annotations=[dragged, other_weak, trusted])  # type: ignore[attr-defined]
    dlg._on_box_edited(dragged)
    assert dragged in it["trusted"]
    assert dragged["hand_label"] is True
    assert it["weak"] == [other_weak]
    assert {id(b) for b in it["trusted"]} == {id(trusted), id(dragged)}
    assert other_weak["trusted"] is False
    assert other_weak["weak"] is True


def _release_event(x: float, y: float) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_p08_09_click_without_drag_does_not_promote_weak_box() -> None:
    """Kutu içine tek tık (sürükleme yok): boxEdited emit edilmez, weak
    kutu el etiketine terfi etmez (D009)."""
    canvas = YOLOPreviewCanvas()
    box = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
           "confidence": 0.5, "trusted": False, "weak": True}
    canvas._annotations = [box]
    canvas._selected_box_idx = 0
    canvas._interaction_mode = "moving"
    canvas._drag_start_pos = QPointF(100.0, 100.0)
    canvas._last_drag_distance = 0.0
    canvas._is_dragging_box = False
    emitted: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(emitted.append)
    canvas.mouseReleaseEvent(_release_event(100.0, 100.0))
    assert emitted == []
    assert box["trusted"] is False
    assert box.get("hand_label") is not True


def test_p08_10_real_drag_promotes_dragged_box() -> None:
    """Gerçek sürükleme (≥8 px): boxEdited emit edilir, kutu terfi eder."""
    canvas = YOLOPreviewCanvas()
    box = {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
           "confidence": 0.5, "trusted": False, "weak": True}
    canvas._annotations = [box]
    canvas._selected_box_idx = 0
    canvas._interaction_mode = "moving"
    canvas._drag_start_pos = QPointF(100.0, 100.0)
    canvas._last_drag_distance = 50.0
    canvas._is_dragging_box = False
    emitted: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(emitted.append)
    canvas.mouseReleaseEvent(_release_event(150.0, 100.0))
    assert len(emitted) == 1
    assert box["trusted"] is True
    assert box["hand_label"] is True
