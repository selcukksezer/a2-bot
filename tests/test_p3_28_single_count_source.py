# -*- coding: utf-8 -*-
"""tests/test_p3_28_single_count_source.py
==========================================
P3-28 (tek sayım kaynağı): ``count_review_state()`` testleri.

Kapı, dashboard çubuğu, web önizleme (``export_web_state``) ve log artık
sayıları tek fonksiyondan alır: ``YOLOFactoryDialog.count_review_state()``
``{pending, reviewed, total, blockers}`` döner. Tanımlar P2-21'deki
gibidir, DEĞİŞMEDİ:

- ``pending``:  ``_pending_count()`` — yüklü kuyrukta ``done`` olmayanlar.
- ``reviewed``: ``_pipeline["reviewed"]`` — oturumda incelenen (onaylanan +
  reddedilen) KÜMÜLATİF sayı; finalize onaylıları listeden çıkarsa da
  sıfırlanmaz (F043-K1).
- ``total``:    ``reviewed + pending`` — çubuk ve web ``reviewProgTxt``
  paydası.
- ``blockers``: ``_pending_train_blockers()`` — pending'den el etiketli
  korumalı kareler çıkarılmış KAPI sayısı (P0-4).

Bu testler üç şeyi kilitler:
1. Bar ≡ web eşitliği: çubuk metni/yüzdesi ile ``export_web_state``
   ``reviewProgTxt`` / ``reviewProgPct`` aynı ``count_review_state()``
   değerlerinden gelir (P2-21'deki "çubuk 0/801, web 379/1180" ayrışması
   bir daha olamaz).
2. Kapı sayısı: ``blockers`` değeri ``_pending_train_blockers()`` ile
   birebir aynıdır; korumalı kareler pending'de sayılır, kapıda sayılmaz.
3. Reviewed kümülatifliği: onaylı kareler listeden çıkınca ``reviewed``
   düşmez, ``total`` paydası korunur.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

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
    """Geçici veri seti dizinli gerçek dialog (test_train_gate_delta ile aynı)."""
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
    """Onaylanmış kare."""
    return {
        "stem": stem,
        "split": "train",
        "weak": [],
        "trusted": [{"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        "done": "ok",
    }


def _set_queue(dlg: YOLOFactoryDialog, items, reviewed: int) -> None:
    dlg._review_items = list(items)
    dlg._review_index = 0
    dlg._pipeline["reviewed"] = int(reviewed)


# --- count_review_state: tek kaynak, dört alan -------------------------------

def test_p328_count_matches_primitives(factory_dialog: YOLOFactoryDialog):
    """Dört alan da sayım ilkelleriyle birebir aynıdır."""
    dlg = factory_dialog
    _set_queue(
        dlg,
        [_plain_pending_item("a"), _protected_item("b"), _done_item("c"), _plain_pending_item("d")],
        reviewed=5,
    )
    state = dlg.count_review_state()
    assert state == {
        "pending": dlg._pending_count(),
        "reviewed": 5,
        "total": 5 + dlg._pending_count(),
        "blockers": dlg._pending_train_blockers(),
    }
    assert state["pending"] == 3  # a, b, d bekliyor; c onaylı
    assert state["total"] == 8
    assert state["blockers"] == 2  # b korumalı, kapıya girmez


def test_p328_count_refreshes_pipeline_pending_cache(factory_dialog: YOLOFactoryDialog):
    """Tek yazıcı: ``_pipeline["pending"]`` bayat kalmaz."""
    dlg = factory_dialog
    _set_queue(dlg, [_plain_pending_item("a"), _plain_pending_item("b")], reviewed=0)
    dlg._pipeline["pending"] = 999  # bayat önbellek
    state = dlg.count_review_state()
    assert state["pending"] == 2
    assert dlg._pipeline["pending"] == 2


def test_p328_count_empty_queue(factory_dialog: YOLOFactoryDialog):
    dlg = factory_dialog
    _set_queue(dlg, [], reviewed=0)
    assert dlg.count_review_state() == {
        "pending": 0,
        "reviewed": 0,
        "total": 0,
        "blockers": 0,
    }


def test_p328_count_only_protected_pending(factory_dialog: YOLOFactoryDialog):
    """Yalnızca korumalı kare bekliyorsa: pending > 0 ama kapı 0."""
    dlg = factory_dialog
    _set_queue(dlg, [_protected_item("a"), _protected_item("b")], reviewed=7)
    state = dlg.count_review_state()
    assert state["pending"] == 2
    assert state["blockers"] == 0
    assert state["total"] == 9


# --- Kapı blockers değeri -----------------------------------------------------

def test_p328_blockers_excludes_only_protected(factory_dialog: YOLOFactoryDialog):
    """Kapı sayısı = bekleyen − korumalı; korumalı olmayan her bekleyen engeldir."""
    dlg = factory_dialog
    items = [
        _plain_pending_item("a"),
        _protected_item("b"),
        _plain_pending_item("c"),
        _done_item("d"),
        _protected_item("e"),
    ]
    _set_queue(dlg, items, reviewed=0)
    assert dlg.count_review_state()["blockers"] == 2  # a, c
    assert dlg._pending_train_blockers() == 2


# --- Reviewed kümülatifliği ---------------------------------------------------

def test_p328_reviewed_survives_finalize_removal(factory_dialog: YOLOFactoryDialog):
    """P2-21/F043-K1: finalize onaylıları listeden çıkarsa da ``reviewed``
    sıfırlanmaz; payda ``reviewed + pending`` korunur."""
    dlg = factory_dialog
    _set_queue(
        dlg,
        [_done_item("a"), _done_item("b"), _plain_pending_item("c")],
        reviewed=379,
    )
    before = dlg.count_review_state()
    assert before == {"pending": 1, "reviewed": 379, "total": 380, "blockers": 1}

    # Finalize: onaylı kareler listeden çıkar (kuyrukta yalnız bekleyen kalır).
    dlg._review_items = [it for it in dlg._review_items if not it.get("done")]
    after = dlg.count_review_state()
    assert after["reviewed"] == 379  # kümülatif — düşmedi
    assert after["pending"] == 1
    assert after["total"] == 380  # payda korunur: 379 + 1


def test_p328_reviewed_counts_rejected_too(factory_dialog: YOLOFactoryDialog):
    """Reddedilen kare de incelenmiştir: reviewed onay + red toplamıdır."""
    dlg = factory_dialog
    _set_queue(dlg, [_plain_pending_item("a")], reviewed=1180)
    state = dlg.count_review_state()
    assert state["reviewed"] == 1180
    assert state["total"] == 1181


# --- Bar ≡ web eşitliği -------------------------------------------------------

def test_p328_bar_equals_web(factory_dialog: YOLOFactoryDialog):
    """Çubuk metni/yüzdesi ile web ``reviewProgTxt`` / ``reviewProgPct``
    aynı ``count_review_state()`` değerlerinden gelir."""
    dlg = factory_dialog
    _set_queue(
        dlg,
        [_plain_pending_item("a"), _protected_item("b"), _done_item("c")],
        reviewed=379,
    )
    state = dlg.count_review_state()
    assert state == {"pending": 2, "reviewed": 379, "total": 381, "blockers": 1}

    # Çubuk (dashboard): _render_review_frame içindeki tek-kaynak okuma.
    dlg._render_review_frame()
    assert dlg.review_prog_txt.text() == "379 / 381"
    assert dlg.review_prog.value() == int(round(100 * 379 / 381))

    # Web: export_web_state aynı sayıları verir.
    web = dlg.export_web_state()
    assert web["reviewProgTxt"] == "379 / 381"
    assert web["reviewProgPct"] == int(round(100 * 379 / 381))
    assert web["reviewProgTxt"] == dlg.review_prog_txt.text()
    assert web["reviewProgPct"] == dlg.review_prog.value()


def test_p328_bar_equals_web_after_finalize(factory_dialog: YOLOFactoryDialog):
    """Finalize sonrası ayrışma (P2-21 örneği: çubuk 0/801, web 379/1180)
    bir daha olamaz: iki okuma da aynı fonksiyondadır."""
    dlg = factory_dialog
    _set_queue(dlg, [_plain_pending_item("a")], reviewed=801)
    # Onaylı kareler listeden çıktıktan sonra bile:
    dlg._review_items = [_plain_pending_item("a")]
    dlg._pipeline["reviewed"] = 801

    dlg._render_review_frame()
    web = dlg.export_web_state()
    assert dlg.review_prog_txt.text() == "801 / 802"
    assert web["reviewProgTxt"] == "801 / 802"
    assert web["reviewProgPct"] == dlg.review_prog.value()


def test_p328_bar_equals_web_empty_queue(factory_dialog: YOLOFactoryDialog):
    dlg = factory_dialog
    _set_queue(dlg, [], reviewed=0)
    dlg._render_review_frame()
    assert dlg.review_prog_txt.text() == "0 / 0"
    assert dlg.review_prog.value() == 100
    web = dlg.export_web_state()
    assert web["reviewProgTxt"] == "0 / 0"
    # NOT: boş kuyrukta yüzde ayrışır (çubuk total<=0 → 100, web
    # max(1,total) → 0) — P2-21'den beri böyle, P3-28 kapsamı dışı;
    # metin ("0 / 0") iki yerde de aynıdır.
    assert web["reviewProgPct"] == 0
