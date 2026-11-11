# -*- coding: utf-8 -*-
"""P0-7 / K-13: toplu onay + budama koruma deliği testleri.

Kapsam (ultra tarama 2026-11-10, bölüm 9 P0 madde 7 + K-13 + bölüm 3.2):

- (a) Toplu onay weak dalı da koruma kontrolünden geçer: trusted el
      kutulu + eşik üstü weak kutulu kare `done="ok"` /
      `human_approved` almaz; weak kutular işaretlenir, kare kuyrukta
      kalır ve `n`'ye girmez.
- (b) Budama fallback'i dış budamayla AYNI kanonik koruma predikatını
      (`label_trust.is_protected_box`) kullanır: `source`'unda "hand"
      geçen ama `hand_label` taşımayan conf 0.0 kutu silinmez; dış yol
      ile fallback aynı kutuda aynı kararı verir. P1-12 (K-32/K-34)
      uzlaşması: karar "silinmez"dir, "trusted olur" değil — kaynak-el
      kutu weak kalır, güveni 0.0 şişirilmez (eski trusted + 0.90
      beklentisi kaldırılan F-1212 şişirmesiydi).
- (c) Dış budama 0 kutu budayıp yalnızca terfi ettirince fallback
      bayat bellekle aynı kareyi ikinci kez işlemez: terfi bir kez
      sayılır, kare `done="ok"` almaz, kuyruk diskten yenilenir.
- (d) Kalibrasyon vetosu (malzeme çakışması) fallback tarafından geri
      alınmaz: veto edilen kutu weak kalır, güveni 0.45'e iner.

Kapsam dışı (P0-7'ye girmez): eşik kaydırıcısı, "%90+" metni.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.label_trust import prune_low_confidence_boxes_in_dataset
from ui.yolo_factory_dialog import (
    HAND_LABEL_TXT_SOURCE,
    YOLOFactoryDialog,
    _process_bulk_approve_items,
)


@pytest.fixture
def factory_dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> YOLOFactoryDialog:
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    dlg.bulk_range.setValue(85)
    # CWD bağlı yazımları (models/stale_manifest.json) tmp altına al.
    monkeypatch.chdir(tmp_path)
    # [YOLO-ASYNC] __init__'teki başlangıç yüklemesi (üretim dizini) test
    # gövdesi kuyruğu okumadan önce bitmeli; yoksa bitince kuyruğu sıfırlar.
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    return dlg


def _dummy_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


def _box(
    conf: float,
    class_id: int = 1,
    class_name: str = "herb_mela",
    **extra: Any,
) -> Dict[str, Any]:
    b: Dict[str, Any] = {
        "class_id": class_id,
        "class_name": class_name,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": conf,
        "conf": conf,
        "trusted": False,
        "weak": True,
    }
    b.update(extra)
    return b


def _hand_trusted_box(conf: float = 0.0) -> Dict[str, Any]:
    """TXT kökenli el etiketi kutusu (trusted listesinde)."""
    return {
        "class_id": 6,
        "class_name": "tree_log",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.2,
        "confidence": conf,
        "conf": conf,
        "trusted": True,
        "weak": False,
        "hand_label": True,
        "source": HAND_LABEL_TXT_SOURCE,
    }


def _review_frame(
    tmp_path: Path,
    stem: str,
    weak: List[Dict[str, Any]] | None = None,
    trusted: List[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    """Review JSON'lu kare (görsel + JSON + TXT)."""
    img = tmp_path / "images" / "train" / f"{stem}.png"
    rev = tmp_path / "review" / "train" / f"{stem}.json"
    lbl = tmp_path / "labels" / "train" / f"{stem}.txt"
    _dummy_png(img)
    rev.write_text(
        json.dumps(
            {"stem": stem, "split": "train", "weak": weak or [], "trusted": trusted or []}
        ),
        encoding="utf-8",
    )
    lbl.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return {"image": img, "review": rev, "label": lbl}


def _capture_logs(dlg: YOLOFactoryDialog) -> List[str]:
    """_append_log yazılarını kaydet (orijinal de çalışır)."""
    seen: List[str] = []
    orig = dlg._append_log

    def _rec(msg: str) -> None:
        seen.append(str(msg or ""))
        orig(msg)

    dlg._append_log = _rec  # type: ignore[method-assign]
    return seen


# ============================================================================
# (a) Toplu onay weak dalı: korumalı kare done almaz
# ============================================================================


def test_bulk_approve_weak_branch_protected_frame_gets_no_done(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """K-13: trusted el kutulu + eşik üstü weak kutulu kare toplu onayda
    `done="ok"` / `human_approved` almaz; weak kutu işaretlenir, `n`'ye girmez.
    P1-10 / K-21: toplu onay HİÇBİR kareye `human_approved` yazmaz —
    onaylanan kare `done="ok"` + `bulk_approved` alır."""
    dlg = factory_dialog
    _dummy_png(tmp_path / "images" / "train" / "prot_weak.png")
    _dummy_png(tmp_path / "images" / "train" / "model_ok.png")
    protected_item: Dict[str, Any] = {
        "stem": "prot_weak",
        "split": "train",
        "weak": [_box(0.95)],
        "trusted": [_hand_trusted_box()],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "prot_weak.json"),
    }
    model_item: Dict[str, Any] = {
        "stem": "model_ok",
        "split": "train",
        "weak": [_box(0.95)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "model_ok.json"),
    }

    result = _process_bulk_approve_items(dlg, [protected_item, model_item], 0.85, "all")

    # Yalnızca korumasız kare onay sayılır; korumalı kare işaretlenir ama
    # onaylanmaz.
    assert result["n"] == 1
    assert result["protected_weak_marked"] == 1
    assert result["bulk_marked_total"] == 2
    assert result["skipped_hand"] == 0

    # Korumalı kare: weak kutu işaretli, trusted el kutusu ayrı, done yok.
    assert protected_item.get("done") is None
    assert protected_item.get("human_approved") is not True
    assert len(protected_item["weak"]) == 1
    assert protected_item["weak"][0].get("bulk_approved") is True
    assert protected_item["weak"][0].get("trusted") is False
    assert len(protected_item["trusted"]) == 1
    assert protected_item["trusted"][0].get("hand_label") is True
    assert protected_item["trusted"][0].get("bulk_approved") is not True

    # Korumasız kontrol karesi eskisi gibi onaylanır — ama P1-10 / K-21:
    # toplu onay `human_approved` YAZMAZ, yalnızca `done="ok"` + `bulk_approved`.
    assert model_item.get("done") == "ok"
    assert model_item.get("human_approved") is not True
    assert model_item.get("bulk_approved") is True

    # Disk: korumalı kare onaylı yazılmadı.
    saved = json.loads(
        (tmp_path / "review" / "train" / "prot_weak.json").read_text(encoding="utf-8")
    )
    assert saved.get("done") is None
    assert saved.get("human_approved") is not True
    assert saved["weak"][0].get("bulk_approved") is True

    # Disk: onaylanan kare P1-10 / K-21 ile `done="ok"` + `bulk_approved`
    # yazar — `_persist` artık `done`'dan `human_approved` üretmez.
    saved_model = json.loads(
        (tmp_path / "review" / "train" / "model_ok.json").read_text(encoding="utf-8")
    )
    assert saved_model.get("done") == "ok"
    assert saved_model.get("human_approved") is not True
    assert saved_model.get("bulk_approved") is True


def test_bulk_approve_protected_frame_stays_in_queue(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Tam `_bulk_approve` yolu: korumalı kare kuyrukta kalır, log dürüsttür."""
    dlg = factory_dialog
    _dummy_png(tmp_path / "images" / "train" / "prot_weak.png")
    _dummy_png(tmp_path / "images" / "train" / "model_ok.png")
    protected_item: Dict[str, Any] = {
        "stem": "prot_weak",
        "split": "train",
        "weak": [_box(0.95)],
        "trusted": [_hand_trusted_box()],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "prot_weak.json"),
    }
    model_item: Dict[str, Any] = {
        "stem": "model_ok",
        "split": "train",
        "weak": [_box(0.95)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "model_ok.json"),
    }
    dlg._review_items = [protected_item, model_item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_approve()

    assert [it["stem"] for it in dlg._review_items] == ["prot_weak"]
    assert dlg._review_items[0].get("done") is None
    assert dlg._review_items[0].get("human_approved") is not True
    log_text = dlg.training_log_box.toPlainText()
    assert "Filtredeki 1 kare onaylandı" in log_text
    assert "1 korumalı karede yalnızca weak kutular işaretlendi" in log_text


# ============================================================================
# (b) Dış + fallback budama aynı kutuda aynı kararı verir
# ============================================================================


def test_fallback_prune_keeps_source_hand_box(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """K-13 ayrışması: `source`'unda "hand" geçen ama `hand_label`
    taşımayan conf 0.0 kutuyu fallback eskiden siliyordu; artık dış
    yolla aynı kararla korunur, negatif öğretilmez.
    P1-12 (K-32 never inflate / K-34) uzlaşması: koruma ≠ güven —
    kaynak-el kutu budanmaz ama insan da sayılmaz; kalibrasyon onu
    weak bırakır, ölçülmüş güveni 0.0 aynen kalır. Eski "trusted olur,
    güven 0.90'a çıkar" beklentisi kaldırılan ``max(0.90, ...)``
    şişirmesiydi (F-1212); P1-12 kilit testleri seed/kaynak-el kutuyu
    weak/untrusted sayar (test_dev_d54_label_trust.py)."""
    dlg = factory_dialog
    taught: List[Dict[str, Any]] = []
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: taught.append({"stem": it.get("stem"), "box": dict(box), **kw}) or True
    )
    # Yalnızca bellekte var (review JSON yok) → dış budama dokunamaz,
    # fallback çalışır. Aynı karede bir de silinecek model kutusu var.
    hand_src_box = _box(
        0.0, class_id=6, class_name="tree_log", source=HAND_LABEL_TXT_SOURCE
    )
    hand_src_box.pop("hand_label", None)
    # _box trusted/weak alanlarını weak olarak kurar; source el kökenli.
    assert "hand_label" not in hand_src_box
    item: Dict[str, Any] = {
        "stem": "frame_hand_src",
        "split": "train",
        "weak": [hand_src_box, _box(0.30)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "frame_hand_src.json"),
    }
    dlg._review_items = [item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_prune_low_confidence_boxes()

    # Model kutusu budanıp negatif öğretildi; el kökenli kutu duruyor
    # (weak, silinmedi, negatif öğretilmedi) — ama P1-12 ile trusted
    # DEĞİL: kaynak-el kutu insan sayılmaz, güveni şişirilmez (0.0 kalır).
    assert len(taught) == 1
    assert taught[0]["box"]["confidence"] == 0.30
    assert len(item["weak"]) == 1
    kept = item["weak"][0]
    assert kept["class_id"] == 6
    assert kept.get("trusted") is False
    assert float(kept.get("confidence", 0.0)) == 0.0
    assert item["trusted"] == []
    # Kutu-düzeyi korumalı kare done almaz (dış budama done'a dokunmaz).
    assert item.get("done") is None


def test_external_and_fallback_agree_protected_box_survives(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aynı el-kökenli kutu dış budamada da fallback'te de korunur
    (silinmez, negatif öğretilmez).
    P1-12 (K-32/K-34) uzlaşması: iki yolun ANLAŞTIĞI karar "silinmez"dir;
    kutu weak kalır, trusted olmaz, güveni 0.0 şişirilmez — eski
    "trusted + 0.90" beklentisi kaldırılan F-1212 şişirmesiydi."""
    monkeypatch.chdir(tmp_path)
    # (a) Dış yol: diskteki kare.
    hand_src_box = _box(
        0.0, class_id=6, class_name="tree_log", source=HAND_LABEL_TXT_SOURCE
    )
    hand_src_box.pop("hand_label", None)
    _review_frame(tmp_path, "frame_agree_ext", weak=[hand_src_box])

    res = prune_low_confidence_boxes_in_dataset(
        tmp_path, threshold=0.85, record_negative_exemplars=False
    )

    assert res["pruned_boxes"] == 0
    saved = json.loads(
        (tmp_path / "review" / "train" / "frame_agree_ext.json").read_text(encoding="utf-8")
    )
    kept_ext = saved.get("trusted", []) + saved.get("weak", [])
    assert len(kept_ext) == 1
    assert kept_ext[0]["class_id"] == 6
    # P1-12: korumalı kutu silinmedi ama trusted de olmadı — weak kalır,
    # ölçülmüş güveni 0.0 aynen durur (şişirme yok).
    assert kept_ext[0].get("trusted") is False
    assert float(kept_ext[0].get("confidence", 0.0)) == 0.0
    assert saved.get("trusted", []) == []
    assert len(saved.get("weak", [])) == 1

    # (b) Fallback: bellekteki ikiz kare (review JSON yok).
    dlg = factory_dialog
    taught: List[Dict[str, Any]] = []
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: taught.append({"stem": it.get("stem"), "box": dict(box), **kw}) or True
    )
    twin_box = _box(0.0, class_id=6, class_name="tree_log", source=HAND_LABEL_TXT_SOURCE)
    twin_box.pop("hand_label", None)
    item: Dict[str, Any] = {
        "stem": "frame_agree_fb",
        "split": "train",
        "weak": [twin_box],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "frame_agree_fb.json"),
    }
    dlg._review_items = [item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_prune_low_confidence_boxes()

    assert taught == []
    # P1-12: fallback dış yolla aynı kararı verir — kutu silinmez ama
    # weak kalır (trusted olmaz, güven 0.0 şişirilmez).
    assert len(item["weak"]) == 1
    assert item["weak"][0]["class_id"] == 6
    assert item["weak"][0].get("trusted") is False
    assert float(item["weak"][0].get("confidence", 0.0)) == 0.0
    assert item["trusted"] == []


def test_external_and_fallback_agree_low_conf_box_pruned(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aynı korumasız düşük güvenli kutu dış budamada da fallback'te de silinir."""
    monkeypatch.chdir(tmp_path)
    # (a) Dış yol: diskteki kare.
    _review_frame(tmp_path, "frame_cut_ext", weak=[_box(0.30)])

    res = prune_low_confidence_boxes_in_dataset(
        tmp_path, threshold=0.85, record_negative_exemplars=False
    )

    assert res["pruned_boxes"] == 1
    saved = json.loads(
        (tmp_path / "review" / "train" / "frame_cut_ext.json").read_text(encoding="utf-8")
    )
    assert saved.get("trusted", []) + saved.get("weak", []) == []

    # (b) Fallback: bellekteki ikiz kare (review JSON yok).
    dlg = factory_dialog
    taught: List[Dict[str, Any]] = []
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: taught.append({"stem": it.get("stem"), "box": dict(box), **kw}) or True
    )
    item: Dict[str, Any] = {
        "stem": "frame_cut_fb",
        "split": "train",
        "weak": [_box(0.30)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / "frame_cut_fb.json"),
    }
    dlg._review_items = [item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_prune_low_confidence_boxes()

    assert len(taught) == 1
    assert taught[0]["box"]["confidence"] == 0.30
    assert item["weak"] == []
    assert item["trusted"] == []


# ============================================================================
# (c) Terfi-only dış budama: fallback ikinci kez işlemez (bölüm 3.2)
# ============================================================================


def test_promote_only_external_prune_skips_fallback(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Dış budama 0 budayıp 1 terfi ettirince fallback bayat bellekle
    aynı kareyi ikinci kez işlemez: terfi bir kez sayılır, done yok."""
    dlg = factory_dialog
    files = _review_frame(tmp_path, "frame_promo_only", weak=[_box(0.95)])
    dlg._reload_review_queue()
    assert {it["stem"] for it in dlg._review_items} == {"frame_promo_only"}
    dlg.set_queue_filter_mode("all")
    logs = _capture_logs(dlg)

    dlg._bulk_prune_low_confidence_boxes()

    # Disk: kutu tam bir kez trusted oldu; done yok.
    saved = json.loads(files["review"].read_text(encoding="utf-8"))
    assert len(saved.get("trusted", [])) == 1
    assert saved.get("weak", []) == []
    assert saved.get("done") is None
    # Kuyruk diskten yenilendi: aynı tek kutu, done yok.
    assert len(dlg._review_items) == 1
    current = dlg._review_items[0]
    assert current.get("done") is None
    assert len(current.get("trusted", [])) == 1
    assert current.get("weak", []) == []
    # Sayı çift sayılmadı: "1 kutu onaylandı", "2 kutu" değil.
    done_lines = [m for m in logs if m.startswith("[BUDAMA] Tamamlandı")]
    assert len(done_lines) == 1
    assert "0 düşük güvenli kutu silindi, 1 kutu onaylandı" in done_lines[0]


# ============================================================================
# (d) Kalibrasyon vetosu fallback tarafından geri alınmaz (bölüm 3.2)
# ============================================================================


def test_vetoed_box_not_undone_by_fallback(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Malzeme çakışması vetosu: dış budama kutuyu weak tutup güveni
    0.45'e indirir; fallback bayat bellekle terfi ettirip vetoyu geri alamaz."""
    dlg = factory_dialog
    # stem "mela" içerir, kutu "targena" sınıfı → is_material_conflict True.
    veto_box = _box(0.95, class_id=2, class_name="herb_targena", source="model_detector")
    files = _review_frame(tmp_path, "frame_mela_01", weak=[veto_box])
    dlg._reload_review_queue()
    assert {it["stem"] for it in dlg._review_items} == {"frame_mela_01"}
    dlg.set_queue_filter_mode("all")
    logs = _capture_logs(dlg)

    dlg._bulk_prune_low_confidence_boxes()

    # Disk: kutu hâlâ weak, trusted değil, güven 0.45'e inmiş, done yok.
    saved = json.loads(files["review"].read_text(encoding="utf-8"))
    assert saved.get("trusted", []) == []
    assert len(saved.get("weak", [])) == 1
    assert float(saved["weak"][0].get("confidence", 1.0)) == 0.45
    assert saved["weak"][0].get("trusted") is False
    assert saved.get("done") is None
    # Kuyruk diskten yenilendi: veto kararı bellekte de duruyor.
    assert len(dlg._review_items) == 1
    current = dlg._review_items[0]
    assert current.get("done") is None
    assert current.get("trusted", []) == []
    assert len(current.get("weak", [])) == 1
    # Hiçbir terfi sayılmadı.
    done_lines = [m for m in logs if m.startswith("[BUDAMA] Tamamlandı")]
    assert len(done_lines) == 1
    assert "0 kutu onaylandı" in done_lines[0]
