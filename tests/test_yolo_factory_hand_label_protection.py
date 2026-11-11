# -*- coding: utf-8 -*-
"""El etiketi (TXT) koruma testleri — fix/yolo-prune-guvenlik.

Düzeltilmiş el etiketleri (labels/*.txt, review JSON'u yok) asla silinemez:

- (a) tarama TXT kökenli kutuları hand_label=True + trusted listesinde çıkarır
      (şüpheli/weak değil); "suspicious" filtresine girmez.
- (b) budama model tahmin kutusunu silerken TXT kökenli karenin labels
      TXT'si bayt-bayt aynen durur (fallback yolu + dış budama yolu).
- (c) korumalı karede Reddet labels TXT'sini ve görseli silmez
      (tekli red, toplu red, `_delete/_batch_delete`).
- (d) dış budama fonksiyonu korumalı kutuyu atlar.
- (e) toplu onay korumalı kareye `done="ok"` / `human_approved` vermez (B-07).

Tüm testler gerçek dosya sistemini (tmp_path) kullanır; repo
`dataset/learned_exemplars.json`'una yazılmaz (negatif örnek öğretimi
kaydediciyle değiştirilir, görseller geçersiz PNG baytıdır → cv2 frame
yoktur → dış budama exemplar yazmaz). `monkeypatch.chdir(tmp_path)` CWD
bağlı `models/stale_manifest.json` yazımını engeller.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PyQt6.QtWidgets import QApplication, QLabel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.label_trust import prune_low_confidence_boxes_in_dataset
from ui.yolo_factory_dialog import HAND_LABEL_TXT_SOURCE, YOLOFactoryDialog

TXT_STEM = "frame_txt_el"
TXT_BYTES = b"6 0.500000 0.500000 0.100000 0.200000\n3 0.250000 0.750000 0.050000 0.050000\n"
MODEL_STEM = "frame_model"


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


def _txt_frame(tmp_path: Path, stem: str = TXT_STEM, txt: bytes = TXT_BYTES) -> Dict[str, Path]:
    """Review JSON'suz, labels TXT'li kare (düzeltilmiş el etiketi)."""
    img = tmp_path / "images" / "train" / f"{stem}.png"
    lbl = tmp_path / "labels" / "train" / f"{stem}.txt"
    img.parent.mkdir(parents=True, exist_ok=True)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    _dummy_png(img)
    lbl.write_bytes(txt)
    return {"image": img, "label": lbl}


def _review_frame(
    tmp_path: Path,
    stem: str = MODEL_STEM,
    weak: List[Dict[str, Any]] | None = None,
    trusted: List[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    """Review JSON'lu model karesi (görsel + JSON + TXT).

    TXT bilerek BOŞ bırakılır: P0-1 uzlaştırması 5 alanlı her TXT
    satırını el etiketi sayar — model karesinin TXT'sinde satır olsaydı
    tarama onu trusted el etiketi yapar ve kare reddedilemez olurdu
    (eski "TXT satırı kanıtsız kalıntıdır" kuralı P0-1 ile çelişir).
    Dosya yine de oluşturulur (boş), silme iddiaları anlamlı kalsın.
    """
    img = tmp_path / "images" / "train" / f"{stem}.png"
    rev = tmp_path / "review" / "train" / f"{stem}.json"
    lbl = tmp_path / "labels" / "train" / f"{stem}.txt"
    img.parent.mkdir(parents=True, exist_ok=True)
    rev.parent.mkdir(parents=True, exist_ok=True)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    _dummy_png(img)
    rev.write_text(
        json.dumps(
            {"stem": stem, "split": "train", "weak": weak or [], "trusted": trusted or []}
        ),
        encoding="utf-8",
    )
    lbl.write_text("", encoding="utf-8")
    return {"image": img, "review": rev, "label": lbl}


def _model_weak_box(conf: float = 0.30) -> Dict[str, Any]:
    return {
        "class_id": 1,
        "class_name": "herb_mela",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": conf,
        "conf": conf,
        "trusted": False,
        "weak": True,
    }


def _scan_items(tmp_path: Path, rejected: set | None = None) -> List[Dict[str, Any]]:
    return YOLOFactoryDialog._scan_review_queue_items(tmp_path, set(rejected or set()))


def _item_by_stem(items: List[Dict[str, Any]], stem: str) -> Dict[str, Any]:
    for it in items:
        if it.get("stem") == stem:
            return it
    raise AssertionError(f"{stem} kuyrukta bulunamadı: {[i.get('stem') for i in items]}")


# ============================================================================
# (a) Tarama: TXT kökenli kutular hand_label + trusted, şüpheli değil
# ============================================================================


def test_scan_txt_frame_is_hand_label_trusted(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    dlg = factory_dialog
    _txt_frame(tmp_path)

    items = _scan_items(tmp_path)
    assert len(items) == 1
    it = items[0]
    assert it["stem"] == TXT_STEM
    assert it["weak"] == []
    assert it["n_weak"] == 0
    assert len(it["trusted"]) == 2
    for b in it["trusted"]:
        assert b["hand_label"] is True
        assert b["trusted"] is True
        assert b["weak"] is False
        assert b["source"] == HAND_LABEL_TXT_SOURCE
        # Güven ölçülmedi (0.0) — uydurulmadı; onay damgası yok (B-07).
        assert float(b.get("confidence", -1.0)) == 0.0
        assert b.get("human_approved") is not True
        assert b.get("bulk_approved") is not True
    assert it.get("done") is None
    assert it.get("human_approved") is not True

    # "Suspicious" filtresine girmez (n_weak == 0) — toplu şüpheli
    # aksiyonları bu kareyi hedefleyemez.
    dlg._review_items = items
    dlg.set_queue_filter_mode("suspicious")
    assert dlg._filtered_review_indices() == []

    # Detay satırı: "şüpheli" değil, el etiketi gösterir.
    row = dlg._create_review_det_row(dict(it["trusted"][0]), is_trusted=True)
    conf_lbl = row.findChild(object, "PipeDetConf")
    assert conf_lbl is not None
    assert conf_lbl.text() == "el"
    assert any(
        "El etiketi" in lbl.text() for lbl in row.findChildren(QLabel)
    )


def test_scan_rejected_txt_frame_files_preserved(tmp_path: Path) -> None:
    """Tarama GC yolu: reddedilmiş TXT karesinin hiçbir dosyası silinmez."""
    txt_files = _txt_frame(tmp_path)
    model_files = _review_frame(tmp_path, weak=[_model_weak_box()])

    items = _scan_items(tmp_path, rejected={TXT_STEM, MODEL_STEM})

    assert items == []
    # Korumalı kare: görsel + TXT duruyor.
    assert txt_files["image"].is_file()
    assert txt_files["label"].is_file()
    assert txt_files["label"].read_bytes() == TXT_BYTES
    # Korumasız kontrol karesi: eskisi gibi tamamen silinir.
    assert not model_files["image"].exists()
    assert not model_files["review"].exists()
    assert not model_files["label"].exists()


# ============================================================================
# (b) Budama: model kutusu silinirken TXT bayt-bayt aynen durur
# ============================================================================


def test_fallback_prune_keeps_txt_byte_identical(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Dış budama 0 kutu budarsa fallback çalışır; TXT kareye dokunulmaz."""
    dlg = factory_dialog
    txt_files = _txt_frame(tmp_path)
    taught: List[Dict[str, Any]] = []
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: taught.append({"stem": it.get("stem"), "box": dict(box), **kw}) or True
    )

    # Bellekteki kuyruk: model kare (review JSON'suz, weak 0.30) + TXT kare.
    # Dış budama diskte review JSON bulamaz → 0 budar → fallback çalışır.
    txt_item = _item_by_stem(_scan_items(tmp_path), TXT_STEM)
    model_item: Dict[str, Any] = {
        "stem": MODEL_STEM,
        "split": "train",
        "weak": [_model_weak_box(0.30)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / f"{MODEL_STEM}.json"),
    }
    dlg._review_items = [model_item, txt_item]
    dlg.set_queue_filter_mode("all")
    txt_before = txt_files["label"].read_bytes()

    dlg._bulk_prune_low_confidence_boxes()

    # Model kutusu budandı ve negatif öğretildi.
    assert len(taught) == 1
    assert taught[0]["stem"] == MODEL_STEM
    assert taught[0]["box"]["confidence"] == 0.30
    assert model_item["weak"] == []
    # TXT kare: bayt-bayt aynen, done yok, kutular aynen.
    assert txt_files["label"].read_bytes() == txt_before == TXT_BYTES
    assert not (tmp_path / "review" / "train" / f"{TXT_STEM}.json").exists()
    txt_after = _item_by_stem(dlg._review_items, TXT_STEM)
    assert txt_after.get("done") is None
    assert txt_after.get("human_approved") is not True
    assert len(txt_after["trusted"]) == 2
    assert all(b["hand_label"] is True for b in txt_after["trusted"])


def test_external_prune_keeps_txt_only_frame(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Dış budama model karesini budar; TXT-only kareye hiç dokunmaz."""
    dlg = factory_dialog
    txt_files = _txt_frame(tmp_path)
    model_files = _review_frame(
        tmp_path,
        weak=[_model_weak_box(0.30)],
        trusted=[{**_model_weak_box(0.95), "trusted": True, "weak": False}],
    )
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: True
    )
    dlg._reload_review_queue()
    assert {it["stem"] for it in dlg._review_items} == {TXT_STEM, MODEL_STEM}
    txt_before = txt_files["label"].read_bytes()

    dlg._bulk_prune_low_confidence_boxes()

    # Model karesi diskte budandı (0.30 weak kutu gitti).
    saved = json.loads(model_files["review"].read_text(encoding="utf-8"))
    assert all(
        float(b.get("confidence", 0.0)) != 0.30
        for b in saved.get("trusted", []) + saved.get("weak", [])
    )
    # TXT kare: bayt-bayt aynen; yeniden yüklemede hâlâ el etiketi/trusted.
    assert txt_files["label"].read_bytes() == txt_before == TXT_BYTES
    assert not (tmp_path / "review" / "train" / f"{TXT_STEM}.json").exists()
    txt_after = _item_by_stem(dlg._review_items, TXT_STEM)
    assert txt_after.get("done") is None
    assert txt_after["weak"] == []
    assert all(b["hand_label"] is True for b in txt_after["trusted"])


# ============================================================================
# (c) Reddet: korumalı karede hiçbir dosya silinmez
# ============================================================================


def test_reject_blocked_for_txt_frame(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    dlg = factory_dialog
    txt_files = _txt_frame(tmp_path)
    model_files = _review_frame(tmp_path, weak=[_model_weak_box()])

    items = _scan_items(tmp_path)
    txt_item = _item_by_stem(items, TXT_STEM)
    model_item = _item_by_stem(items, MODEL_STEM)

    # Tekli silme: korumalı karede [] döner, dosyalar durur.
    assert dlg._delete_review_sample(txt_item) == []
    assert txt_files["image"].is_file()
    assert txt_files["label"].is_file()
    assert txt_files["label"].read_bytes() == TXT_BYTES
    rejected_path = tmp_path / "rejected_stems.json"
    if rejected_path.is_file():
        assert TXT_STEM not in json.loads(rejected_path.read_text(encoding="utf-8"))

    # Tekli red akışı (_review_done "no"): kuyruktan çıkmaz, sayılmaz.
    dlg._review_items = [txt_item]
    dlg._review_index = 0
    reviewed_before = int(dlg._pipeline.get("reviewed", 0))
    dlg._review_done("no")
    assert [it["stem"] for it in dlg._review_items] == [TXT_STEM]
    assert int(dlg._pipeline.get("reviewed", 0)) == reviewed_before
    assert txt_files["image"].is_file()
    assert txt_files["label"].is_file()

    # Toplu silme: model kare silinir, TXT kare korunur.
    removed = dlg._batch_delete_review_samples([txt_item, model_item])
    assert removed > 0
    assert not model_files["image"].exists()
    assert not model_files["review"].exists()
    assert not model_files["label"].exists()
    assert txt_files["image"].is_file()
    assert txt_files["label"].is_file()
    assert txt_files["label"].read_bytes() == TXT_BYTES
    if rejected_path.is_file():
        rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
        assert TXT_STEM not in rejected
        assert MODEL_STEM in rejected


def test_bulk_reject_filtered_keeps_txt_frame_in_queue(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Toplu red ("all" filtresi): TXT kare kuyrukta kalır, model kare gider."""
    dlg = factory_dialog
    txt_files = _txt_frame(tmp_path)
    model_files = _review_frame(tmp_path, weak=[_model_weak_box()])
    dlg._reload_review_queue()
    assert {it["stem"] for it in dlg._review_items} == {TXT_STEM, MODEL_STEM}
    dlg.set_queue_filter_mode("all")

    dlg._bulk_reject_filtered()

    assert [it["stem"] for it in dlg._review_items] == [TXT_STEM]
    assert txt_files["image"].is_file()
    assert txt_files["label"].is_file()
    assert txt_files["label"].read_bytes() == TXT_BYTES
    assert not model_files["image"].exists()
    assert not model_files["review"].exists()


# ============================================================================
# (d) Dış budama fonksiyonu korumalı kutuyu atlar
# ============================================================================


def test_external_prune_skips_hand_label_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dış budama dosyaları yeniden yazınca mark_model_stale CWD bağlı
    # models/ dizinine bakabilir — tmp altına al.
    monkeypatch.chdir(tmp_path)
    hand_box = {
        "class_id": 6,
        "class_name": "tree_log",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.2,
        "confidence": 0.0,
        "conf": 0.0,
        "trusted": True,
        "weak": False,
        "hand_label": True,
        "source": HAND_LABEL_TXT_SOURCE,
    }
    stem = "frame_mixed"
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    rev = tmp_path / "review" / "train" / f"{stem}.json"
    lbl = tmp_path / "labels" / "train" / f"{stem}.txt"
    _dummy_png(tmp_path / "images" / "train" / f"{stem}.png")
    rev.write_text(
        json.dumps(
            {"stem": stem, "split": "train", "weak": [_model_weak_box(0.30)], "trusted": [hand_box]}
        ),
        encoding="utf-8",
    )
    lbl.write_text("6 0.5 0.5 0.1 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    res = prune_low_confidence_boxes_in_dataset(
        tmp_path, threshold=0.85, record_negative_exemplars=False
    )

    # Yalnızca model kutusu budandı; el etiketi kutusu duruyor.
    assert res["pruned_boxes"] == 1
    saved = json.loads(rev.read_text(encoding="utf-8"))
    kept = saved.get("trusted", []) + saved.get("weak", [])
    assert len(kept) == 1
    assert kept[0]["class_id"] == 6
    assert float(kept[0]["cx"]) == 0.5 and float(kept[0]["cy"]) == 0.5
    assert kept[0].get("hand_label") is True
    assert kept[0].get("human_approved") is not True
    # TXT yeniden yazıldıysa korumalı kutu aynen geri yazılır.
    txt_lines = lbl.read_text(encoding="utf-8").strip().splitlines()
    assert len(txt_lines) == 1
    assert txt_lines[0].split()[0] == "6"


# ============================================================================
# (e) Toplu onay (B-07): korumalı kare onaylanmaz
# ============================================================================


def test_bulk_approve_skips_txt_frame(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    dlg = factory_dialog
    _txt_frame(tmp_path)
    _dummy_png(tmp_path / "images" / "train" / f"{MODEL_STEM}.png")
    dlg._learn_box_exemplar = (  # type: ignore[method-assign]
        lambda it, box, **kw: True
    )
    txt_item = _item_by_stem(_scan_items(tmp_path), TXT_STEM)
    model_item: Dict[str, Any] = {
        "stem": MODEL_STEM,
        "split": "train",
        "weak": [_model_weak_box(0.95)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / f"{MODEL_STEM}.json"),
    }
    dlg._review_items = [txt_item, model_item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_approve()

    # Model kare toplu onaylandı (kuyruktan çıktı); TXT kare kuyrukta,
    # onaysız, review JSON'suz duruyor.
    assert [it["stem"] for it in dlg._review_items] == [TXT_STEM]
    txt_after = dlg._review_items[0]
    assert txt_after.get("done") is None
    assert txt_after.get("human_approved") is not True
    assert not (tmp_path / "review" / "train" / f"{TXT_STEM}.json").exists()
    assert (tmp_path / "labels" / "train" / f"{TXT_STEM}.txt").read_bytes() == TXT_BYTES


# ---------------------------------------------------------------------------
# P1-16 (K-17): fail-closed koruma + tarama GC karantinası.
# ---------------------------------------------------------------------------

def test_rejected_stem_is_hand_protected_fail_closed_on_corrupt_review(tmp_path: Path) -> None:
    """Bozuk / dict-olmayan review JSON korumalı sayılır — eski davranış
    False dönüp tarama GC'nin kareyi silmesine yol açıyordu (K-17)."""
    rev = tmp_path / "review" / "train" / "frame_x.json"
    lbl = tmp_path / "labels" / "train" / "frame_x.txt"
    rev.parent.mkdir(parents=True, exist_ok=True)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("", encoding="utf-8")

    rev.write_text("{bozuk json", encoding="utf-8")
    assert YOLOFactoryDialog._rejected_stem_is_hand_protected(rev, lbl) is True

    rev.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert YOLOFactoryDialog._rejected_stem_is_hand_protected(rev, lbl) is True

    # Kontrol: geçerli review, el etiketi yok → korumasız.
    rev.write_text(
        json.dumps({"stem": "frame_x", "trusted": [], "weak": [_model_weak_box()]}),
        encoding="utf-8",
    )
    assert YOLOFactoryDialog._rejected_stem_is_hand_protected(rev, lbl) is False


def test_scan_quarantines_corrupt_review_of_rejected_protected_frame(tmp_path: Path) -> None:
    """Reddedilen + bozuk review'lu kare fail-closed korumalı sayılır:
    görsel + TXT silinmez, bozuk review karantinaya alınır, olay kayda
    geçer, kare kuyruğa girmez."""
    stem = "frame_corrupt_rev"
    img = tmp_path / "images" / "train" / f"{stem}.png"
    rev = tmp_path / "review" / "train" / f"{stem}.json"
    lbl = tmp_path / "labels" / "train" / f"{stem}.txt"
    img.parent.mkdir(parents=True, exist_ok=True)
    rev.parent.mkdir(parents=True, exist_ok=True)
    lbl.parent.mkdir(parents=True, exist_ok=True)
    _dummy_png(img)
    rev.write_text("{bozuk json", encoding="utf-8")
    lbl.write_bytes(TXT_BYTES)

    events: List[Dict[str, Any]] = []
    items = YOLOFactoryDialog._scan_review_queue_items(
        tmp_path, {stem}, quarantine_events=events
    )

    assert [it["stem"] for it in items] == []
    assert len(events) == 1
    assert events[0]["stem"] == stem
    assert events[0]["kind"] == "review JSON"
    assert events[0]["quarantined_to"]
    assert not rev.exists()
    assert img.is_file()
    assert lbl.is_file()
    assert lbl.read_bytes() == TXT_BYTES
    assert list((tmp_path / "quarantine" / "review" / "train").glob(f"{stem}__*.json"))


def test_scan_quarantines_corrupt_txt_of_rejected_unprotected_frame(tmp_path: Path) -> None:
    """Reddedilen + korumasız karede bozuk TXT karantinaya alınır
    (silinmez); kalan dosyalar reddet kuralıyla silinir."""
    stem = "frame_corrupt_txt"
    files = _review_frame(tmp_path, stem=stem, weak=[_model_weak_box()])
    _dummy_png(files["image"])
    files["label"].write_text("6 0.5 0.5 0.1 0.1\n3 0.25\n", encoding="utf-8")

    events: List[Dict[str, Any]] = []
    items = YOLOFactoryDialog._scan_review_queue_items(
        tmp_path, {stem}, quarantine_events=events
    )

    assert [it["stem"] for it in items] == []
    assert len(events) == 1
    assert events[0]["kind"] == "labels TXT"
    assert events[0]["quarantined_to"]
    assert "2" in events[0]["reason"]  # bozuk satır no
    assert not files["image"].exists()
    assert not files["review"].exists()
    assert not files["label"].exists()
    assert list((tmp_path / "quarantine" / "labels" / "train").glob(f"{stem}__*.txt"))
