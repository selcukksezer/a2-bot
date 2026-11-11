# -*- coding: utf-8 -*-
"""
P1-17 (K-06 + K-09) hedefli testler — turbo/relabel guard'ları.

Kilitlenen davranış (taban: feat/all-combined-v16 @ 3e739f7):
- Turbo `_process_one`, mevcut review/TXT'yi okuyup el etiketlerini
  model çıktısıyla BİRLEŞTİRİR: trusted'taki el kutusu trusted'ta,
  weak'teki (demote edilmiş) el kutusu weak'te kalır; model kutusu
  korunan bir el kutusuyla çakışıyorsa eklenmez.
- Bozuk review JSON fail-closed'dur: üzerine yazılmaz, kare `error`
  sayılır, diskteki review/TXT aynen kalır.
- Bootstrap idempotent'tir: review'da kutu varsa ikinci çalışmada
  atlanır; commit sonrası review JSON silinmez (`review.unlink()`
  kalktı); `--force` iki kontrolü de aşar.
- `_auto_label_unlabeled_suspicious`, el etiketli kareyi hedefe
  sokmaz ve çalışmadan önce onay sorar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.dataset_curator import (
    DatasetCurator,
    _is_hand_box,
    _turbo_box_identity_match,
    imwrite_unicode,
)
from tools.bootstrap_learn_labels import _review_has_boxes


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _create_dummy_image(path: Path, width: int = 320, height: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), (60, 120, 180), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (120, 120), (200, 220, 50), -1)
    assert imwrite_unicode(path, img) is True
    return path


def _hand_box(cid: int = 0, cx: float = 0.5, cy: float = 0.5,
              w: float = 0.2, h: float = 0.2, **extra: Any) -> Dict[str, Any]:
    box = {
        "class_id": cid,
        "class_name": f"class_{cid}",
        "cx": cx, "cy": cy, "w": w, "h": h,
        "confidence": 0.95,
        "hand_label": True,
        "source": "manual_draw",
        "trusted": True,
        "weak": False,
    }
    box.update(extra)
    return box


def _model_box(cid: int = 0, cx: float = 0.5, cy: float = 0.5,
               w: float = 0.2, h: float = 0.2, conf: float = 0.93) -> Dict[str, Any]:
    return {
        "class_id": cid,
        "class_name": f"class_{cid}",
        "cx": cx, "cy": cy, "w": w, "h": h,
        "confidence": conf,
        "source": "yolo_model_v2",
    }


def _seed_review(dataset_dir: Path, stem: str, split: str = "train",
                 trusted: List[Dict[str, Any]] | None = None,
                 weak: List[Dict[str, Any]] | None = None) -> Path:
    review_path = dataset_dir / "review" / split / f"{stem}.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(
            {"stem": stem, "trusted": trusted or [], "weak": weak or []},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return review_path


def _read_review(dataset_dir: Path, stem: str, split: str = "train") -> Dict[str, Any]:
    return json.loads(
        (dataset_dir / "review" / split / f"{stem}.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# 1. Saf predikatlar
# ---------------------------------------------------------------------------

def test_is_hand_box_predicate() -> None:
    assert _is_hand_box({"hand_label": True}) is True
    assert _is_hand_box({"source": "manual_draw"}) is True
    assert _is_hand_box({"hand_label": True, "source": "manual_draw"}) is True
    # TXT uzlaştırması TXT-only el etiketini hand_label=True +
    # source="hand_label_txt" ile taşır — predikat onu da yakalar.
    assert _is_hand_box({"hand_label": True, "source": "hand_label_txt"}) is True
    assert _is_hand_box({"source": "yolo_model_v2"}) is False
    assert _is_hand_box({"hand_label": False}) is False
    assert _is_hand_box({}) is False
    assert _is_hand_box(None) is False  # type: ignore[arg-type]
    assert _is_hand_box("manual_draw") is False  # type: ignore[arg-type]


def test_turbo_box_identity_match() -> None:
    hand = _hand_box(0, 0.5, 0.5, 0.2, 0.2)
    assert _turbo_box_identity_match(hand, hand) is True
    # Aynı sınıf + 0.02 tolerans içi koordinat: aynı kutu.
    assert _turbo_box_identity_match(
        _model_box(0, 0.51, 0.49, 0.21, 0.19), hand
    ) is True
    # Farklı sınıf: farklı kutu.
    assert _turbo_box_identity_match(_model_box(1, 0.5, 0.5, 0.2, 0.2), hand) is False
    # Koordinat farkı tolerans dışı: farklı kutu.
    assert _turbo_box_identity_match(_model_box(0, 0.8, 0.5, 0.2, 0.2), hand) is False
    assert _turbo_box_identity_match(_model_box(0, 0.5, 0.5, 0.4, 0.2), hand) is False


# ---------------------------------------------------------------------------
# 2. Turbo birleştirme (DatasetCurator.relabel_dataset_batch)
# ---------------------------------------------------------------------------

def test_turbo_preserves_hand_boxes_when_model_finds_nothing(tmp_path: Path) -> None:
    """K-06: model 0 kutu bulsa bile review'daki el etiketleri silinmez."""
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)
    img_path = _create_dummy_image(curator.images_train_dir / "hand_frame_01.jpg")

    hand_trusted = _hand_box(0, 0.5, 0.5, 0.2, 0.2)
    hand_demoted = _hand_box(
        1, 0.2, 0.3, 0.15, 0.15,
        trusted=False, weak=True, demoted_by_user=True,
    )
    _seed_review(dataset_dir, "hand_frame_01", trusted=[hand_trusted], weak=[hand_demoted])

    class EmptyLabeler:
        def auto_label_frame(self, frame, fast_mode=True, source_path=None, full_scan=False):
            return []

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", EmptyLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[img_path], force_all=True, max_workers=1
        )

    assert res["errors"] == 0
    assert res["labeled"] == 1
    rev = _read_review(dataset_dir, "hand_frame_01")
    # Trusted el kutusu trusted'ta, demote edilmiş el kutusu weak'te kalır.
    assert len(rev["trusted"]) == 1
    assert rev["trusted"][0]["class_id"] == 0
    assert rev["trusted"][0].get("hand_label") is True
    assert len(rev["weak"]) == 1
    assert rev["weak"][0]["class_id"] == 1
    assert rev["weak"][0].get("demoted_by_user") is True
    # El etiketinin TXT karşılığı da durur.
    txt_lines = [
        ln.strip()
        for ln in (dataset_dir / "labels" / "train" / "hand_frame_01.txt")
        .read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(txt_lines) == 1
    assert int(txt_lines[0].split()[0]) == 0


def test_turbo_preserves_txt_only_hand_label(tmp_path: Path) -> None:
    """K-06/K-01: review JSON yokken TXT-only el etiketi de korunur."""
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)
    img_path = _create_dummy_image(curator.images_train_dir / "txt_hand_01.jpg")
    label_path = dataset_dir / "labels" / "train" / "txt_hand_01.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")

    class EmptyLabeler:
        def auto_label_frame(self, frame, fast_mode=True, source_path=None, full_scan=False):
            return []

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", EmptyLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[img_path], force_all=True, max_workers=1
        )

    assert res["errors"] == 0
    assert res["labeled"] == 1
    rev = _read_review(dataset_dir, "txt_hand_01")
    assert len(rev["trusted"]) == 1
    assert rev["trusted"][0]["class_id"] == 0
    assert rev["trusted"][0].get("hand_label") is True
    assert label_path.read_text(encoding="utf-8").strip() != ""


def test_turbo_model_box_duplicating_hand_box_is_not_added(tmp_path: Path) -> None:
    """K-06: model, korunan el kutusunun kopyasını eklemez; farklı kutuyu ekler."""
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)
    img_path = _create_dummy_image(curator.images_train_dir / "dup_hand_01.jpg")
    _seed_review(dataset_dir, "dup_hand_01", trusted=[_hand_box(0, 0.5, 0.5, 0.2, 0.2)])

    class DupLabeler:
        def auto_label_frame(self, frame, fast_mode=True, source_path=None, full_scan=False):
            return [
                # El kutusuyla çakışan model kopyası (aynı cid + koordinat).
                _model_box(0, 0.51, 0.49, 0.2, 0.2, conf=0.93),
                # Çakışmayan yeni model kutusu (cid 2, human_approved ile trusted).
                dict(_model_box(2, 0.8, 0.8, 0.1, 0.1, conf=0.91),
                     human_approved=True),
            ]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", DupLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[img_path], force_all=True, max_workers=1
        )

    assert res["errors"] == 0
    rev = _read_review(dataset_dir, "dup_hand_01")
    cid0 = [b for b in rev["trusted"] if b["class_id"] == 0]
    assert len(cid0) == 1, "el kutusu tam bir kez kalmalı (model kopyası eklenmemeli)"
    assert cid0[0].get("hand_label") is True, "kalan cid-0 kutusu el versiyonu olmalı"
    assert {b["class_id"] for b in rev["trusted"]} == {0, 2}


def test_turbo_corrupt_review_is_fail_closed(tmp_path: Path) -> None:
    """P1-16/K-05: bozuk review JSON üzerine yazılmaz; kare error sayılır."""
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)
    img_path = _create_dummy_image(curator.images_train_dir / "corrupt_01.jpg")
    review_path = dataset_dir / "review" / "train" / "corrupt_01.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("{bozuk json", encoding="utf-8")
    label_path = dataset_dir / "labels" / "train" / "corrupt_01.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
    review_before = review_path.read_bytes()
    label_before = label_path.read_bytes()

    class GoodLabeler:
        def auto_label_frame(self, frame, fast_mode=True, source_path=None, full_scan=False):
            return [_model_box(3, 0.3, 0.3, 0.1, 0.1, conf=0.95)]

    with patch("dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", GoodLabeler):
        res = curator.relabel_dataset_batch(
            image_paths=[img_path], force_all=True, max_workers=1
        )

    assert res["errors"] == 1
    assert "corrupt_01" in res["error_stems"]
    assert res["labeled"] == 0
    assert review_path.read_bytes() == review_before, "bozuk review aynen kalmalı"
    assert label_path.read_bytes() == label_before, "TXT aynen kalmalı"


# ---------------------------------------------------------------------------
# 3. Bootstrap idempotency (tools/bootstrap_learn_labels.py)
# ---------------------------------------------------------------------------

def test_review_has_boxes_predicate(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert _review_has_boxes(missing) is False

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"trusted": [], "weak": []}), encoding="utf-8")
    assert _review_has_boxes(empty) is False

    no_keys = tmp_path / "nokeys.json"
    no_keys.write_text(json.dumps({"stem": "x"}), encoding="utf-8")
    assert _review_has_boxes(no_keys) is False

    trusted = tmp_path / "trusted.json"
    trusted.write_text(
        json.dumps({"trusted": [_hand_box(0)], "weak": []}), encoding="utf-8"
    )
    assert _review_has_boxes(trusted) is True

    weak = tmp_path / "weak.json"
    weak.write_text(
        json.dumps({"trusted": [], "weak": [_model_box(2)]}), encoding="utf-8"
    )
    assert _review_has_boxes(weak) is True

    # Bozuk/okunamaz JSON fail-closed: True (üzerine yazılmaz).
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bozuk", encoding="utf-8")
    assert _review_has_boxes(corrupt) is True

    not_dict = tmp_path / "list.json"
    not_dict.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _review_has_boxes(not_dict) is True


def _bootstrap_dataset(tmp_path: Path) -> Path:
    """Tek learn_monsters karesi + gerçek game_learning JSON'lu dataset."""
    ds = tmp_path / "dataset"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "game_learning" / "monsters").mkdir(parents=True)
    (ds / "images" / "train" / "learn_monsters_01.jpg").write_bytes(b"fakejpg")
    (ds / "game_learning" / "monsters" / "learn_monsters_01.json").write_text(
        json.dumps({"boxes": [{"class_id": 1, "box": [0.5, 0.5, 0.2, 0.2]}]}),
        encoding="utf-8",
    )
    return ds


def _run_bootstrap(ds: Path, extra_args: List[str] | None = None, capsys=None) -> Dict[str, Any]:
    """main() çalıştır, stdout'taki JSON özeti döndür."""
    import tools.bootstrap_learn_labels as boot

    argv = ["bootstrap_learn_labels.py", "--dataset", str(ds)] + (extra_args or [])
    with patch.object(sys, "argv", argv):
        rc = boot.main()
    assert rc == 0
    assert capsys is not None
    out = capsys.readouterr().out
    start = out.index("{")
    return json.loads(out[start:])


def test_bootstrap_second_run_skips_and_keeps_review(tmp_path: Path, capsys) -> None:
    """K-09: ikinci çalışmada review'u kutulu kare atlanır; review silinmez."""
    ds = _bootstrap_dataset(tmp_path)

    first = _run_bootstrap(ds, capsys=capsys)
    assert first["rewritten"] == 1
    review_path = ds / "review" / "train" / "learn_monsters_01.json"
    assert review_path.is_file()
    assert _review_has_boxes(review_path) is True
    review_before = review_path.read_bytes()
    txt_path = ds / "labels" / "train" / "learn_monsters_01.txt"
    txt_before = txt_path.read_bytes() if txt_path.is_file() else None

    second = _run_bootstrap(ds, capsys=capsys)
    assert second["rewritten"] == 0
    assert second["skipped"] >= 1
    # Commit sonrası review.unlink() kalktı: review diske kalır, aynen durur.
    assert review_path.is_file()
    assert review_path.read_bytes() == review_before
    if txt_before is not None:
        assert txt_path.read_bytes() == txt_before


def test_bootstrap_force_reruns_boxed_review(tmp_path: Path, capsys) -> None:
    """K-09: --force, review'da kutu olsa bile yeniden yazar."""
    ds = _bootstrap_dataset(tmp_path)
    assert _run_bootstrap(ds, capsys=capsys)["rewritten"] == 1
    assert _run_bootstrap(ds, capsys=capsys)["rewritten"] == 0
    forced = _run_bootstrap(ds, extra_args=["--force"], capsys=capsys)
    assert forced["rewritten"] == 1
    assert forced["force"] is True


def test_bootstrap_skips_hand_labeled_frame_with_empty_txt(tmp_path: Path, capsys) -> None:
    """K-09/C011: el etiketli kare (review'da el kutusu + boş TXT) atlanır;
    boş TXT, el etiketini commit'te sıfırlatmaz."""
    ds = _bootstrap_dataset(tmp_path)
    review_path = ds / "review" / "train" / "learn_monsters_01.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps({"stem": "learn_monsters_01", "trusted": [_hand_box(0)], "weak": []}),
        encoding="utf-8",
    )
    txt_path = ds / "labels" / "train" / "learn_monsters_01.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("", encoding="utf-8")
    review_before = review_path.read_bytes()

    summary = _run_bootstrap(ds, capsys=capsys)
    assert summary["rewritten"] == 0
    assert summary["skipped"] >= 1
    assert review_path.read_bytes() == review_before
    assert txt_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# 4. Dialog seviyesi: el-etiketli filtre + onay dialogu
#    (YOLOFactoryDialog — module-scope tek dialog; yapım ~13 sn.)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(scope="module")
def factory_dialog(qapp, tmp_path_factory):
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    dlg = YOLOFactoryDialog()
    base = tmp_path_factory.mktemp("p117_ui")
    dlg._dataset_dir = lambda: base  # type: ignore[assignment]
    dlg._auto_confirm_dialogs = True
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base / "review" / "train").mkdir(parents=True, exist_ok=True)
    yield dlg, base
    try:
        # Yapım sırasında başlayan yükleme thread'ini tek desenle durdur.
        dlg._stop_yolo_threads()
        dlg.close()
    except Exception:
        pass


def _ui_item(stem: str, trusted=None, weak=None) -> Dict[str, Any]:
    return {
        "stem": stem, "split": "train",
        "trusted": list(trusted or []), "weak": list(weak or []),
    }


def test_item_is_hand_labeled_trusted_hand_box(factory_dialog) -> None:
    dlg, _base = factory_dialog
    assert dlg._item_is_hand_labeled(_ui_item("x", trusted=[_hand_box(0)])) is True
    assert dlg._item_is_hand_labeled(
        _ui_item("x", trusted=[_hand_box(0)], weak=[_model_box(2)])
    ) is True


def test_item_is_hand_labeled_demoted_weak_hand_box(factory_dialog) -> None:
    dlg, _base = factory_dialog
    demoted = _hand_box(1, trusted=False, weak=True, demoted_by_user=True)
    assert dlg._item_is_hand_labeled(_ui_item("x", weak=[demoted])) is True


def test_item_is_hand_labeled_model_only_is_false(factory_dialog) -> None:
    dlg, _base = factory_dialog
    assert dlg._item_is_hand_labeled(
        _ui_item("nohand", trusted=[_model_box(0)], weak=[_model_box(3)])
    ) is False
    assert dlg._item_is_hand_labeled(_ui_item("nohand2")) is False


def test_item_is_hand_labeled_txt_only_frame(factory_dialog) -> None:
    """K-06: bellek bayrakları boş olsa bile diskteki TXT-only el etiketi korur."""
    dlg, base = factory_dialog
    (base / "labels" / "train" / "txt_only_ui.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
    )
    assert dlg._item_is_hand_labeled(_ui_item("txt_only_ui")) is True


def _mock_relabel():
    from dataset_factory.dataset_curator import DatasetCurator

    return patch.object(
        DatasetCurator,
        "relabel_dataset_batch",
        return_value={"processed": 2, "fps": 1.0, "total_boxes": 3},
    )


def _write_ui_image(base: Path, stem: str) -> Path:
    p = base / "images" / "train" / f"{stem}.jpg"
    p.write_bytes(b"fakejpg")
    return p


def test_auto_label_excludes_hand_labeled_frames(factory_dialog) -> None:
    """K-06 kanıtı: trusted'ta el etiketi + 1 weak kutu olan kare hedefe girmez."""
    from PyQt6.QtWidgets import QMessageBox

    dlg, base = factory_dialog
    # P2-20: worker asenkron — Yes testindeki çözüm: önceki turdan sızan
    # turbo worker + YOLO yükleme bitmeden mock kurulmaz / _review_items
    # yazılmaz; modül-kapsamlı dialog paylaşılır, sızan worker hem mock'u
    # kirletir hem de kuyruk yenilemesiyle _review_items'ı ezer.
    dlg.wait_for_turbo_relabel(timeout_ms=30000)
    dlg.wait_for_yolo_load(timeout_ms=30000)
    dlg._auto_confirm_dialogs = True
    hand_img = _write_ui_image(base, "ui_hand")
    empty_img = _write_ui_image(base, "ui_empty")
    susp_img = _write_ui_image(base, "ui_susp")
    dlg._review_items = [
        _ui_item("ui_hand", trusted=[_hand_box(0)], weak=[_model_box(2)]),
        _ui_item("ui_empty"),
        _ui_item("ui_susp", trusted=[_model_box(0)], weak=[_model_box(3)]),
    ]
    with (
        _mock_relabel() as mock_relabel,
        patch("ui.yolo_factory_dialog.QMessageBox.information"),
    ):
        dlg._auto_label_unlabeled_suspicious()
        # P2-20: worker asenkron — mock kalkmadan bitmesini bekle.
        dlg.wait_for_turbo_relabel(timeout_ms=30000)

    assert mock_relabel.call_count == 1
    called_paths = [Path(p) for p in mock_relabel.call_args.kwargs["image_paths"]]
    assert hand_img not in called_paths
    assert set(called_paths) == {empty_img, susp_img}
    log_text = dlg.training_log_box.toPlainText()
    assert "1 el etiketli kare hariç tutuldu" in log_text


def test_auto_label_confirm_no_cancels(factory_dialog) -> None:
    """K-06/B-06: onay dialogunda Hayır → turbo hiç çalışmaz."""
    from PyQt6.QtWidgets import QMessageBox

    dlg, base = factory_dialog
    dlg._auto_confirm_dialogs = False
    _write_ui_image(base, "ui_empty2")
    dlg._review_items = [_ui_item("ui_empty2")]
    try:
        with (
            _mock_relabel() as mock_relabel,
            patch(
                "ui.yolo_factory_dialog.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ),
        ):
            dlg._auto_label_unlabeled_suspicious()
    finally:
        dlg._auto_confirm_dialogs = True

    assert mock_relabel.call_count == 0
    assert "iptal edildi" in dlg.training_log_box.toPlainText()


def test_auto_label_confirm_yes_runs(factory_dialog) -> None:
    """K-06/B-06: onay dialogunda Evet → yalnızca el-etiketsiz hedefler çalışır."""
    from PyQt6.QtWidgets import QMessageBox

    dlg, base = factory_dialog
    # Önceki testten sızan turbo worker + onun tetiklediği kuyruk
    # yenilemesi bitmeden mock kurulmaz / _review_items yazılmaz:
    # modül-kapsamlı dialog paylaşılır; sızan worker hem mock'u
    # kirletir hem de kuyruk yenilemesiyle _review_items'ı ezer.
    dlg.wait_for_turbo_relabel(timeout_ms=30000)
    dlg.wait_for_yolo_load(timeout_ms=30000)
    dlg._auto_confirm_dialogs = False
    empty_img = _write_ui_image(base, "ui_empty3")
    _write_ui_image(base, "ui_hand3")
    dlg._review_items = [
        _ui_item("ui_hand3", trusted=[_hand_box(0)]),
        _ui_item("ui_empty3"),
    ]
    try:
        with (
            _mock_relabel() as mock_relabel,
            patch(
                "ui.yolo_factory_dialog.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch("ui.yolo_factory_dialog.QMessageBox.information"),
        ):
            dlg._auto_label_unlabeled_suspicious()
            # P2-20: worker asenkron — mock kalkmadan bitmesini bekle.
            dlg.wait_for_turbo_relabel(timeout_ms=30000)
    finally:
        dlg._auto_confirm_dialogs = True

    assert mock_relabel.call_count == 1
    called_paths = [Path(p) for p in mock_relabel.call_args.kwargs["image_paths"]]
    assert called_paths == [empty_img]


def test_auto_label_no_targets_reports_protected_hand_frames(factory_dialog) -> None:
    """K-06: hedef yoksa ve el etiketli kareler korunduysa bilgi mesajı bunu söyler."""
    dlg, base = factory_dialog
    dlg._auto_confirm_dialogs = True
    dlg._review_items = [_ui_item("ui_hand4", trusted=[_hand_box(0)])]
    with (
        _mock_relabel() as mock_relabel,
        patch("ui.yolo_factory_dialog.QMessageBox.information") as mock_info,
    ):
        dlg._auto_label_unlabeled_suspicious()

    assert mock_relabel.call_count == 0
    assert mock_info.call_count == 1
    assert "korundu" in str(mock_info.call_args.args[2])
