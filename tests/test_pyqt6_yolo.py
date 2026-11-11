# -*- coding: utf-8 -*-
"""tests/test_pyqt6_yolo.py — W4 (YOLO) PyQt6 çevirisi: prototip panel-2

YOLO Fabrikası inceleme/onay ekranının (Adım 2 — Onayla) HTML prototipteki
`yolo-fabrikasi-yeni-tasarim.html` panel-2 karşılıklarını doğrular:

- Fotoğraf/etiketleme alanı kocaman (16:9 canvas, min yükseklik >= 580),
  kutular + etiketler üzerinde, canvas ✕ kaldırma sinyaline bağlı.
- Sol onay kuyruğu okunur: küçük resim + kare no + güven + kutu sayısı +
  saat/sınıf/kutu satırı kesilmeden (özel satır widget'ı).
- Filtre & Arama: dosya/stem arama, durum filtresi
  (Tümü/Bekleyen/Onaylanan/Reddedilen/Yüksek/Düşük güvenli), tüm sınıfları
  içeren sınıf filtresi (player_character dahil), güven kaydırıcısı,
  "Sadece onaylananları eğit" anahtarı, onay sayacı,
  "Kuyruk · hepsi (n kare)" başlığı, Toplu İşlemler → "Filtredekileri
  Onayla (n)" — hepsi gerçek kuyruk verisini süzüp onaylar.
- Fotoğraf üstü: canlı önizleme satırı, Hız/Kalan göstergesi, Tam Ekran
  (Lightbox); fotoğraf altı: tıklayınca sınıf filtresi uygulayan renk lejantı.
- Alt aksiyonlar ONAYLA VE SONRAKİ (A), REDDET (R), Atla (S),
  Karakterimi Etiketle (K) + klavye kısayolları.
- Sağ panel (Sıradaki İş / Engeller / Çalıştırmalar / Canlı Log) görünür;
  altta filtreli Canlı Günlük şeridi (Tümü/INFO/Önizleme/Temizle).
- KİNAH TUTARI BU EKRANDA YOK: rozet, det-satır rozeti ve onay logundaki
  para değeri kaldırıldı (görünmez `_session_approved_kinah` sayacı
  ekonomi katmanında tutulur, ekranda gösterilmez).
- Eğitim akışı: "Eğit adımına git" butonu `_go_step(4)`'e bağlıdır
  (dondurma kilidi pipeline kuralı olarak korunur).

Sahte/demo veri yok: filtre testleri sentetik kuyruk öğeleriyle
`_filtered_review_indices`'in gerçek mantığını, toplu onay testi ise
tmp dataset + gerçek disk dosyalarıyla `_bulk_approve`'u çalıştırır.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from dataset_factory.approved_pool import ApprovedDatasetPool
import dataset_factory.approved_pool as approved_pool_mod
import dataset_factory.exemplar_memory as exemplar_memory_mod
from dataset_factory.exemplar_memory import ExemplarMemory
from ui.yolo_factory_dialog import YOLOFactoryDialog, YOLOPreviewCanvas


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _box(cid: int, conf: float, cname: str = "") -> Dict[str, Any]:
    return {
        "class_id": cid,
        "class_name": cname or f"class_{cid}",
        "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
        "confidence": conf, "conf": conf,
        "trusted": False, "weak": True,
    }


def _item(stem: str, weak: List[Dict[str, Any]], trusted: List[Dict[str, Any]],
          done: Any = None, mtime: float = 1762616400.0) -> Dict[str, Any]:
    return {
        "path": Path(f"/tmp/nope/{stem}.json"),
        "split": "train", "stem": stem,
        "weak": weak, "trusted": trusted,
        "done": done, "mtime": mtime,
        "n_weak": len(weak), "requeued": False,
    }


def _synthetic_items() -> List[Dict[str, Any]]:
    """4 kare: bekleyen-yüksek, bekleyen-düşük, onaylanan, reddedilen."""
    return [
        _item("gather_aria_s1_0001", [_box(0, 0.95, "herb_aria")], []),
        _item("gather_mela_s1_0002", [_box(1, 0.70, "herb_mela")], []),
        _item("gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip",
              [], [_box(0, 0.97, "herb_aria")], done="ok"),
        _item("player_char_frame_0003",
              [], [_box(26, 0.60, "player_character")], done="no"),
    ]


def _load_synthetic(dlg: YOLOFactoryDialog) -> None:
    dlg._review_items = _synthetic_items()
    dlg._review_index = 0
    dlg._queue_filter_mode = "all"
    dlg._queue_min_conf = 0.0
    dlg._render_queue_list()


def _visible_queue_stems(dlg: YOLOFactoryDialog) -> List[str]:
    return [
        str(dlg._review_items[int(dlg.queue_list.item(r).data(int(Qt.ItemDataRole.UserRole)))]["stem"])
        for r in range(dlg.queue_list.count())
    ]


# ----------------------------------------------------------------------
# 1. Kontroller var + canvas kocaman
# ----------------------------------------------------------------------

def test_review_controls_exist_and_canvas_is_large(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        # Fotoğraf alanı ekranın yıldızı: en az ~580px yükseklik (gerçek: 620).
        assert dlg.review_preview.minimumHeight() >= 580
        assert dlg.review_preview.minimumHeight() == 620

        # Filtre & Arama paneli
        assert dlg.review_search_edit.placeholderText().startswith("🔍 Dosya / stem ara")
        assert [dlg.queue_filter.itemData(i) for i in range(dlg.queue_filter.count())] == [
            "all", "pending", "ok", "no", "high", "low",
        ]
        assert dlg.queue_conf_slider.minimum() == 0
        assert dlg.queue_conf_slider.maximum() == 95
        assert dlg.queue_conf_val.text() == "≥ 0,00"
        assert dlg.btn_clear_filters.text() == "Filtreleri temizle"
        assert dlg.chk_only_approved.isChecked() is True
        assert dlg.lbl_approved_count is not None

        # Sınıf filtresi tüm sınıfları içerir (player_character = 26 dahil).
        class_data = [dlg.review_class_filter.itemData(i) for i in range(dlg.review_class_filter.count())]
        assert 26 in class_data
        assert -1 in class_data

        # Toplu İşlemler → Filtredekileri Onayla
        assert dlg.btn_approve_filtered.text().startswith("✓ Filtredekileri Onayla")
        assert dlg.btn_bulk is not None

        # Fotoğraf üstü: canlı önizleme satırı TEK — canvas paintEvent çizer
        # (lbl_preview_live çift satırdı, kaldırıldı; başlık preview_canvas'te).
        assert not hasattr(dlg, "lbl_preview_live")
        assert isinstance(dlg.preview_canvas, YOLOPreviewCanvas)
        assert dlg.preview_canvas.objectName() == "PipeCapturePreview"
        assert "Hız:" in dlg.lbl_speed_meter.text() and "Kalan:" in dlg.lbl_speed_meter.text()
        assert "Tam Ekran (Lightbox)" in dlg.btn_open_lightbox.text()

        # Fotoğraf altı renk lejantı: tüm sınıflar + Tümü
        assert -1 in dlg._legend_buttons
        assert 0 in dlg._legend_buttons and 26 in dlg._legend_buttons
        assert "herb_aria" in dlg._legend_buttons[0].text()
        assert "player_character" in dlg._legend_buttons[26].text()

        # Alt aksiyonlar + kısayollar
        assert "ONAYLA VE SONRAKİ" in dlg.btn_approve.text()
        assert "REDDET" in dlg.btn_reject.text()
        assert "Atla" in dlg.btn_skip.text()
        assert "Karakterimi Etiketle" in dlg.btn_add_player_char.text()
        assert dlg.btn_goto_train.text() == "Eğit adımına git →"
        for name in ("_shortcut_approve_a", "_shortcut_reject_r", "_shortcut_skip_s", "_shortcut_player_k"):
            assert hasattr(dlg, name), f"eksik kısayol: {name}"

        # Sağ panel: Sıradaki İş / Engeller / Çalıştırmalar / Canlı Log
        assert dlg.next_title is not None
        assert dlg.block_count is not None
        assert dlg.run_list_host is not None
        assert dlg.training_log_box is not None

        # Alttaki filtreli Canlı Günlük şeridi
        assert dlg.live_journal_box is not None
        assert [b.text() for b in dlg._journal_filter_group.buttons()] == ["Tümü", "INFO", "Önizleme"]
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 2. Bu ekranda Kinah tutarı yok
# ----------------------------------------------------------------------

def test_no_kinah_amount_on_review_screen(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        _load_synthetic(dlg)
        # Rozet widget'ı kaldırıldı.
        assert not hasattr(dlg, "lbl_review_kinah")
        assert dlg.findChild(object, "ReviewKinahBadge") is None
        # Alt başlıkta para değeri yok (Kinah sınıfı kutu olsa bile).
        assert "Kinah" not in dlg.review_sub.text()
        assert "💰" not in dlg.review_sub.text() and "🪙" not in dlg.review_sub.text()
        # İnceleme sayfasındaki etiketlerde para rozeti / Kinah değeri yok.
        review_page = dlg.step_stack.widget(1)
        for lbl in review_page.findChildren(QLabel):
            assert "💰" not in lbl.text() and "🪙" not in lbl.text(), lbl.text()
            assert "Kinah" not in lbl.text(), lbl.text()
        for btn in review_page.findChildren(QPushButton):
            assert "Kinah" not in btn.text(), btn.text()
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 3. Durum filtresi modları gerçek kuyruğu süzer
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode,label,expected",
    [
        ("all", "hepsi", ["gather_aria_s1_0001", "gather_mela_s1_0002",
                 "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip",
                 "player_char_frame_0003"]),
        ("pending", "bekleyen", ["gather_aria_s1_0001", "gather_mela_s1_0002"]),
        ("ok", "onaylanan", ["gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip"]),
        ("no", "reddedilen", ["player_char_frame_0003"]),
        ("high", "yüksek güvenli", ["gather_aria_s1_0001",
                  "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip"]),
        ("low", "düşük güvenli", ["gather_mela_s1_0002", "player_char_frame_0003"]),
    ],
)
def test_status_filter_modes(qapp: QApplication, mode: str, label: str, expected: List[str]) -> None:
    dlg = YOLOFactoryDialog()
    try:
        _load_synthetic(dlg)
        # Combo üzerinden seç (gerçek wiring): veri değişince kuyruk yenilenir.
        dlg.queue_filter.setCurrentIndex(dlg.queue_filter.findData(mode))
        assert dlg._queue_filter_mode == mode
        assert _visible_queue_stems(dlg) == expected
        assert dlg.queue_filter_lbl.text() == f"Kuyruk · {label} ({len(expected)} kare)"
    finally:
        dlg.close()


def test_search_class_conf_filters_and_clear(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        _load_synthetic(dlg)
        # Dosya/stem arama
        dlg.review_search_edit.setText("aria")
        assert _visible_queue_stems(dlg) == [
            "gather_aria_s1_0001",
            "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip",
        ]
        dlg.review_search_edit.clear()

        # Sınıf filtresi: player_character (26) yalnızca o kareyi bırakır.
        dlg.review_class_filter.setCurrentIndex(dlg.review_class_filter.findData(26))
        assert _visible_queue_stems(dlg) == ["player_char_frame_0003"]
        dlg.review_class_filter.setCurrentIndex(0)

        # Güven kaydırıcısı: ≥ 0,85 düşük güvenli kareyi eler.
        dlg.queue_conf_slider.setValue(85)
        assert dlg.queue_conf_val.text() == "≥ 0,85"
        assert _visible_queue_stems(dlg) == [
            "gather_aria_s1_0001",
            "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip",
        ]

        # Filtreleri temizle: hepsi sıfırlanır, 4 kare geri gelir.
        dlg.btn_clear_filters.click()
        assert dlg.review_search_edit.text() == ""
        assert dlg.queue_filter.currentData() == "all"
        assert dlg.review_class_filter.currentData() == -1
        assert dlg.queue_conf_slider.value() == 0
        assert len(_visible_queue_stems(dlg)) == 4
        assert dlg.queue_filter_lbl.text() == "Kuyruk · hepsi (4 kare)"
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 4. Kuyruk satırları okunur: küçük resim + tam stem + güven/kutu/saat
# ----------------------------------------------------------------------

def test_queue_rows_are_readable_widgets(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        _load_synthetic(dlg)
        assert dlg.queue_list.count() == 4
        assert dlg.queue_filter_lbl.text() == "Kuyruk · hepsi (4 kare)"
        long_stem = "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip"
        row_idx = _visible_queue_stems(dlg).index(long_stem)
        widget = dlg.queue_list.itemWidget(dlg.queue_list.item(row_idx))
        assert widget is not None
        labels = widget.findChildren(QLabel)
        texts = [lb.text() for lb in labels]
        # Kare no kesilmeden: tam stem satırda durur.
        assert any(f"#{long_stem}" in t for t in texts), texts
        # Güven + kutu sayısı + saat/sınıf satırı.
        assert any("%97" in t and "1 kutu" in t for t in texts), texts
        assert any("herb_aria×1" in t for t in texts), texts
        # Küçük resim etiketi 72x54.
        thumbs = [lb for lb in labels if lb.size().width() == 72]
        assert len(thumbs) == 1 and thumbs[0].size().height() == 54
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 5. Lejant tıklaması sınıf filtresi uygular (tekrarı sıfırlar)
# ----------------------------------------------------------------------

def test_legend_click_applies_class_filter(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        _load_synthetic(dlg)
        dlg._legend_buttons[0].click()
        assert dlg.review_class_filter.currentData() == 0
        assert _visible_queue_stems(dlg) == [
            "gather_aria_s1_0001",
            "gather_t2s_aria_s1_6992862_0001_aug06_night_fog_hflip",
        ]
        dlg._legend_buttons[0].click()
        assert dlg.review_class_filter.currentData() == -1
        assert len(_visible_queue_stems(dlg)) == 4
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 6. Filtredekileri Onayla: gerçek disk + havuz, Kinah'sız log
# ----------------------------------------------------------------------

def test_bulk_approve_filtered_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "review" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    pool = ApprovedDatasetPool(project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)
    exemplar_path = tmp_path / "learned_exemplars.json"
    monkeypatch.setattr(
        exemplar_memory_mod, "_GLOBAL_EXEMPLAR_MEMORY", ExemplarMemory(json_path=exemplar_path)
    )

    import json as _json

    rng = np.random.default_rng(11)
    for stem, cid, conf in (("bulk_high_0001", 0, 0.95), ("bulk_low_0002", 3, 0.70)):
        img_p = dataset_dir / "images" / "train" / f"{stem}.jpg"
        assert cv2.imwrite(str(img_p), rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
        rev = {
            "weak": [{"class_id": cid, "class_name": "herb_aria" if cid == 0 else "ore_orichalcum",
                      "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
                      "confidence": conf, "trusted": False, "weak": True}],
            "trusted": [],
        }
        (dataset_dir / "review" / "train" / f"{stem}.json").write_text(
            _json.dumps(rev), encoding="utf-8"
        )

    dlg = YOLOFactoryDialog()
    try:
        # [YOLO-ASYNC] __init__'teki başlangıç yüklemesi bitmeden dizin
        # değiştirilip senkron yükleme yapılırsa, bitince kuyruğu ezer.
        assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
        dlg._auto_confirm_dialogs = True
        dlg._builtin_helper.dataset_dir = dataset_dir
        ctrl = getattr(dlg, "factory_controller", None)
        if ctrl is not None and hasattr(ctrl, "dataset_dir"):
            ctrl.dataset_dir = dataset_dir
        assert dlg._dataset_dir() == dataset_dir
        dlg._reload_review_queue()
        assert len(dlg._review_items) == 2

        # Yüksek güvenli filtresi yalnızca 0,95'lik kareyi bırakır.
        dlg.queue_filter.setCurrentIndex(dlg.queue_filter.findData("high"))
        assert _visible_queue_stems(dlg) == ["bulk_high_0001"]
        assert dlg.btn_approve_filtered.text() == "✓ Filtredekileri Onayla (1)"

        dlg.btn_approve_filtered.click()
        # Onaylanan kare kuyruktan çıktı, düşük güvenli beklemede kaldı.
        assert dlg._pending_count() == 1
        assert _visible_queue_stems(dlg) == []
        # Onay logunda Kinah tutarı yok.
        assert "Kinah" not in dlg.training_log_box.toPlainText()
        assert "Filtredeki 1 kare onaylandı" in dlg.training_log_box.toPlainText()
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 7. Canlı Günlük şeridi filtreleri + Temizle
# ----------------------------------------------------------------------

def test_live_journal_strip_filters(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        dlg._append_log("[ÖNİZLEME] journal-preview-probe")
        dlg._append_log("[BİLGİ] journal-info-probe")
        dlg._set_journal_filter("all")
        assert "journal-preview-probe" in dlg.live_journal_box.toPlainText()
        assert "journal-info-probe" in dlg.live_journal_box.toPlainText()
        dlg._set_journal_filter("preview")
        assert "journal-preview-probe" in dlg.live_journal_box.toPlainText()
        assert "journal-info-probe" not in dlg.live_journal_box.toPlainText()
        dlg._set_journal_filter("info")
        assert "journal-preview-probe" not in dlg.live_journal_box.toPlainText()
        assert "journal-info-probe" in dlg.live_journal_box.toPlainText()
        dlg._clear_journal()
        assert "journal-info-probe" not in dlg.live_journal_box.toPlainText()
        assert dlg.live_journal_box.toPlainText().strip() == "[HAZIR] Canlı günlük açık."
    finally:
        dlg.close()


# ----------------------------------------------------------------------
# 8. "Sadece onaylananları eğit" anahtarı + "Eğit adımına git"
# ----------------------------------------------------------------------

def test_only_approved_toggle_logs_and_train_goto(qapp: QApplication) -> None:
    dlg = YOLOFactoryDialog()
    try:
        dlg.chk_only_approved.setChecked(False)
        assert "Sadece onaylananları eğit: KAPALI" in dlg.training_log_box.toPlainText()
        dlg.chk_only_approved.setChecked(True)
        assert "Sadece onaylananları eğit: AÇIK" in dlg.training_log_box.toPlainText()
        # Eğitim kilidi: dondurulmamış veri setiyle Eğit adımı Hazırla'ya döner
        # (pipeline soft-gate kuralı korunur; buton _go_step(4)'e bağlıdır).
        # Not: dialog diskteki pipeline durumunu yükler; repo durumunda frozen
        # True olabilir — kilit davranışını test etmek için False'a çekilir.
        dlg._pipeline["frozen"] = False
        dlg.btn_goto_train.click()
        assert dlg.step_stack.currentIndex() == 2  # Adım 3 — Hazırla
    finally:
        dlg.close()
