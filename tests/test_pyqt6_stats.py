# -*- coding: utf-8 -*-
"""PyQt6 İstatistik (AnalyticsDialog) — gatherion-istatistik-prototipi uyumu.

Kapsam: ui/analytics_dialog.py yalnızca.
- Boş durumda sahte/demo veri yok: tablo "henüz kayıt yok", içgörüler "—",
  grafik boş durum metni.
- Gerçek TelemetryEngine olaylarıyla kartlar / tablo / içgörüler / detay
  paneli (son olaylar + olay detayı) hesaplanır.
- Tüm kontroller gerçek handler'a bağlıdır (butonlar, filtreler, dilim,
  sıralama, günlük filtreleri, dışa aktarma, sıfırlama).
- Lifetime sıfırlama testi geçici dosyadaki store üzerinde çalışır;
  gerçek kalıcı dosyaya dokunulmaz.

Çalıştırma (offscreen):
    QT_QPA_PLATFORM=offscreen ~/workspace/a2-bot-venv/bin/python -m pytest tests/test_pyqt6_stats.py -v
"""
import csv
import json
import logging
import os

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

import core.lifetime_stats as lifetime_stats_mod
from core.lifetime_stats import LifetimeStatsStore
from core.telemetry import TelemetryEngine
from ui import analytics_dialog as ad_mod
from ui.analytics_dialog import AnalyticsDialog


@pytest.fixture(scope="module")
def app():
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture()
def tmp_lifetime(tmp_path, monkeypatch):
    """Lifetime store'u geçici dosyaya yönlendir (gerçek dosyaya yazılmaz)."""
    store = LifetimeStatsStore(path=str(tmp_path / "lifetime_stats.json"))
    monkeypatch.setattr(lifetime_stats_mod, "get_lifetime_stats", lambda path=None: store)
    monkeypatch.setattr(ad_mod, "get_lifetime_stats", lambda path=None: store)
    return store


@pytest.fixture()
def dialog(app, tmp_lifetime):
    dlg = AnalyticsDialog()
    dlg.show()
    yield dlg
    dlg.close()


def _seed_events(tel: TelemetryEngine) -> None:
    tel.record_gather(1, "Altgard", "Dredgion Crash Site", "Aether Bitkisi", 4.6, 90)
    tel.record_gather(2, "Altgard", "Dredgion Crash Site", "Aether Bitkisi", 5.1, 88)
    tel.record_gather(3, "Altgard", "Brusthonin Yolu", "Demir Cevheri", 7.4, 75)
    tel.record_gather(4, "Altgard", "Ishalgen Geçidi", "Aria Kökü", 6.4, 95)
    tel.record_failure(5, reason="timeout")
    tel.record_combat(12.5, outcome="CLEARED")


# ---------------------------------------------------------------- boş durum
def test_empty_state_shows_no_fake_data(dialog, app):
    dialog.refresh_metrics()
    app.processEvents()
    # Tablo: tek bilgilendirme satırı, sahte satır yok.
    assert dialog.table.rowCount() == 1
    assert "henüz kayıt yok" in dialog.table.item(0, 0).text()
    # İçgörüler: "—" / "Henüz veri yok".
    assert dialog.ins_rota.val_lbl.text() == "—"
    assert dialog.ins_kinah.val_lbl.text() == "—"
    assert dialog.ins_rota.sub_lbl.text() == "Henüz veri yok"
    # Kartlar: 0 / bilinmiyor (asla uydurma %100 başarı yok).
    assert dialog.card_gathered.value_lbl.text() == "0"
    assert dialog.card_success.value_lbl.text() == "%—"
    assert dialog.card_success.sub_lbl.text() == "henüz deneme yok"
    # Grafik: kayıt yokken kova listesi boş (boş durum metni çizilir).
    assert dialog.chart_widget.buckets == []
    # Kalıcı kartlar: geçici store boş -> 0.
    assert dialog.card_lt_materials.value_lbl.text() == "0"
    assert dialog.card_lt_kills.value_lbl.text() == "0"
    # Detay paneli son olaylar: boş durum.
    assert dialog.detail_events_list.count() == 1
    assert "henüz kayıt yok" in dialog.detail_events_list.item(0).text()
    # Canlı günlük: açılış kaydı var, sahte hazır satır yok.
    assert dialog.log_view.toPlainText().strip() != ""
    assert "İstatistik sayfası açıldı" in dialog.log_view.toPlainText()


def test_seeded_events_drive_all_panels(dialog, app):
    tel = dialog.telemetry
    _seed_events(tel)
    dialog.refresh_metrics()
    app.processEvents()

    # Başlık: 4 toplama, 1 savaş, 1 hata.
    assert "Savaşlar: 1" in dialog.session_info_lbl.text()
    assert "Hatalar: 1" in dialog.session_info_lbl.text()
    # Kartlar.
    assert dialog.card_gathered.value_lbl.text() == "4"
    assert dialog.card_success.value_lbl.text() == "%80.0"  # 4/5 deneme
    assert dialog.card_walk.value_lbl.text() != "0.00sn"
    # Tablo: 3 grup (Aether x2, Demir, Aria).
    assert dialog.table.rowCount() == 3
    first_res = dialog.table.item(0, 0).text()
    assert first_res == "Aether Bitkisi"  # adet=2 ile en üstte
    assert dialog.table.item(0, 3).text() == "2"
    # Kayıt sayacı: 5 kayıt (4 toplama + 1 başarısız).
    assert ">5<" in dialog.rec_count_lbl.text()
    # İçgörü: en verimli rota fiyat bilgisi yoksa adede göre not düşer.
    assert dialog.ins_rota.val_lbl.text() != "—"
    # Detay: son olaylar listesi dolu (6 olay: 4 toplama + 1 fail + 1 savaş).
    assert dialog.detail_events_list.count() == 6
    # Grafik: 12 dilim dolu.
    assert len(dialog.chart_widget.buckets) == 12
    assert sum(b["count"] for b in dialog.chart_widget.buckets) == 4


def test_event_click_opens_event_detail(dialog, app):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()
    assert dialog.detail_events_list.count() == 6
    item = dialog.detail_events_list.item(0)
    dialog.detail_events_list.itemClicked.emit(item)
    app.processEvents()
    # Olay detayı: KV satırları olay alanlarını gösterir.
    labels = [kl.text() for kl, _vl in dialog.detail_kv_labels]
    assert "Eşleşme güveni" in labels or "Sonuç" in labels
    assert dialog.btn_clear_selection.isVisible()
    assert dialog.btn_clear_selection.text() == "← Özete Dön"
    # Özet dönüşü.
    dialog.btn_clear_selection.click()
    app.processEvents()
    assert dialog._selected_event is None
    assert dialog.detail_title_lbl.text() in ("Bugün Özeti", "Oturum Özeti", "Son 7 Gün Özeti", "Kalıcı Özet")


def test_table_row_click_opens_group_detail_and_group_csv(dialog, app, tmp_path, monkeypatch):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()
    dialog.table.selectRow(0)
    app.processEvents()
    assert dialog._selected_group_key is not None
    assert dialog.detail_kicker_lbl.text() == "SEÇİLİ KAYNAK GRUBU"
    assert dialog.btn_group_csv.isVisible()

    out = tmp_path / "grup.csv"
    monkeypatch.setattr(
        "ui.analytics_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), "CSV Dosyaları (*.csv)"),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dialog.on_export_group_csv()
    app.processEvents()
    assert out.exists()
    text = out.read_text(encoding="utf-8-sig")
    assert "Aether Bitkisi" in text
    assert "Dredgion Crash Site" in text


# ---------------------------------------------------------------- filtreler
def test_region_route_filters_narrow_table(dialog, app):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()
    assert dialog.table.rowCount() == 3

    # Bölge filtresi: tek bölge var; "Tümü" dışında bir şey seçilemez ama
    # rota filtresi grupları daraltır.
    idx = dialog.route_filter.findData("Brusthonin Yolu")
    assert idx >= 0
    dialog.route_filter.setCurrentIndex(idx)
    app.processEvents()
    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 0).text() == "Demir Cevheri"
    assert dialog.table.item(0, 2).text() == "Brusthonin Yolu"

    # Filtre sıfırlanınca 3 grup geri gelir.
    dialog.route_filter.setCurrentIndex(0)
    app.processEvents()
    assert dialog.table.rowCount() == 3


def test_range_buttons_and_target_spin(dialog, app):
    _seed_events(dialog.telemetry)
    dialog._on_range_selected("session")
    app.processEvents()
    assert dialog._active_range == "session"
    assert "OTURUM" in dialog.range_sec_title.text()
    dialog._on_range_selected("7d")
    app.processEvents()
    assert "SON 7 GÜN" in dialog.range_sec_title.text()
    dialog._on_range_selected("all")
    app.processEvents()
    assert "KALICI" in dialog.range_sec_title.text()
    # Kalıcı aralıkta olay listesi tutulmaz.
    assert dialog.detail_events_list.count() == 1
    assert "tutulmaz" in dialog.detail_events_list.item(0).text()
    dialog._on_range_selected("today")

    # Hedef/sa: 40 -> 10 değişince hız kartı ipucu güncellenir.
    dialog.target_rate_spin.setValue(10)
    app.processEvents()
    assert "Hedef 10/sa" in dialog.card_rate.sub_lbl.text()


def test_chart_slice_chip_filters_and_clears(dialog, app):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()
    dialog._on_chart_slice_clicked(0)
    app.processEvents()
    assert dialog._selected_slice == 0
    assert dialog.slice_chip_btn.isVisible()
    assert "Dilim:" in dialog.slice_chip_btn.text()
    # Aynı dilime tekrar tıklayınca filtre kalkar.
    dialog._on_chart_slice_clicked(0)
    app.processEvents()
    assert dialog._selected_slice is None
    assert not dialog.slice_chip_btn.isVisible()


def test_header_sorting_changes_order(dialog, app):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()
    # Varsayılan: adet azalan -> Aether (2) üstte.
    assert dialog.table.item(0, 0).text() == "Aether Bitkisi"
    # KAYNAK TÜRÜ başlığına tıkla: alfabetik artan.
    dialog._on_header_clicked(0)
    app.processEvents()
    assert dialog.table.item(0, 0).text() == "Aether Bitkisi"
    dialog._on_header_clicked(0)  # tersine çevir
    app.processEvents()
    assert dialog.table.item(0, 0).text() == "Demir Cevheri"


# ---------------------------------------------------------------- günlük
def test_live_log_filters_and_clear(dialog, app):
    # Gerçek logger kaydı canlı günlüğe düşer.
    logging.getLogger("Gatherion").warning("test uyarı kaydı")
    app.processEvents()
    assert "test uyarı kaydı" in dialog.log_view.toPlainText()

    dialog._on_log_filter_selected("INFO")
    app.processEvents()
    assert "test uyarı kaydı" not in dialog.log_view.toPlainText()
    assert "İstatistik sayfası açıldı" in dialog.log_view.toPlainText()

    dialog._on_log_filter_selected("WARNING")
    app.processEvents()
    assert "test uyarı kaydı" in dialog.log_view.toPlainText()

    dialog._on_log_filter_selected("all")
    dialog._on_clear_log()
    app.processEvents()
    assert "Günlük temizlendi." in dialog.log_view.toPlainText()
    assert "test uyarı kaydı" not in dialog.log_view.toPlainText()


def test_log_tail_parses_real_bot_log(tmp_path, monkeypatch, app):
    # logs/bot.log kuyruğu: gerçek satırlar ayrıştırılıp gösterilir.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(ad_mod.__file__)))
    log_path = os.path.join(repo_root, "logs", "bot.log")
    existed = os.path.exists(log_path)
    backup = None
    if existed:
        backup = log_path + ".bak-test"
        os.replace(log_path, backup)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[21:15:58] [Gatherion.StateMachine] [INFO] BotWorker thread started.\n")
            f.write("[21:16:01] [Gatherion.HardwareInputPort] [WARNING] NetworkHidAdapter UDP no PONG\n")
        dlg = AnalyticsDialog()
        dlg.show()
        app.processEvents()
        text = dlg.log_view.toPlainText()
        assert "BotWorker thread started." in text
        assert "NetworkHidAdapter UDP no PONG" in text
        dlg.close()
    finally:
        if existed and backup:
            os.replace(backup, log_path)
        elif os.path.exists(log_path):
            os.remove(log_path)


# ------------------------------------------------------- aksiyonlar / sıfırla
def test_all_controls_are_wired(dialog, app):
    """Her buton/filtre gerçek bir handler'a bağlı (alıcı sayısı > 0)."""
    wired = [
        (dialog.btn_live, dialog.btn_live.clicked),
        (dialog.btn_refresh, dialog.btn_refresh.clicked),
        (dialog.btn_export_csv, dialog.btn_export_csv.clicked),
        (dialog.btn_save_json, dialog.btn_save_json.clicked),
        (dialog.btn_reset, dialog.btn_reset.clicked),
        (dialog.btn_reset_lifetime, dialog.btn_reset_lifetime.clicked),
        (dialog.btn_close, dialog.btn_close.clicked),
        (dialog.slice_chip_btn, dialog.slice_chip_btn.clicked),
        (dialog.btn_clear_selection, dialog.btn_clear_selection.clicked),
        (dialog.btn_group_csv, dialog.btn_group_csv.clicked),
        (dialog.btn_clear_log, dialog.btn_clear_log.clicked),
        (dialog.speed_combo, dialog.speed_combo.currentIndexChanged),
        (dialog.region_filter, dialog.region_filter.currentIndexChanged),
        (dialog.route_filter, dialog.route_filter.currentIndexChanged),
        (dialog.target_rate_spin, dialog.target_rate_spin.valueChanged),
    ]
    for widget, signal in wired:
        assert widget.receivers(signal) > 0, f"{widget} has no handler connected"
    # Hız seçenekleri prototip değerleri: 2600 / 1500 / 700 ms.
    assert [dialog.speed_combo.itemData(i) for i in range(3)] == [2600, 1500, 700]
    # Canlı akış düğmesi zamanlayıcıyı gerçekten durdurur/başlatır.
    assert dialog._refresh_timer.isActive()
    dialog.btn_live.click()
    app.processEvents()
    assert not dialog._refresh_timer.isActive()
    assert dialog.btn_live.text() == "Canlı Akışı Başlat"
    dialog.btn_live.click()
    app.processEvents()
    assert dialog._refresh_timer.isActive()


def test_export_csv_and_json_write_real_files(dialog, app, tmp_path, monkeypatch):
    _seed_events(dialog.telemetry)
    dialog.refresh_metrics()
    app.processEvents()

    csv_out = tmp_path / "stats.csv"
    monkeypatch.setattr(
        "ui.analytics_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(csv_out), "CSV Dosyaları (*.csv)"),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    dialog.on_export_csv()
    app.processEvents()
    assert csv_out.exists()
    assert "Aether" in csv_out.read_text(encoding="utf-8", errors="replace")

    json_out = tmp_path / "report.json"
    monkeypatch.setattr(
        "ui.analytics_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(json_out), "JSON Dosyaları (*.json)"),
    )
    dialog.on_save_json()
    app.processEvents()
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(data, (dict, list))


def test_reset_session_keeps_lifetime(dialog, app, tmp_lifetime, monkeypatch):
    _seed_events(dialog.telemetry)
    tmp_lifetime.record_gold(5000)
    dialog.refresh_metrics()
    app.processEvents()
    assert len(dialog.telemetry.gather_events) == 4

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    dialog.on_reset_stats()
    app.processEvents()
    assert len(dialog.telemetry.gather_events) == 0
    assert dialog.table.rowCount() == 1  # boş durum
    # Kalıcı sayaç korunur.
    assert tmp_lifetime.snapshot().get("gold_earned", 0) == 5000


def test_reset_lifetime_requires_confirm_and_wipes(dialog, app, tmp_lifetime, monkeypatch):
    tmp_lifetime.record_material("Aether Bitkisi", count=7)
    tmp_lifetime.record_gold(2500)
    dialog.refresh_metrics()
    app.processEvents()
    assert dialog.card_lt_materials.value_lbl.text() == "7"

    # Önce reddet: veri durur.
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    dialog.on_reset_lifetime_stats()
    app.processEvents()
    assert tmp_lifetime.snapshot().get("materials_total", 0) == 7

    # Onayla: veri silinir.
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dialog.on_reset_lifetime_stats()
    app.processEvents()
    snap = tmp_lifetime.snapshot()
    assert snap.get("materials_total", 0) == 0
    assert snap.get("gold_earned", 0) == 0
    assert dialog.card_lt_materials.value_lbl.text() == "0"
