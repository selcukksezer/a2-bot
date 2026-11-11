# -*- coding: utf-8 -*-
"""v20 istatistik kaydı düzeltmeleri (fix/v20-istatistik-kayit).

Kapsam:
- R1: "Bugün Özeti"ndeki "Öldürülen canavar" satırı artık aralığın
  kill'ini gösterir; kalıcı kill (89) yalnızca "Kalıcı" aralığında
  "(kalıcı)" etiketiyle görünür.
- R2: `TelemetryEngine` geçmiş oturum olaylarını
  `logs/telemetry_history.json` dosyasından geri yükler; "Bugün" /
  "Son 7 Gün" restart sonrası da bugünün daha önceki oturumlarını kapsar.
  Geçmiş olaylar AYRI listelerdedir — canlı oturum listeleri ve
  kümülatif sayaçlar kirletilmez.
- R3: Ayarlar penceresinden açılan bağımsız istatistik diyaloğu canlı
  telemetri motorunu alır (`_resolve_analytics_telemetry`).

Çalıştırma (offscreen — testler KOŞULMAZ, ana ajan koşar):
    QT_QPA_PLATFORM=offscreen ~/workspace/a2-bot-venv/bin/python -m pytest tests/test_v20_istatistik_kayit.py -v
"""
import json
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

import core.lifetime_stats as lifetime_stats_mod
from core.lifetime_stats import LifetimeStatsStore
from core.telemetry import TelemetryEngine
from ui import analytics_dialog as ad_mod
from ui.analytics_dialog import AnalyticsDialog
from ui.settings_window import _resolve_analytics_telemetry


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


def _gather_event(node_id: int, ts: float, resource: str = "Targena") -> dict:
    return {
        "timestamp": ts,
        "iso_time": "2026-11-11T10:00:00",
        "node_id": node_id,
        "region": "Altgard",
        "route": "All",
        "resource_type": resource,
        "walk_duration_sec": 5.0,
        "match_pct": 90,
        "items_gathered": 3,
    }


def _history_file(tmp_path, snapshots) -> str:
    p = tmp_path / "telemetry_history.json"
    p.write_text(json.dumps(snapshots, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _snapshot(gathers, failures=None, combats=None) -> dict:
    return {
        "saved_at": "2026-11-11T10:00:00",
        "session_start": "2026-11-11T09:00:00",
        "metrics": {},
        "recent_gathers": gathers,
        "recent_failures": failures or [],
        "recent_combats": combats or [],
    }


# ---------------------------------------------------------------- R2: geçmiş yükleme
def test_history_load_restores_today_events_deduped(tmp_path):
    now = time.time()
    ev_today = _gather_event(7, now - 3600)
    ev_today2 = _gather_event(9, now - 1800)
    ev_old = _gather_event(8, now - 8 * 86400)
    # Aynı olay iki snapshot'ta geçiyor (otomatik kayıt her 10 dk'da yazar).
    path = _history_file(
        tmp_path,
        [_snapshot([ev_today]), _snapshot([ev_today, ev_today2, ev_old])],
    )
    tel = TelemetryEngine(history_path=path)
    assert len(tel.history_gather_events) == 2
    assert {e["node_id"] for e in tel.history_gather_events} == {7, 9}
    # Canlı oturum listeleri/sayaçları kirlenmez:
    assert tel.gather_events == []
    assert tel._gathered_total == 0


def test_history_load_missing_file_is_noop(tmp_path):
    tel = TelemetryEngine(history_path=str(tmp_path / "yok.json"))
    assert tel.history_gather_events == []
    assert tel.history_failure_events == []
    assert tel.history_combat_events == []


def test_history_load_corrupt_file_is_noop(tmp_path):
    p = tmp_path / "bozuk.json"
    p.write_text("{bu json değil", encoding="utf-8")
    tel = TelemetryEngine(history_path=str(p))
    assert tel.history_gather_events == []


# ---------------------------------------------------------------- R1: aralık kill'i
def test_today_summary_shows_range_kills_not_lifetime(app, tmp_lifetime):
    # Kalıcı: 3 kill (a/b/c ayrı anahtar — debounce'a takılmaz).
    for key in ("mob-a", "mob-b", "mob-c"):
        assert tmp_lifetime.record_monster_kill(mob_key=key) is True
    assert tmp_lifetime.snapshot()["monsters_killed"] == 3

    tel = TelemetryEngine()
    tel.record_combat(12.0, outcome="CLEARED", mob_key="m1")
    tel.record_combat(9.0, outcome="CLEARED", mob_key="m2")
    tel.record_combat(4.0, outcome="DIED", mob_key="")
    # record_combat CLEARED'leri kalıcıya da yazar: 3 + 2 = 5.
    assert tmp_lifetime.snapshot()["monsters_killed"] == 5

    dlg = AnalyticsDialog(telemetry=tel)
    dlg.show()
    try:
        dlg._active_range = "today"
        dlg.refresh_metrics()
        app.processEvents()
        key_lbl, val_lbl = dlg.detail_kv_labels[4]
        assert key_lbl.text() == "Öldürülen canavar"
        assert val_lbl.text() == "2"  # yalnızca 2 CLEARED; DIED sayılmaz, kalıcı 5 sızmaz.

        dlg._active_range = "all"
        dlg.refresh_metrics()
        app.processEvents()
        key_lbl, val_lbl = dlg.detail_kv_labels[4]
        assert key_lbl.text() == "Öldürülen canavar (kalıcı)"
        assert val_lbl.text() == "5"
    finally:
        dlg.close()


def test_merged_events_dedupes_live_vs_history():
    ev = _gather_event(7, 1000.0)
    merged = AnalyticsDialog._merged_events([ev], [dict(ev)], "gather")
    assert len(merged) == 1
    merged2 = AnalyticsDialog._merged_events([], [_gather_event(7, 1000.0), _gather_event(9, 2000.0)], "gather")
    assert [e["node_id"] for e in merged2] == [7, 9]


# ---------------------------------------------------------------- R2: bugün metriği
def test_history_events_count_in_today_metrics(app, tmp_path, tmp_lifetime):
    now = time.time()
    path = _history_file(tmp_path, [_snapshot([_gather_event(7, now - 3600)])])
    tel = TelemetryEngine(history_path=path)
    dlg = AnalyticsDialog(telemetry=tel)
    dlg.show()
    try:
        dlg._active_range = "today"
        dlg.refresh_metrics()
        app.processEvents()
        assert dlg.card_gathered.value_lbl.text() == "1"
        # Oturum aralığı geçmişi saymaz — canlı sayaçlar 0.
        dlg._active_range = "session"
        dlg.refresh_metrics()
        app.processEvents()
        assert dlg.card_gathered.value_lbl.text() == "0"
    finally:
        dlg.close()


# ---------------------------------------------------------------- R3: çözümleyici
def test_resolve_analytics_telemetry_prefers_live():
    tel = object()
    assert _resolve_analytics_telemetry(SimpleNamespace(_live_telemetry=tel)) is tel
    host = SimpleNamespace(_live_telemetry=None, _shell=SimpleNamespace(_telemetry=tel))
    assert _resolve_analytics_telemetry(host) is tel
    assert _resolve_analytics_telemetry(SimpleNamespace()) is None
