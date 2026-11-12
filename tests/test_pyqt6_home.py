# -*- coding: utf-8 -*-
"""Anasayfa (W1): HTML prototip -> PyQt6 dönüşümünün duman ve bağlantı testleri.

Kapsam: ui/app_shell.py içindeki _HomePage + GatherionShell navigasyonu.
Prototip: gatherion-anasayfa-prototipi.html (21 butonluk komut ızgarası,
üst bar, sol menü, durum satırı, sayılar şeridi, minimap/rota, sistem
kartı anahtarları, son tespitler, canlı günlük).

Kural: sahte veri yok — her buton gerçek bir işleyiciye bağlı olmalı,
karşılığı olmayan buton testte "bağlı değil" diye işaretlenir.
"""

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QSizePolicy, QWidget

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.schema import SettingsManager  # noqa: E402
from ui.app_shell import GatherionShell, _HomePage  # noqa: E402
from ui.f9_menu import F9Menu  # noqa: E402

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

# Prototipteki 21 komut, sırasıyla.
PROTOTYPE_GRID = [
    "Başlat",
    "Devam Et",
    "Duraklat",
    "Toplama",
    "Savaş",
    "Keşif",
    "Oto-Kayıt",
    "Konum Ekle",
    "Minimap",
    "Rota",
    "Radar",
    "Harita",
    "YOLO",
    "İstatistik",
    "Kalibrasyon",
    "Eğit",
    "Ayarlar",
    "Günlük",
    "Kılavuz",
    "Tanılama",
    "Çıkış",
]

# Izgara butonu -> F9Menu.action_triggered ile beklenen eylem kimliği.
# "Rota" alt menü açar (sinyal yaymaz), "Günlük"/"Kılavuz"/"Tanılama"
# sayfa açar ya da panoya yazar (sinyal yaymaz) — onlar ayrı test edilir.
GRID_ACTION_EMIT = {
    "Devam Et": "resume",
    "Toplama": "route",
    "Savaş": "combat",
    "Keşif": "explorer",
    "Oto-Kayıt": "auto_record",
    "Konum Ekle": "manual_record",
    "Minimap": "minimap_auto_record",
    "Radar": "radar",
    "İstatistik": "analytics",
    "Kalibrasyon": "calibrate",
    "Eğit": "yolo_factory",
    "Ayarlar": "settings",
    "Çıkış": "exit",
}


class _FakeWorker:
    """is_paused üzerinden çalışan/duraklı durumunu veren en küçük worker."""

    def __init__(self, paused: bool = True) -> None:
        self.is_paused = paused
        self.state = "PAUSED" if paused else "WALKING"


class _FakeRouteMgr:
    """route_mgr'in anasayfanın kullandığı üç metodunu verir."""

    def __init__(self) -> None:
        self.active_region = "Altgard"
        self.active_route = "Dredgion Crash Site"

    def get_routes(self, region=None):
        return ["Rota A", "Rota B", "Dredgion Crash Site"]

    def get_route_stats(self, region=None, route=None, rest_duration_sec=None):
        return (195, 0)

    def get_route_nodes(self, region_name=None, route_name=None, **_kw):
        return []


class _FakeMobileServer:
    def __init__(self) -> None:
        self._running = False
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> str:
        self._running = True
        self.start_calls += 1
        return "http://127.0.0.1:8765"

    def stop(self) -> None:
        self._running = False
        self.stop_calls += 1


def _grid_texts(home: _HomePage) -> list:
    grid = home._action_grid
    texts = []
    for row in range(3):
        for col in range(7):
            item = grid.itemAtPosition(row, col)
            assert item is not None, f"ızgara ({row},{col}) boş"
            widget = item.widget()
            assert widget is not None, f"ızgara ({row},{col}) widgetsiz"
            texts.append(widget.text())
    return texts


def _grid_button(home: _HomePage, label: str):
    for button in home.findChildren(QPushButton):
        if button.text() == label and home._action_grid.indexOf(button) != -1:
            return button
    raise AssertionError(f"ızgarada {label!r} butonu yok")


@pytest.fixture
def shell_with_menu(tmp_path):
    """Gerçek F9Menu + gerçek SettingsManager (geçici dosya) ile shell."""
    settings_path = tmp_path / "settings.json"
    mgr = SettingsManager(settings_path=str(settings_path))
    menu = F9Menu(route_manager=None)
    shell = GatherionShell(
        settings_mgr=mgr,
        route_mgr=None,
        f9_menu=menu,
        worker=None,
    )
    shell.show()
    yield shell, menu, mgr
    shell._home_timer.stop()
    shell._log_timer.stop()
    shell.close()


def test_home_grid_matches_prototype_order(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    assert isinstance(home, _HomePage)
    assert _grid_texts(home) == PROTOTYPE_GRID
    # Menü prototipte yok: ızgarada görünmez, akor kaydı durur.
    assert "Menü" not in _grid_texts(home)
    assert home._chord_buttons["toggle_menu_f9"].isVisible() is False


def test_home_grid_buttons_all_connected(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    grid = home._action_grid
    assert grid.count() == 21
    for index in range(grid.count()):
        button = grid.itemAt(index).widget()
        assert button is not None
        assert button.receivers(button.clicked) > 0, (
            f"{button.text()!r} butonunun clicked işleyicisi yok"
        )


def test_home_grid_actions_reach_f9_menu(shell_with_menu):
    shell, menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    emitted = []
    menu.action_triggered.connect(emitted.append)
    for label, action_id in GRID_ACTION_EMIT.items():
        emitted.clear()
        QTest.mouseClick(_grid_button(home, label), Qt.MouseButton.LeftButton)
        assert emitted == [action_id], (
            f"{label!r} -> {action_id!r} beklenirken {emitted!r} geldi"
        )


def test_home_grid_page_shortcuts_switch_tabs(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    QTest.mouseClick(_grid_button(home, "Harita"), Qt.MouseButton.LeftButton)
    assert shell.current_key() == "map"
    QTest.mouseClick(_grid_button(home, "YOLO"), Qt.MouseButton.LeftButton)
    assert shell.current_key() == "yolo"


def test_home_grid_rota_opens_submenu_and_gunluk_opens_logs(shell_with_menu):
    shell, menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    QTest.mouseClick(_grid_button(home, "Rota"), Qt.MouseButton.LeftButton)
    assert menu._route_submenu is not None
    QTest.mouseClick(_grid_button(home, "Günlük"), Qt.MouseButton.LeftButton)
    assert shell.current_key() == "logs"
    QTest.mouseClick(_grid_button(home, "Kılavuz"), Qt.MouseButton.LeftButton)
    assert shell.current_key() == "guide"


def test_status_row_buttons(shell_with_menu):
    shell, menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    calls = {"pause": 0, "record": 0}
    shell.set_chord_handlers(
        record_node=lambda: calls.__setitem__("record", calls["record"] + 1),
        pause_resume=lambda: calls.__setitem__("pause", calls["pause"] + 1),
    )
    # Başlat/Duraklat (üst bar) akoru çağırır.
    QTest.mouseClick(home.btn_status_run, Qt.MouseButton.LeftButton)
    assert calls["pause"] == 1
    # Konumu Kaydet F8 kaydını çağırır.
    QTest.mouseClick(home.btn_status_save, Qt.MouseButton.LeftButton)
    assert calls["record"] == 1
    # Mobil Bağlan mobil sayfayı açar.
    QTest.mouseClick(home.btn_status_mobile, Qt.MouseButton.LeftButton)
    assert shell.current_key() == "mobile"


def test_pause_only_button_pauses_only_when_running(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    calls = {"pause": 0}
    shell.set_chord_handlers(
        pause_resume=lambda: calls.__setitem__("pause", calls["pause"] + 1),
    )
    pause_btn = home._chord_buttons["pause_only"]
    # Duraklıyken buton kapalıdır (worker yok -> çalışmıyor).
    assert pause_btn.isEnabled() is False
    # Çalışırken açılır ve tek tıklama duraklatır.
    shell._worker = _FakeWorker(paused=False)
    shell._publish_home()
    assert home._running is True
    assert pause_btn.isEnabled() is True
    QTest.mouseClick(pause_btn, Qt.MouseButton.LeftButton)
    assert calls["pause"] == 1
    # Duraklıyken _run_chord("pause_only") hiçbir şey yapmaz.
    shell._worker = _FakeWorker(paused=True)
    shell._publish_home()
    assert pause_btn.isEnabled() is False
    shell._run_chord("pause_only")
    assert calls["pause"] == 1


def test_topbar_and_rail_match_prototype(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    assert shell.findChild(type(shell.top_page_meta), "TopBrand").text() == "GATHERION"
    assert "Anasayfa" in shell.top_page_meta.text()
    perf = shell.top_perf_label.text()
    for token in ("FPS", "GPU", "CPU", "LAT", "MOD"):
        assert token in perf, f"üst bar performansında {token} yok: {perf}"
    assert shell.rail_labels() == [
        "Anasayfa",
        "Radar",
        "Harita",
        "YOLO",
        "Kalibrasyon",
        "İstatistik",
        "Mobil",
        "Kılavuz",
        "Log",
        "Ayarlar",
    ]


def test_log_card_filters_present(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    labels = {b.text() for b in shell.findChildren(QPushButton)}
    assert {"Tümü", "INFO", "Önizleme", "Temizle"} <= labels
    wrap = shell.findChild(QWidget, "ShellLogWrap")
    assert wrap is not None
    assert wrap.minimumWidth() == 360 and wrap.maximumWidth() == 360
    assert shell.log_view.minimumHeight() >= 120
    assert shell.log_view.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding


def test_avoid_combat_toggle_writes_settings(shell_with_menu, tmp_path):
    shell, _menu, mgr = shell_with_menu
    home = shell.pages["home"]
    assert mgr.settings.route.fight_back is True
    QTest.mouseClick(home.tog_avoid_combat, Qt.MouseButton.LeftButton)
    assert home.tog_avoid_combat.isChecked() is True
    assert mgr.settings.route.fight_back is False
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["route"]["fight_back"] is False
    QTest.mouseClick(home.tog_avoid_combat, Qt.MouseButton.LeftButton)
    assert mgr.settings.route.fight_back is True


def test_auto_record_toggle_switches_mode_without_f8(shell_with_menu):
    shell, menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    emitted = []
    menu.action_triggered.connect(emitted.append)
    f8_calls = []
    shell.set_chord_handlers(record_node=lambda: f8_calls.append("f8"))
    QTest.mouseClick(home.tog_auto_rec, Qt.MouseButton.LeftButton)
    assert home.tog_auto_rec.isChecked() is True
    assert emitted == ["auto_record"]
    assert menu._active_mode == "auto_record"
    emitted.clear()
    QTest.mouseClick(home.tog_auto_rec, Qt.MouseButton.LeftButton)
    assert home.tog_auto_rec.isChecked() is False
    assert emitted == ["manual_record"]
    assert menu._active_mode == "manual_record"
    assert f8_calls == [], "Oto-Kayıt kapatılırken F8 kaydı tetiklenmemeli"


def test_mobile_panel_toggle_starts_and_stops_server(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    fake = _FakeMobileServer()
    shell._mobile_server = fake
    shell._on_home_toggle("mobile_panel", True)
    assert fake.start_calls == 1
    assert fake.is_running is True
    shell._publish_home()
    assert home.tog_mobile_panel.isChecked() is True
    shell._on_home_toggle("mobile_panel", False)
    assert fake.stop_calls == 1
    assert fake.is_running is False
    shell._publish_home()
    assert home.tog_mobile_panel.isChecked() is False


def test_mobile_panel_toggle_without_server_is_noop(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    shell._mobile_server = None
    QTest.mouseClick(home.tog_mobile_panel, Qt.MouseButton.LeftButton)
    # Sunucu yok: sahte açık durumu gösterilmez, yayın geri kapatır.
    assert home.tog_mobile_panel.isChecked() is False


def test_route_number_in_snapshot_and_status_line(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    shell.route_mgr = _FakeRouteMgr()
    shell.get_active_region = lambda: "Altgard"
    shell.get_active_route = lambda: "Dredgion Crash Site"
    snap = shell._home_snapshot()
    assert snap["route_no"] == "3"
    assert snap["active"] == "195"
    home.apply_snapshot(
        {
            "state": "Duraklatıldı",
            "region": "Altgard",
            "route": "Dredgion Crash Site",
            "route_no": "3",
            "active": "195",
        }
    )
    assert "Rota #3" in home.route_summary_label.text()
    assert "195 node" in home.route_summary_label.text()


def test_home_empty_states_show_no_fake_data(shell_with_menu):
    shell, _menu, _mgr = shell_with_menu
    home = shell.pages["home"]
    home.update_detections([], None)
    assert "Henüz tespit yok" in home.detections_layout.itemAt(0).widget().text()
    home.update_route_steps("Altgard", "Dredgion Crash Site", [], active_idx=0)
    assert home.minimap_pbar.value() == 0
