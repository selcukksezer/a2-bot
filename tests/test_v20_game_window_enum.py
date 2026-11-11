# -*- coding: utf-8 -*-
"""
v20-hotfix/game-window-enum — EnumWindows numaralandırma düzeltmesi testleri.

Kök neden (canlı log, v20, 2026-11-11 16:54:19):
    [Gatherion.Watchdog] [WARNING] Game window not found;
    visible top-level windows (0): []
Oyun AÇIKKEN, bot + tarayıcı da açıkken 0 pencere imkânsızdır.

Kök neden (kod kanıtı): `core/watchdog.py::_find_game_window` iki
katmanda da `wintypes.WNDENUMPROC` okuyordu — ama ctypes.wintypes'te
WNDENUMPROC YOKTUR (yalnızca HANDLE türlerini tanımlar; geri-çağrı tipi
ctypes.WINFUNCTYPE ile kurulur). AttributeError, EnumWindows hiç
çağrılmadan DEBUG'a gömülüyordu ("Window title fallback failed"),
visible_titles hep boş kalıyordu. v19/S147'den (fb0cd14) beri ölüydü;
v20 tanısı "(0): []" ile ortaya çıkardı.

Düzeltme (core/watchdog.py):
1) Geri-çağrı tipi `ctypes.WINFUNCTYPE` ile kurulur (modül düzeyi
   `_WNDENUMPROC`; Linux'ta None — Windows dalı orada çalışmaz).
2) Numaralandırma hatası WARNING'e çıkar (istisna metniyle, seyreltilmiş
   tanı üzerinden); DEBUG'a gömülmez.
3) Gezilen pencere sayacı (visited) + EnumWindows dönüş değeri kontrolü.
4) Geri-çağrı nesnesi kalıcı referansta tutulur (self._enum_cb_ref).
5) EnumWindows 0 pencere gezerse GetTopWindow/GetNextWindow zincir yedeği.

NOT: pytest bu VM'de bu görevde KOŞULMAZ (görev kuralı); dosya yalnızca
yazıldı, derlemesi py_compile ile doğrulandı. Windows yolu (ctypes)
Linux'ta çalışmaz — user32 + WINFUNCTYPE mocklanır.
"""

import contextlib
import ctypes
import logging
import sys
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest

from core import watchdog as watchdog_mod
from core.watchdog import BotWatchdog

# core.watchdog'ta TARGETS adında bir sembol yok; hedef başlıklar
# BotWatchdog'un varsayılanıdır (watchdog.py __init__).
TARGETS = ["AION 2", "Aion 2", "AION", "Aion"]


def _make_fake_user32(titles, *, enum_mode="normal", chain_hwnds=()):
    """ctypes.windll.user32 taklidi.

    titles: {hwnd:int -> title:str} — GetWindowTextW'un döndürdüğü başlıklar.
    enum_mode:
      "normal" — EnumWindows her hwnd için callback'i çağırır, True döner.
      "raises" — EnumWindows RuntimeError("boom") atar.
      "false" — EnumWindows callback'i hiç çağırmadan False döner.
      "empty" — EnumWindows callback'i çağırmadan True döner.
    chain_hwnds: GetTopWindow/GetNextWindow zincirinin döndüreceği hwnd listesi.
    """
    fake = MagicMock(name="user32")
    fake.FindWindowW.side_effect = lambda _cls, _title: 0
    fake.IsWindowVisible.side_effect = lambda _hwnd: True

    def _get_text(hwnd, buf, _n):
        text = titles.get(int(hwnd), "")
        buf.value = text
        return len(text)

    fake.GetWindowTextW.side_effect = _get_text
    # Süreç yedeği (katman 3): pid 0 kalırsa callback erken True döner.
    fake.GetWindowThreadProcessId.side_effect = lambda _hwnd, _pid: 1

    def _enum(callback, _lparam):
        if enum_mode == "raises":
            raise RuntimeError("boom")
        if enum_mode == "false":
            return False
        if enum_mode == "empty":
            return True
        for hwnd in list(titles):
            if callback(hwnd, 0) is False:
                return False
        return True

    fake.EnumWindows.side_effect = _enum

    _chain = list(chain_hwnds)
    fake.GetTopWindow.side_effect = lambda _hwnd: _chain[0] if _chain else 0

    def _get_next(hwnd, _cmd):
        try:
            idx = _chain.index(int(hwnd))
        except ValueError:
            return 0
        return _chain[idx + 1] if idx + 1 < len(_chain) else 0

    fake.GetNextWindow.side_effect = _get_next
    return fake


@contextlib.contextmanager
def _windows_env(fake_user32):
    """Win32 ortamını taklit et: platform + windll + WINFUNCTYPE.

    _WNDENUMPROC Linux'ta None'dur; testte pass-through fabrika ile
    yamalanır (gerçek Windows'ta WINFUNCTYPE, callback'i aynen sarar).
    """
    fake_windll = SimpleNamespace(
        user32=fake_user32,
        kernel32=MagicMock(name="kernel32"),
    )
    with (
        mock.patch.object(sys, "platform", "win32"),
        mock.patch.object(ctypes, "windll", fake_windll, create=True),
        mock.patch.object(watchdog_mod, "_WNDENUMPROC", lambda fn: fn),
    ):
        yield


def _watchdog():
    return BotWatchdog(target_window_titles=list(TARGETS))


def test_enum_collects_titles_and_diag_reports_them(caplog):
    """Callback başlıkları toplar, tanı onları + sayacı raporlar."""
    titles = {0x1001: "Google Chrome", 0x1002: "Discord", 0x1003: "Notepad"}
    fake = _make_fake_user32(titles)
    watchdog = _watchdog()
    with _windows_env(fake), caplog.at_level(
        logging.WARNING, logger="Gatherion.Watchdog"
    ):
        assert watchdog.is_game_window_present() is False
    assert fake.EnumWindows.call_count >= 1
    assert "enumerated 3 windows" in caplog.text
    assert "visible top-level windows (3)" in caplog.text
    assert "Google Chrome" in caplog.text
    assert "Discord" in caplog.text
    assert "Notepad" in caplog.text


def test_enum_exception_surfaces_as_warning(caplog):
    """EnumWindows atarsa hata WARNING'e çıkar (DEBUG'a gömülmez)."""
    fake = _make_fake_user32({}, enum_mode="raises")
    watchdog = _watchdog()
    with _windows_env(fake), caplog.at_level(
        logging.WARNING, logger="Gatherion.Watchdog"
    ):
        assert watchdog.is_game_window_present() is False
    assert "Game window enumeration failed" in caplog.text
    assert "boom" in caplog.text
    assert "enumerated 0 windows" in caplog.text


def test_enum_returns_false_is_warning(caplog):
    """EnumWindows False dönerse tanı bunu + 0 sayacı gösterir."""
    fake = _make_fake_user32({}, enum_mode="false")
    watchdog = _watchdog()
    with _windows_env(fake), caplog.at_level(
        logging.WARNING, logger="Gatherion.Watchdog"
    ):
        assert watchdog.is_game_window_present() is False
    assert "EnumWindows returned False" in caplog.text
    assert "enumerated 0 windows" in caplog.text


def test_top_window_chain_fallback_finds_game():
    """EnumWindows 0 gezerse zincir yedeği 'AION 2'yi bulur."""
    titles = {0x2001: "AION 2", 0x2002: "Discord"}
    fake = _make_fake_user32(titles, enum_mode="empty", chain_hwnds=[0x2001, 0x2002])
    watchdog = _watchdog()
    with _windows_env(fake):
        assert watchdog.is_game_window_present() is True
    assert fake.GetTopWindow.call_count >= 1


def test_enum_callback_kept_alive():
    """Geri-çağrı nesnesi kalıcı referansta tutulur (GC çökmesi yok)."""
    titles = {0x1001: "Google Chrome"}
    fake = _make_fake_user32(titles)
    watchdog = _watchdog()
    with _windows_env(fake):
        assert watchdog.is_game_window_present() is False
    assert watchdog._enum_cb_ref is not None


def test_wndenumproc_is_none_off_windows():
    """Linux'ta modül düzeyi _WNDENUMPROC None'dur (WINFUNCTYPE yok)."""
    if sys.platform.startswith("win"):
        pytest.skip("Windows-only guard check")
    assert watchdog_mod._WNDENUMPROC is None
