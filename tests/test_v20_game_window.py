# -*- coding: utf-8 -*-
"""
v20/game-window — pencere tanıma düzeltmesi testleri.

Sorun (canlı log, v19): Oyun AÇIK ve DXGI yakalayabiliyorken tüm oturum
boyunca her ~2 sn "[Gatherion.Watchdog] [WARNING] Game window not found
(game may have crashed)." — detektör sürekli False dönüyordu.

Kök neden: _find_game_window yalnızca (1) FindWindowW tam başlık
eşleşmesi ve (2) S147 önek eşleşmesi (text.startswith(hedef)) deniyordu.
Canlı AION 2 penceresinin gerçek başlığı bu dört tam başlıkla ne tam ne
de önek eşleşmesi yapıyor.

Düzeltme (core/watchdog.py):
1) Katmanlı eşleşme: tam → önek → süreç exe adı ("aion" içeren) →
   büyük/küçük harf duyarsız alt-dize. Eşleştirme saf fonksiyona
   (match_game_window_title / game_process_exe_matches) çıkarıldı.
2) Tanı: detektör False döndüğünde o anki ilk ~10 görünür üst-düzey
   pencere başlığı loga yazılır — False'a ilk geçişte ve sonra en fazla
   30 sn'de bir. Gerçek başlık TAHMİN EDİLMEZ; canlı logda görülür.

NOT: pytest bu VM'de bu görevde KOŞULMAZ (görev kuralı); dosya yalnızca
yazıldı, derlemesi py_compile ile doğrulandı. Windows yolundaki
(_find_game_window / tanı) davranış canlı PC'de doğrulanmalıdır.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.watchdog import (
    BotWatchdog,
    _GAME_WINDOW_DIAG_INTERVAL_SEC,
    game_process_exe_matches,
    match_game_window_title,
)

TARGETS = ["AION 2", "Aion 2", "AION", "Aion"]


# --- 1) Saf eşleştirme: tam / önek / alt-dize -------------------------------


def test_exact_match_wins():
    assert match_game_window_title("AION 2", TARGETS) == ("AION 2", 0)
    assert match_game_window_title("Aion", TARGETS) == ("Aion", 0)


def test_prefix_match_s147():
    # S147: sürüm/sunucu eki olan başlık önekiyle tutar.
    assert match_game_window_title("AION 2 (64-bit)", TARGETS) == ("AION 2", 1)
    assert match_game_window_title("Aion 2 - NA", TARGETS) == ("Aion 2", 1)


def test_substring_match_case_insensitive_v20():
    # v20: başlık "aion" ile başlamıyorsa bile içeriyorsa tutar.
    assert match_game_window_title("aion2", TARGETS) == ("AION", 2)
    assert match_game_window_title("My AION 2 Client", TARGETS) == ("AION 2", 2)
    # "AION2.EXE - Game" "AION" ile BAŞLAR → önek (1), alt-dize değil.
    assert match_game_window_title("AION2.EXE - Game", TARGETS) == ("AION", 1)
    assert match_game_window_title("My AION2 Client", TARGETS) == ("AION", 2)


def test_prefix_beats_substring():
    # "Aion 2 Rehberi" — "Aion 2" öneki (1), "AION" alt-dizesinden (2)
    # önceliklidir; tarayıcı sekmesi gerçek pencereden önce gelmez.
    assert match_game_window_title("Aion 2 Rehberi", TARGETS) == ("Aion 2", 1)


def test_no_match():
    assert match_game_window_title("Google Chrome", TARGETS) is None
    assert match_game_window_title("", TARGETS) is None
    assert match_game_window_title("AION 2", []) is None
    assert match_game_window_title("AION 2", [""]) is None


# --- 2) Süreç yedeği: exe adı ----------------------------------------------


def test_process_exe_matches():
    assert game_process_exe_matches("AION2.exe") is True
    assert game_process_exe_matches("aion.exe") is True
    assert game_process_exe_matches("Aion2_x64.exe") is True
    assert game_process_exe_matches("chrome.exe") is False
    assert game_process_exe_matches("NCLauncher2.exe") is False
    assert game_process_exe_matches("") is False


# --- 3) Tanı seyretme (Windows dışı: erken dönüş, log yok) ------------------


def test_diag_throttle_state_defaults():
    watchdog = BotWatchdog(target_window_titles=list(TARGETS))
    assert watchdog._last_game_window_present is None
    assert watchdog._last_game_window_diag_ts == 0.0
    assert _GAME_WINDOW_DIAG_INTERVAL_SEC == 30.0


def test_diag_noop_off_windows(caplog):
    # VM Linux: tanı Windows dışında hiçbir şey loglamaz.
    watchdog = BotWatchdog(target_window_titles=list(TARGETS))
    with caplog.at_level(logging.WARNING, logger="Gatherion.Watchdog"):
        watchdog._maybe_log_game_window_diagnostic(["AION 2", "Chrome"])
    assert "visible top-level windows" not in caplog.text


# --- 4) Mevcut davranış korunur (Windows dışı) ------------------------------


def test_is_game_window_present_unknown_off_windows():
    watchdog = BotWatchdog(target_window_titles=list(TARGETS))
    assert watchdog.is_game_window_present() is None


def test_is_game_window_present_unknown_empty_targets():
    watchdog = BotWatchdog(target_window_titles=[])
    assert watchdog.is_game_window_present() is None


# --- 5) Windows-only: canlı PC doğrulaması ---------------------------------
# Aşağıdaki testler yalnızca Windows'ta çalışır; VM'de atlanır.
# Canlı PC'de (oyun açıkken) beklenen: is_game_window_present() is True
# ve logda "Game window not found" + tanı satırı GÖRÜLMEMELİ.
# Oyun kapalıyken beklenen: False + tanı satırında gerçek başlıklar.


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="Windows-only live check"
)
def test_live_game_window_present_when_game_open():
    watchdog = BotWatchdog(target_window_titles=list(TARGETS))
    assert watchdog.is_game_window_present() is True


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="Windows-only live check"
)
def test_live_diag_lists_visible_titles_when_game_closed(caplog):
    watchdog = BotWatchdog(target_window_titles=["__no_such_game__"])
    with caplog.at_level(logging.WARNING, logger="Gatherion.Watchdog"):
        assert watchdog.is_game_window_present() is False
    assert "visible top-level windows" in caplog.text
