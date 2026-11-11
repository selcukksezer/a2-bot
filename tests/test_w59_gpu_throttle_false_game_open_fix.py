# -*- coding: utf-8 -*-
"""W59: Oyun kapalıyken core/game_gpu_throttle.py modülünün yanlışlıkla 'oyun açık'
demediğini ve eğitimi bekletmediğini / kısıtlamadığını doğrulayan birim testleri.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import game_gpu_throttle as ggt


def test_browser_and_ide_titles_with_aion2_do_not_trigger_game_open() -> None:
    """Tarayıcı, IDE, terminal ve Gezgin başlıklarında 'Aion 2' geçse bile oyun açık sayılmaz."""
    false_positive_titles = [
        "Aion 2 Wiki - Google Chrome",
        "Aion 2rehberi - Microsoft Edge",
        "aion2_bot.py - Gatherion - Cursor",
        "Aion 2 - Visual Studio Code",
        "Windows PowerShell - aion2",
        "aion2 - Dosya Gezgini",
        "Aion 2 - File Explorer",
        "Gatherion Control Suite - Aion 2",
    ]
    for title in false_positive_titles:
        assert ggt.title_is_aion_game(title) is False, f"Başlık yanlışlıkla oyun sayıldı: {title}"


def test_excluded_executables_never_match_as_aion_game() -> None:
    """Hariç tutulan exe'ler (python, tarayıcılar, IDE'ler, kabuklar) asla oyun süreci sayılmaz."""
    non_game_exes = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Users\user\AppData\Local\Programs\cursor\Cursor.exe",
        r"C:\Windows\explorer.exe",
        r"C:\Python311\python.exe",
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\NCSOFT\Aion2\tools\python.exe",
        r"C:\NCSOFT\Aion\code.exe",
    ]
    for exe in non_game_exes:
        assert ggt.exe_is_aion_game(exe) is False, f"Exe yanlışlıkla oyun sayıldı: {exe}"


def test_find_aion_game_processes_filters_out_non_game_pids_when_game_closed() -> None:
    """Pencere veya süreç taramasından sahte/tarayıcı PID'leri gelse bile exe doğrulaması bunları eler.

    F-1518: find_aion_game_processes() Linux'ta (os.name != "nt") en başta
    boş liste döner, yani yamalanan _pids_from_* yardımcıları hiç çağrılmaz
    ve test boşuna geçerdi. os.name "nt" yapılarak Windows dalı zorlanır,
    böylece exe filtresi gerçekten çalıştırılır ve yama çağrıları doğrulanır.
    """
    fake_pid_exes = {
        1010: r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        2020: r"C:\Users\user\AppData\Local\Programs\cursor\Cursor.exe",
        3030: r"C:\Windows\System32\cmd.exe",
    }
    with (
        patch.object(os, "name", "nt"),
        patch.object(ggt, "_pids_from_window_titles", return_value=[1010, 2020]) as mock_titles,
        patch.object(ggt, "_pids_from_process_names", return_value=[3030]) as mock_procs,
        patch.object(ggt, "_get_exe_for_pid", side_effect=lambda pid, *_a: fake_pid_exes.get(pid, "")) as mock_exe,
    ):
        assert ggt.find_aion_game_processes() == []
        assert ggt.is_game_running() is False
        # The filter helpers must actually have run — otherwise this test
        # would pass vacuously on the early-return path.
        mock_titles.assert_called()
        mock_procs.assert_called()
        assert mock_exe.call_count >= 3


def test_game_gpu_throttle_does_not_apply_when_game_is_closed() -> None:
    """Oyun kapalıyken GameGpuThrottle.game_found False kalır ve hiçbir sürece kısıtlama uygulanmaz."""
    with patch.object(ggt, "_apply_to_pid") as mock_apply:
        with ggt.GameGpuThrottle(finder=lambda: []) as throttle:
            assert throttle.game_found is False
            assert throttle.applied is False
            assert throttle.pids == []
            mock_apply.assert_not_called()


def test_capped_train_batch_preserves_full_batch_when_game_closed() -> None:
    """Oyun kapalıyken (game_found=False) eğitim batch boyutu düşürülmez ve eğitim bekletilmez."""
    with ggt.GameGpuThrottle(finder=lambda: []) as throttle:
        effective_batch = ggt.capped_train_batch(requested_batch=16, game_found=throttle.game_found)
        assert throttle.game_found is False
        assert effective_batch == 16
