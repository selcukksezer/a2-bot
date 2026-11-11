# -*- coding: utf-8 -*-
"""Oturum bazli log dosyalari (v20/session-logs).

Selcuk'un istegi (2026-11-11): tek upuzun ``bot.log`` yerine bot her
acildiginda yeni bir oturum dosyasi acilsin; logun sonu gorulebilsin.

Bicim: ``logs/session_YYYY-MM-DD_HHMMSS.log`` (yerel saat; main.py'deki
LOG_DATEFMT ile ayni saat dilimi). Her baslatmada yeni dosya acilir;
en son 20 oturum tutulur, eskileri budanir.

Bu modul yalnizca stdlib'e dayanir (Qt yok, repo ici import yok);
main.py import aninda, ui ve core'dan guvenle import edilebilir.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

SESSION_LOG_PREFIX = "session_"
SESSION_LOG_SUFFIX = ".log"
SESSION_LOG_KEEP = 20
_LEGACY_LOG_NAME = "bot.log"

_SESSION_NAME_RE = re.compile(r"^session_\d{4}-\d{2}-\d{2}_\d{6}\.log$")


def session_log_filename(now: Optional[datetime] = None) -> str:
    """Oturum dosyasi adi: ``session_YYYY-MM-DD_HHMMSS.log``.

    ``now`` yalnizca testler icin verilir (zaman mock'u); verilmezse
    o anki yerel saat kullanilir.
    """
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return f"{SESSION_LOG_PREFIX}{stamp}{SESSION_LOG_SUFFIX}"


def resolve_session_log_path(
    log_dir: Union[str, Path], now: Optional[datetime] = None
) -> Path:
    """Bu baslatma icin oturum dosyasi yolu (henuz olusmamissa bile)."""
    return Path(log_dir) / session_log_filename(now)


def list_session_logs(log_dir: Union[str, Path]) -> List[Path]:
    """Oturum dosyalari, ada gore sirali (ad kronolojiktir)."""
    try:
        directory = Path(log_dir)
        if not directory.is_dir():
            return []
        found = [
            entry
            for entry in directory.iterdir()
            if entry.is_file() and _SESSION_NAME_RE.match(entry.name)
        ]
    except Exception:
        return []
    return sorted(found, key=lambda entry: entry.name)


def current_session_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
    """En yeni oturum dosyasi; yoksa None. Hicbir zaman hata firlatmaz."""
    logs = list_session_logs(log_dir)
    return logs[-1] if logs else None


def resolve_current_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
    """Okuyucular icin: guncel oturum dosyasi; yoksa eski ``bot.log``
    yedegi (varsa); ikisi de yoksa None. Hicbir zaman hata firlatmaz."""
    current = current_session_log_path(log_dir)
    if current is not None:
        return current
    try:
        legacy = Path(log_dir) / _LEGACY_LOG_NAME
        if legacy.is_file():
            return legacy
    except Exception:
        pass
    return None


def prune_old_sessions(
    log_dir: Union[str, Path], keep: int = SESSION_LOG_KEEP
) -> List[Path]:
    """En yeni ``keep`` oturum dosyasini tut, eskileri sil.

    Yalnizca ``session_YYYY-MM-DD_HHMMSS.log`` bicimindeki dosyalar
    silinir; ``bot.log`` / ``gatherion.log`` ve diger dosyalara
    dokunulmaz. Silinenlerin listesini dondurur; hata firlatmaz
    (silinemeyen dosya atlanir).
    """
    logs = list_session_logs(log_dir)
    keep = max(0, int(keep))
    doomed = logs[: max(0, len(logs) - keep)] if len(logs) > keep else []
    deleted: List[Path] = []
    for path in doomed:
        try:
            os.remove(path)
            deleted.append(path)
        except Exception:
            continue
    return deleted


def default_log_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """main.py'deki ``_resolve_log_dir`` ile ayni cozum (F-1254):

    GATHERION_HOME (varsa) -> ``<home>/logs``; Windows'ta ``F:\\a2``
    varsa ``F:\\a2\\logs``, ``F:\\`` varsa ``F:\\Gatherion\\logs``;
    yoksa ``<repo>/logs``. Okuyucularin (panel, tanilama) main.py'ye
    baglanmadan log dizinini bulmasi icin.
    """
    env_home = os.environ.get("GATHERION_HOME", "").strip()
    if env_home and os.path.isdir(env_home):
        return Path(env_home) / "logs"
    if os.name == "nt":
        if os.path.isdir(r"F:\a2"):
            return Path(r"F:\a2\logs")
        if os.path.isdir(r"F:\\"):
            return Path(r"F:\Gatherion\logs")
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    return root / "logs"
