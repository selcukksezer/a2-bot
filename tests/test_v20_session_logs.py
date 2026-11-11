# -*- coding: utf-8 -*-
"""v20/session-logs testleri.

Oturum dosyasi adlandirma (zaman `now` parametresiyle mock'lanir),
guncel oturum cozumu ve 20 oturumda budama.

Testler kosulmazken bile okunabilir olmali: core/session_log yalnizca
stdlib kullanir (Qt yok). Gecici dosyalar ~/workspace altinda tutulur,
/tmp asla kullanilmaz (VM'de /tmp 512M tmpfs).
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.session_log import (
    SESSION_LOG_KEEP,
    current_session_log_path,
    default_log_dir,
    list_session_logs,
    prune_old_sessions,
    resolve_current_log_path,
    resolve_session_log_path,
    session_log_filename,
)

WORK = Path("/home/hatch/workspace/dev-test-v20-session-logs-tmp")


@pytest.fixture()
def workdir():
    WORK.mkdir(parents=True, exist_ok=True)
    yield WORK
    for child in WORK.iterdir():
        child.unlink()
    WORK.rmdir()


def _mk_session(workdir: Path, day: int, hour: int) -> Path:
    name = f"session_2026-11-{day:02d}_{hour:02d}0000.log"
    path = workdir / name
    path.write_text("log satiri\n", encoding="utf-8")
    return path


def test_session_filename_format():
    assert session_log_filename(datetime(2026, 11, 11, 17, 5, 9)) == (
        "session_2026-11-11_170509.log"
    )
    assert session_log_filename(datetime(2026, 1, 2, 3, 4, 5)) == (
        "session_2026-01-02_030405.log"
    )
    # Varsayilan (o anki saat) ayni bicime uyar.
    assert re.match(r"^session_\d{4}-\d{2}-\d{2}_\d{6}\.log$", session_log_filename())
    # Farkli baslatma anlari farkli dosya adlari verir (zaman mock'lu).
    assert session_log_filename(
        datetime(2026, 11, 11, 17, 5, 9)
    ) != session_log_filename(datetime(2026, 11, 11, 17, 6, 10))


def test_resolve_session_log_path_under_log_dir(workdir):
    path = resolve_session_log_path(workdir, datetime(2026, 11, 11, 17, 5, 9))
    assert path == workdir / "session_2026-11-11_170509.log"


def test_current_session_returns_newest(workdir):
    assert current_session_log_path(workdir) is None
    first = _mk_session(workdir, 10, 9)
    second = _mk_session(workdir, 11, 17)
    # Oturum disi dosyalar guncel oturum secimini etkilemez.
    (workdir / "bot.log").write_text("eski tek dosya\n", encoding="utf-8")
    (workdir / "gatherion.log").write_text("stdio\n", encoding="utf-8")
    (workdir / "notes.txt").write_text("x\n", encoding="utf-8")
    assert current_session_log_path(workdir) == second
    assert list_session_logs(workdir) == [first, second]


def test_resolve_current_prefers_session_then_legacy(workdir):
    assert resolve_current_log_path(workdir) is None
    legacy = workdir / "bot.log"
    legacy.write_text("eski\n", encoding="utf-8")
    assert resolve_current_log_path(workdir) == legacy
    session = _mk_session(workdir, 11, 17)
    assert resolve_current_log_path(workdir) == session


def test_prune_keeps_newest_20(workdir):
    made = [_mk_session(workdir, 1 + i // 24, i % 24) for i in range(25)]
    # Oturum disi dosyalar budanmaz.
    legacy = workdir / "bot.log"
    legacy.write_text("eski\n", encoding="utf-8")
    stdio = workdir / "gatherion.log"
    stdio.write_text("stdio\n", encoding="utf-8")

    deleted = prune_old_sessions(workdir)
    assert len(deleted) == 5
    assert sorted(deleted) == sorted(made[:5])
    assert sorted(list_session_logs(workdir)) == sorted(made[5:])
    assert len(list_session_logs(workdir)) == SESSION_LOG_KEEP
    assert legacy.is_file() and stdio.is_file()

    # Ikinci budama bosa doner.
    assert prune_old_sessions(workdir) == []
    assert len(list_session_logs(workdir)) == SESSION_LOG_KEEP


def test_prune_empty_dir_is_noop(workdir):
    assert prune_old_sessions(workdir) == []
    assert list_session_logs(workdir) == []


def test_default_log_dir_env_and_repo(workdir, monkeypatch):
    monkeypatch.setenv("GATHERION_HOME", os.fspath(workdir))
    assert default_log_dir() == workdir / "logs"
    monkeypatch.delenv("GATHERION_HOME")
    # Linux VM'de F: yok; repo kokune duser.
    assert default_log_dir() == _ROOT / "logs"
