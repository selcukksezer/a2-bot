# -*- coding: utf-8 -*-
"""D70 (launch-tools) gelistirme testleri.

Kapsar (hepsi Linux'ta kosulabilir; gercek Win32 yollari windows-only'dir):
- tools/bot_elevated_launch.py: tembel Win32 yukleme (import Linux'ta
  cokmemeli), is_admin/probe_pid graceful, register_task/status_report
  schtasks/powershell yokken FileNotFoundError firtlatmamali, --status
  --json saf JSON basmali.
- tools/steam_unelevated_launch.py: list_processes tasklist hatasinda []
  donmeli (timeout=15), task_exists schtasks yokken False, kayit sonrasi
  schtasks /Query dogrulamasi gorunmeyen gorevi hata saymali.
- tools/steam_shortcut_gatherion.py: yol cozumleme onceligi
  (CLI > env > varsayilan), Steam cfg kesfi, --dry-run hicbir sey
  yazmamali, yazma ozel appname ile VDF uretmeli, yabanci kisayol
  varsa uzerine yazmayi reddetmeli.

NOT (2026-11-08 21:12, Selcuk talimati): kodlama ajanlari pytest
kosturmaz; bu dosya test ajanlari tarafindan kosulmak uzere yazildi.
"""

import json
import subprocess
import sys

import pytest

from tools import bot_elevated_launch as bel
from tools import steam_shortcut_gatherion as ssg
from tools import steam_unelevated_launch as sul

WIN32 = sys.platform == "win32"
NON_WIN = pytest.mark.skipif(WIN32, reason="non-Windows only: Win32 tooling absent")


# ---------------------------------------------------------------- bot_elevated
@NON_WIN
def test_bel_imports_and_win32_loader_reports_unavailable():
    assert bel._load_win32() is False
    assert bel.kernel32 is None
    assert bel.shell32 is None


@NON_WIN
def test_bel_is_admin_false_and_probe_pid_unavailable():
    assert bel.is_admin() is False
    info = bel.probe_pid(1234)
    assert info == {
        "pid": 1234,
        "ok": False,
        "error": "Windows-only token probe unavailable on this platform",
    }


def _raise_oserror(*args, **kwargs):
    raise FileNotFoundError(2, "No such file or directory", args[0][0])


def test_bel_task_exists_false_when_schtasks_missing(monkeypatch):
    monkeypatch.setattr(bel, "_run", _raise_oserror)
    assert bel.task_exists() is False


def test_bel_task_exists_from_returncode(monkeypatch):
    def fake_run(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(bel, "_run", fake_run)
    assert bel.task_exists() is True

    def fake_run_fail(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(bel, "_run", fake_run_fail)
    assert bel.task_exists() is False


def test_bel_list_gatherion_pythonw_empty_when_powershell_missing(monkeypatch):
    monkeypatch.setattr(bel, "_run", _raise_oserror)
    assert bel.list_gatherion_pythonw() == []


@NON_WIN
def test_bel_register_task_graceful_dict_on_non_windows():
    # Once: task_exists() FileNotFoundError firlatip graceful dict yolunu
    # kapatiyordu; artik erisilebilir olmali.
    result = bel.register_task()
    assert result["ok"] is False
    assert result["registered"] is False
    assert result["uac_used"] is False
    assert "Windows" in result["error"]


@NON_WIN
def test_bel_status_report_is_real_dict_on_non_windows(monkeypatch):
    monkeypatch.setattr(bel.subprocess, "check_output", _raise_oserror)
    monkeypatch.setattr(bel, "_run", _raise_oserror)
    report = bel.status_report()
    assert report["task_registered"] is False
    assert report["bot_high_elevated"] is False
    assert report["gatherion_bots"] == []
    assert report["self_admin"] is False


@NON_WIN
def test_bel_status_json_is_pure_json(monkeypatch, capsys):
    monkeypatch.setattr(bel.subprocess, "check_output", _raise_oserror)
    monkeypatch.setattr(bel, "_run", _raise_oserror)
    rc = bel.main(["--status", "--json"])
    assert rc == 2
    out = capsys.readouterr().out
    parsed = json.loads(out)  # saf JSON degilse burada patlar
    assert parsed["bot_high_elevated"] is False
    assert parsed["task_registered"] is False


# -------------------------------------------------------------- steam_unelevated
@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError(2, "No such file", "tasklist"),
        subprocess.TimeoutExpired("tasklist", 15),
        subprocess.CalledProcessError(1, "tasklist"),
    ],
)
def test_sul_list_processes_returns_empty_on_tasklist_failure(monkeypatch, exc):
    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(sul.subprocess, "check_output", boom)
    assert sul.list_processes("steam.exe") == []


def test_sul_list_processes_has_timeout(monkeypatch):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return ""

    monkeypatch.setattr(sul.subprocess, "check_output", spy)
    sul.list_processes("steam.exe")
    assert seen.get("timeout") == 15


def test_sul_task_exists_false_when_schtasks_missing(monkeypatch):
    monkeypatch.setattr(sul, "_run", _raise_oserror)
    assert sul.task_exists() is False


def test_sul_ensure_medium_task_rejects_invisible_task(monkeypatch):
    # PowerShell 0 dondu ama schtasks /Query gorevi goremiyor -> hata.
    def fake_run(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sul, "_run", fake_run)
    monkeypatch.setattr(sul, "task_exists", lambda: False)
    with pytest.raises(RuntimeError, match="not visible via schtasks"):
        sul.ensure_medium_task("C:\\Steam\\steam.exe", "")


def test_sul_ensure_medium_task_ok_when_task_visible(monkeypatch):
    def fake_run(cmd, check=False):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sul, "_run", fake_run)
    monkeypatch.setattr(sul, "task_exists", lambda: True)
    sul.ensure_medium_task("C:\\Steam\\steam.exe", "")  # raise etmemeli


@NON_WIN
def test_sul_status_report_on_non_windows():
    report = sul.status_report()
    assert report["steam_medium_unelevated"] is False
    assert report["processes"] == []


# -------------------------------------------------------------- steam_shortcut
def test_ssg_resolve_launch_paths_precedence(monkeypatch):
    monkeypatch.setenv(ssg.ENV_EXE, "E:\\env\\pythonw.exe")
    monkeypatch.setenv(ssg.ENV_START_DIR, "E:\\env")
    monkeypatch.setenv(ssg.ENV_MAIN_PY, "E:\\env\\main.py")
    monkeypatch.setenv(ssg.ENV_APPNAME, "EnvApp")
    got = ssg.resolve_launch_paths()
    assert str(got["exe"]) == "E:\\env\\pythonw.exe"
    assert got["appname"] == "EnvApp"
    # CLI env'i yener.
    got = ssg.resolve_launch_paths(exe="E:\\cli\\pythonw.exe", appname="CliApp")
    assert str(got["exe"]) == "E:\\cli\\pythonw.exe"
    assert got["appname"] == "CliApp"
    # Ne CLI ne env: hard-coded varsayilan (Selcuk'un makinesi) korunur.
    monkeypatch.delenv(ssg.ENV_EXE)
    monkeypatch.delenv(ssg.ENV_APPNAME)
    got = ssg.resolve_launch_paths()
    assert got["exe"] == ssg.EXE
    assert got["appname"] == ssg.APPNAME


def test_ssg_discover_cfg_explicit_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ssg.ENV_STEAM_CFG, str(tmp_path / "env-cfg"))
    assert ssg.discover_steam_cfg(str(tmp_path / "cli-cfg")) == tmp_path / "cli-cfg"
    assert ssg.discover_steam_cfg() == tmp_path / "env-cfg"


def test_ssg_discover_cfg_scans_steam_roots(tmp_path):
    root = tmp_path / "Steam"
    cfg = root / "userdata" / "123" / "config"
    cfg.mkdir(parents=True)
    (cfg / "shortcuts.vdf").write_bytes(b"x")
    assert ssg.discover_steam_cfg(steam_roots=[root]) == cfg


def test_ssg_shortcut_app_ids_use_given_exe():
    ids_default = ssg.shortcut_app_ids('"X" "M"', "Gatherion")
    ids_custom = ssg.shortcut_app_ids('"Y" "M"', "Gatherion", "Y")
    assert ids_default != ids_custom
    assert all(v & 0x80000000 for v in ids_custom.values())


def test_ssg_dry_run_writes_nothing(tmp_path, capsys):
    cfg = tmp_path / "cfg"
    rc = ssg.main(["--steam-cfg", str(cfg), "--dry-run"])
    assert rc == 0
    assert not (cfg / "shortcuts.vdf").exists()
    assert "dry-run" in capsys.readouterr().out


def test_ssg_write_custom_appname_roundtrip(tmp_path):
    cfg = tmp_path / "cfg"
    rc = ssg.main(["--steam-cfg", str(cfg), "--appname", "GatherionTest"])
    assert rc == 0
    data = (cfg / "shortcuts.vdf").read_bytes()
    assert ssg._existing_app_names(data) == ["GatherionTest"]


def test_ssg_refuses_to_overwrite_foreign_shortcuts(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    foreign = ssg.build_shortcuts_vdf('"E" "M"', '"D"', "BaskaOyun")
    (cfg / "shortcuts.vdf").write_bytes(foreign)
    rc = ssg.main(["--steam-cfg", str(cfg)])
    assert rc == 1
    assert (cfg / "shortcuts.vdf").read_bytes() == foreign
