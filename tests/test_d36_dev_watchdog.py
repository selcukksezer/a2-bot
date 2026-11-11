# -*- coding: utf-8 -*-
"""D36 (watchdog) dev-improvement tests.

Covers the three D36 improvements in core/watchdog.py:
  A. probe results cached for probe_cache_ttl_sec (display probe +
     memory stats) + same-type display alerts suppressed 60 s;
  B. _read_last_log_lines reads only the tail of bot.log;
  C. incident snapshots run outside BotWatchdog._lock, so a slow
     IncidentRecorder can no longer stall record_heartbeat().

Runs on plain python3 (no PyQt6 / mss needed: probes are monkeypatched
where the outcome would depend on the machine). Temporary files live
under ~/workspace, never /tmp (the VM's /tmp tmpfs is full).
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import core.watchdog as wdmod
from core.watchdog import BotWatchdog, SupportDiagnostics

WORK = Path("/home/hatch/workspace/dev-test-D36-tmp")


@pytest.fixture()
def workdir():
    WORK.mkdir(parents=True, exist_ok=True)
    yield WORK
    for child in WORK.iterdir():
        child.unlink()
    WORK.rmdir()


def _counting_wrappers(monkeypatch):
    """Count OS-probe invocations; return the counts dict."""
    counts = {"resolution": 0, "capture": 0, "memory": 0}
    orig_res = wdmod._query_system_resolution
    orig_mem = wdmod._query_memory_stats
    orig_cap = BotWatchdog._verify_screen_capture_health

    def counting_res():
        counts["resolution"] += 1
        return orig_res()

    def counting_mem():
        counts["memory"] += 1
        return orig_mem()

    def counting_cap(self):
        counts["capture"] += 1
        return orig_cap(self)

    monkeypatch.setattr(wdmod, "_query_system_resolution", counting_res)
    monkeypatch.setattr(wdmod, "_query_memory_stats", counting_mem)
    monkeypatch.setattr(
        BotWatchdog, "_verify_screen_capture_health", counting_cap
    )
    return counts


def test_probe_results_cached_within_ttl(monkeypatch):
    counts = _counting_wrappers(monkeypatch)
    wd = BotWatchdog(stall_timeout_sec=60.0, check_interval_sec=1.0)
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    for _ in range(20):
        wd.get_status()
        wd.check_health()
    # ctor: 1 resolution query. First poll: 1 resolution + 1 capture +
    # 1 memory query. The other 39 polls answer from the cache.
    assert counts == {"resolution": 2, "capture": 1, "memory": 1}, counts


def test_probe_reprobes_after_ttl(monkeypatch):
    counts = _counting_wrappers(monkeypatch)
    wd = BotWatchdog(
        stall_timeout_sec=60.0, check_interval_sec=1.0, probe_cache_ttl_sec=0.05
    )
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    wd.get_status()
    time.sleep(0.07)
    wd.get_status()
    assert counts["resolution"] == 3, counts  # ctor + 2 probes
    assert counts["capture"] == 2, counts
    assert counts["memory"] == 2, counts  # heartbeat + 2 polls, 2nd past TTL


def test_ttl_zero_disables_cache(monkeypatch):
    counts = _counting_wrappers(monkeypatch)
    wd = BotWatchdog(
        stall_timeout_sec=60.0, check_interval_sec=1.0, probe_cache_ttl_sec=0.0
    )
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    for _ in range(3):
        wd.get_status()
    assert counts["capture"] == 3, counts
    assert counts["resolution"] == 4, counts  # ctor + 3 probes


def test_simulate_and_reset_invalidate_cache(monkeypatch):
    # Deterministic probes: real resolution + healthy capture, so the
    # first verdict is True and only a fresh probe can turn it False.
    counts = {"resolution": 0, "capture": 0}

    def counting_res():
        counts["resolution"] += 1
        return (1920, 1080)

    def counting_cap(self):
        counts["capture"] += 1
        return True

    monkeypatch.setattr(wdmod, "_query_system_resolution", counting_res)
    monkeypatch.setattr(
        BotWatchdog, "_verify_screen_capture_health", counting_cap
    )
    wd = BotWatchdog(
        stall_timeout_sec=60.0, check_interval_sec=1.0, probe_cache_ttl_sec=60.0
    )
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    assert wd.get_status()["display_valid"] is True
    assert counts == {"resolution": 2, "capture": 1}, counts

    wd.simulate_display_anomaly(resolution=(1024, 768), minimized=False)
    # Fresh probe (not the cached True): resolution differs -> False,
    # and the RESOLUTION_CHANGED notification proves the probe ran.
    # (A simulated resolution needs no OS query, so resolution stays 2.)
    assert wd.is_display_valid() is False
    assert counts == {"resolution": 2, "capture": 1}, counts
    assert any(
        n["event_type"] == "RESOLUTION_CHANGED" for n in wd._notifications
    )

    wd.reset_display_baseline()
    assert wd.get_status()["display_valid"] is True
    # reset itself queries the resolution once, then the poll probes again.
    assert counts == {"resolution": 4, "capture": 2}, counts


def test_same_display_alert_not_spammed(monkeypatch):
    monkeypatch.setattr(
        wdmod, "_query_system_resolution", lambda: (1920, 1080)
    )
    monkeypatch.setattr(
        BotWatchdog, "_verify_screen_capture_health", lambda self: False
    )
    alerts = []
    wd = BotWatchdog(
        stall_timeout_sec=60.0,
        check_interval_sec=1.0,
        probe_cache_ttl_sec=0.05,
        target_window_titles=[],
        display_alert_callback=lambda ev, info: alerts.append(ev),
    )
    for _ in range(3):
        assert wd.is_display_valid() is False
        time.sleep(0.07)
    failed = [
        n for n in wd._notifications if n["event_type"] == "CAPTURE_FAILED"
    ]
    assert len(failed) == 1, f"expected 1 CAPTURE_FAILED, got {len(failed)}"
    assert alerts == ["CAPTURE_FAILED"], alerts


def test_different_display_alert_type_still_fires(monkeypatch):
    monkeypatch.setattr(
        wdmod, "_query_system_resolution", lambda: (1920, 1080)
    )
    capture_ok = {"value": True}

    def fake_capture(self):
        return capture_ok["value"]

    monkeypatch.setattr(BotWatchdog, "_verify_screen_capture_health", fake_capture)
    wd = BotWatchdog(
        stall_timeout_sec=60.0,
        check_interval_sec=1.0,
        probe_cache_ttl_sec=0.05,
        target_window_titles=[],
    )
    wd.simulate_display_anomaly(minimized=True)
    assert wd.is_display_valid() is False
    wd.reset_display_baseline()
    capture_ok["value"] = False
    assert wd.is_display_valid() is False
    types = [n["event_type"] for n in wd._notifications]
    assert "WINDOW_MINIMIZED" in types, types
    assert "CAPTURE_FAILED" in types, types


def _reference_tail(path, max_lines):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = [line.rstrip() for line in raw[-max_lines:]]
    return tail if tail else ["(Log file is empty)"]


def test_log_tail_matches_full_read(workdir):
    cases = {
        "forty.txt": [f"line {i}" for i in range(40)],
        "five.txt": [f"line {i}" for i in range(5)],
        "empty.txt": [],
        "no_trailing_newline.txt": None,  # written without trailing \n
        "crlf.txt": None,
        "unicode.txt": [f"Türkçe satır {i} — ğüşöçı" for i in range(35)],
    }
    for name, lines in cases.items():
        path = workdir / name
        if lines is None and "no_trailing" in name:
            path.write_text("\n".join(f"line {i}" for i in range(40)), encoding="utf-8")
        elif lines is None:  # crlf
            path.write_bytes(
                b"\r\n".join(f"line {i}".encode() for i in range(40)) + b"\r\n"
            )
        elif lines:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
        got = SupportDiagnostics._read_last_log_lines(path, workdir, max_lines=30)
        assert got == _reference_tail(path, 30), name

    # max_lines=0 keeps the old semantics (whole file, [-0:] == [0:]).
    path = workdir / "forty.txt"
    assert SupportDiagnostics._read_last_log_lines(
        path, workdir, max_lines=0
    ) == _reference_tail(path, 0)

    # Missing file message unchanged.
    missing = SupportDiagnostics._read_last_log_lines(
        workdir / "nope.log", workdir, max_lines=30
    )
    assert missing == [f"(Log file not found at {workdir / 'nope.log'})"], missing

    # One very long line before the tail: the partial first line is dropped.
    path = workdir / "longline.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("X" * 100_000 + "\n")
        for i in range(35):
            f.write(f"short {i}\n")
    got = SupportDiagnostics._read_last_log_lines(path, workdir, max_lines=30)
    assert got == [f"short {i}" for i in range(5, 35)], got


def test_log_tail_large_file_correct_and_fast(workdir):
    path = workdir / "big.log"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(250_000):
            f.write(
                f"[12:00:00] [INFO] Synthetic log entry #{i:06d} "
                "with padding text to reach ~100 bytes per line.\n"
            )
    assert path.stat().st_size > 20_000_000
    t0 = time.perf_counter()
    got = SupportDiagnostics._read_last_log_lines(path, workdir, max_lines=30)
    dt = time.perf_counter() - t0
    assert len(got) == 30, got
    assert got[-1].endswith("#249999 with padding text to reach ~100 bytes per line.")
    assert got[0].endswith("#249970 with padding text to reach ~100 bytes per line.")
    # Old implementation read the whole file (~0.3-0.6 s warm for 20 MB);
    # the tail read touches only the last 64 KB chunk.
    assert dt < 0.5, f"tail read took {dt:.3f}s"


class _GatedRecorder:
    """record_incident blocks until released; proves it runs off the lock."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def record_incident(self, event_type="manual_snapshot", details=""):
        self.calls.append((event_type, details))
        self.entered.set()
        self.release.wait(timeout=10.0)
        return {"ok": True}


def test_incident_snapshot_does_not_block_heartbeat():
    wd = BotWatchdog(stall_timeout_sec=0.05, check_interval_sec=1.0)
    recorder = _GatedRecorder()
    wd._incident_recorder = recorder
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    time.sleep(0.12)  # heartbeat stale -> check_health sees a stall
    t = threading.Thread(target=wd.check_health, daemon=True)
    t.start()
    try:
        assert recorder.entered.wait(timeout=5.0), "snapshot never started"
        t0 = time.perf_counter()
        wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
        dt = time.perf_counter() - t0
        # Before the fix the heartbeat blocked on _lock for the whole
        # snapshot (~0.5 s in the bench); now it returns immediately.
        assert dt < 0.3, f"heartbeat blocked {dt:.3f}s by incident snapshot"
    finally:
        recorder.release.set()
        t.join(timeout=10.0)
    assert recorder.calls, "stall snapshot was lost"
    assert recorder.calls[0][0] == "stuck", recorder.calls


def test_stall_detection_and_recovery_unchanged():
    recoveries = []
    wd = BotWatchdog(
        stall_timeout_sec=0.1,
        check_interval_sec=0.05,
        recovery_callback=lambda info: recoveries.append(info),
    )
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=11.0)
    time.sleep(0.15)
    health = wd.check_health()
    assert health["stalled"] is True
    assert len(recoveries) == 1
    wd.record_heartbeat(state="GATHERING", mode="route", latency_ms=10.0)
    assert not wd.is_stalled()


def test_vitals_death_snapshot_still_recorded():
    recorder = _GatedRecorder()
    # No gate needed here: release immediately so record_incident returns.
    recorder.release.set()
    worker = SimpleNamespace(hp_ratio=0.0, state="DEAD")
    wd = BotWatchdog(stall_timeout_sec=60.0, check_interval_sec=1.0, worker=worker)
    wd._incident_recorder = recorder
    wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.0)
    wd.check_health()
    assert any(
        n["event_type"] == "WORKER_DEAD" for n in wd._notifications
    ), list(wd._notifications)
    assert recorder.calls, "death snapshot was lost"
    # "Character dead" carries no stall/stuck keyword -> manual_snapshot.
    assert recorder.calls[0][0] == "manual_snapshot", recorder.calls
