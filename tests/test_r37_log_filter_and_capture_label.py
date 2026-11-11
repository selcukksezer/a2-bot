# -*- coding: utf-8 -*-
"""R37 regression tests.

F-0627: _LogPage._filter_log_content INFO dalı yalnızca INFO etiketli
satırları geçirir — etiketsiz satırlar ve DEBUG satırları geçemez.
S-0902: _query_capture_metrics fps<=0 / hata dalında ölçülmemiş gecikme
"0.0 ms" diye iddia edilmez; etiket "0 FPS · — ms" olur.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import ui.app_shell as app_shell_mod
from core.system_monitor import _query_capture_metrics


def _filter(level: str, text: str) -> str:
    # _LogPage bir QWidget: __init__ cagrilmadan ornek alinca PyQt
    # hasattr erisiminde RuntimeError atiyor; o yuzden bagli olmayan
    # metot sade bir ad alaniyla (fake self) cagriliyor.
    fake = SimpleNamespace(_level_filter=level)
    return app_shell_mod._LogPage._filter_log_content(fake, text)


SAMPLE = "\n".join([
    "12:00:01 [INFO] Herb gathered",
    "12:00:02 [DEBUG] debug detail",
    "12:00:03 plain line without level",
    "12:00:04 [WARN] something odd",
    "12:00:05 [ERROR] boom",
    "12:00:06 [info] lowercase info",
])


def test_f0627_info_filter_keeps_only_info_lines():
    out = _filter("INFO", SAMPLE)
    lines = out.splitlines()
    assert lines == [
        "12:00:01 [INFO] Herb gathered",
        "12:00:06 [info] lowercase info",
    ]


def test_f0627_warn_filter_unchanged_exclusive():
    out = _filter("WARN", SAMPLE)
    assert out.splitlines() == ["12:00:04 [WARN] something odd"]


def test_f0627_error_filter_unchanged():
    out = _filter("ERROR", SAMPLE)
    assert out.splitlines() == ["12:00:05 [ERROR] boom"]


def test_f0627_all_filter_keeps_everything():
    out = _filter("ALL", SAMPLE)
    assert out.splitlines() == SAMPLE.splitlines()


def test_s0902_zero_fps_label_shows_unmeasured_latency():
    cap = _query_capture_metrics(None)
    assert cap["label_str"] == "0 FPS · — ms"
    assert cap["latency_measured"] is False
    assert cap["latency_ms"] == 0.0  # numeric field unchanged for arithmetic

    mock_zero = MagicMock()
    mock_zero.last_fps = 0.0
    # MagicMock auto-attrs would fake a latency; pin them to None.
    for attr in ("last_latency_ms", "latency_ms", "frame_latency_ms",
                 "last_frame_time_ms", "frame_time_ms"):
        setattr(mock_zero, attr, None)
    cap2 = _query_capture_metrics(mock_zero)
    assert cap2["label_str"] == "0 FPS · — ms"
    assert cap2["latency_measured"] is False


def test_s0902_positive_fps_unmeasured_latency_still_dash():
    mock = MagicMock()
    mock.last_fps = 60.0
    for attr in ("last_latency_ms", "latency_ms", "frame_latency_ms",
                 "last_frame_time_ms", "frame_time_ms"):
        setattr(mock, attr, None)
    cap = _query_capture_metrics(mock)
    assert cap["label_str"] == "60 FPS · — ms"


def test_s0902_positive_fps_measured_latency_shown():
    mock = MagicMock()
    mock.last_fps = 60.0
    mock.last_latency_ms = 12.34
    cap = _query_capture_metrics(mock)
    assert cap["label_str"] == "60 FPS · 12.3 ms"
    assert cap["latency_measured"] is True
