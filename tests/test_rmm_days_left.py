# -*- coding: utf-8 -*-
"""Tests for tools/rmm_days_left.py (S-4906: wait_days parsing guard)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rmm_days_left import main


def _write_start(tmp_path, **overrides):
    data = {"start_date": "2026-11-07", "model": "SM-G950F", "wait_days": 7, "note": ""}
    data.update(overrides)
    path = tmp_path / "rmm_wait_start.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_wait_days_ok(tmp_path, capsys):
    path = _write_start(tmp_path, wait_days=7)
    assert main(["--start-file", str(path), "--as-of", "2026-11-10"]) == 0
    assert "wait_days=7" in capsys.readouterr().out


def test_non_numeric_wait_days_no_traceback(tmp_path, capsys):
    # S-4906: wait_days='seven' used to raise an uncaught ValueError traceback.
    path = _write_start(tmp_path, wait_days="seven")
    assert main(["--start-file", str(path), "--as-of", "2026-11-10"]) == 2
    captured = capsys.readouterr()
    assert "FAIL: invalid wait_days 'seven'" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_float_string_wait_days_rejected(tmp_path, capsys):
    path = _write_start(tmp_path, wait_days="7.5")
    assert main(["--start-file", str(path), "--as-of", "2026-11-10"]) == 2
    assert "FAIL: invalid wait_days" in capsys.readouterr().err


def test_zero_or_negative_wait_days_falls_back_to_default(tmp_path, capsys):
    for bad in (0, -3):
        path = _write_start(tmp_path, wait_days=bad)
        assert main(["--start-file", str(path), "--as-of", "2026-11-10"]) == 0
        assert "wait_days=7" in capsys.readouterr().out
