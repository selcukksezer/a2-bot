# -*- coding: utf-8 -*-
"""D01 (dev) tests — actionbus-incidents slice.

Locks in the D01 improvements; nothing here touches the 225 audit findings.
All fixtures are fakes / temp dirs under ~/workspace (never /tmp).
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.incident_recorder as ir
from core.game_action_bus import GameActionBus
from core.incident_recorder import (
    EVENT_METADATA_MAP,
    IncidentRecorder,
    MobileTokenAuthenticator,
    _to_ascii_tr,
    validate_incident_recorder_contract,
)


def _tmpdir():
    return tempfile.TemporaryDirectory(dir=os.path.expanduser("~/workspace"))


# --------------------------------------------------------------------------
# Fake driver / reader / timing for GameActionBus
# --------------------------------------------------------------------------
class FakeTiming:
    def __init__(self):
        self.calls = []

    def sample_gaussian_jitter(self, base_sec, std_sec, min_sec, max_sec):
        self.calls.append((base_sec, std_sec, min_sec, max_sec))
        return base_sec

    def get_key_release_guard_sec(self):
        return 0.02


class FakeDriver:
    def __init__(self):
        self.ensure_calls = 0
        self._human_timing = FakeTiming()
        self.keyboard_inject_trusted = True  # plain bool attr (not callable)

    def ensure_serial_hid(self, force=False):
        self.ensure_calls += 1
        return True

    def press_key(self, *args, **kwargs):
        return 0.1

    def _hw_key_down(self, key, post=False):
        return True

    def _hw_key_up(self, key, post=False):
        return True

    def _hw_mouse_up(self, btn):
        return True

    def release_modifier_keys(self):
        return None

    def focus_game_window(self):
        return None

    def key_down(self, key):
        return True

    def key_up(self, key):
        return True


class FakeReader:
    def __init__(self, fail=False):
        self.marks = []
        self.fail = fail

    def mark_slot_pressed(self, key):
        if self.fail:
            raise RuntimeError("reader boom")
        self.marks.append(key)

    def slot_center_px(self, frame, key):
        return (100, 100)


# --------------------------------------------------------------------------
# 1-3. Probe-chain memoization
# --------------------------------------------------------------------------
def _reset_probe_cache():
    ir._probe_cache["ts"] = 0.0
    ir._probe_cache["value"] = None


def test_probe_chain_runs_probe_once_per_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_probe():
        calls["n"] += 1
        return True

    monkeypatch.setattr(ir, "_probe_incident_recorder_contract", counting_probe)
    _reset_probe_cache()
    try:
        assert validate_incident_recorder_contract() is True
        assert validate_incident_recorder_contract() is True
        assert validate_incident_recorder_contract() is True
        snap = ir.get_incident_recorder_heartbeat_snapshot()
        assert snap["alive"] is True
        # heartbeat walks digest -> audit -> readiness -> metrics (4 probes)
        # + telemetry -> seal -> certificate -> probe -> summary (5 more)
        # before the cache; now the whole chain costs ONE probe.
        assert calls["n"] == 1
    finally:
        _reset_probe_cache()


def test_probe_cache_expires_after_ttl(monkeypatch):
    calls = {"n": 0}

    def counting_probe():
        calls["n"] += 1
        return True

    monkeypatch.setattr(ir, "_probe_incident_recorder_contract", counting_probe)
    monkeypatch.setattr(ir, "_PROBE_CACHE_TTL_SEC", 0.0)
    _reset_probe_cache()
    try:
        assert validate_incident_recorder_contract() is True
        assert validate_incident_recorder_contract() is True
        assert calls["n"] == 2  # TTL 0 -> every call re-probes
    finally:
        _reset_probe_cache()


def test_validate_real_probe_true():
    # Integration: the real probe (temp dir + token + one cv2 record) still
    # returns True; the cache only memoizes that result for 60 s.
    _reset_probe_cache()
    try:
        assert validate_incident_recorder_contract() is True
    finally:
        _reset_probe_cache()


# --------------------------------------------------------------------------
# 4. Event metadata dedup + copy-on-resolve
# --------------------------------------------------------------------------
def test_event_meta_aliases_and_copy():
    assert EVENT_METADATA_MAP["ölüm"]["code"] == "character_dead"
    assert EVENT_METADATA_MAP["death"]["code"] == "character_dead"
    assert EVENT_METADATA_MAP["takılma"]["code"] == "stuck"
    assert EVENT_METADATA_MAP["uzun savaş >2dk"]["code"] == "prolonged_combat"
    assert EVENT_METADATA_MAP["manual_snapshot"]["severity"] == "info"
    # Aliases share the canonical dict object now.
    assert EVENT_METADATA_MAP["ölüm"] is EVENT_METADATA_MAP["character_dead"]
    # ...but callers get a copy, so a mutation cannot poison the map.
    meta = IncidentRecorder._resolve_event_meta("ölüm")
    meta["code"] = "MUTATED"
    assert IncidentRecorder._resolve_event_meta("ölüm")["code"] == "character_dead"
    assert IncidentRecorder._resolve_event_meta("ÖLÜM")["code"] == "character_dead"
    assert IncidentRecorder._resolve_event_meta("bilinmeyen_olay")["severity"] == "warning"


# --------------------------------------------------------------------------
# 5. Transliteration helper
# --------------------------------------------------------------------------
def test_to_ascii_tr_folds_all_turkish_letters():
    assert _to_ascii_tr("İıŞşĞğÜüÖöÇç") == "IiSsGgUuOoCc"
    assert _to_ascii_tr("KRİTİK: KARAKTER ÖLDÜ") == "KRITIK: KARAKTER OLDU"
    assert _to_ascii_tr("plain ascii 123") == "plain ascii 123"


def test_annotate_snapshot_shape_unchanged():
    with _tmpdir() as td:
        rec = IncidentRecorder(incidents_dir=os.path.join(td, "inc"), cooldown_sec=0.0)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        out = rec._annotate_snapshot(
            frame, EVENT_METADATA_MAP["stuck"],
            "2026-11-08 18:00:00", "Takılma detayı ğüşöçı",
            "Altgard", 3, 0.5,
        )
        assert out.shape == (540, 960, 3)
        assert out.dtype == np.uint8


# --------------------------------------------------------------------------
# 6. Lazy thumbnail base64
# --------------------------------------------------------------------------
def test_lazy_thumbnail_b64():
    with _tmpdir() as td:
        rec = IncidentRecorder(
            incidents_dir=os.path.join(td, "inc"), cooldown_sec=0.0, max_incidents=5
        )
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        returned = rec.record_incident("stuck", frame=frame, force=True)
        assert returned is not None
        # Return value keeps the data-URI contract for callers.
        assert str(returned["thumbnail_b64"]).startswith("data:image/jpeg;base64,")
        # ...but the stored in-memory entry carries NO eager base64 anymore.
        stored = rec._incidents[0]
        assert stored["thumbnail_b64"] == ""
        assert os.path.isfile(stored["file_path"])
        assert os.path.isfile(os.path.join(rec.incidents_dir, stored["thumbnail_file"]))
        # Readers still get the data URI, materialized lazily from the file.
        assert str(rec.get_incidents()[0]["thumbnail_b64"]).startswith(
            "data:image/jpeg;base64,"
        )
        # Manifest on disk never carried base64 (and still does not).
        with open(rec.manifest_path, encoding="utf-8") as f:
            manifest_text = f.read()
        assert "thumbnail_b64" not in manifest_text


# --------------------------------------------------------------------------
# 7. Trusted pass-through: one trust probe per cast / tab
# --------------------------------------------------------------------------
def test_cast_skill_probes_trust_once():
    drv = FakeDriver()
    reader = FakeReader()
    bus = GameActionBus(input_driver=drv, skill_bar_reader=reader)
    res = bus.cast_skill("1")
    assert res["ok"] is True
    assert res["method"] == "key"
    assert res["trusted"] is True
    assert res["burst_duration_sec"] > 0.0
    assert res["gcd_align_sec"] > 0.0
    # Before D01: 2 ensure_serial_hid calls per cast (cast_skill + _key_fallback).
    assert drv.ensure_calls == 1
    assert reader.marks == ["1"]


def test_press_tab_target_probes_trust_once():
    drv = FakeDriver()
    bus = GameActionBus(input_driver=drv)
    res = bus.press_tab_target(soft_aim=False)
    assert res["ok"] is True
    assert res["method"] == "key"
    assert drv.ensure_calls == 1


def test_gather_still_probes_when_no_prior_probe():
    drv = FakeDriver()
    bus = GameActionBus(input_driver=drv)
    res = bus.gather_interact(frame=None, preferred_key="f")
    assert res["ok"] is True
    assert res["method"] == "key_fallback"
    assert drv.ensure_calls == 1  # _key_fallback's own probe (trusted=None)


# --------------------------------------------------------------------------
# 8. mark_slot_pressed dedup (best-effort, failure must not break the cast)
# --------------------------------------------------------------------------
def test_mark_slot_pressed_failure_does_not_break_cast():
    drv = FakeDriver()
    bus = GameActionBus(input_driver=drv, skill_bar_reader=FakeReader(fail=True))
    res = bus.cast_skill("2")
    assert res["ok"] is True
    assert res["method"] == "key"


# --------------------------------------------------------------------------
# 9. Jitter helper
# --------------------------------------------------------------------------
def test_sample_jitter_prefers_timing_sampler():
    timing = FakeTiming()
    out = GameActionBus._sample_jitter(
        timing, base_sec=0.03, std_sec=0.003,
        min_sec=0.024, max_sec=0.04,
        fallback_min_sec=0.027, fallback_max_sec=0.035,
    )
    assert out == 0.03
    assert timing.calls == [(0.03, 0.003, 0.024, 0.04)]


def test_sample_jitter_fallback_stays_in_window():
    for _ in range(200):
        out = GameActionBus._sample_jitter(
            None, base_sec=0.03, std_sec=0.003,
            min_sec=0.024, max_sec=0.04,
            fallback_min_sec=0.027, fallback_max_sec=0.035,
        )
        assert 0.027 <= out <= 0.035


def test_sample_jitter_no_timing_object_attribute():
    class NoTiming:
        pass

    out = GameActionBus._sample_jitter(
        NoTiming(), base_sec=0.01, std_sec=0.001,
        min_sec=0.004, max_sec=0.014,
        fallback_min_sec=0.006, fallback_max_sec=0.011,
    )
    assert 0.006 <= out <= 0.011
