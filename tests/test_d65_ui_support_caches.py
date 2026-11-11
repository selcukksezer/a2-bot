# -*- coding: utf-8 -*-
"""D65: cache behaviour for QR pixmaps, guide frames and the ui probe.

- ``MobileConnectDialog._generate_qr_pixmap`` must not re-encode the same
  (url, size) on every manual refresh / diagnostics pass, and must NOT
  cache the "unavailable" fallback card.
- ``GuideDialog._load_frame_pixmap`` must not re-read the same frame file
  from disk on every step navigation.
- ``ui._probe_ui_package`` must run the real filesystem probe at most
  once per TTL window even though the readiness helpers nest 8 deep.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

import ui as ui_pkg
import ui.guide_dialog as guide_mod
import ui.mobile_connect_dialog as mobile_mod

app = QApplication.instance() or QApplication(sys.argv)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _counting_qr_stub(counter):
    """Stand-in for _generate_qr_pixmap_uncached returning a real pixmap."""
    pix = QPixmap(10, 10)
    assert not pix.isNull()

    def _stub(self, url, size=250):
        counter["n"] += 1
        return pix, True

    return _stub


def test_qr_pixmap_cached_for_same_url_and_size(monkeypatch):
    """Same (url, size) encodes once; new url/size re-encodes."""
    dlg = mobile_mod.MobileConnectDialog.__new__(mobile_mod.MobileConnectDialog)
    counter = {"n": 0}
    monkeypatch.setattr(
        mobile_mod.MobileConnectDialog,
        "_generate_qr_pixmap_uncached",
        _counting_qr_stub(counter),
    )
    url = "http://192.168.1.42:8080"

    pix1 = dlg._generate_qr_pixmap(url, size=252)
    pix2 = dlg._generate_qr_pixmap(url, size=252)
    assert counter["n"] == 1
    assert pix1 is pix2

    dlg._generate_qr_pixmap("http://192.168.1.43:8080", size=252)
    assert counter["n"] == 2
    dlg._generate_qr_pixmap(url, size=200)
    assert counter["n"] == 3


def test_qr_pixmap_fallback_card_is_not_cached(monkeypatch):
    """A failed producer must be retried, not frozen in the cache."""
    dlg = mobile_mod.MobileConnectDialog.__new__(mobile_mod.MobileConnectDialog)
    counter = {"n": 0}
    pix = QPixmap(10, 10)

    def _stub(self, url, size=250):
        counter["n"] += 1
        return pix, False

    monkeypatch.setattr(
        mobile_mod.MobileConnectDialog, "_generate_qr_pixmap_uncached", _stub
    )
    dlg._generate_qr_pixmap("http://192.168.1.42:8080", size=252)
    dlg._generate_qr_pixmap("http://192.168.1.42:8080", size=252)
    assert counter["n"] == 2


def test_qr_pixmap_cache_evicts_oldest(monkeypatch):
    """Cache is bounded: the oldest entry is evicted past the cap."""
    dlg = mobile_mod.MobileConnectDialog.__new__(mobile_mod.MobileConnectDialog)
    counter = {"n": 0}
    monkeypatch.setattr(
        mobile_mod.MobileConnectDialog,
        "_generate_qr_pixmap_uncached",
        _counting_qr_stub(counter),
    )
    cap = mobile_mod.MobileConnectDialog._QR_PIXMAP_CACHE_MAX_ENTRIES
    urls = [f"http://192.168.1.{i}:8080" for i in range(cap + 1)]
    for u in urls:
        dlg._generate_qr_pixmap(u, size=252)
    assert counter["n"] == cap + 1
    dlg._generate_qr_pixmap(urls[0], size=252)
    assert counter["n"] == cap + 2  # oldest entry was evicted, re-encoded
    dlg._generate_qr_pixmap(urls[-1], size=252)
    assert counter["n"] == cap + 2  # newest entry still cached


def test_guide_frame_pixmap_cached_from_disk(monkeypatch):
    """Revisiting a guide step reuses the pixmap; disk is read once."""
    dlg = guide_mod.GuideDialog.__new__(guide_mod.GuideDialog)
    dlg.frames_dir = REPO_ROOT / "video_frames"
    calls = {"n": 0}
    orig_imread = guide_mod.imread_unicode

    def _counting_imread(path, *args, **kwargs):
        calls["n"] += 1
        return orig_imread(path, *args, **kwargs)

    monkeypatch.setattr(guide_mod, "imread_unicode", _counting_imread)
    pix1 = dlg._load_frame_pixmap("frame_03s.jpg", "step title")
    pix2 = dlg._load_frame_pixmap("frame_03s.jpg", "step title")
    assert not pix1.isNull()
    assert calls["n"] == 1
    assert pix1 is pix2
    # A different frame is a separate cache entry.
    pix3 = dlg._load_frame_pixmap("frame_12s.jpg", "other step")
    assert not pix3.isNull()
    assert calls["n"] == 2


def test_guide_frame_missing_file_falls_back_without_caching(monkeypatch):
    """Missing frame draws the fallback banner and never poisons the cache."""
    dlg = guide_mod.GuideDialog.__new__(guide_mod.GuideDialog)
    dlg.frames_dir = REPO_ROOT / "video_frames"
    calls = {"n": 0}
    orig_imread = guide_mod.imread_unicode

    def _counting_imread(path, *args, **kwargs):
        calls["n"] += 1
        return orig_imread(path, *args, **kwargs)

    monkeypatch.setattr(guide_mod, "imread_unicode", _counting_imread)
    pix1 = dlg._load_frame_pixmap("no_such_frame.jpg", "missing")
    pix2 = dlg._load_frame_pixmap("no_such_frame.jpg", "missing")
    assert not pix1.isNull() and not pix2.isNull()
    assert calls["n"] == 0  # missing file never reaches imread
    assert pix1 is not pix2  # fallback banner is redrawn, not cached


def test_ui_probe_memoised_within_ttl(monkeypatch):
    """One top-level certificate call used to probe the disk 8 times."""
    calls = {"n": 0}
    orig_probe = ui_pkg._probe_ui_package_uncached

    def _counting_probe():
        calls["n"] += 1
        return orig_probe()

    monkeypatch.setattr(ui_pkg, "_probe_ui_package_uncached", _counting_probe)
    monkeypatch.setattr(ui_pkg, "_probe_cache", {})
    monkeypatch.setattr(ui_pkg, "_probe_cache_ts", 0.0)

    first = ui_pkg.get_ui_package_compliance_certificate()
    second = ui_pkg.get_ui_package_compliance_certificate()
    assert calls["n"] == 1
    assert first == second

    # Returned snapshots are copies: mutating one must not corrupt the cache.
    probe = ui_pkg._probe_ui_package()
    probe["modules_present"]["menu"] = False
    probe["missing_modules"].append("menu")
    fresh = ui_pkg._probe_ui_package()
    assert fresh["modules_present"]["menu"] is True
    assert "menu" not in fresh["missing_modules"]

    # Forcing TTL expiry re-probes exactly once: stale the cached
    # timestamp instead of zeroing the TTL (with TTL 0 every nested
    # helper inside one certificate call would re-probe).
    import time as _time

    monkeypatch.setattr(ui_pkg, "_probe_cache_ts", _time.monotonic() - 10.0)
    ui_pkg.get_ui_package_compliance_certificate()
    assert calls["n"] == 2
