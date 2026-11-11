# -*- coding: utf-8 -*-
"""R19 duzeltme testleri: core/mobile_tunnel.py.

Kapsar:
- F-0138 (Yuksek): regenerate_topic / regenerate_ntfy_topic olu kod degil;
  zayif konu goc yolu regenerate_topic() uzerinden gecer, sarmalayici
  delege eder ve konu kalici olarak doner.
- S-0504 (Dusuk): find_cloudflared_binary adaylari dogrulanmadan
  dondurmez; dogrulama basarisiz ikili yoksayilir (tools/ ve PATH).
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.mobile_tunnel as mt
from core.mobile_tunnel import FreeCloudTunnelManager, NtfyCloudRelay

STRONG_TOPIC_RE = re.compile(r"^gatherion_tr_[0-9a-f]{32}$")
WEAK_TOPIC = "gatherion_tr_ab12cd34"
SECRET = "a" * 64


def _write_relay_config(path: Path, topic: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"topic": topic, "command_secret": SECRET}),
        encoding="utf-8",
    )


def _read_topic(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8")).get("topic", ""))


# ---------------------------------------------------------------- F-0138


def test_regenerate_topic_rotates_and_persists(tmp_path):
    cfg = tmp_path / "mobile_relay.json"
    _write_relay_config(cfg, "gatherion_tr_" + "b" * 32)
    relay = NtfyCloudRelay(config_path=cfg)
    old = relay.topic
    new = relay.regenerate_topic()
    assert new != old
    assert STRONG_TOPIC_RE.match(new)
    assert relay.topic == new
    assert _read_topic(cfg) == new


def test_weak_topic_auto_rotates_on_load(tmp_path):
    """Zayif (8-hex) konu yuklemede regenerate_topic() ile doner (F-0138)."""
    cfg = tmp_path / "mobile_relay.json"
    _write_relay_config(cfg, WEAK_TOPIC)
    relay = NtfyCloudRelay(config_path=cfg)
    assert relay.topic != WEAK_TOPIC
    assert STRONG_TOPIC_RE.match(relay.topic)
    assert _read_topic(cfg) == relay.topic


def test_manager_regenerate_ntfy_topic_delegates(tmp_path, monkeypatch):
    cfg = tmp_path / "mobile_relay.json"
    _write_relay_config(cfg, "gatherion_tr_" + "c" * 32)
    monkeypatch.setattr(mt, "DEFAULT_RELAY_CONFIG_PATH", cfg)
    mgr = FreeCloudTunnelManager()
    old = mgr.ntfy_relay.topic
    new = mgr.regenerate_ntfy_topic()
    assert new != old
    assert STRONG_TOPIC_RE.match(new)
    assert mgr.ntfy_relay.topic == new
    assert _read_topic(cfg) == new


# ---------------------------------------------------------------- S-0504


def _manager_with_tmp_config(tmp_path, monkeypatch):
    cfg = tmp_path / "mobile_relay.json"
    _write_relay_config(cfg, "gatherion_tr_" + "d" * 32)
    monkeypatch.setattr(mt, "DEFAULT_RELAY_CONFIG_PATH", cfg)
    return FreeCloudTunnelManager()


def test_verify_accepts_real_executable_and_rejects_small(tmp_path, monkeypatch):
    mgr = _manager_with_tmp_config(tmp_path, monkeypatch)
    real_python = Path(sys.executable).resolve()
    assert real_python.stat().st_size > 1_000_000
    assert mgr._verify_cloudflared_binary(real_python) is True
    small = tmp_path / "small.bin"
    small.write_bytes(b"MZ" + b"\x00" * 100)
    assert mgr._verify_cloudflared_binary(small) is False


def test_find_cloudflared_rejects_unverified_tools_binary(tmp_path, monkeypatch):
    """MZ baslikli ama calismayan >1MB dosya tools/'da olsa bile donmez."""
    fake = tmp_path / "cloudflared.exe"
    fake.write_bytes(b"MZ" + b"\x00" * 1_000_001)
    monkeypatch.setattr(mt, "CLOUDFLARED_EXE_PATH", fake)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    mgr = _manager_with_tmp_config(tmp_path, monkeypatch)
    assert mgr.find_cloudflared_binary() is None


def test_find_cloudflared_accepts_verified_path_binary(tmp_path, monkeypatch):
    real_python = Path(sys.executable).resolve()
    assert real_python.stat().st_size > 1_000_000
    monkeypatch.setattr(mt, "CLOUDFLARED_EXE_PATH", tmp_path / "yok.exe")
    monkeypatch.setattr(shutil, "which", lambda *a, **k: str(real_python))
    mgr = _manager_with_tmp_config(tmp_path, monkeypatch)
    assert mgr.find_cloudflared_binary() == str(real_python)


def test_find_cloudflared_rejects_bad_path_binary(tmp_path, monkeypatch):
    bad = tmp_path / "bad-cloudflared"
    bad.write_bytes(b"not-an-executable\n" * 70000)  # >1MB, MZ/ELF degil
    assert bad.stat().st_size > 1_000_000
    monkeypatch.setattr(mt, "CLOUDFLARED_EXE_PATH", tmp_path / "yok.exe")
    monkeypatch.setattr(shutil, "which", lambda *a, **k: str(bad))
    mgr = _manager_with_tmp_config(tmp_path, monkeypatch)
    assert mgr.find_cloudflared_binary() is None
