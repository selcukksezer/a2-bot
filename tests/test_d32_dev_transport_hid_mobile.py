# -*- coding: utf-8 -*-
"""D32 DEV iyilestirme testleri.

Dilim: core/hardware_input_port.py, core/mobile_tunnel.py
(Tasima katmani degistirilmedi; R/T eslemesi ve sert kapiya dokunulmadi.)

Kapsar:
  I1  SoftInputAdapter send_key/mouse_button teklesmesi (davranis ayni)
  I2  settings.json (mtime, boyut) onbellegi + gecersiz kilma
  I3  get_lan_ip 30 sn TTL onbellegi; get_status binary aramasi 1 kez
  I4  _run_tunnel_process ortak govdesi (cloudflare + ssh baslaticilar)
  I5  ntfy yoklama uyarlanabilir geri cekilme (backoff)
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import mobile_tunnel as mt
from core.hardware_input_port import (
    SoftInputAdapter,
    _SETTINGS_JSON_CACHE,
    _load_settings_json_dict,
    resolve_serial_hid_port,
)


def _tmp_relay() -> mt.NtfyCloudRelay:
    tmpdir = tempfile.mkdtemp(dir="/home/hatch/workspace")
    return mt.NtfyCloudRelay(config_path=Path(tmpdir) / "relay.json")


def _tmp_manager() -> mt.FreeCloudTunnelManager:
    tmpdir = tempfile.mkdtemp(dir="/home/hatch/workspace")
    with patch.object(mt, "DEFAULT_RELAY_CONFIG_PATH", Path(tmpdir) / "relay.json"):
        return mt.FreeCloudTunnelManager(local_port=18765)


class _FakeProc:
    """subprocess.Popen benzeri: stdout/stderr StringIO, poll None."""

    def __init__(self, stdout_text: str = "", stderr_text: str = "") -> None:
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO(stderr_text)
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class _FakeResp:
    def __init__(self, body: bytes = b"") -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestSoftInputAdapterDedup(unittest.TestCase):
    def test_send_key_full_kwargs_success(self):
        driver = MagicMock()
        driver.press_key.return_value = 0.1
        adapter = SoftInputAdapter(input_driver=driver)
        self.assertTrue(adapter.send_key("r"))
        driver.press_key.assert_called_once_with(
            "r", min_hold=0.08, max_hold=0.12,
            release_mouse=True, post_to_window=False,
        )

    def test_send_key_legacy_driver_fallback(self):
        driver = MagicMock()

        def _press(key, **kwargs):
            if "release_mouse" in kwargs:
                raise TypeError("legacy driver")
            return None  # legacy: kabul, donus yok

        driver.press_key.side_effect = _press
        adapter = SoftInputAdapter(input_driver=driver)
        self.assertTrue(adapter.send_key("t"))
        self.assertEqual(driver.press_key.call_count, 2)

    def test_send_key_zero_hold_and_false_are_failure(self):
        driver = MagicMock()
        driver.press_key.return_value = 0.0
        self.assertFalse(SoftInputAdapter(input_driver=driver).send_key("r"))
        driver.press_key.return_value = False
        self.assertFalse(SoftInputAdapter(input_driver=driver).send_key("r"))
        self.assertFalse(SoftInputAdapter(input_driver=None).send_key("r"))

    def test_mouse_button_hw_path_down_and_up(self):
        driver = MagicMock()
        driver._hw_mouse_down.return_value = None
        driver._hw_mouse_up.return_value = True
        adapter = SoftInputAdapter(input_driver=driver)
        self.assertTrue(adapter.mouse_button("left", True))
        driver._hw_mouse_down.assert_called_once_with("left")
        self.assertTrue(adapter.mouse_button("right", False))
        driver._hw_mouse_up.assert_called_once_with("right")

    def test_mouse_button_no_hw_path_refuses_without_sendinput_fallback(self):
        """N-3101: baypas kaldırıldı, Interception yoksa girdi başarısız sayılır.

        Sürücüde güvenilir `_hw_mouse_down/up` yoksa adaptör artık False
        döner; silinen `_sendinput_mouse_button` yedeğine düşmez — sahte
        SendInput yolu hiç çağrılmaz."""
        sendinput = MagicMock(return_value=None)
        driver = SimpleNamespace(
            _sendinput_mouse_button=sendinput
        )
        adapter = SoftInputAdapter(input_driver=driver)
        self.assertFalse(adapter.mouse_button("left", True))
        self.assertFalse(adapter.mouse_button("left", False))
        sendinput.assert_not_called()

    def test_mouse_button_no_driver(self):
        self.assertFalse(SoftInputAdapter(input_driver=None).mouse_button("left", True))


class TestSettingsJsonCache(unittest.TestCase):
    def setUp(self):
        _SETTINGS_JSON_CACHE.clear()
        _SETTINGS_JSON_CACHE.update({"key": None, "data": {}})

    def tearDown(self):
        _SETTINGS_JSON_CACHE.clear()
        _SETTINGS_JSON_CACHE.update({"key": None, "data": {}})

    def test_second_call_does_not_re_read(self):
        import pathlib

        reads = []
        orig_read = pathlib.Path.read_text

        def _counting(self, *args, **kwargs):
            if self.name == "settings.json":
                reads.append(1)
            return orig_read(self, *args, **kwargs)

        with patch.object(pathlib.Path, "read_text", _counting):
            first = _load_settings_json_dict()
            second = _load_settings_json_dict()
        self.assertEqual(first, second)
        self.assertTrue(first, "config/settings.json okunmali")
        self.assertEqual(len(reads), 1, "2. cagri disk okumamali")

    def test_mtime_change_re_reads_and_new_content_wins(self):
        import pathlib

        orig_stat = pathlib.Path.stat
        orig_read = pathlib.Path.read_text
        bumped = {"on": False}

        def _stat(self, *args, **kwargs):
            st = orig_stat(self, *args, **kwargs)
            if self.name == "settings.json" and bumped["on"]:
                # is_file() st_mode ister (S_ISREG kontrolu)
                import stat as _statmod

                return SimpleNamespace(
                    st_mtime_ns=st.st_mtime_ns + 1,
                    st_size=st.st_size,
                    st_mode=_statmod.S_IFREG | 0o644,
                )
            return st

        def _read(self, *args, **kwargs):
            if self.name == "settings.json" and bumped["on"]:
                return '{"input": {"serial_port": "COM9"}}'
            return orig_read(self, *args, **kwargs)

        with patch.object(pathlib.Path, "stat", _stat), patch.object(
            pathlib.Path, "read_text", _read
        ), patch.dict(os.environ, {"GATHERION_HID_COM": "", "GATHERION_HID_TCP": ""}):
            before = resolve_serial_hid_port()
            self.assertNotEqual(before, "COM9")
            bumped["on"] = True
            self.assertEqual(resolve_serial_hid_port(), "COM9")


class TestLanIpCacheAndStatus(unittest.TestCase):
    def setUp(self):
        mt._LAN_IP_CACHE.clear()
        mt._LAN_IP_CACHE.update({"ip": None, "at": 0.0})

    def tearDown(self):
        mt._LAN_IP_CACHE.clear()
        mt._LAN_IP_CACHE.update({"ip": None, "at": 0.0})

    def test_get_lan_ip_cached_within_ttl(self):
        calls = []
        with patch.object(
            mt, "_detect_lan_ip", side_effect=lambda: calls.append(1) or "10.1.2.3"
        ):
            self.assertEqual(mt.get_lan_ip(), "10.1.2.3")
            self.assertEqual(mt.get_lan_ip(), "10.1.2.3")
        self.assertEqual(len(calls), 1, "TTL icinde 2. cagri tespit yapmamali")
        mt._LAN_IP_CACHE["at"] = time.monotonic() - 31.0
        with patch.object(
            mt, "_detect_lan_ip", side_effect=lambda: calls.append(1) or "10.1.2.3"
        ):
            self.assertEqual(mt.get_lan_ip(), "10.1.2.3")
        self.assertEqual(len(calls), 2, "TTL dolunca yeniden tespit edilmeli")

    def test_get_status_calls_binary_lookup_once(self):
        mgr = _tmp_manager()
        calls = []
        orig = mt.FreeCloudTunnelManager.find_cloudflared_binary

        def _counting(self):
            calls.append(1)
            return orig(self)

        with patch.object(
            mt.FreeCloudTunnelManager, "find_cloudflared_binary", _counting
        ):
            status = mgr.get_status()
        self.assertEqual(len(calls), 1, "get_status binary aramasini 1 kez yapmali")
        self.assertIs(status["cloudflared_installed"], False)
        self.assertTrue(str(status["cloudflared_path"]).endswith("cloudflared.exe"))


class TestTunnelProcessHelper(unittest.TestCase):
    def test_cloudflare_tunnel_url_found_once(self):
        mgr = _tmp_manager()
        fired = []
        mgr.on_url_ready = lambda url, method: fired.append((url, method))
        proc = _FakeProc(stdout_text="x\nhttps://abc-123.trycloudflare.com ok\n")
        with patch.object(mt.subprocess, "Popen", return_value=proc), patch.object(
            mt.FreeCloudTunnelManager, "find_cloudflared_binary",
            return_value="/fake/cloudflared.exe",
        ):
            url = mgr.start_cloudflare_tunnel(
                local_port=18765, auto_download=False, timeout_sec=5.0
            )
        self.assertEqual(url, "https://abc-123.trycloudflare.com")
        self.assertEqual(mgr.public_url, url)
        self.assertEqual(mgr.active_method, "cloudflare")
        self.assertEqual(fired, [(url, "cloudflare")])
        mgr._stop_tunnel_process()

    def test_ssh_tunnel_url_found(self):
        mgr = _tmp_manager()
        proc = _FakeProc(stderr_text="x\nhttps://demo-1.lhr.life\n")
        with patch.object(mt.subprocess, "Popen", return_value=proc), patch.object(
            mt.shutil, "which", return_value="/usr/bin/ssh"
        ):
            url = mgr.start_ssh_tunnel(
                local_port=18765, provider="localhost.run", timeout_sec=5.0
            )
        self.assertEqual(url, "https://demo-1.lhr.life")
        self.assertEqual(mgr.active_method, "ssh_localhost_run")
        mgr._stop_tunnel_process()

    def test_tunnel_timeout_returns_none(self):
        mgr = _tmp_manager()
        proc = _FakeProc()
        with patch.object(mt.subprocess, "Popen", return_value=proc), patch.object(
            mt.FreeCloudTunnelManager, "find_cloudflared_binary",
            return_value="/fake/cloudflared.exe",
        ):
            url = mgr.start_cloudflare_tunnel(
                local_port=18765, auto_download=False, timeout_sec=0.3
            )
        self.assertIsNone(url)
        self.assertIn("zaman", mgr._last_error)


class TestNtfyPollBackoff(unittest.TestCase):
    def test_backoff_grows_on_errors_and_resets_on_success(self):
        relay = _tmp_relay()
        err = urllib.error.URLError("down")
        calls = {"n": 0}

        def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise err
            return _FakeResp(b"")

        with patch.object(mt.urllib.request, "urlopen", _flaky):
            relay.poll_commands_once()
            self.assertEqual(relay._consecutive_poll_errors, 1)
            self.assertEqual(relay._current_poll_interval_sec, 7.0)
            relay.poll_commands_once()
            self.assertEqual(relay._current_poll_interval_sec, 14.0)
            relay.poll_commands_once()
            self.assertEqual(relay._current_poll_interval_sec, 28.0)
            relay.poll_commands_once()
            self.assertEqual(relay._consecutive_poll_errors, 0)
            self.assertEqual(relay._current_poll_interval_sec, 3.5)

    def test_backoff_caps_at_30s(self):
        relay = _tmp_relay()
        with patch.object(
            mt.urllib.request, "urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            for _ in range(6):
                relay.poll_commands_once()
        self.assertEqual(relay._current_poll_interval_sec, 30.0)
        status = relay.get_status()
        self.assertEqual(status["poll_interval_sec"], 30.0)
        self.assertEqual(status["consecutive_poll_errors"], 6)

    def test_listener_uses_backoff_interval(self):
        relay = _tmp_relay()
        relay.poll_interval_sec = 0.2
        relay._current_poll_interval_sec = 0.2
        with patch.object(
            mt.urllib.request, "urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            relay.start_command_listener()
            try:
                time.sleep(0.6)
                self.assertGreaterEqual(relay._consecutive_poll_errors, 1)
                self.assertGreater(relay._current_poll_interval_sec, 0.2)
            finally:
                relay.stop_command_listener()


if __name__ == "__main__":
    unittest.main()
