# -*- coding: utf-8 -*-
"""Quick unit checks for HID endpoint resolution (no hardware)."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.hardware_input_port import (
    NetworkHidAdapter,
    is_tcp_hid_endpoint,
    is_udp_hid_endpoint,
    normalize_hid_endpoint,
    open_hid_adapter,
    resolve_serial_hid_port,
)


class HidEndpointTests(unittest.TestCase):
    def test_normalize_tcp(self) -> None:
        self.assertEqual(normalize_hid_endpoint("192.168.1.20:8765"), "tcp://192.168.1.20:8765")
        self.assertEqual(normalize_hid_endpoint("tcp://10.0.0.2:9000"), "tcp://10.0.0.2:9000")
        self.assertEqual(normalize_hid_endpoint("COM5"), "COM5")
        self.assertTrue(is_tcp_hid_endpoint("10.0.0.1:8765"))
        self.assertFalse(is_tcp_hid_endpoint("COM3"))

    def test_normalize_udp(self) -> None:
        self.assertEqual(normalize_hid_endpoint("udp://192.168.1.20:8765"), "udp://192.168.1.20:8765")
        self.assertEqual(normalize_hid_endpoint("udp:10.0.0.2:9000"), "udp://10.0.0.2:9000")
        self.assertTrue(is_udp_hid_endpoint("udp://127.0.0.1:18765"))
        self.assertTrue(is_tcp_hid_endpoint("udp://127.0.0.1:18765"))  # network family
        self.assertFalse(is_udp_hid_endpoint("tcp://127.0.0.1:18765"))

    def test_resolve_env_tcp(self) -> None:
        old_com = os.environ.pop("GATHERION_HID_COM", None)
        old_tcp = os.environ.pop("GATHERION_HID_TCP", None)
        try:
            os.environ["GATHERION_HID_TCP"] = "192.168.0.5:8765"
            self.assertEqual(resolve_serial_hid_port(), "tcp://192.168.0.5:8765")
            os.environ["GATHERION_HID_TCP"] = "udp://192.168.0.5:8765"
            self.assertEqual(resolve_serial_hid_port(), "udp://192.168.0.5:8765")
            os.environ["GATHERION_HID_COM"] = "COM7"
            self.assertEqual(resolve_serial_hid_port(), "COM7")
        finally:
            if old_com is None:
                os.environ.pop("GATHERION_HID_COM", None)
            else:
                os.environ["GATHERION_HID_COM"] = old_com
            if old_tcp is None:
                os.environ.pop("GATHERION_HID_TCP", None)
            else:
                os.environ["GATHERION_HID_TCP"] = old_tcp

    def test_open_factory_tcp_offline(self) -> None:
        hid = open_hid_adapter("127.0.0.1:1")
        self.assertIsInstance(hid, NetworkHidAdapter)
        self.assertEqual(hid.transport, "tcp")
        self.assertFalse(hid.available)

    def test_open_factory_udp_offline(self) -> None:
        hid = open_hid_adapter("udp://127.0.0.1:1")
        self.assertIsInstance(hid, NetworkHidAdapter)
        self.assertEqual(hid.transport, "udp")
        # Dead peer: no PONG → unavailable so Interception cascade is not skipped
        self.assertFalse(hid.available)
        hid.close()
        self.assertFalse(hid.available)

    def test_open_factory_udp_with_pong_peer(self) -> None:
        import socket
        import threading
        import time

        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("127.0.0.1", 0))
        port = int(srv.getsockname()[1])
        stop = threading.Event()

        def _pong() -> None:
            srv.settimeout(0.3)
            while not stop.is_set():
                try:
                    data, addr = srv.recvfrom(256)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if b"PING" in data.upper():
                    srv.sendto(b"PONG\n", addr)

        threading.Thread(target=_pong, daemon=True).start()
        time.sleep(0.05)
        try:
            hid = open_hid_adapter(f"udp://127.0.0.1:{port}")
            self.assertTrue(hid.available)
            self.assertEqual(hid.transport, "udp")
            hid.close()
        finally:
            stop.set()
            srv.close()

    def test_setup_android_usb_hid_dry_run(self) -> None:
        script = ROOT / "tools" / "setup_android_usb_hid.py"
        self.assertTrue(script.is_file())
        proc = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("android_usb_hid_server.py", proc.stdout)
        self.assertIn("ANDROID_USB_NO_BT_TR.md", proc.stdout)

    def test_set_hid_tcp_dry_run(self) -> None:
        script = ROOT / "tools" / "set_hid_tcp.py"
        self.assertTrue(script.is_file())
        proc = subprocess.run(
            [sys.executable, str(script), "udp://10.0.0.9:8765", "--dry-run"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("GATHERION_HID_TCP=udp://10.0.0.9:8765", proc.stdout)

    def test_interception_vendor_path_listed(self) -> None:
        from core.input_driver import InterceptionBridge

        # Avoid __init__ driver probe; only check search paths.
        bridge = object.__new__(InterceptionBridge)
        paths = InterceptionBridge._candidate_dll_paths(bridge)
        joined = os.path.normcase("\n".join(paths))
        self.assertIn(os.path.normcase("vendor" + os.sep + "interception"), joined)

    def test_rootless_bridge_parse_line(self) -> None:
        from tools.rootless_hid_bridge import parse_line

        self.assertEqual(parse_line("PING"), ("PING", ()))
        self.assertEqual(parse_line("KEYDOWN tab"), ("KEYDOWN", ("tab",)))
        self.assertEqual(parse_line("KEYUP 1"), ("KEYUP", ("1",)))
        self.assertEqual(parse_line("KEY 2 120"), ("KEY", ("2", 120)))
        self.assertEqual(parse_line("KEY 3"), ("KEY", ("3", 80)))
        self.assertEqual(parse_line("NOPE"), ("", ()))

    def test_rootless_and_interception_helpers_exist(self) -> None:
        self.assertTrue((ROOT / "tools" / "rootless_hid_bridge.py").is_file())
        self.assertTrue((ROOT / "tools" / "interception_status.py").is_file())
        self.assertTrue((ROOT / "docs" / "ANDROID_USB_NO_BT_TR.md").is_file())
        doc = (ROOT / "docs" / "ANDROID_USB_NO_BT_TR.md").read_text(encoding="utf-8")
        self.assertIn("SM-G950F", doc)
        self.assertIn("rootless_hid_bridge", doc)
        self.assertIn("Barrier", doc)
        self.assertIn("Mouse Without Borders", doc)
        self.assertIn("Input Director", doc)

    def test_interception_status_exit_codes(self) -> None:
        script = ROOT / "tools" / "interception_status.py"
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        # 0 = driver ready, 2 = dll/driver missing — both OK for offline CI
        self.assertIn(proc.returncode, (0, 2), msg=proc.stdout + proc.stderr)
        self.assertIn("Interception status", proc.stdout)
        self.assertIn("vendor", proc.stdout.lower())
        self.assertIn("process elevated", proc.stdout.lower())
        self.assertTrue(
            (ROOT / "vendor" / "interception" / "install-interception.exe").is_file()
            or (
                ROOT
                / "vendor"
                / "interception"
                / "_extract"
                / "Interception"
                / "command line installer"
                / "install-interception.exe"
            ).is_file(),
            msg="install-interception.exe should be staged under vendor/interception/",
        )

    def test_resolve_reads_settings_json_when_no_env(self) -> None:
        old_com = os.environ.pop("GATHERION_HID_COM", None)
        old_tcp = os.environ.pop("GATHERION_HID_TCP", None)
        try:
            ep = resolve_serial_hid_port()
            # Tonight default in config/settings.json is 1B rootless UDP
            self.assertTrue(ep, msg="expected hid endpoint from config/settings.json")
            self.assertTrue(
                is_tcp_hid_endpoint(ep) or ep.upper().startswith("COM"),
                msg=f"unexpected endpoint {ep!r}",
            )
        finally:
            if old_com is None:
                os.environ.pop("GATHERION_HID_COM", None)
            else:
                os.environ["GATHERION_HID_COM"] = old_com
            if old_tcp is None:
                os.environ.pop("GATHERION_HID_TCP", None)
            else:
                os.environ["GATHERION_HID_TCP"] = old_tcp

    def test_set_hid_tcp_rootless_dry_run(self) -> None:
        script = ROOT / "tools" / "set_hid_tcp.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--rootless", "--dry-run"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("GATHERION_HID_TCP=udp://127.0.0.1:18766", proc.stdout)

    def test_s8_root_checklist_runs(self) -> None:
        script = ROOT / "tools" / "s8_root_checklist.py"
        self.assertTrue(script.is_file())
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("SM-G950F", proc.stdout)
        self.assertIn("Magisk", proc.stdout)
        self.assertIn("Knox", proc.stdout)
        self.assertIn("s8_root", proc.stdout)
        self.assertIn("vbmeta", proc.stdout.lower())
        # Staged flash kit (may be absent on fresh clones — then URLs must still print)
        self.assertIn("dl.twrp.me/dreamlte", proc.stdout)
        self.assertIn("topjohnwu/Magisk", proc.stdout)


if __name__ == "__main__":
    unittest.main()
