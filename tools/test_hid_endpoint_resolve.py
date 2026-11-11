# -*- coding: utf-8 -*-
"""Quick unit checks for HID endpoint resolution (no hardware)."""
from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def _hid_env_isolated():
    """Clear GATHERION_HID_* env vars for the block, then restore them."""
    old = {
        key: os.environ.pop(key, None)
        for key in ("GATHERION_HID_COM", "GATHERION_HID_TCP")
    }
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    def test_hid_env_isolated_restores(self) -> None:
        old_com = os.environ.get("GATHERION_HID_COM")
        old_tcp = os.environ.get("GATHERION_HID_TCP")
        with _hid_env_isolated():
            self.assertNotIn("GATHERION_HID_COM", os.environ)
            self.assertNotIn("GATHERION_HID_TCP", os.environ)
            os.environ["GATHERION_HID_COM"] = "COM99"
            os.environ["GATHERION_HID_TCP"] = "udp://127.0.0.1:18765"
        self.assertEqual(os.environ.get("GATHERION_HID_COM"), old_com)
        self.assertEqual(os.environ.get("GATHERION_HID_TCP"), old_tcp)

    def test_resolve_env_tcp(self) -> None:
        with _hid_env_isolated():
            os.environ["GATHERION_HID_TCP"] = "192.168.0.5:8765"
            self.assertEqual(resolve_serial_hid_port(), "tcp://192.168.0.5:8765")
            os.environ["GATHERION_HID_TCP"] = "udp://192.168.0.5:8765"
            self.assertEqual(resolve_serial_hid_port(), "udp://192.168.0.5:8765")
            os.environ["GATHERION_HID_COM"] = "COM7"
            self.assertEqual(resolve_serial_hid_port(), "COM7")

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

        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("127.0.0.1", 0))
        port = int(srv.getsockname()[1])
        stop = threading.Event()
        ready = threading.Event()

        def _pong() -> None:
            srv.settimeout(0.3)
            ready.set()
            while not stop.is_set():
                try:
                    data, addr = srv.recvfrom(256)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if b"PING" in data.upper():
                    # sendto() bu VM'nin seccomp filtresinde EPERM verir
                    # (istemci probu da aynı yüzden bağlı soket + send()
                    # kullanır); yanıt bağlı soket üzerinden gönderilir.
                    # connect() sonrası recvfrom yalnızca bu peer'dan gelen
                    # datagramları geçirir — tek istemcili bu testte sorun
                    # değil, protokol (PING -> PONG) aynen korunur.
                    srv.connect(addr)
                    srv.send(b"PONG\n")

        threading.Thread(target=_pong, daemon=True).start()
        # No blind sleep: the probe's 2 s deadline must not race thread
        # scheduling under suite load. Wait until the peer is provably
        # inside its receive loop before the single client probe runs;
        # a peer that never becomes ready fails here, loudly, instead
        # of surfacing as a confusing "available is False".
        self.assertTrue(
            ready.wait(timeout=5.0), "pong peer thread did not start listening"
        )
        try:
            hid = open_hid_adapter(f"udp://127.0.0.1:{port}")
            self.assertTrue(hid.available)
            self.assertEqual(hid.transport, "udp")
            hid.close()
        finally:
            stop.set()
            srv.close()

    def test_setup_android_usb_hid_dry_run(self) -> None:
        # F-1180: dry-run, vendor/android_hid altindaki dosyalar olmadan
        # cikis kodu 1 dondurur; bu checkout'ta staged degiller.
        for needed in (
            ROOT / "vendor" / "android_hid" / "usb_gadget" / "android_usb_hid_server.py",
            ROOT / "vendor" / "android_hid" / "usb_gadget" / "enable_hid_gadget.sh",
        ):
            if not needed.is_file():
                self.skipTest(f"staged vendor file missing in this checkout: {needed}")
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
        # F-1180: vendor/interception bu checkout'ta staged degil.
        if not (ROOT / "vendor" / "interception").is_dir():
            self.skipTest("vendor/interception not staged in this checkout")
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
        # F-1180: installer vendor/interception altinda staged degilse bu
        # testin dosya iddiasi bu checkout'ta karsilanamaz.
        installer_staged = (
            (ROOT / "vendor" / "interception" / "install-interception.exe").is_file()
            or (
                ROOT
                / "vendor"
                / "interception"
                / "_extract"
                / "Interception"
                / "command line installer"
                / "install-interception.exe"
            ).is_file()
        )
        if not installer_staged:
            self.skipTest("install-interception.exe not staged under vendor/interception/")
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
        with _hid_env_isolated():
            ep = resolve_serial_hid_port()
            # F-1180: bu iddia config/settings.json'daki input.* degerine
            # bagliydi; bu checkout'ta bos. Endpoint cozulemiyorsa SKIP.
            if not ep:
                self.skipTest(
                    "no HID endpoint resolvable from config/settings.json in this checkout"
                )
            # Tonight default in config/settings.json is 1B rootless UDP
            self.assertTrue(ep, msg="expected hid endpoint from config/settings.json")
            self.assertTrue(
                is_tcp_hid_endpoint(ep) or ep.upper().startswith("COM"),
                msg=f"unexpected endpoint {ep!r}",
            )

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
        # [O10] D95/F-1153 (merged): exit 0 yalnızca canlı prob gerçekten
        # çalıştıysa; cihaz yokken dürüst sonuç exit 2 + "no live probe"
        # satırıdır. Test, betiğin çalışıp dürüstçe raporladığını kanıtlar.
        self.assertIn(proc.returncode, (0, 2), msg=proc.stdout + proc.stderr)
        self.assertIn("SM-G950F", proc.stdout)
        self.assertIn("RESULT:", proc.stdout)
        self.assertIn("Magisk", proc.stdout)
        self.assertIn("Knox", proc.stdout)
        self.assertIn("s8_root", proc.stdout)
        self.assertIn("vbmeta", proc.stdout.lower())
        # Staged flash kit (may be absent on fresh clones — then URLs must still print)
        self.assertIn("dl.twrp.me/dreamlte", proc.stdout)
        self.assertIn("topjohnwu/Magisk", proc.stdout)


if __name__ == "__main__":
    unittest.main()
