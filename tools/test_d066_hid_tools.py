# -*- coding: utf-8 -*-
"""D66 tests: HID setup-tool improvements (no hardware, Linux-safe).

Covers:
  1. stored_hid_tcp_endpoint() — single source of truth for the persisted
     endpoint form (was duplicated inline in set_hid_tcp.py and
     setup_android_usb_hid.py).
  2. write_settings_hid_tcp() dedup — setup_android_usb_hid.py now imports
     the one implementation from tools.set_hid_tcp (identity check +
     behaviour: key preservation, corrupt-settings backup).
  3. run_adb() timeout — a hung adb used to hang the tool forever;
     it now fails fast with rc 124.
  4. interception_status.py no longer builds an uninitialized bridge via
     object.__new__; output / exit code unchanged on Linux.
  5. set_hid_tcp.py CLI dry-run + invalid-endpoint rejection.

NOTE: /tmp on this VM is a 512M tmpfs and full — temp dirs are created
under ~/workspace and removed in tearDown.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORKSPACE = Path.home() / "workspace"

from tools.set_hid_tcp import (  # noqa: E402
    main as set_hid_tcp_main,
    stored_hid_tcp_endpoint,
    write_settings_hid_tcp,
)
from tools.setup_android_usb_hid import (  # noqa: E402
    adb_devices,
    run_adb,
)
from tools.setup_android_usb_hid import (  # noqa: E402
    write_settings_hid_tcp as setup_write_settings_hid_tcp,
)
from tools.interception_status import main as interception_status_main  # noqa: E402


def _scratch_dir(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="d066-test-", dir=str(WORKSPACE)))
    test.addCleanup(shutil.rmtree, path, True)
    return path


class StoredEndpointTests(unittest.TestCase):
    def test_bare_host_port_stays_bare(self) -> None:
        self.assertEqual(
            stored_hid_tcp_endpoint("192.168.1.55:8765"), "192.168.1.55:8765"
        )

    def test_tcp_scheme_stripped(self) -> None:
        self.assertEqual(
            stored_hid_tcp_endpoint("tcp://10.0.0.2:9000"), "10.0.0.2:9000"
        )

    def test_udp_scheme_kept(self) -> None:
        self.assertEqual(
            stored_hid_tcp_endpoint("udp://127.0.0.1:18766"),
            "udp://127.0.0.1:18766",
        )

    def test_whitespace_and_case(self) -> None:
        self.assertEqual(
            stored_hid_tcp_endpoint("  TCP://127.0.0.1:18765 "),
            "127.0.0.1:18765",
        )
        self.assertEqual(
            stored_hid_tcp_endpoint("  udp://127.0.0.1:18766 "),
            "udp://127.0.0.1:18766",
        )

    def test_non_network_passthrough(self) -> None:
        # Validation upstream rejects these; the helper itself never crashes.
        self.assertEqual(stored_hid_tcp_endpoint("COM5"), "COM5")
        self.assertEqual(stored_hid_tcp_endpoint(""), "")


class WriterDedupTests(unittest.TestCase):
    def test_single_implementation(self) -> None:
        # setup_android_usb_hid must not carry its own copy anymore.
        self.assertIs(setup_write_settings_hid_tcp, write_settings_hid_tcp)

    def test_write_preserves_other_keys(self) -> None:
        scratch = _scratch_dir(self)
        path = scratch / "settings.json"
        path.write_text(
            json.dumps({"input": {"serial_port": "COM5"}, "other": 1}),
            encoding="utf-8",
        )
        out = setup_write_settings_hid_tcp("192.168.1.55:8765", settings_path=path)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["input"]["hid_tcp"], "192.168.1.55:8765")
        self.assertEqual(data["input"]["serial_port"], "COM5")
        self.assertEqual(data["other"], 1)

    def test_corrupt_settings_backed_up_not_wiped(self) -> None:
        scratch = _scratch_dir(self)
        path = scratch / "settings.json"
        path.write_text("{corrupt json", encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            out = setup_write_settings_hid_tcp("127.0.0.1:18765", settings_path=path)
        backups = list(scratch.glob("settings.json.corrupt*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "{corrupt json")
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["input"]["hid_tcp"], "127.0.0.1:18765")


class RunAdbTimeoutTests(unittest.TestCase):
    def _fake_adb(self, scratch: Path) -> Path:
        script = scratch / "adb"
        script.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "sleepy" ]; then sleep 30; fi\n'
            'echo "List of devices attached"\n'
            'echo "FAKESERIAL\tdevice"\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def test_hung_adb_fails_fast_with_rc_124(self) -> None:
        scratch = _scratch_dir(self)
        adb = self._fake_adb(scratch)
        with mock.patch.dict(os.environ, {"ADB": str(adb)}):
            start = time.monotonic()
            proc = run_adb(["sleepy"], timeout_sec=1)
            elapsed = time.monotonic() - start
        self.assertEqual(proc.returncode, 124)
        self.assertIn("timed out", proc.stderr)
        self.assertLess(elapsed, 5.0, f"hung adb took {elapsed:.1f}s")

    def test_fast_adb_still_works(self) -> None:
        scratch = _scratch_dir(self)
        adb = self._fake_adb(scratch)
        with mock.patch.dict(os.environ, {"ADB": str(adb)}):
            proc = run_adb(["devices"], timeout_sec=10)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(adb_devices(), ["FAKESERIAL"])


class SetHidTcpMainTests(unittest.TestCase):
    def test_dry_run_valid_endpoint(self) -> None:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["set_hid_tcp.py", "--dry-run", "192.168.1.55:8765"]):
            with redirect_stdout(out):
                rc = set_hid_tcp_main()
        self.assertEqual(rc, 0)
        self.assertIn("set GATHERION_HID_TCP=192.168.1.55:8765", out.getvalue())

    def test_dry_run_udp_keeps_scheme(self) -> None:
        out = io.StringIO()
        with mock.patch.object(
            sys, "argv", ["set_hid_tcp.py", "--dry-run", "--rootless"]
        ):
            with redirect_stdout(out):
                rc = set_hid_tcp_main()
        self.assertEqual(rc, 0)
        self.assertIn("set GATHERION_HID_TCP=udp://127.0.0.1:18766", out.getvalue())

    def test_com_port_rejected(self) -> None:
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["set_hid_tcp.py", "COM5"]):
            with redirect_stderr(err):
                rc = set_hid_tcp_main()
        self.assertEqual(rc, 2)
        self.assertIn("invalid HID TCP endpoint", err.getvalue())


class InterceptionStatusTests(unittest.TestCase):
    def test_main_reports_without_driver_on_linux(self) -> None:
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["interception_status.py"]):
            with redirect_stdout(out):
                rc = interception_status_main()
        text = out.getvalue()
        # No driver on this Linux VM: exit 2, operator steps printed.
        self.assertEqual(rc, 2)
        self.assertIn("candidate search paths", text)
        self.assertIn("live.available=False", text)
        self.assertIn("RESULT: DLL may be present but driver NOT installed", text)
        self.assertNotIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
