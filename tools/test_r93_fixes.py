# -*- coding: utf-8 -*-
"""Regression tests for R93 fixes (S-4703, S-4901, S-4902, S-4905).

No hardware, no Windows-only deps: handle_line is exercised with a fake
injector, _resolve_port with monkeypatched port helpers, and
write_settings_serial_port against temp files.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.hid_com_util import write_settings_serial_port  # noqa: E402
from tools.rootless_hid_bridge import LocalKeyInjector, handle_line  # noqa: E402
import tools.hid_com_smoke as hid_com_smoke  # noqa: E402


class _FakeInjector:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def key_down(self, name: str) -> bool:
        return True

    def key_up(self, name: str) -> bool:
        return True

    def key_pulse(self, name: str, hold_ms: int = 80) -> bool:
        return True


class RootlessSoftTests(unittest.TestCase):
    def test_default_allow_soft_is_false(self) -> None:
        # S-4703: soft SendInput must be opt-in, never the default.
        sig = inspect.signature(LocalKeyInjector.__init__)
        self.assertIs(sig.parameters["allow_soft"].default, False)

    def test_soft_success_is_reported_as_err(self) -> None:
        # S-4703: a soft "success" must reach the bot as an ERR reply,
        # otherwise the bot believes a key AION 2 ignored actually landed.
        inj = _FakeInjector("soft")
        for line in ("KEYDOWN tab", "KEYUP tab", "KEY 1 80"):
            reply = handle_line(inj, line)
            self.assertIsNotNone(reply, line)
            assert reply is not None
            self.assertTrue(reply.startswith("ERR"), reply)
            self.assertIn("soft", reply.lower())
            self.assertIn("Interception", reply)

    def test_interception_success_returns_none(self) -> None:
        inj = _FakeInjector("interception")
        self.assertIsNone(handle_line(inj, "KEYDOWN tab"))
        self.assertIsNone(handle_line(inj, "KEYUP tab"))
        self.assertIsNone(handle_line(inj, "KEY 1 80"))
        self.assertEqual(handle_line(inj, "PING"), "PONG")

    def test_interception_failure_still_err(self) -> None:
        class _Fail(_FakeInjector):
            def key_down(self, name: str) -> bool:
                return False

        reply = handle_line(_Fail("interception"), "KEYDOWN tab")
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertTrue(reply.startswith("ERR KEYDOWN"))


class WaitPlugAmbiguousTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(port="", wait_plug=True, auto=False, timeout=90.0)

    def test_ambiguous_fresh_ports_rejected(self) -> None:
        # S-4901: pick_best_port returning '' must NOT fall back to fresh[0].
        with (
            patch.object(hid_com_smoke, "port_devices", return_value=set()),
            patch.object(
                hid_com_smoke, "wait_for_new_com", return_value=["COM7", "COM8"]
            ),
            patch.object(hid_com_smoke, "pick_best_port", return_value=""),
            patch.object(hid_com_smoke, "hid_candidates", return_value=[]),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("GATHERION_HID_COM", None)
            self.assertEqual(hid_com_smoke._resolve_port(self._args()), "")

    def test_unambiguous_fresh_port_accepted(self) -> None:
        with (
            patch.object(hid_com_smoke, "port_devices", return_value=set()),
            patch.object(hid_com_smoke, "wait_for_new_com", return_value=["COM7"]),
            patch.object(hid_com_smoke, "pick_best_port", return_value="COM7"),
        ):
            self.assertEqual(hid_com_smoke._resolve_port(self._args()), "COM7")


class WriteSettingsAtomicTests(unittest.TestCase):
    def test_corrupt_settings_backed_up_not_crashed(self) -> None:
        # S-4902: bare json.loads used to raise on corrupt settings.json.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text("{not valid json", encoding="utf-8")
            out = write_settings_serial_port("COM5", settings_path=path)
            self.assertEqual(out, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["input"]["serial_port"], "COM5")
            backups = list(Path(td).glob("settings.json.corrupt*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "{not valid json")
            # No temp file left behind by the atomic write.
            self.assertFalse((Path(td) / "settings.json.tmp").exists())

    def test_roundtrip_preserves_other_keys(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text(
                '{"input": {"hid_tcp": "127.0.0.1:1"}, "x": 1}\n', encoding="utf-8"
            )
            write_settings_serial_port("COM12", settings_path=path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["input"]["serial_port"], "COM12")
            self.assertEqual(data["input"]["hid_tcp"], "127.0.0.1:1")
            self.assertEqual(data["x"], 1)
            self.assertFalse((Path(td) / "settings.json.tmp").exists())


class RelabelWeakClearTests(unittest.TestCase):
    def test_clear_failure_skips_frame(self) -> None:
        # S-4905: source-level regression guard — the stale-txt clear must
        # skip the frame (continue) on failure instead of silently passing
        # and letting commit_split_labels resurrect the stale txt.
        body = (ROOT / "tools" / "relabel_weak_only.py").read_text(encoding="utf-8")
        self.assertNotIn("except Exception:\n                pass", body)
        self.assertIn("could not clear stale label", body)
        self.assertIn("skipped += 1\n                continue", body)


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
