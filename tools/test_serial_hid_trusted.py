# -*- coding: utf-8 -*-
"""SerialHID combat trust matches NetworkHID (no soft cascade when line HID live)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.game_action_bus import GameActionBus
from core.input_driver import InputController


class _FakeLineHid:
    transport = "serial"

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.available = True

    def key_down(self, key: str) -> bool:
        self.lines.append(f"KEYDOWN {key}")
        return True

    def key_up(self, key: str) -> bool:
        self.lines.append(f"KEYUP {key}")
        return True


class SerialHidTrustedTests(unittest.TestCase):
    def test_keyboard_inject_trusted_for_serial_transport(self) -> None:
        ctrl = InputController.__new__(InputController)
        ctrl._serial_hid = _FakeLineHid()
        ctrl._interception = None
        ctrl._faker = None
        ctrl._serial_port_cfg = "COM9"
        ctrl._hid_retry_ts = 0.0
        ctrl.ensure_serial_hid = lambda force=False, min_interval_sec=2.0: True  # type: ignore
        self.assertTrue(ctrl.keyboard_inject_trusted)
        self.assertEqual(ctrl.active_backend_name, "SerialHID")

    def test_hw_key_down_serial_skips_soft_when_ok(self) -> None:
        ctrl = InputController.__new__(InputController)
        fake = _FakeLineHid()
        ctrl._serial_hid = fake
        ctrl._interception = None
        ctrl._faker = None
        ctrl._sending_virtual_key = False
        ctrl._last_virtual_key_ts = 0.0
        ctrl.ensure_serial_hid = lambda force=False, min_interval_sec=2.0: True  # type: ignore
        soft = MagicMock(return_value=True)
        ctrl._sendinput_key = soft  # type: ignore
        ctrl._keybd_event_key = soft  # type: ignore
        ok = InputController._hw_key_down(ctrl, "tab", post_to_window=False)
        self.assertTrue(ok)
        self.assertEqual(fake.lines, ["KEYDOWN tab"])
        soft.assert_not_called()

    def test_bus_tab_and_skills_hardware_only_when_serial_trusted(self) -> None:
        held: list[tuple] = []

        class Drv:
            def keyboard_inject_trusted(self):
                return True

            def focus_game_window(self):
                return True

            def release_modifier_keys(self):
                return None

            def prepare_game_keyboard(self):
                return None

            def _hw_key_down(self, key, post_to_window=False):
                held.append(("down", key, post_to_window))
                return True

            def _hw_key_up(self, key, post_to_window=False):
                held.append(("up", key, post_to_window))

            def press_key(self, *a, **k):
                raise AssertionError("soft press_key must not run when SerialHID trusted")

        bus = GameActionBus(input_driver=Drv(), mouse_look_mode=True)
        tab = bus.press_tab_target(soft_aim=False)
        self.assertTrue(tab["ok"])
        self.assertEqual(tab["method"], "key")
        skill = bus.cast_skill("1")
        self.assertTrue(skill["ok"])
        self.assertEqual(skill["method"], "key")
        self.assertTrue(any(h[1] == "tab" and h[2] is False for h in held if h[0] == "down"))
        self.assertTrue(any(h[1] == "1" and h[2] is False for h in held if h[0] == "down"))


class HidComUtilOfflineTests(unittest.TestCase):
    def test_write_settings_roundtrip(self) -> None:
        import tempfile

        from tools.hid_com_util import write_settings_serial_port

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            path.write_text('{"input": {"hid_tcp": "127.0.0.1:1"}, "x": 1}\n', encoding="utf-8")
            write_settings_serial_port("COM12", settings_path=path)
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["input"]["serial_port"], "COM12")
            self.assertEqual(data["input"]["hid_tcp"], "127.0.0.1:1")
            self.assertEqual(data["x"], 1)

    def test_smoke_cli_flags_present(self) -> None:
        body = (ROOT / "tools" / "hid_com_smoke.py").read_text(encoding="utf-8")
        self.assertIn("--wait-plug", body)
        self.assertIn("--write-settings", body)
        self.assertIn("SerialHID", body)
        self.assertTrue((ROOT / "tools" / "flash_pico.bat").is_file())
        self.assertTrue((ROOT / "docs" / "PICO_OPERATOR_TR.md").is_file())


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
