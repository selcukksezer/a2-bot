# -*- coding: utf-8 -*-
"""Offline checks for COM HID smoke helper (no hardware)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class HidComSmokeOfflineTests(unittest.TestCase):
    def test_module_imports(self) -> None:
        path = ROOT / "tools" / "hid_com_smoke.py"
        spec = importlib.util.spec_from_file_location("hid_com_smoke", path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.main))

    def test_firmware_paths_exist(self) -> None:
        pico = ROOT / "vendor" / "usb_hid" / "pico" / "GatherionHidKeyboard"
        leo = ROOT / "vendor" / "usb_hid" / "leonardo" / "GatherionHidKeyboard"
        self.assertTrue((pico / "GatherionHidKeyboard.ino").is_file())
        self.assertTrue((pico / "GatherionHidKeys.h").is_file())
        self.assertTrue((leo / "GatherionHidKeyboard.ino").is_file())
        self.assertTrue((leo / "GatherionHidKeys.h").is_file())
        doc = ROOT / "docs" / "USB_HID_KEYBOARD.md"
        self.assertTrue(doc.is_file())
        body = doc.read_text(encoding="utf-8")
        self.assertIn("GATHERION_HID_COM", body)
        self.assertIn("KEYDOWN", body)
        self.assertTrue((ROOT / "docs" / "PICO_OPERATOR_TR.md").is_file())
        smoke = (ROOT / "tools" / "hid_com_smoke.py").read_text(encoding="utf-8")
        self.assertIn("--wait-plug", smoke)
        self.assertIn("--write-settings", smoke)
        self.assertTrue((ROOT / "tools" / "set_hid_com.py").is_file())
        # Slice 4 Pico gate — offline path presence (docs + flash helpers + firmware)
        self.assertTrue((ROOT / "tools" / "flash_pico.bat").is_file())
        self.assertTrue((ROOT / "tools" / "flash_pico.ps1").is_file())
        self.assertTrue((ROOT / "vendor" / "usb_hid" / "README.md").is_file())
        self.assertTrue(
            (ROOT / "vendor" / "usb_hid" / "pico" / "GatherionHidKeyboard").is_dir()
        )

    def test_protocol_strings_match_adapter(self) -> None:
        from core.hardware_input_port import _LineHidMixin

        class Capture(_LineHidMixin):
            def __init__(self) -> None:
                self.lines: list[str] = []

            def _send(self, payload: str) -> bool:
                self.lines.append(payload)
                return True

        c = Capture()
        c.key_down("tab")
        c.key_up("1")
        c.send_key("2", 0.08, 0.08)
        self.assertEqual(c.lines[0], "KEYDOWN tab")
        self.assertEqual(c.lines[1], "KEYUP 1")
        self.assertEqual(c.lines[2], "KEY 2 80")


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
