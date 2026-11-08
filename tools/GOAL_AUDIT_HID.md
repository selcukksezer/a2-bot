# Goal audit — Android USB HID / Interception (KEEP OPEN)

**Date:** 2026-11-07 evening refresh · **Goal:** OPEN (do not mark complete)  
**Objective:** Wire old Android as USB HID keyboard (no BT required) into Gatherion bot.

| # | Scope | Status | Evidence |
|---|--------|--------|----------|
| **1A** | Root USB Gadget/HID + Wi‑Fi UDP/TCP bridge | **BLOCKED** | SM-G950F Magisk hit RMM “only official binaries”; Prenormal ~7d from `tools/rmm_wait_start.json` start **2026-11-07**. Magisk tonight **NO**. Staged: `vendor/android_hid/usb_gadget`, `tools/setup_android_usb_hid.py`, `tools/after_magisk_hid.bat`, `docs/ANDROID_USB_NO_BT_TR.md` |
| **1B** | Rootless ADB/soft KEY bridge | **DONE** (PC path) | `--self-smoke` → `1B_SMOKE_OK`. UDP **PONG** gate (dead peer → Interception). `resolve_serial_hid_port()` reads `config/settings.json` without env. Live bridge → bot log `NETWORK_HID_OK … (udp)` + `NetworkHID+Interception`. Helpers: `tools/start_rootless_hid.bat`, `set_hid_tcp.py --rootless`. Phone ADB empty tonight |
| **2** | Soft KVM / network keyboard options | **DONE** (docs) | `docs/ANDROID_USB_NO_BT_TR.md` §KVM: Barrier/Input Leap, Mouse Without Borders, Input Director — soft only; not on Gatherion cascade |
| **3** | Interception kernel driver on bot input path | **DONE** | `python tools/interception_status.py` → `RESULT: INTERCEPTION_OK`; DLL `drivers/interception.dll`; cascade Network/Serial HID → Interception → FakerInput → soft |
| **4** | BT dongle / Arduino-Pico compatible with HID pipeline | **PARTIAL** | Pico/Leonardo: `vendor/usb_hid/`, `docs/USB_HID_KEYBOARD.md`, `GATHERION_HID_COM`. BT card ready: `tools/BT_DONGLE_TOMORROW_TR.md` + `vendor/android_hid/GatherionHidBridge-debug.apk`. Dongle not purchased yet |

## Verification criteria

| Check | Result |
|-------|--------|
| `input_driver` / `hardware_input_port` via phone **or** Interception | **Interception proven**; phone USB HID **pending 1A**; 1B NetworkHid→Interception proven via self-smoke |
| Operator docs in repo | `docs/ANDROID_USB_NO_BT_TR.md`, `docs/ANDROID_HID_KEYBOARD.md`, `docs/USB_HID_KEYBOARD.md`, this file, `tools/BT_DONGLE_TOMORROW_TR.md` |

## Operator tonight (no Magisk)

```bat
python tools\interception_status.py
tools\start_rootless_hid.bat
REM or: python tools\rootless_hid_bridge.py --self-smoke --no-soft
baslat.bat
```

Bot uses Interception even if bridge window closed (UDP no-PONG → cascade). With bridge open: `NETWORK_HID_OK` via `udp://127.0.0.1:18766`.

## After RMM clears (~7d) / Magisk OK

1. Follow `docs/S8_MAGISK_SEQUENCE_SAFE.md` / `docs/S8_RMM_WAIT_CHECKLIST.md` — Download check-only first.  
2. `tools\after_magisk_hid.bat` → `python tools\set_hid_tcp.py --adb-forward` → `GATHERION_HID_TCP=127.0.0.1:18765`.  
3. Prove `/dev/hidg0` + live keys → then 1A can flip toward complete.

## After BT dongle (tomorrow)

`tools/BT_DONGLE_TOMORROW_TR.md` → install APK → pair → `GATHERION_HID_TCP=TELEFON_IP:8765` → `NETWORK_HID_OK`.

## UpdateGoal

**Keep OPEN.** Interception + 1B alone do **not** close the goal. Close only when **1A** also proves rooted `su`, `/dev/hidg0`, and bot live keys via phone USB HID (or accepted BT HID hardware path as substitute per operator).
