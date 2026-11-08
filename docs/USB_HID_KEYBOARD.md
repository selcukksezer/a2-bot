# USB Pico / Leonardo — Gatherion real keyboard (no Bluetooth)

AION2 drops soft `SendInput` keys. Gatherion already speaks a **real HID keyboard**
line protocol over serial (`SerialHidAdapter`). This is the **primary path when the
PC has no Bluetooth**.

Mouse / Alt+skillbar / soft-aim is **not** a keyboard substitute. Skills need
`KEYDOWN 1` / `KEYDOWN tab` from a physical USB HID device.

```
Gatherion (PC)  --COM serial-->  Pico or Leonardo
                                      |
                                      v
                               USB HID keyboard reports
                                      |
                                      v
                               Windows → AION2  (trusted keys)
```

---

## What to buy (pick one)

| Buy | Why | Approx | Notes |
|-----|-----|--------|-------|
| **Raspberry Pi Pico (RP2040)** — **recommended** | Cheap USB HID + CDC COM, firmware staged here | ~$4–8 | Micro-USB or USB-C clone OK; no BT needed |
| Arduino Leonardo / Pro Micro (ATmega32U4) | Native USB HID, same protocol | ~$8–15 | Also staged; slightly simpler Arduino boards list |
| USB Bluetooth dongle (for PC) | Only if you prefer **phone** as keyboard | dongle + phone | Then use `GatherionHidBridge` + `GATHERION_HID_TCP` — see `docs/ANDROID_HID_KEYBOARD.md` |

**Samsung S8 + no PC Bluetooth:** either **Pico/Leonardo USB** (this doc) **or** a **USB BT dongle** so Android BT-HID can pair. Wi‑Fi alone cannot inject HID.

---

## Flash

**Turkish 3-step card:** `docs/PICO_OPERATOR_TR.md`

### Pico (recommended)

Helper (UF2 / picotool / arduino-cli / BOOTSEL detect — no UAC):

```bat
tools\flash_pico.bat
tools\flash_pico.bat -WaitCom -SetHid
```

Manual:

1. Install [Arduino IDE](https://www.arduino.cc/en/software).
2. Boards Manager → install **Raspberry Pi Pico/RP2040** by Earle F. Philhower.
3. Open `vendor/usb_hid/pico/GatherionHidKeyboard/GatherionHidKeyboard.ino`
   (keep `GatherionHidKeys.h` in the same folder).
4. Board: **Raspberry Pi Pico**. First flash: hold **BOOTSEL**, plug USB, then **Upload**.
5. Auto-bind COM:

```bat
python tools\set_hid_com.py --wait-plug
```

### Leonardo / Pro Micro

1. Arduino IDE → Board **Arduino Leonardo** (or Pro Micro 5V / 16MHz).
2. Open `vendor/usb_hid/leonardo/GatherionHidKeyboard/GatherionHidKeyboard.ino`
3. Upload → `python tools\set_hid_com.py --wait-plug`

---

## PC wiring

```bat
python tools\hid_com_smoke.py --list
python tools\hid_com_smoke.py --wait-plug --keys tab,1,2,3,f --write-settings
```

Or manually:

```bat
set GATHERION_HID_COM=COM5
python tools\hid_com_smoke.py --port COM5 --keys tab,1,2,3,f
```

Focus Notepad — Tab / digits / `f` must appear from the USB device.

Then start Gatherion with the same env (or `input.serial_port` already written):

```bat
set GATHERION_HID_COM=COM5
baslat-mcp.bat
```

Look for `SERIAL_HID_OK` / `SerialHID` backend in logs.  
When SerialHID (or NetworkHID) is connected, combat Tab/1/2/3 are **hardware-only** — soft SendInput is not cascaded.  
`GATHERION_HID_COM` wins over `GATHERION_HID_TCP` if both are set.

| Setting | Example |
|---------|---------|
| Env `GATHERION_HID_COM` | `COM5` |
| `config/settings.json` → `input.serial_port` | `"COM5"` |
| Helper | `python tools\set_hid_com.py COM5` |

Baud: **115200** (fixed in `SerialHidAdapter`).

---

## Protocol

Newline UTF-8 (same as Android NetworkHID):

```
KEYDOWN tab
KEYUP tab
KEY 1 80
KEYDOWN alt
KEYUP alt
PING
```

`PING` → firmware replies `PONG`.

---

## Files

| Path | Role |
|------|------|
| `vendor/usb_hid/pico/GatherionHidKeyboard/` | Pico firmware sketch |
| `vendor/usb_hid/leonardo/GatherionHidKeyboard/` | Leonardo / Pro Micro sketch |
| `vendor/usb_hid/common/GatherionHidKeys.h` | Shared key-name map |
| `vendor/usb_hid/README.md` | Short shopping + flash index |
| `docs/PICO_OPERATOR_TR.md` | Turkish operator card (3 steps) |
| `tools/flash_pico.bat` / `.ps1` | UF2 / picotool / BOOTSEL helper |
| `tools/set_hid_com.py` | Auto-detect COM → settings.json |
| `tools/hid_com_smoke.py` | PC → COM protocol smoke (+ `--wait-plug`) |
| `tools/hid_com_util.py` | Shared COM list / wait helpers |
| `core/hardware_input_port.py` | `SerialHidAdapter` (already wired) |
| `docs/ANDROID_HID_KEYBOARD.md` | Phone BT-HID when PC has Bluetooth |
| `vendor/interception/` | Optional staged installer (do **not** auto-install; ban risk) |

---

## Blockers (honest)

- Firmware is **staged / source-ready**. Proof requires physical Pico/Leonardo plugged in.
- Soft keyboard / mouse skill clicks are **not** claimed as working skill casts.
- Android BT-HID remains available after you add a **USB BT dongle**; do not delete that stack.
