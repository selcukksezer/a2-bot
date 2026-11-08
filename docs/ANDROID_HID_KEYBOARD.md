# Android phone as Gatherion HID keyboard

AION2 drops synthetic `SendInput` / soft keyboard. Gatherion already prefers a
**real HID keyboard** path via `SerialHidAdapter` (`KEYDOWN` / `KEYUP` / `KEY`
line protocol). This doc makes an **Android phone** that path — “phone as Arduino”.

UAC / High-bot / RMB-only soft-aim is **not** the fix for missing keyboard.

---

## Ranked methods (honest)

| Rank | Method | Root? | Works as real PC keyboard? | Gatherion talks how? | Tonight? |
|------|--------|-------|----------------------------|----------------------|----------|
| **0** | **USB Pico / Leonardo HID** (no Bluetooth) | n/a | **Yes** | `GATHERION_HID_COM=COMx` | **Primary if PC has no BT** — see `docs/USB_HID_KEYBOARD.md` |
| **1** | **Bluetooth HID peripheral + TCP** (`GatherionHidBridge`) | **No** (Android 9+) | **Yes** — needs **PC Bluetooth** (dongle OK) | Wi‑Fi TCP `KEYDOWN tab` | After USB BT dongle |
| **2** | **USB HID gadget** (`/dev/hidg0`) on rooted S8 + Termux TCP/UDP | **Yes** + HID kernel | **Yes** — USB keyboard, **no PC Bluetooth** | `GATHERION_HID_TCP` (tcp/udp) | Master: `docs/ANDROID_USB_NO_BT_TR.md` · S8: `docs/S8_USB_HID_GADGET_TR.md` · `vendor/android_hid/usb_gadget/` |
| **3** | Play Store “Bluetooth keyboard” apps (Keywe, etc.) | No | Yes for *manual* taps | **No PC command API** | Useless for bot alone |
| **4** | USB “keyboard” apps without root / gadget | — | **No** on stock Android | — | Dead end |
| **5** | Rootless PC KEY bridge → Interception/soft | No | **Not** USB HID — PC inject | `GATHERION_HID_TCP=udp://127.0.0.1:18766` | Interim: `tools/rootless_hid_bridge.py` |
| **✗** | Soft keyboard / mouse-only skill click as “keyboard” | — | **No** for AION2 skills | — | Rejected |
| **✗** | Game DLL inject / memory write | — | — | — | Rejected |

Locked phone target for USB gadget (Rank 2): **SM-G950F** Android 9 — see `docs/ANDROID_USB_NO_BT_TR.md`.

### Why not plain USB without root?

Stock Android is a **USB device** for MTP/ADB, not a configurable HID gadget.
Writing `/dev/hidg0` needs **root + ConfigFS** (e.g. [tejado/android-usb-gadget](https://github.com/tejado/android-usb-gadget/)).
No-root USB-as-keyboard is not a reliable path. Use Rank 5 only as an honest PC-side interim.

### Why Bluetooth HID?

Android 9+ `BluetoothHidDevice` API lets an app register as a **keyboard peripheral**.
Windows pairs it like any BT keyboard — same trust class as Leonardo for games.
Gatherion cannot “push keys into” a UI-only app, so we add a **TCP line server** on the phone that speaks the existing Gatherion protocol.

Your existing **Bluetooth keyboard peripheral is irrelevant** — Windows is the HID *host*; the phone must be the HID *device*.

---

## Recommended stack (Rank 1)

```
Gatherion (PC)  --Wi‑Fi TCP-->  GatherionHidBridge (phone)
                                      |
                                      v
                               Bluetooth HID reports
                                      |
                                      v
                               Windows (AION2)  ← real keyboard
```

Protocol (same as Leonardo serial):

```
KEYDOWN tab
KEYUP tab
KEY 1 80
KEYDOWN alt
KEYUP alt
PING
```

---

## Tonight — what you do on the phone

**Prebuilt APK (no Android Studio needed):**  
`F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk`

```bat
adb install -r F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk
```

(Or copy the APK to the phone and open it. USB debugging optional if you sideload via file share.)

1. ~~Install Android Studio~~ — skip if using the prebuilt APK above.
2. ~~Open / Run from Android Studio~~ — only if rebuilding from `android/GatherionHidBridge`.
3. On phone: grant **Nearby devices / Bluetooth** permission.
4. In the app: **Start HID** → pair phone from **Windows Bluetooth settings** (phone appears as a keyboard).
5. Tap **Connect to PC** in the app (select your PC if listed) until status = **HID connected**.
6. Note phone Wi‑Fi IP (app shows it), keep phone unlocked / app in foreground service.
7. On PC (PowerShell):

```bat
set GATHERION_HID_TCP=PHONE_IP:8765
python tools\hid_tcp_smoke.py --host PHONE_IP --keys tab,1,2,3,f
```

8. Focus Notepad on PC — you should see Tab / digits / `f` from the phone HID.
9. Start Gatherion with the same env (or `config/settings.json` → `input.hid_tcp`).

```bat
set GATHERION_HID_TCP=192.168.1.20:8765
baslat-mcp.bat
```

(or unelevated / normal start — **HID keyboard does not need admin bot**)

10. In bot logs look for: `NETWORK_HID_OK: keyboard path via tcp://... (tcp)` and backend `NetworkHID`
   (COM Leonardo still logs `SERIAL_HID_OK`).

Firewall: allow inbound TCP **8765** on the phone’s LAN (Android usually allows the app; PC is the **client**, so PC firewall is fine).

---

## PC wiring (already in Gatherion)

| Setting | Example |
|---------|---------|
| Env `GATHERION_HID_TCP` | `192.168.1.20:8765` |
| Env `GATHERION_HID_COM` | `COM5` (Leonardo; wins over TCP if both set) |
| `config/settings.json` → `input.hid_tcp` | `"192.168.1.20:8765"` |
| `config/settings.json` → `input.serial_port` | `"COM5"` |

Cascade (keyboard): **Network/Serial HID → Interception → FakerInput → soft SendInput**.

Smoke:

```bat
python tools\hid_tcp_smoke.py --host 192.168.1.20
```

---

## Rank 2 (rooted USB gadget) — short

Full TR checklist (S8 Android 9 OEM unlock / TWRP / Magisk / Knox / hidg0):
**`docs/ANDROID_USB_NO_BT_TR.md`**.

1. Root + kernel with `CONFIG_USB_CONFIGFS_F_HID` (stock S8 often needs NetHunter / USB Gadget Tool).
2. Enable HID gadget ([android-usb-gadget](https://github.com/tejado/android-usb-gadget/)) or `enable_hid_gadget.sh`.
3. Phone: `android_usb_hid_server.py` → `/dev/hidg0` (`vendor/android_hid/usb_gadget/`).
4. PC: `python tools/setup_android_usb_hid.py --push --forward --write-settings` then smoke / bot.

---

## Ban / anti-cheat note (brief)

Emulating a **standard Bluetooth/USB keyboard** is the same class as a physical
Leonardo or a real BT keyboard. It is still automation: ToS / anti-cheat risk
exists; do not treat HID as “undetectable”. Prefer same machine, local Wi‑Fi only,
no cloud relay.

---

## Files

| Path | Role |
|------|------|
| `core/hardware_input_port.py` | `NetworkHidAdapter` + `open_hid_adapter` |
| `core/input_driver.py` | Uses TCP/COM endpoint in keyboard cascade |
| `android/GatherionHidBridge/` | Phone BT-HID + TCP bridge (build in Android Studio) |
| `tools/hid_tcp_smoke.py` | PC → phone protocol test |
| `docs/ANDROID_HID_KEYBOARD.md` | This doc (BT primary) |
| `docs/ANDROID_USB_NO_BT_TR.md` | No-BT USB root path (S8) |
| `tools/setup_android_usb_hid.py` | adb push / forward / hidg0 check |

---

## Staged debug APK (PC already built)

- APK: `F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk`
- Install note: `F:\a2\vendor\android_hid\INSTALL_APK.md`
- When USB debugging is on:
  `adb install -r F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk`
  `adb shell am start -n com.gatherion.hidbridge/.MainActivity`
