# GatherionHidBridge (Android)

Makes this phone a **Bluetooth HID keyboard** for the Windows PC and accepts
Gatherion’s line protocol over **TCP :8765**.

```
PC Gatherion --TCP KEYDOWN--> phone app --BT HID report--> Windows / AION2
```

## Install (prebuilt)

Phone charged later — no Studio required:

```bat
adb install -r F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk
```

## Rebuild (optional)

```bat
cd F:\a2\android\GatherionHidBridge
gradlew.bat assembleDebug
copy app\build\outputs\apk\debug\app-debug.apk F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk
```

Min SDK 28 (Android 9) — required for `BluetoothHidDevice`.

## First run

1. Grant Bluetooth / Nearby devices.
2. **Start HID service**.
3. On **Windows**: Settings → Bluetooth → add device → select the phone (keyboard).
4. In app: **Connect to PC** (pick the paired PC) until **HID connected**.
5. Confirm phone IP on screen; PC:

```bat
set GATHERION_HID_TCP=<phone-ip>:8765
python F:\a2\tools\hid_tcp_smoke.py --host <phone-ip>
```

## Protocol

Newline-terminated UTF-8 (identical to Leonardo serial):

- `KEYDOWN <key>`
- `KEYUP <key>`
- `KEY <key> <hold_ms>`
- `PING` → replies `PONG`

Keys used by Gatherion: `tab`, `1`–`5`, `f`, `alt`, `q`, `r`, `v`, `e`, `space`, `esc`, …

## Notes

- Keep the foreground service running; do not swipe the app away mid-farm.
- Some OEMs kill BT HID in battery saver — exclude the app.
- This is **not** USB gadget mode (that needs root). See `docs/ANDROID_HID_KEYBOARD.md`.
