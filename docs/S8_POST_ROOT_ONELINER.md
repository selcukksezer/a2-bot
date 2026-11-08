# SM-G950F — next after Magisk (one-liner)

Goal: live HID keys via `/dev/hidg0`. Docs alone ≠ done.

## Preferred one-shot

```bat
tools\after_magisk_hid.bat
```

Runs: `--check` → `--push --enable-gadget --forward --write-settings --start-server` → `hid_tcp_smoke.py`.

## Manual equivalent

```bat
adb shell su -c id && adb shell su -c "sh /data/local/tmp/enable_hid_gadget.sh" && python tools\setup_android_usb_hid.py --push --forward --write-settings --enable-gadget --start-server
```

If `enable_hid_gadget.sh` is not on phone yet, the setup script `--push` copies it. Order of intent:

1. `adb shell su -c id` → must show `uid=0(root)`
2. Enable gadget: `su -c "sh /data/local/tmp/enable_hid_gadget.sh"` (or USB Gadget Tool / NetHunter)
3. `python tools\setup_android_usb_hid.py --push --forward --write-settings --enable-gadget --start-server`
4. Smoke: `set GATHERION_HID_TCP=127.0.0.1:18765` then `python tools\hid_tcp_smoke.py --host 127.0.0.1:18765`

Interim (no phone HID yet): `python tools\interception_status.py` — never auto-installs.
