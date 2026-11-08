# Galaxy S8 SM-G950F → USB HID klavye (root + gadget)

PC’de Bluetooth yok: **root’lu SM-G950F + USB HID Gadget**.
Windows telefonu **gerçek USB klavye** görür.

**Master rehber (root adımları + 1B/KVM):** `docs/ANDROID_USB_NO_BT_TR.md`  
**Canlı checklist (adb probe, flash yok):** `python tools\s8_root_checklist.py`

```
PC Gatherion  --TCP/UDP (Wi‑Fi veya adb forward)-->  android_usb_hid_server.py
                                                      --/dev/hidg0--> USB --> Windows
```

## Kilitli hedef

| Alan | Değer |
|------|--------|
| Model | **SM-G950F** (Exynos dreamlte) — operatör onaylı |
| OS | **Android 9** |
| Unlock | Canlı `flash.locked=0` + `orange` → OEM **SKIP**. **G950U değil** |
| Download mode | Vol− + Bixby + Power |
| TWRP / Magisk | `vendor/android_hid/s8_root/` · Odin **AP** = dreamlte `.img.tar` |

---

## Tutarsızlık / risk (dürüst)

| Konu | Durum |
|------|--------|
| Soft gibi tuş yutma | HID açılırsa **hayır** |
| Ban / ToS | Otomasyon riski durur |
| Root / Knox | Unlock + Magisk → Knox trip, garanti/banka |
| Kernel | Stok S8’de `/dev/hidg0` **sık yok**; NetHunter / USB Gadget Tool / HID’li kernel |
| Brick | TWRP/Odin hatası pahalı |
| Rootless | Stok USB ≠ HID — `tools/rootless_hid_bridge.py` sadece PC Interception/soft |

---

## Önkoşul checklist (G950F · Android 9)

1. `adb shell getprop ro.product.model` → **SM-G950F**  
2. `ro.build.version.release` → **9**; `ro.product.device` → **dreamlte**  
3. Yedek alındı  
4. Unlock teyit: `flash.locked=0` / `orange` ise OEM unlock **atla**; USB debugging açık  
5. Odin AP ← staged TWRP → Magisk ZIP → `adb shell su -c id` → `uid=0`  
6. USB HID gadget → `ls -l /dev/hidg0` **var**  
7. Data USB kablo; Aygıt Yöneticisi’nde HID Keyboard  
8. Termux/`python` ile script **root** çalışır  

Detay: `vendor/android_hid/s8_root/README.md` · `python tools\s8_root_checklist.py`  
`hidg0` yoksa bot kurma; önce gadget.

---

## PC tek komut (root sonrası)

```bat
python tools\setup_android_usb_hid.py --check
python tools\setup_android_usb_hid.py --push --forward --write-settings --enable-gadget --start-server
python tools\hid_tcp_smoke.py --host 127.0.0.1:18765
```

Env: `set GATHERION_HID_TCP=127.0.0.1:18765`

Eski bat: `tools\push_s8_hid_server.bat`

---

## Root yokken (ara)

```bat
python tools\interception_status.py
python tools\rootless_hid_bridge.py
set GATHERION_HID_TCP=udp://127.0.0.1:18766
```

veya Pico: `docs/USB_HID_KEYBOARD.md` / `docs/PICO_OPERATOR_TR.md`  
BT dongle yolu: `docs/ANDROID_HID_KEYBOARD.md`

---

## Termux / adb elle

Dosya: `vendor/android_hid/usb_gadget/android_usb_hid_server.py`

```bash
su -c "python /data/local/tmp/android_usb_hid_server.py"
```

Protokol: `PING` / `KEYDOWN tab` / `KEYUP tab` / `KEY 1 80`
