# Yarın — USB Bluetooth dongle (Rank 1) · scope item #4

**Amaç:** PC’de BT yok → dongle tak → telefon `GatherionHidBridge` ile gerçek HID klavye.  
**Şimdi:** `#3` Interception OK · `#1B` rootless OK · **1A** Magisk/RMM ~7g açık. Goal **OPEN** (`tools/GOAL_AUDIT_HID.md`).  
**Pipeline uyumu:** Aynı `GATHERION_HID_TCP` / `NetworkHidAdapter` / `input_driver` cascade — Pico COM (`GATHERION_HID_COM`) ile yan yana; COM önce gelir.

## Alışveriş

| Al | Not |
|----|-----|
| **USB Bluetooth dongle** (PC) | Windows 10/11 uyumlu; BLE/BT 4.0+ yeterli |
| Telefon (S8 veya Android 9+) | Aynı Wi‑Fi; BT açık |
| *(opsiyon)* Pico | Dongle yoksa USB HID — `docs/PICO_OPERATOR_TR.md` |

## APK

`F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk`

```bat
adb install -r F:\a2\vendor\android_hid\GatherionHidBridge-debug.apk
```

## 5 adım (dongle sonrası)

1. Dongle’ı PC’ye tak → Windows Bluetooth aç → çalıştığını doğrula.
2. APK kur → uygulamada Bluetooth izni → **Start HID**.
3. Windows Bluetooth’tan telefonu (klavye) eşleştir → uygulamada **Connect to PC** → **HID connected**.
4. PC’de telefon IP ile:
   ```bat
   set GATHERION_HID_TCP=TELEFON_IP:8765
   python tools\hid_tcp_smoke.py --host TELEFON_IP --keys tab,1,2,3,f
   ```
5. Notepad’te tuş gör → aynı `GATHERION_HID_TCP` ile botu başlat (`NETWORK_HID_OK`).

Detay: `docs/ANDROID_HID_KEYBOARD.md`
