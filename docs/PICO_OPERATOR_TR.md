# Pico operatör kartı (klavye HID) — 3 adım

**Hedef:** AION2 Tab + skill 1/2/3 gerçek USB klavye. Soft SendInput yetmez. Mouse-only istemiyoruz.  
**Donanım masada yoksa hedef tamamlanmaz.**

## Satın al → flash → COM → bot

### 1) Satın al
- **Raspberry Pi Pico (RP2040)** — birincil (Bluetooth gerekmez)
- Alternatif: Arduino Leonardo / Pro Micro
- *(İsteğe bağlı sonra)* PC’ye USB Bluetooth dongle → Android telefon yolu (`docs/ANDROID_HID_KEYBOARD.md`)

### 2) Flash
```bat
tools\flash_pico.bat
```
İlk flaş: Pico’da **BOOTSEL** basılı tut → USB tak → Arduino IDE (Philhower Pico) ile  
`vendor\usb_hid\pico\GatherionHidKeyboard\` Upload  
veya hazır `.uf2` dosyasını `RPI-RP2` sürücüsüne sürükle.

### 3) COM → bot
```bat
python tools\set_hid_com.py --wait-plug
python tools\hid_com_smoke.py --keys tab,1,2,3
baslat-mcp.bat
```
Log’da `SerialHID` / `SERIAL_HID_OK` ara. Notepad’de Tab/1/2/3 Pico’dan gelmeli.

---

| Ayar | Örnek |
|------|--------|
| `GATHERION_HID_COM` | `COM5` |
| `config/settings.json` → `input.serial_port` | `"COM5"` |

`GATHERION_HID_COM` varsa TCP’den önce gelir. Baud: 115200.

## Android (opsiyonel)
PC’de Bluetooth yok → Wi‑Fi HID yapmaz. BT dongle sonrası telefon köprüsü: `docs/ANDROID_HID_KEYBOARD.md`.

## Yazılım yedek (ban notu)
`vendor/interception/` altına installer **indirilebilir** (kurma / UAC yok). Kernel sürücü anti-cheat riski taşır — **birincil öneri Pico**.  
Durum: `python tools\interception_status.py` — **asla** otomatik install yok.  
Rootless 1B: `python tools\rootless_hid_bridge.py` + `GATHERION_HID_TCP=udp://127.0.0.1:18766`.
