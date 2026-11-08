# Android USB HID (Bluetooth yok) — Gatherion

PC’de Bluetooth yok; **kök hedef** telefonun **USB kablo** ile gerçek HID klavye olmasıdır.
Bu yol **root gerektirir**. Stok Android USB’de sadece MTP/ADB olur; HID gadget değil.

| Yol | Root? | Bu gece? |
|-----|-------|----------|
| **A. SM-G950F root + `/dev/hidg0` + TCP/UDP** | **Evet** | Asıl hedef (Bölüm 1–2) |
| **B. Rootless PC köprü (1B)** | Hayır | Stok telefon ≠ USB HID; KEY → Interception/soft — `tools/rootless_hid_bridge.py` |
| **C. Interception (PC kernel)** | Hayır (telefon) | Ara test — `vendor/interception/` + `tools/interception_status.py` |
| **D. Yazılımsal KVM** | Hayır | Barrier / MWB / Input Director — soft; AION2 yutabilir (Bölüm KVM) |
| **E. Pico / Leonardo USB** | n/a | Telefon yoksa birincil yedek — `docs/USB_HID_KEYBOARD.md` |
| **F. GatherionHidBridge BT** | Hayır | PC’de BT dongle şart — `docs/ANDROID_HID_KEYBOARD.md` |

```
PC Gatherion  --TCP/UDP (Wi‑Fi veya adb forward)-->  Termux/root server
                                                      |
                                                      v
                                                 /dev/hidg0
                                                      |
                                                      v
                                              USB kablo → Windows (gerçek klavye)
```

Komut yolu ağ; **tuş sinyali USB**. Soft `SendInput` değil (yol A).

---

## Kilitli hedef cihaz (operatör onaylı)

| Alan | Değer |
|------|--------|
| Model | **SM-G950F** (Exynos, EU/global dreamlte) |
| Yazılım | **Android 9** (bootloader `G950FXXU…` ailesi) |
| Unlock | Canlı: `flash.locked=0` + `verifiedbootstate=orange` → **zaten unlock** (OEM adımını atla). G950U değil. |
| Kablo | **Data** USB (şarj-only değil) |
| Root (canlı ölçüm) | Unlock olsa da genelde **unrooted** (`su` yok) → TWRP + Magisk şart |
| Live serial (örnek) | `ce0617164cd200600d` — `device` + model `SM-G950F` |
| Flash staging | `vendor/android_hid/s8_root/` (TWRP tar + Magisk APK/ZIP) |

`*#1234#` veya Ayarlar’dan modeli doğrula. **G950U / G950W** ise bu gece checklist’i **uygulama** — Pico veya Interception’a geç.

S8 kısa kart: `docs/S8_USB_HID_GADGET_TR.md`  
Dosyalar: `vendor/android_hid/usb_gadget/`

---

## Dürüst riskler (okumadan flash yok)

| Konu | Gerçek |
|------|--------|
| Knox / garanti | OEM unlock + özel recovery → **Knox 0x1**, garanti/banka/Samsung Pay kırılır |
| Brick | Yanlış Odin paketi / TWRP → soft/hard brick riski |
| Ban / ToS | HID “gerçek klavye” sınıfı; otomasyon riski yine var |
| Stok S8 kernel | Çoğu stok romda **`/dev/hidg0` yok** — ConfigFS HID için NetHunter / USB Gadget Tool / HID’li custom kernel gerekebilir |
| Bu makine | Telefondan **uzaktan OEM-unlock / Odin yapamaz** — adımları **sen** çalıştırırsın |

---

## Bölüm 0 — Root yokken (1B + ara test)

**Dürüst:** Stok SM-G950F USB’de gerçek HID klavye **olamaz**. ADB ile “telefon HID” illüzyonu yok.

### 0.1 Rootless KEY köprü (tercih edilen 1B)

PC dinler; bot aynı protokolü kullanır; inject **Interception** (varsa) veya soft SendInput.
`NetworkHidAdapter` UDP için **PONG** ister — köprü kapalıysa `NETWORK_HID_MISSING` → cascade **Interception** (tuşlar boşa gitmez).

```bat
python tools\interception_status.py
tools\start_rootless_hid.bat
REM elle:
python tools\set_hid_tcp.py --rootless
python tools\rootless_hid_bridge.py --no-soft
python tools\hid_tcp_smoke.py --host udp://127.0.0.1:18766 --keys tab,1,2,3
python tools\rootless_hid_bridge.py --self-smoke --no-soft
```

`config/settings.json` varsayılan (RMM beklerken): `input.hid_tcp=udp://127.0.0.1:18766`.  
1A sonrası: `python tools\set_hid_tcp.py --adb-forward` → `127.0.0.1:18765`.

Alternatif: Interception sürücüsü kuruluysa bot cascade zaten Interception kullanır — localhost köprü şart değil (`INTERCEPTION_OK` log).

ADB rolü (opsiyonel): sadece cihaz durumu / ileride root push. Stok telefonda KEY üreten USB gadget yok.

### 0.2 Interception (elle)

`vendor/interception/README.md` — installer’ı **elle** çalıştır; bot sessiz kurmaz (UAC sizin).

```bat
python tools\interception_status.py
tools\install_interception_admin.bat
```

Status aracı elevated mi söyler; bat `Start-Process -Verb RunAs` ile UAC ister. Alternatif Admin CMD:

```bat
cd /d F:\a2\vendor\interception
install-interception.exe /install
pause
```

Smoke (sürücü hazırsa / `INTERCEPTION_OK`): `python tools\interception_status.py --smoke --confirm-smoke --key tab`  
1B tek config: `python tools\set_hid_tcp.py --rootless` → `GATHERION_HID_TCP=udp://127.0.0.1:18766`

### 0.3 Pico

`docs/USB_HID_KEYBOARD.md` + `GATHERION_HID_COM=COMx`.

Soft klavye AION2’de yutulur — “telefonsuz soft” çözüm değil; Interception/Pico/root HID gerekir.

---

## Bölüm KVM — yazılımsal ağ klavye paylaşımı (madde 2)

Bunlar **vendor edilmez**; operatör isterse kurar. Hepsi Windows’a **yazılımsal** girdi verir (gerçek USB HID değil). AION2 Raw Input çoğu soft yolu **yutar**.

| Araç | Ne | Gatherion ile |
|------|-----|----------------|
| **Barrier** (veya Input Leap) | Açık kaynak KVM; PC’ler arası klavye/fare | Soft; AION2 için güvenilir değil |
| **Mouse Without Borders** (Microsoft) | LAN çoklu PC paylaşımı | Soft |
| **Input Director** | Windows KVM | Soft |

**Neden soft/virtual:** Kaynak PC tuşu yakalar → ağ → hedefte SendInput/benzeri. USB gadget / Interception / Pico değil.

Gatherion birincil yol: **Network/Serial HID → Interception → FakerInput → soft**. KVM bu zincire **bağlanmaz**; sadece “başka PC’den oyna” senaryosu.

---

## Bölüm 1 — SM-G950F · Android 9 root (bu gece checklist)

**Yedek al** (foto/WhatsApp). Knox trip geri dönmez.  
Hedef model: **yalnızca SM-G950F** (Exynos dreamlte).

Canlı paket (flash yok; adb probe + kopyala-yapıştır adımlar):

```bat
python tools\s8_root_checklist.py
```

### 1.1 Bilgi topla (PC)

```bat
python tools\s8_root_checklist.py
adb devices -l
adb shell getprop ro.product.model
adb shell getprop ro.boot.bootloader
adb shell getprop ro.build.version.release
adb shell getprop ro.product.device
adb shell getprop ro.boot.verifiedbootstate
adb shell getprop ro.boot.flash.locked
```

`verifiedbootstate=orange` + `flash.locked=0` → bootloader **zaten unlock** → **OEM unlock adımını atla**; yine de `su` yoksa Magisk/TWRP gerekir.

**Beklenen (canlı kilitli hedef):**

| Prop | Beklenen |
|------|----------|
| `ro.product.model` | `SM-G950F` |
| `ro.product.device` | `dreamlte` |
| `ro.build.version.release` | `9` |
| `ro.boot.bootloader` | `G950F…` ile başlar (örn. `G950FXXUCDZE7`) |
| `ro.boot.flash.locked` | `0` → OEM unlock **SKIP** |
| `ro.boot.verifiedbootstate` | `orange` → unlock teyidi |

`SM-G950U` görürsen **dur** — bu checklist Exynos içindir.

### 1.2 OEM unlock (yalnızca hâlâ kilitliyken)

`flash.locked=0` / `orange` ise bu bölümü **geç**. Aksi halde:

1. Ayarlar → Telefon hakkında → Yapı numarasına 7× → Geliştirici seçenekleri  
2. **OEM kilit açma** + **USB hata ayıklama** aç; bu PC’yi yetkilendir  
3. Telefonu kapat → **Download (Odin) mode**: **Vol− + Bixby + Power** (G950F)  
4. XDA / güncel G950F rehberine göre bootloader unlock.  
5. Unlock sonrası genelde **factory reset**

### 1.2b RMM / KG Prenormal (stok sonrası “only official binaries”)

Unlock props (`flash.locked=0` / orange) olsa bile Download’da **RMM/KG Prenormal** custom AP’yi engeller.  
Check-only kart (flash yok): `docs/S8_DOWNLOAD_CHECK_TR.md` — KG/RMM/OEM LOCK oku, **Vol− + Power** ile çık.  
Bekleme checklist: `docs/S8_RMM_WAIT_CHECKLIST.md` (~7 gün, OEM ON, custom flash yok).  
Güvenli Magisk sırası (multidisabler → Magisk, flash yok bu oturumda): `docs/S8_MAGISK_SEQUENCE_SAFE.md`.  
Root sonrası tek dosya: `tools\after_magisk_hid.bat`.

### 1.3 TWRP + Magisk (G950F · Odin AP)

Staging klasörü (tercih): `vendor/android_hid/s8_root/` — `python tools\s8_root_checklist.py` hash + yolları basar.

| Ne | Link / dosya |
|----|------|
| TWRP dreamlte tar | `vendor/android_hid/s8_root/twrp-3.7.0_9-0-dreamlte.img.tar` · https://dl.twrp.me/dreamlte/twrp-3.7.0_9-0-dreamlte.img.tar |
| Multidisabler v3.1 | `vendor/android_hid/s8_root/multidisabler-samsung-3.1-flashable.zip` · https://github.com/ianmacd/multidisabler-samsung/releases/tag/v3.1 |
| Magisk | `vendor/android_hid/s8_root/Magisk-v31.1.apk` (+ aynı bayt `Magisk-v31.1.zip`) · https://github.com/topjohnwu/Magisk/releases |
| Odin | XDA / Samsung Odin 3.x (repo otomatik indirmez) |

**Birebir gece flash (yalnızca Prenormal temiz + operatör onayı):**

1. Yedek. Knox trip kabul. Prenormal yok (Download metni).  
2. Download mode: **Vol− + Bixby + Power**. Odin’de mavi COM.  
3. Odin: **Auto Reboot OFF**, Re-Partition OFF → **yalnızca AP** = TWRP `.img.tar` → Start → PASS.  
4. Hemen recovery: **Vol+ + Bixby + Power** (sisteme boot etme; stok recovery TWRP’yi ezer).  
5. TWRP: şifreli data → **Format Data**; önce **`multidisabler-samsung-3.1-flashable.zip`**, sonra `Magisk-v31.1.zip` Install → Reboot System.  
6. **vbmeta (A9):** Klasik Exynos S8 Pie’de genelde ayrı `vbmeta` slot yok — A10+ “vbmeta disable” img’lerini **flash etme**. Magisk boot patch yeterli olmalı; twrp.me dm-verity notunu oku.  
7. Doğrula:

```bat
adb shell su -c id
```

`uid=0(root)` şart. Tekrar: `python tools\s8_root_checklist.py`

### 1.4 HID gadget (stok kernel gerçeği)

Root **tek başına** `/dev/hidg0` açmaz.

1. Magisk modülü / **USB Gadget Tool** (tejado) veya **NetHunter** HID  
2. Best-effort ConfigFS:

```bat
python tools\setup_android_usb_hid.py --push --enable-gadget
```

3. Telefonda:

```bat
vendor\android_sdk\platform-tools\adb.exe shell su -c "ls -l /dev/hidg0"
```

**Yoksa** bot TCP HID kurma — önce kernel/gadget. Stok Android 9 G950F’de sık sonuç: script FAIL → NetHunter/custom kernel.

---

## Bölüm 2 — Sunucu + PC köprüsü (root + hidg0 sonrası)

### 2.1 Push + forward (PC)

```bat
python tools\setup_android_usb_hid.py --check
python tools\setup_android_usb_hid.py --push --forward --write-settings
python tools\setup_android_usb_hid.py --enable-gadget
python tools\setup_android_usb_hid.py --start-server
```

`settings.json` → `input.hid_tcp` = `127.0.0.1:18765`  
Env: `set GATHERION_HID_TCP=127.0.0.1:18765`

Wi‑Fi (adb forward yok):

```bat
python tools\set_hid_tcp.py TELEFON_IP:8765
set GATHERION_HID_TCP=TELEFON_IP:8765
```

UDP:

```bat
set GATHERION_HID_TCP=udp://TELEFON_IP:8765
```

### 2.2 Telefonda elle

```bash
su -c "python /data/local/tmp/android_usb_hid_server.py"
```

Eski ad: `s8_hid_tcp_server.py` → `usb_gadget/android_usb_hid_server.py` sarmalayıcısı.

### 2.3 Smoke + bot

```bat
python tools\hid_tcp_smoke.py --host 127.0.0.1:18765 --keys tab,1,2,3
```

Notepad’te tuşlar **USB klavye** gibi gelmeli. Gatherion log: `NETWORK_HID_OK` / backend `NetworkHID`.

Windows Aygıt Yöneticisi’nde telefon bağlıyken **HID Keyboard** (gadget açıksa).

---

## Bölüm 3 — Gatherion ayarları

| Anahtar | Örnek |
|---------|--------|
| `GATHERION_HID_TCP` | `127.0.0.1:18765` / `udp://127.0.0.1:18766` (1B) / `udp://telefon:8765` |
| `config/settings.json` → `input.hid_tcp` | aynı |
| `GATHERION_HID_COM` | Pico varsa COM; **COM, TCP’den önce gelir** |

Cascade: **Network/Serial HID → Interception → FakerInput → soft**.

---

## Dosya haritası

| Path | Rol |
|------|-----|
| `docs/ANDROID_USB_NO_BT_TR.md` | Bu master rehber |
| `docs/S8_USB_HID_GADGET_TR.md` | S8 / G950F kısa özet |
| `vendor/android_hid/usb_gadget/android_usb_hid_server.py` | Root TCP+UDP → hidg0 |
| `vendor/android_hid/usb_gadget/enable_hid_gadget.sh` | ConfigFS best-effort |
| `tools/setup_android_usb_hid.py` | adb push / forward / check |
| `tools/rootless_hid_bridge.py` | 1B KEY → Interception/soft |
| `tools/interception_status.py` | DLL/driver durumu (install yok; Admin komutu basar) |
| `tools/s8_root_checklist.py` | G950F canlı adb + TWRP/Magisk checklist (OEM skip if unlocked) |
| `vendor/android_hid/s8_root/` | Staged TWRP dreamlte `.img.tar` + Magisk APK/ZIP |
| `tools/set_hid_tcp.py` | `input.hid_tcp` yaz (`--rootless` / `--adb-forward`) |
| `vendor/interception/` | Ara test (elle install; `install-interception.exe` staged) |
| `docs/USB_HID_KEYBOARD.md` / `docs/PICO_OPERATOR_TR.md` | Pico yedek |
| `docs/ANDROID_HID_KEYBOARD.md` | BT dongle + APK yolu |

---

## Bu gece (SM-G950F özet)

1. Canlı `adb`: **SM-G950F** / Android **9** / `flash.locked=0` + `orange` → OEM **SKIP**; `su` yok → Magisk şart.  
2. Yedek → Odin **AP** = staged TWRP (`vendor/android_hid/s8_root/`) → Magisk ZIP → `su -c id`.  
3. `/dev/hidg0` aç (gadget tool / NetHunter / script).  
4. `python tools\setup_android_usb_hid.py --push --forward --write-settings --start-server`  
5. Smoke → bot `GATHERION_HID_TCP`.  
6. Root bitene kadar: `rootless_hid_bridge` + Interception veya Pico — **gerçek telefon USB HID değil**.
