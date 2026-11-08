# SM-G950F — safest Magisk sequence (PREPARED — do not flash yet)

**HOLD:** Do **not** run this until:

1. User explicitly confirms flash, **and**
2. Download mode shows RMM/KG **not** Prenormal (see `docs/S8_RMM_WAIT_CHECKLIST.md`), **and**
3. Live evidence still shows OEM / bootloader unlock: `ro.boot.flash.locked=0` + `verifiedbootstate=orange` (re-run `python tools\s8_root_checklist.py`).

Flashing while Prenormal → “Only official released binaries…” / brick risk.

## Staged artifacts (already on disk)

| Order | File | Role |
|------:|------|------|
| 0 | `vendor/android_hid/s8_root/twrp-3.7.0_9-0-dreamlte.img.tar` | Odin **AP** only |
| 1 | `vendor/android_hid/s8_root/multidisabler-samsung-3.1-flashable.zip` | TWRP Install **FIRST** (Ian Macdonald v3.1) |
| 2 | `vendor/android_hid/s8_root/Magisk-v31.1.zip` | TWRP Install **SECOND** (same bytes as `.apk`) |
| — | `odin/Odin3_v3.14.1/Odin3 v3.14.1.exe` | Admin; Auto Reboot **OFF** for TWRP |

Multidisabler SHA256 (flashable zip): `96C42E8C7BE0EAE7EA0E07862E5A62B8BA8BA6BD321B83A53806702836ABF948`  
`update-binary` matches GitHub tag `v3.1` (`ianmacd/multidisabler-samsung`).  
Also staged: `multidisabler-samsung-v3.1-github-src.zip` (source archive).

## Sequence (operator; agent does not Odin)

1. Backup. Charge >60%. Knox already `warranty_bit=1` on this unit — still accept bank/Pay loss.
2. Confirm Prenormal cleared + unlock props (above).
3. Download: **Vol− + Bixby + Power**. Odin Admin → blue COM → **Auto Reboot OFF**, Re-Partition OFF → **AP** = TWRP tar → Start → PASS.
4. Immediately recovery: **Vol+ + Bixby + Power** (do not boot system).
5. TWRP: if encrypted → **Format Data** (`yes`).
6. `adb push` (or MTP) both zips to `/sdcard`:
   - `multidisabler-samsung-3.1-flashable.zip`
   - `Magisk-v31.1.zip`
7. TWRP **Install** → **multidisabler** → reboot recovery if script asks / or stay in TWRP.
8. TWRP **Install** → **Magisk-v31.1.zip** → Reboot System.
9. Verify: `adb shell su -c id` → `uid=0`. Then `/dev/hidg0` path (`docs/S8_USB_HID_GADGET_TR.md`).

**Order rule:** multidisabler **before** Magisk. Magisk-first after Format Data on Samsung Pie often loses Vaultkeeper / FBE / stock-recovery-restore protections and soft-bricks the next boot.

**vbmeta:** Do not flash random A10+ vbmeta disable images on A9 dreamlte.

## Explicit non-actions this turn

- No Odin Start for TWRP/Magisk.
- No `adb reboot bootloader` unlock games.
- No Magisk APK sideload as a root substitute (needs patched boot / TWRP zip).
