# SM-G950F — RMM / KG Prenormal wait (no custom flash)

**Status (2026-11-07 evening):** Magisk path failed on RMM again → user recovering with **stock Odin** (`G950FXXUCDZE7`). Wait clock **reset** in `tools/rmm_wait_start.json` to **2026-11-07**. Treat **RMM / KG Prenormal** as active (~7d) until Download-mode proves otherwise. **No Magisk/TWRP tonight.**

**Do not flash TWRP / Magisk / multidisabler while Prenormal is active.** Waiting is the safe path; forcing custom AP can soft-brick or loop.

## Why adb alone is not enough

Android `getprop` on this build usually **does not** expose `rmm` / `kg` / `prenormal` strings. Gate status is read on the **Download** screen.

**TR operator card (check-only taps):** `docs/S8_DOWNLOAD_CHECK_TR.md`

1. Power off.
2. **Vol− + Bixby + Power** → Download / Warning → **Vol+** to continue.
3. Read the text lines (wording varies by bootloader):
   - **RMM STATE: Prenormal** or **KG STATUS: Prenormal** → **WAIT** (this checklist).
   - **KG Checking / Completed** + **OEM LOCK: OFF** → Magisk path unlocks (still need user confirm + `S8_MAGISK_SEQUENCE_SAFE.md`).
   - **Checking** / **Normal** / line absent + OEM unlock already off → custom flash may be allowed (still need user confirm before any Odin).
4. Leave Download with **Vol− + Power** (or long-press Power) — **do not Start Odin** on this check-only visit.

`adb devices` empty in Download is normal (Odin uses `VID_04E8&PID_685D` modem, not adb).

## 7-day wait checklist (keep OEM flashable)

| # | Do | Why |
|---|----|-----|
| 1 | Keep **OEM unlocking** (OEM kilit açma) **ON** in Developer options | Turning it off restarts / worsens the timer |
| 2 | Stay on **stock** `G950FXXUCDZE7` — **no** TWRP, Magisk, custom kernels, or unofficial Odin packages | Any unofficial binary can reset Prenormal |
| 3 | Connect to **Wi‑Fi or mobile data** daily; leave the phone **on** (screen off OK) | RMM/KG needs network time |
| 4 | Prefer **SIM inserted** + normal clock/date (auto time) | Helps Checking → Normal transition |
| 5 | Avoid factory reset / wipe unless recovering a brick | Reset often restarts the ~7 day window |
| 6 | Do **not** chase “unlock again” in Download while Prenormal | Unlock/OEM already done (`flash.locked=0`) |
| 7 | After **~7 full days** of continuous stock uptime + OEM ON: re-enter Download and re-read RMM/KG | Only then consider the Magisk sequence |

Mark calendar: wait started ≈ first clean stock boot after the failed custom flash (this session phone uptime was short after restore — **restart the 7-day clock from that stock boot**, not from adb connect time).

**Not primary:** do **not** rely on manual date/time hacks to skip Prenormal. Prefer OEM ON + network + ~7 days, then re-check Download (`tools/check_rmm_ready.md`).

## When wait is over — still no auto-flash

Gate only when **all** are true:

1. Download: RMM/KG **not** Prenormal.
2. Android: still `flash.locked=0` / `orange`, `sys.oem_unlock_allowed=1`.
3. Operator **explicitly confirms** flash.
4. Follow `docs/S8_MAGISK_SEQUENCE_SAFE.md` — **multidisabler first**, then Magisk. Never Magisk-only after Format Data on Pie Samsung without multidisabler.

Staged zip (Ian Macdonald v3.1, verified):  
`vendor/android_hid/s8_root/multidisabler-samsung-3.1-flashable.zip`

## Interim HID (goal #3 proven without phone root)

- Interception: `python tools\interception_status.py --smoke --confirm-smoke --key tab`
- Rootless 1B: `tools/rootless_hid_bridge.py` + docs in `ANDROID_USB_NO_BT_TR.md`

1A remains **open** until `su` + `/dev/hidg0` + `NETWORK_HID` evidence.
