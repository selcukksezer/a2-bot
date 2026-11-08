# RMM / KG ready? (check-only)

**Not a flash guide.** Decide whether Magisk/TWRP is allowed yet.

## Primary path (preferred)

1. Keep **OEM unlocking ON**, stay on **stock**, phone **on** with **Wi‑Fi or mobile data** daily.
2. Wait about **7 full days** from the first clean stock boot after the failed custom flash (restart the clock if you wiped/restored again).
3. Track the clock: `tools/rmm_wait_start.json` + `python tools/rmm_days_left.py` (prints elapsed/remaining; reminds Download check-only when due).
4. Re-enter **Download** (check-only): `docs/S8_DOWNLOAD_CHECK_TR.md`
5. Gate open only when Download shows **KG Checking/Completed** (not Prenormal) **and** **OEM LOCK: OFF**.
6. Full wait checklist: `docs/S8_RMM_WAIT_CHECKLIST.md`

**Optional early check:** when convenient, run Download check-only before day 7. If KG is already clear + OEM LOCK OFF, Magisk can start early.

## Do not treat as primary

- **Manual date/time tricks** (set clock forward/back to “skip” Prenormal) are **risky and unreliable** — do not use them as the main plan. Prefer the 7-day OEM-ON + internet wait.
- Do **not** flash custom while Prenormal / “only official binaries” is active.

## After the gate opens

Operator confirm → `docs/S8_MAGISK_SEQUENCE_SAFE.md` → then `tools/after_magisk_hid.bat`.
