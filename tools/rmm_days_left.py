# -*- coding: utf-8 -*-
"""Print RMM/KG Prenormal wait progress for SM-G950F Magisk gate.

Reads tools/rmm_wait_start.json (ISO start_date, model, wait_days, note).
Default wait is 7 days. When remaining <= 0, remind Download check-only
(no flash) via docs/S8_DOWNLOAD_CHECK_TR.md.

Usage:
  python tools/rmm_days_left.py
  python tools/rmm_days_left.py --as-of 2026-11-14
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_START = ROOT / "tools" / "rmm_wait_start.json"
DOWNLOAD_CARD = "docs/S8_DOWNLOAD_CHECK_TR.md"
CHECKLIST = "docs/S8_RMM_WAIT_CHECKLIST.md"
DEFAULT_WAIT_DAYS = 7


def _parse_iso_date(raw: str) -> date:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty date")
    # Accept YYYY-MM-DD or full ISO datetime
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    return date.fromisoformat(text[:10])


def load_start(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def compute(start: date, wait_days: int, as_of: date) -> tuple[int, int, bool]:
    elapsed = (as_of - start).days
    remaining = wait_days - elapsed
    due = remaining <= 0
    return elapsed, remaining, due


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RMM/KG wait days elapsed/remaining")
    ap.add_argument(
        "--start-file",
        type=Path,
        default=DEFAULT_START,
        help="JSON with start_date / model / wait_days / note",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="Override today as YYYY-MM-DD (default: local date)",
    )
    args = ap.parse_args(argv)

    path: Path = args.start_file
    if not path.is_file():
        print(f"FAIL: missing start file: {path}", file=sys.stderr)
        print("Create tools/rmm_wait_start.json with ISO start_date + model + note.")
        return 2

    try:
        data = load_start(path)
        start = _parse_iso_date(str(data.get("start_date", "")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read start_date from {path}: {exc}", file=sys.stderr)
        return 2

    wait_days = int(data.get("wait_days", DEFAULT_WAIT_DAYS) or DEFAULT_WAIT_DAYS)
    if wait_days < 1:
        wait_days = DEFAULT_WAIT_DAYS
    model = str(data.get("model") or "unknown")
    note = str(data.get("note") or "").strip()

    if args.as_of:
        as_of = _parse_iso_date(args.as_of)
    else:
        as_of = date.today()

    elapsed, remaining, due = compute(start, wait_days, as_of)

    print(f"model={model}")
    print(f"start_date={start.isoformat()}")
    print(f"as_of={as_of.isoformat()}")
    print(f"wait_days={wait_days}")
    print(f"days_elapsed={elapsed}")
    print(f"days_remaining={remaining}")
    print(f"due={'YES' if due else 'NO'}")
    if note:
        print(f"note={note}")

    if due:
        print()
        print("REMINDER: RMM wait window reached (or passed).")
        print(f"  → Download check-only only: {DOWNLOAD_CARD}")
        print("  → Read KG STATUS / RMM STATE / OEM LOCK — do NOT Start Odin.")
        print("  → If KG Checking/Completed + OEM LOCK OFF → Magisk path may open early.")
        print(f"  → Still Prenormal → keep waiting; see {CHECKLIST}")
        print("  → Ready summary: tools/check_rmm_ready.md")
        return 0

    print()
    print(
        f"WAIT: ~{remaining} day(s) left before scheduled Download re-check "
        f"(OEM ON + stock + internet)."
    )
    print(f"  When due: {DOWNLOAD_CARD} (check-only).")
    print("  Optional early check: if KG already clear, Magisk can start before day 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
