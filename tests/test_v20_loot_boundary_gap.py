# -*- coding: utf-8 -*-
"""v20 loot start/end — honest gap lock (no fabricated detector).

Gather (node F) already has: gather prompt + top cast bar rise/complete
in core/state_machine.py. Combat loot F still has no proven loot-specific
UI signal in-repo; CombatRecovery uses gather cast bar as an UNVERIFIED
proxy and keeps blind second-F blocked until PC proof exists.

This test locks that honesty: if someone claims loot boundaries are
'solved' without new evidence (template/frame/inventory delta), these
assertions fail when the disclaimer or gate is quietly removed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_combat_recovery_documents_unproven_loot_cast_proxy() -> None:
    src = (ROOT / "core" / "combat_recovery.py").read_text(encoding="utf-8")
    assert "Loot'un gerçekten AYNI barda kanal gösterip göstermediği" in src
    assert "vekil BLOKE" in src or "VEKİL" in src
    assert "LOOT_CAST_BAR_COMPLETE_PCT" in src
    # Blind success on timeout must stay forbidden (F-0050 twin).
    assert "kör zaman aşımı" in src.lower() or "ASLA başarı" in src


def test_state_machine_has_gather_cast_but_open_full_bar_question() -> None:
    src = (ROOT / "core" / "state_machine.py").read_text(encoding="utf-8")
    assert "_check_gather_cast_progress" in src
    assert "_stage_cast_rose" in src
    # Known open question: cast bar that starts already full never 'rises'.
    assert "dolu başlayıp" in src or "açık soru" in src


def test_no_dedicated_loot_template_or_inventory_delta_detector() -> None:
    """Evidence gap: no loot-only template / inventory OCR delta module."""
    # Combat recovery is the only loot-cast poller; it reuses gather APIs.
    cr = (ROOT / "core" / "combat_recovery.py").read_text(encoding="utf-8")
    assert "verify_gather_cast_bar" in cr
    # No dedicated loot template asset name in combat_recovery.
    assert "loot_template" not in cr.lower()
    assert "inventory_delta" not in cr.lower()
