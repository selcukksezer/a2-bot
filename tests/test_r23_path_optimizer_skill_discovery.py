# -*- coding: utf-8 -*-
"""R23 regression tests.

S-0203: validate_path_optimizer_contract() must run the real
RouteOptimizer._self_check() probe instead of always returning True from a
static non-empty capability list.
S-1303: confirm_tooltip_hint() must classify "increases physical attack" as
a buff (scroll_buff), not as an attack, while a real attack skill
("deals physical attack damage") still classifies as attack.
F-1615: R/T mapping (right-click -> T, left-click -> R) is recorded as a
code comment at the top of core/skill_discovery.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.path_optimizer import (
    RouteOptimizer,
    get_path_optimizer_audit_record,
    get_path_optimizer_compliance_certificate,
    get_path_optimizer_integrity_seal,
    get_path_optimizer_operational_digest,
    get_path_optimizer_readiness_status,
    validate_path_optimizer_contract,
)
from core.skill_discovery import SkillDiscovery


def test_validate_contract_runs_real_self_check(monkeypatch):
    # Healthy optimizer -> True.
    assert validate_path_optimizer_contract() is True
    # A broken self-check must make the whole chain fail (no fake success).
    monkeypatch.setattr(
        RouteOptimizer, "_self_check", lambda self: (False, "boom")
    )
    assert validate_path_optimizer_contract() is False
    assert get_path_optimizer_readiness_status()["ready"] is False
    assert get_path_optimizer_audit_record()["audit_passed"] is False
    assert get_path_optimizer_operational_digest()["operational"] is False
    assert get_path_optimizer_integrity_seal()["sealed"] is False
    assert get_path_optimizer_compliance_certificate()["certified"] is False


def test_validate_contract_false_when_no_capabilities(monkeypatch):
    import core.path_optimizer as po

    monkeypatch.setattr(po, "get_path_optimizer_capabilities", lambda: [])
    assert validate_path_optimizer_contract() is False


def _discovery(tmp_path):
    return SkillDiscovery(
        lexicon_path=str(tmp_path / "lexicon.json"),
        db_path=str(tmp_path / "lexicon.sqlite"),
    )


def test_tooltip_increases_physical_attack_is_buff(tmp_path):
    sd = _discovery(tmp_path)
    assert sd.confirm_tooltip_hint("1", "Increases physical attack by 15") == "scroll_buff"
    assert sd.confirm_tooltip_hint("2", "Attack speed scroll") == "scroll_buff"
    assert sd.confirm_tooltip_hint("3", "Physical attack buff scroll") == "scroll_buff"


def test_tooltip_physical_attack_skill_is_attack(tmp_path):
    sd = _discovery(tmp_path)
    assert sd.confirm_tooltip_hint("1", "Deals physical attack damage") == "attack"
    assert sd.confirm_tooltip_hint("2", "Strike: 120 damage") == "attack"


def test_rt_mapping_recorded_in_skill_discovery_source():
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "core", "skill_discovery.py"
    )
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    assert "right-click attack = T" in src
    assert "left-click attack = R" in src
