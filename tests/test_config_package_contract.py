# -*- coding: utf-8 -*-
"""Tests for the config package contract probe (S-2601, R67).

Regression test: get_config_package_summary(),
get_config_package_ready_flag() and validate_config_package_contract()
were tautologies — the summary try block could never raise, the ready
flag was hardcoded True, and validate() was bool() of a constant
3-element capabilities list, so all three always reported healthy.
They now delegate to the real schema health probe
(config.schema _probe_schema_health via validate_config_schema_contract),
so a failing probe must surface as False.
"""

import config
from config import (
    get_config_package_readiness_status,
    get_config_package_ready_flag,
    get_config_package_summary,
    validate_config_package_contract,
)


def test_contract_true_on_healthy_defaults():
    assert validate_config_package_contract() is True
    assert get_config_package_summary() == {"config_ready": True}
    assert get_config_package_ready_flag() == {"package": "config", "ready": True}


def test_contract_false_when_probe_fails(monkeypatch):
    monkeypatch.setattr(config, "validate_config_schema_contract", lambda: False)
    assert validate_config_package_contract() is False
    assert get_config_package_summary() == {"config_ready": False}
    assert get_config_package_ready_flag() == {"package": "config", "ready": False}
    assert get_config_package_readiness_status()["ready"] is False


def test_contract_false_when_probe_raises(monkeypatch):
    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(config, "validate_config_schema_contract", _boom)
    assert validate_config_package_contract() is False
    assert get_config_package_summary() == {"config_ready": False}
    assert get_config_package_ready_flag() == {"package": "config", "ready": False}
