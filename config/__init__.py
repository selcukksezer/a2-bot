# -*- coding: utf-8 -*-
"""Config package: re-exports the settings types owned by config.schema."""

from config.schema import BotSettings, SettingsManager, validate_config_schema_contract
from version import __version__

__all__ = [
    "BotSettings",
    "SettingsManager",
    "get_config_package_summary",
    "get_config_package_ready_flag",
    "get_config_package_version_info",
    "get_config_package_diagnostic_snapshot",
    "get_config_package_capabilities",
    "validate_config_package_contract",
    "get_config_package_baseline_metrics",
    "get_config_package_readiness_status",
    "get_config_package_audit_record",
]


def get_config_package_summary() -> dict:
    # S-2601: was a tautology (try block could never raise). Now reports the
    # real schema health probe; the except path is reachable when the probe
    # itself raises.
    try:
        return {"config_ready": validate_config_package_contract()}
    except Exception:
        return {"config_ready": False}


def get_config_package_ready_flag() -> dict:
    # S-2601: was hardcoded {"ready": True}; now delegates to the real probe.
    return {"package": "config", "ready": validate_config_package_contract()}


def get_config_package_version_info() -> dict:
    # S-6501: was hardcoded {"ready": True}; now reflects the real probe
    # (validate_config_package_contract -> config.schema health probe),
    # mirroring config.schema.get_schema_version_info (S-0403).
    return {"package": "config", "version": __version__, "ready": validate_config_package_contract()}


def get_config_package_diagnostic_snapshot() -> dict:
    # S-6501: was hardcoded {"status": "ok", "ready": True}; status/ready
    # now reflect the real probe, mirroring get_schema_diagnostic_snapshot.
    ready = validate_config_package_contract()
    return {"package": "config", "status": "ok" if ready else "error", "ready": ready}


def get_config_package_capabilities() -> list:
    return ["config_package", "diagnostic_snapshot", "version_info"]


def validate_config_package_contract() -> bool:
    # S-2601: was bool(get_config_package_capabilities()) — the capabilities
    # list is a constant 3-element list, so this was always True. Now
    # delegates to the real schema health probe (config.schema
    # _probe_schema_health via validate_config_schema_contract): default
    # settings must validate, canonical hotkey chords must parse, and a
    # settings payload must survive a merge round-trip. Any probe failure
    # (or exception) makes the contract False.
    try:
        return bool(validate_config_schema_contract())
    except Exception:
        return False


def get_config_package_baseline_metrics() -> dict:
    return {"module": "config_package", "contract_valid": validate_config_package_contract(), "capability_count": len(get_config_package_capabilities())}


def get_config_package_readiness_status() -> dict:
    return {"module": "config_package", "ready": validate_config_package_contract(), "metrics": get_config_package_baseline_metrics()}


def get_config_package_audit_record() -> dict:
    return {"module": "config_package", "audit_passed": validate_config_package_contract(), "readiness": get_config_package_readiness_status()}


validate_config_init_contract = validate_config_package_contract
get_config_init_audit_record = get_config_package_audit_record


def get_config_init_operational_digest() -> dict:
    return {"module": "config_init", "operational": validate_config_init_contract(), "audit": get_config_init_audit_record()}


def get_config_init_telemetry_payload() -> dict:
    return {"module": "config_init", "telemetry_ready": validate_config_init_contract(), "digest": get_config_init_operational_digest()}


get_config_package_telemetry_payload = get_config_init_telemetry_payload


def get_config_package_integrity_seal() -> dict:
    return {"module": "config_package", "sealed": validate_config_package_contract(), "telemetry": get_config_package_telemetry_payload()}


def get_config_package_compliance_certificate() -> dict:
    return {"module": "config_package", "certified": validate_config_package_contract(), "seal": get_config_package_integrity_seal()}


validate_config_pkg_contract = validate_config_package_contract
get_config_pkg_compliance_certificate = get_config_package_compliance_certificate


def get_config_pkg_readiness_probe() -> dict:
    return {"module": "config_pkg", "ready": validate_config_pkg_contract(), "certificate": get_config_pkg_compliance_certificate()}


def get_config_pkg_audit_summary() -> dict:
    return {"module": "config_pkg", "audit_passed": validate_config_pkg_contract(), "probe": get_config_pkg_readiness_probe()}


get_config_init_audit_summary = get_config_pkg_audit_summary


def get_config_init_heartbeat_snapshot() -> dict:
    return {"module": "config_init", "alive": validate_config_init_contract(), "audit": get_config_init_audit_summary()}


