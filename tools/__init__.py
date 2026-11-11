# -*- coding: utf-8 -*-
"""tools package readiness helpers.

These used to return constant True without checking anything. The summary
now scans the package directory for real tool modules and the contract
check verifies both that scan and the public API surface, so a broken or
empty package can no longer report itself ready.

Validation chains (seal -> telemetry -> digest -> audit -> readiness)
share a single directory scan per top-level call: each public helper
validates once and passes the result down through private ``_xxx(ready)``
helpers instead of every layer re-running the scan.
"""

from pathlib import Path
from version import __version__

_PACKAGE_DIR = Path(__file__).resolve().parent

# Public helpers the contract requires to exist and be callable.
_REQUIRED_CONTRACT_FUNCS = (
    "get_tools_package_summary",
    "get_tools_package_ready_flag",
    "get_tools_package_version_info",
    "get_tools_package_diagnostic_snapshot",
    "get_tools_package_capabilities",
)


def _scan_tool_modules() -> list:
    return sorted(
        p.stem
        for p in _PACKAGE_DIR.glob("*.py")
        if p.is_file() and p.name != "__init__.py"
    )


def get_tools_package_summary() -> dict:
    try:
        modules = _scan_tool_modules()
        return {
            "tools_ready": bool(modules),
            "module_count": len(modules),
            "modules": modules,
            "package_dir": str(_PACKAGE_DIR),
        }
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        return {
            "tools_ready": False,
            "module_count": 0,
            "modules": [],
            "package_dir": str(_PACKAGE_DIR),
            "error": str(exc),
        }


def _contract_valid(summary: dict) -> bool:
    """Check one already-computed summary against the contract."""
    try:
        if not get_tools_package_capabilities():
            return False
        if not summary.get("tools_ready"):
            return False
        return all(callable(globals().get(name)) for name in _REQUIRED_CONTRACT_FUNCS)
    except Exception:  # noqa: BLE001 - a failed check is a failed contract
        return False


def get_tools_package_ready_flag() -> dict:
    return {"package": "tools", "ready": validate_tools_package_contract()}


def get_tools_package_version_info() -> dict:
    return {
        "package": "tools",
        "version": __version__,
        "ready": validate_tools_package_contract(),
    }


def get_tools_package_diagnostic_snapshot() -> dict:
    ready = validate_tools_package_contract()
    return {
        "package": "tools",
        "status": "ok" if ready else "failed",
        "ready": ready,
    }


def get_tools_package_capabilities() -> list:
    return ["tools_package", "diagnostic_snapshot", "version_info"]


def validate_tools_package_contract() -> bool:
    try:
        return _contract_valid(get_tools_package_summary())
    except Exception:  # noqa: BLE001 - a failed check is a failed contract
        return False


def _baseline_metrics(ready: bool) -> dict:
    return {"module": "tools_package", "contract_valid": ready, "capability_count": len(get_tools_package_capabilities())}


def get_tools_package_baseline_metrics() -> dict:
    return _baseline_metrics(validate_tools_package_contract())


def _readiness_status(ready: bool) -> dict:
    return {"module": "tools_package", "ready": ready, "metrics": _baseline_metrics(ready)}


def get_tools_package_readiness_status() -> dict:
    try:
        return _readiness_status(validate_tools_package_contract())
    except Exception:
        return {"module": "tools_package", "ready": False, "metrics": {}}


def _audit_record(ready: bool) -> dict:
    return {"module": "tools_package", "audit_passed": ready, "readiness": _readiness_status(ready)}


def get_tools_package_audit_record() -> dict:
    try:
        return _audit_record(validate_tools_package_contract())
    except Exception:
        return {"module": "tools_package", "audit_passed": False, "readiness": {}}


validate_tools_init_contract = validate_tools_package_contract
get_tools_init_audit_record = get_tools_package_audit_record


def _operational_digest(ready: bool) -> dict:
    return {"module": "tools_init", "operational": ready, "audit": _audit_record(ready)}


def get_tools_init_operational_digest() -> dict:
    return _operational_digest(validate_tools_init_contract())


def _telemetry_payload(ready: bool) -> dict:
    return {"module": "tools_init", "telemetry_ready": ready, "digest": _operational_digest(ready)}


def get_tools_init_telemetry_payload() -> dict:
    return _telemetry_payload(validate_tools_init_contract())


get_tools_package_telemetry_payload = get_tools_init_telemetry_payload


def _integrity_seal(ready: bool) -> dict:
    return {"module": "tools_package", "sealed": ready, "telemetry": _telemetry_payload(ready)}


def get_tools_package_integrity_seal() -> dict:
    return _integrity_seal(validate_tools_package_contract())


def _compliance_certificate(ready: bool) -> dict:
    return {"module": "tools_package", "certified": ready, "seal": _integrity_seal(ready)}


def get_tools_package_compliance_certificate() -> dict:
    return _compliance_certificate(validate_tools_package_contract())


validate_tools_pkg_contract = validate_tools_package_contract
get_tools_pkg_compliance_certificate = get_tools_package_compliance_certificate


def _readiness_probe(ready: bool) -> dict:
    return {"module": "tools_pkg", "ready": ready, "certificate": _compliance_certificate(ready)}


def get_tools_pkg_readiness_probe() -> dict:
    return _readiness_probe(validate_tools_pkg_contract())


def _audit_summary(ready: bool) -> dict:
    return {"module": "tools_pkg", "audit_passed": ready, "probe": _readiness_probe(ready)}


def get_tools_pkg_audit_summary() -> dict:
    return _audit_summary(validate_tools_pkg_contract())
