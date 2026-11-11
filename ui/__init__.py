# -*- coding: utf-8 -*-
"""ui package.

Readiness helpers in this module used to return hardcoded ``{"ready": True}``
without checking anything. They now probe for real: PyQt6 must be
importable and the package's core dialog/menu modules must exist on disk.
Importing this package stays light — the probe uses ``find_spec`` and file
existence checks only, it does not import PyQt6 or any dialog module.
The probe result is memoised for 2 s (``_PROBE_CACHE_TTL_SECONDS``) because
the readiness helpers nest into each other and one top-level call used to
run the filesystem probe 8 times; the memo still reflects a real probe.
"""

import importlib.util
import time
from pathlib import Path
from version import __version__

# Core UI modules the package is expected to ship. Kept as a plain list so a
# missing/renamed module makes readiness False instead of going unnoticed.
_EXPECTED_UI_MODULES = (
    "menu",
    "f9_menu",
    "app_shell",
    "overlay",
    "overlay_hud",
    "settings_window",
    "settings_dialog",
    "map_viewer",
    "cartography_radar",
    "yolo_factory_dialog",
)


def _probe_ui_package_uncached() -> dict:
    """Probe the real state of the ui package; never claims blind readiness."""
    pkg_dir = Path(__file__).resolve().parent
    modules_present = {
        name: (pkg_dir / f"{name}.py").is_file() for name in _EXPECTED_UI_MODULES
    }
    pyqt6_available = importlib.util.find_spec("PyQt6.QtWidgets") is not None
    ready = bool(pyqt6_available and all(modules_present.values()))
    return {
        "pyqt6_available": pyqt6_available,
        "modules_present": modules_present,
        "missing_modules": sorted(n for n, ok in modules_present.items() if not ok),
        "ready": ready,
    }


# The readiness helpers below nest into each other (certificate -> seal ->
# telemetry -> digest -> audit -> readiness -> baseline -> validate), so one
# top-level call used to run the filesystem probe 8 times. The probe result
# is memoised for a short window: readiness is not time-critical, and a
# 2 s-old snapshot is still a real probe, not a hardcoded claim.
_PROBE_CACHE_TTL_SECONDS = 2.0
_probe_cache: dict = {}
_probe_cache_ts: float = 0.0


def _probe_ui_package() -> dict:
    """TTL-cached probe: at most one real filesystem probe per window.

    Returns a copy (including a copy of the nested ``modules_present``
    dict) so callers cannot mutate the cached snapshot.
    """
    global _probe_cache, _probe_cache_ts
    now = time.monotonic()
    if _probe_cache and (now - _probe_cache_ts) < _PROBE_CACHE_TTL_SECONDS:
        snapshot = dict(_probe_cache)
        snapshot["modules_present"] = dict(_probe_cache["modules_present"])
        snapshot["missing_modules"] = list(_probe_cache["missing_modules"])
        return snapshot
    probe = _probe_ui_package_uncached()
    _probe_cache = dict(probe)
    _probe_cache["modules_present"] = dict(probe["modules_present"])
    _probe_cache["missing_modules"] = list(probe["missing_modules"])
    _probe_cache_ts = now
    snapshot = dict(probe)
    snapshot["modules_present"] = dict(probe["modules_present"])
    snapshot["missing_modules"] = list(probe["missing_modules"])
    return snapshot


def get_ui_package_summary() -> dict:
    probe = _probe_ui_package()
    return {
        "ui_ready": probe["ready"],
        "pyqt6_available": probe["pyqt6_available"],
        "missing_modules": probe["missing_modules"],
    }


def get_ui_package_ready_flag() -> dict:
    return {"package": "ui", "ready": _probe_ui_package()["ready"]}


def get_ui_package_version_info() -> dict:
    return {"package": "ui", "version": __version__, "ready": _probe_ui_package()["ready"]}


def get_ui_package_diagnostic_snapshot() -> dict:
    probe = _probe_ui_package()
    return {
        "package": "ui",
        "status": "ok" if probe["ready"] else "degraded",
        "ready": probe["ready"],
        "pyqt6_available": probe["pyqt6_available"],
        "missing_modules": probe["missing_modules"],
    }


def get_ui_package_capabilities() -> list:
    return ["ui_package", "diagnostic_snapshot", "version_info"]


def validate_ui_package_contract() -> bool:
    return bool(get_ui_package_capabilities()) and _probe_ui_package()["ready"]


def get_ui_package_baseline_metrics() -> dict:
    probe = _probe_ui_package()
    return {
        "module": "ui_package",
        "contract_valid": validate_ui_package_contract(),
        "capability_count": len(get_ui_package_capabilities()),
        "pyqt6_available": probe["pyqt6_available"],
        "missing_modules": probe["missing_modules"],
    }


def get_ui_package_readiness_status() -> dict:
    return {"module": "ui_package", "ready": validate_ui_package_contract(), "metrics": get_ui_package_baseline_metrics()}


def get_ui_package_audit_record() -> dict:
    return {"module": "ui_package", "audit_passed": validate_ui_package_contract(), "readiness": get_ui_package_readiness_status()}


validate_ui_init_contract = validate_ui_package_contract
get_ui_init_audit_record = get_ui_package_audit_record


def get_ui_init_operational_digest() -> dict:
    return {"module": "ui_init", "operational": validate_ui_init_contract(), "audit": get_ui_init_audit_record()}


def get_ui_init_telemetry_payload() -> dict:
    return {"module": "ui_init", "telemetry_ready": validate_ui_init_contract(), "digest": get_ui_init_operational_digest()}


get_ui_package_telemetry_payload = get_ui_init_telemetry_payload


def get_ui_package_integrity_seal() -> dict:
    return {"module": "ui_package", "sealed": validate_ui_package_contract(), "telemetry": get_ui_package_telemetry_payload()}


def get_ui_package_compliance_certificate() -> dict:
    return {"module": "ui_package", "certified": validate_ui_package_contract(), "seal": get_ui_package_integrity_seal()}


validate_ui_pkg_contract = validate_ui_package_contract
get_ui_pkg_compliance_certificate = get_ui_package_compliance_certificate


def get_ui_pkg_readiness_probe() -> dict:
    return {"module": "ui_pkg", "ready": validate_ui_pkg_contract(), "certificate": get_ui_pkg_compliance_certificate()}


def get_ui_pkg_audit_summary() -> dict:
    return {"module": "ui_pkg", "audit_passed": validate_ui_pkg_contract(), "probe": get_ui_pkg_readiness_probe()}


get_ui_init_audit_summary = get_ui_pkg_audit_summary


def get_ui_init_heartbeat_snapshot() -> dict:
    return {"module": "ui_init", "alive": validate_ui_init_contract(), "audit": get_ui_init_audit_summary()}
