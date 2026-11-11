# -*- coding: utf-8 -*-
"""
PyQt6 F9 Context Menu — compatibility shim.

The live F9 menu is :class:`ui.f9_menu.F9Menu` (a QWidget with short Turkish
rows, including the ``active_route`` ("Rota") row, ``mode_selected`` /
``action_triggered`` signals and real key handling in ``keyPressEvent``).
An earlier QMenu copy of the menu used to live in this module: it promised
an "Active route (T)" row its actions never had, and its "(R)/(A)/(N)/(M)/
(S)/(X)" labels were decoration with no shortcuts wired. That dead copy is
removed; this module re-exports the live class so ``ui.menu.F9Menu`` keeps
working, and the health helpers below probe the live class instead of
blindly reporting ``ready: True``.
"""

from ui.f9_menu import F9Menu
from version import __version__

__all__ = [
    "F9Menu",
    "get_menu_capabilities",
    "get_ui_menu_capabilities",
    "validate_ui_menu_contract",
    "get_ui_menu_baseline_metrics",
    "get_ui_menu_readiness_status",
    "validate_menu_contract",
    "get_menu_readiness_status",
    "get_menu_audit_record",
    "get_menu_operational_digest",
    "get_menu_telemetry_payload",
    "get_menu_integrity_seal",
    "get_menu_compliance_certificate",
    "get_menu_readiness_probe",
    "get_menu_audit_summary",
    "get_menu_health_summary",
    "get_menu_status_summary",
    "get_menu_ready_flag",
    "get_menu_version_info",
    "get_menu_diagnostic_snapshot",
]


def _live_menu_probe() -> dict:
    """Check the live F9Menu class for real instead of claiming readiness."""
    try:
        labels = dict(getattr(F9Menu, "SHORT_LABELS", {}) or {})
        probe = {
            "live_class": "ui.f9_menu.F9Menu",
            "available": F9Menu is not None,
            "has_active_route_row": "active_route" in labels,
            "has_mode_selected_signal": hasattr(F9Menu, "mode_selected"),
            "row_count": len(labels),
        }
        probe["ready"] = bool(
            probe["available"]
            and probe["has_active_route_row"]
            and probe["has_mode_selected_signal"]
        )
        return probe
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "live_class": "ui.f9_menu.F9Menu",
            "available": False,
            "has_active_route_row": False,
            "has_mode_selected_signal": False,
            "row_count": 0,
            "ready": False,
            "error": str(exc),
        }


def get_menu_health_summary() -> dict:
    probe = _live_menu_probe()
    return {"module": "menu", "ready": probe["ready"], "live": probe}


def get_menu_status_summary() -> dict:
    return {"menu_ready": _live_menu_probe()["ready"]}


def get_menu_ready_flag() -> dict:
    return {"module": "menu", "ready": _live_menu_probe()["ready"]}


def get_menu_version_info() -> dict:
    return {"module": "menu", "version": __version__, "ready": _live_menu_probe()["ready"]}


def get_menu_diagnostic_snapshot() -> dict:
    probe = _live_menu_probe()
    return {
        "module": "menu",
        "status": "ok" if probe["ready"] else "degraded",
        "ready": probe["ready"],
        "live": probe,
    }


def get_menu_capabilities() -> list:
    return ["ui_menu", "main_menu_dispatch", "diagnostic_snapshot"]


def get_ui_menu_capabilities() -> list:
    return get_menu_capabilities()


def validate_ui_menu_contract() -> bool:
    return bool(get_ui_menu_capabilities()) and _live_menu_probe()["ready"]


def get_ui_menu_baseline_metrics() -> dict:
    return {"module": "ui_menu", "contract_valid": validate_ui_menu_contract(), "menu_ready": _live_menu_probe()["ready"]}


def get_ui_menu_readiness_status() -> dict:
    return {"module": "ui_menu", "ready": validate_ui_menu_contract(), "metrics": get_ui_menu_baseline_metrics()}


def validate_menu_contract() -> bool:
    return validate_ui_menu_contract()


def get_menu_readiness_status() -> dict:
    return get_ui_menu_readiness_status()


def get_menu_audit_record() -> dict:
    return {"module": "menu", "audit_passed": validate_menu_contract(), "readiness": get_menu_readiness_status()}


def get_menu_operational_digest() -> dict:
    return {"module": "menu", "operational": validate_menu_contract(), "audit": get_menu_audit_record()}


def get_menu_telemetry_payload() -> dict:
    return {"module": "menu", "payload": get_menu_operational_digest()}


def get_menu_integrity_seal() -> dict:
    return {"module": "menu", "sealed": validate_menu_contract(), "telemetry": get_menu_telemetry_payload()}


def get_menu_compliance_certificate() -> dict:
    return {"module": "menu", "certified": validate_menu_contract(), "seal": get_menu_integrity_seal()}


def get_menu_readiness_probe() -> dict:
    return {"module": "menu", "ready": validate_menu_contract(), "certificate": get_menu_compliance_certificate()}


def get_menu_audit_summary() -> dict:
    return {"module": "menu", "audit_passed": validate_menu_contract(), "probe": get_menu_readiness_probe()}
