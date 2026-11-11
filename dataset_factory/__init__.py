"""
Dataset Factory Package.

Provides automated training data collection for YOLO resource detection,
including the 2-second pre-gather time machine ring buffer (`FrameSnapshot`,
`PreGatherTimeMachineBuffer`).
"""

from __future__ import annotations

from .atomic_review_sync import AtomicReviewTxtSynchronizer
from .frame_time_machine import FrameSnapshot, PreGatherTimeMachineBuffer
from .lru_frame_cache import (  # F-2000 / R65 / R46: public cache API export
    DEFAULT_CACHE_CAPACITY,
    FrameCacheEntry,
    LRUFrameDecodeCache,
    get_lru_frame_cache,
    imread_unicode,
)
from .player_self_exclusion_guard import PlayerSelfExclusionGuard
from .stem_material_validator import (
    StemMaterialValidator,
    StemValidationResult,
    extract_stem_material_hints,
    is_stem_material_conflict,
    validate_and_demote_box,
)
from version import __version__

__all__ = [
    "AtomicReviewTxtSynchronizer",
    "DEFAULT_CACHE_CAPACITY",
    "FrameCacheEntry",
    "FrameSnapshot",
    "LRUFrameDecodeCache",
    "PlayerSelfExclusionGuard",
    "PreGatherTimeMachineBuffer",
    # F-2001: StemMaterialValidator üretimde kullanılabilir olmalı —
    # tools/recalibrate_dataset_trust.py Rule 4 bu export üzerinden
    # bağlanır; daha önce export yoktu ve validator yalnızca testler
    # tarafından import ediliyordu.
    "StemMaterialValidator",
    "StemValidationResult",
    "extract_stem_material_hints",
    "get_lru_frame_cache",
    "imread_unicode",
    "is_stem_material_conflict",
    "validate_and_demote_box",
    "get_dataset_factory_package_readiness_status",
    "get_dataset_factory_package_audit_record",
    "get_dataset_factory_init_operational_digest",
    "get_dataset_factory_init_telemetry_payload",
    "get_dataset_factory_package_integrity_seal",
    "get_dataset_factory_package_compliance_certificate",
    "get_dataset_factory_pkg_readiness_probe",
    "get_dataset_factory_pkg_audit_summary",
    "get_dataset_factory_init_heartbeat_snapshot",
]


def _probe_dataset_factory_package() -> dict:
    """Real package probe (S-2402, Karar 3): every core submodule must import
    and the pre-gather ring buffer must construct with sane defaults. A
    component that fails to import (or a buffer that cannot be built) makes
    the package not ready — the answer is never a constant."""
    import importlib

    results: dict = {}
    for name in ("frame_time_machine", "quality_gate", "approved_pool", "atomic_review_sync", "lru_frame_cache"):
        try:
            importlib.import_module(f"dataset_factory.{name}")
            results[name] = True
        except Exception:
            results[name] = False
    try:
        buffer = PreGatherTimeMachineBuffer()
        results["frame_time_machine"] = bool(
            results.get("frame_time_machine")
            and buffer.max_capacity >= 8
            and buffer.max_duration_sec >= 0.5
        )
    except Exception:
        results["frame_time_machine"] = False
    return results


def get_dataset_factory_package_summary() -> dict:
    probe = _probe_dataset_factory_package()
    return {"dataset_factory_ready": all(probe.values()), "components": probe}


def get_dataset_factory_package_ready_flag() -> dict:
    return {"package": "dataset_factory", "ready": validate_dataset_factory_package_contract()}


def get_dataset_factory_package_version_info() -> dict:
    return {"package": "dataset_factory", "version": __version__, "ready": validate_dataset_factory_package_contract()}


def get_dataset_factory_package_diagnostic_snapshot() -> dict:
    ready = validate_dataset_factory_package_contract()
    return {"package": "dataset_factory", "status": "ok" if ready else "degraded", "ready": ready}


def get_dataset_factory_package_capabilities() -> list:
    return ["dataset_factory_package", "diagnostic_snapshot", "version_info"]


def validate_dataset_factory_package_contract() -> bool:
    return bool(get_dataset_factory_package_capabilities()) and all(
        _probe_dataset_factory_package().values()
    )


def get_dataset_factory_package_baseline_metrics() -> dict:
    return _pkg_baseline_metrics(validate_dataset_factory_package_contract())


def get_dataset_factory_package_readiness_status() -> dict:
    return _pkg_readiness_status(validate_dataset_factory_package_contract())


def get_dataset_factory_package_audit_record() -> dict:
    return _pkg_audit_record(validate_dataset_factory_package_contract())


validate_dataset_factory_init_contract = validate_dataset_factory_package_contract
get_dataset_factory_init_audit_record = get_dataset_factory_package_audit_record


def get_dataset_factory_init_operational_digest() -> dict:
    return _init_operational_digest(validate_dataset_factory_package_contract())


def get_dataset_factory_init_telemetry_payload() -> dict:
    return _init_telemetry_payload(validate_dataset_factory_package_contract())


get_dataset_factory_package_telemetry_payload = get_dataset_factory_init_telemetry_payload


def get_dataset_factory_package_integrity_seal() -> dict:
    return _pkg_integrity_seal(validate_dataset_factory_package_contract())


def get_dataset_factory_package_compliance_certificate() -> dict:
    return _pkg_compliance_certificate(validate_dataset_factory_package_contract())


validate_dataset_factory_pkg_contract = validate_dataset_factory_package_contract
get_dataset_factory_pkg_compliance_certificate = get_dataset_factory_package_compliance_certificate


def get_dataset_factory_pkg_readiness_probe() -> dict:
    return _pkg_readiness_probe(validate_dataset_factory_package_contract())


def get_dataset_factory_pkg_audit_summary() -> dict:
    return _pkg_audit_summary(validate_dataset_factory_package_contract())


get_dataset_factory_init_audit_summary = get_dataset_factory_pkg_audit_summary


def get_dataset_factory_init_heartbeat_snapshot() -> dict:
    return _init_heartbeat_snapshot(validate_dataset_factory_package_contract())


# Private builders: each public function above evaluates the contract
# exactly once and threads the result down, instead of every nesting level
# re-running validate_dataset_factory_package_contract() (measured:
# a single heartbeat snapshot used to evaluate it 5 times, 4 redundant).
# Return values are unchanged.
def _pkg_baseline_metrics(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_package",
        "contract_valid": contract_valid,
        "capability_count": len(get_dataset_factory_package_capabilities()),
    }


def _pkg_readiness_status(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_package",
        "ready": contract_valid,
        "metrics": _pkg_baseline_metrics(contract_valid),
    }


def _pkg_audit_record(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_package",
        "audit_passed": contract_valid,
        "readiness": _pkg_readiness_status(contract_valid),
    }


def _init_operational_digest(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_init",
        "operational": contract_valid,
        "audit": _pkg_audit_record(contract_valid),
    }


def _init_telemetry_payload(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_init",
        "telemetry_ready": contract_valid,
        "digest": _init_operational_digest(contract_valid),
    }


def _pkg_integrity_seal(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_package",
        "sealed": contract_valid,
        "telemetry": _init_telemetry_payload(contract_valid),
    }


def _pkg_compliance_certificate(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_package",
        "certified": contract_valid,
        "seal": _pkg_integrity_seal(contract_valid),
    }


def _pkg_readiness_probe(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_pkg",
        "ready": contract_valid,
        "certificate": _pkg_compliance_certificate(contract_valid),
    }


def _pkg_audit_summary(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_pkg",
        "audit_passed": contract_valid,
        "probe": _pkg_readiness_probe(contract_valid),
    }


def _init_heartbeat_snapshot(contract_valid: bool) -> dict:
    return {
        "module": "dataset_factory_init",
        "alive": contract_valid,
        "audit": _pkg_audit_summary(contract_valid),
    }


