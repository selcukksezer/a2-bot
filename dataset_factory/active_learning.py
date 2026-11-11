from __future__ import annotations

from typing import Optional

from dataset_factory.active_learning_loop import (
    CANONICAL_12_YOLO_CLASSES,
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    ActiveLearningLoop,
    HardExampleRecord,
    ShadowGameplayRecorder,
    normalize_material_class_key,
    resolve_class_id,
)
from version import __version__


class ActiveLearningEngine(ActiveLearningLoop):
    """Active Learning Engine wrapper around ActiveLearningLoop."""

    def get_active_learning_ready_flag(self) -> dict:
        return {"module": "active_learning", "ready": validate_active_learning_contract()}

    def get_active_learning_version_info(self) -> dict:
        return {
            "module": "active_learning",
            "version": __version__,
            "ready": validate_active_learning_contract(),
        }

    def get_active_learning_diagnostic_snapshot(self) -> dict:
        valid = validate_active_learning_contract()
        return {
            "module": "active_learning",
            "status": "ok" if valid else "contract_invalid",
            "ready": valid,
        }


class ActiveLearning(ActiveLearningEngine):
    """Alias wrapper for ActiveLearningEngine (inherits the engine's methods)."""


__all__ = [
    "ActiveLearningEngine",
    "ActiveLearning",
    "ActiveLearningLoop",
    "HardExampleRecord",
    "ShadowGameplayRecorder",
    "CANONICAL_12_YOLO_CLASSES",
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "normalize_material_class_key",
    "resolve_class_id",
    "get_active_learning_readiness_status",
    "get_active_learning_audit_record",
    "get_active_learning_operational_digest",
    "get_active_learning_telemetry_payload",
    "get_active_learning_integrity_seal",
    "get_active_learning_compliance_certificate",
    "get_active_learning_readiness_probe",
    "get_active_learning_audit_summary",
]


def get_active_learning_capabilities() -> list:
    return ["active_learning", "uncertainty_sampling", "diagnostic_snapshot"]


def validate_active_learning_contract() -> bool:
    """Real contract check for the active-learning module.

    Previously this only checked that a hard-coded capability list was
    non-empty (a tautology), and the readiness/audit/seal/certificate chain
    built on it certified nothing. Now it verifies the actual invariants:
    the loop/recorder classes exist and the 12-class list is exactly the
    canonical taxonomy ids 0..11, in canonical order.
    """
    try:
        if not get_active_learning_capabilities():
            return False
        if not isinstance(ActiveLearningLoop, type):
            return False
        if not isinstance(ShadowGameplayRecorder, type):
            return False
        from dataset_factory.class_taxonomy import CANONICAL_NAMES

        expected = tuple(CANONICAL_NAMES[cid] for cid in range(12))
        if tuple(CANONICAL_12_YOLO_CLASSES) != expected:
            return False
        return True
    except Exception:
        return False


def _contract_valid_once(contract_valid: Optional[bool] = None) -> bool:
    """Return the (single) contract verdict for an audit-chain call.

    The readiness/audit/seal/probe chain used to re-run
    ``validate_active_learning_contract()`` at every level (8 calls for one
    ``get_active_learning_audit_summary()``). The verdict is now computed
    once at the outermost call and threaded down via ``_contract_valid``;
    outputs are unchanged.
    """
    return validate_active_learning_contract() if contract_valid is None else bool(contract_valid)


def get_active_learning_baseline_metrics(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_baseline_metrics(_contract_valid_once(_contract_valid))


def get_active_learning_readiness_status(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_readiness_status(_contract_valid_once(_contract_valid))


def get_active_learning_audit_record(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_audit_record(_contract_valid_once(_contract_valid))


def get_active_learning_operational_digest(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_operational_digest(_contract_valid_once(_contract_valid))


def get_active_learning_telemetry_payload(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_telemetry_payload(_contract_valid_once(_contract_valid))


def get_active_learning_integrity_seal(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_integrity_seal(_contract_valid_once(_contract_valid))


def get_active_learning_compliance_certificate(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_compliance_certificate(_contract_valid_once(_contract_valid))


def get_active_learning_readiness_probe(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_readiness_probe(_contract_valid_once(_contract_valid))


def get_active_learning_audit_summary(_contract_valid: Optional[bool] = None) -> dict:
    return _active_learning_audit_summary(_contract_valid_once(_contract_valid))


# Private builders: each public function above evaluates the contract
# exactly once and threads the result down, instead of every nesting level
# re-running validate_active_learning_contract() (measured: a single
# audit_summary() call used to evaluate it 8 times, 7 redundant, each
# re-importing class_taxonomy and rebuilding the 12-class tuple).
# Return values are unchanged.
def _active_learning_baseline_metrics(contract_valid: bool) -> dict:
    return {"module": "active_learning", "contract_valid": contract_valid, "capability_count": len(get_active_learning_capabilities())}


def _active_learning_readiness_status(contract_valid: bool) -> dict:
    return {"module": "active_learning", "ready": contract_valid, "metrics": _active_learning_baseline_metrics(contract_valid)}


def _active_learning_audit_record(contract_valid: bool) -> dict:
    return {"module": "active_learning", "audit_passed": contract_valid, "readiness": _active_learning_readiness_status(contract_valid)}


def _active_learning_operational_digest(contract_valid: bool) -> dict:
    return {"module": "active_learning", "operational": contract_valid, "audit": _active_learning_audit_record(contract_valid)}


def _active_learning_telemetry_payload(contract_valid: bool) -> dict:
    return {"module": "active_learning", "payload": _active_learning_operational_digest(contract_valid)}


def _active_learning_integrity_seal(contract_valid: bool) -> dict:
    return {"module": "active_learning", "sealed": contract_valid, "telemetry": _active_learning_telemetry_payload(contract_valid)}


def _active_learning_compliance_certificate(contract_valid: bool) -> dict:
    return {"module": "active_learning", "certified": contract_valid, "seal": _active_learning_integrity_seal(contract_valid)}


def _active_learning_readiness_probe(contract_valid: bool) -> dict:
    return {"module": "active_learning", "ready": contract_valid, "certificate": _active_learning_compliance_certificate(contract_valid)}


def _active_learning_audit_summary(contract_valid: bool) -> dict:
    return {"module": "active_learning", "audit_passed": contract_valid, "probe": _active_learning_readiness_probe(contract_valid)}



