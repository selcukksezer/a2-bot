from __future__ import annotations

from dataset_factory.quality_dedup_gate import DatasetQualityGate, QualityGateDecision
from version import __version__


class QualityGate(DatasetQualityGate):
    """Quality Gate wrapper around DatasetQualityGate."""

    def get_quality_gate_ready_flag(self) -> dict:
        # S-1902 (Karar 3): ready is the live contract probe, not a constant.
        return {"module": "quality_gate", "ready": validate_quality_gate_contract()}

    def get_quality_gate_version_info(self) -> dict:
        return {"module": "quality_gate", "version": __version__, "ready": validate_quality_gate_contract()}

    def get_quality_gate_diagnostic_snapshot(self) -> dict:
        ready = validate_quality_gate_contract()
        return {"module": "quality_gate", "status": "ok" if ready else "degraded", "ready": ready}


__all__ = [
    "QualityGate",
    "DatasetQualityGate",
    "QualityGateDecision",
    "get_quality_gate_readiness_status",
    "get_quality_gate_audit_record",
    "get_quality_gate_operational_digest",
    "get_quality_gate_telemetry_payload",
    "get_quality_gate_integrity_seal",
    "get_quality_gate_compliance_certificate",
    "get_quality_gate_readiness_probe",
    "get_quality_gate_audit_summary",
]


def get_quality_gate_capabilities() -> list:
    return ["quality_gate", "frame_validation", "diagnostic_snapshot"]


def _probe_quality_gate() -> bool:
    """Real readiness probe (S-1902, Karar 3): exercise DatasetQualityGate
    instead of trusting a constant. The gate must construct with sane
    thresholds, keep one valid annotation, and drop one degenerate box;
    any failure means the gate is not actually ready."""
    try:
        gate = DatasetQualityGate()
        if not (
            gate.min_sharpness_threshold > 0
            and 0.0 < gate.val_ratio < 1.0
            and 0.0 < gate.nms_iou_threshold < 1.0
            and gate.min_hamming_distance > 0
        ):
            return False
        kept = gate.validate_yolo_annotations(
            [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
        )
        if len(kept) != 1:
            return False
        dropped = gate.validate_yolo_annotations(
            [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1}]
        )
        if dropped:
            return False
        return True
    except Exception:
        return False


def validate_quality_gate_contract() -> bool:
    return bool(get_quality_gate_capabilities()) and _probe_quality_gate()


def get_quality_gate_baseline_metrics() -> dict:
    return {"module": "quality_gate", "contract_valid": validate_quality_gate_contract(), "capability_count": len(get_quality_gate_capabilities())}


def get_quality_gate_readiness_status() -> dict:
    return {"module": "quality_gate", "ready": validate_quality_gate_contract(), "metrics": get_quality_gate_baseline_metrics()}


def get_quality_gate_audit_record() -> dict:
    return {"module": "quality_gate", "audit_passed": validate_quality_gate_contract(), "readiness": get_quality_gate_readiness_status()}


def get_quality_gate_operational_digest() -> dict:
    return {"module": "quality_gate", "operational": validate_quality_gate_contract(), "audit": get_quality_gate_audit_record()}


def get_quality_gate_telemetry_payload() -> dict:
    return {"module": "quality_gate", "payload": get_quality_gate_operational_digest()}


def get_quality_gate_integrity_seal() -> dict:
    return {"module": "quality_gate", "sealed": validate_quality_gate_contract(), "telemetry": get_quality_gate_telemetry_payload()}


def get_quality_gate_compliance_certificate() -> dict:
    return {"module": "quality_gate", "certified": validate_quality_gate_contract(), "seal": get_quality_gate_integrity_seal()}


def get_quality_gate_readiness_probe() -> dict:
    return {"module": "quality_gate", "ready": validate_quality_gate_contract(), "certificate": get_quality_gate_compliance_certificate()}


def get_quality_gate_audit_summary() -> dict:
    return {"module": "quality_gate", "audit_passed": validate_quality_gate_contract(), "probe": get_quality_gate_readiness_probe()}


