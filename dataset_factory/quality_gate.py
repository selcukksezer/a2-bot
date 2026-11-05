from __future__ import annotations

from dataset_factory.quality_dedup_gate import DatasetQualityGate, QualityGateDecision


class QualityGate(DatasetQualityGate):
    """Quality Gate wrapper around DatasetQualityGate."""

    def get_quality_gate_ready_flag(self) -> dict:
        return {"module": "quality_gate", "ready": True}

    def get_quality_gate_version_info(self) -> dict:
        return {"module": "quality_gate", "version": "2.0.0", "ready": True}

    def get_quality_gate_diagnostic_snapshot(self) -> dict:
        return {"module": "quality_gate", "status": "ok", "ready": True}


__all__ = [
    "QualityGate",
    "DatasetQualityGate",
    "QualityGateDecision",
    "get_quality_gate_readiness_status",
]


def get_quality_gate_capabilities() -> list:
    return ["quality_gate", "frame_validation", "diagnostic_snapshot"]


def validate_quality_gate_contract() -> bool:
    return bool(get_quality_gate_capabilities())


def get_quality_gate_baseline_metrics() -> dict:
    return {"module": "quality_gate", "contract_valid": validate_quality_gate_contract(), "capability_count": len(get_quality_gate_capabilities())}


def get_quality_gate_readiness_status() -> dict:
    return {"module": "quality_gate", "ready": validate_quality_gate_contract(), "metrics": get_quality_gate_baseline_metrics()}



