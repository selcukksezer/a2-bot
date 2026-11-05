from __future__ import annotations

from dataset_factory.quality_dedup_gate import DatasetQualityGate, QualityGateDecision


class QualityGate(DatasetQualityGate):
    """Quality Gate wrapper around DatasetQualityGate."""

    def get_quality_gate_ready_flag(self) -> dict:
        return {"module": "quality_gate", "ready": True}


__all__ = [
    "QualityGate",
    "DatasetQualityGate",
    "QualityGateDecision",
]
