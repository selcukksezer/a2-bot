from __future__ import annotations

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


class ActiveLearningEngine(ActiveLearningLoop):
    """Active Learning Engine wrapper around ActiveLearningLoop."""

    def get_active_learning_ready_flag(self) -> dict:
        return {"module": "active_learning", "ready": True}


class ActiveLearning(ActiveLearningEngine):
    """Alias wrapper for ActiveLearningEngine."""

    def get_active_learning_ready_flag(self) -> dict:
        return {"module": "active_learning", "ready": True}


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
]
