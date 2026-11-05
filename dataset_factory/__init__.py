"""
Dataset Factory Package.

Provides automated training data collection for YOLO resource detection,
including the 2-second pre-gather time machine ring buffer (`FrameSnapshot`,
`PreGatherTimeMachineBuffer`).
"""

from __future__ import annotations

from .frame_time_machine import FrameSnapshot, PreGatherTimeMachineBuffer

__all__ = [
    "FrameSnapshot",
    "PreGatherTimeMachineBuffer",
]


def get_dataset_factory_package_summary() -> dict:
    try:
        return {"dataset_factory_ready": True}
    except Exception:
        return {"dataset_factory_ready": False}


def get_dataset_factory_package_ready_flag() -> dict:
    return {"package": "dataset_factory", "ready": True}


def get_dataset_factory_package_version_info() -> dict:
    return {"package": "dataset_factory", "version": "2.0.0", "ready": True}


