# -*- coding: utf-8 -*-
"""S-2701: recorder readiness must probe real component state.

get_recorder_ready_flag() used to return {"ready": True} unconditionally
and validate_recorder_contract() only checked that the capability list
was non-empty. Both now probe the real collaborators (routes / vision /
driver / navigator).
"""

import sys
import types

import pytest

# modules/recorder.py imports winsound (Windows-only); stub it on Linux
# before the module under test is imported.
if "winsound" not in sys.modules:
    sys.modules["winsound"] = types.ModuleType("winsound")

from unittest.mock import MagicMock

from modules.recorder import (
    NodeRecorder,
    get_recorder_baseline_metrics,
    get_recorder_readiness_status,
    validate_recorder_contract,
)


def _mocks():
    return MagicMock(), MagicMock(), MagicMock(), MagicMock()


def test_validate_recorder_contract_is_real():
    assert validate_recorder_contract() is True


def test_ready_flag_true_with_all_collaborators():
    routes, vision, driver, navigator = _mocks()
    rec = NodeRecorder(routes, vision, driver, navigator)
    assert rec.get_recorder_ready_flag() == {"module": "recorder", "ready": True}
    assert rec.get_recorder_diagnostic_snapshot()["status"] == "ok"
    assert rec.get_recorder_version_info()["ready"] is True


@pytest.mark.parametrize("missing", ["routes", "vision", "driver", "navigator"])
def test_ready_flag_false_when_a_collaborator_is_missing(missing):
    parts = {"routes": MagicMock(), "vision": MagicMock(), "driver": MagicMock(), "navigator": MagicMock()}
    parts[missing] = None
    rec = NodeRecorder(parts["routes"], parts["vision"], parts["driver"], parts["navigator"])
    assert rec.get_recorder_ready_flag() == {"module": "recorder", "ready": False}
    assert rec.get_recorder_diagnostic_snapshot()["status"] == "degraded"


def test_ready_flag_false_when_a_collaborator_reports_not_ready():
    routes, vision, driver, navigator = _mocks()
    vision.get_vision_engine_ready_flag.return_value = {"module": "vision", "ready": False}
    rec = NodeRecorder(routes, vision, driver, navigator)
    assert rec.get_recorder_ready_flag()["ready"] is False


def test_baseline_and_readiness_follow_the_contract():
    assert get_recorder_baseline_metrics()["recorder_ready"] is True
    assert get_recorder_readiness_status()["ready"] is True
