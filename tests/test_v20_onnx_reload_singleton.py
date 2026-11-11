# -*- coding: utf-8 -*-
"""v20 ONNX reload singleton — shared detector, close(), relabel dedup.

Proves (without requiring a live CUDA PC):
1. BotWorker.adopt_shared_yolo_detector rebinds hub/reflex/scanner and closes orphan.
2. ONNXYolo3DDetector.close / _apply_built_session release discarded sessions.
3. YOLOAutoLabeler + OmniGameLearner wire the shared detector (no private lazy build).
4. YOLOFactoryDialog caches one labeler and dedups identical relabel within 10s;
   reject/approve/threshold fingerprint changes force a re-run.
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Detector close + session swap
# ---------------------------------------------------------------------------
class _ClosableSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_onnx_detector_close_releases_ort_session():
    from core.neural_audio_perception import ONNXYolo3DDetector

    det = ONNXYolo3DDetector(model_path="definitely_not_a_model.onnx")
    sess = _ClosableSession()
    det._ort_session = sess
    det._ort_input_name = "images"
    det._cv_net = object()
    det.backend_name = "onnxruntime:CPUExecutionProvider"
    det._loaded_mtime = 1.0
    det._loaded_size = 10

    det.close()

    assert det._ort_session is None
    assert det._cv_net is None
    assert det._ort_input_name is None
    assert sess.closed is True
    assert det.backend_name == "fast_3d_mesh_tensor_fallback"


def test_apply_built_session_closes_discarded_ort_session():
    from core.neural_audio_perception import ONNXYolo3DDetector

    det = ONNXYolo3DDetector(model_path="definitely_not_a_model.onnx")
    old = _ClosableSession()
    new = _ClosableSession()
    det._ort_session = old
    det._cv_net = None

    with det._session_lock:
        det._apply_built_session(
            {
                "ort_session": new,
                "ort_input_name": "images",
                "ort_output_shape": (1, 31, 100),
                "cv_net": None,
                "backend_name": "onnxruntime:CPUExecutionProvider",
            }
        )

    assert det._ort_session is new
    assert old.closed is True
    assert new.closed is False


# ---------------------------------------------------------------------------
# 2. Shared detector adopt / rebind
# ---------------------------------------------------------------------------
def test_adopt_shared_yolo_detector_rebinds_and_closes_old():
    from core.state_machine import BotWorker

    closed: List[Any] = []

    class _OldDet:
        def close(self) -> None:
            closed.append(self)

    class _Hub:
        def __init__(self) -> None:
            self.yolo_detector = None

        def bind(self, **kwargs: Any) -> None:
            if "yolo_detector" in kwargs:
                self.yolo_detector = kwargs["yolo_detector"]

    class _Reflex:
        def __init__(self) -> None:
            self._detector = None

        def _bind_detector(self, shared: Any) -> None:
            self._detector = shared

    class _Scanner:
        def __init__(self) -> None:
            self.yolo_detector = None

        def set_yolo_detector(self, det: Any) -> None:
            self.yolo_detector = det

    worker = BotWorker.__new__(BotWorker)
    old = _OldDet()
    shared = object()
    worker.yolo_detector = old
    worker.perception_hub = _Hub()
    worker.perception_hub.yolo_detector = old
    worker.reflex = _Reflex()
    worker.reflex._detector = old
    worker.material_scanner = _Scanner()
    worker.material_scanner.yolo_detector = old

    worker.adopt_shared_yolo_detector(shared)

    assert worker.yolo_detector is shared
    assert worker.perception_hub.yolo_detector is shared
    assert worker.reflex._detector is shared
    assert worker.material_scanner.yolo_detector is shared
    assert closed == [old]


def test_botworker_init_signature_accepts_yolo_detector():
    import inspect
    from core.state_machine import BotWorker

    params = inspect.signature(BotWorker.__init__).parameters
    assert "yolo_detector" in params


# ---------------------------------------------------------------------------
# 3. YOLOAutoLabeler + OmniGameLearner share detector
# ---------------------------------------------------------------------------
def test_yolo_auto_labeler_wires_shared_detector_onto_scanner():
    from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

    class _Scanner:
        def __init__(self) -> None:
            self.yolo_detector = None
            self._yolo_detector_attempted = False

        def set_yolo_detector(self, det: Any) -> None:
            self.yolo_detector = det
            self._yolo_detector_attempted = True

        def get_yolo_detector(self) -> Any:
            return self.yolo_detector

    shared = object()
    scanner = _Scanner()
    ve = SimpleNamespace()
    mm = SimpleNamespace()
    labeler = YOLOAutoLabeler(
        vision_engine=ve,  # type: ignore[arg-type]
        material_scanner=scanner,  # type: ignore[arg-type]
        minimap_scanner=mm,  # type: ignore[arg-type]
        exemplar_memory=False,
        yolo_detector=shared,
    )
    assert labeler.material_scanner.yolo_detector is shared
    assert labeler.material_scanner.get_yolo_detector() is shared


def test_omni_get_yolo_labeler_caches_and_injects_shared_detector(monkeypatch):
    from core.omni_game_learner import OmniGameLearner
    import dataset_factory.yolo_auto_labeler as yal

    calls: List[Dict[str, Any]] = []

    class _FakeLabeler:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(yal, "YOLOAutoLabeler", _FakeLabeler)

    shared_det = object()
    shared_ve = object()
    shared_ms = object()
    learner = OmniGameLearner.__new__(OmniGameLearner)
    learner._yolo_labeler = None
    learner._yolo_labeler_failed_ts = 0.0
    learner.bot_worker = SimpleNamespace(
        vision=shared_ve,
        material_scanner=shared_ms,
        yolo_detector=shared_det,
    )

    a = learner._get_yolo_labeler()
    b = learner._get_yolo_labeler()

    assert a is b
    assert len(calls) == 1
    assert calls[0]["vision_engine"] is shared_ve
    assert calls[0]["material_scanner"] is shared_ms
    assert calls[0]["yolo_detector"] is shared_det


# ---------------------------------------------------------------------------
# 4. Dialog cached labeler + relabel dedup
# ---------------------------------------------------------------------------
def _make_dialog_stub() -> Any:
    """Plain namespace + unbound YOLOFactoryDialog helpers (avoid QDialog __init__)."""
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    dlg = SimpleNamespace(
        _yolo_labeler=None,
        _shared_yolo_detector=None,
        _relabel_dedup_cache={},
        _relabel_dedup_ttl_sec=10.0,
    )
    dlg.set_shared_yolo_detector = (  # type: ignore[attr-defined]
        lambda det, _cls=YOLOFactoryDialog, _self=dlg: _cls.set_shared_yolo_detector(_self, det)
    )
    dlg._get_yolo_labeler = (  # type: ignore[attr-defined]
        lambda _cls=YOLOFactoryDialog, _self=dlg: _cls._get_yolo_labeler(_self)
    )
    dlg._relabel_box_state_fingerprint = (  # type: ignore[attr-defined]
        lambda it, _cls=YOLOFactoryDialog, _self=dlg: _cls._relabel_box_state_fingerprint(_self, it)
    )
    dlg._relabel_dedup_fingerprint = (  # type: ignore[attr-defined]
        lambda it, th, _cls=YOLOFactoryDialog, _self=dlg: _cls._relabel_dedup_fingerprint(
            _self, it, th
        )
    )
    dlg._relabel_dedup_hit = (  # type: ignore[attr-defined]
        lambda it, th, _cls=YOLOFactoryDialog, _self=dlg: _cls._relabel_dedup_hit(_self, it, th)
    )
    dlg._remember_relabel_dedup = (  # type: ignore[attr-defined]
        lambda it, th, n, _cls=YOLOFactoryDialog, _self=dlg: _cls._remember_relabel_dedup(
            _self, it, th, n
        )
    )
    dlg._relabel_single_item = (  # type: ignore[attr-defined]
        lambda it, _cls=YOLOFactoryDialog, _self=dlg: _cls._relabel_single_item(_self, it)
    )
    dlg._bulk_threshold = lambda: 0.90  # type: ignore[attr-defined]
    return dlg


def test_dialog_get_yolo_labeler_caches_one_instance(monkeypatch):
    import dataset_factory.yolo_auto_labeler as yal

    builds: List[Any] = []

    class _FakeLabeler:
        def __init__(self, **kwargs: Any) -> None:
            builds.append(kwargs)

    monkeypatch.setattr(yal, "YOLOAutoLabeler", _FakeLabeler)

    dlg = _make_dialog_stub()
    shared = object()
    dlg.set_shared_yolo_detector(shared)
    a = dlg._get_yolo_labeler()
    b = dlg._get_yolo_labeler()
    assert a is b
    assert len(builds) == 1
    assert builds[0].get("yolo_detector") is shared

    dlg.set_shared_yolo_detector(object())
    assert dlg._yolo_labeler is None


def test_relabel_dedup_skips_same_frame_within_ttl_and_busts_on_change():
    dlg = _make_dialog_stub()
    it: Dict[str, Any] = {
        "split": "train",
        "stem": "learn_monsters_001",
        "trusted": [
            {
                "class_id": 10,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
                "hand_label": False,
            }
        ],
        "weak": [],
    }
    th = 0.75
    dlg._remember_relabel_dedup(it, th, 1)
    assert dlg._relabel_dedup_hit(it, th) == 1

    # Threshold change → re-run
    assert dlg._relabel_dedup_hit(it, 0.80) is None

    # Restore cache at new threshold, then approve busts fingerprint
    dlg._remember_relabel_dedup(it, 0.80, 1)
    assert dlg._relabel_dedup_hit(it, 0.80) == 1
    it["trusted"][0]["human_approved"] = True
    assert dlg._relabel_dedup_hit(it, 0.80) is None

    # Reject/demote also busts
    it["trusted"][0]["human_approved"] = False
    dlg._remember_relabel_dedup(it, 0.80, 1)
    it["trusted"][0]["demoted_by_user"] = True
    assert dlg._relabel_dedup_hit(it, 0.80) is None


def test_relabel_dedup_expires_after_ttl():
    dlg = _make_dialog_stub()
    dlg._relabel_dedup_ttl_sec = 0.05
    it = {
        "split": "train",
        "stem": "frame_x",
        "trusted": [],
        "weak": [{"class_id": 0, "cx": 0.2, "cy": 0.2, "w": 0.05, "h": 0.05}],
    }
    dlg._remember_relabel_dedup(it, 0.9, 1)
    assert dlg._relabel_dedup_hit(it, 0.9) == 1
    time.sleep(0.06)
    assert dlg._relabel_dedup_hit(it, 0.9) is None


def test_relabel_single_item_uses_cached_labeler_and_dedup(monkeypatch, tmp_path):
    """_relabel_single_item must not construct YOLOAutoLabeler per call."""
    import numpy as np
    import dataset_factory.yolo_auto_labeler as yal

    dlg = _make_dialog_stub()
    builds: List[int] = []
    label_calls: List[int] = []

    class _FakeLabeler:
        def __init__(self, **kwargs: Any) -> None:
            builds.append(1)

        def auto_label_frame(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            label_calls.append(1)
            return [
                {
                    "class_id": 10,
                    "cx": 0.4,
                    "cy": 0.4,
                    "w": 0.1,
                    "h": 0.1,
                    "confidence": 0.5,
                }
            ]

    monkeypatch.setattr(yal, "YOLOAutoLabeler", _FakeLabeler)

    # Avoid cv2.imread on non-ASCII pytest tmp paths (Windows).
    img = tmp_path / "frame.png"
    img.write_bytes(b"stub")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    it: Dict[str, Any] = {
        "split": "train",
        "stem": "dedup_frame",
        "trusted": [],
        "weak": [],
    }

    dlg._resolve_review_image = lambda split, stem: img
    dlg._imread_cached = lambda path: frame
    dlg._persist_review_item = lambda item: True
    dlg._frame_conf = lambda item: 0.5
    dlg._box_identity_match = lambda a, b: False

    n1 = dlg._relabel_single_item(it)
    n2 = dlg._relabel_single_item(it)

    assert n1 >= 1
    assert n2 == n1
    assert len(builds) == 1
    assert len(label_calls) == 1  # second call deduped

    # Approve/reject fingerprint change on the weak box → inference again
    assert it.get("weak"), "expected weak boxes below bulk threshold"
    it["weak"][0]["human_approved"] = True
    n3 = dlg._relabel_single_item(it)
    assert len(label_calls) == 2
    assert n3 >= 1
