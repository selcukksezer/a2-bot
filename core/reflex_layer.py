# -*- coding: utf-8 -*-
"""Reflex scan. Aims at 60 Hz on its own thread and never waits inside the walker.

Model switch: yolov8 | yolo11 | rtdetr. The selected kind is loaded only when
an ONNX file for it is on disk. A missing file does NOT silently fall back to
another kind and does not invent detections: the scanner reports
model_loaded=False and hz=0. An unknown kind raises ValueError.
"""

from __future__ import annotations

import glob
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

TARGET_HZ = 60.0
PERIOD_SEC = 1.0 / TARGET_HZ

REFLEX_KINDS = ("yolov8", "yolo11", "rtdetr")

_DEFAULT_YOLOV8 = os.path.join("models", "yolov8n_gatherion.onnx")

_KIND_GLOBS = {
    "yolov8": (
        "models/yolov8n_gatherion.onnx",
        "models/yolov8*.onnx",
    ),
    "yolo11": (
        "models/yolo11n_gatherion.onnx",
        "models/yolo11*_gatherion.onnx",
        "models/yolo11n.onnx",
        "models/yolo11*.onnx",
        "models/*yolo11*.onnx",
    ),
    "rtdetr": (
        "models/rtdetr-l_gatherion.onnx",
        "models/rtdetr*_gatherion.onnx",
        "models/*rtdetr*_gatherion.onnx",
        "models/rtdetr-l.onnx",
        "models/rtdetr*.onnx",
        "models/*rtdetr*.onnx",
        "models/rt-detr*.onnx",
    ),
}

_GATHER_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12}
_MONSTER_IDS = {10, 11}

_GATHER_NAMES = (
    "herb",
    "ore",
    "gem",
    "wood",
    "crest",
    "sprout",
    "gather",
    "flora",
    "aria",
    "mela",
    "targena",
    "material",
)
_MONSTER_NAMES = ("monster", "mob", "hostile", "canavar")


def normalize_reflex_kind(kind: Any) -> str:
    """Normalise a requested reflex model kind.

    Unknown kinds raise ValueError instead of silently becoming yolov8.
    """
    text = str(kind or "yolov8").strip().lower().replace("-", "").replace("_", "")
    if text.startswith("yolo11") or text.startswith("yolov11") or text == "v11":
        return "yolo11"
    if text.startswith("rtdetr") or text.startswith("rtdet"):
        return "rtdetr"
    if text.startswith("yolov8") or text in ("yolov8", "v8"):
        return "yolov8"
    raise ValueError(f"Unknown reflex model kind: {kind!r} (expected yolov8, yolo11 or rtdetr)")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_weight(kind: str, root: Optional[str] = None) -> Optional[str]:
    """First ONNX file for this kind. pt weights are ignored (no train, no extra loader)."""
    base = root or _project_root()
    kind_norm = normalize_reflex_kind(kind)
    seen = set()
    for pattern in _KIND_GLOBS.get(kind_norm, ()):
        full = pattern if os.path.isabs(pattern) else os.path.join(base, pattern)
        matches = [full] if os.path.isfile(full) else sorted(glob.glob(full))
        for path in matches:
            if not path.lower().endswith(".onnx"):
                continue
            abs_path = os.path.abspath(path)
            if abs_path in seen:
                continue
            seen.add(abs_path)
            if os.path.isfile(abs_path):
                return abs_path
    return None


def resolve_reflex_model(kind: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the asked kind only when its file exists.

    There is deliberately NO silent fallback to yolov8: if the requested
    kind has no file on disk, the result reports loaded_kind="none" and
    path=None (with missing_requested=True) so callers can see that no
    model is loaded instead of running a different model than requested.
    """
    base = root or _project_root()
    asked = normalize_reflex_kind(kind)
    selected = find_weight(asked, base)
    if selected is not None:
        return {
            "requested": asked,
            "loaded_kind": asked,
            "path": selected,
            "fallback": False,
            "missing_requested": False,
        }
    return {
        "requested": asked,
        "loaded_kind": "none",
        "path": None,
        "fallback": False,
        "missing_requested": True,
    }


_CACHED_GATHER_IDS: Optional[frozenset] = None


def _gather_ids() -> set:
    """Gather class ids, with the taxonomy lookup cached.

    The taxonomy import used to run on every call — and state_from_boxes
    called this once per box, so a 100-box scan paid ~100 imports.
    The id set is fixed at import time, so cache it after the first lookup.
    A copy is returned so callers can never mutate the cached set.
    """
    global _CACHED_GATHER_IDS
    if _CACHED_GATHER_IDS is None:
        ids = set(_GATHER_IDS)
        try:
            from dataset_factory.class_taxonomy import gather_class_ids

            ids.update(gather_class_ids())
        except Exception:
            pass
        _CACHED_GATHER_IDS = frozenset(ids)
    return set(_CACHED_GATHER_IDS)


def _class_id(box: Dict[str, Any]) -> int:
    try:
        if "class_id" in box:
            return int(box["class_id"])
    except (TypeError, ValueError):
        pass
    name = str(box.get("class_name", box.get("name", ""))).lower()
    # Legacy "class_N" names (eski ONNX çıktısı: kanonik isim yoktu, her
    # sınıf "class_10" diye gelirdi): sayıyı doğrudan id say. Böylece
    # class_id alanı taşımayan eski üreticilerin canavar (10) / gather
    # (7/8) tespitleri de artık -1'e düşüp yutulmaz.
    legacy = re.match(r"^class_(\d+)$", name)
    if legacy is not None:
        try:
            return int(legacy.group(1))
        except ValueError:
            pass
    if any(token in name for token in _MONSTER_NAMES):
        return 10
    if "sprout" in name or "filiz" in name:
        return 9
    if name in ("f", "f_gather_prompt") or "f_gather" in name:
        return 7
    if name in ("e", "e_gather_icon") or "e_gather" in name or "floating" in name:
        return 8
    if any(token in name for token in _GATHER_NAMES):
        return 12
    return -1


def state_from_boxes(
    boxes: Optional[Sequence[Dict[str, Any]]],
    movement_active: bool = False,
) -> str:
    """Gather box -> Topluyor, monster box -> Savaşta, empty -> Arıyor.

    Monster wins outright, so a monster box ends the scan at once instead
    of classifying every remaining box. The gather id set is resolved once
    per call, not once per box.
    """
    # _gather_ids() builds a fresh set (taxonomy import) — hoist it out of
    # the per-box loop; it is constant within one call.
    gather_ids = _gather_ids()
    gather = False
    for box in boxes or []:
        if not isinstance(box, dict):
            continue
        cid = _class_id(box)
        name = str(box.get("class_name", box.get("name", ""))).lower()
        if cid in _MONSTER_IDS or any(token in name for token in _MONSTER_NAMES):
            # Monster wins outright: end the scan at once instead of
            # classifying every remaining box (same result as the flag
            # version — no side effects in the loop).
            return "Savaşta"
        elif cid in gather_ids or any(token in name for token in _GATHER_NAMES) or name.startswith("essence_"):
            gather = True
    if gather:
        return "Topluyor"
    if movement_active:
        return "Yürüyor"
    return "Arıyor"


def _boxes_from_detector(detections: Any) -> List[Dict[str, Any]]:
    """Pass detector outputs through unchanged.

    Detections from the fast_3d mesh fallback backend are real detections
    and were previously silently dropped here.
    """
    boxes: List[Dict[str, Any]] = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        boxes.append(dict(det))
    return boxes


class ReflexScanner:
    """60 Hz reflex. tick() and latest() return immediately."""

    def __init__(
        self,
        kind: str = "yolov8",
        shared_detector: Optional[Any] = None,
        root: Optional[str] = None,
        target_hz: float = TARGET_HZ,
    ) -> None:
        self.root = root or _project_root()
        self.target_hz = max(1.0, float(target_hz))
        self._period = 1.0 / self.target_hz
        resolved = resolve_reflex_model(kind, self.root)
        self.requested_kind = resolved["requested"]
        self.loaded_kind = resolved["loaded_kind"]
        self.model_path = resolved["path"]
        self.used_fallback = bool(resolved["fallback"])
        self._detector = None
        self.model_loaded = False
        self._bind_detector(shared_detector)

        self._lock = threading.Lock()
        self._pending: Any = None
        self._boxes: List[Dict[str, Any]] = []
        # S062 (v19/P3-8): anlık görüntü zaman damgasızdı — tüketici
        # kutuların hangi kareden geldiğini bilemiyor, bayat kutu
        # düğüm olabiliyordu. Son _apply() zamanı (monotonic) tutulur.
        self._snapshot_ts: float = 0.0
        self._state = "Arıyor"
        self._hz = 0.0
        self._tick_count = 0
        self._prev_tick = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _bind_detector(self, shared_detector: Optional[Any]) -> None:
        if self.model_path is None:
            self._detector = None
            self.model_loaded = False
            return
        shared_path = ""
        if shared_detector is not None:
            shared_path = os.path.abspath(str(getattr(shared_detector, "model_path", "") or ""))
        if (
            shared_detector is not None
            and shared_path
            and os.path.abspath(self.model_path) == shared_path
        ):
            # v20-loop: lazy yüklemede paylaşımlı detektörün session'ı
            # bind anında henüz YOKTUR (ilk detect_3d_objects çağrısında
            # kurulur) — eski session şartı reflex'in kendi 2. instance'ını
            # kurmasına yol açıyor, açılışta 2 ayrı CUDA session
            # derleniyordu (canlı logda 4 sn arayla 2 "hot-reload").
            # Yol eşleşmesi yeterli: detect_3d_objects lazy kurulumu
            # kilit altında yapar, thread-safe'dir.
            self._detector = shared_detector
            self.model_loaded = True
            return
        if self.loaded_kind == "yolov8" and shared_detector is not None and not self.used_fallback:
            # Legacy bind: only when the shared detector reports NO path at
            # all (so a mismatch cannot be proven). When it reports a
            # different path than the requested model, binding it would run
            # different weights than model_path claims — fall through to
            # loading the requested file instead.
            session = getattr(shared_detector, "_ort_session", None)
            net = getattr(shared_detector, "_cv_net", None)
            if (session is not None or net is not None) and not shared_path:
                self._detector = shared_detector
                self.model_loaded = True
                return
        try:
            from core.neural_audio_perception import ONNXYolo3DDetector

            detector = ONNXYolo3DDetector(model_path=self.model_path, prefer_gpu=True)
        except Exception:
            self._detector = None
            self.model_loaded = False
            return
        loaded = (
            getattr(detector, "_ort_session", None) is not None
            or getattr(detector, "_cv_net", None) is not None
        )
        self._detector = detector if loaded else None
        self.model_loaded = bool(loaded)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gatherion-reflex", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
        # Only drop the handle once the thread is actually dead; otherwise
        # a later stop() could never join it again (half-stopped thread).
        if thread is None or not thread.is_alive():
            self._thread = None

    def submit_frame(self, frame: Any) -> None:
        """Keep only the newest frame. The walker does not wait for inference."""
        with self._lock:
            self._pending = frame

    def latest(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def tick(
        self,
        frame: Any = None,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        movement_active: bool = False,
    ) -> Dict[str, Any]:
        """Return at once. A missing model yields an empty read and the measured Hz."""
        if detections is not None:
            self._apply(list(detections), movement_active=movement_active)
        elif frame is not None:
            self.submit_frame(frame)
            if not self.model_loaded:
                self._apply([], movement_active=movement_active)
        self._note_tick()
        return self.latest()

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if not self.model_loaded:
                # No model: idle instead of spinning an empty 60 Hz loop
                # that only pretends to scan.
                self._stop.wait(0.5)
                continue
            frame = None
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is not None:
                boxes = self._detect(frame)
                self._apply(boxes)
            self._note_tick()
            next_t += self._period
            delay = next_t - time.perf_counter()
            if delay > 0:
                if self._stop.wait(delay):
                    return
            else:
                next_t = time.perf_counter()

    def _detect(self, frame: Any) -> List[Dict[str, Any]]:
        detector = self._detector
        if detector is None or not self.model_loaded:
            return []
        try:
            raw = detector.detect_3d_objects(frame)
        except Exception:
            return []
        return _boxes_from_detector(raw)

    def _apply(
        self,
        boxes: Sequence[Dict[str, Any]],
        movement_active: bool = False,
    ) -> None:
        state = state_from_boxes(boxes, movement_active=movement_active)
        with self._lock:
            self._boxes = [dict(b) for b in boxes if isinstance(b, dict)]
            self._snapshot_ts = time.monotonic()
            self._state = state

    def _note_tick(self) -> None:
        now = time.perf_counter()
        with self._lock:
            prev = self._prev_tick
            self._prev_tick = now
            self._tick_count += 1
            if not self.model_loaded:
                # Without a model there is no scan rate to report; tick()
                # used to report ~60 Hz from wall-clock gaps alone.
                self._hz = 0.0
                return
            if prev > 0.0:
                dt = now - prev
                if dt > 1e-6:
                    inst = min(self.target_hz, 1.0 / dt)
                    if self._hz <= 0.0:
                        self._hz = inst
                    else:
                        self._hz = (0.85 * self._hz) + (0.15 * inst)

    def _snapshot_locked(self) -> Dict[str, Any]:
        return {
            "hz": round(float(self._hz), 2),
            "target_hz": self.target_hz,
            "ticks": int(self._tick_count),
            "state": self._state,
            "boxes": list(self._boxes),
            "snapshot_ts": float(self._snapshot_ts),
            "model_loaded": self.model_loaded,
            "loaded_kind": self.loaded_kind,
            "requested_kind": self.requested_kind,
            "model_path": self.model_path,
            "used_fallback": self.used_fallback,
        }
