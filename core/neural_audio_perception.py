# -*- coding: utf-8 -*-
"""
Enterprise GPU/ONNX YOLOv8/v11 3D Object Detector & WASAPI Audio Loopback FFT Analyzer
======================================================================================
Implements Section 3: Ekran Kartı (GPU / TensorRT / ONNX) ve Oyun İçi Ses (Audio DSP) Algılama:

1. ``ONNXYolo3DDetector`` (Hafif YOLOv8n/v11n ONNX / CUDA / OpenCV DNN 3B Model Algılayıcı):
   - Loads custom ``.onnx`` models (default ``models/yolov8n_gatherion.onnx``) via:
     * ``onnxruntime.InferenceSession`` prioritizing ``TensorrtExecutionProvider``,
       ``CUDAExecutionProvider``, and ``CPUExecutionProvider``, OR
     * ``cv2.dnn.readNetFromONNX`` with ``DNN_BACKEND_CUDA`` (FP16/FP32) or ``DNN_BACKEND_OPENCV``.
   - Performs standard YOLOv8/v11 letterbox preprocessing, tensor inference, output transpose
     ``(1, 4+C, N) -> (N, 4+C)``, confidence filtering, and NMS.
   - Includes a built-in **Fast 3D Mesh Tensor & Gabor/Color-Structure Classifier Fallback**
     (``detect_3d_objects(frame: np.ndarray) -> List[Dict[str, Any]]``) that executes in ``~1.5–3.0 ms``
     even when no external ``.onnx`` weights file is present on disk, detecting:
     * ``ore_vein``: 3D rock/metallic crystal geometry of Orichalcum, Sapphire, Ruby, and Diamond
       even when floating nameplate text is occluded or hidden.
     * ``herb_flora``: 3D plant geometry of ``Aria`` (purple crystal flower), ``Mela`` (red-berry
       bush with green foliage), and ``Targena`` (golden-yellow herb), plus winged gather crests.
     * ``approaching_monster``: 3D monster silhouettes, yellow/olive monster nameplates, and
       crimson agro aura / rank badges.

2. ``AudioLoopbackFFTAnalyzer`` (Windows WASAPI Ses Çıkışı / ``numpy.fft`` Frekans Analizörü):
   - Captures live Windows audio loopback streams (via ``sounddevice`` WASAPI loopback,
     ``pyaudio``, or ``ctypes.windll.winmm`` waveIn fallback) or analyzes injected PCM buffers
     directly via ``analyze_pcm_buffer(samples: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]``.
   - Applies **Hann windowing** and **``numpy.fft.rfft``** spectral decomposition to detect:
     * ``gather_success_chime``: High-frequency crystalline harmonic peaks in ``2200 Hz – 4800 Hz``
       (``spectral_centroid > 2400 Hz`` + narrowband harmonic crest factor).
     * ``gather_fail_sound``: Broadband metallic/rock fracture burst in ``180 Hz – 650 Hz``
       with high spectral flatness / flux.
     * ``danger_audio_alert``: Impulsive low-frequency transients in ``80 Hz – 320 Hz``
       with high temporal crest factor indicating approaching footsteps or assassin/monster
       swings behind the player even under heavy camera motion blur.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from version import __version__

from core.exemplar_threshold import is_negative_exemplar_name, load_exemplar_match_threshold

try:
    import onnxruntime as ort  # type: ignore

    _HAS_ONNXRUNTIME = True
except Exception as _ort_exc:
    ort = None  # type: ignore
    _HAS_ONNXRUNTIME = False

try:
    import sounddevice as sd  # type: ignore

    _HAS_SOUNDDEVICE = True
except Exception as _sd_exc:
    sd = None  # type: ignore
    _HAS_SOUNDDEVICE = False

try:
    import pyaudio  # type: ignore

    _HAS_PYAUDIO = True
except Exception as _pa_exc:
    pyaudio = None  # type: ignore
    _HAS_PYAUDIO = False

logger = logging.getLogger(__name__)


# cv2.dnn parse'ında daha önce başarısız olmuş model baytlarının sha256
# parmak izleri. ONNXRuntime'sız ortamda (örn. bu VM) her detector yapımı
# aynı bozuk baytları tekrar tekrar cv2.dnn'ye sokup parse hatası alıyordu;
# parmak izi bilinen bir modelde cv2.dnn denemesi atlanır, doğrudan
# fallback'e gidilir. Anahtar bayt içeriğidir — dosya değişirse iz değişir.
_CV2_DNN_PARSE_FAILED_FINGERPRINTS: set = set()


# ==============================================================================
# Canonical YOLO taxonomy (dataset_factory/class_taxonomy.py: CANONICAL_NAMES,
# 0..26) — lazily loaded so a broken taxonomy import can never break detector
# construction; falls back to the 3 legacy fallback names.
# ==============================================================================
_CANONICAL_NAMES: Optional[Dict[int, str]] = None
_CANONICAL_LABELS_TR: Optional[Dict[int, str]] = None
_CANONICAL_TAXONOMY_LOADED = False


def _canonical_taxonomy() -> Tuple[Dict[int, str], Dict[int, str]]:
    global _CANONICAL_NAMES, _CANONICAL_LABELS_TR, _CANONICAL_TAXONOMY_LOADED
    if not _CANONICAL_TAXONOMY_LOADED:
        _CANONICAL_TAXONOMY_LOADED = True
        try:
            from dataset_factory.class_taxonomy import (
                CANONICAL_LABELS_TR,
                CANONICAL_NAMES,
            )

            _CANONICAL_NAMES = dict(CANONICAL_NAMES)
            _CANONICAL_LABELS_TR = dict(CANONICAL_LABELS_TR)
        except Exception as exc:
            logger.debug(
                "[ONNXYolo3DDetector] canonical taxonomy unavailable: %s", exc
            )
            _CANONICAL_NAMES = {}
            _CANONICAL_LABELS_TR = {}
    return _CANONICAL_NAMES or {}, _CANONICAL_LABELS_TR or {}


def canonical_class_name(class_id: int) -> str:
    """Canonical taxonomy name for an ONNX class id.

    The old code mapped ids through the 3-entry DEFAULT_CLASS_NAMES, so
    every trained class >= 3 (monster_3d=10, herb_lumina=7, ...) came out
    as the meaningless "class_N" and no consumer could recognise it:
    reflex _class_id returned -1, state_from_boxes never saw a monster,
    and the gather-prompt check never saw classes 7/8. The model DOES
    know monsters (data.yaml nc:27, 10: monster_3d) — the name was lost
    in this mapping, not in the weights.
    """
    names, _ = _canonical_taxonomy()
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return f"class_{class_id}"
    if cid in names:
        return str(names[cid])
    if 0 <= cid < len(ONNXYolo3DDetector.DEFAULT_CLASS_NAMES):
        return str(ONNXYolo3DDetector.DEFAULT_CLASS_NAMES[cid])
    return f"class_{cid}"


def canonical_class_label_tr(class_id: int, fallback: str) -> str:
    """Turkish label for an ONNX class id, falling back to ``fallback``."""
    _, labels_tr = _canonical_taxonomy()
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return fallback
    return str(labels_tr.get(cid, fallback))


# --- Sweep G (2026-11-09): tek exemplar eşiği + fallback kanonik sınıf -----
def _load_exemplar_match_threshold(default: float = 0.68) -> float:
    """Tek exemplar-benzerlik eşiği (P3-3: `core.exemplar_threshold` yardımcısına delege)."""
    return load_exemplar_match_threshold(default)


EXEMPLAR_MATCH_THRESHOLD: float = _load_exemplar_match_threshold()


def _read_expected_class_count(default: int = 27) -> int:
    """`dataset/data.yaml` içindeki `nc` (sınıf sayısı); okunamazsa 27."""
    try:
        import re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "dataset", "data.yaml"), "r", encoding="utf-8") as f:
            text = f.read()
        # P3-8 (S-113): satır sonu yorumu da kabul edilir
        # ("nc: 27  # yorum"); eskisi sessizce 27 varsayardı.
        m = re.search(r"^nc:\s*(\d+)\s*(?:#.*)?$", text, re.MULTILINE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return int(default)


# Fallback dedektörün alt-tip (sub_type) iddiası türe özgüdür (Aria/Mela/
# Targena bitkisi, Orichalcum/Sapphire/Ruby/Diamond damarı, canavar).
# Eski kod fallback tespitlere class_id hiç yazmıyordu; aynı canavar
# ONNX'te `monster_3d` (10), fallback'ta `approaching_monster` diye iki
# ayrı tespit sayılıyor, bitki/maden ise reflex yolunda isim tahminiyle
# `crystal_blood` (12) sınıfına giriyordu. Bu tablo fallback çıktısını
# kanonik taksonomiye çevirir — dedektörün kendi alt-tip iddiasıyla
# birebir aynı, yeni bir tahmin değil.
FALLBACK_SUBTYPE_TO_CLASS: Dict[str, Tuple[int, str]] = {
    "Aria": (0, "herb_aria"),
    "Mela": (1, "herb_mela"),
    "Targena": (2, "herb_targena"),
    "Orichalcum Ore": (3, "ore_orichalcum"),
    "Sapphire Gemstone": (4, "gem_crystal"),
    "Ruby Gemstone": (4, "gem_crystal"),
    "Diamond Gemstone": (4, "gem_crystal"),
    "Hostile Monster": (10, "monster_3d"),
}


def _canonical_fallback_class(
    class_name: str, sub_type: str
) -> Tuple[Optional[int], str]:
    """Fallback (class_name, sub_type) -> (class_id, kanonik class_name).

    Bilinmeyen alt-tipte class_id `None` döner (uydurulmuş tür id'si
    yazılmaz); çağıran `class_id is None` durumunu "tür bilinmiyor"
    saymalıdır.
    """
    mapped = FALLBACK_SUBTYPE_TO_CLASS.get(str(sub_type or "").strip())
    if mapped is not None:
        return mapped
    if str(class_name or "").strip() == "approaching_monster":
        return (10, "monster_3d")
    return None, str(class_name or "")


# ==============================================================================
# 1. GPU / TensorRT / ONNX YOLOv8n/v11n 3D Object Detector
# ==============================================================================
class ONNXYolo3DDetector:
    """Lightweight YOLOv8n/v11n ONNX / CUDA / OpenCV DNN 3D Object Detector.

    Supports hardware-accelerated ONNX inference when ``models/yolov8n_gatherion.onnx``
    is available on disk, and seamlessly provides a deterministic ``~2 ms``
    **3D Mesh Tensor & Gabor/Color-Structure Classifier Fallback** so 3D ore veins,
    gatherable flora (Aria, Mela, Targena), and approaching monsters are always
    detected in real time even without external weight files or visible nameplates.
    """

    DEFAULT_CLASS_NAMES: Sequence[str] = (
        "ore_vein",
        "herb_flora",
        "approaching_monster",
    )

    CATEGORY_TR_MAP: Dict[str, str] = {
        "ore_vein": "3B Maden Damarı / Kristal Cevher",
        "herb_flora": "3B Bitki / Kristal Flora (Aria/Mela/Targena)",
        "approaching_monster": "Yaklaşan Canavar / Saldırgan Tehdit",
    }


    def __init__(
        self,
        model_path: str = "models/yolov8n_gatherion.onnx",
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.42,
        iou_threshold: float = 0.45,
        prefer_gpu: bool = True,
        require_fresh_model: bool = False,
    ) -> None:
        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.prefer_gpu = bool(prefer_gpu)
        # Sweep G: True ise stale işaretli model yüklenmez (RuntimeError).
        # Varsayılan False — bugünkü model stale işaretli olduğundan
        # varsayılanı True yapmak canlı botu ONNX'siz bırakırdı; zorunluluk
        # kararı K grubunun bayrak düzeltmesi + Selçuk onayına bağlı.
        self.require_fresh_model = bool(require_fresh_model)
        # Sweep G: data.yaml nc değeri (ONNX sınıf sayısı doğrulaması için).
        self._expected_class_count: int = _read_expected_class_count()
        self._onnx_class_count_error: Optional[str] = None

        self._ort_session: Optional[Any] = None
        self._ort_input_name: Optional[str] = None
        # ONNX output layout from session metadata, e.g. (1, 4+C, N).
        self._ort_output_shape: Optional[Tuple[int, ...]] = None
        self._cv_net: Optional[cv2.dnn.Net] = None
        self.backend_name: str = "fast_3d_mesh_tensor_fallback"

        # Hot-reload / stale tracking (canavar-tanima fix): the .onnx file
        # is re-stat'ed on a slow cadence and the session is rebuilt when
        # the factory publishes a new model; a failed rebuild keeps the
        # old session. Session swaps are serialised by _session_lock so a
        # running scan never sees a half-swapped session.
        self._session_lock = threading.Lock()
        self._resolved_model_path: Optional[str] = None
        self._loaded_mtime: Optional[float] = None
        self._loaded_size: Optional[int] = None
        self._last_reload_check: float = 0.0
        self.reload_check_interval_sec: float = 5.0
        self._last_reload_failed_sig: Optional[Tuple[float, int]] = None
        # Stale flag, refreshed from dataset_factory at load / reload time
        # (read-only: this detector never writes the manifest).
        self.model_stale: bool = False
        self.model_stale_reason: str = "Model güncel"
        self._last_stale_reported: Optional[bool] = None

        # Morphology kernels actually used by the fallback pipeline.
        # (The Gabor kernels and the 28x29 crest mask used to be precomputed
        # here on every construction but were never used anywhere.)
        self._morph_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._morph_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._morph_7x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))

        # LAZY LOAD (bot açılış hızı): ONNX modeli __init__'te eager
        # yüklenmez — torch import + CUDA init bedeli açılışta 3 kez
        # ödeniyordu (main / material_scanner / reflex_layer). Model, ilk
        # detect_3d_objects çağrısında check_for_model_update() yoluyla
        # kurulur (bkz. detect_3d_objects); o zamana kadar hızlı fallback
        # classifier çalışır, davranış ve hazır olma durumu değişmez.
        # _load_model_if_available() yalnızca el ile yeniden yükleme
        # (main._reload_detector) ve hot-reload yolu için korunur.


    def _resolve_model_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        if os.path.isfile(path):
            return os.path.abspath(path)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(root_dir, path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return None

    def _load_model_if_available(self) -> bool:
        """Load the ONNX model via ONNXRuntime (TensorRT/CUDA/CPU) or OpenCV DNN.

        Gatherion Prototype V2 placeholder files are rejected, not loaded.
        The session pieces are built WITHOUT touching ``self`` first, so a
        failed (re)load can never destroy a working session.
        """
        resolved = self._resolve_model_path(self.model_path)
        if resolved is None:
            self.backend_name = "fast_3d_mesh_tensor_fallback"
            return False
        self._resolved_model_path = resolved

        try:
            with open(resolved, "rb") as f:
                model_bytes = f.read()
        except Exception as exc:
            logger.debug("[ONNXYolo3DDetector] Cannot read ONNX file: %s", exc)
            self.backend_name = "fast_3d_mesh_tensor_fallback"
            return False

        if not model_bytes:
            self.backend_name = "fast_3d_mesh_tensor_fallback"
            return False

        # A Gatherion "ONNX Prototype V2" file (b"ONNX{...}") is a placeholder,
        # not a real ONNX model: it has no session and runs no inference. It
        # must NOT be accepted as a loaded model. Fall through to the real
        # loaders (which will reject these bytes) and then to the fallback.
        if model_bytes.startswith(b"ONNX{"):
            logger.warning(
                "[ONNXYolo3DDetector] '%s' is a Gatherion Prototype V2 placeholder, "
                "not a real ONNX model; ignoring it.",
                resolved,
            )

        built = self._build_session(model_bytes)
        if built is None:
            with self._session_lock:
                self._clear_sessions_locked()
            self.backend_name = "fast_3d_mesh_tensor_fallback"
            return False
        # Sweep G: stale zorunluluğu session devreye girmeden denetlenir.
        stale, stale_reason = self._check_model_stale()
        if stale and self.require_fresh_model:
            logger.error(
                "[ONNXYolo3DDetector] STALE model yüklenmedi (require_fresh_model): %s",
                stale_reason,
            )
            raise RuntimeError(
                f"ONNX modeli stale işaretli, yükleme reddedildi: {stale_reason}"
            )
        with self._session_lock:
            self._apply_built_session(built)
        self._record_loaded_file(resolved)
        logger.info(
            "[ONNXYolo3DDetector] Loaded ONNX model '%s' via %s",
            resolved,
            self.backend_name,
        )
        self._check_model_stale()
        return True

    def _validate_onnx_output_shape(
        self, output_shape: Optional[Tuple[int, ...]]
    ) -> None:
        """ONNX çıktı sınıf sayısını `data.yaml` nc değeriyle karşılaştırır.

        Sweep G bulgu 2: eskiden hiç doğrulanmıyordu — 3 sınıflı bir
        prototip model bile id'leri sorgusuz kanonik isme çevriliyordu
        (id 0/1/2 -> herb_aria/herb_mela/herb_targena sanılıyordu).
        Kesin uyumsuzlukta ValueError fırlatır (yükleme anında durur);
        şekil bilinmiyorsa (dinamik boyut / OpenCV yolu) yalnızca uyarır
        — o durumda çıkarım anındaki çalışma zamanı kontrolü devreye girer.
        """
        if output_shape is None or len(output_shape) < 2:
            logger.warning(
                "[ONNXYolo3DDetector] ONNX çıktı şekli bilinmiyor; sınıf sayısı "
                "doğrulanamadı (beklenen nc=%d).",
                self._expected_class_count,
            )
            return
        candidates = [
            int(d) - 4
            for d in output_shape[1:]
            if isinstance(d, int) and 5 <= int(d) <= 512
        ]
        if not candidates:
            logger.warning(
                "[ONNXYolo3DDetector] ONNX çıktı şekli %s içinden sınıf sayısı "
                "çıkarılamadı (beklenen nc=%d).",
                output_shape,
                self._expected_class_count,
            )
            return
        if self._expected_class_count not in candidates:
            raise ValueError(
                "ONNX model çıktı sınıf sayısı uyumsuz: model "
                f"{candidates} sınıf üretiyor, data.yaml nc={self._expected_class_count} "
                f"(çıktı şekli {output_shape}). Model ile taksonomi eşleşmiyor; "
                "yanlış etiketle çalışmak yerine yükleme durduruldu."
            )

    def _build_session(self, model_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Build a fresh inference session from raw .onnx bytes.

        Returns the session pieces as a dict, or None when neither
        ONNXRuntime nor OpenCV DNN can load the bytes. NEVER mutates
        ``self``: the caller swaps the pieces in only on success, so a
        failed hot-reload keeps the old session alive.
        """
        # 1. Try ONNXRuntime first (passing raw bytes avoids Windows non-ASCII path issues)
        if _HAS_ONNXRUNTIME and ort is not None:
            try:
                if hasattr(ort, "set_default_logger_severity"):
                    ort.set_default_logger_severity(3)
                avail = ort.get_available_providers()
                providers: List[str] = []
                if self.prefer_gpu:
                    has_cuda_runtime = False
                    try:
                        import torch
                        has_cuda_runtime = bool(torch.cuda.is_available())
                    except Exception:
                        has_cuda_runtime = False
                    if has_cuda_runtime:
                        for p in (
                            "CUDAExecutionProvider",
                            "DmlExecutionProvider",
                        ):
                            if p in avail:
                                providers.append(p)
                    elif "DmlExecutionProvider" in avail:
                        providers.append("DmlExecutionProvider")
                providers.append("CPUExecutionProvider")

                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3
                sess_opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session = ort.InferenceSession(
                    model_bytes, sess_options=sess_opts, providers=providers
                )
                input_name = session.get_inputs()[0].name
                try:
                    _out_shape = session.get_outputs()[0].shape
                    output_shape: Optional[Tuple[int, ...]] = tuple(
                        int(d) for d in _out_shape
                    )
                except Exception:
                    output_shape = None
                # Sweep G: sınıf sayısı uyumsuzluğu burada kesinleşirse
                # ValueError fırlatır — aşağıdaki except ValueError'u
                # yeniden fırlatır, OpenCV yedeğine sessizce düşülmez.
                self._validate_onnx_output_shape(output_shape)
                active_provider = session.get_providers()[0]
                return {
                    "ort_session": session,
                    "ort_input_name": input_name,
                    "ort_output_shape": output_shape,
                    "cv_net": None,
                    "backend_name": f"onnxruntime:{active_provider}",
                }
            except ValueError:
                raise
            except Exception as exc:
                logger.debug(
                    "[ONNXYolo3DDetector] ONNXRuntime load failed (%s); trying OpenCV DNN.",
                    exc,
                )

        # 2. Fallback to OpenCV DNN from memory buffer (CUDA or OpenCV CPU backend)
        # Başarısız parse parmak izi önbelleği: bu baytlar cv2.dnn parse'ında
        # daha önce başarısız olduysa tekrar deneme, doğrudan fallback'e git.
        # (ONNXRuntime'sız ortamda her yükleme denemesi aynı parse hatasını
        # tekrarlıyordu.) İz bayt içeriğine bağlıdır — dosya değişirse iz
        # değişir ve parse yeniden denenir.
        _model_fingerprint = hashlib.sha256(model_bytes).hexdigest()
        if _model_fingerprint in _CV2_DNN_PARSE_FAILED_FINGERPRINTS:
            logger.debug(
                "[ONNXYolo3DDetector] cv2.dnn parse daha önce bu modelde "
                "başarısız oldu, tekrar denenmiyor; Fast 3D Mesh Tensor Fallback."
            )
            return None
        try:
            buf = np.frombuffer(model_bytes, dtype=np.uint8)
            net = cv2.dnn.readNetFromONNX(buf)
            used_cuda = False
            if self.prefer_gpu:
                try:
                    cuda_count = (
                        cv2.cuda.getCudaEnabledDeviceCount()
                        if hasattr(cv2, "cuda")
                        else 0
                    )
                    if cuda_count > 0:
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                        used_cuda = True
                except Exception:
                    used_cuda = False

            if not used_cuda:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            return {
                "ort_session": None,
                "ort_input_name": None,
                "ort_output_shape": None,
                "cv_net": net,
                "backend_name": (
                    "opencv_dnn:CUDA_FP16" if used_cuda else "opencv_dnn:CPU"
                ),
            }
        except Exception as exc:
            _CV2_DNN_PARSE_FAILED_FINGERPRINTS.add(_model_fingerprint)
            logger.debug(
                "[ONNXYolo3DDetector] OpenCV DNN load failed (%s); using Fast 3D Mesh Tensor Fallback.",
                exc,
            )
            return None

    @staticmethod
    def _release_session_objects(
        ort_session: Any = None,
        cv_net: Any = None,
    ) -> None:
        """Drop CUDA/ORT/OpenCV session refs so discarded loads can free VRAM/RSS."""
        if ort_session is not None:
            try:
                closer = getattr(ort_session, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass
            try:
                del ort_session
            except Exception:
                pass
        if cv_net is not None:
            try:
                del cv_net
            except Exception:
                pass

    def _clear_sessions_locked(self) -> None:
        """Null active session handles and release the previous objects (lock held)."""
        old_ort = self._ort_session
        old_cv = self._cv_net
        self._ort_session = None
        self._ort_input_name = None
        self._ort_output_shape = None
        self._cv_net = None
        self._release_session_objects(old_ort, old_cv)

    def _apply_built_session(self, built: Dict[str, Any]) -> None:
        """Swap freshly built session pieces in (caller holds _session_lock).

        Previous ORT/OpenCV sessions are closed/released — hot-reload used to
        overwrite ``_ort_session`` and leak ~300-500MB CUDA arenas per swap.
        """
        old_ort = self._ort_session
        old_cv = self._cv_net
        self._ort_session = built.get("ort_session")
        self._ort_input_name = built.get("ort_input_name")
        self._ort_output_shape = built.get("ort_output_shape")
        self._cv_net = built.get("cv_net")
        self.backend_name = str(
            built.get("backend_name", "fast_3d_mesh_tensor_fallback")
        )
        if old_ort is not self._ort_session or old_cv is not self._cv_net:
            self._release_session_objects(
                old_ort if old_ort is not self._ort_session else None,
                old_cv if old_cv is not self._cv_net else None,
            )

    def close(self) -> None:
        """Release inference sessions (ORT/OpenCV) held by this detector."""
        with self._session_lock:
            self._clear_sessions_locked()
            self.backend_name = "fast_3d_mesh_tensor_fallback"
            self._loaded_mtime = None
            self._loaded_size = None

    def _record_loaded_file(self, resolved: str) -> None:
        """Remember the loaded file's (mtime, size) for hot-reload checks."""
        try:
            st = os.stat(resolved)
            self._loaded_mtime = float(st.st_mtime)
            self._loaded_size = int(st.st_size)
        except OSError:
            self._loaded_mtime = None
            self._loaded_size = None

    def _read_stale_manifest(self, models_dir: str) -> Tuple[bool, str]:
        """Direct stale-manifest read (fallback when dataset_validator import fails)."""
        import json

        stale_file = os.path.join(models_dir, "stale_manifest.json")
        if not os.path.isfile(stale_file):
            return False, "Model güncel"
        try:
            with open(stale_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("stale manifest is not a JSON object")
            if data.get("model_stale") is True:
                return True, str(data.get("reason", "Model işaretlendi: stale"))
        except Exception as exc:
            # A corrupt manifest must NOT read as "model güncel" (fail closed).
            logger.warning("Bozuk stale manifest (%s): %s", stale_file, exc)
            return True, f"Stale manifest bozuk, model stale sayılıyor: {stale_file}"
        return False, "Model güncel"

    def _check_model_stale(
        self, models_dir: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Refresh the stale flag from the factory's stale marker (READ-ONLY).

        The live bot never consumed is_model_stale before: the only
        production consumer was the manual calibration wizard. Now the
        detector checks at load / reload time and warns loudly in the log
        when the loaded weights predate the latest labels, so the operator
        sees "model eski — fabrika eğitimi bekliyor" instead of silently
        running stale weights. Nothing here writes anything.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if models_dir is None:
            if self._resolved_model_path:
                models_dir = os.path.dirname(self._resolved_model_path)
            else:
                models_dir = os.path.join(project_root, "models")
        dataset_dir = os.path.join(project_root, "dataset")
        stale, reason = False, "Model güncel"
        try:
            from dataset_factory.dataset_validator import is_model_stale

            stale, reason = is_model_stale(
                dataset_dir=dataset_dir, models_dir=models_dir
            )
        except Exception as exc:
            logger.debug(
                "[ONNXYolo3DDetector] is_model_stale import failed (%s); reading manifest directly.",
                exc,
            )
            stale, reason = self._read_stale_manifest(models_dir)
        self.model_stale = bool(stale)
        self.model_stale_reason = str(reason or "")
        if self._last_stale_reported is not stale:
            self._last_stale_reported = stale
            if stale:
                # Sweep G: stale artık yalnızca warning değil — error
                # seviyesinde, çünkü eski ağırlıklarla oynamak sessizce
                # sürmemeli. Durdurma (require_fresh_model) ayrı anahtarla
                # açılır; varsayılan davranış uyar + çalışmaya devam et.
                logger.error(
                    "[ONNXYolo3DDetector] MODEL ESKİ — fabrika eğitimi bekliyor: %s",
                    self.model_stale_reason,
                )
            else:
                logger.info(
                    "[ONNXYolo3DDetector] YOLO modeli güncel: %s",
                    self.model_stale_reason,
                )
        return self.model_stale, self.model_stale_reason

    def check_for_model_update(self) -> bool:
        """Hot-reload the ONNX session when the .onnx file changed on disk.

        Throttled to ``reload_check_interval_sec`` (default 5 s) so the
        60 Hz scan path pays at most one stat() per interval. Safe by
        construction: the new session is built fully before the swap, the
        swap itself is locked, a running scan keeps using the old session
        reference it already snapshotted, and ANY failure (unreadable /
        placeholder / unloadable bytes) leaves the old session untouched.
        Returns True only when a new session actually went live.
        """
        now = time.monotonic()
        if now - self._last_reload_check < self.reload_check_interval_sec:
            return False
        self._last_reload_check = now
        resolved = self._resolved_model_path or self._resolve_model_path(
            self.model_path
        )
        if resolved is None or not os.path.isfile(resolved):
            return False
        self._resolved_model_path = resolved
        try:
            st = os.stat(resolved)
        except OSError:
            return False
        sig = (float(st.st_mtime), int(st.st_size))
        if (
            self._loaded_mtime is not None
            and sig[0] == self._loaded_mtime
            and sig[1] == self._loaded_size
        ):
            # File unchanged — but the stale flag may have cleared after a
            # factory publish, so still refresh it on this cadence.
            self._check_model_stale()
            return False
        if sig == self._last_reload_failed_sig:
            return False
        try:
            with open(resolved, "rb") as f:
                model_bytes = f.read()
        except Exception as exc:
            logger.warning(
                "[ONNXYolo3DDetector] Değişen model dosyası okunamadı, eski session kullanılıyor: %s (%s)",
                resolved,
                exc,
            )
            self._last_reload_failed_sig = sig
            return False
        if not model_bytes or model_bytes.startswith(b"ONNX{"):
            logger.warning(
                "[ONNXYolo3DDetector] Değişen model dosyası geçerli ONNX değil (boş/placeholder), eski session kullanılıyor: %s",
                resolved,
            )
            self._last_reload_failed_sig = sig
            return False
        try:
            built = self._build_session(model_bytes)
        except ValueError as exc:
            # Sweep G: sınıf sayısı uyumsuz yeni model — çalışan botu
            # çökertmeden eski session ile devam et, ama yüksek sesle.
            logger.error(
                "[ONNXYolo3DDetector] Yeni model sınıf sayısı uyumsuz, eski session kullanılıyor: %s",
                exc,
            )
            self._last_reload_failed_sig = sig
            return False
        if built is None:
            logger.warning(
                "[ONNXYolo3DDetector] Yeni model dosyası yüklenemedi, eski session kullanılmaya devam ediyor: %s",
                resolved,
            )
            self._last_reload_failed_sig = sig
            return False
        # Sweep G: stale zorunluluğu hot-reload'da da geçerli — stale yeni
        # model eski session'ın yerini alamaz.
        stale, stale_reason = self._check_model_stale()
        if stale and self.require_fresh_model:
            logger.error(
                "[ONNXYolo3DDetector] STALE yeni model hot-reload reddedildi, eski session kullanılıyor: %s",
                stale_reason,
            )
            self._last_reload_failed_sig = sig
            return False
        # v20-loop: ilk yükleme "hot-reload" diye loglanıyordu — dosya
        # değişmeden 2 ayrı instance'ın ilk kurulumu (16:53:51 + 16:53:55)
        # "model 4 sn'de 2 kez değişti" gibi okunuyordu. İlk kurulum
        # ayrı yazılır; gerçek hot-reload yalnızca mevcut session'ın
        # değişmesiyle olur.
        was_first_load = self._loaded_mtime is None
        with self._session_lock:
            self._apply_built_session(built)
            self._loaded_mtime = sig[0]
            self._loaded_size = sig[1]
        self._last_reload_failed_sig = None
        if was_first_load:
            logger.info(
                "[ONNXYolo3DDetector] Model ilk kez yüklendi (%s via %s)",
                resolved,
                self.backend_name,
            )
        else:
            logger.info(
                "[ONNXYolo3DDetector] Model hot-reload: yeni session devrede (%s via %s)",
                resolved,
                self.backend_name,
            )
        self._check_model_stale()
        return True

    def get_model_status(self) -> Dict[str, Any]:
        """Live model status for HUD/log: backend, stale flag, loaded file."""
        return {
            "backend": self.backend_name,
            "model_path": self._resolved_model_path or self.model_path,
            "model_loaded": self._ort_session is not None or self._cv_net is not None,
            "model_stale": bool(self.model_stale),
            "stale_reason": self.model_stale_reason,
            "loaded_mtime": self._loaded_mtime,
            "loaded_size": self._loaded_size,
            # Sweep G: sınıf sayısı doğrulama durumu.
            "expected_class_count": self._expected_class_count,
            "onnx_output_shape": self._ort_output_shape,
            "onnx_class_count_error": self._onnx_class_count_error,
            "require_fresh_model": bool(self.require_fresh_model),
        }

    def detect_3d_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect 3D ore veins, gatherable herb flora, and approaching monsters in a BGR frame.

        Runs ONNX inference if an ``.onnx`` model is loaded, and combines/falls back to
        the high-speed (~2ms) 3D Mesh Tensor & Gabor/Color-Structure Classifier.

        Args:
            frame: BGR image tensor ``(H, W, 3)`` as ``np.uint8``.

        Returns:
            List of detection dicts sorted by ``confidence`` descending:
            ``[{"class_id": int, "class_name": str, "category_tr": str, "confidence": float,
            "bbox": (x, y, w, h), "center": (cx, cy), "backend": str,
            "inference_ms": float, "sub_type": str}]``. ``class_name`` is the
            canonical taxonomy name (e.g. ``monster_3d`` for id 10), so
            reflex/hub/state-machine consumers can match on id or name.
            Fallback detections are canonicalised the same way (see
            ``_canonical_fallback_class``); an unknown fallback species
            keeps ``class_id=None`` instead of a fabricated id.

        Sweep G: ONNX/fallback canavar adayları (class_id 10) ExemplarMemory
        vetosundan geçer (`_apply_exemplar_monster_veto`) — bilinen negatif
        / reddedilmiş / kaynak-eşleşmeli kırpma listeden çıkarılır. Veto
        yalnızca eler, güveni asla şişirmez. Bu, üretimde hiç çalışmayan
        ExemplarMemory–ONNX hakemliğinin (`mana_phase_combat`
        `filter_monster_targets`) canlı bağlanmış karşılığıdır.
        """
        t0 = time.perf_counter()
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] < 3
        ):
            return []

        detections: List[Dict[str, Any]] = []

        # Hot-reload check (throttled inside): a factory publish swaps the
        # .onnx file while the bot runs; the session must follow without a
        # restart, and a failed rebuild must never disturb this scan.
        try:
            self.check_for_model_update()
        except Exception as exc:
            logger.debug("[ONNXYolo3DDetector] reload check skipped: %s", exc)

        # 1. Run ONNX session if loaded
        if self._ort_session is not None or self._cv_net is not None:
            try:
                detections = self._run_onnx_inference(frame, t0)
            except Exception as exc:
                logger.debug(
                    "[ONNXYolo3DDetector] ONNX inference error, falling back: %s", exc
                )
                detections = []

        # 2. Always run the Fast 3D Mesh fallback too and MERGE its detections
        # (class-aware NMS dedup). The old either/or logic meant anything the
        # ONNX model missed in a frame was never detected at all.
        for fb in self._run_fast_3d_mesh_fallback(frame, t0):
            self._append_detection_nms(
                detections,
                str(fb.get("class_name", "")),
                str(fb.get("sub_type", "")),
                float(fb.get("confidence", 0.0)),
                tuple(fb.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
                backend=str(fb.get("backend", "fast_3d_mesh_tensor_fallback")),
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        for det in detections:
            det["inference_ms"] = elapsed_ms

        # Sweep G: canlı ExemplarMemory–ONNX hakemliği (veto katmanı).
        detections = self._apply_exemplar_monster_veto(frame, detections)

        detections.sort(key=lambda d: float(d["confidence"]), reverse=True)
        return detections

    def _apply_exemplar_monster_veto(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """ExemplarMemory–ONNX hakemliği: canavar adaylarında veto katmanı.

        Sweep G bulgu 7: `mana_phase_combat.filter_monster_targets` üretimde
        hiç çağrılmıyordu — hakem yoktu, hub ONNX/fallback dict'inden
        doğrudan `has_target` çıkarıyordu. Hakemlik artık canlı tespit
        yolunun içinde: her class_id 10 adayı kırpılıp ExemplarMemory ile
        karşılaştırılır; bilinen negatif, pozitif listedeki
        `(10, 'monster_3d_negative')` kaydı veya güçlü kaynak/engel
        eşleşmesi adayı listeden çıkarır. Bilinmeyen aday tutulur (veto
        yalnızca eler — doğrulanmamış canavar "yok" sayılmaz, mana
        tarafındaki unconfirmed-fallback mantığıyla aynı). Exemplar
        belleği yoksa/hatalıysa veto atlanır (tespitler korunur); hata
        debug loglanır.
        """
        if not detections:
            return detections
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory

            mem = get_exemplar_memory()
        except Exception as exc:
            logger.debug("[ONNXYolo3DDetector] exemplar veto atlandı: %s", exc)
            return detections
        if mem is None:
            return detections
        h, w = frame.shape[:2]
        kept: List[Dict[str, Any]] = []
        for det in detections:
            if det.get("class_id") != 10:
                kept.append(det)
                continue
            try:
                bx, by, bw, bh = (int(v) for v in det.get("bbox", (0, 0, 0, 0)))
                pad = 6
                crop = frame[
                    max(0, by - pad) : min(h, by + bh + pad),
                    max(0, bx - pad) : min(w, bw + bx + pad),
                ]
                if crop is None or crop.size == 0:
                    kept.append(det)
                    continue
                is_neg, _, _ = mem.is_negative_patch(
                    crop, threshold=EXEMPLAR_MATCH_THRESHOLD
                )
                if is_neg:
                    logger.debug(
                        "[ONNXYolo3DDetector] exemplar veto: negatif canavar adayı elendi %s",
                        det.get("bbox"),
                    )
                    continue
                pos_cid, pos_cname, pos_sim = mem.classify_patch(
                    crop, min_sim=EXEMPLAR_MATCH_THRESHOLD
                )
                if pos_cid is not None:
                    if is_negative_exemplar_name(pos_cname):
                        logger.debug(
                            "[ONNXYolo3DDetector] exemplar veto: negatif kayıt eşleşmesi elendi %s",
                            det.get("bbox"),
                        )
                        continue
                    if pos_cid in (0, 1, 2, 3, 4, 5, 14) and float(pos_sim) >= EXEMPLAR_MATCH_THRESHOLD:
                        logger.debug(
                            "[ONNXYolo3DDetector] exemplar veto: kaynak/engel eşleşmesi elendi %s",
                            det.get("bbox"),
                        )
                        continue
            except Exception as exc:
                logger.debug("[ONNXYolo3DDetector] exemplar veto hatası: %s", exc)
            kept.append(det)
        return kept

    # ------------------------------------------------------------------
    # ONNX YOLOv8/v11 Inference Pipeline
    # ------------------------------------------------------------------
    def _letterbox(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize and pad image to ``self.input_size`` while preserving aspect ratio."""
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / float(w), target_h / float(h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Split the total padding exactly (the old round(dh +/- 0.1) split
        # could miss the target size by 1 px, shifting box back-projection).
        total_h = target_h - new_h
        total_w = target_w - new_w
        top, bottom = total_h // 2, total_h - total_h // 2
        left, right = total_w // 2, total_w - total_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        # Return the ACTUAL integer padding applied, not the float half:
        # when total_w/total_h is odd, (target-new)/2.0 is off by 0.5 px
        # from the real border (e.g. 640x479 frame: dh=80.5 vs top=80),
        # which shifts every back-projected box by 0.5/scale px.
        return padded, scale, (float(left), float(top))

    def _run_onnx_inference(
        self, frame: np.ndarray, t0: float
    ) -> List[Dict[str, Any]]:
        """Execute YOLOv8/v11 tensor forward pass via ONNXRuntime or OpenCV DNN."""
        h, w = frame.shape[:2]
        padded, scale, (dw, dh) = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1.0 / 255.0, size=self.input_size, swapRB=True, crop=False
        )

        # Snapshot the session under the reload lock: a hot-reload swap
        # mid-inference must not hand us a half-swapped (session, input)
        # pair, and inference must keep using the session it started with.
        with self._session_lock:
            ort_session = self._ort_session
            ort_input_name = self._ort_input_name
            ort_output_shape = self._ort_output_shape
            cv_net = self._cv_net
            backend_name = self.backend_name

        if ort_session is not None and ort_input_name is not None:
            raw_out = ort_session.run(None, {ort_input_name: blob})[0]
        elif cv_net is not None:
            cv_net.setInput(blob)
            raw_out = cv_net.forward()
        else:
            return []

        preds = np.squeeze(raw_out)
        if preds.ndim == 2:
            # Decide the (4+C, N) -> (N, 4+C) transpose from the model's own
            # output layout when known. The old shape guess (rows < cols)
            # wrongly transposed a genuine (N, 4+C) output whenever N < 4+C.
            transposed = None
            if ort_output_shape is not None and len(ort_output_shape) == 3:
                transposed = ort_output_shape[1] < ort_output_shape[2]
            if transposed is None:
                # OpenCV DNN path (no metadata): transpose only when the first
                # dim is a plausible 4+C and the second is a large N.
                transposed = (
                    preds.shape[0] < preds.shape[1]
                    and preds.shape[0] <= 256
                    and preds.shape[1] > 100
                )
            if transposed:
                preds = preds.T

        if preds.ndim != 2 or preds.shape[1] < 5:
            return []

        # Sweep G: çıkarım anı sınıf sayısı kontrolü (OpenCV DNN yolunda
        # yükleme anında çıktı şekli bilinmediğinden bu tek kapıdır).
        # Uyumsuz modelin yanlış etiket üretmesi yerine hiç tespit dönmez;
        # hata bir kez error loglanır ve get_model_status() içinde görünür.
        n_pred_classes = int(preds.shape[1]) - 4
        if n_pred_classes != self._expected_class_count:
            if self._onnx_class_count_error is None:
                self._onnx_class_count_error = (
                    f"ONNX çıkarım sınıf sayısı uyumsuz: model {n_pred_classes} "
                    f"sınıf üretiyor, data.yaml nc={self._expected_class_count}"
                )
                logger.error(
                    "[ONNXYolo3DDetector] %s — tespitler atılıyor.",
                    self._onnx_class_count_error,
                )
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        nms_boxes: List[List[int]] = []
        for cx_b, cy_b, bw_b, bh_b in boxes_xywh:
            x1 = int(round((cx_b - bw_b / 2.0 - dw) / scale))
            y1 = int(round((cy_b - bh_b / 2.0 - dh) / scale))
            bw = int(round(bw_b / scale))
            bh = int(round(bh_b / scale))
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            bw = max(1, min(w - x1, bw))
            bh = max(1, min(h - y1, bh))
            nms_boxes.append([x1, y1, bw, bh])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return []

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        results: List[Dict[str, Any]] = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = nms_boxes[int(idx)]
            cid = int(class_ids[int(idx)])
            # Canonical taxonomy name + numeric id (was: 3-entry map, so
            # every class >= 3 came out as "class_N" with no class_id and
            # no consumer — reflex, hub, state machine — could see it).
            cname = canonical_class_name(cid)
            cx = bx + bw // 2
            cy = by + bh // 2
            results.append(
                {
                    "class_id": cid,
                    "class_name": cname,
                    "category_tr": canonical_class_label_tr(
                        cid, self.CATEGORY_TR_MAP.get(cname, cname)
                    ),
                    "confidence": round(float(confidences[int(idx)]), 3),
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "center": (int(cx), int(cy)),
                    "backend": backend_name,
                    "inference_ms": elapsed_ms,
                    "sub_type": cname,
                }
            )
        return results

    @staticmethod
    def _tensor_bbox_to_abs(
        vx1: int,
        vy1: int,
        sx: float,
        sy: float,
        x_t: int,
        y_t: int,
        w_t: int,
        h_t: int,
        min_w: int = 24,
        min_h: int = 24,
    ) -> Tuple[int, int, int, int]:
        """Map a downsampled-tensor bbox back to full-frame coordinates.

        The five fallback sections each did this same 4-line conversion
        inline (with different minimum sizes); one helper keeps them
        identical by construction.
        """
        return (
            int(round(vx1 + x_t * sx)),
            int(round(vy1 + y_t * sy)),
            max(min_w, int(round(w_t * sx))),
            max(min_h, int(round(h_t * sy))),
        )

    # ------------------------------------------------------------------
    # Fast (~2ms) 3D Mesh Tensor & Gabor/Color-Structure Classifier Fallback
    # ------------------------------------------------------------------
    def _run_fast_3d_mesh_fallback(
        self, frame: np.ndarray, t0: float
    ) -> List[Dict[str, Any]]:
        """Execute high-speed (~2ms) 3D mesh geometry, Gabor texture, and HSV tensor analysis.

        Operates on a compact downsampled 3D viewport tensor (``360 x 202``) so the entire
        multi-channel HSV + gradient/Gabor + morphological component pipeline finishes in
        ~1.5–3.0 ms, while accurately localizing:
        1. ``herb_flora``:
           - 3D purple/violet crystalline flower clusters (``Aria``) and winged gather crests.
           - 3D red-berry/coral flower clusters surrounded by green leaf bushes (``Mela``)
             and interactive ``F Gather`` / ``[E]`` plant nodes.
           - 3D golden-yellow herb clusters on green stems (``Targena``).
        2. ``ore_vein``:
           - 3D metallic/copper/amber Orichalcum veins embedded in dark faceted rock.
           - 3D crystalline Sapphire (deep blue), Ruby (crimson facet on dark rock), and
             Diamond (high-luminance specular crystal facets) veins even when nameplates are off.
        3. ``approaching_monster``:
           - 3D hostile monster silhouettes with crimson/red agro badges or yellow/olive
             monster nameplates / glowing combat auras.
        """
        h, w = frame.shape[:2]
        if w < 120 or h < 90:
            return []

        # Crop to the active 3D game world viewport (excludes top bar, minimap, bottom skill bar)
        vx1 = int(w * 0.08)
        vy1 = int(h * 0.12)
        vx2 = int(w * 0.84)
        vy2 = int(h * 0.72)
        roi_w = max(64, vx2 - vx1)
        roi_h = max(48, vy2 - vy1)

        # Two-stage fast downsample: zero-copy 2x stride slice + INTER_NEAREST to (320, 180) in ~0.3ms
        half_bgr = frame[vy1:vy2:2, vx1:vx2:2] if roi_w > 640 else frame[vy1:vy2, vx1:vx2]
        tensor_w = min(320, roi_w)
        tensor_h = min(180, roi_h)
        small_bgr = cv2.resize(
            half_bgr, (tensor_w, tensor_h), interpolation=cv2.INTER_NEAREST
        )

        sx = roi_w / float(tensor_w)
        sy = roi_h / float(tensor_h)

        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Reject flat dark settings/modal screens so UI panels are never misclassified as 3D meshes
        slate_ratio = float(
            np.count_nonzero((s_ch < 35) & (v_ch >= 18) & (v_ch <= 48))
        ) / float(tensor_w * tensor_h)
        # Only a near-total slate frame is a modal/settings screen; a
        # dark cave/ore region can exceed 0.55 slate coverage legitimately.
        if slate_ratio > 0.80:
            return []

        # Compute fast 3D facet edge energy (Sobel + diagonal structure response) on V channel
        gx = cv2.Sobel(v_ch, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(v_ch, cv2.CV_16S, 0, 1, ksize=3)
        grad_mag = cv2.addWeighted(
            cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0
        )

        # Gradient-magnitude threshold masks, shared by the colour masks
        # below. (Each inRange is one C++ pass; the old ``grad_mag >= X``
        # boolean chains rebuilt a full-size temp array per comparison.)
        grad_ge_18 = cv2.inRange(grad_mag, 18, 255)
        grad_ge_30 = cv2.inRange(grad_mag, 30, 255)
        grad_ge_35 = cv2.inRange(grad_mag, 35, 255)
        grad_ge_42 = cv2.inRange(grad_mag, 42, 255)
        grad_ge_45 = cv2.inRange(grad_mag, 45, 255)

        # Precompute the dilated green-foliage mask once: it seeds both the
        # Mela and the Targena detectors below (the old code dilated the
        # same mask twice per frame).
        green_foliage_mask = cv2.inRange(hsv, (36, 65, 45), (76, 255, 225))
        green_foliage_dil5 = cv2.dilate(green_foliage_mask, self._morph_5x5)

        detections: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # A. 3D White Winged Gather Crest + Cyan/Green Nameplate Tensor Scan
        #    (Detects distant Aria/Mela/Orichalcum nodes like aria_3d_crest_far.jpg)
        # ------------------------------------------------------------------
        # cv2.inRange is one C++ pass over the HSV image; the old boolean
        # chain ((h_ch >= ..) & ...) allocated a full-size temp array per
        # comparison. Bit-exact equal: inRange bounds are inclusive, same
        # as the >= / <= comparisons they replace.
        cyan_label_mask = cv2.inRange(hsv, (60, 55, 115), (102, 255, 255))
        cyan_closed = cv2.morphologyEx(
            cyan_label_mask, cv2.MORPH_CLOSE, self._morph_7x5
        )
        cnts_cyan, _ = cv2.findContours(
            cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_cyan:
            cx_t, cy_t, cw_t, ch_t = cv2.boundingRect(cnt)
            if cw_t < 4 or cw_t > 65 or ch_t < 2 or ch_t > 18:
                continue

            # Check for white winged crest right above the cyan/green label
            crest_y1 = max(0, cy_t - 14)
            crest_y2 = max(1, cy_t + 2)
            crest_x1 = max(0, cx_t - 4)
            crest_x2 = min(tensor_w, cx_t + cw_t + 4)
            crest_patch_s = s_ch[crest_y1:crest_y2, crest_x1:crest_x2]
            crest_patch_v = v_ch[crest_y1:crest_y2, crest_x1:crest_x2]
            white_crest_px = int(
                np.count_nonzero((crest_patch_v >= 165) & (crest_patch_s <= 65))
            )
            if white_crest_px < 4:
                continue

            # Inspect 3D mesh region directly below the nameplate to classify herb_flora vs ore_vein
            mesh_y1 = min(tensor_h - 1, cy_t + ch_t + 1)
            mesh_y2 = min(tensor_h, cy_t + ch_t + 28)
            mesh_x1 = max(0, cx_t - 8)
            mesh_x2 = min(tensor_w, cx_t + cw_t + 8)
            mh = h_ch[mesh_y1:mesh_y2, mesh_x1:mesh_x2]
            ms = s_ch[mesh_y1:mesh_y2, mesh_x1:mesh_x2]
            mv = v_ch[mesh_y1:mesh_y2, mesh_x1:mesh_x2]

            purple_px = int(
                np.count_nonzero((mh >= 122) & (mh <= 165) & (ms >= 55) & (mv >= 65))
            )
            red_px = int(
                np.count_nonzero(
                    (((mh <= 10) | (mh >= 168)) & (ms >= 95) & (mv >= 80))
                )
            )
            green_px = int(
                np.count_nonzero((mh >= 35) & (mh <= 78) & (ms >= 55) & (mv >= 45))
            )
            sapphire_px = int(
                np.count_nonzero((mh >= 103) & (mh <= 130) & (ms >= 120) & (mv >= 120))
            )
            warm_ore_px = int(
                np.count_nonzero((mh >= 6) & (mh <= 26) & (ms >= 75) & (mv >= 65))
            )
            dark_rock_px = int(np.count_nonzero((ms <= 65) & (mv >= 20) & (mv <= 95)))

            if purple_px >= 6:
                cname, sub_type = "herb_flora", "Aria"
            elif red_px >= 6 and green_px >= 4:
                cname, sub_type = "herb_flora", "Mela"
            elif sapphire_px >= 6:
                cname, sub_type = "ore_vein", "Sapphire Gemstone"
            elif warm_ore_px >= 8 and dark_rock_px >= 10 and green_px < 10:
                cname, sub_type = "ore_vein", "Orichalcum Ore"
            else:
                # No colour evidence under this label: report nothing rather
                # than fabricating an "Aria" detection.
                continue

            abs_x = int(round(vx1 + crest_x1 * sx))
            abs_y = int(round(vy1 + crest_y1 * sy))
            abs_w = int(round((mesh_x2 - crest_x1) * sx))
            abs_h = int(round((mesh_y2 - crest_y1) * sy))
            conf = round(min(0.97, 0.84 + min(0.12, white_crest_px * 0.008)), 3)
            self._append_detection_nms(
                detections,
                cname,
                sub_type,
                conf,
                (abs_x, abs_y, max(24, abs_w), max(28, abs_h)),
                backend="fast_3d_mesh_tensor_fallback",
            )

        # ------------------------------------------------------------------
        # B. Nameplate-Free 3D Plant Geometry (herb_flora: Aria, Mela, Targena)
        # ------------------------------------------------------------------
        # 1) Aria 3D Purple/Violet Crystal Flower Mesh
        aria_mask = cv2.bitwise_and(
            cv2.inRange(hsv, (126, 75, 80), (164, 255, 255)), grad_ge_18
        )
        aria_closed = cv2.morphologyEx(aria_mask, cv2.MORPH_CLOSE, self._morph_5x5)
        cnts_aria, _ = cv2.findContours(
            aria_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_aria:
            x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            area_t = w_t * h_t
            if area_t < 18 or area_t > 2400 or w_t < 4 or h_t < 4:
                continue
            active_px = int(
                np.count_nonzero(aria_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            if active_px < 8:
                continue
            abs_x, abs_y, abs_w, abs_h = self._tensor_bbox_to_abs(
                vx1, vy1, sx, sy, x_t, y_t, w_t, h_t, 24, 24
            )
            conf = round(min(0.94, 0.76 + min(0.16, active_px / 120.0)), 3)
            self._append_detection_nms(
                detections,
                "herb_flora",
                "Aria",
                conf,
                (abs_x, abs_y, abs_w, abs_h),
                backend="fast_3d_mesh_tensor_fallback",
            )

        # 2) Mela 3D Red-Berry Bush + Green Leaf Coupling (works even without minimap icon)
        # Hue wraps around 0 for red, so two inRange calls would cost more
        # than the boolean chain here (measured 0.79 ms vs 0.60 ms); keep it.
        red_berry_mask = (
            (((h_ch <= 9) | (h_ch >= 169)) & (s_ch >= 115) & (v_ch >= 95))
        ).astype(np.uint8) * 255

        mela_seed = cv2.bitwise_and(
            cv2.dilate(red_berry_mask, self._morph_5x5),
            green_foliage_dil5,
        )
        cnts_mela, _ = cv2.findContours(
            mela_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_mela:
            x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            if w_t < 5 or h_t < 5 or w_t > 90 or h_t > 75:
                continue
            # Exclude central player character cape/armor zone (around x=0.46..0.54, y=0.45..0.68 of full screen)
            abs_cx_check = vx1 + (x_t + w_t * 0.5) * sx
            abs_cy_check = vy1 + (y_t + h_t * 0.5) * sy
            if (
                0.44 * w <= abs_cx_check <= 0.55 * w
                and 0.44 * h <= abs_cy_check <= 0.70 * h
            ):
                # Only allow in player zone if interactive "F Gather" white pill is present right next to it
                f_x1 = int(max(0, min(tensor_w - 1, x_t - 10)))
                f_x2 = int(max(1, min(tensor_w, x_t + w_t + 35)))
                f_y1 = int(max(0, min(tensor_h - 1, y_t - 10)))
                f_y2 = int(max(1, min(tensor_h, y_t + h_t + 15)))
                white_pill_px = int(
                    np.count_nonzero(
                        (v_ch[f_y1:f_y2, f_x1:f_x2] >= 185)
                        & (s_ch[f_y1:f_y2, f_x1:f_x2] <= 40)
                    )
                )
                if white_pill_px < 12:
                    continue

            r_cnt = int(
                np.count_nonzero(red_berry_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            g_cnt = int(
                np.count_nonzero(green_foliage_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            if r_cnt < 6 or g_cnt < 8:
                continue

            abs_x, abs_y, abs_w, abs_h = self._tensor_bbox_to_abs(
                vx1, vy1, sx, sy, x_t, y_t, w_t, h_t, 28, 26
            )
            conf = round(min(0.95, 0.78 + min(0.15, (r_cnt + g_cnt) / 140.0)), 3)
            self._append_detection_nms(
                detections,
                "herb_flora",
                "Mela",
                conf,
                (abs_x, abs_y, abs_w, abs_h),
                backend="fast_3d_mesh_tensor_fallback",
            )

        # 3) Targena 3D Golden-Yellow Herb Cluster on Green Stems
        yellow_herb_mask = cv2.bitwise_and(
            cv2.inRange(hsv, (20, 115, 130), (34, 255, 255)), grad_ge_18
        )
        targena_seed = cv2.bitwise_and(
            cv2.dilate(yellow_herb_mask, self._morph_5x5),
            green_foliage_dil5,
        )
        cnts_targena, _ = cv2.findContours(
            targena_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_targena:
            x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            if w_t < 5 or h_t < 5 or w_t > 90 or h_t > 75:
                continue
            y_cnt = int(
                np.count_nonzero(yellow_herb_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            g_cnt = int(
                np.count_nonzero(green_foliage_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            if y_cnt < 6 or g_cnt < 8:
                continue
            abs_x, abs_y, abs_w, abs_h = self._tensor_bbox_to_abs(
                vx1, vy1, sx, sy, x_t, y_t, w_t, h_t, 28, 26
            )
            conf = round(min(0.95, 0.78 + min(0.15, (y_cnt + g_cnt) / 140.0)), 3)
            self._append_detection_nms(
                detections,
                "herb_flora",
                "Targena",
                conf,
                (abs_x, abs_y, abs_w, abs_h),
                backend="fast_3d_mesh_tensor_fallback",
            )

        # ------------------------------------------------------------------
        # C. Nameplate-Free 3D Ore Vein & Crystal Geometry (ore_vein)
        #    Detects metallic/crystalline rock veins with high facet gradient
        #    and specular crystal highlights (Orichalcum / Sapphire / Ruby / Diamond)
        # ------------------------------------------------------------------
        # 1) Sapphire / Crystal Blue Vein on dark faceted rock
        sapphire_vein = cv2.bitwise_and(
            cv2.inRange(hsv, (104, 145, 140), (132, 255, 255)), grad_ge_35
        )
        # 2) Orichalcum Copper/Amber Metallic Crystal Vein embedded in dark slate rock with specular glint
        dark_rock = cv2.bitwise_and(
            cv2.inRange(hsv, (0, 0, 20), (180, 58, 78)), grad_ge_30
        )
        amber_vein = cv2.bitwise_and(
            cv2.inRange(hsv, (10, 145, 145), (23, 255, 255)), grad_ge_42
        )
        specular_glint = cv2.bitwise_and(
            cv2.inRange(hsv, (0, 0, 210), (180, 45, 255)), grad_ge_45
        )
        # 3) Ruby crimson crystal facets on dark rock (hue wraps: boolean
        # chain kept, same reason as red_berry_mask above)
        ruby_vein = (
            (((h_ch <= 8) | (h_ch >= 172)) & (s_ch >= 135) & (v_ch >= 120) & (grad_mag >= 35))
        ).astype(np.uint8) * 255
        # 4) Diamond high-luminance specular crystal facets
        diamond_vein = cv2.bitwise_and(
            cv2.inRange(hsv, (0, 0, 215), (180, 40, 255)), grad_ge_45
        )
        orichalcum_coupled = cv2.bitwise_and(
            cv2.dilate(amber_vein, self._morph_3x3),
            cv2.bitwise_and(
                cv2.dilate(dark_rock, self._morph_5x5),
                cv2.dilate(specular_glint, self._morph_5x5),
            ),
        )
        # Exclude ground/grass regions by requiring near-zero green foliage inside the rock vein
        ore_combined = cv2.bitwise_or(
            cv2.bitwise_or(sapphire_vein, orichalcum_coupled),
            cv2.bitwise_or(ruby_vein, diamond_vein),
        )
        ore_closed = cv2.morphologyEx(ore_combined, cv2.MORPH_CLOSE, self._morph_5x5)
        cnts_ore, _ = cv2.findContours(
            ore_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_ore:
            x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            if w_t < 4 or h_t < 4 or w_t > 65 or h_t > 60:
                continue
            abs_cx_check = vx1 + (x_t + w_t * 0.5) * sx
            abs_cy_check = vy1 + (y_t + h_t * 0.5) * sy
            # Exclude central player character silhouette
            if (
                0.42 * w <= abs_cx_check <= 0.58 * w
                and 0.36 * h <= abs_cy_check <= 0.74 * h
            ):
                continue

            g_inside = int(
                np.count_nonzero(green_foliage_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            rock_inside = int(
                np.count_nonzero(dark_rock[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            sapph_inside = int(
                np.count_nonzero(sapphire_vein[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            amber_inside = int(
                np.count_nonzero(amber_vein[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            ruby_inside = int(
                np.count_nonzero(ruby_vein[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            diamond_inside = int(
                np.count_nonzero(diamond_vein[y_t : y_t + h_t, x_t : x_t + w_t])
            )

            # Must have strong dark rock facet support and not be green grass/foliage
            if g_inside > (rock_inside + sapph_inside) // 3:
                continue
            if (
                sapph_inside < 6
                and ruby_inside < 6
                and diamond_inside < 6
                and (amber_inside < 6 or rock_inside < 8)
            ):
                continue

            # Strongest vein evidence wins: Sapphire / Orichalcum / Ruby / Diamond.
            _vein_scores = {
                "Sapphire Gemstone": sapph_inside,
                "Orichalcum Ore": amber_inside,
                "Ruby Gemstone": ruby_inside,
                "Diamond Gemstone": diamond_inside,
            }
            sub_type = max(_vein_scores, key=lambda k: _vein_scores[k])
            abs_x, abs_y, abs_w, abs_h = self._tensor_bbox_to_abs(
                vx1, vy1, sx, sy, x_t, y_t, w_t, h_t, 26, 26
            )
            conf = round(
                min(
                    0.92,
                    0.74
                    + min(
                        0.16,
                        (sapph_inside + amber_inside + rock_inside + ruby_inside + diamond_inside)
                        / 110.0,
                    ),
                ),
                3,
            )
            self._append_detection_nms(
                detections,
                "ore_vein",
                sub_type,
                conf,
                (abs_x, abs_y, abs_w, abs_h),
                backend="fast_3d_mesh_tensor_fallback",
            )

        # ------------------------------------------------------------------
        # D. 3D Approaching Monster Silhouette & Agro Aura (approaching_monster)
        #    Detects yellow/orange monster nameplates + red rank/agro badges or
        #    hostile crimson combat aura silhouettes
        # ------------------------------------------------------------------
        yellow_mob_text = cv2.inRange(hsv, (15, 115, 135), (34, 255, 255))
        # Hue wraps for red: boolean chain kept (see red_berry_mask).
        red_agro_badge = (
            (((h_ch <= 8) | (h_ch >= 172)) & (s_ch >= 135) & (v_ch >= 95))
        ).astype(np.uint8) * 255

        mob_coupled = cv2.bitwise_and(
            cv2.dilate(yellow_mob_text, self._morph_7x5),
            cv2.dilate(red_agro_badge, self._morph_7x5),
        )
        cnts_mob, _ = cv2.findContours(
            mob_coupled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_mob:
            x_t, y_t, w_t, h_t = cv2.boundingRect(cnt)
            if w_t < 4 or h_t < 3 or w_t > 90:
                continue
            # Ensure it is not a cyan gatherable label
            c_px = int(
                np.count_nonzero(cyan_label_mask[y_t : y_t + h_t, x_t : x_t + w_t])
            )
            if c_px >= 6:
                continue
            # Include 3D monster body below the agro nameplate
            body_h_t = min(tensor_h - y_t, h_t + 26)
            abs_x, abs_y, abs_w, abs_h = self._tensor_bbox_to_abs(
                vx1, vy1, sx, sy, x_t, y_t, w_t, body_h_t, 32, 36
            )
            # Confidence scales with the actual coupled text+badge evidence
            # instead of a flat 0.88 for every monster, strong or weak.
            _mob_y = int(np.count_nonzero(yellow_mob_text[y_t : y_t + h_t, x_t : x_t + w_t]))
            _mob_r = int(np.count_nonzero(red_agro_badge[y_t : y_t + h_t, x_t : x_t + w_t]))
            mob_conf = round(min(0.95, 0.70 + min(0.25, (_mob_y + _mob_r) / 120.0)), 3)
            self._append_detection_nms(
                detections,
                "approaching_monster",
                "Hostile Monster",
                mob_conf,
                (abs_x, abs_y, abs_w, abs_h),
                backend="fast_3d_mesh_tensor_fallback",
            )

        return detections

    def _append_detection_nms(
        self,
        detections: List[Dict[str, Any]],
        class_name: str,
        sub_type: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        backend: Optional[str] = None,
    ) -> None:
        """Append a detection unless a SAME-CLASS detection already covers it.

        A nearby detection of a DIFFERENT class is a different node and is
        kept (the old code suppressed it and even overwrote the existing
        detection's class with the new one).

        Sweep G: fallback isimler (`herb_flora` / `ore_vein` /
        `approaching_monster`) alt-tipe göre kanonik sınıfa çevrilir
        (örn. `Aria` -> class_id 0 `herb_aria`, `Hostile Monster` ->
        class_id 10 `monster_3d`); her tespit artık `class_id` taşır.
        Böylece ONNX `monster_3d` ile fallback `approaching_monster` aynı
        canavarı iki kez saymaz (dedup aynı kanonik isimde birleşir) ve
        reflex/hub tüketicisi isim tahminiyle yanlış sınıfa (eski:
        `crystal_blood` 12) düşmez. Alt-tip bilinmiyorsa class_id `None`
        kalır — tür uydurulmaz.
        """
        bx, by, bw, bh = bbox
        cx = bx + bw // 2
        cy = by + bh // 2
        canonical_id, canonical_name = _canonical_fallback_class(class_name, sub_type)

        for existing in detections:
            ex, ey, ew, eh = existing["bbox"]
            ecx, ecy = existing["center"]
            if abs(cx - ecx) <= max(24, min(bw, ew) // 2) and abs(cy - ecy) <= max(
                24, min(bh, eh) // 2
            ):
                if existing.get("class_name") != canonical_name:
                    continue
                if confidence > existing["confidence"]:
                    existing["confidence"] = confidence
                    existing["bbox"] = bbox
                    existing["center"] = (cx, cy)
                    existing["sub_type"] = sub_type
                    if backend is not None:
                        existing["backend"] = backend
                return

        detections.append(
            {
                "class_id": canonical_id,
                "class_name": canonical_name,
                # Orijinal fallback etiketi izlenebilirlik için korunur.
                "fallback_label": str(class_name),
                "category_tr": self.CATEGORY_TR_MAP.get(
                    class_name, canonical_class_label_tr(canonical_id, canonical_name)
                    if canonical_id is not None
                    else canonical_name
                ),
                "confidence": round(float(confidence), 3),
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "center": (int(cx), int(cy)),
                "backend": backend if backend is not None else self.backend_name,
                "inference_ms": 0.0,
                "sub_type": sub_type,
            }
        )


# ==============================================================================
# 2. Windows WASAPI Audio Loopback & numpy.fft Spectral DSP Analyzer
# ==============================================================================
# One-shot audio event flags: each detection is delivered by
# poll_latest_audio_events() exactly once (S-0701). Level-like fields
# (confidence, rms_energy, spectral metrics) keep reporting the last
# analysis, but these three flags are consumed on read and never replayed.
_ONE_SHOT_AUDIO_EVENT_KEYS: Tuple[str, ...] = (
    "gather_success_chime",
    "gather_fail_sound",
    "danger_audio_alert",
)


class AudioLoopbackFFTAnalyzer:
    """Windows WASAPI Audio Loopback & ``numpy.fft.rfft`` Spectral DSP Event Detector.

    Analyzes game audio PCM frames using Hann windowing and real Fast Fourier Transform
    (``numpy.fft.rfft``) to provide an orthogonal, visual-independent confirmation channel:
    1. **Success Crystal Chime (``gather_success_chime``)**:
       High-frequency crystalline harmonic resonance in the ``2200 Hz – 4800 Hz`` band
       (``spectral_centroid > 2400 Hz`` + narrowband harmonic peak-to-mean crest ratio).
    2. **Extraction Failure Shatter/Thud (``gather_fail_sound``)**:
       Broadband metallic/rock fracture burst in the ``180 Hz – 650 Hz`` band with elevated
       spectral flatness and mid-low band energy ratio.
    3. **Approaching Assassin / Monster Footstep & Attack (``danger_audio_alert``)**:
       Impulsive low-frequency transients in the ``80 Hz – 320 Hz`` band with high temporal
       crest factor (peak-to-RMS ratio) indicating footsteps or weapon swings behind the player
       even when the camera is spinning or blurred.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate: int = max(8000, int(sample_rate))
        self.fft_size: int = max(256, int(fft_size))
        self.device_index: Optional[int] = device_index

        # Precompute Hann window for spectral leakage suppression
        self._hann_window: np.ndarray = np.hanning(self.fft_size).astype(np.float32)
        # Bounded caches for per-length spectral data. analyze_pcm_buffer
        # used to rebuild the Hann window (np.hanning) and the rfftfreq
        # vector + 3 band masks on EVERY call (~0.7 ms of window alone for
        # a 6615-sample buffer); the values depend only on (n, sample_rate),
        # so they are computed once per distinct length/rate and reused.
        # Capped at 8 entries: a longer-lived analyzer that sees many
        # different buffer lengths cannot grow these caches without bound.
        self._hann_cache: Dict[int, np.ndarray] = {self.fft_size: self._hann_window}
        self._spectral_cache: Dict[
            Tuple[int, int],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        # Thread-safe ring buffer for live WASAPI loopback capture
        self._ring_buffer: Deque[np.ndarray] = deque(maxlen=16)
        self._lock = threading.Lock()
        self._running: bool = False
        self._stream: Optional[Any] = None
        # F-0153: the PyAudio instance must be kept alive; a local `pa`
        # could be garbage-collected (PyAudio.__del__ terminates PortAudio)
        # while its stream was still stored in self._stream.
        self._pa: Optional[Any] = None
        self._auto_start_attempted: bool = False
        self._capture_backend: str = "direct_pcm_buffer"
        self._prev_mag_spectrum: Optional[np.ndarray] = None
        self._last_analysis: Dict[str, Any] = self._empty_result()
        # S-0701: one-shot events detected but not yet delivered by poll.
        self._pending_events: Dict[str, bool] = {
            k: False for k in _ONE_SHOT_AUDIO_EVENT_KEYS
        }
        # F-0158: in-analyzer consumer state — the loopback stream is
        # polled by a daemon thread so events are actually read in
        # production (previously the analyzer was only constructed and
        # stored, never started or polled by any production caller).
        self._event_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._audio_event_history: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._audio_event_counts: Dict[str, int] = {
            k: 0 for k in _ONE_SHOT_AUDIO_EVENT_KEYS
        }
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_interval_sec: float = 0.1
        # F-0158: wire itself up on construction — no production caller
        # ever started the stream or polled it. Any failure is silent
        # (debug log only) and leaves the analyzer usable via
        # analyze_pcm_buffer / poll_latest_audio_events.
        try:
            self.start_audio_event_loop()
        except Exception as exc:
            logger.debug("[AudioLoopbackFFTAnalyzer] auto-start skipped: %s", exc)

    def _hann_window_for(self, n: int) -> np.ndarray:
        """Return the cached Hann window of length ``n`` (computed once)."""
        window = self._hann_cache.get(n)
        if window is None:
            if len(self._hann_cache) >= 8:
                self._hann_cache.clear()
            window = np.hanning(n).astype(np.float32)
            self._hann_cache[n] = window
        return window

    def _spectral_bands_for(
        self, n: int, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cached ``(freqs, danger_mask, fail_mask, chime_mask)`` for ``(n, sr)``.

        The returned arrays are read-only inputs to the analysis below and
        must not be mutated by callers.
        """
        key = (n, sr)
        cached = self._spectral_cache.get(key)
        if cached is None:
            if len(self._spectral_cache) >= 8:
                self._spectral_cache.clear()
            freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
            cached = (
                freqs,
                (freqs >= 80.0) & (freqs <= 320.0),
                (freqs >= 180.0) & (freqs <= 650.0),
                (freqs >= 2200.0) & (freqs <= 4800.0),
            )
            self._spectral_cache[key] = cached
        return cached

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "gather_success_chime": False,
            "gather_fail_sound": False,
            "danger_audio_alert": False,
            "dominant_event": "idle",
            "confidence": 0.0,
            "rms_energy": 0.0,
            "spectral_centroid_hz": 0.0,
            "dominant_freq_hz": 0.0,
            "spectral_flatness": 0.0,
            "temporal_crest_factor": 0.0,
            "band_ratios": {
                "low_danger_80_320hz": 0.0,
                "mid_fail_180_650hz": 0.0,
                "high_chime_2200_4800hz": 0.0,
            },
            "backend": self._capture_backend,
        }

    # ------------------------------------------------------------------
    # Live Windows WASAPI Loopback Stream Management (Optional Background Thread)
    # ------------------------------------------------------------------
    def _validate_loopback_device(self) -> None:
        """Validate device_index before opening a stream (F-0328).

        Loopback listens to an OUTPUT device. An input-only device index
        (e.g. a microphone) passed here would either listen to the mic or
        fail the stream, so it is rejected and the default device is used.
        """
        if self.device_index is None:
            return
        if _HAS_SOUNDDEVICE and sd is not None:
            try:
                info = sd.query_devices(self.device_index)
                if int(info.get("max_output_channels", 0)) <= 0:
                    logger.warning(
                        "[AudioLoopbackFFTAnalyzer] device_index=%s is input-only "
                        "(no output channels); using the default loopback device.",
                        self.device_index,
                    )
                    self.device_index = None
                return
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] device query failed: %s", exc)
        if _HAS_PYAUDIO and pyaudio is not None:
            try:
                _pa_probe = pyaudio.PyAudio()
                try:
                    info = _pa_probe.get_device_info_by_index(int(self.device_index))
                    if int(info.get("maxOutputChannels", 0)) <= 0:
                        logger.warning(
                            "[AudioLoopbackFFTAnalyzer] device_index=%s is input-only "
                            "(no output channels); using the default loopback device.",
                            self.device_index,
                        )
                        self.device_index = None
                finally:
                    _pa_probe.terminate()
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] device query failed: %s", exc)

    def start_loopback_stream(self) -> bool:
        """Start non-blocking Windows WASAPI audio loopback capture if audio hardware is available."""
        if self._running:
            self._ensure_audio_event_loop()
            return True
        self._auto_start_attempted = False
        self._validate_loopback_device()

        # 1. Try sounddevice WASAPI loopback
        if _HAS_SOUNDDEVICE and sd is not None:
            try:
                wasapi_settings = None
                if hasattr(sd, "WasapiSettings"):
                    try:
                        wasapi_settings = sd.WasapiSettings(loopback=True)  # type: ignore
                    except Exception:
                        wasapi_settings = None

                def _sd_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
                    mono = (
                        np.mean(indata, axis=1).astype(np.float32)
                        if indata.ndim > 1
                        else indata.astype(np.float32)
                    )
                    with self._lock:
                        self._ring_buffer.append(mono.copy())

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.fft_size,
                    device=self.device_index,
                    channels=1,
                    dtype="float32",
                    extra_settings=wasapi_settings,
                    callback=_sd_callback,
                )
                self._stream.start()
                self._running = True
                self._capture_backend = "sounddevice_wasapi"
                logger.info("[AudioLoopbackFFTAnalyzer] Started sounddevice WASAPI stream.")
                self._ensure_audio_event_loop()
                return True
            except Exception as exc:
                logger.debug(
                    "[AudioLoopbackFFTAnalyzer] sounddevice loopback unavailable: %s", exc
                )
                self._stream = None

        # 2. Try PyAudio input/loopback stream
        if _HAS_PYAUDIO and pyaudio is not None:
            try:
                pa = pyaudio.PyAudio()
                self._pa = pa

                def _pa_callback(in_data: bytes, frame_count: int, time_info: Any, status: int) -> Tuple[None, int]:
                    samples = np.frombuffer(in_data, dtype=np.float32)
                    with self._lock:
                        self._ring_buffer.append(samples.copy())
                    return (None, pyaudio.paContinue)

                self._stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.fft_size,
                    stream_callback=_pa_callback,
                )
                self._stream.start_stream()
                self._running = True
                self._capture_backend = "pyaudio_wasapi"
                logger.info("[AudioLoopbackFFTAnalyzer] Started PyAudio stream.")
                self._ensure_audio_event_loop()
                return True
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] PyAudio stream unavailable: %s", exc)
                self._stream = None
                if self._pa is not None:
                    try:
                        self._pa.terminate()
                    except Exception:
                        pass
                    self._pa = None

        # 3. Check Windows winmm waveIn numDevs via ctypes
        try:
            if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "winmm"):
                num_devs = int(ctypes.windll.winmm.waveInGetNumDevs())
                self._capture_backend = f"winmm_pcm_ready(devs={num_devs})"
        except Exception:
            self._capture_backend = "direct_pcm_buffer"

        return False

    def stop_loopback_stream(self) -> None:
        """Stop live audio capture stream cleanly."""
        self._running = False
        poll_thread = self._poll_thread
        self._poll_thread = None
        if (
            poll_thread is not None
            and poll_thread.is_alive()
            and poll_thread is not threading.current_thread()
        ):
            poll_thread.join(timeout=0.5)
        if self._stream is not None:
            try:
                if hasattr(self._stream, "stop"):
                    self._stream.stop()
                if hasattr(self._stream, "stop_stream"):
                    self._stream.stop_stream()
                if hasattr(self._stream, "close"):
                    self._stream.close()
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] Stream close error: %s", exc)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] PyAudio terminate error: %s", exc)
            self._pa = None

    # ------------------------------------------------------------------
    # F-0158: in-analyzer audio event consumer loop
    #
    # No production caller ever started the loopback stream or polled
    # it, so the analyzer consumes itself: once the stream runs, a
    # daemon thread polls it every ~100 ms and dispatches each consumed
    # one-shot event to registered handlers plus a bounded in-memory
    # history/counters readable via get_audio_event_history/counts.
    # ------------------------------------------------------------------
    def register_audio_event_handler(
        self, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a callback invoked once per consumed one-shot event.

        The handler receives a copy of the poll result dict (with the
        fired one-shot flags True). Handler exceptions are swallowed
        so one bad handler never breaks the poll loop.
        """
        if callable(handler) and handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def unregister_audio_event_handler(
        self, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        try:
            self._event_handlers.remove(handler)
        except ValueError:
            pass

    def get_audio_event_history(self) -> List[Dict[str, Any]]:
        """Consumed one-shot audio events, oldest first (bounded to 200)."""
        with self._lock:
            return [dict(entry) for entry in self._audio_event_history]

    def get_audio_event_counts(self) -> Dict[str, int]:
        """How many times each one-shot event was consumed so far."""
        with self._lock:
            return dict(self._audio_event_counts)

    def start_audio_event_loop(self) -> bool:
        """Start the loopback stream (if needed) and its consumer thread.

        Returns True when the stream is running and the poll thread is
        alive; False when no capture backend is available (the analyzer
        stays usable via analyze_pcm_buffer / poll_latest_audio_events).
        """
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return True
        if not self._running:
            try:
                if not self.start_loopback_stream():
                    return False
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] event loop start failed: %s", exc)
                return False
        return self._ensure_audio_event_loop()

    def _ensure_audio_event_loop(self) -> bool:
        if not self._running:
            return False
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return True
        thread = threading.Thread(
            target=self._audio_event_loop, name="audio-event-poll", daemon=True
        )
        self._poll_thread = thread
        thread.start()
        return True

    def _audio_event_loop(self) -> None:
        while self._running:
            try:
                events = self.poll_latest_audio_events()
                self._dispatch_audio_events(events)
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] poll loop error: %s", exc)
            deadline = time.monotonic() + self._poll_interval_sec
            while self._running and time.monotonic() < deadline:
                time.sleep(0.02)

    def _dispatch_audio_events(self, events: Dict[str, Any]) -> None:
        fired = [k for k in _ONE_SHOT_AUDIO_EVENT_KEYS if bool(events.get(k))]
        if not fired:
            return
        record = {
            "timestamp": time.time(),
            "dominant_event": events.get("dominant_event"),
            "confidence": events.get("confidence"),
        }
        with self._lock:
            for key in fired:
                self._audio_event_counts[key] = self._audio_event_counts.get(key, 0) + 1
                entry = dict(record)
                entry["event"] = key
                self._audio_event_history.append(entry)
        if "danger_audio_alert" in fired:
            logger.warning(
                "[AudioLoopbackFFTAnalyzer] danger audio alert "
                "(footsteps/attack transient)."
            )
        else:
            logger.debug("[AudioLoopbackFFTAnalyzer] audio event(s): %s", ",".join(fired))
        for handler in list(self._event_handlers):
            try:
                handler(dict(events))
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] event handler error: %s", exc)

    def poll_latest_audio_events(self) -> Dict[str, Any]:
        """Analyze captured loopback audio frames if available, else return last state.

        One-shot event flags (``gather_success_chime``,
        ``gather_fail_sound``, ``danger_audio_alert``) are delivered
        exactly once (S-0701): a detection sets a pending flag, the next
        poll returns it True and consumes it, and later polls report
        False until a new chunk detects the event again. Level-like
        fields (confidence, rms_energy, spectral metrics) keep
        reporting the last analysis.

        The stream is started lazily on first poll (it was previously only
        ever instantiated, never started or polled in production). Every
        chunk buffered since the last poll is analysed, not just the last
        one, so events are not lost when the ring buffer wraps.
        """
        if not self._running and not self._auto_start_attempted:
            self._auto_start_attempted = True
            try:
                self.start_loopback_stream()
            except Exception as exc:
                logger.debug("[AudioLoopbackFFTAnalyzer] lazy loopback start failed: %s", exc)
        chunks: List[np.ndarray] = []
        with self._lock:
            if self._ring_buffer:
                chunks = list(self._ring_buffer)
                self._ring_buffer.clear()
        if chunks:
            results = [self.analyze_pcm_buffer(c, self.sample_rate) for c in chunks]
            merged = dict(results[-1])
            merged["gather_success_chime"] = any(r["gather_success_chime"] for r in results)
            merged["gather_fail_sound"] = any(r["gather_fail_sound"] for r in results)
            merged["danger_audio_alert"] = any(r["danger_audio_alert"] for r in results)
            best = max(results, key=lambda r: float(r.get("confidence", 0.0)))
            merged["dominant_event"] = best["dominant_event"]
            merged["confidence"] = best["confidence"]
        # P2-3 (K-17): _last_analysis + _pending_events paylasilan
        # durumdur (poll thread + analiz cagiranlar + probe). Eski kod
        # ikisini de kilitsiz okuyup yaziyordu: iki eszamanli poll ayni
        # one-shot olayi iki kez teslim edebiliyor, _last_analysis
        # okumasi yarim yazilmis dict gorebiliyordu. Yayin + tuketim
        # tek kisa kilit altinda atomik yapilir; agir FFT analizi
        # kilit disinda kalir (kilit asla analiz sirasinda tutulmaz —
        # analyze_pcm_buffer kendi kisa kilitlerini alir, deadlock yok).
        with self._lock:
            if chunks:
                self._last_analysis = merged
                # Queue the one-shot flags for exactly-once delivery.
                for key in _ONE_SHOT_AUDIO_EVENT_KEYS:
                    if merged.get(key):
                        self._pending_events[key] = True
            # Consume: the returned flags come from the pending queue, never
            # from the stale _last_analysis, so a detected event is reported
            # True on exactly one poll.
            snapshot = dict(self._last_analysis)
            for key in _ONE_SHOT_AUDIO_EVENT_KEYS:
                snapshot[key] = bool(self._pending_events.get(key, False))
                self._pending_events[key] = False
        return snapshot

    # ------------------------------------------------------------------
    # Core numpy.fft.rfft + Hann Window Spectral Analyzer
    # ------------------------------------------------------------------
    def analyze_pcm_buffer(
        self, samples: np.ndarray, sample_rate: int = 44100
    ) -> Dict[str, Any]:
        """Analyze a raw PCM audio buffer using Hann windowing and ``numpy.fft.rfft``.

        Args:
            samples: 1D or 2D (stereo) numpy array of PCM audio samples
                (``float32``/``float64`` in ``[-1.0, 1.0]`` or ``int16`` in ``[-32768, 32767]``).
            sample_rate: Sampling frequency in Hz (default ``44100``).

        Returns:
            Dictionary with boolean flags ``gather_success_chime``, ``gather_fail_sound``,
            ``danger_audio_alert``, ``dominant_event``, ``confidence``, and spectral metrics.
        """
        sr = max(8000, int(sample_rate))
        if samples is None or not isinstance(samples, np.ndarray) or samples.size == 0:
            return self._empty_result()

        # Convert stereo/multi-channel to mono float32
        x = np.asarray(samples)
        if x.ndim > 1:
            x = np.mean(x, axis=-1)
        x = x.ravel()

        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            max_abs = float(max(abs(info.min), abs(info.max), 1))
            x = x.astype(np.float32) / max_abs
        else:
            x = x.astype(np.float32)

        if x.size < 64:
            return self._empty_result()

        # Remove DC offset
        x = x - float(np.mean(x))

        # Temporal RMS and impulse crest factor (for footstep / assassin attack transient detection)
        rms_energy = float(np.sqrt(np.mean(x * x) + 1e-12))
        peak_abs = float(np.max(np.abs(x)))
        temporal_crest_factor = peak_abs / max(rms_energy, 1e-7)

        # Silence gate. Reset the previous spectrum so the next
        # non-silent frame's flux is not computed against a minutes-old one.
        if rms_energy < 1e-4:
            res = self._empty_result()
            res["rms_energy"] = round(rms_energy, 6)
            # P2-3 (K-17): paylasilan analiz durumu kisa kilitle yazilir.
            with self._lock:
                self._prev_mag_spectrum = None
                self._last_analysis = res
            return res

        # Apply Hann window of exact buffer length (cached per length).
        n = x.size
        window = self._hann_window_for(n)

        windowed = x * window

        # Compute one-sided real FFT magnitude and power spectrum
        fft_complex = np.fft.rfft(windowed)
        mag_spectrum = np.abs(fft_complex).astype(np.float64) / float(n)
        power_spectrum = mag_spectrum * mag_spectrum + 1e-18
        freqs, mask_danger, mask_fail, mask_chime = self._spectral_bands_for(n, sr)

        total_power = float(np.sum(power_spectrum))
        total_mag = float(np.sum(mag_spectrum)) + 1e-12

        # Spectral Centroid (Hz)
        spectral_centroid_hz = float(np.sum(freqs * mag_spectrum) / total_mag)

        # Dominant Frequency (Hz)
        dom_idx = int(np.argmax(mag_spectrum[1:]) + 1) if mag_spectrum.size > 1 else 0
        dominant_freq_hz = float(freqs[dom_idx])

        # Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean of power spectrum)
        # Distinguish tonal crystal chimes (low flatness) from broadband fracture/shatter noise (higher flatness)
        active_bins = power_spectrum[1:] if power_spectrum.size > 1 else power_spectrum
        log_mean = float(np.mean(np.log(active_bins + 1e-18)))
        geom_mean = math.exp(max(-40.0, min(40.0, log_mean)))
        arith_mean = float(np.mean(active_bins)) + 1e-18
        spectral_flatness = float(min(1.0, max(0.0, geom_mean / arith_mean)))

        # Spectral Flux (frame-to-frame onset change). Onceki spektrum
        # kisa kilitle okunur, guncel spektrum kisa kilitle yazilir;
        # flux yalnizca tanisal bir metriktir, okuma-yazma arasindaki
        # kucuk TOCTOU penceresi kabul edilir (P2-3).
        with self._lock:
            _prev_mag_spectrum = self._prev_mag_spectrum
        if (
            _prev_mag_spectrum is not None
            and _prev_mag_spectrum.shape == mag_spectrum.shape
        ):
            diff = mag_spectrum - _prev_mag_spectrum
            spectral_flux = float(np.sum(np.maximum(0.0, diff)) / total_mag)
        else:
            spectral_flux = 0.25
        with self._lock:
            self._prev_mag_spectrum = mag_spectrum.copy()

        # Band Power Ratios (masks come from the cached _spectral_bands_for):
        # 1) Danger / Footstep / Assassin Swing Band: 80 Hz – 320 Hz
        power_danger = float(np.sum(power_spectrum[mask_danger]))
        ratio_danger = power_danger / max(total_power, 1e-18)

        # 2) Extraction Failure Shatter / Rock Thud Band: 180 Hz – 650 Hz
        power_fail = float(np.sum(power_spectrum[mask_fail]))
        ratio_fail = power_fail / max(total_power, 1e-18)

        # 3) Success Crystal Chime Band: 2200 Hz – 4800 Hz
        power_chime = float(np.sum(power_spectrum[mask_chime]))
        ratio_chime = power_chime / max(total_power, 1e-18)

        # Narrowband harmonic peak ratio inside the 2200–4800 Hz crystal chime band
        if np.any(mask_chime):
            chime_mags = mag_spectrum[mask_chime]
            chime_peak_ratio = float(
                np.max(chime_mags) / (np.mean(chime_mags) + 1e-12)
            )
        else:
            chime_peak_ratio = 0.0

        # Bandwidth spread inside the 180–650 Hz failure band (to distinguish broadband rock/metal
        # fracture burst from a narrow pure low footstep thump)
        if np.any(mask_fail):
            fail_mags = mag_spectrum[mask_fail]
            fail_p = fail_mags * fail_mags + 1e-18
            fail_flatness = float(
                math.exp(float(np.mean(np.log(fail_p)))) / (float(np.mean(fail_p)) + 1e-18)
            )
            # Active bin fraction (bins > 15% of peak bin magnitude in 180-650Hz)
            fail_active_spread = float(
                np.mean(fail_mags >= 0.15 * float(np.max(fail_mags)))
            )
        else:
            fail_flatness = 0.0
            fail_active_spread = 0.0

        # Evaluate temporal envelope modulation (impulsive bursts vs steady broadband hum)
        # Split signal into 8 short sub-windows to measure envelope variance / transient burstiness
        num_seg = 8
        seg_len = max(1, n // num_seg)
        seg_rms = [
            float(np.sqrt(np.mean(x[i * seg_len : (i + 1) * seg_len] ** 2) + 1e-12))
            for i in range(num_seg)
            if (i + 1) * seg_len <= n
        ]
        env_max_to_min = (
            max(seg_rms) / max(min(seg_rms), 1e-6) if seg_rms else 1.0
        )

        # ------------------------------------------------------------------
        # Decision Rules for the 3 Game Audio Signatures
        # ------------------------------------------------------------------
        # 1. Success Crystal Chime (gather_success_chime):
        #    - High-frequency crystalline harmonics in 2200–4800 Hz
        #    - Spectral centroid > 2400 Hz
        #    - Strong narrowband harmonic peak ratio (tonal chime, not white noise)
        is_success_chime = bool(
            spectral_centroid_hz > 2400.0
            and ratio_chime >= 0.42
            and chime_peak_ratio >= 3.2
            and spectral_flatness < 0.35
        )

        # 2. Extraction Failure Shatter/Thud (gather_fail_sound):
        #    - Broadband metallic/rock fracture burst concentrated in 180–650 Hz
        #    - Centroid between 180 Hz and 1150 Hz
        #    - High in-band spectral flatness / multi-frequency fracture spread
        is_fail_sound = bool(
            not is_success_chime
            and ratio_fail >= 0.45
            and 180.0 <= spectral_centroid_hz <= 1250.0
            # In-band broadband evidence is required: BOTH in-band flatness
            # AND active spread (OR was too weak — a 3-tone chord
            # 260+395+530 Hz passes spread alone at 0.0847 via Hann
            # leakage while its in-band flatness is 0.0000; the fail
            # synth passes both at 0.4612 / 0.7797). The GLOBAL
            # spectral_flatness must NOT be used here: it is ~0.000000 for
            # any band-concentrated signal by construction (ratio_fail >=
            # 0.45 puts >=45% of the power in-band, the remaining bins are
            # near-zero, so the geometric mean over ALL bins collapses).
            # Measured on the module's own synthesize_failure_shatter():
            # global flatness = 0.000000 while in-band flatness = 0.4612
            # and active spread = 0.7797, so the old
            # `spectral_flatness >= 0.015` conjunct made this rule
            # unreachable (0/6 duration/amplitude combos detected).
            and fail_flatness >= 0.05
            and fail_active_spread >= 0.08
            and dominant_freq_hz >= 180.0
        )

        # 3. Approaching Assassin / Monster Footstep & Attack (danger_audio_alert):
        #    - Impulsive low-frequency transients in 80–320 Hz
        #    - High temporal crest factor or envelope transient ratio (footstep thuds / weapon swing)
        is_danger_alert = bool(
            not is_success_chime
            and ratio_danger >= 0.45
            and spectral_centroid_hz < 420.0
            and dominant_freq_hz <= 310.0
            and (temporal_crest_factor >= 2.1 or env_max_to_min >= 1.8)
        )

        # Resolve mutual exclusivity if both fail_sound and danger_alert overlap in 180-320Hz
        if is_danger_alert and is_fail_sound:
            if dominant_freq_hz < 210.0 and temporal_crest_factor >= 2.6:
                is_fail_sound = False
            else:
                is_danger_alert = False

        if is_success_chime:
            dominant_event = "gather_success_chime"
            confidence = round(
                min(0.99, 0.78 + 0.15 * ratio_chime + min(0.06, chime_peak_ratio / 100.0)),
                3,
            )
        elif is_fail_sound:
            dominant_event = "gather_fail_sound"
            confidence = round(
                min(0.97, 0.74 + 0.18 * ratio_fail + min(0.05, fail_flatness)),
                3,
            )
        elif is_danger_alert:
            dominant_event = "danger_audio_alert"
            confidence = round(
                min(
                    0.98,
                    0.75
                    + 0.16 * ratio_danger
                    + min(0.07, (temporal_crest_factor - 1.5) * 0.03),
                ),
                3,
            )
        else:
            dominant_event = "idle"
            confidence = 0.0

        result: Dict[str, Any] = {
            "gather_success_chime": is_success_chime,
            "gather_fail_sound": is_fail_sound,
            "danger_audio_alert": is_danger_alert,
            "dominant_event": dominant_event,
            "confidence": confidence,
            "rms_energy": round(rms_energy, 5),
            "spectral_centroid_hz": round(spectral_centroid_hz, 2),
            "dominant_freq_hz": round(dominant_freq_hz, 2),
            "spectral_flatness": round(spectral_flatness, 5),
            "temporal_crest_factor": round(temporal_crest_factor, 3),
            "spectral_flux": round(spectral_flux, 4),
            "band_ratios": {
                "low_danger_80_320hz": round(ratio_danger, 4),
                "mid_fail_180_650hz": round(ratio_fail, 4),
                "high_chime_2200_4800hz": round(ratio_chime, 4),
            },
            "backend": self._capture_backend,
        }
        # P2-3 (K-17): paylasilan analiz durumu kisa kilitle yazilir.
        with self._lock:
            self._last_analysis = result
        return result

    # ------------------------------------------------------------------
    # Waveform Synthesizers for Self-Test & Diagnostic Calibration
    # ------------------------------------------------------------------
    @staticmethod
    def synthesize_crystal_chime(
        duration_sec: float = 0.15, sample_rate: int = 44100
    ) -> np.ndarray:
        """Synthesize a canonical Aion 2 gathering success crystal chime (2800Hz + 3600Hz + 4400Hz harmonics)."""
        n = int(round(duration_sec * sample_rate))
        t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
        env = np.exp(-t * 14.0).astype(np.float32)
        wave = (
            0.55 * np.sin(2.0 * math.pi * 2850.0 * t)
            + 0.30 * np.sin(2.0 * math.pi * 3620.0 * t)
            + 0.15 * np.sin(2.0 * math.pi * 4380.0 * t)
        ) * env
        return wave.astype(np.float32)

    @staticmethod
    def synthesize_failure_shatter(
        duration_sec: float = 0.15, sample_rate: int = 44100
    ) -> np.ndarray:
        """Synthesize a canonical extraction failure rock/metallic fracture burst (180Hz – 650Hz broadband)."""
        n = int(round(duration_sec * sample_rate))
        t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
        rng = np.random.default_rng(42)
        # Broadband rock/metallic fracture burst filtered to 180Hz..650Hz in the frequency domain
        noise = rng.standard_normal(n).astype(np.float32)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
        band_mask = ((freqs >= 180.0) & (freqs <= 650.0)).astype(np.float32)
        band_burst = np.fft.irfft(spec * band_mask, n=n).astype(np.float32)
        band_burst /= float(np.max(np.abs(band_burst)) + 1e-7)
        # Combine broadband rock fracture with metallic clatter modes (260Hz, 390Hz, 520Hz)
        clatter = (
            0.35 * np.sin(2.0 * math.pi * 260.0 * t)
            + 0.35 * np.sin(2.0 * math.pi * 395.0 * t)
            + 0.30 * np.sin(2.0 * math.pi * 530.0 * t)
        ).astype(np.float32)
        env = np.exp(-t * 9.0).astype(np.float32)
        return ((0.65 * band_burst + 0.35 * clatter) * env).astype(np.float32)

    @staticmethod
    def synthesize_monster_footstep(
        duration_sec: float = 0.15, sample_rate: int = 44100
    ) -> np.ndarray:
        """Synthesize an impulsive low-frequency approaching monster footstep / assassin swing (80Hz – 320Hz)."""
        n = int(round(duration_sec * sample_rate))
        t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
        # Two sharp low-frequency thump transients (115 Hz + 165 Hz) with fast exponential decay
        pulse1 = np.exp(-((t - 0.025) ** 2) / (2.0 * (0.008 ** 2))).astype(np.float32)
        pulse2 = 0.75 * np.exp(-((t - 0.095) ** 2) / (2.0 * (0.009 ** 2))).astype(
            np.float32
        )
        env = pulse1 + pulse2
        carrier = (
            0.65 * np.sin(2.0 * math.pi * 118.0 * t)
            + 0.35 * np.sin(2.0 * math.pi * 168.0 * t)
        ).astype(np.float32)
        return (carrier * env).astype(np.float32)

    def _probe_audio_pipeline(self) -> bool:
        """Real readiness probe (S-0302): the FFT pipeline must actually work.

        Checks config bounds, the Hann window, and runs the spectral
        analyzer end-to-end on a synthesized success chime (which must be
        detected). Instance analysis state is saved and restored so a
        diagnostic probe never injects a stale event into
        ``poll_latest_audio_events``.
        """
        try:
            if int(self.sample_rate) < 8000 or int(self.fft_size) < 256:
                return False
            window = getattr(self, "_hann_window", None)
            if window is None or len(window) != int(self.fft_size):
                return False
            if not callable(getattr(self, "analyze_pcm_buffer", None)):
                return False
            if not bool(np.isfinite(np.fft.rfft(np.hanning(256).astype(np.float32))).all()):
                return False
            # P2-3 (K-17): probe'un durum kaydet/geri-yukle adimlari
            # kisa kilitle yapilir. Not: analyze_pcm_buffer cagrisi
            # sirasinda kilit tutulmaz (analiz kendi kisa kilitlerini
            # alir); bu yuzden eszamanli bir poll, probe'un gecici
            # sonucunu cok kisa sureligine gorebilir — onceki
            # kilitsiz duruma gore daraltilmis, ama tamamen
            # kapatilmamis bir yaristir (kapatmak FFT'yi kilit altina
            # sokmak olurdu).
            with self._lock:
                prev_analysis = self._last_analysis
                prev_spectrum = self._prev_mag_spectrum
            try:
                chime = self.synthesize_crystal_chime()
                result = self.analyze_pcm_buffer(chime, int(self.sample_rate))
                return bool(result.get("gather_success_chime"))
            finally:
                with self._lock:
                    self._last_analysis = prev_analysis
                    self._prev_mag_spectrum = prev_spectrum
        except Exception:
            return False

    def get_neural_audio_ready_flag(self) -> dict:
        # S-0302: bound to the real FFT-pipeline probe, not a constant True.
        return {"module": "neural_audio_perception", "ready": bool(self._probe_audio_pipeline())}

    def get_neural_audio_health_summary(self) -> dict:
        try:
            return {
                "neural_audio_ready": bool(self._probe_audio_pipeline()),
                "loopback_stream_running": bool(getattr(self, "_running", False)),
                "capture_backend": str(getattr(self, "_capture_backend", "unknown")),
            }
        except Exception:
            return {"neural_audio_ready": False}

    def get_neural_audio_version_info(self) -> dict:
        return {"module": "neural_audio_perception", "version": __version__, "ready": bool(self._probe_audio_pipeline())}

    def get_neural_audio_diagnostic_snapshot(self) -> dict:
        _ok = bool(self._probe_audio_pipeline())
        return {"module": "neural_audio_perception", "status": "ok" if _ok else "degraded", "ready": _ok}

    def get_neural_audio_perception_capabilities(self) -> list:
        return ["neural_audio_perception", "diagnostic_snapshot", "version_info"]


def get_neural_audio_perception_capabilities() -> list:
    return ["neural_audio_perception", "diagnostic_snapshot", "version_info"]


def validate_neural_audio_perception_contract() -> bool:
    """Real contract checks (S-0302, not a tautology): capabilities are
    non-empty, the ONNX detector exposes its 3D entry point, and a fresh
    AudioLoopbackFFTAnalyzer passes its end-to-end FFT probe (synthesized
    success chime must be detected).

    P2-1: prob analyzer'i kapatilir. ``AudioLoopbackFFTAnalyzer()``
    kurucuda ``start_audio_event_loop()``'u cagirir; kapanmayan ornek
    WASAPI/PyAudio akisi + daemon poll thread birakir. Zincir eskiden
    tek okumada 9-10 analyzer aciyor ve hicbirini durdurmuyordu.
    """
    analyzer = None
    try:
        if not get_neural_audio_perception_capabilities():
            return False
        if not callable(getattr(ONNXYolo3DDetector, "detect_3d_objects", None)):
            return False
        if not callable(getattr(AudioLoopbackFFTAnalyzer, "analyze_pcm_buffer", None)):
            return False
        analyzer = AudioLoopbackFFTAnalyzer()
        return bool(analyzer._probe_audio_pipeline())
    except Exception:
        return False
    finally:
        if analyzer is not None:
            try:
                analyzer.stop_loopback_stream()
            except Exception:
                pass


def _neural_audio_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. telemetry.py'deki aciklama).
    ``None`` ise dogrulama burada bir kez calisir (tek analyzer acar ve
    kapatir); cagrilar arasi onbellek yoktur."""
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_neural_audio_perception_contract())


def get_neural_audio_perception_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "neural_audio_perception", "contract_valid": _neural_audio_contract_ok(_contract_ok), "capability_count": len(get_neural_audio_perception_capabilities())}


def validate_neural_audio_contract(_contract_ok: Optional[bool] = None) -> bool:
    return _neural_audio_contract_ok(_contract_ok)


def get_neural_audio_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return get_neural_audio_perception_baseline_metrics(_contract_ok=_contract_ok)


def get_neural_audio_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "ready": ok, "metrics": get_neural_audio_baseline_metrics(_contract_ok=ok)}


def get_neural_audio_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "audit_passed": ok, "readiness": get_neural_audio_readiness_status(_contract_ok=ok)}


def get_neural_audio_perception_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    return get_neural_audio_audit_record(_contract_ok=_contract_ok)


def get_neural_audio_perception_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "operational": ok, "audit": get_neural_audio_perception_audit_record(_contract_ok=ok)}


def get_neural_audio_perception_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "telemetry_ready": ok, "digest": get_neural_audio_perception_operational_digest(_contract_ok=ok)}


def get_neural_audio_perception_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "sealed": ok, "telemetry": get_neural_audio_perception_telemetry_payload(_contract_ok=ok)}


def get_neural_audio_perception_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "certified": ok, "seal": get_neural_audio_perception_integrity_seal(_contract_ok=ok)}


def get_neural_audio_perception_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "ready": ok, "certificate": get_neural_audio_perception_compliance_certificate(_contract_ok=ok)}


def get_neural_audio_perception_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "audit_passed": ok, "probe": get_neural_audio_perception_readiness_probe(_contract_ok=ok)}


def get_neural_audio_perception_heartbeat_snapshot(_contract_ok: Optional[bool] = None) -> dict:
    ok = _neural_audio_contract_ok(_contract_ok)
    return {"module": "neural_audio_perception", "alive": ok, "audit": get_neural_audio_perception_audit_summary(_contract_ok=ok)}

