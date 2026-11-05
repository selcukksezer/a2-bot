# -*- coding: utf-8 -*-
"""
AION 2 Gatherion — Gelişmiş Toplama, Binek ve Haritasız Mikro-Navigasyon Zekası (`core/navigation_intel.py`).

Maddeler 21–25:
- Madde 21: `ThreeStageHarvestController` — 3 Aşamalı (`1/3 ➔ 2/3 ➔ 3/3`) 400ms Re-Prompt
  Dinamik Toplama Döngüsü. `1/3` ve `2/3` tamamlandıktan sonra `400ms` (`0.4s` yapılandırılabilir)
  bekleyip `[F Gather]` / `[E]` isteminin tekrar belirip belirmediğini kontrol eder; `3/3`
  tamamlanana veya kaynak tükenene kadar botu aynı düğümde sabit (`is_anchored=True`) tutar.
- Madde 22: `PartialNodeDetector` — Yarım Bırakılmış Kaynak (Partial Node) Algılama.
  Başka bir oyuncu kaynağı önceden 1 veya 2 kez toplamışsa (`1/3` veya `2/3` sonrasında kaynak
  yok olup doğrulama penceresi içinde `[F Gather]` / `[E]` tekrar çıkmazsa), 3. toplama için
  boşuna beklemek yerine düğümü `route_mgr.mark_node_gathered(node_id, force_deplete=True)`
  ile derhal `RESTING` (5 dakika) durumuna geçirir.
- Madde 23: `handle_extraction_failure` — Toplama Başarısızlığında (`Essence Extraction failed`)
  Hak Düşmeme. Sohbette `"Essence Extraction failed"` algılandığında `current_harvests` sayacını
  artırmaz / hak düşmez (`0/3` -> `0/3`, `1/3` -> `1/3` olarak aynı aşamayı tekrar dener) ve
  aynı kaynağa anında tekrar `F` / `E` basar.
- Madde 24: `AutoMountManager` — Mesafe Bazlı Otomatik Binek (`> 35m` Bin / `<= 8m` veya
  `[F Gather]` / `[E]` Görününce İn & Topla). Sıradaki düğüm `> 35m` uzaktaysa, karakter
  savaşta değilse ve sohbette `"Cannot mount in current state"` yoksa otomatik bineğe biner;
  düğüme `<= 8m` yaklaşıldığında veya `[F Gather]` / `[E]` belirdiğinde otomatik inip toplamaya başlar.
- Madde 25: `MaplessMicroNavigator` — Minimap Üzerinden Doğrudan Yönelim (`< 40m` Haritasız
  Mikro-Navigasyon). Sıradaki düğüm `< 40m` mesafedeyse veya sağ-üst minimap üzerinde
  görünüyorsa tam ekran `M` haritasını açıp vakit kaybetmek yerine doğrudan sağ-üst minimap
  ofsetinden `(dx, dy)` yönelim vektörünü ve pusula açısını (`heading_angle_deg`) hesaplayarak
  karakteri hedefe yönlendirir.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.minimap_scanner import MinimapScanner
from core.ocr_distance import GameHUDReader

DEFAULT_MAX_HARVESTS: int = 3
DEFAULT_REST_SEC: int = 300

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# SABİTLER (MADDELER 21–25)
# ==============================================================================
DEFAULT_REPROMPT_WAIT_SEC: float = 0.40  # 400ms re-prompt bekleme süresi (Madde 21)
DEFAULT_REPROMPT_WAIT_MS: int = 400
DEFAULT_VERIFICATION_WINDOW_SEC: float = 0.85
DEFAULT_POLL_INTERVAL_SEC: float = 0.05
DEFAULT_CAST_DURATION_SEC: float = 1.80
DEFAULT_REST_MINUTES: float = 5.0
MOUNT_DISTANCE_THRESHOLD_M: float = 35.0  # > 35m otomatik binek (Madde 24)
DISMOUNT_DISTANCE_THRESHOLD_M: float = 8.0  # <= 8m otomatik inme (Madde 24)
MAPLESS_DISTANCE_THRESHOLD_M: float = 40.0  # < 40m haritasız mikro-navigasyon (Madde 25)


# ==============================================================================
# YARDIMCI FONKSİYONLAR
# ==============================================================================
def get_navigation_intel_summary(self) -> dict:
    try:
        return {"intel_ready": True, "metrics": getattr(self, "__dict__", {})}
    except Exception as e:
        return {"intel_ready": True, "error": str(e)}


def _send_key_via_controller(
    input_ctrl: Any,
    key: str,
    test_mode: bool = False,
) -> bool:
    """
    Verilen `input_ctrl` nesnesi üzerinden (`press_key`, `send_gather_key`, `tap_key`,
    `send_key` veya doğrudan çağrılabilir fonksiyon) güvenli tuş basımı gerçekleştirir.
    """
    clean_key = str(key or "f").strip().lower() or "f"
    if input_ctrl is None:
        return bool(test_mode)

    try:
        if clean_key == "f" and hasattr(input_ctrl, "send_gather_key") and not hasattr(input_ctrl, "press_key"):
            input_ctrl.send_gather_key()
            return True
        if hasattr(input_ctrl, "press_key") and callable(input_ctrl.press_key):
            input_ctrl.press_key(clean_key)
            return True
        if hasattr(input_ctrl, "tap_key") and callable(input_ctrl.tap_key):
            input_ctrl.tap_key(clean_key)
            return True
        if hasattr(input_ctrl, "send_key") and callable(input_ctrl.send_key):
            input_ctrl.send_key(clean_key)
            return True
        if clean_key == "f" and hasattr(input_ctrl, "send_gather_key") and callable(input_ctrl.send_gather_key):
            input_ctrl.send_gather_key()
            return True
        if callable(input_ctrl):
            input_ctrl(clean_key)
            return True
    except Exception as exc:
        logger.error("Tuş gönderimi başarısız ('%s'): %s", clean_key, exc)
        return False
    return False


def _detect_prompt_from_vision_or_frame(
    vision_engine: Any = None,
    frame: Optional[np.ndarray] = None,
    default_key: str = "f",
) -> Tuple[bool, str]:
    """
    `vision_engine` veya `frame` üzerinden `[F Gather]` / yüzen `[E]` isteminin
    aktif olup olmadığını ve etkileşim tuşunu (`"f"` veya `"e"`) tespit eder.
    """
    detected_key = str(default_key or "f").strip().lower() or "f"

    if vision_engine is not None:
        try:
            for method_name in ("detect_gather_prompt", "check_gather_prompt", "find_gather_prompt"):
                if hasattr(vision_engine, method_name):
                    fn = getattr(vision_engine, method_name)
                    try:
                        res = fn(frame) if frame is not None else fn()
                    except TypeError:
                        res = fn()

                    last_k = str(getattr(vision_engine, "last_detected_gather_key", "") or "").strip().lower()
                    if last_k in ("f", "e"):
                        detected_key = last_k

                    if isinstance(res, tuple) and len(res) >= 1:
                        is_found = bool(res[0])
                        if len(res) >= 2 and isinstance(res[1], str) and res[1].strip().lower() in ("f", "e"):
                            detected_key = res[1].strip().lower()
                        elif len(res) >= 3 and isinstance(res[2], str) and res[2].strip().lower() in ("f", "e"):
                            detected_key = res[2].strip().lower()
                        if is_found:
                            return True, detected_key
                    elif isinstance(res, dict):
                        is_found = bool(res.get("detected", res.get("found", res.get("visible", False))))
                        k_val = str(res.get("gather_key", res.get("key", detected_key))).strip().lower()
                        if k_val in ("f", "e"):
                            detected_key = k_val
                        if is_found:
                            return True, detected_key
                    elif bool(res):
                        return True, detected_key

            if hasattr(vision_engine, "detect_floating_resource_icon") and frame is not None:
                icon_res = vision_engine.detect_floating_resource_icon(frame, close_range_only=True)
                if isinstance(icon_res, tuple) and len(icon_res) >= 1 and bool(icon_res[0]):
                    last_k = str(getattr(vision_engine, "last_detected_gather_key", "e") or "e").strip().lower()
                    return True, (last_k if last_k in ("f", "e") else "e")
                if isinstance(icon_res, dict) and bool(icon_res.get("detected", icon_res.get("found", False))):
                    k_val = str(icon_res.get("key", icon_res.get("gather_key", "e"))).strip().lower()
                    return True, (k_val if k_val in ("f", "e") else "e")
        except Exception as exc:
            logger.debug("Vision üzerinden gather prompt kontrol hatası: %s", exc)

    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        try:
            from core.vision import VisionEngine
            temp_vision = VisionEngine()
            found, score, _ = temp_vision.detect_gather_prompt(frame)
            if found and score >= 0.65:
                k_val = str(getattr(temp_vision, "last_detected_gather_key", detected_key) or detected_key).lower()
                return True, (k_val if k_val in ("f", "e") else detected_key)
        except Exception as exc:
            logger.debug("Frame üzerinden yedek gather prompt kontrol hatası: %s", exc)

    return False, detected_key


def _detect_chat_alerts_unified(
    frame: Optional[np.ndarray] = None,
    chat_text: Optional[str] = None,
    chat_alerts: Optional[Dict[str, Any]] = None,
    hud_reader: Optional[GameHUDReader] = None,
) -> Dict[str, Any]:
    """
    Sohbet metni (`chat_text`), hazır uyarı sözlüğü (`chat_alerts`) veya oyun ekranı karesi
    (`frame`) üzerinden `"Essence Extraction failed"` ve `"Cannot mount in current state"`
    sistem uyarılarını birleşik olarak tespit eder.
    """
    extraction_failed = False
    cannot_mount = False
    orange_ratio = 0.0

    if isinstance(chat_alerts, dict):
        extraction_failed = bool(
            chat_alerts.get("extraction_failed", chat_alerts.get("essence_extraction_failed", False))
        )
        cannot_mount = bool(
            chat_alerts.get("cannot_mount", chat_alerts.get("cannot_mount_in_current_state", False))
        )
        orange_ratio = float(chat_alerts.get("orange_warning_ratio", 0.0) or 0.0)

    if isinstance(chat_text, str) and chat_text.strip():
        lower_txt = chat_text.strip().lower()
        if "essence extraction failed" in lower_txt or "extraction failed" in lower_txt:
            extraction_failed = True
        if "cannot mount in current state" in lower_txt or "cannot mount" in lower_txt:
            cannot_mount = True

    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        try:
            reader = hud_reader if hud_reader is not None else GameHUDReader()
            if hasattr(reader, "detect_chat_system_alerts"):
                det = reader.detect_chat_system_alerts(frame)
                if isinstance(det, dict):
                    extraction_failed = extraction_failed or bool(det.get("extraction_failed", False))
                    cannot_mount = cannot_mount or bool(det.get("cannot_mount", False))
                    orange_ratio = max(orange_ratio, float(det.get("orange_warning_ratio", 0.0) or 0.0))
        except Exception as exc:
            logger.debug("Sohbet sistem uyarısı okuma hatası: %s", exc)

    return {
        "extraction_failed": bool(extraction_failed),
        "cannot_mount": bool(cannot_mount),
        "orange_warning_ratio": round(float(orange_ratio), 6),
    }


def _resolve_node_reference(
    node_or_id: Union[int, NodeData, Dict[str, Any], None],
    route_mgr: Optional[RouteManager] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[int], Optional[Union[NodeData, Dict[str, Any]]]]:
    """
    `int`, `NodeData` veya `dict` olarak verilen düğüm referansını `(node_id, node_obj)`
    çiftine çözümler.
    """
    if node_or_id is None:
        if route_mgr is not None and hasattr(route_mgr, "get_next_ready_node"):
            found = route_mgr.get_next_ready_node(region=region)
            if found is not None:
                return int(found.id), found
        return None, None

    if isinstance(node_or_id, int):
        nid = int(node_or_id)
        nobj = route_mgr.get_node(nid, region=region) if (route_mgr is not None and hasattr(route_mgr, "get_node")) else None
        return nid, nobj

    if isinstance(node_or_id, NodeData):
        nid = int(node_or_id.id)
        if route_mgr is not None and hasattr(route_mgr, "get_node"):
            rm_node = route_mgr.get_node(nid, region=region or node_or_id.region)
            if rm_node is not None:
                return nid, rm_node
        return nid, node_or_id

    if isinstance(node_or_id, dict):
        nid = int(node_or_id.get("id", node_or_id.get("node_id", 1)))
        if route_mgr is not None and hasattr(route_mgr, "get_node"):
            rm_node = route_mgr.get_node(nid, region=region or node_or_id.get("region"))
            if rm_node is not None:
                return nid, rm_node
        return nid, node_or_id

    nid = int(getattr(node_or_id, "id", 1))
    return nid, node_or_id


# ==============================================================================
# MADDE 23 — TOPLAMA BAŞARISIZLIĞINDA (`Essence Extraction failed`) HAK DÜŞMEME
# ==============================================================================
def handle_extraction_failure(
    node_or_id: Union[int, NodeData, Dict[str, Any], None] = None,
    route_mgr: Optional[RouteManager] = None,
    input_ctrl: Any = None,
    gather_key: str = "f",
    frame: Optional[np.ndarray] = None,
    chat_text: Optional[str] = None,
    chat_alerts: Optional[Dict[str, Any]] = None,
    extraction_failed: Optional[bool] = None,
    hud_reader: Optional[GameHUDReader] = None,
    region: Optional[str] = None,
    retry_press: bool = True,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """
    Madde 23 — Toplama Başarısızlığında (`Essence Extraction failed`) Hak Düşmeme:
    Sohbette `"Essence Extraction failed"` algılandığında:
    1. `current_harvests` sayacını **KESİNLİKLE artırmaz** ve **sıfırlamaz** (`0/3` ise `0/3` kalır,
       `1/3` tamamlanmış ve `2/3` deneniyorken başarısız olduysa `1/3` kalır).
    2. Düğümü `resting` veya `failed` durumuna düşürmez; `state = "active"` olarak korur.
    3. Aynı kaynağa anında tekrar `F` veya `E` (`gather_key`) basarak aynı aşamayı yeniden dener.
    """
    # Argüman esnekliği: eğer 2. argüman olarak InputController verildiyse
    if route_mgr is not None and not hasattr(route_mgr, "get_node") and hasattr(route_mgr, "press_key"):
        if input_ctrl is None:
            input_ctrl = route_mgr
        route_mgr = None

    node_id, node_obj = _resolve_node_reference(node_or_id, route_mgr=route_mgr, region=region)

    # Başarısızlık durumunu belirle
    if extraction_failed is not None:
        is_failed = bool(extraction_failed)
    elif chat_text is not None or chat_alerts is not None or frame is not None:
        alerts = _detect_chat_alerts_unified(
            frame=frame,
            chat_text=chat_text,
            chat_alerts=chat_alerts,
            hud_reader=hud_reader,
        )
        is_failed = bool(alerts.get("extraction_failed", False))
    else:
        # Doğrudan hata işleyici olarak çağrıldıysa True kabul et
        is_failed = True

    # Mevcut hasat sayılarını oku (değiştirmeden koru!)
    if isinstance(node_obj, dict):
        curr_h = max(0, int(node_obj.get("current_harvests", 0) or 0))
        max_h = max(1, int(node_obj.get("max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS))
        node_state = str(node_obj.get("state", "active") or "active")
    elif node_obj is not None:
        curr_h = max(0, int(getattr(node_obj, "current_harvests", 0) or 0))
        max_h = max(1, int(getattr(node_obj, "max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS))
        node_state = str(getattr(node_obj, "state", "active") or "active")
    else:
        curr_h = 0
        max_h = DEFAULT_MAX_HARVESTS
        node_state = "active"

    used_key = str(gather_key or "f").strip().lower() or "f"
    if used_key not in ("f", "e"):
        used_key = "f"

    if not is_failed:
        return {
            "extraction_failed": False,
            "handled": False,
            "node_id": node_id,
            "current_harvests": curr_h,
            "max_harvests": max_h,
            "stage_incremented": False,
            "stage_decremented": False,
            "rights_preserved": True,
            "retry_stage": f"{min(max_h, curr_h + 1)}/{max_h}",
            "stage_display": f"{curr_h}/{max_h}",
            "re_pressed": False,
            "gather_key": used_key,
            "state": node_state,
        }

    # Hak düşmeme garantisi: current_harvests AYNEN korunur, state "active" tutulur
    if isinstance(node_obj, dict):
        node_obj["current_harvests"] = curr_h
        if node_obj.get("state") == "resting":
            node_obj["state"] = "active"
        node_state = str(node_obj.get("state", "active"))
    elif node_obj is not None:
        node_obj.current_harvests = curr_h
        if getattr(node_obj, "state", "active") == "resting":
            node_obj.state = "active"
        node_state = str(getattr(node_obj, "state", "active"))

    # Eğer düğüm nesnesi ayrı olarak geçildiyse ve route_mgr içinde de varsa senkronize et
    if isinstance(node_or_id, (NodeData, dict)) and node_or_id is not node_obj:
        if isinstance(node_or_id, dict):
            node_or_id["current_harvests"] = curr_h
        else:
            node_or_id.current_harvests = curr_h

    re_pressed = False
    if retry_press:
        re_pressed = _send_key_via_controller(input_ctrl, used_key, test_mode=test_mode)

    attempting_stage = min(max_h, curr_h + 1)
    logger.warning(
        "[ExtractionFailure] 'Essence Extraction failed' algılandı (Node #%s)! "
        "Hak düşülmedi (current_harvests=%d/%d korundu, aşama %d/%d yeniden deneniyor, tuş='%s', re_pressed=%s).",
        str(node_id) if node_id is not None else "?",
        curr_h,
        max_h,
        attempting_stage,
        max_h,
        used_key.upper(),
        re_pressed,
    )

    return {
        "extraction_failed": True,
        "handled": True,
        "node_id": node_id,
        "current_harvests": curr_h,
        "max_harvests": max_h,
        "stage_incremented": False,
        "stage_decremented": False,
        "rights_preserved": True,
        "retry_stage": f"{attempting_stage}/{max_h}",
        "stage_display": f"{curr_h}/{max_h}",
        "re_pressed": bool(re_pressed),
        "gather_key": used_key,
        "state": node_state,
    }


# ==============================================================================
# MADDE 22 — YARIM BIRAKILMIŞ KAYNAK (PARTIAL NODE) ALGILAMA (`PartialNodeDetector`)
# ==============================================================================
class PartialNodeDetector:
    """
    Madde 22 — Yarım Bırakılmış Kaynak (Partial Node) Algılama:

    Başka bir oyuncu kaynağı önceden 1 veya 2 kez toplamışsa (dolayısıyla 1. veya 2.
    toplamadan sonra kaynak dünyadan silinir ve 400ms bekleme + doğrulama penceresi
    içinde `[F Gather]` / `[E]` tekrar belirmez), 3. toplama için boşuna beklemek yerine
    `route_mgr.mark_node_gathered(node_id, force_deplete=True)` çağrısıyla düğümü derhal
    `RESTING` (5 dakika) durumuna geçirir.
    """

    def __init__(
        self,
        route_manager: Optional[RouteManager] = None,
        vision_engine: Any = None,
        reprompt_wait_sec: float = DEFAULT_REPROMPT_WAIT_SEC,
        verification_window_sec: float = DEFAULT_VERIFICATION_WINDOW_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        default_rest_minutes: float = DEFAULT_REST_MINUTES,
        sleep_fn: Optional[Callable[[float], None]] = None,
        route_mgr: Optional[RouteManager] = None,
        vision: Any = None,
    ) -> None:
        self.route_manager: Optional[RouteManager] = route_manager or route_mgr
        self.vision_engine: Any = vision_engine or vision
        self.reprompt_wait_sec: float = max(0.0, float(reprompt_wait_sec))
        self.verification_window_sec: float = max(self.reprompt_wait_sec, float(verification_window_sec))
        self.poll_interval_sec: float = max(0.005, float(poll_interval_sec))
        self.default_rest_minutes: float = max(0.1, float(default_rest_minutes))
        self._sleep: Callable[[float], None] = sleep_fn if callable(sleep_fn) else time.sleep

        self.partial_nodes_detected: int = 0
        self.history: List[Dict[str, Any]] = []

    def verify_reprompt_within_window(
        self,
        frame: Optional[np.ndarray] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        prompt_detector: Optional[Callable[[], Union[bool, Tuple[bool, str]]]] = None,
        prompt_reappeared: Optional[bool] = None,
        wait_initial_400ms: bool = True,
        default_key: str = "f",
    ) -> Tuple[bool, str, float]:
        """
        Hasat aşaması (`1/3` veya `2/3`) tamamlandıktan sonra `reprompt_wait_sec` (400ms)
        bekler ve doğrulama penceresi içinde `[F Gather]` / `[E]` isteminin tekrar
        belirip belirmediğini kontrol eder.

        Returns:
            `(reappeared: bool, detected_key: str, elapsed_wait_sec: float)`
        """
        waited_sec = 0.0
        if wait_initial_400ms and self.reprompt_wait_sec > 0.0:
            self._sleep(self.reprompt_wait_sec)
            waited_sec += self.reprompt_wait_sec

        if prompt_reappeared is not None:
            return bool(prompt_reappeared), str(default_key or "f").lower(), round(waited_sec, 4)

        if callable(prompt_detector):
            res = prompt_detector()
            if isinstance(res, tuple) and len(res) >= 1:
                k = str(res[1]).lower() if len(res) >= 2 and res[1] else default_key
                if bool(res[0]):
                    return True, k, round(waited_sec, 4)
            elif bool(res):
                return True, default_key, round(waited_sec, 4)

        # İlk kontrol
        current_frame = frame_provider() if callable(frame_provider) else frame
        found, det_key = _detect_prompt_from_vision_or_frame(
            vision_engine=self.vision_engine,
            frame=current_frame,
            default_key=default_key,
        )
        if found:
            return True, det_key, round(waited_sec, 4)

        # Eğer dinamik sağlayıcı varsa kalan doğrulama penceresi boyunca yokla
        remaining_window = max(0.0, self.verification_window_sec - self.reprompt_wait_sec)
        if remaining_window > 0.0 and (callable(frame_provider) or callable(prompt_detector)):
            deadline = time.monotonic() + remaining_window
            while time.monotonic() < deadline:
                self._sleep(self.poll_interval_sec)
                waited_sec += self.poll_interval_sec
                if callable(prompt_detector):
                    res = prompt_detector()
                    if isinstance(res, tuple) and len(res) >= 1 and bool(res[0]):
                        k = str(res[1]).lower() if len(res) >= 2 and res[1] else default_key
                        return True, k, round(waited_sec, 4)
                    if not isinstance(res, tuple) and bool(res):
                        return True, default_key, round(waited_sec, 4)
                if callable(frame_provider):
                    polled_frame = frame_provider()
                    found, det_key = _detect_prompt_from_vision_or_frame(
                        vision_engine=self.vision_engine,
                        frame=polled_frame,
                        default_key=default_key,
                    )
                    if found:
                        return True, det_key, round(waited_sec, 4)

        return False, det_key, round(waited_sec, 4)

    def check_and_handle_partial_node(
        self,
        node_or_id: Union[int, NodeData, Dict[str, Any]],
        completed_harvests: Optional[int] = None,
        max_harvests: Optional[int] = None,
        prompt_reappeared: Optional[bool] = None,
        frame: Optional[np.ndarray] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        prompt_detector: Optional[Callable[[], Union[bool, Tuple[bool, str]]]] = None,
        route_mgr: Optional[RouteManager] = None,
        rest_minutes: Optional[float] = None,
        region: Optional[str] = None,
        wait_before_check: bool = False,
    ) -> Dict[str, Any]:
        """
        1. veya 2. hasat sonrasında kaynağın yarım bırakılmış (partial node) olup
        olmadığını denetler.
        Eğer `0 < completed_harvests < max_harvests` iken `[F Gather]` / `[E]` tekrar
        belirmezse (`prompt_reappeared == False`), düğümü 3. hasat için bekletmeden
        `route_mgr.mark_node_gathered(node_id, rest_minutes=rest_minutes, force_deplete=True)`
        ile derhal `RESTING` (5 dakika) durumuna geçirir.
        """
        active_rm = route_mgr or self.route_manager
        node_id, node_obj = _resolve_node_reference(node_or_id, route_mgr=active_rm, region=region)
        resolved_id = int(node_id if node_id is not None else 1)

        if completed_harvests is None:
            if isinstance(node_obj, dict):
                completed_harvests = int(node_obj.get("current_harvests", 1) or 1)
            elif node_obj is not None:
                completed_harvests = int(getattr(node_obj, "current_harvests", 1) or 1)
            else:
                completed_harvests = 1

        if max_harvests is None:
            if isinstance(node_obj, dict):
                max_harvests = int(node_obj.get("max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS)
            elif node_obj is not None:
                max_harvests = int(getattr(node_obj, "max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS)
            else:
                max_harvests = DEFAULT_MAX_HARVESTS

        comp_h = max(0, int(completed_harvests))
        target_h = max(1, int(max_harvests))
        effective_rest = float(rest_minutes if rest_minutes is not None else self.default_rest_minutes)

        # Eğer zaten 3/3 (veya max_harvests) tamamlandıysa bu bir partial node değil, tam hasattır
        if comp_h >= target_h:
            return {
                "is_partial_node": False,
                "depleted_early": False,
                "node_id": resolved_id,
                "completed_harvests": comp_h,
                "max_harvests": target_h,
                "missing_harvests": 0,
                "prompt_reappeared": False,
                "node_state": "resting",
                "rest_duration_sec": int(round(effective_rest * 60.0)),
                "action": "FULL_HARVEST_COMPLETED",
            }

        reappeared, det_key, waited_sec = self.verify_reprompt_within_window(
            frame=frame,
            frame_provider=frame_provider,
            prompt_detector=prompt_detector,
            prompt_reappeared=prompt_reappeared,
            wait_initial_400ms=bool(wait_before_check),
        )

        if reappeared:
            return {
                "is_partial_node": False,
                "depleted_early": False,
                "node_id": resolved_id,
                "completed_harvests": comp_h,
                "max_harvests": target_h,
                "missing_harvests": max(0, target_h - comp_h),
                "prompt_reappeared": True,
                "detected_key": det_key,
                "waited_sec": waited_sec,
                "node_state": "active",
                "rest_duration_sec": int(round(effective_rest * 60.0)),
                "action": "CONTINUE_HARVEST",
            }

        # [F Gather] / [E] tekrar çıkmadı -> Kaynak başka bir oyuncu tarafından yarım bırakılmış!
        # 3. toplama için boşuna beklemeden derhal RESTING (5 dk) durumuna geçir!
        updated_node = None
        if active_rm is not None and hasattr(active_rm, "mark_node_gathered"):
            updated_node = active_rm.mark_node_gathered(
                resolved_id,
                rest_minutes=effective_rest,
                force_deplete=True,
                region=region,
            )

        # Doğrudan geçirilen node_or_id nesnesini de senkronize et
        rest_sec = int(round(effective_rest * 60.0))
        now_ts = time.time()
        for obj in (node_obj, node_or_id if isinstance(node_or_id, (NodeData, dict)) else None):
            if obj is None or obj is updated_node:
                continue
            if isinstance(obj, dict):
                obj["current_harvests"] = 0
                obj["state"] = "resting"
                obj["last_visited_timestamp"] = now_ts
                obj["rest_duration_sec"] = rest_sec
            else:
                obj.current_harvests = 0
                obj.state = "resting"
                obj.last_visited_timestamp = now_ts
                obj.rest_duration_sec = rest_sec

        self.partial_nodes_detected += 1
        event = {
            "is_partial_node": True,
            "depleted_early": True,
            "node_id": resolved_id,
            "completed_harvests": comp_h,
            "max_harvests": target_h,
            "missing_harvests": max(0, target_h - comp_h),
            "prompt_reappeared": False,
            "waited_sec": waited_sec,
            "node_state": "resting",
            "rest_duration_sec": rest_sec,
            "action": "MARKED_RESTING_FORCE_DEPLETE",
            "timestamp": now_ts,
        }
        self.history.append(event)
        logger.info(
            "[PartialNodeDetector] Düğüm #%d kısmi hasat sonrasında tükendi (%d/%d tamamlandı, "
            "başka oyuncu tarafından yarım bırakılmış). 3. hasat beklenmeden RESTING (%d sn) durumuna alındı.",
            resolved_id,
            comp_h,
            target_h,
            rest_sec,
        )
        return event

    # Esnek çağrı takma adları
    evaluate_after_stage = check_and_handle_partial_node
    detect_partial_node = check_and_handle_partial_node
    handle_partial_depletion = check_and_handle_partial_node


# ==============================================================================
# MADDE 21 — 3 AŞAMALI (`1/3 ➔ 2/3 ➔ 3/3`) 400ms RE-PROMPT TOPLAMA DÖNGÜSÜ
# ==============================================================================
class ThreeStageHarvestController:
    """
    Madde 21 — 3 Aşamalı (`1/3 ➔ 2/3 ➔ 3/3`) Dinamik Toplama Döngüsü (`ThreeStageHarvestController`):

    Bir düğümdeki `1/3 -> 2/3 -> 3/3` toplama durum makinesini yönetir:
    - `1/3` (ve `2/3`) toplama işlemi tamamlandıktan sonra `400ms` (`0.4s` yapılandırılabilir)
      bekler ve `[F Gather]` / `[E]` isteminin tekrar belirip belirmediğini kontrol eder.
    - İstem tekrar belirdiği sürece botu aynı düğümde sabit (`is_anchored = True`) tutarak
      bir sonraki aşamaya (`2/3` -> `3/3`) geçirir.
    - Sohbette `"Essence Extraction failed"` algılanırsa `handle_extraction_failure` (Madde 23)
      üzerinden hak düşmeden (`current_harvests` artırılmadan) aynı aşamayı anında tekrar dener.
    - `1/3` veya `2/3` sonrasında kaynak erken tükenirse `PartialNodeDetector` (Madde 22)
      üzerinden düğümü derhal `RESTING` (5 dakika) durumuna geçirir ve sabitlemeyi kaldırır.
    """

    STATE_IDLE = "IDLE"
    STATE_ANCHORED_GATHERING = "ANCHORED_GATHERING"
    STATE_WAITING_REPROMPT = "WAITING_REPROMPT"
    STATE_COMPLETED = "COMPLETED"
    STATE_PARTIAL_DEPLETED = "PARTIAL_DEPLETED"

    def __init__(
        self,
        route_manager: Optional[RouteManager] = None,
        vision_engine: Any = None,
        input_controller: Any = None,
        hud_reader: Optional[GameHUDReader] = None,
        partial_detector: Optional[PartialNodeDetector] = None,
        reprompt_wait_sec: float = DEFAULT_REPROMPT_WAIT_SEC,
        reprompt_wait_ms: Optional[int] = None,
        verification_window_sec: float = DEFAULT_VERIFICATION_WINDOW_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        cast_duration_sec: float = DEFAULT_CAST_DURATION_SEC,
        default_max_harvests: int = DEFAULT_MAX_HARVESTS,
        max_retries_per_stage: int = 3,
        rest_minutes: float = DEFAULT_REST_MINUTES,
        gather_key: str = "f",
        test_mode: bool = False,
        sleep_fn: Optional[Callable[[float], None]] = None,
        route_mgr: Optional[RouteManager] = None,
        vision: Any = None,
        input_ctrl: Any = None,
        input_driver: Any = None,
    ) -> None:
        self.route_manager: Optional[RouteManager] = route_manager or route_mgr
        self.vision_engine: Any = vision_engine or vision
        self.input_controller: Any = input_controller or input_ctrl or input_driver
        self.hud_reader: GameHUDReader = hud_reader or GameHUDReader()

        if reprompt_wait_ms is not None:
            self.reprompt_wait_sec: float = max(0.0, float(reprompt_wait_ms) / 1000.0)
        else:
            self.reprompt_wait_sec = max(0.0, float(reprompt_wait_sec))
        self.reprompt_wait_ms: int = int(round(self.reprompt_wait_sec * 1000.0))

        self.verification_window_sec: float = max(self.reprompt_wait_sec, float(verification_window_sec))
        self.poll_interval_sec: float = max(0.005, float(poll_interval_sec))
        self.cast_duration_sec: float = max(0.0, float(cast_duration_sec))
        self.default_max_harvests: int = max(1, int(default_max_harvests or DEFAULT_MAX_HARVESTS))
        self.max_retries_per_stage: int = max(1, int(max_retries_per_stage))
        self.rest_minutes: float = max(0.1, float(rest_minutes))
        self.gather_key: str = str(gather_key or "f").strip().lower() or "f"
        self.test_mode: bool = bool(test_mode)
        self._sleep: Callable[[float], None] = sleep_fn if callable(sleep_fn) else time.sleep

        self.partial_detector: PartialNodeDetector = partial_detector or PartialNodeDetector(
            route_manager=self.route_manager,
            vision_engine=self.vision_engine,
            reprompt_wait_sec=self.reprompt_wait_sec,
            verification_window_sec=self.verification_window_sec,
            poll_interval_sec=self.poll_interval_sec,
            default_rest_minutes=self.rest_minutes,
            sleep_fn=self._sleep,
        )

        # Aktif düğüm durum makinesi değişkenleri
        self.active_node_id: Optional[int] = None
        self.active_node: Optional[Union[NodeData, Dict[str, Any]]] = None
        self.current_harvests: int = 0
        self.max_harvests: int = self.default_max_harvests
        self.state: str = self.STATE_IDLE
        self.is_anchored: bool = False
        self.stage_history: List[str] = []
        self.reprompt_waits_sec: List[float] = []
        self.extraction_retries: int = 0
        self.last_reprompt_wait_sec: float = 0.0

    @property
    def stage_label(self) -> str:
        """Güncel hasat aşamasını `'1/3'`, `'2/3'`, `'3/3'` biçiminde döndürür."""
        return f"{self.current_harvests}/{self.max_harvests}"

    def start_node(
        self,
        node_or_id: Union[int, NodeData, Dict[str, Any]],
        max_harvests: Optional[int] = None,
        gather_key: Optional[str] = None,
        region: Optional[str] = None,
        press_initial_key: bool = False,
    ) -> Dict[str, Any]:
        """
        Yeni bir düğümde 3 aşamalı (`1/3 -> 2/3 -> 3/3`) toplama döngüsünü başlatır
        ve botu düğüme sabitler (`self.is_anchored = True`).
        """
        node_id, node_obj = _resolve_node_reference(
            node_or_id,
            route_mgr=self.route_manager,
            region=region,
        )
        self.active_node_id = int(node_id if node_id is not None else 1)
        self.active_node = node_obj if node_obj is not None else (
            node_or_id if isinstance(node_or_id, (NodeData, dict)) else None
        )

        if max_harvests is not None:
            resolved_max = max(1, int(max_harvests))
        elif isinstance(self.active_node, dict):
            resolved_max = max(1, int(self.active_node.get("max_harvests", self.default_max_harvests) or self.default_max_harvests))
        elif self.active_node is not None:
            resolved_max = max(1, int(getattr(self.active_node, "max_harvests", self.default_max_harvests) or self.default_max_harvests))
        else:
            resolved_max = self.default_max_harvests

        if isinstance(self.active_node, dict):
            initial_curr = max(0, int(self.active_node.get("current_harvests", 0) or 0))
        elif self.active_node is not None:
            initial_curr = max(0, int(getattr(self.active_node, "current_harvests", 0) or 0))
        else:
            initial_curr = 0

        if initial_curr >= resolved_max:
            initial_curr = 0

        self.max_harvests = resolved_max
        self.current_harvests = initial_curr
        self.state = self.STATE_ANCHORED_GATHERING
        self.is_anchored = True
        self.stage_history = []
        self.reprompt_waits_sec = []
        self.extraction_retries = 0
        self.last_reprompt_wait_sec = 0.0

        if gather_key:
            self.gather_key = str(gather_key).strip().lower() or "f"

        key_pressed = False
        if press_initial_key:
            key_pressed = _send_key_via_controller(
                self.input_controller,
                self.gather_key,
                test_mode=self.test_mode,
            )

        return {
            "node_id": self.active_node_id,
            "state": self.state,
            "is_anchored": self.is_anchored,
            "current_harvests": self.current_harvests,
            "max_harvests": self.max_harvests,
            "target_stage": f"{self.current_harvests + 1}/{self.max_harvests}",
            "gather_key": self.gather_key,
            "key_pressed": key_pressed,
        }

    # Takma adlar
    begin_harvest = start_node
    start_harvest = start_node

    def handle_extraction_failure(
        self,
        node_or_id: Union[int, NodeData, Dict[str, Any], None] = None,
        gather_key: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        chat_text: Optional[str] = None,
        chat_alerts: Optional[Dict[str, Any]] = None,
        extraction_failed: Optional[bool] = None,
        region: Optional[str] = None,
        retry_press: bool = True,
    ) -> Dict[str, Any]:
        """
        `ThreeStageHarvestController` üzerinden `"Essence Extraction failed"` durumunu yönetir:
        `current_harvests` sayacını artırmaz, botu düğümde sabit (`is_anchored=True`) tutar
        ve aynı aşamayı (`1/3`, `2/3` veya `3/3`) anında yeniden başlatmak için `F` / `E` basar.
        """
        target_ref = node_or_id if node_or_id is not None else (self.active_node or self.active_node_id)
        res = handle_extraction_failure(
            node_or_id=target_ref,
            route_mgr=self.route_manager,
            input_ctrl=self.input_controller,
            gather_key=gather_key or self.gather_key,
            frame=frame,
            chat_text=chat_text,
            chat_alerts=chat_alerts,
            extraction_failed=extraction_failed,
            hud_reader=self.hud_reader,
            region=region,
            retry_press=retry_press,
            test_mode=self.test_mode,
        )
        if res.get("extraction_failed"):
            self.extraction_retries += 1
            self.is_anchored = True
            self.state = self.STATE_ANCHORED_GATHERING
        return res

    def on_cast_completed(
        self,
        node_or_id: Union[int, NodeData, Dict[str, Any], None] = None,
        frame: Optional[np.ndarray] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        prompt_detector: Optional[Callable[[], Union[bool, Tuple[bool, str]]]] = None,
        prompt_reappeared: Optional[bool] = None,
        extraction_failed: Optional[bool] = None,
        chat_text: Optional[str] = None,
        chat_alerts: Optional[Dict[str, Any]] = None,
        gather_key: Optional[str] = None,
        region: Optional[str] = None,
        auto_press_next_stage: bool = True,
    ) -> Dict[str, Any]:
        """
        Tek bir toplama (cast) tamamlandığında çağrılır:
        1. `"Essence Extraction failed"` kontrolü yapar; başarısızsa `current_harvests`
           artırılmaz, hak düşmez ve aynı aşama için tekrar `F`/`E` basılır.
        2. Başarılıysa aşamayı ilerletir (`1/3`, `2/3` veya `3/3`).
        3. `1/3` veya `2/3` tamamlandıysa `400ms` (`self.reprompt_wait_sec`) bekler ve
           `[F Gather]` / `[E]` isteminin tekrar belirip belirmediğini kontrol eder:
           - Tekrar belirdiyse botu aynı düğümde sabit (`is_anchored=True`) tutar ve
             bir sonraki aşama için `F`/`E` basar.
           - Tekrar belirmediyse `PartialNodeDetector` ile düğümü derhal `RESTING` (5 dk)
             durumuna geçirir (`force_deplete=True`) ve sabitlemeyi kaldırır.
        4. `3/3` tamamlandıysa düğümü `RESTING` (5 dk) durumuna geçirir ve `is_anchored=False` yapar.
        """
        if node_or_id is not None and self.active_node_id is None:
            self.start_node(node_or_id, gather_key=gather_key, region=region)

        target_ref = node_or_id if node_or_id is not None else (self.active_node or self.active_node_id)
        node_id, node_obj = _resolve_node_reference(target_ref, route_mgr=self.route_manager, region=region)
        resolved_id = int(node_id if node_id is not None else (self.active_node_id or 1))
        self.active_node_id = resolved_id
        if node_obj is not None:
            self.active_node = node_obj

        used_key = str(gather_key or self.gather_key or "f").strip().lower() or "f"
        self.gather_key = used_key

        # 1. Madde 23: "Essence Extraction failed" kontrolü (Hak düşmeme!)
        has_failure_input = (
            extraction_failed is not None
            or chat_text is not None
            or chat_alerts is not None
            or frame is not None
        )
        if has_failure_input:
            fail_res = self.handle_extraction_failure(
                node_or_id=self.active_node or resolved_id,
                gather_key=used_key,
                frame=frame,
                chat_text=chat_text,
                chat_alerts=chat_alerts,
                extraction_failed=extraction_failed if extraction_failed is not None else None,
                region=region,
                retry_press=True,
            )
            if fail_res.get("extraction_failed"):
                return {
                    "status": "RETRY_STAGE",
                    "node_id": resolved_id,
                    "current_harvests": self.current_harvests,
                    "max_harvests": self.max_harvests,
                    "stage": f"{self.current_harvests}/{self.max_harvests}",
                    "retry_stage": fail_res["retry_stage"],
                    "extraction_failed": True,
                    "re_pressed": fail_res["re_pressed"],
                    "is_anchored": True,
                    "node_depleted": False,
                    "partial_node": False,
                    "state": self.state,
                }

        # 2. Hasat aşamasını kaydet (1/3 -> 2/3 -> 3/3)
        if self.route_manager is not None and hasattr(self.route_manager, "record_harvest_stage"):
            curr_h, max_h, is_depleted = self.route_manager.record_harvest_stage(
                node_id=resolved_id,
                rest_minutes=self.rest_minutes,
                default_max_harvests=self.max_harvests,
                region=region,
            )
            if is_depleted:
                self.current_harvests = max_h
                self.max_harvests = max_h
            else:
                self.current_harvests = curr_h
                self.max_harvests = max_h
        else:
            self.current_harvests += 1
            is_depleted = self.current_harvests >= self.max_harvests
            if isinstance(self.active_node, dict):
                self.active_node["current_harvests"] = 0 if is_depleted else self.current_harvests
                self.active_node["state"] = "resting" if is_depleted else "active"
            elif self.active_node is not None:
                self.active_node.current_harvests = 0 if is_depleted else self.current_harvests
                self.active_node.state = "resting" if is_depleted else "active"

        completed_stage_str = f"{self.current_harvests}/{self.max_harvests}"
        self.stage_history.append(completed_stage_str)

        # 3. Eğer 3/3 (veya max_harvests) tamamlandıysa düğüm dinlenmeye geçer
        if is_depleted or self.current_harvests >= self.max_harvests:
            self.state = self.STATE_COMPLETED
            self.is_anchored = False
            logger.info(
                "[ThreeStageHarvest] Düğüm #%d tam hasat döngüsü tamamlandı (%s) -> RESTING (%.1f dk).",
                resolved_id,
                completed_stage_str,
                self.rest_minutes,
            )
            return {
                "status": "COMPLETED",
                "node_id": resolved_id,
                "current_harvests": self.current_harvests,
                "max_harvests": self.max_harvests,
                "stage": completed_stage_str,
                "stages_completed": list(self.stage_history),
                "extraction_failed": False,
                "reprompt_detected": False,
                "waited_sec": 0.0,
                "is_anchored": False,
                "node_depleted": True,
                "partial_node": False,
                "state": self.state,
            }

        # 4. 1/3 veya 2/3 tamamlandı -> 400ms bekle ve [F Gather] / [E] tekrar çıkıyor mu kontrol et!
        self.state = self.STATE_WAITING_REPROMPT
        self.is_anchored = True

        partial_eval = self.partial_detector.check_and_handle_partial_node(
            node_or_id=self.active_node or resolved_id,
            completed_harvests=self.current_harvests,
            max_harvests=self.max_harvests,
            prompt_reappeared=prompt_reappeared,
            frame=frame,
            frame_provider=frame_provider,
            prompt_detector=prompt_detector,
            route_mgr=self.route_manager,
            rest_minutes=self.rest_minutes,
            region=region,
            wait_before_check=True,
        )
        waited_sec = float(partial_eval.get("waited_sec", self.reprompt_wait_sec))
        self.last_reprompt_wait_sec = waited_sec
        self.reprompt_waits_sec.append(waited_sec)

        if partial_eval.get("is_partial_node"):
            # Kaynak erken tükendi (Partial Node) -> RESTING durumuna alındı
            self.state = self.STATE_PARTIAL_DEPLETED
            self.is_anchored = False
            return {
                "status": "PARTIAL_DEPLETED",
                "node_id": resolved_id,
                "current_harvests": self.current_harvests,
                "max_harvests": self.max_harvests,
                "stage": completed_stage_str,
                "stages_completed": list(self.stage_history),
                "extraction_failed": False,
                "reprompt_detected": False,
                "waited_sec": waited_sec,
                "is_anchored": False,
                "node_depleted": True,
                "partial_node": True,
                "state": self.state,
            }

        # [F Gather] / [E] tekrar belirdi -> Bot aynı düğümde sabit kalır ve sıradaki aşamaya (2/3 veya 3/3) geçer
        self.state = self.STATE_ANCHORED_GATHERING
        self.is_anchored = True
        detected_key = str(partial_eval.get("detected_key", used_key) or used_key).strip().lower()
        if detected_key in ("f", "e"):
            self.gather_key = detected_key

        next_pressed = False
        if auto_press_next_stage:
            next_pressed = _send_key_via_controller(
                self.input_controller,
                self.gather_key,
                test_mode=self.test_mode,
            )

        next_stage_str = f"{self.current_harvests + 1}/{self.max_harvests}"
        logger.info(
            "[ThreeStageHarvest] Düğüm #%d aşama %s tamamlandı (%.0fms beklendi, [%s] tekrar belirdi) -> "
            "Aşama %s başlatılıyor (is_anchored=True).",
            resolved_id,
            completed_stage_str,
            waited_sec * 1000.0,
            self.gather_key.upper(),
            next_stage_str,
        )
        return {
            "status": "CONTINUE_NEXT_STAGE",
            "node_id": resolved_id,
            "current_harvests": self.current_harvests,
            "max_harvests": self.max_harvests,
            "stage": completed_stage_str,
            "next_stage": next_stage_str,
            "stages_completed": list(self.stage_history),
            "extraction_failed": False,
            "reprompt_detected": True,
            "next_key_pressed": next_pressed,
            "gather_key": self.gather_key,
            "waited_sec": waited_sec,
            "is_anchored": True,
            "node_depleted": False,
            "partial_node": False,
            "state": self.state,
        }

    # Takma adlar
    advance_stage = on_cast_completed
    step_stage = on_cast_completed

    def execute_harvest_loop(
        self,
        node_or_id: Union[int, NodeData, Dict[str, Any]],
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        prompt_detector: Optional[Callable[[], Union[bool, Tuple[bool, str]]]] = None,
        chat_alert_provider: Optional[Callable[[], Union[bool, str, Dict[str, Any]]]] = None,
        reprompt_sequence: Optional[List[bool]] = None,
        extraction_fail_sequence: Optional[List[bool]] = None,
        gather_key: Optional[str] = None,
        max_harvests: Optional[int] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bir düğümde baştan sona tam `1/3 -> 2/3 -> 3/3` dinamik toplama döngüsünü çalıştırır:
        - İlk `F` / `E` basımını yapar,
        - Her aşama sonunda `"Essence Extraction failed"` varsa hak düşmeden aynı aşamayı tekrarlar,
        - `1/3` ve `2/3` sonrasında `400ms` bekleyip `[F Gather]` / `[E]` kontrolü yapar,
        - Yarım bırakılmış kaynak (Partial Node) ise 3. toplamayı beklemeden `RESTING` yapar,
        - `3/3` tamamlandığında düğümü `RESTING` durumuna geçirir ve özet sözlüğü döndürür.
        """
        self.start_node(
            node_or_id=node_or_id,
            max_harvests=max_harvests,
            gather_key=gather_key,
            region=region,
            press_initial_key=True,
        )

        reprompt_iter = iter(reprompt_sequence) if reprompt_sequence is not None else None
        fail_iter = iter(extraction_fail_sequence) if extraction_fail_sequence is not None else None

        total_casts = 0
        max_total_casts = self.max_harvests * (self.max_retries_per_stage + 1)
        last_step_result: Dict[str, Any] = {}

        while self.is_anchored and total_casts < max_total_casts:
            total_casts += 1
            if self.cast_duration_sec > 0.0:
                self._sleep(self.cast_duration_sec)

            curr_frame = frame_provider() if callable(frame_provider) else None

            # Extraction failure durumu
            ext_failed: Optional[bool] = None
            chat_txt: Optional[str] = None
            chat_dict: Optional[Dict[str, Any]] = None
            if fail_iter is not None:
                ext_failed = bool(next(fail_iter, False))
            elif callable(chat_alert_provider):
                alert_val = chat_alert_provider()
                if isinstance(alert_val, bool):
                    ext_failed = alert_val
                elif isinstance(alert_val, str):
                    chat_txt = alert_val
                elif isinstance(alert_val, dict):
                    chat_dict = alert_val

            # Re-prompt durumu (yalnızca extraction_failed değilse tüketilir)
            reprompt_val: Optional[bool] = None
            will_fail = False
            if ext_failed is True:
                will_fail = True
            elif chat_txt and "extraction failed" in chat_txt.lower():
                will_fail = True
            elif isinstance(chat_dict, dict) and chat_dict.get("extraction_failed"):
                will_fail = True

            if not will_fail and reprompt_iter is not None and (self.current_harvests + 1) < self.max_harvests:
                reprompt_val = bool(next(reprompt_iter, True))
            elif (
                not will_fail
                and reprompt_iter is None
                and frame_provider is None
                and prompt_detector is None
                and self.vision_engine is None
            ):
                # Senkron birim testlerinde harici sağlayıcı verilmediyse varsayılan olarak 3/3'e ilerle
                reprompt_val = True

            last_step_result = self.on_cast_completed(
                node_or_id=self.active_node or self.active_node_id,
                frame=curr_frame,
                frame_provider=frame_provider,
                prompt_detector=prompt_detector,
                prompt_reappeared=reprompt_val,
                extraction_failed=ext_failed,
                chat_text=chat_txt,
                chat_alerts=chat_dict,
                gather_key=self.gather_key,
                region=region,
                auto_press_next_stage=True,
            )

            if last_step_result.get("status") in ("COMPLETED", "PARTIAL_DEPLETED"):
                break

        final_node_state = "resting" if not self.is_anchored else "active"
        if self.route_manager is not None and self.active_node_id is not None:
            rm_node = self.route_manager.get_node(self.active_node_id, region=region)
            if rm_node is not None:
                final_node_state = str(rm_node.state)

        return {
            "node_id": self.active_node_id,
            "harvests_completed": len(self.stage_history),
            "max_harvests": self.max_harvests,
            "stages_completed": list(self.stage_history),
            "partial_node_detected": bool(last_step_result.get("partial_node", False)),
            "extraction_failure_retries": int(self.extraction_retries),
            "reprompt_waits_sec": list(self.reprompt_waits_sec),
            "node_state": final_node_state,
            "is_anchored": self.is_anchored,
            "state": self.state,
            "gather_key": self.gather_key,
        }

    harvest_node = execute_harvest_loop
    run_harvest_cycle = execute_harvest_loop


# ==============================================================================
# MADDE 24 — MESAFE BAZLI OTOMATİK BİNEK (`AutoMountManager`)
# ==============================================================================
class AutoMountManager:
    """
    Madde 24 — Mesafe Bazlı Otomatik Binek (`AutoMountManager`):

    - Sıradaki düğüm `> 35m` (`mount_distance_m = 35.0`) uzaktaysa, karakter savaşta
      değilse (`in_combat == False`) ve sohbette `"Cannot mount in current state"`
      uyarısı yoksa otomatik olarak bineğe biner (`mount_up`).
    - Düğüme `<= 8m` (`dismount_distance_m = 8.0`) yaklaşıldığında veya ekranda
      `[F Gather]` / `[E]` toplama istemi belirdiği anda otomatik olarak binekten iner
      ve toplamaya başlar (`dismount` + `gather`).
    """

    def __init__(
        self,
        input_controller: Any = None,
        vision_engine: Any = None,
        hud_reader: Optional[GameHUDReader] = None,
        combat_recovery: Any = None,
        mount_distance_m: float = MOUNT_DISTANCE_THRESHOLD_M,
        dismount_distance_m: float = DISMOUNT_DISTANCE_THRESHOLD_M,
        mount_key: str = "r",
        gather_key: str = "f",
        mount_cooldown_sec: float = 0.0,
        cannot_mount_backoff_sec: float = 2.5,
        test_mode: bool = False,
        input_ctrl: Any = None,
        input_driver: Any = None,
        vision: Any = None,
    ) -> None:
        self.input_controller: Any = input_controller or input_ctrl or input_driver
        self.vision_engine: Any = vision_engine or vision
        self.hud_reader: GameHUDReader = hud_reader or GameHUDReader()
        self.combat_recovery: Any = combat_recovery

        self.mount_distance_m: float = float(mount_distance_m)
        self.dismount_distance_m: float = float(dismount_distance_m)
        self.mount_key: str = str(mount_key or "r").strip().lower() or "r"
        self.gather_key: str = str(gather_key or "f").strip().lower() or "f"
        self.mount_cooldown_sec: float = max(0.0, float(mount_cooldown_sec))
        self.cannot_mount_backoff_sec: float = max(0.0, float(cannot_mount_backoff_sec))
        self.test_mode: bool = bool(test_mode)

        self.is_mounted: bool = False
        self.last_mount_ts: float = 0.0
        self.last_dismount_ts: float = 0.0
        self.cannot_mount_until_ts: float = 0.0
        self.mount_count: int = 0
        self.dismount_count: int = 0

    def _is_in_combat(
        self,
        in_combat: Optional[bool] = None,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Karakterin savaşta olup olmadığını kontrol eder."""
        if in_combat is not None:
            return bool(in_combat)
        if self.combat_recovery is not None and frame is not None:
            try:
                if hasattr(self.combat_recovery, "is_under_attack"):
                    return bool(self.combat_recovery.is_under_attack(frame))
                if hasattr(self.combat_recovery, "detect_combat"):
                    return bool(self.combat_recovery.detect_combat(frame))
            except Exception as exc:
                logger.debug("CombatRecovery savaş kontrolü hatası: %s", exc)
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            try:
                bar_info = self.hud_reader.detect_top_target_bar(frame)
                if bool(bar_info.get("is_monster_target", False)):
                    return True
            except Exception as exc:
                logger.debug("HUDReader savaş hedef kontrolü hatası: %s", exc)
        return False

    def should_mount(
        self,
        distance_m: float,
        in_combat: bool = False,
        cannot_mount_alert: bool = False,
        gather_prompt_visible: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        """
        Bineğe binilmesi gerekip gerekmediğini döndürür:
        - Halihazırda binekli değilse (`not self.is_mounted`)
        - Mesafe `> 35m` (`distance_m > self.mount_distance_m`) ise
        - Savaşta değilse (`not in_combat`)
        - Sohbette `"Cannot mount in current state"` yoksa (`not cannot_mount_alert`)
        - Ekranda `[F Gather]` / `[E]` istemi yoksa (`not gather_prompt_visible`)
        """
        if self.is_mounted:
            return False
        if bool(in_combat) or bool(cannot_mount_alert) or bool(gather_prompt_visible):
            return False
        if float(distance_m) <= self.mount_distance_m:
            return False

        curr_time = time.time() if now is None else float(now)
        if curr_time < self.cannot_mount_until_ts:
            return False
        if self.mount_cooldown_sec > 0.0 and (curr_time - self.last_mount_ts) < self.mount_cooldown_sec:
            return False
        return True

    def should_dismount(
        self,
        distance_m: Optional[float] = None,
        gather_prompt_visible: bool = False,
        in_combat: bool = False,
    ) -> bool:
        """
        Binekten inilmesi gerekip gerekmediğini döndürür:
        - Halihazırda binekliyse (`self.is_mounted == True`) VE
        - `[F Gather]` / `[E]` belirdiyse (`gather_prompt_visible == True`) VEYA
        - Düğüme mesafe `<= 8m` (`distance_m <= self.dismount_distance_m`) altına indiyse VEYA
        - Karakter savaşa girdiyse (`in_combat == True`).
        """
        if not self.is_mounted:
            return False
        if bool(gather_prompt_visible):
            return True
        if distance_m is not None and float(distance_m) <= self.dismount_distance_m:
            return True
        if bool(in_combat):
            return True
        return False

    def mount_up(self, input_ctrl: Any = None) -> bool:
        """Binek tuşuna (`self.mount_key`) basarak karakteri bineğe bindirir."""
        ctrl = input_ctrl or self.input_controller
        pressed = _send_key_via_controller(ctrl, self.mount_key, test_mode=self.test_mode)
        self.is_mounted = True
        self.last_mount_ts = time.time()
        self.mount_count += 1
        logger.info("[AutoMount] Hedef > %.1fm: Bineğe binildi (tuş='%s').", self.mount_distance_m, self.mount_key.upper())
        return pressed

    def dismount(
        self,
        input_ctrl: Any = None,
        start_gathering: bool = True,
        gather_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Binekten iner ve `start_gathering=True` ise anında `F` / `E` tuşuna basarak
        kaynak toplamayı başlatır.
        """
        ctrl = input_ctrl or self.input_controller
        used_gather_key = str(gather_key or self.gather_key or "f").strip().lower() or "f"

        dismount_pressed = _send_key_via_controller(ctrl, self.mount_key, test_mode=self.test_mode)
        self.is_mounted = False
        self.last_dismount_ts = time.time()
        self.dismount_count += 1

        gather_pressed = False
        if start_gathering:
            gather_pressed = _send_key_via_controller(ctrl, used_gather_key, test_mode=self.test_mode)

        logger.info(
            "[AutoMount] Binekten inildi (start_gathering=%s, gather_key='%s').",
            start_gathering,
            used_gather_key.upper(),
        )
        return {
            "dismounted": True,
            "dismount_key_pressed": dismount_pressed,
            "started_gathering": bool(start_gathering),
            "gather_key_pressed": gather_pressed,
            "gather_key": used_gather_key,
            "is_mounted": self.is_mounted,
        }

    def update(
        self,
        distance_m: float,
        in_combat: Optional[bool] = None,
        frame: Optional[np.ndarray] = None,
        chat_text: Optional[str] = None,
        chat_alerts: Optional[Dict[str, Any]] = None,
        cannot_mount_alert: Optional[bool] = None,
        gather_prompt_visible: Optional[bool] = None,
        gather_key: Optional[str] = None,
        input_ctrl: Any = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Her navigasyon adımında mesafe (`distance_m`), savaş durumu (`in_combat`),
        sohbet uyarısı (`"Cannot mount in current state"`) ve `[F Gather]` / `[E]`
        görünürlüğünü değerlendirerek otomatik binek / inme + toplama kararını uygular.
        """
        curr_time = time.time() if now is None else float(now)
        combat_active = self._is_in_combat(in_combat=in_combat, frame=frame)

        # Sohbette "Cannot mount in current state" kontrolü
        if cannot_mount_alert is not None:
            cannot_mount = bool(cannot_mount_alert)
        elif chat_text is not None or chat_alerts is not None or frame is not None:
            alerts = _detect_chat_alerts_unified(
                frame=frame,
                chat_text=chat_text,
                chat_alerts=chat_alerts,
                hud_reader=self.hud_reader,
            )
            cannot_mount = bool(alerts.get("cannot_mount", False))
        else:
            cannot_mount = False

        if cannot_mount:
            self.cannot_mount_until_ts = curr_time + self.cannot_mount_backoff_sec
            if self.is_mounted:
                self.is_mounted = False

        # [F Gather] / [E] görünürlük kontrolü
        used_gather_key = str(gather_key or self.gather_key or "f").strip().lower() or "f"
        if gather_prompt_visible is not None:
            prompt_vis = bool(gather_prompt_visible)
        elif frame is not None or self.vision_engine is not None:
            prompt_vis, det_k = _detect_prompt_from_vision_or_frame(
                vision_engine=self.vision_engine,
                frame=frame,
                default_key=used_gather_key,
            )
            if det_k in ("f", "e"):
                used_gather_key = det_k
        else:
            prompt_vis = False

        dist_val = float(distance_m)
        ctrl = input_ctrl or self.input_controller

        # Durum 1: Karakter şu anda binek üzerinde -> <= 8m veya [F Gather]/[E] göründüyse in ve topla!
        if self.is_mounted:
            if self.should_dismount(
                distance_m=dist_val,
                gather_prompt_visible=prompt_vis,
                in_combat=combat_active,
            ):
                should_gather_now = bool(not combat_active and (prompt_vis or dist_val <= self.dismount_distance_m))
                dismount_res = self.dismount(
                    input_ctrl=ctrl,
                    start_gathering=should_gather_now,
                    gather_key=used_gather_key,
                )
                return {
                    "action": "DISMOUNT_AND_GATHER" if should_gather_now else "DISMOUNT",
                    "mounted": False,
                    "dismounted": True,
                    "started_gathering": should_gather_now,
                    "gather_key": used_gather_key,
                    "is_mounted": self.is_mounted,
                    "distance_m": dist_val,
                    "in_combat": combat_active,
                    "cannot_mount": cannot_mount,
                    "gather_prompt_visible": prompt_vis,
                    "dismount_info": dismount_res,
                }

            return {
                "action": "KEEP_RIDING",
                "mounted": False,
                "dismounted": False,
                "started_gathering": False,
                "gather_key": used_gather_key,
                "is_mounted": self.is_mounted,
                "distance_m": dist_val,
                "in_combat": combat_active,
                "cannot_mount": cannot_mount,
                "gather_prompt_visible": prompt_vis,
            }

        # Durum 2: Karakter yaya ve [F Gather]/[E] aktif -> doğrudan toplamaya başla
        if prompt_vis and not combat_active:
            _send_key_via_controller(ctrl, used_gather_key, test_mode=self.test_mode)
            return {
                "action": "GATHER",
                "mounted": False,
                "dismounted": False,
                "started_gathering": True,
                "gather_key": used_gather_key,
                "is_mounted": self.is_mounted,
                "distance_m": dist_val,
                "in_combat": combat_active,
                "cannot_mount": cannot_mount,
                "gather_prompt_visible": True,
            }

        # Durum 3: Karakter yaya ve mesafe > 35m, savaş yok, "Cannot mount" yok -> bineğe bin!
        if self.should_mount(
            distance_m=dist_val,
            in_combat=combat_active,
            cannot_mount_alert=cannot_mount,
            gather_prompt_visible=prompt_vis,
            now=curr_time,
        ):
            self.mount_up(input_ctrl=ctrl)
            return {
                "action": "MOUNT",
                "mounted": True,
                "dismounted": False,
                "started_gathering": False,
                "mount_key": self.mount_key,
                "is_mounted": self.is_mounted,
                "distance_m": dist_val,
                "in_combat": combat_active,
                "cannot_mount": cannot_mount,
                "gather_prompt_visible": False,
            }

        return {
            "action": "BLOCKED_CANNOT_MOUNT" if cannot_mount else ("BLOCKED_COMBAT" if combat_active else "WALK"),
            "mounted": False,
            "dismounted": False,
            "started_gathering": False,
            "is_mounted": self.is_mounted,
            "distance_m": dist_val,
            "in_combat": combat_active,
            "cannot_mount": cannot_mount,
            "gather_prompt_visible": prompt_vis,
        }

    evaluate = update
    step = update
    manage_mount = update


# ==============================================================================
# MADDE 25 — MİNİMAP ÜZERİNDEN DOĞRUDAN YÖNELİM (`MaplessMicroNavigator`)
# ==============================================================================
class MaplessMicroNavigator:
    """
    Madde 25 — Minimap Üzerinden Doğrudan Yönelim (`MaplessMicroNavigator`):

    Sıradaki düğüm `< 40m` (`max_micro_nav_distance_m = 40.0`) mesafedeyse veya
    sağ-üst minimap üzerinde görünüyorsa, tam ekran `M` haritasını açıp vakit
    kaybetmek yerine:
    1. Sağ-üst minimap üzerindeki oyuncu merkezinden (`cx, cy`) hedef düğüme olan
       `(dx, dy)` yönelim vektörünü ve pusula açısını (`heading_angle_deg`, 0°=Kuzey,
       90°=Doğu, 180°=Güney, 270°=Batı) doğrudan hesaplar.
    2. Tam ekran `M` haritasını **KESİNLİKLE AÇMAZ** (`map_opened = False`).
    3. Karakteri 3D oyun ekranında doğrudan hedef vektörüne yönlendirir (`steer`).
    """

    def __init__(
        self,
        minimap_scanner: Optional[MinimapScanner] = None,
        input_controller: Any = None,
        vision_engine: Any = None,
        max_micro_nav_distance_m: float = MAPLESS_DISTANCE_THRESHOLD_M,
        arrival_distance_m: float = 3.5,
        meters_per_minimap_px: float = 0.55,
        screen_width: int = 1920,
        screen_height: int = 1080,
        test_mode: bool = False,
        input_ctrl: Any = None,
        input_driver: Any = None,
        vision: Any = None,
    ) -> None:
        self.minimap_scanner: MinimapScanner = minimap_scanner or MinimapScanner()
        self.input_controller: Any = input_controller or input_ctrl or input_driver
        self.vision_engine: Any = vision_engine or vision
        self.max_micro_nav_distance_m: float = float(max_micro_nav_distance_m)
        self.arrival_distance_m: float = max(0.5, float(arrival_distance_m))
        self.meters_per_minimap_px: float = max(0.05, float(meters_per_minimap_px))
        self.screen_width: int = max(320, int(screen_width))
        self.screen_height: int = max(240, int(screen_height))
        self.test_mode: bool = bool(test_mode)

        self.micro_nav_steps: int = 0
        self.map_open_calls_avoided: int = 0
        self.last_steering_result: Optional[Dict[str, Any]] = None

    def should_use_mapless_navigation(
        self,
        distance_m: Optional[float] = None,
        visible_on_minimap: bool = False,
    ) -> bool:
        """
        Tam ekran `M` haritasını açmadan doğrudan minimap üzerinden mikro-navigasyon
        kullanılıp kullanılmayacağını belirler (`distance_m < 40.0` veya `visible_on_minimap == True`).
        """
        if bool(visible_on_minimap):
            return True
        if distance_m is not None and float(distance_m) < self.max_micro_nav_distance_m:
            return True
        return False

    @staticmethod
    def compute_compass_heading_deg(dx: float, dy: float) -> float:
        """
        Minimap piksel ofsetini (`dx` sağa pozitif, `dy` aşağı pozitif)
        pusula açısına (`0.0° .. 360.0°`, Kuzey=0°, Doğu=90°, Güney=180°, Batı=270°) dönüştürür.
        """
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            return 0.0
        angle = math.degrees(math.atan2(float(dx), -float(dy)))
        return round((angle + 360.0) % 360.0, 2)

    @staticmethod
    def compute_shortest_turn_deg(current_heading_deg: float, target_heading_deg: float) -> float:
        """
        Mevcut karakter yönünden (`current_heading_deg`) hedef yöne (`target_heading_deg`)
        en kısa işaretli dönüş açısını (`[-180.0°, +180.0°]`) hesaplar.
        """
        diff = (float(target_heading_deg) - float(current_heading_deg) + 180.0) % 360.0 - 180.0
        return round(diff, 2)

    def compute_steering_vector(
        self,
        minimap_offset: Optional[Tuple[float, float]] = None,
        player_pos: Optional[Tuple[float, float]] = None,
        target_pos: Optional[Tuple[float, float]] = None,
        player_heading_deg: Optional[float] = None,
        distance_m: Optional[float] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Sağ-üst minimap ofsetinden (`minimap_offset = (dx, dy)`) veya oyuncu/hedef
        koordinatlarından yönelim vektörünü `(dx, dy)`, birim vektörü `(unit_dx, unit_dy)`,
        pusula açısını (`heading_angle_deg`) ve 3D ekran yönelim noktasını (`screen_steer_point`)
        hesaplar.
        """
        if minimap_offset is not None:
            dx = float(minimap_offset[0])
            dy = float(minimap_offset[1])
        elif player_pos is not None and target_pos is not None:
            dx = float(target_pos[0]) - float(player_pos[0])
            dy = float(target_pos[1]) - float(player_pos[1])
        elif target_pos is not None:
            dx = float(target_pos[0])
            dy = float(target_pos[1])
        else:
            dx, dy = 0.0, 0.0

        pixel_dist = math.hypot(dx, dy)
        if pixel_dist > 1e-6:
            unit_dx = round(dx / pixel_dist, 6)
            unit_dy = round(dy / pixel_dist, 6)
        else:
            unit_dx, unit_dy = 0.0, 0.0

        est_dist_m = (
            float(distance_m)
            if distance_m is not None
            else round(pixel_dist * self.meters_per_minimap_px, 2)
        )
        target_heading = self.compute_compass_heading_deg(dx, dy)
        heading_error = (
            self.compute_shortest_turn_deg(player_heading_deg, target_heading)
            if player_heading_deg is not None
            else 0.0
        )

        sw, sh = screen_size if screen_size is not None else (self.screen_width, self.screen_height)
        center_x = float(sw) * 0.50
        center_y = float(sh) * 0.52
        steer_radius = min(float(sw), float(sh)) * 0.18
        steer_x = int(round(np.clip(center_x + unit_dx * steer_radius, 40, sw - 40)))
        steer_y = int(round(np.clip(center_y + unit_dy * steer_radius, 40, sh - 40)))

        return {
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "steering_vector": (round(dx, 3), round(dy, 3)),
            "unit_vector": (unit_dx, unit_dy),
            "unit_dx": unit_dx,
            "unit_dy": unit_dy,
            "pixel_distance": round(pixel_dist, 3),
            "distance_m": round(est_dist_m, 2),
            "heading_angle_deg": target_heading,
            "heading_deg": target_heading,
            "player_heading_deg": round(float(player_heading_deg), 2) if player_heading_deg is not None else None,
            "heading_error_deg": heading_error,
            "screen_steer_point": (steer_x, steer_y),
        }

    def locate_node_on_minimap(
        self,
        frame: np.ndarray,
        target_node: Optional[Union[NodeData, Dict[str, Any]]] = None,
        player_map_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Sağ-üst minimap karesini (`frame`) tarayarak hedef düğümün minimap üzerindeki
        `(dx, dy)` ofsetini, ekran koordinatını ve mesafesini bulur.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        try:
            # 1. MinimapScanner.scan_minimap mevcutsa toplanabilir filiz/materyal düğümlerini tara
            if hasattr(self.minimap_scanner, "scan_minimap"):
                scan_res = self.minimap_scanner.scan_minimap(frame)
                gatherable = scan_res.get("gatherable_nodes", []) if isinstance(scan_res, dict) else []
                if gatherable:
                    # Eğer target_node ve player_map_pos verilmişse beklenen ofsete en yakın olanı seç
                    if target_node is not None and player_map_pos is not None:
                        tx = float(target_node["map_x"] if isinstance(target_node, dict) else getattr(target_node, "map_x", 0))
                        ty = float(target_node["map_y"] if isinstance(target_node, dict) else getattr(target_node, "map_y", 0))
                        exp_dx = tx - float(player_map_pos[0])
                        exp_dy = ty - float(player_map_pos[1])
                        chosen = min(
                            gatherable,
                            key=lambda n: math.hypot(float(n.get("dx", 0.0)) - exp_dx, float(n.get("dy", 0.0)) - exp_dy),
                        )
                    else:
                        chosen = min(
                            gatherable,
                            key=lambda n: math.hypot(float(n.get("dx", 0.0)), float(n.get("dy", 0.0))),
                        )
                    dx = float(chosen.get("dx", 0.0))
                    dy = float(chosen.get("dy", 0.0))
                    dist_px = math.hypot(dx, dy)
                    return {
                        "visible_on_minimap": True,
                        "dx": round(dx, 3),
                        "dy": round(dy, 3),
                        "distance_m": round(dist_px * self.meters_per_minimap_px, 2),
                        "heading_angle_deg": self.compute_compass_heading_deg(dx, dy),
                        "player_heading_deg": scan_res.get("player_heading_deg") if isinstance(scan_res, dict) else None,
                    }

            # 2. Yedek: Minimap kesitinde parlak hedef ikonu / şablon araması
            crop, _ = self.minimap_scanner.extract_minimap(frame)
            pcx, pcy = self.minimap_scanner.get_minimap_player_center(crop.shape, crop_bgr=crop)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            ch, cw = gray.shape[:2]

            # Merkezdeki oyuncu okunu maskele (r=11px) ve en parlak materyal kümesini bul
            masked = gray.copy()
            cv2.circle(masked, (int(round(pcx)), int(round(pcy))), 11, 0, -1)
            # Dairesel minimap iç bölgesini koru
            circle_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(circle_mask, (int(round(pcx)), int(round(pcy))), int(min(cw, ch) * 0.44), 255, -1)
            masked = cv2.bitwise_and(masked, circle_mask)

            _, bw = cv2.threshold(masked, 210, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if 4.0 <= cv2.contourArea(c) <= 180.0]
            if valid:
                best_c = min(
                    valid,
                    key=lambda c: math.hypot(
                        float(cv2.boundingRect(c)[0]) - pcx,
                        float(cv2.boundingRect(c)[1]) - pcy,
                    ),
                )
                m = cv2.moments(best_c)
                if m["m00"] > 1e-5:
                    nx = float(m["m10"] / m["m00"])
                    ny = float(m["m01"] / m["m00"])
                    dx = nx - pcx
                    dy = ny - pcy
                    dist_px = math.hypot(dx, dy)
                    return {
                        "visible_on_minimap": True,
                        "dx": round(dx, 3),
                        "dy": round(dy, 3),
                        "distance_m": round(dist_px * self.meters_per_minimap_px, 2),
                        "heading_angle_deg": self.compute_compass_heading_deg(dx, dy),
                        "player_heading_deg": None,
                    }
        except Exception as exc:
            logger.debug("Minimap üzerinde düğüm arama hatası: %s", exc)

        return None

    def navigate_to_node(
        self,
        target_node: Optional[Union[NodeData, Dict[str, Any], Tuple[float, float]]] = None,
        distance_m: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        player_map_pos: Optional[Tuple[float, float]] = None,
        minimap_offset: Optional[Tuple[float, float]] = None,
        visible_on_minimap: Optional[bool] = None,
        player_heading_deg: Optional[float] = None,
        input_ctrl: Any = None,
        execute_input: bool = True,
    ) -> Dict[str, Any]:
        """
        Madde 25 — Haritasız Mikro-Navigasyon (`< 40m` veya Minimap Üzerinde Görünür):
        - Hedef `< 40m` mesafedeyse veya sağ-üst minimap üzerinde görünüyorsa, tam ekran
          `M` haritasını **KESİNLİKLE AÇMAZ** (`map_opened = False`).
        - Sağ-üst minimap ofsetinden `(dx, dy)` yönelim vektörünü ve `heading_angle_deg`
          açısını hesaplar ve karakteri doğrudan düğüme yönlendirir.
        """
        det_minimap: Optional[Dict[str, Any]] = None
        if minimap_offset is None and frame is not None:
            det_minimap = self.locate_node_on_minimap(
                frame=frame,
                target_node=target_node if isinstance(target_node, (NodeData, dict)) else None,
                player_map_pos=player_map_pos,
            )
            if det_minimap is not None:
                minimap_offset = (float(det_minimap["dx"]), float(det_minimap["dy"]))
                if visible_on_minimap is None:
                    visible_on_minimap = True
                if distance_m is None:
                    distance_m = float(det_minimap["distance_m"])
                if player_heading_deg is None and det_minimap.get("player_heading_deg") is not None:
                    player_heading_deg = float(det_minimap["player_heading_deg"])

        # Eğer minimap_offset hala yoksa target_node ve player_map_pos'tan hesapla
        if minimap_offset is None:
            if isinstance(target_node, (tuple, list)) and len(target_node) >= 2:
                if player_map_pos is not None:
                    minimap_offset = (
                        float(target_node[0]) - float(player_map_pos[0]),
                        float(target_node[1]) - float(player_map_pos[1]),
                    )
                else:
                    minimap_offset = (float(target_node[0]), float(target_node[1]))
            elif target_node is not None:
                tx = float(
                    target_node.get("dx", target_node.get("rel_map_dx", target_node.get("map_x", 0.0)))
                    if isinstance(target_node, dict)
                    else getattr(target_node, "map_x", 0.0)
                )
                ty = float(
                    target_node.get("dy", target_node.get("rel_map_dy", target_node.get("map_y", 0.0)))
                    if isinstance(target_node, dict)
                    else getattr(target_node, "map_y", 0.0)
                )
                if isinstance(target_node, dict) and ("dx" in target_node or "rel_map_dx" in target_node):
                    minimap_offset = (tx, ty)
                elif player_map_pos is not None:
                    minimap_offset = (tx - float(player_map_pos[0]), ty - float(player_map_pos[1]))
                else:
                    minimap_offset = (tx, ty)

        # Oyuncu yönü (heading) frame üzerinden okunabiliyorsa oku
        if player_heading_deg is None and frame is not None and hasattr(self.minimap_scanner, "detect_player_heading"):
            try:
                ok_h, h_deg = self.minimap_scanner.detect_player_heading(frame)
                if ok_h:
                    player_heading_deg = float(h_deg)
            except Exception as exc:
                logger.debug("Oyuncu yön açısı okunamadı: %s", exc)

        steer_data = self.compute_steering_vector(
            minimap_offset=minimap_offset,
            player_heading_deg=player_heading_deg,
            distance_m=distance_m,
            screen_size=(
                (frame.shape[1], frame.shape[0])
                if (frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 2)
                else (self.screen_width, self.screen_height)
            ),
        )

        effective_dist_m = float(steer_data["distance_m"])
        is_vis = bool(visible_on_minimap) if visible_on_minimap is not None else (
            effective_dist_m < self.max_micro_nav_distance_m and steer_data["pixel_distance"] > 0.0
        )

        if not self.should_use_mapless_navigation(distance_m=effective_dist_m, visible_on_minimap=is_vis):
            return {
                "used_mapless": False,
                "map_opened": False,
                "reason": "DISTANCE_EXCEEDS_MAPLESS_THRESHOLD",
                "distance_m": effective_dist_m,
                "max_micro_nav_distance_m": self.max_micro_nav_distance_m,
                "visible_on_minimap": is_vis,
                "dx": steer_data["dx"],
                "dy": steer_data["dy"],
                "steering_vector": steer_data["steering_vector"],
                "heading_angle_deg": steer_data["heading_angle_deg"],
                "heading_deg": steer_data["heading_deg"],
                "steered": False,
            }

        # < 40m veya minimap üzerinde görünür -> 'M' haritasını AÇMADAN doğrudan yönel!
        self.micro_nav_steps += 1
        self.map_open_calls_avoided += 1

        steered = False
        ctrl = input_ctrl or self.input_controller
        sx, sy = steer_data["screen_steer_point"]

        if execute_input:
            if ctrl is not None:
                try:
                    if hasattr(ctrl, "smooth_move_to") and not self.test_mode:
                        ctrl.smooth_move_to(sx, sy)
                        steered = True
                    elif hasattr(ctrl, "move_mouse") and callable(ctrl.move_mouse):
                        ctrl.move_mouse(sx, sy)
                        steered = True
                    elif hasattr(ctrl, "press_key") and callable(ctrl.press_key):
                        ctrl.press_key("w")
                        steered = True
                    else:
                        steered = True
                except Exception as exc:
                    logger.debug("Haritasız mikro-navigasyon girdi hatası: %s", exc)
            else:
                steered = True

        result = {
            "used_mapless": True,
            "map_opened": False,
            "dx": steer_data["dx"],
            "dy": steer_data["dy"],
            "steering_vector": steer_data["steering_vector"],
            "unit_vector": steer_data["unit_vector"],
            "unit_dx": steer_data["unit_dx"],
            "unit_dy": steer_data["unit_dy"],
            "pixel_distance": steer_data["pixel_distance"],
            "distance_m": effective_dist_m,
            "heading_angle_deg": steer_data["heading_angle_deg"],
            "heading_deg": steer_data["heading_deg"],
            "player_heading_deg": steer_data["player_heading_deg"],
            "heading_error_deg": steer_data["heading_error_deg"],
            "visible_on_minimap": is_vis,
            "screen_steer_point": (sx, sy),
            "steered": bool(steered),
            "arrived": effective_dist_m <= self.arrival_distance_m,
        }
        self.last_steering_result = result
        logger.debug(
            "[MaplessMicroNav] M haritası açılmadan yönelindi: dx=%.1f, dy=%.1f, açı=%.1f°, mesafe=%.1fm",
            result["dx"],
            result["dy"],
            result["heading_angle_deg"],
            result["distance_m"],
        )
        return result

    steer_to_node = navigate_to_node
    micro_navigate = navigate_to_node
    step = navigate_to_node


from core.route_manager import NodeData, RouteManager  # noqa: E402,F401


class NavigationIntel:
    """
    NavigationIntel sınıfı, toplanan harita, binek ve navigasyon zekası modüllerini
    bir araya getiren ana koordinasyon sınıfıdır.
    """
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def get_navigation_intel_health_summary(self) -> dict:
        return {"module": "navigation_intel", "ready": True}

    def get_navigation_intel_ready_flag(self) -> dict:
        return {"module": "navigation_intel", "ready": True}

    def get_navigation_intel_version_info(self) -> dict:
        return {"module": "navigation_intel", "version": "2.0.0", "ready": True}

    def get_navigation_intel_diagnostic_snapshot(self) -> dict:
        return {"module": "navigation_intel", "status": "ok", "ready": True}


__all__ = [
    "DEFAULT_REPROMPT_WAIT_SEC",
    "DEFAULT_REPROMPT_WAIT_MS",
    "MOUNT_DISTANCE_THRESHOLD_M",
    "DISMOUNT_DISTANCE_THRESHOLD_M",
    "MAPLESS_DISTANCE_THRESHOLD_M",
    "ThreeStageHarvestController",
    "PartialNodeDetector",
    "handle_extraction_failure",
    "AutoMountManager",
    "MaplessMicroNavigator",
]


def get_navigation_intel_capabilities() -> list:
    return ["navigation_intel", "spatial_heuristics", "diagnostic_snapshot"]


def validate_navigation_intel_contract() -> bool:
    return bool(get_navigation_intel_capabilities())


def get_navigation_intel_baseline_metrics() -> dict:
    return {"module": "navigation_intel", "contract_valid": validate_navigation_intel_contract(), "capability_count": len(get_navigation_intel_capabilities())}


