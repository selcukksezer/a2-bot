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
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from core.minimap_scanner import MinimapScanner
from core.ocr_distance import GameHUDReader
from version import __version__

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

DEFAULT_STUCK_TIMEOUT_SEC: float = 3.0
DEFAULT_MAX_HISTORY: int = 200  # history listeleri için üst sınır (sınırsız büyüme engeli)


# H-sweep (dilim 12 O-5): kerteriz / açı farkı / dönüş sönümleme
# corridor_navigator.py'deki birebir kopyanın tekilleştirilmiş hâlidir.
# İki ayrı tanım tutuluyordu; biri düzeltilince diğeri eski kalıyordu
# (kanıt: F-0011/F-0012 corridor'da düzelmiş, buradaki kopyada duruyordu).
# İsimler yeniden ihraç edilir; `from core.navigation_intel import
# compute_bearing` çalışmaya devam eder.
# P3-3: ölü-bölge sabitleri (6.0/5.0/8.0) de corridor_navigator tek
# kaynağından gelir; aşağıdaki `__all__` girdileri aynen korunur.
from core.corridor_navigator import (  # noqa: E402
    DEFAULT_DEADBAND_DEG,
    MAX_DEADBAND_DEG,
    MINIMAP_METERS_PER_PX,
    MIN_DEADBAND_DEG,
    WORLD_MAP_METERS_PER_PX,
    TurnDampingController,
    apply_turn_damping,
    compute_angle_diff,
    compute_bearing,
)
# P3-3 v19: DEFAULT_WAYPOINT_RADIUS_PX/M tek tanımı `core/navigator.py`
# dedir (20.0 px / 3.0 m); buradaki kopya silindi, adlar yeniden ihraç
# edilir (`from core.navigation_intel import DEFAULT_WAYPOINT_RADIUS_PX`
# ve aşağıdaki `__all__` girdileri çalışmaya devam eder). Döngüsel import
# yoktur: navigator -> {corridor_navigator, input_driver, vision,
# route_manager} zincirinin hiçbir halkası navigation_intel'i üst düzeyde
# içe aktarmaz (route_manager yalnızca tembel içe aktarır).
from core.navigator import (  # noqa: E402
    DEFAULT_WAYPOINT_RADIUS_M,
    DEFAULT_WAYPOINT_RADIUS_PX,
)
class RadarStuckDetector:
    """
    Radar Intel Tabanlı Takılma Tespiti ve Kurtarma:
    Oyuncu hareket tuşları (W / strafe) basılı iken oyuncunun minimap / dünya
    konumunun değişip değişmediğini izler. Konum > 3.0 saniye boyunca değişmezse
    takılma (stuck) tespit eder ve kurtulma manevrasını (Space -> A+W/D+W -> Realign) tetikler.
    """
    def __init__(
        self,
        stuck_timeout_sec: float = DEFAULT_STUCK_TIMEOUT_SEC,
        movement_threshold_px: float = 2.0,
        movement_threshold_m: float = 0.5,
        strafe_duration_sec: float = 1.0,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.stuck_timeout_sec: float = float(max(0.5, stuck_timeout_sec))
        self.movement_threshold_px: float = float(max(0.1, movement_threshold_px))
        self.movement_threshold_m: float = float(max(0.05, movement_threshold_m))
        self.strafe_duration_sec: float = float(max(0.1, strafe_duration_sec))
        self.max_history: int = max(1, int(max_history))

        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_pos_ts: Optional[float] = None
        self.stuck_start_ts: Optional[float] = None
        self.is_stuck: bool = False
        self.stuck_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.last_pos = None
        self.last_pos_ts = None
        self.stuck_start_ts = None
        self.is_stuck = False

    def update_position(
        self,
        player_pos: Optional[Tuple[float, float]],
        movement_keys_held: bool = True,
        distance_is_meters: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Oyuncu konumunu günceller ve hareket tuşları basılı iken > 3.0s
        boyunca konum değişmezse is_stuck=True durumuna geçer.
        """
        curr_ts = time.time() if now is None else float(now)

        if player_pos is None or not movement_keys_held:
            self.last_pos = player_pos
            self.last_pos_ts = curr_ts
            self.stuck_start_ts = None
            self.is_stuck = False
            return {
                "is_stuck": False,
                "stuck_elapsed_sec": 0.0,
                "movement_distance": 0.0,
                "movement_keys_held": bool(movement_keys_held),
            }

        px, py = float(player_pos[0]), float(player_pos[1])
        thresh = self.movement_threshold_m if distance_is_meters else self.movement_threshold_px

        if self.last_pos is None:
            self.last_pos = (px, py)
            self.last_pos_ts = curr_ts
            self.stuck_start_ts = None
            self.is_stuck = False
            return {
                "is_stuck": False,
                "stuck_elapsed_sec": 0.0,
                "movement_distance": 0.0,
                "movement_keys_held": True,
            }

        dist = math.hypot(px - self.last_pos[0], py - self.last_pos[1])

        if dist >= thresh:
            self.last_pos = (px, py)
            self.last_pos_ts = curr_ts
            self.stuck_start_ts = None
            self.is_stuck = False
            stuck_elapsed = 0.0
        else:
            if self.stuck_start_ts is None:
                self.stuck_start_ts = self.last_pos_ts if self.last_pos_ts is not None else curr_ts
            stuck_elapsed = max(0.0, curr_ts - self.stuck_start_ts)
            if stuck_elapsed >= self.stuck_timeout_sec:
                self.is_stuck = True

        return {
            "is_stuck": bool(self.is_stuck),
            "stuck_elapsed_sec": round(stuck_elapsed, 3),
            "movement_distance": round(dist, 3),
            "movement_keys_held": True,
            "threshold": thresh,
            "stuck_timeout_sec": self.stuck_timeout_sec,
        }

    def execute_unstick_maneuver(
        self,
        input_controller: Any = None,
        target_bearing: Optional[float] = None,
        player_heading: Optional[float] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
    ) -> Dict[str, Any]:
        """
        Takılma durumunda fiziksel kurtulma manevrası:
        1. Space (Zıpla)
        2. A+W veya dönüşümlü D+W strafe
        3. En kısa açı farkıyla hedefe yeniden hizalanma (Realign)
        """
        strafe_key = "a" if (self.stuck_count % 2 == 0) else "d"
        self.stuck_count += 1
        # H-sweep (dilim 12 Y-1 — F-0011/F-0012'nin bu dosyadaki kopyası):
        # bekleme verilmezse gerçek bekleme (time.sleep) kullanılır; no-op
        # varsayılan A+W strafe'i anında bas-bırak yapıyordu.
        wait_fn = sleep_fn if callable(sleep_fn) else time.sleep
        actions: List[str] = []
        attempted = False

        if input_controller is not None:
            actions.append("SPACE")
            if hasattr(input_controller, "tap_key") and callable(input_controller.tap_key):
                input_controller.tap_key("space")
                attempted = True
            elif hasattr(input_controller, "press_key") and callable(input_controller.press_key):
                input_controller.press_key("space")
                attempted = True
            elif hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
                input_controller.key_down("space")
                wait_fn(0.06)
                input_controller.key_up("space")
                attempted = True

        combo = f"{strafe_key.upper()}+W"
        if input_controller is not None:
            actions.append(combo)
            if hasattr(input_controller, "hold_keys") and callable(input_controller.hold_keys):
                input_controller.hold_keys([strafe_key, "w"], duration=self.strafe_duration_sec)
                attempted = True
            elif hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
                input_controller.key_down("w")
                input_controller.key_down(strafe_key)
                wait_fn(self.strafe_duration_sec)
                input_controller.key_up(strafe_key)
                input_controller.key_up("w")
                attempted = True
            elif hasattr(input_controller, "press_key") and callable(input_controller.press_key):
                input_controller.press_key(strafe_key)
                input_controller.press_key("w")
                wait_fn(self.strafe_duration_sec)
                attempted = True

        angle_diff = 0.0
        damped = None
        if target_bearing is not None and player_heading is not None:
            angle_diff = compute_angle_diff(target_bearing, player_heading)
            damped = apply_turn_damping(angle_diff, deadband_deg=deadband_deg)

        if input_controller is not None:
            actions.append("REALIGN")
            if damped is None or damped["turn_needed"]:
                if hasattr(input_controller, "turn_by_angle") and callable(input_controller.turn_by_angle):
                    turn_dur = damped.get("turn_duration_sec") if damped else None
                    try:
                        input_controller.turn_by_angle(angle_diff, duration=turn_dur)
                    except TypeError:
                        input_controller.turn_by_angle(angle_diff)
                    attempted = True
                elif hasattr(input_controller, "realign_to_target") and callable(input_controller.realign_to_target):
                    input_controller.realign_to_target(target_bearing)
                    attempted = True

        # H-sweep (dilim 12 Y-1): kontrolcü yokken hiçbir tuşa basılmadı;
        # "executed: True" üst katmana manevra uygulandı yalanını söylerdi.
        # executed yalnızca gerçekten bir girdi denendiyse True'dur.
        executed = bool(attempted)
        if executed:
            self.stuck_start_ts = None
            self.is_stuck = False
            self.last_pos = None

        logger.info(
            "[RadarStuckDetector] Kurtarma manevrası #%d %s: %s (strafe=%.2fs, açı=%.1f°)",
            self.stuck_count,
            "uygulandı" if executed else "UYGULANAMADI (kontrolcü yok)",
            " -> ".join(actions) if actions else "-",
            self.strafe_duration_sec,
            angle_diff,
        )

        res = {
            "executed": executed,
            "actions": actions,
            "sequence": actions,
            "strafe_key": strafe_key,
            "strafe_combo": combo,
            "strafe_duration_sec": self.strafe_duration_sec,
            "angle_diff_deg": round(angle_diff, 2),
            "in_deadband": bool(damped.get("in_deadband", False)) if damped else False,
            "turn_needed": bool(damped.get("turn_needed", True)) if damped else True,
            "stuck_count": self.stuck_count,
        }
        _append_bounded(self.history, res, self.max_history)
        return res


class WaypointArrivalDetector:
    """
    Hedef düğüm varış yarıçapı (örneğin <= 20px veya <= 3.0m) tespiti ve
    sıradaki ardışık düğüme (Next in Sequence) temiz geçiş yöneticisi.
    """
    def __init__(
        self,
        waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
        waypoint_radius_m: float = DEFAULT_WAYPOINT_RADIUS_M,
        # H-sweep (dilim 12 Y-3): ölçek corridor_navigator.MINIMAP_METERS_PER_PX
        # tek kaynağından gelir (0.55); 0.65/0.28/0.25 değerleri başka
        # uzaylara/modüllere aittir, buraya yazılmaz.
        meters_per_px: float = MINIMAP_METERS_PER_PX,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.waypoint_radius_px: float = float(max(1.0, waypoint_radius_px))
        self.waypoint_radius_m: float = float(max(0.5, waypoint_radius_m))
        self.meters_per_px: float = float(max(0.05, meters_per_px))
        self.max_history: int = max(1, int(max_history))
        self.transitions_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def is_arrival_reached(
        self,
        player_pos: Optional[Tuple[float, float]],
        waypoint_pos: Optional[Tuple[float, float]],
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Oyuncunun hedef düğüm yarıçapına ulaşıp ulaşmadığını belirler.
        """
        if pixel_distance is not None:
            px_dist = float(pixel_distance)
        elif player_pos is not None and waypoint_pos is not None:
            px_dist = math.hypot(
                float(waypoint_pos[0]) - float(player_pos[0]),
                float(waypoint_pos[1]) - float(player_pos[1]),
            )
        else:
            px_dist = 9999.0

        if distance_m is not None:
            m_dist = float(distance_m)
            distance_source = "measured"  # OCR / HUD'dan bağımsız ölçüm
        else:
            m_dist = px_dist * self.meters_per_px
            distance_source = "derived_from_px"  # pikselden türetildi

        reached_px = px_dist <= self.waypoint_radius_px
        reached_m = m_dist <= self.waypoint_radius_m
        # F-0324: varış için İKİ ölçümün de eşik içinde olması gerekir. OR
        # mantığı 20px (≈11m) ile 3m arasındaki tutarsızlığı gizleyip
        # 11m uzaktan "vardım" dedirtiyordu; AND ile iki ölçüm de
        # doğrulanmadan varış sayılmaz.
        # H-sweep (dilim 12 O-6): distance_m türetilmişse iki koşul aynı
        # sayının iki yazımıdır — AND anlamsızlaşmaz ama bağlayıcı olan
        # metre eşiğidir (px<=20 ⟺ m<=11 iken m<=3, yani px<=5.45).
        # Davranış korunur (F-0324 koruması); yalnızca kaynak döndürülür
        # ki çağıran iki bağımsız ölçüm var sanmasın.
        arrived = bool(reached_px and reached_m)

        return {
            "arrived": arrived,
            "pixel_distance": round(px_dist, 3),
            "distance_m": round(m_dist, 2),
            "distance_source": distance_source,
            "radius_px": self.waypoint_radius_px,
            "radius_m": self.waypoint_radius_m,
            "reached_px": reached_px,
            "reached_m": reached_m,
        }

    def transition_to_next_node(
        self,
        nodes: Sequence[Any],
        current_index: int,
        route_planner: Optional[Any] = None,
        stay_on_partial_harvest: bool = False,
        now: Optional[float] = None,
    ) -> Tuple[Optional[Any], int, bool]:
        """
        Sıradaki düğüme temiz geçiş yapar.
        """
        node_list = list(nodes) if nodes else []
        n = len(node_list)
        if n == 0:
            return None, -1, False

        curr_idx = int(current_index) if current_index >= 0 else 0

        if route_planner is not None and hasattr(route_planner, "select_next_sequential_node"):
            next_node, next_idx = route_planner.select_next_sequential_node(
                nodes=node_list,
                current_seq_pos=curr_idx,
                stay_on_partial_harvest=stay_on_partial_harvest,
                now=now,
            )
            self.transitions_count += 1
            return next_node, next_idx, True

        next_idx = (curr_idx + 1) % n
        next_node = node_list[next_idx]
        self.transitions_count += 1
        return next_node, next_idx, True

    def check_and_transition(
        self,
        player_pos: Optional[Tuple[float, float]],
        waypoint_pos: Optional[Tuple[float, float]],
        nodes: Sequence[Any],
        current_index: int,
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
        route_planner: Optional[Any] = None,
        stay_on_partial_harvest: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Varış kontrolü yapar ve ulaşıldıysa sıradaki düğüme temiz geçiş tetikler.
        """
        arrival_info = self.is_arrival_reached(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            distance_m=distance_m,
            pixel_distance=pixel_distance,
        )

        if not arrival_info["arrived"]:
            node_list = list(nodes) if nodes else []
            active_node = node_list[current_index] if (node_list and 0 <= current_index < len(node_list)) else None
            return {
                "arrived": False,
                "transitioned": False,
                "current_index": current_index,
                "active_node": active_node,
                "next_index": current_index,
                "next_node": active_node,
                **arrival_info,
            }

        next_node, next_idx, transitioned = self.transition_to_next_node(
            nodes=nodes,
            current_index=current_index,
            route_planner=route_planner,
            stay_on_partial_harvest=stay_on_partial_harvest,
            now=now,
        )

        record = {
            "arrived": True,
            "transitioned": bool(transitioned),
            "previous_index": current_index,
            "current_index": next_idx,
            "active_node": next_node,
            "next_index": next_idx,
            "next_node": next_node,
            **arrival_info,
        }
        _append_bounded(self.history, record, self.max_history)
        logger.info(
            "[WaypointArrival] Hedef düğüme ulaşıldı (mesafe=%.1fpx / %.2fm <= eşik %.1fpx / %.1fm) -> "
            "Sıradaki düğüme (#%d) geçildi.",
            arrival_info["pixel_distance"],
            arrival_info["distance_m"],
            self.waypoint_radius_px,
            self.waypoint_radius_m,
            next_idx,
        )
        return record


# ==============================================================================
# YARDIMCI FONKSİYONLAR
# ==============================================================================
def _probe_navigation_intel_ready(obj: Any) -> Tuple[bool, Dict[str, bool]]:
    """
    F-0478: NavigationIntel hazır olma probu — gerçek bileşen durumunu
    yoklar; hazır bayrağı artık koşulsuz True değildir. Karar 3
    (F-0208) gereği hazır bayrakları gerçek proba bağlanır.
    """
    checks: Dict[str, bool] = {}
    try:
        stuck = getattr(obj, "stuck_detector", None)
        checks["stuck_detector"] = bool(
            stuck is not None
            and hasattr(stuck, "stuck_count")
            and hasattr(stuck, "update_position")
        )
        waypoint = getattr(obj, "waypoint_detector", None)
        checks["waypoint_detector"] = bool(
            waypoint is not None
            and hasattr(waypoint, "transitions_count")
            and hasattr(waypoint, "is_arrival_reached")
        )
        damper = getattr(obj, "turn_damper", None)
        checks["turn_damper"] = bool(
            damper is not None and hasattr(damper, "calculate_turn")
        )
        micro = getattr(obj, "micro_navigator", None)
        checks["micro_navigator"] = bool(micro is not None)
        try:
            deadband = float(getattr(obj, "deadband_deg"))
            checks["deadband_in_range"] = MIN_DEADBAND_DEG <= deadband <= MAX_DEADBAND_DEG
        except (TypeError, ValueError):
            checks["deadband_in_range"] = False
        try:
            checks["waypoint_radius_valid"] = (
                float(getattr(obj, "waypoint_radius_px")) > 0
                and float(getattr(obj, "waypoint_radius_m")) > 0
            )
        except (TypeError, ValueError):
            checks["waypoint_radius_valid"] = False
        try:
            checks["stuck_timeout_valid"] = float(getattr(obj, "stuck_timeout_sec")) > 0
        except (TypeError, ValueError):
            checks["stuck_timeout_valid"] = False
    except Exception:
        return False, checks
    return bool(checks and all(checks.values())), checks


def get_navigation_intel_summary(self) -> dict:
    try:
        # F-0478: başarı yolu da proba bağlı — eskiden hata durumunda
        # bile intel_ready=True dönüyordu, başarı yolunda ise koşulsuz
        # True + ham __dict__ dönüyordu.
        ready, checks = _probe_navigation_intel_ready(self)
        return {"intel_ready": ready, "checks": checks, "metrics": getattr(self, "__dict__", {})}
    except Exception as e:
        return {"intel_ready": False, "error": str(e)}


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
        # F-0325: kontrolcü yokken hiçbir tuş basılmadı; test_mode
        # True bile olsa başarı raporlanmaz (önceden bool(test_mode)
        # dönüp basılmamış tuşu "basıldı" sayıyordu).
        return False

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


# ==============================================================================
# YEDEK (FALLBACK) GÖRSEL MOTORLAR — gecikmeli (lazy) paylaşılan örnekler
# ==============================================================================
# _detect_prompt_from_vision_or_frame / _detect_chat_alerts_unified, çağıran
# kendi motorunu vermediğinde her çağrıda VisionEngine() (~1.0 sn) /
# GameHUDReader() (~0.13-0.47 sn) kuruyordu; 50 ms poll döngüsünde bu,
# doğrulama penceresini kat kat aşan gecikme demekti. Motorların detect_*
# metotları yalnızca frame alır ve çağrılar arasında paylaşılabilir.
# B-03 disiplini: paylaşılan yedek motorun durumu (özellikle
# VisionEngine.last_detected_gather_key) HER yedek kullanım öncesi
# sıfırlanır — önceki, alakasız kareden kalan F/E tuşu sonraki çağrıda
# "tespit" olarak dönemez. Yedek kullanım debug-loglanır ve sohbet
# uyarısı sonucu "source" ile etiketlenir (birincil mi yedek mi belli).
# Kurulum başarısız olursa önbelleğe alınmaz: eski davranış gibi bir
# sonraki çağrıda yeniden denenir.
_FALLBACK_VISION_ENGINE: Optional[Any] = None
_FALLBACK_HUD_READER: Optional[Any] = None
# S081 (v19/P3-8): yedek motor singleton'ları kilitsiz kuruluyordu —
# iki thread aynı anda VisionEngine() (~1 sn) kurup birini
# ezebiliyordu; B-03 sıfırlamasıyla yarış da mümkündü. Çift
# kontrollü kilit (stability_guards.py deseni).
_FALLBACK_ENGINES_LOCK = threading.Lock()


def _get_fallback_vision_engine() -> Optional[Any]:
    """Paylaşılan yedek VisionEngine örneğini döndürür (ilk çağrıda kurar)."""
    global _FALLBACK_VISION_ENGINE
    if _FALLBACK_VISION_ENGINE is None:
        with _FALLBACK_ENGINES_LOCK:
            if _FALLBACK_VISION_ENGINE is None:
                try:
                    from core.vision import VisionEngine

                    _FALLBACK_VISION_ENGINE = VisionEngine()
                except Exception as exc:
                    logger.debug("Yedek VisionEngine kurulamadı: %s", exc)
                    return None
    return _FALLBACK_VISION_ENGINE


def _get_fallback_hud_reader() -> Optional[Any]:
    """Paylaşılan yedek GameHUDReader örneğini döndürür (ilk çağrıda kurar)."""
    global _FALLBACK_HUD_READER
    if _FALLBACK_HUD_READER is None:
        with _FALLBACK_ENGINES_LOCK:
            if _FALLBACK_HUD_READER is None:
                try:
                    _FALLBACK_HUD_READER = GameHUDReader()
                except Exception as exc:
                    logger.debug("Yedek GameHUDReader kurulamadı: %s", exc)
                    return None
    return _FALLBACK_HUD_READER


def _reset_fallback_vision_state(engine: Any, default_key: str = "f") -> None:
    """Paylaşılan yedek motorun çağrılar-arası durumunu sıfırla (B-03).

    VisionEngine.last_detected_gather_key önceki kareden kalmışsa bu
    karede hiç tespit yapılmasa bile eski F/E tuşu "tespit" gibi okunur.
    Yedek motor her kullanım öncesi varsayılan tuşa sıfırlanır; tespit
    metodu bu karede bulursa anahtarı kendisi günceller.
    """
    if engine is None:
        return
    try:
        if hasattr(engine, "last_detected_gather_key"):
            setattr(engine, "last_detected_gather_key", str(default_key or "f").strip().lower() or "f")
    except Exception as exc:
        logger.debug("Yedek motor durumu sıfırlanamadı: %s", exc)


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
                # F-0034: vision.detect_floating_resource_icon(frame, threshold,
                # use_roi, roi_norm) imzası close_range_only kabul etmez; eski
                # çağrı TypeError fırlatıp dış try'da sessizce yutuluyordu.
                # Desteklenen imza ile çağır, 4'lü dönüşü (found, conf,
                # center, key) doğrudan oku.
                icon_res = vision_engine.detect_floating_resource_icon(frame)
                if isinstance(icon_res, tuple) and len(icon_res) >= 1 and bool(icon_res[0]):
                    k_val = (
                        str(icon_res[3]).strip().lower()
                        if len(icon_res) >= 4 and isinstance(icon_res[3], str)
                        else ""
                    )
                    if k_val not in ("f", "e"):
                        last_k = str(getattr(vision_engine, "last_detected_gather_key", "e") or "e").strip().lower()
                        k_val = last_k if last_k in ("f", "e") else "e"
                    return True, k_val
                if isinstance(icon_res, dict) and bool(icon_res.get("detected", icon_res.get("found", False))):
                    k_val = str(icon_res.get("key", icon_res.get("gather_key", "e"))).strip().lower()
                    return True, (k_val if k_val in ("f", "e") else "e")
        except Exception as exc:
            logger.debug("Vision üzerinden gather prompt kontrol hatası: %s", exc)

    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        try:
            temp_vision = _get_fallback_vision_engine()
            if temp_vision is None:
                return False, detected_key
            # B-03: paylaşılan yedek motorun önceki kareden kalan
            # last_detected_gather_key durumu bu kullanım öncesi sıfırlanır;
            # aksi halde eski F/E tuşu bu karede "tespit" gibi dönerdi.
            _reset_fallback_vision_state(temp_vision, detected_key)
            logger.debug("Gather prompt yedek VisionEngine ile kontrol ediliyor (kaynak: fallback).")
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

    # B-03: sonucun kaynağı etiketlenir — çağıran kararın birincil
    # motordan mı yedek motordan mı geldiğini ayırt edebilir.
    source = "none"
    if isinstance(chat_alerts, dict) and chat_alerts:
        source = "chat_alerts"
    if isinstance(chat_text, str) and chat_text.strip():
        source = "chat_text"
    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        try:
            reader = hud_reader if hud_reader is not None else _get_fallback_hud_reader()
            if reader is not None and hasattr(reader, "detect_chat_system_alerts"):
                source = "hud_reader" if hud_reader is not None else "fallback_hud_reader"
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
        "source": source,
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
        # F-0326: bozuk id çökmesin.
        try:
            nid = int(node_or_id.id)
        except (TypeError, ValueError):
            return None, node_or_id
        if route_mgr is not None and hasattr(route_mgr, "get_node"):
            rm_node = route_mgr.get_node(nid, region=region or node_or_id.region)
            if rm_node is not None:
                return nid, rm_node
        return nid, node_or_id

    if isinstance(node_or_id, dict):
        # F-0152/F-0326: id'siz / bozuk id'li referans sessizce düğüm #1
        # olmasın; çözülemezse (None, node_or_id) dön, çağıran karar versin.
        raw_id = node_or_id.get("id", node_or_id.get("node_id"))
        if raw_id is None:
            return None, node_or_id
        try:
            nid = int(raw_id)
        except (TypeError, ValueError):
            return None, node_or_id
        if route_mgr is not None and hasattr(route_mgr, "get_node"):
            rm_node = route_mgr.get_node(nid, region=region or node_or_id.get("region"))
            if rm_node is not None:
                return nid, rm_node
        return nid, node_or_id

    # F-0326: bilinmeyen tipte nesnede korumasız int(...) çöküyordu.
    raw_id = getattr(node_or_id, "id", None)
    if raw_id is None:
        return None, node_or_id
    try:
        nid = int(raw_id)
    except (TypeError, ValueError):
        return None, node_or_id
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
        # F-0035: kanıt yokken (extraction_failed, chat_text, chat_alerts,
        # frame hiçbiri verilmediyse) başarısız varsaymak yanlış; bilinmeyen
        # durum "başarısız değil"dir, çağıran kanıtla çağırmalı.
        is_failed = False

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
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self.route_manager: Optional[RouteManager] = route_manager or route_mgr
        self.vision_engine: Any = vision_engine or vision
        self.reprompt_wait_sec: float = max(0.0, float(reprompt_wait_sec))
        self.verification_window_sec: float = max(self.reprompt_wait_sec, float(verification_window_sec))
        self.poll_interval_sec: float = max(0.005, float(poll_interval_sec))
        self.default_rest_minutes: float = max(0.1, float(default_rest_minutes))
        self.max_history: int = max(1, int(max_history))
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

        # S076 (v19/P3-8) — 2 kare kuralı: tek karelik prompt kabulü,
        # şablon parlamasında sahte F'e yol açıyordu. İlk olumlu tespit,
        # taze kaynaktan (aynı prompt_detector / frame_provider) bir kez
        # daha doğrulanır. Taze kaynak yoksa (yalnızca statik kare)
        # ikinci kare üretilemez — tek kare kabul edilir (belgelendi).
        # prompt_reappeared=True kısayolu (F-0036) çağıranın açık
        # kanıtıdır, görsel tespit değildir; o etkilenmez.
        def _second_positive(via: str) -> bool:
            # Doğrulama, olumlu tespitin geldiği kaynaktan yapılır.
            if via == "detector":
                if not callable(prompt_detector):
                    return True
                try:
                    res2 = prompt_detector()
                except Exception:
                    return False
                if isinstance(res2, tuple):
                    return bool(res2[0]) if len(res2) >= 1 else False
                return bool(res2)
            if not callable(frame_provider):
                return True
            try:
                frame2 = frame_provider()
            except Exception:
                return False
            found2, _ = _detect_prompt_from_vision_or_frame(
                vision_engine=self.vision_engine,
                frame=frame2,
                default_key=default_key,
            )
            return bool(found2)

        def _confirm_first_positive(via: str) -> bool:
            nonlocal waited_sec
            self._sleep(self.poll_interval_sec)
            waited_sec += self.poll_interval_sec
            return _second_positive(via)

        # F-0036: prompt_reappeared=True olumlu kanıttır, kısayol sayılır;
        # False ise görsel kontrol yapılmadan RESTING'e götüren kısayol
        # OLAMAZ — False geldiğinde aşağıdaki gerçek görsel doğrulama
        # çalışır, yalnızca o da bulamazsa False döner.
        if prompt_reappeared is True:
            return True, str(default_key or "f").lower(), round(waited_sec, 4)

        if callable(prompt_detector):
            res = prompt_detector()
            if isinstance(res, tuple) and len(res) >= 1:
                k = str(res[1]).lower() if len(res) >= 2 and res[1] else default_key
                if bool(res[0]) and _confirm_first_positive("detector"):
                    return True, k, round(waited_sec, 4)
            elif bool(res) and _confirm_first_positive("detector"):
                return True, default_key, round(waited_sec, 4)

        # İlk kontrol
        current_frame = frame_provider() if callable(frame_provider) else frame
        found, det_key = _detect_prompt_from_vision_or_frame(
            vision_engine=self.vision_engine,
            frame=current_frame,
            default_key=default_key,
        )
        if found and _confirm_first_positive("frame"):
            return True, det_key, round(waited_sec, 4)

        # F-0327: pencere, gerçekten beklenen süreden kısaltılır;
        # wait_initial_400ms=False iken bekleme 0 olmasına rağmen eskiden
        # her zaman 0.85 − 0.40 = 0.45 sn oluyordu.
        remaining_window = max(0.0, self.verification_window_sec - waited_sec)
        if remaining_window > 0.0 and (callable(frame_provider) or callable(prompt_detector)):
            deadline = time.monotonic() + remaining_window
            while time.monotonic() < deadline:
                self._sleep(self.poll_interval_sec)
                waited_sec += self.poll_interval_sec
                if callable(prompt_detector):
                    res = prompt_detector()
                    if isinstance(res, tuple) and len(res) >= 1 and bool(res[0]):
                        k = str(res[1]).lower() if len(res) >= 2 and res[1] else default_key
                        if _confirm_first_positive("detector"):
                            return True, k, round(waited_sec, 4)
                    if not isinstance(res, tuple) and bool(res):
                        if _confirm_first_positive("detector"):
                            return True, default_key, round(waited_sec, 4)
                if callable(frame_provider):
                    polled_frame = frame_provider()
                    found, det_key = _detect_prompt_from_vision_or_frame(
                        vision_engine=self.vision_engine,
                        frame=polled_frame,
                        default_key=default_key,
                    )
                    if found and _confirm_first_positive("frame"):
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
        # F-0152: çözülemeyen referans sessizce düğüm #1 olmasın; id yoksa
        # hiçbir düğüm dinlendirilmez, çözümsüzlük raporlanır.
        if node_id is None:
            return {
                "is_partial_node": False,
                "depleted_early": False,
                "node_id": None,
                "completed_harvests": completed_harvests,
                "max_harvests": max_harvests,
                "missing_harvests": None,
                "prompt_reappeared": prompt_reappeared,
                "node_state": "unknown",
                "rest_duration_sec": 0,
                "action": "UNRESOLVED_NODE",
            }
        resolved_id = int(node_id)

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
            # F-0146: dal "resting" yazıp düğümü dinlendirmiyordu; tam
            # hasat da gerçekten RESTING'e alınır (route_mgr + yerel nesne).
            rest_sec = int(round(effective_rest * 60.0))
            now_ts = time.time()
            updated_node = None
            if active_rm is not None and hasattr(active_rm, "mark_node_gathered"):
                updated_node = active_rm.mark_node_gathered(
                    resolved_id,
                    rest_minutes=effective_rest,
                    force_deplete=True,
                    region=region,
                )
            _sync_resting_nodes(node_obj, node_or_id, updated_node, rest_sec, now_ts)
            return {
                "is_partial_node": False,
                "depleted_early": False,
                "node_id": resolved_id,
                "completed_harvests": comp_h,
                "max_harvests": target_h,
                "missing_harvests": 0,
                "prompt_reappeared": False,
                "node_state": "resting",
                "rest_duration_sec": rest_sec,
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
        _sync_resting_nodes(node_obj, node_or_id, updated_node, rest_sec, now_ts)

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
        _append_bounded(self.history, event, self.max_history)
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
def _append_bounded(
    history: List[Dict[str, Any]],
    record: Dict[str, Any],
    max_len: int,
) -> None:
    """
    history listesine kayıt ekler; liste max_len'i aşarsa en eski kayıtları
    atar. Uzun bot oturumlarında sınırsız büyüme (bellek sızıntısı) engellenir;
    sayaçlar (stuck_count, transitions_count, partial_nodes_detected) ayrı
    tutulduğundan kırpma hiçbir sayacı sıfırlamaz.
    """
    history.append(record)
    cap = max(1, int(max_len))
    excess = len(history) - cap
    if excess > 0:
        del history[:excess]


def _apply_node_harvest_state(
    node_obj: Any,
    *,
    current_harvests: int,
    state: str,
    rest_sec: int = 0,
    now_ts: Optional[float] = None,
    stamp_rest: bool = False,
) -> None:
    """
    Tek düğüm nesnesine (dict veya NodeData) hasat sonrası durumu yazar.
    dict/nesne ayrımı tek noktada yapılır; çağıranlar aynı yazımı 3 ayrı
    yerde tekrarlamaz. stamp_rest=True ise dinlenme zaman damgası da yazılır
    (tükenmeyen düğümde last_visited/rest_duration yazılmaz).
    """
    if node_obj is None:
        return
    if isinstance(node_obj, dict):
        node_obj["current_harvests"] = int(current_harvests)
        node_obj["state"] = str(state)
        if stamp_rest:
            node_obj["last_visited_timestamp"] = now_ts
            node_obj["rest_duration_sec"] = int(rest_sec)
    else:
        node_obj.current_harvests = int(current_harvests)
        node_obj.state = str(state)
        if stamp_rest:
            node_obj.last_visited_timestamp = now_ts
            node_obj.rest_duration_sec = int(rest_sec)


def _sync_resting_nodes(
    node_obj: Any,
    node_or_id: Any,
    updated_node: Any,
    rest_sec: int,
    now_ts: float,
) -> None:
    """
    route_mgr'in döndürdüğü (updated_node) dışındaki düğüm referanslarını
    RESTING durumuna senkronize eder: current_harvests=0, state="resting",
    dinlenme zaman damgası yazılır. PartialNodeDetector'ın tam-hasat ve
    kısmi-hasat dallarındaki aynı döngünün tek kaynağıdır.
    """
    for obj in (node_obj, node_or_id if isinstance(node_or_id, (NodeData, dict)) else None):
        if obj is None or obj is updated_node:
            continue
        _apply_node_harvest_state(
            obj,
            current_harvests=0,
            state="resting",
            rest_sec=rest_sec,
            now_ts=now_ts,
            stamp_rest=True,
        )


def _node_ready_for_fresh_cycle(node_obj: Any, now: Optional[float] = None) -> bool:
    """
    F-0320: tam hasatlı (current >= max) düğümde yeni döngüye başlamadan
    önce hazır olma kontrolü. NodeData.is_ready() varsa onu, dict ise
    state/last_visited_timestamp/rest_duration_sec ile aynı mantığı kullanır.
    Bilgi yoksa hazır sayılır (eski davranış korunur).
    """
    if node_obj is None:
        return True
    if hasattr(node_obj, "is_ready") and callable(node_obj.is_ready):
        try:
            return bool(node_obj.is_ready(now=now))
        except Exception:
            return True
    if isinstance(node_obj, dict):
        state = str(node_obj.get("state", "active") or "active")
        try:
            curr = int(node_obj.get("current_harvests", 0) or 0)
            max_h = max(1, int(node_obj.get("max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS))
        except (TypeError, ValueError):
            return True
        if state in ("active", "gathering") and 0 < curr < max_h:
            return True
        try:
            last = float(node_obj.get("last_visited_timestamp", 0) or 0)
        except (TypeError, ValueError):
            return True
        if last <= 0.0:
            return state != "resting"
        try:
            duration = int(node_obj.get("rest_duration_sec", 0) or 0)
        except (TypeError, ValueError):
            return True
        if duration <= 0:
            return True
        current_time = time.time() if now is None else float(now)
        return (current_time - last) >= float(duration)
    return True


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
        # F-0152: çözülemeyen referans sessizce düğüm #1 olmasın.
        if node_id is None:
            return {
                "node_id": None,
                "state": self.state,
                "is_anchored": False,
                "current_harvests": self.current_harvests,
                "max_harvests": self.max_harvests,
                "target_stage": f"{self.current_harvests + 1}/{self.max_harvests}",
                "gather_key": self.gather_key,
                "key_pressed": False,
                "not_ready": True,
                "reason": "UNRESOLVED_NODE",
            }
        self.active_node_id = int(node_id)
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

        # F-0320: tamamlanmış düğümde sayacı sıfırlayıp yeniden toplama;
        # önce hazır mı kontrol edilir — dinlenmedeyse başlatılmaz.
        if initial_curr >= resolved_max:
            ready_obj = node_obj
            if ready_obj is None and self.route_manager is not None and hasattr(self.route_manager, "get_node"):
                try:
                    ready_obj = self.route_manager.get_node(int(node_id), region=region)
                except Exception:
                    ready_obj = None
            if not _node_ready_for_fresh_cycle(ready_obj if ready_obj is not None else self.active_node):
                self.active_node_id = int(node_id)
                return {
                    "node_id": int(node_id),
                    "state": self.state,
                    "is_anchored": False,
                    "current_harvests": initial_curr,
                    "max_harvests": resolved_max,
                    "target_stage": f"{initial_curr + 1}/{resolved_max}",
                    "gather_key": self.gather_key,
                    "key_pressed": False,
                    "not_ready": True,
                    "reason": "NODE_NOT_READY",
                }
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
        # F-0474: max_retries_per_stage artık doğrudan çağrı yolunda da
        # uygulanır — kota dolunca yeniden basılmaz, aşama bırakılır.
        retries_left = self.max_retries_per_stage - self.extraction_retries
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
            retry_press=bool(retry_press) and retries_left > 0,
            test_mode=self.test_mode,
        )
        if res.get("extraction_failed"):
            self.extraction_retries += 1
            if self.extraction_retries >= self.max_retries_per_stage:
                # Kota doldu: düğümde çakılı kalma, sabiti bırak.
                res["retries_exhausted"] = True
                self.is_anchored = False
                self.state = self.STATE_IDLE
            else:
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
            start_res = self.start_node(node_or_id, gather_key=gather_key, region=region)
            if start_res.get("not_ready"):
                return {
                    "status": "NODE_NOT_READY",
                    "node_id": start_res.get("node_id"),
                    "current_harvests": self.current_harvests,
                    "max_harvests": self.max_harvests,
                    "stage": f"{self.current_harvests}/{self.max_harvests}",
                    "extraction_failed": False,
                    "re_pressed": False,
                    "is_anchored": False,
                    "node_depleted": False,
                    "partial_node": False,
                    "reason": start_res.get("reason", "NODE_NOT_READY"),
                    "state": self.state,
                }

        target_ref = node_or_id if node_or_id is not None else (self.active_node or self.active_node_id)
        node_id, node_obj = _resolve_node_reference(target_ref, route_mgr=self.route_manager, region=region)
        # F-0152: çözülemeyen referans sessizce düğüm #1 olmasın.
        if node_id is None and self.active_node_id is None:
            return {
                "status": "UNRESOLVED_NODE",
                "node_id": None,
                "current_harvests": self.current_harvests,
                "max_harvests": self.max_harvests,
                "stage": f"{self.current_harvests}/{self.max_harvests}",
                "extraction_failed": False,
                "re_pressed": False,
                "is_anchored": self.is_anchored,
                "node_depleted": False,
                "partial_node": False,
                "state": self.state,
            }
        resolved_id = int(node_id if node_id is not None else self.active_node_id)
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
                # F-0474: kota dolduysa RETRY_EXHAUSTED dön, çakılı kalma.
                if fail_res.get("retries_exhausted"):
                    return {
                        "status": "RETRY_EXHAUSTED",
                        "node_id": resolved_id,
                        "current_harvests": self.current_harvests,
                        "max_harvests": self.max_harvests,
                        "stage": f"{self.current_harvests}/{self.max_harvests}",
                        "retry_stage": fail_res["retry_stage"],
                        "extraction_failed": True,
                        "retries_exhausted": True,
                        "re_pressed": fail_res["re_pressed"],
                        "is_anchored": False,
                        "node_depleted": False,
                        "partial_node": False,
                        "state": self.state,
                    }
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
            # F-0474: aşama başarıyla ilerledi — retry sayacı sıfırlanır.
            self.extraction_retries = 0
        else:
            self.current_harvests += 1
            # F-0474: aşama başarıyla ilerledi — retry sayacı bu aşama için sıfırlanır.
            self.extraction_retries = 0
            is_depleted = self.current_harvests >= self.max_harvests
            # F-0147: yedek dalda tükenen düğüme dinlenme zaman damgası da
            # yazılır; yoksa last_visited=0 kalıp düğüm kalıcı ölü olurdu.
            rest_sec = int(round(self.rest_minutes * 60.0))
            now_ts = time.time()
            _apply_node_harvest_state(
                self.active_node,
                current_harvests=0 if is_depleted else self.current_harvests,
                state="resting" if is_depleted else "active",
                rest_sec=rest_sec,
                now_ts=now_ts,
                stamp_rest=bool(is_depleted),
            )

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
        # H-sweep (dilim 12 O-3): cast bitimini görsel kanıtla bekleyen
        # kanca. Verilmezse eski kör bekleme (cast_duration_sec) çalışır
        # ama sonuç "cast_verified: False" diye işaretlenir — aşama
        # sayacı "ölçüldü" sayılmaz.
        cast_complete_detector: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Bir düğümde baştan sona tam `1/3 -> 2/3 -> 3/3` dinamik toplama döngüsünü çalıştırır:
        - İlk `F` / `E` basımını yapar,
        - Her aşama sonunda `"Essence Extraction failed"` varsa hak düşmeden aynı aşamayı tekrarlar,
        - `1/3` ve `2/3` sonrasında `400ms` bekleyip `[F Gather]` / `[E]` kontrolü yapar,
        - Yarım bırakılmış kaynak (Partial Node) ise 3. toplamayı beklemeden `RESTING` yapar,
        - `3/3` tamamlandığında düğümü `RESTING` durumuna geçirir ve özet sözlüğü döndürür.
        """
        start_res = self.start_node(
            node_or_id=node_or_id,
            max_harvests=max_harvests,
            gather_key=gather_key,
            region=region,
            press_initial_key=True,
        )
        # F-0320: düğüm hazır değilse döngü hiç başlamaz.
        if start_res.get("not_ready"):
            return {
                "status": start_res.get("reason", "NODE_NOT_READY"),
                "node_id": start_res.get("node_id"),
                "harvests_completed": 0,
                "max_harvests": self.max_harvests,
                "stages_completed": [],
                "partial_node_detected": False,
                "extraction_failure_retries": 0,
                "reprompt_waits_sec": [],
                "node_state": "unknown",
                "is_anchored": False,
                "state": self.state,
                "gather_key": self.gather_key,
                "cast_verified": False,
            }

        reprompt_iter = iter(reprompt_sequence) if reprompt_sequence is not None else None
        fail_iter = iter(extraction_fail_sequence) if extraction_fail_sequence is not None else None

        total_casts = 0
        max_total_casts = self.max_harvests * (self.max_retries_per_stage + 1)
        last_step_result: Dict[str, Any] = {}
        cast_verified = False

        while self.is_anchored and total_casts < max_total_casts:
            total_casts += 1
            if callable(cast_complete_detector):
                # Cast bitimi görsel kanıtla beklenir (en fazla 3x süre).
                deadline = time.time() + max(self.cast_duration_sec * 3.0, 1.0)
                verified = False
                while time.time() < deadline:
                    try:
                        if bool(cast_complete_detector()):
                            verified = True
                            break
                    except Exception as exc:
                        logger.debug("cast_complete_detector hatası: %s", exc)
                        break
                    self._sleep(self.poll_interval_sec)
                cast_verified = bool(verified)
            elif self.cast_duration_sec > 0.0:
                self._sleep(self.cast_duration_sec)
                cast_verified = False

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
            # F-0148: sağlayıcı yokken re-prompt varsayılanı True kaldırıldı —
            # görsel kanıt olmadan 3/3'e ilerlenmez; reprompt_val None kalır ve
            # verify_reprompt_within_window gerçek görsel kontrolü yapar
            # (kanıt yoksa False → partial-node yolu işletilir).

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

            if last_step_result.get("status") in ("COMPLETED", "PARTIAL_DEPLETED", "RETRY_EXHAUSTED", "NODE_NOT_READY", "UNRESOLVED_NODE"):
                break

        # F-0032: kota dolup döngü bittiyse bot düğüme çakılı kalmasın;
        # status her durumda raporlanır.
        loop_status = str(last_step_result.get("status", "") or "")
        quota_hit = total_casts >= max_total_casts and self.is_anchored
        if quota_hit:
            self.is_anchored = False
            self.state = self.STATE_IDLE
            loop_status = "QUOTA_EXCEEDED"
        elif not loop_status:
            loop_status = "STOPPED"

        final_node_state = "resting" if not self.is_anchored else "active"
        if self.route_manager is not None and self.active_node_id is not None:
            rm_node = self.route_manager.get_node(self.active_node_id, region=region)
            if rm_node is not None:
                final_node_state = str(rm_node.state)

        return {
            "status": loop_status,
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
            # H-sweep (dilim 12 O-3): True yalnızca cast_complete_detector
            # cast bitimini gördüyse; kör 1.80 sn beklemede False kalır.
            "cast_verified": bool(cast_verified),
        }

    harvest_node = execute_harvest_loop
    run_harvest_cycle = execute_harvest_loop


# ==============================================================================
# MADDE 24 — MESAFE BAZLI OTOMATİK BİNEK (`AutoMountManager`)
# ==============================================================================
# H-sweep (dilim 12 Y-4) — KAPANDI (Selçuk kararı, 2026-11-11): binek tuşu
# "C"dir, "R" değil. R, sol-tık skill tuşudur (sağ tık -> T, sol tık -> R);
# binek C olduğu için çakışma yoktur. Varsayılan buna göre "c"dir.
MOUNT_KEY_DEFAULT: str = "c"
# H-sweep (dilim 12 D-4): başarısız binek denemesi her tick'te
# tekrarlanmasın diye varsayılan bekleme (eskiden 0.0 = spam engeli kapalı).
MOUNT_COOLDOWN_DEFAULT_SEC: float = 5.0


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
        mount_key: str = MOUNT_KEY_DEFAULT,
        gather_key: str = "f",
        mount_cooldown_sec: float = MOUNT_COOLDOWN_DEFAULT_SEC,
        cannot_mount_backoff_sec: float = 2.5,
        gather_press_cooldown_sec: float = 1.0,
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
        self.mount_key: str = str(mount_key or MOUNT_KEY_DEFAULT).strip().lower() or MOUNT_KEY_DEFAULT
        self.gather_key: str = str(gather_key or "f").strip().lower() or "f"
        self.mount_cooldown_sec: float = max(0.0, float(mount_cooldown_sec))
        self.cannot_mount_backoff_sec: float = max(0.0, float(cannot_mount_backoff_sec))
        # F-0321: GATHER dalında her tick'te F basmayı engelleyen debounce.
        self.gather_press_cooldown_sec: float = max(0.0, float(gather_press_cooldown_sec))
        self._last_gather_press_ts: float = 0.0
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
        # F-0149: tuş basımı başarısızsa binekli sayılmaz — eskiden
        # pressed=False olsa bile is_mounted=True yapılıyordu.
        if not pressed:
            return False
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
        # S-0402: tuş basımı başarısızsa (kontrolcü yok / gönderilemedi)
        # binekten inilmiş sayılmaz — mount_up() için F-0149 ile
        # düzeltilen hatanın dismount dalındaki karşılığı. Durum
        # bayrakları ve sayaçlar yalnızca gerçek basımda güncellenir;
        # toplama tuşuna da basılmaz (hâlâ binekli sayılır).
        if not dismount_pressed:
            return {
                "dismounted": False,
                "dismount_key_pressed": False,
                "started_gathering": False,
                "gather_key_pressed": False,
                "gather_key": used_gather_key,
                "is_mounted": self.is_mounted,
            }
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
            # S-2102: started_gathering toplama tuşunun gerçekten basılmasına
            # bağlıdır — gather_pressed False iken start_gathering=True
            # dönmek, basılamayan toplamayı başlamış gibi raporlardı
            # (update() Durum 2'deki doğru desenle aynı).
            "started_gathering": bool(gather_pressed),
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
                # F-0150: binekli sanılırken "Cannot mount" geldiyse bayrak
                # tek başına indirilmez; iniş tuşuna gerçekten basılır.
                self.dismount(
                    input_ctrl=input_ctrl or self.input_controller,
                    start_gathering=False,
                )

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
                # S-0402: dismount() başarısızsa (tuş basılamadı) iniş
                # raporlanmaz — eskiden burada da koşulsuz
                # "dismounted": True dönülüyordu.
                dismounted_ok = bool(dismount_res.get("dismounted", False))
                return {
                    "action": (
                        ("DISMOUNT_AND_GATHER" if should_gather_now else "DISMOUNT")
                        if dismounted_ok
                        else "DISMOUNT_FAILED"
                    ),
                    "mounted": self.is_mounted,
                    "dismounted": dismounted_ok,
                    "started_gathering": bool(should_gather_now and dismounted_ok),
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
                # F-0150: biniliyor iken mounted False dönülemez.
                "mounted": True,
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
            # F-0321: her tick'te F basılmaz; debounce aralığı beklenir.
            gather_pressed = False
            if (curr_time - self._last_gather_press_ts) >= self.gather_press_cooldown_sec:
                gather_pressed = _send_key_via_controller(ctrl, used_gather_key, test_mode=self.test_mode)
                if gather_pressed:
                    self._last_gather_press_ts = curr_time
            return {
                "action": "GATHER",
                "mounted": False,
                "dismounted": False,
                "started_gathering": bool(gather_pressed),
                "gather_key_pressed": bool(gather_pressed),
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
            # F-0149: mount_up başarısızsa MOUNT raporlanmaz.
            mounted_ok = self.mount_up(input_ctrl=ctrl)
            return {
                "action": "MOUNT" if mounted_ok else "MOUNT_FAILED",
                "mounted": bool(mounted_ok),
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


# H-sweep (dilim 12 Y-2 / dilim 13 O-9): hedef eşleşme sınırı. Beklenen
# ofsetten bu kadar uzaktaki bir filiz "hedef" sayılmaz — gerçek hedef
# minimap dışındaysa listedeki başka bir düğüme yönelmek yanlıştır.
MAX_TARGET_OFFSET_DEVIATION_PX: float = 60.0
# Yedek dalda parlak kontur, canavar noktasının bu yarıçapı içindeyse
# hedef sayılmaz (beyaz canavar noktası parlak kontur olarak görünür).
MONSTER_EXCLUSION_RADIUS_PX: float = 10.0


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
        arrival_distance_m: float = DEFAULT_WAYPOINT_RADIUS_M,
        meters_per_minimap_px: float = MINIMAP_METERS_PER_PX,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
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
        self.deadband_deg: float = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))
        self.waypoint_radius_px: float = float(max(1.0, waypoint_radius_px))
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
        hesaplar. ±5°..±8° ölü bölge sönümlemesi uygulayarak yalpalanmayı önler.
        """
        if minimap_offset is not None:
            dx = float(minimap_offset[0])
            dy = float(minimap_offset[1])
        elif player_pos is not None and target_pos is not None:
            dx = float(target_pos[0]) - float(player_pos[0])
            dy = float(target_pos[1]) - float(player_pos[1])
        else:
            # F-0322: oyuncu konumu yokken mutlak harita koordinatı minimap
            # ofseti sayılmaz; yön bilinmiyor demektir (0, 0).
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
        damped = apply_turn_damping(heading_error, deadband_deg=self.deadband_deg)

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
            "deadband_deg": self.deadband_deg,
            "in_deadband": damped["in_deadband"],
            "turn_needed": damped["turn_needed"],
            "turn_direction": damped["turn_direction"],
            "turn_duration_sec": damped["turn_duration_sec"],
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
            # 1. MinimapScanner.scan_gatherable_materials ile toplanabilir
            # filiz/materyal düğümlerini tara (F-0151: scan_minimap diye bir
            # metot yok; bu dal ölü koddu).
            # H-sweep (dilim 13 K2): canavar noktaları artık yalnızca radar
            # çizimi için değil, hedef seçiminde de kullanılır — yedek dal
            # canavar noktasını hedef saymaz, sonuçta sayısı döndürülür.
            scan_res: Dict[str, Any] = {}
            if hasattr(self.minimap_scanner, "scan_gatherable_materials"):
                scan_res = self.minimap_scanner.scan_gatherable_materials(frame)
                gatherable = scan_res.get("gatherable_nodes", []) if isinstance(scan_res, dict) else []
                if gatherable:
                    # Eğer target_node ve player_map_pos verilmişse beklenen ofsete en yakın olanı seç
                    deviation_px: Optional[float] = None
                    if target_node is not None and player_map_pos is not None:
                        tx = float(target_node["map_x"] if isinstance(target_node, dict) else getattr(target_node, "map_x", 0))
                        ty = float(target_node["map_y"] if isinstance(target_node, dict) else getattr(target_node, "map_y", 0))
                        exp_dx = tx - float(player_map_pos[0])
                        exp_dy = ty - float(player_map_pos[1])
                        chosen = min(
                            gatherable,
                            key=lambda n: math.hypot(float(n.get("dx", 0.0)) - exp_dx, float(n.get("dy", 0.0)) - exp_dy),
                        )
                        deviation_px = math.hypot(
                            float(chosen.get("dx", 0.0)) - exp_dx,
                            float(chosen.get("dy", 0.0)) - exp_dy,
                        )
                        # H-sweep (dilim 12 Y-2): maksimum sapma eşiği yoktu —
                        # gerçek hedef minimap dışındaysa listedeki başka bir
                        # düğüm hedef diye seçiliyordu. Eşik aşılırsa hedef
                        # minimap'te görünmüyor demektir; yanlış düğüme
                        # yönelmek yerine None dönülür.
                        if deviation_px > MAX_TARGET_OFFSET_DEVIATION_PX:
                            return None
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
                        "target_match_deviation_px": (round(deviation_px, 2) if deviation_px is not None else None),
                        "nearby_monster_dots": len(scan_res.get("monster_dots", []) or []) + len(scan_res.get("hostile_dots", []) or []),
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
                # H-sweep (dilim 13 O-9 + K2): yedek dal en parlak konturu
                # hedef sayıyordu; beyaz canavar noktası da parlaktır ve
                # bot canavara yönlendirilebilirdi. Canavar/düşman
                # noktasına yakın konturlar elenir (konumlar kırpım
                # koordinatındadır, kontur merkezleriyle aynı uzay).
                monster_positions: List[Tuple[float, float]] = []
                for _key in ("monster_dots", "hostile_dots"):
                    for _dot in scan_res.get(_key, []) or []:
                        _pos = _dot.get("pos") if isinstance(_dot, dict) else None
                        if _pos and len(_pos) >= 2:
                            monster_positions.append((float(_pos[0]), float(_pos[1])))
                excluded = 0

                def _contour_centroid(c: Any) -> Tuple[float, float]:
                    m = cv2.moments(c)
                    if m["m00"] > 1e-5:
                        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
                    x, y, w, h = cv2.boundingRect(c)
                    return float(x + w / 2.0), float(y + h / 2.0)

                def _near_monster(c: Any) -> bool:
                    cx, cy = _contour_centroid(c)
                    return any(
                        math.hypot(cx - mx, cy - my) <= MONSTER_EXCLUSION_RADIUS_PX
                        for mx, my in monster_positions
                    )

                valid = [c for c in valid if not _near_monster(c)]
                if not valid:
                    logger.debug(
                        "locate_node_on_minimap: yedek dalda yalnızca canavar "
                        "noktası konturu bulundu, hedef yok sayıldı"
                    )
                    return None
                # F-0475: en yakın kontur ağırlık merkeziyle (moments) seçilir;
                # eskiden boundingRect sol-üst köşesiyle seçiliyordu.

                best_c = min(
                    valid,
                    key=lambda c: math.hypot(
                        _contour_centroid(c)[0] - pcx,
                        _contour_centroid(c)[1] - pcy,
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
                        "fallback_bright_contour": True,
                        "nearby_monster_dots": len(monster_positions),
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
                # F-0322: oyuncu konumu yokken mutlak (x, y) ofset sayılmaz.
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
                # F-0322: oyuncu konumu yokken mutlak map_x/map_y ofset sayılmaz.

        # F-0033: yönelilecek ofset yoksa (hedefsiz navigasyon) "vardım"
        # denemez; sıfır vektör varış sayılmaz.
        if minimap_offset is None:
            return {
                "used_mapless": False,
                "map_opened": False,
                "reason": "NO_TARGET",
                "distance_m": float(distance_m) if distance_m is not None else 0.0,
                "max_micro_nav_distance_m": self.max_micro_nav_distance_m,
                "visible_on_minimap": False,
                "dx": 0.0,
                "dy": 0.0,
                "steering_vector": (0.0, 0.0),
                "heading_angle_deg": 0.0,
                "heading_deg": 0.0,
                "steered": False,
                "arrived": False,
            }

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
                        # F-0323: tanınan girdi metodu yokken yönelindi sayılmaz.
                        steered = False
                except Exception as exc:
                    logger.debug("Haritasız mikro-navigasyon girdi hatası: %s", exc)
            # F-0323: kontrolcü yokken girdi gönderilmedi; steered False kalır.

        # F-0323: sayaçlar girdiden önce değil, başarılı yönelimden sonra artar.
        if steered:
            self.micro_nav_steps += 1
            self.map_open_calls_avoided += 1

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
            "deadband_deg": steer_data.get("deadband_deg", self.deadband_deg),
            "in_deadband": steer_data.get("in_deadband", False),
            "turn_needed": steer_data.get("turn_needed", True),
            "turn_direction": steer_data.get("turn_direction", "straight"),
            "turn_duration_sec": steer_data.get("turn_duration_sec", 0.0),
            "visible_on_minimap": is_vis,
            "screen_steer_point": (sx, sy),
            "steered": bool(steered),
            # F-0033/F-0324: varış için hem metre hem piksel eşiği birlikte
            # sağlanmalı (OR, 11m uzaktan "vardım" dedirtiyordu).
            "arrived": bool(effective_dist_m <= self.arrival_distance_m and steer_data["pixel_distance"] <= self.waypoint_radius_px),
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


class NavigationIntel:
    """
    NavigationIntel sınıfı, toplanan harita, binek, radar takılma tespiti,
    ölü bölge (deadband) sönümlemesi ve ardışık düğüm navigasyon zekası modüllerini
    bir araya getiren ana koordinasyon sınıfıdır.
    """
    def __init__(
        self,
        waypoint_radius_px: float = DEFAULT_WAYPOINT_RADIUS_PX,
        waypoint_radius_m: float = DEFAULT_WAYPOINT_RADIUS_M,
        stuck_timeout_sec: float = DEFAULT_STUCK_TIMEOUT_SEC,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        input_controller: Any = None,
        route_manager: Optional[RouteManager] = None,
        vision_engine: Any = None,
        **kwargs: Any,
    ) -> None:
        # H-sweep (dilim 12 D-2): **kwargs eskiden sessizce yutuluyordu —
        # yanlış yazılmış / desteklenmeyen ayar adı çağırana hiç belli
        # olmadan kayboluyordu. Bilinmeyen anahtar artık uyarıyla düşer.
        if kwargs:
            logger.warning(
                "NavigationIntel: bilinmeyen ayar(lar) yok sayıldı: %s",
                sorted(str(k) for k in kwargs),
            )
        self.kwargs = kwargs
        self.input_controller = input_controller
        self.route_manager = route_manager
        self.vision_engine = vision_engine

        self.waypoint_radius_px = float(waypoint_radius_px)
        self.waypoint_radius_m = float(waypoint_radius_m)
        self.stuck_timeout_sec = float(stuck_timeout_sec)
        self.deadband_deg = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))

        self.stuck_detector = RadarStuckDetector(
            stuck_timeout_sec=self.stuck_timeout_sec,
        )
        self.waypoint_detector = WaypointArrivalDetector(
            waypoint_radius_px=self.waypoint_radius_px,
            waypoint_radius_m=self.waypoint_radius_m,
        )
        self.turn_damper = TurnDampingController(
            deadband_deg=self.deadband_deg,
        )
        self.micro_navigator = MaplessMicroNavigator(
            input_controller=self.input_controller,
            vision_engine=self.vision_engine,
            deadband_deg=self.deadband_deg,
            waypoint_radius_px=self.waypoint_radius_px,
            arrival_distance_m=self.waypoint_radius_m,
        )

    def compute_bearing(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> float:
        """Calculates compass bearing (0°=North, 90°=East) from from_pos to to_pos."""
        return compute_bearing(from_pos, to_pos)

    def compute_angle_diff(self, target_bearing: float, player_heading: float) -> float:
        """Calculates shortest rotational angle difference in [-180°..+180°]."""
        return compute_angle_diff(target_bearing, player_heading)

    def apply_turn_damping(
        self,
        angle_diff: float,
        deadband_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Applies deadband (±5° to ±8°) and smooth turn duration interpolation."""
        db = float(deadband_deg) if deadband_deg is not None else self.deadband_deg
        return apply_turn_damping(angle_diff, deadband_deg=db)

    def check_waypoint_arrival(
        self,
        player_pos: Optional[Tuple[float, float]],
        waypoint_pos: Optional[Tuple[float, float]],
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Checks if player reached waypoint radius (<=20px or <=3m)."""
        return self.waypoint_detector.is_arrival_reached(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            distance_m=distance_m,
            pixel_distance=pixel_distance,
        )

    def transition_to_next_node(
        self,
        nodes: Sequence[Any],
        current_index: int,
        route_planner: Optional[Any] = None,
        stay_on_partial_harvest: bool = False,
        now: Optional[float] = None,
    ) -> Tuple[Optional[Any], int, bool]:
        """Triggers clean transition to next sequential node."""
        return self.waypoint_detector.transition_to_next_node(
            nodes=nodes,
            current_index=current_index,
            route_planner=route_planner or self.route_manager,
            stay_on_partial_harvest=stay_on_partial_harvest,
            now=now,
        )

    def check_arrival_and_transition(
        self,
        player_pos: Optional[Tuple[float, float]],
        waypoint_pos: Optional[Tuple[float, float]],
        nodes: Sequence[Any],
        current_index: int,
        distance_m: Optional[float] = None,
        pixel_distance: Optional[float] = None,
        route_planner: Optional[Any] = None,
        stay_on_partial_harvest: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Checks arrival within radius (20px / 3m) and triggers sequential transition."""
        return self.waypoint_detector.check_and_transition(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            nodes=nodes,
            current_index=current_index,
            distance_m=distance_m,
            pixel_distance=pixel_distance,
            route_planner=route_planner or self.route_manager,
            stay_on_partial_harvest=stay_on_partial_harvest,
            now=now,
        )

    def update_stuck_status(
        self,
        player_pos: Optional[Tuple[float, float]],
        movement_keys_held: bool = True,
        distance_is_meters: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Tracks player position. Detects stuck if position unchanged for >3.0s while moving."""
        return self.stuck_detector.update_position(
            player_pos=player_pos,
            movement_keys_held=movement_keys_held,
            distance_is_meters=distance_is_meters,
            now=now,
        )

    def execute_unstick(
        self,
        input_controller: Any = None,
        target_bearing: Optional[float] = None,
        player_heading: Optional[float] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """Triggers unstick maneuver (Space -> Strafe A+W/D+W -> Realign)."""
        ctrl = input_controller or self.input_controller
        return self.stuck_detector.execute_unstick_maneuver(
            input_controller=ctrl,
            target_bearing=target_bearing,
            player_heading=player_heading,
            sleep_fn=sleep_fn,
            deadband_deg=self.deadband_deg,
        )

    def step_navigation_cycle(
        self,
        player_pos: Optional[Tuple[float, float]],
        player_heading: Optional[float],
        waypoint_pos: Optional[Tuple[float, float]],
        nodes: Sequence[Any],
        current_index: int,
        movement_keys_held: bool = True,
        input_controller: Any = None,
        distance_m: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Unified navigation cycle step:
        1. Stuck detection: monitors position over 3.0s while movement keys held.
        2. Arrival & sequencing: within 20px / 3m -> clean transition to next sequential node.
        3. Bearing & shortest turn calculation (North=0°, East=90°, 180° inversion eliminated).
        4. Turn damping & deadband (±5° to ±8°) preventing rapid steering oscillation.
        """
        curr_ts = time.time() if now is None else float(now)
        ctrl = input_controller or self.input_controller

        # 1. Stuck check
        stuck_info = self.update_stuck_status(
            player_pos=player_pos,
            movement_keys_held=movement_keys_held,
            now=curr_ts,
        )
        unstick_res: Optional[Dict[str, Any]] = None
        if stuck_info["is_stuck"]:
            target_b = compute_bearing(player_pos, waypoint_pos) if (player_pos and waypoint_pos) else None
            unstick_res = self.execute_unstick(
                input_controller=ctrl,
                target_bearing=target_b,
                player_heading=player_heading,
            )

        # 2. Waypoint arrival check & transition
        arrival_res = self.check_arrival_and_transition(
            player_pos=player_pos,
            waypoint_pos=waypoint_pos,
            nodes=nodes,
            current_index=current_index,
            distance_m=distance_m,
            now=curr_ts,
        )

        # 3. Bearing & Damped Steering
        # Durum taşıyan TurnDampingController kullanılır: ham apply_turn_damping
        # çağrısı hızlı yön tersine çevirmelerdeki reversal-damping'i (0.65x,
        # 0.12 sn soğuma) hiç uygulamıyordu — __init__'te kurulan turn_damper
        # bu döngüde ölüydü. deadband canlı öznitelikten senkronize edilir.
        bearing = None
        turn_data = None
        if player_pos is not None and waypoint_pos is not None:
            bearing = compute_bearing(player_pos, waypoint_pos)
            if player_heading is not None:
                self.turn_damper.deadband_deg = max(
                    MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(self.deadband_deg))
                )
                turn_data = self.turn_damper.calculate_turn(
                    bearing, player_heading, now=curr_ts
                )

        return {
            "stuck_detected": bool(stuck_info["is_stuck"]),
            "stuck_info": stuck_info,
            "unstick_executed": bool(unstick_res is not None),
            "unstick_info": unstick_res,
            "arrival_info": arrival_res,
            "target_bearing": bearing,
            "player_heading": player_heading,
            "turn_data": turn_data,
        }

    def get_navigation_intel_health_summary(self) -> dict:
        # F-0478: hazır bayrağı gerçek proba bağlı — eskiden koşulsuz True.
        ready, checks = _probe_navigation_intel_ready(self)
        return {"module": "navigation_intel", "ready": ready, "checks": checks}

    def get_navigation_intel_ready_flag(self) -> dict:
        # F-0478: hazır bayrağı gerçek proba bağlı — eskiden koşulsuz True.
        ready, checks = _probe_navigation_intel_ready(self)
        return {"module": "navigation_intel", "ready": ready, "checks": checks}

    def get_navigation_intel_version_info(self) -> dict:
        # F-0478: sürüm bilgisi sabit kalır ama ready proba bağlıdır.
        ready, checks = _probe_navigation_intel_ready(self)
        return {
            "module": "navigation_intel",
            "version": __version__,
            "ready": ready,
            "checks": checks,
        }

    def get_navigation_intel_diagnostic_snapshot(self) -> dict:
        # F-0478: status/ready proba bağlı; sayaçlar gerçek durumdur.
        # Prob başarısızken (bileşen eksikse) sayaç okuma çökmez.
        ready, checks = _probe_navigation_intel_ready(self)
        return {
            "module": "navigation_intel",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "checks": checks,
            "stuck_count": getattr(
                getattr(self, "stuck_detector", None), "stuck_count", None
            ),
            "transitions_count": getattr(
                getattr(self, "waypoint_detector", None), "transitions_count", None
            ),
            "deadband_deg": getattr(self, "deadband_deg", None),
        }


__all__ = [
    "DEFAULT_REPROMPT_WAIT_SEC",
    "DEFAULT_REPROMPT_WAIT_MS",
    "MOUNT_DISTANCE_THRESHOLD_M",
    "DISMOUNT_DISTANCE_THRESHOLD_M",
    "MAPLESS_DISTANCE_THRESHOLD_M",
    "DEFAULT_DEADBAND_DEG",
    "MIN_DEADBAND_DEG",
    "MAX_DEADBAND_DEG",
    "DEFAULT_WAYPOINT_RADIUS_PX",
    "DEFAULT_WAYPOINT_RADIUS_M",
    "DEFAULT_STUCK_TIMEOUT_SEC",
    "ThreeStageHarvestController",
    "install_three_stage_harvest_controller",
    "install_navigation_intel",
    "PartialNodeDetector",
    "handle_extraction_failure",
    "AutoMountManager",
    "MaplessMicroNavigator",
    "RadarStuckDetector",
    "WaypointArrivalDetector",
    "TurnDampingController",
    "NavigationIntel",
    "compute_bearing",
    "compute_angle_diff",
    "apply_turn_damping",
    "get_navigation_intel_contract_checks",
    "get_navigation_intel_readiness_status",
    "get_navigation_intel_audit_record",
    "get_navigation_intel_operational_digest",
    "get_navigation_intel_telemetry_payload",
    "get_navigation_intel_integrity_seal",
    "get_navigation_intel_compliance_certificate",
    "get_navigation_intel_readiness_probe",
    "get_navigation_intel_audit_summary",
]


def get_navigation_intel_capabilities() -> list:
    return ["navigation_intel", "spatial_heuristics", "heading_alignment", "turn_damping", "stuck_detection", "waypoint_sequencing", "diagnostic_snapshot"]


# S-2101: readiness zinciri koşulsuz True dönemez. Yukarıdaki capabilities()
# listesi sabittir, `bool(...)` hiçbir zaman False olamazdı; readiness /
# audit / seal / certificate fonksiyonları da gerçek durum yoklamıyordu.
# Düzeltme: her yetenek adı bu modüldeki gerçek bir sembole bağlanır ve
# kontrat doğrulaması o sembollerin tanımlı/çağrılabilir olduğunu yoklar.
# Bir NavigationIntel örneği verilirse ayrıca canlı prob
# (_probe_navigation_intel_ready, F-0478) çalışır; verilmezse sonuç
# yalnızca modül-kontrat yoklamasına dayanır ve "instance_probed": False
# ile işaretlenir — canlı durum yoklanmamışken yoklanmış gibi raporlanmaz.
_NAVIGATION_INTEL_CAPABILITY_SYMBOLS: Dict[str, str] = {
    "navigation_intel": "NavigationIntel",
    "spatial_heuristics": "compute_bearing",
    "heading_alignment": "compute_angle_diff",
    "turn_damping": "TurnDampingController",
    "stuck_detection": "RadarStuckDetector",
    "waypoint_sequencing": "WaypointArrivalDetector",
    "diagnostic_snapshot": "get_navigation_intel_summary",
}


def get_navigation_intel_contract_checks() -> Dict[str, bool]:
    """Her yetenek için gerçek sembol yoklaması (True = sembol tanımlı ve çağrılabilir)."""
    checks: Dict[str, bool] = {}
    for capability, symbol in _NAVIGATION_INTEL_CAPABILITY_SYMBOLS.items():
        obj = globals().get(symbol)
        checks[capability] = obj is not None and callable(obj)
    return checks


def validate_navigation_intel_contract() -> bool:
    checks = get_navigation_intel_contract_checks()
    return bool(checks) and all(checks.values())


# H-sweep (dilim 12 K-1): üretim bağlantı bayrağı. Varsayılan False'dur;
# yalnızca install_three_stage_harvest_controller() /
# install_navigation_intel() True yapar. "ready/certified" bayrakları
# modül öz-deneyidir; üretime bağlılık ayrı ölçülür.
_PRODUCTION_WIRED: bool = False


def install_three_stage_harvest_controller(
    route_manager: Any = None,
    input_controller: Any = None,
    **kwargs: Any,
) -> "ThreeStageHarvestController":
    """`ThreeStageHarvestController` üretim kurulumu (C grubu işi).

    state_machine.py tarafı bu yardımcıyı çağırıp dönen denetleyiciyi
    toplama döngüsüne bağlamalıdır; çağrılmadığı sürece Harvest sayacı
    üretimde ilerleyemez (dilim 12 K-1 — controller'ın kurulumu yoktu).
    Kurulum bayrağını kaldırır, denetleyiciyi döndürür.
    """
    global _PRODUCTION_WIRED
    controller = ThreeStageHarvestController(
        route_manager=route_manager,
        input_controller=input_controller,
        **kwargs,
    )
    _PRODUCTION_WIRED = True
    return controller


def install_navigation_intel(**kwargs: Any) -> "NavigationIntel":
    """`NavigationIntel` üretim kurulumu; bayrağı kaldırır, örneği döndürür."""
    global _PRODUCTION_WIRED
    intel = NavigationIntel(**kwargs)
    _PRODUCTION_WIRED = True
    return intel


def _navigation_intel_verdict(intel: Any = None) -> Tuple[bool, Dict[str, bool], bool]:
    """(verdict, checks, instance_probed): kontrat yoklaması her zaman çalışır;
    örnek verilirse canlı prob da koşulur ve karar ikisinin VE'sidir.
    "production_wired" kontrat kararı değildir — checks içinde ayrıca
    taşınır, ölü modül "hazır" görünmesin diye.
    """
    checks = get_navigation_intel_contract_checks()
    contract_ok = bool(checks) and all(checks.values())
    if intel is None:
        merged = dict(checks)
        merged["production_wired"] = bool(_PRODUCTION_WIRED)
        return contract_ok, merged, False
    try:
        ready, probe_checks = _probe_navigation_intel_ready(intel)
    except Exception:
        merged = dict(checks)
        merged["instance_probe"] = False
        merged["production_wired"] = bool(_PRODUCTION_WIRED)
        return False, merged, True
    merged = dict(checks)
    merged.update({f"instance:{key}": value for key, value in probe_checks.items()})
    merged["production_wired"] = bool(_PRODUCTION_WIRED)
    return bool(contract_ok and ready), merged, True


def get_navigation_intel_baseline_metrics() -> dict:
    return {"module": "navigation_intel", "contract_valid": validate_navigation_intel_contract(), "capability_count": len(get_navigation_intel_capabilities()), "contract_checks": get_navigation_intel_contract_checks()}


def get_navigation_intel_readiness_status(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "ready": ready, "instance_probed": probed, "checks": checks, "metrics": get_navigation_intel_baseline_metrics()}


def get_navigation_intel_audit_record(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "audit_passed": ready, "instance_probed": probed, "checks": checks, "readiness": get_navigation_intel_readiness_status(intel)}


def get_navigation_intel_operational_digest(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "operational": ready, "instance_probed": probed, "checks": checks, "audit": get_navigation_intel_audit_record(intel)}


def get_navigation_intel_telemetry_payload(intel: Any = None) -> dict:
    return {"module": "navigation_intel", "payload": get_navigation_intel_operational_digest(intel)}


def get_navigation_intel_integrity_seal(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "sealed": ready, "instance_probed": probed, "checks": checks, "telemetry": get_navigation_intel_telemetry_payload(intel)}


def get_navigation_intel_compliance_certificate(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "certified": ready, "instance_probed": probed, "checks": checks, "seal": get_navigation_intel_integrity_seal(intel)}


def get_navigation_intel_readiness_probe(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "ready": ready, "instance_probed": probed, "checks": checks, "certificate": get_navigation_intel_compliance_certificate(intel)}


def get_navigation_intel_audit_summary(intel: Any = None) -> dict:
    ready, checks, probed = _navigation_intel_verdict(intel)
    return {"module": "navigation_intel", "audit_passed": ready, "instance_probed": probed, "checks": checks, "probe": get_navigation_intel_readiness_probe(intel)}


# F-0476: NodeData/RouteManager import'u dosya ortasından buraya (dosya
# sonuna) taşındı. Dairesel import güvenliği: route_manager.py NodeData'yı
# (satır 1243) ve RouteManager'ı (satır 1584) kendi 4703. satırındaki
# `from core.navigation_intel import ...`tan ÖNCE tanımlar; bu modülün
# route_manager'ın kullandığı adları (PartialNodeDetector,
# handle_extraction_failure) ise bu satırdan önce tanımlanmıştır. Böylece
# import sırası (önce navigation_intel ya da önce route_manager) fark
# etmeksizin iki yön de çözülür; ortaya taşınırsa çöker.
from core.route_manager import NodeData, RouteManager  # noqa: E402,F401


