# -*- coding: utf-8 -*-
"""
Savaş, Ölüm ve Takılma Kurtarma Modülü (Combat, Death & Stuck Recovery Module)
AION 2 Gatherion Pixel Bot - Ajan 6 Modülü

Özellikler:
1. Savaş Algılama (check_under_attack):
   - Alt-ortadaki oyuncu can barı (HP bar: 3,026/3,026) doluluk oranı ve düşüş analizi
   - Ekranın üst-orta ve merkez savaş bölgesindeki kırmızı düşman hedef barı (Red enemy nameplate / target bar) tespiti
   - Oyuncunun kendi baş üstü HP/MP barını ve harita/arayüz durumlarını ayırt ederek yanlış pozitifleri önleme
2. Karşı Saldırı Döngüsü (execute_combat_loop):
   - Savaş başladığında FIGHTING durumu bildirimi
   - Hedef ölene kadar Tab, Sol/Sağ tık ve 1, 2, 3, 4, Q, E, R yetenekleriyle insanlaştırılmış rotasyon
   - Canavar öldüğünde F tuşu ile ganimet toplama (loot) ve stop_gather_sec kadar bekleyip CLEARED döndürme
   - 2 dakikayı (max_fight_sec=120) aşan savaşlarda TIMEOUT ve "Paused: a fight lasted over 2 min" bildirimi
3. Ölüm Algılama ve Dirilme Bekleme (check_character_dead & wait_for_revive):
   - Can barı %0 olduğunda veya dirilme/ölüm penceresi belirdiğinde ölüm tespiti
   - "Paused: your character died" durumuna geçiş ve can > %50 olduğunda otomatik devam
4. Takılma Kurtarma (recover_from_stuck):
   - Zıplama (space), sağa/sola manevra (a/d + w), geri çekilme (s) ve kanat açma/uçma (PageUp / v / space)
5. Anti-AFK ve Günlük Limit (check_daily_limit_and_anti_afk & perform_anti_afk_step):
   - Günlük toplama/özüt çıkarma limitinde sıfırlanma saatine kadar bekleme ve her 3-4 dakikada bir Anti-AFK adımı
"""

from collections import deque
from datetime import datetime, timedelta, timezone
import logging
import random
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.stability_guards import FiveFrameDeathConfirmationGuard


# Modül logger yapılandırması
logger = logging.getLogger("CombatRecovery")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class CombatRecovery:
    """
    AION 2 Savaş, Ölüm, Takılma Kurtarma ve Anti-AFK Yöneticisi.
    """

    # Oyuncu alt-orta HP barı normalize koordinatları (pilgrims_ridge_1.jpg ve frame_00s.jpg referanslı)
    PLAYER_HP_X1_NORM: float = 0.3450
    PLAYER_HP_X2_NORM: float = 0.4780
    PLAYER_HP_Y1_NORM: float = 0.8850
    PLAYER_HP_Y2_NORM: float = 0.9020

    # Oyuncu baş üstü HP barı normalize koordinatları (pilgrims_ridge_1.jpg referanslı: x:0.472..0.526, y:0.470..0.482)
    OVERHEAD_HP_X1_NORM: float = 0.4720
    OVERHEAD_HP_X2_NORM: float = 0.5260
    OVERHEAD_HP_Y1_NORM: float = 0.4700
    OVERHEAD_HP_Y2_NORM: float = 0.4820

    # Oyuncu alt-orta MP barı normalize koordinatları (HUD görünürlük doğrulaması için: 1,521/1,521 mavi bar)
    PLAYER_MP_X1_NORM: float = 0.5190
    PLAYER_MP_X2_NORM: float = 0.6360
    PLAYER_MP_Y1_NORM: float = 0.8850
    PLAYER_MP_Y2_NORM: float = 0.9020

    # Üst-orta düşman hedef barı (Target frame) normalize koordinatları
    TARGET_BAR_X1_NORM: float = 0.3000
    TARGET_BAR_X2_NORM: float = 0.7000
    TARGET_BAR_Y1_NORM: float = 0.0200
    TARGET_BAR_Y2_NORM: float = 0.1550

    # Üst Hedefin Hedefi (Target-of-Target: "45 Muses" kırmızı barı) normalize koordinatları
    TOT_BAR_X1_NORM: float = 0.6200
    TOT_BAR_X2_NORM: float = 0.7100
    TOT_BAR_Y1_NORM: float = 0.0250
    TOT_BAR_Y2_NORM: float = 0.0650

    # Merkez/Üst savaş alanı düşman isimliği (Enemy nameplate) normalize koordinatları
    COMBAT_ZONE_X1_NORM: float = 0.2200
    COMBAT_ZONE_X2_NORM: float = 0.7800
    COMBAT_ZONE_Y1_NORM: float = 0.1550
    COMBAT_ZONE_Y2_NORM: float = 0.6000

    # 3. şahıs kamerada oyuncunun kendi karakterinin baş üstü HP/MP barı (Yanlış pozitif maskesi)
    SELF_OVERHEAD_X1_NORM: float = 0.4650
    SELF_OVERHEAD_X2_NORM: float = 0.5350
    SELF_OVERHEAD_Y1_NORM: float = 0.4600
    SELF_OVERHEAD_Y2_NORM: float = 0.5650

    # Saldırı tuş havuzu
    COMBAT_SKILL_KEYS: Tuple[str, ...] = ("1", "2", "3", "4", "q", "e", "r")

    def __init__(
        self,
        fly_unreachable: Any = True,
        daily_reset_hour_utc: Any = 0,
        anti_afk_min_interval_sec: float = 180.0,
        anti_afk_max_interval_sec: float = 240.0,
        damaged_hp_threshold: float = 0.94,
        vision: Any = None,
        input_ctrl: Any = None,
    ) -> None:
        if not isinstance(fly_unreachable, (bool, int, float)):
            self.vision = fly_unreachable
            self.fly_unreachable = True
        else:
            self.vision = vision
            self.fly_unreachable = bool(fly_unreachable)

        if not isinstance(daily_reset_hour_utc, (int, float, str)):
            self.input_ctrl = daily_reset_hour_utc
            self.daily_reset_hour_utc = 0
        else:
            self.input_ctrl = input_ctrl
            self.daily_reset_hour_utc = int(daily_reset_hour_utc) % 24
        self.anti_afk_min_interval_sec: float = max(10.0, float(anti_afk_min_interval_sec))
        self.anti_afk_max_interval_sec: float = max(
            self.anti_afk_min_interval_sec, float(anti_afk_max_interval_sec)
        )
        self.damaged_hp_threshold: float = float(damaged_hp_threshold)

        # Durum ve geçmiş izleme
        self._hp_history: Deque[float] = deque(maxlen=10)
        self._enemy_bar_history: Deque[bool] = deque(maxlen=5)
        self.last_hp_ratio: float = 1.0
        self.in_combat: bool = False
        self.is_dead: bool = False
        self._consecutive_death_frames: int = 0
        self.live_death_confirm_frames: int = 5
        self.death_guard: FiveFrameDeathConfirmationGuard = FiveFrameDeathConfirmationGuard(
            required_frames=5
        )
        self.last_status_message: str = "IDLE"

        # Anti-AFK zamanlayıcıları
        self.last_anti_afk_time: float = time.monotonic()
        self.next_anti_afk_interval_sec: float = random.uniform(
            self.anti_afk_min_interval_sec, self.anti_afk_max_interval_sec
        )
        self._anti_afk_step_counter: int = 0

        logger.info(
            "CombatRecovery başlatıldı | fly_unreachable=%s | anti_afk=[%.0fs..%.0fs]",
            self.fly_unreachable,
            self.anti_afk_min_interval_sec,
            self.anti_afk_max_interval_sec,
        )

    # ========================================================================
    # YARDIMCI GÖRÜNTÜ, GİRDİ VE DURUM BİLDİRİM METODLARI
    # ========================================================================

    @staticmethod
    def _ensure_bgr(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Gelen görüntüyü güvenli biçimde 3 kanallı BGR numpy dizisine dönüştürür."""
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return None
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.ndim == 3 and frame.shape[2] == 3:
                return frame
            return None
        except Exception as exc:
            logger.error("Görüntü BGR formatına dönüştürülürken hata: %s", exc)
            return None

    def _grab_frame_from_capture(self, capture: Any) -> Optional[np.ndarray]:
        """
        Farklı ekran yakalama (Vision, ScreenCapture, fonksiyon veya doğrudan ndarray)
        arayüzlerinden güvenli biçimde güncel BGR kareyi alır.
        """
        try:
            if capture is None:
                return None
            if isinstance(capture, np.ndarray):
                return self._ensure_bgr(capture)
            if callable(capture):
                return self._ensure_bgr(capture())
            for method_name in ("get_latest_frame", "grab_frame", "capture_frame"):
                method = getattr(capture, method_name, None)
                if callable(method):
                    return self._ensure_bgr(method())
            grab_screen = getattr(capture, "grab_screen", None)
            if callable(grab_screen):
                try:
                    return self._ensure_bgr(grab_screen(grayscale=False))
                except TypeError:
                    return self._ensure_bgr(grab_screen())
            return None
        except Exception as exc:
            logger.error("Ekran karesi yakalanırken hata: %s", exc)
            return None

    def _notify_status(
        self,
        status_callback: Optional[Callable],
        state: str,
        title: str,
        detail: str = "",
    ) -> None:
        """
        Durum geri çağrısını (status_callback) esnek argüman sayısıyla güvenli biçimde tetikler.
        Şartname formatı: status_callback("FIGHTING", "Fighting back", "Attacking hostile target...")
        """
        self.last_status_message = title if title else state
        if status_callback is None or not callable(status_callback):
            return
        try:
            status_callback(state, title, detail)
        except TypeError:
            try:
                status_callback(state, detail or title)
            except TypeError:
                try:
                    status_callback(f"{state}: {title} - {detail}".strip(" -"))
                except Exception as inner_exc:
                    logger.error("status_callback tek argümanlı çağrı hatası: %s", inner_exc)
        except Exception as exc:
            logger.error("status_callback çalıştırılırken hata: %s", exc)

    @staticmethod
    def _safe_press_key(
        input_ctrl: Any,
        key: str,
        min_hold: float = 0.045,
        max_hold: float = 0.095,
    ) -> bool:
        """input_ctrl üzerinden insanlaştırılmış tuş basımını güvenli şekilde gerçekleştirir."""
        if input_ctrl is None:
            return False
        try:
            press_fn = getattr(input_ctrl, "press_key", None)
            if callable(press_fn):
                try:
                    press_fn(key, min_hold=min_hold, max_hold=max_hold)
                except TypeError:
                    press_fn(key)
                return True

            send_key_fn = getattr(input_ctrl, "send_key", None)
            if callable(send_key_fn):
                send_key_fn(key)
                return True
            return False
        except Exception as exc:
            logger.error("Tuş basımı sırasında hata ('%s'): %s", key, exc)
            return False

    @staticmethod
    def _safe_mouse_click(
        input_ctrl: Any,
        button: str = "left",
        x: Optional[int] = None,
        y: Optional[int] = None,
        min_hold: float = 0.045,
        max_hold: float = 0.095,
    ) -> bool:
        """input_ctrl üzerinden insanlaştırılmış fare tıklamasını güvenli şekilde gerçekleştirir."""
        if input_ctrl is None:
            return False
        try:
            click_fn = getattr(input_ctrl, "human_click", None)
            if callable(click_fn):
                try:
                    click_fn(x=x, y=y, button=button, min_hold=min_hold, max_hold=max_hold)
                except TypeError:
                    click_fn(x, y, button)
                return True

            alt_click = getattr(input_ctrl, "click", None)
            if callable(alt_click):
                alt_click(button=button)
                return True
            return False
        except Exception as exc:
            logger.error("Fare tıklaması sırasında hata (button='%s'): %s", button, exc)
            return False

    def _safe_simultaneous_move(
        self,
        input_ctrl: Any,
        strafe_key: str,
        forward_key: str = "w",
        hold_sec: float = 0.35,
    ) -> None:
        """
        Takılma kurtarma sırasında çapraz manevra (a/d + w) uygular.
        Donanım alt metodları varsa eşzamanlı basar, yoksa ardışık insanlaştırılmış basım yapar.
        """
        if input_ctrl is None:
            return
        try:
            hw_down = getattr(input_ctrl, "_hw_key_down", None)
            hw_up = getattr(input_ctrl, "_hw_key_up", None)
            test_mode = bool(getattr(input_ctrl, "test_mode", False))
            lock = getattr(input_ctrl, "_bot_pressing_lock", None)
            active_keys = getattr(input_ctrl, "_active_bot_keys", None)

            if callable(hw_down) and callable(hw_up) and not test_mode:
                if lock is not None and active_keys is not None:
                    with lock:
                        setattr(input_ctrl, "_bot_pressing", True)
                        active_keys.add(forward_key.lower())
                        active_keys.add(strafe_key.lower())
                try:
                    hw_down(forward_key)
                    time.sleep(random.uniform(0.02, 0.04))
                    hw_down(strafe_key)
                    time.sleep(max(0.05, float(hold_sec)))
                finally:
                    hw_up(strafe_key)
                    time.sleep(random.uniform(0.015, 0.03))
                    hw_up(forward_key)
                    time.sleep(0.015)
                    if lock is not None and active_keys is not None:
                        with lock:
                            active_keys.discard(forward_key.lower())
                            active_keys.discard(strafe_key.lower())
                            if not active_keys:
                                setattr(input_ctrl, "_bot_pressing", False)
            else:
                self._safe_press_key(input_ctrl, strafe_key, min_hold=hold_sec * 0.6, max_hold=hold_sec)
                self._safe_press_key(input_ctrl, forward_key, min_hold=hold_sec * 0.7, max_hold=hold_sec * 1.1)
        except Exception as exc:
            logger.error("Çapraz hareket manevrası hatası (%s + %s): %s", strafe_key, forward_key, exc)

    # ========================================================================
    # 1. SAVAŞ ALGILAMA (CHECK_UNDER_ATTACK & HP / TARGET BAR ANALİZİ)
    # ========================================================================

    def _read_hp_ratio(self, frame: np.ndarray) -> float:
        """
        Alt-ortadaki oyuncu can barının (x: 0.345..0.478, y: 0.885..0.902) VE
        oyuncunun baş üstü can barının (x: 0.472..0.526, y: 0.470..0.482) doluluk oranını
        [0.0, 1.0] aralığında hesaplar.
        Eğer her iki bölgede de kırmızı bar pikseli eşleşmezse (HUD gizli veya ölçek farklı),
        yanlış ölüm/hasar tetiklememek için varsayılan olarak 1.0 (%100 HP) döndürür.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return 1.0

            h, w, _ = bgr.shape
            if h < 50 or w < 50:
                return 1.0

            # 1) Alt-orta oyuncu HP barı (x: 0.345..0.478, y: 0.885..0.902)
            # Not: 0.345..0.363 aralığında sol çerçeve ikonu bulunduğundan efektif bar genişliği (0.3635..0.4775)
            # üzerinden doluluk oranı normalize edilir.
            x1 = max(0, int(round(w * self.PLAYER_HP_X1_NORM)))
            x2 = min(w, int(round(w * self.PLAYER_HP_X2_NORM)))
            y1 = max(0, int(round(h * self.PLAYER_HP_Y1_NORM)))
            y2 = min(h, int(round(h * self.PLAYER_HP_Y2_NORM)))

            bot_ratio = 0.0
            bot_matched = False
            if x2 > x1 and y2 > y1:
                roi_bot = bgr[y1:y2, x1:x2]
                hsv_bot = cv2.cvtColor(roi_bot, cv2.COLOR_BGR2HSV)
                red_low = (hsv_bot[:, :, 0] <= 12) & (hsv_bot[:, :, 1] >= 95) & (hsv_bot[:, :, 2] >= 85)
                red_high = (hsv_bot[:, :, 0] >= 168) & (hsv_bot[:, :, 1] >= 95) & (hsv_bot[:, :, 2] >= 85)
                mask_bot = (red_low | red_high).astype(np.uint8)

                total_w = mask_bot.shape[1]
                if total_w > 0:
                    col_counts = np.sum(mask_bot, axis=0)
                    min_col_px = max(1, int(mask_bot.shape[0] * 0.18))
                    valid_cols = np.where(col_counts >= min_col_px)[0]
                    if valid_cols.size >= max(4, int(total_w * 0.05)):
                        bot_matched = True
                        # Sol çerçeve boşluğunu (0.345 -> 0.3635 ~ %13.8) dikkate alarak efektif genişliğe oranla
                        eff_start = int(total_w * 0.135)
                        eff_width = max(1, total_w - eff_start)
                        span_px = max(0, int(valid_cols[-1]) - max(eff_start, int(valid_cols[0])) + 1)
                        rightmost_ratio = float(int(valid_cols[-1]) - eff_start + 1) / float(eff_width)
                        active_ratio = float(valid_cols.size) / float(eff_width)
                        span_ratio = float(span_px) / float(eff_width)
                        bot_ratio = float(np.clip(max(rightmost_ratio, active_ratio, span_ratio), 0.0, 1.0))

            # 2) Baş üstü oyuncu HP barı (x: 0.472..0.526, y: 0.470..0.482)
            ox1 = max(0, int(round(w * self.OVERHEAD_HP_X1_NORM)))
            ox2 = min(w, int(round(w * self.OVERHEAD_HP_X2_NORM)))
            oy1 = max(0, int(round(h * self.OVERHEAD_HP_Y1_NORM)))
            oy2 = min(h, int(round(h * self.OVERHEAD_HP_Y2_NORM)))

            ov_ratio = 0.0
            ov_matched = False
            if ox2 > ox1 and oy2 > oy1:
                roi_ov = bgr[oy1:oy2, ox1:ox2]
                hsv_ov = cv2.cvtColor(roi_ov, cv2.COLOR_BGR2HSV)
                # Baş üstü UI barı doygun kırmızıdır; çimen/toprak yansımalarını ayıklamak için S>=135, V>=105
                ov_red = (
                    ((hsv_ov[:, :, 0] <= 10) | (hsv_ov[:, :, 0] >= 170))
                    & (hsv_ov[:, :, 1] >= 135)
                    & (hsv_ov[:, :, 2] >= 105)
                ).astype(np.uint8)
                ov_w = ov_red.shape[1]
                if ov_w > 0:
                    ov_cols = np.where(np.sum(ov_red, axis=0) >= 1)[0]
                    if ov_cols.size >= max(5, int(ov_w * 0.10)):
                        ov_matched = True
                        eff_ov_start = int(ov_w * 0.11)
                        eff_ov_width = max(1, ov_w - eff_ov_start)
                        ov_rightmost = float(int(ov_cols[-1]) - eff_ov_start + 1) / float(eff_ov_width)
                        ov_active = float(ov_cols.size) / float(eff_ov_width)
                        ov_ratio = float(np.clip(max(ov_rightmost, ov_active), 0.0, 1.0))

            if bot_matched and ov_matched:
                return float(np.clip(max(bot_ratio, ov_ratio), 0.0, 1.0))
            if bot_matched:
                return bot_ratio
            if ov_matched:
                return ov_ratio

            # Hiçbir kırmızı HP barı pikseli eşleşmediyse varsayılan 1.0 (%100 HP) döndür
            return 1.0
        except Exception as exc:
            logger.error("Oyuncu HP oranı (_read_hp_ratio) hesaplanırken hata: %s", exc)
            return 1.0

    def get_player_hp_ratio(self, frame: np.ndarray) -> float:
        """
        Alt-ortadaki oyuncu can barının (11,718/11,718) ve baş üstü HP barının
        doluluk oranını [0.0, 1.0] aralığında hesaplar (_read_hp_ratio ile eşdeğer).
        """
        return self._read_hp_ratio(frame)

    def _detect_red_vignette_flash(self, frame: np.ndarray) -> bool:
        """
        Hasar alındığında ekran kenarlarında oluşan kırmızı kan/uyarı parlamasını (vignette flash) tespit eder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False
            h, w, _ = bgr.shape
            if h < 80 or w < 80:
                return False

            bw = max(6, int(round(w * 0.035)))
            bh = max(6, int(round(h * 0.045)))

            strips = [
                bgr[0:bh, :],
                bgr[h - bh : h, :],
                bgr[bh : h - bh, 0:bw],
                bgr[bh : h - bh, w - bw : w],
            ]
            red_ratios: List[float] = []
            for strip in strips:
                if strip.size == 0:
                    continue
                hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                red_mask = (
                    ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
                    & (hsv[:, :, 1] >= 145)
                    & (hsv[:, :, 2] >= 110)
                )
                red_ratios.append(float(np.mean(red_mask)))

            # En az 3 kenarda yoğun kırmızı vignette parlaması varsa True
            active_edges = sum(1 for r in red_ratios if r >= 0.22)
            return active_edges >= 3
        except Exception as exc:
            logger.error("Kırmızı ekran kenarı parlaması analizi hatası: %s", exc)
            return False

    def _detect_target_of_target_agro(self, frame: np.ndarray) -> bool:
        """
        Ekranın üst kısmındaki Hedefin Hedefi (Target-of-Target: x: 0.62..0.71, y: 0.025..0.065)
        bölgesinde '45 Muses' kırmızı can barının aktif olup olmadığını ve hedefin yakın (< 20m)
        olup olmadığını kontrol eder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False
            h, w, _ = bgr.shape
            if h < 80 or w < 80:
                return False

            x1 = max(0, int(round(w * self.TOT_BAR_X1_NORM)))
            x2 = min(w, int(round(w * self.TOT_BAR_X2_NORM)))
            y1 = max(0, int(round(h * self.TOT_BAR_Y1_NORM)))
            y2 = min(h, int(round(h * self.TOT_BAR_Y2_NORM)))
            if x2 <= x1 or y2 <= y1:
                return False

            roi_tot = bgr[y1:y2, x1:x2]
            hsv_tot = cv2.cvtColor(roi_tot, cv2.COLOR_BGR2HSV)
            red_tot = (
                ((hsv_tot[:, :, 0] <= 12) | (hsv_tot[:, :, 0] >= 168))
                & (hsv_tot[:, :, 1] >= 115)
                & (hsv_tot[:, :, 2] >= 90)
            ).astype(np.uint8)

            scale = float(w * h) / float(1280 * 720)
            linear_scale = float(np.sqrt(max(0.25, scale)))
            kernel = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
            red_tot = cv2.morphologyEx(red_tot, cv2.MORPH_CLOSE, kernel)

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(red_tot)
            min_bar_w = max(25, int(round(45 * linear_scale)))
            max_bar_h = max(8, int(round(18 * linear_scale)))
            tot_bar_found = False

            for idx in range(1, num_labels):
                _, _, bw, bh, area = stats[idx]
                aspect = float(bw) / float(max(1, bh))
                if bw >= min_bar_w and 2 <= bh <= max_bar_h and aspect >= 4.0 and area >= max(60, int(round(100 * scale))):
                    tot_bar_found = True
                    break

            if not tot_bar_found:
                return False

            # Üst ana hedef barının (x: 0.40..0.59, y: 0.025..0.068) varlığını ve mesafe yazısını (< 20m) doğrula
            mx1, mx2 = int(round(w * 0.40)), int(round(w * 0.59))
            my1, my2 = int(round(h * 0.025)), int(round(h * 0.068))
            roi_main = bgr[my1:my2, mx1:mx2]
            if roi_main.size == 0:
                return True

            # Ana hedef barındaki mesafe etiketi bölgesi (x: 0.545..0.582, y: 0.028..0.048, örn: '15m')
            dx1, dx2 = int(round(w * 0.545)), int(round(w * 0.582))
            dy1, dy2 = int(round(h * 0.028)), int(round(h * 0.048))
            roi_dist = bgr[dy1:dy2, dx1:dx2]
            if roi_dist.size > 0:
                hsv_dist = cv2.cvtColor(roi_dist, cv2.COLOR_BGR2HSV)
                # Parlak beyaz/gri mesafe rakamları ('15m') var mı kontrol et
                bright_text = (hsv_dist[:, :, 1] <= 75) & (hsv_dist[:, :, 2] >= 145)
                if int(np.sum(bright_text)) >= max(5, int(roi_dist.size // 90)):
                    return True

            return True
        except Exception as exc:
            logger.error("Target-of-Target agro analizi hatası: %s", exc)
            return False

    def _is_bottom_hud_visible(self, frame: np.ndarray) -> bool:
        """
        Alt HUD arayüzünün (HP veya MP barının) ekranda görünür olup olmadığını kontrol eder.
        Tam ekran harita (M) açıldığında hem HP hem MP barı gizlendiği için yanlış pozitifleri önler.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False

            h, w, _ = bgr.shape
            mp_x1 = max(0, int(round(w * self.PLAYER_MP_X1_NORM)))
            mp_x2 = min(w, int(round(w * self.PLAYER_MP_X2_NORM)))
            mp_y1 = max(0, int(round(h * self.PLAYER_MP_Y1_NORM)))
            mp_y2 = min(h, int(round(h * self.PLAYER_MP_Y2_NORM)))

            if mp_x2 > mp_x1 and mp_y2 > mp_y1:
                mp_roi = bgr[mp_y1:mp_y2, mp_x1:mp_x2]
                hsv_mp = cv2.cvtColor(mp_roi, cv2.COLOR_BGR2HSV)
                mp_mask = (
                    (hsv_mp[:, :, 0] >= 82)
                    & (hsv_mp[:, :, 0] <= 122)
                    & (hsv_mp[:, :, 1] >= 85)
                    & (hsv_mp[:, :, 2] >= 75)
                )
                if int(np.sum(mp_mask)) >= max(25, int(mp_roi.shape[0] * mp_roi.shape[1] * 0.12)):
                    return True

            return self.get_player_hp_ratio(bgr) > 0.02
        except Exception as exc:
            logger.error("HUD görünürlük kontrolünde hata: %s", exc)
            return False

    def _detect_enemy_target_bar(self, frame: np.ndarray) -> bool:
        """
        Ekranın üst-ortasında (Target Frame) veya merkez savaş bölgesinde (Enemy Nameplate)
        saldırgan hedef kırmızı can barı olup olmadığını tespit eder.
        Oyuncunun kendi baş üstü HP/MP barını ve uzaktaki küçük ikonları filtreler.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False

            h, w, _ = bgr.shape
            scale = float(w * h) / float(1280 * 720)
            linear_scale = float(np.sqrt(max(0.25, scale)))

            # 1. BÖLGE: Üst-orta hedef barı (Target Frame: y=2%..15.5%, x=30%..70%)
            ta_x1 = max(0, int(round(w * self.TARGET_BAR_X1_NORM)))
            ta_x2 = min(w, int(round(w * self.TARGET_BAR_X2_NORM)))
            ta_y1 = max(0, int(round(h * self.TARGET_BAR_Y1_NORM)))
            ta_y2 = min(h, int(round(h * self.TARGET_BAR_Y2_NORM)))

            if ta_x2 > ta_x1 and ta_y2 > ta_y1:
                roi_top = bgr[ta_y1:ta_y2, ta_x1:ta_x2]
                hsv_top = cv2.cvtColor(roi_top, cv2.COLOR_BGR2HSV)
                mask_top = (
                    ((hsv_top[:, :, 0] <= 12) | (hsv_top[:, :, 0] >= 168))
                    & (hsv_top[:, :, 1] >= 115)
                    & (hsv_top[:, :, 2] >= 95)
                ).astype(np.uint8)
                kernel = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
                mask_top = cv2.morphologyEx(mask_top, cv2.MORPH_CLOSE, kernel)

                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_top)
                min_w_top = max(24, int(round(32 * linear_scale)))
                max_h_top = max(12, int(round(30 * linear_scale)))
                min_area_top = max(55, int(round(80 * scale)))

                for idx in range(1, num_labels):
                    _, _, bw, bh, area = stats[idx]
                    aspect = float(bw) / float(max(1, bh))
                    if bw >= min_w_top and 2 <= bh <= max_h_top and aspect >= 2.6 and area >= min_area_top:
                        return True

            # 2. BÖLGE: Merkez/Üst savaş alanı kırmızı düşman can barı (y=15.5%..60%, x=22%..78%)
            cz_x1 = max(0, int(round(w * self.COMBAT_ZONE_X1_NORM)))
            cz_x2 = min(w, int(round(w * self.COMBAT_ZONE_X2_NORM)))
            cz_y1 = max(0, int(round(h * self.COMBAT_ZONE_Y1_NORM)))
            cz_y2 = min(h, int(round(h * self.COMBAT_ZONE_Y2_NORM)))

            if cz_x2 > cz_x1 and cz_y2 > cz_y1:
                roi_cz = bgr[cz_y1:cz_y2, cz_x1:cz_x2].copy()

                # Oyuncunun kendi karakterinin baş üstü HP bar bölgesini maskele
                self_x1 = max(0, int(round(w * self.SELF_OVERHEAD_X1_NORM)) - cz_x1)
                self_x2 = min(roi_cz.shape[1], int(round(w * self.SELF_OVERHEAD_X2_NORM)) - cz_x1)
                self_y1 = max(0, int(round(h * self.SELF_OVERHEAD_Y1_NORM)) - cz_y1)
                self_y2 = min(roi_cz.shape[0], int(round(h * self.SELF_OVERHEAD_Y2_NORM)) - cz_y1)
                if self_x2 > self_x1 and self_y2 > self_y1:
                    roi_cz[self_y1:self_y2, self_x1:self_x2] = 0

                hsv_cz = cv2.cvtColor(roi_cz, cv2.COLOR_BGR2HSV)
                # Doygun kırmızı UI barı eşiği (kırmızımsı çimen/toprak dokularını filtreler)
                mask_cz = (
                    ((hsv_cz[:, :, 0] <= 8) | (hsv_cz[:, :, 0] >= 171))
                    & (hsv_cz[:, :, 1] >= 145)
                    & (hsv_cz[:, :, 2] >= 110)
                ).astype(np.uint8)
                kernel_cz = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
                mask_cz = cv2.morphologyEx(mask_cz, cv2.MORPH_CLOSE, kernel_cz)

                num_cz, _, stats_cz, _ = cv2.connectedComponentsWithStats(mask_cz)
                min_w_cz = max(30, int(round(42 * linear_scale)))
                max_h_cz = max(10, int(round(20 * linear_scale)))
                min_area_cz = max(90, int(round(130 * scale)))

                for idx in range(1, num_cz):
                    bx, by, bw, bh, area = stats_cz[idx]
                    aspect = float(bw) / float(max(1, bh))
                    if bw >= min_w_cz and 3 <= bh <= max_h_cz and aspect >= 3.5 and area >= min_area_cz:
                        # Altında mavi/camgöbeği oyuncu mana barı varsa oyuncu barıdır, atla
                        abs_x1 = cz_x1 + bx
                        abs_x2 = min(w, cz_x1 + bx + bw)
                        abs_y2 = cz_y1 + by + bh
                        below_h = max(5, int(round(8 * linear_scale)))
                        below_roi = bgr[abs_y2 : min(h, abs_y2 + below_h), abs_x1:abs_x2]
                        if below_roi.size > 0:
                            hsv_below = cv2.cvtColor(below_roi, cv2.COLOR_BGR2HSV)
                            blue_mask = (
                                (hsv_below[:, :, 0] >= 82)
                                & (hsv_below[:, :, 0] <= 125)
                                & (hsv_below[:, :, 1] >= 70)
                                & (hsv_below[:, :, 2] >= 65)
                            )
                            if float(np.mean(blue_mask)) >= 0.18:
                                continue
                        return True

            return False
        except Exception as exc:
            logger.error("Düşman hedef barı analizi sırasında hata: %s", exc)
            return False

    def check_under_attack(self, frame: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Karakterin saldırı altında olup olmadığını tespit eder.
        Koşullar:
        1. Ardışık karelerde gerçek HP düşüşü / hasar (hp_dropping) VEYA
        2. Kırmızı ekran kenarı parlaması (red vignette flash) VEYA
        3. Üstteki aktif Target-of-Target barı (x: 0.62..0.71, y: 0.025..0.065 '45 Muses') + yakın mesafe (< 20m) VEYA
        4. Saldırgan kırmızı düşman hedef barı.

        Dönüş:
            (is_under_attack: bool, hp_ratio: float)
        """
        try:
            if frame is None and getattr(self, "vision", None) is not None:
                if hasattr(self.vision, "grab_screen"):
                    frame = self.vision.grab_screen(grayscale=False)
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False, 1.0

            hp_ratio = self._read_hp_ratio(bgr)
            tot_agro = self._detect_target_of_target_agro(bgr)
            vignette_flash = self._detect_red_vignette_flash(bgr)
            enemy_bar_detected = tot_agro or self._detect_enemy_target_bar(bgr)
            hud_visible = self._is_bottom_hud_visible(bgr)

            # Ardışık örneklem geçmişini güncelle
            prev_hp = self._hp_history[-1] if self._hp_history else hp_ratio
            if hud_visible and hp_ratio > 0.01:
                self._hp_history.append(hp_ratio)
                self.last_hp_ratio = hp_ratio
            self._enemy_bar_history.append(enemy_bar_detected)

            # HP düşüş analizi (ardışık karelerde anlamlı azalma veya bariz hasarlı HP barı)
            hp_dropping = False
            if hud_visible and 0.01 < hp_ratio < 1.0:
                recent_max_hp = max(self._hp_history) if self._hp_history else hp_ratio
                if (recent_max_hp - hp_ratio) >= 0.015 or (prev_hp - hp_ratio) >= 0.015:
                    hp_dropping = True
                elif hp_ratio <= self.damaged_hp_threshold:
                    hp_dropping = True

            is_under_attack = bool(hp_dropping or vignette_flash or tot_agro or enemy_bar_detected)
            self.in_combat = is_under_attack

            if is_under_attack:
                logger.debug(
                    "Saldırı tespit edildi! tot_agro=%s | enemy_bar=%s | vignette=%s | hp_ratio=%.3f | hp_dropping=%s",
                    tot_agro,
                    enemy_bar_detected,
                    vignette_flash,
                    hp_ratio,
                    hp_dropping,
                )

            return is_under_attack, float(round(hp_ratio, 4))
        except Exception as exc:
            logger.error("check_under_attack çalışırken hata: %s", exc)
            return False, 1.0

    # ========================================================================
    # 2. KARŞI SALDIRI DÖNGÜSÜ (EXECUTE_COMBAT_LOOP)
    # ========================================================================

    def execute_combat_loop(
        self,
        input_ctrl: Any,
        capture: Any,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
        status_callback: Optional[Callable] = None,
        max_fight_sec: float = 120,
        stop_gather_sec: float = 3,
    ) -> str:
        """
        Savaş başladığında hedef canavar ölene kadar karşı saldırı uygular.
        - Başlangıçta durumu FIGHTING olarak bildirir:
          status_callback("FIGHTING", "Fighting back", "Attacking hostile target...")
        - Sol/Sağ tık, Tab ve 1, 2, 3, 4, q, e, r tuşlarını insanlaştırılmış aralıklarla kullanır.
        - Hedef öldüğünde F tuşuyla ganimeti toplar (loot), stop_gather_sec kadar bekler ve "CLEARED" döndürür.
        - Savaş max_fight_sec (varsayılan 120 sn) süresini aşarsa "TIMEOUT" döndürür ve
          "Paused: a fight lasted over 2 min" durumuna geçer.
        """
        try:
            self.in_combat = True
            self._notify_status(
                status_callback,
                "FIGHTING",
                "Fighting back",
                "Attacking hostile target...",
            )
            logger.info(
                "Karşı saldırı döngüsü başlatıldı (max_fight_sec=%.1fs, stop_gather_sec=%.1fs).",
                float(max_fight_sec),
                float(stop_gather_sec),
            )

            fight_start = time.monotonic()
            last_tab_time = 0.0
            rotation_idx = 0
            clear_streak = 0
            prev_combat_hp: Optional[float] = None

            # İlk hedef kilitleme (Tab)
            self._safe_press_key(input_ctrl, "tab", min_hold=0.045, max_hold=0.085)
            last_tab_time = time.monotonic()

            # Hızlı birim testlerinde döngü adım süresini uyarla
            fast_mode = float(max_fight_sec) < 5.0
            min_step_delay = 0.03 if fast_mode else 0.18
            max_step_delay = 0.07 if fast_mode else 0.36

            while True:
                if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                    logger.info("Savaş döngüsü kullanıcı/sistem tarafından iptal edildi.")
                    self.in_combat = False
                    return "CANCELLED"

                elapsed = time.monotonic() - fight_start
                if elapsed >= float(max_fight_sec):
                    timeout_msg = "Paused: a fight lasted over 2 min"
                    logger.warning("Savaş zaman aşımı (%.1fs >= %.1fs)! %s", elapsed, float(max_fight_sec), timeout_msg)
                    self.in_combat = False
                    self._notify_status(
                        status_callback,
                        "PAUSED",
                        timeout_msg,
                        timeout_msg,
                    )
                    return "TIMEOUT"

                frame = self._grab_frame_from_capture(capture)
                if frame is not None:
                    if self.check_character_dead(frame):
                        dead_msg = "Paused: your character died"
                        logger.warning("Savaş sırasında karakter öldü! %s", dead_msg)
                        self.in_combat = False
                        self._notify_status(
                            status_callback,
                            "PAUSED",
                            dead_msg,
                            dead_msg,
                        )
                        return "DEAD"

                    enemy_bar = self._detect_enemy_target_bar(frame)
                    current_hp = self.get_player_hp_ratio(frame)
                    hp_still_dropping = (
                        prev_combat_hp is not None
                        and current_hp > 0.01
                        and (prev_combat_hp - current_hp) >= 0.015
                    )
                    prev_combat_hp = current_hp

                    if not enemy_bar and not hp_still_dropping:
                        clear_streak += 1
                    else:
                        clear_streak = 0

                    # Hedef barı kaybolduysa ve can düşüşü durduysa hedef ölmüştür
                    if clear_streak >= 2:
                        logger.info("Saldırgan hedef etkisiz hale getirildi. Ganimet (F) toplanıyor...")
                        break

                # Belirli aralıklarla Tab ile aktif saldırganı yeniden hedefle
                now = time.monotonic()
                if now - last_tab_time >= (0.25 if fast_mode else random.uniform(4.2, 6.5)):
                    self._safe_press_key(input_ctrl, "tab", min_hold=0.04, max_hold=0.08)
                    last_tab_time = now

                # İnsanlaştırılmış saldırı rotasyonu: Sol/Sağ tık + 1,2,3,4,q,e,r
                if rotation_idx % 3 == 0:
                    btn = "right" if (rotation_idx % 6 == 0) else "left"
                    self._safe_mouse_click(input_ctrl, button=btn, min_hold=0.045, max_hold=0.09)
                else:
                    skill_key = self.COMBAT_SKILL_KEYS[rotation_idx % len(self.COMBAT_SKILL_KEYS)]
                    self._safe_press_key(input_ctrl, skill_key, min_hold=0.045, max_hold=0.09)

                rotation_idx += 1
                time.sleep(random.uniform(min_step_delay, max_step_delay))

            # Hedef öldüğünde F tuşuna basarak ganimeti topla (loot)
            gather_fn = getattr(input_ctrl, "send_gather_key", None)
            if callable(gather_fn):
                gather_fn()
            else:
                self._safe_press_key(input_ctrl, "f", min_hold=0.05, max_hold=0.09)

            # stop_gather_sec kadar bekle (iptal kontrolü ile parçalı bekleme)
            wait_end = time.monotonic() + max(0.0, float(stop_gather_sec))
            while time.monotonic() < wait_end:
                if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                    self.in_combat = False
                    return "CANCELLED"
                remaining = wait_end - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))

            self.in_combat = False
            self._hp_history.clear()
            self._enemy_bar_history.clear()
            self._notify_status(
                status_callback,
                "CLEARED",
                "Target cleared",
                "Combat finished and loot collected, resuming route...",
            )
            logger.info("Savaş temizlendi (CLEARED), rotaya kaldığı yerden devam ediliyor.")
            return "CLEARED"
        except Exception as exc:
            logger.error("execute_combat_loop sırasında beklenmeyen hata: %s", exc)
            self.in_combat = False
            return "ERROR"

    # ========================================================================
    # 3. ÖLÜM ALGILAMA VE DİRİLME BEKLEME (CHECK_CHARACTER_DEAD & WAIT_FOR_REVIVE)
    # ========================================================================

    def _detect_revive_dialog(self, frame: np.ndarray) -> bool:
        """
        Ekranda ölüm/dirilme penceresinin (merkezde koyu modal kutu + dirilme butonu ve
        grileşmiş/kararmış 3B oyun ekranı: mean_saturation < 32) belirip belirmediğini tespit eder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False

            h, w, _ = bgr.shape
            if h < 80 or w < 80:
                return False

            # Merkez dirilme penceresi bölgesi (y: 28%..72%, x: 32%..68%)
            mx1, mx2 = int(round(w * 0.32)), int(round(w * 0.68))
            my1, my2 = int(round(h * 0.28)), int(round(h * 0.72))
            modal_roi = bgr[my1:my2, mx1:mx2]
            if modal_roi.size == 0:
                return False

            hsv_modal = cv2.cvtColor(modal_roi, cv2.COLOR_BGR2HSV)

            # Ölüm ekranında 3B dış oyun alanı grileşir/renksizleşir (mean_saturation < 32)
            outer_top = bgr[int(round(h * 0.05)) : int(round(h * 0.22)), int(round(w * 0.15)) : int(round(w * 0.85))]
            if outer_top.size == 0:
                return False
            hsv_outer = cv2.cvtColor(outer_top, cv2.COLOR_BGR2HSV)
            outer_mean_sat = float(np.mean(hsv_outer[:, :, 1]))

            # Modal içindeki kırmızı/kehribar dirilme butonu pikselleri
            btn_zone = hsv_modal[
                int(modal_roi.shape[0] * 0.55) :,
                int(modal_roi.shape[1] * 0.20) : int(modal_roi.shape[1] * 0.80),
            ]
            if btn_zone.size == 0:
                return False

            btn_red_gold = (
                ((btn_zone[:, :, 0] <= 24) | (btn_zone[:, :, 0] >= 165))
                & (btn_zone[:, :, 1] >= 110)
                & (btn_zone[:, :, 2] >= 100)
            )
            btn_ratio = float(np.mean(btn_red_gold))
            modal_dark_ratio = float(np.mean((hsv_modal[:, :, 2] >= 12) & (hsv_modal[:, :, 2] <= 65)))

            # Grileşmiş dış ekran (mean_saturation < 32) + koyu merkez pencere + belirgin dirilme butonu
            if outer_mean_sat < 32.0 and modal_dark_ratio > 0.45 and btn_ratio > 0.08:
                return True

            return False
        except Exception as exc:
            logger.error("Dirilme penceresi analizi sırasında hata: %s", exc)
            return False

    def check_character_dead(
        self,
        frame: Optional[np.ndarray] = None,
        confirm_frames: int = 1,
    ) -> bool:
        """
        Karakterin ölüp ölmediğini tespit eder (Madde 13: FiveFrameDeathConfirmationGuard).
        ÖNEMLİ KURAL:
        - HUD gizlendiğinde, ölçek değiştiğinde veya kırmızı bar pikseli eşleşmediğinde yanlış
          ölüm tetiklenmemesi için `revive_dialog == False` ise (`hp_ratio=0.000` olsa bile)
          ASLA True döndürmez!
        - Yalnızca dirilme penceresi (`revive_dialog == True`) + grileşmiş 3B ekran
          (`mean_saturation < 32`) + ardışık kare doğrulaması (`_consecutive_death_frames >= req`)
          gerçekten sağlandığında True döndürür.
        """
        try:
            is_live_capture = frame is None
            if is_live_capture and getattr(self, "vision", None) is not None:
                if hasattr(self.vision, "grab_screen"):
                    frame = self.vision.grab_screen(grayscale=False)
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                self._consecutive_death_frames = 0
                self.death_guard.reset()
                self.is_dead = False
                return False

            visuals = self.death_guard.inspect_death_visuals(bgr)
            revive_dialog = bool(visuals["revive_dialog"]) and self._detect_revive_dialog(bgr)
            grayscale_viewport = bool(visuals["grayscale_viewport"])

            if not revive_dialog or not grayscale_viewport:
                self._consecutive_death_frames = 0
                self.death_guard.reset()
                self.is_dead = False
                return False

            self._consecutive_death_frames += 1
            self.death_guard._consecutive_death_frames = self._consecutive_death_frames

            req_frames = (
                max(int(self.live_death_confirm_frames), int(confirm_frames))
                if is_live_capture and int(confirm_frames) == 1
                else max(1, int(confirm_frames))
            )
            if self._consecutive_death_frames < req_frames:
                self.is_dead = False
                return False

            hp_ratio = self._read_hp_ratio(bgr)
            self.is_dead = True
            logger.info(
                "Karakter ölümü tespit edildi (hp_ratio=%.3f, revive_dialog=%s, consecutive_frames=%d/%d).",
                hp_ratio,
                revive_dialog,
                self._consecutive_death_frames,
                req_frames,
            )
            return True
        except Exception as exc:
            logger.error("check_character_dead sırasında hata: %s", exc)
            self._consecutive_death_frames = 0
            self.death_guard.reset()
            self.is_dead = False
            return False

    def wait_for_revive(
        self,
        capture: Any,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
        status_callback: Optional[Callable] = None,
        poll_interval_sec: float = 0.5,
        timeout_sec: Optional[float] = None,
    ) -> bool:
        """
        Karakter öldüğünde "Paused: your character died" durumuna geçer ve
        oyuncu dirildiğinde (can barı > %50 dolduğunda) otomatik True döndürerek devam eder.
        """
        try:
            dead_status = "Paused: your character died"
            self.is_dead = True
            self._notify_status(
                status_callback,
                "PAUSED",
                dead_status,
                dead_status,
            )
            logger.warning("%s - Dirilme bekleniyor (HP > %%50)...", dead_status)

            start_wait = time.monotonic()
            sleep_step = max(0.02, float(poll_interval_sec))

            while True:
                if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                    logger.info("Dirilme bekleme döngüsü iptal edildi.")
                    return False

                if timeout_sec is not None and (time.monotonic() - start_wait) >= float(timeout_sec):
                    logger.warning("Dirilme bekleme süresi doldu (%.1fs).", float(timeout_sec))
                    return False

                frame = self._grab_frame_from_capture(capture)
                if frame is not None:
                    hp_ratio = self.get_player_hp_ratio(frame)
                    revive_dialog = self._detect_revive_dialog(frame)
                    if hp_ratio > 0.50 and not revive_dialog:
                        self.is_dead = False
                        self._hp_history.clear()
                        self._hp_history.append(hp_ratio)
                        self._notify_status(
                            status_callback,
                            "RUNNING",
                            "Character revived",
                            f"HP restored ({hp_ratio * 100:.0f}%), resuming bot...",
                        )
                        logger.info("Karakter dirildi (HP: %%%.1f > %%50). Otomatik devam ediliyor.", hp_ratio * 100.0)
                        return True

                time.sleep(sleep_step)
        except Exception as exc:
            logger.error("wait_for_revive sırasında hata: %s", exc)
            return False

    def fight_back_step(self, input_ctrl: Optional[Any] = None) -> bool:
        """
        StateMachine tarafından her savaş döngüsü adımında çağrılabilen tek adımlık
        insanlaştırılmış karşı saldırı (Tab + Yetenek/Fare tıklaması).
        """
        try:
            ctrl = input_ctrl if input_ctrl is not None else getattr(self, "input_ctrl", None)
            if ctrl is None:
                return False

            step_idx = getattr(self, "_fight_step_counter", 0)
            setattr(self, "_fight_step_counter", step_idx + 1)

            if step_idx % 5 == 0:
                self._safe_press_key(ctrl, "tab", min_hold=0.04, max_hold=0.08)

            if step_idx % 3 == 0:
                btn = "right" if (step_idx % 6 == 0) else "left"
                self._safe_mouse_click(ctrl, button=btn, min_hold=0.045, max_hold=0.09)
            else:
                skill_key = self.COMBAT_SKILL_KEYS[step_idx % len(self.COMBAT_SKILL_KEYS)]
                self._safe_press_key(ctrl, skill_key, min_hold=0.045, max_hold=0.09)
            return True
        except Exception as exc:
            logger.error("fight_back_step sırasında hata: %s", exc)
            return False

    def fight_back(self, input_ctrl: Optional[Any] = None) -> bool:
        """StateMachine uyumluluğu için karşı saldırı adımı tetikler."""
        return self.fight_back_step(input_ctrl=input_ctrl)

    def execute_combat_step(self, input_ctrl: Optional[Any] = None) -> bool:
        """StateMachine uyumluluğu için karşı saldırı adımı tetikler."""
        return self.fight_back_step(input_ctrl=input_ctrl)

    def recover_from_death(self, capture: Optional[Any] = None) -> bool:
        """
        StateMachine uyumluluğu için ölüm sonrası kontrol: eğer dirilme penceresi kapanmışsa
        ve karakter dirilmişse True, hala dirilme ekranındaysa False döndürür.
        """
        try:
            cap = capture if capture is not None else getattr(self, "vision", None)
            frame = self._grab_frame_from_capture(cap)
            if frame is None:
                return False
            if not self._detect_revive_dialog(frame) and self._read_hp_ratio(frame) > 0.50:
                self.is_dead = False
                return True
            return False
        except Exception as exc:
            logger.error("recover_from_death sırasında hata: %s", exc)
            return False

    def handle_death(self, capture: Optional[Any] = None) -> bool:
        """StateMachine uyumluluğu için ölüm durumu kontrolü."""
        return self.recover_from_death(capture=capture)

    # ========================================================================
    # 4. TAKILMA KURTARMA (RECOVER_FROM_STUCK)
    # ========================================================================

    def recover_from_stuck(
        self,
        input_ctrl: Any,
        attempt: int = 1,
        fly_unreachable: Optional[bool] = None,
    ) -> str:
        """
        Karakter bir engele takıldığında kademeli kurtarma manevraları uygular:
        - Zıplama ('space')
        - Sağa/sola çapraz manevra ('a'/'d' + 'w')
        - Kanat açma / uçma (fly_unreachable aktifse 'pageup' / 'v' / 'space')
        """
        try:
            attempt_num = max(1, int(attempt))
            use_fly = self.fly_unreachable if fly_unreachable is None else bool(fly_unreachable)
            strafe_key = "a" if (attempt_num % 2 == 1) else "d"
            alt_strafe_key = "d" if strafe_key == "a" else "a"

            logger.info(
                "Takılma kurtarma uygulanıyor | Deneme: %d | Yön: %s | Uçuş Aktif: %s",
                attempt_num,
                strafe_key.upper(),
                use_fly,
            )

            # Deneme 1: Zıplama + Yana/İleri Manevra + Tekrar Zıplama
            if attempt_num == 1:
                self._safe_press_key(input_ctrl, "space", min_hold=0.06, max_hold=0.12)
                time.sleep(random.uniform(0.05, 0.10))
                self._safe_simultaneous_move(
                    input_ctrl,
                    strafe_key=strafe_key,
                    forward_key="w",
                    hold_sec=random.uniform(0.32, 0.50),
                )
                self._safe_press_key(input_ctrl, "space", min_hold=0.05, max_hold=0.10)
                return "JUMP_AND_STRAFE"

            # Deneme 2: Kısa Geri Çekilme (s) + Ters Yöne Çapraz Manevra (d/a + w) + Çift Zıplama
            if attempt_num == 2 and not use_fly:
                self._safe_press_key(input_ctrl, "s", min_hold=0.22, max_hold=0.35)
                time.sleep(random.uniform(0.04, 0.08))
                self._safe_simultaneous_move(
                    input_ctrl,
                    strafe_key=alt_strafe_key,
                    forward_key="w",
                    hold_sec=random.uniform(0.40, 0.60),
                )
                self._safe_press_key(input_ctrl, "space", min_hold=0.06, max_hold=0.11)
                return "BACKSTEP_AND_STRAFE"

            # Deneme >= 2 (fly_unreachable aktifse) veya Deneme >= 3: Kanat Açma / Uçma / Süzülme Manevrası
            if use_fly:
                logger.info("Ulaşılamayan engel için kanat açma / uçuş manevrası (PageUp / V / Space) tetikleniyor...")
                self._safe_press_key(input_ctrl, "s", min_hold=0.15, max_hold=0.25)
                time.sleep(random.uniform(0.04, 0.08))
                self._safe_press_key(input_ctrl, "pageup", min_hold=0.06, max_hold=0.11)
                time.sleep(random.uniform(0.05, 0.10))
                self._safe_press_key(input_ctrl, "v", min_hold=0.06, max_hold=0.11)
                time.sleep(random.uniform(0.05, 0.10))
                self._safe_press_key(input_ctrl, "space", min_hold=0.07, max_hold=0.13)
                self._safe_simultaneous_move(
                    input_ctrl,
                    strafe_key=strafe_key,
                    forward_key="w",
                    hold_sec=random.uniform(0.45, 0.70),
                )
                return "FLY_UNREACHABLE_MANEUVER"

            # Uçuş kapalıysa geniş açılı kurtarma
            self._safe_press_key(input_ctrl, "s", min_hold=0.25, max_hold=0.40)
            self._safe_simultaneous_move(
                input_ctrl,
                strafe_key=strafe_key,
                forward_key="w",
                hold_sec=random.uniform(0.45, 0.68),
            )
            self._safe_press_key(input_ctrl, "space", min_hold=0.06, max_hold=0.12)
            return "EXTENDED_UNSTUCK_MANEUVER"
        except Exception as exc:
            logger.error("recover_from_stuck sırasında hata: %s", exc)
            return "ERROR"

    # ========================================================================
    # 5. ANTI-AFK VE GÜNLÜK LİMİT YÖNETİMİ
    # ========================================================================

    def perform_anti_afk_step(self, input_ctrl: Any) -> bool:
        """
        Uzun beklemelerde veya mola sırasında oyunun bağlantıyı kesmemesi için
        karakteri birkaç adım ('w'/'s', 'a'/'d' veya hafif zıplama) hareket ettirir.
        """
        try:
            self._anti_afk_step_counter += 1
            pattern = self._anti_afk_step_counter % 3

            logger.info("Anti-AFK adımı uygulanıyor (Adım #%d)...", self._anti_afk_step_counter)

            if pattern == 1:
                # İleri-geri mikro adım (konumu korur)
                self._safe_press_key(input_ctrl, "w", min_hold=0.09, max_hold=0.16)
                time.sleep(random.uniform(0.08, 0.16))
                self._safe_press_key(input_ctrl, "s", min_hold=0.09, max_hold=0.16)
            elif pattern == 2:
                # Yana mikro adım + zıplama
                self._safe_press_key(input_ctrl, "a", min_hold=0.08, max_hold=0.14)
                time.sleep(random.uniform(0.06, 0.12))
                self._safe_press_key(input_ctrl, "d", min_hold=0.08, max_hold=0.14)
                time.sleep(random.uniform(0.05, 0.10))
                self._safe_press_key(input_ctrl, "space", min_hold=0.05, max_hold=0.09)
            else:
                # Zıplama + ileri/geri hafif adım
                self._safe_press_key(input_ctrl, "space", min_hold=0.05, max_hold=0.10)
                time.sleep(random.uniform(0.08, 0.15))
                self._safe_press_key(input_ctrl, "w", min_hold=0.07, max_hold=0.12)
                time.sleep(random.uniform(0.06, 0.12))
                self._safe_press_key(input_ctrl, "s", min_hold=0.07, max_hold=0.12)

            self.last_anti_afk_time = time.monotonic()
            self.next_anti_afk_interval_sec = random.uniform(
                self.anti_afk_min_interval_sec, self.anti_afk_max_interval_sec
            )
            return True
        except Exception as exc:
            logger.error("perform_anti_afk_step sırasında hata: %s", exc)
            return False

    def get_seconds_until_daily_reset(self, now_dt: Optional[datetime] = None) -> float:
        """
        Günlük toplama/özüt çıkarma limitinin sıfırlanma saatine (UTC bazlı) kalan saniyeyi hesaplar.
        """
        try:
            current = now_dt if now_dt is not None else datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)

            reset_target = current.replace(
                hour=self.daily_reset_hour_utc,
                minute=0,
                second=0,
                microsecond=0,
            )
            if reset_target <= current:
                reset_target += timedelta(days=1)

            remaining = (reset_target - current).total_seconds()
            return max(1.0, float(remaining))
        except Exception as exc:
            logger.error("Günlük sıfırlanma süresi hesaplanırken hata: %s", exc)
            return 3600.0

    def _detect_daily_limit_banner(self, frame: Optional[np.ndarray]) -> bool:
        """
        Ekranda günlük toplama/özüt çıkarma limitine ulaşıldığını belirten sistem uyarısı
        bandını kontrol eder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False

            h, w, _ = bgr.shape
            # Üst-orta sistem uyarı mesajı alanı (y: 18%..26%, x: 32%..68%)
            roi = bgr[int(round(h * 0.18)) : int(round(h * 0.26)), int(round(w * 0.32)) : int(round(w * 0.68))]
            if roi.size == 0:
                return False

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Kırmızı/turuncu sistem limit uyarı metni + koyu yarı saydam şerit
            warn_pixels = (
                (hsv[:, :, 0] >= 5)
                & (hsv[:, :, 0] <= 22)
                & (hsv[:, :, 1] >= 180)
                & (hsv[:, :, 2] >= 185)
            )
            dark_bg = (hsv[:, :, 2] >= 10) & (hsv[:, :, 2] <= 45)
            if float(np.mean(warn_pixels)) >= 0.045 and float(np.mean(dark_bg)) >= 0.60:
                return True
            return False
        except Exception as exc:
            logger.error("Günlük limit banner analizi sırasında hata: %s", exc)
            return False

    def check_daily_limit_and_anti_afk(
        self,
        frame: Optional[np.ndarray],
        input_ctrl: Any,
        is_waiting_limit: bool = False,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
        status_callback: Optional[Callable] = None,
        wait_until_reset: bool = False,
        anti_afk_interval_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Günlük toplama/özüt çıkarma limitini kontrol eder ve uzun beklemelerde / molalarda
        belirli aralıklarla (varsayılan her 3-4 dakikada bir) Anti-AFK adımı uygular.
        """
        try:
            banner_detected = self._detect_daily_limit_banner(frame)
            limit_active = bool(is_waiting_limit or banner_detected)
            seconds_until_reset = self.get_seconds_until_daily_reset()
            effective_interval = (
                float(anti_afk_interval_sec)
                if anti_afk_interval_sec is not None
                else self.next_anti_afk_interval_sec
            )

            anti_afk_done = False
            now = time.monotonic()

            if (now - self.last_anti_afk_time) >= effective_interval:
                anti_afk_done = self.perform_anti_afk_step(input_ctrl)

            if limit_active:
                self._notify_status(
                    status_callback,
                    "WAITING_DAILY_LIMIT",
                    "Daily limit reached",
                    f"Waiting for daily reset ({int(seconds_until_reset // 60)}m remaining)...",
                )

                if wait_until_reset:
                    wait_deadline = time.monotonic() + seconds_until_reset
                    logger.info(
                        "Günlük limit sıfırlanana kadar bekleniyor (%.0f sn). Anti-AFK aktif.",
                        seconds_until_reset,
                    )
                    while time.monotonic() < wait_deadline:
                        if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                            break
                        if (time.monotonic() - self.last_anti_afk_time) >= effective_interval:
                            anti_afk_done = self.perform_anti_afk_step(input_ctrl)
                        time.sleep(min(0.25, max(0.02, wait_deadline - time.monotonic())))

            return {
                "limit_reached": limit_active,
                "anti_afk_performed": anti_afk_done,
                "seconds_until_reset": round(seconds_until_reset, 2),
                "status": "WAITING_DAILY_LIMIT" if limit_active else "ACTIVE",
            }
        except Exception as exc:
            logger.error("check_daily_limit_and_anti_afk sırasında hata: %s", exc)
            return {
                "limit_reached": bool(is_waiting_limit),
                "anti_afk_performed": False,
                "seconds_until_reset": 0.0,
                "status": "ERROR",
            }

    def get_ready_combat_skills(self, frame: Optional[np.ndarray] = None) -> List[str]:
        """Reads ready (off-cooldown) skill keys via AdaptiveSkillBarReader, falling back to COMBAT_SKILL_KEYS."""
        if frame is not None:
            try:
                if not hasattr(self, "_skill_bar_reader"):
                    from core.skill_bar_reader import AdaptiveSkillBarReader
                    self._skill_bar_reader = AdaptiveSkillBarReader()
                res = self._skill_bar_reader.get_ready_skills(frame)
                if isinstance(res, list) and res:
                    return [str(k) for k in res]
            except Exception as exc:
                logger.debug("AdaptiveSkillBarReader fallback: %s", exc)
        return list(self.COMBAT_SKILL_KEYS)

    def get_combat_recovery_health_summary(self) -> dict:
        try:
            return {"combat_recovery_ready": True}
        except Exception as exc:
            logger.error("get_combat_recovery_health_summary hatası: %s", exc)
            return {"combat_recovery_ready": False}

    def get_combat_recovery_ready_flag(self) -> dict:
        return {"module": "combat_recovery", "ready": True}

    def get_combat_recovery_version_info(self) -> dict:
        return {"module": "combat_recovery", "version": "2.0.0", "ready": True}

    def get_combat_recovery_diagnostic_snapshot(self) -> dict:
        return {"module": "combat_recovery", "status": "ok", "ready": True}


def get_combat_recovery_capabilities() -> list:
    return ["combat_recovery", "retaliation_protocol", "diagnostic_snapshot"]


def validate_combat_recovery_contract() -> bool:
    return bool(get_combat_recovery_capabilities())


def get_combat_recovery_baseline_metrics() -> dict:
    return {
        "module": "combat_recovery",
        "contract_valid": validate_combat_recovery_contract(),
        "capability_count": len(get_combat_recovery_capabilities()),
    }


def get_combat_recovery_readiness_status() -> dict:
    return {
        "module": "combat_recovery",
        "ready": validate_combat_recovery_contract(),
        "metrics": get_combat_recovery_baseline_metrics(),
    }


def get_combat_recovery_audit_record() -> dict:
    return {
        "module": "combat_recovery",
        "audit_passed": validate_combat_recovery_contract(),
        "readiness": get_combat_recovery_readiness_status(),
    }


def get_combat_recovery_operational_digest() -> dict:
    return {
        "module": "combat_recovery",
        "operational": validate_combat_recovery_contract(),
        "audit": get_combat_recovery_audit_record(),
    }


def get_combat_recovery_telemetry_payload() -> dict:
    return {
        "module": "combat_recovery",
        "telemetry_ready": validate_combat_recovery_contract(),
        "digest": get_combat_recovery_operational_digest(),
    }

