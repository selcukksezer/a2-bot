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
   - Hedef ölene kadar Tab, R/T tuşları (sağ tık=T, sol tık=R — fare tıklaması yok) ve 1, 2, 3, 4, Q, E, R yetenekleriyle insanlaştırılmış rotasyon
   - Canavar öldüğünde TEK F tuşu ile ganimet toplama (loot, no27/F-0419
     doğrulanmış akış: kör ikinci F yok) ve stop_gather_sec kadar yerleşme
     bekleyip CLEARED döndürme
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
import os
import random
import time
import zlib
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.stability_guards import FiveFrameDeathConfirmationGuard, get_shared_stability_guards
from version import __version__


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

    # Saldırı tuş havuzu (sağ tık=T, sol tık=R — fare tıklaması yok)
    COMBAT_SKILL_KEYS: Tuple[str, ...] = ("1", "2", "3", "4", "q", "e", "r", "t")

    # Olive-green mob detection HSV threshold bounds (AION 2 Yellow-Maned Zaif, Sparkie, etc.)
    OLIVE_MOB_HUE_MIN: int = 26
    OLIVE_MOB_HUE_MAX: int = 68
    OLIVE_MOB_SAT_MIN: int = 80
    OLIVE_MOB_VAL_MIN: int = 80

    # Fast TAB acquisition timing
    FAST_TAB_INTERVAL_SEC: float = 0.55

    # Hedef barı FIGHTING girişi doğrulama eşiği (yanlış "saldırı altında"
    # düzeltmesi): tek karelik ham bar tespiti savaş girişi sayılmaz —
    # görsel parazit / 1 karelik bar yanıp sönmesi FIGHTING'e sokmaz.
    # Barın giriş sayılması için ARDIŞIK 2 FARKLI karede ham tespit gerekir.
    # ÇIKIŞ davranışı değişmez: doğrulanmış kilit, 1 karelik kayboluşta
    # flicker grace ile sürer, 2. kayboluş karesinde düşer.
    TARGET_BAR_ENTRY_CONFIRM_FRAMES: int = 2

    # R06 — CANLI YOLA TAŞINDI (sweep-fix Grup B; eskiden yalnızca ölü
    # CombatSystem'da vardı, core/combat.py:68,1315-1343): hedef başına
    # ilk atışta 0,30 sn bekleme + hedef HP yeniden okuma. Canlı sınıfın
    # kendi durgunluk sayacı yoktu; eşik/sayaç CombatSystem'daki
    # LineOfSightTracker ile aynıdır (damage_stagnation_threshold=3,
    # hasar sayılması için en az %0,5 HP düşüşü). 2-kare kuralı
    # (TARGET_BAR_ENTRY_CONFIRM_FRAMES) aynen korunur — settle onun
    # ÜSTÜNE eklenir, hedef yokluğu eşiğini değiştirmez.
    POSTCAST_HP_SETTLE_SEC: float = 0.30
    DAMAGE_STAGNATION_THRESHOLD: int = 3
    POSTCAST_HP_DROP_EPS_PCT: float = 0.5

    # no27 (F-0419) — doğrulanmış loot akışı, cast-bar VEKİLİ eşikleri.
    # Vekil: verify_gather_cast_bar (core/ocr_distance.py:1101) — üst-orta
    # kanal barı (x 0.422..0.571, y 0.048..0.059; cyan/teal H 72..118).
    # Loot'un gerçekten AYNI barda kanal gösterip göstermediği repo'da
    # HİÇBİR yerde kanıtlanmadı (loot karesi / şablonu / yorumu yok) —
    # vekil BLOKE: PC doğrulaması bekliyor. Bu yüzden kör zaman aşımı
    # ASLA başarı sayılmaz (F-0050 ikizi) ve bar görülmezse kör ikinci
    # F basılmaz; akış "tek F, kör tekrar yok" olarak çalışır.
    # Bitiş kuralı gather emsalinden (core/state_machine.py:4428-4493):
    # bar %90+ OLDU veya bar görüldükten sonra kayboldu.
    LOOT_CAST_BAR_COMPLETE_PCT: float = 90.0
    LOOT_START_TIMEOUT_SEC: float = 1.5
    LOOT_END_TIMEOUT_SEC: float = 3.0
    LOOT_CAST_POLL_INTERVAL_SEC: float = 0.10
    LOOT_CAST_POLL_INTERVAL_FAST_SEC: float = 0.01
    # Kare bütçesi: 1,5 sn / 0,10 sn = 15 yoklama; 3,0 sn / 0,10 sn = 30.
    # Süre (monotonic) + kare sayısı İKİSİ de sınırdır — donmuş saatte
    # (birim testleri) kare sayısı, gerçek saatte süre durdurur.
    LOOT_START_MAX_POLLS: int = 15
    LOOT_END_MAX_POLLS: int = 30
    LOOT_END_MISSING_FRAMES: int = 2

    def __init__(
        self,
        fly_unreachable: Any = True,
        daily_reset_hour_utc: Any = 0,
        anti_afk_min_interval_sec: float = 180.0,
        anti_afk_max_interval_sec: float = 240.0,
        damaged_hp_threshold: float = 0.94,
        vision: Any = None,
        input_ctrl: Any = None,
        cast_bar_reader: Any = None,
    ) -> None:
        # Legacy positional compat: CombatRecovery(vision, input_ctrl) is used by
        # state_machine/main, so a non-bool first arg is treated as vision — but only
        # when it actually looks like a vision/capture object; anything else is a
        # programming error and now raises instead of silently mis-assigning (F-0418).
        if not isinstance(fly_unreachable, (bool, int, float)):
            if fly_unreachable is None or callable(fly_unreachable) or any(
                hasattr(fly_unreachable, m)
                for m in ("grab_screen", "get_latest_frame", "grab_frame", "capture_frame")
            ):
                logger.warning(
                    "CombatRecovery positional vision arg used (legacy shim); "
                    "prefer CombatRecovery(vision=..., input_ctrl=...)."
                )
                self.vision = fly_unreachable
                self.fly_unreachable = True
            else:
                raise TypeError(
                    "CombatRecovery fly_unreachable must be bool/int/float or a "
                    "vision object, got %r" % type(fly_unreachable).__name__
                )
        else:
            self.vision = vision
            self.fly_unreachable = bool(fly_unreachable)

        if not isinstance(daily_reset_hour_utc, (int, float, str)):
            if daily_reset_hour_utc is None or any(
                hasattr(daily_reset_hour_utc, m) for m in ("press_key", "send_key")
            ):
                logger.warning(
                    "CombatRecovery positional input_ctrl arg used (legacy shim); "
                    "prefer CombatRecovery(vision=..., input_ctrl=...)."
                )
                self.input_ctrl = daily_reset_hour_utc
                self.daily_reset_hour_utc = 0
            else:
                raise TypeError(
                    "CombatRecovery daily_reset_hour_utc must be int/float/str or an "
                    "input controller, got %r" % type(daily_reset_hour_utc).__name__
                )
        else:
            self.input_ctrl = input_ctrl
            self.daily_reset_hour_utc = int(daily_reset_hour_utc) % 24
        self.anti_afk_min_interval_sec: float = max(10.0, float(anti_afk_min_interval_sec))
        self.anti_afk_max_interval_sec: float = max(
            self.anti_afk_min_interval_sec, float(anti_afk_max_interval_sec)
        )
        # NOT: damaged_hp_threshold saldırı kararında KULLANILMAZ.
        # Eskiden check_under_attack'ta `hp_ratio <= 0.94` dalı her karede
        # "saldırı altında" üretiyordu (sabit %90 HP bile). HP seviyesi
        # yalnızca iksir mantığını besler; saldırı bayrağını asla.
        # Parametre, eski çağıranlarla uyumluluk için tutulur.
        self.damaged_hp_threshold: float = float(damaged_hp_threshold)

        # Durum ve geçmiş izleme
        self._hp_history: Deque[float] = deque(maxlen=10)
        self.last_hp_ratio: float = 1.0
        self.in_combat: bool = False
        # fix/v20-panic-combat: son check_under_attack() turunda Hedefin
        # Hedefi (ToT) barı oyuncuyu hedefliyor muydu — saldıran düşman
        # kanıtı. Rota FIGHTING kapısı (state_machine
        # _route_combat_signal_confirmed) barı bu işaretle doğrular;
        # HP okuyucudan bağımsız ikinci kanıt kanalıdır. check_under_attack
        # her turda yazar; tur dışı okuma bayat değerdir.
        self.last_target_of_target_agro: bool = False
        self.is_dead: bool = False
        self._consecutive_death_frames: int = 0
        self.live_death_confirm_frames: int = 5
        # F-1741: kendi FiveFrameDeathConfirmationGuard örneğini kurmak yerine
        # paylaşılan StabilityGuards singleton'ının ölüm korumasını kullan —
        # get_shared_stability_guards() üretim yolunda gerçekten çağrılır.
        # Paylaşım güvenlidir: CombatRecovery kararlarını kendi
        # _consecutive_death_frames sayacından verir; guard'ın sayacı her
        # ölüm-adayı karede ondan senkronlanır, diğer durumda resetlenir
        # (check_character_dead). required_frames varsayılanı da aynı (5).
        self.death_guard: FiveFrameDeathConfirmationGuard = (
            get_shared_stability_guards().death_guard
        )
        self.last_status_message: str = "IDLE"

        # no27 (F-0419): doğrulanmış loot akışı.
        # cast_bar_reader: verify_gather_cast_bar(frame) -> (bool, float)
        # veya read_cast_bar_progress(frame) -> float sunan okuyucu
        # (örn. GameHUDReader). None ise doğrulama YAPILAMAZ — bu
        # durumda akış "tek F, kör tekrar yok" olarak çalışır, loot
        # var sayılmaz. Üretimde henüz enjekte edilmedi (vekil BLOKE:
        # PC doğrulaması bekliyor); testler sahte okuyucu enjekte eder.
        self.cast_bar_reader: Any = cast_bar_reader
        # Son loot döngüsünün doğrulama sonucu: "none" | "verified" |
        # "not_started" | "not_finished" | "no_reader" | "cancelled".
        self.last_loot_outcome: str = "none"
        self.loot_verification_outcomes: List[str] = []

        # Fast TAB cycling and targeting flicker grace (sustains lock during combat VFX / damage numbers)
        self.fast_tab_interval_sec: float = float(self.FAST_TAB_INTERVAL_SEC)
        self.last_tab_time: float = 0.0
        self.tab_cycle_count: int = 0
        self.max_flicker_grace_ticks: int = int(os.environ.get("GATHERION_FLICKER_GRACE_TICKS", 1))
        self._flicker_grace_counter: int = 0
        self.is_flicker_grace_active: bool = False
        self._last_raw_target_detected: bool = False
        # Hedef barı giriş doğrulaması (TARGET_BAR_ENTRY_CONFIRM_FRAMES):
        # ardışık ham tespit sayacı (yalnızca FARKLI kareler sayılır —
        # aynı tick içinde check_under_attack + has_bar aynı frame_bgr'yi
        # iki kez okur; ikinci okuma yeni kare sayılmaz), son tespit
        # edilen karenin içerik parmak izi ve kilidin doğrulanmışlık durumu.
        self._consecutive_raw_target_frames: int = 0
        self._last_target_frame_fp: Optional[Tuple[Tuple[int, ...], int]] = None
        self._target_lock_confirmed: bool = False
        self._last_grace_tick_ts: float = 0.0

        # Anti-AFK zamanlayıcıları
        self.last_anti_afk_time: float = time.monotonic()
        self.next_anti_afk_interval_sec: float = random.uniform(
            self.anti_afk_min_interval_sec, self.anti_afk_max_interval_sec
        )
        self._anti_afk_step_counter: int = 0
        self._fight_step_counter: int = 0
        self._fight_skill_idx: int = 0
        self._no_target_steps: int = 0
        # R06 (canlı yol): hedef HP bazı + ardışık hasarsız atış sayacı.
        # Baz None = "henüz okuma yok": (yeniden) hedef sonrası ilk
        # okuma yalnızca baz olur, hasar hükmü vermez (S-1303 kuralı —
        # uydurma %100 baz yok). Hedef kaybında / Tab ile hedef
        # değişiminde sıfırlanır (fight_back_step).
        self._postcast_target_hp: Optional[float] = None
        self._consecutive_stagnant_casts: int = 0
        self._consecutive_no_target_frames: int = 0
        # F-0069: minimum interval between skill/attack presses in fight_back_step
        self._last_fight_back_press_ts: float = 0.0
        self.fight_back_min_press_interval_sec: float = 0.08

        logger.info(
            "CombatRecovery başlatıldı | fly_unreachable=%s | anti_afk=[%.0fs..%.0fs] | fast_tab=%.2fs | flicker_grace=%d",
            self.fly_unreachable,
            self.anti_afk_min_interval_sec,
            self.anti_afk_max_interval_sec,
            self.fast_tab_interval_sec,
            self.max_flicker_grace_ticks,
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
                try:
                    alt_click(button=button)
                except TypeError:
                    try:
                        alt_click(button)
                    except TypeError:
                        alt_click()
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
    # FAST TAB CYCLING & 1-FRAME TARGETING FLICKER GRACE
    # ========================================================================

    def reset_flicker_grace(self) -> None:
        """Resets the targeting flicker grace counters and status."""
        self._flicker_grace_counter = 0
        self._consecutive_no_target_frames = 0
        self.is_flicker_grace_active = False
        self._last_raw_target_detected = False
        self._last_grace_tick_ts = 0.0
        # Giriş doğrulaması da sıfırlanır: kilit düşer, sonraki bar
        # yeniden 2 farklı karede doğrulanmak zorunda kalır.
        self._consecutive_raw_target_frames = 0
        self._last_target_frame_fp = None
        self._target_lock_confirmed = False

    def is_targeting_flicker_grace(self) -> bool:
        """Returns True if the current target lock is sustained via 1-frame flicker grace."""
        return bool(self.is_flicker_grace_active)

    def fast_tab_cycle(
        self,
        input_ctrl: Any,
        min_interval_sec: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """
        Fast TAB target cycling for rapid target acquisition.
        Dispatches humanized hardware-compliant key press (35-65ms) with
        configurable interval (default fast_tab_interval_sec ~0.55s).
        """
        if input_ctrl is None:
            return False
        now = time.monotonic()
        interval = (
            float(min_interval_sec)
            if min_interval_sec is not None
            else self.fast_tab_interval_sec
        )
        fast_mode = getattr(input_ctrl, "test_mode", False)
        if fast_mode:
            interval = min(0.05, interval)

        if force or (now - self.last_tab_time) >= interval:
            pressed = self._safe_press_key(
                input_ctrl, "tab", min_hold=0.035, max_hold=0.065
            )
            if pressed:
                self.last_tab_time = now
                self.tab_cycle_count += 1
                return True
        return False

    @staticmethod
    def _red_mask_from_hsv(
        hsv: np.ndarray,
        s_min: int,
        v_min: int,
        h_lo: int = 12,
        h_hi: int = 168,
    ) -> np.ndarray:
        """
        Wrap-around red HSV mask via cv2.inRange: hue 0..h_lo OR h_hi..179,
        Sat >= s_min, Val >= v_min. Returns a standard OpenCV 0/255 mask.
        cv2.inRange bounds are inclusive, so this is pixel-identical to the
        old numpy boolean chain ((h <= h_lo) | (h >= h_hi)) & (s >= s_min)
        & (v >= v_min) — but a single optimized C call instead of ~5
        temporary boolean arrays.
        """
        lo = cv2.inRange(hsv, (0, s_min, v_min), (h_lo, 255, 255))
        hi = cv2.inRange(hsv, (h_hi, s_min, v_min), (179, 255, 255))
        return cv2.bitwise_or(lo, hi)

    @staticmethod
    def _hostile_bar_mask_from_hsv(
        hsv: np.ndarray,
        red_s: int = 90,
        red_v: int = 80,
        pink_s: int = 60,
        pink_v: int = 100,
    ) -> np.ndarray:
        """
        Hostile target-bar mask (red | pink/magenta | olive-green mob fill)
        via cv2.inRange. Returns a standard OpenCV 0/255 mask. Thresholds
        match the old boolean chains exactly (red h<=12/h>=168, pink
        h 140..175, olive h 26..68 / S,V >= 80).
        """
        red = CombatRecovery._red_mask_from_hsv(hsv, red_s, red_v, h_lo=12, h_hi=168)
        pink = cv2.inRange(
            hsv, (140, pink_s, pink_v), (175, 255, 255)
        )
        olive = cv2.inRange(
            hsv,
            (
                CombatRecovery.OLIVE_MOB_HUE_MIN,
                CombatRecovery.OLIVE_MOB_SAT_MIN,
                CombatRecovery.OLIVE_MOB_VAL_MIN,
            ),
            (CombatRecovery.OLIVE_MOB_HUE_MAX, 255, 255),
        )
        return cv2.bitwise_or(cv2.bitwise_or(red, pink), olive)

    @staticmethod
    def get_olive_mob_mask(bgr_roi: np.ndarray) -> np.ndarray:
        """
        Extracts binary mask for olive-green mob HP bars and nameplates
        (Hue 26..68, Sat 80..255, Val 80..255).
        Returns a 0/1 mask (callers / tests assert mask == 1 for olive).
        """
        if bgr_roi is None or not isinstance(bgr_roi, np.ndarray) or bgr_roi.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        try:
            hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
            # cv2.inRange is inclusive on both ends, identical to the old
            # >= / <= boolean chain; // 255 maps the 0/255 output to 0/1.
            mask_olive = cv2.inRange(
                hsv,
                (
                    CombatRecovery.OLIVE_MOB_HUE_MIN,
                    CombatRecovery.OLIVE_MOB_SAT_MIN,
                    CombatRecovery.OLIVE_MOB_VAL_MIN,
                ),
                (CombatRecovery.OLIVE_MOB_HUE_MAX, 255, 255),
            )
            mask_olive //= 255
            return mask_olive
        except Exception as exc:
            logger.error("Olive mob mask extraction error: %s", exc)
            return np.zeros((bgr_roi.shape[0], bgr_roi.shape[1]), dtype=np.uint8)

    def detect_olive_mob_bar(
        self,
        frame: np.ndarray,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """
        Detects horizontal olive-green mob HP bar or nameplate in the specified
        or default combat zone ROI (x: 22%..78%, y: 12%..55%).
        Filters out self-overhead player bar.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False
            h, w, _ = bgr.shape
            if roi_norm is not None:
                rx1, rx2, ry1, ry2 = roi_norm
                x1 = max(0, int(round(w * rx1)))
                x2 = min(w, int(round(w * rx2)))
                y1 = max(0, int(round(h * ry1)))
                y2 = min(h, int(round(h * ry2)))
            else:
                x1 = max(0, int(round(w * self.COMBAT_ZONE_X1_NORM)))
                x2 = min(w, int(round(w * self.COMBAT_ZONE_X2_NORM)))
                y1 = max(0, int(round(h * self.COMBAT_ZONE_Y1_NORM)))
                y2 = min(h, int(round(h * min(0.55, self.COMBAT_ZONE_Y2_NORM))))

            if x2 <= x1 or y2 <= y1:
                return False

            roi = bgr[y1:y2, x1:x2].copy()
            # Mask self overhead bar
            sx1 = max(0, int(round(w * self.SELF_OVERHEAD_X1_NORM)) - x1)
            sx2 = min(roi.shape[1], int(round(w * self.SELF_OVERHEAD_X2_NORM)) - x1)
            sy1 = max(0, int(round(h * self.SELF_OVERHEAD_Y1_NORM)) - y1)
            sy2 = min(roi.shape[0], int(round(h * self.SELF_OVERHEAD_Y2_NORM)) - y1)
            if sx2 > sx1 and sy2 > sy1:
                roi[sy1:sy2, sx1:sx2] = 0

            mask_olive = self.get_olive_mob_mask(roi)
            if mask_olive.size == 0:
                return False

            scale = float(w * h) / float(1280 * 720)
            linear_scale = float(np.sqrt(max(0.25, scale)))
            kernel = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
            mask_olive = cv2.morphologyEx(mask_olive, cv2.MORPH_CLOSE, kernel)

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_olive)
            min_w = max(26, int(round(34 * linear_scale)))
            max_h = max(12, int(round(22 * linear_scale)))
            min_area = max(55, int(round(80 * scale)))

            for idx in range(1, num_labels):
                _, _, bw, bh, area = stats[idx]
                aspect = float(bw) / float(max(1, bh))
                if bw >= min_w and 2 <= bh <= max_h and aspect >= 2.2 and area >= min_area:
                    return True
            return False
        except Exception as exc:
            logger.error("detect_olive_mob_bar error: %s", exc)
            return False

    @staticmethod
    def _frame_fingerprint(frame: Optional[np.ndarray]) -> Optional[Tuple[Tuple[int, ...], int]]:
        """
        Kare içeriğinin ucuz parmak izi: (şekil, strided örnek CRC32).

        Tam kareyi hash'lemek 1440p'de ~8 MB okuma demek — her tick
        pahalı. Bunun yerine kare 16'şar piksel atlayarak örneklenir
        (2560x1440 BGR'de 160x90x3 = ~43 KB, tam karenin ~%0,5'i) ve
        örneğin CRC32'si alınır; şekil de CRC tohumuna katılır, böylece
        farklı çözünürlükler aynı örneği verse bile çakışmaz.

        Aynı pikselleri taşıyan FARKLI dizi nesneleri (örn. _ensure_bgr
        her çağrıda yeni dizi üretse bile) aynı parmak izini verir;
        ardışık gerçek kareler (oyun animasyonu, hasar sayıları,
        partikül/gürültü) örneklenen piksellerde neredeyse her zaman
        farklılık gösterdiğinden farklı parmak izi alır. Bu, eski
        id(frame) tekilleştirmesinin üretimdeki çakışma hatasını
        düzeltir: _step_combat_mode her tick'te yeni dizi alır, eski
        dizi serbest kalınca yeni tahsis aynı adresi yeniden kullanırdı
        ve id() ardışık kareleri "aynı kare" sayıp doğrulama sayacını
        hiç ilerletmezdi.

        Geçersiz karede (None / boş / dizi değil) veya hata durumunda
        None döner — çağıran bunu "bilinmeyen kare" sayar ve kareyi
        yeni kabul eder (tekilleştirme yapılmaz, tespit akışı bozulmaz).
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return None
            sample = frame[::16, ::16]
            seed = zlib.crc32(repr(frame.shape).encode("ascii"))
            return (tuple(int(d) for d in frame.shape), zlib.crc32(sample.tobytes(), seed))
        except Exception:
            return None

    def _apply_targeting_flicker_grace(
        self,
        raw_detected: bool,
        use_flicker_grace: bool = True,
        frame: Optional[np.ndarray] = None,
        require_entry_confirmation: bool = True,
    ) -> bool:
        """
        Applies 1-frame targeting flicker grace ticks to prevent premature combat aborts,
        plus 2-frame ENTRY confirmation to prevent false combat entries.

        GİRİŞ (raw True): tek karelik ham tespit FIGHTING girişi sayılmaz.
        Bar, ARDIŞIK TARGET_BAR_ENTRY_CONFIRM_FRAMES (2) FARKLI karede ham
        tespit edilmeden kilit doğrulanmaz ve False döner. Aynı karenin
        aynı tick içinde iki kez okunması (check_under_attack + has_bar
        aynı frame_bgr'yi okur) yeni kare sayılmaz — o okumada o anki
        kilit durumu döner. "Aynı kare" kararı İÇERİK parmak iziyle
        verilir (_frame_fingerprint: 16'şar piksel strided örnek +
        CRC32 + şekil): aynı pikselleri taşıyan farklı dizi nesneleri
        aynı sayılır, içerik değişen kare farklı sayılır. Eski id()
        tekilleştirmesi üretimde çakışıyordu (serbest kalan dizinin
        adresi yeni tahsiste yeniden kullanılıyordu) ve doğrulama
        sayacını kilitliyordu — bu sınır kalktı. Kalan sınır: sahne
        tamamen donmuşsa (bayt-aynı iki kare, örn. duraklatılmış
        görüntü) ardışık kareler aynı parmak izini verir ve giriş
        1 kare gecikir; yanlış girişe yol açmaz.
        ÇIKIŞ (raw False): yalnızca DOĞRULANMIŞ kilit grace alır —
        doğrulanmamış tek karelik tespit, sonraki boş karede grace ile
        uzatılmaz. Doğrulanmış kilit kaybolursa grace davranışı eskisi
        gibidir: max_flicker_grace_ticks kare sürer, sonra düşer.

        use_flicker_grace=False: ham sonuç aynen döner (eski davranış);
        doğrulama sayacı yine de ilerler ama kilit, sayaç eşiğe ulaşmadan
        "doğrulanmış" sayılmaz.

        require_entry_confirmation=False: SAVAŞ-İÇİ TAKİP — döngü zaten
        savaşın içindedir, giriş kararı yoktur; ham True kilidi ANINDA
        doğrular (v9 davranışı: confirmed=True, last_raw=True, grace
        sıfır, True döner; doğrulama sayacı eşiğe çekilir, kare parmak
        izi güncellenir). Ham False'ta grace eskisi gibi işler.
        Yalnızca execute_combat_loop ana tespiti ve fight_back_step
        hedef kontrolü False geçirir; savaş GİRİŞİ kararları
        (check_under_attack, perception_hub, rota kapısı) True bırakır.
        """
        if raw_detected:
            if not require_entry_confirmation:
                # Savaş-içi takip: ham True kilidi anında doğrular (v9).
                self._target_lock_confirmed = True
                self._last_raw_target_detected = True
                self._flicker_grace_counter = 0
                self.is_flicker_grace_active = False
                self._consecutive_raw_target_frames = (
                    self.TARGET_BAR_ENTRY_CONFIRM_FRAMES
                )
                self._consecutive_no_target_frames = 0
                self._last_target_frame_fp = self._frame_fingerprint(frame)
                return True
            frame_fp = self._frame_fingerprint(frame)
            if (
                frame_fp is not None
                and self._last_target_frame_fp is not None
                and frame_fp == self._last_target_frame_fp
            ):
                # Aynı karenin ikinci değerlendirmesi — yeni kare değil.
                if use_flicker_grace:
                    return bool(self._target_lock_confirmed)
                return True
            self._last_target_frame_fp = frame_fp
            self._consecutive_raw_target_frames = getattr(
                self, "_consecutive_raw_target_frames", 0
            ) + 1
            self._consecutive_no_target_frames = 0
            if (
                self._consecutive_raw_target_frames
                >= self.TARGET_BAR_ENTRY_CONFIRM_FRAMES
            ):
                self._target_lock_confirmed = True
                self._last_raw_target_detected = True
                self._flicker_grace_counter = 0
                self.is_flicker_grace_active = False
                return True
            if not use_flicker_grace:
                # Ham yoklama: ilk karede bile True (eski davranış), ama
                # kilit henüz doğrulanmış sayılmaz — grace buna dayanmaz.
                self._last_raw_target_detected = True
                return True
            return False

        # Target lost on this frame. Check if eligible for 1-frame flicker grace.
        # NOT: grace verilirse doğrulama sayacı SIFIRLANMAZ — kilit hâlâ
        # sürüyordur; bar geri geldiğinde aynı kilidin devamı sayılır,
        # yeniden 2 kare beklenmez. Sayaç yalnızca kilit gerçekten
        # düştüğünde (aşağıdaki kayıp dalı) sıfırlanır.
        self._consecutive_no_target_frames = getattr(self, "_consecutive_no_target_frames", 0) + 1
        if (
            use_flicker_grace
            and self._last_raw_target_detected
            and self._target_lock_confirmed
            and self._consecutive_no_target_frames <= self.max_flicker_grace_ticks
        ):
            self._flicker_grace_counter = self._consecutive_no_target_frames
            self.is_flicker_grace_active = True
            logger.debug(
                "Targeting flicker grace active (frame %d/%d) - combat sustained",
                self._consecutive_no_target_frames,
                self.max_flicker_grace_ticks,
            )
            return True

        # Grace expired or not eligible -> confirmed target loss
        self.is_flicker_grace_active = False
        self._last_raw_target_detected = False
        self._target_lock_confirmed = False
        self._consecutive_raw_target_frames = 0
        self._last_target_frame_fp = None
        self._flicker_grace_counter = 0
        return False

    # ========================================================================
    # 1. SAVAŞ ALGILAMA (CHECK_UNDER_ATTACK & HP / TARGET BAR ANALİZİ)
    # ========================================================================

    def _read_hp_ratio_or_none(self, frame: np.ndarray) -> Optional[float]:
        """
        Alt-ortadaki oyuncu can barının (x: 0.345..0.478, y: 0.885..0.902) VE
        oyuncunun baş üstü can barının (x: 0.472..0.526, y: 0.470..0.482) doluluk oranını
        [0.0, 1.0] aralığında hesaplar.
        Eğer her iki bölgede de kırmızı bar pikseli eşleşmezse (HUD gizli veya ölçek farklı),
        kare yoksa ya da analiz hatası olursa None döndürür — okunamayan HP asla
        %100 varsayılmaz (F-0010/F-0072: fail-open kaldırıldı).
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return None

            h, w, _ = bgr.shape
            if h < 50 or w < 50:
                return None

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
                # 0/255 -> 0/1: downstream column math counts pixels, not 255s.
                mask_bot = self._red_mask_from_hsv(hsv_bot, 95, 85, h_lo=12, h_hi=168) // 255

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
                        # active_ratio yalnızca efektif pencere içindeki
                        # sütunları sayar: çerçeve/icon bölgesindeki kırmızı
                        # pikseller HP'yi olduğundan yüksek gösterirdi.
                        eff_valid = valid_cols[valid_cols >= eff_start]
                        span_px = max(0, int(valid_cols[-1]) - max(eff_start, int(valid_cols[0])) + 1)
                        rightmost_ratio = float(int(valid_cols[-1]) - eff_start + 1) / float(eff_width)
                        active_ratio = float(eff_valid.size) / float(eff_width)
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
                # 0/255 -> 0/1: downstream column math counts pixels, not 255s.
                ov_red = self._red_mask_from_hsv(hsv_ov, 135, 105, h_lo=10, h_hi=170) // 255
                ov_w = ov_red.shape[1]
                if ov_w > 0:
                    ov_cols = np.where(np.sum(ov_red, axis=0) >= 1)[0]
                    if ov_cols.size >= max(5, int(ov_w * 0.10)):
                        ov_matched = True
                        eff_ov_start = int(ov_w * 0.11)
                        eff_ov_width = max(1, ov_w - eff_ov_start)
                        # active yalnızca efektif pencere içinden sayılır
                        # (alt bardaki düzeltmeyle aynı gerekçe).
                        eff_ov_cols = ov_cols[ov_cols >= eff_ov_start]
                        ov_rightmost = float(int(ov_cols[-1]) - eff_ov_start + 1) / float(eff_ov_width)
                        ov_active = float(eff_ov_cols.size) / float(eff_ov_width)
                        ov_ratio = float(np.clip(max(ov_rightmost, ov_active), 0.0, 1.0))

            if bot_matched and ov_matched:
                return float(np.clip(max(bot_ratio, ov_ratio), 0.0, 1.0))
            if bot_matched:
                return bot_ratio
            if ov_matched:
                return ov_ratio

            # Hiçbir kırmızı HP barı pikseli eşleşmediyse varsayılan 1.0 (%100 HP) döndür
            return None
        except Exception as exc:
            logger.error("Oyuncu HP oranı (_read_hp_ratio) hesaplanırken hata: %s", exc)
            return None


    def _read_hp_ratio(self, frame: np.ndarray) -> float:
        """
        HP oranını [0.0, 1.0] aralığında döndürür. Okunamazsa 0.0 döner — 0.0
        "bilinmiyor" demektir, "ölü" demek değildir: ölüm kararı yalnızca
        dirilme penceresi + gri ekran + 5 kare doğrulamasıyla verilir.
        Okunabilirliği ayırt etmek isteyen çağıranlar read_hp_ratio_or_none
        kullanmalıdır.
        """
        ratio = self._read_hp_ratio_or_none(frame)
        return 0.0 if ratio is None else float(ratio)

    def read_hp_ratio_or_none(self, frame: np.ndarray) -> Optional[float]:
        """Public alias: HP barı eşleşmezse None (okunamadı), yoksa [0.0, 1.0]."""
        return self._read_hp_ratio_or_none(frame)

    def _hp_bar_matched(self, frame: np.ndarray) -> bool:
        """True ise HP barı pikselleri gerçekten eşleşti (HP okunabiliyor)."""
        return self._read_hp_ratio_or_none(frame) is not None

    # -- R06: hedef HP okuma + atış-sonrası durgunluk takibi (canlı yol) --
    def _read_target_hp_pct(self, frame: np.ndarray) -> Optional[float]:
        """Hedef HP'sini [0.0, 100.0] okur; okunamazsa None.

        Kaynak: vision.read_player_hp_mp_bars(frame) — CombatSystem'ın
        kullandığı okuyucunun aynısı (core/vision.py:2433; has_target +
        target_hp_pct). Okuyucu yoksa / hedef yoksa / değer yoksa None —
        bilinmeyen HP asla %100 varsayılmaz.
        """
        vision = getattr(self, "vision", None)
        reader = getattr(vision, "read_player_hp_mp_bars", None)
        if not callable(reader):
            return None
        try:
            vitals = reader(frame)
        except Exception as exc:
            logger.debug("Hedef HP okunamadı: %s", exc)
            return None
        if not isinstance(vitals, dict) or not vitals.get("has_target", False):
            return None
        hp = vitals.get("target_hp_pct")
        if hp is None:
            return None
        try:
            return float(hp)
        except (TypeError, ValueError):
            return None

    def _can_grab_fresh_frame(self) -> bool:
        """Vision'dan taze kare yakalanabilir mi (settle sonrası re-read için)?"""
        vision = getattr(self, "vision", None)
        if vision is None or isinstance(vision, np.ndarray):
            return False
        if callable(vision):
            return True
        return any(
            callable(getattr(vision, m, None))
            for m in ("get_latest_frame", "grab_frame", "capture_frame", "grab_screen")
        )

    def _record_postcast_outcome(self, target_hp_pct: float) -> bool:
        """Bir atışın hedef HP sonucunu kaydeder; hasar varsa True.

        CombatSystem'daki LineOfSightTracker.record_cast_outcome
        (core/combat.py:745) ile aynı anlambilim: ilk okuma yalnızca
        bazdır (hüküm yok); %0,5'ten fazla düşüş hasardır (sayaç
        sıfırlanır); aksi halde ardışık hasarsız atış sayacı +1.
        """
        curr_hp = float(target_hp_pct)
        prev_hp = self._postcast_target_hp
        self._postcast_target_hp = curr_hp
        if prev_hp is None:
            return False
        if curr_hp < float(prev_hp) - float(self.POSTCAST_HP_DROP_EPS_PCT):
            self._consecutive_stagnant_casts = 0
            return True
        self._consecutive_stagnant_casts += 1
        return False

    def _postcast_settle_and_reread(self) -> Optional[float]:
        """R06: 0,30 sn bekle, TAZE kare yakala, hedef HP'yi yeniden oku.

        CombatSystem'daki aynı-kare yeniden okumanın aksine (orada
        `frame` parametresi — donmuş kare — yeniden okunur, yani hep
        aynı HP döner): settle sonrası vision'dan yeni kare alınır.
        Taze kare yoksa None döner — bekleme de yapılmaz, davranış
        değişmez.
        """
        if not self._can_grab_fresh_frame():
            return None
        time.sleep(float(self.POSTCAST_HP_SETTLE_SEC))
        fresh = self._grab_frame_from_capture(getattr(self, "vision", None))
        if fresh is None:
            return None
        return self._read_target_hp_pct(fresh)

    def _after_cast_postcast_tracking(
        self, pre_cast_target_hp: Optional[float], input_ctrl: Any
    ) -> None:
        """Atış sonrası R06 takibi: baz + settle + durgunluk kararı.

        pre_cast_target_hp None ise (okuyucu yok / Tab ile hedef
        değişti) hiçbir şey yapılmaz. Durgunluk eşiğine (3 ardışık
        hasarsız atış) ulaşılırsa kanıtlı Tab-reset yapılır — bu,
        CombatSystem'daki LoS kurtarma zincirinin son fazıdır
        (TAB_RESET); strafe/flank fazları taşınmadı: canlı adımda
        hedef konumu/obstacle girdisi yok, uydurma konumla manevra
        yapılmaz.
        """
        if pre_cast_target_hp is None:
            return
        is_first_reading = self._postcast_target_hp is None
        self._record_postcast_outcome(pre_cast_target_hp)
        if is_first_reading:
            post_hp = self._postcast_settle_and_reread()
            if post_hp is not None:
                self._record_postcast_outcome(post_hp)
        if self._consecutive_stagnant_casts >= int(self.DAMAGE_STAGNATION_THRESHOLD):
            logger.warning(
                "Hasar durgunluğu: %d ardışık atışta hedef HP düşmedi — "
                "hedef değiştiriliyor (Tab).",
                int(self._consecutive_stagnant_casts),
            )
            self.fast_tab_cycle(input_ctrl, force=True)
            self._postcast_target_hp = None
            self._consecutive_stagnant_casts = 0

    def _is_revived(self, frame: np.ndarray) -> bool:
        """
        Gerçek dirilme: HP barı eşleşiyor VE oranı > %50 VE dirilme penceresi
        kapalı. Okunamayan HP (None) asla 'dirildi' sayılmaz (F-0070/F-0009).
        """
        ratio = self._read_hp_ratio_or_none(frame)
        if ratio is None or ratio <= 0.50:
            return False
        return not self._detect_revive_dialog(frame)

    def get_player_hp_ratio(self, frame: np.ndarray) -> float:
        """
        Alt-ortadaki oyuncu can barının ve baş üstü HP barının doluluk oranını
        [0.0, 1.0] aralığında hesaplar (_read_hp_ratio ile eşdeğer).
        Okunamazsa 0.0 döner — 0.0 "bilinmiyor" demektir; okunabilirlik için
        read_hp_ratio_or_none kullanın.
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

            # Tek tam-kare HSV dönüşümü: 4 şerit için ayrı cvtColor,
            # aynı pikselleri 4 kez dönüştürüyordu (cvtColor piksel-bağımsız
            # olduğundan sonuç aynıdır).
            hsv_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            strips = [
                hsv_full[0:bh, :],
                hsv_full[h - bh : h, :],
                hsv_full[bh : h - bh, 0:bw],
                hsv_full[bh : h - bh, w - bw : w],
            ]
            red_ratios: List[float] = []
            for strip in strips:
                if strip.size == 0:
                    continue
                # strip zaten hsv_full'un HSV dilimi: ikinci cvtColor yok.
                # _red_mask_from_hsv (OURS yardımcısı) HSV alır, 0/255 maske verir.
                red_mask = self._red_mask_from_hsv(strip, 145, 110, h_lo=10, h_hi=170)
                # inRange mask is 0/255; divide back to a true pixel fraction.
                red_ratios.append(float(np.mean(red_mask)) / 255.0)

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
            # 0/255 mask: morphology + connectedComponents treat any nonzero
            # as foreground, same as the old 0/1 mask.
            red_tot = self._red_mask_from_hsv(hsv_tot, 115, 90, h_lo=12, h_hi=168)

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
                return False

            # Ana hedef barındaki mesafe etiketi bölgesi (x: 0.545..0.582, y: 0.028..0.048, örn: '15m')
            dx1, dx2 = int(round(w * 0.545)), int(round(w * 0.582))
            dy1, dy2 = int(round(h * 0.028)), int(round(h * 0.048))
            roi_dist = bgr[dy1:dy2, dx1:dx2]
            if roi_dist.size > 0:
                hsv_dist = cv2.cvtColor(roi_dist, cv2.COLOR_BGR2HSV)
                # Parlak beyaz/gri mesafe rakamları ('15m') var mı kontrol et
                # 0/255 mask: count_nonzero keeps the old boolean-sum count.
                bright_text = cv2.inRange(hsv_dist, (0, 0, 145), (179, 75, 255))
                if int(np.count_nonzero(bright_text)) >= max(5, int(roi_dist.size // 90)):
                    return True

            # Mesafe yazısı bulunamadı: '<20m' doğrulanamadı, yakın-agro sayılmaz.
            return False
        except Exception as exc:
            logger.error("Target-of-Target agro analizi hatası: %s", exc)
            return False

    def _is_bottom_hud_visible(
        self,
        frame: np.ndarray,
        hp_matched: Optional[bool] = None,
        hp_ratio_or_none: Optional[float] = None,
        hp_ratio_known: bool = False,
    ) -> bool:
        """
        Alt HUD arayüzünün (HP veya MP barının) ekranda görünür olup olmadığını kontrol eder.
        Tam ekran harita (M) açıldığında hem HP hem MP barı gizlendiği için yanlış pozitifleri önler.
        hp_ratio_known=True ise HP okuması çağıran tarafından zaten yapılmıştır
        (hp_ratio_or_none verilir); hp_matched verildiyse (çağıran HP barını
        bu karede zaten okuduysa) o bool kullanılır; ikisi de yoksa eski
        davranışla HP okunarak karar verilir. Her durumda aynı kare için
        _read_hp_ratio_or_none tekrar çalıştırılmaz.
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
                mp_mask = cv2.inRange(hsv_mp, (82, 85, 75), (122, 255, 255))
                if int(np.count_nonzero(mp_mask)) >= max(25, int(mp_roi.shape[0] * mp_roi.shape[1] * 0.12)):
                    return True

            # HP barı gerçekten eşleştiyse HUD görünür; okunamayan HP (None)
            # eskisi gibi 1.0 varsayılıp HUD'u 'görünür' saymaz (F-0010).
            if hp_ratio_known:
                return hp_ratio_or_none is not None
            if hp_matched is not None:
                return bool(hp_matched)
            return self._read_hp_ratio_or_none(bgr) is not None
        except Exception as exc:
            logger.error("HUD görünürlük kontrolünde hata: %s", exc)
            return False

    def _raw_detect_enemy_target_bar(self, frame: np.ndarray) -> bool:
        """
        Doğrudan bilgisayarlı görü ile ekranın üst-ortasında (Target Frame) veya
        merkez savaş bölgesinde (Enemy Nameplate / Mob Bar) kırmızı, pembe veya
        zeytin-yeşili (olive-green) hedef can barı olup olmadığını zamansal tolerans
        olmadan (raw) tespit eder.
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
                # AION2 top frame: red + pink/magenta fill + olive-green mob fill.
                # 0/255 mask: morphology + connectedComponents treat any nonzero
                # as foreground, same as the old 0/1 mask.
                mask_top = self._hostile_bar_mask_from_hsv(
                    hsv_top, red_s=90, red_v=80, pink_s=60, pink_v=100
                )
                kernel = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
                mask_top = cv2.morphologyEx(mask_top, cv2.MORPH_CLOSE, kernel)

                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_top)
                min_w_top = max(28, int(round(36 * linear_scale)))
                max_h_top = max(14, int(round(32 * linear_scale)))
                min_area_top = max(60, int(round(90 * scale)))

                for idx in range(1, num_labels):
                    _, _, bw, bh, area = stats[idx]
                    aspect = float(bw) / float(max(1, bh))
                    if bw >= min_w_top and 2 <= bh <= max_h_top and aspect >= 2.4 and area >= min_area_top:
                        return True

            # 2. BÖLGE: üst-orta savaş alanı pembe/kırmızı/zeytin-yeşili hedef HP şeridi.
            try:
                if self._detect_combat_zone_hostile_bar(
                    bgr, linear_scale=linear_scale, scale=scale
                ):
                    return True
            except Exception:
                pass

            # 3. BÖLGE: Özel olive-green mob HP barı / isimliği tespiti
            try:
                if self.detect_olive_mob_bar(bgr):
                    return True
            except Exception:
                pass

            # 4. BÖLGE: Soft cyan/blue selection reticle under character (AION2 lock ring).
            try:
                if self._detect_selection_reticle(
                    bgr, linear_scale=linear_scale, scale=scale
                ):
                    return True
            except Exception:
                pass
            return False
        except Exception as exc:
            logger.error("Düşman hedef barı analizi sırasında hata: %s", exc)
            return False

    def _detect_enemy_target_bar(
        self,
        frame: np.ndarray,
        use_flicker_grace: bool = True,
        require_entry_confirmation: bool = True,
    ) -> bool:
        """
        Ekranın üst-ortasında (Target Frame) veya merkez savaş bölgesinde (Enemy Nameplate)
        saldırgan hedef (kırmızı, pembe veya zeytin-yeşili mob) barı olup olmadığını tespit eder.
        1-frame targeting flicker grace ticks uygular, böylece savaş esnasındaki
        anlık görsel titreşimlerde savaşın erken sonlanmasını (abort) önler.
        require_entry_confirmation=False: savaş-içi takipte ham True kilidi
        anında doğrular (giriş doğrulaması uygulanmaz).
        """
        raw = self._raw_detect_enemy_target_bar(frame)
        return self._apply_targeting_flicker_grace(
            raw,
            use_flicker_grace=use_flicker_grace,
            frame=frame,
            require_entry_confirmation=require_entry_confirmation,
        )

    def detect_enemy_target_bar(
        self,
        frame: np.ndarray,
        use_flicker_grace: bool = True,
        require_entry_confirmation: bool = True,
    ) -> bool:
        """Public arayüz: Hedef canavar barı / zeytin-yeşili mob tespiti."""
        return self._detect_enemy_target_bar(
            frame,
            use_flicker_grace=use_flicker_grace,
            require_entry_confirmation=require_entry_confirmation,
        )

    def _detect_combat_zone_hostile_bar(
        self,
        bgr: np.ndarray,
        *,
        linear_scale: float = 1.0,
        scale: float = 1.0,
    ) -> bool:
        """Pink/red/olive hostile & mob HP strip in mid viewport — not bottom HUD / self overhead."""
        h, w, _ = bgr.shape
        y1 = max(0, int(round(h * self.COMBAT_ZONE_Y1_NORM)))
        y2 = min(h, int(round(h * min(0.42, self.COMBAT_ZONE_Y2_NORM))))
        x1 = max(0, int(round(w * self.COMBAT_ZONE_X1_NORM)))
        x2 = min(w, int(round(w * self.COMBAT_ZONE_X2_NORM)))
        if y2 <= y1 or x2 <= x1:
            return False
        roi = bgr[y1:y2, x1:x2].copy()
        # Mask self overhead HP/MP so player bars never count as enemy lock.
        sx1 = max(0, int(round(w * self.SELF_OVERHEAD_X1_NORM)) - x1)
        sx2 = min(roi.shape[1], int(round(w * self.SELF_OVERHEAD_X2_NORM)) - x1)
        sy1 = max(0, int(round(h * self.SELF_OVERHEAD_Y1_NORM)) - y1)
        sy2 = min(roi.shape[0], int(round(h * self.SELF_OVERHEAD_Y2_NORM)) - y1)
        if sx2 > sx1 and sy2 > sy1:
            roi[sy1:sy2, sx1:sx2] = 0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 0/255 mask: morphology + connectedComponents treat any nonzero
        # as foreground, same as the old 0/1 mask.
        mask = self._hostile_bar_mask_from_hsv(hsv, red_s=100, red_v=90, pink_s=70, pink_v=110)
        kernel = np.ones((2, max(3, int(round(5 * linear_scale)))), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        min_w = max(36, int(round(48 * linear_scale)))
        max_h = max(12, int(round(22 * linear_scale)))
        min_area = max(80, int(round(110 * scale)))
        for idx in range(1, num_labels):
            _, _, bw, bh, area = stats[idx]
            aspect = float(bw) / float(max(1, bh))
            if bw >= min_w and 3 <= bh <= max_h and aspect >= 3.0 and area >= min_area:
                return True
        return False

    def _detect_selection_reticle(
        self,
        bgr: np.ndarray,
        *,
        linear_scale: float = 1.0,
        scale: float = 1.0,
    ) -> bool:
        """True when a cyan/blue soft-target ring is under/near the character."""
        try:
            h, w, _ = bgr.shape
            # Karakter ayakları / yakın zemin (3. şahıs)
            y1 = max(0, int(round(h * 0.48)))
            y2 = min(h, int(round(h * 0.78)))
            x1 = max(0, int(round(w * 0.32)))
            x2 = min(w, int(round(w * 0.68)))
            if y2 <= y1 or x2 <= x1:
                return False
            roi = bgr[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 0/255 mask: count_nonzero / sum(axis) > 0 checks below only
            # test for nonzero, so the 255 scale changes nothing.
            cyan = cv2.inRange(hsv, (85, 85, 110), (115, 255, 255))
            px = int(np.count_nonzero(cyan))
            # Stricter than HUD chrome / water tint false positives.
            min_px = max(180, int(round(320 * scale)))
            if px < min_px:
                return False
            # Yatay yayılım: halka genişliği dar bir blob olmamalı
            cols = np.where(cyan.sum(axis=0) > 0)[0]
            rows = np.where(cyan.sum(axis=1) > 0)[0]
            if cols.size < 2 or rows.size < 2:
                return False
            span_x = int(cols[-1] - cols[0] + 1)
            span_y = int(rows[-1] - rows[0] + 1)
            min_span = max(36, int(round(52 * linear_scale)))
            max_span_x = max(120, int(round(220 * linear_scale)))
            # Full-width cyan UI bands are not a soft-target ring.
            if span_x > max_span_x:
                return False
            return span_x >= min_span and span_y >= max(14, int(round(18 * linear_scale)))
        except Exception:
            return False

    def check_under_attack(self, frame: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Karakterin saldırı altında olup olmadığını tespit eder.
        Koşullar:
        1. Ardışık karelerde gerçek HP düşüşü / hasar (hp_dropping) VEYA
        2. Kırmızı ekran kenarı parlaması (red vignette flash) VEYA
        3. Üstteki aktif Target-of-Target barı (x: 0.62..0.71, y: 0.025..0.065 '45 Muses') + yakın mesafe (< 20m) VEYA
        4. Saldırgan kırmızı düşman hedef barı (ardışık 2 farklı karede doğrulanmış).

        Sabit düşük HP SEVİYESİ tek başına saldırı sayılmaz (bkz. hp_dropping
        notu) — saldırı ancak gerçek düşüşle, barla, vignette ile veya
        ToT ile sayılır.

        Dönüş:
            (is_under_attack: bool, hp_ratio: float)
        """
        try:
            if frame is None and getattr(self, "vision", None) is not None:
                if hasattr(self.vision, "grab_screen"):
                    frame = self.vision.grab_screen(grayscale=False)
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return False, 0.0

            # HP barı bu karede YALNIZCA bir kez okunur; sonuç hem oran hem
            # HUD görünürlüğü için paylaşılır (_is_bottom_hud_visible'a
            # hp_ratio_or_none + hp_ratio_known olarak verilir, ikinci
            # tarama yapılmaz).
            hp_ratio_or_none = self._read_hp_ratio_or_none(bgr)
            hp_ratio = 0.0 if hp_ratio_or_none is None else float(hp_ratio_or_none)
            tot_agro = self._detect_target_of_target_agro(bgr)
            # fix/v20-panic-combat: ToT aggro işareti yayınlanır (rota
            # kapısı + panik katmanı için saldıran-düşman kanıtı).
            self.last_target_of_target_agro = bool(tot_agro)
            vignette_flash = self._detect_red_vignette_flash(bgr)
            enemy_bar_detected = tot_agro or self._detect_enemy_target_bar(bgr, use_flicker_grace=True)
            hud_visible = self._is_bottom_hud_visible(
                bgr, hp_ratio_or_none=hp_ratio_or_none, hp_ratio_known=True
            )

            # Ardışık örneklem geçmişini güncelle
            prev_hp = self._hp_history[-1] if self._hp_history else hp_ratio
            if hud_visible and hp_ratio > 0.01:
                self._hp_history.append(hp_ratio)
                self.last_hp_ratio = hp_ratio

            # HP düşüş analizi (ardışık karelerde anlamlı azalma).
            # DÜZELTME (yanlış "saldırı altında"): HP SEVİYESİ saldırı
            # kararı vermez. Sabit %90 HP (hasar almadan %94'ün altında
            # duran bir bar) saldırı değildir; eski
            # `hp_ratio <= damaged_hp_threshold (0.94)` dalı, hiç hasar
            # gelmese bile her karede hp_dropping=True üretiyordu.
            # damaged_hp_threshold artık saldırı bayrağını asla
            # beslemez (yalnızca iksir eşiği niyetiyle parametre olarak
            # durur; repo'da iksir kodu da onu okumuyor — bkz. rapor).
            # Saldırı ancak GERÇEK düşüşle sayılır: son 10 karenin
            # max'ından veya bir önceki kareden >= %1.5 azalma.
            hp_dropping = False
            if hud_visible and 0.01 < hp_ratio < 1.0:
                recent_max_hp = max(self._hp_history) if self._hp_history else hp_ratio
                if (recent_max_hp - hp_ratio) >= 0.015 or (prev_hp - hp_ratio) >= 0.015:
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
            # Hata = bilinmiyor: 'saldırı yok + tam can' (False, 1.0) fail-open
            # raporlanmaz; HP 0.0 (bilinmiyor) döner (F-0067).
            return False, 0.0

    # ========================================================================
    # no27 (F-0419): DOĞRULANMIŞ LOOT AKIŞI
    # Selçuk'un tasarımı: "İkinci F'yi basarsa bozulur. F basınca loot'un
    # BAŞLADIĞI kontrol edilecek, BİTİŞİ kontrol edilip ancak sonra tekrar
    # F basılacak." Başlangıç = kanal barı görüldü; bitiş = bar %90+ oldu
    # veya görüldükten sonra kayboldu. Bar hiç görülmezse loot yok / hedef
    # yok sayılır ve kör ikinci F ASLA basılmaz (F-0050: kör zaman aşımı
    # başarı sayılmaz). Vekil BLOKE olduğundan (PC doğrulaması bekliyor)
    # bu yöntemler sinyal yokken "loot oldu" demez — yalnızca bar
    # GÖRÜLDÜĞÜNDE doğrular.
    # ========================================================================

    def _check_loot_cast_progress(self, frame: Optional[np.ndarray]) -> Tuple[bool, float]:
        """
        Tek karede loot kanal barı yoklaması (gather cast-bar vekili).

        Returns:
            (is_present, progress_pct 0.0..100.0). Okuyucu yoksa veya kare
            yoksa (False, 0.0) — bu "sinyal yok" demektir, "loot bitti"
            değil; çağıran bunu asla başarı saymaz.
        """
        if frame is None:
            return False, 0.0
        reader = self.cast_bar_reader
        if reader is None:
            vision = getattr(self, "vision", None)
            if vision is not None and (
                hasattr(vision, "verify_gather_cast_bar")
                or hasattr(vision, "read_cast_bar_progress")
            ):
                reader = vision
        if reader is None:
            return False, 0.0
        try:
            if hasattr(reader, "verify_gather_cast_bar"):
                is_present, pct = reader.verify_gather_cast_bar(frame)
                if is_present:
                    return True, float(pct)
            if hasattr(reader, "read_cast_bar_progress"):
                ratio = reader.read_cast_bar_progress(frame)
                if ratio > 0.0:
                    return True, round(float(ratio) * 100.0, 1)
        except Exception as exc:
            logger.debug("Loot cast bar okuma hatası: %s", exc)
        return False, 0.0

    def _has_loot_cast_reader(self) -> bool:
        """no27: enjekte edilmiş (veya vision üzerinden) cast-bar okuyucusu var mı?"""
        if self.cast_bar_reader is not None and (
            hasattr(self.cast_bar_reader, "verify_gather_cast_bar")
            or hasattr(self.cast_bar_reader, "read_cast_bar_progress")
        ):
            return True
        vision = getattr(self, "vision", None)
        return vision is not None and (
            hasattr(vision, "verify_gather_cast_bar")
            or hasattr(vision, "read_cast_bar_progress")
        )

    def _wait_for_verified_loot(
        self,
        capture: Any,
        is_cancelled_fn: Optional[Callable[[], bool]] = None,
        fast_mode: bool = False,
    ) -> str:
        """
        F basıldıktan sonra loot'un BAŞLANGIÇ ve BİTİŞ doğrulamasını bekler.

        Returns:
            "verified" — bar görüldü (başlangıç) VE %90+ oldu ya da
                görüldükten sonra 2 kare kayboldu (bitiş). Yalnızca bu
                durumda loot DOĞRULANDI sayılır.
            "not_started" — başlangıç zaman aşımı (1,5 sn / 15 yoklama)
                içinde bar HİÇ görülmedi: loot yok / hedef yok sayılır.
            "not_finished" — bar görüldü ama bitiş zaman aşımında
                (3 sn / 30 yoklama) ne %90+ oldu ne kayboldu.
            "no_reader" — cast-bar okuyucusu enjekte edilmedi:
                doğrulama yapılamadı (üretimde bugünkü durum).
            "cancelled" — bekleme sırasında iptal edildi.
        Kör zaman aşımı ASLA başarı sayılmaz; "not_started" /
        "not_finished" / "no_reader" durumlarında kör ikinci F
        basılmaz — tek F ile yetinilir.
        """
        if not self._has_loot_cast_reader():
            return "no_reader"
        interval = (
            self.LOOT_CAST_POLL_INTERVAL_FAST_SEC
            if fast_mode
            else self.LOOT_CAST_POLL_INTERVAL_SEC
        )

        # Faz 1 — BAŞLANGIÇ: kanal barı görülene kadar yokla.
        start_deadline = time.monotonic() + float(self.LOOT_START_TIMEOUT_SEC)
        bar_seen = False
        pct = 0.0
        for _ in range(int(self.LOOT_START_MAX_POLLS)):
            if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                return "cancelled"
            if time.monotonic() >= start_deadline:
                break
            frame = self._grab_frame_from_capture(capture)
            if frame is not None:
                bar_seen, pct = self._check_loot_cast_progress(frame)
                if bar_seen:
                    break
            time.sleep(interval)
        if not bar_seen:
            logger.info(
                "no27: loot başlangıcı doğrulanamadı (%.1f sn içinde kanal barı "
                "görülmedi) — kör ikinci F basılmıyor.",
                float(self.LOOT_START_TIMEOUT_SEC),
            )
            return "not_started"
        if pct >= float(self.LOOT_CAST_BAR_COMPLETE_PCT):
            logger.info("no27: loot kanal barı ilk görüşte %.1f%% — başlangıç+bitiş doğrulandı.", pct)
            return "verified"

        # Faz 2 — BİTİŞ: bar %90+ olana ya da görüldükten sonra kaybolana
        # kadar yokla (gather emsali: core/state_machine.py:4428-4493).
        end_deadline = time.monotonic() + float(self.LOOT_END_TIMEOUT_SEC)
        missing_frames = 0
        for _ in range(int(self.LOOT_END_MAX_POLLS)):
            if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                return "cancelled"
            if time.monotonic() >= end_deadline:
                break
            frame = self._grab_frame_from_capture(capture)
            if frame is None:
                # Kare yok = gözlem yok; kayboluş sayılmaz.
                time.sleep(interval)
                continue
            present, pct = self._check_loot_cast_progress(frame)
            if present:
                missing_frames = 0
                if pct >= float(self.LOOT_CAST_BAR_COMPLETE_PCT):
                    logger.info("no27: loot bitti doğrulandı (kanal barı %.1f%%).", pct)
                    return "verified"
            else:
                missing_frames += 1
                if missing_frames >= int(self.LOOT_END_MISSING_FRAMES):
                    logger.info("no27: loot bitti doğrulandı (kanal barı görüldükten sonra kayboldu).")
                    return "verified"
            time.sleep(interval)
        logger.info(
            "no27: loot bitişi doğrulanamadı (bar görüldü ama %.1f sn içinde "
            "bitmedi) — bir sonraki F beklenir, kör basılmaz.",
            float(self.LOOT_END_TIMEOUT_SEC),
        )
        return "not_finished"

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
        - R/T tuşları (sağ tık=T, sol tık=R — fare tıklaması yok), Tab ve 1, 2, 3, 4, q, e, r
          tuşlarını insanlaştırılmış aralıklarla kullanır.
        - Hedef öldüğünde TEK F ile ganimeti toplar (loot) ve no27 (F-0419)
          doğrulanmış loot akışını işletir: başlangıç (kanal barı görüldü)
          + bitiş (%90+ ya da görüldükten sonra kayboluş) doğrulanmadan
          bir sonraki F'ye izin verilmez; kör ikinci F ASLA basılmaz.
          Doğrulama yoksa (okuyucu yok / bar görülmedi) tek F ile
          yetinilir; "loot toplandı" iddiası yapılmaz. F10 (v18):
          "CLEARED" ölüm kanıtına bakar, loot kanıtına değil — hedef
          FIGHTING kapısında doğrulanmış (ardışık 2 farklı karede bar)
          ve bar ardışık karelerde kaybolmuşsa (clear_streak >= 2 +
          HP düşüşü durdu) ölüm kanıtlıdır, loot doğrulanmasa da
          "CLEARED" döner. Loot doğrulaması yalnızca ikinci kanaldır:
          ikisi de yoksa sonuç "CLEARED_UNVERIFIED" olur.
        - Savaş max_fight_sec (varsayılan 120 sn) süresini aşarsa "TIMEOUT" döndürür ve
          "Paused: a fight lasted over 2 min" durumuna geçer.
        """
        # F-0234: fast_mode grace'i kalıcı olarak 1'e çekmesin — finally'de geri al.
        saved_flicker_grace_ticks = self.max_flicker_grace_ticks
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

            # F10 (v18): döngüye girerken FIGHTING kapısının doğruladığı
            # kilidin anlık görüntüsü — gerçek-ölüm kanıtının birinci
            # kanalı. Döngüye, kapı hedef barını ardışık 2 farklı karede
            # doğruladıktan sonra girilir (check_under_attack koşul 4);
            # döngü-içi tespit (require_entry_confirmation=False) giriş
            # kararı değildir, yalnızca takiptir — anında-doğrulama bu
            # bayrağı döngü içinde True'ya çeker, o yüzden giriş anındaki
            # değer saklanır. Kapı doğrulaması yoksa (soğuk giriş) clear
            # dalındaki kayboluş tek başına öldürme kanıtı sayılmaz.
            kill_verified_by_gate = bool(
                getattr(self, "_target_lock_confirmed", False)
            )

            fight_start = time.monotonic()
            last_tab_time = 0.0
            rotation_idx = 0
            clear_streak = 0
            prev_combat_hp: Optional[float] = None
            # F-0006: kare yokken bile tanımlı olmalı (tanımsız okuma çökertirdi).
            enemy_bar: bool = False
            # F-0419 / no27: loot bayrağı artık "basıldı" değil "DOĞRULANDI"
            # anlamına gelir; her loot döngüsü (clear dalı) başında sıfırlanır.
            loot_verified: bool = False

            # İlk hedef kilitleme (Fast TAB)
            self.fast_tab_cycle(input_ctrl, force=True)
            last_tab_time = time.monotonic()

            # Hızlı birim testlerinde döngü adım süresini uyarla
            fast_mode = float(max_fight_sec) < 5.0
            if fast_mode:
                self.max_flicker_grace_ticks = 1
            min_step_delay = 0.03 if fast_mode else 0.18
            max_step_delay = 0.07 if fast_mode else 0.36

            while True:
                if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                    logger.info("Savaş döngüsü kullanıcı/sistem tarafından iptal edildi.")
                    self.in_combat = False
                    self.reset_flicker_grace()
                    return "CANCELLED"

                elapsed = time.monotonic() - fight_start
                if elapsed >= float(max_fight_sec):
                    # F-0231: mesaj gerçek max_fight_sec'i yansıtır (120 sn -> 'over 2 min').
                    _mfs = float(max_fight_sec)
                    timeout_msg = (
                        f"Paused: a fight lasted over {_mfs / 60.0:.0f} min"
                        if _mfs >= 60.0
                        else f"Paused: a fight lasted over {_mfs:.0f}s"
                    )
                    logger.warning("Savaş zaman aşımı (%.1fs >= %.1fs)! %s", elapsed, float(max_fight_sec), timeout_msg)
                    self.in_combat = False
                    self.reset_flicker_grace()
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
                        # F-0232 notu: burada wait_for_revive() BİLEREK
                        # çağrılmaz — o, bloklayan (300 sn) bir giriş
                        # noktası API'sidir; çağıran (state_machine) ölüm
                        # sonrası beklemeyi tick tabanlı, sınırlı
                        # recover_from_death() yoklamasıyla kendisi yürütür.
                        # Bu dal "DEAD" döndürüp denetimi çağırana bırakır.
                        dead_msg = "Paused: your character died"
                        logger.warning("Savaş sırasında karakter öldü! %s", dead_msg)
                        self.in_combat = False
                        self.reset_flicker_grace()
                        self._notify_status(
                            status_callback,
                            "PAUSED",
                            dead_msg,
                            dead_msg,
                        )
                        return "DEAD"

                    enemy_bar = self._detect_enemy_target_bar(
                        frame, use_flicker_grace=True, require_entry_confirmation=False
                    )
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
                    # (1-frame flicker grace + 2 confirmed frames = 3 frames total protection)
                    if clear_streak >= 2:
                        logger.info("Saldırgan canavar etkisiz hale getirildi. Ganimet (F) toplanıyor...")
                        # no27 (F-0419) — doğrulanmış loot akışı (Selçuk'un tasarımı):
                        # TEK F basılır, sonra BAŞLANGIÇ (kanal barı görüldü) ve
                        # BİTİŞ (%90+ ya da görüldükten sonra kayboluş) doğrulanır.
                        # Eski 0,35 sn kör bekleme kalktı — yerini doğrulama
                        # beklemesi aldı; okuyucu yoksa bekleme hiç yapılmaz.
                        # Bar görülmezse loot yok/hedef yok sayılır, kör ikinci
                        # F basılmaz; bayrak loot döngüsü başına sıfırlanır.
                        loot_verified = False
                        self._safe_press_key(input_ctrl, "f", min_hold=0.05, max_hold=0.09)
                        loot_outcome = self._wait_for_verified_loot(
                            capture, is_cancelled_fn, fast_mode
                        )
                        self.last_loot_outcome = loot_outcome
                        self.loot_verification_outcomes.append(loot_outcome)
                        # D1: liste sınırsız büyüyordu — son 100 sonuç tutulur.
                        del self.loot_verification_outcomes[:-100]
                        if loot_outcome == "verified":
                            loot_verified = True
                            logger.info("no27: loot doğrulandı (başlangıç + bitiş).")
                        elif loot_outcome == "cancelled":
                            logger.info("Savaş döngüsü loot doğrulaması sırasında iptal edildi.")
                            self.in_combat = False
                            self.reset_flicker_grace()
                            return "CANCELLED"
                        # "not_started" / "not_finished" / "no_reader": loot
                        # doğrulanmadı — yine de kör ikinci F YOK, TAB'a geçilir.
                        # 2. Sıradaki canavarı seçmek için Fast TAB tuşuna bas!
                        logger.info("Sıradaki canavarı hedeflemek için Fast TAB cycling...")
                        self.fast_tab_cycle(input_ctrl, force=True)
                        last_tab_time = time.monotonic()
                        time.sleep(0.25 if not fast_mode else 0.05)

                        # Sıradaki canavar hedeflendi mi kontrol et
                        new_frame = self._grab_frame_from_capture(capture)
                        if new_frame is not None and self._detect_enemy_target_bar(new_frame, use_flicker_grace=False):
                            logger.info("Yeni canavar hedeflendi! Savaş döngüsüne devam ediliyor...")
                            clear_streak = 0
                            continue
                        break
                else:
                    # F-0417 (P1-3 + P1-11 + P3-4): kare yokken hedef YOK sayılır.
                    # Eski kod son bilinen hedef durumunu _enemy_bar_history'den
                    # okuyordu; o geçmiş yalnızca check_under_attack'ta yazılır
                    # ve savaş döngüsü boyunca donuktur — bayat True, kanıtsız
                    # hedef varsayımıdır (F-0421'deki "kare yoksa hedef var
                    # varsayılmaz" kuralıyla çelişir). Okuma kaldırılınca tek
                    # okuyucusu kalmayan geçmiş de kaldırıldı (init/append/
                    # clear silindi). Artık kare yoksa enemy_bar=False; TAB
                    # aralığı hızlı aramaya döner. (clear_streak burada
                    # artırılmaz — kör kare temizlik sayılmaz.)
                    enemy_bar = False

                # Fast TAB cycling: Hedef aranıyorsa (~0.55s), kilitliyse periyodik (~4.2-6.5s)
                now = time.monotonic()
                tab_cadence = (
                    self.fast_tab_interval_sec
                    if (not enemy_bar or self.is_flicker_grace_active)
                    else random.uniform(4.2, 6.5)
                )
                if fast_mode:
                    tab_cadence = 0.10
                if (now - last_tab_time) >= tab_cadence:
                    if self.fast_tab_cycle(input_ctrl, min_interval_sec=tab_cadence):
                        last_tab_time = now

                # İnsanlaştırılmış saldırı rotasyonu: R/T tuş + 1,2,3,4,q,e,r,t (sağ tık=T, sol tık=R)
                if rotation_idx % 3 == 0:
                    attack_key = "t" if (rotation_idx % 6 == 0) else "r"
                    self._safe_press_key(input_ctrl, attack_key, min_hold=0.045, max_hold=0.09)
                else:
                    skill_key = self.COMBAT_SKILL_KEYS[rotation_idx % len(self.COMBAT_SKILL_KEYS)]
                    # v20/madde 2: skill tuşunda güvenli alt sınır 0.08-0.12 sn
                    # (45 ms basım oyunda kayda geçmeyebilir).
                    self._safe_press_key(input_ctrl, skill_key, min_hold=0.08, max_hold=0.12)

                rotation_idx += 1
                time.sleep(random.uniform(min_step_delay, max_step_delay))

            # no27 (F-0419): döngü sonrası kör final F KALDIRILDI.
            # Döngü yalnızca clear dalındaki F'ten sonra break eder, yani
            # buraya gelindiğinde bir F zaten basılmıştır; ikinci (kör) F
            # loot'u bozar (Selçuk: "İkinci F'yi basarsa bozulur").
            # loot_verified False ise bile tekrar basılmaz — doğrulanamayan
            # loot kör F ile düzeltilmez; tek F ile yetinilir. Eski davranış
            # (loot_pressed False ise send_gather_key / F + kör bekleme)
            # no27'nin yasakladığı kör ikinci F idi; test_2 de bu yüzden
            # 12 tuş / 3×F bekliyordu (bkz. tests/test_w52_...py:212-216).

            # stop_gather_sec kadar bekle (iptal kontrolü ile parçalı bekleme).
            # Not: bu artık bir "loot beklemesi" değil, savaş-sonrası yerleşme
            # beklemesidir — loot beklemesi clear dalındaki doğrulama
            # yoklamasında yapılır. Çağıranlar 0.0 geçerek kapatabilir.
            wait_end = time.monotonic() + max(0.0, float(stop_gather_sec))
            while time.monotonic() < wait_end:
                if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                    self.in_combat = False
                    self.reset_flicker_grace()
                    return "CANCELLED"
                remaining = wait_end - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))

            self.in_combat = False
            self.reset_flicker_grace()
            self._hp_history.clear()
            # F10 (v18): "CLEARED", iki bağımsız doğrulama kanalından
            # biriyle verilir — (a) öldürme kanıtı: FIGHTING kapısı
            # doğrulamış hedefin barı clear dalında ardışık karelerde
            # kayboldu (clear_streak >= 2 + HP düşüşü durdu); ya da
            # (b) loot kanıtı: loot başlangıç+bitiş doğrulandı.
            # P1-11'in haklı çekirdeği korunur: "loot collected" iddiası
            # YALNIZCA (b)'de yapılır; (a)'da loot doğrulanmadıysa
            # bildirim bunu açıkça söyler ama ölüm kanıtlı olduğu için
            # sonuç CLEARED'dir (v17 davranışı). İki kanal da yoksa
            # (soğuk giriş + doğrulanmamış loot) sonuç
            # "CLEARED_UNVERIFIED" olur.
            if loot_verified or kill_verified_by_gate:
                if loot_verified:
                    self._notify_status(
                        status_callback,
                        "CLEARED",
                        "Target cleared",
                        "Combat finished and loot collected, resuming route...",
                    )
                    logger.info(
                        "Savaş temizlendi (CLEARED), loot doğrulandı, "
                        "rotaya kaldığı yerden devam ediliyor.",
                    )
                else:
                    self._notify_status(
                        status_callback,
                        "CLEARED",
                        "Target cleared",
                        "Combat finished but loot was NOT verified (outcome: %s), "
                        "resuming route..." % (self.last_loot_outcome,),
                    )
                    logger.info(
                        "Savaş temizlendi (CLEARED) — hedef ölümü kapı "
                        "doğrulamasıyla kanıtlı, loot DOĞRULANMADI (%s), "
                        "rotaya kaldığı yerden devam ediliyor.",
                        self.last_loot_outcome,
                    )
                return "CLEARED"
            self._notify_status(
                status_callback,
                "CLEARED_UNVERIFIED",
                "Target cleared, loot unverified",
                "Combat finished but loot was NOT verified (outcome: %s), "
                "resuming route..." % (self.last_loot_outcome,),
            )
            logger.info(
                "Savaş temizlendi ama loot DOĞRULANMADI (CLEARED_UNVERIFIED), "
                "loot doğrulama: %s, rotaya kaldığı yerden devam ediliyor.",
                self.last_loot_outcome,
            )
            return "CLEARED_UNVERIFIED"
        except Exception as exc:
            logger.error("execute_combat_loop sırasında beklenmeyen hata: %s", exc)
            self.in_combat = False
            self.reset_flicker_grace()
            return "ERROR"
        finally:
            self.max_flicker_grace_ticks = saved_flicker_grace_ticks

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

            # inRange masks are 0/255; count_nonzero / size keeps the old
            # boolean-mean ratios exact.
            btn_red_gold = self._red_mask_from_hsv(btn_zone, 110, 100, h_lo=24, h_hi=165)
            btn_ratio = float(np.count_nonzero(btn_red_gold)) / float(btn_zone.size)
            dark_mask = cv2.inRange(hsv_modal, (0, 0, 12), (179, 255, 65))
            modal_dark_ratio = float(np.count_nonzero(dark_mask)) / float(dark_mask.size)

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

            # F-0007/F-0008: kareyle çağrılınca bile 1 kareye düşmez — alt sınır her
            # zaman live_death_confirm_frames (varsayılan 5) olur.
            req_frames = max(int(self.live_death_confirm_frames), int(confirm_frames))
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
        timeout_sec: Optional[float] = 300.0,
    ) -> bool:
        """
        Karakter öldüğünde "Paused: your character died" durumuna geçer ve
        oyuncu dirildiğinde (can barı > %50 dolduğunda) otomatik True döndürerek devam eder.
        timeout_sec varsayılanı 300 sn'dir (F-0009: eskiden None = sonsuz döngü).
        Dirilme yalnızca HP barı gerçekten okunup > %50 ise sayılır; okunamayan
        HP asla 'dirildi' sayılmaz.
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
                    if self._is_revived(frame):
                        hp_ratio = self.get_player_hp_ratio(frame)
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

    def fight_back_step(
        self,
        input_ctrl: Optional[Any] = None,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        StateMachine tarafından her savaş döngüsü adımında çağrılabilen
        insanlaştırılmış karşı saldırı (Tab + yetenek rotasyonu).
        P1-11: hedef yokken False döner — kör tuş basılmaz, saldırı
        yapılmış sayılmaz; loot F'i bu adımda basılmaz (doğrulanmış
        loot akışı state_machine / execute_combat_loop'a aittir).
        """
        try:
            ctrl = input_ctrl if input_ctrl is not None else getattr(self, "input_ctrl", None)
            if ctrl is None and hasattr(self, "vision") and hasattr(self.vision, "driver"):
                ctrl = self.vision.driver
            if ctrl is None and hasattr(self, "driver"):
                ctrl = getattr(self, "driver", None)
            if ctrl is None:
                return False

            step_idx = getattr(self, "_fight_step_counter", 0)
            setattr(self, "_fight_step_counter", step_idx + 1)

            # Hedef kontrolü
            img = frame
            if img is None and hasattr(self, "vision") and self.vision is not None:
                img = self._grab_frame_from_capture(self.vision)

            # F-0421: kare yoksa hedef var varsayılmaz — kör saldırı yok.
            has_target = False
            if img is not None:
                has_target = self._detect_enemy_target_bar(
                    img, use_flicker_grace=True, require_entry_confirmation=False
                )

            if not has_target:
                no_tgt = getattr(self, "_no_target_steps", 0) + 1
                setattr(self, "_no_target_steps", no_tgt)
                # R06: hedef kaybı durgunluk bazını sıfırlar — sonraki
                # atış yeni hedef için ilk atıştır.
                self._postcast_target_hp = None
                self._consecutive_stagnant_casts = 0
                if no_tgt >= 2:
                    # Y2 DÜZELTME: 2 adım hedef yokluğu "canavar öldü"
                    # sayılmaz — bar kaybı flicker/kaçış/okuma hatası da
                    # olabilir. Kör F + kör Tab KALDIRILDI (2-kare eşiği
                    # aynen korunur, yalnızca bu dalın kör aksiyonu
                    # kalktı): loot + kesim sayacı state_machine'ın
                    # korumalı akışına aittir (_loot_confirmed_fight /
                    # death_ready: had_target_bar korumalı tek F). Bu
                    # dalın kör F'i onunla çift F basıyordu (no27
                    # ihlali — ikinci F loot kanalını bozar). Burada ne
                    # F basılır ne Tab çevrilir ne de kesim yazılır.
                    logger.info(
                        "Hedef 2 adımdır yok — ölüm varsayılmıyor "
                        "(kör F/Tab yok); loot state_machine akışına bırakıldı."
                    )
                    setattr(self, "_no_target_steps", 0)
                    # P1-11: hedefsiz adım saldırı sayılmaz — False döner
                    # (eskiden True dönüyor, basılmayan adım "saldırı
                    # yapıldı" sayılıyordu).
                    return False
                # İlk kayıp adım: flicker grace zaten korur — kör Tab yok.
                # P1-11: hedefsiz adım False döner.
                return False
            else:
                setattr(self, "_no_target_steps", 0)

            # Savaşırken düzenli olarak Tab ile en yakın hedefi kilitle (her 3 adımda bir).
            # Tab gerçekten basıldıysa hedef değişmiş olabilir — durgunluk
            # bazı sıfırlanır ve bu adımın atışı kaydedilmez (eski
            # hedefin HP'si yeni hedefe yazılmaz).
            retargeted_this_step = False
            if step_idx % 3 == 0:
                if self.fast_tab_cycle(ctrl, min_interval_sec=1.2):
                    retargeted_this_step = True
                    self._postcast_target_hp = None
                    self._consecutive_stagnant_casts = 0

            # F-0069: spam gate — çok sık ardışık basımları atla (tuş spam'i yok).
            now_ts = time.monotonic()
            min_interval = float(getattr(self, "fight_back_min_press_interval_sec", 0.08))
            if (now_ts - float(getattr(self, "_last_fight_back_press_ts", 0.0))) < min_interval:
                return True
            setattr(self, "_last_fight_back_press_ts", now_ts)

            # R06 (canlı yol): atış öncesi hedef HP'si — atış sonrası
            # durgunluk kararı bu okumayla verilir. Okunamazsa kayıt
            # atlanır, davranış değişmez.
            pre_cast_target_hp: Optional[float] = None
            if img is not None and not retargeted_this_step:
                pre_cast_target_hp = self._read_target_hp_pct(img)

            # Yetenek ve vuruş rotasyonu (sağ tık=T, sol tık=R — fare tıklaması yok)
            pressed = False
            if step_idx % 2 == 0:
                attack_key = "t" if (step_idx % 4 == 0) else "r"
                pressed = self._safe_press_key(ctrl, attack_key, min_hold=0.045, max_hold=0.09)
            else:
                skill_idx = getattr(self, "_fight_skill_idx", 0)
                combat_keys = list(self.COMBAT_SKILL_KEYS)
                # F-0236: hazır skill'ler varsa onları kullan (ölü entegrasyon canlandı).
                # v20/madde 2: hazır liste tek tuşa indiyse (örn. okuyucu
                # yalnızca '2'yi hazır görüyorsa) rotasyon o tuşa
                # kilitlenmesin — hazır tuşlar önce, ardından tam havuz.
                if img is not None:
                    try:
                        ready = self.get_ready_combat_skills(img)
                        if ready:
                            ready_keys = [k for k in ready if k in self.COMBAT_SKILL_KEYS]
                            if ready_keys:
                                combat_keys = ready_keys + [
                                    k for k in self.COMBAT_SKILL_KEYS if k not in ready_keys
                                ]
                    except Exception:
                        pass
                skill_key = combat_keys[skill_idx % len(combat_keys)]
                setattr(self, "_fight_skill_idx", (skill_idx + 1) % len(combat_keys))
                # v20/madde 2: skill tuşunda güvenli alt sınır 0.08-0.12 sn
                # (45 ms basım oyunda kayda geçmeyebilir).
                pressed = self._safe_press_key(ctrl, skill_key, min_hold=0.08, max_hold=0.12)
            # O2 (canlı yol): basılmayan adım "saldırı yapıldı" sayılmaz —
            # girdi yoksa False dönülür, atış sonucu da kaydedilmez.
            if not pressed:
                return False
            self._after_cast_postcast_tracking(pre_cast_target_hp, ctrl)
            return True
        except Exception as exc:
            logger.error("fight_back_step sırasında hata: %s", exc)
            return False

    def fight_back(self, input_ctrl: Optional[Any] = None, frame: Optional[np.ndarray] = None) -> bool:
        """StateMachine uyumluluğu için karşı saldırı adımı tetikler."""
        return self.fight_back_step(input_ctrl=input_ctrl, frame=frame)

    def execute_combat_step(self, input_ctrl: Optional[Any] = None, frame: Optional[np.ndarray] = None) -> bool:
        """StateMachine uyumluluğu için karşı saldırı adımı tetikler."""
        return self.fight_back_step(input_ctrl=input_ctrl, frame=frame)

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
            if self._is_revived(frame):
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
            # (F-0233: eskiden 'and not use_fly' koşulu yüzünden varsayılan configde
            # hiç çalışmıyordu — deneme 2 artık her zaman geri-çekilme yapar.)
            if attempt_num == 2:
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
            # inRange masks are 0/255; count_nonzero / size keeps the old
            # boolean-mean ratios exact.
            warn_mask = cv2.inRange(hsv, (5, 180, 185), (22, 255, 255))
            dark_mask = cv2.inRange(hsv, (0, 0, 10), (179, 255, 45))
            warn_ratio = float(np.count_nonzero(warn_mask)) / float(warn_mask.size)
            dark_ratio = float(np.count_nonzero(dark_mask)) / float(dark_mask.size)
            if warn_ratio >= 0.045 and dark_ratio >= 0.60:
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
        max_wait_sec: float = 3600.0,
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
            wait_capped = False
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
                    # F-0235: bekleme max_wait_sec (varsayılan 3600 sn) ile sınırlı —
                    # 24 saate kadar donma yok.
                    wait_started = time.monotonic()
                    wait_deadline = wait_started + seconds_until_reset
                    capped_deadline = min(
                        wait_deadline, wait_started + max(60.0, float(max_wait_sec))
                    )
                    logger.info(
                        "Günlük limit sıfırlanana kadar bekleniyor (%.0f sn, cap %.0f sn). Anti-AFK aktif.",
                        seconds_until_reset,
                        max(60.0, float(max_wait_sec)),
                    )
                    while time.monotonic() < capped_deadline:
                        if is_cancelled_fn is not None and callable(is_cancelled_fn) and is_cancelled_fn():
                            break
                        if (time.monotonic() - self.last_anti_afk_time) >= effective_interval:
                            anti_afk_done = self.perform_anti_afk_step(input_ctrl)
                        time.sleep(min(0.25, max(0.02, capped_deadline - time.monotonic())))
                    if time.monotonic() >= capped_deadline and time.monotonic() < wait_deadline:
                        wait_capped = True
                        logger.warning(
                            "Günlük limit beklemesi %.0f sn cap ile kesildi; "
                            "sıfırlanmaya hâlâ %.0f sn var.",
                            max(60.0, float(max_wait_sec)),
                            wait_deadline - time.monotonic(),
                        )

            return {
                "limit_reached": limit_active,
                "anti_afk_performed": anti_afk_done,
                "seconds_until_reset": round(seconds_until_reset, 2),
                "status": "WAITING_DAILY_LIMIT" if limit_active else "ACTIVE",
                "wait_capped": wait_capped,
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
        """Reads ready (off-cooldown) skill keys via AdaptiveSkillBarReader, falling back to COMBAT_SKILL_KEYS.

        O3: geri düşüş SESSİZ değildir — okuyucu yoksa / boş dönerse
        (60 sn'de en çok bir kez) warning loglanır. Davranış aynen
        korunur (tüm tuşlar): okuyucu bugün güvenilmez olduğu için
        (hazır/beklemede ayrımı hatalı — dilim 10/28) fail-closed
        yapılsaydı bot skill basamazdı; kör rotasyon, bilinçli ve
        görünür bir geri düşüştür.
        """
        if frame is not None:
            try:
                # G→B devri (v11 entegrasyonu): paylaşımlı okuyucu — kilit
                # (0,15 sn basım kilidi) örnek başına tutulduğu için kendi
                # örneğini yaratan combat_recovery, game_action_bus'ın
                # kaydettiği basımı görmüyordu; tek örnek = sistem
                # genelinde tek kilit (skill_bar_reader.py:94).
                from core.skill_bar_reader import get_shared_skill_bar_reader

                res = get_shared_skill_bar_reader().get_ready_skills(frame)
                if isinstance(res, list) and res:
                    return [str(k) for k in res]
            except Exception as exc:
                self._warn_skill_reader_fallback("okunamadı: %s" % exc)
                return list(self.COMBAT_SKILL_KEYS)
        self._warn_skill_reader_fallback("hazır skill döndürmedi")
        return list(self.COMBAT_SKILL_KEYS)

    def _warn_skill_reader_fallback(self, reason: str) -> None:
        """O3 geri-düşüş uyarısı — log şişmesin diye 60 sn'de en çok bir kez."""
        now = time.monotonic()
        last = float(getattr(self, "_last_skill_fallback_warn_ts", 0.0) or 0.0)
        if now - last < 60.0:
            return
        self._last_skill_fallback_warn_ts = now
        logger.warning(
            "Skill bar okuyucusu %s — tüm skill tuşları hazır sayılıyor "
            "(kör rotasyon).",
            reason,
        )

    def get_combat_recovery_health_summary(self) -> dict:
        # F-0071: gerçek kontrol — skill havuzu, 5 kare ölüm doğrulaması ve grace.
        try:
            ok = bool(
                self.COMBAT_SKILL_KEYS
                and int(self.live_death_confirm_frames) >= 5
                and int(self.death_guard.required_frames) >= 5
                and int(self.max_flicker_grace_ticks) >= 1
            )
            return {
                "combat_recovery_ready": bool(ok),
                "death_confirm_frames": int(self.live_death_confirm_frames),
                "skill_keys": len(self.COMBAT_SKILL_KEYS),
                "flicker_grace_ticks": int(self.max_flicker_grace_ticks),
            }
        except Exception as exc:
            logger.error("get_combat_recovery_health_summary hatası: %s", exc)
            return {"combat_recovery_ready": False}

    def get_combat_recovery_ready_flag(self) -> dict:
        return {
            "module": "combat_recovery",
            "ready": bool(self.get_combat_recovery_health_summary().get("combat_recovery_ready", False)),
        }

    def get_combat_recovery_version_info(self) -> dict:
        return {
            "module": "combat_recovery",
            "version": __version__,
            "ready": bool(self.get_combat_recovery_health_summary().get("combat_recovery_ready", False)),
        }

    def get_combat_recovery_diagnostic_snapshot(self) -> dict:
        summary = self.get_combat_recovery_health_summary()
        ready = bool(summary.get("combat_recovery_ready", False))
        return {
            "module": "combat_recovery",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "health": summary,
        }


def get_combat_recovery_capabilities() -> list:
    return ["combat_recovery", "retaliation_protocol", "diagnostic_snapshot"]


def validate_combat_recovery_contract() -> bool:
    # F-0071: yalnızca liste varlığına değil, gerçek sağlık özetine bakılır.
    try:
        if not get_combat_recovery_capabilities():
            return False
        probe = CombatRecovery()
        return bool(probe.get_combat_recovery_health_summary().get("combat_recovery_ready", False))
    except Exception:
        return False


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


def get_combat_recovery_integrity_seal() -> dict:
    return {
        "module": "combat_recovery",
        "sealed": validate_combat_recovery_contract(),
        "telemetry": get_combat_recovery_telemetry_payload(),
    }


def get_combat_recovery_compliance_certificate() -> dict:
    return {
        "module": "combat_recovery",
        "certified": validate_combat_recovery_contract(),
        "seal": get_combat_recovery_integrity_seal(),
    }


def get_combat_recovery_readiness_probe() -> dict:
    return {
        "module": "combat_recovery",
        "ready": validate_combat_recovery_contract(),
        "certificate": get_combat_recovery_compliance_certificate(),
    }


def get_combat_recovery_audit_summary() -> dict:
    return {
        "module": "combat_recovery",
        "audit_passed": validate_combat_recovery_contract(),
        "probe": get_combat_recovery_readiness_probe(),
    }



