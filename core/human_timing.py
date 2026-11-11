# -*- coding: utf-8 -*-
"""
Katman III: İstatistiksel İnsanlaştırma [Ex-Gaussian & Log-Normal], Görsel Tepki Gecikmesi,
Dinamik Mola Planlayıcı ve Organik Anti-AFK Motoru (Ajan 23 Modülü).

Özellikler:
1. HumanTimingEngine:
   - Gerçek insan nöro-motor tepki süresi dağılımı (Ex-Gaussian: Gaussian algı/motor fazı +
     Üstel karar/dikkat kuyruğu `tau` ve Log-Normal dağılım).
   - Görsel Tepki Gecikmesi (180ms - 380ms): "F Gather" belirdiği an 0ms'de basmayı önler.
   - F Basım Aralığı (1.5s - 3.0s): Çan eğrisi / Ex-Gaussian dağılımlı tekrar süresi.
   - Harita Açılış Okuma (400ms - 850ms) ve Harita Kapanış (250ms - 500ms) gecikmeleri.
   - Dinamik Yeniden Tıklama Süresi (8.5s - 13.2s): Sabit 10.0s periyodik tıklama izini kırar.
2. DynamicBreakScheduler:
   - Makro Mola (Su/Kahve Molası): Her 45 ± 12 dakikada bir, 6 ± 3 dakika süren mola.
   - Mikro Duraksama (İnsani Dalgınlık): Her 8 - 15 dakikada bir, 3.5s - 9.0s süren duraksama.
3. OrganicAntiAFK:
   - Düzensiz aralıklarla (140s - 260s Ex-Gaussian) sağ tık basılı WindMouse kamera çevirme,
     zıplama veya rastgele yöne birkaç adım yürüme eylemlerinden birini organik biçimde uygular.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from version import __version__

# Modül logger yapılandırması
logger = logging.getLogger("HumanTiming")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# 1. İSTATİSTİKSEL İNSAN ZAMANLAMA MOTORU (EX-GAUSSIAN & LOG-NORMAL)
# ============================================================================

class HumanTimingEngine:
    """
    Düz `random.uniform` yerine gerçek insan nöro-motor tepki dağılımını
    (Ex-Gaussian ve Log-Normal) kullanan istatistiksel zamanlama motoru.

    Matematiksel Model (Ex-Gaussian):
        X = Y + Z
        Y ~ Normal(mu, sigma^2)   -> Bazal duyusal-motor iletim süresi
        Z ~ Exponential(1 / tau)  -> Merkezi sinir sistemi dikkat/karar kuyruğu (sağa çarpıklık)
    """

    def __init__(self, seed: Optional[int] = None, fatigue_enabled: bool = True) -> None:
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.fatigue_enabled: bool = bool(fatigue_enabled)
        self._session_start_ts: float = time.monotonic()
        self._total_samples_generated: int = 0

    @property
    def total_samples_generated(self) -> int:
        """Number of samples produced by this engine so far.

        The counter was already incremented by every sampler but never
        exposed — write-only telemetry. Read under the RNG lock.
        """
        with self._lock:
            return self._total_samples_generated

    def get_fatigue_multiplier(self) -> float:
        """
        Oturum süresine bağlı olarak nöro-motor yorgunluk çarpanını (1.00 - 1.08) hesaplar.
        Saatler ilerledikçe insan tepki süresi çok hafif sağa kayar ve dalgalanır.
        """
        try:
            if not self.fatigue_enabled:
                return 1.0
            elapsed_hours = max(0.0, (time.monotonic() - self._session_start_ts) / 3600.0)
            # İlk 2 saatte ~1.00-1.03, uzun oturumlarda en fazla %6-%8 artış + ultradian dalga
            base_fatigue = min(0.06, elapsed_hours * 0.018)
            ultradian_wave = 0.015 * math.sin(elapsed_hours * math.pi / 1.5)
            return max(0.98, min(1.08, 1.0 + base_fatigue + ultradian_wave))
        except Exception as exc:
            logger.error("Yorgunluk çarpanı hesaplama hatası: %s", exc)
            return 1.0

    def sample_ex_gaussian(
        self,
        mu: float,
        sigma: float,
        tau: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """
        İnsan tepki süresini (Gaussian bileşen `mu, sigma` + üstel kuyruk `tau`)
        matematiksel olarak üretir ve `[min_val, max_val]` aralığında sınırlandırır.

        Keskin sınır yığılmalarını (boundary spike) önlemek için önce reddetme örneklemesi
        (rejection sampling) uygular, çok nadir uç değerlerde yumuşak geri dönüş yapar.
        """
        try:
            low = float(min(min_val, max_val))
            high = float(max(min_val, max_val))
            if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
                return low

            clean_sigma = max(1e-5, float(sigma))
            clean_tau = max(1e-5, float(tau))
            clean_mu = float(mu)

            with self._lock:
                self._total_samples_generated += 1
                # Reddetme örneklemesi (Rejection Sampling) ile doğal dağılım kuyruğunu koru
                for _ in range(32):
                    gaussian_part = self._rng.gauss(clean_mu, clean_sigma)
                    exponential_part = self._rng.expovariate(1.0 / clean_tau)
                    candidate = gaussian_part + exponential_part
                    if low <= candidate <= high:
                        return float(candidate)

                # Nadir taşmalarda aralığın teorik merkezine yakın yumuşak örnekleme
                fallback_mean = max(low, min(high, clean_mu + clean_tau))
                fallback = self._rng.gauss(fallback_mean, clean_sigma)
                return float(max(low, min(high, fallback)))
        except Exception as exc:
            logger.error(
                "Ex-Gaussian örnekleme hatası (mu=%s, sigma=%s, tau=%s): %s",
                mu,
                sigma,
                tau,
                exc,
            )
            return float((min_val + max_val) * 0.5)

    def sample_log_normal(
        self,
        median: float,
        sigma: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """
        Log-Normal dağılım kullanarak sağa çarpık (pozitif kuyruklu) insani süre üretir.
        `median` dağılımın medyan değerini (`exp(mu_log)`), `sigma` ise log-ölçek sapmasını belirtir.
        """
        try:
            low = float(min(min_val, max_val))
            high = float(max(min_val, max_val))
            if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
                return low

            clean_median = max(1e-4, float(median))
            mu_log = math.log(clean_median)
            clean_sigma = max(1e-4, float(sigma))

            with self._lock:
                self._total_samples_generated += 1
                for _ in range(32):
                    candidate = self._rng.lognormvariate(mu_log, clean_sigma)
                    if low <= candidate <= high:
                        return float(candidate)

                fallback = self._rng.gauss(clean_median, (high - low) * 0.15)
                return float(max(low, min(high, fallback)))
        except Exception as exc:
            logger.error(
                "Log-Normal örnekleme hatası (median=%s, sigma=%s): %s",
                median,
                sigma,
                exc,
            )
            return float((min_val + max_val) * 0.5)

    def get_visual_reaction_delay_sec(self) -> float:
        """
        Görsel Tepki Gecikmesi:
        Ekranda 'F Gather' yazısı belirdiği milisaniyede F tuşuna basılmaması için
        180 ms ile 380 ms (0.18s - 0.38s) arasında Ex-Gaussian insani refleks gecikmesi üretir.

        Tipik insan görsel-motor basit reaksiyon süresi (SRT):
        Tepe noktası ~230-255ms, sağ kuyruk 380ms'ye uzanır.
        """
        try:
            fatigue = self.get_fatigue_multiplier()
            mu = 0.218 * min(1.04, fatigue)
            sigma = 0.019
            tau = 0.038 * min(1.06, fatigue)
            delay = self.sample_ex_gaussian(
                mu=mu,
                sigma=sigma,
                tau=tau,
                min_val=0.180,
                max_val=0.380,
            )
            logger.debug("Görsel tepki gecikmesi üretildi: %.1f ms (%.4f s)", delay * 1000.0, delay)
            return delay
        except Exception as exc:
            logger.error("get_visual_reaction_delay_sec hatası: %s", exc)
            return 0.245

    def get_f_press_interval_sec(self, min_s: float = 1.5, max_s: float = 3.0) -> float:
        """
        F Basım Aralığı:
        `random.uniform` yerine çan eğrisi / Ex-Gaussian dağılımlı F tekrar süresi üretir.
        Varsayılan aralık: 1.5s - 3.0s.
        """
        try:
            low = max(0.1, float(min(min_s, max_s)))
            high = max(low + 0.01, float(max(min_s, max_s)))
            span = high - low

            # Çan eğrisi + hafif sağ kuyruk (Ex-Gaussian) parametreleri
            mu = low + span * 0.34
            sigma = span * 0.13
            tau = span * 0.16

            interval = self.sample_ex_gaussian(
                mu=mu,
                sigma=sigma,
                tau=tau,
                min_val=low,
                max_val=high,
            )
            logger.debug("F basım aralığı üretildi: %.3f s (aralık: %.2f-%.2f s)", interval, low, high)
            return interval
        except Exception as exc:
            logger.error("get_f_press_interval_sec hatası: %s", exc)
            return 2.15

    def get_map_open_reading_delay_sec(self) -> float:
        """
        Harita Açılış / Okuma Beklemesi:
        'M' tuşuna basıp haritayı açtıktan sonra 400 ms - 850 ms (0.40s - 0.85s)
        'haritaya bakma / hedefi bulma' süresi üretir.
        """
        try:
            fatigue = self.get_fatigue_multiplier()
            mu = 0.495 * min(1.04, fatigue)
            sigma = 0.048
            tau = 0.095 * min(1.05, fatigue)
            delay = self.sample_ex_gaussian(
                mu=mu,
                sigma=sigma,
                tau=tau,
                min_val=0.400,
                max_val=0.850,
            )
            logger.debug("Harita açılış okuma gecikmesi üretildi: %.1f ms (%.4f s)", delay * 1000.0, delay)
            return delay
        except Exception as exc:
            logger.error("get_map_open_reading_delay_sec hatası: %s", exc)
            return 0.580

    def get_map_close_delay_sec(self) -> float:
        """
        Harita Kapanış Beklemesi:
        Harita üzerinde hedefe tıkladıktan sonra 250 ms - 500 ms (0.25s - 0.50s)
        haritayı kapatma gecikmesi üretir.
        """
        try:
            fatigue = self.get_fatigue_multiplier()
            mu = 0.305 * min(1.04, fatigue)
            sigma = 0.030
            tau = 0.052 * min(1.05, fatigue)
            delay = self.sample_ex_gaussian(
                mu=mu,
                sigma=sigma,
                tau=tau,
                min_val=0.250,
                max_val=0.500,
            )
            logger.debug("Harita kapanış gecikmesi üretildi: %.1f ms (%.4f s)", delay * 1000.0, delay)
            return delay
        except Exception as exc:
            logger.error("get_map_close_delay_sec hatası: %s", exc)
            return 0.355

    def get_reclick_interval_sec(self, base_sec: float = 10.0) -> float:
        """
        Dinamik Yeniden Tıklama Süresi:
        Sabit 10.0 saniye yerine 8.5 ile 13.2 saniye arasında Ex-Gaussian / Log-Normal
        değişken süre döndürür. `base_sec` 10.0'dan farklı verilirse orantısal ölçeklenir.
        """
        try:
            scale = max(0.1, float(base_sec) / 10.0)
            min_s = 8.5 * scale
            max_s = 13.2 * scale
            mu = 9.45 * scale
            sigma = 0.58 * scale
            tau = 0.95 * scale

            interval = self.sample_ex_gaussian(
                mu=mu,
                sigma=sigma,
                tau=tau,
                min_val=min_s,
                max_val=max_s,
            )
            logger.debug(
                "Dinamik yeniden tıklama süresi üretildi: %.3f s (hedef aralık: %.2f-%.2f s)",
                interval,
                min_s,
                max_s,
            )
            return interval
        except Exception as exc:
            logger.error("get_reclick_interval_sec hatası: %s", exc)
            return 10.25

    def get_key_hold_duration_sec(self, min_s: float = 0.042, max_s: float = 0.095) -> float:
        """
        Fiziksel tuş basılı tutma süresini Log-Normal dağılımla (42ms - 95ms) üretir.
        """
        try:
            low = max(0.015, float(min(min_s, max_s)))
            high = max(low + 0.005, float(max(min_s, max_s)))
            median = low + (high - low) * 0.42
            return self.sample_log_normal(median=median, sigma=0.22, min_val=low, max_val=high)
        except Exception as exc:
            logger.error("get_key_hold_duration_sec hatası: %s", exc)
            return 0.062

    def sample_gaussian_jitter(
        self,
        base_sec: float,
        std_sec: float = 0.005,
        min_sec: Optional[float] = None,
        max_sec: Optional[float] = None,
    ) -> float:
        """
        Gaussian (normal) dağılımla ortalama `base_sec` ve standart sapma `std_sec`
        kullanarak insani jitter üretir. Sınırlar verilmezse [base - 3*std, base + 3*std]
        ve en az 0.001s olarak otomatik sınırlandırır.
        """
        try:
            clean_base = max(0.001, float(base_sec))
            clean_std = max(0.0001, float(std_sec))
            low = float(min_sec) if min_sec is not None else max(0.001, clean_base - 3.0 * clean_std)
            high = float(max_sec) if max_sec is not None else (clean_base + 3.0 * clean_std)
            if low > high:
                low, high = high, low
            with self._lock:
                self._total_samples_generated += 1
                for _ in range(32):
                    candidate = self._rng.gauss(clean_base, clean_std)
                    if low <= candidate <= high:
                        return float(candidate)
                return float(max(low, min(high, clean_base)))
        except Exception as exc:
            logger.error("sample_gaussian_jitter hatası: %s", exc)
            return float(base_sec)

    def get_key_release_guard_sec(self) -> float:
        """
        Tuş bırakıldıktan sonraki koruma gecikmesi:
        Sabit 20ms (0.020s) yerine Gaussian insan jitter'ı (16ms - 28ms, ort ~21ms).
        """
        fatigue = self.get_fatigue_multiplier()
        mu = 0.021 * min(1.05, fatigue)
        sigma = 0.0025
        return self.sample_gaussian_jitter(
            base_sec=mu,
            std_sec=sigma,
            min_sec=0.016,
            max_sec=0.028,
        )

    def get_burst_inter_key_delay_sec(self, trusted: bool = True) -> float:
        """
        Çoklu beceri (burst) basımları arasındaki bekleme süresi:
        Sabit 35ms (trusted) / 60ms (untrusted) yerine Gaussian insan jitter'ı.
        Trusted: 28ms - 45ms (ort ~35ms, std 3.5ms).
        Untrusted: 48ms - 75ms (ort ~60ms, std 5.5ms).
        """
        fatigue = self.get_fatigue_multiplier()
        if trusted:
            mu = 0.035 * min(1.05, fatigue)
            return self.sample_gaussian_jitter(
                base_sec=mu,
                std_sec=0.0035,
                min_sec=0.028,
                max_sec=0.045,
            )
        else:
            mu = 0.060 * min(1.05, fatigue)
            return self.sample_gaussian_jitter(
                base_sec=mu,
                std_sec=0.0055,
                min_sec=0.048,
                max_sec=0.075,
            )

    def align_to_gcd(
        self,
        burst_duration_sec: float,
        gcd_sec: float = 0.12,
        min_gap_sec: float = 0.020,
    ) -> float:
        """
        Global Cooldown (GCD) ile zaman uyumu:
        Beceri kullanım süresinin ardından bir sonraki becerinin GCD bitimine denk
        gelmesi için kalan süreyi hesaplar ve insan parmak jitter'ı ekler.
        `min_gap_sec` minimum dinlenme payını garanti eder.
        """
        try:
            clean_gcd = max(0.05, float(gcd_sec))
            clean_dur = max(0.0, float(burst_duration_sec))
            clean_min = max(0.005, float(min_gap_sec))

            rem = clean_gcd - (clean_dur % clean_gcd) if clean_gcd > 0 else clean_min
            if rem < clean_min:
                rem += clean_gcd

            jittered = self.sample_gaussian_jitter(
                base_sec=rem,
                std_sec=0.004,
                min_sec=clean_min,
                max_sec=rem + 0.015,
            )
            return max(clean_min, jittered)
        except Exception as exc:
            logger.error("align_to_gcd hatası: %s", exc)
            # F-0102: hata dalı kendisi exception fırlatmamalı — float()
            # dönüşümleri de başarısız olabilir (None / sayısal olmayan
            # girdi); her parçayı ayrı ayrı güvenli ayrıştır, aksi hâlde
            # varsayılanlara düş.
            try:
                _gcd = float(gcd_sec)
            except (TypeError, ValueError):
                _gcd = 0.12
            try:
                _dur = float(burst_duration_sec)
            except (TypeError, ValueError):
                _dur = 0.0
            try:
                _min = float(min_gap_sec)
            except (TypeError, ValueError):
                _min = 0.020
            return max(_min, _gcd - _dur)

    def sleep_gaussian_jitter(
        self,
        base_sec: float,
        std_sec: float = 0.005,
        min_sec: Optional[float] = None,
        max_sec: Optional[float] = None,
    ) -> float:
        """Gaussian jitter süresi kadar bekler ve uygulanan süreyi döndürür."""
        delay = self.sample_gaussian_jitter(base_sec, std_sec=std_sec, min_sec=min_sec, max_sec=max_sec)
        try:
            time.sleep(delay)
        except Exception as exc:
            logger.error("sleep_gaussian_jitter hatası: %s", exc)
        return delay

    def sleep_visual_reaction(self) -> float:
        """Görsel tepki gecikmesi kadar bekler ve beklenen süreyi döndürür."""
        delay = self.get_visual_reaction_delay_sec()
        try:
            time.sleep(delay)
        except Exception as exc:
            logger.error("sleep_visual_reaction sırasında hata: %s", exc)
        return delay

    def get_human_timing_ready_flag(self) -> dict:
        # S-1701: kosulsuz "ready: True" degil — motoru gercekten yokla.
        # _probe_human_timing_engine modul sonunda tanimli; cagri aninda cozulur.
        ready, _checks = _probe_human_timing_engine(self)
        return {"module": "human_timing", "ready": bool(ready)}

    def get_human_timing_version_info(self) -> dict:
        # S-1701: surum bilgisi hazirlik iddiasiyla birlikte yalnizca prob gecerse True.
        ready, _checks = _probe_human_timing_engine(self)
        return {"module": "human_timing", "version": __version__, "ready": bool(ready)}

    def get_human_timing_diagnostic_snapshot(self) -> dict:
        """Diagnostic snapshot with real live values (not a constant).

        S-1701: "status: ok" yalnızca prob geçerse; aksi halde degraded.
        `total_samples_generated` is the engine's real sample counter,
        `fatigue_multiplier` the current session-fatigue factor and
        `session_elapsed_sec` the monotonic session age — so a caller can
        tell a fresh engine from one that has sampled thousands of times.
        The counter is captured BEFORE the readiness probe runs: the probe
        itself genuinely samples the engine (11 samples), so reading the
        counter after probing would report the probe's own samples as if
        the caller had produced them. A diagnostic read must not inflate
        the quantity it measures.
        """
        # Sayaç prob ÖNCESİ okunur (prob 11 örnek üretir; bkz. docstring).
        total_samples = self.total_samples_generated
        ready, checks = _probe_human_timing_engine(self)
        return {
            "module": "human_timing",
            "status": "ok" if ready else "degraded",
            "ready": bool(ready),
            "checks": checks,
            "total_samples_generated": total_samples,
            "fatigue_multiplier": round(self.get_fatigue_multiplier(), 4),
            "session_elapsed_sec": round(time.monotonic() - self._session_start_ts, 2),
        }


# ============================================================================
# 2. DİNAMİK MOLA PLANLAYICI (MİKRO VE MAKRO MOLALAR)
# ============================================================================

class DynamicBreakScheduler:
    """
    Saatlerce aralıksız çalışan bot davranışını kırmak için:
    - Makro Mola (Su/Kahve Molası): Her 45 ± 12 dakikada bir (33 - 57 dk),
      6 ± 3 dakika (3 - 9 dk / 180s - 540s) mola planlar.
    - Mikro Duraksama (İnsani Dalgınlık): Her 8 - 15 dakikada bir (480s - 900s),
      3.5s - 9.0s süren kısa insani duraksama üretir.
    """

    def __init__(
        self,
        timing_engine: Optional[HumanTimingEngine] = None,
        macro_interval_min: float = 45.0,
        macro_interval_jitter_min: float = 12.0,
        macro_duration_min: float = 6.0,
        macro_duration_jitter_min: float = 3.0,
        micro_interval_min_range: Tuple[float, float] = (8.0, 15.0),
        micro_duration_sec_range: Tuple[float, float] = (3.5, 9.0),
        enabled: bool = True,
    ) -> None:
        self.timing = timing_engine or HumanTimingEngine()
        self.enabled: bool = bool(enabled)
        self._lock = threading.Lock()

        # Makro mola ayarları (dakika)
        self.macro_interval_min: float = max(5.0, float(macro_interval_min))
        self.macro_interval_jitter_min: float = max(1.0, float(macro_interval_jitter_min))
        self.macro_duration_min: float = max(1.0, float(macro_duration_min))
        self.macro_duration_jitter_min: float = max(0.5, float(macro_duration_jitter_min))

        # Mikro duraksama ayarları
        self.micro_interval_min_range: Tuple[float, float] = (
            max(1.0, float(min(micro_interval_min_range))),
            max(2.0, float(max(micro_interval_min_range))),
        )
        self.micro_duration_sec_range: Tuple[float, float] = (
            max(0.5, float(min(micro_duration_sec_range))),
            max(1.0, float(max(micro_duration_sec_range))),
        )

        now = time.monotonic()
        self._last_macro_ts: float = now
        self._last_micro_ts: float = now
        self._next_macro_interval_sec: float = self._sample_macro_interval_sec()
        self._next_micro_interval_sec: float = self._sample_micro_interval_sec()

        self.total_macro_breaks: int = 0
        self.total_micro_pauses: int = 0
        # F-0278: get_break_scheduler_status'un okuduğu alan gerçekten
        # atanır; mola bitiş anı da tutulur (is_on_break buradan türetilir).
        self._is_on_break: bool = False
        self._break_until_ts: float = 0.0

        logger.info(
            "DynamicBreakScheduler başlatıldı | Sonraki Mikro: %.1f dk | Sonraki Makro: %.1f dk",
            self._next_micro_interval_sec / 60.0,
            self._next_macro_interval_sec / 60.0,
        )

    def _sample_macro_interval_sec(self) -> float:
        """45 ± 12 dakika (33 - 57 dk) aralığında Ex-Gaussian makro mola aralığı üretir."""
        min_sec = max(60.0, (self.macro_interval_min - self.macro_interval_jitter_min) * 60.0)
        max_sec = max(min_sec + 60.0, (self.macro_interval_min + self.macro_interval_jitter_min) * 60.0)
        span = max_sec - min_sec
        return self.timing.sample_ex_gaussian(
            mu=min_sec + span * 0.42,
            sigma=span * 0.14,
            tau=span * 0.14,
            min_val=min_sec,
            max_val=max_sec,
        )

    def _sample_macro_duration_sec(self) -> float:
        """6 ± 3 dakika (180s - 540s) aralığında Ex-Gaussian makro mola süresi üretir."""
        min_sec = max(30.0, (self.macro_duration_min - self.macro_duration_jitter_min) * 60.0)
        max_sec = max(min_sec + 30.0, (self.macro_duration_min + self.macro_duration_jitter_min) * 60.0)
        span = max_sec - min_sec
        return self.timing.sample_ex_gaussian(
            mu=min_sec + span * 0.38,
            sigma=span * 0.13,
            tau=span * 0.16,
            min_val=min_sec,
            max_val=max_sec,
        )

    def _sample_micro_interval_sec(self) -> float:
        """8 - 15 dakika (480s - 900s) aralığında Ex-Gaussian mikro duraksama aralığı üretir."""
        min_sec = self.micro_interval_min_range[0] * 60.0
        max_sec = self.micro_interval_min_range[1] * 60.0
        span = max_sec - min_sec
        return self.timing.sample_ex_gaussian(
            mu=min_sec + span * 0.38,
            sigma=span * 0.14,
            tau=span * 0.16,
            min_val=min_sec,
            max_val=max_sec,
        )

    def _sample_micro_duration_sec(self) -> float:
        """3.5s - 9.0s aralığında Ex-Gaussian mikro duraksama süresi üretir."""
        min_sec, max_sec = self.micro_duration_sec_range
        span = max_sec - min_sec
        return self.timing.sample_ex_gaussian(
            mu=min_sec + span * 0.34,
            sigma=span * 0.13,
            tau=span * 0.18,
            min_val=min_sec,
            max_val=max_sec,
        )

    def should_take_macro_break(self, now: Optional[float] = None) -> Tuple[bool, float]:
        """
        Makro mola zamanının gelip gelmediğini kontrol eder.
        Zamanı geldiyse `(True, mola_süresi_saniye)` döndürür ve bir sonraki makro molayı planlar.
        Zamanı gelmediyse `(False, 0.0)` döndürür.
        `now` verilmezse monotonic saat kullanılır (F-0277: wall-clock
        sıçramaları aralıkları bozmaz).
        """
        try:
            if not self.enabled:
                return False, 0.0

            current_ts = float(now if now is not None else time.monotonic())
            with self._lock:
                elapsed = current_ts - self._last_macro_ts
                if elapsed < self._next_macro_interval_sec:
                    return False, 0.0

                duration_sec = self._sample_macro_duration_sec()
                # F-0276: _last_* mola başlangıcına (current_ts) kurulur.
                # Eskiden current_ts + duration_sec yazılıyordu; bu,
                # get_scheduler_status'taki kalan süre hesabında mola
                # süresini çift sayıyordu (kalan = aralık + mola).
                self._last_macro_ts = current_ts
                self._last_micro_ts = current_ts
                self._is_on_break = True
                self._break_until_ts = current_ts + duration_sec
                self._next_macro_interval_sec = self._sample_macro_interval_sec()
                self._next_micro_interval_sec = self._sample_micro_interval_sec()
                self.total_macro_breaks += 1

                logger.info(
                    "[Makro Mola] Su/Kahve molası tetiklendi (#%d) | Süre: %.1f dk (%.1f s) | Sonraki: %.1f dk sonra",
                    self.total_macro_breaks,
                    duration_sec / 60.0,
                    duration_sec,
                    self._next_macro_interval_sec / 60.0,
                )
                return True, float(duration_sec)
        except Exception as exc:
            logger.error("should_take_macro_break kontrol hatası: %s", exc)
            return False, 0.0

    def should_take_micro_pause(self, now: Optional[float] = None) -> Tuple[bool, float]:
        """
        Mikro duraksama (insani dalgınlık) zamanının gelip gelmediğini kontrol eder.
        Zamanı geldiyse `(True, duraksama_süresi_saniye)` döndürür ve bir sonraki mikro duraksamayı planlar.
        Zamanı gelmediyse `(False, 0.0)` döndürür.
        """
        try:
            if not self.enabled:
                return False, 0.0

            current_ts = float(now if now is not None else time.monotonic())
            with self._lock:
                elapsed = current_ts - self._last_micro_ts
                if elapsed < self._next_micro_interval_sec:
                    return False, 0.0

                duration_sec = self._sample_micro_duration_sec()
                # F-0276: mikro duraksamada da _last_micro_ts başlangıca
                # kurulur (current_ts + duration_sec çift sayıyordu).
                self._last_micro_ts = current_ts
                self._next_micro_interval_sec = self._sample_micro_interval_sec()
                self.total_micro_pauses += 1

                logger.info(
                    "[Mikro Duraksama] İnsani dalgınlık molası (#%d) | Süre: %.2f s | Sonraki: %.1f dk sonra",
                    self.total_micro_pauses,
                    duration_sec,
                    self._next_micro_interval_sec / 60.0,
                )
                return True, float(duration_sec)
        except Exception as exc:
            logger.error("should_take_micro_pause kontrol hatası: %s", exc)
            return False, 0.0

    def get_scheduler_status(self, now: Optional[float] = None) -> Dict[str, float]:
        """Mevcut mola zamanlayıcılarının kalan sürelerini ve istatistiklerini döndürür."""
        current_ts = float(now if now is not None else time.monotonic())
        with self._lock:
            rem_macro = max(0.0, self._next_macro_interval_sec - (current_ts - self._last_macro_ts))
            rem_micro = max(0.0, self._next_micro_interval_sec - (current_ts - self._last_micro_ts))
            return {
                "remaining_macro_sec": rem_macro,
                "remaining_micro_sec": rem_micro,
                "total_macro_breaks": float(self.total_macro_breaks),
                "total_micro_pauses": float(self.total_micro_pauses),
            }

    def get_break_scheduler_status(self, now: Optional[float] = None) -> dict:
        """Mola zamanlayıcısının durumunu döndürür.

        F-0278: ``_is_on_break`` artık __init__'te atanır ve makro mola
        tetiklendiğinde güncellenir; burada mola bitiş anına göre
        yeniden hesaplanır. ``now`` verilmezse monotonic saat kullanılır.
        """
        try:
            current_ts = float(now if now is not None else time.monotonic())
            with self._lock:
                on_break = current_ts < float(getattr(self, "_break_until_ts", 0.0))
                self._is_on_break = bool(on_break)
                return {
                    "enabled": bool(getattr(self, "enabled", True)),
                    "is_on_break": bool(on_break),
                }
        except Exception:
            return {"enabled": True, "is_on_break": False}

    def get_human_timing_health_summary(self) -> dict:
        """Health snapshot with real scheduler state (not a constant).

        S-1701: `human_timing_ready` bağlı motorun gerçek prob sonucunu
        yansıtır (yalnızca `enabled` bayrağı değil); the rest is live
        break/pause state, so a caller can tell a disabled, idle or
        on-break scheduler apart.
        """
        try:
            # S-1701: bağlı motoru gerçekten yokla.
            engine = getattr(self, "timing", None) or get_human_timing_engine()
            probe_ready, _checks = _probe_human_timing_engine(engine)
            current_ts = time.monotonic()
            with self._lock:
                return {
                    "human_timing_ready": bool(probe_ready and self.enabled),
                    "enabled": bool(self.enabled),
                    "is_on_break": bool(current_ts < self._break_until_ts),
                    "total_macro_breaks": self.total_macro_breaks,
                    "total_micro_pauses": self.total_micro_pauses,
                    "remaining_macro_sec": round(
                        max(0.0, self._next_macro_interval_sec - (current_ts - self._last_macro_ts)), 2
                    ),
                    "remaining_micro_sec": round(
                        max(0.0, self._next_micro_interval_sec - (current_ts - self._last_micro_ts)), 2
                    ),
                }
        except Exception:
            ready = False
        return {"human_timing_ready": bool(ready)}


# ============================================================================
# 3. ORGANİK ANTİ-AFK MOTORU (MOLA SIRASINDA İNSANİ EYLEMLER)
# ============================================================================

# S-1103: ekran çözünürlüğü çözümleme yardımcıları. Kamera çevirmesi
# ekran merkezinden başlar; çözünürlük okunamadan 1920x1080 varsaymak
# gerçek çözünürlük farklıysa çevirmeyi yanlış noktadan başlatır.
# Bu yardımcılar çözünürlüğü doğrulanmış kaynaklardan sırayla okur;
# hiçbir kaynak geçerli değer vermezse son çare 1920x1080 döner ama
# bunu "unknown-fallback" diye etiketler ve çağıran yüksek sesle
# uyarır — sessiz uydurma yok. (Sıkı "bilinmiyorsa atla" burada
# uygulanamaz: repo testi test_anti_afk_reentrancy_returns_busy
# çözünürlük kaynağı olmayan sahte bir ctrl ile kamera çevirmesinin
# fare yöntemine ulaşmasını bekler; üretimde Windows'ta user32
# sorgusu zaten gerçek çözünürlüğü verir, geri dönüşe düşülmez.)

_RESOLUTION_MIN_WIDTH = 640
_RESOLUTION_MAX_WIDTH = 7680
_RESOLUTION_MIN_HEIGHT = 480
_RESOLUTION_MAX_HEIGHT = 4320


def _is_valid_resolution(width: Any, height: Any) -> bool:
    """Çözünürlük değerlerinin makul bir (genişlik, yükseklik) çifti olup olmadığını doğrular."""
    if isinstance(width, bool) or isinstance(height, bool):
        return False
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    return (
        _RESOLUTION_MIN_WIDTH <= width <= _RESOLUTION_MAX_WIDTH
        and _RESOLUTION_MIN_HEIGHT <= height <= _RESOLUTION_MAX_HEIGHT
    )


def _resolve_screen_resolution(input_ctrl: Any) -> Tuple[int, int, str]:
    """Ekran çözünürlüğünü doğrulanmış kaynaklardan sırayla okur.

    Sıra: (1) input_ctrl.get_screen_resolution(), (2) input_ctrl
    screen_width/screen_height öznitelikleri, (3) Windows'ta doğrudan
    user32 GetSystemMetrics sorgusu (core/input_driver.py'deki
    get_screen_resolution ile aynı yol — ctrl'de yöntem yoksa bile
    gerçek çözünürlük okunur). Hiçbiri geçerli bir değer vermezse
    son çare (1920, 1080, "unknown-fallback") döner.
    """
    getter = getattr(input_ctrl, "get_screen_resolution", None)
    if callable(getter):
        try:
            resolved = getter()
            width, height = resolved[0], resolved[1]
            if _is_valid_resolution(width, height):
                return int(width), int(height), "input_ctrl"
            logger.debug("Geçersiz çözünürlük değeri yok sayıldı: %r", resolved)
        except Exception as exc:
            logger.debug("Çözünürlük okunamadı (input_ctrl): %s", exc)

    try:
        width = getattr(input_ctrl, "screen_width", None)
        height = getattr(input_ctrl, "screen_height", None)
        if _is_valid_resolution(width, height):
            return int(width), int(height), "input_ctrl_attrs"
    except Exception as exc:
        logger.debug("Çözünürlük okunamadı (öznitelik): %s", exc)

    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception as dpi_exc:
            logger.debug("DPI farkındalığı ayarlanırken bilgi: %s", dpi_exc)
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if _is_valid_resolution(width, height):
            return width, height, "os"
    except Exception as exc:
        logger.debug("Çözünürlük okunamadı (OS sorgusu): %s", exc)

    return 1920, 1080, "unknown-fallback"


class OrganicAntiAFK:
    """
    Uzun beklemelerde veya makro molalarda hep aynı saniyede ileri-geri yapmak yerine,
    düzensiz aralıklarla (140s - 260s Ex-Gaussian) aşağıdaki insani eylemlerden birini seçip uygular:
      1. Kamera açısını hafifçe çevirme (fare sağ tık basılı hafif WindMouse kaydırma)
      2. Zıplama ('space')
      3. Rastgele yöne ('w', 'a', 's', 'd') birkaç adım yürüme
    """

    ACTION_CAMERA_PAN: str = "camera_windmouse_pan"
    ACTION_JUMP: str = "organic_jump"
    ACTION_MICRO_WALK: str = "random_micro_walk"

    def __init__(
        self,
        timing_engine: Optional[HumanTimingEngine] = None,
        min_interval_sec: float = 140.0,
        max_interval_sec: float = 260.0,
    ) -> None:
        self.timing = timing_engine or HumanTimingEngine()
        self.min_interval_sec: float = max(30.0, float(min(min_interval_sec, max_interval_sec)))
        self.max_interval_sec: float = max(self.min_interval_sec + 10.0, float(max(min_interval_sec, max_interval_sec)))
        self._rng = random.Random()
        self._lock = threading.Lock()

        self._last_action_ts: float = time.monotonic()
        self._next_interval_sec: float = self.get_next_interval_sec()
        self._last_action_name: str = ""
        self.total_actions_executed: int = 0
        # F-0279: reentrancy koruması — iki eşzamanlı çağrı aynı eylemi
        # iki kez çalıştırmasın.
        self._executing: bool = False

    def get_next_interval_sec(self) -> float:
        """
        140s - 260s arasında Ex-Gaussian dağılımlı düzensiz Anti-AFK bekleme süresi üretir.
        """
        try:
            span = self.max_interval_sec - self.min_interval_sec
            interval = self.timing.sample_ex_gaussian(
                mu=self.min_interval_sec + span * 0.36,
                sigma=span * 0.13,
                tau=span * 0.17,
                min_val=self.min_interval_sec,
                max_val=self.max_interval_sec,
            )
            logger.debug("Sonraki Organik Anti-AFK aralığı: %.2f s", interval)
            return interval
        except Exception as exc:
            logger.error("get_next_interval_sec hatası: %s", exc)
            return 190.0

    def should_trigger_anti_afk(self, now: Optional[float] = None) -> Tuple[bool, float]:
        """
        Anti-AFK eylemi zamanının gelip gelmediğini kontrol eder.
        Dönüş: `(tetiklenmeli_mi, kalan_veya_yeni_aralık_sn)`
        """
        current_ts = float(now if now is not None else time.monotonic())
        with self._lock:
            elapsed = current_ts - self._last_action_ts
            if elapsed >= self._next_interval_sec:
                return True, 0.0
            return False, max(0.0, self._next_interval_sec - elapsed)

    def _generate_windmouse_points(
        self,
        start_x: int,
        start_y: int,
        dest_x: int,
        dest_y: int,
        gravity: float = 9.0,
        wind: float = 3.5,
        min_wait: float = 0.004,
        max_wait: float = 0.012,
        max_step: float = 12.0,
        target_area: float = 6.0,
    ) -> List[Tuple[int, int, float]]:
        """
        Gerçek WindMouse fiziksel imleç yörüngesi (yerçekimi + rüzgar türbülansı) üretir.
        Her adım için `(x, y, bekleme_süresi)` listesi döndürür.
        """
        points: List[Tuple[int, int, float]] = []
        try:
            cur_x, cur_y = float(start_x), float(start_y)
            vel_x, vel_y = 0.0, 0.0
            wind_x, wind_y = 0.0, 0.0
            sqrt2 = math.sqrt(2.0)
            sqrt3 = math.sqrt(3.0)
            sqrt5 = math.sqrt(5.0)

            for _ in range(45):
                dist = math.hypot(dest_x - cur_x, dest_y - cur_y)
                if dist < 1.5:
                    break

                wind_mag = min(wind, dist)
                if dist >= target_area:
                    wind_x = wind_x / sqrt3 + (2.0 * self._rng.random() - 1.0) * wind_mag / sqrt5
                    wind_y = wind_y / sqrt3 + (2.0 * self._rng.random() - 1.0) * wind_mag / sqrt5
                else:
                    wind_x /= sqrt2
                    wind_y /= sqrt2
                    if max_step < 3.0:
                        max_step = self._rng.uniform(3.0, 6.0)
                    else:
                        max_step /= sqrt5

                vel_x += wind_x + gravity * (dest_x - cur_x) / dist
                vel_y += wind_y + gravity * (dest_y - cur_y) / dist

                vel_mag = math.hypot(vel_x, vel_y)
                if vel_mag > max_step:
                    random_dist = max_step / 2.0 + self._rng.random() * (max_step / 2.0)
                    vel_x = (vel_x / vel_mag) * random_dist
                    vel_y = (vel_y / vel_mag) * random_dist

                cur_x += vel_x
                cur_y += vel_y
                step_ratio = min(1.0, math.hypot(vel_x, vel_y) / max(1.0, max_step))
                wait = min_wait + (max_wait - min_wait) * (1.0 - step_ratio)
                points.append((int(round(cur_x)), int(round(cur_y)), wait))

            points.append((int(dest_x), int(dest_y), min_wait))
        except Exception as exc:
            logger.error("WindMouse yörünge üretim hatası: %s", exc)
            points.append((int(dest_x), int(dest_y), 0.008))
        return points

    def _execute_camera_pan(self, input_ctrl: Any) -> Dict[str, Any]:
        """
        Fare sağ tık basılı tutarak WindMouse eğrisiyle kamerayı hafifçe (35-115 px) çevirir.
        """
        test_mode = bool(getattr(input_ctrl, "test_mode", False))
        # S-1103: çözünürlüğü doğrulanmış kaynaklardan oku (ctrl yöntemi
        # -> öznitelikler -> doğrudan OS sorgusu). Hiçbir kaynak
        # okunamazsa son çare 1920x1080 kullanılır ama bu artık sessiz
        # değildir: warning loglanır ve sonuç resolution_source ile
        # "unknown-fallback" diye etiketlenir.
        width, height, res_source = _resolve_screen_resolution(input_ctrl)
        if res_source == "unknown-fallback":
            logger.warning(
                "[OrganicAntiAFK] Ekran çözünürlüğü okunamadı; varsayılan "
                "1920x1080 merkezi kullanılıyor — gerçek çözünürlük farklıysa "
                "kamera çevirmesi yanlış noktadan başlayabilir"
            )

        start_x, start_y = width // 2, height // 2
        if hasattr(input_ctrl, "get_cursor_position"):
            try:
                cx, cy = input_ctrl.get_cursor_position()
                if 100 < cx < width - 100 and 100 < cy < height - 100:
                    start_x, start_y = cx, cy
            except Exception as exc:
                logger.debug("İmleç konumu okunamadı: %s", exc)

        # Yatayda ±(35..115) piksel, dikeyde ±(6..28) piksel hafif kamera dönüşü
        dir_x = -1 if self._rng.random() < 0.5 else 1
        delta_x = dir_x * int(round(self.timing.sample_ex_gaussian(58.0, 14.0, 18.0, 35.0, 115.0)))
        delta_y = int(round(self._rng.gauss(0.0, 9.0)))
        delta_y = max(-28, min(28, delta_y))

        dest_x = max(120, min(width - 120, start_x + delta_x))
        dest_y = max(120, min(height - 120, start_y + delta_y))

        trajectory = self._generate_windmouse_points(start_x, start_y, dest_x, dest_y)
        pre_hold = self.timing.get_key_hold_duration_sec(0.035, 0.075)
        post_hold = self.timing.get_key_hold_duration_sec(0.030, 0.065)

        if test_mode:
            logger.info(
                "[TEST MODE] [OrganicAntiAFK] Kamera WindMouse Çevirme: (%d, %d) -> (%d, %d) | Adım: %d",
                start_x,
                start_y,
                dest_x,
                dest_y,
                len(trajectory),
            )
            return {
                "action": self.ACTION_CAMERA_PAN,
                "start": (start_x, start_y),
                "dest": (dest_x, dest_y),
                "steps": len(trajectory),
                "duration_sec": round(pre_hold + post_hold + sum(p[2] for p in trajectory), 4),
                "success": True,
                "executed": True,
                "resolution": (width, height),
                "resolution_source": res_source,
            }

        # F-0104: girdi metodu yokken sessizce "başarı" döndürme — hangi
        # yöntemin eksik olduğunu söyleyen bir skipped sonucu döndür.
        has_hw = hasattr(input_ctrl, "_hw_mouse_down") and hasattr(input_ctrl, "_hw_mouse_up")
        has_smooth = hasattr(input_ctrl, "smooth_move_to")
        if not has_hw and not has_smooth:
            logger.warning("[OrganicAntiAFK] Kamera çevirme atlandı: input_ctrl'te fare yöntemi yok")
            return {
                "action": self.ACTION_CAMERA_PAN,
                "start": (start_x, start_y),
                "dest": (dest_x, dest_y),
                "steps": 0,
                "duration_sec": 0.0,
                "success": False,
                "executed": False,
                "resolution": (width, height),
                "resolution_source": res_source,
                "error": "no mouse input method on input_ctrl (need _hw_mouse_down/_hw_mouse_up or smooth_move_to)",
            }
        if has_hw and not hasattr(input_ctrl, "_hw_move_to"):
            logger.warning("[OrganicAntiAFK] Kamera çevirme atlandı: _hw_move_to yok")
            return {
                "action": self.ACTION_CAMERA_PAN,
                "start": (start_x, start_y),
                "dest": (dest_x, dest_y),
                "steps": 0,
                "duration_sec": 0.0,
                "success": False,
                "executed": False,
                "resolution": (width, height),
                "resolution_source": res_source,
                "error": "no _hw_move_to on input_ctrl",
            }

        # P0-5: has_smooth dalında dönüş başarıyı kanıtlamaz (konum
        # tuple'ı döner); pan_ok yalnızca _hw dalında izlenir.
        pan_ok = True
        try:
            if has_hw:
                # P0-5: her adımın dönüşü denetlenir — reddedilen basış/
                # taşıma/bırakma artık sessizce "başarı" sayılmaz. None =
                # eski sürücü (dönüşsüz kabul), yalnızca açık False
                # başarısızlıktır.
                if input_ctrl._hw_mouse_down("right") is False:
                    pan_ok = False
                try:
                    time.sleep(pre_hold)
                    for px, py, wait_s in trajectory:
                        if hasattr(input_ctrl, "_hw_move_to"):
                            if input_ctrl._hw_move_to(px, py) is False:
                                pan_ok = False
                        time.sleep(wait_s)
                    time.sleep(post_hold)
                finally:
                    if input_ctrl._hw_mouse_up("right") is False:
                        pan_ok = False
            elif has_smooth:
                input_ctrl.smooth_move_to(dest_x, dest_y)
        except Exception as exc:
            logger.error("Kamera çevirme eylemi sırasında hata: %s", exc)
            return {
                "action": self.ACTION_CAMERA_PAN,
                "start": (start_x, start_y),
                "dest": (dest_x, dest_y),
                "steps": len(trajectory),
                "duration_sec": round(pre_hold + post_hold + sum(p[2] for p in trajectory), 4),
                "success": False,
                "executed": False,
                "resolution": (width, height),
                "resolution_source": res_source,
                "error": str(exc),
            }

        return {
            "action": self.ACTION_CAMERA_PAN,
            "start": (start_x, start_y),
            "dest": (dest_x, dest_y),
            "steps": len(trajectory),
            "duration_sec": round(pre_hold + post_hold + sum(p[2] for p in trajectory), 4),
            "success": bool(pan_ok),
            "executed": True,
            "resolution": (width, height),
            "resolution_source": res_source,
        }

    def _execute_organic_jump(self, input_ctrl: Any) -> Dict[str, Any]:
        """İnsani basılı tutma süresiyle zıplama ('space') eylemi gerçekleştirir."""
        hold_sec = self.timing.get_key_hold_duration_sec(0.055, 0.115)
        test_mode = bool(getattr(input_ctrl, "test_mode", False))

        if test_mode:
            logger.info("[TEST MODE] [OrganicAntiAFK] Zıplama ('space') | Tutma: %.1f ms", hold_sec * 1000.0)
            return {"action": self.ACTION_JUMP, "key": "space", "duration_sec": round(hold_sec, 4),
                    "success": True, "executed": True}

        # F-0104: press_key yoksa sessizce başarı döndürme.
        if not hasattr(input_ctrl, "press_key"):
            logger.warning("[OrganicAntiAFK] Zıplama atlandı: input_ctrl'te press_key yok")
            return {"action": self.ACTION_JUMP, "key": "space", "duration_sec": round(hold_sec, 4),
                    "success": False, "executed": False,
                    "error": "no press_key method on input_ctrl"}
        try:
            actual = input_ctrl.press_key("space", min_hold=hold_sec * 0.9, max_hold=hold_sec * 1.1)
            if actual is not None:
                hold_sec = float(actual)
        except Exception as exc:
            logger.error("Organik zıplama eylemi hatası: %s", exc)
            return {"action": self.ACTION_JUMP, "key": "space", "duration_sec": round(hold_sec, 4),
                    "success": False, "executed": False, "error": str(exc)}

        return {"action": self.ACTION_JUMP, "key": "space", "duration_sec": round(hold_sec, 4),
                "success": True, "executed": True}

    def _execute_micro_walk(self, input_ctrl: Any) -> Dict[str, Any]:
        """Rastgele bir yöne ('w', 'a', 's', 'd') birkaç adım (180ms - 460ms) yürür."""
        direction_key = self._rng.choice(["w", "a", "s", "d"])
        walk_sec = self.timing.sample_ex_gaussian(
            mu=0.245,
            sigma=0.035,
            tau=0.055,
            min_val=0.180,
            max_val=0.460,
        )
        test_mode = bool(getattr(input_ctrl, "test_mode", False))

        if test_mode:
            logger.info(
                "[TEST MODE] [OrganicAntiAFK] Rastgele Yöne Yürüme ('%s') | Süre: %.1f ms",
                direction_key,
                walk_sec * 1000.0,
            )
            return {
                "action": self.ACTION_MICRO_WALK,
                "key": direction_key,
                "duration_sec": round(walk_sec, 4),
                "success": True,
                "executed": True,
            }

        # F-0104: press_key yoksa sessizce başarı döndürme.
        if not hasattr(input_ctrl, "press_key"):
            logger.warning("[OrganicAntiAFK] Mikro yürüme atlandı: input_ctrl'te press_key yok")
            return {"action": self.ACTION_MICRO_WALK, "key": direction_key,
                    "duration_sec": round(walk_sec, 4),
                    "success": False, "executed": False,
                    "error": "no press_key method on input_ctrl"}
        try:
            actual = input_ctrl.press_key(direction_key, min_hold=walk_sec * 0.95, max_hold=walk_sec * 1.05)
            if actual is not None:
                walk_sec = float(actual)
        except Exception as exc:
            logger.error("Organik mikro yürüme eylemi hatası: %s", exc)
            return {"action": self.ACTION_MICRO_WALK, "key": direction_key,
                    "duration_sec": round(walk_sec, 4),
                    "success": False, "executed": False, "error": str(exc)}

        return {
            "action": self.ACTION_MICRO_WALK,
            "key": direction_key,
            "duration_sec": round(walk_sec, 4),
            "success": True,
            "executed": True,
        }

    def execute_organic_anti_afk(self, input_ctrl: Any) -> Dict[str, Any]:
        """
        Kamera çevirme (WindMouse sağ tık), zıplama veya rastgele yöne birkaç adım yürüme
        eylemlerinden birini ağırlıklı olasılıkla (ve ardışık tekrarı önleyerek) seçip uygular.
        Bir sonraki Anti-AFK zamanlayıcısını 140s - 260s Ex-Gaussian dağılımla günceller.
        """
        # F-0279: reentrancy koruması — zaten bir eylem sürüyorsa ikinci
        # çağrı "busy" döner, eylemi tekrar çalıştırmaz.
        with self._lock:
            if self._executing:
                remaining = max(0.0, self._next_interval_sec - (time.monotonic() - self._last_action_ts))
                return {
                    "action": "busy",
                    "success": False,
                    "error": "anti-afk action already executing",
                    "next_interval_sec": round(remaining, 2),
                }
            self._executing = True
        try:
            with self._lock:
                # Ağırlıklı seçim: %45 kamera çevirme, %35 mikro yürüme, %20 zıplama
                candidates = [
                    (self.ACTION_CAMERA_PAN, 0.45),
                    (self.ACTION_MICRO_WALK, 0.35),
                    (self.ACTION_JUMP, 0.20),
                ]
                # Hep aynı hareketi ardışık yapmamak için son eylemin ağırlığını düşür
                adjusted: List[Tuple[str, float]] = []
                for name, weight in candidates:
                    if name == self._last_action_name:
                        adjusted.append((name, weight * 0.35))
                    else:
                        adjusted.append((name, weight))

                total_w = sum(w for _, w in adjusted)
                pick = self._rng.uniform(0.0, total_w)
                chosen_action = self.ACTION_CAMERA_PAN
                acc = 0.0
                for name, weight in adjusted:
                    acc += weight
                    if pick <= acc:
                        chosen_action = name
                        break

                self._last_action_name = chosen_action

            if chosen_action == self.ACTION_CAMERA_PAN:
                result = self._execute_camera_pan(input_ctrl)
            elif chosen_action == self.ACTION_JUMP:
                result = self._execute_organic_jump(input_ctrl)
            else:
                result = self._execute_micro_walk(input_ctrl)

            # F-0104 devamı: eylem uygulanamadıysa (girdi metodu yok /
            # hata) zamanlayıcıyı tam aralıkla ilerletme — 30 sn sonra
            # tekrar dene ve bunu uygulanmış sayma.
            if not result.get("success", True):
                with self._lock:
                    self._last_action_ts = time.monotonic()
                    self._next_interval_sec = 30.0
                result["next_interval_sec"] = 30.0
                result["action_index"] = self.total_actions_executed
                logger.warning(
                    "[OrganicAntiAFK] Eylem uygulanamadı: %s | 30.0 s sonra tekrar denenecek",
                    result.get("error", result.get("action")),
                )
                return result

            with self._lock:
                self._last_action_ts = time.monotonic()
                self._next_interval_sec = self.get_next_interval_sec()
                self.total_actions_executed += 1
                next_iv = self._next_interval_sec
                action_count = self.total_actions_executed

            result["next_interval_sec"] = round(next_iv, 2)
            result["action_index"] = action_count
            result["success"] = True
            logger.info(
                "[OrganicAntiAFK] Eylem #%d uygulandı: %s | Sonraki Anti-AFK: %.1f s sonra",
                action_count,
                result.get("action"),
                next_iv,
            )
            return result
        except Exception as exc:
            logger.error("execute_organic_anti_afk genel hatası: %s", exc)
            # F-0279: hata dalında da zamanlayıcıyı güncelle — aksi hâlde
            # should_trigger_anti_afk her çağrıda True döner ve hata
            # döngüye girer. 180 sn geri çekilme (backoff).
            with self._lock:
                self._last_action_ts = time.monotonic()
                self._next_interval_sec = 180.0
            return {
                "action": "error",
                "success": False,
                "error": str(exc),
                "next_interval_sec": 180.0,
            }
        finally:
            with self._lock:
                self._executing = False


# ============================================================================
# 3b. İNSAN ZAMANLAMA KOORDİNATÖRÜ (F-0103)
# ============================================================================

class HumanTimingCoordinator:
    """
    F-0103: DynamicBreakScheduler ve OrganicAntiAFK üretimde hiç
    çalıştırılmıyordu — ikisi de yalnızca inşa ediliyor (`main.py`,
    `core/state_machine.py`), tick/should_* metotları hiçbir yerden
    çağrılmıyordu.

    Bu sınıf ikisini tek bir `tick()` çağrısında birleştirir: üretim
    döngüsü (BotWorker ana döngüsü) her yinelemede `tick(now, input_ctrl)`
    çağırır; mola zamanı geldiyse mola süresini, Anti-AFK zamanı geldiyse
    eylem sonucunu döndürür. `tick()` asla kendisi uyumaz — molayı
    uygulamak (beklemek) çağıranın işidir.

    NOT (F-0103 kablolama kancası — P1-10 doğrulaması): `tick()`'in canlı
    çağrı yeri `core/state_machine.py::_perform_anti_afk_step`'tir
    (`_handle_break_step` ← ana döngü, BREAK sırasında 25-45 sn'de bir);
    o adım `run_human_timing_tick(self.input_driver, coordinator=...)`
    üzerinden çağırır. `coordinator` verilmezse tekil (global)
    koordinatör kullanılır. Mola sinyalleri (`macro_break` /
    `micro_pause`) bu adımda yalnızca gözlenir/loglanır — mola
    zamanlamasının kendisi `_should_enter_break` / `_handle_break_step`
    içinde ayar (`break_every_min` / `break_lasts_min`) ile yürür;
    `DynamicBreakScheduler`'ın 45±12 dk'lık kendi takviminin devreye
    girmesi kullanıcı ayarını ezeceği için bilinçli olarak
    bağlanmamıştır (P1-10: koordinatör kararı bekliyor).
    """

    def __init__(
        self,
        timing_engine: Optional[HumanTimingEngine] = None,
        break_scheduler: Optional[DynamicBreakScheduler] = None,
        anti_afk: Optional[OrganicAntiAFK] = None,
    ) -> None:
        self.timing = timing_engine or HumanTimingEngine()
        self.break_scheduler = break_scheduler or DynamicBreakScheduler(timing_engine=self.timing)
        self.anti_afk = anti_afk or OrganicAntiAFK(timing_engine=self.timing)

    def tick(self, now: Optional[float] = None, input_ctrl: Any = None) -> Dict[str, Any]:
        """Mola + Anti-AFK durumunu tek çağrıda yoklar.

        `now` verilmezse monotonic saat kullanılır. Dönen sözlük:
        `macro_break` / `macro_break_duration_sec` (mola geldiyse True),
        `micro_pause` / `micro_pause_duration_sec`,
        `anti_afk` (eylem sonucu, bekliyorsa `{"action": "waiting", ...}`,
        `input_ctrl` yoksa `{"action": "pending", "success": False, ...}`).
        """
        current_ts = float(now if now is not None else time.monotonic())
        out: Dict[str, Any] = {
            "macro_break": False,
            "macro_break_duration_sec": 0.0,
            "micro_pause": False,
            "micro_pause_duration_sec": 0.0,
            "anti_afk": None,
        }
        macro_due, macro_dur = self.break_scheduler.should_take_macro_break(now=current_ts)
        if macro_due:
            out["macro_break"] = True
            out["macro_break_duration_sec"] = float(macro_dur)
        else:
            micro_due, micro_dur = self.break_scheduler.should_take_micro_pause(now=current_ts)
            if micro_due:
                out["micro_pause"] = True
                out["micro_pause_duration_sec"] = float(micro_dur)

        due, remaining = self.anti_afk.should_trigger_anti_afk(now=current_ts)
        if not due:
            out["anti_afk"] = {"action": "waiting", "success": True, "executed": False,
                               "next_interval_sec": round(float(remaining), 2)}
        elif input_ctrl is None:
            # Sessizce başarı döndürme (F-0104): girdi yoksa beklemede bırak.
            out["anti_afk"] = {"action": "pending", "success": False, "executed": False,
                               "error": "no input_ctrl supplied to tick()",
                               "next_interval_sec": 0.0}
        else:
            out["anti_afk"] = self.anti_afk.execute_organic_anti_afk(input_ctrl)
        return out


# ============================================================================
# İSTATİSTİKSEL DAĞILIM DOĞRULAMA TEST ARACI (1000 ÖRNEKLEM)
# ============================================================================

def run_statistical_distribution_verification(n_samples: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    1000 örneklemli istatistiksel dağılım testini çalıştırarak tüm sürelerin
    (180-380ms, 400-850ms, 250-500ms, 8.5-13.2s, 1.5-3.0s, 140-260s)
    sınırlarına ve Ex-Gaussian sağa çarpık (pozitif skewness) dağılıma uyduğunu doğrular.
    """
    engine = HumanTimingEngine(seed=42, fatigue_enabled=False)
    anti_afk = OrganicAntiAFK(timing_engine=engine)

    generators = {
        "visual_reaction_sec (0.18-0.38s)": (
            engine.get_visual_reaction_delay_sec,
            0.180,
            0.380,
        ),
        "f_press_interval_sec (1.50-3.00s)": (
            engine.get_f_press_interval_sec,
            1.500,
            3.000,
        ),
        "map_open_reading_sec (0.40-0.85s)": (
            engine.get_map_open_reading_delay_sec,
            0.400,
            0.850,
        ),
        "map_close_delay_sec (0.25-0.50s)": (
            engine.get_map_close_delay_sec,
            0.250,
            0.500,
        ),
        "reclick_interval_sec (8.50-13.20s)": (
            engine.get_reclick_interval_sec,
            8.500,
            13.200,
        ),
        "anti_afk_interval_sec (140.0-260.0s)": (
            anti_afk.get_next_interval_sec,
            140.0,
            260.0,
        ),
        "key_release_guard_sec (0.016-0.028s)": (
            engine.get_key_release_guard_sec,
            0.016,
            0.028,
        ),
        "burst_inter_key_trusted (0.028-0.045s)": (
            lambda: engine.get_burst_inter_key_delay_sec(trusted=True),
            0.028,
            0.045,
        ),
        "burst_inter_key_untrusted (0.048-0.075s)": (
            lambda: engine.get_burst_inter_key_delay_sec(trusted=False),
            0.048,
            0.075,
        ),
    }

    report: Dict[str, Dict[str, float]] = {}
    for label, (fn, low, high) in generators.items():
        samples = [float(fn()) for _ in range(n_samples)]
        samples_sorted = sorted(samples)
        mean_val = sum(samples) / float(n_samples)
        median_val = samples_sorted[n_samples // 2]
        variance = sum((x - mean_val) ** 2 for x in samples) / float(n_samples)
        std_val = math.sqrt(max(1e-12, variance))
        skewness = (
            sum(((x - mean_val) / std_val) ** 3 for x in samples) / float(n_samples)
        )
        min_obs = samples_sorted[0]
        max_obs = samples_sorted[-1]

        in_bounds = (min_obs >= low - 1e-9) and (max_obs <= high + 1e-9)
        if not in_bounds:
            raise AssertionError(
                f"Dağılım sınır ihlali ({label}): gözlenen=[{min_obs:.4f}, {max_obs:.4f}], beklenen=[{low:.4f}, {high:.4f}]"
            )

        report[label] = {
            "count": float(n_samples),
            "min": round(min_obs, 4),
            "max": round(max_obs, 4),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "skewness": round(skewness, 4),
        }
        logger.info(
            "[Dağılım Doğrulandı] %-36s | Min: %.4f | Medyan: %.4f | Ort: %.4f | Max: %.4f | Std: %.4f | Çarpıklık(Skew): %+.3f",
            label,
            min_obs,
            median_val,
            mean_val,
            max_obs,
            std_val,
            skewness,
        )

    return report


# ============================================================================
# 4. GLOBAL TEKİL (SINGLETON) VE MODÜL SEVİYESİ YARDIMCI FONKSİYONLAR
# ============================================================================

_GLOBAL_HUMAN_TIMING_ENGINE: Optional[HumanTimingEngine] = None
_GLOBAL_TIMING_LOCK = threading.Lock()

# F-0444: Modül düzeyi yardımcılar HumanTimingEngine metotlarıyla aynı
# adı taşır (geriye dönük uyumluluk için korunur — core/input_driver.py
# ve testler bunları import eder). Bunlar ayrı bir RNG akışı DEĞİLDİR:
# hepsi aşağıdaki tekil (singleton) motora delege eder, dolayısıyla
# metot çağrılarıyla aynı rastgele akışı paylaşır. İsim gölgelemesini
# azaltmak için `__all__` bu adları açıkça listeler.
__all__ = [
    "HumanTimingEngine",
    "DynamicBreakScheduler",
    "OrganicAntiAFK",
    "HumanTimingCoordinator",
    "get_human_timing_engine",
    "get_human_timing_coordinator",
    "run_human_timing_tick",
    "get_gaussian_jitter",
    "get_key_release_guard_sec",
    "get_burst_inter_key_delay_sec",
    "align_to_gcd",
    "sleep_gaussian_jitter",
    "run_statistical_distribution_verification",
]


def get_human_timing_engine() -> HumanTimingEngine:
    """Global HumanTimingEngine tekil (singleton) örneğini döndürür."""
    global _GLOBAL_HUMAN_TIMING_ENGINE
    if _GLOBAL_HUMAN_TIMING_ENGINE is None:
        with _GLOBAL_TIMING_LOCK:
            if _GLOBAL_HUMAN_TIMING_ENGINE is None:
                _GLOBAL_HUMAN_TIMING_ENGINE = HumanTimingEngine()
    return _GLOBAL_HUMAN_TIMING_ENGINE


# F-0103: HumanTimingCoordinator tekil örneği + tek çağrılık kablolama
# noktası. Canlı çağrı yeri core/state_machine.py::_perform_anti_afk_step
# (R26; _handle_break_step ← ana döngü, BREAK sırasında 25-45 sn'de bir)
# `run_human_timing_tick(input_ctrl, coordinator=...)` çağırır; koordinatör
# mola + Anti-AFK durumunu tek tick'te yoklar. tick() asla kendisi uyumaz —
# makro mola geldiyse çağıran beklemeyi uygular. Tekil koordinatör
# get_human_timing_engine() ile aynı motoru paylaşır (ayrı RNG akışı
# yoktur); state_machine/main.py'de ayrıca OrganicAntiAFK inşa etmeye
# gerek yoktur.
_GLOBAL_HUMAN_TIMING_COORDINATOR: Optional[HumanTimingCoordinator] = None
_GLOBAL_COORDINATOR_LOCK = threading.Lock()


def get_human_timing_coordinator() -> HumanTimingCoordinator:
    """Global HumanTimingCoordinator tekil (singleton) örneğini döndürür."""
    global _GLOBAL_HUMAN_TIMING_COORDINATOR
    if _GLOBAL_HUMAN_TIMING_COORDINATOR is None:
        with _GLOBAL_COORDINATOR_LOCK:
            if _GLOBAL_HUMAN_TIMING_COORDINATOR is None:
                _GLOBAL_HUMAN_TIMING_COORDINATOR = HumanTimingCoordinator(
                    timing_engine=get_human_timing_engine()
                )
    return _GLOBAL_HUMAN_TIMING_COORDINATOR


def run_human_timing_tick(
    input_ctrl: Any,
    now: Optional[float] = None,
    coordinator: Any = None,
) -> Dict[str, Any]:
    """F-0103 kablolama noktası: koordinatörün tick() çağrısı.

    Canlı çağrı yeri `core/state_machine.py::_perform_anti_afk_step`
    (BREAK sırasındaki Anti-AFK adımı) — `coordinator` olarak
    state machine'in kendi alt sistemlerinden kurulu koordinatör
    (`_get_human_timing_coordinator()`) verilir, böylece tekil
    (global) motor yerine botun kendi HumanTimingEngine /
    OrganicAntiAFK örnekleri kullanılır. `coordinator` verilmezse
    tekil koordinatör (`get_human_timing_coordinator()`) kullanılır.

    Dönen sözlük HumanTimingCoordinator.tick() ile aynıdır:
    `macro_break` / `macro_break_duration_sec` (mola geldiyse True —
    beklemek çağıranın işidir), `micro_pause` / `micro_pause_duration_sec`,
    `anti_afk` (eylem sonucu; bekliyorsa `{"action": "waiting", ...}`,
    girdi yoksa `{"action": "pending", ...}`).
    """
    coord = coordinator if coordinator is not None else get_human_timing_coordinator()
    return coord.tick(now=now, input_ctrl=input_ctrl)


def get_gaussian_jitter(
    base_sec: float,
    std_sec: float = 0.005,
    min_sec: Optional[float] = None,
    max_sec: Optional[float] = None,
) -> float:
    """Global HumanTimingEngine üzerinden Gaussian jitter üretir.

    F-0444: HumanTimingEngine.sample_gaussian_jitter ile aynı addaki
    metodu gölgelemez — tekil motora delege eden ince bir sarmalayıcıdır
    (ayrı RNG akışı yoktur).
    """
    return get_human_timing_engine().sample_gaussian_jitter(
        base_sec, std_sec=std_sec, min_sec=min_sec, max_sec=max_sec
    )


def get_key_release_guard_sec() -> float:
    """Global HumanTimingEngine üzerinden tuş bırakma koruma süresi üretir.

    F-0444: tekil motora delege eder (ayrı RNG akışı yoktur).
    """
    return get_human_timing_engine().get_key_release_guard_sec()


def get_burst_inter_key_delay_sec(trusted: bool = True) -> float:
    """Global HumanTimingEngine üzerinden çoklu tuş aralığı gecikmesi üretir.

    F-0444: tekil motora delege eder (ayrı RNG akışı yoktur).
    """
    return get_human_timing_engine().get_burst_inter_key_delay_sec(trusted=trusted)


def align_to_gcd(
    burst_duration_sec: float,
    gcd_sec: float = 0.12,
    min_gap_sec: float = 0.020,
) -> float:
    """Global HumanTimingEngine üzerinden GCD uyumlu bekleme süresi hesaplar.

    F-0444: tekil motora delege eder (ayrı RNG akışı yoktur).
    """
    return get_human_timing_engine().align_to_gcd(
        burst_duration_sec=burst_duration_sec, gcd_sec=gcd_sec, min_gap_sec=min_gap_sec
    )


def sleep_gaussian_jitter(
    base_sec: float,
    std_sec: float = 0.005,
    min_sec: Optional[float] = None,
    max_sec: Optional[float] = None,
) -> float:
    """Gaussian jitter süresi kadar bekler ve uygulanan süreyi döndürür.

    F-0444: tekil motora delege eder (ayrı RNG akışı yoktur). Eskiden
    buradaki sleep mantığı HumanTimingEngine.sleep_gaussian_jitter ile
    birebir kopyaydı; artık tek tanım motorda, burası ince sarmalayıcıdır.
    """
    return get_human_timing_engine().sleep_gaussian_jitter(
        base_sec, std_sec=std_sec, min_sec=min_sec, max_sec=max_sec
    )


if __name__ == "__main__":
    run_statistical_distribution_verification(1000)


def _probe_human_timing_engine(engine: Any) -> Tuple[bool, Dict[str, Any]]:
    """S-1701: insan zamanlama motorunu gercekten yoklar.

    Sabit liste / kosulsuz True yerine motorun canli durumunu okur:
    yorgunluk carpani aralikta mi, gorsel tepki / jitter / Ex-Gaussian
    ornekleri sozlesme araliklarinda mi, RNG degiskenlik uretiyor mu
    (8 ornekte en az 2 farkli deger) ve ornek sayaci ilerliyor mu.
    Hicbir ornek uyutmaz (yalnizca dagilim orneklemesi).
    """
    checks: Dict[str, Any] = {}
    try:
        if engine is None:
            return False, {"error": "engine yok"}
        fatigue = float(engine.get_fatigue_multiplier())
        checks["fatigue_in_range"] = 0.98 <= fatigue <= 1.08
        reaction = float(engine.get_visual_reaction_delay_sec())
        checks["reaction_in_range"] = 0.180 <= reaction <= 0.380
        jitter = float(
            engine.sample_gaussian_jitter(0.05, std_sec=0.005, min_sec=0.02, max_sec=0.08)
        )
        checks["jitter_in_range"] = 0.02 <= jitter <= 0.08
        ex = float(engine.sample_ex_gaussian(0.218, 0.019, 0.038, 0.180, 0.380))
        checks["ex_gaussian_in_range"] = 0.180 <= ex <= 0.380
        samples = [
            float(engine.sample_gaussian_jitter(0.05, std_sec=0.005, min_sec=0.02, max_sec=0.08))
            for _ in range(8)
        ]
        checks["rng_varies"] = len(set(samples)) >= 2
        checks["samples_generated"] = int(getattr(engine, "_total_samples_generated", 0) or 0) > 0
        return bool(all(checks.values())), checks
    except Exception as exc:
        logger.warning("human_timing motor probu basarisiz: %s", exc)
        checks["error"] = str(exc)
        return False, checks


def get_human_timing_capabilities() -> list:
    return ["human_timing", "gaussian_jitter", "diagnostic_snapshot"]


def _validate_human_timing_contract_detailed() -> Tuple[bool, Dict[str, Any]]:
    """S-1701: eskiden `bool(get_human_timing_capabilities())` idi — sabit,
    bos olmayan bir liste her zaman True donduruyordu; readiness/audit/
    digest/integrity/compliance zincirinin tamami sahte basari uretiyordu.
    Simdi: capability isimleri gercek motor ozelliklerine baglanir ve motor
    `_probe_human_timing_engine` ile canli yoklanir; herhangi bir denetim
    gecemezse False doner.

    P2-1: probu bir kez calistirip `(sonuc, denetimler)` ciftini dondurur;
    ``validate_human_timing_contract()`` ve
    ``get_human_timing_audit_summary()`` ikisi de buna delege eder, boylece
    tek ust duzey okumada motor yalnizca bir kez yoklanir (prob 11 ornek
    uretir; 9 kez tekrarlanmasi ornek sayacini da 9 kat sisisiriyordu).
    """
    try:
        caps = get_human_timing_capabilities()
        if not caps:
            return False, {"error": "capability listesi bos"}
        engine = get_human_timing_engine()
        if not isinstance(engine, HumanTimingEngine):
            return False, {"error": "motor yok"}
        capability_checks = {
            "human_timing": isinstance(engine, HumanTimingEngine),
            "gaussian_jitter": callable(getattr(engine, "sample_gaussian_jitter", None)),
            "diagnostic_snapshot": callable(
                getattr(engine, "get_human_timing_diagnostic_snapshot", None)
            ),
        }
        for name in caps:
            if name in capability_checks and not capability_checks[name]:
                logger.warning("human_timing sozlesme ihlali: capability eksik: %s", name)
                return False, {"error": f"capability eksik: {name}"}
        ok, details = _probe_human_timing_engine(engine)
        return bool(ok), details
    except Exception as exc:
        logger.warning("human_timing sozlesme dogrulamasi basarisiz: %s", exc)
        return False, {"error": str(exc)}


def validate_human_timing_contract() -> bool:
    ok, _details = _validate_human_timing_contract_detailed()
    return bool(ok)


def _human_timing_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. telemetry.py'deki aciklama).
    ``None`` ise dogrulama burada bir kez calisir; cagrilar arasi
    onbellek yoktur — her ust duzey okuma taze prob yapar."""
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_human_timing_contract())


def get_human_timing_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "human_timing", "contract_valid": _human_timing_contract_ok(_contract_ok), "capability_count": len(get_human_timing_capabilities())}


def get_human_timing_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "ready": ok, "metrics": get_human_timing_baseline_metrics(_contract_ok=ok)}


def get_human_timing_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "audit_passed": ok, "readiness": get_human_timing_readiness_status(_contract_ok=ok)}


def get_human_timing_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "operational": ok, "audit": get_human_timing_audit_record(_contract_ok=ok)}


def get_human_timing_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "human_timing", "payload": get_human_timing_operational_digest(_contract_ok=_contract_ok)}


def get_human_timing_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "sealed": ok, "telemetry": get_human_timing_telemetry_payload(_contract_ok=ok)}


def get_human_timing_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "certified": ok, "seal": get_human_timing_integrity_seal(_contract_ok=ok)}


def get_human_timing_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _human_timing_contract_ok(_contract_ok)
    return {"module": "human_timing", "ready": ok, "certificate": get_human_timing_compliance_certificate(_contract_ok=ok)}


def get_human_timing_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    # S-1701: audit_passed artik validate zincirinin gercek sonucu; ayrintili
    # prob sonuclari da eklenir ki basarisizlikta hangi denetim dustugu gorunsun.
    # P2-1: prob yalnizca bir kez calisir — ayrintili dogrulama hem `ok` hem
    # `checks`'i tek yoklamadan uretir, alt zincire `ok` aktarilir.
    if _contract_ok is None:
        ok, details = _validate_human_timing_contract_detailed()
    else:
        # Disaridan aktarilmis sonuc: `checks` ayrintisi o yoklamaya ait
        # degildir; bos birakilir (taze ayrinti icin parametresiz cagir).
        ok, details = bool(_contract_ok), {}
    return {
        "module": "human_timing",
        "audit_passed": bool(ok),
        "checks": details,
        "probe": get_human_timing_readiness_probe(_contract_ok=bool(ok)),
    }

