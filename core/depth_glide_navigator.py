# -*- coding: utf-8 -*-
"""
Section 5: Derinlik Haritası (3D Monocular Depth Estimation),
Akıllı Engel Kaçınma (Local Obstacle Avoidance) ve
Otomatik Süzülme / Uçurum Atlama Noktası Kontrolcüsü (Glide Waypoint Controller).

Mimari Bileşenler:
1. MonocularDepthObstacleAvoidance:
   - `models/` dizininde hafif ONNX derinlik modeli (`Depth-Anything-Small` / `MiDaS-small`)
     bulunursa `onnxruntime` veya `cv2.dnn` ile yükler.
   - Model dosyası olmasa dahi kutudan çıktığı anda çalışan **< 5ms Gerçek Zamanlı
     Geometrik + Dikey Perspektif + Sobel Doku Gradyanı + Gökyüzü/Uçurum Segmentasyonlu
     Derinlik Tahmincisi** (`estimate_depth_map`) içerir.
   - Karakterin önündeki hareket koridorunu (`Left`, `Center`, `Right` sektörleri) analiz eder
     (`analyze_forward_corridor`), yakın engelleri (`wall`, `rock`, `tree_trunk`) ve ani
     derinlik kopmalarını (`cliff_drop`) tespit ederek yumuşak yerel kaçınma açısı
     (`recommended_steer_deg`) üretir.

2. GlideWaypointController:
   - Yüksek rakımlı `current_node` ile alçak rakımlı / vadi-uçurum ötesi `next_node`
     arasındaki geçişin **Glide Kısayolu** (`is_glide_waypoint` / `should_use_glide`)
     olup olmadığını değerlendirir (40sn koşu süresini ~5sn süzülmeye indirir).
   - `execute_glide_maneuver`: Uçurum kenarına hizalanma, ileri depar (`W`), uçurumdan
     sıçrama (`Space`), tepe noktasında (`0.35s`) kanat açma (`Space`), hedef kaynağa
     doğru havada yön/irtifa koruma ve hedef noktada kanat kapatıp (`Space`) iniş
     sekansını yürütür.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from version import __version__

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except Exception:
    ort = None
    HAS_ONNXRUNTIME = False

logger = logging.getLogger("DepthGlideNavigator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
logger.propagate = False


class MonocularDepthObstacleAvoidance:
    """
    Gerçek Zamanlı 3B Monoküler Derinlik Tahmini ve Yerel Engel Kaçınma Sistemi.

    Derinlik Haritası Konvansiyonu (`estimate_depth_map` çıktısı):
    - `np.ndarray` (`dtype=np.float32`, değer aralığı `[0.0, 1.0]`)
    - `0.0` -> Kameraya/karaktere çok yakın (engel: kaya, duvar, ağaç gövdesi)
    - `1.0` -> Çok uzak mesafe / açık ufuk / gökyüzü / uçurum boşluğu
    """

    CANDIDATE_ONNX_MODELS: Tuple[str, str, str, str] = (
        "depth_anything_v2_vits.onnx",
        "depth_anything_small.onnx",
        "midas_v21_small_256.onnx",
        "midas_small.onnx",
    )

    def __init__(
        self,
        model_dir: str = "models",
        work_size: Tuple[int, int] = (160, 90),
        prefer_realtime_geometric: bool = True,
        obstacle_clearance_threshold: float = 0.35,
        cliff_drop_threshold: float = 0.58,
        max_steer_deg: float = 42.0,
    ) -> None:
        self.model_dir = model_dir
        self.work_width = max(64, int(work_size[0]))
        self.work_height = max(36, int(work_size[1]))
        self.prefer_realtime_geometric = bool(prefer_realtime_geometric)
        self.obstacle_clearance_threshold = float(obstacle_clearance_threshold)
        self.cliff_drop_threshold = float(cliff_drop_threshold)
        self.max_steer_deg = float(max_steer_deg)

        self.onnx_session: Optional[Any] = None
        self.cv_net: Optional[Any] = None
        self.loaded_model_path: Optional[str] = None
        self.backend_name: str = "geometric_perspective_sobel_realtime"

        # < 5ms çalışma için önceden hesaplanmış dikey perspektif ve merkez ağırlık matrisleri
        self._vertical_prior = self._build_vertical_perspective_prior(
            self.work_height, self.work_width
        )
        self._corridor_mask = self._build_corridor_vignette(
            self.work_height, self.work_width
        )
        self._smoothed_steer_deg: float = 0.0
        self._last_inference_ms: float = 0.0

        self._try_load_depth_model()
        # İlk OpenCV JIT/SIMD ısınmasını başlangıçta tamamla (<5ms çalışma garantisi için)
        _warmup = np.zeros((180, 320, 3), dtype=np.uint8)
        self.estimate_depth_map(_warmup)

    @staticmethod
    def _build_vertical_perspective_prior(height: int, width: int) -> np.ndarray:
        """
        Üçüncü şahıs (TPS) kamera geometrisine uygun dikey perspektif derinlik öncülü:
        Ekranın alt kısmı (karakterin ayak hizası) yakın (0.10), ufuk çizgisi ve üstü uzak (0.95).
        """
        y_norm = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, np.newaxis]
        # TPS kamerada ufuk genelde üst %25-%55 bandındadır; non-lineer perspektif eğrisi
        prior_1d = np.clip(np.power(y_norm, 0.78), 0.05, 0.98).astype(np.float32)
        return np.repeat(prior_1d, width, axis=1)

    @staticmethod
    def _build_corridor_vignette(height: int, width: int) -> np.ndarray:
        """Ekranın yatay merkez eksenini vurgulayan hafif ağırlık matrisi."""
        x_norm = np.linspace(-1.0, 1.0, width, dtype=np.float32)[np.newaxis, :]
        horiz = 1.0 - 0.12 * (x_norm * x_norm)
        return np.repeat(horiz.astype(np.float32), height, axis=0)

    def _try_load_depth_model(self) -> bool:
        """
        `models/` dizininde hafif bir ONNX derinlik modeli (`Depth-Anything-Small` veya
        `MiDaS-small`) mevcutsa `onnxruntime` ya da `cv2.dnn` üzerinden yükler.
        """
        if not os.path.isdir(self.model_dir):
            return False

        for fname in self.CANDIDATE_ONNX_MODELS:
            candidate = os.path.join(self.model_dir, fname)
            if not os.path.isfile(candidate):
                continue

            # F-1832 / P2-10: ONNX modeli imzasız yüklenmesin — yükleme
            # için `<model>.sha256` sidecar ZORUNLUDUR. Sidecar yoksa
            # model yüklenmez (hash'i loglayıp yüklemek, kural ile
            # çelişiyordu); sidecar varsa özet doğrulanır, uyuşmazlıkta
            # yükleme reddedilir.
            try:
                with open(candidate, "rb") as fh:
                    model_sha256 = hashlib.sha256(fh.read()).hexdigest()
            except Exception as exc:
                logger.warning("ONNX model dosyası okunamadı (%s): %s", candidate, exc)
                continue
            sidecar = candidate + ".sha256"
            if not os.path.isfile(sidecar):
                logger.error(
                    "ONNX model SHA-256 sidecar yok, yükleme reddedildi: %s "
                    "(%s eksik — hash doğrulaması yapılamaz; imzasız model yüklenmez)",
                    candidate,
                    sidecar,
                )
                continue
            try:
                with open(sidecar, "r", encoding="utf-8") as sfh:
                    expected = sfh.read().strip().split()[0].lower()
                if expected != model_sha256:
                    logger.error(
                        "ONNX model SHA-256 uyuşmazlığı, yükleme reddedildi: %s "
                        "(beklenen=%s..., gerçek=%s...)",
                        candidate,
                        expected[:16],
                        model_sha256[:16],
                    )
                    continue
            except Exception as exc:
                logger.warning("ONNX model SHA-256 sidecar okunamadı (%s): %s", candidate, exc)
                continue
            logger.info("ONNX derinlik modeli SHA-256: %s %s", model_sha256[:16], candidate)

            if HAS_ONNXRUNTIME and ort is not None:
                try:
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = 2
                    sess_opts.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    self.onnx_session = ort.InferenceSession(
                        candidate,
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self.loaded_model_path = candidate
                    self.backend_name = f"onnxruntime:{fname}"
                    logger.info("ONNX derinlik modeli yüklendi: %s", candidate)
                    return True
                except Exception as exc:
                    logger.debug("onnxruntime model yükleme uyarısı (%s): %s", candidate, exc)

            try:
                self.cv_net = cv2.dnn.readNetFromONNX(candidate)
                self.loaded_model_path = candidate
                self.backend_name = f"cv2.dnn:{fname}"
                logger.info("OpenCV DNN derinlik modeli yüklendi: %s", candidate)
                return True
            except Exception as exc:
                logger.debug("cv2.dnn model yükleme uyarısı (%s): %s", candidate, exc)

        return False

    def _estimate_depth_onnx(self, small_bgr: np.ndarray) -> Optional[np.ndarray]:
        """ONNX modeli yüklüyse derinlik tahmini yapar ve [0.0=yakın, 1.0=uzak] normalize eder."""
        try:
            rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (rgb - mean) / std
            blob = np.transpose(norm, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

            if self.onnx_session is not None:
                input_name = self.onnx_session.get_inputs()[0].name
                pred = self.onnx_session.run(None, {input_name: blob})[0]
            elif self.cv_net is not None:
                self.cv_net.setInput(blob)
                pred = self.cv_net.forward()
            else:
                return None

            depth_raw = np.squeeze(pred).astype(np.float32)
            if depth_raw.ndim != 2:
                return None

            d_min = float(np.min(depth_raw))
            d_max = float(np.max(depth_raw))
            inv_depth = (depth_raw - d_min) / max(1e-6, d_max - d_min)
            # MiDaS / Depth-Anything ters derinlik (disparity) verir: yüksek değer = yakın.
            # Konvansiyonumuza (0.0=yakın, 1.0=uzak) çeviriyoruz:
            metric_like = 1.0 - np.clip(inv_depth, 0.0, 1.0)
            return cv2.resize(
                metric_like,
                (self.work_width, self.work_height),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception as exc:
            logger.debug("ONNX derinlik çıkarımı geri dönüşü: %s", exc)
            return None

    def estimate_depth_map(
        self,
        frame: np.ndarray,
        return_full_resolution: bool = False,
    ) -> np.ndarray:
        """
        < 5ms Gerçek Zamanlı Geometrik + Dikey Perspektif + Sobel Doku Gradyanı +
        Gökyüzü/Uçurum Segmentasyonlu Derinlik Tahmincisi.

        Parametreler:
        - `frame`: BGR veya Grayscale oyun karesi (`np.ndarray`).
        - `return_full_resolution`: `True` ise çıktıyı orijinal kare boyutuna ölçekler,
          `False` (varsayılan) ise yüksek hızlı analiz çözünürlüğünde (`90x160`) döndürür.

        Dönüş:
        - `depth_map`: `float32` `[0.0, 1.0]` (`0.0` = çok yakın engel, `1.0` = uzak/ufuk/uçurum).
        """
        t0 = time.perf_counter()

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            # F-1835: Boş/None karede 0.5'lik harita fail-open'dır (koridor "açık"
            # görünür). Fail-closed: 0.0 (çok yakın/engelli) döndür ki koridor
            # analizi engel saysın.
            logger.warning("estimate_depth_map: boş/geçersiz kare, fail-closed 0.0 harita dönülüyor.")
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return np.zeros((self.work_height, self.work_width), dtype=np.float32)

        # F-1822: Tek kanallı 3B kare (h, w, 1) / 2 kanallı kare BGR varsayımıyla
        # çökerdi — kanal sayısını normalize et.
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[2] == 2:
            frame = frame[:, :, 0]
        elif frame.ndim not in (2, 3):
            logger.warning(
                "estimate_depth_map: beklenmeyen kare boyutu %s, fail-closed 0.0 harita dönülüyor.",
                getattr(frame, "shape", None),
            )
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return np.zeros((self.work_height, self.work_width), dtype=np.float32)

        orig_h, orig_w = frame.shape[:2]

        # 1. Ultra-hızlı alt örnekleme (<1ms): Bitişik (contiguous) bellek üzerinde
        # doğrudan INTER_NEAREST ile 160x90'a indirgeme (~0.15ms)
        if frame.ndim == 2:
            small_gray_u8 = cv2.resize(
                frame, (self.work_width, self.work_height), interpolation=cv2.INTER_NEAREST
            )
            small_bgr = cv2.cvtColor(small_gray_u8, cv2.COLOR_GRAY2BGR)
        else:
            small_bgr = cv2.resize(
                frame,
                (self.work_width, self.work_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if small_bgr.shape[2] > 3:
                small_bgr = small_bgr[:, :, :3]
            small_gray_u8 = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # İsteğe bağlı ONNX modeli (prefer_realtime_geometric=False ve model yüklüyse)
        if not self.prefer_realtime_geometric and (
            self.onnx_session is not None or self.cv_net is not None
        ):
            onnx_depth = self._estimate_depth_onnx(small_bgr)
            if onnx_depth is not None:
                self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
                if return_full_resolution and (
                    orig_h != self.work_height or orig_w != self.work_width
                ):
                    return cv2.resize(
                        onnx_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                    )
                return onnx_depth

        # 2. Normalize parlaklık ve renk kanalları (vektörize float32)
        b_ch = small_bgr[:, :, 0].astype(np.float32) * (1.0 / 255.0)
        g_ch = small_bgr[:, :, 1].astype(np.float32) * (1.0 / 255.0)
        r_ch = small_bgr[:, :, 2].astype(np.float32) * (1.0 / 255.0)
        luma = small_gray_u8.astype(np.float32) * (1.0 / 255.0)

        # 3. Sobel yatay/dikey doku ve kenar gradyanı
        # Yakın engeller (kaya, ağaç gövdesi, duvar) yüksek mikro-doku ve dikey kenar yoğunluğuna sahiptir
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        grad_smooth = cv2.blur(grad_mag, (7, 7))
        grad_norm = np.clip(grad_smooth * 1.15, 0.0, 1.0)

        # Dikey kenar baskınlığı (ağaç gövdesi / kaya sütunu / duvar tespiti için)
        # D10: blur doğrusal bir işlemdir: blur(a) - 0.6*blur(b) == blur(a - 0.6*b)
        # (float32 yuvarlama farkı ~2.4e-07). İki (5,9) blur yerine tek blur
        # çağrısı yapılır; çıktı pratikte aynıdır, çağrı başına ~0.7ms kazandırır.
        vertical_structure = np.clip(
            cv2.blur(np.abs(gx) - 0.60 * np.abs(gy), (5, 9)) * 1.6, 0.0, 1.0
        )

        # 4. Gökyüzü / Atmosferik Perspektif / Uçurum Boşluğu Segmentasyonu
        # Uzak vadi, sis ve gökyüzü yüksek parlaklık + mavi/açık-gri ton + düşük doku gradyanı gösterir
        max_rgb = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        saturation = (max_rgb - min_rgb) / np.maximum(max_rgb, 1e-4)

        blue_sky_bias = np.clip((b_ch - 0.72 * r_ch) + 0.38 * luma, 0.0, 1.0)
        atmospheric_haze = np.clip(
            luma * (1.0 - 0.60 * saturation) * (1.0 - 0.65 * grad_norm), 0.0, 1.0
        )
        sky_or_valley_score = np.clip(
            0.55 * atmospheric_haze + 0.45 * blue_sky_bias, 0.0, 1.0
        )

        # 5. Yakın Karanlık / Kütlesel Engel Tespiti (Kaya, Duvar, Ağaç Gövdesi)
        dark_mass = np.clip((0.50 - luma) * 1.85, 0.0, 1.0)
        near_obstacle_response = np.clip(
            0.38 * grad_norm + 0.34 * vertical_structure + 0.38 * dark_mass,
            0.0,
            1.0,
        )

        # 6. Dikey Perspektif Öncülü + Doku Gradyanı + Gökyüzü/Uçurum Birleşimi
        # F-1824: Önceki ağırlıklar (0.56 öncül) haritayı büyük ölçüde sabit
        # dikey öncüle bağlıyor, görüntü kanıtı (engel tepkisi) zayıf kalıyordu.
        # Engel tepkisi artık baskın; öncül yalnızca zayıf bir önsel.
        depth_est = (
            0.34 * self._vertical_prior
            + 0.40 * sky_or_valley_score
            - 0.60 * near_obstacle_response
            + 0.10
        )
        depth_est = depth_est * self._corridor_mask

        # Karakterin kendi modeli (ekranın alt-orta %18'lik kısmı) engel sayılmasın diye nötrlenir
        h_self_start = int(self.work_height * 0.80)
        w_self_left = int(self.work_width * 0.41)
        w_self_right = int(self.work_width * 0.59)
        depth_est[h_self_start:, w_self_left:w_self_right] = np.maximum(
            depth_est[h_self_start:, w_self_left:w_self_right],
            self._vertical_prior[h_self_start:, w_self_left:w_self_right],
        )

        depth_map = np.clip(depth_est, 0.0, 1.0).astype(np.float32)

        if return_full_resolution and (
            orig_h != self.work_height or orig_w != self.work_width
        ):
            depth_map = cv2.resize(
                depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return depth_map

    def analyze_forward_corridor(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Karakterin önündeki hareket koridorunu `Left`, `Center` ve `Right` sektörlerine
        ayırarak inceler; yakın engelleri (`wall`, `rock`, `tree_trunk`) veya ani derinlik
        kopmalarını (`cliff_drop`) tespit eder ve yerel kaçınma direksiyon açısı
        (`recommended_steer_deg`) hesaplar.

        Dönüş Sözlüğü:
        - `recommended_steer_deg`: float (-max_steer_deg .. +max_steer_deg, negatif=Sol, pozitif=Sağ)
        - `obstacle_detected`: bool
        - `obstacle_type`: str (`"none"`, `"wall"`, `"rock"`, `"tree_trunk"`, `"cliff_drop"`)
        - `clearance_left`: float (`[0.0, 1.0]`)
        - `clearance_center`: float (`[0.0, 1.0]`)
        - `clearance_right`: float (`[0.0, 1.0]`)
        - `cliff_drop_detected`: bool
        - `inference_ms`: float
        - `depth_map_shape`: Tuple[int, int]
        """
        t0 = time.perf_counter()
        depth_map = self.estimate_depth_map(frame, return_full_resolution=False)
        dh, dw = depth_map.shape[:2]

        # İleri hareket koridoru ROI: Dikey %26 (ufuk altı) ile %76 (ayak üstü) arası,
        # Yatay %12 ile %88 arası (3 sektöre bölünür: Left, Center, Right)
        y1 = int(dh * 0.26)
        y2 = int(dh * 0.76)
        x_left_start = int(dw * 0.10)
        x_left_end = int(dw * 0.37)
        x_center_end = int(dw * 0.63)
        x_right_end = int(dw * 0.90)

        sector_left = depth_map[y1:y2, x_left_start:x_left_end]
        sector_center = depth_map[y1:y2, x_left_end:x_center_end]
        sector_right = depth_map[y1:y2, x_center_end:x_right_end]

        clearance_left = float(np.mean(sector_left))
        clearance_center = float(np.mean(sector_center))
        clearance_right = float(np.mean(sector_right))

        # Ani uçurum kopması (cliff_drop) analizi:
        # Alt-orta zemin (y: %52..%78) beklenenden çok yüksek derinlik/boşluk değerine sahipse
        # veya üst-alt geçişte keskin derinlik sıçraması varsa
        lower_ground_center = depth_map[int(dh * 0.54) : int(dh * 0.78), x_left_end:x_center_end]
        upper_horizon_center = depth_map[int(dh * 0.22) : int(dh * 0.44), x_left_end:x_center_end]
        lower_ground_depth = float(np.mean(lower_ground_center))
        upper_horizon_depth = float(np.mean(upper_horizon_center))

        # Normal zeminde lower_ground_depth ~0.25-0.48 aralığındadır; uçurum kenarında >0.66 olur
        # D10: üç ayrı np.percentile çağrısı üç kez sıralama yapar; [15, 22, 90]
        # tek çağrıda hesaplanır (fark <= ~1.2e-08, float32 yuvarlama gürültüsü).
        _p15, _p22, _p90 = np.percentile(sector_center, [15, 22, 90])
        vertical_discontinuity = float(_p90 - _p15)
        cliff_drop_detected = bool(
            lower_ground_depth >= self.cliff_drop_threshold
            or (
                lower_ground_depth >= 0.60
                and upper_horizon_depth >= 0.74
                and vertical_discontinuity >= 0.34
            )
        )

        # Yakın engel tespiti ve sınıflandırması (`wall`, `rock`, `tree_trunk`)
        near_quantile_center = float(_p22)
        col_means = np.mean(sector_center, axis=0)
        min_col_clearance = float(np.min(col_means)) if col_means.size > 0 else clearance_center
        obstacle_detected = False
        obstacle_type = "none"

        if (
            near_quantile_center < self.obstacle_clearance_threshold * 0.80
            or clearance_center < self.obstacle_clearance_threshold
            or min_col_clearance < self.obstacle_clearance_threshold * 0.78
        ):
            obstacle_detected = True
            prior_center = self._vertical_prior[y1:y2, x_left_end:x_center_end]
            obstacle_type = self._classify_center_obstacle(
                sector_center=sector_center,
                prior_center=prior_center,
                clearance_left=clearance_left,
                clearance_center=clearance_center,
                clearance_right=clearance_right,
            )
        elif cliff_drop_detected:
            obstacle_detected = True
            obstacle_type = "cliff_drop"

        # F-1823: Uçurum tespit edildiyse türü asla yakın-engel sınıfıyla
        # (wall/rock/tree_trunk) maskelenmesin — uçurum güvenlik açısından
        # önceliklidir, aksi halde uçurum tipi engel kaybolur.
        if cliff_drop_detected:
            obstacle_detected = True
            obstacle_type = "cliff_drop"

        # Yerel kaçınma direksiyon açısı (recommended_steer_deg) hesabı:
        # Negatif açı = Sola dönüş, Pozitif açı = Sağa dönüş
        corridor_asymmetry = clearance_right - clearance_left
        if obstacle_detected and obstacle_type != "cliff_drop":
            urgency = np.clip(
                (self.obstacle_clearance_threshold - min(clearance_center, near_quantile_center))
                / max(0.15, self.obstacle_clearance_threshold),
                0.15,
                1.0,
            )
            direction_sign = 1.0 if clearance_right >= clearance_left else -1.0
            raw_steer = direction_sign * (
                12.0 + float(urgency) * (self.max_steer_deg - 12.0)
            ) + (corridor_asymmetry * 18.0)
            raw_steer = float(
                np.clip(raw_steer, -self.max_steer_deg, self.max_steer_deg)
            )
        elif obstacle_type == "cliff_drop":
            # Uçurum tespit edildiğinde (glide aktif değilse) daha güvenli/sığ tarafa yönel
            safer_right = abs(clearance_right - 0.45) < abs(clearance_left - 0.45)
            raw_steer = 24.0 if safer_right else -24.0
        else:
            # Koridor açıkken bile taş/ağaç kenarlarından uzak tutan hafif merkezleme düzeltmesi
            if abs(corridor_asymmetry) > 0.045:
                raw_steer = float(
                    np.clip(corridor_asymmetry * 28.0, -14.0, 14.0)
                )
            else:
                raw_steer = 0.0

        # EMA ile yumuşak direksiyon geçişi
        self._smoothed_steer_deg = (
            0.68 * raw_steer + 0.32 * self._smoothed_steer_deg
        )
        recommended_steer_deg = round(float(self._smoothed_steer_deg), 2)
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "recommended_steer_deg": recommended_steer_deg,
            "obstacle_detected": bool(obstacle_detected),
            "obstacle_type": obstacle_type,
            "clearance_left": round(clearance_left, 4),
            "clearance_center": round(clearance_center, 4),
            "clearance_right": round(clearance_right, 4),
            "cliff_drop_detected": bool(cliff_drop_detected),
            "lower_ground_depth": round(lower_ground_depth, 4),
            "inference_ms": round(total_ms, 3),
            "depth_map_shape": (int(dh), int(dw)),
            "backend": self.backend_name,
        }

    def _classify_center_obstacle(
        self,
        sector_center: np.ndarray,
        prior_center: np.ndarray,
        clearance_left: float,
        clearance_center: float,
        clearance_right: float,
    ) -> str:
        """
        Merkez koridordaki engelin geometrik dağılımına göre türünü belirler:
        - `wall`: Sol, merkez ve sağ koridorların tamamı yakın/kapalı veya yatay varyans çok düşük
        - `tree_trunk`: Dikey boyunca dar sütun şeklinde yakınlık (yatay varyans yüksek, üst-alt benzer)
        - `rock`: Alt-orta bölgede yoğunlaşmış kaya/kütle
        """
        sh = sector_center.shape[0]
        # Dikey perspektif öncülünden arındırılmış engel tepkisi
        # (öncül ağırlığı estimate_depth_map ile aynı: 0.34)
        residual = sector_center - 0.34 * prior_center
        top_half = float(np.mean(residual[: max(1, sh // 2), :]))
        bottom_half = float(np.mean(residual[max(1, sh // 2) :, :]))
        col_profile = np.mean(residual, axis=0)
        horiz_std = float(np.std(col_profile))
        col_span = float(np.max(col_profile) - np.min(col_profile))

        # Dar dikey sütun (yüksek yatay kontrast + üst-alt yarıda sürekli engel) -> tree_trunk
        if col_span >= 0.085 and abs(top_half - bottom_half) < 0.065:
            return "tree_trunk"

        # Tüm sektörler kapalıysa ve yatay profil tekdüze ise -> wall
        if (
            clearance_left < self.obstacle_clearance_threshold * 1.08
            and clearance_right < self.obstacle_clearance_threshold * 1.08
            and horiz_std < 0.018
            and abs(top_half - bottom_half) < 0.045
        ):
            return "wall"

        # Alt-orta bölgede yoğunlaşmış kaya/kaynak kütleleri -> rock
        return "rock"

    def apply_avoidance_steering(
        self,
        input_ctrl: Optional[Any],
        corridor_info: Dict[str, Any],
        pixels_per_degree: float = 4.2,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        `analyze_forward_corridor` çıktısındaki `recommended_steer_deg` değerini
        fare hareketine dönüştürerek botun kayalara ve ağaçlara takılmadan önce
        yumuşakça etrafından dolaşmasını sağlar.

        F-0081: Varsayılan `test_mode=True` üretimde fareyi hiç oynatmıyor,
        üstelik fare hatası yutulup `applied=True` dönüyordu. Varsayılan artık
        `False` (üretimde gerçekten uygular); `applied` yalnızca fare gerçekten
        oynatıldıysa (veya açıkça simülasyon istendiyse) `True` döner.
        """
        steer_deg = float(corridor_info.get("recommended_steer_deg", 0.0))
        if abs(steer_deg) < 1.5:
            return {
                "applied": False,
                "steer_deg": steer_deg,
                "mouse_dx": 0,
                "reason": "corridor_clear",
            }

        mouse_dx = int(round(steer_deg * float(pixels_per_degree)))
        if test_mode:
            return {
                "applied": True,
                "steer_deg": steer_deg,
                "mouse_dx": mouse_dx,
                "obstacle_type": corridor_info.get("obstacle_type", "none"),
                "test_mode": True,
                "simulated": True,
            }

        moved = False
        move_error: Optional[str] = None
        if input_ctrl is not None:
            try:
                if hasattr(input_ctrl, "get_cursor_position") and hasattr(
                    input_ctrl, "smooth_move_to"
                ):
                    cx, cy = input_ctrl.get_cursor_position()
                    input_ctrl.smooth_move_to(cx + mouse_dx, cy, duration=0.045)
                    moved = True
                # smooth_move_to mutlak hedef alır; imleç konumu okunamıyorsa
                # göreli kaydırma uydurulamaz — uygulanmadı sayılır.
            except Exception as exc:
                move_error = str(exc)
                logger.warning("Kaçınma fare yönlendirme hatası: %s", exc)

        return {
            "applied": bool(moved),
            "steer_deg": steer_deg,
            "mouse_dx": mouse_dx,
            "obstacle_type": corridor_info.get("obstacle_type", "none"),
            "test_mode": False,
            "simulated": False,
            "reason": (
                "steering_applied"
                if moved
                else ("mouse_move_failed: " + move_error if move_error else "no_usable_input_controller")
            ),
        }

    def _probe_depth_readiness(self) -> Dict[str, Any]:
        """
        S-0401: Gerçek hazır probu — eşik sağlığı + sentetik kare üzerinde
        derinlik haritası ve koridor analizi ölçülür, sabit True döndürülmez
        (S-0602'nin GlideWaypointController'daki proba bağlama kararı).
        Pozitif kontrol: dikey gradyanlı sentetik kare geçerli, değişken bir
        derinlik haritası ve sınırlı bir direksiyon açısı üretmeli.
        Negatif kontrol: fail-closed sıfır harita (std 0) probu geçirmez.
        """
        checks: Dict[str, Any] = {}
        try:
            checks["thresholds_sane"] = bool(
                self.work_width >= 64
                and self.work_height >= 36
                and 0.0 < self.obstacle_clearance_threshold < 1.0
                and 0.0 < self.cliff_drop_threshold < 1.0
                and self.max_steer_deg > 0
                and self._vertical_prior.shape
                == (self.work_height, self.work_width)
                and self._corridor_mask.shape
                == (self.work_height, self.work_width)
            )
            checks["backend_known"] = bool(
                isinstance(self.backend_name, str) and self.backend_name
            )
            # Sentetik dikey gradyan kare: üst açık (parlak), alt yakın (koyu).
            rows = np.linspace(255.0, 0.0, 180, dtype=np.float32)[:, None, None]
            frame = np.repeat(np.repeat(rows, 320, axis=1), 3, axis=2).astype(
                np.uint8
            )
            depth_map = self.estimate_depth_map(frame, return_full_resolution=False)
            checks["depth_map_shape_ok"] = bool(
                depth_map.shape == (self.work_height, self.work_width)
                and depth_map.dtype == np.float32
            )
            checks["depth_map_finite_in_range"] = bool(
                np.all(np.isfinite(depth_map))
                and float(depth_map.min()) >= 0.0
                and float(depth_map.max()) <= 1.0
            )
            checks["depth_map_varies"] = bool(float(np.std(depth_map)) > 0.0)
            corridor = self.analyze_forward_corridor(frame)
            steer = float(corridor.get("recommended_steer_deg", float("nan")))
            checks["corridor_probe_ok"] = bool(
                corridor.get("depth_map_shape")
                == (self.work_height, self.work_width)
                and np.isfinite(steer)
                and abs(steer) <= self.max_steer_deg
            )
        except Exception as exc:
            checks["error"] = str(exc)
        checks["passed"] = bool(checks.get("thresholds_sane")) and all(
            bool(v) for k, v in checks.items() if k not in ("error", "passed")
        )
        return checks

    def _ready_state(self) -> bool:
        """Gerçek hazır durumu: öncül matrisler doğru şekilde kurulu ve arka uç biliniyor."""
        try:
            return bool(
                self._vertical_prior.shape == (self.work_height, self.work_width)
                and self._corridor_mask.shape == (self.work_height, self.work_width)
                and self.backend_name
                and np.isfinite(self._last_inference_ms)
            )
        except Exception:
            return False

    def _combined_ready(self) -> bool:
        # S-0401 probu (sentetik kare + koridor) VE D10 hazır durumu
        # (öncül matrisler + arka uç + son çıkarım süresi) ikisi de geçmeli.
        return bool(self._probe_depth_readiness().get("passed", False)) and bool(
            self._ready_state()
        )

    def get_depth_glide_navigator_ready_flag(self) -> dict:
        # S-0401 + D10: koşulsuz True değil; prob zinciri ve hazır durumu ikisi de bakılır.
        ready = self._combined_ready()
        return {
            "module": "depth_glide_navigator",
            "ready": ready,
            "status": "ok" if ready else "not_initialized",
            "backend": self.backend_name,
            "last_inference_ms": round(float(self._last_inference_ms), 3),
        }

    def get_depth_glide_navigator_version_info(self) -> dict:
        return {"module": "depth_glide_navigator", "version": __version__, "ready": self._combined_ready()}

    def get_depth_glide_navigator_diagnostic_snapshot(self) -> dict:
        # S-0401: Sabit True yok — gerçek prob zincirine delege edilir.
        probe = self._probe_depth_readiness()
        ready = bool(probe.get("passed", False))
        return {
            "module": "depth_glide_navigator",
            "status": "ok" if ready else "not_ready",
            "ready": ready,
            "backend": self.backend_name,
            "probe": probe,
        }



class GlideWaypointController:
    """
    Otomatik Süzülme (Glide) ve Uçurum Atlama Noktası Kontrolcüsü.

    AION 2'de yüksek rakımlı bir kaynaktan (`current_node`) vadi/uçurum aşağısındaki
    bir kaynağa (`next_node`) yürüyerek ~40 saniyede inmek yerine, uçurumdan atlayıp
    kanat açarak (Glide) ~4.5 - 5 saniyede doğrudan kaynağın üzerine inmeyi sağlar.
    """

    # P1-11: süzülme süresi üst sınırı — düğümden gelen
    # `glide_duration_sec` de, hesaplanan süre de bu aralığı aşamaz.
    # (P2-10 da aynı sınıra dokunur; birleşimde tek kaynak kalmalı.)
    MIN_GLIDE_DURATION_SEC: float = 2.5
    MAX_GLIDE_DURATION_SEC: float = 8.5

    def __init__(
        self,
        min_elevation_drop: float = 12.0,
        min_horizontal_dist: float = 22.0,
        max_horizontal_dist: float = 450.0,
        apex_delay_sec: float = 0.35,
        sprint_lead_sec: float = 0.45,
    ) -> None:
        self.min_elevation_drop = float(min_elevation_drop)
        self.min_horizontal_dist = float(min_horizontal_dist)
        self.max_horizontal_dist = float(max_horizontal_dist)
        self.apex_delay_sec = float(apex_delay_sec)
        self.sprint_lead_sec = float(sprint_lead_sec)

    @staticmethod
    def _extract_elevation(node: Optional[Dict[str, Any]]) -> Optional[float]:
        """Düğüm sözlüğünden yükseklik (z / elevation / height / alt) değerini çıkarır."""
        if not isinstance(node, dict):
            return None
        for key in ("elevation", "z", "height", "altitude", "world_z", "pos_z"):
            val = node.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_xy(node: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        """Düğüm sözlüğünden yatay koordinatları (x/y veya map_x/map_y) çıkarır."""
        if not isinstance(node, dict):
            return (0.0, 0.0)
        x = node.get("x", node.get("world_x", node.get("map_x", 0.0)))
        y = node.get("y", node.get("world_y", node.get("map_y", 0.0)))
        try:
            return (float(x or 0.0), float(y or 0.0))
        except (TypeError, ValueError):
            return (0.0, 0.0)

    @staticmethod
    def _is_glide_flag(value: Any) -> bool:
        """P2-10: glide bayrağı katı ayrıştırma. Ham ``bool()`` ``"false"``
        stringini bile ``True`` sayıyordu; yalnızca gerçek ``bool True``,
        ``1`` sayısı veya ``"true"``/``"1"``/``"yes"`` stringi glide
        bayrağı sayılır. ``"false"``/``"0"``/``"no"``/``None`` reddedilir."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return False

    def _pair_supports_glide(
        self,
        current_node: Dict[str, Any],
        next_node: Dict[str, Any],
    ) -> bool:
        """P2-10: açık bayrak bile mesafe/yükseklik kontrolünden geçmek
        zorunda — bayrak, uçurum/mesafe kanıtının yerini tutmaz.
        Yatay mesafe ``[min_horizontal_dist, max_horizontal_dist]``
        içinde olmalı (F-0250: mesafe 0 / eksik koordinat reddedilir);
        iki düğümde de yükseklik biliniyorsa düşüş en az
        ``min_elevation_drop`` olmalı. Yükseklik bilinmiyorsa yalnızca
        mesafe denetlenir (bayrak + mesafe yeter)."""
        x1, y1 = self._extract_xy(current_node)
        x2, y2 = self._extract_xy(next_node)
        horiz_dist = math.hypot(x2 - x1, y2 - y1)
        if not (self.min_horizontal_dist <= horiz_dist <= self.max_horizontal_dist):
            return False
        cur_z = self._extract_elevation(current_node)
        nxt_z = self._extract_elevation(next_node)
        if cur_z is not None and nxt_z is not None:
            if (cur_z - nxt_z) < self.min_elevation_drop:
                return False
        return True

    def is_glide_waypoint(
        self,
        current_node: Optional[Dict[str, Any]],
        next_node: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verilen düğümün veya `(current_node -> next_node)` geçişinin süzülme (Glide)
        noktası olup olmadığını belirler.
        """
        return self.should_use_glide(current_node, next_node)

    def should_use_glide(
        self,
        current_node: Optional[Dict[str, Any]],
        next_node: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        `current_node` (yüksek rakım) -> `next_node` (alçak rakım / vadi ötesi) geçişinin
        Glide kısayolu için uygun olup olmadığını değerlendirir.

        Kriterler:
        1. `current_node` veya `next_node` üzerinde açıkça `is_glide_waypoint=True`,
           `glide=True`, `waypoint_type="glide"` veya `action="glide"` tanımlanmışsa
           VE çift biliniyorsa mesafe/yükseklik kapısı da geçiliyorsa `True`
           (P2-10: açık bayrak kapıyı baypas etmez; bayrak değeri katı
           ayrıştırılır — `"false"` stringi bayrak sayılmaz).
        2. Her iki düğümde yükseklik (`elevation` / `z`) bilgisi varsa ve
           `elevation_drop >= min_elevation_drop` (varsayılan 12m) + yatay mesafe
           uygun aralıktaysa `True`.
        """
        if not isinstance(current_node, dict):
            return False

        # 1. Açık (Explicit) Glide Etiketi Kontrolü
        # P2-10: ham bool() "false" stringini bile True sayıyordu —
        # bayraklar katı ayrıştırma ile okunur; waypoint_type/action
        # karşılaştırması zaten tam-eşleşme ("glide") ister.
        explicit_flags = (
            self._is_glide_flag(current_node.get("is_glide_waypoint")),
            self._is_glide_flag(current_node.get("glide")),
            self._is_glide_flag(current_node.get("use_glide")),
            str(current_node.get("waypoint_type", "")).lower() == "glide",
            str(current_node.get("action", "")).lower() == "glide",
        )
        next_flags = False
        if isinstance(next_node, dict):
            next_flags = (
                self._is_glide_flag(next_node.get("glide_from_prev"))
                or self._is_glide_flag(next_node.get("is_glide_target"))
                or str(next_node.get("approach", "")).lower() == "glide"
            )
        if any(explicit_flags) or next_flags:
            # P2-10: açık bayrak mesafe/yükseklik kontrolünü baypas
            # edemez — çift biliniyorsa çiftin kendisi doğrulanır
            # (mesafe aralığı + bilinen yüksekliklerde düşüş).
            # Tek düğüm sorgusunda doğrulanacak çift yoktur; bayrak
            # tek başına geçer (çağıran çifti verince kapı çalışır).
            if isinstance(next_node, dict):
                return self._pair_supports_glide(current_node, next_node)
            return True

        if isinstance(next_node, dict):
            # 2. Rakım Farkı (Elevation Drop) ve Yatay Vadi Mesafesi Analizi
            cur_z = self._extract_elevation(current_node)
            nxt_z = self._extract_elevation(next_node)
            if cur_z is not None and nxt_z is not None:
                elevation_drop = cur_z - nxt_z
                if elevation_drop >= self.min_elevation_drop:
                    x1, y1 = self._extract_xy(current_node)
                    x2, y2 = self._extract_xy(next_node)
                    horiz_dist = math.hypot(x2 - x1, y2 - y1)
                    # F-0250: Yatay mesafe 0 iken (aynı nokta / eksik koordinat)
                    # glide onaylanmaz — süzülecek mesafe yok.
                    if self.min_horizontal_dist <= horiz_dist <= self.max_horizontal_dist:
                        return True

        return False

    def estimate_glide_parameters(
        self,
        current_node: Dict[str, Any],
        next_node: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        İki düğüm arasındaki hedef yön açısını (`target_heading_deg`), süzülme süresini
        (`glide_duration_sec`) ve kazanılan süreyi (`estimated_time_saved_sec`) hesaplar.
        """
        x1, y1 = self._extract_xy(current_node)
        x2, y2 = self._extract_xy(next_node)
        dx = x2 - x1
        dy = y2 - y1
        horiz_dist = math.hypot(dx, dy)

        # F-1825: `glide_heading_deg=None` (veya sayı-dışı) float() çökertir,
        # eksik yükseklik `or 100.0/60.0` ile uydurulurdu. Güvenli ayrıştırma;
        # bilinmeyen değer uydurulmaz, `None`/bayrak ile raporlanır.
        heading_source = "unknown"
        heading_deg: Optional[float] = None
        raw_heading = current_node.get("glide_heading_deg") if isinstance(current_node, dict) else None
        if raw_heading is not None:
            try:
                heading_deg = float(raw_heading)
                heading_source = "node"
            except (TypeError, ValueError):
                heading_deg = None
        if heading_deg is None and horiz_dist > 1e-3:
            heading_deg = math.degrees(math.atan2(dx, -dy)) % 360.0
            heading_source = "geometry"
        if heading_deg is None:
            heading_deg = 0.0

        cur_z = self._extract_elevation(current_node)
        nxt_z = self._extract_elevation(next_node)
        elevation_known = cur_z is not None and nxt_z is not None
        elevation_drop: Optional[float] = None
        if elevation_known:
            elevation_drop = max(0.0, float(cur_z) - float(nxt_z))

        glide_duration: Optional[float] = None
        if "glide_duration_sec" in current_node:
            try:
                glide_duration = float(current_node["glide_duration_sec"])
            except (TypeError, ValueError):
                glide_duration = None
            # P1-11: sonlu olmayan düğüm süresi (NaN/inf) hesap yoluna düşer.
            if glide_duration is not None and not math.isfinite(glide_duration):
                glide_duration = None
        if glide_duration is None:
            if horiz_dist > 1.0:
                # Süzülme hızı ~22 birim/sn olarak modellenir.
                glide_duration = float(
                    np.clip(
                        horiz_dist / 22.0,
                        self.MIN_GLIDE_DURATION_SEC,
                        self.MAX_GLIDE_DURATION_SEC,
                    )
                )
            else:
                glide_duration = 4.5
        else:
            # P1-11: düğümden gelen süre de üst sınırsız geçemez —
            # [MIN, MAX] aralığına çekilir.
            glide_duration = float(
                np.clip(
                    glide_duration,
                    self.MIN_GLIDE_DURATION_SEC,
                    self.MAX_GLIDE_DURATION_SEC,
                )
            )

        walk_time_est = max(glide_duration * 6.5, 40.0)
        return {
            "should_glide": self.should_use_glide(current_node, next_node),
            "target_heading_deg": round(heading_deg, 2),
            "heading_source": heading_source,
            "glide_duration_sec": round(glide_duration, 2),
            "elevation_drop": round(elevation_drop, 2) if elevation_drop is not None else None,
            "elevation_known": bool(elevation_known),
            "horizontal_distance": round(horiz_dist, 2),
            "estimated_walk_sec": round(walk_time_est, 1),
            "estimated_time_saved_sec": round(max(0.0, walk_time_est - glide_duration), 1),
        }

    def execute_glide_maneuver(
        self,
        input_ctrl: Optional[Any] = None,
        target_heading_deg: float = 0.0,
        glide_duration_sec: float = 4.5,
        test_mode: bool = False,
        current_heading_deg: Optional[float] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Tam Otomatik Uçurum Atlama ve Süzülme (Glide) Manevrasını Yürütür:
        1. `align_cliff_edge`: Hedef uçurum/kaynak açısına (`target_heading_deg`) hizalanır.
        2. `sprint_forward`: İleri (`W`) tuşunu basılı tutarak uçurum kenarına depar atar.
        3. `jump_off_ledge`: Uçurum kenarında sıçrama (`Space`) yapar.
        4. `deploy_wings_apex`: Zıplama tepe noktasında (`0.35s` sonra) ikinci `Space`
           vuruşuyla kanatları (Glide) açar.
        5. `aerial_pitch_and_steer`: `glide_duration_sec` boyunca havada hedef kaynağa
           doğru süzülme açısını ve ileri momentumu korur.
        6. `fold_wings_and_land`: Hedef kaynağın tam üzerinde kanatları kapatıp (`Space`)
           yumuşak iniş yapar ve `W` tuşunu bırakır.

        F-1815: Varsayılan `test_mode=True` üretimde hiçbir girdi göndermeden
        `success=True` döndürüyordu. Varsayılan artık `False`; açıkça simülasyon
        istenirse `test_mode=True` verilir ve sonuç `simulated=True` işaretlenir.
        F-1816: Mutlak hedef açı fare deltasına çevrilemez; `current_heading_deg`
        verilmezse yön düzeltmesi yapılmaz (0 px) ve bu kaydedilir.
        F-1817: `W` tutuşu `try/finally` ile güvenceye alındı; basma dönüş
        değerleri kontrol ediliyor.
        F-1829: `cancel_event` (`.is_set()` olan nesne) her süzülme diliminde
        kontrol edilir; iniş doğrulanamadığı için `landed_at_target` artık
        doğrulanmış iniş yokken `False` + `landing_verified=False` döner.
        P1-11: W tutuşu/tap'i, sıçrama veya kanat açma Space'i
        gönderilemezse manevra durur (`success=False`, `reason` ile);
        `glide_duration_sec` MAX_GLIDE_DURATION_SEC (8.5 sn) ile sınırlıdır.
        """
        t_start = time.perf_counter()
        # P1-11: süreye üst sınır — doğrudan çağrıda verilen
        # glide_duration_sec de MAX_GLIDE_DURATION_SEC'i aşamaz.
        duration = float(glide_duration_sec)
        if not math.isfinite(duration):
            duration = 4.5
        duration = min(self.MAX_GLIDE_DURATION_SEC, max(0.8, duration))
        heading = float(target_heading_deg)
        timeline_events: List[Dict[str, Any]] = []

        def _record_step(step_name: str, action_desc: str, extra: Optional[Dict[str, Any]] = None) -> None:
            entry: Dict[str, Any] = {
                "step": step_name,
                "action": action_desc,
                "elapsed_sec": round(time.perf_counter() - t_start, 4),
            }
            if extra:
                entry.update(extra)
            timeline_events.append(entry)

        # F-1815: Kullanılabilir girdi denetleyicisi yoksa (ör. yanlışlıkla düğüm
        # sözlüğü geçildiyse) beklemeye girmeden hızlıca başarısız dön — sahte
        # başarı ve boş bekleme yok.
        usable_input = False
        if test_mode:
            usable_input = True
        elif input_ctrl is not None:
            usable_input = any(
                hasattr(input_ctrl, name)
                for name in ("_hw_key_down", "_hw_key_up", "press_key", "smooth_move_to", "get_cursor_position")
            )
        if not usable_input:
            logger.warning(
                "[GlideController] Kullanılabilir girdi denetleyicisi yok; süzülme manevrası iptal edildi."
            )
            return {
                "success": False,
                "maneuver": "cliff_jump_glide",
                "target_heading_deg": heading,
                "planned_glide_duration_sec": duration,
                "apex_deploy_delay_sec": self.apex_delay_sec,
                "wings_deployed": False,
                "landed_at_target": False,
                "landing_verified": False,
                "test_mode": bool(test_mode),
                "simulated": bool(test_mode),
                "reason": "no_usable_input_controller",
                "execution_elapsed_sec": round(time.perf_counter() - t_start, 4),
                "timeline": timeline_events,
            }

        logger.info(
            "[GlideController] Süzülme manevrası başlatılıyor -> Yön: %.1f°, Süre: %.2fs (test_mode=%s)",
            heading,
            duration,
            test_mode,
        )

        # Adım 1: Uçurum kenarına ve hedef kaynağa kamera/yön hizalama
        # F-1816: Mutlak açı delta değildir — yalnızca mevcut açı biliniyorsa
        # göreli farktan fare kayması hesaplanır; bilinmiyorsa kayma 0'dır.
        heading_delta_deg: Optional[float] = None
        if current_heading_deg is not None:
            try:
                heading_delta_deg = (heading - float(current_heading_deg) + 180.0) % 360.0 - 180.0
            except (TypeError, ValueError):
                heading_delta_deg = None
        steer_offset_px = int(round(heading_delta_deg * 3.2)) if heading_delta_deg is not None else 0
        if not test_mode and input_ctrl is not None and abs(steer_offset_px) > 4:
            try:
                if hasattr(input_ctrl, "get_cursor_position") and hasattr(
                    input_ctrl, "smooth_move_to"
                ):
                    cx, cy = input_ctrl.get_cursor_position()
                    input_ctrl.smooth_move_to(cx + steer_offset_px, cy, duration=0.08)
            except Exception as exc:
                logger.debug("Glide hizalama fare uyarısı: %s", exc)
        _record_step(
            "align_cliff_edge",
            "aligned_to_cliff_heading",
            {
                "target_heading_deg": heading,
                "current_heading_deg": current_heading_deg,
                "heading_delta_deg": heading_delta_deg,
                "steer_offset_px": steer_offset_px,
            },
        )

        # Adım 2-6: W tutuşu try/finally güvencesinde (F-1817) — manevra
        # iptal edilirse/çökerse bile W bırakılır.
        w_held = False
        w_hold_method: Optional[str] = None
        wings_deployed = False
        cancelled = False
        # P1-11: W/sıçrama/kanat gönderim başarısızlığında manevra durur.
        aborted = False
        abort_reason: Optional[str] = None
        try:
            # Adım 2: Uçurum kenarına ileri depar (W basılı tutma)
            # P1-11: W tutuşu/tap'i başarısızsa manevra DURUR — W'suz
            # depar yok, uçurum kenarına varılmadan sıçranmaz.
            if not test_mode and input_ctrl is not None:
                if hasattr(input_ctrl, "_hw_key_down"):
                    try:
                        w_held = bool(input_ctrl._hw_key_down("w"))
                    except Exception as exc:
                        logger.warning("Glide W tutuşu başarısız: %s", exc)
                        w_held = False
                    if w_held:
                        w_hold_method = "hw_key_down"
                    else:
                        w_hold_method = "hw_key_down_failed"
                        aborted = True
                        abort_reason = "w_hold_failed"
                elif hasattr(input_ctrl, "press_key"):
                    # press_key bas-bırak yapar; tutuş değildir — tap olarak kaydet.
                    tap_result = input_ctrl.press_key("w", min_hold=0.15, max_hold=0.22)
                    if isinstance(tap_result, (int, float)) and tap_result <= 0.0:
                        logger.warning("Glide W tap gönderilemedi (press_key 0.0 döndü).")
                        aborted = True
                        abort_reason = "w_tap_failed"
                    w_hold_method = "press_key_tap"
                else:
                    # W gönderecek yöntem yok — ileri momentum imkânsız.
                    logger.warning("Glide W gönderilemedi: girdi denetleyicisinde W yöntemi yok.")
                    aborted = True
                    abort_reason = "no_w_dispatch_method"
                if not aborted:
                    time.sleep(self.sprint_lead_sec)
            _record_step(
                "sprint_forward",
                "key_down('w')" if w_held else "key_tap_or_simulated('w')",
                {"sprint_lead_sec": self.sprint_lead_sec, "w_held": w_held, "w_hold_method": w_hold_method},
            )

            # Adım 3: Uçurum kenarından sıçrama (İlk Space)
            # P1-11: sıçrama gönderilemezse manevra DURUR.
            if not aborted and not test_mode and input_ctrl is not None:
                if hasattr(input_ctrl, "press_key"):
                    jump_result = input_ctrl.press_key("space", min_hold=0.045, max_hold=0.075)
                    if isinstance(jump_result, (int, float)) and jump_result <= 0.0:
                        logger.warning("Glide sıçrama Space gönderilemedi (press_key 0.0 döndü).")
                        aborted = True
                        abort_reason = "jump_failed"
                else:
                    logger.warning("Glide sıçrama gönderilemedi: press_key yok.")
                    aborted = True
                    abort_reason = "jump_no_press_key"
            _record_step(
                "jump_off_ledge",
                "press_key('space') [jump]" if not aborted else "aborted: jump not dispatched",
                {"abort_reason": abort_reason} if aborted else None,
            )

            # Adım 4: Sıçrama tepe noktası (Apex = 0.35s) beklemesi ve Kanat Açma (İkinci Space)
            # P1-11: kanat açma Space'i gönderilemezse manevra DURUR —
            # kanatsız süzülme yoktur, başarı sayılmaz.
            if not aborted and not test_mode:
                time.sleep(self.apex_delay_sec)
                if input_ctrl is not None and hasattr(input_ctrl, "press_key"):
                    wings_result = input_ctrl.press_key("space", min_hold=0.045, max_hold=0.075)
                    if isinstance(wings_result, (int, float)) and wings_result <= 0.0:
                        logger.warning("Glide kanat açma Space gönderilemedi (press_key 0.0 döndü).")
                        aborted = True
                        abort_reason = "wings_deploy_failed"
                    else:
                        wings_deployed = True
                else:
                    logger.warning("Glide kanat açma gönderilemedi: press_key yok.")
                    aborted = True
                    abort_reason = "wings_no_press_key"
            elif not aborted:
                wings_deployed = True  # simülasyon
            _record_step(
                "deploy_wings_apex",
                "press_key('space') [wings_deployed_after_0.35s]",
                {"apex_delay_sec": self.apex_delay_sec, "wings_deployed": wings_deployed},
            )

            # Adım 5: Havada süzülme, irtifa/pitch koruma ve mikro yön düzeltmeleri
            # F-1829: Bu adım artık gerçekten W tutuşunu korur (canlıda) ve
            # iptal bayrağını her dilimde kontrol eder; girdi gönderilmeden
            # "irtifa koruyor" kaydedilmez.
            glide_segments = max(2, min(6, int(round(duration))))
            segment_sleep = duration / float(glide_segments)
            for seg_idx in range(glide_segments):
                # P1-11: W/sıçrama/kanat başarısızlığında süzülme
                # dilimlerine girilmez — manevra durduruldu.
                if aborted:
                    _record_step(
                        "aerial_glide_hold",
                        "aborted_before_glide_hold",
                        {"segment": seg_idx + 1, "abort_reason": abort_reason},
                    )
                    break
                if cancel_event is not None:
                    try:
                        if bool(cancel_event.is_set()):
                            cancelled = True
                            _record_step(
                                "aerial_glide_hold",
                                "cancelled_by_cancel_event",
                                {"segment": seg_idx + 1},
                            )
                            break
                    except Exception:
                        pass
                if not test_mode:
                    time.sleep(segment_sleep)
                _record_step(
                    "aerial_glide_hold",
                    (
                        "maintaining_pitch_and_heading_segment_%d/%d (w_held=%s)"
                        % (seg_idx + 1, glide_segments, w_held)
                        if not test_mode
                        else "simulated_pitch_and_heading_segment_%d/%d" % (seg_idx + 1, glide_segments)
                    ),
                    {
                        "segment": seg_idx + 1,
                        "planned_segment_sec": round(segment_sleep, 3),
                        "w_held": w_held,
                        "input_sent": bool(not test_mode and w_held),
                    },
                )
                if cancelled:
                    break

            # Adım 6: Hedef kaynak üzerinde kanatları kapatma (Space) ve ileri tuşunu (W) bırakma
            # P1-11: durdurulan manevrada iniş adımı da çalışmaz.
            if not cancelled and not aborted:
                if not test_mode and input_ctrl is not None:
                    if hasattr(input_ctrl, "press_key"):
                        fold_result = input_ctrl.press_key("space", min_hold=0.045, max_hold=0.075)
                        if isinstance(fold_result, (int, float)) and fold_result <= 0.0:
                            logger.warning("Glide kanat kapatma Space gönderilemedi (press_key 0.0 döndü).")
                _record_step(
                    "fold_wings_and_land",
                    "press_key('space') [fold_wings] + key_up('w') [land_at_node]",
                )
        finally:
            # F-1817: W tutulduysa her durumda bırak (iptal/hata dahil).
            if w_held and input_ctrl is not None and hasattr(input_ctrl, "_hw_key_up"):
                try:
                    input_ctrl._hw_key_up("w")
                except Exception as exc:
                    logger.warning("Glide W bırakma hatası: %s", exc)
                w_held = False

        elapsed_total = round(time.perf_counter() - t_start, 4)
        logger.info(
            "[GlideController] Süzülme manevrası tamamlandı (Adım sayısı: %d, Süre: %.3fs, iptal=%s, durdurma=%s).",
            len(timeline_events),
            elapsed_total,
            cancelled,
            abort_reason,
        )

        return {
            # F-1815/F-1829: Başarı yalnızca manevra gerçekten yürütülüp
            # iptal edilmediyse; P1-11: W/sıçrama/kanat gönderim
            # başarısızlığı da manevrayı durdurur (success=False).
            # İniş doğrulanamadığı için landed_at_target her zaman False
            # + landing_verified=False (doğrulama yok).
            "success": bool(not cancelled and not aborted),
            "maneuver": "cliff_jump_glide",
            "target_heading_deg": heading,
            "planned_glide_duration_sec": duration,
            "apex_deploy_delay_sec": self.apex_delay_sec,
            "wings_deployed": bool(wings_deployed),
            "landed_at_target": False,
            "landing_verified": False,
            "cancelled": bool(cancelled),
            "reason": abort_reason,
            "test_mode": bool(test_mode),
            "simulated": bool(test_mode),
            "execution_elapsed_sec": elapsed_total,
            "timeline": timeline_events,
        }

    def _probe_glide_readiness(self) -> Dict[str, Any]:
        """
        S-0602: Gerçek hazır probu — eşik sağlığı + sentetik glide kararı
        ölçülür, sabit True döndürülmez (Karar 3: proba bağlama).
        Pozitif kontrol: yüksek->alçak sentetik çift glide olmalı;
        negatif kontrol: düz (eş yükseltili) çift glide olmamalı.
        """
        checks: Dict[str, Any] = {}
        try:
            checks["thresholds_sane"] = bool(
                self.min_elevation_drop > 0
                and self.min_horizontal_dist > 0
                and self.max_horizontal_dist > self.min_horizontal_dist
                and self.apex_delay_sec >= 0
                and self.sprint_lead_sec >= 0
            )
            high = {"x": 0.0, "y": 0.0, "elevation": 100.0}
            low = {"x": 100.0, "y": 0.0, "elevation": 50.0}
            checks["synthetic_glide_decision"] = bool(self.should_use_glide(high, low))
            flat = {"x": 100.0, "y": 0.0, "elevation": 100.0}
            checks["flat_pair_rejected"] = not bool(self.should_use_glide(high, flat))
            params = self.estimate_glide_parameters(high, low)
            checks["glide_estimate_sane"] = bool(
                params.get("should_glide")
                and params.get("glide_duration_sec") is not None
                and 0 < float(params["glide_duration_sec"]) <= 8.5
                and float(params.get("estimated_time_saved_sec") or 0) > 0
            )
        except Exception as exc:
            checks["error"] = str(exc)
        checks["passed"] = bool(checks.get("thresholds_sane")) and all(
            bool(v) for k, v in checks.items() if k not in ("error", "passed")
        )
        return checks

    def get_navigator_telemetry_summary(self) -> dict:
        """Returns a telemetry summary of the navigator state and stats safely."""
        summary = {
            # S-0602: Sabit True yok — bu örneğin gerçek prob sonucu.
            "navigator_ready": bool(self._probe_glide_readiness().get("passed", False)),
            "glide_enabled": True,
        }
        try:
            stats = {}
            for k, v in self.__dict__.items():
                if isinstance(v, (int, float, str, bool, list, dict)):
                    stats[k] = v
            summary["stats"] = stats
        except Exception as e:
            summary["stats_error"] = str(e)
        return summary

    def get_depth_glide_health_summary(self) -> dict:
        try:
            # S-0602: Sabit True yok — bu örneğin gerçek prob sonucu.
            return {"depth_glide_ready": bool(self._probe_glide_readiness().get("passed", False))}
        except Exception:
            return {"depth_glide_ready": False}

    def _config_valid(self) -> bool:
        """Yapılandırma tutarlılığı: eşikler pozitif ve mesafe aralığı sıralı."""
        try:
            return bool(
                self.min_elevation_drop > 0.0
                and 0.0 < self.min_horizontal_dist < self.max_horizontal_dist
                and self.apex_delay_sec > 0.0
                and self.sprint_lead_sec >= 0.0
            )
        except Exception:
            return False

    def _glide_not_ready_status(self) -> str:
        """Hazır-olmama ayrımı (R04 S-0602 + D10 — iki testin de pinlediği
        girdiler korunur):

        - "misconfigured": mesafe aralığı kendi içinde çelişkili
          (min_horizontal_dist >= max_horizontal_dist) — hiçbir girdi
          aralığı sağlayamaz; D10 bunu pinler
          (test_glide_ready_flag_rejects_misconfigured_thresholds).
        - "not_ready": eşikler dejenere (pozitif olmayan düşüş/mesafe,
          pozitif olmayan apex gecikmesi) — gerçek bir glide tarif
          edilemez; R04/S-0602 bunu pinler
          (test_degenerate_thresholds_report_not_ready, örn.
          min_elevation_drop=-5.0).
        """
        if self.min_horizontal_dist >= self.max_horizontal_dist:
            return "misconfigured"
        return "not_ready"

    def get_depth_glide_navigator_ready_flag(self) -> dict:
        # S-0602 + D10: Sabit True yok — gerçek prob zinciri VE eşik/mesafe
        # yapılandırma tutarlılığı ikisi de geçmeli.
        probe_ready = bool(self._probe_glide_readiness().get("passed", False))
        config_ready = self._config_valid()
        ready = probe_ready and config_ready
        status = (
            "ok"
            if ready
            else (self._glide_not_ready_status() if not config_ready else "not_ready")
        )
        return {
            "module": "depth_glide_navigator",
            "ready": ready,
            "status": status,
            "min_elevation_drop": self.min_elevation_drop,
            "min_horizontal_dist": self.min_horizontal_dist,
            "max_horizontal_dist": self.max_horizontal_dist,
        }

    def get_depth_glide_navigator_version_info(self) -> dict:
        ready = bool(self._probe_glide_readiness().get("passed", False)) and self._config_valid()
        return {"module": "depth_glide_navigator", "version": __version__, "ready": ready}

    def get_depth_glide_navigator_diagnostic_snapshot(self) -> dict:
        probe_ready = bool(self._probe_glide_readiness().get("passed", False))
        config_ready = self._config_valid()
        ready = probe_ready and config_ready
        status = (
            "ok"
            if ready
            else (self._glide_not_ready_status() if not config_ready else "not_ready")
        )
        return {"module": "depth_glide_navigator", "status": status, "ready": ready}


def get_depth_glide_navigator_capabilities() -> list:
    return ["depth_glide_navigator", "vertical_glide_control", "diagnostic_snapshot"]


def _run_depth_glide_navigator_contract_checks() -> dict:
    """
    S-0602: 'hazır/audit/certified' zinciri artık gerçek ölçüm yapar —
    varsayılan GlideWaypointController üzerinde eşik + sentetik glide
    kararı probu çalıştırılır (sabit True / boş-olmayan liste yok).
    """
    checks: Dict[str, Any] = {}
    try:
        checks["capabilities_present"] = bool(get_depth_glide_navigator_capabilities())
        probe = GlideWaypointController()._probe_glide_readiness()
        for key, value in probe.items():
            if key != "passed":
                checks["glide_" + key] = value
        checks["passed"] = bool(checks["capabilities_present"]) and bool(probe.get("passed", False))
    except Exception as exc:
        checks["error"] = str(exc)
        checks["passed"] = False
    return checks


def validate_depth_glide_navigator_contract() -> bool:
    """Gerçek prob: sınıflar kurulur, tahminci sentetik karede çalışır, glide mantığı cevap verir.

    D10: önceden yalnızca yetenek listesinin boş olmadığını döndürüyordu
    (totoloji — liste sabitti, her zaman True). Artık kurulum + çıkarım +
    glide kararı gerçekten denenir; herhangi biri başarısız olursa False.
    S-0602 zinciri de ayrıca çalışır (yetenekler + GlideWaypointController probu).
    """
    contract_ok = bool(_run_depth_glide_navigator_contract_checks().get("passed", False))
    d10_ok = _validate_depth_glide_d10_probe()
    return bool(contract_ok and d10_ok)


def _validate_depth_glide_d10_probe() -> bool:
    """D10 inline probu: sentetik karede derinlik çıkarımı + glide kararı pozitif/negatif kontrol."""
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            nav = MonocularDepthObstacleAvoidance(model_dir=tmp, work_size=(64, 36))
            depth = nav.estimate_depth_map(np.zeros((72, 128, 3), dtype=np.uint8))
            if depth.shape != (36, 64) or depth.dtype != np.float32:
                return False
            if not bool(np.all((depth >= 0.0) & (depth <= 1.0))):
                return False
        ctrl = GlideWaypointController()
        if not ctrl.should_use_glide(
            {"x": 0.0, "y": 0.0, "elevation": 100.0},
            {"x": 100.0, "y": 0.0, "elevation": 50.0},
        ):
            return False
        if ctrl.should_use_glide(
            {"x": 0.0, "y": 0.0, "elevation": 50.0},
            {"x": 100.0, "y": 0.0, "elevation": 50.0},
        ):
            return False
        return True
    except Exception:
        return False


def _depth_glide_navigator_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. telemetry.py'deki aciklama).

    ``validate_depth_glide_navigator_contract()`` her cagrida iki agir
    prob kosar: GlideWaypointController sentetik probu +
    ``_validate_depth_glide_d10_probe`` (gecici dizin + MonocularDepth
    kurulumu + sentetik cikarim + glide karari). Zincir bunu ust duzey
    okuma basina bir kez yapar. Cagrilar arasi onbellek yoktur.
    """
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_depth_glide_navigator_contract())


def get_depth_glide_navigator_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "depth_glide_navigator", "contract_valid": _depth_glide_navigator_contract_ok(_contract_ok), "capability_count": len(get_depth_glide_navigator_capabilities())}


def get_depth_glide_navigator_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "ready": ok,
        "metrics": get_depth_glide_navigator_baseline_metrics(_contract_ok=ok),
    }


def get_depth_glide_navigator_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "audit_passed": ok,
        "readiness": get_depth_glide_navigator_readiness_status(_contract_ok=ok),
    }


def get_depth_glide_navigator_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "operational": ok,
        "audit": get_depth_glide_navigator_audit_record(_contract_ok=ok),
    }


def get_depth_glide_navigator_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    return {
        "module": "depth_glide_navigator",
        "payload": get_depth_glide_navigator_operational_digest(_contract_ok=_contract_ok),
    }


def get_depth_glide_navigator_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "sealed": ok,
        "telemetry": get_depth_glide_navigator_telemetry_payload(_contract_ok=ok),
    }


def get_depth_glide_navigator_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "certified": ok,
        "seal": get_depth_glide_navigator_integrity_seal(_contract_ok=ok),
    }


def get_depth_glide_navigator_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "ready": ok,
        "certificate": get_depth_glide_navigator_compliance_certificate(_contract_ok=ok),
    }


def get_depth_glide_navigator_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _depth_glide_navigator_contract_ok(_contract_ok)
    return {
        "module": "depth_glide_navigator",
        "audit_passed": ok,
        "probe": get_depth_glide_navigator_readiness_probe(_contract_ok=ok),
    }


