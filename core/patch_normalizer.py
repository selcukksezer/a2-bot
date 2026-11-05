# -*- coding: utf-8 -*-
"""
50-Nokta Üretim Spesifikasyonu — Madde 6, 7, 8, 9 ve 10:
Yama (Patch) Bozulma Koruması, Sobel+CLAHE Aydınlatma Normalizasyonu,
2K (2560x1440) Sıfır-Kopya ROI Hızlandırıcı, Watershed Kümelenmiş İkon Ayırıcı
ve Laplacian Hareket Bulanıklığı (Motion Blur) Kapısı (core/patch_normalizer.py).

Kapsanan 5 Kritik Alt Sistem:
1. Madde 6 — Kamera Açısı Kaynaklı Yama (Patch) Bozulma Koruması (`MinimapOrientationGuard`):
   - Sağ üstteki minimap'in Sabit-Kuzey (Fixed-North) modunda mı yoksa Dönen (Rotating)
     modda mı olduğunu tespit eder (`verify_fixed_north_minimap(frame) -> dict`).
   - Minimap başlık çubuğunu (`Pilgrim's Ridge`), yatay/dikey kenar hizalamasını ve
     sağ alttaki yön değiştirme (`⇄` / pusula) göstergesini analiz eder.
   - `normalize_patch_to_north(patch_bgr, camera_yaw_deg=0.0) -> np.ndarray` ile
     80x80 düğüm yamalarını (`node_XX.png`) kamera dönüş açısına karşı kuzey-yukarı
     (north-up) kanonik yönelime döndürür ve kenar artefaktı olmadan kırpar.

2. Madde 7 — Gölgeli ve Gece/Gündüz Harita Kesiti Normalizasyonu (`SobelClahePatchNormalizer`):
   - 80x80 harita/minimap yamalarını (`normalize_patch(patch_bgr) -> np.ndarray`)
     CLAHE (`clipLimit=2.5, tileGridSize=(4, 4)`) + Sobel Kenar Büyüklüğü
     (`cv2.Sobel` dx/dy, `%65 CLAHE + %35 Sobel edges` harmanı) ile normalize eder.
   - Gece/gündüz, bulut gölgesi ve ortam ışığı değişimlerinde şablon eşleştirme
     skorlarının `0.53` veya `0.61` seviyesine düşmesini önler
     (`match_normalized_patch(search_roi_bgr, template_bgr) -> Tuple[float, Tuple[int, int]]`).

3. Madde 8 — Yüksek Çözünürlük (`2560×1440`) İçin Alt-Bölge (ROI) Kırpma (`HighResROICropper`):
   - 2K (`2560x1440`), 1080p (`1920x1080`) ve 2000px referans ekranlarda sıfır-kopya
     (`NumPy view`, `copy=False`) dilimleme yardımcıları sağlar:
     * `crop_f_gather_roi(frame)` -> `(x=0.45, y=0.40, w=0.25, h=0.25)`
     * `crop_3d_crest_horizon_roi(frame)` -> `(x=0.15, y=0.18, w=0.70, h=0.52)`
     * `crop_minimap_roi(frame)` -> `(x=0.855, y=0.065, w=0.137, h=0.190)`
   - CPU piksel tarama yükünü %80'in üzerinde azaltır ve yerel ROI koordinatlarını
     tam ekran koordinatlarına geri dönüştüren yardımcılar sunar.

4. Madde 9 — Üst Üste Binen (Clustered) Minimap İkonlarını Ayırma (`WatershedClusterSeparator`):
   - Minimap üzerinde 2 veya 3 adet 3-yapraklı filiz (`sprout`) ikonu birbirine değip
     tek bir birleşik leke (connected component blob) oluşturduğunda, Öklid Mesafe
     Dönüşümü (`cv2.distanceTransform`) + yerel maksimum tohumları + **Watershed
     (`cv2.watershed`)** algoritmasıyla her bir dokunan ikonun bağımsız merkez
     koordinatlarını `[(cx1, cy1), (cx2, cy2), ...]` hassasiyetle ayırır.

5. Madde 10 — Hareket Bulanıklığı (Motion Blur) Filtresi (`MotionBlurGate`):
   - Kare keskinliğini **Laplacian varyansı (`cv2.Laplacian(gray, cv2.CV_64F).var()`)**
     ile ölçer (`is_frame_blurry(frame, threshold=45.0) -> Tuple[bool, float]`).
   - Binek (mount) ile hızlı koşarken veya ani kamera dönüşlerinde yakalanan bulanık
     karelerin otomatik düğüm kaydını (auto-node recording) bozmasını ve yanlış
     pozitif üretmesini engeller.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

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


# Kanonik düğüm yaması (patch) boyutu: 80x80
CANONICAL_PATCH_SIZE: int = 80

# Standart normalize ROI dikdörtgenleri (x, y, w, h)
ROI_F_GATHER_NORM: Tuple[float, float, float, float] = (0.45, 0.40, 0.25, 0.25)
ROI_3D_CREST_HORIZON_NORM: Tuple[float, float, float, float] = (0.15, 0.18, 0.70, 0.52)
ROI_MINIMAP_NORM: Tuple[float, float, float, float] = (0.855, 0.065, 0.137, 0.190)
ROI_MINIMAP_INNER_NORM: Tuple[float, float, float, float] = (0.858, 0.088, 0.132, 0.158)


@dataclass(frozen=True)
class ROISliceMeta:
    """
    Sıfır-kopya (zero-copy) kırpılmış alt-bölge (ROI) meta verisi.
    Yerel ROI koordinatlarını tam ekran piksel koordinatlarına geri çevirmek için kullanılır.
    """

    name: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    frame_width: int
    frame_height: int
    norm_rect: Tuple[float, float, float, float]

    @property
    def pixel_reduction_ratio(self) -> float:
        """Tam ekrana kıyasla taranmayan (tasarruf edilen) piksel oranı (0.0 .. 1.0)."""
        total_px = max(1, self.frame_width * self.frame_height)
        roi_px = max(0, self.width * self.height)
        return float(max(0.0, 1.0 - (roi_px / float(total_px))))

    def to_full_screen(self, local_x: Union[int, float], local_y: Union[int, float]) -> Tuple[int, int]:
        """ROI içindeki yerel `(local_x, local_y)` koordinatını tam ekran `(screen_x, screen_y)` koordinatına çevirir."""
        return (
            int(round(self.x_offset + float(local_x))),
            int(round(self.y_offset + float(local_y))),
        )


# ==============================================================================
# MADDE 6: Kamera Açısı Kaynaklı Yama (Patch) Bozulma Koruması
# ==============================================================================


class MinimapOrientationGuard:
    """
    Madde 6 — Kamera Açısı Kaynaklı Yama (Patch) Bozulma Koruması.

    Sağ üstteki minimap'in Sabit-Kuzey (Fixed-North) modunda mı yoksa Dönen (Rotating)
    modda mı olduğunu doğrular ve `node_XX.png` (80x80) yamalarını kamera yaw açısına
    göre Kuzey-Yukarı (North-Up) kanonik düzleme normalize eder.
    """

    DEFAULT_MINIMAP_NORM_RECT: Tuple[float, float, float, float] = ROI_MINIMAP_NORM
    CANONICAL_SIZE: int = CANONICAL_PATCH_SIZE

    def __init__(
        self,
        minimap_norm_rect: Tuple[float, float, float, float] = DEFAULT_MINIMAP_NORM_RECT,
        canonical_size: int = CANONICAL_SIZE,
    ) -> None:
        self.minimap_norm_rect: Tuple[float, float, float, float] = (
            float(minimap_norm_rect[0]),
            float(minimap_norm_rect[1]),
            float(minimap_norm_rect[2]),
            float(minimap_norm_rect[3]),
        )
        self.canonical_size: int = max(16, int(canonical_size))

    def _extract_minimap_candidate(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Girdi tam ekran bir kareyse sağ üst minimap bölgesini kırpar;
        zaten kırpılmış bir minimap yakın çekimiyse (örn. 365x309) doğrudan döndürür.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame boş veya None olamaz.")

        h, w = frame.shape[:2]
        # Tam ekran kare (örn. 1280x720, 1920x1080, 2000x1124, 2560x1440)
        if w >= 800 and h >= 500:
            nx, ny, nw, nh = self.minimap_norm_rect
            x1 = max(0, min(w - 1, int(round(nx * w))))
            y1 = max(0, min(h - 1, int(round(ny * h))))
            x2 = max(x1 + 1, min(w, int(round((nx + nw) * w))))
            y2 = max(y1 + 1, min(h, int(round((ny + nh) * h))))
            return frame[y1:y2, x1:x2], True
        return frame, False

    def verify_fixed_north_minimap(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Minimap'in Sabit-Kuzey (Fixed-North) modunda olup olmadığını analiz eder.

        Analiz Kriterleri:
        1. Minimap üst başlık ayırıcı çizgisi ve dikdörtgen çerçeve hizalaması (`axis_alignment_score`):
           Sobel yatay/dikey gradyan enerjisinin diyagonal gradyan enerjisine oranı.
           Sabit-Kuzey kutu minimap'te güçlü yatay/dikey çerçeve çizgileri bulunur.
        2. Sağ alt mod değiştirme (`⇄` sabit-kuzey / pusula) ikonu ve başlık çubuğu (`header_bar_score`):
           Üst %18'lik karanlık başlık şeridi (`Pilgrim's Ridge`) ile altındaki harita
           tuvali arasındaki yatay geçiş kontrastı ve sağ alt köşedeki beyaz `⇄` sembolü.
        3. Merkez oyuncu oku / görüş konisi (`view_cone_detected`):
           Sabit-Kuzey modunda harita sabit kalırken merkezdeki oyuncu oku ve sarı/camgöbeği
           görüş dairesi kamera açısına göre döner.

        Dönüş:
            {
                "is_fixed_north": bool,
                "mode": "fixed_north" | "rotating",
                "confidence": float,
                "estimated_map_rotation_deg": float,
                "axis_alignment_score": float,
                "header_bar_score": float,
                "toggle_icon_score": float,
                "player_heading_deg": Optional[float],
                "is_fullscreen_input": bool,
            }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "is_fixed_north": False,
                "mode": "rotating",
                "confidence": 0.0,
                "estimated_map_rotation_deg": 0.0,
                "axis_alignment_score": 0.0,
                "header_bar_score": 0.0,
                "toggle_icon_score": 0.0,
                "player_heading_deg": None,
                "is_fullscreen_input": False,
            }

        mm_roi, is_fullscreen = self._extract_minimap_candidate(frame)
        if mm_roi.ndim == 3:
            gray = cv2.cvtColor(mm_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = mm_roi.copy()

        mh, mw = gray.shape[:2]
        if mh < 24 or mw < 24:
            return {
                "is_fixed_north": True,
                "mode": "fixed_north",
                "confidence": 0.5,
                "estimated_map_rotation_deg": 0.0,
                "axis_alignment_score": 0.5,
                "header_bar_score": 0.5,
                "toggle_icon_score": 0.0,
                "player_heading_deg": 0.0,
                "is_fullscreen_input": is_fullscreen,
            }

        # 1. Yatay ve dikey çerçeve / başlık çizgisi hizalaması (Axis Alignment Score)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        abs_gx = np.abs(gx)
        abs_gy = np.abs(gy)

        # Üst başlık ayırıcı çizgisi (y: %10 .. %26 arası) güçlü yatay gradyan üretir
        y_hdr1 = max(1, int(round(mh * 0.08)))
        y_hdr2 = max(y_hdr1 + 2, min(mh - 1, int(round(mh * 0.26))))
        header_band_gy = abs_gy[y_hdr1:y_hdr2, :]
        row_peaks = np.mean(header_band_gy, axis=1)
        max_row_grad = float(np.max(row_peaks)) if row_peaks.size > 0 else 0.0
        mean_global_grad = float(np.mean(abs_gy) + 1e-5)
        header_line_ratio = min(1.0, max_row_grad / max(15.0, mean_global_grad * 2.2))

        # Kardinal (0/90 derece) gradyan enerjisi vs diyagonal enerji
        cardinal_energy = float(np.mean(np.maximum(abs_gx, abs_gy)))
        total_energy = float(np.mean(np.hypot(gx, gy)) + 1e-5)
        # Tam eksen hizalı kenarlarda max(|gx|,|gy|) / hypot(gx,gy) -> 1.0, 45 derecede -> 0.7071
        raw_axis_ratio = cardinal_energy / total_energy
        axis_alignment_score = float(np.clip((raw_axis_ratio - 0.72) / (0.92 - 0.72), 0.0, 1.0))

        # 2. Üst karanlık başlık şeridi ("Pilgrim's Ridge") kontrastı
        top_strip = gray[: max(2, int(round(mh * 0.18))), :]
        mid_canvas = gray[int(round(mh * 0.25)) : int(round(mh * 0.75)), :]
        top_mean = float(np.mean(top_strip))
        mid_mean = float(np.mean(mid_canvas)) if mid_canvas.size > 0 else top_mean
        # Başlık şeridi genellikle koyu yarı saydamdır (top_mean < mid_mean) veya parlak beyaz yazı içerir
        top_bright_text_ratio = float(np.mean(top_strip > 185))
        has_dark_header_with_text = (top_mean < mid_mean + 15.0) and (0.005 <= top_bright_text_ratio <= 0.35)
        header_bar_score = float(
            np.clip(
                0.60 * header_line_ratio + (0.40 if has_dark_header_with_text else 0.15),
                0.0,
                1.0,
            )
        )

        # 3. Sağ alt köşe mod değiştirme (`⇄` sabit-kuzey / saat yanındaki ikon) kontrolü
        bx1 = max(0, int(round(mw * 0.78)))
        by1 = max(0, int(round(mh * 0.78)))
        br_corner = gray[by1:, bx1:]
        if br_corner.size > 0:
            bright_icon_ratio = float(np.mean(br_corner > 190))
            toggle_icon_score = float(np.clip(bright_icon_ratio / 0.04, 0.0, 1.0))
        else:
            toggle_icon_score = 0.0

        # 4. Merkezdeki oyuncu yön oku (cyan/beyaz ok) açısını tahmin et
        player_heading_deg = self._estimate_player_arrow_heading(mm_roi)

        # Toplam Sabit-Kuzey güven skoru
        confidence = float(
            np.clip(
                0.45 * header_bar_score + 0.35 * axis_alignment_score + 0.20 * toggle_icon_score,
                0.0,
                1.0,
            )
        )
        is_fixed_north = bool(confidence >= 0.42)

        return {
            "is_fixed_north": is_fixed_north,
            "mode": "fixed_north" if is_fixed_north else "rotating",
            "confidence": round(confidence, 4),
            "estimated_map_rotation_deg": 0.0 if is_fixed_north else float(player_heading_deg or 0.0),
            "axis_alignment_score": round(axis_alignment_score, 4),
            "header_bar_score": round(header_bar_score, 4),
            "toggle_icon_score": round(toggle_icon_score, 4),
            "player_heading_deg": round(player_heading_deg, 2) if player_heading_deg is not None else None,
            "is_fullscreen_input": is_fullscreen,
        }

    def _estimate_player_arrow_heading(self, mm_roi: np.ndarray) -> Optional[float]:
        """
        Minimap merkezindeki oyuncu yön okunun (camgöbeği/beyaz ok) baktığı açıyı
        (0° = Kuzey, 90° = Doğu, 180° = Güney, 270° = Batı) hesaplar.
        """
        if mm_roi is None or mm_roi.size == 0:
            return None
        mh, mw = mm_roi.shape[:2]
        cx = int(round(mw * 0.48))
        cy = int(round(mh * 0.56))
        rad = max(10, int(round(min(mw, mh) * 0.11)))

        x1 = max(0, cx - rad)
        y1 = max(0, cy - rad)
        x2 = min(mw, cx + rad)
        y2 = min(mh, cy + rad)
        center_crop = mm_roi[y1:y2, x1:x2]
        if center_crop.size == 0:
            return None

        if center_crop.ndim == 3:
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            # Oyuncu oku parlak beyaz + camgöbeği (cyan) kenarlıdır
            cyan_mask = cv2.inRange(hsv, np.array([75, 40, 150], dtype=np.uint8), np.array([115, 255, 255], dtype=np.uint8))
            white_mask = cv2.inRange(hsv, np.array([0, 0, 210], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))
            arrow_mask = cv2.bitwise_or(cyan_mask, white_mask)
        else:
            _, arrow_mask = cv2.threshold(center_crop, 200, 255, cv2.THRESH_BINARY)

        ys, xs = np.nonzero(arrow_mask)
        if xs.size < 6:
            return 0.0

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        dx = xs.astype(np.float32) - mean_x
        dy = ys.astype(np.float32) - mean_y
        dists = np.hypot(dx, dy)
        # En uçtaki %25'lik piksel grubu okun sivri ucunu (tip) gösterir
        q75 = float(np.percentile(dists, 75))
        tip_mask = dists >= max(1.0, q75)
        if not np.any(tip_mask):
            return 0.0

        tip_dx = float(np.mean(dx[tip_mask]))
        tip_dy = float(np.mean(dy[tip_mask]))
        # Kuzey (yukarı: dy < 0, dx = 0) = 0°, Doğu (sağ: dx > 0, dy = 0) = 90°
        angle_rad = math.atan2(tip_dx, -tip_dy)
        heading_deg = (math.degrees(angle_rad) + 360.0) % 360.0
        return float(heading_deg)

    def normalize_patch_to_north(
        self,
        patch_bgr: np.ndarray,
        camera_yaw_deg: float = 0.0,
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        80x80 düğüm yamasını (`node_XX.png`) kamera yaw dönüş açısına (`camera_yaw_deg`)
        karşı ters yönde döndürerek her zaman **Kuzey-Yukarı (Fixed-North)** kanonik
        düzleminde tutar.

        - `camera_yaw_deg == 0.0` (veya Sabit-Kuzey minimap'ten alınmış yama) ise
          döndürme bulanıklığı eklemeden doğrudan hedef boyuta (`80x80`) getirir.
        - Dönen minimap'ten alınmış bir yamada `camera_yaw_deg != 0.0` verildiğinde,
          yamayı merkez etrafında `-camera_yaw_deg` döndürür, `BORDER_REFLECT_101` ile
          siyah köşe artefaktlarını önler ve `80x80` kanonik boyutta döndürür.
        """
        if patch_bgr is None or not isinstance(patch_bgr, np.ndarray) or patch_bgr.size == 0:
            raise ValueError("patch_bgr boş veya None olamaz.")

        out_size = int(target_size if target_size is not None else self.canonical_size)
        ph, pw = patch_bgr.shape[:2]

        # Açı 0 dereceye çok yakınsa interpolasyon kaybı olmadan doğrudan boyutlandır
        norm_yaw = float(camera_yaw_deg) % 360.0
        if norm_yaw < 0.25 or (360.0 - norm_yaw) < 0.25:
            if ph == out_size and pw == out_size:
                return patch_bgr.copy()
            return cv2.resize(patch_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

        # Merkez etrafında Kuzey-Yukarı hizasına döndür
        center = (pw / 2.0, ph / 2.0)
        # OpenCV getRotationMatrix2D pozitif açıyı saat yönünün tersine (CCW) döndürür.
        # Kamera saat yönünde +camera_yaw_deg döndüğünde harita CCW döner; kuzeye geri
        # hizalamak için rotasyon matrisi açısı = +camera_yaw_deg kullanılır.
        rot_mat = cv2.getRotationMatrix2D(center, float(norm_yaw), 1.0)
        rotated = cv2.warpAffine(
            patch_bgr,
            rot_mat,
            (pw, ph),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Eğer girdi yaması 80x80'den büyükse (örn. döndürme payı ile 114x114 kırpılmışsa),
        # merkezdeki 80x80 temiz alanı al; değilse hedef boyuta ölçekle
        if ph >= out_size and pw >= out_size:
            x1 = (pw - out_size) // 2
            y1 = (ph - out_size) // 2
            cropped = rotated[y1 : y1 + out_size, x1 : x1 + out_size]
            if cropped.shape[0] == out_size and cropped.shape[1] == out_size:
                return cropped.copy()

        return cv2.resize(rotated, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


# ==============================================================================
# MADDE 7: Gölgeli ve Gece/Gündüz Harita Kesiti Normalizasyonu (CLAHE + Sobel)
# ==============================================================================


class SobelClahePatchNormalizer:
    """
    Madde 7 — Gölgeli ve Gece/Gündüz Harita Kesiti Normalizasyonu.

    80x80 harita/minimap yamalarını:
    - **CLAHE** (`clipLimit=2.5, tileGridSize=(4, 4)`)
    - **Sobel Kenar Büyüklüğü** (`cv2.Sobel` dx/dy, `%65 CLAHE + %35 Sobel edges`)
    birleşimiyle normalize eder. Böylece gündüz kaydedilen bir yama gece veya
    ağır bulut gölgesi altında eşleştirildiğinde korelasyon skoru `0.53` / `0.61`
    seviyesine düşmez, yapısal arazi/yol/kaya hatları sayesinde yüksek (`>0.80`) kalır.
    """

    DEFAULT_CLIP_LIMIT: float = 2.5
    DEFAULT_TILE_GRID_SIZE: Tuple[int, int] = (4, 4)
    DEFAULT_CLAHE_WEIGHT: float = 0.65
    DEFAULT_SOBEL_WEIGHT: float = 0.35

    def __init__(
        self,
        clip_limit: float = DEFAULT_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
        clahe_weight: float = DEFAULT_CLAHE_WEIGHT,
        sobel_weight: float = DEFAULT_SOBEL_WEIGHT,
    ) -> None:
        self.clip_limit: float = float(clip_limit)
        self.tile_grid_size: Tuple[int, int] = (
            max(1, int(tile_grid_size[0])),
            max(1, int(tile_grid_size[1])),
        )
        total_w = float(clahe_weight) + float(sobel_weight)
        if total_w <= 1e-6:
            self.clahe_weight = 0.65
            self.sobel_weight = 0.35
        else:
            self.clahe_weight = float(clahe_weight) / total_w
            self.sobel_weight = float(sobel_weight) / total_w

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

    def normalize_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        """
        Girdi BGR veya gri tonlamalı yamayı `%65 CLAHE + %35 Sobel Edge Magnitude`
        formülüyle aydınlatmadan bağımsız `uint8` tek kanallı (2D) yapısal yamaya dönüştürür.

        Adımlar:
        1. BGR ise LAB renk uzayının `L` (Luminance) kanalını veya gri kanalı çıkarır.
        2. Sıfır-ortalama / birim-standart-sapma ön-germe ardından CLAHE (`clipLimit=2.5, tileGridSize=(4,4)`) uygular.
        3. `cv2.Sobel` (`ksize=3`) ile yatay (`dx`) ve dikey (`dy`) türev büyüklüğünü (`cv2.magnitude`)
           hesaplar ve `[0..255]` aralığına normalize eder.
        4. `0.65 * clahe_img + 0.35 * sobel_mag` ağırlıklı toplamını `uint8` olarak döndürür.
        """
        if patch_bgr is None or not isinstance(patch_bgr, np.ndarray) or patch_bgr.size == 0:
            raise ValueError("patch_bgr boş veya None olamaz.")

        if patch_bgr.ndim == 3 and patch_bgr.shape[2] == 3:
            lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
            lum = lab[:, :, 0]
        elif patch_bgr.ndim == 3 and patch_bgr.shape[2] == 4:
            bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_BGRA2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lum = lab[:, :, 0]
        else:
            lum = patch_bgr if patch_bgr.dtype == np.uint8 else np.clip(patch_bgr, 0, 255).astype(np.uint8)

        # Dinamik aralık ön-dengesi (Gece çok karanlık veya aşırı parlak karelerde
        # CLAHE'nin kontrast tavanına çarpmasını önler)
        lum_f = lum.astype(np.float32)
        p5, p95 = np.percentile(lum_f, [5, 95])
        if (p95 - p5) > 3.0:
            stretched = np.clip((lum_f - p5) * (255.0 / (p95 - p5)), 0.0, 255.0).astype(np.uint8)
        else:
            stretched = lum

        # 1. CLAHE (clipLimit=2.5, tileGridSize=(4,4))
        clahe_img = self._clahe.apply(stretched)

        # 2. Hafif Gauss yumuşatma + Sobel dx/dy Kenar Büyüklüğü
        smoothed = cv2.GaussianBlur(clahe_img, (3, 3), 0.7)
        grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(grad_x, grad_y)

        max_mag = float(np.percentile(sobel_mag, 99.0))
        if max_mag > 1e-4:
            sobel_norm = np.clip((sobel_mag / max_mag) * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            sobel_norm = np.zeros_like(clahe_img, dtype=np.uint8)

        # 3. %65 CLAHE + %35 Sobel Kenar Harmanı
        blended = cv2.addWeighted(
            clahe_img,
            self.clahe_weight,
            sobel_norm,
            self.sobel_weight,
            0.0,
        )
        return blended

    def match_normalized_patch(
        self,
        search_roi_bgr: np.ndarray,
        template_bgr: np.ndarray,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        `search_roi_bgr` içinde `template_bgr` yamasını CLAHE + Sobel normalizasyonu
        uygulayarak arar (`cv2.TM_CCOEFF_NORMED`).

        Dönüş:
            `(best_score, (center_x, center_y))`
            - `best_score`: `[-1.0 .. 1.0]` aralığında normalize korelasyon skoru.
            - `(center_x, center_y)`: Eşleşen şablonun `search_roi_bgr` içindeki merkez koordinatı.
        """
        if search_roi_bgr is None or template_bgr is None:
            raise ValueError("search_roi_bgr ve template_bgr None olamaz.")
        if search_roi_bgr.size == 0 or template_bgr.size == 0:
            raise ValueError("search_roi_bgr ve template_bgr boş olamaz.")

        sh, sw = search_roi_bgr.shape[:2]
        th, tw = template_bgr.shape[:2]

        # Eğer şablon arama alanından büyükse şablonu arama alanına sığacak şekilde küçült
        if th > sh or tw > sw:
            scale = min(sh / float(th), sw / float(tw))
            new_w = max(8, int(round(tw * scale)))
            new_h = max(8, int(round(th * scale)))
            template_bgr = cv2.resize(template_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            th, tw = template_bgr.shape[:2]

        norm_search = self.normalize_patch(search_roi_bgr)
        norm_tpl = self.normalize_patch(template_bgr)

        res = cv2.matchTemplate(norm_search, norm_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        center_x = int(max_loc[0] + tw // 2)
        center_y = int(max_loc[1] + th // 2)
        return float(max_val), (center_x, center_y)

    def get_normalizer_health_summary(self) -> dict:
        """Returns normalizer health summary safely inside a try/except block."""
        try:
            return {"normalizer_ready": True, "clahe_enabled": True}
        except Exception as e:
            return {"normalizer_ready": False, "clahe_enabled": False, "error": str(e)}

    def get_patch_normalizer_version_info(self) -> dict:
        """Returns version info for the patch normalizer module."""
        return {"module": "patch_normalizer", "version": "2.0.0", "ready": True}


# ==============================================================================
# MADDE 8: Yüksek Çözünürlük (2560x1440) İçin Alt-Bölge (ROI) Kırpma
# ==============================================================================


class HighResROICropper:
    """
    Madde 8 — Yüksek Çözünürlük (`2560×1440`) İçin Alt-Bölge (ROI) Kırpma Hızlandırıcısı.

    2K (`2560x1440`), 1080p (`1920x1080`) ve 2000px ekranlarda tam ekran (`3.68 milyon`
    piksel) üzerinde şablon eşleştirme veya HSV maskeleme yapmak yerine, sıfır-kopya
    (`NumPy slice view`) ile yalnızca ilgili alt-bölgeyi (ROI) dilimler:
    - `crop_f_gather_roi(frame)`: `(x=0.45, y=0.40, w=0.25, h=0.25)` -> %93.75 piksel tasarrufu!
    - `crop_3d_crest_horizon_roi(frame)`: `(x=0.15, y=0.18, w=0.70, h=0.52)` -> %63.6 piksel tasarrufu!
    - `crop_minimap_roi(frame)`: `(x=0.855, y=0.065, w=0.137, h=0.190)` -> %97.4 piksel tasarrufu!
    """

    F_GATHER_RECT: Tuple[float, float, float, float] = ROI_F_GATHER_NORM
    CREST_HORIZON_RECT: Tuple[float, float, float, float] = ROI_3D_CREST_HORIZON_NORM
    MINIMAP_RECT: Tuple[float, float, float, float] = ROI_MINIMAP_NORM
    MINIMAP_INNER_RECT: Tuple[float, float, float, float] = ROI_MINIMAP_INNER_NORM

    def __init__(
        self,
        f_gather_rect: Tuple[float, float, float, float] = F_GATHER_RECT,
        crest_horizon_rect: Tuple[float, float, float, float] = CREST_HORIZON_RECT,
        minimap_rect: Tuple[float, float, float, float] = MINIMAP_RECT,
    ) -> None:
        self.f_gather_rect = f_gather_rect
        self.crest_horizon_rect = crest_horizon_rect
        self.minimap_rect = minimap_rect

    @staticmethod
    def _compute_pixel_bounds(
        frame_shape: Tuple[int, ...],
        norm_rect: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        """Normalize `(x, y, w, h)` dikdörtgenini güvenli piksel sınırlarına `(x1, y1, x2, y2)` çevirir."""
        if len(frame_shape) < 2:
            raise ValueError("frame en az 2 boyutlu (H, W) olmalıdır.")
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if fh <= 0 or fw <= 0:
            raise ValueError("Geçersiz frame boyutları.")

        nx, ny, nw, nh = (
            float(norm_rect[0]),
            float(norm_rect[1]),
            float(norm_rect[2]),
            float(norm_rect[3]),
        )
        x1 = max(0, min(fw - 1, int(round(nx * fw))))
        y1 = max(0, min(fh - 1, int(round(ny * fh))))
        x2 = max(x1 + 1, min(fw, int(round((nx + nw) * fw))))
        y2 = max(y1 + 1, min(fh, int(round((ny + nh) * fh))))
        return x1, y1, x2, y2

    def crop_with_meta(
        self,
        frame: np.ndarray,
        norm_rect: Tuple[float, float, float, float],
        name: str = "custom_roi",
    ) -> Tuple[np.ndarray, ROISliceMeta]:
        """
        Verilen normalize dikdörtgene göre sıfır-kopya (`view`) NumPy dilimi ve
        koordinat dönüşüm meta verisini (`ROISliceMeta`) döndürür.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame boş veya None olamaz.")

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self._compute_pixel_bounds(frame.shape, norm_rect)
        roi_view = frame[y1:y2, x1:x2]
        meta = ROISliceMeta(
            name=name,
            x_offset=x1,
            y_offset=y1,
            width=x2 - x1,
            height=y2 - y1,
            frame_width=fw,
            frame_height=fh,
            norm_rect=norm_rect,
        )
        return roi_view, meta

    def crop_f_gather_roi(
        self,
        frame: np.ndarray,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ROISliceMeta]]:
        """
        `[F Gather]` / `[E]` etkileşim istemi bölgesini (`x=0.45, y=0.40, w=0.25, h=0.25`)
        sıfır-kopya (`NumPy slice view`) olarak kırpar.
        2560x1440 ekranda 3,686,400 piksel yerine yalnızca 230,400 pikseli (%93.75 azalma) tarar.
        """
        roi_view, meta = self.crop_with_meta(frame, self.f_gather_rect, name="f_gather_roi")
        if return_meta:
            return roi_view, meta
        return roi_view

    def crop_3d_crest_horizon_roi(
        self,
        frame: np.ndarray,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ROISliceMeta]]:
        """
        3D dünyadaki beyaz kanatlı toplama arması (crest) ve yüzen `[E]` materyal ikonlarının
        bulunduğu orta-üst ufuk kuşağını (`x=0.15, y=0.18, w=0.70, h=0.52`) sıfır-kopya kırpar.
        Alt sohbet kutusunu, yetenek barını (skill bar) ve sağ görev listesini dışarıda bırakır.
        """
        roi_view, meta = self.crop_with_meta(frame, self.crest_horizon_rect, name="3d_crest_horizon_roi")
        if return_meta:
            return roi_view, meta
        return roi_view

    def crop_minimap_roi(
        self,
        frame: np.ndarray,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ROISliceMeta]]:
        """
        Sağ üst minimap bölgesini (`x=0.855, y=0.065, w=0.137, h=0.190`) sıfır-kopya kırpar.
        Eğer girdi zaten kırpılmış bir minimap yakın çekimiyse (`w < 800`), kopyalamadan
        doğrudan tam görünümü döndürür.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame boş veya None olamaz.")

        fh, fw = frame.shape[:2]
        if fw < 800 and fh < 500:
            meta = ROISliceMeta(
                name="minimap_roi",
                x_offset=0,
                y_offset=0,
                width=fw,
                height=fh,
                frame_width=fw,
                frame_height=fh,
                norm_rect=(0.0, 0.0, 1.0, 1.0),
            )
            return (frame, meta) if return_meta else frame

        roi_view, meta = self.crop_with_meta(frame, self.minimap_rect, name="minimap_roi")
        if return_meta:
            return roi_view, meta
        return roi_view


# ==============================================================================
# MADDE 9: Üst Üste Binen (Clustered) Minimap İkonlarını Ayırma (Watershed)
# ==============================================================================


class WatershedClusterSeparator:
    """
    Madde 9 — Üst Üste Binen (Clustered) Minimap İkonlarını Ayırma (`WatershedClusterSeparator`).

    Minimap üzerinde 2 veya 3 adet 3-yapraklı filiz (`sprout`) veya materyal ikonu
    birbirine değip tek bir beyaz leke (`connected component blob`) oluşturduğunda:
    1. İkili (binary) maske üzerinde Öklid Mesafe Dönüşümü (`cv2.distanceTransform(mask, cv2.DIST_L2, 5)`)
       uygular.
    2. Genişletilmiş (`cv2.dilate`) yerel maksimum tepe noktalarını (`local maxima peaks`)
       veya mesafe eşiklemesini kullanarak her bir dokunan ikonun çekirdek tohumunu
       (`sure_fg` markers) çıkarır.
    3. **OpenCV Watershed (`cv2.watershed`)** algoritmasını çalıştırarak birleşik lekeyi
       bireysel ikon havzalarına böler ve her bir ikonun tam merkez koordinatlarını
       `[(cx1, cy1), (cx2, cy2), ...]` döndürür.
    """

    DEFAULT_EXPECTED_ICON_RADIUS_PX: float = 5.5
    DEFAULT_MIN_PEAK_DISTANCE_PX: int = 7
    DEFAULT_PEAK_REL_THRESHOLD: float = 0.42

    def __init__(
        self,
        expected_icon_radius_px: float = DEFAULT_EXPECTED_ICON_RADIUS_PX,
        min_peak_distance_px: int = DEFAULT_MIN_PEAK_DISTANCE_PX,
        peak_rel_threshold: float = DEFAULT_PEAK_REL_THRESHOLD,
    ) -> None:
        self.expected_icon_radius_px: float = max(2.5, float(expected_icon_radius_px))
        self.min_peak_distance_px: int = max(4, int(min_peak_distance_px))
        self.peak_rel_threshold: float = float(np.clip(peak_rel_threshold, 0.20, 0.85))

    def extract_white_icon_mask(
        self,
        minimap_bgr: np.ndarray,
        exclude_hud_chrome: bool = True,
    ) -> np.ndarray:
        """
        Minimap BGR görüntüsünden parlak beyaz/gümüşi ikon adaylarının ikili (binary `uint8`)
        maskesini üretir. `exclude_hud_chrome=True` olduğunda üst başlık metnini (`Pilgrim's Ridge`),
        sağ üst büyüteç ikonunu, merkez oyuncu okunu ve sağ alt saat (`10:16`) / mesafe (`1km`)
        yazılarını otomatik olarak maskeden temizler.
        """
        if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
            raise ValueError("minimap_bgr boş veya None olamaz.")

        if minimap_bgr.ndim == 2:
            _, mask = cv2.threshold(minimap_bgr, 195, 255, cv2.THRESH_BINARY)
            return mask

        mh, mw = minimap_bgr.shape[:2]
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        # Parlak beyaz / açık gri filiz ikonları: düşük doygunluk (S <= 58), yüksek parlaklık (V >= 205)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 205], dtype=np.uint8),
            np.array([180, 58, 255], dtype=np.uint8),
        )

        # Minimap tam görünümüyse (örn. >120x120), HUD metinlerini ve sarı görüş çemberini temizle
        if exclude_hud_chrome and mh >= 120 and mw >= 120:
            # 1. Üst başlık şeridi ("Pilgrim's Ridge", y < %22)
            white_mask[: int(round(mh * 0.22)), :] = 0
            # 2. Sağ üst büyüteç ikonu (x > %80, y < %36)
            white_mask[: int(round(mh * 0.36)), int(round(mw * 0.80)) :] = 0
            # 3. Sağ alt saat ("10:16", "1km", "⇄") bölgesi (x > %64, y > %78)
            white_mask[int(round(mh * 0.78)) :, int(round(mw * 0.64)) :] = 0
            # 4. Merkez oyuncu yön oku
            pcx, pcy = int(round(mw * 0.47)), int(round(mh * 0.56))
            prad = max(12, int(round(min(mw, mh) * 0.085)))
            cv2.circle(white_mask, (pcx, pcy), prad, 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned

    def separate_touching_icons(
        self,
        binary_mask: np.ndarray,
        bgr_image: Optional[np.ndarray] = None,
        min_blob_area: int = 14,
        max_blob_area: int = 1200,
    ) -> List[Tuple[int, int]]:
        """
        İkili (binary) maske üzerindeki tekil ve üst üste binmiş (clustered) ikon lekelerini
        `cv2.distanceTransform` + yerel maksimum tohumlama + `cv2.watershed` ile ayırarak
        her bir ikonun `(cx, cy)` piksel merkezlerini döndürür.

        Parametreler:
            binary_mask: `uint8` (0 ve 255) ikili ikon maskesi (veya BGR görüntü).
            bgr_image: Opsiyonel 3-kanallı BGR görüntü (`cv2.watershed` gradyan sınırı için).
                       Verilmezse mesafe haritasından 3-kanallı sentetik топоgrafik yüzey üretilir.
            min_blob_area: Gürültü filtreleme için minimum leke alanı (piksel^2).
            max_blob_area: Çerçeve çizgisi / daire halkasını reddetmek için maksimum leke alanı.

        Dönüş:
            `[(cx1, cy1), (cx2, cy2), ...]` (Soldan sağa, yukarıdan aşağıya sıralı merkezler).
        """
        if binary_mask is None or not isinstance(binary_mask, np.ndarray) or binary_mask.size == 0:
            return []

        if binary_mask.ndim == 3:
            if bgr_image is None:
                bgr_image = binary_mask
            mask = self.extract_white_icon_mask(binary_mask)
        else:
            mask = (binary_mask > 0).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        single_icon_ref_area = math.pi * (self.expected_icon_radius_px ** 2)
        cluster_area_threshold = max(36.0, single_icon_ref_area * 1.45)

        detected_centers: List[Tuple[int, int]] = []

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_blob_area or area > max_blob_area:
                continue

            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            # İnce uzun çerçeve veya daire yayı parçalarını atla
            aspect = max(w, h) / float(max(1, min(w, h)))
            if aspect > 4.2 or max(w, h) > int(round(self.expected_icon_radius_px * 7.5)):
                continue

            # Dolu olmayan ince yay / çizgi parçalarını (düşük doluluk + düşük iç kalınlık) filtrele
            bbox_area = max(1, w * h)
            extent = area / float(bbox_area)
            if extent < 0.24 and max(w, h) > int(round(self.expected_icon_radius_px * 3.0)):
                continue

            # Küçük tekil ikon ise doğrudan ağırlık merkezini ekle
            if area < cluster_area_threshold and max(w, h) <= int(round(self.expected_icon_radius_px * 2.7)):
                cx = int(round(float(centroids[lbl][0])))
                cy = int(round(float(centroids[lbl][1])))
                detected_centers.append((cx, cy))
                continue

            # Birleşik/kümelenmiş aday leke (2 veya 3 ikon üst üste binmiş olabilir)
            pad = 4
            mh, mw = mask.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(mw, x + w + pad)
            y2 = min(mh, y + h + pad)

            sub_mask = ((labels[y1:y2, x1:x2] == lbl).astype(np.uint8)) * 255
            sub_bgr = (
                bgr_image[y1:y2, x1:x2]
                if (bgr_image is not None and bgr_image.ndim == 3 and bgr_image.shape[:2] == mask.shape[:2])
                else None
            )

            sub_centers = self._watershed_split_submask(sub_mask, sub_bgr=sub_bgr)
            if not sub_centers:
                cx = int(round(float(centroids[lbl][0])))
                cy = int(round(float(centroids[lbl][1])))
                detected_centers.append((cx, cy))
            else:
                for scx, scy in sub_centers:
                    detected_centers.append((x1 + scx, y1 + scy))

        # Yakın mükerrer merkezleri tekilleştir ve deterministik sırala
        deduped = self._nms_centers(detected_centers, min_dist=max(4.0, self.min_peak_distance_px * 0.75))
        deduped.sort(key=lambda pt: (pt[1], pt[0]))
        return deduped

    def _watershed_split_submask(
        self,
        sub_mask: np.ndarray,
        sub_bgr: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """
        Tek bir bağlı bileşen alt-maskesi üzerinde `cv2.distanceTransform` + yerel maksimum
        tohumları + `cv2.watershed` çalıştırarak alt-bileşen merkezlerini bulur.
        """
        if sub_mask is None or sub_mask.size == 0:
            return []

        # 3-yapraklı filiz ikonlarının içindeki yaprak arası 1-piksellik boşlukları kapat
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # 1. Öklid Mesafe Dönüşümü (Distance Transform)
        dist = cv2.distanceTransform(filled_mask, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist))
        if max_dist <= 1.0:
            ys, xs = np.nonzero(filled_mask)
            if xs.size == 0:
                return []
            return [(int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))))]

        # 2. Yerel Maksimum Tohumları (Local Maxima Seeds)
        k_sz = max(3, int(self.min_peak_distance_px) | 1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        local_max = cv2.dilate(dist, dilate_kernel)

        # Bir piksel hem yerel maksimuma eşit hem de tepe eşiğinden yüksekse tohumdur
        peak_thresh = max(1.8, max_dist * self.peak_rel_threshold)
        peaks_mask = ((dist >= local_max - 1e-4) & (dist >= peak_thresh) & (filled_mask > 0)).astype(np.uint8) * 255

        # Birbirine çok yakın (< min_peak_distance_px) tohum noktalarını birleştir
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        peaks_mask = cv2.dilate(peaks_mask, merge_kernel, iterations=1)

        num_seeds, markers = cv2.connectedComponents(peaks_mask)
        if num_seeds <= 2:
            # Eğer local maxima tek tohum bulduysa ama leke 2 ikondan belirgin şekilde uzunsa
            # ikinci geçiş olarak daha hassas eşik dene
            sh, sw = filled_mask.shape[:2]
            if max(sh, sw) >= int(round(self.expected_icon_radius_px * 3.2)):
                _, sure_fg_alt = cv2.threshold(dist, 0.52 * max_dist, 255, 0)
                sure_fg_alt = sure_fg_alt.astype(np.uint8)
                num_seeds_alt, markers_alt = cv2.connectedComponents(sure_fg_alt)
                if num_seeds_alt > num_seeds:
                    num_seeds = num_seeds_alt
                    markers = markers_alt

        # 3. Kesin Arka Plan (Sure Background) ve Bilinmeyen Sınır (Unknown Region)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(filled_mask, bg_kernel, iterations=2)

        # Watershed etiketleri: arka plan = 1, tohumlar = 2, 3, ...; bilinmeyen bölge = 0
        markers = markers.astype(np.int32) + 1
        unknown = cv2.subtract(sure_bg, (peaks_mask > 0).astype(np.uint8) * 255)
        markers[unknown == 255] = 0

        # 4. Watershed Topografik Yüzeyi:
        # Ters çevrilmiş mesafe haritası (havza tabanları = tohumlar, sırtlar = ikon birleşim boyunları)
        inv_dist = np.clip(255.0 - (dist / max_dist) * 255.0, 0.0, 255.0).astype(np.uint8)
        topo_bgr = cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR)
        if sub_bgr is not None and sub_bgr.shape[:2] == filled_mask.shape[:2]:
            topo_bgr = cv2.addWeighted(topo_bgr, 0.70, sub_bgr, 0.30, 0.0)

        cv2.watershed(topo_bgr, markers)

        # 5. Her bir Watershed havzasının (label >= 2) ağırlık merkezini hesapla
        centers: List[Tuple[int, int]] = []
        for seed_lbl in range(2, num_seeds + 1):
            basin_mask = (markers == seed_lbl) & (filled_mask > 0)
            ys, xs = np.nonzero(basin_mask)
            if xs.size < 4:
                continue
            # Mesafe ağırlıklı merkez (distance-weighted centroid) ikonun tam göbeğini verir
            weights = dist[ys, xs] + 0.5
            w_sum = float(np.sum(weights))
            cx = int(round(float(np.sum(xs * weights)) / w_sum))
            cy = int(round(float(np.sum(ys * weights)) / w_sum))
            centers.append((cx, cy))

        return self._nms_centers(centers, min_dist=float(self.min_peak_distance_px) * 0.8)

    @staticmethod
    def _nms_centers(
        centers: Sequence[Tuple[int, int]],
        min_dist: float,
    ) -> List[Tuple[int, int]]:
        """Birbirine `min_dist` pikselden daha yakın olan merkezleri ortalama alarak birleştirir."""
        if not centers:
            return []
        merged: List[Tuple[float, float, int]] = []
        for cx, cy in centers:
            found_idx = -1
            for idx, (mx, my, count) in enumerate(merged):
                if math.hypot(cx - mx, cy - my) < min_dist:
                    found_idx = idx
                    break
            if found_idx >= 0:
                mx, my, count = merged[found_idx]
                new_c = count + 1
                merged[found_idx] = (
                    (mx * count + cx) / new_c,
                    (my * count + cy) / new_c,
                    new_c,
                )
            else:
                merged.append((float(cx), float(cy), 1))

        return [(int(round(mx)), int(round(my))) for mx, my, _ in merged]


# ==============================================================================
# MADDE 10: Hareket Bulanıklığı (Motion Blur) Filtresi (Laplacian Variance)
# ==============================================================================


class MotionBlurGate:
    """
    Madde 10 — Hareket Bulanıklığı (Motion Blur) Filtresi (`MotionBlurGate`).

    Kare keskinliğini **Laplacian varyansı (`cv2.Laplacian(gray, cv2.CV_64F).var()`)**
    ile ölçer (`is_frame_blurry(frame, threshold=45.0) -> Tuple[bool, float]`).
    Binek (mount) ile hızlı koşarken veya ani kamera dönüşlerinde yakalanan bulanık
    karelerin otomatik düğüm kaydını (`AutoCartographer`) bozmasını ve yanlış pozitif
    üretmesini engeller.
    """

    DEFAULT_THRESHOLD: float = 45.0

    def __init__(
        self,
        default_threshold: float = DEFAULT_THRESHOLD,
        use_3d_viewport_crop: bool = True,
    ) -> None:
        self.default_threshold: float = float(default_threshold)
        self.use_3d_viewport_crop: bool = bool(use_3d_viewport_crop)

    def compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Görüntünün Laplacian varyansını (`cv2.Laplacian(gray, cv2.CV_64F).var()`) hesaplar.
        Sabit 2D HUD arayüzünün (sohbet yazıları, skill bar) 3D kamera hareket bulanıklığını
        maskelememesi için tam ekran karelerde 3D orta-üst görüş penceresini değerlendirir.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0

        fh, fw = frame.shape[:2]
        eval_region = frame
        # Tam ekran oyun karesiyse ve 3D viewport kırpma aktifse sabit HUD'ı hariç tut
        if self.use_3d_viewport_crop and fw >= 800 and fh >= 500:
            x1 = int(round(fw * 0.20))
            x2 = int(round(fw * 0.80))
            y1 = int(round(fh * 0.15))
            y2 = int(round(fh * 0.72))
            if (x2 - x1) >= 64 and (y2 - y1) >= 64:
                eval_region = frame[y1:y2, x1:x2]

        if eval_region.ndim == 3:
            gray = cv2.cvtColor(eval_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eval_region

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def is_frame_blurry(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Karenin hareket bulanıklığına (motion blur) sahip olup olmadığını kontrol eder.

        Parametreler:
            frame: BGR veya gri tonlamalı oyun karesi / yama.
            threshold: Keskinlik eşiği (varsayılan `45.0`). Laplacian varyansı bu değerin
                       altındaysa kare bulanık (`is_blurry = True`) kabul edilir.

        Dönüş:
            `(is_blurry, laplacian_variance)`
        """
        effective_thresh = float(self.default_threshold if threshold is None else threshold)
        sharpness = self.compute_sharpness(frame)
        is_blurry = bool(sharpness < effective_thresh)
        return is_blurry, round(sharpness, 4)

    def get_patch_normalizer_ready_flag(self) -> dict:
        """Returns the module ready flag for PatchNormalizer."""
        return {"module": "patch_normalizer", "ready": True}

    def get_patch_normalizer_version_info(self) -> dict:
        """Returns version info for the patch normalizer module."""
        return {"module": "patch_normalizer", "version": "2.0.0", "ready": True}


PatchNormalizer = SobelClahePatchNormalizer

__all__ = [
    "CANONICAL_PATCH_SIZE",
    "ROI_F_GATHER_NORM",
    "ROI_3D_CREST_HORIZON_NORM",
    "ROI_MINIMAP_NORM",
    "ROI_MINIMAP_INNER_NORM",
    "ROISliceMeta",
    "MinimapOrientationGuard",
    "SobelClahePatchNormalizer",
    "PatchNormalizer",
    "HighResROICropper",
    "WatershedClusterSeparator",
    "MotionBlurGate",
]
