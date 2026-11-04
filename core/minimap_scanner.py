# -*- coding: utf-8 -*-
"""
Gerçek Zamanlı Minimap Materyal (3 Yapraklı Filiz) ve Radar Tarayıcı Motoru.

Oyun ekranının sağ üst köşesindeki minimap bölgesini (`norm_rect = (0.855, 0.065, 0.137, 0.190)`,
yani `x: 0.855..0.992, y: 0.065..0.255`) veya doğrudan kırpılmış minimap yakın çekimlerini
analiz ederek:
1. Oyuncu yön açısını (`detect_player_heading`: 0°-360°, Kuzey=0°, Doğu=90°),
2. Yalnızca gerçek toplanabilir materyal ikonu olan **beyaz 3 yapraklı filiz (`sprout_material`)**
   düğümlerini (`gatherable_nodes`),
3. Küçük düz yuvarlak beyaz canavar noktalarını (`monster_dots`, `•` - materyal olarak REDDEDİLİR),
4. Beyaz tüy (`minimap_feather_odyle.png`), mavi kanat ve yeşil yığın rozetlerini (`rejected_icons` -
   materyal olarak REDDEDİLİR, toplanmaz),
5. Kırmızı düşman noktalarını (`hostile_dots`),
6. Minimap solundaki `Minimap Filter Settings` açılır penceresinin durumunu (`detect_minimap_filter_state`),
7. Ardışık kareler arasındaki yürüyüş ilerlemesini (`detect_movement_progress`)
yüksek doğrulukla tespit eder.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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


class MinimapScanner:
    """
    Minimap 3-Yapraklı Filiz (`sprout_material`) Tarayıcı, Canavar/Tüy Filtreleyici,
    Oyuncu Yönü (Heading) ve Yürüyüş İlerleme Analizörü.
    """

    # Sağ üst minimap kutusu: x in [0.855..0.992], y in [0.065..0.255]
    DEFAULT_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.137, 0.190)
    PLAYER_CENTER_REL: Tuple[float, float] = (0.502, 0.545)

    # 332x257 referans minimap ölçeğinde kanonik 16x16 beyaz 3-yapraklı filiz (sprout) şablonu
    _CANONICAL_SPROUT_TPL: np.ndarray = np.array(
        [
            [81, 82, 84, 85, 77, 84, 62, 122, 92, 69, 84, 81, 88, 87, 83, 78],
            [83, 90, 88, 79, 87, 66, 90, 230, 159, 53, 83, 81, 80, 81, 81, 79],
            [83, 83, 82, 79, 83, 34, 157, 232, 200, 80, 73, 93, 85, 85, 85, 84],
            [84, 76, 80, 90, 59, 85, 240, 210, 227, 150, 56, 86, 92, 89, 87, 87],
            [87, 86, 86, 86, 44, 184, 245, 214, 227, 202, 83, 73, 90, 87, 85, 84],
            [85, 94, 82, 58, 93, 236, 235, 185, 214, 238, 145, 38, 86, 89, 90, 87],
            [81, 85, 73, 54, 181, 255, 255, 165, 216, 255, 199, 35, 73, 87, 95, 90],
            [84, 73, 73, 82, 229, 255, 232, 155, 195, 255, 248, 121, 50, 74, 90, 86],
            [87, 81, 46, 175, 249, 254, 228, 143, 202, 245, 255, 220, 53, 77, 88, 84],
            [81, 70, 57, 166, 245, 254, 244, 114, 190, 255, 246, 214, 65, 40, 86, 93],
            [79, 53, 38, 94, 185, 245, 234, 116, 173, 255, 205, 93, 29, 69, 71, 79],
            [66, 102, 146, 160, 122, 183, 234, 89, 179, 210, 131, 182, 139, 97, 78, 48],
            [220, 253, 246, 254, 227, 96, 174, 72, 140, 136, 214, 246, 254, 246, 206, 102],
            [67, 152, 251, 252, 241, 218, 80, 45, 87, 200, 243, 253, 237, 175, 84, 68],
            [67, 63, 131, 182, 141, 78, 40, 44, 43, 78, 124, 196, 142, 69, 78, 85],
            [100, 76, 50, 60, 78, 84, 100, 93, 91, 85, 61, 47, 64, 78, 85, 88],
        ],
        dtype=np.uint8,
    )

    _CANONICAL_SPROUT_MASK: np.ndarray = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    def __init__(
        self,
        norm_rect: Tuple[float, float, float, float] = DEFAULT_NORM_RECT,
        movement_shift_threshold: float = 0.22,
        movement_diff_threshold: float = 3.2,
        templates_dir: Optional[str] = None,
    ) -> None:
        self.norm_rect: Tuple[float, float, float, float] = (
            float(norm_rect[0]),
            float(norm_rect[1]),
            float(norm_rect[2]),
            float(norm_rect[3]),
        )
        self.movement_shift_threshold: float = float(movement_shift_threshold)
        self.movement_diff_threshold: float = float(movement_diff_threshold)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.templates_dir: str = (
            templates_dir if templates_dir is not None else os.path.join(base_dir, "templates")
        )
        self._external_sprout_templates: List[Tuple[np.ndarray, np.ndarray]] = []
        self._load_optional_templates()

    def _load_optional_templates(self) -> None:
        """Diskte mevcutsa filiz (sprout) şablonlarını Unicode-güvenli olarak yükler."""
        self._external_sprout_templates.clear()
        for fname in ("minimap_sprout_node.png", "minimap_gatherable_sprout_icon.png"):
            fpath = os.path.join(self.templates_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                raw_bytes = np.fromfile(fpath, dtype=np.uint8)
                if raw_bytes.size == 0:
                    continue
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                th, tw = gray.shape[:2]
                if 10 <= tw <= 28 and 10 <= th <= 28:
                    med = float(np.median(gray))
                    mask = ((gray < max(68.0, med - 15.0)) | (gray > min(180.0, med + 25.0))).astype(
                        np.uint8
                    )
                    if int(mask.sum()) >= 25:
                        self._external_sprout_templates.append((gray, mask))
            except Exception as exc:
                logger.debug("Şablon yükleme atlandı (%s): %s", fname, exc)

    # ------------------------------------------------------------------
    # Geometrik ve Görüntü Kesit Yardımcıları
    # ------------------------------------------------------------------
    def get_minimap_pixel_bounds(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """
        Ekran veya yakın çekim boyutuna göre minimap (x1, y1, x2, y2) piksel sınırlarını döndürür.
        - Tam ekran (w >= 1000): Sağ üst minimap (`x: 0.855..0.992, y: 0.065..0.255`)
          (Alt kısmı kırpılmış 16:9 ekran görüntülerinde, örn. 2000x981, sağ üst UI 16:9 yüksekliğe
          sabitlendiği için `ref_h = round(w * 9 / 16)` kullanılır).
        - Yakın çekim + Filtre penceresi (500 <= w < 1000 ve h < 500, örn. `minimap_filter_closeup.jpg`):
          Sağ yarıdaki minimap kutusunu döndürür.
        - Doğrudan minimap yakın çekimi (w < 500 ve h < 450, örn. `minimap_sprout_closeup.jpg`):
          Minimap kutusu sınırlarını döndürür.
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])

        if self.norm_rect == self.DEFAULT_NORM_RECT:
            if w < 500 and h < 450:
                if 340 <= w <= 390 and 290 <= h <= 335:
                    return (
                        int(round(w * 0.019)),
                        int(round(h * 0.107)),
                        int(round(w * 0.929)),
                        int(round(h * 0.939)),
                    )
                return 0, 0, w, h

            if 500 <= w < 1000 and h < 500:
                return (
                    int(round(w * 0.498)),
                    int(round(h * 0.201)),
                    int(round(w * 0.961)),
                    int(round(h * 0.915)),
                )

            ref_h = h
            if w >= 1200:
                expected_16_9_h = int(round(w * 9.0 / 16.0))
                if int(round(expected_16_9_h * 0.80)) <= h <= int(round(expected_16_9_h * 1.05)):
                    ref_h = expected_16_9_h

            nx, ny, nw, nh = self.norm_rect
            x1 = max(0, min(int(round(w * nx)), w - 1))
            y1 = max(0, min(int(round(ref_h * ny)), h - 1))
            x2 = max(x1 + 1, min(int(round(w * (nx + nw))), w))
            y2 = max(y1 + 1, min(int(round(ref_h * (ny + nh))), h))
            return x1, y1, x2, y2

        nx, ny, nw, nh = self.norm_rect
        x1 = max(0, min(int(round(w * nx)), w - 1))
        y1 = max(0, min(int(round(h * ny)), h - 1))
        x2 = max(x1 + 1, min(int(round(w * (nx + nw))), w))
        y2 = max(y1 + 1, min(int(round(h * (ny + nh))), h))
        return x1, y1, x2, y2

    def extract_minimap(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Kareden minimap alt görüntüsünü (BGR) ve ekran koordinat sınırlarını çıkarır."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Geçersiz veya boş görüntü karesi (frame).")

        if frame.ndim == 2:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr_frame = frame

        x1, y1, x2, y2 = self.get_minimap_pixel_bounds(bgr_frame.shape)
        crop = bgr_frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def get_minimap_player_center(
        self,
        crop_shape: Tuple[int, ...],
        crop_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Minimap kesiti içerisindeki oyuncu okunun merkez (cx, cy) koordinatını döndürür.
        Eğer `crop_bgr` verilmişse merkezdeki parlak oyuncu okunun gerçek ağırlık merkezini bulur.
        """
        ch, cw = int(crop_shape[0]), int(crop_shape[1])
        default_cx = float(cw) * self.PLAYER_CENTER_REL[0]
        default_cy = float(ch) * self.PLAYER_CENTER_REL[1]

        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return default_cx, default_cy

        try:
            rx1 = max(0, int(round(cw * 0.46)))
            ry1 = max(0, int(round(ch * 0.49)))
            rx2 = min(cw, int(round(cw * 0.55)))
            ry2 = min(ch, int(round(ch * 0.59)))
            roi = crop_bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return default_cx, default_cy

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            _, bw = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return default_cx, default_cy

            best_cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best_cnt) < 3.0:
                return default_cx, default_cy

            m = cv2.moments(best_cnt)
            if m["m00"] > 1e-5:
                cx = float(rx1) + float(m["m10"] / m["m00"])
                cy = float(ry1) + float(m["m01"] / m["m00"])
                return cx, cy
        except Exception as exc:
            logger.debug("get_minimap_player_center dinamik iyileştirme hatası: %s", exc)

        return default_cx, default_cy

    @staticmethod
    def _vector_to_compass_deg(dx: float, dy: float) -> float:
        """
        Ekran koordinat farkını (dx sağa pozitif, dy aşağı pozitif)
        pusula açısına (0.0° - 360.0°, Kuzey=0°, Doğu=90°, Güney=180°, Batı=270°) dönüştürür.
        """
        angle = math.degrees(math.atan2(float(dx), -float(dy)))
        return round((angle + 360.0) % 360.0, 2)

    # ------------------------------------------------------------------
    # 1. Oyuncu Yön Açısı (Player Heading) Tespiti
    # ------------------------------------------------------------------
    def detect_player_heading(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Minimap merkezindeki beyaz/parlak oyuncu okunun baktığı yön açısını
        (0.0° - 360.0°, Kuzey=0°, Doğu=90°) kontur köşe analiziyle hesaplar.
        """
        try:
            crop, _ = self.extract_minimap(frame)
            ch, cw = crop.shape[:2]
            if ch < 20 or cw < 20:
                return False, 0.0

            cx_f, cy_f = self.get_minimap_player_center(crop.shape, crop_bgr=crop)
            ax = int(round(cx_f))
            ay = int(round(cy_f))
            half_win = max(6, int(round(min(cw, ch) * 0.045)))
            rx1 = max(0, ax - half_win)
            ry1 = max(0, ay - half_win)
            rx2 = min(cw, ax + half_win + 1)
            ry2 = min(ch, ay + half_win + 1)

            roi = crop[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return False, 0.0

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            scale = 8
            up_gray = cv2.resize(
                gray,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

            best_contour = None
            for thresh_val in (205, 195, 185, 175):
                _, bw = cv2.threshold(up_gray, thresh_val, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                valid_cnts = [c for c in contours if cv2.contourArea(c) >= 40.0]
                if valid_cnts:
                    best_contour = max(valid_cnts, key=cv2.contourArea)
                    break

            if best_contour is None:
                return False, 0.0

            moments = cv2.moments(best_contour)
            if moments["m00"] <= 1e-5:
                return False, 0.0
            mcx = moments["m10"] / moments["m00"]
            mcy = moments["m01"] / moments["m00"]

            peri = cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, 0.03 * peri, True).reshape(-1, 2)

            if len(approx) >= 3:
                best_tri: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                max_perim = -1.0
                for p1, p2, p3 in itertools.combinations(approx, 3):
                    d12 = float(np.linalg.norm(p1 - p2))
                    d23 = float(np.linalg.norm(p2 - p3))
                    d31 = float(np.linalg.norm(p3 - p1))
                    tri_p = d12 + d23 + d31
                    if tri_p > max_perim:
                        max_perim = tri_p
                        best_tri = (p1, p2, p3)

                if best_tri is not None:
                    p1, p2, p3 = best_tri
                    candidates = [(p1, p2, p3), (p2, p3, p1), (p3, p1, p2)]
                    chosen_apex = p1
                    chosen_base_mid = 0.5 * (p2 + p3)
                    lowest_notch_intensity = 999.0

                    uh, uw = up_gray.shape[:2]
                    for apex, wing1, wing2 in candidates:
                        mid_wings = 0.5 * (wing1 + wing2)
                        mx = int(np.clip(round(mid_wings[0]), 0, uw - 1))
                        my = int(np.clip(round(mid_wings[1]), 0, uh - 1))
                        notch_val = float(up_gray[my, mx])
                        if notch_val < lowest_notch_intensity:
                            lowest_notch_intensity = notch_val
                            chosen_apex = apex
                            chosen_base_mid = mid_wings

                    dx = float(chosen_apex[0] - chosen_base_mid[0])
                    dy = float(chosen_apex[1] - chosen_base_mid[1])
                    if math.hypot(dx, dy) > 1e-3:
                        heading = self._vector_to_compass_deg(dx, dy)
                        return True, heading

            pts = best_contour.reshape(-1, 2)
            dists = np.hypot(pts[:, 0] - mcx, pts[:, 1] - mcy)
            tip_pt = pts[int(np.argmax(dists))]
            dx = float(tip_pt[0] - mcx)
            dy = float(tip_pt[1] - mcy)
            if math.hypot(dx, dy) <= 1e-3:
                return False, 0.0

            heading = self._vector_to_compass_deg(dx, dy)
            return True, heading

        except Exception as exc:
            logger.error("detect_player_heading hatası: %s", exc)
            return False, 0.0

    # ------------------------------------------------------------------
    # 2. Minimap Filter Settings Açılır Penceresi Durum Tespiti
    # ------------------------------------------------------------------
    def detect_minimap_filter_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Minimap solundaki `Minimap Filter Settings` açılır penceresinin açık olup olmadığını,
        `Monster` ve `Gatherable M...` onay kutularının seçili olup olmadığını tespit eder.
        """
        result: Dict[str, Any] = {
            "popup_open": False,
            "quest_checked": False,
            "monster_checked": False,
            "gatherable_checked": False,
            "npc_checked": False,
        }
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return result

            bgr = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim == 2
                else (
                    cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if frame.shape[2] == 4
                    else frame
                )
            )
            h, w = bgr.shape[:2]
            if w < 500 or h < 250:
                return result

            if w >= 1000:
                ref_h = h
                expected_16_9_h = int(round(w * 9.0 / 16.0))
                if int(round(expected_16_9_h * 0.80)) <= h <= int(round(expected_16_9_h * 1.05)):
                    ref_h = expected_16_9_h
                px1 = int(round(w * 0.723))
                py1 = int(round(ref_h * 0.066))
                px2 = int(round(w * 0.842))
                py2 = min(h, int(round(ref_h * 0.248)))
            else:
                px1 = int(round(w * 0.020))
                py1 = int(round(h * 0.200))
                px2 = int(round(w * 0.433))
                py2 = int(round(h * 0.920))

            if px2 <= px1 + 20 or py2 <= py1 + 20:
                return result

            panel = bgr[py1:py2, px1:px2]
            hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
            ph, pw = panel.shape[:2]

            # İç arka plan (metin ve onay kutusu dışındaki koyu arduvaz zemin)
            bg_strip = hsv[int(ph * 0.22) : int(ph * 0.90), int(pw * 0.65) : int(pw * 0.78)]
            if bg_strip.size == 0:
                return result

            mean_v = float(np.mean(bg_strip[:, :, 2]))
            std_v = float(np.std(bg_strip[:, :, 2]))
            mean_s = float(np.mean(bg_strip[:, :, 1]))

            if not (22.0 <= mean_v <= 68.0 and std_v <= 28.0 and 15.0 <= mean_s <= 90.0):
                return result

            cb_col = hsv[:, int(pw * 0.84) : int(pw * 0.96), 2]
            if cb_col.size == 0:
                return result

            q_max = float(np.max(cb_col[int(ph * 0.22) : int(ph * 0.39), :]))
            m_max = float(np.max(cb_col[int(ph * 0.41) : int(ph * 0.57), :]))
            g_max = float(np.max(cb_col[int(ph * 0.59) : int(ph * 0.75), :]))
            n_max = float(np.max(cb_col[int(ph * 0.77) : int(ph * 0.93), :]))

            if max(q_max, m_max, g_max, n_max) < 165.0:
                return result

            result["popup_open"] = True
            result["quest_checked"] = bool(q_max >= 170.0)
            result["monster_checked"] = bool(m_max >= 170.0)
            result["gatherable_checked"] = bool(g_max >= 170.0)
            result["npc_checked"] = bool(n_max >= 170.0)
            return result

        except Exception as exc:
            logger.debug("detect_minimap_filter_state hatası: %s", exc)
            return result

    # ------------------------------------------------------------------
    # 3. Maskeli Normalize Çapraz Korelasyon (3-Yapraklı Filiz Tespiti)
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_ncc(
        gray_img: np.ndarray,
        tpl: np.ndarray,
        mask: np.ndarray,
        min_local_contrast: float = 26.0,
    ) -> np.ndarray:
        """
        Yalnızca ikon + koyu dış kontur maskesi (`mask > 0`) üzerinde sıfır-ortalama
        birim-varyanslı çapraz korelasyon hesaplar; böylece minimap zemin parlaklığından
        (parlak sarı radar halkası veya koyu çim zemin) bağımsız olarak 3-yapraklı
        filiz ikonunu yüksek hassasiyetle yakalar.
        """
        m = (mask > 0).astype(np.float32)
        n = float(np.sum(m))
        if n < 12.0:
            return np.zeros(gray_img.shape[:2], dtype=np.float32)

        t = tpl.astype(np.float32)
        t_mean = float(np.sum(t * m)) / n
        t_zero = (t - t_mean) * m
        t_norm = float(np.sqrt(np.sum(t_zero * t_zero))) + 1e-6
        t_kernel = t_zero / t_norm

        img_f = gray_img.astype(np.float32)
        loc_sum = cv2.filter2D(img_f, cv2.CV_32F, m)
        loc_sq_sum = cv2.filter2D(img_f * img_f, cv2.CV_32F, m)
        loc_var = np.maximum(loc_sq_sum - (loc_sum * loc_sum) / n, 1.0)
        loc_std = np.sqrt(loc_var)

        corr = cv2.filter2D(img_f, cv2.CV_32F, t_kernel) / loc_std
        contrast = loc_std / math.sqrt(n)
        corr[contrast < float(min_local_contrast)] = 0.0
        return corr

    def _verify_sprout_patch_structure(
        self,
        gray_crop: np.ndarray,
        hsv_crop: np.ndarray,
        cx: int,
        cy: int,
        scale: float,
    ) -> bool:
        """
        Aday (cx, cy) noktasının gerçekten beyaz 3-yapraklı filiz (`sprout_material`)
        olduğunu doğrular ve yuvarlak beyaz canavar noktalarını (`•`) veya tüy ikonlarını reddeder:
        1. Düşük doygunluklu parlak beyaz iç yaprak pikselleri (`V > 180, S < 65`),
        2. Çevresinde koyu dış kontur/damar pikselleri (`gray < 115`),
        3. Alt yarıdaki beyaz yaprak genişliğinin (sol+sağ yan yapraklar) üst yarıdaki sivri uçtan
           daha geniş olması veya orta dikey damar çukurunun bulunması.
        """
        ch, cw = gray_crop.shape[:2]
        rad = max(5, int(round(8.0 * scale)))
        x1 = max(0, cx - rad)
        y1 = max(0, cy - rad)
        x2 = min(cw, cx + rad + 1)
        y2 = min(ch, cy + rad + 1)

        patch_g = gray_crop[y1:y2, x1:x2]
        patch_hsv = hsv_crop[y1:y2, x1:x2]
        ph, pw = patch_g.shape[:2]
        if ph < 9 or pw < 9:
            return False

        # Parlak beyaz/gümüş iç pikseller (mavi kanat / yeşil rozet elenir)
        white_mask = (patch_hsv[:, :, 2] >= 180) & (patch_hsv[:, :, 1] <= 68)
        white_count = int(np.sum(white_mask))
        if white_count < max(8, int(round(18.0 * scale * scale))):
            return False

        # Yüksek doygunluklu mavi/camgöbeği veya yeşil piksel oranı yüksekse reddet (mavi kanat / yeşil rozet)
        high_sat_colored = (patch_hsv[:, :, 1] >= 85) & (patch_hsv[:, :, 0] >= 55) & (
            patch_hsv[:, :, 0] <= 120
        )
        if float(np.mean(high_sat_colored)) > 0.22:
            return False

        # Koyu dış sınır / orta damar kontrastı kontrolü
        p_min = float(np.min(patch_g))
        p_max = float(np.max(patch_g))
        if (p_max - p_min) < 80.0 or p_min > 120.0:
            return False

        # Yuvarlak canavar noktası kontrolü: eğer aday tek bir küçük dairesel kontursa (circularity > 0.76) reddet!
        bw_local = (white_mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(bw_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            main_c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(main_c))
            peri = float(cv2.arcLength(main_c, True))
            circ = 4.0 * math.pi * area / (peri * peri + 1e-6)
            hull_area = float(cv2.contourArea(cv2.convexHull(main_c)))
            solidity = area / (hull_area + 1e-6)
            if circ > 0.76 and solidity > 0.88:
                return False

        return True

    # ------------------------------------------------------------------
    # 4. Ana Minimap Kesit Analizörü (3-Yapraklı Filiz vs Canavar Noktası)
    # ------------------------------------------------------------------
    def scan_minimap_crop(
        self,
        crop_bgr: np.ndarray,
        offset_xy: Tuple[int, int] = (0, 0),
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Doğrudan bir minimap BGR kesiti (`crop_bgr`) üzerinde çalışarak:
        - `gatherable_nodes`: YALNIZCA beyaz 3-yapraklı filiz (`kind="sprout_material"`) düğümlerini,
        - `monster_dots`: Küçük yuvarlak beyaz canavar noktalarını (`•`),
        - `hostile_dots`: Kırmızı düşman noktalarını,
        - `rejected_icons`: Beyaz tüy (`odyle_feather`), mavi kanat (`cyan_wing`) ve yeşil rozetleri
        birbirinden ayırarak döndürür.
        """
        gatherable_nodes: List[Dict[str, Any]] = []
        monster_dots: List[Dict[str, Any]] = []
        hostile_dots: List[Dict[str, Any]] = []
        rejected_icons: List[Dict[str, Any]] = []

        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return {
                "gatherable_nodes": gatherable_nodes,
                "monster_dots": monster_dots,
                "hostile_dots": hostile_dots,
                "rejected_icons": rejected_icons,
                "player_center": (0.0, 0.0),
            }

        if crop_bgr.ndim == 2:
            crop = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
        elif crop_bgr.shape[2] == 4:
            crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR)
        else:
            crop = crop_bgr

        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 20:
            return {
                "gatherable_nodes": gatherable_nodes,
                "monster_dots": monster_dots,
                "hostile_dots": hostile_dots,
                "rejected_icons": rejected_icons,
                "player_center": (0.0, 0.0),
            }

        ox, oy = int(offset_xy[0]), int(offset_xy[1])
        full_h = int(frame_shape[0]) if frame_shape is not None and len(frame_shape) >= 2 else ch
        full_w = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) >= 2 else cw

        center_x, center_y = self.get_minimap_player_center(crop.shape, crop_bgr=crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # UI kenar maskesi: üst başlık ("Pilgrim's Ridge" / "Western Altamia Highland"),
        # sağ üst büyüteç ("Q"), sağ alt saat ("10:09" / "10:16") ve merkez oyuncu oku
        valid_mask = np.zeros((ch, cw), dtype=np.uint8)
        top_margin = int(round(ch * 0.16))
        bottom_margin = int(round(ch * 0.96))
        side_margin = int(round(cw * 0.04))
        valid_mask[top_margin:bottom_margin, side_margin : cw - side_margin] = 255

        # Sağ üst büyüteç ikonu maskesi
        valid_mask[: int(round(ch * 0.32)), int(round(cw * 0.82)) :] = 0
        # Sağ alt saat ve mesafe ("1km / 10:16") maskesi
        valid_mask[int(round(ch * 0.76)) :, int(round(cw * 0.64)) :] = 0

        # Merkezdeki oyuncu okunu maskele
        arrow_mask_rad = max(5, int(round(min(cw, ch) * 0.042)))
        cv2.circle(
            valid_mask,
            (int(round(center_x)), int(round(center_y))),
            arrow_mask_rad,
            0,
            -1,
        )

        # Ölçek faktörü (332px referans minimap genişliğine göre)
        scale_ref = float(cw) / 332.0

        # ------------------------------------------------------------------
        # A) Kırmızı Düşman Noktaları (hostile_dots)
        # (Minimap altındaki kırmızı XP çubuğunu hariç tutmak için y < 0.89*ch sınırı uygulanır)
        # ------------------------------------------------------------------
        red_valid = valid_mask.copy()
        red_valid[int(round(ch * 0.89)) :, :] = 0
        red_low1 = cv2.inRange(hsv, np.array([0, 115, 110]), np.array([12, 255, 255]))
        red_low2 = cv2.inRange(hsv, np.array([165, 115, 110]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(red_low1, red_low2), red_valid)

        red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if area < 3.0 or area > 95.0 or bw > 14 or bh > 14:
                continue
            ix = float(bx) + float(bw) * 0.5
            iy = float(by) + float(bh) * 0.5
            dx = round(ix - center_x, 2)
            dy = round(iy - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)
            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + ix))
            sy = int(round(oy + iy))

            hostile_dots.append(
                {
                    "type": "hostile_dot",
                    "category": "hostile_dots",
                    "dx": dx,
                    "dy": dy,
                    "dist_px": dist_px,
                    "angle_deg": angle_deg,
                    "pos": (int(round(ix)), int(round(iy))),
                    "screen_pos": (sx, sy),
                    "screen_x": sx,
                    "screen_y": sy,
                    "area": round(area, 1),
                    "is_hostile": True,
                    "is_odyle": False,
                }
            )

        # ------------------------------------------------------------------
        # B) Küçük Yuvarlak Beyaz Canavar Noktaları (monster_dots, `•` - REDDEDİLİR!)
        # ------------------------------------------------------------------
        white_mask = ((hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
        white_mask = cv2.bitwise_and(white_mask, valid_mask)
        white_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in white_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)

            if area < 7.0 or area > 85.0 or bw < 4 or bh < 4 or bw > 12 or bh > 12:
                continue

            aspect = float(bw) / float(max(1, bh))
            if not (0.72 <= aspect <= 1.38):
                continue

            peri = float(cv2.arcLength(cnt, True))
            circularity = 4.0 * math.pi * area / (peri * peri + 1e-6)
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity = area / (hull_area + 1e-6)

            if circularity < 0.78 or solidity < 0.88:
                continue

            mask_c = np.zeros_like(gray)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)
            ring = cv2.dilate(mask_c, np.ones((5, 5), dtype=np.uint8)) - mask_c
            ring_vals = gray[ring > 0]
            if len(ring_vals) == 0 or float(np.mean(ring_vals < 115)) < 0.26:
                continue

            ix = float(bx) + float(bw) * 0.5
            iy = float(by) + float(bh) * 0.5
            dx = round(ix - center_x, 2)
            dy = round(iy - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)
            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + ix))
            sy = int(round(oy + iy))

            monster_dots.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "dist_px": dist_px,
                    "angle_deg": angle_deg,
                    "pos": (int(round(ix)), int(round(iy))),
                    "screen_x": sx,
                    "screen_y": sy,
                    "circularity": round(circularity, 3),
                    "area": round(area, 1),
                }
            )

        # ------------------------------------------------------------------
        # C) Mavi Kanat / Beyaz Tüy İkonlarını Tespit Et ve Materyal Olarak REDDET!
        # ("tüy odyle değil tüy toplamayacağız")
        # ------------------------------------------------------------------
        cyan_mask = cv2.inRange(hsv, np.array([82, 68, 125]), np.array([115, 255, 255]))
        cyan_mask = cv2.bitwise_and(cyan_mask, valid_mask)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        cyan_cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cyan_cnts:
            area = float(cv2.contourArea(cnt))
            if area >= 6.0:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                ix = float(bx) + float(bw) * 0.5
                iy = float(by) + float(bh) * 0.5
                rejected_icons.append(
                    {
                        "kind": "rejected_cyan_wing",
                        "dx": round(ix - center_x, 2),
                        "dy": round(iy - center_y, 2),
                        "pos": (int(round(ix)), int(round(iy))),
                        "area": round(area, 1),
                    }
                )

        # Uzun diyagonal beyaz tüy konturlarını da rejected_icons listesine kaydet
        # (Radar çemberi yaylarını hariç tutmak için solidity >= 0.55 ve koyu çerçeve şartı aranır)
        for cnt in white_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if area >= 35.0 and (bh >= 15 or bw >= 15) and bw <= 26 and bh <= 32:
                hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
                solidity = area / (hull_area + 1e-6)
                if solidity < 0.55:
                    continue
                mask_c = np.zeros_like(gray)
                cv2.drawContours(mask_c, [cnt], -1, 255, -1)
                ring = cv2.dilate(mask_c, np.ones((5, 5), dtype=np.uint8)) - mask_c
                ring_vals = gray[ring > 0]
                if len(ring_vals) == 0 or float(np.mean(ring_vals < 90)) < 0.40:
                    continue
                ix = float(bx) + float(bw) * 0.5
                iy = float(by) + float(bh) * 0.5
                rejected_icons.append(
                    {
                        "kind": "rejected_feather",
                        "dx": round(ix - center_x, 2),
                        "dy": round(iy - center_y, 2),
                        "pos": (int(round(ix)), int(round(iy))),
                        "area": round(area, 1),
                    }
                )

        # ------------------------------------------------------------------
        # D) Gerçek Toplanabilir Materyal: Beyaz 3-Yapraklı Filiz (`sprout_material`)
        # ------------------------------------------------------------------
        templates_to_eval: List[Tuple[np.ndarray, np.ndarray]] = [
            (self._CANONICAL_SPROUT_TPL, self._CANONICAL_SPROUT_MASK)
        ] + self._external_sprout_templates

        combined_score_map = np.zeros((ch, cw), dtype=np.float32)

        for base_tpl, base_mask in templates_to_eval:
            for mult in (0.94, 1.00, 1.06):
                ks = max(9, int(round(16.0 * scale_ref * mult)))
                t_scaled = cv2.resize(base_tpl, (ks, ks), interpolation=cv2.INTER_AREA)
                m_scaled = (
                    cv2.resize(
                        base_mask.astype(np.float32),
                        (ks, ks),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    > 0.40
                ).astype(np.uint8)
                score_map = self._masked_ncc(gray, t_scaled, m_scaled, min_local_contrast=26.0)
                combined_score_map = np.maximum(combined_score_map, score_map)

        # Geçerli minimap alanı dışını sıfırla
        combined_score_map[valid_mask == 0] = 0.0

        # Reddedilen mavi kanat ve tüy ikonlarının yakın çevresini sıfırla
        for rej in rejected_icons:
            rx, ry = rej["pos"]
            cv2.circle(
                combined_score_map,
                (int(rx), int(ry)),
                max(10, int(round(18.0 * scale_ref))),
                0.0,
                -1,
            )

        # Monster dot merkezlerinin tam üzerini sıfırla
        for mdot in monster_dots:
            mx, my = mdot["pos"]
            cv2.circle(
                combined_score_map,
                (int(mx), int(my)),
                max(5, int(round(7.0 * scale_ref))),
                0.0,
                -1,
            )

        # Tepe noktalarını (peaks >= 0.70) çıkar ve yapısal doğrulamadan geçir
        work_map = combined_score_map.copy()
        nms_rad = max(8, int(round(12.0 * scale_ref)))

        while True:
            _, max_val, _, max_loc = cv2.minMaxLoc(work_map)
            if float(max_val) < 0.70:
                break

            px, py = int(max_loc[0]), int(max_loc[1])
            cv2.circle(work_map, (px, py), nms_rad, 0.0, -1)

            if not self._verify_sprout_patch_structure(gray, hsv, px, py, scale_ref):
                continue

            dx = round(float(px) - center_x, 2)
            dy = round(float(py) - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)
            if dist_px < 5.0:
                continue

            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + px))
            sy = int(round(oy + py))
            norm_x = round(float(sx) / float(max(1, full_w)), 4)
            norm_y = round(float(sy) / float(max(1, full_h)), 4)
            conf = round(min(1.0, max(0.0, float(max_val))), 3)

            gatherable_nodes.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "dist_px": dist_px,
                    "angle_deg": angle_deg,
                    "kind": "sprout_material",
                    "stack_count": 1,
                    "harvests_available": 3,
                    "screen_x": sx,
                    "screen_y": sy,
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                    "confidence": conf,
                    "type": "resource",
                    "category": "resource_icon",
                    "pos": (px, py),
                    "screen_pos": (sx, sy),
                    "area": round(16.0 * scale_ref * scale_ref, 1),
                    "is_hostile": False,
                    "is_odyle": False,
                    "is_gatherable": True,
                }
            )

        gatherable_nodes.sort(key=lambda d: float(d["dist_px"]))
        monster_dots.sort(key=lambda d: float(d["dist_px"]))
        hostile_dots.sort(key=lambda d: float(d["dist_px"]))

        return {
            "gatherable_nodes": gatherable_nodes,
            "monster_dots": monster_dots,
            "hostile_dots": hostile_dots,
            "rejected_icons": rejected_icons,
            "player_center": (round(center_x, 2), round(center_y, 2)),
        }

    # ------------------------------------------------------------------
    # 5. Tam Kare veya Yakın Çekim Materyal ve Odyle Tarama Metodları
    # ------------------------------------------------------------------
    def scan_gatherable_materials(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Tam ekran veya yakın çekim (`minimap_sprout_closeup.jpg`, `minimap_filter_closeup.jpg`)
        karesinde minimap'i çıkarır ve:
        - `gatherable_nodes` (yalnızca `sprout_material` 3-yapraklı filiz ikonları),
        - `monster_dots` (reddedilen yuvarlak beyaz canavar noktaları `•`),
        - `hostile_dots` (kırmızı düşman noktaları),
        - `rejected_icons` (reddedilen beyaz tüy ve mavi kanat ikonları),
        - `filter_state` (`Minimap Filter Settings` penceresi durumu),
        - `minimap_bounds` (`(x1, y1, x2, y2)`)
        bilgilerini döndürür.
        """
        try:
            crop, (x1, y1, x2, y2) = self.extract_minimap(frame)
            scan_res = self.scan_minimap_crop(
                crop_bgr=crop,
                offset_xy=(x1, y1),
                frame_shape=frame.shape,
            )
            scan_res["minimap_bounds"] = (x1, y1, x2, y2)
            scan_res["filter_state"] = self.detect_minimap_filter_state(frame)
            return scan_res
        except Exception as exc:
            logger.error("scan_gatherable_materials hatası: %s", exc)
            return {
                "gatherable_nodes": [],
                "monster_dots": [],
                "hostile_dots": [],
                "rejected_icons": [],
                "player_center": (0.0, 0.0),
                "minimap_bounds": (0, 0, 0, 0),
                "filter_state": self.detect_minimap_filter_state(frame)
                if isinstance(frame, np.ndarray)
                else {},
            }

    def scan_odyle_dots(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Yalnızca gerçek toplanabilir materyal (`sprout_material` 3-yapraklı filiz)
        düğümlerini döndürür. Küçük yuvarlak beyaz canavar noktalarını (`•`) ve
        beyaz tüy (`minimap_feather_odyle.png`) ikonlarını ASLA döndürmez!
        """
        res = self.scan_gatherable_materials(frame)
        return list(res.get("gatherable_nodes", []))

    def get_nearest_odyle_offset(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        En yakın gerçek toplanabilir materyalin (`sprout_material`) oyuncuya göre
        `(dx, dy, dist_px)` ofsetini döndürür; yoksa `None` döner.
        """
        nodes = self.scan_odyle_dots(frame)
        if not nodes:
            return None
        nearest = nodes[0]
        return (
            float(nearest["dx"]),
            float(nearest["dy"]),
            float(nearest["dist_px"]),
        )

    def scan_minimap_resources_and_odyle(
        self,
        frame: np.ndarray,
        check_odyle_air: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Minimap üzerindeki gerçek toplanabilir materyal ikonlarını (`sprout_material`) ve
        kırmızı düşman noktalarını (`hostile_dots`) tespit eder.
        Küçük yuvarlak beyaz canavar noktalarını (`•`) ve beyaz tüy (`minimap_feather_odyle.png`)
        ikonlarını kesin olarak filtreler.
        """
        _ = check_odyle_air
        detections: List[Dict[str, Any]] = []
        try:
            res = self.scan_gatherable_materials(frame)
            detections.extend(res.get("hostile_dots", []))
            detections.extend(res.get("gatherable_nodes", []))
            detections.sort(key=lambda d: float(d.get("dist_px", 9999.0)))
            return detections
        except Exception as exc:
            logger.error("scan_minimap_resources_and_odyle hatası: %s", exc)
            return detections

    # ------------------------------------------------------------------
    # 6. Minimap Dokusu Kayması ile Hareket / Takılma (Stuck) Tespiti
    # ------------------------------------------------------------------
    def detect_movement_progress(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Minimap dokusunun ardışık kareler arasındaki kaymasını (`cv2.phaseCorrelate`
        ve yoğunluk farkı ile) ölçerek karakterin gerçekten yürüyüp yürümediğini
        veya bir duvara/kayaya takılıp takılmadığını (`is_moving: bool, speed_score: float`)
        tespit eder.
        """
        try:
            if prev_frame is None or curr_frame is None:
                return False, 0.0

            crop1, _ = self.extract_minimap(prev_frame)
            crop2, _ = self.extract_minimap(curr_frame)

            if crop1.shape != crop2.shape:
                crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))

            ch, cw = crop1.shape[:2]
            top_pad = int(round(ch * 0.16))
            bot_pad = int(round(ch * 0.88))
            side_pad = int(round(cw * 0.08))

            g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)[top_pad:bot_pad, side_pad : cw - side_pad]
            g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)[top_pad:bot_pad, side_pad : cw - side_pad]

            if g1.size == 0 or g2.size == 0:
                return False, 0.0

            f1 = g1.astype(np.float32)
            f2 = g2.astype(np.float32)

            ih, iw = f1.shape[:2]
            arrow_x = int(round(cw * self.PLAYER_CENTER_REL[0])) - side_pad
            arrow_y = int(round(ch * self.PLAYER_CENTER_REL[1])) - top_pad
            mean1 = float(np.mean(f1))
            mean2 = float(np.mean(f2))
            cv2.circle(f1, (arrow_x, arrow_y), 8, mean1, -1)
            cv2.circle(f2, (arrow_x, arrow_y), 8, mean2, -1)

            window = cv2.createHanningWindow((iw, ih), cv2.CV_32F)
            (shift_x, shift_y), _ = cv2.phaseCorrelate(f1, f2, window)
            shift_mag = float(math.hypot(shift_x, shift_y))

            mean_abs_diff = float(np.mean(np.abs(f1 - f2)))

            diff_contrib = max(0.0, (mean_abs_diff - 2.2) * 0.35)
            speed_score = round(shift_mag + diff_contrib, 4)

            is_moving = bool(
                shift_mag >= self.movement_shift_threshold
                or mean_abs_diff >= self.movement_diff_threshold
            )
            return is_moving, speed_score

        except Exception as exc:
            logger.error("detect_movement_progress hatası: %s", exc)
            return False, 0.0
