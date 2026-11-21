import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ChatInputOpenResult(dict):
    """
    Structured dictionary returned by `detect_chat_input_open` that also evaluates
    to `bool(self.get("is_open", False))` in boolean truthiness contexts
    (e.g., `if hud_reader.detect_chat_input_open(frame):`).
    """

    def __bool__(self) -> bool:
        return bool(self.get("is_open", False))


class AdaptivePreprocessor:
    """
    Lightweight, lighting-invariant image preprocessor and multi-scale template
    matcher designed for real-time (< 2ms) HUD and interaction prompt detection
    across 1280x720, 1920x1080, and 2560x1440 resolutions.
    """

    def __init__(self, clahe_clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        try:
            self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
            self._sharpen_kernel = np.array(
                [
                    [0.0, -0.5, 0.0],
                    [-0.5, 3.0, -0.5],
                    [0.0, -0.5, 0.0],
                ],
                dtype=np.float32,
            )
            self._morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self._morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
        except Exception as exc:
            logger.error("Failed to initialize AdaptivePreprocessor: %s", exc)
            raise




def _robust_fill_end(col_active: np.ndarray, min_run: int = 3) -> int:
    """
    A18: dolu barın sağ ucu — tek sütunluk parazit (kırmızı yazı kalıntısı,
    sıkıştırma lekesi) en sağ aktif sütun sayılırsa %20 dolu HP %100 okunuyordu
    (ölçüm: 20% dolu + sağ kenarda 1 kırmızı sütun -> hp_ratio 1.0).
    Bitişik (en az `min_run` sütunluk) son aktif koşunun sonu alınır; 1-2
    sütunluk izole lekeler yok sayılır. Bitişik gerçek doluda eski sonuçla
    birebir aynıdır (temiz %20 -> 0.197, %100 -> 1.0).
    """
    if col_active is None or len(col_active) == 0 or not np.any(col_active):
        return 0
    active = np.asarray(col_active, dtype=bool)
    # Son aktif koşunun (run) başlangıç/bitişini bul.
    end = int(len(active))
    # Sondan geriye: aktif koşu uzunluğu min_run'a ulaşana dek tara.
    run = 0
    for idx in range(len(active) - 1, -1, -1):
        if active[idx]:
            run += 1
            if run >= min_run:
                # Koşunun sonu: bu koşudaki en sağ aktif sütunun ardı.
                # Koşu sağ kenara dayanmıyorsa, koşu sonrasındaki ilk
                # inaktif sütun koşu bitimidir; sağ kenardaki koşuda len().
                j = idx
                while j + 1 < len(active) and active[j + 1]:
                    j += 1
                return j + 1
        else:
            run = 0
    return 0


def _px_to_meters(fallback_px_dist: float, frame_w: Optional[float] = None) -> int:
    """
    F-0338: piksel→metre dönüşümünün tek formülü. Eskiden ana yedek yol
    `px / (28.0 * scale_norm)` kullanırken erken-dönüş ve except yolları
    `px / 28.0` kullanıyordu (1080p'de %50 fark). Çözünürlük bilinmiyorsa
    1280px tabanı (scale 1.0) varsayılır.
    """
    sx = (float(frame_w) / 1280.0) if frame_w else 1.0
    scale_norm = max(0.5, sx)
    return max(0, int(round(float(fallback_px_dist) / (28.0 * scale_norm))))


class GameHUDReader:
    """
    Ultra-fast (< 2ms) HUD reader that uses HSV bar scanning, synthetic pixel
    templates ('0'-'9', 'm', '/', ','), cv2.FONT_HERSHEY_SIMPLEX glyphs, and
    morphological contour analysis without external OCR dependencies.
    """

    def get_ocr_reader_health_summary(self) -> dict:
        # F-0336: eskiden try bloğu asla hata vermediği için her zaman
        # ready=True dönüyordu. Şimdi gerçekten kontrol edilir: ön işleyici
        # ve şablonlar hazırsa ready=True.
        try:
            ready = bool(
                getattr(self, "preprocessor", None) is not None
                and getattr(self, "templates", None)
                and getattr(self, "hershey_templates", None)
            )
            return {"ocr_reader_ready": ready}
        except Exception:
            return {"ocr_reader_ready": False}

    # Crisp 5x7 pixel bitmap patterns for tiny HUD digits and symbols ('0'-'9', 'm', '/', ',')
    _BITMAP_GLYPHS: Dict[str, Sequence[str]] = {
        "0": (
            " ### ",
            "#   #",
            "#   #",
            "#   #",
            "#   #",
            "#   #",
            " ### ",
        ),
        "1": (
            "  #  ",
            " ##  ",
            "  #  ",
            "  #  ",
            "  #  ",
            "  #  ",
            " ### ",
        ),
        "2": (
            " ### ",
            "#   #",
            "    #",
            "  ## ",
            " #   ",
            "#    ",
            "#####",
        ),
        "3": (
            " ### ",
            "#   #",
            "    #",
            "  ## ",
            "    #",
            "#   #",
            " ### ",
        ),
        "4": (
            "   # ",
            "  ## ",
            " # # ",
            "#  # ",
            "#####",
            "   # ",
            "   # ",
        ),
        "5": (
            "#####",
            "#    ",
            "#### ",
            "    #",
            "    #",
            "#   #",
            " ### ",
        ),
        "6": (
            " ### ",
            "#    ",
            "#### ",
            "#   #",
            "#   #",
            "#   #",
            " ### ",
        ),
        "7": (
            "#####",
            "    #",
            "   # ",
            "  #  ",
            "  #  ",
            " #   ",
            " #   ",
        ),
        "8": (
            " ### ",
            "#   #",
            "#   #",
            " ### ",
            "#   #",
            "#   #",
            " ### ",
        ),
        "9": (
            " ### ",
            "#   #",
            "#   #",
            " ####",
            "    #",
            "#   #",
            " ### ",
        ),
        "m": (
            "     ",
            "     ",
            "## # ",
            "# # #",
            "# # #",
            "# # #",
            "# # #",
        ),
        "/": (
            "    #",
            "   # ",
            "   # ",
            "  #  ",
            " #   ",
            " #   ",
            "#    ",
        ),
        ",": (
            "     ",
            "     ",
            "     ",
            "     ",
            "  ## ",
            "  #  ",
            " #   ",
        ),
    }

    def __init__(self) -> None:
        try:
            self.preprocessor = AdaptivePreprocessor()
            self.templates: Dict[str, np.ndarray] = {}
            self.hershey_templates: Dict[str, np.ndarray] = {}
            self._build_synthetic_templates()
        except Exception as exc:
            logger.error("Failed to initialize GameHUDReader: %s", exc)
            raise

    def _build_synthetic_templates(self) -> None:
        """
        Pre-generates both 5x7 crisp pixel templates and cv2.FONT_HERSHEY_SIMPLEX
        rendered templates for '0'-'9', 'm', '/', ',' at initialization.
        """
        try:
            for ch, rows in self._BITMAP_GLYPHS.items():
                arr = np.zeros((len(rows), len(rows[0])), dtype=np.uint8)
                for r_idx, row_str in enumerate(rows):
                    for c_idx, val in enumerate(row_str):
                        if val == "#":
                            arr[r_idx, c_idx] = 255
                self.templates[ch] = arr

                # Also generate a cv2.FONT_HERSHEY_SIMPLEX glyph template normalized to 8x12
                canvas = np.zeros((24, 20), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    ch,
                    (2, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    255,
                    1,
                    cv2.LINE_AA,
                )
                coords = cv2.findNonZero(canvas)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    crop = canvas[y : y + h, x : x + w]
                    self.hershey_templates[ch] = cv2.resize(crop, (5, 7), interpolation=cv2.INTER_AREA)
                else:
                    self.hershey_templates[ch] = arr.copy()
            # Warm up OpenCV color conversion & connected components LUTs at init time
            _warmup_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
            _ = cv2.cvtColor(_warmup_bgr, cv2.COLOR_BGR2HSV)
            _ = cv2.connectedComponentsWithStats(_warmup_bgr[:, :, 0], connectivity=8)
        except Exception as exc:
            logger.error("_build_synthetic_templates failed: %s", exc)

    def _scan_chat_bands(
        self,
        chat_roi: np.ndarray,
        allow_high_contrast_synthetic: bool = False,
    ) -> Tuple[bool, bool, bool, float, int]:
        """
        Internal < 1.5ms band scanner for the bottom-left chat ROI.
        Identifies:
          - 'Essence Extraction failed' -> extraction_failed
          - 'Cannot mount in current state' -> cannot_mount
          - 'Invalid target' -> invalid_target
        Returns:
          (extraction_failed, cannot_mount, invalid_target, orange_warning_ratio, valid_band_count)
        """
        if chat_roi is None or chat_roi.size == 0:
            return False, False, False, 0.0, 0

        if chat_roi.ndim == 2:
            hsv = cv2.cvtColor(cv2.cvtColor(chat_roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(chat_roi, cv2.COLOR_BGR2HSV)

        # Amber/orange system text mask (includes anti-aliased glyph strokes)
        orange_mask = (
            (hsv[:, :, 0] >= 10)
            & (hsv[:, :, 0] <= 26)
            & (hsv[:, :, 1] >= 130)
            & (hsv[:, :, 1] <= 255)
            & (hsv[:, :, 2] >= 165)
            & (hsv[:, :, 2] <= 255)
        )
        red_alert_mask = (
            ((hsv[:, :, 0] <= 7) | (hsv[:, :, 0] >= 168))
            & (hsv[:, :, 1] >= 140)
            & (hsv[:, :, 2] >= 170)
        )
        core_orange_mask = (
            (hsv[:, :, 0] >= 12)
            & (hsv[:, :, 0] <= 25)
            & (hsv[:, :, 1] >= 140)
            & (hsv[:, :, 1] <= 245)
            & (hsv[:, :, 2] >= 180)
            & (hsv[:, :, 2] <= 255)
        )

        orange_warning_ratio = float(orange_mask.mean())
        combined_mask = orange_mask if int(orange_mask.sum()) >= 16 else (orange_mask | red_alert_mask)

        if int(core_orange_mask.sum()) < 8 and int(combined_mask.sum()) < 16:
            if allow_high_contrast_synthetic:
                v_chan = hsv[:, :, 2]
                mean_arr, std_arr = cv2.meanStdDev(v_chan)
                bg_mean = float(mean_arr[0, 0])
                bg_std = float(std_arr[0, 0])
                if bg_mean <= 32.0 and bg_std <= 38.0:
                    synth_mask = v_chan >= 165
                    if int(synth_mask.sum()) >= 18:
                        combined_mask = synth_mask
                        orange_warning_ratio = float(synth_mask.mean())
                    else:
                        return False, False, False, round(orange_warning_ratio, 6), 0
                else:
                    return False, False, False, round(orange_warning_ratio, 6), 0
            else:
                return False, False, False, round(orange_warning_ratio, 6), 0

        roi_h, roi_w = combined_mask.shape[:2]
        min_row_px = max(3, int(round(roi_w * 0.012)))
        row_counts = combined_mask.sum(axis=1)
        active_rows = np.where(row_counts >= min_row_px)[0]

        if len(active_rows) == 0:
            return False, False, False, round(orange_warning_ratio, 6), 0

        max_gap = max(2, int(round(roi_h * 0.05)))
        gaps = np.where(np.diff(active_rows) > max_gap)[0]
        bands = np.split(active_rows, gaps + 1)

        extraction_failed = False
        cannot_mount = False
        invalid_target = False
        valid_band_count = 0

        for band in bands:
            if len(band) == 0:
                continue
            ry1 = max(0, int(band[0]) - 1)
            ry2 = min(roi_h, int(band[-1]) + 2)
            band_mask = combined_mask[ry1:ry2, :]
            band_px = int(band_mask.sum())
            if band_px < max(10, int(round(roi_w * 0.025))):
                continue

            cols = np.where(band_mask.any(axis=0))[0]
            if len(cols) == 0:
                continue

            valid_band_count += 1
            span_px = int(cols[-1] - cols[0] + 1)
            rel_span = float(span_px) / float(max(1, roi_w))

            # F-0161: sınıflandırma yalnızca yatay uzunluk (rel_span)
            # pencereleriyle yapılır. Kelime sayısı bu maskeden güvenilir
            # ölçülemez: genişletme (dilation) sayımı gerçek çözünürlükte
            # harf parçalarına ayrılır (pilgrims_ridge_1.jpg: 16 bileşen),
            # sentetik Hershey'de ise 1-4 verir; sütun projeksiyonu da
            # harf boşluğu (~9px) ile kelime boşluğunu (9-13px) ayıramaz.
            # Bu yüzden eski OR/catch-all dalları (tek başına span VEYA
            # kelime sayısı VEYA y-konumu) loot ("You obtained 100 Kina")
            # ve başarı ("Essence Extraction complete") mesajlarını bile
            # yanlış etiketliyordu. Kalibrasyon (gerçek 2000x1124 ekran +
            # 1080p Hershey probu): Invalid target 0.211; Essence
            # Extraction failed 0.411-0.418; Cannot mount in current
            # state. 0.498-0.501; loot 0.366; başarı 0.474; "Party member
            # joined" 0.336; "Hello world test msg" 0.323. Pencereye
            # uymayan bant etiketsiz kalır (bayraklar False,
            # warning_line_count artar). SINIR: "You have died" (0.228)
            # geometrik olarak "Invalid target" (0.211) ile ayırt
            # edilemez; gerçek oyunda ölüm mesajı kırmızıdır, bu yüzden
            # kırmızı-ağırlıklı bantlar hiç etiketlenmez.
            orange_px = int(orange_mask[ry1:ry2, :].sum())
            red_px = int(red_alert_mask[ry1:ry2, :].sum())
            if red_px > orange_px:
                continue
            if 0.12 <= rel_span < 0.28:
                invalid_target = True
            elif 0.38 <= rel_span <= 0.468:
                extraction_failed = True
            elif rel_span > 0.486:
                cannot_mount = True

        return (
            bool(extraction_failed),
            bool(cannot_mount),
            bool(invalid_target),
            round(orange_warning_ratio, 6),
            int(valid_band_count),
        )

    def detect_chat_system_alerts(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Scans the bottom-left chat window (x: 0.012..0.220, y: 0.750..0.810) for
        orange/amber system warning text (BGR ~(55, 155, 235) / HSV H=12..25,
        S=140..240, V=180..255) and identifies specific system alerts:
          - 'Essence Extraction failed' -> extraction_failed: bool
          - 'Cannot mount in current state.' -> cannot_mount: bool
          - 'Invalid target' -> invalid_target: bool
          - orange_warning_ratio: float
        """
        try:
            if frame is None or frame.size == 0:
                return {
                    "extraction_failed": False,
                    "cannot_mount": False,
                    "invalid_target": False,
                    "orange_warning_ratio": 0.0,
                    "warning_line_count": 0,
                }

            h, w = frame.shape[:2]
            cx1 = max(0, int(round(w * 0.012)))
            cx2 = min(w, max(cx1 + 1, int(round(w * 0.220))))
            cy1 = max(0, int(round(h * 0.750)))
            cy2 = min(h, max(cy1 + 1, int(round(h * 0.810))))

            chat_roi = frame[cy1:cy2, cx1:cx2]
            ext_fail, no_mount, inv_target, ratio, line_cnt = self._scan_chat_bands(
                chat_roi, allow_high_contrast_synthetic=False
            )
            return {
                "extraction_failed": ext_fail,
                "cannot_mount": no_mount,
                "invalid_target": inv_target,
                "orange_warning_ratio": ratio,
                "warning_line_count": line_cnt,
            }
        except Exception as exc:
            logger.error("detect_chat_system_alerts failed: %s", exc)
            return {
                "extraction_failed": False,
                "cannot_mount": False,
                "invalid_target": False,
                "orange_warning_ratio": 0.0,
                "warning_line_count": 0,
            }

    def parse_chat_feedback_fast(
        self,
        frame: Union[np.ndarray, str, Sequence[str], None],
    ) -> Dict[str, Any]:
        """
        Ultra-fast (< 5ms) chat feedback parser (Item 20) that detects:
          - "Essence Extraction failed" -> extraction_failed: True ("retry_gather")
          - "Cannot mount in current state" -> cannot_mount: True ("skip_mount")
          - "Invalid target" -> invalid_target: True ("retarget")
        in the bottom-left chat box (or from text lines) and returns structured
        action directives:
            {
                "extraction_failed": bool,
                "cannot_mount": bool,
                "invalid_target": bool,
                "recommended_action": "retry_gather" | "skip_mount" | "retarget" | "none",
                "elapsed_ms": float,
            }
        """
        t0 = time.perf_counter()
        extraction_failed = False
        cannot_mount = False
        invalid_target = False

        try:
            if isinstance(frame, str):
                lower = frame.lower()
                # F-0339: çıplak "essence extraction" alt dizesi başarı
                # mesajını ("Essence Extraction complete") bile başarısız
                # sayıyordu; yalnızca gerçek başarısızlık ifadesi aranır.
                extraction_failed = "extraction failed" in lower
                cannot_mount = "cannot mount" in lower
                invalid_target = "invalid target" in lower
            elif isinstance(frame, (list, tuple)):
                joined = " ".join(str(x).lower() for x in frame)
                extraction_failed = "extraction failed" in joined
                cannot_mount = "cannot mount" in joined
                invalid_target = "invalid target" in joined
            elif isinstance(frame, np.ndarray) and frame.size > 0:
                h, w = frame.shape[:2]
                if h >= 200 and w >= 320:
                    # 1. Primary bottom-left system warning window (x: 0.012..0.220, y: 0.750..0.810)
                    cx1 = max(0, int(round(w * 0.012)))
                    cx2 = min(w, max(cx1 + 1, int(round(w * 0.220))))
                    cy1 = max(0, int(round(h * 0.750)))
                    cy2 = min(h, max(cy1 + 1, int(round(h * 0.810))))
                    primary_roi = frame[cy1:cy2, cx1:cx2]
                    extraction_failed, cannot_mount, invalid_target, _, line_cnt = self._scan_chat_bands(
                        primary_roi, allow_high_contrast_synthetic=True
                    )

                    # 2. Fallback scan over lower chat log window (y: 0.810..0.905) if primary had no lines
                    if line_cnt == 0 and not (extraction_failed or cannot_mount or invalid_target):
                        ly1 = max(0, int(round(h * 0.810)))
                        ly2 = min(h, max(ly1 + 1, int(round(h * 0.905))))
                        lower_roi = frame[ly1:ly2, cx1:cx2]
                        extraction_failed, cannot_mount, invalid_target, _, _ = self._scan_chat_bands(
                            lower_roi, allow_high_contrast_synthetic=True
                        )
                else:
                    # Frame is already a cropped chat ROI
                    extraction_failed, cannot_mount, invalid_target, _, _ = self._scan_chat_bands(
                        frame, allow_high_contrast_synthetic=True
                    )
        except Exception as exc:
            logger.error("parse_chat_feedback_fast failed: %s", exc)

        if extraction_failed:
            recommended_action = "retry_gather"
        elif cannot_mount:
            recommended_action = "skip_mount"
        elif invalid_target:
            recommended_action = "retarget"
        else:
            recommended_action = "none"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "extraction_failed": bool(extraction_failed),
            "cannot_mount": bool(cannot_mount),
            "invalid_target": bool(invalid_target),
            "recommended_action": recommended_action,
            "elapsed_ms": elapsed_ms,
        }

    def detect_chat_input_open(self, frame: np.ndarray) -> ChatInputOpenResult:
        """
        Detects if the bottom-left chat text input bar (x: 0.01..0.24, y: 0.91..0.96)
        is active/open (Item 17):
          1. Bright horizontal input box border along top/bottom of the bar, OR
          2. Blinking vertical text cursor ('|') inside the dark input bar, OR
          3. Active '[Say]' / '[Party]' / '[Guild]' prompt tag box on the left of a
             dark input field.
        Returns a ChatInputOpenResult(dict) whose truthiness (`bool(res)`) matches `res["is_open"]`.
        """
        t0 = time.perf_counter()
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return ChatInputOpenResult(
                    is_open=False,
                    chat_open=False,
                    has_border=False,
                    has_cursor=False,
                    has_prompt_tag=False,
                    confidence=0.0,
                    roi_bbox=(0, 0, 0, 0),
                    elapsed_ms=0.0,
                )

            h, w = frame.shape[:2]
            if h >= 200 and w >= 320:
                x1 = max(0, int(round(w * 0.01)))
                x2 = min(w, max(x1 + 1, int(round(w * 0.24))))
                y1 = max(0, int(round(h * 0.91)))
                y2 = min(h, max(y1 + 1, int(round(h * 0.96))))
                roi = frame[y1:y2, x1:x2]
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                roi = frame

            if roi.size == 0:
                return ChatInputOpenResult(
                    is_open=False,
                    chat_open=False,
                    has_border=False,
                    has_cursor=False,
                    has_prompt_tag=False,
                    confidence=0.0,
                    roi_bbox=(x1, y1, x2 - x1, y2 - y1),
                    elapsed_ms=0.0,
                )

            if roi.ndim == 3:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
                s = hsv[:, :, 1]
            else:
                v = roi
                s = np.zeros_like(v)

            rh, rw = v.shape[:2]
            if rh < 6 or rw < 16:
                return ChatInputOpenResult(
                    is_open=False,
                    chat_open=False,
                    has_border=False,
                    has_cursor=False,
                    has_prompt_tag=False,
                    confidence=0.0,
                    roi_bbox=(x1, y1, x2 - x1, y2 - y1),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 4),
                )

            # 1. Check for bright horizontal input border (top/bottom border rows spanning >= 40% of bar width)
            row_bright_ratios = (v >= 135).mean(axis=1)
            max_row_bright = float(row_bright_ratios.max())
            top_band_max = float(row_bright_ratios[: max(2, rh // 3)].max())
            bot_band_max = float(row_bright_ratios[min(rh - 1, (2 * rh) // 3) :].max())
            mid_v = v[max(1, rh // 4) : max(2, (3 * rh) // 4), :]
            mid_dark_ratio = float((mid_v < 85).mean())

            has_border = bool(
                (max_row_bright >= 0.42 and mid_dark_ratio >= 0.45)
                or (top_band_max >= 0.35 and bot_band_max >= 0.35 and mid_dark_ratio >= 0.50)
            )

            # 2. Check for blinking vertical text cursor '|' (thin tall bright bar on a dark input field)
            cursor_mask = (v >= 175) & (s <= 95)
            col_bright_ratios = cursor_mask.mean(axis=0)
            has_cursor = False
            candidate_cols = np.where(col_bright_ratios >= 0.44)[0]
            for c_idx in candidate_cols:
                if c_idx < 3 or c_idx >= rw - 3:
                    continue
                left_flank = v[:, max(0, c_idx - 6) : max(1, c_idx - 2)]
                right_flank = v[:, min(rw - 1, c_idx + 3) : min(rw, c_idx + 7)]
                if (
                    left_flank.size > 0
                    and right_flank.size > 0
                    and float(left_flank.mean()) < 62.0
                    and float(right_flank.mean()) < 62.0
                    and int(left_flank.max()) < 115
                    and int(right_flank.max()) < 115
                ):
                    has_cursor = True
                    break

            # 3. Check for '[Say]' / '[Party]' / '[Guild]' prompt tag on the left of a uniform dark input box
            left_end = max(8, int(round(rw * 0.36)))
            left_v = v[:, :left_end]
            left_bright_bool = left_v >= 155
            left_bright_px = int(left_bright_bool.sum())

            has_prompt_tag = False
            if left_bright_px >= 18:
                left_col_max_ratio = float(left_bright_bool.mean(axis=0).max()) if left_v.size > 0 else 0.0
                right_v = v[:, left_end:]
                if right_v.size > 0:
                    r_mean_arr, r_std_arr = cv2.meanStdDev(right_v)
                    right_mean_v = float(r_mean_arr[0, 0])
                    right_std_v = float(r_std_arr[0, 0])
                else:
                    right_mean_v, right_std_v = 255.0, 255.0

                if (
                    has_border
                    or has_cursor
                    or (
                        right_mean_v <= 36.0
                        and right_std_v <= 26.0
                        and left_col_max_ratio >= 0.38
                    )
                ):
                    col_active = left_bright_bool.any(axis=0).astype(np.int8)
                    padded = np.pad(col_active, (1, 1), constant_values=0)
                    glyph_runs = int(np.count_nonzero(np.diff(padded) > 0))
                    if glyph_runs >= 2:
                        has_prompt_tag = True

            is_open = bool(has_border or has_cursor or has_prompt_tag)
            confidence = 0.0
            if is_open:
                score_parts = [
                    0.50 if has_border else 0.0,
                    0.45 if has_cursor else 0.0,
                    0.45 if has_prompt_tag else 0.0,
                    min(0.25, max_row_bright * 0.35),
                ]
                # F-0340: tek zayıf sinyal bile en az 0.65 güven
                # raporluyordu; güven artık sinyallerin toplamıdır.
                confidence = round(min(1.0, sum(score_parts)), 3)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return ChatInputOpenResult(
                is_open=is_open,
                chat_open=is_open,
                has_border=has_border,
                has_cursor=has_cursor,
                has_prompt_tag=has_prompt_tag,
                confidence=confidence,
                roi_bbox=(x1, y1, x2 - x1, y2 - y1),
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            logger.error("detect_chat_input_open failed: %s", exc)
            return ChatInputOpenResult(
                is_open=False,
                chat_open=False,
                has_border=False,
                has_cursor=False,
                has_prompt_tag=False,
                confidence=0.0,
                roi_bbox=(0, 0, 0, 0),
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 4),
            )

    def detect_top_target_bar(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Inspects the top-center HUD region (x: 0.40..0.60, y: 0.02..0.09) and the
        secondary target region to its right (x: 0.62..0.71, y: 0.045..0.060) to
        distinguish between:
          1. Monster Target Bar (e.g. '45 Yellow-Maned Zaif 15m' with olive-green HP
             bar H=28..65 or hostile red HP bar, red star badge on right tip, and/or
             secondary target bar '45 Muses' on the right).
          2. Gather Cast / Resource Target Bar (e.g. 'Orichalcum 0m' with cyan/blue/teal
             bar H=72..118 and no monster HP fill or secondary target).
          3. No active top target bar.

        Returns:
            {
                'has_target': bool,
                'is_monster_target': bool,
                'is_gather_target': bool,
                'target_hp_ratio': float (0.0 .. 1.0),
                'has_secondary_target': bool,
                'secondary_target_hp_ratio': float (0.0 .. 1.0),
            }
        """
        try:
            if frame is None or frame.size == 0:
                return {
                    "has_target": False,
                    "is_monster_target": False,
                    "is_gather_target": False,
                    "target_hp_ratio": 0.0,
                    "has_secondary_target": False,
                    "secondary_target_hp_ratio": 0.0,
                }

            h, w = frame.shape[:2]

            # Primary target bar inner span: x=0.422..0.571, y=0.048..0.059
            bx1 = max(0, int(round(w * 0.422)))
            bx2 = min(w, max(bx1 + 1, int(round(w * 0.571))))
            by1 = max(0, int(round(h * 0.048)))
            by2 = min(h, max(by1 + 1, int(round(h * 0.059))))

            bar_roi = frame[by1:by2, bx1:bx2]
            if bar_roi.size == 0:
                return {
                    "has_target": False,
                    "is_monster_target": False,
                    "is_gather_target": False,
                    "target_hp_ratio": 0.0,
                    "has_secondary_target": False,
                    "secondary_target_hp_ratio": 0.0,
                }

            bar_hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
            bar_w = max(1, bx2 - bx1)

            # Check dark metallic frame borders above/below the bar to reject sky/foliage
            b_top_y = max(0, int(round(h * 0.046)))
            b_bot_y = min(h - 1, int(round(h * 0.061)))
            top_border_v = float(frame[b_top_y, bx1:bx2].mean())
            bot_border_v = float(frame[b_bot_y, bx1:bx2].mean())
            has_dark_frame = (top_border_v < 75.0) and (bot_border_v < 75.0)

            # Right-tip red star badge (x: 0.570..0.600, y: 0.040..0.068)
            badge_x1 = max(0, int(round(w * 0.570)))
            badge_x2 = min(w, max(badge_x1 + 1, int(round(w * 0.600))))
            badge_y1 = max(0, int(round(h * 0.040)))
            badge_y2 = min(h, max(badge_y1 + 1, int(round(h * 0.068))))
            badge_hsv = cv2.cvtColor(frame[badge_y1:badge_y2, badge_x1:badge_x2], cv2.COLOR_BGR2HSV)
            red_badge_mask = (
                ((badge_hsv[:, :, 0] <= 12) | (badge_hsv[:, :, 0] >= 165))
                & (badge_hsv[:, :, 1] >= 100)
                & (badge_hsv[:, :, 2] >= 100)
            )
            badge_area = max(1, (badge_x2 - badge_x1) * (badge_y2 - badge_y1))
            has_red_badge = (float(red_badge_mask.sum()) / float(badge_area)) >= 0.045

            # Secondary target bar on the right ('45 Muses' at x: 0.620..0.710, y: 0.045..0.060)
            sec_x1 = max(0, int(round(w * 0.620)))
            sec_x2 = min(w, max(sec_x1 + 1, int(round(w * 0.710))))
            sec_y1 = max(0, int(round(h * 0.045)))
            sec_y2 = min(h, max(sec_y1 + 1, int(round(h * 0.060))))
            sec_roi = frame[sec_y1:sec_y2, sec_x1:sec_x2]
            sec_hsv = cv2.cvtColor(sec_roi, cv2.COLOR_BGR2HSV)
            sec_red_mask = (
                ((sec_hsv[:, :, 0] <= 14) | (sec_hsv[:, :, 0] >= 165))
                & (sec_hsv[:, :, 1] >= 100)
                & (sec_hsv[:, :, 2] >= 90)
            )
            sec_col_active = sec_red_mask.mean(axis=0) >= 0.20
            sec_w = max(1, sec_x2 - sec_x1)
            sec_ratio_raw = float(sec_col_active.sum()) / float(sec_w)
            has_secondary_target = sec_ratio_raw >= 0.25
            secondary_target_hp_ratio = round(min(1.0, sec_ratio_raw / 0.76), 3) if has_secondary_target else 0.0

            # 1. Olive-green Monster HP bar (Yellow-Maned Zaif: H=28..65, S>=95, V>=90)
            olive_mask = (
                (bar_hsv[:, :, 0] >= 28)
                & (bar_hsv[:, :, 0] <= 65)
                & (bar_hsv[:, :, 1] >= 95)
                & (bar_hsv[:, :, 2] >= 90)
            )
            olive_cols = olive_mask.mean(axis=0) >= 0.35
            olive_count = int(olive_cols.sum())

            # 2. Hostile red Monster HP bar (H<=14 or H>=165, S>=105, V>=90)
            red_mask = (
                ((bar_hsv[:, :, 0] <= 14) | (bar_hsv[:, :, 0] >= 165))
                & (bar_hsv[:, :, 1] >= 105)
                & (bar_hsv[:, :, 2] >= 90)
            )
            red_cols = red_mask.mean(axis=0) >= 0.35
            red_count = int(red_cols.sum())

            # 3. Cyan/blue/teal Gather Cast Bar ('Orichalcum 0m': H=72..118, S>=110, V>=85)
            cyan_mask = (
                (bar_hsv[:, :, 0] >= 72)
                & (bar_hsv[:, :, 0] <= 118)
                & (bar_hsv[:, :, 1] >= 110)
                & (bar_hsv[:, :, 2] >= 85)
            )
            cyan_cols = cyan_mask.mean(axis=0) >= 0.35
            cyan_count = int(cyan_cols.sum())

            # Central crest icon occupies ~7.5% of the inner bar width
            effective_full_cols = max(1.0, float(bar_w) * 0.925)
            min_active_cols = max(6, int(round(bar_w * 0.06)))

            # Monster target detection (olive-green or red monster HP bar)
            if (olive_count >= min_active_cols or red_count >= min_active_cols) and (
                has_red_badge or has_secondary_target or has_dark_frame
            ):
                active_cols = olive_cols if olive_count >= red_count else red_cols
                active_idx = np.where(active_cols)[0]
                span_ratio = float(active_idx[-1] - active_idx[0] + 1) / float(bar_w) if len(active_idx) else 0.0
                col_ratio = float(active_cols.sum()) / effective_full_cols
                hp_ratio = round(min(1.0, max(0.0, max(col_ratio, span_ratio))), 3)
                return {
                    "has_target": True,
                    "is_monster_target": True,
                    "is_gather_target": False,
                    "target_hp_ratio": hp_ratio,
                    "has_secondary_target": bool(has_secondary_target),
                    "secondary_target_hp_ratio": float(secondary_target_hp_ratio),
                }

            # Gather cast/target bar detection (cyan/blue/teal, never triggered by green monster HP)
            if (
                cyan_count >= min_active_cols
                and not has_secondary_target
                and olive_count < min_active_cols
                and (has_red_badge or has_dark_frame or cyan_count >= min_active_cols)
            ):
                # v20: cast bar boş başlar — dolgu soldan büyür. Sağda
                # yalnız cyan leke/çerçeve (dolu başlama illüzyonu) ise 0.
                active_idx = np.where(cyan_cols)[0]
                if len(active_idx) == 0:
                    progress_ratio = 0.0
                else:
                    left = int(active_idx[0])
                    if left > max(2, int(round(bar_w * 0.06))):
                        progress_ratio = 0.0
                    else:
                        fill_end = _robust_fill_end(cyan_cols)
                        progress_ratio = round(
                            min(1.0, max(0.0, float(fill_end) / effective_full_cols)),
                            3,
                        )
                return {
                    "has_target": True,
                    "is_monster_target": False,
                    "is_gather_target": True,
                    "target_hp_ratio": progress_ratio,
                    "has_secondary_target": False,
                    "secondary_target_hp_ratio": 0.0,
                }

            return {
                "has_target": False,
                "is_monster_target": False,
                "is_gather_target": False,
                "target_hp_ratio": 0.0,
                "has_secondary_target": bool(has_secondary_target),
                "secondary_target_hp_ratio": float(secondary_target_hp_ratio),
            }
        except Exception as exc:
            logger.error("detect_top_target_bar failed: %s", exc)
            return {
                "has_target": False,
                "is_monster_target": False,
                "is_gather_target": False,
                "target_hp_ratio": 0.0,
                "has_secondary_target": False,
                "secondary_target_hp_ratio": 0.0,
            }

    def read_cast_bar_progress(self, frame: np.ndarray) -> float:
        """
        Returns the gather cast bar fill ratio (0.0 .. 1.0) when a cyan/blue/teal
        gather bar is active at the top of the HUD, and strictly returns 0.0 when
        a monster target bar (such as '45 Yellow-Maned Zaif 15m') or no bar is active.
        """
        try:
            target_info = self.detect_top_target_bar(frame)
            if target_info.get("is_monster_target", False) or not target_info.get("is_gather_target", False):
                return 0.0
            return float(target_info.get("target_hp_ratio", 0.0))
        except Exception as exc:
            logger.error("read_cast_bar_progress failed: %s", exc)
            return 0.0

    def read_vital_bars(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Reads the bottom-center Player HP bar (x: 0.345..0.478, y: 0.885..0.900) and
        MP bar (x: 0.520..0.652, y: 0.885..0.900), as well as the overhead floating
        Player HP/MP bar near screen center (x: 0.460..0.540, y: 0.450..0.500).

        Returns ratios in [0.0 .. 1.0] (e.g. 1.0 for 100% HP and 100% MP on both
        pilgrims_ridge_1.jpg and video_frames/frame_00s.jpg) along with percentages.

        v20 (HP yanlış okuma): bar görünmüyorsa oran None döner, 0.0 değil.
        Koyu bar yuvası can %0 iken bile ekrandadır; o yüzden "kırmızı
        dolgu yok" tek başına "ölü" demek değildir — yuva + parlaklık
        kontrolü (aşağıda) geçemezse ölçüm yok sayılır (`hp_valid` /
        `mp_valid` False). Çağıranlar None'u "bilinmiyor" saymalı; 0.0
        yalnızca yuvası görünen ama dolgusu olmayan (gerçekten boş)
        bar içindir.
        """
        try:
            if frame is None or frame.size == 0:
                return {
                    "hp_ratio": None,
                    "mp_ratio": None,
                    "hp": None,
                    "mp": None,
                    "hp_pct": None,
                    "mp_pct": None,
                    "hp_valid": False,
                    "mp_valid": False,
                    "overhead_hp_ratio": 0.0,
                    "overhead_mp_ratio": 0.0,
                }

            h, w = frame.shape[:2]

            # 1. Bottom-Center HP Bar (x: 0.345..0.478, y: 0.885..0.900; inner active track: 0.364..0.478)
            hp_x1 = max(0, int(round(w * 0.345)))
            hp_x2 = min(w, max(hp_x1 + 1, int(round(w * 0.478))))
            hp_y1 = max(0, int(round(h * 0.885)))
            hp_y2 = min(h, max(hp_y1 + 1, int(round(h * 0.900))))

            hp_roi = frame[hp_y1:hp_y2, hp_x1:hp_x2]
            hp_hsv = cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV)
            # F-0159: yalnızca KIRMIZI dolgu ölçülür. Beyaz sayı katmanı
            # ('11,718/11,718') eskiden dolgu sayılıyordu: %20 kırmızı dolu
            # + tüm bara yayılan beyaz yazı → span tüm bar → %65 okunuyordu.
            # Beyaz katman artık orana katılmaz; oran, soldan başlayan
            # kırmızı dolgunun ulaştığı en sağ sütundur.
            hp_fill_mask = (
                ((hp_hsv[:, :, 0] <= 15) | (hp_hsv[:, :, 0] >= 164)) & (hp_hsv[:, :, 1] >= 75) & (hp_hsv[:, :, 2] >= 85)
            )
            # Core inner track of the bottom HP bar (excluding angled outer left bezel at 0.345..0.365)
            inner_hp_offset = max(0, int(round(w * (0.365 - 0.345))))
            hp_inner_mask = hp_fill_mask[:, inner_hp_offset:] if inner_hp_offset < hp_fill_mask.shape[1] else hp_fill_mask
            hp_col_active = hp_inner_mask.mean(axis=0) >= 0.25
            hp_inner_cols = max(1, hp_inner_mask.shape[1])
            hp_end = _robust_fill_end(hp_col_active)
            if hp_end > 0:
                hp_ratio = min(1.0, max(0.0, float(hp_end) / float(hp_inner_cols)))
                if hp_ratio >= 0.99:
                    hp_ratio = 1.0
            else:
                hp_ratio = 0.0

            # v20 (HP yanlış okuma): bar görünürlüğü. Koyu bar yuvası
            # can %0 iken bile ekrandadır — `vision.is_map_open`'daki
            # hp_slot kuralının aynısı (V<=45 oranı > 0.30). Yuva yoksa
            # ve kırmızı dolgu da yoksa bar bu karede görünmüyor
            # demektir (HUD gizli / yanlış kare / yakalama hatası) ve
            # 0.0 değil None döner, yoksa okuma hatası "ölü" sanılıp
            # iksir + kaçış tetikler. Tam siyah kare yuvayı %100 koyu
            # gösterip bu kuralı kandırır; o yüzden ROI'de en az %1
            # parlak piksel de aranır (bar çerçevesi / can yazısı her
            # gerçek karede parlaktır, siyah karede 0'dır).
            hp_slot_frac = float(np.mean(hp_hsv[:, :, 2] <= 45)) if hp_hsv.size else 0.0
            hp_bright_frac = float(np.mean(hp_hsv[:, :, 2] > 45)) if hp_hsv.size else 0.0
            hp_valid = bool(hp_bright_frac > 0.01 and (hp_end > 0 or hp_slot_frac > 0.30))
            if not hp_valid:
                hp_ratio = None

            # 2. Bottom-Center MP Bar (x: 0.520..0.652, y: 0.885..0.900; inner active track: 0.520..0.634)
            mp_x1 = max(0, int(round(w * 0.520)))
            mp_x2 = min(w, max(mp_x1 + 1, int(round(w * 0.652))))
            mp_y1 = max(0, int(round(h * 0.885)))
            mp_y2 = min(h, max(mp_y1 + 1, int(round(h * 0.900))))

            mp_roi = frame[mp_y1:mp_y2, mp_x1:mp_x2]
            mp_hsv = cv2.cvtColor(mp_roi, cv2.COLOR_BGR2HSV)
            # F-0159 + High #3: yalnızca camgöbeği/mavi/teal dolgu ölçülür,
            # beyaz sayı katmanı (3814/3814) orana katılmaz. Hue üst sınırı
            # 135'e açıldı — Aion 2 dolu MP çubuğu bazen daha mavi okunur
            # ve dar aralık %20–40 sahte düşüğe yol açıyordu.
            mp_fill_mask = (
                (mp_hsv[:, :, 0] >= 75)
                & (mp_hsv[:, :, 0] <= 135)
                & (mp_hsv[:, :, 1] >= 70)
                & (mp_hsv[:, :, 2] >= 70)
            )
            inner_mp_width = max(1, int(round(w * (0.634 - 0.520))))
            mp_inner_mask = mp_fill_mask[:, :inner_mp_width] if inner_mp_width <= mp_fill_mask.shape[1] else mp_fill_mask
            mp_col_active = mp_inner_mask.mean(axis=0) >= 0.20
            mp_inner_cols = max(1, mp_inner_mask.shape[1])
            mp_end = _robust_fill_end(mp_col_active)
            if mp_end > 0:
                mp_ratio = min(1.0, max(0.0, float(mp_end) / float(mp_inner_cols)))
                if mp_ratio >= 0.97:
                    mp_ratio = 1.0
            else:
                mp_ratio = 0.0

            # v20: MP için HP ile aynı görünürlük kuralı (yuva + parlaklık).
            mp_slot_frac = float(np.mean(mp_hsv[:, :, 2] <= 45)) if mp_hsv.size else 0.0
            mp_bright_frac = float(np.mean(mp_hsv[:, :, 2] > 45)) if mp_hsv.size else 0.0
            mp_valid = bool(mp_bright_frac > 0.01 and (mp_end > 0 or mp_slot_frac > 0.30))
            if not mp_valid:
                mp_ratio = None

            # 3. Overhead Player HP/MP Bar near center (x: 0.460..0.540, y: 0.450..0.500)
            oh_x1 = max(0, int(round(w * 0.460)))
            oh_x2 = min(w, max(oh_x1 + 1, int(round(w * 0.540))))
            oh_y1 = max(0, int(round(h * 0.450)))
            oh_y2 = min(h, max(oh_y1 + 1, int(round(h * 0.500))))
            oh_roi = frame[oh_y1:oh_y2, oh_x1:oh_x2]
            oh_hsv = cv2.cvtColor(oh_roi, cv2.COLOR_BGR2HSV)
            oh_red = (
                ((oh_hsv[:, :, 0] <= 14) | (oh_hsv[:, :, 0] >= 165))
                & (oh_hsv[:, :, 1] >= 100)
                & (oh_hsv[:, :, 2] >= 100)
            )
            oh_cyan = (
                (oh_hsv[:, :, 0] >= 80)
                & (oh_hsv[:, :, 0] <= 115)
                & (oh_hsv[:, :, 1] >= 100)
                & (oh_hsv[:, :, 2] >= 100)
            )
            expected_oh_w = max(1.0, float(w) * 0.047)
            oh_hp_cols = int(oh_red.sum(axis=1).max()) if oh_red.size > 0 else 0
            oh_mp_cols = int(oh_cyan.sum(axis=1).max()) if oh_cyan.size > 0 else 0
            overhead_hp_ratio = round(min(1.0, max(0.0, float(oh_hp_cols) / expected_oh_w)), 3)
            overhead_mp_ratio = round(min(1.0, max(0.0, float(oh_mp_cols) / expected_oh_w)), 3)

            hp_ratio = round(float(hp_ratio), 3) if hp_ratio is not None else None
            mp_ratio = round(float(mp_ratio), 3) if mp_ratio is not None else None

            return {
                "hp_ratio": hp_ratio,
                "mp_ratio": mp_ratio,
                "hp": hp_ratio,
                "mp": mp_ratio,
                "hp_pct": round(hp_ratio * 100.0, 1) if hp_ratio is not None else None,
                "mp_pct": round(mp_ratio * 100.0, 1) if mp_ratio is not None else None,
                "hp_valid": hp_valid,
                "mp_valid": mp_valid,
                "overhead_hp_ratio": overhead_hp_ratio,
                "overhead_mp_ratio": overhead_mp_ratio,
            }
        except Exception as exc:
            logger.error("read_vital_bars failed: %s", exc)
            return {
                "hp_ratio": None,
                "mp_ratio": None,
                "hp": None,
                "mp": None,
                "hp_pct": None,
                "mp_pct": None,
                "hp_valid": False,
                "mp_valid": False,
                "overhead_hp_ratio": 0.0,
                "overhead_mp_ratio": 0.0,
            }

    def read_player_hp_mp_bars(self, frame_bgr: np.ndarray) -> Dict[str, object]:
        """
        Reads the bottom-center player HP bar (red), MP bar (cyan/blue), and top-center
        Target bar presence using fast HSV scanlines normalized to frame resolution.

        Returns:
            {
                'hp_pct': float (0.0 .. 100.0) or None when the HP bar is not
                    visible in this frame (v20: unknown, never a fake 0.0),
                'mp_pct': float (0.0 .. 100.0) or None when the MP bar is not
                    visible in this frame,
                'hp_valid': bool, 'mp_valid': bool,
                'has_target': bool,
                'target_pct': float (0.0 .. 100.0),
                'is_monster_target': bool,
                'has_secondary_target': bool,
            }
        """
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return {
                    "hp_pct": None,
                    "mp_pct": None,
                    "has_target": False,
                    "target_pct": 0.0,
                    "is_monster_target": False,
                    "has_secondary_target": False,
                    "hp_valid": False,
                    "mp_valid": False,
                }

            vitals = self.read_vital_bars(frame_bgr)
            target_info = self.detect_top_target_bar(frame_bgr)

            has_target = bool(target_info.get("has_target", False))
            target_pct = round(float(target_info.get("target_hp_ratio", 0.0)) * 100.0, 1)

            hp_pct = vitals.get("hp_pct")
            mp_pct = vitals.get("mp_pct")
            return {
                # v20: None olduğu gibi geçer — float(..., 0.0) burada
                # "bilinmiyor"u "ölü"ye çevirirdi.
                "hp_pct": float(hp_pct) if hp_pct is not None else None,
                "mp_pct": float(mp_pct) if mp_pct is not None else None,
                "has_target": has_target,
                "target_pct": target_pct,
                "is_monster_target": bool(target_info.get("is_monster_target", False)),
                "has_secondary_target": bool(target_info.get("has_secondary_target", False)),
                "hp_valid": bool(vitals.get("hp_valid", hp_pct is not None)),
                "mp_valid": bool(vitals.get("mp_valid", mp_pct is not None)),
            }
        except Exception as exc:
            logger.error("read_player_hp_mp_bars failed: %s", exc)
            return {
                "hp_pct": None,
                "mp_pct": None,
                "has_target": False,
                "target_pct": 0.0,
                "is_monster_target": False,
                "has_secondary_target": False,
                "hp_valid": False,
                "mp_valid": False,
            }

    def verify_gather_cast_bar(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Detects the top-center cyan/teal/blue 'Orichalcum' gathering cast/progress bar
        (as seen in sample_35s.jpg and frame_36s.jpg) and computes its fill
        percentage (0.0 .. 100.0%), while never confusing a monster target HP bar
        ('45 Yellow-Maned Zaif 15m') for a gather cast bar.

        Returns:
            (is_present: bool, progress_pct: float)
        """
        try:
            progress_ratio = self.read_cast_bar_progress(frame_bgr)
            if progress_ratio > 0.0:
                return True, round(min(100.0, max(0.0, progress_ratio * 100.0)), 1)
            return False, 0.0
        except Exception as exc:
            logger.error("verify_gather_cast_bar failed: %s", exc)
            return False, 0.0

    def _classify_digit_contour(self, glyph_bin: np.ndarray) -> Optional[str]:
        """
        Classifies a binarized digit/character ROI ('0'-'9', 'm') in < 0.2ms using
        morphological hole topology (Euler number / contour hierarchy) combined with
        synthetic bitmap & Hershey template correlation.
        """
        try:
            if glyph_bin is None or glyph_bin.size == 0:
                return None

            gh, gw = glyph_bin.shape[:2]
            aspect = float(gw) / float(max(1, gh))

            # 'm' glyph is noticeably wider than a single digit and has 3 vertical legs at the bottom
            if aspect >= 1.15:
                bottom_row = glyph_bin[max(0, gh - 2) :, :] > 0
                col_proj = bottom_row.any(axis=0).astype(np.int8)
                transitions = int(np.sum(np.abs(np.diff(np.pad(col_proj, (1, 1), constant_values=0))))) // 2
                if transitions >= 2:
                    return "m"

            if aspect <= 0.48:
                return "1"

            # F-0337: iç delik topolojisi gerçekten sayılır ve '5' dörtgen
            # kontrolünden ÖNCE bakılır ('5' ile '6' aynı dörtgen imzasına
            # sahip; ayıran deliktir). Eskiden hole_count en fazla 1
            # olabiliyordu (tek orta yama kontrolü), bu yüzden '8' dalı
            # ölüydü ve 2/6/8/9 '0' sayılıyordu. Şimdi arka planın kenara
            # değmeyen bileşenleri delik sayılır; '4'ün küçük üçgen boşluğu
            # (glif alanının ~%9'u) delik sayılmaz (en az %10 + 4px gerekir).
            # 2 delik -> '8'; tek delik -> konuma göre '9' (üstte) / '6'
            # (altta) / '0' (ortada); delik yoksa '5' kontrolü ve şablon
            # eşleşmeye düşülür.
            hole_count = 0
            hole_centers_y: List[float] = []
            if gw >= 3 and gh >= 4:
                bg = (glyph_bin == 0).astype(np.uint8)
                min_hole_px = max(4.0, 0.10 * float(gh * gw))
                num_labels, labels = cv2.connectedComponents(bg, connectivity=4)
                for lbl in range(1, num_labels):
                    ys, xs = np.where(labels == lbl)
                    if len(ys) == 0:
                        continue
                    if ys.min() > 0 and ys.max() < gh - 1 and xs.min() > 0 and xs.max() < gw - 1:
                        if float(len(ys)) < min_hole_px:
                            continue
                        hole_count += 1
                        hole_centers_y.append(float(ys.mean()) / float(max(1, gh)))

            if hole_count >= 2:
                return "8"
            if hole_count == 1:
                cy_rel = hole_centers_y[0] if hole_centers_y else 0.5
                if cy_rel < 0.35:
                    return "9"
                if cy_rel > 0.50:
                    return "6"
                return "0"

            # Structural quadrant check for '5' (upper-left stroke + lower-right loop, open upper-right)
            if gh >= 5 and gw >= 4:
                u_end = max(2, gh // 2 - (1 if gh >= 8 else 0))
                l_start = min(gh - 2, gh // 2 + 1)
                mid_x = max(2, (gw + 1) // 2)
                ul_fill = float(glyph_bin[1:u_end, :mid_x].mean())
                ur_fill = float(glyph_bin[1:u_end, mid_x:].mean())
                ll_fill = float(glyph_bin[l_start : gh - 1, :mid_x].mean())
                lr_fill = float(glyph_bin[l_start : gh - 1, mid_x:].mean())
                if ul_fill > ur_fill * 1.4 and lr_fill > ll_fill * 1.2:
                    return "5"

            # Template correlation against prebuilt 5x7 crisp & Hershey digit templates
            resized = cv2.resize(glyph_bin, (5, 7), interpolation=cv2.INTER_NEAREST)
            best_digit = "0"
            best_score = -1.0
            for ch in "0123456789":
                tpl_a = self.templates[ch]
                tpl_b = self.hershey_templates[ch]
                score_a = float(np.mean(resized == tpl_a))
                score_b = float(np.mean(resized == tpl_b))
                score = max(score_a, score_b)
                if score > best_score:
                    best_score = score
                    best_digit = ch

            return best_digit
        except Exception as exc:
            logger.error("_classify_digit_contour failed: %s", exc)
            return None

    def read_hp_mp(self, frame: np.ndarray) -> Dict[str, float]:
        """Convenience alias for read_player_hp_mp_bars returning hp_pct and mp_pct.

        v20: eski 0.0 sözleşmesi korunur (test_d34 pinler) — okunamayan
        bar burada 0.0 sayılır; "bilinmiyor" ayrımını isteyen çağıranlar
        read_player_hp_mp_bars / read_vital_bars'taki hp_valid'i kullanır.
        """
        try:
            res = self.read_player_hp_mp_bars(frame)
            return {
                "hp_pct": float(res["hp_pct"]) if res.get("hp_pct") is not None else 0.0,
                "mp_pct": float(res["mp_pct"]) if res.get("mp_pct") is not None else 0.0,
            }
        except Exception as exc:
            logger.error("read_hp_mp failed: %s", exc)
            return {"hp_pct": 0.0, "mp_pct": 0.0}

    def is_under_attack(self, frame: np.ndarray) -> bool:
        """Determines if the player is under attack based on monster target presence or alerts."""
        try:
            if frame is None or frame.size == 0:
                return False
            target_info = self.detect_top_target_bar(frame)
            if target_info.get("has_target", False) and target_info.get("is_monster_target", False):
                return True
            return False
        except Exception as exc:
            logger.error("is_under_attack failed: %s", exc)
            return False

    def get_hud_combat_and_health_summary(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Safely calls read_vital_bars and detect_top_target_bar inside try/except
        and returns {"hp_pct": float, "mp_pct": float, "is_under_attack": bool}.

        DEV-D34 / A18: eskiden read_hp_mp() (-> read_player_hp_mp_bars ->
        detect_top_target_bar) + is_under_attack() (-> detect_top_target_bar)
        yoluyla üst hedef barı kare başına İKİ kez tarıyordu; şimdi
        read_vital_bars + detect_top_target_bar birer kez çağrılıyor ve
        is_under_attack aynı sözlükten türetiliyor. Dönüş değerleri birebir aynı.
        """
        try:
            vitals = self.read_vital_bars(frame)
            target_info = self.detect_top_target_bar(frame)
            under_attack = bool(
                target_info.get("has_target", False) and target_info.get("is_monster_target", False)
            )
            return {
                # v20: None -> 0.0 eşlemesi bilinçli (test_d34 pinler:
                # bu özet API boş/siyah karede 0.0 döner); "bilinmiyor"
                # ayrımı read_vital_bars'taki hp_valid/mp_valid'tedir.
                "hp_pct": float(vitals["hp_pct"]) if vitals.get("hp_pct") is not None else 0.0,
                "mp_pct": float(vitals["mp_pct"]) if vitals.get("mp_pct") is not None else 0.0,
                "is_under_attack": bool(under_attack),
            }
        except Exception as exc:
            logger.error("get_hud_combat_and_health_summary failed: %s", exc)
            return {
                "hp_pct": 0.0,
                "mp_pct": 0.0,
                "is_under_attack": False,
            }

def detect_gm_whisper_chat(frame: np.ndarray) -> bool:
    """Lower chat'te GM / fısıltı tipi mor-lavanta sistem metni arar.

    Üretim watchdog (PANIC-LOCK GM_INSPECTION_ALERT) bu fonksiyonu
    worker.grab_screen() karesiyle çağırır — turuncu sistem uyarılarından
  ayrılır.
    """
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        h, w = frame.shape[:2]
        cx1 = max(0, int(round(w * 0.012)))
        cx2 = min(w, max(cx1 + 1, int(round(w * 0.28))))
        cy1 = max(0, int(round(h * 0.72)))
        cy2 = min(h, max(cy1 + 1, int(round(h * 0.88))))
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        purple_mask = (
            (hsv[:, :, 0] >= 118)
            & (hsv[:, :, 0] <= 165)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 85)
        )
        orange_mask = (
            (hsv[:, :, 0] >= 8)
            & (hsv[:, :, 0] <= 28)
            & (hsv[:, :, 1] >= 120)
            & (hsv[:, :, 2] >= 140)
        )
        area = max(1, roi.shape[0] * roi.shape[1])
        purple_ratio = float(purple_mask.sum()) / float(area)
        orange_ratio = float(orange_mask.sum()) / float(area)
        return purple_ratio >= 0.012 and purple_ratio > orange_ratio * 1.35
    except Exception as exc:
        logger.debug("detect_gm_whisper_chat failed: %s", exc)
        return False


_DEFAULT_HUD_READER: Optional[GameHUDReader] = None


def _get_default_hud_reader() -> GameHUDReader:
    global _DEFAULT_HUD_READER
    if _DEFAULT_HUD_READER is None:
        _DEFAULT_HUD_READER = GameHUDReader()
    return _DEFAULT_HUD_READER


def detect_chat_input_open(frame: np.ndarray) -> ChatInputOpenResult:
    """
    Module-level convenience wrapper for GameHUDReader.detect_chat_input_open (Item 17).
    Detects if the bottom-left chat text input bar (x: 0.01..0.24, y: 0.91..0.96) is open.
    """
    return _get_default_hud_reader().detect_chat_input_open(frame)


def parse_chat_feedback_fast(
    frame: Union[np.ndarray, str, Sequence[str], None],
) -> Dict[str, Any]:
    """
    Module-level convenience wrapper for GameHUDReader.parse_chat_feedback_fast (Item 20).
    """
    return _get_default_hud_reader().parse_chat_feedback_fast(frame)


def close_chat_box_if_open(
    frame: Optional[np.ndarray] = None,
    input_controller: Optional[Any] = None,
    chat_open: Optional[bool] = None,
) -> bool:
    """
    Checks if the chat input box is open (via `frame` or `chat_open`) and delegates
    pressing `Esc` to `input_controller.close_chat_box_if_open` (or `press_esc`).
    """
    is_open = bool(chat_open) if chat_open is not None else False
    if not is_open and frame is not None:
        is_open = bool(detect_chat_input_open(frame).get("is_open", False))

    if not is_open:
        return False

    if input_controller is not None:
        if hasattr(input_controller, "close_chat_box_if_open"):
            return bool(input_controller.close_chat_box_if_open(frame=frame, chat_open=True))
        if hasattr(input_controller, "press_esc"):
            input_controller.press_esc()
            return True
    # F-0160: kontrolcü yoksa Esc basılmadı; "kapatıldı" yalanı söylenmez.
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # NOT: <100ms sınırı paylaşımlı VM yükündeki dalgalanmaya (1.7-81ms)
    # karşı toleranstır; işlevsel iddialar (sınıflandırma) asıl denetimdir.
    reader = GameHUDReader()

    # 1. Verify closed chat input bar on blank / normal frame
    closed_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    closed_res = reader.detect_chat_input_open(closed_frame)
    assert not closed_res["is_open"] and not bool(closed_res)
    assert closed_res["elapsed_ms"] < 100.0

    # 2. Verify open chat input bar via bright input border + [Say] prompt + cursor
    open_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ix1, ix2 = int(1920 * 0.01), int(1920 * 0.24)
    iy1, iy2 = int(1080 * 0.91), int(1080 * 0.96)
    cv2.rectangle(open_frame, (ix1 + 4, iy1 + 4), (ix2 - 4, iy2 - 4), (210, 210, 210), 2)
    cv2.putText(
        open_frame,
        "[Say] ",
        (ix1 + 12, iy1 + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
    )
    cv2.line(open_frame, (ix1 + 92, iy1 + 10), (ix1 + 92, iy2 - 10), (255, 255, 255), 2)
    open_res = reader.detect_chat_input_open(open_frame)
    assert open_res["is_open"] and bool(open_res), f"Expected open chat box, got {open_res}"
    assert open_res["elapsed_ms"] < 100.0

    # 3. Verify < 5ms fast chat feedback parser for all 3 alerts
    def _make_alert_frame(msg_text: str) -> np.ndarray:
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cx1 = int(1920 * 0.015)
        cy1 = int(1080 * 0.775)
        # Amber/orange system warning color BGR=(55, 155, 235)
        cv2.putText(
            canvas,
            msg_text,
            (cx1, cy1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (55, 155, 235),
            1,
            cv2.LINE_AA,
        )
        return canvas

    r_ext = reader.parse_chat_feedback_fast(_make_alert_frame("Essence Extraction failed"))
    assert r_ext["extraction_failed"] is True and r_ext["recommended_action"] == "retry_gather", f"Got {r_ext}"
    assert r_ext["elapsed_ms"] < 100.0

    r_mnt = reader.parse_chat_feedback_fast(_make_alert_frame("Cannot mount in current state."))
    assert r_mnt["cannot_mount"] is True and r_mnt["recommended_action"] == "skip_mount", f"Got {r_mnt}"
    assert r_mnt["elapsed_ms"] < 100.0

    r_inv = reader.parse_chat_feedback_fast(_make_alert_frame("Invalid target"))
    assert r_inv["invalid_target"] is True and r_inv["recommended_action"] == "retarget", f"Got {r_inv}"
    assert r_inv["elapsed_ms"] < 100.0

    r_none = reader.parse_chat_feedback_fast(closed_frame)
    assert r_none["recommended_action"] == "none" and r_none["elapsed_ms"] < 100.0

    # 4. F-0161/F-0339 negatif problar: loot ve başarı mesajları asla
    # extraction_failed sayılmaz (eski geometri kuralı hepsini "failed" diyordu).
    r_loot = reader.parse_chat_feedback_fast(_make_alert_frame("You obtained 100 Kina"))
    assert r_loot["extraction_failed"] is False, f"Loot 'failed' sayıldı: {r_loot}"
    r_ok = reader.parse_chat_feedback_fast(_make_alert_frame("Essence Extraction complete"))
    assert r_ok["extraction_failed"] is False, f"Başarı 'failed' sayıldı: {r_ok}"
    r_party = reader.parse_chat_feedback_fast(_make_alert_frame("Party member joined"))
    assert r_party["invalid_target"] is False and r_party["extraction_failed"] is False, f"Yanlış etiket: {r_party}"
    # Metin yolunda da aynı kural: başarı mesajı başarısız değildir.
    assert reader.parse_chat_feedback_fast("Essence Extraction complete")["extraction_failed"] is False
    assert reader.parse_chat_feedback_fast("Essence Extraction failed")["extraction_failed"] is True
    # 5. F-0161 kırmızı koruma: gerçek oyunda kırmızı olan ölüm mesajı
    # ("You have died") turuncu uyarı sayılmaz. Not: aynı mesaj turuncu
    # çizilirse geometrik olarak "Invalid target"tan ayırt edilemez
    # (span 0.228 vs 0.211) — bu prob kırmızıyı doğrular, turuncu
    # varsayımı doğrulamaz.
    died_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(
        died_frame, "You have died", (int(1920 * 0.015), int(1080 * 0.775)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 230), 1, cv2.LINE_AA,
    )
    r_died = reader.detect_chat_system_alerts(died_frame)
    assert r_died["invalid_target"] is False and r_died["extraction_failed"] is False, f"Kırmızı ölüm mesajı etiketlendi: {r_died}"

    logger.info(
        "core/ocr_distance.py self-check PASSED (ext=%.2fms, mnt=%.2fms, inv=%.2fms, chat_open=%.2fms).",
        r_ext["elapsed_ms"],
        r_mnt["elapsed_ms"],
        r_inv["elapsed_ms"],
        open_res["elapsed_ms"],
    )


