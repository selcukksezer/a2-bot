# -*- coding: utf-8 -*-
"""
High-Speed Computer Vision & Template Matching Engine (core/vision.py)

Provides:
- Dual Gather Prompt detection (`detect_gather_prompt` / `check_gather_prompt` / `find_gather_prompt`):
  - Multi-scale template & pill-region detection for the 'F Gather' prompt loading from both
    `templates/` (`f_gather_template.png`, `f_gather_pill.png`, `f_gather.png`) and
    `assets/f_gather_template.png` (including the `x=42..118` dark pill region resilient to
    rock/grass backgrounds).
  - Floating `[E]` / `[F]` resource box detection (`detect_floating_resource_icon`) across the
    mid-screen 3D viewport (`x: 0.25..0.75, y: 0.25..0.70`) and within close interaction range
    (`x: 0.42..0.69, y: 0.38..0.66`), storing `self.last_detected_gather_key` (`"f"`, `"e"`, or `""` when unknown).
- 3D White Winged Gather Crest detection (`detect_3d_gather_crests`) for gatherable plants
  such as `Aria` and `Mela` that do not appear on the minimap (`aria_3d_crest_far.jpg`,
  `aria_f_gather_not_on_minimap.jpg`, `mela_f_gather_not_on_minimap.jpg`).
- Minimap Filter Settings popup & checkbox inspection (`detect_minimap_filter_settings`)
  verifying `Quest`, `Monster`, `Gatherable Material`, and `Functional NPC` checkmarks.
- Open World Map (`M`) verification (`is_map_open`) resilient across all zones (`Altgard`,
  `Pilgrim's Ridge`, `Western Altamia Highland`) and resolutions (`720p`, `1080p`, `2000x1125`, `1440p`).
- Blocking in-game window detection (`is_game_window_blocking`) for Inventory/Cube,
  Character, Quest, and right/center menu panels (`frame_03s.jpg`).
- Overlay collision detection (`is_overlay_covering_point`).
- Clean 80x80 grayscale map patch extraction (`extract_center_map_patch`)
  with player arrow Telea inpainting.
"""

from __future__ import annotations

import base64
import logging
import os
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from core.capture import REF_HEIGHT, REF_WIDTH, ScreenCapture
# P1-7 (K-20): dünya-harita metre/piksel ölçeğinin tek kaynağı.
from core.corridor_navigator import WORLD_MAP_METERS_PER_PX
from version import __version__
from core.exemplar_threshold import is_negative_exemplar_name, load_exemplar_match_threshold

logger = logging.getLogger(__name__)

# Şablon decode önbelleği: (mutlak yol, mtime, boyut, imread flags) ->
# decode edilmiş ndarray. VisionEngine her yapımda aynı şablonları
# diskten okuyup decode ediyordu (2 instance x 6 şablon = 12 decode);
# dosya değişmediyse önbellekten dönülür. Piramitler instance başına
# kalır — yalnızca ham decode önbelleğe alınır. Çağırana ndarray'in
# kopyası verilir, böylece çağıran diziyi değiştirse bile önbellekteki
# asıl bozulmaz. Boyut sınırı: _coerce_frame gibi büyük kareler de bu
# yoldan geçtiği için FIFO ile 32 girdide tutulur.
_TEMPLATE_DECODE_CACHE: Dict[Tuple[str, float, int, int], np.ndarray] = {}
_TEMPLATE_DECODE_CACHE_MAX_ENTRIES = 32

# --- Sweep G (2026-11-09): tek yedek ROI + tek exemplar eşiği -----------------
# HP barı için dosya içinde iki farklı sabit yedek ROI vardı (is_map_open:
# x 0.36-0.47 / y 0.87-0.92; read_player_hp_mp_bars: 0.345-0.485 /
# 0.880-0.915). Tek doğruluk kaynağı: kalibre edilmiş player_hp_bar
# kaydıyla birebir uyuşan (0.345, 0.880, 0.485, 0.915) dikdörtgeni.
# MP ve hedef barı yedekleri de aynı kurala bağlandı: MP
# (0.515, 0.880, 0.655, 0.915), hedef bar (0.350, 0.030, 0.650, 0.075).
# Kalibre kaydı varsa o kullanılır; bu sabitler yalnızca kalibre yokken
# devreye girer.
FALLBACK_PLAYER_HP_BAR_ROI: Tuple[float, float, float, float] = (0.345, 0.880, 0.485, 0.915)
FALLBACK_PLAYER_MP_BAR_ROI: Tuple[float, float, float, float] = (0.515, 0.880, 0.655, 0.915)
FALLBACK_TARGET_HP_BAR_ROI: Tuple[float, float, float, float] = (0.350, 0.030, 0.650, 0.075)


def _load_exemplar_match_threshold(default: float = 0.68) -> float:
    """Tek exemplar-benzerlik eşiği (P3-3: `core.exemplar_threshold` yardımcısına delege).

    Eskiden 0.65 / 0.68 / 0.70 / 0.72 / 0.75 gibi dağınık değerler
    kullanılıyordu; hiçbiri ayarda değildi.
    """
    return load_exemplar_match_threshold(default)


EXEMPLAR_MATCH_THRESHOLD: float = _load_exemplar_match_threshold()


DEFAULT_TEMPLATE_PATHS: Sequence[str] = (
    "templates/f_gather_template.png",
    "templates/f_gather_pill.png",
    "templates/f_gather.png",
    "templates/f_gather_aria.png",
    "templates/legacy/f_gather_video.png",
    "templates/legacy/f_gather_pill_video.png",
    "assets/f_gather_template.png",
)

E_GATHER_ICON_TEMPLATE_PATHS: Sequence[str] = (
    "templates/e_gather_icon.png",
    "assets/e_gather_icon.png",
)

GATHER_CREST_TEMPLATE_PATHS: Sequence[str] = (
    "templates/world_gather_crest.png",
    "templates/world_gather_crest_near.png",
    "templates/gather_crest.png",
    "assets/gather_crest.png",
)

MAP_HEADER_TEMPLATE_PATH: str = "templates/map_header.png"
MAP_HEADER_LEGACY_TEMPLATE_PATH: str = "templates/legacy/map_header_video.png"
PLAYER_ARROW_TEMPLATE_PATH: str = "templates/player_arrow.png"
PLAYER_ARROW_LEGACY_TEMPLATE_PATH: str = "templates/legacy/player_arrow_video.png"

# Embedded 36x37 grayscale reference template for the floating [E] gatherable plant box
# (extracted from reference_screenshots/pilgrims_ridge_1.jpg at y=487..523, x=1312..1349)
_EMBEDDED_E_GATHER_BOX_B64: str = (
    "eNp9VFGV7TAIrIVaqIVYwAIWsIAFLMRCLMRCLMRCLXRnoN19X4/b0+a2kwEmQGv/MZHflaqa"
    "5q0e9bRvoUBqp0VetDFHLfpcc+DfsCbWI3A5LoN5eBlWg/AxXAByaRfcXqBXUTdlAHTbk3OE"
    "iA8/G2jtEBXJ8BKEW6fnGAF3I057PDRBhIG9FuVw9nR36q2tHfwiTeEdP6IcuYw+ozULO/VZ"
    "K45SRbwESlgAtMgUdui+zuMsjJCqFZv1e43dwR4Kpt79Ejcxx64ghFQe+9lPb2Q6r4BKp+85"
    "fUCnHpqiiPd172c0Nac+3Kpj77X2AI5BXVTDPSpwng4xTYDa9z2HXTCQQSwDLyhDM61L4HLd"
    "N3DbgLgyP6oFXnOCct8ADa+ZKrA+nFTOjC1z59bY9wbZoggtTzRBFZO2DEGAAubeTumbXIkC"
    "ExVngSiZ1Cf8PfeU4vlAkkzqayLg1ZnfgwRb5pcHEGTKwIGdcPRsluJGVC8IOkbFlAGwEtfe"
    "qI01+1560j9ASKxAFM1QFislv4FakUK11AnuuhVJSoSoqefA1mwooTspMdWBWjv1vmmTBSpv"
    "TCwHzSNCqQ4Uw2B/DFZ2CiqWintJENkX2VXVKqZvnffIysQ7UKD12E3vsWYJsRk9SnG+CcZv"
    "+jU/TilTU/2OJffBq70DIqX+KgXfX3fVjSY1Hr52qeDcssbtnR8M29/UC0OIJ9NfSjl3ys3b"
    "8ww+pwrBhfudSInSF8TABwfPn1ltqAc06Bg9TSPn02f/LPkPow2n+AM3tCl8"
)

# Embedded 29x29 grayscale reference template for the 3D white winged gather crest
# (extracted from reference_screenshots/aria_3d_crest_far.jpg at y=471..500, x=779..808)
_EMBEDDED_3D_GATHER_CREST_B64: str = (
    "eNo9Ugu1xSAMq4VawAIWZgELWMACFrBQC1iohVqohb5QuA92NrYsafrhglUL83nSjkZMeLur"
    "Yr+3wrS7Dibg96Yfff2c6bzuUmR/RB9zY16xWZTKnEtBktqWyFqd+i7dzUYMQtTSbB5mhmeq"
    "KsvVxG9IqrIVge4iseXbVQvdv4vFA2FgbtXtW+oo12yPRLBbHXObxPy27ae8jwh4w6Q3Uwkb"
    "qrEyV1rl6FJFrLFMZ4SMHt6oYE8w8dy22jQXmPCPwIefTB6i062wuHecUIcRJidcXvmlmwnx"
    "Fx7muHlNMyiau218mwihkVBAOY0CMvxtBhtcd5wFtJ3iG4oG0BTNQxO5eWLIh+mzQzQfsPfq"
    "tlI4wQDo8h1rdwSI1lHeAFvaKfTaf9ENpiRovmumVPjNC3X3ZPZQF7rFfCAyjwuiHPErP/9G"
    "Sw540oQsGlwSK6+DCllD7AnQ6o91iWT/oLtnIj/m8QFQa1YWaabaP7jQOX+g3w79Ju1YhyPI"
    "nWrod428WcvT8jMOmJhGt7PMf682PAY="
)

# Maximum raw crest candidates kept before NMS + alignment proof (F-1684:
# unbounded candidates made many-match frames take seconds).
MAX_CREST_CANDIDATES: int = 200

# Map-header template scales for `VisionEngine.is_map_open`. The legacy
# 720p English header must cover a native 2560x1440 UI (about 2.0x-2.67x).
MAP_HEADER_SCALES: Tuple[float, ...] = (
    0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.40, 1.50,
    1.5625, 1.65, 1.80, 2.0, 2.25, 2.50, 2.67,
)

# Shared 5x5 erosion kernel for `verify_crest_alignment_proof` (allocated
# once; previously rebuilt from `np.ones` on every crest candidate).
_ERODE_KERNEL_5X5: np.ndarray = np.ones((5, 5), dtype=np.uint8)

# ==============================================================================
# Player Center Self-Exclusion Zone (Normalized X: 0.43..0.57, Y: 0.38..0.64)
# Eliminates false 3D monster and resource detections on player model, cape, mount
# ==============================================================================
PLAYER_CENTER_EXCLUSION_ZONE: Tuple[float, float, float, float] = (0.43, 0.38, 0.57, 0.64)
PLAYER_SELF_EXCLUSION_ZONE: Tuple[float, float, float, float] = PLAYER_CENTER_EXCLUSION_ZONE


def is_in_player_exclusion_zone(
    x: Union[int, float],
    y: Union[int, float],
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    exclusion_zone: Tuple[float, float, float, float] = PLAYER_CENTER_EXCLUSION_ZONE,
) -> bool:
    """
    Determines whether a point (x, y) or bounding box falls within the player center
    self-exclusion zone (normalized X in [0.43, 0.57], Y in [0.38, 0.64]).
    Used to eliminate false 3D monster and resource detections originating from
    the player's own character model, cape, or mount.
    """
    try:
        fw = float(frame_w) if (frame_w is not None and frame_w > 0) else float(REF_WIDTH)
        fh = float(frame_h) if (frame_h is not None and frame_h > 0) else float(REF_HEIGHT)

        zx1, zy1, zx2, zy2 = exclusion_zone

        if bbox is not None and len(bbox) >= 4:
            bx, by, bw, bh = bbox[:4]
            cx = float(bx) + float(bw) / 2.0
            cy = float(by) + float(bh) / 2.0
        else:
            cx, cy = float(x), float(y)

        # Handle normalized vs absolute pixel coordinates
        if (0.0 <= cx <= 1.0) and (0.0 <= cy <= 1.0):
            nx = cx
            ny = cy
        else:
            nx = cx / fw
            ny = cy / fh

        if zx1 <= nx <= zx2 and zy1 <= ny <= zy2:
            return True

        # Check significant bounding box overlap (> 40% of bbox)
        if bbox is not None and len(bbox) >= 4:
            bx, by, bw, bh = bbox[:4]
            if bw > 0 and bh > 0:
                pix_zx1 = zx1 * fw
                pix_zx2 = zx2 * fw
                pix_zy1 = zy1 * fh
                pix_zy2 = zy2 * fh
                ox = max(0.0, min(float(bx + bw), pix_zx2) - max(float(bx), pix_zx1))
                oy = max(0.0, min(float(by + bh), pix_zy2) - max(float(by), pix_zy1))
                if (ox * oy) / float(bw * bh) >= 0.40:
                    return True

        return False
    except Exception:
        return False


def is_in_player_self_exclusion_zone(
    x: Union[int, float],
    y: Union[int, float],
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    exclusion_zone: Tuple[float, float, float, float] = PLAYER_CENTER_EXCLUSION_ZONE,
) -> bool:
    """Alias for is_in_player_exclusion_zone."""
    return is_in_player_exclusion_zone(
        x=x, y=y, frame_w=frame_w, frame_h=frame_h, bbox=bbox, exclusion_zone=exclusion_zone
    )


def detect_inventory_panel(frame_bgr: Optional[np.ndarray]) -> bool:
    """
    Çanta + kuşanlı eşya ızgarası açık mı?

    Sağ-üst minimap görünür kalsa da (``is_game_window_blocking`` bunu menü sanmaz)
    veya panel çok karanlık olsa da (ortalama parlaklık < 22 gece sanılır) envanter
    ikonları dünya materyali değildir. Koyu pencerede sütun hizalı ikon ızgarası aranır.
    """
    try:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return False
        if frame_bgr.ndim != 3 or frame_bgr.size == 0 or frame_bgr.shape[2] < 3:
            return False
        img = frame_bgr[:, :, :3]
        frame_h, frame_w = img.shape[:2]
        if frame_h < 200 or frame_w < 280:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        windows = [(0, 0, frame_w, frame_h)]
        # Çanta genelde ortaya oturur; tam kare parlak dünya yüzünden eşiği kaçırır.
        for wf, hf, ax, ay in (
            (0.42, 0.94, 0.28, 0.02),
            (0.50, 0.90, 0.22, 0.04),
            (0.55, 0.85, 0.18, 0.06),
        ):
            ww, hh = int(frame_w * wf), int(frame_h * hf)
            x0 = min(max(0, int(frame_w * ax)), max(0, frame_w - ww))
            y0 = min(max(0, int(frame_h * ay)), max(0, frame_h - hh))
            if ww >= 200 and hh >= 220:
                windows.append((x0, y0, ww, hh))
        for wf, hf in ((0.62, 0.78), (0.48, 0.70), (0.85, 0.92)):
            ww, hh = int(frame_w * wf), int(frame_h * hf)
            if ww < 200 or hh < 220:
                continue
            step_x = max(1, ww // 3)
            step_y = max(1, hh // 3)
            y0 = 0
            while y0 <= max(0, frame_h - hh):
                x0 = 0
                while x0 <= max(0, frame_w - ww):
                    windows.append((x0, y0, ww, hh))
                    if x0 == frame_w - ww:
                        break
                    x0 = min(frame_w - ww, x0 + step_x)
                if y0 == frame_h - hh:
                    break
                y0 = min(frame_h - hh, y0 + step_y)
            if len(windows) > 28:
                break

        for x0, y0, ww, hh in windows[:28]:
            roi_g = gray[y0 : y0 + hh, x0 : x0 + ww]
            if roi_g.size == 0:
                continue
            mean_g = float(roi_g.mean())
            dark_frac = float(np.mean(roi_g < 48))
            if mean_g > 40.0 or dark_frac < 0.74:
                continue
            roi_h = hsv[y0 : y0 + hh, x0 : x0 + ww]
            icon = cv2.inRange(roi_h, (0, 30, 28), (180, 255, 255))
            contours, _ = cv2.findContours(icon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xs: List[float] = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 8 or ch < 8 or cw > int(ww * 0.16) or ch > int(hh * 0.12):
                    continue
                if not (0.35 <= (float(cw) / float(max(1, ch))) <= 2.6):
                    continue
                xs.append(x + cw / 2.0)
            if len(xs) < 30:
                continue
            cols: List[List[float]] = []
            tol = max(8.0, ww * 0.02)
            for x in sorted(xs):
                if not cols or abs(x - float(np.mean(cols[-1]))) > tol:
                    cols.append([x])
                else:
                    cols[-1].append(x)
            rich4 = sum(1 for col in cols if len(col) >= 4)
            rich6 = sum(1 for col in cols if len(col) >= 6)
            if rich4 >= 6 and rich6 >= 4:
                return True
        return False
    except Exception as exc:
        logger.debug("detect_inventory_panel hatası: %s", exc)
        return False


class VisionEngine:
    """
    Production-grade computer vision engine for AION 2 gathering & map navigation.
    """

    def __init__(
        self,
        template_path: str = "assets/f_gather_template.png",
        monitor_idx: int = 1,
    ) -> None:
        self.capture: ScreenCapture = ScreenCapture(
            monitor_idx=monitor_idx,
            ref_width=REF_WIDTH,
            ref_height=REF_HEIGHT,
        )
        self.monitor: Dict[str, int] = self.capture.monitor
        self.screen_w: int = self.capture.screen_w
        self.screen_h: int = self.capture.screen_h
        # P1-7 (K-20): dünya-harita metre/piksel ölçeği tek kaynaktan
        # (WORLD_MAP_METERS_PER_PX, 0.28). Eski 0.25 değeri kaldırıldı;
        # route_manager artık bu özniteliği okumuyor (kendi ölçeğini
        # aynı sabitten alır) — öznitelik geriye dönük uyumluluk için
        # korunur. Değer canlı ölçümle kalibre edilmedi.
        self.meters_per_pixel: float = WORLD_MAP_METERS_PER_PX

        self.template_path: str = template_path
        self.gather_template: Optional[np.ndarray] = None
        self.gather_pill_template: Optional[np.ndarray] = None
        self.gather_pill_tight: Optional[np.ndarray] = None
        self.e_gather_template: Optional[np.ndarray] = None
        self.e_badge_template: Optional[np.ndarray] = None
        self.gather_crest_template: Optional[np.ndarray] = None
        self.map_header_template: Optional[np.ndarray] = None
        self.map_header_legacy_template: Optional[np.ndarray] = None
        self.player_arrow_template: Optional[np.ndarray] = None
        self.player_arrow_legacy_template: Optional[np.ndarray] = None
        self._template_variants: List[Dict[str, Any]] = []
        self._e_icon_scales: List[Tuple[float, np.ndarray]] = []
        self._crest_scales: List[Tuple[float, np.ndarray]] = []
        # Lazily built per-template resized map-header pyramids for
        # `is_map_open` (entries keep the source template alive so the
        # identity check below cannot match a recycled id()).
        self._map_header_scale_cache: List[Tuple[np.ndarray, List[Tuple[float, np.ndarray]]]] = []

        # Tracks which interaction key ('f' or 'e') was most recently detected.
        # Sweep G: ilk değer "" (bilinmiyor) — eskiden "f" idi ve hiçbir
        # tespit yapılamasa bile başarısız çağrılar bu bayat değeri taze
        # tespitmiş gibi döndürüyordu. Başarısızlık artık "" döner.
        self.last_detected_gather_key: str = ""

        # Default ROI for 'F Gather' prompt in normalized coordinates
        # Covers the character interaction zone while excluding right-edge quest tracker / bottom-right HUD
        self.prompt_norm_roi: Tuple[float, float, float, float] = (0.35, 0.35, 0.48, 0.48)

        # Close interaction range for floating [E]/[F] box & 'F Gather' prompt
        self.close_interaction_norm_roi: Tuple[float, float, float, float] = (0.42, 0.38, 0.69, 0.66)
        # Mid-screen 3D viewport ROI for floating [E]/[F] resource icon detection
        self.floating_icon_norm_roi: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.70)

        # Player center self-exclusion zone in 3D viewport (x: 0.43..0.57, y: 0.38..0.64 normalized)
        # Eliminates false 3D monster or resource detections originating from player model/cape/mount
        self.player_center_exclusion_zone: Tuple[float, float, float, float] = PLAYER_CENTER_EXCLUSION_ZONE

        self.load_template()

    # ------------------------------------------------------------------
    # Template Loading & Multi-Scale Pyramid Preparation
    # ------------------------------------------------------------------
    def _resolve_asset_path(self, rel_or_abs: str) -> Optional[str]:
        """Resolves a relative template/asset path against CWD and project root."""
        if not rel_or_abs:
            return None
        if os.path.exists(rel_or_abs):
            return rel_or_abs
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(base_dir, rel_or_abs)
        if os.path.exists(alt):
            return alt
        return None

    @staticmethod
    def _safe_imread(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
        """Unicode-safe image reader for Windows paths containing Turkish characters.

        Decode önbellekli: aynı dosya (yol, mtime, boyut, flags) ikinci kez
        istendiğinde diskten okunup decode edilmez, önbellekten kopya döner.
        """
        try:
            if not path or not os.path.isfile(path):
                return None
            cache_key: Optional[Tuple[str, float, int, int]] = None
            try:
                st = os.stat(path)
                cache_key = (
                    os.path.abspath(path),
                    float(st.st_mtime),
                    int(st.st_size),
                    int(flags),
                )
                cached = _TEMPLATE_DECODE_CACHE.get(cache_key)
                if cached is not None:
                    return cached.copy()
            except OSError:
                cache_key = None
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size == 0:
                return None
            decoded = cv2.imdecode(raw, flags)
            if decoded is None:
                return cv2.imread(path, flags)
            if cache_key is not None:
                if len(_TEMPLATE_DECODE_CACHE) >= _TEMPLATE_DECODE_CACHE_MAX_ENTRIES:
                    _TEMPLATE_DECODE_CACHE.pop(next(iter(_TEMPLATE_DECODE_CACHE)))
                _TEMPLATE_DECODE_CACHE[cache_key] = decoded
            return decoded.copy()
        except Exception:
            return cv2.imread(path, flags)

    def load_template(self) -> bool:
        """
        Loads all available 'F Gather' templates from `templates/` and `assets/`,
        builds both full-template and dark-pill (`x=42..118`) multi-scale pyramids,
        loads floating `[E]` resource icon and 3D white winged gather crest templates,
        and loads auxiliary map templates (`map_header.png`, `player_arrow.png`).
        """
        self._template_variants.clear()
        self._e_icon_scales.clear()
        self._crest_scales.clear()

        candidate_paths: List[str] = []
        for path in DEFAULT_TEMPLATE_PATHS:
            if path not in candidate_paths:
                candidate_paths.append(path)
        if self.template_path and self.template_path not in candidate_paths:
            candidate_paths.append(self.template_path)

        loaded_any = False
        seen_hashes: set[bytes] = set()

        for rel_or_abs in candidate_paths:
            resolved = self._resolve_asset_path(rel_or_abs)
            if resolved is None:
                continue

            try:
                img_gray = self._safe_imread(resolved, cv2.IMREAD_GRAYSCALE)
                if img_gray is None or img_gray.size == 0:
                    continue

                th, tw = img_gray.shape[:2]
                fname = os.path.basename(resolved).lower()
                is_pill_file = "pill" in fname

                if self.gather_template is None and not is_pill_file:
                    self.gather_template = img_gray
                if is_pill_file and self.gather_pill_template is None:
                    self.gather_pill_template = img_gray

                sig = img_gray.tobytes()
                if sig in seen_hashes:
                    loaded_any = True
                    continue
                seen_hashes.add(sig)

                if is_pill_file:
                    if tw >= 70:
                        pill_ox, pill_oy, pill_fw, pill_fh = 42, 4, 130, 36
                    else:
                        pill_ox, pill_oy, pill_fw, pill_fh = 27, 3, 86, 23
                else:
                    pill_ox, pill_oy, pill_fw, pill_fh = 0, 0, tw, th

                # Register template (supporting both 130x36 HD pill and 86x23 legacy pill)
                self._add_template_variant(
                    name=f"full:{os.path.basename(resolved)}",
                    template_gray=img_gray,
                    offset_x=pill_ox,
                    offset_y=pill_oy,
                    full_w=pill_fw,
                    full_h=pill_fh,
                    is_pill=is_pill_file,
                )

                # For full templates (130x36 HD, 120x35 asset, or 86x23 legacy),
                # extract the right-hand 'F Gather' dark pill region
                if not is_pill_file and tw >= 80 and th >= 20:
                    if tw >= 125:
                        x1, x2 = 42, min(tw, 128)
                        y1, y2 = 4, min(th, 32)
                    elif tw >= 115:
                        x1, x2 = 42, min(tw, 118)
                        y1, y2 = 5, min(th, 30)
                    else:
                        x1 = int(round(tw * (42.0 / 120.0)))
                        x2 = int(round(tw * (118.0 / 120.0)))
                        y1 = int(round(th * (5.0 / 35.0)))
                        y2 = int(round(th * (30.0 / 35.0)))

                    pill_slice = img_gray[:, x1:x2].copy()
                    if self.gather_pill_template is None:
                        self.gather_pill_template = pill_slice

                    if y2 - y1 >= 12 and x2 - x1 >= 28:
                        pill_tight = img_gray[y1:y2, x1:x2].copy()
                        self.gather_pill_tight = pill_tight
                        tight_sig = pill_tight.tobytes()
                        if tight_sig not in seen_hashes:
                            seen_hashes.add(tight_sig)
                            self._add_template_variant(
                                name=f"pill_tight:{os.path.basename(resolved)}",
                                template_gray=pill_tight,
                                offset_x=x1,
                                offset_y=y1,
                                full_w=tw,
                                full_h=th,
                                is_pill=True,
                            )

                loaded_any = True
                logger.info(
                    "Loaded gather template '%s' (shape=%dx%d)",
                    resolved,
                    tw,
                    th,
                )
            except Exception as exc:
                logger.error("Failed to load template '%s': %s", resolved, exc)

        # Load optional map_header and player_arrow templates (both HD and legacy video)
        try:
            mh_path = self._resolve_asset_path(MAP_HEADER_TEMPLATE_PATH)
            if mh_path is not None:
                self.map_header_template = self._safe_imread(mh_path, cv2.IMREAD_GRAYSCALE)
            mh_leg_path = self._resolve_asset_path(MAP_HEADER_LEGACY_TEMPLATE_PATH)
            if mh_leg_path is not None:
                self.map_header_legacy_template = self._safe_imread(mh_leg_path, cv2.IMREAD_GRAYSCALE)
            pa_path = self._resolve_asset_path(PLAYER_ARROW_TEMPLATE_PATH)
            if pa_path is not None:
                self.player_arrow_template = self._safe_imread(pa_path, cv2.IMREAD_GRAYSCALE)
            pa_leg_path = self._resolve_asset_path(PLAYER_ARROW_LEGACY_TEMPLATE_PATH)
            if pa_leg_path is not None:
                self.player_arrow_legacy_template = self._safe_imread(pa_leg_path, cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            logger.debug("Optional map template load warning: %s", exc)

        # Load or reconstruct floating [E] resource box template and 3D gather crest template
        self._load_e_gather_and_crest_templates()

        if not loaded_any:
            # No real template on disk: report failure and match against
            # nothing. A synthetic fake template would match arbitrary screen
            # content and report gathers that never happened.
            logger.error(
                "No gather template image found on disk; refusing to install "
                "a synthetic fallback template."
            )

        return loaded_any

    def _load_e_gather_and_crest_templates(self) -> None:
        """
        Loads `templates/e_gather_icon.png` and `templates/gather_crest.png` if present,
        or falls back to lossless reference crops / embedded compressed arrays.
        """
        # 1. Floating [E] resource icon template
        e_gray: Optional[np.ndarray] = None
        for rel_path in E_GATHER_ICON_TEMPLATE_PATHS:
            resolved = self._resolve_asset_path(rel_path)
            if resolved is not None:
                loaded = self._safe_imread(resolved, cv2.IMREAD_GRAYSCALE)
                if loaded is not None and loaded.size > 0:
                    e_gray = loaded
                    break

        if e_gray is None:
            pr1_path = self._resolve_asset_path("reference_screenshots/pilgrims_ridge_1.jpg")
            if pr1_path is not None:
                pr1 = self._safe_imread(pr1_path, cv2.IMREAD_GRAYSCALE)
                if pr1 is not None and pr1.size > 0:
                    # Reference crop was y=487..523, x=1312..1349 on a 1124x2000
                    # frame; scale it to the actual image size instead of
                    # hard-coding pixels from one reference resolution.
                    ph, pw = pr1.shape[:2]
                    cy1, cy2 = int(round(ph * 487 / 1124)), int(round(ph * 523 / 1124))
                    cx1, cx2 = int(round(pw * 1312 / 2000)), int(round(pw * 1349 / 2000))
                    if cy2 > cy1 and cx2 > cx1:
                        e_gray = pr1[cy1:cy2, cx1:cx2].copy()

        if e_gray is None:
            try:
                raw = zlib.decompress(base64.b64decode(_EMBEDDED_E_GATHER_BOX_B64))
                e_gray = np.frombuffer(raw, dtype=np.uint8).reshape(36, 37).copy()
            except Exception as exc:
                logger.debug("Embedded E gather icon decode warning: %s", exc)

        if e_gray is not None and e_gray.size > 0:
            self.e_gather_template = e_gray
            if e_gray.shape[0] >= 16 and e_gray.shape[1] >= 16:
                self.e_badge_template = e_gray[1:15, 1:15].copy()
            eh, ew = e_gray.shape[:2]
            for s in (0.60, 0.65, 0.72, 0.80, 0.90, 0.96, 1.00, 1.08, 1.20, 1.35):
                sw = max(12, int(round(ew * s)))
                sh = max(12, int(round(eh * s)))
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                resized = e_gray if abs(s - 1.0) < 1e-4 else cv2.resize(e_gray, (sw, sh), interpolation=interp)
                self._e_icon_scales.append((float(s), resized))

        # 2. 3D White Winged Gather Crest template
        c_gray: Optional[np.ndarray] = None
        for rel_path in GATHER_CREST_TEMPLATE_PATHS:
            resolved = self._resolve_asset_path(rel_path)
            if resolved is not None:
                loaded = self._safe_imread(resolved, cv2.IMREAD_GRAYSCALE)
                if loaded is not None and loaded.size > 0:
                    c_gray = loaded
                    break

        if c_gray is None:
            far_path = self._resolve_asset_path("reference_screenshots/aria_3d_crest_far.jpg")
            if far_path is not None:
                far_im = self._safe_imread(far_path, cv2.IMREAD_GRAYSCALE)
                if far_im is not None and far_im.size > 0:
                    # Reference crop was y=471..500, x=779..808 on a 981x2000
                    # frame; scale it to the actual image size.
                    fh, fw = far_im.shape[:2]
                    cy1, cy2 = int(round(fh * 471 / 981)), int(round(fh * 500 / 981))
                    cx1, cx2 = int(round(fw * 779 / 2000)), int(round(fw * 808 / 2000))
                    if cy2 > cy1 and cx2 > cx1:
                        c_gray = far_im[cy1:cy2, cx1:cx2].copy()

        if c_gray is None:
            try:
                raw_c = zlib.decompress(base64.b64decode(_EMBEDDED_3D_GATHER_CREST_B64))
                c_gray = np.frombuffer(raw_c, dtype=np.uint8).reshape(29, 29).copy()
            except Exception as exc:
                logger.debug("Embedded 3D gather crest decode warning: %s", exc)

        if c_gray is not None and c_gray.size > 0:
            self.gather_crest_template = c_gray
            ch, cw = c_gray.shape[:2]
            for s in (0.60, 0.65, 0.70, 0.78, 0.85, 0.92, 1.00, 1.08, 1.18, 1.30):
                sw = max(10, int(round(cw * s)))
                sh = max(10, int(round(ch * s)))
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                resized = c_gray if abs(s - 1.0) < 1e-4 else cv2.resize(c_gray, (sw, sh), interpolation=interp)
                self._crest_scales.append((float(s), resized))

    def _add_template_variant(
        self,
        name: str,
        template_gray: np.ndarray,
        offset_x: int,
        offset_y: int,
        full_w: int,
        full_h: int,
        is_pill: bool,
    ) -> None:
        """Pre-computes multi-scale resized templates for fast matching across 720p/1080p/2000p/1440p."""
        scales = (
            1.00,
            0.6667,
            0.75,
            0.85,
            1.15,
            1.3333,
            1.50,
        )
        scaled_items: List[Tuple[float, np.ndarray]] = []
        th, tw = template_gray.shape[:2]
        for s in scales:
            sw = max(8, int(round(tw * s)))
            sh = max(6, int(round(th * s)))
            if abs(s - 1.0) < 1e-4:
                resized = template_gray
            else:
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(template_gray, (sw, sh), interpolation=interp)
            scaled_items.append((float(s), resized))

        self._template_variants.append(
            {
                "name": name,
                "base": template_gray,
                "offset_x": int(offset_x),
                "offset_y": int(offset_y),
                "full_w": int(full_w),
                "full_h": int(full_h),
                "is_pill": bool(is_pill),
                "scales": scaled_items,
            }
        )

    def _get_map_header_scales(
        self, tmpl: np.ndarray
    ) -> List[Tuple[float, np.ndarray]]:
        """Returns the cached `MAP_HEADER_SCALES` pyramid for `tmpl`.

        `is_map_open` used to `cv2.resize` the same header template at all
        15 scales on every call; the resized pyramid only depends on the
        template pixels, so it is built once per template object and reused.
        The cache entry keeps a reference to the source template, so the
        identity check below can never match a different array that reused
        a freed id().
        """
        for cached_tmpl, cached_scales in self._map_header_scale_cache:
            if cached_tmpl is tmpl:
                return cached_scales
        th, tw = tmpl.shape[:2]
        scaled: List[Tuple[float, np.ndarray]] = []
        for s in MAP_HEADER_SCALES:
            sw, sh = max(10, int(round(tw * s))), max(6, int(round(th * s)))
            resized = (
                tmpl
                if abs(s - 1.0) < 1e-4
                else cv2.resize(tmpl, (sw, sh))
            )
            scaled.append((float(s), resized))
        self._map_header_scale_cache.append((tmpl, scaled))
        return scaled

    # ------------------------------------------------------------------
    # Screen Grabbing & Frame Coercion Helpers
    # ------------------------------------------------------------------
    def grab_screen(
        self,
        region: Optional[Union[Dict[str, int], Tuple[int, int, int, int]]] = None,
        grayscale: bool = True,
    ) -> np.ndarray:
        """Grabs full screen or specified region via thread-safe `ScreenCapture`."""
        if region is None:
            frame = self.capture.grab_full(grayscale=grayscale)
        else:
            frame = self.capture.grab_region(region, grayscale=grayscale)
        self.screen_w = self.capture.screen_w
        self.screen_h = self.capture.screen_h
        return frame

    def capture_screen(
        self,
        region: Optional[Union[Dict[str, int], Tuple[int, int, int, int]]] = None,
        grayscale: bool = False,
    ) -> np.ndarray:
        """Alias for `grab_screen` defaulting to BGR output."""
        return self.grab_screen(region=region, grayscale=grayscale)

    def validate_frame(self, frame: Optional[np.ndarray]) -> Tuple[bool, str]:
        """Delegates frame validation (including black screen detection) to `ScreenCapture`."""
        return self.capture.validate_frame(frame)

    def _coerce_frame(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]],
        grayscale: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Normalizes `frame` input:
        - If `None`, captures live screen via `ScreenCapture`.
        - If `str` or `PathLike`, loads image from disk (with fallback lookup in `video_analysis/`).
        - If `np.ndarray`, converts to grayscale or BGR as requested.
        """
        try:
            if frame is None:
                return self.grab_screen(region=None, grayscale=grayscale)

            if isinstance(frame, (str, os.PathLike)):
                path_str = str(frame)
                if not os.path.exists(path_str):
                    alt_path = os.path.join("video_analysis", os.path.basename(path_str))
                    if os.path.exists(alt_path):
                        path_str = alt_path
                flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                loaded = self._safe_imread(path_str, flag)
                return loaded

            if isinstance(frame, np.ndarray):
                if frame.size == 0:
                    return None
                if grayscale:
                    if frame.ndim == 2:
                        return frame
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    if frame.ndim == 2:
                        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    return frame
            return None
        except Exception as exc:
            logger.error("_coerce_frame failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 1. Dual Gather Prompt Detection ('F Gather' Pill + Floating [E]/[F] Box)
    # ------------------------------------------------------------------
    def detect_floating_resource_icon(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        threshold: float = 0.65,
        use_roi: bool = True,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        map_open: Optional[bool] = None,
    ) -> Tuple[bool, float, Optional[Tuple[int, int]], str]:
        """
        Detects the square floating `[E]` / `[F]` gatherable material icon in the 3D viewport
        (such as in `reference_screenshots/pilgrims_ridge_1.jpg`).

        Uses `templates/e_gather_icon.png` (or embedded reference template) with multi-scale
        template matching plus top-left `[E]`/`[F]` badge & metallic square border fallback.

        Args:
            frame: BGR/Grayscale image or path. If `None`, captures live screen.
            threshold: Minimum confidence in `[0.0, 1.0]` (default `0.65`).
            use_roi: Restrict search to mid-screen 3D viewport (`x: 0.25..0.75, y: 0.25..0.70`).
            roi_norm: Optional custom `(x1_norm, y1_norm, x2_norm, y2_norm)` bounds.
            map_open: Optional precomputed `is_map_open` verdict for this exact
                frame. Callers that already checked the map (e.g.
                `detect_gather_prompt`) pass it to avoid running the full
                map check a second time on the same frame; `None` (default)
                runs the check as before.

        Returns:
            `(found: bool, confidence: float, center_xy: Optional[Tuple[int, int]], detected_key: str)`
        """
        try:
            if not self._e_icon_scales:
                self._load_e_gather_and_crest_templates()

            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                # Sweep G: okunamayan karede bayat tuş dönme — bilinmiyor ("").
                return False, 0.0, None, ""

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr
            )
            h, w = img_gray.shape[:2]
            if h < 120 or w < 160:
                return False, 0.0, None, ""

            if map_open is None:
                map_open = self.is_map_open(img_bgr)
            # Sweep G: is_map_open artık üç durumlu (True/False/None).
            # Harita durumu bilinmiyorsa (None) harita açıkmış gibi davran:
            # bilinmeyen karede toplama aramak fail-open'dır.
            if map_open is not False:
                return False, 0.0, None, ""

            if use_roi:
                nx1, ny1, nx2, ny2 = roi_norm if roi_norm is not None else self.floating_icon_norm_roi
                rx1 = max(0, int(round(w * nx1)))
                ry1 = max(0, int(round(h * ny1)))
                rx2 = min(w, int(round(w * nx2)))
                ry2 = min(h, int(round(h * ny2)))
                search_gray = img_gray[ry1:ry2, rx1:rx2]
            else:
                rx1, ry1 = 0, 0
                search_gray = img_gray

            sh, sw = search_gray.shape[:2]
            best_score: float = 0.0
            best_center: Optional[Tuple[int, int]] = None
            # Sweep G: rozet okunamazsa tuş bilinmiyor ("") — son bilinen
            # tuş bu karede taze tespitmiş gibi taşınmaz.
            detected_key: str = ""

            # Pass 1: Multi-scale template matching on full floating [E] resource icon
            for _scale, templ in self._e_icon_scales:
                th, tw = templ.shape[:2]
                if th >= sh or tw >= sw:
                    continue
                res = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)
                if score <= best_score:
                    continue

                mx = int(max_loc[0])
                my = int(max_loc[1])
                patch = search_gray[my : my + th, mx : mx + tw]
                if patch.size > 0:
                    # Verify top-left bright key badge + square frame contrast
                    badge = patch[1 : max(5, int(th * 0.42)), 1 : max(5, int(tw * 0.42))]
                    if badge.size > 0 and (float(np.max(badge)) < 165.0 or float(np.min(badge)) > 95.0):
                        score *= 0.65

                if score > best_score:
                    best_score = score
                    cx = rx1 + mx + tw // 2
                    cy = ry1 + my + th // 2
                    best_center = (cx, cy)
                    badge_key = self._classify_badge_key(patch)
                    if badge_key in ("e", "f"):
                        detected_key = badge_key

            # Pass 2: Top-left [E] badge + square metallic border fallback (for different plant graphics inside box)
            if best_score < float(threshold) and self.e_badge_template is not None:
                bth, btw = self.e_badge_template.shape[:2]
                if bth < sh and btw < sw:
                    res_b = cv2.matchTemplate(search_gray, self.e_badge_template, cv2.TM_CCOEFF_NORMED)
                    _, mv_b, _, ml_b = cv2.minMaxLoc(res_b)
                    if float(mv_b) >= 0.72:
                        bx = int(ml_b[0]) - 1
                        by = int(ml_b[1]) - 1
                        box_w, box_h = 37, 36
                        if 0 <= bx and 0 <= by and bx + box_w <= sw and by + box_h <= sh:
                            box = search_gray[by : by + box_h, bx : bx + box_w]
                            top_b = float(np.mean(box[1:3, 4:33]))
                            bot_b = float(np.mean(box[33:35, 4:33]))
                            left_b = float(np.mean(box[4:33, 1:3]))
                            right_b = float(np.mean(box[4:33, 34:36]))
                            if min(top_b, bot_b, left_b, right_b) >= 75.0:
                                score_b = float(mv_b) * 0.95
                                if score_b > best_score:
                                    best_score = score_b
                                    best_center = (rx1 + bx + box_w // 2, ry1 + by + box_h // 2)
                                    badge_key = self._classify_badge_key(box)
                                    if badge_key in ("e", "f"):
                                        detected_key = badge_key

            if best_center is not None and img_bgr is not None:
                try:
                    from dataset_factory.exemplar_memory import get_exemplar_memory
                    mem = get_exemplar_memory()
                    bcx, bcy = best_center
                    crop = img_bgr[max(0, bcy - 20) : min(img_bgr.shape[0], bcy + 20), max(0, bcx - 20) : min(img_bgr.shape[1], bcx + 20)]
                    if crop.size > 0:
                        # Sweep G: exemplar yalnızca VETO yapar — negatif
                        # eşleşme tespiti düşürür, pozitif benzerlik skoru
                        # yukarı şişirmez (eski `max(best_score, pos_sim)`
                        # bayat/yanlış exemplarla güveni tek yönlü artırıyordu).
                        is_neg, _, _ = mem.is_negative_patch(crop)
                        if is_neg:
                            best_score = 0.0
                            best_center = None
                except Exception as exc:
                    # Exemplar verification is auxiliary: a failure must not
                    # silently pass the detection through as verified.
                    logger.debug("exemplar verification skipped for floating icon: %s", exc)

            found = bool(best_score >= float(threshold) and best_center is not None)
            if found and detected_key in ("e", "f"):
                self.last_detected_gather_key = detected_key
            return found, round(float(np.clip(best_score, 0.0, 1.0)), 4), best_center if found else None, detected_key
        except Exception as exc:
            logger.error("detect_floating_resource_icon failed: %s", exc)
            return False, 0.0, None, ""

    @staticmethod
    def _classify_badge_key(box_gray: np.ndarray) -> str:
        """
        Distinguishes whether the top-left key badge in a floating resource box is 'e' or 'f'
        by checking the bottom horizontal bar of the character glyph.
        Returns "unknown" when the glyph matches neither letter confidently:
        an ambiguous badge must never be silently reported as 'e' (callers
        decide what to do with "unknown" instead of pressing the wrong key).
        """
        try:
            h, w = box_gray.shape[:2]
            glyph = box_gray[max(1, int(h * 0.14)) : max(4, int(h * 0.34)), max(1, int(w * 0.14)) : max(4, int(w * 0.30))]
            if glyph.size == 0:
                return "unknown"
            # 'E' has a bright bottom horizontal stroke; 'F' is open at bottom-right
            bot_row = float(np.mean(glyph[-2:, -3:]))
            top_row = float(np.mean(glyph[:2, -3:]))
            if top_row > 140.0 and bot_row < 75.0:
                return "f"
            if top_row > 140.0 and bot_row >= 140.0:
                return "e"
            return "unknown"
        except Exception:
            return "unknown"

    def detect_gather_prompt(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        threshold: float = 0.68,
        use_roi: bool = True,
    ) -> Tuple[bool, int, Optional[Tuple[int, int, int, int]]]:
        """
        Detects the in-game gather interaction prompt on screen:
        1. Checks the classic `'F Gather'` pill/template (`video_frames/frame_33s.jpg`,
           `aria_f_gather_not_on_minimap.jpg`, `mela_f_gather_not_on_minimap.jpg`).
           When found, sets `self.last_detected_gather_key = "f"`.
        2. Checks for a close-range floating `[E]`/`[F]` interaction box inside the
           interaction ROI (`x: 0.42..0.69, y: 0.38..0.66`, as in `pilgrims_ridge_1.jpg`).
           When found, sets `self.last_detected_gather_key = "e"` (or `"f"`).

        Args:
            frame: Optional BGR/Grayscale image or file path. If `None`, captures live screen.
            threshold: Normalized match threshold in `[0.0, 1.0]` (default `0.68`).
            use_roi: Whether to restrict search to the central/character interaction zone
                     for speed and reduced false positives.

        Returns:
            `(is_found: bool, confidence_pct: int, bbox: tuple[int, int, int, int] | None)`
            where `bbox = (x, y, w, h)` in full-frame pixel coordinates.
        """
        try:
            if not self._template_variants:
                self.load_template()

            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return False, 0, None

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr
            )
            h, w = img_gray.shape[:2]

            # Quick reject if map is open — or its state is unknown.
            # Sweep G: is_map_open None (bilinmiyor) dönerse de toplama
            # aranmaz; bozuk karede harita açıkken F/E basmak fail-open'dır.
            if self.is_map_open(img_bgr) is not False:
                return False, 0, None

            if use_roi and w >= 320 and h >= 240:
                nx, ny, nw, nh = self.prompt_norm_roi
                rx1 = max(0, int(round(w * nx)))
                ry1 = max(0, int(round(h * ny)))
                rx2 = min(w, int(round(w * (nx + nw))))
                ry2 = min(h, int(round(h * (ny + nh))))
                search_gray = img_gray[ry1:ry2, rx1:rx2]
            else:
                rx1, ry1 = 0, 0
                search_gray = img_gray

            sh, sw = search_gray.shape[:2]
            best_score: float = 0.0
            best_bbox: Optional[Tuple[int, int, int, int]] = None

            for variant in self._template_variants:
                is_pill: bool = variant["is_pill"]
                offset_x: int = variant["offset_x"]
                offset_y: int = variant["offset_y"]
                full_w: int = variant["full_w"]
                full_h: int = variant["full_h"]

                for scale, templ in variant["scales"]:
                    th, tw = templ.shape[:2]
                    if th >= sh or tw >= sw:
                        continue

                    res = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    score = float(max_val)

                    if score <= best_score:
                        continue

                    match_x = rx1 + int(max_loc[0])
                    match_y = ry1 + int(max_loc[1])

                    # Verify dark pill + bright white 'F Gather' text contrast
                    if is_pill:
                        patch = img_gray[match_y : match_y + th, match_x : match_x + tw]
                        if patch.size > 0:
                            p25 = float(np.percentile(patch, 25))
                            p92 = float(np.percentile(patch, 92))
                            if p25 > 80.0 or (p92 - p25) < 60.0:
                                score *= 0.60

                    if score > best_score:
                        best_score = score
                        box_x = max(0, int(round(match_x - offset_x * scale)))
                        box_y = max(0, int(round(match_y - offset_y * scale)))
                        box_w = min(w - box_x, max(tw, int(round(full_w * scale))))
                        box_h = min(h - box_y, max(th, int(round(full_h * scale))))
                        best_bbox = (box_x, box_y, box_w, box_h)
                        if best_score >= 0.94:
                            break
                if best_score >= 0.94:
                    break

            if best_score >= float(threshold) and best_bbox is not None:
                self.last_detected_gather_key = "f"
                confidence_pct = int(round(np.clip(best_score, 0.0, 1.0) * 100.0))
                return True, confidence_pct, best_bbox

            # Check close-range floating [E]/[F] interaction box (x: 0.42..0.69, y: 0.38..0.66).
            # The map is already known closed here (early return above), so
            # pass the verdict down instead of re-running is_map_open on
            # the same frame inside detect_floating_resource_icon.
            e_found, e_conf, e_center, e_key = self.detect_floating_resource_icon(
                frame=img_bgr,
                threshold=max(0.65, float(threshold)),
                use_roi=use_roi,
                roi_norm=self.close_interaction_norm_roi,
                map_open=False,
            )
            if e_found and e_center is not None and e_conf >= best_score:
                if e_key in ("e", "f"):
                    self.last_detected_gather_key = e_key
                cx, cy = e_center
                bw, bh = 37, 36
                bx = max(0, cx - bw // 2)
                by = max(0, cy - bh // 2)
                confidence_pct = int(round(np.clip(e_conf, 0.0, 1.0) * 100.0))
                return True, confidence_pct, (bx, by, bw, bh)

            confidence_pct = int(round(np.clip(best_score, 0.0, 1.0) * 100.0))
            return False, confidence_pct, None
        except Exception as exc:
            logger.error("detect_gather_prompt failed: %s", exc)
            return False, 0, None

    def check_gather_prompt(
        self,
        threshold: float = 0.68,
        use_roi: bool = True,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Tuple[bool, int, Optional[Tuple[int, int, int, int]]]:
        """Backward-compatible alias for `detect_gather_prompt`."""
        return self.detect_gather_prompt(frame=frame, threshold=threshold, use_roi=use_roi)

    def find_gather_prompt(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        threshold: float = 0.68,
    ) -> Tuple[bool, int, Optional[Tuple[int, int, int, int]]]:
        """Alias for `detect_gather_prompt`."""
        return self.detect_gather_prompt(frame=frame, threshold=threshold, use_roi=True)

    # ------------------------------------------------------------------
    # 1B. Player Self-Exclusion Zone & 3D Gather Crest Alignment Proof
    # ------------------------------------------------------------------
    def is_in_player_exclusion_zone(
        self,
        x: Union[int, float],
        y: Union[int, float],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """
        Determines whether coordinate (x, y) or bbox falls within the player center
        self-exclusion zone (normalized X in [0.43, 0.57], Y in [0.38, 0.64]).
        Used to eliminate false 3D monster and resource detections originating from
        the player's own character model, cape, or mount.
        """
        return is_in_player_exclusion_zone(
            x=x,
            y=y,
            frame_w=frame_w or self.screen_w,
            frame_h=frame_h or self.screen_h,
            bbox=bbox,
            exclusion_zone=self.player_center_exclusion_zone,
        )

    @staticmethod
    def _unverified_crest_proof(raw_score: float) -> Dict[str, Any]:
        """Unverified-crest result shared by all no-proof exits of
        `verify_crest_alignment_proof`: genuine low confidence in
        [0.50..0.68] scaled from the raw template score (never a fixed
        placeholder value)."""
        base_norm = np.clip((raw_score - 0.74) / 0.26, 0.0, 1.0) if raw_score >= 0.74 else 0.5
        conf = round(float(np.clip(0.50 + 0.18 * base_norm, 0.50, 0.68)), 4)
        return {
            "has_proof": False,
            "trusted": False,
            "weak": True,
            "proof_type": "none",
            "dx": None,
            "dy": None,
            "confidence": conf,
            "nameplate_detected": False,
            "prompt_detected": False,
        }

    def verify_crest_alignment_proof(
        self,
        crest_xy: Tuple[Union[int, float], Union[int, float]],
        frame_bgr: Optional[np.ndarray] = None,
        raw_score: float = 0.85,
        max_abs_dx: int = 45,
        min_dy: int = 0,
        max_dy: int = 70,
    ) -> Dict[str, Any]:
        """
        Verifies vertical spatial alignment between a detected 3D winged crest (`\\Y/`)
        and either:
        (1) Cyan/blue gatherable nameplate text (`‹ Aria ›`, `‹ Mela ›`), or
        (2) An `[F]` gather interaction prompt or floating key icon.

        Alignment criteria:
            `abs(dx) <= 45px` and `0 <= dy <= 70px`
            where `dx = target_cx - crest_cx` and `dy = target_cy - crest_cy`.

        - If verified by alignment proof:
          assigns confidence in [0.88 .. 0.96] (`trusted = True, weak = False`),
          measured from the proof strength (nameplate glyph count / prompt
          template-match score) — not a fixed formula.
        - If crest is an unverified isolated contour without nameplate or prompt proof:
          assigns genuine confidence in [0.50 .. 0.68] (`trusted = False, weak = True`).

        Returns:
            Dict containing:
            - `has_proof: bool` (True if vertically aligned with nameplate or [F] prompt)
            - `trusted: bool` (True if verified, False if unverified)
            - `weak: bool` (False if verified, True if unverified)
            - `proof_type: str` ('nameplate', 'f_prompt', 'nameplate+f_prompt', or 'none')
            - `dx: Optional[float]` (horizontal offset in pixels)
            - `dy: Optional[float]` (vertical downward offset in pixels)
            - `confidence: float` (measured 0.88..0.96 if verified; 0.50..0.68 if unverified)
            - `nameplate_detected: bool`
            - `prompt_detected: bool`
        """
        crest_cx = int(round(crest_xy[0]))
        crest_cy = int(round(crest_xy[1]))

        if frame_bgr is None or frame_bgr.size == 0:
            return self._unverified_crest_proof(raw_score)

        h, w = frame_bgr.shape[:2]
        # Search window directly below and horizontally centered on the crest
        x1 = max(0, crest_cx - max_abs_dx - 20)
        x2 = min(w, crest_cx + max_abs_dx + 20)
        y1 = max(0, crest_cy - 5)
        y2 = min(h, crest_cy + max_dy + 30)

        win_bgr = frame_bgr[y1:y2, x1:x2]
        if win_bgr.size == 0 or win_bgr.shape[0] < 4 or win_bgr.shape[1] < 4:
            return self._unverified_crest_proof(raw_score)

        win_hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
        win_gray = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------------------
        # 1. Cyan/Blue Gatherable Nameplate Text Detection
        # --------------------------------------------------------------
        mask_cyan_blue = (
            ((win_hsv[:, :, 0] >= 45) & (win_hsv[:, :, 0] <= 135) & (win_hsv[:, :, 1] >= 32) & (win_hsv[:, :, 2] >= 105))
            | ((win_hsv[:, :, 1] <= 40) & (win_hsv[:, :, 2] >= 190))
        )
        local_min = cv2.erode(win_gray, _ERODE_KERNEL_5X5)
        local_contrast = ((win_gray.astype(np.int16) - local_min.astype(np.int16)) >= 45) & (local_min <= 95)
        mask_text = mask_cyan_blue & (local_contrast | (win_hsv[:, :, 1] >= 60))

        cnts, _ = cv2.findContours(mask_text.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_glyphs: List[Tuple[float, float, int, int]] = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if 2 <= bh <= 28 and 1 <= bw <= 55 and area >= 3:
                gcx = float(x1 + bx) + float(bw) / 2.0
                gcy = float(y1 + by) + float(bh) / 2.0
                gdx = gcx - float(crest_cx)
                gdy = gcy - float(crest_cy)
                if abs(gdx) <= max_abs_dx and min_dy <= gdy <= max_dy:
                    valid_glyphs.append((gcx, gcy, bw, bh))

        nameplate_detected = len(valid_glyphs) >= 2 or (
            len(valid_glyphs) >= 1 and sum(g[2] * g[3] for g in valid_glyphs) >= 24
        )
        nameplate_dx: Optional[float] = None
        nameplate_dy: Optional[float] = None
        if nameplate_detected and valid_glyphs:
            nameplate_dx = float(np.median([g[0] for g in valid_glyphs])) - float(crest_cx)
            nameplate_dy = float(np.median([g[1] for g in valid_glyphs])) - float(crest_cy)

        # --------------------------------------------------------------
        # 2. [F] Gather Prompt or Floating Key Badge Detection
        # --------------------------------------------------------------
        prompt_detected = False
        prompt_dx: Optional[float] = None
        prompt_dy: Optional[float] = None
        # Measured proof strength in [0.0, 1.0] (template-match score or
        # bright-pixel evidence); drives the verified confidence below.
        prompt_evidence: float = 0.0

        # 2a. Template match gather pill tight if available
        if (
            self.gather_pill_tight is not None
            and self.gather_pill_tight.shape[0] < win_gray.shape[0]
            and self.gather_pill_tight.shape[1] < win_gray.shape[1]
        ):
            res_p = cv2.matchTemplate(win_gray, self.gather_pill_tight, cv2.TM_CCOEFF_NORMED)
            _, mv_p, _, ml_p = cv2.minMaxLoc(res_p)
            if float(mv_p) >= 0.65:
                pcx = float(x1 + ml_p[0]) + float(self.gather_pill_tight.shape[1]) / 2.0
                pcy = float(y1 + ml_p[1]) + float(self.gather_pill_tight.shape[0]) / 2.0
                pdx = pcx - float(crest_cx)
                pdy = pcy - float(crest_cy)
                if abs(pdx) <= max_abs_dx and min_dy <= pdy <= max_dy:
                    prompt_detected = True
                    prompt_dx = pdx
                    prompt_dy = pdy
                    prompt_evidence = float(np.clip((float(mv_p) - 0.65) / 0.35, 0.0, 1.0))

        # 2b. Dark pill container with bright [F] text check
        if not prompt_detected:
            dark_pill = (win_gray >= 12) & (win_gray <= 65)
            p_cnts, _ = cv2.findContours(dark_pill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pc in p_cnts:
                pbx, pby, pbw, pbh = cv2.boundingRect(pc)
                if pbw >= 20 and 8 <= pbh <= 40 and float(pbw) / float(max(1, pbh)) >= 1.4:
                    sub = win_gray[pby : pby + pbh, pbx : pbx + pbw]
                    bright_px = int(np.count_nonzero(sub > 175))
                    if bright_px >= 6 or float(np.mean(sub > 175)) >= 0.012:
                        pcx = float(x1 + pbx) + float(pbw) / 2.0
                        pcy = float(y1 + pby) + float(pbh) / 2.0
                        pdx = pcx - float(crest_cx)
                        pdy = pcy - float(crest_cy)
                        if abs(pdx) <= max_abs_dx and min_dy <= pdy <= max_dy:
                            prompt_detected = True
                            prompt_dx = pdx
                            prompt_dy = pdy
                            prompt_evidence = float(np.clip(bright_px / 40.0, 0.0, 1.0))
                            break

        # 2c. Floating [E]/[F] key icon template match
        if (
            not prompt_detected
            and self.e_badge_template is not None
            and self.e_badge_template.shape[0] < win_gray.shape[0]
            and self.e_badge_template.shape[1] < win_gray.shape[1]
        ):
            res_e = cv2.matchTemplate(win_gray, self.e_badge_template, cv2.TM_CCOEFF_NORMED)
            _, mv_e, _, ml_e = cv2.minMaxLoc(res_e)
            if float(mv_e) >= 0.68:
                ecx = float(x1 + ml_e[0]) + float(self.e_badge_template.shape[1]) / 2.0
                ecy = float(y1 + ml_e[1]) + float(self.e_badge_template.shape[0]) / 2.0
                edx = ecx - float(crest_cx)
                edy = ecy - float(crest_cy)
                if abs(edx) <= max_abs_dx and min_dy <= edy <= max_dy:
                    prompt_detected = True
                    prompt_dx = edx
                    prompt_dy = edy
                    prompt_evidence = float(np.clip((float(mv_e) - 0.68) / 0.32, 0.0, 1.0))

        # --------------------------------------------------------------
        # 3. Decision & Genuine Confidence Calibration
        # --------------------------------------------------------------
        has_proof = bool(nameplate_detected or prompt_detected)

        if has_proof:
            if nameplate_detected and prompt_detected:
                proof_type = "nameplate+f_prompt"
                chosen_dx = nameplate_dx
                chosen_dy = nameplate_dy
            elif nameplate_detected:
                proof_type = "nameplate"
                chosen_dx = nameplate_dx
                chosen_dy = nameplate_dy
            else:
                proof_type = "f_prompt"
                chosen_dx = prompt_dx
                chosen_dy = prompt_dy

            cdx = chosen_dx if chosen_dx is not None else 0.0
            cdy = chosen_dy if chosen_dy is not None else 35.0

            # Measured confidence: the strongest measured proof evidence —
            # nameplate glyph count (a full "‹ Aria ›" nameplate yields ~6
            # glyphs) or the prompt template-match / bright-pixel evidence.
            # Alignment geometry alone must not mint a high confidence.
            evidence = 0.0
            if nameplate_detected:
                evidence = max(evidence, min(1.0, len(valid_glyphs) / 6.0))
            if prompt_detected:
                evidence = max(evidence, float(np.clip(prompt_evidence, 0.0, 1.0)))
            conf = round(float(np.clip(0.88 + 0.08 * evidence, 0.88, 0.96)), 4)
            return {
                "has_proof": True,
                "trusted": True,
                "weak": False,
                "proof_type": proof_type,
                "dx": round(cdx, 2),
                "dy": round(cdy, 2),
                "confidence": conf,
                "nameplate_detected": bool(nameplate_detected),
                "prompt_detected": bool(prompt_detected),
            }
        else:
            return self._unverified_crest_proof(raw_score)

    def detect_3d_gather_crests(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        threshold: float = 0.74,
    ) -> List[Dict[str, Any]]:
        """
        Detects the white winged gather crest icons floating above gatherable plants
        (such as `Aria` and `Mela` in `aria_3d_crest_far.jpg`, `aria_f_gather_not_on_minimap.jpg`,
        `mela_f_gather_not_on_minimap.jpg`, and `video_frames/frame_33s.jpg`) in the 3D world.

        Excludes false detections originating from the player center self-exclusion zone
        (normalized X in [0.43, 0.57], Y in [0.38, 0.64] - player character model, cape, or mount).

        Performs mandatory vertical spatial alignment proof with cyan/blue gatherable nameplate
        text or [F] prompt (abs(dx) <= 45px and 0 <= dy <= 70px):
        - If verified by alignment proof: assigns measured confidence 0.88 .. 0.96 (trusted = True, weak = False).
        - If crest is an unverified isolated contour without proof: assigns 0.50 .. 0.68 (trusted = False, weak = True).

        Returns:
            List of dicts sorted by confidence descending, each containing:
            `{"x": int, "y": int, "center": (int, int), "bbox": (int, int, int, int), "confidence": float,
              "trusted": bool, "weak": bool, "has_alignment_proof": bool, "proof_type": str, ...}`
        """
        try:
            if not self._crest_scales:
                self._load_e_gather_and_crest_templates()

            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return []

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr
            )
            h, w = img_gray.shape[:2]
            if h < 180 or w < 240:
                return []

            # P1-3 (K-07) istisnası: harita durumu bilinmiyorsa (None) crest
            # aranır — yalnızca pozitif "harita açık" (True) kanıtı aramayı
            # durdurur. `detect_gather_prompt` / `detect_floating_resource_icon`
            # aksine bu fonksiyon tuş basmaz (fail-open eylemi yok); pasif
            # tespittir ve her aday zaten zorunlu hizalama-kanıtı adımından
            # geçer (kanıtsız aday weak 0.50..0.68, trusted=False). v18'de
            # `is_map_open` kanıtsız karede None dönmeye başlayınca eski
            # `is not False` kapısı bilinmeyen her karede tespiti
            # körlüyordu (5 yapıştırılmış crest → 0).
            if self.is_map_open(img_bgr) is True:
                return []

            # Search 3D world viewport excluding top bars, right minimap/quest tracker, and bottom skill bar
            rx1, rx2 = int(round(w * 0.12)), int(round(w * 0.82))
            ry1, ry2 = int(round(h * 0.18)), int(round(h * 0.76))
            roi_gray = img_gray[ry1:ry2, rx1:rx2]
            rh, rw = roi_gray.shape[:2]

            candidates: List[Dict[str, Any]] = []
            for _scale, templ in self._crest_scales:
                th, tw = templ.shape[:2]
                if th >= rh or tw >= rw:
                    continue

                res = cv2.matchTemplate(roi_gray, templ, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= float(threshold))
                for my, mx in zip(ys.tolist(), xs.tolist()):
                    score = float(res[my, mx])
                    patch = roi_gray[my : my + th, mx : mx + tw]
                    if patch.size == 0:
                        continue
                    # Verify bright white crest pixels (> 185) and dark outline contrast
                    p90 = float(np.percentile(patch, 90))
                    p25 = float(np.percentile(patch, 25))
                    if p90 < 175.0 or (p90 - p25) < 65.0:
                        continue

                    abs_x = rx1 + mx
                    abs_y = ry1 + my
                    cx = abs_x + tw // 2
                    cy = abs_y + th // 2
                    candidates.append(
                        {
                            "x": int(cx),
                            "y": int(cy),
                            "center": (int(cx), int(cy)),
                            "bbox": (int(abs_x), int(abs_y), int(tw), int(th)),
                            "confidence": round(score, 4),
                        }
                    )

            if not candidates:
                return []

            # Cap candidates before NMS + alignment proof: a many-match frame
            # (thousands of raw template hits) would otherwise spend seconds
            # in the per-candidate proof. Keep the strongest matches only.
            if len(candidates) > MAX_CREST_CANDIDATES:
                candidates.sort(key=lambda d: d["confidence"], reverse=True)
                candidates = candidates[:MAX_CREST_CANDIDATES]

            # Non-maximum suppression (merge detections within 28px radius)
            candidates.sort(key=lambda d: d["confidence"], reverse=True)
            # Resolve the exemplar-memory singleton once per call, not once
            # per candidate (up to MAX_CREST_CANDIDATES lock acquisitions
            # and imports for the same object). A failure still only skips
            # verification; candidates continue to the alignment proof.
            try:
                from dataset_factory.exemplar_memory import get_exemplar_memory

                crest_mem = get_exemplar_memory()
            except Exception as exc:
                logger.debug("exemplar memory unavailable for crest candidates: %s", exc)
                crest_mem = None
            kept: List[Dict[str, Any]] = []
            for cand in candidates:
                cx, cy = cand["x"], cand["y"]
                if any((cx - k["x"]) ** 2 + (cy - k["y"]) ** 2 <= 28 ** 2 for k in kept):
                    continue

                # Player self-exclusion zone: reject false detections on player model/cape/mount
                if self.is_in_player_exclusion_zone(cx, cy, w, h, bbox=cand.get("bbox")):
                    continue

                # Exemplar memory verification (singleton resolved once above)
                if crest_mem is not None:
                    try:
                        bx, by, bw, bh = cand["bbox"]
                        crop = img_bgr[by : by + bh, bx : bx + bw]
                        if crop.size > 0:
                            # Sweep G: exemplar yalnızca VETO yapar — negatif
                            # eşleşme adayı eler. Pozitif benzerlikle güveni
                            # yukarı şişiren `max(conf, pos_sim)` kaldırıldı.
                            is_neg, _, _ = crest_mem.is_negative_patch(crop)
                            if is_neg:
                                continue
                    except Exception as exc:
                        # Exemplar verification is auxiliary: log the skip instead
                        # of silently treating the candidate as verified.
                        logger.debug("exemplar verification skipped for crest candidate: %s", exc)

                # Vertical spatial alignment proof with cyan/blue nameplate text or [F] prompt
                proof = self.verify_crest_alignment_proof(
                    crest_xy=(cx, cy),
                    frame_bgr=img_bgr,
                    raw_score=cand["confidence"],
                )
                cand["has_alignment_proof"] = bool(proof["has_proof"])
                cand["proof_type"] = str(proof["proof_type"])
                cand["trusted"] = bool(proof["trusted"])
                cand["weak"] = bool(proof["weak"])
                cand["confidence"] = float(proof["confidence"])
                if proof.get("dx") is not None:
                    cand["alignment_dx"] = float(proof["dx"])
                if proof.get("dy") is not None:
                    cand["alignment_dy"] = float(proof["dy"])

                kept.append(cand)

            kept.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
            return kept
        except Exception as exc:
            logger.error("detect_3d_gather_crests failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # 1C. Minimap Filter Settings Popup & Checkbox Verification
    # ------------------------------------------------------------------
    def detect_minimap_filter_settings(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Dict[str, Any]:
        """
        Detects whether the `Minimap Filter Settings` popup is open (both in closeup crops
        like `reference_screenshots/minimap_filter_closeup.jpg` and full-screen frames like
        `aria_f_gather_not_on_minimap.jpg`) and inspects the gold/white checkmark ticks
        inside the 4 filter checkboxes (`Quest`, `Monster`, `Gatherable Material`, `Functional NPC`).

        Returns:
            Dict containing:
            - `is_open: bool`
            - `panel_bbox: Optional[Tuple[int, int, int, int]]`
            - `checkboxes: Dict[str, bool]`
            - `quest: bool`, `monster: bool`, `gatherable_material: bool`, `functional_npc: bool`
            - `all_checked: bool`
            - `checkbox_coords: Dict[str, Tuple[int, int]]`
        """
        default_result: Dict[str, Any] = {
            "is_open": False,
            "panel_bbox": None,
            "checkboxes": {},
            "quest": False,
            "monster": False,
            "gatherable_material": False,
            "functional_npc": False,
            "all_checked": False,
            "checkbox_coords": {},
        }
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return default_result

            h, w = img_bgr.shape[:2]
            if h < 120 or w < 160:
                return default_result

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # In full-screen frames (w >= 1000), the popup sits to the left of the minimap (x: 0.68..0.86, y: 0.04..0.34).
            # In closeup crops (like minimap_filter_closeup.jpg, 718x362), it sits in the left half (x: 0.0..0.50, y: 0.15..0.96).
            if w >= 1000:
                sx1, sx2 = int(w * 0.68), int(w * 0.86)
                sy1, sy2 = int(h * 0.04), int(h * 0.34)
            else:
                sx1, sx2 = 0, int(w * 0.50)
                sy1, sy2 = int(h * 0.15), int(h * 0.96)

            roi_g = gray[sy1:sy2, sx1:sx2]
            roi_h = hsv[sy1:sy2, sx1:sx2]
            if roi_g.size == 0:
                return default_result

            slate = (roi_g >= 28) & (roi_g <= 55) & (roi_h[:, :, 1] <= 80)

            valid_rows = np.where(np.mean(slate, axis=1) > 0.45)[0]
            if len(valid_rows) < 15:
                return default_result
            py1, py2 = int(valid_rows[0]), int(valid_rows[-1])

            col_slate = np.mean(slate[py1 : py2 + 1, :], axis=0)
            valid_cols = np.where(col_slate > 0.55)[0]
            if len(valid_cols) < 15:
                return default_result
            px1, px2 = int(valid_cols[0]), int(valid_cols[-1])

            # Refine vertical bounds strictly within [px1:px2]
            valid_rows2 = np.where(np.mean(slate[:, px1 : px2 + 1], axis=1) > 0.55)[0]
            if len(valid_rows2) < 15:
                return default_result
            py1, py2 = int(valid_rows2[0]), int(valid_rows2[-1])
            ph = py2 - py1 + 1

            # Refine horizontal bounds strictly within [py1:py2]
            valid_cols2 = np.where(np.mean(slate[py1 : py2 + 1, :], axis=0) > 0.55)[0]
            if len(valid_cols2) < 15:
                return default_result
            px1, px2 = int(valid_cols2[0]), int(valid_cols2[-1])
            pw = px2 - px1 + 1

            if pw < 110 or ph < 90 or not (0.90 <= pw / float(ph) <= 1.40):
                return default_result

            # Verify uniform horizontal separator strip below 'Minimap Filter Settings' header
            sep_strip = roi_g[
                py1 + int(ph * 0.13) : py1 + int(ph * 0.23),
                px1 + int(pw * 0.08) : px1 + int(pw * 0.92),
            ]
            if sep_strip.size == 0 or float(np.min(np.std(sep_strip, axis=1))) > 4.2:
                return default_result

            # Verify bright white 'Minimap Filter Settings' title text
            hdr = roi_g[
                py1 + int(ph * 0.03) : py1 + int(ph * 0.15),
                px1 + int(pw * 0.05) : px1 + int(pw * 0.85),
            ]
            if hdr.size == 0 or float(np.mean(hdr > 160)) < 0.05:
                return default_result

            abs_px = sx1 + px1
            abs_py = sy1 + py1
            cx1 = abs_px + int(round(pw * 0.85))
            cx2 = abs_px + int(round(pw * 0.96))

            rows_spec = (
                ("quest", "Quest", 0.22, 0.35),
                ("monster", "Monster", 0.40, 0.53),
                ("gatherable_material", "Gatherable Material", 0.59, 0.72),
                ("functional_npc", "Functional NPC", 0.78, 0.91),
            )
            checkboxes: Dict[str, bool] = {}
            checkbox_coords: Dict[str, Tuple[int, int]] = {}

            for key, _label, yr1, yr2 in rows_spec:
                cy1 = abs_py + int(round(ph * yr1))
                cy2 = abs_py + int(round(ph * yr2))
                cbox_g = gray[cy1:cy2, cx1:cx2]
                inner_g = (
                    cbox_g[2:-2, 2:-2]
                    if cbox_g.shape[0] > 6 and cbox_g.shape[1] > 6
                    else cbox_g
                )
                max_val = float(np.max(inner_g)) if inner_g.size > 0 else 0.0
                bright_ratio = float(np.mean(inner_g > 155)) if inner_g.size > 0 else 0.0
                checked = bool(max_val >= 175.0 and bright_ratio >= 0.04)
                # One canonical key per checkbox ("quest", "monster",
                # "gatherable_material", "functional_npc"); the display label
                # is not a second key for the same box.
                checkboxes[key] = checked
                center_pt = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                checkbox_coords[key] = center_pt

            all_checked = all(
                checkboxes[k]
                for k in ("quest", "monster", "gatherable_material", "functional_npc")
            )
            return {
                "is_open": True,
                "panel_bbox": (int(abs_px), int(abs_py), int(pw), int(ph)),
                "checkboxes": checkboxes,
                "quest": checkboxes["quest"],
                "monster": checkboxes["monster"],
                "gatherable_material": checkboxes["gatherable_material"],
                "functional_npc": checkboxes["functional_npc"],
                "all_checked": bool(all_checked),
                "checkbox_coords": checkbox_coords,
            }
        except Exception as exc:
            logger.error("detect_minimap_filter_settings failed: %s", exc)
            return default_result

    # ------------------------------------------------------------------
    # 2. World Map ('M') Open Detection
    # ------------------------------------------------------------------
    def is_map_open(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Optional[bool]:
        """
        Determines whether the in-game World Map (`M`) is currently open.

        Combines multi-scale `templates/map_header.png` template matching with
        structural & color verification across all zones (`Altgard`, `Pilgrim's Ridge`,
        `Western Altamia Highland`) and resolutions (`720p`, `1080p`, `2000x1125`, `1440p`).

        Sweep G: üç durumlu dönüş — `True` (harita açık), `False` (harita
        kapalı, oyun ekranı) veya `None` (bilinmiyor: bozuk/eksik kare,
        doğrulama hatası). Eskiden her hata `False` dönüyordu (fail-open):
        bozuk karede bot harita açıkken bile toplama arayıp F/E basıyordu.
        Çağıranlar `None`'u "harita açık olabilir" saymalı — `bool(None)`
        ile `False`'a indirgemek fail-open'ı geri getirir. Bu dosyadaki
        iç çağıranlar (`detect_gather_prompt`, `is_game_window_blocking`,
        `detect_floating_resource_icon`) `None`'u açık sayar.
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return None

            valid, _ = self.capture.validate_frame(img_bgr)
            if not valid:
                return None

            h, w = img_bgr.shape[:2]
            if h < 120 or w < 160:
                return None

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Grayscale is only needed by the chrome/header checks below;
            # gameplay frames already return at step 1, so it is built
            # lazily there instead of on every call.

            # 1. Check bottom-center HP & MP bars (visible during normal 3D world gameplay)
            hp_rect = None
            mp_rect = None
            try:
                from dataset_factory.exemplar_memory import get_calibrated_rois
                cm = get_calibrated_rois()
                hp_rect = cm.get_normalized_rect("player_hp_bar")
                mp_rect = cm.get_normalized_rect("player_mp_bar")
            except Exception as exc:
                # Sweep G: kalibre yükleme hatası artık sessizce yutulmuyor.
                logger.debug("is_map_open calibrated ROI load failed: %s", exc)

            if hp_rect:
                hp_roi = hsv[int(h * hp_rect[1]) : int(h * hp_rect[3]), int(w * hp_rect[0]) : int(w * hp_rect[2])]
            else:
                # Sweep G: tek yedek ROI (FALLBACK_PLAYER_HP_BAR_ROI) —
                # eskiden buradaki (0.36-0.47, 0.87-0.92) dilim,
                # read_player_hp_mp_bars yedeğinden (0.345-0.485, 0.880-0.915)
                # farklıydı; iki fonksiyon aynı bar hakkında çelişebiliyordu.
                hp_roi = hsv[int(h * FALLBACK_PLAYER_HP_BAR_ROI[1]) : int(h * FALLBACK_PLAYER_HP_BAR_ROI[3]), int(w * FALLBACK_PLAYER_HP_BAR_ROI[0]) : int(w * FALLBACK_PLAYER_HP_BAR_ROI[2])]

            if mp_rect:
                mp_roi = hsv[int(h * mp_rect[1]) : int(h * mp_rect[3]), int(w * mp_rect[0]) : int(w * mp_rect[2])]
            else:
                mp_roi = hsv[int(h * FALLBACK_PLAYER_MP_BAR_ROI[1]) : int(h * FALLBACK_PLAYER_MP_BAR_ROI[3]), int(w * FALLBACK_PLAYER_MP_BAR_ROI[0]) : int(w * FALLBACK_PLAYER_MP_BAR_ROI[2])]
            hp_red = float(
                np.mean(
                    ((hp_roi[:, :, 0] < 12) | (hp_roi[:, :, 0] > 168))
                    & (hp_roi[:, :, 1] > 120)
                    & (hp_roi[:, :, 2] > 100)
                )
            )
            mp_blue = float(
                np.mean(
                    (mp_roi[:, :, 0] > 85)
                    & (mp_roi[:, :, 0] < 115)
                    & (mp_roi[:, :, 1] > 120)
                    & (mp_roi[:, :, 2] > 100)
                )
            )
            # Bar-slot presence: the dark bar container stays on screen even
            # when the fill is nearly empty, so the veto must not disappear
            # when HP or MP drops below ~18% fill. A bar counts as present
            # when it has fill (>5%) or a visible dark slot (>30% dark px).
            hp_slot = float(np.mean(hp_roi[:, :, 2] <= 45)) if hp_roi.size else 0.0
            mp_slot = float(np.mean(mp_roi[:, :, 2] <= 45)) if mp_roi.size else 0.0
            hp_present = hp_red > 0.05 or hp_slot > 0.30
            mp_present = mp_blue > 0.05 or mp_slot > 0.30
            # Gameplay HUD (bottom HP + MP together) means the world map is closed.
            # A single red map pin must not take this path.
            if hp_present and mp_present:
                return False

            # Grayscale is only needed from here on (chrome + header checks);
            # gameplay frames already returned at step 1.
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 2. Language-independent world-map chrome. A native 1440p client (Turkish
            # header text, vivid terrain) does not match the 720p English crop, and a
            # saturation cap of 98 rejected that colorful canvas. The signature is a dark
            # left list, a brighter map canvas, a lit right edge, and the top-right close
            # control — without the bottom gameplay bars.
            # This cheap ROI-statistics check runs BEFORE the header template
            # pyramid: on a genuinely open map it returns True without any
            # matchTemplate work, and the boolean result is identical either
            # way (both checks only ever return True and have no side effects).
            left_hsv = hsv[int(h * 0.18) : int(h * 0.72), int(w * 0.012) : int(w * 0.16)]
            left_gray = gray[int(h * 0.18) : int(h * 0.72), int(w * 0.012) : int(w * 0.16)]
            if left_hsv.size == 0 or left_gray.size == 0:
                return None
            left_sat = float(np.mean(left_hsv[:, :, 1]))
            left_val = float(np.mean(left_hsv[:, :, 2]))
            left_text_ratio = float(np.mean(left_gray > 145))

            canvas_hsv = hsv[int(h * 0.16) : int(h * 0.78), int(w * 0.22) : int(w * 0.90)]
            if canvas_hsv.size == 0:
                return None
            canvas_val = float(np.mean(canvas_hsv[:, :, 2]))

            right_edge_val = float(
                np.mean(hsv[int(h * 0.16) : int(h * 0.70), int(w * 0.86) : int(w * 0.97), 2])
            )
            x_gray = gray[int(h * 0.012) : int(h * 0.070), int(w * 0.955) : int(w * 0.995)]
            x_bright = float(np.mean(x_gray > 150)) if x_gray.size else 0.0
            top_gray = gray[int(h * 0.012) : int(h * 0.10), int(w * 0.20) : int(w * 0.80)]
            top_mean = float(np.mean(top_gray)) if top_gray.size else 255.0

            if (
                left_sat < 80.0
                and 12.0 <= left_val <= 130.0
                and left_text_ratio > 0.006
                and canvas_val > 78.0
                and canvas_val > left_val + 28.0
                and canvas_val > top_mean + 35.0
                and right_edge_val > 60.0
                and x_bright > 0.004
            ):
                return True

            # 3. Header banner. templates/map_header.png is the 28px minimap pin and is
            # skipped (width < 60). The legacy crop is the 720p English header, so scales
            # must cover a native 2560x1440 UI (about 2.0x–2.67x), not only the video frame.
            # Resized pyramids come from the per-template cache
            # (`_get_map_header_scales`) instead of being re-resized per call.
            for hdr_tmpl in (self.map_header_legacy_template, self.map_header_template):
                if hdr_tmpl is None or hdr_tmpl.shape[1] < 60:
                    continue
                y1, y2 = int(h * 0.00), max(int(h * 0.16), int(h * 0.00) + 8)
                x1, x2 = int(w * 0.12), max(int(w * 0.88), int(w * 0.12) + 8)
                top_strip = gray[y1:y2, x1:x2]
                if top_strip.size == 0:
                    continue
                for _s, rt in self._get_map_header_scales(hdr_tmpl):
                    if rt.shape[0] < top_strip.shape[0] and rt.shape[1] < top_strip.shape[1]:
                        res = cv2.matchTemplate(top_strip, rt, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, _ = cv2.minMaxLoc(res)
                        if float(mv) >= 0.68:
                            return True

            # Overhead HP bar on a bottom-cropped gameplay shot (aria_3d_crest_far.jpg).
            # Checked after chrome so red markers on an open map are not a veto.
            # F8 (v18): esikler MultiCriteriaMapVerifier._is_world_3d_hud_active
            # ile ayni (sat>140, val>120, oran>0.008). Eski siki esik
            # (sat>160, val>140, oran>0.025) bu karedeki ust HP barini
            # (olculen oran 0.0118) iskaliyor; alt HP/MP barlari kirpilmis
            # karede bu tek oyun kanitiydi ve P1-3 sonrasi dusus None
            # (bilinmiyor) donup #14 'vis_open is False' kosulunu kiriyordu.
            char_hp_roi = hsv[int(h * 0.40) : int(h * 0.60), int(w * 0.44) : int(w * 0.56)]
            if char_hp_roi.size > 0:
                char_hp_red = float(
                    np.mean(
                        ((char_hp_roi[:, :, 0] < 10) | (char_hp_roi[:, :, 0] > 170))
                        & (char_hp_roi[:, :, 1] > 140)
                        & (char_hp_roi[:, :, 2] > 120)
                    )
                )
                if char_hp_red > 0.008:
                    return False
            # P1-3 (K-07): buraya kadar ne harita-açık ne oyun-ekranı kanıtı
            # bulundu — kanıtsızlıkta False ("harita kapalı") dönmek
            # fail-open'dı: yükleme/karartı/menü karesinde bot harita
            # açıkken bile toplama arayıp F/E basıyordu. Kanıt yoksa
            # bilinmiyor (None) döner; çağıranlar None'u "açık olabilir"
            # sayar (bu dosyadaki iç çağıranlar `is not False` ile).
            return None
        except Exception as exc:
            logger.error("is_map_open failed: %s", exc)
            # Sweep G: hata = bilinmiyor (None), "harita kapalı" değil.
            return None

    # ------------------------------------------------------------------
    # 3. Blocking Game Window Detection (Cube / Menu / Dialogs)
    # ------------------------------------------------------------------
    def is_game_window_blocking(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Optional[bool]:
        """
        Checks whether an in-game modal or menu panel (Inventory/Cube, Character,
        Quest, Settings dialog, or the right-hand Escape menu as in `frame_03s.jpg`)
        is currently open and blocking normal gameplay.

        Sweep G: `is_map_open` gibi üç durumlu — `None` bilinmiyor demek
        (bozuk kare / hata); eskiden hata `False` ("engel yok") dönüyordu.
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return None

            valid, _ = self.capture.validate_frame(img_bgr)
            if not valid:
                return None

            h, w = img_bgr.shape[:2]
            if h < 380 or w < 500:
                return None

            map_status = self.is_map_open(img_bgr)
            if map_status is True:
                return False
            if map_status is None:
                # Harita durumu bilinmiyorsa engel durumu da bilinmiyor.
                return None

            # Envanter minimap'i kapatmadan ve ekranı gece kadar karartarak açılır.
            # Parlak-minimap ve "mean < 22 = gece" kapıları bu paneli kaçırır.
            if detect_inventory_panel(img_bgr):
                return True

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            mean_brightness = float(np.mean(gray))
            if mean_brightness < 22.0:
                # Dark screen / night / loading screen is not a blocking menu
                return False

            # 1. Top-right Minimap visibility (x=0.86..0.98, y=0.09..0.24)
            mmap_roi = hsv[int(h * 0.09) : int(h * 0.24), int(w * 0.86) : int(w * 0.98)]
            mmap_val = float(np.mean(mmap_roi[:, :, 2]))

            # If minimap is visibly active and has normal brightness, menus are closed
            if mmap_val >= 55.0:
                return False

            # 2. Right-hand menu panel area (x=0.71..0.96, y=0.01..0.70)
            right_panel = hsv[int(h * 0.01) : int(h * 0.70), int(w * 0.71) : int(w * 0.96)]
            r_sat = float(np.mean(right_panel[:, :, 1]))
            r_val = float(np.mean(right_panel[:, :, 2]))

            # 3. Top-right close 'X' icon (x=0.965..0.994, y=0.018..0.062)
            x_gray = gray[int(h * 0.018) : int(h * 0.062), int(w * 0.965) : int(w * 0.994)]
            x_bright = float(np.mean(x_gray > 190))

            # Center modal dialog check (e.g. inventory/cube or system prompt dialog)
            center_dialog = gray[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
            center_edges = cv2.Canny(center_dialog, 60, 180)
            center_edge_density = float(np.mean(center_edges > 0))

            # Menus must show strict characteristic dark modal backdrop & high X icon intensity
            right_menu_open = (
                mmap_val < 38.0
                and r_sat < 30.0
                and r_val < 35.0
                and x_bright > 0.15
            )

            center_modal_open = (
                center_edge_density > 0.18
                and x_bright > 0.16
                and mmap_val < 35.0
            )

            return bool(right_menu_open or center_modal_open)
        except Exception as exc:
            logger.error("is_game_window_blocking failed: %s", exc)
            return None

    def is_inventory_open(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> bool:
        """Çanta/ekipman ızgarası görünür mü? Açıkken ikonlar materyal kaydı değildir."""
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            return bool(detect_inventory_panel(img_bgr))
        except Exception as exc:
            logger.debug("is_inventory_open hatası: %s", exc)
            return False

    def check_blocking_window(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Optional[bool]:
        """Alias for `is_game_window_blocking` (three-state: True/False/None)."""
        return self.is_game_window_blocking(frame=frame)

    # ------------------------------------------------------------------
    # 4. Overlay Panel Collision Check
    # ------------------------------------------------------------------
    def is_overlay_covering_point(
        self,
        x: int,
        y: int,
        overlay_rect: Union[Tuple[int, int, int, int], Dict[str, int], Any],
        node_id: Optional[Union[int, str]] = None,
    ) -> Tuple[bool, str]:
        """
        Checks whether map target coordinate `(x, y)` lies underneath the top-left
        Overlay HUD panel (`overlay_rect = (ox, oy, ow, oh)`).

        Returns:
            `(is_covered: bool, warning_message: str)`
            When covered, returns `(True, f"Node {node_id} is under the overlay panel")`.
            On failure returns `(False, "error: ...")` — never a clean "OK",
            so callers can tell an error apart from a verified clear point.
        """
        try:
            if overlay_rect is None:
                return False, "OK"

            px, py = int(round(x)), int(round(y))
            ox, oy, ow, oh = 0, 0, 0, 0

            if isinstance(overlay_rect, (tuple, list)) and len(overlay_rect) >= 4:
                ox, oy, ow, oh = (
                    int(overlay_rect[0]),
                    int(overlay_rect[1]),
                    int(overlay_rect[2]),
                    int(overlay_rect[3]),
                )
            elif isinstance(overlay_rect, dict):
                ox = int(overlay_rect.get("left", overlay_rect.get("x", 0)))
                oy = int(overlay_rect.get("top", overlay_rect.get("y", 0)))
                ow = int(overlay_rect.get("width", overlay_rect.get("w", 0)))
                oh = int(overlay_rect.get("height", overlay_rect.get("h", 0)))
            elif all(hasattr(overlay_rect, attr) for attr in ("x", "y", "width", "height")):
                ox = int(overlay_rect.x() if callable(overlay_rect.x) else overlay_rect.x)
                oy = int(overlay_rect.y() if callable(overlay_rect.y) else overlay_rect.y)
                ow = int(overlay_rect.width() if callable(overlay_rect.width) else overlay_rect.width)
                oh = int(overlay_rect.height() if callable(overlay_rect.height) else overlay_rect.height)
            else:
                return False, f"invalid overlay_rect: {type(overlay_rect).__name__}"

            if ow <= 0 or oh <= 0:
                return False, "invalid overlay_rect: non-positive size"

            inside = (ox <= px <= ox + ow) and (oy <= py <= oy + oh)
            if inside:
                label = node_id if node_id is not None else f"({px}, {py})"
                msg = f"Node {label} is under the overlay panel"
                logger.warning(msg)
                return True, msg

            return False, "OK"
        except Exception as exc:
            logger.error("is_overlay_covering_point failed: %s", exc)
            return False, f"error: {exc}"

    # ------------------------------------------------------------------
    # 5. Center Map Patch Extraction (80x80 with Arrow Masking)
    # ------------------------------------------------------------------
    def extract_center_map_patch(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        patch_size: int = 80,
        mask_player_arrow: bool = True,
    ) -> np.ndarray:
        """
        Extracts a `patch_size x patch_size` (default `80x80`) grayscale map patch
        around the center of the open map, masking and Telea-inpainting the directional
        player arrow so subsequent template matching is unaffected by player heading.
        Raises RuntimeError when no usable frame is available: a zero patch
        would silently match nothing (or everything) downstream.
        """
        size = max(16, int(patch_size))
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                raise RuntimeError("extract_center_map_patch: no frame available")

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr.copy()
            )
            h, w = img_gray.shape[:2]
            player_pos = self.find_player_on_map(img_bgr)
            if player_pos is None:
                # Sweep G: ok bulunamadı — ekran merkezi gerçek konummuş
                # gibi yama kesilmez; RuntimeError bu fonksiyonun sözleşmesi.
                raise RuntimeError("extract_center_map_patch: player arrow not found on map")
            cx, cy = player_pos
            half = size // 2

            cx = int(np.clip(cx, half, max(half, w - half)))
            cy = int(np.clip(cy, half, max(half, h - half)))

            y1, y2 = cy - half, cy + half
            x1, x2 = cx - half, cx + half
            patch_gray = img_gray[y1:y2, x1:x2].copy()

            if patch_gray.shape[0] != size or patch_gray.shape[1] != size:
                patch_gray = cv2.resize(patch_gray, (size, size), interpolation=cv2.INTER_AREA)

            if not mask_player_arrow:
                return np.asarray(patch_gray, dtype=np.uint8)

            # Build player arrow mask around patch center
            mask = np.zeros((size, size), dtype=np.uint8)
            core_radius = max(8, int(round(size * 0.14)))
            cv2.circle(mask, (half, half), core_radius, 255, -1)

            if img_bgr.ndim == 3:
                patch_bgr = img_bgr[y1:y2, x1:x2]
                if patch_bgr.shape[0] != size or patch_bgr.shape[1] != size:
                    patch_bgr = cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                outer_zone = np.zeros((size, size), dtype=np.uint8)
                outer_radius = max(12, int(round(size * 0.24)))
                cv2.circle(outer_zone, (half, half), outer_radius, 255, -1)
                arrow_px = (
                    ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 130))
                    | (hsv[:, :, 2] > 215)
                ).astype(np.uint8) * 255
                arrow_px = cv2.dilate(arrow_px, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(arrow_px, outer_zone))

            cleaned = cv2.inpaint(patch_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return np.asarray(cleaned, dtype=np.uint8)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("extract_center_map_patch failed: %s", exc)
            raise RuntimeError(f"extract_center_map_patch failed: {exc}") from exc

    def find_player_on_map(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Locates the player arrow near the center of the open world map using
        `templates/player_arrow.png` or cyan/white arrow color segmentation.

        Sweep G: ok bulunamazsa `None` döner — eskiden ekran merkezi
        `(w // 2, h // 2)` gerçek konummuş gibi dönüyor, çağıranlar da
        (örn. `extract_center_map_patch`) bu uydurma merkezi sorgusuz
        kullanıyordu. Dış çağıranların tamamı `isinstance(pos, tuple)`
        ile korunduğundan `None` onlarda kendi yedeklerine düşer.
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return None

            h, w = img_bgr.shape[:2]
            cx, cy = w // 2, h // 2
            r = max(36, int(round(min(w, h) * 0.10)))
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)

            roi_bgr = img_bgr[y1:y2, x1:x2]
            if roi_bgr.size > 0:
                roi_gray = (
                    cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    if roi_bgr.ndim == 3
                    else roi_bgr
                )
                for pa_tmpl in (self.player_arrow_template, self.player_arrow_legacy_template):
                    if pa_tmpl is None:
                        continue
                    th, tw = pa_tmpl.shape[:2]
                    if th < roi_gray.shape[0] and tw < roi_gray.shape[1]:
                        res = cv2.matchTemplate(roi_gray, pa_tmpl, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, ml = cv2.minMaxLoc(res)
                        if float(mv) >= 0.72:
                            px = x1 + int(ml[0]) + tw // 2
                            py = y1 + int(ml[1]) + th // 2
                            return px, py

            if roi_bgr.ndim == 3 and roi_bgr.size > 0:
                hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                cyan_mask = (
                    (hsv[:, :, 0] >= 88)
                    & (hsv[:, :, 0] <= 132)
                    & (hsv[:, :, 1] > 95)
                    & (hsv[:, :, 2] > 145)
                )
                ys, xs = np.where(cyan_mask)
                if len(xs) >= 8:
                    return int(round(x1 + float(np.mean(xs)))), int(round(y1 + float(np.mean(ys))))
            # No arrow found by template or colour: report "not found"
            # instead of silently pretending the screen centre is the player.
            logger.warning(
                "find_player_on_map: player arrow not detected; returning None (no fallback to screen centre)",
            )
            return None
        except Exception as exc:
            logger.warning("find_player_on_map failed, returning None: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 6. Target HP Bar Reading
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_target_bar_hsv(hsv_crop: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Detects red, orange, yellow, olive, and magenta/pink hostile target HP bar pixels.
        Returns (combined_mask, dominant_color_name).
        """
        if hsv_crop is None or hsv_crop.size == 0:
            return np.zeros((0, 0), dtype=np.uint8), "none"

        # Red: Crimson/Ruby hostile HP bar (lower & upper wrap)
        red1 = (hsv_crop[:, :, 0] <= 12) & (hsv_crop[:, :, 1] >= 75) & (hsv_crop[:, :, 2] >= 70)
        red2 = (hsv_crop[:, :, 0] >= 165) & (hsv_crop[:, :, 1] >= 75) & (hsv_crop[:, :, 2] >= 70)
        mask_red = red1 | red2

        # Orange: Amber/Tangerine hostile/damaged HP bar
        mask_orange = (
            (hsv_crop[:, :, 0] >= 13)
            & (hsv_crop[:, :, 0] <= 25)
            & (hsv_crop[:, :, 1] >= 75)
            & (hsv_crop[:, :, 2] >= 80)
        )

        # Yellow: Gold/Yellow hostile or neutral-aggro HP bar
        mask_yellow = (
            (hsv_crop[:, :, 0] >= 26)
            & (hsv_crop[:, :, 0] <= 40)
            & (hsv_crop[:, :, 1] >= 70)
            & (hsv_crop[:, :, 2] >= 80)
        )

        # Olive: Olive/Yellow-green HP bar
        mask_olive = (
            (hsv_crop[:, :, 0] >= 41)
            & (hsv_crop[:, :, 0] <= 65)
            & (hsv_crop[:, :, 1] >= 65)
            & (hsv_crop[:, :, 2] >= 80)
        )

        # Magenta/Pink: Special boss or shield HP bar
        mask_magenta = (
            (hsv_crop[:, :, 0] >= 140)
            & (hsv_crop[:, :, 0] <= 164)
            & (hsv_crop[:, :, 1] >= 60)
            & (hsv_crop[:, :, 2] >= 90)
        )

        combined = (mask_red | mask_orange | mask_yellow | mask_olive | mask_magenta).astype(np.uint8)

        # Determine dominant color
        counts = {
            "red": int(np.count_nonzero(mask_red)),
            "orange": int(np.count_nonzero(mask_orange)),
            "yellow": int(np.count_nonzero(mask_yellow)),
            "olive": int(np.count_nonzero(mask_olive)),
            "magenta": int(np.count_nonzero(mask_magenta)),
        }
        best_col, best_count = max(counts.items(), key=lambda item: item[1])
        dom_color = best_col if best_count >= 10 else "none"
        return combined, dom_color

    def read_target_hp_bar(self, frame_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Accurately reads the target monster's health bar percentage [0.0..100.0] and targeting status.
        Eliminates false 'HEDEF YOK' dropouts by inspecting:
        1. Multi-gamut hostile fill (Red, Orange, Yellow, Olive, Magenta).
        2. HP bar track/slot container bounds.
        3. Cyan / light-blue diamond bracket tips `< Monster Name >`.
        4. Target frame header text and agro badge indicators.
        5. Dynamically respects calibrated 'target_hp_bar' ROI from config/calibrated_rois.json.
        """
        img = self._coerce_frame(frame_bgr, grayscale=False)
        default_res = {
            "has_target": False,
            "target_hp_pct": 0.0,
            "target_pct": 0.0,
            "is_hostile": False,
            "is_monster_target": False,
            "bar_bbox": (0, 0, 0, 0),
            "confidence": 0.0,
            "bar_color": "none",
            "has_diamond_brackets": False,
            "nameplate_detected": False,
            # Sweep G: target_hp_pct=0.0 artık üç anlama gelmiyor.
            # read_ok=False -> okunamadı (bozuk kare / hata / dejeneratif ROI).
            # read_ok=True + has_target=False -> hedef yok.
            # read_ok=True + has_target=True + pct 0.0 -> hedef %0 canda.
            "read_ok": False,
            "read_error": None,
        }
        if img is None or img.size == 0:
            return default_res

        try:
            h, w = img.shape[:2]
            scale_x = w / 2000.0
            scale_y = h / 1124.0

            # Dynamic calibrated ROI support with auto-reload
            target_rect = None
            try:
                from dataset_factory.exemplar_memory import get_calibrated_rois
                cm = get_calibrated_rois()
                cm.check_reload()
                cand_rect = cm.get_normalized_rect("target_hp_bar") or cm.get_normalized_rect("target_bar")
                # Ensure the rect is in the upper HUD region (y1 < 0.25)
                if cand_rect and cand_rect[1] < 0.25:
                    target_rect = cand_rect
            except Exception as exc:
                logger.debug("target_hp_bar calibrated ROI load: %s", exc)

            if target_rect is None:
                # Sweep G: tek yedek ROI (FALLBACK_TARGET_HP_BAR_ROI).
                # Eskiden burası (0.300, 0.015, 0.700, 0.120) iken
                # skill_bar_reader / mana_phase_combat (0.350, 0.030,
                # 0.650, 0.075) kullanıyordu — aynı bar, iki farklı bölge.
                target_rect = FALLBACK_TARGET_HP_BAR_ROI

            tx1 = max(0, int(round(w * target_rect[0])))
            ty1 = max(0, int(round(h * target_rect[1])))
            tx2 = min(w, int(round(w * target_rect[2])))
            ty2 = min(h, int(round(h * target_rect[3])))

            if tx2 <= tx1 or ty2 <= ty1:
                return default_res

            target_crop = img[ty1:ty2, tx1:tx2]
            hsv = cv2.cvtColor(target_crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)

            # 1. Hostile color mask & dominant color
            color_mask, dom_color = self._detect_target_bar_hsv(hsv)

            # 2. Cyan diamond brackets tips `< ... >` (left tip `<` and right tip `>`)
            cyan_mask = (
                (hsv[:, :, 0] >= 78)
                & (hsv[:, :, 0] <= 108)
                & (hsv[:, :, 1] >= 70)
                & (hsv[:, :, 2] >= 125)
            ).astype(np.uint8)
            cyan_cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_brackets = []
            min_bracket_w = max(3, int(round(4 * scale_x)))
            max_bracket_w = max(18, int(round(26 * scale_x)))
            min_bracket_h = max(4, int(round(5 * scale_y)))
            max_bracket_h = max(24, int(round(30 * scale_y)))
            for cc in cyan_cnts:
                cx, cy, cw_b, ch_b = cv2.boundingRect(cc)
                if min_bracket_w <= cw_b <= max_bracket_w and min_bracket_h <= ch_b <= max_bracket_h:
                    valid_brackets.append((cx, cy, cw_b, ch_b))

            has_diamond_brackets = False
            if len(valid_brackets) >= 2:
                valid_brackets.sort(key=lambda b: b[0])
                for i in range(len(valid_brackets) - 1):
                    for j in range(i + 1, len(valid_brackets)):
                        b1 = valid_brackets[i]
                        b2 = valid_brackets[j]
                        dx = (b2[0] + b2[2] // 2) - (b1[0] + b1[2] // 2)
                        dy = abs((b2[1] + b2[3] // 2) - (b1[1] + b1[3] // 2))
                        if dy <= max(8, int(round(10 * scale_y))) and max(40, int(round(60 * scale_x))) <= dx <= max(350, int(round(500 * scale_x))):
                            has_diamond_brackets = True
                            break
                    if has_diamond_brackets:
                        break

            # 3. Target nameplate high-contrast text detection inside the crop
            bright_text = (hsv[:, :, 1] <= 85) & (hsv[:, :, 2] >= 165)
            text_px = int(np.count_nonzero(bright_text))
            nameplate_detected = text_px >= max(20, int(round(35 * scale_x * scale_y)))

            # 4. Empty bar container slot: dark gray / dark red slot
            dark_slot = (
                (hsv[:, :, 2] <= 45)
                | (((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] <= 85))
            ).astype(np.uint8)

            # Combined bar strip (color fill + dark empty slot)
            combined_bar = cv2.bitwise_or(color_mask, dark_slot)
            kernel_comb = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(round(12 * scale_x))), 2))
            closed_combined = cv2.morphologyEx(combined_bar, cv2.MORPH_CLOSE, kernel_comb)

            # Standalone color fill
            kernel_col = cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, int(round(8 * scale_x))), 2))
            closed_color = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_col)

            # Find contours in combined bar
            cnts_comb, _ = cv2.findContours(closed_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_color, _ = cv2.findContours(closed_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_bar_w = max(24, int(round(40 * scale_x)))
            min_bar_h = 3
            max_bar_h = max(18, int(round(36 * scale_y)))

            best_container = None
            for c in cnts_comb:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(max(1, bh))
                if bw >= min_bar_w and min_bar_h <= bh <= max_bar_h and aspect >= 2.4:
                    sub_col = color_mask[by : by + bh, bx : bx + bw]
                    col_px = int(np.count_nonzero(sub_col))
                    # Accept if it contains color fill OR confirmed diamond brackets
                    if col_px >= max(4, int(round(8 * scale_x))) or has_diamond_brackets:
                        if best_container is None or bw > best_container[2]:
                            best_container = (bx, by, bw, bh)

            # Also check standalone color contours
            best_color_bar = None
            for c in cnts_color:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(max(1, bh))
                if bw >= max(14, int(round(20 * scale_x))) and min_bar_h <= bh <= max_bar_h and aspect >= 2.0:
                    if best_color_bar is None or bw > best_color_bar[2]:
                        best_color_bar = (bx, by, bw, bh)

            has_target = False
            target_hp_pct = 0.0
            chosen_bbox = (0, 0, 0, 0)
            conf = 0.0

            if best_container is not None:
                bx, by, bw, bh = best_container
                has_target = True
                chosen_bbox = (tx1 + bx, ty1 + by, bw, bh)
                sub_col = color_mask[by : by + bh, bx : bx + bw]
                col_sums = np.mean(sub_col, axis=0) > 0.15
                fill_w = int(np.sum(col_sums))
                target_hp_pct = round(min(100.0, max(0.0, (fill_w / float(max(1, bw))) * 100.0)), 1)
                conf = 0.94 if has_diamond_brackets else 0.88
            elif best_color_bar is not None:
                bx, by, bw, bh = best_color_bar
                has_target = True
                chosen_bbox = (tx1 + bx, ty1 + by, bw, bh)
                sub_col = color_mask[by : by + bh, bx : bx + bw]
                col_sums = np.mean(sub_col, axis=0) > 0.15
                fill_w = int(np.sum(col_sums))
                # Report the measured fill. A wide bar with <85% fill is a
                # damaged target, not a full-health one — never overwrite the
                # measurement with 100%.
                target_hp_pct = round(min(100.0, max(0.0, (fill_w / float(max(1, bw))) * 100.0)), 1)
                conf = 0.86
            elif has_diamond_brackets or (nameplate_detected and int(np.count_nonzero(color_mask)) >= 5):
                # Target locked but HP is near 0% or empty
                has_target = True
                target_hp_pct = 0.0
                conf = 0.90 if has_diamond_brackets else 0.75
                chosen_bbox = (tx1 + target_crop.shape[1] // 4, ty1 + 10, target_crop.shape[1] // 2, 16)

            is_hostile = has_target and (dom_color in ("red", "orange", "yellow", "olive", "magenta") or has_diamond_brackets)

            return {
                "has_target": has_target,
                "target_hp_pct": target_hp_pct,
                "target_pct": target_hp_pct,
                "is_hostile": is_hostile,
                "is_monster_target": is_hostile,
                "bar_bbox": chosen_bbox,
                "confidence": round(conf, 3),
                "bar_color": dom_color,
                "has_diamond_brackets": has_diamond_brackets,
                "nameplate_detected": nameplate_detected,
                "read_ok": True,
                "read_error": None,
            }
        except Exception as exc:
            logger.error("read_target_hp_bar failed: %s", exc)
            failed_res = dict(default_res)
            failed_res["read_error"] = str(exc)
            return failed_res

    def detect_target_brackets(
        self, frame_bgr: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Detects cyan diamond bracket tips `< ... >` indicative of a targeted monster.
        Returns only blobs that form a left+right pair (similar y, plausible
        separation): a single unmatched cyan blob is not a bracket pair and
        is dropped instead of being reported as a target.
        Returns list of bracket detections with coordinates and center positions.
        """
        img = self._coerce_frame(frame_bgr, grayscale=False)
        if img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        scale_x = w / 2000.0
        scale_y = h / 1124.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cyan_mask = (
            (hsv[:, :, 0] >= 78)
            & (hsv[:, :, 0] <= 108)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 125)
        ).astype(np.uint8)

        cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brackets = []
        min_bw = max(3, int(round(4 * scale_x)))
        max_bw = max(22, int(round(30 * scale_x)))
        min_bh = max(4, int(round(5 * scale_y)))
        max_bh = max(26, int(round(34 * scale_y)))

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if min_bw <= bw <= max_bw and min_bh <= bh <= max_bh:
                brackets.append({
                    "x": bx,
                    "y": by,
                    "w": bw,
                    "h": bh,
                    "center_x": bx + bw // 2,
                    "center_y": by + bh // 2,
                })

        # Pairing filter: keep only blobs with a partner on the same row at
        # bracket distance, mirroring read_target_hp_bar's pairing rule.
        min_dx = max(40, int(round(60 * scale_x)))
        max_dx = max(350, int(round(500 * scale_x)))
        max_dy = max(8, int(round(10 * scale_y)))
        paired: List[Dict[str, Any]] = []
        for i, b1 in enumerate(brackets):
            for j, b2 in enumerate(brackets):
                if i == j:
                    continue
                dx = abs(b2["center_x"] - b1["center_x"])
                dy = abs(b2["center_y"] - b1["center_y"])
                if min_dx <= dx <= max_dx and dy <= max_dy:
                    paired.append(b1)
                    break
        return paired

    def has_active_target(self, frame_bgr: Optional[np.ndarray] = None) -> bool:
        """Returns True if a target monster is currently selected/locked."""
        res = self.read_target_hp_bar(frame_bgr)
        return bool(res.get("has_target", False))

    def get_target_hp_percentage(self, frame_bgr: Optional[np.ndarray] = None) -> Optional[float]:
        """Returns the target monster's health percentage [0.0..100.0].

        Sweep G: üç anlam artık ayrık — `None` hedef yok VEYA okuma
        başarısız demek (ayrım için `read_target_hp_bar()` içindeki
        `has_target` / `read_ok` alanlarına bakılır); `0.0` yalnızca
        hedef varken ve gerçekten %0 candayken döner.
        """
        res = self.read_target_hp_bar(frame_bgr)
        if not res.get("read_ok", False) or not res.get("has_target", False):
            return None
        return float(res.get("target_hp_pct", 0.0))

    def read_player_hp_mp_bars(self, frame_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Dynamically reads player HP and MP fill percentages [0.0..100.0] and target HP percentage
        using config/calibrated_rois.json when calibrated, auto-reloaded on file modification.
        Eliminates false 'HEDEF YOK' targeting dropouts by using robust multi-gamut target bar reading.
        """
        img = self._coerce_frame(frame_bgr, grayscale=False)
        if img is None or img.size == 0:
            # Unreadable frame: report unknown, never a fake healthy 100%.
            return {
                "hp_pct": None,
                "mp_pct": None,
                "target_hp_pct": None,
                "target_pct": None,
                "has_target": False,
                "is_monster_target": False,
                "is_hostile": False,
                "bar_color": "none",
                "has_diamond_brackets": False,
                "hp_valid": False,
                "mp_valid": False,
                "valid": False,
            }

        try:
            h, w = img.shape[:2]
            from dataset_factory.exemplar_memory import get_calibrated_rois
            cm = get_calibrated_rois()
            cm.check_reload()

            hp_rect = cm.get_normalized_rect("player_hp_bar") or FALLBACK_PLAYER_HP_BAR_ROI
            mp_rect = cm.get_normalized_rect("player_mp_bar") or FALLBACK_PLAYER_MP_BAR_ROI

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Player HP (red fill)
            hy1, hy2 = int(h * hp_rect[1]), int(h * hp_rect[3])
            hx1, hx2 = int(w * hp_rect[0]), int(w * hp_rect[2])
            hp_crop = hsv[hy1:hy2, hx1:hx2]
            if hp_crop.size > 0:
                red_mask = ((hp_crop[:, :, 0] < 12) | (hp_crop[:, :, 0] > 168)) & (hp_crop[:, :, 1] > 110) & (hp_crop[:, :, 2] > 90)
                col_fill = np.mean(red_mask, axis=0) > 0.25
                hp_ratio = float(np.sum(col_fill)) / float(max(1, len(col_fill)))
                hp_pct: Optional[float] = round(hp_ratio * 100.0, 1)
                hp_valid = True
            else:
                # Empty ROI: no measurement — report unknown, not 100%.
                hp_pct = None
                hp_valid = False

            # Player MP (cyan/blue fill)
            my1, my2 = int(h * mp_rect[1]), int(h * mp_rect[3])
            mx1, mx2 = int(w * mp_rect[0]), int(w * mp_rect[2])
            mp_crop = hsv[my1:my2, mx1:mx2]
            if mp_crop.size > 0:
                blue_mask = (mp_crop[:, :, 0] > 85) & (mp_crop[:, :, 0] < 135) & (mp_crop[:, :, 1] > 100) & (mp_crop[:, :, 2] > 80)
                col_fill_mp = np.mean(blue_mask, axis=0) > 0.25
                mp_ratio = float(np.sum(col_fill_mp)) / float(max(1, len(col_fill_mp)))
                mp_pct: Optional[float] = round(mp_ratio * 100.0, 1)
                mp_valid = True
            else:
                mp_pct = None
                mp_valid = False

            # Target HP bar read via robust multi-gamut detector
            target_info = self.read_target_hp_bar(img)

            return {
                "hp_pct": hp_pct,
                "mp_pct": mp_pct,
                # Sweep G: hedef yoksa / okunamadıysa None — 0.0 yalnızca
                # hedef varken %0 can demek (bkz. read_target_hp_bar).
                "target_hp_pct": target_info["target_hp_pct"] if (target_info.get("read_ok") and target_info.get("has_target")) else None,
                "target_pct": target_info["target_hp_pct"] if (target_info.get("read_ok") and target_info.get("has_target")) else None,
                "has_target": target_info["has_target"],
                "target_read_ok": bool(target_info.get("read_ok", False)),
                "is_monster_target": target_info["is_monster_target"],
                "is_hostile": target_info["is_hostile"],
                "bar_color": target_info["bar_color"],
                "has_diamond_brackets": target_info["has_diamond_brackets"],
                "target_bbox": target_info["bar_bbox"],
                "hp_valid": hp_valid,
                "mp_valid": mp_valid,
                "valid": bool(hp_valid and mp_valid),
            }
        except Exception as exc:
            logger.error("read_player_hp_mp_bars failed: %s", exc)
            return {
                "hp_pct": None,
                "mp_pct": None,
                "target_hp_pct": None,
                "target_pct": None,
                "has_target": False,
                "target_read_ok": False,
                "is_monster_target": False,
                "is_hostile": False,
                "bar_color": "none",
                "has_diamond_brackets": False,
                "hp_valid": False,
                "mp_valid": False,
                "valid": False,
            }

    def detect_monsters_3d(
        self,
        frame_bgr: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detects 3D monsters in the game viewport and validates/refines them against ExemplarMemory:
        - Detects hostile red/orange/yellow monster HP bars and nameplates (aspect >= 2.5).
        - Detects targeted monsters with cyan diamond bracket tips `< Monster Name >` (boosts conf to 0.95).
        - Detects yellow/olive monster nameplates in the 3D world.
        - Excludes player's own overhead HP bar (center of screen: x: 0.44..0.56, y: 0.42..0.58).
        - Rejects candidates matching negative exemplars in ExemplarMemory.
        - Elevates confidence for candidates matching 10: monster_3d.
        - Also detects top HUD target HP bar as class_id 18: target_hp_bar.
        """
        img = self._coerce_frame(frame_bgr, grayscale=False)
        if img is None or img.size == 0:
            return []

        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            mem = get_exemplar_memory()
        except Exception:
            mem = None

        h, w = img.shape[:2]
        scale_x = w / 2000.0
        scale_y = h / 1124.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        monsters: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 1. Top HUD target HP bar check (class_id = 18: target_hp_bar)
        # ------------------------------------------------------------------
        tgt_info = self.read_target_hp_bar(img)
        if tgt_info.get("has_target") and tgt_info.get("bar_bbox", (0, 0, 0, 0))[2] > 0:
            bx, by, bw, bh = tgt_info["bar_bbox"]
            monsters.append({
                "screen_x": bx + bw // 2,
                "screen_y": by + bh // 2,
                "bbox": (bx, by, bw, bh),
                "confidence": float(tgt_info.get("confidence", 0.92)),
                "class_id": 18,
                "class_name": "target_hp_bar",
                "has_diamond_brackets": bool(tgt_info.get("has_diamond_brackets", False)),
                "target_hp_pct": float(tgt_info.get("target_hp_pct") or 0.0),
                "is_hostile": bool(tgt_info.get("is_hostile", True)),
                "is_targeted": True,
            })

        # ------------------------------------------------------------------
        # 2. 3D Viewport Search Area (0.12*H..0.78*H, 0.10*W..0.88*W)
        # ------------------------------------------------------------------
        vy1, vy2 = int(h * 0.12), int(h * 0.78)
        vx1, vx2 = int(w * 0.10), int(w * 0.88)
        viewport_hsv = hsv[vy1:vy2, vx1:vx2]

        # Player center self-exclusion zone in 3D viewport (x: 0.43..0.57, y: 0.38..0.64 normalized)
        # Eliminates false 3D monster detections originating from player model/cape/mount
        pex_x1, pex_y1, pex_x2, pex_y2 = self.player_center_exclusion_zone
        center_y1 = max(0, int(h * pex_y1) - vy1)
        center_y2 = min(vy2 - vy1, int(h * pex_y2) - vy1)
        center_x1 = max(0, int(w * pex_x1) - vx1)
        center_x2 = min(vx2 - vx1, int(w * pex_x2) - vx1)

        valid_vp_mask = np.ones((vy2 - vy1, vx2 - vx1), dtype=bool)
        if center_y2 > center_y1 and center_x2 > center_x1:
            valid_vp_mask[center_y1:center_y2, center_x1:center_x2] = False

        # Hostile red, orange, yellow nameplate/bar in 3D viewport
        hsv_h = viewport_hsv[:, :, 0]
        hsv_s = viewport_hsv[:, :, 1]
        hsv_v = viewport_hsv[:, :, 2]

        red_vp = ((hsv_h <= 12) | (hsv_h >= 165)) & (hsv_s >= 115) & (hsv_v >= 105)
        orange_vp = (hsv_h >= 13) & (hsv_h <= 25) & (hsv_s >= 115) & (hsv_v >= 110)
        yellow_vp = (hsv_h >= 26) & (hsv_h <= 34) & (hsv_s >= 105) & (hsv_v >= 115)

        hostile_vp = (red_vp | orange_vp | yellow_vp) & valid_vp_mask

        # Cyan diamond bracket tips in 3D viewport
        cyan_vp = (hsv_h >= 78) & (hsv_h <= 108) & (hsv_s >= 70) & (hsv_v >= 125) & valid_vp_mask
        cyan_cnts, _ = cv2.findContours(cyan_vp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vp_brackets = []
        for cc in cyan_cnts:
            cx, cy, cw_b, ch_b = cv2.boundingRect(cc)
            if 3 <= cw_b <= max(20, int(round(28 * scale_x))) and 4 <= ch_b <= max(24, int(round(32 * scale_y))):
                vp_brackets.append((cx, cy, cw_b, ch_b))

        # Horizontal morphology close to bridge nameplate letters and HP fill
        kw = max(10, int(round(14 * scale_x)))
        kh = max(2, int(round(3 * scale_y)))
        kernel_vp = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        closed_vp = cv2.morphologyEx(hostile_vp.astype(np.uint8), cv2.MORPH_CLOSE, kernel_vp)

        contours, _ = cv2.findContours(closed_vp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = max(25, int(round(35 * scale_x)))
        max_w = max(160, int(round(320 * scale_x)))
        min_h = 3
        max_h = max(22, int(round(32 * scale_y)))

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(max(1, bh))
            if not (min_w <= bw <= max_w and min_h <= bh <= max_h and aspect >= 2.5):
                continue

            abs_x = vx1 + bx
            abs_y = vy1 + by
            cand_cx = abs_x + bw // 2
            cand_cy = abs_y + bh // 2

            # Reject false 3D monster detections originating from player model/cape/mount
            if self.is_in_player_exclusion_zone(cand_cx, cand_cy, w, h, bbox=(abs_x, abs_y, bw, bh)):
                continue

            # Check if this candidate is flanked by cyan brackets
            has_brackets = False
            for b in vp_brackets:
                b_abs_x = vx1 + b[0]
                b_abs_y = vy1 + b[1]
                if abs(b_abs_y - abs_y) <= 12 and (abs(b_abs_x - abs_x) <= 35 or abs(b_abs_x - (abs_x + bw)) <= 35):
                    has_brackets = True
                    break

            crop = img[max(0, abs_y - 10) : min(h, abs_y + bh + 40), max(0, abs_x - 15) : min(w, abs_x + bw + 15)]

            # Exemplar memory evaluation
            conf = 0.95 if has_brackets else 0.82
            if mem is not None and crop.size > 0:
                is_neg, neg_sim, _ = mem.is_negative_patch(crop)
                if is_neg:
                    continue
                pos_cid, pos_cname, pos_sim = mem.classify_patch(crop, min_sim=EXEMPLAR_MATCH_THRESHOLD)
                # Sweep G: (a) pozitif listeye sızmış (10,
                # 'monster_3d_negative') kaydı canavar kanıtı sayılmaz —
                # adı negatif olan eşleşme adayı eler. (b) Güven artık
                # exemplar benzerliğiyle yukarı şişirilmiyor (eski
                # `max(conf, pos_sim)` kaldırıldı). (c) Kaynak/engel
                # sınıfına güçlü eşleşme de canavar değildir.
                if is_negative_exemplar_name(pos_cname):
                    continue
                if pos_cid in (0, 1, 2, 3, 4, 5, 14) and float(pos_sim) >= EXEMPLAR_MATCH_THRESHOLD:
                    continue

            monsters.append({
                "screen_x": abs_x + bw // 2,
                "screen_y": abs_y + bh // 2,
                "bbox": (abs_x, abs_y, bw, bh),
                "confidence": round(conf, 3),
                "class_id": 10,
                "class_name": "monster_3d",
                "has_diamond_brackets": has_brackets,
                "is_targeted": has_brackets,
                "is_hostile": True,
            })

        # Deduplicate and cluster vertically/horizontally adjacent monster detections
        # (e.g. nameplate + hostile red HP bar above it belonging to the same monster)
        merged_monsters: List[Dict[str, Any]] = []
        for m in monsters:
            if m.get("class_id") != 10:
                merged_monsters.append(m)
                continue
            bx, by, bw, bh = m["bbox"]
            cx, cy = m["screen_x"], m["screen_y"]
            merged = False
            for existing in merged_monsters:
                if existing.get("class_id") != 10:
                    continue
                ebx, eby, ebw, ebh = existing["bbox"]
                ecx, ecy = existing["screen_x"], existing["screen_y"]
                if abs(cx - ecx) <= max(35, int(round(50 * scale_x))) and abs(cy - ecy) <= max(30, int(round(40 * scale_y))):
                    nx1 = min(bx, ebx)
                    ny1 = min(by, eby)
                    nx2 = max(bx + bw, ebx + ebw)
                    ny2 = max(by + bh, eby + ebh)
                    existing["bbox"] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    existing["screen_x"] = nx1 + (nx2 - nx1) // 2
                    existing["screen_y"] = ny1 + (ny2 - ny1) // 2
                    existing["confidence"] = max(existing["confidence"], m["confidence"])
                    existing["has_diamond_brackets"] = existing["has_diamond_brackets"] or m.get("has_diamond_brackets", False)
                    existing["is_targeted"] = existing["is_targeted"] or m.get("is_targeted", False)
                    merged = True
                    break
            if not merged:
                merged_monsters.append(m)

        return merged_monsters

    def _is_engine_ready(self) -> bool:
        """True when gather templates loaded and capture dimensions are sane."""
        templates_ok = bool(self._template_variants) or self.gather_template is not None
        capture_ok = self.screen_w > 0 and self.screen_h > 0
        return bool(templates_ok and capture_ok)

    def get_vision_engine_health_summary(self) -> dict:
        """Returns vision engine health summary from real engine state."""
        templates_ok = bool(self._template_variants) or self.gather_template is not None
        capture_ok = self.screen_w > 0 and self.screen_h > 0
        return {
            "vision_ready": bool(templates_ok and capture_ok),
            "templates_loaded": bool(templates_ok),
            "template_variants": len(self._template_variants),
            "capture_ok": bool(capture_ok),
        }

    def get_vision_engine_ready_flag(self) -> dict:
        """Returns vision engine ready flag from real engine state."""
        return {"module": "vision", "ready": self._is_engine_ready()}

    def get_vision_engine_version_info(self) -> dict:
        return {"module": "vision", "version": __version__, "ready": self._is_engine_ready()}

    def get_vision_engine_diagnostic_snapshot(self) -> dict:
        ready = self._is_engine_ready()
        return {
            "module": "vision",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "templates_loaded": bool(self._template_variants) or self.gather_template is not None,
        }


# Backward-compatible class alias required across existing modules
Vision = VisionEngine


def get_vision_capabilities() -> list:
    return ["vision_engine", "template_matching", "diagnostic_snapshot"]


def validate_vision_contract() -> bool:
    """
    Real contract check (not an unconditional True): capabilities are present,
    the required public methods exist on VisionEngine, and the badge
    classifier distinguishes 'e' / 'f' while admitting "unknown".
    """
    try:
        if not get_vision_capabilities():
            return False
        for name in (
            "detect_gather_prompt",
            "detect_floating_resource_icon",
            "is_map_open",
            "read_player_hp_mp_bars",
            "detect_monsters_3d",
            "detect_3d_gather_crests",
        ):
            if not callable(getattr(VisionEngine, name, None)):
                return False
        box_f = np.full((36, 37), 60, dtype=np.uint8)
        box_f[5:7, 8:11] = 200
        box_f[10:12, 8:11] = 30
        if VisionEngine._classify_badge_key(box_f) != "f":
            return False
        box_e = np.full((36, 37), 60, dtype=np.uint8)
        box_e[5:7, 8:11] = 200
        box_e[10:12, 8:11] = 200
        if VisionEngine._classify_badge_key(box_e) != "e":
            return False
        box_u = np.full((36, 37), 100, dtype=np.uint8)
        if VisionEngine._classify_badge_key(box_u) != "unknown":
            return False
        return True
    except Exception:
        return False


def get_vision_baseline_metrics() -> dict:
    return {"module": "vision", "contract_valid": validate_vision_contract(), "capability_count": len(get_vision_capabilities())}


def get_vision_readiness_status() -> dict:
    return {"module": "vision", "ready": validate_vision_contract(), "metrics": get_vision_baseline_metrics()}


def get_vision_audit_record() -> dict:
    return {"module": "vision", "audit_passed": validate_vision_contract(), "readiness": get_vision_readiness_status()}


def get_vision_operational_digest() -> dict:
    return {"module": "vision", "operational": validate_vision_contract(), "audit": get_vision_audit_record()}


def get_vision_telemetry_payload() -> dict:
    return {"module": "vision", "telemetry_ready": validate_vision_contract(), "digest": get_vision_operational_digest()}


def get_vision_integrity_seal() -> dict:
    return {"module": "vision", "sealed": validate_vision_contract(), "telemetry": get_vision_telemetry_payload()}


def get_vision_compliance_certificate() -> dict:
    return {"module": "vision", "certified": validate_vision_contract(), "seal": get_vision_integrity_seal()}


def get_vision_readiness_probe() -> dict:
    return {"module": "vision", "ready": validate_vision_contract(), "certificate": get_vision_compliance_certificate()}


def get_vision_audit_summary() -> dict:
    return {"module": "vision", "audit_passed": validate_vision_contract(), "probe": get_vision_readiness_probe()}


__all__ = [
    "VisionEngine",
    "Vision",
    "PLAYER_CENTER_EXCLUSION_ZONE",
    "PLAYER_SELF_EXCLUSION_ZONE",
    "is_in_player_exclusion_zone",
    "is_in_player_self_exclusion_zone",
    "detect_inventory_panel",
]



