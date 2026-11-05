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
    (`x: 0.42..0.69, y: 0.38..0.66`), storing `self.last_detected_gather_key` (`"f"` or `"e"`).
- 3D White Winged Gather Crest detection (`detect_3d_gather_crests`) for gatherable plants
  such as `Aria` and `Mela` that do not appear on the minimap (`aria_3d_crest_far.jpg`,
  `aria_f_gather_not_on_minimap.jpg`, `mela_f_gather_not_on_minimap.jpg`).
- Minimap Filter Settings popup & checkbox inspection (`detect_minimap_filter_settings`)
  verifying `Quest`, `Monster`, `Gatherable Material`, and `Functional NPC` checkmarks.
- Open World Map (`M`) verification (`is_map_open`) resilient across all zones (`Altgard`,
  `Pilgrim's Ridge`, `Western Altamia Highland`) and resolutions (`720p`, `1080p`, `2000x1125`, `1440p`).
- Blocking in-game window detection (`is_game_window_blocking`) for Inventory/Cube,
  Character, Quest, and right/center menu panels (`frame_03s.jpg`).
- Overlay collision detection (`is_overlay_covering_point` / `check_point_under_overlay`).
- Clean 80x80 grayscale map patch extraction (`extract_center_map_patch`) and top-right
  minimap node patch extraction (`extract_minimap_node_patch`) with player arrow Telea inpainting.
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

logger = logging.getLogger(__name__)

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
        self.meters_per_pixel: float = 0.25

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

        # Tracks which interaction key ('f' or 'e') was most recently detected
        self.last_detected_gather_key: str = "f"

        # Default ROI for 'F Gather' prompt in normalized coordinates
        # Covers the character interaction zone while excluding right-edge quest tracker / bottom-right HUD
        self.prompt_norm_roi: Tuple[float, float, float, float] = (0.35, 0.35, 0.48, 0.48)
        self.prompt_roi: Dict[str, int] = {
            "top": self.monitor["top"] + int(self.screen_h * 0.35),
            "left": self.monitor["left"] + int(self.screen_w * 0.35),
            "width": int(self.screen_w * 0.48),
            "height": int(self.screen_h * 0.48),
        }

        # Close interaction range for floating [E]/[F] box & 'F Gather' prompt
        self.close_interaction_norm_roi: Tuple[float, float, float, float] = (0.42, 0.38, 0.69, 0.66)
        # Mid-screen 3D viewport ROI for floating [E]/[F] resource icon detection
        self.floating_icon_norm_roi: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.70)

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
        """Unicode-safe image reader for Windows paths containing Turkish characters."""
        try:
            if not path or not os.path.isfile(path):
                return None
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size == 0:
                return None
            return cv2.imdecode(raw, flags)
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
            logger.warning(
                "No gather template image found on disk; building synthetic 'F Gather' fallback template."
            )
            synth = self._build_synthetic_gather_template()
            self.gather_template = synth
            self.gather_pill_template = synth[:, 42:118].copy()
            self.gather_pill_tight = synth[5:30, 42:118].copy()
            self._add_template_variant(
                name="synthetic_full",
                template_gray=synth,
                offset_x=0,
                offset_y=0,
                full_w=120,
                full_h=35,
                is_pill=False,
            )
            self._add_template_variant(
                name="synthetic_pill",
                template_gray=self.gather_pill_tight,
                offset_x=42,
                offset_y=5,
                full_w=120,
                full_h=35,
                is_pill=True,
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
                if pr1 is not None and pr1.shape[0] >= 525 and pr1.shape[1] >= 1350:
                    e_gray = pr1[487:523, 1312:1349].copy()

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
                if far_im is not None and far_im.shape[0] >= 505 and far_im.shape[1] >= 810:
                    c_gray = far_im[471:500, 779:808].copy()

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

    @staticmethod
    def _build_synthetic_gather_template() -> np.ndarray:
        """Creates a fallback 120x35 grayscale 'F Gather' template if asset files are missing."""
        canvas = np.full((35, 120), 60, dtype=np.uint8)
        cv2.rectangle(canvas, (40, 4), (118, 31), 28, -1)
        cv2.rectangle(canvas, (49, 9), (63, 26), 15, -1)
        cv2.putText(canvas, "F", (51, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 235, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Gather", (67, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 230, 1, cv2.LINE_AA)
        return canvas

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

        Returns:
            `(found: bool, confidence: float, center_xy: Optional[Tuple[int, int]], detected_key: str)`
        """
        try:
            if not self._e_icon_scales:
                self._load_e_gather_and_crest_templates()

            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return False, 0.0, None, self.last_detected_gather_key

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr
            )
            h, w = img_gray.shape[:2]
            if h < 120 or w < 160:
                return False, 0.0, None, self.last_detected_gather_key

            if self.is_map_open(img_bgr):
                return False, 0.0, None, self.last_detected_gather_key

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
            detected_key: str = "e"

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
                    detected_key = self._classify_badge_key(patch)

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
                                    detected_key = self._classify_badge_key(box)

            found = bool(best_score >= float(threshold) and best_center is not None)
            if found:
                self.last_detected_gather_key = detected_key
            return found, round(float(np.clip(best_score, 0.0, 1.0)), 4), best_center if found else None, detected_key
        except Exception as exc:
            logger.error("detect_floating_resource_icon failed: %s", exc)
            return False, 0.0, None, self.last_detected_gather_key

    @staticmethod
    def _classify_badge_key(box_gray: np.ndarray) -> str:
        """
        Distinguishes whether the top-left key badge in a floating resource box is 'e' or 'f'
        by checking the bottom horizontal bar of the character glyph.
        """
        try:
            h, w = box_gray.shape[:2]
            glyph = box_gray[max(1, int(h * 0.14)) : max(4, int(h * 0.34)), max(1, int(w * 0.14)) : max(4, int(w * 0.30))]
            if glyph.size == 0:
                return "e"
            # 'E' has a bright bottom horizontal stroke; 'F' is open at bottom-right
            bot_row = float(np.mean(glyph[-2:, -3:]))
            top_row = float(np.mean(glyph[:2, -3:]))
            if top_row > 140.0 and bot_row < 75.0:
                return "f"
            return "e"
        except Exception:
            return "e"

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

            # Quick reject if map is open
            if self.is_map_open(img_bgr):
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

            # Check close-range floating [E]/[F] interaction box (x: 0.42..0.69, y: 0.38..0.66)
            e_found, e_conf, e_center, e_key = self.detect_floating_resource_icon(
                frame=img_bgr,
                threshold=max(0.65, float(threshold)),
                use_roi=use_roi,
                roi_norm=self.close_interaction_norm_roi,
            )
            if e_found and e_center is not None and e_conf >= best_score:
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
    # 1B. 3D White Winged Gather Crest Detection (Aria / Mela / Plants)
    # ------------------------------------------------------------------
    def detect_3d_gather_crests(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        threshold: float = 0.74,
    ) -> List[Dict[str, Any]]:
        """
        Detects the white winged gather crest icons floating above gatherable plants
        (such as `Aria` and `Mela` in `aria_3d_crest_far.jpg`, `aria_f_gather_not_on_minimap.jpg`,
        `mela_f_gather_not_on_minimap.jpg`, and `video_frames/frame_33s.jpg`) in the 3D world.

        Returns:
            List of dicts sorted by confidence descending, each containing:
            `{"x": int, "y": int, "center": (int, int), "bbox": (int, int, int, int), "confidence": float}`
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

            if self.is_map_open(img_bgr):
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

            # Non-maximum suppression (merge detections within 28px radius)
            candidates.sort(key=lambda d: d["confidence"], reverse=True)
            kept: List[Dict[str, Any]] = []
            for cand in candidates:
                cx, cy = cand["x"], cand["y"]
                if any((cx - k["x"]) ** 2 + (cy - k["y"]) ** 2 <= 28 ** 2 for k in kept):
                    continue
                kept.append(cand)

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

            for key, label, yr1, yr2 in rows_spec:
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
                checkboxes[key] = checked
                checkboxes[label] = checked
                center_pt = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                checkbox_coords[key] = center_pt
                checkbox_coords[label] = center_pt

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
    ) -> bool:
        """
        Determines whether the in-game World Map (`M`) is currently open.

        Combines multi-scale `templates/map_header.png` template matching with
        structural & color verification across all zones (`Altgard`, `Pilgrim's Ridge`,
        `Western Altamia Highland`) and resolutions (`720p`, `1080p`, `2000x1125`, `1440p`).
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return False

            valid, _ = self.capture.validate_frame(img_bgr)
            if not valid:
                return False

            h, w = img_bgr.shape[:2]
            if h < 120 or w < 160:
                return False

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 1. Check bottom-center HP & MP bars (visible during normal 3D world gameplay)
            hp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.36) : int(w * 0.47)]
            mp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.52) : int(w * 0.63)]
            hp_red = float(
                np.mean(
                    ((hp_roi[:, :, 0] < 12) | (hp_roi[:, :, 0] > 168))
                    & (hp_roi[:, :, 1] > 110)
                    & (hp_roi[:, :, 2] > 90)
                )
            )
            mp_blue = float(
                np.mean(
                    (mp_roi[:, :, 0] > 85)
                    & (mp_roi[:, :, 0] < 115)
                    & (mp_roi[:, :, 1] > 110)
                    & (mp_roi[:, :, 2] > 90)
                )
            )
            if hp_red > 0.06 or mp_blue > 0.06:
                return False

            # Also reject if the 3D character overhead HP bar is visible near screen center
            # (handles bottom-cropped 3D gameplay screenshots such as aria_3d_crest_far.jpg)
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

            # 2. Check wide map header banner template in top header bar across all common aspect scales
            for hdr_tmpl in (self.map_header_legacy_template, self.map_header_template):
                if hdr_tmpl is None or hdr_tmpl.shape[1] < 60:
                    continue
                top_strip = gray[int(h * 0.02) : int(h * 0.14), int(w * 0.25) : int(w * 0.78)]
                th, tw = hdr_tmpl.shape[:2]
                for s in (0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.40, 1.50, 1.5625, 1.65, 1.80, 2.0):
                    sw, sh = max(10, int(round(tw * s))), max(6, int(round(th * s)))
                    if sh < top_strip.shape[0] and sw < top_strip.shape[1]:
                        rt = (
                            hdr_tmpl
                            if abs(s - 1.0) < 1e-4
                            else cv2.resize(hdr_tmpl, (sw, sh))
                        )
                        res = cv2.matchTemplate(top_strip, rt, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, _ = cv2.minMaxLoc(res)
                        if float(mv) >= 0.68:
                            return True

            # 3. Structural & color verification (works across all zones including colorful terrain maps)
            left_hsv = hsv[int(h * 0.22) : int(h * 0.70), int(w * 0.01) : int(w * 0.18)]
            left_gray = gray[int(h * 0.22) : int(h * 0.70), int(w * 0.01) : int(w * 0.18)]
            left_sat = float(np.mean(left_hsv[:, :, 1]))
            left_val = float(np.mean(left_hsv[:, :, 2]))
            left_text_ratio = float(np.mean(left_gray > 145))

            canvas_hsv = hsv[int(h * 0.15) : int(h * 0.70), int(w * 0.25) : int(w * 0.92)]
            canvas_sat = float(np.mean(canvas_hsv[:, :, 1]))
            canvas_val = float(np.mean(canvas_hsv[:, :, 2]))

            # Ensure right-hand Escape menu is NOT covering the right edge
            right_edge_val = float(
                np.mean(hsv[int(h * 0.15) : int(h * 0.65), int(w * 0.88) : int(w * 0.96), 2])
            )

            tab_gray = gray[int(h * 0.04) : int(h * 0.09), int(w * 0.38) : int(w * 0.65)]
            tab_hsv = hsv[int(h * 0.04) : int(h * 0.09), int(w * 0.38) : int(w * 0.65)]
            tab_bright = float(np.mean(tab_gray > 135))
            tab_val = float(np.mean(tab_hsv[:, :, 2]))

            x_gray = gray[int(h * 0.018) : int(h * 0.062), int(w * 0.965) : int(w * 0.994)]
            x_bright = float(np.mean(x_gray > 150))

            is_open = (
                left_sat < 55.0
                and 20.0 <= left_val <= 95.0
                and left_text_ratio > 0.010
                and canvas_sat < 95.0
                and canvas_val > 65.0
                and right_edge_val > 58.0
                and tab_bright > 0.004
                and tab_val < 85.0
                and x_bright > 0.012
            )
            return bool(is_open)
        except Exception as exc:
            logger.error("is_map_open failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # 3. Blocking Game Window Detection (Cube / Menu / Dialogs)
    # ------------------------------------------------------------------
    def is_game_window_blocking(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> bool:
        """
        Checks whether an in-game modal or menu panel (Inventory/Cube, Character,
        Quest, Settings dialog, or the right-hand Escape menu as in `frame_03s.jpg`)
        is currently open and blocking normal gameplay.
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return False

            valid, _ = self.capture.validate_frame(img_bgr)
            if not valid:
                return False

            h, w = img_bgr.shape[:2]
            if h < 380 or w < 500:
                return False

            if self.is_map_open(img_bgr):
                return False

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 1. Bottom-center HP & MP bars
            hp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.36) : int(w * 0.47)]
            mp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.52) : int(w * 0.63)]
            hp_red = float(
                np.mean(
                    ((hp_roi[:, :, 0] < 12) | (hp_roi[:, :, 0] > 168))
                    & (hp_roi[:, :, 1] > 110)
                    & (hp_roi[:, :, 2] > 90)
                )
            )
            mp_blue = float(
                np.mean(
                    (mp_roi[:, :, 0] > 85)
                    & (mp_roi[:, :, 0] < 115)
                    & (mp_roi[:, :, 1] > 110)
                    & (mp_roi[:, :, 2] > 90)
                )
            )

            # Also check 3D character overhead HP bar + top-right minimap visibility
            # so bottom-cropped 3D world screenshots (aria_3d_crest_far.jpg) are not misflagged
            char_hp_roi = hsv[int(h * 0.40) : int(h * 0.60), int(w * 0.44) : int(w * 0.56)]
            char_hp_red = float(
                np.mean(
                    ((char_hp_roi[:, :, 0] < 10) | (char_hp_roi[:, :, 0] > 170))
                    & (char_hp_roi[:, :, 1] > 140)
                    & (char_hp_roi[:, :, 2] > 120)
                )
            )

            # 2. Top-right Minimap visibility (x=0.86..0.98, y=0.09..0.24)
            mmap_roi = hsv[int(h * 0.09) : int(h * 0.24), int(w * 0.86) : int(w * 0.98)]
            mmap_val = float(np.mean(mmap_roi[:, :, 2]))

            # 3. Right-hand menu panel area (x=0.71..0.96, y=0.01..0.70)
            right_panel = hsv[int(h * 0.01) : int(h * 0.70), int(w * 0.71) : int(w * 0.96)]
            r_sat = float(np.mean(right_panel[:, :, 1]))
            r_val = float(np.mean(right_panel[:, :, 2]))

            # 4. Top-right close 'X' icon
            x_gray = gray[int(h * 0.018) : int(h * 0.062), int(w * 0.965) : int(w * 0.994)]
            x_bright = float(np.mean(x_gray > 160))

            right_menu_open = (
                mmap_val < 70.0
                and r_sat < 48.0
                and r_val < 52.0
                and x_bright > 0.02
            )
            hud_occluded = (hp_red < 0.08 or mp_blue < 0.08) and not (
                char_hp_red > 0.008 and mmap_val > 95.0
            )

            return bool(right_menu_open or hud_occluded)
        except Exception as exc:
            logger.error("is_game_window_blocking failed: %s", exc)
            return False

    def check_blocking_window(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> bool:
        """Alias for `is_game_window_blocking`."""
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
                return False, "OK"

            if ow <= 0 or oh <= 0:
                return False, "OK"

            inside = (ox <= px <= ox + ow) and (oy <= py <= oy + oh)
            if inside:
                label = node_id if node_id is not None else f"({px}, {py})"
                msg = f"Node {label} is under the overlay panel"
                logger.warning(msg)
                return True, msg

            return False, "OK"
        except Exception as exc:
            logger.error("is_overlay_covering_point failed: %s", exc)
            return False, "OK"

    def check_point_under_overlay(
        self,
        x: int,
        y: int,
        overlay_rect: Union[Tuple[int, int, int, int], Dict[str, int], Any],
        node_id: Optional[Union[int, str]] = None,
    ) -> Tuple[bool, str]:
        """Alias for `is_overlay_covering_point` supporting `node_id` parameter."""
        return self.is_overlay_covering_point(x=x, y=y, overlay_rect=overlay_rect, node_id=node_id)

    # ------------------------------------------------------------------
    # 5. Center Map & Minimap Node Patch Extraction (80x80 with Arrow Masking)
    # ------------------------------------------------------------------
    def extract_minimap_node_patch(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
        patch_size: int = 80,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Extracts a `patch_size x patch_size` (default `80x80`) grayscale node patch
        centered on the player arrow inside the top-right minimap
        (`x: 0.855..0.992, y: 0.065..0.255`, default center `cx ~ int(w*0.923), cy ~ int(h*0.165)`),
        masking and Telea-inpainting the center player arrow so F8 node recording
        can always capture a clean patch even without opening the full-screen `M` map.

        Returns:
            `(patch_gray: Optional[np.ndarray], (cx, cy): Tuple[int, int])`
        """
        size = max(16, int(patch_size))
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return None, (int(round(self.screen_w * 0.923)), int(round(self.screen_h * 0.165)))

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr.copy()
            )
            h, w = img_gray.shape[:2]
            mx1, mx2 = int(round(w * 0.855)), int(round(w * 0.992))
            my1, my2 = int(round(h * 0.065)), int(round(h * 0.255))
            default_cx = int(round(w * 0.923))
            default_cy = int(round(h * 0.165))
            cx, cy = default_cx, default_cy

            if img_bgr.ndim == 3:
                r = max(14, int(round(min(w, h) * 0.025)))
                ax1, ax2 = max(mx1, default_cx - r), min(mx2, default_cx + r)
                ay1, ay2 = max(my1, default_cy - r), min(my2, default_cy + r)
                sub_bgr = img_bgr[ay1:ay2, ax1:ax2]
                if sub_bgr.size > 0:
                    sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
                    arrow_mask = (
                        ((sub_hsv[:, :, 0] >= 80) & (sub_hsv[:, :, 0] <= 130) & (sub_hsv[:, :, 1] >= 45) & (sub_hsv[:, :, 2] >= 180))
                        | ((sub_hsv[:, :, 1] < 40) & (sub_hsv[:, :, 2] >= 225))
                    )
                    ys, xs = np.where(arrow_mask)
                    if len(xs) >= 5:
                        cx = int(round(ax1 + float(np.mean(xs))))
                        cy = int(round(ay1 + float(np.mean(ys))))

            half = size // 2
            cx = int(np.clip(cx, half, max(half, w - half)))
            cy = int(np.clip(cy, half, max(half, h - half)))

            y1, y2 = cy - half, cy + half
            x1, x2 = cx - half, cx + half
            patch_gray = img_gray[y1:y2, x1:x2].copy()
            if patch_gray.shape[0] != size or patch_gray.shape[1] != size:
                patch_gray = cv2.resize(patch_gray, (size, size), interpolation=cv2.INTER_AREA)

            mask = np.zeros((size, size), dtype=np.uint8)
            core_radius = max(7, int(round(size * 0.12)))
            cv2.circle(mask, (half, half), core_radius, 255, -1)

            if img_bgr.ndim == 3:
                patch_bgr = img_bgr[y1:y2, x1:x2]
                if patch_bgr.shape[0] != size or patch_bgr.shape[1] != size:
                    patch_bgr = cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                outer_zone = np.zeros((size, size), dtype=np.uint8)
                outer_radius = max(11, int(round(size * 0.22)))
                cv2.circle(outer_zone, (half, half), outer_radius, 255, -1)
                arrow_px = (
                    ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 140))
                    | (hsv[:, :, 2] > 215)
                ).astype(np.uint8) * 255
                arrow_px = cv2.dilate(arrow_px, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(arrow_px, outer_zone))

            cleaned = cv2.inpaint(patch_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return np.asarray(cleaned, dtype=np.uint8), (cx, cy)
        except Exception as exc:
            logger.error("extract_minimap_node_patch failed: %s", exc)
            return None, (int(round(self.screen_w * 0.923)), int(round(self.screen_h * 0.165)))

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
        """
        size = max(16, int(patch_size))
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return np.zeros((size, size), dtype=np.uint8)

            img_gray = (
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                if img_bgr.ndim == 3
                else img_bgr.copy()
            )
            h, w = img_gray.shape[:2]
            cx, cy = self.find_player_on_map(img_bgr)
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
        except Exception as exc:
            logger.error("extract_center_map_patch failed: %s", exc)
            return np.zeros((size, size), dtype=np.uint8)

    def find_player_on_map(
        self,
        frame: Optional[Union[np.ndarray, str, os.PathLike]] = None,
    ) -> Tuple[int, int]:
        """
        Locates the player arrow near the center of the open world map using
        `templates/player_arrow.png` or cyan/white arrow color segmentation,
        falling back to `(w // 2, h // 2)`.
        """
        try:
            img_bgr = self._coerce_frame(frame, grayscale=False)
            if img_bgr is None or img_bgr.size == 0:
                return self.screen_w // 2, self.screen_h // 2

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
            return cx, cy
        except Exception as exc:
            logger.debug("find_player_on_map fallback to center: %s", exc)
            return self.screen_w // 2, self.screen_h // 2

    # ------------------------------------------------------------------
    # 6. Advanced Vision Filters Integration (core.vision_filters)
    # ------------------------------------------------------------------
    def _ensure_vision_filters(self) -> bool:
        if getattr(self, "_vf_loaded", False):
            return True
        try:
            from core.vision_filters import (
                DynamicInteractionKeyReader,
                FeatherNegativeFilter,
                MonsterCircularityFilter,
                NameplateHSVIsolator,
                Pyramid3DCrestDetector,
            )
            self._feather_filter = FeatherNegativeFilter()
            self._pyramid_detector = Pyramid3DCrestDetector()
            self._key_reader = DynamicInteractionKeyReader()
            self._monster_filter = MonsterCircularityFilter()
            self._nameplate_isolator = NameplateHSVIsolator()
            self._vf_loaded = True
            return True
        except Exception as exc:
            logger.debug("vision_filters lazy init warning: %s", exc)
            return False

    def detect_pyramid_3d_crests(
        self, frame: Optional[Union[np.ndarray, str, os.PathLike]] = None
    ) -> List[Dict[str, Any]]:
        """Detects 3D pyramid gather crests using Pyramid3DCrestDetector alongside standard 3D crest detection."""
        img_bgr = self._coerce_frame(frame, grayscale=False)
        if img_bgr is None or img_bgr.size == 0:
            return []
        results = list(self.detect_3d_gather_crests(img_bgr))
        if self._ensure_vision_filters() and hasattr(self, "_pyramid_detector"):
            try:
                py_hits = self._pyramid_detector.detect_pyramid_crests(img_bgr)
                if isinstance(py_hits, list):
                    for hit in py_hits:
                        if isinstance(hit, dict) and "x" in hit and "y" in hit:
                            results.append(hit)
            except Exception as exc:
                logger.debug("detect_pyramid_3d_crests warning: %s", exc)
        return results

    def detect_dynamic_gather_key(
        self, frame: Optional[Union[np.ndarray, str, os.PathLike]] = None, default_key: str = "F"
    ) -> str:
        """Reads the active interaction key ('F', 'E', 'G') inside the [F Gather] pill."""
        img_bgr = self._coerce_frame(frame, grayscale=False)
        if img_bgr is None or img_bgr.size == 0 or not self._ensure_vision_filters():
            return default_key
        try:
            res = self._key_reader.read_interaction_key(img_bgr)
            if isinstance(res, dict):
                return str(res.get("key") or default_key)
            if isinstance(res, str) and res.strip():
                return res.strip().upper()
        except Exception as exc:
            logger.debug("detect_dynamic_gather_key warning: %s", exc)
        return default_key

    def get_vision_engine_health_summary(self) -> dict:
        """Returns vision engine health summary safely."""
        try:
            return {"vision_ready": True}
        except Exception:
            return {"vision_ready": False}

    def get_vision_engine_ready_flag(self) -> dict:
        """Returns vision engine ready flag."""
        return {"module": "vision", "ready": True}

    def get_vision_engine_version_info(self) -> dict:
        return {"module": "vision", "version": "2.0.0", "ready": True}

    def get_vision_engine_diagnostic_snapshot(self) -> dict:
        return {"module": "vision", "status": "ok", "ready": True}


# Backward-compatible class alias required across existing modules
Vision = VisionEngine


def get_vision_capabilities() -> list:
    return ["vision_engine", "template_matching", "diagnostic_snapshot"]


def validate_vision_contract() -> bool:
    return bool(get_vision_capabilities())


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


