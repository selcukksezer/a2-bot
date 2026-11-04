"""3D Game World & Minimap Automatic Ore/Gem/Material Scanner for Aion 2.

Automatically detects gatherable materials (Aria, Mela, Orichalcum Ore,
Diamond/Sapphire/Ruby Gemstones, etc.) directly from the 3D game viewport
(white symmetrical winged gather crests above cyan/green labels, diamond-tipped
resource nameplates, and floating square [E]/[F] gatherable material icons),
the interactive "F Gather" prompt, and the top-right minimap (white 3-leaf
sprout icons) while rejecting monster nameplates, monster dots, and feathers.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedMaterial:
    """Represents a gatherable resource node detected on screen or minimap."""

    material_type: str
    source: str
    screen_x: int
    screen_y: int
    bbox: tuple[int, int, int, int]
    rel_map_dx: float
    rel_map_dy: float
    estimated_distance_m: float
    confidence: float
    in_air: bool = False
    stack_count: int = 1
    harvests_remaining: int = 3


class MaterialScanner:
    """Scans game frames for 3D world material crests/nameplates, gather prompts, floating [E]/[F] boxes, and minimap sprout icons."""

    # Reference resolution for normalized coordinate and perspective calculations
    REF_WIDTH: int = 1280
    REF_HEIGHT: int = 720

    # Player anchor position in 3D screen space (normalized)
    PLAYER_ANCHOR_X_RATIO: float = 0.50
    PLAYER_ANCHOR_Y_RATIO: float = 0.54

    # Minimap normalized ROI bounds (x: 0.855..0.992, y: 0.065..0.255)
    MINIMAP_X1_RATIO: float = 0.855
    MINIMAP_Y1_RATIO: float = 0.065
    MINIMAP_X2_RATIO: float = 0.992
    MINIMAP_Y2_RATIO: float = 0.255

    # Inner minimap radar canvas bounds (excluding header text bar and outer frame)
    MINIMAP_INNER_X1_RATIO: float = 0.858
    MINIMAP_INNER_Y1_RATIO: float = 0.088
    MINIMAP_INNER_X2_RATIO: float = 0.990
    MINIMAP_INNER_Y2_RATIO: float = 0.246

    # Primary HSV range for cyan/teal diamond-tipped resource labels (< Orichalcum >)
    CYAN_PRIMARY_LOWER: tuple[int, int, int] = (78, 90, 130)
    CYAN_PRIMARY_UPPER: tuple[int, int, int] = (102, 255, 255)

    # Secondary HSV range for un-targeted teal/green resource text in 3D world (Aria, Mela)
    CYAN_SECONDARY_LOWER: tuple[int, int, int] = (60, 50, 110)
    CYAN_SECONDARY_UPPER: tuple[int, int, int] = (102, 255, 255)

    # White/silver resource crest icon HSV range above 3D nameplates
    WHITE_CREST_LOWER: tuple[int, int, int] = (0, 0, 168)
    WHITE_CREST_UPPER: tuple[int, int, int] = (180, 60, 255)

    # Canonical 28x29 bit-packed binary mask of the 3D white symmetrical winged gather crest
    _CREST_BITMASK_ROWS: Sequence[int] = (
        81920,
        278529,
        302533659,
        369117195,
        335568911,
        469786639,
        402669583,
        427835506,
        432013430,
        207618246,
        241172878,
        242221468,
        133695484,
        133959676,
        66981880,
        16650208,
        8261504,
        4071168,
        1974016,
        1973760,
        1973760,
        1974016,
        1974016,
        1974016,
        1974016,
        4071168,
        924672,
        202752,
    )

    # Canonical 12x12 V-channel template for the full-screen minimap 3-leaf white sprout icon
    _SPROUT_12X12_V: Sequence[Sequence[int]] = (
        (193, 185, 195, 127, 178, 131, 146, 193, 196, 195, 203, 202),
        (186, 193, 165, 129, 233, 200, 115, 177, 201, 199, 208, 204),
        (176, 186, 120, 179, 230, 239, 137, 138, 194, 200, 205, 204),
        (182, 159, 124, 227, 215, 226, 212, 110, 176, 199, 201, 204),
        (191, 119, 162, 255, 199, 219, 255, 134, 135, 186, 198, 208),
        (161, 105, 205, 255, 174, 204, 255, 192, 110, 166, 194, 214),
        (113, 137, 255, 254, 169, 181, 255, 248, 135, 146, 192, 219),
        (85, 123, 237, 255, 143, 181, 255, 225, 104, 125, 213, 220),
        (127, 131, 174, 241, 146, 177, 235, 148, 132, 139, 128, 147),
        (245, 237, 206, 187, 116, 141, 215, 210, 240, 232, 166, 108),
        (206, 255, 242, 139, 63, 68, 179, 255, 255, 174, 149, 196),
        (111, 152, 146, 127, 139, 121, 113, 116, 135, 145, 205, 209),
    )

    # Canonical 17x17 V-channel template for the closeup minimap 3-leaf white sprout icon
    _SPROUT_17X17_V: Sequence[Sequence[int]] = (
        (89, 88, 87, 82, 88, 62, 40, 59, 83, 87, 89, 89, 90, 87, 82, 78, 79),
        (87, 89, 89, 79, 86, 64, 124, 94, 71, 86, 83, 90, 89, 85, 80, 81, 83),
        (95, 93, 83, 89, 68, 92, 232, 161, 55, 85, 83, 82, 83, 83, 81, 82, 83),
        (88, 87, 83, 85, 35, 158, 233, 202, 82, 75, 95, 87, 87, 87, 86, 84, 81),
        (81, 85, 94, 60, 86, 241, 211, 229, 152, 58, 88, 94, 91, 89, 89, 85, 80),
        (91, 91, 90, 45, 185, 246, 215, 228, 203, 85, 75, 92, 89, 87, 86, 86, 82),
        (99, 87, 61, 94, 237, 236, 186, 215, 239, 147, 40, 88, 91, 92, 89, 87, 85),
        (90, 78, 57, 182, 255, 255, 165, 217, 255, 200, 36, 75, 89, 97, 92, 86, 85),
        (77, 77, 85, 229, 255, 234, 155, 196, 255, 249, 122, 51, 76, 92, 88, 83, 84),
        (83, 48, 176, 249, 255, 232, 145, 204, 245, 255, 221, 54, 78, 90, 86, 81, 82),
        (72, 58, 166, 247, 255, 248, 116, 192, 255, 246, 214, 66, 41, 88, 95, 80, 93),
        (55, 39, 94, 185, 245, 236, 116, 173, 255, 206, 94, 30, 70, 73, 81, 90, 82),
        (104, 147, 161, 122, 183, 234, 89, 179, 210, 132, 183, 140, 98, 80, 50, 37, 62),
        (255, 248, 255, 228, 97, 175, 73, 141, 137, 215, 247, 255, 248, 208, 104, 48, 81),
        (154, 253, 254, 243, 220, 82, 47, 89, 202, 245, 254, 239, 177, 86, 70, 88, 76),
        (66, 133, 184, 144, 81, 43, 47, 46, 81, 127, 198, 144, 71, 80, 87, 82, 91),
        (80, 53, 63, 81, 88, 103, 96, 94, 88, 64, 50, 67, 80, 87, 90, 89, 83),
    )

    def __init__(
        self,
        perspective_scale_x: float = 0.085,
        perspective_scale_y: float = 0.115,
        meters_per_map_pixel: float = 0.65,
        templates_dir: str = "templates",
    ) -> None:
        """Initialize the material scanner with perspective projection parameters and templates.

        Args:
            perspective_scale_x: Horizontal conversion scale from 3D screen offset to map pixels.
            perspective_scale_y: Vertical conversion scale from 3D screen offset to map pixels.
            meters_per_map_pixel: Conversion factor from relative map pixels to estimated meters.
            templates_dir: Directory containing optional PNG templates.
        """
        self.perspective_scale_x = float(perspective_scale_x)
        self.perspective_scale_y = float(perspective_scale_y)
        self.meters_per_map_pixel = float(meters_per_map_pixel)
        self.templates_dir = templates_dir

        self._crest_template_mask = self._build_crest_template_mask()
        self._sprout_12_tpl = np.array(self._SPROUT_12X12_V, dtype=np.uint8)
        self._sprout_17_tpl = np.array(self._SPROUT_17X17_V, dtype=np.uint8)

        self._disk_crest_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "world_gather_crest.png"
        )
        self._disk_e_box_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "e_gather_icon.png"
        )
        self._disk_sprout_tpl: Optional[np.ndarray] = self._load_optional_gray_template(
            "minimap_sprout.png"
        )

    def _build_crest_template_mask(self) -> np.ndarray:
        """Unpack the 28x29 canonical binary mask for the 3D white winged gather crest."""
        h = len(self._CREST_BITMASK_ROWS)
        w = 29
        mask = np.zeros((h, w), dtype=np.uint8)
        for r, row_bits in enumerate(self._CREST_BITMASK_ROWS):
            for c in range(w):
                if (row_bits >> c) & 1:
                    mask[r, c] = 255
        return mask

    def _load_optional_gray_template(self, filename: str) -> Optional[np.ndarray]:
        """Load an optional grayscale template from the templates directory if present."""
        path = os.path.join(self.templates_dir, filename)
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                return img
        return None

    def scan_frame_for_materials(
        self, frame_bgr: np.ndarray
    ) -> list[DetectedMaterial]:
        """Scan a single BGR game frame for all visible materials across 3D world and minimap.

        Executes:
        1. 3D world white winged gather crest + cyan/green nameplate scanning (Aria, Mela, Orichalcum)
           with strict monster nameplate rejection (Yellow-Maned Zaif, Highland Odyle Sparkie).
        2. 3D world floating square [E]/[F] gatherable material icon scanning.
        3. Interactive "F Gather" button prompt scanning near the character.
        4. Top-right minimap white 3-leaf sprout icon scanning (rejecting monster dots and feathers).

        Args:
            frame_bgr: Input image in BGR format (numpy ndarray).

        Returns:
            List of DetectedMaterial objects sorted by confidence descending.
        """
        try:
            if (
                frame_bgr is None
                or not isinstance(frame_bgr, np.ndarray)
                or frame_bgr.size == 0
                or frame_bgr.ndim != 3
                or frame_bgr.shape[2] < 3
            ):
                logger.warning("Invalid frame provided to scan_frame_for_materials.")
                return []

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, w = frame_bgr.shape[:2]

            # If frame is a minimap closeup crop (w < 1000), only run minimap material scanning
            if w < 1000:
                minimap_nodes = self.scan_minimap_materials(frame_bgr, hsv)
                minimap_nodes.sort(key=lambda item: item.confidence, reverse=True)
                return minimap_nodes

            # Reject full-screen modal settings/import dialogs so UI elements are never misclassified
            if self._is_fullscreen_modal_open(frame_bgr, hsv):
                logger.debug("Full-screen modal/settings dialog detected; skipping material scan.")
                return []

            results: list[DetectedMaterial] = []

            # 1. Scan 3D world white winged gather crests and cyan/teal resource nameplates
            world_materials = self.scan_3d_world_nameplates(frame_bgr, hsv)
            results.extend(world_materials)

            # 2. Scan floating square [E] / [F] gatherable material icons in the 3D world
            floating_boxes = self._scan_floating_gather_boxes(frame_bgr, hsv)
            results.extend(floating_boxes)

            # 3. Scan "F Gather" interactive prompt near the player
            gather_prompts = self.scan_gather_prompt(frame_bgr, hsv, world_materials)
            results.extend(gather_prompts)

            # 4. Scan top-right minimap for 3-leaf sprout gatherable material icons
            minimap_nodes = self.scan_minimap_materials(frame_bgr, hsv)
            results.extend(minimap_nodes)

            results.sort(key=lambda item: item.confidence, reverse=True)
            return results

        except Exception as exc:
            logger.error("Error in scan_frame_for_materials: %s", exc, exc_info=True)
            return []

    def _scan_3d_gather_crests(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect white symmetrical winged gather crests floating above cyan/green labels (Aria, Mela).

        Rejects yellow/olive monster labels (e.g. Highland Odyle Sparkie, Yellow-Maned Zaif)
        and player character nameplates.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_world".
        """
        detections: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return detections

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            if w < 600 or h < 360:
                return detections

            scale_x = w / 2000.0
            scale_y = h / 1124.0

            # Restrict search to 3D world viewport
            vx1 = int(w * 0.08)
            vy1 = int(h * 0.12)
            vx2 = int(w * 0.82)
            vy2 = int(h * 0.78)

            roi_hsv = hsv[vy1:vy2, vx1:vx2]
            if roi_hsv.size == 0:
                return detections

            white_mask = cv2.inRange(
                roi_hsv, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER
            )

            # Multi-scale template matching using canonical winged crest mask (and disk template if present)
            candidate_boxes: list[tuple[float, int, int, int, int]] = []
            scales = [1.0]
            if abs(scale_x - 1.0) > 0.08:
                scales.append(max(0.55, min(1.35, scale_x)))

            for s in scales:
                tw = max(16, int(round(self._crest_template_mask.shape[1] * s)))
                th = max(16, int(round(self._crest_template_mask.shape[0] * s)))
                if tw >= white_mask.shape[1] or th >= white_mask.shape[0]:
                    continue
                tpl = (
                    self._crest_template_mask
                    if (tw == self._crest_template_mask.shape[1] and th == self._crest_template_mask.shape[0])
                    else cv2.resize(self._crest_template_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                )
                res = cv2.matchTemplate(white_mask, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= 0.58)
                for y_pt, x_pt in zip(ys, xs):
                    candidate_boxes.append(
                        (float(res[y_pt, x_pt]), int(x_pt), int(y_pt), int(tw), int(th))
                    )

            if self._disk_crest_tpl is not None:
                d_th, d_tw = self._disk_crest_tpl.shape[:2]
                if d_tw < roi_hsv.shape[1] and d_th < roi_hsv.shape[0]:
                    res_d = cv2.matchTemplate(
                        roi_hsv[:, :, 2], self._disk_crest_tpl, cv2.TM_CCOEFF_NORMED
                    )
                    ys_d, xs_d = np.where(res_d >= 0.62)
                    for y_pt, x_pt in zip(ys_d, xs_d):
                        candidate_boxes.append(
                            (float(res_d[y_pt, x_pt]), int(x_pt), int(y_pt), int(d_tw), int(d_th))
                        )

            candidate_boxes.sort(key=lambda item: item[0], reverse=True)
            kept_crests: list[tuple[float, int, int, int, int]] = []
            for score, cx_b, cy_b, cw, ch in candidate_boxes:
                if any(
                    abs(cx_b - kx) <= max(16, cw // 2) and abs(cy_b - ky) <= max(16, ch // 2)
                    for _, kx, ky, _, _ in kept_crests
                ):
                    continue
                kept_crests.append((score, cx_b, cy_b, cw, ch))

            for score, cx_b, cy_b, cw, ch in kept_crests:
                sub_wm = white_mask[cy_b : cy_b + ch, cx_b : cx_b + cw]
                white_px = int(np.count_nonzero(sub_wm))
                if white_px < max(60, int(round(110 * scale_x * scale_y))):
                    continue

                # Verify horizontal left-right mirror symmetry of the winged crest
                flipped = np.fliplr(sub_wm)
                inter = int(np.count_nonzero(sub_wm & flipped))
                union = int(np.count_nonzero(sub_wm | flipped))
                sym_ratio = inter / float(max(1, union))
                if sym_ratio < 0.42:
                    continue

                # Inspect the label region directly below the winged crest for cyan/green text (Aria, Mela)
                tx1 = max(0, cx_b - int(round(12 * scale_x)))
                tx2 = min(roi_hsv.shape[1], cx_b + cw + int(round(12 * scale_x)))
                ty1 = max(0, cy_b + ch)
                ty2 = min(roi_hsv.shape[0], cy_b + ch + max(18, int(round(32 * scale_y))))
                if ty2 <= ty1 or tx2 <= tx1:
                    continue

                label_hsv = roi_hsv[ty1:ty2, tx1:tx2]
                cg_mask = cv2.inRange(
                    label_hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
                )
                cg_px = int(np.count_nonzero(cg_mask))
                min_cg_px = max(25, int(round(50 * scale_x * scale_y)))
                if cg_px < min_cg_px:
                    continue

                abs_x = vx1 + cx_b
                abs_y = vy1 + cy_b
                abs_label_y2 = vy1 + ty2

                # Reject if this is a monster nameplate (yellow text + red agro badge)
                if self._is_monster_nameplate(
                    hsv,
                    abs_x,
                    vy1 + ty1,
                    cw,
                    ty2 - ty1,
                    w / float(self.REF_WIDTH),
                    h / float(self.REF_HEIGHT),
                ):
                    continue

                material_type, in_air = self._classify_3d_material(
                    frame_bgr,
                    hsv,
                    abs_x,
                    vy1 + ty1,
                    cw,
                    ty2 - ty1,
                )

                node_screen_x = int(abs_x + cw // 2)
                node_screen_y = int(
                    min(h - 1, abs_label_y2 + int(round(24 * (h / float(self.REF_HEIGHT)))))
                )
                full_bbox = (
                    int(vx1 + min(cx_b, tx1)),
                    int(abs_y),
                    int(max(cw, tx2 - tx1)),
                    int(abs_label_y2 - abs_y),
                )

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(
                    node_screen_x, node_screen_y, w, h
                )
                conf = round(min(0.99, 0.84 + 0.12 * score + 0.03 * sym_ratio), 3)

                detections.append(
                    DetectedMaterial(
                        material_type=material_type,
                        source="3d_world",
                        screen_x=node_screen_x,
                        screen_y=node_screen_y,
                        bbox=full_bbox,
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=in_air,
                        stack_count=1,
                        harvests_remaining=3,
                    )
                )

            return detections

        except Exception as exc:
            logger.error("Error in _scan_3d_gather_crests: %s", exc, exc_info=True)
            return detections

    def _is_monster_nameplate(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
        scale_x: float,
        scale_y: float,
    ) -> bool:
        """Check whether a 3D world nameplate belongs to a monster (e.g. Yellow-Maned Zaif, Highland Odyle Sparkie).

        Monsters exhibit:
        1. Yellow/orange/olive text between the diamond bracket tips (H=14..38, S=90..255, V=120..255)
           with little or no cyan/green text inside the inner text span, OR
        2. A red/orange square agro/rank badge ('4') and level number directly below the health bar.
        """
        h, w = hsv.shape[:2]

        # 1. Inspect inner text region (excluding left/right 18% where cyan diamond tips sit)
        pad_x = max(2, int(round(cw * 0.18)))
        ix1 = max(0, x + pad_x)
        ix2 = min(w, x + cw - pad_x)
        iy1 = max(0, y - 2)
        iy2 = min(h, y + ch + 4)

        if ix2 > ix1 and iy2 > iy1:
            inner_hsv = hsv[iy1:iy2, ix1:ix2]
            yellow_text_mask = cv2.inRange(inner_hsv, (14, 95, 125), (38, 255, 255))
            cyan_text_mask = cv2.inRange(
                inner_hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
            )
            yellow_px = int(np.count_nonzero(yellow_text_mask))
            cyan_inner_px = int(np.count_nonzero(cyan_text_mask))
            # If the label contains substantial cyan/green gatherable text, it is NOT a monster nameplate
            if cyan_inner_px >= 18 and yellow_px < cyan_inner_px * 2:
                return False
            if cyan_inner_px < 18 and yellow_px > max(18, cyan_inner_px * 2):
                return True

        # 2. Inspect region directly below the nameplate for a red/orange agro/rank square badge ('4')
        bx1 = max(0, x + cw // 4)
        bx2 = min(w, x + (cw * 3) // 4)
        by1 = min(h, y + ch)
        by2 = min(h, y + ch + max(24, int(round(42 * scale_y))))
        if bx2 > bx1 and by2 > by1:
            below_hsv = hsv[by1:by2, bx1:bx2]
            red1 = cv2.inRange(below_hsv, (0, 110, 80), (12, 255, 255))
            red2 = cv2.inRange(below_hsv, (168, 110, 80), (180, 255, 255))
            red_badge_px = int(np.count_nonzero(red1 | red2))
            if red_badge_px >= max(16, int(round(20 * scale_x * scale_y))):
                return True

        return False

    def scan_3d_world_nameplates(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect 3D world gatherable materials via winged gather crests and cyan/teal diamond-tipped nameplates.

        Strictly filters out monster nameplates (such as '‹ Yellow-Maned Zaif ›' and
        'Highland Odyle Sparkie').

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_world".
        """
        detections: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return detections

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # First run dedicated 3D white symmetrical winged gather crest + cyan/green label scanner (Aria, Mela)
            crest_detections = self._scan_3d_gather_crests(frame_bgr, hsv)
            detections.extend(crest_detections)

            h, w = frame_bgr.shape[:2]
            scale_x = w / float(self.REF_WIDTH)
            scale_y = h / float(self.REF_HEIGHT)

            mask_primary = cv2.inRange(
                hsv, self.CYAN_PRIMARY_LOWER, self.CYAN_PRIMARY_UPPER
            )
            mask_secondary = cv2.inRange(
                hsv, self.CYAN_SECONDARY_LOWER, self.CYAN_SECONDARY_UPPER
            )
            cyan_mask = cv2.bitwise_or(mask_primary, mask_secondary)

            # Restrict to 3D viewport (exclude top HUD bar, bottom skill bar, side panels)
            top_cut = int(h * 0.10)
            bottom_cut = int(h * 0.80)
            left_cut = int(w * 0.08)
            right_cut = int(w * 0.82)

            cyan_mask[:top_cut, :] = 0
            cyan_mask[bottom_cut:, :] = 0
            cyan_mask[:, :left_cut] = 0
            cyan_mask[:, right_cut:] = 0

            kernel_w = max(8, int(round(14 * scale_x)))
            kernel_h = max(3, int(round(4 * scale_y)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
            closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_w = max(16, int(round(18 * scale_x)))
            max_w = max(90, int(round(145 * scale_x)))
            min_h = max(4, int(round(5 * scale_y)))
            max_h = max(20, int(round(30 * scale_y)))

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < min_w or cw > max_w or ch < min_h or ch > max_h:
                    continue

                # Skip if already covered by a 3D winged crest detection
                cx = x + cw // 2
                cy = y + ch // 2
                if any(
                    abs(cx - (d.bbox[0] + d.bbox[2] // 2)) <= max(28, cw)
                    and abs(cy - (d.bbox[1] + d.bbox[3] // 2)) <= max(36, ch * 3)
                    for d in detections
                ):
                    continue

                roi_mask = cyan_mask[y : y + ch, x : x + cw]
                cyan_px = int(np.count_nonzero(roi_mask))
                min_px = max(12, int(round(15 * scale_x * scale_y)))
                fill_ratio = cyan_px / float(max(1, cw * ch))

                # Text + diamond tips have moderate fill ratio, never a solid UI rectangle
                if cyan_px < min_px or fill_ratio > 0.65:
                    continue

                hue_values = hsv[y : y + ch, x : x + cw, 0][roi_mask > 0]
                if hue_values.size == 0:
                    continue
                mean_hue = float(np.mean(hue_values))

                # Filter out pure blue UI selection highlights (which have H >= 98)
                if mean_hue > 97.5:
                    continue

                # Reject monster nameplates (e.g. ‹ Yellow-Maned Zaif › with yellow text / red agro badge)
                if self._is_monster_nameplate(hsv, x, y, cw, ch, scale_x, scale_y):
                    continue

                # Verify the bright white/silver crest icon directly above the nameplate
                crest_y1 = max(0, y - int(round(38 * scale_y)))
                crest_y2 = max(0, y - int(round(2 * scale_y)))
                crest_half_w = max(14, int(round(22 * scale_x)))
                crest_x1 = max(0, cx - crest_half_w)
                crest_x2 = min(w, cx + crest_half_w)

                if crest_y2 <= crest_y1 or crest_x2 <= crest_x1:
                    continue

                crest_hsv = hsv[crest_y1:crest_y2, crest_x1:crest_x2]
                crest_mask = cv2.inRange(
                    crest_hsv, self.WHITE_CREST_LOWER, self.WHITE_CREST_UPPER
                )
                white_px = int(np.count_nonzero(crest_mask))
                min_white_px = max(10, int(round(14 * scale_x * scale_y)))
                max_white_px = max(140, int(round(280 * scale_x * scale_y)))

                if not (min_white_px <= white_px <= max_white_px):
                    continue

                # Check horizontal symmetry of the white crest above the nameplate
                flipped_crest = np.fliplr(crest_mask)
                c_inter = int(np.count_nonzero(crest_mask & flipped_crest))
                c_union = int(np.count_nonzero(crest_mask | flipped_crest))
                if (c_inter / float(max(1, c_union))) < 0.25:
                    continue

                # Check if diamond bracket tips (< Orichalcum >) are illuminated (active target)
                has_diamond_tips = cw >= int(round(80 * scale_x)) and cyan_px >= int(
                    round(140 * scale_x * scale_y)
                )

                material_type, in_air = self._classify_3d_material(
                    frame_bgr, hsv, x, y, cw, ch
                )

                # Target node position on the ground right below the nameplate
                node_screen_x = int(cx)
                node_screen_y = int(min(h - 1, y + ch + int(round(28 * scale_y))))
                full_bbox = (
                    int(min(x, crest_x1)),
                    int(crest_y1),
                    int(max(cw, crest_x2 - crest_x1)),
                    int((y + ch) - crest_y1),
                )

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(
                    node_screen_x, node_screen_y, w, h
                )

                base_conf = 0.88 if has_diamond_tips else 0.80
                crest_bonus = min(0.08, (white_px / float(max_white_px)) * 0.10)
                cyan_bonus = min(0.04, (cyan_px / 250.0) * 0.04)
                confidence = round(min(0.99, base_conf + crest_bonus + cyan_bonus), 3)

                detections.append(
                    DetectedMaterial(
                        material_type=material_type,
                        source="3d_world",
                        screen_x=node_screen_x,
                        screen_y=node_screen_y,
                        bbox=full_bbox,
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=confidence,
                        in_air=in_air,
                        stack_count=1,
                        harvests_remaining=3,
                    )
                )

            return detections

        except Exception as exc:
            logger.error("Error in scan_3d_world_nameplates: %s", exc, exc_info=True)
            return detections

    def _scan_floating_gather_boxes(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Detect floating square [E] / [F] gatherable material boxes (36x36 to 44x44 px) in the 3D world.

        Detects the square metallic/olive border frame with strong 4-side Sobel edges,
        top-left white [E]/[F] keybind badge on a dark key cap, and glowing resource graphic inside.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="3d_floating_icon" and material_type="gatherable_node".
        """
        boxes: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return boxes

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            if w < 600 or h < 360:
                return boxes

            x1 = int(w * 0.08)
            y1 = int(h * 0.12)
            x2 = int(w * 0.82)
            y2 = int(h * 0.78)

            roi_hsv = hsv[y1:y2, x1:x2]
            if roi_hsv.size == 0:
                return boxes

            v_chan = roi_hsv[:, :, 2]
            s_chan = roi_hsv[:, :, 1]

            # 1. Optional template matching if templates/e_gather_icon.png exists on disk
            raw_candidates: list[tuple[float, int, int, int, int]] = []
            if self._disk_e_box_tpl is not None:
                th, tw = self._disk_e_box_tpl.shape[:2]
                if tw < v_chan.shape[1] and th < v_chan.shape[0]:
                    res_tpl = cv2.matchTemplate(
                        v_chan, self._disk_e_box_tpl, cv2.TM_CCOEFF_NORMED
                    )
                    ys_t, xs_t = np.where(res_tpl >= 0.72)
                    for y_pt, x_pt in zip(ys_t, xs_t):
                        raw_candidates.append(
                            (
                                float(res_tpl[y_pt, x_pt]),
                                int(x1 + x_pt),
                                int(y1 + y_pt),
                                int(tw),
                                int(th),
                            )
                        )

            # 2. Structural + HSV + Sobel detection for the square metallic/olive frame + top-left E/F badge
            gx = np.abs(cv2.Sobel(v_chan, cv2.CV_32F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(v_chan, cv2.CV_32F, 0, 1, ksize=3))

            # Top-left keybind letter (E or F) has bright white pixels (V>=175, S<=48) adjacent to dark key cap (V<=52)
            key_white = ((v_chan >= 175) & (s_chan <= 48)).astype(np.uint8) * 255
            key_dark = (v_chan <= 52).astype(np.uint8) * 255
            badge_seed = cv2.bitwise_and(
                cv2.dilate(key_white, np.ones((3, 3), np.uint8)),
                cv2.dilate(key_dark, np.ones((3, 3), np.uint8)),
            )
            contours, _ = cv2.findContours(
                badge_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            scaled_sz = max(28, min(46, int(round(36 * (w / 2000.0)))))
            candidate_sizes = sorted({scaled_sz, 36})

            for cnt in contours:
                kx, ky, kw, kh = cv2.boundingRect(cnt)
                if not (2 <= kw <= 16 and 3 <= kh <= 16):
                    continue

                best_match: Optional[tuple[float, int, int, int, int]] = None
                for box_sz in candidate_sizes:
                    for dy in range(-8, -1):
                        for dx in range(-8, -1):
                            bx = kx + dx
                            by = ky + dy
                            bw = box_sz
                            bh = box_sz
                            if (
                                bx < 0
                                or by < 0
                                or bx + bw >= roi_hsv.shape[1]
                                or by + bh >= roi_hsv.shape[0]
                            ):
                                continue

                            top_gy = float(np.mean(gy[by : by + 3, bx : bx + bw]))
                            bot_gy = float(
                                np.mean(gy[by + bh - 3 : by + bh, bx : bx + bw])
                            )
                            left_gx = float(np.mean(gx[by : by + bh, bx : bx + 3]))
                            right_gx = float(
                                np.mean(gx[by : by + bh, bx + bw - 3 : bx + bw])
                            )
                            min_grad = min(top_gy, bot_gy, left_gx, right_gx)
                            if min_grad < 120.0:
                                continue

                            b_top = roi_hsv[by : by + 3, bx : bx + bw]
                            b_bot = roi_hsv[by + bh - 3 : by + bh, bx : bx + bw]
                            b_left = roi_hsv[by : by + bh, bx : bx + 3]
                            b_right = roi_hsv[by : by + bh, bx + bw - 3 : bx + bw]

                            o_top = float(
                                np.mean(
                                    (b_top[:, :, 0] >= 35)
                                    & (b_top[:, :, 0] <= 68)
                                    & (b_top[:, :, 1] >= 75)
                                    & (b_top[:, :, 2] >= 85)
                                )
                            )
                            o_bot = float(
                                np.mean(
                                    (b_bot[:, :, 0] >= 35)
                                    & (b_bot[:, :, 0] <= 68)
                                    & (b_bot[:, :, 1] >= 75)
                                    & (b_bot[:, :, 2] >= 85)
                                )
                            )
                            o_left = float(
                                np.mean(
                                    (b_left[:, :, 0] >= 35)
                                    & (b_left[:, :, 0] <= 68)
                                    & (b_left[:, :, 1] >= 75)
                                    & (b_left[:, :, 2] >= 85)
                                )
                            )
                            o_right = float(
                                np.mean(
                                    (b_right[:, :, 0] >= 35)
                                    & (b_right[:, :, 0] <= 68)
                                    & (b_right[:, :, 1] >= 75)
                                    & (b_right[:, :, 2] >= 85)
                                )
                            )
                            o_min = min(o_top, o_bot, o_left, o_right)
                            if o_min < 0.65:
                                continue

                            # Verify interior glowing graphic has high brightness contrast
                            inner_v = v_chan[
                                by + 4 : by + bh - 4, bx + 4 : bx + bw - 4
                            ]
                            if inner_v.size == 0 or float(np.std(inner_v)) < 32.0:
                                continue

                            conf = min(
                                0.96,
                                0.82
                                + min(0.08, (min_grad - 120.0) / 1200.0)
                                + min(0.06, (o_min - 0.65) * 0.25),
                            )
                            if best_match is None or conf > best_match[0]:
                                best_match = (
                                    conf,
                                    int(x1 + bx),
                                    int(y1 + by),
                                    int(bw),
                                    int(bh),
                                )

                if best_match is not None:
                    raw_candidates.append(best_match)

            raw_candidates.sort(key=lambda item: item[0], reverse=True)
            for conf, abs_x, abs_y, bw, bh in raw_candidates:
                cx = abs_x + bw // 2
                cy = abs_y + bh // 2
                if any(
                    abs(cx - d.screen_x) <= max(24, bw)
                    and abs(cy - d.screen_y) <= max(24, bh)
                    for d in boxes
                ):
                    continue

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(cx, cy, w, h)
                boxes.append(
                    DetectedMaterial(
                        material_type="gatherable_node",
                        source="3d_floating_icon",
                        screen_x=int(cx),
                        screen_y=int(cy),
                        bbox=(int(abs_x), int(abs_y), int(bw), int(bh)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=round(float(conf), 3),
                        in_air=False,
                        stack_count=1,
                        harvests_remaining=3,
                    )
                )

            return boxes

        except Exception as exc:
            logger.error("Error in _scan_floating_gather_boxes: %s", exc, exc_info=True)
            return boxes

    def scan_gather_prompt(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
        world_detections: Optional[list[DetectedMaterial]] = None,
    ) -> list[DetectedMaterial]:
        """Detect the interactive 'F Gather' button prompt near the character center.

        Args:
            frame_bgr: BGR game frame.
            hsv: Optional precomputed HSV image.
            world_detections: Optional 3D world detections to link material type.

        Returns:
            List of DetectedMaterial instances with source="gather_prompt".
        """
        prompts: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return prompts

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            scale_x = w / float(self.REF_WIDTH)
            scale_y = h / float(self.REF_HEIGHT)

            # "F Gather" appears slightly right/below character center
            rx1 = int(w * 0.46)
            ry1 = int(h * 0.50)
            rx2 = int(w * 0.66)
            ry2 = int(h * 0.66)

            roi_hsv = hsv[ry1:ry2, rx1:rx2]
            if roi_hsv.size == 0:
                return prompts

            # Support both full brightness ("F Gather" active) and semi-faded ("F Gather" transition)
            white_mask = cv2.inRange(roi_hsv, (0, 0, 140), (180, 48, 255))
            kw = max(10, int(round(18 * scale_x)))
            kh = max(4, int(round(6 * scale_y)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_w = max(42, int(round(56 * scale_x)))
            max_w = max(70, int(round(98 * scale_x)))
            min_h = max(8, int(round(11 * scale_y)))
            max_h = max(20, int(round(30 * scale_y)))
            icon_split = max(16, int(round(24 * scale_x)))

            for cnt in contours:
                fx, fy, fcw, fch = cv2.boundingRect(cnt)
                if fcw < min_w or fcw > max_w or fch < min_h or fch > max_h:
                    continue

                sub_mask = white_mask[fy : fy + fch, fx : fx + fcw]
                total_white = int(np.count_nonzero(sub_mask))
                left_icon_white = int(np.count_nonzero(sub_mask[:, :icon_split]))
                right_text_white = int(np.count_nonzero(sub_mask[:, icon_split:]))

                if left_icon_white < int(round(28 * scale_x * scale_y)):
                    continue
                if right_text_white < int(round(28 * scale_x * scale_y)):
                    continue

                # Verify dark translucent pill background behind "Gather" text
                pill_v = roi_hsv[fy : fy + fch, fx + icon_split : fx + fcw, 2]
                dark_pixels = pill_v[pill_v < 135]
                if dark_pixels.size == 0:
                    continue
                dark_mean = float(np.mean(dark_pixels))
                if not (20.0 <= dark_mean <= 78.0):
                    continue

                abs_x = rx1 + fx
                abs_y = ry1 + fy
                cx = abs_x + fcw // 2
                cy = abs_y + fch // 2

                # Inherit material_type from nearby 3D nameplate if present, else default to Orichalcum Ore
                mat_type = "Orichalcum Ore"
                in_air = False
                if world_detections:
                    nearest = min(
                        world_detections,
                        key=lambda d: math.hypot(d.screen_x - cx, d.screen_y - cy),
                    )
                    mat_type = nearest.material_type
                    in_air = nearest.in_air

                rel_dx, rel_dy, dist_m = self._project_3d_to_map_offset(cx, cy, w, h)
                # When "F Gather" is visible, the player is within interaction range (~1.5m - 2.5m)
                dist_m = round(min(dist_m, 2.2), 2)

                conf = 0.94 if total_white >= 130 else 0.86
                prompts.append(
                    DetectedMaterial(
                        material_type=mat_type,
                        source="gather_prompt",
                        screen_x=int(cx),
                        screen_y=int(cy),
                        bbox=(int(abs_x), int(abs_y), int(fcw), int(fch)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=in_air,
                        stack_count=1,
                        harvests_remaining=3,
                    )
                )

            return prompts

        except Exception as exc:
            logger.error("Error in scan_gather_prompt: %s", exc, exc_info=True)
            return prompts

    def scan_minimap_materials(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Scan the top-right minimap (x: 0.855..0.992, y: 0.065..0.255) for white 3-leaf sprout icons.

        Strictly detects ONLY the white 3-leaf sprout icon ('Gatherable Material') and
        rejects plain circular white monster dots ('•'), white feathers, blue wings,
        player arrows, and minimap UI text/chrome.

        Args:
            frame_bgr: BGR game frame (full screen or minimap closeup crop).
            hsv: Optional precomputed HSV image.

        Returns:
            List of DetectedMaterial instances with source="minimap".
        """
        nodes: list[DetectedMaterial] = []
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                return nodes

            if hsv is None:
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]

            # Support both full-screen frames (w >= 1000) and minimap closeup crops (w < 1000)
            if w >= 1000:
                mx1 = int(round(w * self.MINIMAP_INNER_X1_RATIO))
                my1 = int(round(h * self.MINIMAP_INNER_Y1_RATIO))
                mx2 = int(round(w * self.MINIMAP_INNER_X2_RATIO))
                my2 = int(round(h * self.MINIMAP_INNER_Y2_RATIO))
                templates = [self._sprout_12_tpl]
                if abs(w - 2000) > 150:
                    s_sz = max(9, min(16, int(round(12 * (w / 2000.0)))))
                    if s_sz != 12:
                        templates.append(
                            cv2.resize(
                                self._sprout_12_tpl,
                                (s_sz, s_sz),
                                interpolation=cv2.INTER_AREA,
                            )
                        )
            elif w > 500:
                mx1, my1, mx2, my2 = 359, 108, min(w, 690), min(h, 329)
                templates = [self._sprout_17_tpl]
            else:
                mx1, my1, mx2, my2 = 8, 68, min(w, 339), min(h, 289)
                templates = [self._sprout_17_tpl]

            if self._disk_sprout_tpl is not None:
                templates.append(self._disk_sprout_tpl)

            mm_hsv = hsv[my1:my2, mx1:mx2]
            if mm_hsv.size == 0:
                return nodes

            # Ensure minimap is actually visible and not occluded by a dark menu/overlay
            mean_v = float(np.mean(mm_hsv[:, :, 2]))
            mean_s = float(np.mean(mm_hsv[:, :, 1]))
            if mean_v < 70.0 or mean_s < 18.0:
                return nodes

            mm_h, mm_w = mm_hsv.shape[:2]
            center_x = mm_w / 2.0
            center_y = mm_h / 2.0
            mm_v = mm_hsv[:, :, 2]

            # Valid radar canvas mask (exclude top-right zoom glass, bottom-right clock/1km, and center player arrow)
            valid_mask = np.ones((mm_h, mm_w), dtype=bool)
            valid_mask[: int(mm_h * 0.18), int(mm_w * 0.80) :] = False
            valid_mask[int(mm_h * 0.78) :, int(mm_w * 0.65) :] = False

            raw_matches: list[tuple[float, int, int, int, int]] = []
            for tpl in templates:
                th, tw = tpl.shape[:2]
                if tw >= mm_w or th >= mm_h:
                    continue
                res = cv2.matchTemplate(mm_v, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= 0.68)
                for y_pt, x_pt in zip(ys, xs):
                    raw_matches.append(
                        (float(res[y_pt, x_pt]), int(x_pt), int(y_pt), int(tw), int(th))
                    )

            raw_matches.sort(key=lambda item: item[0], reverse=True)
            for score, nx, ny, nw, nh in raw_matches:
                node_cx = nx + nw / 2.0
                node_cy = ny + nh / 2.0

                if not valid_mask[min(mm_h - 1, int(node_cy)), min(mm_w - 1, int(node_cx))]:
                    continue

                # Exclude center player arrow
                if (
                    abs(node_cx - center_x) < mm_w * 0.06
                    and abs(node_cy - center_y) < mm_h * 0.06
                ):
                    continue

                # Avoid duplicate detections of the same sprout icon
                abs_cx = int(round(mx1 + node_cx))
                abs_cy = int(round(my1 + node_cy))
                if any(
                    abs(abs_cx - d.screen_x) <= max(10, nw)
                    and abs(abs_cy - d.screen_y) <= max(10, nh)
                    for d in nodes
                ):
                    continue

                # Verify sprout is white/silver (low saturation on bright pixels) with dark border contrast
                sub_hsv = mm_hsv[ny : ny + nh, nx : nx + nw]
                sub_v = sub_hsv[:, :, 2]
                sub_s = sub_hsv[:, :, 1]
                white_bright_mask = (sub_v > 175) & (sub_s < 58)
                bright_px = int(np.count_nonzero(white_bright_mask))
                if bright_px < 6:
                    continue

                contrast = float(np.max(sub_v)) - float(np.min(sub_v))
                if contrast < 85.0:
                    continue

                # Reject plain circular white monster dots ('•'): monster dots have top width ≈ bottom width
                # and no central vertical vein split, whereas the 3-leaf sprout widens at the bottom
                top_half_bright = int(np.count_nonzero(white_bright_mask[: nh // 2, :]))
                bot_half_bright = int(np.count_nonzero(white_bright_mask[nh // 2 :, :]))
                if bot_half_bright < top_half_bright:
                    continue

                rel_dx = round(float(node_cx - center_x), 2)
                rel_dy = round(float(node_cy - center_y), 2)
                dist_m = round(
                    math.hypot(rel_dx, rel_dy) * self.meters_per_map_pixel, 2
                )
                conf = round(min(0.96, 0.78 + 0.18 * score), 3)

                nodes.append(
                    DetectedMaterial(
                        material_type="Gatherable Material",
                        source="minimap",
                        screen_x=abs_cx,
                        screen_y=abs_cy,
                        bbox=(int(mx1 + nx), int(my1 + ny), int(nw), int(nh)),
                        rel_map_dx=rel_dx,
                        rel_map_dy=rel_dy,
                        estimated_distance_m=dist_m,
                        confidence=conf,
                        in_air=False,
                        stack_count=1,
                        harvests_remaining=3,
                    )
                )

            return nodes

        except Exception as exc:
            logger.error("Error in scan_minimap_materials: %s", exc, exc_info=True)
            return nodes

    def scan_minimap_nodes(
        self,
        frame_bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> list[DetectedMaterial]:
        """Backward-compatible alias for scan_minimap_materials."""
        return self.scan_minimap_materials(frame_bgr, hsv)

    def _project_3d_to_map_offset(
        self, screen_x: int, screen_y: int, width: int, height: int
    ) -> tuple[float, float, float]:
        """Convert 3D screen coordinates into relative map pixel offsets (dx, dy) and distance (m).

        Uses the player anchor at (W * 0.50, H * 0.54) and applies perspective foreshortening
        so objects higher up toward the horizon correspond to larger map distances.
        """
        player_x = width * self.PLAYER_ANCHOR_X_RATIO
        player_y = height * self.PLAYER_ANCHOR_Y_RATIO

        raw_dx = float(screen_x) - player_x
        raw_dy = float(screen_y) - player_y

        # Normalize to reference resolution (1280x720)
        norm_dx = raw_dx * (self.REF_WIDTH / float(max(1, width)))
        norm_dy = raw_dy * (self.REF_HEIGHT / float(max(1, height)))

        # Perspective depth multiplier increases as screen_y moves toward the upper horizon
        vertical_ratio = max(0.25, min(1.8, (player_y / float(max(1, screen_y))) ** 1.15))

        rel_map_dx = round(norm_dx * self.perspective_scale_x * vertical_ratio, 2)
        rel_map_dy = round(norm_dy * self.perspective_scale_y * vertical_ratio, 2)

        map_pixel_dist = math.hypot(rel_map_dx, rel_map_dy)
        estimated_distance_m = round(
            max(0.5, map_pixel_dist * self.meters_per_map_pixel), 2
        )
        return rel_map_dx, rel_map_dy, estimated_distance_m

    def _classify_3d_material(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        x: int,
        y: int,
        cw: int,
        ch: int,
    ) -> tuple[str, bool]:
        """Classify the 3D resource type across Aion 2's 6 Essence Extraction categories from visual context.

        Categories supported:
        - Herbs: Aria (purple crystal herb), Targena / Calendula (golden-yellow herb)
        - Cooking & Flora: Mela (red-berry/flower bush with green leaves)
        - Gemstones: Sapphire Gemstone (blue crystal), Ruby Gemstone (red crystal), Diamond Gemstone (white crystal)
        - Ores: Orichalcum Ore (copper/amber/metallic rock vein)
        - Logs & Trees: Yggdrasil Log / Asvata Log (woody brown tree trunk)
        - Odyle: Odyle (3D world energy orb, never minimap feathers)
        """
        h, w = frame_bgr.shape[:2]
        in_air = bool(y < int(h * 0.22))

        # Inspect the 3D mesh region directly below the nameplate for crystal/plant/ore/gem/log color signatures
        mesh_y1 = min(h - 1, y + ch + 4)
        mesh_y2 = min(h, y + ch + int(round(75 * (h / float(self.REF_HEIGHT)))))
        mesh_x1 = max(0, x - 20)
        mesh_x2 = min(w, x + cw + 20)

        if mesh_y2 > mesh_y1 and mesh_x2 > mesh_x1:
            mesh_hsv = hsv[mesh_y1:mesh_y2, mesh_x1:mesh_x2]

            # 1. Purple/violet crystal foliage signature for Aria
            aria_purple_mask = cv2.inRange(mesh_hsv, (125, 65, 70), (164, 255, 255))
            aria_purple_px = int(np.count_nonzero(aria_purple_mask))
            if aria_purple_px >= 120:
                return "Aria", in_air

            # 2. Red/coral berry/flower + green leaf bush signature for Mela
            mela_red_mask = cv2.inRange(
                mesh_hsv, (0, 110, 90), (10, 255, 255)
            ) | cv2.inRange(mesh_hsv, (168, 110, 90), (180, 255, 255))
            mela_green_mask = cv2.inRange(mesh_hsv, (36, 65, 50), (75, 255, 220))
            red_px = int(np.count_nonzero(mela_red_mask))
            green_px = int(np.count_nonzero(mela_green_mask))
            if red_px >= 120 and green_px >= 40:
                return "Mela", in_air

            # 3. Gemstones: Sapphire (blue crystal), Ruby (red crystal on rock), Diamond (bright white crystal)
            sapphire_mask = cv2.inRange(mesh_hsv, (105, 130, 130), (130, 255, 255))
            ruby_mask = cv2.inRange(mesh_hsv, (165, 120, 120), (180, 255, 255))
            diamond_mask = cv2.inRange(mesh_hsv, (0, 0, 215), (180, 35, 255))
            sapphire_px = int(np.count_nonzero(sapphire_mask))
            ruby_px = int(np.count_nonzero(ruby_mask))
            diamond_px = int(np.count_nonzero(diamond_mask))

            if sapphire_px >= 25:
                return "Sapphire Gemstone", in_air
            if ruby_px >= 25 and green_px < 20:
                return "Ruby Gemstone", in_air
            if diamond_px >= 65 and green_px < 20:
                return "Diamond Gemstone", in_air

            # 4. Golden-yellow herb with green foliage (Targena / Calendula)
            yellow_herb_mask = cv2.inRange(mesh_hsv, (20, 120, 130), (34, 255, 255))
            if int(np.count_nonzero(yellow_herb_mask)) >= 80 and green_px >= 40:
                return "Targena", in_air

            # 5. Warm copper/amber/orange glow on dark rock for Orichalcum Ore veins
            warm_ore_mask = cv2.inRange(mesh_hsv, (5, 70, 60), (26, 255, 255))
            dark_rock_mask = cv2.inRange(mesh_hsv, (0, 0, 20), (180, 65, 95))
            warm_px = int(np.count_nonzero(warm_ore_mask))
            rock_px = int(np.count_nonzero(dark_rock_mask))
            if warm_px >= 12 and rock_px >= 30:
                return "Orichalcum Ore", in_air

            # 6. Woody brown trunk signature for Yggdrasil Log / Asvata Log
            wood_brown_mask = cv2.inRange(mesh_hsv, (10, 65, 45), (22, 180, 145))
            if int(np.count_nonzero(wood_brown_mask)) >= 140 and rock_px < 25:
                return "Yggdrasil Log", in_air

            if warm_px >= 12:
                return "Orichalcum Ore", in_air

        if in_air:
            return "Odyle", True

        return "Orichalcum Ore", in_air

    def _is_fullscreen_modal_open(
        self, frame_bgr: np.ndarray, hsv: np.ndarray
    ) -> bool:
        """Detect whether a large flat dark UI window (e.g. Gatherion settings/import) covers the screen."""
        h, w = frame_bgr.shape[:2]
        # Check central viewport area where settings dialogs sit
        cx1, cy1 = int(w * 0.15), int(h * 0.12)
        cx2, cy2 = int(w * 0.85), int(h * 0.85)
        center_hsv = hsv[cy1:cy2, cx1:cx2]

        # Flat dark slate UI background has very low saturation (S < 35) and narrow value band (18 <= V <= 48)
        slate_mask = cv2.inRange(center_hsv, (0, 0, 18), (180, 35, 48))
        slate_ratio = float(np.count_nonzero(slate_mask)) / float(
            max(1, center_hsv.shape[0] * center_hsv.shape[1])
        )
        return slate_ratio > 0.52
