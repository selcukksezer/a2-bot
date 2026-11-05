# -*- coding: utf-8 -*-
r"""
Dataset Factory Sub-Agent 2: Multi-Class 3D + Minimap Automatic Bounding Box & YOLO `.txt` Labeler
==================================================================================================
Automatically generates high-precision, normalized YOLO format labels
(`class_id cx_norm cy_norm w_norm h_norm`, all values strictly in `[0.000000, 1.000000]`)
from any game frame + optional gather event metadata.

Combines:
- `VisionEngine` (3D white winged gather crest `\Y/`, `[F Gather]` prompt pill, floating `[E]`/`[F]` box)
- `MaterialScanner` (3D world nameplates, floating boxes, gather prompts, minimap sprouts, monster nameplate checks)
- `MinimapScanner` (3-leaf sprout `minimap_sprout` icons & circular white `minimap_monster_dot` negative markers)
- `classify_3d_material_patch` + HSV saturation/value contour mesh segmentation directly below any detected
  `world_gather_crest` (`x - 36..x + 36, y + 18..y + 115`) or around floating `[E]`/`[F]` icons to bound and
  classify the 3D Plant/Ore/Gem/Log mesh (`0..5`).
- Dedicated 3D monster & nameplate detector (`10: monster_3d`, e.g. `< Yellow-Maned Zaif >`,
  `Highland Odyle Sparkie`) so the YOLO detector learns both positive gatherables and negative obstacles/monsters.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from config.materials_db import classify_3d_material_patch, is_rejected_non_material
from core.material_scanner import DetectedMaterial, MaterialScanner
from core.minimap_scanner import MinimapScanner
from core.vision import VisionEngine

logger = logging.getLogger(__name__)

# ==============================================================================
# 12 Official Gatherion YOLO Classes (English & Turkish)
# ==============================================================================
YOLO_CLASSES: Dict[int, str] = {
    0: "herb_aria",
    1: "herb_mela",
    2: "herb_targena",
    3: "ore_orichalcum",
    4: "gem_crystal",
    5: "tree_log",
    6: "world_gather_crest",
    7: "f_gather_prompt",
    8: "e_floating_icon",
    9: "minimap_sprout",
    10: "monster_3d",
    11: "minimap_monster_dot",
}

YOLO_CLASSES_TR: Dict[int, str] = {
    0: "Aria / Mor Kristal Bitki",
    1: "Mela / Kırmızı Meyveli Çalı",
    2: "Targena / Azpha / Calendula Bitkisi",
    3: "Orichalcum / Maden Cevheri",
    4: "Sapphire / Ruby / Diamond Gem Taşı",
    5: "Yggdrasil / Asvata Ağaç Kerestesi",
    6: r"3B Beyaz Kanatlı Toplama Amblemi \Y/",
    7: "[F Gather] Etkileşim Butonu",
    8: "[E] / [F] Yüzen Kaynak Kutusu",
    9: "Minimap 3 Yapraklı Filiz İkonu",
    10: "3B Canavar & İsim Plakası - Negatif Sınıf",
    11: "Minimap Yuvarlak Canavar Noktası • - Negatif Sınıf",
}

YOLO_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in YOLO_CLASSES.items()}

POSITIVE_CLASS_IDS: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
NEGATIVE_CLASS_IDS: Sequence[int] = (10, 11)

# Distinct BGR colors for UI preview rendering per class_id
YOLO_CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (255, 80, 215),   # herb_aria: vibrant purple/magenta
    1: (60, 75, 255),    # herb_mela: bright coral red
    2: (50, 230, 120),   # herb_targena: emerald/golden green
    3: (40, 175, 255),   # ore_orichalcum: warm amber/copper
    4: (255, 215, 45),   # gem_crystal: cyan/sapphire blue
    5: (55, 135, 195),   # tree_log: woody bronze
    6: (255, 255, 110),  # world_gather_crest: bright sky-cyan
    7: (85, 255, 85),    # f_gather_prompt: lime green
    8: (65, 240, 200),   # e_floating_icon: mint/teal
    9: (120, 255, 120),  # minimap_sprout: bright green
    10: (35, 35, 235),   # monster_3d: danger crimson red
    11: (70, 95, 250),   # minimap_monster_dot: orange-red negative dot
}


def map_material_name_to_yolo_class_id(material_name: str, default_id: int = 3) -> int:
    """
    Map any canonical or raw Aion 2 material name / category string to one of the
    6 3D resource mesh YOLO class IDs (`0..5`), or UI/monster classes (`6..11`) if specified.
    """
    if not material_name or not isinstance(material_name, str):
        return int(default_id)

    norm = material_name.strip().lower()
    if not norm:
        return int(default_id)

    # Direct match against YOLO_CLASSES names
    if norm in YOLO_NAME_TO_ID:
        return YOLO_NAME_TO_ID[norm]

    # Negative / monster keywords
    if any(k in norm for k in ("monster_dot", "minimap_monster", "yuvarlak canavar")):
        return 11
    if any(k in norm for k in ("monster", "zaif", "sparkie", "skyray", "canavar", "hostile", "mob")):
        return 10

    # UI / Icon classes
    if "crest" in norm or "amblem" in norm:
        return 6
    if "f_gather" in norm or "gather_prompt" in norm or "f gather" in norm:
        return 7
    if "floating" in norm or "e_floating" in norm or "e_gather" in norm:
        return 8
    if "sprout" in norm or "filiz" in norm or norm == "minimap":
        return 9

    # 3D Mesh classes (0..5)
    if "aria" in norm or "pella" in norm or "valerian" in norm or "amilis" in norm:
        return 0
    if "mela" in norm or "berry" in norm or "meyve" in norm or "çalı" in norm:
        return 1
    if any(
        k in norm
        for k in (
            "targena",
            "azpha",
            "calendula",
            "lura",
            "gravi",
            "leaf",
            "miana",
            "rina",
            "mandragora",
            "atri",
            "kalgelly",
            "lolo",
            "marle",
            "caprauna",
            "herb",
            "bitki",
            "ot",
        )
    ):
        return 2
    if any(
        k in norm
        for k in (
            "sapphire",
            "ruby",
            "diamond",
            "gem",
            "crystal",
            "lazuli",
            "garnet",
            "moonstone",
            "topaz",
            "aquamarine",
            "opal",
            "emerald",
            "amethyst",
            "odyle",
            "aether",
            "lunime",
            "kristal",
        )
    ):
        return 4
    if any(
        k in norm
        for k in (
            "yggdrasil",
            "asvata",
            "log",
            "tree",
            "wood",
            "branch",
            "acacia",
            "cypress",
            "elm",
            "oak",
            "karpaso",
            "nepenthes",
            "kereste",
            "ağaç",
        )
    ):
        return 5
    if any(
        k in norm
        for k in (
            "orichalcum",
            "iron",
            "titanium",
            "adamantium",
            "copper",
            "silver",
            "gold",
            "platinum",
            "mithril",
            "catalium",
            "ore",
            "maden",
            "cevher",
        )
    ):
        return 3

    return int(default_id)


class YOLOAutoLabeler:
    r"""
    Multi-Class 3D + Minimap Automatic Bounding Box & YOLO `.txt` Labeler.

    Automatically detects and generates normalized `[0.000000, 1.000000]` YOLO bounding boxes for:
    - `0..5`: 3D resource meshes (`herb_aria`, `herb_mela`, `herb_targena`, `ore_orichalcum`,
              `gem_crystal`, `tree_log`) segmented below every `world_gather_crest` (`x-36..x+36, y+18..y+115`)
              and around floating `[E]`/`[F]` resource boxes.
    - `6`: `world_gather_crest` (`3B Beyaz Kanatlı Toplama Amblemi \Y/`)
    - `7`: `f_gather_prompt` (`[F Gather] Etkileşim Butonu`)
    - `8`: `e_floating_icon` (`[E] / [F] Yüzen Kaynak Kutusu`)
    - `9`: `minimap_sprout` (`Minimap 3 Yapraklı Filiz İkonu`)
    - `10`: `monster_3d` (`3B Canavar & İsim Plakası - Negatif Sınıf`)
    - `11`: `minimap_monster_dot` (`Minimap Yuvarlak Canavar Noktası • - Negatif Sınıf`)
    """

    def get_vision_filters_status(self) -> Dict[str, bool]:
        """
        Returns a dictionary showing whether `FeatherNegativeFilter`,
        `Pyramid3DCrestDetector`, and `MonsterCircularityFilter` are available,
        wrapped in try/except blocks.
        """
        status = {
            "FeatherNegativeFilter": False,
            "Pyramid3DCrestDetector": False,
            "MonsterCircularityFilter": False,
        }
        try:
            from core.vision_filters import FeatherNegativeFilter
            status["FeatherNegativeFilter"] = FeatherNegativeFilter is not None
        except Exception:
            pass

        try:
            from core.vision_filters import Pyramid3DCrestDetector
            status["Pyramid3DCrestDetector"] = Pyramid3DCrestDetector is not None
        except Exception:
            pass

        try:
            from core.vision_filters import MonsterCircularityFilter
            status["MonsterCircularityFilter"] = MonsterCircularityFilter is not None
        except Exception:
            pass

        return status

    def __init__(
        self,
        vision_engine: Optional[VisionEngine] = None,
        material_scanner: Optional[MaterialScanner] = None,
        minimap_scanner: Optional[MinimapScanner] = None,
        crest_threshold: float = 0.72,
        prompt_threshold: float = 0.68,
        floating_icon_threshold: float = 0.65,
    ) -> None:
        self.vision_engine: VisionEngine = (
            vision_engine if vision_engine is not None else VisionEngine()
        )
        self.material_scanner: MaterialScanner = (
            material_scanner if material_scanner is not None else MaterialScanner()
        )
        self.minimap_scanner: MinimapScanner = (
            minimap_scanner if minimap_scanner is not None else MinimapScanner()
        )
        self.crest_threshold: float = float(crest_threshold)
        self.prompt_threshold: float = float(prompt_threshold)
        self.floating_icon_threshold: float = float(floating_icon_threshold)

    # ------------------------------------------------------------------
    # Coordinate Normalization & Bounding Box Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_bgr_frame(frame: Union[np.ndarray, str, os.PathLike]) -> Optional[np.ndarray]:
        """Convert path or ndarray input into a valid 3-channel BGR uint8 image."""
        if frame is None:
            return None
        if isinstance(frame, (str, os.PathLike)):
            path_str = str(frame)
            if not os.path.isfile(path_str):
                return None
            raw = np.fromfile(path_str, dtype=np.uint8)
            if raw.size == 0:
                return None
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    @staticmethod
    def _pixel_xywh_to_yolo_norm(
        x: float,
        y: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float, float, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Clamp pixel bounding box `(x, y, w, h)` to image bounds `(img_w, img_h)` and compute
        strictly normalized `(cx_norm, cy_norm, w_norm, h_norm)` in `[0.000000, 1.000000]`.
        """
        fw = float(max(1, img_w))
        fh = float(max(1, img_h))

        x1 = float(np.clip(x, 0.0, fw - 1.0))
        y1 = float(np.clip(y, 0.0, fh - 1.0))
        x2 = float(np.clip(x + max(1.0, w), x1 + 1.0, fw))
        y2 = float(np.clip(y + max(1.0, h), y1 + 1.0, fh))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + 0.5 * bw
        cy = y1 + 0.5 * bh

        cx_norm = float(np.clip(cx / fw, 0.0, 1.0))
        cy_norm = float(np.clip(cy / fh, 0.0, 1.0))
        w_norm = float(np.clip(bw / fw, 0.0, 1.0))
        h_norm = float(np.clip(bh / fh, 0.0, 1.0))

        # Guarantee [cx ± w/2, cy ± h/2] stays within [0.0, 1.0]
        nx1 = float(np.clip(cx_norm - 0.5 * w_norm, 0.0, 1.0))
        ny1 = float(np.clip(cy_norm - 0.5 * h_norm, 0.0, 1.0))
        nx2 = float(np.clip(cx_norm + 0.5 * w_norm, 0.0, 1.0))
        ny2 = float(np.clip(cy_norm + 0.5 * h_norm, 0.0, 1.0))

        w_norm = round(float(np.clip(nx2 - nx1, 0.0005, 1.0)), 6)
        h_norm = round(float(np.clip(ny2 - ny1, 0.0005, 1.0)), 6)
        cx_norm = round(float(np.clip(nx1 + 0.5 * w_norm, 0.0, 1.0)), 6)
        cy_norm = round(float(np.clip(ny1 + 0.5 * h_norm, 0.0, 1.0)), 6)

        px_xywh = (int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh)))
        px_xyxy = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        return cx_norm, cy_norm, w_norm, h_norm, px_xywh, px_xyxy

    def _make_annotation(
        self,
        class_id: int,
        x: float,
        y: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
        confidence: float = 0.90,
        source: str = "auto_labeler",
        material_name: str = "",
    ) -> Dict[str, Any]:
        """Construct a standardized YOLO annotation dictionary with normalized and pixel bounds."""
        cid = int(class_id)
        if cid not in YOLO_CLASSES:
            cid = 3

        cx_n, cy_n, w_n, h_n, px_xywh, px_xyxy = self._pixel_xywh_to_yolo_norm(
            x, y, w, h, img_w, img_h
        )
        cls_name = YOLO_CLASSES[cid]
        cls_name_tr = YOLO_CLASSES_TR[cid]
        conf_clamped = round(float(np.clip(confidence, 0.01, 1.0)), 4)

        return {
            "class_id": cid,
            "class_name": cls_name,
            "class_name_tr": cls_name_tr,
            "material_name": material_name or cls_name,
            "cx": cx_n,
            "cy": cy_n,
            "w": w_n,
            "h": h_n,
            "width": w_n,
            "height": h_n,
            "bbox_norm": (cx_n, cy_n, w_n, h_n),
            "yolo_bbox": [cx_n, cy_n, w_n, h_n],
            "bbox": px_xywh,
            "bbox_xywh": px_xywh,
            "bbox_xyxy": list(px_xyxy),
            "confidence": conf_clamped,
            "source": str(source),
            "is_negative": bool(cid in NEGATIVE_CLASS_IDS),
        }

    @staticmethod
    def _compute_iou_xyxy(
        box_a: Sequence[Union[int, float]],
        box_b: Sequence[Union[int, float]],
    ) -> float:
        """Compute Intersection-over-Union (IoU) between two `(x1, y1, x2, y2)` pixel boxes."""
        ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return float(inter / (area_a + area_b - inter + 1e-6))

    def _deduplicate_annotations(
        self,
        annotations: List[Dict[str, Any]],
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Apply per-class Non-Maximum Suppression (NMS) so overlapping detections of the same
        object from multiple sub-scanners are merged into the highest-confidence bounding box.
        """
        if not annotations:
            return []

        ordered = sorted(
            annotations,
            key=lambda a: (int(a["class_id"]), -float(a.get("confidence", 0.0))),
        )
        kept: List[Dict[str, Any]] = []

        for cand in ordered:
            cid = int(cand["class_id"])
            c_xyxy = cand["bbox_xyxy"]
            c_cx = 0.5 * (c_xyxy[0] + c_xyxy[2])
            c_cy = 0.5 * (c_xyxy[1] + c_xyxy[3])

            duplicate = False
            for existing in kept:
                if int(existing["class_id"]) != cid:
                    continue
                e_xyxy = existing["bbox_xyxy"]
                iou = self._compute_iou_xyxy(c_xyxy, e_xyxy)
                if iou >= iou_threshold:
                    duplicate = True
                    break
                # Also merge if centers are extremely close for the same class
                e_cx = 0.5 * (e_xyxy[0] + e_xyxy[2])
                e_cy = 0.5 * (e_xyxy[1] + e_xyxy[3])
                min_dim = max(8.0, min(c_xyxy[2] - c_xyxy[0], c_xyxy[3] - c_xyxy[1]) * 0.65)
                if math.hypot(c_cx - e_cx, c_cy - e_cy) <= min_dim:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(cand)

        return kept

    # ------------------------------------------------------------------
    # 1. 3D Mesh Segmentation & Classification Below `world_gather_crest`
    # ------------------------------------------------------------------
    def _segment_and_classify_3d_mesh_below_crest(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        crest_cx: int,
        crest_cy: int,
        crest_bbox: Tuple[int, int, int, int],
        hint_material_name: str = "",
    ) -> Dict[str, Any]:
        """
        Given a detected `world_gather_crest` centered at `(crest_cx, crest_cy)`:
        1. Extracts the 3D mesh region directly below the crest (`x - 36..x + 36, y + 18..y + 115`,
           scaled proportionally to frame resolution and extended slightly downward when a tall crystal/bush
           continues below `y + 115`).
        2. Classifies the 3D mesh into one of the 6 material classes (`0..5`) via
           `classify_3d_material_patch` and `hint_material_name`.
        3. Refines the bounding box around the 3D plant/ore/gem/log using class-specific HSV
           saturation/value contour segmentation, falling back to the canonical
           `(x - 36..x + 36, y + 18..y + 115)` window if background lighting is low-contrast.
        """
        img_h, img_w = frame_bgr.shape[:2]
        scale_x = max(0.55, min(1.5, img_w / 2000.0))
        scale_y = max(0.55, min(1.5, img_h / 1124.0))

        # Canonical window requested in mission spec: x - 36..x + 36, y + 18..y + 115
        half_w = max(24, int(round(36 * scale_x)))
        y_off_top = max(12, int(round(18 * scale_y)))
        y_off_bot = max(64, int(round(115 * scale_y)))

        win_x1 = max(0, int(crest_cx - half_w))
        win_x2 = min(img_w, int(crest_cx + half_w))
        win_y1 = max(0, min(img_h - 2, int(crest_cy + y_off_top)))
        win_y2 = max(win_y1 + 2, min(img_h, int(crest_cy + y_off_bot)))

        # Extended search window (down to y + 152) to capture the full base of tall Aria crystals / Mela bushes
        ext_y1 = max(0, min(img_h - 2, int(crest_cy + int(round(32 * scale_y)))))
        ext_y2 = max(ext_y1 + 2, min(img_h, int(crest_cy + int(round(152 * scale_y)))))
        ext_x1 = max(0, int(crest_cx - int(round(44 * scale_x))))
        ext_x2 = min(img_w, int(crest_cx + int(round(44 * scale_x))))

        canonical_crop = frame_bgr[win_y1:win_y2, win_x1:win_x2]
        extended_crop = frame_bgr[ext_y1:ext_y2, ext_x1:ext_x2]

        patch_info = classify_3d_material_patch(
            canonical_crop if canonical_crop.size > 0 else extended_crop,
            label_width_px=int(crest_bbox[2]),
        )
        detected_mat_name = str(patch_info.get("name", "Orichalcum Ore"))
        patch_conf = float(patch_info.get("confidence", 0.85))

        # If caller or MaterialScanner provided a specific material name, honor it when consistent
        if hint_material_name and not is_rejected_non_material(hint_material_name):
            hint_low = hint_material_name.strip().lower()
            if hint_low not in ("gatherable material", "gatherable_node", "resource", "unknown"):
                hint_cid = map_material_name_to_yolo_class_id(hint_material_name, default_id=-1)
                patch_cid = map_material_name_to_yolo_class_id(detected_mat_name, default_id=3)
                # Prefer visual Aria/Mela if high confidence, otherwise use hint
                if patch_cid in (0, 1) and patch_conf >= 0.90:
                    mesh_class_id = patch_cid
                elif 0 <= hint_cid <= 5:
                    mesh_class_id = hint_cid
                    detected_mat_name = hint_material_name
                else:
                    mesh_class_id = patch_cid
            else:
                mesh_class_id = map_material_name_to_yolo_class_id(detected_mat_name, default_id=3)
        else:
            mesh_class_id = map_material_name_to_yolo_class_id(detected_mat_name, default_id=3)

        # Refine 3D mesh bounding box via HSV saturation/value contour segmentation
        roi_hsv = hsv[ext_y1:ext_y2, ext_x1:ext_x2]
        if roi_hsv.size == 0:
            return self._make_annotation(
                class_id=mesh_class_id,
                x=win_x1,
                y=win_y1,
                w=win_x2 - win_x1,
                h=win_y2 - win_y1,
                img_w=img_w,
                img_h=img_h,
                confidence=patch_conf,
                source="3d_mesh_below_crest",
                material_name=detected_mat_name,
            )

        # Exclude cyan nameplate text row at the top of the ROI (first ~16px below crest)
        text_cut = max(0, min(roi_hsv.shape[0] // 3, int(round(16 * scale_y))))
        seg_mask = self._build_class_hsv_mesh_mask(roi_hsv, mesh_class_id)
        if text_cut > 0:
            seg_mask[:text_cut, :] = 0

        refined_box = self._extract_refined_mesh_bbox(
            seg_mask=seg_mask,
            roi_offset_x=ext_x1,
            roi_offset_y=ext_y1,
            fallback_box=(win_x1, win_y1, win_x2 - win_x1, win_y2 - win_y1),
            scale_x=scale_x,
            scale_y=scale_y,
        )

        rx, ry, rw, rh = refined_box
        return self._make_annotation(
            class_id=mesh_class_id,
            x=rx,
            y=ry,
            w=rw,
            h=rh,
            img_w=img_w,
            img_h=img_h,
            confidence=max(0.84, patch_conf),
            source="3d_mesh_below_crest",
            material_name=detected_mat_name,
        )

    @staticmethod
    def _build_class_hsv_mesh_mask(roi_hsv: np.ndarray, mesh_class_id: int) -> np.ndarray:
        """Build an HSV foreground segmentation mask tailored to the 3D resource class (`0..5`)."""
        if mesh_class_id == 0:  # herb_aria: purple/violet crystal petals
            mask = cv2.inRange(roi_hsv, np.array([120, 55, 65]), np.array([168, 255, 255]))
        elif mesh_class_id == 1:  # herb_mela: red/coral berries + green bush foliage
            red1 = cv2.inRange(roi_hsv, np.array([0, 95, 75]), np.array([12, 255, 255]))
            red2 = cv2.inRange(roi_hsv, np.array([166, 95, 75]), np.array([180, 255, 255]))
            green = cv2.inRange(roi_hsv, np.array([32, 55, 45]), np.array([78, 255, 235]))
            mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), green)
        elif mesh_class_id == 2:  # herb_targena / azpha / calendula
            gold_green = cv2.inRange(roi_hsv, np.array([18, 70, 75]), np.array([72, 255, 255]))
            marigold = cv2.inRange(roi_hsv, np.array([10, 110, 120]), np.array([24, 255, 255]))
            azpha = cv2.inRange(roi_hsv, np.array([105, 65, 85]), np.array([136, 255, 255]))
            mask = cv2.bitwise_or(cv2.bitwise_or(gold_green, marigold), azpha)
        elif mesh_class_id == 4:  # gem_crystal: sapphire, ruby, diamond, odyle
            sapphire = cv2.inRange(roi_hsv, np.array([92, 100, 125]), np.array([128, 255, 255]))
            ruby = cv2.inRange(roi_hsv, np.array([165, 115, 115]), np.array([180, 255, 255]))
            diamond = cv2.inRange(roi_hsv, np.array([80, 10, 195]), np.array([120, 70, 255]))
            odyle = cv2.inRange(roi_hsv, np.array([56, 100, 135]), np.array([95, 255, 255]))
            mask = cv2.bitwise_or(cv2.bitwise_or(sapphire, ruby), cv2.bitwise_or(diamond, odyle))
        elif mesh_class_id == 5:  # tree_log: woody bark + canopy
            bark = cv2.inRange(roi_hsv, np.array([8, 45, 35]), np.array([23, 175, 155]))
            canopy = cv2.inRange(roi_hsv, np.array([30, 50, 45]), np.array([78, 255, 235]))
            mask = cv2.bitwise_or(bark, canopy)
        else:  # 3: ore_orichalcum (metallic copper/amber/teal vein on dark rock)
            warm_vein = cv2.inRange(roi_hsv, np.array([5, 65, 55]), np.array([26, 245, 245]))
            teal_vein = cv2.inRange(roi_hsv, np.array([72, 45, 65]), np.array([98, 220, 235]))
            dark_rock = cv2.inRange(roi_hsv, np.array([0, 0, 22]), np.array([180, 65, 95]))
            mask = cv2.bitwise_or(cv2.bitwise_or(warm_vein, teal_vein), dark_rock)

        # Fallback: if specific mask has very few pixels, segment high-saturation/contrast foreground
        if int(np.count_nonzero(mask)) < 35:
            s_chan = roi_hsv[:, :, 1]
            v_chan = roi_hsv[:, :, 2]
            s_med = float(np.median(s_chan))
            v_med = float(np.median(v_chan))
            fg = ((s_chan > max(65.0, s_med + 18.0)) | (np.abs(v_chan.astype(np.float32) - v_med) > 36.0))
            mask = (fg.astype(np.uint8)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        return mask

    @staticmethod
    def _extract_refined_mesh_bbox(
        seg_mask: np.ndarray,
        roi_offset_x: int,
        roi_offset_y: int,
        fallback_box: Tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> Tuple[int, int, int, int]:
        """Extract tight contour bounding box from `seg_mask` with morphological padding."""
        cnts, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(24.0, 45.0 * scale_x * scale_y)
        valid_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

        if not valid_cnts:
            return fallback_box

        # Combine contours near the horizontal center of the ROI
        rh, rw = seg_mask.shape[:2]
        center_x = rw * 0.5
        selected_pts: List[np.ndarray] = []
        for cnt in valid_cnts:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            ccx = bx + bw * 0.5
            if abs(ccx - center_x) <= rw * 0.42:
                selected_pts.append(cnt)

        if not selected_pts:
            selected_pts = [max(valid_cnts, key=cv2.contourArea)]

        merged = np.vstack(selected_pts)
        mx, my, mw, mh = cv2.boundingRect(merged)

        # Ensure minimum reasonable mesh dimensions and add small 4px context margin
        pad_x = max(3, int(round(5 * scale_x)))
        pad_y = max(3, int(round(5 * scale_y)))
        x1 = max(0, mx - pad_x)
        y1 = max(0, my - pad_y)
        x2 = min(rw, mx + mw + pad_x)
        y2 = min(rh, my + mh + pad_y)

        final_w = max(int(round(28 * scale_x)), x2 - x1)
        final_h = max(int(round(32 * scale_y)), y2 - y1)
        abs_x = roi_offset_x + x1
        abs_y = roi_offset_y + y1
        return int(abs_x), int(abs_y), int(final_w), int(final_h)

    # ------------------------------------------------------------------
    # 2. 3D Monster & Nameplate Detector (`class_id = 10: monster_3d`)
    # ------------------------------------------------------------------
    def _scan_3d_monsters(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        existing_positive_boxes: Sequence[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        """
        Detect 3D monster nameplates and bodies in the 3D viewport (`class_id = 10: monster_3d`),
        such as:
        - Targeted monsters with cyan diamond bracket tips + yellow text + green/red health bar +
          red `'4'` agro badge (e.g. `< Yellow-Maned Zaif >` in `pilgrims_ridge_1.jpg`).
        - Untargeted yellow/olive monster nameplates in the 3D world (e.g. `Highland Odyle Sparkie`
          in `aria_3d_crest_far.jpg` and `mela_f_gather_not_on_minimap.jpg`).
        Strictly excludes:
        - The player's own central nameplate (`Two-Faced / Muses`),
        - HUD/UI bars (top boss bar, right quest tracker, minimap, bottom skill/chat bars),
        - Any positive gatherable crest/resource already detected.
        """
        monsters: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 600 or img_h < 360:
            return monsters

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # High-contrast stroke filter (Top-Hat) to separate thin text glyphs from sunlit ground/rocks
        th_sz = max(5, int(round(7 * scale_x)) | 1)
        tophat = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (th_sz, th_sz)),
        )

        # Monster nameplate yellow/olive text: Hue 21..30, Saturation >= 132, Value >= 145
        # (Sunlit rocks/dry grass have Hue <= 20 or low saturation/stroke contrast)
        yellow_stroke = (
            (hsv[:, :, 0] >= 21)
            & (hsv[:, :, 0] <= 30)
            & (hsv[:, :, 1] >= 132)
            & (hsv[:, :, 2] >= 145)
            & (tophat >= 10)
        )
        ym_u8 = (yellow_stroke.astype(np.uint8)) * 255

        # Mask out non-3D-viewport HUD areas:
        # 1) Top HUD & target boss bar (y < 0.11 * h)
        ym_u8[: int(img_h * 0.11), :] = 0
        # 2) Bottom skill bar & XP bar (y > 0.78 * h)
        ym_u8[int(img_h * 0.78) :, :] = 0
        # 3) Right-side Quest Tracker & Minimap (x > 0.852 * w, and y < 0.30 * h for x > 0.71 * w)
        ym_u8[:, int(img_w * 0.852) :] = 0
        ym_u8[: int(img_h * 0.30), int(img_w * 0.71) :] = 0
        # 4) Bottom-left chat box (y > 0.66 * h and x < 0.26 * w)
        ym_u8[int(img_h * 0.66) :, : int(img_w * 0.26)] = 0
        # 5) Player's own central nameplate ("Two-Faced / Muses" at x: 0.44..0.56, y: 0.42..0.58)
        ym_u8[
            int(img_h * 0.42) : int(img_h * 0.58),
            int(img_w * 0.44) : int(img_w * 0.56),
        ] = 0

        kw = max(10, int(round(16 * scale_x)))
        kh = max(3, int(round(4 * scale_y)))
        closed = cv2.morphologyEx(
            ym_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
        )
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = max(72, int(round(115 * scale_x)))
        max_w = max(160, int(round(250 * scale_x)))
        min_h = max(6, int(round(7 * scale_y)))
        max_h = max(20, int(round(28 * scale_y)))

        for cnt in cnts:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if not (min_w <= cw <= max_w and min_h <= ch <= max_h):
                continue

            aspect = float(cw) / float(max(1, ch))
            if aspect < 4.5:
                continue

            sub_ym = ym_u8[y : y + ch, x : x + cw]
            stroke_px = int(np.count_nonzero(sub_ym))
            min_stroke = max(40, int(round(105 * scale_x * scale_y)))

            # Check for cyan diamond bracket tips (`< Monster >`) at left and right ends of targeted monster nameplate
            lx1 = max(0, x - int(round(26 * scale_x)))
            lx2 = min(img_w, x + int(round(6 * scale_x)))
            rx1 = max(0, x + cw - int(round(6 * scale_x)))
            rx2 = min(img_w, x + cw + int(round(26 * scale_x)))
            ty1 = max(0, y - int(round(6 * scale_y)))
            ty2 = min(img_h, y + ch + int(round(6 * scale_y)))

            left_cyan = cv2.inRange(hsv[ty1:ty2, lx1:lx2], np.array([78, 80, 140]), np.array([105, 255, 255]))
            right_cyan = cv2.inRange(hsv[ty1:ty2, rx1:rx2], np.array([78, 80, 140]), np.array([105, 255, 255]))
            has_diamond_tips = (
                int(np.count_nonzero(left_cyan)) >= max(8, int(round(12 * scale_x * scale_y)))
                and int(np.count_nonzero(right_cyan)) >= max(8, int(round(12 * scale_x * scale_y)))
            )

            # Check for centered red agro badge ('4') directly below the nameplate
            cx_label = x + cw // 2
            bx1 = max(0, cx_label - int(round(18 * scale_x)))
            bx2 = min(img_w, cx_label + int(round(18 * scale_x)))
            by1 = min(img_h, y + ch)
            by2 = min(img_h, y + ch + int(round(48 * scale_y)))
            below_hsv = hsv[by1:by2, bx1:bx2]
            red1 = cv2.inRange(below_hsv, np.array([0, 135, 95]), np.array([8, 255, 255]))
            red2 = cv2.inRange(below_hsv, np.array([170, 135, 95]), np.array([180, 255, 255]))
            red_badge_px = int(np.count_nonzero(red1 | red2))
            has_agro_badge = has_diamond_tips or (
                red_badge_px >= max(24, int(round(45 * scale_x * scale_y)))
            )

            # Count distinct character-like connected components inside the text line
            num_labels, _ = cv2.connectedComponents(sub_ym)
            if not has_agro_badge and (stroke_px < min_stroke or num_labels < 7):
                continue

            # Verify dark text outline/shadow pixels exist immediately around the yellow glyphs
            dilated_stroke = cv2.dilate(sub_ym, np.ones((3, 3), dtype=np.uint8))
            border_ring = dilated_stroke - sub_ym
            ring_gray = gray[y : y + ch, x : x + cw][border_ring > 0]
            if ring_gray.size == 0:
                continue
            dark_outline_ratio = float(np.mean(ring_gray < 120))
            if not has_agro_badge and dark_outline_ratio < 0.20:
                continue

            # Expand bounding box to cover diamond tips + monster body below nameplate if targeted/close
            if has_agro_badge:
                # Targeted monster like `< Yellow-Maned Zaif >`: include diamond tips & 3D creature body below
                final_x = max(0, x - int(round(24 * scale_x)))
                final_y = max(0, y - int(round(8 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(48 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(105 * scale_y)))
                conf = 0.95
            else:
                # Untargeted monster nameplate (`Highland Odyle Sparkie`)
                final_x = max(0, x - int(round(6 * scale_x)))
                final_y = max(0, y - int(round(4 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(12 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(8 * scale_y)))
                conf = round(min(0.92, 0.78 + min(0.14, stroke_px / 900.0)), 4)

            cand_xyxy = (final_x, final_y, final_x + final_w, final_y + final_h)
            if any(self._compute_iou_xyxy(cand_xyxy, pos_box) >= 0.25 for pos_box in existing_positive_boxes):
                continue

            monsters.append(
                self._make_annotation(
                    class_id=10,
                    x=final_x,
                    y=final_y,
                    w=final_w,
                    h=final_h,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=conf,
                    source="3d_monster_scanner",
                    material_name="monster_3d",
                )
            )

        return monsters

    # ------------------------------------------------------------------
    # 3. Core Frame Auto-Labeling Pipeline
    # ------------------------------------------------------------------
    def auto_label_frame(
        self,
        frame: np.ndarray,
        extra_detections: Optional[List[Dict[str, Any]]] = None,
        known_target_xy: Optional[Tuple[int, int]] = None,
        known_material_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Automatically detect and label all 12 Gatherion YOLO classes in `frame`:
        1. Every `world_gather_crest` (`class_id = 6`) AND the **3D Plant/Ore/Gem Mesh**
           directly below the crest (`x - 36..x + 36, y + 18..y + 115`, refined via HSV
           saturation/value contour segmentation) classified into:
           `0: herb_aria`, `1: herb_mela`, `2: herb_targena`, `3: ore_orichalcum`,
           `4: gem_crystal`, or `5: tree_log`.
        2. Any `[F Gather]` prompt (`class_id = 7`) and `[E]` floating resource box (`class_id = 8`),
           plus the 3D resource mesh under/around the floating `[E]` box.
        3. Every top-right minimap 3-leaf sprout icon (`class_id = 9: minimap_sprout`) AND
           round white monster dot (`class_id = 11: minimap_monster_dot`).
        4. Any 3D monster nameplate like `Yellow-Maned Zaif` or `Highland Odyle Sparkie`
           (`class_id = 10: monster_3d`).
        """
        bgr = self._coerce_bgr_frame(frame)
        if bgr is None or bgr.size == 0:
            return []

        img_h, img_w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        annotations: List[Dict[str, Any]] = []

        # Run MaterialScanner for holistic 3D world, floating box, gather prompt & minimap detections
        ms_detections: List[DetectedMaterial] = self.material_scanner.scan_frame_for_materials(bgr)

        # ------------------------------------------------------------------
        # Stage 1: 3D White Winged Gather Crests (`6: world_gather_crest`)
        #          + 3D Plant/Ore/Gem Mesh Below Crest (`0..5`)
        # ------------------------------------------------------------------
        ve_crests = self.vision_engine.detect_3d_gather_crests(
            bgr, threshold=self.crest_threshold
        )

        # Merge crests from VisionEngine and MaterialScanner (`source == "3d_world"`)
        merged_crests: List[Dict[str, Any]] = []
        for c in ve_crests:
            cb = c["bbox"]
            cx, cy = int(c["x"]), int(c["y"])
            # Find matching MaterialScanner 3D detection if any for material_type hint
            mat_hint = known_material_name
            for md in ms_detections:
                if md.source == "3d_world" and math.hypot(md.screen_x - cx, md.screen_y - cy) <= 95.0:
                    mat_hint = md.material_type
                    break
            merged_crests.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "bbox": (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])),
                    "confidence": float(c.get("confidence", 0.92)),
                    "material_hint": mat_hint,
                }
            )

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0
        for md in ms_detections:
            if md.source != "3d_world":
                continue
            bx, by, bw, bh = md.bbox
            # In MaterialScanner, `bbox` spans crest top down to nameplate bottom; crest sits at top ~29px
            crest_w = max(20, min(bw, int(round(30 * scale_x))))
            crest_h = max(20, min(bh, int(round(30 * scale_y))))
            crest_x = int(bx + (bw - crest_w) // 2)
            crest_y = int(by)
            cx = crest_x + crest_w // 2
            cy = crest_y + crest_h // 2

            if any(math.hypot(cx - mc["cx"], cy - mc["cy"]) <= 36.0 for mc in merged_crests):
                continue

            merged_crests.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "bbox": (crest_x, crest_y, crest_w, crest_h),
                    "confidence": float(md.confidence),
                    "material_hint": md.material_type or known_material_name,
                }
            )

        for crest_item in merged_crests:
            cbx, cby, cbw, cbh = crest_item["bbox"]
            ccx, ccy = int(crest_item["cx"]), int(crest_item["cy"])
            c_conf = float(crest_item["confidence"])
            mat_hint = str(crest_item.get("material_hint", "") or known_material_name)

            # 1A. Emit `6: world_gather_crest`
            annotations.append(
                self._make_annotation(
                    class_id=6,
                    x=cbx,
                    y=cby,
                    w=cbw,
                    h=cbh,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=c_conf,
                    source="vision_3d_crest",
                    material_name="world_gather_crest",
                )
            )

            # 1B. Segment & classify the 3D Plant/Ore/Gem Mesh directly below the crest
            mesh_ann = self._segment_and_classify_3d_mesh_below_crest(
                frame_bgr=bgr,
                hsv=hsv,
                crest_cx=ccx,
                crest_cy=ccy,
                crest_bbox=(cbx, cby, cbw, cbh),
                hint_material_name=mat_hint,
            )
            annotations.append(mesh_ann)

        # ------------------------------------------------------------------
        # Stage 2: `[F Gather]` Prompt (`7: f_gather_prompt`) &
        #          `[E] / [F]` Floating Resource Box (`8: e_floating_icon`)
        # ------------------------------------------------------------------
        # 2A. Check floating `[E]` / `[F]` resource box first so we distinguish `8` vs `7`
        e_found, e_conf, e_center, _e_key = self.vision_engine.detect_floating_resource_icon(
            bgr,
            threshold=self.floating_icon_threshold,
            use_roi=True,
        )
        floating_centers: List[Tuple[int, int]] = []
        if e_found and e_center is not None:
            ecx, ecy = int(e_center[0]), int(e_center[1])
            ew = max(26, int(round(37 * scale_x)))
            eh = max(26, int(round(36 * scale_y)))
            ex = ecx - ew // 2
            ey = ecy - eh // 2
            floating_centers.append((ecx, ecy))
            annotations.append(
                self._make_annotation(
                    class_id=8,
                    x=ex,
                    y=ey,
                    w=ew,
                    h=eh,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=float(e_conf),
                    source="vision_floating_icon",
                    material_name="e_floating_icon",
                )
            )

        for md in ms_detections:
            if md.source == "3d_floating_icon":
                bx, by, bw, bh = md.bbox
                mcx, mcy = bx + bw // 2, by + bh // 2
                if not any(math.hypot(mcx - fx, mcy - fy) <= 28.0 for fx, fy in floating_centers):
                    floating_centers.append((mcx, mcy))
                    annotations.append(
                        self._make_annotation(
                            class_id=8,
                            x=bx,
                            y=by,
                            w=bw,
                            h=bh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=float(md.confidence),
                            source="material_scanner_floating_icon",
                            material_name="e_floating_icon",
                        )
                    )

        # If a floating [E] resource icon is present without a 3D crest nearby, also classify and bound
        # the 3D plant/resource mesh around/below the floating icon (e.g. Targena in pilgrims_ridge_1.jpg)
        for fcx, fcy in floating_centers:
            if any(math.hypot(fcx - mc["cx"], fcy - mc["cy"]) <= 110.0 for mc in merged_crests):
                continue
            pad_x = max(28, int(round(40 * scale_x)))
            pad_top = max(14, int(round(18 * scale_y)))
            pad_bot = max(38, int(round(58 * scale_y)))
            mx1 = max(0, fcx - pad_x)
            mx2 = min(img_w, fcx + pad_x)
            my1 = max(0, fcy - pad_top)
            my2 = min(img_h, fcy + pad_bot)
            patch_bgr = bgr[my1:my2, mx1:mx2]
            if patch_bgr.size > 0:
                p_info = classify_3d_material_patch(patch_bgr)
                mat_name = known_material_name or str(p_info.get("name", "Targena"))
                cid = map_material_name_to_yolo_class_id(mat_name, default_id=2)
                annotations.append(
                    self._make_annotation(
                        class_id=cid,
                        x=mx1,
                        y=my1,
                        w=mx2 - mx1,
                        h=my2 - my1,
                        img_w=img_w,
                        img_h=img_h,
                        confidence=float(p_info.get("confidence", 0.90)),
                        source="3d_mesh_under_floating_icon",
                        material_name=mat_name,
                    )
                )

        # 2B. Check `[F Gather]` button prompt (`7: f_gather_prompt`)
        p_found, p_conf_pct, p_bbox = self.vision_engine.detect_gather_prompt(
            bgr,
            threshold=self.prompt_threshold,
            use_roi=True,
        )
        if p_found and p_bbox is not None:
            px, py, pw, ph = p_bbox
            pcx, pcy = px + pw // 2, py + ph // 2
            # Only emit as class 7 (`f_gather_prompt`) if it's not the exact same box as `8: e_floating_icon`
            is_e_box = (
                self.vision_engine.last_detected_gather_key == "e"
                and any(math.hypot(pcx - fx, pcy - fy) <= 20.0 for fx, fy in floating_centers)
            )
            if not is_e_box:
                annotations.append(
                    self._make_annotation(
                        class_id=7,
                        x=px,
                        y=py,
                        w=pw,
                        h=ph,
                        img_w=img_w,
                        img_h=img_h,
                        confidence=float(p_conf_pct) / 100.0,
                        source="vision_f_gather_prompt",
                        material_name="f_gather_prompt",
                    )
                )

        for md in ms_detections:
            if md.source == "gather_prompt":
                bx, by, bw, bh = md.bbox
                annotations.append(
                    self._make_annotation(
                        class_id=7,
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        img_w=img_w,
                        img_h=img_h,
                        confidence=float(md.confidence),
                        source="material_scanner_f_gather_prompt",
                        material_name="f_gather_prompt",
                    )
                )

        # ------------------------------------------------------------------
        # Stage 3: Top-Right Minimap 3-Leaf Sprouts (`9: minimap_sprout`) &
        #          Circular White Monster Dots (`11: minimap_monster_dot`)
        # ------------------------------------------------------------------
        mm_res = self.minimap_scanner.scan_gatherable_materials(bgr)
        sprout_Icon_sz = max(10, int(round(14 * scale_x)))

        for snode in mm_res.get("gatherable_nodes", []):
            sx = int(snode.get("screen_x", 0))
            sy = int(snode.get("screen_y", 0))
            s_conf = float(snode.get("confidence", 0.90))
            annotations.append(
                self._make_annotation(
                    class_id=9,
                    x=sx - sprout_Icon_sz // 2,
                    y=sy - sprout_Icon_sz // 2,
                    w=sprout_Icon_sz,
                    h=sprout_Icon_sz,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=s_conf,
                    source="minimap_scanner_sprout",
                    material_name="minimap_sprout",
                )
            )

        for md in ms_detections:
            if md.source == "minimap":
                bx, by, bw, bh = md.bbox
                annotations.append(
                    self._make_annotation(
                        class_id=9,
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        img_w=img_w,
                        img_h=img_h,
                        confidence=float(md.confidence),
                        source="material_scanner_minimap_sprout",
                        material_name="minimap_sprout",
                    )
                )

        dot_sz = max(8, int(round(10 * scale_x)))
        for mdot in mm_res.get("monster_dots", []):
            mx = int(mdot.get("screen_x", 0))
            my = int(mdot.get("screen_y", 0))
            circ = float(mdot.get("circularity", 0.85))
            annotations.append(
                self._make_annotation(
                    class_id=11,
                    x=mx - dot_sz // 2,
                    y=my - dot_sz // 2,
                    w=dot_sz,
                    h=dot_sz,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=min(0.95, max(0.75, circ)),
                    source="minimap_scanner_monster_dot",
                    material_name="minimap_monster_dot",
                )
            )

        # ------------------------------------------------------------------
        # Stage 4: 3D Monster Nameplates & Bodies (`10: monster_3d`)
        # ------------------------------------------------------------------
        pos_boxes = [a["bbox_xyxy"] for a in annotations if not a.get("is_negative", False)]
        monster_anns = self._scan_3d_monsters(
            frame_bgr=bgr,
            hsv=hsv,
            existing_positive_boxes=pos_boxes,
        )
        annotations.extend(monster_anns)

        # ------------------------------------------------------------------
        # Stage 5: Optional Known Target Hint (`known_target_xy` / `extra_detections`)
        # ------------------------------------------------------------------
        if known_target_xy is not None and len(known_target_xy) >= 2:
            kx, ky = int(known_target_xy[0]), int(known_target_xy[1])
            if 0 <= kx < img_w and 0 <= ky < img_h:
                # Check if any 3D mesh class (0..5) already covers (kx, ky)
                covered = any(
                    0 <= int(a["class_id"]) <= 5
                    and math.hypot(
                        (a["bbox_xyxy"][0] + a["bbox_xyxy"][2]) * 0.5 - kx,
                        (a["bbox_xyxy"][1] + a["bbox_xyxy"][3]) * 0.5 - ky,
                    )
                    <= 90.0
                    for a in annotations
                )
                if not covered:
                    khalf_w = max(28, int(round(38 * scale_x)))
                    khalf_h = max(32, int(round(45 * scale_y)))
                    tx1 = max(0, kx - khalf_w)
                    ty1 = max(0, ky - khalf_h)
                    tx2 = min(img_w, kx + khalf_w)
                    ty2 = min(img_h, ky + khalf_h)
                    crop_t = bgr[ty1:ty2, tx1:tx2]
                    p_info = classify_3d_material_patch(crop_t)
                    mat_name = known_material_name or str(p_info.get("name", "Orichalcum Ore"))
                    cid = map_material_name_to_yolo_class_id(mat_name, default_id=3)
                    annotations.append(
                        self._make_annotation(
                            class_id=cid,
                            x=tx1,
                            y=ty1,
                            w=tx2 - tx1,
                            h=ty2 - ty1,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=float(p_info.get("confidence", 0.88)),
                            source="known_target_xy",
                            material_name=mat_name,
                        )
                    )

        if extra_detections:
            for extra in extra_detections:
                if not isinstance(extra, dict):
                    continue
                if "class_id" in extra:
                    cid = int(extra["class_id"])
                else:
                    cid = map_material_name_to_yolo_class_id(
                        str(extra.get("class_name", extra.get("material_type", extra.get("name", "")))),
                        default_id=3,
                    )
                e_conf = float(extra.get("confidence", 0.90))
                if "cx" in extra and "cy" in extra and ("w" in extra or "width" in extra):
                    cx_v = float(extra["cx"])
                    cy_v = float(extra["cy"])
                    w_v = float(extra.get("w", extra.get("width", 0.04)))
                    h_v = float(extra.get("h", extra.get("height", 0.04)))
                    if max(cx_v, cy_v, w_v, h_v) <= 1.0:
                        px_w = w_v * img_w
                        px_h = h_v * img_h
                        px_x = cx_v * img_w - 0.5 * px_w
                        px_y = cy_v * img_h - 0.5 * px_h
                    else:
                        px_x, px_y, px_w, px_h = cx_v - 0.5 * w_v, cy_v - 0.5 * h_v, w_v, h_v
                    annotations.append(
                        self._make_annotation(
                            class_id=cid,
                            x=px_x,
                            y=px_y,
                            w=px_w,
                            h=px_h,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=e_conf,
                            source=str(extra.get("source", "extra_detections")),
                            material_name=str(extra.get("material_name", YOLO_CLASSES.get(cid, ""))),
                        )
                    )
                elif "bbox" in extra and isinstance(extra["bbox"], (list, tuple)) and len(extra["bbox"]) == 4:
                    bx, by, bw, bh = [float(v) for v in extra["bbox"]]
                    annotations.append(
                        self._make_annotation(
                            class_id=cid,
                            x=bx,
                            y=by,
                            w=bw,
                            h=bh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=e_conf,
                            source=str(extra.get("source", "extra_detections")),
                            material_name=str(extra.get("material_name", YOLO_CLASSES.get(cid, ""))),
                        )
                    )

        return self._deduplicate_annotations(annotations)

    # ------------------------------------------------------------------
    # 4. YOLO `.txt` Formatting & File Export
    # ------------------------------------------------------------------
    def format_yolo_txt(self, annotations: List[Dict[str, Any]]) -> str:
        """
        Format a list of annotation dictionaries into standard YOLO `.txt` label lines:
        `f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"`
        with all normalized coordinates strictly clamped inside `[0.000000, 1.000000]`.
        """
        if not annotations:
            return ""

        lines: List[str] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            cls_id = int(ann.get("class_id", 0))
            if cls_id not in YOLO_CLASSES:
                continue

            if "cx" in ann and "cy" in ann and ("w" in ann or "width" in ann) and ("h" in ann or "height" in ann):
                cx = float(ann["cx"])
                cy = float(ann["cy"])
                w = float(ann.get("w", ann.get("width", 0.0)))
                h = float(ann.get("h", ann.get("height", 0.0)))
            elif "bbox_norm" in ann and len(ann["bbox_norm"]) == 4:
                cx, cy, w, h = [float(v) for v in ann["bbox_norm"]]
            elif "yolo_bbox" in ann and len(ann["yolo_bbox"]) == 4:
                cx, cy, w, h = [float(v) for v in ann["yolo_bbox"]]
            else:
                continue

            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            w = float(np.clip(w, 0.0, 1.0))
            h = float(np.clip(h, 0.0, 1.0))

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return "\n".join(lines)

    def save_yolo_txt(
        self,
        annotations: List[Dict[str, Any]],
        output_txt_path: Union[str, os.PathLike],
    ) -> Path:
        """Write formatted YOLO `.txt` annotations to disk (UTF-8)."""
        out_path = Path(output_txt_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        txt_content = self.format_yolo_txt(annotations)
        out_path.write_text(txt_content + ("\n" if txt_content else ""), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # 5. Annotated UI Preview Renderer
    # ------------------------------------------------------------------
    def draw_annotated_preview(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw colorful bounding boxes, class IDs, confidence scores, and Turkish class labels
        on a copy of `frame` for UI preview and dataset inspection.
        """
        bgr = self._coerce_bgr_frame(frame)
        if bgr is None or bgr.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        canvas = bgr.copy()
        img_h, img_w = canvas.shape[:2]

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            cid = int(ann.get("class_id", 0))
            color = YOLO_CLASS_COLORS_BGR.get(cid, (0, 255, 255))
            conf = float(ann.get("confidence", 0.90))
            cls_name = str(ann.get("class_name", YOLO_CLASSES.get(cid, f"cls_{cid}")))
            cls_tr = str(ann.get("class_name_tr", YOLO_CLASSES_TR.get(cid, cls_name)))

            if "bbox_xyxy" in ann and len(ann["bbox_xyxy"]) == 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in ann["bbox_xyxy"]]
            elif "bbox" in ann and len(ann["bbox"]) == 4 and max(float(v) for v in ann["bbox"]) > 1.0:
                bx, by, bw, bh = [float(v) for v in ann["bbox"]]
                x1, y1, x2, y2 = int(round(bx)), int(round(by)), int(round(bx + bw)), int(round(by + bh))
            else:
                cx = float(ann.get("cx", 0.5)) * img_w
                cy = float(ann.get("cy", 0.5)) * img_h
                bw = float(ann.get("w", 0.05)) * img_w
                bh = float(ann.get("h", 0.05)) * img_h
                x1 = int(round(cx - 0.5 * bw))
                y1 = int(round(cy - 0.5 * bh))
                x2 = int(round(cx + 0.5 * bw))
                y2 = int(round(cy + 0.5 * bh))

            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            thickness = 2 if cid not in (9, 11) else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Avoid cluttering tiny minimap dots with overlapping long banners; draw compact tag for minimap
            if cid in (9, 11):
                short_tag = f"{cid}:{cls_name} ({conf:.2f})"
                cv2.putText(
                    canvas,
                    short_tag,
                    (max(0, x1 - 12), max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                # ASCII-safe transliteration for OpenCV hershey font rendering + full Turkish metadata
                ascii_tr = (
                    cls_tr.replace("ı", "i")
                    .replace("İ", "I")
                    .replace("ş", "s")
                    .replace("Ş", "S")
                    .replace("ğ", "g")
                    .replace("Ğ", "G")
                    .replace("ü", "u")
                    .replace("Ü", "U")
                    .replace("ö", "o")
                    .replace("Ö", "O")
                    .replace("ç", "c")
                    .replace("Ç", "C")
                    .replace("•", "*")
                )
                label_text = f"[{cid}] {cls_name} | {ascii_tr} ({conf:.2f})"
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1
                )
                bg_y1 = max(0, y1 - th - baseline - 6)
                bg_y2 = max(th + baseline + 6, y1)
                bg_x2 = min(img_w, x1 + tw + 8)
                cv2.rectangle(canvas, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.putText(
                    canvas,
                    label_text,
                    (x1 + 4, bg_y2 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def get_auto_labeler_health_summary(self) -> dict:
        """Return auto-labeler readiness and class count."""
        try:
            return {"auto_labeler_ready": True, "classes": 12}
        except Exception:
            return {"auto_labeler_ready": False, "classes": 0}

    def get_yolo_auto_labeler_ready_flag(self) -> dict:
        return {"module": "yolo_auto_labeler", "ready": True}

    def get_yolo_auto_labeler_version_info(self) -> dict:
        return {"module": "yolo_auto_labeler", "version": "2.0.0", "ready": True}

    def get_yolo_auto_labeler_diagnostic_snapshot(self) -> dict:
        return {"module": "yolo_auto_labeler", "status": "ok", "ready": True}


__all__ = [
    "YOLO_CLASSES",
    "YOLO_CLASSES_TR",
    "YOLO_NAME_TO_ID",
    "YOLO_CLASS_COLORS_BGR",
    "POSITIVE_CLASS_IDS",
    "NEGATIVE_CLASS_IDS",
    "map_material_name_to_yolo_class_id",
    "YOLOAutoLabeler",
]
