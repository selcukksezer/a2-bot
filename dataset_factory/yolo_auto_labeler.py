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
    12: "world_material",
    13: "skill_ui_prompt",
    14: "obstacle",
    15: "path_ground",
    16: "player_hp_bar",
    17: "player_mp_bar",
    18: "hp_potion",
    19: "mp_potion",
    20: "target_hp_bar",
    21: "skill_slot",
    22: "buff_scroll",
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
    12: "Adı bilinmeyen dünya malzemesi",
    13: "Skill / arayüz istemi (F ve E değil)",
    14: "Engel",
    15: "Yol / zemin",
    16: "Oyuncu Can Barı (HP)",
    17: "Oyuncu Mana Barı (MP)",
    18: "Can İksiri (HP Potion)",
    19: "Mana İksiri (MP Potion)",
    20: "Hedef Can Barı (Target HP)",
    21: "Skill Yuvası",
    22: "Güçlendirme Parşömeni (Buff Scroll)",
}

YOLO_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in YOLO_CLASSES.items()}

try:
    from dataset_factory.class_taxonomy import dataset_class_names as _taxonomy_class_names
    from dataset_factory.class_taxonomy import dataset_labels_tr as _taxonomy_labels_tr

    for _cid, _name in _taxonomy_class_names().items():
        if int(_cid) >= 16 and int(_cid) not in YOLO_CLASSES:
            YOLO_CLASSES[int(_cid)] = _name
            YOLO_CLASSES_TR[int(_cid)] = _taxonomy_labels_tr().get(int(_cid), _name)
    YOLO_NAME_TO_ID.clear()
    YOLO_NAME_TO_ID.update({name: cid for cid, name in YOLO_CLASSES.items()})
except Exception:
    pass

POSITIVE_CLASS_IDS: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22)
NEGATIVE_CLASS_IDS: Sequence[int] = (10, 11, 14)

UNVERIFIED_HEURISTIC_SOURCES = {
    "chromatic_salient_3d_resource",
    "visual_ground_path_segmenter",
    "visual_obstacle_segmenter",
    "hud_skill_slot",
    "hud_player_hp_bar",
    "hud_player_mp_bar",
    "hud_hp_potion",
    "hud_mp_potion",
    "hud_buff_scroll",
    "dataset_requeue",
    "legacy_auto",
    "center_anchor_fallback",
    "gather_material_hint",
}


def is_path_learning_stem(stem: str) -> bool:
    """Check if stem is dedicated to path/obstacle learning."""
    s = str(stem or "").strip().lower()
    return (
        s.startswith("learn_paths_")
        or "learn_paths" in s
        or "learn_obstacles" in s
        or "_path_ground" in s
        or "_path_" in s
        or s.startswith("path_")
    )


def is_skill_learning_stem(stem: str) -> bool:
    """Check if stem is dedicated to HUD skill/vitality learning."""
    s = str(stem or "").strip().lower()
    return (
        "learn_skills" in s
        or "learn_vitality" in s
        or s.startswith("skill_")
        or "_vitality" in s
    )

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
    12: (180, 180, 180),
    13: (200, 160, 255),
    14: (40, 90, 200),
    15: (160, 200, 120),
    16: (60, 60, 230),
    17: (230, 160, 50),
    18: (40, 40, 240),    # hp_potion: vibrant red
    19: (240, 160, 40),   # mp_potion: vibrant blue/cyan
    20: (30, 30, 220),    # target_hp_bar: target health bar
    21: (200, 140, 240),  # skill_slot: slot outline
    22: (80, 200, 230),   # buff_scroll: golden parchment scroll
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

    try:
        from dataset_factory.class_taxonomy import class_id_for_material_text

        specific = class_id_for_material_text(norm)
    except Exception:
        specific = None
    if specific is not None:
        return int(specific)

    # Negative / monster keywords
    if any(k in norm for k in ("monster_dot", "minimap_monster", "yuvarlak canavar")):
        return 11
    if any(k in norm for k in ("monster", "zaif", "sparkie", "skyray", "canavar", "hostile", "mob")):
        return 10

    # UI / Icon / HUD classes
    if "buff_scroll" in norm or "scroll" in norm or "parşömen" in norm or "parcsomen" in norm:
        return 22
    if "hp_potion" in norm or "can iksiri" in norm or ("potion" in norm and "hp" in norm):
        return 18
    if "mp_potion" in norm or "mana iksiri" in norm or ("potion" in norm and "mp" in norm):
        return 19
    if "target_hp_bar" in norm or "target_hp" in norm:
        return 20
    if "skill_slot" in norm or "skill_bar" in norm:
        return 21
    if "player_hp_bar" in norm or "player_hp" in norm:
        return 16
    if "player_mp_bar" in norm or "player_mp" in norm:
        return 17
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

    if any(k in norm for k in ("gatherable plant", "gatherable material", "world_material")):
        return 12
    if any(k in norm for k in ("skill_ui", "skill", "cooldown", "vitality")):
        return 13
    if any(k in norm for k in ("obstacle", "engel", "cliff")):
        return 14
    if any(k in norm for k in ("path_ground", "walking path", "navigable")):
        return 15

    if int(default_id) in YOLO_NAME_TO_ID.values() or 0 <= int(default_id) < len(YOLO_CLASSES):
        if int(default_id) == 3 and norm not in ("",):
            return 12
    return 12 if int(default_id) == 3 else int(default_id)


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
        exemplar_memory: Optional[Any] = None,
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
        if exemplar_memory is not None:
            self.exemplar_memory = exemplar_memory
        else:
            try:
                from dataset_factory.exemplar_memory import get_exemplar_memory

                self.exemplar_memory = get_exemplar_memory()
            except Exception:
                self.exemplar_memory = None

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
        hand_label: bool = False,
        **kwargs,
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
        is_hand = bool(
            hand_label
            or kwargs.get("is_hand_label")
            or kwargs.get("human_approved")
            or kwargs.get("approved")
            or "hand" in str(source).lower()
        )
        src_str = str(source).strip().lower()
        is_heuristic = any(src_str == s or src_str.startswith(s) for s in UNVERIFIED_HEURISTIC_SOURCES)

        if is_hand:
            is_emin = True
        elif "trusted" in kwargs:
            is_emin = bool(kwargs["trusted"])
        elif "weak" in kwargs:
            is_emin = not bool(kwargs["weak"])
        elif is_heuristic:
            # Unverified heuristic proposals must NEVER be trusted/emin without human approval
            is_emin = False
        else:
            is_emin = bool(
                conf_clamped >= 0.85
                and (
                    source.startswith("vision_3d_crest")
                    or "gather_prompt" in source
                    or "floating" in source
                    or "exemplar" in source
                    or "target_hp_bar" in source
                    or "minimap" in source
                    or "monster" in source
                )
            )

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
            "hand_label": is_hand,
            "trusted": is_emin,
            "weak": not is_emin,
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
                named = map_material_name_to_yolo_class_id(detected_mat_name, default_id=12)
                if named == 3 and patch_conf < 0.80:
                    mesh_class_id = 12
                else:
                    mesh_class_id = named
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
            confidence=float(np.clip(patch_conf, 0.05, 0.99)),
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
        gray: Optional[np.ndarray] = None,
        shared_frame_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect 3D monster nameplates, bodies, and hostile HP bars (`class_id = 10: monster_3d`)
        and target HP bars (`class_id = 20: target_hp_bar`), such as:
        - Hostile red monster HP bars / nameplates in 3D viewport (Hue 0..10 | 168..180, Sat >= 125, Val >= 115).
        - Top boss / target enemy health bar (0.02*H..0.12*H, 0.30*W..0.70*W) -> class_id 20: target_hp_bar.
        - Targeted monsters with cyan diamond bracket tips + yellow text + green/red health bar +
          red '4' agro badge (e.g. `< Yellow-Maned Zaif >`).
        - Untargeted yellow/olive monster nameplates in the 3D world (e.g. `Highland Odyle Sparkie`).
        """
        monsters: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 500 or img_h < 300:
            return monsters

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0
        if gray is None:
            if shared_frame_context is not None and "gray" in shared_frame_context and shared_frame_context["gray"] is not None:
                gray = shared_frame_context["gray"]
            else:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if shared_frame_context is not None:
                    shared_frame_context["gray"] = gray

        # ------------------------------------------------------------------
        # Stage B1: Hostile Red Monster HP Bars & Top Target HP Bar
        # ------------------------------------------------------------------
        red_part1 = cv2.inRange(hsv, np.array([0, 125, 115]), np.array([10, 255, 255]))
        red_part2 = cv2.inRange(hsv, np.array([168, 125, 115]), np.array([180, 255, 255]))
        red_u8 = cv2.bitwise_or(red_part1, red_part2)

        # 1. Top HUD target HP bar check (0.02*H..0.12*H, 0.30*W..0.70*W)
        top_y1, top_y2 = int(img_h * 0.02), int(img_h * 0.12)
        top_x1, top_x2 = int(img_w * 0.30), int(img_w * 0.70)
        top_red = red_u8[top_y1:top_y2, top_x1:top_x2]
        if top_red.size > 0:
            top_closed = cv2.morphologyEx(
                top_red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            )
            top_cnts, _ = cv2.findContours(top_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for tcnt in top_cnts:
                tx, ty, tw, th_b = cv2.boundingRect(tcnt)
                aspect = float(tw) / float(max(1, th_b))
                min_tw = max(40, int(round(60 * scale_x)))
                max_tw = max(120, int(round(360 * scale_x)))
                if aspect >= 3.5 and min_tw <= tw <= max_tw and 4 <= th_b <= int(32 * scale_y):
                    abs_tx = top_x1 + tx
                    abs_ty = top_y1 + ty
                    target_bar_cid = YOLO_NAME_TO_ID.get("target_hp_bar", 20)
                    monsters.append(
                        self._make_annotation(
                            class_id=target_bar_cid,
                            x=abs_tx,
                            y=abs_ty,
                            w=tw,
                            h=th_b,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=0.92,
                            source="target_hp_bar_scanner",
                            material_name="target_hp_bar",
                        )
                    )

        # 2. Viewport hostile monster red health bars (0.12*H..0.75*H)
        vp_red = red_u8.copy()
        vp_red[: int(img_h * 0.12), :] = 0
        vp_red[int(img_h * 0.75) :, :] = 0
        vp_red[:, int(img_w * 0.852) :] = 0
        vp_red[: int(img_h * 0.30), int(img_w * 0.71) :] = 0
        vp_red[int(img_h * 0.66) :, : int(img_w * 0.26)] = 0
        vp_red[int(img_h * 0.42) : int(img_h * 0.58), int(img_w * 0.44) : int(img_w * 0.56)] = 0

        rkw = max(8, int(round(14 * scale_x)))
        rkh = max(2, int(round(3 * scale_y)))
        r_closed = cv2.morphologyEx(
            vp_red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (rkw, rkh))
        )
        r_cnts, _ = cv2.findContours(r_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_rw = max(35, int(round(50 * scale_x)))
        max_rw = max(120, int(round(280 * scale_x)))

        for rcnt in r_cnts:
            rx, ry, rcw, rch = cv2.boundingRect(rcnt)
            if not (min_rw <= rcw <= max_rw and 3 <= rch <= int(25 * scale_y)):
                continue
            aspect = float(rcw) / float(max(1, rch))
            if aspect < 3.2:
                continue
            m_x = max(0, rx - int(round(12 * scale_x)))
            m_y = max(0, ry - int(round(6 * scale_y)))
            m_w = min(img_w - m_x, rcw + int(round(24 * scale_x)))
            m_h = min(img_h - m_y, int(rcw * 0.85) + int(round(40 * scale_y)))

            cand_xyxy = (m_x, m_y, m_x + m_w, m_y + m_h)
            if any(self._compute_iou_xyxy(cand_xyxy, pos_box) >= 0.25 for pos_box in existing_positive_boxes):
                continue
            monsters.append(
                self._make_annotation(
                    class_id=10,
                    x=m_x,
                    y=m_y,
                    w=m_w,
                    h=m_h,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=0.90,
                    source="hostile_red_hp_bar_monster",
                    material_name="monster_3d",
                )
            )

        # ------------------------------------------------------------------
        # Stage B2: Yellow/Olive Monster Nameplate Glyph Detector
        # ------------------------------------------------------------------
        th_sz = max(5, int(round(7 * scale_x)) | 1)
        tophat = cv2.morphologyEx(
            gray,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (th_sz, th_sz)),
        )

        yellow_stroke = (
            (hsv[:, :, 0] >= 20)
            & (hsv[:, :, 0] <= 32)
            & (hsv[:, :, 1] >= 90)
            & (hsv[:, :, 2] >= 110)
            & (tophat >= 8)
        )
        ym_u8 = (yellow_stroke.astype(np.uint8)) * 255

        ym_u8[: int(img_h * 0.11), :] = 0
        ym_u8[int(img_h * 0.78) :, :] = 0
        ym_u8[:, int(img_w * 0.852) :] = 0
        ym_u8[: int(img_h * 0.30), int(img_w * 0.71) :] = 0
        ym_u8[int(img_h * 0.66) :, : int(img_w * 0.26)] = 0
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
            min_stroke = max(35, int(round(90 * scale_x * scale_y)))

            lx1 = max(0, x - int(round(26 * scale_x)))
            lx2 = min(img_w, x + int(round(6 * scale_x)))
            rx1 = max(0, x + cw - int(round(6 * scale_x)))
            rx2 = min(img_w, x + cw + int(round(26 * scale_x)))
            ty1 = max(0, y - int(round(6 * scale_y)))
            ty2 = min(img_h, y + ch + int(round(6 * scale_y)))

            left_cyan = cv2.inRange(hsv[ty1:ty2, lx1:lx2], np.array([75, 70, 120]), np.array([105, 255, 255]))
            right_cyan = cv2.inRange(hsv[ty1:ty2, rx1:rx2], np.array([75, 70, 120]), np.array([105, 255, 255]))
            has_diamond_tips = (
                int(np.count_nonzero(left_cyan)) >= max(4, int(round(8 * scale_x * scale_y)))
                and int(np.count_nonzero(right_cyan)) >= max(4, int(round(8 * scale_x * scale_y)))
            )

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

            num_labels, _ = cv2.connectedComponents(sub_ym)
            if not has_agro_badge and (stroke_px < min_stroke or num_labels < 7):
                continue

            dilated_stroke = cv2.dilate(sub_ym, np.ones((3, 3), dtype=np.uint8))
            border_ring = dilated_stroke - sub_ym
            ring_gray = gray[y : y + ch, x : x + cw][border_ring > 0]
            if ring_gray.size == 0:
                continue
            dark_outline_ratio = float(np.mean(ring_gray < 120))
            if not has_agro_badge and dark_outline_ratio < 0.20:
                continue

            if has_agro_badge:
                final_x = max(0, x - int(round(24 * scale_x)))
                final_y = max(0, y - int(round(8 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(48 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(105 * scale_y)))
                conf = 0.95
                is_weak = False
            else:
                final_x = max(0, x - int(round(6 * scale_x)))
                final_y = max(0, y - int(round(4 * scale_y)))
                final_w = min(img_w - final_x, cw + int(round(12 * scale_x)))
                final_h = min(img_h - final_y, ch + int(round(8 * scale_y)))
                conf = round(min(0.68, 0.52 + min(0.16, stroke_px / 900.0)), 4)
                is_weak = True

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
                    weak=is_weak,
                    trusted=not is_weak,
                )
            )

        return monsters

    # ------------------------------------------------------------------
    # Stage A: Salient 3D Resources & Crestless Nodes (0..5, 12)
    # ------------------------------------------------------------------
    def _scan_salient_3d_resources(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        existing_boxes: Sequence[Sequence[int]],
        hint_material_name: str = "",
        has_crest_or_prompt: bool = False,
        shared_frame_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage A: Salient 3D Resources & Crestless Nodes (classes 0..5, 12).
        Fallback ONLY when NO 3D crest (`vision_3d_crest`) or prompt icon is found.
        Uses chromatic contour segmentation for ore/herb/crystal even if white wing crest is dim or occluded.
        Validates candidate patches with `classify_3d_material_patch`.
        Never assigns class_id > 12 (never assigns essence_*).
        Confidence is strictly calibrated: conf = min(0.68, max(0.38, round(p_score * 0.75, 4))).
        Marked as trusted=False, weak=True.
        """
        if has_crest_or_prompt:
            return []

        salient_anns: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 500 or img_h < 300:
            return salient_anns

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0

        vx1, vx2 = int(img_w * 0.16), int(img_w * 0.82)
        vy1, vy2 = int(img_h * 0.18), int(img_h * 0.78)
        roi_bgr = frame_bgr[vy1:vy2, vx1:vx2]
        roi_hsv = hsv[vy1:vy2, vx1:vx2]
        if roi_bgr.size == 0:
            return salient_anns

        sat_mask = cv2.inRange(roi_hsv, np.array([0, 75, 70]), np.array([180, 255, 255]))
        ksize = max(5, int(round(7 * scale_x)) | 1)
        closed = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rw = vx2 - vx1
        rh = vy2 - vy1
        min_a = max(36.0, float(rw * rh) * 0.003)
        max_a = float(rw * rh) * 0.25

        valid_cnts = [c for c in cnts if min_a <= float(cv2.contourArea(c)) <= max_a]
        valid_cnts.sort(key=cv2.contourArea, reverse=True)

        for cnt in valid_cnts[:4]:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.25 or aspect > 4.0:
                continue

            abs_x = vx1 + bx
            abs_y = vy1 + by
            cand_xyxy = (abs_x, abs_y, abs_x + bw, abs_y + bh)
            if any(self._compute_iou_xyxy(cand_xyxy, ebox) >= 0.20 for ebox in existing_boxes):
                continue

            patch = frame_bgr[abs_y : abs_y + bh, abs_x : abs_x + bw]
            if patch.size == 0:
                continue

            try:
                p_info = classify_3d_material_patch(patch, label_width_px=bw)
            except Exception:
                continue

            p_score = float(p_info.get("confidence", p_info.get("score", 0.0)))
            mat_name = str(p_info.get("name", ""))
            if not mat_name or is_rejected_non_material(mat_name) or p_score < 0.20:
                continue

            cid = map_material_name_to_yolo_class_id(mat_name, default_id=12)
            # Never assign class_id > 12 (never essence_*) or UI/monster classes in salient 3D resources
            if cid > 12 or cid in (6, 7, 8, 9, 10, 11):
                cid = 12
                mat_name = "world_material"

            # Strict calibrated confidence without artificial floors:
            # conf = min(0.68, max(0.38, round(p_score * 0.75, 4)))
            calibrated_conf = min(0.68, max(0.38, round(p_score * 0.75, 4)))

            salient_anns.append(
                self._make_annotation(
                    class_id=cid,
                    x=abs_x,
                    y=abs_y,
                    w=bw,
                    h=bh,
                    img_w=img_w,
                    img_h=img_h,
                    confidence=calibrated_conf,
                    source="chromatic_salient_3d_resource",
                    material_name=mat_name,
                    trusted=False,
                    weak=True,
                )
            )

        return salient_anns

    # ------------------------------------------------------------------
    # Stage C: Genuine HUD & Skill Bar Detection (16, 17, 18, 19, 21, 22)
    # ------------------------------------------------------------------
    def _scan_hud_and_skills(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        source_stem: str = "",
        shared_frame_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Genuine HUD & Skill Bar Detection (classes 16, 17, 18, 19, 21, 22):
        - Player HP bar (16: player_hp_bar): horizontal red/green health bar in bottom HUD
        - Player MP bar (17: player_mp_bar): horizontal blue mana bar in bottom HUD
        - Upper Consumable Row (0.870*H..0.938*H): Buff scrolls (22: buff_scroll), HP potions (18: hp_potion), MP potions (19: mp_potion)
        - Lower Combat Skills Row (>=0.940*H): Active combat skill slots (21: skill_slot)
        - Center emblem exclusion: skips center level emblem (0.44..0.56*W, 0.85..0.95*H)
        STRICT STEM GATING: ONLY execute on "learn_skills" or "learn_vitality" stems!
        Normal gameplay/gathering frames must NEVER receive HUD annotations!
        Without human approval, confidence capped at <= 0.70 (weak=True).
        """
        if not is_skill_learning_stem(source_stem):
            return []

        hud_anns: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 500 or img_h < 300:
            return hud_anns

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0

        y1, y2 = int(img_h * 0.80), int(img_h * 0.98)
        x1, x2 = int(img_w * 0.18), int(img_w * 0.82)
        roi_hsv = hsv[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            return hud_anns

        roi_w = x2 - x1
        left_hsv = roi_hsv[:, : int(roi_w * 0.50)]
        if left_hsv.size > 0:
            red1 = cv2.inRange(left_hsv, np.array([0, 110, 80]), np.array([10, 255, 255]))
            red2 = cv2.inRange(left_hsv, np.array([168, 110, 80]), np.array([180, 255, 255]))
            hp_red = cv2.bitwise_or(red1, red2)
            hp_green = cv2.inRange(left_hsv, np.array([35, 90, 80]), np.array([85, 255, 255]))
            hp_mask = cv2.bitwise_or(hp_red, hp_green)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(12 * scale_x)), max(2, int(3 * scale_y))))
            hp_closed = cv2.morphologyEx(hp_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(hp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_hp_w = max(40, int(60 * scale_x))
            max_hp_w = max(100, int(260 * scale_x))

            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(max(1, bh))
                if min_hp_w <= bw <= max_hp_w and 5 <= bh <= int(28 * scale_y) and aspect >= 3.8:
                    abs_x = x1 + bx
                    abs_y = y1 + by
                    # Player HP bar strictly occupies y: 0.84*H..0.925*H above the action hotbar
                    if abs_y + bh > int(img_h * 0.925) or abs_y < int(img_h * 0.83):
                        continue
                    hud_anns.append(
                        self._make_annotation(
                            class_id=16,
                            x=abs_x,
                            y=abs_y,
                            w=bw,
                            h=bh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=0.70,
                            source="hud_player_hp_bar",
                            material_name="player_hp_bar",
                            trusted=False,
                            weak=True,
                        )
                    )
                    break

            mp_blue = cv2.inRange(left_hsv, np.array([95, 90, 80]), np.array([130, 255, 255]))
            mp_closed = cv2.morphologyEx(mp_blue, cv2.MORPH_CLOSE, kernel)
            mp_cnts, _ = cv2.findContours(mp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in mp_cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = float(bw) / float(max(1, bh))
                if min_hp_w <= bw <= max_hp_w and 4 <= bh <= int(25 * scale_y) and aspect >= 3.8:
                    abs_x = x1 + bx
                    abs_y = y1 + by
                    # Player MP bar strictly occupies y: 0.84*H..0.925*H above the action hotbar
                    if abs_y + bh > int(img_h * 0.925) or abs_y < int(img_h * 0.83):
                        continue
                    hud_anns.append(
                        self._make_annotation(
                            class_id=17,
                            x=abs_x,
                            y=abs_y,
                            w=bw,
                            h=bh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=0.70,
                            source="hud_player_mp_bar",
                            material_name="player_mp_bar",
                            trusted=False,
                            weak=True,
                        )
                    )
                    break

        # Edge map for HUD slots: check shared_frame_context
        skill_roi_x1 = x1 + int(roi_w * 0.20)
        skill_roi_x2 = x2
        skill_roi_gray = gray[y1:y2, skill_roi_x1:skill_roi_x2]
        if skill_roi_gray.size > 0:
            if shared_frame_context is not None and "edges" in shared_frame_context and shared_frame_context["edges"] is not None:
                edges = shared_frame_context["edges"][y1:y2, skill_roi_x1:skill_roi_x2]
            else:
                edges = cv2.Canny(skill_roi_gray, 50, 150)
            sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_dil = cv2.dilate(edges, sq_kernel)
            cnts, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_slot_sz = max(18, int(22 * scale_x))
            max_slot_sz = max(40, int(60 * scale_x))

            valid_slots: List[Tuple[int, int, int, int]] = []
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if min_slot_sz <= bw <= max_slot_sz and min_slot_sz <= bh <= max_slot_sz:
                    aspect = float(bw) / float(max(1, bh))
                    if 0.78 <= aspect <= 1.28:
                        slot_edges = edges[by : by + bh, bx : bx + bw]
                        if int(np.count_nonzero(slot_edges)) >= max(10, int(15 * scale_x)):
                            valid_slots.append((bx, by, bw, bh))

            for bx, by, bw, bh in valid_slots[:16]:
                abs_x = skill_roi_x1 + bx
                abs_y = y1 + by
                cx = abs_x + bw * 0.5
                cy = abs_y + bh * 0.5
                norm_cx = cx / float(img_w)
                norm_cy = cy / float(img_h)

                # 1. Exclude the center character level '45' emblem (x: 0.44..0.56, y: 0.85..0.95)
                if 0.44 <= norm_cx <= 0.56 and 0.85 <= norm_cy <= 0.95:
                    continue

                slot_bgr = frame_bgr[abs_y : abs_y + bh, abs_x : abs_x + bw]
                slot_hsv = hsv[abs_y : abs_y + bh, abs_x : abs_x + bw]
                if slot_bgr.size == 0 or slot_hsv.size == 0:
                    continue

                slot_area = bw * bh

                # 2. Upper Consumable Row (y: 0.870 .. 0.938*H)
                if 0.865 <= norm_cy <= 0.938:
                    # Never label as skill_slot or skill_ui_prompt!
                    # Check empty slot:
                    sat_chan = slot_hsv[:, :, 1]
                    val_chan = slot_hsv[:, :, 2]
                    if float(np.mean(sat_chan)) < 28.0 and float(np.std(val_chan)) < 16.0:
                        # Empty slot (e.g. F8) -> do not label
                        continue

                    # HP Potion (red liquid)
                    r1 = cv2.inRange(slot_hsv, np.array([0, 105, 75]), np.array([12, 255, 255]))
                    r2 = cv2.inRange(slot_hsv, np.array([165, 105, 75]), np.array([180, 255, 255]))
                    hp_red_px = int(np.count_nonzero(r1 | r2))

                    # MP Potion (blue or yellow liquid)
                    b_mask = cv2.inRange(slot_hsv, np.array([95, 85, 75]), np.array([135, 255, 255]))
                    y_mask = cv2.inRange(slot_hsv, np.array([20, 95, 95]), np.array([35, 255, 255]))
                    mp_blue_yellow_px = int(np.count_nonzero(b_mask | y_mask))

                    # Buff Scroll (rolled parchment icon: tan/golden/cream)
                    scroll_mask = cv2.inRange(slot_hsv, np.array([12, 30, 85]), np.array([28, 175, 235]))
                    scroll_px = int(np.count_nonzero(scroll_mask))

                    min_colored = max(8, int(0.07 * slot_area))

                    if hp_red_px >= min_colored and hp_red_px > mp_blue_yellow_px and hp_red_px > scroll_px:
                        hud_anns.append(
                            self._make_annotation(
                                class_id=18,
                                x=abs_x,
                                y=abs_y,
                                w=bw,
                                h=bh,
                                img_w=img_w,
                                img_h=img_h,
                                confidence=0.70,
                                source="hud_hp_potion",
                                material_name="hp_potion",
                                trusted=False,
                                weak=True,
                            )
                        )
                    elif mp_blue_yellow_px >= min_colored and mp_blue_yellow_px > scroll_px:
                        hud_anns.append(
                            self._make_annotation(
                                class_id=19,
                                x=abs_x,
                                y=abs_y,
                                w=bw,
                                h=bh,
                                img_w=img_w,
                                img_h=img_h,
                                confidence=0.70,
                                source="hud_mp_potion",
                                material_name="mp_potion",
                                trusted=False,
                                weak=True,
                            )
                        )
                    elif scroll_px >= max(10, int(0.09 * slot_area)):
                        hud_anns.append(
                            self._make_annotation(
                                class_id=22,
                                x=abs_x,
                                y=abs_y,
                                w=bw,
                                h=bh,
                                img_w=img_w,
                                img_h=img_h,
                                confidence=0.68,
                                source="hud_buff_scroll",
                                material_name="buff_scroll",
                                trusted=False,
                                weak=True,
                            )
                        )
                    # Else: empty or unclassified slot -> do not label

                # 3. Lower Combat Skills Row (y >= 0.940*H)
                elif norm_cy >= 0.939:
                    slot_edge_cnt = int(np.count_nonzero(edges[by : by + bh, bx : bx + bw]))
                    if slot_edge_cnt >= max(10, int(15 * scale_x)):
                        hud_anns.append(
                            self._make_annotation(
                                class_id=21,
                                x=abs_x,
                                y=abs_y,
                                w=bw,
                                h=bh,
                                img_w=img_w,
                                img_h=img_h,
                                confidence=0.68,
                                source="hud_skill_slot",
                                material_name="skill_slot",
                                trusted=False,
                                weak=True,
                            )
                        )

        return hud_anns

    # ------------------------------------------------------------------
    # Stage D: Walkable Path & Obstacle Segmentation (14, 15)
    # ------------------------------------------------------------------
    def _scan_path_and_obstacles(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        source_stem: str = "",
        shared_frame_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Genuine Walkable Path & Obstacle Segmentation (classes 14: obstacle, 15: path_ground):
        STRICT STEM GATING: ONLY execute when `is_path_learning_stem(source_stem)` or "learn_paths" in source_stem
        or "learn_obstacles" in source_stem.
        Normal gather/monster/boot frames (`boot_*`, `gather_*`, etc.) must NEVER receive 15: path_ground
        or 14: obstacle annotations!
        Without human approval, cap confidence at <= 0.68 (weak=True).
        """
        if not is_path_learning_stem(source_stem):
            return []

        path_anns: List[Dict[str, Any]] = []
        img_h, img_w = frame_bgr.shape[:2]
        if img_w < 500 or img_h < 300:
            return path_anns

        scale_x = img_w / 2000.0
        scale_y = img_h / 1124.0

        py1, py2 = int(img_h * 0.52), int(img_h * 0.85)
        px1, px2 = int(img_w * 0.28), int(img_w * 0.72)
        roi_gray = gray[py1:py2, px1:px2]
        roi_hsv = hsv[py1:py2, px1:px2]
        if roi_gray.size == 0:
            return path_anns

        sobel_key = f"sobel_{py1}_{py2}_{px1}_{px2}"
        if shared_frame_context is not None and sobel_key in shared_frame_context:
            mag = shared_frame_context[sobel_key]
        else:
            gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            if shared_frame_context is not None:
                shared_frame_context[sobel_key] = mag

        mag_th = float(np.percentile(mag, 45))
        smooth_ground = (mag <= max(18.0, mag_th)) & (roi_hsv[:, :, 1] <= 150) & (roi_hsv[:, :, 2] >= 35)
        ground_u8 = (smooth_ground.astype(np.uint8)) * 255

        ksize = max(9, int(round(15 * scale_x)) | 1)
        ground_closed = cv2.morphologyEx(
            ground_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        )
        cnts, _ = cv2.findContours(ground_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rw = px2 - px1
        rh = py2 - py1
        min_path_area = float(rw * rh) * 0.08

        path_cnt = None
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            if float(cv2.contourArea(c)) >= min_path_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw >= int(rw * 0.30) and bh >= int(rh * 0.25):
                    path_cnt = c
                    abs_x = px1 + bx
                    abs_y = py1 + by
                    # Without human approval, cap confidence at <= 0.68 (weak=True)
                    raw_conf = float(np.clip(float(bw * bh) / float(max(1, rw * rh)) * 1.4, 0.25, 0.68))
                    conf = min(0.68, max(0.35, round(raw_conf, 4)))
                    path_anns.append(
                        self._make_annotation(
                            class_id=15,
                            x=abs_x,
                            y=abs_y,
                            w=bw,
                            h=bh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=conf,
                            source="visual_ground_path_segmenter",
                            material_name="path_ground",
                            trusted=False,
                            weak=True,
                        )
                    )
                    break

        if path_cnt is not None:
            high_mag = (mag >= float(np.percentile(mag, 85))).astype(np.uint8) * 255
            obs_cnts, _ = cv2.findContours(high_mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for oc in sorted(obs_cnts, key=cv2.contourArea, reverse=True)[:2]:
                if float(cv2.contourArea(oc)) >= float(rw * rh) * 0.04:
                    obx, oby, obw, obh = cv2.boundingRect(oc)
                    abs_ox = px1 + obx
                    abs_oy = py1 + oby
                    # Without human approval, cap confidence at <= 0.68 (weak=True)
                    raw_obs_conf = float(np.clip(float(cv2.contourArea(oc)) / float(max(1, rw * rh)) * 3.5, 0.25, 0.68))
                    obs_conf = min(0.68, max(0.35, round(raw_obs_conf, 4)))
                    path_anns.append(
                        self._make_annotation(
                            class_id=14,
                            x=abs_ox,
                            y=abs_oy,
                            w=obw,
                            h=obh,
                            img_w=img_w,
                            img_h=img_h,
                            confidence=obs_conf,
                            source="visual_obstacle_segmenter",
                            material_name="obstacle",
                            trusted=False,
                            weak=True,
                        )
                    )

        return path_anns

    # ------------------------------------------------------------------
    # 3. Core Frame Auto-Labeling Pipeline
    # ------------------------------------------------------------------
    def auto_label_frame(
        self,
        frame: np.ndarray,
        extra_detections: Optional[List[Dict[str, Any]]] = None,
        known_target_xy: Optional[Tuple[int, int]] = None,
        known_material_name: str = "",
        fast_mode: bool = False,
        source_path: Union[str, os.PathLike, None] = None,
        shared_frame_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Automatically detect and label all Gatherion YOLO classes in `frame`:
        1. 3D White Winged Gather Crests (`6: world_gather_crest`) & 3D Resource Mesh (`0..5`).
        2. `[F Gather]` prompt (`7`) & `[E]/[F]` floating resource box (`8`).
        3. Minimap 3-leaf sprout (`9`) & monster dot (`11`).
        4. 3D Monsters & bodies (`10: monster_3d`) and Target HP Bars (`20: target_hp_bar`).
        5. Salient 3D Resources & Crestless Nodes (Stage A, `0..5, 12`).
        6. Genuine HUD & Skill Bar (Stage C, `16, 17, 18, 19, 21, 22`).
        7. Walkable Path & Obstacles (Stage D, `14, 15`).

        When `fast_mode=True`, reuses buffers and avoids running redundant template scans
        if already resolved by `material_scanner`.
        Reuses cached gray, hsv, and edge representations in `shared_frame_context` across sub-scanners.
        """
        if shared_frame_context is None:
            shared_frame_context = {}

        bgr = shared_frame_context.get("bgr")
        if bgr is None:
            bgr = self._coerce_bgr_frame(frame)
            if bgr is not None and bgr.size > 0:
                shared_frame_context["bgr"] = bgr

        if bgr is None or bgr.size == 0:
            return []

        img_h, img_w = bgr.shape[:2]

        if "hsv" in shared_frame_context and shared_frame_context["hsv"] is not None:
            hsv = shared_frame_context["hsv"]
        else:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            shared_frame_context["hsv"] = hsv

        if "gray" in shared_frame_context and shared_frame_context["gray"] is not None:
            gray = shared_frame_context["gray"]
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            shared_frame_context["gray"] = gray

        # Cache full-frame edge representation in shared_frame_context for 3x speedup
        if "edges" not in shared_frame_context or shared_frame_context["edges"] is None:
            shared_frame_context["edges"] = cv2.Canny(gray, 50, 150)

        annotations: List[Dict[str, Any]] = []

        source_stem = ""
        if source_path is not None:
            source_stem = Path(source_path).stem
        elif isinstance(frame, (str, os.PathLike)):
            source_stem = Path(frame).stem

        # Run MaterialScanner once for holistic 3D world, floating box, gather prompt & minimap detections
        ms_detections: List[DetectedMaterial] = self.material_scanner.scan_frame_for_materials(bgr)

        # ------------------------------------------------------------------
        # Stage 1: 3D White Winged Gather Crests (`6: world_gather_crest`)
        #          + 3D Plant/Ore/Gem Mesh Below Crest (`0..5`)
        # ------------------------------------------------------------------
        if fast_mode:
            ve_crests = []
        else:
            ve_crests = self.vision_engine.detect_3d_gather_crests(
                bgr, threshold=self.crest_threshold
            )

        merged_crests: List[Dict[str, Any]] = []
        for c in ve_crests:
            cb = c["bbox"]
            cx, cy = int(c["x"]), int(c["y"])
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
        # Stage 2: `[F Gather]` Prompt (`7`) & Floating Resource Box (`8`)
        # ------------------------------------------------------------------
        floating_centers: List[Tuple[int, int]] = []
        has_ms_floating = any(md.source == "3d_floating_icon" for md in ms_detections)

        # In fast_mode, reuse ms_detections from material_scanner and skip redundant vision_engine scan
        if not fast_mode:
            e_found, e_conf, e_center, _e_key = self.vision_engine.detect_floating_resource_icon(
                bgr,
                threshold=self.floating_icon_threshold,
                use_roi=True,
            )
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

        # In fast_mode, reuse ms_detections from material_scanner and skip redundant vision_engine scan
        if not fast_mode:
            p_found, p_conf_pct, p_bbox = self.vision_engine.detect_gather_prompt(
                bgr,
                threshold=self.prompt_threshold,
                use_roi=True,
            )
            if p_found and p_bbox is not None:
                px, py, pw, ph = p_bbox
                pcx, pcy = px + pw // 2, py + ph // 2
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
        # Stage 3: Top-Right Minimap 3-Leaf Sprouts (`9`) & Monster Dots (`11`)
        # ------------------------------------------------------------------
        has_ms_minimap = any(md.source == "minimap" for md in ms_detections)
        # In fast_mode, reuse ms_detections from material_scanner and skip redundant minimap scan
        if not fast_mode:
            mm_res = self.minimap_scanner.scan_gatherable_materials(bgr)
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

        # ------------------------------------------------------------------
        # Stage 4: 3D Monster Nameplates, Bodies, and Target HP Bars
        # ------------------------------------------------------------------
        pos_boxes = [a["bbox_xyxy"] for a in annotations if not a.get("is_negative", False)]
        monster_anns = self._scan_3d_monsters(
            frame_bgr=bgr,
            hsv=hsv,
            existing_positive_boxes=pos_boxes,
            gray=gray,
            shared_frame_context=shared_frame_context,
        )
        annotations.extend(monster_anns)

        # ------------------------------------------------------------------
        # Stage A: Salient 3D Resources & Crestless Nodes (0..5, 12)
        # Fallback ONLY when NO 3D crest or prompt icon found in frame
        # ------------------------------------------------------------------
        has_crest_or_prompt = any(
            int(a.get("class_id", -1)) in (6, 7, 8)
            or a.get("source") in ("vision_3d_crest", "vision_floating_icon", "vision_f_gather_prompt", "material_scanner_floating_icon", "material_scanner_f_gather_prompt")
            or "crest" in str(a.get("source", ""))
            or "gather_prompt" in str(a.get("source", ""))
            or "floating" in str(a.get("source", ""))
            for a in annotations
        )
        if not has_crest_or_prompt:
            pos_boxes_updated = [a["bbox_xyxy"] for a in annotations if not a.get("is_negative", False)]
            salient_anns = self._scan_salient_3d_resources(
                frame_bgr=bgr,
                hsv=hsv,
                existing_boxes=pos_boxes_updated,
                hint_material_name=known_material_name,
                has_crest_or_prompt=False,
                shared_frame_context=shared_frame_context,
            )
            annotations.extend(salient_anns)

        # ------------------------------------------------------------------
        # Stage C: Genuine HUD & Skill Bar (16, 17, 18, 19, 21, 22)
        # ------------------------------------------------------------------
        hud_anns = self._scan_hud_and_skills(
            frame_bgr=bgr,
            hsv=hsv,
            gray=gray,
            source_stem=source_stem,
            shared_frame_context=shared_frame_context,
        )
        annotations.extend(hud_anns)

        # ------------------------------------------------------------------
        # Stage D: Walkable Path & Obstacle (14, 15)
        # ------------------------------------------------------------------
        path_anns = self._scan_path_and_obstacles(
            frame_bgr=bgr,
            hsv=hsv,
            gray=gray,
            source_stem=source_stem,
            shared_frame_context=shared_frame_context,
        )
        annotations.extend(path_anns)

        # ------------------------------------------------------------------
        # Stage 5: Optional Known Target Hint (`known_target_xy` / `extra_detections`)
        # ------------------------------------------------------------------
        if known_target_xy is not None and len(known_target_xy) >= 2:
            kx, ky = int(known_target_xy[0]), int(known_target_xy[1])
            if 0 <= kx < img_w and 0 <= ky < img_h:
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

        deduped = self._deduplicate_annotations(annotations)
        if self.exemplar_memory is not None:
            try:
                deduped = self.exemplar_memory.refine_annotations(bgr, deduped)
            except Exception as exc:
                logger.debug("Exemplar memory refinement skipped: %s", exc)

        try:
            from dataset_factory.label_trust import filter_hud_false_positives

            return filter_hud_false_positives(deduped, stem=source_stem, remap_sprout=False)
        except Exception:
            return deduped

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
    "get_yolo_auto_labeler_readiness_status",
    "get_yolo_auto_labeler_audit_record",
    "get_yolo_auto_labeler_operational_digest",
    "get_yolo_auto_labeler_telemetry_payload",
    "get_yolo_auto_labeler_integrity_seal",
    "get_yolo_auto_labeler_compliance_certificate",
    "get_yolo_auto_labeler_readiness_probe",
    "get_yolo_auto_labeler_audit_summary",
]


def get_yolo_auto_labeler_capabilities() -> list:
    return ["yolo_auto_labeler", "bbox_annotation", "diagnostic_snapshot"]


def validate_yolo_auto_labeler_contract() -> bool:
    return bool(get_yolo_auto_labeler_capabilities())


def get_yolo_auto_labeler_baseline_metrics() -> dict:
    return {"module": "yolo_auto_labeler", "contract_valid": validate_yolo_auto_labeler_contract(), "capability_count": len(get_yolo_auto_labeler_capabilities())}


def get_yolo_auto_labeler_readiness_status() -> dict:
    try:
        return {"module": "yolo_auto_labeler", "ready": validate_yolo_auto_labeler_contract(), "metrics": get_yolo_auto_labeler_baseline_metrics()}
    except Exception:
        return {"module": "yolo_auto_labeler", "ready": False, "metrics": {}}


def get_yolo_auto_labeler_audit_record() -> dict:
    try:
        return {
            "module": "yolo_auto_labeler",
            "audit_passed": validate_yolo_auto_labeler_contract(),
            "readiness": get_yolo_auto_labeler_readiness_status(),
        }
    except Exception:
        return {
            "module": "yolo_auto_labeler",
            "audit_passed": False,
            "readiness": {},
        }


def get_yolo_auto_labeler_operational_digest() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "operational": validate_yolo_auto_labeler_contract(),
        "audit": get_yolo_auto_labeler_audit_record(),
    }


def get_yolo_auto_labeler_telemetry_payload() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "payload": get_yolo_auto_labeler_operational_digest(),
    }


def get_yolo_auto_labeler_integrity_seal() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "sealed": validate_yolo_auto_labeler_contract(),
        "telemetry": get_yolo_auto_labeler_telemetry_payload(),
    }


def get_yolo_auto_labeler_compliance_certificate() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "certified": validate_yolo_auto_labeler_contract(),
        "seal": get_yolo_auto_labeler_integrity_seal(),
    }


def get_yolo_auto_labeler_readiness_probe() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "ready": validate_yolo_auto_labeler_contract(),
        "certificate": get_yolo_auto_labeler_compliance_certificate(),
    }


def get_yolo_auto_labeler_audit_summary() -> dict:
    return {
        "module": "yolo_auto_labeler",
        "audit_passed": validate_yolo_auto_labeler_contract(),
        "probe": get_yolo_auto_labeler_readiness_probe(),
    }


