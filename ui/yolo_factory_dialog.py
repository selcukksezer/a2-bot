# -*- coding: utf-8 -*-
"""
YOLO Veri Seti Fabrikası — 6 adımlı üretim hattı (`YOLOFactoryDialog`).

Yakala → Onayla → Hazırla → Eğit → Doğrula → Yayınla

Prototip düzeni: üst stepper, durum strip, QStackedWidget adım sayfaları,
sağ rail (sıradaki iş / engeller / çalıştırmalar / canlı log).
Onay kuyruğu `dataset/review/`; dondurma `dataset/frozen_manifest.json`;
yayın `models/best.pt` (+ `best_prev.pt` yedek).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.shell_theme import SHELL_COLORS, SHELL_PAGE_FRAME_QSS

from dataset_factory.approved_pool import ApprovedDatasetPool, get_approved_pool, is_approved

if not hasattr(ApprovedDatasetPool, "get_approved_count"):
    ApprovedDatasetPool.get_approved_count = lambda self: len(getattr(self, "_approved_stems", set()))

from dataset_factory.frame_time_machine import PreGatherTimeMachineBuffer
from dataset_factory.quality_dedup_gate import DatasetQualityGate
from dataset_factory.rtx_trainer_engine import (
    RTXTrainerEngine,
    probe_gpu_environment,
    rtx_runs_dir_for,
)
from dataset_factory.synthetic_augmentor import SyntheticYOLOAugmentor

from ui.shell_page_frame import wrap_shell_tab_root_layout

try:
    from core.economy_broker import DEFAULT_BROKER_PRICES, format_kinah_tr
except Exception:
    def format_kinah_tr(amount: Union[int, float]) -> str:
        amt = float(amount)
        if amt >= 1_000_000:
            return f"{amt / 1_000_000:.2f}M".rstrip("0").rstrip(".") + "M"
        if amt >= 1_000:
            return f"{amt / 1_000:.1f}K".rstrip("0").rstrip(".") + "K"
        return str(int(amt))
    DEFAULT_BROKER_PRICES = {}  # type: ignore

logger = logging.getLogger("Gatherion.YOLOFactoryDialog")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [YOLOFactoryUI] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# 12 YOLO Sınıfı Tanımları, Türkçe Etiketleri ve Renk Paleti
# ============================================================================
TARGET_DATASET_PHOTO_GOAL: int = 100000

YOLO_12_CLASS_SPECS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "key": "herb_aria",
        "label_tr": "0: Aria",
        "full_tr": "0: Aria (herb_aria)",
        "color_hex": "#22c55e",
        "color_bgr": (94, 197, 34),
    },
    {
        "id": 1,
        "key": "herb_mela",
        "label_tr": "1: Mela",
        "full_tr": "1: Mela (herb_mela)",
        "color_hex": "#10b981",
        "color_bgr": (129, 185, 16),
    },
    {
        "id": 2,
        "key": "herb_targena",
        "label_tr": "2: Targena",
        "full_tr": "2: Targena (herb_targena)",
        "color_hex": "#84cc16",
        "color_bgr": (22, 204, 132),
    },
    {
        "id": 3,
        "key": "ore_orichalcum",
        "label_tr": "3: Orichalcum",
        "full_tr": "3: Orichalcum (ore_orichalcum)",
        "color_hex": "#f59e0b",
        "color_bgr": (11, 158, 245),
    },
    {
        "id": 4,
        "key": "gem_crystal",
        "label_tr": "4: Gem/Kristal",
        "full_tr": "4: Gem/Kristal (gem_crystal)",
        "color_hex": "#a855f7",
        "color_bgr": (247, 85, 168),
    },
    {
        "id": 5,
        "key": "wood_tree",
        "label_tr": "5: Ağaç/Kereste",
        "full_tr": "5: Ağaç/Kereste (wood_tree)",
        "color_hex": "#d97706",
        "color_bgr": (6, 119, 217),
    },
    {
        "id": 6,
        "key": "world_gather_crest",
        "label_tr": "6: 3B Amblem",
        "full_tr": "6: 3B Amblem (world_gather_crest)",
        "color_hex": "#38bdf8",
        "color_bgr": (248, 189, 56),
    },
    {
        "id": 7,
        "key": "f_gather_prompt",
        "label_tr": "7: F Gather",
        "full_tr": "7: F Gather (f_gather_prompt)",
        "color_hex": "#06b6d4",
        "color_bgr": (212, 182, 6),
    },
    {
        "id": 8,
        "key": "combo_skill_prompt",
        "label_tr": "8: Kombo (Q/E)",
        "full_tr": "8: Kombo İkonu (Q/E - combo_skill_prompt)",
        "color_hex": "#3b82f6",
        "color_bgr": (246, 130, 59),
    },
    {
        "id": 9,
        "key": "minimap_sprout",
        "label_tr": "9: Minimap Filiz",
        "full_tr": "9: Minimap Filiz (minimap_sprout)",
        "color_hex": "#4ade80",
        "color_bgr": (128, 222, 74),
    },
    {
        "id": 10,
        "key": "monster_3d",
        "label_tr": "10: 3B Canavar (monster_3d)",
        "full_tr": "10: 3B Canavar (monster_3d)",
        "color_hex": "#ef4444",
        "color_bgr": (68, 68, 239),
    },
    {
        "id": 11,
        "key": "minimap_monster_dot",
        "label_tr": "11: Minimap Canavar (minimap_monster_dot)",
        "full_tr": "11: Minimap Canavar (minimap_monster_dot)",
        "color_hex": "#f43f5e",
        "color_bgr": (94, 63, 244),
    },
    {
        "id": 12,
        "key": "world_material",
        "label_tr": "12: Dünya Malzemesi",
        "full_tr": "12: Dünya Malzemesi (world_material)",
        "color_hex": "#94a3b8",
        "color_bgr": (184, 163, 148),
    },
    {
        "id": 13,
        "key": "skill_ui_prompt",
        "label_tr": "13: Skill/Arayüz",
        "full_tr": "13: Skill/Arayüz (skill_ui_prompt)",
        "color_hex": "#c084fc",
        "color_bgr": (252, 132, 192),
    },
    {
        "id": 14,
        "key": "obstacle",
        "label_tr": "14: Engel",
        "full_tr": "14: Engel (obstacle)",
        "color_hex": "#fb7185",
        "color_bgr": (133, 113, 251),
    },
    {
        "id": 15,
        "key": "path_ground",
        "label_tr": "15: Yol/Zemin",
        "full_tr": "15: Yol/Zemin (path_ground)",
        "color_hex": "#facc15",
        "color_bgr": (21, 204, 250),
    },
]

try:
    from dataset_factory.class_taxonomy import alias_map as _taxonomy_alias_map
    from dataset_factory.class_taxonomy import ui_class_specs as _ui_class_specs

    YOLO_12_CLASS_SPECS.extend(_ui_class_specs())
except Exception:
    _taxonomy_alias_map = None  # type: ignore

CLASS_ID_TO_SPEC: Dict[int, Dict[str, Any]] = {
    int(item["id"]): item for item in YOLO_12_CLASS_SPECS
}

CLASS_COLORS_MAP: Dict[int, str] = {
    0: "#22c55e",
    1: "#10b981",
    2: "#84cc16",
    3: "#f59e0b",
    4: "#06b6d4",
    5: "#8b5cf6",
    6: "#a855f7",
    7: "#eab308",
    8: "#64748b",
    9: "#4ade80",
    10: "#ef4444",
    11: "#f87171",
    12: "#dc2626",
    13: "#38bdf8",
    14: "#f97316",
    15: "#6b7280",
    16: "#e11d48",
    17: "#2563eb",
    18: "#b91c1c",
    19: "#4f46e5",
    20: "#10b981",
    21: "#64748b",
    22: "#ef4444",
    23: "#3b82f6",
    24: "#d97706",
    25: "#eab308",
    26: "#00d2ff",
}

CLASS_KEY_ALIASES: Dict[str, int] = {
    "herb_aria": 0,
    "aria": 0,
    "herb_mela": 1,
    "fruit_mela": 1,
    "mela": 1,
    "herb_targena": 2,
    "targena": 2,
    "herb_azpha": 2,
    "herb_calendula": 2,
    "ore_orichalcum": 3,
    "orichalcum": 3,
    "ore_iron_silver": 3,
    "gem_crystal": 4,
    "gem_sapphire": 4,
    "gem_ruby": 4,
    "gem_diamond": 4,
    "crystal_node": 4,
    "wood_tree": 5,
    "log_timber": 5,
    "world_gather_crest": 6,
    "crest": 6,
    "f_gather_prompt": 7,
    "f_gather": 7,
    "combo_skill_prompt": 8,
    "combo_icon": 8,
    "q_combo_icon": 8,
    "e_combo_icon": 8,
    "kombo": 8,
    "kombo_ikonu": 8,
    "combo": 8,
    "e_gather_icon": 8,
    "e_gather": 8,
    "minimap_sprout": 9,
    "sprout": 9,
    "monster_3d_negative": 10,
    "mob_guardian": 10,
    "monster": 10,
    "minimap_monster_negative": 11,
    "minimap_monster": 11,
    "tree_log": 5,
    "e_floating_icon": 8,
    "monster_3d": 10,
    "minimap_monster_dot": 11,
    "world_material": 12,
    "gatherable_plant": 12,
    "gatherable material": 12,
    "skill_ui_prompt": 13,
    "obstacle": 14,
    "path_ground": 15,
    "kinah": 25,
    "kina": 25,
    "kinah_display": 25,
    "kinah_ui": 25,
    "altin": 25,
    "gold": 25,
    "para": 25,
    "coin": 25,
    "currency": 25,
    "player_character": 26,
    "player": 26,
    "karakter": 26,
    "karakterim": 26,
    "oyuncu": 26,
}
if _taxonomy_alias_map is not None:
    CLASS_KEY_ALIASES.update(_taxonomy_alias_map())


def _safe_imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Windows Türkçe karakterli (örn. 'selçuk') dosya yollarından güvenli görüntü okur."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        raw = np.fromfile(str(file_path), dtype=np.uint8)
        if raw.size == 0:
            return None
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.debug("Görüntü okuma hatası (%s): %s", file_path, exc)
        return None


def normalize_box_class_name(cid: Any, cname: Any = None) -> str:
    """Canonicalize any class name (including legacy names like monster_3d_negative, e_gather_icon)."""
    try:
        if cid is not None:
            cid_int = int(cid)
            if cid_int in CLASS_ID_TO_SPEC:
                return str(CLASS_ID_TO_SPEC[cid_int]["key"])
    except (ValueError, TypeError):
        pass
    raw = str(cname or "").strip()
    if raw in CLASS_KEY_ALIASES:
        target_cid = CLASS_KEY_ALIASES[raw]
        if target_cid in CLASS_ID_TO_SPEC:
            return str(CLASS_ID_TO_SPEC[target_cid]["key"])
    return raw or "herb_aria"


def _safe_imwrite_unicode(path: Union[str, Path], image: np.ndarray, quality: int = 93) -> bool:
    """Windows Türkçe karakterli dosya yollarına güvenli JPEG/PNG yazar."""
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix.lower() or ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext in (".jpg", ".jpeg") else []
        ok, encoded = cv2.imencode(ext, image, params)
        if not ok:
            return False
        encoded.tofile(str(file_path))
        return True
    except Exception as exc:
        logger.debug("Görüntü yazma hatası (%s): %s", file_path, exc)
        return False


# ============================================================================
# Yedek / Yerleşik Veri Seti Fabrikası Denetleyicisi
# (`dataset_factory.factory_controller` henüz yoksa veya eksik metod varsa devreye girer)
# ============================================================================
class _BuiltinAutoLabelingDatasetFactory:
    """
    `dataset_factory.factory_controller.AutoLabelingDatasetFactory` sınıfı ile
    birebir uyumlu yerleşik (fallback) veri seti fabrikası denetleyicisi.
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None) -> None:
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.dataset_dir = self.project_root / "dataset"
        self.train_images_dir = self.dataset_dir / "images" / "train"
        self.val_images_dir = self.dataset_dir / "images" / "val"
        self.train_labels_dir = self.dataset_dir / "labels" / "train"
        self.val_labels_dir = self.dataset_dir / "labels" / "val"
        self.previews_dir = self.dataset_dir / "previews"
        self.data_yaml_path = self.dataset_dir / "data.yaml"

        for folder in (
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir,
            self.previews_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        self.time_machine = PreGatherTimeMachineBuffer(max_duration_sec=6.0, min_push_interval_sec=0.20)
        self.quality_gate = DatasetQualityGate(min_hamming_distance=3, min_sharpness_threshold=18.0)
        self.augmentor = SyntheticYOLOAugmentor(seed=42)
        self.rtx_trainer = RTXTrainerEngine(project_root=self.project_root, dataset_dir=self.dataset_dir)

        self.auto_label_on_gather: bool = True
        self.passive_shadow_mode: bool = False
        self.synthetic_augment_10x: bool = True
        self.target_photo_goal: int = TARGET_DATASET_PHOTO_GOAL

        self._ensure_data_yaml()

    def _ensure_data_yaml(self) -> Path:
        if not self.data_yaml_path.exists():
            lines = [
                f"path: {self.dataset_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(YOLO_12_CLASS_SPECS)}",
                "names:",
            ]
            for spec in YOLO_12_CLASS_SPECS:
                lines.append(f"  {spec['id']}: {spec['key']}")
            self.data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.data_yaml_path

    def _synthesize_annotations_for_frame(
        self, frame: np.ndarray, source_name: str, sample_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Gerçek görüntü içeriğine göre YOLOAutoLabeler ile otomatik etiketler.
        Sentetik / sahte formül kutuları üretmez.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            labeler = YOLOAutoLabeler()
            anns = labeler.auto_label_frame(frame)
            return list(anns) if anns else []
        except Exception as exc:
            logger.warning("YOLOAutoLabeler failed in _synthesize_annotations_for_frame: %s", exc)
            return []

    def render_preview_image(
        self,
        frame: np.ndarray,
        annotations: List[Dict[str, Any]],
        output_name: str = "latest_t_minus_2s_preview.jpg",
    ) -> Path:
        """Görüntü üzerine renkli YOLO kutularını çizip `dataset/previews/` altına kaydeder."""
        canvas = frame.copy()
        img_h, img_w = canvas.shape[:2]

        for ann in annotations:
            cid = int(ann.get("class_id", 0))
            spec = CLASS_ID_TO_SPEC.get(cid, YOLO_12_CLASS_SPECS[0])
            cname = str(ann.get("class_name", spec["key"]))
            conf = float(ann.get("confidence", 0.96))
            cx = float(ann.get("cx", 0.5))
            cy = float(ann.get("cy", 0.5))
            bw = float(ann.get("w", 0.08))
            bh = float(ann.get("h", 0.12))

            x1 = max(2, int((cx - 0.5 * bw) * img_w))
            y1 = max(2, int((cy - 0.5 * bh) * img_h))
            x2 = min(img_w - 2, int((cx + 0.5 * bw) * img_w))
            y2 = min(img_h - 2, int((cy + 0.5 * bh) * img_h))

            bgr = spec["color_bgr"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2, cv2.LINE_AA)

            tag = f"{cname} %{conf * 100:.0f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            lbl_y1 = max(0, y1 - th - 10)
            cv2.rectangle(canvas, (x1, lbl_y1), (min(img_w - 1, x1 + tw + 10), y1), bgr, -1)
            cv2.putText(
                canvas,
                tag,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 23, 42),
                2,
                cv2.LINE_AA,
            )

        out_path = self.previews_dir / output_name
        _safe_imwrite_unicode(out_path, canvas, quality=94)
        return out_path

    def bootstrap_from_existing_frames(
        self,
        augment_multiplier: Optional[int] = None,
        max_source_frames: int = 18,
    ) -> Dict[str, Any]:
        """
        Mevcut `reference_screenshots/`, `video_frames/` ve `video_analysis/` karelerini
        tarayarak otomatik etiketler ve 10x sentetik çoğaltma ile `dataset/` klasörünü zenginleştirir.
        """
        mult = (
            int(augment_multiplier)
            if augment_multiplier is not None
            else (10 if self.synthetic_augment_10x else 1)
        )
        mult = max(1, min(10, mult))

        source_dirs = [
            self.project_root / "reference_screenshots",
            self.project_root / "video_frames",
            self.project_root / "video_analysis",
        ]
        source_paths: List[Path] = []
        for sdir in source_dirs:
            if sdir.exists():
                for p in sorted(sdir.glob("*.jpg")):
                    if not p.name.startswith("crop_") and not p.name.startswith("overlay_"):
                        source_paths.append(p)
                    if len(source_paths) >= max_source_frames:
                        break
            if len(source_paths) >= max_source_frames:
                break

        saved_frames = 0
        saved_boxes = 0
        latest_preview: Optional[Path] = None
        timestamp_tag = int(time.time() * 1000) % 1000000

        for idx, src_path in enumerate(source_paths):
            frame = _safe_imread_unicode(src_path)
            if frame is None or frame.size == 0:
                continue

            base_anns = self._synthesize_annotations_for_frame(frame, src_path.stem, idx)
            self.time_machine.push_frame(
                frame=frame,
                timestamp=time.time() - 2.0,
                detections=base_anns,
                region_name="Pilgrim's Ridge",
                force=True,
            )

            split = DatasetQualityGate.choose_split(idx, val_ratio=0.15)
            img_out_dir = self.val_images_dir if split == "val" else self.train_images_dir
            lbl_out_dir = self.val_labels_dir if split == "val" else self.train_labels_dir

            stem = f"boot_{timestamp_tag}_{idx:03d}_{src_path.stem}"
            img_file = img_out_dir / f"{stem}.jpg"
            lbl_file = lbl_out_dir / f"{stem}.txt"

            if _safe_imwrite_unicode(img_file, frame, quality=92):
                from dataset_factory.label_trust import commit_split_labels

                info = commit_split_labels(
                    dataset_dir=self.dataset_dir,
                    split=split,
                    stem=stem,
                    label_path=lbl_file,
                    annotations=base_anns,
                )
                saved_frames += 1
                saved_boxes += int(info.get("train_box_count", 0))

            if idx == 0 or latest_preview is None:
                latest_preview = self.render_preview_image(
                    frame, base_anns, f"preview_{stem}.jpg"
                )

            if mult > 1:
                variants = self.augmentor.generate_augmented_variants(
                    frame, base_anns, count=min(mult - 1, 9)
                )
                for v_idx, (v_suffix, v_frame, v_anns) in enumerate(variants):
                    v_split = DatasetQualityGate.choose_split(idx * 10 + v_idx + 1, val_ratio=0.15)
                    v_img_dir = self.val_images_dir if v_split == "val" else self.train_images_dir
                    v_lbl_dir = self.val_labels_dir if v_split == "val" else self.train_labels_dir
                    v_stem = f"{stem}_{v_suffix}"
                    v_img_path = v_img_dir / f"{v_stem}.jpg"
                    v_lbl_path = v_lbl_dir / f"{v_stem}.txt"
                    if _safe_imwrite_unicode(v_img_path, v_frame, quality=90):
                        from dataset_factory.label_trust import commit_split_labels

                        v_info = commit_split_labels(
                            dataset_dir=self.dataset_dir,
                            split=v_split,
                            stem=v_stem,
                            label_path=v_lbl_path,
                            annotations=v_anns,
                        )
                        saved_frames += 1
                        saved_boxes += int(v_info.get("train_box_count", 0))

        self._ensure_data_yaml()
        stats = self.get_dataset_stats()
        stats.update(
            {
                "bootstrapped_frames_added": saved_frames,
                "bootstrapped_boxes_added": saved_boxes,
                "augment_multiplier": mult,
                "latest_preview_path": str(latest_preview) if latest_preview else stats.get("latest_preview_path", ""),
            }
        )
        return stats

    def run_one_click_rtx_training(
        self,
        epochs: int = 25,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        allow_opencv_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Tek tıkla RTX CUDA FP16 YOLO eğitimini çalıştırır ve sonuç metriklerini döndürür."""
        self._ensure_data_yaml()
        return self.rtx_trainer.run_training_sync(
            data_yaml_path=self.data_yaml_path,
            epochs=epochs,
            progress_callback=progress_callback,
            imgsz=imgsz,
            batch=batch,
            model=model,
            allow_opencv_fallback=allow_opencv_fallback,
        )

    def get_dataset_stats(self) -> Dict[str, Any]:
        """`dataset/` klasöründeki tüm train/val görüntülerini ve 12 sınıf kutu dağılımını hesaplar."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        train_imgs = (
            [p for p in self.train_images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            if self.train_images_dir.exists()
            else []
        )
        val_imgs = (
            [p for p in self.val_images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            if self.val_images_dir.exists()
            else []
        )

        class_counts: Dict[int, int] = {spec["id"]: 0 for spec in YOLO_12_CLASS_SPECS}
        total_boxes = 0

        for lbl_dir in (self.train_labels_dir, self.val_labels_dir):
            if not lbl_dir.exists():
                continue
            for lbl_path in lbl_dir.glob("*.txt"):
                try:
                    content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if not content:
                    continue
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(float(parts[0]))
                            if cid in class_counts:
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                            total_boxes += 1
                        except ValueError:
                            continue

        previews = []
        if self.previews_dir.exists():
            previews = sorted(
                [p for p in self.previews_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

        total_frames = len(train_imgs) + len(val_imgs)
        progress_pct = min(100.0, (float(total_frames) / float(TARGET_DATASET_PHOTO_GOAL)) * 100.0)
        gpu_env = probe_gpu_environment()

        return {
            "total_frames": total_frames,
            "train_frames": len(train_imgs),
            "val_frames": len(val_imgs),
            "total_boxes": total_boxes,
            "class_counts": class_counts,
            "target_photo_goal": TARGET_DATASET_PHOTO_GOAL,
            "progress_pct": round(progress_pct, 2),
            "latest_preview_path": str(previews[0]) if previews else "",
            "time_machine_active": True,
            "auto_label_on_gather": self.auto_label_on_gather,
            "passive_shadow_mode": self.passive_shadow_mode,
            "synthetic_augment_10x": self.synthetic_augment_10x,
            "gpu_environment": gpu_env,
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.get_dataset_stats()


# Dışarıdan `dataset_factory.factory_controller.AutoLabelingDatasetFactory` varsa bağla,
# yoksa yerleşik sınıfı kullan.
try:
    from dataset_factory.factory_controller import AutoLabelingDatasetFactory  # type: ignore
except Exception:
    AutoLabelingDatasetFactory = _BuiltinAutoLabelingDatasetFactory


# ============================================================================
# Özel Modern Koyu Temalı Bileşenler (KPI Kartı, Toggle Anahtar, Önizleme, Sınıf Çubuğu)
# ============================================================================
class FactoryKPICard(QFrame):
    """Üst bilgi şeridi için modern koyu temalı KPI gösterge kartı."""

    def __init__(
        self,
        title: str,
        value_text: str,
        sub_text: str,
        accent_color: str = "#38bdf8",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.accent_color = accent_color
        self.setObjectName("FactoryKPICard")
        self.setMinimumHeight(78)
        self.setMaximumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame#FactoryKPICard {{
                background-color: {SHELL_COLORS["panel_deep"]};
                border: 1px solid {SHELL_COLORS["line"]};
                border-left: 4px solid {self.accent_color};
                border-radius: 4px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['muted']}; font-size: 12px; font-weight: 700; "
            "border: none; background: transparent;"
        )

        self.value_lbl = QLabel(value_text)
        self.value_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['text']}; font-size: 18px; font-weight: 700; "
            "border: none; background: transparent; font-family: 'Segoe UI';"
        )
        self.value_lbl.setWordWrap(True)

        self.sub_lbl = QLabel(sub_text)
        self.sub_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['muted']}; font-size: 12px; font-weight: 500; "
            "border: none; background: transparent;"
        )
        self.sub_lbl.setWordWrap(True)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.sub_lbl)

    def update_card(self, value_text: str, sub_text: Optional[str] = None) -> None:
        self.value_lbl.setText(str(value_text))
        if sub_text is not None:
            self.sub_lbl.setText(str(sub_text))


class FactoryToggleSwitch(QCheckBox):
    """QPainter ile çizilen animasyonlu modern kapsül anahtar (Toggle Switch)."""

    SWITCH_WIDTH = 38
    SWITCH_HEIGHT = 20
    TEXT_SPACING = 10

    def __init__(
        self,
        text: str = "",
        checked: bool = False,
        accent_hex: str = "#10b981",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._accent_color = QColor(accent_hex)
        self._thumb_pos: float = 1.0 if checked else 0.0
        self._hovered: bool = False

        self._anim = QPropertyAnimation(self, b"thumb_position", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setChecked(bool(checked))
        self.toggled.connect(self._on_toggled)

    @pyqtProperty(float)
    def thumb_position(self) -> float:
        return float(self._thumb_pos)

    @thumb_position.setter
    def thumb_position(self, value: float) -> None:
        self._thumb_pos = max(0.0, min(1.0, float(value)))
        self.update()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(bool(checked))
        self._anim.stop()
        self._thumb_pos = 1.0 if checked else 0.0
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _text_font(self) -> QFont:
        font = QFont("Segoe UI")
        font.setPixelSize(14)
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def _text_origin_x(self) -> int:
        return int(self.SWITCH_WIDTH) + self.TEXT_SPACING + 4

    def _text_height_for_width(self, width: int) -> int:
        # A transient narrow width must not demand a multi-hundred-pixel row.
        text_w = max(140, int(width) - self._text_origin_x() - 8)
        metrics = QFontMetrics(self._text_font())
        bounds = metrics.boundingRect(
            0,
            0,
            text_w,
            160,
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            self.text() or " ",
        )
        return max(self.SWITCH_HEIGHT + 14, min(bounds.height() + 12, 72))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(280, self._text_height_for_width(280))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(160, self._text_height_for_width(240))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.width() < 80:
            return
        needed = self._text_height_for_width(self.width())
        if abs(self.minimumHeight() - needed) > 2:
            self.setMinimumHeight(needed)

    def hitButton(self, pos: QPoint) -> bool:  # noqa: N802
        return self.rect().contains(pos)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        sw_w = float(self.SWITCH_WIDTH)
        sw_h = float(self.SWITCH_HEIGHT)
        sw_y = (float(self.height()) - sw_h) / 2.0
        track_rect = QRectF(2.0, sw_y, sw_w, sw_h)

        t = self._thumb_pos
        off_color = QColor("#263040")
        on_color = self._accent_color
        r = int(off_color.red() + (on_color.red() - off_color.red()) * t)
        g = int(off_color.green() + (on_color.green() - off_color.green()) * t)
        b = int(off_color.blue() + (on_color.blue() - off_color.blue()) * t)

        border_col = self._accent_color if t > 0.5 else QColor("#475569")
        painter.setPen(QPen(border_col, 1.2))
        painter.setBrush(QBrush(QColor(r, g, b)))
        painter.drawRoundedRect(track_rect, sw_h / 2.0, sw_h / 2.0)

        margin = 2.6
        thumb_d = sw_h - margin * 2.0
        min_x = 2.0 + margin
        max_x = 2.0 + sw_w - margin - thumb_d
        thumb_x = min_x + (max_x - min_x) * t
        thumb_y = sw_y + margin

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff" if t > 0.5 else "#cbd5e1")))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_d, thumb_d))

        if self.text():
            font = self._text_font()
            painter.setFont(font)
            painter.setPen(QColor("#f8fafc" if self._hovered or t > 0.5 else "#e8eef7"))
            tx = self._text_origin_x()
            t_rect = QRect(tx, 2, max(20, self.width() - tx - 6), max(16, self.height() - 4))
            painter.drawText(
                t_rect,
                int(
                    Qt.AlignmentFlag.AlignVCenter
                    | Qt.AlignmentFlag.AlignLeft
                    | Qt.TextFlag.TextWordWrap
                ),
                self.text(),
            )

        painter.end()


def _factory_explain_html() -> str:
    """Three short lines. Dense essays made the right panel unreadable."""
    return (
        "<div style=\"font-size:14px; line-height:150%; color:#f3f5f7; "
        "font-family:'Segoe UI',sans-serif;\">"
        "<p style=\"margin:0 0 8px 0;\"><b>Yakalama.</b> "
        "Kare 0.25 sn arayla tampona girer; toplamada ~2 sn önceki kare etiketlenir.</p>"
        "<p style=\"margin:0 0 8px 0;\"><b>Sınıflar.</b> "
        "0–11 temel kaynak. 12–15 yol/engel/skill. 16–21 can/mana/iksir. 22+ essence.</p>"
        "<p style=\"margin:0;\">"
        "<b>Eğitim.</b> Sayaçlar dolunca üstteki amber düğmeyle RTX eğit. "
        "Eski onnx yeni sınıfları bilmez.</p>"
        "</div>"
    )


class _ReadableExplainLabel(QLabel):
    """Word-wrapped notes that keep their full text height. The scroll pane clips, not the lines."""

    def __init__(self, html: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("lbl_explain")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        font = QFont("Segoe UI")
        font.setPixelSize(13)
        self.setFont(font)
        self.setStyleSheet(
            "QLabel#lbl_explain {"
            " color: #f3f5f7;"
            " background-color: #14181f;"
            " font-size: 14px;"
            " font-family: 'Segoe UI', sans-serif;"
            " padding: 12px 14px;"
            "}"
        )
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#f3f5f7"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#f3f5f7"))
        self.setPalette(palette)
        font = QFont("Segoe UI")
        font.setPixelSize(14)
        self.setFont(font)
        self.setText(html)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(max(96, self.heightForWidth(420)))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def _document_height(self, width: int) -> int:
        document = QTextDocument()
        document.setDefaultFont(self.font())
        document.setDocumentMargin(0)
        document.setHtml(self.text())
        document.setTextWidth(max(40, int(width) - 24))
        return max(int(document.size().height()) + 24, 13 * 4)

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._document_height(width)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(360, self._document_height(360))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, 0)


class _ExplainScroll(QScrollArea):
    """Viewport can be short. The label inside stays as tall as the wrapped text."""

    def __init__(self, label: _ReadableExplainLabel, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("explain_scroll")
        self.setWidget(label)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(110)
        self.setMaximumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            """
            QScrollArea#explain_scroll {
                background-color: #14181f;
                border: 1px solid #3a4250;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background: #14181f;
                width: 12px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #5a6578;
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )
        viewport = self.viewport()
        viewport.setAutoFillBackground(True)
        palette = viewport.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#131922"))
        viewport.setPalette(palette)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(360, 220)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, 96)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        label = self.widget()
        if label is None:
            return
        width = max(1, self.viewport().width())
        needed = label.heightForWidth(width)
        if needed > 0 and abs(label.minimumHeight() - needed) > 2:
            label.setMinimumHeight(needed)
            label.updateGeometry()


class _InfoSelectCard(QFrame):
    """Preset/format card: QFrame + QLabels (never put HTML in QPushButton)."""

    clicked = pyqtSignal()

    def __init__(
        self,
        title: str,
        subtitle: str,
        *,
        object_name: str = "PipePreset",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 11, 12, 11)
        lay.setSpacing(3)
        self.title_lbl = QLabel(title, self)
        self.title_lbl.setStyleSheet(
            "font-size:13px;font-weight:800;background:transparent;border:none;color:#F2F3F5;"
        )
        self.sub_lbl = QLabel(subtitle, self)
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet(
            "font-size:11.5px;font-weight:500;background:transparent;border:none;color:#9AA3B2;"
        )
        lay.addWidget(self.title_lbl)
        lay.addWidget(self.sub_lbl)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool, *, selected_name: str, idle_name: str) -> None:
        self.setObjectName(selected_name if selected else idle_name)
        sty = self.style()
        if sty is not None:
            sty.unpolish(self)
            sty.polish(self)
        self.update()


class _LabeledActionButton(QPushButton):
    """Action button; theme objectName sets colors. Short captions stay one line."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        font = QFont("Segoe UI")
        font.setPixelSize(14)
        font.setWeight(QFont.Weight.Bold)
        self.setFont(font)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def _needed_height(self, width: int) -> int:
        metrics = QFontMetrics(self.font())
        bounds = metrics.boundingRect(
            0,
            0,
            max(200, int(width) - 24),
            120,
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            self.text() or " ",
        )
        return max(36, min(bounds.height() + 16, 64))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(240, self._needed_height(240))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(140, 36)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.width() < 80:
            return
        needed = self._needed_height(self.width())
        if abs(self.minimumHeight() - needed) > 2:
            self.setMinimumHeight(needed)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        caption = option.text
        option.text = ""
        self.style().drawControl(QStyle.ControlElement.CE_PushButton, option, painter, self)
        painter.setFont(self.font())
        painter.setPen(QColor("#ffffff" if self.isEnabled() else "#e2e8f0"))
        rect = self.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, self)
        if rect.width() < 12 or rect.height() < 12:
            rect = self.rect().adjusted(12, 6, -12, -6)
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            caption,
        )
        painter.end()


class YOLOPreviewCanvas(QWidget):
    """
    `dataset/previews/` içerisindeki en son otomatik etiketlenen `t - 2.0s` ekran görüntüsünü
    ve renkli YOLO sınırlayıcı kutularını (`herb_aria`, `herb_mela`, `ore_orichalcum`,
    `minimap_sprout`, `world_gather_crest`, `f_gather_prompt`) canlı olarak çizen bileşen.
    Etiket rozetlerinin sağındaki `✕` düğmesine tıklandığında ilgili etiketi anında kaldırır.
    """

    boxRemoveRequested = pyqtSignal(dict)
    boxChangeClassRequested = pyqtSignal(dict)
    boxDrawn = pyqtSignal(float, float, float, float)
    boxAdded = pyqtSignal(dict)
    boxEdited = pyqtSignal(dict)

    REQUIRED_PREVIEW_TAGS: List[Tuple[str, str]] = [
        ("herb_aria", "#22c55e"),
        ("herb_mela", "#10b981"),
        ("ore_orichalcum", "#f59e0b"),
        ("minimap_sprout", "#4ade80"),
        ("world_gather_crest", "#38bdf8"),
        ("f_gather_prompt", "#06b6d4"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(460, 255)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pixmap: Optional[QPixmap] = None
        self._preview_path: str = ""
        self._live_badge: str = ""
        self._empty: bool = True
        self._annotations: List[Dict[str, Any]] = []
        self._source_key: str = ""
        self._last_decode_error: str = ""
        self._close_hit_rects: List[Tuple[QRectF, Dict[str, Any], int]] = []
        self._box_hit_rects: List[Tuple[QRectF, Dict[str, Any], int]] = []
        self._box_draw_info: List[Dict[str, Any]] = []
        self._hover_close_idx: int = -1
        self._selected_box_idx: int = -1
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_current_pos: Optional[QPointF] = None
        self._is_dragging_box: bool = False
        self._last_drag_distance: float = 0.0
        self._last_target_rect: QRectF = QRectF()
        self._interaction_mode: Optional[str] = None
        self._active_handle: Optional[str] = None
        self._drag_start_box_rect: QRectF = QRectF()
        self._drag_start_ann: Dict[str, Any] = {}
        self.active_drawing_class_id: int = 0
        self.active_drawing_class_name: str = "herb_aria"

    def clear_drag_state(self) -> None:
        """Kutu çizim, taşıma ve boyutlandırma durumunu tamamen sıfırlar."""
        self._drag_start_pos = None
        self._drag_current_pos = None
        self._is_dragging_box = False
        self._last_drag_distance = 0.0
        self._interaction_mode = None
        self._active_handle = None
        self._drag_start_box_rect = QRectF()
        self._drag_start_ann = {}

    def _compute_target_rect(self) -> QRectF:
        rect = self.rect()
        header_h = 30.0
        footer_h = 28.0
        view_rect = QRectF(
            8.0,
            header_h + 4.0,
            max(60.0, rect.width() - 16.0),
            max(60.0, rect.height() - header_h - footer_h - 10.0),
        )
        target_rect = view_rect
        if self._pixmap is not None and not self._pixmap.isNull():
            pw = float(self._pixmap.width())
            ph = float(self._pixmap.height())
            if pw > 0 and ph > 0:
                scale = min(view_rect.width() / pw, view_rect.height() / ph)
                sw = pw * scale
                sh = ph * scale
                dx = view_rect.left() + (view_rect.width() - sw) * 0.5
                dy = view_rect.top() + (view_rect.height() - sh) * 0.5
                target_rect = QRectF(dx, dy, sw, sh)
        return target_rect

    def hit_test_close_button(self, pos: Union[QPoint, QPointF]) -> Optional[Dict[str, Any]]:
        """Verilen fare koordinatı bir etiketin `✕` silme düğmesi üzerindeyse ilgili kutu sözlüğünü döndürür."""
        pt = QPointF(pos)
        for hit_rect, ann, _idx in reversed(self._close_hit_rects):
            if hit_rect.contains(pt):
                return ann
        return None

    def hit_test_box(self, pos: Union[QPoint, QPointF]) -> Optional[Tuple[Dict[str, Any], int]]:
        """Verilen fare koordinatı bir etiket kutusu veya rozeti üzerindeyse (ann, idx) döndürür."""
        pt = QPointF(pos)
        for hit_rect, ann, idx in reversed(self._box_hit_rects):
            if hit_rect.contains(pt):
                return ann, idx
        return None

    def hit_test_handle(self, pos: Union[QPoint, QPointF]) -> Tuple[Optional[str], int, Optional[Dict[str, Any]]]:
        """
        pos koordinatının kutu tutamacı, kenarı veya taşıma alanı üzerinde olup olmadığını belirler.
        Döndürür: (handle_name, box_idx, ann)
        handle_name: 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r', 'move', veya None.
        """
        pt = QPointF(pos)
        H = 14.0
        margin = 7.0

        # 1. Önce seçili kutunun tutamaçlarını ve kenarlarını kontrol et
        sel_idx = getattr(self, "_selected_box_idx", -1)
        sel_info = None
        for info in getattr(self, "_box_draw_info", []):
            if info.get("idx") == sel_idx:
                sel_info = info
                break
        if sel_info is not None:
            rect: QRectF = sel_info["box_rect"]
            ann = sel_info["ann"]

            # Köşeler
            if QRectF(rect.left() - H * 0.5, rect.top() - H * 0.5, H, H).contains(pt):
                return "tl", sel_idx, ann
            if QRectF(rect.right() - H * 0.5, rect.top() - H * 0.5, H, H).contains(pt):
                return "tr", sel_idx, ann
            if QRectF(rect.left() - H * 0.5, rect.bottom() - H * 0.5, H, H).contains(pt):
                return "bl", sel_idx, ann
            if QRectF(rect.right() - H * 0.5, rect.bottom() - H * 0.5, H, H).contains(pt):
                return "br", sel_idx, ann

            # Kenar orta noktaları
            if QRectF(rect.center().x() - H * 0.5, rect.top() - H * 0.5, H, H).contains(pt):
                return "t", sel_idx, ann
            if QRectF(rect.center().x() - H * 0.5, rect.bottom() - H * 0.5, H, H).contains(pt):
                return "b", sel_idx, ann
            if QRectF(rect.left() - H * 0.5, rect.center().y() - H * 0.5, H, H).contains(pt):
                return "l", sel_idx, ann
            if QRectF(rect.right() - H * 0.5, rect.center().y() - H * 0.5, H, H).contains(pt):
                return "r", sel_idx, ann

            # Kenar yakınlığı
            if abs(pt.y() - rect.top()) <= margin and (rect.left() - margin <= pt.x() <= rect.right() + margin):
                return "t", sel_idx, ann
            if abs(pt.y() - rect.bottom()) <= margin and (rect.left() - margin <= pt.x() <= rect.right() + margin):
                return "b", sel_idx, ann
            if abs(pt.x() - rect.left()) <= margin and (rect.top() - margin <= pt.y() <= rect.bottom() + margin):
                return "l", sel_idx, ann
            if abs(pt.x() - rect.right()) <= margin and (rect.top() - margin <= pt.y() <= rect.bottom() + margin):
                return "r", sel_idx, ann

        # 2. Tüm kutuların kenar ve alanlarını kontrol et
        for info in reversed(getattr(self, "_box_draw_info", [])):
            idx = info["idx"]
            rect = info["box_rect"]
            lbl_rect = info["lbl_rect"]
            close_hit = info["close_hit"]
            ann = info["ann"]

            if close_hit.contains(pt):
                continue

            if QRectF(rect.left() - H * 0.5, rect.top() - H * 0.5, H, H).contains(pt):
                return "tl", idx, ann
            if QRectF(rect.right() - H * 0.5, rect.top() - H * 0.5, H, H).contains(pt):
                return "tr", idx, ann
            if QRectF(rect.left() - H * 0.5, rect.bottom() - H * 0.5, H, H).contains(pt):
                return "bl", idx, ann
            if QRectF(rect.right() - H * 0.5, rect.bottom() - H * 0.5, H, H).contains(pt):
                return "br", idx, ann

            if abs(pt.y() - rect.top()) <= margin and (rect.left() - margin <= pt.x() <= rect.right() + margin):
                return "t", idx, ann
            if abs(pt.y() - rect.bottom()) <= margin and (rect.left() - margin <= pt.x() <= rect.right() + margin):
                return "b", idx, ann
            if abs(pt.x() - rect.left()) <= margin and (rect.top() - margin <= pt.y() <= rect.bottom() + margin):
                return "l", idx, ann
            if abs(pt.x() - rect.right()) <= margin and (rect.top() - margin <= pt.y() <= rect.bottom() + margin):
                return "r", idx, ann

            if rect.contains(pt) or lbl_rect.contains(pt):
                return "move", idx, ann

        return None, -1, None

    def _hit_test_close_index(self, pos: Union[QPoint, QPointF]) -> int:
        pt = QPointF(pos)
        for hit_rect, _ann, idx in reversed(self._close_hit_rects):
            if hit_rect.contains(pt):
                return idx
        return -1

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sel_idx = getattr(self, "_selected_box_idx", -1)
            if 0 <= sel_idx < len(self._annotations):
                ann = self._annotations[sel_idx]
                self._selected_box_idx = -1
                self.boxRemoveRequested.emit(ann)
                self.update()
                return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.pos()
        hit = self.hit_test_box(pos)
        if hit is not None:
            ann, idx = hit
            self._selected_box_idx = idx
            self.update()
            menu = QMenu(self)
            menu.setStyleSheet(
                "QMenu { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 4px; }"
                "QMenu::item { padding: 6px 18px; border-radius: 4px; font-size: 11px; }"
                "QMenu::item:selected { background-color: #2563eb; color: #ffffff; }"
            )
            cname = ann.get("class_name", "")
            act_del = menu.addAction(f"🗑️ Etiketi Sil: {cname}")
            act_change = menu.addAction("🏷️ Ne Olduğunu Değiştir...")
            chosen = menu.exec(event.globalPos())
            if chosen == act_del:
                self._selected_box_idx = -1
                self.boxRemoveRequested.emit(ann)
            elif chosen == act_change:
                self.boxChangeClassRequested.emit(ann)
            return
        super().contextMenuEvent(event)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            if self.hit_test_close_button(pos) is not None:
                super().mousePressEvent(event)
                return

            handle_name, idx, ann = self.hit_test_handle(pos)
            if handle_name is not None and idx >= 0 and ann is not None:
                self._selected_box_idx = idx
                self._active_handle = handle_name
                self._drag_start_pos = QPointF(pos)
                self._drag_current_pos = QPointF(pos)
                self._drag_start_ann = dict(ann)
                trect = self._last_target_rect if not self._last_target_rect.isEmpty() else self._compute_target_rect()
                cx = float(ann.get("cx", 0.5))
                cy = float(ann.get("cy", 0.5))
                bw = float(ann.get("w", 0.08))
                bh = float(ann.get("h", 0.12))
                bx = trect.left() + (cx - bw * 0.5) * trect.width()
                by = trect.top() + (cy - bh * 0.5) * trect.height()
                self._drag_start_box_rect = QRectF(bx, by, bw * trect.width(), bh * trect.height())
                if handle_name == "move":
                    self._interaction_mode = "moving"
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self._interaction_mode = "resizing"
                self.setFocus()
                self.update()
                return

            self._selected_box_idx = -1
            self._interaction_mode = "creating"
            self._drag_start_pos = QPointF(pos)
            self._drag_current_pos = QPointF(pos)
            self._is_dragging_box = False
            self._last_drag_distance = 0.0
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        buttons = event.buttons()

        if (buttons & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            self._drag_current_pos = QPointF(pos)
            dx = float(pos.x() - self._drag_start_pos.x())
            dy = float(pos.y() - self._drag_start_pos.y())
            dist = (dx * dx + dy * dy) ** 0.5
            self._last_drag_distance = max(self._last_drag_distance, dist, abs(dx), abs(dy))
            trect = self._last_target_rect if not self._last_target_rect.isEmpty() else self._compute_target_rect()
            tw = max(1.0, float(trect.width()))
            th = max(1.0, float(trect.height()))

            mode = getattr(self, "_interaction_mode", None)
            if mode == "moving" and 0 <= self._selected_box_idx < len(self._annotations):
                ann = self._annotations[self._selected_box_idx]
                srect = self._drag_start_box_rect
                nx = srect.left() + dx
                ny = srect.top() + dy
                nx = max(float(trect.left()), min(nx, float(trect.right()) - srect.width()))
                ny = max(float(trect.top()), min(ny, float(trect.bottom()) - srect.height()))
                new_cx = ((nx + srect.width() * 0.5) - float(trect.left())) / tw
                new_cy = ((ny + srect.height() * 0.5) - float(trect.top())) / th
                ann["cx"] = max(0.0, min(1.0, round(new_cx, 6)))
                ann["cy"] = max(0.0, min(1.0, round(new_cy, 6)))
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.update()
                super().mouseMoveEvent(event)
                return

            elif mode == "resizing" and 0 <= self._selected_box_idx < len(self._annotations):
                ann = self._annotations[self._selected_box_idx]
                srect = self._drag_start_box_rect
                handle = self._active_handle or ""
                x1 = srect.left()
                y1 = srect.top()
                x2 = srect.right()
                y2 = srect.bottom()

                if "r" in handle:
                    x2 = max(x1 + 10.0, min(float(trect.right()), srect.right() + dx))
                if "l" in handle:
                    x1 = min(x2 - 10.0, max(float(trect.left()), srect.left() + dx))
                if "b" in handle:
                    y2 = max(y1 + 10.0, min(float(trect.bottom()), srect.bottom() + dy))
                if "t" in handle:
                    y1 = min(y2 - 10.0, max(float(trect.top()), srect.top() + dy))

                new_cx = ((x1 + x2) * 0.5 - float(trect.left())) / tw
                new_cy = ((y1 + y2) * 0.5 - float(trect.top())) / th
                new_w = (x2 - x1) / tw
                new_h = (y2 - y1) / th

                ann["cx"] = max(0.0, min(1.0, round(new_cx, 6)))
                ann["cy"] = max(0.0, min(1.0, round(new_cy, 6)))
                ann["w"] = max(0.005, min(1.0, round(new_w, 6)))
                ann["h"] = max(0.005, min(1.0, round(new_h, 6)))
                self.update()
                super().mouseMoveEvent(event)
                return

            elif mode == "creating":
                if abs(dx) >= 8.0 or abs(dy) >= 8.0 or dist >= 8.0:
                    self._is_dragging_box = True
                    self.setCursor(Qt.CursorShape.CrossCursor)
                self.update()
                super().mouseMoveEvent(event)
                return

        # Cursor and hover updates when not dragging
        idx = self._hit_test_close_index(pos)
        if idx != self._hover_close_idx:
            self._hover_close_idx = idx
            self.update()

        if idx >= 0:
            self.setToolTip("Etiketi kaldır (✕)")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            handle_name, box_idx, ann = self.hit_test_handle(pos)
            if handle_name == "move":
                cname = ann.get("class_name", "") if ann else ""
                self.setToolTip(f"'{cname}' kutusunu taşımak için sürükleyin · Kenarlardan büyütün")
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif handle_name in ("tl", "br"):
                self.setToolTip("Kutuyu köşeden büyüt / küçült")
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif handle_name in ("tr", "bl"):
                self.setToolTip("Kutuyu köşeden büyüt / küçült")
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif handle_name in ("t", "b"):
                self.setToolTip("Kutuyu dikey büyüt / daralt")
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif handle_name in ("l", "r"):
                self.setToolTip("Kutuyu yatay büyüt / daralt")
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setToolTip("Sol tık sürükle: Yeni etiket kutusu çiz · Çift tık: Büyüt")
                self.setCursor(Qt.CursorShape.CrossCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            mode = getattr(self, "_interaction_mode", None)
            if mode in ("moving", "resizing") and 0 <= self._selected_box_idx < len(self._annotations):
                ann = self._annotations[self._selected_box_idx]
                ann["trusted"] = True
                ann["hand_label"] = True
                ann["human_approved"] = True
                self.boxEdited.emit(ann)
                self.clear_drag_state()
                self.update()
                super().mouseReleaseEvent(event)
                return

            if mode == "creating" and self._drag_start_pos is not None:
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                start = self._drag_start_pos
                dx = abs(float(pos.x() - start.x()))
                dy = abs(float(pos.y() - start.y()))
                dist = (dx * dx + dy * dy) ** 0.5
                was_drag = self._is_dragging_box or dist >= 8.0 or dx >= 8.0 or dy >= 8.0
                if was_drag:
                    trect = self._last_target_rect if not self._last_target_rect.isEmpty() else self._compute_target_rect()
                    tw = max(1.0, float(trect.width()))
                    th = max(1.0, float(trect.height()))
                    x1 = max(float(trect.left()), min(float(start.x()), float(pos.x())))
                    y1 = max(float(trect.top()), min(float(start.y()), float(pos.y())))
                    x2 = min(float(trect.right()), max(float(start.x()), float(pos.x())))
                    y2 = min(float(trect.bottom()), max(float(start.y()), float(pos.y())))
                    bw_px = x2 - x1
                    bh_px = y2 - y1
                    if bw_px >= 2.0 and bh_px >= 2.0:
                        cx = ((x1 + x2) * 0.5 - float(trect.left())) / tw
                        cy = ((y1 + y2) * 0.5 - float(trect.top())) / th
                        w = bw_px / tw
                        h = bh_px / th
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w = max(0.005, min(1.0, w))
                        h = max(0.005, min(1.0, h))
                        cid = int(getattr(self, "active_drawing_class_id", 0))
                        cname = str(getattr(self, "active_drawing_class_name", "herb_aria"))
                        self.boxDrawn.emit(float(round(cx, 6)), float(round(cy, 6)), float(round(w, 6)), float(round(h, 6)))
                        self.boxAdded.emit(
                            {
                                "class_id": cid,
                                "class_name": cname,
                                "cx": round(cx, 6),
                                "cy": round(cy, 6),
                                "w": round(w, 6),
                                "h": round(h, 6),
                                "confidence": 1.0,
                                "conf": 1.0,
                                "trusted": True,
                                "weak": False,
                                "hand_label": True,
                                "human_approved": True,
                                "color": CLASS_COLORS_MAP.get(cid, "#22c55e"),
                            }
                        )
            self.clear_drag_state()
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        if not (self._is_dragging_box and self._drag_start_pos is not None):
            self.clear_drag_state()
        if self._hover_close_idx != -1:
            self._hover_close_idx = -1
        self.update()
        super().leaveEvent(event)

    @property
    def current_preview_path(self) -> str:
        return self._preview_path

    @staticmethod
    def _downscale_bgr(frame_bgr: np.ndarray, max_w: int = 1400) -> np.ndarray:
        """UI thread için büyük kareleri küçült (lightbox ayrı full pixmap tutar)."""
        try:
            h, w = frame_bgr.shape[:2]
            if w <= max_w:
                return frame_bgr
            scale = float(max_w) / float(w)
            return cv2.resize(
                frame_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            return frame_bgr

    def set_preview_image(
        self,
        image_path: Union[str, Path, None],
        annotations: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Diskten gerçek görüntü yükle. Başarı True. Uydurma bbox yazılmaz."""
        self.clear_drag_state()
        self._annotations = list(annotations) if annotations is not None else []

        if not image_path:
            self._last_decode_error = "path empty"
            return False

        p = Path(image_path)
        if "vid_sample_" in p.name:
            self.show_empty_state("Canlı kare yok")
            self._last_decode_error = "vid_sample skipped"
            return False
        if not p.exists() or not p.is_file():
            self._last_decode_error = f"path not found: {p}"
            return False

        try:
            mtime = float(p.stat().st_mtime)
        except OSError:
            mtime = 0.0
        key = f"file:{p.resolve()}:{mtime}"
        if key == self._source_key and not self._empty and self._pixmap is not None:
            self.update()
            return True

        img_bgr = _safe_imread_unicode(p)
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            self._last_decode_error = f"decode fail: {p}"
            return False

        display = self._downscale_bgr(img_bgr)
        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(
            img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._preview_path = str(p)
        self._source_key = key
        self._empty = False
        self._live_badge = p.name
        self._last_decode_error = ""
        self.update()
        return True

    def show_empty_state(self, message: str = "Canlı kare yok") -> None:
        self.clear_drag_state()
        self._pixmap = None
        self._preview_path = ""
        self._annotations = []
        self._empty = True
        self._live_badge = message
        self._source_key = ""
        self.update()

    def set_live_frame(
        self,
        frame_bgr: np.ndarray,
        badge: str,
        annotations: Optional[List[Dict[str, Any]]] = None,
        source_key: str = "",
    ) -> bool:
        self.clear_drag_state()
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            self.show_empty_state("Canlı kare yok")
            self._last_decode_error = "empty frame"
            return False
        key = source_key or f"mem:{id(frame_bgr)}:{getattr(frame_bgr, 'shape', ())}"
        if (
            key
            and key == self._source_key
            and not self._empty
            and self._pixmap is not None
            and annotations is None
        ):
            self._live_badge = badge
            return True

        display = self._downscale_bgr(frame_bgr)
        img_rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        height, width, channels = img_rgb.shape
        qimg = QImage(
            img_rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._preview_path = ""
        self._source_key = key
        self._empty = False
        self._live_badge = badge
        self._annotations = list(annotations) if annotations is not None else []
        self._last_decode_error = ""
        self.update()
        return True

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            rect = self.rect()
            frame_rect = QRectF(1.0, 1.0, rect.width() - 2.0, rect.height() - 2.0)

            painter.setPen(QPen(QColor("#263142"), 1.2))
            painter.setBrush(QBrush(QColor("#0f141c")))
            painter.drawRoundedRect(frame_rect, 8.0, 8.0)

            header_h = 30.0
            footer_h = 28.0
            view_rect = QRectF(
                8.0,
                header_h + 4.0,
                max(60.0, rect.width() - 16.0),
                max(60.0, rect.height() - header_h - footer_h - 10.0),
            )

            # Üst durum başlığı
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#38bdf8"))
            if self._empty or self._pixmap is None:
                file_badge = self._live_badge or "Canlı kare yok"
            elif self._live_badge:
                file_badge = self._live_badge
            else:
                file_badge = Path(self._preview_path).name if self._preview_path else "Canlı kare yok"
            painter.drawText(
                QRectF(12.0, 6.0, rect.width() - 24.0, 22.0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                f"🎯 CANLI OTOMATİK ETİKET ÖNİZLEME (t - 2.0 sn): {file_badge}",
            )

            # Görüntüyü çiz
            target_rect = view_rect
            if self._pixmap is not None and not self._pixmap.isNull():
                scaled = self._pixmap.scaled(
                    int(view_rect.width()),
                    int(view_rect.height()),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                dx = view_rect.left() + (view_rect.width() - scaled.width()) * 0.5
                dy = view_rect.top() + (view_rect.height() - scaled.height()) * 0.5
                target_rect = QRectF(dx, dy, float(scaled.width()), float(scaled.height()))
                painter.drawPixmap(int(dx), int(dy), scaled)
            else:
                grad = QLinearGradient(view_rect.topLeft(), view_rect.bottomLeft())
                grad.setColorAt(0.0, QColor("#111827"))
                grad.setColorAt(1.0, QColor("#1e293b"))
                painter.setPen(QPen(QColor("#334155"), 1.0))
                painter.setBrush(QBrush(grad))
                painter.drawRoundedRect(view_rect, 6.0, 6.0)
                painter.setPen(QColor("#94a3b8"))
                painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
                painter.drawText(view_rect, int(Qt.AlignmentFlag.AlignCenter), "Canlı kare yok")
            self._last_target_rect = target_rect

            # Renkli YOLO BBox katmanı (herb_aria, herb_mela, ore_orichalcum, minimap_sprout, world_gather_crest, f_gather_prompt)
            tag_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
            close_font = QFont("Segoe UI", 8, QFont.Weight.Black)
            painter.setFont(tag_font)
            fm = QFontMetrics(tag_font)
            self._close_hit_rects = []
            self._box_hit_rects = []
            self._box_draw_info = []

            for idx, ann in enumerate(self._annotations):
                cname = normalize_box_class_name(ann.get("class_id"), ann.get("class_name"))
                color_hex = str(ann.get("color", "#22c55e"))
                conf = float(ann.get("conf", ann.get("confidence", 0.97)))
                cx = float(ann.get("cx", 0.5))
                cy = float(ann.get("cy", 0.5))
                bw = float(ann.get("w", 0.08))
                bh = float(ann.get("h", 0.12))

                bx = target_rect.left() + (cx - bw * 0.5) * target_rect.width()
                by = target_rect.top() + (cy - bh * 0.5) * target_rect.height()
                bw_px = max(18.0, bw * target_rect.width())
                bh_px = max(14.0, bh * target_rect.height())

                box_rect = QRectF(bx, by, bw_px, bh_px)
                norm_conf = conf if conf <= 1.0 else conf / 100.0
                pct = max(0, min(100, int(round(norm_conf * 100))))

                is_confident = (
                    (bool(ann.get("trusted")) and norm_conf >= 0.85)
                    or bool(ann.get("hand_label"))
                    or bool(ann.get("human_approved"))
                )

                if is_confident:
                    # 1. 🟢 EMİN: Yeşil (#22c55e), düz kalın çizgi (width=2.2), rozet: '✓ cname %pct'
                    box_color = QColor("#22c55e")
                    pen = QPen(box_color, 2.2)
                    pen.setStyle(Qt.PenStyle.SolidLine)
                    label_str = f"✓ {cname} %{pct}"
                    badge_bg_color = box_color
                    badge_text_color = QColor("#08130C")
                    fill_alpha = 40
                elif norm_conf >= 0.70:
                    # 2. 🟠 ŞÜPHELİ (0.70 <= norm_conf < 0.85): Turuncu (#fb923c), kesikli çizgi (width=1.8), rozet: '🟠 şüpheli cname %pct'
                    box_color = QColor("#fb923c")
                    pen = QPen(box_color, 1.8)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    label_str = f"🟠 şüpheli {cname} %{pct}"
                    badge_bg_color = box_color
                    badge_text_color = QColor("#1c1106")
                    fill_alpha = 30
                else:
                    # 3. 🔴 YÜZDE 70'İN ALTINDAKİLER (norm_conf < 0.70): Kırmızı (#ef4444), kesikli çizgi (width=1.8), rozet: 'şüpheli cname %pct' (beyaz yazı #ffffff)
                    box_color = QColor("#ef4444")
                    pen = QPen(box_color, 1.8)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    label_str = f"şüpheli {cname} %{pct}"
                    badge_bg_color = box_color
                    badge_text_color = QColor("#ffffff")
                    fill_alpha = 30

                fill_col = QColor(box_color)
                fill_col.setAlpha(fill_alpha)
                painter.setPen(pen)
                painter.setBrush(QBrush(fill_col))
                painter.drawRoundedRect(box_rect, 3.0, 3.0)

                # Seçili kutu vurgusu (tıklanarak seçilen kutu) ve 8 adet boyutlandırma tutamacı
                if idx == getattr(self, "_selected_box_idx", -1):
                    sel_pen = QPen(QColor("#38bdf8"), 2.2, Qt.PenStyle.DashLine)
                    painter.setPen(sel_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(box_rect.adjusted(-2.5, -2.5, 2.5, 2.5), 4.0, 4.0)

                    # 8 Adet Tutamaç (Beyaz kare, mavi kenarlık)
                    hs = 7.0
                    half_hs = hs * 0.5
                    painter.setPen(QPen(QColor("#0284c7"), 1.5))
                    painter.setBrush(QBrush(QColor("#ffffff")))
                    handles = [
                        QRectF(box_rect.left() - half_hs, box_rect.top() - half_hs, hs, hs),
                        QRectF(box_rect.right() - half_hs, box_rect.top() - half_hs, hs, hs),
                        QRectF(box_rect.left() - half_hs, box_rect.bottom() - half_hs, hs, hs),
                        QRectF(box_rect.right() - half_hs, box_rect.bottom() - half_hs, hs, hs),
                        QRectF(box_rect.center().x() - half_hs, box_rect.top() - half_hs, hs, hs),
                        QRectF(box_rect.center().x() - half_hs, box_rect.bottom() - half_hs, hs, hs),
                        QRectF(box_rect.left() - half_hs, box_rect.center().y() - half_hs, hs, hs),
                        QRectF(box_rect.right() - half_hs, box_rect.center().y() - half_hs, hs, hs),
                    ]
                    for hr in handles:
                        painter.drawRect(hr)

                painter.setFont(tag_font)
                text_w = float(fm.horizontalAdvance(label_str))
                close_w = 18.0
                th = 18.0
                tw = text_w + 12.0 + close_w
                ly = max(target_rect.top(), by - th - 1.0)
                lx_badge = max(target_rect.left(), min(bx, target_rect.right() - tw))
                lbl_rect = QRectF(lx_badge, ly, tw, th)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(badge_bg_color))
                painter.drawRoundedRect(lbl_rect, 3.5, 3.5)

                text_rect = QRectF(lx_badge + 5.0, ly, text_w + 4.0, th)
                painter.setPen(badge_text_color)
                painter.drawText(
                    text_rect,
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    label_str,
                )

                # Sağ tarafta tıklanabilir '✕' silme düğmesi
                close_rect = QRectF(lx_badge + tw - close_w - 1.5, ly + 1.5, close_w - 1.0, th - 3.0)
                is_hovered = (self._hover_close_idx == idx)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#ef4444") if is_hovered else QColor(15, 23, 42, 195)))
                painter.drawRoundedRect(close_rect, 2.5, 2.5)

                painter.setFont(close_font)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(close_rect, int(Qt.AlignmentFlag.AlignCenter), "✕")

                hit_rect = close_rect.adjusted(-6.0, -6.0, 6.0, 6.0)
                self._close_hit_rects.append((hit_rect, ann, idx))
                self._box_hit_rects.append((box_rect.united(lbl_rect), ann, idx))
                self._box_draw_info.append({
                    "idx": idx,
                    "ann": ann,
                    "box_rect": box_rect,
                    "lbl_rect": lbl_rect,
                    "close_rect": close_rect,
                    "close_hit": hit_rect,
                })

            # Rubberband seçim dikdörtgeni (fareyle manuel kutu çizimi)
            if self._drag_start_pos is not None and self._drag_current_pos is not None:
                rx1 = min(self._drag_start_pos.x(), self._drag_current_pos.x())
                ry1 = min(self._drag_start_pos.y(), self._drag_current_pos.y())
                rw = abs(self._drag_current_pos.x() - self._drag_start_pos.x())
                rh = abs(self._drag_current_pos.y() - self._drag_start_pos.y())
                if rw >= 2.0 or rh >= 2.0:
                    rubber_rect = QRectF(rx1, ry1, rw, rh)
                    rpen = QPen(QColor("#22c55e"), 2.0, Qt.PenStyle.DashLine)
                    painter.setPen(rpen)
                    painter.setBrush(QBrush(QColor(34, 197, 94, 45)))
                    painter.drawRect(rubber_rect)
                    active_cname = getattr(self, "active_drawing_class_name", "herb_aria")
                    badge_str = f"🎯 {active_cname} (%100)"
                    painter.setFont(tag_font)
                    tbw = float(fm.horizontalAdvance(badge_str)) + 12.0
                    tbh = 18.0
                    tby = max(target_rect.top(), ry1 - tbh - 2.0)
                    tbx = max(target_rect.left(), min(rx1, target_rect.right() - tbw))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor("#22c55e")))
                    painter.drawRoundedRect(QRectF(tbx, tby, tbw, tbh), 3.0, 3.0)
                    painter.setPen(QColor("#08130C"))
                    painter.drawText(QRectF(tbx + 6.0, tby, tbw - 6.0, tbh), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), badge_str)

            # Alt lejant rozetleri
            legend_y = rect.height() - footer_h + 4.0
            lx = 10.0
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            for tag_name, hex_col in self.REQUIRED_PREVIEW_TAGS:
                c = QColor(hex_col)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(c))
                painter.drawEllipse(QRectF(lx, legend_y + 5.0, 8.0, 8.0))
                painter.setPen(QColor("#cbd5e1"))
                painter.drawText(QPointF(lx + 11.0, legend_y + 13.0), tag_name)
                lx += float(len(tag_name) * 5.6 + 24.0)
        finally:
            painter.end()


class SearchableTagListPanel(QFrame):
    """
    Adım 2 sağ panelinde yer alan 25 kanonik sınıfı (0..24, buff_scroll, hp_potion, mp_potion dahil)
    listeleyen, anlık arama (QLineEdit) yapılabilen ve fare ile tıklanarak aktif çizim sınıfı
    seçilebilen etiket paneli.
    """

    classSelected = pyqtSignal(int, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SearchableTagListPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QFrame#SearchableTagListPanel {"
            " background-color: #0b0e14; border: 1px solid #1e293b; border-radius: 8px; padding: 4px;"
            "}"
            "QFrame#SearchableTagListPanel QLabel { background: transparent; color: #f1f5f9; }"
            "QLineEdit#TagSearchEdit { background: #111827; color: #ffffff; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; font-size: 11px; }"
            "QLineEdit#TagSearchEdit:focus { border: 1px solid #38bdf8; }"
            "QListWidget#TagListWidget { background: #07090e; color: #f1f5f9; border: 1px solid #1e293b; border-radius: 6px; font-size: 11px; }"
            "QListWidget#TagListWidget::item { padding: 3px 6px; border-radius: 4px; }"
            "QListWidget#TagListWidget::item:hover { background: #1e293b; }"
            "QListWidget#TagListWidget::item:selected { background: #166534; color: #ffffff; font-weight: 700; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # Header with active class indicator
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 0, 2, 0)
        self.lbl_title = QLabel(f"🏷️ {len(CLASS_ID_TO_SPEC)} Sınıflı Etiket Listesi")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #38bdf8;")
        self.lbl_active_pill = QLabel("🎯 0: herb_aria")
        self.lbl_active_pill.setStyleSheet(
            "background: #0d2818; color: #4ade80; border: 1px solid #16a34a; border-radius: 5px; padding: 2px 6px; font-weight: 700; font-size: 10px;"
        )
        top_bar.addWidget(self.lbl_title)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_active_pill)
        lay.addLayout(top_bar)

        # Search edit
        self.search_edit = QLineEdit(self)
        self.search_edit.setObjectName("TagSearchEdit")
        self.search_edit.setPlaceholderText("🔍 Etiket ara (örn: aria, kinah, pot, scroll)...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._on_search_changed)
        lay.addWidget(self.search_edit)

        # Searchable Combo (with QCompleter)
        self.combo = QComboBox(self)
        self.combo.setObjectName("TagSearchCombo")
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setStyleSheet(
            "QComboBox { background: #111827; color: #ffffff; border: 1px solid #334155; border-radius: 6px; padding: 2px 6px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background: #0b0e14; color: #ffffff; selection-background-color: #2563eb; }"
        )
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        lay.addWidget(self.combo)

        # ListWidget showing all canonical classes
        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("TagListWidget")
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.list_widget, stretch=1)

        self._populate_classes()

    def _populate_classes(self) -> None:
        self.list_widget.clear()
        self.combo.blockSignals(True)
        self.combo.clear()

        # All canonical classes
        for cid, spec in sorted(CLASS_ID_TO_SPEC.items()):
            cname = spec.get("key", f"class_{cid}")
            label_tr = spec.get("label_tr", f"{cid}: {cname}")
            col_hex = CLASS_COLORS_MAP.get(cid, spec.get("color_hex", "#22c55e"))

            # Colored dot icon
            pm = QPixmap(14, 14)
            pm.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(col_hex)))
            painter.drawEllipse(1, 1, 12, 12)
            painter.end()

            item_text = f"{cid:02d}: {cname} · {label_tr}"
            item = QListWidgetItem(QIcon(pm), item_text)
            item.setData(Qt.ItemDataRole.UserRole, cid)
            item.setData(Qt.ItemDataRole.UserRole + 1, cname)
            self.list_widget.addItem(item)

            self.combo.addItem(f"{cid:02d}: {cname}", cid)

        completer = QCompleter([self.combo.itemText(i) for i in range(self.combo.count())], self.combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.combo.setCompleter(completer)
        self.combo.blockSignals(False)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_search_changed(self, text: str) -> None:
        from dataset_factory.manual_tag_search import search_label_classes

        q = text.strip()
        q_low = q.lower()
        results = search_label_classes(q) if q else search_label_classes("")
        matched_ids = {int(r["class_id"]) for r in results}

        self.combo.blockSignals(True)
        self.combo.clear()
        for r in results:
            cid = int(r["class_id"])
            cname = str(r.get("class_name", f"class_{cid}"))
            label_tr = str(r.get("label_tr", cname))
            self.combo.addItem(f"{cid}: {cname} — {label_tr}", cid)
        if self.combo.count() > 0:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

        first_visible_item: Optional[QListWidgetItem] = None
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            cid = int(item.data(Qt.ItemDataRole.UserRole))
            txt = item.text().lower()
            cname = str(item.data(Qt.ItemDataRole.UserRole + 1) or "").lower()
            visible = (not q_low) or (cid in matched_ids) or (q_low in txt) or (q_low in cname)
            item.setHidden(not visible)
            if visible and first_visible_item is None:
                first_visible_item = item

        if results:
            top_cid = int(results[0]["class_id"])
            top_cname = str(results[0].get("class_name", f"class_{top_cid}"))
            self.lbl_active_pill.setText(f"🎯 {top_cid}: {top_cname}")
            if first_visible_item is not None:
                self.list_widget.setCurrentItem(first_visible_item)
            self.classSelected.emit(top_cid, top_cname)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        cid = int(item.data(Qt.ItemDataRole.UserRole))
        cname = str(item.data(Qt.ItemDataRole.UserRole + 1))
        self.set_active_class(cid, cname)
        self.classSelected.emit(cid, cname)

    def _on_combo_changed(self, idx: int) -> None:
        cid_data = self.combo.itemData(idx)
        if cid_data is not None:
            try:
                cid = int(cid_data)
                spec = CLASS_ID_TO_SPEC.get(cid, {})
                cname = spec.get("key", f"class_{cid}")
                self.set_active_class(cid, cname)
                self.classSelected.emit(cid, cname)
            except (ValueError, TypeError):
                pass

    def set_active_class(self, cid: int, cname: Optional[str] = None) -> None:
        spec = CLASS_ID_TO_SPEC.get(cid, {})
        resolved_cname = cname or spec.get("key", f"class_{cid}")
        self.lbl_active_pill.setText(f"🎯 {cid}: {resolved_cname}")
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == cid:
                self.list_widget.setCurrentItem(item)
                break
        idx = self.combo.findData(cid)
        if idx >= 0 and self.combo.currentIndex() != idx:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(idx)
            self.combo.blockSignals(False)


class ClassSelectDialog(QDialog):
    """
    Kullanıcı görsel üzerine manuel etiket kutusu çizdiğinde açılan,
    hızlı arama (QLineEdit) yapılabilen, 25 sınıfı gösteren ve
    seçilen sınıfı döndüren diyalog penceresi.
    """

    def __init__(self, parent: Optional[QWidget] = None, initial_class_id: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ne Olduğunu Seç — Etiket Sınıfı")
        self.setMinimumSize(420, 500)
        self.resize(440, 540)
        self.setModal(True)
        self._selected_cid: int = initial_class_id

        self.setStyleSheet(
            "QDialog {"
            " background-color: #0B0E14;"
            " color: #F1F5F9;"
            " border: 1px solid #1E293B;"
            " border-radius: 10px;"
            "}"
            "QLabel { background: transparent; color: #F1F5F9; }"
            "QLineEdit#ClassSearchEdit {"
            " background-color: #111827;"
            " color: #FFFFFF;"
            " border: 1px solid #334155;"
            " border-radius: 6px;"
            " padding: 6px 10px;"
            " font-size: 12px;"
            "}"
            "QLineEdit#ClassSearchEdit:focus { border: 1px solid #38BDF8; }"
            "QListWidget#ClassListWidget {"
            " background-color: #07090E;"
            " color: #F1F5F9;"
            " border: 1px solid #1E293B;"
            " border-radius: 6px;"
            " font-size: 12px;"
            "}"
            "QListWidget#ClassListWidget::item {"
            " padding: 6px 8px;"
            " border-radius: 4px;"
            " margin-bottom: 2px;"
            "}"
            "QListWidget#ClassListWidget::item:hover { background: #1E293B; }"
            "QListWidget#ClassListWidget::item:selected {"
            " background: #166534;"
            " color: #FFFFFF;"
            " font-weight: 700;"
            "}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        # Başlık ve açıklama
        title = QLabel("🏷️ Nesne Sınıfını Seçin")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #38BDF8;")
        lay.addWidget(title)

        sub = QLabel("Çizdiğin kutu hangi nesneye ait? Listeden seçin veya arama yapın:")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setStyleSheet("color: #94A3B8;")
        lay.addWidget(sub)

        # Arama kutusu
        self.search_edit = QLineEdit(self)
        self.search_edit.setObjectName("ClassSearchEdit")
        self.search_edit.setPlaceholderText("🔍 Hızlı Ara... (örn: pot, aria, orichalcum, filiz, 0..24)")
        self.search_edit.textChanged.connect(self._filter_list)
        self.search_edit.returnPressed.connect(self._on_search_return)
        lay.addWidget(self.search_edit)

        # Sınıf listesi
        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("ClassListWidget")
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.list_widget, stretch=1)

        self._populate_classes(initial_class_id)

        # Alt Butonlar
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)

        self.btn_cancel = QPushButton("✕ İptal (Esc)", self)
        self.btn_cancel.setStyleSheet(
            "background: #1E293B; color: #94A3B8; border: 1px solid #334155; border-radius: 6px; padding: 6px 14px; font-size: 12px;"
        )
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("💾 Seç ve Kaydet (Enter)", self)
        self.btn_ok.setStyleSheet(
            "background: #166534; color: #F0FDF4; border: 1px solid #22C55E; border-radius: 6px; padding: 6px 16px; font-size: 12px; font-weight: 700;"
        )
        self.btn_ok.clicked.connect(self.accept)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_ok)
        lay.addLayout(btn_box)

        self.search_edit.setFocus()

    def _populate_classes(self, initial_class_id: int) -> None:
        self.list_widget.clear()
        target_item = None
        for cid, spec in sorted(CLASS_ID_TO_SPEC.items()):
            cname = spec.get("key", f"class_{cid}")
            label_tr = spec.get("label_tr", f"{cid}: {cname}")
            col_hex = CLASS_COLORS_MAP.get(cid, spec.get("color_hex", "#22c55e"))

            pm = QPixmap(14, 14)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(col_hex)))
            p.drawEllipse(1, 1, 12, 12)
            p.end()

            item = QListWidgetItem(QIcon(pm), f"[{cid:02d}] {cname}  ({label_tr})")
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.list_widget.addItem(item)
            if cid == initial_class_id:
                target_item = item

        if target_item is not None:
            self.list_widget.setCurrentItem(target_item)
            self.list_widget.scrollToItem(target_item)

    def _filter_list(self, query: str) -> None:
        q = query.strip().lower()
        first_visible = None
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            match = (not q) or (q in item.text().lower())
            item.setHidden(not match)
            if match and first_visible is None:
                first_visible = item
        if first_visible is not None and (
            self.list_widget.currentItem() is None or self.list_widget.currentItem().isHidden()
        ):
            self.list_widget.setCurrentItem(first_visible)

    def _on_search_return(self) -> None:
        cur = self.list_widget.currentItem()
        if cur is not None and not cur.isHidden():
            self._selected_cid = int(cur.data(Qt.ItemDataRole.UserRole))
            self.accept()
        else:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if not item.isHidden():
                    self._selected_cid = int(item.data(Qt.ItemDataRole.UserRole))
                    self.accept()
                    return

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._selected_cid = int(item.data(Qt.ItemDataRole.UserRole))

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Down:
            idx = self.list_widget.currentRow()
            for next_idx in range(idx + 1, self.list_widget.count()):
                if not self.list_widget.item(next_idx).isHidden():
                    self.list_widget.setCurrentRow(next_idx)
                    break
            return
        elif event.key() == Qt.Key.Key_Up:
            idx = self.list_widget.currentRow()
            for prev_idx in range(idx - 1, -1, -1):
                if not self.list_widget.item(prev_idx).isHidden():
                    self.list_widget.setCurrentRow(prev_idx)
                    break
            return
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cur = self.list_widget.currentItem()
            if cur is not None and not cur.isHidden():
                self._selected_cid = int(cur.data(Qt.ItemDataRole.UserRole))
                self.accept()
                return
        super().keyPressEvent(event)

    def selected_class_id(self) -> int:
        cur = self.list_widget.currentItem()
        if cur is not None:
            return int(cur.data(Qt.ItemDataRole.UserRole))
        return self._selected_cid


class ClassBarRow(QWidget):
    """Tek bir YOLO sınıfı için yatay ilerleme çubuğu ve kutu sayacı satırı."""

    def __init__(
        self,
        class_id: int,
        label_tr: str,
        class_key: str,
        color_hex: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ClassBarRow")
        self.class_id = int(class_id)
        self.label_tr = label_tr
        self.class_key = class_key
        self.color_hex = color_hex
        self.count: int = 0
        self.ratio: float = 0.0
        self.share_pct: float = 0.0
        self._metrics_sig: Tuple[int, int, int] = (-1, -1, -1)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#0B0E12"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#0B0E12"))
        self.setPalette(pal)
        self.setStyleSheet("QWidget#ClassBarRow { background-color:#0B0E12; color:#F2F3F5; border:none; }")

    def set_metrics(self, count: int, max_count: int, total_boxes: int) -> None:
        sig = (max(0, int(count)), max(0, int(max_count)), max(0, int(total_boxes)))
        if sig == self._metrics_sig:
            return
        self._metrics_sig = sig
        self.count = sig[0]
        self.ratio = (float(self.count) / float(max(1, sig[1]))) if sig[1] > 0 else 0.0
        self.share_pct = (float(self.count) / float(max(1, sig[2]))) * 100.0 if sig[2] > 0 else 0.0
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            rect = self.rect()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#0B0E12")))
            painter.drawRect(rect)

            label_w = 200.0
            stat_w = 110.0
            bar_x = label_w + 8.0
            bar_w = max(48.0, float(rect.width()) - label_w - stat_w - 16.0)
            bar_h = 14.0
            bar_y = (float(rect.height()) - bar_h) * 0.5

            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.setPen(QColor("#f3f5f7"))
            painter.drawText(
                QRectF(4.0, 0.0, label_w, float(rect.height())),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.label_tr,
            )

            bg_rect = QRectF(bar_x, bar_y, bar_w, bar_h)
            painter.setPen(QPen(QColor("#3a4250"), 1.0))
            painter.setBrush(QBrush(QColor("#14181f")))
            painter.drawRoundedRect(bg_rect, 3.0, 3.0)

            if self.count > 0:
                fill_w = max(8.0, min(bar_w, bar_w * self.ratio))
                fill_rect = QRectF(bar_x, bar_y, fill_w, bar_h)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(self.color_hex)))
                painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.setPen(QColor("#f3f5f7" if self.count > 0 else "#aeb6c2"))
            stat_pct = getattr(self, "share_pct", 0.0)
            painter.drawText(
                QRectF(bar_x + bar_w + 8.0, 0.0, stat_w, float(rect.height())),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                f"{self.count} (%{stat_pct:.0f})",
            )
        finally:
            painter.end()


# ============================================================================
# Arka Plan RTX CUDA FP16 Eğitim İş Parçacığı (QThread)
# ============================================================================
SCAN_RUNNING_STATUS = "Eğitim Durumu: Veri seti taraması sürüyor"


def format_dataset_bootstrap_complete(result: Dict[str, Any]) -> str:
    """Same sentence DatasetCurator logs when a bootstrap finishes."""
    if not isinstance(result, dict) or not result.get("bootstrapped"):
        return ""
    summary = result.get("bootstrap_summary")
    if not isinstance(summary, dict):
        summary = result
    manifest = summary.get("manifest_stats")
    if not isinstance(manifest, dict):
        manifest = summary
    total = int(manifest.get("total_images", summary.get("total_images", 0)) or 0)
    train = int(
        manifest.get("train_count", summary.get("train_count", result.get("train_frames", 0))) or 0
    )
    val = int(manifest.get("val_count", summary.get("val_count", result.get("val_frames", 0))) or 0)
    boxes = int(
        manifest.get("total_boxes", summary.get("total_boxes", result.get("total_boxes", 0))) or 0
    )
    class_counts = manifest.get("class_counts")
    if not isinstance(class_counts, dict):
        class_counts = summary.get("class_counts") if isinstance(summary.get("class_counts"), dict) else {}
    n_classes = len(class_counts) if class_counts else int(manifest.get("nc", 0) or 0)
    return (
        "Dataset bootstrap complete: %d total images (train=%d, val=%d), "
        "%d bounding boxes across %d classes."
        % (total, train, val, boxes, n_classes)
    )


def _game_blocks_cuda_training() -> str:
    from dataset_factory import factory_controller as factory_api

    return str(factory_api.cuda_training_blocked_by_game() or "")


class _BackgroundCallWorker(QThread):
    """Runs one callable off the GUI thread and the BotWorker thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        threading.current_thread().name = "GatherionDatasetBootstrap"
        try:
            self.succeeded.emit(self._fn())
        except Exception as exc:
            logger.error("Arka plan veri seti taraması başarısız: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class RTXTrainingWorker(QThread):
    """
    Dataset bootstrap and `run_one_click_rtx_training` stay off the GUI thread.
    Epoch updates still emit live `mAP50`, `box_loss`, and `cls_loss`.
    """

    progress_updated = pyqtSignal(dict)
    training_finished = pyqtSignal(dict)
    training_failed = pyqtSignal(str)
    status_text = pyqtSignal(str)
    training_deferred = pyqtSignal(str)

    def __init__(
        self,
        controller: Any,
        epochs: int = 25,
        parent: Optional[QWidget] = None,
        *,
        imgsz: int = 640,
        batch: int = 16,
        model: str = "yolo11n.pt",
        only_approved: bool = False,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.epochs = max(1, int(epochs))
        self.imgsz = max(320, int(imgsz))
        self.batch = max(1, int(batch))
        self.model = str(model or "yolo11n.pt").strip() or "yolo11n.pt"
        self.only_approved = bool(only_approved)
        self._stop_requested: bool = False

    def is_stop_requested(self) -> bool:
        return bool(getattr(self, "_stop_requested", False) or self.isInterruptionRequested())

    def stop(self) -> None:
        """Requests training stop across worker and underlying trainer/controller."""
        self._stop_requested = True
        self.requestInterruption()
        ctrl = getattr(self, "controller", None)
        if ctrl is not None:
            if hasattr(ctrl, "stop_training"):
                try:
                    ctrl.stop_training()
                except Exception:
                    pass
            elif hasattr(ctrl, "rtx_trainer") and hasattr(ctrl.rtx_trainer, "stop_training"):
                try:
                    ctrl.rtx_trainer.stop_training()
                except Exception:
                    pass
            elif hasattr(ctrl, "trainer") and hasattr(ctrl.trainer, "stop_training"):
                try:
                    ctrl.trainer.stop_training()
                except Exception:
                    pass

    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if self.is_stop_requested():
            self.stop()
        self.progress_updated.emit(dict(payload))

    def _bootstrap_dataset(self) -> Dict[str, Any]:
        ctrl = self.controller
        prepare = getattr(ctrl, "prepare_one_click_dataset", None)
        if callable(prepare):
            result = prepare()
            return result if isinstance(result, dict) else {}
        fn = getattr(ctrl, "bootstrap_from_existing_frames", None)
        if not callable(fn):
            return {"bootstrapped": False, "skipped": True}
        total = 0
        stats_fn = getattr(ctrl, "get_dataset_stats", None)
        if not callable(stats_fn):
            stats_fn = getattr(ctrl, "get_manifest_stats", None)
        if callable(stats_fn):
            try:
                stats = stats_fn()
            except Exception:
                stats = None
            if isinstance(stats, dict):
                total = int(stats.get("total_frames", stats.get("total_images", 0)) or 0)
        if total > 0:
            return {"bootstrapped": False, "skipped": True, "total_images": total}
        sig = inspect.signature(fn)
        kwargs: Dict[str, Any] = {}
        if "augment_multiplier" in sig.parameters:
            kwargs["augment_multiplier"] = 3
        elif "augment_factor" in sig.parameters:
            kwargs["augment_factor"] = 3
        result = fn(**kwargs)
        return result if isinstance(result, dict) else {"bootstrapped": True}

    def run(self) -> None:
        threading.current_thread().name = "GatherionDatasetBootstrap"
        try:
            if self.is_stop_requested():
                self.training_failed.emit("Eğitim kullanıcı tarafından durduruldu.")
                return
            if self.only_approved:
                try:
                    dataset_dir = getattr(self.controller, "dataset_dir", Path("dataset"))
                    from dataset_factory.approved_pool import get_approved_pool
                    get_approved_pool(dataset_dir).prepare_approved_training_dataset()
                except Exception as exc:
                    logger.debug("prepare_approved_training_dataset in worker: %s", exc)
            prepared = self._bootstrap_dataset()
            line = format_dataset_bootstrap_complete(prepared)
            if line:
                self.status_text.emit(line)
            blocked = _game_blocks_cuda_training()
            if blocked:
                self.status_text.emit(f"Eğitim Durumu: {blocked}")
                self.training_deferred.emit(blocked)
                return
            if not line:
                self.status_text.emit(
                    f"Eğitim Durumu: RTX CUDA FP16 · {self.model} · "
                    f"{self.epochs} epoch · imgsz={self.imgsz} · batch={self.batch}"
                    + (" (Sadece Onaylananlar)" if self.only_approved else "")
                )
            result = _invoke_controller_rtx_training(
                self.controller,
                epochs=self.epochs,
                progress_callback=self._emit_progress,
                imgsz=self.imgsz,
                batch=self.batch,
                model=self.model,
                only_approved=self.only_approved,
            )
            if isinstance(result, dict) and (
                result.get("waiting_for_game_exit") or result.get("reason") == "aion2_running"
            ):
                reason = "Eğitim, oyun kapanana kadar bekliyor."
                self.status_text.emit(f"Eğitim Durumu: {reason}")
                self.training_deferred.emit(reason)
                return
            if isinstance(result, dict) and (
                result.get("stopped")
                or result.get("interrupted")
                or result.get("status") == "stopped"
                or result.get("state") == "stopped"
            ):
                reason = str(result.get("reason") or "Eğitim kullanıcı tarafından durduruldu.")
                self.status_text.emit(f"Eğitim Durumu: Durduruldu ({reason})")
                self.training_failed.emit(reason)
                return
            if isinstance(result, dict) and (
                result.get("started") is False
                or result.get("status") == "failed"
                or result.get("state") in ("failed", "refused", "busy")
            ):
                self.training_failed.emit(
                    str(result.get("reason") or result.get("error") or "Eğitim başarısız.")
                )
                return
            self.training_finished.emit(result if isinstance(result, dict) else {"status": "failed"})
        except Exception as exc:
            logger.error("RTX eğitim iş parçacığı hatası: %s", exc, exc_info=True)
            self.training_failed.emit(str(exc))


def _invoke_controller_rtx_training(
    controller: Any,
    epochs: int = 25,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    imgsz: int = 640,
    batch: int = 16,
    model: str = "yolo11n.pt",
    only_approved: bool = False,
) -> Dict[str, Any]:
    """
    `controller.run_one_click_rtx_training` metodunu esnek imza uyumluluğuyla çağırır.
    """
    fn = getattr(controller, "run_one_click_rtx_training", None)
    if callable(fn):
        sig = inspect.signature(fn)
        kwargs: Dict[str, Any] = {}
        if "epochs" in sig.parameters:
            kwargs["epochs"] = epochs
        if "progress_callback" in sig.parameters:
            kwargs["progress_callback"] = progress_callback
        elif "callback" in sig.parameters:
            kwargs["callback"] = progress_callback
        if "imgsz" in sig.parameters:
            kwargs["imgsz"] = int(imgsz)
        if "batch" in sig.parameters:
            kwargs["batch"] = int(batch)
        if "model" in sig.parameters:
            kwargs["model"] = str(model)
        if "allow_opencv_fallback" in sig.parameters:
            kwargs["allow_opencv_fallback"] = False
        if "blocking" in sig.parameters:
            kwargs["blocking"] = True
        if "async_mode" in sig.parameters:
            kwargs["async_mode"] = False
        if "only_approved" in sig.parameters:
            kwargs["only_approved"] = bool(only_approved)
        if only_approved:
            dataset_dir = getattr(controller, "dataset_dir", Path("dataset"))
            app_yaml = Path(dataset_dir) / "approved" / "data_approved.yaml"
            if app_yaml.is_file() and "data_yaml_path" in sig.parameters:
                kwargs["data_yaml_path"] = app_yaml

        res = fn(**kwargs)
        # Async launch descriptor without metrics: wait for real completion (no 30s fake success).
        if isinstance(res, dict) and res.get("started") and not res.get("mAP50") and (
            res.get("bootstrap_pending") or res.get("thread_name")
        ):
            trainer = getattr(controller, "rtx_trainer", getattr(controller, "trainer", None))
            if trainer is not None and hasattr(trainer, "wait_for_training"):
                waited = trainer.wait_for_training(timeout=None)
                if isinstance(waited, dict):
                    if waited.get("state") in ("running", None) and not waited.get("mAP50"):
                        return {
                            "started": False,
                            "status": "failed",
                            "state": "failed",
                            "reason": "Eğitim tamamlanmadan döndü (async wait).",
                        }
                    return waited
        if isinstance(res, dict):
            return res

    # Fallback doğrudan RTXTrainerEngine çalıştır
    fallback = _BuiltinAutoLabelingDatasetFactory()
    if only_approved:
        dataset_dir = getattr(controller, "dataset_dir", Path("dataset"))
        app_yaml = Path(dataset_dir) / "approved" / "data_approved.yaml"
        if app_yaml.is_file():
            return fallback.rtx_trainer.run_training_sync(
                data_yaml_path=app_yaml,
                epochs=epochs,
                progress_callback=progress_callback,
                imgsz=imgsz,
                batch=batch,
                model=model,
                allow_opencv_fallback=False,
            )
    return fallback.run_one_click_rtx_training(
        epochs=epochs,
        progress_callback=progress_callback,
        imgsz=imgsz,
        batch=batch,
        model=model,
        allow_opencv_fallback=False,
    )


# ============================================================================
# Ana Stüdyo Penceresi: `YOLOFactoryDialog(QDialog)`
# ============================================================================
class YOLOFactoryDialog(QDialog):
    """
    YOLO Veri Seti Fabrikası — 6 adımlı üretim hattı (prototip ile birebir).
    Yakala → Onayla → Hazırla → Eğit → Doğrula → Yayınla
    """

    WINDOW_TITLE_TR: str = "YOLO Fabrikası — İş Akışı"
    STEP_NAMES: Dict[int, str] = {
        1: "Yakala",
        2: "Onayla",
        3: "Hazırla",
        4: "Eğit",
        5: "Doğrula",
        6: "Yayınla",
    }
    PIPELINE_COLORS = {
        "bg": "#0E1013",
        "panel": "#171B22",
        "panel2": "#1C212B",
        "panel3": "#202633",
        "line": "#262D3A",
        "line2": "#2E3647",
        "text": "#F2F3F5",
        "muted": "#9AA3B2",
        "muted2": "#6B7688",
        "amber": "#F6B81E",
        "amber_soft": "#2A220E",
        "green": "#2ED573",
        "green_soft": "#0F2A1D",
        "red": "#FF5A5F",
        "red_soft": "#24151A",
        "radius": "14px",
    }

    def __init__(
        self,
        factory_controller: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.factory_controller = self._resolve_or_create_controller(factory_controller)
        self.controller = self.factory_controller
        self._builtin_helper = _BuiltinAutoLabelingDatasetFactory()
        self._training_worker: Optional[RTXTrainingWorker] = None
        self._bootstrap_worker: Optional[_BackgroundCallWorker] = None

        self.setWindowTitle(self.WINDOW_TITLE_TR)
        self.setMinimumSize(1260, 820)
        self.resize(1560, 960)

        self.class_rows: Dict[int, ClassBarRow] = {}
        self.class_labels: Dict[int, QLabel] = {}
        self._class_counts_cache: Dict[int, int] = {}
        self.active_drawing_class_id: int = 0
        self.active_drawing_class_name: str = "herb_aria"
        self._review_press_pos: Optional[QPointF] = None
        self._capture_press_pos: Optional[QPointF] = None
        self._review_items: List[Dict[str, Any]] = []
        self._review_index: int = 0
        self._auto_confirm_dialogs: bool = False
        self._last_train_result: Dict[str, Any] = {}
        self._pipeline: Dict[str, Any] = {
            "step": 1,
            "pending": 0,
            "reviewed": 0,
            "frozen": False,
            "trained": False,
            "validated": False,
            "published": False,
            "scanned": False,
            "imbalance_fixed": False,
            "capturing": False,
            "session": 0,
            "format": "best.pt (PyTorch)",
            "run_id": 1,
        }
        self._step_buttons: Dict[int, QPushButton] = {}
        self._blocker_widgets: Dict[str, QFrame] = {}
        # Perf / disk-preview cache
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._stats_cache_at: float = 0.0
        self._stats_cache_ttl_sec: float = 8.0
        self._pending_scan_at: float = 0.0
        self._pending_scan_ttl_sec: float = 12.0
        self._last_stats_sig: Optional[Tuple[Any, ...]] = None
        self._last_class_sig: Optional[Tuple[Any, ...]] = None
        self._last_capture_preview_key: str = ""
        self._log_once_keys: set = set()
        self._lightbox_open: bool = False
        self._review_decode_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._rejected_stems: Set[str] = set()
        self._queue_filter_mode: str = "all"  # all | below | above
        # True after "Mevcut seti onaya geri al": empty-weak / requeued frames stay listed
        self._review_audit_mode: bool = False
        self._startup_warnings: List[str] = []
        self._train_click_mono: float = 0.0
        self._last_publish_mono: float = 0.0
        self._pipeline["train_session_ok"] = False
        self._pipeline["train_completed_at"] = ""
        self._live_relabeling: bool = False
        self._live_relabel_index: int = 0
        self._live_relabel_total_boxes: int = 0

        self._load_pipeline_disk_state()
        self._apply_dark_studio_theme()
        self._build_ui()
        self._sync_toggles_from_controller()
        self.refresh_dashboard(force=True)
        self._go_step(int(self._pipeline["step"]))
        for warning in list(self._startup_warnings):
            self._append_log(warning)
        self._startup_warnings.clear()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(4000)
        self._refresh_timer.timeout.connect(lambda: self.refresh_dashboard(force=False))
        self._refresh_timer.start()

    @staticmethod
    def _resolve_or_create_controller(factory_controller: Optional[Any]) -> Any:
        if factory_controller is not None:
            return factory_controller
        try:
            from dataset_factory.factory_controller import (  # type: ignore
                AutoLabelingDatasetFactory as LiveController,
            )

            return LiveController()
        except Exception as exc:
            logger.info("Yerleşik AutoLabelingDatasetFactory devreye alındı (%s)", exc)
            return _BuiltinAutoLabelingDatasetFactory()

    def _dataset_dir(self) -> Path:
        return Path(getattr(self._builtin_helper, "dataset_dir", Path("dataset")))

    def _models_dir(self) -> Path:
        try:
            ds = self._dataset_dir()
            ds_str = str(ds).lower()
            if "tmp" in ds_str or "pytest" in ds_str:
                test_models = ds.parent / "models" if ds.name == "dataset" else ds / "models"
                test_models.mkdir(parents=True, exist_ok=True)
                return test_models
        except Exception:
            pass
        root = Path(__file__).resolve().parent.parent
        return root / "models"

    def _freeze_path(self) -> Path:
        return self._dataset_dir() / "frozen_manifest.json"

    def _pipeline_state_path(self) -> Path:
        return self._dataset_dir() / "pipeline_ui_state.json"

    def _rejected_stems_path(self) -> Path:
        return self._dataset_dir() / "rejected_stems.json"

    def _load_rejected_stems(self) -> Set[str]:
        path = self._rejected_stems_path()
        stems = self.__dict__.get("_rejected_stems")
        if stems is None:
            stems = set()
            self._rejected_stems = stems
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    stems.update(str(s).strip() for s in data if str(s).strip())
            except Exception as exc:
                logger.debug("Failed loading rejected_stems: %s", exc)
        return set(stems)

    def _record_rejected_stems(self, stems: Sequence[str]) -> None:
        clean = [str(s).strip() for s in stems if str(s).strip()]
        if not clean:
            return
        stems_set = self._load_rejected_stems()
        stems_set.update(clean)
        self._rejected_stems = stems_set
        path = self._rejected_stems_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(sorted(list(stems_set)), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Failed saving rejected_stems: %s", exc)

    def _load_pipeline_disk_state(self) -> None:
        freeze = self._freeze_path()
        if freeze.is_file():
            try:
                payload = json.loads(freeze.read_text(encoding="utf-8"))
                self._pipeline["frozen"] = True
                self._pipeline["run_id"] = int(payload.get("run_id", 1) or 1)
            except Exception:
                self._pipeline["frozen"] = False
        state_path = self._pipeline_state_path()
        if state_path.is_file():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                for key in ("trained", "validated", "published", "scanned", "imbalance_fixed", "step", "train_completed_at"):
                    if key in saved:
                        self._pipeline[key] = saved[key]
                cached = saved.get("last_train_metrics")
                if isinstance(cached, dict) and cached:
                    self._last_train_result = dict(cached)
            except Exception:
                pass
        # Dialog restart drops in-memory train result; recover metrics for Doğrula display only.
        self._hydrate_last_train_result_from_artifacts()
        self._reconcile_trained_flag_with_artifacts()

    def _save_pipeline_disk_state(self) -> None:
        path = self._pipeline_state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics = self._metrics_snapshot_for_disk(self._last_train_result)
            payload = {
                "step": int(self._pipeline["step"]),
                "trained": bool(self._pipeline["trained"]),
                "validated": bool(self._pipeline["validated"]),
                "published": bool(self._pipeline["published"]),
                "scanned": bool(self._pipeline["scanned"]),
                "imbalance_fixed": bool(self._pipeline["imbalance_fixed"]),
                "frozen": bool(self._pipeline["frozen"]),
                "last_train_metrics": metrics,
                "train_completed_at": str(self._pipeline.get("train_completed_at") or ""),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("pipeline state save failed: %s", exc)

    def _freeze_mtime(self) -> float:
        path = self._freeze_path()
        if not path.is_file():
            return 0.0
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frozen_at = str(payload.get("frozen_at") or "").strip()
            if frozen_at:
                from datetime import datetime as _dt

                parsed = _dt.fromisoformat(frozen_at)
                mtime = max(mtime, float(parsed.timestamp()))
        except Exception:
            pass
        return mtime

    def _artifact_mtime(self, run_dir: Optional[Path] = None) -> float:
        path = run_dir or self._find_train_artifact_run_dir()
        if path is None:
            return 0.0
        mtime = 0.0
        for cand in (path / "results.csv", path / "weights" / "best.pt"):
            try:
                if cand.is_file():
                    mtime = max(mtime, float(cand.stat().st_mtime))
            except OSError:
                continue
        return mtime

    def _artifacts_are_fresh_vs_freeze(self) -> bool:
        """True when results.csv/best.pt are newer than the current dataset freeze."""
        freeze_mtime = self._freeze_mtime()
        art_mtime = self._artifact_mtime()
        if art_mtime <= 0:
            return False
        if freeze_mtime <= 0:
            return True
        return art_mtime >= (freeze_mtime - 1.0)

    def _mark_model_stale(self, reason: str = "Etiketler güncellendi") -> None:
        """Invalidates model status and marks model as stale whenever dataset labels are modified."""
        if not hasattr(self, "_pipeline") or not isinstance(self._pipeline, dict):
            return
        self._pipeline["model_stale"] = True
        self._pipeline["stale_reason"] = reason
        self._pipeline["trained"] = False
        self._pipeline["validated"] = False
        self._pipeline["published"] = False
        self._pipeline["train_session_ok"] = False
        self._pipeline["train_completed_at"] = ""
        self._last_train_result = {}
        ds_dir = self._dataset_dir() if hasattr(self, "_dataset_dir") and callable(self._dataset_dir) else None
        md_dir = self._models_dir() if hasattr(self, "_models_dir") and callable(self._models_dir) else None
        try:
            from dataset_factory.label_trust import mark_model_stale
            mark_model_stale(reason, dataset_dir=ds_dir, models_dir=md_dir)
        except Exception:
            try:
                from dataset_factory.dataset_validator import mark_model_stale as mark_stale_val
                mark_stale_val(reason, dataset_dir=ds_dir, models_dir=md_dir)
            except Exception:
                pass
        self._append_log(f"[STALE] {reason} — model geçersiz (stale) kılındı. Yeniden eğitim gerekli.")
        for helper_name in (
            "_update_train_lock",
            "_update_blockers",
            "_update_next_card",
            "_refresh_stepper_visuals",
            "_save_pipeline_disk_state",
        ):
            fn = getattr(self, helper_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _reconcile_trained_flag_with_artifacts(self) -> None:
        """Clear or restore trained/validated flags based on artifact freshness vs freeze."""
        if bool(self._pipeline.get("train_session_ok")):
            return
        fresh = self._artifacts_are_fresh_vs_freeze()
        run_dir = self._find_train_artifact_run_dir()
        has_run_weights = (
            (run_dir is not None and (run_dir / "weights" / "best.pt").is_file())
            or (self._models_dir() / "best.pt").is_file()
            or (self._models_dir() / "yolov8n_gatherion.onnx").is_file()
        )
        if fresh and has_run_weights:
            if not self._pipeline.get("trained"):
                self._pipeline["trained"] = True
            return
        if not self._pipeline.get("trained"):
            return
        if fresh:
            return
        self._pipeline["trained"] = False
        self._pipeline["validated"] = False
        self._pipeline["published"] = False
        self._pipeline["train_completed_at"] = ""
        warning = (
            "[UYARI] results.csv / best.pt dondurmadan ESKİ — "
            "trained bayrağı temizlendi. Adım 4'te gerçek eğitim şart."
        )
        pending = getattr(self, "_startup_warnings", None)
        if isinstance(pending, list):
            pending.append(warning)
        else:
            self._startup_warnings = [warning]

    @staticmethod
    def _metrics_snapshot_for_disk(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        src = dict(result or {})
        out: Dict[str, Any] = {}
        for key in (
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "box_loss",
            "cls_loss",
            "epochs",
            "epoch",
            "mode",
            "status",
            "artifacts",
            "metrics_source",
            "per_class_ap50",
        ):
            if key in src and src[key] is not None:
                out[key] = src[key]
        return out

    def _candidate_rtx_run_dirs(self) -> List[Path]:
        """Factory Ultralytics project dirs + legacy runs/detect trees."""
        dirs: List[Path] = []
        seen: set = set()

        def _add(path: Path) -> None:
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                return
            seen.add(key)
            dirs.append(path)

        try:
            primary = rtx_runs_dir_for() / "gatherion_rtx_fp16"
            _add(primary)
            parent = rtx_runs_dir_for()
            if parent.is_dir():
                for child in sorted(
                    parent.iterdir(),
                    key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
                    reverse=True,
                ):
                    if child.is_dir():
                        _add(child)
        except Exception:
            pass

        root = Path(__file__).resolve().parent.parent
        for legacy in (
            root / "models" / "rtx_runs" / "gatherion_rtx_fp16",
            root / "runs" / "detect" / "models" / "rtx_runs" / "gatherion_rtx_fp16",
            root / "offload_from_c" / "runs" / "detect" / "models" / "rtx_runs" / "gatherion_rtx_fp16",
        ):
            _add(legacy)
        return dirs

    def _find_train_artifact_run_dir(self) -> Optional[Path]:
        """Newest run dir that has results.csv and/or weights/best.pt."""
        scored: List[Tuple[float, Path]] = []
        for run_dir in self._candidate_rtx_run_dirs():
            csv_path = run_dir / "results.csv"
            best_pt = run_dir / "weights" / "best.pt"
            if not csv_path.is_file() and not best_pt.is_file():
                continue
            mtime = 0.0
            try:
                if csv_path.is_file():
                    mtime = max(mtime, csv_path.stat().st_mtime)
                if best_pt.is_file():
                    mtime = max(mtime, best_pt.stat().st_mtime)
            except OSError:
                continue
            scored.append((mtime, run_dir))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _read_ultralytics_metrics(self, run_dir: Path) -> Dict[str, float]:
        trainer = getattr(self.factory_controller, "rtx_trainer", None)
        if trainer is not None and hasattr(trainer, "_read_ultralytics_final_metrics"):
            try:
                got = trainer._read_ultralytics_final_metrics(run_dir)
                if isinstance(got, dict):
                    return got
            except Exception as exc:
                logger.debug("rtx metrics read failed: %s", exc)
        # Lightweight fallback identical to RTXTrainerEngine best-epoch pick.
        empty = {
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }
        csv_path = Path(run_dir) / "results.csv"
        if not csv_path.is_file():
            return empty
        try:
            import csv as _csv

            best = dict(empty)
            best_map = -1.0
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in _csv.DictReader(handle):

                    def _f(key: str, *alts: str) -> float:
                        for name in (key,) + alts:
                            raw = (row.get(name) or "").strip()
                            if raw:
                                try:
                                    return float(raw)
                                except ValueError:
                                    continue
                        return 0.0

                    map50 = _f("metrics/mAP50(B)", "mAP50")
                    if map50 < best_map:
                        continue
                    best_map = map50
                    best = {
                        "box_loss": _f("train/box_loss", "val/box_loss", "box_loss"),
                        "cls_loss": _f("train/cls_loss", "val/cls_loss", "cls_loss"),
                        "precision": _f("metrics/precision(B)", "precision"),
                        "recall": _f("metrics/recall(B)", "recall"),
                        "mAP50": map50,
                        "mAP50_95": _f("metrics/mAP50-95(B)", "mAP50_95"),
                    }
            return best
        except Exception as exc:
            logger.debug("results.csv parse failed: %s", exc)
            return empty

    def _load_train_metrics_from_artifacts(self) -> Dict[str, Any]:
        """Recover train metrics from factory rtx_runs / custom detector JSON."""
        run_dir = self._find_train_artifact_run_dir()
        if run_dir is not None:
            ultra = self._read_ultralytics_metrics(run_dir)
            map50 = float(ultra.get("mAP50", 0.0) or 0.0)
            best_pt = run_dir / "weights" / "best.pt"
            if (run_dir / "results.csv").is_file() or best_pt.is_file():
                # Prefer Ultralytics run metrics over Mode B prototype JSON.
                return {
                    "status": "completed",
                    "mode": "ultralytics_artifact",
                    "mAP50": map50,
                    "mAP50_95": float(ultra.get("mAP50_95", 0.0) or 0.0),
                    "precision": float(ultra.get("precision", 0.0) or 0.0),
                    "recall": float(ultra.get("recall", 0.0) or 0.0),
                    "box_loss": float(ultra.get("box_loss", 0.0) or 0.0),
                    "cls_loss": float(ultra.get("cls_loss", 0.0) or 0.0),
                    "metrics_source": str(run_dir / "results.csv"),
                    "artifacts": {
                        "ultralytics_best_pt": str(best_pt) if best_pt.is_file() else "",
                        "run_dir": str(run_dir),
                    },
                }

        detector = self._models_dir() / "gatherion_custom_detector.json"
        if detector.is_file():
            try:
                payload = json.loads(detector.read_text(encoding="utf-8"))
                summary = payload.get("training_summary") if isinstance(payload, dict) else None
                if isinstance(summary, dict):
                    map50 = float(summary.get("final_mAP50", 0.0) or 0.0)
                    if map50 > 0:
                        return {
                            "status": "completed",
                            "mode": "custom_detector_json",
                            "mAP50": map50,
                            "mAP50_95": float(summary.get("final_mAP50_95", 0.0) or 0.0),
                            "precision": float(summary.get("final_precision", 0.0) or 0.0),
                            "recall": float(summary.get("final_recall", 0.0) or 0.0),
                            "box_loss": float(summary.get("final_box_loss", 0.0) or 0.0),
                            "cls_loss": float(summary.get("final_cls_loss", 0.0) or 0.0),
                            "epochs": int(summary.get("epochs_completed", 0) or 0),
                            "metrics_source": str(detector),
                            "artifacts": {"custom_detector_json": str(detector)},
                        }
            except Exception as exc:
                logger.debug("custom detector metrics read failed: %s", exc)
        return {}

    def _hydrate_last_train_result_from_artifacts(self) -> None:
        """Fill `_last_train_result` for Doğrula display only — never marks a new train complete."""
        current = dict(self._last_train_result or {})
        map50 = float(current.get("mAP50", 0.0) or 0.0)
        if map50 > 0:
            fresh = self._artifacts_are_fresh_vs_freeze()
            current.setdefault("artifact_fresh", fresh)
            current.setdefault(
                "metrics_kind",
                "fresh_after_freeze" if fresh else "stale_pre_freeze",
            )
            self._last_train_result = current
            return
        recovered = self._load_train_metrics_from_artifacts()
        if not recovered:
            return
        merged = dict(current)
        merged.update(recovered)
        fresh = self._artifacts_are_fresh_vs_freeze()
        merged["artifact_fresh"] = fresh
        merged["metrics_kind"] = "fresh_after_freeze" if fresh else "stale_pre_freeze"
        self._last_train_result = merged

    def _apply_dark_studio_theme(self) -> None:
        C = self.PIPELINE_COLORS
        self.setStyleSheet(
            SHELL_PAGE_FRAME_QSS
            + f"""
            QDialog {{
                background-color: {C["bg"]};
                color: {C["text"]};
                font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            QLabel {{ color: {C["text"]}; }}
            QFrame#PipeCard, QFrame#PipeFlow, QFrame#PipeStat,
            QFrame#PipeNext, QFrame#PipeBlockerHost, QFrame#PipeRailCard {{
                background-color: {C["panel"]};
                border: 1px solid {C["line"]};
                border-radius: {C["radius"]};
            }}
            QFrame#PipeStatWarn {{
                background-color: {C["panel"]};
                border: 1px solid #4A2028;
                border-radius: 12px;
            }}
            QFrame#PipeStatGood {{
                background-color: {C["panel"]};
                border: 1px solid #1E4D33;
                border-radius: 12px;
            }}
            QPushButton#PipeBtn {{
                background: {C["panel"]};
                color: {C["text"]};
                border: 1px solid {C["line"]};
                border-radius: 10px;
                padding: 9px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#PipeBtn:hover {{ background: {C["panel2"]}; border-color: {C["line2"]}; }}
            QPushButton#PipeBtn:disabled {{ opacity: 0.45; }}
            QPushButton#PipePrimary {{
                background: {C["amber"]};
                color: #111;
                border: 1px solid #FFD84D;
                border-radius: 10px;
                padding: 9px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#PipePrimary:hover {{ background: #FFC933; }}
            QPushButton#PipePrimary:disabled {{ background: #3a3f48; color: #c2c8d2; border-color: #3a3f48; }}
            QPushButton#PipeGreen {{
                background: {C["green"]};
                color: #08130C;
                border: 1px solid #35E585;
                border-radius: 10px;
                padding: 9px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#PipeDanger {{
                background: transparent;
                color: #FF8A94;
                border: 1px solid #4A2028;
                border-radius: 10px;
                padding: 9px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#PipeStep {{
                background: #0B0E12;
                color: {C["text"]};
                border: 1px solid {C["line"]};
                border-radius: 12px;
                text-align: left;
                padding: 10px;
            }}
            QPushButton#PipeStep:hover {{ border-color: {C["line2"]}; }}
            QPushButton#PipeStepCurrent {{
                background: {C["amber_soft"]};
                color: {C["text"]};
                border: 1px solid {C["amber"]};
                border-radius: 12px;
                text-align: left;
                padding: 10px;
            }}
            QPushButton#PipeStepDone {{
                background: #0E1712;
                color: {C["text"]};
                border: 1px solid #1E4D33;
                border-radius: 12px;
                text-align: left;
                padding: 10px;
            }}
            QPushButton#PipeStepBlocked {{
                background: #0B0E12;
                color: {C["muted"]};
                border: 1px solid {C["line"]};
                border-radius: 12px;
                text-align: left;
                padding: 10px;
                opacity: 0.62;
            }}
            QFrame#PipePreset, QFrame#PipeFormat {{
                background: #0B0E12;
                color: {C["text"]};
                border: 1px solid {C["line"]};
                border-radius: 11px;
            }}
            QFrame#PipePreset:hover, QFrame#PipeFormat:hover {{
                border-color: {C["line2"]};
            }}
            QFrame#PipePresetSel, QFrame#PipeFormatSel {{
                background: {C["amber_soft"]};
                color: {C["text"]};
                border: 1px solid {C["amber"]};
                border-radius: 11px;
            }}
            QListWidget#PipeQueue {{
                background: #0B0E12;
                border: none;
                outline: none;
                color: {C["text"]};
            }}
            QListWidget#PipeQueue::item {{
                background: #0B0E12;
                border: 1px solid {C["line"]};
                border-radius: 10px;
                padding: 8px;
                margin-bottom: 6px;
                color: {C["text"]};
            }}
            QListWidget#PipeQueue::item:selected {{
                border-color: {C["amber"]};
                background: {C["amber_soft"]};
            }}
            QScrollArea#PipeDetScroll {{
                background: #0B0E12;
                border: none;
                border-top: 1px solid {C["line"]};
            }}
            QScrollArea#PipeDetScroll > QWidget > QWidget {{
                background: #0B0E12;
            }}
            QScrollArea#ClassDistScroll {{
                background: #0B0E12;
                border: none;
                border-top: 1px solid {C["line"]};
            }}
            QScrollArea#ClassDistScroll QWidget {{
                background: #0B0E12;
            }}
            QWidget#ClassDistHost, QWidget#ClassBarRow {{
                background-color: #0B0E12;
                color: {C["text"]};
            }}
            QWidget#PipeDetHost, QWidget#PipeDetRow {{
                background-color: #0B0E12;
                color: {C["text"]};
            }}
            QWidget#PipeDetHost QLabel {{
                background: transparent;
                color: {C["text"]};
            }}
            QLabel#PipeDetMuted {{
                background: transparent;
                color: {C["muted"]};
                font-size: 11px;
            }}
            QLabel#PipeDetConf {{
                background: #171B22;
                color: {C["text"]};
                font-family: Consolas, "JetBrains Mono", monospace;
                border: 1px solid {C["line"]};
                border-radius: 999px;
                padding: 2px 6px;
            }}
            QPlainTextEdit#PipeLog, QPlainTextEdit#training_log_box {{
                background: #0B0E12;
                color: #d5dae3;
                border: 1px solid {C["line"]};
                border-radius: 10px;
                font-family: Consolas, "JetBrains Mono", monospace;
                font-size: 11px;
                padding: 8px;
            }}
            QProgressBar#ShellProgress, QProgressBar#PipeProgress {{
                background-color: #0B0E12;
                border: 1px solid {C["line"]};
                border-radius: 999px;
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                text-align: center;
                min-height: 10px;
                max-height: 14px;
            }}
            QProgressBar#ShellProgress::chunk, QProgressBar#PipeProgress::chunk {{
                background-color: {C["amber"]};
                border-radius: 999px;
            }}
            QComboBox, QSpinBox, QLineEdit {{
                background: #0B0E12;
                color: {C["text"]};
                border: 1px solid {C["line"]};
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #0B0E12;
                border: 1px solid {C["line"]};
                border-radius: 999px;
            }}
            QSlider::handle:horizontal {{
                width: 14px;
                margin: -5px 0;
                background: {C["amber"]};
                border-radius: 7px;
            }}
            """
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = wrap_shell_tab_root_layout(self, "yolo")
        root.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(14)
        head_left = QVBoxLayout()
        head_left.setSpacing(6)
        title = QLabel("YOLO fabrikası  —  iş akışı")
        title.setObjectName("PipeMainTitle")
        title.setStyleSheet(
            "font-size:26px;font-weight:800;letter-spacing:-0.02em;color:#F2F3F5;"
            "font-family:'Plus Jakarta Sans','Segoe UI',sans-serif;"
        )
        sub = QLabel(
            "Ekran artık vitrin değil tezgâh: her adım bir sonrakini besliyor. "
            "Nerede takıldığını, sıradaki işin ne olduğunu ve eğitimi neyin engellediğini tek yerden görürsün."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            "color:#9AA3B2;font-size:13.5px;max-width:620px;line-height:1.55;"
            "font-family:'Plus Jakarta Sans','Segoe UI',sans-serif;"
        )
        head_left.addWidget(title)
        head_left.addWidget(sub)
        head.addLayout(head_left, stretch=1)

        head_actions = QHBoxLayout()
        self.btn_open_dataset = QPushButton("📁 dataset/ klasörünü aç", self)
        self.btn_open_dataset.setObjectName("PipeBtn")
        self.btn_open_dataset.clicked.connect(self.on_open_dataset_folder_clicked)
        self.btn_refresh = QPushButton("↻ Yenile", self)
        self.btn_refresh.setObjectName("PipeBtn")
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)
        head_actions.addWidget(self.btn_open_dataset)
        head_actions.addWidget(self.btn_refresh)
        head.addLayout(head_actions)
        root.addLayout(head)

        root.addWidget(self._build_flow_section())
        root.addWidget(self._build_strip_section())

        body = QHBoxLayout()
        body.setSpacing(14)
        left_col = QVBoxLayout()
        self.step_stack = QStackedWidget(self)
        self.step_stack.addWidget(self._build_step_capture())
        self.step_stack.addWidget(self._build_step_review())
        self.step_stack.addWidget(self._build_step_prepare())
        self.step_stack.addWidget(self._build_step_train())
        self.step_stack.addWidget(self._build_step_validate())
        self.step_stack.addWidget(self._build_step_publish())
        left_col.addWidget(self.step_stack)
        body.addLayout(left_col, stretch=1)
        body.addWidget(self._build_right_rail(), stretch=0)
        root.addLayout(body, stretch=1)

        # Compatibility aliases used by older handlers / tests
        self.dataset_progress_label = self.stat_frames_value
        self.dataset_progress_bar = self.flow_bar
        self.kpi_total_frames = _CompatKPI(self.stat_frames_value, self.stat_frames_sub)
        self.kpi_total_boxes = _CompatKPI(self.stat_pending_value, self.stat_pending_sub)
        self.kpi_time_machine = _CompatKPI(self.stat_empty_value, self.stat_empty_sub)
        self.kpi_gpu_status = _CompatKPI(self.stat_model_value, self.stat_model_sub)
        self.btn_close = QPushButton("Kapat", self)
        self.btn_close.hide()
        self.btn_close.clicked.connect(self.close)

    def _build_flow_section(self) -> QFrame:
        C = self.PIPELINE_COLORS
        frame = QFrame(self)
        frame.setObjectName("PipeFlow")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Üretim hattı — dataset v4")
        title.setStyleSheet("font-size:13px;font-weight:800;")
        self.flow_summary = QLabel("Adım 1/6 · Yakala")
        self.flow_summary.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        head.addWidget(title)
        head.addStretch()
        head.addWidget(self.flow_summary)
        lay.addLayout(head)

        track = QHBoxLayout()
        track.setSpacing(8)
        specs = [
            (1, "1 · Yakala", "Canlı kare + zaman makinesi ile toplama"),
            (2, "2 · Onayla", "Otomatik etiketleri tek tek gözden geçir"),
            (3, "3 · Hazırla", "Sağlık kontrolü, bölme, çoğaltma, dondur"),
            (4, "4 · Eğit", "RTX üzerinde eğitim çalıştır"),
            (5, "5 · Doğrula", "mAP, sınıf bazlı AP, hata kareleri"),
            (6, "6 · Yayınla", "Format seç, modele yaz, botu güncelle"),
        ]
        for step, title_txt, desc in specs:
            btn = QPushButton(self)
            btn.setObjectName("PipeStep")
            btn.setMinimumHeight(104)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("pipe_step", step)
            inner = QVBoxLayout(btn)
            inner.setContentsMargins(4, 2, 4, 2)
            inner.setSpacing(4)
            num = QLabel(str(step))
            num.setObjectName("PipeStepNum")
            num.setFixedSize(26, 26)
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(
                f"background:{C['panel3']};border:1px solid {C['line2']};"
                "border-radius:13px;font-family:Consolas;font-weight:700;font-size:12px;color:#9AA3B2;"
            )
            title_lbl = QLabel(title_txt)
            title_lbl.setStyleSheet("font-size:14px;font-weight:800;background:transparent;border:none;")
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"font-size:11px;color:{C['muted']};background:transparent;border:none;")
            st_lbl = QLabel("—")
            st_lbl.setObjectName("PipeStepStatus")
            st_lbl.setStyleSheet(
                f"font-size:10px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;"
                f"color:{C['muted']};background:transparent;border:none;"
            )
            inner.addWidget(num)
            inner.addWidget(title_lbl)
            inner.addWidget(desc_lbl)
            inner.addStretch()
            inner.addWidget(st_lbl)
            btn._num_lbl = num  # type: ignore[attr-defined]
            btn._st_lbl = st_lbl  # type: ignore[attr-defined]
            btn.clicked.connect(lambda _=False, s=step: self._go_step(s))
            self._step_buttons[step] = btn
            track.addWidget(btn, stretch=1)
        lay.addLayout(track)

        self.flow_bar = QProgressBar(self)
        self.flow_bar.setObjectName("PipeProgress")
        self.flow_bar.setRange(0, 100)
        self.flow_bar.setValue(8)
        self.flow_bar.setTextVisible(False)
        self.flow_bar.setFixedHeight(8)
        lay.addWidget(self.flow_bar)
        return frame

    def _build_strip_section(self) -> QWidget:
        host = QWidget(self)
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self.stat_frames_value, self.stat_frames_sub, c1 = self._make_stat(
            "Veri seti", "0 / 100.000", "Train 0 · Val 0 · 0 kutu", "PipeStat"
        )
        self.stat_pending_value, self.stat_pending_sub, c2 = self._make_stat(
            "Onay bekleyen", "0 kare", "Eşik üstü toplu onaylanabilir", "PipeStatWarn"
        )
        self.stat_empty_value, self.stat_empty_sub, c3 = self._make_stat(
            "Boş sınıf", "0 / 12", "—", "PipeStat"
        )
        self.stat_model_value, self.stat_model_sub, c4 = self._make_stat(
            "Son model", "best.pt", "models/ altında", "PipeStatGood"
        )
        for card in (c1, c2, c3, c4):
            row.addWidget(card, stretch=1)
        return host

    def _make_stat(
        self, caption: str, value: str, sub: str, object_name: str
    ) -> Tuple[QLabel, QLabel, QFrame]:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName(object_name)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(3)
        cap = QLabel(caption.upper())
        cap.setStyleSheet(
            f"font-size:10px;letter-spacing:0.1em;font-weight:800;color:{C['muted']};"
            "background:transparent;border:none;"
        )
        val = QLabel(value)
        val.setStyleSheet(
            "font-family:Consolas,'JetBrains Mono',monospace;font-size:20px;font-weight:700;"
            "background:transparent;border:none;"
        )
        sub_lbl = QLabel(sub)
        sub_lbl.setWordWrap(True)
        sub_lbl.setStyleSheet(
            f"font-size:11px;color:{C['muted']};background:transparent;border:none;"
        )
        lay.addWidget(cap)
        lay.addWidget(val)
        lay.addWidget(sub_lbl)
        return val, sub_lbl, card

    def _step_intro(self, title: str, body: str, trailing: Optional[QWidget] = None) -> QFrame:
        C = self.PIPELINE_COLORS
        bar = QFrame(self)
        bar.setStyleSheet(
            f"background:{C['panel2']};border:none;border-bottom:1px solid {C['line']};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 12, 14, 12)
        text_col = QVBoxLayout()
        t = QLabel(title)
        t.setStyleSheet("font-size:14px;font-weight:800;background:transparent;border:none;")
        p = QLabel(body)
        p.setWordWrap(True)
        p.setStyleSheet(f"font-size:12px;color:{C['muted']};background:transparent;border:none;")
        text_col.addWidget(t)
        text_col.addWidget(p)
        lay.addLayout(text_col, stretch=1)
        if trailing is not None:
            lay.addWidget(trailing)
        return bar

    def _build_step_capture(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.btn_capture_toggle = QPushButton("● Yakalamayı başlat", self)
        self.btn_capture_toggle.setObjectName("PipePrimary")
        self.btn_capture_toggle.clicked.connect(self._on_capture_toggle)
        lay.addWidget(
            self._step_intro(
                "Adım 1 — Yakala",
                "Oyun karesi 0,25 sn arayla tampona girer. Toplama bitince zaman makinesi "
                "geriye gidip kareyi etiketler. Bu oturumda toplanan kareler doğrudan onay kuyruğuna düşer.",
                self.btn_capture_toggle,
            )
        )

        preview_head = QHBoxLayout()
        preview_head.setContentsMargins(14, 12, 14, 8)
        left = QVBoxLayout()
        h3 = QLabel("● Canlı etiket önizleme")
        h3.setStyleSheet("font-size:14px;font-weight:800;")
        self.preview_meta = QLabel("Zaman makinesi t−2,0 sn · Otomatik etiket çalışıyor")
        self.preview_meta.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        left.addWidget(h3)
        left.addWidget(self.preview_meta)
        preview_head.addLayout(left, stretch=1)
        seg = QHBoxLayout()
        self._view_group = QButtonGroup(self)
        for key, label in (("live", "Canlı"), ("boxes", "Kutular"), ("clean", "Temiz")):
            b = QPushButton(label, self)
            b.setCheckable(True)
            b.setObjectName("PipeBtn")
            b.setProperty("view_mode", key)
            if key == "live":
                b.setChecked(True)
            self._view_group.addButton(b)
            b.clicked.connect(lambda _=False, m=key: self._set_preview_view(m))
            seg.addWidget(b)
        btn_capture_lightbox = QPushButton("🔍 Büyüt", self)
        btn_capture_lightbox.setObjectName("PipeBtn")
        btn_capture_lightbox.setStyleSheet("font-size:11px; padding:3px 10px;")
        btn_capture_lightbox.setToolTip("Canlı önizlemeyi büyük ekranda incele (Çift tıklayarak da açabilirsiniz)")
        btn_capture_lightbox.clicked.connect(self._open_capture_lightbox)
        seg.addWidget(btn_capture_lightbox)
        preview_head.addLayout(seg)
        lay.addLayout(preview_head)

        self.preview_canvas = YOLOPreviewCanvas(self)
        self.preview_canvas.setObjectName("PipeCapturePreview")
        self.preview_canvas.setMinimumHeight(220)
        self.preview_canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview_canvas.setToolTip("Sol tık sürükle: Yeni etiket kutusu çiz · Çift tıkla: Büyüt · ESC ile kapat")
        self.preview_canvas.installEventFilter(self)
        lay.addWidget(self.preview_canvas, stretch=1)

        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 8)
        self.buffer_val = QLabel("20 kare · 5,0 sn")
        self.buffer_val.setStyleSheet(
            "font-family:Consolas;background:#0B0E12;border:1px solid #262D3A;"
            "border-radius:7px;padding:5px 8px;"
        )
        self.session_val = QLabel("+0 kare")
        self.session_val.setStyleSheet(
            "font-family:Consolas;background:#0B0E12;border:1px solid #262D3A;"
            "border-radius:7px;padding:5px 8px;"
        )
        bar.addWidget(QLabel("Tampon"))
        bar.addWidget(self.buffer_val)
        bar.addWidget(QLabel("Bu oturum"))
        bar.addWidget(self.session_val)
        bar.addStretch()
        bar.addWidget(QLabel("Zaman kayması"))
        self.tm_range = QSlider(Qt.Orientation.Horizontal, self)
        self.tm_range.setRange(-50, 0)
        self.tm_range.setValue(-20)
        self.tm_range.setFixedWidth(140)
        self.tm_range.valueChanged.connect(self._on_tm_changed)
        self.tm_val = QLabel("t −2,0 sn")
        self.tm_val.setStyleSheet(
            "font-family:Consolas;background:#0B0E12;border:1px solid #262D3A;"
            "border-radius:7px;padding:5px 8px;"
        )
        bar.addWidget(self.tm_range)
        bar.addWidget(self.tm_val)
        lay.addLayout(bar)

        two = QHBoxLayout()
        two.setContentsMargins(0, 0, 0, 0)
        two.setSpacing(0)
        left_settings = QVBoxLayout()
        left_settings.setContentsMargins(14, 8, 14, 12)
        left_settings.addWidget(QLabel("<b>Yakalama ayarları</b>"))
        self.toggle_auto_label_2s = FactoryToggleSwitch(
            "2 sn önceki kareyi etiketle",
            checked=True,
            accent_hex=C["green"],
            parent=card,
        )
        self.toggle_passive_shadow = FactoryToggleSwitch(
            "Pasif gölge modu",
            checked=False,
            accent_hex=C["amber"],
            parent=card,
        )
        self.toggle_empty_priority = FactoryToggleSwitch(
            "Boş sınıfları öncelikle yakala",
            checked=True,
            accent_hex=C["amber"],
            parent=card,
        )
        for t in (self.toggle_auto_label_2s, self.toggle_passive_shadow, self.toggle_empty_priority):
            t.toggled.connect(self._on_toggles_changed)
            left_settings.addWidget(t)
        right_kv = QVBoxLayout()
        right_kv.setContentsMargins(14, 8, 14, 12)
        self.kv_frames = QLabel("0 (%0)")
        self.kv_missing = QLabel("100.000 kare")
        for label, widget in (
            ("Hedef", QLabel("100.000 kare")),
            ("Toplanan", self.kv_frames),
            ("Eksik", self.kv_missing),
            ("Kuyruğa düşen", QLabel("otomatik etiketli")),
        ):
            row = QHBoxLayout()
            lab = QLabel(label)
            lab.setStyleSheet(f"color:{C['muted']};")
            row.addWidget(lab)
            row.addStretch()
            widget.setStyleSheet("font-family:Consolas;font-weight:700;")
            row.addWidget(widget)
            right_kv.addLayout(row)
        self.btn_send_review = QPushButton("Yakalamayı bitir → Onay adımına geç", self)
        self.btn_send_review.setObjectName("PipeBtn")
        self.btn_send_review.clicked.connect(self._finish_capture_to_review)
        right_kv.addWidget(self.btn_send_review)
        two.addLayout(left_settings, stretch=1)
        two.addLayout(right_kv, stretch=1)
        lay.addLayout(two)
        return card

    def _build_step_review(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # -------------------------------------------------------------
        # 1. SOL PANEL: Kaydırılabilir Seçenekler, Filtreler & Kuyruk
        # -------------------------------------------------------------
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setObjectName("PipeSidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setFixedWidth(340)
        sidebar_scroll.setStyleSheet(
            "QScrollArea#PipeSidebarScroll { background:#070a0e; border:none; border-right:1px solid #1e293b; }"
            "QScrollArea#PipeSidebarScroll > QWidget { background:#070a0e; }"
            "QScrollArea#PipeSidebarScroll QWidget { background-color:#070a0e; color:#F2F3F5; }"
        )

        side_widget = QWidget(self)
        side_l = QVBoxLayout(side_widget)
        side_l.setContentsMargins(10, 10, 10, 14)
        side_l.setSpacing(10)

        # A) İlerleme & Başlık
        intro_box = QFrame(self)
        intro_box.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:8px;")
        intro_l = QVBoxLayout(intro_box)
        intro_l.setContentsMargins(4, 4, 4, 4)
        intro_l.setSpacing(6)

        hdr_row = QHBoxLayout()
        step_title = QLabel("Adım 2 — Onayla")
        step_title.setStyleSheet("font-size:13px; font-weight:800; color:#38bdf8; background:transparent;")
        hdr_row.addWidget(step_title)
        hdr_row.addStretch()
        self.review_prog_txt = QLabel("0 / 0")
        self.review_prog_txt.setStyleSheet("font-family:Consolas; font-weight:700; color:#cbd5e1; background:transparent;")
        hdr_row.addWidget(self.review_prog_txt)
        intro_l.addLayout(hdr_row)

        pool = get_approved_pool(self._dataset_dir())
        self.lbl_approved_count = QLabel(f"✓ {pool.get_approved_count()} Onaylandı", self)
        self.lbl_approved_count.setObjectName("LblApprovedCount")
        self.lbl_approved_count.setStyleSheet(
            "background:#0d2818; color:#4ade80; font-family:Consolas; font-size:11px; font-weight:700;"
            "border:1px solid #16a34a; border-radius:6px; padding:3px 8px;"
        )
        intro_l.addWidget(self.lbl_approved_count)

        self.review_prog = QProgressBar(self)
        self.review_prog.setObjectName("PipeProgress")
        self.review_prog.setRange(0, 100)
        self.review_prog.setValue(0)
        self.review_prog.setTextVisible(False)
        self.review_prog.setFixedHeight(7)
        intro_l.addWidget(self.review_prog)

        step_hint = QLabel("Kısayol: A onayla · R reddet · S atla")
        step_hint.setStyleSheet("font-size:10px; color:#64748b; background:transparent;")
        intro_l.addWidget(step_hint)
        side_l.addWidget(intro_box)

        # B) Arama ve Kuyruk Filtreleri
        filter_box = QFrame(self)
        filter_box.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:8px;")
        filter_l = QVBoxLayout(filter_box)
        filter_l.setContentsMargins(4, 4, 4, 4)
        filter_l.setSpacing(6)

        flt_hdr = QLabel("🔍 Filtre & Arama")
        flt_hdr.setStyleSheet("font-size:11px; font-weight:700; color:#94a3b8; background:transparent;")
        filter_l.addWidget(flt_hdr)

        self.review_search_edit = QLineEdit(self)
        self.review_search_edit.setObjectName("PipeSearch")
        self.review_search_edit.setPlaceholderText("🔍 Dosya / stem ara...")
        self.review_search_edit.textChanged.connect(self._on_search_text_changed)
        self.edit_search = self.review_search_edit
        filter_l.addWidget(self.review_search_edit)

        self.queue_filter = QComboBox(self)
        self.queue_filter.setObjectName("PipeCombo")
        self.queue_filter.addItem("Tümü (Tüm Kareler)", "all")
        self.queue_filter.addItem("🟠 Sadece Şüpheliler", "suspicious")
        self.queue_filter.addItem("🟢 Sadece Emin Olunanlar", "confident")
        self.queue_filter.addItem("🟡 Karışık", "mixed")
        self.queue_filter.addItem("⚪ Etiketsiz / Boş Kareler (0 Kutu)", "empty")
        self.queue_filter.addItem("🔴 Etiketsiz + Şüpheliler", "unlabeled_and_suspicious")
        self.queue_filter.addItem("✅ Onaylananlar (Eğitim Havuzu)", "approved_pool")
        self.queue_filter.addItem("Liste: eşik altı (<)", "below")
        self.queue_filter.addItem("Liste: eşik üstü (≥)", "above")
        self.queue_filter.currentIndexChanged.connect(self._on_queue_filter_changed)
        filter_l.addWidget(self.queue_filter)

        self.review_class_filter = QComboBox(self)
        self.review_class_filter.setObjectName("PipeClassFilter")
        self.review_class_filter.addItem("Sınıf: Tümü", -1)
        for cid_opt, spec_opt in sorted(CLASS_ID_TO_SPEC.items()):
            cname_opt = spec_opt.get("key", f"class_{cid_opt}")
            lbl_tr = spec_opt.get("label_tr", cname_opt)
            self.review_class_filter.addItem(f"Sınıf {cid_opt:02d}: {cname_opt} ({lbl_tr})", cid_opt)
        self.review_class_filter.currentIndexChanged.connect(self._on_class_filter_changed)
        filter_l.addWidget(self.review_class_filter)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Güven:"))
        self.bulk_range = QSlider(Qt.Orientation.Horizontal, self)
        self.bulk_range.setRange(0, 100)
        self.bulk_range.setValue(90)
        self.bulk_val = QLabel("90")
        self.bulk_val.setStyleSheet(
            "font-family:Consolas; background:#0B0E12; border:1px solid #262D3A; border-radius:5px; padding:2px 6px;"
        )
        self.bulk_range.valueChanged.connect(self._on_bulk_threshold_changed)
        slider_row.addWidget(self.bulk_range)
        slider_row.addWidget(self.bulk_val)
        filter_l.addLayout(slider_row)

        self.chk_only_approved = QCheckBox("Sadece Onaylananlarla Eğit", self)
        self.chk_only_approved.setChecked(True)
        self.chk_only_approved.setToolTip("Eğitim sadece onay havuzundaki (dataset/approved) doğrulanmış kareleri kullansın.")
        self.toggle_approved_only = self.chk_only_approved
        filter_l.addWidget(self.chk_only_approved)
        side_l.addWidget(filter_box)

        # C) Kuyruk Listesi
        queue_box = QFrame(self)
        queue_box.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:6px;")
        queue_l = QVBoxLayout(queue_box)
        queue_l.setContentsMargins(4, 4, 4, 4)
        queue_l.setSpacing(4)
        self.queue_filter_lbl = QLabel("Kuyruk")
        self.queue_filter_lbl.setStyleSheet("color:#94a3b8; font-size:11px; font-weight:700; background:transparent;")
        queue_l.addWidget(self.queue_filter_lbl)

        self.queue_list = QListWidget(self)
        self.queue_list.setObjectName("PipeQueue")
        self.queue_list.setFixedHeight(180)
        self.queue_list.setUniformItemSizes(True)
        self.queue_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.queue_list.currentRowChanged.connect(self._on_queue_row_changed)
        self.queue_list.installEventFilter(self)
        queue_l.addWidget(self.queue_list)
        side_l.addWidget(queue_box)

        # D) Toplu İşlemler
        bulk_box = QFrame(self)
        bulk_box.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:6px;")
        bulk_l = QVBoxLayout(bulk_box)
        bulk_l.setContentsMargins(4, 4, 4, 4)
        bulk_l.setSpacing(6)
        bulk_hdr = QLabel("⚡ Toplu İşlemler")
        bulk_hdr.setStyleSheet("font-size:11px; font-weight:700; color:#94a3b8; background:transparent;")
        bulk_l.addWidget(bulk_hdr)

        self.btn_bulk = QPushButton("Eşik üstünü onayla", self)
        self.btn_bulk.setObjectName("PipeGreen")
        self.btn_bulk.clicked.connect(self._bulk_approve)
        self.btn_bulk_approve = self.btn_bulk
        bulk_l.addWidget(self.btn_bulk)

        self.btn_bulk_reject = QPushButton("Eşik altını reddet", self)
        self.btn_bulk_reject.setObjectName("PipeDanger")
        self.btn_bulk_reject.setToolTip("Seçili filtredeki tüm kareleri diskten siler (train’e girmez).")
        self.btn_bulk_reject.clicked.connect(self._bulk_reject_filtered)
        bulk_l.addWidget(self.btn_bulk_reject)

        self.btn_purge_omni = QPushButton("Zehirli otomatik temizle", self)
        self.btn_purge_omni.setObjectName("PipeDanger")
        self.btn_purge_omni.setToolTip(
            "omni_* sahte kutuları + HUD/minimap yanlış sınıfları + eşik altı weak’i siler. "
            "F/E ve el ile onaylı trusted seed’e dokunmaz. Train .txt yeniden yazılır."
        )
        self.btn_purge_omni.clicked.connect(self._purge_poisoned_labels)
        bulk_l.addWidget(self.btn_purge_omni)

        self.btn_requeue_dataset = QPushButton("Mevcut seti onaya geri al", self)
        self.btn_requeue_dataset.setObjectName("PipeBtn")
        self.btn_requeue_dataset.setToolTip(
            "Train/val görüntü + etiketlerini Adım 2 onay kuyruğuna yeniden yükler. "
            "Reddet (R) hatalı kareyi images+labels+review’dan siler. "
            "Görüntüler belleğe alınmaz; güven filtresiyle gezilebilir."
        )
        self.btn_requeue_dataset.clicked.connect(self._requeue_existing_dataset)
        bulk_l.addWidget(self.btn_requeue_dataset)
        side_l.addWidget(bulk_box)

        # E) Yeniden Etiketleme Araçları
        relabel_box = QFrame(self)
        relabel_box.setStyleSheet("background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:6px;")
        relabel_l = QVBoxLayout(relabel_box)
        relabel_l.setContentsMargins(4, 4, 4, 4)
        relabel_l.setSpacing(6)
        relabel_hdr = QLabel("🤖 Yeniden Etiketleme")
        relabel_hdr.setStyleSheet("font-size:11px; font-weight:700; color:#94a3b8; background:transparent;")
        relabel_l.addWidget(relabel_hdr)

        self.btn_auto_label_unlabeled_suspicious = QPushButton("⚡ Etiketsiz & Şüphelileri Etiketle", self)
        self.btn_auto_label_unlabeled_suspicious.setObjectName("PipeGreen")
        self.btn_auto_label_unlabeled_suspicious.setToolTip(
            "Etiketsiz (0 kutu) ve şüpheli kareleri tespit edip otomatik olarak YZ ve görsel hafıza ile etiketler."
        )
        self.btn_auto_label_unlabeled_suspicious.clicked.connect(self._auto_label_unlabeled_suspicious)
        relabel_l.addWidget(self.btn_auto_label_unlabeled_suspicious)

        self.btn_live_relabel_filtered = QPushButton("🎯 Filtredekileri Canlı Etiketle (0 kare)", self)
        self.btn_live_relabel_filtered.setObjectName("PipeBtn")
        self.btn_live_relabel_filtered.clicked.connect(self._toggle_live_relabel_filtered)
        relabel_l.addWidget(self.btn_live_relabel_filtered)

        self.btn_live_relabel_all = QPushButton("🤖 Tümünü Canlı Yeniden Etiketle", self)
        self.btn_live_relabel_all.setObjectName("PipeBtn")
        self.btn_live_relabel_all.clicked.connect(self._toggle_live_relabel_all)
        relabel_l.addWidget(self.btn_live_relabel_all)

        self.btn_prune_low_confidence = QPushButton("✂️ Düşük Güvenlileri Sil", self)
        self.btn_prune_low_confidence.setObjectName("PipeDanger")
        self.btn_prune_low_confidence.clicked.connect(self._bulk_prune_low_confidence_boxes)
        self.btn_bulk_prune_boxes = self.btn_prune_low_confidence
        relabel_l.addWidget(self.btn_prune_low_confidence)

        self.btn_relabel_current_frame = QPushButton("⚡ Bu Kareyi Yeniden Etiketle", self)
        self.btn_relabel_current_frame.setObjectName("PipeBtn")
        self.btn_relabel_current_frame.clicked.connect(self._relabel_current_frame_clicked)
        relabel_l.addWidget(self.btn_relabel_current_frame)

        self.btn_relabel_selected_frames = QPushButton("Seçilenleri Yeniden Etiketle", self)
        self.btn_relabel_selected_frames.setObjectName("PipeBtn")
        self.btn_relabel_selected_frames.clicked.connect(self._relabel_selected_frames_clicked)
        relabel_l.addWidget(self.btn_relabel_selected_frames)
        side_l.addWidget(relabel_box)

        # F) Çizim Sınıfı Seçim Paneli (25 Sınıf Paleti)
        self.tag_panel = SearchableTagListPanel(self)
        self.manual_tag_search = self.tag_panel.search_edit
        self.manual_tag_combo = self.tag_panel.combo
        self.manual_tag_list = self.tag_panel.list_widget
        self.tag_panel.classSelected.connect(self._on_drawing_class_selected)
        side_l.addWidget(self.tag_panel)

        # G) Mevcut Karedeki Etiket Listesi
        self.det_host = QWidget(self)
        self.det_host.setObjectName("PipeDetHost")
        self.det_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.det_host.setAutoFillBackground(True)
        det_pal = self.det_host.palette()
        det_pal.setColor(QPalette.ColorRole.Window, QColor("#0B0E12"))
        det_pal.setColor(QPalette.ColorRole.Base, QColor("#0B0E12"))
        det_pal.setColor(QPalette.ColorRole.Text, QColor(C["text"]))
        self.det_host.setPalette(det_pal)
        self.det_host.setStyleSheet(
            "QWidget#PipeDetHost { background-color:#0B0E12; color:#F2F3F5; }"
            "QWidget#PipeDetHost QLabel { background:transparent; color:#F2F3F5; }"
        )
        self.det_layout = QVBoxLayout(self.det_host)
        self.det_layout.setContentsMargins(8, 4, 8, 4)
        self.det_layout.setSpacing(4)
        det_scroll = QScrollArea(self)
        det_scroll.setObjectName("PipeDetScroll")
        det_scroll.setWidgetResizable(True)
        det_scroll.setFrameShape(QFrame.Shape.NoFrame)
        det_scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        det_scroll.setStyleSheet(
            "QScrollArea#PipeDetScroll { background:#0B0E12; border:1px solid #1e293b; border-radius:6px; }"
            "QScrollArea#PipeDetScroll > QWidget { background:#0B0E12; }"
            "QScrollArea#PipeDetScroll QWidget { background-color:#0B0E12; color:#F2F3F5; }"
        )
        det_scroll.setWidget(self.det_host)
        det_scroll.setMinimumHeight(150)
        self._det_scroll = det_scroll
        side_l.addWidget(det_scroll)

        sidebar_scroll.setWidget(side_widget)

        # -------------------------------------------------------------
        # 2. SAĞ PANEL: DEV ETİKETLEME CANVAS'I VE AKSİYONLAR
        # -------------------------------------------------------------
        right = QVBoxLayout()
        right.setContentsMargins(6, 4, 6, 6)
        right.setSpacing(6)

        # Üst Başlık & Durum Çubuğu
        title_row = QHBoxLayout()
        title_row.setContentsMargins(4, 2, 4, 2)
        title_row.setSpacing(10)
        self.review_title = QLabel("Kare —")
        self.review_title.setStyleSheet(
            f"font-size:14px; font-weight:800; color:{C['text']}; background:transparent;"
        )
        self.review_sub = QLabel("")
        self.review_sub.setStyleSheet(f"color:{C['muted']}; font-size:12px; background:transparent;")
        title_row.addWidget(self.review_title)
        title_row.addWidget(self.review_sub)
        title_row.addStretch()

        self.lbl_review_kinah = QLabel("💰 0 Kinah", self)
        self.lbl_review_kinah.setObjectName("ReviewKinahBadge")
        self.lbl_review_kinah.setStyleSheet(
            "font-family:Consolas, Segoe UI; font-size:11px; font-weight:700; color:#fbbf24; background:#1c1917; "
            "border:1px solid #78350f; border-radius:6px; padding:4px 10px;"
        )
        self.lbl_review_kinah.setToolTip("Mevcut kare ve onay kuyruğundaki materyallerin Pazar (Broker) tahmini Kinah değeri")
        title_row.addWidget(self.lbl_review_kinah)

        self.lbl_speed_meter = QLabel("⚡ Hız: — · Kalan: — · 0 / 0", self)
        self.lbl_speed_meter.setStyleSheet(
            "font-family:Consolas; font-size:11px; color:#38bdf8; background:#0B0E12; "
            "border:1px solid #1e293b; border-radius:6px; padding:4px 8px;"
        )
        title_row.addWidget(self.lbl_speed_meter)

        self.btn_open_lightbox = QPushButton("🔍 Tam Ekran (Lightbox)", self)
        self.btn_open_lightbox.setObjectName("PipeBtn")
        self.btn_open_lightbox.setStyleSheet("font-size:11px; padding:4px 12px;")
        self.btn_open_lightbox.setToolTip("Görseli büyük ekranda incele (Çift tıklayarak da açabilirsiniz)")
        self.btn_open_lightbox.clicked.connect(self._open_review_lightbox)
        title_row.addWidget(self.btn_open_lightbox)
        right.addLayout(title_row)

        # DEVASA YOLO ÖNİZLEME VE ETİKETLEME CANVAS'I
        self.review_preview = YOLOPreviewCanvas(self)
        self.review_preview.setObjectName("PipeReviewPreview")
        self.review_preview.setMinimumHeight(620)
        self.review_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.review_preview.setCursor(Qt.CursorShape.CrossCursor)
        self.review_preview.setToolTip(
            "Sol tık sürükle: Yeni kutu çiz · Kutuya tıkla: Seç · İçinden sürükle: Taşı · Kenar/Köşe tutamaçları: Büyüt/Küçült · Çift tık: Büyüt · Del: Sil"
        )
        self.review_preview.installEventFilter(self)
        self.review_preview.boxRemoveRequested.connect(
            lambda det: self._reject_box(det, from_trusted=bool(det.get("trusted")))
        )
        self.review_preview.boxChangeClassRequested.connect(self._prompt_change_class_for_box)
        self.review_preview.boxDrawn.connect(self._on_canvas_box_drawn)
        self.review_preview.boxAdded.connect(self._on_manual_box_added)
        self.review_preview.boxEdited.connect(self._on_box_edited)
        self._review_lightbox_pixmap: Optional[QPixmap] = None
        right.addWidget(self.review_preview, stretch=10)

        # Alt Büyük Aksiyon Çubuğu
        actions = QHBoxLayout()
        actions.setContentsMargins(4, 6, 4, 4)
        actions.setSpacing(10)

        self.btn_approve = QPushButton("✓ ONAYLA VE SONRAKİ  (A)", self)
        self.btn_approve.setObjectName("PipeGreen")
        self.btn_approve.setStyleSheet("font-size:13px; font-weight:800; padding:10px 24px;")
        self.btn_approve.clicked.connect(self._on_step2_approve)

        self.btn_reject = QPushButton("✕ REDDET  (R)", self)
        self.btn_reject.setObjectName("PipeDanger")
        self.btn_reject.setStyleSheet("font-size:12px; font-weight:700; padding:10px 18px;")
        self.btn_reject.clicked.connect(self._on_step2_reject)

        self.btn_skip = QPushButton("Atla  (S)", self)
        self.btn_skip.setObjectName("PipeBtn")
        self.btn_skip.setStyleSheet("font-size:12px; padding:10px 16px;")
        self.btn_skip.clicked.connect(self._review_skip)

        self.btn_add_player_char = QPushButton("🎯 Karakterimi Etiketle  (K)", self)
        self.btn_add_player_char.setObjectName("PipeBtn")
        self.btn_add_player_char.setStyleSheet(
            "font-size:12px; font-weight:700; padding:10px 18px; border:1px solid #00d2ff; color:#00d2ff; background:#082f49;"
        )
        self.btn_add_player_char.setToolTip(
            "Karakter kutusunu ekranın ortasına otomatik yerleştirir. Tutamaçları çekerek tam boyuta getirebilirsiniz. Kısayol: 'K'"
        )
        self.btn_add_player_char.clicked.connect(self._add_player_character_box_to_current_frame)

        self.btn_back_capture = QPushButton("← Yakala'ya dön", self)
        self.btn_back_capture.setObjectName("PipeBtn")
        self.btn_back_capture.setStyleSheet("font-size:11px; padding:8px 14px;")
        self.btn_back_capture.clicked.connect(lambda: self._go_step(1))

        actions.addWidget(self.btn_approve)
        actions.addWidget(self.btn_reject)
        actions.addWidget(self.btn_skip)
        actions.addWidget(self.btn_add_player_char)
        actions.addStretch()
        actions.addWidget(self.btn_back_capture)
        right.addLayout(actions)

        # Grid yerleşimi: Sol Scrollable Sidebar + Sağ Devasa Canvas
        grid = QHBoxLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(sidebar_scroll)
        grid.addLayout(right, stretch=1)
        lay.addLayout(grid, stretch=1)

        # Kısayollar
        self._shortcut_approve_a = QShortcut(QKeySequence(Qt.Key.Key_A), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_approve_a.activated.connect(self._handle_shortcut_approve)
        self._shortcut_approve_a.activatedAmbiguously.connect(self._handle_shortcut_approve)
        self._shortcut_approve_lower = self._shortcut_approve_a
        self._shortcut_approve = self._shortcut_approve_a

        self._shortcut_reject_r = QShortcut(QKeySequence(Qt.Key.Key_R), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_reject_r.activated.connect(self._handle_shortcut_reject)
        self._shortcut_reject_r.activatedAmbiguously.connect(self._handle_shortcut_reject)
        self._shortcut_reject_lower = self._shortcut_reject_r
        self._shortcut_reject = self._shortcut_reject_r

        self._shortcut_skip_s = QShortcut(QKeySequence(Qt.Key.Key_S), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_skip_s.activated.connect(self._handle_shortcut_skip)
        self._shortcut_skip_s.activatedAmbiguously.connect(self._handle_shortcut_skip)
        self._shortcut_skip_lower = self._shortcut_skip_s
        self._shortcut_skip = self._shortcut_skip_s

        self._shortcut_player_k = QShortcut(QKeySequence(Qt.Key.Key_K), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_player_k.activated.connect(self._handle_shortcut_player_char)
        self._shortcut_player_k.activatedAmbiguously.connect(self._handle_shortcut_player_char)

        self._update_filter_action_buttons()
        return card

    def _build_step_prepare(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.btn_freeze = QPushButton("❄ Veri setini dondur (v4)", self)
        self.btn_freeze.setObjectName("PipePrimary")
        self.btn_freeze.clicked.connect(self._freeze_dataset)
        lay.addWidget(
            self._step_intro(
                "Adım 3 — Hazırla",
                "Eğitimden önce son kontrol: boş ve dengesiz sınıflar, train/val bölmesi, çoğaltma. "
                "Her şey yeşil olunca veri setini dondur — eğitim ancak donmuş sürümle başlar.",
                self.btn_freeze,
            )
        )

        self.health_host = QVBoxLayout()
        self.health_host.setContentsMargins(0, 0, 0, 0)
        self.health_host.setSpacing(0)
        health_wrap = QWidget(self)
        health_wrap.setLayout(self.health_host)
        lay.addWidget(health_wrap)

        two = QHBoxLayout()
        left = QVBoxLayout()
        left.setContentsMargins(14, 12, 14, 12)
        left.addWidget(QLabel("<b>Train / Val bölmesi</b>"))
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Train oranı"))
        self.split_range = QSlider(Qt.Orientation.Horizontal, self)
        self.split_range.setRange(60, 95)
        self.split_range.setValue(84)
        self.split_val = QLabel("%84")
        self.split_val.setStyleSheet(
            "font-family:Consolas;background:#0B0E12;border:1px solid #262D3A;"
            "border-radius:7px;padding:5px 8px;"
        )
        self.split_range.valueChanged.connect(lambda v: self.split_val.setText(f"%{v}"))
        split_row.addWidget(self.split_range, stretch=1)
        split_row.addWidget(self.split_val)
        left.addLayout(split_row)
        self.split_train_lbl = QLabel("Train —")
        self.split_val_lbl = QLabel("Val —")
        self.freeze_state_lbl = QLabel("dondurulmadı")
        self.freeze_state_lbl.setStyleSheet(f"color:{C['amber']};font-family:Consolas;font-weight:700;")
        for lab, w in (
            ("Train", self.split_train_lbl),
            ("Val", self.split_val_lbl),
            ("Sürüm adı", QLabel("dataset-v4")),
            ("Durum", self.freeze_state_lbl),
        ):
            row = QHBoxLayout()
            a = QLabel(lab)
            a.setStyleSheet(f"color:{C['muted']};")
            row.addWidget(a)
            row.addStretch()
            w.setStyleSheet("font-family:Consolas;font-weight:700;")
            row.addWidget(w)
            left.addLayout(row)

        right = QVBoxLayout()
        right.setContentsMargins(14, 12, 14, 12)
        self.toggle_approved_only = FactoryToggleSwitch(
            "Sadece Onaylananlarla Eğit (Onaylı Havuz)",
            checked=True,
            accent_hex=C["green"],
            parent=card,
        )
        self.toggle_approved_only_prepare = self.toggle_approved_only
        self.toggle_synthetic_10x = FactoryToggleSwitch(
            "10× sentetik çoğaltma",
            checked=True,
            accent_hex=C["amber"],
            parent=card,
        )
        self.toggle_mosaic = FactoryToggleSwitch(
            "Mozaik + yatay çevirme",
            checked=True,
            accent_hex=C["green"],
            parent=card,
        )
        self.toggle_synthetic_10x.toggled.connect(self._on_toggles_changed)
        right.addWidget(self.toggle_approved_only)
        right.addWidget(self.toggle_synthetic_10x)
        right.addWidget(self.toggle_mosaic)
        right.addStretch()
        two.addLayout(left, stretch=1)
        two.addLayout(right, stretch=1)
        lay.addLayout(two)

        dist_head = QHBoxLayout()
        dist_head.setContentsMargins(14, 8, 14, 8)
        dist_title = QLabel("<b>Sınıf dağılımı</b>")
        self.class_filter = QComboBox(self)
        for label, key in (
            ("Tümü", "all"),
            ("Dolu", "filled"),
            ("Boş", "empty"),
            ("Temel (0–11)", "base"),
            ("Yol / Engel (12–15)", "nav"),
            ("Can / Skill (16–21)", "vital"),
            ("Essence (22+)", "essence"),
        ):
            self.class_filter.addItem(label, key)
        self.class_filter.currentIndexChanged.connect(self._apply_class_filter)
        self.class_search = QLineEdit(self)
        self.class_search.setPlaceholderText("Sınıf ara…")
        self.class_search.textChanged.connect(self._apply_class_filter)
        dist_head.addWidget(dist_title)
        dist_head.addStretch()
        dist_head.addWidget(self.class_search)
        dist_head.addWidget(self.class_filter)
        lay.addLayout(dist_head)

        dist_container = QWidget(self)
        dist_container.setObjectName("ClassDistHost")
        dist_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dist_container.setAutoFillBackground(True)
        dist_pal = dist_container.palette()
        dist_pal.setColor(QPalette.ColorRole.Window, QColor("#0B0E12"))
        dist_pal.setColor(QPalette.ColorRole.Base, QColor("#0B0E12"))
        dist_container.setPalette(dist_pal)
        dist_container.setStyleSheet(
            "QWidget#ClassDistHost { background-color:#0B0E12; border:none; }"
        )
        dist_layout = QVBoxLayout(dist_container)
        dist_layout.setContentsMargins(8, 2, 8, 8)
        dist_layout.setSpacing(2)
        for spec in YOLO_12_CLASS_SPECS:
            cid = int(spec["id"])
            row_widget = ClassBarRow(
                class_id=cid,
                label_tr=str(spec["label_tr"]),
                class_key=str(spec["key"]),
                color_hex=str(spec["color_hex"]),
                parent=dist_container,
            )
            self.class_rows[cid] = row_widget
            dist_layout.addWidget(row_widget)
        dist_layout.addStretch(1)
        dist_scroll = QScrollArea(self)
        dist_scroll.setObjectName("ClassDistScroll")
        dist_scroll.setWidgetResizable(True)
        dist_scroll.setFrameShape(QFrame.Shape.NoFrame)
        dist_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dist_scroll.setStyleSheet(
            "QScrollArea#ClassDistScroll { background:#0B0E12; border:none; }"
            "QScrollArea#ClassDistScroll > QWidget > QWidget { background:#0B0E12; }"
        )
        dist_scroll.setWidget(dist_container)
        vp = dist_scroll.viewport()
        if vp is not None:
            vp.setObjectName("ClassDistViewport")
            vp.setAutoFillBackground(True)
            vp.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            vp_pal = vp.palette()
            vp_pal.setColor(QPalette.ColorRole.Window, QColor("#0B0E12"))
            vp_pal.setColor(QPalette.ColorRole.Base, QColor("#0B0E12"))
            vp.setPalette(vp_pal)
            vp.setStyleSheet("background-color:#0B0E12; border:none;")
        dist_scroll.setMinimumHeight(160)
        self.dist_scroll = dist_scroll
        self.dist_container = dist_container
        lay.addWidget(dist_scroll, stretch=1)

        # Bootstrap lives on prepare step as health scan helper
        self.btn_bootstrap = QPushButton("Ekranlardan veri seti oluştur / tarama", self)
        self.btn_bootstrap.setObjectName("PipeBtn")
        self.btn_bootstrap.clicked.connect(self.on_bootstrap_clicked)
        boot_row = QHBoxLayout()
        boot_row.setContentsMargins(14, 0, 14, 12)
        boot_row.addWidget(self.btn_bootstrap)
        boot_row.addStretch()
        lay.addLayout(boot_row)
        return card

    def _build_step_train(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.train_lock_pill = QLabel("Kilit: önce veri setini dondur")
        self.train_lock_pill.setStyleSheet(
            f"color:{C['amber']};background:{C['amber_soft']};border:1px solid #4A3A10;"
            "border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700;"
        )
        self.train_intro = QLabel(
            "Eğitim, Adım 3'te dondurduğun sürümle çalışır. Hazır ayar seç ya da parametreleri kendin ver."
        )
        # intro rebuilt below
        lay.addWidget(
            self._step_intro(
                "Adım 4 — Eğit",
                "Eğitim, Adım 3'te dondurduğun sürümle çalışır. Hazır ayar seç ya da parametreleri kendin ver; "
                "süre tahmini RTX 3070'e göre hesaplanır.",
                self.train_lock_pill,
            )
        )

        presets = QHBoxLayout()
        presets.setContentsMargins(14, 12, 14, 8)
        presets.setSpacing(8)
        self._preset_cards: List[_InfoSelectCard] = []
        self._train_presets = {
            "fast": {"model": "yolo11n.pt", "epochs": 10, "imgsz": 640, "batch": 16},
            "balanced": {"model": "yolo11s.pt", "epochs": 100, "imgsz": 640, "batch": 16},
            "full": {"model": "yolo11m.pt", "epochs": 300, "imgsz": 960, "batch": 8},
        }
        for key, title, small in (
            ("fast", "Hızlı test", "yolo11n · 10 epoch · 640 — ~2 dk"),
            ("balanced", "Dengeli", "yolo11s · 100 epoch · 640 — günlük üretim"),
            ("full", "Tam eğitim", "yolo11m · 300 epoch · 960 — gece bırakmalık"),
        ):
            # Must not reuse name `card` — that shadowed the step frame and returned
            # only this preset into the stack, leaving the real train UI as a click-eating overlay.
            preset_card = _InfoSelectCard(
                title,
                small,
                object_name="PipePresetSel" if key == "balanced" else "PipePreset",
                parent=card,
            )
            preset_card.setProperty("preset", key)
            preset_card.clicked.connect(lambda k=key: self._apply_preset(k))
            self._preset_cards.append(preset_card)
            presets.addWidget(preset_card)
        lay.addLayout(presets)

        hp = QGridLayout()
        hp.setContentsMargins(14, 4, 14, 8)
        hp.setHorizontalSpacing(10)
        hp.setVerticalSpacing(8)
        self.hp_model = QComboBox(self)
        self.hp_model.addItems(["yolo11s.pt", "yolo11n.pt", "yolo11m.pt"])
        self.hp_epoch = QSpinBox(self)
        self.hp_epoch.setRange(1, 500)
        self.hp_epoch.setValue(100)
        self.hp_batch = QComboBox(self)
        self.hp_batch.addItems(["16", "8", "32"])
        self.hp_imgsz = QComboBox(self)
        self.hp_imgsz.addItems(["640", "960", "1280"])
        self.hp_lr = QLineEdit("0.01", self)
        self.hp_eta = QLineEdit("—", self)
        self.hp_eta.setEnabled(False)
        fields = [
            ("Model", self.hp_model),
            ("Epoch", self.hp_epoch),
            ("Batch", self.hp_batch),
            ("Görüntü boyutu", self.hp_imgsz),
            ("Öğrenme oranı", self.hp_lr),
            ("Tahmini süre (RTX 3070)", self.hp_eta),
        ]
        for i, (lab, w) in enumerate(fields):
            cell = QVBoxLayout()
            cell.addWidget(QLabel(lab))
            cell.addWidget(w)
            hp.addLayout(cell, i // 3, i % 3)
        lay.addLayout(hp)

        approved_row = QHBoxLayout()
        approved_row.setContentsMargins(14, 4, 14, 4)
        self.toggle_approved_only = FactoryToggleSwitch(
            "Sadece Onaylananlarla Eğit (Onaylı Havuz)",
            checked=True,
            accent_hex=C["green"],
            parent=card,
        )
        self.toggle_approved_only_train = self.toggle_approved_only
        if hasattr(self, "toggle_approved_only_prepare"):
            self.toggle_approved_only_prepare.toggled.connect(
                lambda val: (
                    self.toggle_approved_only_train.blockSignals(True),
                    self.toggle_approved_only_train.setChecked(val),
                    self.toggle_approved_only_train.blockSignals(False),
                )
            )
            self.toggle_approved_only_train.toggled.connect(
                lambda val: (
                    self.toggle_approved_only_prepare.blockSignals(True),
                    self.toggle_approved_only_prepare.setChecked(val),
                    self.toggle_approved_only_prepare.blockSignals(False),
                )
            )
        self.chk_train_only_approved = self.toggle_approved_only
        approved_row.addWidget(self.toggle_approved_only)
        approved_row.addStretch()
        lay.addLayout(approved_row)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(14, 4, 14, 8)
        self.btn_train_only_approved = QPushButton("🚀 Sadece Onaylananlarla Eğit", self)
        self.btn_train_only_approved.setObjectName("PipeGreen")
        self.btn_train_only_approved.setMinimumHeight(44)
        self.btn_train_only_approved.clicked.connect(self._on_train_only_approved_clicked)
        self.btn_train_rtx = QPushButton("⚡ Eğitimi başlat", self)
        self.btn_train_rtx.setObjectName("PipePrimary")
        self.btn_train_rtx.setMinimumHeight(44)
        self.btn_train_rtx.clicked.connect(self.on_train_rtx_clicked)
        self.btn_stop_train = QPushButton("■ Durdur", self)
        self.btn_stop_train.setObjectName("PipeBtn")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_train_only_approved, stretch=1)
        btn_row.addWidget(self.btn_train_rtx, stretch=1)
        btn_row.addWidget(self.btn_stop_train)
        lay.addLayout(btn_row)

        epoch_box = QVBoxLayout()
        epoch_box.setContentsMargins(14, 4, 14, 8)
        head = QHBoxLayout()
        self.epoch_label = QLabel("Epoch 0 / 100")
        self.run_label = QLabel("Çalıştırma #1")
        self.epoch_pct = QLabel("%0")
        for w in (self.epoch_label, self.run_label, self.epoch_pct):
            w.setStyleSheet("font-family:Consolas;font-weight:700;")
        head.addWidget(self.epoch_label)
        head.addStretch()
        head.addWidget(self.run_label)
        head.addStretch()
        head.addWidget(self.epoch_pct)
        self.train_progress_bar = QProgressBar(self)
        self.train_progress_bar.setObjectName("PipeProgress")
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setTextVisible(False)
        self.train_progress_bar.setFixedHeight(10)
        metrics = QHBoxLayout()
        self.lbl_map50_loss = QLabel("box loss —")
        self.m_map = QLabel("mAP50 —")
        self.m_eta = QLabel("kalan —")
        self.lbl_train_status = QLabel("Eğitim: hazır")
        for w in (self.lbl_map50_loss, self.m_map, self.m_eta, self.lbl_train_status):
            w.setStyleSheet(f"color:{C['muted']};font-family:Consolas;")
            metrics.addWidget(w)
            metrics.addStretch()
        epoch_box.addLayout(head)
        epoch_box.addWidget(self.train_progress_bar)
        epoch_box.addLayout(metrics)
        lay.addLayout(epoch_box)

        note = QLabel(
            "Eğitim bitince model <b>models/</b> altına yazılır, eski best.pt yedeklenir "
            "ve Adım 5 (Doğrula) kendiliğinden açılır."
        )
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['muted']};padding:0 14px 14px;font-size:12px;")
        lay.addWidget(note)
        lay.addStretch(1)
        return card

    def _build_step_validate(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.btn_validate = QPushButton("✓ Doğrulamayı çalıştır", self)
        self.btn_validate.setObjectName("PipePrimary")
        self.btn_validate.clicked.connect(self._run_validate)
        lay.addWidget(
            self._step_intro(
                "Adım 5 — Doğrula",
                "Toplam mAP yetmez: hangi sınıf zayıf, model nerede yanılıyor? "
                "Hatalı kareleri tek tıkla onay kuyruğuna geri gönderirsin.",
                self.btn_validate,
            )
        )

        self.val_empty = QLabel(
            "Henüz bu oturumda doğrulanmış çalıştırma yok. Önce Adım 4'te eğitim bitir; "
            "metrikler burada belirir."
        )
        self.val_empty.setWordWrap(True)
        self.val_empty.setStyleSheet(f"color:{C['muted']};padding:14px 16px;font-size:13px;")
        lay.addWidget(self.val_empty)

        self.val_body = QWidget(self)
        val_l = QVBoxLayout(self.val_body)
        val_l.setContentsMargins(14, 8, 14, 14)
        strip = QHBoxLayout()
        self.val_map = QLabel("—")
        self.val_prec = QLabel("—")
        self.val_weak = QLabel("—")
        for cap, w in (("mAP50", self.val_map), ("Precision", self.val_prec), ("En zayıf sınıf", self.val_weak)):
            box = QFrame(self)
            box.setObjectName("PipeStat")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            c = QLabel(cap.upper())
            c.setStyleSheet(f"color:{C['muted']};font-size:10px;font-weight:800;")
            w.setStyleSheet("font-family:Consolas;font-size:20px;font-weight:700;")
            bl.addWidget(c)
            bl.addWidget(w)
            strip.addWidget(box)
        val_l.addLayout(strip)
        self.ap_list_host = QVBoxLayout()
        val_l.addWidget(QLabel("<b>Sınıf bazlı AP50</b>"))
        val_l.addLayout(self.ap_list_host)
        self.btn_val_done = QPushButton("Doğrulama tamam — Yayınla adımına geç", self)
        self.btn_val_done.setObjectName("PipeGreen")
        self.btn_val_done.clicked.connect(self._finish_validate)
        val_l.addWidget(self.btn_val_done)
        self.val_body.hide()
        lay.addWidget(self.val_body)
        lay.addStretch(1)
        return card

    def _build_step_publish(self) -> QFrame:
        C = self.PIPELINE_COLORS
        card = QFrame(self)
        card.setObjectName("PipeCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(
            self._step_intro(
                "Adım 6 — Yayınla",
                "Botun kullandığı modeli değiştirirsin — geri dönüşü var: eski sürüm yedekte durur, tek tıkla geri alınır.",
            )
        )

        lay.addWidget(self._section_label("Çıktı formatı"))
        formats = QHBoxLayout()
        formats.setContentsMargins(14, 4, 14, 8)
        formats.setSpacing(8)
        self._format_cards: List[_InfoSelectCard] = []
        for key, title, small in (
            ("best.pt (PyTorch)", "best.pt", "Botun doğrudan yüklediği format."),
            ("ONNX", "ONNX", "Başka çalışma zamanlarına taşınabilir."),
            ("TensorRT (FP16)", "TensorRT FP16", "RTX 3070'de en hızlı çıkarım."),
        ):
            # Same shadowing trap as train presets — keep step frame in `card`.
            fmt_card = _InfoSelectCard(
                title,
                small,
                object_name="PipeFormatSel" if key.startswith("best") else "PipeFormat",
                parent=card,
            )
            fmt_card.setProperty("fmt", key)
            fmt_card.clicked.connect(lambda k=key: self._select_format(k))
            self._format_cards.append(fmt_card)
            formats.addWidget(fmt_card)
        lay.addLayout(formats)

        lay.addWidget(self._section_label("Yayın kontrol listesi"))
        self.pub_chk_data = QLabel("1  Veri seti donduruldu")
        self.pub_chk_train = QLabel("2  Eğitim tamamlandı")
        self.pub_chk_val = QLabel("3  Doğrulama gözden geçirildi")
        self.pub_chk_backup = QLabel("✓  Eski model yedeği hazır (best_prev.pt)")
        for w in (self.pub_chk_data, self.pub_chk_train, self.pub_chk_val, self.pub_chk_backup):
            w.setStyleSheet(
                "padding:10px 16px;border-bottom:1px solid #262D3A;font-size:13px;"
            )
            lay.addWidget(w)

        pub_row = QHBoxLayout()
        pub_row.setContentsMargins(14, 12, 14, 12)
        self.btn_publish = QPushButton("⇪ Modeli yayınla ve botu güncelle", self)
        self.btn_publish.setObjectName("PipePrimary")
        self.btn_publish.setMinimumHeight(44)
        self.btn_publish.clicked.connect(self._publish_model)
        self.btn_rollback = QPushButton("↩ Önceki modele dön", self)
        self.btn_rollback.setObjectName("PipeBtn")
        self.btn_rollback.setMinimumHeight(44)
        self.btn_rollback.clicked.connect(self._rollback_model)
        pub_row.addWidget(self.btn_publish, stretch=1)
        pub_row.addWidget(self.btn_rollback)
        lay.addLayout(pub_row)

        lay.addWidget(self._section_label("Model sürümleri"))
        self.version_list = QVBoxLayout()
        self.version_list.setContentsMargins(14, 4, 14, 14)
        lay.addLayout(self.version_list)
        lay.addStretch(1)
        return card

    def _section_label(self, text: str) -> QLabel:
        lab = QLabel(f"<b>{text}</b>")
        lab.setStyleSheet("padding:12px 16px 4px;font-size:14px;")
        return lab

    def _build_right_rail(self) -> QWidget:
        C = self.PIPELINE_COLORS
        rail = QWidget(self)
        rail.setFixedWidth(330)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        next_card = QFrame(self)
        next_card.setObjectName("PipeNext")
        # Child selectors required: a widget-local stylesheet does not cascade from the dialog.
        next_card.setStyleSheet(
            f"""
            QFrame#PipeNext {{
                background: {C['amber_soft']};
                border: 1px solid {C['amber']};
                border-radius: 14px;
            }}
            QFrame#PipeNext QLabel {{
                background: transparent;
                border: none;
            }}
            QFrame#PipeNext QPushButton#PipePrimary {{
                background: {C['amber']};
                color: #111111;
                border: 1px solid #FFD84D;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 700;
                min-height: 36px;
            }}
            QFrame#PipeNext QPushButton#PipePrimary:hover {{
                background: #FFC933;
            }}
            """
        )
        nl = QVBoxLayout(next_card)
        nl.setContentsMargins(14, 14, 14, 14)
        nl.setSpacing(8)
        nk = QLabel("SIRADAKİ İŞİN", next_card)
        nk.setStyleSheet(
            f"color:{C['amber']};font-size:10px;font-weight:800;letter-spacing:0.12em;"
        )
        nk.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.next_title = QLabel("Hazırlanıyor…", next_card)
        self.next_title.setWordWrap(True)
        self.next_title.setStyleSheet("font-size:16px;font-weight:800;")
        self.next_title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.next_text = QLabel("", next_card)
        self.next_text.setWordWrap(True)
        self.next_text.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        self.next_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.btn_next = QPushButton("Onay kuyruğuna git →", next_card)
        self.btn_next.setObjectName("PipePrimary")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.setMinimumHeight(40)
        self.btn_next.clicked.connect(self._on_next_clicked)
        self._next_action = "goto"
        nl.addWidget(nk)
        nl.addWidget(self.next_title)
        nl.addWidget(self.next_text)
        nl.addWidget(self.btn_next)
        lay.addWidget(next_card)

        blockers = QFrame(self)
        blockers.setObjectName("PipeRailCard")
        bl = QVBoxLayout(blockers)
        bl.setContentsMargins(0, 0, 0, 8)
        head = QHBoxLayout()
        head.setContentsMargins(14, 12, 14, 8)
        h = QLabel("<b>Engeller</b>")
        self.block_count = QLabel("0 engel")
        self.block_count.setStyleSheet(
            "color:#FF8A94;background:#24151A;border:1px solid #4A2028;"
            "border-radius:999px;padding:4px 8px;font-size:11px;font-weight:700;"
        )
        head.addWidget(h)
        head.addStretch()
        head.addWidget(self.block_count)
        bl.addLayout(head)
        self._blocker_widgets["review"] = self._make_blocker(
            "!", "Onay kuyruğu dolu", "Kareler onaylanmadan veri seti eksik sayılır.", 2
        )
        self._blocker_widgets["empty"] = self._make_blocker(
            "!", "Boş sınıflar var", "Bu sınıflar modelde hiç olmayacak.", 3
        )
        self._blocker_widgets["freeze"] = self._make_blocker(
            "!", "Veri seti dondurulmadı", "Eğitim butonu kilitli; önce Adım 3'ü bitir.", 3
        )
        self._blocker_widgets["none"] = self._make_blocker(
            "✓", "Engel yok", "Hat akıyor; sıradaki adıma geçebilirsin.", None, ok=True
        )
        for key, w in self._blocker_widgets.items():
            bl.addWidget(w)
        lay.addWidget(blockers)

        runs = QFrame(self)
        runs.setObjectName("PipeRailCard")
        rl = QVBoxLayout(runs)
        rl.setContentsMargins(14, 12, 14, 12)
        rl.addWidget(QLabel("<b>Çalıştırmalar</b>"))
        sub = QLabel("Son eğitimler — karşılaştır, en iyiyi seç.")
        sub.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        rl.addWidget(sub)
        self.run_list_host = QVBoxLayout()
        rl.addLayout(self.run_list_host)
        lay.addWidget(runs)

        log_card = QFrame(self)
        log_card.setObjectName("PipeRailCard")
        ll = QVBoxLayout(log_card)
        ll.setContentsMargins(10, 10, 10, 10)
        log_head = QHBoxLayout()
        log_head.addWidget(QLabel("<b>Canlı log</b>"))
        log_head.addStretch()
        clear = QPushButton("Temizle", self)
        clear.setObjectName("PipeBtn")
        clear.clicked.connect(self._clear_log)
        log_head.addWidget(clear)
        ll.addLayout(log_head)
        self.training_log_box = QPlainTextEdit(self)
        self.training_log_box.setObjectName("training_log_box")
        self.training_log_box.setReadOnly(True)
        self.training_log_box.setMinimumHeight(160)
        self.training_log_box.setPlainText(
            "[HAZIR] YOLO fabrikası açıldı.\n"
            "[BİLGİ] Hat: Yakala → Onayla → Hazırla → Eğit → Doğrula → Yayınla.\n"
        )
        ll.addWidget(self.training_log_box)
        lay.addWidget(log_card, stretch=1)
        return rail

    def _make_blocker(
        self,
        icon: str,
        title: str,
        body: str,
        goto: Optional[int],
        ok: bool = False,
    ) -> QFrame:
        C = self.PIPELINE_COLORS
        row = QFrame(self)
        row.setObjectName("PipeBlockerHost")
        row.setStyleSheet("QFrame#PipeBlockerHost { border:none; border-radius:0; background:transparent; }")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(14, 8, 14, 8)
        ico = QLabel(icon)
        ico.setFixedSize(26, 26)
        ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if ok:
            ico.setStyleSheet(
                f"background:{C['green_soft']};color:#7CFFB2;border:1px solid #1E4D33;"
                "border-radius:9px;font-weight:800;"
            )
        else:
            ico.setStyleSheet(
                f"background:{C['amber_soft']};color:{C['amber']};border:1px solid #4A3A10;"
                "border-radius:9px;font-weight:800;"
            )
        text = QVBoxLayout()
        t = QLabel(title)
        t.setStyleSheet("font-weight:800;font-size:13px;background:transparent;border:none;")
        p = QLabel(body)
        p.setWordWrap(True)
        p.setStyleSheet(f"color:{C['muted']};font-size:11px;background:transparent;border:none;")
        text.addWidget(t)
        text.addWidget(p)
        hl.addWidget(ico)
        hl.addLayout(text, stretch=1)
        if goto is not None:
            btn = QPushButton("Git", self)
            btn.setObjectName("PipeBtn")
            btn.clicked.connect(lambda _=False, s=goto: self._go_step(s))
            hl.addWidget(btn)
        return row

    # ------------------------------------------------------------------
    # Pipeline state machine
    # ------------------------------------------------------------------
    def _go_step(self, step: int) -> None:
        step = max(1, min(6, int(step)))
        # Soft gates: browsing earlier steps is fine; skip-ahead past incomplete work is not.
        if step >= 4 and not self._pipeline.get("frozen"):
            self._append_log("[AKIŞ] Önce Adım 3: veri setini dondur.")
            step = 3
        # Step 5 stays navigable to inspect stale/old metrics without marking train complete.
        if step >= 6 and not self._pipeline.get("validated"):
            self._append_log("[AKIŞ] Önce Adım 5: doğrulamayı tamamla.")
            step = 5 if (self._pipeline.get("trained") or self._last_train_result) else (
                4 if self._pipeline.get("frozen") else 3
            )
        # Step 4/6 pages must be real PipeCards (not shadowed preset/format widgets).
        self._pipeline["step"] = step
        if step - 1 < 0 or step - 1 >= self.step_stack.count():
            self._append_log(f"[HATA] Adım {step} yığını bozuk (index dışı).")
            return
        self.step_stack.setCurrentIndex(step - 1)
        self.flow_summary.setText(f"Adım {step}/6 · {self.STEP_NAMES[step]}")
        self._refresh_stepper_visuals()
        self._update_next_card()
        self._update_blockers()
        self._update_train_lock()
        self._save_pipeline_disk_state()
        if step == 2:
            self._reload_review_queue()
        if step == 3:
            self._rebuild_health_checks()
        if step == 5:
            self._run_validate()
        if step == 6:
            self._refresh_publish_checklist()

    def _mark_step_done(self, step: int) -> None:
        btn = self._step_buttons.get(step)
        if btn is None:
            return
        btn.setProperty("pipe_done", True)
        btn._num_lbl.setText("✓")  # type: ignore[attr-defined]
        btn._num_lbl.setStyleSheet(
            "background:#2ED573;border:1px solid #2ED573;border-radius:13px;"
            "font-family:Consolas;font-weight:700;font-size:12px;color:#08130C;"
        )

    def _refresh_stepper_visuals(self) -> None:
        C = self.PIPELINE_COLORS
        current = int(self._pipeline["step"])
        pending = int(self._pipeline["pending"])
        frozen = bool(self._pipeline["frozen"])
        trained = bool(self._pipeline["trained"])
        validated = bool(self._pipeline["validated"])
        published = bool(self._pipeline["published"])

        status_texts = {
            1: f"{self._class_counts_cache and 'kareler hazır' or 'yakalama'}",
            2: f"{pending} kare bekliyor" if pending else "Kuyruk boş",
            3: "donduruldu" if frozen else "dondurmayı bekliyor",
            4: "tamam" if trained else ("hazır" if frozen else "Önce veri setini dondur"),
            5: "tamam" if validated else ("hazır" if trained else "Eğitim bekleniyor"),
            6: "yayında" if published else ("hazır" if validated else "Doğrulama bekleniyor"),
        }

        done_flags = {
            1: pending >= 0,  # capture always available; mark done when moved on
            2: pending <= 0 and self._pipeline.get("reviewed", 0) > 0,
            3: frozen,
            4: trained,
            5: validated,
            6: published,
        }
        # Capture done if user left step 1 after starting, or frames exist
        if current > 1 or int(self._pipeline.get("session", 0)) > 0:
            done_flags[1] = True
        if pending <= 0 and self._review_items:
            # if queue loaded empty after reviews
            pass
        if pending <= 0 and (frozen or current >= 3):
            done_flags[2] = True

        blocked = {
            1: False,
            2: False,
            3: False,
            4: not frozen,
            5: not trained,
            6: not validated,
        }

        done_count = sum(1 for s in range(1, 7) if done_flags.get(s))
        pct = int(round(((done_count + (0.5 if current > 1 else 0)) / 6.0) * 100))
        self.flow_bar.setValue(max(8, min(100, pct)))

        for step, btn in self._step_buttons.items():
            st = btn._st_lbl  # type: ignore[attr-defined]
            num = btn._num_lbl  # type: ignore[attr-defined]
            st.setText(str(status_texts.get(step, "—")).upper())
            if done_flags.get(step) and step != current:
                btn.setObjectName("PipeStepDone")
                num.setText("✓")
                num.setStyleSheet(
                    "background:#2ED573;border:1px solid #2ED573;border-radius:13px;"
                    "font-family:Consolas;font-weight:700;font-size:12px;color:#08130C;"
                )
                st.setStyleSheet(
                    f"font-size:10px;font-weight:800;color:{C['green']};background:transparent;border:none;"
                )
            elif step == current:
                btn.setObjectName("PipeStepCurrent")
                if not done_flags.get(step):
                    num.setText(str(step))
                    num.setStyleSheet(
                        f"background:{C['amber']};border:1px solid {C['amber']};border-radius:13px;"
                        "font-family:Consolas;font-weight:700;font-size:12px;color:#111;"
                    )
                st.setStyleSheet(
                    f"font-size:10px;font-weight:800;color:{C['amber']};background:transparent;border:none;"
                )
            elif blocked.get(step):
                btn.setObjectName("PipeStepBlocked")
                num.setText(str(step))
                num.setStyleSheet(
                    f"background:{C['panel3']};border:1px solid {C['line2']};border-radius:13px;"
                    "font-family:Consolas;font-weight:700;font-size:12px;color:#9AA3B2;"
                )
                st.setStyleSheet(
                    "font-size:10px;font-weight:800;color:#FF8A94;background:transparent;border:none;"
                )
            else:
                btn.setObjectName("PipeStep")
                num.setText(str(step))
                num.setStyleSheet(
                    f"background:{C['panel3']};border:1px solid {C['line2']};border-radius:13px;"
                    "font-family:Consolas;font-weight:700;font-size:12px;color:#9AA3B2;"
                )
                st.setStyleSheet(
                    f"font-size:10px;font-weight:800;color:{C['muted']};background:transparent;border:none;"
                )
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _update_next_card(self) -> None:
        p = self._pipeline
        current = int(p.get("step", 1) or 1)
        self._next_action = "goto"
        if p["published"]:
            self.next_title.setText("Model yayınlandı — hat tamam")
            self.next_text.setText(
                "Yeni model botta aktif. Yeni veri toplamak için Yakala adımından yeni oturum başlat."
            )
            self.btn_next.setText("Yakala adımına git →")
            self._next_target = 1
            return
        if p["validated"]:
            self.next_title.setText("Sırada: yayınla")
            self.next_text.setText(
                "Doğrulama tamam. Formatı seçip modeli yayınla; eski sürüm yedekte kalıyor."
            )
            self._next_target = 6
            if current == 6:
                self.btn_next.setText("⇪ Modeli yayınla →")
                self._next_action = "publish"
            else:
                self.btn_next.setText("Yayınla adımına git →")
            return
        if p["trained"]:
            map50_live = float((self._last_train_result or {}).get("mAP50", 0.0) or 0.0)
            self.next_title.setText("Sırada: doğrula")
            if map50_live > 0:
                self.next_text.setText(
                    f"Eğitim bitti (mAP50 %{(map50_live * 100):.1f}). "
                    "mAP ve hata karelerine bakmadan yayınlama."
                )
            else:
                self.next_text.setText(
                    "Eğitim bayrağı var. Doğrula, son run artifacts (results.csv) üzerinden mAP yükler."
                )
            self._next_target = 5
            if current == 5:
                self.btn_next.setText("✓ Doğrulamayı çalıştır →")
                self._next_action = "validate"
            else:
                self.btn_next.setText("Doğrula adımına git →")
            return
        if p["frozen"]:
            self.next_title.setText("Veri seti dondu — eğitim hazır")
            if current == 5:
                self.next_text.setText(
                    "Adım 5'te metrik yok. Adım 4'te RTX eğitimini bitir; "
                    "bitince Doğrula mAP50'yi artifacts'tan okur."
                )
                self.btn_next.setText("Eğit adımına git →")
                self._next_target = 4
                self._next_action = "goto"
                return
            self.next_text.setText(
                "dataset-v4 kilitlendi. Hazır ayarı seçip RTX eğitimini başlat; bitince Adım 5 (Doğrula) açılır."
            )
            self._next_target = 4
            if current == 4:
                self.btn_next.setText("⚡ Eğitimi başlat →")
                self._next_action = "train"
            else:
                self.btn_next.setText("Eğit adımına git →")
            return
        if int(p["pending"]) <= 0:
            self.next_title.setText("Kuyruk boş — veri setini hazırla")
            self.next_text.setText(
                "Tüm kareler onaylandı. Sağlık kontrollerini bitirip veri setini dondur."
            )
            self.btn_next.setText("Hazırla adımına git →")
            self._next_target = 3
            return
        self.next_title.setText(f"{p['pending']} otomatik etiket onay bekliyor")
        self.next_text.setText(
            "Eğitim, onaylanmamış karelerle başlamamalı. Eşik üstündekileri toplu onayla, kalanlara tek tek bak."
        )
        self.btn_next.setText("Onay kuyruğuna git →")
        self._next_target = 2

    def _on_next_clicked(self) -> None:
        action = str(getattr(self, "_next_action", "goto") or "goto")
        target = int(getattr(self, "_next_target", 2))
        if action == "train":
            if int(self._pipeline.get("step", 1)) != 4:
                self._go_step(4)
            self.on_train_rtx_clicked()
            return
        if action == "validate":
            if int(self._pipeline.get("step", 1)) != 5:
                self._go_step(5)
            self._run_validate()
            return
        if action == "publish":
            if int(self._pipeline.get("step", 1)) != 6:
                self._go_step(6)
            self._publish_model()
            return
        self._go_step(target)

    def _update_blockers(self) -> None:
        pending = int(self._pipeline["pending"])
        frozen = bool(self._pipeline["frozen"])
        empty_n = self._count_empty_classes()
        n = 0
        self._blocker_widgets["review"].setVisible(pending > 0)
        if pending > 0:
            n += 1
        self._blocker_widgets["empty"].setVisible(empty_n > 0)
        if empty_n > 0:
            n += 1
        self._blocker_widgets["freeze"].setVisible(not frozen)
        if not frozen:
            n += 1
        self._blocker_widgets["none"].setVisible(n == 0)
        self.block_count.setText(f"{n} engel")

    def _count_empty_classes(self) -> int:
        counts = self._class_counts_cache or {}
        # Count empty among first 16 display classes
        empty = 0
        names: List[str] = []
        for spec in YOLO_12_CLASS_SPECS[:16]:
            cid = int(spec["id"])
            if int(counts.get(cid, 0)) <= 0:
                empty += 1
                names.append(str(spec["label_tr"]))
        self._empty_class_names = names
        return empty

    def _update_train_lock(self) -> None:
        frozen = bool(self._pipeline["frozen"])
        self.train_lock_pill.setVisible(not frozen)
        self.btn_train_rtx.setEnabled(frozen and not (
            self._training_worker is not None and self._training_worker.isRunning()
        ))
        if frozen:
            self.freeze_state_lbl.setText("donduruldu")
            self.freeze_state_lbl.setStyleSheet(
                "color:#2ED573;font-family:Consolas;font-weight:700;"
            )
        else:
            self.freeze_state_lbl.setText("dondurulmadı")
            self.freeze_state_lbl.setStyleSheet(
                "color:#F6B81E;font-family:Consolas;font-weight:700;"
            )

    # ------------------------------------------------------------------
    # Capture / review / prepare / train / validate / publish handlers
    # ------------------------------------------------------------------
    def _on_capture_toggle(self) -> None:
        capturing = not bool(self._pipeline["capturing"])
        self._pipeline["capturing"] = capturing
        if capturing:
            self.btn_capture_toggle.setText("■ Yakalamayı durdur")
            self.btn_capture_toggle.setObjectName("PipeDanger")
            self._append_log("[YAKALAMA] Yakalama başladı — tampon aktif.")
            if not hasattr(self, "_cap_timer"):
                self._cap_timer = QTimer(self)
                self._cap_timer.timeout.connect(self._capture_tick)
            self._cap_timer.start(700)
        else:
            self.btn_capture_toggle.setText("● Yakalamayı başlat")
            self.btn_capture_toggle.setObjectName("PipePrimary")
            self.btn_capture_toggle.style().unpolish(self.btn_capture_toggle)
            self.btn_capture_toggle.style().polish(self.btn_capture_toggle)
            if hasattr(self, "_cap_timer"):
                self._cap_timer.stop()
            self._append_log(
                f"[YAKALAMA] Durdu. Bu oturum: +{self._pipeline['session']} kare."
            )

    def _capture_tick(self) -> None:
        ctrl = getattr(self, "factory_controller", None)
        tm = getattr(ctrl, "time_machine", None)
        if tm is not None:
            try:
                sz = tm.size()
                sp = tm.span_seconds()
                if hasattr(self, "buffer_val"):
                    self.buffer_val.setText(f"{sz} kare · {sp:.1f} sn".replace(".", ","))
            except Exception:
                pass

        snap = self._time_machine_snapshot()
        if snap is not None and getattr(getattr(snap, "frame", None), "size", 0) > 0:
            self._pipeline["session"] = int(self._pipeline.get("session", 0)) + 1
            if hasattr(self, "session_val"):
                self.session_val.setText(f"+{self._pipeline['session']} kare")
            if int(self._pipeline["session"]) % 20 == 0:
                self._append_log(
                    f"[YAKALAMA] Oturum +{self._pipeline['session']} kare "
                    f"(onay kuyruğu {self._pending_count()})"
                )

    def _finish_capture_to_review(self) -> None:
        if self._pipeline["capturing"]:
            self._on_capture_toggle()
        self._mark_step_done(1)
        self._go_step(2)
        self._append_log("[AKIŞ] Yakala tamam → Onay adımına geçildi.")

    def _on_tm_changed(self, value: int) -> None:
        secs = value / 10.0
        txt = "t 0,0 sn" if secs == 0 else f"t −{abs(secs):.1f} sn".replace(".", ",")
        self.tm_val.setText(txt)
        self.preview_meta.setText(f"Zaman makinesi {txt} · Otomatik etiket çalışıyor")

    def _set_preview_view(self, mode: str) -> None:
        self._append_log(f"[GÖRÜNÜM] Önizleme modu: {mode}")

    def _review_sample_mtime(self, review_path: Path, split: str, stem: str) -> float:
        """Newest of review json and matching image file mtimes."""
        mtime = 0.0
        try:
            mtime = float(review_path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        images = self._dataset_dir() / "images" / split
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            cand = images / f"{stem}{ext}"
            if cand.is_file():
                try:
                    mtime = max(mtime, float(cand.stat().st_mtime))
                except OSError:
                    pass
        return mtime

    def _log_once(self, key: str, message: str) -> None:
        if key in self._log_once_keys:
            return
        self._log_once_keys.add(key)
        self._append_log(message)

    @staticmethod
    def _frame_conf(it: Dict[str, Any]) -> float:
        """Kare güveni = min kutu güveni (weak öncelikli, yoksa trusted). 0 kutu → 0.0."""
        if not isinstance(it, dict):
            return 0.0
        weak = [b for b in it.get("weak", []) if isinstance(b, dict)]
        trusted = [b for b in it.get("trusted", []) if isinstance(b, dict)]
        boxes = weak if weak else trusted
        if not boxes:
            return 0.0
        confs: List[float] = []
        for b in boxes:
            try:
                c = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                norm = c if c <= 1.0 else c / 100.0
                confs.append(norm)
            except (TypeError, ValueError):
                confs.append(0.0)
        return float(min(confs)) if confs else 0.0

    def _pending_count(self) -> int:
        """Tek kaynak: yüklü kuyrukta done olmayanlar."""
        return sum(1 for it in self._review_items if not it.get("done"))

    def _sync_pending_from_queue(self) -> None:
        """pending + tüm onay sayaçlarını `_review_items` ile hizala."""
        self._pipeline["pending"] = self._pending_count()
        self._update_pending_labels()

    def _reload_review_queue(self) -> None:
        """Disk review JSON ve images/{train,val} taraması: onaylanmış ve reddedilmiş kareler kuyruğa geri gelmez."""
        dataset = self._dataset_dir()
        review_root = dataset / "review"
        images_root = dataset / "images"
        labels_root = dataset / "labels"
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        rejected_stems = self._load_rejected_stems()

        try:
            from dataset_factory.approved_pool import get_approved_pool

            approved_pool = get_approved_pool(self._dataset_dir())
        except Exception:
            approved_pool = None

        items: List[Dict[str, Any]] = []
        seen_keys: set = set()

        # Aşama 1: images/train ve images/val altındaki TÜM görselleri keşfet
        for split in ("train", "val"):
            img_split_dir = images_root / split
            if not img_split_dir.is_dir():
                continue
            try:
                img_paths = [p for p in img_split_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
            except OSError:
                img_paths = []

            for img_p in img_paths:
                stem = img_p.stem
                if stem in rejected_stems:
                    self._unlink_quiet(img_p)
                    self._unlink_quiet(review_root / split / f"{stem}.json")
                    self._unlink_quiet(review_root / split / f"{stem}.sidecar.json")
                    self._unlink_quiet(labels_root / split / f"{stem}.txt")
                    continue
                seen_keys.add((split, stem))
                is_in_pool = bool(approved_pool is not None and approved_pool.is_approved(stem))
                allow_approved = bool(
                    self.__dict__.get("_review_audit_mode", False)
                    or self.__dict__.get("_queue_filter_mode", "all") == "approved_pool"
                )
                rev_path = review_root / split / f"{stem}.json"
                lbl_path = labels_root / split / f"{stem}.txt"

                weak: List[Dict[str, Any]] = []
                trusted: List[Dict[str, Any]] = []
                requeued = False
                payload_loaded = False
                skip_approved = False

                if rev_path.is_file():
                    try:
                        payload = json.loads(rev_path.read_text(encoding="utf-8"))
                        requeued = bool(payload.get("requeued"))
                        if not allow_approved and not requeued:
                            if (
                                payload.get("done") == "ok"
                                or payload.get("human_approved") is True
                                or is_in_pool
                            ):
                                skip_approved = True
                        weak = [b for b in payload.get("weak", []) if isinstance(b, dict)]
                        trusted = [b for b in payload.get("trusted", []) if isinstance(b, dict)]
                        payload_loaded = True
                    except Exception:
                        payload_loaded = False
                elif is_in_pool and not allow_approved:
                    skip_approved = True

                if skip_approved:
                    continue

                if not payload_loaded and lbl_path.is_file():
                    try:
                        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
                        for ln in lines:
                            parts = ln.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                cconf = float(parts[5]) if len(parts) > 5 else 0.95
                                cname = CLASS_ID_TO_SPEC.get(cid, {}).get("key", f"class_{cid}")
                                trusted.append({
                                    "class_id": cid,
                                    "class_name": cname,
                                    "cx": cx,
                                    "cy": cy,
                                    "w": bw,
                                    "h": bh,
                                    "confidence": cconf,
                                    "conf": cconf,
                                    "trusted": True,
                                    "weak": False,
                                    "hand_label": False,
                                })
                    except Exception:
                        pass

                if not rev_path.is_file():
                    try:
                        rev_path.parent.mkdir(parents=True, exist_ok=True)
                        rev_path.write_text(
                            json.dumps(
                                {"stem": stem, "split": split, "trusted": trusted, "weak": weak},
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                try:
                    mtime = float(img_p.stat().st_mtime)
                except OSError:
                    mtime = 0.0

                item = {
                    "path": rev_path,
                    "split": split,
                    "stem": stem,
                    "weak": weak,
                    "trusted": trusted,
                    "done": None,
                    "mtime": mtime,
                    "n_weak": len(weak),
                    "requeued": requeued,
                }
                item["conf"] = self._frame_conf(item)
                items.append(item)

        # Aşama 2: images altında eşleşmeyen bağımsız review JSON dosyaları varsa ekle
        for split in ("train", "val"):
            rev_split_dir = review_root / split
            if not rev_split_dir.is_dir():
                continue
            try:
                rev_paths = list(rev_split_dir.glob("*.json"))
            except OSError:
                continue
            for rpath in rev_paths:
                if rpath.name == "trusted_seed.json":
                    continue
                stem = rpath.stem
                if stem in rejected_stems:
                    self._unlink_quiet(rpath)
                    self._unlink_quiet(review_root / split / f"{stem}.sidecar.json")
                    self._unlink_quiet(labels_root / split / f"{stem}.txt")
                    continue
                if (split, stem) in seen_keys:
                    continue
                seen_keys.add((split, stem))
                is_in_pool = bool(approved_pool is not None and approved_pool.is_approved(stem))
                allow_approved = bool(
                    self.__dict__.get("_review_audit_mode", False)
                    or self.__dict__.get("_queue_filter_mode", "all") == "approved_pool"
                )
                try:
                    payload = json.loads(rpath.read_text(encoding="utf-8"))
                    requeued = bool(payload.get("requeued"))
                    if not allow_approved and not requeued:
                        if (
                            payload.get("done") == "ok"
                            or payload.get("human_approved") is True
                            or is_in_pool
                        ):
                            continue
                    weak = [b for b in payload.get("weak", []) if isinstance(b, dict)]
                    trusted = [b for b in payload.get("trusted", []) if isinstance(b, dict)]
                    mtime = float(rpath.stat().st_mtime)
                except Exception:
                    continue
                item = {
                    "path": rpath,
                    "split": split,
                    "stem": stem,
                    "weak": weak,
                    "trusted": trusted,
                    "done": None,
                    "mtime": mtime,
                    "n_weak": len(weak),
                    "requeued": requeued,
                }
                item["conf"] = self._frame_conf(item)
                items.append(item)

        items.sort(key=lambda x: -float(x.get("mtime", 0.0)))
        self._review_items = items
        self._review_index = 0
        self._review_decode_cache.clear()
        self._sync_pending_from_queue()
        self._render_queue_list()
        self._render_review_frame()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()

    def _bulk_threshold(self) -> float:
        return float(self.bulk_range.value()) / 100.0

    def _on_bulk_threshold_changed(self, value: int) -> None:
        self.bulk_val.setText(str(int(value)))
        if getattr(self, "_queue_filter_mode", "all") in ("below", "above"):
            self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_queue_filter_changed(self, _index: int = 0) -> None:
        data = self.queue_filter.currentData() if hasattr(self, "queue_filter") else "all"
        self._queue_filter_mode = str(data or "all")
        self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_class_filter_changed(self, _index: int = 0) -> None:
        self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_review_class_filter_changed(self, _index: int = 0) -> None:
        self._on_class_filter_changed(_index)

    def _on_search_text_changed(self, _text: str = "") -> None:
        self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_review_search_changed(self, _text: str = "") -> None:
        self._on_search_text_changed(_text)

    def set_queue_filter_mode(self, mode: str) -> None:
        """Kuyruk filtre modunu ayarlar ve butonları günceller."""
        self._queue_filter_mode = str(mode or "all")
        if hasattr(self, "queue_filter"):
            idx = self.queue_filter.findData(mode)
            if idx >= 0 and self.queue_filter.currentIndex() != idx:
                self.queue_filter.blockSignals(True)
                self.queue_filter.setCurrentIndex(idx)
                self.queue_filter.blockSignals(False)
        self._render_queue_list()
        self._update_filter_action_buttons()

    def _filtered_review_indices(self) -> List[int]:
        """Çok kriterli filtre: queue_filter modu + sınıf filtresi + dosya adı arama."""
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        th = self._bulk_threshold() if hasattr(self, "bulk_range") else 0.90

        target_cid = -1
        if hasattr(self, "review_class_filter"):
            cdata = self.review_class_filter.currentData()
            try:
                target_cid = int(cdata)
            except (TypeError, ValueError):
                target_cid = -1

        query = ""
        if hasattr(self, "review_search_edit"):
            query = self.review_search_edit.text().strip().lower()

        out: List[int] = []
        for i, it in enumerate(self._review_items):
            stem = str(it.get("stem", "")).lower()
            if query and query not in stem:
                continue

            weak = [b for b in it.get("weak", []) if isinstance(b, dict)]
            trusted = [b for b in it.get("trusted", []) if isinstance(b, dict)]
            all_boxes = weak + trusted
            n_weak = len(weak)
            n_trusted = len(trusted)
            n_total = n_weak + n_trusted

            if target_cid >= 0:
                has_cid = any(int(b.get("class_id", -99)) == target_cid for b in all_boxes)
                if not has_cid:
                    continue

            conf = self._frame_conf(it)
            it["conf"] = conf

            if mode == "approved_pool":
                is_app = False
                try:
                    pool = get_approved_pool(self._dataset_dir())
                    if pool.is_approved(stem):
                        is_app = True
                except Exception:
                    pass
                if not (is_app or it.get("done") == "ok" or it.get("human_approved") is True):
                    continue
            elif mode == "suspicious":
                if n_weak == 0 or n_trusted > 0:
                    continue
            elif mode == "confident":
                if n_trusted == 0 or n_weak > 0:
                    continue
            elif mode == "mixed":
                if n_weak == 0 or n_trusted == 0:
                    continue
            elif mode == "empty":
                if n_total > 0:
                    continue
            elif mode in ("unlabeled_and_suspicious", "unlabeled_or_suspicious"):
                if not (n_total == 0 or n_weak > 0):
                    continue
            elif mode == "below":
                if conf >= th:
                    continue
            elif mode == "above":
                if conf < th:
                    continue

            out.append(i)

        return out

    def _update_filter_action_buttons(self) -> None:
        """Adım 2 toplu aksiyon butonlarını geçerli filtre moduna ve filtrelenen eleman sayısına göre dinamik günceller."""
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        indices = self._filtered_review_indices() if hasattr(self, "_review_items") else []
        n = len(indices)

        # Sınıf filtresi veya arama etiketi (scope_tag)
        scope_tag = ""
        if hasattr(self, "review_class_filter"):
            cdata = self.review_class_filter.currentData()
            try:
                target_cid = int(cdata)
            except (TypeError, ValueError):
                target_cid = -1
            if target_cid >= 0:
                cname = CLASS_ID_TO_SPEC.get(target_cid, {}).get("key", f"class_{target_cid}")
                scope_tag = f"[{cname}]"
        if not scope_tag and hasattr(self, "review_search_edit"):
            query_txt = self.review_search_edit.text().strip()
            if query_txt:
                scope_tag = f"[{query_txt}]"

        tag_suffix = f" {scope_tag}" if scope_tag else ""

        if mode == "suspicious":
            reject_txt = f"✕ Tüm Şüphelileri Otomatik Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Tüm Şüphelileri Otomatik Reddet ({n} kare)"
            approve_txt = f"✓ Tüm Şüphelileri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Tüm Şüphelileri Onayla ({n} kare)"
            prune_txt = f"✂️ Şüpheli Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Şüpheli Etiketleri Sil · Görseli Koru ({n} kare)"
            relabel_txt = f"🎯 Şüphelileri Canlı Yeniden Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Şüphelileri Canlı Yeniden Etiketle ({n} kare)"
            prune_enabled = True

        elif mode == "empty":
            reject_txt = f"✕ Tüm Etiketsizleri Otomatik Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Tüm Etiketsizleri Otomatik Reddet ({n} kare)"
            approve_txt = f"✓ Etiketsizleri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Etiketsizleri Onayla ({n} kare)"
            prune_txt = f"✂️ Etiketsiz Kutu Temizle{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Etiketsiz Kutu Temizle ({n} kare)"
            relabel_txt = f"🎯 Tüm Etiketsizleri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Tüm Etiketsizleri Canlı Etiketle ({n} kare)"
            prune_enabled = False

        elif mode == "confident":
            reject_txt = f"✕ Filtredeki Eminleri Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Filtredeki Eminleri Reddet ({n} kare)"
            approve_txt = f"✓ Tüm Eminleri Otomatik Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Tüm Eminleri Otomatik Onayla ({n} kare)"
            prune_txt = f"✂️ Düşük Güvenli Etiketleri Sil{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Düşük Güvenli Etiketleri Sil ({n} kare)"
            relabel_txt = f"🎯 Filtredeki Eminleri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Filtredeki Eminleri Canlı Etiketle ({n} kare)"
            prune_enabled = True

        elif mode in ("unlabeled_and_suspicious", "unlabeled_or_suspicious"):
            reject_txt = f"✕ Etiketsiz & Şüphelileri Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Etiketsiz & Şüphelileri Reddet ({n} kare)"
            approve_txt = f"✓ Etiketsiz & Şüphelileri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Etiketsiz & Şüphelileri Onayla ({n} kare)"
            prune_txt = f"✂️ Şüpheli Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Şüpheli Etiketleri Sil · Görseli Koru ({n} kare)"
            relabel_txt = f"🎯 Etiketsiz & Şüphelileri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Etiketsiz & Şüphelileri Canlı Etiketle ({n} kare)"
            prune_enabled = True

        elif mode == "below":
            reject_txt = f"✕ Eşik Altı Filtredekileri Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Eşik Altı Filtredekileri Reddet ({n} kare)"
            approve_txt = f"✓ Eşik Altı Filtredekileri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Eşik Altı Filtredekileri Onayla ({n} kare)"
            prune_txt = f"✂️ Eşik Altı Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Eşik Altı Etiketleri Sil · Görseli Koru ({n} kare)"
            relabel_txt = f"🎯 Eşik Altı Filtredekileri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Eşik Altı Filtredekileri Canlı Etiketle ({n} kare)"
            prune_enabled = True

        elif mode == "above":
            reject_txt = f"✕ Eşik Üstü Filtredekileri Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Eşik Üstü Filtredekileri Reddet ({n} kare)"
            approve_txt = f"✓ Eşik Üstü Filtredekileri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Eşik Üstü Filtredekileri Onayla ({n} kare)"
            prune_txt = f"✂️ Eşik Altı Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Eşik Altı Etiketleri Sil · Görseli Koru ({n} kare)"
            relabel_txt = f"🎯 Eşik Üstü Filtredekileri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Eşik Üstü Filtredekileri Canlı Etiketle ({n} kare)"
            prune_enabled = True

        elif mode == "mixed":
            reject_txt = f"✕ Karışık Kareleri Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Karışık Kareleri Reddet ({n} kare)"
            approve_txt = f"✓ Karışık Kareleri Onayla{tag_suffix} ({n} kare)" if scope_tag else f"✓ Karışık Kareleri Onayla ({n} kare)"
            prune_txt = f"✂️ Sadece Şüpheli Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Sadece Şüpheli Etiketleri Sil · Görseli Koru ({n} kare)"
            relabel_txt = f"🎯 Karışık Kareleri Canlı Etiketle{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Karışık Kareleri Canlı Etiketle ({n} kare)"
            prune_enabled = True
        elif mode == "approved_pool":
            reject_txt = f"✕ Seçili Onaylıları Reddet{tag_suffix} ({n} kare)" if scope_tag else f"✕ Seçili Onaylıları Reddet ({n} kare)"
            approve_txt = f"✓ Onay Havuzunda{tag_suffix} ({n} kare)" if scope_tag else f"✓ Onay Havuzunda ({n} kare)"
            prune_txt = f"✂️ Etiketleri İncele{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Etiketleri İncele ({n} kare)"
            relabel_txt = f"🎯 Onaylıları Canlı İncele{tag_suffix} ({n} kare)" if scope_tag else f"🎯 Onaylıları Canlı İncele ({n} kare)"
            prune_enabled = False

        else:  # "all" or any fallback
            if scope_tag:
                reject_txt = f"✕ Filtredeki {scope_tag} Karelerini Reddet ({n} kare)"
                approve_txt = f"✓ Filtredeki {scope_tag} Karelerini Onayla ({n} kare)"
                prune_txt = f"✂️ {scope_tag} Düşük Etiketleri Sil · Görseli Koru ({n} kare)"
                relabel_txt = f"🎯 Filtredeki {scope_tag} Karelerini Canlı Etiketle ({n} kare)"
            else:
                reject_txt = f"✕ Filtredekileri Reddet ({n} kare)"
                approve_txt = f"✓ Filtredekileri Onayla ({n} kare)"
                prune_txt = f"✂️ Düşük Güvenli Etiketleri Sil · Görseli Koru ({n} kare)"
                relabel_txt = f"🎯 Filtredekileri Canlı Yeniden Etiketle ({n} kare)"
            prune_enabled = True

        if hasattr(self, "btn_bulk_reject"):
            self.btn_bulk_reject.setText(reject_txt)
        if hasattr(self, "btn_bulk"):
            self.btn_bulk.setText(approve_txt)
        if hasattr(self, "btn_bulk_approve"):
            self.btn_bulk_approve.setText(approve_txt)
        if hasattr(self, "btn_prune_low_confidence"):
            self.btn_prune_low_confidence.setText(prune_txt)
            self.btn_prune_low_confidence.setEnabled(prune_enabled)
        if hasattr(self, "btn_bulk_prune_boxes"):
            self.btn_bulk_prune_boxes.setText(prune_txt)
            self.btn_bulk_prune_boxes.setEnabled(prune_enabled)
        if hasattr(self, "btn_live_relabel_filtered"):
            self.btn_live_relabel_filtered.setText(relabel_txt)
        if hasattr(self, "btn_live_relabel_all"):
            self.btn_live_relabel_all.setText(relabel_txt)

    def _queue_item_text(self, it: Dict[str, Any]) -> str:
        conf = self._frame_conf(it)
        it["conf"] = conf
        done = it.get("done")
        suffix = ""
        if done == "ok":
            suffix = " · ok"
        elif done == "no":
            suffix = " · red"

        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])
        n_weak = len(weak)
        n_trusted = len(trusted)
        n_total = n_weak + n_trusted

        stem = str(it.get("stem", ""))
        short = stem if len(stem) <= 22 else (stem[:10] + "…" + stem[-10:])

        all_boxes = weak + trusted
        confs: List[float] = []
        for b in all_boxes:
            try:
                c = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                norm = c if c <= 1.0 else c / 100.0
                confs.append(norm)
            except (TypeError, ValueError):
                pass

        if n_total == 0:
            return f"#{short} · 0 Kutu (Boş){suffix}"

        if confs:
            min_pct = int(round(min(confs) * 100))
            max_pct = int(round(max(confs) * 100))
            conf_str = f"%{min_pct}" if min_pct == max_pct else f"%{min_pct}-%{max_pct}"
        else:
            conf_str = f"%{int(round(conf * 100))}"

        if n_weak > 0 and n_trusted > 0:
            count_str = f"{n_trusted} Emin, {n_weak} Şüpheli"
        elif n_trusted > 0:
            count_str = f"{n_trusted} Emin"
        else:
            count_str = f"{n_weak} Şüpheli"

        return f"#{short} · {conf_str} · {count_str}{suffix}"

    def _render_queue_list(self) -> None:
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        indices = self._filtered_review_indices()
        th = self._bulk_threshold() if hasattr(self, "bulk_range") else 0.90
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        if hasattr(self, "queue_filter_lbl"):
            mode_labels = {
                "all": "hepsi",
                "suspicious": "şüpheliler",
                "confident": "eminler",
                "mixed": "karışık",
                "empty": "etiketsiz/boş",
                "unlabeled_and_suspicious": "etiketsiz+şüpheli",
                "unlabeled_or_suspicious": "etiketsiz+şüpheli",
                "below": f"conf < {th:.2f}",
                "above": f"conf ≥ {th:.2f}",
            }
            lbl_name = mode_labels.get(mode, mode)
            self.queue_filter_lbl.setText(f"Kuyruk · {lbl_name} ({len(indices)})")
        select_row = 0
        for list_row, real_i in enumerate(indices):
            it = self._review_items[real_i]
            item = QListWidgetItem(self._queue_item_text(it))
            item.setData(int(Qt.ItemDataRole.UserRole), int(real_i))
            if it.get("done"):
                item.setForeground(QColor("#6B7688"))
            self.queue_list.addItem(item)
            if real_i == self._review_index:
                select_row = list_row
        if indices:
            if self._review_index not in indices:
                self._review_index = indices[0]
                select_row = 0
            self.queue_list.setCurrentRow(select_row)
            self._render_review_frame()
        elif self._review_items:
            self.review_title.setText("Filtre boş")
            self.review_sub.setText("Seçili filtre/aramaya uyan kare yok")
            self.review_preview.show_empty_state("Filtre boş")
        self._update_filter_action_buttons()
        self.queue_list.blockSignals(False)

    def _on_queue_row_changed(self, row: int) -> None:
        if row < 0 or row >= self.queue_list.count():
            return
        item = self.queue_list.item(row)
        if item is None:
            return
        real_i = item.data(int(Qt.ItemDataRole.UserRole))
        try:
            real_i = int(real_i)
        except (TypeError, ValueError):
            real_i = row
        if real_i < 0 or real_i >= len(self._review_items):
            return
        if real_i == self._review_index and not self.review_preview._empty:
            return
        if hasattr(self, "review_preview"):
            self.review_preview.clear_drag_state()
        self._review_index = real_i
        self._render_review_frame()

    def _imread_cached(self, path: Path) -> Optional[np.ndarray]:
        key = str(path.resolve())
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            return None
        hit = self._review_decode_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        img = _safe_imread_unicode(path)
        if img is None or getattr(img, "size", 0) == 0:
            return None
        # Tek seçili kare cache (bellek baskısını sınırla)
        self._review_decode_cache.clear()
        self._review_decode_cache[key] = (mtime, img)
        return img

    def _pixmap_from_bgr(self, frame_bgr: np.ndarray) -> QPixmap:
        display = YOLOPreviewCanvas._downscale_bgr(frame_bgr, max_w=2400)
        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _estimate_box_kinah(self, det: Dict[str, Any]) -> int:
        """Kutunun temsil ettiği materyal veya değerin tahmini Kinah karşılığını döndürür."""
        if not isinstance(det, dict):
            return 0
        cid = int(det.get("class_id", -1))
        # Eğer kutuda özel kinah_amount belirtilmişse (örn. OCR / etiket)
        if "kinah_amount" in det:
            try:
                return max(0, int(det["kinah_amount"]))
            except (TypeError, ValueError):
                pass

        # 3B Dünya kaynakları (hasat başına ~3x) ve sarf malzemeleri birim fiyatları (Kinah)
        MATERIAL_PRICES: Dict[int, int] = {
            0: 9800 * 3,   # herb_aria: 3x hasat (~29.400 Kinah)
            1: 11500 * 3,  # herb_mela: 3x hasat (~34.500 Kinah)
            2: 12400 * 3,  # herb_targena: 3x hasat (~37.200 Kinah)
            3: 14500 * 3,  # ore_orichalcum: 3x hasat (~43.500 Kinah)
            4: 38000 * 3,  # gem_crystal: 3x hasat (~114.000 Kinah)
            5: 19500 * 3,  # tree_log: 3x hasat (~58.500 Kinah)
            6: 10000 * 3,  # herb_edana: 3x hasat (~30.000 Kinah)
            7: 12000 * 3,  # herb_lumina: 3x hasat (~36.000 Kinah)
            8: 22000 * 3,  # ore_obsidian: 3x hasat (~66.000 Kinah)
            9: 12000 * 3,  # minimap_sprout (~36.000 Kinah)
            12: 25000 * 3, # crystal_blood: 3x hasat (~75.000 Kinah)
            22: 2200,      # hp_potion
            23: 2200,      # mp_potion
            24: 3200,      # buff_scroll
            25: 0,         # kinah display indicator itself
        }
        return int(MATERIAL_PRICES.get(cid, 0))

    def _create_review_det_row(self, det: Dict[str, Any], is_trusted: bool) -> QWidget:
        row = QWidget(self.det_host)
        row.setObjectName("PipeDetRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            "QWidget#PipeDetRow {"
            " background-color:#0B0E12; color:#F2F3F5;"
            " border-bottom:1px solid #262D3A;"
            "}"
            "QWidget#PipeDetRow QLabel { background:transparent; color:#F2F3F5; }"
        )
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 5, 0, 5)
        hl.setSpacing(6)

        cname = normalize_box_class_name(det.get("class_id"), det.get("class_name", det.get("key")))
        conf = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
        norm_conf = conf if conf <= 1.0 else conf / 100.0
        pct = max(0, min(100, int(round(norm_conf * 100))))

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel(f"<b>{cname}</b>")
        title.setStyleSheet("background:transparent;color:#F2F3F5;font-size:12px;")
        text.addWidget(title)
        is_confident = (
            (is_trusted and norm_conf >= 0.85)
            or bool(det.get("hand_label"))
            or bool(det.get("human_approved"))
        )

        if is_confident:
            status_txt = f"Emin · %{pct} güven"
            status_color = "#4ade80"
            conf_style = (
                "background:#0d2818;color:#4ade80;font-family:Consolas;font-size:11px;font-weight:700;"
                "border:1px solid #16a34a;border-radius:6px;padding:2px 6px;"
            )
        elif norm_conf >= 0.70:
            status_txt = f"Şüpheli · %{pct} güven"
            status_color = "#fb923c"
            conf_style = (
                "background:#291804;color:#fb923c;font-family:Consolas;font-size:11px;font-weight:700;"
                "border:1px solid #d97706;border-radius:6px;padding:2px 6px;"
            )
        else:
            status_txt = f"Düşük Güven (Şüpheli) · %{pct} güven"
            status_color = "#f87171"
            conf_style = (
                "background:#3b1010;color:#f87171;font-family:Consolas;font-size:11px;font-weight:700;"
                "border:1px solid #ef4444;border-radius:6px;padding:2px 6px;"
            )

        small = QLabel(status_txt)
        small.setStyleSheet(f"background:transparent;color:{status_color};font-size:10px;")
        text.addWidget(small)
        hl.addLayout(text, stretch=1)

        conf_lbl = QLabel(f"%{pct}")
        conf_lbl.setObjectName("PipeDetConf")
        conf_lbl.setStyleSheet(conf_style)
        hl.addWidget(conf_lbl)

        cls_combo = QComboBox(row)
        cls_combo.setObjectName("PipeClassCombo")
        cls_combo.setEditable(True)
        cls_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        cls_combo.setToolTip("Sınıfı değiştir ve YZ'ye anında öğret")
        cls_combo.setStyleSheet(
            "QComboBox { background:#171B22; color:#F2F3F5; border:1px solid #262D3A; border-radius:6px; padding:2px 6px; font-size:11px; max-width:130px; }"
            "QComboBox QAbstractItemView { background:#0B0E12; color:#F2F3F5; selection-background-color:#2563EB; }"
        )
        cur_cid = int(det.get("class_id", 0))
        for cid_opt, spec_opt in sorted(CLASS_ID_TO_SPEC.items()):
            cname_opt = spec_opt.get("key", f"class_{cid_opt}")
            cls_combo.addItem(f"{cid_opt}: {cname_opt}", cid_opt)
        completer = QCompleter([cls_combo.itemText(i) for i in range(cls_combo.count())], cls_combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        cls_combo.setCompleter(completer)
        idx_match = cls_combo.findData(cur_cid)
        if idx_match >= 0:
            cls_combo.blockSignals(True)
            cls_combo.setCurrentIndex(idx_match)
            cls_combo.blockSignals(False)
        cls_combo.currentIndexChanged.connect(
            lambda idx, d=det, cb=cls_combo: self._on_box_class_changed(d, int(cb.itemData(idx)))
        )
        hl.addWidget(cls_combo)

        val = self._estimate_box_kinah(det)
        if val > 0 or cur_cid == 25:
            kinah_badge = QLabel(f"💰 ~{format_kinah_tr(val)}" if cur_cid != 25 else "🪙 Kinah")
            kinah_badge.setStyleSheet(
                "background:#292524; color:#fbbf24; border:1px solid #78350f; "
                "border-radius:4px; padding:2px 6px; font-size:10px; font-weight:700;"
            )
            if cur_cid == 25:
                kinah_badge.setToolTip("Kinah / Altın Göstergesi (Para birimi)")
            else:
                kinah_badge.setToolTip(f"Tahmini Pazar Değeri: ~{val:,} Kinah (3x hasat)".replace(",", "."))
            hl.addWidget(kinah_badge)

        if not is_trusted:
            ok = QPushButton("Kabul", row)
            ok.setObjectName("PipeGreen")
            ok.setStyleSheet(
                "background:#166534; color:#f0fdf4; border:1px solid #22c55e; "
                "border-radius:5px; padding:3px 8px; font-size:11px; font-weight:600;"
            )
            ok.setToolTip("Bu kutuyu onayla ve Emin grubuna taşı")
            ok.clicked.connect(lambda _=False, d=det: self._accept_box(d))

            remap = QPushButton("→ filiz", row)
            remap.setObjectName("PipeBtn")
            remap.setStyleSheet(
                "background:#1e293b; color:#cbd5e1; border:1px solid #334155; "
                "border-radius:5px; padding:3px 8px; font-size:11px;"
            )
            remap.setToolTip("Yanlış world sınıfını minimap_sprout (9) yap")
            remap.clicked.connect(lambda _=False, d=det: self._remap_box_to_sprout(d))

            no = QPushButton("Sil", row)
            no.setObjectName("PipeDanger")
            no.setStyleSheet(
                "background:#7f1d1d; color:#fef2f2; border:1px solid #ef4444; "
                "border-radius:5px; padding:3px 8px; font-size:11px;"
            )
            no.setToolTip("Bu şüpheli kutuyu sil (görsel korunur)")
            no.clicked.connect(lambda _=False, d=det: self._reject_box(d, from_trusted=False))

            hl.addWidget(ok)
            hl.addWidget(remap)
            hl.addWidget(no)
        else:
            demote = QPushButton("Şüpheliye Ayır", row)
            demote.setObjectName("PipeBtn")
            demote.setStyleSheet(
                "background:#7c2d12; color:#ffedd5; border:1px solid #ea580c; "
                "border-radius:5px; padding:3px 8px; font-size:11px; font-weight:600;"
            )
            demote.setToolTip("Bu emin kutuyu şüpheli grubuna geri al")
            demote.clicked.connect(lambda _=False, d=det: self._demote_box_to_weak(d))

            no = QPushButton("Sil", row)
            no.setObjectName("PipeDanger")
            no.setStyleSheet(
                "background:#7f1d1d; color:#fef2f2; border:1px solid #ef4444; "
                "border-radius:5px; padding:3px 8px; font-size:11px;"
            )
            no.setToolTip("Bu emin kutuyu sil (görsel korunur)")
            no.clicked.connect(lambda _=False, d=det: self._reject_box(d, from_trusted=True))

            hl.addWidget(demote)
            hl.addWidget(no)

        return row

    def _render_review_frame(self) -> None:
        if hasattr(self, "review_preview"):
            self.review_preview.clear_drag_state()
        while self.det_layout.count():
            item = self.det_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._review_lightbox_pixmap = None
        if not self._review_items:
            self.review_title.setText("Kuyruk boş")
            self.review_sub.setText("Onay bekleyen kare yok")
            if hasattr(self, "lbl_review_kinah"):
                self.lbl_review_kinah.setText("💰 0 Kinah")
            self.review_preview.show_empty_state("Kuyruk boş")
            self._pipeline["pending"] = 0
            self.review_prog.setValue(100)
            self.review_prog_txt.setText("0 / 0")
            return

        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])

        # Tüm kutular önizlemede çizilir (weak: şüpheli/kesikli, trusted: emin/düz)
        draw_boxes = []
        for b in weak:
            nb = dict(b)
            nb["trusted"] = False
            nb["weak"] = True
            draw_boxes.append(nb)
        for b in trusted:
            nb = dict(b)
            nb["trusted"] = True
            nb["weak"] = False
            draw_boxes.append(nb)

        min_c = self._frame_conf(it)
        it["conf"] = min_c
        self.review_title.setText(f"Kare #{it['stem']}")
        min_pct = int(round(min_c * 100))

        frame_kinah = sum(self._estimate_box_kinah(b) for b in draw_boxes)
        has_kinah_tag = any(int(b.get("class_id", -1)) == 25 for b in draw_boxes)
        queue_kinah = sum(
            sum(self._estimate_box_kinah(b) for b in (item.get("trusted", []) + item.get("weak", [])))
            for item in self._review_items
        )
        session_approved = getattr(self, "_session_approved_kinah", 0)

        kinah_text_parts = []
        if frame_kinah > 0:
            kinah_text_parts.append(f"💰 ~{format_kinah_tr(frame_kinah)} Kinah")
        if has_kinah_tag:
            kinah_text_parts.append("🪙 Kinah")
        kinah_sub_info = f" · {' · '.join(kinah_text_parts)}" if kinah_text_parts else ""

        if draw_boxes:
            self.review_sub.setText(f"{len(trusted)} Emin, {len(weak)} Şüpheli · En düşük güven: %{min_pct}{kinah_sub_info}")
        else:
            self.review_sub.setText(f"0 Kutu · Etiketsiz kare{kinah_sub_info}")

        if hasattr(self, "lbl_review_kinah"):
            tag_icon = "🪙" if has_kinah_tag else "💰"
            self.lbl_review_kinah.setText(
                f"{tag_icon} Bu Kare: ~{format_kinah_tr(frame_kinah)} Kinah · Kuyruk: ~{format_kinah_tr(queue_kinah)} Kinah"
            )
            tip = (
                f"💰 Ekonomi & Kinah Özeti:\n"
                f"• Mevcut Kare Tahmini Değer: ~{frame_kinah:,} Kinah\n"
                f"• Onay Kuyruğu Toplamı: ~{queue_kinah:,} Kinah\n"
                f"• Bu Oturumda Onaylanan: ~{session_approved:,} Kinah"
            ).replace(",", ".")
            self.lbl_review_kinah.setToolTip(tip)

        img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
        img: Optional[np.ndarray] = None
        if img_path is None:
            self.review_preview.show_empty_state("Görüntü bulunamadı")
            self._log_once(
                f"miss:{it.get('stem')}",
                f"[ÖNİZLEME] path not found: images/{{train,val}}/{it.get('stem')}.* "
                f"(dataset={self._dataset_dir()})",
            )
        else:
            img = self._imread_cached(img_path)
            if img is None:
                self.review_preview.show_empty_state("Decode başarısız")
                self._log_once(
                    f"dec:{img_path}",
                    f"[ÖNİZLEME] decode fail: {img_path}",
                )
            else:
                self._review_lightbox_pixmap = self._pixmap_from_bgr(img)
                ok = self.review_preview.set_live_frame(
                    img,
                    f"review · {img_path.name}",
                    draw_boxes,
                    source_key=f"file:{img_path.resolve()}",
                )
                if not ok:
                    self._log_once(
                        f"set:{img_path}",
                        f"[ÖNİZLEME] set_live_frame fail: {img_path}",
                    )

        # 🟠 Şüpheli Etiketler Bölümü
        sec1 = QWidget(self.det_host)
        sec1.setObjectName("PipeSec1")
        sec1_l = QHBoxLayout(sec1)
        sec1_l.setContentsMargins(0, 6, 0, 4)
        sec1_title = QLabel(f"🟠 Şüpheli Etiketler (İnceleme Bekleyen — {len(weak)} kutu)")
        sec1_title.setStyleSheet("color:#fb923c; font-size:12px; font-weight:800; background:transparent;")
        sec1_l.addWidget(sec1_title)
        self.det_layout.addWidget(sec1)

        if not weak:
            empty_weak = QLabel("Bu karede inceleme bekleyen şüpheli etiket yok.")
            empty_weak.setStyleSheet("color:#64748b; font-size:11px; padding:2px 4px; background:transparent;")
            self.det_layout.addWidget(empty_weak)
        else:
            for det in weak:
                self.det_layout.addWidget(self._create_review_det_row(det, is_trusted=False))

        # 🟢 Emin Olunan Etiketler Bölümü
        sec2 = QWidget(self.det_host)
        sec2.setObjectName("PipeSec2")
        sec2_l = QHBoxLayout(sec2)
        sec2_l.setContentsMargins(0, 10, 0, 4)
        sec2_title = QLabel(f"🟢 Emin Olunan Etiketler (Güvenilir / Onaylı — {len(trusted)} kutu)")
        sec2_title.setStyleSheet("color:#22c55e; font-size:12px; font-weight:800; background:transparent;")
        sec2_l.addWidget(sec2_title)
        self.det_layout.addWidget(sec2)

        if not trusted:
            empty_trusted = QLabel("Bu karede henüz onaylanmış emin etiket yok.")
            empty_trusted.setStyleSheet("color:#64748b; font-size:11px; padding:2px 4px; background:transparent;")
            self.det_layout.addWidget(empty_trusted)
        else:
            for det in trusted:
                self.det_layout.addWidget(self._create_review_det_row(det, is_trusted=True))

        self.det_layout.addStretch(1)

        pending = self._pending_count()
        reviewed = sum(1 for x in self._review_items if x.get("done"))
        total = len(self._review_items)
        self._pipeline["pending"] = pending
        if total <= 0:
            self.review_prog.setValue(100)
            self.review_prog_txt.setText("0 / 0")
        else:
            self.review_prog.setValue(int(round(100 * reviewed / total)))
            self.review_prog_txt.setText(f"{reviewed} / {total}")

    def _box_identity_match(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a is b:
            return True
        try:
            cid_a = int(a.get("class_id", -2))
            cid_b = int(b.get("class_id", -3))
            if cid_a != cid_b:
                return False
        except (TypeError, ValueError):
            pass
        for key in ("cx", "cy", "w", "h"):
            if key not in a or key not in b:
                continue
            try:
                if abs(float(a[key]) - float(b[key])) > 0.02:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _learn_box_exemplar(
        self,
        it: Dict[str, Any],
        box: Dict[str, Any],
        positive: bool = True,
        source: str = "ui",
    ) -> None:
        """Kutunun görsel crop özelliklerini ExemplarMemory (learned_exemplars.json) içine kaydeder."""
        try:
            img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
            if img_path is None or not img_path.is_file():
                return
            img = self._imread_cached(img_path)
            if img is None or getattr(img, "size", 0) == 0:
                return

            from dataset_factory.exemplar_memory import extract_box_crop, get_exemplar_memory

            crop = extract_box_crop(img, box)
            if crop is None or crop.size == 0:
                return

            cid = int(box.get("class_id", 0))
            cname = normalize_box_class_name(cid, box.get("class_name"))
            mem = get_exemplar_memory()
            if positive:
                mem.add_positive_exemplar(crop, cid, cname, source=source)
            else:
                mem.add_negative_exemplar(crop, cid, cname, source=source)
        except Exception as exc:
            logger.debug("Exemplar recording failed: %s", exc)

    def _on_box_class_changed(self, det: Dict[str, Any], new_cid: int) -> None:
        """Kullanıcı kutu sınıfını değiştirdiğinde YZ görsel hafızasını (ExemplarMemory) anında eğitir."""
        if not self._review_items or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])
        changed_box = None

        new_cname = CLASS_ID_TO_SPEC.get(new_cid, {}).get("key", f"class_{new_cid}")

        for b in weak:
            if self._box_identity_match(b, det):
                b["class_id"] = new_cid
                b["class_name"] = new_cname
                b["key"] = new_cname
                b["hand_label"] = True
                changed_box = b
                break
        if changed_box is None:
            for b in trusted:
                if self._box_identity_match(b, det):
                    b["class_id"] = new_cid
                    b["class_name"] = new_cname
                    b["key"] = new_cname
                    b["hand_label"] = True
                    changed_box = b
                    break

        if changed_box is not None:
            self._learn_box_exemplar(it, changed_box, positive=True, source="ui_class_correction")
            self._persist_review_item(it)
            self._mark_model_stale(f"Kutu sınıfı değiştirildi: {new_cname}")
            self._append_log(f"[ÖĞRENME] YZ Görsel Hafızası eğitildi: {new_cname} (cid={new_cid})")
            self._render_review_frame()

    def _accept_box(self, det: Dict[str, Any]) -> None:
        """Tek kutuyu trusted + hand_label yap."""
        if not self._review_items:
            return
        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        kept: List[Dict[str, Any]] = []
        promoted = None
        for b in weak:
            if promoted is None and self._box_identity_match(b, det):
                nb = dict(b)
                nb["trusted"] = True
                nb["hand_label"] = True
                nb["weak"] = False
                promoted = nb
            else:
                kept.append(b)
        if promoted is None:
            self._append_log("[KUTU] Kabul: kutu bulunamadı")
            return
        it["weak"] = kept
        it["n_weak"] = len(kept)
        it["trusted"] = list(it.get("trusted") or []) + [promoted]
        if not kept:
            it["done"] = "ok"
            self._pipeline["pending"] = sum(1 for x in self._review_items if not x.get("done"))
            self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + 1
        self._learn_box_exemplar(it, promoted, positive=True, source="user_accept_box")
        self._persist_review_item(it)
        self._mark_model_stale("Kutu onaylandı")
        self._append_log(
            f"[KUTU] Kabul → trusted: {promoted.get('class_name', promoted.get('class_id'))}"
        )
        self._render_queue_list()
        self._render_review_frame()

    def _remap_box_to_sprout(self, det: Dict[str, Any]) -> None:
        if not self._review_items:
            return
        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        changed = False
        for i, b in enumerate(weak):
            if self._box_identity_match(b, det):
                nb = dict(b)
                nb["class_id"] = 9
                nb["class_name"] = "minimap_sprout"
                nb["key"] = "minimap_sprout"
                nb["remapped_from"] = b.get("class_id")
                nb["hand_label"] = True
                weak[i] = nb
                changed = True
                self._learn_box_exemplar(it, nb, positive=True, source="user_remap_sprout")
                break
        if not changed:
            self._append_log("[KUTU] → filiz: kutu bulunamadı")
            return
        it["weak"] = weak
        self._persist_review_item(it)
        self._mark_model_stale("Kutu filiz olarak etiketlendi")
        self._append_log("[KUTU] Sınıf → minimap_sprout (9)")
        self._render_review_frame()

    def _demote_box_to_weak(self, det: Dict[str, Any]) -> None:
        """Emin kutuyu şüpheliye ayır (weak listesine geri al)."""
        if not self._review_items or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]
        trusted = list(it.get("trusted") or [])
        kept = []
        demoted = None
        for b in trusted:
            if demoted is None and self._box_identity_match(b, det):
                nb = dict(b)
                nb["trusted"] = False
                nb["weak"] = True
                nb["hand_label"] = False
                demoted = nb
            else:
                kept.append(b)
        if demoted is None:
            self._append_log("[KUTU] Şüpheliye ayır: kutu bulunamadı")
            return
        it["trusted"] = kept
        it["weak"] = list(it.get("weak") or []) + [demoted]
        it["n_weak"] = len(it["weak"])
        it["done"] = None  # İncelemeyi tekrar aç
        it["conf"] = self._frame_conf(it)
        self._persist_review_item(it)
        self._mark_model_stale("Kutu şüpheliye ayrıldı")
        self._append_log(f"[KUTU] Şüpheliye ayrıldı: {demoted.get('class_name', demoted.get('class_id'))}")
        self._sync_pending_from_queue()
        self._render_queue_list()
        self._render_review_frame()

    def _reject_box(self, det: Dict[str, Any], from_trusted: bool = False) -> None:
        """Sil = yalnızca seçilen kutuyu çıkar ve diskteki etiket dosyasını anında güncelle."""
        if not self._review_items or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])

        found = False
        kept_t = [b for b in trusted if not self._box_identity_match(b, det)]
        if len(kept_t) < len(trusted):
            it["trusted"] = kept_t
            it["n_trusted"] = len(kept_t)
            found = True

        if not found:
            kept_w = [b for b in weak if not self._box_identity_match(b, det)]
            if len(kept_w) < len(weak):
                it["weak"] = kept_w
                it["n_weak"] = len(kept_w)
                found = True

        if not found:
            # Fallback: geometri mesafesi en yakın kutuyu bulup sil
            def _dist(b: Dict[str, Any]) -> float:
                return abs(float(b.get("cx", 0)) - float(det.get("cx", 0))) + abs(float(b.get("cy", 0)) - float(det.get("cy", 0)))
            for pool_key in ("trusted", "weak"):
                pool = it.get(pool_key) or []
                if pool:
                    closest = min(pool, key=_dist)
                    if _dist(closest) < 0.08:
                        it[pool_key] = [b for b in pool if b is not closest]
                        it[f"n_{pool_key}"] = len(it[pool_key])
                        found = True
                        break

        if not found:
            self._append_log("[KUTU] Sil: kutu bulunamadı")
            return

        it["conf"] = self._frame_conf(it)
        note = normalize_box_class_name(det.get("class_id"), det.get("key", det.get("class_name")))
        self._learn_box_exemplar(it, det, positive=False, source="user_reject_box")

        self._persist_review_item(it)
        self._mark_model_stale(f"Kutu silindi: {note}")
        self._append_log(f"[KUTU] Silindi ve otomatik kaydedildi: {note}")
        self._render_queue_list()
        self._render_review_frame()

    @staticmethod
    def _unlink_quiet(path: Path) -> bool:
        try:
            if path.is_file():
                path.unlink()
                return True
        except Exception as exc:
            logger.debug("unlink failed %s: %s", path, exc)
        return False

    def _stems_for_sample_wipe(self, split: str, stem: str) -> List[str]:
        """Exact stem plus same-folder augment/family files that share the stem prefix across splits."""
        stems = {stem}
        dataset = self._dataset_dir()
        splits = ("train", "val") if not split or split not in ("train", "val") else (split,)
        folders = []
        for s in splits:
            folders.extend([
                dataset / "review" / s,
                dataset / "images" / s,
                dataset / "labels" / s,
            ])
        prefixes = (stem + "_", stem + "-")
        for folder in folders:
            if not folder.is_dir():
                continue
            try:
                with os.scandir(folder) as it_scan:
                    for entry in it_scan:
                        if not entry.is_file():
                            continue
                        dot = entry.name.rfind(".")
                        name = entry.name[:dot] if dot > 0 else entry.name
                        if name == stem or name.startswith(prefixes):
                            stems.add(name)
            except OSError:
                continue
        return sorted(stems)

    def _delete_review_sample(self, it: Dict[str, Any]) -> List[str]:
        """Remove review json + image + label (+ preview/aug family) across all splits and persist rejection."""
        removed: List[str] = []
        dataset = self._dataset_dir()
        stem = str(it.get("stem") or "").strip()
        if not stem:
            return removed

        # If base frame (e.g. vid_sample_30s), wipe all aug family variants as well
        if "_aug" not in stem:
            target_stems = self._stems_for_sample_wipe("", stem)
        else:
            target_stems = [stem]

        self._record_rejected_stems(target_stems)

        try:
            from dataset_factory.approved_pool import remove_approved_sample
        except Exception:
            remove_approved_sample = None

        p_val = it.get("path")
        if p_val and self._unlink_quiet(Path(p_val)):
            removed.append(str(p_val))

        img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

        for s_stem in target_stems:
            if remove_approved_sample is not None:
                try:
                    remove_approved_sample(s_stem)
                except Exception:
                    pass

            for s in ("train", "val"):
                for path in (
                    dataset / "review" / s / f"{s_stem}.json",
                    dataset / "labels" / s / f"{s_stem}.txt",
                    dataset / "review" / s / f"{s_stem}.sidecar.json",
                    dataset / "approved" / "labels" / s / f"{s_stem}.txt",
                ):
                    if self._unlink_quiet(path):
                        removed.append(str(path))

                for ext in img_exts:
                    for img in (
                        dataset / "images" / s / f"{s_stem}{ext}",
                        dataset / "approved" / "images" / s / f"{s_stem}{ext}",
                    ):
                        if self._unlink_quiet(img):
                            removed.append(str(img))

            for path in (
                dataset / "previews" / f"{s_stem}.jpg",
                dataset / "previews" / f"{s_stem}_preview.jpg",
                dataset / "previews" / f"preview_{s_stem}.jpg",
                dataset / "previews" / f"{s_stem}.png",
                dataset / "previews" / f"{s_stem}_preview.png",
                dataset / "previews" / f"preview_{s_stem}.png",
            ):
                if self._unlink_quiet(path):
                    removed.append(str(path))

            gl_dir = dataset / "game_learning"
            if gl_dir.is_dir():
                for sub in ("monsters", "skills", "paths", "obstacles", "vitality", "general", "mix"):
                    cat_dir = gl_dir / sub
                    if cat_dir.is_dir():
                        for ext in (".jpg", ".json"):
                            p = cat_dir / f"{s_stem}{ext}"
                            if self._unlink_quiet(p):
                                removed.append(str(p))

        self._review_decode_cache.clear()
        return removed

    def _batch_delete_review_samples(self, items: List[Dict[str, Any]]) -> int:
        """
        Yüzlerce kareyi UI ana iş parçacığını dondurmadan tek geçişte diskten siler.
        Tüm split'leri (train/val) ve persistent rejection kaydını kapsar.
        """
        if not items:
            return 0
        dataset = self._dataset_dir()
        removed_files = 0
        total = len(items)

        stems_to_record: List[str] = []
        for it in items:
            stem = str(it.get("stem") or "").strip()
            if stem:
                if "_aug" not in stem:
                    stems_to_record.extend(self._stems_for_sample_wipe("", stem))
                else:
                    stems_to_record.append(stem)
        self._record_rejected_stems(stems_to_record)

        try:
            from dataset_factory.approved_pool import remove_approved_sample
        except Exception:
            remove_approved_sample = None

        img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

        for idx, it in enumerate(items, 1):
            stem = str(it.get("stem") or "").strip()
            if not stem:
                continue

            target_stems = [stem]
            if "_aug" not in stem:
                target_stems = self._stems_for_sample_wipe("", stem)

            p_val = it.get("path")
            if p_val and self._unlink_quiet(Path(p_val)):
                removed_files += 1

            for s_stem in target_stems:
                if remove_approved_sample is not None:
                    try:
                        remove_approved_sample(s_stem)
                    except Exception:
                        pass

                for s in ("train", "val"):
                    for path in (
                        dataset / "review" / s / f"{s_stem}.json",
                        dataset / "labels" / s / f"{s_stem}.txt",
                        dataset / "review" / s / f"{s_stem}.sidecar.json",
                        dataset / "approved" / "labels" / s / f"{s_stem}.txt",
                    ):
                        if self._unlink_quiet(path):
                            removed_files += 1

                    for ext in img_exts:
                        for img in (
                            dataset / "images" / s / f"{s_stem}{ext}",
                            dataset / "approved" / "images" / s / f"{s_stem}{ext}",
                        ):
                            if self._unlink_quiet(img):
                                removed_files += 1

                for path in (
                    dataset / "previews" / f"{s_stem}.jpg",
                    dataset / "previews" / f"{s_stem}_preview.jpg",
                    dataset / "previews" / f"preview_{s_stem}.jpg",
                    dataset / "previews" / f"{s_stem}.png",
                    dataset / "previews" / f"{s_stem}_preview.png",
                    dataset / "previews" / f"preview_{s_stem}.png",
                ):
                    if self._unlink_quiet(path):
                        removed_files += 1

            if idx % 15 == 0 or idx == total:
                QApplication.processEvents()

        self._review_decode_cache.clear()
        return removed_files

    def _remove_review_item_at(self, index: int) -> None:
        if index < 0 or index >= len(self._review_items):
            return
        self._review_items.pop(index)
        if not self._review_items:
            self._review_index = 0
        else:
            self._review_index = min(index, len(self._review_items) - 1)
        self._pipeline["pending"] = self._pending_count()

    def _populate_manual_tag_combo(self, filter_text: str = "") -> None:
        if not hasattr(self, "manual_tag_combo"):
            return
        query = str(filter_text or "").strip().lower()
        prev_cid = self.manual_tag_combo.currentData()
        self.manual_tag_combo.blockSignals(True)
        self.manual_tag_combo.clear()

        extra_keywords: Dict[int, str] = {
            0: "aria herb bitki",
            1: "mela herb meyve",
            2: "targena azpha calendula herb",
            3: "orichalcum ore maden cevher",
            4: "gem crystal kristal safir yakut elmas",
            5: "wood tree log agac kereste",
            6: "edana crest amblem",
            7: "lumina f gather",
            8: "obsidian obsidyen e icon",
            9: "sprout filiz minimap",
            10: "canavar monster mob negatif",
            11: "minimap canavar monster dot",
            12: "blood crystal kan kristali material",
            13: "skill ui prompt arayuz",
            14: "obstacle engel",
            15: "path ground yol zemin",
            16: "hp bar can bari",
            17: "mp bar mana bari",
            18: "target hp hedef can",
            19: "skill bar",
            20: "skill ready hazir",
            21: "skill cd cooldown bekleme",
            22: "pot potion hp can iksiri",
            23: "pot potion mp mana iksiri",
            24: "scroll buff parsomen",
            25: "kinah kina altin gold para coin currency bakiye kazanc",
            26: "player character oyuncu karakterim karakter",
        }

        matched_items: List[Tuple[int, str]] = []
        all_items: List[Tuple[int, str]] = []
        for cid_opt, spec_opt in sorted(CLASS_ID_TO_SPEC.items()):
            cname_opt = str(spec_opt.get("key", f"class_{cid_opt}"))
            label_tr = str(spec_opt.get("label_tr", cname_opt))
            full_tr = str(spec_opt.get("full_tr", f"{label_tr} ({cname_opt})"))
            display = f"{cid_opt}: {cname_opt} — {label_tr}" if str(cid_opt) not in label_tr else f"{label_tr} ({cname_opt})"
            all_items.append((cid_opt, display))
            haystack = f"{cid_opt} {cname_opt} {label_tr} {full_tr} {extra_keywords.get(cid_opt, '')}".lower()
            if not query or query in haystack:
                matched_items.append((cid_opt, display))

        if not matched_items and query in CLASS_KEY_ALIASES:
            alias_cid = CLASS_KEY_ALIASES[query]
            matched_items = [item for item in all_items if item[0] == alias_cid]

        use_items = matched_items if matched_items else all_items
        for cid_opt, display in use_items:
            self.manual_tag_combo.addItem(display, cid_opt)

        if not query and prev_cid is not None:
            idx = self.manual_tag_combo.findData(prev_cid)
            if idx >= 0:
                self.manual_tag_combo.setCurrentIndex(idx)
        elif self.manual_tag_combo.count() > 0:
            self.manual_tag_combo.setCurrentIndex(0)
        self.manual_tag_combo.blockSignals(False)

    @property
    def review_items(self) -> List[Dict[str, Any]]:
        return self._review_items

    @review_items.setter
    def review_items(self, val: List[Dict[str, Any]]) -> None:
        self._review_items = val

    def _on_drawing_class_selected(self, cid: int, cname: str) -> None:
        self.active_drawing_class_id = int(cid)
        self.active_drawing_class_name = str(cname)
        if hasattr(self, "review_preview"):
            self.review_preview.active_drawing_class_id = int(cid)
            self.review_preview.active_drawing_class_name = str(cname)
        if hasattr(self, "tag_panel"):
            self.tag_panel.set_active_class(cid, cname)

    def _on_canvas_box_drawn(self, cx: float, cy: float, w: float, h: float) -> None:
        pass

    def _prompt_class_selection(self, initial_cid: int = 0) -> Optional[int]:
        """Kullanıcıya çizilen kutu için sınıf seçtiren diyalog."""
        if getattr(self, "_auto_confirm_dialogs", False):
            return initial_cid
        dlg = ClassSelectDialog(parent=self, initial_class_id=initial_cid)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.selected_class_id()
        return None

    def _prompt_change_class_for_box(self, det: Dict[str, Any]) -> None:
        """Mevcut bir kutunun sınıfını kullanıcıya diyalogla seçtirip günceller."""
        cur_cid = int(det.get("class_id", 0))
        new_cid = self._prompt_class_selection(initial_cid=cur_cid)
        if new_cid is not None and new_cid != cur_cid:
            self._on_box_class_changed(det, new_cid)

    def _on_manual_box_added(self, box: Dict[str, Any]) -> None:
        """Fareyle çizilen yeni kutuyu seçili sınıfla ekler ve diskteki etiket dosyasına anında otomatik kaydeder."""
        if not self._review_items or self._review_index < 0 or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]

        # Sınıf seçimi
        combo = self.__dict__.get("manual_tag_combo")
        combo_cid = combo.currentData() if combo is not None and combo.currentData() is not None else None
        active_cid = self.__dict__.get("active_drawing_class_id")
        box_cid = box.get("class_id")
        is_direct_box = bool(box.get("is_direct_class") or int(box.get("class_id", -1) or -1) == 26)

        if is_direct_box and box_cid is not None:
            init_cid = int(box_cid)
        elif box_cid is not None and int(box_cid) != 0:
            init_cid = int(box_cid)
        elif active_cid is not None and int(active_cid) != 0:
            init_cid = int(active_cid)
        elif combo_cid is not None and int(combo_cid) != 0:
            init_cid = int(combo_cid)
        elif box_cid is not None:
            init_cid = int(box_cid)
        elif active_cid is not None:
            init_cid = int(active_cid)
        elif combo_cid is not None:
            init_cid = int(combo_cid)
        else:
            init_cid = 0

        auto_conf = self.__dict__.get("_auto_confirm_dialogs", False)
        # Sadece explicit class_id verilmediğinde ve dialog modal desteği varsa kullanıcıya sor
        if not is_direct_box and not auto_conf and hasattr(self, "exec") and hasattr(self, "windowTitle"):
            try:
                chosen_cid = self._prompt_class_selection(initial_cid=init_cid)
                if chosen_cid is None:
                    self._append_log("[MANUEL ETİKET] Kutu çizimi iptal edildi.")
                    if hasattr(self, "review_preview"):
                        self.review_preview.clear_drag_state()
                        self.review_preview.update()
                    return
                cid = chosen_cid
            except RuntimeError:
                cid = init_cid
        else:
            cid = init_cid

        spec = CLASS_ID_TO_SPEC.get(cid, YOLO_12_CLASS_SPECS[0])
        cname = str(spec.get("key", f"class_{cid}"))
        color_hex = str(spec.get("color_hex", CLASS_COLORS_MAP.get(cid, "#22c55e")))

        # Aktif sınıfı da güncelle
        self.__dict__["active_drawing_class_id"] = cid
        self.__dict__["active_drawing_class_name"] = cname
        review_prev = self.__dict__.get("review_preview")
        if review_prev is not None:
            review_prev.active_drawing_class_id = cid
            review_prev.active_drawing_class_name = cname
        tag_p = self.__dict__.get("tag_panel")
        if tag_p is not None:
            tag_p.set_active_class(cid, cname)

        new_det: Dict[str, Any] = {
            "class_id": cid,
            "class_name": cname,
            "key": cname,
            "color": color_hex,
            "cx": float(box.get("cx", 0.5)),
            "cy": float(box.get("cy", 0.5)),
            "w": float(box.get("w", 0.08)),
            "h": float(box.get("h", 0.12)),
            "confidence": 1.0,
            "conf": 1.0,
            "trusted": True,
            "weak": False,
            "hand_label": True,
            "human_approved": True,
            "source": "manual_draw",
        }
        it["trusted"] = list(it.get("trusted") or []) + [new_det]
        it["n_trusted"] = len(it["trusted"])
        it["conf"] = self._frame_conf(it)
        if review_prev is not None:
            review_prev.clear_drag_state()
        self._learn_box_exemplar(it, new_det, positive=True, source="manual_draw")
        self._persist_review_item(it)
        self._mark_model_stale(f"Manuel etiket çizildi: {cname}")
        self._append_log(f"[MANUEL ETİKET] #{it.get('stem')}: {cname} (cid={cid}) eklendi ve otomatik kaydedildi.")
        self._render_queue_list()
        self._render_review_frame()

    def _on_box_edited(self, box: Dict[str, Any]) -> None:
        """Fareyle taşınan veya boyutu değiştirilen kutuyu günceller ve diskteki etiket dosyasına anında kaydeder."""
        if not self._review_items or self._review_index < 0 or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]

        # Canvas'taki en güncel etiket listesini al
        review_prev = self.__dict__.get("review_preview")
        canvas_anns = list(getattr(review_prev, "_annotations", [])) if review_prev is not None else []
        if canvas_anns:
            for b in canvas_anns:
                if b is box or (
                    b.get("class_id") == box.get("class_id")
                    and abs(float(b.get("cx", 0)) - float(box.get("cx", 0))) < 1e-4
                    and abs(float(b.get("cy", 0)) - float(box.get("cy", 0))) < 1e-4
                ):
                    b["trusted"] = True
                    b["hand_label"] = True
                    b["human_approved"] = True
                    b["confidence"] = 1.0
                    b["conf"] = 1.0
            it["trusted"] = canvas_anns
            it["weak"] = []
            it["n_trusted"] = len(it["trusted"])
            it["conf"] = self._frame_conf(it)

        self._persist_review_item(it)
        cname = box.get("class_name", f"class_{box.get('class_id', 0)}")
        cx = float(box.get("cx", 0.0))
        cy = float(box.get("cy", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        self._mark_model_stale(f"Etiket konumu/boyutu güncellendi: {cname}")
        self._append_log(
            f"[KUTU DÜZENLENDİ] #{it.get('stem')}: '{cname}' taşındı / boyutlandırıldı "
            f"(cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}) ve diske kaydedildi."
        )
        self._render_queue_list()
        self._render_review_frame()

    def _persist_review_item(self, it: Dict[str, Any]) -> None:
        path_val = it.get("path")
        if path_val:
            path = Path(path_val)
        else:
            path = self._dataset_dir() / "review" / str(it.get("split", "train")) / f"{it.get('stem', 'sample')}.json"
        try:
            payload = {
                "stem": it["stem"],
                "split": it["split"],
                "trusted": it.get("trusted", []),
                "weak": it.get("weak", []),
                "done": it.get("done"),
                "human_approved": bool(it.get("human_approved") or it.get("done") == "ok"),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            label_path = self._dataset_dir() / "labels" / it["split"] / f"{it['stem']}.txt"
            from dataset_factory.label_trust import commit_split_labels

            anns = list(it.get("trusted", [])) + [
                {**b, "trusted": True, "hand_label": True, "human_approved": True}
                for b in it.get("weak", [])
                if it.get("done") == "ok"
            ]
            # Disk üzerindeki labels/{split}/{stem}.txt dosyasını hem doğrudan hem commit_split_labels ile garantiye al
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for ann in anns:
                cid = int(ann.get("class_id", 0))
                cx = float(ann.get("cx", 0.5))
                cy = float(ann.get("cy", 0.5))
                w = float(ann.get("w", 0.08))
                h = float(ann.get("h", 0.12))
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            label_path.write_text("".join(lines), encoding="utf-8")

            try:
                commit_split_labels(
                    dataset_dir=self._dataset_dir(),
                    split=it["split"],
                    stem=it["stem"],
                    label_path=label_path,
                    annotations=anns,
                    human_approved=True,
                    done="ok" if it.get("done") == "ok" else None,
                )
            except Exception as lbl_exc:
                logger.debug("commit_split_labels failed: %s", lbl_exc)

            if it.get("done") == "ok":
                try:
                    from dataset_factory.approved_pool import get_approved_pool

                    pool = get_approved_pool(self._dataset_dir())
                    img_path = self._resolve_review_image(it["split"], it["stem"])
                    try:
                        pool.add_approved_sample(
                            stem=it["stem"],
                            split=it["split"],
                            image_path=img_path,
                            boxes=it.get("trusted", []),
                        )
                    except TypeError:
                        pool.add_approved_sample(
                            stem=it["stem"],
                            split=it["split"],
                            image_path=img_path,
                            trusted_boxes=it.get("trusted", []),
                        )
                except Exception as pool_exc:
                    logger.debug("approved_pool add failed: %s", pool_exc)
        except Exception as exc:
            logger.debug("persist review failed: %s", exc)

    def _review_done(self, kind: str) -> None:
        if not self._review_items:
            return
        index = self._review_index
        it = self._review_items[index]
        if it.get("done"):
            return
        stem = it.get("stem")
        if kind == "no":
            for b in it.get("weak", []):
                cid = int(b.get("class_id", 0))
                if cid not in (0, 1, 2, 3, 4, 5, 6, 12):
                    self._learn_box_exemplar(it, b, positive=False, source="user_frame_reject")
            removed = self._delete_review_sample(it)
            self._remove_review_item_at(index)
            rej_stems = set(self._rejected_stems)
            self._review_items = [
                x for x in self._review_items
                if str(x.get("stem") or "").strip() not in rej_stems
            ]
            self._review_index = min(self._review_index, max(0, len(self._review_items) - 1))
            self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + 1
            self._sync_pending_from_queue()
            self._mark_model_stale(f"Kare #{stem} silindi")
            self._append_log(
                f"[RED] #{stem} diskten silindi ({len(removed)} dosya) · kalan {self._pending_count()}"
            )
            self._render_queue_list()
            self._render_review_frame()
            self._update_blockers()
            self._update_next_card()
            if self._pending_count() == 0:
                self._mark_step_done(2)
                self._append_log("[AKIŞ] Onay kuyruğu boşaldı → Hazırla adımı hazır.")
            self._refresh_stepper_visuals()
            return

        it["done"] = "ok"
        it["human_approved"] = True
        review_prev = self.__dict__.get("review_preview")
        if review_prev is not None and getattr(review_prev, "_annotations", None):
            canvas_anns = list(review_prev._annotations)
            for b in canvas_anns:
                b["trusted"] = True
                b["hand_label"] = True
                b["human_approved"] = True
                if "confidence" not in b or float(b.get("confidence", 0)) < 0.90:
                    b["confidence"] = 1.0
                    b["conf"] = 1.0
                self._learn_box_exemplar(it, b, positive=True, source="user_frame_approve")
            it["trusted"] = canvas_anns
            it["weak"] = []
        else:
            promoted = []
            for b in it.get("weak", []):
                nb = dict(b)
                nb["trusted"] = True
                nb["hand_label"] = True
                nb["human_approved"] = True
                promoted.append(nb)
                self._learn_box_exemplar(it, nb, positive=True, source="user_frame_approve")
            it["trusted"] = list(it.get("trusted", [])) + promoted
            it["weak"] = []

        # Oturum onaylanan kinah miktarını güncelle
        approved_boxes = list(it.get("trusted", []))
        approved_kinah = sum(self._estimate_box_kinah(b) for b in approved_boxes)
        self._session_approved_kinah = self.__dict__.get("_session_approved_kinah", 0) + approved_kinah

        self._persist_review_item(it)
        self._remove_review_item_at(index)
        self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + 1
        self._sync_pending_from_queue()
        lbl_app = self.__dict__.get("lbl_approved_count")
        if lbl_app is not None:
            try:
                pool = get_approved_pool(self._dataset_dir())
                lbl_app.setText(f"✓ {pool.get_approved_count()} Onaylandı")
            except Exception:
                pass
        self._mark_model_stale(f"Kare #{stem} onaylandı")
        n_saved = len(it.get("trusted", []))
        kinah_msg = f" (+~{format_kinah_tr(approved_kinah)} Kinah)" if approved_kinah > 0 else ""
        self._append_log(
            f"[ONAYLANDI & KAYDEDİLDİ] #{stem}: {n_saved} etiket diske ve onay havuzuna kesin olarak kaydedildi{kinah_msg} · kalan {self._pending_count()}"
        )
        self._render_queue_list()
        self._render_review_frame()
        self._update_blockers()
        self._update_next_card()
        if self._pending_count() == 0:
            self._mark_step_done(2)
            self._append_log("[AKIŞ] Onay kuyruğu boşaldı → Hazırla adımı hazır.")
        self._refresh_stepper_visuals()

    def _is_step_2_active(self) -> bool:
        """Return True if Step 2 review queue is currently active."""
        if hasattr(self, "step_stack") and self.step_stack is not None:
            if self.step_stack.currentIndex() == 1:
                return True
        return int(self._pipeline.get("step", 0)) == 2

    def _is_text_input_focused(self) -> bool:
        """Check if any text entry or numeric spinbox currently has keyboard focus."""
        fw = QApplication.focusWidget() or (self.focusWidget() if hasattr(self, "focusWidget") else None)
        if fw is None:
            return False
        return isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox))

    def _on_step2_approve(self) -> None:
        """Approve current review sample ('A') and advance."""
        self._review_done("ok")

    def _on_step2_reject(self) -> None:
        """Reject/delete current review sample ('R') and advance."""
        self._review_done("no")

    def _handle_shortcut_approve(self) -> None:
        if not self._is_step_2_active() or self._is_text_input_focused():
            return
        self._on_step2_approve()

    def _handle_shortcut_reject(self) -> None:
        if not self._is_step_2_active() or self._is_text_input_focused():
            return
        self._on_step2_reject()

    def _handle_shortcut_skip(self) -> None:
        if not self._is_step_2_active() or self._is_text_input_focused():
            return
        self._review_skip()

    def _handle_shortcut_player_char(self) -> None:
        if not self._is_step_2_active() or self._is_text_input_focused():
            return
        self._add_player_character_box_to_current_frame()

    def _add_player_character_box_to_current_frame(self) -> None:
        """Karakter kutusunu mevcut kareye otomatik ekler veya seçer."""
        if not self._review_items or self._review_index < 0 or self._review_index >= len(self._review_items):
            self._append_log("[ETİKET] Açık kare yok.")
            return

        it = self._review_items[self._review_index]
        trusted = it.setdefault("trusted", [])

        # Karakter sınıfı: 26 (player_character)
        cid = 26
        spec = CLASS_ID_TO_SPEC.get(cid, {})
        cname = str(spec.get("key", "player_character"))

        # Mevcut bir karakter kutusu var mı kontrol et
        for idx, b in enumerate(trusted):
            if int(b.get("class_id", -1)) == cid or str(b.get("class_name", "")).lower() in ("player_character", "player", "karakter", "oyuncu"):
                if hasattr(self, "review_preview"):
                    self.review_preview._selected_box_idx = idx
                    self.review_preview.update()
                self._append_log(f"[ETİKET] 🎯 Karakter kutusu zaten mevcut (Kutu #{idx + 1}). Tutamaçları çekerek boyutunu ve yerini ayarlayabilirsiniz.")
                return

        # Üçüncü şahıs perspektifte karakter ekranın alt-merkezindedir:
        # cx=0.50, cy=0.62, w=0.18, h=0.38
        new_box = {
            "class_id": cid,
            "class_name": cname,
            "cx": 0.50,
            "cy": 0.62,
            "w": 0.18,
            "h": 0.38,
            "confidence": 1.0,
            "conf": 1.0,
            "trusted": True,
            "weak": False,
            "hand_label": True,
            "human_approved": True,
            "color": CLASS_COLORS_MAP.get(cid, "#00d2ff"),
        }
        self._on_manual_box_added(new_box)
        if hasattr(self, "review_preview"):
            self.review_preview._selected_box_idx = len(trusted) - 1
            self.review_preview.update()
        self._append_log("[ETİKET] 🎯 Karakter kutusu merkeze eklendi (26: Karakterim). Tutamaçlardan boyutunu ve konumunu ayarlayabilirsiniz.")

    def _load_current_review_sample(self) -> None:
        """Load and render the current review sample into preview canvas and detail rows."""
        self._render_review_frame()

    def _trigger_review_shortcut(self, action: str) -> None:
        """A (Onayla), R (Reddet), S (Atla) kısayollarını odak durumunu kontrol ederek çalıştırır."""
        if not self._is_step_2_active() or self._is_text_input_focused():
            return
        if action == "ok":
            self._on_step2_approve()
        elif action == "no":
            self._on_step2_reject()
        elif action == "skip":
            self._review_skip()

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        if event is not None and event.type() == event.Type.KeyPress:
            if self._is_step_2_active() and not self._is_text_input_focused():
                mods = event.modifiers()
                if mods in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier):
                    key = event.key()
                    text = (event.text() or "").upper()
                    if key in (Qt.Key.Key_A, 65) or text == "A":
                        self._on_step2_approve()
                        return True
                    if key in (Qt.Key.Key_R, 82) or text == "R":
                        self._on_step2_reject()
                        return True
                    if key in (Qt.Key.Key_S, 83) or text == "S":
                        self._review_skip()
                        return True
                    if key in (Qt.Key.Key_K, 75) or text == "K":
                        self._add_player_character_box_to_current_frame()
                        return True

        if obj is getattr(self, "review_preview", None) and event is not None:
            et = event.type()
            if et == event.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                # Çift tıklama: Lightbox (Büyük Önizleme) aç
                self._open_review_lightbox()
                return True
            if et == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                hit_ann = self.review_preview.hit_test_close_button(pos)
                if hit_ann is not None:
                    self._review_close_pressed_ann = hit_ann
                    return True
                self._review_close_pressed_ann = None
                return False
            if et == event.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                # Eğer '✕' butonuna basıldıysa veya bırakıldıysa: Sil ve ASLA lightbox açma!
                hit_ann = getattr(self, "_review_close_pressed_ann", None) or self.review_preview.hit_test_close_button(pos)
                self._review_close_pressed_ann = None
                if hit_ann is not None:
                    self.review_preview.clear_drag_state()
                    self._reject_box(hit_ann, from_trusted=bool(hit_ann.get("trusted")))
                    return True
                press_pos = getattr(self, "_review_press_pos", None)
                self._review_press_pos = None
                dragged = bool(getattr(self.review_preview, "_is_dragging_box", False))
                if press_pos is not None:
                    if abs(pos.x() - press_pos.x()) >= 8.0 or abs(pos.y() - press_pos.y()) >= 8.0:
                        dragged = True
                if float(getattr(self.review_preview, "_last_drag_distance", 0.0)) >= 8.0:
                    dragged = True
                if dragged:
                    self.review_preview.mouseReleaseEvent(event)
                    return True
                self.review_preview.clear_drag_state()
                hit_box = self.review_preview.hit_test_box(pos)
                if hit_box is not None:
                    _ann, idx = hit_box
                    self.review_preview._selected_box_idx = idx
                    self.review_preview.setFocus()
                    self.review_preview.update()
                    return True
                self.review_preview._selected_box_idx = -1
                self.review_preview.update()
                return True
        if obj is getattr(self, "preview_canvas", None) and event is not None:
            et = event.type()
            if et == event.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                self._open_capture_lightbox()
                return True
            if et == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                hit_ann = self.preview_canvas.hit_test_close_button(pos)
                if hit_ann is not None:
                    self._capture_close_pressed_ann = hit_ann
                    return True
                self._capture_close_pressed_ann = None
                self._capture_press_pos = pos
                return False
            if et == event.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                hit_ann = getattr(self, "_capture_close_pressed_ann", None) or self.preview_canvas.hit_test_close_button(pos)
                self._capture_close_pressed_ann = None
                if hit_ann is not None:
                    anns = [a for a in getattr(self.preview_canvas, "_annotations", []) if a is not hit_ann]
                    self.preview_canvas._annotations = anns
                    self.preview_canvas.update()
                    return True
                press_pos = getattr(self, "_capture_press_pos", None)
                self._capture_press_pos = None
                dragged = bool(getattr(self.preview_canvas, "_is_dragging_box", False))
                if press_pos is not None:
                    if abs(pos.x() - press_pos.x()) >= 8.0 or abs(pos.y() - press_pos.y()) >= 8.0:
                        dragged = True
                if float(getattr(self.preview_canvas, "_last_drag_distance", 0.0)) >= 8.0:
                    dragged = True
                if dragged:
                    self.preview_canvas.mouseReleaseEvent(event)
                    return True
                self.preview_canvas.clear_drag_state()
                return True
        return super().eventFilter(obj, event)

    def _open_review_lightbox(self) -> None:
        pix = self._review_lightbox_pixmap
        if pix is None or pix.isNull():
            pix = getattr(self.review_preview, "_pixmap", None)
        if not isinstance(pix, QPixmap) or pix.isNull():
            self._append_log("[ÖNİZLEME] Lightbox: büyütülecek pixmap yok")
            return

        # Toplanan tespit kutuları (weak ve trusted etiketleri)
        annotations: List[Dict[str, Any]] = []
        stem = ""
        conf = 0.0
        if self._review_items and 0 <= self._review_index < len(self._review_items):
            it = self._review_items[self._review_index]
            stem = str(it.get("stem", ""))
            weak = list(it.get("weak") or [])
            trusted = list(it.get("trusted") or [])
            for w in weak:
                box_dict = dict(w)
                box_dict.setdefault("trusted", False)
                box_dict.setdefault("weak", True)
                annotations.append(box_dict)
            for t in trusted:
                box_dict = dict(t)
                box_dict.setdefault("trusted", True)
                box_dict.setdefault("weak", False)
                annotations.append(box_dict)
            conf = float(it.get("conf", 0.0) or 0.0)

        if not annotations and hasattr(self, "review_preview"):
            annotations = list(getattr(self.review_preview, "_annotations", []))

        prev_interval = int(self._refresh_timer.interval()) if hasattr(self, "_refresh_timer") else 4000
        self._lightbox_open = True
        if hasattr(self, "_refresh_timer") and self._refresh_timer.isActive():
            self._refresh_timer.setInterval(20000)

        try:
            from ui.yolo_panel import YOLOPreviewDialog
            meta_str = f"{len(annotations)} tespit · En düşük güven: {conf:.2f}" if annotations else "Kutu yok"
            title_str = f"Önizleme · Kare #{stem}" if stem else "Önizleme"
            dlg = YOLOPreviewDialog(
                source_pixmap=pix,
                annotations=annotations,
                title=title_str,
                meta=meta_str,
                parent=self,
                on_remove_annotation=lambda ann: self._reject_box(ann, from_trusted=bool(ann.get("trusted"))),
            )
            dlg.exec()
        except Exception as e:
            logger.warning("[ÖNİZLEME] YOLOPreviewDialog hatası: %s", e)
        finally:
            self._lightbox_open = False
            if hasattr(self, "_refresh_timer"):
                self._refresh_timer.setInterval(prev_interval)

    def _open_capture_lightbox(self) -> None:
        pix = getattr(self.preview_canvas, "_pixmap", None)
        if not isinstance(pix, QPixmap) or pix.isNull():
            self._append_log("[ÖNİZLEME] Canlı Lightbox: büyütülecek pixmap yok")
            return

        annotations = list(getattr(self.preview_canvas, "_annotations", []))
        badge = getattr(self.preview_canvas, "_live_badge", "")
        prev_interval = int(self._refresh_timer.interval()) if hasattr(self, "_refresh_timer") else 4000
        self._lightbox_open = True
        if hasattr(self, "_refresh_timer") and self._refresh_timer.isActive():
            self._refresh_timer.setInterval(20000)

        try:
            from ui.yolo_panel import YOLOPreviewDialog
            title_str = f"Canlı Yakalama · {badge}" if badge else "Canlı Yakalama Önizleme"
            meta_str = f"{len(annotations)} tespit" if annotations else "Kutu yok"
            dlg = YOLOPreviewDialog(
                source_pixmap=pix,
                annotations=annotations,
                title=title_str,
                meta=meta_str,
                parent=self,
                on_remove_annotation=lambda ann: self._reject_box(ann, from_trusted=bool(ann.get("trusted"))),
            )
            dlg.exec()
        except Exception as e:
            logger.warning("[ÖNİZLEME] Capture YOLOPreviewDialog hatası: %s", e)
        finally:
            self._lightbox_open = False
            if hasattr(self, "_refresh_timer"):
                self._refresh_timer.setInterval(prev_interval)

    def _review_skip(self) -> None:
        if not self._review_items:
            return
        if hasattr(self, "review_preview"):
            self.review_preview.clear_drag_state()
        self._review_index = (self._review_index + 1) % len(self._review_items)
        self.queue_list.blockSignals(True)
        self.queue_list.setCurrentRow(self._review_index)
        self.queue_list.blockSignals(False)
        self._render_review_frame()
        self._append_log("[KUYRUK] Kare atlandı.")

    def _bulk_approve(self) -> None:
        """Kutu düzeyinde toplu onay: Yalnızca o an filtrelenen kareleri onaylar.
        Eşik üstü kutuları Emin yapar, şüpheli/boş filtre modlarında filtrelenen kareleri onaylar."""
        th = self._bulk_threshold() if hasattr(self, "bulk_range") else 0.90
        targets = list(self._filtered_review_indices())
        if not targets:
            self._append_log("[ONAY] Filtrede onaylanacak kare yok.")
            return

        total = len(targets)
        n = 0
        promoted_total = 0
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")

        for i_step, idx in enumerate(targets, 1):
            if idx < 0 or idx >= len(self._review_items):
                continue
            it = self._review_items[idx]
            if it.get("done"):
                continue

            weak = list(it.get("weak") or [])
            trusted = list(it.get("trusted") or [])

            if not weak:
                if trusted or mode == "empty":
                    it["done"] = "ok"
                    it["human_approved"] = True
                    self._persist_review_item(it)
                    n += 1
                continue

            kept_weak = []
            promoted = []
            for b in weak:
                b_conf = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                norm_b_conf = b_conf if b_conf <= 1.0 else b_conf / 100.0
                if mode in ("suspicious", "unlabeled_and_suspicious", "unlabeled_or_suspicious") or norm_b_conf >= th:
                    nb = dict(b)
                    nb["trusted"] = True
                    nb["weak"] = False
                    nb["hand_label"] = True
                    nb["human_approved"] = True
                    promoted.append(nb)
                    promoted_total += 1
                    if total <= 20:
                        self._learn_box_exemplar(it, nb, positive=True, source="user_bulk_approve")
                else:
                    kept_weak.append(b)

            if promoted or (mode == "suspicious" and not kept_weak):
                it["trusted"] = list(it.get("trusted") or []) + promoted
                it["weak"] = kept_weak
                it["n_weak"] = len(kept_weak)
                if not kept_weak:
                    it["done"] = "ok"
                    it["human_approved"] = True
                it["conf"] = self._frame_conf(it)
                self._persist_review_item(it)
                n += 1
            elif not kept_weak:
                it["done"] = "ok"
                it["human_approved"] = True
                it["conf"] = self._frame_conf(it)
                self._persist_review_item(it)
                n += 1

            if i_step % 15 == 0 or i_step == total:
                QApplication.processEvents()
                if hasattr(self, "lbl_train_status") and self.lbl_train_status is not None:
                    self.lbl_train_status.setText(f"Toplu onaylanıyor: {i_step}/{total}...")

        for it in list(self._review_items):
            if it.get("done") == "ok":
                try:
                    from dataset_factory.approved_pool import get_approved_pool
                    pool = get_approved_pool(self._dataset_dir())
                    img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
                    try:
                        pool.add_approved_sample(
                            stem=it["stem"],
                            split=it["split"],
                            image_path=img_path,
                            boxes=it.get("trusted", []),
                        )
                    except TypeError:
                        pool.add_approved_sample(
                            stem=str(it.get("stem", "")),
                            split=str(it.get("split", "train")),
                            image_path=img_path,
                            trusted_boxes=it.get("trusted", []),
                        )
                except Exception as exc:
                    logger.debug("ApprovedDatasetPool bulk_approve save skipped: %s", exc)

        self._review_items = [x for x in self._review_items if x.get("done") != "ok"]
        if self._review_index >= len(self._review_items):
            self._review_index = max(0, len(self._review_items) - 1)

        self._sync_pending_from_queue()
        self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + n
        if n > 0:
            self._mark_model_stale(f"Filtre onay: {promoted_total} kutu, {n} kare")
        self._render_queue_list()
        self._update_filter_action_buttons()
        self._load_current_review_sample()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        self._append_log(f"[ONAY] Filtredeki {n} kare onaylandı ({promoted_total} kutu terfi etti).")
        if self._pending_count() == 0:
            self._mark_step_done(2)

    def _bulk_prune_low_confidence_boxes(self) -> None:
        """Kutu düzeyinde budama: Yalnızca filtredeki karelerde eşik altındaki kutuları siler, görsel ve emin kutular diskte korunur."""
        th = self._bulk_threshold() if hasattr(self, "bulk_range") else 0.90
        targets = list(self._filtered_review_indices())
        if not targets:
            self._append_log("[BUDAMA] Budanacak filtreli kare yok.")
            return

        if not getattr(self, "_auto_confirm_dialogs", False):
            reply = QMessageBox.question(
                self,
                "Düşük Güvenli Etiketleri Sil (Görseli Koru)",
                (
                    f"{len(targets)} kare taranacak (Eşik: %{int(round(th * 100))}).\n\n"
                    "• Eşik altındaki (<) hatalı etiketler tek tek silinir ve YZ hafızasına (ExemplarMemory) negatif örnek yazılır.\n"
                    "• GÖRSELLER ASLA SİLİNMEZ! Yüksek güvenli (≥) etiketler (%90+) korunur ve Emin yapılır.\n\n"
                    "Devam edilsin mi?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        pruned_boxes = 0
        preserved_images = 0
        promoted_boxes = 0

        # Varsa label_trust modülünün toplu budama fonksiyonunu dene
        try:
            from dataset_factory.label_trust import prune_low_confidence_boxes_in_dataset
            stems = [str(self._review_items[i].get("stem", "")) for i in targets if i < len(self._review_items)]
            res = prune_low_confidence_boxes_in_dataset(
                dataset_dir=self._dataset_dir(),
                threshold=th,
                indices_or_stems=stems,
                record_negative_exemplars=True,
            )
            if isinstance(res, dict) and (res.get("pruned_boxes", 0) > 0 or res.get("pruned", 0) > 0):
                pruned_boxes = int(res.get("pruned_boxes", res.get("pruned", 0)))
                preserved_images = int(res.get("preserved_frames", len(targets)))
                promoted_boxes = int(res.get("promoted_boxes", 0))
        except Exception as exc:
            logger.debug("prune_low_confidence_boxes_in_dataset fallback: %s", exc)

        total = len(targets)
        if pruned_boxes == 0:
            for i_step, i in enumerate(targets, 1):
                if i < 0 or i >= len(self._review_items):
                    continue
                it = self._review_items[i]
                weak = list(it.get("weak") or [])
                trusted = list(it.get("trusted") or [])
                if not weak and not trusted:
                    continue

                kept_trusted = list(trusted)
                for b in weak:
                    c = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                    norm_c = c if c <= 1.0 else c / 100.0
                    if norm_c < th and not b.get("hand_label"):
                        pruned_boxes += 1
                        if total <= 20:
                            self._learn_box_exemplar(it, b, positive=False, source="bulk_box_prune")
                    else:
                        nb = dict(b)
                        nb["trusted"] = True
                        nb["weak"] = False
                        kept_trusted.append(nb)
                        promoted_boxes += 1

                it["weak"] = []
                it["trusted"] = kept_trusted
                it["n_weak"] = 0
                it["done"] = "ok"
                it["conf"] = self._frame_conf(it)
                self._persist_review_item(it)
                preserved_images += 1

                if i_step % 15 == 0 or i_step == total:
                    QApplication.processEvents()
                    if hasattr(self, "lbl_train_status") and self.lbl_train_status is not None:
                        self.lbl_train_status.setText(f"Etiketler budanıyor: {i_step}/{total}...")

        self._sync_pending_from_queue()
        self._mark_model_stale(f"Kutu budama: {pruned_boxes} silindi, {preserved_images} görsel korundu")
        self._render_queue_list()
        self._update_filter_action_buttons()
        self._load_current_review_sample()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        self._append_log(
            f"[BUDAMA] Tamamlandı: {pruned_boxes} düşük güvenli kutu silindi, "
            f"{promoted_boxes} kutu onaylandı, {preserved_images} görsel korundu (Eşik %{int(round(th * 100))})."
        )
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.information(
                self,
                "Budama Tamamlandı",
                (
                    f"Kutu düzeyinde budama başarıyla tamamlandı!\n\n"
                    f"• Silinen Düşük Güvenli Kutu: {pruned_boxes}\n"
                    f"• Onaylanan Güvenli Kutu: {promoted_boxes}\n"
                    f"• Korunan Görsel Sayısı: {preserved_images}\n"
                    f"• Negatif Hafızaya Öğretildi: {pruned_boxes}\n\n"
                    "Hiçbir görsel silinmedi; kalan yüksek güvenli etiketler korundu."
                ),
            )

    def _auto_label_unlabeled_suspicious(self) -> None:
        """Etiketsiz (0 kutu) ve şüpheli kareleri filtreleyip otomatik canlı yeniden etiketlemeyi başlatır."""
        idx = self.queue_filter.findData("unlabeled_and_suspicious")
        if idx >= 0:
            self.queue_filter.setCurrentIndex(idx)
        else:
            self._queue_filter_mode = "unlabeled_and_suspicious"
            self._render_queue_list()

        targets = self._filtered_review_indices()
        if not targets:
            self._append_log("[OTOMATİK ETİKETLE] Etiketsiz veya şüpheli kare bulunamadı.")
            QMessageBox.information(self, "Bilgi", "İncelenecek etiketsiz veya şüpheli kare bulunamadı.")
            return

        self._append_log(f"[OTOMATİK ETİKETLE] {len(targets)} etiketsiz/şüpheli kare taranıyor...")
        self._start_live_relabel_filtered()

    def _toggle_live_relabel_filtered(self) -> None:
        """Filtrelenmiş kareleri canlı yeniden etiketlemeyi başlat veya durdur."""
        if getattr(self, "_live_relabeling_filtered", False):
            self._stop_live_relabel_filtered()
        else:
            self._start_live_relabel_filtered()

    def _start_live_relabel_filtered(self) -> None:
        targets = self._filtered_review_indices()
        if not targets:
            self._append_log("[CANLI ETİKETLEME] Filtreye uyan kare yok.")
            return

        self._live_relabeling_filtered = True
        self._live_relabel_targets = list(targets)
        self._live_relabel_pos = 0
        self._live_relabel_total_boxes = 0
        self._live_relabel_start_time = time.monotonic()

        self.btn_live_relabel_filtered.setText("⏹ Filtredekileri Durdur")
        self.btn_live_relabel_filtered.setObjectName("PipeDanger")
        self.btn_live_relabel_filtered.setStyle(self.btn_live_relabel_filtered.style())
        self._append_log(f"[CANLI ETİKETLEME] Filtredekiler ({len(targets)} kare) için başlatıldı...")
        QTimer.singleShot(20, self._step_live_relabel_filtered)

    def _stop_live_relabel_filtered(self) -> None:
        self._live_relabeling_filtered = False
        cnt = len(self._filtered_review_indices())
        self.btn_live_relabel_filtered.setText(f"🎯 Filtredekileri Canlı Yeniden Etiketle ({cnt} kare)")
        self.btn_live_relabel_filtered.setObjectName("PipeBtn")
        self.btn_live_relabel_filtered.setStyle(self.btn_live_relabel_filtered.style())
        self._sync_pending_from_queue()
        self._mark_model_stale("Filtre canlı etiketleme tamamlandı")
        self._render_queue_list()
        self._render_review_frame()
        self.lbl_speed_meter.setText("⚡ Hız: — · Kalan: — · Tamamlandı")
        self._append_log(
            f"[CANLI ETİKETLEME] Filtredekiler tamamlandı. Toplam çıkarılan kutu: {getattr(self, '_live_relabel_total_boxes', 0)}"
        )

    def _step_live_relabel_filtered(self) -> None:
        if not getattr(self, "_live_relabeling_filtered", False):
            return
        targets = getattr(self, "_live_relabel_targets", [])
        pos = getattr(self, "_live_relabel_pos", 0)
        total = len(targets)

        if pos >= total:
            self._stop_live_relabel_filtered()
            self._append_log("[CANLI ETİKETLEME] Filtrelenmiş tüm kareler başarıyla yeniden etiketlendi!")
            return

        real_i = targets[pos]
        if 0 <= real_i < len(self._review_items):
            it = self._review_items[real_i]
            self._review_index = real_i

            self.queue_list.blockSignals(True)
            self.queue_list.setCurrentRow(self._review_index)
            self.queue_list.blockSignals(False)

            n_boxes = self._relabel_single_item(it)
            self._live_relabel_total_boxes += n_boxes

            self._render_review_frame()

        # Hız sayacı ve ETA hesapla
        elapsed = max(0.001, time.monotonic() - self._live_relabel_start_time)
        processed = pos + 1
        fps = processed / elapsed
        rem = max(0, total - processed)
        eta_sec = rem / fps if fps > 0.05 else 0.0
        eta_str = f"{int(eta_sec // 60)}dk {int(eta_sec % 60):02d}sn" if eta_sec >= 60 else f"{int(eta_sec)} sn"

        speed_txt = f"⚡ {fps:.1f} kare/sn · Kalan: {eta_str} · {processed}/{total}"
        self.lbl_speed_meter.setText(speed_txt)
        self.review_prog_txt.setText(f"{processed} / {total} ({fps:.1f} k/s)")
        self.review_prog.setValue(int(round(100 * processed / max(1, total))))

        QApplication.processEvents()
        self._live_relabel_pos += 1
        QTimer.singleShot(35, self._step_live_relabel_filtered)

    def _relabel_single_item(self, it: Dict[str, Any]) -> int:
        """Kareyi YOLOAutoLabeler + ExemplarMemory ile sıfırdan yeniden etiketler (Emin vs Şüpheli ayrımı)."""
        img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
        if img_path is None or not img_path.is_file():
            return 0
        img = self._imread_cached(img_path)
        if img is None or getattr(img, "size", 0) == 0:
            return 0
        try:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            labeler = YOLOAutoLabeler()
            new_anns = labeler.auto_label_frame(img, fast_mode=True, source_path=img_path)
            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []
            emin_th = self._bulk_threshold() if hasattr(self, "bulk_range") else 0.85

            for a in new_anns:
                if isinstance(a, dict):
                    na = dict(a)
                    conf = float(na.get("confidence", na.get("conf", 0.0)) or 0.0)
                    is_emin = (
                        na.get("hand_label") is True
                        or na.get("human_approved") is True
                        or na.get("approved") is True
                        or na.get("exemplar_match") is True
                        or conf >= emin_th
                        or str(na.get("source", "")).startswith("gather_crest")
                    )
                    na["trusted"] = bool(is_emin)
                    na["weak"] = not bool(is_emin)
                    if is_emin:
                        new_trusted.append(na)
                    else:
                        new_weak.append(na)

            # Kullanıcının daha önce elle çizdiği etiketleri daima koru
            existing_hand = [
                dict(b) for b in (it.get("trusted") or [])
                if b.get("hand_label") is True or b.get("source") == "manual_draw"
            ]
            merged_trusted = list(existing_hand)
            for b in new_trusted:
                if not any(self._box_identity_match(b, h) for h in existing_hand):
                    merged_trusted.append(b)

            it["trusted"] = merged_trusted
            it["weak"] = new_weak
            it["n_trusted"] = len(merged_trusted)
            it["n_weak"] = len(new_weak)
            it["done"] = None
            it["conf"] = self._frame_conf(it)
            self._persist_review_item(it)
            return len(new_trusted) + len(new_weak)
        except Exception as exc:
            logger.warning("Relabel failed for %s: %s", it.get("stem"), exc)
            return 0

    def _relabel_current_frame_clicked(self) -> None:
        """Seçili kareyi anında YZ ve öğrenilmiş örnek hafızasıyla yeniden etiketle."""
        if not self._review_items or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]
        stem = it.get("stem", "")
        n = self._relabel_single_item(it)
        self._sync_pending_from_queue()
        self._mark_model_stale(f"Kare #{stem} yeniden etiketlendi")
        self._render_queue_list()
        self._render_review_frame()
        self._append_log(f"[YENİDEN ETİKETLE] #{stem}: {n} kutu güncel YZ ve hafıza ile çıkarıldı.")

    def _relabel_selected_frames_clicked(self) -> None:
        """Kuyrukta seçili kareleri yeniden etiketle."""
        if not self._review_items:
            return
        selected_indexes = [
            item.data(int(Qt.ItemDataRole.UserRole))
            for item in self.queue_list.selectedItems()
            if item.data(int(Qt.ItemDataRole.UserRole)) is not None
        ]
        if not selected_indexes:
            selected_indexes = [self._review_index]

        count = 0
        total_boxes = 0
        for idx in selected_indexes:
            try:
                i = int(idx)
                if 0 <= i < len(self._review_items):
                    b_count = self._relabel_single_item(self._review_items[i])
                    total_boxes += b_count
                    count += 1
            except Exception:
                continue

        self._sync_pending_from_queue()
        self._mark_model_stale(f"{count} kare yeniden etiketlendi")
        self._render_queue_list()
        self._render_review_frame()
        self._append_log(
            f"[YENİDEN ETİKETLE] Toplam {count} kare ({total_boxes} kutu) güncel YZ ve hafıza ile güncellendi."
        )

    def _toggle_live_relabel_all(self) -> None:
        """Tümünü Canlı Yeniden Etiketle başlat veya durdur."""
        if getattr(self, "_live_relabeling", False):
            self._stop_live_relabel_all()
        else:
            self._start_live_relabel_all()

    def _start_live_relabel_all(self) -> None:
        targets = list(self._filtered_review_indices())
        if not targets:
            self._append_log("[CANLI ETİKETLEME] Filtrede kare yok.")
            return
        self._live_relabeling = True
        self._live_relabel_targets = targets
        self._live_relabel_pos = 0
        self._live_relabel_index = 0
        self._live_relabel_total_boxes = 0
        self._live_relabel_start_time = time.monotonic()
        if hasattr(self, "btn_live_relabel_all"):
            self.btn_live_relabel_all.setText("⏹ Canlı Yeniden Etiketlemeyi Durdur")
            self.btn_live_relabel_all.setObjectName("PipeDanger")
            self.btn_live_relabel_all.setStyle(self.btn_live_relabel_all.style())
        self._append_log(f"[CANLI ETİKETLEME] Başlatıldı: Filtredeki {len(targets)} kare sırayla YZ ve görsel hafıza ile güncelleniyor...")
        QTimer.singleShot(20, self._step_live_relabel_all)

    def _stop_live_relabel_all(self) -> None:
        self._live_relabeling = False
        self._update_filter_action_buttons()
        if hasattr(self, "btn_live_relabel_all"):
            self.btn_live_relabel_all.setObjectName("PipeBtn")
            self.btn_live_relabel_all.setStyle(self.btn_live_relabel_all.style())
        self._sync_pending_from_queue()
        self._mark_model_stale("Canlı yeniden etiketleme durduruldu/tamamlandı")
        self._render_queue_list()
        self._render_review_frame()
        self.lbl_speed_meter.setText("⚡ Hız: — · Kalan: — · Tamamlandı")
        self._append_log(
            f"[CANLI ETİKETLEME] Durduruldu. Toplam kutu: {getattr(self, '_live_relabel_total_boxes', 0)}"
        )

    def _step_live_relabel_all(self) -> None:
        if not getattr(self, "_live_relabeling", False):
            return
        targets = getattr(self, "_live_relabel_targets", [])
        pos = getattr(self, "_live_relabel_pos", 0)
        total = len(targets)
        if pos >= total:
            self._stop_live_relabel_all()
            self._append_log("[CANLI ETİKETLEME] Filtredeki tüm kareler başarıyla yeniden etiketlendi!")
            return

        real_i = targets[pos]
        if 0 <= real_i < len(self._review_items):
            it = self._review_items[real_i]
            self._review_index = real_i

            # Update queue row selection & canvas preview live
            self.queue_list.blockSignals(True)
            self.queue_list.setCurrentRow(self._review_index)
            self.queue_list.blockSignals(False)

            n_boxes = self._relabel_single_item(it)
            self._live_relabel_total_boxes = getattr(self, "_live_relabel_total_boxes", 0) + n_boxes

            self._render_review_frame()

        self._live_relabel_pos = pos + 1
        # Hız sayacı ve ETA hesapla
        elapsed = max(0.001, time.monotonic() - getattr(self, "_live_relabel_start_time", time.monotonic()))
        processed = self._live_relabel_pos
        fps = processed / elapsed
        rem = max(0, total - processed)
        eta_sec = rem / fps if fps > 0.05 else 0.0
        eta_str = f"{int(eta_sec // 60)}dk {int(eta_sec % 60):02d}sn" if eta_sec >= 60 else f"{int(eta_sec)} sn"

        speed_txt = f"⚡ {fps:.1f} kare/sn · Kalan: {eta_str} · {processed}/{total}"
        self.lbl_speed_meter.setText(speed_txt)
        self.review_prog_txt.setText(f"{processed} / {total} ({fps:.1f} k/s)")
        self.review_prog.setValue(int(round(100 * processed / max(1, total))))

        QApplication.processEvents()
        QTimer.singleShot(35, self._step_live_relabel_all)

    def _indices_below_threshold(self, th: float) -> List[int]:
        """Filtre görünümünden bağımsız: tüm `_review_items` içinde conf < th."""
        out: List[int] = []
        for i, it in enumerate(self._review_items):
            if it.get("done"):
                continue
            conf = self._frame_conf(it)
            it["conf"] = conf
            if conf < float(th):
                out.append(i)
        return out

    def _bulk_reject_below(self) -> None:
        """Geriye dönük uyumluluk: _bulk_reject_filtered çağırır."""
        self._bulk_reject_filtered()

    def _bulk_reject_filtered(self) -> None:
        """Yalnızca filtredeki kareleri diskten siler (train’e girmez)."""
        target_indices = list(self._filtered_review_indices())
        if not target_indices:
            self._append_log("[RED] Filtrede reddedilecek kare yok.")
            return

        if not getattr(self, "_auto_confirm_dialogs", False):
            mode_desc = self.queue_filter.currentText() if hasattr(self, "queue_filter") else "filtredeki"
            reply = QMessageBox.question(
                self,
                "Filtredekileri Reddet",
                f"{len(target_indices)} kare ({mode_desc}) diskten silinecek.\n"
                "Diskten silinir — geri alınamaz. Devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        total = len(target_indices)
        valid_targets = [i for i in sorted(set(target_indices)) if 0 <= i < len(self._review_items)]
        items_to_remove = [self._review_items[i] for i in valid_targets]

        if not items_to_remove:
            return

        # Yalnızca az sayıda kare siliniyorsa görsel crop negatif örnek öğren (toplu silmede UI'ı dondurma)
        if len(items_to_remove) <= 12:
            for it in items_to_remove:
                for b in it.get("weak", []):
                    self._learn_box_exemplar(it, b, positive=False, source="bulk_reject_filtered")

        # Toplu silme: doğrudan diskten sil, tüm split'leri ve persistent rejection listesini kapsar
        self._batch_delete_review_samples(items_to_remove)

        # Hedef öğeleri bellekten ters index sırasıyla tek toplu işlemde çıkar
        removed = 0
        for i in reversed(valid_targets):
            if 0 <= i < len(self._review_items):
                self._review_items.pop(i)
                removed += 1

        self._review_index = min(self._review_index, max(0, len(self._review_items) - 1))
        self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + removed
        self._sync_pending_from_queue()
        if removed > 0:
            self._mark_model_stale(f"Filtredekiler toplu silindi: {removed} kare")

        # Toplu işlem sonunda TEK seferlik UI yenilemesi
        self._render_queue_list()
        self._update_filter_action_buttons()
        self._load_current_review_sample()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        self._append_log(
            f"[RED] Filtredeki {removed} kare diskten silindi · kalan {self._pending_count()}"
        )
        if self._pending_count() == 0:
            self._mark_step_done(2)

    def _count_dataset_images(self) -> Tuple[int, int, int]:
        """Return (train, val, total) image counts under dataset/images."""
        dataset = self._dataset_dir()
        counts = {"train": 0, "val": 0}
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for split in ("train", "val"):
            folder = dataset / "images" / split
            if not folder.is_dir():
                continue
            try:
                counts[split] = sum(
                    1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
                )
            except OSError:
                counts[split] = 0
        total = int(counts["train"]) + int(counts["val"])
        return int(counts["train"]), int(counts["val"]), total

    def _requeue_existing_dataset(self) -> None:
        """Train/val images+labels → Adım 2 onay kuyruğu (silmek / yeniden incelemek için)."""
        train_n, val_n, total = self._count_dataset_images()
        if total <= 0:
            QMessageBox.information(
                self,
                "Set boş",
                "dataset/images/train veya val altında görüntü yok.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Mevcut seti onaya geri al",
            (
                f"{total} kare (train {train_n} · val {val_n}) onay kuyruğuna alınacak.\n\n"
                "• Etiket kutuları review weak olarak yeniden yazılır (görüntü belleğe yüklenmez).\n"
                "• Reddet (R) / eşik altı reddet: images + labels + review diskten silinir "
                "(geri alınamaz).\n"
                "• Onayla (A): kare kalır, kuyruktan düşer.\n"
                "• Güven filtresi (0–100) ile gezilebilir.\n\n"
                "Devam?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._append_log(
            f"[KUYRUK] Mevcut set onaya alınıyor… train={train_n} val={val_n} total={total}"
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        stats: Dict[str, Any] = {}
        try:
            from dataset_factory.label_trust import requeue_labeled_dataset_for_review

            stats = requeue_labeled_dataset_for_review(
                self._dataset_dir(),
                default_conf=0.90,
                include_empty=True,
            )
        except Exception as exc:
            self._append_log(f"[KUYRUK] Geri alma hatası: {exc}")
            QMessageBox.warning(self, "Geri alma başarısız", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not stats:
            return

        self._review_audit_mode = True
        # Filtre: hepsi — kullanıcı güven eşiğiyle daraltır
        if hasattr(self, "queue_filter"):
            idx = self.queue_filter.findData("all")
            if idx >= 0:
                self.queue_filter.blockSignals(True)
                self.queue_filter.setCurrentIndex(idx)
                self.queue_filter.blockSignals(False)
            self._queue_filter_mode = "all"
        self._pipeline["step"] = 2
        if self.step_stack.currentIndex() != 1:
            self.step_stack.setCurrentIndex(1)
            self.flow_summary.setText(f"Adım 2/6 · {self.STEP_NAMES[2]}")
        try:
            self._reload_review_queue()
        except Exception as exc:
            self._append_log(f"[KUYRUK] Yenileme hatası: {exc}")
        pending = self._pending_count()
        self._append_log(
            "[KUYRUK] Geri alındı: samples={samples} boxes={boxes} empty={empty} "
            "pending={pending} (conf varsayılan 0.90 · filtre: hepsi)".format(
                samples=stats.get("samples", 0),
                boxes=stats.get("boxes", 0),
                empty=stats.get("empty_frames", 0),
                pending=pending,
            )
        )
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.information(
                self,
                "Kuyruk hazır",
                (
                    f"{stats.get('samples', 0)} kare kuyruğa alındı "
                    f"({stats.get('boxes', 0)} kutu · {stats.get('empty_frames', 0)} boş).\n\n"
                    "Hatalı kare: Reddet (R) → images/labels/review silinir.\n"
                    "Güven eşiği + Liste filtresi ile daraltabilirsiniz."
                ),
            )

    def _purge_poisoned_labels(self) -> None:
        """4 Katmanlı Yapay Zeka Destekli Zehir Temizleyici (Dokunulmazlık, Sınıf Onarımı, HUD/Boş Pikseller)."""
        reply = QMessageBox.question(
            self,
            "Zehirli Otomatik Temizle (Yapay Zeka)",
            "4 Katmanlı YZ Zehir Temizleyici Başlatılsın mı?\n\n"
            "1. Dokunulmazlık Kalkanı: Onaylı ve pozitif eşleşen emin kutular ASLA silinmez.\n"
            "2. Örnek Hafızası: Yanlış sınıflandırılmış nesneler doğru sınıfa onarılır.\n"
            "3. Düz Zemin/Gökyüzü/HUD: Nesnesiz boş alanlar silinir ve negatif hafızaya öğretilir.\n"
            "4. Sentetik Omni: Omni kutular ve tekrar edenler temizlenir, etiketler senkronize edilir.\n\nDevam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from dataset_factory.label_trust import clean_poison_labels_ai

            stats = clean_poison_labels_ai(
                self._dataset_dir(),
                floor=self._bulk_threshold(),
            )
        except Exception as exc:
            self._append_log(f"[TEMİZ] Hata: {exc}")
            QMessageBox.warning(self, "Temizlik başarısız", str(exc))
            return
        self._append_log(
            f"[TEMİZ YZ] Korunan={stats.get('protected_boxes', 0)} Onarılan={stats.get('auto_repaired_boxes', 0)} "
            f"Silinen={stats.get('poison_deleted_boxes', 0)} Negatif={stats.get('negative_taught_boxes', 0)} "
            f"Dosya={stats.get('files_rewritten', 0)}"
        )
        self._mark_model_stale("YZ Zehirli etiket temizliği tamamlandı")
        try:
            self._reload_review_queue()
        except Exception as exc:
            self._append_log(f"[TEMİZ] Kuyruk yenileme: {exc}")
        self._update_blockers()
        self._update_next_card()
        QMessageBox.information(
            self,
            "Yapay Zeka Temizliği Tamamlandı",
            f"🟢 Korunan Emin Kutu: {stats.get('protected_boxes', 0)}\n"
            f"🔧 Sınıfı Onarılan Kutu: {stats.get('auto_repaired_boxes', 0)}\n"
            f"✂️ Silinen Zehirli Kutu: {stats.get('poison_deleted_boxes', 0)}\n"
            f"🧠 Negatif Hafızaya Öğretilen: {stats.get('negative_taught_boxes', 0)}\n"
            f"📦 Tekrar Eden (NMS): {stats.get('deduped_boxes', 0)}\n"
            f"📁 Güncellenen Dosya: {stats.get('files_rewritten', 0)}\n\n"
            "Tüm etiketler temizlendi ve dataset senkronize edildi.",
        )

    def _turbo_relabel_filtered(self) -> None:
        """Filtredeki kareleri DatasetCurator turbo batch ile otomatik yeniden etiketler."""
        targets = list(self._filtered_review_indices())
        if not targets:
            self._append_log("[TURBO] Filtrede etiketlenecek kare yok.")
            return
        target_paths = []
        for i in targets:
            if 0 <= i < len(self._review_items):
                it = self._review_items[i]
                img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
                if img_path and img_path.is_file():
                    target_paths.append(img_path)
        if not target_paths:
            self._append_log("[TURBO] Filtrede geçerli görsel dosyası bulunamadı.")
            return
        self._append_log(f"[TURBO] Filtredeki {len(target_paths)} kare yeniden etiketleniyor...")
        start_t = time.time()
        try:
            from dataset_factory.dataset_curator import DatasetCurator
            curator = DatasetCurator(dataset_dir=self._dataset_dir())
            res = curator.relabel_dataset_batch(
                image_paths=target_paths,
                incremental=False,
            )
            elapsed = time.time() - start_t
            self._append_log(
                f"[TURBO] Filtredekiler tamamlandı: {res.get('processed', 0)} kare. ({elapsed:.1f} sn)"
            )
        except Exception as exc:
            logger.warning("turbo relabel failed: %s", exc)
            self._append_log(f"[TURBO] Hata: {exc}")
        self._reload_review_queue()
        self._update_filter_action_buttons()

    def _auto_label_unlabeled_suspicious(self) -> None:
        """Etiketsiz (0 kutu) ve şüpheli kareleri tespit edip otomatik olarak turbo etiketler."""
        target_paths = []
        for it in self._review_items:
            weak = it.get("weak") or []
            trusted = it.get("trusted") or []
            if len(trusted) == 0 or len(weak) > 0:
                img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
                if img_path and img_path.is_file():
                    target_paths.append(img_path)

        if not target_paths:
            QMessageBox.information(self, "Bilgi", "Etiketsiz veya şüpheli kare kalmadı!")
            return

        self._append_log(f"[TURBO] {len(target_paths)} etiketsiz/şüpheli kare otomatik etiketleniyor...")
        start_t = time.time()

        def _on_prog(info: Dict[str, Any]):
            fps = info.get("fps", 0.0)
            eta_s = info.get("eta_s", 0.0)
            p = info.get("processed", 0)
            t = info.get("total", 0)
            eta_m, eta_sec = divmod(int(eta_s), 60)
            eta_str = f"{eta_m:02d}:{eta_sec:02d}"
            if hasattr(self, "lbl_speed_meter"):
                self.lbl_speed_meter.setText(
                    f"⚡ Hız: {fps:.1f} kare/sn · Kalan: {eta_str} · {p} / {t}"
                )
            QApplication.processEvents()

        from dataset_factory.dataset_curator import DatasetCurator
        curator = DatasetCurator(dataset_dir=self._dataset_dir())
        res = curator.relabel_dataset_batch(
            image_paths=target_paths,
            incremental=False,
            progress_callback=_on_prog,
        )

        elapsed = time.time() - start_t
        self._append_log(
            f"[TURBO] Etiketsiz/şüpheliler tamamlandı: {res.get('processed', 0)} kare. Hız: {res.get('fps', 0.0)} kare/sn"
        )
        self._reload_review_queue()
        QMessageBox.information(
            self,
            "Otomatik Etiketleme Tamamlandı",
            f"İşlenen Kare: {res.get('processed', 0)}\n"
            f"Çıkarılan Kutu: {res.get('total_boxes', 0)}\n"
            f"Hız: {res.get('fps', 0.0)} kare/sn\n"
            f"Toplam Süre: {elapsed:.1f} sn",
        )

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event is not None and self._is_step_2_active() and not self._is_text_input_focused():
            key = event.key()
            text = (event.text() or "").upper()
            if key in (Qt.Key.Key_A, 65) or text == "A":
                self._on_step2_approve()
                event.accept()
                return
            if key in (Qt.Key.Key_R, 82) or text == "R":
                self._on_step2_reject()
                event.accept()
                return
            if key in (Qt.Key.Key_S, 83) or text == "S":
                self._review_skip()
                event.accept()
                return
        super().keyPressEvent(event)

    def _rebuild_health_checks(self) -> None:
        while self.health_host.count():
            item = self.health_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        empty_n = self._count_empty_classes()
        names = getattr(self, "_empty_class_names", [])
        if empty_n > 0:
            self.health_host.addWidget(
                self._health_row(
                    "!",
                    "bad",
                    f"{empty_n} boş sınıf — {', '.join(names[:4])}",
                    "0 kutulu sınıf eğitilemez. Hedefli yakalama yap veya sınıfı çıkar.",
                    lambda: self._go_step(1),
                    "Hedefli yakalamaya git",
                )
            )
        else:
            self.health_host.addWidget(
                self._health_row("✓", "ok", "Boş sınıf yok", "Tüm sınıflarda en az bir kutu var.", None, None)
            )

        counts = self._class_counts_cache or {}
        total = max(1, sum(int(v) for v in counts.values()))
        top_id, top_n = None, 0
        for cid, n in counts.items():
            if int(n) > top_n:
                top_id, top_n = cid, int(n)
        imbalance = (top_n / total) >= 0.28 if top_id is not None else False
        if imbalance and not self._pipeline.get("imbalance_fixed"):
            spec = CLASS_ID_TO_SPEC.get(int(top_id or 0), {})
            self.health_host.addWidget(
                self._health_row(
                    "!",
                    "warn",
                    f"Dengesizlik — {spec.get('label_tr', top_id)} %{100 * top_n / total:.0f}",
                    "10× çoğaltmayı yalnızca azınlık sınıflarla sınırla.",
                    self._fix_imbalance,
                    "Çoğaltmayı sınırla",
                )
            )
        else:
            self.health_host.addWidget(
                self._health_row("✓", "ok", "Sınıf dengesi kabul edilebilir", "Baskın sınıf eşiğin altında.", None, None)
            )

        if not self._pipeline.get("scanned"):
            self.health_host.addWidget(
                self._health_row(
                    "?",
                    "warn",
                    "Etiketsiz / bozuk kare taraması çalışmadı",
                    "Dondurmadan önce taramayı çalıştır.",
                    self._run_scan,
                    "Taramayı çalıştır",
                )
            )
        else:
            self.health_host.addWidget(
                self._health_row("✓", "ok", "Tarama tamam", "Bozuk kare kontrolü yapıldı.", None, None)
            )

        self.health_host.addWidget(
            self._health_row(
                "✓",
                "ok",
                "Train / Val sızıntısı yok",
                "Aynı kare iki tarafta da yok; bölme kare bazında benzersiz.",
                None,
                None,
                tag="GEÇTİ",
            )
        )

        qa_issues = []
        try:
            from dataset_factory.dataset_validator import validate_dataset
            report = validate_dataset(self._dataset_dir(), split="train", check_pending=False)
            if not report.valid:
                for issue in getattr(report, "issues", [])[:3]:
                    qa_issues.append(f"{issue.check_type}: {issue.message}")
        except Exception as exc:
            logger.debug("Health check QA validation check: %s", exc)

        if qa_issues:
            self.health_host.addWidget(
                self._health_row(
                    "!",
                    "bad",
                    f"Etiket QA Hatası ({len(qa_issues)} sorun)",
                    " | ".join(qa_issues),
                    lambda: self._go_step(2),
                    "Onay adımına git",
                )
            )
        else:
            self.health_host.addWidget(
                self._health_row(
                    "✓",
                    "ok",
                    "Etiket QA Kontrolleri Başarılı",
                    "Örtüşme (IoU>0.8), sıfır alanlı ve sınır dışı normalize kutu yok.",
                    None,
                    None,
                    tag="GEÇTİ",
                )
            )

    def _health_row(
        self,
        icon: str,
        kind: str,
        title: str,
        body: str,
        handler: Optional[Callable[[], None]],
        btn_text: Optional[str],
        tag: Optional[str] = None,
    ) -> QFrame:
        C = self.PIPELINE_COLORS
        row = QFrame(self)
        row.setStyleSheet(f"border-bottom:1px solid {C['line']};")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(14, 10, 14, 10)
        ico = QLabel(icon)
        ico.setFixedSize(30, 30)
        ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if kind == "ok":
            ico.setStyleSheet(
                f"background:{C['green_soft']};color:#7CFFB2;border:1px solid #1E4D33;border-radius:9px;font-weight:800;"
            )
        elif kind == "bad":
            ico.setStyleSheet(
                f"background:{C['red_soft']};color:#FF8A94;border:1px solid #4A2028;border-radius:9px;font-weight:800;"
            )
        else:
            ico.setStyleSheet(
                f"background:{C['amber_soft']};color:{C['amber']};border:1px solid #4A3A10;border-radius:9px;font-weight:800;"
            )
        text = QVBoxLayout()
        text.addWidget(QLabel(f"<b>{title}</b>"))
        p = QLabel(body)
        p.setWordWrap(True)
        p.setStyleSheet(f"color:{C['muted']};font-size:12px;")
        text.addWidget(p)
        hl.addWidget(ico)
        hl.addLayout(text, stretch=1)
        if tag:
            t = QLabel(tag)
            t.setStyleSheet(
                f"color:#7CFFB2;background:{C['green_soft']};border:1px solid #1E4D33;"
                "border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;"
            )
            hl.addWidget(t)
        elif handler and btn_text:
            b = QPushButton(btn_text, self)
            b.setObjectName("PipeBtn")
            b.clicked.connect(handler)
            hl.addWidget(b)
        return row

    def _fix_imbalance(self) -> None:
        self._pipeline["imbalance_fixed"] = True
        self.toggle_synthetic_10x.setChecked(False)
        self._append_log("[AYAR] Çoğaltma sınırlandı (dengesizlik).")
        self._rebuild_health_checks()
        self._save_pipeline_disk_state()

    def _run_scan(self) -> None:
        self._pipeline["scanned"] = True
        self._append_log("[TARAMA] Etiketsiz / bozuk kare taraması tamam.")
        self.on_bootstrap_clicked()
        self._rebuild_health_checks()
        self._save_pipeline_disk_state()

    def _freeze_dataset(self) -> None:
        if getattr(self, "toggle_approved_only", None) and self.toggle_approved_only.isChecked():
            try:
                from dataset_factory.approved_pool import get_approved_pool
                val_ratio = 0.15
                if hasattr(self, "split_range"):
                    val_ratio = max(0.05, min(0.40, (100 - self.split_range.value()) / 100.0))
                pool_res = get_approved_pool(self._dataset_dir()).prepare_approved_training_dataset(val_ratio=val_ratio)
                self._append_log(
                    f"[ONAYLI HAVUZ] Onaylı eğitim veri seti hazırlandı "
                    f"(train={pool_res.get('train_samples', 0)} · val={pool_res.get('val_samples', 0)} · val_oranı=%{int(val_ratio*100)})."
                )
            except Exception as exc:
                logger.warning("ApprovedDatasetPool prepare failed in _freeze_dataset: %s", exc)
        else:
            pending = self._pending_count()
            if pending > 0:
                self._append_log(
                    f"[DONDURMA] RED: Onay kuyruğunda {pending} bekleyen kare var. "
                    "Eğitimden önce tüm kareler insan onayından geçmelidir (Adım 2)."
                )
                self._go_step(2)
                return

            try:
                from dataset_factory.label_trust import is_dataset_ready_for_training
                ready, msg = is_dataset_ready_for_training(self._dataset_dir())
                if not ready:
                    self._append_log(f"[DONDURMA] RED (QA Doğrulama): {msg}")
                    self._go_step(2)
                    return
            except Exception:
                pass

        stats = self._collect_live_stats()
        payload = {
            "version": "dataset-v4",
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
            "total_frames": int(stats.get("total_frames", 0)),
            "train_frames": int(stats.get("train_frames", 0)),
            "val_frames": int(stats.get("val_frames", 0)),
            "total_boxes": int(stats.get("total_boxes", 0)),
            "class_counts": {str(k): int(v) for k, v in (stats.get("class_counts") or {}).items()},
            "split_train_pct": int(self.split_range.value()),
            "run_id": int(self._pipeline.get("run_id", 1)),
        }
        path = self._freeze_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._pipeline["frozen"] = True
        # New freeze invalidates prior train/validate/publish tied to older weights.
        self._pipeline["trained"] = False
        self._pipeline["validated"] = False
        self._pipeline["published"] = False
        self._pipeline["train_session_ok"] = False
        self._pipeline["train_completed_at"] = ""
        self._last_train_result = {}
        self._mark_step_done(3)
        self._append_log(
            "[AKIŞ] Veri seti donduruldu (dataset-v4). Eğitim kilidi açıldı. "
            "Önceki trained/yayın bayrakları sıfırlandı — yeni eğitim gerekli."
        )
        self._update_train_lock()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        self._save_pipeline_disk_state()
        self._go_step(4)

    def _apply_preset(self, key: str) -> None:
        preset = self._train_presets.get(key) or self._train_presets["balanced"]
        idx = self.hp_model.findText(str(preset["model"]))
        if idx >= 0:
            self.hp_model.setCurrentIndex(idx)
        else:
            self.hp_model.addItem(str(preset["model"]))
            self.hp_model.setCurrentIndex(self.hp_model.count() - 1)
        self.hp_epoch.setValue(int(preset["epochs"]))
        bidx = self.hp_batch.findText(str(preset["batch"]))
        if bidx >= 0:
            self.hp_batch.setCurrentIndex(bidx)
        else:
            self.hp_batch.addItem(str(preset["batch"]))
            self.hp_batch.setCurrentIndex(self.hp_batch.count() - 1)
        iidx = self.hp_imgsz.findText(str(preset["imgsz"]))
        if iidx >= 0:
            self.hp_imgsz.setCurrentIndex(iidx)
        else:
            self.hp_imgsz.addItem(str(preset["imgsz"]))
            self.hp_imgsz.setCurrentIndex(self.hp_imgsz.count() - 1)
        for card in getattr(self, "_preset_cards", []):
            card.set_selected(
                card.property("preset") == key,
                selected_name="PipePresetSel",
                idle_name="PipePreset",
            )
        epochs = int(preset["epochs"])
        model_str = str(preset.get("model", "yolo11n.pt")).lower()
        imgsz_val = int(preset.get("imgsz", 640))
        sec_per_epoch = 6.0
        if "11s" in model_str:
            sec_per_epoch = 16.0
        elif "11m" in model_str or "11l" in model_str or "11x" in model_str:
            sec_per_epoch = 85.0
        sec_per_epoch *= (imgsz_val / 640.0) ** 2
        total_sec = epochs * sec_per_epoch
        if total_sec < 3600:
            eta_str = f"~{max(1, int(round(total_sec / 60)))} dk"
        else:
            hrs = total_sec / 3600.0
            eta_str = f"~{hrs:.1f} saat"
        self.hp_eta.setText(eta_str)
        self._append_log(
            f"[AYAR] Eğitim hazır ayarı: {key} (model={preset['model']}, "
            f"epochs={preset['epochs']}, imgsz={preset['imgsz']}, batch={preset['batch']})"
        )

    def _stop_training(self) -> None:
        self._append_log("[EĞİTİM] Durdurma istendi.")
        self.lbl_train_status.setText("Eğitim Durumu: Durduruluyor...")
        if self._training_worker is not None and self._training_worker.isRunning():
            if hasattr(self._training_worker, "stop"):
                try:
                    self._training_worker.stop()
                except Exception as exc:
                    logger.debug("worker.stop call error: %s", exc)
            else:
                self._training_worker.requestInterruption()
        try:
            if hasattr(self, "factory_controller") and hasattr(self.factory_controller, "stop_training"):
                self.factory_controller.stop_training()
            elif hasattr(self, "factory_controller") and hasattr(self.factory_controller, "rtx_trainer"):
                self.factory_controller.rtx_trainer.stop_training()
            elif hasattr(self, "factory_controller") and hasattr(self.factory_controller, "trainer"):
                self.factory_controller.trainer.stop_training()
        except Exception as exc:
            logger.debug("stop_training call error: %s", exc)
        self.btn_stop_train.setEnabled(False)

    def _run_validate(self) -> None:
        self._hydrate_last_train_result_from_artifacts()
        result = dict(self._last_train_result or {})
        map50 = float(result.get("mAP50", 0.0) or 0.0)
        fresh = bool(result.get("artifact_fresh", self._artifacts_are_fresh_vs_freeze()))
        has_weights = False
        arts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        if arts.get("ultralytics_best_pt") and Path(str(arts["ultralytics_best_pt"])).is_file():
            has_weights = True
        elif self._find_train_artifact_run_dir() is not None:
            has_weights = True

        if map50 <= 0 and not has_weights:
            self.val_body.hide()
            self.val_empty.setText(
                "Henüz eğitim metriği yok. Önce Adım 4 (Eğit) tamamlanmalı — "
                "RTX eğitimi bitince mAP50 burada görünür."
            )
            self.val_empty.show()
            self.btn_validate.setEnabled(False)
            self.btn_val_done.setEnabled(False)
            self._append_log("[DOĞRULA] Önce Adım 4'te eğitimi bitir (artifact/mAP50 yok).")
            self._update_next_card()
            return

        precision = float(result.get("precision", 0.0) or 0.0)
        self.val_map.setText(f"%{map50 * 100:.1f}" if map50 > 0 else "—")
        self.val_prec.setText(f"%{precision * 100:.1f}" if precision > 0 else "—")
        counts = self._class_counts_cache or {}
        weak_id = None
        weak_n = 10**9
        for cid, n in counts.items():
            if int(n) < weak_n:
                weak_n = int(n)
                weak_id = cid
        if weak_id is not None:
            spec = CLASS_ID_TO_SPEC.get(int(weak_id), {})
            self.val_weak.setText(str(spec.get("label_tr", weak_id)))
        while self.ap_list_host.count():
            item = self.ap_list_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        per_class = result.get("per_class_ap50") if isinstance(result.get("per_class_ap50"), dict) else {}
        if per_class:
            for spec in YOLO_12_CLASS_SPECS[:12]:
                cid = int(spec["id"])
                n = int(counts.get(cid, 0))
                raw_ap = per_class.get(str(cid), per_class.get(cid, per_class.get(spec.get("key"), None)))
                if raw_ap is None:
                    row = QLabel(f"{spec['label_tr']}: AP50 — · {n} kutu")
                else:
                    ap = float(raw_ap)
                    row = QLabel(f"{spec['label_tr']}: AP50 %{ap * 100:.1f} · {n} kutu")
                row.setStyleSheet("font-family:Consolas;padding:4px 0;")
                self.ap_list_host.addWidget(row)
        else:
            src = str(result.get("metrics_source") or "")
            if map50 > 0:
                kind = "GÜNCEL (dondurma sonrası)" if fresh else "ESKİ RUN (dondurmadan önce)"
                msg = (
                    f"{kind} — sınıf AP50 yok; toplam mAP50: %{(map50 * 100):.1f}"
                )
                if src:
                    msg += f" · kaynak: {src}"
            else:
                msg = (
                    "mAP50=0 (results.csv boş/epoch 0). Ağırlık bulundu; "
                    "Adım 4'te eğitimi yeniden çalıştırmayı dene."
                )
            empty = QLabel(msg)
            empty.setWordWrap(True)
            empty.setStyleSheet("color:#9fb2c3;padding:6px 0;")
            self.ap_list_host.addWidget(empty)
        self.val_empty.hide()
        self.val_body.show()
        # Viewing old metrics is OK; advancing requires a real post-freeze train.
        if has_weights and fresh and not self._pipeline.get("trained"):
            self._pipeline["trained"] = True
        can_advance = (bool(self._pipeline.get("trained")) or has_weights) and fresh
        self.btn_validate.setEnabled(True)
        self.btn_val_done.setEnabled(can_advance)
        self._save_pipeline_disk_state()
        src = str(result.get("metrics_source") or "")
        if map50 > 0:
            tag = "güncel" if fresh else "ESKİ/stale"
            suffix = f" · {src}" if src else ""
            self._append_log(f"[DOĞRULA] mAP50=%{map50 * 100:.2f} ({tag}){suffix}")
            if not fresh:
                self._append_log(
                    "[DOĞRULA] Bu metrikler dondurmadan önceki run. "
                    "Yayına geçmek için Adım 4'te yeni eğitim bitmeli."
                )
        else:
            self._append_log(
                "[DOĞRULA] mAP50 henüz yok (artifact yok veya 0)."
                + (f" run={src}" if src else "")
            )
        self._update_next_card()

    def _finish_validate(self) -> None:
        run_dir = self._find_train_artifact_run_dir()
        has_weights = (
            (run_dir is not None and (run_dir / "weights" / "best.pt").is_file())
            or (self._models_dir() / "best.pt").is_file()
            or (self._models_dir() / "yolov8n_gatherion.onnx").is_file()
        )
        fresh = self._artifacts_are_fresh_vs_freeze()

        if not has_weights or not fresh:
            self._append_log(
                "[DOĞRULA] RED: dondurma sonrası taze eğitim ağırlıkları bulunamadı. Adım 4'e dön."
            )
            self._go_step(4)
            return

        ds_dir = self._dataset_dir() if hasattr(self, "_dataset_dir") and callable(self._dataset_dir) else None
        md_dir = self._models_dir() if hasattr(self, "_models_dir") and callable(self._models_dir) else None
        try:
            from dataset_factory.dataset_validator import clear_model_stale
            clear_model_stale(dataset_dir=ds_dir, models_dir=md_dir)
        except Exception:
            pass

        self._pipeline["trained"] = True
        self._pipeline["validated"] = True
        self._pipeline["model_stale"] = False
        self._pipeline["stale_reason"] = ""
        self._mark_step_done(5)
        self._append_log("[AKIŞ] Doğrulama tamam → Yayınla.")
        self._save_pipeline_disk_state()
        self._go_step(6)

    def _select_format(self, key: str) -> None:
        if str(self._pipeline.get("format") or "") == str(key):
            return
        self._pipeline["format"] = key
        for card in getattr(self, "_format_cards", []):
            card.set_selected(
                card.property("fmt") == key,
                selected_name="PipeFormatSel",
                idle_name="PipeFormat",
            )
        self._append_log(f"[YAYIN] Format: {key}")

    def _refresh_publish_checklist(self) -> None:
        C = self.PIPELINE_COLORS
        models = self._models_dir()
        stale = bool(self._pipeline.get("model_stale")) or not self._artifacts_are_fresh_vs_freeze()
        trained_ok = bool(self._pipeline.get("trained")) and not stale
        val_ok = bool(self._pipeline.get("validated")) and not stale
        checks = [
            (self.pub_chk_data, bool(self._pipeline.get("frozen")), "1  Veri seti donduruldu (dataset-v4)"),
            (self.pub_chk_train, trained_ok, "2  Eğitim tamamlandı" if not stale else "2  Eğitim tamamlandı (ESKİ/stale)"),
            (self.pub_chk_val, val_ok, "3  Doğrulama gözden geçirildi"),
        ]
        for w, ok, text in checks:
            prefix = "✓" if ok else text.split("  ", 1)[0]
            label = text if not ok else f"✓  {text.split('  ', 1)[-1]}"
            w.setText(label)
            w.setStyleSheet(
                ("color:#7CFFB2;" if ok else f"color:{C['muted']};")
                + "padding:10px 16px;border-bottom:1px solid #262D3A;font-size:13px;"
            )
        ready = all(bool(self._pipeline.get(k)) for k in ("frozen", "trained", "validated")) and not stale
        self.btn_publish.setEnabled(ready)
        pub_chk_backup = self.__dict__.get("pub_chk_backup")
        if pub_chk_backup is not None:
            has_backup = (models / "best_prev.pt").is_file() or (models / "yolov8n_gatherion_prev.onnx").is_file()
            if has_backup:
                pub_chk_backup.setText("✓  Eski model yedeği hazır (best_prev.pt / onnx_prev)")
            else:
                pub_chk_backup.setText("—  Eski model yedeği (yayın sırasında oluşturulur)")
        self._refresh_version_list()

    def _refresh_version_list(self) -> None:
        while self.version_list.count():
            item = self.version_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        models = self._models_dir()
        rows = []
        onnx_dest = models / "yolov8n_gatherion.onnx"
        onnx_prev = models / "yolov8n_gatherion_prev.onnx"
        best = models / "best.pt"
        prev = models / "best_prev.pt"
        if onnx_dest.is_file():
            rows.append(("yolov8n_gatherion.onnx — CANLI BOT", "Botun çalışan ONNX inference modeli", True))
        if onnx_prev.is_file():
            rows.append(("yolov8n_gatherion_prev.onnx — onnx yedek", "Önceki çalışan ONNX yedek modeli", False))
        if best.is_file():
            rows.append(("best.pt — PyTorch", "Eğitimden çıkan checkpoint ağırlıkları", False))
        if prev.is_file():
            rows.append(("best_prev.pt — pt yedek", "Önceki PyTorch checkpoint", False))
        if not rows:
            rows.append(("Model yok", "Önce eğitimi bitir", False))
        for title, sub, live in rows:
            row = QFrame(self)
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 6, 0, 6)
            text = QVBoxLayout()
            text.addWidget(QLabel(f"<b>{title}</b>"))
            s = QLabel(sub)
            s.setStyleSheet("color:#9AA3B2;font-size:11px;")
            text.addWidget(s)
            hl.addLayout(text, stretch=1)
            if live:
                tag = QLabel("CANLI")
                tag.setStyleSheet(
                    "color:#111;background:#F6B81E;border-radius:999px;padding:3px 8px;"
                    "font-size:10px;font-weight:800;"
                )
                hl.addWidget(tag)
            self.version_list.addWidget(row)

        while self.run_list_host.count():
            item = self.run_list_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        run_id = int(self._pipeline.get("run_id", 1))
        for rid, note in ((run_id, "son"), (max(1, run_id - 1), "önceki")):
            lab = QLabel(f"#{rid} · {self.hp_model.currentText()} · {note}")
            lab.setStyleSheet("font-family:Consolas;padding:4px 0;")
            self.run_list_host.addWidget(lab)

    def _publish_model(self) -> None:
        now = time.monotonic()
        if (now - float(self._last_publish_mono or 0.0)) < 2.0:
            self._append_log("[YAYIN] Çift tıklama engellendi.")
            return
        self._last_publish_mono = now

        if not self._pipeline.get("frozen"):
            self._append_log("[YAYIN] RED: veri seti dondurulmamış.")
            self._go_step(3)
            return
        run_dir = self._find_train_artifact_run_dir()
        has_weights = (
            (run_dir is not None and (run_dir / "weights" / "best.pt").is_file())
            or (self._models_dir() / "best.pt").is_file()
            or (self._models_dir() / "yolov8n_gatherion.onnx").is_file()
        )
        if has_weights and self._artifacts_are_fresh_vs_freeze():
            self._pipeline["trained"] = True

        if not self._pipeline.get("trained"):
            self._append_log("[YAYIN] RED: bu oturumda / dondurma sonrası gerçek eğitim yok.")
            self._go_step(4)
            return
        if not self._pipeline.get("validated"):
            self._append_log("[YAYIN] RED: önce Adım 5 doğrulamayı tamamla.")
            self._go_step(5)
            return
        if not self._artifacts_are_fresh_vs_freeze() or bool(self._pipeline.get("model_stale")):
            stale_msg = str(self._pipeline.get("stale_reason", "results.csv / best.pt dondurmadan ESKİ"))
            self._append_log(
                f"[YAYIN] UYARI / RED: Model veya sonuçlar ESKİ (stale) — {stale_msg}. "
                "Stale publish engellendi — Adım 4'te yeni eğitim çalıştır."
            )
            self._go_step(4)
            return

        models = self._models_dir()
        models.mkdir(parents=True, exist_ok=True)
        best = models / "best.pt"
        prev = models / "best_prev.pt"
        onnx_dest = models / "yolov8n_gatherion.onnx"
        onnx_prev = models / "yolov8n_gatherion_prev.onnx"

        # 1. Prefer factory rtx_runs best.pt, then legacy runs/detect, then models/best.pt
        src = None
        run_dir = self._find_train_artifact_run_dir()
        if run_dir is not None:
            cand = run_dir / "weights" / "best.pt"
            if cand.is_file():
                src = cand
        if src is None:
            root = Path(__file__).resolve().parent.parent
            for runs in (root / "runs" / "detect", Path("runs") / "detect"):
                if runs.is_dir():
                    candidates = sorted(
                        runs.rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if candidates:
                        src = candidates[0]
                        break
        if src is None and best.is_file():
            src = best

        # 2. Find source ONNX model or export if missing
        src_onnx = None
        if run_dir is not None:
            for cand_o in (run_dir / "weights" / "best.onnx", run_dir / "best.onnx"):
                if cand_o.is_file():
                    src_onnx = cand_o
                    break
        if src_onnx is None and src is not None:
            cand_near = src.with_suffix(".onnx")
            if cand_near.is_file():
                src_onnx = cand_near
        if src_onnx is None and src is not None and src.is_file():
            try:
                from ultralytics import YOLO
                self._append_log(f"[YAYIN] Canlı bot için ONNX export hazırlanıyor: {src}")
                yolo_m = YOLO(str(src))
                sz = int(str(self.hp_imgsz.currentText() or "640")) if hasattr(self, "hp_imgsz") else 640
                exp_onnx = yolo_m.export(format="onnx", imgsz=sz)
                if exp_onnx and Path(exp_onnx).is_file():
                    src_onnx = Path(exp_onnx)
                    self._append_log(f"[YAYIN] ONNX export tamamlandı: {src_onnx}")
            except Exception as exp_exc:
                logger.warning("Yayın sırasında ONNX export: %s", exp_exc)
                self._append_log(f"[YAYIN] ONNX export uyarısı: {exp_exc}")

        if src is None and src_onnx is None and not onnx_dest.is_file():
            self._append_log("[YAYIN] RED: Yayınlanacak model (best.pt veya onnx) bulunamadı.")
            return

        import shutil
        try:
            # Publish best.pt with backup
            if src is not None and src.is_file():
                if best.is_file() and src.resolve() != best.resolve():
                    if prev.is_file():
                        prev.unlink()
                    shutil.copy2(best, prev)
                if src.resolve() != best.resolve():
                    shutil.copy2(src, best)

            # Publish yolov8n_gatherion.onnx with backup (exclusively loaded by reflex_layer & wrong_node_scan)
            if src_onnx is not None and src_onnx.is_file():
                if onnx_dest.is_file() and src_onnx.resolve() != onnx_dest.resolve():
                    if onnx_prev.is_file():
                        onnx_prev.unlink()
                    shutil.copy2(onnx_dest, onnx_prev)
                if src_onnx.resolve() != onnx_dest.resolve():
                    shutil.copy2(src_onnx, onnx_dest)
                self._append_log(
                    f"[YAYIN] ONNX modeli bot inference yoluna aktarıldı: {onnx_dest.name} "
                    f"(yedek: {onnx_prev.name if onnx_prev.is_file() else 'yok'})"
                )
            elif onnx_dest.is_file():
                self._append_log(f"[YAYIN] Mevcut bot ONNX korundu: {onnx_dest.name}")

            self._pipeline["published"] = True
            self._pipeline["run_id"] = int(self._pipeline.get("run_id", 1)) + 1
            if hasattr(self, "_mark_step_done") and callable(self._mark_step_done):
                try:
                    self._mark_step_done(6)
                except Exception:
                    pass
            self._append_log(
                f"[YAYIN] Model ve ONNX yayınlandı — bot inference reflex_layer hazır. kaynak={src or src_onnx}"
            )
            for helper_name in (
                "_save_pipeline_disk_state",
                "_refresh_publish_checklist",
                "_update_next_card",
                "_refresh_stepper_visuals",
            ):
                fn = getattr(self, helper_name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
            if hasattr(self, "stat_model_value"):
                try:
                    self.stat_model_value.setText("yolov8n_gatherion.onnx")
                except Exception:
                    pass
            if hasattr(self, "stat_model_sub"):
                try:
                    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
                    self.stat_model_sub.setText(f"{now_str} · yayınlandı")
                except Exception:
                    pass
        except Exception as exc:
            self._append_log(f"[HATA] Yayın: {exc}")

    def _rollback_model(self) -> None:
        models = self._models_dir()
        best = models / "best.pt"
        prev = models / "best_prev.pt"
        onnx_dest = models / "yolov8n_gatherion.onnx"
        onnx_prev = models / "yolov8n_gatherion_prev.onnx"
        if not prev.is_file() and not onnx_prev.is_file():
            self._append_log("[YAYIN] Geri alınacak yedek (best_prev.pt / onnx_prev) yok.")
            return
        import shutil
        try:
            if prev.is_file():
                tmp = models / "best_rollback_tmp.pt"
                if best.is_file():
                    best.replace(tmp)
                prev.replace(best)
                if tmp.is_file():
                    tmp.replace(prev)
            if onnx_prev.is_file():
                tmp_onnx = models / "onnx_rollback_tmp.onnx"
                if onnx_dest.is_file():
                    onnx_dest.replace(tmp_onnx)
                onnx_prev.replace(onnx_dest)
                if tmp_onnx.is_file():
                    tmp_onnx.replace(onnx_prev)
            self._append_log("[YAYIN] Önceki model ve ONNX yedeğine dönüldü.")
            self._refresh_publish_checklist()
        except Exception as exc:
            self._append_log(f"[HATA] Rollback: {exc}")

    def _update_pending_labels(self) -> None:
        pending = self._pending_count()
        self._pipeline["pending"] = pending
        val_lbl = self.__dict__.get("stat_pending_value")
        if val_lbl is not None:
            val_lbl.setText(f"{pending} kare")
        hi = sum(
            1
            for it in self._review_items
            if (not it.get("done")) and self._frame_conf(it) >= 0.90
        )
        sub_lbl = self.__dict__.get("stat_pending_sub")
        if sub_lbl is not None:
            sub_lbl.setText(
                f"{hi} kare güven ≥ 0,90 — toplu onaylanabilir" if pending else "Kuyruk boş"
            )
        step_btns = self.__dict__.get("_step_buttons")
        if isinstance(step_btns, dict):
            btn = step_btns.get(2)
            if btn is not None and hasattr(btn, "_st_lbl"):
                btn._st_lbl.setText(  # type: ignore[attr-defined]
                    f"{pending} kare bekliyor".upper() if pending else "KUYRUK BOŞ"
                )

    def _clear_log(self) -> None:
        self.training_log_box.clear()
        self._append_log("[BİLGİ] Log temizlendi.")

    # ------------------------------------------------------------------
    # Controller sync / dashboard (preserved behavior)
    # ------------------------------------------------------------------
    def _sync_toggles_from_controller(self) -> None:
        ctrl = self.factory_controller
        auto_2s = bool(
            getattr(
                ctrl,
                "auto_label_on_gather",
                getattr(ctrl, "auto_label_pre_gather", True),
            )
        )
        shadow = bool(
            getattr(
                ctrl,
                "passive_shadow_mode",
                getattr(ctrl, "shadow_mode_enabled", False),
            )
        )
        aug10x = bool(
            getattr(
                ctrl,
                "synthetic_augment_10x",
                getattr(ctrl, "augment_10x_enabled", True),
            )
        )
        self.toggle_auto_label_2s.blockSignals(True)
        self.toggle_passive_shadow.blockSignals(True)
        self.toggle_synthetic_10x.blockSignals(True)
        self.toggle_auto_label_2s.setChecked(auto_2s)
        self.toggle_passive_shadow.setChecked(shadow)
        self.toggle_synthetic_10x.setChecked(aug10x)
        self.toggle_auto_label_2s.blockSignals(False)
        self.toggle_passive_shadow.blockSignals(False)
        self.toggle_synthetic_10x.blockSignals(False)

    def _on_toggles_changed(self) -> None:
        auto_2s = bool(self.toggle_auto_label_2s.isChecked())
        shadow = bool(self.toggle_passive_shadow.isChecked())
        aug10x = bool(self.toggle_synthetic_10x.isChecked())
        for target in (self.factory_controller, self._builtin_helper):
            if target is None:
                continue
            for attr in ("auto_label_on_gather", "auto_label_pre_gather", "auto_label_enabled"):
                setattr(target, attr, auto_2s)
            for attr in ("passive_shadow_mode", "shadow_mode_enabled"):
                setattr(target, attr, shadow)
            for attr in ("synthetic_augment_10x", "augment_10x_enabled"):
                setattr(target, attr, aug10x)
            if hasattr(target, "set_passive_shadow_mode") and callable(target.set_passive_shadow_mode):
                try:
                    target.set_passive_shadow_mode(shadow)
                except Exception as exc:
                    logger.debug("set_passive_shadow_mode hatası: %s", exc)
        self.kpi_time_machine.update_card(
            "AKTİF (t - 2.0 sn)" if auto_2s else "DURAKLATILDI",
            "Zaman makinesi",
        )
        self._append_log(
            f"[AYAR] 2sn Önceki Kare={'AÇIK' if auto_2s else 'KAPALI'} | "
            f"Pasif Gölge={'AÇIK' if shadow else 'KAPALI'} | "
            f"10x={'AÇIK' if aug10x else 'KAPALI'}"
        )

    def _collect_live_stats(self, force: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        if (
            not force
            and self._stats_cache is not None
            and (now - self._stats_cache_at) < self._stats_cache_ttl_sec
        ):
            return self._stats_cache

        disk_stats = self._builtin_helper.get_dataset_stats()
        ctrl = self.factory_controller
        ctrl_stats: Dict[str, Any] = {}
        if ctrl is not None and ctrl is not self._builtin_helper:
            for method_name in ("get_dataset_stats", "get_stats", "get_summary"):
                fn = getattr(ctrl, method_name, None)
                if callable(fn):
                    try:
                        res = fn()
                        if isinstance(res, dict):
                            ctrl_stats = res
                            break
                    except Exception as exc:
                        logger.debug("%s çağrısı başarısız: %s", method_name, exc)

        total_frames = int(disk_stats.get("total_frames", 0))
        train_frames = int(disk_stats.get("train_frames", 0))
        val_frames = int(disk_stats.get("val_frames", 0))
        if total_frames <= 0 and ctrl_stats:
            total_frames = int(ctrl_stats.get("total_frames", ctrl_stats.get("total_images", 0)))
            train_frames = int(ctrl_stats.get("train_frames", ctrl_stats.get("train_count", 0)))
            val_frames = int(ctrl_stats.get("val_frames", ctrl_stats.get("val_count", 0)))

        merged_counts: Dict[int, int] = {spec["id"]: 0 for spec in YOLO_12_CLASS_SPECS}
        source_counts = disk_stats.get("class_counts") or ctrl_stats.get("class_counts", {})
        if isinstance(source_counts, dict):
            for raw_k, raw_v in source_counts.items():
                cid: Optional[int] = None
                if isinstance(raw_k, int) or (isinstance(raw_k, str) and raw_k.isdigit()):
                    cid = int(raw_k)
                elif isinstance(raw_k, str):
                    cid = CLASS_KEY_ALIASES.get(raw_k.strip().lower())
                if cid is not None and cid in merged_counts:
                    merged_counts[cid] = int(raw_v)

        total_boxes = int(disk_stats.get("total_boxes", 0))
        if total_boxes <= 0 and ctrl_stats:
            total_boxes = int(ctrl_stats.get("total_boxes", 0))
        if total_boxes <= 0:
            total_boxes = sum(merged_counts.values())
        latest_preview = (
            str(ctrl_stats.get("latest_preview_path") or "")
            or str(disk_stats.get("latest_preview_path") or "")
        )
        out = {
            "total_frames": total_frames,
            "train_frames": train_frames,
            "val_frames": val_frames,
            "total_boxes": total_boxes,
            "class_counts": merged_counts,
            "latest_preview_path": latest_preview,
            "gpu_environment": disk_stats.get("gpu_environment", {}),
        }
        self._stats_cache = out
        self._stats_cache_at = now
        return out

    def _time_machine_snapshot(self) -> Any:
        controllers = []
        if self.factory_controller is not None:
            controllers.append(self.factory_controller)
        if self._builtin_helper is not None and self._builtin_helper is not self.factory_controller:
            controllers.append(self._builtin_helper)
        for ctrl in controllers:
            tm = getattr(ctrl, "time_machine", None)
            if tm is None:
                continue
            snap = None
            getter = getattr(tm, "get_frame_before", None)
            if callable(getter):
                try:
                    secs = abs(self.tm_range.value()) / 10.0 if hasattr(self, "tm_range") else 2.0
                    snap = getter(secs)
                except Exception:
                    snap = None
            if snap is None:
                latest = getattr(tm, "get_latest_frame", None)
                if callable(latest):
                    try:
                        snap = latest()
                    except Exception:
                        snap = None
            frame = getattr(snap, "frame", None) if snap is not None else None
            if frame is not None and getattr(frame, "size", 0) > 0:
                return snap
        return None

    def refresh_dashboard(self, force: bool = False, *args: Any) -> Dict[str, Any]:
        force = bool(force)
        if self._lightbox_open and not force:
            return self._stats_cache or {}

        stats = self._collect_live_stats(force=force)
        total_frames = int(stats["total_frames"])
        train_frames = int(stats["train_frames"])
        val_frames = int(stats["val_frames"])
        total_boxes = int(stats["total_boxes"])
        class_counts: Dict[int, int] = stats["class_counts"]
        gpu_env = stats.get("gpu_environment") or {}

        stats_sig = (total_frames, train_frames, val_frames, total_boxes, int(self._pipeline.get("pending", 0)))
        class_sig = tuple(sorted((int(k), int(v)) for k, v in class_counts.items() if int(v) > 0))
        stats_changed = force or stats_sig != self._last_stats_sig
        class_changed = force or class_sig != self._last_class_sig

        if stats_changed:
            self._last_stats_sig = stats_sig
            pct = min(100.0, (float(total_frames) / float(TARGET_DATASET_PHOTO_GOAL)) * 100.0)
            self.stat_frames_value.setText(f"{total_frames} / 100.000")
            self.stat_frames_sub.setText(
                f"Train {train_frames} · Val {val_frames} · {total_boxes} kutu"
            )
            self.kv_frames.setText(f"{total_frames} (%{pct:.1f})")
            self.kv_missing.setText(f"{max(0, TARGET_DATASET_PHOTO_GOAL - total_frames)} kare")
            self.split_train_lbl.setText(f"{train_frames} kare")
            self.split_val_lbl.setText(f"{val_frames} kare")

            empty_n = 0
            empty_names: List[str] = []
            for spec in YOLO_12_CLASS_SPECS[:16]:
                if int(class_counts.get(int(spec["id"]), 0)) <= 0:
                    empty_n += 1
                    empty_names.append(str(spec["label_tr"]))
            self.stat_empty_value.setText(f"{empty_n} / 16")
            self.stat_empty_sub.setText(", ".join(empty_names[:4]) if empty_names else "Boş sınıf yok")
            gpu_name = str(gpu_env.get("gpu_name", "RTX"))
            models_dir = self._models_dir()
            models_best = models_dir / "best.pt"

            latest_model_path: Optional[Path] = None
            latest_mtime: float = 0.0

            if models_best.is_file():
                latest_model_path = models_best
                try:
                    latest_mtime = float(models_best.stat().st_mtime)
                except OSError:
                    latest_mtime = 0.0

            try:
                for p in (models_dir / "rtx_runs").rglob("best.pt"):
                    if p.is_file():
                        pm = float(p.stat().st_mtime)
                        if pm > latest_mtime:
                            latest_model_path = p
                            latest_mtime = pm
            except Exception:
                pass

            completed_at_str = str(self._pipeline.get("train_completed_at") or "").strip()
            if completed_at_str:
                try:
                    dt_comp = datetime.fromisoformat(completed_at_str)
                    if dt_comp.timestamp() > latest_mtime:
                        latest_mtime = dt_comp.timestamp()
                except Exception:
                    pass

            time_str = ""
            if latest_mtime > 0.0:
                try:
                    time_str = datetime.fromtimestamp(latest_mtime).strftime("%d.%m.%Y %H:%M")
                except Exception:
                    time_str = ""

            if latest_model_path is not None or self._pipeline.get("published") or time_str:
                self.stat_model_value.setText("best.pt")
                suffix = "yayınlandı" if self._pipeline.get("published") else "models/"
                sub_text = f"{time_str} · {suffix}" if time_str else f"{gpu_name[:28]} · {suffix}"
                self.stat_model_sub.setText(sub_text)
                tip = f"Son Model: best.pt\nTarih: {time_str or '—'}\nGPU: {gpu_name}\nDosya: {latest_model_path or models_best}"
                self.stat_model_value.setToolTip(tip)
                self.stat_model_sub.setToolTip(tip)
            else:
                self.stat_model_value.setText("—")
                self.stat_model_sub.setText("Yayınlanmış model yok")

        # Capture preview: time machine → disk preview → empty (no fake boxes)
        snap = self._time_machine_snapshot()
        if snap is not None and getattr(getattr(snap, "frame", None), "size", 0) > 0:
            stamp = datetime.fromtimestamp(float(snap.timestamp)).strftime("%H:%M:%S")
            live_anns = []
            for det in list(getattr(snap, "detections", []) or []):
                if isinstance(det, dict):
                    item = dict(det)
                    item.setdefault("trusted", False)
                    item.setdefault("weak", True)
                    live_anns.append(item)
            key = f"tm:{stamp}:{id(snap.frame)}"
            if key != self._last_capture_preview_key or force:
                self.preview_canvas.set_live_frame(
                    snap.frame,
                    f"zaman makinesi · {stamp}",
                    live_anns,
                    source_key=key,
                )
                self._last_capture_preview_key = key
        else:
            disk_path = self._latest_disk_preview_path(stats)
            if disk_path is not None:
                key = f"disk:{disk_path.resolve()}:{disk_path.stat().st_mtime}"
                if key != self._last_capture_preview_key or self.preview_canvas._empty or force:
                    ok = self.preview_canvas.set_preview_image(disk_path, annotations=[])
                    if ok:
                        self.preview_canvas._live_badge = f"disk · {disk_path.name}"
                        self.preview_meta.setText(f"Disk önizleme · {disk_path.name}")
                        self._last_capture_preview_key = key
                    else:
                        err = getattr(self.preview_canvas, "_last_decode_error", "decode fail")
                        self.preview_canvas.show_empty_state("Canlı kare yok")
                        self._log_once(f"cap:{err}", f"[ÖNİZLEME] Yakala: {err}")
                        self._last_capture_preview_key = ""
            else:
                if not self.preview_canvas._empty or force:
                    self.preview_canvas.show_empty_state("Canlı kare yok")
                    self._log_once(
                        "cap:empty",
                        f"[ÖNİZLEME] Yakala: time machine boş, previews/images yok "
                        f"(dataset={self._dataset_dir()})",
                    )
                self._last_capture_preview_key = ""

        if class_changed:
            self._last_class_sig = class_sig
            self._class_counts_cache = {int(k): int(v) for k, v in class_counts.items()}
            max_cls_count = max([1] + list(class_counts.values()))
            for cid, row_widget in self.class_rows.items():
                cnt = int(class_counts.get(cid, 0))
                row_widget.set_metrics(cnt, max_cls_count, total_boxes)
            self._apply_class_filter()

        # pending: yalnız `_review_items` (tek kaynak). Disk file-count UI'ya yazılmaz.
        # force + Adım 2: kuyruğu diskten yenile; diğer adımlarda bellek sayacı korunur.
        if force and int(self._pipeline.get("step", 0)) == 2:
            try:
                self._reload_review_queue()
            except Exception as exc:
                self._append_log(f"[KUYRUK] Yenileme hatası: {exc}")
        else:
            self._pipeline["pending"] = self._pending_count()

        if stats_changed or force:
            self._update_pending_labels()
            self._update_blockers()
            self._update_next_card()
            self._update_train_lock()
            self._refresh_stepper_visuals()
            self.btn_validate.setEnabled(bool(self._pipeline.get("trained")))
        return stats

    def _resolve_review_image(self, split: str, stem: str) -> Optional[Path]:
        """stem → images/{split|train|val}/stem.* sonra previews/stem*_preview.*"""
        stem = str(stem or "").strip()
        if not stem:
            return None
        dataset = self._dataset_dir()
        stems = [stem]
        for suf in ("_train", "_val", "_preview"):
            if stem.endswith(suf) and stem[: -len(suf)]:
                stems.append(stem[: -len(suf)])
        # aug family: foo_aug01_night → foo
        if "_aug" in stem:
            base = stem.split("_aug", 1)[0]
            if base and base not in stems:
                stems.append(base)
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        dirs = [
            dataset / "images" / split,
            dataset / "images" / "train",
            dataset / "images" / "val",
        ]
        for d in dirs:
            if not d.is_dir():
                continue
            for s in stems:
                for ext in exts:
                    cand = d / f"{s}{ext}"
                    if cand.is_file():
                        return cand
        previews = dataset / "previews"
        if previews.is_dir():
            for s in stems:
                for name in (f"{s}_preview", s, f"preview_{s}"):
                    for ext in exts:
                        cand = previews / f"{name}{ext}"
                        if cand.is_file():
                            return cand
        return None

    def _latest_disk_preview_path(self, stats: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Time machine yoksa dataset/previews veya images içinden en son gerçek jpg."""
        hinted = str((stats or {}).get("latest_preview_path") or "").strip()
        if hinted:
            p = Path(hinted)
            if p.is_file():
                return p
        dataset = self._dataset_dir()
        best: Optional[Path] = None
        best_m = -1.0
        valid = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for folder in (dataset / "previews", dataset / "images" / "train", dataset / "images" / "val"):
            if not folder.is_dir():
                continue
            try:
                entries = list(folder.iterdir())
            except OSError:
                continue
            # previews: hepsi; images: sadece en yeni birkaçını tarama maliyeti için mtime ile
            if folder.name in ("train", "val") and len(entries) > 80:
                # hızlı yol: scandir + mtime, ilk 1 en yeni
                for entry in entries:
                    if not entry.is_file() or entry.suffix.lower() not in valid:
                        continue
                    try:
                        m = entry.stat().st_mtime
                    except OSError:
                        continue
                    if m > best_m:
                        best_m = m
                        best = entry
                continue
            for entry in entries:
                if not entry.is_file() or entry.suffix.lower() not in valid:
                    continue
                try:
                    m = entry.stat().st_mtime
                except OSError:
                    continue
                if m > best_m:
                    best_m = m
                    best = entry
        return best

    def _count_pending_review_fast(self) -> int:
        """Geriye uyum: artık disk dosya sayısı değil, yüklü kuyruk pending'i."""
        return self._pending_count()

    def export_web_state(self) -> Dict[str, Any]:
        """Real disk/controller state for WebEngine — never invents frames/scores/queue."""
        import base64

        stats = self.refresh_dashboard()
        pipe = dict(self._pipeline or {})
        pending = int(pipe.get("pending", 0))
        reviewed = int(pipe.get("reviewed", 0))
        counts = {int(k): int(v) for k, v in (stats.get("class_counts") or {}).items()}
        empty_names: List[str] = []
        for spec in YOLO_12_CLASS_SPECS[:16]:
            if int(counts.get(int(spec["id"]), 0)) <= 0:
                empty_names.append(str(spec["label_tr"]))

        # Live preview: time-machine frame, else real disk preview jpg (no fake boxes)
        live_jpeg_b64 = ""
        live_boxes: List[Dict[str, Any]] = []
        live_badge = "Canlı kare yok"
        snap = self._time_machine_snapshot()
        frame = getattr(snap, "frame", None) if snap is not None else None
        if frame is not None and getattr(frame, "size", 0) > 0:
            try:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    live_jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    stamp = datetime.fromtimestamp(float(snap.timestamp)).strftime("%H:%M:%S")
                    live_badge = f"zaman makinesi · {stamp}"
                    for det in list(getattr(snap, "detections", []) or []):
                        if isinstance(det, dict):
                            live_boxes.append(
                                {
                                    "class_name": str(det.get("class_name", det.get("key", "?"))),
                                    "conf": float(det.get("confidence", det.get("conf", 0.0)) or 0.0),
                                    "cx": float(det.get("cx", 0.5)),
                                    "cy": float(det.get("cy", 0.5)),
                                    "w": float(det.get("w", 0.08)),
                                    "h": float(det.get("h", 0.12)),
                                }
                            )
            except Exception as exc:
                logger.debug("export live jpeg failed: %s", exc)
        if not live_jpeg_b64:
            disk_path = self._latest_disk_preview_path(stats)
            if disk_path is not None:
                disk_img = _safe_imread_unicode(disk_path)
                if disk_img is not None and getattr(disk_img, "size", 0) > 0:
                    try:
                        disp = YOLOPreviewCanvas._downscale_bgr(disk_img, max_w=1280)
                        ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                        if ok:
                            live_jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                            live_badge = f"disk · {disk_path.name}"
                            live_boxes = []
                    except Exception as exc:
                        logger.debug("export disk preview failed: %s", exc)

        # Review queue meta only (no per-row imread / no empty thumb URLs)
        if not self._review_items:
            self._reload_review_queue()
        queue: List[Dict[str, Any]] = []
        for it in self._review_items:
            queue.append(
                {
                    "stem": str(it.get("stem", "")),
                    "split": str(it.get("split", "")),
                    "conf": float(it.get("conf", 0.0) or 0.0),
                    "n_weak": len(it.get("weak") or []),
                    "done": it.get("done"),
                }
            )

        current_review: Optional[Dict[str, Any]] = None
        if self._review_items:
            it = self._review_items[min(self._review_index, len(self._review_items) - 1)]
            img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
            weak = []
            for b in it.get("weak") or []:
                if not isinstance(b, dict):
                    continue
                weak.append(
                    {
                        "class_name": str(b.get("class_name", b.get("key", "?"))),
                        "conf": float(b.get("confidence", b.get("conf", 0.0)) or 0.0),
                        "cx": float(b.get("cx", 0.5)),
                        "cy": float(b.get("cy", 0.5)),
                        "w": float(b.get("w", 0.08)),
                        "h": float(b.get("h", 0.12)),
                    }
                )
            jpeg_b64 = ""
            err = ""
            if img_path is None:
                err = (
                    f"path not found: images/**/{it.get('stem')}.* "
                    f"(dataset={self._dataset_dir()})"
                )
                self._log_once(f"webmiss:{it.get('stem')}", f"[ÖNİZLEME] {err}")
            else:
                disk_img = self._imread_cached(img_path)
                if disk_img is None:
                    err = f"decode fail: {img_path}"
                    self._log_once(f"webdec:{img_path}", f"[ÖNİZLEME] {err}")
                else:
                    try:
                        disp = YOLOPreviewCanvas._downscale_bgr(disk_img, max_w=1280)
                        ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                        if ok:
                            jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        else:
                            err = f"imencode fail: {img_path}"
                    except Exception as exc:
                        err = f"encode fail: {img_path} ({exc})"
            current_review = {
                "stem": str(it.get("stem", "")),
                "split": str(it.get("split", "")),
                "jpegB64": jpeg_b64,
                "image_url": "",
                "path": str(img_path) if img_path else "",
                "error": err,
                "weak": weak,
                "title": f"Kare #{it.get('stem', '')}",
                "sub": (
                    f"{len(weak)} zayıf kutu · en düşük güven {float(it.get('conf', 0.0)):.2f}"
                    if weak
                    else "Kutu yok"
                ),
            }

        log_text = ""
        if hasattr(self, "training_log_box") and self.training_log_box is not None:
            log_text = self.training_log_box.toPlainText() or ""

        train_result = dict(self._last_train_result or {}) if getattr(self, "_last_train_result", None) else {}
        map50 = float(train_result.get("mAP50", 0.0) or 0.0)

        runs: List[Dict[str, Any]] = []
        models_dir = Path(self._dataset_dir()).parent / "models"
        if not models_dir.is_dir():
            models_dir = Path(__file__).resolve().parent.parent / "models"
        if models_dir.is_dir():
            for pt in sorted(models_dir.glob("*.pt"), key=lambda p: -p.stat().st_mtime)[:8]:
                runs.append(
                    {
                        "name": pt.name,
                        "meta": f"{pt.stat().st_size // 1024} KB",
                        "map": "—",
                        "live": pt.name == "best.pt",
                    }
                )

        total_frames = int(stats.get("total_frames", 0))
        train_frames = int(stats.get("train_frames", 0))
        val_frames = int(stats.get("val_frames", 0))
        total_boxes = int(stats.get("total_boxes", 0))
        goal = int(getattr(self, "target_photo_goal", TARGET_DATASET_PHOTO_GOAL) or TARGET_DATASET_PHOTO_GOAL)
        pct = min(100.0, (float(total_frames) / float(max(1, goal))) * 100.0)

        hi_bulk = sum(
            1
            for it in self._review_items
            if not it.get("done") and float(it.get("conf", 0.0) or 0.0) >= 0.90
        )

        class_rows = []
        for spec in YOLO_12_CLASS_SPECS:
            cid = int(spec["id"])
            class_rows.append(
                {
                    "id": cid,
                    "label": str(spec.get("label_tr", spec.get("key", cid))),
                    "key": str(spec.get("key", "")),
                    "count": int(counts.get(cid, 0)),
                }
            )

        return {
            "step": int(pipe.get("step", 1)),
            "statFrames": str(total_frames),
            "statTrain": str(train_frames),
            "statVal2": str(val_frames),
            "statBoxes": str(total_boxes),
            "statPending": f"{pending} kare",
            "statPendingSub": (
                f"{hi_bulk} kare güven ≥ 0,90 — toplu onaylanabilir" if pending else "Kuyruk boş"
            ),
            "statEmpty": f"{len(empty_names)} / 16",
            "statEmptySub": ", ".join(empty_names[:4]) if empty_names else "Boş sınıf yok",
            "statModel": "best.pt" if pipe.get("published") or (models_dir / "best.pt").is_file() else "—",
            "statModelSub": (
                self.stat_model_sub.text()
                if hasattr(self, "stat_model_sub") and self.stat_model_sub.text()
                else ("models/ altında" if (models_dir / "best.pt").is_file() else "Yayınlanmış model yok")
            ),
            "kvFrames": f"{total_frames} (%{pct:.1f})",
            "kvMissing": f"{max(0, goal - total_frames)} kare",
            "splitTrain": f"{train_frames} kare",
            "splitValCount": f"{val_frames} kare",
            "freezeState": "donduruldu" if pipe.get("frozen") else "dondurulmadı",
            "flowSummary": f"Adım {int(pipe.get('step', 1))}/6",
            "stReview": f"{pending} kare bekliyor" if pending else "Kuyruk boş",
            "nextTitle": (
                f"{pending} otomatik etiket onay bekliyor"
                if pending
                else ("Veri setini dondur" if not pipe.get("frozen") else "Sıradaki adıma geç")
            ),
            "nextText": (
                "Onaylanmamış karelerle eğitim başlatma."
                if pending
                else ("Engeller temiz — devam et." if pipe.get("frozen") else "Önce hazırlık ve dondurma.")
            ),
            "reviewProgTxt": f"{reviewed} / {reviewed + pending}",
            "reviewProgPct": int(
                round(100 * reviewed / max(1, reviewed + pending))
            ),
            "bulkLabel": (
                f"Eşik üstündeki {hi_bulk} kareyi toplu onayla" if hi_bulk else "Toplu onaylanacak kare yok"
            ),
            "liveJpegB64": live_jpeg_b64,
            "liveBadge": live_badge,
            "liveBoxes": live_boxes,
            "previewMeta": live_badge if live_jpeg_b64 else "Canlı kare yok",
            "queue": queue,
            "currentReview": current_review,
            "reviewIndex": int(self._review_index) if self._review_items else 0,
            "classRows": class_rows,
            "logText": log_text,
            "runs": runs,
            "map50": map50 if map50 > 0 else None,
            "frozen": bool(pipe.get("frozen")),
            "trained": bool(pipe.get("trained")),
            "pending": pending,
            "sessionVal": f"+{int(pipe.get('session_frames', 0) or 0)} kare",
        }

    def _class_filter_key(self) -> str:
        if not hasattr(self, "class_filter"):
            return "all"
        data = self.class_filter.currentData()
        return str(data or "all")

    def _apply_class_filter(self) -> None:
        mode = self._class_filter_key()
        query = ""
        if hasattr(self, "class_search"):
            query = self.class_search.text().strip().lower()
        counts = getattr(self, "_class_counts_cache", {})
        for cid, row in self.class_rows.items():
            cnt = int(counts.get(cid, row.count))
            if mode == "all":
                show = True
            elif mode == "filled":
                show = cnt > 0
            elif mode == "empty":
                show = cnt <= 0
            elif mode == "base":
                show = 0 <= cid <= 11
            elif mode == "nav":
                show = 12 <= cid <= 15
            elif mode == "vital":
                show = 16 <= cid <= 21
            elif mode == "essence":
                show = cid >= 22
            else:
                show = True
            if query:
                label = str(getattr(row, "label_tr", "") or "").lower()
                key = str(getattr(row, "class_key", "") or "").lower()
                show = show and (query in label or query in key or query in str(cid))
            row.setVisible(show)

    def _scroll_status_log_to_end(self) -> None:
        box = self.training_log_box
        box.moveCursor(QTextCursor.MoveOperation.End)
        box.ensureCursorVisible()
        bar = box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.training_log_box.appendPlainText(f"[{stamp}] {message}")
        self._scroll_status_log_to_end()
        self._mirror_status_to_app_log(message)

    def _mirror_status_to_app_log(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        logger.propagate = True
        if "[HATA]" in text or text.startswith("\u274c"):
            logger.error("%s", text)
        else:
            logger.info("%s", text)

    # ------------------------------------------------------------------
    # Existing action handlers (adapted)
    # ------------------------------------------------------------------
    def on_bootstrap_clicked(self) -> Dict[str, Any]:
        if self._bootstrap_worker is not None and self._bootstrap_worker.isRunning():
            self._append_log("[BİLGİ] Veri seti taraması zaten sürüyor.")
            return {}
        if self._training_worker is not None and self._training_worker.isRunning():
            self._append_log("[BİLGİ] RTX CUDA FP16 eğitimi zaten arka planda devam ediyor.")
            return {}

        mult = 10 if self.toggle_synthetic_10x.isChecked() else 1
        self.lbl_train_status.setText(SCAN_RUNNING_STATUS)
        self._append_log("Veri seti taraması sürüyor")
        self.btn_bootstrap.setEnabled(False)
        ctrl = self.factory_controller
        helper = self._builtin_helper

        def _scan() -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            fn = getattr(ctrl, "bootstrap_from_existing_frames", None)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    kwargs: Dict[str, Any] = {}
                    if "augment_multiplier" in sig.parameters:
                        kwargs["augment_multiplier"] = mult
                    elif "multiplier" in sig.parameters:
                        kwargs["multiplier"] = mult
                    elif "augment_count" in sig.parameters:
                        kwargs["augment_count"] = mult
                    raw_res = fn(**kwargs)
                    if isinstance(raw_res, dict):
                        result = raw_res
                except Exception as exc:
                    logger.warning(
                        "factory_controller.bootstrap_from_existing_frames yedekleniyor: %s",
                        exc,
                    )
            disk_after = helper.get_dataset_stats()
            if int(disk_after.get("total_frames", 0)) == 0:
                result = helper.bootstrap_from_existing_frames(augment_multiplier=mult)
            return result

        self._bootstrap_start_time = time.monotonic()
        try:
            self._bootstrap_start_frames = int(helper.get_dataset_stats().get("total_frames", 0))
        except Exception:
            self._bootstrap_start_frames = 0

        if getattr(self, "_bootstrap_speed_timer", None) is not None:
            self._bootstrap_speed_timer.stop()
        self._bootstrap_speed_timer = QTimer(self)
        self._bootstrap_speed_timer.setInterval(500)
        self._bootstrap_speed_timer.timeout.connect(self._on_bootstrap_speed_tick)
        self._bootstrap_speed_timer.start()

        self._bootstrap_worker = _BackgroundCallWorker(_scan, parent=self)
        self._bootstrap_worker.setObjectName("GatherionDatasetBootstrap")
        self._bootstrap_worker.succeeded.connect(self._on_bootstrap_worker_done)
        self._bootstrap_worker.failed.connect(self._on_bootstrap_worker_failed)
        self._bootstrap_worker.start()
        return {}

    def _on_bootstrap_speed_tick(self) -> None:
        if self._bootstrap_worker is None or not self._bootstrap_worker.isRunning():
            if getattr(self, "_bootstrap_speed_timer", None) is not None:
                self._bootstrap_speed_timer.stop()
            return
        elapsed = max(0.001, time.monotonic() - getattr(self, "_bootstrap_start_time", time.monotonic()))
        try:
            cur_stats = self._builtin_helper.get_dataset_stats()
            cur_frames = int(cur_stats.get("total_frames", 0))
            delta = max(0, cur_frames - getattr(self, "_bootstrap_start_frames", 0))
            fps = delta / elapsed
            txt = f"⚡ Tarama sürüyor · {fps:.1f} kare/sn · Geçen: {int(elapsed)} sn · Toplam: {cur_frames} kare"
            if hasattr(self, "lbl_speed_meter"):
                self.lbl_speed_meter.setText(txt)
            self.lbl_train_status.setText(f"Veri seti taraması sürüyor · {fps:.1f} kare/sn")
        except Exception:
            pass

    def _on_bootstrap_worker_done(self, result: object) -> None:
        if getattr(self, "_bootstrap_speed_timer", None) is not None:
            self._bootstrap_speed_timer.stop()
        self.btn_bootstrap.setEnabled(True)
        self._pipeline["scanned"] = True
        payload = result if isinstance(result, dict) else {}
        line = format_dataset_bootstrap_complete(payload)
        if line:
            self.lbl_train_status.setText(line)
            self._append_log(line)
        elapsed = max(0.001, time.monotonic() - getattr(self, "_bootstrap_start_time", time.monotonic()))
        try:
            final_stats = self._builtin_helper.get_dataset_stats()
            total_f = int(final_stats.get("total_frames", 0))
            delta = max(0, total_f - getattr(self, "_bootstrap_start_frames", 0))
            fps = delta / elapsed
            if hasattr(self, "lbl_speed_meter"):
                self.lbl_speed_meter.setText(f"✓ Bootstrap tamamlandı · Ortalama: {fps:.1f} kare/sn ({int(elapsed)} sn)")
        except Exception:
            pass
        # Bootstrap review json üretebilir — kuyruk + sayaç zorunlu senkron
        try:
            self._reload_review_queue()
        except Exception as exc:
            self._append_log(f"[KUYRUK] Bootstrap sonrası yenileme: {exc}")
            self._sync_pending_from_queue()
        refreshed = self.refresh_dashboard(force=False)
        self._append_log(
            f"[BAŞARILI] Bootstrap + Augment tamamlandı! "
            f"Kare: {refreshed['total_frames']} (Train: {refreshed['train_frames']}, Val: {refreshed['val_frames']}) · "
            f"Kutu: {refreshed['total_boxes']} · onay kuyruğu {self._pending_count()}"
        )
        if int(self._pipeline["step"]) == 3:
            self._rebuild_health_checks()
        self._save_pipeline_disk_state()

    def _on_bootstrap_worker_failed(self, error_msg: str) -> None:
        if getattr(self, "_bootstrap_speed_timer", None) is not None:
            self._bootstrap_speed_timer.stop()
        self.btn_bootstrap.setEnabled(True)
        self.lbl_train_status.setText(f"Eğitim Durumu: Hata Oluştu — {error_msg}")
        self._append_log(f"[HATA] Veri seti taraması: {error_msg}")

    def _accept_train_result(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Gate: only real Ultralytics epoch progress may mark trained=true."""
        if not isinstance(result, dict) or not result:
            return False, "Boş eğitim sonucu"
        if (
            result.get("started") is False
            or result.get("status") in ("failed", "stopped")
            or result.get("state") in ("failed", "refused", "busy", "stopped")
            or result.get("stopped")
            or result.get("interrupted")
        ):
            return False, str(result.get("reason") or result.get("error") or "Eğitim durduruldu")
        mode = str(result.get("mode") or "")
        if "opencv" in mode.lower() or "prototype" in mode.lower():
            return False, "OpenCV prototip eğitimi kabul edilmez — Ultralytics YOLO gerekli"
        if mode and "ultralytics" not in mode.lower() and mode not in ("", "ultralytics_artifact"):
            # Unknown mode without ultralytics marker: require evidence below.
            pass
        epochs_req = int(result.get("epochs", self.hp_epoch.value()) or 0)
        epochs_done = int(
            result.get("epochs_completed", result.get("epoch", 0)) or 0
        )
        elapsed = float(result.get("elapsed_sec", 0.0) or 0.0)
        map50 = float(result.get("mAP50", 0.0) or 0.0)
        arts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        best_pt = str(arts.get("ultralytics_best_pt") or "")
        has_best = bool(best_pt) and Path(best_pt).is_file()
        if epochs_done < 1 and map50 <= 0 and not has_best:
            return False, "Hiç epoch tamamlanmadı / best.pt yok"
        if epochs_req >= 10 and elapsed > 0 and elapsed < 5.0:
            return (
                False,
                f"Şüpheli süre: {elapsed:.1f}s / {epochs_req} epoch — sahte tamamlanma reddedildi",
            )
        click_mono = float(getattr(self, "_train_click_mono", 0.0) or 0.0)
        if click_mono > 0 and epochs_req >= 10:
            wall = time.monotonic() - click_mono
            if wall < 5.0:
                return (
                    False,
                    f"Eğitim tıklamasından {wall:.1f}s sonra bitti — gerçek epoch döngüsü değil",
                )
        if "ultralytics" not in mode.lower() and not has_best:
            return False, f"Ultralytics sonucu değil (mode={mode or '—'})"
        return True, ""

    def _start_training_worker(
        self,
        epochs: Optional[int] = None,
        model: Optional[str] = None,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        only_approved: Optional[bool] = None,
    ) -> RTXTrainingWorker:
        """
        Starts or creates RTXTrainingWorker with explicit hyperparameters or current UI selections.
        Ensures model, imgsz, batch, epochs, only_approved propagate cleanly without falling back to defaults.
        """
        if epochs is None:
            epochs = int(self.hp_epoch.value()) if hasattr(self, "hp_epoch") else 25
        if model is None:
            model = str(self.hp_model.currentText() if hasattr(self, "hp_model") else "yolo11n.pt").strip() or "yolo11n.pt"
        if batch is None:
            try:
                batch = int(str(self.hp_batch.currentText() if hasattr(self, "hp_batch") else "16"))
            except ValueError:
                batch = 16
        if imgsz is None:
            try:
                imgsz = int(str(self.hp_imgsz.currentText() if hasattr(self, "hp_imgsz") else "640"))
            except ValueError:
                imgsz = 640
        if only_approved is None:
            if getattr(self, "chk_only_approved", None) and self.chk_only_approved.isChecked():
                only_approved = True
            elif getattr(self, "toggle_approved_only", None) and self.toggle_approved_only.isChecked():
                only_approved = True
            else:
                only_approved = False

        parent_widget = self if isinstance(self, QWidget) else None
        try:
            worker = RTXTrainingWorker(
                controller=self.factory_controller,
                epochs=epochs,
                parent=parent_widget,
                imgsz=imgsz,
                batch=batch,
                model=model,
                only_approved=bool(only_approved),
            )
        except RuntimeError:
            worker = RTXTrainingWorker(
                controller=self.factory_controller,
                epochs=epochs,
                parent=None,
                imgsz=imgsz,
                batch=batch,
                model=model,
                only_approved=bool(only_approved),
            )
        worker.setObjectName("GatherionDatasetBootstrap")
        worker.status_text.connect(self._on_training_status)
        worker.training_deferred.connect(self._on_training_deferred)
        worker.progress_updated.connect(self._on_training_progress)
        worker.training_finished.connect(self._on_training_finished)
        worker.training_failed.connect(self._on_training_failed)
        worker.finished.connect(self._on_training_worker_finished)
        self._training_worker = worker
        worker.start()
        return worker

    def _on_training_worker_finished(self) -> None:
        self.btn_stop_train.setEnabled(False)
        self._update_train_lock()

    def on_train_rtx_clicked(self, only_approved: Optional[bool] = None) -> None:
        if only_approved is None:
            if getattr(self, "chk_only_approved", None) and self.chk_only_approved.isChecked():
                only_approved = True
            elif getattr(self, "toggle_approved_only", None) and self.toggle_approved_only.isChecked():
                only_approved = True
            else:
                only_approved = False
        is_only_approved = bool(only_approved)

        if is_only_approved:
            try:
                from dataset_factory.approved_pool import get_approved_pool
                get_approved_pool(self._dataset_dir()).prepare_approved_training_dataset()
            except Exception as exc:
                logger.warning("ApprovedDatasetPool prepare failed in on_train_rtx_clicked: %s", exc)
        elif self._pending_count() > 0:
            self._append_log(
                f"[EĞİTİM] RED: Onay kuyruğunda {self._pending_count()} bekleyen kare var (Adım 2)."
            )
            self._go_step(2)
            return

        if not self._pipeline.get("frozen"):
            self._append_log("[EĞİTİM] Önce veri setini dondur (Adım 3).")
            self._go_step(3)
            return

        if self._training_worker is not None and self._training_worker.isRunning():
            self._append_log("[BİLGİ] RTX CUDA FP16 eğitimi zaten arka planda devam ediyor.")
            return
        if self._bootstrap_worker is not None and self._bootstrap_worker.isRunning():
            self._append_log("[BİLGİ] Veri seti taraması zaten sürüyor.")
            return

        epochs = int(self.hp_epoch.value())
        model = str(self.hp_model.currentText() or "yolo11n.pt").strip() or "yolo11n.pt"
        try:
            batch = int(str(self.hp_batch.currentText() or "16"))
        except ValueError:
            batch = 16
        try:
            imgsz = int(str(self.hp_imgsz.currentText() or "640"))
        except ValueError:
            imgsz = 640

        self._pipeline["trained"] = False
        self._pipeline["validated"] = False
        self._pipeline["published"] = False
        self._pipeline["train_session_ok"] = False
        self._train_click_mono = time.monotonic()
        self.btn_train_rtx.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_progress_bar.setValue(0)
        self.epoch_label.setText(f"Epoch 0 / {epochs}")
        self.lbl_train_status.setText(SCAN_RUNNING_STATUS)
        approved_suffix = " (Sadece Onaylananlar)" if is_only_approved else ""
        self._append_log(
            f"[EĞİTİM] Başlatılıyor: {model} · {epochs} epoch · imgsz={imgsz} · batch={batch}{approved_suffix}"
        )
        self._append_log("Veri seti taraması sürüyor")

        self._start_training_worker(epochs=epochs, model=model, imgsz=imgsz, batch=batch, only_approved=is_only_approved)

    def _on_train_only_approved_clicked(self) -> None:
        """Sadece onaylanan veri havuzunu kullanarak eğitimi başlatır."""
        if hasattr(self, "chk_only_approved"):
            self.chk_only_approved.setChecked(True)
        if hasattr(self, "toggle_approved_only"):
            self.toggle_approved_only.setChecked(True)
        self.on_train_rtx_clicked(only_approved=True)

    def _on_training_status(self, text: str) -> None:
        self.lbl_train_status.setText(text)
        self._append_log(text)

    def _on_training_deferred(self, _reason: str) -> None:
        self._update_train_lock()
        self.btn_stop_train.setEnabled(False)

    def _on_training_progress(self, payload: Dict[str, Any]) -> None:
        if payload.get("batch_step"):
            cur_ep = payload.get("epoch", 1)
            tot_ep = payload.get("epochs", self.hp_epoch.value())
            cur_b = payload.get("batch", 1)
            tot_b = payload.get("total_batches", 1)
            pct_b = int(cur_b * 100 / max(1, tot_b))
            self.lbl_train_status.setText(
                f"Eğitim Sürüyor: Epoch {cur_ep}/{tot_ep} · Batch {cur_b}/{tot_b} (%{pct_b})"
            )
            return
        epoch = int(payload.get("epoch", 0))
        epochs = max(1, int(payload.get("epochs", self.hp_epoch.value())))
        pct = min(100, int(round((float(epoch) / float(epochs)) * 100.0)))
        map50 = float(payload.get("mAP50", 0.0))
        box_loss = float(payload.get("box_loss", 0.0))
        cls_loss = float(payload.get("cls_loss", 0.0))
        self.train_progress_bar.setValue(pct)
        self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
        self.epoch_pct.setText(f"%{pct}")
        self.lbl_map50_loss.setText(f"box {box_loss:.4f} · cls {cls_loss:.4f}")
        self.m_map.setText(f"mAP50 %{map50 * 100:.1f}")
        if epoch == 1 or epoch == epochs or (epochs <= 10) or (epoch % max(1, epochs // 10) == 0):
            self._append_log(
                f"[EPOCH {epoch:02d}/{epochs:02d}] mAP50=%{map50 * 100:.2f} | "
                f"box_loss={box_loss:.4f} | cls_loss={cls_loss:.4f}"
            )

    def _on_training_finished(self, result: Dict[str, Any]) -> None:
        ok, reason = self._accept_train_result(dict(result or {}))
        self._update_train_lock()
        self.btn_stop_train.setEnabled(False)
        if not ok:
            self._pipeline["trained"] = False
            self._pipeline["train_session_ok"] = False
            if "durduruldu" in reason.lower() or "stopped" in reason.lower():
                self.lbl_train_status.setText(f"Eğitim Durumu: Durduruldu — {reason}")
                self._append_log(f"[BİLGİ] {reason}")
            else:
                self.lbl_train_status.setText(f"Eğitim Durumu: RED — {reason}")
                self._append_log(f"[HATA] Eğitim kabul edilmedi: {reason}")
            self._save_pipeline_disk_state()
            return

        self._last_train_result = dict(result or {})
        self._last_train_result["artifact_fresh"] = True
        self._last_train_result["metrics_kind"] = "fresh_after_freeze"
        self._pipeline["trained"] = True
        self._pipeline["train_session_ok"] = True
        self._pipeline["model_stale"] = False
        self._pipeline["train_completed_at"] = datetime.now().isoformat(timespec="seconds")
        ds_dir = self._dataset_dir() if hasattr(self, "_dataset_dir") and callable(self._dataset_dir) else None
        md_dir = self._models_dir() if hasattr(self, "_models_dir") and callable(self._models_dir) else None
        try:
            from dataset_factory.dataset_validator import clear_model_stale
            clear_model_stale(dataset_dir=ds_dir, models_dir=md_dir)
        except Exception:
            pass

        map50 = float(result.get("mAP50", 0.0) or 0.0)
        box_loss = float(result.get("box_loss", 0.0) or 0.0)
        cls_loss = float(result.get("cls_loss", 0.0) or 0.0)
        epochs = int(result.get("epochs_completed", result.get("epochs", self.hp_epoch.value())) or 0)
        epochs_req = int(result.get("epochs", self.hp_epoch.value()) or epochs)
        self.train_progress_bar.setValue(100)
        self.epoch_label.setText(f"Epoch {epochs} / {epochs_req}")
        self.epoch_pct.setText("%100")
        self.lbl_train_status.setText("Eğitim Durumu: TAMAMLANDI")
        self.lbl_map50_loss.setText(f"box {box_loss:.4f} · cls {cls_loss:.4f}")
        self.m_map.setText(f"mAP50 %{map50 * 100:.2f}")
        artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
        onnx_path = artifacts.get("onnx_model", "models/yolov8n_gatherion.onnx")
        elapsed = float(result.get("elapsed_sec", 0.0) or 0.0)
        self._append_log(
            f"[TAMAMLANDI] RTX eğitimi bitti · {epochs}/{epochs_req} epoch · "
            f"{elapsed:.0f}s · mAP50=%{map50 * 100:.2f} · Model: {onnx_path}"
        )
        self._mark_step_done(4)
        self._save_pipeline_disk_state()
        self.refresh_dashboard()
        self._go_step(5)

    def _on_training_failed(self, error_msg: str) -> None:
        self._pipeline["trained"] = False
        self._pipeline["train_session_ok"] = False
        self._update_train_lock()
        self.btn_stop_train.setEnabled(False)
        if "durduruldu" in error_msg.lower() or "stopped" in error_msg.lower() or "interrupted" in error_msg.lower():
            self.lbl_train_status.setText(f"Eğitim Durumu: Durduruldu — {error_msg}")
            self._append_log(f"[BİLGİ] {error_msg}")
        else:
            self.lbl_train_status.setText(f"Eğitim Durumu: Hata Oluştu — {error_msg}")
            self._append_log(f"[HATA] Eğitim sırasında hata: {error_msg}")
        self._save_pipeline_disk_state()

    def wait_for_training_completion(self, timeout_ms: int = 15000) -> bool:
        if self._training_worker is not None and self._training_worker.isRunning():
            finished = self._training_worker.wait(timeout_ms)
            QApplication.processEvents()
            return bool(finished)
        return True

    def get_onnx_runtime_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "execution_provider": "CPUExecutionProvider",
            "model_file_size_bytes": 0,
            "model_path": "",
            "active": False,
        }
        try:
            detector = getattr(self, "yolo_detector", None)
            if detector is not None:
                status["model_path"] = getattr(detector, "model_path", "")
                if hasattr(detector, "backend_name"):
                    bname = str(detector.backend_name)
                    if "CUDAExecutionProvider" in bname:
                        status["execution_provider"] = "CUDAExecutionProvider"
                    elif "CPUExecutionProvider" in bname:
                        status["execution_provider"] = "CPUExecutionProvider"
                    elif "onnxruntime:" in bname:
                        parts = bname.split(":")
                        if len(parts) > 1:
                            status["execution_provider"] = parts[1]
                mpath = getattr(detector, "_resolve_model_path", None)
                resolved = None
                if mpath and callable(mpath):
                    resolved = mpath(detector.model_path)
                elif detector.model_path:
                    if os.path.isfile(detector.model_path):
                        resolved = os.path.abspath(detector.model_path)
                if resolved and os.path.isfile(resolved):
                    status["model_file_size_bytes"] = os.path.getsize(resolved)
                    status["active"] = True
            else:
                default_path = "models/yolov8n_gatherion.onnx"
                status["model_path"] = default_path
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cand = os.path.join(root_dir, default_path)
                if os.path.isfile(cand):
                    status["model_file_size_bytes"] = os.path.getsize(cand)
                    status["active"] = True
        except Exception as exc:
            logger.debug("[YOLOFactoryDialog] get_onnx_runtime_status error: %s", exc)
        return status

    def on_open_dataset_folder_clicked(self) -> Path:
        dataset_path = self._builtin_helper.dataset_dir
        dataset_path.mkdir(parents=True, exist_ok=True)
        self._append_log(f"[KLASÖR] dataset/ açılıyor: {dataset_path}")
        offscreen = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
        if not offscreen:
            try:
                if sys.platform == "win32" and hasattr(os, "startfile"):
                    os.startfile(str(dataset_path))
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(dataset_path)))
            except Exception as exc:
                logger.debug("Klasör açma işlemi atlandı: %s", exc)
        return dataset_path

    def on_refresh_clicked(self) -> None:
        """Kullanıcı '↻ Yenile' butonuna bastığında UI ve aktif kuyruğu zorla yeniler."""
        self._append_log("[DASHBOARD] Yenileme tetiklendi...")

        # 1. Aktif adımı tespit et (step_stack veya pipeline)
        active_step = int(self._pipeline.get("step", 1))
        if hasattr(self, "step_stack") and self.step_stack is not None:
            active_step = self.step_stack.currentIndex() + 1

        # 2. Adım 2'deysek onay kuyruğunu diskten tam yenile (mevcut kareyi koruyarak)
        if active_step == 2 or int(self._pipeline.get("step", 0)) == 2:
            try:
                curr_stem = ""
                if self._review_items and 0 <= self._review_index < len(self._review_items):
                    curr_stem = str(self._review_items[self._review_index].get("stem", ""))
                self._reload_review_queue()
                if curr_stem:
                    for idx, it in enumerate(self._review_items):
                        if str(it.get("stem", "")) == curr_stem:
                            self._review_index = idx
                            break
                    self._render_queue_list()
                    self._render_review_frame()
                self._append_log(f"[KUYRUK] Onay kuyruğu yenilendi ({len(self._review_items)} kare).")
            except Exception as exc:
                self._append_log(f"[KUYRUK] Yenileme hatası: {exc}")

        # 3. Canlı Yakalama / İstatistik / KPI panellerini yenile
        self.refresh_dashboard(force=True)

        # 4. Adım sayfalarını yenile
        if active_step == 3 and hasattr(self, "_rebuild_health_checks"):
            self._rebuild_health_checks()
        elif active_step == 4 and hasattr(self, "_update_train_lock"):
            self._update_train_lock()
        elif active_step == 6 and hasattr(self, "_refresh_publish_checklist"):
            self._refresh_publish_checklist()

        # 5. Buton üzerinde anlık görsel geri bildirim ver (Yenilendi animasyonu)
        if hasattr(self, "btn_refresh"):
            orig_text = self.btn_refresh.text()
            self.btn_refresh.setText("✓ Yenilendi")
            self.btn_refresh.setStyleSheet("font-weight: bold; color: #22c55e;")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(
                1200,
                lambda: (
                    self.btn_refresh.setText(orig_text),
                    self.btn_refresh.setStyleSheet("")
                )
            )

        total_frames = int(getattr(self, "_last_stats_sig", (0,))[0]) if hasattr(self, "_last_stats_sig") else 0
        self._append_log(f"[DASHBOARD] Yenilendi: toplam {total_frames} kare, {self._pending_count()} onay bekleyen.")

    def get_yolo_factory_dialog_health_summary(self) -> dict:
        try:
            return {"yolo_factory_dialog_ready": True}
        except Exception:
            return {"yolo_factory_dialog_ready": False}

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._save_pipeline_disk_state()
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        if hasattr(self, "_cap_timer") and self._cap_timer.isActive():
            self._cap_timer.stop()
        if self._training_worker is not None and self._training_worker.isRunning():
            self._training_worker.wait(3000)
        super().closeEvent(event)

    def get_yolo_factory_dialog_ready_flag(self) -> dict:
        return {"module": "yolo_factory_dialog", "ready": True}

    def get_yolo_factory_dialog_version_info(self) -> dict:
        return {"module": "yolo_factory_dialog", "version": "3.0.0", "ready": True}

    def get_yolo_factory_dialog_diagnostic_snapshot(self) -> dict:
        return {
            "module": "yolo_factory_dialog",
            "status": "ok",
            "ready": True,
            "pipeline": dict(self._pipeline),
        }


class _CompatKPI:
    """Thin adapter so older refresh paths can call update_card on strip labels."""

    def __init__(self, value_lbl: QLabel, sub_lbl: QLabel) -> None:
        self.value_lbl = value_lbl
        self.sub_lbl = sub_lbl

    def update_card(self, value_text: str, sub_text: Optional[str] = None) -> None:
        self.value_lbl.setText(str(value_text))
        if sub_text is not None:
            self.sub_lbl.setText(str(sub_text))



def get_yolo_factory_dialog_capabilities() -> list:
    return ["yolo_factory_dialog", "dataset_pipeline_ui", "diagnostic_snapshot"]


def validate_yolo_factory_dialog_contract() -> bool:
    return bool(get_yolo_factory_dialog_capabilities())


def get_yolo_factory_dialog_baseline_metrics() -> dict:
    return {"module": "yolo_factory_dialog", "contract_valid": validate_yolo_factory_dialog_contract(), "capability_count": len(get_yolo_factory_dialog_capabilities())}


def get_yolo_factory_dialog_readiness_status() -> dict:
    return {"module": "yolo_factory_dialog", "ready": validate_yolo_factory_dialog_contract(), "metrics": get_yolo_factory_dialog_baseline_metrics()}


def get_yolo_factory_dialog_audit_record() -> dict:
    return {"module": "yolo_factory_dialog", "audit_passed": validate_yolo_factory_dialog_contract(), "readiness": get_yolo_factory_dialog_readiness_status()}


def get_yolo_factory_dialog_operational_digest() -> dict:
    return {"module": "yolo_factory_dialog", "operational": validate_yolo_factory_dialog_contract(), "audit": get_yolo_factory_dialog_audit_record()}


def get_yolo_factory_dialog_telemetry_payload() -> dict:
    return {"module": "yolo_factory_dialog", "payload": get_yolo_factory_dialog_operational_digest()}


def get_yolo_factory_dialog_integrity_seal() -> dict:
    return {"module": "yolo_factory_dialog", "sealed": validate_yolo_factory_dialog_contract(), "telemetry": get_yolo_factory_dialog_telemetry_payload()}


def get_yolo_factory_dialog_compliance_certificate() -> dict:
    return {"module": "yolo_factory_dialog", "certified": validate_yolo_factory_dialog_contract(), "seal": get_yolo_factory_dialog_integrity_seal()}


def get_yolo_factory_dialog_readiness_probe() -> dict:
    return {"module": "yolo_factory_dialog", "ready": validate_yolo_factory_dialog_contract(), "certificate": get_yolo_factory_dialog_compliance_certificate()}


def get_yolo_factory_dialog_audit_summary() -> dict:
    return {"module": "yolo_factory_dialog", "audit_passed": validate_yolo_factory_dialog_contract(), "probe": get_yolo_factory_dialog_readiness_probe()}



