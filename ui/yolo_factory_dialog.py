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

import atexit
import copy
import inspect
import json
import logging
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
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
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QImageReader,
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

from dataset_factory.approval_state import is_frame_closed
from dataset_factory.approved_pool import ApprovedDatasetPool, get_approved_pool, is_approved
from dataset_factory.box_match import has_box_geometry, same_box  # P1-13 (K-33): tek "ayni kutu" tanimi
from version import __version__
# P3-7 (v18): 3-tier güven renkleri/eşikleri TEK kaynak —
# dataset_factory/confidence_tier_coloring.py (test-pinned, test_w27).
from dataset_factory.confidence_tier_coloring import (
    CONFIDENT_COLOR_HEX,
    CONFIDENT_STATUS_COLOR_HEX,
    CONFIDENT_TEXT_COLOR_HEX,
    LOW_CONFIDENCE_COLOR_HEX,
    LOW_CONFIDENCE_STATUS_COLOR_HEX,
    LOW_CONFIDENCE_TEXT_COLOR_HEX,
    SUSPICIOUS_COLOR_HEX,
    SUSPICIOUS_STATUS_COLOR_HEX,
    SUSPICIOUS_TEXT_COLOR_HEX,
    SUSPICIOUS_THRESHOLD,
)

# P1-12 (K-31/K-32/K-34/K-35): TEK güven sabitleri — dataset_factory/label_trust.py.
# label_trust UI import etmez, döngü yoktur; yedek değerler yalnızca
# import arızasında devreye girer.
try:
    from dataset_factory.label_trust import (
        CONF_BULK_DEFAULT,
        CONF_REQUEUE_DEFAULT,
        TRAIN_CONF_FLOOR,
    )
except Exception:  # pragma: no cover - import fallback
    CONF_BULK_DEFAULT = 0.90
    CONF_REQUEUE_DEFAULT = 0.58
    TRAIN_CONF_FLOOR = 0.45


def _measured_confidence(ann: Dict[str, Any]) -> Optional[float]:
    """Ölçülmüş kutu güveni (confidence/conf/score, [0, 1] normalize) ya da None.

    P1-12 (K-32): ölçülmemiş güven "—" gösterilir, %97/%90 varsayılanı
    yazılmaz. 1.0 üstü değerler yüzde sayılır (eski norm_conf kuralı).
    """
    if not isinstance(ann, dict):
        return None
    for key in ("confidence", "conf", "score"):
        if key not in ann:
            continue
        try:
            value = float(ann[key])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN: not a measurement
            continue
        return value / 100.0 if value > 1.0 else value
    return None

# F-1957 [O3] çözüldü: get_approved_count artık ApprovedDatasetPool'un
# gerçek metodu (dataset_factory/approved_pool.py); buradaki
# monkey-patch kaldırıldı.



from dataset_factory.frame_time_machine import PreGatherTimeMachineBuffer
from dataset_factory.lru_frame_cache import get_lru_frame_cache  # F-2000 / R44
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
            return f"{amt / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
        if amt >= 1_000:
            return f"{amt / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
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
# Kanonik YOLO Sınıfı Tanımları, Türkçe Etiketleri ve Renk Paleti
# (P1-14: eski "12 YOLO Sınıfı" başlığı yanlıştı — tek otorite
# dataset_factory.class_taxonomy.CANONICAL_NAMES, bugün 29 sınıf 0..28.
# YOLO_12_CLASS_SPECS adı uyumluluk için korunuyor ama içeriği kanonik kümedir.)
# ============================================================================
TARGET_DATASET_PHOTO_GOAL: int = 100000

# Diyalogun yerel 16 sınıf tanımı (0..15) kanonik taksonomiyle birebir aynıdır
# (F-1946 remap: tek otorite dataset_factory/class_taxonomy.py CANONICAL_NAMES;
# 8 = ore_obsidian, 5 = tree_log, 6 = herb_edana, 7 = herb_lumina, 12 = crystal_blood).
# P1-14 (K-38): bu birebirlik _build_yolo_class_specs'te import anında
# doğrulanır; taksonomi yüklenemezse veya kanonik küme tamamlanamazsa sert hata
# verilir — sessiz 16 sınıfa düşüş yok.
# Eski yerel adlar (combo_skill_prompt vb.) yalnızca CLASS_KEY_ALIASES'te
# kanonik 13 (skill_ui_prompt) için legacy alias olarak durur.
_BASE_CLASS_SPECS: List[Dict[str, Any]] = [
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
        "key": "tree_log",
        "label_tr": "5: Ağaç/Kereste",
        "full_tr": "5: Ağaç/Kereste (tree_log)",
        "color_hex": "#d97706",
        "color_bgr": (6, 119, 217),
    },
    {
        "id": 6,
        "key": "herb_edana",
        "label_tr": "6: Edana",
        "full_tr": "6: Edana (herb_edana)",
        "color_hex": "#38bdf8",
        "color_bgr": (248, 189, 56),
    },
    {
        "id": 7,
        "key": "herb_lumina",
        "label_tr": "7: Lumina",
        "full_tr": "7: Lumina (herb_lumina)",
        "color_hex": "#06b6d4",
        "color_bgr": (212, 182, 6),
    },
    {
        "id": 8,
        "key": "ore_obsidian",
        "label_tr": "8: Obsidyen",
        "full_tr": "8: Obsidyen (ore_obsidian)",
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
        "key": "crystal_blood",
        "label_tr": "12: Kan Kristali",
        "full_tr": "12: Kan Kristali (crystal_blood)",
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
    from dataset_factory.class_taxonomy import CANONICAL_NAMES as _TAXONOMY_CANONICAL_NAMES
    from dataset_factory.class_taxonomy import alias_map as _taxonomy_alias_map
except Exception as exc:
    # P1-14 (K-38): taksonomi yükleme hatası SERT hatadır — sessiz
    # 16 sınıfa düşüş ve eksik alias tablosuyla çalışma yok.
    raise RuntimeError(
        "Taksonomi yüklenemedi (dataset_factory.class_taxonomy): "
        "YOLO Factory dialog kanonik sınıf kümesi olmadan açılamaz"
    ) from exc


def _build_yolo_class_specs() -> List[Dict[str, Any]]:
    """Tek kaynaktan kanonik sınıf spec'leri: yerel 16 (0..15, kanonikle
    birebir — F-1946) + canonical taksonomiden 16..N (ui_class_specs).

    P1-14 (K-38): sınıf sayısı tek kaynaktır (CANONICAL_NAMES). Yerel tablo
    taksonomiyle çelişirse, taksonomi genişletilemezse veya sonuç kanonik
    kümeyi tam kapsamazsa SERT hata verilir; eskiden olduğu gibi warning
    loglayıp sessizce 16 sınıfla devam edilmez.
    """
    for item in _BASE_CLASS_SPECS:
        cid = int(item["id"])
        if _TAXONOMY_CANONICAL_NAMES.get(cid) != item["key"]:
            raise RuntimeError(
                "Sınıf taksonomisi çakışması: yerel spec id=%s key=%r, "
                "kanonik ad=%r"
                % (cid, item["key"], _TAXONOMY_CANONICAL_NAMES.get(cid))
            )
    specs = [dict(item) for item in _BASE_CLASS_SPECS]
    seen = {int(item["id"]) for item in specs}
    from dataset_factory.class_taxonomy import ui_class_specs as _ui_class_specs

    for extra in _ui_class_specs():
        cid = int(extra["id"])
        if cid in seen:
            raise RuntimeError(
                "Sınıf taksonomisi çakışması: id=%s zaten yerel listede" % cid
            )
        seen.add(cid)
        specs.append(dict(extra))
    specs.sort(key=lambda item: int(item["id"]))
    if {int(item["id"]) for item in specs} != set(_TAXONOMY_CANONICAL_NAMES):
        _max_id = max(_TAXONOMY_CANONICAL_NAMES) if _TAXONOMY_CANONICAL_NAMES else -1
        raise RuntimeError(
            "Sınıf taksonomisi eksik: %d spec üretildi, kanonik %d sınıf "
            "bekleniyor (0..%d)"
            % (len(specs), len(_TAXONOMY_CANONICAL_NAMES), _max_id)
        )
    return specs


YOLO_12_CLASS_SPECS: List[Dict[str, Any]] = _build_yolo_class_specs()

CLASS_ID_TO_SPEC: Dict[int, Dict[str, Any]] = {
    int(item["id"]): item for item in YOLO_12_CLASS_SPECS
}

# Renkler tek kaynaktan türetilir: sınıf spec'leri. Eskiden elle yazılmış ikinci
# bir harita vardı ve aynı sınıf iki panelde farklı renkte görünüyordu (F-1953).
CLASS_COLORS_MAP: Dict[int, str] = {
    int(item["id"]): str(item.get("color_hex", "#22c55e")) for item in YOLO_12_CLASS_SPECS
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
    "gem_crystal": 4,
    "gem_sapphire": 4,
    "gem_ruby": 4,
    "gem_diamond": 4,
    "crystal_node": 4,
    "wood_tree": 5,
    "log_timber": 5,
    "tree_log": 5,
    # Legacy prompt/crest adları kanonik 13 (skill_ui_prompt) alias'ıdır (F-1946):
    # taksonomi alias_map'i de aynı eşlemeyi yapar; buradaki literaller import
    # başarısız olsa bile kanonik kalır.
    "world_gather_crest": 13,
    "crest": 13,
    "f_gather_prompt": 13,
    "f_gather": 13,
    "combo_skill_prompt": 13,
    "combo_icon": 13,
    "q_combo_icon": 13,
    "e_combo_icon": 13,
    "kombo": 13,
    "kombo_ikonu": 13,
    "combo": 13,
    "e_gather_icon": 13,
    "e_gather": 13,
    "e_floating_icon": 13,
    "herb_edana": 6,
    "edana": 6,
    "herb_lumina": 7,
    "lumina": 7,
    "ore_obsidian": 8,
    "obsidian": 8,
    "minimap_sprout": 9,
    "sprout": 9,
    "monster_3d_negative": 10,
    "mob_guardian": 10,
    "monster": 10,
    "minimap_monster_negative": 11,
    "minimap_monster": 11,
    "monster_3d": 10,
    "minimap_monster_dot": 11,
    "world_material": 12,
    "crystal_blood": 12,
    "blood_crystal": 12,
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
    """Canonicalize any class name (including legacy names like monster_3d_negative, e_gather_icon).

    P1-14 (K-37): İSİM ÖNCELİKLİDİR. Eskiden geçerli bir cid, cname'i hiç
    okumadan kazanıyordu — `cid=6, cname="crest"` sessizce herb_edana
    oluyordu. Şimdi: adı çözülebilen kutu adının sınıfını alır ("crest"
    -> skill_ui_prompt / 13); adı boş olan kutu cid'in kanonik adını alır;
    adı da cid'i de çözülemeyen kutu "bilinmeyen" döner ve warning loglanır
    — bilinmeyen ad artık 12 (crystal_blood) çöplüğüne veya ham adın
    aynen geçirilmesine düşmez.
    """
    raw = str(cname or "").strip()
    if raw:
        for key in (raw, raw.lower()):
            target_cid = CLASS_KEY_ALIASES.get(key)
            if target_cid is not None and target_cid in CLASS_ID_TO_SPEC:
                return str(CLASS_ID_TO_SPEC[target_cid]["key"])
        logger.warning(
            "Bilinmeyen sınıf adı: cname=%r (cid=%r) — 'bilinmeyen' sayıldı, "
            "sınıf 12'ye zorlanmadı",
            raw,
            cid,
        )
        return "bilinmeyen"
    try:
        cid_int = int(cid)
        if cid_int in CLASS_ID_TO_SPEC:
            return str(CLASS_ID_TO_SPEC[cid_int]["key"])
    except (ValueError, TypeError):
        pass
    logger.warning(
        "Çözülemeyen kutu sınıfı: cid=%r, cname=%r — 'bilinmeyen' sayıldı",
        cid,
        cname,
    )
    return "bilinmeyen"


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


def _format_loss(value: Any) -> str:
    """Loss değerini 4 ondalıkla biçimlendir; yoksa "—" döner.

    v19 telemetri: loss alınamadığında 0.0000 basmak YASAK — 0.0000
    "loss sıfır" gibi okunuyor, oysa anlamı "loss alınamadı". Engine
    alınamayan loss'u ``None`` gönderir; bu fonksiyon ``None``'u (ve
    sayıya çevrilemeyeni) "—" yapar.
    """
    if value is None:
        return "—"
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if num != num or num in (float("inf"), float("-inf")):
        return "—"
    return f"{num:.4f}"


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
        self._yolo_labeler: Optional[Any] = None
        self._shared_yolo_detector: Optional[Any] = None

        self._ensure_data_yaml()

    def _ensure_data_yaml(self) -> Path:
        # data.yaml bir kez yazılıp bırakılmaz; sınıf listesi değişince
        # güncellenir (F-1964).
        # P1-14 (K-38): ATOMİK yazım + yalnızca içerik değiştiyse yaz.
        # Eskiden düz write_text ile üzerine yazılıyordu: yazım yarıda
        # kesilirse data.yaml bozuk kalır, eğitim düşerdi. Şimdi aynı
        # dizinde geçici dosyaya yazılıp fsync + os.replace ile atomik
        # taşınır; içerik aynıysa dosyaya hiç dokunulmaz (mtime korunur).
        lines = [
            f"path: {self.dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(YOLO_12_CLASS_SPECS)}",
            "names:",
        ]
        for spec in YOLO_12_CLASS_SPECS:
            lines.append(f"  {spec['id']}: {spec['key']}")
        expected = "\n".join(lines) + "\n"
        current = ""
        if self.data_yaml_path.exists():
            try:
                current = self.data_yaml_path.read_text(encoding="utf-8")
            except OSError:
                current = ""
        if current != expected:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".data.yaml.", suffix=".tmp", dir=str(self.data_yaml_path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    tmp.write(expected)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                os.replace(tmp_name, self.data_yaml_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        return self.data_yaml_path

    def set_shared_yolo_detector(self, detector: Any) -> None:
        """Bind the app-wide ONNX detector; drop cached labeler so it rewires."""
        self._shared_yolo_detector = detector
        self._yolo_labeler = None

    def _get_yolo_labeler(self) -> Any:
        """Return the dialog-scoped YOLOAutoLabeler (created once, reused)."""
        if self._yolo_labeler is not None:
            return self._yolo_labeler
        from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

        kwargs: Dict[str, Any] = {}
        if self._shared_yolo_detector is not None:
            kwargs["yolo_detector"] = self._shared_yolo_detector
        self._yolo_labeler = YOLOAutoLabeler(**kwargs)
        return self._yolo_labeler

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
            labeler = self._get_yolo_labeler()
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
    ) -> Optional[Path]:
        """Görüntü üzerine renkli YOLO kutularını çizip `dataset/previews/` altına kaydeder."""
        canvas = frame.copy()
        img_h, img_w = canvas.shape[:2]

        for ann in annotations:
            # P2-23 (Y002-K1): geometrisi eksik kutuya şablon
            # (0.5/0.5/0.08) uydurulmaz — çizilmez.
            if any(ann.get(k) is None for k in ("cx", "cy", "w", "h")):
                continue
            cid = int(ann.get("class_id", 0))
            spec = CLASS_ID_TO_SPEC.get(cid)
            cname = str(ann.get("class_name") or (spec["key"] if spec else f"class_{cid}"))
            conf_raw = ann.get("confidence", ann.get("conf"))
            conf = float(conf_raw) if conf_raw is not None else None
            cx = float(ann.get("cx", 0.5))
            cy = float(ann.get("cy", 0.5))
            bw = float(ann.get("w", 0.08))
            bh = float(ann.get("h", 0.12))

            x1 = max(2, int((cx - 0.5 * bw) * img_w))
            y1 = max(2, int((cy - 0.5 * bh) * img_h))
            x2 = min(img_w - 2, int((cx + 0.5 * bw) * img_w))
            y2 = min(img_h - 2, int((cy + 0.5 * bh) * img_h))

            bgr = spec["color_bgr"] if spec is not None else (34, 197, 94)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2, cv2.LINE_AA)

            tag = f"{cname} %{conf * 100:.0f}" if conf is not None else f"{cname} · güven yok"
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
        if not _safe_imwrite_unicode(out_path, canvas, quality=94):
            logger.warning("Önizleme yazılamadı: %s", out_path)
            return None
        return out_path

    def bootstrap_from_existing_frames(
        self,
        augment_multiplier: Optional[int] = None,
        max_source_frames: int = 18,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Mevcut `reference_screenshots/`, `video_frames/` ve `video_analysis/` karelerini
        tarayarak otomatik etiketler ve 10x sentetik çoğaltma ile `dataset/` klasörünü zenginleştirir.

        P2-8 (DF.9): `should_stop` verilirse tarama kare sınırında
        durur — o ana kadar kaydedilen kareler diskte kalır (kısmi
        sonuç geçerlidir), dönüş sözlüğü `cancelled: True` taşır.
        İptal, yazılmış kareyi geri almaz; yarım kare yazılmaz
        (kontrol yalnızca kare/varyant sınırında yapılır).
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
        # P2-21 (D004-K1 / C012-K1): commit 0 kutu yazan kare
        # "kaydedildi" sayılmaz — ayrı sayılır.
        zero_box_frames = 0
        gate_skipped = 0
        # P2-8: iptal bayrağı — kare/varyant sınırında kontrol edilir.
        cancelled = False

        def _stop_requested() -> bool:
            if should_stop is None:
                return False
            try:
                return bool(should_stop())
            except Exception:
                return False

        existing_stems: Set[str] = set()
        for _img_dir in (self.train_images_dir, self.val_images_dir):
            if _img_dir.exists():
                existing_stems.update(p.stem for p in _img_dir.iterdir() if p.is_file())
        latest_preview: Optional[Path] = None
        timestamp_tag = int(time.time() * 1000)

        for idx, src_path in enumerate(source_paths):
            # P2-8: iptal yalnızca kare sınırında — yarım kare yazılmaz.
            if _stop_requested():
                cancelled = True
                logger.info(
                    "Bootstrap iptal edildi (%d/%d kare işlendi).",
                    idx,
                    len(source_paths),
                )
                break
            frame = _safe_imread_unicode(src_path)
            if frame is None or frame.size == 0:
                continue
            if any(s.endswith(f"_{src_path.stem}") for s in existing_stems):
                continue  # bu kaynak zaten bootstrap edilmiş (F-1952)

            base_anns = self._synthesize_annotations_for_frame(frame, src_path.stem, idx)
            # Kalite kapısı bootstrap yolunda da çalışır (F-1947): bulanık /
            # tekrar kareler ile geçersiz etiketler diske yazılmaz.
            gate_decision = self.quality_gate.evaluate_frame(frame, base_anns)
            if not gate_decision.accepted:
                gate_skipped += 1
                logger.debug(
                    "Bootstrap: kare atlandı (%s): %s", src_path.name, gate_decision.reason
                )
                continue
            base_anns = list(gate_decision.validated_annotations)
            split = gate_decision.split
            self.time_machine.push_frame(
                frame=frame,
                timestamp=time.time() - 2.0,
                detections=base_anns,
                region_name="Pilgrim's Ridge",
                force=True,
            )

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
                _n_boxes = int(info.get("train_box_count", 0))
                saved_boxes += _n_boxes
                if _n_boxes > 0:
                    saved_frames += 1
                else:
                    zero_box_frames += 1
                    logger.warning(
                        "Bootstrap: %s commit 0 kutu yazdı (dropped=%s) — kare kaydedildi sayılmadı",
                        stem,
                        info.get("dropped"),
                    )

            if idx == 0 or latest_preview is None:
                latest_preview = self.render_preview_image(
                    frame, base_anns, f"preview_{stem}.jpg"
                )

            if mult > 1:
                variants = self.augmentor.generate_augmented_variants(
                    frame, base_anns, count=min(mult - 1, 9)
                )
                for v_idx, (v_suffix, v_frame, v_anns) in enumerate(variants):
                    # P2-8: iptal varyant sınırında da kontrol edilir.
                    if _stop_requested():
                        cancelled = True
                        logger.info(
                            "Bootstrap iptal edildi (varyant %d, kaynak %s).",
                            v_idx,
                            src_path.name,
                        )
                        break
                    # Augment varyantlar baz kareyle AYNI split'te kalır; bağımsız
                    # split train/val sızıntısı yaratır (F-1945).
                    v_split = split
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
                        _n_v_boxes = int(v_info.get("train_box_count", 0))
                        saved_boxes += _n_v_boxes
                        if _n_v_boxes > 0:
                            saved_frames += 1
                        else:
                            zero_box_frames += 1

        self._ensure_data_yaml()
        stats = self.get_dataset_stats()
        stats.update(
            {
                "bootstrapped_frames_added": saved_frames,
                "bootstrapped_boxes_added": saved_boxes,
                "bootstrapped_zero_box_frames": zero_box_frames,
                "bootstrapped_gate_skipped": gate_skipped,
                "augment_multiplier": mult,
                # P2-8: kısmi tarama geçerli sonuçtur; iptal açıkça işaretlenir.
                "cancelled": bool(cancelled),
                "interrupted": bool(cancelled),
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

    # P2-20: GPU probe (nvidia-smi + import denemeleri) önbellek süresi.
    # Gerçek controller'daki `_get_gpu_info_cached` (15 sn) ile aynı
    # fikir: donanım bilgisi saniyeler içinde değişmez.
    _GPU_ENV_CACHE_TTL_SEC: float = 30.0

    def _get_gpu_environment_cached(self) -> Dict[str, Any]:
        """`probe_gpu_environment()` sonucunu 30 sn önbellekle döner.

        Yükleme worker'ı zaten probe eder; UI thread'deki tekrar
        çağrılar (dashboard, bootstrap bitişi) aynı sonucu önbellekten
        alır — her çağrıda subprocess + import denemesi yapılmaz.
        """
        now = time.monotonic()
        cached = getattr(self, "_gpu_env_cache", None)
        cached_at = float(getattr(self, "_gpu_env_cache_at", 0.0) or 0.0)
        if cached is not None and (now - cached_at) < self._GPU_ENV_CACHE_TTL_SEC:
            return dict(cached)
        env = probe_gpu_environment()
        self._gpu_env_cache = dict(env)
        self._gpu_env_cache_at = now
        return dict(env)

    def get_dataset_stats(
        self, gpu_environment: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """`dataset/` klasöründeki tüm train/val görüntülerini ve 12 sınıf kutu dağılımını hesaplar.

        P2-20: `gpu_environment` verilirse (yükleme worker'ının probe
        sonucu) `probe_gpu_environment()` çalıştırılmaz; verilmezse
        30 sn TTL'li örnek önbelleği kullanılır — UI thread'deki
        tekrar çağrılar (dashboard yenileme, bootstrap bitişi) her
        seferinde nvidia-smi + import denemesi yapmaz.
        """
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
        # P2-23 (D061-K1): bilinmeyen sınıf id'si sessizce atılmaz —
        # sayılır, id'leri toplanır, imza değişince uyarıyla işaretlenir.
        total_label_lines = 0
        unknown_class_boxes = 0
        unknown_class_ids: set = set()

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
                        except ValueError:
                            continue
                        total_label_lines += 1
                        if cid in class_counts:
                            class_counts[cid] = class_counts.get(cid, 0) + 1
                            total_boxes += 1
                        else:
                            unknown_class_boxes += 1
                            unknown_class_ids.add(cid)

        # Değişmeyen imzada tekrar uyarı yok (dashboard ~8 sn'de bir tazelenir).
        unknown_sig = (unknown_class_boxes, tuple(sorted(unknown_class_ids)))
        if unknown_class_boxes > 0 and unknown_sig != getattr(
            self, "_last_unknown_class_sig", None
        ):
            self._last_unknown_class_sig = unknown_sig
            logger.warning(
                "Bilinmeyen sınıf id'li %d kutu sayılmadı (id'ler: %s); "
                "satır=%d, sayılan=%d, bilinmeyen=%d",
                unknown_class_boxes,
                sorted(unknown_class_ids)[:20],
                total_label_lines,
                total_boxes,
                unknown_class_boxes,
            )

        previews = []
        if self.previews_dir.exists():
            previews = sorted(
                [p for p in self.previews_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

        total_frames = len(train_imgs) + len(val_imgs)
        progress_pct = min(100.0, (float(total_frames) / float(TARGET_DATASET_PHOTO_GOAL)) * 100.0)
        if isinstance(gpu_environment, dict) and gpu_environment:
            gpu_env = dict(gpu_environment)
        else:
            gpu_env = self._get_gpu_environment_cached()

        return {
            "total_frames": total_frames,
            "train_frames": len(train_imgs),
            "val_frames": len(val_imgs),
            "total_boxes": total_boxes,
            # P2-23 (D061-K1): total_boxes == sum(class_counts) değişmezi
            # bilinmeyen id'ler düşülünce körleşir; karşılaştırma için
            # ham satır sayısı + bilinmeyen sayımı da verilir.
            "total_label_lines": total_label_lines,
            "unknown_class_boxes": unknown_class_boxes,
            "unknown_class_ids": sorted(unknown_class_ids),
            "class_counts": class_counts,
            "target_photo_goal": TARGET_DATASET_PHOTO_GOAL,
            "progress_pct": round(progress_pct, 2),
            "latest_preview_path": str(previews[0]) if previews else "",
            "time_machine_active": bool(
                getattr(self.time_machine, "_total_pushed", 0) > 0 or len(self.time_machine) > 0
            ),
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
        self.setMinimumHeight(84)
        self.setMaximumHeight(96)
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
        layout.setSpacing(4)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['muted']}; font-size: 13px; font-weight: 700; "
            "border: none; background: transparent;"
        )

        self.value_lbl = QLabel(value_text)
        self.value_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['text']}; font-size: 20px; font-weight: 700; "
            "border: none; background: transparent; font-family: 'Segoe UI';"
        )
        self.value_lbl.setWordWrap(True)

        self.sub_lbl = QLabel(sub_text)
        self.sub_lbl.setStyleSheet(
            f"color: {SHELL_COLORS['muted']}; font-size: 13px; font-weight: 500; "
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

    SWITCH_WIDTH = 44
    SWITCH_HEIGHT = 22
    TEXT_SPACING = 12

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
            " padding: 14px 16px;"
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
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)
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
                width: 14px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #5a6578;
                border-radius: 4px;
                min-height: 32px;
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
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(5)
        self.title_lbl = QLabel(title, self)
        self.title_lbl.setStyleSheet(
            "font-size:14px;font-weight:800;background:transparent;border:none;color:#F2F3F5;"
        )
        self.sub_lbl = QLabel(subtitle, self)
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setStyleSheet(
            "font-size:12px;font-weight:500;background:transparent;border:none;color:#B7BFCB;"
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
        self.setMinimumHeight(44)
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
        return max(44, min(bounds.height() + 16, 72))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(240, self._needed_height(240))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(140, 44)

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
    `minimap_sprout`, `herb_edana`, `herb_lumina`) canlı olarak çizen bileşen.
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
        ("herb_edana", "#38bdf8"),
        ("herb_lumina", "#06b6d4"),
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
        # Önizleme görünüm modu: live (görüntü + kutular), boxes (yalnız kutular),
        # clean (yalnız görüntü). _set_preview_view ile değiştirilir (F-0605).
        self._view_mode: str = "live"
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

    def _emin_badge_threshold(self) -> float:
        """Rozet "Emin" eşiği = toplu onay eşiği (P1-12/K-31).

        paintEvent dialog'un metodunu çağırıyordu ama metot yalnızca
        YOLOFactoryDialog'da tanımlıydı — canvas'ta AttributeError ile
        boya baştan çöküyor, katman kalemleri + ✕ vuruş alanları hiç
        oluşmuyordu (w27/w70). Canvas ebeveyn dialog varsa onun
        eşiğini (bulk_range kaydırıcısı) alır, yoksa (başsız test)
        yedek CONF_BULK_DEFAULT (0.90) döner.
        """
        parent = self.parent()
        if parent is not None and hasattr(parent, "_emin_badge_threshold"):
            try:
                return float(parent._emin_badge_threshold())  # type: ignore[attr-defined]
            except Exception:
                pass
        return CONF_BULK_DEFAULT

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
        header_h = 32.0
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
                "QMenu::item { padding: 8px 20px; border-radius: 4px; font-size: 12px; }"
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
                # P0-8: her taşıma/boyutlandırma press'i mesafe sayacını
                # sıfırlar — önceki "creating" sürüklemesinden kalan
                # _is_dragging_box/_last_drag_distance, kutu içine tek
                # tıkı sürükleme saymasın (release eşiği bunları okur).
                self._is_dragging_box = False
                self._last_drag_distance = 0.0
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
                # P0-8 (K-12/D009): kutu içine tek tık (sürükleme yok) bir
                # düzenleme değildir — weak kutuyu el etiketine terfi
                # ettirmez. Eşik, "creating" dalındaki 8.0 px ile aynı:
                # mouseMove'da izlenen en büyük mesafe + bırakma anındaki
                # mesafe. Eşik altında boxEdited emit edilmez.
                pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                start = self._drag_start_pos
                if start is not None:
                    rdx = abs(float(pos.x() - start.x()))
                    rdy = abs(float(pos.y() - start.y()))
                    rdist = (rdx * rdx + rdy * rdy) ** 0.5
                else:
                    rdx = rdy = rdist = 0.0
                was_drag = (
                    self._is_dragging_box
                    or float(getattr(self, "_last_drag_distance", 0.0) or 0.0) >= 8.0
                    or rdist >= 8.0
                    or rdx >= 8.0
                    or rdy >= 8.0
                )
                if not was_drag:
                    self.clear_drag_state()
                    self.update()
                    super().mouseReleaseEvent(event)
                    return
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
            painter.drawRoundedRect(frame_rect, 10.0, 10.0)

            header_h = 32.0
            footer_h = 28.0
            view_rect = QRectF(
                8.0,
                header_h + 4.0,
                max(60.0, rect.width() - 16.0),
                max(60.0, rect.height() - header_h - footer_h - 10.0),
            )

            # Üst durum başlığı
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#38bdf8"))
            if self._empty or self._pixmap is None:
                file_badge = self._live_badge or "Canlı kare yok"
            elif self._live_badge:
                file_badge = self._live_badge
            else:
                file_badge = Path(self._preview_path).name if self._preview_path else "Canlı kare yok"
            painter.drawText(
                QRectF(12.0, 6.0, rect.width() - 24.0, 24.0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                f"🎯 CANLI OTOMATİK ETİKET ÖNİZLEME (t - 2.0 sn): {file_badge}",
            )

            # Görüntüyü çiz ("boxes" modunda görüntü çizilmez, yalnız kutular)
            target_rect = view_rect
            if getattr(self, "_view_mode", "live") == "boxes":
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#0f141c")))
                painter.drawRoundedRect(view_rect, 6.0, 6.0)
            elif self._pixmap is not None and not self._pixmap.isNull():
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
                # Boş durumun gerçek nedeni ortada görünsün: "Canlı kare
                # yok" sabit yazısı, görüntüsü bulunamayan karede de aynı
                # çıktığı için kullanıcı neden onaylayamadığını anlayamıyordu.
                empty_msg = self._live_badge or "Canlı kare yok"
                painter.drawText(
                    view_rect,
                    int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                    empty_msg,
                )
            self._last_target_rect = target_rect

            # Renkli YOLO BBox katmanı (herb_aria, herb_mela, ore_orichalcum, minimap_sprout, herb_edana, herb_lumina)
            tag_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            close_font = QFont("Segoe UI", 9, QFont.Weight.Black)
            painter.setFont(tag_font)
            fm = QFontMetrics(tag_font)
            self._close_hit_rects = []
            self._box_hit_rects = []
            self._box_draw_info = []
            # Bu çizim turunda yerleştirilen rozet dikdörtgenleri:
            # aynı noktaya yığılan rozetler aşağı kademelendirilir.
            placed_lbl_rects: List[QRectF] = []

            # "clean" modunda kutu katmanı çizilmez (F-0605).
            for idx, ann in enumerate(
                [] if getattr(self, "_view_mode", "live") == "clean" else self._annotations
            ):
                # P2-23 (Y002-K1): geometrisi eksik kutuya şablon
                # (0.5/0.5/0.08) uydurulmaz — çizilmez, ✕ vuruş alanı
                # da oluşmaz.
                if any(ann.get(k) is None for k in ("cx", "cy", "w", "h")):
                    continue
                cname = normalize_box_class_name(ann.get("class_id"), ann.get("class_name"))
                color_hex = str(ann.get("color", "#22c55e"))
                # P1-12 (K-32): ölçülmemiş güven "—" gösterilir — eski
                # 0.97 varsayılanı ölçülmemiş kutuyu %97 gösteriyordu.
                measured_conf = _measured_confidence(ann)
                norm_conf = measured_conf if measured_conf is not None else 0.0
                pct_txt = (
                    f"%{max(0, min(100, int(round(norm_conf * 100))))}"
                    if measured_conf is not None
                    else "—"
                )
                cx = float(ann.get("cx", 0.5))
                cy = float(ann.get("cy", 0.5))
                bw = float(ann.get("w", 0.08))
                bh = float(ann.get("h", 0.12))

                bx = target_rect.left() + (cx - bw * 0.5) * target_rect.width()
                by = target_rect.top() + (cy - bh * 0.5) * target_rect.height()
                bw_px = max(18.0, bw * target_rect.width())
                bh_px = max(14.0, bh * target_rect.height())

                box_rect = QRectF(bx, by, bw_px, bh_px)

                # P1-12 (K-31): rozet "Emin" eşiği = toplu onay eşiği
                # (bulk_range kaydırıcısı); ölçülmemiş güven Emin sayılmaz.
                is_confident = (
                    (
                        bool(ann.get("trusted"))
                        and measured_conf is not None
                        and norm_conf >= self._emin_badge_threshold()
                    )
                    or bool(ann.get("hand_label"))
                    or bool(ann.get("human_approved"))
                )

                if is_confident:
                    # 1. 🟢 EMİN: Yeşil (CONFIDENT_COLOR_HEX), düz kalın çizgi (width=2.2), rozet: '✓ cname %pct'
                    box_color = QColor(CONFIDENT_COLOR_HEX)
                    pen = QPen(box_color, 2.2)
                    pen.setStyle(Qt.PenStyle.SolidLine)
                    # El etiketi + ölçülmemiş güven (TXT kökenli, conf 0.0):
                    # uydurma "%0" yerine el işareti — güven yok, el var.
                    if bool(ann.get("hand_label")) and norm_conf <= 0.0:
                        label_str = f"✋ el {cname}"
                    else:
                        label_str = f"✓ {cname} {pct_txt}"
                    badge_bg_color = box_color
                    badge_text_color = QColor(CONFIDENT_TEXT_COLOR_HEX)
                    fill_alpha = 40
                elif norm_conf >= SUSPICIOUS_THRESHOLD:
                    # 2. 🟠 ŞÜPHELİ (SUSPICIOUS_THRESHOLD <= norm_conf < 0.85): Turuncu (SUSPICIOUS_COLOR_HEX), kesikli çizgi (width=1.8), rozet: '🟠 şüpheli cname %pct'
                    box_color = QColor(SUSPICIOUS_COLOR_HEX)
                    pen = QPen(box_color, 1.8)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    label_str = f"🟠 şüpheli {cname} {pct_txt}"
                    badge_bg_color = box_color
                    badge_text_color = QColor(SUSPICIOUS_TEXT_COLOR_HEX)
                    fill_alpha = 30
                else:
                    # 3. 🔴 ŞÜPHELİ EŞİĞİNİN ALTINDAKİLER (norm_conf < SUSPICIOUS_THRESHOLD): Kırmızı (LOW_CONFIDENCE_COLOR_HEX), kesikli çizgi (width=1.8), rozet: 'şüpheli cname %pct' (beyaz yazı #ffffff)
                    box_color = QColor(LOW_CONFIDENCE_COLOR_HEX)
                    pen = QPen(box_color, 1.8)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    label_str = f"şüpheli {cname} {pct_txt}"
                    badge_bg_color = box_color
                    badge_text_color = QColor(LOW_CONFIDENCE_TEXT_COLOR_HEX)
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
                # Rozet çakışması: aynı noktaya yığılan rozetler (örn.
                # kümelenmiş minimap noktaları) üst üste binip okunmaz
                # olur; çakışanı bir rozet boyu aşağı kademelendir.
                _probe = QRectF(lx_badge, ly, tw, th)
                for _ in range(16):
                    if not any(_probe.intersects(r) for r in placed_lbl_rects):
                        break
                    _probe.translate(0.0, th + 3.0)
                    if _probe.bottom() > target_rect.bottom():
                        _probe.moveBottom(target_rect.bottom())
                        break
                ly = _probe.top()
                lx_badge = _probe.left()
                placed_lbl_rects.append(_probe)
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
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            for tag_name, hex_col in self.REQUIRED_PREVIEW_TAGS:
                c = QColor(hex_col)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(c))
                painter.drawEllipse(QRectF(lx, legend_y + 5.0, 8.0, 8.0))
                painter.setPen(QColor("#cbd5e1"))
                painter.drawText(QPointF(lx + 11.0, legend_y + 13.0), tag_name)
                lx += float(len(tag_name) * 5.6 + 24.0)
        except Exception as exc:
            # Bozuk annotation UI thread'ini çökertmez; hata loglanır (F-1950).
            logger.warning("YOLOPreviewCanvas paintEvent hatası: %s", exc)
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
            " background-color: #0b0e14; border: 1px solid #1e293b; border-radius: 8px; padding: 6px;"
            "}"
            "QFrame#SearchableTagListPanel QLabel { background: transparent; color: #f1f5f9; }"
            "QLineEdit#TagSearchEdit { background: #111827; color: #ffffff; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; font-size: 12px; }"
            "QLineEdit#TagSearchEdit:focus { border: 1px solid #38bdf8; }"
            "QListWidget#TagListWidget { background: #07090e; color: #f1f5f9; border: 1px solid #1e293b; border-radius: 6px; font-size: 12px; }"
            "QListWidget#TagListWidget::item { padding: 5px 8px; border-radius: 4px; }"
            "QListWidget#TagListWidget::item:hover { background: #1e293b; }"
            "QListWidget#TagListWidget::item:selected { background: #166534; color: #ffffff; font-weight: 700; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Header with active class indicator
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 0, 2, 0)
        self.lbl_title = QLabel(f"🏷️ {len(CLASS_ID_TO_SPEC)} Sınıflı Etiket Listesi")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #38bdf8;")
        self.lbl_active_pill = QLabel("🎯 0: herb_aria")
        self.lbl_active_pill.setStyleSheet(
            "background: #0d2818; color: #4ade80; border: 1px solid #16a34a; border-radius: 5px; padding: 3px 8px; font-weight: 700; font-size: 11px;"
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
            "QComboBox { background: #111827; color: #ffffff; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; font-size: 12px; }"
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
        from dataset_factory.manual_tag_search import get_tag_registry, search_label_classes

        q = text.strip()
        q_low = q.lower()
        if q:
            results = search_label_classes(q)
        else:
            # P1-8 (K-21): search("") returns [] now (empty query matches
            # nothing). Clearing the search box must list the whole catalog
            # again, so the empty case lists explicitly via
            # get_all_classes() instead of searching.
            results = get_tag_registry().get_all_classes()
        matched_ids = {int(r["class_id"]) for r in results}
        # Arama, aktif çizim sınıfını değiştirmez (F-1948): yalnızca listeyi
        # filtreler; combo seçimi önceki aktif sınıfta kalır.
        prev_cid = self.combo.currentData()

        self.combo.blockSignals(True)
        self.combo.clear()
        for r in results:
            cid = int(r["class_id"])
            cname = str(r.get("class_name", f"class_{cid}"))
            label_tr = str(r.get("label_tr", cname))
            self.combo.addItem(f"{cid}: {cname} — {label_tr}", cid)
        if prev_cid is not None:
            restore_idx = self.combo.findData(prev_cid)
            if restore_idx >= 0:
                self.combo.setCurrentIndex(restore_idx)
        self.combo.blockSignals(False)

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            cid = int(item.data(Qt.ItemDataRole.UserRole))
            txt = item.text().lower()
            cname = str(item.data(Qt.ItemDataRole.UserRole + 1) or "").lower()
            visible = (not q_low) or (cid in matched_ids) or (q_low in txt) or (q_low in cname)
            item.setHidden(not visible)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        cid = int(item.data(Qt.ItemDataRole.UserRole))
        cname = str(item.data(Qt.ItemDataRole.UserRole + 1))
        self.set_active_class(cid, cname)
        self.classSelected.emit(cid, cname)

    def _on_combo_changed(self, idx: int) -> None:
        if idx < 0:
            return
        cid_data = self.combo.itemData(idx)
        if cid_data is not None:
            try:
                cid = int(cid_data)
                spec = CLASS_ID_TO_SPEC.get(cid, {})
                cname = spec.get("key", f"class_{cid}")
                self.set_active_class(cid, cname)
                self.classSelected.emit(cid, cname)
            except (ValueError, TypeError) as exc:
                logger.warning("Sınıf combo değişimi yoksayıldı (idx=%s): %s", idx, exc)

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
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)

        # Başlık ve açıklama
        title = QLabel("🏷️ Nesne Sınıfını Seçin")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #38BDF8;")
        lay.addWidget(title)

        sub = QLabel("Çizdiğin kutu hangi nesneye ait? Listeden seçin veya arama yapın:")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet("color: #A8B4C4;")
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
            "background: #1E293B; color: #C3CAD6; border: 1px solid #334155; border-radius: 6px; padding: 8px 16px; font-size: 13px;"
        )
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_ok = QPushButton("💾 Seç ve Kaydet (Enter)", self)
        self.btn_ok.setStyleSheet(
            "background: #166534; color: #F0FDF4; border: 1px solid #22C55E; border-radius: 6px; padding: 8px 18px; font-size: 13px; font-weight: 700;"
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
        self.setFixedHeight(34)
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

            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
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

            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
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

# B-02 (dilim 24) / F1-F8 (dilim 05): yayın/doğrulama kabul eşiği.
# Tek kaynak dataset_factory/dataset_validator.py'deki MIN_PUBLISH_MAP50'dir
# (motor + dialog aynı eşiği okur; v11 tazelik kapısı). config/schema.py'de
# mAP eşiği anahtarı YOK (doğrulandı) — tek kaynak bu sabittir. Motorun
# `target_accuracy_map50: 0.85` değeri ölü ayardır (hiçbir kod okumuyor);
# kapı 0.85 değil 0.50'dir: 0.32'lik yarım eğitimi engeller, ulaşılabilir
# eğitimi geçirir.
from dataset_factory.dataset_validator import MIN_PUBLISH_MAP50


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
    """Runs one callable off the GUI thread and the BotWorker thread.

    P2-18: işbirlikçi `stop()` bayrağı — kapanış ağı bu worker'ı da
    durdurur.
    P2-8 (K-17 / DF.9): `run()` artık tek senkron çağrıyı körlemesine
    beklemez — `_fn` `should_stop` parametresini kabul ediyorsa
    (açık parametre veya `**kwargs`) worker kendi
    `is_stop_requested` bayrağını ona geçirir; tarama kare
    sınırında durur. Kabul etmeyen eski çağrılabilirler eskisi gibi
    `fn()` ile çağrılır (çağrı öncesi/sonrası bayrak kontrolü aynen
    durur). Böylece diyalog kapanırken worker dakikalarca süren
    taramanın bitmesini beklemez — kapanış SIGABRT'siz tamamlanır.
    """

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._stop_requested = False

    def is_stop_requested(self) -> bool:
        return bool(
            getattr(self, "_stop_requested", False) or self.isInterruptionRequested()
        )

    def stop(self) -> None:
        """Kapanıştan durdurma isteği: başlamadıysa hiç çalışmaz."""
        self._stop_requested = True
        self.requestInterruption()

    @staticmethod
    def _fn_accepts_should_stop(fn: Callable[..., Any]) -> bool:
        """`fn(should_stop=...)` çağrılabilir mi? İmza okunamazsa False."""
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        if "should_stop" in params:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def run(self) -> None:
        threading.current_thread().name = "GatherionDatasetBootstrap"
        if self.is_stop_requested():
            self.failed.emit("Veri seti taraması durduruldu.")
            return
        try:
            if self._fn_accepts_should_stop(self._fn):
                result = self._fn(should_stop=self.is_stop_requested)
            else:
                result = self._fn()
        except Exception as exc:
            logger.error("Arka plan veri seti taraması başarısız: %s", exc, exc_info=True)
            self.failed.emit(str(exc))
            return
        if self.is_stop_requested():
            self.failed.emit("Veri seti taraması durduruldu.")
            return
        self.succeeded.emit(result)


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
                except Exception as exc:
                    logger.warning("stop_training hatası: %s", exc)
            elif hasattr(ctrl, "rtx_trainer") and hasattr(ctrl.rtx_trainer, "stop_training"):
                try:
                    ctrl.rtx_trainer.stop_training()
                except Exception as exc:
                    logger.warning("rtx_trainer.stop_training hatası: %s", exc)
            elif hasattr(ctrl, "trainer") and hasattr(ctrl.trainer, "stop_training"):
                try:
                    ctrl.trainer.stop_training()
                except Exception as exc:
                    logger.warning("trainer.stop_training hatası: %s", exc)

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
        # Veri setinde kare olması bootstrap'ı tamamen atlamaz (F-1952);
        # bootstrap_from_existing_frames yalnızca zaten eklenmiş kaynakları atlar.
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
                # B-10: havuz hazırlığı eskiden burada da yutuluyordu
                # (logger.debug + devam) — eğitim eski/yanlış setle
                # başlıyordu. Dondurma havuzu hazırladı; worker yalnızca
                # onaylı yaml'ın varlığını doğrular, yoksa eğitimi
                # başlatmaz.
                try:
                    dataset_dir = getattr(self.controller, "dataset_dir", Path("dataset"))
                    app_yaml = Path(dataset_dir) / "approved" / "data_approved.yaml"
                    if not app_yaml.is_file():
                        raise RuntimeError(f"Onaylı eğitim yaml'ı yok: {app_yaml}")
                except Exception as exc:
                    self.training_failed.emit(
                        f"Onaylı eğitim verisi hazır değil ({exc}) — önce Adım 3'te dondurun."
                    )
                    return
            prepared = self._bootstrap_dataset()
            line = format_dataset_bootstrap_complete(prepared)
            if line:
                self.status_text.emit(line)
            blocked = _game_blocks_cuda_training()
            if blocked:
                self.status_text.emit(f"Eğitim Durumu: {blocked}")
                self.training_deferred.emit(blocked)
                return
            # P0-5 tur 2 (K-30): yaml kapısının bootstrap-sonrası durağı.
            # Tık anından buraya taşındı: bootstrap etiketli kareleri
            # yazıp `data.yaml`'ı düzeltebilir, kapı onarımı veto
            # etmemeli. Eğitim başlamadan önce MUTLAKA çalışır — motor
            # kapısı (`resolve_training_yaml`) ikinci savunma hattıdır.
            # Oyun ertelemesi önceliklidir (üstteki kapı), yaml hatası
            # terminaldir. Doğrulama kendisi çökerse kilitleme (fail-open).
            try:
                from dataset_factory.yolo_trainer import validate_training_yaml as _validate_yaml

                _worker_ds = Path(getattr(self.controller, "dataset_dir", Path("dataset")))
                _worker_yaml = (
                    _worker_ds / "approved" / "data_approved.yaml"
                    if self.only_approved
                    else _worker_ds / "data.yaml"
                )
                _worker_report = _validate_yaml(_worker_yaml)
            except Exception as exc:  # doğrulama kendisi çökerse eğitimi kilitleme
                logger.warning("Eğitim yaml doğrulaması çalıştırılamadı: %s", exc)
                _worker_report = {"ok": True, "errors": []}
            if not _worker_report.get("ok", False):
                _worker_reason = "; ".join(
                    str(e) for e in _worker_report.get("errors") or ["bilinmeyen yaml hatası"]
                )
                _worker_msg = f"Eğitim yaml'ı geçersiz, eğitim başlatılmadı: {_worker_reason}"
                logger.warning("%s", _worker_msg)
                # Non-modal: training_failed → _on_training_failed statü
                # etiketi + log yazar; QMessageBox yok, hang yok (8bfb9c0).
                self.training_failed.emit(_worker_msg)
                return
            # P1-09 (K-24): dondurma-sayı kapısı, worker tarafı —
            # bootstrap sonrası (bootstrap havuzu değiştirmez, ama
            # dondurma sonrası onay/budama havuzu değiştirmiş olabilir).
            # Yalnızca onaylı modda ve manifesto karşılaştırılabilirse
            # geçerlidir; kontrolün kendisi çökerse eğitim kilitlenmez
            # (motor kapısı ikinci savunma hattıdır).
            if self.only_approved:
                try:
                    from dataset_factory.approved_pool import (
                        check_frozen_manifest_consistency as _check_frozen,
                    )

                    _frozen_check = _check_frozen(
                        Path(getattr(self.controller, "dataset_dir", Path("dataset")))
                    )
                    if _frozen_check.get("checked") and not _frozen_check.get("ok"):
                        _frozen_msg = (
                            "Dondurulan veri seti havuzla uyuşmuyor, eğitim başlatılmadı: "
                            + str(_frozen_check.get("reason") or "")
                            + " Önce Adım 3'te yeniden dondurun."
                        )
                        logger.warning("%s", _frozen_msg)
                        self.training_failed.emit(_frozen_msg)
                        return
                except Exception as exc:
                    logger.warning("Dondurma karşılaştırması çalıştırılamadı (worker): %s", exc)
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
        # P2-18: süresiz bekleme yok — `wait_for_training` sonlu timeout
        # (3600 sn) ile çağrılır; süre dolarsa motor `timed_out` döner ve
        # aşağıdaki `state == "running"` dalı bunu başarısız sayar.
        if isinstance(res, dict) and res.get("started") and not res.get("mAP50") and (
            res.get("bootstrap_pending") or res.get("thread_name")
        ):
            trainer = getattr(controller, "rtx_trainer", getattr(controller, "trainer", None))
            if trainer is not None and hasattr(trainer, "wait_for_training"):
                waited = trainer.wait_for_training(timeout=3600.0)
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
def _resolve_review_image_path(dataset_dir: Path, split: str, stem: str) -> Optional[Path]:
    """stem → images/{split|train|val}/stem.* sonra previews/stem*_preview.* (saf Path işi)."""
    stem = str(stem or "").strip()
    if not stem:
        return None
    dataset = Path(dataset_dir)
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
    # GameLearner (core/omni_game_learner._save_learning_sample): learn_*
    # kareleri önce game_learning/<kategori>/ altına yazılır; train
    # kopyası commit 0 kutu yazınca geri alınır (silinir) ama review JSON
    # + game_learning kopyası inceleme kuyruğu için kalır. Bu yüzden
    # images/ altında bulunamayan learn_*/mix_* karelerinin görüntüsü
    # buradadır — aranmazsa kuyruk kartı "…" kalır, ana panel boş görünür.
    gl_root = dataset / "game_learning"
    if gl_root.is_dir():
        cat_dirs: List[Path] = []
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "learn" and parts[1]:
            cat_dirs.append(gl_root / parts[1])
        try:
            subdirs = [p for p in gl_root.iterdir() if p.is_dir()]
        except OSError:
            subdirs = []
        for d in cat_dirs + [p for p in subdirs if p not in cat_dirs]:
            for s in stems:
                for ext in exts:
                    cand = d / f"{s}{ext}"
                    if cand.is_file():
                        return cand
    return None



_QUEUE_RENDER_SYNC_LIMIT = 60  # bu satır sayısına kadar liste senkron kurulur
_QUEUE_RENDER_CHUNK = 20  # büyük listelerde olay-döngüsü başına eklenen satır

# El etiketi (TXT) köken işareti: tarama, review JSON'u olmayan ama labels
# TXT'si olan karenin kutularına bu kaynağı yazar (düzeltilmiş el
# etiketleri, örn. 1193 karelik yükleme). Bu kutular budama/reddet
# korumalıdır; kaynak işareti persist'te korunur
# (`_normalize_box_dict` source'u saklar), böylece koruma yeniden
# yüklemeden sonra da sürer. `human_approved` değildir — B-07: ölçülmemiş
# güven toplu onaya giremez; yalnızca silinmeye karşı korunur.
HAND_LABEL_TXT_SOURCE = "hand_label_txt"


# [YOLO-ASYNC] Arka plan yükleme: dataset taraması + kuyruk kurulumu +
# görünür satır küçük resimleri. Bu sınıflar QWidget'e DOKUNMAZ; sonuçlar
# signal/slot ile UI thread'e döner.


# ----------------------------------------------------------------------
# [YOLO-ASYNC] Arka plan thread güvenlik ağı
# ----------------------------------------------------------------------
# Dialog, kendi buton/kuyruk sinyallerine bağlı yöntemlerle C++ tarafında
# güçlü referansla tutulur; CPython GC bu döngüyü göremez, dialog test
# fixture'larında close() edilmeden bırakılırsa hiç toplanmaz. Çalışan
# bir QThread kapanışta yok edilirse süreç abort olur (exit 134). Bu kayıt
# defteri + atexit kancası, dialog toplanmasa bile kapanışta tüm YOLO arka
# plan thread'lerini durdurur (stop + quit + wait). Bitmiş/silinmiş
# thread'ler WeakSet'ten kendiliğinden düşer.
_YOLO_BACKGROUND_THREADS: "weakref.WeakSet[Any]" = weakref.WeakSet()


# P2-18 kapanış bütçeleri (seri en kötü durum ~32 sn: eğitim 10 + diğerleri
# 5'er + küçük resim 2; pratikte aynı anda en fazla 1-2 thread çalışır).
_YOLO_THREAD_STOP_TIMEOUT_MS = 5000
_YOLO_TRAIN_STOP_TIMEOUT_MS = 10000
_YOLO_THUMB_STOP_TIMEOUT_MS = 2000
_YOLO_TERMINATE_WAIT_MS = 2000


def _stop_qthread_gracefully(
    thread: Any,
    name: str,
    timeout_ms: int = _YOLO_THREAD_STOP_TIMEOUT_MS,
    worker: Any = None,
) -> bool:
    """TEK kapanış deseni (main.py:2851-2861, F-1194).

    Sıra: işbirlikçi `stop()` → `requestInterruption()` → `quit()` →
    sonucu kontrol edilen `wait(timeout)` → durmadıysa `terminate()` +
    tekrar `wait()` + log. `wait()` sonucu hiçbir yerde yok sayılmaz;
    False dönerse terminate/log yolu çalışır.

    `stop()` yalnızca gerçekten tanımlıysa çağrılır: düz `QThread`'te
    `stop()` yoktur (P2-18) — `worker` QObject'i (load/bulk/purge) ya da
    `stop()` tanımlı QThread alt sınıfı (eğitim/küçük resim/bootstrap)
    verilirse onun `stop()`'u çağrılır, düz thread'de adım atlanır.
    Çalışmayan/silinmiş thread'de hemen True döner.
    """
    if thread is None:
        return True
    try:
        if not thread.isRunning():
            return True
    except RuntimeError:
        return True  # C++ nesnesi silinmiş
    except Exception as exc:
        logger.warning("[YOLO] %s durumu okunamadı: %s", name, exc)
        return False
    stop_target = worker if worker is not None else thread
    stop = getattr(stop_target, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception as exc:
            logger.warning("[YOLO] %s stop() hatası: %s", name, exc)
    try:
        thread.requestInterruption()
    except Exception:
        pass
    try:
        thread.quit()
    except Exception:
        pass
    try:
        stopped = bool(thread.wait(int(timeout_ms)))
    except Exception as exc:
        stopped = False
        logger.warning("[YOLO] %s bekleme hatası: %s", name, exc)
    if stopped:
        return True
    try:
        if not thread.isRunning():
            return True
    except Exception:
        return False
    logger.warning(
        "[YOLO] %s %d ms'de durmadı; son çare olarak terminate() uygulanıyor.",
        name,
        int(timeout_ms),
    )
    try:
        thread.terminate()
        stopped = bool(thread.wait(_YOLO_TERMINATE_WAIT_MS))
    except Exception as exc:
        stopped = False
        logger.warning("[YOLO] %s terminate hatası: %s", name, exc)
    if not stopped:
        logger.error("[YOLO] %s terminate() sonrası bile çalışıyor.", name)
    return bool(stopped)


def _stop_all_yolo_background_threads() -> None:
    """Kapanışta kayıtlı tüm YOLO arka plan thread'lerini durdur (atexit).

    P2-18: atexit'in kendi `stop()` denemesi + yok sayılan `wait(5000)`
    kopyası kaldırıldı — atexit artık closeEvent/`__del__` ile AYNI
    `_stop_qthread_gracefully` desenini kullanır. Düz `QThread`'te
    `stop()` olmadığı için atexit thread'e `stop()` denemez; işbirlikçi
    durdurma yalnızca `stop()` tanımlı alt sınıf thread'lerde çalışır.
    """
    for thread in list(_YOLO_BACKGROUND_THREADS):
        try:
            _stop_qthread_gracefully(thread, "YOLO arka plan thread'i")
        except Exception:
            pass


atexit.register(_stop_all_yolo_background_threads)


class _YoloDataLoadWorker(QObject):
    """Worker thread: dataset istatistikleri + onay kuyruğu taraması.

    Yalnızca disk I/O yapar (helper.get_dataset_stats + saf kuyruk taraması);
    sonuçlar finished/failed signal'leriyle UI thread'e döner.
    """

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict, list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dataset_dir: Path,
        rejected_stems: set,
        review_audit_mode: bool,
        queue_filter_mode: str,
        helper: Any,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_dir = Path(dataset_dir)
        self._rejected_stems = set(rejected_stems or set())
        self._review_audit_mode = bool(review_audit_mode)
        self._queue_filter_mode = str(queue_filter_mode or "all")
        self._helper = helper
        # P1-16 (K-17): taramada karantinaya alınan bozuk dosyalar —
        # finished sinyali sonrası UI thread'de `_on_yolo_data_loaded`
        # tarafından okunup kullanıcıya bildirilir.
        self.quarantine_events: List[Dict[str, Any]] = []
        # P2-18: işbirlikçi durdurma bayrağı — `quit()` çalışan `run()`
        # slotunu bölmez, bu bayrak faz sınırlarında kontrol edilir.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Kapanıştan durdurma isteği (faz sınırında durur)."""
        self._stop_event.set()

    def is_stop_requested(self) -> bool:
        return bool(getattr(self, "_stop_event", None) is not None and self._stop_event.is_set())

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.progress.emit("dataset taranıyor…")
            if self.is_stop_requested():
                self.failed.emit("Yükleme durduruldu.")
                return
            disk_stats: Dict[str, Any] = {}
            if self._helper is not None:
                disk_stats = self._helper.get_dataset_stats() or {}
            self.progress.emit("onay kuyruğu kuruluyor…")
            quarantine_events: List[Dict[str, Any]] = []
            if self.is_stop_requested():
                self.failed.emit("Yükleme durduruldu.")
                return
            items = YOLOFactoryDialog._scan_review_queue_items(
                self._dataset_dir,
                self._rejected_stems,
                None,
                self._review_audit_mode,
                self._queue_filter_mode,
                quarantine_events=quarantine_events,
            )
            if self.is_stop_requested():
                # Tarama bitti ama kapanış istendi: sonucu uygulama.
                self.failed.emit("Yükleme durduruldu.")
                return
            self.quarantine_events = list(quarantine_events)
            self.progress.emit(f"kuyruk kuruldu ({len(items)} kare)")
            self.finished.emit(dict(disk_stats), list(items))
        except Exception as exc:
            logger.warning("[YOLO] worker yükleme hatası: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


# ----------------------------------------------------------------------
# [YOLO-BULK-ASYNC] Toplu onay / zehir temizliği arka plan işçileri
# ----------------------------------------------------------------------
# Canlı log kanıtı (2026-11-10, Selçuk'un PC'si): "Filtredekileri Onayla"
# UI thread'i 50,8 sn kilitledi (379 kare; 14 kutu toplu işaretlendi, 770
# el etiketi korumalı kare atlandı), "Zehirli otomatik temizle" 18,8 sn
# kilitledi — pencere "(Yanıt Vermiyor)" oldu. Bu işçiler disk yazımını
# worker thread'e taşır; ilerleme signal ile UI thread'e döner, overlay
# "132/1180" sayacını ana thread'de günceller.
#
# Eşik (_BULK_ASYNC_MIN_ITEMS): 40 kareye kadar senkron yol korunur —
# küçük toplu işlem anlık biter, mevcut senkron testler (35 karelik
# processEvents sayacı, tıklama sonrası anlık assert) aynen geçer.
# Ölçüm (VM, sentetik tmp dataset, 2026-11-10): senkron onay ~22 ms/kare
# (60 kare 1,31 sn) — 40 kare ~0,9 sn, processEvents 15 karede bir
# UI boşluğunu ~0,3 sn'de tutar. Canlı veride kare başı ~134 ms
# (50,8 sn / 379 kare) olduğundan 40 kare üstü async'e girer.
_BULK_ASYNC_MIN_ITEMS = 40


def _process_bulk_approve_items(
    dialog: Any,
    items: List[Dict[str, Any]],
    threshold: float,
    mode: str,
    *,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional["threading.Event"] = None,
) -> Dict[str, Any]:
    """Toplu onay çekirdeği — senkron ve async yolun TEK kaynağı.

    `items` yerinde güncellenir (senkron yolda canlı liste, async yolda
    worker'a verilen derin kopyalar). QWidget'e dokunmaz: UI yazıları
    `log_fn` ile arayana bırakılır, ilerleme `progress_cb(done, total)`
    ile raporlanır. Dialog metotlarından yalnızca disk-saf olanlar
    çağrılır (`_is_hand_label_protected_item`, `_persist_review_item`,
    `_frame_conf`) — hiçbiri QWidget okumaz/yazmaz.

    Koruma (prune-güvenlik dalı, aynen): el etiketi korumalı kare atlanır
    (`skipped_hand`, `[KORUMA]` log). W46: 0 kutulu (boş) kare toplu
    onaya GİREMEZ — eski kod `mode == "empty"` ise `done="ok"` yazıp
    `_persist_review_item` B-02 ön-kontrolünde reddediyordu (yanıltıcı
    `[HATA]` loguyla); artık açıkça atlanır (`skipped_empty`, `[W46]` log),
    diske hiç yazılmaz.

    `n` YALNIZCA `done="ok"` alıp diske yazılan kareleri sayar (P2-21 /
    F029-K2): eşik altı kutusu kalıp kuyrukta kalan kısmi kare sayılmaz;
    trusted-only (el etiketli) kare ayrıca `trusted_only_approved`
    ile sayılır. İlerleme `progress_cb(islenen, toplam, onaylanan)`
    üçlüsüyle raporlanır — overlay metni onaylananı sayar (F043-K2).

    P0-7/K-13: koruma kontrolü weak dalında da çalışır — `not weak`
    dalındakiyle AYNI predikat (`_is_hand_label_protected_item`).
    Korumalı karede weak kutular işaretlenebilir ama trusted el kutuları
    ayrı tutulur ve kare asla `done="ok"` / `human_approved` almaz
    (tek tek inceleme için kuyrukta kalır, `n`'ye girmez;
    `protected_weak_marked` sayılır).
    K-21 / P1-10: toplu onay HİÇBİR kareye `human_approved` yazmaz —
    onaylanan kare `done="ok"` + kare düzeyi `bulk_approved` alır;
    havuz kapısı (`_pool_add_approved` + dondurma) bunu tek başına
    kabul etmez, yalnızca kutu düzeyinde insan onayı / el etiketi
    olan kutular havuza girer.
    """
    total = len(items)
    n = 0
    bulk_marked_total = 0
    skipped_hand = 0
    skipped_empty = 0
    protected_weak_marked = 0
    # P2-21 (F029-K2): 0 kutu işaretleyerek onaylanan el-etiketli
    # (trusted-only) kare sayısı — özet logda `n`'den ayrı gösterilir,
    # "379 kare, 14 kutu işaretlendi" çelişkisi kalmaz.
    trusted_only_approved = 0
    errors: List[str] = []
    stopped = False
    th = float(threshold)

    def _emit_progress(step: int) -> None:
        # P2-21 (F043-K2): ilerleme 3'lüdür — (işlenen, toplam,
        # onaylanan). Overlay metni onaylananı sayar; çubuk işleneni
        # gösterir (çubuk onaylananla dolsa atlanan karelerde takılır
        # görünürdü). 2 argümanlı eski çağrılar yok — iki üretim
        # çağrı noktası da (_sync_progress, _BulkApproveWorker) 3'lü
        # imzaya geçirildi.
        if progress_cb is not None:
            progress_cb(step, total, n)

    for i_step, it in enumerate(items, 1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        if not isinstance(it, dict):
            _emit_progress(i_step)
            continue
        if it.get("done"):
            _emit_progress(i_step)
            continue

        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])

        if not weak:
            # B-07 + el etiketi koruması: TXT kökenli el etiketi karesi
            # toplu onaya GİREMEZ — kare kuyrukta kalır, tek tek incelenir.
            if dialog._is_hand_label_protected_item(it):
                skipped_hand += 1
                logger.debug(
                    "[KORUMA] #%s: el etiketi korumalı kare toplu onaya girmedi",
                    it.get("stem"),
                )
            elif not trusted:
                # [W46] 0 kutulu kare: ne weak ne trusted var — onaylanacak
                # geçerli kutu yok; done="ok" yazılmaz, diske yazılmaz.
                skipped_empty += 1
                logger.info(
                    "[W46] #%s: 0 kutulu kare toplu onaya girmedi (boş kare onaylanmaz)",
                    it.get("stem"),
                )
            else:
                # K-21 / P1-10: toplu onay kareye `human_approved`
                # YAZMAZ — yalnızca `done="ok"` + `bulk_approved` izi.
                # Kutulara tek tek bakılmadı; havuz kapısı bu karesi
                # kökensiz trusted kutuyu almaz (`_pool_add_approved`).
                it["done"] = "ok"
                it["bulk_approved"] = True
                if dialog._persist_review_item(it):
                    # P2-21 (F029-K2): trusted-only kare 0 kutu
                    # işaretleyerek onaylanır — kare onaylıdır (tüm
                    # kutuları zaten trusted el etiketi) ama özet
                    # logda ayrı sayılır.
                    n += 1
                    trusted_only_approved += 1
                else:
                    it["done"] = None  # S-1502: yazılamadıysa kare kuyrukta kalır
                    it.pop("bulk_approved", None)
                    msg = f"[HATA] #{it.get('stem')}: toplu onay kaydı diske yazılamadı."
                    errors.append(msg)
                    if log_fn is not None:
                        log_fn(msg)
            _emit_progress(i_step)
            continue

        # P0-7/K-13: koruma kontrolü weak dalında da — `not weak`
        # dalındakiyle AYNI predikat. Korumalı karede weak kutular
        # işaretlenir ama kare `done="ok"` / `human_approved` almaz.
        frame_protected = dialog._is_hand_label_protected_item(it)

        kept_weak = []
        bulk_marked = []
        for b in weak:
            b_conf = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
            norm_b_conf = b_conf if b_conf <= 1.0 else b_conf / 100.0
            # P1-12 (K-35/F017-K2): uydurma güven ölçüm değildir — eşik
            # üstü görünse bile toplu onay işareti almaz, kare done="ok"
            # olmaz; weak olarak kalır, insan tek tek inceler.
            if b.get("confidence_fabricated") is True:
                kept_weak.append(b)
                continue
            if norm_b_conf >= th:
                # Eşik üstü kutu toplu işaret alır, terfi etmez (B-03).
                nb = dict(b)
                nb["bulk_approved"] = True
                nb["trusted"] = False
                nb["weak"] = True
                bulk_marked.append(nb)
                bulk_marked_total += 1
            else:
                kept_weak.append(b)

        if bulk_marked or (mode == "suspicious" and not kept_weak):
            it["weak"] = bulk_marked + kept_weak
            it["n_weak"] = len(it["weak"])
            if not kept_weak and not frame_protected:
                # K-21 / P1-10: weak-only kare `done="ok"` + yalnızca
                # `bulk_approved` alır — `human_approved` almaz; havuza
                # 0 katkıyla girer (trusted kutu yok, havuz adımı
                # atlanır). Kuyruktan çıkması kapıları kilitlemesin
                # diye `done` verilir, onay sayılmaz.
                it["done"] = "ok"
                it["bulk_approved"] = True
            it["conf"] = dialog._frame_conf(it)
            if dialog._persist_review_item(it):
                if frame_protected:
                    # İşaretler diske yazıldı ama kare onaylanmadı:
                    # `n`'ye girmez, kuyrukta kalır.
                    protected_weak_marked += 1
                elif it.get("done") == "ok":
                    # P2-21 (F029-K2): kısmi kare sayılmaz — eşik altı
                    # kutusu kalıp `done` almadan kuyrukta kalan kare
                    # `n`'ye giremez; eskiden persist başarısı
                    # tek başına `n += 1` yapıyordu.
                    n += 1
            else:
                it["done"] = None  # S-1502: yazılamadıysa kare kuyrukta kalır
                it.pop("bulk_approved", None)
                msg = f"[HATA] #{it.get('stem')}: toplu onay kaydı diske yazılamadı."
                errors.append(msg)
                if log_fn is not None:
                    log_fn(msg)

        _emit_progress(i_step)

    return {
        "n": n,
        "bulk_marked_total": bulk_marked_total,
        "skipped_hand": skipped_hand,
        "skipped_empty": skipped_empty,
        "protected_weak_marked": protected_weak_marked,
        "trusted_only_approved": trusted_only_approved,
        "errors": errors,
        "stopped": stopped,
        "total": total,
    }


def _bulk_approve_pool_readd(
    dialog: Any, items: List[Dict[str, Any]]
) -> Tuple[List[str], int]:
    """Onaylı (done="ok") + trusted kutulu kareleri havuza tekrar ekle.

    `_persist` zaten her kareyi havuza ekledi; bu tur yalnızca
    tekrar-eklemedir. Hatası sessizce yutulmaz (B-02): eklenemeyen
    stem'ler döner, arayan loglar. Senkron yolda canlı liste, async
    yolda worker kopyaları üzerinde çalışır.

    K-21 / P1-10: havuz kapısı (`_pool_add_approved`) kökensiz kutuyu
    eler — bu HATA değil, 0 katkıdır. Döner:
    ``(pool_failed, skipped_no_provenance)`` — ikincisi hiçbir kutusu
    havuza girmeden atlanan kare sayısıdır (özet log dürüst olsun).
    """
    pool_failed: List[str] = []
    skipped_no_provenance = 0
    for it in list(items):
        if it.get("done") == "ok" and list(it.get("trusted", []) or []):
            if not dialog._pool_add_approved(it, list(it.get("trusted", []))):
                pool_failed.append(str(it.get("stem", "")))
            elif int(it.get("_pool_added_boxes", 0) or 0) == 0:
                skipped_no_provenance += 1
    if pool_failed:
        logger.warning(
            "bulk_approve: %d kare havuza tekrar eklenemedi: %s",
            len(pool_failed),
            ", ".join(pool_failed[:10]),
        )
    return pool_failed, skipped_no_provenance


class _BulkApproveWorker(QObject):
    """Worker thread: toplu onay disk yazımı + havuz tekrar-ekleme.

    Girdi `items` (hedef karelerin derin kopyaları) + `all_items`
    (havuz tekrar-ekleme turu için kuyruğun derin kopyası); sonuç
    `finished` ile UI thread'e döner — worker QWidget'e dokunmaz.
    `stop()` işbirlikçi durdurmadır: kare sınırında durur, kısmi
    sonuç `finished` ile döner (`stopped=True`).
    """

    progress = pyqtSignal(int, int, int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dialog: Any,
        items: List[Dict[str, Any]],
        all_items: List[Dict[str, Any]],
        threshold: float,
        mode: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dialog = dialog
        self._items = items
        self._all_items = all_items
        self._threshold = float(threshold)
        self._mode = str(mode or "all")
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = _process_bulk_approve_items(
                self._dialog,
                self._items,
                self._threshold,
                self._mode,
                progress_cb=lambda d, t, a: self.progress.emit(int(d), int(t), int(a)),
                log_fn=None,
                stop_event=self._stop_event,
            )
            # Havuz tekrar-ekleme: işlenmiş kopyaları tam kuyruk
            # kopyasına bindir, sonra done="ok" + trusted turu.
            by_stem = {str(it.get("stem") or ""): it for it in self._all_items}
            for it in self._items:
                by_stem[str(it.get("stem") or "")] = it
            result["pool_failed"], result["pool_skipped_no_provenance"] = (
                _bulk_approve_pool_readd(self._dialog, list(by_stem.values()))
            )
            result["items"] = self._items
            result["stopped"] = bool(result.get("stopped"))
            self.finished.emit(dict(result))
        except Exception as exc:
            logger.warning("[ONAY] worker hatası: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


# ----------------------------------------------------------------------
# [YOLO-P2-20] Kutu budama çekirdeği + worker
# ----------------------------------------------------------------------
# Budama eskiden tamamen senkrondu (F022-K1, D022): dış budama + fallback
# döngüsü + sonda senkron reload UI thread'de çalışıyordu. Çekirdek
# `_process_prune_items` senkron ve async yolun TEK kaynağıdır
# (`_process_bulk_approve_items` deseni): senkron yolda canlı liste,
# async yolda worker'a verilen derin kopyalar üzerinde çalışır.
# QWidget'e dokunmaz: ilerleme `progress_cb(done, total)` ile, log
# `log_fn` ile arayana bırakılır. Dialog metotlarından yalnızca
# disk-saf olanlar çağrılır (`_is_hand_label_protected_item`,
# `_persist_review_item`, `_frame_conf`, `_resolve_review_image`,
# `_learn_box_exemplar` — sonuncusu thread-safe LRU decode önbelleği
# üzerinden okur). `processEvents()` bu çekirdekte YOKTUR; senkron
# sarmalayıcı 15 karede bir kendi `processEvents`'ini yapar.
#
# Eşik (`_BULK_ASYNC_MIN_ITEMS`, toplu onayla aynı): 40 kareye kadar
# senkron yol korunur — küçük budama anlık biter, mevcut senkron
# testler (1-2 karelik prune guard testleri) aynen geçer. Üzeri
# worker'a gider; sonda kuyruk yenileme worker bitişine bağlanır
# (senkron yolda `_reload_review_queue`, async yolda
# `_request_queue_reload`).
def _process_prune_items(
    dialog: Any,
    items: List[Dict[str, Any]],
    threshold: float,
    *,
    dataset_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional["threading.Event"] = None,
) -> Dict[str, Any]:
    """Budama çekirdeği — dış + fallback, senkron ve async yolun TEK kaynağı.

    `items` yerinde güncellenir. Dış budama (`label_trust.
    prune_low_confidence_boxes_in_dataset`) yetkilidir: diske yazdıysa
    fallback aynı kareleri ikinci kez işlemez (`external_handled`).
    Dönüş: pruned/promoted/preserved/negative_taught sayıları,
    `external_handled`, `stopped`, `errors`, `total`, `threshold`.
    """
    th = float(threshold)
    total = len(items)
    pruned_boxes = 0
    promoted_boxes = 0
    negative_taught_boxes = 0
    # S-3502: öğretilen negatif örnek sayısı ayrı ölçülür — budanan kutu
    # sayısı ile eşitlenmez; yalnızca _learn_box_exemplar True dönen
    # (geçerli crop ile hafızaya yazılan) kutular sayılır.
    errors: List[str] = []
    stopped = False
    # F-0610: dış budama diske yazdıysa bellek içi _review_items eski kalır;
    # bu bayrak başarı yolunda kuyruk yenilemeyi tetikler.
    # P0-7/K-13 (bölüm 3.2): "dış budama dokundu" yalnızca budanan
    # kutuya bakmaz — 0 budayıp yalnızca terfi eden kare de diskte
    # değişti; bayat bellekle fallback aynı kareyi ikinci kez işleyip
    # terfi sayısını çift sayar ve kalibrasyon vetosunu geri alır.
    external_handled = False

    # Varsa label_trust modülünün toplu budama fonksiyonunu dene
    try:
        from dataset_factory.label_trust import prune_low_confidence_boxes_in_dataset

        stems = [str(it.get("stem", "")) for it in items if isinstance(it, dict)]
        res = prune_low_confidence_boxes_in_dataset(
            dataset_dir=Path(dataset_dir),
            threshold=th,
            indices_or_stems=stems,
            record_negative_exemplars=True,
        )
        if isinstance(res, dict):
            pruned_boxes = int(res.get("pruned_boxes", res.get("pruned", 0)) or 0)
            promoted_boxes = int(res.get("promoted_boxes", 0) or 0)
            # S-3502: dış budama gerçek öğretim sayısını dönerse o
            # kullanılır; dönmezse (anahtar yok) 0 kalır — budanan sayı
            # asla "öğretildi" diye yazılmaz.
            negative_taught_boxes = int(res.get("negative_taught_boxes", 0) or 0)
            if (
                int(res.get("files_rewritten", 0) or 0) > 0
                or pruned_boxes > 0
                or promoted_boxes > 0
            ):
                # Disk değişti (ya da dış yol değiştiğini raporladı):
                # dış yol yetkilidir — fallback aynı kareleri ikinci
                # kez işlemez; kuyruk aşağıda diskten yenilenir.
                external_handled = True
    except Exception as exc:
        logger.debug("prune_low_confidence_boxes_in_dataset fallback: %s", exc)

    if not external_handled:
        # K-13/P0-7: fallback, dış budamayla AYNI kanonik koruma
        # predikatını (label_trust.is_protected_box) ve AYNI
        # kalibrasyonu (calibrate_box_confidence_and_trust,
        # trust_by_confidence=True) kullanır — eskiden fallback
        # yalnızca `hand_label`'a bakıp conf 0.0 TXT kutuyu siliyor,
        # veto edilmiş kutuyu vetosuz terfi ettiriyordu. Import
        # edilemezse (label_trust yüklenemedi — dış budama da o
        # yüzden atlandı) yerel ayna AYNI predikat mantığını uygular;
        # kalibrasyon yoksa eski terfi davranışı sürer ama budama
        # kararı yine kanonik predikatla verilir.
        try:
            from dataset_factory.label_trust import (
                calibrate_box_confidence_and_trust as _calibrate_prune_box,
            )
            from dataset_factory.label_trust import is_protected_box as _is_protected_prune_box
        except Exception:
            _calibrate_prune_box = None

            def _is_protected_prune_box(b: Dict[str, Any]) -> bool:  # type: ignore[no-redef]
                # Yerel ayna — kanonik tanım:
                # dataset_factory/label_trust.py::is_protected_box
                if not isinstance(b, dict):
                    return False
                if b.get("hand_label") is True:
                    return True
                if b.get("human_approved") is True:
                    return True
                if b.get("approved") is True:
                    return True
                _src = str(b.get("source", "")).lower()
                if any(t in _src for t in ("hand", "manual", "human")):
                    return True
                try:
                    _cid = int(b.get("class_id", -1))
                except (TypeError, ValueError):
                    _cid = -1
                _key = str(b.get("gather_key", b.get("key", ""))).strip().lower()
                return bool(
                    _cid == 13
                    and (
                        _key in ("f", "e")
                        or "f_gather" in _src
                        or "e_gather" in _src
                        or "gather_prompt" in _src
                        or "f_template" in _src
                        or "e_template" in _src
                        or "e_floating" in _src
                        or "floating" in _src
                    )
                )

        for i_step, it in enumerate(items, 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if not isinstance(it, dict):
                if progress_cb is not None:
                    progress_cb(i_step, total)
                continue
            weak = list(it.get("weak") or [])
            trusted = list(it.get("trusted") or [])
            if not weak and not trusted:
                if progress_cb is not None:
                    progress_cb(i_step, total)
                continue

            # P0-7/K-13: kare koruması — kare-düzeyi predikat VEYA
            # kutu-düzeyi kanonik predikata uyan en az bir weak kutu
            # varsa kare done="ok" almaz (dış budama done'a hiç
            # dokunmaz; fallback da dokunmamalı).
            frame_protected = dialog._is_hand_label_protected_item(it) or any(
                _is_protected_prune_box(b) for b in weak
            )

            kept_trusted = list(trusted)
            kept_weak: List[Dict[str, Any]] = []
            for b in weak:
                c = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                norm_c = c if c <= 1.0 else c / 100.0
                if norm_c < th and not _is_protected_prune_box(b):
                    pruned_boxes += 1
                    # S-1503: toplu onaydaki (F-0775) kapak kaldırıldı; budamada
                    # da `total <= 20` kapağı yok — budanan her kutu negatif
                    # örnek olarak öğretilir, sessiz atlama olmaz.
                    # S-3502: sayaç yalnızca öğrenme gerçekten
                    # gerçekleştiyse artar (görsel/crop yoksa False döner).
                    if dialog._learn_box_exemplar(it, b, positive=False, source="bulk_box_prune"):
                        negative_taught_boxes += 1
                elif _calibrate_prune_box is not None:
                    # Dış yolla aynı karar: kalibrasyon vetoları önce
                    # çalışır — veto edilen kutu weak kalır (terfi
                    # etmez), korumalı kutu trusted olur.
                    cal = _calibrate_prune_box(
                        dict(b), stem=str(it.get("stem") or ""),
                        emin_threshold=th, trust_by_confidence=True,
                    )
                    if cal.get("trusted") is True:
                        kept_trusted.append(cal)
                        promoted_boxes += 1
                    else:
                        kept_weak.append(cal)
                else:
                    nb = dict(b)
                    nb["trusted"] = True
                    nb["weak"] = False
                    kept_trusted.append(nb)
                    promoted_boxes += 1

            # Bu turda kareye dokunuldu mu? (içerik karşılaştırması:
            # yalnızca kalibrasyon normalleştirmesi bile yazımı
            # gerektirir; hiçbir şey değişmediyse kareye dokunulmaz.)
            frame_changed = kept_trusted != trusted or kept_weak != weak
            # El etiketi koruması: bu turda hiçbir kutusu budanmayan /
            # terfi etmeyen VE el etiketi korumalı kareye dokunulmaz —
            # persist yok (labels TXT bayt-bayt durur), `done="ok"` yok.
            # TXT kökenli el etiketi karesi (weak boş) hep bu daldan
            # çıkar. Korumasız trusted-only karede eski davranış aynen
            # sürer (W46 testi: 'above' filtresinde Emin kare done="ok").
            if not frame_changed and dialog._is_hand_label_protected_item(it):
                if progress_cb is not None:
                    progress_cb(i_step, total)
                continue

            it["weak"] = kept_weak
            it["trusted"] = kept_trusted
            it["n_weak"] = len(kept_weak)
            # F-2055 [O3]: kutuları tamamen silinen kare onaylanmaz —
            # sıfır kutulu kare "ok" değildir; yeniden etiketleme /
            # boş-onay için beklemede kalır. Yalnızca en az bir Emin
            # kutusu kalan VE korumalı olmayan kare done="ok" olur
            # (P0-7/K-13: korumalı kare done almaz).
            if kept_trusted and not frame_protected:
                it["done"] = "ok"
            it["conf"] = dialog._frame_conf(it)
            if dialog._persist_review_item(it):
                pass
            else:
                it["done"] = None  # S-1502: yazılamadıysa kare kuyrukta kalır
                msg = f"[HATA] #{it.get('stem')}: budama kaydı diske yazılamadı."
                errors.append(msg)
                if log_fn is not None:
                    log_fn(msg)

            if progress_cb is not None:
                progress_cb(i_step, total)

    # Korunan görsel sayısı bellekten değil diskten ölçülür (F-0610).
    preserved_images = 0
    for _it in items:
        if not isinstance(_it, dict):
            continue
        _img = dialog._resolve_review_image(
            str(_it.get("split", "train")), str(_it.get("stem", ""))
        )
        if _img is not None and _img.is_file():
            preserved_images += 1

    return {
        "pruned_boxes": pruned_boxes,
        "promoted_boxes": promoted_boxes,
        "preserved_images": preserved_images,
        "negative_taught_boxes": negative_taught_boxes,
        "external_handled": external_handled,
        "errors": errors,
        "stopped": stopped,
        "total": total,
        "threshold": th,
    }


class _BulkPruneWorker(QObject):
    """Worker thread: kutu budama (dış + fallback) disk yazımı.

    Girdi `items` (hedef karelerin derin kopyaları); sonuç `finished`
    ile UI thread'e döner — worker QWidget'e dokunmaz. `stop()`
    işbirlikçi durdurmadır: yalnızca fallback döngüsü kare sınırında
    durur; dış budama tek senkron çağrıdır, kare ortasında bölünemez
    (zehir temizliği worker'ındaki sınırlamayla aynı).
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dialog: Any,
        items: List[Dict[str, Any]],
        dataset_dir: Path,
        threshold: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dialog = dialog
        self._items = items
        self._dataset_dir = Path(dataset_dir)
        self._threshold = float(threshold)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = _process_prune_items(
                self._dialog,
                self._items,
                self._threshold,
                dataset_dir=self._dataset_dir,
                progress_cb=lambda d, t: self.progress.emit(int(d), int(t)),
                log_fn=None,
                stop_event=self._stop_event,
            )
            result["items"] = self._items
            result["stopped"] = bool(result.get("stopped"))
            self.finished.emit(dict(result))
        except Exception as exc:
            logger.warning("[BUDAMA] worker hatası: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class _TurboRelabelWorker(QObject):
    """Worker thread: turbo yeniden etiketleme (`relabel_dataset_batch`).

    P2-20 (F022-K2, F068-K2): `DatasetCurator.relabel_dataset_batch`
    tüm filtre setinin YOLO inference'ını yapar — UI thread'de
    çalışırsa pencere dakikalarca donar; ilerleme callback'indeki
    `processEvents()` ise reentrancy penceresi açar. Bu worker
    küratörü kendi thread'inde kurar, ilerlemeyi `progress` signal'i
    ile UI thread'e döner (QWidget'e dokunmaz, `processEvents` yok).
    `stop()` işbirlikçi değildir — `relabel_dataset_batch`'in
    ilerleme callback'indeki hata yutulur, yarıda kesilemez; kapanışta
    thread quit+wait ile beklenir.
    """

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dataset_dir: Path,
        image_paths: List[Path],
        # P1-17 / K-06: varsayılan incremental=True — değişmemiş kare
        # atlanır (P2-20'nin False varsayılanı v16'dan mekanik taşınmıştı).
        incremental: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_dir = Path(dataset_dir)
        self._image_paths = [Path(p) for p in image_paths]
        self._incremental = bool(incremental)

    @pyqtSlot()
    def run(self) -> None:
        try:
            from dataset_factory.dataset_curator import DatasetCurator

            curator = DatasetCurator(dataset_dir=self._dataset_dir)
            res = curator.relabel_dataset_batch(
                image_paths=self._image_paths,
                incremental=self._incremental,
                progress_callback=lambda info: self.progress.emit(dict(info)),
            )
            self.finished.emit(dict(res or {}))
        except Exception as exc:
            logger.warning("[TURBO] worker hatası: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class _PoisonPurgeWorker(QObject):
    """Worker thread: `clean_poison_labels_ai` (4 katmanlı YZ temizlik).

    İlerleme `progress_callback(done, total)` üzerinden signal'e döner
    (label_trust kümülatif raporlar: train+val toplamı). `stop()`
    yalnızca çağrı öncesi/sonrası etkilidir — temizleyici tek senkron
    çağrıdır, kare ortasında bölünemez; kapanışta thread quit+wait
    ile beklenir (aynı güvenlik ağı).
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dataset_dir: Path,
        floor: float,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_dir = Path(dataset_dir)
        self._floor = float(floor)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            from dataset_factory.label_trust import clean_poison_labels_ai

            if self._stop_event.is_set():
                self.finished.emit({"stopped": True})
                return
            stats = clean_poison_labels_ai(
                self._dataset_dir,
                floor=self._floor,
                progress_callback=lambda d, t: self.progress.emit(int(d), int(t)),
            )
            stats = dict(stats or {})
            stats["stopped"] = False
            self.finished.emit(stats)
        except Exception as exc:
            logger.warning("[TEMİZ] worker hatası: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class _ThumbPixmapLruCache:
    """Sınırlı boyutlu gerçek LRU küçük-resim önbelleği (P2-20, F050-K2).

    Eskiden düz dict + `len(cache) > 300` ise `cache.clear()` idi: 301.
    girdide 300 pixmap birden atılıyor, kuyruk kaydırıldıkça aynı kareler
    sürekli yeniden decode ediliyordu. Artık en az kullanılan girdi tek
    tek atılır (`OrderedDict`, `get`/`__setitem__` erişimi tazeler).
    Yalnızca UI thread'den kullanılır (QPixmap GUI thread'e aittir).
    """

    def __init__(self, maxsize: int = 300, initial: Optional[Dict[Any, Any]] = None) -> None:
        self._maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[Tuple[str, str], QPixmap]" = OrderedDict()
        if initial:
            for key, value in dict(initial).items():
                self[key] = value

    def get(self, key: Tuple[str, str], default: Any = None) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            return default
        self._data.move_to_end(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __setitem__(self, key: Tuple[str, str], value: QPixmap) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


def _coerce_thumb_cache(cache: Any) -> _ThumbPixmapLruCache:
    """Düz dict önbelleği (eski/harness yapımı) gerçek LRU'ya çevir."""
    if isinstance(cache, _ThumbPixmapLruCache):
        return cache
    return _ThumbPixmapLruCache(initial=cache if isinstance(cache, dict) else None)


def _decode_thumb_qimage(dataset_dir: Path, split: str, stem: str) -> Optional[QImage]:
    """Tek karenin 72x54 küçük resmini worker thread'de üret.

    P2-20 (F050-K1): `QImageReader` ölçekli okur — tam çözünürlük JPEG
    decode + sonra `cv2.resize` kalktı; 1180 kare = 1180 tam decode
    sorunu biter (JPEG decoder hedef boyutta kısmi decode yapar).
    QImage worker thread'de güvenle kurulur (QPixmap kurulmaz); numpy
    tamponundan .copy() ile ayrılır, signal ile UI thread'e gider.
    Reader ölçekli okuyamazsa (desteklenmeyen format/plug-in yokluğu)
    eski tam-decode yolu yedek olarak çalışır.
    """
    path = _resolve_review_image_path(dataset_dir, split, stem)
    if path is None:
        return None
    try:
        reader = QImageReader(str(path))
        orig = reader.size()
        if orig is not None and not orig.isEmpty():
            w, h = int(orig.width()), int(orig.height())
            scale = min(72.0 / max(1, w), 54.0 / max(1, h))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            reader.setScaledSize(QSize(nw, nh))
            img = reader.read()
            if img is not None and not img.isNull():
                if img.width() != nw or img.height() != nh:
                    img = img.scaled(
                        nw,
                        nh,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                return img
    except Exception as exc:
        logger.debug("QImageReader ölçekli okuma başarısız %s: %s", path, exc)
    # Yedek: önceki tam-decode + resize yolu (yalnızca reader yolu
    # sonuç vermediyse; davranış birebir korunur).
    img_bgr = _safe_imread_unicode(path)
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return None
    h, w = img_bgr.shape[:2]
    scale = min(72.0 / max(1, w), 54.0 / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()


class _QueueThumbThread(QThread):
    """Görünür kuyruk satırlarının küçük resimlerini arka planda üretir."""

    thumb_ready = pyqtSignal(str, str, QImage)

    def __init__(self, dataset_dir: Path, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._dataset_dir = Path(dataset_dir)
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._queued: set = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # [YOLO-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(self)

    def request(self, split: str, stem: str) -> None:
        key = (str(split), str(stem))
        with self._lock:
            if key in self._queued or self._stop.is_set():
                return
            self._queued.add(key)
        self._queue.put(key)

    def reset(self) -> None:
        with self._lock:
            self._queued.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                key = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._queued.discard(key)
            if self._stop.is_set():
                continue
            try:
                qimg = _decode_thumb_qimage(self._dataset_dir, key[0], key[1])
            except Exception as exc:
                logger.debug("küçük resim üretilemedi %s: %s", key, exc)
                qimg = None
            if qimg is not None and not qimg.isNull():
                self.thumb_ready.emit(key[0], key[1], qimg)


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
        # Yerleşik fabrika tembel kurulur: her dialog açılışında dataset
        # klasörleri + data.yaml yan etkisi olmasın (F-1966).
        self._builtin_helper: Optional[_BuiltinAutoLabelingDatasetFactory] = None
        self._training_worker: Optional[RTXTrainingWorker] = None
        # P2-22 (bölüm 6.2, F040-K1): senkron yeniden-giriş bayrağı —
        # yalnızca asenkron `isRunning()` kontrolü çift tıkta ikinci eğitimi
        # engellemeye yetmez (ikinci tık `self._training_worker` üzerine
        # yazabiliyordu). Bayrak tık anında senkron kurulur; kapı RED'inde
        # tıklama işleyicisi, gerçek başlangıçta bitiş sinyalleri +
        # 1 sn kilit zamanlayıcısı temizler.
        self._training_starting: bool = False
        self._bootstrap_worker: Optional[_BackgroundCallWorker] = None
        # [YOLO-ASYNC] Arka plan yükleme / küçük resim durumu.
        self._yolo_load_thread: Optional[QThread] = None
        self._yolo_load_worker: Optional[Any] = None
        self._yolo_load_running: bool = False
        self._yolo_load_preserve_stem: Optional[str] = None
        self._yolo_load_started_at: float = 0.0
        self._yolo_closing: bool = False
        self._thumb_thread: Optional[Any] = None
        self._thumb_labels: Dict[str, QLabel] = {}
        self._yolo_loading_overlay: Optional[QFrame] = None
        self._yolo_loading_detail: Optional[QLabel] = None
        self._yolo_loading_title: Optional[QLabel] = None
        self._yolo_loading_bar: Optional[QProgressBar] = None
        # [YOLO-BULK-ASYNC] Toplu onay / zehir temizliği arka plan durumu.
        self._bulk_approve_thread: Optional[QThread] = None
        self._bulk_approve_worker: Optional[Any] = None
        self._bulk_approve_running: bool = False
        self._bulk_approve_started_at: float = 0.0
        self._bulk_approve_total: int = 0
        # P1-09 (K-24): dondurma çalışırken toplu onay başlayamaz, çift
        # tıklama ikinci dondurma çalıştıramaz — `_freeze_dataset`
        # sarmalayıcısı bu bayrağı + btn_freeze kilidini yönetir.
        self._freeze_running: bool = False
        self._purge_thread: Optional[QThread] = None
        self._purge_worker: Optional[Any] = None
        self._purge_running: bool = False
        self._purge_started_at: float = 0.0
        # [P2-19/F044-02] Bot-pause kancaları: dialog açıkken botun R
        # basışı dialoga gitmesin diye bot duraklatılır. Kancalar main.py
        # (_connect_yolo_factory_signals) tarafından bağlanır; bağlanmadıysa
        # None kalır ve davranış değişmez. `_bot_paused_by_factory` yalnızca
        # BU dialogun duraklattığı botu geri devam ettirir (kullanıcının
        # önceden duraklattığı bota dokunulmaz).
        self._bot_pause_fn: Optional[Any] = None
        self._bot_resume_fn: Optional[Any] = None
        self._bot_is_paused_fn: Optional[Any] = None
        self._bot_paused_by_factory: bool = False
        # v20-onnx-reload: one cached YOLOAutoLabeler (+ optional shared
        # ONNX detector). Relabel used to build a new labeler/scanner/
        # CUDA session on every click (~300-500MB + UI freeze).
        self._yolo_labeler: Optional[Any] = None
        self._shared_yolo_detector: Optional[Any] = None
        self._relabel_dedup_cache: Dict[str, Dict[str, Any]] = {}
        self._relabel_dedup_ttl_sec: float = 10.0

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
            # B1: stale bayrağı dialog state dosyasında da yaşar — varsayılan
            # tanımlar burada, round-trip _load/_save_pipeline_disk_state'te.
            "model_stale": False,
            "stale_reason": "",
            "stale_at": "",
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
        self._queue_filter_mode: str = "all"  # all | pending | ok | no | high | low (+ eski modlar)
        self._queue_min_conf: float = 0.0  # Kuyruk güven filtresi (qConf kaydırıcısı, 0.00..0.95)
        self._queue_thumb_cache: _ThumbPixmapLruCache = _ThumbPixmapLruCache()
        self._journal_lines: List[str] = []  # Canlı günlük şeridi (altta) için ham log satırları
        self._journal_filter: str = "all"  # all | info | preview
        self._legend_buttons: Dict[int, QPushButton] = {}
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
        # [YOLO-ASYNC] İlk açılış: ağır dataset/kuyruk taraması worker
        # thread'de; sayfa anında "Yükleniyor..." durumuyla açılır, UI
        # thread'i hiç bloklanmaz. _go_step(2) aynı yüklemeyi tekrar
        # istemez: _request_queue_reload çalışan yüklemeyi görünce düşer.
        self._show_yolo_loading("YOLO yükleniyor: dataset taranıyor…")
        self._start_yolo_data_load(initial=True)
        self._go_step(int(self._pipeline["step"]))
        for warning in list(self._startup_warnings):
            self._append_log(warning)
        self._startup_warnings.clear()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(4000)
        self._refresh_timer.timeout.connect(lambda: self.refresh_dashboard(force=False))
        self._refresh_timer.start()

        # P2-22 (bölüm 6.2, F048-O1): eğitim kilidi yalnızca worker
        # sinyallerine bağlı kalmasın — sinyal kaçarsa butonlar kilitli
        # kalıyordu. 1 sn'de bir yer-doğruluk (`_training_worker.isRunning()`)
        # üzerinden kilit tazelenir (`_refresh_training_locks`); özet satırı
        # disk okuduğu için bu yolda tazelenmez.
        self._train_lock_timer = QTimer(self)
        self._train_lock_timer.setInterval(1000)
        self._train_lock_timer.timeout.connect(
            lambda: self._refresh_training_locks(refresh_summary=False)
        )
        self._train_lock_timer.start()

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
            logger.warning("Canlı AutoLabelingDatasetFactory kurulamadı, yerleşik yedeğe düşüldü (%s)", exc)
            return _BuiltinAutoLabelingDatasetFactory()

    def _get_builtin_helper(self) -> _BuiltinAutoLabelingDatasetFactory:
        if self._builtin_helper is None:
            self._builtin_helper = _BuiltinAutoLabelingDatasetFactory()
        return self._builtin_helper

    def _dataset_dir(self) -> Path:
        ctrl = getattr(self, "factory_controller", None)
        ctrl_ds = getattr(ctrl, "dataset_dir", None) if ctrl is not None else None
        if ctrl_ds is not None:
            return Path(ctrl_ds)
        return Path(self._get_builtin_helper().dataset_dir)

    def _models_dir(self) -> Path:
        try:
            ds = self._dataset_dir()
            # Test tespiti yol adındaki "tmp"/"pytest" alt dizesine bakmaz (F-1962):
            # gerçek kullanıcı yolu "tmp" içerebilir.
            under_test = bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules
            if under_test:
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
                # B-02 (dilim 27): manifest parse ediliyor diye `frozen=True`
                # yetmez — diskte 0 kare varken manifest 456 kare iddia
                # edebiliyor, UI "donduruldu" gösterip eğitimi açıyordu.
                # Dondurma, manifestin anlattığı dizinde gerçekten kare
                # varsa geçerlidir.
                claimed = int(payload.get("total_frames", 0) or 0)
                if claimed > 0:
                    if str(payload.get("source", "")) == "approved_pool":
                        img_roots = (
                            self._dataset_dir() / "approved" / "images",
                        )
                    else:
                        img_roots = (self._dataset_dir() / "images",)
                    disk_images = 0
                    for _root in img_roots:
                        for _sub in ("train", "val"):
                            _d = _root / _sub
                            if _d.is_dir():
                                disk_images += sum(
                                    1
                                    for _f in _d.iterdir()
                                    if _f.is_file()
                                    and _f.suffix.lower()
                                    in (".jpg", ".png", ".jpeg", ".webp", ".bmp")
                                )
                    if disk_images == 0:
                        self._pipeline["frozen"] = False
                        self._startup_warnings.append(
                            "[UYARI] frozen_manifest.json "
                            f"{claimed} kare iddia ediyor ama diskte kare yok "
                            "— dondurma geçersiz sayıldı, Adım 3'te yeniden dondurun."
                        )
                        logger.warning(
                            "frozen_manifest (%d kare) diskle uyuşmuyor (0 kare) — frozen=False",
                            claimed,
                        )
                    else:
                        self._pipeline["frozen"] = True
                else:
                    self._pipeline["frozen"] = True
                self._pipeline["run_id"] = int(payload.get("run_id", 1) or 1)
            except Exception as exc:
                logger.warning("Bozuk frozen_manifest yoksayıldı: %s", exc)
                self._startup_warnings.append("[UYARI] frozen_manifest.json bozuk — dondurma durumu sıfırlandı.")
                self._pipeline["frozen"] = False
        state_path = self._pipeline_state_path()
        if state_path.is_file():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                for key in ("trained", "validated", "published", "scanned", "imbalance_fixed", "step", "train_completed_at", "model_stale", "stale_reason", "stale_at"):
                    if key in saved:
                        self._pipeline[key] = saved[key]
                cached = saved.get("last_train_metrics")
                if isinstance(cached, dict) and cached:
                    self._last_train_result = dict(cached)
            except Exception as exc:
                logger.warning("Bozuk pipeline state yoksayıldı: %s", exc)
                self._startup_warnings.append("[UYARI] pipeline state dosyası bozuk — eğitim durumu sıfırlandı.")
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
                # B1: stale bayrağı daha önce payload'da yoktu — _mark_model_stale
                # sonunda _save_pipeline_disk_state çağırıp kendi bayrağını
                # siliyordu. Artık round-trip korunur.
                "model_stale": bool(self._pipeline.get("model_stale", False)),
                "stale_reason": str(self._pipeline.get("stale_reason", "") or ""),
                "stale_at": str(self._pipeline.get("stale_at", "") or ""),
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
        """Newest mtime of the run's WEIGHT files (best.pt / best.onnx).

        P1-11 (K-26 / D020-K1): `results.csv` tek başına tazelik kanıtı
        değildir — ağırlık dosyası olmayan bir run (yalnızca taze
        `results.csv`) `fresh=True` üretip `trained=True` diriltiyordu.
        Bu yüzden yalnızca ağırlık dosyalarının mtime'ı sayılır.
        `models/` altındaki canlı ONNX bilerek dışarıda bırakılır: o,
        önceki döngünün yayın çıktısıdır, mevcut run'ın kanıtı değil.
        """
        path = run_dir or self._find_train_artifact_run_dir()
        if path is None:
            return 0.0
        mtime = 0.0
        for cand in (
            path / "weights" / "best.pt",
            path / "weights" / "best.onnx",
            path / "best.onnx",
        ):
            try:
                if cand.is_file():
                    mtime = max(mtime, float(cand.stat().st_mtime))
            except OSError:
                continue
        return mtime

    def _artifacts_are_fresh_vs_freeze(self) -> bool:
        """True when the run's weight files are newer than the current dataset freeze.

        P1-11 (K-26 / D020-K1): tazelik artık ağırlık dosyasına (best.pt /
        best.onnx) dayanır; taze `results.csv` tek başına `fresh=True`
        vermez. ONNX tazeliği ZORUNLU değildir (B5: yayın öncesi ONNX hep
        eski döngüden kalır) — taze best.pt tek başına yeter, ama taze
        results.csv + eski ağırlık yetmez.
        """
        freeze_mtime = self._freeze_mtime()
        art_mtime = self._artifact_mtime()
        if art_mtime <= 0:
            return False
        if freeze_mtime <= 0:
            return True
        return art_mtime >= (freeze_mtime - 1.0)

    @staticmethod
    def _read_run_requested_epochs(run_dir: Optional[Path]) -> int:
        """Requested epoch count from the run's Ultralytics `args.yaml`.

        Artifact dosyaları tamamlanan epoch sayısını (`results.csv` satır
        sayısı) verir ama istenen sayıyı vermez; yarım-run kapısı için
        istenen sayı run dizinindeki `args.yaml`'dan okunur (Ultralytics
        her eğitimde yazar). Dosya yoksa/okunamazsa 0 döner — kapı o
        zaman atlanır (eski davranış).
        """
        if run_dir is None:
            return 0
        try:
            text = (Path(run_dir) / "args.yaml").read_text(encoding="utf-8")
        except Exception:
            return 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("epochs:"):
                raw = stripped.split(":", 1)[1].strip().strip("\"").strip("'")
                try:
                    return int(raw)
                except ValueError:
                    try:
                        return int(float(raw))
                    except ValueError:
                        return 0
        return 0

    def _same_run_early_stopped(self, run_dir: Optional[Path]) -> bool:
        """True when the accepted in-memory result belongs to this run dir
        and the engine flagged it early-stopped (patience).

        Erken durdurma kanıtı artifact dosyalarında yoktur — yalnızca
        kabul edilmiş sonucun kendisinde yaşar. Bu yüzden muafiyet
        yalnızca bellek sonucu BU run dizinine aitse geçerlidir; başka
        (ör. reddedilen) bir run'ın muafiyeti kapıyı açamaz.
        """
        if run_dir is None:
            return False
        try:
            accepted = dict(self._last_train_result or {})
            if not accepted.get("early_stopped"):
                return False
            arts = accepted.get("artifacts") if isinstance(accepted.get("artifacts"), dict) else {}
            accepted_run = str(arts.get("run_dir") or "")
            if not accepted_run:
                return False
            target = Path(run_dir).resolve()
            # Motor run_dir'i göreli de yazabilir
            # ("models/rtx_runs/gatherion_rtx_fp16") — CWD'ye ve repo
            # köküne karşı da çözümlenir.
            candidates = {Path(accepted_run).resolve()}
            try:
                repo_root = Path(__file__).resolve().parent.parent
                candidates.add((repo_root / accepted_run).resolve())
            except Exception:
                pass
            return target in candidates
        except Exception:
            return False

    def _disk_model_stale(self) -> Tuple[bool, str]:
        """Disk stale bayrağı (models/stale_manifest.json + pipeline state).

        P1-11 (K-26): `_reconcile` bayrağı hiç okumuyordu — diskte bayat
        işaretli model yeniden açılışta "TAMAMLANDI" diriliyordu.
        Dizinler dialogun kendi `_dataset_dir`/`_models_dir`'inden
        çözülemezse (örn. test stub'ı) sinyal yok sayılır — CWD varsayılan
        yollarına düşülmez, yoksa repo durumu teste sızar. Bozuk
        manifestin kendisi `is_model_stale` içinde fail-closed okunur
        (stale=True).
        """
        try:
            ds_fn = getattr(self, "_dataset_dir", None)
            md_fn = getattr(self, "_models_dir", None)
            if not callable(ds_fn) or not callable(md_fn):
                return False, ""
            ds_dir = ds_fn()
            md_dir = md_fn()
        except Exception:
            return False, ""
        try:
            from dataset_factory.dataset_validator import is_model_stale

            return is_model_stale(dataset_dir=ds_dir, models_dir=md_dir)
        except Exception:
            return False, ""

    def _mark_model_stale(self, reason: str = "Etiketler güncellendi") -> None:
        """Invalidates model status and marks model as stale whenever dataset labels are modified."""
        if not hasattr(self, "_pipeline") or not isinstance(self._pipeline, dict):
            return
        self._pipeline["model_stale"] = True
        self._pipeline["stale_reason"] = reason
        self._pipeline["stale_at"] = datetime.now().isoformat(timespec="seconds")
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
        # P1-11 (K-26): `model_stale` bayrağı burada hiç okunmuyordu —
        # diskte bayat işaretli model yeniden açılışta "TAMAMLANDI"
        # diriliyordu. Bayatlık, oturum bayrağından da önce gelir: taze
        # oturumda bile stale yazılmışsa eğitim geçersiz sayılır.
        disk_stale, disk_reason = self._disk_model_stale()
        if disk_stale or bool(self._pipeline.get("model_stale")):
            self._pipeline["trained"] = False
            self._pipeline["validated"] = False
            self._pipeline["published"] = False
            self._pipeline["train_session_ok"] = False
            self._pipeline["train_completed_at"] = ""
            warning = (
                "[UYARI] Model bayat (stale) işaretli — ESKİ eğitim trained "
                f"sayılmaz ({disk_reason or self._pipeline.get('stale_reason') or 'sebep belirtilmemiş'}). "
                "Adım 4'te gerçek eğitim şart."
            )
            pending = getattr(self, "_startup_warnings", None)
            if isinstance(pending, list):
                pending.append(warning)
            else:
                self._startup_warnings = [warning]
            return
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
            # B-03: taze ağırlık tek başına trained=True yapmıyordu artık —
            # eskiden diskte taze best.pt varsa kabul kapısı hiç
            # sorulmadan bayrak diriltiliyordu. Artık mAP kanıtı şart:
            # eşik altı (örn. yarım eğitimin 0.32'si) veya metriksiz
            # ağırlık trained sayılmaz.
            metrics = self._load_train_metrics_from_artifacts()
            map50 = float(metrics.get("mAP50", 0.0) or 0.0)
            # OpenCV prototip (Mode B) metriği asla eğitim kanıtı sayılmaz —
            # yalnızca Ultralytics artifact'ı trained diriltir.
            if metrics.get("mode") not in ("ultralytics_artifact",):
                self._pipeline["trained"] = False
                self._pipeline["validated"] = False
                self._pipeline["published"] = False
                return
            if map50 < MIN_PUBLISH_MAP50:
                self._pipeline["trained"] = False
                self._pipeline["validated"] = False
                self._pipeline["published"] = False
                warning = (
                    "[UYARI] Taze ağırlık var ama mAP50 kanıtı yok/eşik altı "
                    f"(%{map50 * 100:.1f} < %{MIN_PUBLISH_MAP50 * 100:.0f}) — "
                    "trained bayrağı kurulmadı. Adım 4'te gerçek eğitim şart."
                )
                pending = getattr(self, "_startup_warnings", None)
                if isinstance(pending, list):
                    pending.append(warning)
                else:
                    self._startup_warnings = [warning]
                return
            # P1-11 (K-26): yarım-run kapısı artık artifact'a dayanır —
            # eskiden yalnızca bellek sonucu (`_last_train_result`)
            # okunuyordu; reddedilen run belleği temizlediği için kapı
            # boşa düşüyor, mAP'i yüksek yarım run trained diriltiyordu.
            # Tamamlanan epoch artifact'tan (`results.csv` satır sayısı),
            # istenen epoch run'ın `args.yaml`'ından okunur; erken
            # durdurma muafiyeti yalnızca AYNI run'a ait kabul edilmiş
            # sonuçta geçerlidir.
            cached = dict(self._last_train_result or {})
            ep_done = int(metrics.get("epochs_completed", 0) or 0)
            ep_req = self._read_run_requested_epochs(run_dir)
            if ep_req <= 0:
                ep_req = int(cached.get("epochs_requested", cached.get("epochs", 0)) or 0)
            if (
                ep_req > 0
                and ep_done < ep_req
                and not self._same_run_early_stopped(run_dir)
            ):
                self._pipeline["trained"] = False
                self._pipeline["validated"] = False
                self._pipeline["published"] = False
                return
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
            "[UYARI] Eğitim ağırlıkları (best.pt / best.onnx) dondurmadan ESKİ — "
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
            "epochs_completed",
            "epochs_requested",
            "early_stopped",
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
                # B-04: artifact kurtarma yolunda da epoch kanıtı taşınır —
                # eskiden yalnızca mAP50>0 bakılıyordu, 33/100 run "completed"
                # sayılıyordu. Satır sayısı = tamamlanan epoch sayısı.
                epochs_completed = 0
                _csv_path = run_dir / "results.csv"
                if _csv_path.is_file():
                    try:
                        import csv as _csv

                        with _csv_path.open("r", encoding="utf-8", newline="") as _h:
                            epochs_completed = sum(1 for _ in _csv.DictReader(_h))
                    except Exception:
                        epochs_completed = 0
                # Prefer Ultralytics run metrics over Mode B prototype JSON.
                # Metriği olmayan run "completed" sayılmaz (F-1951).
                return {
                    "status": "completed" if map50 > 0 else "incomplete",
                    "mode": "ultralytics_artifact",
                    "mAP50": map50,
                    "mAP50_95": float(ultra.get("mAP50_95", 0.0) or 0.0),
                    "precision": float(ultra.get("precision", 0.0) or 0.0),
                    "recall": float(ultra.get("recall", 0.0) or 0.0),
                    "box_loss": float(ultra.get("box_loss", 0.0) or 0.0),
                    "cls_loss": float(ultra.get("cls_loss", 0.0) or 0.0),
                    "epochs_completed": int(epochs_completed),
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
                padding: 10px 16px;
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
                padding: 10px 16px;
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
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton#PipeDanger {{
                background: transparent;
                color: #FF8A94;
                border: 1px solid #4A2028;
                border-radius: 10px;
                padding: 10px 16px;
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
        root.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(16)
        head_left = QVBoxLayout()
        head_left.setSpacing(8)
        title = QLabel("YOLO fabrikası  —  iş akışı")
        title.setObjectName("PipeMainTitle")
        title.setStyleSheet(
            "font-size:28px;font-weight:800;letter-spacing:-0.02em;color:#F2F3F5;"
            "font-family:'Plus Jakarta Sans','Segoe UI',sans-serif;"
        )
        sub = QLabel(
            "Ekran artık vitrin değil tezgâh: her adım bir sonrakini besliyor. "
            "Nerede takıldığını, sıradaki işin ne olduğunu ve eğitimi neyin engellediğini tek yerden görürsün."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            "color:#B7BFCB;font-size:14px;max-width:620px;line-height:1.55;"
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
        body.setSpacing(16)
        left_col = QVBoxLayout()
        self.step_stack = QStackedWidget(self)
        self.step_stack.addWidget(self._build_step_capture())
        # W3 (dikey tasma): Onayla adimi dikey kaydirmali alana alinir.
        # 1560x960'da govde ~384 px; canvas (min 620) + lejant + aksiyon
        # cubugu ~800 px sigmadigi icin lejant ve aksiyon cubugu
        # erisilemez kaliyordu. Kaydirma alani her boyutta alt satira
        # inmeyi saglar; yatay kaydirma yalnizca pencere cok
        # daraltilirsa devreye girer.
        review_page = self._build_step_review()
        review_scroll = QScrollArea(self)
        review_scroll.setObjectName("PipeReviewScroll")
        review_scroll.setWidgetResizable(True)
        review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        review_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        review_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        review_scroll.setStyleSheet(
            "QScrollArea#PipeReviewScroll { background: transparent; border: none; }"
        )
        review_scroll.setWidget(review_page)
        self._review_scroll = review_scroll
        self.step_stack.addWidget(review_scroll)
        self.step_stack.addWidget(self._build_step_prepare())
        self.step_stack.addWidget(self._build_step_train())
        self.step_stack.addWidget(self._build_step_validate())
        self.step_stack.addWidget(self._build_step_publish())
        left_col.addWidget(self.step_stack)
        body.addLayout(left_col, stretch=1)
        body.addWidget(self._build_right_rail(), stretch=0)
        root.addLayout(body, stretch=1)

        # Alttaki filtreli Canlı Günlük şeridi (prototip): Tümü / INFO /
        # Önizleme + Temizle. Sağdaki Canlı log ile aynı akışı gösterir.
        self.live_journal_strip = self._build_live_journal_strip()
        root.addWidget(self.live_journal_strip)

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

        # [YOLO-ASYNC] Yükleme katmanı + görünür satır küçük resim tetikleyicisi.
        self._build_yolo_loading_overlay()
        self._wire_queue_thumb_requests()

    def _build_flow_section(self) -> QFrame:
        C = self.PIPELINE_COLORS
        frame = QFrame(self)
        frame.setObjectName("PipeFlow")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)

        head = QHBoxLayout()
        title = QLabel("Üretim hattı — dataset v4")
        title.setStyleSheet("font-size:14px;font-weight:800;")
        self.flow_summary = QLabel("Adım 1/6 · Yakala")
        self.flow_summary.setStyleSheet(f"color:{C['muted']};font-size:13px;")
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
            btn.setMinimumHeight(112)
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
                "border-radius:13px;font-family:Consolas;font-weight:700;font-size:13px;color:#B7BFCB;"
            )
            title_lbl = QLabel(title_txt)
            title_lbl.setStyleSheet("font-size:15px;font-weight:800;background:transparent;border:none;")
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"font-size:12px;color:{C['muted']};background:transparent;border:none;")
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
            "Boş sınıf", f"0 / {len(YOLO_12_CLASS_SPECS)}", "—", "PipeStat"
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
        # R05: sabit 420 px (v1 sözleşmesi 340 px'ti). 340 px'te kuyruk
        # kartlarındaki metin (stem + güven + sınıf özeti) sığmıyor ve
        # sınıf filtresi combo'sunun en uzun öğesi (~485 px) paneli yatay
        # kaydırmaya zorluyordu. 420 px'te kuyruk metni okunur; 2560
        # hedef ekranda merkez canvas hâlâ açık ara en geniş alan.
        # W3'ün gerçek taşma düzeltmeleri (dikey kaydırma, kısaltılmış
        # çubuk, esnek sağ ray) aynen duruyor.
        sidebar_scroll.setFixedWidth(420)
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

        step_hint = QLabel("Kısayol: A onayla · R reddet · S atla · K karakterimi etiketle")
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
        # Prototip (panel-2) durum filtresi: Tümü / Bekleyen / Onaylanan /
        # Reddedilen / Yüksek güvenli / Düşük güvenli. Eski modlar
        # (suspicious/confident/...) _filtered_review_indices'te programatik
        # olarak desteklenmeye devam eder, combo'da gösterilmez.
        self.queue_filter.addItem("Tümü (Tüm Kareler)", "all")
        self.queue_filter.addItem("Bekleyenler", "pending")
        self.queue_filter.addItem("Onaylananlar", "ok")
        self.queue_filter.addItem("Reddedilenler", "no")
        self.queue_filter.addItem("Yüksek güvenli (≥ kaydırıcı eşiği)", "high")
        self.queue_filter.addItem("Düşük güvenli (< kaydırıcı eşiği)", "low")
        self.queue_filter.currentIndexChanged.connect(self._on_queue_filter_changed)
        filter_l.addWidget(self.queue_filter)

        self.review_class_filter = QComboBox(self)
        self.review_class_filter.setObjectName("PipeClassFilter")
        # Sınıf filtresi en uzun öğeye göre ~485 px minimum genişlik
        # ister; bu, sol paneli yatay kaydırmaya zorlayıp kuyruk
        # kartlarını kırpıyordu. İçerik uzunluğuna göre ayarı kapatıp
        # panel genişliğine uyması sağlanır (seçili metin elide edilir,
        # tam metin açılır listede görünür).
        self.review_class_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.review_class_filter.setMinimumContentsLength(10)
        self.review_class_filter.addItem("Sınıf: Tümü", -1)
        for cid_opt, spec_opt in sorted(CLASS_ID_TO_SPEC.items()):
            cname_opt = spec_opt.get("key", f"class_{cid_opt}")
            lbl_tr = spec_opt.get("label_tr", cname_opt)
            self.review_class_filter.addItem(f"Sınıf {cid_opt:02d}: {cname_opt} ({lbl_tr})", cid_opt)
        self.review_class_filter.currentIndexChanged.connect(self._on_class_filter_changed)
        filter_l.addWidget(self.review_class_filter)

        # Kuyruk güven kaydırıcısı (prototip qConf): 0..95, adım 5 — kuyruğu
        # minimum güvene göre süzer. Toplu-onay eşiği (bulk_range) ayrıdır.
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Güven:"))
        self.queue_conf_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.queue_conf_slider.setObjectName("PipeQueueConf")
        self.queue_conf_slider.setRange(0, 95)
        self.queue_conf_slider.setSingleStep(5)
        self.queue_conf_slider.setPageStep(5)
        self.queue_conf_slider.setValue(0)
        self.queue_conf_val = QLabel("≥ 0,00")
        self.queue_conf_val.setStyleSheet(
            "font-family:Consolas; background:#0B0E12; border:1px solid #262D3A; border-radius:5px; padding:2px 6px;"
        )
        self.queue_conf_slider.valueChanged.connect(self._on_queue_conf_changed)
        conf_row.addWidget(self.queue_conf_slider)
        conf_row.addWidget(self.queue_conf_val)
        filter_l.addLayout(conf_row)

        self.btn_clear_filters = QPushButton("Filtreleri temizle", self)
        self.btn_clear_filters.setObjectName("PipeBtn")
        self.btn_clear_filters.setToolTip("Arama, durum, sınıf ve güven filtresini sıfırlar; kuyruğu hepsine döndürür.")
        self.btn_clear_filters.clicked.connect(self._on_clear_filters)
        filter_l.addWidget(self.btn_clear_filters)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Güven:"))
        self.bulk_range = QSlider(Qt.Orientation.Horizontal, self)
        self.bulk_range.setRange(0, 100)
        self.bulk_range.setValue(int(round(CONF_BULK_DEFAULT * 100)))
        self.bulk_val = QLabel(str(int(round(CONF_BULK_DEFAULT * 100))))
        self.bulk_val.setStyleSheet(
            "font-family:Consolas; background:#0B0E12; border:1px solid #262D3A; border-radius:5px; padding:2px 6px;"
        )
        # P1-12 (K-31): bu kaydırıcı TEK eşiktir — toplu onay, "Emin"
        # rozeti, yüksek/düşük filtreleri ve sayaçlar aynı değeri okur.
        self.bulk_range.setToolTip(
            "Toplu onay + Emin rozeti eşiği: bu değer ve üstü toplu onaylanır, "
            "rozet Emin sayılır; altı şüpheli kalır. Temizlik floor'u bu değer "
            "DEĞİLDİR (temizlik her zaman TRAIN_CONF_FLOOR 0.45 kullanır)."
        )
        self.bulk_range.valueChanged.connect(self._on_bulk_threshold_changed)
        self.bulk_range.sliderReleased.connect(self._on_bulk_threshold_released)
        slider_row.addWidget(self.bulk_range)
        slider_row.addWidget(self.bulk_val)
        filter_l.addLayout(slider_row)

        self.chk_only_approved = QCheckBox("Sadece Onaylananlarla Eğit", self)
        self.chk_only_approved.setChecked(True)
        self.chk_only_approved.setToolTip("Eğitim sadece onay havuzundaki (dataset/approved) doğrulanmış kareleri kullansın.")
        self.chk_only_approved.toggled.connect(self._on_only_approved_toggled)
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
        # Okunur kuyruk: özel satır widget'ları değişken yükseklikli olduğu için
        # uniform item sizes kapalı; liste 180px'e sıkıştırılmaz.
        self.queue_list.setMinimumHeight(240)
        self.queue_list.setUniformItemSizes(False)
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

        # Prototip "Filtredekileri Onayla (n)": o anki filtredeki karelerin
        # tamamını onaylar — _bulk_approve zaten filtrelenen indekste çalışır.
        self.btn_approve_filtered = QPushButton("✓ Filtredekileri Onayla (0)", self)
        self.btn_approve_filtered.setObjectName("PipeGreen")
        self.btn_approve_filtered.setToolTip("O anki arama/durum/sınıf/güven filtresine uyan karelerin tamamını onaylar.")
        self.btn_approve_filtered.clicked.connect(self._bulk_approve)
        bulk_l.addWidget(self.btn_approve_filtered)

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
            "Etiketsiz (0 kutu) ve şüpheli kareleri tespit edip otomatik olarak YZ ve görsel hafıza ile etiketler. "
            "El etiketli kareler hedefe girmez."
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
        right.addLayout(title_row)

        # Fotoğraf üstü şerit (prototip scene-head): Hız/Kalan göstergesi +
        # Tam Ekran (Lightbox). Canlı önizleme satırı TEK: canvas'ın kendi
        # paintEvent başlığı çizer (lbl_preview_live çift satırdı, kaldırıldı).
        # Bu ekranda Kinah tutarı gösterilmez.
        preview_head = QHBoxLayout()
        preview_head.setContentsMargins(4, 2, 4, 2)
        preview_head.setSpacing(10)
        preview_head.addStretch(1)

        self.lbl_speed_meter = QLabel("⚡ Hız: — · Kalan: — · 0 / 0", self)
        self.lbl_speed_meter.setObjectName("PipeSpeedMeter")
        self.lbl_speed_meter.setStyleSheet(
            "font-family:Consolas; font-size:11px; color:#38bdf8; background:#0B0E12; "
            "border:1px solid #1e293b; border-radius:6px; padding:4px 8px;"
        )
        preview_head.addWidget(self.lbl_speed_meter)

        self.btn_open_lightbox = QPushButton("⛶ Tam Ekran (Lightbox)", self)
        self.btn_open_lightbox.setObjectName("PipeBtn")
        self.btn_open_lightbox.setStyleSheet("font-size:11px; padding:4px 12px;")
        self.btn_open_lightbox.setToolTip("Görseli büyük ekranda incele (Çift tıklayarak da açabilirsiniz)")
        self.btn_open_lightbox.clicked.connect(self._open_review_lightbox)
        preview_head.addWidget(self.btn_open_lightbox)
        right.addLayout(preview_head)

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
        self.review_preview.boxAdded.connect(self._on_manual_box_added)
        self.review_preview.boxEdited.connect(self._on_box_edited)
        self._review_lightbox_pixmap: Optional[QPixmap] = None
        right.addWidget(self.review_preview, stretch=10)

        # Fotoğraf altı renk lejantı (prototip classLegend): tıklayınca kuyruğu
        # o sınıfa süzer; aynı sınıfa tekrar tıklayınca filtre sıfırlanır.
        self._build_class_legend()
        right.addWidget(self.class_legend_scroll)

        # Alt Büyük Aksiyon Çubuğu
        # W3 (yatay tasma): buton yatay padding'leri kisildi (metinler ayni);
        # cubuk minimumu ~930 px'ten ~838 px'e indi, 1560 px pencerede
        # butonlar artik minimum genisliklerinin altinda sikismiyor.
        actions = QHBoxLayout()
        actions.setContentsMargins(4, 6, 4, 4)
        actions.setSpacing(10)

        self.btn_approve = QPushButton("✓ ONAYLA VE SONRAKİ  (A)", self)
        self.btn_approve.setObjectName("PipeGreen")
        self.btn_approve.setStyleSheet("font-size:13px; font-weight:800; padding:10px 12px;")
        self.btn_approve.clicked.connect(self._on_step2_approve)

        self.btn_reject = QPushButton("✕ REDDET  (R)", self)
        self.btn_reject.setObjectName("PipeDanger")
        self.btn_reject.setStyleSheet("font-size:12px; font-weight:700; padding:10px 10px;")
        self.btn_reject.clicked.connect(self._on_step2_reject)

        self.btn_skip = QPushButton("Atla  (S)", self)
        self.btn_skip.setObjectName("PipeBtn")
        self.btn_skip.setStyleSheet("font-size:12px; padding:10px 10px;")
        self.btn_skip.clicked.connect(self._review_skip)

        self.btn_add_player_char = QPushButton("🎯 Karakterimi Etiketle  (K)", self)
        self.btn_add_player_char.setObjectName("PipeBtn")
        self.btn_add_player_char.setStyleSheet(
            "font-size:12px; font-weight:700; padding:10px 10px; border:1px solid #00d2ff; color:#00d2ff; background:#082f49;"
        )
        self.btn_add_player_char.setToolTip(
            "Karakter kutusunu ekranın ortasına otomatik yerleştirir. Tutamaçları çekerek tam boyuta getirebilirsiniz. Kısayol: 'K'"
        )
        self.btn_add_player_char.clicked.connect(self._add_player_character_box_to_current_frame)

        self.btn_back_capture = QPushButton("← Yakala'ya dön", self)
        self.btn_back_capture.setObjectName("PipeBtn")
        self.btn_back_capture.setStyleSheet("font-size:11px; padding:8px 8px;")
        self.btn_back_capture.clicked.connect(lambda: self._go_step(1))

        self.btn_goto_train = QPushButton("Eğit adımına git →", self)
        self.btn_goto_train.setObjectName("PipeBtn")
        self.btn_goto_train.setStyleSheet("font-size:11px; font-weight:700; padding:8px 8px;")
        self.btn_goto_train.setToolTip("Adım 4 — Eğit'e gider (veri seti dondurulmadan eğitim kilitlidir).")
        self.btn_goto_train.clicked.connect(lambda: self._go_step(4))

        actions.addWidget(self.btn_approve)
        actions.addWidget(self.btn_reject)
        actions.addWidget(self.btn_skip)
        actions.addWidget(self.btn_add_player_char)
        actions.addStretch()
        actions.addWidget(self.btn_back_capture)
        actions.addWidget(self.btn_goto_train)
        right.addLayout(actions)

        # Grid yerleşimi: Sol Scrollable Sidebar + Sağ Devasa Canvas
        grid = QHBoxLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(sidebar_scroll)
        grid.addLayout(right, stretch=1)
        lay.addLayout(grid, stretch=1)

        # Kısayollar
        # [P2-19/F044-03] Basılı tutulan A/R/S/K art arda kare
        # onaylamasın/silmesin: QShortcut otomatik tekrarı kapalıdır
        # (fiziksel tek basış = tek tetik). eventFilter/keyPressEvent
        # yolunda aynı koruma `event.isAutoRepeat()` ile yapılır
        # (`_review_shortcuts_allowed`).
        self._shortcut_approve_a = QShortcut(QKeySequence(Qt.Key.Key_A), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_approve_a.activated.connect(self._handle_shortcut_approve)
        self._shortcut_approve_a.activatedAmbiguously.connect(self._handle_shortcut_approve)
        self._shortcut_approve_lower = self._shortcut_approve_a
        self._shortcut_approve = self._shortcut_approve_a
        try:
            self._shortcut_approve_a.setAutoRepeat(False)
        except Exception:
            pass

        self._shortcut_reject_r = QShortcut(QKeySequence(Qt.Key.Key_R), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_reject_r.activated.connect(self._handle_shortcut_reject)
        self._shortcut_reject_r.activatedAmbiguously.connect(self._handle_shortcut_reject)
        self._shortcut_reject_lower = self._shortcut_reject_r
        self._shortcut_reject = self._shortcut_reject_r
        try:
            self._shortcut_reject_r.setAutoRepeat(False)
        except Exception:
            pass

        self._shortcut_skip_s = QShortcut(QKeySequence(Qt.Key.Key_S), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_skip_s.activated.connect(self._handle_shortcut_skip)
        self._shortcut_skip_s.activatedAmbiguously.connect(self._handle_shortcut_skip)
        self._shortcut_skip_lower = self._shortcut_skip_s
        self._shortcut_skip = self._shortcut_skip_s
        try:
            self._shortcut_skip_s.setAutoRepeat(False)
        except Exception:
            pass

        self._shortcut_player_k = QShortcut(QKeySequence(Qt.Key.Key_K), self, context=Qt.ShortcutContext.WindowShortcut)
        self._shortcut_player_k.activated.connect(self._handle_shortcut_player_char)
        self._shortcut_player_k.activatedAmbiguously.connect(self._handle_shortcut_player_char)
        try:
            self._shortcut_player_k.setAutoRepeat(False)
        except Exception:
            pass

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
        # P1-09 (K-24 / F063): varsayılan 84 idi → val oranı %16; kodun
        # her yerindeki varsayılan 0.15 ile uyuşmuyordu (dondurma %16,
        # hazırlık/eğitim %15 varsayıyordu). Kaydırıcı artık 85: train
        # %85 / val %15 — kod varsayılanıyla aynı.
        self.split_range.setValue(85)
        self.split_val = QLabel("%85")
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
        self.freeze_state_lbl.setStyleSheet(f"color:{C['amber']};font-family:Consolas;font-weight:700;font-size:13px;")
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
        # P2-22: Hazırla toggle'ı değişince sarı buton etiketi kapsamı anında
        # güncellenir (kilit tazelenmesi + 1 sn zamanlayıcı da aynı etiketi
        # tazeler; senkronlama lambdaları blockSignals kullandığı için bu
        # bağlantı ayrıca gerekli).
        self.toggle_approved_only_prepare.toggled.connect(
            lambda _checked: self._refresh_train_rtx_label()
        )
        self.toggle_synthetic_10x = FactoryToggleSwitch(
            "10× sentetik çoğaltma",
            checked=True,
            accent_hex=C["amber"],
            parent=card,
        )
        self.toggle_synthetic_10x.toggled.connect(self._on_toggles_changed)
        right.addWidget(self.toggle_approved_only)
        right.addWidget(self.toggle_synthetic_10x)
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
            ("Dünya / Arayüz (12–15)", "nav"),
            ("Can / Skill (16–21)", "vital"),
            ("İksir / Kinah / Oyuncu (22–26)", "essence"),
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
        dist_layout.setContentsMargins(10, 4, 10, 10)
        dist_layout.setSpacing(4)
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
        dist_scroll.setMinimumHeight(180)
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
            "border-radius:999px;padding:6px 12px;font-size:12px;font-weight:700;"
        )
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
        self.hp_eta = QLineEdit("—", self)
        self.hp_eta.setEnabled(False)
        fields = [
            ("Model", self.hp_model),
            ("Epoch", self.hp_epoch),
            ("Batch", self.hp_batch),
            ("Görüntü boyutu", self.hp_imgsz),
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
            # Üçlü senkron (F-0603/D028): Hazırla toggle'ı değişince Eğit
            # toggle'ı VE Adım 2 checkbox'ı da aynı değere çekilir — daha
            # önce yalnızca Eğit toggle'ı senkronlanıyor, bayat checkbox
            # AÇIK kalıp _only_approved_enabled() üzerinden eğitimi
            # sessizce onaylı-havuz moduna sokuyordu.
            self.toggle_approved_only_prepare.toggled.connect(
                lambda val: (
                    self.toggle_approved_only_train.blockSignals(True),
                    self.toggle_approved_only_train.setChecked(val),
                    self.toggle_approved_only_train.blockSignals(False),
                    (
                        self.chk_only_approved.blockSignals(True),
                        self.chk_only_approved.setChecked(val),
                        self.chk_only_approved.blockSignals(False),
                    )
                    if hasattr(self, "chk_only_approved")
                    else None,
                )
            )
            self.toggle_approved_only_train.toggled.connect(
                lambda val: (
                    self.toggle_approved_only_prepare.blockSignals(True),
                    self.toggle_approved_only_prepare.setChecked(val),
                    self.toggle_approved_only_prepare.blockSignals(False),
                )
            )
        if hasattr(self, "chk_only_approved"):
            # Adım 2 checkbox'ı da aynı tek duruma bağlanır (F-0603).
            self.chk_only_approved.toggled.connect(
                lambda val: (
                    self.toggle_approved_only_train.blockSignals(True),
                    self.toggle_approved_only_train.setChecked(val),
                    self.toggle_approved_only_train.blockSignals(False),
                    self.toggle_approved_only_prepare.blockSignals(True),
                    self.toggle_approved_only_prepare.setChecked(val),
                    self.toggle_approved_only_prepare.blockSignals(False),
                )
            )
            self.toggle_approved_only_train.toggled.connect(
                lambda val: (
                    self.chk_only_approved.blockSignals(True),
                    self.chk_only_approved.setChecked(val),
                    self.chk_only_approved.blockSignals(False),
                )
            )
        self.chk_train_only_approved = self.toggle_approved_only
        # P2-22: Eğit toggle'ı değişince sarı buton etiketi kapsamı anında
        # güncellenir (kilit tazelenmesi + 1 sn zamanlayıcı da aynı etiketi
        # tazeler).
        self.toggle_approved_only_train.toggled.connect(
            lambda _checked: self._refresh_train_rtx_label()
        )
        approved_row.addWidget(self.toggle_approved_only)
        approved_row.addStretch()
        lay.addLayout(approved_row)

        self.lbl_train_summary = QLabel("", self)
        self.lbl_train_summary.setWordWrap(True)
        self.lbl_train_summary.setStyleSheet("font-size:12px; color:#D7DEE8; padding:2px 14px;")
        lay.addWidget(self.lbl_train_summary)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(14, 4, 14, 8)
        self.btn_train_only_approved = QPushButton("🚀 Sadece Onaylananlarla Eğit", self)
        self.btn_train_only_approved.setObjectName("PipeGreen")
        self.btn_train_only_approved.setMinimumHeight(48)
        self.btn_train_only_approved.clicked.connect(self._on_train_only_approved_clicked)
        self.btn_train_rtx = QPushButton("⚡ Eğitimi başlat", self)
        self.btn_train_rtx.setObjectName("PipePrimary")
        self.btn_train_rtx.setMinimumHeight(48)
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
        self.train_progress_bar.setFixedHeight(12)
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
        note.setStyleSheet(f"color:{C['muted']};padding:0 16px 16px;font-size:13px;")
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
        self.val_empty.setStyleSheet(f"color:{C['muted']};padding:16px 18px;font-size:14px;")
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
            c.setStyleSheet(f"color:{C['muted']};font-size:11px;font-weight:800;")
            w.setStyleSheet("font-family:Consolas;font-size:22px;font-weight:700;")
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
                "padding:12px 18px;border-bottom:1px solid #262D3A;font-size:13px;"
            )
            lay.addWidget(w)

        pub_row = QHBoxLayout()
        pub_row.setContentsMargins(14, 12, 14, 12)
        self.btn_publish = QPushButton("⇪ Modeli yayınla ve botu güncelle", self)
        self.btn_publish.setObjectName("PipePrimary")
        self.btn_publish.setMinimumHeight(48)
        self.btn_publish.clicked.connect(self._publish_model)
        self.btn_rollback = QPushButton("↩ Önceki modele dön", self)
        self.btn_rollback.setObjectName("PipeBtn")
        self.btn_rollback.setMinimumHeight(48)
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

    def _build_live_journal_strip(self) -> QFrame:
        """Alttaki filtreli Canlı Günlük şeridi (prototip): Tümü / INFO /
        Önizleme + Temizle. Sağdaki Canlı log ile aynı akışı gösterir;
        satırlar _append_log ile beslenir."""
        strip = QFrame(self)
        strip.setObjectName("PipeJournalStrip")
        lay = QVBoxLayout(strip)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("<b>Canlı günlük</b>")
        title.setStyleSheet("font-size:13px;")
        head.addWidget(title)
        sub = QLabel("Tüm hat olayları tek akışta — türe göre süzülür, sağdaki logla birlikte güncellenir.")
        sub.setStyleSheet("color:#9AA3B2; font-size:11px;")
        head.addWidget(sub)
        head.addStretch()
        self._journal_filter_group = QButtonGroup(self)
        for key, label in (("all", "Tümü"), ("info", "INFO"), ("preview", "Önizleme")):
            btn = QPushButton(label, self)
            btn.setObjectName("PipeBtn")
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.clicked.connect(lambda _=False, k=key: self._set_journal_filter(k))
            self._journal_filter_group.addButton(btn)
            head.addWidget(btn)
        clear = QPushButton("Temizle", self)
        clear.setObjectName("PipeBtn")
        clear.clicked.connect(self._clear_journal)
        head.addWidget(clear)
        lay.addLayout(head)
        self.live_journal_box = QPlainTextEdit(self)
        self.live_journal_box.setObjectName("live_journal_box")
        self.live_journal_box.setReadOnly(True)
        self.live_journal_box.setMaximumHeight(110)
        self.live_journal_box.setMinimumHeight(64)
        self.live_journal_box.setPlainText("[HAZIR] Canlı günlük açık.")
        lay.addWidget(self.live_journal_box)
        return strip

    def _set_journal_filter(self, mode: str) -> None:
        self._journal_filter = mode
        self._render_journal()

    def _journal_line_visible(self, line: str) -> bool:
        mode = getattr(self, "_journal_filter", "all")
        if mode == "preview":
            return "[ÖNİZLEME]" in line
        if mode == "info":
            return "[ÖNİZLEME]" not in line
        return True

    def _render_journal(self) -> None:
        box = getattr(self, "live_journal_box", None)
        if box is None:
            return
        lines = [ln for ln in getattr(self, "_journal_lines", []) if self._journal_line_visible(ln)]
        box.setPlainText("\n".join(lines[-200:]) or "[HAZIR] Canlı günlük açık.")
        box.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_journal(self) -> None:
        self._journal_lines = []
        self._render_journal()

    def _build_right_rail(self) -> QWidget:
        C = self.PIPELINE_COLORS
        rail = QWidget(self)
        # W3 (yatay tasma): sabit 330 px yerine esnek genislik (280-330);
        # dar pencerede kuculup Onayla adimina yer acar. Varsayilan
        # boyutta 330 px'te kalir.
        rail.setMinimumWidth(280)
        rail.setMaximumWidth(330)
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
            # Ağır kuyruk taraması worker thread'de (UI donmasın); zaten
            # çalışan bir yükleme varsa istek düşer (o yükleme kuyruğu kapsar).
            self._request_queue_reload()
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
        # P3-28: pending tek kaynaktan (count_review_state) — önbellek
        # okunmaz, bayat kalmaz (F008-06).
        pending = self.count_review_state()["pending"]
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
            1: False,  # yakalama yalnızca oturum karesi varsa / adım geçildiyse biter (F-0692)
            2: pending <= 0 and self._pipeline.get("reviewed", 0) > 0,
            3: frozen,
            4: trained,
            5: validated,
            6: published,
        }
        # Capture done if user left step 1 after starting, or frames exist
        if current > 1 or int(self._pipeline.get("session", 0)) > 0:
            done_flags[1] = True

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
        # P3-28: pending tek kaynaktan — count_review_state() aynı zamanda
        # `_pipeline["pending"]` önbelleğini tazeler; aşağıdaki iki kullanım
        # aynı sayıyı görür.
        pending = self.count_review_state()["pending"]
        if pending <= 0:
            self.next_title.setText("Kuyruk boş — veri setini hazırla")
            self.next_text.setText(
                "Tüm kareler onaylandı. Sağlık kontrollerini bitirip veri setini dondur."
            )
            self.btn_next.setText("Hazırla adımına git →")
            self._next_target = 3
            return
        self.next_title.setText(f"{pending} otomatik etiket onay bekliyor")
        self.next_text.setText(
            "Eğitim, onaylanmamış karelerle başlamamalı. Eşik üstündekileri toplu onayla, kalanlara tek tek bak."
        )
        self.btn_next.setText("Onay kuyruğuna git →")
        self._next_target = 2

    def _on_next_clicked(self) -> None:
        action = str(getattr(self, "_next_action", "goto") or "goto")
        target = int(getattr(self, "_next_target", 2))
        if action == "train":
            # P2-22: btn_next üçüncü "Eğitimi başlat" girişidir — yeşil/sarı
            # butonla aynı senkron kilidi kullanır; çift tık adım
            # değiştirtmeden yoksayılır.
            _worker = getattr(self, "_training_worker", None)
            if getattr(self, "_training_starting", False) or (
                _worker is not None and _worker.isRunning()
            ):
                self._append_log(
                    "[BİLGİ] Eğitim zaten başlatılıyor/devam ediyor — "
                    "'Sıradaki' tıkı yoksayıldı."
                )
                return
            if int(self._pipeline.get("step", 1)) != 4:
                self._go_step(4)
            self.on_train_rtx_clicked()
            return
        if action == "validate":
            # _go_step(5) zaten _run_validate çağırır; ikinci çağrı çift
            # doğrulama yapardı (F-0771).
            self._go_step(5)
            return
        if action == "publish":
            if int(self._pipeline.get("step", 1)) != 6:
                self._go_step(6)
            self._publish_model()
            return
        self._go_step(target)

    def _update_blockers(self) -> None:
        # P3-28: pending tek kaynaktan (count_review_state) — önbellek
        # okunmaz; bazı çağrı yollarında bayat kalıyordu (F008-06).
        pending = self.count_review_state()["pending"]
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
        # Count empty among ALL classes (F-0701: kapsam 27 olmalı)
        empty = 0
        names: List[str] = []
        for spec in YOLO_12_CLASS_SPECS:
            cid = int(spec["id"])
            if int(counts.get(cid, 0)) <= 0:
                empty += 1
                names.append(str(spec["label_tr"]))
        self._empty_class_names = names
        return empty

    def _update_train_lock(self, refresh_summary: bool = True) -> None:
        frozen = bool(self._pipeline["frozen"])
        # P0-4 (K-20 delta): buton görünümü kapı kararıyla tutarlı olmalı —
        # dondurulunca aktif görünüp tıklayınca RED atan durum kalmasın.
        # Kapı her iki modda da `_pending_train_blockers` sayar (el etiketli
        # korumalı kareler sayılmaz); engel varken iki eğitim butonu da
        # kilitli görünür. Hesap hatası fail-closed: butonlar kilitli kalır.
        try:
            # P3-28: kapı sayısı tek kaynaktan (count_review_state).
            train_blockers = self.count_review_state()["blockers"]
        except Exception:
            train_blockers = 1
        training_running = (
            self._training_worker is not None and self._training_worker.isRunning()
        )
        # P2-22: senkron bayrak da kapıya girer — tık anında kurulan bayrak
        # varken worker henüz `isRunning()` dönmese bile butonlar açılmaz.
        can_train = (
            frozen
            and train_blockers == 0
            and not training_running
            and not getattr(self, "_training_starting", False)
        )
        self.train_lock_pill.setVisible(not frozen)
        self.btn_train_rtx.setEnabled(can_train)
        if hasattr(self, "btn_train_only_approved"):
            self.btn_train_only_approved.setEnabled(can_train)
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
        try:
            self._refresh_train_rtx_label()
        except Exception:
            pass
        if refresh_summary:
            try:
                self._refresh_train_summary()
            except Exception:
                pass

    def _refresh_train_rtx_label(self) -> None:
        """P2-22 (bölüm 6.2, F040-K2): sarı buton etiketi kapsamı yazar.

        Toggle AÇIK iken sarı buton onaylı havuzla, KAPALI iken tüm veriyle
        eğitir — eski "⚡ Eğitimi başlat" etiketi bunu söylemiyordu (toggle
        AÇIK iken sarı ≡ yeşil). Etiket + tooltip her kilit tazelenmesinde
        toggle durumundan okunur.
        """
        btn = getattr(self, "btn_train_rtx", None)
        if btn is None:
            return
        if self._only_approved_enabled():
            btn.setText("⚡ Eğitimi başlat — yalnızca onaylı")
            btn.setToolTip(
                "Yalnızca onaylı havuzla eğitir (dataset/approved) — "
                "'Sadece Onaylananlarla Eğit' anahtarı AÇIK."
            )
        else:
            btn.setText("⚡ Eğitimi başlat — tüm veri")
            btn.setToolTip(
                "Tüm veriyle eğitir (dataset/train + val, onaylanmamış "
                "kareler dahil) — 'Sadece Onaylananlarla Eğit' anahtarı KAPALI."
            )

    def _refresh_training_locks(self, refresh_summary: bool = True) -> None:
        """P2-22 (bölüm 6.2, F048-O1): kilitleri sinyal beklemeden tazele.

        Worker sinyali (`finished` / `training_finished` /
        `training_failed` / `training_deferred`) kaçarsa eğitim butonları
        kilitli kalıyordu. Bu yol yer-doğruluktan
        (`_training_worker.isRunning()`) okur: çalışan worker yoksa bayat
        senkron bayrak temizlenir, kilit yeniden hesaplanır, Durdur butonu
        çalışan worker yoksa kapatılır. Periyodik zamanlayıcı (1 sn) ve
        eğitim bitiş sinyal işleyicileri aynı yolu kullanır. Durdur butonu
        bu yoldan asla AÇILMAZ — yalnızca yeni eğitim başlangıcı açar
        (durdurma isteğinden sonra worker kapanırken butonun geri açılması
        istenmez).
        """
        try:
            running = (
                self._training_worker is not None
                and self._training_worker.isRunning()
            )
        except Exception:
            running = False
        try:
            if not running and getattr(self, "_training_starting", False):
                self._training_starting = False
            self._update_train_lock(refresh_summary=refresh_summary)
            btn_stop = getattr(self, "btn_stop_train", None)
            if btn_stop is not None and not running:
                btn_stop.setEnabled(False)
        except Exception as exc:
            logger.debug("training lock refresh error: %s", exc)

    def _refresh_train_summary(self) -> None:
        """Eğitim ekranı özeti: basılan butona göre hangi setle, kaç kareyle
        eğitim yapılacağı. Havuz sayıları manifestten, tüm-veri sayıları
        dataset/images altından sayılır."""
        lbl = getattr(self, "lbl_train_summary", None)
        if lbl is None:
            return
        lines: List[str] = []
        try:
            from dataset_factory.approved_pool import get_approved_pool

            stats = get_approved_pool().get_approved_stats()
            total = int(stats.get("total_approved_images", 0) or 0)
            if total > 0:
                lines.append(
                    "🚀 Sadece Onaylananlarla: "
                    f"{total} kare (train {int(stats.get('train_count', 0) or 0)} + "
                    f"val {int(stats.get('val_count', 0) or 0)} · "
                    f"{int(stats.get('total_boxes', 0) or 0)} kutu) — kaynak: dataset/approved"
                )
            else:
                lines.append("🚀 Sadece Onaylananlarla: havuz boş — önce Adım 3'te dondur.")
        except Exception:
            lines.append("🚀 Sadece Onaylananlarla: havuz okunamadı — önce Adım 3'te dondur.")
        try:
            ds = self._dataset_dir()
            counts: Dict[str, int] = {}
            total_full = 0
            for split in ("train", "val"):
                folder = ds / "images" / split
                n = 0
                if folder.is_dir():
                    with os.scandir(folder) as it_scan:
                        for entry in it_scan:
                            if entry.is_file() and entry.name.lower().endswith(
                                (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                            ):
                                n += 1
                counts[split] = n
                total_full += n
            lines.append(
                "⚡ Eğitimi başlat: "
                f"{total_full} kare (train {counts['train']} + val {counts['val']}) — kaynak: dataset"
            )
        except Exception:
            lines.append("⚡ Eğitimi başlat: veri seti sayılamadı.")
        if not self._pipeline.get("frozen"):
            lines.append("⛔ Önce Adım 3'te dondur — eğitim kilitli.")
        else:
            # P0-4 (K-20 delta): pending kapısı her iki modda da geçerli —
            # özet satırı kapı kararını aynen yansıtır (onaylı modda yeşil,
            # normal modda sarı buton). Sayı tek kaynaktan (P3-28).
            blockers = self.count_review_state()["blockers"]
            if blockers > 0:
                btn_name = "Yeşil buton" if self._only_approved_enabled() else "Sarı buton"
                lines.append(
                    f"⛔ {btn_name} şu an başlamaz: {blockers} kare onay bekliyor "
                    "(el etiketleri sayılmaz) — Adım 2'de tek tek onayla/reddet."
                )
        lbl.setText("\n".join(lines))

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

    def _save_capture_snapshot(self, snap: Any) -> bool:
        """Zaman makinesi karesini diske (images + review JSON) yazar.

        Yalnızca yazma başarılıysa True döner; sayaç yalnızca bu durumda
        artar (F-0604) — daha önce kare diske yazılmadan sayaç artıyordu.
        """
        frame = getattr(snap, "frame", None)
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        snap_ts = float(getattr(snap, "timestamp", 0.0) or 0.0)
        if snap_ts and snap_ts == float(getattr(self, "_last_capture_snap_ts", 0.0) or 0.0):
            return False  # aynı kare tekrar sayılmaz
        now_mono = time.monotonic()
        if now_mono - float(getattr(self, "_last_capture_save_mono", 0.0) or 0.0) < 1.0:
            return False  # disk selini önlemek için en fazla ~1 kare/sn
        stem = f"cap_{int(time.time() * 1000)}_{int(self._pipeline.get('session', 0)) + 1:04d}"
        img_path = self._dataset_dir() / "images" / "train" / f"{stem}.jpg"
        if not _safe_imwrite_unicode(img_path, frame, quality=92):
            logger.warning("Yakalama karesi diske yazılamadı: %s", img_path)
            return False
        rev_path = self._dataset_dir() / "review" / "train" / f"{stem}.json"
        try:
            rev_path.parent.mkdir(parents=True, exist_ok=True)
            anns = [dict(a) for a in (getattr(snap, "detections", None) or []) if isinstance(a, dict)]
            rev_path.write_text(
                json.dumps(
                    {"stem": stem, "split": "train", "trusted": [], "weak": anns},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Yakalama review JSON yazılamadı (%s): %s", rev_path, exc)
            try:
                img_path.unlink()
            except OSError:
                pass
            return False
        self._last_capture_snap_ts = snap_ts
        self._last_capture_save_mono = now_mono
        return True

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
            if not self._save_capture_snapshot(snap):
                return
            self._pipeline["session"] = int(self._pipeline.get("session", 0)) + 1
            if hasattr(self, "session_val"):
                self.session_val.setText(f"+{self._pipeline['session']} kare")
            if int(self._pipeline["session"]) % 20 == 0:
                # P3-28: logdaki kuyruk sayısı tek kaynaktan.
                self._append_log(
                    f"[YAKALAMA] Oturum +{self._pipeline['session']} kare "
                    f"(onay kuyruğu {self.count_review_state()['pending']})"
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
        mode = str(mode or "live").lower()
        if mode not in ("live", "boxes", "clean"):
            mode = "live"
        self._preview_mode = mode
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            canvas._view_mode = mode
            canvas.update()
        self._append_log(f"[GÖRÜNÜM] Önizleme modu: {mode}")


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

    def _pending_train_blockers(self) -> int:
        """Eğitim kapısı için bekleyen sayısı: done olmayanlardan el etiketi
        korumalılar sayılmaz — onlar dondurmayla havuza girer (faz 1), kuyrukta
        kalmaları eğitimi kilitlememeli. Kalanlar (güven eşiği altı / boş kare)
        tek tek incelenir."""
        return sum(
            1
            for it in self._review_items
            if not it.get("done") and not self._is_hand_label_protected_item(it)
        )

    def count_review_state(self) -> Dict[str, int]:
        """Tek sayım kaynağı (P3-28) — inceleme kuyruğu sayıları.

        Kapı, dashboard, web önizleme (`export_web_state`) ve log BU
        fonksiyonu kullanır; hiçbiri kendi sayım kuralını yazmaz, hiçbiri
        `_pipeline["pending"]` / `_pipeline["reviewed"]`'u doğrudan okuyup
        kendi formülünü kurmaz. Tanımlar P2-21'deki gibidir, DEĞİŞMEDİ:

        - ``pending``:  ``_pending_count()`` — yüklü kuyrukta ``done``
          olmayanlar (bellek kuyruğu; disk dosya sayısı değil).
        - ``reviewed``: ``_pipeline["reviewed"]`` — oturumda incelenen
          (onaylanan + reddedilen) KÜMÜLATİF sayı; finalize onaylıları
          listeden çıkarsa da sıfırlanmaz (F043-K1).
        - ``total``:    ``reviewed + pending`` — çubuk ve web
          ``reviewProgTxt`` paydası (P2-21'de iki yerde aynı formüldü,
          artık tek yerden gelir).
        - ``blockers``: ``_pending_train_blockers()`` — pending'den el
          etiketli korumalı kareler çıkarılmış KAPI sayısı (P0-4);
          eğitim/dondurma kapıları bu sayıya bakar.

        Yan etki: ``_pipeline["pending"]`` bu çağrıda tazelenir — eskiden
        her okuyucu kendi anında yazıyordu (biri yazmayı unutunca okuyucu
        bayat kalıyordu, F008-06); artık tek yazıcı burasıdır.
        ``_pending_count()`` / ``_pending_train_blockers()`` sayım
        ilkelleri olarak kalır — testler onları doğrudan çağırır; bu
        fonksiyon onların ÜZERİNDEKİ tek tüketici arayüzüdür.
        """
        pending = self._pending_count()
        reviewed = int(self._pipeline.get("reviewed", 0))
        blockers = self._pending_train_blockers()
        self._pipeline["pending"] = pending
        return {
            "pending": pending,
            "reviewed": reviewed,
            "total": reviewed + pending,
            "blockers": blockers,
        }

    def _sync_pending_from_queue(self) -> None:
        """pending + tüm onay sayaçlarını `_review_items` ile hizala."""
        self.count_review_state()
        self._update_pending_labels()

    @staticmethod
    def _rejected_stem_is_hand_protected(review_path: Path, label_path: Path) -> bool:
        """Reddedilmiş karenin el etiketi koruması var mı? (tarama GC yolu).

        Worker thread'den de çağrılır: QWidget'e dokunmaz, yalnızca disk okur.
        Korumalı sayılır: (a) labels TXT var + review JSON yok → TXT
        kökenli düzeltilmiş el etiketi karesi; (b) review JSON var ve
        içindeki kutulardan en az biri hand_label → el işi kaybolmasın.
        Korumalı karede tarama HİÇBİR dosyayı silmez (görsel + TXT durur),
        kare yalnızca kuyruktan dışarıda kalır.

        P1-16 (K-17) fail-closed: review JSON okunamazsa / payload dict
        değilse kare KORUMALI sayılır — bozuk dosya "korumasız" sayılıp
        silinmez; bozuk dosyanın karantinası tarama GC dalında yapılır.
        """
        try:
            rev = Path(review_path)
            lbl = Path(label_path)
            if lbl.is_file() and not rev.is_file():
                return True
            if rev.is_file():
                payload = json.loads(rev.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    logger.warning(
                        "[KORUMA] %s: review JSON dict değil — korumalı sayılıyor",
                        rev,
                    )
                    return True
                for b in list(payload.get("weak") or []) + list(payload.get("trusted") or []):
                    if isinstance(b, dict) and b.get("hand_label") is True:
                        return True
        except Exception as exc:
            # Fail-closed: okuma/çözümleme hatası = içerik bilinmiyor =
            # korumalı say. (Eski davranış False dönüp GC'nin kareyi
            # silmesine yol açıyordu — K-17.)
            logger.warning(
                "[KORUMA] %s / %s okunamadı (%s) — korumalı sayılıyor",
                review_path,
                label_path,
                exc,
            )
            return True
        return False

    @staticmethod
    def _quarantine_corrupt_gc_files(
        dataset: Path,
        split: str,
        stem: str,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Tarama GC silme kümesindeki bozuk review JSON / labels TXT'yi karantinaya al.

        P1-16 (K-17/K-05): bozuk dosya silinmez — ``<dataset>/quarantine``
        altına taşınır. Bulunan her bozuk dosya için bir olay dict'i
        döner (ve ``events`` verildiyse ona da eklenir); UI thread'deki
        çağıran bunları ``_append_log`` ile kullanıcıya bildirir.
        Karantina kendisi başarısız olursa dosya yerinde bırakılır ve
        olayda ``quarantined_to=None`` olur — yine de silinmez.
        """
        found: List[Dict[str, Any]] = []
        try:
            from dataset_factory.label_trust import (
                label_txt_corrupt_lines as _txt_bad_lines,
                quarantine_corrupt_file as _quarantine,
                review_json_is_corrupt as _rev_corrupt,
            )
        except Exception as exc:  # pragma: no cover - import hatası
            logger.warning("[KARANTİNA] karantina yardımcısı import edilemedi: %s", exc)
            return found
        ds = Path(dataset)
        rev = ds / "review" / split / f"{stem}.json"
        if _rev_corrupt(rev):
            dest = _quarantine(
                rev, dataset_dir=ds, reason="bozuk review JSON (tarama GC, K-17)"
            )
            found.append({
                "stem": stem,
                "split": split,
                "kind": "review JSON",
                "file": str(rev),
                "quarantined_to": str(dest) if dest is not None else None,
                "reason": "JSON çözümlenemedi",
            })
        lbl = ds / "labels" / split / f"{stem}.txt"
        bad_lines = _txt_bad_lines(lbl)
        if bad_lines:
            if bad_lines == [-1]:
                reason = "dosya okunamadı"
            else:
                shown = ",".join(str(n) for n in bad_lines[:10])
                reason = f"bozuk satır(lar): {shown}" + (
                    f" (+{len(bad_lines) - 10} satır daha)" if len(bad_lines) > 10 else ""
                )
            dest = _quarantine(
                lbl, dataset_dir=ds, reason=f"bozuk labels TXT (tarama GC, K-17): {reason}"
            )
            found.append({
                "stem": stem,
                "split": split,
                "kind": "labels TXT",
                "file": str(lbl),
                "quarantined_to": str(dest) if dest is not None else None,
                "reason": reason,
            })
        if events is not None:
            events.extend(found)
        return found

    @staticmethod
    def _scan_review_queue_items(
        dataset_dir: Path,
        rejected_stems: set,
        approved_pool: Any = None,
        review_audit_mode: bool = False,
        queue_filter_mode: str = "all",
        quarantine_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Disk review JSON + images/{train,val} taraması (saf veri).

        Worker thread'den de çağrılır (YOLO açılış donmasını önlemek
        için): dosya okuma/silme dışında yan etkisi yoktur, QWidget'e
        dokunmaz. Onaylanmış ve reddedilmiş kareler kuyruğa geri gelmez.

        ``quarantine_events`` verilirse, tarama GC'nin karantinaya aldığı
        bozuk dosyalar bu listeye eklenir — UI thread'deki çağıran
        kullanıcıya bildirir (P1-16/K-17).

        Kuyruk-kapalı kararı tek kanonik yoldan okunur
        (``approval_state.is_frame_closed`` — P3-27).
        """
        dataset = Path(dataset_dir)
        review_root = dataset / "review"
        images_root = dataset / "images"
        labels_root = dataset / "labels"
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if approved_pool is None:
            try:
                from dataset_factory.approved_pool import get_approved_pool

                approved_pool = get_approved_pool(dataset)
            except Exception:
                approved_pool = None

        # P0-1 / K-02: tek kanonik TXT uzlaştırması label_trust'ta yaşar;
        # dialog kendi kuralını uydurmaz, onu çağırır. Import edilemezse
        # tarama uzlaştırmasız devam eder (eski davranış).
        _reconcile_txt = None
        try:
            from dataset_factory.label_trust import (
                _reconcile_txt_hand_labels as _reconcile_txt,
            )
        except Exception as exc:  # pragma: no cover - import hatası
            logger.debug("TXT uzlaştırması import edilemedi: %s", exc)

        def _stamp_txt_hand_names(boxes: List[Dict[str, Any]]) -> None:
            """Uzlaştırmayla trusted'a taşınan TXT kutusuna class_name yaz."""
            for b in boxes:
                if (
                    b.get("source") == HAND_LABEL_TXT_SOURCE
                    and b.get("class_name") is None
                    and isinstance(b.get("class_id"), int)
                ):
                    b["class_name"] = CLASS_ID_TO_SPEC.get(b["class_id"], {}).get(
                        "key", f"class_{b['class_id']}"
                    )

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
                    # El etiketi koruması: TXT kökenli (veya el etiketli
                    # review JSON'lu) karede HİÇBİR dosya silinmez —
                    # görsel + labels TXT durur, kare yalnızca kuyruğa girmez.
                    # P1-16 (K-17) fail-closed: koruma kontrolü hata verirse
                    # korumalı sayılır. Her iki dalda da silmeden ÖNCE bozuk
                    # review JSON / labels TXT karantinaya alınır, silinmez.
                    if not YOLOFactoryDialog._rejected_stem_is_hand_protected(
                        review_root / split / f"{stem}.json",
                        labels_root / split / f"{stem}.txt",
                    ):
                        YOLOFactoryDialog._quarantine_corrupt_gc_files(
                            dataset, split, stem, quarantine_events
                        )
                        # Kullanıcı kararı (11 Kasım): reddet kalıcı
                        # silmedir — çöp/geri yükleme yok, düz unlink.
                        # (Bozuk dosya yukarıda karantinaya taşındıysa
                        # unlink sessizce no-op olur — helper içindeki
                        # `_unlink_quiet` aynı davranışı korur.)
                        YOLOFactoryDialog._delete_rejected_scan_files(
                            split,
                            stem,
                            [
                                img_p,
                                review_root / split / f"{stem}.json",
                                review_root / split / f"{stem}.sidecar.json",
                                labels_root / split / f"{stem}.txt",
                            ],
                        )
                    else:
                        # Korumalı kare: bozuk dosya varsa karantinaya al,
                        # kalan dosyalar silinmez (içerik bilinmiyor —
                        # fail-closed; kare K-18 limbosunda kalır, kanıt
                        # karantinada saklanır).
                        quarantined = YOLOFactoryDialog._quarantine_corrupt_gc_files(
                            dataset, split, stem, quarantine_events
                        )
                        if quarantined:
                            logger.warning(
                                "[KARANTİNA] reddedilen korumalı kare %s/%s: %d bozuk dosya karantinaya alındı, diğer dosyalar silinmedi",
                                split,
                                stem,
                                len(quarantined),
                            )
                        else:
                            logger.debug(
                                "[KORUMA] reddedilen el etiketi kare korundu (tarama): %s/%s",
                                split, stem,
                            )
                    continue
                seen_keys.add((split, stem))
                is_in_pool = bool(approved_pool is not None and approved_pool.is_approved(stem))
                allow_approved = bool(
                    review_audit_mode
                    or queue_filter_mode == "approved_pool"
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
                                is_frame_closed(payload)
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

                if payload_loaded and _reconcile_txt is not None and lbl_path.is_file():
                    # JSON dalı (K-03): review JSON'u olan karede bile TXT
                    # kutuları aynı kural ile bellek içi trusted el etiketi
                    # olur. YALNIZCA bellek içi — okuma yolunda review JSON
                    # yazılmaz (F-0772).
                    trusted, weak = _reconcile_txt(trusted, weak, lbl_path)
                    _stamp_txt_hand_names(trusted)

                if not payload_loaded and lbl_path.is_file():
                    # P1-16 (K-17): TXT satır satır çözümlenir — tek bozuk
                    # satır tüm el etiketlerini bellekten düşürmez; bozuk
                    # satır atlanır, sağlam kutular korunur, satır
                    # numaraları loga yazılır.
                    bad_txt_lines: List[int] = []
                    try:
                        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
                    except Exception as exc:
                        logger.warning(
                            "[KORUMA] #%s/%s: labels TXT okunamadı (%s) — kutular belirsiz, kare kuyruğa boş giriyor",
                            split, stem, exc,
                        )
                        lines = []
                    for _ln_no, ln in enumerate(lines, 1):
                        parts = ln.strip().split()
                        if len(parts) >= 5:
                            try:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                cconf = float(parts[5]) if len(parts) > 5 else 0.0
                            except (ValueError, TypeError):
                                bad_txt_lines.append(_ln_no)
                                continue
                            # B-07 + el etiketi koruması: .txt'te güven
                            # sütunu yoksa (YOLO .txt 5 sütunludur) kutu
                            # eskiden 0.95 güvenle "Emin" sayılıyordu —
                            # ölçülmemiş varsayılan, ölçülmüş güvenmiş
                            # gibi toplu onaya giriyordu. Bu dalda
                            # review JSON'u olmayan ama labels TXT'si
                            # olan kare, düzeltilmiş EL ETİKETİ sayılır
                            # (model çıktısı değil): kutu hand_label
                            # damgasıyla TRUSTED listesine girer, güven
                            # ölçülmemiş olarak 0.0 kalır (uydurulmaz).
                            # Koruma ≠ onay: `human_approved` /
                            # `bulk_approved` / `done="ok"` verilmez —
                            # yalnızca budama/reddet silmesine karşı
                            # korunur; onay ancak kare tek tek
                            # incelenip onaylanınca olur (B-07 aynen).
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
                                "hand_label": True,
                                "source": HAND_LABEL_TXT_SOURCE,
                            })
                        elif parts:
                            bad_txt_lines.append(_ln_no)
                    if bad_txt_lines:
                        _shown = ",".join(str(n) for n in bad_txt_lines[:10])
                        logger.warning(
                            "[KORUMA] #%s/%s: labels TXT'de %d bozuk satır atlandı (satır %s) — diğer kutular korundu",
                            split, stem, len(bad_txt_lines), _shown,
                        )

                # NOT: okuma yolunda review JSON yazılmaz (F-0772); dosya yalnızca
                # kullanıcı onayı/reddiyle _persist_review_item'da yazılır.

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
                item["conf"] = YOLOFactoryDialog._frame_conf(item)
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
                    # El etiketi koruması (Aşama 1 ile aynı): el etiketli
                    # review JSON'un labels TXT'si silinmez. P1-16 (K-17):
                    # silmeden önce bozuk dosyalar karantinaya alınır.
                    if not YOLOFactoryDialog._rejected_stem_is_hand_protected(
                        rpath, labels_root / split / f"{stem}.txt"
                    ):
                        YOLOFactoryDialog._quarantine_corrupt_gc_files(
                            dataset, split, stem, quarantine_events
                        )
                        # Kullanıcı kararı (11 Kasım): reddet kalıcı
                        # silmedir — çöp/geri yükleme yok, düz unlink.
                        YOLOFactoryDialog._delete_rejected_scan_files(
                            split,
                            stem,
                            [
                                rpath,
                                review_root / split / f"{stem}.sidecar.json",
                                labels_root / split / f"{stem}.txt",
                            ],
                        )
                    else:
                        quarantined = YOLOFactoryDialog._quarantine_corrupt_gc_files(
                            dataset, split, stem, quarantine_events
                        )
                        if quarantined:
                            logger.warning(
                                "[KARANTİNA] reddedilen korumalı kare %s/%s: %d bozuk dosya karantinaya alındı, diğer dosyalar silinmedi",
                                split,
                                stem,
                                len(quarantined),
                            )
                        else:
                            logger.debug(
                                "[KORUMA] reddedilen el etiketi kare korundu (tarama, review): %s/%s",
                                split, stem,
                            )
                    continue
                if (split, stem) in seen_keys:
                    continue
                seen_keys.add((split, stem))
                is_in_pool = bool(approved_pool is not None and approved_pool.is_approved(stem))
                allow_approved = bool(
                    review_audit_mode
                    or queue_filter_mode == "approved_pool"
                )
                try:
                    payload = json.loads(rpath.read_text(encoding="utf-8"))
                    requeued = bool(payload.get("requeued"))
                    if not allow_approved and not requeued:
                        if (
                            is_frame_closed(payload)
                            or is_in_pool
                        ):
                            continue
                    weak = [b for b in payload.get("weak", []) if isinstance(b, dict)]
                    trusted = [b for b in payload.get("trusted", []) if isinstance(b, dict)]
                    if _reconcile_txt is not None:
                        # Aşama 1 JSON dalıyla aynı kural (K-03): TXT
                        # kutuları bellek içi trusted el etiketi olur;
                        # okuma yolunda review JSON yazılmaz (F-0772).
                        _lbl2 = labels_root / split / f"{stem}.txt"
                        if _lbl2.is_file():
                            trusted, weak = _reconcile_txt(trusted, weak, _lbl2)
                            _stamp_txt_hand_names(trusted)
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
                item["conf"] = YOLOFactoryDialog._frame_conf(item)
                items.append(item)

        items.sort(key=lambda x: -float(x.get("mtime", 0.0)))
        return items

    def _report_quarantine_events(self, events: Optional[List[Dict[str, Any]]]) -> None:
        """Taramada karantinaya alınan bozuk dosyaları kullanıcıya bildir.

        UI thread'de çağrılır (worker thread'den değil — ``_append_log``
        QWidget'e yazar). Olaylar ayrıca ``logger.warning`` ile logdadır.
        """
        for ev in list(events or []):
            dest = ev.get("quarantined_to")
            if dest:
                self._append_log(
                    f"[KARANTİNA] #{ev.get('stem')}: bozuk {ev.get('kind')} karantinaya alındı → {dest} "
                    f"(kaynak: {ev.get('file')}; {ev.get('reason')})"
                )
            else:
                self._append_log(
                    f"[KARANTİNA] #{ev.get('stem')}: bozuk {ev.get('kind')} karantinaya ALINAMADI — dosya yerinde duruyor, silinmedi "
                    f"({ev.get('file')}; {ev.get('reason')})"
                )

    def _reload_review_queue(self) -> None:
        """Disk review JSON ve images/{train,val} taraması (senkron).

        Yalnızca anında tutarlılık gereken iç akışlar (onay/red/budama)
        kullanır; açılış, adım geçişi ve Yenile butonu async
        `_request_queue_reload()` yolunu kullanır.
        """
        quarantine_events: List[Dict[str, Any]] = []
        items = self._scan_review_queue_items(
            self._dataset_dir(),
            self._load_rejected_stems(),
            None,
            bool(self.__dict__.get("_review_audit_mode", False)),
            str(self.__dict__.get("_queue_filter_mode", "all") or "all"),
            quarantine_events=quarantine_events,
        )
        self._apply_review_queue_items(items)
        self._report_quarantine_events(quarantine_events)

    def _apply_review_queue_items(
        self, items: List[Dict[str, Any]], preserve_stem: Optional[str] = None
    ) -> None:
        """Taranmış kuyruk verisini UI thread'de uygula (liste + önizleme)."""
        self._review_items = list(items)
        if preserve_stem:
            restored = False
            for idx, it in enumerate(self._review_items):
                if str(it.get("stem", "")) == preserve_stem:
                    self._review_index = idx
                    restored = True
                    break
            if not restored:
                self._review_index = 0
        else:
            self._review_index = 0
        self._review_decode_cache.clear()
        # __dict__ ile oku: __new__ ile kurulan (super().__init__() görmemiş)
        # harness'larda getattr sip __getattr__'ına düşüp RuntimeError verir.
        thumb_thread = self.__dict__.get("_thumb_thread")
        if thumb_thread is not None:
            thumb_thread.reset()
        self._sync_pending_from_queue()
        self._render_queue_list()
        self._render_review_frame()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()

    # ------------------------------------------------------------------
    # [YOLO-ASYNC] Arka plan yükleme (QThread) + loading katmanı
    # ------------------------------------------------------------------
    def _request_queue_reload(self, preserve_stem: Optional[str] = None) -> bool:
        """Kuyruk yenilemeyi worker thread'de iste; yükleme sürüyorsa düşer.

        Çalışan yükleme zaten kuyruğu kapsar (disk o yükleme başladıktan
        sonra değişmediyse veri aynıdır); üst üste tarama yapılmaz.
        """
        if getattr(self, "_yolo_load_running", False):
            logger.debug("[YOLO] kuyruk yenileme isteği düştü: yükleme sürüyor")
            return False
        return self._start_yolo_data_load(initial=False, preserve_stem=preserve_stem)

    def _start_yolo_data_load(
        self, initial: bool = False, preserve_stem: Optional[str] = None
    ) -> bool:
        """Dataset istatistikleri + kuyruk taramasını worker thread'de başlat."""
        if getattr(self, "_yolo_load_running", False):
            return False
        try:
            dataset_dir = self._dataset_dir()
            rejected = self._load_rejected_stems()
            helper = self._get_builtin_helper()
        except Exception as exc:
            logger.warning("[YOLO] yükleme hazırlığı başarısız: %s", exc)
            self._on_yolo_data_load_failed(str(exc))
            return False
        self._yolo_load_running = True
        self._yolo_load_preserve_stem = preserve_stem
        self._yolo_load_started_at = time.monotonic()
        if initial:
            # İlerleme logunu worker'ın ilk progress sinyali yazar
            # (çift satır olmasın diye burada loglanmaz).
            self._set_yolo_loading_detail("dataset taranıyor…")
        else:
            logger.info("[YOLO] yükleniyor: onay kuyruğu yenileniyor…")
            self._show_yolo_loading("Onay kuyruğu yenileniyor…")
        worker = _YoloDataLoadWorker(
            dataset_dir=dataset_dir,
            rejected_stems=rejected,
            review_audit_mode=bool(self.__dict__.get("_review_audit_mode", False)),
            queue_filter_mode=str(self.__dict__.get("_queue_filter_mode", "all") or "all"),
            helper=helper,
        )
        thread = QThread(self)
        # [YOLO-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(thread)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_yolo_load_progress)
        worker.finished.connect(self._on_yolo_data_loaded)
        worker.failed.connect(self._on_yolo_data_load_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # finished sonrası C++ nesnesi silinir; sarkan wrapper'a
        # closeEvent'ten dokunulmaması için referansı temizle.
        thread.finished.connect(self._clear_yolo_load_thread)
        self._yolo_load_thread = thread
        self._yolo_load_worker = worker
        try:
            thread.start()
        except Exception as exc:
            self._on_yolo_data_load_failed(str(exc))
            return False
        return True

    def _clear_yolo_load_thread(self) -> None:
        self._yolo_load_thread = None
        self._yolo_load_worker = None

    def _stop_yolo_threads(self) -> None:
        """Tüm YOLO arka plan thread'lerini TEK düzende durdur (P2-18).

        closeEvent, `__del__` ve atexit bu yöntemi / aynı
        `_stop_qthread_gracefully` desenini kullanır: her thread'de
        `stop()` + `quit()` + sonucu kontrol edilen `wait()`; durmayana
        son çare `terminate()` + log (main.py:2851-2861 deseni).
        Kapsam: eğitim + bootstrap + küçük resim + yükleme + toplu onay
        + zehir temizliği. Zaten durmuş thread'de ucuz no-op'tur, bu
        yüzden closeEvent sonrası `__del__`/atexit tekrarı güvenlidir.
        Bir worker'ın durdurulamaması diğerlerini engellemez (her biri
        kendi try/except'inde). __dict__ ile okunur — C++ nesnesi
        silinmişse bile patlamaz.
        """
        # Eğitim worker'ı (RTXTrainingWorker, QThread alt sınıfı):
        # stop() epoch sınırında durur + alttaki trainer'a stop gönderir.
        training = self.__dict__.get("_training_worker")
        if training is not None:
            try:
                _stop_qthread_gracefully(
                    training, "eğitim worker'ı", _YOLO_TRAIN_STOP_TIMEOUT_MS
                )
            except Exception:
                pass
        # Bootstrap worker'ı (_BackgroundCallWorker, QThread alt sınıfı).
        bootstrap = self.__dict__.get("_bootstrap_worker")
        if bootstrap is not None:
            try:
                _stop_qthread_gracefully(
                    bootstrap, "bootstrap worker'ı", _YOLO_THREAD_STOP_TIMEOUT_MS
                )
            except Exception:
                pass
        thumb_thread = self.__dict__.get("_thumb_thread")
        if thumb_thread is not None:
            try:
                _stop_qthread_gracefully(
                    thumb_thread, "küçük resim thread'i", _YOLO_THUMB_STOP_TIMEOUT_MS
                )
            except Exception:
                pass
        load_thread = self.__dict__.get("_yolo_load_thread")
        if load_thread is not None:
            try:
                _stop_qthread_gracefully(
                    load_thread,
                    "yükleme thread'i",
                    _YOLO_THREAD_STOP_TIMEOUT_MS,
                    worker=self.__dict__.get("_yolo_load_worker"),
                )
            except Exception:
                pass
        # [YOLO-BULK-ASYNC] Toplu onay / zehir temizliği / budama / turbo
        # thread'leri: önce işbirlikçi durdurma isteği (worker kare
        # sınırında durur), sonra quit+wait — çalışan QThread yok
        # edilmeden bitmelidir.
        for _key in ("_bulk_approve", "_purge", "_bulk_prune", "_turbo_relabel"):
            worker = self.__dict__.get(f"{_key}_worker")
            thread = self.__dict__.get(f"{_key}_thread")
            if thread is not None:
                try:
                    _stop_qthread_gracefully(
                        thread,
                        f"{_key} thread'i",
                        _YOLO_THREAD_STOP_TIMEOUT_MS,
                        worker=worker,
                    )
                except Exception:
                    pass

    def wait_for_yolo_load(self, timeout_ms: int = 10000) -> bool:
        """Async kuyruk yüklemesi bitene kadar olay döngüsünü çalıştır.

        Test yardımcısı: `_go_step(2)` / Yenile sonrası kuyruk worker'da
        dolar; senkron okuyan testler bu yöntemle yüklemenin bitmesini
        bekler. Yükleme yoksa hemen True döner; zaman aşımında False.
        Üretim akışını değiştirmez.
        """
        if not self.__dict__.get("_yolo_load_running", False):
            return True
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(25)
        poll.timeout.connect(
            lambda: loop.quit()
            if not self.__dict__.get("_yolo_load_running", False)
            else None
        )
        poll.start()
        QTimer.singleShot(int(timeout_ms), loop.quit)
        loop.exec()
        poll.stop()
        return not self.__dict__.get("_yolo_load_running", False)

    def _on_yolo_load_progress(self, text: str) -> None:
        if getattr(self, "_yolo_closing", False):
            return
        logger.info("[YOLO] yükleniyor: %s", text)
        self._set_yolo_loading_detail(str(text))

    def _on_yolo_data_loaded(
        self, disk_stats: Dict[str, Any], items: List[Dict[str, Any]]
    ) -> None:
        if getattr(self, "_yolo_closing", False):
            self._yolo_load_running = False
            return
        started = getattr(self, "_yolo_load_started_at", 0.0) or time.monotonic()
        self._yolo_load_running = False
        # P1-16 (K-17): worker'ın karantina olayları thread bitmeden okunur
        # (thread.finished → _clear_yolo_load_thread bu slottan sonra çalışır).
        quarantine_events = list(
            getattr(self.__dict__.get("_yolo_load_worker"), "quarantine_events", []) or []
        )
        try:
            preserve = self._yolo_load_preserve_stem
            self._yolo_load_preserve_stem = None
            # P2-20 (F051-K2): yükleme-biti GPU probe'u worker'a taşındı.
            # Worker `get_dataset_stats()` içinde zaten
            # `probe_gpu_environment()` çalıştırdı (nvidia-smi + import
            # denemeleri); sonucu gerçek controller'ın GPU önbelleğine
            # tohumla ki aşağıdaki `_collect_live_stats(force=True)` →
            # `get_factory_status()` → `_get_gpu_info_cached()` UI
            # thread'de aynı probe'u tekrar etmesin. Sonraki dashboard
            # yenilemeleri de bu tohumlanmış bilgiyi kullanır.
            gpu_env = (disk_stats or {}).get("gpu_environment")
            if isinstance(gpu_env, dict) and gpu_env:
                self._last_worker_gpu_env = dict(gpu_env)
                seed = getattr(getattr(self, "factory_controller", None), "seed_gpu_info", None)
                if callable(seed):
                    try:
                        seed(dict(gpu_env))
                    except Exception:
                        pass
            # Worker'ın disk istatistiğini normal birleştirme yoluna ver
            # (canlı controller istatistikleri UI thread'de eklenir).
            self._stats_cache = None
            self._stats_cache_at = 0.0
            stats = self._collect_live_stats(force=True, disk_stats=disk_stats or {})
            self._apply_review_queue_items(items, preserve_stem=preserve)
            self.refresh_dashboard(force=True, _skip_queue_reload=True)
            self._hide_yolo_loading()
            total = int(stats.get("total_frames", 0))
            elapsed = time.monotonic() - started
            logger.info(
                "[YOLO] yüklendi: %d kare kuyruk, %d kare dataset (%.1f sn)",
                len(items),
                total,
                elapsed,
            )
            self._append_log(
                f"[YOLO] Yüklendi: {len(items)} kare kuyruk, {total} kare dataset."
            )
            self._report_quarantine_events(quarantine_events)
        except Exception as exc:
            logger.warning("[YOLO] yükleme sonucu uygulanamadı: %s", exc, exc_info=True)
            self._hide_yolo_loading()
            self._append_log(f"[YOLO] Yükleme hatası: {exc}")

    def _on_yolo_data_load_failed(self, message: str) -> None:
        self._yolo_load_running = False
        if getattr(self, "_yolo_closing", False):
            return
        logger.warning("[YOLO] yükleme başarısız: %s", message)
        self._hide_yolo_loading()
        self._append_log(f"[YOLO] Yükleme hatası: {message}")
        # Son çare: senkron yükleme (yalnızca hata yolunda; sayfa boş kalmasın).
        try:
            self._reload_review_queue()
        except Exception as exc:
            logger.warning("[YOLO] senkron geri dönüş de başarısız: %s", exc)

    def _build_yolo_loading_overlay(self) -> None:
        overlay = QFrame(self)
        overlay.setObjectName("YoloLoadingOverlay")
        overlay.setStyleSheet(
            "QFrame#YoloLoadingOverlay { background-color: rgba(14, 16, 19, 235); border: none; }"
        )
        lay = QVBoxLayout(overlay)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("YOLO yükleniyor…", overlay)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:22px; font-weight:800; color:#F2F3F5; background:transparent; border:none;"
        )
        self._yolo_loading_detail = QLabel("", overlay)
        self._yolo_loading_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._yolo_loading_detail.setStyleSheet(
            "font-size:14px; color:#94a3b8; background:transparent; border:none;"
        )
        bar = QProgressBar(overlay)
        bar.setRange(0, 0)  # busy indicator
        bar.setFixedWidth(280)
        bar.setTextVisible(False)
        lay.addWidget(title)
        lay.addWidget(self._yolo_loading_detail)
        lay.addWidget(bar, alignment=Qt.AlignmentFlag.AlignCenter)
        overlay.hide()
        self._yolo_loading_overlay = overlay
        self._yolo_loading_title = title
        self._yolo_loading_bar = bar
        self._sync_yolo_loading_geometry()

    def _sync_yolo_loading_geometry(self) -> None:
        overlay = getattr(self, "_yolo_loading_overlay", None)
        if overlay is None:
            return
        overlay.setGeometry(0, 0, max(1, self.width()), max(1, self.height()))
        if overlay.isVisible():
            overlay.raise_()

    def _show_yolo_loading(self, detail: str = "", title: Optional[str] = None) -> None:
        overlay = getattr(self, "_yolo_loading_overlay", None)
        if overlay is None:
            return
        if title is not None:
            title_lbl = getattr(self, "_yolo_loading_title", None)
            if title_lbl is not None:
                title_lbl.setText(str(title))
        # Belirli ilerleme yoksa meşgul göstergesine dön (yeni işin
        # eski dolu çubuğuyla başlamaması için).
        bar = getattr(self, "_yolo_loading_bar", None)
        if bar is not None:
            bar.setRange(0, 0)
            bar.setValue(0)
        self._set_yolo_loading_detail(detail)
        self._sync_yolo_loading_geometry()
        overlay.show()
        overlay.raise_()

    def _set_yolo_loading_progress(self, done: int, total: int) -> None:
        """Loading çubuğunu belirli ilerlemeye geçir (done/total).

        Sayaç metni detail label'dadır ("132/1180"); çubuk yalnızca
        doluluk gösterir. total <= 0 ise meşgul göstergesine döner.
        """
        bar = getattr(self, "_yolo_loading_bar", None)
        if bar is None:
            return
        if total and total > 0:
            bar.setRange(0, int(total))
            bar.setValue(max(0, min(int(done), int(total))))
        else:
            bar.setRange(0, 0)
            bar.setValue(0)

    def _hide_yolo_loading(self) -> None:
        overlay = getattr(self, "_yolo_loading_overlay", None)
        if overlay is not None:
            overlay.hide()
        bar = getattr(self, "_yolo_loading_bar", None)
        if bar is not None:
            bar.setRange(0, 0)
            bar.setValue(0)
        title_lbl = getattr(self, "_yolo_loading_title", None)
        if title_lbl is not None:
            title_lbl.setText("YOLO yükleniyor…")

    def _set_yolo_loading_detail(self, text: str) -> None:
        lbl = getattr(self, "_yolo_loading_detail", None)
        if lbl is not None:
            lbl.setText(str(text or ""))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_yolo_loading_geometry()

    # ------------------------------------------------------------------
    # [P2-19/F044-02] Odak/modalite KARARI: dialog açıkken bot durur.
    #
    # Gerekçe: Interception HID enjeksiyonu OS seviyesinde gerçek tuş
    # gibi görünür — Qt, botun R basışını insanın R basışından ayırt
    # edemez, dialog tarafında "botun tuşunu yoksay" ayrımı yapılamaz.
    # Modal yapmak da çözmez: modal dialog odağı üstüne çeker, enjekte
    # R yine dialoga gider (daha da kötü). Mevcut Alt-Tab auto-pause da
    # kurtarmıyor: `is_game_window_focused(allow_gatherion_ui=True)`
    # kendi sürecindeki (own-PID) tüm pencereleri "Gatherion UI" sayılıp
    # odaklı sayıyor (`core/dpi_focus_guard.py`), fabrika dialogu aynı
    # süreçte olduğu için bot durmuyor. Tek gerçek çözüm: dialog
    # görünürken botu `pause()` ile durdurmak, gizlenince/kapanınca
    # yalnızca dialogun duraklattığı botu `resume()` ile devam ettirmek.
    # Kancalar main.py (`_connect_yolo_factory_signals`) tarafından
    # bağlanır; bağlanmadıysa (test/yalnız kullanım) hiçbir şey yapılmaz.
    # ------------------------------------------------------------------
    def set_bot_pause_hooks(
        self,
        pause_fn: Optional[Any] = None,
        resume_fn: Optional[Any] = None,
        is_paused_fn: Optional[Any] = None,
    ) -> None:
        """Bot duraklat/devam kancalarını bağla (main.py çağırır).

        `pause_fn`/`resume_fn` BotWorker.pause/resume, `is_paused_fn`
        BotWorker.is_paused karşılığıdır. Kancalar çağrılabilir değilse
        yoksayılır — dialog kancasız da çalışır.
        """
        self._bot_pause_fn = pause_fn if callable(pause_fn) else None
        self._bot_resume_fn = resume_fn if callable(resume_fn) else None
        self._bot_is_paused_fn = is_paused_fn if callable(is_paused_fn) else None

    def _pause_bot_for_factory(self) -> None:
        """Dialog görünür oldu: bot çalışıyorsa duraklat (F044-02)."""
        if getattr(self, "_bot_paused_by_factory", False):
            return
        pause_fn = getattr(self, "_bot_pause_fn", None)
        if not callable(pause_fn):
            return
        is_paused_fn = getattr(self, "_bot_is_paused_fn", None)
        try:
            if callable(is_paused_fn) and bool(is_paused_fn()):
                # Bot zaten duraklatılmış (kullanıcı / Alt-Tab) — biz
                # duraklatmadık, kapanışta devam da ettirmeyiz.
                return
        except Exception:
            pass
        try:
            pause_fn()
        except Exception as exc:
            logger.debug("Fabrika bot-pause kancası çalışmadı: %s", exc)
            return
        self._bot_paused_by_factory = True
        try:
            self._append_log("[BOT] YOLO Fabrikası açık — bot duraklatıldı (kapanınca devam eder).")
        except Exception:
            pass

    def _resume_bot_if_paused_by_factory(self) -> None:
        """Dialog gizlendi/kapandı: yalnızca bizim duraklattığımız botu devam ettir."""
        if not getattr(self, "_bot_paused_by_factory", False):
            return
        self._bot_paused_by_factory = False
        resume_fn = getattr(self, "_bot_resume_fn", None)
        if not callable(resume_fn):
            return
        is_paused_fn = getattr(self, "_bot_is_paused_fn", None)
        try:
            if callable(is_paused_fn) and not bool(is_paused_fn()):
                # Kullanıcı dialog açıkken botu kendisi devam ettirmiş —
                # ikinci resume ile gereksiz "Resumed" logu basılmaz.
                return
        except Exception:
            pass
        try:
            resume_fn()
        except Exception as exc:
            logger.debug("Fabrika bot-resume kancası çalışmadı: %s", exc)
            return
        try:
            self._append_log("[BOT] YOLO Fabrikası kapandı — bot devam etti.")
        except Exception:
            pass

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self._pause_bot_for_factory()

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._resume_bot_if_paused_by_factory()
        super().hideEvent(event)

    # NOT: closeEvent bu sınıfta dosya sonunda ayrıca tanımlıdır
    # (oradaki gövdeye `_resume_bot_if_paused_by_factory()` eklenmiştir);
    # buraya ikinci bir closeEvent yazılmaz — sınıf gövdesindeki son
    # tanım öncekinin üstünü yazar.

    # ------------------------------------------------------------------
    # [YOLO-ASYNC] Görünür satır küçük resimleri (worker thread)
    # ------------------------------------------------------------------
    def _wire_queue_thumb_requests(self) -> None:
        try:
            self.queue_list.verticalScrollBar().valueChanged.connect(
                lambda _v: self._schedule_visible_thumbs()
            )
        except Exception:
            logger.debug("kuyruk kaydırma sinyali bağlanamadı", exc_info=True)

    def _ensure_thumb_thread(self) -> Any:
        thread = getattr(self, "_thumb_thread", None)
        if thread is not None and thread.isRunning():
            return thread
        thread = _QueueThumbThread(self._dataset_dir())
        thread.thumb_ready.connect(self._on_thumb_ready)
        thread.start()
        self._thumb_thread = thread
        return thread

    def _schedule_visible_thumbs(self) -> None:
        QTimer.singleShot(120, self._request_visible_thumbs)

    def _visible_queue_rows(self) -> Tuple[int, int]:
        qlist = self.queue_list
        count = qlist.count()
        if count == 0:
            return (0, -1)
        vp = qlist.viewport()
        top = qlist.indexAt(QPoint(0, 0))
        bottom = qlist.indexAt(QPoint(0, max(0, vp.height() - 1)))
        first = top.row() if top.isValid() else 0
        last = bottom.row() if bottom.isValid() else min(count - 1, first + 15)
        if last < first:
            last = min(count - 1, first + 15)
        return (first, last)

    def _request_visible_thumbs(self) -> None:
        qlist = getattr(self, "queue_list", None)
        if qlist is None or qlist.count() == 0:
            return
        if getattr(self, "_yolo_closing", False):
            return
        first, last = self._visible_queue_rows()
        if last < first:
            return
        thread = self._ensure_thumb_thread()
        # P2-20: `or {}` boş LRU'yu düz dict ile değiştirirdi (LRU boşken
        # falsy'dir); yalnızca None ise kur.
        cache = getattr(self, "_queue_thumb_cache", None)
        if cache is None:
            cache = self._queue_thumb_cache = _ThumbPixmapLruCache()
        for row in range(max(0, first - 2), min(qlist.count(), last + 21)):
            item = qlist.item(row)
            if item is None:
                continue
            real_i = item.data(int(Qt.ItemDataRole.UserRole))
            try:
                real_i = int(real_i)
            except (TypeError, ValueError):
                continue
            if real_i < 0 or real_i >= len(self._review_items):
                continue
            it = self._review_items[real_i]
            key = (str(it.get("split", "train")), str(it.get("stem", "")))
            if key in cache:
                continue
            thread.request(key[0], key[1])

    def _on_thumb_ready(self, split: str, stem: str, image: QImage) -> None:
        if getattr(self, "_yolo_closing", False):
            return
        if image is None or image.isNull():
            return
        key = (str(split), str(stem))
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            return
        cache = getattr(self, "_queue_thumb_cache", None)
        if cache is None:
            cache = self._queue_thumb_cache = _ThumbPixmapLruCache()
        elif not isinstance(cache, _ThumbPixmapLruCache):
            # Eski/harness yapımı düz dict — gerçek LRU'ya çevir.
            cache = self._queue_thumb_cache = _coerce_thumb_cache(cache)
        # P2-20 (F050-K2): `len > 300: clear()` kalktı — ekleme en az
        # kullanılan girdiyi tek tek atar (`_ThumbPixmapLruCache`).
        cache[key] = pix
        labels = getattr(self, "_thumb_labels", None) or {}
        lbl = labels.get(f"{split}|{stem}")
        if lbl is not None:
            lbl.setPixmap(
                pix.scaled(72, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            lbl.setText("")

    def _bulk_threshold(self) -> float:
        return float(self.bulk_range.value()) / 100.0

    def _emin_badge_threshold(self) -> float:
        """Rozet "Emin" eşiği = toplu onay eşiği (P1-12/K-31, F014-04).

        Eskiden rozet 0.85, toplu onay 0.90 bakıyordu: 0.85–0.89 kutu
        rozette Emin, onayda eşik altı kalıyordu. Artık ikisi de
        bulk_range kaydırıcısını okur; kaydırıcı yokken (başsız test)
        yedek CONF_BULK_DEFAULT'tur — toplu onay yedeğiyle aynı değer.
        """
        return self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT

    def _on_bulk_threshold_changed(self, value: int) -> None:
        self.bulk_val.setText(str(int(value)))
        if getattr(self, "_queue_filter_mode", "all") in ("below", "above"):
            self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_bulk_threshold_released(self) -> None:
        """Kaydırıcı bırakılınca rozet eşiği değiştiği için mevcut kare
        yeniden çizilir (P1-12/K-31: rozet = toplu onay eşiği). valueChanged
        sırasında çizilmez — sürüklerken her tıkta yeniden çizim pahalıdır."""
        try:
            if getattr(self, "_review_items", None):
                self._render_review_frame()
        except Exception:
            pass

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

    def _on_queue_conf_changed(self, value: int) -> None:
        """Kuyruk güven kaydırıcısı (prototip qConf): minimum güven filtresi."""
        self._queue_min_conf = max(0.0, min(0.95, float(value) / 100.0))
        if hasattr(self, "queue_conf_val"):
            self.queue_conf_val.setText(f"≥ {self._queue_min_conf:.2f}".replace(".", ","))
        self._render_queue_list()
        self._update_filter_action_buttons()

    def _on_clear_filters(self) -> None:
        """Prototip 'Filtreleri temizle': arama + durum + sınıf + güven sıfırlanır."""
        if hasattr(self, "review_search_edit"):
            self.review_search_edit.clear()
        if hasattr(self, "queue_filter"):
            idx = self.queue_filter.findData("all")
            if idx >= 0:
                self.queue_filter.setCurrentIndex(idx)
        if hasattr(self, "review_class_filter"):
            self.review_class_filter.setCurrentIndex(0)
        if hasattr(self, "queue_conf_slider"):
            self.queue_conf_slider.setValue(0)
        self._queue_min_conf = 0.0
        self._queue_filter_mode = "all"
        self._render_queue_list()
        self._update_filter_action_buttons()
        self._append_log("[FİLTRE] Filtreler temizlendi — kuyruk hepsine döndü.")

    def _on_only_approved_toggled(self, checked: bool) -> None:
        """'Sadece onaylananları eğit' anahtarı: eğitim/freeze bu bayrağı
        _only_approved_enabled() üzerinden okur; burada durum loglanır ve
        eğitim kilidi tazelenir."""
        self._append_log(
            "[AYAR] Sadece onaylananları eğit: " + ("AÇIK" if checked else "KAPALI")
        )
        try:
            self._update_train_lock()
        except Exception:
            pass

    def _build_class_legend(self) -> None:
        """Fotoğraf altı renk lejantı (prototip classLegend): her sınıf bir
        renk düğmesi; tıklayınca sınıf filtresi uygulanır."""
        bar = QWidget(self)
        bar_l = QHBoxLayout(bar)
        bar_l.setContentsMargins(4, 4, 4, 4)
        bar_l.setSpacing(6)
        all_btn = QPushButton("Tümü", self)
        all_btn.setObjectName("PipeLegendBtn")
        all_btn.setStyleSheet("font-size:11px; padding:3px 10px;")
        all_btn.clicked.connect(lambda: self._on_legend_class_clicked(-1))
        bar_l.addWidget(all_btn)
        self._legend_buttons = {-1: all_btn}
        for cid in sorted(CLASS_ID_TO_SPEC):
            spec = CLASS_ID_TO_SPEC[cid]
            cname = str(spec.get("key", f"class_{cid}"))
            color = CLASS_COLORS_MAP.get(cid, "#22c55e")
            btn = QPushButton(f"■ {cname}", self)
            btn.setObjectName("PipeLegendBtn")
            btn.setToolTip(f"{cname} — tıklayınca kuyruğu bu sınıfa süzer")
            btn.setStyleSheet(
                f"font-size:11px; color:{color}; padding:3px 8px; "
                "border:1px solid #1e293b; border-radius:6px; background:#0B0E12;"
            )
            btn.clicked.connect(lambda _=False, c=cid: self._on_legend_class_clicked(c))
            bar_l.addWidget(btn)
            self._legend_buttons[cid] = btn
        bar_l.addStretch()
        scroll = QScrollArea(self)
        scroll.setObjectName("PipeLegendScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(34)
        scroll.setWidget(bar)
        self.class_legend_bar = bar
        self.class_legend_scroll = scroll
        # Lejant yüksekliği: yatay kaydırma çubuğu görünürken çubuk payı
        # eklenir — yoksa viewport 34-14=20 px'e iner ve sınıf çipleri
        # alttan kırpılır. rangeChanged hem görünme hem kaybolma anında
        # tetiklenir.
        scroll.horizontalScrollBar().rangeChanged.connect(
            lambda _mn, _mx: self._update_legend_scroll_height()
        )
        self._update_legend_scroll_height()

    def _update_legend_scroll_height(self) -> None:
        """Lejant şeridinin sabit yüksekliğini içeriğe uydur: yatay
        kaydırma çubuğu görünürken çubuk yüksekliği eklenir, görünmezken
        yalnızca çubuk yüksekliği kullanılır. Böylece sınıf çipleri
        hiçbir pencere genişliğinde alttan kırpılmaz."""
        scroll = getattr(self, "class_legend_scroll", None)
        bar = getattr(self, "class_legend_bar", None)
        if scroll is None or bar is None:
            return
        bar_h = bar.sizeHint().height()
        hbar = scroll.horizontalScrollBar()
        sb_h = hbar.sizeHint().height() if hbar.maximum() > 0 else 0
        scroll.setFixedHeight(bar_h + sb_h)

    def _on_legend_class_clicked(self, cid: int) -> None:
        """Lejant düğmesi: sınıf filtresini uygular; aynı sınıfa tekrar
        tıklayınca filtre sıfırlanır (Tümü)."""
        if not hasattr(self, "review_class_filter"):
            return
        try:
            current = int(self.review_class_filter.currentData())
        except (TypeError, ValueError):
            current = -1
        target = -1 if current == cid else cid
        idx = self.review_class_filter.findData(target)
        if idx >= 0:
            self.review_class_filter.setCurrentIndex(idx)
        else:
            self._on_class_filter_changed()
        self._refresh_legend_active()

    def _refresh_legend_active(self) -> None:
        """Aktif sınıf filtresini lejantta vurgula."""
        buttons = getattr(self, "_legend_buttons", {}) or {}
        if not buttons:
            return
        try:
            current = int(self.review_class_filter.currentData())
        except (TypeError, ValueError, AttributeError):
            current = -1
        for cid, btn in buttons.items():
            color = "#fbbf24" if cid == -1 else CLASS_COLORS_MAP.get(cid, "#22c55e")
            if cid == current:
                btn.setStyleSheet(
                    f"font-size:11px; font-weight:800; color:{color}; padding:3px 8px; "
                    "border:2px solid #38bdf8; border-radius:6px; background:#0f172a;"
                )
            else:
                btn.setStyleSheet(
                    f"font-size:11px; color:{color}; padding:3px 8px; "
                    "border:1px solid #1e293b; border-radius:6px; background:#0B0E12;"
                )


    def _create_queue_item_widget(self, it: Dict[str, Any]) -> QWidget:
        """Okunur kuyruk satırı: küçük resim + kare no + güven + kutu sayısı +
        saat/sınıf/kutu satırı — kesilmeden (uzun stem iki satıra sarılır)."""
        conf = self._frame_conf(it)
        it["conf"] = conf
        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])
        n_weak = len(weak)
        n_trusted = len(trusted)
        n_total = n_weak + n_trusted
        stem = str(it.get("stem", ""))
        done = it.get("done")
        status = ""
        if done == "ok":
            status = " · ✓ onaylandı"
        elif done == "no":
            status = " · ✕ reddedildi"

        if n_total == 0:
            conf_str = "0 Kutu (Boş)"
        else:
            confs: List[float] = []
            for b in weak + trusted:
                try:
                    c = float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
                    confs.append(c if c <= 1.0 else c / 100.0)
                except (TypeError, ValueError):
                    pass
            if confs:
                mn = int(round(min(confs) * 100))
                mx = int(round(max(confs) * 100))
                conf_str = f"%{mn}" if mn == mx else f"%{mn}-%{mx}"
            else:
                conf_str = f"%{int(round(conf * 100))}"

        # Saat (dosya mtime) + sınıf/kutu özeti
        mtime = float(it.get("mtime", 0.0) or 0.0)
        time_str = datetime.fromtimestamp(mtime).strftime("%H:%M") if mtime > 0 else "—"
        if n_total == 0:
            class_str = "Etiketsiz / boş kare"
        else:
            counts: Dict[str, int] = {}
            for b in weak + trusted:
                try:
                    cid = int(b.get("class_id", -1))
                except (TypeError, ValueError):
                    cid = -1
                cname = CLASS_ID_TO_SPEC.get(cid, {}).get("key") or str(b.get("class_name", f"class_{cid}"))
                counts[str(cname)] = counts.get(str(cname), 0) + 1
            class_str = " · ".join(f"{k}×{v}" for k, v in sorted(counts.items()))

        row = QWidget(self)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)
        thumb = QLabel(row)
        thumb.setObjectName("queue_thumb")
        thumb.setFixedSize(72, 54)
        thumb.setStyleSheet("background:#0B0E12; border:1px solid #1e293b; border-radius:4px;")
        # [YOLO-ASYNC] Küçük resim artık satır kurulumunda decode edilmez
        # (417 kare x tam çözünürlük decode = 30+ sn UI donması). Önbellek
        # isabeti varsa anında basılır, yoksa yer tutucu kalır; görünür
        # satırların küçük resmi worker thread'de üretilir (QImage).
        split = str(it.get("split", "train"))
        thumb_key = f"{split}|{stem}"
        thumb.setProperty("thumb_key", thumb_key)
        labels = getattr(self, "_thumb_labels", None)
        if labels is not None:
            labels[thumb_key] = thumb
        pix = self._queue_thumb_cache.get((split, stem))
        if pix is not None and not pix.isNull():
            thumb.setPixmap(
                pix.scaled(72, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            thumb.setText("…")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet("background:#0B0E12; border:1px solid #1e293b; border-radius:4px; color:#64748b;")
        lay.addWidget(thumb)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        l1 = QLabel(f"#{stem}", row)
        l1.setWordWrap(True)
        l1.setStyleSheet("font-size:11.5px; font-weight:700; color:#F2F3F5; background:transparent;")
        text_col.addWidget(l1)
        l2 = QLabel(f"{conf_str} · {n_total} kutu ({n_trusted} Emin, {n_weak} Şüpheli){status}", row)
        l2.setWordWrap(True)
        l2.setStyleSheet("font-size:11px; color:#cbd5e1; background:transparent;")
        text_col.addWidget(l2)
        l3 = QLabel(f"{time_str} · {class_str}", row)
        l3.setWordWrap(True)
        l3.setStyleSheet("font-size:10.5px; color:#94a3b8; background:transparent;")
        text_col.addWidget(l3)
        lay.addLayout(text_col, stretch=1)
        row.setToolTip(stem)
        return row

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
        th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT

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

            # Kuyruk güven kaydırıcısı (qConf): minimum güvenin altındaki kareler
            # durum modundan bağımsız olarak elenir.
            min_conf = float(getattr(self, "_queue_min_conf", 0.0) or 0.0)
            if min_conf > 0.0 and conf < min_conf:
                continue

            # Prototip durum modları: Bekleyen / Onaylanan / Reddedilen /
            # Yüksek güvenli (≥ kaydırıcı eşiği) / Düşük güvenli (< kaydırıcı eşiği).
            # Not: canlı akışta
            # onaylanan/reddedilen kare kuyruktan çıkarıldığı için ok/no
            # listeleri genelde boştur; filtre mantığı sentetik kuyrukla
            # doğrulanır (tests/test_pyqt6_yolo.py).
            if mode == "pending":
                if it.get("done"):
                    continue
            elif mode == "ok":
                if not (it.get("done") == "ok" or it.get("human_approved") is True):
                    continue
            elif mode == "no":
                if it.get("done") != "no":
                    continue
            elif mode == "high":
                # P1-12 (K-31/F014-01): yüksek/düşük kaydırıcı eşiğinde
                # ayrılır — 0.80-0.89 arası kare hiçbir modda kaybolmaz.
                if conf < th:
                    continue
            elif mode == "low":
                if conf >= th:
                    continue
            elif mode == "approved_pool":
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
                # P0-8 (K-14): requeue edilmiş + el etiketli kare
                # suspicious değildir (trusted boş olsa bile).
                if YOLOFactoryDialog._is_requeued_hand_frame(it):
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
            # F-2064 [O3]: 'above' dalında budama butonu "Eşik Üstü" der (copy-paste "Eşik Altı" idi).
            prune_txt = f"✂️ Eşik Üstü Etiketleri Sil · Görseli Koru{tag_suffix} ({n} kare)" if scope_tag else f"✂️ Eşik Üstü Etiketleri Sil · Görseli Koru ({n} kare)"
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
        if hasattr(self, "btn_approve_filtered"):
            # Prototip "Filtredekileri Onayla (n)": sayı her zaman filtrelenen
            # kare sayısıdır; boş filtrede devre dışı kalır.
            self.btn_approve_filtered.setText(f"✓ Filtredekileri Onayla ({n})")
            self.btn_approve_filtered.setEnabled(n > 0)
        self._refresh_legend_active()
        if hasattr(self, "btn_prune_low_confidence"):
            self.btn_prune_low_confidence.setText(prune_txt)
            self.btn_prune_low_confidence.setEnabled(prune_enabled)
        if hasattr(self, "btn_bulk_prune_boxes"):
            self.btn_bulk_prune_boxes.setText(prune_txt)
            self.btn_bulk_prune_boxes.setEnabled(prune_enabled)
        if hasattr(self, "btn_live_relabel_filtered"):
            self.btn_live_relabel_filtered.setText(relabel_txt)
        if hasattr(self, "btn_live_relabel_all"):
            # "Tümünü" butonu filtre kapsamlı metin almaz; tüm kuyruğu sayar (F-0606).
            total_all = len(getattr(self, "_review_items", []) or [])
            self.btn_live_relabel_all.setText(f"🤖 Tümünü Canlı Yeniden Etiketle ({total_all} kare)")

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
        # [YOLO-ASYNC] Parçalı render: gösterilen bir QListWidget'a 456
        # satır eklemek tek seferde ~3 sn UI bloğu yapar (addItem başına
        # ~5 ms). Küçük listeler (<=60 satır — tüm test verisi dahil)
        # eskisi gibi senkron kurulur; büyük listelerde ilk 20 satır
        # senkron, kalanı 20'şer satırlık olay-döngüsü parçalarıyla akar.
        self._queue_render_gen = getattr(self, "_queue_render_gen", 0) + 1
        gen = self._queue_render_gen
        if hasattr(self, "_thumb_labels"):
            self._thumb_labels.clear()
        self._queue_render_frame_index = None
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        indices = self._filtered_review_indices()
        th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        if hasattr(self, "queue_filter_lbl"):
            mode_labels = {
                "all": "hepsi",
                "pending": "bekleyen",
                "ok": "onaylanan",
                "no": "reddedilen",
                "high": "yüksek güvenli",
                "low": "düşük güvenli",
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
            self.queue_filter_lbl.setText(f"Kuyruk · {lbl_name} ({len(indices)} kare)")
        if indices and self._review_index not in indices:
            self._review_index = indices[0]
        self._queue_render_indices = list(indices)
        self._queue_render_pos = 0
        self._queue_render_select_row = 0
        if len(indices) <= _QUEUE_RENDER_SYNC_LIMIT:
            self._render_queue_rows(0, len(indices))
            self._finalize_queue_list_render(gen)
            return
        self._render_queue_rows(0, _QUEUE_RENDER_CHUNK)
        # Önizleme + filtre durumu veriye dayalıdır, liste bitmeden de
        # gösterilebilir: büyük önizleme hemen gelir.
        self._render_review_frame()
        self._queue_render_frame_index = self._review_index
        self._update_filter_action_buttons()
        # İlk parçadan sonra liste tıklanabilir olsun (kalan parçalar
        # aşağıya eklenir; seçim finalize'da güncel indekse göre kurulur).
        self.queue_list.blockSignals(False)
        QTimer.singleShot(0, lambda: self._render_queue_list_chunk(gen))

    def _render_queue_rows(self, start: int, end: int) -> None:
        """_queue_render_indices[start:end] satırlarını listeye ekler."""
        indices = getattr(self, "_queue_render_indices", []) or []
        for list_row in range(start, min(end, len(indices))):
            real_i = indices[list_row]
            it = self._review_items[real_i]
            item = QListWidgetItem(self._queue_item_text(it))
            item.setData(int(Qt.ItemDataRole.UserRole), int(real_i))
            if it.get("done"):
                item.setForeground(QColor("#6B7688"))
            # Okunur satır: küçük resim + kare no + güven + kutu sayısı +
            # saat/sınıf/kutu satırı. Uzun stem iki satıra sarılır.
            widget = self._create_queue_item_widget(it)
            stem_len = len(str(it.get("stem", "")))
            row_h = 112 if stem_len > 28 else 92
            # [YOLO-ASYNC] Satır yüksekliği burada ölçülmez: 456 satırda
            # resize+activate+sizeHint ~3 sn UI bloğu yapıyordu (açılışta
            # UI-WATCHDOG uyarısı). Sezgisel yükseklikle kurulur; gerçek
            # yükseklikler `_refresh_queue_row_heights` ile olay
            # döngüsüne bölünmüş parçalar halinde düzeltilir.
            item.setSizeHint(QSize(200, row_h))
            self.queue_list.addItem(item)
            self.queue_list.setItemWidget(item, widget)
            if real_i == self._review_index:
                self._queue_render_select_row = list_row
        self._queue_render_pos = min(end, len(indices))

    def _render_queue_list_chunk(self, gen: int) -> None:
        if gen != getattr(self, "_queue_render_gen", 0):
            return
        indices = getattr(self, "_queue_render_indices", []) or []
        pos = getattr(self, "_queue_render_pos", 0)
        end = min(len(indices), pos + _QUEUE_RENDER_CHUNK)
        self._render_queue_rows(pos, end)
        if end < len(indices):
            QTimer.singleShot(0, lambda: self._render_queue_list_chunk(gen))
        else:
            self._finalize_queue_list_render(gen)

    def _finalize_queue_list_render(self, gen: int) -> None:
        if gen != getattr(self, "_queue_render_gen", 0):
            return
        indices = getattr(self, "_queue_render_indices", []) or []
        if indices:
            # Parçalı render sürerken kullanıcı başka satır seçmiş
            # olabilir: seçim güncel _review_index'e göre kurulur.
            select_row = 0
            for list_row, real_i in enumerate(indices):
                if real_i == self._review_index:
                    select_row = list_row
                    break
            self.queue_list.blockSignals(True)
            if self.queue_list.currentRow() != select_row:
                self.queue_list.setCurrentRow(select_row)
            self.queue_list.blockSignals(False)
            if getattr(self, "_queue_render_frame_index", None) != self._review_index:
                self._render_review_frame()
        elif self._review_items:
            self.review_title.setText("Filtre boş")
            self.review_sub.setText("Seçili filtre/aramaya uyan kare yok")
            self.review_preview.show_empty_state("Filtre boş")
        self._update_filter_action_buttons()
        self.queue_list.blockSignals(False)
        self._queue_render_indices = []
        self._queue_render_pos = 0
        # Satır yüksekliklerini ilk yerleşimden sonra düzelt: liste
        # henüz yerleşmemişse (pencere açılmadan önce) yükseklikler
        # sezgisel kalır; yerleşim bitince gerçek genişlikte ölçülür.
        # [YOLO-ASYNC] Parçalı çalışır (nesil korumalı): yeni render
        # eski geçişi geçersiz kılar.
        self._queue_heights_dirty = True
        self._queue_heights_gen = getattr(self, "_queue_heights_gen", 0) + 1
        _hgen = self._queue_heights_gen
        QTimer.singleShot(0, lambda: self._refresh_queue_row_heights(0, _hgen))
        # [YOLO-ASYNC] Görünür satırların küçük resimlerini worker'dan iste.
        QTimer.singleShot(120, self._request_visible_thumbs)

    def _refresh_queue_row_heights(self, _start: int = 0, _gen: Optional[int] = None) -> None:
        """Kuyruk satır yüksekliklerini liste genişliğine göre düzelt.

        Sarılan satırlar (uzun stem / sınıf özeti) sabit 92/112 px'e
        sığmazsa satır dikeyde sıkışır ve metin kesilir. Satır widget'ı
        liste genişliğinde yatırılıp sizeHint yüksekliği okunur; yetersiz
        kalan satırlar büyütülür (küçültme yapılmaz — titreme olmasın).

        [YOLO-ASYNC] 456 satırı tek seferde ölçmek ~3 sn UI bloğu
        yapıyordu; o yüzden 25 satırlık parçalar halinde, olay döngüsüne
        dönerek çalışır. Yeni bir render başlarsa eski geçiş nesil
        numarasıyla iptal olur.
        """
        if _gen is not None and _gen != getattr(self, "_queue_heights_gen", 0):
            return
        qlist = getattr(self, "queue_list", None)
        if qlist is None:
            self._queue_heights_pending = False
            return
        avail_w = qlist.viewport().width()
        if avail_w < 200:
            self._queue_heights_pending = False
            return
        if _start == 0:
            measured_w = getattr(self, "_queue_heights_measured_w", -1)
            if not getattr(self, "_queue_heights_dirty", True) and abs(avail_w - measured_w) < 4:
                self._queue_heights_pending = False
                return
        self._queue_heights_pending = True
        end = min(qlist.count(), _start + 25)
        for r in range(_start, end):
            item = qlist.item(r)
            wdg = qlist.itemWidget(item) if item is not None else None
            if item is None or wdg is None:
                continue
            meas_w = wdg.width() if wdg.width() >= 200 else max(200, avail_w - 18)
            wdg.resize(meas_w, wdg.sizeHint().height())
            row_layout = wdg.layout()
            if row_layout is not None:
                row_layout.activate()
            need_h = min(wdg.sizeHint().height() + 24, 240)
            if need_h > item.sizeHint().height():
                item.setSizeHint(QSize(200, need_h))
        if end < qlist.count():
            _gen2 = _gen if _gen is not None else getattr(self, "_queue_heights_gen", 0)
            QTimer.singleShot(0, lambda: self._refresh_queue_row_heights(end, _gen2))
        else:
            self._queue_heights_pending = False
            self._queue_heights_dirty = False
            self._queue_heights_measured_w = avail_w

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
        # F-2000 / R44: Step 2 review scrubbing decodes through the shared
        # LRUFrameDecodeCache (mtime-validated, 32 frames) instead of only a
        # hand-rolled single-entry dict, so scrubbing back/forth across
        # frames no longer re-decodes from disk every time.
        try:
            img = get_lru_frame_cache().get_or_decode(path)
            if img is not None and getattr(img, "size", 0) > 0:
                return img
        except Exception as exc:
            logger.debug("LRU frame cache read failed, falling back: %s", exc)
        # Fallback: previous single-entry behaviour (kept so a cache failure
        # can never break review rendering).
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
        # P1-12 (K-32): ölçülmemiş güven "—" gösterilir, uydurma %0 yok.
        measured_conf = _measured_confidence(det)
        norm_conf = measured_conf if measured_conf is not None else 0.0
        pct_txt = (
            f"%{max(0, min(100, int(round(norm_conf * 100))))}"
            if measured_conf is not None
            else "—"
        )

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel(f"<b>{cname}</b>")
        title.setStyleSheet("background:transparent;color:#F2F3F5;font-size:12px;")
        text.addWidget(title)
        # P1-12 (K-31): rozet "Emin" eşiği = toplu onay eşiği
        # (bulk_range kaydırıcısı); ölçülmemiş güven Emin sayılmaz.
        is_confident = (
            (
                is_trusted
                and measured_conf is not None
                and norm_conf >= self._emin_badge_threshold()
            )
            or bool(det.get("hand_label"))
            or bool(det.get("human_approved"))
        )

        # El etiketi + ölçülmemiş güven (TXT kökenli, conf 0.0): rozette
        # uydurma "%0" yok — el etiketi olduğu yazılır.
        is_hand_unmeasured = bool(det.get("hand_label")) and norm_conf <= 0.0

        if is_confident:
            status_txt = "El etiketi · güven ölçülmedi" if is_hand_unmeasured else f"Emin · {pct_txt} güven"
            status_color = CONFIDENT_STATUS_COLOR_HEX
            conf_style = (
                f"background:#0d2818;color:{CONFIDENT_STATUS_COLOR_HEX};font-family:Consolas;font-size:12px;font-weight:700;"
                "border:1px solid #16a34a;border-radius:6px;padding:3px 8px;"
            )
        elif norm_conf >= SUSPICIOUS_THRESHOLD:
            status_txt = f"Şüpheli · {pct_txt} güven"
            status_color = SUSPICIOUS_STATUS_COLOR_HEX
            conf_style = (
                f"background:#291804;color:{SUSPICIOUS_STATUS_COLOR_HEX};font-family:Consolas;font-size:12px;font-weight:700;"
                "border:1px solid #d97706;border-radius:6px;padding:3px 8px;"
            )
        else:
            # Ölçülmemiş güven (el dışı): yüzde uydurulmaz.
            status_txt = (
                "Güven ölçülmedi (Şüpheli)"
                if measured_conf is None
                else f"Düşük Güven (Şüpheli) · {pct_txt} güven"
            )
            status_color = LOW_CONFIDENCE_STATUS_COLOR_HEX
            conf_style = (
                f"background:#3b1010;color:{LOW_CONFIDENCE_STATUS_COLOR_HEX};font-family:Consolas;font-size:12px;font-weight:700;"
                f"border:1px solid {LOW_CONFIDENCE_COLOR_HEX};border-radius:6px;padding:3px 8px;"
            )

        small = QLabel(status_txt)
        small.setStyleSheet(f"background:transparent;color:{status_color};font-size:11px;")
        text.addWidget(small)
        hl.addLayout(text, stretch=1)

        conf_lbl = QLabel("el" if is_hand_unmeasured else pct_txt)
        conf_lbl.setObjectName("PipeDetConf")
        conf_lbl.setStyleSheet(conf_style)
        hl.addWidget(conf_lbl)

        cls_combo = QComboBox(row)
        cls_combo.setObjectName("PipeClassCombo")
        cls_combo.setEditable(True)
        cls_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        cls_combo.setToolTip("Sınıfı değiştir ve YZ'ye anında öğret")
        cls_combo.setStyleSheet(
            "QComboBox { background:#171B22; color:#F2F3F5; border:1px solid #262D3A; border-radius:6px; padding:4px 8px; font-size:12px; max-width:150px; }"
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

        # Bu ekranda Kinah tutarı gösterilmez (prototip: etiketleme
        # ekranında para değeri yok). _estimate_box_kinah yalnızca
        # ekonomi katmanında kalır, inceleme UI'sında kullanılmaz.

        if not is_trusted:
            ok = QPushButton("Kabul", row)
            ok.setObjectName("PipeGreen")
            ok.setStyleSheet(
                "background:#166534; color:#f0fdf4; border:1px solid #22c55e; "
                "border-radius:5px; padding:5px 10px; font-size:12px; font-weight:600;"
            )
            ok.setToolTip("Bu kutuyu onayla ve Emin grubuna taşı")
            ok.clicked.connect(lambda _=False, d=det: self._accept_box(d))

            remap = QPushButton("→ filiz", row)
            remap.setObjectName("PipeBtn")
            remap.setStyleSheet(
                "background:#1e293b; color:#cbd5e1; border:1px solid #334155; "
                "border-radius:5px; padding:5px 10px; font-size:12px;"
            )
            remap.setToolTip("Yanlış world sınıfını minimap_sprout (9) yap")
            remap.clicked.connect(lambda _=False, d=det: self._remap_box_to_sprout(d))

            no = QPushButton("Sil", row)
            no.setObjectName("PipeDanger")
            no.setStyleSheet(
                "background:#7f1d1d; color:#fef2f2; border:1px solid #ef4444; "
                "border-radius:5px; padding:5px 10px; font-size:12px;"
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
                "border-radius:5px; padding:5px 10px; font-size:12px; font-weight:600;"
            )
            demote.setToolTip("Bu emin kutuyu şüpheli grubuna geri al")
            demote.clicked.connect(lambda _=False, d=det: self._demote_box_to_weak(d))

            no = QPushButton("Sil", row)
            no.setObjectName("PipeDanger")
            no.setStyleSheet(
                "background:#7f1d1d; color:#fef2f2; border:1px solid #ef4444; "
                "border-radius:5px; padding:5px 10px; font-size:12px;"
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

        if draw_boxes:
            self.review_sub.setText(f"{len(trusted)} Emin, {len(weak)} Şüpheli · En düşük güven: %{min_pct}")
        else:
            self.review_sub.setText("0 Kutu · Etiketsiz kare")

        img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
        img: Optional[np.ndarray] = None
        if img_path is None:
            self.review_preview.show_empty_state(f"Görüntü bulunamadı: {it.get('stem')}")
            self._log_once(
                f"miss:{it.get('stem')}",
                f"[ÖNİZLEME] path not found: images/{{train,val}}/{it.get('stem')}.*, "
                f"previews/, game_learning/*/ (dataset={self._dataset_dir()})",
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
        sec1_title.setStyleSheet("color:#fb923c; font-size:13px; font-weight:800; background:transparent;")
        sec1_l.addWidget(sec1_title)
        self.det_layout.addWidget(sec1)

        if not weak:
            empty_weak = QLabel("Bu karede inceleme bekleyen şüpheli etiket yok.")
            empty_weak.setStyleSheet("color:#7E8DA1; font-size:12px; padding:3px 6px; background:transparent;")
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
        sec2_title.setStyleSheet("color:#22c55e; font-size:13px; font-weight:800; background:transparent;")
        sec2_l.addWidget(sec2_title)
        self.det_layout.addWidget(sec2)

        if not trusted:
            empty_trusted = QLabel("Bu karede henüz onaylanmış emin etiket yok.")
            empty_trusted.setStyleSheet("color:#7E8DA1; font-size:12px; padding:3px 6px; background:transparent;")
            self.det_layout.addWidget(empty_trusted)
        else:
            for det in trusted:
                self.det_layout.addWidget(self._create_review_det_row(det, is_trusted=True))

        self.det_layout.addStretch(1)

        # P3-28: çubuk sayıları tek kaynaktan (count_review_state) — P2-21
        # tanımı aynen korunur (reviewed kümülatif `_pipeline["reviewed"]`,
        # payda reviewed + pending); artık web `reviewProgTxt` ile aynı
        # fonksiyonu okur, formül iki yerde yazılmaz.
        _review_state = self.count_review_state()
        pending = _review_state["pending"]
        reviewed = _review_state["reviewed"]
        total = _review_state["total"]
        if total <= 0:
            self.review_prog.setValue(100)
            self.review_prog_txt.setText("0 / 0")
        else:
            self.review_prog.setValue(int(round(100 * reviewed / total)))
            self.review_prog_txt.setText(f"{reviewed} / {total}")
        # Hız/Kalan göstergesi (prototip speedChip): inceleme hızı bootstrap
        # anında _on_bootstrap_speed_tick ile dolar; burada kalan/ilerleme
        # her karede tazelenir.
        if hasattr(self, "lbl_speed_meter"):
            self.lbl_speed_meter.setText(f"⚡ Hız: — · Kalan: {pending} · {reviewed} / {total}")

    def _box_identity_match(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Tek "ayni kutu" tanimi (P1-13 / K-33): ``same_box()``.

        Eski tanim (koordinat basina 0.02 mutlak fark; eksik anahtar
        atlanirdi) kaldirildi: 0.02x0.02 kutu 0.02 kaymisken EVET diyordu
        (IoU=0.000), 0.50x0.30 kutu 0.05 kaymisken HAYIR (IoU=0.818)
        (F073-F1); ``w/h``'si olmayan ``det`` listedeki ilk ayni-sinif
        kutuyla eslesip yanlis kutunun silinmesine yol aciyordu (F073-F2).
        Eksik geometrili/sinifsiz kutu artik eslesmez; ``_reject_box``
        eslesme yoksa hicbir sey silmez. ``a is b`` yalnizca nesne
        kimligi kisa yoludur — ikinci bir tanim degildir.
        """
        if a is b:
            return True
        return same_box(a, b)

    def _learn_box_exemplar(
        self,
        it: Dict[str, Any],
        box: Dict[str, Any],
        positive: bool = True,
        source: str = "ui",
    ) -> bool:
        """Kutunun görsel crop özelliklerini ExemplarMemory (learned_exemplars.json) içine kaydeder.

        S-3502: öğrenme başarısını döner — görsel yoksa / imread None ise /
        crop boşsa / hata olursa False döner (sessiz yutma yok); crop geçerli
        olup hafızaya yazım denendiyse True döner (benzer örneğin tazelenmesi
        de yazım sayılır).
        """
        try:
            img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
            if img_path is None or not img_path.is_file():
                return False
            img = self._imread_cached(img_path)
            if img is None or getattr(img, "size", 0) == 0:
                return False

            from dataset_factory.exemplar_memory import extract_box_crop, get_exemplar_memory

            crop = extract_box_crop(img, box)
            if crop is None or crop.size == 0:
                return False

            cid = int(box.get("class_id", 0))
            cname = normalize_box_class_name(cid, box.get("class_name"))
            mem = get_exemplar_memory()
            if positive:
                mem.add_positive_exemplar(crop, cid, cname, source=source)
            else:
                mem.add_negative_exemplar(crop, cid, cname, source=source)
            return True
        except Exception as exc:
            logger.debug("Exemplar recording failed: %s", exc)
            return False

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
            if not self._persist_review_item(it):
                self._append_log(f"[HATA] #{it.get('stem')}: kutu sınıfı değişikliği diske kaydedilemedi.")
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
                # P0-8: açık kabul, önceki demote kararını geçersiz kılar.
                nb.pop("demoted_by_user", None)
                promoted = nb
            else:
                kept.append(b)
        if promoted is None:
            self._append_log("[KUTU] Kabul: kutu bulunamadı")
            return
        it["weak"] = kept
        it["n_weak"] = len(kept)
        it["trusted"] = list(it.get("trusted") or []) + [promoted]
        # S-4401: done/reviewed persist ÖNCESİ işaretlenir; yazım
        # başarısız olursa aşağıda geri alınır (S-1502 deseni).
        prev_done = it.get("done")
        marked_done = False
        if not kept:
            it["done"] = "ok"
            marked_done = True
            # P3-28: pending tek kaynaktan (inline sum kalktı — aynı kural).
            self.count_review_state()
            self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + 1
        self._learn_box_exemplar(it, promoted, positive=True, source="user_accept_box")
        if not self._persist_review_item(it):
            self._append_log(f"[HATA] #{it.get('stem')}: kutu onayı diske kaydedilemedi.")
            if marked_done:
                it["done"] = prev_done
                self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) - 1
                # P3-28: pending tek kaynaktan (inline sum kalktı — aynı kural).
                self.count_review_state()
        else:
            self._append_log(
                f"[KUTU] Kabul → trusted: {promoted.get('class_name', promoted.get('class_id'))}"
            )
        self._mark_model_stale("Kutu onaylandı")
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
        if not self._persist_review_item(it):
            self._append_log(f"[HATA] #{it.get('stem')}: filiz eşlemesi diske kaydedilemedi.")
        else:
            # S-4401: başarı logu yalnızca yazım gerçekten olduysa basılır.
            self._append_log("[KUTU] Sınıf → minimap_sprout (9)")
        self._mark_model_stale("Kutu filiz olarak etiketlendi")
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
                # P0-8 (K-12): demote yalnızca trusted/weak ayrımını
                # değiştirir — köken bayrağı (hand_label, source) ayrı
                # tutulur, ASLA False yazılmaz. Kullanıcının şüpheliye
                # ayırma kararı ayrı bayrakla kaydedilir
                # (demoted_by_user; P0-1 uzlaştırması bu adı sayar).
                # human_approved da temizlenir: persist aynı çağrıda
                # onay bayrağıyla kutuyu trusted'a geri yazmasın.
                # NOT: hand_label'ı korumak tek başına yetmez — persist
                # (_normalize_box_dict) hand_label'dan da trusted
                # çıkarır; demoted_by_user vetosu orada uygulanır.
                nb["demoted_by_user"] = True
                nb["human_approved"] = False
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
        if not self._persist_review_item(it):
            self._append_log(f"[HATA] #{it.get('stem')}: şüpheliye ayırma diske kaydedilemedi.")
        else:
            # S-4401: başarı logu yalnızca yazım gerçekten olduysa basılır.
            self._append_log(f"[KUTU] Şüpheliye ayrıldı: {demoted.get('class_name', demoted.get('class_id'))}")
        self._mark_model_stale("Kutu şüpheliye ayrıldı")
        self._sync_pending_from_queue()
        self._render_queue_list()
        self._render_review_frame()

    def _is_reject_protected_box(
        self, it: Dict[str, Any], det: Dict[str, Any], matched: Dict[str, Any]
    ) -> bool:
        """P0-3 (K-11): reddedilecek kutu el etiketi korumasında mı?

        Kutu-düzeyi kontrol — kare-düzeyi ``_is_hand_label_protected_item``
        tek başına kullanılmaz: karışık karede (TXT el etiketi + weak
        model kutusu bir arada — örn. demote sonrası review JSON'unda
        trusted'da TXT kutu, weak'te şüpheli kutu varken) korumasız weak
        kutunun reddi aynen çalışmalıdır; kare-düzeyi guard onu da
        engellerdi. Korumalı sayılır:
        - eşleşen kutunun kendisi TXT el etiketi ise, veya
        - gelen ``det`` kendisi TXT el etiketi bayrağı taşıyorsa
          (canvas/lightbox kopyası bayrakları korur), veya
        - kare diskte TXT-only ise (review JSON yok → bellekteki
          bayraklar eksik/bozuk olabilir; karedeki her kutu el etiketi
          sayılır).
        """
        if YOLOFactoryDialog._is_txt_hand_box(det):
            return True
        if YOLOFactoryDialog._is_txt_hand_box(matched):
            return True
        try:
            if YOLOFactoryDialog._stem_is_txt_only_frame(
                self._dataset_dir(), str(it.get("stem") or "")
            ):
                return True
        except Exception:
            pass
        return False

    def _reject_box(self, det: Dict[str, Any], from_trusted: bool = False) -> bool:
        """Sil = yalnızca seçilen kutuyu çıkar ve diskteki etiket dosyasını anında güncelle.

        P0-3 (K-11): TXT kökenli el etiketi kutusu reddedilemez. Guard
        fonksiyonun başındadır — 6 çağrı noktasının hepsini kapsar:
        şüpheli/emin satırındaki "Sil" butonları, canvas ✕, canvas
        Delete/Backspace + sağ-tık silme, inceleme/canlı lightbox ✕.
        Korumalı kutuda: bellekten çıkarılmaz, negatif örnek
        öğretilmez, diske yazılmaz; log + kullanıcı mesajı basılır.

        P2-23 (Y004-K1): disk-önce sıra + bool dönüş. Yazım başarısız
        olursa bellek geri alınır (aşağıda) ve False döner — arayan
        (lightbox ✕) UI'dan kutuyu çıkarmaz; başarıda True döner.
        Negatif örnek öğretimi (`_learn_box_exemplar`) yazım
        başarısından SONRA yapılır: yazım başarısız olursa kutu
        diskte dururken exemplar hafızasına "reddedildi" işlenmez.
        """
        if not self._review_items or self._review_index >= len(self._review_items):
            return False
        it = self._review_items[self._review_index]
        weak = list(it.get("weak") or [])
        trusted = list(it.get("trusted") or [])

        # P1-13 (K-33): geometrisi eksik `det` eslesmeden once elenir
        # (``same_box`` False doner) — artik ayni-sinif ilk kutuyla
        # eslesip yanlis kutunun silinmesine yol acamaz (F073-F2).
        if not has_box_geometry(det):
            self._append_log("[KUTU] Sil: kutu geometrisi eksik (cx/cy/w/h yok)")
            return

        # Önce reddedilecek kutuyu bul — henüz değiştirme. Guard,
        # gerçekten silinecek kutuya bakmalıdır: gelen `det` canvas/
        # lightbox kopyası olabilir, bayrakları eksik olabilir.
        matched: Optional[Dict[str, Any]] = None
        matched_from_trusted = False
        for b in trusted:
            if self._box_identity_match(b, det):
                matched = b
                matched_from_trusted = True
                break
        if matched is None:
            for b in weak:
                if self._box_identity_match(b, det):
                    matched = b
                    break

        # P0-3 guard: korumalı kutu reddedilemez — dosya silinmez,
        # negatif örnek öğretilmez, log + kullanıcı mesajı basılır.
        if matched is not None and self._is_reject_protected_box(it, det, matched):
            stem = str(it.get("stem") or "")
            logger.warning(
                "[KORUMA] #%s: el etiketi korumalı kutu reddedilemedi — dosya duruyor",
                stem,
            )
            self._append_log(
                f"[KORUMA] #{stem}: el etiketi korumalı kutu silinemez — dosya duruyor. "
                "Kutuyu değiştirmek için önce kareyi inceleyip onaylayın."
            )
            return False

        # Eşleşme yoksa en yakın kutu SİLİNMEZ (F-0693): yanlış kutuyu silmektense
        # hiçbir şey silinmez.
        if matched is None:
            self._append_log("[KUTU] Sil: kutu bulunamadı")
            return False

        # S-1502: yazım başarısız olursa bellek geri alınır — diskte duran
        # kutu kuyrukta da durur, kullanıcı tekrar deneyebilir.
        prev_trusted = list(trusted)
        prev_weak = list(weak)
        prev_conf = it.get("conf")
        # Yalnızca eşleşen TEK kutu çıkarılır (kimlikle, nesne kimliğiyle):
        # eskiden aynı sınıf+geometriye sahip tüm kopyalar bir tıkta
        # siliniyordu; korumalı kopya eşleşse guard zaten yukarıda durur.
        if matched_from_trusted:
            it["trusted"] = [b for b in trusted if b is not matched]
            it["n_trusted"] = len(it["trusted"])
        else:
            it["weak"] = [b for b in weak if b is not matched]
            it["n_weak"] = len(it["weak"])

        it["conf"] = self._frame_conf(it)
        note = normalize_box_class_name(det.get("class_id"), det.get("key", det.get("class_name")))

        _saved = self._persist_review_item(it)
        if _saved:
            # P2-23: negatif örnek öğretimi disk yazımından SONRA —
            # yazım başarısız olursa kutu diskte dururken exemplar
            # hafızasına "reddedildi" işlenmemeli.
            self._learn_box_exemplar(it, det, positive=False, source="user_reject_box")
            self._mark_model_stale(f"Kutu silindi: {note}")
            self._append_log(f"[KUTU] Silindi ve otomatik kaydedildi: {note}")
        else:
            it["trusted"] = prev_trusted
            it["weak"] = prev_weak
            it["n_trusted"] = len(prev_trusted)
            it["n_weak"] = len(prev_weak)
            it["conf"] = prev_conf
            self._append_log(
                f"[HATA] #{it.get('stem')}: kutu silme diske kaydedilemedi — kutu kuyrukta bırakıldı."
            )
        self._render_queue_list()
        self._render_review_frame()
        return bool(_saved)

    @staticmethod
    def _unlink_quiet(path: Path) -> bool:
        try:
            if path.is_file():
                path.unlink()
                return True
        except Exception as exc:
            logger.debug("unlink failed %s: %s", path, exc)
        return False

    @staticmethod
    def _delete_rejected_scan_files(split: str, stem: str, paths: Sequence[Path]) -> List[str]:
        """Tarama GC: reddedilmiş stem'in dosyalarını sil + stem/neden logla.

        P2-21 (K-07): tarama her YOLO açılışında/kuyruk yenilemede
        reddedilmiş stem'lerin görsel+review+TXT dosyalarını hiçbir iz
        bırakmadan siliyordu. Sessiz silme kalktı — silinen her dosya
        stem + neden (rejected_stems.json) ile tek satırda loglanır.
        Korumalı (el etiketli) kare bu fonksiyona hiç ulaşmaz; o dal
        ayrıca debug loglar (yukarıdaki [KORUMA] satırları).
        """
        deleted = [str(p) for p in paths if YOLOFactoryDialog._unlink_quiet(Path(p))]
        if deleted:
            logger.info(
                "[GC] reddedilen kare silindi (tarama): %s/%s — neden: rejected_stems.json — %s",
                split,
                stem,
                ", ".join(deleted),
            )
        return deleted

    # --- El etiketi (TXT) koruması: budama/reddet silmesine karşı ---
    @staticmethod
    def _is_txt_hand_box(b: Dict[str, Any]) -> bool:
        """TXT kökenli düzeltilmiş el etiketi kutusu mu?

        Tarama bu kutulara ``hand_label=True`` + ``source="hand_label_txt"``
        yazar; işaret persist'te korunur. Fabrika içi tekil onay
        (`_accept_box`, manuel çizim) bu kaynağı taşımaz — o kutular
        zaten tekil insan kararıyla korunur/değişir, bu koruma onları
        hedeflemez.
        """
        return (
            isinstance(b, dict)
            and b.get("hand_label") is True
            and b.get("source") == HAND_LABEL_TXT_SOURCE
        )

    def _item_has_txt_hand_boxes(self, it: Dict[str, Any]) -> bool:
        """Karede en az bir TXT kökenli el etiketi kutusu var mı?"""
        if not isinstance(it, dict):
            return False
        for b in list(it.get("weak") or []) + list(it.get("trusted") or []):
            if YOLOFactoryDialog._is_txt_hand_box(b):
                return True
        return False

    def _item_is_hand_labeled(self, it: Dict[str, Any]) -> bool:
        """Karede el etiketi var mı? (P1-17 / K-06)

        Bellekteki el kutusu (``hand_label`` / ``manual_draw`` /
        demote edilmiş el kutusu) VEYA diskteki TXT-only kare — ikisi
        de otomatik etiketlemenin hedefi olamaz. ``_relabel_single_item``
        ve turbo birleştirme el kutusunu korusa da, bu buton el
        etiketli kareyi hiç hedefe sokmaz (savunma derinliği).
        """
        if not isinstance(it, dict):
            return False
        for b in list(it.get("weak") or []) + list(it.get("trusted") or []):
            if not isinstance(b, dict):
                continue
            if (
                b.get("hand_label") is True
                or b.get("source") == "manual_draw"
                or b.get("demoted_by_user") is True
            ):
                return True
        return self._is_hand_label_protected_item(it)

    @staticmethod
    def _is_requeued_hand_frame(it: Dict[str, Any]) -> bool:
        """P0-8 (K-14): requeue edilmiş + el etiketli kare mi?

        Requeue sonrası trusted boşalabilir (demote edilmiş el kutuları
        weak'tedir); böyle kare `n_weak>0 VE n_trusted==0` ile suspicious
        sayılır ve toplu redde yem olurdu. Bu kare suspicious değildir —
        kökeni insan kararıdır, yeniden incelenir, topluca reddedilmez.
        Kaynak daraltması yapılmaz: hand_label/demoted_by_user hangi
        kaynaktan gelirse gelsin köken kanıtıdır.
        """
        if not isinstance(it, dict) or it.get("requeued") is not True:
            return False
        for b in list(it.get("weak") or []) + list(it.get("trusted") or []):
            if not isinstance(b, dict):
                continue
            if b.get("hand_label") is True or b.get("demoted_by_user") is True:
                return True
        return False

    @staticmethod
    def _stem_review_has_hand_boxes(dataset: Path, stem: str) -> bool:
        """Disk kanıtı: karenin review JSON'unda el etiketi kutusu var mı?

        `_stem_is_txt_only_frame` yalnızca TXT-only kareyi yakalar; review
        JSON'u olan kardeş (aug) karenin el etiketi bu yolla görünmez.
        Aile silmesi bu kontrolü de yapar (P0-8/K-14): kardeş karenin el
        etiketi koruması, ana karenin silmesiyle yok sayılmaz.
        """
        stem = str(stem or "").strip()
        if not stem:
            return False
        ds = Path(dataset)
        for s in ("train", "val"):
            p = ds / "review" / s / f"{stem}.json"
            if not p.is_file():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                # P1-16 (K-17) fail-closed: okunamayan review JSON'un
                # içeriği bilinmiyor — el etiketi varmış gibi sayılır.
                logger.warning(
                    "[KORUMA] %s okunamadı (%s) — el etiketli sayılıyor",
                    p,
                    exc,
                )
                return True
            if not isinstance(payload, dict):
                logger.warning(
                    "[KORUMA] %s: review JSON dict değil — el etiketli sayılıyor",
                    p,
                )
                return True
            for b in list(payload.get("weak") or []) + list(payload.get("trusted") or []):
                if not isinstance(b, dict):
                    continue
                if b.get("hand_label") is True or b.get("demoted_by_user") is True:
                    return True
        return False

    @staticmethod
    def _stem_is_txt_only_frame(dataset: Path, stem: str) -> bool:
        """Disk kanıtı: labels TXT var + review JSON yok (her iki split'te bakılır).

        Tarama bu kareyi TXT kökenli el etiketi sayar; kutu bayrakları
        bellekte bozulmuş/eksik olsa bile disk gerçeği korur.
        """
        stem = str(stem or "").strip()
        if not stem:
            return False
        ds = Path(dataset)
        for s in ("train", "val"):
            if (ds / "labels" / s / f"{stem}.txt").is_file() and not (
                ds / "review" / s / f"{stem}.json"
            ).is_file():
                return True
        return False

    def _is_hand_label_protected_item(self, it: Dict[str, Any]) -> bool:
        """Korumalı kare mi? TXT kökenli el etiketi kutusu varsa veya diskte
        TXT-only kare ise True — budama/reddet/toplu onay dokunamaz.

        P1-16 (K-17) fail-closed: disk kontrolü hata verirse kare
        KORUMALI sayılır — hata "korumasız" diye yutulmaz.
        """
        if not isinstance(it, dict):
            return False
        if self._item_has_txt_hand_boxes(it):
            return True
        try:
            if YOLOFactoryDialog._stem_is_txt_only_frame(
                self._dataset_dir(), str(it.get("stem") or "")
            ):
                return True
        except Exception as exc:
            logger.warning(
                "[KORUMA] #%s: TXT-only kontrolü hata verdi (%s) — korumalı sayılıyor",
                it.get("stem"),
                exc,
            )
            return True
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

    def _sample_wipe_paths(
        self, dataset: Path, it: Dict[str, Any], target_stems: List[str]
    ) -> List[Path]:
        """Reddedilecek karenin tüm dosyaları (review + görsel + TXT + önizleme + game_learning).

        Dosyalar burada silinmez, yalnızca toplanır — silme çağıranda
        düz ``unlink`` ile yapılır (kullanıcı kararı 11 Kasım: reddet
        kalıcı silmedir, çöp/geri yükleme yok). ``it["path"]`` review
        JSON'u ile aynı dosyayı iki kez vermesin diye liste tekilleştirilir.
        """
        candidates: List[Path] = []
        p_val = it.get("path")
        if p_val:
            candidates.append(Path(p_val))
        img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        for s_stem in target_stems:
            for s in ("train", "val"):
                candidates.extend(
                    [
                        dataset / "review" / s / f"{s_stem}.json",
                        dataset / "labels" / s / f"{s_stem}.txt",
                        dataset / "review" / s / f"{s_stem}.sidecar.json",
                        dataset / "approved" / "labels" / s / f"{s_stem}.txt",
                    ]
                )
                for ext in img_exts:
                    candidates.append(dataset / "images" / s / f"{s_stem}{ext}")
                    candidates.append(dataset / "approved" / "images" / s / f"{s_stem}{ext}")
            candidates.extend(
                [
                    dataset / "previews" / f"{s_stem}.jpg",
                    dataset / "previews" / f"{s_stem}_preview.jpg",
                    dataset / "previews" / f"preview_{s_stem}.jpg",
                    dataset / "previews" / f"{s_stem}.png",
                    dataset / "previews" / f"{s_stem}_preview.png",
                    dataset / "previews" / f"preview_{s_stem}.png",
                ]
            )
            gl_dir = dataset / "game_learning"
            if gl_dir.is_dir():
                for sub in ("monsters", "skills", "paths", "obstacles", "vitality", "general", "mix"):
                    for ext in (".jpg", ".json"):
                        candidates.append(gl_dir / sub / f"{s_stem}{ext}")
        seen: Set[str] = set()
        deduped: List[Path] = []
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        return deduped

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

        # El etiketi koruması: korumalı kare reddedilemez — ne labels TXT'si
        # ne görseli silinir, ne de rejected_stems'e yazılır. Red sessizce
        # geçiştirilmez: log satırı + [] dönüşü çağıranı da durdurur.
        # P0-8 (K-14): aile (aug) kontrolü yalnızca TXT-only diske değil,
        # kardeş karenin review JSON'undaki el etiketi kutusuna da bakar.
        if self._is_hand_label_protected_item(it) or any(
            YOLOFactoryDialog._stem_is_txt_only_frame(dataset, s_stem)
            for s_stem in target_stems
        ) or any(
            YOLOFactoryDialog._stem_review_has_hand_boxes(dataset, s_stem)
            for s_stem in target_stems
        ):
            logger.warning(
                "[KORUMA] #%s: el etiketi korumalı kare reddedilemedi — dosyalar duruyor",
                stem,
            )
            self._append_log(
                f"[KORUMA] #{stem}: el etiketi korumalı kare reddedilemedi — dosyalar duruyor, tek tek inceleyip onaylayın."
            )
            return removed

        self._record_rejected_stems(target_stems)

        try:
            from dataset_factory.approved_pool import remove_approved_sample
        except Exception:
            remove_approved_sample = None

        for s_stem in target_stems:
            if remove_approved_sample is not None:
                try:
                    remove_approved_sample(s_stem)
                except Exception:
                    pass

        # Kullanıcı kararı (11 Kasım): reddet kalıcı silmedir — çöp/geri
        # yükleme yok, dosyalar düz unlink ile silinir.
        candidates = self._sample_wipe_paths(dataset, it, target_stems)
        for p in candidates:
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

        # El etiketi koruması: korumalı kareler toplu silmeden çıkarılır —
        # ne dosyaları silinir ne rejected_stems'e yazılır; sessiz geçiştirme yok.
        deletable: List[Dict[str, Any]] = []
        blocked_stems: List[str] = []
        for it in items:
            stem = str(it.get("stem") or "").strip()
            if not stem:
                continue
            if "_aug" not in stem:
                fam = self._stems_for_sample_wipe("", stem)
            else:
                fam = [stem]
            # P0-8 (K-14): aile (aug) kontrolü yalnızca TXT-only diske
            # değil, kardeş karenin review JSON'undaki el etiketi
            # kutusuna da bakar.
            if self._is_hand_label_protected_item(it) or any(
                YOLOFactoryDialog._stem_is_txt_only_frame(dataset, s) for s in fam
            ) or any(
                YOLOFactoryDialog._stem_review_has_hand_boxes(dataset, s) for s in fam
            ):
                blocked_stems.append(stem)
                continue
            deletable.append(it)
        if blocked_stems:
            logger.warning(
                "[KORUMA] %d el etiketi korumalı kare toplu reddedilmedi: %s",
                len(blocked_stems), ", ".join(blocked_stems[:10]),
            )
            self._append_log(
                f"[KORUMA] {len(blocked_stems)} el etiketi korumalı kare toplu reddedilmedi "
                f"({', '.join(blocked_stems[:5])}) — dosyalar duruyor."
            )
        if not deletable:
            return 0
        items = deletable
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

        # Kullanıcı kararı (11 Kasım): reddet kalıcı silmedir — çöp/geri
        # yükleme yok, dosyalar düz unlink ile silinir.
        for idx, it in enumerate(items, 1):
            stem = str(it.get("stem") or "").strip()
            if not stem:
                continue

            target_stems = [stem]
            if "_aug" not in stem:
                target_stems = self._stems_for_sample_wipe("", stem)

            for s_stem in target_stems:
                if remove_approved_sample is not None:
                    try:
                        remove_approved_sample(s_stem)
                    except Exception:
                        pass

            candidates = self._sample_wipe_paths(dataset, it, target_stems)
            for p in candidates:
                if self._unlink_quiet(p):
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
        # P3-28: pending tek kaynaktan (count_review_state önbelleği tazeler).
        self.count_review_state()

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

        spec = CLASS_ID_TO_SPEC.get(cid)
        if spec is None:
            # Bilinmeyen cid sınıf-0 adıyla kaydedilmez (F-0694).
            logger.warning("Bilinmeyen sınıf id ile manuel kutu: cid=%s", cid)
            cname = f"class_{cid}"
        else:
            cname = str(spec.get("key", f"class_{cid}"))
        color_hex = str((spec or {}).get("color_hex", CLASS_COLORS_MAP.get(cid, "#22c55e")))

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
        _saved = self._persist_review_item(it)
        self._mark_model_stale(f"Manuel etiket çizildi: {cname}")
        if _saved:
            self._append_log(f"[MANUEL ETİKET] #{it.get('stem')}: {cname} (cid={cid}) eklendi ve otomatik kaydedildi.")
        else:
            self._append_log(f"[HATA] #{it.get('stem')}: manuel etiket diske kaydedilemedi.")
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
            # P0-8 (K-12/D046): eskiden burası TÜM canvas kutularını
            # trusted yapıp weak=[] sıfırlıyordu — tek kutu sürüklenince
            # karedeki weak kutular da el etiketine terfi ediyordu.
            # Şimdi yalnızca sürüklenen kutu terfi eder; diğer kutular
            # demote öncesi trusted/weak üyeliğini korur (aynı dict
            # nesnesi öncelikli, geometri anahtarı yedek eşleşmedir).
            old_trusted = list(it.get("trusted") or [])
            old_weak = list(it.get("weak") or [])

            # P1-13 (K-33): uyelik tek tanim ``same_box()`` ile kurulur;
            # eski ``_box_key`` (4 ondaliga yuvarlama, ~1e-4) ve cx/cy-only
            # 1e-4 ``is_edited`` karsilastirmasi kaldirildi — w/h yok
            # sayiliyor, eksik geometrili kutu ilk ayni-sinif kutuyla
            # eslesebiliyordu (F073-F2).
            old_trusted_ids = {id(b) for b in old_trusted}
            old_weak_ids = {id(b) for b in old_weak}
            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []
            for b in canvas_anns:
                is_edited = b is box or same_box(b, box)
                if is_edited:
                    # Sürükleme açık bir insan düzenlemesidir: kutu
                    # terfi eder ve önceki demote kararı kalkar.
                    b["trusted"] = True
                    b["hand_label"] = True
                    b["human_approved"] = True
                    b["confidence"] = 1.0
                    b["conf"] = 1.0
                    b.pop("demoted_by_user", None)
                    new_trusted.append(b)
                elif id(b) in old_trusted_ids or any(same_box(b, ob) for ob in old_trusted):
                    new_trusted.append(b)
                elif id(b) in old_weak_ids or any(same_box(b, ob) for ob in old_weak):
                    new_weak.append(b)
                elif b.get("trusted"):
                    new_trusted.append(b)
                else:
                    new_weak.append(b)
            it["trusted"] = new_trusted
            it["weak"] = new_weak
            it["n_trusted"] = len(new_trusted)
            it["n_weak"] = len(new_weak)
            it["conf"] = self._frame_conf(it)

        _saved = self._persist_review_item(it)
        if not _saved:
            self._append_log(f"[HATA] #{it.get('stem')}: kutu düzenlemesi diske kaydedilemedi.")
        cname = box.get("class_name", f"class_{box.get('class_id', 0)}")
        cx = float(box.get("cx", 0.0))
        cy = float(box.get("cy", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        self._mark_model_stale(f"Etiket konumu/boyutu güncellendi: {cname}")
        if _saved:
            self._append_log(
                f"[KUTU DÜZENLENDİ] #{it.get('stem')}: '{cname}' taşındı / boyutlandırıldı "
                f"(cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}) ve diske kaydedildi."
            )
        self._render_queue_list()
        self._render_review_frame()

    def _validate_review_boxes(
        self, boxes: Any, list_name: str, normalize_box: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Kutuları senkronizör kuralıyla süz: geçerli → temiz liste,
        geçersiz → denetim kaydı (B-04).

        Karar tek kaynaktan: `AtomicReviewTxtSynchronizer._normalize_box_dict`
        (sınıfı/koordinatı eksik, sonlu olmayan, aralık dışı ya da
        sıfır/negatif boyutlu kutu → None). Şablon doldurma yok: eksik
        alanlı kutu `class_id=0, cx/cy=0.5, w=0.08, h=0.12` olarak
        yazılmaz; reddedilir ve gerekçesiyle işaretlenir.
        """
        clean: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for idx, b in enumerate(list(boxes or [])):
            if not isinstance(b, dict) or normalize_box(
                b,
                default_trusted=(list_name == "trusted"),
                preserve_flags=True,
            ) is None:
                summary = (
                    {
                        k: b.get(k)
                        for k in ("class_id", "cid", "cls", "cx", "cy", "w", "h", "confidence", "source")
                        if k in b
                    }
                    if isinstance(b, dict)
                    else {}
                )
                rejected.append(
                    {
                        "list": list_name,
                        "index": idx,
                        "reason": self._review_box_rejection_reason(b),
                        "box": summary,
                    }
                )
            else:
                clean.append(b)
        return clean, rejected

    @staticmethod
    def _review_box_rejection_reason(raw: Any) -> str:
        """Geçersiz kutunun insan-okunur gerekçesi (yalnızca açıklama;
        ret kararı `_normalize_box_dict`'ındır)."""
        if not isinstance(raw, dict):
            return "kutu dict değil"
        if not any(k in raw and raw[k] is not None for k in ("class_id", "cid", "cls")):
            return "class_id eksik (sınıfsız kutu)"
        geom_keys = {
            "cx": ("cx", "x_center", "x"),
            "cy": ("cy", "y_center", "y"),
            "w": ("w", "width"),
            "h": ("h", "height"),
        }
        for field, keys in geom_keys.items():
            if not any(k in raw and raw[k] is not None for k in keys):
                return f"{field} eksik"
        try:
            cx = float(raw.get("cx", raw.get("x_center", raw.get("x"))))
            cy = float(raw.get("cy", raw.get("y_center", raw.get("y"))))
            w = float(raw.get("w", raw.get("width")))
            h = float(raw.get("h", raw.get("height")))
        except (TypeError, ValueError):
            return "koordinat sayı değil"
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            return "koordinat sonlu değil"
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return "koordinat aralık dışı (cx/cy [0,1], w/h (0,1])"
        return "senkronizör reddetti"

    def _pool_add_approved(self, it: Dict[str, Any], boxes: List[Dict[str, Any]]) -> bool:
        """Onaylı kareyi havuza ekler; False/hata = başarısız (B-02).

        Hata yutulmaz: False döner, arayan kareyi kuyrukta bırakır.

        K-21 / P1-10 havuz kapısı: havuza yalnızca kutu düzeyinde
        gerçek insan onayı / el etiketi taşıyan kutular girer
        (``box_has_human_provenance`` — ``hand_label`` veya per-box
        ``human_approved``). ``done="ok"`` / ``bulk_approved`` tek
        başına hiçbir kutuyu havuza sokmaz: toplu onayda kutulara tek
        tek bakılmadı. Kökeni kanıtlı kutu yoksa havuz adımı sessizce
        atlanır (True döner — bu hata değil, 0 katkıdır); kare
        kuyruktan çıkar ama eğitime 0 kutu verir. Eklenen kutu sayısı
        ``it["_pool_added_boxes"]``'a yazılır (yalnızca bellek içi,
        diske yazılmaz) ki arayan logu dürüst tutabilsin.
        """
        try:
            from dataset_factory.approved_pool import (
                box_has_human_provenance,
                get_approved_pool,
            )

            eligible = [
                b for b in (boxes or [])
                if box_has_human_provenance(b)
            ]
            if not eligible:
                it["_pool_added_boxes"] = 0
                logger.info(
                    "approved_pool skip #%s: havuza girmedi — hiçbir kutuda "
                    "per-box insan onayı / el etiketi yok "
                    "(bulk_approved tek başına yetmez)",
                    it.get("stem"),
                )
                return True
            pool = get_approved_pool(self._dataset_dir())
            img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
            try:
                ok = bool(
                    pool.add_approved_sample(
                        stem=it["stem"],
                        split=it["split"],
                        image_path=img_path,
                        boxes=eligible,
                    )
                )
            except TypeError:
                ok = bool(
                    pool.add_approved_sample(
                        stem=str(it.get("stem", "")),
                        split=str(it.get("split", "train")),
                        image_path=img_path,
                        trusted_boxes=eligible,
                    )
                )
            it["_pool_added_boxes"] = len(eligible) if ok else 0
            return ok
        except Exception as exc:
            logger.warning("approved_pool add failed #%s: %s", it.get("stem"), exc)
            return False

    def _persist_review_item(self, it: Dict[str, Any]) -> bool:
        """Review kalemini diske yazar; başarıda True, hatada False döner.

        Tek yazıcı `AtomicReviewTxtSynchronizer.write_synchronized_pair`'dır
        (B-01): review JSON + labels TXT aynı çağrıda atomik (.tmp +
        os.replace) yazılır. Eski el yapımı `write_text` çift yazımı ve
        `commit_split_labels` ikinci yazımı kaldırıldı — iki yazıcı
        JSON/TXT'yi farklı üretiyordu (B-01/B-11); label_trust durumu
        review JSON'unun kendisinden okunur (`_load_state`), o yol
        etkilenmez. Havuz ekleme hatası artık yutulmuyor (B-02): pool
        False dönerse / hata fırlatırsa False döner, kare kuyrukta kalır.
        Eksik alanlı kutu şablon değerle yazılmaz (B-04): reddedilir,
        review JSON'a `invalid_boxes` denetim kaydıyla işaretlenir,
        TXT'ye ve havuza girmez. B-02'nin boş-onay koruması "hiç geçerli
        kutu yok" demektir: weak-only kare (toplu onay, B-03) diske
        yazılır — kutular weak kalır, havuza yalnızca trusted girer,
        trusted yoksa havuz adımı atlanır (hata sayılmaz).
        K-21 / P1-10: `done="ok"` tek başına insan onayı ÜRETMEZ —
        kare düzeyi `human_approved` yalnızca arayan açıkça yazdıysa
        (tek-kare onay, canvas onayı) diske yazılır; yoksa senkronizöre
        None geçilir ve diskteki mevcut değer korunur (tarama kare
        düzeyi bayrakları belleğe yüklemez — False geçmek eski gerçek
        onayı silerdi). Toplu onay kare düzeyi `bulk_approved=True`
        yazar (iz olarak diske de yazılır); havuz kapısı bunu tek
        başına kabul etmez — havuza giriş için kutu düzeyinde insan
        onayı / el etiketi gerekir (`_pool_add_approved`).
        Weak-only kare `done="ok"` alıp kuyruktan çıkabilir ama havuza
        0 katkıyla girmez: havuz adımı yalnızca kökeni kanıtlı trusted
        kutu varsa çalışır (aşağıdaki havuz adımı).
        """
        stem = str(it.get("stem") or "").strip()
        split = str(it.get("split") or "train").strip() or "train"
        if not stem:
            logger.warning("persist review failed: stem yok")
            return False
        # [P2-19/F019-K2] Diriltme engeli: reddedilen (diskten silinen)
        # stem asla geri yazılmaz. `_delete_review_sample` /
        # `_batch_delete_review_samples` silmeden ÖNCE
        # `_record_rejected_stems` çağırır; async toplu-onay worker'ı
        # (`_BulkApproveWorker`, worker thread'den bu metodu çağırır)
        # reddet ile yarışsa bile silinen kare `_persist` ile geri
        # yazılamaz. Set yalnızca atanarak değiştirilir
        # (`_record_rejected_stems`/`_load_rejected_stems`), yerinde
        # mutate edilmez — worker thread'den üyelik testi GIL altında
        # güvenlidir; kalıcı liste diskten okunur, dialoglar arası da
        # geçerlidir.
        try:
            rejected = self.__dict__.get("_rejected_stems") or set()
            if stem in rejected:
                logger.warning(
                    "persist review reddedildi (%s): stem reddedilenler listesinde — silinen kare geri yazılmaz",
                    stem,
                )
                return False
        except Exception:
            pass
        try:
            from dataset_factory.atomic_review_sync import (
                AtomicReviewTxtSynchronizer,
                _normalize_box_dict,
            )
        except Exception as exc:
            logger.warning("persist review failed (%s): senkronizör import edilemedi: %s", stem, exc)
            return False

        done_ok = it.get("done") == "ok"
        # K-21 / P1-10: `done="ok"` ⇒ `human_approved=True` dönüşümü
        # KALDIRILDI — kare düzeyinde `done` tek başına insan onayı
        # üretmez. Yalnızca arayanın bellek içi `human_approved`
        # (tek-kare / canvas onayı) diske yazılır; yoksa None geçilir
        # ve senkronizör diskteki mevcut değeri korur.
        # (P3-27'nin `derive_persist_human_approved` çağrısı bu
        # dönüşümü geri getirirdi — P1-10 kazanır, kullanılmaz.)
        human_approved = bool(it.get("human_approved") is True)

        # B-04: eksik/bozuk alanlı kutuyu reddet — şablon doldurma yok.
        clean_trusted, rejected = self._validate_review_boxes(
            it.get("trusted", []), "trusted", _normalize_box_dict
        )
        clean_weak, rejected_weak = self._validate_review_boxes(
            it.get("weak", []), "weak", _normalize_box_dict
        )
        rejected = rejected + rejected_weak
        if rejected:
            logger.warning(
                "persist review (%s): %d geçersiz kutu reddedildi (TXT/havuza girmedi): %s",
                stem,
                len(rejected),
                "; ".join(str(r.get("reason")) for r in rejected[:5]),
            )
        # Bellek = disk: reddedilen kutu kuyrukta da görünmez.
        it["trusted"] = clean_trusted
        it["weak"] = clean_weak

        # B-02 ön kontrol: sıfır geçerli kutuyla onay havuza giremez
        # (F-0784) — onay bayrağı diske yazılmadan önce yakala. "Geçerli
        # kutu" yalnızca trusted demek değildir: toplu onayda (B-03)
        # kutular weak kalır ama `bulk_approved` iziyle onaylanmıştır —
        # geometrisi/sınıfı geçerli bir weak kutusu olan kare, boş onay
        # değildir. Yalnızca hiç geçerli kutusu olmayan (trusted da weak
        # de boş) kare reddedilir; weak-only kare JSON'a yazılır, havuza
        # ise yalnızca trusted kutular girer (aşağıdaki havuz adımı).
        if done_ok and not clean_trusted and not clean_weak:
            logger.warning(
                "persist review failed (%s): onaylanacak geçerli kutu yok — havuz sıfır kutuyu reddeder",
                stem,
            )
            return False

        try:
            syncer = AtomicReviewTxtSynchronizer(dataset_dir=self._dataset_dir())
            # K-21 / P1-10: toplu onay izi kare düzeyinde `bulk_approved`
            # olarak diske yazılır (yalnızca True ise — yoksa diskteki
            # eski iz korunur, False ile silinmez). `human_approved`
            # yalnızca açık onayda True geçilir, yoksa None (koru).
            extra_fields: Dict[str, Any] = {"invalid_boxes": rejected}
            if it.get("bulk_approved") is True:
                extra_fields["bulk_approved"] = True
            result = syncer.write_synchronized_pair(
                split=split,
                stem=stem,
                trusted=clean_trusted,
                weak=clean_weak,
                human_approved=True if human_approved else None,
                done=it.get("done"),
                extra_fields=extra_fields,
            )
            if not bool(getattr(result, "synced", False)):
                logger.warning("persist review failed (%s): senkronizör synced=False döndü", stem)
                return False
        except Exception as exc:
            logger.warning("persist review failed (%s): %s", stem, exc)
            return False

        # Havuza yalnızca kökeni kanıtlı trusted kutular girer (B-03 +
        # K-21 / P1-10): weak-only onayda havuza eklenecek kutu yoktur
        # — bu başarısızlık değil, atlanacak adımdır. Sıfır kutuyla
        # havuz çağrısı (F-0784) yapılmaz; `_pool_add_approved`
        # kökensiz kutuyu kendisi eler (bulk_approved tek başına
        # yetmez) — elenen kare kuyrukta kalmaz, 0 katkıyla çıkar.
        # Yalnızca gerçek havuz hatasında (False) kare kuyrukta kalır
        # (B-02).
        if done_ok and clean_trusted and not self._pool_add_approved(it, clean_trusted):
            logger.warning(
                "persist review failed (%s): onay havuza eklenemedi — kare kuyrukta bırakıldı",
                stem,
            )
            return False
        return True

    def _review_done(self, kind: str) -> None:
        if not self._review_items:
            return
        index = self._review_index
        it = self._review_items[index]
        if it.get("done"):
            return
        stem = it.get("stem")
        if kind == "no":
            # El etiketi koruması: korumalı kare tekli redde de silinmez —
            # kuyruktan çıkarılmaz, incelenmiş sayılmaz; log görünür.
            if self._is_hand_label_protected_item(it):
                logger.warning("[KORUMA] #%s: tekli red engellendi (el etiketi korumalı)", stem)
                self._append_log(
                    f"[KORUMA] #{stem}: el etiketi korumalı kare reddedilemedi — dosyalar duruyor, tek tek inceleyip onaylayın."
                )
                return
            for b in it.get("weak", []):
                # Negatif örnek öğrenimi tüm sınıfları kapsar (F-0774): en yaygın
                # sınıflar hariç tutulmuyor.
                self._learn_box_exemplar(it, b, positive=False, source="user_frame_reject")
            removed = self._delete_review_sample(it)
            # P0-3 (K-11 / S-4401): silme başarısı kontrolü — 0 dosya
            # silindiyse kare kuyruktan çıkarılmaz, `reviewed` artmaz,
            # "[RED] diskten silindi" başarı logu basılmaz. (Negatif örnek
            # öğretimi yukarıda kaldı: görsel crop diskten okunur, silme
            # sonrası görsel yoktur; 0 dosya zaten görselin çoktan
            # gitmiş olması demektir, o durumda öğretim de False döner.)
            if len(removed) == 0:
                logger.warning(
                    "[RED] #%s: kare diskten silinemedi (0 dosya) — kare kuyrukta bırakıldı",
                    stem,
                )
                self._append_log(
                    f"[HATA] #{stem}: kare diskten silinemedi (0 dosya) — kare kuyrukta bırakıldı, tekrar deneyin."
                )
                return
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
                f"[RED] #{stem} diskten silindi ({len(removed)} dosya) · kalan {self.count_review_state()['pending']}"
            )
            self._render_queue_list()
            self._render_review_frame()
            self._update_blockers()
            self._update_next_card()
            if self.count_review_state()["pending"] == 0:
                self._mark_step_done(2)
                self._append_log("[AKIŞ] Onay kuyruğu boşaldı → Hazırla adımı hazır.")
            self._refresh_stepper_visuals()
            return

        # B-12: yazım hatasında S-1502 deseni — eskiden `done="ok"` +
        # `human_approved` + terfi, persist ÖNCESİ yapılıyor, hata
        # durumunda geri alınmıyordu; fonksiyon başındaki
        # `if it.get("done"): return` kareyi tamamen kilitliyordu.
        # (Exemplar öğrenimi geri alınamaz — o, onay niyetiyle yapıldı;
        # ama disk kaydı düşmemiş karenin durumu kuyrukta kalır.)
        # B-02/B-03 kilidi (canvassız tek-kare onay): kullanıcı kutuları
        # canvas'ta hiç görmediyse weak-only kare onaylanamaz. Persist'in
        # toplu-onay kabulü (trusted-VEYA-weak, Fix-BH) yalnızca
        # _bulk_approve içindir — orada kullanıcı filtreleyip onay
        # penceresinde Evet dedi (açık insan kararı); canvassız tek-kare
        # onayda ise kutular incelenmedi, kare pending kalır. Zaten
        # trusted kutusu olan kare etkilenmez (eski davranış: onaylanır).
        review_prev = self.__dict__.get("review_preview")
        _has_canvas = review_prev is not None and bool(
            getattr(review_prev, "_annotations", None)
        )
        if not _has_canvas and not list(it.get("trusted") or []):
            self._append_log(
                f"[B-02] #{stem} canvassız onaylanamadı — zayıf kutular "
                "incelenmedi, kare kuyrukta bırakıldı."
            )
            return
        _prev_done = it.get("done")
        _prev_approved = it.get("human_approved")
        _prev_trusted = list(it.get("trusted") or [])
        _prev_weak = list(it.get("weak") or [])
        it["done"] = "ok"
        it["human_approved"] = True
        if review_prev is not None and getattr(review_prev, "_annotations", None):
            canvas_anns = list(review_prev._annotations)
            for b in canvas_anns:
                b["trusted"] = True
                b["hand_label"] = True
                b["human_approved"] = True
                # Onay, kutuyu güvenilir yapar; güven skorunu 1.0'a zorlamaz —
                # zayıf kutunun ölçülmüş güveni korunur (F-0608).
                self._learn_box_exemplar(it, b, positive=True, source="user_frame_approve")
            it["trusted"] = canvas_anns
            it["weak"] = []
        else:
            # Canvas yok ama karede zaten trusted kutu var (weak-only
            # kare buraya ulaşamaz — B-02/B-03 kilidi yukarıda döndü):
            # weak kutular tek tek incelenmedi — toplu terfi YOK (B-03).
            # Kare düzeyinde onaylanır (done="ok"), ama kutular weak
            # kalır, yalnızca `bulk_approved` izi taşır: ne
            # `hand_label`/`human_approved` damgası alırlar ne de pozitif
            # örnek olarak öğretilirler (K1/E-04: şablon kutu el etiketi
            # gibi görünemez). TXT'ye/havuza yalnızca zaten trusted olan
            # kutular girer.
            bulk_marked = []
            for b in it.get("weak", []):
                nb = dict(b)
                nb["bulk_approved"] = True
                nb["trusted"] = False
                nb["weak"] = True
                bulk_marked.append(nb)
            it["weak"] = bulk_marked

        # Oturum onaylanan Kinah sayacı görünmez defter tutma olarak kalır
        # (ekonomi katmanı; eski testin sayaç beklentisi); bu ekranda
        # hiçbir yerde gösterilmez, loga da yazılmaz.
        approved_boxes = list(it.get("trusted", []))
        approved_kinah = sum(self._estimate_box_kinah(b) for b in approved_boxes)
        self._session_approved_kinah = self.__dict__.get("_session_approved_kinah", 0) + approved_kinah

        if not self._persist_review_item(it):
            it["done"] = _prev_done
            it["human_approved"] = _prev_approved
            it["trusted"] = _prev_trusted
            it["weak"] = _prev_weak
            self._append_log(f"[HATA] #{stem} onay kaydı diske yazılamadı — kare kuyrukta bırakıldı.")
            return
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
        # K-21 / P1-10: havuza giren kutu sayısı `_persist`'in
        # `_pool_added_boxes` izinden okunur — kökensiz trusted kutu
        # havuza girmez, log "havuza kaydedildi" diyemez.
        n_pool = int(it.get("_pool_added_boxes", 0) or 0)
        if n_saved > 0 and n_pool > 0:
            self._append_log(
                f"[ONAYLANDI & KAYDEDİLDİ] #{stem}: {n_pool} etiket diske ve onay havuzuna kesin olarak kaydedildi · kalan {self.count_review_state()['pending']}"
            )
        elif n_saved > 0:
            # Trusted kutu var ama hiçbirinde kutu düzeyinde insan
            # onayı / el etiketi yok (örn. canvassız onayda güvenle
            # terfi etmiş model kutular): diske yazıldı, havuza girmedi.
            self._append_log(
                f"[ONAYLANDI & KAYDEDİLDİ] #{stem}: {n_saved} trusted etiket diske kaydedildi, havuza girmedi (kutu düzeyinde insan onayı / el etiketi yok) · kalan {self.count_review_state()['pending']}"
            )
        else:
            # Weak-only onay (B-03): kutular weak kaldı, havuza girmedi —
            # log havuza yazıldığını iddia etmemeli, yalnızca inceleme
            # JSON'unun kaydedildiğini söylemeli.
            self._append_log(
                f"[ONAYLANDI & KAYDEDİLDİ] #{stem}: 0 trusted etiket (yalnızca zayıf kutu — havuza girmedi, inceleme JSON'una kaydedildi) · kalan {self.count_review_state()['pending']}"
            )
        self._render_queue_list()
        self._render_review_frame()
        self._update_blockers()
        self._update_next_card()
        if self.count_review_state()["pending"] == 0:
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

    def _review_shortcuts_allowed(self, event: Any = None) -> bool:
        """A/R/S/K kısayollarının ÜÇ paralel tetik yolunda da ortak kapısı
        (P2-19/F044, F019-K3, F020-K1): QShortcut handler'ları,
        `eventFilter` ve dialog `keyPressEvent` aynı guard'ı görür.

        Kapalı olduğu durumlar:
        - adım 2 aktif değilse / metin girişi odaklıysa (eski koşul),
        - `event` verildiyse: otomatik tekrar (`isAutoRepeat` — basılı
          tutulan tuş art arda kare onaylamaz/silmez) veya Ctrl/Alt
          gibi değiştirici varsa (Ctrl+A/Alt+A onayı — F044),
        - YOLO veri yüklemesi sürüyorsa (`_yolo_load_running`) veya
          loading overlay görünür durumdaysa (yükleme / toplu onay /
          zehir temizliği sırasında kısayol çalışmaz),
        - toplu onay veya zehir temizliği sürüyorsa (senkron toplu
          onayda overlay yoktur — bayrak kontrolü o yolu da kapatır).
        """
        if not self._is_step_2_active() or self._is_text_input_focused():
            return False
        if event is not None:
            try:
                if bool(event.isAutoRepeat()):
                    return False
            except Exception:
                pass
            try:
                mods = event.modifiers()
            except Exception:
                mods = None
            if mods is not None and mods not in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ShiftModifier,
            ):
                return False
        if getattr(self, "_yolo_load_running", False):
            return False
        if getattr(self, "_bulk_approve_running", False):
            return False
        if getattr(self, "_purge_running", False):
            return False
        overlay = getattr(self, "_yolo_loading_overlay", None)
        try:
            if overlay is not None and overlay.isVisible():
                return False
        except Exception:
            pass
        return True

    def _handle_shortcut_approve(self) -> None:
        # QShortcut olayı `isAutoRepeat` taşımaz — tekrar koruması
        # kısayol kurulumundaki `setAutoRepeat(False)` ile yapılır.
        if not self._review_shortcuts_allowed():
            return
        self._on_step2_approve()

    def _handle_shortcut_reject(self) -> None:
        if not self._review_shortcuts_allowed():
            return
        self._on_step2_reject()

    def _handle_shortcut_skip(self) -> None:
        if not self._review_shortcuts_allowed():
            return
        self._review_skip()

    def _handle_shortcut_player_char(self) -> None:
        if not self._review_shortcuts_allowed():
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
        # cx=0.50, cy=0.62, w=0.18, h=0.38 — bu yalnızca TAHMİNİ bir konumdur;
        # görüntüye bakılmadan eklendiği için güven 1.0 / Emin sayılmaz, zayıf
        # kutu olarak incelemeye düşer (F-0695).
        # NOT (F-C): `hand_label` False olmalı — bu kutuyu insan eliyle
        # çizmedi, makine tahmini. True basılırsa senkronizör
        # (`_normalize_box_dict`, weak-terfi ~880, `_passes_conf_floor`
        # ~808) kutuyu insan doğrulaması sayıp conf 0.0 ile trusted'a
        # terfi ettirir ve eğitim txt'sine yazar. Konum, kullanıcı
        # tutamaçlarla düzelttiğinde `_on_box_edited`/mouseRelease
        # üzerinden `hand_label=True` damgasını alır.
        new_box = {
            "class_id": cid,
            "class_name": cname,
            "cx": 0.50,
            "cy": 0.62,
            "w": 0.18,
            "h": 0.38,
            "confidence": 0.0,
            "conf": 0.0,
            "trusted": False,
            "weak": True,
            "hand_label": False,
            "human_approved": False,
            "color": CLASS_COLORS_MAP.get(cid, "#00d2ff"),
            "source": "manual_char_guess",
        }
        it.setdefault("weak", []).append(new_box)
        it["n_weak"] = len(it["weak"])
        it["conf"] = self._frame_conf(it)
        if not self._persist_review_item(it):
            self._append_log(f"[HATA] #{it.get('stem')}: karakter kutusu diske kaydedilemedi.")
        self._mark_model_stale("Karakter kutusu eklendi (konum doğrulanmadı)")
        if hasattr(self, "review_preview"):
            self.review_preview._selected_box_idx = -1
            self.review_preview.update()
        self._append_log("[ETİKET] 🎯 Karakter kutusu tahmini konumda ZAYIF olarak eklendi (26: Karakterim, güven yok). Tutamaçlarla konumunu doğrulayın.")

    def _load_current_review_sample(self) -> None:
        """Load and render the current review sample into preview canvas and detail rows."""
        self._render_review_frame()


    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        if event is not None and event.type() == event.Type.KeyPress:
            if self._review_shortcuts_allowed(event):
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
        Eşik üstü kutular toplu işaret alır (`bulk_approved`), Emin yapılmaz
        (B-03/E-04): kimse kutuya tek tek bakmadı — `hand_label` /
        `human_approved` damgası yok, pozitif örnek olarak öğretilmez.
        Şüpheli/boş filtre modlarında filtrelenen kareleri onaylar.

        [YOLO-BULK-ASYNC] 40 kareye kadar senkron yol (anlık geri bildirim,
        mevcut testlerle uyumlu); üstü worker thread'de çalışır — loading
        overlay + "132/1180" sayacı, UI yanıt verir kalır. Onay penceresi
        (QMessageBox.question) worker başlamadan ÖNCE sorulur (B-06).
        """
        # [P2-19] Karşılıklı dışlama: toplu onay sürerken ikinci toplu
        # onay başlamaz (processEvents reentrancy — senkron yol bayrağı
        # kurar, bkz. `_bulk_approve_sync`); zehir temizliği sürerken
        # toplu onay başlamaz ve tersi (`_purge_poisoned_labels`).
        if getattr(self, "_bulk_approve_running", False):
            self._append_log("[ONAY] Toplu onay zaten sürüyor — bitmeden yeni onay başlatılmaz.")
            return
        if getattr(self, "_purge_running", False):
            self._append_log("[ONAY] Zehir temizliği sürüyor — bitmeden toplu onay başlatılmaz.")
            return
        th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        targets = list(self._filtered_review_indices())
        if not targets:
            self._append_log("[ONAY] Filtrede onaylanacak kare yok.")
            return

        # P1-09 (K-24): dondurma sürerken toplu onay başlayamaz — onay,
        # dondurmanın hazırladığı havuza sessizce girer, dondurulan set
        # bayatlardı. (Async yol `_start_bulk_approve_async` içinde de
        # `_bulk_approve_running` kontrolü vardır; burası iki yolu da keser.)
        if getattr(self, "_freeze_running", False):
            self._append_log(
                "[ONAY] RED: veri seti donduruluyor — dondurma bitmeden toplu onay başlatılamaz."
            )
            return

        # B-06: toplu onay eskiden tek tıkla, onay penceresi sormadan
        # 388 kareye kadar model güvenini `human_approved` yapıyordu —
        # budama ve toplu ret sorarken onay sormuyordu. Selçuk %90 güven
        # skorlarına güvenmediğini söyledi; toplu onay artık sorularak yapılır.
        if not getattr(self, "_auto_confirm_dialogs", False):
            reply = QMessageBox.question(
                self,
                "Filtredekileri Toplu Onayla",
                (
                    f"{len(targets)} kare toplu onaylanacak (Eşik: %{int(round(th * 100))}).\n\n"
                    "• Eşik üstü (≥) kutular toplu işaret alır (bulk_approved); insan onaylı sayılmaz ve onay havuzuna girmez — havuza yalnızca tek tek onaylanmış / el etiketli kutular girer.\n"
                    "• Model güven skoru insan onayı DEĞİLDİR — karelere bakmadıysanız onaylamayın.\n"
                    "• Bu işlem geri alınamaz (etiket + havuz diske yazılır).\n\n"
                    "Devam edilsin mi?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if len(targets) > _BULK_ASYNC_MIN_ITEMS:
            self._start_bulk_approve_async(targets, th)
            return
        self._bulk_approve_sync(targets, th)

    def _bulk_approve_sync(self, targets: List[int], th: float) -> None:
        """Senkron toplu onay (<= _BULK_ASYNC_MIN_ITEMS kare).

        Davranış eski gövdeyle birebir: 15 karede bir processEvents +
        durum etiketi. Çekirdek `_process_bulk_approve_items` ile async
        yolla paylaşılır; W46 boş-kare atlaması burada da geçerlidir.

        P1-09 (K-24): senkron yol eskiden `_bulk_approve_running`
        bayrağını hiç kurmuyordu — `processEvents` sırasında gelen
        dondurma tıklaması onayın ortasına girebiliyordu. Bayrak artık
        burada da kurulur; `_freeze_dataset` sarmalayıcısı onu okur.

        [P2-19/F019-K1] Senkron yol da `_bulk_approve_running` bayrağını
        kurar: `processEvents()` reentrancy ile senkron onay sürerken
        ikinci toplu onay / toplu ret / purge başlayamaz (`_bulk_approve`,
        `_bulk_reject_filtered`, `_purge_poisoned_labels` giriş kapıları
        bayrağı görür). Bayrak `try/finally` ile kalkar — hata yolunda
        da asılı kalmaz.
        """
        # Reentrancy kapısı: `_bulk_approve` giriş kontrolünden sonra
        # (onay penceresi sırasında) ikinci çağrı buraya ulaşırsa durur.
        if getattr(self, "_bulk_approve_running", False):
            self._append_log("[ONAY] Toplu onay zaten sürüyor — bitmeden yeni onay başlatılmaz.")
            return
        if getattr(self, "_purge_running", False):
            self._append_log("[ONAY] Zehir temizliği sürüyor — bitmeden toplu onay başlatılmaz.")
            return
        self._bulk_approve_running = True
        self._bulk_approve_total = len(targets)
        self._bulk_approve_started_at = time.monotonic()
        try:
            self._bulk_approve_sync_inner(targets, th)
        finally:
            self._bulk_approve_running = False

    def _bulk_approve_sync_inner(self, targets: List[int], th: float) -> None:
        """Senkron toplu onayın bayraksız gövdesi (bayrağı çağıran kurar)."""
        total = len(targets)
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        items = [
            self._review_items[idx]
            for idx in targets
            if 0 <= idx < len(self._review_items)
        ]

        def _sync_progress(done: int, _total: int, approved: int = 0) -> None:
            if done % 15 == 0 or done == total:
                QApplication.processEvents()
                if hasattr(self, "lbl_train_status") and self.lbl_train_status is not None:
                    # P2-21 (F043-K2): etiket onaylananı sayar.
                    self.lbl_train_status.setText(f"Toplu onaylanıyor: {approved} onaylandı · {done}/{total} işlendi...")

        self._bulk_approve_running = True
        try:
            result = _process_bulk_approve_items(
                self,
                items,
                th,
                mode,
                progress_cb=_sync_progress,
                log_fn=self._append_log,
            )
        finally:
            self._bulk_approve_running = False
        result["pool_failed"], result["pool_skipped_no_provenance"] = (
            _bulk_approve_pool_readd(self, list(self._review_items))
        )
        self._finalize_bulk_approve(result)

    def _start_bulk_approve_async(self, targets: List[int], th: float) -> bool:
        """Toplu onayı worker thread'de başlat (toplam > _BULK_ASYNC_MIN_ITEMS).

        Onay penceresi çoktan soruldu. Hedef kareler + tam kuyruk
        derin kopyalanır (worker canlı dict'lere dokunmaz); ilerleme
        overlay'de "done/total" sayacıyla gösterilir.
        """
        if getattr(self, "_bulk_approve_running", False):
            self._append_log("[ONAY] Toplu onay zaten sürüyor — bitmeden yeni onay başlatılmaz.")
            return False
        # [P2-19] Toplu onay ↔ purge karşılıklı dışlama (async başlatıcı
        # doğrudan da çağrılabilir — `_bulk_approve` giriş kapısına ek).
        if getattr(self, "_purge_running", False):
            self._append_log("[ONAY] Zehir temizliği sürüyor — bitmeden toplu onay başlatılmaz.")
            return False
        mode = str(getattr(self, "_queue_filter_mode", "all") or "all")
        items = [
            copy.deepcopy(self._review_items[idx])
            for idx in targets
            if 0 <= idx < len(self._review_items)
        ]
        if not items:
            return False
        total = len(items)
        self._bulk_approve_running = True
        self._bulk_approve_total = total
        self._bulk_approve_started_at = time.monotonic()
        logger.info("[ONAY] toplu onay başladı (async): %d kare, eşik %.2f", total, th)
        self._append_log(f"[ONAY] Toplu onay başladı: {total} kare (arka planda, ilerleme ekranda)…")
        self._show_yolo_loading(f"Toplu onay: 0/{total}", title="Toplu onaylanıyor…")
        self._set_yolo_loading_progress(0, total)
        worker = _BulkApproveWorker(
            dialog=self,
            items=items,
            all_items=copy.deepcopy(list(self._review_items)),
            threshold=th,
            mode=mode,
        )
        thread = QThread(self)
        # [YOLO-BULK-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(thread)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_bulk_approve_progress)
        worker.finished.connect(self._on_bulk_approve_finished)
        worker.failed.connect(self._on_bulk_approve_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_bulk_approve_thread)
        self._bulk_approve_thread = thread
        self._bulk_approve_worker = worker
        try:
            thread.start()
        except Exception as exc:
            self._on_bulk_approve_failed(str(exc))
            return False
        return True

    def _clear_bulk_approve_thread(self) -> None:
        self._bulk_approve_thread = None
        self._bulk_approve_worker = None

    def wait_for_bulk_approve(self, timeout_ms: int = 60000) -> bool:
        """Async toplu onay bitene kadar olay döngüsünü çalıştır.

        Test yardımcısı (`wait_for_yolo_load` deseni): onay yoksa hemen
        True döner; zaman aşımında False. Üretim akışını değiştirmez.
        """
        if not self.__dict__.get("_bulk_approve_running", False):
            return True
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(25)
        poll.timeout.connect(
            lambda: loop.quit()
            if not self.__dict__.get("_bulk_approve_running", False)
            else None
        )
        poll.start()
        QTimer.singleShot(int(timeout_ms), loop.quit)
        loop.exec()
        poll.stop()
        return not self.__dict__.get("_bulk_approve_running", False)

    def _on_bulk_approve_progress(self, done: int, total: int, approved: int = 0) -> None:
        if getattr(self, "_yolo_closing", False):
            return
        # P2-21 (F043-K2): overlay metni ONAYLANANI sayar ("Toplu onay:
        # 1149/1149" yalanı biter — 1149 işlenen, onaylanan değil);
        # çubuk işlenenle dolar (takılma izlenimi vermemek için).
        self._set_yolo_loading_detail(f"Toplu onay: {approved} onaylandı · {done}/{total} işlendi")
        self._set_yolo_loading_progress(done, total)
        if done == total or done % 100 == 0:
            logger.info("[ONAY] toplu onay ilerleme: %d onaylandı, %d/%d işlendi", approved, done, total)

    def _on_bulk_approve_finished(self, result: Dict[str, Any]) -> None:
        self._bulk_approve_running = False
        if getattr(self, "_yolo_closing", False):
            return
        started = getattr(self, "_bulk_approve_started_at", 0.0) or time.monotonic()
        elapsed = time.monotonic() - started
        # Worker kopyalarındaki sonucu canlı kuyruğa stem ile uygula
        # (kuyruk arada yenilenmiş olsa bile doğru kare güncellenir;
        # dict kimliği korunur — clear+update yerinde değiştirir).
        by_stem = {str(it.get("stem") or ""): it for it in self._review_items}
        for updated in result.get("items", []):
            live = by_stem.get(str(updated.get("stem") or ""))
            if live is not None:
                live.clear()
                live.update(copy.deepcopy(updated))
        for msg in result.get("errors", []):
            self._append_log(str(msg))
        self._hide_yolo_loading()
        self._finalize_bulk_approve(result)
        logger.info(
            "[ONAY] toplu onay bitti (async): %d kare onaylandı (%d'i el-etiketli/trusted-only, 0 kutu işaretleyerek), "
            "%d kutu işaretlendi, "
            "%d korumalı + %d boş atlandı, %d korumalı karede yalnızca weak işaretlendi (%.1f sn)",
            int(result.get("n", 0)),
            int(result.get("trusted_only_approved", 0)),
            int(result.get("bulk_marked_total", 0)),
            int(result.get("skipped_hand", 0)),
            int(result.get("skipped_empty", 0)),
            int(result.get("protected_weak_marked", 0)),
            elapsed,
        )

    def _on_bulk_approve_failed(self, message: str) -> None:
        self._bulk_approve_running = False
        if getattr(self, "_yolo_closing", False):
            return
        logger.warning("[ONAY] toplu onay başarısız: %s", message)
        self._hide_yolo_loading()
        self._append_log(f"[HATA] Toplu onay yarıda kaldı: {message}")
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.warning(self, "Toplu onay başarısız", str(message))

    def _finalize_bulk_approve(self, result: Dict[str, Any]) -> None:
        """Toplu onay sonrası UI yenilemesi (yalnızca UI thread).

        Senkron ve async yol ortaktır: onaylanan kareler kuyruktan çıkar,
        sayaçlar/etiketler tazelenir, özet log yazılır. Havuz tekrar-ekleme
        bu fonksiyonda YAPILMAZ — senkron yolda önce, async yolda worker'da
        tamamlanmıştır (`result["pool_failed"]` ile gelir).
        """
        n = int(result.get("n", 0))
        bulk_marked_total = int(result.get("bulk_marked_total", 0))
        skipped_hand = int(result.get("skipped_hand", 0))
        skipped_empty = int(result.get("skipped_empty", 0))
        protected_weak_marked = int(result.get("protected_weak_marked", 0))
        trusted_only_approved = int(result.get("trusted_only_approved", 0))
        pool_failed = list(result.get("pool_failed", []) or [])
        pool_skipped_no_provenance = int(result.get("pool_skipped_no_provenance", 0) or 0)
        stopped = bool(result.get("stopped", False))

        self._review_items = [x for x in self._review_items if x.get("done") != "ok"]
        if self._review_index >= len(self._review_items):
            self._review_index = max(0, len(self._review_items) - 1)

        self._sync_pending_from_queue()
        self._pipeline["reviewed"] = int(self._pipeline.get("reviewed", 0)) + n
        if n > 0:
            self._mark_model_stale(f"Filtre onay: {bulk_marked_total} kutu toplu işaretlendi, {n} kare")
        self._render_queue_list()
        self._update_filter_action_buttons()
        self._load_current_review_sample()
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        self._append_log(
            f"[ONAY] Filtredeki {n} kare onaylandı ({bulk_marked_total} kutu toplu işaretlendi"
            + (f"; {trusted_only_approved}'i el-etiketli kare, 0 kutu işaretleyerek onaylandı" if trusted_only_approved else "")
            + "; kutu düzeyinde onay damgası yok)."
            + (f" Havuz hatası: {len(pool_failed)} kare." if pool_failed else "")
            + (f" {pool_skipped_no_provenance} kare havuza girmedi (kutu düzeyinde insan onayı / el etiketi yok — yalnızca toplu işaret)." if pool_skipped_no_provenance else "")
            + (f" {skipped_hand} el etiketi korumalı kare toplu onaya girmedi (tek tek incelenir)." if skipped_hand else "")
            + (f" {protected_weak_marked} korumalı karede yalnızca weak kutular işaretlendi (kare onaylanmadı, kuyrukta)." if protected_weak_marked else "")
            + (f" [W46] {skipped_empty} boş kare toplu onaya girmedi (0 kutu onaylanmaz)." if skipped_empty else "")
            + (" (Yarıda durduruldu — kalan kareler kuyrukta.)" if stopped else "")
        )
        if self.count_review_state()["pending"] == 0:
            self._mark_step_done(2)

    def _bulk_prune_low_confidence_boxes(self) -> None:
        """Kutu düzeyinde budama: Yalnızca filtredeki karelerde eşik altındaki kutuları siler, görsel ve emin kutular diskte korunur.

        P2-20: 40 kareye kadar senkron yol (küçük budama anlık biter,
        senkron testler aynen geçer); üzeri `_BulkPruneWorker` ile arka
        planda çalışır, sonda kuyruk yenileme worker bitişine bağlanır.
        """
        th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
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
                    f"• GÖRSELLER ASLA SİLİNMEZ! Yüksek güvenli (≥ %{int(round(th * 100))}) etiketler korunur ve Emin yapılır.\n\n"
                    "Devam edilsin mi?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if len(targets) > _BULK_ASYNC_MIN_ITEMS:
            self._start_bulk_prune_async(targets, th)
            return
        self._bulk_prune_sync(targets, th)

    def _bulk_prune_sync(self, targets: List[int], th: float) -> None:
        """Senkron budama (<= _BULK_ASYNC_MIN_ITEMS kare).

        Çekirdek `_process_prune_items` ile async yolla paylaşılır;
        15 karede bir processEvents + durum etiketi (eski gövdeyle
        birebir). Dış budama diske yazdıysa kuyruk diskten yenilenir.
        """
        total = len(targets)
        items = [
            self._review_items[idx]
            for idx in targets
            if 0 <= idx < len(self._review_items)
        ]
        if not items:
            return
        # İlerleme paydası çekirdeğin gördüğü liste uzunluğudur.
        total = len(items)

        def _sync_progress(done: int, _total: int) -> None:
            if done % 15 == 0 or done == total:
                QApplication.processEvents()
                if hasattr(self, "lbl_train_status") and self.lbl_train_status is not None:
                    self.lbl_train_status.setText(f"Etiketler budanıyor: {done}/{total}...")

        result = _process_prune_items(
            self,
            items,
            th,
            dataset_dir=self._dataset_dir(),
            progress_cb=_sync_progress,
            log_fn=self._append_log,
        )
        if result.get("external_handled"):
            # F-0610 + P0-7/K-13: dış budama review JSON'larını diskte
            # yeniden yazdı ama bellek içi _review_items eski kaldı.
            # _render_queue_list() ve _load_current_review_sample()
            # bellekten okuduğu için kuyruk diskten yeniden yüklenir.
            try:
                self._reload_review_queue()
            except Exception as exc:
                self._append_log(f"[BUDAMA] Kuyruk yenileme: {exc}")
        self._finalize_bulk_prune(result)

    def _start_bulk_prune_async(self, targets: List[int], th: float) -> bool:
        """Budamayı worker thread'de başlat (toplam > _BULK_ASYNC_MIN_ITEMS).

        Onay penceresi çoktan soruldu. Hedef kareler derin kopyalanır
        (worker canlı dict'lere dokunmaz); ilerleme overlay'de
        "done/total" sayacıyla gösterilir.
        """
        if getattr(self, "_bulk_prune_running", False):
            self._append_log("[BUDAMA] Budama zaten sürüyor — bitmeden yeni budama başlatılmaz.")
            return False
        items = [
            copy.deepcopy(self._review_items[idx])
            for idx in targets
            if 0 <= idx < len(self._review_items)
        ]
        if not items:
            return False
        total = len(items)
        self._bulk_prune_running = True
        self._bulk_prune_total = total
        self._bulk_prune_started_at = time.monotonic()
        logger.info("[BUDAMA] budama başladı (async): %d kare, eşik %.2f", total, th)
        self._append_log(f"[BUDAMA] Budama başladı: {total} kare (arka planda, ilerleme ekranda)…")
        self._show_yolo_loading(f"Budama: 0/{total}", title="Budanıyor…")
        self._set_yolo_loading_progress(0, total)
        worker = _BulkPruneWorker(
            dialog=self,
            items=items,
            dataset_dir=self._dataset_dir(),
            threshold=th,
        )
        thread = QThread(self)
        # [YOLO-BULK-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(thread)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_bulk_prune_progress)
        worker.finished.connect(self._on_bulk_prune_finished)
        worker.failed.connect(self._on_bulk_prune_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_bulk_prune_thread)
        self._bulk_prune_thread = thread
        self._bulk_prune_worker = worker
        try:
            thread.start()
        except Exception as exc:
            self._on_bulk_prune_failed(str(exc))
            return False
        return True

    def _clear_bulk_prune_thread(self) -> None:
        self._bulk_prune_thread = None
        self._bulk_prune_worker = None

    def wait_for_bulk_prune(self, timeout_ms: int = 60000) -> bool:
        """Async budama bitene kadar olay döngüsünü çalıştır.

        Test yardımcısı (`wait_for_bulk_approve` deseni): budama yoksa
        hemen True döner; zaman aşımında False. Üretim akışını değiştirmez.
        """
        if not self.__dict__.get("_bulk_prune_running", False):
            return True
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(25)
        poll.timeout.connect(
            lambda: loop.quit()
            if not self.__dict__.get("_bulk_prune_running", False)
            else None
        )
        poll.start()
        QTimer.singleShot(int(timeout_ms), loop.quit)
        loop.exec()
        poll.stop()
        return not self.__dict__.get("_bulk_prune_running", False)

    def _on_bulk_prune_progress(self, done: int, total: int) -> None:
        if getattr(self, "_yolo_closing", False):
            return
        self._set_yolo_loading_detail(f"Budama: {done}/{total}")
        self._set_yolo_loading_progress(done, total)
        if done == total or done % 100 == 0:
            logger.info("[BUDAMA] budama ilerleme: %d/%d", done, total)

    def _on_bulk_prune_finished(self, result: Dict[str, Any]) -> None:
        self._bulk_prune_running = False
        if getattr(self, "_yolo_closing", False):
            return
        started = getattr(self, "_bulk_prune_started_at", 0.0) or time.monotonic()
        elapsed = time.monotonic() - started
        # Worker kopyalarındaki sonucu canlı kuyruğa stem ile uygula
        # (kuyruk arada yenilenmiş olsa bile doğru kare güncellenir).
        by_stem = {str(it.get("stem") or ""): it for it in self._review_items}
        for updated in result.get("items", []):
            live = by_stem.get(str(updated.get("stem") or ""))
            if live is not None:
                live.clear()
                live.update(copy.deepcopy(updated))
        for msg in result.get("errors", []):
            self._append_log(str(msg))
        self._hide_yolo_loading()
        if result.get("external_handled"):
            # P2-20: sonda senkron `_reload_review_queue` worker bitişine
            # bağlandı — dış budama diski değiştirdiyse kuyruk async
            # yükleme worker'ıyla diskten yenilenir (UI thread taramaz).
            self._request_queue_reload()
        self._finalize_bulk_prune(result)
        logger.info(
            "[BUDAMA] budama bitti (async): %d kutu silindi, %d kutu onaylandı, "
            "%d görsel korundu (%.1f sn)",
            int(result.get("pruned_boxes", 0)),
            int(result.get("promoted_boxes", 0)),
            int(result.get("preserved_images", 0)),
            elapsed,
        )

    def _on_bulk_prune_failed(self, message: str) -> None:
        self._bulk_prune_running = False
        if getattr(self, "_yolo_closing", False):
            return
        logger.warning("[BUDAMA] budama başarısız: %s", message)
        self._hide_yolo_loading()
        self._append_log(f"[BUDAMA] Hata: {message}")
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.warning(self, "Budama başarısız", str(message))

    def _finalize_bulk_prune(self, result: Dict[str, Any]) -> None:
        """Budama sonrası UI yenilemesi (yalnızca UI thread).

        Senkron ve async yol ortaktır: sayaçlar/etiketler tazelenir,
        özet log + bilgi penceresi gösterilir.
        """
        pruned_boxes = int(result.get("pruned_boxes", 0))
        promoted_boxes = int(result.get("promoted_boxes", 0))
        preserved_images = int(result.get("preserved_images", 0))
        negative_taught_boxes = int(result.get("negative_taught_boxes", 0))
        try:
            th = float(result.get("threshold", 0.90))
        except (TypeError, ValueError):
            th = 0.90
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
                    f"• Negatif Hafızaya Öğretildi: {negative_taught_boxes}\n\n"
                    "Hiçbir görsel silinmedi; kalan yüksek güvenli etiketler korundu."
                ),
            )

    # NOT: _auto_label_unlabeled_suspicious tek tanımdır (aşağıda, turbo curator
    # yolu). Aynı isimle ikinci bir tanım Python'da ilki ezerdi; ölü ilk tanım
    # kaldırıldı (F-0541).
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

    def _stop_live_relabel_filtered(self, completed: bool = False) -> None:
        """completed=True yalnızca tüm hedefler işlendiğinde verilir; kullanıcı
        durdurması 'tamamlandı' sayılmaz (F-0696)."""
        self._live_relabeling_filtered = False
        cnt = len(self._filtered_review_indices())
        self.btn_live_relabel_filtered.setText(f"🎯 Filtredekileri Canlı Yeniden Etiketle ({cnt} kare)")
        self.btn_live_relabel_filtered.setObjectName("PipeBtn")
        self.btn_live_relabel_filtered.setStyle(self.btn_live_relabel_filtered.style())
        self._sync_pending_from_queue()
        if completed:
            self._mark_model_stale("Filtre canlı etiketleme tamamlandı")
            self.lbl_speed_meter.setText("⚡ Hız: — · Kalan: — · Tamamlandı")
            self._append_log(
                f"[CANLI ETİKETLEME] Filtredekiler tamamlandı. Toplam çıkarılan kutu: {getattr(self, '_live_relabel_total_boxes', 0)}"
            )
        else:
            self.lbl_speed_meter.setText("⚡ Hız: — · Kalan: — · Durduruldu")
            self._append_log(
                f"[CANLI ETİKETLEME] Filtredekiler kullanıcı tarafından durduruldu (tamamlanmadı). "
                f"Şu ana kadar çıkarılan kutu: {getattr(self, '_live_relabel_total_boxes', 0)}"
            )
        self._render_queue_list()
        self._render_review_frame()

    def _step_live_relabel_filtered(self) -> None:
        if not getattr(self, "_live_relabeling_filtered", False):
            return
        targets = getattr(self, "_live_relabel_targets", [])
        pos = getattr(self, "_live_relabel_pos", 0)
        total = len(targets)

        if pos >= total:
            self._stop_live_relabel_filtered(completed=True)
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

    def set_shared_yolo_detector(self, detector: Any) -> None:
        """Bind the app-wide ONNX detector; drop cached labeler so it rewires."""
        self._shared_yolo_detector = detector
        self._yolo_labeler = None

    def _get_yolo_labeler(self) -> Any:
        """Return the dialog-scoped YOLOAutoLabeler (created once, reused)."""
        if self._yolo_labeler is not None:
            return self._yolo_labeler
        from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

        kwargs: Dict[str, Any] = {}
        if getattr(self, "_shared_yolo_detector", None) is not None:
            kwargs["yolo_detector"] = self._shared_yolo_detector
        self._yolo_labeler = YOLOAutoLabeler(**kwargs)
        return self._yolo_labeler

    def _relabel_box_state_fingerprint(self, it: Dict[str, Any]) -> str:
        """Stable fingerprint of current box state for relabel dedup."""
        parts: List[Tuple[Any, ...]] = []
        for kind in ("trusted", "weak"):
            for b in (it.get(kind) or []):
                if not isinstance(b, dict):
                    continue
                try:
                    cid = int(b.get("class_id", -1) or -1)
                except (TypeError, ValueError):
                    cid = -1
                try:
                    cx = round(float(b.get("cx", 0.0) or 0.0), 4)
                    cy = round(float(b.get("cy", 0.0) or 0.0), 4)
                    bw = round(float(b.get("w", 0.0) or 0.0), 4)
                    bh = round(float(b.get("h", 0.0) or 0.0), 4)
                except (TypeError, ValueError):
                    cx = cy = bw = bh = 0.0
                parts.append(
                    (
                        kind,
                        cid,
                        cx,
                        cy,
                        bw,
                        bh,
                        bool(b.get("hand_label")),
                        bool(b.get("human_approved")),
                        bool(b.get("approved")),
                        bool(b.get("demoted_by_user")),
                        str(b.get("source") or ""),
                    )
                )
        parts.sort()
        return repr(parts)

    def _relabel_dedup_fingerprint(
        self, it: Dict[str, Any], threshold: float
    ) -> Tuple[str, str, str, float]:
        """Frame identity + box state + Emin threshold (reject/approve/threshold change busts cache)."""
        return (
            str(it.get("split") or "train"),
            str(it.get("stem") or ""),
            self._relabel_box_state_fingerprint(it),
            round(float(threshold), 4),
        )

    def _relabel_dedup_hit(
        self, it: Dict[str, Any], threshold: float
    ) -> Optional[int]:
        """Return prior box count when same frame/boxes/threshold was relabeled < TTL ago."""
        cache = getattr(self, "_relabel_dedup_cache", None)
        if not isinstance(cache, dict):
            return None
        fp = self._relabel_dedup_fingerprint(it, threshold)
        key = f"{fp[0]}|{fp[1]}"
        entry = cache.get(key)
        if not isinstance(entry, dict):
            return None
        ttl = float(getattr(self, "_relabel_dedup_ttl_sec", 10.0) or 10.0)
        try:
            age = time.monotonic() - float(entry.get("ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if age > ttl or entry.get("fp") != fp:
            return None
        try:
            return int(entry.get("n", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _remember_relabel_dedup(
        self, it: Dict[str, Any], threshold: float, n_boxes: int
    ) -> None:
        cache = getattr(self, "_relabel_dedup_cache", None)
        if not isinstance(cache, dict):
            self._relabel_dedup_cache = {}
            cache = self._relabel_dedup_cache
        fp = self._relabel_dedup_fingerprint(it, threshold)
        key = f"{fp[0]}|{fp[1]}"
        cache[key] = {"fp": fp, "ts": time.monotonic(), "n": int(n_boxes)}

    def _relabel_single_item(self, it: Dict[str, Any]) -> int:
        """Kareyi YOLOAutoLabeler + ExemplarMemory ile sıfırdan yeniden etiketler (Emin vs Şüpheli ayrımı)."""
        emin_th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        dedup_n = self._relabel_dedup_hit(it, emin_th)
        if dedup_n is not None:
            return dedup_n
        img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
        if img_path is None or not img_path.is_file():
            return 0
        img = self._imread_cached(img_path)
        if img is None or getattr(img, "size", 0) == 0:
            return 0
        try:
            labeler = self._get_yolo_labeler()
            # V20-YZ-ETİKET: "Bu Kareyi Yeniden Etiketle" karedeki HER ŞEYİ
            # etiketler (full_scan=True). Stem learn_monsters_* olsa bile
            # HUD (16-24) ve yol/engel (14/15) kutuları üretilir; yoksa
            # relabel yalnızca canavar + materyal + minimap kutusu
            # üretebilirdi. Kutular zayıf-cap'lidir, onaysız TXT'ye girmez.
            new_anns = labeler.auto_label_frame(
                img, fast_mode=True, source_path=img_path, full_scan=True
            )
            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []

            for a in new_anns:
                if isinstance(a, dict):
                    na = dict(a)
                    conf = float(na.get("confidence", na.get("conf", 0.0)) or 0.0)
                    # P1-12 (K-31/K-34): relabel Emin eşiği kaydırıcıdır
                    # (yedek CONF_BULK_DEFAULT); çıplak "approved" insan sayılmaz.
                    is_emin = (
                        na.get("hand_label") is True
                        or na.get("human_approved") is True
                        or na.get("exemplar_match") is True
                        or conf >= emin_th
                    )
                    na["trusted"] = bool(is_emin)
                    na["weak"] = not bool(is_emin)
                    if is_emin:
                        new_trusted.append(na)
                    else:
                        new_weak.append(na)

            # Kullanıcının daha önce elle çizdiği etiketleri daima koru
            # (P1-17 / K-06): trusted'taki el kutusu trusted'ta kalır;
            # weak'teki el kutusu (şüpheliye ayrılmış / demote edilmiş)
            # weak'te kalır — weak listesi model çıktısıyla bütünüyle
            # değiştirilirken el etiketleri kaybolmaz. Terfi yok:
            # kullanıcı kararı korunur. Model kutusu, korunan bir el
            # kutusuyla çakışıyorsa el versiyonu kalır, model kopyası
            # eklenmez.
            existing_hand = [
                dict(b) for b in (it.get("trusted") or [])
                if b.get("hand_label") is True or b.get("source") == "manual_draw"
            ]
            existing_hand_weak = [
                dict(b) for b in (it.get("weak") or [])
                if b.get("hand_label") is True
                or b.get("source") == "manual_draw"
                or b.get("demoted_by_user") is True
            ]
            existing_hand_all = existing_hand + existing_hand_weak
            merged_trusted = list(existing_hand)
            for b in new_trusted:
                if not any(self._box_identity_match(b, h) for h in existing_hand_all):
                    merged_trusted.append(b)
            merged_weak = list(existing_hand_weak)
            for b in new_weak:
                if not any(self._box_identity_match(b, h) for h in existing_hand_all):
                    merged_weak.append(b)

            it["trusted"] = merged_trusted
            it["weak"] = merged_weak
            it["n_trusted"] = len(merged_trusted)
            it["n_weak"] = len(merged_weak)
            it["done"] = None
            it["conf"] = self._frame_conf(it)
            if not self._persist_review_item(it):
                self._append_log(f"[HATA] #{it.get('stem')}: yeniden etiketleme diske kaydedilemedi.")
                return 0
            n_out = len(new_trusted) + len(new_weak)
            # Dedup fingerprint is the POST-relabel box state: a second click
            # with unchanged boxes+threshold within TTL skips inference.
            self._remember_relabel_dedup(it, emin_th, n_out)
            return n_out
        except Exception as exc:
            logger.warning("Relabel failed for %s: %s", it.get("stem"), exc)
            return 0

    def _relabel_current_frame_clicked(self) -> None:
        """Seçili kareyi anında YZ ve öğrenilmiş örnek hafızasıyla yeniden etiketle."""
        if not self._review_items or self._review_index >= len(self._review_items):
            return
        it = self._review_items[self._review_index]
        stem = it.get("stem", "")
        emin_th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        skipped = self._relabel_dedup_hit(it, emin_th) is not None
        n = self._relabel_single_item(it)
        self._sync_pending_from_queue()
        if skipped:
            self._append_log(
                f"[YENİDEN ETİKETLE] #{stem}: atlandı (aynı kare/kutu/eş, {int(getattr(self, '_relabel_dedup_ttl_sec', 10))}sn)."
            )
            self._render_queue_list()
            self._render_review_frame()
            return
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
        # "Tümünü" gerçekten TÜM kuyruğu hedefler — filtrelenmiş alt küme değil
        # (F-0606: buton metni tüm sayıyı gösterirken işleyici filtredekileri
        # yapıyordu; metin ve davranış tutarlı olmalı).
        targets = list(range(len(getattr(self, "_review_items", []) or [])))
        if not targets:
            self._append_log("[CANLI ETİKETLEME] Kuyrukta kare yok.")
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
        self._append_log(f"[CANLI ETİKETLEME] Başlatıldı: Tüm {len(targets)} kare sırayla YZ ve görsel hafıza ile güncelleniyor...")
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
        """Yalnızca filtredeki kareleri diskten siler (train’e girmez).

        [YOLO-BULK-ASYNC] SENKRON BIRAKILDI — ölçüm (VM, sentetik tmp
        dataset, 2026-11-10): 200 boş kare toplu reddi 1,43 sn sürdü,
        `processEvents` 15 karede bir UI boşluğunu ~0,1 sn'de tutuyor;
        bloklama yok (50,8 sn'lik onay / 18,8 sn'lik temizlik sorununun
        kaynağı burası değil — silme başına yalnızca scandir+unlink var,
        havuz yazımı yok). Async'e taşımak mevcut senkron testleri
        (50 karede processEvents sayacı == 4) bozardı, faydasız.

        [P2-19/F019-K2] Async toplu onay sürerken toplu ret başlamaz:
        ret diskten silerken worker aynı stem'i `_persist` ile geri
        yazardı (silinen kare dirilir). Ek disk-seviyesi koruma
        `_persist_review_item`'daki reddedilen-stem kontrolüdür.
        Zehir temizliği sürerken de başlamaz (aynı dosyaları yazar).
        """
        if getattr(self, "_bulk_approve_running", False):
            self._append_log("[RED] Toplu onay sürüyor — bitmeden toplu ret başlatılmaz.")
            return
        if getattr(self, "_purge_running", False):
            self._append_log("[RED] Zehir temizliği sürüyor — bitmeden toplu ret başlatılmaz.")
            return
        target_indices = list(self._filtered_review_indices())
        if not target_indices:
            self._append_log("[RED] Filtrede reddedilecek kare yok.")
            return

        valid_targets = [i for i in sorted(set(target_indices)) if 0 <= i < len(self._review_items)]
        if not valid_targets:
            return

        # El etiketi koruması: korumalı kareler reddedilmez — onay
        # penceresine girmeden ayrılır, kuyrukta kalır, sessiz geçiştirme yok.
        deletable_targets: List[int] = []
        blocked_stems: List[str] = []
        for i in valid_targets:
            it = self._review_items[i]
            if self._is_hand_label_protected_item(it):
                blocked_stems.append(str(it.get("stem") or ""))
            else:
                deletable_targets.append(i)
        if blocked_stems:
            logger.warning(
                "[KORUMA] %d el etiketi korumalı kare toplu reddedilmedi: %s",
                len(blocked_stems), ", ".join(blocked_stems[:10]),
            )
            self._append_log(
                f"[KORUMA] {len(blocked_stems)} el etiketi korumalı kare toplu reddedilmedi — dosyalar duruyor, kuyrukta kaldı."
            )
        if not deletable_targets:
            self._append_log("[RED] Filtrede reddedilecek (korumasız) kare yok.")
            return

        if not getattr(self, "_auto_confirm_dialogs", False):
            mode_desc = self.queue_filter.currentText() if hasattr(self, "queue_filter") else "filtredeki"
            blocked_note = (
                f" — {len(blocked_stems)} el etiketi korumalı kare korunacak."
                if blocked_stems else ""
            )
            reply = QMessageBox.question(
                self,
                "Filtredekileri Reddet",
                f"{len(deletable_targets)} kare ({mode_desc}) diskten silinecek.{blocked_note}\n"
                "Diskten silinir — geri alınamaz. Devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        total = len(deletable_targets)
        items_to_remove = [self._review_items[i] for i in deletable_targets]

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
        # (yalnızca silinenler — korumalı kareler kuyrukta kalır).
        removed = 0
        for i in reversed(deletable_targets):
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
            f"[RED] Filtredeki {removed} kare diskten silindi · kalan {self.count_review_state()['pending']}"
        )
        if self.count_review_state()["pending"] == 0:
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
                "• El etiketi / tek tek onaylı kareler kuyruğa geri alınmaz (asla geri dönmez kuralı).\n"
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
                # P1-12 (K-32/K-35): ölçülmemiş güven yer tutucusu
                # CONF_REQUEUE_DEFAULT (0.58) + confidence_fabricated işaretli;
                # eski 0.90 varsayılanı ölçülmemiş kutuyu %90 gösteriyordu.
                default_conf=CONF_REQUEUE_DEFAULT,
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
        # P3-28: logdaki pending tek kaynaktan.
        pending = self.count_review_state()["pending"]
        self._append_log(
            "[KUYRUK] Geri alındı: samples={samples} boxes={boxes} empty={empty} "
            "pending={pending} korumalı_atlanan={skipped_protected} (conf varsayılan {requeue_conf:.2f} = ölçülmemiş güven yer tutucusu, "
            "confidence_fabricated işaretli · filtre: hepsi)".format(
                samples=stats.get("samples", 0),
                boxes=stats.get("boxes", 0),
                empty=stats.get("empty_frames", 0),
                pending=pending,
                skipped_protected=stats.get("skipped_protected", 0),
                requeue_conf=CONF_REQUEUE_DEFAULT,
            )
        )
        self._update_blockers()
        self._update_next_card()
        self._refresh_stepper_visuals()
        if not getattr(self, "_auto_confirm_dialogs", False):
            _n_protected = int(stats.get("skipped_protected", 0) or 0)
            _protected_note = (
                f"\n\nEl etiketi / tek tek onaylı {_n_protected} kare kuyruğa geri alınmadı "
                "(asla geri dönmez kuralı)."
                if _n_protected > 0
                else ""
            )
            QMessageBox.information(
                self,
                "Kuyruk hazır",
                (
                    f"{stats.get('samples', 0)} kare kuyruğa alındı "
                    f"({stats.get('boxes', 0)} kutu · {stats.get('empty_frames', 0)} boş).{_protected_note}\n\n"
                    "Hatalı kare: Reddet (R) → images/labels/review silinir.\n"
                    "Güven eşiği + Liste filtresi ile daraltabilirsiniz."
                ),
            )

    def _purge_poisoned_labels(self) -> None:
        """4 Katmanlı Yapay Zeka Destekli Zehir Temizleyici (Dokunulmazlık, Sınıf Onarımı, HUD/Boş Pikseller).

        [YOLO-BULK-ASYNC] Senkron `clean_poison_labels_ai` çağrısı canlı
        logda UI thread'i 18,8 sn kilitledi (2026-11-10 21:16:15) — artık
        worker thread'de çalışır; overlay + "done/total" sayacı, UI yanıt
        verir kalır. Onay penceresi worker başlamadan ÖNCE sorulur.
        Bitişteki `_reload_review_queue` + bilgi penceresi ana thread'de
        kalır (eski davranış).
        """
        if getattr(self, "_purge_running", False):
            self._append_log("[TEMİZ] Zehir temizliği zaten sürüyor — bitmeden yeniden başlatılmaz.")
            return
        # [P2-19] Toplu onay ↔ purge karşılıklı dışlama: toplu onay
        # sürerken purge başlamaz (iki yazıcı aynı review JSON/TXT'yi
        # yazar — onay worker'ı `_persist`, purge `clean_poison_labels_ai`).
        if getattr(self, "_bulk_approve_running", False):
            self._append_log("[TEMİZ] Toplu onay sürüyor — bitmeden zehir temizliği başlatılmaz.")
            return
        if not getattr(self, "_auto_confirm_dialogs", False):
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
            dataset_dir = self._dataset_dir()
            # P1-12 (K-35): temizlik floor'u TRAIN_CONF_FLOOR'dur (0.45) —
            # kaydırıcı (toplu onay eşiği, varsayılan 0.90) floor DEĞİLDİR;
            # eskiden 0.45–0.90 arası meşru kutular "sub-floor" diye siliniyordu.
            floor = TRAIN_CONF_FLOOR
        except Exception as exc:
            self._append_log(f"[TEMİZ] Hata: {exc}")
            if not getattr(self, "_auto_confirm_dialogs", False):
                QMessageBox.warning(self, "Temizlik başarısız", str(exc))
            return
        self._purge_running = True
        self._purge_started_at = time.monotonic()
        logger.info("[TEMİZ] zehir temizliği başladı (async)")
        self._append_log(
            f"[TEMİZ] Zehir temizliği başladı (arka planda, ilerleme ekranda)… floor={floor:.2f}"
        )
        self._show_yolo_loading("Zehir temizliği: taranıyor…", title="Zehir temizleniyor…")
        worker = _PoisonPurgeWorker(dataset_dir=dataset_dir, floor=floor)
        thread = QThread(self)
        # [YOLO-BULK-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(thread)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_purge_progress)
        worker.finished.connect(self._on_purge_finished)
        worker.failed.connect(self._on_purge_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_purge_thread)
        self._purge_thread = thread
        self._purge_worker = worker
        try:
            thread.start()
        except Exception as exc:
            self._on_purge_failed(str(exc))

    def _clear_purge_thread(self) -> None:
        self._purge_thread = None
        self._purge_worker = None

    def wait_for_bulk_purge(self, timeout_ms: int = 120000) -> bool:
        """Async zehir temizliği bitene kadar olay döngüsünü çalıştır.

        Test yardımcısı (`wait_for_yolo_load` deseni): temizlik yoksa
        hemen True döner; zaman aşımında False. Üretim akışını değiştirmez.
        """
        if not self.__dict__.get("_purge_running", False):
            return True
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(25)
        poll.timeout.connect(
            lambda: loop.quit()
            if not self.__dict__.get("_purge_running", False)
            else None
        )
        poll.start()
        QTimer.singleShot(int(timeout_ms), loop.quit)
        loop.exec()
        poll.stop()
        return not self.__dict__.get("_purge_running", False)

    def _on_purge_progress(self, done: int, total: int) -> None:
        if getattr(self, "_yolo_closing", False):
            return
        self._set_yolo_loading_detail(f"Zehir temizliği: {done}/{total}")
        self._set_yolo_loading_progress(done, total)
        if done == total or done % 100 == 0:
            logger.info("[TEMİZ] zehir temizliği ilerleme: %d/%d", done, total)

    def _on_purge_finished(self, stats: Dict[str, Any]) -> None:
        self._purge_running = False
        if getattr(self, "_yolo_closing", False):
            return
        started = getattr(self, "_purge_started_at", 0.0) or time.monotonic()
        elapsed = time.monotonic() - started
        self._hide_yolo_loading()
        if stats.get("stopped"):
            self._append_log("[TEMİZ] Zehir temizliği durduruldu (kapanış).")
            return
        # P2-21 (F056-K1/K2, F030-K2): `ok` / `manifest_error` /
        # `unreadable_files` başarı yolunda okunur — manifest arızası
        # veya okunamayan review dosyası varken "Tüm etiketler
        # temizlendi" denmez. Katman dökümünün tamamı (14 alan) loga
        # düşer; dialog eskiden yalnızca 5 toplam basıyordu.
        # F056-K3: `protected_boxes` onarılan kutuları da içerir
        # (testte sabitli semantik) — toplanınca çifte sayım olmasın
        # diye ekranda "korunan" onarılandan ayrık gösterilir.
        ok = stats.get("ok", True) is not False and not stats.get("manifest_error")
        manifest_error = stats.get("manifest_error")
        unreadable = int(stats.get("unreadable_files", 0) or 0)
        protected = int(stats.get("protected_boxes", 0) or 0)
        repaired = int(stats.get("auto_repaired_boxes", 0) or 0)
        protected_excl_repaired = max(0, protected - repaired)
        layer_detail = (
            f"katman: negatif_eslesme={stats.get('negative_match_deleted', 0)} "
            f"hud={stats.get('hud_poison_deleted', 0)} "
            f"duz_zemin={stats.get('flat_texture_deleted', 0)} "
            f"taban_alti={stats.get('sub_floor_deleted', 0)} "
            f"omni={stats.get('omni_demoted_boxes', 0)} "
            f"nms_tekrar={stats.get('deduped_boxes', 0)} "
            f"negatif_ogretilen={stats.get('negative_taught_boxes', 0)} "
            f"gorselsiz_kutu={stats.get('boxes_without_visual', 0)} "
            f"taranan={stats.get('scanned', 0)} "
            f"okunamayan={unreadable}"
            + (f" okunamayan_dosyalar={list(stats.get('unreadable_file_names', []) or [])[:5]}" if unreadable else "")
            + (f" manifest_hatasi={manifest_error}" if manifest_error else "")
        )
        self._append_log(
            f"[TEMİZ YZ] Korunan={protected_excl_repaired} (onarılan {repaired} hariç) Onarılan={repaired} "
            f"Silinen={stats.get('poison_deleted_boxes', 0)} Negatif={stats.get('negative_taught_boxes', 0)} "
            f"Dosya={stats.get('files_rewritten', 0)} (yeniden yazılan review dosyası; değişen dosya sayısı değil) ({elapsed:.1f} sn) — {layer_detail}"
        )
        if ok and unreadable == 0:
            logger.info("[TEMİZ] zehir temizliği bitti (async): %.1f sn — %s", elapsed, layer_detail)
        else:
            logger.warning(
                "[TEMİZ] zehir temizliği EKSİK bitti (async): ok=%s manifest_error=%s okunamayan=%d (%.1f sn) — %s",
                ok,
                manifest_error,
                unreadable,
                elapsed,
                layer_detail,
            )
            self._append_log(
                "[TEMİZ] DİKKAT: manifest hatası / okunamayan dosya var — "
                f"{unreadable} dosya temizlenmedi, istatistikler bayat olabilir."
                + (f" Manifest hatası: {manifest_error}" if manifest_error else "")
            )
        self._mark_model_stale("YZ Zehirli etiket temizliği tamamlandı")
        try:
            self._reload_review_queue()
        except Exception as exc:
            self._append_log(f"[TEMİZ] Kuyruk yenileme: {exc}")
        self._update_blockers()
        self._update_next_card()
        if not getattr(self, "_auto_confirm_dialogs", False):
            if ok and unreadable == 0:
                QMessageBox.information(
                    self,
                    "Yapay Zeka Temizliği Tamamlandı",
                    f"🟢 Korunan Emin Kutu: {protected_excl_repaired} (onarılan {repaired} hariç)\n"
                    f"🔧 Sınıfı Onarılan Kutu: {repaired}\n"
                    f"✂️ Silinen Zehirli Kutu: {stats.get('poison_deleted_boxes', 0)}\n"
                    f"🧠 Negatif Hafızaya Öğretilen: {stats.get('negative_taught_boxes', 0)}\n"
                    f"📦 Tekrar Eden (NMS): {stats.get('deduped_boxes', 0)}\n"
                    f"📁 Güncellenen Dosya: {stats.get('files_rewritten', 0)}\n\n"
                    "Tüm etiketler temizlendi ve dataset senkronize edildi.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Yapay Zeka Temizliği Eksik Tamamlandı",
                    f"🟢 Korunan Emin Kutu: {protected_excl_repaired} (onarılan {repaired} hariç)\n"
                    f"🔧 Sınıfı Onarılan Kutu: {repaired}\n"
                    f"✂️ Silinen Zehirli Kutu: {stats.get('poison_deleted_boxes', 0)}\n"
                    f"📁 Güncellenen Dosya: {stats.get('files_rewritten', 0)}\n\n"
                    f"⚠️ {unreadable} review dosyası okunamadı (temizlenmedi)"
                    + (f"; manifest hatası: {manifest_error}" if manifest_error else "")
                    + ". Tüm etiketlerin temizlendiği söylenemez — logdaki katman dökümüne bakın.",
                )

    def _on_purge_failed(self, message: str) -> None:
        self._purge_running = False
        if getattr(self, "_yolo_closing", False):
            return
        logger.warning("[TEMİZ] zehir temizliği başarısız: %s", message)
        self._hide_yolo_loading()
        self._append_log(f"[TEMİZ] Hata: {message}")
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.warning(self, "Temizlik başarısız", str(message))

    def _turbo_relabel_filtered(self) -> None:
        """Filtredeki kareleri DatasetCurator turbo batch ile otomatik yeniden etiketler.

        P2-20: `relabel_dataset_batch` (YOLO inference) artık UI thread'de
        çalışmaz — `_TurboRelabelWorker` arka planda yapar, ilerleme
        signal ile döner (`processEvents()` reentrancy penceresi kapandı).
        Bitişte kuyruk async yükleme worker'ıyla yenilenir.
        """
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
        self._start_turbo_relabel(target_paths, kind="filtered")

    def _auto_label_unlabeled_suspicious(self) -> None:
        """Etiketsiz (0 kutu) ve şüpheli kareleri tespit edip otomatik olarak turbo etiketler.

        P1-17 / K-06: el etiketli kare hedefe girmez — tek weak kutu
        taşıyan ama trusted'ta el etiketi olan kare bile eskiden
        hedefe girip turbo'da el etiketini kaybediyordu. TXT-only
        el etiketli kare de "şüpheli" sayılmaz: disk kanıtı
        (``_is_hand_label_protected_item``) bellek bayrakları
        bozulmuş olsa bile korur. Çalışmadan önce onay diyaloğu
        sorulur (P1-17); diyalog P2-20 birleşiminde kalkmıştı, v17
        düzeltmesiyle geri kondu — worker yalnızca Evet ile başlar.

        P2-20: `_turbo_relabel_filtered` ile aynı worker yolu — YOLO
        inference UI thread'de yapılmaz, ilerleme signal ile döner.
        """
        target_paths = []
        excluded_hand = 0
        for it in self._review_items:
            # El etiketli kare asla hedef değildir — weak kutu taşısa
            # bile (K-06 kanıtı: trusted'ta el etiketi + 1 weak kutu
            # hedefe giriyordu).
            if self._item_is_hand_labeled(it):
                excluded_hand += 1
                continue
            weak = it.get("weak") or []
            trusted = it.get("trusted") or []
            if len(trusted) == 0 or len(weak) > 0:
                img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))
                if img_path and img_path.is_file():
                    target_paths.append(img_path)

        if not target_paths:
            if excluded_hand:
                QMessageBox.information(
                    self,
                    "Bilgi",
                    "Etiketsiz veya şüpheli kare kalmadı!\n"
                    f"({excluded_hand} el etiketli kare korundu, hedefe girmedi.)",
                )
            else:
                QMessageBox.information(self, "Bilgi", "Etiketsiz veya şüpheli kare kalmadı!")
            return

        # K-06: tek tıkla, onaysız, el-etiketli kareleri de kapsayan
        # toplu etiketleme yok — kapsamı gösterip sorulur (B-06 deseni).
        if not getattr(self, "_auto_confirm_dialogs", False):
            hand_note = (
                f"\n• {excluded_hand} el etiketli kare hedefe girmedi (korundu)."
                if excluded_hand
                else ""
            )
            reply = QMessageBox.question(
                self,
                "Etiketsiz & Şüphelileri Etiketle",
                (
                    f"{len(target_paths)} kare otomatik etiketlenecek (model + görsel hafıza).\n\n"
                    "• Yalnızca etiketsiz (0 kutu) ve şüpheli kareler hedeflenir.\n"
                    "• El etiketli kareler bu işleme girmez; model kutuları el etiketinin üzerine yazılmaz."
                    f"{hand_note}\n"
                    "• Bu işlem geri alınamaz (etiketler diske yazılır).\n\n"
                    "Devam edilsin mi?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._append_log("[TURBO] Etiketsiz/şüpheli etiketleme kullanıcı tarafından iptal edildi.")
                return

        self._append_log(
            f"[TURBO] {len(target_paths)} etiketsiz/şüpheli kare otomatik etiketleniyor"
            f" ({excluded_hand} el etiketli kare hariç tutuldu)..."
        )
        self._start_turbo_relabel(target_paths, kind="unlabeled_suspicious")

    def _start_turbo_relabel(self, target_paths: List[Path], kind: str) -> bool:
        """Turbo yeniden etiketlemeyi worker thread'de başlat (P2-20)."""
        if getattr(self, "_turbo_relabel_running", False):
            self._append_log("[TURBO] Otomatik etiketleme zaten sürüyor — bitmeden yeni tur başlatılmaz.")
            return False
        total = len(target_paths)
        self._turbo_relabel_running = True
        self._turbo_relabel_kind = str(kind)
        self._turbo_relabel_total = total
        self._turbo_relabel_started_at = time.monotonic()
        self._append_log(f"[TURBO] {total} kare yeniden etiketleniyor (arka planda)…")
        self._show_yolo_loading(f"Otomatik etiketleme: 0/{total}", title="Otomatik etiketleniyor…")
        self._set_yolo_loading_progress(0, total)
        # P1-17 / K-06: incremental=True — varsayılan incremental=True:
        # değişmemiş kare atlanır, turbo her çalıştığında tüm hedefleri
        # sıfırdan etiketlemez (P2-20'nin incremental=False'u v16'dan
        # mekanik taşınmış eski argümandı).
        worker = _TurboRelabelWorker(
            dataset_dir=self._dataset_dir(),
            image_paths=list(target_paths),
            incremental=True,
        )
        thread = QThread(self)
        # [YOLO-BULK-ASYNC] Güvenlik ağı: kapanışta atexit ile durdurulabilsin.
        _YOLO_BACKGROUND_THREADS.add(thread)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_turbo_relabel_progress)
        worker.finished.connect(self._on_turbo_relabel_finished)
        worker.failed.connect(self._on_turbo_relabel_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_turbo_relabel_thread)
        self._turbo_relabel_thread = thread
        self._turbo_relabel_worker = worker
        try:
            thread.start()
        except Exception as exc:
            self._on_turbo_relabel_failed(str(exc))
            return False
        return True

    def _clear_turbo_relabel_thread(self) -> None:
        self._turbo_relabel_thread = None
        self._turbo_relabel_worker = None

    def wait_for_turbo_relabel(self, timeout_ms: int = 300000) -> bool:
        """Async turbo etiketleme bitene kadar olay döngüsünü çalıştır.

        Test yardımcısı (`wait_for_bulk_approve` deseni): etiketleme
        yoksa hemen True döner; zaman aşımında False. Üretim akışını
        değiştirmez. Varsayılan 5 dk — tam filtre turu uzun sürebilir.
        """
        if not self.__dict__.get("_turbo_relabel_running", False):
            return True
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(25)
        poll.timeout.connect(
            lambda: loop.quit()
            if not self.__dict__.get("_turbo_relabel_running", False)
            else None
        )
        poll.start()
        QTimer.singleShot(int(timeout_ms), loop.quit)
        loop.exec()
        poll.stop()
        return not self.__dict__.get("_turbo_relabel_running", False)

    def _on_turbo_relabel_progress(self, info: Dict[str, Any]) -> None:
        """Worker ilerleme signal'i — hız göstergesi + overlay (UI thread).

        P2-20: eski `_on_prog` callback'i `QApplication.processEvents()`
        çağırıp reentrancy penceresi açıyordu; artık yalnızca signal
        slot'u yazar, olay döngüsü işletilmez.
        """
        if getattr(self, "_yolo_closing", False):
            return
        fps = float(info.get("fps", 0.0) or 0.0)
        eta_s = float(info.get("eta_s", 0.0) or 0.0)
        p = int(info.get("processed", 0) or 0)
        t = int(info.get("total", 0) or 0)
        eta_m, eta_sec = divmod(int(eta_s), 60)
        eta_str = f"{eta_m:02d}:{eta_sec:02d}"
        if hasattr(self, "lbl_speed_meter") and self.lbl_speed_meter is not None:
            self.lbl_speed_meter.setText(
                f"⚡ Hız: {fps:.1f} kare/sn · Kalan: {eta_str} · {p} / {t}"
            )
        self._set_yolo_loading_detail(f"Otomatik etiketleme: {p}/{t}")
        self._set_yolo_loading_progress(p, t)

    def _on_turbo_relabel_finished(self, res: Dict[str, Any]) -> None:
        self._turbo_relabel_running = False
        if getattr(self, "_yolo_closing", False):
            return
        started = getattr(self, "_turbo_relabel_started_at", 0.0) or time.monotonic()
        elapsed = time.monotonic() - started
        kind = str(getattr(self, "_turbo_relabel_kind", "filtered"))
        self._hide_yolo_loading()
        processed = int(res.get("processed", 0) or 0)
        if kind == "unlabeled_suspicious":
            self._append_log(
                f"[TURBO] Etiketsiz/şüpheliler tamamlandı: {processed} kare. Hız: {res.get('fps', 0.0)} kare/sn"
            )
        else:
            self._append_log(
                f"[TURBO] Filtredekiler tamamlandı: {processed} kare. ({elapsed:.1f} sn)"
            )
        # P2-20: sonda senkron `_reload_review_queue` worker bitişine
        # bağlandı — kuyruk async yükleme worker'ıyla diskten yenilenir
        # (UI thread taramaz).
        self._request_queue_reload()
        if kind == "filtered":
            self._update_filter_action_buttons()
        if kind == "unlabeled_suspicious":
            if not getattr(self, "_auto_confirm_dialogs", False):
                QMessageBox.information(
                    self,
                    "Otomatik Etiketleme Tamamlandı",
                    f"İşlenen Kare: {processed}\n"
                    f"Çıkarılan Kutu: {res.get('total_boxes', 0)}\n"
                    f"Hız: {res.get('fps', 0.0)} kare/sn\n"
                    f"Toplam Süre: {elapsed:.1f} sn",
                )

    def _on_turbo_relabel_failed(self, message: str) -> None:
        self._turbo_relabel_running = False
        if getattr(self, "_yolo_closing", False):
            return
        logger.warning("[TURBO] otomatik etiketleme başarısız: %s", message)
        self._hide_yolo_loading()
        self._append_log(f"[TURBO] Hata: {message}")
        if not getattr(self, "_auto_confirm_dialogs", False):
            QMessageBox.warning(self, "Otomatik etiketleme başarısız", str(message))

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        # [P2-19] Üçüncü tetik yolu da aynı guard'ı görür — eskiden burada
        # değiştirici kontrolü yoktu (Ctrl+A/Alt+A onaylıyordu) ve
        # `isAutoRepeat` kontrolü yoktu.
        if event is not None and self._review_shortcuts_allowed(event):
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

        # Train/Val sızıntısı gerçekten ölçülür: iki tarafta da aynı stem
        # varsa GEÇTİ basılmaz (F-0698).
        _img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        _train_stems: Set[str] = set()
        _val_stems: Set[str] = set()
        for _split, _acc in (("train", _train_stems), ("val", _val_stems)):
            _d = self._dataset_dir() / "images" / _split
            if _d.is_dir():
                try:
                    _acc.update(
                        p.stem for p in _d.iterdir() if p.is_file() and p.suffix.lower() in _img_exts
                    )
                except OSError:
                    pass
        _leaked = sorted(_train_stems & _val_stems)
        if _leaked:
            self.health_host.addWidget(
                self._health_row(
                    "!",
                    "bad",
                    f"Train / Val sızıntısı VAR ({len(_leaked)} kare)",
                    "Aynı kare iki tarafta da var: " + ", ".join(_leaked[:4]),
                    lambda: self._go_step(2),
                    "Onay adımına git",
                    tag="KALDI",
                )
            )
        else:
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
        hl.setContentsMargins(16, 12, 16, 12)
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
        p.setStyleSheet(f"color:{C['muted']};font-size:13px;")
        text.addWidget(p)
        hl.addWidget(ico)
        hl.addLayout(text, stretch=1)
        if tag:
            t = QLabel(tag)
            t.setStyleSheet(
                f"color:#7CFFB2;background:{C['green_soft']};border:1px solid #1E4D33;"
                "border-radius:999px;padding:5px 10px;font-size:12px;font-weight:800;"
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

    def _only_approved_enabled(self) -> bool:
        """Tek doğruluk kaynağı: 'Sadece Onaylananlarla Eğit' üç ayrı kontrolde
        tutulur (Adım 2 checkbox, Hazırla toggle, Eğit toggle); herhangi biri
        açıksa onaylı-havuz modu etkindir (F-0603)."""
        for attr in (
            "chk_only_approved",
            "toggle_approved_only_prepare",
            "toggle_approved_only_train",
            "toggle_approved_only",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    if widget.isChecked():
                        return True
                except Exception:
                    continue
        return False

    def _freeze_dataset(self) -> None:
        # P1-09 (K-24 / F060-K3): dondurma sırasındaki onay kaçağı +
        # çift tıklama. Toplu onay sürerken dondurma başlayamaz (onay,
        # dondurmanın hazırladığı havuza sessizce girerdi); dondurma
        # sürerken ikinci tıklama yoksayılır. Buton kilidi, senkron
        # çalışmada kuyruğa giren ikinci tıklamanın sinyal üretmesini
        # engeller; bayrak da programatik çift çağrıyı keser.
        if getattr(self, "_bulk_approve_running", False):
            self._append_log(
                "[DONDURMA] RED: toplu onay sürüyor — onay bitmeden dondurma başlatılamaz."
            )
            return
        if getattr(self, "_freeze_running", False):
            self._append_log("[BİLGİ] Dondurma zaten sürüyor — ikinci tıklama yoksayıldı.")
            return
        self._freeze_running = True
        if hasattr(self, "btn_freeze"):
            try:
                self.btn_freeze.setEnabled(False)
            except Exception:
                pass
        try:
            self._freeze_dataset_impl()
        finally:
            self._freeze_running = False
            if hasattr(self, "btn_freeze"):
                try:
                    self.btn_freeze.setEnabled(True)
                except Exception:
                    pass

    def _freeze_dataset_impl(self) -> None:
        # B-11 (dilim 24): manifesto hangi seti anlattığını söyler —
        # onaylı modda ham dataset sayıları değil, havuz sayıları yazılır.
        freeze_source = "dataset"
        freeze_pool_counts: Optional[Dict[str, Any]] = None
        if self._only_approved_enabled():
            try:
                from dataset_factory.approved_pool import get_approved_pool
                val_ratio = 0.15
                if hasattr(self, "split_range"):
                    val_ratio = max(0.05, min(0.40, (100 - self.split_range.value()) / 100.0))
                pool = get_approved_pool(self._dataset_dir())
                # B-07 (dilim 27): kaydırıcı eskiden uygulanmıyordu —
                # `force_rebalance` yalnızca testlerde geçiyordu, havuz
                # eski split'leriyle kalıyor, log yine de "val_oranı=%16"
                # yazıyordu. Artık dondurma anında kaydırıcı TEK kaynaktır:
                # havuz bu oranla yeniden dengelenir.
                pool_res = pool.prepare_approved_training_dataset(
                    val_ratio=val_ratio, force_rebalance=True
                )
                freeze_source = "approved_pool"
                freeze_pool_counts = dict(pool_res or {})
                # Havuz `train_count/val_count` döner; eski log
                # `train_samples/val_samples` okuyup hep 0 yazıyordu.
                self._append_log(
                    f"[ONAYLI HAVUZ] Onaylı eğitim veri seti hazırlandı "
                    f"(train={pool_res.get('train_count', 0)} · val={pool_res.get('val_count', 0)} · val_oranı=%{int(val_ratio*100)})."
                )
            except Exception as exc:
                logger.warning("ApprovedDatasetPool prepare failed in _freeze_dataset: %s", exc)
                self._append_log("[DONDURMA] RED: onaylı havuz hazırlanamadı — dondurma iptal.")
                self._go_step(2)
                return
            # Onaylı-havuz yolu da QA kapısından geçer (F-0542): hazırlanan
            # onaylı set doğrulanmadan dondurulmaz. Kapı HAZIRLANAN seti
            # denetler — ham veri seti dizinini değil: ham dizinde .txt
            # etiket olmayabilir, hazırlanan havuzda etiketler vardır.
            try:
                from dataset_factory.label_trust import is_dataset_ready_for_training
                approved_root = Path(getattr(pool, "pool_root", None) or (Path(self._dataset_dir()) / "approved"))
                ready, msg = is_dataset_ready_for_training(approved_root)
                if not ready:
                    self._append_log(f"[DONDURMA] RED (QA Doğrulama, onaylı havuz): {msg}")
                    self._go_step(2)
                    return
            except Exception as exc:
                # B-09: QA kapısı istisnada eskiden sessizce geçiliyordu
                # (yalnızca warning + dondurmaya devam). Kapının kendisi
                # bozulduysa dondurma durur — bozuk veri seti dondurulamaz.
                logger.warning("QA check failed in _freeze_dataset (approved-only): %s", exc)
                self._append_log("[DONDURMA] RED: QA doğrulaması çalışmadı (hata) — dondurma iptal.")
                self._go_step(2)
                return
            # Bekleyen kareler onaylı-havuz eğitimine girmez; sayıları bilgi
            # olarak loglanır (normal yolda olduğu gibi engel değildir, çünkü
            # bu kareler bu dondurmada zaten eğitim dışı bırakılır).
            # P0-4 (K-20 delta): havuza girmeyen yalnızca engel sayılan
            # karelerdir — el etiketli korumalı kareler dondurmayla havuza
            # girer (faz 1), o yüzden sayaç `_pending_train_blockers`.
            pending = self.count_review_state()["blockers"]
            if pending > 0:
                self._append_log(
                    f"[DONDURMA] Bilgi: onay kuyruğunda {pending} bekleyen kare var; "
                    "onaylı-havuz dondurmasına dahil edilmediler "
                    "(el etiketli kareler dondurmayla havuza girer)."
                )
        else:
            # P0-4 (K-20 delta): dondurma kapısı da sarı buton kapısıyla
            # aynı sayacı kullanır — el etiketli korumalı kareler (sarmalın
            # nedeni) dondurmayı kilitlemez; yalnızca gerçekten bekleyen
            # kareler engeldir.
            pending = self.count_review_state()["blockers"]
            if pending > 0:
                self._append_log(
                    f"[DONDURMA] RED: Onay kuyruğunda {pending} bekleyen kare var "
                    "(el etiketleri sayılmaz — onlar dondurmayla havuza girer). "
                    "Eğitimden önce kalan kareler insan onayından geçmelidir (Adım 2)."
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
            except Exception as exc:
                # B-09 (ham yol): `except Exception: pass` ile QA kapısı
                # sessizce geçiliyordu. Kapı bozulduysa dondurma durur.
                logger.warning("QA check failed in _freeze_dataset (raw): %s", exc)
                self._append_log("[DONDURMA] RED: QA doğrulaması çalışmadı (hata) — dondurma iptal.")
                self._go_step(2)
                return

        stats = self._collect_live_stats()
        if freeze_pool_counts is not None:
            # Onaylı modda manifesto havuzu anlatır (B-11): ham dataset
            # sayılarıyla eğitim seti farklı şeylerdi.
            payload = {
                "version": "dataset-v4",
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "source": freeze_source,
                "total_frames": int(freeze_pool_counts.get("total_approved_images", 0)),
                "train_frames": int(freeze_pool_counts.get("train_count", 0)),
                "val_frames": int(freeze_pool_counts.get("val_count", 0)),
                "total_boxes": int(freeze_pool_counts.get("total_boxes", 0)),
                "class_counts": {str(k): int(v) for k, v in (freeze_pool_counts.get("classes") or {}).items()},
                "split_train_pct": int(self.split_range.value()),
                "run_id": int(self._pipeline.get("run_id", 1)),
            }
        else:
            payload = {
                "version": "dataset-v4",
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "source": freeze_source,
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
        # B12: dondurma bayrakları sıfırlıyor ama model_stale çağırmıyordu —
        # bot tarafı is_model_stale() False dönerken dialog publish
        # checklist'i stale görüyordu. Dondurma, önceki eğitimi geçersiz
        # kılar: stale bayrağı disk manifestiyle birlikte yazılır.
        self._mark_model_stale("Veri seti donduruldu — önceki eğitim geçersiz")
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
        # Önceki run'dan ölçülmüş epoch süresi varsa onu kullan (F-0699);
        # yoksa sabit katsayılarla TAHMİNİ süre gösterilir, ölçüm gibi değil.
        measured_sec_per_epoch = None
        try:
            _last = self._last_train_result or {}
            _el = float(_last.get("elapsed_sec", 0.0) or 0.0)
            _ep = int(_last.get("epochs_completed", _last.get("epochs", 0)) or 0)
            if _el > 0 and _ep > 0:
                measured_sec_per_epoch = _el / _ep
        except (TypeError, ValueError):
            measured_sec_per_epoch = None
        if measured_sec_per_epoch is not None:
            sec_per_epoch = measured_sec_per_epoch
            eta_kind = "ölçülen"
        else:
            sec_per_epoch = 6.0
            if "11s" in model_str:
                sec_per_epoch = 16.0
            elif "11m" in model_str or "11l" in model_str or "11x" in model_str:
                sec_per_epoch = 85.0
            sec_per_epoch *= (imgsz_val / 640.0) ** 2
            eta_kind = "tahmini (ölçülmedi)"
        total_sec = epochs * sec_per_epoch
        if total_sec < 3600:
            eta_str = f"~{max(1, int(round(total_sec / 60)))} dk ({eta_kind})"
        else:
            hrs = total_sec / 3600.0
            eta_str = f"~{hrs:.1f} saat ({eta_kind})"
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
        per_class_weak = result.get("per_class_ap50") if isinstance(result.get("per_class_ap50"), dict) else {}
        weak_id = None
        weak_ap = None
        weak_n = 10**9
        for spec in YOLO_12_CLASS_SPECS:
            cid = int(spec["id"])
            raw_ap = per_class_weak.get(str(cid), per_class_weak.get(cid, per_class_weak.get(spec.get("key"))))
            if raw_ap is not None:
                try:
                    ap = float(raw_ap)
                except (TypeError, ValueError):
                    ap = None
                if ap is not None and (weak_ap is None or ap < weak_ap):
                    weak_ap = ap
                    weak_id = cid
        if weak_id is None:
            # AP yoksa en az kutulu sınıf (F-0700: AP varken en düşük AP kazanır).
            for cid, n in counts.items():
                if int(n) < weak_n:
                    weak_n = int(n)
                    weak_id = cid
        if weak_id is not None:
            spec = CLASS_ID_TO_SPEC.get(int(weak_id), {})
            weak_txt = str(spec.get("label_tr", weak_id))
            if weak_ap is not None:
                weak_txt += f" (AP50 %{weak_ap * 100:.1f})"
            self.val_weak.setText(weak_txt)
        while self.ap_list_host.count():
            item = self.ap_list_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        per_class = result.get("per_class_ap50") if isinstance(result.get("per_class_ap50"), dict) else {}
        if per_class:
            for spec in YOLO_12_CLASS_SPECS:
                cid = int(spec["id"])
                n = int(counts.get(cid, 0))
                raw_ap = per_class.get(str(cid), per_class.get(cid, per_class.get(spec.get("key"), None)))
                if raw_ap is None:
                    row = QLabel(f"{spec['label_tr']}: AP50 — · {n} kutu")
                else:
                    ap = float(raw_ap)
                    row = QLabel(f"{spec['label_tr']}: AP50 %{ap * 100:.1f} · {n} kutu")
                row.setStyleSheet("font-family:Consolas;padding:6px 0;")
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
            empty.setStyleSheet("color:#9fb2c3;padding:8px 0;")
            self.ap_list_host.addWidget(empty)
        self.val_empty.hide()
        self.val_body.show()
        # Doğrulama ekranını açmak trained=True yapmaz; yalnızca _finish_validate
        # gerçek doğrulama sonrası trained/validated işaretler (F-0543).
        # B-02: mAP eşiği — eskiden `(trained or has_weights) and fresh`
        # yeterliydi; mAP50=0 bile ağırlık varsa geçiyordu.
        map_ok = map50 >= MIN_PUBLISH_MAP50
        can_advance = (bool(self._pipeline.get("trained")) or has_weights) and fresh and map_ok
        if not map_ok:
            self._append_log(
                f"[DOĞRULA] RED: mAP50 %{map50 * 100:.1f} kabul eşiğinin "
                f"(%{MIN_PUBLISH_MAP50 * 100:.0f}) altında — doğrulama ilerlemez, "
                "Adım 4'te eğitimi yenile."
            )
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

        # B-02/B-03: _finish_validate eskiden yalnızca ağırlık+tazelik
        # bakıyor, _accept_train_result'ı hiç çağırmıyordu — reddedilmiş
        # bir eğitim Adım 5'ten yürüyerek doğrulanabiliyordu. Artık mAP
        # eşiği + epoch bütünlüğü burada da kapıdır.
        # P1-11 (K-26): metrikler HER ZAMAN artifact dosyasından okunur —
        # bellek (`_last_train_result`) eski/kabul edilmiş bir run'a ait
        # olabilir; reddedilen run'ın ağırlıkları o metriklerle kapıdan
        # geçemez. `result` yalnızca artifact'tır.
        art = self._load_train_metrics_from_artifacts()
        result = dict(art) if art else {}
        map50 = float(result.get("mAP50", 0.0) or 0.0)
        if map50 < MIN_PUBLISH_MAP50:
            self._append_log(
                f"[DOĞRULA] RED: mAP50 %{map50 * 100:.1f} kabul eşiğinin "
                f"(%{MIN_PUBLISH_MAP50 * 100:.0f}) altında. Adım 4'e dön."
            )
            self._go_step(4)
            return
        # Tamamlanan epoch artifact'tan (`results.csv` satır sayısı),
        # istenen epoch run'ın `args.yaml`'ından okunur; `args.yaml`
        # yoksa artifact'taki istenen-sayı alanına düşülür, o da yoksa
        # kapı atlanır (eski davranış — örn. args.yaml'sız fixture).
        # Erken durdurma muafiyeti yalnızca AYNI run'a ait kabul edilmiş
        # sonuçta geçerlidir.
        art_run_dir = None
        _arts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
        if str(_arts.get("run_dir") or ""):
            art_run_dir = Path(str(_arts["run_dir"]))
        ep_done = int(result.get("epochs_completed", result.get("epoch", 0)) or 0)
        ep_req = self._read_run_requested_epochs(art_run_dir)
        if ep_req <= 0:
            ep_req = int(result.get("epochs_requested", result.get("epochs", 0)) or 0)
        if ep_req > 0 and ep_done < ep_req and not self._same_run_early_stopped(art_run_dir):
            self._append_log(
                f"[DOĞRULA] RED: yarım eğitim ({ep_done}/{ep_req} epoch). Adım 4'e dön."
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
        self._pipeline["stale_at"] = ""
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
                + "padding:12px 18px;border-bottom:1px solid #262D3A;font-size:13px;"
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
            hl.setContentsMargins(0, 8, 0, 8)
            text = QVBoxLayout()
            text.addWidget(QLabel(f"<b>{title}</b>"))
            s = QLabel(sub)
            s.setStyleSheet("color:#B7BFCB;font-size:12px;")
            text.addWidget(s)
            hl.addLayout(text, stretch=1)
            if live:
                tag = QLabel("CANLI")
                tag.setStyleSheet(
                    "color:#111;background:#F6B81E;border-radius:999px;padding:4px 10px;"
                    "font-size:11px;font-weight:800;"
                )
                hl.addWidget(tag)
            self.version_list.addWidget(row)

        while self.run_list_host.count():
            item = self.run_list_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        run_id = int(self._pipeline.get("run_id", 1))
        # Önceki run'ın modeli bilinmez; güncel model adıyla etiketlenmez (F-0776).
        for rid, model_txt, note in (
            (run_id, str(self.hp_model.currentText()), "son"),
            (max(1, run_id - 1), "bilinmiyor", "önceki"),
        ):
            lab = QLabel(f"#{rid} · {model_txt} · {note}")
            lab.setStyleSheet("font-family:Consolas;padding:6px 0;")
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

        # B-02/B-03: yayın kapısında da mAP kanıtı şart — doğrulama
        # baypası (yeniden açılış / doğrudan yayın) eşik altı modeli
        # geçiremesin.
        pub_result = dict(self._last_train_result or {})
        pub_map50 = float(pub_result.get("mAP50", 0.0) or 0.0)
        if pub_map50 <= 0:
            art = self._load_train_metrics_from_artifacts()
            if art:
                pub_result = dict(art)
                pub_map50 = float(art.get("mAP50", 0.0) or 0.0)
        if pub_map50 < MIN_PUBLISH_MAP50:
            self._append_log(
                f"[YAYIN] RED: mAP50 %{pub_map50 * 100:.1f} kabul eşiğinin "
                f"(%{MIN_PUBLISH_MAP50 * 100:.0f}) altında — yayın durduruldu. Adım 4'e dön."
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

        # B5: _artifacts_are_fresh_vs_freeze yalnızca results.csv/best.pt
        # mtime'ına bakar — eski run'dan kalma bir ONNX taze sanılıp
        # yayınlanabilirdi. Dondurmadan eski kaynak ONNX atılır; taze
        # best.pt'den export zorlanır. (ONNX tazeliğini bu fonksiyona
        # eklemek validate akışını kilitlerdi: yayın öncesi ONNX hep eski
        # döngüden kalır — doğru yer burası.)
        if src_onnx is not None and src_onnx.is_file():
            try:
                freeze_mtime = float(self._freeze_mtime())
                if freeze_mtime > 0 and float(src_onnx.stat().st_mtime) < (freeze_mtime - 1.0):
                    self._append_log(
                        f"[YAYIN] UYARI: kaynak ONNX dondurmadan eski ({src_onnx.name}) — "
                        "taze best.pt'den yeniden export edilecek."
                    )
                    src_onnx = None
                    if src is not None and src.is_file():
                        try:
                            from ultralytics import YOLO
                            yolo_m = YOLO(str(src))
                            sz = int(str(self.hp_imgsz.currentText() or "640")) if hasattr(self, "hp_imgsz") else 640
                            exp_onnx = yolo_m.export(format="onnx", imgsz=sz)
                            if exp_onnx and Path(exp_onnx).is_file():
                                src_onnx = Path(exp_onnx)
                                self._append_log(f"[YAYIN] ONNX export tamamlandı: {src_onnx}")
                        except Exception as exp_exc:
                            logger.warning("Yayın sırasında ONNX re-export: %s", exp_exc)
                            self._append_log(f"[YAYIN] ONNX re-export uyarısı: {exp_exc}")
            except OSError:
                pass

        if src is None and src_onnx is None and not onnx_dest.is_file():
            self._append_log("[YAYIN] RED: Yayınlanacak model (best.pt veya onnx) bulunamadı.")
            return

        import shutil
        best_published = False
        onnx_published = False
        try:
            # Publish best.pt with backup
            if src is not None and src.is_file():
                if best.is_file() and src.resolve() != best.resolve():
                    if prev.is_file():
                        prev.unlink()
                    shutil.copy2(best, prev)
                if src.resolve() != best.resolve():
                    shutil.copy2(src, best)
                best_published = bool(best.is_file())

            # Publish yolov8n_gatherion.onnx with backup (exclusively loaded by reflex_layer & wrong_node_scan)
            if src_onnx is not None and src_onnx.is_file():
                if onnx_dest.is_file() and src_onnx.resolve() != onnx_dest.resolve():
                    if onnx_prev.is_file():
                        onnx_prev.unlink()
                    shutil.copy2(onnx_dest, onnx_prev)
                if src_onnx.resolve() != onnx_dest.resolve():
                    shutil.copy2(src_onnx, onnx_dest)
                onnx_published = bool(onnx_dest.is_file())
                self._append_log(
                    f"[YAYIN] ONNX modeli bot inference yoluna aktarıldı: {onnx_dest.name} "
                    f"(yedek: {onnx_prev.name if onnx_prev.is_file() else 'yok'})"
                )
            elif onnx_dest.is_file():
                # Eski ONNX yerinde duruyor diye 'yayınlandı' sayılmaz (F-0611);
                # yeni ONNX üretilmeden/kopyalanmadan yayın tamamlanmaz.
                self._append_log(
                    f"[YAYIN] Yeni ONNX yok; mevcut bot ONNX ({onnx_dest.name}) yayın sayılmaz."
                )

            if not onnx_published:
                # B-05: best.pt tek başına yayın sayılmaz — bot inference
                # yolunu yalnızca yolov8n_gatherion.onnx besler. ONNX
                # üretilemeden/kopyalanmadan yayın "başarılı" sayılmaz;
                # eski ONNX yerinde duruyor diye de sayılmaz (F-0611).
                self._append_log(
                    "[YAYIN] RED: yeni ONNX yayınlanamadı (export/kopya "
                    "başarısız) — best.pt tek başına yayın sayılmaz, bot "
                    "eski modelle çalışmaya devam eder."
                )
                return
            if not onnx_published:
                # B5: published=True taze ONNX ister — bot inference'ı
                # ONNX'u kullanır (reflex_layer / wrong_node_scan). Yalnızca
                # best.pt yayınlanıp published=True demek, botun eski
                # ONNX'la çalışmaya devam etmesi demekti (F-0611).
                self._append_log(
                    "[YAYIN] RED: taze ONNX yayınlanamadı — published işaretlenmedi. "
                    "ONNX export başarısızsa Adım 4'te eğitimi tekrarla."
                )
                return
            self._pipeline["published"] = True
            self._pipeline["run_id"] = int(self._pipeline.get("run_id", 1)) + 1
            if hasattr(self, "_mark_step_done") and callable(self._mark_step_done):
                try:
                    self._mark_step_done(6)
                except Exception:
                    pass
            published_parts = []
            if best_published:
                published_parts.append("best.pt")
            if onnx_published:
                published_parts.append("yolov8n_gatherion.onnx")
            self._append_log(
                f"[YAYIN] Yayınlandı ({', '.join(published_parts)}) — bot inference reflex_layer hazır. kaynak={src or src_onnx}"
            )
            # F8: yayınlanan modelin metrikleri diskte kayda geçer — "bu
            # modelin mAP'si kaçtı, kaç epoch eğitildi" sorusu sonradan
            # diskten cevaplanabilir (eskiden yalnızca run dizinindeki
            # results.csv'de kalıyordu).
            try:
                _pub_arts = pub_result.get("artifacts") if isinstance(pub_result.get("artifacts"), dict) else {}
                _metrics_payload = {
                    "published_at": datetime.now().isoformat(timespec="seconds"),
                    "mAP50": float(pub_map50),
                    "mAP50_95": float(pub_result.get("mAP50_95", 0.0) or 0.0),
                    "precision": float(pub_result.get("precision", 0.0) or 0.0),
                    "recall": float(pub_result.get("recall", 0.0) or 0.0),
                    "epochs_completed": int(pub_result.get("epochs_completed", pub_result.get("epoch", 0)) or 0),
                    "epochs_requested": int(pub_result.get("epochs_requested", pub_result.get("epochs", 0)) or 0),
                    "min_map50": MIN_PUBLISH_MAP50,
                    "data_yaml_path": str(pub_result.get("data_yaml_path") or ""),
                    "run_dir": str(_pub_arts.get("run_dir") or (run_dir or "")),
                    "best_pt": str(src or ""),
                    "onnx": str(onnx_dest),
                }
                (models / "published_model_metrics.json").write_text(
                    json.dumps(_metrics_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("published_model_metrics yazılamadı: %s", exc)
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
            # B-15: rollback eskiden pipeline bayraklarına dokunmuyor,
            # disk durumunu kaydetmiyor, `_last_train_result`'ı
            # güncellemiyordu — diskte artık eski model varken arayüz ve
            # kayıtlı durum "yeni model yayında" göstermeye devam ediyordu.
            # Geri alınan model canlıdır ama güncel dondurmaya karşı
            # doğrulama kanıtı yoktur: trained/validated düşer, yayın
            # bayrağı önceki modelin yayında olduğunu söyler.
            self._last_train_result = {}
            self._pipeline["train_completed_at"] = ""
            self._pipeline["trained"] = False
            self._pipeline["validated"] = False
            self._pipeline["published"] = True  # önceki model yayında
            self._save_pipeline_disk_state()
            self._append_log(
                "[YAYIN] Önceki model ve ONNX yedeğine dönüldü — canlı model "
                "artık önceki yayındır; doğrulama kanıtı sıfırlandı, Adım 5'te "
                "yeniden doğrulayın."
            )
            self._refresh_publish_checklist()
        except Exception as exc:
            self._append_log(f"[HATA] Rollback: {exc}")

    def _update_pending_labels(self) -> None:
        # P3-28: pending tek kaynaktan (hi sayacı kapsam dışı — F074).
        pending = self.count_review_state()["pending"]
        val_lbl = self.__dict__.get("stat_pending_value")
        if val_lbl is not None:
            val_lbl.setText(f"{pending} kare")
        # P1-12 (K-31/F014-02): sayaç kaydırıcıyı okur — sabit 0.90 yok.
        th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        hi = sum(
            1
            for it in self._review_items
            if (not it.get("done")) and self._frame_conf(it) >= th
        )
        sub_lbl = self.__dict__.get("stat_pending_sub")
        if sub_lbl is not None:
            sub_lbl.setText(
                f"{hi} kare güven ≥ {th:.2f} — toplu onaylanabilir" if pending else "Kuyruk boş"
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
        for target in (self.factory_controller, self._get_builtin_helper()):
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

    def _collect_live_stats(
        self, force: bool = False, disk_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        now = time.monotonic()
        if (
            not force
            and self._stats_cache is not None
            and (now - self._stats_cache_at) < self._stats_cache_ttl_sec
        ):
            return self._stats_cache

        if disk_stats is None:
            # P2-20: worker'ın tohumladığı GPU bilgisi varsa probe etme
            # (helper 30 sn TTL'li önbelleğiyle de korunur).
            disk_stats = self._get_builtin_helper().get_dataset_stats(
                gpu_environment=getattr(self, "_last_worker_gpu_env", None)
            )
        ctrl = self.factory_controller
        ctrl_stats: Dict[str, Any] = {}
        if ctrl is not None and ctrl is not self._get_builtin_helper():
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
            # P2-23 (D061-K1): bilinmeyen-sınıf sayımı disk istatistiğinden
            # aynen geçer — web önizleme total_boxes'u sınıf toplamıyla
            # karşılaştırır (export_web_state).
            "total_label_lines": int(disk_stats.get("total_label_lines", 0) or 0),
            "unknown_class_boxes": int(disk_stats.get("unknown_class_boxes", 0) or 0),
            "unknown_class_ids": list(disk_stats.get("unknown_class_ids") or []),
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
        _builtin = self._get_builtin_helper()
        if _builtin is not self.factory_controller:
            controllers.append(_builtin)
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

    def refresh_dashboard(
        self, force: bool = False, *args: Any, _skip_queue_reload: bool = False
    ) -> Dict[str, Any]:
        force = bool(force)
        if self._lightbox_open and not force:
            return self._stats_cache or {}
        if not force and getattr(self, "_yolo_load_running", False):
            # P2-20 (F051-K2): yükleme sürerken 4 sn'lik zamanlayıcı
            # dashboard'u tazelemesin — yükleme worker'ı bitmeden UI
            # thread'de `get_dataset_stats()` → GPU probe çalışmasın.
            # Yükleme katmanı zaten ekranda; sıfır istatistik döner,
            # yükleme bitince `force=True` yolu gerçek sayıları yazar.
            return {
                "total_frames": 0,
                "train_frames": 0,
                "val_frames": 0,
                "total_boxes": 0,
                "class_counts": {spec["id"]: 0 for spec in YOLO_12_CLASS_SPECS},
                "latest_preview_path": "",
                "gpu_environment": getattr(self, "_last_worker_gpu_env", None) or {},
            }

        stats = self._collect_live_stats(force=force)
        # P3-28: dashboard sayıları tek kaynaktan — stats_sig de taze
        # pending'i görür (eskiden önbellek okunuyordu, bayat kalabiliyordu).
        _review_state = self.count_review_state()
        total_frames = int(stats["total_frames"])
        train_frames = int(stats["train_frames"])
        val_frames = int(stats["val_frames"])
        total_boxes = int(stats["total_boxes"])
        class_counts: Dict[int, int] = stats["class_counts"]
        gpu_env = stats.get("gpu_environment") or {}

        stats_sig = (total_frames, train_frames, val_frames, total_boxes, _review_state["pending"])
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
            for spec in YOLO_12_CLASS_SPECS:
                if int(class_counts.get(int(spec["id"]), 0)) <= 0:
                    empty_n += 1
                    empty_names.append(str(spec["label_tr"]))
            self.stat_empty_value.setText(f"{empty_n} / {len(YOLO_12_CLASS_SPECS)}")
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
        # P3-28: pending yukarıdaki count_review_state() ile tazelendi;
        # force + Adım 2'de kuyruk diskten yenilenir (async — sayı yenileme
        # bitince _sync_pending_from_queue ile tazelenir).
        if force and int(self._pipeline.get("step", 0)) == 2 and not _skip_queue_reload:
            try:
                self._request_queue_reload()
            except Exception as exc:
                self._append_log(f"[KUYRUK] Yenileme hatası: {exc}")

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
        return _resolve_review_image_path(self._dataset_dir(), split, stem)

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
        # P3-28: çubukla aynı sayılar — tek kaynak count_review_state()
        # (refresh_dashboard zaten çağırdı; buradaki çağrı pipeline'ın
        # arada değişmiş olmasına karşı tazeliği garantiler).
        review_state = self.count_review_state()
        pending = review_state["pending"]
        reviewed = review_state["reviewed"]
        counts = {int(k): int(v) for k, v in (stats.get("class_counts") or {}).items()}
        empty_names: List[str] = []
        for spec in YOLO_12_CLASS_SPECS:
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
                    # P2-23 (D061/D063-K1): web kuyruk satırı emin kutu
                    # sayısını da taşır — TXT-only el etiketli kare
                    # "0 zayıf" diye boş görünmez.
                    "n_trusted": len(it.get("trusted") or []),
                    "done": it.get("done"),
                }
            )

        current_review: Optional[Dict[str, Any]] = None
        if self._review_items:
            it = self._review_items[min(self._review_index, len(self._review_items) - 1)]
            img_path = self._resolve_review_image(str(it.get("split", "train")), str(it.get("stem", "")))

            def _web_box(b: Any, trusted_flag: bool) -> Optional[Dict[str, Any]]:
                # P2-23 (D061-K1 + Y002-K1): trusted kutular da ihraç
                # edilir (eskiden yalnızca weak vardı — TXT-only el
                # etiketli karede "Kutu yok" yalanı çıkıyordu). Geometrisi
                # eksik kutuya şablon (0.5/0.5/0.08) uydurulmaz — atlanır
                # ve sayılır; `trusted` alanı yoksa False sayılır.
                if not isinstance(b, dict):
                    return None
                if any(b.get(k) is None for k in ("cx", "cy", "w", "h")):
                    return None
                try:
                    return {
                        "class_name": str(b.get("class_name", b.get("key", "?"))),
                        "conf": float(b.get("confidence", b.get("conf", 0.0)) or 0.0),
                        "cx": float(b.get("cx")),
                        "cy": float(b.get("cy")),
                        "w": float(b.get("w")),
                        "h": float(b.get("h")),
                        "trusted": bool(trusted_flag),
                    }
                except (TypeError, ValueError):
                    return None

            weak = [
                wb for wb in (
                    _web_box(b, False) for b in it.get("weak") or []
                ) if wb is not None
            ]
            trusted = [
                tb for tb in (
                    _web_box(b, True) for b in it.get("trusted") or []
                ) if tb is not None
            ]
            invalid_boxes = (
                len(it.get("weak") or []) + len(it.get("trusted") or [])
                - len(weak) - len(trusted)
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
                # P2-23 (D061/D063-K1): el etiketli (trusted) kutular
                # web önizlemede de çizilir/listelenir.
                "trusted": trusted,
                "n_trusted": len(trusted),
                "invalid_boxes": invalid_boxes,
                "title": f"Kare #{it.get('stem', '')}",
                "sub": (
                    f"{len(trusted)} emin, {len(weak)} zayıf kutu · en düşük güven {float(it.get('conf', 0.0)):.2f}"
                    if (weak or trusted)
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
        # P2-23 (D061-K1): total_boxes sınıf toplamıyla karşılaştırılır —
        # eskiden diskten körü körüne alınıp hiç karşılaştırılmıyordu.
        # Bilinmeyen-sınıf kutuları total_boxes'a girmez; o yüzden
        # `total_label_lines == total_boxes + unknown_class_boxes`
        # değişmezi de denetlenir.
        class_sum = sum(counts.values())
        unknown_class_boxes = int(stats.get("unknown_class_boxes", 0) or 0)
        total_label_lines = int(stats.get("total_label_lines", 0) or 0)
        box_count_ok = total_boxes == class_sum and (
            total_label_lines == 0
            or total_label_lines == total_boxes + unknown_class_boxes
        )
        if not box_count_ok:
            # İmza değişmeyince tekrar uyarı yok (web push 2,5 sn'de bir koşar).
            box_sig = (total_boxes, class_sum, total_label_lines, unknown_class_boxes)
            if box_sig != getattr(self, "_last_box_count_sig", None):
                self._last_box_count_sig = box_sig
                logger.warning(
                    "[ÖNİZLEME] kutu sayımı uyuşmuyor: total_boxes=%d sınıf "
                    "toplamı=%d satır=%d bilinmeyen-sınıf=%d",
                    total_boxes,
                    class_sum,
                    total_label_lines,
                    unknown_class_boxes,
                )
        goal = int(getattr(self, "target_photo_goal", TARGET_DATASET_PHOTO_GOAL) or TARGET_DATASET_PHOTO_GOAL)
        pct = min(100.0, (float(total_frames) / float(max(1, goal))) * 100.0)

        # P1-12 (K-31/F014-02): sayaç kaydırıcıyı okur — sabit 0.90 yok.
        _hi_th = self._bulk_threshold() if self.__dict__.get("bulk_range") else CONF_BULK_DEFAULT
        hi_bulk = sum(
            1
            for it in self._review_items
            if not it.get("done") and float(it.get("conf", 0.0) or 0.0) >= _hi_th
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
            # P2-23 (D061-K1): bilinmeyen-sınıf sayımı + total_boxes /
            # sınıf-toplamı çapraz kontrolü web katmanına da taşınır.
            "unknownClassBoxes": unknown_class_boxes,
            "unknownClassIds": list(stats.get("unknown_class_ids") or []),
            "boxCountCheck": {
                "total_boxes": total_boxes,
                "class_sum": class_sum,
                "total_label_lines": total_label_lines,
                "unknown_class_boxes": unknown_class_boxes,
                "ok": bool(box_count_ok),
            },
            "statPending": f"{pending} kare",
            "statPendingSub": (
                f"{hi_bulk} kare güven ≥ {_hi_th:.2f} — toplu onaylanabilir" if pending else "Kuyruk boş"
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
            "reviewProgTxt": f"{reviewed} / {review_state['total']}",
            "reviewProgPct": int(
                round(100 * reviewed / max(1, review_state["total"]))
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
        # Alttaki Canlı Günlük şeridi aynı akışı filtreli gösterir.
        line = f"[{stamp}] {message}"
        journal = getattr(self, "_journal_lines", None)
        if journal is not None:
            journal.append(line)
            if len(journal) > 500:
                del journal[: len(journal) - 500]
            box = getattr(self, "live_journal_box", None)
            if box is not None and self._journal_line_visible(line):
                box.appendPlainText(line)
                box.moveCursor(QTextCursor.MoveOperation.End)

    def _mirror_status_to_app_log(self, message: str) -> None:
        # P2-21 (F031-03): dialog logları artık logger.info/error olarak
        # bot.log'a AKMAZ — 172 çağrı noktasının tamamı bot.log'u
        # şişiriyor, her dialog açılışında ~5 "yükleniyor" satırı
        # bırakıyordu. Ayna debug düzeyindedir: dialog log kutusu
        # kaydın kendisi olmaya devam eder; bot.log'a yalnızca
        # DEBUG düzeyinde bakılırsa görünür. Gerçek hatalar zaten
        # kendi logger.* çağrılarıyla (bu fonksiyon dışında) loglanır.
        text = str(message or "").strip()
        if not text:
            return
        logger.debug("[DIALOG] %s", text)

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
        helper = self._get_builtin_helper()

        def _scan(should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
            # P2-8 (DF.9): worker `should_stop` bayrağını geçirir
            # (`_BackgroundCallWorker` imzayı kontrol eder); tarama
            # kare sınırında durur, kapanış SIGABRT'siz tamamlanır.
            result: Dict[str, Any] = {}

            def _maybe_pass_stop(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
                if should_stop is None:
                    return kwargs
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    return kwargs
                if "should_stop" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    kwargs["should_stop"] = should_stop
                return kwargs

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
                    kwargs = _maybe_pass_stop(fn, kwargs)
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
                helper_fn = getattr(helper, "bootstrap_from_existing_frames", None)
                if callable(helper_fn):
                    result = helper_fn(
                        **_maybe_pass_stop(
                            helper_fn, {"augment_multiplier": mult}
                        )
                    )
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
        # [P2-18] Bootstrap worker'ı kapanış ağına kayıtlı: closeEvent /
        # __del__ / atexit aynı desende durdurur.
        _YOLO_BACKGROUND_THREADS.add(self._bootstrap_worker)
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
            cur_stats = self._get_builtin_helper().get_dataset_stats()
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
        # P2-18: kapanış sırasında gelen bitiş sinyali UI'a dokunmaz
        # (özellikle `_reload_review_queue` senkron taraması çalışmaz).
        if getattr(self, "_yolo_closing", False):
            return
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
            final_stats = self._get_builtin_helper().get_dataset_stats()
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
        added = int(payload.get("bootstrapped_frames_added", 0) or 0)
        if added == 0:
            try:
                added = max(
                    0,
                    int(refreshed.get("total_frames", 0))
                    - int(getattr(self, "_bootstrap_start_frames", 0) or 0),
                )
            except (TypeError, ValueError):
                added = 0
        if added > 0:
            self._append_log(
                f"[BAŞARILI] Bootstrap + Augment tamamlandı! "
                f"Kare: {refreshed['total_frames']} (Train: {refreshed['train_frames']}, Val: {refreshed['val_frames']}) · "
                f"Kutu: {refreshed['total_boxes']} · onay kuyruğu {self.count_review_state()['pending']}"
            )
        else:
            self._append_log(
                "[BİLGİ] Bootstrap taraması bitti — yeni kare eklenmedi "
                f"(toplam {refreshed['total_frames']} kare · onay kuyruğu {self.count_review_state()['pending']})."
            )
        if int(self._pipeline["step"]) == 3:
            self._rebuild_health_checks()
        self._save_pipeline_disk_state()

    def _on_bootstrap_worker_failed(self, error_msg: str) -> None:
        # P2-18: kapanış sırasında gelen hata sinyali UI'a dokunmaz.
        if getattr(self, "_yolo_closing", False):
            return
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
        if epochs_done < 1:
            # Diskteki eski best.pt tek başına yeterli kanıt değildir; epoch
            # tamamlanmadan gelen sonuç kabul edilmez (F-0544).
            return False, "Hiç epoch tamamlanmadı (diskteki eski best.pt yeterli değil)"
        # B-04: 33/100 gibi yarım eğitim eskiden kabul ediliyordu.
        # Erken durdurma (patience) hariç: motor `early_stopped=True`
        # işaretlerse eksik epoch meşrudur, yoksa reddedilir.
        if epochs_req > 0 and epochs_done < epochs_req and not result.get("early_stopped"):
            return (
                False,
                f"Yarım eğitim: {epochs_done}/{epochs_req} epoch tamamlandı — "
                "kesintiye uğramış eğitim kabul edilmez",
            )
        # B-02: mAP50 kabul eşiği. Eskiden mAP50=0 bile best.pt varsa
        # geçiyordu; yarım eğitim (mAP50 0.32) bu kapıdan yayınlanıyordu.
        if map50 < MIN_PUBLISH_MAP50:
            return (
                False,
                f"mAP50 %{map50 * 100:.1f} kabul eşiğinin "
                f"(%{MIN_PUBLISH_MAP50 * 100:.0f}) altında — model yayınlanamaz",
            )
        if map50 <= 0 and not has_best:
            return False, "Hiç epoch metriği yok / best.pt yok"
        if has_best and map50 <= 0:
            # best.pt kanıt sayılacaksa dondurma sonrası taze olmalı.
            try:
                best_mtime = float(Path(best_pt).stat().st_mtime)
                freeze_mtime = float(self._freeze_mtime()) if hasattr(self, "_freeze_mtime") else 0.0
            except Exception:
                best_mtime, freeze_mtime = 0.0, 0.0
            if freeze_mtime > 0 and best_mtime < (freeze_mtime - 1.0):
                return False, "best.pt dondurma öncesinden (eski) — yeni eğitim gerekli"
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
            only_approved = self._only_approved_enabled()

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
        # [P2-18] Eğitim worker'ı kapanış ağına kayıtlı: closeEvent /
        # __del__ / atexit aynı desende durdurur.
        _YOLO_BACKGROUND_THREADS.add(worker)
        worker.start()
        return worker

    def _on_training_worker_finished(self) -> None:
        # P2-18: kapanış sırasında gelen bitiş sinyali UI'a dokunmaz.
        if getattr(self, "_yolo_closing", False):
            return
        # P2-22: bitiş kilidi sinyal + zamanlayıcı ortak yolundan tazelenir
        # (bayat senkron bayrak temizlenir, Durdur kapatılır).
        self._refresh_training_locks()

    def on_train_rtx_clicked(self, only_approved: Optional[bool] = None) -> None:
        if only_approved is None:
            only_approved = self._only_approved_enabled()
        is_only_approved = bool(only_approved)

        # P2-22 (bölüm 6.2, F040-K1): senkron yeniden-giriş bayrağı.
        # Aşağıdaki `isRunning()` kontrolü asenkron olduğu için çift tık
        # (ya da yeşil + sarı + btn_next'in aynı anda basılması) ikinci
        # eğitimi başlatıp `self._training_worker` üzerine yazabiliyordu.
        # Bayrak kapı kontrollerinden ÖNCE senkron kurulur ve iki eğitim
        # butonu da aynı anda kilitlenir; kapı RED'inde aşağıda temizlenir,
        # gerçek başlangıçta bitiş sinyalleri + 1 sn zamanlayıcı temizler.
        if getattr(self, "_training_starting", False):
            self._append_log("[BİLGİ] Eğitim başlatma zaten sürüyor — çift tık yoksayıldı.")
            return
        self._training_starting = True
        self.btn_train_rtx.setEnabled(False)
        if hasattr(self, "btn_train_only_approved"):
            self.btn_train_only_approved.setEnabled(False)

        if is_only_approved:
            # B-10: havuz hazırlık hatası eskiden yalnızca loglanıp yola
            # devam ediliyordu — eğitim eski/yanlış setle çalışıyordu.
            # Dondurma zaten havuzu hazırladı; burada yalnızca varlığı
            # doğrulanır, hazırlık başarısızlığı eğitimi durdurur.
            try:
                _app_yaml = Path(self._dataset_dir()) / "approved" / "data_approved.yaml"
                if not _app_yaml.is_file():
                    raise RuntimeError(f"Onaylı eğitim yaml'ı yok: {_app_yaml}")
            except Exception as exc:
                logger.warning("ApprovedDatasetPool check failed in on_train_rtx_clicked: %s", exc)
                self._append_log(
                    f"[EĞİTİM] RED: onaylı eğitim verisi hazır değil ({exc}) — "
                    "önce Adım 3'te dondurun."
                )
                self._go_step(3)
                self._training_starting = False
                self._update_train_lock()
                return
        # P0-4 (K-20 delta): pending kapısı onaylı modda da geçerli — eskiden
        # `elif` yüzünden yeşil buton yalnızca yaml varlığına bakıyor, bekleyen
        # kare olsa bile eğitimi başlatıyordu. Engel varsa eğitim başlamaz;
        # kullanıcıya neden + sayı söylenir.
        # P3-28: kapı sayısı tek kaynaktan (eskiden iki ayrı hesaptı).
        blockers = self.count_review_state()["blockers"]
        if blockers > 0:
            self._append_log(
                f"[EĞİTİM] RED: Onay kuyruğunda {blockers} bekleyen kare var "
                "(Adım 2; el etiketleri sayılmaz — tek tek onayla/reddet)."
            )
            self._go_step(2)
            self._training_starting = False
            self._update_train_lock()
            return

        if not self._pipeline.get("frozen"):
            self._append_log("[EĞİTİM] Önce veri setini dondur (Adım 3).")
            self._go_step(3)
            self._training_starting = False
            self._update_train_lock()
            return

        # P1-09 (K-24): dondurma-sayı kapısı — bayrak `frozen` olsa bile
        # dondurma sonrası onay/budama havuzu değiştirdiyse eğitim
        # başlamaz; kullanıcı Adım 3'te yeniden dondurur. (Motor +
        # worker kapıları da aynı kontrolü yapar; bu, tıklama anındaki
        # kullanıcıya dönük erken uyarıdır.)
        if is_only_approved:
            try:
                from dataset_factory.approved_pool import (
                    check_frozen_manifest_consistency as _check_frozen,
                )

                _frozen_check = _check_frozen(self._dataset_dir())
                if _frozen_check.get("checked") and not _frozen_check.get("ok"):
                    self._append_log(
                        "[EĞİTİM] RED: "
                        + str(_frozen_check.get("reason") or "dondurulan set havuzla uyuşmuyor.")
                    )
                    self._go_step(3)
                    return
            except Exception as exc:
                logger.warning("Dondurma karşılaştırması çalıştırılamadı (tık): %s", exc)

        # P0-5 tur 2: tık-öncesi yaml kapısı KALDIRILDI — kapı bootstrap
        # sonrasına taşındı (`RTXTrainingWorker.run`, oyun kapısından
        # sonra, eğitim çağrısından önce). Gerekçe: bootstrap
        # (`bootstrap_from_existing_frames`) etiketli kareleri yazıp
        # `data.yaml`'ı mutlak yolla yeniden oluşturur; tık anında
        # 0-etiketli diye veto etmek onarımı engelliyordu (boş
        # worktree'de 3 tık testi bu yüzden hiç worker'a ulaşamıyordu).
        # Kapı atlanmadı — eğitim başlamadan önce worker + motor
        # kapıları aynı kontrolü yapar. Non-modal bildirim (8bfb9c0)
        # aynen korunur: burada QMessageBox yoktu, olmayacak.

        if self._training_worker is not None and self._training_worker.isRunning():
            self._append_log("[BİLGİ] RTX CUDA FP16 eğitimi zaten arka planda devam ediyor.")
            self._training_starting = False
            self._update_train_lock()
            return
        if self._bootstrap_worker is not None and self._bootstrap_worker.isRunning():
            self._append_log("[BİLGİ] Veri seti taraması zaten sürüyor.")
            self._training_starting = False
            self._update_train_lock()
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
        if hasattr(self, "btn_train_only_approved"):
            self.btn_train_only_approved.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_progress_bar.setValue(0)
        self.epoch_label.setText(f"Epoch 0 / {epochs}")
        self.lbl_train_status.setText(SCAN_RUNNING_STATUS)
        approved_suffix = " (Sadece Onaylananlar)" if is_only_approved else ""
        self._append_log(
            f"[EĞİTİM] Başlatılıyor: {model} · {epochs} epoch · imgsz={imgsz} · batch={batch}{approved_suffix}"
        )
        self._append_log("Veri seti taraması sürüyor")

        # Bayrak bilinçli olarak True bırakılır: worker çalışırken ikinci
        # başlatma denemesi `_training_starting` + `isRunning()` ile düşer.
        # Başlatma istisnayla yarıda kalırsa bayrak hemen temizlenir
        # (yedeği: 1 sn kilit zamanlayıcısı).
        try:
            self._start_training_worker(epochs=epochs, model=model, imgsz=imgsz, batch=batch, only_approved=is_only_approved)
        except Exception:
            self._training_starting = False
            self._update_train_lock()
            raise

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
        self._refresh_training_locks()

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
        # v19 telemetri: loss yoksa "—" (None) — 0.0000 yanıltıcı sıfırdır.
        box_txt = _format_loss(payload.get("box_loss"))
        cls_txt = _format_loss(payload.get("cls_loss"))
        self.train_progress_bar.setValue(pct)
        self.epoch_label.setText(f"Epoch {epoch} / {epochs}")
        self.epoch_pct.setText(f"%{pct}")
        self.lbl_map50_loss.setText(f"box {box_txt} · cls {cls_txt}")
        self.m_map.setText(f"mAP50 %{map50 * 100:.1f}")
        if epoch == 1 or epoch == epochs or (epochs <= 10) or (epoch % max(1, epochs // 10) == 0):
            self._append_log(
                f"[EPOCH {epoch:02d}/{epochs:02d}] mAP50=%{map50 * 100:.2f} | "
                f"box_loss={box_txt} | cls_loss={cls_txt}"
            )

    def _on_training_finished(self, result: Dict[str, Any]) -> None:
        # P2-18: kapanış sırasında gelen bitiş sinyali UI'a dokunmaz.
        if getattr(self, "_yolo_closing", False):
            return
        ok, reason = self._accept_train_result(dict(result or {}))
        self._refresh_training_locks()
        if not ok:
            # P1-11 (K-26): reddedilen run eski kabul edilmiş run'ın
            # metriklerini bellekten/diskten diriltmesin — `_last_train_result`
            # temizlenir; ardından `_save_pipeline_disk_state` diskteki
            # `last_train_metrics`'i de boş yazar, yeniden açılışta
            # hydrate eski metriği geri getiremez.
            self._last_train_result = {}
            self._pipeline["trained"] = False
            self._pipeline["validated"] = False
            self._pipeline["train_session_ok"] = False
            self._pipeline["train_completed_at"] = ""
            if "durduruldu" in reason.lower() or "stopped" in reason.lower():
                self.lbl_train_status.setText(f"Eğitim Durumu: Durduruldu — {reason}")
                self._append_log(f"[BİLGİ] {reason}")
            else:
                self.lbl_train_status.setText(f"Eğitim Durumu: RED — {reason}")
                self._append_log(f"[HATA] Eğitim kabul edilmedi: {reason}")
            self._save_pipeline_disk_state()
            return

        self._last_train_result = dict(result or {})
        # Tazelik ve ONNX yolu ölçülür, varsayılmaz (F-0690).
        fresh = bool(self._artifacts_are_fresh_vs_freeze())
        onnx_path = self._models_dir() / "yolov8n_gatherion.onnx"
        self._last_train_result["artifact_fresh"] = fresh
        self._last_train_result["metrics_kind"] = "fresh_after_freeze" if fresh else "stale_pre_freeze"
        self._last_train_result["onnx_path"] = str(onnx_path)
        self._last_train_result["onnx_exists"] = bool(onnx_path.is_file())
        if not fresh:
            self._append_log("[EĞİTİM] UYARI: eğitim bitti ama artifact'lar dondurmaya göre taze değil — trained işaretlenmedi.")
        self._pipeline["trained"] = fresh
        self._pipeline["train_session_ok"] = fresh
        self._pipeline["model_stale"] = not fresh
        self._pipeline["train_completed_at"] = datetime.now().isoformat(timespec="seconds")
        ds_dir = self._dataset_dir() if hasattr(self, "_dataset_dir") and callable(self._dataset_dir) else None
        md_dir = self._models_dir() if hasattr(self, "_models_dir") and callable(self._models_dir) else None
        if fresh:
            # B2: disk stale bayrağı yalnızca taze eğitimde temizlenir —
            # eski kod model_stale = not fresh kurup hemen ardından
            # koşulsuz clear_model_stale() çağırıyordu: bayat eğitim
            # bot tarafında "güncel" görünüyordu.
            try:
                from dataset_factory.dataset_validator import clear_model_stale
                clear_model_stale(dataset_dir=ds_dir, models_dir=md_dir)
            except Exception:
                pass
            self._pipeline["stale_reason"] = ""
            self._pipeline["stale_at"] = ""
        else:
            # Bayat eğitimde bayrak korunur: disk manifesti de bellek
            # durumuyla aynı fikirde olmalı (bot is_model_stale() okur).
            try:
                from dataset_factory.dataset_validator import mark_model_stale
                mark_model_stale(
                    "Eğitim artifact'ları dondurmaya göre taze değil",
                    dataset_dir=ds_dir,
                    models_dir=md_dir,
                )
            except Exception:
                pass
            self._pipeline["stale_reason"] = "Eğitim artifact'ları dondurmaya göre taze değil"
            self._pipeline["stale_at"] = datetime.now().isoformat(timespec="seconds")

        map50 = float(result.get("mAP50", 0.0) or 0.0)
        # v19 telemetri: final sonuçta da loss yoksa "—" (eksik anahtar/None).
        box_txt = _format_loss(result.get("box_loss"))
        cls_txt = _format_loss(result.get("cls_loss"))
        epochs = int(result.get("epochs_completed", result.get("epochs", self.hp_epoch.value())) or 0)
        epochs_req = int(result.get("epochs", self.hp_epoch.value()) or epochs)
        self.train_progress_bar.setValue(100)
        self.epoch_label.setText(f"Epoch {epochs} / {epochs_req}")
        self.epoch_pct.setText("%100")
        self.lbl_train_status.setText("Eğitim Durumu: TAMAMLANDI")
        self.lbl_map50_loss.setText(f"box {box_txt} · cls {cls_txt}")
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
        # P2-18: kapanış sırasında gelen hata sinyali UI'a dokunmaz.
        if getattr(self, "_yolo_closing", False):
            return
        # P1-11 (K-26) ile aynı delik: başarısız run da eski kabul edilmiş
        # run'ın metriklerini bellekte bırakmamalı.
        self._last_train_result = {}
        self._pipeline["trained"] = False
        self._pipeline["validated"] = False
        self._pipeline["train_session_ok"] = False
        self._pipeline["train_completed_at"] = ""
        self._refresh_training_locks()
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
            if not finished:
                # P2-18: wait() sonucu yok sayılmaz — zaman aşımı loglanır
                # ve False döner (arayana kalır; bu yöntem durdurma
                # istemez, yalnızca bekler).
                logger.warning(
                    "[YOLO] eğitim worker'ı %d ms'de bitmedi (bekleme zaman aşımı).",
                    int(timeout_ms),
                )
            QApplication.processEvents()
            return bool(finished)
        return True


    def on_open_dataset_folder_clicked(self) -> Path:
        dataset_path = self._get_builtin_helper().dataset_dir
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
                # Ağır tarama worker thread'de; bitince kuyruk yenilenir,
                # mevcut kare korunur (preserve_stem) ve sayı loglanır.
                self._request_queue_reload(preserve_stem=curr_stem or None)
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
            self.btn_refresh.setStyleSheet("font-weight: bold; color: #22c55e; font-size: 13px;")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(
                1200,
                lambda: (
                    self.btn_refresh.setText(orig_text),
                    self.btn_refresh.setStyleSheet("")
                )
            )

        total_frames = int(getattr(self, "_last_stats_sig", (0,))[0]) if hasattr(self, "_last_stats_sig") else 0
        self._append_log(f"[DASHBOARD] Yenilendi: toplam {total_frames} kare, {self.count_review_state()['pending']} onay bekleyen.")

    def get_yolo_factory_dialog_health_summary(self) -> dict:
        # Her durumda ready=True dönülmez; gerçek durum ölçülür (F-0768).
        # S-3501: hesaplanan dataset_dir_exists ready hesabına girer —
        # controller varken dataset klasörü silinmiş/eksikse ready=False olur.
        try:
            ds_dir = self._dataset_dir()
            ds_exists = bool(ds_dir is not None and ds_dir.exists())
            ready = bool(ds_exists and self.factory_controller is not None)
            return {
                "yolo_factory_dialog_ready": ready,
                "dataset_dir_exists": ds_exists,
                "controller": type(self.factory_controller).__name__
                if self.factory_controller is not None
                else None,
            }
        except Exception as exc:
            return {"yolo_factory_dialog_ready": False, "error": str(exc)}

    def __del__(self) -> None:
        # [YOLO-ASYNC] Test fixture'ları dialogu close() etmeden bırakır;
        # GC, hâlâ çalışan QThread'i yok ederse süreç abort olur (exit 134).
        # closeEvent dışındaki yok edilme yolunda da thread'leri durdur.
        # P2-18: closeEvent ile AYNI `_stop_yolo_threads` tek düzeni —
        # closeEvent zaten durdurduysa her thread durmuş olduğundan
        # burası ucuz no-op'tur; koordinasyonsuz üçlü durdurma yok.
        try:
            self._stop_yolo_threads()
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        # [P2-19/F044-02] Kapanışta yalnızca dialogun duraklattığı bot
        # devam ettirilir (bayrak korumalı — hideEvent de ateşlenir, çift
        # resume olmaz).
        self._resume_bot_if_paused_by_factory()
        self._save_pipeline_disk_state()
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        if hasattr(self, "_cap_timer") and self._cap_timer.isActive():
            self._cap_timer.stop()
        # P2-18: kapanış bayrağı ÖNCE kurulur — bitiş slotları kapanış
        # sırasında UI'a dokunmaz / senkron geri dönüş yapmaz. Eğitim
        # worker'ının ayrı `wait(3000)`'i kaldırıldı: eğitim dahil tüm
        # worker'lar TEK `_stop_yolo_threads` düzeninde durdurulur
        # (stop + quit + sonucu kontrol edilen wait + terminate/log).
        self._yolo_closing = True
        self._stop_yolo_threads()
        super().closeEvent(event)

    def get_yolo_factory_dialog_ready_flag(self) -> dict:
        # S-1501 (Karar 3): ölçmeden True dönülmez; gerçek durum
        # get_yolo_factory_dialog_health_summary() probundan alınır.
        try:
            ready = bool(
                self.get_yolo_factory_dialog_health_summary().get(
                    "yolo_factory_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {"module": "yolo_factory_dialog", "ready": ready}

    def get_yolo_factory_dialog_version_info(self) -> dict:
        # S-1501 (Karar 3): ready health_summary probundan alınır.
        try:
            ready = bool(
                self.get_yolo_factory_dialog_health_summary().get(
                    "yolo_factory_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {"module": "yolo_factory_dialog", "version": __version__, "ready": ready}

    def get_yolo_factory_dialog_diagnostic_snapshot(self) -> dict:
        # S-1501 (Karar 3): status/ready health_summary probundan alınır.
        try:
            ready = bool(
                self.get_yolo_factory_dialog_health_summary().get(
                    "yolo_factory_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {
            "module": "yolo_factory_dialog",
            "status": "ok" if ready else "degraded",
            "ready": ready,
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
