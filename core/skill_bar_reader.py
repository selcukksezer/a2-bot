# -*- coding: utf-8 -*-
"""
Dynamic Skill Bar Reader & Micro Camera Look-Around (core/skill_bar_reader.py)

Implements:
- Item 44: `SkillBarReader` — Inspects the bottom skill bar slots (`1, 2, 3, 4, 5, 6, 7, 8,
  Q, E, R, T` at `x: 0.345..0.660, y: 0.940..0.991` in `pilgrims_ridge_1.jpg`), measuring each
  skill icon's brightness/saturation and blue mana-availability indicator so during combat
  recovery the bot only presses skills that are off cooldown (`ready=True`) and have enough
  mana (`select_best_ready_skill(frame) -> str`) instead of mashing `1-2-3-4-5` blindly.
  Slot centres are MEASURED (G2, 2026-11-10), not equal-split: 4 frames from 3 sessions
  (606, 524, 743, 866; 2560x1440), divider minima + frame-row cross-check + labelled
  .txt files, all A layout (layout_map.json: 365 A, 12 B28_TO_A, 0 B28). The old 8-slot
  equal split put slot 5 at 0.514 (true 0.463) and never saw T / 6 / 7 / 8 at all.
- Item 45: `MicroCameraLookAround` — During long straight walks, every `15–25 seconds`
  (Gaussian interval), gently sways the camera by `±8°` left/right and back using smooth
  WindMouse micro-steps (`perform_micro_camera_sway(input_ctrl)`) so the character never
  looks like a rigid straight-line robot.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# F-0540 tek kaynak: HP/MP iksir eşikleri config/schema.py sabitlerinden
# okunur (sözleşme no 10: HP 65 / MP 45).
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT

from core.exemplar_threshold import is_negative_exemplar_name, load_exemplar_match_threshold
from core.windmouse_humanizer import WindMouseEngine
from version import __version__

# P1-1: slot-state ids come from the single taxonomy source — they are
# NOT canonical class ids (see the SLOT_STATE_* constants on the manager).
from dataset_factory.class_taxonomy import (
    SLOT_STATE_AUTOMATIC_ID,
    SLOT_STATE_EMPTY_ID,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATED_ROIS_PATH = PROJECT_ROOT / "config" / "calibrated_rois.json"

# --- Sweep G (2026-11-09): tek exemplar eşiği + negatif-kayıt savunması -----
def _load_exemplar_match_threshold(default: float = 0.68) -> float:
    """Tek exemplar-benzerlik eşiği (P3-3: `core.exemplar_threshold` yardımcısına delege).

    Eskiden bu dosyada 0.65 / 0.68 / 0.70 / 0.72 / 0.75 dağınık
    kullanılıyordu; hiçbiri ayarda değildi.
    """
    return load_exemplar_match_threshold(default)


EXEMPLAR_MATCH_THRESHOLD: float = _load_exemplar_match_threshold()


# Kalibre-yok uyarısı her anahtar seti için bir kez loglanır (her karede
# spam yapmaz, ama sessiz düşüş de olmaz — Sweep G bulgu 1).
_ROI_FALLBACK_WARNED: set = set()


def _warn_roi_fallback_once(key: str, reason: str) -> None:
    if key in _ROI_FALLBACK_WARNED:
        return
    _ROI_FALLBACK_WARNED.add(key)
    logger.warning(
        "SkillBarReader: kalibre edilmiş ROI yok (%s); sabit yedek ROI kullanılıyor — "
        "config/calibrated_rois.json içine kalibre kaydı eklenene kadar bar konumu "
        "kayarsa okuma yanlış olur.",
        reason,
    )


_SHARED_SKILL_BAR_READER: Optional["SkillBarReader"] = None
_SHARED_SKILL_BAR_READER_LOCK = threading.Lock()


def get_shared_skill_bar_reader() -> "SkillBarReader":
    """Tek paylaşımlı SkillBarReader örneği (0,15 sn basım kilidi için).

    Sweep G bulgu 3: kilit örnek başına (`self._last_pressed_ts`) tutuluyor;
    `combat_recovery` kendi okuyucusunu yarattığı için `game_action_bus`'ın
    kaydettiği basımı görmüyordu. Tüm çağıranlar bu paylaşımlı örneği
    kullanırsa kilit sistem genelinde tek olur. (Çağıran tarafın geçişi
    B grubunda — combat_recovery.py — bu dosya yalnızca mekanizmayı verir.)
    """
    global _SHARED_SKILL_BAR_READER
    if _SHARED_SKILL_BAR_READER is None:
        with _SHARED_SKILL_BAR_READER_LOCK:
            if _SHARED_SKILL_BAR_READER is None:
                _SHARED_SKILL_BAR_READER = SkillBarReader()
    return _SHARED_SKILL_BAR_READER

try:
    from dataset_factory.exemplar_memory import ExemplarMemory, compute_similarity, extract_crop_features
except Exception:
    ExemplarMemory = None  # type: ignore
    extract_crop_features = None  # type: ignore
    compute_similarity = None  # type: ignore



def _validate_roi_tuple(value: Any, label: str) -> Tuple[float, float, float, float]:
    """Validates a normalized (x1, y1, x2, y2) ROI assignment."""
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be 4 numbers (x1, y1, x2, y2)")
    if len(vals) != 4 or not (0.0 <= vals[0] < vals[2] <= 1.0 and 0.0 <= vals[1] < vals[3] <= 1.0):
        raise ValueError(f"{label} must be a valid normalized rect (x1, y1, x2, y2)")
    return vals


def _parse_roi_entry(entry: Any) -> Optional[Tuple[float, float, float, float]]:
    """Parses one calibrated_rois.json entry into an (x1, y1, x2, y2) rect.

    Accepts the explicit `norm_x`/`norm_y`/`norm_w`/`norm_h` fields, or a
    `normalized_roi` list — which is ALWAYS [x, y, w, h] per the stored
    contract (config/calibrated_rois.json; S099/S105, v19/P3-8). The old
    corner-guess branch misread an [x, y, w, h] ROI whose w/h overflowed
    the frame edge as corners. Returns None when the entry carries no
    usable rectangle.
    """
    if not isinstance(entry, dict):
        return None
    if "norm_x" in entry and "norm_y" in entry:
        # Explicit x/y/w/h fields are authoritative (no guessing).
        x = float(entry["norm_x"])
        y = float(entry["norm_y"])
        w = float(entry.get("norm_w", 0.0))
        h = float(entry.get("norm_h", 0.0))
        return (round(x, 6), round(y, 6), round(min(1.0, x + w), 6), round(min(1.0, y + h), 6))
    if "normalized_roi" in entry and len(entry["normalized_roi"]) == 4:
        n = [float(v) for v in entry["normalized_roi"]]
        return (
            round(n[0], 6),
            round(n[1], 6),
            round(min(1.0, n[0] + n[2]), 6),
            round(min(1.0, n[1] + n[3]), 6),
        )
    return None


def _resolve_roi_from_json_status(
    json_path: Optional[Union[str, Path]],
    entry_keys: Sequence[str],
    fallback: Tuple[float, float, float, float],
    min_width: float = 0.10,
    calibrated_keys: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[float, float, float, float], bool, str]:
    """Shared calibrated-ROI resolution for the skill bar and consumable row.

    Returns `(rect, calibrated, reason)`: `calibrated=True` yalnızca
    `config/calibrated_rois.json` içinde geçerli bir kayıt bulunduğunda;
    dosya yoksa / kayıt yoksa / kayıt `min_width`'dan dar ise / JSON
    bozuksa `calibrated=False` ve yedek döner. Sweep G: düşüş artık
    sessiz değil — sebep bir kez warning olarak loglanır.
    """
    path = Path(json_path) if json_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
    if not path.is_file():
        reason = f"kalibre dosyası yok: {path}"
        _warn_roi_fallback_once(",".join(entry_keys), reason)
        return fallback, False, reason
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rois_dict: Dict[str, Any] = {}
        if isinstance(data, dict):
            if "rois" in data and isinstance(data["rois"], dict):
                rois_dict.update(data["rois"])
            for k, v in data.items():
                if k not in ("rois", "updated_at", "version") and isinstance(v, dict):
                    rois_dict[k] = v

        for key in entry_keys:
            rect = _parse_roi_entry(rois_dict.get(key))
            if rect and (rect[2] - rect[0]) >= min_width:
                return rect, True, f"kalibre kaydı: {key}"

        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            for key in (calibrated_keys if calibrated_keys is not None else entry_keys):
                rect = get_calibrated_rois(path).get_normalized_rect(key)
                if rect and len(rect) == 4 and (rect[2] - rect[0]) >= min_width:
                    return rect, True, f"kalibre kaydı: {key}"
        except Exception as exc:
            reason = f"kalibre yükleyici hatası: {exc}"
            _warn_roi_fallback_once(",".join(entry_keys), reason)
            return fallback, False, reason
    except Exception as exc:
        reason = f"kalibre dosyası okunamadı/bozuk: {exc}"
        _warn_roi_fallback_once(",".join(entry_keys), reason)
        return fallback, False, reason
    reason = f"kalibre kaydı yok veya çok dar: {list(entry_keys)} (min_width={min_width})"
    _warn_roi_fallback_once(",".join(entry_keys), reason)
    return fallback, False, reason


def _resolve_roi_from_json(
    json_path: Optional[Union[str, Path]],
    entry_keys: Sequence[str],
    fallback: Tuple[float, float, float, float],
    min_width: float = 0.10,
    calibrated_keys: Optional[Sequence[str]] = None,
) -> Tuple[float, float, float, float]:
    """Backwards-compatible wrapper: returns only the rect (see _status)."""
    rect, _calibrated, _reason = _resolve_roi_from_json_status(
        json_path, entry_keys, fallback, min_width, calibrated_keys
    )
    return rect


# --- Sweep G2 (2026-11-10): 12 slot, olculmus merkezler -----------------------
# Bar 12 slotlu (1-8, Q, E, R, T). Eskiden SKILL_SLOTS 8'liydi ve ROI 8 esit
# sutuna bolunuyordu: slot 5'in penceresi 0.514'e dusuyordu (gercek 0.463),
# T ile 6/7/8 hic okunmuyordu (B-05'in okuyucu yarisi; learner tarafi K'de
# duzeltildi: omni_game_learner.combat_skill_keys artik 12'li).
#
# Merkezler OLCTU (tahmin degil): 4 kare / 3 oturum (2560x1440), y 0.945..0.990
# seridinde sutun-parlaklik minimumlariyla ayirici cizgiler bulundu, merkez =
# bitisik iki ayiricinin ortasi; cerceve-satirlari yontemi + isaretli serit
# gorseli + 4 etiketli .txt ile capraz dogrulandi. 6/7/8 BOS ama GERCEK kareler
# (kenarlikli bos yuva, icinde soluk yildiz; 606 serit gorseliyle dogrulandi).
#   kare  1     2     3     4     5     6     7     8     Q     E     R     T
#   606   .362  .386  .411  .438  .463  .488  .513  .539  .565  .588  .613  .639
#   866   .361  .386  .411  .437  .463  .487  .512  .539  .565  .588  .613  .642
#   743   .363  .386  .411  .438  .466   --   --   .541  .564  .587  .614  .638
#   etiket .361  .388  .413  .439  .461   --   --    --   .567  .594  .620  .644
# 743'te 6/7 ayiricisi tek karede yakalanamadi (birlesik merkez 0.505 anlamsiz);
# 8 o karede de 0.541. E/R/T'de etiket-piksel farki <=0.007 (etiket tablodan
# kopya; piksel olcumu 3 yontemde de tutarli, o kullanildi).
# NOT: brifingdeki REF degerleri (1:0.360 ... T:0.679) YANLIS — NOTES.md'nin
# kendisi duzeltiyor: "REF slot degerleri hicbir olculmus kareye uymuyor".
# layout_map.json: 365 A + 12 B28_TO_A, B28 kalmadi; tum kareler A yerlesimi.
# Yerlesimler degisebilir (Selcuk'un uyarisi): bu degerler varsayilan ROI
# icin gecerli; kalibre ROI farkliysa ayni ORANLAR o ROI'ye olceklenir.
# Yeni yerlesim gorulurse kalibrasyon sihirbazi yeniden olcmeli.
SLOT_CENTERS_NORM: Dict[str, float] = {
    "1": 0.361, "2": 0.387, "3": 0.412, "4": 0.438, "5": 0.463,
    "6": 0.488, "7": 0.513, "8": 0.539,
    "Q": 0.566, "E": 0.589, "R": 0.614, "T": 0.640,
}
# Varsayilan ROI genisligine gore kesirler (tek dogruluk kaynagi: yukaridaki
# merkezler + DEFAULT_SKILL_BAR_ROI; elle yazilmadi, kod hesapliyor).
_DEFAULT_SKILL_BAR_ROI_X: Tuple[float, float] = (0.345, 0.660)


def _slot_center_fractions() -> Tuple[float, ...]:
    """Olculmus merkezlerin varsayilan ROI genisligindeki kesirleri (12'li)."""
    x1, x2 = _DEFAULT_SKILL_BAR_ROI_X
    w = x2 - x1
    return tuple((SLOT_CENTERS_NORM[k] - x1) / w for k in
                 ("1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T"))


SLOT_CENTER_FRACTIONS: Tuple[float, ...] = _slot_center_fractions()


def slot_bounds_fractions() -> Tuple[float, ...]:
    """13 kenar kesri: bitisik merkezlerin orta noktalari (Voronoi).

    Slot penceresi = iki komsu orta nokta arasi; ilk/son pencere yarim
    adim disari uzar. Esit bolunmenin yerini alir; inspect, _slot_center_px
    ve calibration_wizard ayni kenarlari kullanir.
    """
    c = SLOT_CENTER_FRACTIONS
    edges = [c[0] - (c[1] - c[0]) / 2.0]
    edges.extend((a + b) / 2.0 for a, b in zip(c, c[1:]))
    edges.append(c[-1] + (c[-1] - c[-2]) / 2.0)
    return tuple(edges)


def _slot_center_px(
    frame: Optional[np.ndarray],
    slot_key: str,
    slots: Sequence[str],
    roi: Tuple[float, float, float, float],
    jitter_px: int = 3,
) -> Optional[Tuple[int, int]]:
    """Shared slot-centre geometry for SkillBarReader / SmartConsumableManager.

    Kanonik 12'li bar (1-8, Q, E, R, T) icin OLCULMUS merkez kesirleri
    (SLOT_CENTER_FRACTIONS) kullanilir; baska bir slot listesi verilirse
    eski davranis korunur: ROI, len(slots) esit sutuna bolunur.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    key = str(slot_key or "").strip()
    if not key:
        return None
    idx = None
    for i, name in enumerate(slots):
        if str(name).upper() == key.upper():
            idx = i
            break
    if idx is None:
        return None
    try:
        fh, fw = frame.shape[:2]
        x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
        y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
        x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
        y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        canonical = [str(s).upper() for s in slots] == [
            "1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T",
        ]
        if canonical:
            # Olculmus merkez: kesir o ROI'ye olceklenir (kalibre ROI'de de
            # ayni oranlar gecerli sayilir; bar butun olarak kaydiysa dogru).
            frac = SLOT_CENTER_FRACTIONS[idx]
            cx = x1 + int(round(frac * bw))
            cy = y1 + bh // 2
        else:
            num_slots = len(slots)
            slot_w = float(bw) / float(num_slots)
            sx1 = x1 + int(round(idx * slot_w))
            sx2 = x1 + int(round((idx + 1) * slot_w))
            cx = (sx1 + sx2) // 2
            cy = y1 + bh // 2
        j = max(0, int(jitter_px))
        if j > 0:
            cx += random.randint(-j, j)
            cy += random.randint(-j, j)
        cx = max(0, min(fw - 1, cx))
        cy = max(0, min(fh - 1, cy))
        return int(cx), int(cy)
    except Exception as exc:
        logger.debug("slot_center_px failed (%s): %s", slot_key, exc)
        return None


def _skill_score_bucket(item: Dict[str, Any]) -> int:
    """0.1-wide score buckets: a clearly better icon wins; near-equal icons
    (score noise from pixel jitter) rotate fairly by press recency."""
    try:
        return int(math.floor(float(item.get("score", 0.0)) / 0.1))
    except (TypeError, ValueError):
        return 0


class _RoiMeta(type):
    """Metaclass for the reader/manager classes.

    A plain ``Cls.ROI = (...)`` assignment does NOT invoke a data descriptor's
    ``__set__`` — CPython replaces the descriptor in the class dict, so the
    assignment used to silently destroy the descriptor (and change nothing).
    Intercepting it here makes class-level ROI assignment actually apply.
    """

    def __setattr__(cls, name: str, value: Any) -> None:
        descr = cls.__dict__.get(name)
        if isinstance(descr, (_SkillBarRoiDescriptor, _ConsumableBarRoiDescriptor)):
            descr._set_class_roi(cls, value)
            return
        super().__setattr__(name, value)


class _SkillBarRoiDescriptor:
    """
    Descriptor ensuring both SkillBarReader.SKILL_BAR_ROI and reader.SKILL_BAR_ROI
    return the dynamically reloaded tuple from config/calibrated_rois.json.
    """

    def __init__(self, fallback_roi: Tuple[float, float, float, float]) -> None:
        self.fallback_roi = fallback_roi

    def __get__(
        self, instance: Optional[Any], owner: Optional[type] = None
    ) -> Tuple[float, float, float, float]:
        if instance is not None and hasattr(instance, "current_skill_bar_roi"):
            return instance.current_skill_bar_roi
        # Accessed on class itself: SkillBarReader.SKILL_BAR_ROI
        try:
            return SkillBarReader._resolve_skill_bar_roi_static(fallback=self.fallback_roi)
        except Exception:
            return self.fallback_roi

    def _set_class_roi(self, owner: Any, value: Sequence[float]) -> None:
        vals = _validate_roi_tuple(value, "SKILL_BAR_ROI")
        self.fallback_roi = vals
        # The instance path reads DEFAULT_SKILL_BAR_ROI, so move it too —
        # otherwise the assignment would only affect class-level reads.
        owner.DEFAULT_SKILL_BAR_ROI = vals

    def __set__(self, instance: Any, value: Sequence[float]) -> None:
        if instance is not None and hasattr(instance, "set_custom_roi"):
            instance.set_custom_roi(value)
        elif instance is None:
            # Direct descriptor call (class assignment goes via _RoiMeta).
            self._set_class_roi(SkillBarReader, value)


class _ConsumableBarRoiDescriptor:
    """
    Descriptor ensuring both SmartConsumableManager.CONSUMABLE_ROW_ROI and
    manager.CONSUMABLE_ROW_ROI return the dynamically reloaded tuple from config/calibrated_rois.json.
    """

    def __init__(self, fallback_roi: Tuple[float, float, float, float]) -> None:
        self.fallback_roi = fallback_roi

    def __get__(
        self, instance: Optional[Any], owner: Optional[type] = None
    ) -> Tuple[float, float, float, float]:
        if instance is not None and hasattr(instance, "current_consumable_roi"):
            return instance.current_consumable_roi
        try:
            return SmartConsumableManager._resolve_consumable_roi_static(fallback=self.fallback_roi)
        except Exception:
            return self.fallback_roi

    def _set_class_roi(self, owner: Any, value: Sequence[float]) -> None:
        vals = _validate_roi_tuple(value, "CONSUMABLE_ROW_ROI")
        self.fallback_roi = vals
        owner.DEFAULT_CONSUMABLE_ROW_ROI = vals

    def __set__(self, instance: Any, value: Sequence[float]) -> None:
        if instance is not None and hasattr(instance, "set_custom_roi"):
            instance.set_custom_roi(value)
        elif instance is None:
            # Direct descriptor call (class assignment goes via _RoiMeta).
            self._set_class_roi(SmartConsumableManager, value)


class SkillBarReader(metaclass=_RoiMeta):
    """
    Item 44 — Dinamik Yetenek (Skill) Barı Okuyucu (`SkillBarReader`).

    Inspects the bottom skill bar region (`x: 0.345..0.660, y: 0.940..0.991` in
    `pilgrims_ridge_1.jpg`) across the 12 combat slots (`1, 2, 3, 4, 5, 6, 7, 8, Q, E, R, T`).
    Slot centres are MEASURED (SLOT_CENTERS_NORM, G2 2026-11-10), not equal-split:
    the old 8-way split put slot 5 at 0.514 (true 0.463) and never saw T/6/7/8.
    Empty slots (6/7/8 today) are read like any slot: dark/low-saturation means
    off_cooldown=False, never ready — they are NOT reported ready.
    The bar sits strictly below the consumable row (which ends at y=0.938) and the
    player HP/MP bars (y=0.880..0.915), so the ROIs never overlap.
    Evaluates:
      1. Brightness (`V` channel mean in HSV) & local contrast (`std`) to detect whether
         a skill slot is off cooldown vs darkened by a cooldown sweep.
      2. Color Saturation (`S` channel mean in HSV) to distinguish active icons from
         greyed-out/disabled slots.
      3. Blue Mana-Availability indicator (`B - R` channel dominance and cyan/blue HSV mask
         in the slot / badge corner) to verify sufficient mana.
    Provides `inspect_skill_bar(frame)` and `select_best_ready_skill(frame) -> str`.
    """

    # AION2 bottom hotbar (12 slots: 1-8, Q, E, R, T). Below the consumable row (ends y=0.938).
    # x2=0.660: T'nin sag kenari 0.654 (eski 0.645 T'yi kesiyordu); y2=0.991: slot alti
    # 0.9905 (eski 0.985 alt 0.0055'i kesiyordu). Olcum: 4 etiketli .txt + piksel yontemi.
    DEFAULT_SKILL_BAR_ROI: Tuple[float, float, float, float] = (0.345, 0.940, 0.660, 0.991)
    SKILL_BAR_ROI: Any = _SkillBarRoiDescriptor(DEFAULT_SKILL_BAR_ROI)
    PLAYER_HP_BAR_ROI: Tuple[float, float, float, float] = (0.345, 0.880, 0.485, 0.915)
    PLAYER_MP_BAR_ROI: Tuple[float, float, float, float] = (0.515, 0.880, 0.655, 0.915)
    TARGET_HP_BAR_ROI: Tuple[float, float, float, float] = (0.350, 0.030, 0.650, 0.075)
    SKILL_SLOTS: Tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T")

    @classmethod
    def _resolve_skill_bar_roi_static(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        fallback: Tuple[float, float, float, float] = (0.345, 0.940, 0.660, 0.991),
    ) -> Tuple[float, float, float, float]:
        # Thin wrapper over the shared resolver (see _resolve_roi_from_json).
        return _resolve_roi_from_json(
            json_path, ("skill_bar",), fallback, calibrated_keys=("skill_bar",)
        )

    @classmethod
    def _resolve_skill_bar_roi_status_static(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        fallback: Tuple[float, float, float, float] = (0.345, 0.940, 0.645, 0.985),
    ) -> Tuple[Tuple[float, float, float, float], bool, str]:
        """Like _resolve_skill_bar_roi_static, but also reports whether the
        rect came from a calibrated entry (Sweep G bulgu 1)."""
        return _resolve_roi_from_json_status(
            json_path, ("skill_bar",), fallback, calibrated_keys=("skill_bar",)
        )

    @property
    def is_skill_bar_roi_calibrated(self) -> bool:
        """True yalnızca skill_bar ROI'si kalibre kaydından geliyorsa.

        Sevk edilen config'de `skill_bar` kaydı YOK — okuyucu her açılışta
        tek kareden alınmış sabit ROI'ye düşüyordu ve bunu kimseye
        söylemiyordu. Artık düşüş açıkça görünür (log + bu bayrak +
        ready-flag çıktısı)."""
        try:
            self.get_skill_bar_roi()
        except Exception:
            return False
        return bool(getattr(self, "_roi_calibrated", False))

    @property
    def skill_bar_roi_source(self) -> str:
        """ROI kaynağı: 'custom' / 'kalibre kaydı: ...' / yedek sebebi."""
        if self._custom_roi is not None:
            return "custom"
        try:
            self.get_skill_bar_roi()
        except Exception:
            pass
        return str(getattr(self, "_roi_source", "bilinmiyor"))

    def get_skill_bar_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        """
        Dynamically reloads and returns SKILL_BAR_ROI from config/calibrated_rois.json.
        Performs an mtime check (<50ms, typically <0.05ms) so changes saved by
        Calibration Wizard or manual edits take effect immediately without restart.
        """
        if self._custom_roi is not None:
            return self._custom_roi

        json_path = self.calibrated_rois_path
        if not json_path or not json_path.is_file():
            # Sweep G: dosya yoksa sessizce sabite düşme — bir kez uyar.
            self._roi_calibrated = False
            self._roi_source = f"kalibre dosyası yok: {json_path}"
            _warn_roi_fallback_once("skill_bar", self._roi_source)
            return self.DEFAULT_SKILL_BAR_ROI

        try:
            mtime = json_path.stat().st_mtime
            if not force and mtime == self._last_roi_mtime and self._cached_roi is not None:
                return self._cached_roi

            with self._lock:
                rect, calibrated, reason = self._resolve_skill_bar_roi_status_static(
                    json_path=json_path,
                    fallback=self.DEFAULT_SKILL_BAR_ROI,
                )
                self._cached_roi = rect
                self._roi_calibrated = bool(calibrated)
                self._roi_source = str(reason)
                self._last_roi_mtime = mtime
                return rect
        except Exception as exc:
            logger.debug("SkillBarReader get_skill_bar_roi error: %s", exc)
            return self._cached_roi or self.DEFAULT_SKILL_BAR_ROI

    def check_reload_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        """Checks mtime of config/calibrated_rois.json and reloads SKILL_BAR_ROI if modified (<50ms)."""
        return self.get_skill_bar_roi(force=force)

    def reload_calibrated_rois(self, force: bool = True) -> Tuple[float, float, float, float]:
        """Force-reload calibrated skill bar ROI from disk."""
        return self.get_skill_bar_roi(force=force)

    def set_custom_roi(self, roi: Sequence[float]) -> None:
        """Manually override skill bar ROI with custom normalized coordinates (x1, y1, x2, y2)."""
        if len(roi) == 4:
            self._custom_roi = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))

    @property
    def current_skill_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads SKILL_BAR_ROI from config/calibrated_rois.json on file modification (<50ms)."""
        return self.get_skill_bar_roi()

    @property
    def current_player_hp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads player_hp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_hp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_hp")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.PLAYER_HP_BAR_ROI

    @property
    def current_player_mp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads player_mp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_mp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_mp")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.PLAYER_MP_BAR_ROI

    @property
    def current_target_hp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads target_hp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("target_hp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("target_hp")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("mob_nameplate")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.TARGET_HP_BAR_ROI

    # Convenience properties
    @property
    def player_hp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_player_hp_bar_roi

    @property
    def player_mp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_player_mp_bar_roi

    @property
    def target_hp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_target_hp_bar_roi

    @property
    def skill_bar_roi(self) -> Tuple[float, float, float, float]:
        return self.current_skill_bar_roi

    def get_calibrated_roi(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Get calibrated normalized rect for named ROI."""
        name_lower = str(name).lower()
        if "skill" in name_lower:
            return self.current_skill_bar_roi
        if "player_hp" in name_lower or name_lower == "hp":
            return self.current_player_hp_bar_roi
        if "player_mp" in name_lower or name_lower == "mp":
            return self.current_player_mp_bar_roi
        if "target" in name_lower or "mob" in name_lower:
            return self.current_target_hp_bar_roi
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            return get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect(name)
        except Exception:
            return None


    # v20/empty-slots: boş-slot eşiği (merkez bölge, simge yokluğu).
    # Simge barın merkezinde oturur; kenar parlaması / köşedeki tuş harfi
    # merkeze girmez. Boş slot = merkezde keskin kenar YOK (Sobel p95
    # düşük) VE canlı simge pikseli YOK (doygunluk p95 düşük). İkisi birden
    # gerekir: renkli ama düz bir simge doygunlukla, soluk ama detaylı bir
    # simge kenarla kurtulur. Eşikler canlı kare olmadan seçildi — PC'de
    # gerçek boş/dolu slotlarla doğrulanacak (bkz. tests/test_v20_empty_slots.py).
    EMPTY_CENTER_EDGE_P95: float = 40.0
    EMPTY_CENTER_SAT_P95: float = 90.0

    # Priority weights for combat rotation when multiple skills are ready & have mana
    # Equal baseline weights to eliminate artificial bias towards slot '2'.
    DEFAULT_PRIORITY_WEIGHTS: Dict[str, float] = {
        "1": 1.00,
        "2": 1.00,
        "3": 1.00,
        "4": 1.00,
        "5": 1.00,
        "6": 1.00,
        "7": 1.00,
        "8": 1.00,
        "Q": 1.00,
        "E": 1.00,
        "R": 1.00,
        "T": 1.00,
    }

    def __init__(
        self,
        min_brightness_ready: float = 50.0,
        min_saturation_ready: float = 45.0,
        min_contrast_std: float = 22.0,
        min_mana_blue_ratio: float = 0.03,
        cooldown_lockout_sec: float = 0.15,
        calibrated_rois_path: Optional[Union[str, Path]] = None,
        exemplar_memory: Optional[Any] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.min_brightness_ready: float = float(min_brightness_ready)
        self.min_saturation_ready: float = float(min_saturation_ready)
        self.min_contrast_std: float = float(min_contrast_std)
        self.min_mana_blue_ratio: float = float(min_mana_blue_ratio)
        self.cooldown_lockout_sec: float = max(0.0, float(cooldown_lockout_sec))

        self.calibrated_rois_path: Path = (
            Path(calibrated_rois_path) if calibrated_rois_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        )
        self._last_roi_mtime: float = -1.0
        self._cached_roi: Optional[Tuple[float, float, float, float]] = None
        self._custom_roi: Optional[Tuple[float, float, float, float]] = None
        # Sweep G: ROI kaynağı takibi (kalibre mi, sabit yedek mi).
        self._roi_calibrated: bool = False
        self._roi_source: str = "bilinmiyor"
        self.exemplar_memory: Optional[Any] = exemplar_memory
        self._consumable_manager: Optional[SmartConsumableManager] = None

        self._last_pressed_ts: Dict[str, float] = {}
        self._last_slot_states: Dict[str, Dict[str, Any]] = {}
        self._call_counter: int = 0
        self._last_press_ts: float = 0.0
        self._vitals_combat: Optional[Any] = None

    def get_exemplar_memory(self) -> Optional[Any]:
        """Resolves active ExemplarMemory instance (injected or global singleton)."""
        if self.exemplar_memory is not None:
            return self.exemplar_memory
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("Could not resolve ExemplarMemory: %s", exc)
            return None

    def reset_history(self) -> None:
        """Clears recent skill press lockout timestamps (and the press-order clock)."""
        with self._lock:
            self._last_pressed_ts.clear()
            self._last_slot_states.clear()
            self._call_counter = 0
            self._last_press_ts = 0.0

    def _next_press_ts(self) -> float:
        """Monotonic timestamp for press ordering.

        Strictly increasing by at least 1us so round-robin ordering is stable,
        but bounded: it never drifts away from time.monotonic() (the old
        ``monotonic() + _call_counter * 1e-6`` hack grew without bound).
        """
        t = time.monotonic()
        if t <= self._last_press_ts:
            t = self._last_press_ts + 1e-6
        self._last_press_ts = t
        self._call_counter += 1
        return t

    def mark_slot_pressed(self, slot_key: str) -> None:
        """Record a UI/key cast so inspect_skill_bar applies cooldown lockout."""
        key = str(slot_key or "").strip()
        if not key:
            return
        # Normalize to reader slot casing (1..8, Q, E, R, T).
        upper = key.upper()
        if upper in ("Q", "E", "R", "T"):
            key = upper
        else:
            key = key.lower()
        with self._lock:
            now_ts = self._next_press_ts()
            self._last_pressed_ts[key] = now_ts
            # Also store letter variants used by combat configs.
            self._last_pressed_ts[key.lower()] = now_ts
            self._last_pressed_ts[key.upper()] = now_ts

    def slot_center_px(
        self,
        frame: Optional[np.ndarray],
        slot_key: str,
        jitter_px: int = 3,
    ) -> Optional[Tuple[int, int]]:
        """
        Pixel center of a skill-bar slot inside SKILL_BAR_ROI.
        Kanonik 12'li bar icin OLCULMUS merkez kesirleri (SLOT_CENTER_FRACTIONS);
        esit bolunme degil (G2). Baska slot listesi verilirse esit bolunme.
        """
        try:
            roi = self.current_skill_bar_roi
        except Exception as exc:
            logger.debug("slot_center_px failed (%s): %s", slot_key, exc)
            return None
        return _slot_center_px(frame, slot_key, self.SKILL_SLOTS, roi, jitter_px)

    def inspect_skill_bar(self, frame: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Inspects all 12 skill slots (`1, 2, 3, 4, 5, 6, 7, 8, Q, E, R, T`) in the bottom HUD skill bar
        (`x: 0.345..0.660, y: 0.940..0.991`) and returns a per-slot diagnostics dictionary:
        `{slot_key: {"slot": str, "ready": bool, "mana_ok": bool, "off_cooldown": bool,
                     "brightness": float, "saturation": float, "blue_ratio": float,
                     "contrast_std": float, "score": float}}`.
        Slot windows use the MEASURED centres (slot_bounds_fractions), not equal
        columns. Empty slots are detected directly by icon absence in the slot
        centre (v20/empty-slots) and are never ready — the old claim that
        "empty slots read dark" was disproved live (v19: '6'/'8' pressed
        while empty; frame glow / gradient passes the whole-slot thresholds).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {}

        try:
            fh, fw = frame.shape[:2]
            roi = self.current_skill_bar_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))

            bar_crop = frame[y1:y2, x1:x2]
            if bar_crop.size == 0:
                return {}

            if bar_crop.ndim == 2:
                bar_bgr = cv2.cvtColor(bar_crop, cv2.COLOR_GRAY2BGR)
            else:
                bar_bgr = bar_crop[:, :, :3]

            # Convert the whole bar ONCE: the old code ran BGR->HSV and
            # BGR->gray separately for each of the 12 slots (24 conversions of
            # the same pixels). cvtColor is per-pixel, so slicing the
            # bar-level results gives bit-identical per-slot metrics.
            bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
            bar_gray = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2GRAY)

            bh, bw = bar_bgr.shape[:2]
            # Olculmus merkezler: pencere kenarlari bitisik merkezlerin orta
            # noktalari (slot_bounds_fractions) — esit bolunme degil (G2).
            # Kenar kesirleri ROI genisligine gore; bar butun olarak kayarsa
            # oranlar korunur.
            edges = slot_bounds_fractions()

            now = time.monotonic()
            results: Dict[str, Dict[str, Any]] = {}
            # Resolve ExemplarMemory once per frame (was: once per slot).
            frame_mem = self.get_exemplar_memory()

            # W51 kilit yarışı (v19): exemplar sorguları kare başına bir kez
            # hesaplanır. Boş bellekte her sorgu zaten nötr döner, o yüzden
            # çıkarma hiç yapılmaz; dolu bellekte aynı karenin özellikleri
            # bir kez çıkarılıp tüm sorgular paylaşır (feats=). has_*
            # çağrıları önce check_reload yapar, yani diskte yeni kayıt
            # belirdiyse sonraki kare normal yola düşer.
            _frame_mem_empty = False
            _frame_mem_use_feats = False
            if frame_mem is not None:
                if hasattr(frame_mem, "has_positive_exemplars") and hasattr(frame_mem, "has_negative_exemplars"):
                    try:
                        _frame_mem_empty = not (frame_mem.has_positive_exemplars() or frame_mem.has_negative_exemplars())
                    except Exception:
                        _frame_mem_empty = False
                _frame_mem_use_feats = (
                    ExemplarMemory is not None
                    and extract_crop_features is not None
                    and isinstance(frame_mem, ExemplarMemory)
                )

            for idx, slot_key in enumerate(self.SKILL_SLOTS):
                sx1 = max(0, int(round(edges[idx] * bw)))
                sx2 = max(sx1 + 2, min(bw, int(round(edges[idx + 1] * bw))))
                slot_img = bar_bgr[:, sx1:sx2]

                # Trim outer 12% border so slot frame lines don't skew icon interior metrics
                sh, sw_local = slot_img.shape[:2]
                pad_y = max(1, int(round(sh * 0.12)))
                pad_x = max(1, int(round(sw_local * 0.12)))
                iy0, iy1 = pad_y, max(pad_y + 2, sh - pad_y)
                ix0, ix1 = sx1 + pad_x, sx1 + max(pad_x + 2, sw_local - pad_x)
                inner = slot_img[iy0:iy1, pad_x : max(pad_x + 2, sw_local - pad_x)]
                if inner.size == 0:
                    inner = slot_img
                    iy0, iy1, ix0, ix1 = 0, sh, sx1, sx2
                # Same inner pixels, sliced from the bar-level conversions.
                inner_hsv = bar_hsv[iy0:iy1, ix0:ix1]
                inner_gray = bar_gray[iy0:iy1, ix0:ix1]

                brightness = float(np.mean(inner_hsv[:, :, 2]))
                saturation = float(np.mean(inner_hsv[:, :, 1]))
                # Spatial contrast: std of the grayscale image. (np.std over the
                # flattened BGR array measures cross-channel spread instead, so a
                # flat single-colour patch scored as "high contrast".)
                contrast_std = float(np.std(inner_gray))

                # v20/empty-slots: BOŞ SLOT TESPİTİ — simge yokluğu doğrudan
                # ölçülür, tüm-slot parlaklık/doygunluk eşiklerine güvenilmez.
                # Simge slotun MERKEZİNDE oturur; çerçeve parlaması, tuş
                # harfi ve bar gradyanı kenarda kalır, o yüzden iç bölgenin
                # ortadaki %50'si ölçülür. Gerçek simge merkezde keskin
                # kenar (Sobel) ve/veya canlı renk bırakır; boş slot
                # (düz dolgu ya da yumuşak gradyan/parlama) ikisini de
                # bırakmaz. Yumuşak gradyan kontrast_std'yi şişirir ama
                # Sobel'i şişirmez — o yüzden kural kenar + doygunluk
                # üzerinedir, kontrast_std kullanılmaz.
                _ch, _cw = inner_gray.shape[:2]
                _cy0, _cy1 = _ch // 4, max(_ch // 4 + 1, _ch - _ch // 4)
                _cx0, _cx1 = _cw // 4, max(_cw // 4 + 1, _cw - _cw // 4)
                _center_gray = inner_gray[_cy0:_cy1, _cx0:_cx1]
                _center_hsv = inner_hsv[_cy0:_cy1, _cx0:_cx1]
                if _center_gray.size >= 9:
                    _gx = cv2.Sobel(_center_gray, cv2.CV_32F, 1, 0, ksize=3)
                    _gy = cv2.Sobel(_center_gray, cv2.CV_32F, 0, 1, ksize=3)
                    _grad = np.sqrt(_gx * _gx + _gy * _gy)
                    center_edge_p95 = float(np.percentile(_grad, 95))
                else:
                    center_edge_p95 = 0.0
                center_sat_p95 = (
                    float(np.percentile(_center_hsv[:, :, 1], 95))
                    if _center_hsv.size
                    else 0.0
                )
                is_empty = bool(
                    center_edge_p95 < self.EMPTY_CENTER_EDGE_P95
                    and center_sat_p95 < self.EMPTY_CENTER_SAT_P95
                )

                # Measure blue mana-availability indicator across the full slot (including mana cost badge)
                b_chan = slot_img[:, :, 0].astype(np.int16)
                g_chan = slot_img[:, :, 1].astype(np.int16)
                r_chan = slot_img[:, :, 2].astype(np.int16)
                blue_dominant_mask = (b_chan - r_chan > 14) & (b_chan > 55) & (g_chan >= r_chan - 10)
                blue_ratio = float(np.mean(blue_dominant_mask))

                # Red "no mana / unavailable" overlay check
                red_dominant_mask = (r_chan - b_chan > 45) & (r_chan - g_chan > 35) & (r_chan > 110)
                red_penalty_ratio = float(np.mean(red_dominant_mask))

                baseline_off_cooldown = bool(
                    brightness >= self.min_brightness_ready
                    and saturation >= self.min_saturation_ready
                    and contrast_std >= self.min_contrast_std
                )

                # Enough mana requires the blue mana indicator to be present AND no
                # red "out of mana" overlay. (The old `or red_penalty < 0.25` clause
                # was true for almost every patch, so mana_ok was effectively always
                # True even with zero blue in the slot.)
                mana_ok = bool(
                    blue_ratio >= self.min_mana_blue_ratio
                    and red_penalty_ratio < 0.38
                )

                with self._lock:
                    last_used = self._last_pressed_ts.get(slot_key, 0.0)
                    recently_pressed = (now - last_used) < self.cooldown_lockout_sec if last_used > 0.0 else False
                    lockout_remaining_sec = max(0.0, self.cooldown_lockout_sec - (now - last_used)) if (last_used > 0.0 and (now - last_used) < self.cooldown_lockout_sec) else 0.0

                off_cooldown = baseline_off_cooldown
                ready = bool(off_cooldown and mana_ok and not recently_pressed)

                # Consult ExemplarMemory for active/ready (class 20) vs cooldown/empty (class 21)
                exemplar_reinforced = False
                exemplar_state: str = "none"
                ready_sim: float = 0.0
                cooldown_sim: float = 0.0
                exemplar_sim: float = 0.0
                exemplar_confidence: float = 0.0

                mem = frame_mem
                if mem is not None and inner.size > 0:
                    try:
                        # W51 kilit yarışı — kök neden: aynı slot karesi için
                        # özellik çıkarımı sorgu başına tekrarlanıyordu
                        # (negatif ×2 + match_class ×2 + classify ×2 = slot
                        # başına 6, kare başına ~72 çıkarma). Her inspect
                        # ~70 ms sürüyordu; 0,15 sn'lik basım kilidi ardışık
                        # seçimler arasında duvar saatinde tükeniyor, kilitli
                        # skill yeniden "hazır" sayılıyordu (önce basılan
                        # slot önce düşer: test 3'te '3', test 4'te '1').
                        # Şimdi her kare bir kez çıkarılır ve tüm sorgular
                        # aynı özellikleri paylaşır (feats=); boş bellekte
                        # sorgular atlanır — boş bellek zaten nötr döner
                        # (aşağıdaki hüküm else dalı: exemplar_state
                        # "neutral", sim 0), o yüzden çıktı birebir aynıdır.
                        _feats_inner = None
                        _feats_slot = None
                        _kw_inner: Dict[str, Any] = {}
                        if not _frame_mem_empty and _frame_mem_use_feats:
                            _feats_inner = extract_crop_features(inner, include_1d_hists=False)
                            _kw_inner = {"feats": _feats_inner}

                        # 1. Negative exemplar check (known false/empty/corrupted slots)
                        # Sweep G: tek eşik (EXEMPLAR_MATCH_THRESHOLD); eskiden
                        # iç kare 0.72, tüm slot 0.75 idi.
                        is_neg, neg_sim = False, 0.0
                        if not _frame_mem_empty:
                            is_neg, neg_sim, _ = mem.is_negative_patch(inner, threshold=EXEMPLAR_MATCH_THRESHOLD, **_kw_inner)
                            if not is_neg and slot_img.size > 0:
                                if _frame_mem_use_feats and _feats_slot is None:
                                    _feats_slot = extract_crop_features(slot_img, include_1d_hists=False)
                                _kw_slot = {"feats": _feats_slot} if _frame_mem_use_feats else {}
                                is_neg, neg_sim, _ = mem.is_negative_patch(slot_img, threshold=EXEMPLAR_MATCH_THRESHOLD, **_kw_slot)

                        # A known negative already decides the slot: skip the
                        # expensive match/classify queries for this slot.
                        pos_cid = None
                        pos_cname = None
                        p_sim = 0.0
                        if not is_neg and not _frame_mem_empty:
                            # 2. Specific class queries: 20=skill_slot_ready, 21=skill_slot_cooldown
                            if hasattr(mem, "match_class"):
                                r_sim, _ = mem.match_class(inner, 20, **_kw_inner)
                                c_sim, _ = mem.match_class(inner, 21, **_kw_inner)
                                ready_sim = max(ready_sim, float(r_sim))
                                cooldown_sim = max(cooldown_sim, float(c_sim))

                            # 3. Classify patch across all learned positive exemplars
                            pos_cid, pos_cname, p_sim = mem.classify_patch(inner, min_sim=EXEMPLAR_MATCH_THRESHOLD, **_kw_inner)
                            if pos_cid is None and slot_img.size > 0:
                                if _frame_mem_use_feats and _feats_slot is None:
                                    _feats_slot = extract_crop_features(slot_img, include_1d_hists=False)
                                _kw_slot = {"feats": _feats_slot} if _frame_mem_use_feats else {}
                                pos_cid, pos_cname, p_sim = mem.classify_patch(slot_img, min_sim=EXEMPLAR_MATCH_THRESHOLD, **_kw_slot)

                        if pos_cid is not None:
                            cname_lower = str(pos_cname or "").lower()
                            p_float = float(p_sim)
                            # Sweep G: pozitif listeye sızmış negatif kayıt
                            # (örn. (10, 'monster_3d_negative')) asla pozitif
                            # kanıt sayılmaz — negatif hükmünde işlem görür.
                            if is_negative_exemplar_name(pos_cname):
                                is_neg = True
                                neg_sim = max(float(neg_sim), p_float)
                            elif pos_cid == 20 or "ready" in cname_lower:
                                ready_sim = max(ready_sim, p_float)
                            elif pos_cid == 21 or "cooldown" in cname_lower or "empty" in cname_lower:
                                cooldown_sim = max(cooldown_sim, p_float)
                            elif pos_cid == 13 or "skill" in cname_lower:
                                ready_sim = max(ready_sim, p_float * 0.90)

                        # Adjudicate exemplar verdict
                        if is_neg:
                            off_cooldown = False
                            ready = False
                            exemplar_state = "negative"
                            exemplar_sim = neg_sim
                            exemplar_confidence = neg_sim
                        elif cooldown_sim >= EXEMPLAR_MATCH_THRESHOLD and cooldown_sim > (ready_sim + 0.05):
                            off_cooldown = False
                            ready = False
                            exemplar_state = "cooldown"
                            exemplar_sim = cooldown_sim
                            exemplar_confidence = cooldown_sim
                        elif ready_sim >= EXEMPLAR_MATCH_THRESHOLD and ready_sim > (cooldown_sim + 0.05):
                            off_cooldown = True
                            # A strong class-20 exemplar match is learned
                            # evidence that this exact icon is a ready slot
                            # (exemplars are approved via UI from real ready
                            # slots). The blue-badge heuristic behind mana_ok
                            # must not veto it: many ready icons carry no blue
                            # badge, so D03's fail-closed blue requirement
                            # made the exemplar-ready path unable to ever
                            # report ready. The red "no mana" overlay stays
                            # the fail-closed guard — a covered icon no longer
                            # matches the exemplar either — so ready still
                            # requires no red overlay and no press lockout.
                            # mana_ok itself is left as measured (blue badge
                            # seen or not) and is reported unchanged.
                            ready = bool(not recently_pressed and red_penalty_ratio < 0.38)
                            exemplar_reinforced = True
                            exemplar_state = "ready"
                            exemplar_sim = ready_sim
                            exemplar_confidence = ready_sim
                        else:
                            exemplar_state = "neutral"
                            exemplar_sim = max(ready_sim, cooldown_sim)
                            exemplar_confidence = exemplar_sim

                    except Exception as exc:
                        logger.debug("Exemplar memory slot evaluation error: %s", exc)

                # v20/empty-slots: BOŞ SLOT VETOSU — simge yokluğu ölçüldü
                # (is_empty); hiçbir yol boş slotu hazır sayamaz: ne taban
                # eşikler, ne exemplar-hazır baypası (class 20/13 eşleşmesi
                # parlak çerçeve/gradyan parçasıyla da tutabilir). Exemplar
                # class 21 (cooldown/empty) zaten veto ederdi ama yalnızca
                # eşleşme kazanırsa; eşleşmesiz "neutral" durumda taban +
                # baypas boş slotu hazır çıkarabiliyordu — canlı v19 kanıtı
                # (15:11:28 SKILL KEY '6','8'). Veto exemplar sim'lerini
                # kaydetmeden SONRA uygulanır ki tanı logu nedeni göstersin.
                if is_empty:
                    if ready or exemplar_state == "ready":
                        logger.debug(
                            "EMPTY SLOT VETO slot=%s brightness=%.1f saturation=%.1f "
                            "contrast=%.1f blue_ratio=%.3f exemplar_state=%s "
                            "ready_sim=%.2f cooldown_sim=%.2f center_edge_p95=%.1f "
                            "center_sat_p95=%.0f",
                            slot_key, brightness, saturation, contrast_std,
                            blue_ratio, exemplar_state, ready_sim,
                            cooldown_sim, center_edge_p95, center_sat_p95,
                        )
                    ready = False
                    off_cooldown = False
                    if exemplar_state in ("ready", "neutral"):
                        exemplar_state = "empty"

                # Unbiased composite readiness score:
                # Primary readiness indicator (off-cooldown, mana available, not locked out)
                # without brightness dominating or biasing towards slot '2'.
                base_readiness = 1.0 if ready else (0.45 if (off_cooldown and mana_ok) else 0.10)
                norm_v = min(1.0, brightness / 100.0)
                norm_s = min(1.0, saturation / 140.0)
                norm_contrast = min(1.0, contrast_std / 50.0)
                priority_mult = self.DEFAULT_PRIORITY_WEIGHTS.get(slot_key, 1.0)
                visual_quality = 0.50 * norm_s + 0.30 * norm_contrast + 0.20 * norm_v

                if exemplar_state == "ready" and ready:
                    # Sweep G: skor tabanı kaldırıldı — eskiden görsel kanıt
                    # 0 olsa bile skor en az 0.75 idi (tek yönlü şişirme).
                    score = round(float(np.clip(0.65 * ready_sim + 0.35 * visual_quality, 0.0, 0.99)) * priority_mult, 4)
                elif exemplar_state in ("cooldown", "negative", "empty") or not off_cooldown:
                    score = round(float(max(0.01, min(0.15, 0.15 * visual_quality))), 4)
                else:
                    score = round(float(base_readiness * visual_quality * priority_mult), 4)
                    if cooldown_sim > 0.40:
                        score = round(float(max(0.05, score * (1.0 - 0.5 * cooldown_sim))), 4)

                results[slot_key] = {
                    "slot": slot_key,
                    "ready": ready,
                    "off_cooldown": off_cooldown,
                    "mana_ok": mana_ok,
                    "recently_pressed": recently_pressed,
                    "lockout_remaining_sec": round(lockout_remaining_sec, 3),
                    "brightness": round(brightness, 2),
                    "saturation": round(saturation, 2),
                    "blue_ratio": round(blue_ratio, 4),
                    "contrast_std": round(contrast_std, 2),
                    "score": score,
                    "exemplar_state": exemplar_state,
                    "exemplar_sim": round(exemplar_sim, 3),
                    "exemplar_ready_sim": round(ready_sim, 3),
                    "exemplar_cooldown_sim": round(cooldown_sim, 3),
                    "exemplar_confidence": round(exemplar_confidence, 3),
                    "exemplar_reinforced": exemplar_reinforced,
                    "empty": bool(is_empty),
                    "center_edge_p95": round(center_edge_p95, 1),
                    "center_sat_p95": round(center_sat_p95, 1),
                }

            with self._lock:
                self._last_slot_states = dict(results)

            return results

        except Exception as exc:
            logger.error("SkillBarReader.inspect_skill_bar failed: %s", exc)
            return {}

    def _sort_ready_candidates(
        self,
        candidates: List[Dict[str, Any]],
        order_map: Optional[Dict[str, int]] = None,
    ) -> None:
        """In-place candidate sort shared by all three selection branches:
        highest score bucket first, then least-recently-pressed, then
        preferred order (when given), then natural slot order."""
        slot_indices = {s: i for i, s in enumerate(self.SKILL_SLOTS)}
        if order_map is not None:
            candidates.sort(
                key=lambda item: (
                    -_skill_score_bucket(item),
                    self._last_pressed_ts.get(item["slot"], 0.0),
                    order_map.get(item["slot"], 999),
                    slot_indices.get(item["slot"], 999),
                )
            )
        else:
            # Unbiased round-robin rotation across all ready skills:
            # Highest score bucket first, then least recently pressed
            # timestamp (oldest first) so every ready skill gets used,
            # breaking ties by natural slot order (1, 2, 3, 4, 5, 6, 7, 8, Q, E, R, T).
            candidates.sort(
                key=lambda item: (
                    -_skill_score_bucket(item),
                    self._last_pressed_ts.get(item["slot"], 0.0),
                    slot_indices.get(item["slot"], 999),
                )
            )

    def select_best_ready_skill(
        self,
        frame: Optional[np.ndarray],
        preferred_order: Optional[Sequence[str]] = None,
        record_use: bool = True,
    ) -> str:
        """
        Evaluates all 12 skill slots (`1, 2, 3, 4, 5, 6, 7, 8, Q, E, R, T`) on `frame` and returns
        the key name (`"1"`, `"2"`, `"3"`, `"4"`, `"5"`, `"6"`, `"7"`, `"8"`, `"Q"`, `"E"`, `"R"`,
        or `"T"`) of the highest-scoring skill that is BOTH off cooldown (`ready=True`) and has
        sufficient mana (`mana_ok=True`).

        Selection order: highest score bucket first (scores are bucketed in 0.1 steps
        so near-equal icons rotate fairly instead of a noise-level score difference
        pinning one slot), then least-recently-pressed, then preferred/slot order.

        Sweep G — tek "hazır" tanımı: yalnızca `inspect_skill_bar` çıktısındaki
        `ready` alanı (`off_cooldown and mana_ok and not recently_pressed`,
        exemplar-hazır yolunda kırmızı veto korunarak). Eskiden hazır yoksa
        önce kilit gevşetiliyor, sonra mana kontrolsüz en iyi `off_cooldown`
        slot seçiliyordu — yani tam da "hiç hazır skill yok" anında kilit ve
        mana devre dışı kalıyordu. Artık hazır aday yoksa `""` döner (fail-closed);
        çağıran boş dönüşü "bilinmiyor / basılacak skill yok" saymalıdır.

        Returns `""` when there is no frame to inspect (no blind `"1"` press on a
        missing frame) or when no skill is strictly ready.
        """
        slot_states = self.inspect_skill_bar(frame)
        if not slot_states:
            return ""

        ready_candidates: List[Dict[str, Any]] = [
            info for info in slot_states.values() if info.get("ready", False)
        ]

        if not ready_candidates:
            return ""

        if preferred_order:
            order_map = {str(k).upper(): i for i, k in enumerate(preferred_order)}
            self._sort_ready_candidates(ready_candidates, order_map)
        else:
            self._sort_ready_candidates(ready_candidates)
        chosen_key = str(ready_candidates[0]["slot"])

        # v20/empty-slots: seçim tanısı — bir tuşa basılacaksa o slotun
        # ölçümü canlı logda görünsün; boş slot basımı bir daha olursa
        # nedeni (parlaklık/doygunluk/ready/exemplar/empty) bu satırda okunur.
        _chosen = ready_candidates[0]
        logger.debug(
            "SELECT READY SKILL slot=%s brightness=%.1f saturation=%.1f contrast=%.1f "
            "blue_ratio=%.3f score=%.4f exemplar_state=%s ready_sim=%.2f cooldown_sim=%.2f "
            "empty=%s center_edge_p95=%.1f center_sat_p95=%.0f",
            chosen_key,
            float(_chosen.get("brightness", 0.0) or 0.0),
            float(_chosen.get("saturation", 0.0) or 0.0),
            float(_chosen.get("contrast_std", 0.0) or 0.0),
            float(_chosen.get("blue_ratio", 0.0) or 0.0),
            float(_chosen.get("score", 0.0) or 0.0),
            str(_chosen.get("exemplar_state", "?")),
            float(_chosen.get("exemplar_ready_sim", 0.0) or 0.0),
            float(_chosen.get("exemplar_cooldown_sim", 0.0) or 0.0),
            bool(_chosen.get("empty", False)),
            float(_chosen.get("center_edge_p95", 0.0) or 0.0),
            float(_chosen.get("center_sat_p95", 0.0) or 0.0),
        )

        if record_use and chosen_key:
            with self._lock:
                now_ts = self._next_press_ts()
                self._last_pressed_ts[chosen_key] = now_ts
                self._last_pressed_ts[chosen_key.lower()] = now_ts
                self._last_pressed_ts[chosen_key.upper()] = now_ts

        return chosen_key

    def get_ready_skills(self, frame: Optional[np.ndarray]) -> List[str]:
        """Returns a list of slot keys that are currently ready and off cooldown.

        Sweep G — tek "hazır" tanımı: yalnızca `inspect_skill_bar` çıktısındaki
        `ready` alanı. Eskiden `ready` VEYA (`off_cooldown` ve `mana_ok`)
        dönüyordu; kilitli (0,15 sn içinde basılmış) skill de "hazır"
        listesine giriyordu. Fail-closed: a missing/empty frame (or no ready
        slot) returns `[]`, never the full slot list — callers must not press
        a skill they cannot see ready.

        Okunamayan kare ile "okundu ama hazır yok" ayrımı gerekiyorsa
        `get_ready_skills_or_none()` kullanın (`None` = okunamadı).
        """
        slot_states = self.inspect_skill_bar(frame)
        if not slot_states:
            return []
        return [
            info["slot"]
            for info in slot_states.values()
            if info.get("ready", False)
        ]

    def get_ready_skills_or_none(self, frame: Optional[np.ndarray]) -> Optional[List[str]]:
        """`get_ready_skills` gibi, ama okunamayan karede `None` döner.

        Sweep G (çağıran fail-open düzeltmesinin okuyucu tarafı):
        `combat_recovery.get_ready_combat_skills` boş liste alınca tüm
        `COMBAT_SKILL_KEYS`'i hazır sayıyordu — çünkü `[]` hem "okunamadı"
        hem "hazır yok" demekti. Bu metot ikisini ayırır: `None` =
        kare okunamadı / okuyucu hata verdi (bilinmiyor, hiçbir skill
        hazır sayılmaz); `[]` = kare okundu, hazır skill yok.
        """
        slot_states = self.inspect_skill_bar(frame)
        if not slot_states:
            return None
        return [
            info["slot"]
            for info in slot_states.values()
            if info.get("ready", False)
        ]

    def get_skill_bar_reader_ready_flag(self) -> dict:
        """Real readiness probe: ROI sane, 12 slots configured, required API present.

        Returns ready=False (with reasons) instead of an unconditional True.
        """
        problems: List[str] = []
        try:
            roi = self.current_skill_bar_roi
            if len(roi) != 4 or not (0.0 <= roi[0] < roi[2] <= 1.0 and 0.0 <= roi[1] < roi[3] <= 1.0):
                problems.append(f"invalid_skill_bar_roi={roi!r}")
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            problems.append(f"roi_error={exc}")
        if len(self.SKILL_SLOTS) != 12:
            problems.append("skill_slots_not_12")
        for name in ("inspect_skill_bar", "select_best_ready_skill", "get_ready_skills"):
            if not callable(getattr(self, name, None)):
                problems.append(f"missing_{name}")
        if not (0.0 < self.min_brightness_ready <= 255.0 and 0.0 <= self.min_contrast_std <= 255.0):
            problems.append("thresholds_out_of_range")
        # Sweep G bulgu 1/5: kalibre yokluğu `ready`'yi bozmaz (sözleşme
        # testi kalibresiz kurulumda ready bekler) ama artık görünür:
        # bayrak çıktısında kalibrasyon durumu + kaynak açıkça yer alır.
        try:
            roi_calibrated = bool(self.is_skill_bar_roi_calibrated)
            roi_source = str(self.skill_bar_roi_source)
        except Exception:
            roi_calibrated = False
            roi_source = "bilinmiyor"
        warnings: List[str] = []
        if not roi_calibrated:
            warnings.append(f"skill_bar_roi_not_calibrated: {roi_source}")
        return {
            "module": "skill_bar_reader",
            "ready": not problems,
            "problems": problems,
            "skill_bar_roi_calibrated": roi_calibrated,
            "skill_bar_roi_source": roi_source,
            "warnings": warnings,
        }

    def get_skill_reader_health_summary(self) -> dict:
        flag = self.get_skill_bar_reader_ready_flag()
        return {"skill_reader_ready": bool(flag.get("ready")), "problems": flag.get("problems", [])}

    def get_skill_bar_reader_capabilities(self) -> list:
        return ["skill_bar_reader", "smart_consumable_manager", "diagnostic_snapshot", "version_info"]

    def read_vitals_from_frame(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        """Convenience method reading player HP/MP and target HP using calibrated ROIs (<50ms).

        Reuses a single ManaPhaseCombat instance (was: a fresh one per call).
        Fail-closed: on any error the vitals are reported as unknown/critical
        (`vitals_valid: False`, HP/MP 0.0), never as fake full health.
        """
        try:
            from core.mana_phase_combat import ManaPhaseCombat
            if self._vitals_combat is None:
                self._vitals_combat = ManaPhaseCombat()
            return self._vitals_combat.read_vitals_from_frame(frame)
        except Exception:
            return {
                "player_hp": 0.0,
                "player_mp": 0.0,
                "target_hp": 0.0,
                "has_target": 0.0,
                "hp_potion_needed": True,
                "mp_potion_needed": True,
                "is_vitality_critical": True,
                "vitals_valid": False,
            }

    def get_skill_bar_reader_version_info(self) -> dict:
        # A18: eskiden "ready": True koşulsuz dönüyordu (plasebo). Artık
        # gerçek prob sonucu raporlanır.
        flag = self.get_skill_bar_reader_ready_flag()
        return {"module": "skill_bar_reader", "version": __version__, "ready": bool(flag.get("ready")), "problems": flag.get("problems", [])}

    def get_skill_bar_reader_diagnostic_snapshot(self) -> dict:
        mem = self.get_exemplar_memory()
        mem_loaded = mem is not None
        pos_cnt = len(getattr(mem, "_positive", [])) if mem else 0
        neg_cnt = len(getattr(mem, "_negative", [])) if mem else 0
        # A18: "status"/"ready" artık gerçek prob sonucudur (eskiden koşulsuz ok/True).
        flag = self.get_skill_bar_reader_ready_flag()
        ready = bool(flag.get("ready"))
        return {
            "module": "skill_bar_reader",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "problems": flag.get("problems", []),
            "current_roi": self.current_skill_bar_roi,
            "calibrated_rois_path": str(self.calibrated_rois_path),
            "last_roi_mtime": self._last_roi_mtime,
            "exemplar_memory_active": mem_loaded,
            "exemplar_positive_count": pos_cnt,
            "exemplar_negative_count": neg_cnt,
            "recent_slots_count": len(self._last_slot_states),
        }

    @property
    def consumable_manager(self) -> SmartConsumableManager:
        """Dynamically provides attached SmartConsumableManager."""
        if self._consumable_manager is None:
            self._consumable_manager = SmartConsumableManager(
                calibrated_rois_path=self.calibrated_rois_path,
                exemplar_memory=self.exemplar_memory,
            )
        return self._consumable_manager

    def inspect_consumables(self, frame: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Inspects upper consumable slots (F1..F8, X, B) via SmartConsumableManager."""
        return self.consumable_manager.inspect_consumables(frame)

    def get_consumable_manager(self) -> SmartConsumableManager:
        """Returns active SmartConsumableManager instance."""
        return self.consumable_manager


class MicroCameraLookAround:
    """
    Item 45 — İnsani Kamera Salınımı (`MicroCameraLookAround`).

    During long straight walks, every `15–25 seconds` (sampled from a Gaussian
    distribution centered at `20.0s`, clamped to `[15.0, 25.0]`), gently sways the
    camera by `±8°` left/right and smoothly returns to the original heading using
    `WindMouseEngine` micro-steps (`perform_micro_camera_sway(input_ctrl)`) so the
    character never looks like a rigid straight-line robot.
    """

    MIN_INTERVAL_SEC: float = 15.0
    MAX_INTERVAL_SEC: float = 25.0
    DEFAULT_SWAY_DEG: float = 8.0
    PIXELS_PER_DEGREE: float = 11.5

    def __init__(
        self,
        min_interval_sec: float = MIN_INTERVAL_SEC,
        max_interval_sec: float = MAX_INTERVAL_SEC,
        sway_degrees: float = DEFAULT_SWAY_DEG,
        pixels_per_degree: float = PIXELS_PER_DEGREE,
        windmouse: Optional[WindMouseEngine] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        # P2-3 (K-17): salinim suresince fare adimlarini serilestiren
        # ayri kilit. Durum kilidi (_lock) salinim boyunca tutulmaz;
        # bu kilit yalnizca iki salinimin fareyi ayni anda surmesini
        # engeller, durum okuyucularini bloklamaz. _lock ile hicbir
        # zaman ic ice tutulmaz (kilit sirasi riski yok).
        self._sway_exec_lock: threading.Lock = threading.Lock()
        self.min_interval_sec: float = max(5.0, float(min_interval_sec))
        self.max_interval_sec: float = max(self.min_interval_sec + 1.0, float(max_interval_sec))
        self.sway_degrees: float = max(1.0, min(25.0, float(sway_degrees)))
        self.pixels_per_degree: float = max(2.0, float(pixels_per_degree))
        self._rng: random.Random = random.Random(seed)
        self.windmouse: WindMouseEngine = windmouse or WindMouseEngine(auto_reveal_cursor=False)

        self._last_sway_ts: float = time.monotonic()
        self._next_interval_sec: float = self.sample_next_interval()
        self.sway_count: int = 0
        self.last_sway_summary: Dict[str, Any] = {}

    def sample_next_interval(self) -> float:
        """
        Samples the next look-around interval in `[15.0, 25.0]` seconds using a
        Gaussian distribution (`mean=20.0s, sigma=2.5s`).
        """
        with self._lock:
            mean_sec = 0.5 * (self.min_interval_sec + self.max_interval_sec)
            sigma_sec = (self.max_interval_sec - self.min_interval_sec) / 4.0
            sampled = self._rng.gauss(mean_sec, sigma_sec)
            clamped = float(np.clip(sampled, self.min_interval_sec, self.max_interval_sec))
            self._next_interval_sec = round(clamped, 3)
            return self._next_interval_sec

    def should_trigger_sway(self, now: Optional[float] = None) -> bool:
        """Returns True if at least `_next_interval_sec` (`15–25s`) has elapsed since the last sway."""
        current = float(now) if now is not None else time.monotonic()
        with self._lock:
            return (current - self._last_sway_ts) >= self._next_interval_sec

    @staticmethod
    def _resolve_cursor_pos(input_ctrl: Any) -> Optional[Tuple[int, int]]:
        """Cursor position, or None when it cannot be read.

        Fail-closed: never invent a position (the old (960, 540) fallback made
        sways compute trajectories from a made-up starting point).
        """
        if input_ctrl is not None:
            get_pos = getattr(input_ctrl, "get_cursor_position", None)
            if callable(get_pos):
                try:
                    pos = get_pos()
                    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                        return int(pos[0]), int(pos[1])
                except Exception as exc:
                    logger.debug("Could not query cursor position from input_ctrl: %s", exc)
        return None

    @staticmethod
    def _resolve_move_fn(input_ctrl: Any) -> Optional[Callable[[int, int], Any]]:
        if input_ctrl is None:
            return None
        # P0-5/K-04: `_sendinput_mouse_move` listeden ÇIKARILDI — yumuşak
        # fare yolu artık çözümlenmez; `_hw_move_to` yoksa genel
        # denetleyici API'leri denenir.
        for attr in ("_hw_move_to", "move_to", "move_mouse_abs", "set_cursor_pos"):
            fn = getattr(input_ctrl, attr, None)
            if callable(fn):
                return fn
        return None

    def perform_micro_camera_sway(
        self,
        input_ctrl: Any,
        direction: Optional[int] = None,
        angle_deg: Optional[float] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Gently sways the camera by `±8°` (`angle_deg`) left or right and smoothly returns
        to the original forward heading using WindMouse micro-steps so the character
        looks around naturally during long straight walks without altering its net path.

        Parameters
        ----------
        input_ctrl : Any
            The active `InputController` or mock input driver.
        direction : int, optional
            `-1` for left, `+1` for right. Randomly chosen if `None`.
        angle_deg : float, optional
            Target sway angle in degrees (defaults to `~8.0°` with slight Gaussian variance).
        force : bool
            If True, executes immediately and resets the 15–25s timer. If False
            (default), only executes if `should_trigger_sway()` is True — the old
            default (True) silently bypassed the 15–25s interval on every call.

        Returns `triggered: False` (never a fake success) when there is no input
        controller or the cursor position cannot be read: a sway that moves
        nothing must not report that it swayed.
        """
        now = time.monotonic()
        if input_ctrl is None:
            return {
                "triggered": False,
                "reason": "no_input_controller",
                "next_interval_sec": self._next_interval_sec,
            }
        if not force and not self.should_trigger_sway(now=now):
            return {
                "triggered": False,
                "reason": "interval_not_elapsed",
                "next_interval_sec": self._next_interval_sec,
            }

        # P2-3 (K-17): kilit yalnizca paylasilan durumu koruyacak kadar
        # kisa tutulur. Eski kod tum salinimi (WindMouse yolu + fare
        # adimlari + time.sleep'ler) `with self._lock:` icinde yapiyordu;
        # salinim suresince tum durum okuyuculari bloklaniyordu.
        # 1. asama (kilitli, kisa): salinim parametrelerini sec. RNG
        # cekisleri de burada yapilir (paylasilan _rng kilit disinda
        # kullanilmaz).
        with self._lock:
            chosen_dir = int(direction) if direction in (-1, 1) else self._rng.choice((-1, 1))
            base_deg = float(angle_deg) if angle_deg is not None else self.sway_degrees
            # Natural micro-variance around ±8° (e.g., 7.2° .. 8.8°)
            effective_deg = float(np.clip(self._rng.gauss(base_deg, 0.45), 5.5, 10.5))
            delta_px = int(round(chosen_dir * effective_deg * self.pixels_per_degree))
            if delta_px == 0:
                delta_px = chosen_dir * int(round(self.sway_degrees * self.pixels_per_degree))

            # Subtle vertical pitch micro-drift (±3..8 px) for organic human hand feel
            pitch_jitter_px = int(round(self._rng.gauss(0.0, 3.0)))
            pause_sec = float(self._rng.uniform(0.035, 0.075))
            next_interval_hint = self._next_interval_sec

        # 2. asama (durum kilidi DISINDA): imlec + yol + fare adimlari.
        # Iki salinimin fareyi karistirmamasi icin yalnizca salinimlar
        # arasi seri kilit tutulur; _lock ile ic ice degildir.
        with self._sway_exec_lock:
            cursor_pos = self._resolve_cursor_pos(input_ctrl)
            if cursor_pos is None:
                return {
                    "triggered": False,
                    "reason": "cursor_position_unavailable",
                    "next_interval_sec": next_interval_hint,
                }
            start_x, start_y = cursor_pos
            peak_x = start_x + delta_px
            peak_y = start_y + pitch_jitter_px

            # Generate smooth WindMouse outward trajectory (center -> ±8°) and return trajectory (±8° -> center)
            outward_path = self.windmouse.generate_windmouse_path(
                start_x=start_x,
                start_y=start_y,
                dest_x=peak_x,
                dest_y=peak_y,
                gravity=9.5,
                wind=2.4,
                max_step=14.0,
            )
            return_path = self.windmouse.generate_windmouse_path(
                start_x=peak_x,
                start_y=peak_y,
                dest_x=start_x,
                dest_y=start_y,
                gravity=10.0,
                wind=2.0,
                max_step=14.0,
            )

            move_fn = self._resolve_move_fn(input_ctrl)
            is_test_mode = bool(getattr(input_ctrl, "test_mode", False)) if input_ctrl is not None else True

            executed_steps = 0
            # P0-5: hiçbir adım kabul edilmediyse sway olmamıştır — adım
            # dönüşü yok sayılıp "triggered: True" dönülmez. None = eski
            # sürücü (dönüşsüz kabul), yalnızca açık False başarısızlıktır.
            step_ok = False
            for step in outward_path:
                px, py = int(step[0]), int(step[1])
                dt = float(step[2]) if len(step) >= 3 else 0.0025
                if move_fn is not None:
                    try:
                        if move_fn(px, py) is not False:
                            step_ok = True
                    except Exception as exc:
                        logger.debug("MicroCameraLookAround outward step error: %s", exc)
                executed_steps += 1
                if not is_test_mode:
                    time.sleep(max(0.001, min(0.012, dt)))

            if not is_test_mode:
                time.sleep(pause_sec)

            for step in return_path:
                px, py = int(step[0]), int(step[1])
                dt = float(step[2]) if len(step) >= 3 else 0.0025
                if move_fn is not None:
                    try:
                        if move_fn(px, py) is not False:
                            step_ok = True
                    except Exception as exc:
                        logger.debug("MicroCameraLookAround return step error: %s", exc)
                executed_steps += 1
                if not is_test_mode:
                    time.sleep(max(0.001, min(0.012, dt)))

            if move_fn is None or not step_ok:
                return {
                    "triggered": False,
                    "reason": "mouse_move_rejected",
                    "next_interval_sec": self._next_interval_sec,
                }


        # 3. asama (kilitli, kisa): zaman damgasi + sayac + ozet kaydi.
        with self._lock:
            self._last_sway_ts = time.monotonic()
            next_interval = self.sample_next_interval()
            self.sway_count += 1

            summary = {
                "triggered": True,
                "direction": "RIGHT" if chosen_dir > 0 else "LEFT",
                "angle_deg": round(effective_deg * chosen_dir, 2),
                "delta_px": delta_px,
                "start_pos": (start_x, start_y),
                "peak_pos": (peak_x, peak_y),
                "end_pos": (start_x, start_y),
                "outward_steps": len(outward_path),
                "return_steps": len(return_path),
                "total_micro_steps": executed_steps,
                "next_interval_sec": next_interval,
                "sway_count": self.sway_count,
            }
            self.last_sway_summary = summary
            return summary

    # NOTE: the skill-bar readiness/capability helpers used to live here as
    # tautologies (always "ready: True"). They now live on SkillBarReader as
    # real checks (see SkillBarReader.get_skill_bar_reader_ready_flag).


def get_skill_bar_reader_capabilities() -> list:
    return [
        "skill_bar_reader",
        "smart_consumable_manager",
        "dynamic_roi_reload",
        "exemplar_memory_confidence",
        "cooldown_lockout_tracking",
        "buff_scroll_expiration_tracking",
        "diagnostic_snapshot",
        "version_info",
    ]


def validate_skill_bar_reader_contract() -> bool:
    """Real contract check: the reader class exposes its API, sane ROI/thresholds,
    and 12 configured slots. (Was: `bool(capability_list)` — always True.)"""
    try:
        reader = SkillBarReader(calibrated_rois_path=Path("__nonexistent_rois__.json"))
        flag = reader.get_skill_bar_reader_ready_flag()
        if not flag.get("ready"):
            return False
        caps = get_skill_bar_reader_capabilities()
        return bool(caps) and "skill_bar_reader" in caps and "inspect_skill_bar" in dir(reader)
    except Exception:
        return False


# The readiness/audit/telemetry helpers below used to nest into each other,
# so one `get_skill_bar_reader_heartbeat_snapshot()` call ran
# `validate_skill_bar_reader_contract()` 10 times (each constructing a
# SkillBarReader and re-reading the ROI file). The public functions still
# return identical dicts, but each now validates ONCE and the nesting is
# built by one helper from that single boolean.
def _skill_bar_reader_chain(contract_valid: bool) -> Dict[str, Dict[str, Any]]:
    """Builds every nested readiness/audit/telemetry dict from one validation
    result. Public functions below return the matching piece."""
    baseline = {"module": "skill_bar_reader", "contract_valid": contract_valid,
                "capability_count": len(get_skill_bar_reader_capabilities())}
    readiness = {"module": "skill_bar_reader", "ready": contract_valid, "metrics": baseline}
    audit_record = {"module": "skill_bar_reader", "audit_passed": contract_valid, "readiness": readiness}
    digest = {"module": "skill_bar_reader", "operational": contract_valid, "audit": audit_record}
    telemetry = {"module": "skill_bar_reader", "telemetry_ready": contract_valid, "digest": digest}
    seal = {"module": "skill_bar_reader", "sealed": contract_valid, "telemetry": telemetry}
    certificate = {"module": "skill_bar_reader", "certified": contract_valid, "seal": seal}
    probe = {"module": "skill_bar_reader", "ready": contract_valid, "certificate": certificate}
    summary = {"module": "skill_bar_reader", "audit_passed": contract_valid, "probe": probe}
    return {
        "baseline": baseline, "readiness": readiness, "audit_record": audit_record,
        "digest": digest, "telemetry": telemetry, "seal": seal,
        "certificate": certificate, "probe": probe, "summary": summary,
    }


def get_skill_bar_reader_baseline_metrics() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["baseline"]


def get_skill_bar_reader_readiness_status() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["readiness"]


def get_skill_bar_reader_audit_record() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["audit_record"]


def get_skill_bar_reader_operational_digest() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["digest"]


def get_skill_bar_reader_telemetry_payload() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["telemetry"]


def get_skill_bar_reader_integrity_seal() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["seal"]


def get_skill_bar_reader_compliance_certificate() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["certificate"]


def get_skill_bar_reader_readiness_probe() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["probe"]


def get_skill_bar_reader_audit_summary() -> dict:
    return _skill_bar_reader_chain(validate_skill_bar_reader_contract())["summary"]


def get_skill_bar_reader_heartbeat_snapshot() -> dict:
    valid = validate_skill_bar_reader_contract()
    return {"module": "skill_bar_reader", "alive": valid, "audit": _skill_bar_reader_chain(valid)["summary"]}


# Alias for backwards compatibility with combat_recovery and perception subsystems
AdaptiveSkillBarReader = SkillBarReader

CONSUMABLE_SLOTS: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "X", "B")


class ConsumableActionResult(dict):
    """
    Hybrid dict/key result returned by `SmartConsumableManager.decide_consumable_action`.
    Works both as a full diagnostic dictionary (`res["hp_potion"]`, `res["keys_to_press"]`, etc.)
    and compares equal to the primary chosen key string (`res == "F1"`, `res in ("F1", "F3")`).
    """

    def __hash__(self) -> int:
        # Hash by primary key, consistent with __eq__ against str keys.
        return hash(str(self.get("key") or ""))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            primary = self.get("key")
            if primary is not None and primary == other:
                return True
            if other in self.get("keys_to_press", ()):
                return True
            if self.get("action") == other:
                return True
            return False
        if other is None:
            return self.get("key") is None and not self.get("keys_to_press")
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if item in self.get("keys_to_press", ()):
            return True
        if item is not None and item == self.get("key"):
            return True
        return False

    def __str__(self) -> str:
        # The full diagnostic dict: str() used to return only the primary key,
        # silently discarding every other field. Use .get("key") for the key.
        return dict.__str__(self)

    def upper(self) -> str:
        return str(self.get("key") or "").upper()

    def lower(self) -> str:
        return str(self.get("key") or "").lower()

    def strip(self, chars: Optional[str] = None) -> str:
        return str(self.get("key") or "").strip(chars)

    def startswith(self, prefix: Any, *args: Any) -> bool:
        return str(self.get("key") or "").startswith(prefix, *args)


class SmartConsumableManager(metaclass=_RoiMeta):
    """
    Item 46 — Akıllı Tüketilebilir (Pot ve Parşömen) Yöneticisi (`SmartConsumableManager`).

    Inspects upper consumable hotbar row (`y ≈ 0.870 .. 0.938*H`, keybinds `F1..F8, X, B`):
    - Differentiates consumables from lower combat skills (`y >= 0.940*H`).
    - Learns slot assignments dynamically using pixel color hue density & ExemplarMemory:
        * hp_potion: Red liquid hue density (H < 10 or H > 168, S >= 50, V >= 35).
        * mp_potion: Blue liquid hue density (H 90..140) or Yellow liquid hue density (H 18..42, high S).
        * buff_scroll: Beige parchment paper icon (`H:10..28, S:30..175, V:85..235`).
        * empty: Low saturation / low contrast (`saturation < 30` or `brightness < 35`).
    - Tracks cooldown / readiness state of each slot (`ready` vs `depleted`/`on_cooldown`).
    - Tracks buff duration (`scroll_duration_sec=300.0`) and expiration timers for scroll slots (`F5, F7`; F6 is an automatic client slot and forbidden).
    - Connects with `ExemplarMemory` so user approvals/rejections directly teach the manager.
    """

    DEFAULT_CONSUMABLE_ROW_ROI: Tuple[float, float, float, float] = (0.345, 0.870, 0.655, 0.938)
    CONSUMABLE_ROW_ROI: Any = _ConsumableBarRoiDescriptor(DEFAULT_CONSUMABLE_ROW_ROI)
    CONSUMABLE_SLOTS: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "X", "B")
    SCROLL_SLOTS: Tuple[str, ...] = ("F5", "F7")
    FORBIDDEN_MANUAL_SLOTS: Tuple[str, ...] = ("X", "F6", "B")

    CLASS_ID_HP_POTION: int = 22
    CLASS_NAME_HP_POTION: str = "hp_potion"
    CLASS_ID_MP_POTION: int = 23
    CLASS_NAME_MP_POTION: str = "mp_potion"
    CLASS_ID_BUFF_SCROLL: int = 24
    CLASS_NAME_BUFF_SCROLL: str = "buff_scroll"
    # P1-1: "empty" / "automatic" are SLOT STATES, not classes — they live
    # in their own namespace now (dataset_factory.class_taxonomy:
    # SLOT_STATE_EMPTY_ID = -100, SLOT_STATE_AUTOMATIC_ID = -101). They
    # used to reuse the canonical ids 25 (kinah) and 26
    # (player_character), so an "empty"-slot exemplar was stored — and
    # read back — as a KINAH exemplar (Y-3). The new ids can never be
    # mistaken for a canonical class: no resolver accepts them, the
    # exemplar memory only admits them on the NEGATIVE list, and the
    # node gate refuses them as a verdict class. 22/23/24 above stay
    # canonical (data.yaml: 22=hp_potion, 23=mp_potion, 24=buff_scroll).
    # Positive exemplar matching still keys on the NAME for "empty"
    # (classify dallarına bak) — cid'ye bakılarak "empty" sayılmaz.
    SLOT_STATE_EMPTY_ID: int = SLOT_STATE_EMPTY_ID
    CLASS_NAME_EMPTY: str = "empty"
    SLOT_STATE_AUTOMATIC_ID: int = SLOT_STATE_AUTOMATIC_ID
    CLASS_NAME_AUTOMATIC: str = "automatic"
    # DEPRECATED aliases (P1-1): the old names survive with the NEW
    # (non-canonical) values so any missed caller keeps working with a
    # safe id instead of AttributeError — do not use in new code.
    CLASS_ID_EMPTY: int = SLOT_STATE_EMPTY_ID
    CLASS_ID_AUTOMATIC: int = SLOT_STATE_AUTOMATIC_ID

    DEFAULT_SLOT_ASSIGNMENTS: Dict[str, str] = {
        "F1": "hp_potion",
        "F2": "mp_potion",
        "F3": "hp_potion",
        "F4": "mp_potion",
        "F5": "buff_scroll",
        "F6": "automatic",
        "F7": "buff_scroll",
        "F8": "empty",
        "X": "automatic",
        "B": "automatic",
    }

    @classmethod
    def _resolve_consumable_roi_static(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        fallback: Tuple[float, float, float, float] = (0.345, 0.870, 0.655, 0.938),
    ) -> Tuple[float, float, float, float]:
        # Thin wrapper over the shared resolver (see _resolve_roi_from_json).
        return _resolve_roi_from_json(
            json_path,
            ("consumable_bar", "consumables", "upper_skill_bar", "consumable_row", "upper_consumable_row"),
            fallback,
            calibrated_keys=("consumable_bar", "consumables", "upper_skill_bar"),
        )

    def __init__(
        self,
        calibrated_rois_path: Optional[Union[str, Path]] = None,
        exemplar_memory: Optional[Any] = None,
        consumable_roi: Optional[Sequence[float]] = None,
        default_hp_cooldown_sec: float = 10.0,
        default_mp_cooldown_sec: float = 10.0,
        default_scroll_cooldown_sec: float = 2.0,
        default_buff_duration_sec: float = 300.0,
        scroll_duration_sec: Optional[float] = None,
        hp_threshold: Optional[float] = None,
        mp_threshold: Optional[float] = None,
        seed_defaults: bool = True,
        forbidden_slots: Optional[Sequence[str]] = None,
        # P1-13: yapılandırılmış buff parşömen tuşu (route.buff_scroll_key).
        preferred_scroll_key: Optional[str] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.calibrated_rois_path: Path = (
            Path(calibrated_rois_path) if calibrated_rois_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        )
        self.exemplar_memory: Optional[Any] = exemplar_memory
        if forbidden_slots is not None:
            self.forbidden_slots: Set[str] = {str(s).strip().upper() for s in forbidden_slots}
        elif seed_defaults:
            self.forbidden_slots = {"X", "F6", "B"}
        else:
            self.forbidden_slots = set()
        self.default_hp_cooldown_sec: float = max(0.5, float(default_hp_cooldown_sec))
        self.default_mp_cooldown_sec: float = max(0.5, float(default_mp_cooldown_sec))
        self.default_scroll_cooldown_sec: float = max(0.2, float(default_scroll_cooldown_sec))

        eff_scroll_dur = float(scroll_duration_sec) if scroll_duration_sec is not None else float(default_buff_duration_sec)
        self.scroll_duration_sec: float = max(1.0, eff_scroll_dur)
        self.default_buff_duration_sec: float = self.scroll_duration_sec
        # F-0540 tek kaynak: None gelirse config/schema.py sabitine (65/45)
        # çözülür; değerler oran (0..1) cinsindendir.
        self.hp_threshold: float = (
            float(HP_POTION_THRESHOLD_PCT) / 100.0 if hp_threshold is None else float(hp_threshold)
        )
        self.mp_threshold: float = (
            float(MP_POTION_THRESHOLD_PCT) / 100.0 if mp_threshold is None else float(mp_threshold)
        )

        self._custom_roi: Optional[Tuple[float, float, float, float]] = None
        if consumable_roi is not None and len(consumable_roi) == 4:
            self._custom_roi = (
                float(consumable_roi[0]),
                float(consumable_roi[1]),
                float(consumable_roi[2]),
                float(consumable_roi[3]),
            )
        self._cached_roi: Optional[Tuple[float, float, float, float]] = None
        self._last_roi_mtime: float = -1.0

        self._slot_assignments: Dict[str, str] = {}
        self._slot_confidences: Dict[str, float] = {}
        self._slot_last_used: Dict[str, float] = {}
        self._slot_cooldowns: Dict[str, float] = {}
        self._buff_durations: Dict[str, float] = {}
        self._buff_applied_at: Dict[str, Optional[float]] = {}
        self._buff_expires_at: Dict[str, Optional[float]] = {}
        self._last_diagnostics: Dict[str, Dict[str, Any]] = {}

        self._exemplar_store: Dict[str, List[Dict[str, Any]]] = {
            "hp_potion": [],
            "mp_potion": [],
            "buff_scroll": [],
            "empty": [],
        }
        self._negative_exemplar_store: List[Dict[str, Any]] = []

        if seed_defaults:
            for s, itype in self.DEFAULT_SLOT_ASSIGNMENTS.items():
                self._slot_assignments[s] = itype
                self._slot_confidences[s] = 0.85
                if itype == "buff_scroll":
                    self._buff_durations[s] = self.scroll_duration_sec
                    self._slot_cooldowns[s] = self.default_scroll_cooldown_sec
                elif itype == "hp_potion":
                    self._slot_cooldowns[s] = self.default_hp_cooldown_sec
                elif itype == "mp_potion":
                    self._slot_cooldowns[s] = self.default_mp_cooldown_sec

        # P1-13: yapılandırılmış parşömen tuşunu buff_scroll slotu olarak
        # işaretle (varsayılan F7 zaten tohumlu; davranış değişmez).
        if preferred_scroll_key is not None:
            self.set_preferred_scroll_key(preferred_scroll_key)

    @staticmethod
    def normalize_key(slot_key: str) -> str:
        s = str(slot_key or "").strip().upper()
        return s

    def get_exemplar_memory(self) -> Optional[Any]:
        if self.exemplar_memory is not None:
            return self.exemplar_memory
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("SmartConsumableManager could not resolve ExemplarMemory: %s", exc)
            return None

    def get_consumable_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        if self._custom_roi is not None:
            return self._custom_roi

        json_path = self.calibrated_rois_path
        if not json_path or not json_path.is_file():
            return self.DEFAULT_CONSUMABLE_ROW_ROI

        try:
            mtime = json_path.stat().st_mtime
            if not force and mtime == self._last_roi_mtime and self._cached_roi is not None:
                return self._cached_roi

            with self._lock:
                rect = self._resolve_consumable_roi_static(
                    json_path=json_path,
                    fallback=self.DEFAULT_CONSUMABLE_ROW_ROI,
                )
                self._cached_roi = rect
                self._last_roi_mtime = mtime
                return rect
        except Exception as exc:
            logger.debug("SmartConsumableManager get_consumable_roi error: %s", exc)
            return self._cached_roi or self.DEFAULT_CONSUMABLE_ROW_ROI

    def check_reload_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi(force=force)

    def reload_calibrated_rois(self, force: bool = True) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi(force=force)

    def set_custom_roi(self, roi: Sequence[float]) -> None:
        if len(roi) == 4:
            self._custom_roi = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))

    @property
    def current_consumable_roi(self) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi()

    @property
    def current_consumable_bar_roi(self) -> Tuple[float, float, float, float]:
        """Convenience alias for current_consumable_roi."""
        return self.get_consumable_roi()

    def slot_center_px(
        self,
        frame: Optional[np.ndarray],
        slot_key: str,
        jitter_px: int = 3,
    ) -> Optional[Tuple[int, int]]:
        try:
            roi = self.current_consumable_roi
        except Exception as exc:
            logger.debug("SmartConsumableManager slot_center_px failed (%s): %s", slot_key, exc)
            return None
        return _slot_center_px(frame, slot_key, self.CONSUMABLE_SLOTS, roi, jitter_px)

    def classify_slot_crop(
        self,
        slot_img: Optional[np.ndarray],
        slot_key: str = "",
        *,
        precomputed_hsv: Optional[np.ndarray] = None,
        precomputed_gray: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Classifies one consumable slot crop.

        `precomputed_hsv` / `precomputed_gray` are optional whole-slot HSV /
        grayscale images (same shape as `slot_img`, e.g. sliced from a
        row-level conversion by `inspect_consumables`) — they skip the two
        per-slot `cv2.cvtColor` calls. A shape mismatch falls back to
        converting the crop itself, so results are identical either way.
        """
        if slot_img is None or not isinstance(slot_img, np.ndarray) or slot_img.size == 0:
            return {
                "item_type": "empty",
                "state": "empty",
                "ready": False,
                "confidence": 0.0,
                "red_density": 0.0,
                "blue_density": 0.0,
                "yellow_density": 0.0,
                "parchment_density": 0.0,
                "brightness": 0.0,
                "saturation": 0.0,
                "contrast_std": 0.0,
                "exemplar_type": None,
                "exemplar_sim": 0.0,
                "is_empty": True,
                "is_negative": False,
            }

        if slot_img.ndim == 2:
            img_bgr = cv2.cvtColor(slot_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = slot_img[:, :, :3]

        sh, sw = img_bgr.shape[:2]
        if sh >= 8 and sw >= 8:
            pad_y = max(1, int(round(sh * 0.12)))
            pad_x = max(1, int(round(sw * 0.12)))
            y0, y1 = pad_y, max(pad_y + 2, sh - pad_y)
            x0, x1 = pad_x, max(pad_x + 2, sw - pad_x)
            inner = img_bgr[y0:y1, x0:x1]
            if inner.size == 0:
                inner = img_bgr
                y0, y1, x0, x1 = 0, sh, 0, sw
        else:
            inner = img_bgr
            y0, y1, x0, x1 = 0, sh, 0, sw

        # HSV / grayscale of the *same* inner pixels. Callers that already
        # converted the whole row pass the precomputed arrays (sliced to this
        # slot); otherwise convert just this crop. Pixel values are identical
        # either way (cvtColor is per-pixel), so metrics are unchanged.
        if (
            precomputed_hsv is not None
            and precomputed_hsv.ndim == 3
            and precomputed_hsv.shape[:2] == img_bgr.shape[:2]
            and precomputed_hsv.shape[2] == 3
        ):
            hsv = precomputed_hsv[y0:y1, x0:x1]
        else:
            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        if (
            precomputed_gray is not None
            and precomputed_gray.ndim == 2
            and precomputed_gray.shape[:2] == img_bgr.shape[:2]
        ):
            gray_inner = precomputed_gray[y0:y1, x0:x1]
        else:
            gray_inner = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        brightness = float(np.mean(v_chan))
        saturation = float(np.mean(s_chan))
        # Spatial contrast (grayscale std); np.std over flattened BGR measured
        # cross-channel spread, so a flat single-colour patch looked "contrasty".
        contrast_std = float(np.std(gray_inner))

        b_chan = inner[:, :, 0].astype(np.int16)
        g_chan = inner[:, :, 1].astype(np.int16)
        r_chan = inner[:, :, 2].astype(np.int16)

        # 1. Red liquid mask (HP Potion: kırmızı sıvı)
        red_mask = (
            ((h_chan < 10) | (h_chan > 168))
            & (s_chan >= 50)
            & (v_chan >= 25)
            & (r_chan > b_chan + 12)
            & (r_chan > g_chan + 10)
        )
        red_density = float(np.mean(red_mask))

        # 2. Blue liquid mask (MP Potion: mavi sıvı)
        blue_mask = (
            (h_chan >= 90)
            & (h_chan <= 140)
            & (s_chan >= 45)
            & (v_chan >= 25)
            & (b_chan > r_chan + 12)
        )
        blue_density = float(np.mean(blue_mask))

        # 3. Beige parchment mask (Buff Scroll: bej parşömen kağıdı H:10..28, S:30..175, V:85..235)
        parchment_mask = (
            (h_chan >= 10)
            & (h_chan <= 28)
            & (s_chan >= 30)
            & (s_chan <= 175)
            & (v_chan >= 85)
            & (v_chan <= 235)
        )
        parchment_density = float(np.mean(parchment_mask))

        # 4. Yellow liquid mask (MP Potion: sarı sıvı — high saturation S > 175 or H:29..42)
        yellow_mask = (
            (((h_chan >= 18) & (h_chan <= 42) & (s_chan > 175) & (v_chan >= 45))
             | ((h_chan >= 29) & (h_chan <= 42) & (s_chan >= 75) & (v_chan >= 45)))
            & (r_chan > b_chan + 15)
            & (g_chan > b_chan + 15)
        )
        yellow_density = float(np.mean(yellow_mask))

        max_color_density = max(red_density, blue_density, yellow_density, parchment_density)
        is_empty = bool(
            max_color_density < 0.12
            and (
                saturation < 30.0
                or brightness < 38.0
                or (brightness < 55.0 and contrast_std < 18.0)
            )
        )

        exemplar_type: Optional[str] = None
        exemplar_sim: float = 0.0
        is_negative: bool = False

        # P2-3 (K-17): exemplar depolari paylasilan durumdur
        # (learn_slot_assignment ayni depolari kilit altinda yazar).
        # Eger sorgu depolari kilitsiz okursa, eszamanli ogrenme
        # dict/list degisimi sirasinda "changed size during iteration"
        # uretebilir; sorguyu kilit altinda tutmak ise her slotta
        # ~4ms+ feature cikarma + eslestirmeyi kilide sokar. Cozum:
        # depolari kisa kilitle kopyala (sig), eslestirmeyi kilitsiz
        # kopya uzerinde yap. Her iki depo da bosken (varsayilan)
        # cikarma hic calismaz — ayni cikti, ~10x hizli inspect yolu.
        with self._lock:
            _neg_exemplars = list(self._negative_exemplar_store)
            _pos_exemplars = {
                itype: list(ex_list) for itype, ex_list in self._exemplar_store.items()
            }
        _has_internal_exemplars = bool(
            _neg_exemplars or any(_pos_exemplars.values())
        )
        if (
            _has_internal_exemplars
            and extract_crop_features is not None
            and compute_similarity is not None
        ):
            try:
                feats = extract_crop_features(inner)
                if feats is not None:
                    for neg_ex in _neg_exemplars:
                        sim = compute_similarity(feats, neg_ex)
                        if sim >= 0.78:
                            is_negative = True
                            exemplar_sim = max(exemplar_sim, sim)
                            break
                    if not is_negative:
                        for itype, ex_list in _pos_exemplars.items():
                            for ex in ex_list:
                                sim = compute_similarity(feats, ex)
                                if sim > exemplar_sim:
                                    exemplar_sim = sim
                                    exemplar_type = itype
            except Exception as exc:
                logger.debug("Internal exemplar store matching error: %s", exc)

        mem = self.get_exemplar_memory()
        if mem is not None:
            try:
                if hasattr(mem, "is_negative_patch"):
                    is_neg, neg_sim, neg_meta = mem.is_negative_patch(inner, threshold=EXEMPLAR_MATCH_THRESHOLD)
                    if is_neg:
                        neg_cid = neg_meta.get("class_id") if isinstance(neg_meta, dict) else getattr(neg_meta, "class_id", None)
                        neg_src = str(neg_meta.get("source", "") if isinstance(neg_meta, dict) else getattr(neg_meta, "source", "")).lower()
                        neg_cname = str(neg_meta.get("class_name", "") if isinstance(neg_meta, dict) else getattr(neg_meta, "class_name", "")).lower()
                        if (
                            neg_cid in (
                                self.CLASS_ID_HP_POTION,
                                self.CLASS_ID_MP_POTION,
                                self.CLASS_ID_BUFF_SCROLL,
                                self.SLOT_STATE_EMPTY_ID,
                            )
                            or "consumable" in neg_src
                            or "consumable" in neg_cname
                            or "rejected" in neg_cname
                            or "empty" in neg_cname
                        ):
                            is_negative = True
                            exemplar_sim = max(exemplar_sim, float(neg_sim))
                if hasattr(mem, "classify_patch"):
                    pos_cid, pos_cname, pos_sim = mem.classify_patch(inner, min_sim=EXEMPLAR_MATCH_THRESHOLD)
                    if pos_cid is not None and float(pos_sim) > exemplar_sim:
                        # Kutu kimliği class_id'dir, isim alt-dizesi değil:
                        # (18, 'target_hp_bar') bir HP BAR'dır, HP potion
                        # değil — ama "hp" alt-dizesi onu hp_potion sayıyordu
                        # (aynı tuzak 16 'player_hp_bar' / 17 'player_mp_bar'
                        # için de geçerli). Yalnızca 22/23/24 (potion/scroll)
                        # eşleşir; "empty" dalı isimle kalır (Y-3: cid 25
                        # pozitif tarafta KINAH'dır, "empty" değil).
                        cname = str(pos_cname or "").lower()
                        if pos_cid == self.CLASS_ID_HP_POTION:
                            exemplar_type = "hp_potion"
                            exemplar_sim = float(pos_sim)
                        elif pos_cid == self.CLASS_ID_MP_POTION:
                            exemplar_type = "mp_potion"
                            exemplar_sim = float(pos_sim)
                        elif pos_cid == self.CLASS_ID_BUFF_SCROLL:
                            exemplar_type = "buff_scroll"
                            exemplar_sim = float(pos_sim)
                        elif "empty" in cname:
                            # Y-3 / P1-1: cid 25 pozitif tarafta KINAH'dir,
                            # "empty" degil — cid'ye bakilmaz, yalnizca isme
                            # bakilir. Eski kod `pos_cid == CLASS_ID_EMPTY`
                            # (25) ile gercek bir kinah eslesmesini "empty"
                            # sayiyordu; "empty" artik -100 ad alaninda,
                            # pozitif tarafta hicbir karsiligi yok.
                            exemplar_type = "empty"
                            exemplar_sim = float(pos_sim)
                if hasattr(mem, "match_class"):
                    # Y-3: "empty" icin pozitif sorgu YOK — pozitif "empty"
                    # sinifi diye bir sey yok; (25, "empty") sorgusu kinah
                    # pozitiflerini dondurup "empty" diye etiketliyordu.
                    for cid, cname in (
                        (self.CLASS_ID_HP_POTION, "hp_potion"),
                        (self.CLASS_ID_MP_POTION, "mp_potion"),
                        (self.CLASS_ID_BUFF_SCROLL, "buff_scroll"),
                    ):
                        m_sim, _ = mem.match_class(inner, cid)
                        if float(m_sim) > exemplar_sim and float(m_sim) >= EXEMPLAR_MATCH_THRESHOLD:
                            exemplar_sim = float(m_sim)
                            exemplar_type = cname
            except Exception as exc:
                logger.debug("External ExemplarMemory query error: %s", exc)

        norm_key = self.normalize_key(slot_key)
        # P2-3 (K-17): atama/guven okumasi kisa kilitle; feature
        # cikarma + ExemplarMemory sorgulari kilitsiz yukarida yapildi.
        with self._lock:
            stored_type = self._slot_assignments.get(norm_key)
            stored_conf = float(self._slot_confidences.get(norm_key, 0.60))

        if is_negative:
            predicted_type = "empty"
            confidence = round(exemplar_sim, 3)
        elif exemplar_type is not None and exemplar_sim >= EXEMPLAR_MATCH_THRESHOLD:
            predicted_type = exemplar_type
            confidence = round(exemplar_sim, 3)
        elif max_color_density >= 0.12:
            mp_density = max(blue_density, yellow_density)
            if red_density >= max(mp_density, parchment_density):
                predicted_type = "hp_potion"
                confidence = round(float(np.clip(red_density * 1.5 + 0.35, 0.65, 0.98)), 3)
            elif parchment_density >= max(red_density, mp_density):
                predicted_type = "buff_scroll"
                confidence = round(float(np.clip(parchment_density * 1.4 + 0.35, 0.65, 0.98)), 3)
            else:
                predicted_type = "mp_potion"
                confidence = round(float(np.clip(mp_density * 1.5 + 0.35, 0.65, 0.98)), 3)
        elif is_empty or not norm_key:
            predicted_type = "empty"
            confidence = 0.85
        elif stored_type is not None:
            predicted_type = stored_type
            confidence = round(stored_conf, 3)
        else:
            predicted_type = "empty"
            confidence = 0.50

        return {
            "item_type": predicted_type,
            "confidence": confidence,
            "is_empty": is_empty,
            "is_negative": is_negative,
            "exemplar_type": exemplar_type,
            "exemplar_sim": round(exemplar_sim, 3),
            "red_density": round(red_density, 4),
            "blue_density": round(blue_density, 4),
            "yellow_density": round(yellow_density, 4),
            "parchment_density": round(parchment_density, 4),
            "brightness": round(brightness, 2),
            "saturation": round(saturation, 2),
            "contrast_std": round(contrast_std, 2),
        }

    def classify_consumable_slot_crop(
        self,
        crop_bgr: Optional[np.ndarray],
        slot_key: str = "",
    ) -> str:
        """
        Classifies a single consumable slot crop into one of:
        - `"hp_potion"` (red liquid)
        - `"mp_potion"` (blue or yellow liquid)
        - `"buff_scroll"` (beige parchment paper `H:10..28, S:30..175, V:85..235`)
        - `"empty"` (low saturation / low contrast)
        Also matches against `ExemplarMemory`.
        """
        diag = self.classify_slot_crop(crop_bgr, slot_key=slot_key)
        itype = str(diag.get("item_type", "empty"))
        if itype not in ("hp_potion", "mp_potion", "buff_scroll", "empty", "automatic"):
            return "empty"
        return itype

    def inspect_consumables(
        self,
        frame: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {}
        try:
            fh, fw = frame.shape[:2]
            roi = self.current_consumable_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))

            row_crop = frame[y1:y2, x1:x2]
            if row_crop.size == 0:
                return {}

            if row_crop.ndim == 2:
                row_bgr = cv2.cvtColor(row_crop, cv2.COLOR_GRAY2BGR)
            else:
                row_bgr = row_crop[:, :, :3]

            # Row-level HSV/gray, sliced per slot below (same hoist as
            # SkillBarReader.inspect_skill_bar: 10 slots x 2 conversions -> 2).
            row_hsv = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2HSV)
            row_gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)

            bh, bw = row_bgr.shape[:2]
            num_slots = len(self.CONSUMABLE_SLOTS)
            slot_w = float(bw) / float(num_slots)

            t = time.monotonic() if now is None else float(now)
            results: Dict[str, Dict[str, Any]] = {}
            is_blank_frame = bool(np.max(row_bgr) == 0)

            # P2-3 (K-17): eski kod tum slot dongusunu (OpenCV +
            # exemplar sorgulari dahil) `with self._lock:` icinde
            # tutuyordu. Paylasilan durum kisa kilitle kopyalanir;
            # siniflandirma + hesap kilitsiz; yazim sonda tek kisa
            # kilitle yapilir. classify_slot_crop kendi icinde de
            # yalnizca kisa kilitli snapshot okur.
            with self._lock:
                snap_assignments = dict(self._slot_assignments)
                snap_last_used = dict(self._slot_last_used)
                snap_cooldowns = dict(self._slot_cooldowns)
                snap_buff_applied = dict(self._buff_applied_at)
                snap_buff_expires = dict(self._buff_expires_at)
            # Bu kareden ogrenilen atama/guven: dongu sonunda tek
            # kisa kilitle geri yazilir (CAS korumali, asagida).
            pending_assignments: Dict[str, Tuple[str, float]] = {}
            for idx, slot_key in enumerate(self.CONSUMABLE_SLOTS):
                norm_key = self.normalize_key(slot_key)
                sx1 = max(0, int(round(idx * slot_w)))
                sx2 = max(sx1 + 2, min(bw, int(round((idx + 1) * slot_w))))
                slot_img = row_bgr[:, sx1:sx2]

                # Pass the real slot key so classify_slot_crop can consult the
                # stored assignment for ambiguous crops (was: slot_key=""),
                # plus the precomputed row-level HSV/gray slices so it does
                # not re-convert this slot from BGR on its own.
                diag = self.classify_slot_crop(
                    slot_img,
                    slot_key=norm_key,
                    precomputed_hsv=row_hsv[:, sx1:sx2],
                    precomputed_gray=row_gray[:, sx1:sx2],
                )
                classified_type = diag["item_type"]
                conf = diag["confidence"]

                last_used = snap_last_used.get(norm_key, 0.0)

                if not is_blank_frame and classified_type in ("hp_potion", "mp_potion", "buff_scroll"):
                    # P2-3: yazim dongu sonunda tek kisa kilitle
                    # yapilir (pending_assignments); burada yalnizca
                    # karar kaydedilir.
                    pending_assignments[norm_key] = (classified_type, conf)
                    effective_type = classified_type
                elif not is_blank_frame and classified_type == "empty":
                    # Visibly empty right now: keep the stored (seeded /
                    # learned) assignment as item_type. The assignment is
                    # the slot's loadout — emergency recovery needs it to
                    # locate potion/scroll keys — while the visual truth
                    # is still reported via detected_type ("empty") and
                    # ready=False (state "depleted"), so no empty slot is
                    # ever pressed (is_slot_ready also honours the
                    # diagnostic). Wiping the assignment here made the bot
                    # forget where its potions live after one empty-frame
                    # inspection, and broke the F-1436 contract that
                    # item_type falls back to DEFAULT_SLOT_ASSIGNMENTS.
                    effective_type = snap_assignments.get(norm_key, "empty")
                else:
                    effective_type = snap_assignments.get(norm_key, classified_type)

                cd_dur = snap_cooldowns.get(norm_key, self._get_default_cd(effective_type))
                cooldown_remaining = max(0.0, cd_dur - (t - last_used)) if last_used > 0.0 else 0.0
                is_locked_out = cooldown_remaining > 0.0

                visual_darkened = bool(
                    not is_blank_frame
                    and (
                        classified_type == "empty"
                        or diag["brightness"] < 55.0
                        or diag["saturation"] < 28.0
                    )
                )

                if classified_type == "empty" and effective_type == "empty":
                    state = "empty"
                    status = "empty"
                    is_ready = False
                    is_depleted = True
                    is_on_cd = False
                elif is_locked_out:
                    state = "on_cooldown"
                    status = "on_cooldown"
                    is_ready = False
                    is_depleted = False
                    is_on_cd = True
                elif visual_darkened:
                    state = "depleted"
                    status = "depleted"
                    is_ready = False
                    is_depleted = True
                    is_on_cd = True
                elif is_blank_frame and effective_type == "empty":
                    state = "empty"
                    status = "empty"
                    is_ready = False
                    is_depleted = True
                    is_on_cd = False
                elif is_blank_frame:
                    # P0-6 / K-13: tamamen siyah/boş karede görsel kanıt
                    # yoktur — tohumlanmış (seeded) atama "ready" sayılmaz.
                    # Atama (item_type) loadout hafızası olarak korunur
                    # (F-1436: acil kurtarma iksir tuşunu bulabilsin),
                    # ama hazır durumu bilinmiyor: ready=False.
                    state = "unknown"
                    status = "unknown"
                    is_ready = False
                    is_depleted = False
                    is_on_cd = False
                else:
                    state = "ready"
                    status = "ready"
                    is_ready = True
                    is_depleted = False
                    is_on_cd = False

                buff_active = False
                buff_expired = False
                buff_remaining_sec = 0.0

                if effective_type == "buff_scroll" or classified_type == "buff_scroll":
                    applied_at = snap_buff_applied.get(norm_key)
                    expires_at = snap_buff_expires.get(norm_key)
                    if applied_at is None or expires_at is None or t >= expires_at:
                        buff_expired = True
                        buff_active = False
                        buff_remaining_sec = 0.0
                    else:
                        buff_expired = False
                        buff_active = True
                        buff_remaining_sec = round(max(0.0, expires_at - t), 2)

                reported_type = classified_type if (not is_blank_frame and classified_type != "empty") else effective_type

                slot_info = {
                    "slot": norm_key,
                    "type": reported_type,
                    "item_type": reported_type,
                    "consumable_type": reported_type,
                    "detected_type": classified_type,
                    "state": state,
                    "status": status,
                    "ready": is_ready,
                    "depleted": is_depleted,
                    "on_cooldown": is_on_cd,
                    "confidence": conf,
                    "cooldown_remaining_sec": round(cooldown_remaining, 3),
                    "buff_active": buff_active,
                    "buff_expired": buff_expired,
                    "expired": buff_expired,
                    "buff_remaining_sec": buff_remaining_sec,
                    "red_density": diag["red_density"],
                    "blue_density": diag["blue_density"],
                    "yellow_density": diag["yellow_density"],
                    "parchment_density": diag["parchment_density"],
                    "brightness": diag["brightness"],
                    "saturation": diag["saturation"],
                    "contrast_std": diag["contrast_std"],
                    "exemplar_sim": diag["exemplar_sim"],
                }
                results[norm_key] = slot_info

            # P2-3 (K-17): tek kisa kilitle geri yazim. Atama/guven
            # yalnizca snapshot'tan beri degismemisse yazilir —
            # arada learn_slot_assignment calistiysa o yazim
            # (kullanici/ogrenme karari) bayat kare goruntusuyle
            # ezilmez. _last_diagnostics last-write-wins'dir.
            with self._lock:
                for norm_key, slot_info in results.items():
                    self._last_diagnostics[norm_key] = slot_info
                for norm_key, (itype, conf) in pending_assignments.items():
                    if self._slot_assignments.get(norm_key) == snap_assignments.get(norm_key):
                        self._slot_assignments[norm_key] = itype
                        self._slot_confidences[norm_key] = conf

            return results
        except Exception as exc:
            logger.error("SmartConsumableManager.inspect_consumables failed: %s", exc)
            return {}

    def _get_default_cd(self, item_type: str) -> float:
        if item_type == "hp_potion":
            return self.default_hp_cooldown_sec
        elif item_type == "mp_potion":
            return self.default_mp_cooldown_sec
        elif item_type == "buff_scroll":
            return self.default_scroll_cooldown_sec
        return 1.0

    def get_slot_assignment(self, slot_key: str) -> Optional[str]:
        """Returns the current learned item type ('hp_potion', 'mp_potion', 'buff_scroll', 'empty') for a slot."""
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            return self._slot_assignments.get(norm_key)

    def get_slot_confidence(self, slot_key: str) -> float:
        """Returns the confidence score [0.0..1.0] for the slot assignment."""
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            return float(self._slot_confidences.get(norm_key, 0.0))

    def learn_slot_assignment(
        self,
        slot_key: str,
        item_type: str,
        crop: Optional[np.ndarray] = None,
        buff_duration_sec: Optional[float] = None,
    ) -> bool:
        norm_key = self.normalize_key(slot_key)
        clean_type = str(item_type or "").strip().lower()
        if clean_type not in ("hp_potion", "mp_potion", "buff_scroll", "empty"):
            # Invalid type: reject outright (was: stored as "unknown" and
            # reported True, polluting the exemplar store with a wrong class).
            return False

        with self._lock:
            self._slot_assignments[norm_key] = clean_type
            # Confidence reflects evidence: 0.99 only with a crop attached for
            # the exemplar store; a bare label claim gets the seed-level 0.85.
            self._slot_confidences[norm_key] = 0.99 if crop is not None else 0.85
            if buff_duration_sec is not None:
                self._buff_durations[norm_key] = max(1.0, float(buff_duration_sec))

            if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
                if extract_crop_features is not None:
                    try:
                        feats = extract_crop_features(crop)
                        if feats is not None:
                            self._exemplar_store.setdefault(clean_type, []).append(feats)
                            if len(self._exemplar_store[clean_type]) > 40:
                                self._exemplar_store[clean_type].pop(0)
                    except Exception as exc:
                        logger.debug("Failed saving internal exemplar: %s", exc)

                mem = self.get_exemplar_memory()
                if mem is not None:
                    # P1-1: paylasilan hafizaya POZITIF exemplar yalnizca
                    # kanonik tuketim siniflari icin yazilir (22/23/24).
                    # "empty"/"automatic"in kanonik sinifi YOKTUR: eski
                    # kod bu dalda (25, "empty") yaziyor, hafiza da onu
                    # (25, "kinah") diye kanoniklestirip KINAH pozitif
                    # prototipi olarak sakliyordu (Y-3 zehri). O dal
                    # artik yazilmaz — yoneticinin kendi ic deposu
                    # (yukaridaki _exemplar_store) ozellikleri aynen
                    # tutmaya devam eder.
                    cid = (
                        self.CLASS_ID_HP_POTION
                        if clean_type == "hp_potion"
                        else (
                            self.CLASS_ID_MP_POTION
                            if clean_type == "mp_potion"
                            else (
                                self.CLASS_ID_BUFF_SCROLL
                                if clean_type == "buff_scroll"
                                else None
                            )
                        )
                    )
                    if cid is None:
                        logger.debug(
                            "Empty/automatic slot onayi paylasilan hafizaya "
                            "pozitif exemplar yazmaz (kanonik sinif yok): %s",
                            clean_type,
                        )
                        return True
                    try:
                        mem.add_positive_exemplar(
                            crop,
                            class_id=cid,
                            class_name=clean_type,
                            source="consumable_manager_approval",
                        )
                    except Exception as exc:
                        logger.debug("External ExemplarMemory add_positive_exemplar: %s", exc)
        return True

    def reject_slot_assignment(
        self,
        slot_key: str,
        crop: Optional[np.ndarray] = None,
    ) -> bool:
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            self._slot_assignments[norm_key] = "empty"
            self._slot_confidences[norm_key] = 0.85

            if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
                if extract_crop_features is not None:
                    try:
                        feats = extract_crop_features(crop)
                        if feats is not None:
                            self._negative_exemplar_store.append(feats)
                            if len(self._negative_exemplar_store) > 100:
                                self._negative_exemplar_store.pop(0)
                    except Exception as exc:
                        logger.debug("Failed saving negative internal exemplar: %s", exc)

                mem = self.get_exemplar_memory()
                if mem is not None:
                    try:
                        # P1-1: reddedilen slot kirpintisi paylasilan
                        # hafizaya NEGATIF exemplar olarak yazilir; id
                        # artik kanonik-disi ad alanindadir
                        # (SLOT_STATE_EMPTY_ID = -100): "empty" bir
                        # sinif degildir, kinah (25) ile cakisamaz.
                        # Isimdeki "negative" kanonik-lestiricinin
                        # kaydi oldugu gibi birakmasini saglar; id
                        # kapisi exemplar_memory tarafinda bu ad alanina
                        # aciktir (yalnizca negatif liste).
                        mem.add_negative_exemplar(
                            crop,
                            class_id=self.SLOT_STATE_EMPTY_ID,
                            class_name="rejected_consumable_negative",
                            source="consumable_manager_rejection",
                        )
                    except Exception as exc:
                        logger.debug("External ExemplarMemory add_negative_exemplar: %s", exc)
        return True

    def mark_used(
        self,
        slot_key: str,
        now: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
    ) -> None:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            self._slot_last_used[norm_key] = t
            if cooldown_sec is not None:
                self._slot_cooldowns[norm_key] = max(0.1, float(cooldown_sec))
            itype = self._slot_assignments.get(norm_key)
            if itype == "buff_scroll":
                dur = self._buff_durations.get(norm_key, self.scroll_duration_sec)
                self._buff_applied_at[norm_key] = t
                self._buff_expires_at[norm_key] = t + dur

    def mark_buff_applied(
        self,
        slot_key: str,
        duration_sec: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            dur = (
                max(1.0, float(duration_sec))
                if duration_sec is not None
                else self._buff_durations.get(norm_key, self.scroll_duration_sec)
            )
            self._buff_durations[norm_key] = dur
            self._buff_applied_at[norm_key] = t
            self._buff_expires_at[norm_key] = t + dur
            self._slot_last_used[norm_key] = t
            self._slot_cooldowns[norm_key] = self.default_scroll_cooldown_sec

    def mark_keys_pressed(
        self,
        keys: Optional[Sequence[str]] = None,
        now: Optional[float] = None,
        *,
        buff_duration_sec: Optional[float] = None,
    ) -> List[str]:
        """Stamp cooldowns/buffs for keys that were ACTUALLY sent (P0-6).

        `decide_consumable_action` only decides; the sender calls this
        with the keys it really dispatched (and only those — a failed or
        skipped send must not be passed in, or the potion/buff would be
        suppressed without ever being used). Buff-scroll keys get
        `mark_buff_applied`, everything else gets `mark_used`. Returns the
        list of keys that were stamped.
        """
        t = time.monotonic() if now is None else float(now)
        stamped: List[str] = []
        for key in keys or []:
            if key is None:
                continue
            norm_key = self.normalize_key(str(key))
            if not norm_key:
                continue
            try:
                if self._slot_assignments.get(norm_key) == "buff_scroll":
                    self.mark_buff_applied(
                        norm_key,
                        duration_sec=buff_duration_sec if buff_duration_sec is not None else self.scroll_duration_sec,
                        now=t,
                    )
                else:
                    self.mark_used(norm_key, now=t)
                stamped.append(norm_key)
            except Exception:
                logger.debug("mark_keys_pressed failed for %s", norm_key)
        return stamped

    def is_slot_known(self, slot_key: str) -> bool:
        """Slotun bilinen bir ataması var mı (tohum/öğrenme/tarama)?

        Ataması olmayan slot hiç denetlenmemiştir — "bilinmiyor" durumudur;
        `is_slot_ready` böyle bir slotu hazır saymaz (P1-4, K-07).
        """
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            return norm_key in self._slot_assignments

    def is_slot_ready(self, slot_key: str, now: Optional[float] = None) -> bool:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            # P1-4 (K-07): hiç denetlenmemiş slot hazır sayılmaz. Ataması
            # olmayan slotun (tohumlanmamış, öğrenilmemiş, taranmamış)
            # hazır dönmesi fail-open'dı — bilinmiyor = hazır değil.
            if norm_key not in self._slot_assignments:
                return False
            itype = self._slot_assignments.get(norm_key, "unknown")
            if itype == "empty":
                return False
            last_used = self._slot_last_used.get(norm_key, 0.0)
            cd_dur = self._slot_cooldowns.get(norm_key, self._get_default_cd(itype))
            if last_used > 0.0 and (t - last_used) < cd_dur:
                return False
            diag = self._last_diagnostics.get(norm_key)
            if diag is not None and not diag.get("ready", True):
                return False
            return True

    def is_buff_expired(self, slot_key: str, now: Optional[float] = None) -> bool:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            itype = self._slot_assignments.get(norm_key)
            if itype != "buff_scroll" and norm_key not in self.SCROLL_SLOTS:
                return False
            applied = self._buff_applied_at.get(norm_key)
            expires = self._buff_expires_at.get(norm_key)
            dur = self._buff_durations.get(norm_key, self.scroll_duration_sec)
            if applied is None:
                return True
            if expires is not None:
                return t >= expires
            return (t - applied) >= dur

    def is_buff_active(self, slot_key: str, now: Optional[float] = None) -> bool:
        return not self.is_buff_expired(slot_key, now=now)

    def get_best_hp_potion(self, now: Optional[float] = None) -> Optional[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            ready_slots = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "hp_potion" and self.is_slot_ready(slot_key, now=t):
                    ready_slots.append(slot_key)
            if not ready_slots:
                return None
            ready_slots.sort(
                key=lambda s: (
                    self._slot_last_used.get(s, 0.0),
                    self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99,
                )
            )
            return ready_slots[0]

    def get_best_mp_potion(self, now: Optional[float] = None) -> Optional[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            ready_slots = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "mp_potion" and self.is_slot_ready(slot_key, now=t):
                    ready_slots.append(slot_key)
            if not ready_slots:
                return None
            ready_slots.sort(
                key=lambda s: (
                    self._slot_last_used.get(s, 0.0),
                    self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99,
                )
            )
            return ready_slots[0]

    def get_expired_buff_scrolls(self, now: Optional[float] = None) -> List[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            expired = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "buff_scroll":
                    if self.is_buff_expired(slot_key, now=t) and self.is_slot_ready(slot_key, now=t):
                        expired.append(slot_key)
            expired.sort(key=lambda s: (self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99))
            return expired

    def identify_consumable_slots(
        self,
        frame: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Inspects and classifies upper consumable slots (`F1..F8, X, B`) from `frame`.
        Returns per-slot dictionary indicating which potion/scroll is in each slot and
        whether it is `ready` (`True`) or `depleted`/`on_cooldown` (`False`).
        """
        return self.inspect_consumables(frame, now=now)

    def set_scroll_duration_sec(self, seconds: float) -> None:
        """P1-13: buff parşömen süresini canlı güncelle (route.buff_scroll_interval_sec).

        `scroll_duration_sec` + `default_buff_duration_sec` ve hâlihazırda
        buff_scroll sayılı slotların `_buff_durations` değerleri yenilenir —
        yani aralık ayarı kaydedildiği anda yürürlüğe girer. Zaten
        `mark_buff_applied` ile işaretlenmiş parşömenler de yeni süreyi
        kullanır (ayar neyse o).
        """
        try:
            dur = max(1.0, float(seconds))
        except (TypeError, ValueError):
            return
        self.scroll_duration_sec = dur
        self.default_buff_duration_sec = dur
        for slot_key, itype in self._slot_assignments.items():
            if itype == "buff_scroll":
                self._buff_durations[slot_key] = dur

    def set_preferred_scroll_key(self, slot_key: Any) -> bool:
        """P1-13: yapılandırılmış tuşu (route.buff_scroll_key) buff_scroll slotu yap.

        Parşömen slotları normalde öğrenilir/tohumlanır (F5, F7). Ayarlanan
        tuş, yalnızca slot atanmamışsa, "empty" ise veya zaten buff_scroll
        ise uygulanır — hp/mp_potion ya da automatic bir slotu asla ezmez;
        yasaklı tuşlar (F6/X/B) ve tüketim barı dışındaki tuşlar yok
        sayılır. Uygulandıysa (veya zaten uygulanmışsa) True döner.
        """
        if slot_key is None:
            return False
        key = self.normalize_key(slot_key)
        if not key:
            return False
        if key in {str(s).upper() for s in getattr(self, "forbidden_slots", set())}:
            return False
        if key not in getattr(self, "CONSUMABLE_SLOTS", ()):
            return False
        current = self._slot_assignments.get(key)
        if current is not None and current not in ("empty", "buff_scroll"):
            return False
        self._slot_assignments[key] = "buff_scroll"
        self._slot_confidences[key] = 0.85
        self._buff_durations[key] = self.scroll_duration_sec
        self._slot_cooldowns[key] = self.default_scroll_cooldown_sec
        return True

    def check_and_refresh_scroll_buffs(
        self,
        frame: Optional[Any] = None,
        now: Optional[float] = None,
        *,
        active_buffs: Optional[Any] = None,
    ) -> List[str]:
        """
        Checks `buff_scroll` slots and returns the list of keys to press
        automatically when their duration (`scroll_duration_sec=300.0`) has expired.
        Supports both `check_and_refresh_scroll_buffs(frame, now=None)` and
        `check_and_refresh_scroll_buffs(active_buffs=..., now=None)`.
        Excludes automatic client slots (`F6, X, B` when enabled).
        """
        actual_frame: Optional[np.ndarray] = None
        if frame is not None:
            if isinstance(frame, np.ndarray):
                actual_frame = frame
            elif isinstance(frame, (int, float)) and now is None:
                now = float(frame)
            elif active_buffs is None and isinstance(frame, (dict, list, set, tuple)):
                active_buffs = frame

        t = time.monotonic() if now is None else float(now)

        if actual_frame is not None and actual_frame.size > 0 and int(np.max(actual_frame)) > 0:
            self.identify_consumable_slots(actual_frame, now=t)

        keys_to_refresh: List[str] = []
        active_keys_set = set()
        if active_buffs is not None:
            if isinstance(active_buffs, dict):
                active_keys_set = {str(k).upper() for k, v in active_buffs.items() if v}
            else:
                active_keys_set = {str(k).upper() for k in active_buffs}

        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            scroll_candidates = [
                slot_key
                for slot_key, itype in self._slot_assignments.items()
                if itype == "buff_scroll"
                and slot_key.upper() not in forbidden
            ]
            if not scroll_candidates:
                scroll_candidates = [
                    s for s in self.SCROLL_SLOTS
                    if s.upper() not in forbidden
                ]

            for slot_key in scroll_candidates:
                if not self.is_slot_ready(slot_key, now=t):
                    continue
                if active_buffs is not None:
                    is_active = (slot_key in active_keys_set) or (slot_key.lower() in active_keys_set)
                    if not is_active:
                        keys_to_refresh.append(slot_key)
                else:
                    if self.is_buff_expired(slot_key, now=t):
                        keys_to_refresh.append(slot_key)
            # NOTE: this is a query — it deliberately does NOT mark the buff
            # applied. Marking here suppressed the scroll for 300s even when the
            # key was never pressed. Callers mark after a real press via
            # mark_keys_pressed() (trigger_emergency_recovery / apply_to_bus).

            keys_to_refresh.sort(key=lambda s: (self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99))
            return keys_to_refresh

    def decide_consumable_action(
        self,
        frame_or_hp: Any = None,
        hp_pct: Optional[float] = None,
        mp_pct: Optional[float] = None,
        now: Optional[float] = None,
        *,
        frame: Optional[np.ndarray] = None,
        player_hp_pct: Optional[float] = None,
        player_mp_pct: Optional[float] = None,
        active_buffs: Optional[Any] = None,
    ) -> ConsumableActionResult:
        """
        Decides consumable action from `frame`, `hp_pct`, `mp_pct`, and `now`:
        - If `hp_pct` is below the HP potion threshold (F-0540 tek kaynak,
          `HP_POTION_THRESHOLD_PCT` = 65): selects a ready `hp_potion` slot key.
        - If `mp_pct` is below the MP potion threshold (F-0540 tek kaynak,
          `MP_POTION_THRESHOLD_PCT` = 45): selects a ready `mp_potion` slot key.
        - If a `buff_scroll` (`F5, F7`; F6 is automatic/forbidden) has expired (`scroll_duration_sec=300.0`): selects the ready `buff_scroll` key.
        Supports both `(frame, hp_pct, mp_pct, now=None)` and `(player_hp_pct, player_mp_pct, active_buffs=None, now=None)`.

        Pure decision: this method never stamps a cooldown or a buff
        duration. A chosen key counts as used only after the caller
        actually sends it — see `mark_keys_pressed(keys, now=t)`, which
        the sender must call with the keys it really dispatched (P0-6 /
        K-13: stamping here suppressed the potion/buff for 300s even when
        the key was never pressed).
        """
        actual_frame: Optional[np.ndarray] = frame
        if isinstance(frame_or_hp, np.ndarray):
            actual_frame = frame_or_hp
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)
        elif frame_or_hp is None:
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)
        elif isinstance(frame_or_hp, (int, float)):
            # Numeric first positional is hp (0..1 ratio or 0..100 percent).
            # Remaining positionals are (mp, now): the mp_pct slot holds `now`
            # when a third positional is present, and a bare mp_pct keyword
            # always means MP (it used to be ignored, reading mp as 1.0).
            eff_hp = float(frame_or_hp)
            mp_value: Optional[float] = None
            if isinstance(hp_pct, (int, float)):
                if mp_pct is None and now is None and float(hp_pct) > 100.0:
                    now = float(hp_pct)  # (hp, now) form: 2nd positional is a timestamp
                else:
                    mp_value = float(hp_pct)
            if isinstance(mp_pct, (dict, list, set, tuple)):
                active_buffs = mp_pct
            elif isinstance(mp_pct, (int, float)):
                if now is None and mp_value is not None and float(mp_pct) > 100.0:
                    now = float(mp_pct)  # (hp, mp, now) form
                elif mp_value is None:
                    mp_value = float(mp_pct)  # mp_pct passed as keyword
            eff_mp = mp_value if mp_value is not None else (
                player_mp_pct if player_mp_pct is not None else 1.0
            )
        else:
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)

        t = time.monotonic() if now is None else float(now)

        # Inspect whenever a frame is given — including an all-black one:
        # inspect_consumables records ready=False for blank frames (P0-6 /
        # K-13), so selection below can never treat a seeded assignment as
        # ready on zero visual evidence. (Was: black frames skipped the
        # inspection, leaving the seeded slot "ready".)
        if actual_frame is not None and isinstance(actual_frame, np.ndarray) and actual_frame.size > 0:
            self.identify_consumable_slots(actual_frame, now=t)

        hp = float(eff_hp) if float(eff_hp) <= 1.0 else float(eff_hp) / 100.0
        mp = float(eff_mp) if float(eff_mp) <= 1.0 else float(eff_mp) / 100.0
        hp = max(0.0, min(1.0, hp))
        mp = max(0.0, min(1.0, mp))

        is_emergency = hp < 0.40
        hp_needed = hp < self.hp_threshold
        mp_needed = mp < self.mp_threshold

        chosen_hp: Optional[str] = None
        chosen_mp: Optional[str] = None
        reasons: List[str] = []

        if hp_needed:
            hp_cand = self.get_best_hp_potion(now=t)
            if hp_cand and self.is_slot_ready(hp_cand, now=t):
                chosen_hp = hp_cand
                # P0-6: no mark_used here — this is only a decision. The
                # caller stamps via mark_keys_pressed() after the key is
                # really sent; otherwise a failed/never-sent press would
                # still start the potion cooldown.
                severity = "EMERGENCY_HP" if is_emergency else "HP_POTION"
                reasons.append(f"{severity}: {hp*100:.1f}% < {self.hp_threshold*100:.0f}% -> [{chosen_hp}]")

        if mp_needed:
            mp_cand = self.get_best_mp_potion(now=t)
            if mp_cand and self.is_slot_ready(mp_cand, now=t):
                chosen_mp = mp_cand
                # P0-6: no mark_used here (see HP branch above).
                reasons.append(f"MP_POTION: {mp*100:.1f}% < {self.mp_threshold*100:.0f}% -> [{chosen_mp}]")

        scroll_keys = self.check_and_refresh_scroll_buffs(
            frame=None,
            now=t,
            active_buffs=active_buffs,
        )
        if scroll_keys:
            reasons.append(f"BUFF_REFRESH -> {scroll_keys}")
            # P0-6: no mark_buff_applied here — a 300s buff must not be
            # recorded as applied before the scroll key is really sent.
            # The sender stamps via mark_keys_pressed() after dispatch.

        chosen_scroll: Optional[str] = scroll_keys[0] if scroll_keys else None

        forbidden = getattr(self, "forbidden_slots", set())
        keys_to_press: List[str] = []
        if chosen_hp and str(chosen_hp).upper() not in forbidden:
            keys_to_press.append(chosen_hp)
        if chosen_mp and str(chosen_mp).upper() not in forbidden:
            keys_to_press.append(chosen_mp)
        if scroll_keys:
            keys_to_press.extend([
                s for s in scroll_keys
                if str(s).upper() not in forbidden
            ])

        primary_key: Optional[str] = keys_to_press[0] if keys_to_press else None
        action_name = "emergency" if is_emergency else ("consumable_buff" if keys_to_press else "none")

        return ConsumableActionResult(
            {
                "action": action_name,
                "key": primary_key,
                "recommended_key": primary_key,
                "emergency": is_emergency,
                "hp_potion": chosen_hp,
                "mp_potion": chosen_mp,
                "buff_scroll": chosen_scroll,
                "hp_potion_key": chosen_hp,
                "mp_potion_key": chosen_mp,
                "scroll_key": chosen_scroll,
                "scroll_buffs": scroll_keys,
                "buff_keys": scroll_keys,
                "keys_to_press": keys_to_press,
                "player_hp": hp,
                "player_mp": mp,
                "hp_pct": hp,
                "mp_pct": mp,
                "reason": "; ".join(reasons) if reasons else "vitals_healthy",
            }
        )

    def get_slot_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._last_diagnostics)

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            mem = self.get_exemplar_memory()
            return {
                "module": "smart_consumable_manager",
                "current_roi": self.current_consumable_roi,
                "slots_count": len(self.CONSUMABLE_SLOTS),
                "learned_assignments": dict(self._slot_assignments),
                "hp_potions": [s for s, t in self._slot_assignments.items() if t == "hp_potion"],
                "mp_potions": [s for s, t in self._slot_assignments.items() if t == "mp_potion"],
                "buff_scrolls": [s for s, t in self._slot_assignments.items() if t == "buff_scroll"],
                "empty_slots": [s for s, t in self._slot_assignments.items() if t == "empty"],
                "exemplar_store_counts": {k: len(v) for k, v in self._exemplar_store.items()},
                "negative_exemplar_count": len(self._negative_exemplar_store),
                "external_exemplar_memory": mem is not None,
            }


def classify_consumable_slot_crop(
    crop_bgr: Optional[np.ndarray],
    exemplar_memory: Optional[Any] = None,
) -> str:
    """Module-level convenience wrapper for `SmartConsumableManager.classify_consumable_slot_crop`."""
    mgr = SmartConsumableManager(exemplar_memory=exemplar_memory, seed_defaults=False)
    return mgr.classify_consumable_slot_crop(crop_bgr)


__all__ = [
    "CONSUMABLE_SLOTS",
    "SkillBarReader",
    "SmartConsumableManager",
    "ConsumableActionResult",
    "MicroCameraLookAround",
    "AdaptiveSkillBarReader",
    "classify_consumable_slot_crop",
    "get_skill_bar_reader_capabilities",
    "validate_skill_bar_reader_contract",
]


