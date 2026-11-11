# -*- coding: utf-8 -*-
"""Tek "ayni kutu" tanimi (P1-13 / K-33).

F073'te sayilan 8 farkli tanim (0.02 mutlak / IoU >= 0.30 / IoU > 0.65 /
IoU > 0.80 / 1e-6 / tam sozluk esitligi / IoU >= 0.45+merkez mesafesi /
IoU >= 0.25-0.20) ayni kutu sorusuna iki yonde zit cevap veriyordu:
0.02x0.02 kutu 0.02 kaymisken identity=EVET (IoU=0.000), 0.50x0.30 kutu
0.05 kaymisken identity=HAYIR (IoU=0.818). Artik tek tanim var:
``same_box()`` — ayni sinif + IoU >= SAME_BOX_IOU + merkez mesafesi
<= SAME_BOX_MAX_CENTER_DIST. Esik, daha onceki 8 tanimdan ilkesel
olani (IoU >= 0.45 + merkez mesafesi) kanoniklestirir.

Eksik anahtarli kutu eslesmeden once elenir: ``cx``/``cy``/``w``/``h``
dordunden biri eksik, sayi degil, sonlu degil ya da ``w/h <= 0`` ise
``same_box()`` False doner — ``w/h``'si olmayan ``det`` artik listedeki
ilk ayni-sinif kutuyla eslesip yanlis kutunun silinmesine yol acamaz
(F073-F2; ``_reject_box`` eslesme yoksa hicbir sey silmez).

Bu modul yalnizca stdlib kullanir: ``dataset_validator``,
``label_trust`` ve ``ui/yolo_factory_dialog.py`` dongusel import
riski olmadan import edebilir.

Esik tablosu — kimlik tek, bastirma esikleri ayri kavramdir:
    =================  ========  ============================================
    Kavram             Esik      Nerede
    =================  ========  ============================================
    same_box (kimlik)  IoU>=0.45 ``dataset_factory/box_match.py::same_box``
                       + merkez   dialog ``_box_identity_match``,
                       <=0.05     ``label_trust._find_slot``,
                                  ``dataset_validator`` weak/trusted
                                  cakisma temizligi, drag ``_on_box_edited``
    SLOT_IOU (eski)    0.30      KALDIRILDI — ``label_trust.SLOT_IOU``
                                  geriye donuk ad olarak SAME_BOX_IOU'ya
                                  esit; %30 ortusme ayni slot sayiliyordu,
                                  yan yana iki nesne tek slot sayilip yeni
                                  gozlem dusebiliyordu (F070)
    NMS bastirma       IoU>0.65  ``label_trust._nms_dedupe`` — bastirma,
                                  kimlik degil; esik aynen kalir
    dedup / overlap    IoU>0.80  ``dataset_validator`` (``sanitize_boxes``,
                                  ``detect_overlapping_boxes``,
                                  ``DEFAULT_IOU_OVERLAP_THRESHOLD``) —
                                  bastirma, kimlik degil; esik aynen kalir
    dialog 0.02 mutlak KALDIRILDI ``_box_identity_match`` eski tanimi
    dialog 1e-4        KALDIRILDI drag ``is_edited`` eski cx/cy-only tanimi
    (yalnizca cx/cy)
    validator 1e-6     KALDIRILDI ``_same_box`` eski dordlu tanimi
    =================  ========  ============================================
Kapsam disi (degismedi, ayri kavram): ``yolo_auto_labeler`` 0.25/0.20
ve ``monster_diamond_detector`` 0.45 dedup esikleri — otomatik
etiketleme sirasinda aday eleme, kutu kimligi degil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Tek "ayni kutu" toleransi (P1-13 / K-33).
SAME_BOX_IOU: float = 0.45
SAME_BOX_MAX_CENTER_DIST: float = 0.05


@dataclass(frozen=True)
class BoxGeometry:
    """Parse edilmis kutu: sinif + normalize (cx, cy, w, h)."""

    class_id: Optional[int]
    cx: float
    cy: float
    w: float
    h: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (
            self.cx - self.w * 0.5,
            self.cy - self.h * 0.5,
            self.cx + self.w * 0.5,
            self.cy + self.h * 0.5,
        )


def _safe_class_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))  # e.g. "3.0"
        except (TypeError, ValueError):
            return None


def _coord(box: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in box:
            try:
                value = float(box[key])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def parse_box(box: Any) -> Optional[BoxGeometry]:
    """Kutuyu parse et; geometri eksik/gecersizse None don.

    ``cx``/``cy``/``w``/``h`` dordunun de var ve sonlu olmasi, ``w/h > 0``
    olmasi gerekir (``w`` yerine ``width``, ``h`` yerine ``height`` de
    kabul edilir — ``label_trust._box_xyxy`` ile ayni kural). ``class_id``
    yoksa/parse edilemezse ``None`` olur ama geometri gecerli sayilir;
    kimlik karsilastirmasi (``same_box``) sinifi ayri ister.
    """
    if not isinstance(box, dict):
        return None
    cx = _coord(box, "cx")
    cy = _coord(box, "cy")
    w = _coord(box, "w", "width")
    h = _coord(box, "h", "height")
    if cx is None or cy is None or w is None or h is None:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return BoxGeometry(
        class_id=_safe_class_id(box.get("class_id")),
        cx=cx,
        cy=cy,
        w=w,
        h=h,
    )


def has_box_geometry(box: Any) -> bool:
    """Kutu eslesmeye girecek kadar tam geometri tasiyor mu?"""
    return parse_box(box) is not None


def iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU — ``label_trust._iou_xyxy`` ile bit-for-bit ayni aritmetik
    (ayni islem sirasi), tek kopya burada yasar."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def box_iou(a: Any, b: Any) -> float:
    """Iki kutu dict'i arasinda IoU; geometrisi eksikse 0.0."""
    pa, pb = parse_box(a), parse_box(b)
    if pa is None or pb is None:
        return 0.0
    return iou_xyxy(pa.xyxy, pb.xyxy)


def _iou_parsed(a: BoxGeometry, b: BoxGeometry) -> float:
    return iou_xyxy(a.xyxy, b.xyxy)


def center_distance(a: Any, b: Any) -> float:
    """Normalize merkezler arasi Oklid mesafesi; geometri eksikse inf."""
    pa, pb = parse_box(a), parse_box(b)
    if pa is None or pb is None:
        return math.inf
    return math.hypot(pa.cx - pb.cx, pa.cy - pb.cy)


def same_box_parsed(a: BoxGeometry, b: BoxGeometry) -> bool:
    """Tek kimlik tanimi, parse edilmis formda: ayni sinif + IoU +
    merkez mesafesi. Sinifi eksik kutu eslesmez (fail-closed)."""
    if a.class_id is None or b.class_id is None:
        return False
    if a.class_id != b.class_id:
        return False
    if _iou_parsed(a, b) < SAME_BOX_IOU:
        return False
    if math.hypot(a.cx - b.cx, a.cy - b.cy) > SAME_BOX_MAX_CENTER_DIST:
        return False
    return True


def same_box(a: Any, b: Any) -> bool:
    """Tek "ayni kutu" tanimi (P1-13 / K-33).

    Eksik anahtarli (``cx``/``cy``/``w``/``h``/``class_id``) kutu False
    doner — eslesmeden once elenir, listedeki ilk ayni-sinif kutuyla
    eslesemez.
    """
    pa, pb = parse_box(a), parse_box(b)
    if pa is None or pb is None:
        return False
    return same_box_parsed(pa, pb)


def find_same_box(
    boxes: Sequence[Dict[str, Any]], candidate: Dict[str, Any]
) -> Optional[int]:
    """``candidate`` ile ayni kutu olan ilk girdinin indisi, yoksa None."""
    cand = parse_box(candidate)
    if cand is None:
        return None
    for i, box in enumerate(boxes):
        parsed = parse_box(box)
        if parsed is not None and same_box_parsed(parsed, cand):
            return i
    return None


__all__: List[str] = [
    "SAME_BOX_IOU",
    "SAME_BOX_MAX_CENTER_DIST",
    "BoxGeometry",
    "parse_box",
    "has_box_geometry",
    "iou_xyxy",
    "box_iou",
    "center_distance",
    "same_box_parsed",
    "same_box",
    "find_same_box",
]
