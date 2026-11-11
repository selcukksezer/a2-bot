# -*- coding: utf-8 -*-
"""
tests/test_v20_yolo_onay_onizleme.py
====================================
v20 onay-kuyruğu önizleme düzeltmesi (Selçuk, 11 Kasım 17:37):
kuyruktaki #learn_monsters_... karesinin küçük resmi "…" kalıyor,
ana panel "Canlı kare yok" gösteriyordu — kare kaydı var, görüntüsü yok.

Kök neden (kod kanıtı):
- core/omni_game_learner._save_learning_sample kareyi hem
  dataset/game_learning/<kategori>/ hem dataset/images/train/ altına
  yazar; commit 0 kutu yazınca train kopyası geri alınır (silinir)
  ama review JSON + game_learning kopyası kuyruk için kalır.
- ui/yolo_factory_dialog._resolve_review_image_path yalnızca
  images/{split,train,val}/ + previews/ arıyordu — game_learning/
  hiç aranmıyordu → None → küçük resim "…" + ana panel boş.

Testler KOŞULMADI — ana ajan koşacak.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ui.yolo_factory_dialog import (
    YOLOPreviewCanvas,
    _decode_thumb_qimage,
    _resolve_review_image_path,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _write_jpg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((90, 160, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), img), f"jpg yazılamadı: {path}"
    return path


STEM_MON = "learn_monsters_1794405235376146500"
STEM_SKL = "learn_skills_1794407256036701500_267"
STEM_MIX = "mix_1794407000000000000_123"


def test_resolve_learn_monsters_from_game_learning(tmp_path: Path) -> None:
    """train kopyası silinmiş learn_monsters karesi game_learning'den bulunur."""
    expected = _write_jpg(tmp_path / "game_learning" / "monsters" / f"{STEM_MON}.jpg")
    assert _resolve_review_image_path(tmp_path, "train", STEM_MON) == expected


def test_resolve_learn_skills_from_game_learning(tmp_path: Path) -> None:
    """Aynı durum skills kategorisi için de geçerlidir."""
    expected = _write_jpg(tmp_path / "game_learning" / "skills" / f"{STEM_SKL}.jpg")
    assert _resolve_review_image_path(tmp_path, "train", STEM_SKL) == expected


def test_resolve_mix_from_game_learning_subdir_scan(tmp_path: Path) -> None:
    """Stem kategori taşımıyorsa (mix_*) alt klasör taraması bulur."""
    expected = _write_jpg(tmp_path / "game_learning" / "mix" / f"{STEM_MIX}.jpg")
    assert _resolve_review_image_path(tmp_path, "train", STEM_MIX) == expected


def test_images_train_wins_over_game_learning(tmp_path: Path) -> None:
    """images/train kopyası varsa öncelik onundur (eski davranış korunur)."""
    train_path = _write_jpg(tmp_path / "images" / "train" / f"{STEM_MON}.jpg")
    _write_jpg(tmp_path / "game_learning" / "monsters" / f"{STEM_MON}.jpg")
    assert _resolve_review_image_path(tmp_path, "train", STEM_MON) == train_path


def test_resolve_missing_everywhere_returns_none(tmp_path: Path) -> None:
    """Hiçbir yerde yoksa None — sessizce uydurma yol dönülmez."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "game_learning" / "monsters").mkdir(parents=True)
    assert _resolve_review_image_path(tmp_path, "train", STEM_MON) is None


def test_decode_thumb_uses_game_learning(tmp_path: Path, qapp: QApplication) -> None:
    """Kuyruk küçük resmi ("…" takılması) game_learning karesinden üretilir."""
    _write_jpg(tmp_path / "game_learning" / "monsters" / f"{STEM_MON}.jpg")
    qimg = _decode_thumb_qimage(tmp_path, "train", STEM_MON)
    assert qimg is not None
    assert not qimg.isNull()
    assert qimg.width() <= 72 and qimg.height() <= 54


def test_decode_thumb_missing_returns_none(tmp_path: Path, qapp: QApplication) -> None:
    """Görüntü yoksa küçük resim None döner (yer tutucu "…" kalır)."""
    assert _decode_thumb_qimage(tmp_path, "train", STEM_MON) is None


def test_empty_state_keeps_real_message(qapp: QApplication) -> None:
    """Boş durum gerçek nedeni taşır — ortadaki yazı artık sabit
    'Canlı kare yok' değil, show_empty_state mesajıdır."""
    canvas = YOLOPreviewCanvas()
    try:
        canvas.show_empty_state(f"Görüntü bulunamadı: {STEM_MON}")
        assert canvas._empty is True
        assert canvas._pixmap is None
        assert canvas._live_badge == f"Görüntü bulunamadı: {STEM_MON}"
        assert canvas._preview_path == ""
        canvas.update()
    finally:
        canvas.deleteLater()
