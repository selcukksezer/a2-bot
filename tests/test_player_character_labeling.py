# -*- coding: utf-8 -*-
"""tests/test_player_character_labeling.py
========================================
Tests for Player Character (26: player_character / Karakterim) labeling & recognition:
1. Canonical registration (ID 26, name 'player_character', Turkish label 'Karakterim').
2. Alias resolution ('player', 'character', 'karakter', 'karakterim', 'oyuncu' -> 26).
3. '🎯 Karakterimi Etiketle' (btn_add_player_char) and shortcut 'K' placing center bounding box.
4. YOLO disk persistence and validation: formats standard `26 cx cy w h` in label file.
5. data.yaml nc: 27 with 26: player_character.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    CANONICAL_NAMES,
    CANONICAL_NAME_TO_ID,
    get_super_category,
    is_player_character,
    is_valid_canonical_class_id,
    sanitize_class_id_and_name,
)
from ui.yolo_factory_dialog import CLASS_ID_TO_SPEC, YOLOFactoryDialog


def test_player_character_taxonomy_registration() -> None:
    """Verify player character is canonical class ID 26 with correct naming and super-category."""
    assert 26 in CANONICAL_NAMES
    assert CANONICAL_NAMES[26] == "player_character"
    assert is_valid_canonical_class_id(26) is True
    assert is_player_character(26) is True
    assert get_super_category(26) == "player"

    # Verify Turkish label in dialog specs
    assert 26 in CLASS_ID_TO_SPEC
    spec = CLASS_ID_TO_SPEC[26]
    assert spec["key"] == "player_character"
    assert "karakter" in spec["label_tr"].lower()


def test_player_character_alias_resolution() -> None:
    """Verify all colloquial aliases for player character resolve to 26."""
    aliases = ["player", "character", "karakter", "karakterim", "oyuncu", "player_3d", "avatar"]
    for alias in aliases:
        cid, cname = sanitize_class_id_and_name(-1, alias)
        assert cid == 26, f"Alias '{alias}' resolved to {cid} instead of 26"
        assert cname == "player_character"


def test_add_player_character_box_to_frame(tmp_path: Path) -> None:
    """Verify clicking 'Karakterimi Etiketle' or pressing 'K' adds a centered box to the frame and writes to disk."""
    dataset_dir = tmp_path / "dataset"
    labels_dir = dataset_dir / "labels" / "train"
    review_dir = dataset_dir / "review" / "train"
    images_dir = dataset_dir / "images" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    stem = "frame_with_player"
    (images_dir / f"{stem}.jpg").write_bytes(b"\xFF\xD8\xFF\xE0MOCK")
    (labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
    (review_dir / f"{stem}.json").write_text(
        json.dumps({
            "stem": stem,
            "split": "train",
            "trusted": [],
            "weak": [],
            "done": None,
        }),
        encoding="utf-8",
    )

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dlg._models_dir = lambda: models_dir

    dlg._review_items = [
        {
            "stem": stem,
            "split": "train",
            "path": review_dir / f"{stem}.json",
            "trusted": [],
            "weak": [],
            "done": None,
        }
    ]
    dlg._review_index = 0

    # User clicks 'Karakterimi Etiketle'
    dlg._add_player_character_box_to_current_frame()

    # Verify box was added in memory
    trusted = dlg._review_items[0]["trusted"]
    assert len(trusted) == 1
    player_box = trusted[0]
    assert player_box["class_id"] == 26
    assert player_box["class_name"] == "player_character"
    assert player_box["cx"] == pytest.approx(0.50, abs=1e-2)
    assert player_box["cy"] == pytest.approx(0.62, abs=1e-2)
    assert player_box["trusted"] is True
    assert player_box["human_approved"] is True

    # Verify persisted to label txt
    lbl_content = (labels_dir / f"{stem}.txt").read_text(encoding="utf-8").strip()
    assert lbl_content.startswith("26 0.500000 0.620000")


def test_data_yaml_has_player_character() -> None:
    """Verify dataset/data.yaml contains nc: 27 and 26: player_character."""
    yaml_path = PROJECT_ROOT / "dataset" / "data.yaml"
    assert yaml_path.is_file()
    content = yaml_path.read_text(encoding="utf-8")
    assert "nc: 27" in content
    assert "26: player_character" in content
