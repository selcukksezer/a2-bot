# -*- coding: utf-8 -*-
"""Unit tests for W47: 4-Layer AI Poison Label Cleaner (`clean_poison_labels_ai`).

Verifies:
- Layer 0: Immunity Shield (hand_label, human_approved, F/E gather seed protection)
- Layer 1: Exemplar Class Repair & Negative Exemplar Suppression
- Layer 2: Flat Ground / Featureless Texture & Minimap HUD Poison Deletion
- Layer 3: NMS Deduplication, Omni Synthetic Demotion & Label TXT Synchronization
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_factory.label_trust import clean_poison_labels_ai


class _DummyExemplarMemory:
    """Lightweight mock ExemplarMemory for tracking negative teaching and exemplar matching."""

    def __init__(self) -> None:
        self.negatives: List[Dict[str, Any]] = []
        self.positive_match_result: Tuple[float, int, str, Dict[str, Any]] = (0.0, -1, "", {})
        self.negative_match_result: Tuple[float, Dict[str, Any]] = (0.0, {})

    def add_negative_exemplar(
        self,
        crop: np.ndarray,
        class_id: int,
        class_name: str = "",
        source: str = "",
    ) -> bool:
        self.negatives.append(
            {
                "shape": tuple(crop.shape) if crop is not None else (),
                "class_id": int(class_id),
                "class_name": str(class_name),
                "source": str(source),
            }
        )
        return True

    def match_positive(self, crop: np.ndarray) -> Tuple[float, int, str, Dict[str, Any]]:
        return self.positive_match_result

    def match_negative(self, crop: np.ndarray, class_id: int = -1) -> Tuple[float, Dict[str, Any]]:
        return self.negative_match_result


def _write_review_file(
    dataset_dir: Path,
    split: str,
    stem: str,
    trusted: List[Dict[str, Any]],
    weak: List[Dict[str, Any]],
) -> Path:
    review_dir = dataset_dir / "review" / split
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / f"{stem}.json"
    payload = {
        "stem": stem,
        "split": split,
        "trusted": trusted,
        "weak": weak,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _make_textured_frame(height: int = 200, width: int = 200) -> np.ndarray:
    """Create a deterministic high-variance checkerboard/noise frame so Laplacian variance >> 10."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def test_layer0_immunity_shield_protects_hand_and_approved_boxes(tmp_path: Path) -> None:
    """Katman 0: Dokunulmazlık kalkanı (`hand_label`, `human_approved`, F/E şablon) kutuları
    düz zeminde veya HUD bölgesinde olsa bile asla silmez ve `trusted=True` olarak korur.
    """
    stem = "scene_immune_01"
    # Flat zero-variance frame that would normally trigger Layer 2 flat texture deletion
    flat_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    hand_box = {
        "class_id": 0,
        "class_name": "herb_aria",
        "cx": 0.90,  # Inside minimap HUD zone, yet immune due to hand_label
        "cy": 0.15,
        "w": 0.10,
        "h": 0.10,
        "confidence": 0.55,
        "hand_label": True,
        "source": "manual_draw",
    }
    approved_box = {
        "class_id": 3,
        "class_name": "mine_orichalcum",
        "cx": 0.40,
        "cy": 0.50,
        "w": 0.12,
        "h": 0.12,
        "confidence": 0.60,
        "human_approved": True,
        "source": "yolo_detector",
    }
    seed_box = {
        "class_id": 7,
        "class_name": "gather_prompt_f",
        "cx": 0.55,
        "cy": 0.55,
        "w": 0.08,
        "h": 0.08,
        "confidence": 0.50,
        "gather_key": "f",
        "source": "f_gather_template",
    }

    review_path = _write_review_file(
        tmp_path,
        "train",
        stem,
        trusted=[hand_box],
        weak=[approved_box, seed_box],
    )
    mem = _DummyExemplarMemory()

    stats = clean_poison_labels_ai(
        tmp_path,
        exemplar_memory=mem,
        frame_map={stem: flat_frame},
    )

    assert stats["ok"] is True
    assert stats["protected_boxes"] == 3
    assert stats["poison_deleted_boxes"] == 0

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 3
    assert len(saved["weak"]) == 0
    for box in saved["trusted"]:
        assert box["trusted"] is True
        assert box["weak"] is False
        assert box["confidence"] >= 0.90

    label_txt = (tmp_path / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(label_txt) == 3


def test_layer1_exemplar_class_repair_and_positive_protection(tmp_path: Path) -> None:
    """Katman 1: Pozitif exemplar eşleşmesi >= 0.82 olan yanlış sınıflandırılmış bir kutunun
    `class_id` ve `class_name` alanlarını otomatik onarır (`repaired_by_exemplar=True`).
    """
    stem = "scene_repair_01"
    textured_frame = _make_textured_frame()

    misclassified_box = {
        "class_id": 1,
        "class_name": "herb_mela",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.20,
        "h": 0.20,
        "confidence": 0.62,
        "source": "yolo_detector",
        "exemplar_repair_sim": 0.91,
        "exemplar_repair_class_id": 0,
        "exemplar_repair_class_name": "herb_aria",
    }

    review_path = _write_review_file(
        tmp_path,
        "train",
        stem,
        trusted=[],
        weak=[misclassified_box],
    )
    mem = _DummyExemplarMemory()

    stats = clean_poison_labels_ai(
        tmp_path,
        exemplar_memory=mem,
        frame_map={stem: textured_frame},
        emin_threshold=0.85,
    )

    assert stats["auto_repaired_boxes"] == 1
    assert stats["protected_boxes"] == 1
    assert stats["poison_deleted_boxes"] == 0

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 1
    repaired = saved["trusted"][0]
    assert repaired["class_id"] == 0
    assert repaired["class_name"] == "herb_aria"
    assert repaired["repaired_by_exemplar"] is True
    assert repaired["confidence"] == pytest.approx(0.91, abs=1e-4)

    label_lines = (tmp_path / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")


def test_layer1_negative_exemplar_suppression(tmp_path: Path) -> None:
    """Katman 1: Negatif exemplar benzerliği >= 0.82 ve pozitif farkı >= 0.15 olan zehirli kutuları
    siler (`negative_match_deleted`), pozitif exemplar eşleşen kutuyu korur.
    """
    stem = "scene_neg_match_01"
    textured_frame = _make_textured_frame()

    poison_neg_box = {
        "class_id": 2,
        "class_name": "herb_targena",
        "cx": 0.30,
        "cy": 0.40,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.75,
        "source": "yolo_detector",
        "exemplar_neg_sim": 0.89,
        "exemplar_pos_sim": 0.30,
    }
    valid_pos_box = {
        "class_id": 2,
        "class_name": "herb_targena",
        "cx": 0.65,
        "cy": 0.65,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.88,
        "source": "yolo_detector",
        "exemplar_pos_sim": 0.86,
        "exemplar_neg_sim": 0.10,
    }

    review_path = _write_review_file(
        tmp_path,
        "train",
        stem,
        trusted=[],
        weak=[poison_neg_box, valid_pos_box],
    )
    mem = _DummyExemplarMemory()

    stats = clean_poison_labels_ai(
        tmp_path,
        exemplar_memory=mem,
        frame_map={stem: textured_frame},
    )

    assert stats["negative_match_deleted"] == 1
    assert stats["poison_deleted_boxes"] == 1
    assert stats["protected_boxes"] == 1

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 1
    assert len(saved["weak"]) == 0
    assert saved["trusted"][0]["cx"] == pytest.approx(0.65)


def test_layer2_flat_ground_and_hud_poison_deletion_with_negative_teaching(tmp_path: Path) -> None:
    """Katman 2: Minimap HUD üzerindeki yasadışı dünya nesnesi kutularını (`hud_poison_deleted`)
    ve dokusuz düz zemin/gökyüzü kutularını (`flat_texture_deleted`, Laplacian var < 10.0) siler
    ve negatif exemplar hafızasına öğretir (`negative_taught_boxes`).
    """
    stem = "scene_layer2_poison"
    # Top-left and whole frame are flat (zeros), except bottom-right where a valid textured object sits
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    frame[120:190, 120:190] = rng.integers(0, 255, size=(70, 70, 3), dtype=np.uint8)

    hud_poison_box = {
        "class_id": 0,  # herb_aria is illegal in minimap HUD
        "class_name": "herb_aria",
        "cx": 0.90,
        "cy": 0.15,
        "w": 0.10,
        "h": 0.10,
        "confidence": 0.80,
        "source": "yolo_detector",
    }
    flat_ground_box = {
        "class_id": 4,  # mine_crystal on completely flat region (top-left)
        "class_name": "mine_crystal",
        "cx": 0.25,
        "cy": 0.25,
        "w": 0.20,
        "h": 0.20,
        "confidence": 0.78,
        "source": "yolo_detector",
    }
    valid_textured_box = {
        "class_id": 4,
        "class_name": "mine_crystal",
        "cx": 0.77,
        "cy": 0.77,
        "w": 0.20,
        "h": 0.20,
        "confidence": 0.90,
        "source": "yolo_detector",
    }

    review_path = _write_review_file(
        tmp_path,
        "train",
        stem,
        trusted=[],
        weak=[hud_poison_box, flat_ground_box, valid_textured_box],
    )
    mem = _DummyExemplarMemory()

    stats = clean_poison_labels_ai(
        tmp_path,
        exemplar_memory=mem,
        frame_map={stem: frame},
        teach_negatives=True,
    )

    assert stats["hud_poison_deleted"] == 1
    assert stats["flat_texture_deleted"] == 1
    assert stats["poison_deleted_boxes"] == 2
    assert stats["negative_taught_boxes"] == 2
    assert len(mem.negatives) == 2
    sources = {item["source"] for item in mem.negatives}
    assert sources == {"poison_hud_illegal", "poison_flat_texture"}

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 1
    assert len(saved["weak"]) == 0
    assert saved["trusted"][0]["cx"] == pytest.approx(0.77)


def test_layer3_nms_deduplication_omni_removal_and_label_sync(tmp_path: Path) -> None:
    """Katman 3: Aynı sınıftan üst üste binen mükerrer kutuları NMS ile tekilleştirir (`deduped_boxes`),
    sentetik `omni_*` ve eşik altı kutuları temizler ve `.txt` etiket dosyasını senkronize yazar.
    """
    stem = "scene_layer3_nms"
    textured_frame = _make_textured_frame()

    best_box = {
        "class_id": 5,
        "class_name": "wood_tree",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.24,
        "h": 0.24,
        "confidence": 0.92,
        "trusted": True,
        "source": "yolo_detector",
    }
    duplicate_lower_1 = {
        "class_id": 5,
        "class_name": "wood_tree",
        "cx": 0.51,
        "cy": 0.50,
        "w": 0.24,
        "h": 0.24,
        "confidence": 0.75,
        "source": "yolo_detector",
    }
    duplicate_lower_2 = {
        "class_id": 5,
        "class_name": "wood_tree",
        "cx": 0.50,
        "cy": 0.51,
        "w": 0.23,
        "h": 0.23,
        "confidence": 0.60,
        "source": "yolo_detector",
    }
    omni_box = {
        "class_id": 10,
        "class_name": "monster",
        "cx": 0.25,
        "cy": 0.60,
        "w": 0.14,
        "h": 0.14,
        "confidence": 0.70,
        "omni_synthetic": True,
        "source": "omni_monster_layout",
    }
    sub_floor_box = {
        "class_id": 3,
        "class_name": "mine_orichalcum",
        "cx": 0.20,
        "cy": 0.25,
        "w": 0.12,
        "h": 0.12,
        "confidence": 0.30,
        "source": "yolo_detector",
    }

    review_path = _write_review_file(
        tmp_path,
        "train",
        stem,
        trusted=[best_box],
        weak=[duplicate_lower_1, duplicate_lower_2, omni_box, sub_floor_box],
    )
    mem = _DummyExemplarMemory()

    stats = clean_poison_labels_ai(
        tmp_path,
        exemplar_memory=mem,
        frame_map={stem: textured_frame},
        floor=0.45,
        emin_threshold=0.85,
    )

    assert stats["deduped_boxes"] == 2
    assert stats["omni_demoted_boxes"] == 1
    assert stats["sub_floor_deleted"] == 1
    assert stats["poison_deleted_boxes"] == 2
    assert stats["files_rewritten"] == 1
    assert stats["labels_rewritten"] == 1

    saved = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 1
    assert len(saved["weak"]) == 0
    assert saved["trusted"][0]["confidence"] == pytest.approx(0.92, abs=1e-4)

    label_path = tmp_path / "labels" / "train" / f"{stem}.txt"
    assert label_path.is_file()
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("5 0.500000 0.500000 0.240000 0.240000")
