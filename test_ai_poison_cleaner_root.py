# -*- coding: utf-8 -*-
"""
Tests for Work Package I (4-Layer AI Poison Cleaner) and Work Package J (Box-Level Bulk Pruning / Görseli Koru).
- Layer 0: Immunity shield for hand-labeled, human-approved, gather seeds, or positive exemplar matches.
- Layer 1: AI exemplar memory class repair and negative suppression.
- Layer 2: Visual flat-texture / HUD poison detection and negative teaching.
- Layer 3: Omni synthetic cull and NMS deduplication.
- Work Package J: 4-box scenario: 3 high confidence (%92, %95, %91), 1 low confidence (%35).
  Verifies that ONLY the 1 low box is pruned, image is preserved, 3 good boxes are kept as Emin (trusted=True),
  and disk .txt and .json are accurately updated.

NOTE: this is the root-level copy of the poison-cleaner suite (the fuller
suite lives in tests/test_ai_poison_cleaner.py). It is deliberately named
``*_root`` so pytest does not see two modules called
``test_ai_poison_cleaner`` — the duplicate basename made the whole slice
uncollectable in a single pytest run (import file mismatch).
"""

import json
from pathlib import Path
import tempfile
import shutil
import pytest
import numpy as np
import cv2

from dataset_factory.label_trust import (
    clean_poison_labels_ai,
    prune_low_confidence_boxes_in_dataset,
    requeue_labeled_dataset_for_review,
)
from dataset_factory.exemplar_memory import ExemplarMemory, extract_box_crop


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Redirect CWD-relative production writes into a throwaway dir.

    ``mark_model_stale`` writes ``models/stale_manifest.json`` and
    ``dataset/pipeline_ui_state.json`` relative to the current working
    directory, so a run from the repo root rewrote repo files. All dataset
    paths in this file are absolute, so changing CWD is safe here.
    """
    monkeypatch.chdir(tmp_path)


def _isolated_memory(tmp_path: Path) -> ExemplarMemory:
    """Empty exemplar memory backed by a temp file.

    The cleaner constructs the repo's shared
    ``dataset/learned_exemplars.json`` when no memory is passed, and with
    ``teach_negatives=True`` it appends negative exemplars to that shared
    store (and pays a ~1.2 MB JSON load+save per call). Tests must neither
    mutate the shared store nor depend on its contents — a test that passed
    or failed depending on previously taught negatives is order-dependent.
    """
    return ExemplarMemory(json_path=tmp_path / "exemplars.json")


@pytest.fixture
def temp_dataset_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="poison_test_"))
    for split in ("train", "val"):
        (temp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "review" / split).mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _create_synthetic_test_image(path: Path, width: int = 400, height: int = 400, flat: bool = False):
    if flat:
        img = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        # Rich textured image (Laplacian variance >> 10). Seeded: the
        # unseeded np.random.randint made every run generate a different
        # image, so failures were not reproducible byte-for-byte.
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_two_tone_test_image(path: Path, width: int = 400, height: int = 400):
    """Deterministic two-tone textured image: red noise on the left half,
    blue noise on the right half (BGR). The halves have near-orthogonal HSV
    histograms, so a crop from one half does NOT match an exemplar learned
    from the other half — while a crop from the same half matches ~1.0.
    Laplacian variance is far above the Layer 2 flat threshold (10.0)."""
    rng = np.random.default_rng(7)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    half = width // 2
    img[:, :half, 0] = rng.integers(0, 60, size=(height, half), dtype=np.uint8)
    img[:, :half, 1] = rng.integers(0, 60, size=(height, half), dtype=np.uint8)
    img[:, :half, 2] = rng.integers(180, 256, size=(height, half), dtype=np.uint8)
    img[:, half:, 0] = rng.integers(180, 256, size=(height, width - half), dtype=np.uint8)
    img[:, half:, 1] = rng.integers(0, 60, size=(height, width - half), dtype=np.uint8)
    img[:, half:, 2] = rng.integers(0, 60, size=(height, width - half), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_layer0_immunity_shield(temp_dataset_dir, tmp_path):
    """Layer 0: Hand-labeled, approved, or seed boxes MUST NEVER be deleted or demoted."""
    img_path = temp_dataset_dir / "images" / "train" / "img_immune.jpg"
    _create_synthetic_test_image(img_path, flat=True)  # flat image would normally fail layer 2

    rev_path = temp_dataset_dir / "review" / "train" / "img_immune.json"
    payload = {
        "stem": "img_immune",
        "split": "train",
        "trusted": [
            {
                "class_id": 0,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.50,  # low confidence but hand_label
                "hand_label": True,
            },
            {
                "class_id": 1,
                "cx": 0.3,
                "cy": 0.3,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.60,
                "human_approved": True,
            },
        ],
        "weak": [],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")
    before = rev_path.read_text(encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir, exemplar_memory=_isolated_memory(tmp_path))
    assert stats["ok"] is True
    assert stats["protected_boxes"] >= 2
    assert stats["poison_deleted_boxes"] == 0
    assert stats["files_rewritten"] == 0

    # P1-15 (K-16, F3): 0 silmede dosyaya dokunulmaz — koruma bellekte
    # uygulanır, review JSON bayt-bayt aynen durur. `trusted` bayrağı
    # diske ancak bir silme/onarım/NMS yazımı tetiklenince çıkar.
    # Bu kök ikiz, P1-15 test güncellemesinde (2efffdc) yalnızca
    # tests/ ikizi güncellendiği için v16 beklentisini ("trusted diske
    # yazılır") koruyordu; tests/test_ai_poison_cleaner.py
    # ::test_layer0_immunity_shield ile aynı beklenti artık burada da
    # kilitlenir. Üretim kodu değişmedi — K-16 bilinçli kararıdır.
    assert rev_path.read_text(encoding="utf-8") == before
    saved = json.loads(before)
    assert len(saved["trusted"]) == 2
    assert saved["trusted"][0].get("hand_label") is True
    assert saved["trusted"][1].get("human_approved") is True


def test_layer1_exemplar_memory_repair_and_suppression(temp_dataset_dir, tmp_path):
    """Layer 1: Auto-repair class if positive exemplar matches >= 0.82; suppress if negative matches.

    REAL exemplar matching: the cleaner is given an ExemplarMemory that
    learned one positive exemplar (class 2, from the red left-half crop)
    and one negative exemplar (from the blue right-half crop). The input
    boxes carry NO exemplar_* answer fields — the repair class and the
    suppression must come out of match_positive / match_negative on the
    actual image crops.
    """
    img_path = temp_dataset_dir / "images" / "train" / "img_repair.jpg"
    _create_two_tone_test_image(img_path)
    frame = cv2.imread(str(img_path))
    assert frame is not None

    repair_box = {
        "class_id": 0,
        "cx": 0.25,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.88,
    }
    poison_box = {
        "class_id": 1,
        "cx": 0.75,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.70,
    }

    memory = ExemplarMemory(json_path=tmp_path / "exemplars.json")
    # Canonical name must match the id: herb_aria IS cid 0, so the cid-2
    # positive exemplar is herb_targena (K canonicalization stores the
    # name-resolved class; a (2, "herb_aria") pair would land on cid 0).
    assert memory.add_positive_exemplar(
        extract_box_crop(frame, repair_box), 2, "herb_targena", source="ui_approval"
    )
    assert memory.add_negative_exemplar(
        extract_box_crop(frame, poison_box), 1, "herb_mela", source="ui_rejection"
    )

    # Sanity: the real matcher really fires on these crops before the cleaner runs.
    pos_sim, pos_cid, _, _ = memory.match_positive(extract_box_crop(frame, repair_box))
    assert pos_sim >= 0.82 and pos_cid == 2
    neg_sim, _ = memory.match_negative(extract_box_crop(frame, poison_box), class_id=1)
    assert neg_sim >= 0.82

    rev_path = temp_dataset_dir / "review" / "train" / "img_repair.json"
    payload = {
        "stem": "img_repair",
        "split": "train",
        "trusted": [],
        "weak": [repair_box, poison_box],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir, exemplar_memory=memory)
    assert stats["ok"] is True
    assert stats["auto_repaired_boxes"] >= 1
    assert stats["poison_deleted_boxes"] >= 1

    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    # One box repaired and kept, the other deleted
    remaining = saved["trusted"] + saved["weak"]
    assert len(remaining) == 1
    assert remaining[0]["class_id"] == 2  # Repaired from 0 to 2 by real exemplar match
    assert remaining[0].get("repaired_by_exemplar") is True


def test_layer2_flat_surface_and_hud_poison(temp_dataset_dir, tmp_path):
    """Layer 2: Boxes on flat texture (Laplacian var < 10) or illegal HUD zones pruned."""
    img_path = temp_dataset_dir / "images" / "train" / "img_flat.jpg"
    _create_synthetic_test_image(img_path, flat=True)  # Completely uniform/flat image

    rev_path = temp_dataset_dir / "review" / "train" / "img_flat.json"
    payload = {
        "stem": "img_flat",
        "split": "train",
        "trusted": [],
        "weak": [
            {
                "class_id": 3,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.70,  # weak box on flat surface without immunity
            }
        ],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir, exemplar_memory=_isolated_memory(tmp_path))
    assert stats["ok"] is True
    assert stats["poison_deleted_boxes"] >= 1

    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 0
    assert len(saved["weak"]) == 0


def test_work_package_j_bulk_prune_preserves_image_and_good_boxes(temp_dataset_dir, tmp_path):
    """Work Package J: Image with 4 boxes (3 high: %92, %95, %91; 1 low: %35).
    Verifies that ONLY the 1 low-confidence box is pruned, the image is preserved,
    the 3 good boxes are kept as Emin (trusted=True), and both .json and .txt labels are updated.
    """
    img_path = temp_dataset_dir / "images" / "train" / "frame_wpj.jpg"
    _create_synthetic_test_image(img_path, flat=False)

    lbl_path = temp_dataset_dir / "labels" / "train" / "frame_wpj.txt"
    rev_path = temp_dataset_dir / "review" / "train" / "frame_wpj.json"

    # 4 boxes: 3 high conf (0.92, 0.95, 0.91), 1 low conf (0.35).
    # E-07: high confidence alone never trusts — the 3 good boxes carry
    # per-box human approval so they survive as Emin (trusted=True).
    boxes = [
        {"class_id": 0, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1, "confidence": 0.92, "human_approved": True},
        {"class_id": 1, "cx": 0.4, "cy": 0.4, "w": 0.1, "h": 0.1, "confidence": 0.95, "human_approved": True},
        {"class_id": 2, "cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1, "confidence": 0.91, "human_approved": True},
        {"class_id": 3, "cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1, "confidence": 0.35},
    ]

    # Write initial .txt
    txt_lines = [
        f"{b['class_id']} {b['cx']} {b['cy']} {b['w']} {b['h']} {b['confidence']}"
        for b in boxes
    ]
    lbl_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    # Write initial review .json
    rev_payload = {
        "stem": "frame_wpj",
        "split": "train",
        "trusted": [b for b in boxes if b["confidence"] >= 0.85],
        "weak": [b for b in boxes if b["confidence"] < 0.85],
    }
    rev_path.write_text(json.dumps(rev_payload, indent=2), encoding="utf-8")

    # Execute box-level bulk pruning with threshold 0.85, teaching the
    # pruned box into an isolated temp memory (never the shared store).
    stats = prune_low_confidence_boxes_in_dataset(
        dataset_dir=temp_dataset_dir,
        min_confidence=0.85,
        teach_negative=True,
        exemplar_memory=_isolated_memory(tmp_path),
    )

    assert stats["ok"] is True
    assert stats["boxes_pruned"] == 1
    assert stats["images_checked"] >= 1

    # 1. Image MUST still exist on disk!
    assert img_path.is_file(), "Görsel silinmemeli, korunmalıdır!"

    # 2. Review .json MUST have exactly 3 trusted (Emin) boxes, 0 weak boxes
    assert rev_path.is_file()
    saved_rev = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved_rev["trusted"]) == 3, f"Expected 3 trusted boxes, got {len(saved_rev['trusted'])}"
    assert len(saved_rev["weak"]) == 0, f"Expected 0 weak boxes, got {len(saved_rev['weak'])}"

    saved_classes = [b["class_id"] for b in saved_rev["trusted"]]
    assert set(saved_classes) == {0, 1, 2}, "Classes 0, 1, and 2 must be preserved!"
    assert all(b.get("trusted") is True for b in saved_rev["trusted"])
    assert all(b.get("confidence") >= 0.85 for b in saved_rev["trusted"])

    # 3. Label .txt MUST be updated with exactly 3 high-confidence boxes
    assert lbl_path.is_file()
    txt_content = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(txt_content) == 3, f"Expected 3 lines in .txt, got {len(txt_content)}"
