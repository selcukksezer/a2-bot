# -*- coding: utf-8 -*-
"""W45 unit tests for box-level low-confidence pruning (`prune_low_confidence_boxes_in_dataset`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.label_trust import prune_low_confidence_boxes_in_dataset


def _make_four_boxes(
    low_conf: float = 0.32,
    high_confs: Tuple[float, float, float] = (0.94, 0.91, 0.88),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (trusted, weak) totaling 4 boxes: 3 high-confidence + 1 low-confidence wrong box."""
    trusted = [
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "cx": 0.25,
            "cy": 0.30,
            "w": 0.12,
            "h": 0.14,
            "confidence": high_confs[0],
            "trusted": True,
            "weak": False,
        },
        {
            "class_id": 3,
            "class_name": "ore_orichalcum",
            "cx": 0.50,
            "cy": 0.55,
            "w": 0.15,
            "h": 0.16,
            "confidence": high_confs[1],
            "trusted": True,
            "weak": False,
        },
    ]
    weak = [
        {
            "class_id": 5,
            "class_name": "wood_tree",
            "cx": 0.72,
            "cy": 0.62,
            "w": 0.14,
            "h": 0.18,
            "confidence": high_confs[2],
            "trusted": False,
            "weak": True,
        },
        {
            "class_id": 14,
            "class_name": "obstacle",
            "cx": 0.40,
            "cy": 0.80,
            "w": 0.10,
            "h": 0.10,
            "confidence": low_conf,
            "trusted": False,
            "weak": True,
        },
    ]
    return trusted, weak


def _setup_frame(
    dataset_dir: Path,
    stem: str,
    trusted: List[Dict[str, Any]],
    weak: List[Dict[str, Any]],
    split: str = "train",
) -> Tuple[Path, Path, Path]:
    img_dir = dataset_dir / "images" / split
    rev_dir = dataset_dir / "review" / split
    lbl_dir = dataset_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # F-1587: a REAL decodable PNG (not dummy bytes that no decoder ever
    # reads). Pruning must preserve a genuine image, and the negative-
    # exemplar path can actually load it via cv2.imread.
    import cv2

    img_path = img_dir / f"{stem}.png"
    assert cv2.imwrite(str(img_path), np.full((200, 200, 3), (60, 120, 180), dtype=np.uint8))
    decoded = cv2.imread(str(img_path))
    assert decoded is not None and decoded.shape == (200, 200, 3)

    rev_path = rev_dir / f"{stem}.json"
    payload = {
        "stem": stem,
        "split": split,
        "trusted": trusted,
        "weak": weak,
    }
    rev_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lbl_path = lbl_dir / f"{stem}.txt"
    all_boxes = list(trusted) + list(weak)
    lines = [
        f"{int(b['class_id'])} {float(b['cx']):.6f} {float(b['cy']):.6f} {float(b['w']):.6f} {float(b['h']):.6f}"
        for b in all_boxes
    ]
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return img_path, rev_path, lbl_path


def test_prune_one_low_conf_box_preserves_image_and_three_high_conf_boxes(tmp_path: Path) -> None:
    """1. 4-box frame (3 high-conf, 1 low-conf): image and 3 boxes are kept, 1 low-conf box deleted."""
    trusted, weak = _make_four_boxes(low_conf=0.32, high_confs=(0.95, 0.90, 0.87))
    img_path, rev_path, _ = _setup_frame(tmp_path, "frame_4box_01", trusted, weak)

    res = prune_low_confidence_boxes_in_dataset(
        dataset_dir=tmp_path,
        threshold=0.85,
        record_negative_exemplars=False,
    )

    assert res["ok"] is True
    assert res["scanned"] == 1
    assert res["preserved_frames"] == 1
    assert res["pruned_boxes"] == 1
    assert res["promoted_boxes"] == 3
    assert img_path.is_file(), "Image file must never be deleted during box-level pruning"

    # F-1587: "image preserved" must mean a still-decodable image, not just
    # an existing file with dummy bytes.
    import cv2

    preserved = cv2.imread(str(img_path))
    assert preserved is not None and preserved.shape == (200, 200, 3)

    updated = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(updated["trusted"]) == 3
    assert len(updated["weak"]) == 0
    kept_cids = [b["class_id"] for b in updated["trusted"]]
    assert kept_cids == [0, 3, 5]
    assert all(b["trusted"] is True and b["weak"] is False for b in updated["trusted"])


def test_prune_updates_yolo_train_txt_to_keep_exact_three_boxes(tmp_path: Path) -> None:
    """2. YOLO train label .txt is synchronized to contain only the 3 high-confidence boxes."""
    trusted, weak = _make_four_boxes(low_conf=0.41, high_confs=(0.93, 0.89, 0.86))
    img_path, _, lbl_path = _setup_frame(tmp_path, "frame_4box_02", trusted, weak)

    assert len(lbl_path.read_text(encoding="utf-8").strip().splitlines()) == 4

    prune_low_confidence_boxes_in_dataset(
        dataset_dir=tmp_path,
        threshold=0.85,
        record_negative_exemplars=False,
    )

    assert img_path.exists()
    lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    parsed_class_ids = [int(ln.split()[0]) for ln in lines]
    assert parsed_class_ids == [0, 3, 5]
    assert 14 not in parsed_class_ids


def test_prune_records_negative_exemplar_only_for_deleted_low_conf_box(tmp_path: Path) -> None:
    """3. Only the 1 pruned low-confidence box is recorded as a negative exemplar in ExemplarMemory.

    F-1509: the old test used a fake in-memory spy, so it only proved the
    prune function *called* something. This test passes a REAL
    ExemplarMemory (tmp JSON path) and verifies the exemplar is actually
    persisted and retrievable through the public match path.
    """
    from dataset_factory.exemplar_memory import ExemplarMemory

    trusted, weak = _make_four_boxes(low_conf=0.25, high_confs=(0.96, 0.92, 0.88))
    img_path, _, _ = _setup_frame(tmp_path, "frame_4box_03", trusted, weak)

    mem_path = tmp_path / "learned_exemplars.json"
    real_mem = ExemplarMemory(json_path=mem_path)
    # K-3 (v12 daraltma): dejenere kapısı artık yalnızca POZİTİF tarafta
    # uygulanır — düz gri çerçeveden çıkan içeriksiz crop negatif olarak
    # kabul edilir ("bu gri yama kaynak değil" bastırma sinyali).
    # Brightness-only noise (A-set K-3 method): the same delta on all
    # channels keeps hue/saturation gray while V varies, so the crop
    # carries real texture (and would pass the gate even on the
    # positive side); the negative is accepted either way under v12.
    rng = np.random.default_rng(7)
    noise = rng.integers(-30, 31, size=(200, 200, 1), dtype=np.int16)
    dummy_frame = np.clip(
        np.full((200, 200, 3), 128, dtype=np.int16) + noise, 0, 255
    ).astype(np.uint8)

    res = prune_low_confidence_boxes_in_dataset(
        dataset_dir=tmp_path,
        threshold=0.85,
        record_negative_exemplars=True,
        exemplar_memory=real_mem,
        frame_map={"frame_4box_03": dummy_frame},
    )

    assert img_path.is_file()
    assert res["pruned_boxes"] == 1
    assert res["promoted_boxes"] == 3

    # Persisted: exactly one negative exemplar, for the pruned box only.
    persisted = json.loads(mem_path.read_text(encoding="utf-8"))
    assert len(persisted["negative"]) == 1
    assert persisted["negative"][0]["class_id"] == 14
    assert persisted["negative"][0]["class_name"] == "obstacle"
    assert persisted["negative"][0]["source"] == "bulk_prune_low_conf"
    assert persisted["positive"] == []

    # Retrievable through the public match path: the pruned box's crop
    # (20x20 gray crop of the dummy frame at cx=0.40, cy=0.80) matches it.
    reloaded = ExemplarMemory(json_path=mem_path)
    probe_crop = dummy_frame[150:170, 70:90]
    neg_sim, neg_ex = reloaded.match_negative(probe_crop, class_id=14)
    assert neg_sim >= 0.99
    assert neg_ex is not None and neg_ex["class_id"] == 14


def test_prune_protects_hand_label_among_three_kept_boxes_while_pruning_fourth_low_conf(
    tmp_path: Path,
) -> None:
    """4. Among 4 boxes (2 high-conf, 1 hand_label, 1 low-conf auto), 3 are preserved and 1 is pruned."""
    trusted, weak = _make_four_boxes(low_conf=0.29, high_confs=(0.92, 0.89, 0.60))
    # Make the 3rd box an explicit hand_label even though its numeric score is 0.60 (< 0.85)
    weak[0]["hand_label"] = True
    img_path, rev_path, lbl_path = _setup_frame(tmp_path, "frame_4box_04", trusted, weak)

    res = prune_low_confidence_boxes_in_dataset(
        dataset_dir=tmp_path,
        threshold=0.85,
        record_negative_exemplars=False,
    )

    assert img_path.is_file()
    assert res["pruned_boxes"] == 1
    assert res["promoted_boxes"] == 3

    updated = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(updated["trusted"]) == 3
    assert len(updated["weak"]) == 0
    assert [b["class_id"] for b in updated["trusted"]] == [0, 3, 5]
    assert len(lbl_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_prune_with_stem_filter_and_min_confidence_alias_on_four_box_frame(tmp_path: Path) -> None:
    """5. `indices_or_stems` + `min_confidence` prunes only the targeted 4-box frame.

    F-1510: the old test used min_confidence=0.85 — identical to the default
    threshold — so it could not tell the alias from the default, and
    teach_negative=False had no observable effect. Here min_confidence=0.92
    is deliberately DIFFERENT from the default: with the default 0.85 only
    the 0.38 box would be pruned (1 pruned / 3 promoted), while 0.92 prunes
    the 0.91 and 0.86 boxes too (3 pruned / 1 promoted). teach_negative=False
    is observed through a real ExemplarMemory that must stay empty.
    """
    from dataset_factory.exemplar_memory import ExemplarMemory

    t1, w1 = _make_four_boxes(low_conf=0.38, high_confs=(0.94, 0.91, 0.86))
    t2, w2 = _make_four_boxes(low_conf=0.20, high_confs=(0.95, 0.90, 0.88))
    img1, rev1, _ = _setup_frame(tmp_path, "target_4box", t1, w1)
    img2, rev2, _ = _setup_frame(tmp_path, "untouched_4box", t2, w2)

    mem_path = tmp_path / "learned_exemplars.json"
    real_mem = ExemplarMemory(json_path=mem_path)

    res = prune_low_confidence_boxes_in_dataset(
        dataset_dir=tmp_path,
        min_confidence=0.92,
        indices_or_stems=["target_4box"],
        teach_negative=False,
        exemplar_memory=real_mem,
    )

    assert res["scanned"] == 1
    # 0.92 alias honored (default 0.85 would give pruned=1, promoted=3):
    assert res["pruned_boxes"] == 3
    assert res["promoted_boxes"] == 1
    assert res["preserved_frames"] == 1
    assert img1.is_file() and img2.is_file()

    up1 = json.loads(rev1.read_text(encoding="utf-8"))
    assert len(up1["trusted"]) == 1 and len(up1["weak"]) == 0
    assert [b["class_id"] for b in up1["trusted"]] == [0]

    up2 = json.loads(rev2.read_text(encoding="utf-8"))
    assert len(up2["trusted"]) == 2 and len(up2["weak"]) == 2

    # teach_negative=False observed: no negative exemplar was recorded even
    # though 3 boxes were pruned and a real memory was supplied.
    assert not mem_path.is_file() or json.loads(mem_path.read_text(encoding="utf-8"))["negative"] == []
