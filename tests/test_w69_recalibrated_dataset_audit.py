# -*- coding: utf-8 -*-
"""W69 real dataset audit tests for recalibrated dataset/review/train and dataset/review/val JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.label_trust import (
    UNVERIFIED_HEURISTIC_SOURCES,
    _is_hand_or_gather_seed,
    annotation_confidence,
    calibrate_box_confidence_and_trust,
    is_material_conflict,
    is_omni_synthetic_source,
)

REVIEW_ROOT = ROOT_DIR / "dataset" / "review"
SPLITS = ("train", "val")


def _has_verified_evidence(box: Dict[str, Any], payload: Dict[str, Any] | None = None) -> bool:
    """Return True if the box (or its parent review payload) carries explicit human/template/exemplar proof."""
    if (
        box.get("hand_label") is True
        or box.get("human_approved") is True
        or box.get("approved") is True
        or box.get("confirmed") is True
        or box.get("exemplar_match") is True
        or _is_hand_or_gather_seed(box)
    ):
        return True
    if isinstance(payload, dict) and (
        payload.get("human_approved") is True
        or str(payload.get("done", "")).strip().lower() == "ok"
    ):
        return True
    return False


def _is_unverified_heuristic_box(box: Dict[str, Any]) -> bool:
    """Return True if box source matches UNVERIFIED_HEURISTIC_SOURCES or synthetic fallback markers."""
    src = str(box.get("source", "")).strip().lower()
    return (
        src in UNVERIFIED_HEURISTIC_SOURCES
        or any(uhs in src for uhs in UNVERIFIED_HEURISTIC_SOURCES)
        or is_omni_synthetic_source(src)
        or box.get("omni_synthetic") is True
        or box.get("synthetic_fallback") is True
    )


@pytest.fixture(scope="module")
def review_records() -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Load all review JSON records from dataset/review/train and dataset/review/val once."""
    records: List[Tuple[str, Path, Dict[str, Any]]] = []
    for split in SPLITS:
        split_dir = REVIEW_ROOT / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.json")):
            if path.name == "trusted_seed.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict), f"Invalid JSON object in {path}"
            records.append((split, path, payload))
    return records


def test_review_splits_exist_and_contain_valid_json_records(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """1. Verify dataset/review/train and dataset/review/val exist and contain well-formed review JSONs."""
    assert (REVIEW_ROOT / "train").is_dir(), "dataset/review/train directory is missing"
    assert (REVIEW_ROOT / "val").is_dir(), "dataset/review/val directory is missing"
    assert len(review_records) > 0, "No review JSON files found across train/val"

    counts_by_split = {"train": 0, "val": 0}
    for split, path, payload in review_records:
        counts_by_split[split] += 1
        assert isinstance(payload.get("trusted", []), list), f"'trusted' is not a list in {path}"
        assert isinstance(payload.get("weak", []), list), f"'weak' is not a list in {path}"

    assert counts_by_split["train"] > 0, "dataset/review/train has no JSON records"
    assert counts_by_split["val"] > 0, "dataset/review/val has no JSON records"


def test_all_trusted_boxes_are_ge_085_or_evidenced(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """2. Verify every trusted box in dataset/review/{train,val} has confidence >= 0.85 or verified evidence."""
    violations: List[str] = []
    trusted_total = 0

    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            trusted_total += 1
            conf = annotation_confidence(box)
            evidenced = _has_verified_evidence(box, payload)

            if box.get("trusted") is not True or box.get("weak") is True:
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: invalid flags trusted={box.get('trusted')} weak={box.get('weak')}"
                )
            if not (conf >= 0.85 or evidenced):
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: conf={conf} < 0.85 without evidence (source={box.get('source')})"
                )

    assert trusted_total > 0, "Expected at least one trusted box across dataset/review/{train,val}"
    assert not violations, "Trusted box audit violations:\n" + "\n".join(violations[:25])


def test_no_unevidenced_unverified_heuristic_sources_in_trusted(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """3. Verify no unevidenced box from UNVERIFIED_HEURISTIC_SOURCES is ever placed in the trusted list."""
    leaked_into_trusted: List[str] = []

    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            if _is_unverified_heuristic_box(box) and not _has_verified_evidence(box, payload):
                leaked_into_trusted.append(
                    f"{split}/{path.name}#trusted[{idx}]: source={box.get('source')} conf={annotation_confidence(box)}"
                )

    assert not leaked_into_trusted, (
        "Unevidenced UNVERIFIED_HEURISTIC_SOURCES found in trusted list:\n"
        + "\n".join(leaked_into_trusted[:25])
    )


def test_all_unevidenced_unverified_heuristic_boxes_in_weak_and_le_068(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """4. Verify all unevidenced UNVERIFIED_HEURISTIC_SOURCES boxes are kept in weak with confidence <= 0.68."""
    violations: List[str] = []
    unverified_weak_count = 0

    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if isinstance(box, dict) and _is_unverified_heuristic_box(box) and not _has_verified_evidence(box, payload):
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: unverified heuristic box in trusted (source={box.get('source')})"
                )

        for idx, box in enumerate(payload.get("weak") or []):
            if not isinstance(box, dict):
                continue
            if _is_unverified_heuristic_box(box) and not _has_verified_evidence(box, payload):
                unverified_weak_count += 1
                conf = annotation_confidence(box)
                if conf > 0.68:
                    violations.append(
                        f"{split}/{path.name}#weak[{idx}]: source={box.get('source')} conf={conf} > 0.68"
                    )
                if box.get("trusted") is True or box.get("weak") is not True:
                    violations.append(
                        f"{split}/{path.name}#weak[{idx}]: invalid flags trusted={box.get('trusted')} weak={box.get('weak')}"
                    )

    assert unverified_weak_count > 0, "Expected real UNVERIFIED_HEURISTIC_SOURCES boxes in weak across dataset"
    assert not violations, (
        "UNVERIFIED_HEURISTIC_SOURCES weak list audit violations:\n"
        + "\n".join(violations[:25])
    )


def test_recalibration_consistency_and_material_conflict_exclusion(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """5. Verify recalibration invariants across train/val: trusted boxes stay trusted under recalibration and material conflicts are demoted to weak (<= 0.45)."""
    violations: List[str] = []
    weak_total = 0

    for split, path, payload in review_records:
        stem = str(payload.get("stem") or path.stem)

        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            cid = int(box.get("class_id", -1))
            cname = str(box.get("class_name", ""))
            evidenced = _has_verified_evidence(box, payload)
            if not evidenced and is_material_conflict(stem, cname, cid):
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: conflicting material {cname} (cid={cid}) in trusted on stem={stem}"
                )
            recal = calibrate_box_confidence_and_trust(dict(box), stem=stem)
            if recal.get("trusted") is not True and not evidenced:
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: trusted box would demote under recalibration (conf={box.get('confidence')}, source={box.get('source')})"
                )

        for idx, box in enumerate(payload.get("weak") or []):
            if not isinstance(box, dict):
                continue
            weak_total += 1
            if box.get("trusted") is True or box.get("weak") is not True:
                violations.append(
                    f"{split}/{path.name}#weak[{idx}]: weak box has trusted={box.get('trusted')} weak={box.get('weak')}"
                )
            cid = int(box.get("class_id", -1))
            cname = str(box.get("class_name", ""))
            if not _has_verified_evidence(box, payload) and is_material_conflict(stem, cname, cid):
                conf = annotation_confidence(box)
                if conf > 0.45:
                    violations.append(
                        f"{split}/{path.name}#weak[{idx}]: material conflict conf={conf} > 0.45 on stem={stem}"
                    )

    assert weak_total > 0, "Expected at least one weak box across dataset/review/{train,val}"
    assert not violations, "Recalibration consistency violations:\n" + "\n".join(violations[:25])
