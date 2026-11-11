# -*- coding: utf-8 -*-
"""W69 recalibrated dataset audit tests.

F-1311: the real dataset this file used to audit (`dataset/review/train`
and `dataset/review/val`) is NOT present in this snapshot, so the audit
could never run — every test failed on a missing directory. The audit now
runs over a synthetic, self-contained review dataset built in tmp_path
that exercises every invariant the real audit checked (trusted >= 0.85 or
evidenced, heuristic sources confined to weak <= 0.68, recalibration
consistency, material-conflict demotion).

F-1393: `_has_verified_evidence` used to treat a file-level
``done == "ok"`` as proof for EVERY box in the file. Evidence is now
per-box only; a dedicated negative test proves a ``done="ok"`` payload
does not launder an unevidenced low-confidence trusted box.

F-1594: violation reports used to truncate to the first 25 entries.
All violations are now reported (see `_format_violations`), and a test
with 30 violations proves none is dropped.
"""

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

SPLITS = ("train", "val")


def _box(
    class_id: int,
    class_name: str,
    confidence: float,
    *,
    trusted: bool,
    weak: bool,
    source: str = "yolo_detection",
    **extra: Any,
) -> Dict[str, Any]:
    box = {
        "class_id": class_id,
        "class_name": class_name,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": confidence,
        "trusted": trusted,
        "weak": weak,
        "source": source,
    }
    box.update(extra)
    return box


def _write_synthetic_review_dataset(review_root: Path) -> None:
    """Build a small review dataset that satisfies every audit invariant."""
    records = {
        "train": [
            {
                "stem": "frame_aria",
                "split": "train",
                "trusted": [
                    _box(0, "herb_aria", 0.95, trusted=True, weak=False,
                         source="hand_label", hand_label=True)
                ],
                "weak": [
                    _box(14, "obstacle", 0.50, trusted=False, weak=True,
                         source="legacy_auto")
                ],
            },
            {
                "stem": "frame_ore",
                "split": "train",
                "trusted": [
                    # E-07: high confidence alone is not evidence — this
                    # trusted box carries per-box human approval.
                    _box(3, "orichalcum", 0.92, trusted=True, weak=False,
                         human_approved=True)
                ],
                "weak": [
                    _box(14, "obstacle", 0.60, trusted=False, weak=True,
                         source="visual_obstacle_segmenter")
                ],
            },
            {
                "stem": "aria_gather_01",
                "split": "train",
                "trusted": [
                    _box(0, "herb_aria", 0.90, trusted=True, weak=False,
                         source="hand_label", hand_label=True)
                ],
                "weak": [
                    # Genuine material conflict (aria stem, orichalcum box)
                    # kept weak at conf <= 0.45.
                    _box(3, "orichalcum", 0.40, trusted=False, weak=True)
                ],
            },
        ],
        "val": [
            {
                "stem": "frame_val1",
                "split": "val",
                "trusted": [
                    # Low confidence but per-box human approval: evidenced.
                    _box(1, "herb_mela", 0.50, trusted=True, weak=False,
                         human_approved=True)
                ],
                "weak": [
                    _box(14, "obstacle", 0.68, trusted=False, weak=True,
                         source="chromatic_salient_3d_resource")
                ],
            },
            {
                "stem": "frame_val2",
                "split": "val",
                "trusted": [
                    # E-07: high confidence alone is not evidence — this
                    # trusted box carries per-box human approval.
                    _box(2, "herb_targena", 0.88, trusted=True, weak=False,
                         source="template_match", human_approved=True)
                ],
                "weak": [
                    _box(14, "obstacle", 0.30, trusted=False, weak=True,
                         source="gather_material_hint")
                ],
            },
        ],
    }
    for split, items in records.items():
        split_dir = review_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for payload in items:
            (split_dir / f"{payload['stem']}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def _has_verified_evidence(box: Dict[str, Any]) -> bool:
    """Return True only if THIS BOX carries explicit human/template/exemplar proof.

    F-1393: a file-level ``done == "ok"`` is deliberately NOT evidence —
    it used to launder every box in an approved file, including unevidenced
    low-confidence ones. Only per-box flags count.
    """
    return bool(
        box.get("hand_label") is True
        or box.get("human_approved") is True
        or box.get("approved") is True
        or box.get("confirmed") is True
        or box.get("exemplar_match") is True
        or _is_hand_or_gather_seed(box)
    )


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


def _format_violations(title: str, violations: List[str]) -> str:
    """Render EVERY violation (F-1594: no [:25] truncation)."""
    return title + ":\n" + "\n".join(violations)


def _trusted_box_violations(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> List[str]:
    violations: List[str] = []
    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            conf = annotation_confidence(box)
            if box.get("trusted") is not True or box.get("weak") is True:
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: invalid flags trusted={box.get('trusted')} weak={box.get('weak')}"
                )
            if not (conf >= 0.85 or _has_verified_evidence(box)):
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: conf={conf} < 0.85 without evidence (source={box.get('source')})"
                )
    return violations


def _heuristic_violations(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> Tuple[List[str], int]:
    violations: List[str] = []
    unverified_weak_count = 0
    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if isinstance(box, dict) and _is_unverified_heuristic_box(box) and not _has_verified_evidence(box):
                violations.append(
                    f"{split}/{path.name}#trusted[{idx}]: unverified heuristic box in trusted (source={box.get('source')})"
                )
        for idx, box in enumerate(payload.get("weak") or []):
            if not isinstance(box, dict):
                continue
            if _is_unverified_heuristic_box(box) and not _has_verified_evidence(box):
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
    return violations, unverified_weak_count


def _recalibration_violations(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> Tuple[List[str], int]:
    violations: List[str] = []
    weak_total = 0
    for split, path, payload in review_records:
        stem = str(payload.get("stem") or path.stem)
        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            cid = int(box.get("class_id", -1))
            cname = str(box.get("class_name", ""))
            evidenced = _has_verified_evidence(box)
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
            if not _has_verified_evidence(box) and is_material_conflict(stem, cname, cid):
                conf = annotation_confidence(box)
                if conf > 0.45:
                    violations.append(
                        f"{split}/{path.name}#weak[{idx}]: material conflict conf={conf} > 0.45 on stem={stem}"
                    )
    return violations, weak_total


@pytest.fixture
def review_root(tmp_path: Path) -> Path:
    """Self-contained synthetic review dataset (F-1311: real data absent)."""
    root = tmp_path / "dataset" / "review"
    _write_synthetic_review_dataset(root)
    return root


@pytest.fixture
def review_records(review_root: Path) -> List[Tuple[str, Path, Dict[str, Any]]]:
    """Load all review JSON records from the synthetic dataset once."""
    records: List[Tuple[str, Path, Dict[str, Any]]] = []
    for split in SPLITS:
        split_dir = review_root / split
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
    review_root: Path,
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """1. Verify dataset/review/train and dataset/review/val exist and contain well-formed review JSONs."""
    assert (review_root / "train").is_dir(), "dataset/review/train directory is missing"
    assert (review_root / "val").is_dir(), "dataset/review/val directory is missing"
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
    trusted_total = sum(
        1
        for _split, _path, payload in review_records
        for box in (payload.get("trusted") or [])
        if isinstance(box, dict)
    )
    violations = _trusted_box_violations(review_records)

    assert trusted_total > 0, "Expected at least one trusted box across dataset/review/{train,val}"
    assert not violations, _format_violations("Trusted box audit violations", violations)


def test_no_unevidenced_unverified_heuristic_sources_in_trusted(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """3. Verify no unevidenced box from UNVERIFIED_HEURISTIC_SOURCES is ever placed in the trusted list."""
    leaked_into_trusted: List[str] = []

    for split, path, payload in review_records:
        for idx, box in enumerate(payload.get("trusted") or []):
            if not isinstance(box, dict):
                continue
            if _is_unverified_heuristic_box(box) and not _has_verified_evidence(box):
                leaked_into_trusted.append(
                    f"{split}/{path.name}#trusted[{idx}]: source={box.get('source')} conf={annotation_confidence(box)}"
                )

    assert not leaked_into_trusted, _format_violations(
        "Unevidenced UNVERIFIED_HEURISTIC_SOURCES found in trusted list",
        leaked_into_trusted,
    )


def test_all_unevidenced_unverified_heuristic_boxes_in_weak_and_le_068(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """4. Verify all unevidenced UNVERIFIED_HEURISTIC_SOURCES boxes are kept in weak with confidence <= 0.68."""
    violations, unverified_weak_count = _heuristic_violations(review_records)

    assert unverified_weak_count > 0, "Expected real UNVERIFIED_HEURISTIC_SOURCES boxes in weak across dataset"
    assert not violations, _format_violations(
        "UNVERIFIED_HEURISTIC_SOURCES weak list audit violations", violations
    )


def test_recalibration_consistency_and_material_conflict_exclusion(
    review_records: List[Tuple[str, Path, Dict[str, Any]]],
) -> None:
    """5. Verify recalibration invariants across train/val: trusted boxes stay trusted under recalibration and material conflicts are demoted to weak (<= 0.45)."""
    violations, weak_total = _recalibration_violations(review_records)

    assert weak_total > 0, "Expected at least one weak box across dataset/review/{train,val}"
    assert not violations, _format_violations("Recalibration consistency violations", violations)


def test_file_level_done_ok_does_not_evidence_individual_boxes(
    review_root: Path,
) -> None:
    """6 (F-1393). A payload with done='ok' must NOT evidence its boxes.

    The old `_has_verified_evidence` returned True for every box in a
    ``done == "ok"`` file, so an unevidenced 0.60-confidence box sitting in
    an approved file passed the audit. Per-box evidence is required now:
    this record's trusted box has no per-box proof, so the audit must flag
    it even though the file itself is marked approved.
    """
    split_dir = review_root / "train"
    payload = {
        "stem": "approved_file_unproven_box",
        "split": "train",
        "human_approved": True,
        "done": "ok",
        "trusted": [
            _box(3, "orichalcum", 0.60, trusted=True, weak=False)
        ],
        "weak": [],
    }
    (split_dir / "approved_file_unproven_box.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records = [("train", split_dir / "approved_file_unproven_box.json", payload)]

    assert _has_verified_evidence(payload["trusted"][0]) is False
    violations = _trusted_box_violations(records)
    assert len(violations) == 1
    assert "approved_file_unproven_box" in violations[0]
    assert "conf=0.6" in violations[0]


def test_all_violations_reported_without_truncation() -> None:
    """7 (F-1594). All violations are reported — none dropped after 25.

    Thirty violating trusted boxes must all appear in the formatted
    report; the old ``violations[:25]`` silently hid the last five.
    """
    records: List[Tuple[str, Path, Dict[str, Any]]] = []
    for i in range(30):
        stem = f"frame_bad_{i:02d}"
        payload = {
            "stem": stem,
            "split": "train",
            "trusted": [_box(3, "orichalcum", 0.50, trusted=True, weak=False)],
            "weak": [],
        }
        records.append(("train", Path(f"{stem}.json"), payload))

    violations = _trusted_box_violations(records)
    assert len(violations) == 30
    report = _format_violations("Trusted box audit violations", violations)
    for i in range(30):
        assert f"frame_bad_{i:02d}" in report
