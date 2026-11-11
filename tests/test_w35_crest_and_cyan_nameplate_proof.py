# -*- coding: utf-8 -*-
r"""
W35 Unit Tests: 3D Winged Gather Crest (`\Y/`) & Cyan/Turquoise Nameplate Alignment Proof.

Verifies that:
1. When a 3D winged gather crest (`\Y/`, `class_id=6`) and a turquoise/cyan material nameplate
   are vertically aligned (`dx <= 45px, 0 <= dy <= 70px`), high confidence (`>= 0.86`, `trusted=True`,
   `weak=False`, `"Emin"`) is produced.
2. When alignment or visual cyan nameplate evidence is missing, misaligned, or belongs to a yellow/red
   monster nameplate, low/suspicious confidence (`< 0.70`, `trusted=False`, `weak=True`, `"Şüpheli"`)
   is produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.vision import VisionEngine
from dataset_factory.crest_nameplate_verifier import (
    CREST_CLASS_ID,
    CrestNameplateVerifier,
)


def _paint_canonical_crest_and_cyan_nameplate(
    frame_bgr: np.ndarray,
    verifier: CrestNameplateVerifier,
    crest_cx: int = 960,
    crest_cy: int = 400,
    draw_crest: bool = True,
    draw_cyan_nameplate: bool = True,
    nameplate_dx: int = 0,
    nameplate_dy: int = 34,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r"""Paint an independent symmetric `\Y/` crest and turquoise/cyan glyph nameplate onto `frame_bgr`.

    The crest is drawn from first principles (two diagonal arms + a vertical
    stem, 29x28) and deliberately does NOT use the verifier's own template
    mask (`verifier._crest_template_mask`): painting with the verifier's mask
    would make detection circular (the test would feed the detector its own
    template). The `verifier` argument is kept for the call signature only.
    """
    del verifier  # Independent of the verifier's internals by design.
    th, tw = 28, 29
    cx1 = int(crest_cx - tw // 2)
    cy1 = int(crest_cy - th // 2)

    if draw_crest:
        cv2.line(frame_bgr, (cx1 + 3, cy1 + 1), (cx1 + 14, cy1 + 13), (250, 250, 250), 5)
        cv2.line(frame_bgr, (cx1 + 25, cy1 + 1), (cx1 + 14, cy1 + 13), (250, 250, 250), 5)
        cv2.line(frame_bgr, (cx1 + 14, cy1 + 13), (cx1 + 14, cy1 + 27), (250, 250, 250), 5)

    np_cx = int(crest_cx + nameplate_dx)
    np_cy = int(crest_cy + nameplate_dy)
    np_w, np_h = 68, 14
    nx1 = int(np_cx - np_w // 2)
    ny1 = int(np_cy - np_h // 2)

    if draw_cyan_nameplate:
        # Dark outline backdrop for local contrast + turquoise/cyan glyph strokes (HSV ~90, 220, 235 -> BGR (235, 225, 30))
        cv2.rectangle(
            frame_bgr,
            (nx1 - 2, ny1 - 2),
            (nx1 + np_w + 2, ny1 + np_h + 2),
            (20, 20, 20),
            -1,
        )
        for gx in range(nx1 + 4, nx1 + np_w - 6, 9):
            cv2.rectangle(
                frame_bgr,
                (gx, ny1 + 3),
                (gx + 5, ny1 + np_h - 3),
                (235, 225, 30),
                -1,
            )

    return (cx1, cy1, tw, th), (nx1, ny1, np_w, np_h)


def test_w35_01_crest_and_cyan_nameplate_vertical_alignment_produces_emin_high_confidence() -> None:
    r"""Test 1: A crest + cyan nameplate pair DETECTED from a painted frame
    (`dx <= 45, dy <= 70`) yields `>= 0.86` Emin.

    The boxes are not hand-made geometry: the crest and the nameplate are
    found by the verifier's own detectors on the painted frame, and the
    alignment check runs on those detected boxes.

    KARAR A (v12, F-0787): the test passes the FULL detector dicts (not
    bare bboxes) and confirms via verify_in_frame() that the crest is
    REALLY in the frame — geometry alone is never verification, and a
    frameless check_vertical_alignment returns Şüpheli. The turquoise
    nameplate pixels and the white crest pixels are measured from the
    frame ROIs (single verification path).

    The synthetic frame is 960x540 (not 1920x1080): the painted crest and
    nameplate keep their pixel size and every threshold is in absolute
    glyph pixels, while full-frame template detection costs ~11x less.
    """
    verifier = CrestNameplateVerifier()

    frame = np.full((540, 960, 3), 85, dtype=np.uint8)
    _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=480,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=4,
        nameplate_dy=34,
    )

    # Real detection: the crest must be found close to where it was painted.
    crests = verifier.detect_winged_crests(frame)
    assert len(crests) >= 1, "Painted crest was not detected"
    crest_det = crests[0]
    assert abs(crest_det["cx"] - 480) <= 15
    assert abs(crest_det["cy"] - 200) <= 15

    nameplates = verifier.detect_turquoise_nameplates(frame)
    assert len(nameplates) >= 1, "Painted cyan nameplate was not detected"
    np_det = nameplates[0]
    assert abs(np_det["cx"] - 484) <= 15
    assert abs(np_det["cy"] - 234) <= 15

    # KARAR A (v12, F-0787): full detector dicts + frame-measured proof.
    # verify_in_frame() confirms the crest is REALLY in the frame (white
    # crest pixels measured in its ROI) and the nameplate's turquoise
    # pixels are measured too — dict-carried numbers are never evidence.
    res_frame = verifier.verify_in_frame(frame, crest_box=crest_det, nameplate_box=np_det)
    assert res_frame.verified is True
    assert res_frame.aligned is True
    assert res_frame.has_evidence is True
    assert res_frame.trusted is True
    assert res_frame.weak is False
    assert res_frame.status == "Emin"
    assert "Emin" in res_frame.status_label
    assert res_frame.confidence >= 0.86
    assert res_frame.turquoise_pixels >= verifier.min_turquoise_pixels
    assert res_frame.dx <= 45.0
    assert res_frame.dy <= 70.0

    # The same full detector dicts, with the frame, also verify via the
    # geometry path (frameless it would return Şüpheli — F-0787).
    res = verifier.check_vertical_alignment(
        crest_det, np_det, img_w=960, img_h=540, frame_bgr=frame
    )
    assert res.verified is True
    assert res.aligned is True
    assert res.has_evidence is True
    assert res.trusted is True
    assert res.weak is False
    assert res.status == "Emin"
    assert "Emin" in res.status_label
    assert res.confidence >= 0.86
    assert res.dx <= 45.0
    assert res.dy <= 70.0


def test_w35_02_visual_frame_crest_with_cyan_nameplate_proof_produces_high_confidence_emin() -> None:
    r"""Test 2: Synthetic frame with `\Y/` crest + cyan nameplate glyphs produces `>= 0.86` Emin in both engines."""
    verifier = CrestNameplateVerifier()
    vision = VisionEngine()

    frame = np.full((540, 960, 3), 85, dtype=np.uint8)
    crest_bbox, np_bbox = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=480,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=4,
        nameplate_dy=34,
    )

    # 2A. CrestNameplateVerifier full visual frame discovery & verification
    res_frame = verifier.verify_in_frame(frame, crest_box=crest_bbox)
    assert res_frame.verified is True
    assert res_frame.trusted is True
    assert res_frame.weak is False
    assert res_frame.status == "Emin"
    assert res_frame.confidence >= 0.86
    assert res_frame.turquoise_pixels >= verifier.min_turquoise_pixels

    # 2B. VisionEngine.verify_crest_alignment_proof on the same frame.
    # Crest position and score come from the verifier's real detection, not
    # from the caller-supplied paint coordinates.
    crests = verifier.detect_winged_crests(frame)
    assert len(crests) >= 1, "Painted crest was not detected"
    crest_det = crests[0]
    assert abs(crest_det["cx"] - 480) <= 15
    assert abs(crest_det["cy"] - 200) <= 15
    proof = vision.verify_crest_alignment_proof(
        crest_xy=(crest_det["cx"], crest_det["cy"]),
        frame_bgr=frame,
        raw_score=crest_det["score"],
        min_dy=16,
    )
    assert proof["has_proof"] is True
    assert proof["nameplate_detected"] is True
    assert proof["trusted"] is True
    assert proof["weak"] is False
    assert proof["confidence"] >= 0.86
    assert "nameplate" in proof["proof_type"]


def test_w35_03_isolated_crest_without_cyan_nameplate_evidence_produces_low_suspicious_confidence() -> None:
    r"""Test 3: Isolated `\Y/` crest without cyan nameplate below it produces `< 0.70` Şüpheli (`trusted=False`)."""
    verifier = CrestNameplateVerifier()
    vision = VisionEngine()

    frame = np.full((540, 960, 3), 85, dtype=np.uint8)
    crest_bbox, _ = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=480,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=False,
    )

    # 3A. CrestNameplateVerifier with crest only and no turquoise pixels below
    res_no_np = verifier.verify_in_frame(frame, crest_box=crest_bbox)
    assert res_no_np.verified is False
    assert res_no_np.trusted is False
    assert res_no_np.weak is True
    assert res_no_np.status == "Şüpheli"
    assert res_no_np.confidence < 0.70

    # 3B. VisionEngine.verify_crest_alignment_proof without nameplate/prompt below the crest.
    # Crest position and score come from the verifier's real detection.
    crests = verifier.detect_winged_crests(frame)
    assert len(crests) >= 1, "Painted crest was not detected"
    crest_det = crests[0]
    proof_no_np = vision.verify_crest_alignment_proof(
        crest_xy=(crest_det["cx"], crest_det["cy"]),
        frame_bgr=frame,
        raw_score=crest_det["score"],
        min_dy=16,
    )
    assert proof_no_np["has_proof"] is False
    assert proof_no_np["nameplate_detected"] is False
    assert proof_no_np["trusted"] is False
    assert proof_no_np["weak"] is True
    assert proof_no_np["confidence"] <= 0.68
    assert proof_no_np["proof_type"] == "none"


def test_w35_04_misaligned_or_inverted_crest_and_cyan_nameplate_rejected_as_suspicious() -> None:
    r"""Test 4: Horizontal offset `dx > 45`, vertical distance `dy > 70`, or nameplate above crest yields Şüpheli."""
    verifier = CrestNameplateVerifier()
    vision = VisionEngine()

    crest_box = (946, 386, 29, 28)  # center ~(960.5, 400.0)

    # Case A: Horizontal misalignment (dx = 85px > 45px)
    far_x_np = (946 + 85, 420, 60, 14)
    res_dx = verifier.check_vertical_alignment(crest_box, far_x_np)
    assert res_dx.verified is False
    assert res_dx.trusted is False
    assert res_dx.weak is True
    assert res_dx.status == "Şüpheli"
    assert res_dx.confidence < 0.70

    # Case B: Vertical misalignment (dy = 110px > 70px)
    far_y_np = (930, 386 + 110, 60, 14)
    res_dy = verifier.check_vertical_alignment(crest_box, far_y_np)
    assert res_dy.verified is False
    assert res_dy.trusted is False
    assert res_dy.weak is True
    assert res_dy.status == "Şüpheli"
    assert res_dy.confidence < 0.70

    # Case C: Inverted order (nameplate above crest, signed_dy < -6px)
    above_np = (930, 340, 60, 14)
    res_inv = verifier.check_vertical_alignment(crest_box, above_np, require_crest_above=True)
    assert res_inv.verified is False
    assert res_inv.trusted is False
    assert res_inv.weak is True
    assert res_inv.status == "Şüpheli"
    assert res_inv.confidence < 0.70

    # Case D: Visual frame where cyan nameplate is 95px to the right (outside dx <= 45px).
    # Crest position and score come from the verifier's real detection.
    frame_misaligned = np.full((540, 960, 3), 85, dtype=np.uint8)
    _paint_canonical_crest_and_cyan_nameplate(
        frame_misaligned,
        verifier,
        crest_cx=480,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=95,
        nameplate_dy=34,
    )
    crests = verifier.detect_winged_crests(frame_misaligned)
    assert len(crests) >= 1, "Painted crest was not detected"
    crest_det = crests[0]
    proof_misaligned = vision.verify_crest_alignment_proof(
        crest_xy=(crest_det["cx"], crest_det["cy"]),
        frame_bgr=frame_misaligned,
        raw_score=crest_det["score"],
        min_dy=16,
    )
    assert proof_misaligned["has_proof"] is False
    assert proof_misaligned["trusted"] is False
    assert proof_misaligned["weak"] is True
    assert proof_misaligned["confidence"] <= 0.68


def test_w35_05_batch_annotation_calibration_and_yellow_monster_nameplate_rejection() -> None:
    r"""Test 5: Batch annotation verification promotes aligned pair to `>= 0.86` Emin & rejects yellow monster nameplate."""
    verifier = CrestNameplateVerifier()

    frame = np.full((540, 960, 3), 35, dtype=np.uint8)
    # Valid aligned crest + cyan nameplate at (300, 200)
    c_bbox, np_bbox = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=300,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=2,
        nameplate_dy=32,
    )

    # Yellow monster nameplate below a second crest at (650, 200) -> must be rejected as monster yellow
    c2_bbox, _ = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=650,
        crest_cy=200,
        draw_crest=True,
        draw_cyan_nameplate=False,
    )
    yellow_np_bbox = (633, 213, 68, 14)
    # Paint bright yellow strokes (HSV ~25, 230, 240 -> BGR (20, 215, 245))
    cv2.rectangle(
        frame,
        (yellow_np_bbox[0], yellow_np_bbox[1]),
        (yellow_np_bbox[0] + yellow_np_bbox[2], yellow_np_bbox[1] + yellow_np_bbox[3]),
        (20, 215, 245),
        -1,
    )

    res_yellow = verifier.verify_in_frame(frame, crest_box=c2_bbox, nameplate_box=yellow_np_bbox)
    assert res_yellow.verified is False
    assert res_yellow.trusted is False
    assert res_yellow.weak is True
    assert res_yellow.status == "Şüpheli"
    assert res_yellow.confidence < 0.70
    assert "monster" in res_yellow.reason

    # Batch verify annotations: aligned pair at x=300 vs isolated unverified resource at x=130
    annotations = [
        {
            "class_id": CREST_CLASS_ID,
            "bbox": c_bbox,
            "confidence": 0.84,
            "source": "vision_3d_crest",
        },
        {
            "class_id": 0,
            "bbox": np_bbox,
            "confidence": 0.82,
            "source": "3d_world_nameplate",
        },
        {
            "class_id": 0,
            "bbox": (130, 250, 52, 48),
            "confidence": 0.91,
            "source": "chromatic_salient_3d_resource",
        },
    ]

    calibrated = verifier.verify_annotations(annotations, frame_bgr=frame, img_w=960, img_h=540)
    assert len(calibrated) == 3

    # First two are aligned crest + cyan nameplate -> promoted to >= 0.86 Emin
    for verified_ann in calibrated[:2]:
        assert verified_ann["crest_nameplate_verified"] is True
        assert verified_ann["verification_status"] == "Emin"
        assert verified_ann["trusted"] is True
        assert verified_ann["weak"] is False
        assert verified_ann["confidence"] >= 0.86

    # Third annotation has no crest or cyan nameplate at (130, 250) -> demoted to < 0.70 Şüpheli
    unverified_ann = calibrated[2]
    assert unverified_ann["crest_nameplate_verified"] is False
    assert unverified_ann["verification_status"] == "Şüpheli"
    assert unverified_ann["trusted"] is False
    assert unverified_ann["weak"] is True
    assert unverified_ann["confidence"] < 0.70
