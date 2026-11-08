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
    r"""Paint canonical 28x29 white `\Y/` crest mask and turquoise/cyan glyph nameplate onto `frame_bgr`."""
    mask = verifier._crest_template_mask
    th, tw = mask.shape[:2]
    cx1 = int(crest_cx - tw // 2)
    cy1 = int(crest_cy - th // 2)

    if draw_crest:
        roi = frame_bgr[cy1 : cy1 + th, cx1 : cx1 + tw]
        roi[mask > 0] = (250, 250, 250)  # Bright white in BGR

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
    r"""Test 1: Vertically aligned `\Y/` crest and cyan nameplate (`dx <= 45, dy <= 70`) yields `>= 0.86` Emin."""
    verifier = CrestNameplateVerifier()

    crest_box = {"class_id": CREST_CLASS_ID, "bbox": (946, 386, 29, 28), "confidence": 0.90}
    nameplate_box = {"class_id": 0, "bbox": (930, 424, 68, 14), "confidence": 0.88}

    res = verifier.check_vertical_alignment(crest_box, nameplate_box, img_w=1920, img_h=1080)
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

    frame = np.full((1080, 1920, 3), 85, dtype=np.uint8)
    crest_bbox, np_bbox = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=960,
        crest_cy=400,
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

    # 2B. VisionEngine.verify_crest_alignment_proof on the same frame
    proof = vision.verify_crest_alignment_proof(
        crest_xy=(960, 400),
        frame_bgr=frame,
        raw_score=0.90,
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

    frame = np.full((1080, 1920, 3), 85, dtype=np.uint8)
    crest_bbox, _ = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=960,
        crest_cy=400,
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

    # 3B. VisionEngine.verify_crest_alignment_proof without nameplate/prompt below the crest
    proof_no_np = vision.verify_crest_alignment_proof(
        crest_xy=(960, 400),
        frame_bgr=frame,
        raw_score=0.92,
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

    # Case D: Visual frame where cyan nameplate is 95px to the right (outside dx <= 45px)
    frame_misaligned = np.full((1080, 1920, 3), 85, dtype=np.uint8)
    _paint_canonical_crest_and_cyan_nameplate(
        frame_misaligned,
        verifier,
        crest_cx=960,
        crest_cy=400,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=95,
        nameplate_dy=34,
    )
    proof_misaligned = vision.verify_crest_alignment_proof(
        crest_xy=(960, 400),
        frame_bgr=frame_misaligned,
        raw_score=0.91,
        min_dy=16,
    )
    assert proof_misaligned["has_proof"] is False
    assert proof_misaligned["trusted"] is False
    assert proof_misaligned["weak"] is True
    assert proof_misaligned["confidence"] <= 0.68


def test_w35_05_batch_annotation_calibration_and_yellow_monster_nameplate_rejection() -> None:
    r"""Test 5: Batch annotation verification promotes aligned pair to `>= 0.86` Emin & rejects yellow monster nameplate."""
    verifier = CrestNameplateVerifier()

    frame = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    # Valid aligned crest + cyan nameplate at (600, 400)
    c_bbox, np_bbox = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=600,
        crest_cy=400,
        draw_crest=True,
        draw_cyan_nameplate=True,
        nameplate_dx=2,
        nameplate_dy=32,
    )

    # Yellow monster nameplate below a second crest at (1300, 400) -> must be rejected as monster yellow
    c2_bbox, _ = _paint_canonical_crest_and_cyan_nameplate(
        frame,
        verifier,
        crest_cx=1300,
        crest_cy=400,
        draw_crest=True,
        draw_cyan_nameplate=False,
    )
    yellow_np_bbox = (1266, 426, 68, 14)
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

    # Batch verify annotations: aligned pair at x=600 vs isolated unverified resource at x=300
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
            "bbox": (260, 500, 52, 48),
            "confidence": 0.91,
            "source": "chromatic_salient_3d_resource",
        },
    ]

    calibrated = verifier.verify_annotations(annotations, frame_bgr=frame, img_w=1920, img_h=1080)
    assert len(calibrated) == 3

    # First two are aligned crest + cyan nameplate -> promoted to >= 0.86 Emin
    for verified_ann in calibrated[:2]:
        assert verified_ann["crest_nameplate_verified"] is True
        assert verified_ann["verification_status"] == "Emin"
        assert verified_ann["trusted"] is True
        assert verified_ann["weak"] is False
        assert verified_ann["confidence"] >= 0.86

    # Third annotation has no crest or cyan nameplate at (260, 500) -> demoted to < 0.70 Şüpheli
    unverified_ann = calibrated[2]
    assert unverified_ann["crest_nameplate_verified"] is False
    assert unverified_ann["verification_status"] == "Şüpheli"
    assert unverified_ann["trusted"] is False
    assert unverified_ann["weak"] is True
    assert unverified_ann["confidence"] < 0.70
