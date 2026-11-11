# -*- coding: utf-8 -*-
"""W64 unit tests: PlayerSelfExclusionGuard center-screen (0.50, 0.50) false box exclusion and [F Gather] / \\Y/ preservation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.player_self_exclusion_guard import (
    DEFAULT_CX_MAX,
    DEFAULT_CX_MIN,
    DEFAULT_CY_MAX,
    DEFAULT_CY_MIN,
    ExclusionDecision,
    ExclusionGuardConfig,
    PlayerSelfExclusionGuard,
)


def test_excludes_unverified_false_center_boxes_at_050_050() -> None:
    """1. Unverified 3D material and monster boxes at screen center (0.50, 0.50) are excluded."""
    guard = PlayerSelfExclusionGuard()

    assert guard.bounds == (DEFAULT_CX_MIN, DEFAULT_CX_MAX, DEFAULT_CY_MIN, DEFAULT_CY_MAX)
    assert guard.is_in_self_exclusion_zone(0.50, 0.50) is True
    assert guard.should_exclude(0.50, 0.50) is True

    decision_mat: ExclusionDecision = guard.evaluate(0.50, 0.50, category="3d_material")
    assert decision_mat.excluded is True
    assert decision_mat.allowed is False
    assert decision_mat.in_exclusion_zone is True
    assert decision_mat.has_gather_evidence is False
    assert decision_mat.has_crest_evidence is False
    assert decision_mat.normalized_cx == 0.50
    assert decision_mat.normalized_cy == 0.50
    assert decision_mat.reason == "excluded_player_mount_center_zone_without_evidence"

    false_monster_box = {
        "class_id": 10,
        "class_name": "monster_3d",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.14,
        "h": 0.22,
    }
    decision_mob = guard.evaluate_candidate(false_monster_box)
    assert decision_mob.excluded is True
    assert decision_mob.allowed is False
    assert decision_mob.reason == "excluded_player_mount_center_zone_without_evidence"


def test_preserves_center_box_with_f_gather_prompt_evidence() -> None:
    """2. Center boxes at (0.50, 0.50) with verified [F Gather] prompt evidence are preserved."""
    guard = PlayerSelfExclusionGuard()

    decision_ocr = guard.evaluate(0.50, 0.50, ocr_text="[F Gather]")
    assert decision_ocr.excluded is False
    assert decision_ocr.allowed is True
    assert decision_ocr.in_exclusion_zone is True
    assert decision_ocr.has_gather_evidence is True
    assert decision_ocr.has_crest_evidence is False
    assert decision_ocr.reason == "allowed_by_f_gather_evidence"

    candidate_with_prompt = {
        "class_id": 0,
        "class_name": "herb_aria",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.16,
        "h": 0.18,
        "prompt_text": "[F Gather] Aria",
        "has_f_gather": True,
        "gather_confidence": 0.91,
    }
    decision_cand = guard.evaluate_candidate(candidate_with_prompt)
    assert decision_cand.excluded is False
    assert decision_cand.allowed is True
    assert decision_cand.has_gather_evidence is True
    assert decision_cand.reason == "allowed_by_f_gather_evidence"

    # Sub-threshold gather confidence without prompt text must still be excluded
    decision_low_conf = guard.evaluate(
        0.50,
        0.50,
        f_gather_detected=True,
        gather_confidence=0.25,
    )
    assert decision_low_conf.excluded is True
    assert decision_low_conf.allowed is False


def test_preserves_center_box_with_winged_y_crest_arma_evidence() -> None:
    """3. Center boxes at (0.50, 0.50) with verified \\Y/ crest or arma evidence are preserved."""
    guard = PlayerSelfExclusionGuard()

    decision_crest = guard.evaluate(0.50, 0.50, ocr_text=r"\Y/")
    assert decision_crest.excluded is False
    assert decision_crest.allowed is True
    assert decision_crest.in_exclusion_zone is True
    assert decision_crest.has_crest_evidence is True
    assert decision_crest.has_gather_evidence is False
    assert decision_crest.reason == "allowed_by_crest_evidence"

    candidate_with_arma = {
        "class_id": 1,
        "class_name": "herb_mela",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.15,
        "h": 0.17,
        "crest_symbol": r"\Y/",
        "has_arma": True,
        "crest_confidence": 0.88,
    }
    decision_cand = guard.evaluate_candidate(candidate_with_arma)
    assert decision_cand.excluded is False
    assert decision_cand.allowed is True
    assert decision_cand.has_crest_evidence is True
    assert decision_cand.reason == "allowed_by_crest_evidence"

    decision_both = guard.evaluate(
        0.50,
        0.50,
        ocr_text=r"[F Gather] \Y/",
        f_gather_detected=True,
        crest_detected=True,
    )
    assert decision_both.excluded is False
    assert decision_both.allowed is True
    assert decision_both.has_gather_evidence is True
    assert decision_both.has_crest_evidence is True
    assert decision_both.reason == "allowed_by_f_gather_and_crest_evidence"


def test_filter_candidates_removes_false_center_boxes_and_keeps_verified_or_peripheral() -> None:
    """4. filter_candidates drops false (0.50, 0.50) boxes while keeping [F Gather], \\Y/, and outside boxes."""
    guard = PlayerSelfExclusionGuard()

    false_player_cape = {
        "id": "false_cape",
        "class_name": "ore_orichalcum",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.12,
        "h": 0.16,
    }
    false_player_mount = {
        "id": "false_mount",
        "class_name": "monster_3d",
        "cx": 0.49,
        "cy": 0.52,
        "w": 0.18,
        "h": 0.24,
    }
    verified_gather_center = {
        "id": "true_f_gather",
        "class_name": "herb_aria",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.14,
        "h": 0.15,
        "ui_prompt": "[F Gather]",
    }
    verified_crest_center = {
        "id": "true_y_crest",
        "class_name": "herb_mela",
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.13,
        "h": 0.15,
        "evidence_tags": [r"\Y/", "arma"],
    }
    peripheral_material = {
        "id": "outside_ore",
        "class_name": "ore_orichalcum",
        "cx": 0.22,
        "cy": 0.30,
        "w": 0.11,
        "h": 0.13,
    }

    kept = guard.filter_candidates(
        [
            false_player_cape,
            false_player_mount,
            verified_gather_center,
            verified_crest_center,
            peripheral_material,
        ]
    )

    kept_ids = [item["id"] for item in kept]
    assert kept_ids == ["true_f_gather", "true_y_crest", "outside_ore"]


def test_filter_yolo_lines_and_pixel_normalization_with_diagnostics() -> None:
    """5. filter_yolo_lines, pixel-center normalization (960, 540 on 1920x1080 -> 0.50, 0.50), and diagnostics work."""
    cfg = ExclusionGuardConfig(strict_3d_only=True)
    guard = PlayerSelfExclusionGuard(config=cfg)

    yolo_lines = [
        "0 0.500000 0.500000 0.120000 0.160000",  # Center (0.50, 0.50) -> excluded when unverified
        "3 0.250000 0.350000 0.100000 0.120000",  # Outside exclusion zone -> kept
        "10 0.500000 0.500000 0.150000 0.200000",  # Center monster -> excluded when unverified
    ]

    filtered_unverified = guard.filter_yolo_lines(yolo_lines)
    assert filtered_unverified == ["3 0.250000 0.350000 0.100000 0.120000"]

    filtered_with_gather = guard.filter_yolo_lines(yolo_lines, f_gather_detected=True)
    assert len(filtered_with_gather) == 3

    filtered_with_crest = guard.filter_yolo_lines(yolo_lines, crest_detected=True)
    assert len(filtered_with_crest) == 3

    # Pixel coordinates (960, 540) on 1920x1080 normalize to (0.50, 0.50)
    pixel_decision = guard.evaluate(
        960.0,
        540.0,
        category="3d_material",
        frame_width=1920,
        frame_height=1080,
    )
    assert pixel_decision.normalized_cx == 0.50
    assert pixel_decision.normalized_cy == 0.50
    assert pixel_decision.excluded is True

    diagnostics = guard.get_diagnostics()
    assert diagnostics["ready"] is True
    assert diagnostics["total_evaluated"] == 10
    assert diagnostics["total_excluded"] == 3
    assert diagnostics["total_overridden_by_evidence"] == 4


@pytest.mark.parametrize(
    "cx,cy",
    [(0.42, 0.50), (0.58, 0.50), (0.50, 0.40), (0.50, 0.62), (0.42, 0.40), (0.58, 0.62)],
)
def test_exclusion_zone_edge_points_are_inside(cx: float, cy: float) -> None:
    """6. F-1591: probe the zone edges (0.42..0.58 x, 0.40..0.62 y), not only (0.50, 0.50).

    Bounds are inclusive: a point exactly on an edge is inside the zone.
    Eskiden 6 noktalık tek döngüydü: ilk hatalı kenarda kalan kenarlar hiç
    denetlenmezdi. Parametreli hali her kenar noktasını ayrı vaka yapar.
    """
    guard = PlayerSelfExclusionGuard()
    assert guard.is_in_self_exclusion_zone(cx, cy) is True, f"Edge point {(cx, cy)} must be inside"
    decision = guard.evaluate(cx, cy, category="3d_material")
    assert decision.in_exclusion_zone is True
    assert decision.excluded is True
    assert decision.reason == "excluded_player_mount_center_zone_without_evidence"


@pytest.mark.parametrize(
    "cx,cy",
    [(0.41, 0.50), (0.59, 0.50), (0.50, 0.39), (0.50, 0.63), (0.41, 0.39), (0.59, 0.63)],
)
def test_exclusion_zone_outside_points_are_outside(cx: float, cy: float) -> None:
    """6b. F-1591: one step outside each edge -> outside, never excluded."""
    guard = PlayerSelfExclusionGuard()
    assert guard.is_in_self_exclusion_zone(cx, cy) is False, f"Point {(cx, cy)} must be outside"
    decision = guard.evaluate(cx, cy, category="3d_material")
    assert decision.in_exclusion_zone is False
    assert decision.excluded is False
    assert decision.allowed is True
    assert decision.reason == "outside_self_exclusion_zone"


@pytest.mark.parametrize(
    "non_3d_category",
    ["skill_ui_prompt", "player_hp_bar", "skill_bar", "f_gather_prompt"],
)
def test_strict_3d_only_exempts_non_3d_categories_at_center(non_3d_category: str) -> None:
    """7. F-1590: strict_3d_only=True must actually run its discriminating branch.

    A center candidate whose category is NOT 3D (e.g. skill_ui_prompt,
    player_hp_bar) is exempt from exclusion under strict mode, while the
    same point/category is excluded under the default config.
    Eskiden 4 kategorilik tek döngüydü: ilk hatalı kategoride kalanlar hiç
    denetlenmezdi. Parametreli hali her kategoriyi ayrı vaka yapar.
    """
    strict_guard = PlayerSelfExclusionGuard(config=ExclusionGuardConfig(strict_3d_only=True))
    # [O5 merge] The product default is strict_3d_only=True since D49/F-0983
    # ("False default closes the category gate"), so a bare
    # PlayerSelfExclusionGuard() is already strict. The non-strict control
    # case below must opt out explicitly; otherwise it asserts the pre-D49
    # default that F-0983 removed.
    default_guard = PlayerSelfExclusionGuard(config=ExclusionGuardConfig(strict_3d_only=False))

    strict_decision = strict_guard.evaluate(0.50, 0.50, category=non_3d_category)
    assert strict_decision.in_exclusion_zone is True
    assert strict_decision.excluded is False
    assert strict_decision.allowed is True
    assert strict_decision.reason == "non_3d_category_exempt"

    default_decision = default_guard.evaluate(0.50, 0.50, category=non_3d_category)
    assert default_decision.excluded is True
    assert default_decision.reason == "excluded_player_mount_center_zone_without_evidence"


def test_strict_3d_only_candidate_and_3d_control() -> None:
    """7b. Same strict branch via evaluate_candidate, plus the 3D control case."""
    strict_guard = PlayerSelfExclusionGuard(config=ExclusionGuardConfig(strict_3d_only=True))

    # Same branch via evaluate_candidate (category derived from class_name).
    strict_candidate = strict_guard.evaluate_candidate(
        {"class_id": 13, "class_name": "skill_ui_prompt", "cx": 0.50, "cy": 0.50, "w": 0.10, "h": 0.10}
    )
    assert strict_candidate.excluded is False
    assert strict_candidate.reason == "non_3d_category_exempt"

    # Control: a genuine 3D category is still excluded under strict mode.
    strict_3d = strict_guard.evaluate(0.50, 0.50, category="3d_material")
    assert strict_3d.excluded is True
    assert strict_3d.reason == "excluded_player_mount_center_zone_without_evidence"
    strict_monster = strict_guard.evaluate(0.50, 0.50, category="monster_3d")
    assert strict_monster.excluded is True


def test_filter_candidates_skips_unparsable_candidate_without_killing_batch() -> None:
    """8. S-2503: one bad candidate (non-numeric / missing center) must not
    crash filter_candidates; it is skipped + counted, the rest still filter."""
    guard = PlayerSelfExclusionGuard()

    bad_cx = {"id": "bad_cx", "class_name": "ore_orichalcum", "cx": "abc", "cy": 0.50}
    bad_cy_none = {"id": "bad_cy_none", "class_name": "ore_orichalcum", "cx": 0.50, "cy": None}
    no_center = {"id": "no_center", "class_name": "ore_orichalcum"}
    bad_tuple = ("not_a_number", 0.50)
    false_center = {"id": "false_center", "class_name": "ore_orichalcum", "cx": 0.50, "cy": 0.50}
    outside = {"id": "outside", "class_name": "ore_orichalcum", "cx": 0.10, "cy": 0.10}
    verified_center = {
        "id": "verified_center",
        "class_name": "herb_aria",
        "cx": 0.50,
        "cy": 0.50,
        "ui_prompt": "[F Gather]",
    }

    kept = guard.filter_candidates(
        [bad_cx, bad_cy_none, no_center, bad_tuple, false_center, outside, verified_center]
    )
    assert [item["id"] for item in kept] == ["outside", "verified_center"]

    diagnostics = guard.get_diagnostics()
    assert diagnostics["candidates_skipped_unparsable"] == 4

    # Single-candidate API still surfaces the error (only the batch swallows it).
    with pytest.raises(ValueError):
        guard.evaluate_candidate({"cx": "abc", "cy": 0.50})
    with pytest.raises(ValueError):
        guard.evaluate_candidate({"cx": 0.50, "cy": None})


def test_filter_yolo_lines_rejects_non_numeric_or_invalid_wh() -> None:
    """9. S-2504: YOLO lines with non-numeric / non-finite / non-positive w/h
    are skipped as unparsable instead of passing as 'valid'."""
    guard = PlayerSelfExclusionGuard()

    yolo_lines = [
        "0 0.500000 0.500000 0.120000 0.160000",  # Valid center -> excluded
        "3 0.250000 0.350000 0.100000 0.120000",  # Valid outside -> kept
        "3 0.250000 0.350000 abc 0.120000",  # Non-numeric w -> skipped
        "3 0.250000 0.350000 0.100000 nan",  # Non-finite h -> skipped
        "3 0.250000 0.350000 0 0.120000",  # Zero w -> skipped
        "3 0.250000 0.350000 0.100000 -0.050000",  # Negative h -> skipped
    ]

    kept = guard.filter_yolo_lines(yolo_lines)
    assert kept == ["3 0.250000 0.350000 0.100000 0.120000"]

    diagnostics = guard.get_diagnostics()
    assert diagnostics["yolo_lines_skipped_unparsable"] == 4
