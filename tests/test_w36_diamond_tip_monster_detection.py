# -*- coding: utf-8 -*-
"""W36 unit tests: Turquoise diamond-tip (`< ... >`) yellow monster nameplate (`Yellow-Maned Zait`)
detection as `monster_3d` in normal and `hflip` frames, with screen-center player exclusion."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import cv2
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.vision import PLAYER_CENTER_EXCLUSION_ZONE, is_in_player_exclusion_zone
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler


def _draw_yellow_maned_zait(
    frame: np.ndarray,
    text_x: int,
    text_y: int,
    with_diamond_tips: bool = True,
) -> Tuple[int, int, int, int]:
    """Render `< Yellow-Maned Zait >` yellow nameplate with turquoise `< ... >` diamond tips and body."""
    hsv_yellow = np.uint8([[[25, 150, 180]]])
    bgr_yellow_arr = cv2.cvtColor(hsv_yellow, cv2.COLOR_HSV2BGR)[0][0]
    bgr_yellow = (int(bgr_yellow_arr[0]), int(bgr_yellow_arr[1]), int(bgr_yellow_arr[2]))

    label_text = "Yellow-Maned Zait"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(
        frame,
        label_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        bgr_yellow,
        2,
    )

    if with_diamond_tips:
        # Turquoise / cyan diamond bracket tips `<` and `>` (Hue: 90, Sat: 180, Val: 200)
        hsv_cyan = np.uint8([[[90, 180, 200]]])
        bgr_cyan_arr = cv2.cvtColor(hsv_cyan, cv2.COLOR_HSV2BGR)[0][0]
        bgr_cyan = (int(bgr_cyan_arr[0]), int(bgr_cyan_arr[1]), int(bgr_cyan_arr[2]))

        # Left diamond tip `<` and right diamond tip `>` flanking the yellow nameplate
        cv2.rectangle(frame, (text_x - 18, text_y - th - 2), (text_x - 4, text_y + 2), bgr_cyan, -1)
        cv2.rectangle(frame, (text_x + tw + 4, text_y - th - 2), (text_x + tw + 18, text_y + 2), bgr_cyan, -1)

    # Monster 3D body below the nameplate
    cv2.rectangle(frame, (text_x - 10, text_y + 10), (text_x + tw + 15, text_y + 95), (40, 45, 80), -1)
    return (text_x - 18, text_y - th - 2, tw + 36, th + 97)


def test_diamond_tip_yellow_maned_zait_normal_frame_detected_as_monster_3d() -> None:
    """1. Normal frame with turquoise `< ... >` diamond-tipped `Yellow-Maned Zait` is detected as `monster_3d`."""
    labeler = YOLOAutoLabeler()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    _draw_yellow_maned_zait(frame, text_x=540, text_y=350, with_diamond_tips=True)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    monsters = labeler._scan_3d_monsters(frame_bgr=frame, hsv=hsv, existing_positive_boxes=[], gray=gray)
    m3d = [m for m in monsters if m["class_id"] == 10 and m["material_name"] == "monster_3d"]
    assert len(m3d) == 1

    hit = m3d[0]
    assert hit["class_id"] == 10
    assert hit["class_name"] == "monster_3d"
    assert hit["material_name"] == "monster_3d"
    assert hit["source"] == "3d_monster_scanner"
    assert hit["confidence"] == 0.95
    assert hit["is_negative"] is True
    assert hit["bbox"][2] >= 150
    assert hit["bbox"][3] >= 90


def test_diamond_tip_yellow_maned_zait_hflip_frame_detected_as_monster_3d() -> None:
    """2. Horizontally flipped (`hflip`) frame with `< Yellow-Maned Zait >` is captured as `monster_3d`."""
    labeler = YOLOAutoLabeler()
    frame_normal = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _draw_yellow_maned_zait(frame_normal, text_x=540, text_y=350, with_diamond_tips=True)

    frame_hflip = cv2.flip(frame_normal, 1)

    hsv_norm = cv2.cvtColor(frame_normal, cv2.COLOR_BGR2HSV)
    gray_norm = cv2.cvtColor(frame_normal, cv2.COLOR_BGR2GRAY)
    norm_hits = [
        m
        for m in labeler._scan_3d_monsters(frame_normal, hsv_norm, [], gray_norm)
        if m["class_id"] == 10 and m["material_name"] == "monster_3d"
    ]
    assert len(norm_hits) == 1

    hsv_flip = cv2.cvtColor(frame_hflip, cv2.COLOR_BGR2HSV)
    gray_flip = cv2.cvtColor(frame_hflip, cv2.COLOR_BGR2GRAY)
    flip_hits = [
        m
        for m in labeler._scan_3d_monsters(frame_hflip, hsv_flip, [], gray_flip)
        if m["class_id"] == 10 and m["material_name"] == "monster_3d"
    ]
    assert len(flip_hits) == 1

    hit_norm = norm_hits[0]
    hit_flip = flip_hits[0]
    assert hit_flip["class_id"] == 10
    assert hit_flip["class_name"] == "monster_3d"
    assert hit_flip["material_name"] == "monster_3d"
    assert hit_flip["confidence"] == 0.95
    # Horizontal center in hflip mirrors normal detection across the frame width
    assert abs((hit_norm["cx"] + hit_flip["cx"]) - 1.0) <= 0.01
    assert abs(hit_norm["cy"] - hit_flip["cy"]) <= 0.005
    assert hit_norm["bbox"][2] == hit_flip["bbox"][2]
    assert hit_norm["bbox"][3] == hit_flip["bbox"][3]


def test_screen_center_player_excluded_in_normal_and_hflip_frames() -> None:
    """3. Any `< Yellow-Maned Zait >` or HP bar inside the screen-center player zone is excluded in normal & hflip."""
    labeler = YOLOAutoLabeler()
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Place `< Yellow-Maned Zait >` directly inside the screen-center player exclusion zone (x: 0.44..0.56, y: 0.42..0.58)
    center_x = int(w * 0.465)
    center_y = int(h * 0.50)
    _draw_yellow_maned_zait(frame, text_x=center_x, text_y=center_y, with_diamond_tips=True)
    # Also draw a red HP bar in the player center zone
    cv2.rectangle(frame, (int(w * 0.47), int(h * 0.47)), (int(w * 0.53), int(h * 0.48)), (30, 30, 225), -1)

    assert is_in_player_exclusion_zone(w // 2, h // 2, w, h, exclusion_zone=PLAYER_CENTER_EXCLUSION_ZONE) is True

    for variant in (frame, cv2.flip(frame, 1)):
        hsv = cv2.cvtColor(variant, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
        anns = labeler._scan_3d_monsters(frame_bgr=variant, hsv=hsv, existing_positive_boxes=[], gray=gray)
        m3d = [a for a in anns if a["class_id"] == 10]
        assert len(m3d) == 0


def test_simultaneous_valid_zait_and_center_player_only_captures_monster_in_normal_and_hflip() -> None:
    """4. With both an off-center `< Yellow-Maned Zait >` and a center player present, only the monster is captured."""
    labeler = YOLOAutoLabeler()
    h, w = 1080, 1920
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Valid off-center monster in the 3D viewport
    _draw_yellow_maned_zait(frame, text_x=520, text_y=350, with_diamond_tips=True)

    # Center player nameplate + red bar inside the player self-exclusion zone (x: 0.44..0.56, y: 0.42..0.58)
    _draw_yellow_maned_zait(frame, text_x=int(w * 0.465), text_y=int(h * 0.50), with_diamond_tips=True)
    cv2.rectangle(frame, (int(w * 0.47), int(h * 0.47)), (int(w * 0.53), int(h * 0.48)), (30, 30, 225), -1)

    for is_flipped, img in ((False, frame), (True, cv2.flip(frame, 1))):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m3d = [
            a
            for a in labeler._scan_3d_monsters(img, hsv, [], gray)
            if a["class_id"] == 10 and a["material_name"] == "monster_3d"
        ]
        assert len(m3d) == 1
        hit = m3d[0]
        assert hit["confidence"] == 0.95
        assert not (0.44 <= hit["cx"] <= 0.56 and 0.42 <= hit["cy"] <= 0.58)
        bx, by, bw, bh = hit["bbox"]
        assert not is_in_player_exclusion_zone(
            bx + bw // 2,
            by + bh // 2,
            w,
            h,
            bbox=hit["bbox"],
            exclusion_zone=PLAYER_CENTER_EXCLUSION_ZONE,
        )
        if not is_flipped:
            assert hit["cx"] < 0.44
        else:
            assert hit["cx"] > 0.56


def test_real_pilgrims_ridge_normal_and_hflip_diamond_tip_monster_and_center_exclusion() -> None:
    """5. Real Pilgrim's Ridge frame (and its hflip) captures the diamond-tip yellow monster and excludes center player."""
    labeler = YOLOAutoLabeler()
    repo_root = Path(__file__).resolve().parent.parent
    normal_path = repo_root / "dataset" / "images" / "train" / "boot_478467_005_pilgrims_ridge_1.jpg"
    hflip_aug_path = (
        repo_root / "dataset" / "images" / "train" / "boot_478467_005_pilgrims_ridge_1_aug02_dusk_hflip.jpg"
    )

    if normal_path.is_file():
        img_normal = cv2.imread(str(normal_path))
    else:
        img_normal = np.zeros((1124, 2000, 3), dtype=np.uint8)
        _draw_yellow_maned_zait(img_normal, text_x=650, text_y=315, with_diamond_tips=True)

    assert img_normal is not None and img_normal.size > 0
    img_hflip = cv2.imread(str(hflip_aug_path)) if hflip_aug_path.is_file() else cv2.flip(img_normal, 1)
    assert img_hflip is not None and img_hflip.size > 0

    for variant_name, img in (("normal", img_normal), ("hflip_direct", cv2.flip(img_normal, 1)), ("hflip_aug", img_hflip)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        anns = labeler._scan_3d_monsters(img, hsv, [], gray)
        m3d = [a for a in anns if a["class_id"] == 10 and a["material_name"] == "monster_3d"]
        diamond_m3d = [a for a in m3d if a["confidence"] >= 0.95]
        assert len(diamond_m3d) >= 1, f"Expected diamond-tip monster_3d in {variant_name}"

        for a in m3d:
            assert not (0.44 <= a["cx"] <= 0.56 and 0.42 <= a["cy"] <= 0.58), (
                f"Center player falsely detected as monster_3d in {variant_name}: {a}"
            )
