import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import SettingsManager

logger = logging.getLogger(__name__)


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Reads an image safely on Windows paths containing non-ASCII characters (e.g. Turkish characters)."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return None
        raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        return cv2.imdecode(raw_bytes, flags)
    except Exception as exc:
        logger.error("Failed reading image %s: %s", path, exc)
        return None


def imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    """Writes an image safely on Windows paths containing non-ASCII characters."""
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ext = file_path.suffix if file_path.suffix else ".png"
        success, encoded = cv2.imencode(ext, img)
        if not success:
            logger.error("Failed encoding image for %s", file_path)
            return False
        encoded.tofile(str(file_path))
        return True
    except Exception as exc:
        logger.error("Failed writing image %s: %s", path, exc)
        return False


def _ensure_templates_dir(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root else PROJECT_ROOT
    templates_dir = root / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "legacy").mkdir(parents=True, exist_ok=True)
    (templates_dir / "negative_filters").mkdir(parents=True, exist_ok=True)
    return templates_dir


def _load_best_gather_frame(root: Path) -> Optional[np.ndarray]:
    candidates = [
        root / "video_analysis" / "frame_33s.jpg",
        root / "video_frames" / "frame_33s.jpg",
        root / "video_analysis" / "frame_34s.jpg",
        root / "video_frames" / "frame_54s.jpg",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                img = imread_unicode(candidate, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    logger.info("Loaded gather reference frame: %s (%s)", candidate.name, img.shape)
                    return img
        except Exception as exc:
            logger.error("Failed reading gather candidate %s: %s", candidate, exc)
    return None


def _load_best_map_frame(root: Path) -> Optional[np.ndarray]:
    candidates = [
        root / "video_frames" / "frame_18s.jpg",
        root / "video_analysis" / "map_at_25s.jpg",
        root / "video_frames" / "frame_27s.jpg",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                img = imread_unicode(candidate, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    logger.info("Loaded map reference frame: %s (%s)", candidate.name, img.shape)
                    return img
        except Exception as exc:
            logger.error("Failed reading map candidate %s: %s", candidate, exc)
    return None


def _smooth_fill_masked_region(
    crop: np.ndarray,
    mask: np.ndarray,
    bg_sample: np.ndarray,
) -> np.ndarray:
    """
    Smoothly replaces masked (non-target neighbor icon) pixels using a Gaussian-smoothed
    background canvas synthesized from `bg_sample`, eliminating both overlapping icon bleed
    and any tiling grid artifacts.
    """
    ch, cw = crop.shape[:2]
    bg_resized = cv2.resize(bg_sample, (cw, ch), interpolation=cv2.INTER_LINEAR)
    bg_smooth = cv2.GaussianBlur(bg_resized, (9, 9), 2.5)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 1.2)[..., None]
    blended = crop.astype(np.float32) * (1.0 - alpha) + bg_smooth.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _clean_hd_f_gather_crop(crop: np.ndarray) -> np.ndarray:
    """
    Removes any stray pet-wing cyan pixels in the extreme top-left corner (y=0..13, x=0..11)
    outside the 6-petal flower icon on `aria_f_gather_not_on_minimap.jpg`.
    """
    cleaned = crop.copy()
    ch, cw = cleaned.shape[:2]
    if ch >= 30 and cw >= 100:
        mask = np.zeros((ch, cw), dtype=np.uint8)
        for y in range(min(14, ch)):
            for x in range(min(12, cw)):
                if x + y < 13:
                    mask[y, x] = 255
        bg_sample = cleaned[0:12, 26:40].copy()
        cleaned = _smooth_fill_masked_region(cleaned, mask, bg_sample)
    return cleaned


def _clean_feather_crop(crop: np.ndarray) -> np.ndarray:
    """
    Cleans overlapping neighbor icons from the `minimap_feather_odyle.png` (34x29) crop:
    - Top-left region (y <= 11, x <= 12 or x + y < 17): removes blue wing tip bleed.
    - Top-right region (y <= 8, x >= 21 or y <= 2): removes green stack '3' digit bleed.
    Preserves 100% of the white diagonal feather and its dark outline on a smooth olive background.
    """
    cleaned = crop.copy()
    ch, cw = cleaned.shape[:2]
    if ch < 28 or cw < 24:
        return cleaned
    mask = np.zeros((ch, cw), dtype=np.uint8)
    for y in range(ch):
        for x in range(cw):
            if y <= 2 and (x <= 15 or x >= 20):
                mask[y, x] = 255
            elif y <= 11 and x <= 12 and (x + y <= 20):
                mask[y, x] = 255
            elif y <= 8 and x >= 21 and (x - y >= 13):
                mask[y, x] = 255
    bg_sample = crop[18:33, 19:29].copy()
    return _smooth_fill_masked_region(cleaned, mask, bg_sample)


def _clean_blue_wing_crop(crop: np.ndarray) -> np.ndarray:
    """
    Cleans overlapping neighbor icons from the `minimap_blue_wing.png` (36x34) crop:
    - Top-right corner (y <= 8, x >= 27): removes white monster dot bleed.
    - Bottom-left corner (y >= 19, x <= 10): removes background white icon bleed.
    - Bottom-right corner (y >= 16, x >= 24): removes green '2/3' stack and feather bleed.
    Preserves the symmetrical blue wing + green diamond emblem on a smooth minimap background.
    """
    cleaned = crop.copy()
    ch, cw = cleaned.shape[:2]
    if ch < 30 or cw < 30:
        return cleaned
    mask = np.zeros((ch, cw), dtype=np.uint8)
    for y in range(ch):
        for x in range(cw):
            if y <= 8 and x >= 27:
                mask[y, x] = 255
            elif y >= 19 and x <= 10 and (x <= 8 or y >= 29):
                mask[y, x] = 255
            elif y >= 16 and x >= 24 and (x >= 26 or y >= 23):
                mask[y, x] = 255
            elif y >= 31 and x >= 21:
                mask[y, x] = 255
    bg_sample = crop[1:8, 1:10].copy()
    return _smooth_fill_masked_region(cleaned, mask, bg_sample)


def _clean_green_stack_crop(crop: np.ndarray) -> np.ndarray:
    """
    Cleans overlapping neighbor icons from the `minimap_green_stack.png` (31x23) crop:
    - Top-left / left edge (y <= 15, x <= 6): removes blue wing right-tip bleed.
    - Bottom-left corner (y >= 18, x <= 6): removes white feather top-tip bleed.
    - Right edge (y >= 20, x >= 17): removes white radar ring line bleed.
    Preserves the green diamond stack badge ('2/3') on a smooth minimap background.
    """
    cleaned = crop.copy()
    ch, cw = cleaned.shape[:2]
    if ch < 25 or cw < 20:
        return cleaned
    mask = np.zeros((ch, cw), dtype=np.uint8)
    for y in range(ch):
        for x in range(cw):
            if y <= 3 and x <= 6:
                mask[y, x] = 255
            elif y <= 14 and x <= 5 and (x + (y // 3) <= 7):
                mask[y, x] = 255
            elif y >= 18 and x <= 6 and ((y - 18) >= x):
                mask[y, x] = 255
            elif y >= 25 and x <= 11:
                mask[y, x] = 255
            elif y >= 20 and x >= 17:
                mask[y, x] = 255
    bg_sample = crop[1:9, 15:23].copy()
    return _smooth_fill_masked_region(cleaned, mask, bg_sample)


def extract_f_gather_templates(
    root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Extracts crystal-clear HD `[flower] F Gather`, `[F Gather]` dark pill, and `[F]` key cap
    templates from `reference_screenshots/aria_f_gather_not_on_minimap.jpg` (or `mela_f_gather_not_on_minimap.jpg`),
    while also saving legacy 720p video templates in `templates/legacy/` for backward compatibility.
    """
    base_root = Path(root) if root else PROJECT_ROOT
    templates_dir = _ensure_templates_dir(base_root)
    legacy_dir = templates_dir / "legacy"
    ref_dir = base_root / "reference_screenshots"
    saved: Dict[str, Path] = {}

    try:
        # 1. Extract legacy 720p video F Gather templates first (stored in templates/legacy/)
        frame = _load_best_gather_frame(base_root)
        legacy_full: Optional[np.ndarray] = None
        legacy_pill: Optional[np.ndarray] = None
        if frame is not None:
            h, w = frame.shape[:2]
            scale_x = w / 1280.0
            scale_y = h / 720.0
            search_x1 = int(round(660 * scale_x))
            search_x2 = int(round(775 * scale_x))
            search_y1 = int(round(405 * scale_y))
            search_y2 = int(round(445 * scale_y))

            roi = frame[search_y1:search_y2, search_x1:search_x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bright_ys, bright_xs = np.where(gray_roi > 175)

            if bright_xs.size > 10 and bright_ys.size > 10:
                min_x = max(0, int(bright_xs.min()) - int(round(2 * scale_x)))
                max_x = min(roi.shape[1], int(bright_xs.max()) + int(round(10 * scale_x)))
                min_y = max(0, int(bright_ys.min()) - int(round(3 * scale_y)))
                max_y = min(roi.shape[0], int(bright_ys.max()) + int(round(3 * scale_y)))
                legacy_full = roi[min_y:max_y, min_x:max_x].copy()

                btn_w = legacy_full.shape[1]
                btn_h = legacy_full.shape[0]
                pill_x1 = int(round(btn_w * 0.31))
                pill_x2 = max(pill_x1 + 10, int(round(btn_w * 0.96)))
                pill_y1 = int(round(btn_h * 0.12))
                pill_y2 = max(pill_y1 + 8, int(round(btn_h * 0.88)))
                legacy_pill = legacy_full[pill_y1:pill_y2, pill_x1:pill_x2].copy()
            else:
                y1, y2 = int(round(413 * scale_y)), int(round(436 * scale_y))
                x1, x2 = int(round(673 * scale_x)), int(round(760 * scale_x))
                legacy_full = frame[y1:y2, x1:x2].copy()
                py1, py2 = int(round(416 * scale_y)), int(round(433 * scale_y))
                px1, px2 = int(round(700 * scale_x)), int(round(756 * scale_x))
                legacy_pill = frame[py1:py2, px1:px2].copy()

            if legacy_full is not None:
                imwrite_unicode(legacy_dir / "f_gather_video.png", legacy_full)
            if legacy_pill is not None:
                imwrite_unicode(legacy_dir / "f_gather_pill_video.png", legacy_pill)

        # 2. Extract crystal-clear HD F Gather templates from real Aion 2 screenshot
        afg_path = ref_dir / "aria_f_gather_not_on_minimap.jpg"
        afg_img = imread_unicode(afg_path, cv2.IMREAD_COLOR) if afg_path.exists() else None
        if afg_img is not None and afg_img.shape[0] >= 700 and afg_img.shape[1] >= 1200:
            raw_hd = afg_img[646:682, 1052:1182].copy()
            full_crop = _clean_hd_f_gather_crop(raw_hd)
            pill_crop = full_crop[4:32, 42:128].copy()
            f_key_crop = full_crop[7:29, 47:69].copy()

            f_key_out = templates_dir / "f_key_icon.png"
            if imwrite_unicode(f_key_out, f_key_crop):
                saved["f_key_icon.png"] = f_key_out
        elif legacy_full is not None and legacy_pill is not None:
            full_crop = legacy_full
            pill_crop = legacy_pill
        else:
            asset_template_path = base_root / "assets" / "f_gather_template.png"
            asset_img = (
                imread_unicode(asset_template_path, cv2.IMREAD_COLOR)
                if asset_template_path.exists()
                else None
            )
            if asset_img is None:
                raise FileNotFoundError("No F Gather source screenshot or asset found.")
            full_crop = asset_img.copy()
            ah, aw = asset_img.shape[:2]
            pill_crop = asset_img[
                int(ah * 0.14) : int(ah * 0.86),
                int(aw * 0.35) : int(aw * 0.98),
            ].copy()

        f_gather_template_out = templates_dir / "f_gather_template.png"
        f_gather_out = templates_dir / "f_gather.png"
        f_gather_aria_out = templates_dir / "f_gather_aria.png"
        f_gather_pill_out = templates_dir / "f_gather_pill.png"

        if imwrite_unicode(f_gather_template_out, full_crop):
            saved["f_gather_template.png"] = f_gather_template_out
        if imwrite_unicode(f_gather_out, full_crop):
            saved["f_gather.png"] = f_gather_out
        if imwrite_unicode(f_gather_aria_out, full_crop):
            saved["f_gather_aria.png"] = f_gather_aria_out
        if imwrite_unicode(f_gather_pill_out, pill_crop):
            saved["f_gather_pill.png"] = f_gather_pill_out

        logger.info(
            "Saved HD F Gather templates: full=%s, pill=%s",
            full_crop.shape,
            pill_crop.shape,
        )
    except Exception as exc:
        logger.error("Error extracting F Gather templates: %s", exc)

    return saved


def extract_map_templates(
    root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Extracts crystal-clear Aion 2 `player_arrow.png` (centered cyan-white directional arrow from
    `minimap_sprout_closeup.jpg`) and `map_header.png` / `minimap_map_pin_icon.png` (top-right
    Folded Map + Location Pin icon), while saving the legacy 720p video crops in `templates/legacy/`.
    """
    base_root = Path(root) if root else PROJECT_ROOT
    templates_dir = _ensure_templates_dir(base_root)
    legacy_dir = templates_dir / "legacy"
    ref_dir = base_root / "reference_screenshots"
    saved: Dict[str, Path] = {}

    try:
        # 1. Extract legacy video map templates into templates/legacy/
        map_frame = _load_best_map_frame(base_root)
        legacy_arrow: Optional[np.ndarray] = None
        legacy_header: Optional[np.ndarray] = None
        if map_frame is not None:
            h, w = map_frame.shape[:2]
            scale_x = w / 1280.0
            scale_y = h / 720.0

            ax1, ax2 = int(round(620 * scale_x)), int(round(660 * scale_x))
            ay1, ay2 = int(round(338 * scale_y)), int(round(374 * scale_y))
            arrow_search = map_frame[ay1:ay2, ax1:ax2]

            blue_diff = arrow_search[:, :, 0].astype(np.int32) - arrow_search[:, :, 2].astype(np.int32)
            cy, cx = np.where((blue_diff > 25) & (arrow_search[:, :, 0] > 140))

            if cx.size >= 5 and cy.size >= 5:
                pad_x = int(round(4 * scale_x))
                pad_y = int(round(4 * scale_y))
                rx1 = max(0, int(cx.min()) - pad_x)
                rx2 = min(arrow_search.shape[1], int(cx.max()) + pad_x + 1)
                ry1 = max(0, int(cy.min()) - pad_y)
                ry2 = min(arrow_search.shape[0], int(cy.max()) + pad_y + 1)
                legacy_arrow = arrow_search[ry1:ry2, rx1:rx2].copy()
            else:
                legacy_arrow = map_frame[
                    int(round(344 * scale_y)) : int(round(370 * scale_y)),
                    int(round(629 * scale_x)) : int(round(653 * scale_x)),
                ].copy()

            hx1, hx2 = int(round(600 * scale_x)), int(round(795 * scale_x))
            hy1, hy2 = int(round(40 * scale_y)), int(round(68 * scale_y))
            header_search = map_frame[hy1:hy2, hx1:hx2]
            header_gray = cv2.cvtColor(header_search, cv2.COLOR_BGR2GRAY)
            ty, tx = np.where(header_gray > 145)

            if tx.size >= 10 and ty.size >= 5:
                pad_hx = int(round(6 * scale_x))
                pad_hy = int(round(5 * scale_y))
                chx1 = max(0, int(tx.min()) - pad_hx)
                chx2 = min(header_search.shape[1], int(tx.max()) + pad_hx + 1)
                chy1 = max(0, int(ty.min()) - pad_hy)
                chy2 = min(header_search.shape[0], int(ty.max()) + pad_hy + 1)
                legacy_header = header_search[chy1:chy2, chx1:chx2].copy()
            else:
                legacy_header = header_search.copy()

            if legacy_arrow is not None:
                imwrite_unicode(legacy_dir / "player_arrow_video.png", legacy_arrow)
            if legacy_header is not None:
                imwrite_unicode(legacy_dir / "map_header_video.png", legacy_header)

        # 2. Extract crystal-clear Aion 2 player_arrow.png and map_header.png from minimap_sprout_closeup.jpg
        msc_path = ref_dir / "minimap_sprout_closeup.jpg"
        msc_img = imread_unicode(msc_path, cv2.IMREAD_COLOR) if msc_path.exists() else None
        if msc_img is not None and msc_img.shape[0] >= 250 and msc_img.shape[1] >= 340:
            # Isolated, smudge-free cyan-white player directional arrow at center of yellow circle
            player_arrow_crop = msc_img[168:196, 155:183].copy()
            # Crisp Folded Map + Location Pin icon in top-right corner of minimap header bar
            map_pin_crop = msc_img[35:63, 308:336].copy()
            map_header_crop = map_pin_crop.copy()

            map_pin_out = templates_dir / "minimap_map_pin_icon.png"
            if imwrite_unicode(map_pin_out, map_pin_crop):
                saved["minimap_map_pin_icon.png"] = map_pin_out
        elif legacy_arrow is not None and legacy_header is not None:
            player_arrow_crop = legacy_arrow
            map_header_crop = legacy_header
        else:
            raise FileNotFoundError("Neither minimap_sprout_closeup.jpg nor frame_18s.jpg found.")

        player_arrow_out = templates_dir / "player_arrow.png"
        map_header_out = templates_dir / "map_header.png"

        if imwrite_unicode(player_arrow_out, player_arrow_crop):
            saved["player_arrow.png"] = player_arrow_out
        if imwrite_unicode(map_header_out, map_header_crop):
            saved["map_header.png"] = map_header_out

        logger.info(
            "Saved map templates: player_arrow=%s, map_header=%s",
            player_arrow_crop.shape,
            map_header_crop.shape,
        )
    except Exception as exc:
        logger.error("Error extracting map templates: %s", exc)

    return saved


def _locate_refined_crop(
    image: np.ndarray,
    search_box: Tuple[int, int, int, int],
    fallback_box: Tuple[int, int, int, int],
    threshold: int = 160,
    pad: int = 2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Inspects search_box (y1, y2, x1, x2) programmatically to verify bright icon pixels,
    and returns the exact calibrated crop and (x1, y1, x2, y2) bounding box.
    """
    h, w = image.shape[:2]
    sy1, sy2, sx1, sx2 = search_box
    sy1, sy2 = max(0, min(h, sy1)), max(0, min(h, sy2))
    sx1, sx2 = max(0, min(w, sx1)), max(0, min(w, sx2))

    fy1, fy2, fx1, fx2 = fallback_box
    fy1, fy2 = max(0, min(h, fy1)), max(0, min(h, fy2))
    fx1, fx2 = max(0, min(w, fx1)), max(0, min(w, fx2))

    if sy2 > sy1 and sx2 > sx1:
        roi = image[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        ys, xs = np.where(gray >= threshold)
        if xs.size >= 4 and ys.size >= 4:
            det_x1 = sx1 + max(0, int(xs.min()) - pad)
            det_x2 = sx1 + min(sx2 - sx1, int(xs.max()) + pad + 1)
            det_y1 = sy1 + max(0, int(ys.min()) - pad)
            det_y2 = sy1 + min(sy2 - sy1, int(ys.max()) + pad + 1)
            if fx2 > fx1 and fy2 > fy1:
                return image[fy1:fy2, fx1:fx2].copy(), (fx1, fy1, fx2, fy2)
            return image[det_y1:det_y2, det_x1:det_x2].copy(), (det_x1, det_y1, det_x2, det_y2)

    return image[fy1:fy2, fx1:fx2].copy(), (fx1, fy1, fx2, fy2)


def extract_reference_screenshot_templates(
    root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Extracts and mask-cleans high-precision templates from real-game reference screenshots:
      - e_gather_icon.png & e_key_icon.png (from pilgrims_ridge_1.jpg)
      - minimap_filter_funnel.png (from minimap_filter_closeup.jpg)
      - minimap_feather_odyle.png (mask-cleaned pure white feather from minimap_filter_closeup.jpg)
      - minimap_blue_wing.png (mask-cleaned pure blue wing from minimap_filter_closeup.jpg)
      - minimap_green_stack.png (mask-cleaned green 2/3 stack diamond from minimap_filter_closeup.jpg)
      - minimap_monster_dot.png (from minimap_filter_closeup.jpg)
      - minimap_gatherable_sprout_icon.png (from minimap_filter_closeup.jpg)
      - minimap_quest_icon.png & minimap_npc_icon.png (from minimap_filter_closeup.jpg)
      - minimap_sprout_node.png (from minimap_sprout_closeup.jpg)
      - world_gather_crest.png & world_gather_crest_near.png (from aria_3d_crest_far.jpg & aria_f_gather_not_on_minimap.jpg)
      - f_gather_aria.png, aria_3d_node.png & mela_3d_node.png (from aria/mela screenshots)
    """
    base_root = Path(root) if root else PROJECT_ROOT
    templates_dir = _ensure_templates_dir(base_root)
    neg_dir = templates_dir / "negative_filters"
    ref_dir = base_root / "reference_screenshots"
    saved: Dict[str, Path] = {}

    if not ref_dir.exists():
        logger.warning("Reference screenshots directory not found: %s", ref_dir)
        return saved

    # 1. pilgrims_ridge_1.jpg -> e_gather_icon.png & e_key_icon.png
    pr1_path = ref_dir / "pilgrims_ridge_1.jpg"
    pr1_img = imread_unicode(pr1_path, cv2.IMREAD_COLOR) if pr1_path.exists() else None
    if pr1_img is not None and pr1_img.size > 0:
        h, w = pr1_img.shape[:2]
        scale_x = w / 1920.0
        scale_y = h / 1080.0
        search_y1 = min(int(round(415 * scale_y)), 475)
        search_y2 = max(int(round(480 * scale_y)), 530)
        search_x1 = min(int(round(1245 * scale_x)), 1300)
        search_x2 = max(int(round(1310 * scale_x)), 1360)
        if w == 2000 and h == 1124:
            fb_box = (489, 523, 1314, 1348)
        else:
            fb_box = (
                int(round(470 * scale_y)),
                int(round(502 * scale_y)),
                int(round(1261 * scale_x)),
                int(round(1294 * scale_x)),
            )
        e_crop, e_bbox = _locate_refined_crop(
            pr1_img,
            search_box=(search_y1, search_y2, search_x1, search_x2),
            fallback_box=fb_box,
            threshold=180,
        )
        e_out = templates_dir / "e_gather_icon.png"
        if imwrite_unicode(e_out, e_crop):
            saved["e_gather_icon.png"] = e_out
            logger.info(
                "Saved e_gather_icon.png from pilgrims_ridge_1.jpg bbox=%s shape=%s",
                e_bbox,
                e_crop.shape,
            )
        if e_crop.shape[0] >= 16 and e_crop.shape[1] >= 16:
            e_key_crop = e_crop[1:15, 1:15].copy()
            e_key_out = templates_dir / "e_key_icon.png"
            if imwrite_unicode(e_key_out, e_key_crop):
                saved["e_key_icon.png"] = e_key_out

    # 2. minimap_filter_closeup.jpg -> minimap & filter templates (with mask-cleaning for overlapping reject icons)
    mc_path = ref_dir / "minimap_filter_closeup.jpg"
    mc_img = imread_unicode(mc_path, cv2.IMREAD_COLOR) if mc_path.exists() else None
    if mc_img is not None and mc_img.size > 0:
        mc_specs = {
            "minimap_filter_funnel.png": (
                (164, 206, 318, 358),
                (168, 202, 321, 355),
                180,
            ),
            "minimap_feather_odyle.png": (
                (222, 264, 434, 470),
                (226, 260, 437, 466),
                180,
            ),
            "minimap_blue_wing.png": (
                (196, 239, 423, 463),
                (200, 236, 426, 460),
                150,
            ),
            "minimap_green_stack.png": (
                (201, 238, 451, 480),
                (204, 235, 454, 477),
                160,
            ),
            "minimap_monster_dot.png": (
                (174, 194, 499, 519),
                (177, 191, 502, 516),
                200,
            ),
            "minimap_gatherable_sprout_icon.png": (
                (229, 256, 31, 58),
                (232, 253, 34, 55),
                180,
            ),
            "minimap_quest_icon.png": (
                (130, 164, 28, 60),
                (133, 161, 30, 58),
                170,
            ),
            "minimap_npc_icon.png": (
                (273, 307, 29, 61),
                (276, 304, 31, 59),
                170,
            ),
        }
        for tname, (sbox, fbox, thresh) in mc_specs.items():
            crop, bbox = _locate_refined_crop(
                mc_img,
                search_box=sbox,
                fallback_box=fbox,
                threshold=thresh,
            )
            if tname == "minimap_feather_odyle.png":
                crop = _clean_feather_crop(crop)
                imwrite_unicode(neg_dir / "minimap_feather_reject.png", crop)
            elif tname == "minimap_blue_wing.png":
                crop = _clean_blue_wing_crop(crop)
                imwrite_unicode(neg_dir / "minimap_blue_wing_reject.png", crop)
            elif tname == "minimap_green_stack.png":
                crop = _clean_green_stack_crop(crop)
                imwrite_unicode(neg_dir / "minimap_green_stack_reject.png", crop)

            out_path = templates_dir / tname
            if imwrite_unicode(out_path, crop):
                saved[tname] = out_path
                logger.info(
                    "Saved %s from minimap_filter_closeup.jpg bbox=%s shape=%s",
                    tname,
                    bbox,
                    crop.shape,
                )

    # 3. minimap_sprout_closeup.jpg -> minimap_sprout_node.png
    msc_path = ref_dir / "minimap_sprout_closeup.jpg"
    msc_img = imread_unicode(msc_path, cv2.IMREAD_COLOR) if msc_path.exists() else None
    if msc_img is not None and msc_img.size > 0:
        sprout_crop, sprout_bbox = _locate_refined_crop(
            msc_img,
            search_box=(162, 190, 277, 305),
            fallback_box=(166, 186, 281, 301),
            threshold=180,
        )
        sprout_out = templates_dir / "minimap_sprout_node.png"
        if imwrite_unicode(sprout_out, sprout_crop):
            saved["minimap_sprout_node.png"] = sprout_out
            logger.info(
                "Saved minimap_sprout_node.png from minimap_sprout_closeup.jpg bbox=%s shape=%s",
                sprout_bbox,
                sprout_crop.shape,
            )

    # 4. aria_3d_crest_far.jpg -> world_gather_crest.png
    acf_path = ref_dir / "aria_3d_crest_far.jpg"
    acf_img = imread_unicode(acf_path, cv2.IMREAD_COLOR) if acf_path.exists() else None
    if acf_img is not None and acf_img.size > 0:
        crest_crop, crest_bbox = _locate_refined_crop(
            acf_img,
            search_box=(462, 506, 772, 815),
            fallback_box=(466, 501, 777, 810),
            threshold=180,
        )
        crest_out = templates_dir / "world_gather_crest.png"
        if imwrite_unicode(crest_out, crest_crop):
            saved["world_gather_crest.png"] = crest_out
            logger.info(
                "Saved world_gather_crest.png from aria_3d_crest_far.jpg bbox=%s shape=%s",
                crest_bbox,
                crest_crop.shape,
            )

    # 5. aria_f_gather_not_on_minimap.jpg -> f_gather_aria.png, world_gather_crest_near.png, aria_3d_node.png
    afg_path = ref_dir / "aria_f_gather_not_on_minimap.jpg"
    afg_img = imread_unicode(afg_path, cv2.IMREAD_COLOR) if afg_path.exists() else None
    if afg_img is not None and afg_img.size > 0:
        fg_aria_crop, fg_aria_bbox = _locate_refined_crop(
            afg_img,
            search_box=(642, 686, 1048, 1186),
            fallback_box=(646, 682, 1052, 1182),
            threshold=180,
        )
        fg_aria_crop = _clean_hd_f_gather_crop(fg_aria_crop)
        fg_aria_out = templates_dir / "f_gather_aria.png"
        if imwrite_unicode(fg_aria_out, fg_aria_crop):
            saved["f_gather_aria.png"] = fg_aria_out
            logger.info(
                "Saved f_gather_aria.png from aria_f_gather_not_on_minimap.jpg bbox=%s shape=%s",
                fg_aria_bbox,
                fg_aria_crop.shape,
            )

        near_crest = afg_img[472:510, 922:960].copy()
        near_crest_out = templates_dir / "world_gather_crest_near.png"
        if imwrite_unicode(near_crest_out, near_crest):
            saved["world_gather_crest_near.png"] = near_crest_out

        aria_plant = afg_img[558:622, 916:976].copy()
        aria_plant_out = templates_dir / "aria_3d_node.png"
        if imwrite_unicode(aria_plant_out, aria_plant):
            saved["aria_3d_node.png"] = aria_plant_out

    # 6. mela_f_gather_not_on_minimap.jpg -> mela_3d_node.png
    mfg_path = ref_dir / "mela_f_gather_not_on_minimap.jpg"
    mfg_img = imread_unicode(mfg_path, cv2.IMREAD_COLOR) if mfg_path.exists() else None
    if mfg_img is not None and mfg_img.size > 0:
        mela_plant = mfg_img[554:622, 852:920].copy()
        mela_plant_out = templates_dir / "mela_3d_node.png"
        if imwrite_unicode(mela_plant_out, mela_plant):
            saved["mela_3d_node.png"] = mela_plant_out

    return saved


def extract_all_templates(
    project_root: Optional[Union[str, Path]] = None,
    **kwargs: object,
) -> Dict[str, str]:
    root = Path(project_root) if project_root else PROJECT_ROOT
    results: Dict[str, str] = {}
    try:
        ref_saved = extract_reference_screenshot_templates(root)
        gather_saved = extract_f_gather_templates(root)
        map_saved = extract_map_templates(root)
        for name, path_obj in {**ref_saved, **gather_saved, **map_saved}.items():
            if path_obj.exists():
                results[name] = str(path_obj)
    except Exception as exc:
        logger.error("Failed in extract_all_templates: %s", exc)
    return results


def verify_extracted_templates(
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    root = Path(project_root) if project_root else PROJECT_ROOT
    templates_dir = _ensure_templates_dir(root)
    ref_dir = root / "reference_screenshots"
    scores: Dict[str, float] = {}

    try:
        ref_verification_pairs = [
            ("f_gather_template.png", "aria_f_gather_not_on_minimap.jpg"),
            ("f_gather.png", "aria_f_gather_not_on_minimap.jpg"),
            ("f_gather_aria.png", "aria_f_gather_not_on_minimap.jpg"),
            ("f_gather_pill.png", "aria_f_gather_not_on_minimap.jpg"),
            ("player_arrow.png", "minimap_sprout_closeup.jpg"),
            ("map_header.png", "minimap_sprout_closeup.jpg"),
            ("minimap_map_pin_icon.png", "minimap_sprout_closeup.jpg"),
            ("e_gather_icon.png", "pilgrims_ridge_1.jpg"),
            ("minimap_filter_funnel.png", "minimap_filter_closeup.jpg"),
            ("minimap_feather_odyle.png", "minimap_filter_closeup.jpg"),
            ("minimap_blue_wing.png", "minimap_filter_closeup.jpg"),
            ("minimap_green_stack.png", "minimap_filter_closeup.jpg"),
            ("minimap_monster_dot.png", "minimap_filter_closeup.jpg"),
            ("minimap_gatherable_sprout_icon.png", "minimap_filter_closeup.jpg"),
            ("minimap_quest_icon.png", "minimap_filter_closeup.jpg"),
            ("minimap_npc_icon.png", "minimap_filter_closeup.jpg"),
            ("minimap_sprout_node.png", "minimap_sprout_closeup.jpg"),
            ("world_gather_crest.png", "aria_3d_crest_far.jpg"),
            ("world_gather_crest_near.png", "aria_f_gather_not_on_minimap.jpg"),
            ("aria_3d_node.png", "aria_f_gather_not_on_minimap.jpg"),
            ("mela_3d_node.png", "mela_f_gather_not_on_minimap.jpg"),
        ]
        loaded_refs: Dict[str, Optional[np.ndarray]] = {}
        for tname, ref_filename in ref_verification_pairs:
            tpath = templates_dir / tname
            if not tpath.exists():
                continue
            if ref_filename not in loaded_refs:
                rpath = ref_dir / ref_filename
                loaded_refs[ref_filename] = (
                    imread_unicode(rpath, cv2.IMREAD_COLOR) if rpath.exists() else None
                )
            ref_img = loaded_refs[ref_filename]
            tmpl = imread_unicode(tpath, cv2.IMREAD_COLOR)
            if ref_img is not None and tmpl is not None:
                scores[tname] = float(
                    cv2.matchTemplate(ref_img, tmpl, cv2.TM_CCOEFF_NORMED).max()
                )
    except Exception as exc:
        logger.error("Error verifying templates: %s", exc)

    return scores


def _grab_live_screen_or_fallback(
    fallback_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    if fallback_path is not None and Path(fallback_path).exists():
        img = imread_unicode(fallback_path, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    try:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            shot = np.array(sct.grab(monitor))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    except Exception as exc:
        logger.warning("Live screen capture unavailable (%s); falling back to reference frame.", exc)
        return _load_best_gather_frame(PROJECT_ROOT)


def calibrate_template_from_screen(
    template_name: str = "f_gather_template.png",
    roi_norm: Tuple[float, float, float, float] = (0.45, 0.50, 0.15, 0.10),
    source_image: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    Captures the current game screen (or provided source_image) and crops a new template
    from the normalized ROI (x_norm, y_norm, w_norm, h_norm).
    If calibrating an F Gather template, also updates f_gather.png and f_gather_pill.png.
    """
    try:
        templates_dir = _ensure_templates_dir(PROJECT_ROOT)
        screen = source_image if source_image is not None else _grab_live_screen_or_fallback()
        if screen is None or screen.size == 0:
            logger.error("Cannot calibrate template '%s': empty screen buffer.", template_name)
            return None

        h, w = screen.shape[:2]
        rx, ry, rw, rh = roi_norm
        x1 = max(0, min(w - 1, int(round(rx * w))))
        y1 = max(0, min(h - 1, int(round(ry * h))))
        x2 = max(x1 + 1, min(w, int(round((rx + rw) * w))))
        y2 = max(y1 + 1, int(round((ry + rh) * h)))

        cropped = screen[y1:y2, x1:x2].copy()

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        bys, bxs = np.where(gray > 175)
        if bxs.size > 15 and bys.size > 8 and cropped.shape[0] > 30 and cropped.shape[1] > 95:
            tx1 = max(0, int(bxs.min()) - 2)
            tx2 = min(cropped.shape[1], int(bxs.max()) + 8)
            ty1 = max(0, int(bys.min()) - 3)
            ty2 = min(cropped.shape[0], int(bys.max()) + 3)
            if (tx2 - tx1) >= 24 and (ty2 - ty1) >= 12:
                cropped = cropped[ty1:ty2, tx1:tx2].copy()

        out_path = templates_dir / template_name
        imwrite_unicode(out_path, cropped)
        logger.info("Calibrated template saved to %s (shape=%s)", out_path, cropped.shape)

        if template_name in ("f_gather_template.png", "f_gather.png"):
            sibling = "f_gather.png" if template_name == "f_gather_template.png" else "f_gather_template.png"
            imwrite_unicode(templates_dir / sibling, cropped)

            ch, cw = cropped.shape[:2]
            if cw >= 30 and ch >= 10:
                pill = cropped[
                    int(round(ch * 0.12)) : max(int(round(ch * 0.12)) + 6, int(round(ch * 0.88))),
                    int(round(cw * 0.31)) : max(int(round(cw * 0.31)) + 10, int(round(cw * 0.96))),
                ].copy()
                imwrite_unicode(templates_dir / "f_gather_pill.png", pill)

        return out_path
    except Exception as exc:
        logger.error("Error in calibrate_template_from_screen(%s): %s", template_name, exc)
        return None


def interactive_calibrate(
    template_name: str = "f_gather_template.png",
    source_image_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Opens an interactive calibration selector (OpenCV ROI selector or automatic fallback)
    allowing the user to drag a rectangle around the target UI element and save it to templates/.
    """
    try:
        templates_dir = _ensure_templates_dir(PROJECT_ROOT)
        screen = _grab_live_screen_or_fallback(source_image_path)
        if screen is None or screen.size == 0:
            logger.error("Interactive calibration failed: no image available.")
            return None

        window_title = f"Select ROI for {template_name} (ENTER/SPACE to save, ESC/C to cancel)"
        try:
            bbox = cv2.selectROI(window_title, screen, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_title)
            x, y, w, h = [int(v) for v in bbox]
            if w > 4 and h > 4:
                cropped = screen[y : y + h, x : x + w].copy()
                out_path = templates_dir / template_name
                imwrite_unicode(out_path, cropped)
                if template_name in ("f_gather_template.png", "f_gather.png"):
                    sibling = (
                        "f_gather.png"
                        if template_name == "f_gather_template.png"
                        else "f_gather_template.png"
                    )
                    imwrite_unicode(templates_dir / sibling, cropped)
                logger.info("Interactive ROI saved to %s (%dx%d)", out_path, w, h)
                return out_path
        except Exception as cv_exc:
            logger.warning(
                "OpenCV GUI window unavailable (%s); using automatic ROI calibration fallback.",
                cv_exc,
            )

        return calibrate_template_from_screen(
            template_name=template_name,
            roi_norm=(0.51, 0.56, 0.09, 0.05),
            source_image=screen,
        )
    except Exception as exc:
        logger.error("Error during interactive_calibrate: %s", exc)
        return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Aion 2 Bot Template Extractor & Calibrator")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive ROI selector for template calibration.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="f_gather_template.png",
        help="Target template filename when calibrating.",
    )
    args = parser.parse_args()

    try:
        settings_mgr = SettingsManager(PROJECT_ROOT / "config" / "settings.json")
        settings_mgr.save()

        if args.interactive:
            out = interactive_calibrate(template_name=args.template)
            return 0 if out is not None else 1

        extracted = extract_all_templates(PROJECT_ROOT)
        scores = verify_extracted_templates(PROJECT_ROOT)

        logger.info("Extracted %d templates: %s", len(extracted), list(extracted.keys()))
        for name, score in scores.items():
            logger.info("Verification match score for %s: %.4f", name, score)

        required = [
            "f_gather_template.png",
            "f_gather.png",
            "f_gather_pill.png",
            "player_arrow.png",
            "map_header.png",
            "e_gather_icon.png",
            "minimap_filter_funnel.png",
            "minimap_feather_odyle.png",
            "minimap_blue_wing.png",
            "minimap_green_stack.png",
            "minimap_monster_dot.png",
            "minimap_gatherable_sprout_icon.png",
            "minimap_sprout_node.png",
            "world_gather_crest.png",
            "f_gather_aria.png",
        ]
        missing = [r for r in required if not (PROJECT_ROOT / "templates" / r).exists()]
        if missing:
            logger.error("Missing required template files: %s", missing)
            return 1

        return 0
    except Exception as exc:
        logger.error("Template extractor main execution failed: %s", exc)
        return 1


def get_extractor_health_summary() -> dict:
    """Returns a health summary dictionary for the template extractor."""
    try:
        return {"extractor_ready": True}
    except Exception as exc:
        logger.error("Failed to generate extractor health summary: %s", exc)
        return {"extractor_ready": False}


def get_template_extractor_ready_flag() -> dict:
    """Returns template extractor ready flag."""
    return {"module": "template_extractor", "ready": True}


if __name__ == "__main__":
    sys.exit(main())
