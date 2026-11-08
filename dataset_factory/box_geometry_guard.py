"""Box geometry validation and filtering guard for dataset factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class GeometryGuardConfig:
    """Configuration thresholds for BoxGeometryGuard."""

    min_width: float = 0.004
    min_height: float = 0.004
    max_width: float = 0.65
    max_height: float = 0.65
    min_coord: float = 0.0
    max_coord: float = 1.0
    clip_partially_out_of_bounds: bool = False


class BoxGeometryGuard:
    """Filters out micro-noise boxes, giant false-positive boxes, and out-of-screen coordinates."""

    def __init__(
        self,
        min_width: float = 0.004,
        min_height: float = 0.004,
        max_width: float = 0.65,
        max_height: float = 0.65,
        min_coord: float = 0.0,
        max_coord: float = 1.0,
        clip_partially_out_of_bounds: bool = False,
    ) -> None:
        self.config = GeometryGuardConfig(
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            min_coord=min_coord,
            max_coord=max_coord,
            clip_partially_out_of_bounds=clip_partially_out_of_bounds,
        )

    def is_valid_yolo_box(
        self,
        xc: float,
        yc: float,
        w: float,
        h: float,
    ) -> Tuple[bool, str]:
        """Validate a normalized YOLO box (xc, yc, w, h) and return (is_valid, reason)."""
        cfg = self.config

        # Check finite numbers and positive dimensions
        if w <= 0.0 or h <= 0.0:
            return False, "non_positive_dimension"

        # Micro-noise check (w < 0.004 or h < 0.004)
        if w < cfg.min_width or h < cfg.min_height:
            return False, "micro_noise_box"

        # Giant false grass/sky box check (w > 0.65 or h > 0.65)
        if w > cfg.max_width or h > cfg.max_height:
            return False, "giant_false_box"

        # Center and corners out-of-screen check
        if xc < cfg.min_coord or xc > cfg.max_coord or yc < cfg.min_coord or yc > cfg.max_coord:
            return False, "center_out_of_screen"

        x1 = xc - (w / 2.0)
        y1 = yc - (h / 2.0)
        x2 = xc + (w / 2.0)
        y2 = yc + (h / 2.0)

        eps = 1e-6
        if (
            x1 < cfg.min_coord - eps
            or y1 < cfg.min_coord - eps
            or x2 > cfg.max_coord + eps
            or y2 > cfg.max_coord + eps
        ):
            return False, "corners_out_of_screen"

        return True, "ok"

    def is_valid_xyxy_box(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_width: Optional[ float ] = None,
        img_height: Optional[ float ] = None,
    ) -> Tuple[bool, str]:
        """Validate an xyxy box (normalized or pixel if img_width/img_height given)."""
        if img_width is not None and img_height is not None:
            if img_width <= 0 or img_height <= 0:
                return False, "invalid_image_dimensions"
            x1 = x1 / float(img_width)
            x2 = x2 / float(img_width)
            y1 = y1 / float(img_height)
            y2 = y2 / float(img_height)

        if x2 <= x1 or y2 <= y1:
            return False, "inverted_or_zero_xyxy"

        w = x2 - x1
        h = y2 - y1
        xc = x1 + (w / 2.0)
        yc = y1 + (h / 2.0)
        return self.is_valid_yolo_box(xc, yc, w, h)

    def sanitize_or_reject_yolo(
        self,
        xc: float,
        yc: float,
        w: float,
        h: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Optionally clip slightly out-of-bound boxes if configured, then validate."""
        cfg = self.config
        if cfg.clip_partially_out_of_bounds:
            x1 = max(cfg.min_coord, xc - (w / 2.0))
            y1 = max(cfg.min_coord, yc - (h / 2.0))
            x2 = min(cfg.max_coord, xc + (w / 2.0))
            y2 = min(cfg.max_coord, yc + (h / 2.0))
            if x2 <= x1 or y2 <= y1:
                return None
            w = x2 - x1
            h = y2 - y1
            xc = x1 + (w / 2.0)
            yc = y1 + (h / 2.0)

        valid, _ = self.is_valid_yolo_box(xc, yc, w, h)
        if not valid:
            return None

        # Clamp slight floating point drift within [0.0, 1.0]
        xc = min(max(xc, cfg.min_coord), cfg.max_coord)
        yc = min(max(yc, cfg.min_coord), cfg.max_coord)
        w = min(max(w, cfg.min_width), cfg.max_width)
        h = min(max(h, cfg.min_height), cfg.max_height)
        return (xc, yc, w, h)

    def filter_yolo_annotations(
        self,
        annotations: Iterable[Union[Sequence[Union[int, float]], Dict[str, Any]]],
    ) -> Tuple[List[Union[Tuple[int, float, float, float, float], Dict[str, Any]]], Dict[str, int]]:
        """Filter an iterable of YOLO annotations.

        Supports either:
        - sequences: (class_id, xc, yc, w, h) or (xc, yc, w, h)
        - dicts with keys: ('xc', 'yc', 'w', 'h') or ('bbox': [xc, yc, w, h])
        Returns (accepted_annotations, rejection_stats).
        """
        accepted: List[Union[Tuple[int, float, float, float, float], Dict[str, Any]]] = []
        stats: Dict[str, int] = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "micro_noise_box": 0,
            "giant_false_box": 0,
            "center_out_of_screen": 0,
            "corners_out_of_screen": 0,
            "non_positive_dimension": 0,
            "malformed": 0,
        }

        for item in annotations:
            stats["total"] += 1
            if isinstance(item, dict):
                if "bbox" in item and len(item["bbox"]) == 4:
                    xc, yc, w, h = (float(v) for v in item["bbox"])
                elif all(k in item for k in ("xc", "yc", "w", "h")):
                    xc = float(item["xc"])
                    yc = float(item["yc"])
                    w = float(item["w"])
                    h = float(item["h"])
                else:
                    stats["rejected"] += 1
                    stats["malformed"] += 1
                    continue

                valid, reason = self.is_valid_yolo_box(xc, yc, w, h)
                if valid:
                    stats["accepted"] += 1
                    accepted.append(item)
                else:
                    stats["rejected"] += 1
                    stats[reason] = stats.get(reason, 0) + 1
            elif isinstance(item, (list, tuple)):
                if len(item) == 5:
                    cls_id = int(item[0])
                    xc, yc, w, h = float(item[1]), float(item[2]), float(item[3]), float(item[4])
                elif len(item) == 4:
                    cls_id = 0
                    xc, yc, w, h = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                else:
                    stats["rejected"] += 1
                    stats["malformed"] += 1
                    continue

                valid, reason = self.is_valid_yolo_box(xc, yc, w, h)
                if valid:
                    stats["accepted"] += 1
                    accepted.append((cls_id, xc, yc, w, h))
                else:
                    stats["rejected"] += 1
                    stats[reason] = stats.get(reason, 0) + 1
            else:
                stats["rejected"] += 1
                stats["malformed"] += 1

        return accepted, stats

    def filter_yolo_label_lines(self, lines: Iterable[str]) -> Tuple[List[str], Dict[str, int]]:
        """Filter raw YOLO label text lines ('class_id xc yc w h')."""
        parsed: List[Tuple[int, float, float, float, float]] = []
        malformed_count = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                malformed_count += 1
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                parsed.append((cls_id, xc, yc, w, h))
            except ValueError:
                malformed_count += 1

        accepted, stats = self.filter_yolo_annotations(parsed)
        stats["malformed"] += malformed_count
        stats["total"] += malformed_count
        stats["rejected"] += malformed_count

        formatted_lines = [
            f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            for cls_id, xc, yc, w, h in accepted  # type: ignore[misc]
        ]
        return formatted_lines, stats
