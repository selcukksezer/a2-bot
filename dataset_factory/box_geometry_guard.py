"""Box geometry validation and filtering guard for dataset factory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


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

        # F-0831: NaN/inf coordinates must never pass — every comparison
        # against NaN is False, so a NaN box used to slip through as "valid".
        try:
            all_finite = all(
                math.isfinite(v) for v in (xc, yc, w, h)
            )
        except TypeError:
            all_finite = False
        if not all_finite:
            return False, "non_finite_coordinate"

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
        # S-2403: exactly one of img_width/img_height given used to silently
        # skip the division, so pixel values (e.g. x2=1920) were validated
        # as if they were normalized [0,1] and usually rejected as a
        # giant_false_box. The signature accepts both as independent
        # Optionals, so a partial pair is a caller error — reject it
        # explicitly instead of returning a silently wrong verdict.
        if (img_width is None) != (img_height is None):
            return False, "partial_image_dimensions"
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
            try:
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


            except (TypeError, ValueError, OverflowError):
                # F-0830: a non-numeric box must reject only itself,
                # never crash the whole filter batch.
                stats["rejected"] += 1
                stats["malformed"] += 1
                continue
        return accepted, stats

    def filter_yolo_arrays(
        self,
        boxes: Union[np.ndarray, Sequence[Sequence[float]]],
        class_ids: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, int]]:
        """Vectorized batch filter for ``(xc, yc, w, h)`` boxes (D42).

        Same acceptance rules and reason precedence as :meth:`is_valid_yolo_box`
        (non-finite → non-positive → micro → giant → center → corners),
        evaluated with one NumPy pass instead of a per-box Python loop:
        50k boxes validate in ~0.005-0.014s vs ~0.21-1.40s for
        :meth:`filter_yolo_annotations` on the same data (this VM, min of
        3 timeit runs per side, load avg 20-40; see dev-bench-d42/bench.py).

        Args:
            boxes: ``(N, 4)`` array-like of ``(xc, yc, w, h)`` rows (a flat
                4-tuple is treated as a single box; empty input is allowed).
            class_ids: optional length-``N`` class ids, carried through for
                accepted rows (truncated toward zero like ``int(...)``).

        Returns:
            ``(accepted_boxes, accepted_class_ids, stats)`` where
            ``accepted_boxes`` is a float64 ``(M, 4)`` array,
            ``accepted_class_ids`` is an int64 array or ``None`` when no ids
            were given, and ``stats`` mirrors the scalar filter's keys
            (``total`` / ``accepted`` / ``rejected`` / per-reason counts).

        Raises:
            ValueError: non-numeric boxes, wrong shape, or a ``class_ids``
                length that does not match the number of boxes.
        """
        try:
            arr = np.asarray(boxes, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("boxes must be numeric (N, 4) values") from exc
        if arr.ndim == 1:
            if arr.size == 0:
                arr = arr.reshape(0, 4)
            elif arr.size == 4:
                arr = arr.reshape(1, 4)
            else:
                raise ValueError(f"boxes must have shape (N, 4), got {arr.shape}")
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {arr.shape}")

        n = int(arr.shape[0])
        stats: Dict[str, int] = {
            "total": n,
            "accepted": 0,
            "rejected": 0,
            "micro_noise_box": 0,
            "giant_false_box": 0,
            "center_out_of_screen": 0,
            "corners_out_of_screen": 0,
            "non_positive_dimension": 0,
            "non_finite_coordinate": 0,
            "malformed": 0,
        }
        if n == 0:
            empty_ids = (
                np.zeros((0,), dtype=np.int64) if class_ids is not None else None
            )
            return arr.reshape(0, 4), empty_ids, stats

        ids: Optional[np.ndarray] = None
        if class_ids is not None:
            try:
                ids = np.asarray(class_ids, dtype=np.float64).ravel().astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("class_ids must be numeric") from exc
            if ids.shape[0] != n:
                raise ValueError(
                    f"class_ids length {ids.shape[0]} does not match {n} boxes"
                )

        cfg = self.config
        xc = arr[:, 0]
        yc = arr[:, 1]
        w = arr[:, 2]
        h = arr[:, 3]
        eps = 1e-6
        x1 = xc - (w / 2.0)
        y1 = yc - (h / 2.0)
        x2 = xc + (w / 2.0)
        y2 = yc + (h / 2.0)

        # Integer reason codes, not strings: np.select over a "<U22" string
        # array pays a full() of 50k wide strings plus six string
        # comparisons for the counts; int8 codes + one bincount do the same
        # work in ~1/5 of the time (0.30s -> 0.06s per 50k boxes). np.select
        # takes the FIRST matching condition, so listing the reasons in
        # is_valid_yolo_box order reproduces its precedence exactly (a NaN
        # row matches only the first, non-finite, condition because every
        # NaN comparison below it is False).
        # Codes: 0 ok, 1 non_finite, 2 non_positive, 3 micro, 4 giant,
        # 5 center, 6 corners.
        reason_codes = np.select(
            [
                ~np.isfinite(arr).all(axis=1),
                (w <= 0.0) | (h <= 0.0),
                (w < cfg.min_width) | (h < cfg.min_height),
                (w > cfg.max_width) | (h > cfg.max_height),
                (xc < cfg.min_coord)
                | (xc > cfg.max_coord)
                | (yc < cfg.min_coord)
                | (yc > cfg.max_coord),
                (x1 < cfg.min_coord - eps)
                | (y1 < cfg.min_coord - eps)
                | (x2 > cfg.max_coord + eps)
                | (y2 > cfg.max_coord + eps),
            ],
            [1, 2, 3, 4, 5, 6],
            default=0,
        )
        counts = np.bincount(reason_codes, minlength=7)
        for code, reason in enumerate(
            (
                "ok",
                "non_finite_coordinate",
                "non_positive_dimension",
                "micro_noise_box",
                "giant_false_box",
                "center_out_of_screen",
                "corners_out_of_screen",
            )
        ):
            if code == 0:
                continue
            stats[reason] = int(counts[code])
        accepted_mask = reason_codes == 0
        stats["accepted"] = int(counts[0])
        stats["rejected"] = n - stats["accepted"]
        accepted_boxes = np.ascontiguousarray(arr[accepted_mask], dtype=np.float64)
        accepted_ids = ids[accepted_mask] if ids is not None else None
        return accepted_boxes, accepted_ids, stats

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
