# -*- coding: utf-8 -*-
"""
dataset_factory/rubberband_box_converter.py
===========================================
Converts user mouse-dragged rubber-band pixel coordinates ``(x1, y1) -> (x2, y2)``
drawn over the preview canvas (`target_rect`) into normalized YOLO
``(cx, cy, w, h)`` bounding box coordinates.

Responsibilities:
- Validates minimum drag threshold (``>= 8px`` by default) to reject accidental clicks
  and micro-slivers.
- Clips pixel coordinates to the active image viewport (`target_rect`) before normalization.
- Normalizes and strictly clamps YOLO coordinates ``(cx, cy, w, h)`` into ``[0.0, 1.0]``.
- Supports reverse projection from normalized YOLO coordinates back to canvas pixel space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

DEFAULT_MIN_DRAG_PX: float = 8.0
MIN_NORMALIZED_COORD: float = 0.0
MAX_NORMALIZED_COORD: float = 1.0


@dataclass(frozen=True)
class RubberBandConverterConfig:
    """Configuration parameters for :class:`RubberBandBoxConverter`."""

    min_drag_px: float = DEFAULT_MIN_DRAG_PX
    min_coord: float = MIN_NORMALIZED_COORD
    max_coord: float = MAX_NORMALIZED_COORD
    require_both_axes: bool = True
    require_clipped_min_drag: bool = True


@dataclass(frozen=True)
class RubberBandConversionResult:
    """Structured outcome of converting a rubber-band drag into a YOLO bounding box."""

    valid: bool
    reason: str
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    x1_px: float = 0.0
    y1_px: float = 0.0
    x2_px: float = 0.0
    y2_px: float = 0.0
    raw_dx_px: float = 0.0
    raw_dy_px: float = 0.0
    raw_distance_px: float = 0.0
    clipped_w_px: float = 0.0
    clipped_h_px: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.valid)

    @property
    def is_valid(self) -> bool:
        return bool(self.valid)

    @property
    def yolo_box(self) -> Tuple[float, float, float, float]:
        """Return normalized ``(cx, cy, w, h)`` tuple."""
        return (self.cx, self.cy, self.w, self.h)

    @property
    def xyxy_norm(self) -> Tuple[float, float, float, float]:
        """Return normalized ``(x1, y1, x2, y2)`` clamped to ``[0.0, 1.0]``."""
        if not self.valid:
            return (0.0, 0.0, 0.0, 0.0)
        nx1 = max(MIN_NORMALIZED_COORD, min(MAX_NORMALIZED_COORD, self.cx - self.w * 0.5))
        ny1 = max(MIN_NORMALIZED_COORD, min(MAX_NORMALIZED_COORD, self.cy - self.h * 0.5))
        nx2 = max(MIN_NORMALIZED_COORD, min(MAX_NORMALIZED_COORD, self.cx + self.w * 0.5))
        ny2 = max(MIN_NORMALIZED_COORD, min(MAX_NORMALIZED_COORD, self.cy + self.h * 0.5))
        return (nx1, ny1, nx2, ny2)

    @property
    def clipped_pixel_rect(self) -> Tuple[float, float, float, float]:
        """Return clipped canvas pixel rectangle ``(left, top, width, height)``."""
        return (self.x1_px, self.y1_px, self.clipped_w_px, self.clipped_h_px)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversion result to a dictionary."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            "bbox": [self.cx, self.cy, self.w, self.h],
            "x1_px": self.x1_px,
            "y1_px": self.y1_px,
            "x2_px": self.x2_px,
            "y2_px": self.y2_px,
            "raw_dx_px": self.raw_dx_px,
            "raw_dy_px": self.raw_dy_px,
            "raw_distance_px": self.raw_distance_px,
            "clipped_w_px": self.clipped_w_px,
            "clipped_h_px": self.clipped_h_px,
        }

    def to_annotation(
        self,
        class_id: int = 0,
        class_name: str = "herb_aria",
        confidence: float = 1.0,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a canonical manual annotation dictionary suitable for preview & export."""
        ann: Dict[str, Any] = {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "w": float(self.w),
            "h": float(self.h),
            "bbox": [float(self.cx), float(self.cy), float(self.w), float(self.h)],
            "conf": float(confidence),
            "confidence": float(confidence),
            "trusted": True,
            "hand_label": True,
            "human_approved": True,
            "source": "manual_rubberband",
        }
        if color:
            ann["color"] = str(color)
        return ann

    def to_yolo_line(self, class_id: int = 0, precision: int = 6) -> str:
        """Format as a single YOLO `.txt` annotation line ``class_id cx cy w h``."""
        fmt = f"{{:.{max(1, int(precision))}f}}"
        return (
            f"{int(class_id)} "
            f"{fmt.format(self.cx)} "
            f"{fmt.format(self.cy)} "
            f"{fmt.format(self.w)} "
            f"{fmt.format(self.h)}"
        )


def _is_number(val: Any) -> bool:
    """Return True if *val* is an int or float (and not a bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _extract_point(pt: Any) -> Optional[Tuple[float, float]]:
    """Extract ``(x, y)`` floats from a QPointF, QPoint, (x, y) tuple/list, or dict."""
    if pt is None:
        return None
    if hasattr(pt, "x") and hasattr(pt, "y"):
        x_attr = getattr(pt, "x")
        y_attr = getattr(pt, "y")
        try:
            px = float(x_attr() if callable(x_attr) else x_attr)
            py = float(y_attr() if callable(y_attr) else y_attr)
            return (px, py)
        except (TypeError, ValueError):
            return None
    if isinstance(pt, dict):
        if "x" in pt and "y" in pt:
            try:
                return (float(pt["x"]), float(pt["y"]))
            except (TypeError, ValueError):
                return None
    if isinstance(pt, (tuple, list)) and len(pt) == 2:
        try:
            return (float(pt[0]), float(pt[1]))
        except (TypeError, ValueError):
            return None
    return None


def _parse_target_rect(target_rect: Any) -> Optional[Tuple[float, float, float, float]]:
    """Extract ``(left, top, width, height)`` from a QRectF, QRect, tuple/list, or dict."""
    if target_rect is None:
        return None

    # 1. Qt QRectF / QRect or custom rect object with methods/attributes
    if hasattr(target_rect, "width") and hasattr(target_rect, "height"):
        w_attr = getattr(target_rect, "width")
        h_attr = getattr(target_rect, "height")
        l_attr = getattr(target_rect, "left", getattr(target_rect, "x", 0.0))
        t_attr = getattr(target_rect, "top", getattr(target_rect, "y", 0.0))
        try:
            left = float(l_attr() if callable(l_attr) else l_attr)
            top = float(t_attr() if callable(t_attr) else t_attr)
            width = float(w_attr() if callable(w_attr) else w_attr)
            height = float(h_attr() if callable(h_attr) else h_attr)
            return (left, top, width, height)
        except (TypeError, ValueError):
            return None

    # 2. Dict representation
    if isinstance(target_rect, dict):
        try:
            if "width" in target_rect and "height" in target_rect:
                left = float(target_rect.get("left", target_rect.get("x", target_rect.get("dx", 0.0))))
                top = float(target_rect.get("top", target_rect.get("y", target_rect.get("dy", 0.0))))
                width = float(target_rect["width"])
                height = float(target_rect["height"])
                return (left, top, width, height)
            if "w" in target_rect and "h" in target_rect:
                left = float(target_rect.get("left", target_rect.get("x", target_rect.get("dx", 0.0))))
                top = float(target_rect.get("top", target_rect.get("y", target_rect.get("dy", 0.0))))
                width = float(target_rect["w"])
                height = float(target_rect["h"])
                return (left, top, width, height)
            if "dw" in target_rect and "dh" in target_rect:
                left = float(target_rect.get("dx", target_rect.get("left", target_rect.get("x", 0.0))))
                top = float(target_rect.get("dy", target_rect.get("top", target_rect.get("y", 0.0))))
                width = float(target_rect["dw"])
                height = float(target_rect["dh"])
                return (left, top, width, height)
            if all(k in target_rect for k in ("x1", "y1", "x2", "y2")):
                x1 = float(target_rect["x1"])
                y1 = float(target_rect["y1"])
                x2 = float(target_rect["x2"])
                y2 = float(target_rect["y2"])
                return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        except (TypeError, ValueError):
            return None
        return None

    # 3. Sequence representation: (left, top, width, height) or (width, height)
    if isinstance(target_rect, (tuple, list)):
        try:
            if len(target_rect) == 4:
                return (
                    float(target_rect[0]),
                    float(target_rect[1]),
                    float(target_rect[2]),
                    float(target_rect[3]),
                )
            if len(target_rect) == 2:
                return (0.0, 0.0, float(target_rect[0]), float(target_rect[1]))
        except (TypeError, ValueError):
            return None

    return None


class RubberBandBoxConverter:
    """
    Converts mouse-dragged rubber-band pixel coordinates ``(x1, y1) -> (x2, y2)``
    on a preview canvas (`target_rect`) into normalized YOLO ``(cx, cy, w, h)``
    coordinates within ``[0.0, 1.0]``, enforcing a minimum drag threshold (``>= 8px``).
    """

    MIN_DRAG_PX: float = DEFAULT_MIN_DRAG_PX
    MIN_COORD: float = MIN_NORMALIZED_COORD
    MAX_COORD: float = MAX_NORMALIZED_COORD

    def __init__(
        self,
        min_drag_px: float = DEFAULT_MIN_DRAG_PX,
        min_coord: float = MIN_NORMALIZED_COORD,
        max_coord: float = MAX_NORMALIZED_COORD,
        require_both_axes: bool = True,
        require_clipped_min_drag: bool = True,
    ) -> None:
        self.config = RubberBandConverterConfig(
            min_drag_px=max(1.0, float(min_drag_px)),
            min_coord=float(min_coord),
            max_coord=float(max_coord),
            require_both_axes=bool(require_both_axes),
            require_clipped_min_drag=bool(require_clipped_min_drag),
        )

    @property
    def min_drag_px(self) -> float:
        """Minimum required drag distance in pixels (default ``8.0``)."""
        return self.config.min_drag_px

    @staticmethod
    def clamp_normalized_coord(
        value: float,
        min_coord: float = MIN_NORMALIZED_COORD,
        max_coord: float = MAX_NORMALIZED_COORD,
    ) -> float:
        """Clamp a normalized value into ``[0.0, 1.0]``."""
        if not math.isfinite(value):
            return min_coord
        return min(max(float(value), min_coord), max_coord)

    def clamp_yolo_box(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
    ) -> Tuple[float, float, float, float]:
        """Clip a normalized YOLO ``(cx, cy, w, h)`` box to ``[0.0, 1.0]`` boundaries."""
        lo = self.config.min_coord
        hi = self.config.max_coord
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            return (0.0, 0.0, 0.0, 0.0)

        nx1 = min(max(cx - w * 0.5, lo), hi)
        ny1 = min(max(cy - h * 0.5, lo), hi)
        nx2 = min(max(cx + w * 0.5, lo), hi)
        ny2 = min(max(cy + h * 0.5, lo), hi)

        nw = min(max(nx2 - nx1, lo), hi - lo)
        nh = min(max(ny2 - ny1, lo), hi - lo)
        ncx = min(max(nx1 + nw * 0.5, lo), hi)
        ncy = min(max(ny1 + nh * 0.5, lo), hi)
        return (ncx, ncy, nw, nh)

    @staticmethod
    def compute_drag_span(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[float, float, float]:
        """Return ``(dx_px, dy_px, euclidean_distance_px)`` for a drag vector."""
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return (0.0, 0.0, 0.0)
        dx = abs(float(x2) - float(x1))
        dy = abs(float(y2) - float(y1))
        dist = math.hypot(dx, dy)
        return (dx, dy, dist)

    def validate_drag_distance(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        min_drag_px: Optional[float] = None,
        require_both_axes: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """
        Validate that the drag from ``(x1, y1)`` to ``(x2, y2)`` satisfies the
        minimum pixel drag threshold (``>= 8px`` by default).
        """
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return False, "invalid_coordinates"

        threshold = float(self.config.min_drag_px if min_drag_px is None else min_drag_px)
        both_axes = self.config.require_both_axes if require_both_axes is None else bool(require_both_axes)
        dx, dy, dist = self.compute_drag_span(x1, y1, x2, y2)

        if dist < threshold:
            return False, "drag_too_short"
        if both_axes and (dx < threshold or dy < threshold):
            return False, "drag_too_short"
        if not both_axes and (dx <= 0.0 or dy <= 0.0):
            return False, "drag_too_short"

        return True, "ok"

    def is_valid_drag_distance(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        min_drag_px: Optional[float] = None,
        require_both_axes: Optional[bool] = None,
    ) -> bool:
        """Return True if the pixel drag distance is ``>= min_drag_px`` (default ``8px``)."""
        valid, _ = self.validate_drag_distance(
            x1,
            y1,
            x2,
            y2,
            min_drag_px=min_drag_px,
            require_both_axes=require_both_axes,
        )
        return valid

    @staticmethod
    def clip_to_target_rect(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        target_rect: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Order and clip raw pixel coordinates ``(x1, y1, x2, y2)`` to the boundaries
        of ``target_rect``, returning ``(clipped_x1, clipped_y1, clipped_x2, clipped_y2)``.
        """
        parsed_rect = _parse_target_rect(target_rect)
        if parsed_rect is None:
            return None
        left, top, width, height = parsed_rect
        if not all(math.isfinite(v) for v in (left, top, width, height)) or width <= 0.0 or height <= 0.0:
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None

        right = left + width
        bottom = top + height

        px_min_x = min(float(x1), float(x2))
        px_max_x = max(float(x1), float(x2))
        px_min_y = min(float(y1), float(y2))
        px_max_y = max(float(y1), float(y2))

        cx1 = min(max(px_min_x, left), right)
        cy1 = min(max(px_min_y, top), bottom)
        cx2 = min(max(px_max_x, left), right)
        cy2 = min(max(px_max_y, top), bottom)
        return (cx1, cy1, cx2, cy2)

    def _unpack_call_arguments(
        self,
        a0: Any = None,
        a1: Any = None,
        a2: Any = None,
        a3: Any = None,
        a4: Any = None,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        start_pos: Any = None,
        end_pos: Any = None,
        target_rect: Any = None,
    ) -> Optional[Tuple[float, float, float, float, Any]]:
        """
        Flexibly resolve ``(x1, y1, x2, y2, target_rect)`` from positional or keyword arguments:
        - ``(x1, y1, x2, y2, target_rect)``
        - ``(target_rect, x1, y1, x2, y2)``
        - ``(start_pos, end_pos, target_rect)``
        - ``(target_rect, start_pos, end_pos)``
        """
        # Keyword points
        if start_pos is not None and end_pos is not None:
            p1 = _extract_point(start_pos)
            p2 = _extract_point(end_pos)
            t_rect = target_rect if target_rect is not None else a0
            if p1 is not None and p2 is not None and t_rect is not None:
                return (p1[0], p1[1], p2[0], p2[1], t_rect)

        # Explicit keyword coordinates
        if all(v is not None for v in (x1, y1, x2, y2)):
            t_rect = target_rect if target_rect is not None else a0
            try:
                return (float(x1), float(y1), float(x2), float(y2), t_rect)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        # 5 positional arguments (or 4 positional + target_rect keyword)
        if a0 is not None and a1 is not None and a2 is not None and a3 is not None:
            if a4 is None and target_rect is not None:
                a4 = target_rect

            if a4 is not None:
                if _is_number(a0) and _is_number(a1) and _is_number(a2) and _is_number(a3):
                    return (float(a0), float(a1), float(a2), float(a3), a4)
                if not _is_number(a0) and _is_number(a1) and _is_number(a2) and _is_number(a3) and _is_number(a4):
                    return (float(a1), float(a2), float(a3), float(a4), a0)

        # 3 positional arguments (or 2 point args + target_rect keyword)
        if a0 is not None and a1 is not None and a3 is None and a4 is None:
            third = a2 if a2 is not None else target_rect
            if third is not None:
                p1 = _extract_point(a0)
                p2 = _extract_point(a1)
                if p1 is not None and p2 is not None:
                    return (p1[0], p1[1], p2[0], p2[1], third)
                p1_alt = _extract_point(a1)
                p2_alt = _extract_point(third)
                if p1_alt is not None and p2_alt is not None:
                    return (p1_alt[0], p1_alt[1], p2_alt[0], p2_alt[1], a0)

        return None

    def convert_result(
        self,
        a0: Any = None,
        a1: Any = None,
        a2: Any = None,
        a3: Any = None,
        a4: Any = None,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        start_pos: Any = None,
        end_pos: Any = None,
        target_rect: Any = None,
    ) -> RubberBandConversionResult:
        """
        Convert pixel drag coordinates on ``target_rect`` into a
        :class:`RubberBandConversionResult` containing normalized YOLO ``(cx, cy, w, h)``.
        """
        unpacked = self._unpack_call_arguments(
            a0,
            a1,
            a2,
            a3,
            a4,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            start_pos=start_pos,
            end_pos=end_pos,
            target_rect=target_rect,
        )
        if unpacked is None:
            return RubberBandConversionResult(valid=False, reason="invalid_coordinates")

        px1, py1, px2, py2, rect_obj = unpacked
        if not all(math.isfinite(v) for v in (px1, py1, px2, py2)):
            return RubberBandConversionResult(valid=False, reason="invalid_coordinates")

        parsed_rect = _parse_target_rect(rect_obj)
        if parsed_rect is None:
            return RubberBandConversionResult(valid=False, reason="invalid_target_rect")

        left, top, width, height = parsed_rect
        if not all(math.isfinite(v) for v in (left, top, width, height)) or width <= 0.0 or height <= 0.0:
            return RubberBandConversionResult(valid=False, reason="invalid_target_rect")

        raw_dx, raw_dy, raw_dist = self.compute_drag_span(px1, py1, px2, py2)
        drag_ok, drag_reason = self.validate_drag_distance(px1, py1, px2, py2)
        if not drag_ok:
            return RubberBandConversionResult(
                valid=False,
                reason=drag_reason,
                raw_dx_px=raw_dx,
                raw_dy_px=raw_dy,
                raw_distance_px=raw_dist,
            )

        clipped = self.clip_to_target_rect(px1, py1, px2, py2, parsed_rect)
        if clipped is None:
            return RubberBandConversionResult(
                valid=False,
                reason="invalid_target_rect",
                raw_dx_px=raw_dx,
                raw_dy_px=raw_dy,
                raw_distance_px=raw_dist,
            )

        cx1, cy1, cx2, cy2 = clipped
        clipped_w = cx2 - cx1
        clipped_h = cy2 - cy1

        if clipped_w <= 0.0 or clipped_h <= 0.0:
            return RubberBandConversionResult(
                valid=False,
                reason="outside_target_rect",
                x1_px=cx1,
                y1_px=cy1,
                x2_px=cx2,
                y2_px=cy2,
                raw_dx_px=raw_dx,
                raw_dy_px=raw_dy,
                raw_distance_px=raw_dist,
                clipped_w_px=max(0.0, clipped_w),
                clipped_h_px=max(0.0, clipped_h),
            )

        if self.config.require_clipped_min_drag:
            clipped_ok, _ = self.validate_drag_distance(cx1, cy1, cx2, cy2)
            if not clipped_ok:
                return RubberBandConversionResult(
                    valid=False,
                    reason="clipped_drag_too_short",
                    x1_px=cx1,
                    y1_px=cy1,
                    x2_px=cx2,
                    y2_px=cy2,
                    raw_dx_px=raw_dx,
                    raw_dy_px=raw_dy,
                    raw_distance_px=raw_dist,
                    clipped_w_px=clipped_w,
                    clipped_h_px=clipped_h,
                )

        lo = self.config.min_coord
        hi = self.config.max_coord

        norm_w = self.clamp_normalized_coord(clipped_w / width, lo, hi - lo)
        norm_h = self.clamp_normalized_coord(clipped_h / height, lo, hi - lo)
        if norm_w <= 0.0 or norm_h <= 0.0:
            return RubberBandConversionResult(
                valid=False,
                reason="outside_target_rect",
                x1_px=cx1,
                y1_px=cy1,
                x2_px=cx2,
                y2_px=cy2,
                raw_dx_px=raw_dx,
                raw_dy_px=raw_dy,
                raw_distance_px=raw_dist,
                clipped_w_px=clipped_w,
                clipped_h_px=clipped_h,
            )

        norm_cx = self.clamp_normalized_coord(((cx1 + cx2) * 0.5 - left) / width, lo, hi)
        norm_cy = self.clamp_normalized_coord(((cy1 + cy2) * 0.5 - top) / height, lo, hi)

        return RubberBandConversionResult(
            valid=True,
            reason="ok",
            cx=norm_cx,
            cy=norm_cy,
            w=norm_w,
            h=norm_h,
            x1_px=cx1,
            y1_px=cy1,
            x2_px=cx2,
            y2_px=cy2,
            raw_dx_px=raw_dx,
            raw_dy_px=raw_dy,
            raw_distance_px=raw_dist,
            clipped_w_px=clipped_w,
            clipped_h_px=clipped_h,
        )

    def convert(
        self,
        a0: Any = None,
        a1: Any = None,
        a2: Any = None,
        a3: Any = None,
        a4: Any = None,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        start_pos: Any = None,
        end_pos: Any = None,
        target_rect: Any = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Convert ``(x1, y1) -> (x2, y2)`` pixel coordinates on ``target_rect`` into
        normalized YOLO ``(cx, cy, w, h)`` in ``[0.0, 1.0]``.
        Returns ``None`` if the drag distance is ``< 8px`` or outside ``target_rect``.
        """
        res = self.convert_result(
            a0,
            a1,
            a2,
            a3,
            a4,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            start_pos=start_pos,
            end_pos=end_pos,
            target_rect=target_rect,
        )
        if not res.valid:
            return None
        return res.yolo_box

    # Convenient method aliases
    pixels_to_yolo = convert
    convert_to_yolo = convert

    def convert_points(
        self,
        start_pos: Any,
        end_pos: Any,
        target_rect: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Convert two point objects ``start_pos`` and ``end_pos`` on ``target_rect`` into ``(cx, cy, w, h)``."""
        return self.convert(start_pos=start_pos, end_pos=end_pos, target_rect=target_rect)

    def convert_to_annotation(
        self,
        a0: Any = None,
        a1: Any = None,
        a2: Any = None,
        a3: Any = None,
        a4: Any = None,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        start_pos: Any = None,
        end_pos: Any = None,
        target_rect: Any = None,
        class_id: int = 0,
        class_name: str = "herb_aria",
        confidence: float = 1.0,
        color: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a rubber-band drag directly into a manual annotation dictionary
        ready to be appended to the preview canvas and saved to the dataset.
        """
        res = self.convert_result(
            a0,
            a1,
            a2,
            a3,
            a4,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            start_pos=start_pos,
            end_pos=end_pos,
            target_rect=target_rect,
        )
        if not res.valid:
            return None
        return res.to_annotation(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            color=color,
        )

    def get_rubberband_draw_rect(
        self,
        a0: Any = None,
        a1: Any = None,
        a2: Any = None,
        a3: Any = None,
        a4: Any = None,
        *,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        start_pos: Any = None,
        end_pos: Any = None,
        target_rect: Any = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Return the clipped ``(left, top, width, height)`` pixel rectangle inside
        ``target_rect`` for rendering the live rubber-band selection overlay.
        """
        unpacked = self._unpack_call_arguments(
            a0,
            a1,
            a2,
            a3,
            a4,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            start_pos=start_pos,
            end_pos=end_pos,
            target_rect=target_rect,
        )
        if unpacked is None:
            return None
        px1, py1, px2, py2, rect_obj = unpacked
        clipped = self.clip_to_target_rect(px1, py1, px2, py2, rect_obj)
        if clipped is None:
            return None
        cx1, cy1, cx2, cy2 = clipped
        return (cx1, cy1, max(0.0, cx2 - cx1), max(0.0, cy2 - cy1))

    def yolo_to_pixel_rect(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        target_rect: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Project normalized YOLO ``(cx, cy, w, h)`` coordinates onto ``target_rect``,
        returning ``(bx, by, bw_px, bh_px)`` in canvas pixel space.
        """
        parsed_rect = _parse_target_rect(target_rect)
        if parsed_rect is None:
            return None
        left, top, width, height = parsed_rect
        if width <= 0.0 or height <= 0.0:
            return None

        ncx, ncy, nw, nh = self.clamp_yolo_box(cx, cy, w, h)
        bx = left + (ncx - nw * 0.5) * width
        by = top + (ncy - nh * 0.5) * height
        bw_px = nw * width
        bh_px = nh * height
        return (bx, by, bw_px, bh_px)

    def yolo_to_pixel_xyxy(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        target_rect: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Project normalized YOLO ``(cx, cy, w, h)`` coordinates onto ``target_rect``,
        returning ``(x1_px, y1_px, x2_px, y2_px)`` in canvas pixel space.
        """
        rect = self.yolo_to_pixel_rect(cx, cy, w, h, target_rect)
        if rect is None:
            return None
        bx, by, bw_px, bh_px = rect
        return (bx, by, bx + bw_px, by + bh_px)


__all__ = [
    "DEFAULT_MIN_DRAG_PX",
    "MIN_NORMALIZED_COORD",
    "MAX_NORMALIZED_COORD",
    "RubberBandConverterConfig",
    "RubberBandConversionResult",
    "RubberBandBoxConverter",
]
