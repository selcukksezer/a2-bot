# -*- coding: utf-8 -*-
"""dataset_factory/confidence_tier_coloring.py
==============================================
3-tier bounding box visual style and confidence color-coding calculator for
the Aion 2 Autonomous Dataset Factory review & live preview canvases.

Tier rules:
1. ``tier="confident"`` (Green ``#22c55e``):
   - ``(conf >= 0.85 and trusted)`` OR ``hand_label=True`` OR ``human_approved=True``
   - Solid thick border (``line_style="solid"``, ``pen_style="SolidLine"``, ``line_width=2.2``)
   - Badge background ``#22c55e``, dark text ``#08130C``, badge text ``"✓ {cname} %{pct}"``
2. ``tier="suspicious"`` (Orange ``#fb923c``):
   - ``0.70 <= conf < 0.85`` (or ``conf >= 0.85`` when untrusted and not human-approved)
   - Dashed border (``line_style="dashed"``, ``pen_style="DashLine"``, ``line_width=1.8``)
   - Badge background ``#fb923c``, dark-brown text ``#1c1106``, badge text ``"🟠 şüpheli {cname} %{pct}"``
3. ``tier="low_confidence"`` (Red ``#ef4444``):
   - ``conf < 0.70`` (when not ``hand_label`` / ``human_approved``)
   - Dashed border (``line_style="dashed"``, ``pen_style="DashLine"``, ``line_width=1.8``)
   - Badge background ``#ef4444``, white text ``#ffffff``, badge text ``"şüpheli {cname} %{pct}"``
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Dict, Mapping, Optional, Tuple

# =============================================================================
# Tier & Threshold Constants
# =============================================================================

TIER_CONFIDENT: str = "confident"
TIER_SUSPICIOUS: str = "suspicious"
TIER_LOW_CONFIDENCE: str = "low_confidence"

CONFIDENT_THRESHOLD: float = 0.85
SUSPICIOUS_THRESHOLD: float = 0.70

CONFIDENT_COLOR_HEX: str = "#22c55e"
SUSPICIOUS_COLOR_HEX: str = "#fb923c"
LOW_CONFIDENCE_COLOR_HEX: str = "#ef4444"

CONFIDENT_TEXT_COLOR_HEX: str = "#08130C"
SUSPICIOUS_TEXT_COLOR_HEX: str = "#1c1106"
LOW_CONFIDENCE_TEXT_COLOR_HEX: str = "#ffffff"

CONFIDENT_STATUS_COLOR_HEX: str = "#4ade80"
SUSPICIOUS_STATUS_COLOR_HEX: str = "#fb923c"
LOW_CONFIDENCE_STATUS_COLOR_HEX: str = "#f87171"

TRUSTED_SOURCES: frozenset[str] = frozenset(
    {
        "trusted",
        "hand_label",
        "human_approved",
    }
)


class ConfidenceTier(str, Enum):
    """Canonical 3-tier confidence classification names."""

    CONFIDENT = TIER_CONFIDENT
    SUSPICIOUS = TIER_SUSPICIOUS
    LOW_CONFIDENCE = TIER_LOW_CONFIDENCE


# =============================================================================
# Helper Functions
# =============================================================================


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a ``#RRGGBB`` hex string to an ``(R, G, B)`` tuple.

    Raises ``ValueError`` on invalid input. Previously any invalid code
    silently became confident-green ``(34, 197, 94)``, so a typo'd colour
    rendered a box as trusted/confident.
    """
    s = str(hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex colour (expected #RRGGBB): {hex_color!r}")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        raise ValueError(f"Invalid hex colour (expected #RRGGBB): {hex_color!r}") from None
    return (r, g, b)


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert a ``#RRGGBB`` hex string to an OpenCV ``(B, G, R)`` tuple."""
    r, g, b = hex_to_rgb(hex_color)
    return (b, g, r)


def _tier_template(
    line_color: str,
    text_color: str,
    status_color: str,
    line_style: str,
    pen_style: str,
    line_width: float,
    fill_alpha: int,
    conf_style: str,
) -> Dict[str, Any]:
    """Precompute the per-tier constant fields of a box style dict.

    The three tier colours are fixed module constants, so their RGB/BGR
    tuples and the long ``conf_style`` CSS strings are parsed/built once at
    import time instead of on every :func:`get_box_visual_style` call (the
    hot path of the review/live-preview canvases, called once per box).
    Only the per-annotation fields (tier flags, badge/status text, class
    name, confidence, pct) are filled in per call.
    """
    rgb = hex_to_rgb(line_color)
    return {
        "line_color": line_color,
        "box_color": line_color,
        "border_color": line_color,
        "color": line_color,
        "color_hex": line_color,
        "color_rgb": rgb,
        "color_bgr": (rgb[2], rgb[1], rgb[0]),
        "badge_bg_color": line_color,
        "badge_bg": line_color,
        "badge_background_color": line_color,
        "text_color": text_color,
        "badge_text_color": text_color,
        "font_color": text_color,
        "line_style": line_style,
        "pen_style": pen_style,
        "border_style": line_style,
        "line_width": line_width,
        "pen_width": line_width,
        "border_width": line_width,
        "fill_alpha": fill_alpha,
        "status_color": status_color,
        "conf_style": conf_style,
    }


_TIER_STYLES: Dict[str, Dict[str, Any]] = {
    TIER_CONFIDENT: _tier_template(
        CONFIDENT_COLOR_HEX,
        CONFIDENT_TEXT_COLOR_HEX,
        CONFIDENT_STATUS_COLOR_HEX,
        "solid",
        "SolidLine",
        2.2,
        40,
        "background:#0d2818;color:#4ade80;font-family:Consolas;font-size:11px;font-weight:700;"
        "border:1px solid #16a34a;border-radius:6px;padding:2px 6px;",
    ),
    TIER_SUSPICIOUS: _tier_template(
        SUSPICIOUS_COLOR_HEX,
        SUSPICIOUS_TEXT_COLOR_HEX,
        SUSPICIOUS_STATUS_COLOR_HEX,
        "dashed",
        "DashLine",
        1.8,
        30,
        "background:#291804;color:#fb923c;font-family:Consolas;font-size:11px;font-weight:700;"
        "border:1px solid #d97706;border-radius:6px;padding:2px 6px;",
    ),
    TIER_LOW_CONFIDENCE: _tier_template(
        LOW_CONFIDENCE_COLOR_HEX,
        LOW_CONFIDENCE_TEXT_COLOR_HEX,
        LOW_CONFIDENCE_STATUS_COLOR_HEX,
        "dashed",
        "DashLine",
        1.8,
        30,
        "background:#3b1010;color:#f87171;font-family:Consolas;font-size:11px;font-weight:700;"
        "border:1px solid #ef4444;border-radius:6px;padding:2px 6px;",
    ),
}


_CANONICAL_NAMES: Optional[Dict[int, str]] = None


def _get_canonical_names() -> Dict[int, str]:
    """Lazily load (once) the canonical class-id -> name map.

    Previously :func:`extract_class_name` re-ran the
    ``from dataset_factory.class_taxonomy import CANONICAL_NAMES`` import on
    every ``class_id``-only annotation; the import itself is cached by
    Python but the lookup ran per call. The map is a module constant, so
    one load is enough.
    """
    global _CANONICAL_NAMES
    if _CANONICAL_NAMES is None:
        try:
            from dataset_factory.class_taxonomy import CANONICAL_NAMES

            _CANONICAL_NAMES = dict(CANONICAL_NAMES)
        except Exception:
            _CANONICAL_NAMES = {}
    return _CANONICAL_NAMES


def _truthy_flag(value: Any) -> bool:
    """Return True if a flag value represents boolean truth."""
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ok", "approved", "trusted"}
    return bool(value)


def normalize_confidence(raw_conf: Any, default: float = 0.0) -> float:
    """Normalize a raw confidence value (``0..1`` or ``0..100``) into ``[0.0, 1.0]``.

    Missing/unparseable confidence defaults to ``0.0`` (unknown), never to
    0.97: an empty annotation used to be displayed as 97% confident.
    """
    if raw_conf is None:
        val = float(default)
    else:
        try:
            val = float(raw_conf)
        except (TypeError, ValueError):
            val = float(default)

    if math.isnan(val) or math.isinf(val):
        val = float(default)

    if val > 1.0:
        val = val / 100.0

    return max(0.0, min(1.0, val))


def extract_box_confidence(ann: Mapping[str, Any], default: float = 0.0) -> float:
    """Extract and normalize confidence from an annotation dictionary."""
    if not isinstance(ann, Mapping):
        return normalize_confidence(None, default=default)

    for key in ("conf", "confidence", "score"):
        if key in ann and ann[key] is not None:
            return normalize_confidence(ann[key], default=default)

    return normalize_confidence(None, default=default)


def extract_class_name(ann: Mapping[str, Any], default: str = "unknown") -> str:
    """Extract display class name from an annotation dictionary."""
    if not isinstance(ann, Mapping):
        return default

    for key in ("class_name", "key", "name", "label_name"):
        val = ann.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    if "class_id" in ann and ann["class_id"] is not None:
        try:
            names = _get_canonical_names()
            cid = int(ann["class_id"])
            if cid in names:
                return str(names[cid])
        except Exception:
            pass
        return str(ann["class_id"])

    return default


def is_box_confident(ann: Mapping[str, Any], norm_conf: Optional[float] = None) -> bool:
    """Return True if the annotation qualifies for the green ``confident`` tier.

    Matches the canonical UI and dataset trust specification:
    - ``hand_label`` is True, OR
    - ``human_approved`` is True, OR
    - ``norm_conf >= 0.85`` AND (``trusted`` is True or ``source`` in
      ``{"trusted", "hand_label", "human_approved"}``).
    """
    if not isinstance(ann, Mapping):
        return False

    if norm_conf is None:
        norm_conf = extract_box_confidence(ann)

    hand_label = _truthy_flag(ann.get("hand_label"))
    human_approved = _truthy_flag(ann.get("human_approved"))
    source_str = str(ann.get("source", "") or "").strip().lower()

    if hand_label or human_approved or source_str in {"hand_label", "human_approved"}:
        return True

    # NOTE: an incoming "tier": "confident" hint is NOT trust evidence -- a
    # box used to promote itself to green just by claiming the confident
    # tier. Only an explicit trusted flag / trusted source counts, plus the
    # >= 0.85 confidence threshold.
    is_trusted = _truthy_flag(ann.get("trusted")) or source_str in TRUSTED_SOURCES
    return bool(is_trusted and norm_conf >= CONFIDENT_THRESHOLD)


def classify_confidence_tier(ann: Mapping[str, Any]) -> str:
    """Classify an annotation dictionary into one of the 3 confidence tiers.

    Returns:
        ``"confident"``, ``"suspicious"``, or ``"low_confidence"``.
    """
    norm_conf = extract_box_confidence(ann)
    if is_box_confident(ann, norm_conf=norm_conf):
        return TIER_CONFIDENT
    if norm_conf >= SUSPICIOUS_THRESHOLD:
        return TIER_SUSPICIOUS
    return TIER_LOW_CONFIDENCE


# =============================================================================
# Core Public API
# =============================================================================


def get_box_visual_style(ann: dict) -> dict:
    """Compute the 3-tier visual style dictionary for a bounding box annotation.

    Tier Rules:
    - ``tier="confident"`` (``>= 0.85`` with ``trusted`` / ``hand_label`` / ``human_approved``):
      Green ``#22c55e``, solid line (width 2.2), badge ``"✓ {cname} %{pct}"``, text ``#08130C``.
    - ``tier="suspicious"`` (``0.70 .. 0.84``, or ``>= 0.85`` without trust/approval):
      Orange ``#fb923c``, dashed line (width 1.8), badge ``"🟠 şüpheli {cname} %{pct}"``, text ``#1c1106``.
    - ``tier="low_confidence"`` (``< 0.70``):
      Red ``#ef4444``, dashed line (width 1.8), badge ``"şüpheli {cname} %{pct}"``, text ``#ffffff``.

    Args:
        ann: Bounding box annotation dictionary (e.g. ``{"class_name": "herb_aria",
            "conf": 0.91, "trusted": True}``).

    Returns:
        Dictionary containing tier name, line color, badge background color,
        badge text color, line style, line width, fill alpha, and formatted badge text.
    """
    safe_ann: Mapping[str, Any] = ann if isinstance(ann, Mapping) else {}
    cname = extract_class_name(safe_ann, default="unknown")
    norm_conf = extract_box_confidence(safe_ann, default=0.0)
    pct = max(0, min(100, int(round(norm_conf * 100))))

    confident = is_box_confident(safe_ann, norm_conf=norm_conf)

    if confident:
        tier = TIER_CONFIDENT
        badge_text = f"✓ {cname} %{pct}"
        status_text = f"Emin · %{pct} güven"
    elif norm_conf >= SUSPICIOUS_THRESHOLD:
        tier = TIER_SUSPICIOUS
        badge_text = f"🟠 şüpheli {cname} %{pct}"
        status_text = f"Şüpheli · %{pct} güven"
    else:
        tier = TIER_LOW_CONFIDENCE
        badge_text = f"şüpheli {cname} %{pct}"
        status_text = f"Düşük Güven (Şüpheli) · %{pct} güven"

    # Constant per-tier fields come from the precomputed template (parsed
    # once at import); only per-annotation fields are filled in here.
    style = dict(_TIER_STYLES[tier])
    style.update(
        {
            "tier": tier,
            "is_confident": tier == TIER_CONFIDENT,
            "is_suspicious": tier == TIER_SUSPICIOUS,
            "is_low_confidence": tier == TIER_LOW_CONFIDENCE,
            "badge_text": badge_text,
            "label_text": badge_text,
            "label_str": badge_text,
            "label": badge_text,
            "status_text": status_text,
            "class_name": cname,
            "conf": norm_conf,
            "confidence": norm_conf,
            "pct": pct,
            "confidence_pct": pct,
        }
    )
    return style


class ConfidenceTierColoring:
    """Stateful/configurable wrapper around :func:`get_box_visual_style`."""

    TIER_CONFIDENT = TIER_CONFIDENT
    TIER_SUSPICIOUS = TIER_SUSPICIOUS
    TIER_LOW_CONFIDENCE = TIER_LOW_CONFIDENCE

    @staticmethod
    def get_box_visual_style(ann: dict) -> dict:
        """Delegate to :func:`get_box_visual_style`."""
        return get_box_visual_style(ann)

    @staticmethod
    def classify_tier(ann: dict) -> str:
        """Delegate to :func:`classify_confidence_tier`."""
        return classify_confidence_tier(ann)


__all__ = [
    "CONFIDENT_COLOR_HEX",
    "CONFIDENT_STATUS_COLOR_HEX",
    "CONFIDENT_TEXT_COLOR_HEX",
    "CONFIDENT_THRESHOLD",
    "ConfidenceTier",
    "ConfidenceTierColoring",
    "LOW_CONFIDENCE_COLOR_HEX",
    "LOW_CONFIDENCE_STATUS_COLOR_HEX",
    "LOW_CONFIDENCE_TEXT_COLOR_HEX",
    "SUSPICIOUS_COLOR_HEX",
    "SUSPICIOUS_STATUS_COLOR_HEX",
    "SUSPICIOUS_TEXT_COLOR_HEX",
    "SUSPICIOUS_THRESHOLD",
    "TIER_CONFIDENT",
    "TIER_LOW_CONFIDENCE",
    "TIER_SUSPICIOUS",
    "TRUSTED_SOURCES",
    "classify_confidence_tier",
    "extract_box_confidence",
    "extract_class_name",
    "get_box_visual_style",
    "hex_to_bgr",
    "hex_to_rgb",
    "is_box_confident",
    "normalize_confidence",
]
