# -*- coding: utf-8 -*-
"""YOLO Fabrikası visual tokens + stylesheet loader.

Operator-console palette (olive-steel neutrals, single amber accent).
QSS lives in ``ui/styles/yolo_factory.qss``; this module holds tokens and
loads/substitutes them. Keep giant stylesheets out of ``yolo_factory_dialog.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

from ui.shell_theme import SHELL_PAGE_FRAME_QSS

# Header stamp — bump when Onayla chrome changes so operators can verify load.
UI_BUILD_STAMP: str = "onayla-r4 · 2026-11-11"

# Olive-steel industrial console — one amber accent, no purple/cyan glow.
PIPELINE_COLORS: Dict[str, str] = {
    "bg": "#0C0E0C",
    "panel": "#141816",
    "panel2": "#1A1E1A",
    "panel3": "#222822",
    "line": "#2A3028",
    "line2": "#3A4238",
    "text": "#E8EBE6",
    "muted": "#8B9288",
    "muted2": "#6A7168",
    "amber": "#E6A817",
    "amber_soft": "#2A220E",
    "green": "#3D9B6A",
    "green_soft": "#0F2A1D",
    "red": "#D95C5C",
    "red_soft": "#24151A",
    "radius": "6px",
    "font": "'Segoe UI Variable', Bahnschrift, 'Segoe UI', sans-serif",
    "mono": "Consolas, 'Cascadia Mono', monospace",
}

# Compact control metrics (operator density).
BTN_HEIGHT_MIN: int = 24
BTN_HEIGHT_MAX: int = 28
# Legacy alias — r4 dropped the left sidebar; filmstrip height is the queue chrome.
SIDEBAR_WIDTH: int = 0
FILMSTRIP_HEIGHT: int = 86

_STYLES_DIR = Path(__file__).resolve().parent / "styles"
_QSS_PATH = _STYLES_DIR / "yolo_factory.qss"


def _substitute(template: str, colors: Mapping[str, str]) -> str:
    out = template
    for key, value in colors.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def load_yolo_factory_qss(
    colors: Optional[Mapping[str, str]] = None,
    *,
    include_shell_frame: bool = True,
) -> str:
    """Load ``yolo_factory.qss`` and substitute ``{{token}}`` placeholders."""
    C: MutableMapping[str, str] = dict(PIPELINE_COLORS)
    if colors:
        C.update(dict(colors))
    raw = _QSS_PATH.read_text(encoding="utf-8")
    body = _substitute(raw, C)
    if include_shell_frame:
        return SHELL_PAGE_FRAME_QSS + "\n" + body
    return body


def apply_yolo_factory_theme(widget, colors: Optional[Mapping[str, str]] = None) -> str:
    """Apply factory QSS to a dialog/widget; returns the stylesheet used."""
    qss = load_yolo_factory_qss(colors)
    widget.setStyleSheet(qss)
    return qss
