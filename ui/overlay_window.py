# -*- coding: utf-8 -*-
"""
ui/overlay_window.py
Alias bridge providing OverlayWindow, OverlayHUD, OverlaySignals, and GameOverlay.
Facilitates Requirement R2 HUD overlay access.
"""
from ui.overlay_hud import (
    OverlayHUD,
    OverlaySignals,
    AdaptiveStatLabel,
    ClickableShortcutWidget,
    RadarProjectionWidget,
    RadarProjectionOverlay,
    GameOverlay,
)

OverlayWindow = OverlayHUD

__all__ = [
    "OverlayWindow",
    "OverlayHUD",
    "OverlaySignals",
    "AdaptiveStatLabel",
    "ClickableShortcutWidget",
    "RadarProjectionWidget",
    "RadarProjectionOverlay",
    "GameOverlay",
]
