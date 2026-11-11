# -*- coding: utf-8 -*-
"""
ui/overlay_window.py
Alias bridge providing OverlayWindow, OverlayHUD, OverlaySignals, and GameOverlay.
Facilitates Requirement R2 HUD overlay access.
NOTE: GameOverlay lives only in ui/overlay.py (single GameOverlay class);
ui/overlay_hud.py no longer aliases OverlayHUD as GameOverlay.
"""
from ui.overlay import GameOverlay
from ui.overlay_hud import (
    OverlayHUD,
    OverlaySignals,
    AdaptiveStatLabel,
    ClickableShortcutWidget,
    RadarProjectionWidget,
    RadarProjectionOverlay,
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
