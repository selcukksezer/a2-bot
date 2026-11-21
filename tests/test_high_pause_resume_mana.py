# -*- coding: utf-8 -*-
"""High #1–7 regression: pause/resume gating, Durdur, mana flicker."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)


def test_mp_flicker_needs_second_sample_before_potion():
    """High #3: full-bar flicker at 20–40% must not pot on first sample."""
    from core.mana_phase_combat import ManaPhaseCombat

    combat = ManaPhaseCombat(
        mp_potion_threshold=0.45,
        auto_use_mp_potion=True,
        mp_potion_cd_sec=1.0,
    )
    # Healthy baseline then sudden 30% flicker (full bar OCR fail).
    combat._mp_low_reading_confirmed(1.0, now=100.0)
    first = combat.check_auto_potions(player_hp=1.0, player_mp=0.30, now=101.0)
    assert first.mp_potion_triggered is False
    # Second consecutive low sample may pot.
    second = combat.check_auto_potions(player_hp=1.0, player_mp=0.30, now=102.0)
    assert second.mp_potion_triggered is True


def test_full_cyan_mp_bar_reads_near_full():
    """High #3: saturated cyan fill across ROI must read ~1.0, not 20–40%."""
    from core.mana_phase_combat import ManaPhaseCombat

    combat = ManaPhaseCombat()
    combat.get_calibrated_roi_bounds = lambda: {  # type: ignore[method-assign]
        "player_hp_bar": (0.10, 0.80, 0.30, 0.85),
        "player_mp_bar": (0.60, 0.80, 0.80, 0.85),
        "target_hp_bar": (0.35, 0.03, 0.65, 0.075),
    }
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # Pure cyan/blue BGR fill for the whole MP ROI + white digit holes.
    frame[800:850, 600:800] = (255, 200, 0)
    frame[810:840, 650:680] = (255, 255, 255)  # white "numbers" punch
    vitals = combat.read_vitals_from_frame(frame)
    assert vitals["player_mp"] >= 0.90
    assert vitals["mp_potion_needed"] is False


def test_overlay_stop_emits_stop_not_f10_toggle():
    """High #7: Durdur must emit stop_clicked (pause-only), not f10 toggle."""
    from ui.overlay_hud import OverlayHUD

    hud = OverlayHUD()
    fired = []
    hud.f10_clicked.connect(lambda: fired.append("f10"))
    hud.stop_clicked.connect(lambda: fired.append("stop"))
    hud.btn_stop.click()
    _app.processEvents()
    assert fired == ["stop"]


def test_overlay_gear_emits_settings_clicked():
    """High #5: dişli opens Ayarlar signal — no mode/resume side effect."""
    from ui.overlay_hud import OverlayHUD

    hud = OverlayHUD()
    fired = []
    hud.settings_clicked.connect(lambda: fired.append("settings"))
    hud.btn_hud_settings.click()
    _app.processEvents()
    assert fired == ["settings"]


def test_activate_bot_mode_set_mode_without_resume():
    """High #1/#5/#6: mode menu action must not call resume()."""
    import main as main_mod

    class _W:
        def __init__(self):
            self.mode_calls = []
            self.resume_calls = 0
            self.is_paused = True

        def set_mode(self, mode):
            self.mode_calls.append(mode)

        def resume(self):
            self.resume_calls += 1

    class _App:
        pass

    app = _App()
    app.worker = _W()
    app.watchdog = SimpleNamespace(set_paused=lambda *a, **k: None)
    main_mod.GatherionOrchestrator._activate_bot_mode(app, "auto_record", "WALKING")
    assert app.worker.mode_calls == ["auto_record"]
    assert app.worker.resume_calls == 0

    main_mod.GatherionOrchestrator._activate_bot_mode(app, None, "WALKING")
    assert app.worker.resume_calls == 1


def test_stop_clicked_while_paused_does_not_resume():
    """High #7: Durdur while already paused stays paused (no toggle)."""
    import main as main_mod

    class _W:
        def __init__(self):
            self.is_paused = True
            self.pause_calls = []
            self.resume_calls = 0

        def pause(self, reason=""):
            self.pause_calls.append(reason)
            self.is_paused = True

        def resume(self):
            self.resume_calls += 1
            self.is_paused = False

    class _Overlay:
        def add_log(self, *_a, **_k):
            pass

    class _App:
        pass

    app = _App()
    app.worker = _W()
    app.overlay = _Overlay()
    app._pause_chord_label = "Ctrl+Alt+F10"
    main_mod.GatherionOrchestrator._pause_bot_from_user(app)
    assert app.worker.is_paused is True
    assert app.worker.resume_calls == 0
    assert len(app.worker.pause_calls) == 1


def test_watchdog_stall_candidate_never_resumes():
    """High #2: stall resume candidate is always None."""
    from core.watchdog import BotWatchdog

    wd = BotWatchdog(stall_timeout_sec=15.0, check_interval_sec=30.0)
    assert wd._stall_resume_candidate_unlocked() is None


def test_f10_shortcut_label_mentions_durdur():
    """High #7: F10 shortcut chip still labeled Durdur (Başlat is btn_run)."""
    from ui.overlay_hud import OverlayHUD

    hud = OverlayHUD()
    tip = hud.f10_shortcut.toolTip() or ""
    assert "Durdur" in tip
    assert "Durdur" in (hud.btn_stop.text() or "")
