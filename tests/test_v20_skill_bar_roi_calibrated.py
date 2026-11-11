# -*- coding: utf-8 -*-
"""v20: skill_bar must be a calibrated 12-slot ROI, not silent 8-slot fallback."""
from __future__ import annotations

import json
from pathlib import Path

from core.skill_bar_reader import SkillBarReader, slot_bounds_fractions

ROOT = Path(__file__).resolve().parent.parent
ROI_PATH = ROOT / "config" / "calibrated_rois.json"


def test_calibrated_rois_json_has_skill_bar_covering_twelve_slots() -> None:
    data = json.loads(ROI_PATH.read_text(encoding="utf-8"))
    entry = None
    if isinstance(data.get("rois"), dict):
        entry = data["rois"].get("skill_bar")
    if entry is None:
        entry = data.get("skill_bar")
    assert isinstance(entry, dict), "config/calibrated_rois.json missing skill_bar"
    x = float(entry["norm_x"])
    y = float(entry["norm_y"])
    w = float(entry["norm_w"])
    h = float(entry["norm_h"])
    assert w >= 0.10, "skill_bar width too narrow for full hotbar"
    assert h > 0.0
    # DEFAULT 12-slot strip: x1=0.345 .. x2=0.660 includes T
    assert abs(x - 0.345) < 1e-6
    assert abs((x + w) - 0.660) < 1e-6
    assert abs(y - 0.940) < 1e-6
    assert abs((y + h) - 0.991) < 1e-6


def test_skill_bar_reader_reports_calibrated_and_twelve_slots() -> None:
    reader = SkillBarReader(calibrated_rois_path=ROI_PATH)
    roi = reader.get_skill_bar_roi(force=True)
    assert reader.is_skill_bar_roi_calibrated is True
    assert "kalibre" in reader.skill_bar_roi_source.lower()
    assert roi == (0.345, 0.940, 0.660, 0.991)
    assert reader.SKILL_SLOTS == ("1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T")
    edges = slot_bounds_fractions()
    assert len(edges) == 13  # 12 slots => 13 Voronoi edges
    assert len(reader.SKILL_SLOTS) == 12
