# -*- coding: utf-8 -*-
"""Gather cast barı boş başlamalı — soldan dolgu ucu."""

from __future__ import annotations

import numpy as np

from core.ocr_distance import GameHUDReader


def test_gather_cast_progress_empty_when_cyan_only_on_right():
    """Tam genişlikte cyan çerçeve ama dolgu yok -> ilerleme ~0."""
    reader = GameHUDReader()
    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    bx1, bx2 = int(w * 0.422), int(w * 0.571)
    by1, by2 = int(h * 0.048), int(h * 0.059)
    b_top, b_bot = int(h * 0.046), int(h * 0.061)
    frame[b_top, bx1:bx2] = (25, 25, 25)
    frame[b_bot, bx1:bx2] = (25, 25, 25)
  # Sağ uçta cyan leke — eski ölçüm bunu %100 sanıyordu.
    frame[by1:by2, bx2 - 18 : bx2 - 2] = (220, 180, 40)
    ratio = reader.read_cast_bar_progress(frame)
    assert ratio < 0.25
    present, pct = reader.verify_gather_cast_bar(frame)
    if present:
        assert pct < 30.0
