# -*- coding: utf-8 -*-
"""GM fısıltı chat taraması + watchdog bağlama."""

from __future__ import annotations

import numpy as np

import cv2

from core.ocr_distance import detect_gm_whisper_chat
from core.opsec_crypto import PanicAlertSystem


def _frame_with_purple_chat_band() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    y1, y2 = int(720 * 0.75), int(720 * 0.82)
    x1, x2 = int(1280 * 0.02), int(1280 * 0.22)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = 140
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 200
    frame[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


def test_detect_gm_whisper_chat_positive():
    assert detect_gm_whisper_chat(_frame_with_purple_chat_band()) is True


def test_detect_gm_whisper_chat_negative_on_blank():
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detect_gm_whisper_chat(blank) is False


def test_bind_production_wires_gm_whisper_from_worker_grab():
    class _Worker:
        def grab_screen(self):
            return _frame_with_purple_chat_band()

    panic = PanicAlertSystem(enable_sound=False)
    wired = panic.bind_production_watchdog_sources(worker=_Worker())
    assert wired["wired"]["gm_whisper"] is True
    assert panic._gm_whisper_detector is not None
    assert panic._gm_whisper_detector() is True
