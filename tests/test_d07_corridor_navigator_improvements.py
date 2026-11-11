# -*- coding: utf-8 -*-
"""D07 — corridor_navigator iyileştirmeleri için kilit testleri.

Kapsar:
1. ContestedNodeDetector kısa devreleri (boş / geçersiz isimlik karelerinde
   pahalı OpenCV adımları atlanır) — davranış birebir korunur.
2. İsimlik yatay-şekil kuralının tek yardımcıya (`_has_valid_nameplate_blob`)
   taşınması — sınır değerleriyle kilitlenir.
3. ExemplarMemory tutamacının navigator başına önbelleklenmesi + başarısız
   yüklemenin 60 sn hatırlanması (uyarı seli yok, yeniden deneme pencereli).
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.corridor_navigator import (
    ContestedNodeDetector,
    CorridorNavigator,
    _has_valid_nameplate_blob,
)

NODE_XY = (960.0, 540.0)


def _valid_color_frame() -> np.ndarray:
    """Beyaz yatay isimlik + isimlik ROI'si altında camgöbeği parıltı."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[420:435, 900:1020] = (255, 255, 255)
    frame[535:575, 900:1020] = (255, 255, 0)  # BGR cyan, nameplate ROI (y2=530) dışında
    return frame


def _valid_gray_frame() -> np.ndarray:
    frame = np.zeros((1080, 1920), dtype=np.uint8)
    frame[420:435, 900:1020] = 255
    frame[535:575:2, 900:1020:2] = 255
    return frame


def test_contested_valid_frames_still_detected_color_and_gray() -> None:
    """Kısa devreler geçerli contested tespitini bozmaz (renkli + gri)."""
    det = ContestedNodeDetector()
    assert det.is_node_contested(_valid_color_frame(), NODE_XY) is True
    assert det.is_node_contested(_valid_gray_frame(), NODE_XY) is True


def test_contested_blank_frames_return_false() -> None:
    """Boş kare: isimlik yok -> False (kısa devre yolu)."""
    det = ContestedNodeDetector()
    assert det.is_node_contested(np.zeros((1080, 1920, 3), dtype=np.uint8), NODE_XY) is False
    assert det.is_node_contested(np.zeros((1080, 1920), dtype=np.uint8), NODE_XY) is False


def test_contested_vertical_bar_is_not_a_nameplate() -> None:
    """Dikey çubuk (genişlik 3 px) yatay isimlik değildir; parıltı olsa da False."""
    det = ContestedNodeDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[420:460, 950:953] = (255, 255, 255)
    frame[535:575, 900:1020] = (255, 255, 0)
    assert det.is_node_contested(frame, NODE_XY) is False

    gray = np.zeros((1080, 1920), dtype=np.uint8)
    gray[420:460, 950:953] = 255
    gray[535:575:2, 900:1020:2] = 255
    assert det.is_node_contested(gray, NODE_XY) is False


def test_contested_tiny_speck_is_not_a_nameplate() -> None:
    """10 px'lik parlak benek (min alan 24'ün altı) -> False."""
    det = ContestedNodeDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[425, 900:910] = (255, 255, 255)
    frame[535:575, 900:1020] = (255, 255, 0)
    assert det.is_node_contested(frame, NODE_XY) is False


def test_contested_nameplate_without_gather_glow_is_false() -> None:
    """İsimlik var ama gather parıltısı yok -> False (aktivite şartı korunur)."""
    det = ContestedNodeDetector()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[420:435, 900:1020] = (255, 255, 255)
    assert det.is_node_contested(frame, NODE_XY) is False


def _stats_for(width: int, height: int, area: int) -> np.ndarray:
    # connectedComponentsWithStats: [left, top, width, height, area]; satır 0 arka plan.
    return np.array([[0, 0, 0, 0, 0], [0, 0, width, height, area]], dtype=np.int32)


def test_nameplate_shape_helper_boundaries() -> None:
    """Yardımcı, eski iki kopyadaki kuralın sınır değerlerini aynen uygular."""
    assert _has_valid_nameplate_blob(2, _stats_for(18, 3, 24)) is True  # sınırda geçerli
    assert _has_valid_nameplate_blob(2, _stats_for(17, 3, 24)) is False  # genişlik eksik
    assert _has_valid_nameplate_blob(2, _stats_for(18, 2, 24)) is False  # yükseklik eksik
    assert _has_valid_nameplate_blob(2, _stats_for(18, 3, 23)) is False  # alan eksik
    assert _has_valid_nameplate_blob(2, _stats_for(18, 10, 200)) is False  # en/boy 1.8 < 2.0
    assert _has_valid_nameplate_blob(2, _stats_for(20, 10, 200)) is True  # en/boy 2.0
    assert _has_valid_nameplate_blob(1, _stats_for(18, 3, 24)) is False  # arka plandan başka blob yok


def _small_frame() -> np.ndarray:
    return np.zeros((200, 300, 3), dtype=np.uint8)


def test_exemplar_memory_load_failure_tried_once_and_flagged() -> None:
    """Yükleme başarısızlığı: 3 çağrıda `get_exemplar_memory` 1 kez denenir,
    her sonuçta `exemplar_memory_error=True` kalır (uyarı seli yok)."""
    nav = CorridorNavigator()
    with patch(
        "dataset_factory.exemplar_memory.get_exemplar_memory",
        side_effect=RuntimeError("simulated load failure"),
    ) as mocked:
        results = [nav.evaluate_corridor_path(_small_frame(), yolo_detections=[]) for _ in range(3)]
    assert mocked.call_count == 1
    assert all(r["exemplar_memory_error"] is True for r in results)
    assert all(r["exemplar_memory_available"] is False for r in results)


def test_exemplar_memory_failure_retried_after_retry_window() -> None:
    """Yeniden deneme penceresi (varsayılan 60 sn) dolunca tekrar denenir."""
    nav = CorridorNavigator()
    with patch(
        "dataset_factory.exemplar_memory.get_exemplar_memory",
        side_effect=RuntimeError("simulated load failure"),
    ) as mocked:
        nav.evaluate_corridor_path(_small_frame(), yolo_detections=[])
        assert mocked.call_count == 1
        nav.exemplar_memory_retry_sec = 0.0  # pencereyi kapat: hemen yeniden dene
        nav.evaluate_corridor_path(_small_frame(), yolo_detections=[])
    assert mocked.call_count == 2


def test_exemplar_memory_success_cached_per_navigator() -> None:
    """Başarılı yükleme bir kez çözülür; 5 çağrıda da aynı tutamaç kullanılır."""
    fake_mem = MagicMock()
    fake_mem.is_negative_patch.return_value = (False, 0.0, None)
    fake_mem.match_class.return_value = (0.0, None)
    fake_mem.classify_patch.return_value = (None, None, 0.0)
    nav = CorridorNavigator()
    with patch(
        "dataset_factory.exemplar_memory.get_exemplar_memory", return_value=fake_mem
    ) as mocked:
        results = [nav.evaluate_corridor_path(_small_frame(), yolo_detections=[]) for _ in range(5)]
    assert mocked.call_count == 1
    assert all(r["exemplar_memory_available"] is True for r in results)
    assert all(r["exemplar_memory_error"] is False for r in results)
    assert all(r == results[0] for r in results)
