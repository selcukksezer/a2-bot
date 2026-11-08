from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.corridor_navigator import CorridorNavigator


def _blank_frame(h: int = 200, w: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_obstacle_class_14_triggers_steer_left(_mock_mem) -> None:
    """Merkez ve sağ şeritte 14: obstacle algılandığında steer_left kaçınması doğrulanır."""
    nav = CorridorNavigator()
    frame = _blank_frame()
    detections = [
        {"class_id": 14, "class_name": "obstacle", "confidence": 0.90, "cx": 0.50, "cy": 0.65},
        {"class_id": 14, "class_name": "obstacle", "confidence": 0.85, "cx": 0.65, "cy": 0.65},
    ]

    res = nav.evaluate_corridor_path(frame, yolo_detections=detections)

    assert res["obstacle_detected"] is True
    assert res["monster_detected"] is False
    assert res["steering_bias"] == "steer_left"
    assert res["recommended_turn_offset_deg"] == -18.0
    assert res["lane_scores"]["left"] > res["lane_scores"]["center"]
    assert res["lane_scores"]["left"] > res["lane_scores"]["right"]


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_obstacle_class_14_triggers_steer_right(_mock_mem) -> None:
    """Merkez ve sol şeritte 14: obstacle algılandığında steer_right kaçınması doğrulanır."""
    nav = CorridorNavigator()
    frame = _blank_frame()
    detections = [
        {"class_id": 14, "class_name": "obstacle", "confidence": 0.90, "cx": 0.50, "cy": 0.68},
        {"class_id": 14, "class_name": "obstacle", "confidence": 0.85, "cx": 0.35, "cy": 0.68},
    ]

    res = nav.evaluate_corridor_path(frame, yolo_detections=detections)

    assert res["obstacle_detected"] is True
    assert res["steering_bias"] == "steer_right"
    assert res["recommended_turn_offset_deg"] == 18.0
    assert res["lane_scores"]["right"] > res["lane_scores"]["center"]
    assert res["lane_scores"]["right"] > res["lane_scores"]["left"]


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_monster_3d_class_10_avoidance_steer_left_and_right(_mock_mem) -> None:
    """10: monster_3d algılandığında steer_left / steer_right kaçınması ve steer_step açı düzeltmesi doğrulanır."""
    nav = CorridorNavigator()
    frame = _blank_frame()

    # Merkezde monster_3d varken sol ve sağ eşitse (l_score >= r_score) steer_left (-20°)
    dets_left = [
        {"class_id": 10, "class_name": "monster_3d", "confidence": 0.90, "cx": 0.50, "cy": 0.65},
    ]
    res_left = nav.evaluate_corridor_path(frame, yolo_detections=dets_left)
    assert res_left["monster_detected"] is True
    assert res_left["steering_bias"] == "steer_left"
    assert res_left["recommended_turn_offset_deg"] == -20.0

    # Merkez ve solda monster_3d varken sağ şerit açıksa steer_right (+20°)
    dets_right = [
        {"class_id": 10, "class_name": "monster_3d", "confidence": 0.90, "cx": 0.50, "cy": 0.65},
        {"class_id": 10, "class_name": "monster_3d", "confidence": 0.85, "cx": 0.32, "cy": 0.65},
    ]
    res_right = nav.evaluate_corridor_path(frame, yolo_detections=dets_right)
    assert res_right["monster_detected"] is True
    assert res_right["steering_bias"] == "steer_right"
    assert res_right["recommended_turn_offset_deg"] == 20.0

    # steer_step entegrasyonu: Kuzey (0°) hedefine giderken +20° sağa kaçınma
    step = nav.steer_step(
        current_pos=(100.0, 100.0),
        target_pos=(100.0, 50.0),
        player_heading=0.0,
        frame=frame,
        yolo_detections=dets_right,
    )
    assert step["target_bearing"] == 0.0
    assert step["adjusted_bearing"] == 20.0
    assert step["turn_direction"] == "right"


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_path_ground_class_15_lane_following(_mock_mem) -> None:
    """15: path_ground algılandığında patika bulunan şeride (steer_left / steer_right) yönelim doğrulanır."""
    nav = CorridorNavigator()
    frame = _blank_frame()

    # Sol şeritte yürünebilir patika (15: path_ground)
    dets_path_left = [
        {"class_id": 15, "class_name": "path_ground", "confidence": 0.90, "cx": 0.34, "cy": 0.70},
    ]
    res_left = nav.evaluate_corridor_path(frame, yolo_detections=dets_path_left)
    assert res_left["path_detected"] is True
    assert res_left["obstacle_detected"] is False
    assert res_left["monster_detected"] is False
    assert res_left["steering_bias"] == "steer_left"
    assert res_left["recommended_turn_offset_deg"] == -10.0

    # Sağ şeritte yürünebilir patika (15: path_ground)
    dets_path_right = [
        {"class_id": 15, "class_name": "path_ground", "confidence": 0.90, "cx": 0.66, "cy": 0.70},
    ]
    res_right = nav.evaluate_corridor_path(frame, yolo_detections=dets_path_right)
    assert res_right["path_detected"] is True
    assert res_right["steering_bias"] == "steer_right"
    assert res_right["recommended_turn_offset_deg"] == 10.0


def test_exemplar_memory_3_lane_patch_matching_and_step_navigation() -> None:
    """ExemplarMemory üzerinden 3 şeritli yama analizinde 14, 10 ve 15 sınıflarının işlendiği doğrulanır."""
    nav = CorridorNavigator()
    frame = _blank_frame(100, 100)
    # Sol şerit (x: 28..44) -> 14: obstacle (kanal 0 = 14)
    # Merkez şerit (x: 44..56) -> 10: monster_3d (kanal 0 = 10)
    # Sağ şerit (x: 56..72) -> 15: path_ground (kanal 0 = 15)
    frame[52:82, 28:44, 0] = 14
    frame[52:82, 44:56, 0] = 10
    frame[52:82, 56:72, 0] = 15

    fake_mem = MagicMock()
    fake_mem.is_negative_patch.return_value = (False, 0.0, None)
    fake_mem.classify_patch.return_value = (None, None, 0.0)

    def _match_class(patch: np.ndarray, class_id: int):
        marker = int(patch[0, 0, 0])
        if marker == class_id:
            return 0.85, "matched"
        return 0.0, None

    fake_mem.match_class.side_effect = _match_class

    with patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=fake_mem):
        res = nav.evaluate_corridor_path(frame)

    assert res["obstacle_detected"] is True
    assert res["monster_detected"] is True
    assert res["path_detected"] is True
    assert res["steering_bias"] == "steer_right"
    assert res["recommended_turn_offset_deg"] == 20.0
    assert res["lane_scores"]["right"] > res["lane_scores"]["left"]
    assert res["lane_scores"]["right"] > res["lane_scores"]["center"]
