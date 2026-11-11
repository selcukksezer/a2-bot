from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.corridor_navigator import CorridorNavigator


def _blank_frame(h: int = 200, w: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_clear_path_no_detections_goes_straight(_mock_mem) -> None:
    """Hiç tespit yokken koridor temizdir: şeritler eşit (0.5), sapma 0°."""
    nav = CorridorNavigator()
    frame = _blank_frame()

    res = nav.evaluate_corridor_path(frame)

    assert res["obstacle_detected"] is False
    assert res["monster_detected"] is False
    assert res["path_detected"] is False
    assert res["steering_bias"] == "clear"
    assert res["recommended_turn_offset_deg"] == 0.0
    assert res["lane_scores"] == {"left": 0.5, "center": 0.5, "right": 0.5}


@patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=None)
def test_obstacle_center_only_tie_breaks_steer_left(_mock_mem) -> None:
    """Yalnızca merkez şeritte 14: obstacle varken sol/sağ eşit (0.5) kalır;
    eşitlik steer_left (-18°) ile bozulur (monster testindeki l>=r kuralıyla tutarlı)."""
    nav = CorridorNavigator()
    frame = _blank_frame()
    detections = [
        {"class_id": 14, "class_name": "obstacle", "confidence": 0.90, "cx": 0.50, "cy": 0.65},
    ]

    res = nav.evaluate_corridor_path(frame, yolo_detections=detections)

    assert res["obstacle_detected"] is True
    assert res["monster_detected"] is False
    assert res["steering_bias"] == "steer_left"
    assert res["recommended_turn_offset_deg"] == -18.0
    assert res["lane_scores"]["left"] == res["lane_scores"]["right"] == 0.5
    assert res["lane_scores"]["center"] < res["lane_scores"]["left"]


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
    """ExemplarMemory üzerinden 3 şeritli yama analizinde 14, 10 ve 15 sınıflarının işlendiği doğrulanır.

    Gerçek `match_class(patch, class_id)` protokolü doğrulanır: üretim kodu her şerit
    yaması için `match_class`'i 14 (obstacle), 10 (monster_3d) ve 15 (path_ground)
    ile ayrı ayrı çağırır; eşleşme yalnızca yamanın görsel içeriği o sınıfa aitse
    yüksek benzerlik döner (sınıf filtresi — tıpkı gerçek ExemplarMemory'nin
    yalnızca o `class_id`'ye ait exemplar'larla karşılaştırması gibi).
    """
    nav = CorridorNavigator()
    frame = _blank_frame(100, 100)
    # Sol şerit (x: 28..44) -> 14: obstacle, merkez (x: 44..56) -> 10: monster_3d,
    # sağ şerit (x: 56..72) -> 15: path_ground. Sınıf kimliği yamanın kanal-0
    # içeriğinde (baskın değer) saklanır; tek piksel işaretçisi kullanılmaz.
    frame[52:82, 28:44, 0] = 14
    frame[52:82, 44:56, 0] = 10
    frame[52:82, 56:72, 0] = 15

    fake_mem = MagicMock()
    fake_mem.is_negative_patch.return_value = (False, 0.0, None)
    fake_mem.classify_patch.return_value = (None, None, 0.0)

    def _match_class(patch: np.ndarray, class_id: int):
        # Gerçek protokol: yamanın içeriğinden çıkarılan özellik, yalnızca
        # istenen class_id'nin exemplar'larıyla karşılaştırılır. Burada yamanın
        # baskın kanal-0 değeri o şeridin görsel içeriğini temsil eder.
        dominant = int(np.bincount(patch[:, :, 0].ravel(), minlength=256).argmax())
        if dominant == int(class_id):
            return 0.85, {"class_id": int(class_id), "similarity": 0.85}
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

    # Protokol kanıtı: her şerit için 14, 10 ve 15 ayrı ayrı sorulmuş olmalı
    # (3 şerit x 3 sınıf = 9 çağrı) ve her çağrı (patch, class_id) biçiminde olmalı.
    assert fake_mem.match_class.call_count == 9
    called_class_ids = []
    for call in fake_mem.match_class.call_args_list:
        if "class_id" in call.kwargs:
            called_class_ids.append(call.kwargs["class_id"])
        else:
            called_class_ids.append(call.args[1])
    assert sorted(called_class_ids) == [10, 10, 10, 14, 14, 14, 15, 15, 15]
    for call in fake_mem.match_class.call_args_list:
        called_patch = call.args[0]
        assert isinstance(called_patch, np.ndarray)
        assert called_patch.ndim == 3

    # Sınıf filtresi kanıtı: aynı yama, içeriğiyle eşleşmeyen sınıfta 0.0 döner.
    left_patch = frame[52:82, 28:44]
    assert _match_class(left_patch, 14)[0] == 0.85
    assert _match_class(left_patch, 10)[0] == 0.0
    assert _match_class(left_patch, 15)[0] == 0.0
