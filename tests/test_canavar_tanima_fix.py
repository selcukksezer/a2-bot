# -*- coding: utf-8 -*-
"""Canavar tanıma düzeltmesi (fix/canavar-tanima) — kilit testleri.

Kapsanan kopukluklar (hepsi diskten doğrulandı, model sorgulanmıyor):
  1. ONNX çıkarımı sınıf adını 3 girdili haritadan çeviriyordu: eğitimli
     her sınıf (monster_3d=10 dahil) "class_N" oluyor, class_id alanı hiç
     yoktu -> reflex/hub/state machine canavarı göremiyordu.
  2. Reflex _class_id eski "class_N" isimlerini -1 sayıyordu.
  3. PerceptionHub._fill_targets detector'da olmayan detect()/infer()
     çağırıyordu (gerçek giriş detect_3d_objects) ve canavar kontrolü
     yalnızca "label"/"class" anahtarına bakıyordu.
  4. vision.detect_monsters_3d'nin üretimde hiç çağırıcısı yoktu.
  5. Canlı bot is_model_stale'u hiç tüketmiyordu; detector yüklenirken
     stale uyarısı vermiyor ve .onnx değişince session'ı yenilemiyordu.

NOT: Bu dosyadaki testler YAZILDI, KOŞULMADI (görev kuralı: pytest yok).
Koşturacak ajan: `pytest tests/test_canavar_tanima_fix.py -v`.
"""
import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_audio_perception import (  # noqa: E402
    ONNXYolo3DDetector,
    canonical_class_label_tr,
    canonical_class_name,
)
from core.perception_hub import FrameCognition, PerceptionHub  # noqa: E402
from core.reflex_layer import _class_id, state_from_boxes  # noqa: E402


def _detector_without_model():
    """Gerçek model dosyası olmadan (fallback backend) detector."""
    return ONNXYolo3DDetector(model_path="definitely_not_a_model.onnx")


# ----------------------------------------------------------------------
# 1. Kanonik sınıf adı + class_id (neural_audio_perception)
# ----------------------------------------------------------------------
class TestCanonicalClassNames:
    def test_monster_3d_is_class_10(self):
        assert canonical_class_name(10) == "monster_3d"

    def test_gather_classes_keep_taxonomy_names(self):
        assert canonical_class_name(0) == "herb_aria"
        assert canonical_class_name(7) == "herb_lumina"
        assert canonical_class_name(8) == "ore_obsidian"
        assert canonical_class_name(12) == "crystal_blood"

    def test_all_27_classes_named(self):
        for cid in range(27):
            name = canonical_class_name(cid)
            assert name != f"class_{cid}", f"class {cid} still unnamed"

    def test_unknown_class_falls_back_to_class_n(self):
        assert canonical_class_name(99) == "class_99"

    def test_turkish_label(self):
        assert canonical_class_label_tr(10, "monster") == "3B Canavar"
        assert canonical_class_label_tr(99, "bilinmiyor") == "bilinmiyor"


class TestOnnxResultMapping:
    """Sahte ORT session ile _run_onnx_inference: sınıf 10 tespiti
    class_id + kanonik isim taşımalı (eski davranış: "class_10", id yok)."""

    def _detector_with_fake_session(self):
        det = _detector_without_model()
        raw = np.zeros((1, 31, 100), dtype=np.float32)
        # (4+C, N) düzeni: satır 0..3 kutu, satır 4+cid sınıf skoru.
        raw[0, 0, 0] = 320.0  # cx (letterbox koordinatı)
        raw[0, 1, 0] = 320.0  # cy
        raw[0, 2, 0] = 100.0  # w
        raw[0, 3, 0] = 80.0  # h
        raw[0, 4 + 10, 0] = 0.9  # monster_3d skoru

        class _FakeSession:
            def run(self, _outputs, inputs):
                return [raw]

        det._ort_session = _FakeSession()
        det._ort_input_name = "images"
        det._ort_output_shape = (1, 31, 100)
        det._cv_net = None
        det.backend_name = "onnxruntime:CPUExecutionProvider"
        return det

    def test_monster_detection_carries_class_id_and_name(self):
        det = self._detector_with_fake_session()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det._run_onnx_inference(frame, time.perf_counter())
        assert len(results) == 1
        det0 = results[0]
        assert det0["class_id"] == 10
        assert det0["class_name"] == "monster_3d"
        assert det0["sub_type"] == "monster_3d"
        assert det0["category_tr"] == "3B Canavar"
        assert det0["confidence"] == pytest.approx(0.9)


# ----------------------------------------------------------------------
# 2. Reflex _class_id + state_from_boxes
# ----------------------------------------------------------------------
class TestReflexClassId:
    def test_legacy_class_n_names_parse_to_id(self):
        assert _class_id({"class_name": "class_10"}) == 10
        assert _class_id({"class_name": "class_7"}) == 7
        assert _class_id({"class_name": "class_8"}) == 8

    def test_canonical_monster_name(self):
        assert _class_id({"class_name": "monster_3d"}) == 10
        assert _class_id({"class_id": 10, "class_name": "monster_3d"}) == 10

    def test_fallback_monster_name_unchanged(self):
        assert _class_id({"class_name": "approaching_monster"}) == 10

    def test_gather_names_unchanged(self):
        assert _class_id({"class_name": "herb_flora"}) == 12

    def test_unknown_still_minus_one(self):
        assert _class_id({"class_name": "something_else"}) == -1


class TestReflexStateFromBoxes:
    def test_onnx_monster_box_means_combat(self):
        boxes = [
            {
                "class_id": 10,
                "class_name": "monster_3d",
                "confidence": 0.9,
                "bbox": (100, 100, 80, 60),
            }
        ]
        assert state_from_boxes(boxes) == "Savaşta"

    def test_legacy_class_10_name_means_combat(self):
        assert state_from_boxes([{"class_name": "class_10"}]) == "Savaşta"

    def test_canonical_name_without_id_means_combat(self):
        assert state_from_boxes([{"class_name": "monster_3d"}]) == "Savaşta"

    def test_fallback_monster_name_still_combat(self):
        assert state_from_boxes([{"class_name": "approaching_monster"}]) == "Savaşta"

    def test_gather_and_empty_unchanged(self):
        assert state_from_boxes([{"class_name": "herb_flora"}]) == "Topluyor"
        assert state_from_boxes([]) == "Arıyor"


# ----------------------------------------------------------------------
# 3. Stale bayrağı (salt okunur) + hot-reload
# ----------------------------------------------------------------------
class TestStaleFlag:
    def test_stale_manifest_true_is_consumed(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "stale_manifest.json").write_text(
            json.dumps(
                {"model_stale": True, "reason": "Etiketler güncellendi: test"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        det = _detector_without_model()
        # dataset_validator yolu (models manifest'i öncelikli okur) ya da
        # doğrudan manifest okuma: her ikisi de True vermeli.
        stale, reason = det._check_model_stale(models_dir=str(models_dir))
        assert stale is True
        assert det.model_stale is True
        assert "test" in det.model_stale_reason

    def test_read_stale_manifest_missing_is_fresh(self, tmp_path):
        det = _detector_without_model()
        assert det._read_stale_manifest(str(tmp_path)) == (False, "Model güncel")

    def test_read_stale_manifest_false_is_fresh(self, tmp_path):
        (tmp_path / "stale_manifest.json").write_text(
            json.dumps({"model_stale": False}), encoding="utf-8"
        )
        det = _detector_without_model()
        stale, _reason = det._read_stale_manifest(str(tmp_path))
        assert stale is False

    def test_read_stale_manifest_corrupt_fails_closed(self, tmp_path):
        (tmp_path / "stale_manifest.json").write_text(
            "{bozuk json", encoding="utf-8"
        )
        det = _detector_without_model()
        stale, _reason = det._read_stale_manifest(str(tmp_path))
        assert stale is True

    def test_get_model_status_reports_stale(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "stale_manifest.json").write_text(
            json.dumps({"model_stale": True, "reason": "x"}), encoding="utf-8"
        )
        det = _detector_without_model()
        det._check_model_stale(models_dir=str(models_dir))
        status = det.get_model_status()
        assert status["model_stale"] is True
        assert status["model_loaded"] is False  # model dosyası yok


class TestHotReload:
    def _detector_with_tmp_model(self, tmp_path, content=b"fake-onnx-v1"):
        model_file = tmp_path / "yolov8n_gatherion.onnx"
        model_file.write_bytes(content)
        det = _detector_without_model()
        det._resolved_model_path = str(model_file)
        det._loaded_mtime = None  # hiç yüklenmemiş gibi
        det.reload_check_interval_sec = 0.0  # throttle'ı testte kapat
        return det, model_file

    def test_no_model_file_returns_false(self):
        det = _detector_without_model()
        det.reload_check_interval_sec = 0.0
        assert det.check_for_model_update() is False

    def test_unchanged_file_returns_false(self, tmp_path):
        det, _f = self._detector_with_tmp_model(tmp_path)
        det._record_loaded_file(det._resolved_model_path)
        assert det.check_for_model_update() is False

    def test_successful_reload_swaps_session(self, tmp_path, monkeypatch):
        det, _f = self._detector_with_tmp_model(tmp_path)
        sentinel_session = object()
        built = {
            "ort_session": sentinel_session,
            "ort_input_name": "images",
            "ort_output_shape": (1, 31, 8400),
            "cv_net": None,
            "backend_name": "onnxruntime:CPUExecutionProvider",
        }
        monkeypatch.setattr(det, "_build_session", lambda _b: built)
        assert det.check_for_model_update() is True
        assert det._ort_session is sentinel_session
        assert det.backend_name == "onnxruntime:CPUExecutionProvider"
        assert det._loaded_mtime is not None

    def test_failed_reload_keeps_old_session(self, tmp_path, monkeypatch):
        det, model_file = self._detector_with_tmp_model(tmp_path)
        old_session = object()
        det._ort_session = old_session
        det._loaded_mtime = 1.0  # eski imza -> dosya "değişmiş" sayılır
        det._loaded_size = 1
        model_file.write_bytes(b"fake-onnx-v2-bozuk")
        monkeypatch.setattr(det, "_build_session", lambda _b: None)
        assert det.check_for_model_update() is False
        assert det._ort_session is old_session

    def test_throttle_blocks_immediate_recheck(self, tmp_path, monkeypatch):
        det, _f = self._detector_with_tmp_model(tmp_path)
        det.reload_check_interval_sec = 60.0
        det._last_reload_check = time.monotonic()
        monkeypatch.setattr(
            det, "_build_session", lambda _b: (_ for _ in ()).throw(AssertionError("must not build"))
        )
        assert det.check_for_model_update() is False


# ----------------------------------------------------------------------
# 4. PerceptionHub: YOLO canavar hedefi + vision yedeği
# ----------------------------------------------------------------------
class _FakeYoloDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect_3d_objects(self, _frame):
        self.calls += 1
        return list(self.boxes)


class _FakeVision:
    def __init__(self, monsters):
        self.monsters = monsters
        self.calls = 0

    def detect_monsters_3d(self, _frame):
        self.calls += 1
        return list(self.monsters)


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestPerceptionHubMonsterTargets:
    def test_yolo_monster_sets_has_target(self):
        det = _FakeYoloDetector(
            [
                {
                    "class_id": 10,
                    "class_name": "monster_3d",
                    "confidence": 0.9,
                    "bbox": (100, 100, 80, 60),
                }
            ]
        )
        hub = PerceptionHub(yolo_detector=det, vision=None, combat_recovery=None)
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is True
        assert det.calls == 1

    def test_yolo_monster_by_name_sets_has_target(self):
        # class_id alanı olmayan eski üretici: isim eşleşmesi yeterli.
        det = _FakeYoloDetector(
            [{"class_name": "approaching_monster", "confidence": 0.8}]
        )
        hub = PerceptionHub(yolo_detector=det, vision=None, combat_recovery=None)
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is True

    def test_yolo_gather_box_does_not_set_has_target(self):
        det = _FakeYoloDetector(
            [{"class_id": 0, "class_name": "herb_aria", "confidence": 0.9}]
        )
        hub = PerceptionHub(yolo_detector=det, vision=None, combat_recovery=None)
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is False

    def test_vision_fallback_sets_has_target_when_yolo_empty(self):
        det = _FakeYoloDetector([])
        vision = _FakeVision(
            [
                {
                    "class_id": 10,
                    "class_name": "monster_3d",
                    "confidence": 0.88,
                    "bbox": (10, 20, 30, 40),
                    "screen_x": 25,
                    "screen_y": 40,
                    "is_hostile": True,
                    "is_targeted": False,
                    "has_diamond_brackets": False,
                }
            ]
        )
        hub = PerceptionHub(yolo_detector=det, vision=vision, combat_recovery=None)
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is True
        assert vision.calls == 1
        assert cog.targets[-1]["source"] == "vision.detect_monsters_3d"
        assert cog.targets[-1]["class_id"] == 10

    def test_vision_fallback_skipped_when_yolo_found_monster(self):
        det = _FakeYoloDetector(
            [{"class_id": 10, "class_name": "monster_3d", "confidence": 0.9}]
        )
        vision = _FakeVision([{"class_id": 10, "is_hostile": True}])
        hub = PerceptionHub(yolo_detector=det, vision=vision, combat_recovery=None)
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is True
        assert vision.calls == 0

    def test_vision_fallback_throttled_to_one_per_second(self):
        det = _FakeYoloDetector([])
        vision = _FakeVision([{"class_id": 10, "is_hostile": True}])
        hub = PerceptionHub(yolo_detector=det, vision=vision, combat_recovery=None)
        hub._fill_targets(_frame(), FrameCognition())
        hub._fill_targets(_frame(), FrameCognition())
        assert vision.calls == 1

    def test_old_detect_api_still_works(self):
        class _OldApiDetector:
            def detect(self, _frame):
                return [{"label": "monster", "confidence": 0.7}]

        hub = PerceptionHub(
            yolo_detector=_OldApiDetector(), vision=None, combat_recovery=None
        )
        cog = FrameCognition()
        hub._fill_targets(_frame(), cog)
        assert cog.has_target is True
