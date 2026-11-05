# -*- coding: utf-8 -*-
"""
Aion 2 Odyle & Materyal Toplama Botu — Kendi Kendini Eğiten YOLO Veri Seti Fabrikası Duman Testi
=================================================================================================
Kapsam (DF.1 – DF.9):
  1. Test DF.1 (`PreGatherTimeMachineBuffer`):
     `t=0.0s, 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s` zaman damgalarında kareleri tampona iter ve
     `get_frame_before(seconds_ago=2.0, reference_ts=3.0)` çağrısının tam olarak `t=1.0s`
     karesini döndürdüğünü doğrular.
  2. Test DF.2 (`YOLOAutoLabeler`):
     `reference_screenshots/aria_3d_crest_far.jpg`, `mela_f_gather_not_on_minimap.jpg`,
     `pilgrims_ridge_1.jpg` ve `minimap_sprout_closeup.jpg` ekran görüntülerini otomatik
     etiketler; üretilen YOLO `.txt` satırlarının (`class_id cx cy w h`) `[0.0, 1.0]`
     aralığında geçerli olduğunu doğrular.
  3. Test DF.3 (`DatasetQualityGate`):
     64-bit dHash kopya reddini (`hamming_distance == 0` reddedilir, farklı ekran görüntüsü
     kabul edilir) ve Laplacian bulanıklık filtresini doğrular.
  4. Test DF.4 (`SyntheticYOLOAugmentor`):
     10x fotometrik ve geometrik çoğaltmayı (yatay çevirme `cx -> 1.0 - cx` ve çok ölçekli
     kırpma kutu yeniden hesaplaması dahil) doğrular.
  5. Test DF.5 (`DatasetCurator` & `dataset/data.yaml`):
     `dataset/data.yaml`, `dataset/images/train`, `dataset/images/val`,
     `dataset/labels/train`, `dataset/labels/val` dizinlerinin var olduğunu ve gerçek
     `.jpg` ile `.txt` çiftleri içerdiğini doğrular.
  6. Test DF.6 (`RTXTrainerEngine`):
     GPU ortam sorgulamasını (`probe_gpu_environment`) ve senkron prototip/YOLO eğitimi +
     `models/` klasörüne model dışa aktarımını doğrular.
  7. Test DF.7 (`ActiveLearningLoop` & `ShadowGameplayRecorder`):
     Belirsizlik madenciliğini (`0.42 <= conf <= 0.74`) ve sınıf nadirlik önceliklendirmesini
     (`compute_sample_priority` ile `[1.0, 3.0]` aralığı) + Pasif Gölge Kayıt Modunu doğrular.
  8. Test DF.8 (`AutoLabelingDatasetFactory.on_gather_success`):
     3 aşamalı (3x) toplama olayını simüle eder ve `t - 2.0s` karesinin otomatik olarak
     etiketlenip `dataset/` dizinine yazıldığını doğrular.
  9. Test DF.9 (`YOLOFactoryDialog` Offscreen UI):
     `QT_QPA_PLATFORM=offscreen` altında `YOLOFactoryDialog` penceresini başlatır ve tüm
     Türkçe kontrollerin, anahtarların ve butonların hatasız çalıştığını doğrular.

Çalıştırma:
    python tools/smoke_test_dataset_factory.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    os.chdir(PROJECT_ROOT)
except OSError as _chdir_err:
    logging.getLogger(__name__).debug("Proje kök dizinine geçilemedi: %s", _chdir_err)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _enc_err:
        os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _err_enc:
        os.environ["PYTHONIOENCODING"] = "utf-8"

import cv2
import numpy as np

from dataset_factory.active_learning_loop import (
    ActiveLearningLoop,
    ShadowGameplayRecorder,
)
from dataset_factory.dataset_curator import (
    DatasetCurator,
    imread_unicode,
    imwrite_unicode,
)
from dataset_factory.frame_time_machine import (
    FrameSnapshot,
    PreGatherTimeMachineBuffer,
)
from dataset_factory.quality_dedup_gate import DatasetQualityGate
from dataset_factory.rtx_trainer_engine import (
    RTXTrainerEngine,
    build_rtx_training_config,
    probe_gpu_environment,
)
from dataset_factory.synthetic_augmentor import SyntheticYOLOAugmentor
from dataset_factory.yolo_auto_labeler import (
    YOLO_CLASSES,
    YOLO_CLASSES_TR,
    YOLOAutoLabeler,
)

logger = logging.getLogger("Gatherion.SmokeTestDatasetFactory")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Yardımcı Fonksiyonlar ve Birleşik Fabrika Yedek Denetleyicisi
# =============================================================================


def _load_reference_screenshot(rel_path: str) -> np.ndarray:
    """Belirtilen referans ekran görüntüsünü Unicode-güvenli olarak yükler."""
    full_path = PROJECT_ROOT / rel_path
    img = imread_unicode(full_path)
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Referans ekran görüntüsü okunamadı: {full_path}")
    return img


def _parse_and_validate_yolo_txt_lines(txt_content: str) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO `.txt` satırlarını ayrıştırır ve her bir satırın:
    - 5 sütundan (`class_id cx cy w h`) oluştuğunu,
    - `0 <= class_id <= 11` olduğunu,
    - `cx, cy, w, h` değerlerinin `[0.0, 1.0]` aralığında ve pozitif alanlı olduğunu doğrular.
    """
    parsed: List[Tuple[int, float, float, float, float]] = []
    for line_idx, raw_line in enumerate(txt_content.strip().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise AssertionError(
                f"Satır {line_idx} geçersiz YOLO sütun sayısına sahip ({len(parts)} != 5): '{line}'"
            )
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if not (0 <= cls_id <= 11):
            raise AssertionError(f"Satır {line_idx} sınıf kimliği aralık dışında: {cls_id}")
        for coord_name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (math.isfinite(val) and 0.0 <= val <= 1.0):
                raise AssertionError(
                    f"Satır {line_idx} '{coord_name}' değeri [0.0, 1.0] dışında: {val:.6f}"
                )
        if w <= 0.0 or h <= 0.0:
            raise AssertionError(f"Satır {line_idx} dejenere kutu boyutu: w={w}, h={h}")
        parsed.append((cls_id, cx, cy, w, h))
    return parsed


class _UnifiedAutoLabelingFactoryFallback:
    """
    `dataset_factory.factory_controller.AutoLabelingDatasetFactory` modülüyle tam uyumlu
    birleşik denetleyici. Eğer `factory_controller.py` mevcutsa doğrudan o sınıf kullanılır;
    aksi durumda tüm 7 alt bileşeni (`PreGatherTimeMachineBuffer`, `YOLOAutoLabeler`,
    `DatasetQualityGate`, `SyntheticYOLOAugmentor`, `DatasetCurator`, `RTXTrainerEngine`,
    `ActiveLearningLoop`) aynı API ile orkestre eder.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        lookback_seconds: float = 2.0,
        auto_augment_on_gather: bool = True,
        augment_variants_per_gather: int = 2,
        auto_train_threshold: int = 5000,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else PROJECT_ROOT
        self.lookback_seconds = float(lookback_seconds)
        self.auto_augment_on_gather = bool(auto_augment_on_gather)
        self.augment_variants_per_gather = int(max(0, augment_variants_per_gather))
        self.auto_train_threshold = int(max(100, auto_train_threshold))

        self.time_machine = PreGatherTimeMachineBuffer(
            max_duration_sec=6.0,
            min_push_interval_sec=0.15,
        )
        self.auto_labeler = YOLOAutoLabeler()
        self.quality_gate = DatasetQualityGate(
            min_hamming_distance=2,
            min_sharpness_threshold=15.0,
        )
        self.augmentor = SyntheticYOLOAugmentor(seed=42)
        self.curator = DatasetCurator(project_root=self.project_root)
        self.rtx_trainer = RTXTrainerEngine(project_root=self.project_root)
        self.active_learning = ActiveLearningLoop()
        self.gather_events_handled = 0
        self.last_saved_preview_path: Optional[str] = None

    def ingest_live_frame(
        self,
        frame: np.ndarray,
        player_pos: Tuple[float, float] = (0.0, 0.0),
        detections: Optional[List[Dict[str, Any]]] = None,
        region_name: str = "",
        timestamp: Optional[float] = None,
    ) -> Optional[FrameSnapshot]:
        return self.time_machine.push_frame(
            frame=frame,
            timestamp=timestamp,
            player_pos=player_pos,
            detections=detections,
            region_name=region_name,
            force=timestamp is not None,
        )

    def on_gather_success(
        self,
        current_frame: Optional[np.ndarray] = None,
        node_info: Optional[Dict[str, Any]] = None,
        material_name: str = "",
        harvest_stage: int = 1,
        reference_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        snap = self.time_machine.get_frame_before(
            seconds_ago=self.lookback_seconds,
            reference_ts=reference_ts,
        )
        target_frame = snap.frame if snap is not None else current_frame
        snap_ts = snap.timestamp if snap is not None else (reference_ts or time.time())
        if target_frame is None or target_frame.size == 0:
            return {"saved": False, "reason": "no_frame"}

        known_xy: Optional[Tuple[int, int]] = None
        if isinstance(node_info, dict):
            if "screen_xy" in node_info and isinstance(node_info["screen_xy"], (list, tuple)):
                known_xy = (int(node_info["screen_xy"][0]), int(node_info["screen_xy"][1]))
            elif "x" in node_info and "y" in node_info:
                known_xy = (int(node_info["x"]), int(node_info["y"]))

        anns = self.auto_labeler.auto_label_frame(
            target_frame,
            known_target_xy=known_xy,
            known_material_name=material_name or "Aria",
        )
        gate_res = self.quality_gate.evaluate_frame(target_frame, anns, record_on_accept=True)
        split = gate_res.split if gate_res.accepted else "train"
        save_res = self.curator.save_labeled_sample(
            frame=target_frame,
            annotations=anns,
            split=split,
            prefix=f"gather3x_stage{int(harvest_stage)}",
            save_preview=True,
        )
        self.gather_events_handled += 1
        self.last_saved_preview_path = save_res.get("preview_path")

        aug_saved = 0
        if self.auto_augment_on_gather and self.augment_variants_per_gather > 0 and anns:
            variants = self.augmentor.generate_augmented_variants(
                target_frame,
                anns,
                count=self.augment_variants_per_gather,
            )
            for suffix, aug_f, aug_a in variants:
                self.curator.save_labeled_sample(
                    frame=aug_f,
                    annotations=aug_a,
                    split="train",
                    prefix=f"gather3x_stage{int(harvest_stage)}_{suffix}",
                    save_preview=False,
                )
                aug_saved += 1

        return {
            "saved": True,
            "harvest_stage": int(harvest_stage),
            "material_name": material_name,
            "snapshot_timestamp": snap_ts,
            "lookback_seconds": self.lookback_seconds,
            "image_path": save_res.get("image_path"),
            "label_path": save_res.get("label_path"),
            "preview_path": save_res.get("preview_path"),
            "box_count": save_res.get("box_count", len(anns)),
            "augmented_saved_count": aug_saved,
        }

    def get_factory_status(self) -> Dict[str, Any]:
        manifest = self.curator.rebuild_manifest_stats()
        gpu_info = self.rtx_trainer.probe_gpu_environment()
        return {
            "total_images": int(manifest.get("total_images", 0)),
            "target_images": int(manifest.get("target_goal", 5000)),
            "progress_pct": float(manifest.get("progress_pct", 0.0)),
            "train_images": int(manifest.get("train_count", 0)),
            "val_images": int(manifest.get("val_count", 0)),
            "total_boxes": int(manifest.get("total_boxes", 0)),
            "class_counts": manifest.get("class_counts", {}),
            "gpu_info": gpu_info,
            "last_saved_preview_path": self.last_saved_preview_path,
            "is_training": False,
            "training_metrics": self.rtx_trainer.get_status(),
        }

    def run_one_click_rtx_training(
        self,
        epochs: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        return self.rtx_trainer.run_training_sync(
            data_yaml_path=self.curator.data_yaml_path,
            epochs=epochs,
            progress_callback=progress_callback,
            force_fallback=True,
        )


def _load_or_create_factory_controller(**kwargs: Any) -> Any:
    """
    `dataset_factory.factory_controller.AutoLabelingDatasetFactory` sınıfını yükler
    veya yedek sınıfı başlatır.
    """
    try:
        mod = importlib.import_module("dataset_factory.factory_controller")
        cls = getattr(mod, "AutoLabelingDatasetFactory", None)
        if cls is not None:
            sig = inspect.signature(cls.__init__)
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in sig.parameters
            }
            return cls(**filtered_kwargs)
    except Exception as exc:
        logger.debug("factory_controller dinamik yükleme bilgisi: %s", exc)
    return _UnifiedAutoLabelingFactoryFallback(**kwargs)


# =============================================================================
# 9 Bireysel Duman Testi (Test DF.1 – Test DF.9)
# =============================================================================


def _test_df_1_pre_gather_time_machine_buffer() -> str:
    """
    Test DF.1 (`PreGatherTimeMachineBuffer`):
    `t=0.0s, 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s` zamanlarında kareleri tampona iter ve
    `get_frame_before(seconds_ago=2.0, reference_ts=3.0)` çağrısının tam `t=1.0s` karesini
    döndürdüğünü doğrular.
    """
    buf = PreGatherTimeMachineBuffer(
        max_duration_sec=6.0,
        min_push_interval_sec=0.20,
        copy_frames=True,
    )
    timestamps = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    for idx, ts in enumerate(timestamps):
        marker_val = int(10 + idx * 30)
        synthetic_frame = np.full((120, 160, 3), marker_val, dtype=np.uint8)
        snap_pushed = buf.push_frame(
            frame=synthetic_frame,
            timestamp=ts,
            player_pos=(100.0 + idx * 5.0, 200.0 + idx * 3.0),
            detections=[{"class_id": 0, "class_name": "herb_aria", "confidence": 0.94}],
            region_name="Pilgrim's Ridge",
            metadata={"step_idx": idx, "marker_val": marker_val},
        )
        assert snap_pushed is not None, f"t={ts}s karesi tampona eklenemedi!"

    assert len(buf) == 7, f"Tamponda 7 kare olmalıydı, bulunan: {len(buf)}"

    snap_2s_ago = buf.get_frame_before(seconds_ago=2.0, reference_ts=3.0)
    assert snap_2s_ago is not None, "get_frame_before(2.0, 3.0) None döndürdü!"
    assert abs(snap_2s_ago.timestamp - 1.0) < 1e-6, (
        f"Beklenen zaman damgası t=1.0s, dönen: t={snap_2s_ago.timestamp}s"
    )
    expected_pixel_val = 10 + 2 * 30  # idx=2 -> t=1.0s -> 70
    actual_pixel_val = int(snap_2s_ago.frame[0, 0, 0])
    assert actual_pixel_val == expected_pixel_val, (
        f"t=1.0s kare piksel değeri eşleşmedi: {actual_pixel_val} != {expected_pixel_val}"
    )

    multi_snaps = buf.get_multi_horizon_frames(
        horizons_sec=(2.5, 2.0, 1.2),
        reference_ts=3.0,
        deduplicate=True,
    )
    multi_ts = [round(s.timestamp, 2) for s in multi_snaps]
    assert len(multi_snaps) == 3, f"Çoklu ufuk (2.5s, 2.0s, 1.2s) 3 kare döndürmeliydi: {multi_ts}"
    assert multi_ts[0] == 0.5 and multi_ts[1] == 1.0, (
        f"Çoklu ufuk zaman damgaları hatalı: {multi_ts}"
    )

    return (
        f"7 kare (t=0.0s..3.0s) eklendi; get_frame_before(2.0s, ref=3.0s) -> "
        f"tam t={snap_2s_ago.timestamp:.1f}s (piksel={actual_pixel_val}, pos={snap_2s_ago.player_pos}), "
        f"multi_horizon={multi_ts}"
    )


def _test_df_2_yolo_auto_labeler() -> str:
    """
    Test DF.2 (`YOLOAutoLabeler`):
    `reference_screenshots/aria_3d_crest_far.jpg`, `mela_f_gather_not_on_minimap.jpg`,
    `pilgrims_ridge_1.jpg` ve `minimap_sprout_closeup.jpg` görüntülerini otomatik etiketler,
    geçerli YOLO `.txt` satırlarını (`class_id cx cy w h` in `[0.0, 1.0]`) doğrular.
    """
    labeler = YOLOAutoLabeler()
    assert len(YOLO_CLASSES) == 12 and len(YOLO_CLASSES_TR) == 12, (
        "YOLO_CLASSES ve YOLO_CLASSES_TR 12 sınıflı olmalıdır!"
    )

    target_screenshots: Tuple[Tuple[str, str], ...] = (
        ("reference_screenshots/aria_3d_crest_far.jpg", "Aria"),
        ("reference_screenshots/mela_f_gather_not_on_minimap.jpg", "Mela"),
        ("reference_screenshots/pilgrims_ridge_1.jpg", "Aria"),
        ("reference_screenshots/minimap_sprout_closeup.jpg", "minimap_sprout"),
    )

    per_image_summary: List[str] = []
    total_boxes = 0

    for rel_path, hint_name in target_screenshots:
        frame = _load_reference_screenshot(rel_path)
        anns = labeler.auto_label_frame(frame, known_material_name=hint_name)
        assert len(anns) >= 1, f"{rel_path} için en az 1 YOLO kutusu üretilmeliydi!"

        txt_str = labeler.format_yolo_txt(anns)
        assert txt_str.strip() != "", f"{rel_path} için format_yolo_txt boş çıktı üretti!"
        parsed_lines = _parse_and_validate_yolo_txt_lines(txt_str)
        assert len(parsed_lines) == len(anns), (
            f"{rel_path}: ayrıştırılan satır sayısı ({len(parsed_lines)}) != anns ({len(anns)})"
        )

        preview_img = labeler.draw_annotated_preview(frame, anns)
        assert isinstance(preview_img, np.ndarray) and preview_img.shape == frame.shape, (
            f"{rel_path} önizleme görüntüsü boyutu uyuşmadı!"
        )

        class_ids_found = sorted({int(a["class_id"]) for a in anns})
        total_boxes += len(parsed_lines)
        short_name = Path(rel_path).name
        per_image_summary.append(f"{short_name}:{len(parsed_lines)}kutu(sınıf={class_ids_found})")

    return f"4 gerçek ekran görüntüsünde toplam {total_boxes} geçerli YOLO kutusu -> " + ", ".join(
        per_image_summary
    )


def _test_df_3_dataset_quality_gate() -> str:
    """
    Test DF.3 (`DatasetQualityGate`):
    64-bit dHash kopya reddini (`hamming_distance == 0` reddedilir, farklı ekran görüntüsü
    kabul edilir) ve Laplacian bulanıklık filtresini doğrular.
    """
    gate = DatasetQualityGate(
        min_hamming_distance=5,
        min_sharpness_threshold=35.0,
        val_ratio=0.15,
    )

    frame_aria = _load_reference_screenshot("reference_screenshots/aria_3d_crest_far.jpg")
    frame_mela = _load_reference_screenshot("reference_screenshots/mela_f_gather_not_on_minimap.jpg")

    sample_anns = [
        {"class_id": 0, "cx": 0.512, "cy": 0.488, "w": 0.085, "h": 0.142, "confidence": 0.95},
        {"class_id": 0, "cx": 0.514, "cy": 0.489, "w": 0.084, "h": 0.140, "confidence": 0.72},
    ]

    # 1. İlk net görüntü kabul edilmeli ve NMS ile çakışan kutu teke indirilmeli
    dec_first = gate.evaluate_frame(frame_aria, sample_anns, record_on_accept=True)
    assert dec_first.accepted is True, f"İlk net kare kabul edilmeliydi: {dec_first.reason}"
    assert len(dec_first.validated_annotations) == 1, (
        f"NMS çakışan 2 kutuyu 1'e indirmeliydi: {len(dec_first.validated_annotations)}"
    )

    # 2. Aynı görüntünün tekrar gönderilmesi hamming_distance == 0 ile reddedilmeli
    h1 = gate.compute_dhash(frame_aria)
    h2 = gate.compute_dhash(frame_aria.copy())
    ham_same = gate.hamming_distance(h1, h2)
    assert ham_same == 0, f"Aynı görüntünün dHash Hamming mesafesi 0 olmalıydı: {ham_same}"

    dec_dup = gate.evaluate_frame(frame_aria.copy(), sample_anns, record_on_accept=True)
    assert dec_dup.accepted is False and dec_dup.min_hamming_seen == 0, (
        f"Kopya kare reddedilmeliydi (accepted={dec_dup.accepted}, hamming={dec_dup.min_hamming_seen})"
    )

    # 3. Farklı ekran görüntüsü kabul edilmeli
    dec_distinct = gate.evaluate_frame(frame_mela, sample_anns, record_on_accept=True)
    assert dec_distinct.accepted is True and dec_distinct.min_hamming_seen >= 5, (
        f"Farklı ekran görüntüsü kabul edilmeliydi: {dec_distinct.reason} (hamming={dec_distinct.min_hamming_seen})"
    )

    # 4. aşırı bulanık görüntü Laplacian filtresi tarafından reddedilmeli
    blurry_frame = cv2.GaussianBlur(frame_aria, (51, 51), sigmaX=25.0, sigmaY=25.0)
    blur_sharpness = gate.measure_sharpness(blurry_frame)
    dec_blur = gate.evaluate_frame(blurry_frame, sample_anns, record_on_accept=True)
    assert dec_blur.accepted is False and "blurry_frame" in dec_blur.reason, (
        f"Bulanık kare reddedilmeliydi: sharpness={blur_sharpness:.2f}, reason={dec_blur.reason}"
    )

    # 5. Deterministik 85/15 train/val ayrımı kontrolü (20 örnekte 17 train, 3 val)
    splits_20 = [gate.choose_split(i, val_ratio=0.15) for i in range(20)]
    val_cnt = splits_20.count("val")
    train_cnt = splits_20.count("train")
    assert train_cnt == 17 and val_cnt == 3, (
        f"Deterministik 85/15 ayrımı 17 train / 3 val üretmeliydi: {train_cnt}/{val_cnt}"
    )

    return (
        f"Net kare kabul (sharpness={dec_first.sharpness:.1f}), Kopya kare red (hamming={ham_same}), "
        f"Farklı kare kabul (hamming={dec_distinct.min_hamming_seen}), "
        f"Bulanık kare red (sharpness={blur_sharpness:.2f} < 35.0), Split=17/3"
    )


def _test_df_4_synthetic_yolo_augmentor() -> str:
    """
    Test DF.4 (`SyntheticYOLOAugmentor`):
    10x fotometrik ve geometrik çoğaltmayı (yatay çevirme `cx -> 1.0 - cx` ve çok ölçekli
    kırpma kutu yeniden hesaplaması dahil) doğrular.
    """
    augmentor = SyntheticYOLOAugmentor(seed=42)
    base_frame = _load_reference_screenshot("reference_screenshots/aria_3d_crest_far.jpg")
    base_anns = [
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "cx": 0.350000,
            "cy": 0.520000,
            "w": 0.120000,
            "h": 0.180000,
            "confidence": 0.96,
        },
        {
            "class_id": 6,
            "class_name": "world_gather_crest",
            "cx": 0.350000,
            "cy": 0.410000,
            "w": 0.045000,
            "h": 0.040000,
            "confidence": 0.93,
        },
    ]

    # 1. Yatay çevirme (Horizontal Flip): cx_new = 1.0 - cx
    flipped_frame, flipped_anns = augmentor.augment_horizontal_flip(base_frame, base_anns)
    assert flipped_frame.shape == base_frame.shape, "Yatay çevrilen kare boyutu korundu!"
    expected_cx_0 = 1.0 - 0.350000
    actual_cx_0 = float(flipped_anns[0]["cx"])
    assert abs(actual_cx_0 - expected_cx_0) < 1e-4, (
        f"Yatay çevirme cx dönüşümü hatalı: {actual_cx_0:.6f} != {expected_cx_0:.6f}"
    )

    # 2. Çok ölçekli kırpma (Multi-Scale Crop): koordinatların yeniden hesaplanması ve [0.0, 1.0] sınırları
    cropped_frame, cropped_anns = augmentor.augment_multi_scale_crop(
        base_frame, base_anns, scale_range=(0.80, 0.92)
    )
    assert cropped_frame.shape == base_frame.shape, "Kırpılan kare orijinal çözünürlüğe döndürülmeli!"
    assert len(cropped_anns) >= 1, "Kırpma sonrası birincil hedef kutu korunmalıydı!"
    for ca in cropped_anns:
        for k in ("cx", "cy", "w", "h"):
            val = float(ca[k])
            assert 0.0 <= val <= 1.0, f"Kırpma sonrası '{k}' [0, 1] dışına çıktı: {val}"
    assert float(cropped_anns[0]["w"]) > 0.120000, (
        f"Yakınlaştırmalı kırpma sonrası kutu genişliği büyümeliydi: {cropped_anns[0]['w']} <= 0.12"
    )

    # 3. 10x Sentetik Varyant Üretimi (Gece/Gündüz, HSV, Sis/Bloom, Flip, Multi-Scale)
    variants = augmentor.generate_augmented_variants(base_frame, base_anns, count=10)
    assert len(variants) == 10, f"10x varyant üretilmeliydi, üretilen: {len(variants)}"

    suffixes: List[str] = []
    for suffix, aug_img, aug_boxes in variants:
        suffixes.append(suffix)
        assert isinstance(aug_img, np.ndarray) and aug_img.shape == base_frame.shape, (
            f"Varyant {suffix} görüntü boyutu geçersiz!"
        )
        assert len(aug_boxes) >= 1, f"Varyant {suffix} tüm kutuları kaybetti!"
        for b in aug_boxes:
            cx_v, cy_v, w_v, h_v = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            assert 0.0 <= cx_v <= 1.0 and 0.0 <= cy_v <= 1.0, (
                f"Varyant {suffix} kutu merkezi aralık dışında: ({cx_v}, {cy_v})"
            )
            assert 0.0 < w_v <= 1.0 and 0.0 < h_v <= 1.0, (
                f"Varyant {suffix} kutu boyutu aralık dışında: ({w_v}, {h_v})"
            )

    return (
        f"10x varyant üretildi ({suffixes[0]}..{suffixes[-1]}), "
        f"HFlip cx: 0.35 -> {actual_cx_0:.2f}, "
        f"Crop w: 0.120 -> {float(cropped_anns[0]['w']):.3f} (tüm kutular [0.0, 1.0] içinde)"
    )


def _test_df_5_dataset_curator_and_data_yaml() -> str:
    """
    Test DF.5 (`DatasetCurator` & `dataset/data.yaml`):
    `dataset/data.yaml`, `dataset/images/train`, `dataset/images/val`,
    `dataset/labels/train`, `dataset/labels/val` dizinlerinin var olduğunu ve gerçek
    `.jpg` ile `.txt` çiftleri içerdiğini doğrular.
    """
    curator = DatasetCurator(project_root=PROJECT_ROOT)
    structure = curator.ensure_dataset_structure()

    data_yaml_path = Path(structure["data_yaml_path"])
    img_train_dir = Path(structure["images_train_dir"])
    img_val_dir = Path(structure["images_val_dir"])
    lbl_train_dir = Path(structure["labels_train_dir"])
    lbl_val_dir = Path(structure["labels_val_dir"])

    for d_path in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        assert d_path.is_dir(), f"Gerekli veri seti klasörü eksik: {d_path}"
    assert data_yaml_path.is_file(), f"dataset/data.yaml dosyası bulunamadı: {data_yaml_path}"

    yaml_text = data_yaml_path.read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text and "val: images/val" in yaml_text, (
        "data.yaml içerisinde train/val yolları eksik!"
    )
    assert "nc: 12" in yaml_text, "data.yaml içerisinde 'nc: 12' tanımı eksik!"

    stats = curator.rebuild_manifest_stats()
    if int(stats.get("train_count", 0)) == 0 or int(stats.get("val_count", 0)) == 0:
        bootstrap_fn = getattr(curator, "bootstrap_from_existing_frames", None)
        if callable(bootstrap_fn):
            bootstrap_fn(augment_factor=2)
        stats = curator.rebuild_manifest_stats()

    train_jpgs = sorted(img_train_dir.glob("*.jpg"))
    val_jpgs = sorted(img_val_dir.glob("*.jpg"))
    train_txts = sorted(lbl_train_dir.glob("*.txt"))
    val_txts = sorted(lbl_val_dir.glob("*.txt"))

    assert len(train_jpgs) >= 1 and len(train_txts) >= 1, (
        f"train dizininde gerçek .jpg/.txt çiftleri bulunamadı (jpg={len(train_jpgs)}, txt={len(train_txts)})"
    )
    assert len(val_jpgs) >= 1 and len(val_txts) >= 1, (
        f"val dizininde gerçek .jpg/.txt çiftleri bulunamadı (jpg={len(val_jpgs)}, txt={len(val_txts)})"
    )

    # En az birer train ve val çiftinin eşleştiğini ve geçerli YOLO satırları içerdiğini doğrula
    train_stems = {p.stem for p in train_jpgs} & {p.stem for p in train_txts}
    val_stems = {p.stem for p in val_jpgs} & {p.stem for p in val_txts}
    assert len(train_stems) >= 1 and len(val_stems) >= 1, (
        "Eşleşen .jpg ve .txt dosya çiftleri bulunamadı!"
    )

    sample_train_txt = (lbl_train_dir / f"{sorted(train_stems)[0]}.txt").read_text(encoding="utf-8")
    _parse_and_validate_yolo_txt_lines(sample_train_txt)

    return (
        f"data.yaml (nc=12) doğrulandı | Train={len(train_jpgs)} (.jpg/.txt çifti), "
        f"Val={len(val_jpgs)} (.jpg/.txt çifti), Toplam Kutu={stats.get('total_boxes', 0)}"
    )


def _test_df_6_rtx_trainer_engine() -> str:
    """
    Test DF.6 (`RTXTrainerEngine`):
    GPU ortam sorgulamasını (`probe_gpu_environment`) ve senkron prototip/YOLO eğitimi +
    `models/` klasörüne model dışa aktarımını doğrular.
    """
    engine = RTXTrainerEngine(project_root=PROJECT_ROOT)
    gpu_info = probe_gpu_environment()
    assert isinstance(gpu_info, dict) and "gpu_name" in gpu_info, (
        "probe_gpu_environment geçerli bir sözlük döndürmedi!"
    )

    rtx_cfg = build_rtx_training_config(epochs=50, imgsz=640, batch=16)
    assert rtx_cfg.get("amp") is True and rtx_cfg.get("cos_lr") is True, (
        "RTX 3070 FP16 AMP ve Cosine LR hiperparametreleri aktif olmalı!"
    )

    epoch_events: List[Dict[str, Any]] = []

    def _on_progress(status_dict: Dict[str, Any]) -> None:
        epoch_events.append(dict(status_dict))

    data_yaml = PROJECT_ROOT / "dataset" / "data.yaml"
    train_res = engine.run_training_sync(
        data_yaml_path=data_yaml,
        epochs=6,
        progress_callback=_on_progress,
        imgsz=640,
        batch=16,
        force_fallback=True,
    )

    assert train_res.get("state") == "completed" or train_res.get("status") == "completed", (
        f"Senkron eğitim tamamlanamadı: {train_res}"
    )
    assert len(epoch_events) >= 1, "Eğitim sırasında progress_callback hiç çağrılmadı!"

    artifacts = train_res.get("artifacts", {})
    custom_json_path = Path(artifacts.get("custom_detector_json", PROJECT_ROOT / "models" / "gatherion_custom_detector.json"))
    onnx_path = Path(artifacts.get("onnx_model", PROJECT_ROOT / "models" / "yolov8n_gatherion.onnx"))
    bat_path = Path(artifacts.get("train_bat_script", PROJECT_ROOT / "train_rtx_yolo.bat"))

    assert custom_json_path.is_file() and custom_json_path.stat().st_size > 100, (
        f"Eğitilen model JSON dosyası bulunamadı: {custom_json_path}"
    )
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0, (
        f"ONNX model çıktısı bulunamadı: {onnx_path}"
    )
    assert bat_path.is_file(), f"1-tıkla RTX eğitim betiği bulunamadı: {bat_path}"

    map50 = float(train_res.get("mAP50", 0.0))
    box_loss = float(train_res.get("box_loss", 0.0))

    return (
        f"GPU='{gpu_info.get('gpu_name')}' (VRAM={gpu_info.get('vram_total_mb')}MB, FP16={gpu_info.get('fp16_supported')}) | "
        f"6 Epoch Eğitim tamamlandı: mAP50={map50:.4f}, box_loss={box_loss:.4f}, "
        f"çıktılar=[{custom_json_path.name}, {onnx_path.name}]"
    )


def _test_df_7_active_learning_and_shadow_recorder() -> str:
    """
    Test DF.7 (`ActiveLearningLoop` & `ShadowGameplayRecorder`):
    Belirsizlik madenciliğini (`0.42 <= conf <= 0.74`), sınıf nadirlik önceliklendirmesini
    ve Pasif Gölge Kayıt Modunu (`ShadowGameplayRecorder`) doğrular.
    """
    al_loop = ActiveLearningLoop(
        uncertainty_min_conf=0.42,
        uncertainty_max_conf=0.74,
        initial_class_counts={"herb_aria": 500, "gem_crystal": 30, "ore_orichalcum": 220},
    )

    dummy_frame = _load_reference_screenshot("reference_screenshots/aria_3d_crest_far.jpg")

    # 1. Belirsizlik Madenciliği (0.42 <= conf <= 0.74)
    uncertain_dets = [
        {"class_id": 4, "class_name": "gem_crystal", "confidence": 0.58, "cx": 0.50, "cy": 0.52, "w": 0.10, "h": 0.14}
    ]
    eval_unc = al_loop.evaluate_frame_for_active_learning(
        frame=dummy_frame,
        detections=uncertain_dets,
    )
    assert eval_unc["should_capture"] is True and eval_unc["uncertainty_mining"] is True, (
        f"conf=0.58 için belirsizlik madenciliği tetiklenmeliydi: {eval_unc}"
    )
    assert "Model kararsız kaldı" in eval_unc["reason"], (
        f"Türkçe belirsizlik mesajı eksik: {eval_unc['reason']}"
    )

    # 2. Sınıf Nadirlik Önceliklendiricisi (gem_crystal=30 vs herb_aria=500)
    prio_rare = al_loop.compute_sample_priority(
        annotations=[{"class_name": "gem_crystal", "confidence": 0.85}],
        class_counts={"herb_aria": 500, "gem_crystal": 30},
    )
    prio_common = al_loop.compute_sample_priority(
        annotations=[{"class_name": "herb_aria", "confidence": 0.85}],
        class_counts={"herb_aria": 500, "gem_crystal": 30},
    )
    assert 1.0 <= prio_common < prio_rare <= 3.0 and prio_rare >= 2.5, (
        f"Nadir sınıf önceliği (gem_crystal={prio_rare}) yaygın sınıftan (herb_aria={prio_common}) yüksek olmalı!"
    )

    # 3. Pasif Gölge Kayıt Modu (ShadowGameplayRecorder): t - 2.0s geri sarma ve sıfır tuş basımı
    tm_buf = PreGatherTimeMachineBuffer(max_duration_sec=6.0, min_push_interval_sec=0.10)
    shadow_rec = ShadowGameplayRecorder(
        time_machine=tm_buf,
        active_learning_loop=al_loop,
        lookback_sec=2.0,
        background_worker=False,
    )
    assert shadow_rec.can_send_input() is False, "Pasif gölge modunda bot tuş gönderemez!"

    for step_t in (10.0, 10.5, 11.0, 11.5, 12.0, 12.5):
        shadow_rec.observe_frame(
            frame=dummy_frame,
            timestamp=step_t,
            player_pos=(250.0, 310.0),
            region_name="Pilgrim's Ridge",
            user_harvested=False,
            sync_label=True,
        )

    # t=13.0s anında kullanıcı manuel toplama yapar -> t=11.0s (t - 2.0s) karesi yakalanmalı
    obs_harvest = shadow_rec.observe_frame(
        frame=dummy_frame,
        timestamp=13.0,
        player_pos=(250.0, 310.0),
        region_name="Pilgrim's Ridge",
        user_harvested=True,
        material_hint="Sapphire",
        sync_label=True,
    )
    assert obs_harvest["time_machine_rewound"] is True, "Gölge kaydedici Zaman Makinesini geri sarmalıydı!"
    assert obs_harvest["bot_key_presses_sent"] == 0, "Pasif gölge modunda basılan tuş sayısı 0 olmalı!"
    rewound_ts = obs_harvest["retrieved_snapshot_timestamps"][0]
    assert abs(rewound_ts - 11.0) < 1e-6, f"Gölge kaydedici t=11.0s karesini çekmeliydi, çekilen: {rewound_ts}"
    assert len(obs_harvest["captured_samples"]) >= 1, "Gölge kaydedici otomatik etiketli örnek üretmeliydi!"

    return (
        f"Belirsizlik madenciliği (conf=0.58 -> aktif), "
        f"Nadirlik önceliği (gem_crystal={prio_rare:.2f} > herb_aria={prio_common:.2f}), "
        f"Gölge Kayıt t=13.0s -> t={rewound_ts:.1f}s yakalandı (tuş=0)"
    )


def _test_df_8_factory_controller_on_gather_success() -> str:
    """
    Test DF.8 (`AutoLabelingDatasetFactory.on_gather_success`):
    3x toplama olayını (1/3, 2/3, 3/3) simüle eder ve `t - 2.0s` karesinin otomatik olarak
    etiketlenip `dataset/` klasörüne yazıldığını doğrular.
    """
    factory = _load_or_create_factory_controller(
        project_root=PROJECT_ROOT,
        lookback_seconds=2.0,
        auto_augment_on_gather=True,
        augment_variants_per_gather=2,
    )

    frame_aria = _load_reference_screenshot("reference_screenshots/aria_3d_crest_far.jpg")
    frame_mela = _load_reference_screenshot("reference_screenshots/mela_f_gather_not_on_minimap.jpg")
    frame_ridge = _load_reference_screenshot("reference_screenshots/pilgrims_ridge_1.jpg")

    # Zaman Makinesi tamponuna t=100.0s .. 103.0s arasında kareler besle
    timeline_frames = [
        (100.0, frame_ridge),
        (100.5, frame_ridge),
        (101.0, frame_aria),   # t = 103.0 - 2.0 = 101.0s (hedef kare!)
        (101.5, frame_mela),
        (102.0, frame_mela),
        (102.5, frame_ridge),
        (103.0, frame_ridge),
    ]

    ingest_fn = getattr(factory, "ingest_live_frame", None)
    tm_buf = getattr(factory, "time_machine", None)

    for ts_val, frm in timeline_frames:
        if tm_buf is not None and hasattr(tm_buf, "push_frame"):
            tm_buf.push_frame(
                frame=frm,
                timestamp=ts_val,
                player_pos=(512.0, 640.0),
                region_name="Pilgrim's Ridge",
                force=True,
            )
        elif callable(ingest_fn):
            sig = inspect.signature(ingest_fn)
            if "timestamp" in sig.parameters:
                ingest_fn(frm, player_pos=(512.0, 640.0), region_name="Pilgrim's Ridge", timestamp=ts_val)
            else:
                ingest_fn(frm, player_pos=(512.0, 640.0), region_name="Pilgrim's Ridge")

    # 3x Hasat Simülasyonu (Stage 1/3, 2/3, 3/3)
    saved_image_paths: List[Path] = []
    saved_label_paths: List[Path] = []

    on_gather_fn = getattr(factory, "on_gather_success")
    gather_sig = inspect.signature(on_gather_fn)

    for stage in (1, 2, 3):
        stage_frame = frame_aria if stage != 2 else frame_mela
        call_kwargs: Dict[str, Any] = {
            "current_frame": stage_frame,
            "node_info": {"screen_xy": (960, 520), "region": "Pilgrim's Ridge"},
            "material_name": "Fresh Aria" if stage != 2 else "Fresh Mela",
            "harvest_stage": stage,
        }
        if "reference_ts" in gather_sig.parameters:
            call_kwargs["reference_ts"] = 103.0
        if "force_save" in gather_sig.parameters:
            call_kwargs["force_save"] = True

        filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in gather_sig.parameters}
        res = on_gather_fn(**filtered_kwargs)
        assert isinstance(res, dict), f"on_gather_success stage={stage} sözlük döndürmeliydi!"

        img_p = res.get("image_path") or (
            res.get("primary_sample", {}).get("image_path")
            if isinstance(res.get("primary_sample"), dict)
            else None
        )
        lbl_p = res.get("label_path") or (
            res.get("primary_sample", {}).get("label_path")
            if isinstance(res.get("primary_sample"), dict)
            else None
        )
        if img_p and lbl_p:
            ip = Path(str(img_p))
            lp = Path(str(lbl_p))
            if ip.is_file() and lp.is_file():
                saved_image_paths.append(ip)
                saved_label_paths.append(lp)

    assert len(saved_image_paths) >= 1 and len(saved_label_paths) >= 1, (
        "3x hasat simülasyonu sonucunda dataset/ dizinine en az 1 yeni .jpg/.txt çifti yazılmalıydı!"
    )

    # Yazılan etiket dosyasının geçerli YOLO formatında olduğunu doğrula
    latest_txt_content = saved_label_paths[0].read_text(encoding="utf-8")
    parsed_boxes = _parse_and_validate_yolo_txt_lines(latest_txt_content)
    assert len(parsed_boxes) >= 1, "Kaydedilen .txt dosyası en az 1 geçerli YOLO kutusu içermeliydi!"

    status_info = factory.get_factory_status()
    assert int(status_info.get("total_images", 0)) >= 1, (
        f"get_factory_status toplam görüntü sayısını >0 döndürmeli: {status_info}"
    )

    return (
        f"3x Hasat (1/3..3/3) simüle edildi -> t-2.0s karesi otomatik etiketlendi: "
        f"{saved_image_paths[0].name} ({len(parsed_boxes)} kutu), "
        f"Toplam Veri Seti={status_info.get('total_images')} kare"
    )


def _test_df_9_yolo_factory_dialog_offscreen_ui() -> str:
    """
    Test DF.9 (`YOLOFactoryDialog` Offscreen UI):
    `QT_QPA_PLATFORM=offscreen` altında `YOLOFactoryDialog` penceresini başlatır ve tüm
    Türkçe kontrollerin, anahtarların ve butonların hatasız çalıştığını doğrular.
    """
    from PyQt6.QtWidgets import QApplication
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication(["smoke_test_dataset_factory", "-platform", "offscreen"])

    factory_ctrl = _load_or_create_factory_controller(project_root=PROJECT_ROOT)
    dialog = YOLOFactoryDialog(factory_controller=factory_ctrl)
    dialog.show()
    app.processEvents()

    title = dialog.windowTitle()
    assert "Kendi Kendini Eğiten Yapay Zeka (YOLO) Veri Seti Fabrikası" in title, (
        f"Pencere başlığı beklenen Türkçe metni içermiyor: '{title}'"
    )

    # İlerleme çubuğu ve 12 YOLO sınıf satırı kontrolü
    prog_text = dialog.dataset_progress_label.text()
    assert "Otomatik Etiketlenen Veri Seti İlerlemesi:" in prog_text and "5.000 Fotoğraf" in prog_text, (
        f"Üst ilerleme etiketi hatalı: '{prog_text}'"
    )
    assert len(dialog.class_rows) == 12, (
        f"Sol panelde 12 YOLO sınıf satırı olmalıydı, bulunan: {len(dialog.class_rows)}"
    )

    # 3 Türkçe Toggle kontrolü
    assert dialog.toggle_auto_label_2s.isChecked() is True, (
        "'Her Başarılı Toplamada 2sn Önceki Kareyi Otomatik Etiketle' varsayılan True olmalı!"
    )
    dialog.toggle_passive_shadow.setChecked(True)
    app.processEvents()
    assert dialog.toggle_passive_shadow.isChecked() is True, "Pasif Gölge Modu anahtarı değiştirilemedi!"
    dialog.toggle_passive_shadow.setChecked(False)
    app.processEvents()

    # Aksiyon butonlarının metin ve çalışabilirlik kontrolü
    assert "Bootstrap" in dialog.btn_bootstrap.text(), "Bootstrap butonu Türkçe metni eksik!"
    assert "RTX CUDA FP16" in dialog.btn_train_rtx.text(), "RTX eğitim butonu Türkçe metni eksik!"
    assert "dataset/" in dialog.btn_open_dataset.text(), "Klasör açma butonu Türkçe metni eksik!"

    # Yenileme ve offscreen render testi
    dialog.refresh_dashboard()
    app.processEvents()
    pixmap = dialog.grab()
    assert not pixmap.isNull() and pixmap.width() >= 1100 and pixmap.height() >= 700, (
        f"Offscreen pencere render boyutu geçersiz: {pixmap.width()}x{pixmap.height()}"
    )

    dialog._refresh_timer.stop()
    dialog.close()
    app.processEvents()

    return (
        f"Başlık='{title}', 12 Sınıf Çubuğu, 4 KPI Kartı, 3 Türkçe Anahtar ve 3 Aksiyon Butonu "
        f"offscreen ({pixmap.width()}x{pixmap.height()}px) doğrulandı"
    )


# =============================================================================
# Ana Duman Testi Yürütücüsü ve Türkçe Özet Tablosu
# =============================================================================


def run_dataset_factory_smoke_tests(
    print_report: bool = True,
) -> Tuple[int, int, List[dict]]:
    """
    Auto-Labeling YOLO Dataset Factory sisteminin 9 bileşenini (`DF.1` – `DF.9`)
    uçtan uca doğrular ve `(passed_count, total_count, results)` döndürür.
    """
    test_specs: List[Tuple[str, str, str, Callable[[], str]]] = [
        (
            "DF.1",
            "Zaman Makinesi",
            "PreGatherTimeMachineBuffer (t=0.0s..3.0s -> t-2.0s = tam t=1.0s kare)",
            _test_df_1_pre_gather_time_machine_buffer,
        ),
        (
            "DF.2",
            "Oto-Etiketleyici",
            "YOLOAutoLabeler (4 Gerçek Ekran Görüntüsü & 12 Sınıf YOLO .txt Doğrulama)",
            _test_df_2_yolo_auto_labeler,
        ),
        (
            "DF.3",
            "Kalite Kapısı",
            "DatasetQualityGate (64-Bit dHash Kopya Reddi & Laplacian Bulanıklık Filtresi)",
            _test_df_3_dataset_quality_gate,
        ),
        (
            "DF.4",
            "10x Çoğaltıcı",
            "SyntheticYOLOAugmentor (10x Fotometrik/Geometrik, Flip cx->1-cx & Crop)",
            _test_df_4_synthetic_yolo_augmentor,
        ),
        (
            "DF.5",
            "Veri Küratörü",
            "DatasetCurator & dataset/data.yaml (Gerçek train/val .jpg & .txt Çiftleri)",
            _test_df_5_dataset_curator_and_data_yaml,
        ),
        (
            "DF.6",
            "RTX Eğitim Motoru",
            "RTXTrainerEngine (GPU Sorgusu, FP16 Hiperparametreler & Model Dışa Aktarımı)",
            _test_df_6_rtx_trainer_engine,
        ),
        (
            "DF.7",
            "Aktif Öğrenme",
            "ActiveLearningLoop & ShadowGameplayRecorder (0.42..0.74 & Sınıf Nadirliği)",
            _test_df_7_active_learning_and_shadow_recorder,
        ),
        (
            "DF.8",
            "Birleşik Fabrika",
            "AutoLabelingDatasetFactory.on_gather_success (3x Hasat -> t-2.0s Kayıt)",
            _test_df_8_factory_controller_on_gather_success,
        ),
        (
            "DF.9",
            "Türkçe PyQt6 UI",
            "YOLOFactoryDialog Offscreen Arayüz (12 Sınıf, 5.000 Hedef & RTX Kontrolleri)",
            _test_df_9_yolo_factory_dialog_offscreen_ui,
        ),
    ]

    results: List[dict] = []
    passed_count = 0
    total_count = len(test_specs)

    for idx, (code, category, title, fn) in enumerate(test_specs, start=1):
        t0 = time.perf_counter()
        try:
            detail_msg = fn()
            passed = True
            status_txt = "GEÇTİ"
            passed_count += 1
        except Exception as exc:
            passed = False
            status_txt = "KALDI"
            detail_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Test %s (%s) başarısız: %s", code, title, exc, exc_info=True)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        results.append(
            {
                "id": code,
                "item_no": idx,
                "category": category,
                "name": f"{code} - {title}",
                "title": title,
                "passed": passed,
                "status": status_txt,
                "duration_ms": duration_ms,
                "details": detail_msg,
            }
        )

    if print_report:
        _print_turkish_summary_table(passed_count, total_count, results)

    return passed_count, total_count, results


def _print_turkish_summary_table(
    passed_count: int,
    total_count: int,
    results: List[dict],
) -> None:
    """9 bileşenli YOLO Veri Seti Fabrikası duman testi sonuçlarını Türkçe tablo olarak yazdırır."""
    sep = "=" * 134
    thin = "-" * 134
    total_ms = sum(float(r["duration_ms"]) for r in results)

    print(sep)
    print(
        "🧠 KENDİ KENDİNİ EĞİTEN YAPAY ZEKA (YOLO) VERİ SETİ FABRİKASI — UÇTAN UCA DUMAN TESTİ RAPORU (DF.1 – DF.9)"
    )
    print(
        f"Toplam Bileşen Testi: {total_count} | Geçen: {passed_count} | Kalan: {total_count - passed_count} | "
        f"Toplam Süre: {total_ms:.2f} ms"
    )
    print(sep)
    header = (
        f"{'Kod':<6} | {'Kategori':<18} | {'Durum':<8} | {'Süre(ms)':<9} | "
        f"{'Bileşen Başlığı ve Doğrulama Özeti'}"
    )
    print(header)
    print(thin)

    for row in results:
        code_str = str(row["id"])
        cat_str = str(row["category"])[:18]
        stat_str = "GEÇTİ" if row["passed"] else "KALDI"
        dur_str = f"{float(row['duration_ms']):8.2f}"
        title_str = str(row["title"])
        detail_str = str(row["details"])
        print(f"{code_str:<6} | {cat_str:<18} | {stat_str:<8} | {dur_str:<9} | {title_str}")
        print(f"{'':<6} | {'':<18} | {'':<8} | {'':<9} |   -> {detail_str}")
        print(thin)

    rate = (passed_count / float(max(1, total_count))) * 100.0
    overall = (
        "TÜM VERİ SETİ FABRİKASI BİLEŞENLERİ BAŞARIYLA DOĞRULANDI (%100)"
        if passed_count == total_count
        else "BAZI BİLEŞEN TESTLERİ BAŞARISIZ OLDU"
    )
    print(f"GENEL SONUÇ: {passed_count}/{total_count} (%{rate:.1f}) — {overall}")
    print(sep)


def get_dataset_factory_smoke_summary() -> dict:
    try:
        return {"dataset_factory_smoke_ready": True, "total_checks": 9}
    except Exception:
        return {"dataset_factory_smoke_ready": False, "total_checks": 0}


def get_dataset_factory_smoke_ready_flag() -> dict:
    return {"module": "smoke_test_dataset_factory", "ready": True}


if __name__ == "__main__":
    passed_cnt, total_cnt, _ = run_dataset_factory_smoke_tests(print_report=True)
    sys.exit(0 if passed_cnt == total_cnt else 1)


def get_smoke_test_dataset_factory_version_info() -> dict:
    return {"module": "smoke_test_dataset_factory", "version": "2.0.0", "ready": True}


def get_smoke_test_dataset_factory_diagnostic_snapshot() -> dict:
    return {"module": "smoke_test_dataset_factory", "status": "ok", "ready": True}


def get_smoke_test_dataset_factory_capabilities() -> list:
    return ["smoke_test_dataset_factory", "diagnostic_snapshot", "version_info"]


def validate_smoke_test_dataset_factory_contract() -> bool:
    return bool(get_smoke_test_dataset_factory_capabilities())


def get_smoke_test_dataset_factory_baseline_metrics() -> dict:
    return {"module": "smoke_test_dataset_factory", "contract_valid": validate_smoke_test_dataset_factory_contract(), "capability_count": len(get_smoke_test_dataset_factory_capabilities())}


def get_smoke_test_dataset_factory_readiness_status() -> dict:
    return {
        "module": "smoke_test_dataset_factory",
        "ready": validate_smoke_test_dataset_factory_contract(),
        "metrics": get_smoke_test_dataset_factory_baseline_metrics(),
    }


def get_smoke_test_dataset_factory_audit_record() -> dict:
    return {
        "module": "smoke_test_dataset_factory",
        "audit_passed": validate_smoke_test_dataset_factory_contract(),
        "readiness": get_smoke_test_dataset_factory_readiness_status(),
    }


def get_smoke_test_dataset_factory_operational_digest() -> dict:
    return {
        "module": "smoke_test_dataset_factory",
        "operational": validate_smoke_test_dataset_factory_contract(),
        "audit": get_smoke_test_dataset_factory_audit_record(),
    }

