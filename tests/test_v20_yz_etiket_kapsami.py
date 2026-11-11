# -*- coding: utf-8 -*-
"""
V20 YZ etiket kapsamı (fix/v20-yz-etiket-kapsami).

Şikayet (Selçuk, 11 Kasım 18:56): çekilen YZ fotolarının hepsi
"learn_monsters" başlıklı; yeniden etiketleme yalnızca canavarı
etiketliyor — karede görülen her şey etiketlenmeli.

Kök neden (kod kanıtlı):
- YOLOAutoLabeler._scan_hud_and_skills / _scan_path_and_obstacles
  stem kapılıdır (is_skill_learning_stem / is_path_learning_stem).
- Yakalama: OmniGameLearner._detect_real_boxes etiketleyiciye
  source_path vermez (stem "") -> iki kapı da kapanır -> HUD
  (16-24) ve yol/engel (14/15) kutuları ASLA üretilemez ->
  _filter_boxes_for_category skills/obstacles/paths/vitality için
  hep boş döner -> yalnızca "monsters" kaydedilebilir.
- Yeniden etiketleme: batch relabel + tekli relabel gerçek stem'i
  (learn_monsters_*) geçirir -> kapılar yine kapanır -> relabel
  yalnızca canavar (10/18) + materyal (0-8/12) + minimap (9/11)
  + prompt (13) kutusu üretebilir.

Düzeltme: YOLOAutoLabeler.auto_label_frame(..., full_scan=False)
opt-in parametresi; iki tarama metoduna allow_full_scan kapısı.
Yakalama + batch relabel + tekli relabel + fabrika yolları
(on_gather/roaming/F8/shadow + curator yedeği) full_scan=True
geçirir. Varsayılan False — W29 stem kilidi korunur; kalibrasyon
sihirbazı, dialog önizleme yardımcısı ve tools/relabel_weak_only.py
değişmedi. Çıkan kutular zayıf-cap'lidir (HUD <= 0.70, yol/engel
<= 0.68), incelemeye düşer, insan onayı olmadan train TXT'ye girmez;
el etiketi koruması (P0-3) ve requeue kurallarına dokunulmadı.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.omni_game_learner import OmniGameLearner, _filter_boxes_for_category
from dataset_factory.dataset_curator import DatasetCurator
from dataset_factory.yolo_auto_labeler import (
    YOLOAutoLabeler,
    is_path_learning_stem,
    is_skill_learning_stem,
)


def _noise_frame(seed=7, shape=(360, 640, 3)):
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _build_synthetic_aria_grass_and_hud_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """W29 sentetik karesi: yol ROI'sinde çim + altta HUD barları/slotları.

    learn_paths_* stem'inde 15: path_ground, learn_skills_* stem'inde
    HUD kutuları üretir (W29 testleri bunu pin'ler); boot/gather
    stem'lerinde kilitlidir.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    py1, py2 = int(height * 0.50), int(height * 0.86)
    px1, px2 = int(width * 0.25), int(width * 0.75)
    frame[py1:py2, px1:px2] = (60, 115, 65)  # BGR çim
    cv2.rectangle(frame, (400, 900), (600, 916), (0, 0, 225), -1)   # kırmızı HP bar
    cv2.rectangle(frame, (400, 924), (600, 940), (225, 105, 0), -1)  # mavi MP bar
    for sx in (660, 710, 760, 810):  # üst tüketim sırası (cy ~0.893)
        cv2.rectangle(frame, (sx, 950), (sx + 36, 986), (175, 175, 175), 2)
        cv2.circle(frame, (sx + 18, 968), 10, (90, 190, 230), -1)
    for sx in (660, 710, 760, 810):  # alt savaş skill sırası (cy ~0.954)
        cv2.rectangle(frame, (sx, 1012), (sx + 38, 1050), (190, 190, 190), 2)
        cv2.line(frame, (sx + 6, 1020), (sx + 32, 1042), (140, 160, 250), 2)
    return frame


def _frame_hsv_gray():
    frame = _build_synthetic_aria_grass_and_hud_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, hsv, gray


MONSTER_STEM = "learn_monsters_1794408251038496600_889"


# ---------------------------------------------------------------------------
# 1. Kapı predikatları: canavar stem'i kilitli kalır
# ---------------------------------------------------------------------------

def test_predicates_still_lock_monster_stems():
    for stem in (
        MONSTER_STEM,
        "learn_monsters_001",
        "learn_monsters_1794405235376146500",
        "",
    ):
        assert is_path_learning_stem(stem) is False, stem
        assert is_skill_learning_stem(stem) is False, stem
    # Pozitif kontrol: adanmış stem'ler açık kalır.
    assert is_path_learning_stem("learn_paths_001_grass") is True
    assert is_skill_learning_stem("learn_skills_001_hud") is True


# ---------------------------------------------------------------------------
# 2. Taramalar varsayılan olarak canavar stem'inde kilitli
# ---------------------------------------------------------------------------

def test_scans_locked_by_default_on_monster_stem():
    labeler = YOLOAutoLabeler()
    frame, hsv, gray = _frame_hsv_gray()
    assert labeler._scan_path_and_obstacles(
        frame_bgr=frame, hsv=hsv, gray=gray, source_stem=MONSTER_STEM
    ) == []
    assert labeler._scan_hud_and_skills(
        frame_bgr=frame, hsv=hsv, gray=gray, source_stem=MONSTER_STEM
    ) == []


# ---------------------------------------------------------------------------
# 3. full_scan kapıyı açar (canavar stem'inde bile)
# ---------------------------------------------------------------------------

def test_full_scan_unlocks_scans_on_monster_stem():
    labeler = YOLOAutoLabeler()
    frame, hsv, gray = _frame_hsv_gray()
    path_anns = labeler._scan_path_and_obstacles(
        frame_bgr=frame, hsv=hsv, gray=gray, source_stem=MONSTER_STEM,
        allow_full_scan=True,
    )
    assert any(a["class_id"] == 15 for a in path_anns), (
        f"full_scan yol taramasını açmalı (15: path_ground): {path_anns}"
    )
    hud_anns = labeler._scan_hud_and_skills(
        frame_bgr=frame, hsv=hsv, gray=gray, source_stem=MONSTER_STEM,
        allow_full_scan=True,
    )
    assert len(hud_anns) >= 1
    assert all(a["class_id"] != 13 for a in hud_anns), (
        "HUD taraması sahte 13: skill_ui_prompt üretmez"
    )


# ---------------------------------------------------------------------------
# 4. Uçtan uca: learn_monsters kaynağında full_scan 15 üretir
# ---------------------------------------------------------------------------

def test_auto_label_frame_full_scan_emits_path_ground_on_monster_stem():
    labeler = YOLOAutoLabeler(exemplar_memory=False)
    frame = _build_synthetic_aria_grass_and_hud_frame()
    monster_path = f"dataset/images/train/{MONSTER_STEM}.jpg"

    unlocked = labeler.auto_label_frame(
        frame=frame, known_material_name="Aria", fast_mode=True,
        source_path=monster_path, full_scan=True,
    )
    unlocked_ids = {int(a["class_id"]) for a in unlocked}
    assert 15 in unlocked_ids, f"full_scan 15: path_ground üretmeli: {unlocked}"

    # Negatif kontrol: varsayılan (full_scan=False) kilitli kalır.
    locked = labeler.auto_label_frame(
        frame=frame, known_material_name="Aria", fast_mode=True,
        source_path=monster_path,
    )
    locked_ids = {int(a["class_id"]) for a in locked}
    assert 15 not in locked_ids
    assert 14 not in locked_ids
    assert 13 not in locked_ids


def test_auto_label_frame_default_still_locks_boot_aria():
    """W29 kilidi korunur: boot stem'inde varsayılan çağrı HUD/yol üretmez."""
    labeler = YOLOAutoLabeler(exemplar_memory=False)
    frame = _build_synthetic_aria_grass_and_hud_frame()
    anns = labeler.auto_label_frame(
        frame=frame, known_material_name="Aria", fast_mode=True,
        source_path="dataset/images/boot_478467_000_aria_3d_crest_far.jpg",
    )
    ids = {int(a["class_id"]) for a in anns}
    assert 15 not in ids and 14 not in ids and 13 not in ids


# ---------------------------------------------------------------------------
# 5. Yakalama: _detect_real_boxes full_scan=True geçirir
# ---------------------------------------------------------------------------

class _RecordingLabeler:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls: List[Dict[str, Any]] = []

    def auto_label_frame(self, frame, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.boxes)


def test_capture_passes_full_scan_to_labeler():
    fake = _RecordingLabeler(
        [{"class_id": 10, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9}]
    )
    learner = OmniGameLearner(bot_worker=None, save_screenshots=False)
    learner._yolo_labeler = fake
    boxes = learner._detect_real_boxes(_noise_frame(), "general")
    assert fake.calls, "etiketleyici hiç çağrılmadı"
    assert fake.calls[0].get("full_scan") is True, fake.calls[0]
    assert boxes and boxes[0]["class_id"] == 10


# ---------------------------------------------------------------------------
# 6. Yakalama: aynı kare içerik kategorilerine ayrılır (yalnızca monsters değil)
# ---------------------------------------------------------------------------

def test_capture_routes_frame_to_content_categories():
    mixed = [
        {"class_id": 10, "box": (0.5, 0.4, 0.2, 0.2), "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.2, "confidence": 0.9},
        {"class_id": 14, "box": (0.3, 0.7, 0.15, 0.1), "cx": 0.3, "cy": 0.7, "w": 0.15, "h": 0.1, "confidence": 0.68},
        {"class_id": 15, "box": (0.5, 0.7, 0.3, 0.15), "cx": 0.5, "cy": 0.7, "w": 0.3, "h": 0.15, "confidence": 0.68},
        {"class_id": 16, "box": (0.4, 0.9, 0.12, 0.02), "cx": 0.4, "cy": 0.9, "w": 0.12, "h": 0.02, "confidence": 0.7},
        {"class_id": 20, "box": (0.6, 0.95, 0.03, 0.04), "cx": 0.6, "cy": 0.95, "w": 0.03, "h": 0.04, "confidence": 0.7},
    ]
    learner = OmniGameLearner(bot_worker=None, save_screenshots=True)
    learner._yolo_labeler = _RecordingLabeler(mixed)
    # bot_worker yok -> _actor_is_moving False olur; yürüyormuş gibi davran.
    learner._actor_is_moving = lambda: True
    recorded: List[tuple] = []

    def _save(**kwargs):
        # Gerçek _save_learning_sample kategori filtresini içeride uygular
        # (_filter_boxes_for_category); sahte de aynısını yapmalı, yoksa
        # paylaşılan liste her kategoriye filtrelenmemiş gider.
        filtered = _filter_boxes_for_category(kwargs["boxes"], kwargs["category"])
        recorded.append(
            (kwargs["category"], sorted(int(b["class_id"]) for b in filtered))
        )
        return True

    learner._save_learning_sample = _save
    learner._evaluate_and_save_learning_samples(
        _noise_frame(),
        time.time(),
        current_state="WALKING",
        current_mode="route",
        monster_info={"has_target": True, "hp_percent": 80, "monster_died": False},
        skill_info={"ready_skills": ["1"], "cooldown_skills": {}, "best_skill": "1"},
        obstacle_info={"obstacle_detected": True, "obstacle_type": "rock",
                       "recommended_steer_deg": 10.0},
        path_info={"is_walking": True},
        vitality_info={"player_hp_percent": 40, "player_mp_percent": 30,
                       "hp_pot_used": True, "mp_pot_used": False},
        gate_ok=True,
    )
    by_category = {cat: set(cids) for cat, cids in recorded}
    # Düzeltme öncesi yalnızca "monsters" kaydedilebilirdi (diğer dallar
    # paylaşımlı tespitte hiç kutu bulamazdı).
    assert set(by_category) == {"monsters", "skills", "obstacles", "paths", "vitality"}, (
        f"beklenen 5 kategori, gelen: {by_category}"
    )
    assert by_category["monsters"] == {10}
    assert by_category["skills"] == {20}
    assert by_category["obstacles"] == {14}
    assert by_category["paths"] == {15}
    assert by_category["vitality"] == {16}


# ---------------------------------------------------------------------------
# 7. Batch relabel: full_scan=True geçirilir
# ---------------------------------------------------------------------------

def test_relabel_batch_passes_full_scan(tmp_path):
    dataset_dir = tmp_path / "dataset"
    curator = DatasetCurator(project_root=tmp_path, dataset_dir=dataset_dir)
    img_path = curator.images_train_dir / "learn_monsters_001.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(img_path), img)

    seen: Dict[str, Any] = {}

    class BatchRecordingLabeler:
        def auto_label_frame(self, frame, fast_mode=True, source_path=None, full_scan=False):
            seen["full_scan"] = full_scan
            seen["source_path"] = str(source_path) if source_path else ""
            return []

    with patch(
        "dataset_factory.yolo_auto_labeler.YOLOAutoLabeler", BatchRecordingLabeler
    ):
        curator.relabel_dataset_batch(
            image_paths=[img_path], force_all=True, max_workers=1
        )
    assert seen.get("full_scan") is True, f"batch relabel full_scan geçirmeli: {seen}"
    assert "learn_monsters_001" in seen.get("source_path", "")


# ---------------------------------------------------------------------------
# 8. Curator wrapper: full_scan YOLOAutoLabeler'a geçirilir
# ---------------------------------------------------------------------------

def test_curator_wrapper_forwards_full_scan(tmp_path):
    seen: Dict[str, Any] = {}

    class WrapperRecordingLabeler:
        def auto_label_frame(self, frame, fast_mode=False, source_path=None, full_scan=False):
            seen["full_scan"] = full_scan
            return [{
                "class_id": 10, "class_name": "class_10",
                "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                "confidence": 0.9, "source": "yolo_model_v2",
            }]

    curator = DatasetCurator(
        project_root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        auto_labeler=WrapperRecordingLabeler(),
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = curator.auto_label_frame(
        frame, source_path="learn_monsters_001.jpg", full_scan=True
    )
    assert seen.get("full_scan") is True, f"wrapper full_scan geçirmeli: {seen}"
    assert any(int(a["class_id"]) == 10 for a in out)
