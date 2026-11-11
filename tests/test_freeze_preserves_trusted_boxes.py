# -*- coding: utf-8 -*-
"""tests/test_freeze_preserves_trusted_boxes.py
================================================
v19 KAPSAM NOTU madde 15 — dondurma (freeze) sonrası kutu sayısı
AZALMAMALI regresyon testleri.

Canlı olay (11 Kasım 13:38-13:39, Selçuk'un ekran görüntüsü): dondurma
denemeleri + 397 kare silinmesiyle canlı dataset 790 kutudan 65 kutuya
düştü. Mekanizma: ``commit_split_labels`` TXT'leri yeniden yazıyor,
yalnızca güvenilir/onaylı kutular kalıyor.

BAZ (kullanıcı kararı 2026-11-11 14:13 — kurtarma İPTAL): zip'ten geri
yükleme yapılmayacak; baz kurtarılmış 33.144 kutuluk dataset DEĞİL,
mevcut ONAYLI HAVUZ'dur — canlı 919 kare / 20.786 kutu. Bu test o havuzu
küçük ölçekte modeller (aynı yapı: kare başına ~22 kutu yerine testte
4 onaylı kare / 14 kutu; havuzdaki kutuların tamamı trusted/onaylı
durumda, köken bayraklı) ve dondurma adımının havuz kutu sayısını
koruduğunu gerçek kod yoluyla kanıtlar.

Dondurma akışı (``ui/yolo_factory_dialog.py::_freeze_dataset_impl``,
onaylı mod): ``ApprovedDatasetPool.prepare_approved_training_dataset``
+ QA kapısı + manifesto. Bu test o akışın iki gerçek adımını çalıştırır:

1. ``commit_split_labels`` — TXT'yi yeniden yazan mekanizma
   (``dataset_factory/label_trust.py:2716``). Dondurma öncesi/sonrası
   her commit güvenilir kutuyu korumalı.
2. ``prepare_approved_training_dataset`` — dondurmanın havuz adımı
   (``dataset_factory/approved_pool.py:1585``). Havuz zaten doluyken
   yeniden dondurma, havuzdaki kutu sayısını azaltmamalı.

Kapsanan kurallar (kural test içinde yeniden yazılmaz — gerçek kod
çağrılır):
- P0-1: ``_rewrite_label_txt`` boş emit'te mevcut dolu TXT'yi reddeder
  (``label_trust.py:923``); ``_reconcile_txt_hand_labels`` 5 alanlı TXT
  satırını trusted el etiketi sayar (``label_trust.py:2470``).
- P0-8: ``requeue_labeled_dataset_for_review`` hand_label / tekli
  human_approved kareyi kuyruğa geri döndürmez
  (``label_trust.py:3121`` + ``approval_state.box_is_never_requeue``);
  toplu onay (bulk_approved) bu korumaya girmez.
- K-21/P1-10: havuz kapısı kare düzeyi human_approved + kutu düzeyi
  insan kökeni ister; toplu onaylı kare havuza girmez ama kaynak
  TXT'si de silinmez.

Bu testler yalnızca tmp_path altında kendi dataset ağaçlarını kurar;
gerçek dataset dizinine dokunmaz. Testler bu turda KOŞULMADI — test
ajanları koşacak.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory.approved_pool import ApprovedDatasetPool
from dataset_factory.label_trust import (
    TRAIN_CONF_FLOOR,
    _rewrite_label_txt,
    commit_split_labels,
    requeue_labeled_dataset_for_review,
)

# Havuz bazlı beklenti (küçük ölçekli model): onaylı havuza giren
# kareler ve kutu sayıları. Canlı havuz 919 kare / 20.786 kutu;
# test aynı yapıyı 4 kare / 14 kutu ile modeller.
EXPECTED_POOL_BOXES = 14
EXPECTED_POOL_IMAGES = 4
# Havuza girmeyen kontrol karesi (K-21): toplu onay tek başına yetmez.
BULK_STEM = "frame_bulk"


def _jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    """Küçük ama gerçek JPEG baytları (SOI..EOI) — havuz görüntüsü için."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2 test JPEG'i kodlayamadı"
    data = bytes(buf)
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    return data


def _box(class_id, cx, cy, w=0.08, h=0.10, **flags):
    """Geometrisi QA'dan geçen kutu (w/h 0.004..0.65, merkez HUD dışı)."""
    d = {"class_id": int(class_id), "cx": float(cx), "cy": float(cy),
         "w": float(w), "h": float(h)}
    d.update(flags)
    return d


def _txt_line(b, with_conf=False):
    line = (f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} "
            f"{b['w']:.6f} {b['h']:.6f}")
    if with_conf:
        line += f" {float(b.get('confidence', 0.0)):.6f}"
    return line


def _txt_lines(boxes, with_conf=False):
    return "".join(_txt_line(b, with_conf) + "\n" for b in boxes)


def _parse_txt(path: Path):
    """TXT satırlarını (class_id, cx, cy, w, h) demetlerine çevirir."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(float(parts[0])), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])))
    return out


def _box_keys(boxes):
    return {(int(b["class_id"]), round(float(b["cx"]), 6),
             round(float(b["cy"]), 6), round(float(b["w"]), 6),
             round(float(b["h"]), 6)) for b in boxes}


def _pool_txt_lines(pool_root: Path, stem: str):
    """Havuz TXT'sini split aramadan okur (train veya val)."""
    for split in ("train", "val"):
        p = pool_root / "labels" / split / f"{stem}.txt"
        if p.is_file():
            return _parse_txt(p)
    return []


# ---------------------------------------------------------------------------
# Fixture: onaylı havuz bazlı sentetik dataset
# ---------------------------------------------------------------------------

def _frame_specs():
    """Onaylı havuzu modelleyen kare tanımları (trusted kutu sayıları)."""
    return {
        # 4 el etiketi kutu — havuzun omurgası.
        "frame_hand": {
            "trusted": [
                _box(6, 0.25, 0.35, hand_label=True),
                _box(0, 0.35, 0.55, hand_label=True),
                _box(2, 0.45, 0.45, hand_label=True),
                _box(9, 0.55, 0.65, hand_label=True),
            ],
            "weak": [],
            "human_approved": True,
            "txt_conf_lines": [],
        },
        # 3 tekli human_approved + 1 eşik üstü (0.90) model kutu.
        "frame_single": {
            "trusted": [
                _box(9, 0.30, 0.40, human_approved=True),
                _box(13, 0.50, 0.50, human_approved=True),
                _box(6, 0.65, 0.60, human_approved=True),
                _box(6, 0.40, 0.70, confidence=0.90),
            ],
            "weak": [],
            "human_approved": True,
            "txt_conf_lines": [],
        },
        # 3 trusted (2 el + 1 eşik üstü) + 2 taban-altı weak + 1 sıfır-alanlı.
        "frame_mixed": {
            "trusted": [
                _box(6, 0.25, 0.50, hand_label=True),
                _box(0, 0.45, 0.35, hand_label=True),
                _box(2, 0.60, 0.55, confidence=0.80),
            ],
            "weak": [
                _box(2, 0.35, 0.60, confidence=0.20),
                _box(9, 0.55, 0.40, confidence=0.20),
            ],
            "human_approved": True,
            "txt_conf_lines": [
                _box(2, 0.35, 0.60, confidence=0.20),
                _box(9, 0.55, 0.40, confidence=0.20),
            ],
            "extra_observations": [_box(6, 0.70, 0.70, w=0.0, h=0.10,
                                         confidence=0.90)],
        },
        # KONTROL (K-21): yalnızca toplu onay — havuza girmez.
        BULK_STEM: {
            "trusted": [
                _box(6, 0.30, 0.45, confidence=0.90, bulk_approved=True),
                _box(0, 0.55, 0.55, confidence=0.90, bulk_approved=True),
            ],
            "weak": [],
            "human_approved": False,
            "bulk_approved": True,
            "txt_conf_lines": [],
        },
        # Review JSON'suz, yalnızca 5 alanlı TXT — havuz TXT yolundan alır.
        "frame_txtonly": {
            "trusted": [
                _box(6, 0.30, 0.50, hand_label=True),
                _box(9, 0.50, 0.60, hand_label=True),
                _box(0, 0.65, 0.45, hand_label=True),
            ],
            "weak": [],
            "review_json": False,
        },
        # P0-1 karesi: trusted YOK, yalnızca taban-altı 6 alanlı TXT.
        "frame_stale": {
            "trusted": [],
            "weak": [
                _box(2, 0.30, 0.50, confidence=0.20),
                _box(9, 0.55, 0.65, confidence=0.20),
            ],
            "human_approved": False,
            "txt_conf_lines": [
                _box(2, 0.30, 0.50, confidence=0.20),
                _box(9, 0.55, 0.65, confidence=0.20),
            ],
        },
    }


@pytest.fixture
def pool_dataset(tmp_path):
    """Kaynak dataset + İLK dondurmayla dolmuş onaylı havuz.

    Dönen sözlük: root/dataset dizinleri, havuz, kare tanımları ve
    ilk dondurma sonrası havuz sayıları (baz çizgisi).
    """
    root = tmp_path / "project"
    ds = root / "dataset"
    specs = _frame_specs()
    jpeg = _jpeg_bytes()
    for stem, spec in specs.items():
        (ds / "images" / "train").mkdir(parents=True, exist_ok=True)
        (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (ds / "images" / "train" / f"{stem}.jpg").write_bytes(jpeg)
        txt_boxes = list(spec["trusted"]) + list(spec.get("txt_conf_lines", []))
        # frame_stale: TXT yalnızca 6 alanlı taban-altı satırlar.
        if stem == "frame_stale":
            txt_boxes = list(spec.get("txt_conf_lines", []))
        (ds / "labels" / "train" / f"{stem}.txt").write_text(
            _txt_lines(txt_boxes[:len(spec["trusted"])], with_conf=False)
            + _txt_lines(spec.get("txt_conf_lines", []), with_conf=True),
            encoding="utf-8",
        )
        if spec.get("review_json", True):
            (ds / "review" / "train").mkdir(parents=True, exist_ok=True)
            payload = {
                "stem": stem,
                "split": "train",
                "floor": TRAIN_CONF_FLOOR,
                "trusted": [dict(b, trusted=True, weak=False)
                            for b in spec["trusted"]],
                "weak": [dict(b, trusted=False, weak=True)
                         for b in spec["weak"]],
                "image_path": str(ds / "images" / "train" / f"{stem}.jpg"),
            }
            if spec.get("human_approved"):
                payload["human_approved"] = True
                payload["done"] = "ok"
            elif spec.get("bulk_approved"):
                payload["done"] = "ok"
                payload["bulk_approved"] = True
            (ds / "review" / "train" / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    pool = ApprovedDatasetPool(project_root=root)
    # İLK dondurma: havuz bu noktada dolar — bu, testlerin baz çizgisidir.
    first = pool.prepare_approved_training_dataset(
        source_dataset_dir=ds, val_ratio=0.15, force_rebalance=True
    )
    return {
        "root": root,
        "dataset": ds,
        "pool": pool,
        "pool_root": pool.pool_root,
        "specs": specs,
        "first_freeze": first,
    }


def _commit_frame(ds: Path, stem: str, spec):
    """Bir karede gerçek commit yolunu çalıştırır."""
    observations = [dict(b) for b in spec["trusted"]]
    observations += [dict(b) for b in spec["weak"]]
    observations += [dict(b) for b in spec.get("extra_observations", [])]
    return commit_split_labels(
        dataset_dir=ds,
        split="train",
        stem=stem,
        label_path=ds / "labels" / "train" / f"{stem}.txt",
        annotations=observations,
    )


def _commit_all_review_frames(ds: Path, specs):
    results = {}
    for stem, spec in specs.items():
        if not spec.get("review_json", True):
            continue
        results[stem] = _commit_frame(ds, stem, spec)
    return results


# ---------------------------------------------------------------------------
# (1) commit güvenilir kutuların tamamını korur
# ---------------------------------------------------------------------------

def test_commit_preserves_all_trusted_boxes(pool_dataset):
    """commit_split_labels: trusted kutuların tamamı TXT'de kalır."""
    ds = pool_dataset["dataset"]
    specs = pool_dataset["specs"]
    for stem, spec in specs.items():
        if not spec.get("review_json", True):
            continue
        result = _commit_frame(ds, stem, spec)
        assert result["train_box_count"] == len(spec["trusted"]), (
            f"{stem}: commit {result['train_box_count']} kutu yazdı, "
            f"trusted {len(spec['trusted'])} olmalıydı"
        )
        txt_keys = set(_parse_txt(ds / "labels" / "train" / f"{stem}.txt"))
        assert _box_keys(spec["trusted"]) <= txt_keys, (
            f"{stem}: trusted kutulardan TXT'de eksik var: "
            f"{_box_keys(spec['trusted']) - txt_keys}"
        )


# ---------------------------------------------------------------------------
# (2) elenen kutular sayılır — sessiz silme yok
# ---------------------------------------------------------------------------

def test_commit_counts_dropped_boxes_not_silently(pool_dataset):
    """frame_mixed: taban-altı kutular weak'te sayılır, sıfır-alanlı
    kutu dropped_qa'da sayılır; hiçbiri sessizce yok olmaz."""
    ds = pool_dataset["dataset"]
    spec = pool_dataset["specs"]["frame_mixed"]
    result = _commit_frame(ds, "frame_mixed", spec)
    assert result["train_box_count"] == 3
    # Taban-altı 2 kutu: TXT'ye girmez ama review weak listesinde durur.
    assert result["weak_box_count"] == 2
    weak_keys = _box_keys(result["weak"])
    assert _box_keys(spec["weak"]) <= weak_keys
    # Sıfır-alanlı kutu QA'da elendi ve sayacı tutulur.
    assert result["dropped_qa"] == 1
    assert result["dropped"] == 1
    assert result["dropped_sub_floor"] == 0  # weak'e düşen, dropped değil
    review = json.loads(
        (ds / "review" / "train" / "frame_mixed.json").read_text(
            encoding="utf-8"))
    assert len(review["weak"]) == 2
    assert len(review["trusted"]) == 3


# ---------------------------------------------------------------------------
# (3) dondurma: dolu havuzun kutu sayısı azalmamalı
# ---------------------------------------------------------------------------

def test_freeze_preserves_approved_pool_box_count(pool_dataset):
    """İlk dondurmada dolan havuz (baz) + commit turu + ikinci dondurma:
    havuz kutu sayısı 14'te kalır; toplu onaylı kontrol karesi havuza
    girmez ama kaynak TXT'si de silinmez."""
    ds = pool_dataset["dataset"]
    pool = pool_dataset["pool"]
    pool_root = pool_dataset["pool_root"]
    specs = pool_dataset["specs"]

    first = pool_dataset["first_freeze"]
    assert first["total_boxes"] == EXPECTED_POOL_BOXES
    assert first["total_approved_images"] == EXPECTED_POOL_IMAGES
    assert first["candidates_found"] == 3  # hand, single, mixed
    assert first["txt_hand_label_candidates_found"] == 1  # txtonly
    assert first["pruned_count"] == 0

    # Dondurma arası commit turu (11 Kasım'daki kayıp bu yazımda oldu).
    _commit_all_review_frames(ds, specs)

    second = pool.prepare_approved_training_dataset(
        source_dataset_dir=ds, val_ratio=0.15, force_rebalance=True
    )
    assert second["total_boxes"] == EXPECTED_POOL_BOXES, (
        f"dondurma havuzu {first['total_boxes']} -> "
        f"{second['total_boxes']} kutuya düşürdü"
    )
    assert second["total_approved_images"] == EXPECTED_POOL_IMAGES
    assert second["pruned_count"] == 0

    # Disk gerçeği: manifesto değil, havuz TXT'lerindeki satırlar sayılır.
    pool_txt_total = 0
    for stem in ("frame_hand", "frame_single", "frame_mixed",
                 "frame_txtonly"):
        lines = _pool_txt_lines(pool_root, stem)
        assert len(lines) == len(specs[stem]["trusted"]), (
            f"havuz TXT {stem}: {len(lines)} satır, "
            f"beklenen {len(specs[stem]['trusted'])}"
        )
        pool_txt_total += len(lines)
    assert pool_txt_total == EXPECTED_POOL_BOXES

    # K-21 kontrolü: toplu onaylı kare havuza girmedi...
    assert pool.get_sample_manifest(BULK_STEM) is None
    assert _pool_txt_lines(pool_root, BULK_STEM) == []
    # ...ama kaynağı duruyor (dondurma kaynak TXT'yi silmez).
    assert len(_parse_txt(ds / "labels" / "train" / f"{BULK_STEM}.txt")) == 2


# ---------------------------------------------------------------------------
# (4) tekrarlanan dondurma + commit döngüleri havuzu eritmemeli
# ---------------------------------------------------------------------------

def test_repeated_freeze_and_commit_cycles_do_not_shrink_pool(pool_dataset):
    """11 Kasım deseni (ard arda dondurma denemeleri): üç tur
    commit + dondurma sonrası havuz hâlâ 14 kutu olmalı."""
    ds = pool_dataset["dataset"]
    pool = pool_dataset["pool"]
    specs = pool_dataset["specs"]
    for cycle in range(3):
        _commit_all_review_frames(ds, specs)
        stats = pool.prepare_approved_training_dataset(
            source_dataset_dir=ds, val_ratio=0.15, force_rebalance=True
        )
        assert stats["total_boxes"] == EXPECTED_POOL_BOXES, (
            f"{cycle + 1}. döngüde havuz {stats['total_boxes']} kutuya düştü"
        )
        assert stats["total_approved_images"] == EXPECTED_POOL_IMAGES


# ---------------------------------------------------------------------------
# (5) P0-8: hand_label / tekli human_approved asla kuyruğa dönmez
# ---------------------------------------------------------------------------

def test_requeue_never_returns_protected_frames_p0_8(pool_dataset):
    """requeue: korumalı 4 kare atlanır, dosyaları bayt-aynı kalır;
    toplu onaylı kare kuyruğa düşer ama kutuları kaybolmaz."""
    ds = pool_dataset["dataset"]
    protected = ["frame_hand", "frame_single", "frame_mixed", "frame_txtonly"]
    before = {}
    for stem in protected + [BULK_STEM]:
        before[(stem, "txt")] = (
            ds / "labels" / "train" / f"{stem}.txt").read_bytes()
        review_path = ds / "review" / "train" / f"{stem}.json"
        before[(stem, "review")] = (
            review_path.read_bytes() if review_path.is_file() else None)

    result = requeue_labeled_dataset_for_review(ds)

    assert result["skipped_protected"] == 4, (
        f"korumalı kare sayısı {result['skipped_protected']}, 4 olmalıydı")
    assert result["samples"] == 2  # bulk + stale kuyruğa düştü
    for stem in protected:
        assert (ds / "labels" / "train" / f"{stem}.txt").read_bytes() == (
            before[(stem, "txt")]), f"{stem} TXT requeue'da değişti"
        review_path = ds / "review" / "train" / f"{stem}.json"
        if before[(stem, "review")] is not None:
            assert review_path.read_bytes() == before[(stem, "review")], (
                f"{stem} review JSON requeue'da değişti")

    # Toplu onaylı kare kuyruğa düştü: onayı sıfırlandı ama 2 kutusu durur.
    bulk_review = json.loads(
        (ds / "review" / "train" / f"{BULK_STEM}.json").read_text(
            encoding="utf-8"))
    assert bulk_review["human_approved"] is False
    assert bulk_review["requeued"] is True
    assert (len(bulk_review["trusted"]) + len(bulk_review["weak"])) == 2


# ---------------------------------------------------------------------------
# (6) P0-1: boş emit mevcut dolu TXT'yi silemez
# ---------------------------------------------------------------------------

def test_empty_emit_never_wipes_nonempty_txt_p0_1(pool_dataset):
    """frame_stale: yalnızca taban-altı kutulu commit'te emit boştur —
    TXT yazılmaz (txt_written False), eski TXT bayt-aynı kalır, weak
    kutular review JSON'da durur."""
    ds = pool_dataset["dataset"]
    label_path = ds / "labels" / "train" / "frame_stale.txt"
    before = label_path.read_bytes()
    assert len(_parse_txt(label_path)) == 2

    spec = pool_dataset["specs"]["frame_stale"]
    result = _commit_frame(ds, "frame_stale", spec)

    assert result["train_box_count"] == 0
    assert result["txt_written"] is False
    assert label_path.read_bytes() == before
    review = json.loads(
        (ds / "review" / "train" / "frame_stale.json").read_text(
            encoding="utf-8"))
    assert len(review["weak"]) == 2
    assert review["trusted"] == []


def test_rewrite_label_txt_refuses_empty_emit_on_nonempty_txt(pool_dataset):
    """P0-1 doğrudan: ``_rewrite_label_txt`` boş emit + mevcut dolu TXT
    → False döner, dosya değişmez."""
    label_path = pool_dataset["dataset"] / "labels" / "train" / "frame_stale.txt"
    before = label_path.read_bytes()
    assert len(_parse_txt(label_path)) == 2
    assert _rewrite_label_txt(label_path, [], allow_empty=False) is False
    assert label_path.read_bytes() == before


# ---------------------------------------------------------------------------
# (8) v19-pool-drop-count: havuz TXT yazımında elenen kutu sayılır
# ---------------------------------------------------------------------------

def test_freeze_counts_pool_dropped_boxes_by_reason(pool_dataset, caplog):
    """Dondurma yolunda havuz TXT'sine yazılamayan geçersiz kutu
    sayılır: kanonik-dışı class, sınır-dışı koordinat ve sıfır-alanlı
    kutu havuza girmez AMA dropped_pool_boxes sayacında nedenleriyle
    görünür; geçerli kutu havuza yazılır, K-21 kapısı kapalı kalır."""
    ds = pool_dataset["dataset"]
    pool = pool_dataset["pool"]
    pool_root = pool_dataset["pool_root"]

    stem = "frame_pool_drops"
    boxes = [
        _box(6, 0.30, 0.50, hand_label=True),  # geçerli — havuza girer
        _box(99, 0.30, 0.50),  # kanonik-dışı class (0..26 dışı)
        _box(6, 1.50, 0.50),  # sınır-dışı merkez
        _box(6, 0.30, 0.50, w=0.0, h=0.10),  # sıfır-alanlı
    ]
    (ds / "images" / "train" / f"{stem}.jpg").write_bytes(_jpeg_bytes())
    (ds / "labels" / "train" / f"{stem}.txt").write_text(
        _txt_lines(boxes, with_conf=False), encoding="utf-8")
    (ds / "review" / "train").mkdir(parents=True, exist_ok=True)
    payload = {
        "stem": stem,
        "split": "train",
        "floor": TRAIN_CONF_FLOOR,
        "trusted": [dict(b, trusted=True, weak=False) for b in boxes],
        "weak": [],
        "image_path": str(ds / "images" / "train" / f"{stem}.jpg"),
        "human_approved": True,
        "done": "ok",
    }
    (ds / "review" / "train" / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with caplog.at_level(logging.WARNING,
                         logger="DatasetFactory.ApprovedPool"):
        result = pool.prepare_approved_training_dataset(
            source_dataset_dir=ds, val_ratio=0.15, force_rebalance=True
        )

    # Geçersiz 3 kutu havuza girmedi ama sayaca nedenleriyle düştü.
    assert result["dropped_pool_boxes"] == 3
    assert result["dropped_pool_boxes_by_reason"] == {
        "non_canonical_class": 1,
        "out_of_bounds": 1,
        "zero_area": 1,
    }
    # Bulk düzeyi taşıma: aynı sayaç instance'da da durur.
    assert pool.last_bulk_dropped_boxes == {
        "non_canonical_class": 1,
        "out_of_bounds": 1,
        "zero_area": 1,
    }
    # Geçerli kutu havuza yazıldı; toplam 14 + 1 kutu, 4 + 1 kare.
    assert result["total_boxes"] == EXPECTED_POOL_BOXES + 1
    assert result["total_approved_images"] == EXPECTED_POOL_IMAGES + 1
    pool_lines = _pool_txt_lines(pool_root, stem)
    assert len(pool_lines) == 1
    assert _box_keys([boxes[0]]) <= set(pool_lines)
    # Elenen her kutu logda: kare kimliği + neden, tek satırda.
    assert stem in caplog.text
    for reason in ("non_canonical_class", "out_of_bounds", "zero_area"):
        assert reason in caplog.text
    # K-21 kapısı hâlâ kapalı: toplu onaylı kare havuza girmedi.
    assert pool.get_sample_manifest(BULK_STEM) is None
