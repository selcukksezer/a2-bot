# -*- coding: utf-8 -*-
"""
v19 telemetri düzeltmeleri — canlı eğitim gösterge hataları (KAPSAM EKİ madde 14).

Canlı kanıt (Selçuk'un PC'si, 2026-11-11): eğitim sağlıklı ilerliyordu
(epoch 20 mAP50 %51.61 → 30 %55.93 → 40 %55.17 → 50 %56.48 → 69 %56.1)
ama iki gösterge bozuktu:
  (1) Batch göstergesi "Batch 1/48 (%2)"de TAKILI — epoch ilerliyordu.
  (2) box_loss / cls_loss HER epoch'ta tam 0.0000.

Kök nedenler (ultralytics 8.4.210 ``engine/trainer.py`` ile teyitli):
  (1) ``BaseTrainer``'da ``batch`` diye bir öznitelik YOK, yalnızca
      ``batch_size`` var (satır ~173) — ``getattr(trainer, "batch", 0)``
      hep 0 dönüp ``cur_b``'yi 1'de kilitliyordu. Epoch-içi batch index'i
      ``_do_train``'deki ``for i, batch in pbar`` döngü değişkenidir ve
      trainer'a yazılmaz; ama ``on_train_batch_end`` her batch sonunda
      tam bir kez ateşlenir (satır ~640). Düzeltme: ``_BatchStepCounter``
      callback çağrılarını sayar.
  (2) ``trainer.tloss`` Ultralytics >=8.3'te dict'tir
      (``{"box_loss": ..., "cls_loss": ..., "dfl_loss": ...}``, epoch
      ortalaması; ``_do_train`` satır ~566, ``label_loss_items`` satır
      ~978). Eski kod dict üzerinde iterasyon yapıp ANAHTARLARI float'a
      çevirmeye çalışıyordu — ``float("box_loss")`` ValueError verip
      sessizce yutuluyor, loss hep 0.0 kalıyordu. Düzeltme:
      ``extract_epoch_losses`` dict'i anahtarla okur; alınamayan loss
      ``None`` döner ve UI "—" yazar (YANILTICI SIFIR YASAK).

Eğitim davranışı (epoch/mAP/erken durdurma) değiştirilmedi — yalnızca
telemetri/gösterge düzeltildi.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("QT_QPA_PLATFORM") is None and os.name == "posix" and not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_factory.rtx_trainer_engine import (
    _BatchStepCounter,
    extract_epoch_losses,
)
from ui.yolo_factory_dialog import _format_loss


class _FakeTrainer:
    """Ultralytics trainer'ın telemetri için okunan yüzeyi.

    ``batch`` özniteliği BİLEREK takılı bırakılır (canlıdaki gibi):
    kimi sürümde hiç yoktur, kimi sürümde ilerlemeyen bir değerdir —
    sayaç ona hiç bakmamalıdır.
    """

    def __init__(self, epoch: int = 0, total_batches: int = 48, batch_attr: object = 0):
        self.epoch = epoch
        self.epochs = 100
        self.train_loader = [object() for _ in range(total_batches)]
        self.batch = batch_attr  # takılı / ilerlemeyen — okunmamalı
        self.tloss = None


# ----------------------------------------------------------------------
# Sorun 1 — batch sayacı epoch içinde ilerler
# ----------------------------------------------------------------------

def test_v19_batch_counter_increments_within_epoch():
    trainer = _FakeTrainer(epoch=0, total_batches=48)
    counter = _BatchStepCounter()
    seen = [counter.next(trainer)[0] for _ in range(48)]
    assert seen == list(range(1, 49)), "batch 1..48 sırayla ilerlemeli (canlıda hep 1'di)"
    cur, total, cur_ep = counter.next(trainer)
    # 49. çağrı epoch-içi yeniden başlatma sayılır (OOM retry) -> 1'e sarar
    assert (cur, total, cur_ep) == (1, 48, 1)


def test_v19_batch_counter_ignores_trainer_batch_attr():
    # trainer.batch takılı da olsa, hiç olmasa da sayaç ilerler.
    stuck = _FakeTrainer(epoch=2, total_batches=48, batch_attr=0)
    counter = _BatchStepCounter()
    assert [counter.next(stuck)[0] for _ in range(5)] == [1, 2, 3, 4, 5]
    weird = _FakeTrainer(epoch=2, total_batches=48, batch_attr=7)
    counter2 = _BatchStepCounter()
    assert [counter2.next(weird)[0] for _ in range(3)] == [1, 2, 3]
    no_attr = _FakeTrainer(epoch=2, total_batches=48)
    del no_attr.batch
    counter3 = _BatchStepCounter()
    assert [counter3.next(no_attr)[0] for _ in range(3)] == [1, 2, 3]


def test_v19_batch_counter_resets_on_epoch_change():
    trainer = _FakeTrainer(epoch=0, total_batches=48)
    counter = _BatchStepCounter()
    assert [counter.next(trainer)[0] for _ in range(48)] == list(range(1, 49))
    trainer.epoch = 1  # yeni epoch
    assert counter.next(trainer) == (1, 48, 2)
    assert counter.next(trainer) == (2, 48, 2)


def test_v19_batch_counter_total_matches_loader_len():
    trainer = _FakeTrainer(epoch=0, total_batches=48)
    counter = _BatchStepCounter()
    cur, total, cur_ep = counter.next(trainer)
    assert (cur, total, cur_ep) == (1, 48, 1)
    # Canlı veri: 768 train karesi / batch 16 = 48 batch.


# ----------------------------------------------------------------------
# Sorun 2 — loss: tloss yokken None/"—", varken gerçek değer
# ----------------------------------------------------------------------

def test_v19_loss_missing_tloss_is_none_not_zero():
    assert extract_epoch_losses(None) == (None, None)
    trainer = _FakeTrainer()
    trainer.tloss = None
    assert extract_epoch_losses(trainer.tloss) == (None, None)
    # Zincir: None -> UI "—", asla "0.0000" değil.
    assert _format_loss(None) == "—"


def test_v19_loss_dict_real_values_pass_through():
    # Ultralytics >=8.3 tloss dict'i — canlıdaki regresyon: eski kod
    # anahtarları float'a çevirmeye çalışıp 0.0'a düşüyordu.
    tloss = {"box_loss": 1.234567, "cls_loss": 0.876543, "dfl_loss": 2.0}
    box, cls = extract_epoch_losses(tloss)
    assert box == pytest.approx(1.23457)
    assert cls == pytest.approx(0.87654)
    assert (box, cls) != (0.0, 0.0)


def test_v19_loss_dict_partial_is_half_missing():
    box, cls = extract_epoch_losses({"box_loss": 0.42})
    assert box == pytest.approx(0.42)
    assert cls is None
    assert _format_loss(box) == "0.4200"
    assert _format_loss(cls) == "—"


def test_v19_loss_legacy_sequence_still_works():
    # Eski Ultralytics sürümlerinde tloss sayı dizisiydi.
    assert extract_epoch_losses([2.5, 1.5, 0.5]) == (2.5, 1.5)
    assert extract_epoch_losses((0.123456, 0.654321)) == (0.12346, 0.65432)


def test_v19_loss_nan_and_garbage_are_missing():
    assert extract_epoch_losses({"box_loss": float("nan"), "cls_loss": 0.5}) == (None, 0.5)
    assert extract_epoch_losses({"box_loss": "bozuk", "cls_loss": None}) == (None, None)
    assert extract_epoch_losses("box_loss") == (None, None)
    assert extract_epoch_losses([0.5]) == (None, None)


def test_v19_loss_format_dash_not_zero():
    assert _format_loss(None) == "—"
    assert _format_loss(float("nan")) == "—"
    assert _format_loss("x") == "—"
    assert _format_loss(0.123456) == "0.1235"
    # Gerçek sıfır (CSV'den gelen gerçek 0.0) hâlâ 0.0000 yazılır —
    # yasak olan, ALINAMAYAN loss'u 0.0000 gibi göstermektir.
    assert _format_loss(0.0) == "0.0000"


def test_v19_missing_loss_chain_renders_dash_not_zero():
    # Uçtan uca zincir (canlı senaryo: tloss alınamadı):
    # epoch payload'u -> UI etiketi "box — · cls —" olmalı, "box 0.0000" değil.
    box, cls = extract_epoch_losses(None)
    label = f"box {_format_loss(box)} · cls {_format_loss(cls)}"
    assert label == "box — · cls —"
    assert "0.0000" not in label

    box2, cls2 = extract_epoch_losses({"box_loss": 1.11111, "cls_loss": 0.22222})
    label2 = f"box {_format_loss(box2)} · cls {_format_loss(cls2)}"
    assert label2 == "box 1.1111 · cls 0.2222"
