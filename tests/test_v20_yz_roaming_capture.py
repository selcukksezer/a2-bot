# -*- coding: utf-8 -*-
"""
v20 YZ kare çekimi (roaming) düzeltmeleri — birim testleri (fix/v20-yz-kare-cekimi).

Canlı kanıt (11 Kasım 2026, v20, PC — Selçuk):
- "toplama yada oto keşif yaparkende yz çekim yapmıyor 0 çekim"
  (8 Kasım 11:25'ten beri bilinen sorun: "oto kayıttada sıkıntı var
  ... yz kare çekimi yapmıyor hiç").
- 17:24-17:29 logunda "YZ kare kaydedildi" satırı hiç yok; GameLearner
  kareleri sürekli "commit 0 kutu yazdı (trusted_in=0)" ile düşüyor.

Kök neden (kod kanıtı):
- Dataset factory yalnızca hasat bitince çalışır: on_gather_success
  çağrı noktaları tüm repoda 3 yerdir ve üçü de hasat bitişidir
  (core/state_machine.py:3615 AUTO_RECORD, :5820 ROUTE hasat aşaması,
  core/explorer_mode.py:2708 EXPLORER fırsatçı hasat). Yürüyüş/keşif
  sırasında fabrika hiç kare kaydetmez — feed_live_frame yalnızca
  time-machine tamponunu doldurur, tampon hasat bitmeden diske yazılmaz.
- Düzeltme: _run_passive_auto_cartography tick'inde (route/explorer/
  auto_record/manual_record — savaşta hiç çalışmaz) kısılmış periyodik
  çekim (varsayılan 30 sn); fabrika tarafında roaming=True ile aynı
  kayıt yolu, ayrı sayaç/etiket/stem — hasat istatistiği kirlenmez.

NOT: Bu dosya YAZILIR, KOŞULMAZ — testleri ana ajan koşar.
Windows-only kısımlar mock'lanır (winsound/keyboard stub'ları aşağıda).
"""
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "winsound" not in sys.modules:
    _winsound = types.ModuleType("winsound")
    _winsound.Beep = lambda *a, **k: None
    sys.modules["winsound"] = _winsound
if "keyboard" not in sys.modules:
    _keyboard = types.ModuleType("keyboard")
    _keyboard.add_hotkey = lambda *a, **k: None
    _keyboard.remove_hotkey = lambda *a, **k: None
    _keyboard.is_pressed = lambda *a, **k: False
    _keyboard.unhook_all = MagicMock(name="unhook_all")
    sys.modules["keyboard"] = _keyboard

from core.state_machine import BotWorker  # noqa: E402
from dataset_factory.factory_controller import AutoLabelingDatasetFactory  # noqa: E402


class _FakeFactory:
    """on_gather_success çağrılarını kaydeden sahte fabrika."""

    def __init__(self):
        self.calls = []

    def on_gather_success(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"saved": True, "roaming": bool(kwargs.get("roaming", False))}


def _make_worker(mode="route", state="WALKING", factory=None):
    """BotWorker.__new__ ile kurulan işçi (loop testlerindeki desen).

    mode/state salt-okunur property'dir — _mode/_state + _lock kurulur.
    """
    worker = BotWorker.__new__(BotWorker)
    worker._lock = threading.RLock()
    worker._mode = mode
    worker._state = state
    worker.config = {}
    worker.dataset_factory = factory
    worker._last_yz_roaming_capture_ts = 0.0
    return worker


class TestRoamingCaptureThrottle(unittest.TestCase):
    """(1) Kanca route modunda 30 sn kısma ile bir kez ateşler."""

    def test_first_call_fires_with_roaming_true(self):
        factory = _FakeFactory()
        worker = _make_worker(mode="route", state="WALKING", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertEqual(len(factory.calls), 1)
        _, kwargs = factory.calls[0]
        self.assertTrue(kwargs.get("roaming"))
        self.assertEqual(kwargs.get("material_name"), "")
        self.assertEqual(kwargs.get("harvest_stage"), 1)
        self.assertEqual(kwargs.get("max_harvests"), 1)

    def test_second_immediate_call_is_throttled(self):
        factory = _FakeFactory()
        worker = _make_worker(mode="route", state="WALKING", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())
        worker._maybe_capture_yz_roaming_frame(object())
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertEqual(len(factory.calls), 1, "kısma yok: her tick çekim yapıyor")

    def test_call_fires_again_after_interval(self):
        factory = _FakeFactory()
        worker = _make_worker(mode="route", state="WALKING", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())
        # 31 sn geçmiş gibi göster (varsayılan aralık 30 sn).
        worker._last_yz_roaming_capture_ts = time.time() - 31.0
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertEqual(len(factory.calls), 2)


class TestRoamingCaptureGates(unittest.TestCase):
    """(2) Kanca savaşta/duraklatmada/fabrikasız çalışmaz, çökmez."""

    def test_combat_mode_never_captures(self):
        factory = _FakeFactory()
        worker = _make_worker(mode="combat", state="FIGHTING", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertEqual(len(factory.calls), 0, "savaşta çekim yapılmamalı")

    def test_explorer_and_auto_record_modes_capture(self):
        for mode in ("explorer", "auto_record", "manual_record"):
            factory = _FakeFactory()
            worker = _make_worker(mode=mode, state="WALKING", factory=factory)
            worker._maybe_capture_yz_roaming_frame(object())
            self.assertEqual(len(factory.calls), 1, f"{mode} modunda çekim yok")

    def test_paused_state_never_captures(self):
        factory = _FakeFactory()
        worker = _make_worker(mode="route", state="PAUSED", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertEqual(len(factory.calls), 0, "duraklatmada çekim yapılmamalı")

    def test_missing_factory_warns_once_and_never_crashes(self):
        worker = _make_worker(mode="route", state="WALKING", factory=None)
        worker._maybe_capture_yz_roaming_frame(object())
        worker._maybe_capture_yz_roaming_frame(object())
        self.assertTrue(worker.__dict__.get("_yz_roaming_factory_warned", False))

    def test_factory_without_on_gather_success_is_skipped(self):
        worker = _make_worker(mode="route", state="WALKING", factory=object())
        worker._maybe_capture_yz_roaming_frame(object())  # çökmemeli
        self.assertTrue(worker.__dict__.get("_yz_roaming_factory_warned", False))

    def test_factory_exception_does_not_break_hook(self):
        factory = _FakeFactory()

        def _boom(*args, **kwargs):
            raise RuntimeError("etiketleyici arızası")

        factory.on_gather_success = _boom
        worker = _make_worker(mode="route", state="WALKING", factory=factory)
        worker._maybe_capture_yz_roaming_frame(object())  # çökmemeli


class TestFactoryRoamingCounters(unittest.TestCase):
    """(3) Gerçek fabrika: roaming sayacı hasat sayacından ayrıdır."""

    def _factory(self):
        tmp = tempfile.mkdtemp(prefix="yz_roaming_test_")
        return AutoLabelingDatasetFactory(project_root=tmp)

    def test_roaming_call_uses_roaming_counter_only(self):
        factory = self._factory()
        # Boş kare + boş tampon: kayıt reddedilir ama sayaçlar ayrılır.
        result = factory.on_gather_success(current_frame=None, roaming=True)
        self.assertFalse(result["saved"])
        self.assertTrue(result["roaming"])
        self.assertEqual(factory._roaming_events_total, 1)
        self.assertEqual(factory._gather_events_total, 0)
        self.assertEqual(factory._roaming_samples_rejected, 1)
        self.assertEqual(factory._gather_samples_rejected, 0)

    def test_gather_call_uses_gather_counter_only(self):
        factory = self._factory()
        result = factory.on_gather_success(current_frame=None)
        self.assertFalse(result["saved"])
        self.assertFalse(result["roaming"])
        self.assertEqual(factory._gather_events_total, 1)
        self.assertEqual(factory._roaming_events_total, 0)
        self.assertEqual(factory._gather_samples_rejected, 1)
        self.assertEqual(factory._roaming_samples_rejected, 0)

    def test_roaming_respects_auto_label_flag(self):
        factory = self._factory()
        factory.auto_label_on_gather = False
        result = factory.on_gather_success(current_frame=None, roaming=True)
        self.assertFalse(result["saved"])
        self.assertEqual(result["reason"], "auto_label_on_gather_disabled")
        self.assertEqual(factory._roaming_events_total, 1)
        self.assertEqual(factory._gather_events_total, 0)


if __name__ == "__main__":
    unittest.main()
