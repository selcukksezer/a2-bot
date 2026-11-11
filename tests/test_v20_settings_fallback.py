"""v20 madde 7 regresyonu: SettingsManager.get fallback hatasi.

Canli v19 logu (15:10:37):
  "[Gatherion.StateMachine] [WARNING] Savas/toplama ayari ayristirilamadi
   (route:fight_back=\"cfg.get failed: SettingsManager.get() got an
   unexpected keyword argument 'fallback'\"); varsayilan kullaniliyor."

Kok neden: ``BotWorker._is_route_fight_back_enabled`` (core/state_machine.py)
``cfg.get("route", "fight_back", fallback=None)`` cagiriyordu ama
``SettingsManager.get(section, key, default)`` (config/schema.py) ve
``ConfigManager.get(section, key, default)`` (core/config.py) ``fallback``
parametresi almiyor. TypeError ``except Exception`` ile yutulup
``_warn_bad_config_once`` cagriliyor, fonksiyon hep ``False`` donuyordu —
kullanicinin ``route.fight_back=True`` ayari hic uygulanmiyordu.

Duzeltme: cagri tarafi ``default=None`` bicimine cevrildi; ``get()``
imzasi genisletilmedi.

Bu test, gercek ``SettingsManager.get`` imzasini birebir taklit eden
katı bir stub kullanir: ``fallback`` kwarg'i gecilirse TypeError
firlatir (duzeltme oncesi davranis), ``default`` ile cagrilirsa degeri
doner. Boylece regresyon, canli hatayi ayni sekilde yeniden uretir.

Calistirma (KOSULMADI — v20 kurali: pytest yasak, sadece py_compile):
  QT_QPA_PLATFORM=offscreen <venv>/bin/python -m pytest \\
      tests/test_v20_settings_fallback.py -q
"""
import os
import sys
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_qapp = QApplication.instance() or QApplication([])

from core.state_machine import BotWorker  # noqa: E402


class _StrictSettingsManagerStub:
    """Gercek SettingsManager.get(section, key, default) imzasi.

    ``fallback`` kwarg'i TypeError firlatir — tipki config/schema.py:898
    ve core/config.py:134'teki gercek imza gibi.
    """

    def __init__(self, values):
        self._values = dict(values)

    def get(self, section, key=None, default=None):
        return self._values.get((section, key), default)


def _make_bot(config):
    return BotWorker(
        config=config,
        vision=MagicMock(),
        input_driver=MagicMock(),
        route_manager=MagicMock(),
        combat_recovery=MagicMock(),
        navigator=MagicMock(),
        material_scanner=MagicMock(),
    )


class TestRouteFightBackSettingsFallback:
    def test_fight_back_true_read_via_settings_manager(self):
        cfg = _StrictSettingsManagerStub({("route", "fight_back"): True})
        worker = _make_bot(cfg)
        assert worker._is_route_fight_back_enabled() is True

    def test_fight_back_false_read_via_settings_manager(self):
        cfg = _StrictSettingsManagerStub({("route", "fight_back"): False})
        worker = _make_bot(cfg)
        assert worker._is_route_fight_back_enabled() is False

    def test_fight_back_when_attacked_fallback_key(self):
        cfg = _StrictSettingsManagerStub(
            {("route", "fight_back_when_attacked"): True}
        )
        worker = _make_bot(cfg)
        assert worker._is_route_fight_back_enabled() is True

    def test_unset_returns_false_without_warning(self):
        cfg = _StrictSettingsManagerStub({})
        worker = _make_bot(cfg)
        assert worker._is_route_fight_back_enabled() is False
        # Hata yutulup uyarilmis olsaydi _bad_config_warned dolardi.
        assert not worker.__dict__.get("_bad_config_warned")

    def test_no_fallback_kwarg_reaches_get(self):
        seen_kwargs = []

        class _RecordingStub(_StrictSettingsManagerStub):
            def get(self, section, key=None, default=None, **kwargs):
                seen_kwargs.append(kwargs)
                return super().get(section, key, default)

        worker = _make_bot(_RecordingStub({("route", "fight_back"): True}))
        assert worker._is_route_fight_back_enabled() is True
        # _make_bot + kontrol sirasinda get() bircok kez cagrilir;
        # onemli olan HICBIR cagriya `fallback` kwarg'inin ulasmamasidir.
        assert seen_kwargs
        assert all("fallback" not in kwargs for kwargs in seen_kwargs)
