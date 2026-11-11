"""F-1741: stability_guards bileşen hazır bayrakları gerçek proba bağlıdır.

Önceden 5 bileşen sınıfının get_stability_guards_ready_flag() metodu
koşulsuz True dönüyordu. Artık her sınıf kendi yapılandırmasını denetleyen
_ready_probe()'a delege eder; bozuk eşikle kurulan bileşen hazır sayılmaz.
"""

import numpy as np
import pytest

from core.stability_guards import (
    CameraLockExceptionRule,
    CursorVisibilityAutomator,
    FiveFrameDeathConfirmationGuard,
    MultiCriteriaMapVerifier,
    SelfInputWhitelistWindow,
    StabilityGuards,
    get_shared_map_verifier,
    get_shared_stability_guards,
    get_stability_guards_ready_flag,
    get_stability_health_report,
    verify_map_open,
)


def test_default_components_report_ready():
    guards = StabilityGuards()
    assert guards.get_stability_guards_ready_flag()["ready"] is True
    assert guards.whitelist.get_stability_guards_ready_flag()["ready"] is True
    assert guards.camera_lock_rule.get_stability_guards_ready_flag()["ready"] is True
    assert guards.cursor_automator.get_stability_guards_ready_flag()["ready"] is True
    assert guards.death_guard.get_stability_guards_ready_flag()["ready"] is True
    assert guards.map_verifier.get_stability_guards_ready_flag()["ready"] is True


def test_death_guard_below_five_frames_not_ready():
    # Madde 13: gerçek ölüm en az 5 ardışık kare ister.
    assert FiveFrameDeathConfirmationGuard(required_frames=5).get_stability_guards_ready_flag()["ready"] is True
    assert FiveFrameDeathConfirmationGuard(required_frames=2).get_stability_guards_ready_flag()["ready"] is False


def test_map_verifier_below_two_criteria_not_ready():
    # Madde 14: en az 2/3 kriter eşiği gerekir.
    assert MultiCriteriaMapVerifier(min_criteria=2).get_stability_guards_ready_flag()["ready"] is True
    assert MultiCriteriaMapVerifier(min_criteria=1).get_stability_guards_ready_flag()["ready"] is False


def test_camera_lock_rule_without_centers_not_ready(monkeypatch):
    assert CameraLockExceptionRule().get_stability_guards_ready_flag()["ready"] is True
    monkeypatch.setattr(CameraLockExceptionRule, "KNOWN_LOCK_CENTERS", ())
    assert CameraLockExceptionRule().get_stability_guards_ready_flag()["ready"] is False


def test_whitelist_without_key_table_not_ready(monkeypatch):
    assert SelfInputWhitelistWindow().get_stability_guards_ready_flag()["ready"] is True
    monkeypatch.setattr(SelfInputWhitelistWindow, "KEY_TO_VK", {})
    assert SelfInputWhitelistWindow().get_stability_guards_ready_flag()["ready"] is False


def test_cursor_automator_empty_toggle_key_not_ready():
    assert CursorVisibilityAutomator().get_stability_guards_ready_flag()["ready"] is True
    # Not: __init__ bos toggle_key'i "alt"a dusurur; prob etkin degeri okur.
    automator = CursorVisibilityAutomator()
    automator.toggle_key = ""
    assert automator.get_stability_guards_ready_flag()["ready"] is False


def test_composite_degrades_when_a_component_degrades():
    guards = StabilityGuards()
    assert guards.get_stability_guards_ready_flag()["ready"] is True
    guards.death_guard = FiveFrameDeathConfirmationGuard(required_frames=2)
    assert guards.get_stability_guards_ready_flag()["ready"] is False


def test_shared_singleton_and_health_report_use_real_probes():
    assert get_shared_stability_guards() is get_shared_stability_guards()
    assert isinstance(get_shared_map_verifier(), MultiCriteriaMapVerifier)
    assert get_stability_guards_ready_flag()["ready"] is True
    report = get_stability_health_report()
    assert report["stability_guards_ready"] is True
    assert all(report["components"].values())


def test_verify_map_open_returns_false_without_frame():
    # F-0178 yardımcısı kendisi çalışır (kare yokken harita açık sayılmaz);
    # üretim çağıranı route_manager/explorer_mode tarafındadır (R03 dışı).
    assert verify_map_open(None) is False
    assert verify_map_open(np.zeros((10, 10, 3), dtype=np.uint8)) is False
