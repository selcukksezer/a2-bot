# -*- coding: utf-8 -*-
"""R21 dilimi regresyon testleri.

S-0402: dismount() tuş basımı başarısızsa binekten inilmiş sayılmaz.
F-0478: navigation_intel hazır bayrakları gerçek proba bağlıdır.
"""
from core.navigation_intel import (
    AutoMountManager,
    NavigationIntel,
    get_navigation_intel_summary,
)


class _MockInputController:
    def __init__(self) -> None:
        self.keys_pressed = []

    def press_key(self, key: str) -> None:
        self.keys_pressed.append(str(key).lower())


class TestDismountFailureS0402:
    def test_dismount_without_controller_reports_not_dismounted(self):
        mgr = AutoMountManager(input_controller=None)
        mgr.is_mounted = True
        before_ts = mgr.last_dismount_ts
        res = mgr.dismount()
        assert res["dismounted"] is False
        assert res["dismount_key_pressed"] is False
        assert res["gather_key_pressed"] is False
        assert mgr.is_mounted is True
        assert mgr.dismount_count == 0
        assert mgr.last_dismount_ts == before_ts

    def test_dismount_with_controller_still_works(self):
        ctrl = _MockInputController()
        mgr = AutoMountManager(input_controller=ctrl)
        mgr.is_mounted = True
        res = mgr.dismount()
        assert res["dismounted"] is True
        assert res["dismount_key_pressed"] is True
        assert mgr.is_mounted is False
        assert mgr.dismount_count == 1
        assert "c" in ctrl.keys_pressed  # Y-4: binek tuşu C (Selçuk kararı, 2026-11-11)

    def test_update_reports_dismount_failed_without_controller(self):
        mgr = AutoMountManager(input_controller=None)
        mgr.is_mounted = True
        res = mgr.update(distance_m=5.0, in_combat=False)
        assert res["action"] == "DISMOUNT_FAILED"
        assert res["dismounted"] is False
        assert res["is_mounted"] is True
        assert mgr.is_mounted is True

    def test_update_reports_dismount_with_controller(self):
        ctrl = _MockInputController()
        mgr = AutoMountManager(input_controller=ctrl)
        mgr.is_mounted = True
        res = mgr.update(distance_m=5.0, in_combat=False)
        assert res["action"] in ("DISMOUNT", "DISMOUNT_AND_GATHER")
        assert res["dismounted"] is True
        assert res["is_mounted"] is False


class TestNavigationIntelReadyProbeF0478:
    def test_default_intel_is_ready(self):
        intel = NavigationIntel()
        assert intel.get_navigation_intel_health_summary()["ready"] is True
        assert intel.get_navigation_intel_ready_flag()["ready"] is True
        assert intel.get_navigation_intel_version_info()["ready"] is True
        assert intel.get_navigation_intel_version_info()["version"] == "2.0.0"
        snap = intel.get_navigation_intel_diagnostic_snapshot()
        assert snap["ready"] is True
        assert snap["status"] == "ok"
        assert snap["stuck_count"] == 0
        assert snap["transitions_count"] == 0
        summary = get_navigation_intel_summary(intel)
        assert summary["intel_ready"] is True

    def test_missing_component_is_not_ready(self):
        intel = NavigationIntel()
        del intel.stuck_detector
        assert intel.get_navigation_intel_health_summary()["ready"] is False
        assert intel.get_navigation_intel_ready_flag()["ready"] is False
        assert intel.get_navigation_intel_version_info()["ready"] is False
        snap = intel.get_navigation_intel_diagnostic_snapshot()
        assert snap["ready"] is False
        assert snap["status"] == "degraded"
        assert snap["stuck_count"] is None
        assert get_navigation_intel_summary(intel)["intel_ready"] is False

    def test_invalid_config_is_not_ready(self):
        intel = NavigationIntel()
        intel.deadband_deg = 99.0
        assert intel.get_navigation_intel_health_summary()["ready"] is False
        assert intel.get_navigation_intel_ready_flag()["checks"]["deadband_in_range"] is False

    def test_bare_object_is_not_ready(self):
        summary = get_navigation_intel_summary(object())
        assert summary["intel_ready"] is False
        assert summary["checks"]["stuck_detector"] is False
