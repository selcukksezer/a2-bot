# -*- coding: utf-8 -*-
"""
D19 — navigation_intel iyileştirme testleri.

Kapsar:
- I1: yedek (fallback) VisionEngine / GameHUDReader örnekleri çağrı başına
      değil, gecikmeli (lazy) tek örnek olarak kurulur.
- I2: RadarStuckDetector / WaypointArrivalDetector / PartialNodeDetector
      history listeleri max_history ile sınırlıdır; sayaçlar korunur.
- I3: NavigationIntel.step_navigation_cycle durum taşıyan
      TurnDampingController kullanır (hızlı yön tersine çevirmede damping).
- I4: düğüm durum yazımı tek yardımcıda (_apply_node_harvest_state /
      _sync_resting_nodes) — davranış eşdeğerliği kilitlenir.
"""

import numpy as np
import pytest

import core.navigation_intel as ni


BLANK_FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


# ==============================================================================
# I1 — Paylaşılan yedek motorlar
# ==============================================================================
class TestFallbackEngineCaching:
    def test_hud_reader_constructed_once(self, monkeypatch):
        """GameHUDReader yedeği 5 çağrıda 1 kez kurulur, sonuçlar aynı kalır."""
        constructions = {"n": 0}

        class CountingHUDReader:
            def __init__(self):
                constructions["n"] += 1

            def detect_chat_system_alerts(self, frame):
                return {
                    "extraction_failed": False,
                    "cannot_mount": False,
                    "orange_warning_ratio": 0.0,
                }

        monkeypatch.setattr(ni, "_FALLBACK_HUD_READER", None)
        monkeypatch.setattr(ni, "GameHUDReader", CountingHUDReader)
        results = [ni._detect_chat_alerts_unified(frame=BLANK_FRAME) for _ in range(5)]
        assert constructions["n"] == 1
        assert all(r == results[0] for r in results)
        assert results[0]["extraction_failed"] is False

    def test_vision_engine_constructed_once(self, monkeypatch):
        """VisionEngine yedeği 5 çağrıda 1 kez kurulur, sonuçlar aynı kalır."""
        import core.vision as vision_mod

        constructions = {"n": 0}

        class CountingVision:
            def __init__(self):
                constructions["n"] += 1

            def detect_gather_prompt(self, frame):
                return (False, 0.0, None)

        monkeypatch.setattr(ni, "_FALLBACK_VISION_ENGINE", None)
        monkeypatch.setattr(vision_mod, "VisionEngine", CountingVision)
        results = [ni._detect_prompt_from_vision_or_frame(frame=BLANK_FRAME) for _ in range(5)]
        assert constructions["n"] == 1
        assert results == [(False, "f")] * 5

    def test_shared_fallback_identity(self, monkeypatch):
        """Aynı yedek örnek iki çağrıda da aynı nesnedir."""
        monkeypatch.setattr(ni, "_FALLBACK_HUD_READER", None)
        first = ni._get_fallback_hud_reader()
        second = ni._get_fallback_hud_reader()
        assert first is not None
        assert first is second

    def test_caller_provided_reader_preferred(self, monkeypatch):
        """Çağıran kendi hud_reader'ını verirse yedek hiç kurulmaz."""
        constructions = {"n": 0}

        class CountingHUDReader:
            def __init__(self):
                constructions["n"] += 1

        class ProvidedReader:
            def detect_chat_system_alerts(self, frame):
                return {
                    "extraction_failed": True,
                    "cannot_mount": False,
                    "orange_warning_ratio": 0.5,
                }

        monkeypatch.setattr(ni, "_FALLBACK_HUD_READER", None)
        monkeypatch.setattr(ni, "GameHUDReader", CountingHUDReader)
        res = ni._detect_chat_alerts_unified(frame=BLANK_FRAME, hud_reader=ProvidedReader())
        assert constructions["n"] == 0
        assert res["extraction_failed"] is True
        assert res["orange_warning_ratio"] == pytest.approx(0.5)

    def test_fallback_construction_failure_retries(self, monkeypatch):
        """Yedek kurulamazsa önbelleğe alınmaz: her çağrı yeniden dener, çökmez."""
        import core.vision as vision_mod

        constructions = {"n": 0}

        class FailingVision:
            def __init__(self):
                constructions["n"] += 1
                raise RuntimeError("no display")

        monkeypatch.setattr(ni, "_FALLBACK_VISION_ENGINE", None)
        monkeypatch.setattr(vision_mod, "VisionEngine", FailingVision)
        results = [ni._detect_prompt_from_vision_or_frame(frame=BLANK_FRAME) for _ in range(3)]
        assert constructions["n"] == 3
        assert results == [(False, "f")] * 3
        assert ni._FALLBACK_VISION_ENGINE is None


# ==============================================================================
# I2 — Sınırlı history
# ==============================================================================
class TestBoundedHistory:
    def test_waypoint_history_capped_counters_preserved(self):
        det = ni.WaypointArrivalDetector(max_history=5)
        nodes = [{"id": 1}, {"id": 2}]
        for i in range(10):
            det.check_and_transition(
                player_pos=(0.0, 0.0),
                waypoint_pos=(0.0, 0.0),
                nodes=nodes,
                current_index=i,
            )
        assert det.transitions_count == 10
        assert len(det.history) == 5
        # En eski 5 kayıt atıldı; kalanlar 5..9. varışlardır.
        assert [r["previous_index"] for r in det.history] == [5, 6, 7, 8, 9]
        assert isinstance(det.history, list)

    def test_stuck_history_capped_counters_preserved(self):
        det = ni.RadarStuckDetector(max_history=3)
        for _ in range(5):
            det.execute_unstick_maneuver(input_controller=None, sleep_fn=lambda s: None)
        assert det.stuck_count == 5
        assert len(det.history) == 3
        assert det.history[-1]["stuck_count"] == 5
        assert [r["stuck_count"] for r in det.history] == [3, 4, 5]

    def test_partial_history_capped_counters_preserved(self):
        det = ni.PartialNodeDetector(max_history=2, sleep_fn=lambda s: None)
        for _ in range(3):
            node = {"id": 7, "current_harvests": 1, "max_harvests": 3}
            res = det.check_and_handle_partial_node(
                node, completed_harvests=1, prompt_reappeared=False
            )
            assert res["is_partial_node"] is True
        assert det.partial_nodes_detected == 3
        assert len(det.history) == 2
        assert all(r["node_id"] == 7 for r in det.history)

    def test_default_history_cap_is_200(self):
        det = ni.WaypointArrivalDetector()
        assert det.max_history == 200
        nodes = [{"id": 1}, {"id": 2}]
        for i in range(250):
            det.check_and_transition(
                player_pos=(0.0, 0.0),
                waypoint_pos=(0.0, 0.0),
                nodes=nodes,
                current_index=i,
            )
        assert len(det.history) == 200
        assert det.transitions_count == 250


# ==============================================================================
# I3 — Birleşik döngüde reversal damping
# ==============================================================================
class TestStepCycleReversalDamping:
    def _intel_and_nodes(self):
        intel = ni.NavigationIntel(deadband_deg=6.0)
        nodes = [
            {"id": 1, "map_x": 100, "map_y": 100},
            {"id": 2, "map_x": 200, "map_y": 100},
        ]
        return intel, nodes

    def test_rapid_reversal_is_damped(self):
        intel, nodes = self._intel_and_nodes()
        # Adım 1: kuzeye bakan oyuncu, doğudaki hedef -> sağa dönüş (90°).
        c1 = intel.step_navigation_cycle(
            player_pos=(100.0, 100.0), player_heading=0.0,
            waypoint_pos=(190.0, 100.0), nodes=nodes, current_index=0, now=100.0,
        )
        assert c1["turn_data"]["turn_direction"] == "right"
        assert c1["turn_data"]["reversal_damped"] is False
        raw_dur = c1["turn_data"]["turn_duration_sec"]
        assert raw_dur == pytest.approx(
            ni.apply_turn_damping(90.0, deadband_deg=6.0)["turn_duration_sec"]
        )
        # Adım 2: 0.05 sn sonra ters yöne (sola) dönüş -> 0.65x damping.
        c2 = intel.step_navigation_cycle(
            player_pos=(100.0, 100.0), player_heading=90.0,
            waypoint_pos=(100.0, 10.0), nodes=nodes, current_index=0, now=100.05,
        )
        assert c2["turn_data"]["turn_direction"] == "left"
        assert c2["turn_data"]["reversal_damped"] is True
        assert c2["turn_data"]["turn_duration_sec"] == pytest.approx(raw_dur * 0.65, abs=0.002)

    def test_no_damping_after_cooldown(self):
        intel, nodes = self._intel_and_nodes()
        intel.step_navigation_cycle(
            player_pos=(100.0, 100.0), player_heading=0.0,
            waypoint_pos=(190.0, 100.0), nodes=nodes, current_index=0, now=100.0,
        )
        intel.step_navigation_cycle(
            player_pos=(100.0, 100.0), player_heading=90.0,
            waypoint_pos=(100.0, 10.0), nodes=nodes, current_index=0, now=100.05,
        )
        # Soğuma (0.12 sn) çoktan geçti: ters dönüş artık damping'siz.
        c3 = intel.step_navigation_cycle(
            player_pos=(100.0, 100.0), player_heading=0.0,
            waypoint_pos=(190.0, 100.0), nodes=nodes, current_index=0, now=101.5,
        )
        assert c3["turn_data"]["turn_direction"] == "right"
        assert c3["turn_data"]["reversal_damped"] is False
        assert c3["turn_data"]["turn_duration_sec"] == pytest.approx(
            ni.apply_turn_damping(90.0, deadband_deg=6.0)["turn_duration_sec"]
        )

    def test_stateless_wrapper_unchanged(self):
        """NavigationIntel.apply_turn_damping ham (durumsuz) sonucu vermeye devam eder."""
        intel = ni.NavigationIntel(deadband_deg=6.0)
        assert intel.apply_turn_damping(90.0) == ni.apply_turn_damping(90.0, deadband_deg=6.0)
        assert "reversal_damped" not in intel.apply_turn_damping(90.0)


# ==============================================================================
# I4 — Düğüm durum yazımı tek yardımcıda (davranış eşdeğerliği)
# ==============================================================================
class TestNodeStateHelperEquivalence:
    def test_partial_node_dict_sync(self):
        det = ni.PartialNodeDetector(sleep_fn=lambda s: None)
        node = {"id": 7, "current_harvests": 1, "max_harvests": 3}
        res = det.check_and_handle_partial_node(node, completed_harvests=1, prompt_reappeared=False)
        assert res["action"] == "MARKED_RESTING_FORCE_DEPLETE"
        assert node["current_harvests"] == 0
        assert node["state"] == "resting"
        assert node["rest_duration_sec"] == 300
        assert node["last_visited_timestamp"] > 0

    def test_full_harvest_dict_sync(self):
        det = ni.PartialNodeDetector(sleep_fn=lambda s: None)
        node = {"id": 8, "current_harvests": 3, "max_harvests": 3}
        res = det.check_and_handle_partial_node(node, completed_harvests=3, prompt_reappeared=False)
        assert res["action"] == "FULL_HARVEST_COMPLETED"
        assert node["current_harvests"] == 0
        assert node["state"] == "resting"
        assert node["rest_duration_sec"] == 300

    def test_harvest_controller_fallback_branch_writes_state(self):
        ctrl = ni.ThreeStageHarvestController(sleep_fn=lambda s: None)
        node = {"id": 9, "current_harvests": 0, "max_harvests": 3}
        start = ctrl.start_node(node)
        assert start["is_anchored"] is True
        # 1/3: tükenmedi -> current=1, state active, zaman damgası YOK.
        r1 = ctrl.on_cast_completed(prompt_reappeared=True)
        assert r1["status"] == "CONTINUE_NEXT_STAGE"
        assert node["current_harvests"] == 1
        assert node["state"] == "active"
        assert "last_visited_timestamp" not in node
        # 2/3: aynı.
        r2 = ctrl.on_cast_completed(prompt_reappeared=True)
        assert r2["status"] == "CONTINUE_NEXT_STAGE"
        assert node["current_harvests"] == 2
        # 3/3: tükendi -> current=0, state resting, zaman damgası VAR.
        r3 = ctrl.on_cast_completed(prompt_reappeared=True)
        assert r3["status"] == "COMPLETED"
        assert node["current_harvests"] == 0
        assert node["state"] == "resting"
        assert node["rest_duration_sec"] == 300
        assert node["last_visited_timestamp"] > 0
