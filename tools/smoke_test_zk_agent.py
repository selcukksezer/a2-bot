# -*- coding: utf-8 -*-
"""
Gatherion ZK Vision Agent — uçtan uca duman testi (ZK-01 .. ZK-12).

Sentetik BGR kare + fake driver ile PerceptionHub / ManaPhase / Discovery /
SlotEconomy / Reward / SoftInput / ActionBus / BotWorker wiring / Config / log markers.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOGS_DIR = Path(PROJECT_ROOT) / "logs"
REPORT_PATH = LOGS_DIR / "smoke_zk_agent.json"


class _FakeHUD:
    def read_player_hp_mp_bars(self, frame):
        return {
            "hp_pct": 82.0,
            "mp_pct": 40.0,
            "has_target": True,
            "target_pct": 55.0,
            "is_monster_target": True,
            "has_secondary_target": False,
        }

    def parse_chat_feedback_fast(self, frame):
        return {}


class _FakeSkillBar:
    def inspect_skill_bar(self, frame):
        return {"1": {"ready": True}, "2": {"ready": True}, "Q": {"ready": False}}

    def select_best_ready_skill(self, frame):
        return "1"

    def slot_center_px(self, frame, key, jitter_px=0):
        return (400, 960)

    def mark_slot_pressed(self, key):
        return None


class _FakeDriver:
    def __init__(self, trusted_keyboard: bool = True):
        self.trusted_keyboard = trusted_keyboard
        self.keys: List[str] = []
        self.clicks: List[Tuple[int, int, str]] = []
        self._mouse_buttons_down = set()
        self.btns: List[Tuple[str, str]] = []
        self.held: List[Tuple[str, str, bool]] = []

    def keyboard_inject_trusted(self) -> bool:
        return bool(self.trusted_keyboard)

    def is_keyboard_trusted(self) -> bool:
        return bool(self.trusted_keyboard)

    def _hw_key_down(self, key, soft_fallback=False):
        self.held.append(("down", str(key).lower(), bool(soft_fallback)))
        return True

    def _hw_key_up(self, key, soft_fallback=False):
        self.held.append(("up", str(key).lower(), bool(soft_fallback)))
        return True

    def focus_game_window(self):
        return True

    def press_key(self, key, **kwargs):
        self.keys.append(str(key).lower())

    def human_click(self, x, y, button="left", **kwargs):
        self.clicks.append((int(x), int(y), str(button)))

    def _hw_mouse_down(self, button="left"):
        b = str(button).lower()
        self._mouse_buttons_down.add(b)
        self.btns.append(("down", b))

    def _hw_mouse_up(self, button="left"):
        b = str(button).lower()
        self._mouse_buttons_down.discard(b)
        self.btns.append(("up", b))


def run_zk_agent_smoke_tests(verbose: bool = True) -> Tuple[int, int, List[dict]]:
    results: List[dict] = []

    def _run(test_id: str, category: str, title: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            results.append(
                {
                    "id": test_id,
                    "category": category,
                    "name": title,
                    "passed": True,
                    "status": "GEÇTİ",
                    "elapsed_ms": elapsed_ms,
                    "detail": str(detail or "Başarılı"),
                }
            )
            if verbose:
                print(f"[PASS] {test_id} {title} ({elapsed_ms} ms) — {detail}")
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            tb_short = traceback.format_exc().splitlines()[-1]
            results.append(
                {
                    "id": test_id,
                    "category": category,
                    "name": title,
                    "passed": False,
                    "status": "KALDI",
                    "elapsed_ms": elapsed_ms,
                    "detail": f"{exc} ({tb_short})",
                }
            )
            if verbose:
                print(f"[FAIL] {test_id} {title} — {exc}")

    # ----- ZK-01 PerceptionHub -----
    def zk01() -> str:
        from core.perception_hub import FrameCognition, PerceptionHub

        hub = PerceptionHub(
            hud_reader=_FakeHUD(),
            skill_bar_reader=_FakeSkillBar(),
            min_interval_sec=10.0,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cog = hub.observe(frame, force=True)
        assert isinstance(cog, FrameCognition)
        assert abs(cog.player_hp - 0.82) < 0.02
        assert abs(cog.player_mp - 0.40) < 0.02
        assert cog.has_target
        assert "1" in cog.ready_skills
        d = cog.as_dict()
        assert "chat_events" in d and "gather_prompt" in d
        cached = hub.observe(frame, force=False)
        assert cached is cog
        return f"hp={cog.player_hp:.2f} mp={cog.player_mp:.2f} ready={cog.ready_skills}"

    _run("ZK-01", "Perception", "PerceptionHub observe + throttle", zk01)

    # ----- ZK-02 chat parse -----
    def zk02() -> str:
        from core.perception_hub import parse_chat_text_line

        k = parse_chat_text_line("You obtained 1,450 Kina")
        assert any(e.get("type") == "kina" and e.get("amount") == 1450 for e in k)
        i = parse_chat_text_line("You acquired Sapphire Gemstone")
        assert any(e.get("type") == "item" for e in i)
        d = parse_chat_text_line("Yellow Zaif defeated")
        assert any(e.get("type") == "defeat" for e in d)
        return "kina/item/defeat parsed"

    _run("ZK-02", "Perception", "parse_chat_text_line loot events", zk02)

    # ----- ZK-03 ManaPhase -----
    def zk03() -> str:
        from core.mana_phase_combat import ManaPhase, ManaPhaseCombat

        brain = ManaPhaseCombat(mp_full_threshold=0.95, hp_panic_threshold=0.5)
        a = brain.decide(player_mp=0.3, player_hp=1.0, ready_skills=["1", "2"])
        assert a.phase == ManaPhase.REGEN and a.hold_left and not a.hold_right
        b = brain.decide(player_mp=1.0, player_hp=1.0, ready_skills=["R"])
        assert b.phase == ManaPhase.BURST and b.hold_right and not b.hold_left
        brain.set_lexicon({"1": "defense_heal", "2": "attack"})
        p = brain.decide(
            player_mp=1.0,
            player_hp=0.2,
            ready_skills=["1", "2"],
            fallback_keys=["1", "2"],
        )
        assert p.skill_keys[0] == "1"
        return "regen/burst/panic OK"

    _run("ZK-03", "Combat", "ManaPhaseCombat regen/burst/panic", zk03)

    # ----- ZK-04 SkillDiscovery -----
    def zk04() -> str:
        from core.skill_discovery import SkillDiscovery

        tmp = tempfile.TemporaryDirectory()
        try:
            disc = SkillDiscovery(
                lexicon_path=os.path.join(tmp.name, "lex.json"),
                db_path=os.path.join(tmp.name, "eco.sqlite"),
            )
            r1 = disc.classify_delta("1", 0.9, 0.5, 1.0, 1.0)
            assert r1.label == "attack"
            r2 = disc.classify_delta("2", 0.5, 0.5, 0.4, 0.6)
            assert r2.label == "defense_heal"
            r3 = disc.classify_delta("3", 0.5, 0.5, 1.0, 1.0, 0, 2)
            assert r3.label == "scroll_buff"
            assert os.path.isfile(os.path.join(tmp.name, "lex.json"))
            disc.confirm_tooltip_hint("1", "Restores HP")
            assert disc.lexicon()["1"] == "attack"
            return f"labels={disc.lexicon()}"
        finally:
            tmp.cleanup()

    _run("ZK-04", "Discovery", "SkillDiscovery HP-delta + lexicon persist", zk04)

    # ----- ZK-05 SlotEconomy -----
    def zk05() -> str:
        from core.slot_economy import SlotEconomy

        tmp = tempfile.TemporaryDirectory()
        try:
            db = os.path.join(tmp.name, "economy.sqlite")
            eco = SlotEconomy(db_path=db, explore_ratio=0.0)
            score = eco.compute_slot_score(1000, 500, 100, 0.5)
            assert abs(score - 2800.0) < 1.0
            eco.enter_slot("Slot_A_000_000", cx=10, cy=20)
            eco.record_drop(kina=3000, item_name="Aria", slot_id="Slot_A_000_000")
            eco.enter_slot("Slot_B_001_001", cx=110, cy=120)
            eco.record_drop(kina=50, slot_id="Slot_B_001_001")
            pick = eco.pick_slot_ucb()
            assert pick is not None and pick.slot_id == "Slot_A_000_000"
            eco.record_obstacle(50, 60)
            eco.record_obstacle(52, 61)
            near = eco.obstacles_near(50, 60)
            assert near and int(near[0]["hits"]) >= 2
            return f"kph_pick={pick.slot_id} hits={near[0]['hits']}"
        finally:
            tmp.cleanup()

    _run("ZK-05", "Economy", "SlotEconomy KPH + UCB + obstacles", zk05)

    # ----- ZK-06 RlRewardLog -----
    def zk06() -> str:
        from core.rl_reward_log import RlRewardLog, compute_reward

        r = compute_reward(kina=100, damage_dealt=50, damage_taken=10, wasted_mana=1, stuck_time=1)
        assert r > 0
        tmp = tempfile.TemporaryDirectory()
        try:
            db = os.path.join(tmp.name, "eco.sqlite")
            jl = os.path.join(tmp.name, "r.jsonl")
            log = RlRewardLog(db_path=db, jsonl_path=jl)
            s = log.log_tick(kina=25, damage_dealt=10)
            assert os.path.isfile(jl)
            assert log.episode_summary()["samples"] == 1
            assert s.reward > 0
            return f"reward={s.reward:.2f}"
        finally:
            tmp.cleanup()

    _run("ZK-06", "RL", "RlRewardLog compute + persist", zk06)

    # ----- ZK-07 ObstacleMemory -----
    def zk07() -> str:
        from core.rl_reward_log import ObstacleMemory

        mem = ObstacleMemory(slot_economy=None)
        mem.stuck_threshold_sec = 0.0
        assert mem.note_position(1.0, 2.0, moving_cmd=True) is None
        stuck = mem.note_position(1.0, 2.0, moving_cmd=True)
        assert stuck is not None and stuck >= 0.0
        return f"stuck_for={stuck:.3f}s"

    _run("ZK-07", "Navigation", "ObstacleMemory stuck threshold", zk07)

    # ----- ZK-08 SoftInput -----
    def zk08() -> str:
        from core.hardware_input_port import (
            SerialHidAdapter,
            SoftInputAdapter,
            build_default_input_port,
        )

        d = _FakeDriver()
        port = SoftInputAdapter(d)
        assert port.send_key("1")
        assert port.mouse_button("left", True)
        assert port.mouse_button("left", False)
        assert d.keys == ["1"]
        hid = SerialHidAdapter(port_name="")
        assert not hid.available
        assert isinstance(build_default_input_port(d), SoftInputAdapter)
        return "soft_ok hid_unavailable"

    _run("ZK-08", "Input", "SoftInputAdapter + SerialHid stub", zk08)

    # ----- ZK-09 GameActionBus -----
    def zk09() -> str:
        from core.game_action_bus import GameActionBus
        from core.skill_bar_reader import SkillBarReader

        d = _FakeDriver()
        released: List[str] = []
        restored: List[str] = []
        bus = GameActionBus(
            input_driver=d,
            skill_bar_reader=SkillBarReader(),
            release_basic_attack=lambda: released.append("up"),
            restore_basic_attack=lambda: restored.append("down"),
            mouse_look_mode=True,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = bus.cast_skill("2", frame=frame)
        assert out["ok"] and out["method"] == "key"
        assert released == [] and restored == []
        bus.hold_left_attack(True)
        bus.hold_left_attack(False)
        assert ("down", "left") in d.btns and ("up", "left") in d.btns
        return f"cast={out['method']} left_hold_ok"

    _run("ZK-09", "Input", "GameActionBus cast + left-hold", zk09)

    # ----- ZK-10 BotWorker wiring -----
    def zk10() -> str:
        from test_false_action_gates import _worker

        worker = _worker()
        # None-safe plain attrs
        for name in (
            "perception_hub",
            "mana_phase_combat",
            "skill_discovery",
            "slot_economy",
            "rl_reward_log",
            "input_port",
        ):
            _ = worker._plain_attr(name)

        # Attach mana brain + left sync path for regen
        from core.mana_phase_combat import ManaPhaseCombat

        logs: List[str] = []
        brain = ManaPhaseCombat(mp_full_threshold=0.95, on_log=logs.append)
        worker.mana_phase_combat = brain
        worker.perception_hub = None
        worker.skill_discovery = None
        worker.game_learner = None
        worker._frame_cognition = None

        mouse = _FakeDriver()
        worker.input_driver = mouse
        worker.action_bus = None
        worker._mode = "combat"
        worker._paused = False
        worker.combat_recovery.bar = True
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0
        # Force burst path (mp defaults 100% from empty cognition)
        worker._step_combat_mode()
        left_attack_ok = "left" in mouse._mouse_buttons_down or ("down", "left") in mouse.btns

        # Regen path: low mp decision + left hold via sync
        d2 = brain.decide(player_mp=0.2, player_hp=1.0, ready_skills=["1"])
        worker._sync_combat_left_button(bool(d2.hold_left))
        left_regen_ok = "left" in mouse._mouse_buttons_down or ("down", "left") in mouse.btns
        assert left_attack_ok, "combat tick must hold left in burst"
        assert left_regen_ok, "regen decide must drive left hold"
        assert any("MANA_PHASE" in m for m in logs)
        return f"left_attack={left_attack_ok} left_regen={left_regen_ok} logs={logs[:2]}"

    _run("ZK-10", "Integration", "BotWorker combat wiring + left regen", zk10)

    # ----- ZK-11 Config -----
    def zk11() -> str:
        from config.schema import RouteSettings

        rs = RouteSettings.from_dict(
            {
                "mp_full_threshold": 0.9,
                "ucb_explore_ratio": 0.25,
                "skill_discovery_enabled": False,
            }
        )
        assert abs(rs.mp_full_threshold - 0.9) < 1e-6
        assert abs(rs.ucb_explore_ratio - 0.25) < 1e-6
        assert rs.skill_discovery_enabled is False
        # clamp
        rs2 = RouteSettings.from_dict({"mp_full_threshold": 2.0, "ucb_explore_ratio": 0.9})
        assert rs2.mp_full_threshold <= 1.0
        assert rs2.ucb_explore_ratio <= 0.5

        from test_false_action_gates import _worker

        w = _worker()
        # Alias path via _get_cfg when settings absent → default
        v = w._get_cfg("mp_full_threshold", 0.95)
        assert float(v) > 0.0
        return f"mp={rs.mp_full_threshold} ucb={rs.ucb_explore_ratio}"

    _run("ZK-11", "Config", "RouteSettings ZK fields + _get_cfg", zk11)

    # ----- ZK-12 Log markers -----
    def zk12() -> str:
        from core.mana_phase_combat import ManaPhaseCombat
        from core.skill_discovery import SkillDiscovery
        from core.slot_economy import SlotEconomy

        logs: List[str] = []
        brain = ManaPhaseCombat(on_log=logs.append)
        brain.decide(player_mp=0.1)
        assert any("MANA_PHASE" in x for x in logs)

        tmp = tempfile.TemporaryDirectory()
        try:
            disc = SkillDiscovery(
                lexicon_path=os.path.join(tmp.name, "lex.json"),
                db_path=os.path.join(tmp.name, "eco.sqlite"),
                on_log=logs.append,
            )
            disc.classify_delta("1", 1.0, 0.4, 1.0, 1.0)
            assert any("DISCOVER" in x for x in logs)

            eco = SlotEconomy(
                db_path=os.path.join(tmp.name, "eco2.sqlite"),
                explore_ratio=0.0,
                on_log=logs.append,
            )
            eco.enter_slot("Slot_X_000_000", cx=1, cy=2)
            eco.record_drop(kina=500, slot_id="Slot_X_000_000")
            eco.pick_slot_ucb()
            assert any("SLOT_UCB" in x for x in logs)
            return "MANA_PHASE DISCOVER SLOT_UCB present"
        finally:
            tmp.cleanup()

    _run("ZK-12", "Telemetry", "Log markers MANA_PHASE/DISCOVER/SLOT_UCB", zk12)

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    return passed, failed, results


def write_zk_smoke_report(
    passed: int,
    failed: int,
    results: List[dict],
) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": "ZK_VISION_AGENT",
        "overall_passed": failed == 0 and passed > 0,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "ts": time.time(),
        "checks": results,
    }
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    passed, failed, results = run_zk_agent_smoke_tests(verbose=True)
    path = write_zk_smoke_report(passed, failed, results)
    print(f"\nZK smoke: {passed}/{passed + failed} PASS → {path}")
    return 0 if failed == 0 and passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
