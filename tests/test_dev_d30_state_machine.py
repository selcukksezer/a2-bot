"""D30 (dev) tests: lock in the two state_machine.py improvements.

I1: _get_cfg aliases hoisted to module-level _CFG_ALIASES — behaviour
    must be identical to the old per-call literal (alias resolution,
    section fallback, defaults, break_every_min special case).
I2: is_material_selected_for_gather fast path + memo — results must
    match the plain match_material_names loop for every input, the
    normalized-selection cache must follow direct assignment, and the
    memo must stay bounded.

Run: QT_QPA_PLATFORM=offscreen <venv>/bin/python -m pytest \\
    tests/test_dev_d30_state_machine.py -q
"""
import os
import sys
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_qapp = QApplication.instance() or QApplication([])

import core.state_machine as sm  # noqa: E402
from core.material_scanner import match_material_names  # noqa: E402
from core.state_machine import BotWorker  # noqa: E402


def _make_bot(config=None):
    return BotWorker(
        config=config if config is not None else {},
        vision=MagicMock(),
        input_driver=MagicMock(),
        route_manager=MagicMock(),
        combat_recovery=MagicMock(),
        navigator=MagicMock(),
        material_scanner=MagicMock(),
    )


# ---------------------------------------------------------------- I1
class TestGetCfgAliasesHoisted:
    def test_module_level_alias_table_exists(self):
        assert isinstance(sm._CFG_ALIASES, dict)
        assert sm._CFG_ALIASES["f_interval_min"] == (
            (None, "f_interval_min"),
            ("gathering", "f_interval_min"),
            ("gathering", "min_f_interval"),
        )
        assert sm._CFG_ALIASES["combat_skill_keys"] == (
            (None, "combat_skill_keys"),
            (None, "skill_keys"),
            ("combat", "combat_skill_keys"),
            ("combat", "skill_keys"),
            ("route", "combat_skill_keys"),
            ("route", "skill_keys"),
        )

    def test_alias_resolution_flat_and_sectioned(self):
        bot = _make_bot(config={
            "f_interval_min": 1.1,
            "gathering": {"f_interval_max": 2.2, "min_f_interval": 9.9},
            "combat": {"skill_keys": ["r", "t"]},
            "route": {"rest_minutes": 7},
        })
        # Flat key wins over the sectioned alias.
        assert bot._get_cfg("f_interval_min", 1.5) == 1.1
        # Sectioned value is found; the later min_f_interval alias is not needed.
        assert bot._get_cfg("f_interval_max", 2.8) == 2.2
        # skill_keys alias resolves combat_skill_keys.
        assert bot._get_cfg("combat_skill_keys", []) == ["r", "t"]
        assert bot._get_cfg("rest_minutes", 5) == 7

    def test_alias_fallback_to_renamed_key(self):
        bot = _make_bot(config={"gathering": {"min_f_interval": 3.3}})
        assert bot._get_cfg("f_interval_min", 1.5) == 3.3

    def test_unknown_key_returns_default(self):
        bot = _make_bot(config={"gathering": {}})
        assert bot._get_cfg("no_such_key_xyz", "dflt") == "dflt"
        assert bot._get_cfg("no_such_key_xyz") is None

    def test_none_config_returns_default(self):
        bot = _make_bot(config={})
        bot.config = None
        assert bot._get_cfg("f_interval_min", 1.5) == 1.5

    def test_break_every_min_disabled_breaks(self):
        bot = _make_bot(config={"breaks": {"enabled": False}})
        assert bot._get_cfg("break_every_min", 30) == 0
        bot2 = _make_bot(config={"breaks": {"enabled": False, "break_every_min": 45}})
        # F-23-06: breaks.enabled=false molayı kapatır — break_every_min
        # değeri ne olursa olsun 0 döner.
        assert bot2._get_cfg("break_every_min", 30) == 0

    def test_sample_f_interval_still_uses_cfg(self):
        bot = _make_bot(config={"gathering": {"f_interval_min": 2.0, "f_interval_max": 2.0}})
        # HumanTimingEngine adds a small Ex-Gaussian jitter around the
        # configured interval; the point is the interval still comes
        # from _get_cfg (far from the 1.5/2.8 defaults).
        assert bot._sample_f_interval() == pytest.approx(2.0, abs=0.5)


# ---------------------------------------------------------------- I2
SELECTED = ["Orichalcum Ore", "Aria", "Edana", "Lumina", "Obsidian"]

# (candidate, expected) — expected computed by hand from the documented
# match_material_names semantics; the loop-comparison test below also
# checks every pair against the reference implementation directly.
CANDIDATES = [
    "Orichalcum Ore", "orichalcum ore", "ORICHALCUM ORE", "ore_orichalcum",
    "Orichalcum", "orichalcum cevheri", "Aria", "aria", "Fresh Aria",
    "Edana", "Lumina", "Obsidian", "Obsidian Ore", "obsidyen",
    "Mela", "Targena", "Blood Crystal", "Sapphire", "Tree", "Yggdrasil Log",
    "", "   ",
]


def _reference_is_selected(selected, mat_str):
    return any(match_material_names(t, mat_str) for t in selected)


class TestMaterialSelectedFastPath:
    def test_results_match_reference_loop(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        for cand in CANDIDATES:
            got = bot.is_material_selected_for_gather(cand)
            want = _reference_is_selected(SELECTED, cand)
            assert got == want, f"{cand!r}: got {got}, reference {want}"

    def test_results_match_reference_loop_after_repeated_calls(self):
        # Second pass must hit the memo and still agree with the reference.
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        for _ in range(3):
            for cand in CANDIDATES:
                assert bot.is_material_selected_for_gather(cand) == \
                    _reference_is_selected(SELECTED, cand)

    def test_exact_hit_uses_fast_path(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        assert bot.is_material_selected_for_gather("Orichalcum Ore") is True
        # Fast path must not have needed the memo at all.
        assert bot.__dict__.get("_match_names_memo", {}) == {}

    def test_memo_populated_on_alias_and_miss(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        assert bot.is_material_selected_for_gather("ore_orichalcum") is True
        assert bot.is_material_selected_for_gather("Mela") is False
        memo = bot.__dict__.get("_match_names_memo", {})
        assert len(memo) > 0
        assert memo[("Orichalcum Ore", "ore_orichalcum")] is True
        assert memo[("Orichalcum Ore", "Mela")] is False

    def test_cache_follows_direct_assignment(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(["Aria"])
        assert bot.is_material_selected_for_gather("Aria") is True
        assert bot.is_material_selected_for_gather("Mela") is False
        # Direct assignment (bypassing the setter) must still be honoured.
        bot.selected_gather_materials = ["Mela"]
        assert bot.is_material_selected_for_gather("Mela") is True
        assert bot.is_material_selected_for_gather("Aria") is False

    def test_empty_selection_means_all(self):
        bot = _make_bot()
        bot.set_selected_gather_materials([], enabled=False)
        assert bot.is_material_selected_for_gather("Mela") is True

    def test_node_dict_and_object_inputs(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        assert bot.is_material_selected_for_gather({"resource_type": "Orichalcum Ore"}) is True
        assert bot.is_material_selected_for_gather({"resource_type": "Mela"}) is False
        node = MagicMock()
        node.resource_type = "Aria"
        # _get_node_field on a MagicMock returns a MagicMock for every
        # field, so use a plain object instead for a deterministic check.
        class Node:
            resource_type = "Edana"
        assert bot.is_material_selected_for_gather(Node()) is True
        assert bot.is_material_selected_for_gather(None) is False
        assert bot.is_material_selected_for_gather("") is False

    def test_memo_stays_bounded(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(SELECTED)
        for i in range(600):
            bot.is_material_selected_for_gather(f"UnknownMaterial{i}")
        memo = bot.__dict__.get("_match_names_memo", {})
        assert len(memo) <= 1024
        # Results stay correct after the bound-clear path ran.
        assert bot.is_material_selected_for_gather("Orichalcum Ore") is True
        assert bot.is_material_selected_for_gather("Mela") is False

    def test_turkish_chars_and_underscores(self):
        bot = _make_bot()
        bot.set_selected_gather_materials(["Obsidyen Cevheri"])
        assert bot.is_material_selected_for_gather("obsidian ore") is True
        assert bot.is_material_selected_for_gather("obsidyen_cevheri") is True
