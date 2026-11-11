# -*- coding: utf-8 -*-
"""R59 regression tests: canonical taxonomy in dataset_factory.multi_mix.

F-0882: F/E gather prompts are class 13 (skill_ui_prompt), not legacy 7
(herb_lumina) / 8 (ore_obsidian). A real prompt stays trusted.
S-2003: fusion must keep canonical classes 16..26 (bars, skills,
potions/scroll, kinah, player_character); only ids outside 0..26 drop.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.class_taxonomy import CANONICAL_NAMES
from dataset_factory.multi_mix import _CLASS_E, _CLASS_F, fuse_sources


def test_prompt_class_ids_are_canonical_skill_ui_prompt() -> None:
    assert _CLASS_F == 13
    assert _CLASS_E == 13
    assert CANONICAL_NAMES[13] == "skill_ui_prompt"
    assert CANONICAL_NAMES[7] == "herb_lumina"
    assert CANONICAL_NAMES[8] == "ore_obsidian"


def test_real_f_prompt_is_class_13_and_trusted() -> None:
    fused = fuse_sources(
        {"gather_prompt": {"cx": 0.5, "cy": 0.6, "gather_key": "f", "real_prompt": True}}
    )
    assert len(fused["annotations"]) == 1
    box = fused["annotations"][0]
    assert box["class_id"] == 13
    assert box["gather_key"] == "f"
    assert box["source"] == "f_gather_prompt"
    assert box["trusted"] is True


def test_real_e_prompt_is_class_13_and_trusted() -> None:
    fused = fuse_sources(
        {"gather_prompt": {"cx": 0.5, "cy": 0.6, "gather_key": "e", "real_prompt": True}}
    )
    assert len(fused["annotations"]) == 1
    box = fused["annotations"][0]
    assert box["class_id"] == 13
    assert box["gather_key"] == "e"
    assert box["source"] == "e_gather_prompt"
    assert box["trusted"] is True


def test_prompt_candidate_is_class_13_but_untrusted() -> None:
    fused = fuse_sources({"gather_prompt": {"cx": 0.5, "cy": 0.6, "gather_key": "f"}})
    assert len(fused["annotations"]) == 1
    box = fused["annotations"][0]
    assert box["class_id"] == 13
    assert box["source"] == "prompt_candidate"
    assert box["trusted"] is False


def test_fusion_keeps_canonical_classes_16_to_26() -> None:
    world = [
        {"class_id": cid, "cx": 0.2, "cy": 0.2, "w": 0.05, "h": 0.05, "confidence": 0.9}
        for cid in (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
    ]
    fused = fuse_sources({"yolo": world})
    kept = sorted(b["class_id"] for b in fused["annotations"])
    assert kept == [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


def test_fusion_drops_ids_outside_canonical_range() -> None:
    world = [
        {"class_id": cid, "cx": 0.2, "cy": 0.2, "w": 0.05, "h": 0.05, "confidence": 0.9}
        for cid in (-1, 27, 98)
    ]
    fused = fuse_sources({"yolo": world})
    assert fused["annotations"] == []


def test_fusion_keeps_core_classes_0_to_15() -> None:
    world = [
        {"class_id": cid, "cx": 0.2, "cy": 0.2, "w": 0.05, "h": 0.05, "confidence": 0.9}
        for cid in range(16)
    ]
    fused = fuse_sources({"yolo": world})
    kept = sorted(b["class_id"] for b in fused["annotations"])
    assert kept == list(range(16))
