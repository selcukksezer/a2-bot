# -*- coding: utf-8 -*-
"""
tests/test_smart_consumables_and_scrolls.py
===========================================
W-suite entry point for the SmartConsumableManager + ManaPhaseCombat tests.

This file used to be a ~550-line byte-for-byte copy of the root-level
``test_smart_consumable_manager.py`` (same helpers, same test bodies, only
the class names carried a ``Suite`` suffix). It now re-runs that exact
suite through thin subclasses instead of duplicating it: the assertions,
test count and test ids are unchanged, and the single source of truth is
``test_smart_consumable_manager.py``.

Do NOT paste test bodies back into this file — add new coverage to the
root-level module (or to a new, genuinely different test module) instead.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path (same as before; the root-level test
# module lives there, not under tests/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_smart_consumable_manager import (
    make_blue_potion_crop,
    make_empty_slot_crop,
    make_gradient_crop,
    make_noisy_potion_crop,
    make_partial_fill_crop,
    make_red_potion_crop,
    make_scroll_crop,
    paint_consumable_frame,
)
import test_smart_consumable_manager as _suite_mod

# NOTE: the base classes are referenced ONLY through ``_suite_mod`` on
# purpose. pytest collects every unittest.TestCase subclass present in a
# test module's namespace — importing the bases under any name here would
# make pytest run each test a third time under the alias. Keep them out
# of this module's namespace.

__all__ = [
    "TestSmartConsumableManagerSuite",
    "TestManaPhaseCombatSmartConsumablesSuite",
    "make_red_potion_crop",
    "make_blue_potion_crop",
    "make_scroll_crop",
    "make_empty_slot_crop",
    "make_noisy_potion_crop",
    "make_gradient_crop",
    "make_partial_fill_crop",
    "paint_consumable_frame",
]


class TestSmartConsumableManagerSuite(_suite_mod.TestSmartConsumableManager):
    """Same suite as TestSmartConsumableManager, under the W-suite name."""


class TestManaPhaseCombatSmartConsumablesSuite(_suite_mod.TestManaPhaseCombatSmartConsumables):
    """Same suite as TestManaPhaseCombatSmartConsumables, under the W-suite name."""


if __name__ == "__main__":
    unittest.main()
