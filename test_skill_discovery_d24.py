# -*- coding: utf-8 -*-
"""D24 SkillDiscovery improvement: tooltip hint precedence.

New file only — existing test_skill_discovery.py is NOT modified.
"""

import os
import tempfile
import unittest

from core.skill_discovery import SkillDiscovery


class SkillDiscoveryD24Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def _fresh(self):
        self._n += 1
        return SkillDiscovery(
            lexicon_path=os.path.join(self.tmp.name, f"l{self._n}.json"),
            db_path=os.path.join(self.tmp.name, f"e{self._n}.sqlite"),
        )

    def test_tooltip_corpus(self):
        # D24: buff-describing phrases ("attack speed", "physical attack")
        # must win over the generic "attack" word; damage/heal wording
        # keeps its old labels.
        corpus = [
            ("Attack Speed +15%", "scroll_buff"),
            ("Increases Physical Attack by 25", "scroll_buff"),
            ("Deals 500 damage to the target", "attack"),
            ("Restores 300 HP over 5 seconds", "defense_heal"),
            ("Scroll of Courage", "scroll_buff"),
            ("Strike the enemy for heavy damage", "attack"),
        ]
        for i, (text, expected) in enumerate(corpus):
            with self.subTest(text=text):
                disc = self._fresh()
                got = disc.confirm_tooltip_hint(f"s{i}", text)
                self.assertEqual(got, expected, f"{text!r} -> {got}")

    def test_tooltip_increases_damage_stays_attack(self):
        # Guard against over-hoisting: "increases" alone must not turn a
        # damage skill into a buff.
        disc = self._fresh()
        got = disc.confirm_tooltip_hint("x", "Increases damage by 10%")
        self.assertEqual(got, "attack")


if __name__ == "__main__":
    unittest.main()
