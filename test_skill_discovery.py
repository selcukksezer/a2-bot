# -*- coding: utf-8 -*-
"""SkillDiscovery HP-delta classification tests."""

import os
import tempfile
import unittest

from core.skill_discovery import SkillDiscovery


class SkillDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lex = os.path.join(self.tmp.name, "lex.json")
        self.db = os.path.join(self.tmp.name, "eco.sqlite")
        self.disc = SkillDiscovery(lexicon_path=self.lex, db_path=self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_classify_attack(self):
        r = self.disc.classify_delta("1", 0.8, 0.5, 1.0, 1.0)
        self.assertEqual(r.label, "attack")
        self.assertEqual(self.disc.lexicon().get("1"), "attack")

    def test_classify_heal(self):
        r = self.disc.classify_delta("2", 0.5, 0.5, 0.4, 0.55)
        self.assertEqual(r.label, "defense_heal")

    def test_classify_buff(self):
        r = self.disc.classify_delta("3", 0.5, 0.5, 1.0, 1.0, buffs_before=0, buffs_after=2)
        self.assertEqual(r.label, "scroll_buff")

    def test_start_pending_and_complete(self):
        self.disc.start(slots=["1", "2"])
        self.assertTrue(self.disc.needs_discovery())
        self.disc.classify_delta("1", 1.0, 0.7, 1.0, 1.0)
        self.disc.classify_delta("2", 1.0, 0.9, 1.0, 1.0)
        self.assertFalse(self.disc.needs_discovery())
        self.assertTrue(os.path.isfile(self.lex))

    def test_tooltip_does_not_overwrite(self):
        self.disc.classify_delta("q", 1.0, 0.5, 1.0, 1.0)
        self.disc.confirm_tooltip_hint("q", "Restores HP")
        self.assertEqual(self.disc.lexicon()["q"], "attack")


if __name__ == "__main__":
    unittest.main()
