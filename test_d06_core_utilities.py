# -*- coding: utf-8 -*-
"""D06 dev-improvement tests (dev/D006).

I1 blacklist region index, I2 SlotEconomy shared connection, I3 RlRewardLog
shared connection + persistent JSONL handle, I4 lifetime debounce cap,
I5 ConfigManager atomic unknown-key merge.
"""
import json
import os
import random
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock

from core.blacklist_manager import BlacklistEntry, BlacklistManager
from core.config import ConfigManager
from core.lifetime_stats import LifetimeStatsStore
from core.rl_reward_log import RlRewardLog
from core.slot_economy import SlotEconomy


def _tmp_paths(testcase, *names):
    d = tempfile.TemporaryDirectory()
    testcase.addCleanup(d.cleanup)
    return d.name, [os.path.join(d.name, n) for n in names]


class BlacklistIndexTests(unittest.TestCase):
    def setUp(self):
        self.dir, (self.bl, self.st) = _tmp_paths(self, "blacklist.json", "settings.json")
        self.mgr = BlacklistManager(blacklist_file=self.bl, settings_file=self.st)

    def _fill(self, n=600, regions=("Altgard", "Verteron", "Morheim", "Eltnen", "Poeta", "Ishalgen")):
        rng = random.Random(42)
        entries = [
            BlacklistEntry(id=f"bl_{i}", region=regions[i % len(regions)],
                           map_x=rng.randrange(0, 5000), map_y=rng.randrange(0, 5000))
            for i in range(n)
        ]
        self.mgr._entries = entries
        self.mgr._rebuild_index()
        return entries

    def _brute(self, region, mx, my, norm_x=None, norm_y=None, threshold_px=25.0):
        from core.blacklist_manager import _resolve_canonical_region
        canon = _resolve_canonical_region(region)
        for e in self.mgr._entries:
            if self.mgr._matches(e, canon, mx, my, norm_x, norm_y, threshold_px):
                return True, e
        return False, None

    def test_index_matches_brute_force(self):
        self._fill()
        rng = random.Random(7)
        regions = ("Altgard", "Verteron", "Morheim", "Eltnen", "Poeta", "Ishalgen")
        for _ in range(500):
            region = rng.choice(regions)
            mx, my = rng.randrange(0, 5000), rng.randrange(0, 5000)
            got = self.mgr.is_blacklisted(region, mx, my)
            want = self._brute(region, mx, my)
            self.assertEqual(got[0], want[0], (region, mx, my))
            if want[1] is not None:
                self.assertEqual(got[1].id, want[1].id)

    def test_index_matches_brute_force_with_norm_coords(self):
        self._fill()
        rng = random.Random(11)
        for i, e in enumerate(self.mgr._entries[:50]):
            ok, hit = self.mgr.is_blacklisted(
                e.region, e.map_x + 5000, e.map_y + 5000,
                norm_x=e.norm_x or 0.5, norm_y=e.norm_y or 0.5)
            want_ok, _ = self._brute(e.region, e.map_x + 5000, e.map_y + 5000,
                                     norm_x=e.norm_x or 0.5, norm_y=e.norm_y or 0.5)
            self.assertEqual(ok, want_ok)

    def test_index_stays_consistent_across_mutations(self):
        self._fill(120)
        self.mgr.add_entry("Altgard", 10, 10)
        self.mgr.add_entry("Verteron", 20, 20)
        self.mgr.remove_entry(self.mgr._entries[0].id)
        self.mgr.remove_by_coordinate("Altgard", 10, 10)
        self.mgr.clear_all(region="Verteron")
        self.mgr.from_list([{"id": "x1", "region": "Morheim", "map_x": 1, "map_y": 2}])
        grouped = {}
        for e in self.mgr._entries:
            grouped.setdefault(e.region, []).append(e)
        self.assertEqual(
            {k: sorted(x.id for x in v) for k, v in self.mgr._by_region.items()},
            {k: sorted(x.id for x in v) for k, v in grouped.items()},
        )
        self.assertTrue(self.mgr.is_blacklisted("Morheim", 1, 2)[0])
        self.assertFalse(self.mgr.is_blacklisted("Altgard", 10, 10)[0])
        self.mgr.clear_all()
        self.assertEqual(self.mgr._by_region, {})
        self.assertEqual(self.mgr.count(), 0)


class SlotEconomySharedConnTests(unittest.TestCase):
    def setUp(self):
        self.dir, (self.db,) = _tmp_paths(self, "eco.sqlite")
        self.eco = SlotEconomy(db_path=self.db, explore_ratio=0.05)
        self.addCleanup(self._close_eco)

    def _close_eco(self):
        try:
            self.eco.close()
        except Exception:
            pass

    def test_wal_mode_active(self):
        with self.eco._connect() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

    def test_concurrent_writes_no_errors_and_counts_correct(self):
        dummy = mock.MagicMock()
        dummy.record_monster_kill.return_value = True
        dummy.record_gold.return_value = 1
        errors = []
        with mock.patch("core.lifetime_stats.get_lifetime_stats", return_value=dummy):
            def worker(n):
                try:
                    for i in range(50):
                        self.eco.record_kill(f"mob_{n}_{i % 5}", ttk_sec=5.0)
                        self.eco.record_drop(kina=10, slot_id="Slot_A_000_000")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
            threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        conn = sqlite3.connect(self.db)
        try:
            drops = conn.execute("SELECT COUNT(*) FROM drops").fetchone()[0]
            kills = conn.execute("SELECT SUM(samples) FROM monsters").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(drops, 400)
        self.assertEqual(kills, 400)

    def test_close_then_reuse_reconnects(self):
        self.eco.enter_slot("Slot_A_000_000", cx=10, cy=20, region="A")
        self.eco.close()
        self.assertIsNone(self.eco._conn)
        with mock.patch("core.lifetime_stats.get_lifetime_stats",
                        return_value=mock.MagicMock()):
            self.eco.record_kill("mob_x", ttk_sec=1.0)
        self.assertIsNotNone(self.eco._conn)
        self.assertEqual(len(self.eco.list_slots()), 1)

    def test_data_visible_to_new_instance_after_close(self):
        self.eco.enter_slot("Slot_A_000_000", cx=10, cy=20, region="A")
        with mock.patch("core.lifetime_stats.get_lifetime_stats",
                        return_value=mock.MagicMock()):
            self.eco.record_drop(kina=2000, slot_id="Slot_A_000_000")
        self.eco.close()
        eco2 = SlotEconomy(db_path=self.db, explore_ratio=0.05)
        try:
            slots = eco2.list_slots()
        finally:
            eco2.close()
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["slot_id"], "Slot_A_000_000")
        self.assertGreater(slots[0]["kph"], 0)


class RlRewardLogSharedConnTests(unittest.TestCase):
    def setUp(self):
        self.dir, (self.db, self.jl) = _tmp_paths(self, "rl.sqlite", "rl.jsonl")
        self.log = RlRewardLog(db_path=self.db, jsonl_path=self.jl)
        self.addCleanup(self._close_log)

    def _close_log(self):
        try:
            self.log.close()
        except Exception:
            pass

    def test_many_ticks_both_sinks_agree(self):
        for _ in range(200):
            s = self.log.log_tick(kina=10, damage_dealt=5, slot_id="Slot_A_000_000")
            self.assertTrue(s.persisted)
        self.log.close()
        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM rl_rewards").fetchone()[0]
        finally:
            conn.close()
        with open(self.jl, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        self.assertEqual(rows, 200)
        self.assertEqual(len(lines), 200)
        self.assertEqual(self.log.episode_summary()["samples"], 200)
        self.assertEqual(self.log.episode_summary()["write_errors"], 0)

    def test_close_then_reuse_reconnects(self):
        self.log.log_tick(kina=5)
        self.log.close()
        self.assertIsNone(self.log._conn)
        self.assertIsNone(self.log._jsonl_fh)
        s = self.log.log_tick(kina=5)
        self.assertTrue(s.persisted)
        self.assertEqual(self.log.episode_summary()["samples"], 2)

    def test_dead_sqlite_sink_jsonl_still_persists(self):
        bad = RlRewardLog(db_path=self.dir, jsonl_path=self.jl)  # db path is a directory
        try:
            s = bad.log_tick(kina=5)
        finally:
            bad.close()
        self.assertTrue(s.persisted)
        self.assertEqual(bad.episode_summary()["samples"], 1)
        self.assertEqual(bad.episode_summary()["write_errors"], 0)

    def test_concurrent_ticks(self):
        errors = []
        def worker():
            try:
                for _ in range(50):
                    self.log.log_tick(kina=1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.log.episode_summary()["samples"], 400)


class LifetimeDebounceCapTests(unittest.TestCase):
    def test_debounce_table_bounded_after_many_distinct_kills(self):
        _, (path,) = _tmp_paths(self, "lifetime.json")
        store = LifetimeStatsStore(path=path)
        # File IO is mocked: this test targets the in-memory debounce
        # table, not the fsync-per-record write path.
        with mock.patch.object(store, "_atomic_write_locked", return_value=None):
            for i in range(1500):
                self.assertTrue(store.record_monster_kill(mob_key=f"mob_{i}", force=True))
        self.assertLessEqual(len(store._last_kill_ts_by_mob), 1024)

    def test_recent_mob_still_debounced_after_pruning(self):
        _, (path,) = _tmp_paths(self, "lifetime.json")
        store = LifetimeStatsStore(path=path)
        with mock.patch.object(store, "_atomic_write_locked", return_value=None):
            for i in range(1500):
                store.record_monster_kill(mob_key=f"mob_{i}", force=True)
            # A mob killed just now is still inside the live window.
            store.record_monster_kill(mob_key="mob_recent", force=True)
            self.assertFalse(store.record_monster_kill(mob_key="mob_recent"))
            self.assertTrue(store.record_monster_kill(mob_key="mob_other"))
            self.assertEqual(store.snapshot()["monsters_killed"], 1502)


class ConfigAtomicMergeTests(unittest.TestCase):
    def test_save_preserves_unknown_keys_and_leaves_valid_json(self):
        _, (cfg,) = _tmp_paths(self, "settings.json")
        mgr = ConfigManager(config_path=cfg)
        mgr.config["my_custom_plugin"] = {"enabled": True, "level": 3}
        mgr.config.setdefault("gathering", {})["my_custom_opt"] = "keep-me"
        self.assertTrue(mgr.save())
        with open(cfg, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["my_custom_plugin"], {"enabled": True, "level": 3})
        self.assertEqual(data["gathering"]["my_custom_opt"], "keep-me")
        leftovers = [n for n in os.listdir(os.path.dirname(cfg)) if ".tmp." in n]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
