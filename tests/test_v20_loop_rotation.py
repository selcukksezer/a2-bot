# -*- coding: utf-8 -*-
"""
v20-loop-rotation düzeltmeleri — birim testleri (fix/v20-loop-rotation).

Canlı kanıt (11 Kasım 2026, v20, PC):
- Ana savaş yolu (MANA_PHASE) yalnızca '2' + 'q' basıyordu; karşı saldırı
  yolu 8 tuşu da basıyordu.
- Döngü ~1 FPS (avg_latency 824 → 1126 ms); ONNX hot-reload 4 sn'de 2 kez;
  wrong-node taraması her düğüm hatasında 2 CPU session derliyordu.
- Toplamada karakter sabit duruyor, "Node 40 not reached in 58s: skipping
  (mesafe hiç tazelenmedi)" ile düğüm sessizce siliniyordu.
- Exemplar repair Targena (2) → Orichalcum (3) çeviriyordu.

NOT: Bu dosya YAZILIR, KOŞULMAZ — testleri ana ajan koşar.
Windows-only kısımlar mock'lanır (winsound/keyboard stub'ları aşağıda).
"""
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "winsound" not in sys.modules:
    _winsound = types.ModuleType("winsound")
    _winsound.Beep = lambda *a, **k: None
    sys.modules["winsound"] = _winsound
if "keyboard" not in sys.modules:
    _keyboard = types.ModuleType("keyboard")
    _keyboard.add_hotkey = lambda *a, **k: None
    _keyboard.remove_hotkey = lambda *a, **k: None
    _keyboard.is_pressed = lambda *a, **k: False
    _keyboard.unhook_all = MagicMock(name="unhook_all")
    sys.modules["keyboard"] = _keyboard

from core.mana_phase_combat import ManaPhaseCombat  # noqa: E402
from dataset_factory.class_taxonomy import is_same_material_family  # noqa: E402

# Canlı PC sözlüğü (dataset/skill_lexicon.json): 1,3 scroll_buff;
# 4,5,e,r unknown; 2,q attack; t sözlükte YOK (varsayılan "attack").
LIVE_LEXICON = {
    "1": "scroll_buff",
    "2": "attack",
    "3": "scroll_buff",
    "4": "unknown",
    "5": "unknown",
    "q": "attack",
    "e": "unknown",
    "r": "unknown",
}
ALL_KEYS = {"1", "2", "3", "4", "q", "e", "r", "t"}


class TestLiveLexiconRotation(unittest.TestCase):
    """(1) Ana savaş yolu canlı sözlükle [2,q]'ya çökmez."""

    def _combat(self, lexicon):
        combat = ManaPhaseCombat()
        combat.set_lexicon(dict(lexicon))
        return combat

    def test_burst_spans_full_hotbar_with_live_lexicon(self):
        combat = self._combat(LIVE_LEXICON)
        seen = set()
        for i in range(10):
            decision = combat.decide(
                player_mp=1.0,
                player_hp=1.0,
                ready_skills=["2", "q"],
                fallback_keys=["1", "2", "3", "4", "q", "e", "r", "t"],
                now=1000.0 + i,
            )
            seen.update(str(k) for k in decision.skill_keys)
        # BURST tam hotbar'ı döndürür: 10 turda 8 tuşun tamamı görülmeli.
        self.assertEqual(seen, ALL_KEYS, f"yalnızca {sorted(seen)} görüldü")

    def test_burst_spans_full_hotbar_even_if_pc_lexicon_marks_t_unknown(self):
        # PC'deki canlı sözlük "t"yi unknown sayıyor olabilir (VM'deki
        # dosyada "t" hiç yok). "unknown" artık "attack" sayıldığı için
        # t yine de rotasyona girmeli.
        lexicon = dict(LIVE_LEXICON)
        lexicon["t"] = "unknown"
        combat = self._combat(lexicon)
        seen = set()
        for i in range(10):
            decision = combat.decide(
                player_mp=1.0,
                player_hp=1.0,
                ready_skills=["2", "q"],
                fallback_keys=["1", "2", "3", "4", "q", "e", "r", "t"],
                now=2000.0 + i,
            )
            seen.update(str(k) for k in decision.skill_keys)
        self.assertIn("t", seen)
        self.assertEqual(seen, ALL_KEYS)

    def test_regen_pairs_include_t_over_time(self):
        # REGEN count=2 çift verir ama havuz [2,q,t] döner; 't' de çıkar.
        combat = self._combat(LIVE_LEXICON)
        seen = set()
        for i in range(9):
            decision = combat.decide(
                player_mp=0.90,
                player_hp=1.0,
                ready_skills=["2", "q"],
                fallback_keys=["1", "2", "3", "4", "q", "e", "r", "t"],
                now=3000.0 + i,
            )
            self.assertEqual(len(decision.skill_keys), 2)
            seen.update(str(k) for k in decision.skill_keys)
        self.assertIn("t", seen)
        self.assertIn("2", seen)
        self.assertIn("q", seen)


class TestRoundRobinAdvance(unittest.TestCase):
    """(2) Küçük havuzda dilim ilerler — [2,q] → [q,2]."""

    def test_two_key_pool_rotates(self):
        combat = ManaPhaseCombat()
        first = combat.get_round_robin_burst_slice(
            count=2, pool=["2", "q"], allow_combo=False, now=1000.0
        )
        second = combat.get_round_robin_burst_slice(
            count=2, pool=["2", "q"], allow_combo=False, now=1001.0
        )
        self.assertEqual(first, ["2", "q"])
        self.assertEqual(second, ["q", "2"])

    def test_filter_skills_unknown_counts_as_attack(self):
        combat = ManaPhaseCombat()
        combat.set_lexicon(dict(LIVE_LEXICON))
        out = combat._filter_skills(["1", "2", "4", "e"], allow=("attack",))
        # 1 scroll_buff elenir; 4/e "unknown" artık attack sayılır.
        self.assertEqual(out, ["2", "4", "e"])


class TestExemplarFamilyGate(unittest.TestCase):
    """(4) Repair aile içinde kalır; bitki → cevher çevrilmez."""

    def test_herb_to_ore_blocked(self):
        self.assertFalse(is_same_material_family(2, 3))  # herb_targena -> ore

    def test_same_family_allowed(self):
        self.assertTrue(is_same_material_family(2, 1))  # targena -> mela
        self.assertTrue(is_same_material_family(3, 8))  # orichalcum -> obsidian
        self.assertTrue(is_same_material_family(4, 12))  # crystal -> blood
        self.assertTrue(is_same_material_family(5, 5))  # tree -> tree

    def test_tree_to_herb_blocked(self):
        self.assertFalse(is_same_material_family(5, 0))

    def test_non_material_classes_pass_through(self):
        # 3D materyal olmayan sınıfları süper-kategori kapısı yönetir.
        self.assertTrue(is_same_material_family(9, 10))
        self.assertTrue(is_same_material_family(10, 10))


class TestWrongNodeDetectorCache(unittest.TestCase):
    """(2b) wrong-node detektörü her çağrıda yeniden derlenmez."""

    def test_second_load_returns_same_instance(self):
        from core import wrong_node_scan as wns

        class _FakeDetector:
            def __init__(self, path, **kwargs):
                self.model_path = path
                self.num_classes = 27
                self.device = "CPUExecutionProvider"

            def matches_dataset(self, names):
                return True

            def is_gatherion_subset(self, names):
                return True

        import tempfile

        root = Path(tempfile.mkdtemp(prefix="v20-loop-wns-"))
        (root / "models").mkdir()
        (root / "dataset").mkdir()
        model = root / "models" / "yolov8n_gatherion.onnx"
        model.write_bytes(b"fake-onnx")
        (root / "dataset" / "data.yaml").write_text(
            "nc: 2\nnames:\n  0: herb_aria\n  1: herb_targena\n", encoding="utf-8"
        )
        old_detector = wns.GatherionOnnxDetector
        wns.GatherionOnnxDetector = _FakeDetector
        try:
            wns._DETECTOR_CACHE.clear()
            wns._DETECTOR_CACHE_SIG.clear()
            first = wns.load_gatherion_detector(str(root))
            second = wns.load_gatherion_detector(str(root))
            self.assertIs(second, first, "ikinci yükleme aynı nesneyi dönmeli")
            # Dosya değişirse önbellek geçersiz sayılır.
            st = os.stat(model)
            os.utime(model, (st.st_atime, st.st_mtime + 10))
            third = wns.load_gatherion_detector(str(root))
            self.assertIsNot(third, first, "değişen dosya yeni session istemeli")
        finally:
            wns.GatherionOnnxDetector = old_detector
            wns._DETECTOR_CACHE.clear()
            wns._DETECTOR_CACHE_SIG.clear()


def _make_bot():
    """BotWorker iskeleti: __new__ ile, yalnızca test edilen metotlar gerçek."""
    from core.state_machine import BotWorker

    bot = BotWorker.__new__(BotWorker)
    bot._lock = threading.RLock()
    bot._paused = False
    bot.active_node = None
    bot._walk_start_time = 0.0
    bot._arrival_at_node_ts = 0.0
    bot._last_reclick_time = 0.0
    bot._walk_distance_ever_fresh = False
    bot._active_node_distance_updated_ts = 0.0
    bot._odometry_dist_ts = 0.0
    bot.active_node_distance_m = 999.0
    bot._last_known_map_open = False
    bot.route_manager = None
    bot.vision = None
    return bot


class _FakeRouteManager:
    def __init__(self):
        self.nodes = {}
        self.saved_regions = []

    def get_node(self, node_id, region=None):
        return self.nodes.get(int(node_id))

    def save_region(self, region):
        self.saved_regions.append(region)


class TestNavFailureRest(unittest.TestCase):
    """(3) Mesafe hiç tazelenmediyse düğüm silinmez, dinlenmeye alınır."""

    def _setup_timeout_walk(self, ever_fresh):
        bot = _make_bot()
        now = time.time()
        bot.active_node = {"id": 40, "map_x": 1001.0, "map_y": 532.0}
        bot._walk_start_time = now - 61.0
        bot._walk_distance_ever_fresh = ever_fresh
        bot._active_node_distance_updated_ts = 0.0
        bot.active_node_distance_m = 999.0
        bot.logs = []
        bot._emit_log = bot.logs.append
        bot._get_cfg = lambda name, default=None: {
            "time_to_reach_sec": 60,
            "reclick_node_sec": 10,
        }.get(name, default)
        bot._get_player_movement_coords = lambda: (0.0, 0.0)
        bot._extract_node_id = lambda n: 40
        bot._is_opportunistic_node = lambda n: False
        bot.failed = []
        bot.registered = []
        bot._mark_node_failed = lambda node_id: bot.failed.append(node_id)
        bot._register_node_failure = lambda: bot.registered.append(True)
        rm = _FakeRouteManager()
        node = SimpleNamespace(id=40, region="Altgard", state="ready")
        rm.nodes[40] = node
        bot.route_manager = rm
        bot._node = node
        return bot, now

    def test_never_fresh_distance_rests_node(self):
        bot, now = self._setup_timeout_walk(ever_fresh=False)
        bot._monitor_walking_to_node(now)
        self.assertEqual(bot.failed, [], "mesafe hiç tazelenmediyse fail sayılmaz")
        self.assertEqual(bot.registered, [])
        self.assertEqual(bot._node.state, "resting")
        self.assertIsNone(bot.active_node)
        self.assertTrue(
            any("NAVİGASYON HATASI" in line for line in bot.logs),
            f"logda görünür hata yok: {bot.logs}",
        )

    def test_fresh_distance_still_fails_node(self):
        bot, now = self._setup_timeout_walk(ever_fresh=True)
        bot._monitor_walking_to_node(now)
        self.assertEqual(bot.failed, [40], "mesafe tazelendiyse eski fail yolu sürer")
        self.assertEqual(bot.registered, [True])
        self.assertEqual(bot._node.state, "ready")


class TestMapBlocksMovement(unittest.TestCase):
    """(3b) Tek bozuk kare (None) yürüme kilitlemez."""

    def test_single_none_frame_does_not_block(self):
        bot = _make_bot()
        results = [None, True, None]
        bot.vision = SimpleNamespace(is_map_open=lambda: results.pop(0))
        self.assertFalse(bot._map_blocks_movement())  # None, hiç bilinmiyor -> açık sayılmaz
        self.assertTrue(bot._map_blocks_movement())  # True -> engeller
        self.assertTrue(bot._map_blocks_movement())  # None -> SON BİLİNEN (True) kullanılır

    def test_exception_still_blocks(self):
        bot = _make_bot()

        def _boom():
            raise RuntimeError("vision hatası")

        bot.vision = SimpleNamespace(is_map_open=_boom)
        self.assertTrue(bot._map_blocks_movement())


class TestOdometryDistance(unittest.TestCase):
    """(3c) Harita açılamazsa mesafe odometriden tazelenir."""

    def test_odometry_refreshes_stale_distance(self):
        bot = _make_bot()
        now = time.time()
        bot.active_node = {"id": 14, "map_x": 110.0, "map_y": 100.0}
        bot._active_node_distance_updated_ts = 0.0
        bot._get_cfg = lambda name, default=None: {"reclick_node_sec": 10}.get(name, default)
        bot._get_player_movement_coords = lambda: (100.0, 100.0)
        bot._refresh_node_distance_from_odometry(now)
        # hypot(110-100, 100-100)=10 px × 0.28 m/px = 2.8 → 3 m.
        self.assertEqual(bot.active_node_distance_m, 3)
        self.assertTrue(bot._active_node_distance_fresh(now))

    def test_odometry_skips_when_position_unknown(self):
        bot = _make_bot()
        now = time.time()
        bot.active_node = {"id": 14, "map_x": 110.0, "map_y": 100.0}
        bot._active_node_distance_updated_ts = 0.0
        bot._get_cfg = lambda name, default=None: {"reclick_node_sec": 10}.get(name, default)
        bot._get_player_movement_coords = lambda: (0.0, 0.0)
        bot._refresh_node_distance_from_odometry(now)
        self.assertEqual(bot.active_node_distance_m, 999)


if __name__ == "__main__":
    unittest.main()
