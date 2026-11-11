# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for İş Paketi D:
- Intelligent Game Frame Gate (dataset_factory/game_frame_gate.py)
- Factory Controller Ingestion Hardening (dataset_factory/factory_controller.py)
- Desktop & Non-Game Screenshot Purging (dataset_factory/clean_dataset.py)
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from dataset_factory.clean_dataset import purge_non_game_screenshots
# AutoLabelingDatasetFactory bilerek üstte içe aktarılmaz: factory_controller
# ağır veri kümesi bağımlılıkları çeker ve bu dosyanın yalnızca belirli bir
# venv'de toplanmasına yol açar (F-1406). Onu kullanan tek test, tembel
# içe aktarır ve içe aktarma başarısız olursa kendini atlar.
from dataset_factory.game_frame_gate import (
    GameFrameGate,
    check_process_and_focus,
    check_visual_hud,
    check_visual_novelty,
    clear_frame_gate_cache,
    get_default_frame_gate,
    is_valid_game_frame,
)


class _StubFocusGuard:
    """Win32 ön plan sorgusunun yerine geçen sahte nesne (F-1323).

    Yalnızca OS sınırı sahtedir: `query_foreground_window_info` hazır bir
    (hwnd, pid, başlık) döndürür; Kapı 1'in başlık/süreç/odak kararı ise
    gerçek `check_gate1_process_and_focus` koduyla çalışır.
    """

    def __init__(self, title: str, pid: int) -> None:
        self._title = title
        self._pid = pid

    def query_foreground_window_info(self):
        return (1234, self._pid, self._title)

    def is_game_window_focused(self):
        return (True, f"{self._title} (PID={self._pid})")


class TestGameFrameGateSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real_game_sample_path = Path("dataset/images/train/boot_478467_000_aria_3d_crest_far.jpg")
        if cls.real_game_sample_path.is_file():
            raw = np.fromfile(str(cls.real_game_sample_path), dtype=np.uint8)
            cls.real_game_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        else:
            cls.real_game_img = None

    def setUp(self) -> None:
        self.gate = GameFrameGate(max_hash_cache=64, min_dhash_distance=6)
        clear_frame_gate_cache()

    def _real_path_gate(self, is_running: bool, title: str = "AION2 Client 64-bit", pid: int = 8888) -> GameFrameGate:
        """Kapı 1'in gerçek dalını kullanan kapı (F-1323).

        `set_simulated_game_state` simülasyon bayrağını atlar; burada oyun
        durumu kurucuya enjekte edilen `is_game_running_fn` ile, ön plan
        penceresi ise `_StubFocusGuard` ile verilir.
        """
        return GameFrameGate(
            max_hash_cache=64,
            min_dhash_distance=6,
            focus_guard=_StubFocusGuard(title, pid),
            is_game_running_fn=lambda: bool(is_running),
        )

    @staticmethod
    def _hud_frame() -> np.ndarray:
        """Gerçek oyun karesi yokken bile Kapı 2'den geçen sentetik HUD karesi (F-1324)."""
        h, w = 1080, 1920
        grad = np.tile(np.linspace(20, 80, w, dtype=np.uint8), (h, 1))
        frame = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (750, 920), (950, 936), (20, 20, 220), -1)
        return frame

    @staticmethod
    def _desktop_canvas() -> np.ndarray:
        """Sentetik masaüstü karesi: duvar kâğıdı + altta koyu görev çubuğu.

        Kapı 2'nin masaüstü reddi ile purge testinin masaüstü girdisi aynı
        kareyi kullanır; tek kaynaktan beslenir.
        """
        desktop = np.full((1080, 1920, 3), (180, 120, 40), dtype=np.uint8)
        desktop[1035:, :] = (25, 25, 25)  # dark taskbar strip
        return desktop

    # ==========================================================================
    # 1. Kapı 1: Process & Focus Guard Tests
    # ==========================================================================
    def test_gate1_rejects_when_game_not_running(self) -> None:
        gate = self._real_path_gate(is_running=False)
        ok, reason = gate.check_gate1_process_and_focus()
        self.assertFalse(ok)
        self.assertIn("game_not_running", reason)

    def test_gate1_rejects_windows_desktop(self) -> None:
        for title, pid in (("Program Manager", 100), ("Desktop", 101)):
            with self.subTest(title=title):
                gate = self._real_path_gate(is_running=True, title=title, pid=pid)
                ok, reason = gate.check_gate1_process_and_focus()
                self.assertFalse(ok)
                self.assertIn("desktop_focused", reason)

    def test_gate1_rejects_gatherion_ui(self) -> None:
        for title, pid in (("Gatherion Autonomous Bot v2.4", 9999), ("Overlay HUD", os.getpid())):
            with self.subTest(title=title):
                gate = self._real_path_gate(is_running=True, title=title, pid=pid)
                ok, reason = gate.check_gate1_process_and_focus()
                self.assertFalse(ok)
                self.assertIn("gatherion_ui_focused", reason)

    def test_gate1_rejects_blocked_apps(self) -> None:
        for blocked_title in ["Google Chrome - New Tab", "Visual Studio Code", "File Explorer", "Task Manager"]:
            with self.subTest(title=blocked_title):
                gate = self._real_path_gate(is_running=True, title=blocked_title, pid=5000)
                ok, reason = gate.check_gate1_process_and_focus()
                self.assertFalse(ok, f"Should reject {blocked_title}")
                # F-1409: 'veya'lı assertion sebebi gizliyordu; dört başlık da
                # engellenen ön plan penceresi olarak reddedilir.
                self.assertIn("blocked_foreground", reason, f"Unexpected reason for {blocked_title}: {reason}")

    def test_gate1_accepts_verified_game_window(self) -> None:
        gate = self._real_path_gate(is_running=True, title="AION2 Client 64-bit", pid=8888)
        ok, reason = gate.check_gate1_process_and_focus()
        self.assertTrue(ok)
        self.assertIn("process_and_focus_ok", reason)

    def test_master_gate_bypasses_gate1_when_require_foreground_false(self) -> None:
        # F-1324: gerçek oyun karesi yokken test boş geçiyordu. Sentetik HUD
        # karesi her ortamda çalışır; oyun çalışmıyorken bile
        # require_foreground=False atlaması doğrulanır.
        gate = GameFrameGate(is_game_running_fn=lambda: False)
        frame = self._hud_frame()
        ok, reason = gate.is_valid_game_frame(frame, require_foreground=False)
        self.assertTrue(ok)
        self.assertIn("accepted_hud_evidence", reason)

        # Aynı kare, Kapı 1 devredeyken oyun çalışmadığı için reddedilir.
        ok2, reason2 = gate.is_valid_game_frame(frame, require_foreground=True)
        self.assertFalse(ok2)
        self.assertIn("game_not_running", reason2)

        if self.real_game_img is not None:
            ok3, reason3 = gate.is_valid_game_frame(self.real_game_img, require_foreground=False)
            self.assertTrue(ok3)
            self.assertIn("accepted_hud_evidence", reason3)

    # ==========================================================================
    # 2. Kapı 2: Visual HUD Evidence Tests
    # ==========================================================================
    def test_gate2_rejects_black_frame(self) -> None:
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(black)
        self.assertFalse(ok)
        self.assertIn("black_frame", reason)

    def test_gate2_rejects_solid_color_frame(self) -> None:
        solid = np.full((720, 1280, 3), 120, dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(solid)
        self.assertFalse(ok)
        self.assertIn("solid_color", reason)

    def test_gate2_rejects_pure_white_canvas(self) -> None:
        white = np.full((720, 1280, 3), 255, dtype=np.uint8)
        ok, reason = self.gate.check_gate2_visual_hud(white)
        self.assertFalse(ok)
        self.assertTrue("solid_color" in reason or "blank_white" in reason)

    def test_gate2_rejects_empty_or_invalid_arrays(self) -> None:
        ok1, _ = self.gate.check_gate2_visual_hud(None)  # type: ignore
        self.assertFalse(ok1)
        ok2, _ = self.gate.check_gate2_visual_hud(np.array([], dtype=np.uint8))
        self.assertFalse(ok2)
        ok3, _ = self.gate.check_gate2_visual_hud(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertFalse(ok3)

    def test_gate2_rejects_synthetic_desktop_canvas(self) -> None:
        # Wallpaper + bottom taskbar (tek kaynak: _desktop_canvas)
        desktop = self._desktop_canvas()
        ok, reason = self.gate.check_gate2_visual_hud(desktop)
        self.assertFalse(ok)
        self.assertTrue("no_hud_evidence" in reason or "desktop" in reason)

    def test_gate2_accepts_player_hp_mp_bar(self) -> None:
        # _hud_frame zaten alt-ortada gerçekçi HP barı çizer (y=920..936,
        # x=750..950); aynı kareyi tekrar kurmak yerine onu kullan.
        frame = self._hud_frame()
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("player_red_hp_bar", reason)

    def test_gate2_accepts_minimap_compass_arrow(self) -> None:
        # Taban kare _hud_frame'den; yalnızca minimap oku eklenir.
        frame = self._hud_frame()
        # Radar center is at (1650, 195)
        pts = np.array([[1650, 185], [1640, 205], [1660, 205]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (250, 250, 250))
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("accepted_hud_evidence", reason)

    def test_gate2_accepts_skill_bar_slots(self) -> None:
        # Taban kare _hud_frame'den (gradyan zemin): düz koyu zemin
        # (np.full 30) alt şeridi tekdüze karanlık gösterip kasıtlı
        # Windows-görev-çubuğu vetosunu (F-0956) tetikliyordu — Kapı 2
        # skill-bar kanıtına hiç bakmadan reddediyordu. Gerçek oyun
        # karesinde alt şerit asla tekdüze değildir; kardeş kabul
        # testleri de aynı gradyan tabanı kullanır.
        frame = self._hud_frame()
        # Draw 4 aligned skill slot boxes at y=980
        for i in range(4):
            x = 600 + i * 60
            cv2.rectangle(frame, (x, 980), (x + 40, 1020), (200, 200, 200), 2)
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("skill_bar_slots", reason)

    def test_gate2_accepts_real_dataset_screenshot(self) -> None:
        # S-3502: dataset/ audit kopyasında yokken bu test sıfır assertion
        # ile geçiyordu. Sentetik HUD karesi (_hud_frame) her ortamda
        # Kapı 2'den geçer; gerçek kare varsa o da ayrıca doğrulanır.
        frame = self._hud_frame()
        ok, reason = self.gate.check_gate2_visual_hud(frame)
        self.assertTrue(ok)
        self.assertIn("accepted_hud_evidence", reason)

        if self.real_game_img is None:
            # Gerçek örnek yokken sessizce geçmek sahte yeşildir: sentetik
            # kare yukarıda doğrulandı, ama gerçek-kare kısmı yapılmadı —
            # bunu açıkça atla olarak işaretle.
            self.skipTest("gerçek oyun örneği yok: dataset/images/train/boot_478467_000_aria_3d_crest_far.jpg")
        ok, reason = self.gate.check_gate2_visual_hud(self.real_game_img)
        self.assertTrue(ok)
        self.assertIn("accepted_hud_evidence", reason)

    # ==========================================================================
    # 3. Kapı 3: Visual Novelty & dHash Tests
    # ==========================================================================
    def test_gate3_dhash_calculation_and_hamming_distance(self) -> None:
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Yatay rampa, vektörize: her sütun x*2 (x=0..99), 3 kanala yayılır.
        img2 = np.tile((np.arange(100, dtype=np.uint8) * 2).reshape(1, 100, 1), (100, 1, 3))

        h1 = GameFrameGate.compute_dhash(img1)
        h2 = GameFrameGate.compute_dhash(img2)
        self.assertIsInstance(h1, int)
        self.assertIsInstance(h2, int)
        dist = GameFrameGate.hamming_distance(h1, h2)
        # F-1407: assertGreaterEqual(dist, 0) her zaman doğruydu. Düz siyah
        # karenin hash'i 0'dır; yatay rampa 9x8 gradyanın 64 bitini de çevirir.
        self.assertEqual(h1, 0)
        self.assertGreater(dist, 0)
        self.assertEqual(dist, 64)
        self.assertEqual(GameFrameGate.hamming_distance(h1, h1), 0)

    def test_gate3_rejects_identical_and_duplicate_frames(self) -> None:
        # S-3501: gerçek kare yokken test assertion'sız dönüyordu.
        # Kapı 3 yalnızca dHash'e bakar; sentetik kareyle duplicate
        # reddi (distance 0 < 6) her ortamda doğrulanır. Gerçek kare
        # varsa aynı senaryo onunla da tekrarlanır.
        self.gate.clear_hash_cache()
        frame = self._hud_frame()
        ok1, _ = self.gate.check_gate3_visual_novelty(frame, update_cache=True)
        self.assertTrue(ok1)
        self.assertEqual(self.gate.hash_cache_size, 1)

        # Same frame -> distance 0 < 6 -> rejected
        ok2, r2 = self.gate.check_gate3_visual_novelty(frame, update_cache=True)
        self.assertFalse(ok2)
        self.assertIn("duplicate_frame", r2)
        self.assertIn("0 < 6", r2)

        if self.real_game_img is None:
            # Gerçek örnek yokken sessizce geçmek sahte yeşildir.
            self.skipTest("gerçek oyun örneği yok: dataset/images/train/boot_478467_000_aria_3d_crest_far.jpg")
        self.gate.clear_hash_cache()
        ok1, _ = self.gate.check_gate3_visual_novelty(self.real_game_img, update_cache=True)
        self.assertTrue(ok1)
        self.assertEqual(self.gate.hash_cache_size, 1)

        # Same frame -> distance 0 < 6 -> rejected
        ok2, r2 = self.gate.check_gate3_visual_novelty(self.real_game_img, update_cache=True)
        self.assertFalse(ok2)
        self.assertIn("duplicate_frame", r2)
        self.assertIn("0 < 6", r2)

    def test_gate3_accepts_distinct_frames(self) -> None:
        self.gate.clear_hash_cache()
        img_a = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_a, (50, 50), 30, (255, 255, 255), -1)

        img_b = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_b, (150, 150), 40, (255, 255, 255), -1)

        ok_a, _ = self.gate.check_gate3_visual_novelty(img_a, update_cache=True)
        self.assertTrue(ok_a)
        ok_b, _ = self.gate.check_gate3_visual_novelty(img_b, update_cache=True)
        self.assertTrue(ok_b)
        self.assertEqual(self.gate.hash_cache_size, 2)

    # ==========================================================================
    # 4. Factory Controller Ingestion Hardening Tests
    # ==========================================================================
    def test_factory_controller_drops_black_live_capture(self) -> None:
        # F-1406: tembel içe aktarma — bu dosya factory_controller olmadan
        # da toplanabilsin; bağımlılık yoksa test kendini atlar.
        try:
            from dataset_factory.factory_controller import AutoLabelingDatasetFactory
        except ImportError as exc:
            self.skipTest(f"factory_controller içe aktarılamadı: {exc}")
        fc = AutoLabelingDatasetFactory(project_root=".")
        black = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Live ingest rejected
        res_live = fc.ingest_live_frame(black)
        self.assertFalse(res_live["ingested"])
        self.assertIn("game_frame_gate_rejected", res_live["reason"])

        # Gather success rejected
        res_gather = fc.on_gather_success(current_frame=black)
        self.assertFalse(res_gather["saved"])
        self.assertIn("game_frame_gate_rejected", res_gather["reason"])

    # ==========================================================================
    # 5. Clean Dataset Purging Tests
    # ==========================================================================
    def test_clean_dataset_purges_invalid_desktop_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            train_img = td / "images" / "train"
            train_lbl = td / "labels" / "train"
            val_img = td / "images" / "val"
            val_lbl = td / "labels" / "val"
            gl_dir = td / "game_learning" / "monsters"

            for d in (train_img, train_lbl, val_img, val_lbl, gl_dir):
                d.mkdir(parents=True)

            # Valid game frame
            if self.real_game_img is not None:
                cv2.imwrite(str(train_img / "valid_aria.jpg"), self.real_game_img)
                (train_lbl / "valid_aria.txt").write_text("0 0.5 0.5 0.1 0.1\n")

            # Invalid desktop screenshot (tek kaynak: _desktop_canvas)
            desktop = self._desktop_canvas()
            cv2.imwrite(str(val_img / "desktop_capture.jpg"), desktop)
            (val_lbl / "desktop_capture.txt").write_text("10 0.2 0.2 0.1 0.1\n")

            # Invalid black screenshot in game_learning
            black = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(gl_dir / "black_bad.jpg"), black)
            (gl_dir / "black_bad.json").write_text('{"state": "error"}\n')

            # 1. Dry run
            dry_res = purge_non_game_screenshots(dataset_dir=td, dry_run=True, verbose=False)
            self.assertEqual(dry_res["invalid_purged"], 2)
            self.assertTrue((val_img / "desktop_capture.jpg").exists())

            # 2. Real remove
            remove_res = purge_non_game_screenshots(dataset_dir=td, action="remove", dry_run=False, verbose=False)
            self.assertEqual(remove_res["invalid_purged"], 2)
            self.assertFalse((val_img / "desktop_capture.jpg").exists())
            self.assertFalse((val_lbl / "desktop_capture.txt").exists())
            self.assertFalse((gl_dir / "black_bad.jpg").exists())
            self.assertFalse((gl_dir / "black_bad.json").exists())

            if self.real_game_img is not None:
                self.assertTrue((train_img / "valid_aria.jpg").exists())
                self.assertTrue((train_lbl / "valid_aria.txt").exists())

    def test_clean_dataset_quarantine_action_and_preview_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            train_img = td / "images" / "train"
            train_lbl = td / "labels" / "train"
            previews_dir = td / "previews"
            q_dir = td / "quarantine_target"

            for d in (train_img, train_lbl, previews_dir):
                d.mkdir(parents=True)

            # Invalid black screenshot with label and preview
            black = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(train_img / "bad_black.jpg"), black)
            (train_lbl / "bad_black.txt").write_text("1 0.5 0.5 0.1 0.1\n")
            (previews_dir / "bad_black.jpg").write_text("fake_preview\n")

            res_move = purge_non_game_screenshots(
                dataset_dir=td,
                action="move",
                quarantine_dir=q_dir,
                dry_run=False,
                verbose=False,
            )
            self.assertEqual(res_move["invalid_purged"], 1)
            self.assertFalse((train_img / "bad_black.jpg").exists())
            self.assertFalse((train_lbl / "bad_black.txt").exists())
            self.assertFalse((previews_dir / "bad_black.jpg").exists())

            # Verified moved into quarantine_target
            self.assertTrue((q_dir / "images" / "train" / "bad_black.jpg").exists())
            self.assertTrue((q_dir / "labels" / "train" / "bad_black.txt").exists())
            self.assertTrue((q_dir / "previews" / "bad_black.jpg").exists())

    def test_gate3_lru_cache_bounded_size(self) -> None:
        small_gate = GameFrameGate(max_hash_cache=5, min_dhash_distance=1)
        for i in range(12):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[i % 50, :] = 255
            small_gate.check_gate3_visual_novelty(img, update_cache=True)
        self.assertLessEqual(small_gate.hash_cache_size, 5)

    def test_module_level_convenience_functions(self) -> None:
        clear_frame_gate_cache()
        self.assertIsNotNone(get_default_frame_gate())
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        ok_hud, r_hud = check_visual_hud(black)
        self.assertFalse(ok_hud)
        self.assertIn("black_frame", r_hud)

        # F-1408: check_process_and_focus artık yalnızca tip kontrolü yapmıyor.
        # Varsayılan kapının simüle durumu yoktur; oyun süreci ve ön plan
        # penceresi mock sınırıyla verilir, gerçek Kapı 1 kararı doğrulanır.
        gate = get_default_frame_gate()
        with mock.patch.object(gate, "is_game_running_fn", return_value=False):
            ok_proc, r_proc = check_process_and_focus()
            self.assertFalse(ok_proc)
            self.assertIn("game_not_running", r_proc)
        with mock.patch.object(gate, "is_game_running_fn", return_value=True):
            with mock.patch.object(
                gate.focus_guard,
                "query_foreground_window_info",
                return_value=(1234, 8888, "AION2 Client 64-bit"),
            ):
                with mock.patch.object(
                    gate.focus_guard,
                    "is_game_window_focused",
                    return_value=(True, "AION2 Client 64-bit (PID=8888)"),
                ):
                    ok_proc2, r_proc2 = check_process_and_focus()
                    self.assertTrue(ok_proc2)
                    self.assertIn("process_and_focus_ok", r_proc2)

        # check_visual_novelty
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        img[20:40, 20:40] = 200
        ok_nov1, _ = check_visual_novelty(img, update_cache=True)
        self.assertTrue(ok_nov1)
        ok_nov2, r_nov2 = check_visual_novelty(img, update_cache=True)
        self.assertFalse(ok_nov2)
        self.assertIn("duplicate_frame", r_nov2)


if __name__ == "__main__":
    unittest.main()
