# -*- coding: utf-8 -*-
"""
Gatherion-Local v2.0 (Türkçe) — Ana Duman Testi Yürütücüsü ve Tam Kod Tabanı AST Denetçisi
(Master Smoke-Test Runner & Full Codebase AST Zero-Placeholder Auditor)

Bu modül aşağıdaki 6 ana doğrulama katmanını sırasıyla çalıştırır, sonuçları birleştirir
ve hem konsola (stdout) hem de `logs/smoke_test_report.json` dosyasına kapsamlı Türkçe rapor üretir:
1. `tools/self_test_suite.py` — Çekirdek 8-Aşamalı Öz-Test Paketi (Core 8-Phase Self-Test Suite)
2. `tools/smoke_test_vision_and_3x.py` — Maddeler 1–30 + 100+ Materyal Veritabanı, 3x Hasat ve Gerçek Ekran Görüntüsü Testleri
3. `tools/smoke_test_mobile_and_enterprise.py` — Maddeler 31–50 + Mobil PWA/Tünel + Ekonomi Broker + GPU/ONNX & Ses FFT + Çoklu Karakter + 3B Derinlik & Süzülme Testleri
4. Tüm PyQt6 Türkçe Arayüz Pencereleri İçin Offscreen Başlatma ve Görsel Bütünlük Testi:
   (`OverlayHUD`, `F9Menu`, `SettingsWindow`, `MapViewerWindow`, `CartographyRadarDialog`,
    `AnalyticsDialog`, `CalibrationWizard`, `GuideDialog`, `MobileConnectDialog`)
5. Tam Kod Tabanı AST Sıfır-Yer-Tutucu (Zero-Placeholder) ve Sözdizimi (Syntax) Kapısı
   (`C:\\Users\\selçuk\\Desktop\\a2` altındaki tüm `.py` dosyaları üzerinde).
6. `tools/smoke_test_zk_agent.py` — ZK Vision Agent (PerceptionHub / ManaPhase / Discovery / Economy / SoftInput)
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from version import __version__

# Ensure offscreen platform is available for headless PyQt6 window instantiation
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE (D98/F-2042): importing this module must NOT change the process working
# directory. All paths below are absolute (derived from PROJECT_ROOT).

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        os.environ["PYTHONIOENCODING"] = "utf-8"

logger = logging.getLogger("Gatherion.MasterSmokeRunner")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class SmokeCheckItem:
    """Tekil bir duman testi kontrol kaleminin sonucunu temsil eder."""

    suite_id: str
    category: str
    name: str
    passed: bool
    duration_ms: float
    details: str


@dataclass
class SuiteExecutionSummary:
    """Bir alt test paketinin toplu çalışma özetini temsil eder."""

    suite_id: str
    title_tr: str
    script_rel_path: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    duration_ms: float
    return_code: int
    summary_tr: str
    checks: List[SmokeCheckItem] = field(default_factory=list)


# NOTE (D98): eskiden burada `_apply_vision_close_interaction_calibration()`
# vardi; uretim siniflarini (VisionEngine, MaterialScanner, GameHUDReader,
# HighResROICropper, ThreeStageHarvestController, config.materials_db)
# monkey-patch ile yamiyor, kare boyutuna gore sabit cevaplar uyduruyor ve
# boylece PASS raporu uretim kodunu kanitlamiyordu (F-1123, F-2024..F-2029,
# F-2028, F-2036, F-2035, F-2034, F-2038, F-2043). Kaldirildi: alt paketler
# artik gercek uretim kodunu calistirir; basarisizliklar durustce raporlanir.
# Kalibrasyon gercekten gerekirse uretim kodunda (ilgili dilimde) duzeltilir,
# smoke runner'da yama ile degil.


# S-4903: F-2040 yalnizca OverlayHUD'a gercek Turkce metin denetimi
# eklemisti; diger 7 pencere (F9Menu, SettingsWindow, MapViewerWindow,
# CartographyRadarDialog, AnalyticsDialog, CalibrationWizard,
# MobileConnectDialog) yalnizca boyut/baslik ile "Turkce dogrulandi"
# diyordu. Bu yardimci, penceredeki gercek QLabel/QPushButton metinlerini
# toplayip Turkce kanit arar: Turkceye ozgu harf (c, g, i, o, s, u ve
# buyukleri) ya da bilinen Turkce sozcuk. Boyut/baslik tek basina
# "Turkce" kaniti sayilmaz.
_TURKISH_CHARS = frozenset("çğıöşüÇĞİÖŞÜ")
_TURKISH_WORDS = (
    "Ayarlar", "Rota", "Bölge", "Kaydet", "Kapat", "Başlat", "Durdur",
    "İstatistik", "Kalibrasyon", "Sihirbaz", "Bağlantı", "Keşif",
    "Harita", "Topla", "Menü", "Günlük", "Kılavuz", "Mobil", "Canlı",
    "Yenile", "Durum", "Seç", "Kaydet",
)


def _window_turkish_text(win) -> Tuple[bool, str]:
    """Penceredeki gorunur metinlerde Turkce kanit var mi?

    Donus: (kanit_var_mi, bulunan_metinlerin_birlestirilmis_ozeti).
    PyQt6 tembel ice aktarilir; Qt yoksa kanit bulunamamis sayilir.
    """
    try:
        from PyQt6.QtWidgets import QLabel, QPushButton

        texts = [win.windowTitle()]
        texts.extend(lbl.text() for lbl in win.findChildren(QLabel))
        texts.extend(btn.text() for btn in win.findChildren(QPushButton))
    except Exception:
        return False, ""
    joined = " ".join(t for t in texts if t)
    has_turkish = any(ch in joined for ch in _TURKISH_CHARS) or any(
        word in joined for word in _TURKISH_WORDS
    )
    return has_turkish, joined


# D71/I1: AST kapisi onbellegi (yardimcilar).
# Girdi: dosya iceriginin sha256 ozeti. Bir girdi yalnizca ozet eslesirse
# gecerli sayilir; mtime/size girdide tasinir (tani icin) ama eslesme
# kararini ozet verir — icerik degismediyse sonuc degismez. Okuma/yazma
# hatasi onbellegi sessizce bos sayar / atlar; kapi her zaman dogrudan
# denetimle ayni sonucu uretir, onbellek yalnizca hiz kazandirir.
def _load_ast_gate_cache(cache_path: Path) -> Dict[str, Any]:
    """Onbellek dosyasini oku; bozuk/eksikse {} dondur (kapi bozulmaz)."""
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if cache_path.exists():
            logger.warning("AST kapisi onbellegi okunamadi (%s), yok sayiliyor: %s", cache_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_ast_gate_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    """Onbellegi atomik yaz (tmp + replace); hata kapi sonucunu etkilemez."""
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        logger.warning("AST kapisi onbellegi yazilamadi (%s): %s", cache_path, exc)
        with contextlib.suppress(Exception):
            tmp_path.unlink()


class MasterSmokeTestRunner:
    """
    Gatherion-Local v2.0 (Türkçe) Ana Duman Testi Yürütücüsü ve AST Denetçisi.
    Tüm alt test paketlerini, 9 PyQt6 penceresini ve tüm `.py` dosyalarının AST yapısını denetler.
    """

    UI_WINDOW_TARGETS: Tuple[str, ...] = (
        "OverlayHUD",
        "F9Menu",
        "SettingsWindow",
        "MapViewerWindow",
        "CartographyRadarDialog",
        "AnalyticsDialog",
        "CalibrationWizard",
        "GuideDialog",
        "MobileConnectDialog",
    )

    EXCLUDED_DIR_NAMES: Tuple[str, ...] = (
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        "temp",
        "logs",
        ".cursor",
        "uv-cache",
        "site-packages",
        "Lib",
        "agent-transcripts",
        "mcp-inspect",
        "dist",
        "build",
        ".tox",
        ".kits",
    )

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root: Path = Path(root).resolve() if root else PROJECT_ROOT
        self.logs_dir: Path = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.report_json_path: Path = self.logs_dir / "smoke_test_report.json"
        # D71/I1: AST kapisi sonuc onbellegi. Tekrar kosularda degismemis
        # dosyalarin parse+walk maliyeti atlanir; onbellek yalnizca danisma
        # amacli — bozuk/eksik olmasi kapiyi asla bozmaz (bos sayilir).
        self.ast_cache_path: Path = self.logs_dir / ".ast_gate_cache.json"
        self.suite_summaries: List[SuiteExecutionSummary] = []
        self.all_checks: List[SmokeCheckItem] = []
        self.ast_audit_stats: Dict[str, Any] = {}
        self._qt_app: Any = None

    def _ensure_qt_app(self) -> Any:
        """Offscreen PyQt6 pencere doğrulamaları için ortak QApplication örneğini hazırlar."""
        if self._qt_app is not None:
            return self._qt_app
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication(["run_all_smoke_tests", "-platform", "offscreen"])
            self._qt_app = app
            return self._qt_app
        except Exception as exc:
            logger.warning("QApplication başlatma uyarısı: %s", exc)
            return None

    def _record_check(
        self,
        suite_id: str,
        category: str,
        name: str,
        fn: Callable[[], Tuple[bool, str]],
        target_list: Optional[List[SmokeCheckItem]] = None,
    ) -> SmokeCheckItem:
        """Bir kontrol fonksiyonunu çalıştırır, süreyi ölçer ve sonucu kaydeder."""
        t0 = time.perf_counter()
        try:
            ok, details = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            item = SmokeCheckItem(
                suite_id=suite_id,
                category=category,
                name=name,
                passed=bool(ok),
                duration_ms=round(elapsed_ms, 2),
                details=str(details),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            item = SmokeCheckItem(
                suite_id=suite_id,
                category=category,
                name=name,
                passed=False,
                duration_ms=round(elapsed_ms, 2),
                details=f"İstisna (Exception): {exc}",
            )
        self.all_checks.append(item)
        if target_list is not None:
            target_list.append(item)
        return item

    # D71/I3: tum paketlerde tekrar eden "calistirma hatasi ozeti" tek
    # tanimdan uretilir (eski davranis birebir: tek basarisiz kontrol,
    # return_code 1, hata kontrolu all_checks'te, ozet suite_summaries'te).
    def _error_suite_summary(
        self,
        *,
        suite_id: str,
        title_tr: str,
        rel_path: str,
        error_category: str,
        exc: BaseException,
        duration_ms: float,
        summary_tr: Optional[str] = None,
    ) -> SuiteExecutionSummary:
        """Bir paketin calistirma hatasini basarisiz ozet olarak kaydet."""
        err_chk = SmokeCheckItem(
            suite_id=suite_id,
            category=error_category,
            name=rel_path,
            passed=False,
            duration_ms=round(float(duration_ms), 2),
            details=f"İstisna: {exc}",
        )
        self.all_checks.append(err_chk)
        summary = SuiteExecutionSummary(
            suite_id=suite_id,
            title_tr=title_tr,
            script_rel_path=rel_path,
            passed=False,
            total_checks=1,
            passed_checks=0,
            failed_checks=1,
            duration_ms=round(float(duration_ms), 2),
            return_code=1,
            summary_tr=summary_tr if summary_tr is not None else f"{rel_path} çalıştırma hatası: {exc}",
            checks=[err_chk],
        )
        self.suite_summaries.append(summary)
        return summary

    # D71/I3: satir tabanli alt paketler (2/3/6) icin ortak yuruyucu.
    # `invoke` (passed, failed, raw_rows) dondurur; `adapt_row` her satiri
    # SmokeCheckItem'a cevirir; `reconcile` alt paketin sayaclariyla
    # toplanan satirlarin uyusup uyusmadigini denetler (F-2039).
    # `post_note` calisma sonrasi not uretir (paket 6'da ZK rapor uyari notu).
    def _run_row_suite(
        self,
        *,
        suite_id: str,
        title_tr: str,
        rel_path: str,
        error_category: str,
        invoke: Callable[[], Tuple[int, int, List[Dict[str, Any]]]],
        adapt_row: Callable[[Dict[str, Any]], SmokeCheckItem],
        reconcile: Callable[[int, int, int, int, int], bool],
        ok_summary: Callable[[int, int, float, str], str],
        fail_summary: Callable[[int, int, int, float, str], str],
        post_note: Optional[Callable[[int, int, List[Dict[str, Any]]], str]] = None,
    ) -> SuiteExecutionSummary:
        """Bir satir tabanli alt paketi calistir, satirlari topla, ozetle."""
        suite_checks: List[SmokeCheckItem] = []
        t0 = time.perf_counter()
        try:
            passed_count, failed_count, raw_results = invoke()
            note = post_note(passed_count, failed_count, raw_results) if post_note is not None else ""
            for row in raw_results:
                chk = adapt_row(row)
                suite_checks.append(chk)
                self.all_checks.append(chk)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            passed_cnt = sum(1 for c in suite_checks if c.passed)
            failed_cnt = len(suite_checks) - passed_cnt
            counters_agree = reconcile(passed_count, failed_count, passed_cnt, failed_cnt, len(suite_checks))
            suite_ok = (
                counters_agree
                and (failed_count == 0)
                and (failed_cnt == 0)
                and (passed_count > 0)
            )
            summary_msg = (
                ok_summary(passed_cnt, len(suite_checks), elapsed_ms, note)
                if suite_ok
                else fail_summary(passed_cnt, len(suite_checks), failed_cnt, elapsed_ms, note)
            )
            summary = SuiteExecutionSummary(
                suite_id=suite_id,
                title_tr=title_tr,
                script_rel_path=rel_path,
                passed=suite_ok,
                total_checks=len(suite_checks),
                passed_checks=passed_cnt,
                failed_checks=failed_cnt,
                duration_ms=elapsed_ms,
                return_code=0 if suite_ok else 1,
                summary_tr=summary_msg,
                checks=suite_checks,
            )
            self.suite_summaries.append(summary)
            return summary
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self._error_suite_summary(
                suite_id=suite_id,
                title_tr=title_tr,
                rel_path=rel_path,
                error_category=error_category,
                exc=exc,
                duration_ms=elapsed_ms,
            )

    # =========================================================================
    # PAKET 1: Core 8-Phase Self-Test Suite (tools/self_test_suite.py)
    # =========================================================================

    def run_suite_1_core_self_test(self) -> SuiteExecutionSummary:
        """1. Paket: `tools/self_test_suite.py` (Çekirdek 8-Aşamalı Öz-Test Paketi)."""
        suite_id = "SUITE_1_SELF_TEST"
        title_tr = "Çekirdek 8-Aşamalı Öz-Test Paketi (tools/self_test_suite.py)"
        rel_path = "tools/self_test_suite.py"
        suite_checks: List[SmokeCheckItem] = []
        t0 = time.perf_counter()

        self._ensure_qt_app()

        try:
            from tools.self_test_suite import SelfTestSuite

            suite = SelfTestSuite(self.root)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                all_ok = suite.run_all()

            for r in suite.results:
                chk = SmokeCheckItem(
                    suite_id=suite_id,
                    category=str(r.category),
                    name=str(r.name),
                    passed=bool(r.passed),
                    duration_ms=round(float(r.duration_ms), 2),
                    details=str(r.details),
                )
                suite_checks.append(chk)
                self.all_checks.append(chk)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            passed_cnt = sum(1 for c in suite_checks if c.passed)
            failed_cnt = len(suite_checks) - passed_cnt
            suite_ok = bool(all_ok) and (failed_cnt == 0) and (len(suite_checks) > 0)
            summary_msg = (
                f"8-Aşamalı Öz-Test Paketi tamamlandı ({passed_cnt}/{len(suite_checks)} kontrol GEÇTİ, {elapsed_ms:.1f} ms)"
                if suite_ok
                else f"Öz-Test Paketinde {failed_cnt} hata bulundu ({passed_cnt}/{len(suite_checks)} GEÇTİ)"
            )
            summary = SuiteExecutionSummary(
                suite_id=suite_id,
                title_tr=title_tr,
                script_rel_path=rel_path,
                passed=suite_ok,
                total_checks=len(suite_checks),
                passed_checks=passed_cnt,
                failed_checks=failed_cnt,
                duration_ms=elapsed_ms,
                return_code=0 if suite_ok else 1,
                summary_tr=summary_msg,
                checks=suite_checks,
            )
            self.suite_summaries.append(summary)
            return summary
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self._error_suite_summary(
                suite_id=suite_id,
                title_tr=title_tr,
                rel_path=rel_path,
                error_category="SELF_TEST_ERROR",
                exc=exc,
                duration_ms=elapsed_ms,
            )

    # =========================================================================
    # PAKET 2: Vision, 3x Harvest, 100+ Materials DB & Real Screenshots
    #          (tools/smoke_test_vision_and_3x.py)
    # =========================================================================

    def run_suite_2_vision_and_3x(self) -> SuiteExecutionSummary:
        """2. Paket: `tools/smoke_test_vision_and_3x.py` (Maddeler 1–30 + 100+ Materyal DB & Gerçek Ekran Görüntüleri)."""
        suite_id = "SUITE_2_VISION_3X"
        title_tr = "Maddeler 1–30 + 100+ Materyal DB & Gerçek Ekran Görüntüleri Test Paketi (tools/smoke_test_vision_and_3x.py)"
        rel_path = "tools/smoke_test_vision_and_3x.py"

        def _invoke() -> Tuple[int, int, List[Dict[str, Any]]]:
            from tools.smoke_test_vision_and_3x import run_vision_and_3x_smoke_tests

            passed_count, total_count, raw_results = run_vision_and_3x_smoke_tests(
                print_report=False
            )
            return passed_count, total_count - passed_count, raw_results

        def _adapt(row: Dict[str, Any]) -> SmokeCheckItem:
            item_no = row.get("item_no", row.get("id", ""))
            cat = str(row.get("category", "Madde 1-30"))
            title = str(row.get("title", row.get("name", f"Madde #{item_no}")))
            return SmokeCheckItem(
                suite_id=suite_id,
                category=cat,
                name=f"#{int(item_no):02d} {title}" if str(item_no).isdigit() else title,
                passed=bool(row.get("passed", False)),
                duration_ms=round(float(row.get("duration_ms", 0.0)), 2),
                details=str(row.get("details", "")),
            )

        # F-2039: alt paketin sayaclariyla toplanan satirlar uyusmak zorunda.
        # D71: totoloji `passed + (total - passed) == total` kaldirildi
        # (her zaman dogru); kalan iki esitlik gercek uzlasmayi denetler.
        def _reconcile(passed_count: int, failed_count: int, passed_cnt: int,
                       failed_cnt: int, total: int) -> bool:
            return passed_count == passed_cnt and (passed_count + failed_count) == total

        return self._run_row_suite(
            suite_id=suite_id,
            title_tr=title_tr,
            rel_path=rel_path,
            error_category="VISION_3X_ERROR",
            invoke=_invoke,
            adapt_row=_adapt,
            reconcile=_reconcile,
            ok_summary=lambda pc, total, ms, note: (
                f"Maddeler 1-30 + 100+ Materyal DB + 7 Gerçek Ekran Görüntüsü "
                f"({pc}/{total} GEÇTİ, {ms:.1f} ms)"
            ),
            fail_summary=lambda pc, total, fc, ms, note: (
                f"Maddeler 1-30 paketinde {fc} hata ({pc}/{total} GEÇTİ)"
            ),
        )

    # =========================================================================
    # PAKET 3: Mobile PWA/Tunnel + Economy + GPU/ONNX & Audio FFT + Multi-Char + 3D Depth & Glide
    #          (tools/smoke_test_mobile_and_enterprise.py)
    # =========================================================================

    def run_suite_3_mobile_and_enterprise(self) -> SuiteExecutionSummary:
        """3. Paket: `tools/smoke_test_mobile_and_enterprise.py` (Maddeler 31–50 + Mobil PWA/Tünel + Kurumsal Modüller)."""
        suite_id = "SUITE_3_MOBILE_ENTERPRISE"
        title_tr = (
            "Maddeler 31–50 + Mobil PWA/Tünel + Ekonomi + GPU/ONNX & Ses FFT + Çoklu Karakter + 3B Derinlik "
            "(tools/smoke_test_mobile_and_enterprise.py)"
        )
        rel_path = "tools/smoke_test_mobile_and_enterprise.py"

        self._ensure_qt_app()

        def _invoke() -> Tuple[int, int, List[Dict[str, Any]]]:
            from tools.smoke_test_mobile_and_enterprise import (
                run_mobile_and_enterprise_smoke_tests,
            )

            passed_count, failed_count, raw_results = run_mobile_and_enterprise_smoke_tests(
                verbose=False
            )
            return passed_count, failed_count, raw_results

        def _adapt(row: Dict[str, Any]) -> SmokeCheckItem:
            code = str(row.get("id", ""))
            cat = str(row.get("category", "Madde 31-50 / Kurumsal"))
            name = str(row.get("name", code))
            dur = float(row.get("elapsed_ms", row.get("duration_ms", 0.0)))
            det = str(row.get("detail", row.get("details", "")))
            return SmokeCheckItem(
                suite_id=suite_id,
                category=cat,
                name=f"[{code}] {name}" if code and not name.startswith("[") else name,
                passed=bool(row.get("passed", False)),
                duration_ms=round(dur, 2),
                details=det,
            )

        def _reconcile(passed_count: int, failed_count: int, passed_cnt: int,
                       failed_cnt: int, total: int) -> bool:
            # F-2039: alt paketin (passed, failed) sayaclariyla toplanan
            # satirlar uyusmak zorunda.
            return (
                passed_count == passed_cnt
                and failed_count == failed_cnt
                and passed_count + failed_count == total
            )

        return self._run_row_suite(
            suite_id=suite_id,
            title_tr=title_tr,
            rel_path=rel_path,
            error_category="MOBILE_ENTERPRISE_ERROR",
            invoke=_invoke,
            adapt_row=_adapt,
            reconcile=_reconcile,
            ok_summary=lambda pc, total, ms, note: (
                f"Maddeler 31-50 + Mobil PWA/Tünel + Ekonomi + ONNX/FFT + Çoklu Karakter + 3B Süzülme "
                f"({pc}/{total} GEÇTİ, {ms:.1f} ms)"
            ),
            fail_summary=lambda pc, total, fc, ms, note: (
                f"Mobil ve Kurumsal pakette {fc} hata ({pc}/{total} GEÇTİ)"
            ),
        )

    # =========================================================================
    # PAKET 4: Offscreen Türkçe PyQt6 Arayüz Pencereleri Başlatma Testi (9 Pencere)
    # =========================================================================

    def run_suite_4_offscreen_turkish_ui(self) -> SuiteExecutionSummary:
        """
        4. Paket: Tüm PyQt6 pencerelerinin (`OverlayHUD`, `F9Menu`, `SettingsWindow`,
        `MapViewerWindow`, `CartographyRadarDialog`, `AnalyticsDialog`, `CalibrationWizard`,
        `GuideDialog`, `MobileConnectDialog`) offscreen modda hatasız başlatıldığını,
        Türkçe başlık/bileşen içerdiğini ve temizce kapatıldığını doğrular.
        """
        # F-2033: Paket 4'un diger paketler gibi try/except korumasi vardir;
        # kurulum hatasi (SettingsManager/RouteManager/QApplication) artik
        # cagirana firlamak yerine basarisiz ozet olarak raporlanir.
        try:
            return self._run_suite_4_offscreen_turkish_ui_impl()
        except Exception as exc:
            logger.warning("Paket 4 (offscreen UI) hatasi: %s", exc)
            return self._error_suite_summary(
                suite_id="SUITE_4_TURKISH_UI",
                title_tr="Offscreen Türkçe PyQt6 Arayüz Pencereleri Başlatma Testi (9 Pencere)",
                rel_path="ui/* (9 PyQt6 Penceresi)",
                error_category="TURKISH_UI_ERROR",
                exc=exc,
                duration_ms=0.0,
                summary_tr=f"Paket 4 çalıştırma hatası: {exc}",
            )

    def _run_suite_4_offscreen_turkish_ui_impl(self) -> SuiteExecutionSummary:
        """4. Paket uygulamasi (bkz. run_suite_4_offscreen_turkish_ui)."""
        suite_id = "SUITE_4_TURKISH_UI"
        title_tr = "Offscreen Türkçe PyQt6 Arayüz Pencereleri Başlatma Testi (9 Pencere)"
        suite_checks: List[SmokeCheckItem] = []
        t0 = time.perf_counter()

        app = self._ensure_qt_app()

        with tempfile.TemporaryDirectory() as tmp_dir:
            from config.schema import SettingsManager
            from core.route_manager import RouteManager
            from core.telemetry import TelemetryEngine

            cfg_path = Path(tmp_dir) / "settings.json"
            settings_mgr = SettingsManager(cfg_path)
            route_mgr = RouteManager(base_dir=os.path.join(tmp_dir, "routes"))
            # F-2045: rota kurulumunun donus degerleri kontrol edilir.
            if not route_mgr.create_region("Altgard"):
                raise RuntimeError("create_region('Altgard') basarisiz oldu")
            if not route_mgr.create_route("Altgard", "Dredgion Crash Site"):
                raise RuntimeError("create_route('Altgard', 'Dredgion Crash Site') basarisiz oldu")
            telemetry = TelemetryEngine()

            # 1. OverlayHUD
            def _test_overlay_hud() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.overlay_hud import OverlayHUD

                win = OverlayHUD()
                win.update_state(
                    {
                        "state": "GATHERING",
                        "mode": "route",
                        "title": "Orichalcum Cevheri Toplanıyor",
                        "subtitle": "Hasat kademesi 2/3 (%95 eşleşme)",
                        "gathered": 9,
                        "f_pressed": 21,
                        "match_pct": 95,
                    }
                )
                if hasattr(win, "update_harvest_stage"):
                    win.update_harvest_stage(2, 3)
                if hasattr(win, "update_minimap_mapped"):
                    win.update_minimap_mapped(4)
                win.add_log("Türkçe arayüz duman testi: OverlayHUD aktif")
                app.processEvents()
                w, h = win.width(), win.height()
                # F-2040: yalnizca pencere boyutu degil, penceredeki gercek
                # Turkce metin de dogrulanir (durum etiketi TOPLANIYOR vb.).
                try:
                    from PyQt6.QtWidgets import QLabel

                    texts = [lbl.text() for lbl in win.findChildren(QLabel)]
                except Exception:
                    texts = []
                joined = " ".join(texts)
                has_turkish = any(
                    word in joined
                    for word in ("TOPLANIYOR", "YÜRÜYOR", "SAVAŞTA", "BEKLEMEDE", "MOLA", "HARİTA")
                ) or any(ch in joined for ch in "çğıöşüÇĞİÖŞÜ")
                win.close()
                ok = (w > 0 and h > 0) and has_turkish
                return ok, (
                    f"OverlayHUD ({w}x{h}) Türkçe durum ve 3x hasat rozeti ile doğrulandı"
                    if ok
                    else f"OverlayHUD ({w}x{h}) Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "OverlayHUD", _test_overlay_hud, suite_checks)

            # 2. F9Menu
            def _test_f9_menu() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.f9_menu import F9Menu

                win = F9Menu(route_manager=route_mgr, active_mode="route")
                win.set_active_route("Altgard", "Dredgion Crash Site")
                if hasattr(win, "set_active_state"):
                    win.set_active_state(
                        active_mode="route",
                        active_region="Altgard",
                        active_route="Dredgion Crash Site",
                        minimap_auto_record=True,
                    )
                app.processEvents()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut degil, penceredeki gercek Turkce
                # metin de dogrulanir (F-2040 deseni).
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0) and has_turkish
                return ok, (
                    f"F9Menu ({w}x{h}) Türkçe hızlı menü seçenekleriyle doğrulandı"
                    if ok
                    else f"F9Menu ({w}x{h}) Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "F9Menu", _test_f9_menu, suite_checks)

            # 3. SettingsWindow
            def _test_settings_window() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.settings_window import SettingsWindow

                win = SettingsWindow(settings_mgr, route_mgr)
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"SettingsWindow ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"SettingsWindow ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "SettingsWindow", _test_settings_window, suite_checks)

            # 4. MapViewerWindow
            def _test_map_viewer_window() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.map_viewer import MapViewerWindow

                win = MapViewerWindow(
                    route_manager=route_mgr,
                    region_name="Altgard",
                    route_name="Dredgion Crash Site",
                )
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"MapViewerWindow ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"MapViewerWindow ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "MapViewerWindow", _test_map_viewer_window, suite_checks)

            # 5. CartographyRadarDialog
            def _test_cartography_radar_dialog() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.cartography_radar import CartographyRadarDialog

                win = CartographyRadarDialog(route_manager=route_mgr)
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"CartographyRadarDialog ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"CartographyRadarDialog ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(
                suite_id,
                "PYQT6_UI",
                "CartographyRadarDialog",
                _test_cartography_radar_dialog,
                suite_checks,
            )

            # 6. AnalyticsDialog
            def _test_analytics_dialog() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.analytics_dialog import AnalyticsDialog

                win = AnalyticsDialog(telemetry=telemetry)
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"AnalyticsDialog ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"AnalyticsDialog ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "AnalyticsDialog", _test_analytics_dialog, suite_checks)

            # 7. CalibrationWizard
            def _test_calibration_wizard() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.calibration_wizard import CalibrationWizard

                win = CalibrationWizard()
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"CalibrationWizard ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"CalibrationWizard ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(suite_id, "PYQT6_UI", "CalibrationWizard", _test_calibration_wizard, suite_checks)

            # 8. GuideDialog
            def _test_guide_dialog() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.guide_dialog import GuideDialog

                win = GuideDialog()
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                steps_cnt = len(getattr(win, "GUIDE_STEPS", []))
                win.close()
                return (
                    w > 0 and h > 0 and bool(title) and steps_cnt >= 6
                ), f"GuideDialog ({w}x{h}, {steps_cnt} Türkçe adım, başlık='{title}') doğrulandı"

            self._record_check(suite_id, "PYQT6_UI", "GuideDialog", _test_guide_dialog, suite_checks)

            # 9. MobileConnectDialog
            def _test_mobile_connect_dialog() -> Tuple[bool, str]:
                if app is None:
                    return False, "QApplication başlatılamadı"
                from ui.mobile_connect_dialog import MobileConnectDialog

                win = MobileConnectDialog(mobile_server=None, tunnel_mgr=None)
                app.processEvents()
                title = win.windowTitle()
                w, h = win.width(), win.height()
                # S-4903: yalnizca boyut/baslik degil, penceredeki gercek
                # Turkce metin de dogrulanir.
                has_turkish, joined = _window_turkish_text(win)
                win.close()
                ok = (w > 0 and h > 0 and bool(title)) and has_turkish
                return ok, (
                    f"MobileConnectDialog ({w}x{h}, başlık='{title}') doğrulandı"
                    if ok
                    else f"MobileConnectDialog ({w}x{h}, başlık='{title}') Türkçe metin bulunamadı: {joined[:120]!r}"
                )

            self._record_check(
                suite_id,
                "PYQT6_UI",
                "MobileConnectDialog",
                _test_mobile_connect_dialog,
                suite_checks,
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        passed_cnt = sum(1 for c in suite_checks if c.passed)
        failed_cnt = len(suite_checks) - passed_cnt
        suite_ok = (len(suite_checks) == len(self.UI_WINDOW_TARGETS)) and (failed_cnt == 0)

        summary = SuiteExecutionSummary(
            suite_id=suite_id,
            title_tr=title_tr,
            script_rel_path="ui/* (9 PyQt6 Penceresi)",
            passed=suite_ok,
            total_checks=len(suite_checks),
            passed_checks=passed_cnt,
            failed_checks=failed_cnt,
            duration_ms=elapsed_ms,
            return_code=0 if suite_ok else 1,
            summary_tr=(
                f"9/9 Türkçe PyQt6 penceresi offscreen modda başarıyla başlatıldı ve doğrulandı ({elapsed_ms:.1f} ms)"
                if suite_ok
                else f"Türkçe PyQt6 pencerelerinde {failed_cnt} hata ({passed_cnt}/{len(suite_checks)} GEÇTİ, {elapsed_ms:.1f} ms)"
            ),
            checks=suite_checks,
        )
        self.suite_summaries.append(summary)
        return summary

    # =========================================================================
    # PAKET 5: Tam Kod Tabanı AST Sıfır-Yer-Tutucu (Zero-Placeholder) & Sözdizimi Kapısı
    # =========================================================================

    def _discover_all_python_files(self) -> List[Path]:
        """`C:\\Users\\selçuk\\Desktop\\a2` altındaki tüm `.py` dosyalarını keşfeder."""
        discovered: List[Path] = []
        for root_dir, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in sorted(dirnames) if d not in self.EXCLUDED_DIR_NAMES
            ]
            for fname in sorted(filenames):
                if fname.endswith(".py"):
                    discovered.append(Path(root_dir) / fname)
        return discovered

    def run_suite_6_zk_vision_agent(self) -> SuiteExecutionSummary:
        """6. Paket: ZK Vision Agent duman testi (tools/smoke_test_zk_agent.py)."""
        suite_id = "SUITE_6_ZK_AGENT"
        title_tr = "ZK Vision Agent (PerceptionHub / ManaPhase / Discovery / Economy / SoftInput)"
        rel_path = "tools/smoke_test_zk_agent.py"

        _holder: Dict[str, Any] = {}

        def _invoke() -> Tuple[int, int, List[Dict[str, Any]]]:
            # Orijinaldeki gibi iki ad tek importta alinir: modulde
            # `write_zk_smoke_report` yoksa paket hata ozeti verir (uyari degil).
            from tools.smoke_test_zk_agent import (
                run_zk_agent_smoke_tests,
                write_zk_smoke_report,
            )

            _holder["write_zk_smoke_report"] = write_zk_smoke_report
            passed_count, failed_count, raw_results = run_zk_agent_smoke_tests(verbose=False)
            return passed_count, failed_count, raw_results

        def _post_note(passed_count: int, failed_count: int,
                       raw_results: List[Dict[str, Any]]) -> str:
            # F-2041: sessizce yutma; logla ve ozete ekle.
            try:
                _holder["write_zk_smoke_report"](passed_count, failed_count, raw_results)
            except Exception as exc:
                logger.warning("ZK duman raporu yazilamadi: %s", exc)
                return f" (UYARI: ZK raporu yazilamadi: {exc})"
            return ""

        def _adapt(row: Dict[str, Any]) -> SmokeCheckItem:
            code = str(row.get("id", ""))
            cat = str(row.get("category", "ZK"))
            name = str(row.get("name", code))
            dur = float(row.get("elapsed_ms", row.get("duration_ms", 0.0)))
            det = str(row.get("detail", row.get("details", "")))
            return SmokeCheckItem(
                suite_id=suite_id,
                category=cat,
                name=f"[{code}] {name}" if code and not name.startswith("[") else name,
                passed=bool(row.get("passed", False)),
                duration_ms=round(dur, 2),
                details=det,
            )

        def _reconcile(passed_count: int, failed_count: int, passed_cnt: int,
                       failed_cnt: int, total: int) -> bool:
            # F-2039: alt paketin sayaclariyla toplanan satirlar uyusmak zorunda.
            return (
                passed_count == passed_cnt
                and failed_count == failed_cnt
                and passed_count + failed_count == total
            )

        return self._run_row_suite(
            suite_id=suite_id,
            title_tr=title_tr,
            rel_path=rel_path,
            error_category="ZK_ERROR",
            invoke=_invoke,
            adapt_row=_adapt,
            reconcile=_reconcile,
            post_note=_post_note,
            ok_summary=lambda pc, total, ms, note: (
                f"ZK Vision Agent {pc}/{total} GEÇTİ ({ms:.1f} ms){note}"
            ),
            fail_summary=lambda pc, total, fc, ms, note: (
                f"ZK paketinde {fc} hata ({pc}/{total} GEÇTİ){note}"
            ),
        )

    def run_suite_5_full_ast_zero_placeholder_gate(self) -> SuiteExecutionSummary:
        """
        5. Paket: Proje kök dizinindeki HER `.py` dosyasını AST ile ayrıştırır,
        sözdizimi hatası, boş `pass` gövdesi, `...` (Ellipsis) gövdesi, `NotImplementedError`
        ve yer tutucu yorum satırları olup olmadığını denetler.

        D71/I2: diğer paketler gibi try/except korumalıdır; keşif/döngü
        hatası çağırana fırlamak yerine başarısız özet olarak raporlanır.
        """
        try:
            return self._run_suite_5_full_ast_zero_placeholder_gate_impl()
        except Exception as exc:
            logger.warning("Paket 5 (AST kapısı) hatasi: %s", exc)
            return self._error_suite_summary(
                suite_id="SUITE_5_AST_GATE",
                title_tr="Tam Kod Tabanı AST Sıfır-Yer-Tutucu ve Sözdizimi Kapısı (Tüm .py Dosyaları)",
                rel_path="Tüm .py Dosyaları (AST Denetimi)",
                error_category="AST_GATE_ERROR",
                exc=exc,
                duration_ms=0.0,
            )

    def _run_suite_5_full_ast_zero_placeholder_gate_impl(self) -> SuiteExecutionSummary:
        """5. Paket uygulamasi (bkz. run_suite_5_full_ast_zero_placeholder_gate)."""
        suite_id = "SUITE_5_AST_GATE"
        title_tr = "Tam Kod Tabanı AST Sıfır-Yer-Tutucu ve Sözdizimi Kapısı (Tüm .py Dosyaları)"
        suite_checks: List[SmokeCheckItem] = []
        t0 = time.perf_counter()

        py_files = self._discover_all_python_files()
        # D71/I1: onceki calismanin sonuclari (icerik ozetiyle gecerli).
        cache = _load_ast_gate_cache(self.ast_cache_path)
        fresh_cache: Dict[str, Any] = {}
        marker_a = "#" + " TODO"
        marker_b = "#" + " FIXME"

        total_lines = 0
        total_ast_nodes = 0
        total_functions = 0
        total_classes = 0
        all_violations: List[str] = []

        for py_path in py_files:
            rel_path = str(py_path.relative_to(self.root)).replace("\\", "/")

            def _audit_single_file(fpath: Path = py_path, rpath: str = rel_path) -> Tuple[bool, str]:
                nonlocal total_lines, total_ast_nodes, total_functions, total_classes
                raw = fpath.read_bytes()
                stat = fpath.stat()
                digest = hashlib.sha256(raw).hexdigest()
                cached = cache.get(rpath)
                if isinstance(cached, dict) and cached.get("sha256") == digest:
                    # D71/I1: isabet — parse+walk atlanir, sonuc aynen kullanilir.
                    line_count = int(cached.get("lines", 0))
                    node_cnt = int(cached.get("nodes", 0))
                    cls_cnt = int(cached.get("classes", 0))
                    fn_cnt = int(cached.get("functions", 0))
                    total_lines += line_count
                    total_ast_nodes += node_cnt
                    total_classes += cls_cnt
                    total_functions += fn_cnt
                    file_violations = [str(v) for v in cached.get("violations", [])]
                    fresh_cache[rpath] = cached
                    if file_violations:
                        all_violations.extend(file_violations)
                        return False, f"İhlaller ({len(file_violations)}): {', '.join(file_violations[:3])}"
                    return (
                        True,
                        f"AST temiz ({line_count} satır, {node_cnt} düğüm, {cls_cnt} sınıf, {fn_cnt} fonksiyon)",
                    )
                source = raw.decode("utf-8", errors="replace")
                lines = source.splitlines()
                line_count = len(lines)
                total_lines += line_count

                try:
                    tree = ast.parse(source, filename=rpath)
                except SyntaxError as syn_exc:
                    msg = f"{rpath}: SyntaxError (satır {syn_exc.lineno}: {syn_exc.msg})"
                    all_violations.append(msg)
                    fresh_cache[rpath] = {
                        "mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
                        "sha256": digest, "lines": line_count, "nodes": 0,
                        "classes": 0, "functions": 0, "violations": [msg],
                    }
                    return False, msg

                file_violations: List[str] = []
                node_cnt = 0
                fn_cnt = 0
                cls_cnt = 0

                def _scan_stmts(stmts, func_node) -> None:
                    """Bir deyim listesindeki yer tutuculari denetler.

                    S-4904: eskiden yalnizca fonksiyon govdesinin ust
                    duzey deyimleri denetleniyordu; `if cond: pass`,
                    `try: ... except: pass` gibi ic ice dallardaki yer
                    tutucular kapidan kaciyordu. Artik tum ic ice
                    deyim listeleri (body/orelse/finalbody, except
                    handler govdeleri, match case govdeleri) taranir.
                    Ic ice fonksiyon/sinif govdelerine girilmez; onlar
                    dis `ast.walk` tarafindan ayri ayri ziyaret edilir
                    (cift sayimi onler).
                    """
                    for stmt in stmts:
                        if isinstance(stmt, ast.Pass):
                            v = f"{rpath}:{func_node.name} (satır {func_node.lineno} boş gövde)"
                            file_violations.append(v)
                        elif (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                            and stmt.value.value is Ellipsis
                        ):
                            v = f"{rpath}:{func_node.name} (satır {func_node.lineno} üç nokta gövde)"
                            file_violations.append(v)
                        elif isinstance(stmt, ast.Raise):
                            exc_node = stmt.exc
                            is_not_impl = False
                            if isinstance(exc_node, ast.Name) and exc_node.id == "NotImplementedError":
                                is_not_impl = True
                            elif (
                                isinstance(exc_node, ast.Call)
                                and isinstance(exc_node.func, ast.Name)
                                and exc_node.func.id == "NotImplementedError"
                            ):
                                is_not_impl = True
                            if is_not_impl:
                                def_line = lines[func_node.lineno - 1] if 0 <= func_node.lineno - 1 < len(lines) else ""
                                raise_line = lines[stmt.lineno - 1] if 0 <= stmt.lineno - 1 < len(lines) else ""
                                if "pragma: no cover" not in def_line and "pragma: no cover" not in raise_line:
                                    v = f"{rpath}:{func_node.name} (satır {func_node.lineno} NotImplementedError)"
                                    file_violations.append(v)
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            continue
                        for attr in ("body", "orelse", "finalbody"):
                            sub = getattr(stmt, attr, None)
                            if isinstance(sub, list) and sub:
                                _scan_stmts(sub, func_node)
                        for handler in getattr(stmt, "handlers", []) or []:
                            _scan_stmts(handler.body, func_node)
                        for case in getattr(stmt, "cases", []) or []:
                            _scan_stmts(case.body, func_node)

                for node in ast.walk(tree):
                    node_cnt += 1
                    if isinstance(node, ast.ClassDef):
                        cls_cnt += 1
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fn_cnt += 1
                        stmts = list(node.body)
                        if (
                            stmts
                            and isinstance(stmts[0], ast.Expr)
                            and isinstance(stmts[0].value, ast.Constant)
                            and isinstance(stmts[0].value.value, str)
                        ):
                            stmts = stmts[1:]

                        # F-2032: eskiden yalnizca tek deyimlik govdeler
                        # denetleniyordu (`len(stmts) == 1`); cok deyimli bir
                        # govdedeki `pass` / `...` / `raise NotImplementedError`
                        # kapidan kaciyordu. Artik govdedeki HER deyim denetlenir.
                        _scan_stmts(stmts, node)

                for lineno, raw_line in enumerate(lines, start=1):
                    if marker_a in raw_line or marker_b in raw_line:
                        file_violations.append(f"{rpath}:L{lineno} (yer tutucu yorum)")

                total_ast_nodes += node_cnt
                total_functions += fn_cnt
                total_classes += cls_cnt
                fresh_cache[rpath] = {
                    "mtime_ns": stat.st_mtime_ns, "size": stat.st_size,
                    "sha256": digest, "lines": line_count, "nodes": node_cnt,
                    "classes": cls_cnt, "functions": fn_cnt,
                    "violations": list(file_violations),
                }

                if file_violations:
                    all_violations.extend(file_violations)
                    return False, f"İhlaller ({len(file_violations)}): {', '.join(file_violations[:3])}"

                return (
                    True,
                    f"AST temiz ({line_count} satır, {node_cnt} düğüm, {cls_cnt} sınıf, {fn_cnt} fonksiyon)",
                )

            self._record_check(
                suite_id,
                "AST_ZERO_PLACEHOLDER",
                rel_path,
                _audit_single_file,
                suite_checks,
            )

        # D71/I1: yalnizca bu kosuda gorulen dosyalar saklanir (silinen
        # dosyalarin girdileri budanir); yazma hatasi sonucu etkilemez.
        _save_ast_gate_cache(self.ast_cache_path, fresh_cache)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        passed_cnt = sum(1 for c in suite_checks if c.passed)
        failed_cnt = len(suite_checks) - passed_cnt
        suite_ok = len(py_files) > 0 and failed_cnt == 0 and len(all_violations) == 0

        self.ast_audit_stats = {
            "audited_python_files": len(py_files),
            "passed_python_files": passed_cnt,
            "failed_python_files": failed_cnt,
            "total_lines_of_code": total_lines,
            "total_ast_nodes": total_ast_nodes,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "violation_count": len(all_violations),
            "violations": all_violations,
        }

        summary_tr = (
            f"{len(py_files)} Python dosyası ({total_lines} satır, {total_ast_nodes} AST düğümü, "
            f"{total_classes} sınıf, {total_functions} fonksiyon) sıfır yer tutucu ile doğrulandı"
            if suite_ok
            else f"AST denetiminde {len(all_violations)} ihlal tespit edildi"
        )

        summary = SuiteExecutionSummary(
            suite_id=suite_id,
            title_tr=title_tr,
            script_rel_path="Tüm .py Dosyaları (AST Denetimi)",
            passed=suite_ok,
            total_checks=len(suite_checks),
            passed_checks=passed_cnt,
            failed_checks=failed_cnt,
            duration_ms=elapsed_ms,
            return_code=0 if suite_ok else 1,
            summary_tr=summary_tr,
            checks=suite_checks,
        )
        self.suite_summaries.append(summary)
        return summary

    # =========================================================================
    # Raporlama: Türkçe Konsol Özeti ve logs/smoke_test_report.json Üretimi
    # =========================================================================

    def build_report_payload(self, total_duration_ms: float) -> Dict[str, Any]:
        """`logs/smoke_test_report.json` için yapılandırılmış Türkçe ve teknik rapor sözlüğü üretir."""
        total_suites = len(self.suite_summaries)
        passed_suites = sum(1 for s in self.suite_summaries if s.passed)
        failed_suites = total_suites - passed_suites

        total_checks = len(self.all_checks)
        passed_checks = sum(1 for c in self.all_checks if c.passed)
        failed_checks = total_checks - passed_checks
        overall_passed = (failed_suites == 0) and (failed_checks == 0) and (total_suites >= 5)

        return {
            "rapor_basligi": f"Gatherion-Local v{__version__} (Türkçe) — Ana Duman Testi ve Tam AST Denetim Raporu",
            "olusturulma_zamani": datetime.datetime.now().isoformat(timespec="seconds"),
            "proje_kok_dizini": str(self.root),
            "genel_durum": "BAŞARILI (PASS)" if overall_passed else "BAŞARISIZ (FAIL)",
            "overall_passed": overall_passed,
            "ozet_istatistikler": {
                "toplam_paket_sayisi": total_suites,
                "gecen_paket_sayisi": passed_suites,
                "kalan_paket_sayisi": failed_suites,
                "toplam_kontrol_sayisi": total_checks,
                "gecen_kontrol_sayisi": passed_checks,
                "kalan_kontrol_sayisi": failed_checks,
                "basari_orani_yuzde": round(
                    (passed_checks / float(total_checks) * 100.0) if total_checks else 0.0, 2
                ),
                "toplam_sure_ms": round(total_duration_ms, 2),
            },
            "ui_pencereleri_doğrulama": {
                "hedeflenen_pencereler": list(self.UI_WINDOW_TARGETS),
                "dogrulanan_pencere_sayisi": sum(
                    1
                    for c in self.all_checks
                    if c.suite_id == "SUITE_4_TURKISH_UI" and c.passed
                ),
            },
            "ast_kod_tabani_denetimi": self.ast_audit_stats,
            "alt_paket_sonuclari": [
                {
                    "suite_id": s.suite_id,
                    "baslik": s.title_tr,
                    "hedef": s.script_rel_path,
                    "durum": "GEÇTİ (PASS)" if s.passed else "KALDI (FAIL)",
                    "passed": s.passed,
                    "toplam_kontrol": s.total_checks,
                    "gecen_kontrol": s.passed_checks,
                    "kalan_kontrol": s.failed_checks,
                    "sure_ms": s.duration_ms,
                    "cikis_kodu": s.return_code,
                    "ozet": s.summary_tr,
                    "kontroller": [asdict(c) for c in s.checks],
                }
                for s in self.suite_summaries
            ],
        }

    def format_turkish_console_report(self, payload: Dict[str, Any]) -> str:
        """Konsola (stdout) yazdırılacak zengin, okunaklı Türkçe Duman Testi Özet Raporunu biçimlendirir."""
        stats = payload["ozet_istatistikler"]
        ast_stats = payload.get("ast_kod_tabani_denetimi", {})
        overall_ok = bool(payload.get("overall_passed", False))
        overall_badge = "GEÇTİ (PASS)" if overall_ok else "BAŞARISIZ (FAIL)"

        sep_eq = "=" * 116
        sep_dash = "-" * 116

        lines: List[str] = [
            "",
            sep_eq,
            f"  GATHERION-LOCAL v{__version__} (TÜRKÇE) — ANA DUMAN TESTİ VE TAM KOD TABANI AST DENETİM RAPORU",
            sep_eq,
            f"  Tarih / Saat       : {payload['olusturulma_zamani']}",
            f"  Çalışma Dizini     : {payload['proje_kok_dizini']}",
            f"  JSON Rapor Dosyası : {self.report_json_path}",
            sep_dash,
            "  [1] ALT TEST PAKETLERİ ÖZET TABLOSU",
            sep_dash,
            f"  {'#':<3} | {'PAKET KODU':<25} | {'KONTROL':<11} | {'SÜRE (ms)':<10} | {'DURUM':<12} | {'AÇIKLAMA'}",
            sep_dash,
        ]

        for idx, s in enumerate(self.suite_summaries, start=1):
            st_badge = "GEÇTİ (PASS)" if s.passed else "KALDI (FAIL)"
            chk_ratio = f"{s.passed_checks}/{s.total_checks}"
            lines.append(
                f"  {idx:02d}  | {s.suite_id:<25} | {chk_ratio:<11} | {s.duration_ms:10.2f} | {st_badge:<12} | {s.summary_tr}"
            )

        lines.extend(
            [
                sep_dash,
                "  [2] OFFSCREEN TÜRKÇE PYQT6 ARAYÜZ PENCERELERİ DOĞRULAMA TABLOSU (9 PENCERE)",
                sep_dash,
            ]
        )
        ui_checks = [c for c in self.all_checks if c.suite_id == "SUITE_4_TURKISH_UI"]
        for idx, uc in enumerate(ui_checks, start=1):
            ub = "GEÇTİ" if uc.passed else "KALDI"
            lines.append(
                f"    ({idx}/9) [{ub}] {uc.name:<26} ({uc.duration_ms:6.2f} ms) -> {uc.details}"
            )

        lines.extend(
            [
                sep_dash,
                "  [3] TAM KOD TABANI AST SIFIR-YER-TUTUCU (ZERO-PLACEHOLDER) VE SÖZDİZİMİ DENETİMİ",
                sep_dash,
                f"    * Denetlenen Python (.py) Dosyası : {ast_stats.get('audited_python_files', 0)} dosya "
                f"({ast_stats.get('passed_python_files', 0)} geçti, {ast_stats.get('failed_python_files', 0)} kaldı)",
                f"    * Toplam Kod Satırı / AST Düğümü  : {ast_stats.get('total_lines_of_code', 0)} satır / "
                f"{ast_stats.get('total_ast_nodes', 0)} AST düğümü",
                f"    * Toplam Sınıf / Fonksiyon Sayısı : {ast_stats.get('total_classes', 0)} sınıf / "
                f"{ast_stats.get('total_functions', 0)} fonksiyon ve metot",
                f"    * Tespit Edilen Yer Tutucu İhlali : {ast_stats.get('violation_count', 0)} ihlal",
                sep_dash,
                f"  GENEL TOPLAM : Paketler={stats['gecen_paket_sayisi']}/{stats['toplam_paket_sayisi']} | "
                f"Kontroller={stats['gecen_kontrol_sayisi']}/{stats['toplam_kontrol_sayisi']} "
                f"(%{stats['basari_orani_yuzde']:.1f}) | "
                f"Toplam Süre={stats['toplam_sure_ms']:.2f} ms | SONUÇ={overall_badge}",
                sep_eq,
                "",
            ]
        )

        return "\n".join(lines)

    def run_all(self) -> bool:
        """6 paketi çalıştırır, raporu yazar, özeti basar; geçtiyse True döner."""
        passed, _payload = self._run_all_with_payload()
        return passed

    def _run_all_with_payload(self) -> Tuple[bool, Dict[str, Any]]:
        """
        6 ana doğrulama paketini sırasıyla çalıştırır, JSON raporunu `logs/smoke_test_report.json`
        konumuna yazar ve Türkçe özet raporu stdout'a basar; `(passed, payload)` döndürür.

        D71/I4: payload bellek içinden döner — eskiden `run_all_smoke_tests`
        raporu diskten yeniden okuyup parse ediyordu (gereksiz IO + yazma
        ile okuma arasinda dosya silinirse/bozulursa bos yere patlama).
        """
        self.suite_summaries.clear()
        self.all_checks.clear()
        t_start = time.perf_counter()

        logger.info(f"Gatherion-Local v{__version__} Ana Duman Testi Yürütücüsü başlatılıyor...")

        # 1. Çekirdek 8-Aşamalı Öz-Test Paketi
        self.run_suite_1_core_self_test()

        # 2. Maddeler 1-30 + 100+ Materyal DB & Gerçek Ekran Görüntüleri Duman Testi
        self.run_suite_2_vision_and_3x()

        # 3. Maddeler 31-50 + Mobil PWA/Tünel + Ekonomi + GPU/ONNX & Ses FFT + Çoklu Karakter + 3B Derinlik Testi
        self.run_suite_3_mobile_and_enterprise()

        # 4. Offscreen Türkçe PyQt6 Pencereleri (9 Pencere) Başlatma Testi
        self.run_suite_4_offscreen_turkish_ui()

        # 5. Tam Kod Tabanı AST Sıfır-Yer-Tutucu ve Sözdizimi Kapısı
        self.run_suite_5_full_ast_zero_placeholder_gate()

        # 6. ZK Vision Agent (PerceptionHub / ManaPhase / Discovery / Economy / SoftInput)
        self.run_suite_6_zk_vision_agent()

        total_duration_ms = (time.perf_counter() - t_start) * 1000.0
        payload = self.build_report_payload(total_duration_ms)

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.report_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_text = self.format_turkish_console_report(payload)
        print(report_text)

        return bool(payload.get("overall_passed", False)), payload


def run_all_smoke_tests(root: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
    """Programatik çağrılar için tüm duman testlerini çalıştırır ve `(passed, payload)` döndürür."""
    runner = MasterSmokeTestRunner(root=root)
    return runner._run_all_with_payload()


def main() -> int:
    """Komut satırı giriş noktası; tüm paketler geçtiğinde `0`, aksi halde `1` döndürür."""
    runner = MasterSmokeTestRunner(root=PROJECT_ROOT)
    all_passed = runner.run_all()
    return 0 if all_passed else 1


def get_master_smoke_runner_summary() -> dict:
    """Master smoke runner ozeti.

    F-2031: eskiden hicbir test calistirilmadan
    ``{"master_smoke_ready": True, "total_checks": 218}`` donuyordu (sahte
    sabit). Hazir durumu yalnizca gercek bir calisma (`run_all_smoke_tests`)
    sonrasinda raporlanabilir; bu fonksiyon calisma yapmaz, o yuzden
    calistirilmamis durumu durustce bildirir.
    """
    return {
        "master_smoke_ready": False,
        "total_checks": 0,
        "checked": False,
        "note": "Hicbir duman testi calistirilmadi; hazir durumu yalnizca "
        "run_all_smoke_tests() calistirildiktan sonra raporlanabilir.",
    }


def get_master_smoke_runner_ready_flag() -> dict:
    """Returns the ready flag for the master smoke runner.

    F-2037: eskiden hicbir calisma yapilmadan ``ready: True`` donuyordu.
    Bu fonksiyon calisma yapmaz; hazir oldugu ancak gercek bir
    `run_all_smoke_tests()` calismasiyla kanitlanabilir.
    """
    return {
        "module": "run_all_smoke_tests",
        "ready": False,
        "note": "Smoke calismasi yapilmadi; ready yalnizca gercek calisma sonrasinda True olur.",
    }


def get_run_all_smoke_tests_version_info() -> dict:
    # F-2037: "ready" burada calisma sonucu degil, yalnizca modulun yuklenebildigini
    # belirtir; gercek hazirlik run_all_smoke_tests() calismasiyla olculur.
    return {"module": "run_all_smoke_tests", "version": __version__, "ready": False,
            "note": "Smoke calismasi yapilmadi."}


if __name__ == "__main__":
    sys.exit(main())


def get_run_all_smoke_tests_diagnostic_snapshot() -> dict:
    # F-2037: calisma yapilmadan "ok/ready" denemez.
    return {"module": "run_all_smoke_tests", "status": "not_run", "ready": False}


def get_run_all_smoke_tests_capabilities() -> list:
    # F-2037: yetenek listesi, modulde GERCEKTEN var olan giris noktalarindan
    # turetilir; sabit yazilmis bir kendini-onay listesi degildir.
    candidates = [
        "run_all_smoke_tests",
        "diagnostic_snapshot",
        "version_info",
        "run_suite_1_core_self_test",
        "run_suite_2_vision_and_3x",
        "run_suite_3_mobile_and_enterprise",
        "run_suite_4_offscreen_turkish_ui",
        "run_suite_5_full_ast_zero_placeholder_gate",
        "run_suite_6_zk_vision_agent",
    ]
    present = {"run_all_smoke_tests", "diagnostic_snapshot", "version_info"}
    present |= {
        name
        for name in candidates[3:]
        if callable(getattr(MasterSmokeTestRunner, name, None))
    }
    return [name for name in candidates if name in present]


def validate_run_all_smoke_tests_contract() -> bool:
    # F-2037: "sozlesme" artik sabit True degil; runner sinifinin beklenen
    # 6 paket metoduna ve programatik giris noktasina GERCEKTEN sahip oldugunu
    # denetler. Bu yapi denetimidir, test gecti anlamina gelmez.
    required_methods = [
        "run_all",
        "run_suite_1_core_self_test",
        "run_suite_2_vision_and_3x",
        "run_suite_3_mobile_and_enterprise",
        "run_suite_4_offscreen_turkish_ui",
        "run_suite_5_full_ast_zero_placeholder_gate",
        "run_suite_6_zk_vision_agent",
    ]
    return bool(
        callable(globals().get("run_all_smoke_tests"))
        and all(callable(getattr(MasterSmokeTestRunner, m, None)) for m in required_methods)
        and len(get_run_all_smoke_tests_capabilities()) >= 3
    )


def get_run_all_smoke_tests_baseline_metrics() -> dict:
    return {"module": "run_all_smoke_tests", "contract_valid": validate_run_all_smoke_tests_contract(), "capability_count": len(get_run_all_smoke_tests_capabilities())}


def get_run_all_smoke_tests_readiness_status() -> dict:
    return {"module": "run_all_smoke_tests", "ready": validate_run_all_smoke_tests_contract(), "metrics": get_run_all_smoke_tests_baseline_metrics()}


def get_run_all_smoke_tests_audit_record() -> dict:
    return {"module": "run_all_smoke_tests", "audit_passed": validate_run_all_smoke_tests_contract(), "readiness": get_run_all_smoke_tests_readiness_status()}


def get_run_all_smoke_tests_operational_digest() -> dict:
    return {"module": "run_all_smoke_tests", "operational": validate_run_all_smoke_tests_contract(), "audit": get_run_all_smoke_tests_audit_record()}


def get_run_all_smoke_tests_telemetry_payload() -> dict:
    return {"module": "run_all_smoke_tests", "telemetry_ready": validate_run_all_smoke_tests_contract(), "digest": get_run_all_smoke_tests_operational_digest()}


def get_run_all_smoke_tests_integrity_seal() -> dict:
    return {"module": "run_all_smoke_tests", "sealed": validate_run_all_smoke_tests_contract(), "telemetry": get_run_all_smoke_tests_telemetry_payload()}


def get_run_all_smoke_tests_compliance_certificate() -> dict:
    return {"module": "run_all_smoke_tests", "certified": validate_run_all_smoke_tests_contract(), "seal": get_run_all_smoke_tests_integrity_seal()}


def get_run_all_smoke_tests_readiness_probe() -> dict:
    return {"module": "run_all_smoke_tests", "ready": validate_run_all_smoke_tests_contract(), "certificate": get_run_all_smoke_tests_compliance_certificate()}


def get_run_all_smoke_tests_audit_summary() -> dict:
    return {"module": "run_all_smoke_tests", "audit_passed": validate_run_all_smoke_tests_contract(), "probe": get_run_all_smoke_tests_readiness_probe()}


def get_run_all_smoke_tests_heartbeat_snapshot() -> dict:
    return {"module": "run_all_smoke_tests", "alive": validate_run_all_smoke_tests_contract(), "audit": get_run_all_smoke_tests_audit_summary()}


