# -*- coding: utf-8 -*-
"""D71: `tools/run_all_smoke_tests.py` iyilestirmeleri icin testler.

Kapsanan iyilestirmeler:
- I1: AST kapisi (paket 5) sonuc onbellegi — degismemis dosyalar tekrar
  parse edilmez (ilk calisma doldurur, sonraki calismalar isabet eder).
- I2: Paket 5 dis kesif/dongu hatasinda coken kosucu yerine basarisiz
  ozet dondurur (diger 5 paket gibi try/except korumali).
- I3: Satir tabanli paketler (2/3/6) ortak `_run_row_suite` yardimcisina
  tasindi; hata ozeti tek `_error_suite_summary` yardimcisindan uretilir;
  paket 2'deki totoloji sayac denetimi kaldirildi. Davranis birebir kilitli.
- I4: `run_all_smoke_tests()` yazdigi JSON raporu diskten yeniden okuyup
  parse etmez; `run_all` ile ayni bellek-ici payload'u dondurur.

NOT (2026-11-08 21:12 Selcuk talimati): 250 ajan calisirken kodlama
ajanlari test kosmaz; bu dosya test ajanlari tarafindan kuyrukla
kosulmak uzere yazildi, D71 ajani tarafindan kosulmadi.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.run_all_smoke_tests import MasterSmokeTestRunner, run_all_smoke_tests  # noqa: E402


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _make_tree(tmp_path: Path) -> Path:
    """3 dosyalik kucuk agac: 1 temiz, 1 `pass` govdeli, 1 TODO yorumlu."""
    (tmp_path / "clean.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def stub():\n    pass\n", encoding="utf-8")
    (tmp_path / "todo.py").write_text("x = 1  # TODO sonra duzelt\n", encoding="utf-8")
    return tmp_path


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _ParseCounter:
    """ast.parse cagrilarini sayan sarici (orijinal davranisi korur)."""

    def __init__(self) -> None:
        self.count = 0
        self._orig = ast.parse

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._orig(*args, **kwargs)


# ---------------------------------------------------------------------------
# I1: AST kapisi onbellegi
# ---------------------------------------------------------------------------

def test_ast_gate_cache_second_run_skips_parse(tmp_path, monkeypatch):
    root = _make_tree(tmp_path)
    runner = MasterSmokeTestRunner(root=root)
    counter = _ParseCounter()
    monkeypatch.setattr(ast, "parse", counter)

    first = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert first.total_checks == 3
    assert first.passed_checks == 1
    assert first.failed_checks == 2
    assert counter.count == 3  # ilk calisma 3 dosyayi da parse eder
    first_violations = list(runner.ast_audit_stats["violations"])
    assert len(first_violations) == 2

    counter.count = 0
    second = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert counter.count == 0  # hicbir dosya degismedi: parse yok
    assert second.total_checks == first.total_checks
    assert second.passed_checks == first.passed_checks
    assert second.failed_checks == first.failed_checks
    assert list(runner.ast_audit_stats["violations"]) == first_violations
    assert (root / "logs" / ".ast_gate_cache.json").is_file()


def test_ast_gate_cache_invalidates_on_change(tmp_path, monkeypatch):
    root = _make_tree(tmp_path)
    runner = MasterSmokeTestRunner(root=root)
    counter = _ParseCounter()
    monkeypatch.setattr(ast, "parse", counter)

    runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert counter.count == 3

    # bad.py duzeltildi: pass govdesi kaldirildi.
    (tmp_path / "bad.py").write_text("def stub():\n    return 2\n", encoding="utf-8")
    counter.count = 0
    second = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert counter.count == 1  # yalnizca degisen dosya yeniden parse edilir
    assert second.passed_checks == 2
    assert second.failed_checks == 1
    assert all("bad.py" not in v for v in runner.ast_audit_stats["violations"])


def test_ast_gate_cache_corrupt_cache_ignored(tmp_path):
    root = _make_tree(tmp_path)
    runner = MasterSmokeTestRunner(root=root)
    first = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert first.failed_checks == 2

    (root / "logs" / ".ast_gate_cache.json").write_text("{bozuk json", encoding="utf-8")
    second = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert second.total_checks == first.total_checks
    assert second.passed_checks == first.passed_checks
    assert second.failed_checks == first.failed_checks


def test_ast_gate_cache_shared_across_runner_instances(tmp_path, monkeypatch):
    root = _make_tree(tmp_path)
    MasterSmokeTestRunner(root=root).run_suite_5_full_ast_zero_placeholder_gate()

    counter = _ParseCounter()
    monkeypatch.setattr(ast, "parse", counter)
    second_runner = MasterSmokeTestRunner(root=root)
    second = second_runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert counter.count == 0  # onbellek diskte kalici, yeni runner da isabet eder
    assert second.failed_checks == 2


# ---------------------------------------------------------------------------
# I2: Paket 5 hata korumasi
# ---------------------------------------------------------------------------

def test_suite5_discovery_failure_returns_failed_summary(tmp_path, monkeypatch):
    runner = MasterSmokeTestRunner(root=tmp_path)

    def _boom():
        raise RuntimeError("kesif patladi")

    monkeypatch.setattr(runner, "_discover_all_python_files", _boom)
    summary = runner.run_suite_5_full_ast_zero_placeholder_gate()
    assert summary.passed is False
    assert summary.total_checks == 1
    assert summary.passed_checks == 0
    assert summary.failed_checks == 1
    assert summary.return_code == 1
    assert summary.checks[0].category == "AST_GATE_ERROR"
    assert "kesif patladi" in summary.checks[0].details
    assert summary in runner.suite_summaries


# ---------------------------------------------------------------------------
# I3: Satir tabanli paketler — davranis kilidi (sahte alt paketlerle)
# ---------------------------------------------------------------------------

def test_suite2_row_mapping_and_counter_reconciliation(tmp_path, monkeypatch):
    rows = [
        {"item_no": "1", "category": "Gorunum", "title": "Birinci",
         "passed": True, "duration_ms": 3.0, "details": "ok"},
        {"item_no": "2", "category": "Gorunum", "title": "Ikinci",
         "passed": False, "duration_ms": 4.0, "details": "bozuk"},
        {"item_no": "3", "category": "Gorunum", "title": "Ucuncu",
         "passed": True, "duration_ms": 5.0, "details": "ok"},
    ]
    fake = _fake_module(
        "tools.smoke_test_vision_and_3x",
        run_vision_and_3x_smoke_tests=lambda print_report=False: (2, 3, rows),
    )
    monkeypatch.setitem(sys.modules, "tools.smoke_test_vision_and_3x", fake)
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner.run_suite_2_vision_and_3x()
    assert summary.passed is False
    assert summary.total_checks == 3
    assert summary.passed_checks == 2
    assert summary.failed_checks == 1
    assert [c.name for c in summary.checks] == ["#01 Birinci", "#02 Ikinci", "#03 Ucuncu"]
    assert summary.checks[1].details == "bozuk"


def test_suite2_counter_mismatch_fails_even_when_rows_pass(tmp_path, monkeypatch):
    rows = [
        {"item_no": "1", "category": "Gorunum", "title": "Birinci",
         "passed": True, "duration_ms": 1.0, "details": "ok"},
    ]
    fake = _fake_module(
        "tools.smoke_test_vision_and_3x",
        run_vision_and_3x_smoke_tests=lambda print_report=False: (99, 1, rows),
    )
    monkeypatch.setitem(sys.modules, "tools.smoke_test_vision_and_3x", fake)
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner.run_suite_2_vision_and_3x()
    # Satirlarin hepsi gecti ama alt paket sayaci (99) toplanan satirla (1)
    # uyusmuyor (F-2039): paket basarisiz sayilmali.
    assert summary.passed is False
    assert summary.passed_checks == 1


def test_suite3_row_mapping(tmp_path, monkeypatch):
    rows = [
        {"id": "M31", "category": "Mobil", "name": "Tünel",
         "passed": True, "elapsed_ms": 2.0, "detail": "ok"},
        {"id": "M32", "category": "Mobil", "name": "Ekonomi",
         "passed": False, "elapsed_ms": 3.0, "detail": "hata"},
    ]
    fake = _fake_module(
        "tools.smoke_test_mobile_and_enterprise",
        run_mobile_and_enterprise_smoke_tests=lambda verbose=False: (1, 1, rows),
    )
    monkeypatch.setitem(sys.modules, "tools.smoke_test_mobile_and_enterprise", fake)
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner.run_suite_3_mobile_and_enterprise()
    assert summary.passed is False
    assert summary.total_checks == 2
    assert summary.passed_checks == 1
    assert [c.name for c in summary.checks] == ["[M31] Tünel", "[M32] Ekonomi"]


def test_suite6_zk_report_failure_is_warning_not_silent(tmp_path, monkeypatch):
    rows = [
        {"id": "ZK1", "category": "ZK", "name": "Algı",
         "passed": True, "elapsed_ms": 2.0, "detail": "ok"},
    ]

    def _boom(*args, **kwargs):
        raise OSError("rapor yazilamadi")

    fake = _fake_module(
        "tools.smoke_test_zk_agent",
        run_zk_agent_smoke_tests=lambda verbose=False: (1, 0, rows),
        write_zk_smoke_report=_boom,
    )
    monkeypatch.setitem(sys.modules, "tools.smoke_test_zk_agent", fake)
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner.run_suite_6_zk_vision_agent()
    assert summary.passed is True
    assert "UYARI: ZK raporu yazilamadi" in summary.summary_tr
    assert "rapor yazilamadi" in summary.summary_tr


def test_suite1_import_failure_returns_error_summary(tmp_path, monkeypatch):
    class _BoomSuite:
        def __init__(self, root):
            raise RuntimeError("self_test import patladi")

    fake = _fake_module("tools.self_test_suite", SelfTestSuite=_BoomSuite)
    monkeypatch.setitem(sys.modules, "tools.self_test_suite", fake)
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner.run_suite_1_core_self_test()
    assert summary.passed is False
    assert summary.total_checks == 1
    assert summary.checks[0].category == "SELF_TEST_ERROR"


def test_error_suite_summary_shape(tmp_path):
    runner = MasterSmokeTestRunner(root=tmp_path)
    summary = runner._error_suite_summary(
        suite_id="SUITE_X",
        title_tr="Deneme Paketi",
        rel_path="tools/deneme.py",
        error_category="DENEME_ERROR",
        exc=RuntimeError("patladi"),
        duration_ms=7.5,
    )
    assert summary.passed is False
    assert summary.total_checks == 1
    assert summary.passed_checks == 0
    assert summary.failed_checks == 1
    assert summary.return_code == 1
    assert summary.duration_ms == 7.5
    assert summary.checks[0].category == "DENEME_ERROR"
    assert "patladi" in summary.checks[0].details
    assert summary.checks[0] in runner.all_checks
    assert summary in runner.suite_summaries


# ---------------------------------------------------------------------------
# I4: run_all_smoke_tests diskten yeniden okuma yapmaz
# ---------------------------------------------------------------------------

def test_run_all_smoke_tests_no_disk_reread(tmp_path, monkeypatch):
    from tools.run_all_smoke_tests import SuiteExecutionSummary, SmokeCheckItem

    def _stub_suite(self):
        chk = SmokeCheckItem(
            suite_id="SUITE_STUB", category="STUB", name="stub",
            passed=True, duration_ms=1.0, details="ok",
        )
        self.all_checks.append(chk)
        summary = SuiteExecutionSummary(
            suite_id="SUITE_STUB", title_tr="Stub", script_rel_path="stub",
            passed=True, total_checks=1, passed_checks=1, failed_checks=0,
            duration_ms=1.0, return_code=0, summary_tr="stub gecti",
            checks=[chk],
        )
        self.suite_summaries.append(summary)
        return summary

    for name in (
        "run_suite_1_core_self_test",
        "run_suite_2_vision_and_3x",
        "run_suite_3_mobile_and_enterprise",
        "run_suite_4_offscreen_turkish_ui",
        "run_suite_5_full_ast_zero_placeholder_gate",
        "run_suite_6_zk_vision_agent",
    ):
        monkeypatch.setattr(MasterSmokeTestRunner, name, _stub_suite)

    report_path = tmp_path / "logs" / "smoke_test_report.json"
    orig_read_text = Path.read_text
    reads = {"n": 0}

    def _counting_read_text(self, *args, **kwargs):
        if str(self) == str(report_path):
            reads["n"] += 1
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counting_read_text)
    passed, payload = run_all_smoke_tests(root=tmp_path)
    assert reads["n"] == 0  # rapor diskten yeniden okunmadi
    assert passed is True
    assert report_path.is_file()
    on_disk = json.loads(orig_read_text(report_path, encoding="utf-8"))
    assert payload == on_disk
