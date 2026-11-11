# -*- coding: utf-8 -*-
"""v20 bot ani kapanma (11 Kasim 2026 17:33:54 sessiz olum) -- kanit altyapisi.

KOSULMADI -- testleri ana ajan kosar (kural: duzeltme ajanlari test kosmaz).
Kapsam: core/crash_evidence.py (faulthandler + excepthook + threading
excepthook + atexit kapanis kaydi + RSS izleme + uzak-cikis izi).

Canli karsiligi: 17:33:54 logu kapanis satiri/traceback olmadan kesildi.
Bu testler o sessizligin bir daha yasanmamasini dogrular: yakalanmamis
istisna crash dosyasina duser, atexit kaydi graceful bayragini yazar,
RSS buyumesi uyarir, uzaktan os._exit oncesi CRITICAL satir birakir.
"""

import faulthandler
import logging
import sys
import threading
import time

import pytest

from core import crash_evidence as ce


@pytest.fixture()
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture(autouse=True)
def _reset():
    yield
    ce._reset_state_for_tests()


def _install(log_dir, **kwargs):
    kwargs.setdefault("rss_interval_sec", 60.0)
    assert ce.install_crash_evidence(str(log_dir), **kwargs) is True
    return log_dir


def _crash_files(log_dir):
    return sorted(log_dir.glob("crash_*.log"))


def test_install_idempotent_ve_faulthandler_acik(log_dir):
    _install(log_dir)
    first_thread = ce._state["monitor_thread"]
    assert faulthandler.is_enabled()
    assert (log_dir / "faulthandler.log").exists()
    # Ikinci kurulum hicbir seyi cogaltmaz.
    assert ce.install_crash_evidence(str(log_dir)) is True
    assert ce._state["monitor_thread"] is first_thread
    assert sys.excepthook is ce._excepthook


def test_unhandled_excepthook_crash_dosyasi_yazar_ve_zincirler(log_dir, caplog):
    chained = []
    sys.excepthook = lambda *a: chained.append(a)
    _install(log_dir)
    with caplog.at_level(logging.DEBUG):
        try:
            raise RuntimeError("deneme-patlamasi")
        except RuntimeError:
            exc_type, exc_value, tb = sys.exc_info()
            ce._excepthook(exc_type, exc_value, tb)
    files = _crash_files(log_dir)
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "kind=unhandled" in text
    assert "RuntimeError" in text
    assert "deneme-patlamasi" in text
    assert "rss_mb=" in text
    assert "threads=" in text
    assert len(chained) == 1  # onceki hook'a zincirlendi (stderr izi korunur)
    assert any("YAKALANMAMIS ISTISNA" in r.message for r in caplog.records)
    assert any(r.levelno >= logging.CRITICAL for r in caplog.records)


def test_keyboard_interrupt_sadece_zincirler_crash_yazmaz(log_dir):
    chained = []
    sys.excepthook = lambda *a: chained.append(a)
    _install(log_dir)
    ce._excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert _crash_files(log_dir) == []
    assert len(chained) == 1


def test_threading_excepthook_crash_dosyasi_yazar_process_olmez(log_dir, caplog):
    _install(log_dir)
    args_cls = getattr(threading, "ExceptHookArgs", None)
    try:
        raise ValueError("isci-thread-patlamasi")
    except ValueError:
        exc_type, exc_value, tb = sys.exc_info()
    if args_cls is not None:
        # structseq: tek argüman olarak 4'lü dizi alır (4 ayrı argüman değil)
        args = args_cls((exc_type, exc_value, tb, threading.current_thread()))
    else:  # pragma: no cover - cok eski Python
        import types

        args = types.SimpleNamespace(
            exc_type=exc_type, exc_value=exc_value, exc_traceback=tb,
            thread=threading.current_thread(),
        )
    with caplog.at_level(logging.DEBUG):
        ce._threading_excepthook(args)
    files = _crash_files(log_dir)
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "ValueError" in text
    assert "isci-thread-patlamasi" in text
    assert any("YASIYOR" in r.message for r in caplog.records)


def test_write_crash_file_asla_patlamaz(log_dir):
    _install(log_dir)
    # Gecersiz girdiyle bile exception firlatmaz (kanit yazimi olumu
    # engelleyemez); gecerli girdiyle dosya yazilir.
    ce._write_crash_file("bozuk", None, None, None)
    try:
        raise RuntimeError("gecerli")
    except RuntimeError:
        exc_type, exc_value, tb = sys.exc_info()
    path = ce._write_crash_file("gecerli", exc_type, exc_value, tb)
    assert path is not None
    assert "gecerli" in path
    assert "RuntimeError" in open(path, encoding="utf-8").read()


def test_rss_buyumesi_uyari_verir_ve_tur_sayimi_doker(log_dir, caplog):
    values = iter([100.0, 100.0, 100.0, 700.0, 700.0, 700.0])

    def fake_reader():
        try:
            return next(values)
        except StopIteration:
            return 700.0

    _install(
        log_dir,
        rss_interval_sec=0.05,
        rss_warn_growth_mb=512.0,
        rss_warn_total_mb=100000.0,
        rss_reader=fake_reader,
    )
    deadline = time.monotonic() + 3.0
    with caplog.at_level(logging.DEBUG):
        while time.monotonic() < deadline:
            if any("BELLEK UYARISI" in r.message for r in caplog.records):
                break
            time.sleep(0.05)
    assert any("BELLEK UYARISI" in r.message for r in caplog.records)
    warn = [r for r in caplog.records if "BELLEK UYARISI" in r.message][0]
    assert "700.0" in warn.message  # RSS degeri dokumde
    assert "en cok nesne turleri" in warn.message


def test_rss_normalken_uyari_yok_bilgi_satiri_var(log_dir, caplog):
    _install(log_dir, rss_interval_sec=0.05, rss_reader=lambda: 123.0)
    deadline = time.monotonic() + 1.0
    with caplog.at_level(logging.DEBUG):
        while time.monotonic() < deadline:
            if any("RSS=123.0 MB" in r.message for r in caplog.records):
                break
            time.sleep(0.05)
    assert any("RSS=123.0 MB" in r.message for r in caplog.records)
    assert not any("BELLEK UYARISI" in r.message for r in caplog.records)


def test_uzaktan_cikis_oncesi_critical_iz_birakir(log_dir, caplog):
    _install(log_dir)
    with caplog.at_level(logging.DEBUG):
        ce.log_imminent_process_exit("quit", 1.5)
    assert any("UZAKTAN CIKIS" in r.message for r in caplog.records)
    assert any("action=quit" in r.message for r in caplog.records)
    assert any(r.levelno >= logging.CRITICAL for r in caplog.records)
    trace = (log_dir / "exit_trace.log").read_text(encoding="utf-8")
    assert "remote_exit=quit" in trace


def test_atexit_kaydi_graceful_bayragini_yazar(log_dir):
    _install(log_dir)
    ce._record_exit()
    trace = (log_dir / "exit_trace.log").read_text(encoding="utf-8")
    assert "graceful=False" in trace
    ce.note_graceful_shutdown()
    ce._record_exit()
    lines = (log_dir / "exit_trace.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "graceful=True" in lines[1]
