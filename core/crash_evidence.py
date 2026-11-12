# -*- coding: utf-8 -*-
"""Sessiz process olumu karsisinda kanit altyapisi (v20 / bot ani kapanma).

11 Kasim 2026, 17:33:54 v20 canli logu BIRDEN kesildi: kapanis satiri,
traceback, hata satiri YOK. Kod taramasi (main.py, core/watchdog.py,
core/opsec_crypto.py, core/mobile_server.py, core/state_machine.py,
core/telemetry.py, core/omni_game_learner.py, dataset_factory/*) sunlari
gosterdi:

1. TRACEBACK BOLUNMESI (kesin, kod kanitli) -- main.py:_attach_windowless_stdio
   (satir ~44) pythonw altinda stderr'i ``logs/gatherion.log``'a baglar;
   setup_logging() ise logger ciktisini ``logs/session_*.log``'a yazar.
   Yani ana thread'de yakalanmamis bir istisnanin traceback'i OTURUM
   LOGUNDA HIC GORUNMEZ; Selcuk'un yapistirdigi logda "traceback yok"
   demek, traceback olmadigi anlamina gelmez. Bu moduldaki excepthook
   traceback'i hem logger'a (oturum logu) hem de ``logs/crash_*.log``'a
   yazar; boylece bir daha "iz yok" durumu olmaz.

2. GRACEFUL KAPANIS IZI VAR, OLAYDA YOK -- GatherionOrchestrator.shutdown()
   (main.py:2942) "kapatiliyor" diye loglar. 17:33:54 logunda bu satir yok;
   yani olum menu/tray/mobil-kapatma yolundan (hepsi shutdown() cagirir)
   GECMEDI. Panik (OpsecCrypto.trigger_panic, core/opsec_crypto.py:1911)
   ve watchdog icinde ``os._exit`` / ``app.quit`` YOK -- panik ve Alt-Tab
   process oldurmez (kod kaniti). Uzaktan kapatma da bu olayda elendi:
   do_POST kimlik dogrulamasi zorunlu ve logda "[Mobil Mudahale] Uzaktan
   kapatma istegi alindi." satiri yok.

3. BELLEK -- watchdog payload'larinda RSS 17:26:18'de 2237 MB -> 17:31:30'da
   3458 MB (~7 dk'da +1,2 GB). Buna ragmen RAM'de sinirsiz buyuyen bir
   Python listesi BULUNAMADI: watchdog bildirimleri deque(maxlen=100),
   telemetri _append_capped ile sinirli, incident max 50, dhash gecmisi
   sinirli, kare ring buffer 3 slot onceden ayrilmis, time-machine
   sinirli deque. v20'deki onbelleksiz wrong_node_scan ONNX session
   uretimi loop duzeltmesinde (b94c1e2 tabaninda var) zaten kapatildi.
   RSS artisinin somut kaynagi bu taramada BULUNAMADI -- o yuzden bu
   modul periyodik RSS izlemesi + esik asiminda tur sayimi dokumu ekler;
   bir dahaki sefere sizi tasiyan somut sayi olur. (OOM olum nedeni
   DEGIL: 3,4 GB / 32 GB, sistem %55 doluydu.)

4. KALAN OLUM YOLLARI (koddan elenemedi, kanit lazim) -- ana thread'de
   yakalanmamis istisna (en olasi; madde 1 yuzunden izsiz kaldi), native
   segfault (ONNX CUDA / DXGI / Qt; traceback cikmaz, log birden kesilir),
   disaridan oldurme (Gorev Yoneticisi / antivirus / oyun anti-hile).
   faulthandler + excepthook + atexit kapanis kaydi bu ucu de kapatir:
   - faulthandler dosyasi varsa -> native cokme,
   - crash_*.log varsa -> yakalanmamis istisna,
   - exit_trace.log'da graceful=True -> normal kapanis,
   - hicbiri yoksa -> os._exit / SIGKILL / disaridan oldurme.

Bu modul yalnizca stdlib'e dayanir (psutil istege bagli, yoksa RSS
izlemesi sessizce atlanir); Qt yok, repo ici import yok. Davranis
degistirmez: panik/watchdog/Alt-Tab mantigina dokunmaz.
"""

from __future__ import annotations

import atexit
import collections
import faulthandler
import gc
import logging
import os
import sys
import threading
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "install_crash_evidence",
    "note_graceful_shutdown",
    "log_imminent_process_exit",
]

_LOG = logging.getLogger("Gatherion.CrashEvidence")

# RSS izleme varsayilanlari: 60 sn'de bir satir; baslangica gore +512 MB
# buyume ya da toplam 6 GB asilirsa WARNING + tur sayimi dokumu.
RSS_INTERVAL_SEC = 60.0
RSS_WARN_GROWTH_MB = 512.0
RSS_WARN_TOTAL_MB = 6144.0
# fix/v20-rss-panik: bellek dokumu TEK SEFERLIK degil — esik asili
# kalirken her 5 dk'da bir tekrarlanir (tur sayimi + tracemalloc
# farki). 12 Kasim kaniti: egitim durduktan SONRA da RSS ~300 MB/dk
# artmaya devam etti; tek dokum hangi turun buyudugunu gostermez,
# tekrarli dokum + fark gosterir. tracemalloc yalnizca ilk uyarida
# baslatilir (normal calismada ek yuk yok); fark ikinci dokumden
# itibaren loglanir.
MEM_DUMP_REPEAT_SEC = 300.0
MEM_DUMP_TOP_TYPES = 15
MEM_DUMP_TOP_TRACES = 5
TRACEMALLOC_FRAMES = 5

_state: Dict[str, Any] = {
    "installed": False,
    "log_dir": None,
    "logger": None,
    "faulthandler_file": None,
    "prev_excepthook": None,
    "prev_threading_excepthook": None,
    "graceful": False,
    "start_mono": None,
    "start_rss_mb": None,
    "monitor_stop": None,
    "monitor_thread": None,
    "rss_reader": None,
    "last_mem_dump_mono": None,
    "tracemalloc_snapshot": None,
    "tracemalloc_started_here": False,
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _log_dir_path() -> Optional[Path]:
    log_dir = _state.get("log_dir")
    if not log_dir:
        return None
    try:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _read_rss_mb() -> Optional[float]:
    """Process RSS (MB). Testlerde _rss_reader ile degistirilebilir."""
    reader = _state.get("rss_reader")
    if reader is not None:
        try:
            value = reader()
            return float(value) if value is not None else None
        except Exception:
            return None
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _flush_logger_handlers() -> None:
    """os._exit oncesi log satirlarinin diske ulastigindan emin ol."""
    try:
        for handler in list(logging.getLogger().handlers):
            try:
                handler.flush()
            except Exception:
                pass
    except Exception:
        pass


def _write_crash_file(kind: str, exc_type: Any, exc_value: Any, tb: Any) -> Optional[str]:
    """Traceback + o anki durum dokumu; hicbir durumda exception firlatmaz."""
    log_dir = _log_dir_path()
    if log_dir is None:
        return None
    fname = "crash_%s_%s.log" % (_now_stamp(), kind)
    path = log_dir / fname
    try:
        lines: List[str] = []
        lines.append("timestamp=%s" % datetime.now().isoformat(timespec="seconds"))
        lines.append("kind=%s" % kind)
        lines.append("graceful_shutdown_noted=%s" % bool(_state.get("graceful")))
        rss = _read_rss_mb()
        lines.append("rss_mb=%s" % ("%.1f" % rss if rss is not None else "unknown"))
        try:
            lines.append("threads=%s" % [t.name for t in threading.enumerate()])
        except Exception:
            pass
        try:
            lines.append("gc_counts=%s" % (gc.get_count(),))
        except Exception:
            pass
        lines.append("")
        try:
            lines.append("".join(traceback.format_exception(exc_type, exc_value, tb)))
        except Exception as exc:
            lines.append("(traceback bicimlenemedi: %r)" % (exc,))
        path.write_text("\n".join(lines), encoding="utf-8", errors="replace")
        return str(path)
    except Exception:
        return None


def _excepthook(exc_type: Any, exc_value: Any, tb: Any) -> None:
    """Ana thread'de yakalanmamis istisna: process kapaniyor demektir."""
    if issubclass(exc_type, KeyboardInterrupt):
        prev = _state.get("prev_excepthook")
        if prev is not None:
            try:
                prev(exc_type, exc_value, tb)
            except Exception:
                pass
        return
    crash_path = _write_crash_file("unhandled", exc_type, exc_value, tb)
    logger = _state.get("logger") or _LOG
    try:
        logger.critical(
            "[CRASH-EVIDENCE] YAKALANMAMIS ISTISNA -- process kapaniyor "
            "(%s: %s) | kanit: %s",
            getattr(exc_type, "__name__", exc_type),
            exc_value,
            crash_path or "yazilamadi",
        )
    except Exception:
        pass
    _flush_logger_handlers()
    prev = _state.get("prev_excepthook")
    if prev is not None:
        try:
            prev(exc_type, exc_value, tb)
        except Exception:
            pass


def _threading_excepthook(args: Any) -> None:
    """Isci thread'de yakalanmamis istisna: process olmez ama kaydedilir."""
    thread_obj = getattr(args, "thread", None)
    thread_name = getattr(thread_obj, "name", "unknown") if thread_obj is not None else "unknown"
    crash_path = _write_crash_file(
        "thread_%s" % thread_name,
        getattr(args, "exc_type", None),
        getattr(args, "exc_value", None),
        getattr(args, "exc_traceback", None),
    )
    logger = _state.get("logger") or _LOG
    try:
        logger.error(
            "[CRASH-EVIDENCE] Is parcaciginda yakalanmamis istisna "
            "(thread=%s, %s: %s; process YASIYOR) | kanit: %s",
            getattr(getattr(args, "thread", None), "name", "?"),
            getattr(getattr(args, "exc_type", None), "__name__", "?"),
            getattr(args, "exc_value", "?"),
            crash_path or "yazilamadi",
        )
    except Exception:
        pass
    prev = _state.get("prev_threading_excepthook")
    if prev is not None:
        try:
            prev(args)
        except Exception:
            pass


def _record_exit() -> None:
    """atexit: normal yorumlayici kapanisinda tek satirlik iz birakir.

    os._exit / segfault / SIGKILL durumunda CALISMAZ -- iste bu yokluk
    da kanittir: crash dosyasi da yoksa olum bu yollardan biridir.
    """
    stop = _state.get("monitor_stop")
    if stop is not None:
        try:
            stop.set()
        except Exception:
            pass
    log_dir = _log_dir_path()
    if log_dir is None:
        return
    try:
        rss = _read_rss_mb()
        line = (
            "timestamp=%s graceful=%s rss_mb=%s pid=%d\n"
            % (
                datetime.now().isoformat(timespec="seconds"),
                bool(_state.get("graceful")),
                ("%.1f" % rss) if rss is not None else "unknown",
                os.getpid(),
            )
        )
        with open(log_dir / "exit_trace.log", "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line)
    except Exception:
        pass
    logger = _state.get("logger")
    if logger is not None:
        try:
            logger.info(
                "[CRASH-EVIDENCE] Process atexit'e ulasti (graceful=%s, rss=%s MB) -- "
                "bu satir yoksa olum os._exit / segfault / disaridan oldurme yoluyladir.",
                bool(_state.get("graceful")),
                ("%.1f" % rss) if rss is not None else "unknown",
            )
        except Exception:
            pass
    _flush_logger_handlers()


def note_graceful_shutdown() -> None:
    """GatherionOrchestrator.shutdown() tarafindan cagrilir: temiz kapanis."""
    _state["graceful"] = True


def log_imminent_process_exit(action: str, delay_sec: float = 0.0) -> None:
    """os._exit(0) oncesi SON iz: mobil uzak-eylem cikis yollarinda kullanilir.

    Eskiden _do_quit/_do_restart_exit/_do_exit_after_elevate sessizce
    os._exit(0) yapardi; oturum logunda yalnizca WARNING satiri kalirdi
    (elevate yolunda hic shutdown da yoktu). Artik CRITICAL satir + flush
    var; bir daha ani kapanmada bu satir varsa sebep uzaktan cikistir.
    """
    logger = _state.get("logger") or _LOG
    try:
        logger.critical(
            "[CRASH-EVIDENCE] UZAKTAN CIKIS: action=%s -- %.1f sn icinde "
            "os._exit(0) yapilacak (bilincli kapatma, ariza degil).",
            action,
            float(delay_sec),
        )
    except Exception:
        pass
    log_dir = _log_dir_path()
    if log_dir is not None:
        try:
            with open(log_dir / "exit_trace.log", "a", encoding="utf-8", errors="replace") as fh:
                fh.write(
                    "timestamp=%s graceful=%s remote_exit=%s\n"
                    % (
                        datetime.now().isoformat(timespec="seconds"),
                        bool(_state.get("graceful")),
                        action,
                    )
                )
        except Exception:
            pass
    _flush_logger_handlers()


def _top_object_types(limit: int = MEM_DUMP_TOP_TYPES) -> str:
    """En cok nesne turleri sayimi (gc.get_objects). Hata durumunda kisa mesaj."""
    try:
        counter: Dict[str, int] = collections.Counter()
        for obj in gc.get_objects():
            try:
                counter[type(obj).__name__] += 1
            except Exception:
                pass
        return ", ".join("%s=%d" % item for item in counter.most_common(limit))
    except Exception as exc:
        return "sayim hatasi: %r" % (exc,)


def _tracemalloc_diff_lines(limit: int = MEM_DUMP_TOP_TRACES) -> str:
    """tracemalloc farki: onceki anlik goruntuyle karsilastirir.

    Ilk cagrida izleme baslatilir + taban anlik goruntu alinir, fark
    henuz yoktur ("taban alindi" doner). Sonraki cagrilar en cok
    buyuyen tahsis satirlarini dondurur. Izleme disaridan zaten
    aciksa ona dokunulmaz, yalnizca fark alinir.
    """
    try:
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start(TRACEMALLOC_FRAMES)
            _state["tracemalloc_started_here"] = True
            tracing = True
        current = tracemalloc.take_snapshot()
        previous = _state.get("tracemalloc_snapshot")
        _state["tracemalloc_snapshot"] = current
        if previous is None:
            return "tracemalloc taban anlik goruntusu alindi; fark bir sonraki dokumde"
        try:
            stats = current.compare_to(previous, "traceback")
        except Exception as exc:
            return "tracemalloc farki alinamadi: %r" % (exc,)
        lines = []
        for stat in stats[:limit]:
            frame = stat.traceback[0] if stat.traceback else None
            where = (
                "%s:%s" % (frame.filename, frame.lineno)
                if frame is not None
                else "bilinmiyor"
            )
            lines.append(
                "%s +%0.1f MB (%+d tahsis)" % (where, stat.size_diff / (1024.0 * 1024.0), stat.count_diff)
            )
        return "; ".join(lines) if lines else "tracemalloc farki bos"
    except Exception as exc:
        return "tracemalloc hatasi: %r" % (exc,)


def _maybe_memory_dump(
    logger: Any,
    rss: float,
    growth: float,
    now_mono: float,
    warn_growth: float,
    warn_total: float,
) -> bool:
    """Esik asili kalirken periyodik bellek dokumu yapar; doktu ise True.

    Tek seferlik degil: esik asildiktan sonra her MEM_DUMP_REPEAT_SEC
    saniyede bir tekrarlar. Esik altina inilirse sayac sifirlanmaz —
    bir sonraki asimda aninda dokulur (last_mem_dump yalnizca dokumde
    guncellenir).
    """
    if growth < warn_growth and rss < warn_total:
        return False
    last = _state.get("last_mem_dump_mono")
    if last is not None and (now_mono - float(last)) < MEM_DUMP_REPEAT_SEC:
        return False
    _state["last_mem_dump_mono"] = now_mono
    top_types = _top_object_types()
    trace_diff = _tracemalloc_diff_lines()
    try:
        logger.warning(
            "[CRASH-EVIDENCE] BELLEK UYARISI: RSS=%.1f MB "
            "(baslangictan %+.1f MB) -- en cok nesne turleri: %s | "
            "tracemalloc farki: %s",
            rss,
            growth,
            top_types,
            trace_diff,
        )
    except Exception:
        pass
    return True


def _rss_monitor_loop() -> None:
    stop = _state.get("monitor_stop")
    interval = float(_state.get("rss_interval_sec") or RSS_INTERVAL_SEC)
    warn_growth = float(_state.get("rss_warn_growth_mb") or RSS_WARN_GROWTH_MB)
    warn_total = float(_state.get("rss_warn_total_mb") or RSS_WARN_TOTAL_MB)
    logger = _state.get("logger") or _LOG
    while stop is None or not stop.wait(timeout=interval):
        try:
            rss = _read_rss_mb()
        except Exception:
            rss = None
        if rss is None:
            # psutil yok/okunamadi: bu tur atlanir, sonraki tur yeniden denenir.
            continue
        if _state.get("start_rss_mb") is None:
            _state["start_rss_mb"] = rss
        start_rss = _state.get("start_rss_mb") or rss
        growth = rss - start_rss
        try:
            gc_counts = gc.get_count()
        except Exception:
            gc_counts = "?"
        try:
            n_threads = len(threading.enumerate())
        except Exception:
            n_threads = -1
        try:
            logger.info(
                "[CRASH-EVIDENCE] RSS=%.1f MB (baslangictan %+.1f MB) "
                "threads=%d gc=%s",
                rss,
                growth,
                n_threads,
                gc_counts,
            )
        except Exception:
            pass
        # fix/v20-rss-panik: tekrarli dokum — esik asili kalirken her
        # 5 dk'da bir tur sayimi + tracemalloc farki (tek dokum hangi
        # turun buyudugunu gostermez).
        try:
            _maybe_memory_dump(
                logger, rss, growth, time.monotonic(), warn_growth, warn_total
            )
        except Exception:
            pass


def install_crash_evidence(
    log_dir: Any,
    logger: Optional[logging.Logger] = None,
    rss_interval_sec: float = RSS_INTERVAL_SEC,
    rss_warn_growth_mb: float = RSS_WARN_GROWTH_MB,
    rss_warn_total_mb: float = RSS_WARN_TOTAL_MB,
    rss_reader: Optional[Callable[[], Optional[float]]] = None,
) -> bool:
    """Kanit altyapisini kurar. Idempotent: ikinci cagri hicbir sey yapmaz.

    main() icinde, setup_logging() sonrasi ve app.exec() oncesi cagrilir.
    faulthandler dosyasi ``logs/faulthandler.log``'dur (native cokmede
    Python traceback'i buraya duser); excepthook ``logs/crash_*.log``
    yazar; atexit ``logs/exit_trace.log``'a tek satir yazar.
    """
    if _state.get("installed"):
        return True
    _state["installed"] = True
    _state["log_dir"] = str(log_dir)
    _state["logger"] = logger
    _state["start_mono"] = time.monotonic()
    _state["rss_interval_sec"] = float(rss_interval_sec)
    _state["rss_warn_growth_mb"] = float(rss_warn_growth_mb)
    _state["rss_warn_total_mb"] = float(rss_warn_total_mb)
    if rss_reader is not None:
        _state["rss_reader"] = rss_reader
    log = logger or _LOG

    # 1) faulthandler: SIGSEGV/SIGABRT gibi native olumlerde iz birakir.
    try:
        path = _log_dir_path()
        if path is not None:
            fh = open(path / "faulthandler.log", "a", encoding="utf-8", errors="replace")
            _state["faulthandler_file"] = fh
            faulthandler.enable(file=fh)
    except Exception as exc:
        try:
            log.warning("[CRASH-EVIDENCE] faulthandler kurulamadi: %s", exc)
        except Exception:
            pass

    # 2) Ana thread excepthook (onceki hook'a zincirlenir).
    try:
        _state["prev_excepthook"] = sys.excepthook
        sys.excepthook = _excepthook
    except Exception as exc:
        try:
            log.warning("[CRASH-EVIDENCE] excepthook kurulamadi: %s", exc)
        except Exception:
            pass

    # 3) Isci thread excepthook (Python 3.8+).
    try:
        if hasattr(threading, "excepthook"):
            _state["prev_threading_excepthook"] = threading.excepthook
            threading.excepthook = _threading_excepthook
    except Exception as exc:
        try:
            log.warning("[CRASH-EVIDENCE] threading excepthook kurulamadi: %s", exc)
        except Exception:
            pass

    # 4) atexit kapanis kaydi.
    try:
        atexit.register(_record_exit)
    except Exception as exc:
        try:
            log.warning("[CRASH-EVIDENCE] atexit kurulamadi: %s", exc)
        except Exception:
            pass

    # 5) RSS izleme (daemon thread; olumu engellemez).
    try:
        stop = threading.Event()
        _state["monitor_stop"] = stop
        thread = threading.Thread(
            target=_rss_monitor_loop, name="CrashEvidenceRSS", daemon=True
        )
        _state["monitor_thread"] = thread
        thread.start()
    except Exception as exc:
        try:
            log.warning("[CRASH-EVIDENCE] RSS izleme baslatilamadi: %s", exc)
        except Exception:
            pass

    try:
        log.info(
            "[CRASH-EVIDENCE] Kanit altyapisi kuruldu: faulthandler=%s "
            "crash dosyalari=%s | exit izi=%s",
            "logs/faulthandler.log" if _state.get("faulthandler_file") is not None else "KAPALI",
            "logs/crash_*.log",
            "logs/exit_trace.log",
        )
    except Exception:
        pass
    return True


def _reset_state_for_tests() -> None:
    """YALNIZCA TESTLER ICIN: hook'lari geri alir, izleme thread'ini durdurur."""
    try:
        prev = _state.get("prev_excepthook")
        if prev is not None:
            sys.excepthook = prev
    except Exception:
        pass
    try:
        prev_t = _state.get("prev_threading_excepthook")
        if prev_t is not None and hasattr(threading, "excepthook"):
            threading.excepthook = prev_t
    except Exception:
        pass
    try:
        stop = _state.get("monitor_stop")
        if stop is not None:
            stop.set()
    except Exception:
        pass
    try:
        fh = _state.get("faulthandler_file")
        if fh is not None:
            try:
                faulthandler.disable()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        atexit.unregister(_record_exit)
    except Exception:
        pass
    try:
        if _state.get("tracemalloc_started_here") and tracemalloc.is_tracing():
            tracemalloc.stop()
    except Exception:
        pass
    _state.update(
        {
            "installed": False,
            "log_dir": None,
            "logger": None,
            "faulthandler_file": None,
            "prev_excepthook": None,
            "prev_threading_excepthook": None,
            "graceful": False,
            "start_mono": None,
            "start_rss_mb": None,
            "monitor_stop": None,
            "monitor_thread": None,
            "rss_reader": None,
            "last_mem_dump_mono": None,
            "tracemalloc_snapshot": None,
            "tracemalloc_started_here": False,
        }
    )
