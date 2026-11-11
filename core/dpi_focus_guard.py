"""
Items 46, 47, 48, and 49:
  - Item 46: RouteBackupManager (10-Min Automated Route Backups & Version History with Corruption Auto-Restore)
  - Item 47: DPIScalingSynchronizer (Multi-Monitor Per-Monitor-V2 DPI Awareness & 125%/150% Coordinate Compensation)
  - Item 48: ForegroundFocusGuard (< 0.1ms GetForegroundWindow Focus Protection & Keystroke Suppression)
  - Item 49: SelfHealingTemplateTuner (Self-Healing 80x80 Node Patch Refresher & Adaptive Threshold Tuner)
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from version import __version__

logger = logging.getLogger("Gatherion.DPIFocusGuard")


# ============================================================================
# ITEM 46: Otomatik Rota Yedekleme ve Sürüm Geçmişi (RouteBackupManager)
# ============================================================================

class RouteBackupManager:
    """
    Item 46 — Automated 10-minute (600s) route JSON backup and version history manager.
    Backs up `routes/<region>/region_data.json` into `backups/region_data_<region>_YYYYMMDD_HHMMSS_ffffff.json`
    (mikrosaniye; aynı mikrosaniyeye düşen çakışmada `_<NNNN>` eki),
    retains the most recent `max_snapshots` (default 30) backups per region, validates JSON integrity,
    and automatically restores the latest valid snapshot if a route JSON file is corrupted or truncated.
    """

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        backup_interval_sec: float = 600.0,
        max_snapshots: int = 30,
        auto_start_daemon: bool = True,
    ) -> None:
        # F-1827 / F-0082: start_periodic_backup_daemon()'un üretimde
        # hiçbir çağıranı yoktu — otomatik 10 dakikalık yedekler hiç
        # çalışmıyordu. Daemon kurulumda başlar (main.py:421 ve
        # state_machine._init_optional_subsystem üretimde varsayılan
        # kurucuyla çağırır); istemeyen çağıran auto_start_daemon=False
        # verir. Thread daemon=True'dur; stop_periodic_backup_daemon()
        # ile durdurulur.
        self.base_dir: Path = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.routes_dir: Path = self.base_dir / "routes"
        self.backups_dir: Path = self.base_dir / "backups"
        self.backup_interval_sec: float = max(5.0, float(backup_interval_sec))
        self.max_snapshots: int = max(1, int(max_snapshots))

        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.routes_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._last_backup_ts_by_region: Dict[str, float] = {}
        self._last_hash_by_region: Dict[str, str] = {}
        # D10: daemon tick'i (5 sn'de bir) çok MB'lık rota dosyasını her seferinde
        # okuyup JSON parse + SHA-256 yapmasın diye: son doğrulanmış-okunuşun
        # (boyut, mtime_ns) anahtarı + imzayla birlikte saklanan stat anahtarı.
        # İki ayrı değişmez (invariant) vardır ve karıştırılmaz:
        # - _last_known_good_stat: YEDEĞİ ALINMIŞ içeriğin stat'ı. Yalnızca
        #   yedek oluşturulunca / geri yüklenince / dosya yedekle birebir
        #   aynı çıkınca güncellenir. Değişmiş ama henüz yedeklenmemiş
        #   dosyayı "biliniyor" işaretlemek, sonraki backup_region çağrısının
        #   hızlı yoldan mevcut yedeği döndürüp yeni yedek atlamasına yol
        #   açar (daemon tick'i gözlemleyip yedeklemeden döner).
        # - _sig_cache_by_region: (stat -> imza) önbelleği. İmza, yedek
        #   hash'inden (_last_hash_by_region) AYRI tutulur; yoksa değişmiş
        #   dosyayı gözlemleyen tick, önbelleğe eski yedek hash'ini
        #   düşürür ve sonraki çağrı değişikliği hiç görmez.
        self._last_known_good_stat: Dict[str, Tuple[int, int]] = {}
        self._sig_cache_by_region: Dict[str, Tuple[Tuple[int, int], str]] = {}
        self._seq_counter: int = 0

        if auto_start_daemon:
            try:
                self.start_periodic_backup_daemon()
            except Exception as exc:
                logger.warning("RouteBackupManager: otomatik yedek daemon'u başlatılamadı: %s", exc)

    @staticmethod
    def _atomic_copy_file(src: Path, dst: Path) -> bool:
        """P2-6 (K-05/K-19): tmp + flush + fsync + os.replace ile atomik kopya.

        Düz ``shutil.copyfile`` çökme anında yarım hedef bırakır; yedek
        adı zaten kesinleşmiş olduğundan yarım dosya "en yeni yedek"
        sayılır. Bu yardımcı hedefe ancak doğrulanmış (boyut + SHA-256)
        tam kopyayı bırakır; başarısızlıkta hedefe dokunulmaz.
        """
        tmp_path = dst.with_name(
            f"{dst.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        try:
            with open(src, "rb") as f_src, open(tmp_path, "wb") as f_dst:
                shutil.copyfileobj(f_src, f_dst)
                f_dst.flush()
                try:
                    os.fsync(f_dst.fileno())
                except OSError:
                    pass
            replace_ok = False
            last_exc: Optional[Exception] = None
            for attempt in range(5):
                try:
                    os.replace(tmp_path, dst)
                    replace_ok = True
                    break
                except OSError as exc:
                    last_exc = exc
                    time.sleep(0.04 * (attempt + 1))
            if not replace_ok:
                raise last_exc if last_exc is not None else OSError("os.replace başarısız")
            src_sig = RouteBackupManager._compute_file_signature(src)
            return bool(src_sig) and src_sig == RouteBackupManager._compute_file_signature(dst)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    @staticmethod
    def _compute_file_signature(file_path: Path) -> str:
        """Returns a deterministic size+mtime+content signature for change detection.

        F-1820: built-in hash() süreçler arası rastgeledir (bytes için
        PYTHONHASHSEED); imza SHA-256 ile deterministiktir.
        """
        try:
            raw = file_path.read_bytes()
            return f"{len(raw)}:{hashlib.sha256(raw).hexdigest()}"
        except OSError:
            return ""

    @staticmethod
    def _stat_key(file_path: Path) -> Optional[Tuple[int, int]]:
        """Cheap (size, mtime_ns) change key; None when the file cannot be stat'ed."""
        try:
            st = file_path.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    def _cached_file_signature(
        self, file_path: Path, region: str, stat_key: Optional[Tuple[int, int]]
    ) -> str:
        """SHA-256 signature, reused when (size, mtime_ns) is unchanged since the
        last computation — the 5 s daemon tick must not re-read + re-hash a
        multi-MB route file every tick (D10: 5 MB dosyada tick ~230-450 ms idi).

        İmza, yedek hash'inden (_last_hash_by_region) ayrı bir önbellekte
        (stat -> imza) tutulur: o hash yalnızca YEDEĞİ ALINMIŞ içeriğindir;
        değişmiş dosyayı gözlemleyip yedeklemeden dönen bir tick, bu
        önbelleğe eski yedek hash'ini düşürürse sonraki çağrı değişikliği
        hiç görmez (bayat imza -> sig-equality erken dönüşü).
        """
        if stat_key is not None:
            cached = self._sig_cache_by_region.get(region)
            if cached is not None and cached[0] == stat_key:
                return cached[1]
        sig = self._compute_file_signature(file_path)
        if sig and stat_key is not None:
            self._sig_cache_by_region[region] = (stat_key, sig)
        return sig

    def _backup_file_matches_signature(self, backup_path: Path, live_sig: str) -> bool:
        """Ground-truth check for the "unchanged" fast paths (W1).

        `backup_region` eskiden mevcut yedeği döndürürken yalnızca
        bellek-içi duruma (`_last_known_good_stat` / `_last_hash_by_region`)
        güveniyordu. O durum otomatik daemon thread'iyle paylaşılır; R11
        (fix/v2r-R11, e58fdc6) bir zehirlenme yolunu kapattı (doğrulama
        sonrası known_good yazımı + imza önbelleği karışması) ama kararın
        kendisi hâlâ yalnızca o duruma dayanıyordu — v3 canlı koşusunda
        aynı test aynı semptomla (yeni yedek yok, new_backup mevcut yedek)
        kırılmaya devam etti. Bu yüzden "değişmemiş" kararı artık diskteki
        son yedek dosyasının imzasıyla da doğrulanır; uymuyorsa yeni yedek
        oluşturulur. Yalnızca ham okuma + SHA-256 yapılır (JSON parse yok)
        ve yalnızca aralık dolmuşken çağrılır — 5 sn'lik daemon tick'inin
        hızlı yolu (D10-2) etkilenmez.
        """
        if not live_sig:
            return False
        disk_sig = self._compute_file_signature(backup_path)
        return bool(disk_sig) and disk_sig == live_sig

    @staticmethod
    def is_valid_route_json(file_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Checks whether `file_path` exists, is non-empty, and parses as a valid route JSON dict
        containing a `nodes` list (or dict structure).
        """
        if not file_path.exists() or not file_path.is_file():
            return False, None, "File does not exist"
        try:
            raw_text = file_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                return False, None, "Empty JSON file"
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                return False, None, "Root JSON element is not an object"
            nodes = parsed.get("nodes", [])
            # F-1833: docstring "list (or dict structure)" diyor; dict
            # nodes da geçerli rota yapısıdır.
            if not isinstance(nodes, (list, dict)):
                return False, None, "'nodes' field is not a list or dict"
            return True, parsed, "OK"
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return False, None, f"Corrupted JSON: {exc}"

    @staticmethod
    def _revalidate_before_restore(
        file_path: Path, attempts: int = 2, delay_sec: float = 0.05
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """W1-v5 (#1): restore çağrılmadan önce ikinci okunuş.

        `is_valid_route_json` tek okunuşla karar verir; eşzamanlı yazım
        (truncate anı), kilitli dosya ya da geçici I/O hatası da ilk
        okunuşta "Corrupted JSON" çıkar ve sağlıklı rota verisi eski
        yedekle ezilir. Bu yüzden restore öncesi kısa beklenip dosya
        yeniden okunur; stat (boyut, mtime_ns) okunuşlar arasında
        değişmişse yazım sürüyor demektir, bir tur daha beklenip okunur.
        İkinci okunuş sağlamsa restore YAPILMAZ, çağıran normal yola
        döner. Gerçekten bozuk dosyada her okunuş aynı hatayı verir —
        restore yolu aynen korunur, mutlu yol değişmez. Geçici
        `OSError` tek başına "bozuk" sayılmaz: yeniden deneme yapılır.
        """
        stat_before = RouteBackupManager._stat_key(file_path)
        last_reason = "Corrupted JSON: re-read attempts exhausted"
        for _ in range(attempts):
            time.sleep(delay_sec)
            valid, data, reason = RouteBackupManager.is_valid_route_json(file_path)
            if valid:
                return True, data, "OK"
            last_reason = reason
            stat_now = RouteBackupManager._stat_key(file_path)
            if (
                stat_before is not None
                and stat_now is not None
                and stat_now != stat_before
            ):
                # Dosya okunuşlar arasında değişti: yazım sürüyor
                # olabilir, bir tur daha bekle ve yeniden oku.
                stat_before = stat_now
                continue
            # Stat aynı ve hâlâ geçersiz: kalıcı bozukluk (veya kalıcı
            # okuma hatası) — restore yolu.
            return False, None, reason
        return False, None, last_reason

    def get_region_json_path(self, region: str) -> Path:
        safe_region = "".join(ch for ch in region.strip().lower() if ch.isalnum() or ch in ("_", "-"))
        exact = self.routes_dir / safe_region / "region_data.json"
        if exact.is_file():
            return exact
        # P3-8 (S-169): diskte "Altgard" gibi büyük harfli dizin varsa
        # küçük harfli arama sessizce boşa düşerdi — büyük/küçük harf
        # duyarsız eşleşme denenir.
        try:
            for entry in self.routes_dir.iterdir():
                if entry.is_dir() and entry.name.lower() == safe_region:
                    return entry / "region_data.json"
        except OSError:
            pass
        return exact

    def list_backups_for_region(self, region: str) -> List[Path]:
        """Returns all backup files for `region` sorted chronologically from oldest to newest.

        F-1819: glob deseni bölge öneki çakışmasına açıktı
        ("alt" deseni "altgard" yedeğini de tutar); dosya adı
        `region_data_<bölge>_YYYYMMDD_HHMMSS[_ffffff][_NNNN].json` kalıbıyla
        tam eşleşmek zorundadır. d10-W1: mikrosaniye eki yeni adlardır;
        saniye hassasiyetindeki eski adlar da hâlâ eşleşir.
        """
        safe_region = "".join(ch for ch in region.strip().lower() if ch.isalnum() or ch in ("_", "-"))
        pattern = f"region_data_{safe_region}_*.json"
        name_rx = re.compile(
            r"^region_data_" + re.escape(safe_region) + r"_\d{8}_\d{6}(?:_\d{6})?(?:_\d{4})?\.json$"
        )
        backups = [p for p in self.backups_dir.glob(pattern) if name_rx.match(p.name)]
        backups.sort(key=lambda p: (p.stem, p.stat().st_mtime if p.exists() else 0.0))
        return backups

    def list_valid_backups_for_region(self, region: str) -> List[Path]:
        """P2-6 (K-19): yalnızca geçerli JSON içeren yedekler, eskiden yeniye.

        Yarım/kesik yedek (çökme anında kalmış, JSON parse edilemeyen)
        "en yeni" sayılmaz: budama kararını bu liste üzerinden verir.
        Hızlı yollar yalnızca en yeni adayı doğrular (tüm listeyi parse
        etmek D10 hızlı yolunu bozar). Ham liste için
        :meth:`list_backups_for_region` korunur (sayım/test uyumluluğu).
        """
        return [
            p
            for p in self.list_backups_for_region(region)
            if self.is_valid_route_json(p)[0]
        ]

    def prune_old_backups(self, region: str) -> int:
        """Prunes old snapshots for `region` beyond the most recent `self.max_snapshots`.

        P2-6: geçersiz (yarım/bozuk) yedekler önce silinir ve kota
        sayılmaz — eskiden kota hesabı ham listeden yapılıyor, sağlıklı
        eski yedekler silinirken en yeni yarım yedek tutulabiliyordu.
        Kota aşılmadıysa doğrulama (JSON parse) yapılmaz — budama
        yalnızca yeni yedek yazılınca çalışır, daemon tick'ini
        yavaşlatmaz.
        """
        with self._lock:
            backups = self.list_backups_for_region(region)
            if len(backups) <= self.max_snapshots:
                return 0
            valid = self.list_valid_backups_for_region(region)
            valid_set = set(valid)
            removed_count = 0
            for bad_path in backups:
                if bad_path not in valid_set:
                    try:
                        bad_path.unlink(missing_ok=True)
                        removed_count += 1
                    except OSError:
                        continue
            excess = len(valid) - self.max_snapshots
            if excess > 0:
                for old_path in valid[:excess]:
                    try:
                        old_path.unlink(missing_ok=True)
                        removed_count += 1
                    except OSError:
                        continue
            return removed_count

    def backup_region(self, region: str, force: bool = False) -> Optional[Path]:
        """
        Creates a timestamped backup of `routes/<region>/region_data.json` inside `backups/`.
        If the live file is corrupted, refuses to overwrite healthy backups and triggers auto-restore.

        D10: eskiden her çağrıda (aralık içinde bile) önce tüm dosya JSON parse
        edilip SHA-256 çıkarılıyordu — 5 MB dosyada daemon tick'i ~230-450 ms.
        Artık dosya son doğrulanmış-okunuşundan beri (boyut, mtime_ns) değişmediyse
        parse + hash atlanır; değiştiyse eski tam yol (doğrulama + gerekirse
        auto-restore) aynen çalışır.
        """
        with self._lock:
            safe_region = "".join(ch for ch in region.strip().lower() if ch.isalnum() or ch in ("_", "-"))
            src_path = self.get_region_json_path(safe_region)
            if not src_path.exists():
                return None

            now = time.time()
            last_ts = self._last_backup_ts_by_region.get(safe_region, 0.0)
            stat_key = self._stat_key(src_path)

            # Hızlı yol: dosya son sağlıklı okunuşundan beri değişmemiş.
            if (
                not force
                and stat_key is not None
                and self._last_known_good_stat.get(safe_region) == stat_key
            ):
                if (now - last_ts) < self.backup_interval_sec:
                    return None
                existing = self.list_backups_for_region(safe_region)
                if existing:
                    # W1: bellek-içi "değişmemiş" kanıtı (known_good) daemon ile
                    # paylaşılan durumdur; diskteki son yedek de aynı içeriği
                    # tutmuyorsa bayat yedek döndürülmez, yeni yedek için düşülür.
                    # P2-6: yalnızca EN YENİ aday doğrulanır (tüm listeyi
                    # parse etmek D10 hızlı yolunu bozar) — yarım yedek
                    # "en yeni" sayılmaz, imza da tutmazsa yeni yedek düşülür.
                    live_sig = self._cached_file_signature(src_path, safe_region, stat_key)
                    if (
                        self.is_valid_route_json(existing[-1])[0]
                        and self._backup_file_matches_signature(existing[-1], live_sig)
                    ):
                        self._last_backup_ts_by_region[safe_region] = now
                        return existing[-1]
                    # Disk uymuyor: bellek-içi durum bayat; ts güncellenmez.
                # Yedek yok ama dosya sağlıklı: normal yola düş, yedek oluştur.

            valid, _, _ = self.is_valid_route_json(src_path)
            if not valid:
                # W1-v5 (#1): tek okunuş "bozuk" kararı vermez. İkinci
                # okunuş sağlamsa geçici okuma hatasıydı — restore ETME,
                # normal yola düş (yedek oluşturulur). İkinci okunuş da
                # geçersizse dosya gerçekten bozuktur, restore yolu aynen
                # çalışır.
                valid, _, _ = self._revalidate_before_restore(src_path)
                if not valid:
                    # Live file is corrupted; do not back up corrupted data—attempt restore instead
                    self.restore_latest_backup(safe_region)
                    return None
            # NOT: _last_known_good_stat burada güncellenmez. O sözlük
            # "yedeği alınmış içeriğin stat'ı"dır; sağlıklı okunuşu burada
            # işaretlemek, yedeklemeden dönen bir daemon tick'inin
            # değişmiş dosyayı "biliniyor" saymasına ve sonraki açık
            # backup_region çağrısının hızlı yoldan mevcut yedeği döndürüp
            # yeni yedek atlamasına yol açar. Yalnızca yedek oluşturulunca
            # (aşağıda, kopya sonrası) güncellenir.

            sig = self._cached_file_signature(src_path, safe_region, stat_key)

            if not force:
                if (now - last_ts) < self.backup_interval_sec:
                    return None
                existing = self.list_backups_for_region(safe_region)
                if (
                    sig
                    and self._last_hash_by_region.get(safe_region) == sig
                    and existing
                    # W1: bellek-içi hash tek başına yetmez; diskteki son yedek
                    # de aynı içeriği tutmuyorsa bayat yedek döndürülmez.
                    # P2-6: en yeni aday geçerli JSON olmalı — yarım yedek
                    # imza tutsa bile (tutamaz) "en yeni" sayılmaz.
                    and self.is_valid_route_json(existing[-1])[0]
                    and self._backup_file_matches_signature(existing[-1], sig)
                ):
                    # Dosya yedekle birebir aynı: yedek günceldir, hızlı
                    # yoldan geçebilmesi için bilinen-iyi stat güncellenir.
                    if stat_key is not None:
                        self._last_known_good_stat[safe_region] = stat_key
                    self._last_backup_ts_by_region[safe_region] = now
                    return existing[-1]

            now_dt = datetime.datetime.now()
            # d10-W1: ad artık mikrosaniye hassasiyetinde. Saniye
            # hassasiyetindeki ad, aynı saniyedeki ikinci yedeğin
            # birincinin üstüne yazılmasına yol açıyordu
            # (test_backup_changed_file_creates_new_backup: iki yedek
            # çağrısı, tek dosya). Aynı mikrosaniyeye düşme (saat
            # çözünürlüğü / süreçler arası) ihtimaline karşı diskteki
            # mevcut dosyalardan türetilen sayaç eki de korunur ve
            # üretilen ad diske karşı doğrulanır — iki ardışık yedek
            # çağrısı asla aynı dosya adına düşmez.
            timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S_%f")
            backup_filename = f"region_data_{safe_region}_{timestamp_str}.json"
            dst_path = self.backups_dir / backup_filename

            if dst_path.exists():
                # F-1831 (d10-W1): sayaç diskten türetilir — _seq_counter
                # bellek içidir ve restart'ta sıfırlanır. Aday ad da
                # diske karşı doğrulanır; aynı ada düşme ihtimali kalmaz.
                seq_rx = re.compile(
                    r"^region_data_" + re.escape(safe_region) + r"_"
                    + re.escape(timestamp_str) + r"(?:_(\d{4}))?\.json$"
                )
                disk_seq = 0
                for p in self.backups_dir.glob(
                    f"region_data_{safe_region}_{timestamp_str}*.json"
                ):
                    m = seq_rx.match(p.name)
                    if m and m.group(1):
                        disk_seq = max(disk_seq, int(m.group(1)))
                self._seq_counter = max(self._seq_counter, disk_seq)
                while True:
                    self._seq_counter += 1
                    backup_filename = (
                        f"region_data_{safe_region}_{timestamp_str}_{self._seq_counter:04d}.json"
                    )
                    dst_path = self.backups_dir / backup_filename
                    if not dst_path.exists():
                        break

            # P2-6 (K-05/K-19): yedek atomik yazılır (tmp + fsync +
            # os.replace) ve yazıldıktan sonra DOĞRULANIR (geçerli JSON
            # + canlı dosyayla aynı imza). Eskiden düz copyfile vardı:
            # çökme anında yarım yedek "en yeni" adıyla kalıyor,
            # üstelik durum (ts/hash/known_good) güncellenip yarım
            # yedek "alındı" sayılıyordu. Doğrulama tutmazsa yedek
            # silinir, hiçbir durum güncellenmez, None döner.
            if not self._atomic_copy_file(src_path, dst_path):
                logger.warning("Yedek kopyası yazılamadı/doğrulanamadı: %s", dst_path)
                return None
            valid_backup, _, _ = self.is_valid_route_json(dst_path)
            backup_sig = self._compute_file_signature(dst_path)
            if not valid_backup or not sig or backup_sig != sig:
                logger.warning(
                    "Yedek bütünlük doğrulaması tutmadı, yedek silindi: %s", dst_path
                )
                try:
                    dst_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None
            self._last_backup_ts_by_region[safe_region] = now
            self._last_hash_by_region[safe_region] = sig
            # Yedek alındı: bilinen-iyi stat artık bu dosyadır.
            if stat_key is not None:
                self._last_known_good_stat[safe_region] = stat_key
            self.prune_old_backups(safe_region)
            return dst_path

    def backup_all_regions(self, force: bool = False) -> Dict[str, Optional[Path]]:
        """Scans `routes/*/` and backs up every region that has a `region_data.json`."""
        results: Dict[str, Optional[Path]] = {}
        if not self.routes_dir.exists():
            return results
        for child in sorted(self.routes_dir.iterdir()):
            if child.is_dir():
                json_file = child / "region_data.json"
                if json_file.exists():
                    results[child.name] = self.backup_region(child.name, force=force)
        return results

    def restore_latest_backup(self, region: str) -> Tuple[bool, Optional[Path], str]:
        """
        Restores the most recent valid snapshot for `region` into `routes/<region>/region_data.json`.
        Iterates backwards from newest to oldest snapshot until a valid JSON snapshot is found.
        """
        with self._lock:
            safe_region = "".join(ch for ch in region.strip().lower() if ch.isalnum() or ch in ("_", "-"))
            backups = self.list_backups_for_region(safe_region)
            if not backups:
                return False, None, f"No backups available for region '{safe_region}'"

            dst_path = self.get_region_json_path(safe_region)
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            for candidate in reversed(backups):
                valid, _, _ = self.is_valid_route_json(candidate)
                if valid:
                    # W1-v5 (#2): restore artık atomik yazılır (.tmp ->
                    # flush -> fsync -> os.replace); yarım kalmış copy2
                    # canlı dosyayı kesik bırakabilirdi. Kalıp üretim
                    # yazıcılarıyla aynıdır (route_manager.save_region).
                    tmp_path = dst_path.with_suffix(".tmp")
                    try:
                        with open(candidate, "rb") as src, open(tmp_path, "wb") as tmp:
                            shutil.copyfileobj(src, tmp)
                            tmp.flush()
                            try:
                                os.fsync(tmp.fileno())
                            except OSError:
                                pass
                        # Windows dosya kilitleme ve erişim yarışlarına
                        # karşı güvenli atomik değiştirme döngüsü.
                        replace_ok = False
                        last_replace_exc: Optional[Exception] = None
                        for attempt in range(5):
                            try:
                                os.replace(tmp_path, dst_path)
                                replace_ok = True
                                break
                            except OSError as r_exc:
                                last_replace_exc = r_exc
                                time.sleep(0.04 * (attempt + 1))
                        if not replace_ok:
                            if last_replace_exc is not None:
                                raise last_replace_exc
                    finally:
                        # os.replace başardıysa tmp zaten yok; başarısız
                        # denemede artık tmp temizlenir.
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    self._last_hash_by_region[safe_region] = self._compute_file_signature(dst_path)
                    # D10: geri yüklenen dosya sağlıklı — sonraki tick'ler hızlı yoldan geçsin.
                    restored_stat = self._stat_key(dst_path)
                    if restored_stat is not None:
                        self._last_known_good_stat[safe_region] = restored_stat
                    return True, candidate, f"Restored '{safe_region}' from {candidate.name}"

            return False, None, f"All {len(backups)} backup snapshots for '{safe_region}' were invalid"

    def ensure_region_integrity_or_restore(self, region: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Verifies `routes/<region>/region_data.json`. If corrupted, automatically restores from the
        latest valid backup and returns the loaded dictionary.
        """
        src_path = self.get_region_json_path(region)
        valid, data, reason = self.is_valid_route_json(src_path)
        if valid and data is not None:
            # NOT: sağlıklı okunuş _last_known_good_stat'a yazılmaz — o
            # sözlük "yedeği alınmış içeriğin stat'ı"dır. Sağlıklı ama
            # henüz yedeklenmemiş dosyayı "biliniyor" işaretlemek,
            # sonraki backup_region çağrısının hızlı yoldan mevcut yedeği
            # döndürüp değişikliği yedeklemeden atlamasına yol açar.
            # Bir sonraki backup tick'i dosyayı bir kez tam doğrular ve
            # yedek oluşturulunca stat güncellenir.
            return True, data, "Healthy"

        # W1-v5 (#1): restore öncesi ikinci okunuş — geçici okuma
        # hatasında sağlıklı dosya eski yedekle ezilmesin. İkinci
        # okunuş sağlamsa restore yok, dosya sağlıklı sayılır.
        valid, data, reason = self._revalidate_before_restore(src_path)
        if valid and data is not None:
            return True, data, "Healthy"

        restored, snap_path, msg = self.restore_latest_backup(region)
        if restored:
            valid_after, data_after, _ = self.is_valid_route_json(src_path)
            if valid_after and data_after is not None:
                return True, data_after, f"Auto-recovered from corruption ({reason}) via {snap_path.name if snap_path else 'backup'}"

        fallback_data: Dict[str, Any] = {"region": region, "nodes": [], "connections": []}
        return False, fallback_data, f"Integrity check failed ({reason}) and restore failed ({msg})"

    def start_periodic_backup_daemon(self) -> None:
        """Starts a background daemon thread that checks and backs up routes every 10 minutes (600s)."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._daemon_loop,
                name="RouteBackupManagerDaemon",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_periodic_backup_daemon(self, timeout: float = 2.0) -> None:
        """Stops the background backup daemon thread cleanly."""
        self._stop_event.set()
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
            if worker.is_alive():
                # F-1827: join zaman aşımına uğradıysa referansı silme;
                # yoksa start() zombi thread varken ikinci bir daemon
                # başlatır. Referans durana kadar saklanır.
                logger.warning(
                    "RouteBackupManager daemon %.1fs içinde durmadı; referans saklanıyor",
                    timeout,
                )
                return
        self._worker_thread = None

    def _daemon_loop(self) -> None:
        # Take an initial snapshot on startup if none exist yet
        self.backup_all_regions(force=False)
        check_step = min(5.0, self.backup_interval_sec)
        while not self._stop_event.wait(timeout=check_step):
            self.backup_all_regions(force=False)


# ============================================================================
# ITEM 47: Çoklu Monitör DPI ve Ölçekleme Telafisi (DPIScalingSynchronizer)
# ============================================================================

class DPIScalingSynchronizer:
    """
    Item 47 — Windows Multi-Monitor DPI & Scaling (100% / 125% / 150% / 175% / 200%) Synchronizer.
    Enforces `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)` (-4) with
    automatic fallbacks to `shcore.SetProcessDpiAwareness(2)` and `user32.SetProcessDPIAware()`,
    queries live system/window DPI, and provides exact coordinate conversion between logical Qt
    coordinates and physical `mss` / `SendInput` screen pixels.
    """

    STANDARD_DPI: int = 96
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: int = -4
    PROCESS_PER_MONITOR_DPI_AWARE: int = 2

    def __init__(self, override_scale_factor: Optional[float] = None) -> None:
        self._awareness_level: str = "uninitialized"
        self._system_dpi: int = self.STANDARD_DPI
        self._override_scale: Optional[float] = (
            float(override_scale_factor) if override_scale_factor is not None and override_scale_factor > 0 else None
        )
        # F-1834: ölçek her çağrıda Win32 sorgusuyla ölçülmez; kısa
        # ömürlü önbellek (TTL 2 sn) kullanılır.
        self._scale_cache: Optional[float] = None
        self._scale_cache_ts: float = 0.0
        self.enforce_per_monitor_dpi_awareness()
        self.refresh_dpi_metrics()

    def enforce_per_monitor_dpi_awareness(self) -> str:
        """
        Applies `SetProcessDpiAwarenessContext(-4)` (`DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`),
        falling back to `shcore.SetProcessDpiAwareness(2)` and `user32.SetProcessDPIAware()`.
        """
        if os.name != "nt":
            self._awareness_level = "non_windows_native_1x"
            return self._awareness_level

        user32 = getattr(ctypes, "windll", None)
        if user32 is None:
            self._awareness_level = "fallback_default"
            return self._awareness_level

        # 1. Try SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4)
        try:
            set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
            set_context.argtypes = [ctypes.c_ssize_t]
            set_context.restype = wintypes.BOOL
            res = set_context(ctypes.c_ssize_t(self.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
            if res:
                self._awareness_level = "PER_MONITOR_AWARE_V2"
                return self._awareness_level
            # Even if already set by Qt/Python host, check last error (ACCESS_DENIED = already set)
            err = ctypes.GetLastError()
            if err == 5:  # ERROR_ACCESS_DENIED means DPI awareness was already locked in on process start
                self._awareness_level = "PER_MONITOR_AWARE_V2_EXISTING"
                return self._awareness_level
        except (AttributeError, OSError) as exc:
            self._awareness_level = f"V2_UNSUPPORTED:{type(exc).__name__}"

        # 2. Fallback: shcore.SetProcessDpiAwareness(2)
        try:
            res_shcore = ctypes.windll.shcore.SetProcessDpiAwareness(self.PROCESS_PER_MONITOR_DPI_AWARE)
            if res_shcore in (0, -2147024891):  # S_OK or E_ACCESSDENIED (already set)
                self._awareness_level = "SHCORE_PER_MONITOR_AWARE"
                return self._awareness_level
        except (AttributeError, OSError) as exc:
            self._awareness_level = f"SHCORE_UNSUPPORTED:{type(exc).__name__}"

        # 3. Fallback: user32.SetProcessDPIAware()
        try:
            res_legacy = ctypes.windll.user32.SetProcessDPIAware()
            if res_legacy:
                self._awareness_level = "USER32_SYSTEM_DPI_AWARE"
                return self._awareness_level
        except (AttributeError, OSError) as exc:
            self._awareness_level = f"LEGACY_UNSUPPORTED:{type(exc).__name__}"

        self._awareness_level = "UNAWARE_FALLBACK"
        return self._awareness_level

    def refresh_dpi_metrics(self, hwnd: Optional[int] = None) -> int:
        """Queries `GetDpiForWindow(hwnd)` or `GetDpiForSystem()` to update the current DPI."""
        if os.name != "nt":
            self._system_dpi = self.STANDARD_DPI
            return self._system_dpi

        try:
            if hwnd:
                get_dpi_win = ctypes.windll.user32.GetDpiForWindow
                get_dpi_win.argtypes = [wintypes.HWND]
                get_dpi_win.restype = wintypes.UINT
                win_dpi = int(get_dpi_win(wintypes.HWND(hwnd)))
                if 48 <= win_dpi <= 576:
                    self._system_dpi = win_dpi
                    return self._system_dpi
        except (AttributeError, OSError):
            hwnd = None

        try:
            get_dpi_sys = ctypes.windll.user32.GetDpiForSystem
            get_dpi_sys.argtypes = []
            get_dpi_sys.restype = wintypes.UINT
            sys_dpi = int(get_dpi_sys())
            if 48 <= sys_dpi <= 576:
                self._system_dpi = sys_dpi
                return self._system_dpi
        except (AttributeError, OSError):
            sys_dpi = 0

        # Fallback via GDI GetDeviceCaps(LOGPIXELSX = 88)
        # F-1830: GetDC/GetDeviceCaps/ReleaseDC için argtypes/restype
        # tanımlanır; tanımsızken 64-bit handle kesilip yanlış DPI okunur.
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            user32.GetDC.argtypes = [wintypes.HWND]
            user32.GetDC.restype = wintypes.HANDLE
            gdi32.GetDeviceCaps.argtypes = [wintypes.HANDLE, ctypes.c_int]
            gdi32.GetDeviceCaps.restype = ctypes.c_int
            user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HANDLE]
            user32.ReleaseDC.restype = ctypes.c_int
            hdc = user32.GetDC(0)
            if hdc:
                try:
                    dpi_x = int(gdi32.GetDeviceCaps(hdc, 88))
                finally:
                    user32.ReleaseDC(0, hdc)
                if 48 <= dpi_x <= 576:
                    self._system_dpi = dpi_x
                    return self._system_dpi
        except (AttributeError, OSError):
            pass

        self._system_dpi = self.STANDARD_DPI
        return self._system_dpi

    def get_scale_factor(self, hwnd: Optional[int] = None) -> float:
        """Returns the active scaling factor (e.g. 1.0 for 100%, 1.25 for 125%, 1.5 for 150%).

        F-1834: Win32 sorgusu her çağrıda tekrarlanmaz; 2 sn TTL'li
        önbellek kullanılır (monitör değişiminde en geç 2 sn içinde
        tazelenir).
        """
        if self._override_scale is not None:
            return self._override_scale
        now = time.monotonic()
        if hwnd is None and self._scale_cache is not None and (now - self._scale_cache_ts) < 2.0:
            return self._scale_cache
        dpi = self.refresh_dpi_metrics(hwnd=hwnd)
        scale = round(float(dpi) / float(self.STANDARD_DPI), 4)
        if hwnd is None:
            self._scale_cache = scale
            self._scale_cache_ts = now
        return scale

    def set_override_scale_factor(self, scale_factor: Optional[float]) -> None:
        """Allows deterministic testing or manual user override for 125% (1.25) / 150% (1.50) scaling."""
        if scale_factor is None:
            self._override_scale = None
        else:
            self._override_scale = max(0.5, min(4.0, float(scale_factor)))

    def qt_to_physical_px(
        self,
        x: float | int,
        y: float | int,
        hwnd: Optional[int] = None,
        monitor_origin_px: Tuple[int, int] = (0, 0),
    ) -> Tuple[int, int]:
        """
        Converts logical PyQt6 widget/screen coordinates `(x, y)` into physical `mss` / `SendInput`
        pixel coordinates, compensating for 125% / 150% Windows display scaling and multi-monitor origin.
        """
        scale = self.get_scale_factor(hwnd=hwnd)
        ox, oy = int(monitor_origin_px[0]), int(monitor_origin_px[1])
        phys_x = ox + int(round(float(x) * scale))
        phys_y = oy + int(round(float(y) * scale))
        return phys_x, phys_y

    def physical_to_qt_px(
        self,
        x: float | int,
        y: float | int,
        hwnd: Optional[int] = None,
        monitor_origin_px: Tuple[int, int] = (0, 0),
    ) -> Tuple[int, int]:
        """
        Converts physical `mss` capture coordinates `(x, y)` into logical PyQt6 overlay coordinates
        so HUD markers align 1:1 on 125% or 150% scaled displays.
        """
        scale = self.get_scale_factor(hwnd=hwnd)
        ox, oy = int(monitor_origin_px[0]), int(monitor_origin_px[1])
        rel_x = float(x) - float(ox)
        rel_y = float(y) - float(oy)
        qt_x = int(round(rel_x / scale))
        qt_y = int(round(rel_y / scale))
        return qt_x, qt_y

    def scale_region_bbox_to_physical(
        self,
        bbox: Dict[str, int],
        hwnd: Optional[int] = None,
        monitor_origin_px: Tuple[int, int] = (0, 0),
    ) -> Dict[str, int]:
        """Scales a logical `{left, top, width, height}` region into physical `mss` capture coordinates.

        F-1834: çoklu monitörde bbox dönüşümü monitör kökenini
        (monitor_origin_px) de uygular; aksi halde ikinci monitördeki
        bölge yanlış konuma kayar.
        """
        scale = self.get_scale_factor(hwnd=hwnd)
        ox, oy = int(monitor_origin_px[0]), int(monitor_origin_px[1])
        return {
            "left": ox + int(round(bbox.get("left", 0) * scale)),
            "top": oy + int(round(bbox.get("top", 0) * scale)),
            "width": max(1, int(round(bbox.get("width", 1) * scale))),
            "height": max(1, int(round(bbox.get("height", 1) * scale))),
        }

    def get_diagnostics(self, hwnd: Optional[int] = None) -> Dict[str, Any]:
        # F-1834: DPI ölçekten türetilip uydurulmaz; ölçülen sistem
        # DPI'si raporlanır.
        scale = self.get_scale_factor(hwnd=hwnd)
        return {
            "awareness_level": self._awareness_level,
            "dpi": int(self._system_dpi),
            "scale_factor": scale,
            "scale_percent": int(round(scale * 100)),
        }

    def get_dpi_focus_guard_diagnostic_snapshot(self) -> dict:
        # F-1826: durum koşulsuz "ok" değil; DPI farkındalığı gerçekten
        # kurulamadıysa degraded raporlanır.
        ready_levels = {
            "PER_MONITOR_AWARE_V2",
            "PER_MONITOR_AWARE_V2_EXISTING",
            "SHCORE_PER_MONITOR_AWARE",
            "USER32_SYSTEM_DPI_AWARE",
            "non_windows_native_1x",
        }
        ready = self._awareness_level in ready_levels
        return {
            "module": "dpi_focus_guard",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "awareness_level": self._awareness_level,
            "dpi": int(self._system_dpi),
        }

    def get_dpi_focus_guard_capabilities(self) -> list:
        return ["dpi_focus_guard", "diagnostic_snapshot", "version_info"]


# ============================================================================
# ITEM 48: Oyun Penceresi Odak Koruması (ForegroundFocusGuard)
# ============================================================================

class ForegroundFocusGuard:
    """
    Item 48 — Ultra-fast (< 0.1ms) Win32 Foreground Window Focus Guard.
    Queries `GetForegroundWindow()`, `GetWindowTextW()`, and `GetWindowThreadProcessId()` to verify
    that the target game window is actively focused before any virtual keystroke (`F`, `M`, `W/A/S/D`,
    `Space`, `Tab`) or mouse click is dispatched. Immediately suppresses inputs if the user Alt-Tabs
    to Discord, Chrome, Explorer, or if a Windows notification steals focus.
    """

    # F-1828: "game"/"rise"/"client" gibi jenerik kelimeler her oyunu
    # (ve oyun dışı pencereyi) geçirir; yalnızca Aion'a özgü
    # işaretler kalır. "unrealwindow" Aion istemcisinin pencere
    # sınıfıdır ve tutulur.
    DEFAULT_ALLOWED_TITLE_KEYWORDS = (
        "aion",
        "aion2",
        "ncsoft",
        "purple",
        "unrealwindow",
        "아이온",
    )

    DEFAULT_BLOCKED_APPS = (
        "discord",
        "chrome",
        "firefox",
        "msedge",
        "opera",
        "brave",
        "explorer",
        "notepad",
        "code",
        "visual studio",
        "powershell",
        "cmd.exe",
        "windows powershell",
        "task manager",
        "görev yöneticisi",
        "telegram",
        "whatsapp",
        "slack",
        "spotify",
        "program manager",
        "desktop",
        "masaüstü",
        "steam",
        "taskview",
        "cortana",
    )

    def __init__(
        self,
        allowed_title_keywords: Optional[List[str]] = None,
        allowed_pids: Optional[List[int]] = None,
        strict_mode: bool = True,
    ) -> None:
        self.allowed_title_keywords: List[str] = [
            kw.strip().lower()
            for kw in (allowed_title_keywords if allowed_title_keywords is not None else self.DEFAULT_ALLOWED_TITLE_KEYWORDS)
            if kw.strip()
        ]
        self.allowed_pids: set[int] = set(int(p) for p in (allowed_pids or []))
        self.strict_mode: bool = bool(strict_mode)
        self._simulated_foreground_title: Optional[str] = None
        self._simulated_foreground_pid: Optional[int] = None
        self._suppressed_keystroke_count: int = 0
        self._last_focus_loss_reason: str = ""
        self._last_check_latency_ms: float = 0.0
        self._verified_game_pids: set[int] = set()
        self._own_pid: int = os.getpid()

        # Monitoring thread state
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()
        self._on_focus_lost_cb: Optional[Callable[[str], None]] = None
        self._on_focus_gained_cb: Optional[Callable[[str], None]] = None
        self._monitor_interval_sec: float = 0.05
        self._monitor_allow_gatherion: bool = True

        # Pre-bind Win32 ctypes signatures for < 0.05ms execution
        self._has_win32 = os.name == "nt" and hasattr(ctypes, "windll")
        if self._has_win32:
            self._GetForegroundWindow = ctypes.windll.user32.GetForegroundWindow
            self._GetForegroundWindow.argtypes = []
            self._GetForegroundWindow.restype = wintypes.HWND

            self._GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
            self._GetWindowTextLengthW.argtypes = [wintypes.HWND]
            self._GetWindowTextLengthW.restype = ctypes.c_int

            self._GetWindowTextW = ctypes.windll.user32.GetWindowTextW
            self._GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            self._GetWindowTextW.restype = ctypes.c_int

            self._GetWindowThreadProcessId = ctypes.windll.user32.GetWindowThreadProcessId
            self._GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            self._GetWindowThreadProcessId.restype = wintypes.DWORD

    def register_game_pids(self, pids: Sequence[int]) -> None:
        """Dynamically registers active game PIDs (e.g. from find_aion_game_processes)."""
        for p in pids:
            try:
                val = int(p)
                if val > 0:
                    self.allowed_pids.add(val)
                    self._verified_game_pids.add(val)
            except (ValueError, TypeError):
                pass

    def set_simulated_foreground_window(self, title: Optional[str], pid: Optional[int] = None) -> None:
        """Used by unit tests and dry-run simulator to emulate Alt-Tab focus transitions.

        F-0429: test kancası kalıcı ve sessizdi; kurulduğunda uyarı
        loglanır ve clear_simulated_foreground_window() ile kapatılır.
        """
        logger.warning(
            "FocusGuard: simulated foreground window set (test hook only): title=%r pid=%s",
            title,
            pid,
        )
        self._simulated_foreground_title = title
        self._simulated_foreground_pid = pid

    def clear_simulated_foreground_window(self) -> None:
        """Clears a previously set simulated foreground window; subsequent
        checks read the real Win32 foreground window again."""
        self._simulated_foreground_title = None
        self._simulated_foreground_pid = None

    def query_foreground_window_info(self) -> Tuple[int, int, str]:
        """
        Returns `(hwnd, pid, window_title)` of the active foreground window in `< 0.1ms`.
        """
        t0 = time.perf_counter()
        if self._simulated_foreground_title is not None:
            pid = self._simulated_foreground_pid if self._simulated_foreground_pid is not None else 9999
            self._last_check_latency_ms = (time.perf_counter() - t0) * 1000.0
            return 1, pid, self._simulated_foreground_title

        if not self._has_win32:
            self._last_check_latency_ms = (time.perf_counter() - t0) * 1000.0
            return 0, 0, ""

        hwnd = int(self._GetForegroundWindow() or 0)
        if not hwnd and self._has_win32:
            try:
                desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0100)
                if desk:
                    ctypes.windll.user32.SetThreadDesktop(desk)
                    hwnd = int(self._GetForegroundWindow() or 0)
                    ctypes.windll.user32.CloseDesktop(desk)
            except Exception:
                pass

        if not hwnd:
            self._last_check_latency_ms = (time.perf_counter() - t0) * 1000.0
            return 0, 0, ""

        pid_dword = wintypes.DWORD(0)
        self._GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid_dword))
        pid = int(pid_dword.value)

        length = int(self._GetWindowTextLengthW(wintypes.HWND(hwnd)))
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            self._GetWindowTextW(wintypes.HWND(hwnd), buf, length + 2)
            title = buf.value.strip()
        else:
            title = ""

        self._last_check_latency_ms = (time.perf_counter() - t0) * 1000.0
        return hwnd, pid, title

    def is_game_window_focused(self, allow_gatherion_ui: bool = True) -> Tuple[bool, str]:
        """
        Returns `(is_focused, active_window_description)` in `< 0.1ms`.
        Immediately returns `False` if the foreground window is empty, matches a blocked non-game app
        (Discord, Chrome, Explorer, etc.), or does not match the allowed game title/PID criteria.
        If `allow_gatherion_ui=True`, Gatherion's own window is treated as allowed so clicking HUD
        buttons does not falsely trigger an Alt-Tab pause.
        """
        hwnd, pid, title = self.query_foreground_window_info()
        if (hwnd == 0 or not title.strip()) and self._simulated_foreground_title is None:
            self._last_focus_loss_reason = "No active foreground window (Desktop/LockScreen)"
            return False, self._last_focus_loss_reason

        title_lower = title.lower().strip()
        if title_lower in ("program manager", "workerw", "progman", "desktop", "masaüstü", ""):
            self._last_focus_loss_reason = f"Masaüstü odakta: '{title or '<Desktop>'}' (PID={pid})"
            return False, self._last_focus_loss_reason

        # 0. Gatherion internal UI exemption (Overlay HUD, F9 Menu, Settings)
        if allow_gatherion_ui:
            if pid == self._own_pid or "gatherion" in title_lower:
                return True, f"Gatherion UI: '{title}' (PID={pid})"

        # 1. Immediate blocklist check (Discord, Chrome, Explorer, IDE, etc.)
        # F-0251: blocklist PID eşleşmesinden ÖNCE gelir; aksi halde
        # doğrulanmış bir PID, pencere Discord/Chrome olsa bile kalıcı
        # olarak oyun sayılır.
        for blocked in self.DEFAULT_BLOCKED_APPS:
            if blocked in title_lower:
                self._last_focus_loss_reason = f"Focus stolen by non-game window: '{title}' (PID={pid})"
                return False, self._last_focus_loss_reason

        # 2. Explicit PID match (if registered or verified)
        if (self.allowed_pids and pid in self.allowed_pids) or (pid in self._verified_game_pids):
            return True, f"{title} (PID={pid})"

        # 3. Allowed game title keyword match
        for kw in self.allowed_title_keywords:
            if kw in title_lower:
                if pid > 0:
                    self._verified_game_pids.add(pid)
                return True, f"{title} (PID={pid})"

        # 4. Check process executable name if available (e.g. AION2.exe)
        if pid > 0 and pid != self._own_pid and self._simulated_foreground_title is None:
            try:
                import psutil
                pname = psutil.Process(pid).name().lower()
                if "aion" in pname or "ncsoft" in pname:
                    self._verified_game_pids.add(pid)
                    return True, f"{title} (PID={pid})"
            except Exception:
                pass

        if not self.strict_mode and title:
            return True, f"{title} (PID={pid})"

        self._last_focus_loss_reason = f"Unrecognized foreground window: '{title or '<Untitled>'}' (PID={pid})"
        return False, self._last_focus_loss_reason

    def start_monitoring(
        self,
        on_focus_lost: Optional[Callable[[str], None]] = None,
        on_focus_gained: Optional[Callable[[str], None]] = None,
        check_interval_sec: float = 0.05,
        allow_gatherion_ui: bool = True,
    ) -> None:
        """
        Starts a background daemon thread that checks GetForegroundWindow every 50ms.
        Immediately fires on_focus_lost(reason) when user Alt-Tabs away from the game.
        """
        self.stop_monitoring()
        self._on_focus_lost_cb = on_focus_lost
        self._on_focus_gained_cb = on_focus_gained
        self._monitor_interval_sec = max(0.01, float(check_interval_sec))
        self._monitor_allow_gatherion = bool(allow_gatherion_ui)
        self._monitor_stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="FocusGuardMonitorThread",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stops the focus monitoring thread."""
        self._monitor_stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            # F-1836: 0.2 sn join çok kısaydı; 1.0 sn beklenir ve hâlâ
            # canlıysa referans saklanır (zombi thread + çift başlatma
            # engellenir).
            self._monitor_thread.join(timeout=1.0)
            if self._monitor_thread.is_alive():
                logger.warning("FocusGuard monitor thread did not stop within 1.0s; keeping reference")
                return
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        last_focused = True
        while not self._monitor_stop_event.is_set():
            try:
                focused, info = self.is_game_window_focused(allow_gatherion_ui=self._monitor_allow_gatherion)
                if last_focused and not focused:
                    last_focused = False
                    if self._on_focus_lost_cb:
                        try:
                            self._on_focus_lost_cb(info)
                        except Exception as cb_exc:
                            logger.error("Error in on_focus_lost callback: %s", cb_exc)
                elif not last_focused and focused:
                    last_focused = True
                    if self._on_focus_gained_cb:
                        try:
                            self._on_focus_gained_cb(info)
                        except Exception as cb_exc:
                            logger.error("Error in on_focus_gained callback: %s", cb_exc)
            except Exception as exc:
                # F-1836: izleme döngüsündeki hata sessiz yutulmaz.
                logger.debug("FocusGuard monitor loop error: %s", exc)
            self._monitor_stop_event.wait(self._monitor_interval_sec)

    def guard_keystroke(
        self,
        key_name: str,
        dispatch_fn: Optional[Callable[[str], Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Wraps a virtual keystroke (`F`, `M`, `W`, `A`, `S`, `D`, `Space`, `Tab`).
        If the game window is NOT focused, suppresses the keystroke and increments the suppression counter.
        """
        focused, info = self.is_game_window_focused()
        if not focused:
            self._suppressed_keystroke_count += 1
            return False, f"SUPPRESSED key '{key_name}' -> {info}"

        if dispatch_fn is None:
            # F-1818/F-0252: gönderim fonksiyonu yokken "DISPATCHED"
            # denmez — tuşa yalnızca İZİN verildi, guard göndermedi.
            return True, f"ALLOWED key '{key_name}' (no dispatch_fn; not sent by guard) -> {info}"
        try:
            dispatch_fn(key_name)
        except Exception as exc:
            # F-1818: gönderim hatası korumasızdı; başarısızlık
            # bildirilir, istisna yutulmaz.
            return False, f"DISPATCH_FAILED key '{key_name}': {exc} -> {info}"
        return True, f"DISPATCHED key '{key_name}' -> {info}"

    def attach_to_input_driver(self, driver: Any, method_names: Sequence[str] = ("press_key",)) -> bool:
        """
        F-0083 groundwork — guard_keystroke'un üretimde çağıranı yok;
        core/input_driver.py'da guard kancası yok (InputController
        focus_monitor/focus_guard özniteliğini okumuyor; main.py:513-514
        focus_monitor'ı input_ctrl'e koyuyor ama hiçbir şey okumuyor).

        Bu yardımcı guard'ı sürücü nesnesine tek satırla bağlar:
        ``driver.focus_guard = self`` kurulur ve varsa ``press_key``
        gibi tuş gönderim metodu guard_keystroke ile sarılır — oyun
        odakta değilken tuş gönderilmez, press_key'in "gönderilmedi"
        karşılığı olan 0.0 döner (input_driver.press_key'in odak
        reddi yoluyla aynı değer). Guard çalışamaz durumdaysa
        (Win32 yok ve simülasyon kurulu değil — örn. Linux) sarmalayıcı
        tuşu engellemez, olduğu gibi geçirir; çalışamayan guard
        girdiyi kilitlememelidir. R/T kuralı etkilenmez: guard yalnızca
        tuşu geçirir/engeller, fare tıklamasını T/R tuşuna çevirmez.

        Kalan adım dilim dışıdır (main.py'da attach çağrısı +
        gerekirse input_driver'da kanca) — o yüzden F-0083 bu dilimde
        blocked — cross-slice kalır. İdempotenttir: ikinci çağrı
        çift sarmalamaz. En az bir gönderim metodu sarıldıysa True,
        yoksa (yalnızca focus_guard özniteliği kurulduysa) False döner.
        """
        if driver is None:
            return False
        try:
            driver.focus_guard = self
        except Exception as exc:
            logger.warning("FocusGuard: driver.focus_guard kurulamadı: %s", exc)

        wrapped_any = False
        for name in method_names:
            try:
                original = getattr(driver, name, None)
            except Exception:
                continue
            if not callable(original) or getattr(original, "_focus_guard_wrapped", False):
                continue
            guard = self

            def _guarded(*args: Any, _orig: Any = original, _guard: Any = guard, **kwargs: Any) -> Any:
                key_name = str(args[0]) if args else ""
                if not _guard._focus_guard_operational():
                    return _orig(*args, **kwargs)
                try:
                    allowed, _msg = _guard.guard_keystroke(key_name, dispatch_fn=None)
                except Exception as exc:
                    logger.warning("FocusGuard: odak denetimi hatası, tuş engellendi (%s): %s", key_name, exc)
                    return 0.0
                if not allowed:
                    return 0.0
                return _orig(*args, **kwargs)

            _guarded._focus_guard_wrapped = True  # type: ignore[attr-defined]
            _guarded._focus_guard_original = original  # type: ignore[attr-defined]
            try:
                setattr(driver, name, _guarded)
                wrapped_any = True
            except Exception as exc:
                logger.warning("FocusGuard: driver.%s sarılamadı: %s", name, exc)
        return wrapped_any

    @property
    def suppressed_keystroke_count(self) -> int:
        return self._suppressed_keystroke_count

    @property
    def last_check_latency_ms(self) -> float:
        return self._last_check_latency_ms

    def _focus_guard_operational(self) -> bool:
        """Gerçek çalışabilirlik: Win32 var ya da test simülasyonu kurulu."""
        return bool(self._has_win32 or self._simulated_foreground_title is not None)

    def get_focus_guard_health_summary(self) -> dict:
        # F-1826: sağlık özeti koşulsuz True değil; Win32 yoksa ve
        # simülasyon da kurulu değilse guard çalışamaz.
        ready = self._focus_guard_operational()
        return {
            "focus_guard_ready": ready,
            "has_win32": bool(self._has_win32),
            "simulated": self._simulated_foreground_title is not None,
            "suppressed_keystrokes": self._suppressed_keystroke_count,
        }

    def get_dpi_focus_guard_version_info(self) -> dict:
        return {"module": "dpi_focus_guard", "version": __version__, "ready": self._focus_guard_operational()}

    def get_dpi_focus_guard_diagnostic_snapshot(self) -> dict:
        ready = self._focus_guard_operational()
        return {
            "module": "dpi_focus_guard",
            "status": "ok" if ready else "unavailable",
            "ready": ready,
            "has_win32": bool(self._has_win32),
        }

    def get_dpi_focus_guard_capabilities(self) -> list:
        return ["dpi_focus_guard", "diagnostic_snapshot", "version_info"]



# ============================================================================
# ITEM 49: Kendi Kendini İyileştiren Şablon Eşikleri (SelfHealingTemplateTuner)
# ============================================================================

class SelfHealingTemplateTuner:
    """
    Item 49 — Self-Healing 80x80 Node Template Refresher & Local Threshold Tuner.
    Tracks template matching confidence scores per `node_id`. When a node matches with a borderline
    confidence score in `[borderline_min, borderline_max]` (default `0.65` to `0.72`) across
    `consecutive_trigger_count` (default `3`) consecutive visits—caused by in-game time-of-day
    lighting shifts or weather transitions—it automatically refreshes the node's saved `80x80`
    reference patch (saving a backup of the previous patch) and tunes the node's local threshold.
    """

    PATCH_SIZE: Tuple[int, int] = (80, 80)

    def __init__(
        self,
        borderline_min: float = 0.65,
        borderline_max: float = 0.72,
        consecutive_trigger_count: int = 3,
        default_threshold: float = 0.70,
    ) -> None:
        self.borderline_min: float = float(borderline_min)
        self.borderline_max: float = float(borderline_max)
        self.consecutive_trigger_count: int = max(1, int(consecutive_trigger_count))
        self.default_threshold: float = float(default_threshold)

        self._score_history: Dict[str, List[float]] = {}
        self._borderline_streak: Dict[str, int] = {}
        self._local_thresholds: Dict[str, float] = {}
        self._refresh_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

    def get_node_threshold(self, node_id: str | int) -> float:
        key = str(node_id)
        with self._lock:
            return self._local_thresholds.get(key, self.default_threshold)

    @staticmethod
    def _normalize_to_80x80(patch_bgr: np.ndarray) -> np.ndarray:
        """Ensures the patch is a valid 3-channel 80x80 uint8 BGR image."""
        if patch_bgr is None or patch_bgr.size == 0:
            raise ValueError("patch_bgr cannot be empty")
        if patch_bgr.ndim == 2:
            patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)
        elif patch_bgr.ndim == 3 and patch_bgr.shape[2] == 4:
            patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_BGRA2BGR)

        h, w = patch_bgr.shape[:2]
        if (w, h) != (80, 80):
            patch_bgr = cv2.resize(patch_bgr, (80, 80), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(patch_bgr, dtype=np.uint8)

    def record_visit_score(
        self,
        node_id: str | int,
        score: float,
        current_patch_bgr: Optional[np.ndarray] = None,
        patch_save_path: Optional[str | Path] = None,
        blend_alpha: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Records a template match score for `node_id`.
        If `score` falls within `[0.65, 0.72]` for `3` consecutive visits and `current_patch_bgr` is
        provided, refreshes and saves the `80x80` template patch on disk and tunes the node's local threshold.
        """
        key = str(node_id)
        val = float(score)

        with self._lock:
            hist = self._score_history.setdefault(key, [])
            hist.append(val)
            if len(hist) > 25:
                del hist[:-25]

            is_borderline = self.borderline_min <= val <= self.borderline_max
            if is_borderline:
                self._borderline_streak[key] = self._borderline_streak.get(key, 0) + 1
            else:
                self._borderline_streak[key] = 0

            streak = self._borderline_streak[key]
            refreshed = False
            saved_path_str: Optional[str] = None
            backup_path_str: Optional[str] = None

            if streak >= self.consecutive_trigger_count and current_patch_bgr is not None:
                saved_to_disk = False
                if patch_save_path is not None:
                    norm_patch = self._normalize_to_80x80(current_patch_bgr)

                    target_path = Path(patch_save_path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # If an existing template exists on disk, blend gently with new lighting & archive old
                    if target_path.exists():
                        existing_img = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
                        bak_path = target_path.with_suffix(".prev.png")
                        try:
                            shutil.copy2(target_path, bak_path)
                            backup_path_str = str(bak_path)
                        except OSError:
                            backup_path_str = None
                        if existing_img is not None and existing_img.size > 0:
                            existing_80 = self._normalize_to_80x80(existing_img)
                            alpha = max(0.3, min(1.0, float(blend_alpha)))
                            norm_patch = cv2.addWeighted(norm_patch, alpha, existing_80, 1.0 - alpha, 0.0)

                    # F-1821: imwrite dönüşü kontrol edilir; dosya
                    # yazılmadıysa refreshed=True denmez.
                    try:
                        wrote = bool(cv2.imwrite(str(target_path), norm_patch))
                    except Exception as exc:
                        logger.warning("SelfHealingTemplateTuner: patch write failed (%s): %s", target_path, exc)
                        wrote = False
                    if wrote:
                        saved_to_disk = True
                        saved_path_str = str(target_path)
                    else:
                        logger.warning(
                            "SelfHealingTemplateTuner: patch write failed, no refresh for node %s",
                            key,
                        )

                if saved_to_disk:
                    # Tune local threshold based on the recent 3 borderline visits
                    recent_borderline = hist[-self.consecutive_trigger_count :]
                    mean_borderline = sum(recent_borderline) / float(len(recent_borderline))
                    tuned_threshold = round(max(self.borderline_min, min(0.75, mean_borderline - 0.015)), 4)
                    self._local_thresholds[key] = tuned_threshold
                    self._borderline_streak[key] = 0
                    self._refresh_counts[key] = self._refresh_counts.get(key, 0) + 1
                    refreshed = True
                # else: dosya yazılmadı (yol verilmedi ya da yazma
                # başarısız) — refreshed=False kalır, eşik oynanmaz ve
                # seri sıfırlanmaz ki sonraki çağrı yolu verilince
                # yenilemeyi tetikleyebilsin.
            elif val > 0.82 and len(hist) >= 3 and all(s > 0.80 for s in hist[-3:]):
                # High-confidence streak gently restores threshold toward default
                cur_t = self._local_thresholds.get(key, self.default_threshold)
                self._local_thresholds[key] = round(min(self.default_threshold, cur_t + 0.005), 4)

            return {
                "node_id": key,
                "score": round(val, 4),
                "is_borderline": is_borderline,
                "borderline_streak": self._borderline_streak[key],
                "refreshed": refreshed,
                "refresh_count": self._refresh_counts.get(key, 0),
                "local_threshold": self.get_node_threshold(key),
                "saved_patch_path": saved_path_str,
                "backup_patch_path": backup_path_str,
            }

    def _tuner_config_valid(self) -> bool:
        """Yapılandırma tutarlılığı: eşik aralığı sıralı ve 0..1 içinde."""
        return (
            0.0 <= self.borderline_min < self.borderline_max <= 1.0
            and self.consecutive_trigger_count >= 1
            and 0.0 < self.default_threshold <= 1.0
        )

    def get_dpi_focus_guard_ready_flag(self) -> dict:
        # F-1826: hazır bayrağı koşulsuz True değil; yapılandırma
        # geçersizse hazır değildir.
        ready = self._tuner_config_valid()
        return {
            "module": "dpi_focus_guard",
            "ready": ready,
            "status": "ok" if ready else "misconfigured",
        }

    def get_dpi_focus_guard_version_info(self) -> dict:
        return {"module": "dpi_focus_guard", "version": __version__, "ready": self._tuner_config_valid()}

    def get_dpi_focus_guard_diagnostic_snapshot(self) -> dict:
        ready = self._tuner_config_valid()
        return {
            "module": "dpi_focus_guard",
            "status": "ok" if ready else "misconfigured",
            "ready": ready,
            "total_refreshes": sum(self._refresh_counts.values()),
            "tracked_nodes": len(self._score_history),
        }

    def get_dpi_focus_guard_capabilities(self) -> list:
        return ["dpi_focus_guard", "diagnostic_snapshot", "version_info"]


def get_dpi_focus_guard_capabilities() -> list:
    return ["dpi_focus_guard", "diagnostic_snapshot", "version_info"]


def validate_dpi_focus_guard_contract() -> bool:
    # F-1826: sözleşme, sabit listedeki eleman sayısıyla değil gerçek
    # çalışabilirlikle doğrulanır: üç sınıf kurulur ve tanı
    # anlık görüntüleri hazır raporlamalıdır.
    try:
        if not get_dpi_focus_guard_capabilities():
            return False
        sync = DPIScalingSynchronizer()
        guard = ForegroundFocusGuard()
        tuner = SelfHealingTemplateTuner()
        return bool(
            sync.get_dpi_focus_guard_diagnostic_snapshot().get("ready")
            and guard.get_dpi_focus_guard_diagnostic_snapshot().get("ready")
            and tuner.get_dpi_focus_guard_ready_flag().get("ready")
        )
    except Exception:
        return False


def get_dpi_focus_guard_baseline_metrics() -> dict:
    return {"module": "dpi_focus_guard", "contract_valid": validate_dpi_focus_guard_contract(), "capability_count": len(get_dpi_focus_guard_capabilities())}


def get_dpi_focus_guard_readiness_status() -> dict:
    return {
        "module": "dpi_focus_guard",
        "ready": validate_dpi_focus_guard_contract(),
        "metrics": get_dpi_focus_guard_baseline_metrics(),
    }


def get_dpi_focus_guard_audit_record() -> dict:
    return {
        "module": "dpi_focus_guard",
        "audit_passed": validate_dpi_focus_guard_contract(),
        "readiness": get_dpi_focus_guard_readiness_status(),
    }


def get_dpi_focus_guard_operational_digest() -> dict:
    return {
        "module": "dpi_focus_guard",
        "operational": validate_dpi_focus_guard_contract(),
        "audit": get_dpi_focus_guard_audit_record(),
    }


def get_dpi_focus_guard_telemetry_payload() -> dict:
    return {
        "module": "dpi_focus_guard",
        "telemetry_ready": validate_dpi_focus_guard_contract(),
        "digest": get_dpi_focus_guard_operational_digest(),
    }


def get_dpi_focus_guard_integrity_seal() -> dict:
    return {
        "module": "dpi_focus_guard",
        "sealed": validate_dpi_focus_guard_contract(),
        "telemetry": get_dpi_focus_guard_telemetry_payload(),
    }


def get_dpi_focus_guard_compliance_certificate() -> dict:
    return {
        "module": "dpi_focus_guard",
        "certified": validate_dpi_focus_guard_contract(),
        "seal": get_dpi_focus_guard_integrity_seal(),
    }


def get_dpi_focus_guard_readiness_probe() -> dict:
    return {
        "module": "dpi_focus_guard",
        "ready": validate_dpi_focus_guard_contract(),
        "certificate": get_dpi_focus_guard_compliance_certificate(),
    }


def get_dpi_focus_guard_audit_summary() -> dict:
    return {
        "module": "dpi_focus_guard",
        "audit_passed": validate_dpi_focus_guard_contract(),
        "probe": get_dpi_focus_guard_readiness_probe(),
    }


def get_dpi_focus_guard_heartbeat_snapshot() -> dict:
    return {
        "module": "dpi_focus_guard",
        "alive": validate_dpi_focus_guard_contract(),
        "audit": get_dpi_focus_guard_audit_summary(),
    }



