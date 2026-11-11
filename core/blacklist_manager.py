# -*- coding: utf-8 -*-
"""
Blacklist Manager (Kara Liste Yöneticisi)
=========================================

Otomatik toplama sırasında 3 kez üst üste toplanacak kaynak bulunamayan
ve yapay zeka (YZ) analiziyle yanlış işaretleme olduğu doğrulanan noktaların
koordinatlarını kara listeye alır ve bu noktalara bir daha kesinlikle
düğüm (node) kaydedilmesini engeller.

Özellikler:
1. Koordinat ve Bölge bazlı kara liste kontrolü (`is_blacklisted`).
2. YZ analiz teşhisi (`ai_verdict`) ve hata nedeni ile kayıt.
3. `config/settings.json` ve `config/blacklist.json` ile kalıcı senkronizasyon.
4. Yakınlık toleransı (`threshold_px=25.0`) ile mükerrer işaretleme önleme.
5. Kara listeden çıkarma, temizleme ve filtreleme API'si.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_BLACKLIST_FILE = os.path.join("config", "blacklist.json")
DEFAULT_SETTINGS_FILE = os.path.join("config", "settings.json")
DEFAULT_THRESHOLD_PX = 25.0


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_canonical_region(region: str) -> str:
    clean = str(region or "").strip()
    if not clean:
        return "Altgard"
    # Basic canonical normalization
    clean_lower = clean.lower()
    if "altgard" in clean_lower:
        return "Altgard"
    if "verteron" in clean_lower:
        return "Verteron"
    if "morheim" in clean_lower:
        return "Morheim"
    if "eltnen" in clean_lower:
        return "Eltnen"
    if "poeta" in clean_lower:
        return "Poeta"
    if "ishalgen" in clean_lower:
        return "Ishalgen"
    return clean


@dataclass
class BlacklistEntry:
    """Kara listedeki tek bir engellenen koordinat kaydı."""

    id: str
    region: str
    map_x: int
    map_y: int
    norm_x: float = 0.0
    norm_y: float = 0.0
    reason: str = "3x başarısız toplama (YZ yanlış işaretleme doğrulaması)"
    ai_verdict: str = "empty_terrain"
    created_at: str = ""
    source_node_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.region = _resolve_canonical_region(self.region)
        self.map_x = int(self.map_x)
        self.map_y = int(self.map_y)
        self.norm_x = float(self.norm_x)
        self.norm_y = float(self.norm_y)
        self.reason = str(self.reason or "")
        self.ai_verdict = str(self.ai_verdict or "")
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.source_node_id is not None:
            try:
                self.source_node_id = int(self.source_node_id)
            except (TypeError, ValueError):
                self.source_node_id = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlacklistEntry":
        # F-1932: a corrupt record (e.g. map_x="abc") must not raise — the
        # field falls back to its default instead of killing the whole load.
        return cls(
            id=str(data.get("id", f"bl_{int(time.time()*1000)}")),
            region=str(data.get("region", "Altgard")),
            map_x=cls._safe_int(data.get("map_x", 0), 0),
            map_y=cls._safe_int(data.get("map_y", 0), 0),
            norm_x=cls._safe_float(data.get("norm_x", 0.0), 0.0),
            norm_y=cls._safe_float(data.get("norm_y", 0.0), 0.0),
            reason=str(data.get("reason", "")),
            ai_verdict=str(data.get("ai_verdict", "")),
            created_at=str(data.get("created_at", "")),
            source_node_id=data.get("source_node_id"),
        )


class BlacklistManager:
    """Kara liste yönetimi, sorgulama ve kalıcı depolama motoru."""

    def __init__(
        self,
        blacklist_file: Optional[str] = None,
        settings_file: Optional[str] = None,
    ) -> None:
        root = _project_root()
        self.blacklist_file = (
            blacklist_file
            if blacklist_file and os.path.isabs(blacklist_file)
            else os.path.join(root, blacklist_file or DEFAULT_BLACKLIST_FILE)
        )
        self.settings_file = (
            settings_file
            if settings_file and os.path.isabs(settings_file)
            else os.path.join(root, settings_file or DEFAULT_SETTINGS_FILE)
        )
        self._entries: List[BlacklistEntry] = []
        # I1 (dev/D006): region-partitioned index over _entries. Lookups
        # (is_blacklisted, called per candidate node) used to scan every
        # entry; they now scan only the queried region's entries.
        # _entries stays the source of truth; _by_region is rebuilt after
        # every mutation that can change region membership.
        self._by_region: Dict[str, List[BlacklistEntry]] = {}
        # S272 (v19/P3-8): UI thread'i (settings_window) ile toplama
        # thread'leri (route_manager, auto_cartographer) aynı anda
        # erişiyordu; kilitsiz singleton + kilitsiz mutasyon vardı.
        # RLock: iç içe çağrılar (add_entry -> is_blacklisted -> save)
        # için yeniden girişli olmalı.
        self._lock = threading.RLock()
        self.load()

    def _rebuild_index(self) -> None:
        index: Dict[str, List[BlacklistEntry]] = {}
        for entry in self._entries:
            index.setdefault(entry.region, []).append(entry)
        self._by_region = index

    def load(self) -> List[BlacklistEntry]:
        """Kara liste kayıtlarını diskten yükler (`config/blacklist.json` veya `settings.json`)."""
        loaded: List[BlacklistEntry] = []
        seen_ids = set()

        # 1. Öncelikli dosya: config/blacklist.json
        if os.path.isfile(self.blacklist_file):
            try:
                with open(self.blacklist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # F-1932: one corrupt record must not swallow
                            # the rest of the file.
                            try:
                                entry = BlacklistEntry.from_dict(item)
                            except Exception as exc:
                                logger.warning("Bozuk kara liste kaydı atlandı: %s (%s)", item, exc)
                                continue
                            if entry.id not in seen_ids:
                                seen_ids.add(entry.id)
                                loaded.append(entry)
                elif isinstance(data, dict) and "blacklist" in data:
                    for item in data.get("blacklist", []):
                        if isinstance(item, dict):
                            try:
                                entry = BlacklistEntry.from_dict(item)
                            except Exception as exc:
                                logger.warning("Bozuk kara liste kaydı atlandı: %s (%s)", item, exc)
                                continue
                            if entry.id not in seen_ids:
                                seen_ids.add(entry.id)
                                loaded.append(entry)
            except Exception as exc:
                logger.warning("Kara liste dosyası okunamadı (%s): %s", self.blacklist_file, exc)

        # 2. config/settings.json içindeki blacklist anahtarından tamamlayıcı yükleme
        if os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                if isinstance(sdata, dict) and "blacklist" in sdata:
                    for item in sdata.get("blacklist", []):
                        if isinstance(item, dict):
                            try:
                                entry = BlacklistEntry.from_dict(item)
                            except Exception as exc:
                                logger.warning("Bozuk kara liste kaydı atlandı (settings): %s (%s)", item, exc)
                                continue
                            if entry.id not in seen_ids:
                                seen_ids.add(entry.id)
                                loaded.append(entry)
            except Exception as exc:
                logger.debug("settings.json kara liste okuma hatası: %s", exc)

        with self._lock:
            self._entries = loaded
            self._rebuild_index()
            logger.info("Kara liste yüklendi: %d kayıt mevcut.", len(self._entries))
            return [copy.copy(e) for e in self._entries]

    def save(self) -> bool:
        """Kara listeyi hem `config/blacklist.json` hem de `config/settings.json` dosyalarına güvenli yazar."""
        with self._lock:
            raw_list = [e.to_dict() for e in self._entries]
        success = True

        # 1. config/blacklist.json kaydı (Atomic save)
        try:
            os.makedirs(os.path.dirname(self.blacklist_file), exist_ok=True)
            tmp_file = f"{self.blacklist_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(raw_list, f, indent=4, ensure_ascii=False)
            os.replace(tmp_file, self.blacklist_file)
        except Exception as exc:
            logger.error("Kara liste dosyası yazılamadı (%s): %s", self.blacklist_file, exc)
            success = False

        # 2. config/settings.json senkronizasyonu
        if os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
                if isinstance(settings_data, dict):
                    settings_data["blacklist"] = raw_list
                    tmp_sett = f"{self.settings_file}.tmp"
                    with open(tmp_sett, "w", encoding="utf-8") as f:
                        json.dump(settings_data, f, indent=4, ensure_ascii=False)
                    os.replace(tmp_sett, self.settings_file)
            except Exception as exc:
                # F-1933: a failed settings sync must not still report True.
                logger.error("settings.json blacklist senkronizasyon hatası: %s", exc)
                success = False

        return success

    def _matches(
        self,
        entry: BlacklistEntry,
        canon_reg: str,
        mx: float,
        my: float,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        threshold_px: float = DEFAULT_THRESHOLD_PX,
        viewport_width: float = 1920.0,
    ) -> bool:
        """Shared proximity test for is_blacklisted / remove_by_coordinate."""
        if entry.region != canon_reg:
            return False
        if math.hypot(mx - entry.map_x, my - entry.map_y) <= threshold_px:
            return True
        if (
            norm_x is not None
            and norm_y is not None
            and entry.norm_x > 0.0
            and entry.norm_y > 0.0
        ):
            norm_dist = math.hypot(float(norm_x) - entry.norm_x, float(norm_y) - entry.norm_y)
            # F-1942: the old code hard-coded a 1920 px viewport. The
            # caller can now pass the real viewport width.
            width = float(viewport_width) if viewport_width else 1920.0
            if norm_dist <= (threshold_px / width):
                return True
        return False

    def is_blacklisted(
        self,
        region: str,
        map_x: Union[int, float],
        map_y: Union[int, float],
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        threshold_px: float = DEFAULT_THRESHOLD_PX,
        viewport_width: float = 1920.0,
    ) -> Tuple[bool, Optional[BlacklistEntry]]:
        """
        Verilen koordinatların kara listede olup olmadığını kontrol eder.

        Parametreler:
            region: Bölge adı (örn: "Altgard")
            map_x, map_y: Piksel koordinatları
            norm_x, norm_y: İsteğe bağlı normalize (0..1) koordinatlar
            threshold_px: Piksel cinsinden yakınlık toleransı (varsayılan 25px)
            viewport_width: Normalize eşik için gerçek görünüm genişliği
                (varsayılan 1920; F-1942 — eskiden sabit 1920 varsayılıyordu)

        Döner:
            (is_blacklisted: bool, matching_entry: Optional[BlacklistEntry])
        """
        canon_reg = _resolve_canonical_region(region)
        # P3-8 (S-274): sayı-dışı koordinat ValueError ile çağırana yayılırdı.
        try:
            mx = float(map_x)
            my = float(map_y)
        except (TypeError, ValueError):
            return False, None

        with self._lock:
            for entry in self._by_region.get(canon_reg, ()):
                if self._matches(entry, canon_reg, mx, my, norm_x, norm_y, threshold_px, viewport_width):
                    return True, copy.copy(entry)

        return False, None

    def add_entry(
        self,
        region: str,
        map_x: Union[int, float],
        map_y: Union[int, float],
        norm_x: float = 0.0,
        norm_y: float = 0.0,
        reason: str = "3x başarısız toplama (YZ yanlış işaretleme doğrulaması)",
        ai_verdict: str = "empty_terrain",
        source_node_id: Optional[int] = None,
        threshold_px: float = DEFAULT_THRESHOLD_PX,
    ) -> BlacklistEntry:
        """
        Yeni bir koordinatı kara listeye ekler ve kaydeder.
        Eğer zaten yakın bir kayıt varsa mevcut kaydı günceller ve döndürür.
        """
        canon_reg = _resolve_canonical_region(region)
        mx = int(round(float(map_x)))
        my = int(round(float(map_y)))

        with self._lock:
            return self._add_entry_locked(
                canon_reg, mx, my, norm_x, norm_y, reason, ai_verdict,
                source_node_id, threshold_px,
            )

    def _add_entry_locked(
        self, canon_reg, mx, my, norm_x, norm_y, reason, ai_verdict,
        source_node_id, threshold_px,
    ) -> BlacklistEntry:
        # S272: is_blacklisted artık kopya döner — güncelleme dalı canlı
        # kaydı id ile yeniden çözer, kopyayı değil.
        is_already, existing = self.is_blacklisted(
            canon_reg, mx, my, norm_x=norm_x, norm_y=norm_y, threshold_px=threshold_px
        )
        if is_already and existing is not None:
            live = next((e for e in self._entries if e.id == existing.id), None)
            if live is None:
                return existing
            if reason:
                live.reason = reason
            if ai_verdict:
                live.ai_verdict = ai_verdict
            if source_node_id is not None:
                live.source_node_id = source_node_id
            self.save()
            logger.info("Kara liste kaydı güncellendi: #%s (%s: %d, %d)", live.id, canon_reg, mx, my)
            return copy.copy(live)

        entry_id = f"bl_{int(time.time()*1000)}_{len(self._entries)+1}"
        entry = BlacklistEntry(
            id=entry_id,
            region=canon_reg,
            map_x=mx,
            map_y=my,
            norm_x=float(norm_x),
            norm_y=float(norm_y),
            reason=str(reason or "3x başarısız toplama (YZ yanlış işaretleme doğrulaması)"),
            ai_verdict=str(ai_verdict or "empty_terrain"),
            source_node_id=source_node_id,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._entries.append(entry)
        self._by_region.setdefault(entry.region, []).append(entry)
        self.save()
        logger.warning(
            "⛔ [KARA LİSTE] Yeni koordinat engellendi: %s (%d, %d) [Neden: %s | YZ Teşhis: %s]",
            canon_reg,
            mx,
            my,
            entry.reason,
            entry.ai_verdict,
        )
        return copy.copy(entry)

    def remove_entry(self, entry_id: str) -> bool:
        """Belirtilen ID'ye sahip kara liste kaydını kaldırır."""
        with self._lock:
            target_id = str(entry_id).strip()
            before_len = len(self._entries)
            self._entries = [e for e in self._entries if e.id != target_id]
            if len(self._entries) < before_len:
                self._rebuild_index()
                self.save()
                logger.info("Kara listeden kayıt kaldırıldı: #%s", target_id)
                return True
            return False

    def remove_by_coordinate(
        self,
        region: str,
        map_x: Union[int, float],
        map_y: Union[int, float],
        threshold_px: float = DEFAULT_THRESHOLD_PX,
        viewport_width: float = 1920.0,
    ) -> int:
        """Belirli koordinata yakın kara liste kayıtlarını siler.

        F-1942: eskiden yalnızca ilk eşleşme silinirdi; artık eşleşen
        TÜM kayıtlar silinir ve silinen sayısı döner.
        """
        canon_reg = _resolve_canonical_region(region)
        # P3-8 (S-274 ailesi): sayı-dışı koordinat ValueError verirdi.
        try:
            mx = float(map_x)
            my = float(map_y)
        except (TypeError, ValueError):
            return 0
        with self._lock:
            doomed = [
                e.id
                for e in self._by_region.get(canon_reg, ())
                if self._matches(e, canon_reg, mx, my, None, None, threshold_px, viewport_width)
            ]
            if not doomed:
                return 0
            doomed_set = set(doomed)
            self._entries = [e for e in self._entries if e.id not in doomed_set]
            self._rebuild_index()
            self.save()
            logger.info("Kara listeden %d kayıt kaldırıldı (%s: %d, %d)", len(doomed), canon_reg, mx, my)
            return len(doomed)

    def clear_all(self, region: Optional[str] = None) -> int:
        """Tüm kara listeyi veya belirtilen bölgedeki tüm kayıtları temizler.

        S022 (v19/P3-8): save() atomik (tmp+replace) ama yedeksizdi —
        toplu silme geri alınamıyordu. Silmeden önce tek nesil `.bak`
        yazılır (repo içi yerleşik desen: approved_pool P0-6).
        """
        # Yedek: mevcut dosya varsa .bak'a kopyala (best-effort).
        try:
            if os.path.isfile(self.blacklist_file):
                shutil.copy2(self.blacklist_file, f"{self.blacklist_file}.bak")
        except Exception as exc:
            logger.error("Kara liste .bak yedeği yazılamadı (%s): %s", self.blacklist_file, exc)
        with self._lock:
            return self._clear_all_locked(region)

    def _clear_all_locked(self, region: Optional[str] = None) -> int:
        if region:
            canon = _resolve_canonical_region(region)
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.region != canon]
            removed = before - len(self._entries)
            self._rebuild_index()
            self.save()
            logger.info("'%s' bölgesine ait %d kara liste kaydı temizlendi.", canon, removed)
            return removed
        else:
            count = len(self._entries)
            self._entries.clear()
            self._by_region.clear()
            self.save()
            logger.info("Tüm kara liste temizlendi (%d kayıt).", count)
            return count

    def get_entries(self, region: Optional[str] = None) -> List[BlacklistEntry]:
        """Kayıtların listesini döndürür (bölgeye göre filtrelenebilir).

        S273 (v19/P3-8): eskiden canlı nesneler dönüyordu — dışarıdan
        `entry.region` değişimi `_by_region` indeksini bozuyordu.
        Artık sığ kopyalar döner; tek okuyucu (settings_window) yalnızca
        görüntüler, kopya davranışını değiştirmez.
        """
        with self._lock:
            if region:
                canon = _resolve_canonical_region(region)
                return [copy.copy(e) for e in self._entries if e.region == canon]
            return [copy.copy(e) for e in self._entries]

    def count(self, region: Optional[str] = None) -> int:
        return len(self.get_entries(region))

    def to_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def from_list(self, items: Sequence[Dict[str, Any]]) -> None:
        with self._lock:
            self._entries = [BlacklistEntry.from_dict(item) for item in items if isinstance(item, dict)]
            self._rebuild_index()
            self.save()


_GLOBAL_BLACKLIST_MANAGER: Optional[BlacklistManager] = None
_GLOBAL_BLACKLIST_MANAGER_LOCK = threading.Lock()


def get_blacklist_manager() -> BlacklistManager:
    """Singleton BlacklistManager erişimi (S272: çift-kontrollü kilit)."""
    global _GLOBAL_BLACKLIST_MANAGER
    if _GLOBAL_BLACKLIST_MANAGER is None:
        with _GLOBAL_BLACKLIST_MANAGER_LOCK:
            if _GLOBAL_BLACKLIST_MANAGER is None:
                _GLOBAL_BLACKLIST_MANAGER = BlacklistManager()
    return _GLOBAL_BLACKLIST_MANAGER


def verify_and_blacklist_node(*args: Any, **kwargs: Any) -> Tuple[bool, str, Any]:
    """core.wrong_node_scan.verify_and_blacklist_node için doğrudan erişim köprüsü."""
    from core.wrong_node_scan import verify_and_blacklist_node as _impl
    return _impl(*args, **kwargs)

