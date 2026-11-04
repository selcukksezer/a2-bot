# -*- coding: utf-8 -*-
"""
Katman IV: Davranışsal Doğallık, 10s Takılı Prompt [Envanter Dolu] Koruması,
Karantina ve Odyle Doğrulayıcı (Behavioral Stealth Guard & Natural Combat Rotation).

Bu modül oyun içi davranış doğallığını ve anti-ban güvenlik kurallarını yürütür:
1. Ağırlıklı İleri Koridor Seçimi (`select_natural_next_node`):
   - Hem rotanın geri dönmeden sırayla ilerlemesini (`Sequential Corridor`) korur,
     hem de her turda %100 robotik aynı mikro-sırayı tekrarlamamak için ileri
     koridordaki en yakın hazır 2-3 aday arasından mesafe-ağırlıklı
     (`softmax / inverse-distance weighted`) doğal seçim yapar.
2. Başarısız Düğümleri Geçici Kara Listeye Alma (`NodeQuarantineManager`):
   - Bir kaynak kayanın içine sıkışmışsa veya ulaşılamıyorsa botun oraya 50 kez
     üst üste tıklamasını önler. 2 veya 3 başarısız denemeden sonra o düğümü
     geçici olarak karantinaya alır ve `"Node #X quarantined after N failures"` loglar.
3. Envanter Dolu / Seviye Yetersiz Algılama (`check_prompt_stuck`):
   - "Prompt stuck for 10s" Kuralı: Eğer bot aynı kaynakta `10.0` saniye boyunca
     `F` bastığı halde `"F Gather"` yazısı kaybolmuyorsa (envanter/Cube dolmuş
     veya meslek seviyesi yetmiyor demektir), anında
     `(True, "Paused: gather prompt stuck for 10s (Inventory full or level requirement not met)")`
     döndürerek botu güvenli şekilde durdurur.
4. Havadaki Odyle (Uçuş Öncesi) Minimap Doğrulaması (`should_fly_to_odyle_node`):
   - Havadaki bir kaynağa (`in_air == True` / Odyle) uçmadan önce minimap üzerinde
     kaynağın gerçekten orada olup olmadığını kontrol eder; kaynak orada yoksa
     boşuna havaya uçup şüphe çekmemek için uçuşu iptal eder ve sıradaki kaynağa geçer.
5. Doğal Savaş Rotasyonu (`NaturalCombatRotation`):
   - Saldırıya uğradığında saniyede 10 kez sol tık spam'lemek yerine, `1`, `2`, `3`, `4`
     tuşlarındaki yeteneklerin kendi dolum sürelerini (`cooldowns = {'1': 2.5, '2': 6.0, '3': 10.0, '4': 15.0}`)
     takip ederek ve araya Ex-Gaussian gecikmeli normal saldırı katarak insani bir kombo uygular.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.minimap_scanner import MinimapScanner

logger = logging.getLogger("BehavioralGuard")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# YARDIMCI FONKSİYONLAR (KOORDİNAT, KİMLİK VE EX-GAUSSIAN DAĞILIM)
# ============================================================================


def _extract_node_id(node: Any) -> Optional[int]:
    """Bir düğüm nesnesinden (`NodeData`, `dict` veya `int`) güvenli biçimde `node_id` çıkarır."""
    try:
        if node is None:
            return None
        if isinstance(node, (int, np.integer)):
            return int(node)
        if isinstance(node, dict):
            val = node.get("id", node.get("node_id"))
            return int(val) if val is not None else None
        val = getattr(node, "id", getattr(node, "node_id", None))
        return int(val) if val is not None else None
    except Exception as exc:
        logger.debug("Düğüm kimliği çıkarılamadı (%r): %s", node, exc)
        return None


def _extract_node_xy(
    node: Any,
    fallback_width: float = 1920.0,
    fallback_height: float = 1080.0,
) -> Tuple[float, float]:
    """Bir düğümden 2B harita koordinatını `(x, y)` piksel cinsinden güvenli biçimde çıkarır."""
    try:
        if node is None:
            return 0.0, 0.0
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return float(node[0]), float(node[1])
        if isinstance(node, dict):
            mx = node.get("map_x", node.get("x", 0.0))
            my = node.get("map_y", node.get("y", 0.0))
            if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
                nx = node.get("norm_x")
                ny = node.get("norm_y")
                if nx is not None and ny is not None:
                    return float(nx) * fallback_width, float(ny) * fallback_height
            return float(mx or 0.0), float(my or 0.0)

        mx = getattr(node, "map_x", getattr(node, "x", 0.0))
        my = getattr(node, "map_y", getattr(node, "y", 0.0))
        if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
            nx = getattr(node, "norm_x", None)
            ny = getattr(node, "norm_y", None)
            if nx is not None and ny is not None:
                return float(nx) * fallback_width, float(ny) * fallback_height
        return float(mx or 0.0), float(my or 0.0)
    except Exception as exc:
        logger.debug("Düğüm koordinatı çıkarılamadı (%r): %s", node, exc)
        return 0.0, 0.0


def _is_node_in_air(node: Any) -> bool:
    """Düğümün havada (`in_air == True` veya Odyle kaynağı) olup olmadığını kontrol eder."""
    try:
        if node is None:
            return False
        if isinstance(node, dict):
            if bool(node.get("in_air", False)) or bool(node.get("is_odyle", False)):
                return True
            res_type = str(node.get("resource_type", "")).lower()
            return "odyle" in res_type or "aether" in res_type
        if bool(getattr(node, "in_air", False)) or bool(getattr(node, "is_odyle", False)):
            return True
        res_type = str(getattr(node, "resource_type", "")).lower()
        return "odyle" in res_type or "aether" in res_type
    except Exception as exc:
        logger.debug("in_air kontrolü başarısız (%r): %s", node, exc)
        return False


def sample_ex_gaussian(
    mu: float = 0.28,
    sigma: float = 0.04,
    tau: float = 0.09,
    min_val: float = 0.12,
    max_val: float = 1.25,
    rng: Optional[random.Random] = None,
) -> float:
    """
    İnsan reaksiyon sürelerini modelleyen Ex-Gaussian (Normal + Üstel kuyruk) dağılımından
    rastgele gecikme süresi (saniye) üretir.
    """
    try:
        gen = rng if rng is not None else random
        gauss_part = gen.gauss(float(mu), max(1e-4, float(sigma)))
        exp_part = gen.expovariate(1.0 / max(1e-4, float(tau))) if tau > 0 else 0.0
        val = gauss_part + exp_part
        return round(float(max(min_val, min(max_val, val))), 4)
    except Exception as exc:
        logger.debug("Ex-Gaussian örnekleme hatası: %s", exc)
        return round(float(max(min_val, min(max_val, mu + tau))), 4)


# ============================================================================
# 2. BAŞARISIZ DÜĞÜMLERİ GEÇİCİ KARA LİSTEYE ALMA (NODE QUARANTINE MANAGER)
# ============================================================================


@dataclass
class QuarantineEntry:
    """Tek bir düğümün hata sayacı ve geçici karantina zaman damgası bilgisi."""

    node_id: int
    failure_count: int = 0
    quarantined_at: float = 0.0
    quarantine_until: float = 0.0
    total_quarantines: int = 0
    last_attempt_at: float = 0.0


class NodeQuarantineManager:
    """
    Başarısız Düğümleri Geçici Kara Listeye Alma Yöneticisi.

    Bir kaynak kayanın içine sıkışmışsa veya ulaşılamıyorsa botun oraya 50 kez
    üst üste tıklamasını önler. `max_failures` (varsayılan 2) başarısız denemeden
    sonra o düğümü `quarantine_minutes` (varsayılan 30 dakika) boyunca geçici
    olarak karantinaya alır.
    """

    def __init__(
        self,
        default_max_failures: int = 2,
        default_quarantine_minutes: int = 30,
    ) -> None:
        self._lock = threading.RLock()
        self.default_max_failures: int = max(1, int(default_max_failures))
        self.default_quarantine_minutes: int = max(1, int(default_quarantine_minutes))
        self._entries: Dict[int, QuarantineEntry] = {}

    def record_node_attempt(
        self,
        node_id: int,
        success: bool,
        max_failures: int = 2,
        quarantine_minutes: int = 30,
        now: Optional[float] = None,
    ) -> bool:
        """
        Düğüm toplama/ulaşma denemesini kaydeder.

        - `success=True` ise düğümün ardışık hata sayacını sıfırlar ve varsa aktif
          karantinasını kaldırır (`False` döner).
        - `success=False` ise ardışık hata sayacını artırır. Hata sayısı `max_failures`
          (örn. 2 veya 3) değerine ulaştığında düğümü `quarantine_minutes` dakika boyunca
          karantinaya alır, loga `"Node #X quarantined after N failures"` yazar ve
          `True` (karantinaya alındı) döndürür.
        """
        try:
            nid = int(node_id)
            current_time = time.time() if now is None else float(now)
            threshold = max(1, int(max_failures)) if max_failures is not None else self.default_max_failures
            q_minutes = (
                max(1, int(quarantine_minutes))
                if quarantine_minutes is not None
                else self.default_quarantine_minutes
            )

            with self._lock:
                entry = self._entries.get(nid)
                if entry is None:
                    entry = QuarantineEntry(node_id=nid)
                    self._entries[nid] = entry

                entry.last_attempt_at = current_time

                if success:
                    entry.failure_count = 0
                    entry.quarantined_at = 0.0
                    entry.quarantine_until = 0.0
                    return False

                entry.failure_count += 1
                if entry.failure_count >= threshold:
                    entry.quarantined_at = current_time
                    entry.quarantine_until = current_time + float(q_minutes) * 60.0
                    entry.total_quarantines += 1
                    logger.warning(
                        "Node #%d quarantined after %d failures (duration: %d min)",
                        nid,
                        entry.failure_count,
                        q_minutes,
                    )
                    return True

                logger.info(
                    "Node #%d attempt failed (%d/%d before quarantine)",
                    nid,
                    entry.failure_count,
                    threshold,
                )
                return False
        except Exception as exc:
            logger.error("record_node_attempt hatası (node_id=%r): %s", node_id, exc)
            return False

    def is_quarantined(self, node_id: Union[int, Any], now: Optional[float] = None) -> bool:
        """
        Belirtilen düğümün şu anda geçici kara listede (karantinada) olup olmadığını döndürür.
        Karantina süresi dolmuşsa otomatik olarak karantinadan çıkarır.
        """
        try:
            nid = _extract_node_id(node_id)
            if nid is None:
                return False

            current_time = time.time() if now is None else float(now)
            with self._lock:
                entry = self._entries.get(nid)
                if entry is None or entry.quarantine_until <= 0.0:
                    return False

                if current_time >= entry.quarantine_until:
                    logger.info("Node #%d quarantine expired; restoring to active pool.", nid)
                    entry.failure_count = 0
                    entry.quarantined_at = 0.0
                    entry.quarantine_until = 0.0
                    return False

                return True
        except Exception as exc:
            logger.error("is_quarantined hatası (node_id=%r): %s", node_id, exc)
            return False

    def get_failure_count(self, node_id: Union[int, Any]) -> int:
        """Düğümün mevcut ardışık başarısız deneme sayısını döndürür."""
        try:
            nid = _extract_node_id(node_id)
            if nid is None:
                return 0
            with self._lock:
                entry = self._entries.get(nid)
                return int(entry.failure_count) if entry is not None else 0
        except Exception as exc:
            logger.debug("get_failure_count hatası: %s", exc)
            return 0

    def get_remaining_quarantine_sec(
        self, node_id: Union[int, Any], now: Optional[float] = None
    ) -> float:
        """Düğümün karantinadan çıkmasına kalan süreyi (saniye) döndürür."""
        try:
            nid = _extract_node_id(node_id)
            if nid is None:
                return 0.0
            current_time = time.time() if now is None else float(now)
            with self._lock:
                entry = self._entries.get(nid)
                if entry is None or entry.quarantine_until <= current_time:
                    return 0.0
                return round(float(entry.quarantine_until - current_time), 2)
        except Exception as exc:
            logger.debug("get_remaining_quarantine_sec hatası: %s", exc)
            return 0.0

    def release_node(self, node_id: Union[int, Any]) -> None:
        """Düğümü manuel olarak karantinadan çıkarır ve hata sayacını sıfırlar."""
        try:
            nid = _extract_node_id(node_id)
            if nid is None:
                return
            with self._lock:
                if nid in self._entries:
                    self._entries[nid].failure_count = 0
                    self._entries[nid].quarantined_at = 0.0
                    self._entries[nid].quarantine_until = 0.0
        except Exception as exc:
            logger.debug("release_node hatası: %s", exc)

    def clear_all(self) -> None:
        """Tüm karantina kayıtlarını temizler."""
        with self._lock:
            self._entries.clear()

    def filter_available_nodes(
        self, nodes: Sequence[Any], now: Optional[float] = None
    ) -> List[Any]:
        """Verilen düğüm listesinden karantinada olmayanları filtreleyip döndürür."""
        try:
            if not nodes:
                return []
            return [n for n in nodes if not self.is_quarantined(n, now=now)]
        except Exception as exc:
            logger.error("filter_available_nodes hatası: %s", exc)
            return list(nodes) if nodes else []


# ============================================================================
# 5. DOĞAL SAVAŞ ROTASYONU (NATURAL COMBAT ROTATION)
# ============================================================================


class NaturalCombatRotation:
    """
    Doğal Savaş Rotasyonu Yöneticisi.

    Saldırıya uğradığında saniyede 10 kez sol tık spam'lemek yerine,
    `1`, `2`, `3`, `4` tuşlarındaki yeteneklerin kendi dolum sürelerini
    (`cooldowns = {'1': 2.5, '2': 6.0, '3': 10.0, '4': 15.0}`) takip ederek
    ve araya Ex-Gaussian gecikmeli normal saldırı katarak insani bir kombo uygular.
    """

    DEFAULT_COOLDOWNS: Dict[str, float] = {
        "1": 2.5,
        "2": 6.0,
        "3": 10.0,
        "4": 15.0,
    }

    def __init__(
        self,
        cooldowns: Optional[Dict[str, float]] = None,
        basic_attack_key: str = "left_click",
        ex_gauss_mu: float = 0.28,
        ex_gauss_sigma: float = 0.045,
        ex_gauss_tau: float = 0.095,
        seed: Optional[int] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.cooldowns: Dict[str, float] = dict(self.DEFAULT_COOLDOWNS)
        if cooldowns:
            for k, v in cooldowns.items():
                self.cooldowns[str(k)] = max(0.1, float(v))

        self.basic_attack_key: str = str(basic_attack_key or "left_click")
        self.ex_gauss_mu: float = float(ex_gauss_mu)
        self.ex_gauss_sigma: float = float(ex_gauss_sigma)
        self.ex_gauss_tau: float = float(ex_gauss_tau)
        self._rng = random.Random(seed)

        # Her yeteneğin son kullanıldığı zaman damgası
        self._last_used_at: Dict[str, float] = {k: 0.0 for k in self.cooldowns}
        self._last_action_at: float = 0.0
        self._next_allowed_action_at: float = 0.0
        self._combo_step: int = 0
        self._consecutive_skills: int = 0

    def reset_cooldowns(self) -> None:
        """Tüm yetenek dolum sürelerini ve kombo sayacını sıfırlar."""
        with self._lock:
            for k in self.cooldowns:
                self._last_used_at[k] = 0.0
            self._last_action_at = 0.0
            self._next_allowed_action_at = 0.0
            self._combo_step = 0
            self._consecutive_skills = 0

    def is_skill_ready(self, key: str, now: Optional[float] = None) -> bool:
        """Belirtilen yetenek tuşunun (`'1'`, `'2'`, `'3'`, `'4'`) dolum süresinin bitip bitmediğini döndürür."""
        try:
            k = str(key)
            current_time = time.time() if now is None else float(now)
            with self._lock:
                cd = self.cooldowns.get(k)
                if cd is None:
                    return False
                last_t = self._last_used_at.get(k, 0.0)
                if last_t <= 0.0:
                    return True
                return (current_time - last_t) >= float(cd)
        except Exception as exc:
            logger.debug("is_skill_ready hatası (%r): %s", key, exc)
            return False

    def get_remaining_cooldown(self, key: str, now: Optional[float] = None) -> float:
        """Yeteneğin tekrar hazır olmasına kalan saniyeyi döndürür (hazırsa 0.0)."""
        try:
            k = str(key)
            current_time = time.time() if now is None else float(now)
            with self._lock:
                cd = self.cooldowns.get(k, 0.0)
                last_t = self._last_used_at.get(k, 0.0)
                if last_t <= 0.0 or cd <= 0.0:
                    return 0.0
                rem = float(cd) - (current_time - last_t)
                return round(max(0.0, rem), 3)
        except Exception as exc:
            logger.debug("get_remaining_cooldown hatası (%r): %s", key, exc)
            return 0.0

    def get_ready_skills(self, now: Optional[float] = None) -> List[str]:
        """Şu an dolum süresi tamamlanmış tüm yetenek tuşlarının listesini döndürür."""
        current_time = time.time() if now is None else float(now)
        with self._lock:
            return [k for k in self.cooldowns if self.is_skill_ready(k, now=current_time)]

    def sample_reaction_delay(self) -> float:
        """Ex-Gaussian dağılımından insani tuş/saldırı arası gecikme süresi (saniye) üretir."""
        return sample_ex_gaussian(
            mu=self.ex_gauss_mu,
            sigma=self.ex_gauss_sigma,
            tau=self.ex_gauss_tau,
            min_val=0.14,
            max_val=0.95,
            rng=self._rng,
        )

    def get_next_combat_action(
        self,
        now: Optional[float] = None,
        allow_weave_basic: bool = True,
    ) -> Dict[str, Any]:
        """
        Sıradaki savaş aksiyonunu belirler, ilgili yeteneğin cooldown sayacını günceller
        ve Ex-Gaussian gecikme bilgisiyle birlikte sözlük olarak döndürür.

        Dönen sözlük yapısı:
        ```python
        {
            "action_type": "skill" | "basic_attack",
            "key": "1" | "2" | "3" | "4" | "left_click",
            "button": "left" | None,
            "delay_sec": float,          # Ex-Gaussian insani bekleme süresi
            "cooldown_sec": float,       # Seçilen yeteneğin toplam CD süresi (basic için 0.0)
            "combo_step": int,           # Kombo adım sayacı
            "ready_skills_before": list, # Seçim öncesi hazır yetenekler
            "timestamp": float,
        }
        ```
        """
        try:
            current_time = time.time() if now is None else float(now)
            with self._lock:
                ready_skills = self.get_ready_skills(now=current_time)
                delay_sec = self.sample_reaction_delay()

                # Art arda 2 yetenek vurulduysa ve allow_weave_basic aktifse araya
                # Ex-Gaussian gecikmeli 1 normal saldırı (auto-attack weave) sıkıştır
                should_weave_basic = (
                    allow_weave_basic
                    and self._consecutive_skills >= 2
                    and len(ready_skills) < len(self.cooldowns)
                )

                chosen_skill: Optional[str] = None
                if ready_skills and not should_weave_basic:
                    # Hazır yetenekler arasından öncelik ve hafif ağırlıklı doğal seçim:
                    # Kısa CD'li kombo başlatıcılar ve hazır güçlü yetenekler dengelenir
                    sorted_ready = sorted(
                        ready_skills,
                        key=lambda k: (
                            0 if self._last_used_at.get(k, 0.0) == 0.0 else 1,
                            float(self.cooldowns.get(k, 99.0)),
                        ),
                    )
                    if len(sorted_ready) == 1 or self._rng.random() < 0.75:
                        chosen_skill = sorted_ready[0]
                    else:
                        chosen_skill = self._rng.choice(sorted_ready[: min(3, len(sorted_ready))])

                self._combo_step += 1
                self._last_action_at = current_time
                self._next_allowed_action_at = current_time + delay_sec

                if chosen_skill is not None:
                    self._last_used_at[chosen_skill] = current_time
                    self._consecutive_skills += 1
                    cd_val = float(self.cooldowns[chosen_skill])
                    return {
                        "action_type": "skill",
                        "key": chosen_skill,
                        "button": None,
                        "delay_sec": delay_sec,
                        "cooldown_sec": cd_val,
                        "combo_step": self._combo_step,
                        "ready_skills_before": ready_skills,
                        "timestamp": current_time,
                    }

                # Hiçbir yetenek hazır değilse veya auto-attack weave adımındaysa:
                # Ex-Gaussian gecikmeli normal saldırı
                self._consecutive_skills = 0
                return {
                    "action_type": "basic_attack",
                    "key": self.basic_attack_key,
                    "button": "left",
                    "delay_sec": delay_sec,
                    "cooldown_sec": 0.0,
                    "combo_step": self._combo_step,
                    "ready_skills_before": ready_skills,
                    "timestamp": current_time,
                }
        except Exception as exc:
            logger.error("get_next_combat_action hatası: %s", exc)
            return {
                "action_type": "basic_attack",
                "key": self.basic_attack_key,
                "button": "left",
                "delay_sec": 0.30,
                "cooldown_sec": 0.0,
                "combo_step": 1,
                "ready_skills_before": [],
                "timestamp": time.time() if now is None else float(now),
            }


# ============================================================================
# DAVRANIŞSAL DOĞALLIK VE GÜVENLİK KORUYUCUSU (BEHAVIORAL STEALTH GUARD)
# ============================================================================


class BehavioralStealthGuard:
    """
    Katman IV: Davranışsal Doğallık, 10s Takılı Prompt Koruması, Karantina ve
    Odyle Uçuş Doğrulayıcı Ana Sınıfı.
    """

    PROMPT_STUCK_PAUSE_MESSAGE: str = (
        "Paused: gather prompt stuck for 10s (Inventory full or level requirement not met)"
    )

    def __init__(
        self,
        quarantine_manager: Optional[NodeQuarantineManager] = None,
        combat_rotation: Optional[NaturalCombatRotation] = None,
        minimap_scanner: Optional[MinimapScanner] = None,
        corridor_top_k: int = 3,
        softmax_temperature: float = 0.65,
        seed: Optional[int] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.quarantine_manager: NodeQuarantineManager = (
            quarantine_manager if quarantine_manager is not None else NodeQuarantineManager()
        )
        self.combat_rotation: NaturalCombatRotation = (
            combat_rotation if combat_rotation is not None else NaturalCombatRotation(seed=seed)
        )
        self.minimap_scanner: MinimapScanner = (
            minimap_scanner if minimap_scanner is not None else MinimapScanner()
        )
        self.corridor_top_k: int = max(2, min(3, int(corridor_top_k)))
        self.softmax_temperature: float = max(0.05, float(softmax_temperature))
        self._rng = random.Random(seed)

        # 10s Takılı Prompt ("F Gather") durum takibi
        self._prompt_visible_since: Optional[float] = None
        self._prompt_stuck_node_id: Optional[int] = None
        self._prompt_stuck_triggered: bool = False

    # ------------------------------------------------------------------
    # 1. Ağırlıklı İleri Koridor Seçimi (Weighted Forward Corridor Selection)
    # ------------------------------------------------------------------
    def select_natural_next_node(
        self,
        ready_nodes: list,
        sequential_candidates: list,
        player_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[object]:
        """
        Ağırlıklı İleri Koridor Seçimi (`select_natural_next_node`).

        Hem rotanın geri dönmeden sırayla ilerlemesini (`Sequential Corridor`) korur,
        hem de her turda %100 robotik aynı mikro-sırayı tekrarlamamak için ileri
        koridordaki en yakın hazır 2-3 aday arasından mesafe-ağırlıklı
        (`softmax / inverse-distance weighted`) doğal seçim yapar.
        Karantinadaki düğümleri otomatik olarak eler.
        """
        try:
            ready_list = list(ready_nodes) if ready_nodes else []
            seq_list = list(sequential_candidates) if sequential_candidates else []

            if not ready_list and not seq_list:
                return None

            # Karantinada olmayan hazır düğümleri filtrele
            valid_ready = self.quarantine_manager.filter_available_nodes(ready_list)
            valid_seq = self.quarantine_manager.filter_available_nodes(seq_list)

            if not valid_ready and not valid_seq:
                return None

            # Hazır düğümlerin kimlik ve referans kümesini oluştur
            ready_ids = {_extract_node_id(n) for n in valid_ready if _extract_node_id(n) is not None}
            ready_obj_ids = {id(n) for n in valid_ready}

            # İleri koridor adaylarından hem hazır hem karantinasız olanları sırayı bozmadan al
            corridor_pool: List[Any] = []
            for cand in valid_seq:
                cid = _extract_node_id(cand)
                if not valid_ready or (cid is not None and cid in ready_ids) or (id(cand) in ready_obj_ids):
                    corridor_pool.append(cand)

            # Eğer kesişim boşsa doğrudan geçerli ileri koridor veya hazır listesine düş
            if not corridor_pool:
                corridor_pool = valid_seq if valid_seq else valid_ready

            if not corridor_pool:
                return None

            if len(corridor_pool) == 1:
                return corridor_pool[0]

            # İleri koridordaki en yakın/sıradaki 2-3 adayı belirle
            top_k = min(self.corridor_top_k, len(corridor_pool))
            if player_pos is not None and len(player_pos) >= 2:
                px, py = float(player_pos[0]), float(player_pos[1])
                # İleri koridor penceresini (ilk 5 ileri adayı) koruyarak mesafe + sıra cezası ile en yakın top_k adayı al
                forward_window = corridor_pool[: min(5, len(corridor_pool))]
                scored_window: List[Tuple[float, int, Any]] = []
                for rank_idx, node in enumerate(forward_window):
                    nx, ny = _extract_node_xy(node)
                    dist = math.hypot(nx - px, ny - py)
                    # Sıra koruma faktörü: ileri koridorda çok uzaktaki düğümün önüne geçmesini önler
                    effective_dist = dist * (1.0 + 0.12 * float(rank_idx))
                    scored_window.append((effective_dist, rank_idx, node))

                scored_window.sort(key=lambda item: (item[0], item[1]))
                candidates_k = [item[:2] + (item[2],) for item in scored_window[:top_k]]
            else:
                # Oyuncu pozisyonu verilmemişse ilk aday referans alınarak veya sıra indeksine göre mesafe hesaplanır
                anchor_x, anchor_y = _extract_node_xy(corridor_pool[0])
                candidates_k = []
                for rank_idx, node in enumerate(corridor_pool[:top_k]):
                    nx, ny = _extract_node_xy(node)
                    base_d = math.hypot(nx - anchor_x, ny - anchor_y) if rank_idx > 0 else 0.0
                    effective_dist = base_d + float(rank_idx + 1) * 25.0
                    candidates_k.append((effective_dist, rank_idx, node))

            # Mesafe-ağırlıklı (Inverse-Distance + Softmax) olasılık hesabı
            dists = [max(1.0, float(item[0])) for item in candidates_k]
            min_d = min(dists)
            max_d = max(dists)
            span_d = max(1.0, max_d - min_d)

            weights: List[float] = []
            for idx, d in enumerate(dists):
                norm_d = (d - min_d) / span_d
                # Softmax negatif enerji + ters mesafe + sıra koruma ağırlığı
                inv_factor = 1.0 / (1.0 + (d / max(15.0, min_d)) * 0.65)
                softmax_factor = math.exp(-norm_d / self.softmax_temperature)
                rank_bonus = math.exp(-0.25 * float(idx))
                w = max(1e-4, inv_factor * softmax_factor * rank_bonus)
                weights.append(w)

            total_w = sum(weights)
            if total_w <= 0.0:
                return candidates_k[0][2]

            probs = [w / total_w for w in weights]
            r = self._rng.random()
            cumulative = 0.0
            chosen_node = candidates_k[-1][2]
            for prob, item in zip(probs, candidates_k):
                cumulative += prob
                if r <= cumulative:
                    chosen_node = item[2]
                    break

            return chosen_node

        except Exception as exc:
            logger.error("select_natural_next_node hatası: %s", exc)
            if sequential_candidates:
                return sequential_candidates[0]
            if ready_nodes:
                return ready_nodes[0]
            return None

    # ------------------------------------------------------------------
    # 2. Başarısız Düğümleri Geçici Kara Listeye Alma Delege Metodları
    # ------------------------------------------------------------------
    def record_node_attempt(
        self,
        node_id: int,
        success: bool,
        max_failures: int = 2,
        quarantine_minutes: int = 30,
        now: Optional[float] = None,
    ) -> bool:
        """`NodeQuarantineManager.record_node_attempt` çağrısını yürütür."""
        return self.quarantine_manager.record_node_attempt(
            node_id=node_id,
            success=success,
            max_failures=max_failures,
            quarantine_minutes=quarantine_minutes,
            now=now,
        )

    def is_quarantined(self, node_id: Union[int, Any], now: Optional[float] = None) -> bool:
        """`NodeQuarantineManager.is_quarantined` çağrısını yürütür."""
        return self.quarantine_manager.is_quarantined(node_id=node_id, now=now)

    # ------------------------------------------------------------------
    # 3. Envanter Dolu / Seviye Yetersiz Algılama ("Prompt stuck for 10s")
    # ------------------------------------------------------------------
    def check_prompt_stuck(
        self,
        is_prompt_visible: bool,
        current_node_id: Optional[int] = None,
        max_stuck_sec: float = 10.0,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        "Prompt stuck for 10s" Kuralı:
        Eğer bot aynı kaynakta `max_stuck_sec` (varsayılan `10.0`) saniye boyunca
        `F` bastığı halde `"F Gather"` yazısı kaybolmuyorsa (envanter/Cube dolmuş
        veya meslek seviyesi yetmiyor demektir), anında:
            `(True, "Paused: gather prompt stuck for 10s (Inventory full or level requirement not met)")`
        döndürerek botu güvenli şekilde durdurur!
        """
        try:
            current_time = time.time() if now is None else float(now)
            threshold_sec = max(0.5, float(max_stuck_sec))
            nid = _extract_node_id(current_node_id)

            with self._lock:
                if not is_prompt_visible:
                    self._prompt_visible_since = None
                    self._prompt_stuck_node_id = None
                    self._prompt_stuck_triggered = False
                    return False, ""

                # Eğer farklı bir düğüme geçildiyse veya sayaç henüz başlamadıysa başlat
                if (
                    self._prompt_visible_since is None
                    or (nid is not None and self._prompt_stuck_node_id is not None and nid != self._prompt_stuck_node_id)
                ):
                    self._prompt_visible_since = current_time
                    self._prompt_stuck_node_id = nid
                    self._prompt_stuck_triggered = False
                    return False, ""

                if self._prompt_stuck_node_id is None and nid is not None:
                    self._prompt_stuck_node_id = nid

                elapsed = current_time - self._prompt_visible_since
                if elapsed >= threshold_sec:
                    if not self._prompt_stuck_triggered:
                        self._prompt_stuck_triggered = True
                        logger.warning(
                            "%s [node_id=%s, elapsed=%.2fs]",
                            self.PROMPT_STUCK_PAUSE_MESSAGE,
                            str(nid),
                            elapsed,
                        )
                    return True, self.PROMPT_STUCK_PAUSE_MESSAGE

                return False, ""
        except Exception as exc:
            logger.error("check_prompt_stuck hatası: %s", exc)
            return False, ""

    def reset_prompt_timer(self) -> None:
        """Takılı prompt sayacını sıfırlar."""
        with self._lock:
            self._prompt_visible_since = None
            self._prompt_stuck_node_id = None
            self._prompt_stuck_triggered = False

    # ------------------------------------------------------------------
    # 4. Havadaki Odyle (Uçuş Öncesi) Minimap Doğrulaması
    # ------------------------------------------------------------------
    def should_fly_to_odyle_node(
        self,
        node: Any,
        frame_bgr: np.ndarray,
        minimap_scanner: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        Havadaki bir kaynağa (`in_air == True` / Odyle) uçmadan önce minimap üzerinde
        kaynağın gerçekten orada olup olmadığını kontrol eder; kaynak orada yoksa
        boşuna havaya uçup şüphe çekmemek için uçuşu iptal eder (`False`) ve sıradaki
        kaynağa geçilmesini sağlar.

        - Eğer düğüm havada değilse (`in_air == False`), uçuş gerekmediğini / normal
          yer düğümü olduğunu belirtir.
        - Eğer düğüm havadaysa (`in_air == True`), minimap tarayıcısı ile Odyle/kaynak
          ikonu varlığını doğrular.
        """
        try:
            nid = _extract_node_id(node)
            node_label = f"Node #{nid}" if nid is not None else "Odyle node"

            if not _is_node_in_air(node):
                return False, f"{node_label} is a ground node (in_air=False); flight not required"

            if self.is_quarantined(node):
                return False, f"Skipped flight to {node_label}: node is currently quarantined"

            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                logger.warning(
                    "Skipped flight to %s: invalid or empty frame for minimap Odyle verification",
                    node_label,
                )
                return False, f"Skipped flight to {node_label}: empty frame for minimap verification"

            scanner = minimap_scanner if minimap_scanner is not None else self.minimap_scanner

            detections: List[Dict[str, Any]] = []
            if hasattr(scanner, "scan_minimap_resources_and_odyle"):
                detections = scanner.scan_minimap_resources_and_odyle(
                    frame_bgr, check_odyle_air=True
                )
            elif callable(scanner):
                raw_res = scanner(frame_bgr)
                if isinstance(raw_res, list):
                    detections = raw_res

            # Minimap tespitleri içinde Odyle halkası/tüyü veya parlak kaynak ikonu var mı?
            odyle_or_resource_hits = [
                d
                for d in detections
                if isinstance(d, dict)
                and not bool(d.get("is_hostile", False))
                and (
                    bool(d.get("is_odyle", False))
                    or d.get("type") in ("odyle_air", "resource")
                    or d.get("category") in ("odyle_air", "resource_icon")
                )
            ]

            if odyle_or_resource_hits:
                best_hit = odyle_or_resource_hits[0]
                dist_px = float(best_hit.get("dist_px", 0.0))
                hit_type = str(best_hit.get("type", "odyle_air"))
                msg = (
                    f"Verified {node_label} on minimap ({hit_type} at {dist_px:.1f}px); "
                    f"cleared for flight"
                )
                logger.info(msg)
                return True, msg

            cancel_msg = (
                f"Cancelled flight to {node_label}: Odyle resource not present on minimap "
                f"(skipping to next node to avoid suspicious empty flight)"
            )
            logger.info(cancel_msg)
            return False, cancel_msg

        except Exception as exc:
            logger.error("should_fly_to_odyle_node hatası: %s", exc)
            return False, f"Cancelled flight due to verification error: {exc}"

    # ------------------------------------------------------------------
    # 5. Doğal Savaş Rotasyonu Delege Metodu
    # ------------------------------------------------------------------
    def get_next_combat_action(self, now: Optional[float] = None) -> Dict[str, Any]:
        """`NaturalCombatRotation.get_next_combat_action` çağrısını yürütür."""
        return self.combat_rotation.get_next_combat_action(now=now)


# ============================================================================
# MODÜL SEVİYESİ KISA YOL FONKSİYONLARI (DOĞRUDAN İMPORT UYUMLULUĞU İÇİN)
# ============================================================================

_DEFAULT_GUARD = BehavioralStealthGuard()


def select_natural_next_node(
    ready_nodes: list,
    sequential_candidates: list,
    player_pos: Optional[Tuple[float, float]] = None,
) -> Optional[object]:
    """Modül seviyesi ağırlıklı ileri koridor düğüm seçici."""
    return _DEFAULT_GUARD.select_natural_next_node(
        ready_nodes=ready_nodes,
        sequential_candidates=sequential_candidates,
        player_pos=player_pos,
    )


def check_prompt_stuck(
    is_prompt_visible: bool,
    current_node_id: Optional[int] = None,
    max_stuck_sec: float = 10.0,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """Modül seviyesi 10s takılı prompt ("F Gather") kontrolcüsü."""
    return _DEFAULT_GUARD.check_prompt_stuck(
        is_prompt_visible=is_prompt_visible,
        current_node_id=current_node_id,
        max_stuck_sec=max_stuck_sec,
        now=now,
    )


def should_fly_to_odyle_node(
    node: Any,
    frame_bgr: np.ndarray,
    minimap_scanner: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Modül seviyesi Odyle uçuş öncesi minimap doğrulayıcısı."""
    return _DEFAULT_GUARD.should_fly_to_odyle_node(
        node=node,
        frame_bgr=frame_bgr,
        minimap_scanner=minimap_scanner,
    )
