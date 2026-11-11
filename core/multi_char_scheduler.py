# -*- coding: utf-8 -*-
"""
Çoklu Karakter (Multi-Character), Zamanlanmış Görev Takvimi (Scheduler) ve Pet/Buff Yönetimi
============================================================================================
Bölüm 4: Kurumsal Seviye Çoklu Karakter Rotasyonu, Otomasyon Takvimi ve Yoldaş/Buff Yöneticisi.

Alt Modüller:
1. `AltCharacterManager`:
   - `config/characters.json` üzerinde karakter profillerini (`CharacterProfile`) yönetir.
   - Günlük toplama limiti (`daily_gather_limit`) veya özüt/enerji tükenmesi (`energy_depleted`)
     durumunda güvenli çıkış (`Esc -> Karakter Seçimi -> Slot #N Seç -> Oyuna Gir`) dizisini
     çalıştırır ve `RouteManager` aktif bölge/rotasını yeni karakterin atamasına geçirir.
2. `AutomationTaskScheduler`:
   - Günlük ve haftalık zaman pencerelerini (`ScheduleEntry`) yönetir.
   - Örnek: `"19:00'da rotayı başlat, 21:30'da Dünya Bossu (World Boss) / Etkinlik saatinde şehre dönüp bekle, 22:15'te toplamaya devam et"`.
   - `evaluate_current_schedule(now_dt=None)` ile anlık direktifi (`recommended_state`, `reason_tr`, `active_entry`) döndürür.
3. `PetAndBuffManager`:
   - Ganimet yoldaşı (Loot Pet) açlık zamanlayıcısını (`pet_feed_interval_sec`, `pet_key`) ve
     toplama hızı parşömeni / verim yemeği (`BuffSlot`) sürelerini takip eder.
   - Ekran üzerindeki buff çubuğu bölgesini (`frame`) OpenCV HSV analiziyle denetler,
     süresi biten veya ekranda sönen buffları ve pet beslemesini otomatik yeniler.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from version import __version__

logger = logging.getLogger("Gatherion.MultiCharScheduler")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


# ============================================================================
# YARDIMCI GİRDİ VE ZAMAN FONKSİYONLARI
# ============================================================================

def _controller_keyboard_trusted(input_ctrl: Any) -> Optional[bool]:
    """
    Kontrolcünün donanım klavye yolunu (Interception / HID) doğrulayıp
    doğrulamadığını döndürür: True = güvenilir yol aktif, False = yok,
    None = kontrolcü bu bilgiyi sunmuyor (bilinmiyor).
    """
    try:
        if hasattr(input_ctrl, "is_keyboard_trusted") and callable(input_ctrl.is_keyboard_trusted):
            return bool(input_ctrl.is_keyboard_trusted())
        if hasattr(input_ctrl, "keyboard_inject_trusted"):
            return bool(input_ctrl.keyboard_inject_trusted)
        if hasattr(input_ctrl, "_hw_key_trusted_available") and callable(input_ctrl._hw_key_trusted_available):
            return bool(input_ctrl._hw_key_trusted_available())
    except Exception:
        return None
    return None


def _controller_mouse_trusted(input_ctrl: Any) -> Optional[bool]:
    """
    Kontrolcünün donanım fare yolunu (Interception / HID) doğrulayıp
    doğrulamadığını döndürür: True = güvenilir yol aktif, False = yok,
    None = kontrolcü bu bilgiyi sunmuyor (bilinmiyor).
    Klavye karşılığı `_controller_keyboard_trusted` ile aynı desen (S-0802);
    üretim `input_driver` fare için böyle bir sinyal sunmaz, o durumda None
    döner ve tıklama engellenmez — yalnızca açıkça False ise gönderilmez.
    """
    try:
        if hasattr(input_ctrl, "is_mouse_trusted") and callable(input_ctrl.is_mouse_trusted):
            return bool(input_ctrl.is_mouse_trusted())
        if hasattr(input_ctrl, "mouse_inject_trusted"):
            return bool(input_ctrl.mouse_inject_trusted)
        if hasattr(input_ctrl, "_hw_mouse_trusted_available") and callable(input_ctrl._hw_mouse_trusted_available):
            return bool(input_ctrl._hw_mouse_trusted_available())
    except Exception:
        return None
    return None


def _click_result_coords(result: Any, default_px: int, default_py: int) -> Tuple[int, int]:
    """
    Tıklama metodunun dönüş değerini koordinata çevirir (S-0802).
    Yalnızca sayısal `(x, y[, ...])` tuple/list başarı sayılır; `None`,
    `False`, `0` veya sayısal olmayan dönüş tıklamanın gerçekleşmediğini
    bildirir ve RuntimeError yükseltir — eskisi gibi varsayılan koordinatla
    "tıklandı" sayılmaz. Üretim `input_driver` tıklamaları her zaman
    `(x, y, hold)` tuple döner; `None` dönen sahte/üçüncü taraf kontrolcüler
    artık sessizce başarılı sayılmaz.
    """
    if isinstance(result, bool):
        raise RuntimeError("Normalize tıklama başarısız döndü (False): tıklama gerçekleşmedi")
    if result is None:
        raise RuntimeError("Normalize tıklama dönüş değeri yok (None): tıklama doğrulanamadı")
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        try:
            return int(result[0]), int(result[1])
        except (TypeError, ValueError):
            raise RuntimeError("Normalize tıklama sayısal olmayan koordinat döndü: tıklama doğrulanamadı")
    if isinstance(result, (int, float)):
        # Tuş gönderimindeki `_dispatch_ok` konvansiyonu: 0/0.0 = iptal/başarısız,
        # pozitif sayı (basılı tutma süresi) = başarı; koordinat bilinmediğinden
        # varsayılan koordinat döner.
        if float(result) == 0.0:
            raise RuntimeError("Normalize tıklama başarısız döndü (0): tıklama gerçekleşmedi")
        return default_px, default_py
    raise RuntimeError("Normalize tıklama bilinmeyen dönüş değeri döndü: tıklama doğrulanamadı")


def _dispatch_ok(result: Any) -> bool:
    """Gönderim metodunun dönüş değerini başarıya çevirir (None = bilgi yok, sayıldı)."""
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        # press_key basılı tutma süresini (sn) döner; 0.0 = iptal/başarısız.
        return float(result) != 0.0
    return bool(result)


def _send_key_safe(input_ctrl: Any, key: str, test_mode: bool = False) -> bool:
    """
    Verilen `input_ctrl` nesnesi üzerinden güvenli tuş basımı gerçekleştirir.
    `test_mode=True` ise fiziksel girdi göndermeden simüle eder (True döner).
    Üretimde (`test_mode=False`) kontrolcü yoksa veya kontrolcü hiçbir gönderme
    metoduna sahip değilse False döner — eskisi gibi sessizce True dönmez.
    AION 2 yumuşak (SendInput) girdiyi yok saydığı için, üretimde kontrolcü
    güvenilir bir donanım yolunu (Interception/HID) bildiriyorsa ve o yol
    kapalıysa tuş hiç gönderilmez, False döner.
    """
    key_clean = str(key).strip().lower()
    if not key_clean:
        return False

    if input_ctrl is None:
        if test_mode:
            logger.info("[Simülasyon] Tuş basımı: '%s' (test_mode=%s)", key_clean, test_mode)
            return True
        logger.error("Tuş gönderilemedi ('%s'): girdi kontrolcüsü yok ve test_mode kapalı", key_clean)
        return False

    if not test_mode:
        trusted = _controller_keyboard_trusted(input_ctrl)
        if trusted is False:
            logger.error(
                "Tuş gönderilmedi ('%s'): güvenilir donanım klavye yolu (Interception/HID) kapalı; "
                "yumuşak girdi AION 2'de yok sayılır",
                key_clean,
            )
            return False

    try:
        if hasattr(input_ctrl, "set_test_mode") and isinstance(test_mode, bool):
            input_ctrl.set_test_mode(test_mode)

        if hasattr(input_ctrl, "press_key"):
            return _dispatch_ok(input_ctrl.press_key(key_clean))
        if hasattr(input_ctrl, "tap_key"):
            return _dispatch_ok(input_ctrl.tap_key(key_clean))
        if hasattr(input_ctrl, "send_key"):
            return _dispatch_ok(input_ctrl.send_key(key_clean))
        if callable(input_ctrl):
            return _dispatch_ok(input_ctrl(key_clean))
    except Exception as exc:
        logger.error("Tuş gönderim hatası ('%s'): %s", key_clean, exc)
        return False

    logger.error("Tuş gönderilemedi ('%s'): kontrolcü hiçbir tuş gönderme metoduna sahip değil", key_clean)
    return False


def _resolve_click_screen_size(input_ctrl: Any) -> Optional[Tuple[int, int]]:
    """Tıklama piksel hesabı için ekran çözünürlüğünü okur.

    P1-7 (K-20): 1920x1080 sabiti kaldırıldı — sıra: açıkça verilen
    boyut > kontrolcünün `screen_width/screen_height` öznitelikleri >
    kontrolcünün çözünürlük metodu (`_cached_screen_resolution` /
    `get_screen_resolution`). Hiçbiri yoksa `None` döner; çağıran
    1920x1080 son-çare yedeğini kullanır (test simülasyonu dahil).
    """
    if input_ctrl is None:
        return None
    for w_attr, h_attr in (
        ("screen_width", "screen_height"),
        ("screen_w", "screen_h"),
    ):
        try:
            w = getattr(input_ctrl, w_attr, None)
            h = getattr(input_ctrl, h_attr, None)
            if w and h and int(w) > 0 and int(h) > 0:
                return int(w), int(h)
        except (TypeError, ValueError):
            continue
    for meth in ("_cached_screen_resolution", "get_screen_resolution"):
        fn = getattr(input_ctrl, meth, None)
        if callable(fn):
            try:
                w, h = fn()
                if w and h and int(w) > 0 and int(h) > 0:
                    return int(w), int(h)
            except Exception:
                continue
    return None


def _click_normalized_safe(
    input_ctrl: Any,
    nx: float,
    ny: float,
    test_mode: bool = False,
    double: bool = False,
    screen_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Çözünürlükten bağımsız normalize koordinata `(nx, ny)` güvenli tıklama yapar.
    NOT: Bu tıklamalar oyun içi arayüz (karakter seçimi / oyuna gir) tıklamasıdır,
    skill saldırısı değildir; R/T kuralı skill atışına uygulanır, arayüz
    tıklaması fareyle kalır.
    Üretimde (`test_mode=False`) kontrolcü yoksa veya kontrolcü hiçbir tıklama
    metoduna sahip değilse RuntimeError yükseltir — sessizce "tıklandı"
    sayılmaz; çağıran adımın başarısız olduğunu bilmelidir (F-1709).
    S-0802: tıklama metodunun dönüş değeri denetlenir — `None`/`False`/`0`
    dönen tıklama artık varsayılan koordinatla başarı sayılmaz, RuntimeError
    yükseltir; kontrolcü açıkça güvenilir donanım fare yolunun kapalı olduğunu
    bildiriyorsa (`_controller_mouse_trusted` False) tıklama hiç gönderilmez
    (yumuşak düşüş YOK); güven bilgisi sunmayan kontrolcüde (bilinmiyor)
    tıklama engellenmez.
    """
    clamped_nx = max(0.0, min(1.0, float(nx)))
    clamped_ny = max(0.0, min(1.0, float(ny)))
    # P1-7 (K-20): piksel yedeği çözünürlükten okunur (2560x1440'da
    # 1920x1080 pikseli yanlış olurdu). Son-çare 1920x1080 yalnızca
    # çözünürlük hiçbir yerden okunamadığında kullanılır.
    _res = screen_size or _resolve_click_screen_size(input_ctrl)
    if _res is None:
        logger.debug(
            "Normalize tıklama: çözünürlük okunamadı, 1920x1080 "
            "son-çare yedeği kullanılıyor."
        )
        _res = (1920, 1080)
    default_px = int(round(clamped_nx * _res[0]))
    default_py = int(round(clamped_ny * _res[1]))

    if input_ctrl is None:
        if test_mode:
            logger.info(
                "[Simülasyon] Normalize tıklama: (%.3f, %.3f) -> (%d, %d) double=%s",
                clamped_nx,
                clamped_ny,
                default_px,
                default_py,
                double,
            )
            return default_px, default_py
        raise RuntimeError("Normalize tıklama yapılamadı: girdi kontrolcüsü yok ve test_mode kapalı")

    if not test_mode:
        mouse_trusted = _controller_mouse_trusted(input_ctrl)
        if mouse_trusted is False:
            logger.error(
                "Normalize tıklama gönderilmedi (%.3f, %.3f): güvenilir donanım fare yolu "
                "(Interception/HID) kapalı; yumuşak girdi AION 2'de yok sayılır",
                clamped_nx,
                clamped_ny,
            )
            raise RuntimeError("Normalize tıklama yapılamadı: güvenilir donanım fare yolu kapalı")

    try:
        if hasattr(input_ctrl, "set_test_mode") and isinstance(test_mode, bool):
            input_ctrl.set_test_mode(test_mode)

        if hasattr(input_ctrl, "click_normalized"):
            res = input_ctrl.click_normalized(clamped_nx, clamped_ny, button="left", double=double)
            return _click_result_coords(res, default_px, default_py)

        if hasattr(input_ctrl, "human_click"):
            res = input_ctrl.human_click(x=default_px, y=default_py, button="left", double=double)
            return _click_result_coords(res, default_px, default_py)

        if hasattr(input_ctrl, "click_map"):
            res = input_ctrl.click_map(default_px, default_py, button="left", double=double)
            return _click_result_coords(res, default_px, default_py)
    except Exception as exc:
        logger.error("Normalize tıklama hatası (%.3f, %.3f): %s", clamped_nx, clamped_ny, exc)
        raise

    raise RuntimeError("Normalize tıklama yapılamadı: kontrolcü hiçbir tıklama metoduna sahip değil")


def _parse_hhmm_to_minutes(hhmm: str) -> int:
    """`'HH:MM'` formatındaki saat dizgesini gün içi dakikaya (0..1439) çevirir."""
    raw = str(hhmm or "00:00").strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Geçersiz saat formatı (HH:MM bekleniyor): {hhmm!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Saat/dakika sınır dışı (00:00..23:59): {hhmm!r}")
    return hour * 60 + minute


def _backup_corrupt_config(config_path: str) -> Optional[str]:
    """Bozuk yapılandırma dosyasını üzerine yazmak yerine zaman damgalı yedeğe taşır."""
    try:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = f"{config_path}.corrupt-{stamp}"
        os.replace(config_path, backup_path)
        logger.warning("Bozuk yapılandırma yedeğe taşındı: %s -> %s", config_path, backup_path)
        return backup_path
    except Exception as exc:
        logger.error("Bozuk yapılandırma yedeklenemedi (%s): %s", config_path, exc)
        return None


# ============================================================================
# 1. ÇOKLU KARAKTER YÖNETİCİSİ (ALT CHARACTER MANAGER)
# ============================================================================

@dataclass
class CharacterProfile:
    """
    Oyun içi tek bir karakterin (Ana veya Yan Karakter / Alt-Char) toplama profilini,
    atanmış bölgesini, rotasını ve günlük hasat kotasını temsil eder.
    """

    slot_index: int
    name: str
    level: int = 50
    region_name: str = "Verteron"
    route_name: str = "All"
    daily_gather_limit: int = 400
    gathered_today: int = 0
    energy_depleted: bool = False
    last_rotated_ts: float = 0.0
    last_reset_date: str = field(default_factory=lambda: datetime.date.today().isoformat())

    def __post_init__(self) -> None:
        self.slot_index = max(1, int(self.slot_index))
        self.name = str(self.name or f"Karakter_{self.slot_index}").strip()
        self.level = max(1, int(self.level))
        self.region_name = str(self.region_name or "Verteron").strip() or "Verteron"
        self.route_name = str(self.route_name or "All").strip() or "All"
        self.daily_gather_limit = max(1, int(self.daily_gather_limit))
        self.gathered_today = max(0, int(self.gathered_today))
        self.energy_depleted = bool(self.energy_depleted)
        self.last_rotated_ts = float(self.last_rotated_ts or 0.0)
        self.last_reset_date = str(self.last_reset_date or datetime.date.today().isoformat())

    @property
    def remaining_quota(self) -> int:
        """Karakterin kalan günlük toplama hakkını döndürür."""
        if self.energy_depleted:
            return 0
        return max(0, int(self.daily_gather_limit) - int(self.gathered_today))

    def is_available(self) -> bool:
        """Karakterin günlük toplama limiti dolmamış ve enerjisi tükenmemişse True döner."""
        return (not self.energy_depleted) and (self.gathered_today < self.daily_gather_limit)

    def to_dict(self) -> Dict[str, Any]:
        """JSON serileştirme için sözlük temsili."""
        return {
            "slot_index": int(self.slot_index),
            "name": str(self.name),
            "level": int(self.level),
            "region_name": str(self.region_name),
            "route_name": str(self.route_name),
            "daily_gather_limit": int(self.daily_gather_limit),
            "gathered_today": int(self.gathered_today),
            "energy_depleted": bool(self.energy_depleted),
            "last_rotated_ts": float(self.last_rotated_ts),
            "last_reset_date": str(self.last_reset_date),
            "remaining_quota": int(self.remaining_quota),
            "available": bool(self.is_available()),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        """Sözlükten `CharacterProfile` nesnesi üretir."""
        return cls(
            slot_index=int(data.get("slot_index", 1)),
            name=str(data.get("name", "Karakter_1")),
            level=int(data.get("level", 50)),
            region_name=str(data.get("region_name", "Verteron")),
            route_name=str(data.get("route_name", "All")),
            daily_gather_limit=int(data.get("daily_gather_limit", 400)),
            gathered_today=int(data.get("gathered_today", 0)),
            energy_depleted=bool(data.get("energy_depleted", False)),
            last_rotated_ts=float(data.get("last_rotated_ts", 0.0)),
            last_reset_date=str(data.get("last_reset_date", datetime.date.today().isoformat())),
        )


class AltCharacterManager:
    """
    Günlük Toplama Limiti Dolunca Otomatik Yan Karaktere Geçiş Yöneticisi.

    Özellikler:
    - Karakter listesini (`CharacterProfile`) `config/characters.json` altında kalıcı olarak saklar.
    - Aktif karakterin günlük toplama sayısını (`current_gathered`) ve enerji durumunu
      (`energy_depleted`) izler.
    - Günlük limit dolduğunda veya enerji tükendiğinde:
      1. `Esc -> Karakter Seçimi -> Slot #N Seç -> Oyuna Gir` güvenli çıkış/giriş dizisini çalıştırır.
      2. Aktif karakter profilini sıradaki uygun yan karaktere günceller.
      3. `RouteManager` üzerindeki aktif bölge (`region_name`) ve rotayı (`route_name`) yeni karakterin
         atanmış bölge ve rotasına otomatik geçirir.
    """

    # Oyun içi arayüz normalize koordinatları (1920x1080 & 2560x1440 bağımsız)
    CHAR_SELECT_BTN_NORM: Tuple[float, float] = (0.50, 0.56)
    CONFIRM_LOGOUT_BTN_NORM: Tuple[float, float] = (0.46, 0.55)
    ENTER_WORLD_BTN_NORM: Tuple[float, float] = (0.50, 0.90)
    SLOT_LIST_X_NORM: float = 0.86
    SLOT_FIRST_Y_NORM: float = 0.22
    SLOT_STEP_Y_NORM: float = 0.085

    def __init__(
        self,
        config_path: str = "config/characters.json",
        profiles: Optional[Sequence[Union[CharacterProfile, Dict[str, Any]]]] = None,
        active_slot_index: int = 1,
        step_delay_sec: float = 0.65,
    ) -> None:
        self._lock = threading.RLock()
        self.config_path: str = str(config_path or "config/characters.json")
        self.step_delay_sec: float = max(0.0, float(step_delay_sec))
        self.active_slot_index: int = max(1, int(active_slot_index))
        self.rotation_history: List[Dict[str, Any]] = []
        self._profiles: List[CharacterProfile] = []
        # F-0143 üretim bağlantısı: state_machine/main tek çağrıyla bu bağlamı
        # bağlar (bind_runtime_context); bağlıysa record_gather her toplamada
        # üç yöneticiyi de tetikler. Bağlı değilse None kalır ve record_gather
        # eski davranışıyla yalnızca sayaç artırır.
        self._runtime_input_ctrl: Any = None
        self._runtime_route_mgr: Any = None
        self._runtime_task_scheduler: Any = None
        self._runtime_pet_buff_mgr: Any = None
        self._runtime_frame_provider: Any = None
        self._runtime_test_mode: bool = False
        self._last_wired_tick: Optional[Dict[str, Any]] = None

        if profiles is not None:
            for item in profiles:
                if isinstance(item, CharacterProfile):
                    self._profiles.append(item)
                elif isinstance(item, dict):
                    self._profiles.append(CharacterProfile.from_dict(item))
            self._sort_and_normalize_profiles()
            # Profiller çağrıcı tarafından verildi: kalıcılık çağrıcınındır,
            # kurulum tek başına dosyayı yazmaz (F-1736).
        else:
            self.load_characters()

    def _sort_and_normalize_profiles(self) -> None:
        """Karakter listesini `slot_index` sırasına göre düzenler ve aktif slotu doğrular."""
        # Aynı slotta yinelenen profil varsa son gelen kazanır (tek kayıt kalır).
        by_slot: Dict[int, CharacterProfile] = {}
        for prof in self._profiles:
            by_slot[int(prof.slot_index)] = prof
        self._profiles = sorted(by_slot.values(), key=lambda c: c.slot_index)
        if not self._profiles:
            self._profiles = self._build_default_profiles()
        valid_slots = [c.slot_index for c in self._profiles]
        if self.active_slot_index not in valid_slots:
            self.active_slot_index = valid_slots[0]

    @staticmethod
    def _build_default_profiles() -> List[CharacterProfile]:
        """Varsayılan 3 karakterlik çoklu karakter kadrosunu oluşturur."""
        today_str = datetime.date.today().isoformat()
        return [
            CharacterProfile(
                slot_index=1,
                name="AnaToplayici",
                level=55,
                region_name="Verteron",
                route_name="All",
                daily_gather_limit=400,
                gathered_today=0,
                energy_depleted=False,
                last_reset_date=today_str,
            ),
            CharacterProfile(
                slot_index=2,
                name="YanKarakter_1",
                level=52,
                region_name="Altgard",
                route_name="All",
                daily_gather_limit=400,
                gathered_today=0,
                energy_depleted=False,
                last_reset_date=today_str,
            ),
            CharacterProfile(
                slot_index=3,
                name="YanKarakter_2",
                level=50,
                region_name="Eltnen",
                route_name="All",
                daily_gather_limit=400,
                gathered_today=0,
                energy_depleted=False,
                last_reset_date=today_str,
            ),
        ]

    # ------------------------------------------------------------------
    # Disk Kalıcılığı (`config/characters.json`)
    # ------------------------------------------------------------------
    def load_characters(self) -> List[CharacterProfile]:
        """`config/characters.json` dosyasından karakter kadrosunu yükler."""
        with self._lock:
            corrupt = False
            if os.path.isfile(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    loaded: List[CharacterProfile] = []
                    chars_raw = raw.get("characters", []) if isinstance(raw, dict) else raw
                    if isinstance(chars_raw, list):
                        for item in chars_raw:
                            if isinstance(item, dict):
                                loaded.append(CharacterProfile.from_dict(item))
                    if isinstance(raw, dict) and "active_slot_index" in raw:
                        self.active_slot_index = max(1, int(raw["active_slot_index"]))
                    if loaded:
                        self._profiles = loaded
                        self._sort_and_normalize_profiles()
                        self.reset_daily_quotas_if_new_day()
                        return list(self._profiles)
                except Exception as exc:
                    logger.warning("Karakter yapılandırması okunamadı (%s): %s", self.config_path, exc)
                    corrupt = True
                    _backup_corrupt_config(self.config_path)

            self._profiles = self._build_default_profiles()
            self._sort_and_normalize_profiles()
            if not corrupt:
                self.save_characters()
            return list(self._profiles)

    def save_characters(self) -> bool:
        """Karakter kadrosunu ve aktif slot bilgisini `config/characters.json` dosyasına atomik yazar."""
        with self._lock:
            try:
                parent_dir = os.path.dirname(os.path.abspath(self.config_path))
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

                payload = {
                    "active_slot_index": int(self.active_slot_index),
                    "updated_at": time.time(),
                    "characters": [c.to_dict() for c in self._profiles],
                }
                tmp_path = f"{self.config_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.config_path)
                return True
            except Exception as exc:
                logger.error("Karakter yapılandırması kaydedilemedi (%s): %s", self.config_path, exc)
                return False

    # ------------------------------------------------------------------
    # Karakter Kadrosu (Roster) Sorgulama ve Güncelleme Metodları
    # ------------------------------------------------------------------
    def reset_daily_quotas_if_new_day(self, today_iso: Optional[str] = None) -> bool:
        """Yeni bir güne geçildiyse tüm karakterlerin günlük toplama sayaçlarını sıfırlar."""
        with self._lock:
            current_day = str(today_iso or datetime.date.today().isoformat())
            changed = False
            for profile in self._profiles:
                if profile.last_reset_date != current_day:
                    profile.gathered_today = 0
                    profile.energy_depleted = False
                    profile.last_reset_date = current_day
                    changed = True
            if changed:
                logger.info("Yeni gün (%s) algılandı: Tüm karakterlerin günlük toplama limitleri sıfırlandı.", current_day)
                self.save_characters()
            return changed

    def reset_all_daily_quotas(self) -> None:
        """Tüm karakterlerin günlük toplama sayaçlarını ve enerji tükenme bayraklarını manuel sıfırlar."""
        with self._lock:
            today_str = datetime.date.today().isoformat()
            for profile in self._profiles:
                profile.gathered_today = 0
                profile.energy_depleted = False
                profile.last_reset_date = today_str
            self.save_characters()

    def record_gather(self, count: int = 1, material_name: Optional[str] = None) -> int:
        """
        Aktif karakterin günlük toplama sayacını artırır ve kalıcı kaydeder.
        `state_machine` her başarılı toplamada `record_gather(count=1)` çağırır;
        bu metot yoksa kota takibi sessizce hiç çalışmaz (hasattr koruması).
        Yeni güne geçildiyse önce sayaçlar sıfırlanır. Güncel sayacı döndürür.
        F-0143: üretim bağlamı bağlıysa (bind_runtime_context) sayaç artımından
        sonra kilit DIŞINDA bağlı yöneticiler tetiklenir (takvim/rotasyon/buff);
        bağlam bağlı değilse yalnızca sayaç artar. Tetikleme hatası sayacı
        ve dönüş değerini etkilemez.
        """
        with self._lock:
            self.reset_daily_quotas_if_new_day()
            active = self.get_active_character()
            active.gathered_today = max(0, int(active.gathered_today) + max(0, int(count)))
            self.save_characters()
            total = int(active.gathered_today)
        # Kilit dışında: donanım G/Ç yapan yönetici tetiklemesi kilit altında yapılmaz (F-1731).
        self._run_wired_tick(total)
        return total

    def bind_runtime_context(
        self,
        input_ctrl: Any = None,
        route_mgr: Any = None,
        task_scheduler: Any = None,
        pet_buff_mgr: Any = None,
        frame_provider: Any = None,
        test_mode: bool = False,
    ) -> None:
        """
        F-0143 üretim bağlantısı: `state_machine`/`main` bu metodu bir kez
        çağırarak üç yöneticiyi birbirine bağlar; sonrasında üretimde zaten
        çalışan tek çağrı (`record_gather`, state_machine.py) her toplamada
        takvim/rotasyon/buff yöneticilerini tetikler. `frame_provider`
        çağrılabilirse buff denetimine güncel kare verir (None dönebilir).
        Bağlam bağlanana kadar hiçbir davranış değişmez.
        """
        with self._lock:
            self._runtime_input_ctrl = input_ctrl
            self._runtime_route_mgr = route_mgr
            self._runtime_task_scheduler = task_scheduler
            self._runtime_pet_buff_mgr = pet_buff_mgr
            self._runtime_frame_provider = frame_provider
            self._runtime_test_mode = bool(test_mode)

    def _run_wired_tick(self, current_gathered: int) -> Optional[Dict[str, Any]]:
        """Bağlı üretim bağlamı varsa `tick_multi_char_managers` çalıştırır; yoksa None döner. Asla yükseltmez."""
        with self._lock:
            input_ctrl = self._runtime_input_ctrl
            route_mgr = self._runtime_route_mgr
            task_scheduler = self._runtime_task_scheduler
            pet_buff_mgr = self._runtime_pet_buff_mgr
            frame_provider = self._runtime_frame_provider
            test_mode = self._runtime_test_mode
        if input_ctrl is None and task_scheduler is None and pet_buff_mgr is None:
            return None
        frame = None
        if callable(frame_provider):
            try:
                frame = frame_provider()
            except Exception as exc:
                logger.debug("Çerçeve sağlayıcı hatası (buff denetimi karesiz çalışır): %s", exc)
                frame = None
        try:
            result = tick_multi_char_managers(
                alt_char_mgr=self,
                task_scheduler=task_scheduler,
                pet_buff_mgr=pet_buff_mgr,
                current_gathered=int(current_gathered),
                input_ctrl=input_ctrl,
                route_mgr=route_mgr,
                frame=frame,
                test_mode=test_mode,
            )
        except Exception as exc:
            logger.error("Bağlı yönetici tick hatası: %s", exc)
            return None
        with self._lock:
            self._last_wired_tick = result
        return result

    def get_roster(self) -> List[CharacterProfile]:
        """Kayıtlı tüm karakter profillerinin kopyasını döndürür."""
        with self._lock:
            return list(self._profiles)

    def get_character_by_slot(self, slot_index: int) -> Optional[CharacterProfile]:
        """Belirtilen slot numarasındaki karakter profilini döndürür."""
        with self._lock:
            target = int(slot_index)
            for profile in self._profiles:
                if profile.slot_index == target:
                    return profile
            return None

    def get_active_character(self) -> CharacterProfile:
        """Şu anda aktif olan karakterin `CharacterProfile` nesnesini döndürür."""
        with self._lock:
            # Hızlı yol: kadro yalnızca değiştiren metotlar üzerinden değişir ve
            # her değiştiren metot zaten `_sort_and_normalize_profiles` çağırır;
            # aktif slot tekil ise tam sıralama/tekilleştirme gerekmez.
            found: Optional[CharacterProfile] = None
            matches = 0
            for profile in self._profiles:
                if profile.slot_index == self.active_slot_index:
                    found = profile
                    matches += 1
            if matches == 1:
                return found  # type: ignore[return-value]
            # Aktif slot yoksa veya yineleniyorsa eski davranış: normalize et,
            # aktif slotu düzelt ve tekrar dene.
            self._sort_and_normalize_profiles()
            for profile in self._profiles:
                if profile.slot_index == self.active_slot_index:
                    return profile
            return self._profiles[0]

    def add_or_update_character(self, profile: Union[CharacterProfile, Dict[str, Any]]) -> CharacterProfile:
        """Yeni bir karakter profili ekler veya aynı `slot_index` mevcutsa günceller."""
        with self._lock:
            char_obj = profile if isinstance(profile, CharacterProfile) else CharacterProfile.from_dict(profile)
            for idx, existing in enumerate(self._profiles):
                if existing.slot_index == char_obj.slot_index:
                    self._profiles[idx] = char_obj
                    self._sort_and_normalize_profiles()
                    self.save_characters()
                    return char_obj
            self._profiles.append(char_obj)
            self._sort_and_normalize_profiles()
            self.save_characters()
            return char_obj

    def remove_character(self, slot_index: int) -> bool:
        """Belirtilen slottaki karakteri kadrodan kaldırır (en az 1 karakter korunur)."""
        with self._lock:
            if len(self._profiles) <= 1:
                logger.warning("Kadroda en az 1 karakter bulunmalıdır; silme işlemi iptal edildi.")
                return False
            target = int(slot_index)
            before = len(self._profiles)
            self._profiles = [c for c in self._profiles if c.slot_index != target]
            if len(self._profiles) < before:
                self._sort_and_normalize_profiles()
                self.save_characters()
                return True
            return False

    def get_slot_normalized_coords(self, slot_index: int) -> Tuple[float, float]:
        """Karakter seçimi ekranında `slot_index` (1..8) için normalize `(nx, ny)` koordinatını hesaplar."""
        slot = int(slot_index)
        if slot < 1 or slot > 8:
            # 9. ve üstü slotlar sessizce 8. slotun koordinatını almamalı:
            # yanlış karakter seçilir. Aralık dışı slot açıkça hatadır.
            raise ValueError(f"Karakter slotu 1..8 olmalı, gelen: {slot_index!r}")
        idx_zero = slot - 1
        nx = self.SLOT_LIST_X_NORM
        ny = min(0.88, self.SLOT_FIRST_Y_NORM + idx_zero * self.SLOT_STEP_Y_NORM)
        return round(nx, 4), round(ny, 4)

    def _find_next_available_character(self, exclude_slot: Optional[int] = None) -> Optional[CharacterProfile]:
        """
        Aktif karakterin slotundan başlayarak döngüsel sırada günlük kotası dolmamış
        ve enerjisi bitmemiş sıradaki uygun karakteri bulur.
        """
        with self._lock:
            if not self._profiles:
                return None

            ordered = sorted(self._profiles, key=lambda c: c.slot_index)
            start_pos = 0
            for i, prof in enumerate(ordered):
                if prof.slot_index == self.active_slot_index:
                    start_pos = i
                    break

            n = len(ordered)
            for offset in range(1, n + 1):
                candidate = ordered[(start_pos + offset) % n]
                if exclude_slot is not None and candidate.slot_index == exclude_slot:
                    continue
                if candidate.is_available():
                    return candidate
            return None

    # ------------------------------------------------------------------
    # Güvenli Karakter Değiştirme ve RouteManager Senkronizasyonu
    # ------------------------------------------------------------------
    def _switch_route_manager_to_character(
        self,
        route_mgr: Any,
        character: CharacterProfile,
    ) -> bool:
        """
        `RouteManager` nesnesini yeni karakterin `region_name` ve `route_name` değerlerine geçirir.
        Bölge veya rota henüz yoksa otomatik oluşturur.
        """
        if route_mgr is None:
            return False

        target_region = character.region_name
        target_route = character.route_name

        try:
            if hasattr(route_mgr, "get_regions") and hasattr(route_mgr, "create_region"):
                existing_regions = route_mgr.get_regions()
                if target_region not in existing_regions:
                    route_mgr.create_region(target_region)

            if hasattr(route_mgr, "get_routes") and hasattr(route_mgr, "create_route"):
                existing_routes = route_mgr.get_routes(target_region)
                if target_route not in existing_routes:
                    route_mgr.create_route(target_region, target_route)

            if hasattr(route_mgr, "active_region"):
                route_mgr.active_region = target_region
            if hasattr(route_mgr, "active_route"):
                route_mgr.active_route = target_route

            if hasattr(route_mgr, "_sync_legacy_data") and callable(route_mgr._sync_legacy_data):
                route_mgr._sync_legacy_data()
            elif hasattr(route_mgr, "data") and isinstance(route_mgr.data, dict):
                route_mgr.data["active_region"] = target_region
                route_mgr.data["active_route"] = target_route

            logger.info(
                "[AltCharacterManager] RouteManager yeni karaktere senkronize edildi: Karakter=%s (Slot #%d) -> Bölge='%s', Rota='%s'",
                character.name,
                character.slot_index,
                target_region,
                target_route,
            )
            return True
        except Exception as exc:
            logger.error("RouteManager bölge/rota geçiş hatası (%s/%s): %s", target_region, target_route, exc)
            return False

    def execute_character_switch_sequence(
        self,
        target_slot_index: int,
        input_ctrl: Any = None,
        test_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Güvenli karakter seçim ekranına çıkış ve yan karaktere giriş dizisini çalıştırır:
        1. `Esc` -> Sistem menüsünü aç
        2. `Karakter Seçimi` butonuna tıkla + Onay (`Tamam`) tıkla
        3. Karakter seçim ekranında `Slot #N` üzerine tıkla
        4. `Oyuna Gir` butonuna tıkla
        Her adım `ok` bayrağıyla döner; bir adım gönderilemezse dizi durur ve
        kalan adımlar `ok=False, skipped=True` işaretlenir. Çağıran, tüm adımlar
        `ok` değilse karakter değişimini kalıcı kayda GEÇİREMEZ (F-1709).
        Hedef slot kadroda yoksa veya 1..8 dışında ise hiçbir girdi
        gönderilmeden ValueError yükseltilir.
        """
        target_slot = int(target_slot_index)
        if self.get_character_by_slot(target_slot) is None:
            raise ValueError(f"Hedef slot kadroda yok: Slot #{target_slot}")
        slot_nx, slot_ny = self.get_slot_normalized_coords(target_slot)

        pause_dt = 0.01 if test_mode else self.step_delay_sec
        steps_log: List[Dict[str, Any]] = []
        failed = False

        def _record(step_no: int, action: str, description_tr: str, ok: bool, **extra: Any) -> None:
            entry: Dict[str, Any] = {
                "step": step_no,
                "action": action,
                "description_tr": description_tr,
                "ok": bool(ok),
            }
            entry.update(extra)
            steps_log.append(entry)

        # Adım 1: ESC ile Sistem Menüsünü Aç
        ok1 = _send_key_safe(input_ctrl, "esc", test_mode=test_mode)
        _record(1, "open_system_menu", "Esc tuşu ile oyun sistem menüsü açıldı", ok1, key="esc")
        if not ok1:
            failed = True
        # Adım başarısızsa sonraki adımlar atlanacak; arayüzün oturması için
        # beklemeye gerek yok — başarısız dizi hızlı dönsün.
        if pause_dt > 0 and not failed:
            time.sleep(pause_dt)

        # Adım 2: Karakter Seçimi Butonuna Tıkla ve Onayla
        if not failed:
            try:
                cs_px, cs_py = _click_normalized_safe(
                    input_ctrl,
                    self.CHAR_SELECT_BTN_NORM[0],
                    self.CHAR_SELECT_BTN_NORM[1],
                    test_mode=test_mode,
                )
                if pause_dt > 0:
                    time.sleep(pause_dt * 0.5)
                cf_px, cf_py = _click_normalized_safe(
                    input_ctrl,
                    self.CONFIRM_LOGOUT_BTN_NORM[0],
                    self.CONFIRM_LOGOUT_BTN_NORM[1],
                    test_mode=test_mode,
                )
                _record(
                    2, "click_character_selection",
                    "Karakter Seçimi ekranına çıkış onaylandı", True,
                    coords=(cs_px, cs_py), confirm_coords=(cf_px, cf_py),
                )
            except Exception as exc:
                logger.error("Karakter seçimi adımı başarısız: %s", exc)
                _record(2, "click_character_selection", "Karakter Seçimi adımı gönderilemedi", False,
                        error=str(exc))
                failed = True
        else:
            _record(2, "click_character_selection", "Önceki adım başarısız olduğu için atlandı", False,
                    skipped=True)
        if pause_dt > 0 and not failed:
            time.sleep(pause_dt)

        # Adım 3: Hedef Karakter Slotunu Seç (Slot #N)
        if not failed:
            try:
                slot_px, slot_py = _click_normalized_safe(
                    input_ctrl, slot_nx, slot_ny, test_mode=test_mode, double=False,
                )
                _record(
                    3, f"select_slot_{target_slot}",
                    f"Slot #{target_slot} seçildi", True,
                    slot_index=target_slot, norm_coords=(slot_nx, slot_ny),
                    coords=(slot_px, slot_py),
                )
            except Exception as exc:
                logger.error("Slot seçimi adımı başarısız: %s", exc)
                _record(3, f"select_slot_{target_slot}", "Slot seçimi gönderilemedi", False,
                        slot_index=target_slot, error=str(exc))
                failed = True
        else:
            _record(3, f"select_slot_{target_slot}", "Önceki adım başarısız olduğu için atlandı", False,
                    slot_index=target_slot, skipped=True)
        if pause_dt > 0 and not failed:
            time.sleep(pause_dt)

        # Adım 4: Oyuna Gir (Enter World) — yalnızca tıklama; ardından Enter
        # basmak çifte aksiyon olur (F-1734), o yüzden Enter gönderilmez.
        if not failed:
            try:
                enter_px, enter_py = _click_normalized_safe(
                    input_ctrl,
                    self.ENTER_WORLD_BTN_NORM[0],
                    self.ENTER_WORLD_BTN_NORM[1],
                    test_mode=test_mode,
                )
                _record(
                    4, "enter_game_world",
                    "Oyuna Gir butonuna tıklandı ve yeni karakterle dünyaya giriş yapıldı", True,
                    coords=(enter_px, enter_py),
                )
            except Exception as exc:
                logger.error("Oyuna giriş adımı başarısız: %s", exc)
                _record(4, "enter_game_world", "Oyuna giriş tıklaması gönderilemedi", False,
                        error=str(exc))
                failed = True
        else:
            _record(4, "enter_game_world", "Önceki adım başarısız olduğu için atlandı", False,
                    skipped=True)

        return steps_log

    def check_and_rotate_character(
        self,
        current_gathered: int,
        energy_depleted: bool = False,
        input_ctrl: Any = None,
        route_mgr: Any = None,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Aktif karakterin günlük toplama sayısını (`current_gathered`) ve enerji durumunu
        (`energy_depleted`) kontrol eder.

        Limit dolmuşsa veya enerji tükenmişse:
        - `Esc -> Karakter Seçimi -> Slot #N Seç -> Oyuna Gir` dizisini çalıştırır,
        - Dizi adımlarının HEPSİ doğrulandıysa aktif karakter profilini sıradaki
          uygun karaktere günceller ve kaydeder (doğrulanamayan değişim kalıcı
          kayda geçmez, F-1709),
        - `route_mgr` varsa yeni karakterin `region_name` ve `route_name` değerlerine geçirir.
        Kilit yalnızca kısa karar/kayıt bölümlerinde tutulur; uykulu ve G/Ç yapan
        değişim dizisi kilit DIŞINDA çalışır (F-1731).
        """
        # 1. aşama (kilitli, kısa): kararı ver, kalıcı sayacı güncelle.
        with self._lock:
            self.reset_daily_quotas_if_new_day()
            active_char = self.get_active_character()

            if current_gathered is not None:
                active_char.gathered_today = max(0, int(current_gathered))
            if energy_depleted:
                active_char.energy_depleted = True

            self.save_characters()

            limit_reached = active_char.gathered_today >= active_char.daily_gather_limit
            should_rotate = limit_reached or active_char.energy_depleted

            if not should_rotate:
                return {
                    "rotated": False,
                    "all_exhausted": False,
                    "recommended_state": "WALKING",
                    "previous_character": active_char.to_dict(),
                    "active_character": active_char.to_dict(),
                    "route_switched": False,
                    "sequence_steps": [],
                    "reason_tr": (
                        f"Aktif karakter '{active_char.name}' (Slot #{active_char.slot_index}) "
                        f"kota dahilinde ({active_char.gathered_today}/{active_char.daily_gather_limit})."
                    ),
                }

            trigger_reason = (
                "özüt/toplama enerjisi tükendi"
                if active_char.energy_depleted and not limit_reached
                else f"günlük toplama limiti doldu ({active_char.gathered_today}/{active_char.daily_gather_limit})"
            )

            next_char = self._find_next_available_character(exclude_slot=active_char.slot_index)
            if next_char is None:
                msg_exhausted = (
                    f"Tüm karakterlerin günlük toplama limiti veya enerjisi doldu! "
                    f"(Son karakter: {active_char.name} Slot #{active_char.slot_index}, sebep: {trigger_reason})."
                )
                logger.warning("[AltCharacterManager] %s", msg_exhausted)
                return {
                    "rotated": False,
                    "all_exhausted": True,
                    "recommended_state": "PAUSED",
                    "previous_character": active_char.to_dict(),
                    "active_character": active_char.to_dict(),
                    "route_switched": False,
                    "sequence_steps": [],
                    "reason_tr": msg_exhausted,
                }

            if not 1 <= int(next_char.slot_index) <= 8:
                msg_slot = (
                    f"Yan karakter '{next_char.name}' Slot #{next_char.slot_index} arayüzde yok "
                    f"(1..8); yanlış slota tıklanmaması için rotasyon iptal edildi."
                )
                logger.warning("[AltCharacterManager] %s", msg_slot)
                return {
                    "rotated": False,
                    "all_exhausted": False,
                    "recommended_state": "WALKING",
                    "previous_character": active_char.to_dict(),
                    "active_character": active_char.to_dict(),
                    "route_switched": False,
                    "sequence_steps": [],
                    "reason_tr": msg_slot,
                }

            prev_snapshot = active_char.to_dict()
            decision_slot = int(self.active_slot_index)

        # 2. aşama (kilitsiz): uykulu/G/Ç yapan değişim dizisi.
        try:
            sequence_steps = self.execute_character_switch_sequence(
                target_slot_index=next_char.slot_index,
                input_ctrl=input_ctrl,
                test_mode=test_mode,
            )
        except ValueError as exc:
            logger.error("[AltCharacterManager] Rotasyon iptal edildi: %s", exc)
            return {
                "rotated": False,
                "all_exhausted": False,
                "recommended_state": "WALKING",
                "previous_character": prev_snapshot,
                "active_character": active_char.to_dict(),
                "route_switched": False,
                "sequence_steps": [],
                "reason_tr": f"Karakter değişimi doğrulanamadı, kalıcı kayıt değiştirilmedi: {exc}",
            }

        steps_ok = bool(sequence_steps) and all(bool(st.get("ok")) for st in sequence_steps)
        if not steps_ok:
            failed_steps = [st.get("step") for st in sequence_steps if not st.get("ok")]
            return {
                "rotated": False,
                "all_exhausted": False,
                "recommended_state": "WALKING",
                "previous_character": prev_snapshot,
                "active_character": active_char.to_dict(),
                "route_switched": False,
                "sequence_steps": sequence_steps,
                "reason_tr": (
                    f"Karakter değişimi doğrulanamadı (başarısız adım: {failed_steps}); "
                    f"aktif karakter kaydı değiştirilmedi."
                ),
            }

        # 3. aşama (kilitli, kısa): doğrulanmış değişimi kalıcı kayda geçir.
        with self._lock:
            if int(self.active_slot_index) != decision_slot:
                return {
                    "rotated": False,
                    "all_exhausted": False,
                    "recommended_state": "WALKING",
                    "previous_character": prev_snapshot,
                    "active_character": active_char.to_dict(),
                    "route_switched": False,
                    "sequence_steps": sequence_steps,
                    "reason_tr": (
                        "Karakter değişimi sırasında aktif slot başka bir işlemle değişti; "
                        "kalıcı kayıt değiştirilmedi."
                    ),
                }
            now_ts = time.time()
            active_char.last_rotated_ts = now_ts
            next_char.last_rotated_ts = now_ts
            self.active_slot_index = next_char.slot_index
            self.save_characters()

        # 4. aşama (kilitsiz): rota senkronizasyonu.
        route_switched = self._switch_route_manager_to_character(route_mgr, next_char)

        reason_tr = (
            f"'{active_char.name}' (Slot #{active_char.slot_index}) için {trigger_reason}. "
            f"Yan karakter '{next_char.name}' (Slot #{next_char.slot_index}) seçildi -> "
            f"Bölge: {next_char.region_name}, Rota: {next_char.route_name}."
        )
        logger.info("[AltCharacterManager] %s", reason_tr)

        rotation_event = {
            "timestamp": now_ts,
            "from_slot": active_char.slot_index,
            "from_name": active_char.name,
            "to_slot": next_char.slot_index,
            "to_name": next_char.name,
            "region_name": next_char.region_name,
            "route_name": next_char.route_name,
            "route_switched": route_switched,
            "reason_tr": reason_tr,
        }
        with self._lock:
            self.rotation_history.append(rotation_event)

        return {
            "rotated": True,
            "all_exhausted": False,
            "recommended_state": "WALKING",
            "previous_character": prev_snapshot,
            "active_character": next_char.to_dict(),
            "target_region": next_char.region_name,
            "target_route": next_char.route_name,
            "route_switched": route_switched,
            "sequence_steps": sequence_steps,
            "reason_tr": reason_tr,
        }

    def get_scheduler_health_summary(self) -> Dict[str, Any]:
        """Returns a health summary of characters, active character name, and rotation history."""
        try:
            with self._lock:
                act = self.get_active_character()
                return {
                    "total_characters": len(self._profiles),
                    "available_characters": sum(1 for c in self._profiles if c.is_available()),
                    "active_character_name": act.name if act else "Main",
                    "active_slot_index": int(self.active_slot_index),
                    "rotation_count": len(self.rotation_history),
                }
        except Exception as exc:
            logger.debug("get_scheduler_health_summary warning: %s", exc)
            return {
                "total_characters": 0,
                "active_character_name": "Main",
                "active_slot_index": 1,
                "rotation_count": 0,
            }

    def _is_ready(self) -> bool:
        with self._lock:
            return bool(self._profiles) and any(
                c.slot_index == self.active_slot_index for c in self._profiles
            )

    def get_multi_char_scheduler_ready_flag(self) -> dict:
        ready = self._is_ready()
        return {
            "module": "multi_char_scheduler",
            "ready": ready,
            "total_characters": len(self._profiles),
            "active_slot_index": int(self.active_slot_index),
        }

    def get_multi_char_scheduler_version_info(self) -> dict:
        return {"module": "multi_char_scheduler", "version": __version__, "ready": self._is_ready()}

    def get_multi_char_scheduler_diagnostic_snapshot(self) -> dict:
        ready = self._is_ready()
        with self._lock:
            act = self.get_active_character() if self._profiles else None
            return {
                "module": "multi_char_scheduler",
                "status": "ok" if ready else "degraded",
                "ready": ready,
                "total_characters": len(self._profiles),
                "available_characters": sum(1 for c in self._profiles if c.is_available()),
                "active_character_name": act.name if act else "Main",
                "active_slot_index": int(self.active_slot_index),
                "rotation_count": len(self.rotation_history),
                "runtime_context_bound": bool(
                    self._runtime_input_ctrl is not None
                    or self._runtime_task_scheduler is not None
                    or self._runtime_pet_buff_mgr is not None
                ),
            }




# ============================================================================
# 2. HAFTALIK / GÜNLÜK OTOMASYON TAKVİMİ (AUTOMATION TASK SCHEDULER)
# ============================================================================

VALID_SCHEDULE_ACTIONS: Tuple[str, str, str, str] = (
    "run_route",
    "town_wait_event",
    "pause",
    "switch_region",
)


@dataclass
class ScheduleEntry:
    """
    Zamanlanmış görev takvimindeki tek bir zaman penceresi kuralını temsil eder.

    Desteklenen `action` değerleri:
    - `"run_route"`: Belirtilen saat aralığında toplama rotasını çalıştırır.
    - `"town_wait_event"`: Dünya Bossu (World Boss) veya Etkinlik saatinde şehre dönüp bekler.
    - `"pause"`: Belirtilen saat aralığında botu güvenli duraklatma (`PAUSED`) modunda tutar.
    - `"switch_region"`: Belirtilen saat aralığında `target_region` bölgesine geçiş yapar ve rotayı çalıştırır.
    """

    name: str
    start_hhmm: str
    end_hhmm: str
    action: str = "run_route"
    target_region: str = ""
    target_route: str = "All"
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    enabled: bool = True
    priority: int = 10
    # `__post_init__` içinde bir kez hesaplanan önbellek: `matches_minutes` /
    # `window_duration_minutes` her çağrıda "HH:MM" ayrıştırmasın diye.
    # (Repo içinde `start_hhmm`/`end_hhmm`/`days_of_week` kurulum sonrası
    # değiştirilmiyor; değiştirilirse önbellek tazelenmez.)
    _start_min: int = field(default=0, init=False, repr=False, compare=False)
    _end_min: int = field(default=0, init=False, repr=False, compare=False)
    _days_set: frozenset = field(default=frozenset(), init=False, repr=False, compare=False)
    _overnight_allowed_days: frozenset = field(default=frozenset(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.name = str(self.name or "Görev").strip()
        self.start_hhmm = str(self.start_hhmm or "00:00").strip()
        self.end_hhmm = str(self.end_hhmm or "23:59").strip()
        self._start_min = _parse_hhmm_to_minutes(self.start_hhmm)
        self._end_min = _parse_hhmm_to_minutes(self.end_hhmm)

        clean_action = str(self.action or "run_route").strip().lower()
        if clean_action not in VALID_SCHEDULE_ACTIONS:
            raise ValueError(
                f"Geçersiz takvim aksiyonu: {self.action!r}. Geçerli değerler: {VALID_SCHEDULE_ACTIONS}"
            )
        self.action = clean_action
        self.target_region = str(self.target_region or "").strip()
        self.target_route = str(self.target_route or "All").strip() or "All"
        if not self.days_of_week:
            self.days_of_week = [0, 1, 2, 3, 4, 5, 6]
        else:
            self.days_of_week = sorted({int(d) % 7 for d in self.days_of_week})
        self._days_set = frozenset(self.days_of_week)
        self._overnight_allowed_days = self._days_set | frozenset((d + 1) % 7 for d in self.days_of_week)
        self.enabled = bool(self.enabled)
        self.priority = int(self.priority)

    def matches_minutes(self, current_minutes: int, weekday: int) -> bool:
        """Verilen gün (`0=Pazartesi .. 6=Pazar`) ve gün içi dakikada bu kuralın aktif olup olmadığını döner."""
        if not self.enabled:
            return False

        start_min = self._start_min
        end_min = self._end_min
        weekday = int(weekday) % 7
        if start_min > end_min:
            # Gece yarısını aşan pencerede gece yarısından sonraki kısım
            # pencerenin BAŞLADIĞI güne aittir (hafta sınırında önceki gün).
            if weekday not in self._overnight_allowed_days:
                return False
        elif weekday not in self._days_set:
            return False

        if start_min <= end_min:
            # Aynı gün içi pencere: başlangıç dahil, bitiş hariç (ancak 23:59 ise gün sonuna kadar dahil)
            if end_min == 23 * 60 + 59:
                return start_min <= current_minutes <= end_min
            return start_min <= current_minutes < end_min
        else:
            # Gece yarısını aşan pencere (ör. 22:15 -> 02:00): gece yarısından
            # sonraki kısım pencerenin BAŞLADIĞI güne aittir; hafta sınırında
            # önceki günün kuralı da geçerlidir.
            # Gece yarısı ÖNCESİ kısım yalnızca pencerenin başladığı gün
            # geçerlidir (F-1722): days=[0] (yalnızca Pazartesi) kuralı
            # Salı 23:00'te eşleşmemeli — +1 gün yalnızca 00:00 sonrası
            # kısım içindir.
            if current_minutes >= start_min:
                return weekday in self.days_of_week
            prev_weekday = (int(weekday) - 1) % 7
            return current_minutes < end_min and prev_weekday in self.days_of_week

    def window_duration_minutes(self) -> int:
        """Zaman penceresinin toplam dakika uzunluğunu döndürür (daha dar/özel pencerelere öncelik vermek için)."""
        start_min = self._start_min
        end_min = self._end_min
        if start_min <= end_min:
            return max(1, end_min - start_min)
        return max(1, (24 * 60 - start_min) + end_min)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_hhmm": self.start_hhmm,
            "end_hhmm": self.end_hhmm,
            "action": self.action,
            "target_region": self.target_region,
            "target_route": self.target_route,
            "days_of_week": list(self.days_of_week),
            "enabled": self.enabled,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleEntry":
        return cls(
            name=str(data.get("name", "Görev")),
            start_hhmm=str(data.get("start_hhmm", "00:00")),
            end_hhmm=str(data.get("end_hhmm", "23:59")),
            action=str(data.get("action", "run_route")),
            target_region=str(data.get("target_region", "")),
            target_route=str(data.get("target_route", "All")),
            days_of_week=list(data.get("days_of_week", [0, 1, 2, 3, 4, 5, 6])),
            enabled=bool(data.get("enabled", True)),
            priority=int(data.get("priority", 10)),
        )


class AutomationTaskScheduler:
    """
    Haftalık / Günlük Otomasyon Takvimi & World Boss / Etkinlik Molası Yöneticisi.

    Varsayılan yerleşik takvim:
    - 19:00 - 21:30: Rotayı başlat (`action="run_route"`)
    - 21:30 - 22:15: Dünya Bossu (World Boss) / Etkinlik saatinde şehre dönüp bekle (`action="town_wait_event"`)
    - 22:15 - 23:59: Toplamaya devam et (`action="run_route"`)
    """

    ACTION_TO_STATE: Dict[str, str] = {
        "run_route": "WALKING",
        "town_wait_event": "WAITING",
        "pause": "PAUSED",
        "switch_region": "WALKING",
    }

    def __init__(
        self,
        config_path: str = "config/schedule.json",
        entries: Optional[Sequence[Union[ScheduleEntry, Dict[str, Any]]]] = None,
        town_recall_key: str = "h",
    ) -> None:
        self._lock = threading.RLock()
        self.config_path: str = str(config_path or "config/schedule.json")
        self.town_recall_key: str = str(town_recall_key or "h").strip().lower()
        self._entries: List[ScheduleEntry] = []
        self._last_action: Optional[str] = None

        if entries is not None:
            for item in entries:
                if isinstance(item, ScheduleEntry):
                    self._entries.append(item)
                elif isinstance(item, dict):
                    self._entries.append(ScheduleEntry.from_dict(item))
            # Kurallar çağrıcı tarafından verildi: kalıcılık çağrıcınındır,
            # kurulum tek başına dosyayı yazmaz (F-1736).
        else:
            self.load_schedule()

    @staticmethod
    def build_default_schedule() -> List[ScheduleEntry]:
        """
        Şartnamedeki örnek yerleşik takvimi oluşturur:
        "19:00'da rotayı başlat, 21:30'da Dünya Bossu (World Boss) / Etkinlik saatinde
        şehre dönüp bekle, 22:15'te toplamaya devam et"
        """
        return [
            ScheduleEntry(
                name="19:00 Akşam Rotası Başlangıcı",
                start_hhmm="19:00",
                end_hhmm="21:30",
                action="run_route",
                target_region="",
                target_route="All",
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                enabled=True,
                priority=10,
            ),
            ScheduleEntry(
                name="21:30 Dünya Bossu (World Boss) / Etkinlik Molası",
                start_hhmm="21:30",
                end_hhmm="22:15",
                action="town_wait_event",
                target_region="Town_Sanctum",
                target_route="All",
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                enabled=True,
                priority=30,
            ),
            ScheduleEntry(
                name="22:15 Gece Toplama Devamı",
                start_hhmm="22:15",
                end_hhmm="23:59",
                action="run_route",
                target_region="",
                target_route="All",
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                enabled=True,
                priority=10,
            ),
        ]

    def load_schedule(self) -> List[ScheduleEntry]:
        """`config/schedule.json` dosyasından zamanlanmış görev kurallarını yükler."""
        with self._lock:
            corrupt = False
            if os.path.isfile(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    raw_list = raw.get("entries", []) if isinstance(raw, dict) else raw
                    loaded: List[ScheduleEntry] = []
                    if isinstance(raw_list, list):
                        for item in raw_list:
                            if isinstance(item, dict):
                                loaded.append(ScheduleEntry.from_dict(item))
                    if loaded:
                        self._entries = loaded
                        return list(self._entries)
                except Exception as exc:
                    logger.warning("Takvim yapılandırması okunamadı (%s): %s", self.config_path, exc)
                    corrupt = True
                    _backup_corrupt_config(self.config_path)

            self._entries = self.build_default_schedule()
            if not corrupt:
                self.save_schedule()
            return list(self._entries)

    def save_schedule(self) -> bool:
        """Zamanlanmış görev listesini `config/schedule.json` dosyasına kaydeder."""
        with self._lock:
            try:
                parent_dir = os.path.dirname(os.path.abspath(self.config_path))
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                payload = {
                    "updated_at": time.time(),
                    "entries": [e.to_dict() for e in self._entries],
                }
                tmp_path = f"{self.config_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.config_path)
                return True
            except Exception as exc:
                logger.error("Takvim kaydedilemedi (%s): %s", self.config_path, exc)
                return False

    def get_entries(self) -> List[ScheduleEntry]:
        """Kayıtlı tüm takvim kurallarını döndürür."""
        with self._lock:
            return list(self._entries)

    def add_entry(self, entry: Union[ScheduleEntry, Dict[str, Any]]) -> ScheduleEntry:
        """Takvime yeni bir zaman penceresi kuralı ekler."""
        with self._lock:
            obj = entry if isinstance(entry, ScheduleEntry) else ScheduleEntry.from_dict(entry)
            self._entries.append(obj)
            self.save_schedule()
            return obj

    def clear_entries(self) -> None:
        """Tüm takvim kurallarını temizler."""
        with self._lock:
            self._entries.clear()
            self.save_schedule()

    @staticmethod
    def _coerce_datetime(now_dt: Optional[Union[datetime.datetime, datetime.time, str]]) -> datetime.datetime:
        """`datetime`, `time` veya `'HH:MM'` dizgesini `datetime.datetime` nesnesine dönüştürür."""
        if now_dt is None:
            return datetime.datetime.now()
        if isinstance(now_dt, datetime.datetime):
            return now_dt
        if isinstance(now_dt, datetime.time):
            today = datetime.date.today()
            return datetime.datetime.combine(today, now_dt)
        if isinstance(now_dt, str):
            clean = now_dt.strip()
            try:
                return datetime.datetime.fromisoformat(clean)
            except ValueError:
                mins = _parse_hhmm_to_minutes(clean)
                today = datetime.date.today()
                return datetime.datetime(today.year, today.month, today.day, mins // 60, mins % 60)
        raise TypeError(f"Desteklenmeyen zaman tipi: {type(now_dt)!r}")

    def evaluate_current_schedule(
        self,
        now_dt: Optional[Union[datetime.datetime, datetime.time, str]] = None,
        route_mgr: Any = None,
        input_ctrl: Any = None,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Verilen zamana (`now_dt`, varsayılan şu an) göre aktif takvim kuralını değerlendirir
        ve botun alması gereken durumu (`recommended_state`, `reason_tr`, `active_entry`) döndürür.

        Eğer aktif kural:
        - `"town_wait_event"` ise: `recommended_state = "WAITING"`, `town_wait_active = True` döner
          (ve ilk geçişte `input_ctrl` verilmişse şehre dönüş tuşunu tetikler).
        - `"switch_region"` ise: `route_mgr` verilmişse hedef bölgeye geçiş yapar ve `recommended_state = "WALKING"` döner.
        - `"pause"` ise: `recommended_state = "PAUSED"` döner.
        - `"run_route"` ise: `recommended_state = "WALKING"` döner.
        """
        # S-1202: kilit yalnizca kisa karar bolumunde tutulur; tus gonderimi
        # ve route_mgr G/C'si kilit DISINDA calisir (check_and_rotate_character
        # ile ayni 3 asamali desen, F-1731). Kilit altinda donanim G/C yapan
        # cagri diger thread'leri blokluyordu.
        # 1. asama (kilitli, kisa): eslesen kurali sec, karari ver.
        with self._lock:
            dt = self._coerce_datetime(now_dt)
            current_minutes = dt.hour * 60 + dt.minute
            current_hhmm = f"{dt.hour:02d}:{dt.minute:02d}"
            weekday = dt.weekday()

            matching: List[ScheduleEntry] = [
                entry for entry in self._entries if entry.matches_minutes(current_minutes, weekday)
            ]

            if not matching:
                return {
                    "matched": False,
                    "current_hhmm": current_hhmm,
                    "weekday": weekday,
                    "action": "pause",
                    "recommended_state": "PAUSED",
                    "town_wait_active": False,
                    "region_switched": False,
                    "target_region": "",
                    "active_entry": None,
                    "reason_tr": f"Saat {current_hhmm} için takvim kuralı yok; takvim dışı saatte bot duraklatıldı.",
                }

            # En yüksek öncelikli ve en spesifik (dar) zaman penceresine sahip kuralı seç
            matching.sort(key=lambda e: (-e.priority, e.window_duration_minutes()))
            winner = matching[0]
            rec_state = self.ACTION_TO_STATE.get(winner.action, "WALKING")
            town_wait_active = winner.action == "town_wait_event"
            action = winner.action
            target_region = winner.target_region
            target_route = winner.target_route
            winner_dict = winner.to_dict()

            # Salt-okuma ("sadece bakayım") çağrısı _last_action'ı değiştirmemeli:
            # aksi halde şehir dönüşü tuşu kalıcı olarak bastırılır (F-1720).
            # Yalnızca gerçekten eylem yapan çağrılar (input_ctrl/route_mgr verilmiş)
            # son aksiyonu günceller.
            is_action_call = input_ctrl is not None or route_mgr is not None
            need_town_recall = (
                action == "town_wait_event"
                and self._last_action != "town_wait_event"
                and input_ctrl is not None
            )
            # S-0801: tuş gönderimi başarısız olursa sahiplenme geri alınabilsin
            # diye önceki değer saklanır; başarısız gönderim kalıcı bastırmaya
            # dönüşmesin, sonraki çağrı tuşu tekrar denesin.
            prev_last_action = self._last_action
            if need_town_recall or is_action_call:
                # Şehir dönüşü geçişini kilit altında sahiplen: eşzamanlı
                # ikinci çağrı aynı tuşu tekrar göndermesin.
                self._last_action = action
            need_region_switch = (
                route_mgr is not None
                and bool(target_region)
                and action in ("switch_region", "run_route")
            )

            if action == "town_wait_event":
                reason_tr = (
                    f"[{winner.start_hhmm}-{winner.end_hhmm}] '{winner.name}': "
                    f"Dünya Bossu (World Boss) / Etkinlik saatinde şehre dönüp bekleniyor."
                )
            elif action == "pause":
                reason_tr = (
                    f"[{winner.start_hhmm}-{winner.end_hhmm}] '{winner.name}': "
                    f"Zamanlanmış mola/duraklatma aktif."
                )
            elif action == "switch_region":
                reason_tr = (
                    f"[{winner.start_hhmm}-{winner.end_hhmm}] '{winner.name}': "
                    f"Bölge '{target_region or 'Varsayılan'}' rotası çalıştırılıyor."
                )
            else:
                reason_tr = (
                    f"[{winner.start_hhmm}-{winner.end_hhmm}] '{winner.name}': "
                    f"Zamanlanmış toplama rotası aktif."
                )

        # 2. asama (kilitsiz): donanim G/C — sehir donusu tusu + rota gecisi.
        town_recall_sent = False
        town_recall_pending = False
        if need_town_recall:
            town_recall_sent = _send_key_safe(input_ctrl, self.town_recall_key, test_mode=test_mode)
            if not town_recall_sent:
                # S-0801: tuş gönderilemediyse sahiplenmeyi geri al — sonraki
                # çağrıda need_town_recall yeniden True olur ve tuş tekrar
                # denenir; bot etkinlik penceresi boyunca WAITING'de sahada
                # beklemez. Yalnızca hâlâ bizim sahiplenmemiz duruyorsa geri al.
                with self._lock:
                    if self._last_action == action:
                        self._last_action = prev_last_action
                town_recall_pending = True
                logger.warning(
                    "Şehir dönüşü tuşu gönderilemedi ('%s'); sonraki takvim "
                    "değerlendirmesinde tekrar denenecek.",
                    self.town_recall_key,
                )

        region_switched = False
        if need_region_switch:
            try:
                # S-0803: yalnızca gerçekten uygulanan bir geçiş True sayılır;
                # route_mgr hiçbir beklenen özniteliğe sahip değilse hiçbir
                # işlem yapılmaz ve region_switched False kalır.
                region_applied = False
                if action == "switch_region":
                    if hasattr(route_mgr, "get_regions") and hasattr(route_mgr, "create_region"):
                        if target_region not in route_mgr.get_regions():
                            route_mgr.create_region(target_region)
                        region_applied = True
                    if hasattr(route_mgr, "active_region"):
                        route_mgr.active_region = target_region
                        region_applied = True
                    if hasattr(route_mgr, "active_route") and target_route:
                        route_mgr.active_route = target_route
                        region_applied = True
                    if hasattr(route_mgr, "_sync_legacy_data") and callable(route_mgr._sync_legacy_data):
                        route_mgr._sync_legacy_data()
                        region_applied = True
                else:  # run_route
                    if hasattr(route_mgr, "active_region"):
                        route_mgr.active_region = target_region
                        region_applied = True
                    if hasattr(route_mgr, "active_route") and target_route:
                        route_mgr.active_route = target_route
                        region_applied = True
                region_switched = region_applied
            except Exception as exc:
                if action == "switch_region":
                    logger.error("Takvim bölge değiştirme hatası (%s): %s", target_region, exc)
                else:
                    logger.debug("Rota bölgesi güncellenemedi: %s", exc)

        return {
            "matched": True,
            "current_hhmm": current_hhmm,
            "weekday": weekday,
            "action": action,
            "recommended_state": rec_state,
            "town_wait_active": town_wait_active,
            "town_recall_sent": town_recall_sent,
            "town_recall_pending": town_recall_pending,
            "region_switched": region_switched,
            "target_region": target_region,
            "target_route": target_route,
            "active_entry": winner_dict,
            "reason_tr": reason_tr,
        }

    def next_scheduled_change(
        self,
        now_dt: Optional[Union[datetime.datetime, datetime.time, str]] = None,
    ) -> Dict[str, Any]:
        """
        Bir sonraki takvim geçişine (etkin kuralın başlangıcı veya bitişi) kadar
        olan süreyi döndürür — çağıran `evaluate_current_schedule` ile yoklama
        yapmak yerine bu ana kadar uyuyabilir.

        Döndürür: `has_next` (etkin kural varsa True), `at` (geçiş anı, ISO),
        `in_seconds` (kalan saniye), `transition` (`"start"`/`"end"`),
        `entry` (ilgili kuralın sözlüğü). Etkin kural yoksa `has_next=False`
        döner. Mevcut durumu değiştirmez (`_last_action` güncellenmez).
        """
        with self._lock:
            now = self._coerce_datetime(now_dt)
            best_dt: Optional[datetime.datetime] = None
            best_entry: Optional[ScheduleEntry] = None
            best_transition = ""

            for entry in self._entries:
                if not entry.enabled:
                    continue
                # -1: dün başlayan gece-yarısı-aşan pencerenin bugünkü bitişi
                # de bir sonraki geçiş olabilir; 0..7: bu hafta içi.
                for day_offset in range(-1, 8):
                    day = now.date() + datetime.timedelta(days=day_offset)
                    if day.weekday() not in entry._days_set:
                        continue
                    start_dt = datetime.datetime.combine(
                        day, datetime.time(entry._start_min // 60, entry._start_min % 60)
                    )
                    end_day = day + (
                        datetime.timedelta(days=1)
                        if entry._start_min > entry._end_min
                        else datetime.timedelta(0)
                    )
                    end_dt = datetime.datetime.combine(
                        end_day, datetime.time(entry._end_min // 60, entry._end_min % 60)
                    )
                    for cand_dt, transition in ((start_dt, "start"), (end_dt, "end")):
                        if cand_dt > now and (best_dt is None or cand_dt < best_dt):
                            best_dt = cand_dt
                            best_entry = entry
                            best_transition = transition

            if best_dt is None or best_entry is None:
                return {
                    "has_next": False,
                    "at": None,
                    "in_seconds": None,
                    "transition": None,
                    "entry": None,
                    "current_hhmm": f"{now.hour:02d}:{now.minute:02d}",
                    "weekday": now.weekday(),
                    "reason_tr": "Etkin takvim kuralı yok; bir sonraki takvim geçişi bilinmiyor.",
                }

            in_seconds = (best_dt - now).total_seconds()
            return {
                "has_next": True,
                "at": best_dt.isoformat(),
                "in_seconds": round(float(in_seconds), 1),
                "transition": best_transition,
                "entry": best_entry.to_dict(),
                "current_hhmm": f"{now.hour:02d}:{now.minute:02d}",
                "weekday": now.weekday(),
                "reason_tr": (
                    f"Bir sonraki takvim geçişi {best_dt.strftime('%H:%M')} "
                    f"({best_transition} — '{best_entry.name}'), "
                    f"{int(in_seconds // 60)} dk {int(in_seconds % 60)} sn sonra."
                ),
            }

    def _is_ready(self) -> bool:
        with self._lock:
            return bool(self._entries)

    def get_multi_char_scheduler_ready_flag(self) -> dict:
        return {
            "module": "multi_char_scheduler",
            "ready": self._is_ready(),
            "total_entries": len(self._entries),
            "enabled_entries": sum(1 for e in self._entries if e.enabled),
        }

    def get_multi_char_scheduler_version_info(self) -> dict:
        return {"module": "multi_char_scheduler", "version": __version__, "ready": self._is_ready()}

    def get_multi_char_scheduler_diagnostic_snapshot(self) -> dict:
        ready = self._is_ready()
        with self._lock:
            return {
                "module": "multi_char_scheduler",
                "status": "ok" if ready else "degraded",
                "ready": ready,
                "total_entries": len(self._entries),
                "enabled_entries": sum(1 for e in self._entries if e.enabled),
                "last_action": self._last_action,
            }




# ============================================================================
# 3. PET / YOLDAŞ BESLEME VE YEMEK/PARŞÖMEN BUFF YÖNETİCİSİ (PET & BUFF MANAGER)
# ============================================================================

@dataclass
class BuffSlot:
    """
    Oyun içi toplama hızı parşömeni, verim yemeği veya iksir buff yuvasını temsil eder.
    """

    name: str
    hotkey: str
    duration_sec: float = 1800.0
    last_used_ts: float = 0.0
    enabled: bool = True
    icon_roi_norm: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    min_saturation_threshold: float = 35.0
    min_brightness_threshold: float = 45.0

    def __post_init__(self) -> None:
        self.name = str(self.name or "Buff").strip()
        self.hotkey = str(self.hotkey or "7").strip().lower()
        self.duration_sec = max(1.0, float(self.duration_sec))
        self.last_used_ts = float(self.last_used_ts or 0.0)
        self.enabled = bool(self.enabled)
        if isinstance(self.icon_roi_norm, (list, tuple)) and len(self.icon_roi_norm) == 4:
            self.icon_roi_norm = (
                float(self.icon_roi_norm[0]),
                float(self.icon_roi_norm[1]),
                float(self.icon_roi_norm[2]),
                float(self.icon_roi_norm[3]),
            )
        else:
            self.icon_roi_norm = (0.0, 0.0, 0.0, 0.0)

    def remaining_sec(self, now_ts: Optional[float] = None) -> float:
        """Buff süresinin bitmesine kalan saniyeyi döndürür."""
        now = float(now_ts if now_ts is not None else time.time())
        if self.last_used_ts <= 0.0:
            # Hiç kullanılmadıysa süre bilinmiyor: zamanlayıcı tek başına
            # yenileme tetiklemez (F-1718); tam süre varsayılır, görsel kanıt gerekir.
            return float(self.duration_sec)
        elapsed = max(0.0, now - self.last_used_ts)
        return max(0.0, self.duration_sec - elapsed)

    def is_expired_by_timer(self, now_ts: Optional[float] = None) -> bool:
        """Zamanlayıcıya göre buff süresinin dolup dolmadığını kontrol eder."""
        if not self.enabled:
            return False
        if self.last_used_ts <= 0.0:
            # Kayıt yoksa süre doldu sayılmaz; açılışta koşulsuz tuş basılmaz.
            return False
        return self.remaining_sec(now_ts=now_ts) <= 0.0

    def to_dict(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hotkey": self.hotkey,
            "duration_sec": self.duration_sec,
            "last_used_ts": self.last_used_ts,
            "remaining_sec": round(self.remaining_sec(now_ts=now_ts), 1),
            "expired": self.is_expired_by_timer(now_ts=now_ts),
            "enabled": self.enabled,
            "icon_roi_norm": list(self.icon_roi_norm),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuffSlot":
        roi_raw = data.get("icon_roi_norm", (0.0, 0.0, 0.0, 0.0))
        return cls(
            name=str(data.get("name", "Buff")),
            hotkey=str(data.get("hotkey", "7")),
            duration_sec=float(data.get("duration_sec", 1800.0)),
            last_used_ts=float(data.get("last_used_ts", 0.0)),
            enabled=bool(data.get("enabled", True)),
            icon_roi_norm=tuple(roi_raw) if isinstance(roi_raw, (list, tuple)) and len(roi_raw) == 4 else (0.0, 0.0, 0.0, 0.0),
            min_saturation_threshold=float(data.get("min_saturation_threshold", 35.0)),
            min_brightness_threshold=float(data.get("min_brightness_threshold", 45.0)),
        )


class PetAndBuffManager:
    """
    Pet / Yoldaş Besleme ve Yemek/Parşömen Buff Yöneticisi.

    Özellikler:
    - Ganimet yoldaşı (Loot Pet) açlık zamanlayıcısını (`pet_feed_interval_sec = 1800.0`, `pet_key = "9"`)
      takip eder ve süresi geldiğinde otomatik besler.
    - Toplama hızı parşömeni (`"Toplama Hızı Parşömeni"`) ve verim yemeği (`"Verim Yemeği"`) gibi
      `BuffSlot` öğelerinin hem zamanlayıcılarını hem de ekran üzerindeki buff çubuğu (`BUFF_BAR_ROI_NORM`)
      görsel varlığını denetler.
    - Süresi dolan veya buff çubuğunda sönen/kaybolan buffları ilgili kısayol tuşuna basarak yeniler.
    """

    # Ekran üzerindeki buff çubuğu normalize bölgesi (x1, y1, x2, y2)
    BUFF_BAR_ROI_NORM: Tuple[float, float, float, float] = (0.28, 0.015, 0.58, 0.065)

    def __init__(
        self,
        pet_feed_interval_sec: float = 1800.0,
        pet_key: str = "9",
        pet_name: str = "Ganimet Yoldaşı (Loot Pet)",
        buff_slots: Optional[Sequence[Union[BuffSlot, Dict[str, Any]]]] = None,
        visual_recast_cooldown_sec: float = 15.0,
    ) -> None:
        self._lock = threading.RLock()
        self.pet_feed_interval_sec: float = max(10.0, float(pet_feed_interval_sec))
        self.pet_key: str = str(pet_key or "9").strip().lower()
        self.pet_name: str = str(pet_name or "Ganimet Yoldaşı (Loot Pet)")
        self.last_pet_fed_ts: float = 0.0
        self.pet_feed_count: int = 0
        self.visual_recast_cooldown_sec: float = max(1.0, float(visual_recast_cooldown_sec))

        self.buff_slots: List[BuffSlot] = []
        if buff_slots is not None:
            # S123 (v19/P3-8): görsel harita (`slot_visual_active`)
            # yuva adıyla anahtarlanıyor — aynı adlı iki yuva
            # çakışır, son yazan kazanırdı. Yinelenen ad
            # benzersizleştirilir ("Ad (2)"), uyarı loglanır.
            _seen_names: Dict[str, int] = {}
            for item in buff_slots:
                _slot = item if isinstance(item, BuffSlot) else (
                    BuffSlot.from_dict(item) if isinstance(item, dict) else None
                )
                if _slot is None:
                    continue
                _count = _seen_names.get(_slot.name, 0) + 1
                _seen_names[_slot.name] = _count
                if _count > 1:
                    _orig = _slot.name
                    _slot.name = f"{_orig} ({_count})"
                    logger.warning(
                        "Yinelenen buff yuvası adı '%s' -> '%s' olarak "
                        "benzersizleştirildi.", _orig, _slot.name
                    )
                self.buff_slots.append(_slot)
        else:
            self.buff_slots = self.build_default_buff_slots()

    @staticmethod
    def build_default_buff_slots() -> List[BuffSlot]:
        """Varsayılan toplama parşömeni ve verim yemeği buff yuvalarını döndürür."""
        return [
            BuffSlot(
                name="Toplama Hızı Parşömeni",
                hotkey="7",
                duration_sec=1800.0,
                last_used_ts=0.0,
                enabled=True,
                icon_roi_norm=(0.30, 0.020, 0.325, 0.055),
            ),
            BuffSlot(
                name="Verim Yemeği",
                hotkey="8",
                duration_sec=3600.0,
                last_used_ts=0.0,
                enabled=True,
                icon_roi_norm=(0.33, 0.020, 0.355, 0.055),
            ),
        ]

    def pet_remaining_sec(self, now_ts: Optional[float] = None) -> float:
        """Pet besleme süresinin dolmasına kalan saniyeyi döndürür."""
        now = float(now_ts if now_ts is not None else time.time())
        if self.last_pet_fed_ts <= 0.0:
            return 0.0
        elapsed = max(0.0, now - self.last_pet_fed_ts)
        return max(0.0, self.pet_feed_interval_sec - elapsed)

    def is_pet_hungry(self, now_ts: Optional[float] = None) -> bool:
        """Pet besleme zamanının gelip gelmediğini döndürür."""
        return self.pet_remaining_sec(now_ts=now_ts) <= 0.0

    def inspect_buff_bar(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Oyun ekranındaki (`frame`) buff çubuğu bölgesini OpenCV HSV renk/parlaklık analiziyle inceler.
        Her bir `BuffSlot` için ikonun ekranda aktif (parlak/renkli) olup olmadığını tespit eder.

        P2-3 (K-17): bu metod kilit TUTMAZ — OpenCV analizi
        (connectedComponents + slot basina cvtColor) yalnizca kisa
        kilitle kopyalanmis yuva listesi uzerinde calisir. Cagiran
        (check_and_refresh_buffs) da analizi 1. asama kilidinin
        disinda yapar; kilit yalnizca karar/kayit icin tutulur.
        """
        # Yuva listesi yalnizca __init__'te kurulur; kisa kilitli kopya
        # olasi bir eszamanli degisime karsi analizi sabit bir liste
        # uzerinde tutar. Yuva nesnelerinin salt-okunur alanlari
        # (name/icon_roi_norm/esikler) analizde kullanilir; last_used_ts
        # gibi degisken alanlar burada okunmaz.
        with self._lock:
            slots = list(self.buff_slots)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "frame_analyzed": False,
                "active_icon_count": 0,
                "mean_saturation": 0.0,
                "mean_brightness": 0.0,
                "slot_visual_active": {},
            }

        try:
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                bgr = frame

            h, w = bgr.shape[:2]
            bx1 = max(0, min(w - 1, int(round(self.BUFF_BAR_ROI_NORM[0] * w))))
            by1 = max(0, min(h - 1, int(round(self.BUFF_BAR_ROI_NORM[1] * h))))
            bx2 = max(bx1 + 1, min(w, int(round(self.BUFF_BAR_ROI_NORM[2] * w))))
            by2 = max(by1 + 1, min(h, int(round(self.BUFF_BAR_ROI_NORM[3] * h))))

            bar_crop = bgr[by1:by2, bx1:bx2]
            if bar_crop.size == 0:
                return {
                    "frame_analyzed": False,
                    "active_icon_count": 0,
                    "mean_saturation": 0.0,
                    "mean_brightness": 0.0,
                    "slot_visual_active": {},
                }

            bar_hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
            mean_sat = float(np.mean(bar_hsv[:, :, 1]))
            mean_val = float(np.mean(bar_hsv[:, :, 2]))

            # Buff çubuğundaki aktif ikon sayısını tahmin et (yüksek doygunluk ve parlaklık kümeleri)
            active_mask = ((bar_hsv[:, :, 1] >= 35) & (bar_hsv[:, :, 2] >= 45)).astype(np.uint8) * 255
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(active_mask, connectivity=8)
            active_icon_count = 0
            for lbl_idx in range(1, num_labels):
                area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
                if area >= 12:
                    active_icon_count += 1

            slot_visual_active: Dict[str, bool] = {}
            for idx, slot in enumerate(slots):
                rx1, ry1, rx2, ry2 = slot.icon_roi_norm
                if rx2 > rx1 and ry2 > ry1:
                    sx1 = max(0, min(w - 1, int(round(rx1 * w))))
                    sy1 = max(0, min(h - 1, int(round(ry1 * h))))
                    sx2 = max(sx1 + 1, min(w, int(round(rx2 * w))))
                    sy2 = max(sy1 + 1, min(h, int(round(ry2 * h))))
                    slot_crop = bgr[sy1:sy2, sx1:sx2]
                else:
                    # Otomatik dilimleme: buff bar içinde idx'inci yuva
                    slot_w = max(1, (bx2 - bx1) // max(1, len(slots) + 2))
                    sx1 = bx1 + idx * slot_w
                    sx2 = min(bx2, sx1 + slot_w)
                    slot_crop = bgr[by1:by2, sx1:sx2]

                if slot_crop.size > 0:
                    slot_hsv = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2HSV)
                    s_mean = float(np.mean(slot_hsv[:, :, 1]))
                    v_mean = float(np.mean(slot_hsv[:, :, 2]))
                    is_vis_active = (
                        s_mean >= slot.min_saturation_threshold
                        and v_mean >= slot.min_brightness_threshold
                    )
                    slot_visual_active[slot.name] = bool(is_vis_active)
                else:
                    slot_visual_active[slot.name] = False

            return {
                "frame_analyzed": True,
                "active_icon_count": int(active_icon_count),
                "mean_saturation": round(mean_sat, 2),
                "mean_brightness": round(mean_val, 2),
                "slot_visual_active": slot_visual_active,
            }
        except Exception as exc:
            logger.error("Buff çubuğu görüntü analizi hatası: %s", exc)
            return {
                "frame_analyzed": False,
                "active_icon_count": 0,
                "mean_saturation": 0.0,
                "mean_brightness": 0.0,
                "slot_visual_active": {},
            }

    def check_and_refresh_buffs(
        self,
        frame: Optional[np.ndarray] = None,
        input_ctrl: Any = None,
        test_mode: bool = False,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Pet açlık zamanlayıcısını, yemek/parşömen buff sürelerini ve (verilmişse) `frame`
        üzerindeki buff çubuğunu kontrol eder.

        - Pet açlık süresi (`pet_feed_interval_sec`) dolmuşsa `pet_key` tuşuna basarak peti besler.
        - Süresi dolan veya ekranda buff ikonunun söndüğü tespit edilen `BuffSlot` öğelerini
          kısayol tuşlarına basarak otomatik yeniler.
        """
        # S-1202: kilit yalnizca kisa karar/kayit bolumlerinde tutulur;
        # tus gonderimi (donanim G/C) kilit DISINDA calisir
        # (check_and_rotate_character ile ayni 3 asamali desen, F-1731).
        # P2-3 (K-17): OpenCV buff-bar analizi de kilit DISINDA yapilir
        # (onceki kod inspect_buff_bar'i 1. asama kilidi icinde
        # cagiriyordu); kilit yalnizca karar/kayit icin tutulur.
        # 0. asama (kilitsiz): buff cubugu goruntusunu analiz et.
        buff_bar_status = self.inspect_buff_bar(frame)
        frame_analyzed = bool(buff_bar_status.get("frame_analyzed", False))
        slot_visual_map: Dict[str, bool] = buff_bar_status.get("slot_visual_active", {})

        # 1. asama (kilitli, kisa): pet/buff kararlarini ver.
        with self._lock:
            now = float(now_ts if now_ts is not None else time.time())

            feed_pet = self.is_pet_hungry(now_ts=now)
            pet_fed_ts_before = self.last_pet_fed_ts

            # Yenilenecek bufflar: (slot, sebep, karar anindaki last_used_ts)
            slots_to_refresh: List[Tuple[BuffSlot, str, float]] = []
            for slot in self.buff_slots:
                if not slot.enabled:
                    continue

                timer_expired = slot.is_expired_by_timer(now_ts=now)
                visually_missing = False
                if frame_analyzed and not slot_visual_map.get(slot.name, True):
                    # Görsel olarak buff çubuğunda yoksa ve son kullanımdan beri en az cooldown geçtiyse
                    if (now - slot.last_used_ts) >= self.visual_recast_cooldown_sec:
                        visually_missing = True

                if timer_expired or visually_missing:
                    trigger_src = "zamanlayıcı süresi doldu" if timer_expired else "buff çubuğunda sönük algılandı"
                    slots_to_refresh.append((slot, trigger_src, slot.last_used_ts))

        # 2. asama (kilitsiz): donanim G/C — tus gonderimleri.
        pressed_keys: List[str] = []
        refreshed_buffs: List[str] = []
        pet_fed = False

        if feed_pet:
            if _send_key_safe(input_ctrl, self.pet_key, test_mode=test_mode):
                pet_fed = True
                pressed_keys.append(self.pet_key)

        for slot, trigger_src, _last_used_before in slots_to_refresh:
            if _send_key_safe(input_ctrl, slot.hotkey, test_mode=test_mode):
                refreshed_buffs.append(slot.name)
                pressed_keys.append(slot.hotkey)
                logger.info(
                    "[PetAndBuffManager] Buff yenilendi: '%s' (Tuş: '%s', Sebep: %s, Süre: %ds).",
                    slot.name,
                    slot.hotkey,
                    trigger_src,
                    int(slot.duration_sec),
                )

        # 3. asama (kilitli, kisa): basarili gonderimleri kalici kayda gecir.
        # Eszamanli baska cagri arada ayni kaydi guncellediyse tekrar yazilmaz
        # (karar anindaki zaman damgasi degismis olur).
        with self._lock:
            if pet_fed and self.last_pet_fed_ts == pet_fed_ts_before:
                self.last_pet_fed_ts = now
                self.pet_feed_count += 1
                logger.info(
                    "[PetAndBuffManager] '%s' beslendi (Tuş: '%s', Besleme #%d).",
                    self.pet_name,
                    self.pet_key,
                    self.pet_feed_count,
                )
            refreshed_names = set(refreshed_buffs)
            for slot, _trigger_src, last_used_before in slots_to_refresh:
                if slot.name in refreshed_names and slot.last_used_ts == last_used_before:
                    slot.last_used_ts = now
            pet_feed_count = self.pet_feed_count
            pet_remaining = round(self.pet_remaining_sec(now_ts=now), 1)
            buff_states = [s.to_dict(now_ts=now) for s in self.buff_slots]

        reasons: List[str] = []
        if pet_fed:
            reasons.append(f"{self.pet_name} beslendi ('{self.pet_key}')")
        if refreshed_buffs:
            reasons.append(f"Buff yenilendi: {', '.join(refreshed_buffs)}")
        if not reasons:
            reasons.append("Pet tok ve tüm toplama buffları aktif.")

        return {
            "pet_fed": pet_fed,
            "pet_name": self.pet_name,
            "pet_key": self.pet_key,
            "pet_feed_count": pet_feed_count,
            "pet_remaining_sec": pet_remaining,
            "refreshed_buffs": refreshed_buffs,
            "pressed_keys": pressed_keys,
            "buff_bar_status": buff_bar_status,
            "buff_states": buff_states,
            "reason_tr": " | ".join(reasons),
        }

    def _is_ready(self) -> bool:
        return bool(self.buff_slots) and bool(self.pet_key)

    def get_multi_char_ready_flag(self) -> bool:
        """Gerçek duruma göre hazır bayrağı; hata durumunda False döner."""
        try:
            return bool(self._is_ready())
        except Exception:
            return False

    def get_multi_char_scheduler_ready_flag(self) -> dict:
        return {
            "module": "multi_char_scheduler",
            "ready": self.get_multi_char_ready_flag(),
            "buff_slot_count": len(self.buff_slots),
            "pet_feed_count": int(self.pet_feed_count),
        }

    def get_multi_char_scheduler_version_info(self) -> dict:
        return {"module": "multi_char_scheduler", "version": __version__, "ready": self.get_multi_char_ready_flag()}

    def get_multi_char_scheduler_diagnostic_snapshot(self) -> dict:
        ready = self.get_multi_char_ready_flag()
        return {
            "module": "multi_char_scheduler",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "buff_slot_count": len(self.buff_slots),
            "enabled_buff_slots": sum(1 for b in self.buff_slots if b.enabled),
            "pet_name": self.pet_name,
            "pet_feed_count": int(self.pet_feed_count),
            "pet_remaining_sec": round(self.pet_remaining_sec(), 1),
        }


# ============================================================================
# 4. ÜRETİM BAĞLANTISI — TEK TICK'TE ÜÇ YÖNETİCİ (F-0143)
# ============================================================================

def tick_multi_char_managers(
    alt_char_mgr: Any = None,
    task_scheduler: Any = None,
    pet_buff_mgr: Any = None,
    *,
    current_gathered: Optional[int] = None,
    energy_depleted: bool = False,
    frame: Optional[np.ndarray] = None,
    input_ctrl: Any = None,
    route_mgr: Any = None,
    test_mode: bool = False,
    now_dt: Optional[Union[datetime.datetime, datetime.time, str]] = None,
) -> Dict[str, Any]:
    """
    F-0143 üretim bağlantısı: takvim (`evaluate_current_schedule`), karakter
    rotasyonu (`check_and_rotate_character`) ve pet/buff (`check_and_refresh_buffs`)
    metotlarını tek çağrıda, doğru sırada tetikler. `state_machine`/`main`
    döngüsünden her toplama adımında veya periyodik tick'te çağrılır;
    `AltCharacterManager.bind_runtime_context` + `record_gather` yoluyla da
    otomatik tetiklenir (üretimde zaten çalışan tek çağrı record_gather'dır).

    Öncelik: takvim WAITING/PAUSED diyorsa (etkinlik beklemesi / mola) karakter
    rotasyonu atlanır — bekleme sırasında şehirden ayrılınmaz; buff yenileme
    çalışmaya devam eder. Hiçbir yönetici verilmemişse veya metot eksikse o
    bölüm `{"called": False}` döner; bir yöneticideki hata diğerlerini
    durdurmaz ve asla yükseltilmez.
    """
    schedule_result: Dict[str, Any] = {"called": False, "reason_tr": "Takvim yöneticisi verilmedi."}
    if task_scheduler is not None and hasattr(task_scheduler, "evaluate_current_schedule"):
        try:
            schedule_result = task_scheduler.evaluate_current_schedule(
                now_dt=now_dt, route_mgr=route_mgr, input_ctrl=input_ctrl, test_mode=test_mode,
            )
            schedule_result = dict(schedule_result)
            schedule_result["called"] = True
        except Exception as exc:
            logger.error("Takvim tick hatası: %s", exc)
            schedule_result = {"called": False, "reason_tr": f"Takvim değerlendirmesi hatalı: {exc}"}

    schedule_state = str(schedule_result.get("recommended_state", "")) if schedule_result.get("called") else ""
    schedule_holds = schedule_state in ("WAITING", "PAUSED")

    rotation_result: Dict[str, Any] = {"called": False, "reason_tr": "Karakter yöneticisi verilmedi."}
    if alt_char_mgr is not None and hasattr(alt_char_mgr, "check_and_rotate_character"):
        if schedule_holds:
            rotation_result = {
                "called": False,
                "rotated": False,
                "recommended_state": schedule_state,
                "reason_tr": f"Takvim {schedule_state} diyor; etkinlik/mola sırasında karakter rotasyonu atlandı.",
            }
        else:
            gathered = current_gathered
            if gathered is None:
                try:
                    gathered = int(alt_char_mgr.get_active_character().gathered_today)
                except Exception:
                    gathered = None
            if gathered is None:
                rotation_result = {"called": False, "reason_tr": "Toplama sayısı bilinmiyor; rotasyon denetlenemedi."}
            else:
                try:
                    rotation_result = alt_char_mgr.check_and_rotate_character(
                        int(gathered),
                        energy_depleted=bool(energy_depleted),
                        input_ctrl=input_ctrl,
                        route_mgr=route_mgr,
                        test_mode=test_mode,
                    )
                    rotation_result = dict(rotation_result)
                    rotation_result["called"] = True
                except Exception as exc:
                    logger.error("Rotasyon tick hatası: %s", exc)
                    rotation_result = {"called": False, "reason_tr": f"Rotasyon denetimi hatalı: {exc}"}

    buffs_result: Dict[str, Any] = {"called": False, "reason_tr": "Pet/buff yöneticisi verilmedi."}
    if pet_buff_mgr is not None and hasattr(pet_buff_mgr, "check_and_refresh_buffs"):
        try:
            buffs_result = pet_buff_mgr.check_and_refresh_buffs(
                frame=frame, input_ctrl=input_ctrl, test_mode=test_mode,
            )
            buffs_result = dict(buffs_result)
            buffs_result["called"] = True
        except Exception as exc:
            logger.error("Pet/buff tick hatası: %s", exc)
            buffs_result = {"called": False, "reason_tr": f"Pet/buff denetimi hatalı: {exc}"}

    if schedule_holds:
        recommended_state = schedule_state
    elif rotation_result.get("called"):
        recommended_state = str(rotation_result.get("recommended_state", "WALKING"))
    elif schedule_result.get("called"):
        recommended_state = str(schedule_result.get("recommended_state", "WALKING"))
    else:
        recommended_state = "WALKING"

    reasons: List[str] = []
    if schedule_result.get("called"):
        reasons.append(f"Takvim: {schedule_result.get('reason_tr', '')}")
    if rotation_result.get("called"):
        reasons.append(f"Rotasyon: {rotation_result.get('reason_tr', '')}")
    elif schedule_holds:
        reasons.append(str(rotation_result.get("reason_tr", "")))
    if buffs_result.get("called"):
        reasons.append(f"Buff: {buffs_result.get('reason_tr', '')}")

    return {
        "recommended_state": recommended_state,
        "schedule": schedule_result,
        "rotation": rotation_result,
        "buffs": buffs_result,
        "reason_tr": " | ".join(r for r in reasons if r),
    }


def get_multi_char_scheduler_capabilities() -> list:
    return ["multi_char_scheduler", "character_rotation", "diagnostic_snapshot"]


def validate_multi_char_scheduler_contract() -> bool:
    """Sözleşme denetimi: üç yöneticinin çekirdek metotları gerçekten var mı?"""
    try:
        required = {
            AltCharacterManager: ("record_gather", "check_and_rotate_character", "get_roster"),
            AutomationTaskScheduler: ("evaluate_current_schedule", "get_entries"),
            PetAndBuffManager: ("check_and_refresh_buffs", "get_multi_char_ready_flag"),
        }
        if not get_multi_char_scheduler_capabilities():
            return False
        return all(
            hasattr(cls, method) for cls, methods in required.items() for method in methods
        )
    except Exception:
        return False


def get_multi_char_scheduler_baseline_metrics() -> dict:
    return {"module": "multi_char_scheduler", "contract_valid": validate_multi_char_scheduler_contract(), "capability_count": len(get_multi_char_scheduler_capabilities())}


def get_multi_char_scheduler_readiness_status() -> dict:
    return {"module": "multi_char_scheduler", "ready": validate_multi_char_scheduler_contract(), "metrics": get_multi_char_scheduler_baseline_metrics()}


def get_multi_char_scheduler_audit_record() -> dict:
    return {"module": "multi_char_scheduler", "audit_passed": validate_multi_char_scheduler_contract(), "readiness": get_multi_char_scheduler_readiness_status()}


def get_multi_char_scheduler_operational_digest() -> dict:
    return {"module": "multi_char_scheduler", "operational": validate_multi_char_scheduler_contract(), "audit": get_multi_char_scheduler_audit_record()}


def get_multi_char_scheduler_telemetry_payload() -> dict:
    return {"module": "multi_char_scheduler", "payload": get_multi_char_scheduler_operational_digest()}


def get_multi_char_scheduler_integrity_seal() -> dict:
    return {"module": "multi_char_scheduler", "sealed": validate_multi_char_scheduler_contract(), "telemetry": get_multi_char_scheduler_telemetry_payload()}


def get_multi_char_scheduler_compliance_certificate() -> dict:
    return {"module": "multi_char_scheduler", "certified": validate_multi_char_scheduler_contract(), "seal": get_multi_char_scheduler_integrity_seal()}


def get_multi_char_scheduler_readiness_probe() -> dict:
    return {"module": "multi_char_scheduler", "ready": validate_multi_char_scheduler_contract(), "certificate": get_multi_char_scheduler_compliance_certificate()}


def get_multi_char_scheduler_audit_summary() -> dict:
    return {"module": "multi_char_scheduler", "audit_passed": validate_multi_char_scheduler_contract(), "probe": get_multi_char_scheduler_readiness_probe()}



