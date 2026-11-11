# -*- coding: utf-8 -*-
"""
PerceptionHub — tek kareden ROI biliş paketi (FrameCognition).

BotWorker her tick'te bir kez observe() çağırır; combat / discovery / economy
aynı snapshot'ı paylaşır (çift YOLO/OCR yok).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Gatherion.PerceptionHub")

CHAT_ROI = (0.012, 0.750, 0.320, 0.905)
BUFF_ROI = (0.35, 0.02, 0.65, 0.08)


@dataclass
class FrameCognition:
    """Unified perception snapshot for one processed frame."""

    ts: float = 0.0
    # F-1920: varsayılanlar artık sahte "sağlıklı ve hazır" durumu iddia
    # etmiyor — gözlem yokken HP/MP 0.0, hazır skill yok, observed=False.
    # (Tüketiciler _check_and_trigger_auto_potions'taki hp<=0.01 korumasıyla
    # 0.0'ı "bilinmiyor/ölü" sayılıp iksir içilmeden atlatır.)
    player_hp: float = 0.0
    player_mp: float = 0.0
    # v20 (HP yanlış okuma): player_hp/mp 0.0 varsayılanı "ölü" değil
    # "bilinmiyor"dur — değerin ölçüm mü varsayılan mı olduğu bu
    # bayraklarla taşınır; as_dict() bunları player_hp_valid /
    # player_mp_valid olarak yayımlar, tüketici 0.0'ı bilinmiyor
    # sanıp iksir/kaçış tetiklemez.
    hp_valid: bool = False
    mp_valid: bool = False
    monster_hp: float = 0.0
    has_target: bool = False
    ready_skills: List[str] = field(default_factory=list)
    slot_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    combo_sequence: List[str] = field(default_factory=list)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    gather_prompt: bool = False
    gather_conf: int = 0
    gather_bbox: Optional[Tuple[int, int, int, int]] = None
    chat_events: List[Dict[str, Any]] = field(default_factory=list)
    buff_count: int = 0
    learned_frames: int = 0
    source: str = "hub"
    observed: bool = False
    # F-1944: throttle ile dönen bayat snapshot bu bayrakla işaretlenir.
    stale: bool = False

    def as_dict(self) -> Dict[str, Any]:
        # A16: eskiden slot_states/targets/chat_events yalnızca dış
        # liste/sözlük kopyasıydı — iç sözlükler paylaşıldığı için
        # as_dict() çıktısını değiştiren tüketici hub'ın önbelleğindeki
        # FrameCognition'ı da değiştiriyordu (kanıt: /tmp/a16_demo7.py).
        return {
            "ts": self.ts,
            "player_hp": self.player_hp,
            "player_mp": self.player_mp,
            "player_hp_percent": int(round(self.player_hp * 100)),
            "player_mp_percent": int(round(self.player_mp * 100)),
            "player_hp_valid": self.hp_valid,
            "player_mp_valid": self.mp_valid,
            "monster_hp": self.monster_hp,
            "monster_hp_percent": int(round(self.monster_hp * 100)),
            "has_target": self.has_target,
            "ready_skills": list(self.ready_skills),
            "slot_states": {
                k: dict(v) if isinstance(v, dict) else v for k, v in self.slot_states.items()
            },
            "combo_sequence": list(self.combo_sequence),
            "targets": [dict(t) if isinstance(t, dict) else t for t in self.targets],
            "gather_prompt": self.gather_prompt,
            "gather_conf": self.gather_conf,
            "gather_bbox": self.gather_bbox,
            "chat_events": [dict(e) if isinstance(e, dict) else e for e in self.chat_events],
            "buff_count": self.buff_count,
            "learned_frames": self.learned_frames,
            "source": self.source,
            "observed": self.observed,
            "stale": self.stale,
        }


class PerceptionHub:
    """Throttle + fuse OmniGameLearner / HUD / SkillBar / YOLO / gather / chat."""

    def __init__(
        self,
        game_learner: Any = None,
        skill_bar_reader: Any = None,
        vision: Any = None,
        hud_reader: Any = None,
        yolo_detector: Any = None,
        combat_recovery: Any = None,
        min_interval_sec: float = 1.0 / 12.0,
    ) -> None:
        self.game_learner = game_learner
        self.skill_bar_reader = skill_bar_reader
        self.vision = vision
        self.hud_reader = hud_reader
        self.yolo_detector = yolo_detector
        self.combat_recovery = combat_recovery
        self.min_interval_sec = max(0.04, float(min_interval_sec))
        self._lock = threading.RLock()
        self._last_ts = 0.0
        self._last: Optional[FrameCognition] = None
        # Vision CV canavar yedek taraması için son çalıştırma anı
        # (_fill_targets: en fazla 1/sn).
        self._last_vision_monster_scan_ts: float = 0.0
        self._hud = hud_reader
        if self._hud is None:
            try:
                from core.ocr_distance import GameHUDReader

                self._hud = GameHUDReader()
            except Exception as exc:
                logger.debug("GameHUDReader init skipped: %s", exc)

    def bind(self, **kwargs: Any) -> None:
        for key, val in kwargs.items():
            if val is None:
                continue
            if key == "hud_reader":
                # F-1921: hud_reader hem nitelik hem _hud olarak tutulur.
                # Eskiden yalnızca self.hud_reader değişiyor, _fill_* ise
                # self._hud'u kullandığı için bind(hud_reader=...) sessizce
                # etkisiz kalıyordu.
                self.hud_reader = val
                self._hud = val
            elif hasattr(self, key):
                setattr(self, key, val)

    def last(self) -> Optional[FrameCognition]:
        with self._lock:
            return self._last

    def observe(self, frame: Optional[np.ndarray], force: bool = False) -> FrameCognition:
        now = time.time()
        with self._lock:
            if (
                not force
                and self._last is not None
                and (now - self._last_ts) < self.min_interval_sec
            ):
                # F-1944: throttle ile dönen bayat snapshot taze gibi
                # dönmesin — aynı nesne döner (test_throttle_returns_cached
                # `is` bekler) ama stale bayrağı işaretlenir.
                self._last.stale = True
                return self._last

        # D24: ağır algı işi (YOLO/OCR/vision) artık kilit DIŞINDA çalışır.
        # Eskiden observe() tüm fill'ler boyunca self._lock'u tutuyordu;
        # yavaş bir YOLO çıkarımı sırasında başka bir thread'in last()
        # çağrısı ~300 ms bloklanıyordu (bench_d24: max 298 ms). Şimdi
        # kilit yalnızca throttle kontrolü ve snapshot yayını için tutulur;
        # last() her zaman anında döner. Yayın anında `now >= _last_ts`
        # koruması, geç biten eski bir observe'un daha taze snapshot'ı
        # ezmesini engeller (en yeni kazanır).
        # A16: _fill_vitals (hp_ok, mp_ok) döner — bayraklar vital başına,
        # böylece learner merge yalnızca HUD'ın gerçekten okuyamadığı
        # vital'i doldurur.
        cog = FrameCognition(ts=now)
        try:
            hp_ok, mp_ok = self._fill_vitals(frame, cog)
            self._fill_skills(frame, cog)
            self._fill_targets(frame, cog)
            self._fill_gather(frame, cog)
            self._fill_chat(frame, cog)
            self._fill_buffs(frame, cog)
            self._merge_learner(cog, hp_ok=hp_ok, mp_ok=mp_ok)
        except Exception as exc:
            logger.debug("PerceptionHub.observe error: %s", exc)

        cog.observed = frame is not None
        cog.stale = False
        with self._lock:
            if now >= self._last_ts:
                self._last = cog
                self._last_ts = now
        return cog

    @staticmethod
    def _apply_hp_mp(vitals: Dict[str, Any], cog: FrameCognition) -> bool:
        """HP/MP yüzdelerini (0..100) 0..1 aralığına yaz; 0 geçerlidir.

        F-1914: `or 100.0` artık 0'ı (ölü/bitkin) "dolu" sanmıyor —
        None açıkça ele alınır, 0 geçerli değerdir; eksik anahtar HP'yi
        (varsayılan 0.0 = "bilinmiyor") değiştirmez. Her iki HUD API'si
        (read_player_hp_mp_bars / read_hp_mp) aynı ayrıştırmayı
        paylaşır; hedef bilgisi (has_target/target_pct) yalnızca
        read_player_hp_mp_bars dalında doldurulur.
        """
        filled = False
        hp_raw = vitals.get("hp_pct")
        if hp_raw is not None:
            cog.player_hp = max(0.0, min(1.0, float(hp_raw) / 100.0))
            filled = True
        mp_raw = vitals.get("mp_pct")
        if mp_raw is not None:
            cog.player_mp = max(0.0, min(1.0, float(mp_raw) / 100.0))
            filled = True
        return filled

    def _fill_vitals(self, frame: Optional[np.ndarray], cog: FrameCognition) -> Tuple[bool, bool]:
        """Fill player HP/MP from the HUD.

        Returns (hp_ok, mp_ok): one flag per vital, so the learner merge
        can fill only the vital the HUD actually missed. A16: the old
        single bool dropped the learner MP whenever the HUD read HP alone
        (MP stayed 0.0 = "unknown/dead" although the learner had 0.6).

        F-1914: `or 100.0` artık 0'ı (ölü/bitkin) "dolu" sanmıyor —
        None açıkça ele alınır, 0 geçerli değerdir; eksik anahtar HP'yi
        (varsayılan 0.0 = "bilinmiyor") değiştirmez. combat_recovery dalı
        da aralığı doğrulayıp yalnızca geçerli değeri yazar.
        """
        if frame is None or self._hud is None:
            return (False, False)
        hp_ok = False
        mp_ok = False
        try:
            if hasattr(self._hud, "read_player_hp_mp_bars"):
                vitals = self._hud.read_player_hp_mp_bars(frame)
                # Ortak ayrıştırma _apply_hp_mp'de (bool döner, D24 testi
                # bunu pinler); A16 bayrakları anahtar varlığından
                # türetilir — helper yalnızca o anahtar varken yazar,
                # bayraklarla birebir aynı karar.
                self._apply_hp_mp(vitals, cog)
                hp_ok = vitals.get("hp_pct") is not None
                mp_ok = vitals.get("mp_pct") is not None
                cog.has_target = bool(vitals.get("has_target", False))
                tgt_raw = vitals.get("target_pct")
                if tgt_raw is not None:
                    cog.monster_hp = max(0.0, min(1.0, float(tgt_raw) / 100.0))
            elif hasattr(self._hud, "read_hp_mp"):
                vitals = self._hud.read_hp_mp(frame)
                self._apply_hp_mp(vitals, cog)
                hp_ok = vitals.get("hp_pct") is not None
                mp_ok = vitals.get("mp_pct") is not None
        except Exception as exc:
            logger.debug("vitals read skipped: %s", exc)

        if self.combat_recovery is not None and hasattr(self.combat_recovery, "read_hp_ratio_or_none"):
            # v20 (HP yanlış okuma): okunabilirliği ayırt eden API
            # kullanılır. Eski get_player_hp_ratio dalı okuyamadığında
            # 0.0 dönüyor ve hub bunu geçerli ölçüm sanıp HUD'ın
            # değerini eziyordu (dokümanı da öyle söyler:
            # "Okunabilirliği ayırt etmek isteyen çağıranlar
            # read_hp_ratio_or_none kullanmalıdır"). None ise ezme yok.
            try:
                hp_r = self.combat_recovery.read_hp_ratio_or_none(frame)
                if hp_r is not None:
                    hp_r = float(hp_r)
                    if 0.0 <= hp_r <= 1.0:
                        cog.player_hp = hp_r
                        hp_ok = True
            except Exception:
                pass
        elif self.combat_recovery is not None and hasattr(self.combat_recovery, "get_player_hp_ratio"):
            # Yalnızca eski API'yi sunan nesneler için yedek: 0.0
            # "bilinmiyor" demektir (omni_game_learner S-3101'deki
            # `0.0 < cr_ratio` kuralının aynısı) — geçerli ölçümü ezmez.
            try:
                hp_r = float(self.combat_recovery.get_player_hp_ratio(frame))
                if 0.0 < hp_r <= 1.0:
                    cog.player_hp = hp_r
                    hp_ok = True
            except Exception:
                pass
        cog.hp_valid = bool(hp_ok)
        cog.mp_valid = bool(mp_ok)
        return (hp_ok, mp_ok)

    def _fill_skills(self, frame: Optional[np.ndarray], cog: FrameCognition) -> None:
        reader = self.skill_bar_reader
        if reader is None or frame is None:
            return
        try:
            if hasattr(reader, "inspect_skill_bar"):
                states = reader.inspect_skill_bar(frame) or {}
                if isinstance(states, dict):
                    cog.slot_states = dict(states)
                    ready = [
                        str(k)
                        for k, v in states.items()
                        if isinstance(v, dict) and bool(v.get("ready", False))
                    ]
                    if ready:
                        cog.ready_skills = ready
            if hasattr(reader, "select_best_ready_skill"):
                best = reader.select_best_ready_skill(frame)
                if best and str(best) not in cog.ready_skills:
                    cog.ready_skills = [str(best)] + list(cog.ready_skills)
        except Exception as exc:
            logger.debug("skill bar inspect skipped: %s", exc)

    def _fill_targets(self, frame: Optional[np.ndarray], cog: FrameCognition) -> None:
        if frame is None:
            return
        if self.combat_recovery is not None and hasattr(
            self.combat_recovery, "_detect_enemy_target_bar"
        ):
            try:
                if bool(self.combat_recovery._detect_enemy_target_bar(frame)):
                    cog.has_target = True
            except Exception:
                pass

        det = self.yolo_detector
        if det is not None:
            try:
                boxes = None
                if hasattr(det, "detect"):
                    boxes = det.detect(frame)
                elif hasattr(det, "infer"):
                    boxes = det.infer(frame)
                elif hasattr(det, "detect_3d_objects"):
                    # ONNXYolo3DDetector'ın gerçek giriş noktası: detect /
                    # infer metodu YOK. Eski kod burada None ile dönüp
                    # YOLO hedeflerini hiç toplamıyordu.
                    boxes = det.detect_3d_objects(frame)
                if boxes:
                    for b in boxes:
                        if isinstance(b, dict):
                            cog.targets.append(b)
                        else:
                            cog.targets.append({"raw": b})
                if cog.targets and not cog.has_target:
                    # Any monster-like class implies potential combat target.
                    for t in cog.targets:
                        if not isinstance(t, dict):
                            continue
                        try:
                            cid = (
                                int(t.get("class_id"))
                                if t.get("class_id") is not None
                                else None
                            )
                        except (TypeError, ValueError):
                            cid = None
                        label = str(
                            t.get(
                                "label",
                                t.get(
                                    "class",
                                    t.get("class_name", t.get("name", "")),
                                ),
                            )
                        ).lower()
                        # class_id 10 monster_3d / 11 minimap_monster_dot:
                        # eski kod yalnızca "label"/"class" anahtarına
                        # bakıyordu, ONNX dict'lerindeki "class_name"i ve
                        # sayısal id'yi hiç okumuyordu.
                        if (cid in (10, 11)) or any(
                            tok in label for tok in ("mob", "monster", "enemy")
                        ):
                            cog.has_target = True
                            break
            except Exception as exc:
                logger.debug("yolo targets skipped: %s", exc)

        # Vision CV canavar tespiti (yedek): YOLO canavar bulamadıysa ve
        # hedef barı da yoksa vision.detect_monsters_3d çalıştır. Bu
        # fonksiyonun üretimde HİÇ çağırıcısı yoktu — ekranda canavar
        # varken panelin "HEDEF YOK" demesinin kopukluklarından biri bu.
        # Maliyet sınırı: saniyede en fazla bir tarama (hub zaten ~12 Hz
        # throttle'lı; bu yedek onsuz her karede CV+exemplar koşardı).
        if not cog.has_target and self.vision is not None and hasattr(
            self.vision, "detect_monsters_3d"
        ):
            now = time.time()
            if now - self._last_vision_monster_scan_ts >= 1.0:
                self._last_vision_monster_scan_ts = now
                try:
                    monsters = self.vision.detect_monsters_3d(frame) or []
                    for m in monsters:
                        if not isinstance(m, dict):
                            continue
                        if m.get("class_id") == 10 or bool(m.get("is_hostile")):
                            cog.has_target = True
                            cog.targets.append(
                                {
                                    "class_id": 10,
                                    "class_name": "monster_3d",
                                    "confidence": float(
                                        m.get("confidence", 0.82) or 0.82
                                    ),
                                    "bbox": m.get("bbox"),
                                    "center": (
                                        m.get("screen_x"),
                                        m.get("screen_y"),
                                    ),
                                    "is_targeted": bool(
                                        m.get("is_targeted", False)
                                    ),
                                    "has_diamond_brackets": bool(
                                        m.get("has_diamond_brackets", False)
                                    ),
                                    "source": "vision.detect_monsters_3d",
                                }
                            )
                except Exception as exc:
                    logger.debug("vision monster targets skipped: %s", exc)

    def _fill_gather(self, frame: Optional[np.ndarray], cog: FrameCognition) -> None:
        vision = self.vision
        if vision is None or not hasattr(vision, "detect_gather_prompt"):
            return
        try:
            raw = None
            try:
                raw = vision.detect_gather_prompt(frame=frame)
            except TypeError:
                raw = vision.detect_gather_prompt()
            if isinstance(raw, tuple):
                found = bool(raw[0]) if len(raw) > 0 else False
                # A16: güven aralığı dışında değer (örn. 999) aynen
                # geçiyordu; 0..100'e kırpılır.
                conf = int(raw[1]) if len(raw) > 1 else 0
                conf = max(0, min(100, conf))
                bbox = raw[2] if len(raw) > 2 else None
                cog.gather_prompt = found
                cog.gather_conf = conf
                if bbox is not None and len(bbox) >= 4:
                    vals = [int(v) for v in bbox[:4]]
                    if frame is not None and isinstance(frame, np.ndarray) and frame.size:
                        # A16: kare dışı bbox (örn. 99999) aynen geçiyordu;
                        # kare sınırlarına kırpılır.
                        fh, fw = frame.shape[:2]
                        vals[0] = max(0, min(fw, vals[0]))
                        vals[1] = max(0, min(fh, vals[1]))
                        vals[2] = max(vals[0], min(fw, vals[2]))
                        vals[3] = max(vals[1], min(fh, vals[3]))
                    cog.gather_bbox = tuple(vals)  # type: ignore[assignment]
            elif isinstance(raw, bool):
                cog.gather_prompt = raw
        except Exception as exc:
            logger.debug("gather detect skipped: %s", exc)

    def _fill_chat(self, frame: Optional[np.ndarray], cog: FrameCognition) -> None:
        """Lightweight chat ROI heuristics (no external OCR dependency)."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = CHAT_ROI
            roi = frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
            if roi.size == 0:
                return
            events = parse_chat_loot_events(roi)
            # F-1924: HUD çözülmüş sohbet metni üretebiliyorsa (örn.
            # read_chat_text) loot olayları metinden gerçekten ayrıştırılır;
            # yoksa parlaklık yalnızca chat_activity üretir, loot uydurulmaz.
            chat_text = None
            if self._hud is not None:
                for meth_name in ("read_chat_text", "ocr_chat_text", "get_chat_text"):
                    if hasattr(self._hud, meth_name):
                        try:
                            chat_text = getattr(self._hud, meth_name)(frame)
                        except Exception:
                            chat_text = None
                        # A16: eskiden ilk metot None/"" dönse bile döngü
                        # kırılıyordu — sonraki metot hiç denenmiyor,
                        # loot olayı kaçıyordu. Yalnızca gerçek metin
                        # dönerse dur.
                        if chat_text:
                            break
            if chat_text:
                events.extend(parse_chat_text_line(str(chat_text)))
            if self._hud is not None and hasattr(self._hud, "parse_chat_feedback_fast"):
                try:
                    fb = self._hud.parse_chat_feedback_fast(frame)
                    if isinstance(fb, dict):
                        for flag in ("extraction_failed", "cannot_mount", "invalid_target"):
                            if fb.get(flag):
                                events.append({"type": flag, "raw": flag})
                except Exception:
                    pass
            cog.chat_events = events
        except Exception as exc:
            logger.debug("chat ROI skipped: %s", exc)

    def _fill_buffs(self, frame: Optional[np.ndarray], cog: FrameCognition) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = BUFF_ROI
            roi = frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
            if roi.size == 0:
                return
            import cv2

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Saturated colorful icons in buff strip.
            mask = cv2.inRange(hsv, (0, 60, 60), (179, 255, 255))
            cnt = int(np.count_nonzero(mask))
            area = max(1, mask.size)
            # Rough icon count via connected components density.
            cog.buff_count = max(0, min(20, int(round((cnt / area) * 40))))
        except Exception:
            pass

    def _merge_learner(self, cog: FrameCognition, hp_ok: bool = False, mp_ok: bool = False) -> None:
        learner = self.game_learner
        if learner is None or not hasattr(learner, "get_live_cognition_snapshot"):
            return
        try:
            snap = learner.get_live_cognition_snapshot() or {}
            cog.learned_frames = int(snap.get("total_learned_frames", 0) or 0)
            # F-1914: HUD vital okuduysa (gerçek %0 dahil) learner onu
            # ezmez — eski `cog.player_hp >= 0.999 or <= 0.0` koşulu gerçek
            # %0 HP'yi learner tahminiyle siliyordu. Yalnızca HUD
            # okumadıysa doldurulur. A16: bayraklar artık vital başına —
            # HUD yalnızca HP okuduysa learner MP'yi doldurabilir.
            if not hp_ok:
                if snap.get("player_hp_ratio") is not None:
                    cog.player_hp = float(snap.get("player_hp_ratio", cog.player_hp))
                    # v20: learner doldurduysa bu değer eskisi gibi
                    # işleme girer (davranış korunur); hiçbir kaynak
                    # doldurmadıysa hp_valid False kalır.
                    cog.hp_valid = True
            if not mp_ok:
                if snap.get("player_mp_ratio") is not None:
                    cog.player_mp = float(snap.get("player_mp_ratio", cog.player_mp))
                    cog.mp_valid = True
            mhp = snap.get("monster_hp_ratio")
            if mhp is not None and float(mhp) > 0:
                cog.monster_hp = float(mhp)
                cog.has_target = True
            ready = snap.get("ready_skills")
            if isinstance(ready, list) and ready and not cog.slot_states:
                cog.ready_skills = [str(x) for x in ready]
            combo = snap.get("combo_sequence")
            if isinstance(combo, list):
                cog.combo_sequence = [str(x) for x in combo]
        except Exception as exc:
            logger.debug("learner merge skipped: %s", exc)


_KINA_RE = re.compile(
    r"(?:obtained|gained|acquired|received|kazand[ıi]|ald[ıi]).{0,24}?([\d.,]+)\s*kina",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(
    r"(?:acquired|obtained|received|ald[ıi]|kazand[ıi])\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9' \-]{2,40})",
    re.IGNORECASE,
)
_DEFEAT_RE = re.compile(
    r"(defeated|slain|killed|öldür|yenildi)",
    re.IGNORECASE,
)


def parse_chat_loot_events(roi_bgr: np.ndarray, text: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Chat ROI'den olay üretir.

    F-1924: metin verilmezse yalnızca parlaklık tabanlı 'chat_activity'
    döner — loot uydurulmaz. Çözülmüş sohbet metni verilirse
    parse_chat_text_line ile gerçek loot olayları (kina/item/defeat)
    üretilir. Eskiden metin parametresi yoktu ve fonksiyon loot olayı
    hiç üretemiyordu.
    """
    events: List[Dict[str, Any]] = []
    if text is not None:
        events.extend(parse_chat_text_line(str(text)))
    if roi_bgr is None or getattr(roi_bgr, "size", 0) == 0:
        return events
    try:
        import cv2

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
        # Bright text-ish pixels
        bright = int(np.count_nonzero(gray > 180))
        if bright > gray.size * 0.02:
            events.append({"type": "chat_activity", "bright_ratio": bright / max(1, gray.size)})
    except Exception:
        pass
    return events


def parse_chat_text_line(text: str) -> List[Dict[str, Any]]:
    """Parse a decoded chat line into structured loot events (used by tests / OCR plug-in)."""
    events: List[Dict[str, Any]] = []
    raw = str(text or "").strip()
    if not raw:
        return events
    if _DEFEAT_RE.search(raw):
        events.append({"type": "defeat", "raw": raw})
    m = _KINA_RE.search(raw)
    if m:
        digits = m.group(1).replace(",", "").replace(".", "")
        try:
            kina = int(digits)
        except ValueError:
            kina = 0
        events.append({"type": "kina", "amount": kina, "raw": raw})
    m2 = _ITEM_RE.search(raw)
    if m2:
        events.append({"type": "item", "item_name": m2.group(1).strip(), "raw": raw})
    return events
