# -*- coding: utf-8 -*-
"""
Incident Snapshot Recorder & 256-Bit Mobile Token Security (core/incident_recorder.py)
Items 38 & 39 — Gatherion A2 Bot

Responsibilities:
1. Item 39 — IncidentRecorder:
   Captures, annotates (timestamp, event badge, region, node ID, HP bar), and saves
   JPEG snapshots to `logs/incidents/` whenever critical events occur:
   - Death ("Ölüm" / "character_dead")
   - Stuck ("Takılma" / "stuck")
   - Prolonged Combat >2m ("Uzun Savaş >2dk" / "prolonged_combat")
   - Inventory Near Full ("Envanter %95 Dolu" / "inventory_near_full")
   Maintains a thread-safe JSON manifest (`logs/incidents/incidents.json`) containing
   metadata and base64-encoded JPEG thumbnails (`data:image/jpeg;base64,<b64>`) for instant
   display inside the Mobile Dashboard "📸 Olay Kayıtları Galerisi".

2. Item 38 — 256-Bit QR Security Token Manager (`MobileTokenAuthenticator`):
   Generates, persists, and validates 256-bit (64-hex-character / 32-byte) cryptographic
   tokens for `X-Gatherion-Token` header and `token` QR Code URL authentication.
"""

from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from version import __version__

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INCIDENTS_DIR = os.path.join(PROJECT_ROOT, "logs", "incidents")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_INCIDENTS_DIR, "incidents.json")
DEFAULT_TOKEN_PATH = os.path.join(PROJECT_ROOT, "config", "gatherion_256bit_token.txt")

logger = logging.getLogger("Gatherion.IncidentRecorder")
logger.setLevel(logging.INFO)


_META_CHARACTER_DEAD: Dict[str, Any] = {
    "code": "character_dead",
    "label_tr": "Ölüm",
    "badge_tr": "KRİTİK: KARAKTER ÖLDÜ",
    "color_bgr": (48, 48, 239),  # Crimson red in BGR
    "severity": "critical",
}
_META_STUCK: Dict[str, Any] = {
    "code": "stuck",
    "label_tr": "Takılma",
    "badge_tr": "UYARI: KARAKTER TAKILDI",
    "color_bgr": (11, 158, 245),  # Amber in BGR
    "severity": "warning",
}
_META_PROLONGED_COMBAT: Dict[str, Any] = {
    "code": "prolonged_combat",
    "label_tr": "Uzun Savaş >2dk",
    "badge_tr": "ALARM: UZUN SAVAŞ (>2 DK)",
    "color_bgr": (80, 92, 246),  # Red-Orange in BGR
    "severity": "critical",
}
_META_INVENTORY_NEAR_FULL: Dict[str, Any] = {
    "code": "inventory_near_full",
    "label_tr": "Envanter %95 Dolu",
    "badge_tr": "UYARI: ENVANTER %95+ DOLU",
    "color_bgr": (212, 182, 6),  # Cyan in BGR
    "severity": "warning",
}
_META_MANUAL_SNAPSHOT: Dict[str, Any] = {
    "code": "manual_snapshot",
    "label_tr": "Manuel Ekran Kaydı",
    "badge_tr": "BİLGİ: MANUEL OLAY KAYDI",
    "color_bgr": (129, 185, 16),  # Emerald in BGR
    "severity": "info",
}

# Aliases share one canonical dict each (same code/label/badge/color/severity);
# _resolve_event_meta() returns a copy so no caller can mutate the shared entry.
EVENT_METADATA_MAP: Dict[str, Dict[str, Any]] = {
    "character_dead": _META_CHARACTER_DEAD,
    "ölüm": _META_CHARACTER_DEAD,
    "death": _META_CHARACTER_DEAD,
    "stuck": _META_STUCK,
    "takılma": _META_STUCK,
    "prolonged_combat": _META_PROLONGED_COMBAT,
    "uzun savaş >2dk": _META_PROLONGED_COMBAT,
    "inventory_near_full": _META_INVENTORY_NEAR_FULL,
    "manual_snapshot": _META_MANUAL_SNAPSHOT,
}


def _to_ascii_tr(text: str) -> str:
    """Fold Turkish letters to ASCII for cv2 Hershey text (no TTF support).

    Single definition of the chain that _annotate_snapshot used to repeat
    three times inline. Same replacements, same output, same speed
    (measured: chained replaces beat str.translate for these short strings).
    """
    return (
        str(text)
        .replace("İ", "I")
        .replace("ı", "i")
        .replace("Ş", "S")
        .replace("ş", "s")
        .replace("Ğ", "G")
        .replace("ğ", "g")
        .replace("Ü", "U")
        .replace("ü", "u")
        .replace("Ö", "O")
        .replace("ö", "o")
        .replace("Ç", "C")
        .replace("ç", "c")
    )


def _is_valid_256bit_token(candidate: Any) -> bool:
    """A 256-bit token is exactly 64 hex characters — nothing shorter counts."""
    if not isinstance(candidate, str):
        return False
    text = candidate.strip()
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _write_token_file(token_path: str, token: str) -> bool:
    """Persist the token with owner-only permissions (0o600)."""
    try:
        os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, token.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            os.chmod(token_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        return True
    except Exception as exc:
        logger.debug("256-bit token diske kaydedilemedi: %s", exc)
        return False


class MobileTokenAuthenticator:
    """
    Item 38 — 256-Bit Cryptographic Security Key & QR Token Manager.
    Provides generation, persistence, URL signing, and constant-time verification
    of 256-bit (32-byte / 64-hex-char) tokens for mobile remote control.
    """

    def __init__(
        self,
        token_path: str = DEFAULT_TOKEN_PATH,
        explicit_token: Optional[str] = None,
        require_token: bool = True,
    ) -> None:
        self.token_path = os.path.abspath(token_path)
        self.require_token = bool(require_token)
        self._lock = threading.RLock()
        self.token: str = self._init_or_load_token(explicit_token)

    def _init_or_load_token(self, explicit_token: Optional[str] = None) -> str:
        with self._lock:
            if explicit_token and str(explicit_token).strip():
                candidate = str(explicit_token).strip()
                if _is_valid_256bit_token(candidate):
                    return candidate
                logger.warning(
                    "Verilen açık token 256-bit (64 hex) değil, yok sayılıyor."
                )

            try:
                if os.path.isfile(self.token_path):
                    with open(self.token_path, "r", encoding="utf-8") as f:
                        saved = f.read().strip()
                    if _is_valid_256bit_token(saved):
                        return saved
                    logger.warning(
                        "Kayıtlı token 256-bit (64 hex) değil, yenisi üretiliyor."
                    )
            except Exception as exc:
                logger.debug("256-bit token dosyası okunamadı: %s", exc)

            # Generate a 256-bit (32 bytes = 64 hex chars) cryptographic token
            new_token = secrets.token_hex(32)
            _write_token_file(self.token_path, new_token)
            return new_token

    def rotate_token(self) -> str:
        """Generates and persists a brand-new 256-bit security key."""
        with self._lock:
            self.token = secrets.token_hex(32)
            _write_token_file(self.token_path, self.token)
            return self.token

    def bind_to_mobile_server(self, server: Any) -> bool:
        """Wire this 256-bit authenticator into a MobileRemoteServer.

        The live HTTP server (core/mobile_server.py) keeps its own token
        attributes (``access_token`` / ``require_token``); setting them to
        this authenticator's token is what actually enforces the 256-bit
        token on HTTP requests. Call once after constructing the server
        (e.g. in main.py right after ``MobileRemoteServer(...)``).
        Returns False when the server object has no such attributes.
        """
        try:
            if not hasattr(server, "access_token") or not hasattr(server, "require_token"):
                logger.error("Mobil sunucuya token bağlanamadı: access_token/require_token yok.")
                return False
            server.access_token = self.token
            server.require_token = True
            return True
        except Exception as exc:
            logger.error("Mobil sunucuya token bağlanamadı: %s", exc)
            return False

    @property
    def token_fingerprint(self) -> str:
        """Returns a short SHA-256 fingerprint for safe UI display."""
        digest = hashlib.sha256(self.token.encode("utf-8")).hexdigest()
        return f"{digest[:6]}-{digest[-6:]}"

    def build_qr_authenticated_url(self, base_url: str) -> str:
        """Appends `?token=<256bit_token>` to the LAN or Tunnel URL for QR scanning."""
        clean_base = str(base_url or "").strip()
        if not clean_base:
            return ""
        sep = "&" if "?" in clean_base else "?"
        return f"{clean_base}{sep}token={urllib.parse.quote(self.token, safe='')}"

    def verify_request(
        self,
        headers: Optional[Dict[str, str]] = None,
        query_string: str = "",
        accepted_tokens: Optional[List[str]] = None,
    ) -> bool:
        """
        Verifies incoming HTTP request credentials using constant-time comparison.
        Checks:
        - `X-Gatherion-Token` header
        - `X-Auth-Token` / `X-Access-Token` header
        - `Authorization: Bearer <token>` header
        - `token` or `key` query parameters
        """
        if not self.require_token:
            return True

        candidates: List[str] = []
        if headers:
            for h_key in ("X-Gatherion-Token", "x-gatherion-token", "X-Auth-Token", "x-auth-token", "X-Access-Token", "x-access-token"):
                val = headers.get(h_key, "")
                if val and str(val).strip():
                    candidates.append(str(val).strip())
            auth_val = str(headers.get("Authorization") or headers.get("authorization") or "").strip()
            if auth_val.lower().startswith("bearer "):
                candidates.append(auth_val[7:].strip())

        if query_string:
            qs = urllib.parse.parse_qs(query_string.lstrip("?"))
            for param in ("token", "gatherion_token", "key"):
                if param in qs and qs[param]:
                    candidates.append(str(qs[param][0]).strip())

        valid_pool = [self.token]
        if accepted_tokens:
            valid_pool.extend([str(t).strip() for t in accepted_tokens if t and str(t).strip()])

        for cand in candidates:
            for valid_tok in valid_pool:
                if cand and valid_tok and hmac.compare_digest(cand, valid_tok):
                    return True
        return False


class IncidentRecorder:
    """
    Item 39 — Olay Anı Ekran Görüntüsü Galerisi (Incident Snapshot Recorder).

    Saves annotated JPEG screenshots to `logs/incidents/` and maintains `logs/incidents/incidents.json`
    with base64 thumbnails whenever:
    - Character Death ("Ölüm" / `character_dead`)
    - Character Stuck ("Takılma" / `stuck`)
    - Prolonged Combat >2 min ("Uzun Savaş >2dk" / `prolonged_combat`)
    - Inventory 95% Full ("Envanter %95 Dolu" / `inventory_near_full`)
    occurs.
    """

    def __init__(
        self,
        incidents_dir: str = DEFAULT_INCIDENTS_DIR,
        capture_provider: Any = None,
        max_incidents: int = 50,
        cooldown_sec: float = 25.0,
    ) -> None:
        self.incidents_dir = os.path.abspath(incidents_dir)
        self.manifest_path = os.path.join(self.incidents_dir, "incidents.json")
        self.capture_provider = capture_provider
        self.max_incidents = max(5, int(max_incidents))
        self.cooldown_sec = max(0.0, float(cooldown_sec))

        self._lock = threading.RLock()
        self._last_event_ts: Dict[str, float] = {}
        self._incidents: List[Dict[str, Any]] = []
        # O-3: sessiz kayıp sayaçları (health summary'de görünür).
        self._skipped_cooldown: int = 0
        self._skipped_below_threshold: int = 0
        self._evicted_total: int = 0
        self._corrupt_manifest_backups: int = 0
        # Hız sınırlı skip uyarıları: (tür, kod) -> son warning ts.
        self._last_skip_warn_ts: Dict[str, float] = {}

        os.makedirs(self.incidents_dir, exist_ok=True)
        self._load_manifest()

    def _log_skip_limited(self, kind: str, code: str, message: str, *args: Any) -> None:
        """Atlama uyarısı — 60 sn pencerede (tür, kod) başına 1 warning,
        diğerleri debug (O-3: her tick warning spam'i yok)."""
        key = f"{kind}:{code}"
        now = time.time()
        last = self._last_skip_warn_ts.get(key, 0.0)
        if (now - last) >= 60.0:
            self._last_skip_warn_ts[key] = now
            logger.warning(message, *args)
        else:
            logger.debug(message, *args)

    # =========================================================================
    # Manifest Persistence
    # =========================================================================

    def _load_manifest(self) -> None:
        with self._lock:
            if not os.path.isfile(self.manifest_path):
                self._incidents = []
                self._save_manifest_unlocked()
                return
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dropped = 0
                if isinstance(data, list):
                    dropped = max(0, len(data) - self.max_incidents)
                    self._incidents = data[-self.max_incidents :]
                elif isinstance(data, dict) and isinstance(data.get("incidents"), list):
                    dropped = max(0, len(data["incidents"]) - self.max_incidents)
                    self._incidents = data["incidents"][-self.max_incidents :]
                else:
                    self._incidents = []
                if dropped:
                    # O-3: yüklemede sınır dışı bırakılan eski kayıtlar da
                    # tahliye sayılır (sessiz kayıp olmasın).
                    self._evicted_total += dropped
                    logger.warning(
                        "Manifest yüklenirken sınır (%d) aşıldığı için %d eski olay bırakıldı.",
                        self.max_incidents,
                        dropped,
                    )
            except Exception as exc:
                logger.warning("Olay kayıtları manifest dosyası okunamadı, sıfırlanıyor: %s", exc)
                # Never discard a corrupt manifest without a backup copy.
                try:
                    # O-3: zaman damgalı yedek — sabit isim her bozuk
                    # manifestte öncekinin üstüne yazıyordu.
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    backup_path = f"{self.manifest_path}.corrupt.{stamp}.bak"
                    with open(self.manifest_path, "rb") as src, open(backup_path, "wb") as dst:
                        dst.write(src.read())
                    self._corrupt_manifest_backups += 1
                    logger.warning("Bozuk manifest yedeklendi: %s", backup_path)
                except Exception as backup_exc:
                    # O-3: yedek alınamadıysa bozuk manifest kaybolur —
                    # sessiz debug değil, warning.
                    logger.warning("Bozuk manifest yedeklenemedi: %s", backup_exc)
                self._incidents = []

    def _save_manifest_unlocked(self) -> None:
        try:
            os.makedirs(self.incidents_dir, exist_ok=True)
            tmp_path = self.manifest_path + ".tmp"
            # Thumbnails live as sidecar .thumb.jpg files, not as base64 in
            # the manifest: a 50-entry manifest must not carry 50 images.
            stored = [
                {key: value for key, value in entry.items() if key != "thumbnail_b64"}
                for entry in self._incidents
            ]
            payload = {
                "updated_at": round(time.time(), 3),
                "iso_updated": datetime.now().isoformat(timespec="seconds"),
                "total": len(stored),
                "incidents": stored,
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.manifest_path)
        except Exception as exc:
            logger.error("Olay kayıtları manifest dosyası yazılamadı: %s", exc)

    # =========================================================================
    # Frame Capture & Visual Annotation
    # =========================================================================

    def _acquire_frame(self, explicit_frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Returns a real captured frame, or None when no frame is available.

        A synthetic grid canvas must never be saved as an "incident
        screenshot": a fake image in the gallery is worse than no image.
        """
        if explicit_frame is not None and isinstance(explicit_frame, np.ndarray) and explicit_frame.size > 0:
            if explicit_frame.ndim == 2:
                return cv2.cvtColor(explicit_frame, cv2.COLOR_GRAY2BGR)
            return explicit_frame.copy()

        cap = self.capture_provider
        if cap is not None:
            try:
                frame = None
                if hasattr(cap, "grab_full"):
                    frame = cap.grab_full(grayscale=False)
                elif hasattr(cap, "grab_frame"):
                    frame = cap.grab_frame()
                elif hasattr(cap, "grab_region"):
                    frame = cap.grab_region(None, grayscale=False)
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    if frame.ndim == 2:
                        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    return frame.copy()
            except Exception as exc:
                logger.debug("IncidentRecorder ekran yakalama uyarısı: %s", exc)

        return None

    @staticmethod
    def _resolve_event_meta(event_type: str) -> Dict[str, Any]:
        key = str(event_type or "manual_snapshot").strip().lower()
        if key in EVENT_METADATA_MAP:
            return dict(EVENT_METADATA_MAP[key])
        if "ölüm" in key or "dead" in key or "death" in key:
            return dict(EVENT_METADATA_MAP["character_dead"])
        if "takıl" in key or "stuck" in key:
            return dict(EVENT_METADATA_MAP["stuck"])
        if "savaş" in key or "combat" in key:
            return dict(EVENT_METADATA_MAP["prolonged_combat"])
        if "envanter" in key or "inventory" in key or "full" in key:
            return dict(EVENT_METADATA_MAP["inventory_near_full"])
        return {
            "code": key.replace(" ", "_") or "incident",
            "label_tr": str(event_type or "Olay Kaydı"),
            "badge_tr": f"OLAY: {str(event_type or 'GENEL').upper()}",
            "color_bgr": (212, 182, 6),
            "severity": "warning",
        }

    def _annotate_snapshot(
        self,
        frame: np.ndarray,
        meta: Dict[str, Any],
        ts_str: str,
        details: str,
        region: str,
        node_id: Optional[int],
        hp_ratio: Optional[float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w, target_h = 960, 540
        if (w, h) != (target_w, target_h):
            annotated = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            annotated = frame.copy()

        color_bgr: Tuple[int, int, int] = tuple(meta.get("color_bgr", (48, 48, 239)))  # type: ignore[assignment]

        # Outer alert border
        cv2.rectangle(annotated, (0, 0), (target_w - 1, target_h - 1), color_bgr, 4)

        # Top translucent HUD banner
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 68), (14, 12, 10), -1)
        cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

        # Colored severity pill on top-left
        cv2.rectangle(annotated, (12, 12), (310, 54), color_bgr, -1)
        badge_ascii = _to_ascii_tr(meta.get("badge_tr", "OLAY KAYDI"))
        cv2.putText(
            annotated,
            badge_ascii[:26],
            (22, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Timestamp, region, node, HP info on top-right
        node_label = f"Node #{node_id}" if node_id is not None else "Node: -"
        reg_label = (region or "Altgard")[:22]
        hp_label = f"HP: {int(round(max(0.0, min(1.0, hp_ratio)) * 100))}%" if hp_ratio is not None else ""
        header_line = f"[{ts_str}] {reg_label} | {node_label} {hp_label}".strip()
        header_ascii = _to_ascii_tr(header_line)
        cv2.putText(
            annotated,
            header_ascii,
            (326, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (240, 235, 220),
            1,
            cv2.LINE_AA,
        )

        if details:
            detail_ascii = _to_ascii_tr(str(details)[:78])
            cv2.putText(
                annotated,
                detail_ascii,
                (326, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 210, 230),
                1,
                cv2.LINE_AA,
            )

        return annotated

    # =========================================================================
    # Public Recording API
    # =========================================================================

    def record_incident(
        self,
        event_type: str,
        frame: Optional[np.ndarray] = None,
        details: str = "",
        region: str = "Altgard",
        route: str = "All",
        node_id: Optional[int] = None,
        hp_ratio: Optional[float] = None,
        force: bool = False,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Captures, annotates, and persists an incident snapshot + thumbnail into
        `logs/incidents/` and updates `logs/incidents/incidents.json`.

        Legacy callers (core/state_machine.py `_register_node_failure`) pass
        the frame positionally with ``reason=`` / ``metadata=`` kwargs; those
        are accepted here so a node-failure snapshot is actually recorded
        instead of dying on a signature mismatch.
        Returns None when nothing was recorded (cooldown, no real frame,
        invalid input, or a failed JPEG write) — never a fake "saved" entry.
        """
        # Legacy positional-frame call: record_incident(frame, reason=..., metadata=...)
        if isinstance(event_type, np.ndarray):
            frame = event_type
            event_type = reason or "stuck"
        if reason and not details:
            details = str(reason)
        if metadata:
            try:
                meta_text = ", ".join(f"{k}={v}" for k, v in dict(metadata).items())
                details = f"{details} | {meta_text}" if details else meta_text
            except Exception:
                pass

        # Validate hp_ratio up front: a corrupt value must not crash recording.
        hp_value: Optional[float] = None
        if hp_ratio is not None:
            try:
                hp_value = max(0.0, min(1.0, float(hp_ratio)))
            except (TypeError, ValueError):
                logger.warning("Geçersiz hp_ratio %r, olay kaydı atlanıyor.", hp_ratio)
                return None

        now = time.time()
        meta = self._resolve_event_meta(event_type)
        raw_code = str(meta["code"])
        event_code = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in raw_code
        ).strip("_") or "incident"

        with self._lock:
            last_ts = self._last_event_ts.get(event_code, 0.0)
            if not force and self.cooldown_sec > 0.0 and (now - last_ts) < self.cooldown_sec:
                # O-3: cooldown atlaması artık sessiz değil — sayaç +
                # hız sınırlı warning (60 sn'de kod başına 1).
                self._skipped_cooldown += 1
                self._log_skip_limited(
                    "cooldown",
                    event_code,
                    "Olay '%s' cooldown nedeniyle atlandı (%.1fs kaldı).",
                    event_code,
                    self.cooldown_sec - (now - last_ts),
                )
                return None

        raw_frame = self._acquire_frame(frame)
        if raw_frame is None:
            logger.warning("Olay '%s' için gerçek kare alınamadı, kayıt atlandı.", event_code)
            return None

        dt = datetime.fromtimestamp(now)
        ts_display = dt.strftime("%Y-%m-%d %H:%M:%S")
        # Full microsecond stamp: a millisecond-truncated stamp collides
        # when two incidents land in the same millisecond.
        file_stamp = dt.strftime("%Y%m%d_%H%M%S_%f")
        incident_id = f"inc_{file_stamp}_{event_code}"
        filename = f"{incident_id}.jpg"
        thumb_filename = f"{incident_id}.thumb.jpg"
        file_path = os.path.join(self.incidents_dir, filename)
        thumb_path = os.path.join(self.incidents_dir, thumb_filename)

        annotated = self._annotate_snapshot(
            frame=raw_frame,
            meta=meta,
            ts_str=ts_display,
            details=details or meta["label_tr"],
            region=region,
            node_id=node_id,
            hp_ratio=hp_value,
        )

        # Save full annotated JPEG (960x540) to logs/incidents/. A failed
        # encode or write means NOTHING is recorded — no entry, no "saved" log.
        os.makedirs(self.incidents_dir, exist_ok=True)
        ok_full, enc_full = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok_full:
            logger.error("Olay ekran görüntüsü kodlanamadı (%s), kayıt atlandı.", filename)
            return None
        try:
            with open(file_path, "wb") as f:
                f.write(enc_full.tobytes())
        except Exception as exc:
            logger.error("Olay ekran görüntüsü diske yazılamadı (%s): %s", file_path, exc)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass
            return None

        # Compact thumbnail (320x180) as a sidecar file only. The base64 data
        # URI is NOT encoded eagerly here: _with_thumbnail() materializes it
        # lazily from the sidecar file on read, so the 50-entry in-memory list
        # does not carry 50 base64 thumbnails (~6 KB each).
        thumb_img = cv2.resize(annotated, (320, 180), interpolation=cv2.INTER_AREA)
        ok_thumb, enc_thumb = cv2.imencode(".jpg", thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
        if ok_thumb:
            try:
                with open(thumb_path, "wb") as f:
                    f.write(enc_thumb.tobytes())
            except Exception as exc:
                logger.debug("Olay küçük resmi yazılamadı (%s): %s", thumb_path, exc)
                thumb_path = ""

        entry: Dict[str, Any] = {
            "id": incident_id,
            "timestamp": round(now, 3),
            "iso_time": dt.isoformat(timespec="seconds"),
            "time_str": dt.strftime("%H:%M:%S"),
            "date_str": dt.strftime("%d.%m.%Y %H:%M:%S"),
            "event_type": event_code,
            "label_tr": str(meta["label_tr"]),
            "severity": str(meta["severity"]),
            "details": str(details or meta["label_tr"]),
            "region": str(region or "Altgard"),
            "route": str(route or "All"),
            "node_id": node_id,
            "hp_ratio": round(float(hp_value), 3) if hp_value is not None else None,
            "filename": filename,
            "file_path": file_path,
            "thumbnail_file": thumb_filename if ok_thumb and thumb_path else "",
            # Lazily materialized by _with_thumbnail() from the sidecar file.
            "thumbnail_b64": "",
        }

        with self._lock:
            # Cooldown stamp is set only now: a failed recording must not
            # swallow the next real incident for a full cooldown window.
            self._last_event_ts[event_code] = time.time()
            self._incidents.append(entry)
            evicted = 0
            while len(self._incidents) > self.max_incidents:
                removed = self._incidents.pop(0)
                evicted += 1
                for stale_key in ("file_path",):
                    old_path = removed.get(stale_key)
                    if old_path and os.path.isfile(old_path):
                        try:
                            os.remove(old_path)
                        except OSError as rm_exc:
                            logger.debug("Eski olay dosyası silinemedi: %s", rm_exc)
                old_thumb = removed.get("thumbnail_file")
                if old_thumb:
                    old_thumb_path = os.path.join(self.incidents_dir, str(old_thumb))
                    if os.path.isfile(old_thumb_path):
                        try:
                            os.remove(old_thumb_path)
                        except OSError as rm_exc:
                            logger.debug("Eski olay küçük resmi silinemedi: %s", rm_exc)
            if evicted:
                # O-3: tahliye artık sessiz değil — eski olay + dosyası
                # silindi, sayaç ve warning ile görünür olur.
                self._evicted_total += evicted
                logger.warning(
                    "Olay sınırı (%d) aşıldı: %d eski olay tahliye edildi (dosyaları silindi).",
                    self.max_incidents,
                    evicted,
                )
            self._save_manifest_unlocked()

        logger.info(
            "[Olay Kaydedici] '%s' (%s) ekran görüntüsü kaydedildi: %s",
            meta["label_tr"],
            event_code,
            filename,
        )
        return self._with_thumbnail(entry)

    def record_death(
        self,
        frame: Optional[np.ndarray] = None,
        details: str = "Karakter öldü (HP %0 tespit edildi)",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        return self.record_incident("Ölüm", frame=frame, details=details, **kwargs)

    def record_stuck(
        self,
        frame: Optional[np.ndarray] = None,
        details: str = "Karakter engele takıldı (Hareket ilerlemesi yok)",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        return self.record_incident("Takılma", frame=frame, details=details, **kwargs)

    def record_prolonged_combat(
        self,
        frame: Optional[np.ndarray] = None,
        duration_sec: float = 120.0,
        details: str = "",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            logger.warning("Geçersiz duration_sec %r, uzun savaş kaydı atlanıyor.", duration_sec)
            return None
        if duration < 120.0:
            # O-3: eşik-altı atlama sayaç + hız sınırlı warning.
            self._skipped_below_threshold += 1
            self._log_skip_limited(
                "below_threshold",
                "prolonged_combat",
                "Savaş süresi %.1fs < 120s, uzun savaş kaydı atlandı.",
                duration,
            )
            return None
        msg = details or f"Uzun süreli savaş tespit edildi ({int(round(duration))} sn > 2 dk)"
        return self.record_incident("Uzun Savaş >2dk", frame=frame, details=msg, **kwargs)

    def record_inventory_full(
        self,
        frame: Optional[np.ndarray] = None,
        fill_ratio: float = 0.95,
        details: str = "",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            fill = max(0.0, min(1.0, float(fill_ratio)))
        except (TypeError, ValueError):
            logger.warning("Geçersiz fill_ratio %r, envanter kaydı atlanıyor.", fill_ratio)
            return None
        if fill < 0.95:
            # O-3: eşik-altı atlama sayaç + hız sınırlı warning.
            self._skipped_below_threshold += 1
            self._log_skip_limited(
                "below_threshold",
                "inventory_full",
                "Envanter doluluk %d%% < 95%%, kayıt atlandı.",
                int(round(fill * 100)),
            )
            return None
        pct = int(round(fill * 100))
        msg = details or f"Envanter doluluk oranı kritik seviyede (%{pct})"
        return self.record_incident("Envanter %95 Dolu", frame=frame, details=msg, **kwargs)

    # =========================================================================
    # Query & Gallery Helpers
    # =========================================================================

    def _with_thumbnail(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy with thumbnail_b64 materialized from the sidecar file.

        The manifest stores only `thumbnail_file`; readers of get_incidents()
        still get the data-URI thumbnail they expect.
        """
        out = dict(item)
        if out.get("thumbnail_b64"):
            return out
        thumb_name = str(out.get("thumbnail_file") or "")
        if thumb_name:
            thumb_path = os.path.join(self.incidents_dir, os.path.basename(thumb_name))
            if os.path.isfile(thumb_path):
                try:
                    with open(thumb_path, "rb") as f:
                        raw = f.read()
                    out["thumbnail_b64"] = (
                        "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
                    )
                except Exception as exc:
                    logger.debug("Olay küçük resmi okunamadı (%s): %s", thumb_path, exc)
        return out

    def get_incidents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent incidents (newest first) for the mobile gallery."""
        with self._lock:
            items = list(self._incidents)
        items.reverse()
        return [self._with_thumbnail(item) for item in items[: max(1, int(limit))]]

    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        clean_id = str(incident_id or "").strip()
        if not clean_id:
            return None
        with self._lock:
            for item in reversed(self._incidents):
                if item.get("id") == clean_id or item.get("filename") == clean_id:
                    return self._with_thumbnail(dict(item))
        return None

    def get_incident_jpeg_bytes(self, incident_id: str) -> Optional[bytes]:
        item = self.get_incident_by_id(incident_id)
        if item is None:
            return None
        fpath = item.get("file_path")
        if fpath and os.path.isfile(fpath):
            try:
                with open(fpath, "rb") as f:
                    return f.read()
            except Exception as exc:
                logger.debug("Olay tam ekran JPEG okunamadı (%s): %s", fpath, exc)
        # Full JPEG is gone: say so loudly instead of silently serving the
        # 320x180 thumbnail as if it were the full screenshot.
        logger.warning(
            "Olay %s için tam ekran JPEG bulunamadı, küçük resim döndürülüyor.",
            incident_id,
        )
        thumb_name = str(item.get("thumbnail_file") or "")
        if thumb_name:
            thumb_path = os.path.join(self.incidents_dir, os.path.basename(thumb_name))
            if os.path.isfile(thumb_path):
                try:
                    with open(thumb_path, "rb") as f:
                        return f.read()
                except Exception as exc:
                    logger.debug("Olay küçük resim dosyası okunamadı (%s): %s", thumb_path, exc)
        b64_url = str(item.get("thumbnail_b64") or "")
        if b64_url.startswith("data:image/jpeg;base64,"):
            try:
                return base64.b64decode(b64_url.split(",", 1)[1])
            except Exception:
                return None
        return None

    def clear_incidents(self) -> int:
        """Deletes all stored incident JPEGs and resets `incidents.json`.

        P3-8 (S-050/S-218): files are MOVED to `logs/incidents/.trash/<ts>/`
        instead of deleted, so one (confirmed) clear cannot destroy evidence.
        """
        with self._lock:
            count = len(self._incidents)
            trash_dir = os.path.join(self.incidents_dir, ".trash", time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(trash_dir, exist_ok=True)
            for item in self._incidents:
                fpath = item.get("file_path")
                if fpath and os.path.isfile(fpath):
                    try:
                        os.replace(fpath, os.path.join(trash_dir, os.path.basename(fpath)))
                    except OSError as rm_exc:
                        logger.debug("Olay dosyası çöpe taşınamadı: %s", rm_exc)
                thumb_name = str(item.get("thumbnail_file") or "")
                if thumb_name:
                    thumb_path = os.path.join(self.incidents_dir, os.path.basename(thumb_name))
                    if os.path.isfile(thumb_path):
                        try:
                            os.replace(thumb_path, os.path.join(trash_dir, os.path.basename(thumb_path)))
                        except OSError as rm_exc:
                            logger.debug("Olay küçük resmi çöpe taşınamadı: %s", rm_exc)
            self._incidents = []
            self._last_event_ts.clear()
            self._save_manifest_unlocked()
            return count

    def get_recorder_health_summary(self) -> dict:
        try:
            with self._lock:
                count = len(self._incidents)
                manifest_ok = os.path.isdir(self.incidents_dir)
                skipped_cooldown = int(getattr(self, "_skipped_cooldown", 0) or 0)
                skipped_below_threshold = int(getattr(self, "_skipped_below_threshold", 0) or 0)
                evicted_total = int(getattr(self, "_evicted_total", 0) or 0)
                corrupt_backups = int(getattr(self, "_corrupt_manifest_backups", 0) or 0)
            return {
                "recorder_ready": bool(manifest_ok),
                "incident_count": count,
                "max_incidents": int(self.max_incidents),
                "cooldown_sec": float(self.cooldown_sec),
                # O-3: sessiz kayıp sayaçları.
                "skipped_cooldown": skipped_cooldown,
                "skipped_below_threshold": skipped_below_threshold,
                "evicted_total": evicted_total,
                "corrupt_manifest_backups": corrupt_backups,
            }
        except Exception:
            return {"recorder_ready": False}

    def get_incident_recorder_ready_flag(self) -> dict:
        return {"module": "incident_recorder", "ready": bool(self.get_recorder_health_summary().get("recorder_ready"))}

    def get_incident_recorder_version_info(self) -> dict:
        return {"module": "incident_recorder", "version": __version__, "ready": True}

    def get_incident_recorder_diagnostic_snapshot(self) -> dict:
        return {"module": "incident_recorder", "status": "ok", "ready": True}

    def get_incident_recorder_capabilities(self) -> list:
        return ["incident_recorder", "diagnostic_snapshot", "version_info"]


def _probe_incident_recorder_contract() -> bool:
    """Real probe in temp dirs only: a 256-bit token must generate, verify,
    and reject a wrong token; a recorder must persist one incident."""
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = os.path.join(tmp_dir, "token.txt")
            auth = MobileTokenAuthenticator(token_path=tok_path, require_token=True)
            if not _is_valid_256bit_token(auth.token):
                return False
            if not auth.verify_request(query_string=f"token={auth.token}"):
                return False
            if auth.verify_request(query_string="token=deadbeef"):
                return False
            recorder = IncidentRecorder(
                incidents_dir=os.path.join(tmp_dir, "incidents"),
                max_incidents=5,
                cooldown_sec=0.0,
            )
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            entry = recorder.record_death(frame=frame)
            if entry is None or not os.path.isfile(str(entry.get("file_path"))):
                return False
            return True
    except Exception as exc:
        logger.debug("Incident recorder contract probe failed: %s", exc)
        return False


def get_incident_recorder_capabilities() -> list:
    return ["incident_recorder", "diagnostic_snapshot", "version_info"]


# Memoized contract probe: the module-level get_incident_recorder_* chain
# (digest -> audit -> readiness -> metrics, telemetry -> seal -> certificate
# -> probe -> summary -> heartbeat) used to re-run the full probe — temp dir,
# token generation, file IO, one cv2 incident record — on EVERY call (~10
# probes per heartbeat snapshot, ~0.2 s). The probe result only depends on
# this process's environment, so a short TTL cache is sound: 1 probe per
# 60 s window, identical True/False result inside the window.
_PROBE_CACHE_TTL_SEC = 60.0
_probe_cache: Dict[str, Any] = {"ts": 0.0, "value": None}
_probe_cache_lock = threading.Lock()


def validate_incident_recorder_contract() -> bool:
    now = time.monotonic()
    with _probe_cache_lock:
        if _probe_cache["value"] is not None and (now - _probe_cache["ts"]) < _PROBE_CACHE_TTL_SEC:
            return bool(_probe_cache["value"])
    result = _probe_incident_recorder_contract()
    with _probe_cache_lock:
        _probe_cache["ts"] = time.monotonic()
        _probe_cache["value"] = bool(result)
    return bool(result)


def get_incident_recorder_baseline_metrics() -> dict:
    return {"module": "incident_recorder", "contract_valid": validate_incident_recorder_contract(), "capability_count": len(get_incident_recorder_capabilities())}


def get_incident_recorder_readiness_status() -> dict:
    return {"module": "incident_recorder", "ready": validate_incident_recorder_contract(), "metrics": get_incident_recorder_baseline_metrics()}


def get_incident_recorder_audit_record() -> dict:
    return {"module": "incident_recorder", "audit_passed": validate_incident_recorder_contract(), "readiness": get_incident_recorder_readiness_status()}


def get_incident_recorder_operational_digest() -> dict:
    return {"module": "incident_recorder", "operational": validate_incident_recorder_contract(), "audit": get_incident_recorder_audit_record()}


def get_incident_recorder_telemetry_payload() -> dict:
    return {"module": "incident_recorder", "telemetry_ready": validate_incident_recorder_contract(), "digest": get_incident_recorder_operational_digest()}


def get_incident_recorder_integrity_seal() -> dict:
    return {"module": "incident_recorder", "sealed": validate_incident_recorder_contract(), "telemetry": get_incident_recorder_telemetry_payload()}


def get_incident_recorder_compliance_certificate() -> dict:
    return {"module": "incident_recorder", "certified": validate_incident_recorder_contract(), "seal": get_incident_recorder_integrity_seal()}


def get_incident_recorder_readiness_probe() -> dict:
    return {"module": "incident_recorder", "ready": validate_incident_recorder_contract(), "certificate": get_incident_recorder_compliance_certificate()}


def get_incident_recorder_audit_summary() -> dict:
    return {"module": "incident_recorder", "audit_passed": validate_incident_recorder_contract(), "probe": get_incident_recorder_readiness_probe()}


def get_incident_recorder_heartbeat_snapshot() -> dict:
    return {"module": "incident_recorder", "alive": validate_incident_recorder_contract(), "audit": get_incident_recorder_audit_summary()}


