# -*- coding: utf-8 -*-
"""
Katman V: Şifreli Yerel Veri Kasası, Native RegisterHotKey (Hook'suz Kısayol) ve Panik/Telegram Uyarı Sistemi
-------------------------------------------------------------------------------------------------------------
Modül bileşenleri:
1. LocalVaultCrypto:
   - Kasaya özel rastgele gizli + makine parmak izinden PBKDF2-HMAC-SHA256 (600.000 iterasyon)
     ile makineye özel 256-bit şifreleme anahtarı türetir (eski 100.000-iterasyon anahtarı
     yalnızca eski kasaların çözülmesinde denenir).
   - HMAC-SHA256 bütünlük doğrulamalı (Authenticated Encryption - Encrypt-then-MAC) CTR-SHA256 akış
     şifrelemesi (veya mevcutsa cryptography AES-GCM) kullanarak konfigürasyon ve rota verilerini
     şifreli olarak saklar ve okur.
   - Her kullanıcıya özel yerel salt ve çalışma zamanı imzası üretir.

2. NativeWin32HotkeyManager:
   - Global SetWindowsHookEx(WH_KEYBOARD_LL) kancası kurmadan, doğrudan Windows standart
     RegisterHotKey ve PeekMessageW (WM_HOTKEY = 0x0312) API'sini kullanan arka plan thread'i.
   - F8 (VK_F8 = 0x77) ve F9 (VK_F9 = 0x78) başta olmak üzere sanal tuş kodlarını dinler.

3. PanicAlertSystem:
   - 15 dakika boşta kalma, 2 dakikadan uzun süren savaş, karakter ölümü veya "Prompt stuck for 10s"
     gibi acil durumlarda:
     1) Anında tüm sanal girdi gönderimini kilitler (emergency_halt_inputs).
     2) Çok tonlu sesli alarm çalar (play_panic_siren).
     3) Telegram Bot API veya Discord Webhook üzerinden anında acil durum bildirimi gönderir.
"""

from __future__ import annotations

import base64
import copy
import ctypes
from ctypes import wintypes
import getpass
import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from version import __version__

logger = logging.getLogger("OpsecCrypto")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# İsteğe bağlı AES-GCM desteği kontrolü
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

    _HAS_AESGCM = True
except Exception:
    AESGCM = None  # type: ignore
    _HAS_AESGCM = False


# ============================================================================
# 1. LocalVaultCrypto (Makineye Özel Şifreli Konfigürasyon & Benzersiz İmza)
# ============================================================================

class LocalVaultCrypto:
    """
    Makineye özel 256-bit anahtar türetimi (PBKDF2-HMAC-SHA256, 600.000 iterasyon)
    ve kimlik doğrulamalı şifreleme (Authenticated Encryption) sağlayan yerel veri kasası.
    """

    PBKDF2_ITERATIONS: int = 600_000
    LEGACY_PBKDF2_ITERATIONS: int = 100_000
    KEY_LENGTH: int = 32  # 256-bit
    NONCE_LENGTH: int = 16  # 128-bit
    MAC_LENGTH: int = 32  # 256-bit HMAC-SHA256
    VERSION_CTR_HMAC: int = 1
    VERSION_AES_GCM: int = 2
    # F-0165: PBKDF2 "parolası" artık yalnızca herkese açık makine parmak izi
    # değil; parmak izi + kasaya özel 32 baytlık rastgele gizli (0600 dosya).
    # Eski kasalar için parmak-izine dayalı eski anahtar yalnızca ÇÖZMEDE
    # denenir (geriye dönük uyumluluk); şifreleme hep yeni anahtarla yapılır.
    VAULT_SECRET_FILENAME: str = ".vault_secret"
    MIN_SALT_BYTES: int = 16

    def __init__(
        self,
        vault_path: str = os.path.join("config", "settings.enc"),
        salt_path: str = os.path.join("config", ".vault_salt"),
        prefer_aes_gcm: bool = True,
        vault_secret_path: Optional[str] = None,
    ) -> None:
        self.vault_path = vault_path
        self.salt_path = salt_path
        if vault_secret_path is None:
            salt_dir = os.path.dirname(os.path.abspath(salt_path))
            vault_secret_path = os.path.join(salt_dir, self.VAULT_SECRET_FILENAME)
        self.vault_secret_path = vault_secret_path
        self.prefer_aes_gcm = bool(prefer_aes_gcm and _HAS_AESGCM)
        if prefer_aes_gcm and not _HAS_AESGCM:
            # F-0350: sessiz downgrade yok — AES-GCM yoksa CTR-HMAC yedeği
            # kullanıldığı açıkça loglanır.
            logger.warning(
                "AES-GCM için 'cryptography' paketi bulunamadı; "
                "CTR-HMAC-SHA256-EtM yedeği kullanılıyor (AES değildir)."
            )
        self._lock = threading.RLock()
        self._local_salt: bytes = self._load_or_create_local_salt()
        # F-0165: anahtar malzemesi artık rastgele kasa gizlisi içerir.
        self._vault_secret: bytes = self._load_or_create_vault_secret()
        # D23: makine parmak izi süreç ömrü boyunca değişmez (kullanıcı,
        # makine adı, MAC, CPU, işletim sistemi, disk seri no); eskiden
        # __init__'te iki kez toplanıyordu (master + legacy türetimi).
        # Bir kez toplanır, iki türetime de aynı baytlar verilir.
        self._machine_fingerprint: bytes = self._collect_machine_fingerprint()
        self._master_key: bytes = self._derive_machine_key(
            self._local_salt,
            secret=self._vault_secret,
            fingerprint=self._machine_fingerprint,
        )
        self._enc_key: bytes = hmac.new(self._master_key, b"ENC-KEY-V1", hashlib.sha256).digest()
        self._mac_key: bytes = hmac.new(self._master_key, b"MAC-KEY-V1", hashlib.sha256).digest()
        # D23: eski anahtar (100.000 iterasyon, gizlisiz parmak izi)
        # artık init'te türetilmez — yalnızca eski formatlı bir kasa
        # çözülürken _get_legacy_key_pair() içinde tembel türetilir.
        # Yeni kasalarda ~178 ms'lik PBKDF2 maliyeti ödenmez.
        self._legacy_key_pair: Optional[Tuple[bytes, bytes]] = None

    def _get_legacy_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Eski (yalnızca çözme) anahtar çifti: ilk ihtiyaçta türetilir ve
        önbelleğe alınır; sonraki çözmeler aynı çifti kullanır.
        """
        with self._lock:
            if self._legacy_key_pair is None:
                legacy_master = self._derive_machine_key(
                    self._local_salt,
                    secret=None,
                    iterations=self.LEGACY_PBKDF2_ITERATIONS,
                    fingerprint=self._machine_fingerprint,
                )
                self._legacy_key_pair = (
                    hmac.new(legacy_master, b"ENC-KEY-V1", hashlib.sha256).digest(),
                    hmac.new(legacy_master, b"MAC-KEY-V1", hashlib.sha256).digest(),
                )
            return self._legacy_key_pair

    @staticmethod
    def _chmod_private(path: str) -> None:
        """Gizli dosyayı yalnızca sahibinin okuyabileceği hale getir (0600)."""
        if os.name == "nt":
            return
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logger.debug("chmod 0600 başarısız (%s): %s", path, exc)

    @staticmethod
    def _write_private_bytes(path: str, data: bytes) -> None:
        """Baytları 0600 izinle yazar (varsa mevcut dosyanın iznini de daraltır)."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        LocalVaultCrypto._chmod_private(path)

    def _load_or_create_local_salt(self) -> bytes:
        """
        Kullanıcıya özel yerel 32-bayt salt değerini diskten okur; yoksa kriptografik
        olarak güvenli rastgele salt üretip 0600 izinle kaydeder.

        F-0040: Salt dosyası bayt bayt okunur — .strip() YOK. Rastgele saltın
        uç baytı boşluk baytı olsa bile ilk oturumda kullanılan salt ile
        yeniden başlatmada okunan salt aynıdır.
        F-0346: 16 bayttan kısa/bozuk salt dosyası sessizce üzerine yazılmaz;
        ValueError fırlatılır (eski şifreli veri kaybolmasın diye).
        F-0486: Salt 0600 izinle yazılır.
        """
        salt_dir = os.path.dirname(os.path.abspath(self.salt_path))
        if salt_dir and not os.path.exists(salt_dir):
            os.makedirs(salt_dir, exist_ok=True)

        if os.path.exists(self.salt_path):
            with open(self.salt_path, "rb") as f:
                raw = f.read()
            if len(raw) < self.MIN_SALT_BYTES:
                raise ValueError(
                    f"Yerel salt dosyası bozuk/kısa ({len(raw)} bayt < "
                    f"{self.MIN_SALT_BYTES}): {self.salt_path} — üzerine yazılmadı; "
                    "dosyayı yedekten geri yükleyin ya da silip kasayı sıfırlayın."
                )
            self._chmod_private(self.salt_path)
            return raw[:32]

        new_salt = secrets.token_bytes(32)
        self._write_private_bytes(self.salt_path, new_salt)
        return new_salt

    def _load_or_create_vault_secret(self) -> bytes:
        """
        F-0165: PBKDF2 anahtar malzemesine giren kasaya özel 32 baytlık rastgele
        gizli. Salt dosyasının yanındaki .vault_secret dosyasında 0600 izinle
        saklanır. Dosya okunamaz/yazılamaz ya da bozuksa deterministik yedek
        (F-0164) kullanılmaz — fail-closed: hata fırlatılır.
        """
        secret_dir = os.path.dirname(os.path.abspath(self.vault_secret_path))
        if secret_dir and not os.path.exists(secret_dir):
            os.makedirs(secret_dir, exist_ok=True)

        if os.path.exists(self.vault_secret_path):
            with open(self.vault_secret_path, "rb") as f:
                raw = f.read()
            if len(raw) < self.MIN_SALT_BYTES:
                raise ValueError(
                    f"Kasa gizli dosyası bozuk/kısa ({len(raw)} bayt): "
                    f"{self.vault_secret_path} — üzerine yazılmadı."
                )
            self._chmod_private(self.vault_secret_path)
            return raw[:32]

        new_secret = secrets.token_bytes(32)
        self._write_private_bytes(self.vault_secret_path, new_secret)
        return new_secret

    @staticmethod
    def _collect_machine_fingerprint() -> bytes:
        """
        Bilgisayarın donanım ve kullanıcı kimliği bileşenlerini birleştirerek
        makineye özgü ham parmak izi bayt dizisi oluşturur.
        """
        try:
            username = getpass.getuser()
        except Exception:
            username = os.environ.get("USERNAME", os.environ.get("USER", "unknown_user"))

        node_name = platform.node() or os.environ.get("COMPUTERNAME", "unknown_host")
        mac_int = uuid.getnode()
        arch = platform.machine() or "unknown_arch"
        proc = platform.processor() or "unknown_cpu"
        win_ver = platform.system() + "-" + platform.release()

        volume_serial = "00000000"
        if os.name == "nt":
            try:
                kernel32 = ctypes.windll.kernel32
                vol_name_buf = ctypes.create_unicode_buffer(261)
                fs_name_buf = ctypes.create_unicode_buffer(261)
                serial_num = wintypes.DWORD(0)
                max_comp_len = wintypes.DWORD(0)
                fs_flags = wintypes.DWORD(0)
                sys_drive = os.environ.get("SystemDrive", "C:") + "\\"
                res = kernel32.GetVolumeInformationW(
                    ctypes.c_wchar_p(sys_drive),
                    vol_name_buf,
                    ctypes.sizeof(vol_name_buf),
                    ctypes.byref(serial_num),
                    ctypes.byref(max_comp_len),
                    ctypes.byref(fs_flags),
                    fs_name_buf,
                    ctypes.sizeof(fs_name_buf),
                )
                if res != 0:
                    volume_serial = f"{serial_num.value:08X}"
            except Exception as exc:
                logger.debug("Disk seri numarası alınamadı: %s", exc)

        raw_identity = (
            f"USER={username}|HOST={node_name}|MAC={mac_int:012x}|"
            f"ARCH={arch}|CPU={proc}|OS={win_ver}|VOL={volume_serial}"
        )
        return raw_identity.encode("utf-8", errors="replace")

    def _derive_machine_key(
        self,
        salt: bytes,
        secret: Optional[bytes] = None,
        iterations: Optional[int] = None,
        fingerprint: Optional[bytes] = None,
    ) -> bytes:
        """
        PBKDF2-HMAC-SHA256 (600.000 iterasyon, OWASP önerisi) ile makineye özel
        256-bit anahtar türetir.

        F-0348: iterasyon 100.000 -> 600.000. Eski kasalar için eski anahtar
        (100.000 iterasyon, gizlisiz parmak izi) yalnızca çözmede denenir.
        F-0165: secret verilirse PBKDF2 parolası = gizli || parmak izi;
        secret=None ise eski davranış (yalnızca parmak izi, çözme uyumluluğu).
        D23: fingerprint verilirse parmak izi yeniden toplanmaz — __init__
        tek toplanan parmak izini master + legacy türetime aynen aktarır.
        """
        if fingerprint is None:
            fingerprint = self._collect_machine_fingerprint()
        password = (secret + b"|" + fingerprint) if secret is not None else fingerprint
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password,
            salt,
            int(iterations) if iterations is not None else self.PBKDF2_ITERATIONS,
            dklen=self.KEY_LENGTH,
        )
        return derived

    @staticmethod
    def _ctr_xor_sha256_stream(key: bytes, nonce: bytes, payload: bytes) -> bytes:
        """
        HMAC-SHA256 tabanlı sayaç (CTR) modunda akış şifrelemesi / çözmesi.
        Her 32 baytlık blok için HMAC-SHA256(key, nonce || uint64_be(counter)) anahtar akışı üretilir.

        D23: XOR blok başına int tabanlı yapılır — eski bayt bayt Python
        döngüsüyle birebir aynı anahtar akışı ve aynı çıktı, ~10 kat hızlı.
        (Bu VM'de 'cryptography' kurulu değil; CTR-HMAC yedeği canlı yoldur.)
        """
        out = bytearray(len(payload))
        for offset in range(0, len(payload), 32):
            chunk = payload[offset : offset + 32]
            counter_bytes = struct.pack(">Q", offset // 32)
            keystream_block = hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest()
            n = len(chunk)
            xored = (
                int.from_bytes(chunk, "big") ^ int.from_bytes(keystream_block[:n], "big")
            ).to_bytes(n, "big")
            out[offset : offset + n] = xored

        return bytes(out)

    def encrypt_json_payload(self, data: Dict[str, Any]) -> str:
        """
        Sözlük (dict) verisini JSON olarak serileştirir ve kimlik doğrulamalı olarak şifreler.
        Dönen değer URL-safe Base64 kodlu token dizgesidir.
        """
        if not isinstance(data, dict):
            raise TypeError("encrypt_json_payload yalnızca dict tipinde veri kabul eder.")

        with self._lock:
            plaintext = json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

            if self.prefer_aes_gcm and AESGCM is not None:
                nonce = secrets.token_bytes(12)
                aad = b"VAULT-V2-AESGCM"
                aesgcm = AESGCM(self._enc_key)
                ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
                # Paket formatı: [version(1B)=2][nonce(12B)][ciphertext+tag]
                packet = bytes([self.VERSION_AES_GCM]) + nonce + ciphertext_and_tag
                return base64.urlsafe_b64encode(packet).decode("ascii")

            # Varsayılan taşınabilir mod: CTR-HMAC-SHA256 + Encrypt-then-MAC (HMAC-SHA256)
            nonce = secrets.token_bytes(self.NONCE_LENGTH)
            ciphertext = self._ctr_xor_sha256_stream(self._enc_key, nonce, plaintext)
            header_and_body = bytes([self.VERSION_CTR_HMAC]) + nonce + ciphertext
            mac_tag = hmac.new(self._mac_key, header_and_body, hashlib.sha256).digest()
            packet = header_and_body + mac_tag
            return base64.urlsafe_b64encode(packet).decode("ascii")

    def decrypt_json_payload(self, token: str) -> Dict[str, Any]:
        """
        encrypt_json_payload ile üretilmiş Base64 token'ı doğrular ve çözer.
        Bütünlük doğrulaması (HMAC / GCM Tag) başarısız olursa ValueError fırlatır.
        """
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Şifreli token boş veya geçersiz.")

        with self._lock:
            try:
                packet = base64.urlsafe_b64decode(token.strip().encode("ascii"))
            except Exception as exc:
                raise ValueError(f"Base64 çözme hatası: {exc}") from exc

            if len(packet) < 2:
                raise ValueError("Şifreli paket boyutu çok kısa.")

            version = packet[0]

            if version == self.VERSION_AES_GCM:
                if not _HAS_AESGCM or AESGCM is None:
                    raise RuntimeError("AES-GCM ile şifrelenmiş veri için 'cryptography' paketi gereklidir.")
                if len(packet) < 1 + 12 + 16:
                    raise ValueError("AES-GCM paketi bozuk veya eksik.")
                nonce = packet[1:13]
                ciphertext_and_tag = packet[13:]
                aad = b"VAULT-V2-AESGCM"
                plaintext = None
                last_exc: Optional[Exception] = None
                try:
                    plaintext = AESGCM(self._enc_key).decrypt(
                        nonce, ciphertext_and_tag, aad
                    )
                except Exception as exc:
                    last_exc = exc
                if plaintext is None:
                    # D23: eski anahtar yalnızca yeni anahtar çözemezse
                    # türetilir (tembel); yeni kasalarda maliyeti ödenmez.
                    try:
                        legacy_enc, _legacy_mac = self._get_legacy_key_pair()
                        plaintext = AESGCM(legacy_enc).decrypt(
                            nonce, ciphertext_and_tag, aad
                        )
                    except Exception as exc:
                        last_exc = exc
                if plaintext is None:
                    raise ValueError(f"AES-GCM bütünlük doğrulaması başarısız: {last_exc}") from last_exc

            elif version == self.VERSION_CTR_HMAC:
                min_len = 1 + self.NONCE_LENGTH + self.MAC_LENGTH
                if len(packet) < min_len:
                    raise ValueError("CTR-HMAC paketi bozuk veya eksik.")
                header_and_body = packet[:-self.MAC_LENGTH]
                received_mac = packet[-self.MAC_LENGTH:]
                plaintext = None
                # D23: önce yeni anahtar; eski anahtar yalnızca yeni anahtar
                # MAC'i doğrulayamazsa türetilir (tembel) — yeni kasalarda
                # 100.000 iterasyonluk PBKDF2 maliyeti ödenmez.
                if hmac.compare_digest(
                    received_mac,
                    hmac.new(self._mac_key, header_and_body, hashlib.sha256).digest(),
                ):
                    nonce = header_and_body[1 : 1 + self.NONCE_LENGTH]
                    ciphertext = header_and_body[1 + self.NONCE_LENGTH :]
                    plaintext = self._ctr_xor_sha256_stream(self._enc_key, nonce, ciphertext)
                else:
                    legacy_enc, legacy_mac = self._get_legacy_key_pair()
                    if hmac.compare_digest(
                        received_mac,
                        hmac.new(legacy_mac, header_and_body, hashlib.sha256).digest(),
                    ):
                        nonce = header_and_body[1 : 1 + self.NONCE_LENGTH]
                        ciphertext = header_and_body[1 + self.NONCE_LENGTH :]
                        plaintext = self._ctr_xor_sha256_stream(legacy_enc, nonce, ciphertext)
                if plaintext is None:
                    raise ValueError("HMAC-SHA256 bütünlük doğrulaması başarısız! Veri değiştirilmiş veya anahtar uyumsuz.")
            else:
                raise ValueError(f"Desteklenmeyen şifreleme paket sürümü: {version}")

            try:
                decoded = json.loads(plaintext.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"JSON ayrıştırma hatası: {exc}") from exc

            if not isinstance(decoded, dict):
                raise ValueError("Çözülen JSON verisi bir sözlük (dict) değil.")

            return decoded

    def save_encrypted_config(self, config_data: Dict[str, Any], target_path: Optional[str] = None) -> str:
        """
        Konfigürasyon veya rota verisini şifreleyerek diske (varsayılan: config/settings.enc) yazar.
        Atomik yazım (geçici dosya + os.replace) kullanır.
        """
        path = target_path or self.vault_path
        with self._lock:
            token = self.encrypt_json_payload(config_data)
            parent_dir = os.path.dirname(os.path.abspath(path))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            tmp_path = f"{path}.tmp.{os.getpid()}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(token)
                self._chmod_private(tmp_path)
                os.replace(tmp_path, path)
                logger.info("Şifreli konfigürasyon kaydedildi: %s (%d bayt)", path, len(token))
                return path
            except Exception as exc:
                logger.error("Şifreli konfigürasyon kaydedilemedi (%s): %s", path, exc)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        logger.debug("Geçici dosya silinemedi: %s", tmp_path)
                raise

    def load_encrypted_config(
        self,
        source_path: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Diskteki şifreli konfigürasyon dosyasını okur, doğrular ve çözer.
        Dosya yoksa ve default verilmişse default'un kopyasını döndürür.
        F-0346: Dosya var ama bozuk/çözülemezse default sessizce döndürülmez —
        hata fırlatılır (bozuk kasa default ile geçiştirilmez).
        """
        path = source_path or self.vault_path
        with self._lock:
            if not os.path.exists(path):
                if default is not None:
                    # P3-8 (S-106): sığ kopya iç içe değiştirilebilir
                    # değerleri paylaşırdı — derin kopya döndürülür.
                    return copy.deepcopy(default)
                raise FileNotFoundError(f"Şifreli konfigürasyon dosyası bulunamadı: {path}")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                if not token:
                    raise ValueError(f"Şifreli konfigürasyon dosyası boş: {path}")
                data = self.decrypt_json_payload(token)
                logger.info("Şifreli konfigürasyon başarıyla yüklendi: %s", path)
                return data
            except Exception as exc:
                logger.error("Şifreli konfigürasyon okuma/çözme hatası (%s): %s", path, exc)
                raise

    def get_vault_encryption_summary(self) -> dict:
        """
        Returns non-sensitive metadata about the vault encryption state
        without exposing any secret key or salt bytes.
        """
        with self._lock:
            # F-0350: CTR-HMAC yedeği "AES" diye etiketlenmez — AES değildir.
            cipher = (
                "AES-256-GCM"
                if self.prefer_aes_gcm
                else "CTR-HMAC-SHA256-EtM (AES-GCM fallback, non-AES)"
            )
            return {
                "vault_ready": True,
                "cipher": cipher,
                "vault_path": self.vault_path,
                "kdf_iterations": self.PBKDF2_ITERATIONS,
                "legacy_decrypt_supported": True,
            }

    def get_opsec_crypto_health_summary(self) -> dict:
        """
        S-1002: totoloji değil gerçek prob — bellek-içi şifrele/çöz
        round-trip'i + anahtar malzemesinin varlığını doğrular. Round-trip
        başarısızsa opsec_crypto_ready False döner (False dalı artık ölü
        değil). Dosyaya yazmaz, gizli bayt döndürmez.
        """
        cipher = (
            "AES-256-GCM"
            if self.prefer_aes_gcm
            else "CTR-HMAC-SHA256-EtM (AES-GCM fallback, non-AES)"
        )
        try:
            with self._lock:
                key_ok = bool(self._master_key) and bool(self._local_salt) and bool(self._enc_key)
            if not key_ok:
                return {
                    "opsec_crypto_ready": False,
                    "roundtrip_ok": False,
                    "cipher": cipher,
                    "vault_path": self.vault_path,
                    "error": "key material missing",
                }
            probe = {"health_probe": True}
            token = self.encrypt_json_payload(probe)
            roundtrip_ok = self.decrypt_json_payload(token) == probe
            return {
                "opsec_crypto_ready": bool(roundtrip_ok),
                "roundtrip_ok": bool(roundtrip_ok),
                "cipher": cipher,
                "vault_path": self.vault_path,
            }
        except Exception as exc:
            return {
                "opsec_crypto_ready": False,
                "roundtrip_ok": False,
                "cipher": cipher,
                "vault_path": self.vault_path,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def get_unique_build_signature(self) -> Dict[str, Any]:
        """
        Her kullanıcıya özel yerel salt ve makine anahtarından türetilen benzersiz
        yapı (build) ve çalışma zamanı (runtime) kimliği döndürür.
        """
        with self._lock:
            build_id_bytes = hmac.new(
                self._master_key,
                b"UNIQUE-BUILD-SIGNATURE-V1|" + self._local_salt,
                hashlib.sha256,
            ).digest()
            salt_fingerprint = hashlib.sha256(self._local_salt).hexdigest()[:16]
            runtime_nonce = secrets.token_hex(8)
            runtime_token = hmac.new(
                self._master_key,
                f"RUNTIME|{os.getpid()}|{runtime_nonce}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:24]

            # Makineye özel deterministik pencere/uygulama başlığı etiketi
            mutex_tag = "Global\\Mtx_" + build_id_bytes[:6].hex().upper()
            app_hash = build_id_bytes.hex()[:32]

            return {
                "build_signature": app_hash,
                "salt_fingerprint": salt_fingerprint,
                "runtime_session_id": runtime_token,
                "mutex_name": mutex_tag,
                "kdf_algorithm": "PBKDF2-HMAC-SHA256",
                "kdf_iterations": self.PBKDF2_ITERATIONS,
                "cipher_suite": "AES-256-GCM" if self.prefer_aes_gcm else "CTR-HMAC-SHA256-EtM",
                "pid": os.getpid(),
            }

    def get_opsec_crypto_ready_flag(self) -> dict:
        # S-2201: sabit True değil — gerçek sözleşme probunun sonucu.
        return {"module": "opsec_crypto", "ready": validate_opsec_crypto_contract()}

    def get_opsec_crypto_version_info(self) -> dict:
        return {"module": "opsec_crypto", "version": __version__, "ready": True}

    def get_opsec_crypto_diagnostic_snapshot(self) -> dict:
        # S-2201: sabit "ok"/True değil — gerçek sağlık probu
        # (bellek-içi şifrele/çöz round-trip'i + anahtar malzemesi).
        health = self.get_opsec_crypto_health_summary()
        ready = bool(health.get("opsec_crypto_ready"))
        snapshot: Dict[str, Any] = {
            "module": "opsec_crypto",
            "status": "ok" if ready else "error",
            "ready": ready,
            "cipher": health.get("cipher"),
            "vault_path": health.get("vault_path"),
        }
        if health.get("error"):
            snapshot["error"] = health["error"]
        return snapshot


# ============================================================================
# 2. NativeWin32HotkeyManager (WH_KEYBOARD_LL Kancası Olmadan Kısayol Dinleyici)
# ============================================================================

class NativeWin32HotkeyManager:
    """
    SetWindowsHookEx(WH_KEYBOARD_LL) kullanmadan, doğrudan Windows standart
    user32.RegisterHotKey ve PeekMessageW (WM_HOTKEY = 0x0312) API'sini kullanan
    arka plan mesaj döngüsü yöneticisi.
    """

    WM_HOTKEY: int = 0x0312
    WM_QUIT: int = 0x0012
    PM_REMOVE: int = 0x0001

    # Modifiers
    MOD_NONE: int = 0x0000
    MOD_ALT: int = 0x0001
    MOD_CONTROL: int = 0x0002
    MOD_SHIFT: int = 0x0004
    MOD_WIN: int = 0x0008
    MOD_NOREPEAT: int = 0x4000

    # Sanal Tuş Kodları (Virtual-Key Codes)
    VK_ESCAPE: int = 0x1B
    VK_F1: int = 0x70
    VK_F2: int = 0x71
    VK_F3: int = 0x72
    VK_F4: int = 0x73
    VK_F5: int = 0x74
    VK_F6: int = 0x75
    VK_F7: int = 0x76
    VK_F8: int = 0x77
    VK_F9: int = 0x78
    VK_F10: int = 0x79
    VK_F11: int = 0x7A
    VK_F12: int = 0x7B

    HOTKEY_ID_F8: int = 108
    HOTKEY_ID_F9: int = 109
    HOTKEY_ID_F10: int = 110
    # README / HUD "F9: Menü" — oyun odağındayken de menüyü açmak için
    # yalın F9 RegisterHotKey (Ctrl+Alt+F9 akoruna ek, tasarımı bozmaz).
    HOTKEY_ID_F9_CLASSIC: int = 111

    # Win32 hata kodu: kısayol başka bir program tarafından zaten kayıtlı.
    ERROR_HOTKEY_ALREADY_REGISTERED: int = 1409

    def __init__(self, poll_interval_sec: float = 0.015) -> None:
        self.poll_interval_sec = max(0.002, float(poll_interval_sec))
        self._lock = threading.RLock()
        self._running = threading.Event()
        self._thread_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None

        # hotkey_id -> (vk_code, modifiers, callback, registered_in_os)
        self._hotkeys: Dict[int, Tuple[int, int, Callable[[], None], bool]] = {}
        # EK (dilim-26 F2): son kayıt hatası + akor, hotkey_id bazında saklanır —
        # register_default_f8_f9 yalnızca Dict[str, bool] döndürdüğü için
        # çağıran (main.py) hata kodunu göremez; get_last_register_error /
        # last_register_errors_by_name ile okunur.
        self._last_register_errors: Dict[int, int] = {}
        self._last_register_chords: Dict[int, str] = {}
        self._command_queue: List[Tuple[str, int, int, int, Optional[Callable[[], None]]]] = []
        self._dispatched_count: int = 0

        self._is_windows = os.name == "nt"
        if self._is_windows:
            self._user32 = ctypes.windll.user32
            self._kernel32 = ctypes.windll.kernel32
        else:
            self._user32 = None
            self._kernel32 = None

    @staticmethod
    def _format_chord(vk_code: int, modifiers: int) -> str:
        """(vk, modifiers) -> 'Ctrl+Alt+F8' gibi okunabilir akor adı."""
        parts = []
        if modifiers & NativeWin32HotkeyManager.MOD_CONTROL:
            parts.append("Ctrl")
        if modifiers & NativeWin32HotkeyManager.MOD_ALT:
            parts.append("Alt")
        if modifiers & NativeWin32HotkeyManager.MOD_SHIFT:
            parts.append("Shift")
        if modifiers & NativeWin32HotkeyManager.MOD_WIN:
            parts.append("Win")
        if 0x70 <= vk_code <= 0x87:
            parts.append(f"F{vk_code - 0x70 + 1}")
        elif vk_code:
            parts.append(f"VK_0x{vk_code:02X}")
        return "+".join(parts) if parts else "yalın tuş"

    def get_last_register_error(self, hotkey_id: int) -> Optional[int]:
        """Son RegisterHotKey hata kodunu döner (yoksa/başarılıysa None).

        EK: register_default_f8_f9 yalnızca bool döndürdüğü için hata kodu
        buradan okunur — örn. 1409 = akor başka programda kayıtlı.
        """
        with self._lock:
            return self._last_register_errors.get(int(hotkey_id))

    def last_register_errors_by_name(self) -> Dict[str, int]:
        """Son kayıt hatalarını {'F8': 1409, ...} biçiminde döner."""
        names = {
            self.HOTKEY_ID_F8: "F8",
            self.HOTKEY_ID_F9: "F9",
            self.HOTKEY_ID_F10: "F10",
            self.HOTKEY_ID_F9_CLASSIC: "F9_CLASSIC",
        }
        with self._lock:
            return {
                names.get(hk_id, f"id_{hk_id}"): err
                for hk_id, err in self._last_register_errors.items()
            }

    def register_native_hotkey(
        self,
        hotkey_id: int,
        vk_code: int,
        modifiers: int,
        callback: Callable[[], None],
    ) -> bool:
        """
        Belirtilen sanal tuş kodunu (örn. VK_F8=0x77, VK_F9=0x78) ve değiştirici bayrakları
        (MOD_NONE / MOD_NOREPEAT vb.) RegisterHotKey ile kaydeder.
        Arka plan thread'i henüz başlatılmamışsa otomatik olarak başlatır.
        """
        if not callable(callback):
            raise TypeError("Hotkey callback çağrılabilir (callable) bir fonksiyon olmalıdır.")

        hotkey_id = int(hotkey_id)
        vk_code = int(vk_code)
        modifiers = int(modifiers)

        with self._lock:
            self._hotkeys[hotkey_id] = (vk_code, modifiers, callback, False)
            self._command_queue.append(("register", hotkey_id, vk_code, modifiers, callback))

        if not self.is_running():
            self.start()

        # Mesaj döngüsünün komutu işlemesi için kısa bekleme
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            with self._lock:
                entry = self._hotkeys.get(hotkey_id)
                if entry and entry[3]:
                    return True
            time.sleep(0.005)

        with self._lock:
            entry = self._hotkeys.get(hotkey_id)
            return bool(entry and entry[3])

    @staticmethod
    def vk_for_key(key: str) -> int:
        name = str(key or "").strip().lower()
        if len(name) >= 2 and name[0] == "f" and name[1:].isdigit():
            number = int(name[1:])
            if 1 <= number <= 24:
                return 0x70 + number - 1
        if len(name) == 1 and "a" <= name <= "z":
            return ord(name.upper())
        if name in ("esc", "escape"):
            return 0x1B
        return 0

    @staticmethod
    def chord_binding(
        spec: str, no_repeat: bool = True, allow_bare: bool = False
    ) -> Tuple[int, int]:
        """
        Map a chord spec to (vk, modifiers). Bare F8/F9/F10 return (0, 0) unless
        allow_bare=True (used only for classic menu F9 RegisterHotKey).
        """
        from config.schema import parse_hotkey_chord

        key, mods = parse_hotkey_chord(spec)
        if not key:
            return 0, 0
        if not mods and not allow_bare:
            return 0, 0
        vk = NativeWin32HotkeyManager.vk_for_key(key)
        if not vk:
            return 0, 0
        bits = 0
        if "alt" in mods:
            bits |= NativeWin32HotkeyManager.MOD_ALT
        if "ctrl" in mods:
            bits |= NativeWin32HotkeyManager.MOD_CONTROL
        if "shift" in mods:
            bits |= NativeWin32HotkeyManager.MOD_SHIFT
        if "win" in mods:
            bits |= NativeWin32HotkeyManager.MOD_WIN
        if no_repeat:
            bits |= NativeWin32HotkeyManager.MOD_NOREPEAT
        return vk, bits

    def register_default_f8_f9(
        self,
        on_f8: Optional[Callable[[], None]] = None,
        on_f9: Optional[Callable[[], None]] = None,
        on_f10: Optional[Callable[[], None]] = None,
        no_repeat: bool = True,
        node_chord: Optional[str] = None,
        menu_chord: Optional[str] = None,
        pause_chord: Optional[str] = None,
        classic_f9: bool = True,
    ) -> Dict[str, bool]:
        """
        Register Ctrl+Alt+F8/F9/F10 (or the given chords) with MOD_NOREPEAT.
        Also registers classic bare F9 → same on_f9 when classic_f9=True so the
        menu opens while the game has focus (RegisterHotKey steals F9 from the game).
        Bare F8/F10 stay with the game.
        """
        from config.schema import DEFAULT_MENU_CHORD, DEFAULT_NODE_CHORD, DEFAULT_PAUSE_CHORD

        chords = (
            ("F8", node_chord or DEFAULT_NODE_CHORD, self.HOTKEY_ID_F8, on_f8),
            ("F9", menu_chord or DEFAULT_MENU_CHORD, self.HOTKEY_ID_F9, on_f9),
            ("F10", pause_chord or DEFAULT_PAUSE_CHORD, self.HOTKEY_ID_F10, on_f10),
        )
        results: Dict[str, bool] = {}
        for name, chord, hotkey_id, callback in chords:
            if callback is None:
                # F-0487: callback yoksa da anahtar sözlükte olur (False) —
                # çağıran results['F10'] deyince KeyError almaz.
                results[name] = False
                continue
            vk_code, modifiers = self.chord_binding(chord, no_repeat=no_repeat)
            if not vk_code:
                logger.warning("Kısayol akoru kaydedilemedi (yalın veya geçersiz): %s", chord)
                results[name] = False
                continue
            with self._lock:
                self._last_register_chords[hotkey_id] = str(chord)
            results[name] = self.register_native_hotkey(hotkey_id, vk_code, modifiers, callback)

        # Klasik F9: Ctrl+Alt+F9 yanında yalın F9 de menüyü açar (oyun odağında
        # da çalışır). F8/F10 yalın kalır — oyuna gider.
        results["F9_CLASSIC"] = False
        if classic_f9 and on_f9 is not None:
            vk_bare, mods_bare = self.chord_binding(
                "f9", no_repeat=no_repeat, allow_bare=True
            )
            if vk_bare:
                with self._lock:
                    self._last_register_chords[self.HOTKEY_ID_F9_CLASSIC] = "f9"
                results["F9_CLASSIC"] = self.register_native_hotkey(
                    self.HOTKEY_ID_F9_CLASSIC, vk_bare, mods_bare, on_f9
                )
                if results["F9_CLASSIC"]:
                    logger.info(
                        "Klasik F9 (yalın) RegisterHotKey ile menüye bağlandı "
                        "(Ctrl+Alt+F9 akoru korunur)."
                    )
                else:
                    logger.warning(
                        "Klasik F9 RegisterHotKey kaydı başarısız; menü için "
                        "klavye kancası / Ctrl+Alt+F9 yedek kalır."
                    )
        return results

    def unregister_native_hotkey(self, hotkey_id: int) -> bool:
        """
        Kayıtlı bir kısayolu UnregisterHotKey ile kaldırır.
        """
        hotkey_id = int(hotkey_id)
        with self._lock:
            if hotkey_id not in self._hotkeys:
                return False
            self._command_queue.append(("unregister", hotkey_id, 0, 0, None))

        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            with self._lock:
                if hotkey_id not in self._hotkeys:
                    return True
            time.sleep(0.005)
        return False

    def is_running(self) -> bool:
        return self._running.is_set() and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """
        Arka plan Win32 mesaj döngüsü (PeekMessageW) thread'ini başlatır.

        P2-4 (K-17): stop() join zaman aşımında hayalet thread bırakmışsa
        ikinci bir döngü ASLA başlatılmaz — iki döngü aynı hotkey
        id'lerini iki thread'den kaydeder (çift callback) ve eski
        döngünün finally bloğu yeni döngünün OS kayıtlarını siler.
        Hayalet thread'e önce kilit DIŞINDA kısa bir çıkış şansı verilir
        (join kilit altında tutulursa döngü `_invoke_callback_safe` /
        `_process_pending_commands` için kilidi bekleyip kilitlenir);
        hâlâ yaşıyorsa False döner — çağıran dürüstçe başarısız olur.
        """
        with self._lock:
            if self.is_running():
                return True
            stale = self._thread
        if stale is not None and stale.is_alive():
            stale.join(timeout=1.5)
            if stale.is_alive():
                logger.error(
                    "NativeWin32HotkeyManager başlatılamadı: önceki mesaj "
                    "döngüsü stop() zaman aşımından sonra hâlâ yaşıyor "
                    "(hayalet thread); çift döngü kurulmadı."
                )
                return False
            with self._lock:
                if self._thread is stale:
                    self._thread = None
                    self._thread_id = None
        with self._lock:
            if self.is_running():
                return True
            if self._thread is not None and self._thread.is_alive():
                logger.error(
                    "NativeWin32HotkeyManager başlatılamadı: mesaj döngüsü "
                    "hâlâ yaşıyor; çift döngü kurulmadı."
                )
                return False
            self._running.set()
            self._thread_ready.clear()
            self._thread = threading.Thread(
                target=self._message_loop_worker,
                name="NativeWin32HotkeyThread",
                daemon=True,
            )
            self._thread.start()

        self._thread_ready.wait(timeout=1.5)
        return self.is_running()

    def stop(self) -> None:
        """
        Tüm aktif RegisterHotKey kayıtlarını çözer ve arka plan thread'ini durdurur.

        P2-4 (K-17): join zaman aşımına uğrarsa thread referansı
        DÜŞÜRÜLMEZ — eskiden `self._thread = None` yapılır, döngü
        hayalet olarak yaşamaya devam eder (OS kayıtları + callback
        dağıtımı sürer) ve sonraki start() ikinci bir döngü kurardı.
        Referans korunur: tekrar stop() çıkışı yeniden dener, start()
        hayalet ölene kadar ikinci döngü kurmaz. Python thread'i
        güvenli biçimde öldürülemez; döngü `_running` bayrağını her
        turda okuduğu için takılı callback dönünce kendiliğinden biter
        ve finally bloğu OS kayıtlarını temizler.
        """
        self._running.clear()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        with self._lock:
            if self._thread is thread:
                if thread is None or not thread.is_alive():
                    self._thread = None
                    self._thread_id = None
                else:
                    logger.warning(
                        "NativeWin32HotkeyManager stop(): mesaj döngüsü "
                        "1.5 sn'de bitmedi — hayalet thread referansı "
                        "korunuyor, ikinci döngü kurulmayacak."
                    )

    def simulate_hotkey_event(self, hotkey_id: int) -> bool:
        """
        Test ve doğrulama amacıyla mesaj kuyruğuna WM_HOTKEY olayı gönderir
        (veya doğrudan thread üzerinden tetikler).

        F-0488: Windows'ta PostThreadMessage başarısız olursa başarısızlık
        maskelenmez — callback doğrudan çalıştırılmaz, False döner. Doğrudan
        callback yalnızca Windows-dışı platformda (OS mesaj kuyruğu yokken)
        kullanılır.
        """
        hotkey_id = int(hotkey_id)
        with self._lock:
            if hotkey_id not in self._hotkeys:
                return False
            tid = self._thread_id

        if self._is_windows:
            if self._user32 is not None and tid is not None:
                posted = bool(
                    self._user32.PostThreadMessageW(
                        wintypes.DWORD(tid),
                        wintypes.UINT(self.WM_HOTKEY),
                        wintypes.WPARAM(hotkey_id),
                        wintypes.LPARAM(0),
                    )
                )
                if posted:
                    return True
                logger.warning(
                    "PostThreadMessageW başarısız (hotkey_id=%d); callback çalıştırılmadı.",
                    hotkey_id,
                )
            return False

        # Windows-dışı: OS mesaj kuyruğu yok, doğrudan callback çalıştır.
        self._invoke_callback_safe(hotkey_id)
        return True

    def _invoke_callback_safe(self, hotkey_id: int) -> None:
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            entry = self._hotkeys.get(hotkey_id)
            if entry is not None:
                callback = entry[2]
                self._dispatched_count += 1

        if callback is not None:
            try:
                callback()
            except Exception as exc:
                logger.error("Hotkey (id=%d) callback çalışma hatası: %s", hotkey_id, exc)

    def _process_pending_commands(self, os_registered_ids: set) -> None:
        with self._lock:
            if not self._command_queue:
                return
            cmds = list(self._command_queue)
            self._command_queue.clear()

        for action, hk_id, vk_code, modifiers, callback in cmds:
            if action == "register" and callback is not None:
                success = False
                if self._is_windows and self._user32 is not None:
                    # Önce varsa eski kaydı temizle
                    if hk_id in os_registered_ids:
                        self._user32.UnregisterHotKey(None, hk_id)
                        os_registered_ids.discard(hk_id)

                    res = self._user32.RegisterHotKey(None, hk_id, modifiers, vk_code)
                    if res != 0:
                        success = True
                        os_registered_ids.add(hk_id)
                        with self._lock:
                            self._last_register_errors.pop(hk_id, None)
                        logger.info(
                            "Native RegisterHotKey başarılı: id=%d, vk=0x%02X, mod=0x%04X",
                            hk_id,
                            vk_code,
                            modifiers,
                        )
                    else:
                        err_code = ctypes.get_last_error()
                        # EK (dilim-26 F2): hata kodu saklanır — bool dönüş
                        # bunu taşımaz; çağıran get_last_register_error ile okur.
                        with self._lock:
                            self._last_register_errors[hk_id] = int(err_code)
                        # EK: eski mesaj yanlıştı — kaydedilen akor yalın F
                        # tuşu DEĞİL Ctrl+Alt+F8/F9/F10'dur (yalın akor
                        # chord_binding'de (0, 0) dönüp buraya hiç gelmez).
                        # err=1409: akor başka bir programda zaten kayıtlı.
                        chord = self._last_register_chords.get(hk_id) or self._format_chord(vk_code, modifiers)
                        if int(err_code) == self.ERROR_HOTKEY_ALREADY_REGISTERED:
                            reason = (
                                f"'{chord}' akoru başka bir program tarafından zaten "
                                "kayıtlı (err=1409, ERROR_HOTKEY_ALREADY_REGISTERED); "
                                "bu akor kısayol olarak kullanılamıyor."
                            )
                        else:
                            reason = (
                                f"'{chord}' akoru kaydedilemedi "
                                f"(RegisterHotKey err={err_code})."
                            )
                        logger.warning(
                            "RegisterHotKey(id=%d, vk=0x%02X, mod=0x%04X) reddedildi: %s",
                            hk_id,
                            vk_code,
                            modifiers,
                            reason,
                        )
                        success = False
                else:
                    # F-0347: Windows-dışı platformda OS'a hiç kayıt yapılmadı;
                    # success=False dürüst değerdir (register_native_hotkey
                    # de False döner; main.py hook dinleyicisine düşer).
                    logger.debug(
                        "RegisterHotKey yalnızca Windows'ta desteklenir; "
                        "id=%d bu platformda OS'a kaydedilmedi.",
                        hk_id,
                    )
                    success = False

                with self._lock:
                    if hk_id in self._hotkeys:
                        self._hotkeys[hk_id] = (vk_code, modifiers, callback, success)

            elif action == "unregister":
                if self._is_windows and self._user32 is not None and hk_id in os_registered_ids:
                    self._user32.UnregisterHotKey(None, hk_id)
                    os_registered_ids.discard(hk_id)
                with self._lock:
                    self._hotkeys.pop(hk_id, None)
                logger.info("Native UnregisterHotKey tamamlandı: id=%d", hk_id)

    def _message_loop_worker(self) -> None:
        """
        RegisterHotKey çağrılarının bağlı olduğu thread'de PeekMessageW ile
        WM_HOTKEY (0x0312) mesajlarını dinler.
        """
        os_registered_ids: set = set()
        if self._is_windows and self._kernel32 is not None:
            self._thread_id = int(self._kernel32.GetCurrentThreadId())
            msg = wintypes.MSG()
            # Thread mesaj kuyruğunu ilklendirmek için başlangıç PeekMessageW çağrısı
            self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        else:
            self._thread_id = threading.get_ident()
            msg = None

        self._thread_ready.set()

        try:
            while self._running.is_set():
                self._process_pending_commands(os_registered_ids)

                if self._is_windows and self._user32 is not None and msg is not None:
                    while (
                        self._user32.PeekMessageW(
                            ctypes.byref(msg),
                            None,
                            0,
                            0,
                            self.PM_REMOVE,
                        )
                        != 0
                    ):
                        if msg.message == self.WM_QUIT:
                            self._running.clear()
                            break
                        if msg.message == self.WM_HOTKEY:
                            hotkey_id = int(msg.wParam)
                            self._invoke_callback_safe(hotkey_id)
                        else:
                            self._user32.TranslateMessage(ctypes.byref(msg))
                            self._user32.DispatchMessageW(ctypes.byref(msg))

                time.sleep(self.poll_interval_sec)
        except Exception as exc:
            logger.error("NativeWin32HotkeyManager mesaj döngüsü hatası: %s", exc)
        finally:
            if self._is_windows and self._user32 is not None:
                for hk_id in list(os_registered_ids):
                    try:
                        self._user32.UnregisterHotKey(None, hk_id)
                    except Exception as exc:
                        logger.debug("UnregisterHotKey temizleme hatası (id=%d): %s", hk_id, exc)
                os_registered_ids.clear()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            registered = {
                hk_id: {
                    "vk_code": f"0x{vk:02X}",
                    "modifiers": f"0x{mod:04X}",
                    "active": active,
                }
                for hk_id, (vk, mod, _, active) in self._hotkeys.items()
            }
            return {
                "running": self.is_running(),
                "thread_id": self._thread_id,
                "dispatched_count": self._dispatched_count,
                "hotkeys": registered,
            }


# ============================================================================
# 3. PanicAlertSystem (15dk Boşta Kalma / 2dk Savaş / GM Kontrolü Alarmı)
# ============================================================================

class PanicAlertSystem:
    """
    15 dakika boyunca hiçbir şey toplayamama, 2 dakikadan uzun süren savaş,
    karakter ölümü veya "Prompt stuck for 10s" durumlarında:
      1. Anında tüm sanal girdi gönderimini kilitler (emergency_halt_inputs).
      2. Çok tonlu sesli alarm çalar (play_panic_siren).
      3. Telegram Bot API veya Discord Webhook üzerinden acil durum bildirimi gönderir.
    """

    DEFAULT_IDLE_LIMIT_SEC: float = 15.0 * 60.0      # 15 dakika
    DEFAULT_COMBAT_LIMIT_SEC: float = 2.0 * 60.0     # 2 dakika
    DEFAULT_PROMPT_STUCK_SEC: float = 10.0           # 10 saniye
    MAX_PANIC_HISTORY: int = 100  # F-0041: history sınırsız büyümez.

    def __init__(
        self,
        idle_limit_sec: float = DEFAULT_IDLE_LIMIT_SEC,
        combat_limit_sec: float = DEFAULT_COMBAT_LIMIT_SEC,
        prompt_stuck_sec: float = DEFAULT_PROMPT_STUCK_SEC,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook: str = "",
        enable_sound: bool = True,
    ) -> None:
        self.idle_limit_sec = float(idle_limit_sec)
        self.combat_limit_sec = float(combat_limit_sec)
        self.prompt_stuck_sec = float(prompt_stuck_sec)
        self.telegram_token = telegram_token.strip()
        self.telegram_chat_id = telegram_chat_id.strip()
        self.discord_webhook = discord_webhook.strip()
        self.enable_sound = bool(enable_sound)

        self._lock = threading.RLock()
        now = time.monotonic()
        self._last_gather_ts: float = now
        self._combat_start_ts: Optional[float] = None
        self._prompt_appear_ts: Optional[float] = None
        # Başlat/Devam sonrası PROMPT_STUCK yanlış pozitif: unlock sonrası
        # grace süresi boyunca istem paniği yeniden kurulmaz; gerçek stuck
        # grace bitince yine tetiklenir.
        self._prompt_stuck_grace_until: float = 0.0
        self._prompt_stuck_confirm_streak: int = 0
        self.PROMPT_STUCK_GRACE_SEC: float = 15.0
        self.PROMPT_STUCK_CONFIRM_POLLS: int = 2
        # Soft pause callback (worker.pause) — F-0049: prompt stuck panik
        # kilidi değil, rutin duraklatmadır.
        self._prompt_stuck_pause_callback: Optional[Callable[[str], None]] = None
        self._inputs_halted: bool = False
        self._panic_triggered: bool = False
        self._panic_history: List[Dict[str, Any]] = []
        self._last_panic_event: Optional[Dict[str, Any]] = None
        # F-0041: kilitlenirken üzerine yazılan orijinal girdi metotları/
        # öznitelikleri — reset_panic() bunları geri yükler.
        self._halted_input_state: Optional[Dict[str, Any]] = None
        # fix/panic-lock: kilidi tetikleyen sebep + detay — _blocked_input
        # uyarısında ve get_panic_status()/diagnostic snapshot'ta görünür;
        # trigger_panic ayrıca CRITICAL log satırına yazar (logs/bot.log).
        self._halt_reason: str = ""
        self._halt_details: str = ""
        # F-0351: async siren sonucu (thread bitince dolar).
        self._siren_done = threading.Event()
        self._siren_played: Optional[bool] = None
        self._siren_thread: Optional[threading.Thread] = None
        # F-0163: watchdog poll thread'i (start_watchdog_polling ile açılır).
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        # F-0163: poll döngüsünün dış sinyalleri okuduğu detektörler.
        # Kaydedilmezse ilgili koşul poll'dan gözlenemez — yalnızca doğrudan
        # check_watchdog_conditions(...) çağrısıyla verilebilir. Üretimde
        # bağlanmaları için bind_production_watchdog_sources() kullanılır
        # (main.py'den tek satır; bu dosya üretim modüllerini import etmez).
        self._character_dead_detector: Optional[Callable[[], bool]] = None
        self._gm_whisper_detector: Optional[Callable[[], bool]] = None
        self._prompt_stuck_detector: Optional[Callable[[], Optional[str]]] = None
        self._gather_heartbeat_detector: Optional[Callable[[], bool]] = None
        self._combat_state_detector: Optional[Callable[[], bool]] = None
        self._prompt_visible_detector: Optional[Callable[[], bool]] = None
        # fix/panik-kilit-oyun-yok: oyun penceresi varlığı detektörü —
        # True = açık, False = kapalı, None = bilinmiyor. Yokken (False)
        # idle sayacı durur; bilinmiyorsa eski davranış korunur.
        self._game_window_present_detector: Optional[Callable[[], Optional[bool]]] = None
        # fix/v20-rss-panik: oyun penceresi simge-durumu detektörü —
        # True = simge durumunda (toplanamaz/savaşılamaz), False/None =
        # değil/bilinmiyor. Simge durumundayken idle + savaş sayacı
        # donar (oyun-yokluğu ile aynı kural).
        self._game_window_minimized_detector: Optional[Callable[[], Optional[bool]]] = None
        # fix/v20-egitim-idle: eğitim-aktif detektörü — True = süreç
        # içinde RTX eğitimi sürüyor (bot eğitim için duraklatılmış,
        # toplanamaz). Eğitim sürerken YALNIZCA idle sayacı donar;
        # savaş/istem/ölüm/GM yollarına dokunulmaz (Selçuk kararı:
        # her duraklatmada değil, yalnızca eğitim sırasında).
        self._training_active_detector: Optional[Callable[[], Optional[bool]]] = None
        # fix/panik-kilit-oyun-yok: idle sayacının oyun-yokluğu yüzünden
        # duraklatıldığının son durumu — geçişte tek debug satırı için.
        self._idle_paused_for_game_window: bool = False
        # fix/v20-rss-panik: idle sayacının simge-durumu yüzünden
        # duraklatıldığının son durumu (aynı geçiş logu deseni).
        self._idle_paused_for_minimized: bool = False
        # fix/v20-egitim-idle: idle sayacının eğitim yüzünden
        # duraklatıldığının son durumu (aynı geçiş logu deseni).
        self._idle_paused_for_training: bool = False
        # fix/v20-panic-combat: bot duraklatıldı mı (BotWorker.pause() /
        # resume() set_paused ile bildirir). Duraklatılıyken savaş sayacı
        # donar — poll bayat in_combat okusa bile COMBAT_TIMEOUT_2M
        # tetiklenmez (canlı v19: bot DURAKLATILMIŞKEN 127.7 sn panik).
        self._paused: bool = False
        # fix/v20-panic-combat: savaş sayacının dondurulma nedenlerinin
        # son durumu — geçişte tek debug satırı için (idle deseni).
        self._combat_frozen_for_pause: bool = False
        self._combat_frozen_for_game_window: bool = False
        # fix/v20-rss-panik: savaş sayacının simge-durumu yüzünden
        # dondurulduğunun son durumu (aynı geçiş logu deseni).
        self._combat_frozen_for_minimized: bool = False
        # fix/v20-panic-combat: poll'un savaş detektörünü ardışık kaç
        # turdur True gördüğü — tek poll'lük True (tek karelik
        # vignette/HP-okuma parlaması) sayacı kuramaz; bkz.
        # _combat_poll_confirmed().
        self._combat_poll_streak: int = 0
        # sweep-21 B-11: detektör arıza takibi — bozulan/timeout'a uğrayan
        # detektör "temiz" sayılmaz; ardışık hata sayacı wiring
        # status'te görünür (faulted_detectors).
        self._detector_failures: Dict[str, int] = {}
        # sweep-21 B-05: kilit başarısız kaldıysa yeniden deneme zamanı
        # (monotonik saat; 0.0 = bekleyen deneme yok).
        self._halt_retry_due_mono: float = 0.0
        # sweep-21 B-06/B-13: kilitliyken bastırılan sebepler — panik
        # epizodu başına her sebep history'ye bir kez yazılır.
        self._suppressed_reasons_logged: set = set()
        self._last_panic_mono: Optional[float] = None

    @property
    def panic_triggered(self) -> bool:
        """Panik kilidi şu an aktif mi (F-0041: _panic_triggered artık okunur)."""
        with self._lock:
            return self._panic_triggered

    @property
    def inputs_halted(self) -> bool:
        with self._lock:
            return self._inputs_halted

    def get_panic_history(self) -> List[Dict[str, Any]]:
        """Panik olay geçmişinin kopyası (en fazla MAX_PANIC_HISTORY kayıt)."""
        with self._lock:
            return list(self._panic_history)

    def _append_panic_history(self, record: Dict[str, Any]) -> None:
        """Panik kaydını history'ye ekler (MAX_PANIC_HISTORY ile sınırlı)."""
        with self._lock:
            self._panic_history.append(record)
            del self._panic_history[: max(0, len(self._panic_history) - self.MAX_PANIC_HISTORY)]

    # sweep-21 B-07/B-08: yerel alarm dosyası dizini (göreli; bot repo
    # kökünden çalışır; main.py'deki LOG_DIR ile aynı kök).
    ALARM_DIR = os.path.join("logs", "alarms")

    def _write_local_alarm_file(
        self, reason: str, details: str, remote_notified: bool, extra: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Panik olayını kalıcı bir alarm dosyasına yazar (best-effort).

        B-07'nin düzelttiği yalan: eskiden "yerel alarm kaydı oluşturuldu"
        denmesine rağmen hiçbir dosya yazılmıyordu. Artık her panik
        logs/alarms/panic-<zaman>-<sebep>.json dosyasına yazılır; yazım
        başarısız olursa None döner ve asla exception fırlatmaz.
        """
        try:
            os.makedirs(self.ALARM_DIR, exist_ok=True)
            safe_reason = "".join(
                ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(reason or "UNKNOWN")
            )[:40] or "UNKNOWN"
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            millis = int((time.time() % 1.0) * 1000.0)
            path = os.path.join(
                self.ALARM_DIR, f"panic-{stamp}-{millis:03d}-{safe_reason}.json"
            )
            payload = {
                "timestamp": time.time(),
                "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "reason": str(reason or ""),
                "details": str(details or ""),
                "remote_notified": bool(remote_notified),
                "host": platform.node() or "",
            }
            if extra:
                payload.update(extra)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return path
        except Exception as exc:
            logger.warning("Yerel alarm dosyası yazılamadı: %s", exc)
            return None

    @staticmethod
    def load_panic_history_from_disk(limit: int = 100) -> List[Dict[str, Any]]:
        """logs/alarms/ altındaki kalıcı panik kayıtlarını okur.

        B-08: history yalnızca RAM'de duruyordu; süreç kapanınca kayboluyordu.
        Bu okuyucu diskteki alarm dosyalarını zaman sırasıyla döner.
        """
        records: List[Dict[str, Any]] = []
        try:
            alarm_dir = os.path.join("logs", "alarms")
            if not os.path.isdir(alarm_dir):
                return []
            names = [
                n
                for n in os.listdir(alarm_dir)
                if n.startswith("panic-") and n.endswith(".json")
            ]
            paths = sorted(
                (os.path.join(alarm_dir, n) for n in names),
                key=lambda p: os.path.getmtime(p),
            )
            for path in paths[-max(1, int(limit)) :]:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        records.append(data)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Kalıcı panik geçmişi okunamadı: %s", exc)
        return records

    def get_last_panic_event(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_panic_event) if self._last_panic_event else None

    def get_panic_status(self) -> Dict[str, Any]:
        """
        fix/panic-lock: panik kilidinin o anki durumu — HUD/log tüketicileri
        için tek okuma noktası. Tetikleyici sebebi + detay + tetiklenme
        zamanı + o zamandan beri geçen süre dahildir. Sebep/detay/zaman
        kilit boyunca sabittir; `elapsed_sec` CANLI hesaplanır (her
        okumada büyür) — o yüzden HUD dedup imzası elapsed_sec
        İÇERMEZ (ui/overlay_hud.py update_panic_status; P2-21/F031-01
        v16'da zaten böyle, test_panik_kilit_oyun_yok.py sabitler).
        reset_panic() ile sıfırlanır.
        """
        with self._lock:
            event = dict(self._last_panic_event) if self._last_panic_event else None
            panicked = self._panic_triggered
            halted = self._inputs_halted
            halt_reason = self._halt_reason
            halt_details = self._halt_details
        event_reason = (event or {}).get("reason") if event else None
        event_details = (event or {}).get("details") if event else None
        event_ts = (event or {}).get("timestamp") if event else None
        return {
            "panic_triggered": panicked,
            "inputs_halted": halted,
            "reason": event_reason or halt_reason or None,
            "details": event_details or halt_details or None,
            "timestamp": event_ts,
            "elapsed_sec": round(time.time() - float(event_ts), 1)
            if event_ts is not None
            else None,
            "history_count": len(self.get_panic_history()),
        }

    def record_successful_gather(self) -> None:
        """Başarılı bir kaynak toplama işleminde boşta kalma sayacını sıfırlar."""
        with self._lock:
            self._last_gather_ts = time.monotonic()

    def set_combat_state(self, in_combat: bool) -> None:
        """Savaş durumunu günceller."""
        with self._lock:
            if in_combat:
                if self._combat_start_ts is None:
                    self._combat_start_ts = time.monotonic()
            else:
                self._combat_start_ts = None

    # fix/v20-panic-combat: savaş detektörü poll başına en fazla bir
    # örnek verir (varsayılan 5 sn). Tek poll'lük True — örn. tek karelik
    # vignette parlaması / tek karelik HP okuma hatası / bayat in_combat
    # — sayacı kuramaz; True ardışık 2 poll'de de görülmeli.
    COMBAT_POLL_CONFIRM_POLLS: int = 2

    def _combat_poll_confirmed(self, in_combat: Optional[bool]) -> bool:
        """Poll'un okuduğu savaş sinyalini doğrulama süzgecinden geçirir.

        True ardışık COMBAT_POLL_CONFIRM_POLLS poll'de görülürse True
        döner (sayacı kurmaya değer); ilk True yalnızca streak'i
        hazırlar. False streak'i sıfırlar ve sayacı temizler. None
        (detektör yok/hatalı/belirsiz) streak'e ve sayaca dokunmaz —
        eski davranış korunur.
        """
        if in_combat is None:
            return False
        with self._lock:
            if not in_combat:
                self._combat_poll_streak = 0
                self._combat_start_ts = None
                return False
            self._combat_poll_streak = int(getattr(self, "_combat_poll_streak", 0) or 0) + 1
            return self._combat_poll_streak >= self.COMBAT_POLL_CONFIRM_POLLS

    def set_paused(self, paused: bool) -> None:
        """Bot duraklatıldı/devam etti bilgisini panik sistemine bildirir.

        fix/v20-panic-combat (canlı v19: bot DURAKLATILMIŞKEN
        COMBAT_TIMEOUT_2M 127.7 sn): duraklatma savaş sayacını
        sıfırlar; check_watchdog_conditions duraklatılıyken sayacı
        dondurur (poll bayat in_combat=True okusa bile yeniden
        kurulamaz). Panik KİLİDİNE dokunmaz — kilit yalnızca
        reset_panic() ile açılır; bu metot yalnızca zamanlayıcıyı
        yönetir. BotWorker.pause()/resume()/stop() çağırır.
        """
        paused = bool(paused)
        with self._lock:
            was_paused = self._paused
            self._paused = paused
            if paused:
                self._combat_start_ts = None
                self._combat_poll_streak = 0
        if paused:
            with self._lock:
                self._prompt_appear_ts = None
                self._prompt_stuck_confirm_streak = 0
        if paused and not was_paused:
            logger.debug(
                "[PANIC] Bot duraklatıldı — savaş ve istem sayaçları sıfırlandı/donduruldu."
            )
        elif not paused and was_paused:
            # Başlat/Devam: eski istem sayacıyla anında PROMPT_STUCK kilidi olmasın.
            self.arm_prompt_stuck_grace()
            logger.debug(
                "[PANIC] Bot devam ediyor — savaş/istem sayacı sıfırdan + grace."
            )

    def set_interact_prompt_visible(self, visible: bool) -> None:
        """Etkileşim ([F]) isteminin ekranda takılı kalma süresini takip eder."""
        with self._lock:
            if visible:
                if self._prompt_appear_ts is None:
                    self._prompt_appear_ts = time.monotonic()
            else:
                self._prompt_appear_ts = None
                self._prompt_stuck_confirm_streak = 0

    def arm_prompt_stuck_grace(self, sec: Optional[float] = None) -> None:
        """Başlat/Devam sonrası PROMPT_STUCK yeniden tetiklenmesini geciktirir."""
        grace = float(self.PROMPT_STUCK_GRACE_SEC if sec is None else sec)
        with self._lock:
            self._prompt_appear_ts = None
            self._prompt_stuck_confirm_streak = 0
            self._prompt_stuck_grace_until = time.monotonic() + max(0.0, grace)

    def set_prompt_stuck_pause_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        """PROMPT_STUCK soft yolunda worker.pause(reason) çağrısı bağlanır."""
        self._prompt_stuck_pause_callback = callback if callable(callback) else None

    def _prompt_stuck_in_grace(self, now: Optional[float] = None) -> bool:
        ts = time.monotonic() if now is None else float(now)
        with self._lock:
            return ts < float(self._prompt_stuck_grace_until or 0.0)

    def _release_inputs_soft(self, input_ctrl: Any) -> None:
        """Panik monkey-patch olmadan hareket/girdi kes (ESC / prompt stuck)."""
        if input_ctrl is None:
            return
        for name in ("abort_motion", "release_all_movement_keys", "release_all_held_keys"):
            fn = getattr(input_ctrl, name, None)
            if not callable(fn):
                continue
            try:
                if name == "release_all_held_keys":
                    fn(reason="prompt_stuck_soft")
                else:
                    fn()
            except Exception as exc:
                logger.debug("soft input release %s failed: %s", name, exc)

    def _handle_prompt_stuck_soft(
        self,
        details: str,
        input_ctrl: Any = None,
    ) -> Dict[str, Any]:
        """
        F-0049: takılı gather istemi güvenlik paniği değil — girdiyi kes,
        botu duraklat, grace kur. Başlat/Devam (resume → reset_panic yolu
        veya doğrudan grace) ile devam edilir; gerçek stuck grace sonrası
        yeniden uyarı verir.
        """
        self._release_inputs_soft(input_ctrl)
        reason_txt = str(details or "Prompt stuck for 10s")
        cb = self._prompt_stuck_pause_callback
        if callable(cb):
            try:
                cb(reason_txt)
            except Exception as exc:
                logger.warning("prompt_stuck pause callback failed: %s", exc)
        self.arm_prompt_stuck_grace()
        event = {
            "timestamp": time.time(),
            "reason": "PROMPT_STUCK_10S",
            "details": reason_txt,
            "inputs_halted": False,
            "soft_pause": True,
            "unlock_hint": "Başlat / Devam veya Ctrl+Alt+F10 ile devam",
            "siren_played": False,
            "siren_started": False,
            "remote_notified": False,
            "concurrent_reasons": [],
        }
        # Gözlemlenebilirlik: panik bayrağı kalkmadan son soft olay kaydı
        # (get_last_panic_event / testler). Hard kilit değildir.
        with self._lock:
            self._last_panic_event = dict(event)
        logger.warning(
            "[PANIC] PROMPT_STUCK soft-pause (panik kilidi yok): %s — %s",
            reason_txt,
            event["unlock_hint"],
        )
        return event

    def maybe_auto_recover_prompt_stuck(
        self, prompt_visible: Optional[bool], input_ctrl: Any = None
    ) -> bool:
        """PROMPT_STUCK panik kilidi varken istem kaybolduysa otomatik unlock."""
        with self._lock:
            panicked = self._panic_triggered
            event = dict(self._last_panic_event) if self._last_panic_event else None
        if not panicked or not event:
            return False
        if str(event.get("reason") or "") != "PROMPT_STUCK_10S":
            return False
        if prompt_visible is not False:
            return False
        ok = self.reset_panic(input_ctrl=input_ctrl)
        if ok:
            self.arm_prompt_stuck_grace()
            logger.info(
                "[PANIC] PROMPT_STUCK otomatik recovery: istem kayboldu, kilit açıldı."
            )
        return bool(ok)

    def emergency_halt_inputs(
        self, input_ctrl: Any = None, reason: str = "", details: str = ""
    ) -> bool:
        """
        Anında tüm sanal girdi gönderimini kilitler ve basılı kalmış olabilecek
        tuşları/fare butonlarını serbest bırakır.

        F-0349: Kısmi kilitlenme artık başarı sayılmaz — herhangi bir
        öznitelik/metot kilitlenemezse False döner (yine de denenebilen her
        kilit uygulanır). F-0041: üzerine yazılan orijinaller saklanır;
        reset_panic() ile geri yüklenebilir.
        fix/panic-lock: reason/details kilidin sebebidir — _blocked_input
        uyarısına ve get_panic_status()'a yazılır; eskiden engel uyarısında
        yalnızca "kilit aktif" vardı, hangi tetikleyici olduğu hiçbir
        satırda görünmüyordu.
        """
        # S-2202: denetleyici yoksa kilitlenecek girdi de yok — True dönmek
        # (ve _inputs_halted bayrağını kaldırmak) dürüst değil: gerçekte
        # hiçbir girdi engellenmez. Bu durumda False döner ve bayrak
        # kalkmaz, çünkü engellenen bir girdi yoktur.
        if input_ctrl is None:
            logger.error(
                "[PANIC] Acil durum girdi kilidi uygulanamadı: kilitlenecek "
                "denetleyici (input_ctrl) verilmedi — girdiler engellenmedi."
            )
            return False

        with self._lock:
            self._inputs_halted = True
            already_halted = self._halted_input_state is not None
            # Sebep yalnızca ilk kilitlemede yazılır — tekrar kilitleme
            # (zaten kilitliyken) orijinal sebebi ezmesin.
            if not already_halted:
                self._halt_reason = str(reason or "")
                self._halt_details = str(details or "")

        _halt_note = (
            f" (sebep: {self._halt_reason} — {self._halt_details})"
            if self._halt_reason
            else ""
        )
        logger.critical(
            "[PANIC] Acil durum girdi kilidi devreye alındı! Tüm sanal girdiler durduruldu.%s",
            _halt_note,
        )

        failures: List[str] = []
        # Yalnızca ilk kilitlemede orijinalleri sakla — tekrar kilitleme
        # _blocked_input'u "orijinal" diye kaydetmesin.
        if not already_halted:
            saved: Dict[str, Any] = {"input_ctrl": input_ctrl, "attrs": {}, "methods": {}}
        else:
            saved = None

        try:
            # Farklı InputController tasarımlarıyla tam uyumluluk
            for attr_name in ("emergency_stop", "halted", "locked", "is_halted", "disabled"):
                if hasattr(input_ctrl, attr_name):
                    if saved is not None:
                        saved["attrs"][attr_name] = getattr(input_ctrl, attr_name)
                    try:
                        setattr(input_ctrl, attr_name, True)
                    except Exception as exc:
                        failures.append(attr_name)
                        logger.warning("input_ctrl.%s ayarlanamadı: %s", attr_name, exc)

            # Açık olan sıfırlama/durdurma metotlarını çağır
            for method_name in ("release_all_inputs", "emergency_halt", "stop_all", "stop"):
                method = getattr(input_ctrl, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        failures.append(method_name)
                        logger.warning("input_ctrl.%s() çağrısı hata verdi: %s", method_name, exc)

            # Girdi gönderim metotlarını güvenli no-op (etkisiz) fonksiyonlarla kilitle
            # fix/panic-lock: engel uyarısı kilidin SEBEBİNİ taşır — eskiden
            # yalnızca "kilit aktif" yazıyordu, hangi tetikleyici olduğu
            # (ve ne kadar süredir) hiçbir log satırında yoktu.
            _block_note = (
                f" (sebep: {self._halt_reason} — {self._halt_details})"
                if self._halt_reason
                else ""
            )

            def _blocked_input(*_args: Any, **_kwargs: Any) -> bool:
                logger.warning(
                    "[PANIC-LOCK] Girdi engellendi: acil durum kilidi aktif.%s",
                    _block_note,
                )
                return False

            for send_method in (
                "press_key",
                "key_down",
                "key_up",
                "click",
                "right_click",
                "left_click",
                "move_mouse",
                "human_move_and_click",
                "send_input",
                "mount_up",
            ):
                if hasattr(input_ctrl, send_method):
                    if saved is not None:
                        saved["methods"][send_method] = getattr(input_ctrl, send_method)
                    try:
                        setattr(input_ctrl, send_method, _blocked_input)
                    except Exception as exc:
                        failures.append(send_method)
                        logger.warning("input_ctrl.%s kilitlenemedi: %s", send_method, exc)

            if saved is not None:
                with self._lock:
                    self._halted_input_state = saved

            if failures:
                logger.error("emergency_halt_inputs kısmi kilitlendi; başarısız: %s", failures)
                return False
            return True
        except Exception as exc:
            logger.error("emergency_halt_inputs sırasında hata: %s", exc)
            return False

    def reset_panic(self, input_ctrl: Any = None) -> bool:
        """
        F-0041: Panik kilidini geri alır — _panic_triggered/_inputs_halted
        temizlenir, kilitlenirken üzerine yazılan girdi metotları/öznitelikleri
        geri yüklenir, zamanlayıcılar sıfırlanır (koşul sürüyorsa watchdog
        hemen yeniden tetiklemez). Geri yükleme kısmen başarısız olursa
        False döner. History korunur.

        sweep-21 B-09: bayraklar geri yüklemeden ÖNCE temizlenmiyordu artık —
        geri yükleme ÖNCE denenir, bayraklar YALNIZCA başarılı geri yüklemede
        temizlenir. Başarısız geri yüklemede kilit aktif kalır ve
        _halted_input_state korunur ki sonraki resume() yeniden deneyebilsin;
        bot "açık" sanılırken girdi-ölü kalmaz.
        """
        ctrl = input_ctrl
        with self._lock:
            saved = self._halted_input_state
            if ctrl is None and saved is not None:
                ctrl = saved.get("input_ctrl")

        # S107 (v19/P3-8): verilen ctrl kilitlenen nesneyle aynı değilse
        # orijinaller yanlış nesneye yazılır, kilitli nesne _blocked_input
        # olarak kalırken bayraklar temizlenirdi (girdi-ölü bot).
        # emergency_halt_inputs input_ctrl=None iken False döner, o yüzden
        # kayıtlı kilit varsa saved["input_ctrl"] her zaman gerçek nesnedir.
        if saved is not None and ctrl is not None and ctrl is not saved.get("input_ctrl"):
            logger.error(
                "reset_panic: verilen input_ctrl kilitlenen nesneyle aynı değil — "
                "yanlış nesneye geri yükleme yapılmaz, kilit korunuyor."
            )
            return False

        if saved is None:
            # Geri yüklenecek kilitli girdi yok — bayrakları temizlemek güvenli.
            with self._lock:
                self._panic_triggered = False
                self._inputs_halted = False
                self._last_panic_event = None
                self._halted_input_state = None
                self._halt_reason = ""
                self._halt_details = ""
                now = time.monotonic()
                self._last_gather_ts = now
                self._combat_start_ts = None
                self._combat_poll_streak = 0
                self._prompt_appear_ts = None
                self._prompt_stuck_confirm_streak = 0
                self._suppressed_reasons_logged = set()
                self._halt_retry_due_mono = 0.0
            self.arm_prompt_stuck_grace()
            return True

        if ctrl is None:
            # Kilitli girdi var ama geri yüklenecek denetleyici yok — bayrak
            # temizlenirse metotlar _blocked_input olarak kalır. Kilit aktif
            # bırakılır; denetleyiciyle yeniden denenebilir.
            logger.error(
                "reset_panic: kilitli girdi var ama input_ctrl yok — kilit açılamadı, bayraklar korunuyor."
            )
            return False

        failures: List[str] = []
        for attr_name, original in saved.get("attrs", {}).items():
            try:
                setattr(ctrl, attr_name, original)
            except Exception as exc:
                failures.append(attr_name)
                logger.warning("reset_panic: input_ctrl.%s geri yüklenemedi: %s", attr_name, exc)
        for method_name, original in saved.get("methods", {}).items():
            try:
                setattr(ctrl, method_name, original)
            except Exception as exc:
                failures.append(method_name)
                logger.warning("reset_panic: input_ctrl.%s geri yüklenemedi: %s", method_name, exc)
        if failures:
            logger.error("reset_panic kısmi geri yükleme; başarısız: %s", failures)
            return False
        with self._lock:
            self._panic_triggered = False
            self._inputs_halted = False
            self._last_panic_event = None
            self._halted_input_state = None
            self._halt_reason = ""
            self._halt_details = ""
            now = time.monotonic()
            self._last_gather_ts = now
            self._combat_start_ts = None
            self._combat_poll_streak = 0
            self._prompt_appear_ts = None
            self._prompt_stuck_confirm_streak = 0
            self._suppressed_reasons_logged = set()
            self._halt_retry_due_mono = 0.0
        self.arm_prompt_stuck_grace()
        logger.info("[PANIC] Panik kilidi geri alındı; girdiler yeniden açıldı.")
        return True

    # Geriye dönük uyumlu takma ad.
    unlock_inputs = reset_panic
    resume_from_panic = reset_panic

    def play_panic_siren(
        self,
        pattern: Optional[List[Tuple[int, int]]] = None,
        async_play: bool = False,
    ) -> bool:
        """
        Çok tonlu sesli acil durum sireni çalar.
        Windows üzerinde winsound.Beep / kernel32.Beep kullanır.
        async_play=True ise dönüş değeri yalnızca thread'in başladığını söyler;
        sirenin gerçekten çalıp çalmadığı wait_for_siren()/get_siren_status()
        ile öğrenilir (F-0351).
        """
        if not self.enable_sound:
            return False

        tone_sequence: List[Tuple[int, int]] = pattern or [
            (880, 140),
            (1320, 140),
            (1760, 180),
            (1320, 140),
            (1980, 220),
        ]

        def _sound_worker() -> bool:
            played = False
            if os.name == "nt":
                try:
                    import winsound

                    for freq, dur_ms in tone_sequence:
                        clamped_freq = max(37, min(32767, int(freq)))
                        clamped_dur = max(20, min(5000, int(dur_ms)))
                        winsound.Beep(clamped_freq, clamped_dur)
                    played = True
                except Exception as exc:
                    logger.debug("winsound.Beep başarısız, kernel32.Beep deneniyor: %s", exc)
                    try:
                        kernel32 = ctypes.windll.kernel32
                        for freq, dur_ms in tone_sequence:
                            kernel32.Beep(int(freq), int(dur_ms))
                        played = True
                    except Exception as inner_exc:
                        logger.warning("Sesli alarm çalınamadı: %s", inner_exc)
            return played

        def _async_wrapper() -> None:
            # F-0351: async sonucunu sakla — thread başladı diye "çalındı" sayılmaz.
            try:
                result = _sound_worker()
            except Exception:
                result = False
            with self._lock:
                self._siren_played = result
            self._siren_done.set()

        if async_play:
            with self._lock:
                self._siren_done.clear()
                self._siren_played = None
            t = threading.Thread(target=_async_wrapper, name="PanicSirenThread", daemon=True)
            with self._lock:
                self._siren_thread = t
            t.start()
            # Dönüş değeri = thread başladı, siren henüz çalınmadı.
            return True

        return _sound_worker()

    def wait_for_siren(self, timeout_sec: float = 5.0) -> Optional[bool]:
        """F-0351: async sirenin gerçek sonucunu bekle (True/False, None=bilinmiyor)."""
        finished = self._siren_done.wait(timeout=timeout_sec)
        if not finished:
            return None
        with self._lock:
            return self._siren_played

    def get_siren_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "siren_done": self._siren_done.is_set(),
                "siren_played": self._siren_played,
                "siren_thread_alive": bool(
                    self._siren_thread is not None and self._siren_thread.is_alive()
                ),
            }

    @staticmethod
    def _telegram_response_ok(body: str) -> bool:
        """
        D23: Telegram sendMessage yanıtı başarılı mı? JSON'daki "ok" alanı
        esas alınır. Eski kontrol ('"ok":true' alt-dizesi) yanlış pozitif
        veriyordu: ok:false olan bir hata gövdesi, description içinde
        '"ok":true' geçiyorsa "gönderildi" sayılıyordu. JSON ayrıştırılamayan
        gövdede (proxy/düz metin) geriye dönük uyumluluk için eski alt-dize
        kontrolüne düşülür.
        """
        try:
            data = json.loads(body)
        except Exception:
            return '"ok":true' in body.replace(" ", "").lower()
        return isinstance(data, dict) and data.get("ok") is True

    def send_panic_notification(
        self,
        reason: str,
        details: str,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook: str = "",
        timeout_sec: float = 6.0,
    ) -> bool:
        """
        Kullanıcı tanımladıysa Telegram Bot API
        (https://api.telegram.org/bot<token>/sendMessage) veya Discord Webhook
        üzerinden telefona/kanala anında acil durum bildirimi gönderir.
        """
        tg_token = (telegram_token or self.telegram_token).strip()
        tg_chat = (telegram_chat_id or self.telegram_chat_id).strip()
        dc_hook = (discord_webhook or self.discord_webhook).strip()

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        host_tag = platform.node() or "UnknownHost"
        formatted_message = (
            f"🚨 [ACİL DURUM / PANİK ALARMI]\n"
            f"• Zaman: {timestamp_str}\n"
            f"• Makine: {host_tag}\n"
            f"• Neden: {reason}\n"
            f"• Detay: {details}\n"
            f"• Durum: Tüm sanal girdiler kilitlendi (Emergency Halt)."
        )

        any_sent = False

        # 1) Telegram Bot API Gönderimi
        if tg_token and tg_chat:
            tg_url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload_bytes = json.dumps(
                {
                    "chat_id": tg_chat,
                    "text": formatted_message,
                    "disable_web_page_preview": True,
                },
                ensure_ascii=False,
            ).encode("utf-8")

            req = urllib.request.Request(
                tg_url,
                data=payload_bytes,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "GatherionPanicMonitor/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    status_code = getattr(resp, "status", 200)
                    body = resp.read().decode("utf-8", errors="replace")
                    if 200 <= status_code < 300 and self._telegram_response_ok(body):
                        any_sent = True
                        logger.info("Telegram acil durum bildirimi başarıyla gönderildi.")
                    else:
                        logger.warning("Telegram API beklenmeyen yanıt döndürdü (HTTP %s): %s", status_code, body)
            except Exception as exc:
                logger.error("Telegram bildirimi gönderilemedi: %s", exc)

        # 2) Discord Webhook Gönderimi
        if dc_hook:
            dc_payload = json.dumps(
                {
                    "username": "OPSEC Panic Monitor",
                    "content": formatted_message,
                },
                ensure_ascii=False,
            ).encode("utf-8")

            req = urllib.request.Request(
                dc_hook,
                data=dc_payload,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "GatherionPanicMonitor/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    status_code = getattr(resp, "status", 204)
                    if 200 <= status_code < 300:
                        any_sent = True
                        logger.info("Discord Webhook acil durum bildirimi başarıyla gönderildi.")
                    else:
                        logger.warning("Discord Webhook beklenmeyen yanıt kodu: %s", status_code)
            except Exception as exc:
                logger.error("Discord Webhook bildirimi gönderilemedi: %s", exc)

        if not (tg_token and tg_chat) and not dc_hook:
            # sweep-21 B-07: eskiden burada "yerel alarm kaydı oluşturuldu"
            # denmesine rağmen hiçbir dosya yazılmıyordu. Gerçek yerel kayıt
            # trigger_panic() içinde logs/alarms/panic-*.json olarak yazılır;
            # bu dal yalnızca kanal yokluğunu dürüstçe söyler.
            logger.warning(
                "Harici bildirim kanalı (Telegram/Discord) tanımlanmamış; "
                "panik bildirimi gönderilemedi — kalıcı iz yalnızca "
                "logs/alarms/ alarm dosyası ve bot.log'daki [PANIC] satırıdır."
            )

        return any_sent

    def trigger_panic(
        self,
        reason: str,
        details: str,
        input_ctrl: Any = None,
        play_siren: bool = True,
        siren_async: bool = True,
        extra_reasons: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Tam acil durum prosedürünü yürütür:
        1. Sanal girdileri kilitler.
        2. Çok tonlu sesli alarm çalar.
        3. Telegram / Discord bildirimi gönderir.

        F-0041: Panik zaten aktifse yeniden tetiklenmez — son olay kaydı
        aynen döner, history şişmez, girdi kilidi yeniden kurulmaz.
        F-0351: async sirende siren_played=False, siren_started=True yazılır
        (siren henüz çalmadı; gerçek sonuç wait_for_siren() ile alınır).
        sweep-21 B-06: kilitliyken gelen YENİ sebep kaybolmaz — kalıcı alarm
        dosyasına bastırılmış kayıt olarak yazılır (epizot başına her sebep
        bir kez). RAM history'sine eklenmez: F-0041 sözleşmesi kilit
        epizodu başına tek history kaydı ister
        (test_no_retrigger_while_panicked_keeps_single_event).
        sweep-21 B-05: kilit başarısız olursa bayrak kalkmaz ama yeniden
        deneme için zaman kurulur (check_watchdog_conditions 30 sn'de bir
        emergency_halt_inputs'i yeniden dener).
        """
        # P2-4 (K-17): bayrak + olay kaydı TEK kritik bölgede birlikte
        # yayımlanır. Eskiden `_panic_triggered = True` kilit altında
        # kalkar, `_last_panic_event` ise halt/siren/bildirim bitene kadar
        # None kalırdı; o pencerede ikinci bir `trigger_panic` çağrısı
        # `... and _last_panic_event is not None` kontrolünü geçip ikinci
        # halt + ikinci siren + ikinci history kaydı üretirdi (F-0041
        # tek-olay sözleşmesi delinirdi). Artık ilk tetikleme, kilit
        # altında hemen bir yer tutucu kayıt yayımlar; eşzamanlı
        # çağıranlar o kaydı aynen alır, ikinci tetikleme yapılmaz.
        # Yer tutucu, halt/siren/bildirim bitince aynı nesne üzerinde
        # son değerlerle güncellenir (history'de tek kayıt kalır).
        suppressed_write: Optional[Tuple[str, str, str]] = None
        with self._lock:
            if self._panic_triggered:
                active_event = self._last_panic_event
                if active_event is not None:
                    active_reason = str(active_event.get("reason") or "")
                    if (
                        str(reason or "") != active_reason
                        and str(reason or "")
                        not in self._suppressed_reasons_logged
                    ):
                        self._suppressed_reasons_logged.add(str(reason or ""))
                        suppressed_write = (
                            str(reason or ""),
                            str(details or ""),
                            active_reason,
                        )
                    return_event = dict(active_event)
                else:
                    # Savunma dalı: bayrak ⇒ kayıt değişmezi ihlal
                    # edildiyse bile ikinci tetikleme yapılmaz.
                    logger.warning(
                        "[PANIC] Panik bayrağı aktif ama olay kaydı yok; "
                        "ikinci tetikleme bastırıldı (sebep=%s).",
                        reason,
                    )
                    return_event = {
                        "timestamp": time.time(),
                        "reason": str(self._halt_reason or reason or ""),
                        "details": str(details or ""),
                        "inputs_halted": bool(self._inputs_halted),
                        "inputs_halt_error": None,
                        "siren_played": False,
                        "siren_started": False,
                        "siren_async": bool(siren_async),
                        "remote_notified": False,
                        "concurrent_reasons": [],
                        "suppressed_while_panicked": True,
                    }
                event_record = None
            else:
                claim_ts = time.time()
                event_record = {
                    "timestamp": claim_ts,
                    "reason": reason,
                    "details": details,
                    # Yer tutucu: kilit/siren/bildirim henüz çalışmadı;
                    # inputs_halted=False + açıklama, S-2202'ye uygun
                    # dürüst "henüz belli değil" değeridir.
                    "inputs_halted": False,
                    "inputs_halt_error": "trigger in progress; final status pending",
                    "siren_played": False,
                    "siren_started": False,
                    "siren_async": bool(siren_async),
                    "remote_notified": False,
                    # sweep-21 B-13: aynı turda doğru olan diğer koşullar.
                    "concurrent_reasons": [
                        {"reason": r, "details": d}
                        for r, d in (extra_reasons or [])
                    ],
                }
                self._panic_triggered = True
                self._suppressed_reasons_logged = set()
                # sweep-21 B-05 yeniden deneme penceresi talep anından
                # başlar: yer tutucu inputs_halted=False taşıdığı için
                # watchdog'un panikli-dalı, tetikleme sürerken
                # emergency_halt_inputs'u ikinci kez çağırmasın.
                self._halt_retry_due_mono = time.monotonic() + 30.0
                self._append_panic_history(event_record)
                self._last_panic_event = dict(event_record)
                self._last_panic_mono = time.monotonic()
                return_event = None
        if suppressed_write is not None:
            # Dosya yazımı kilit DIŞINDA yapılır (IO kilit altında tutulmaz).
            self._write_local_alarm_file(
                suppressed_write[0],
                suppressed_write[1],
                False,
                extra={
                    "suppressed_while_panicked": True,
                    "active_reason": suppressed_write[2],
                },
            )
            logger.warning(
                "[PANIC] Kilitliyken yeni sebep bastırıldı (alarm dosyasına yazıldı): %s",
                suppressed_write[0],
            )
        if return_event is not None:
            return return_event
        assert event_record is not None  # kilit dalında kuruldu

        halted = self.emergency_halt_inputs(input_ctrl, reason=reason, details=details)
        if not halted:
            # sweep-21 B-05: kilit kurulamadı — bayrak kalkmaz (watchdog
            # denetlemeye devam eder) ama 30 sn sonra yeniden deneme
            # planlanır; aksi halde her poll'da CRITICAL spam'i olurdu.
            with self._lock:
                self._halt_retry_due_mono = time.monotonic() + 30.0
            logger.error(
                "[PANIC] Girdi kilidi kurulamadı (inputs_halted=False) — "
                "watchdog 30 sn sonra kilitlemeyi yeniden deneyecek."
            )
        siren_started = False
        siren_ok = False
        if play_siren:
            if siren_async:
                siren_started = self.play_panic_siren(async_play=True)
            else:
                siren_ok = self.play_panic_siren(async_play=False)
        notified = self.send_panic_notification(reason=reason, details=details)

        # fix/panic-lock: tetikleyici + süre yalnızca PC belleğindeki
        # _last_panic_event/_panic_history'de kalıyordu, dosyaya yazılmıyordu.
        # Bu CRITICAL satır main.py FileHandler'ı (logs/bot.log) üzerinden
        # dosyaya düşer — PC logundaki CRITICAL'lerin sebebi artık satırın
        # kendisinde yazar (20:08-20:14 arası 4 CRITICAL'in sebebi ancak
        # böyle geriye dönük okunabilir).
        logger.critical(
            "[PANIC] Acil durum tetiklendi: sebep=%s | detay=%s | girdiler_kilitli=%s",
            reason,
            details,
            halted,
        )

        # Yer tutucu kayıt (yukarıda history + _last_panic_event'e
        # yayımlandı) son değerlerle güncellenir — timestamp talep
        # anındaki kalır; eşzamanlı çağıranların aldığı kopyayla aynı
        # olaydır, ikinci kayıt oluşmaz.
        event_record.update(
            {
                "inputs_halted": halted,
                # S-2202: halt başarısızsa nedeni kayda yazılır — denetleyicisiz
                # çağrıda "inputs_halted": False artık "kilitlendi" sayılmaz.
                "inputs_halt_error": None
                if halted
                else (
                    "no input controller provided; inputs not blocked"
                    if input_ctrl is None
                    else "partial lock failure; see log"
                ),
                "siren_played": siren_ok,
                "siren_started": siren_started,
                "siren_async": bool(siren_async),
                "remote_notified": notified,
            }
        )
        # sweep-21 B-07/B-08: her panik kalıcı bir alarm dosyasına yazılır
        # (logs/alarms/panic-*.json) — RAM history'si süreç kapanınca
        # kayboluyordu; bu dosya load_panic_history_from_disk() ile okunur.
        alarm_file = self._write_local_alarm_file(reason, details, notified)
        if alarm_file is not None:
            event_record["alarm_file"] = alarm_file
        with self._lock:
            # Yer tutucu zaten history'de (aynı nesne) ve
            # _last_panic_event'te; yalnızca son değerlerle güncellenir,
            # ikinci kayıt eklenmez. reset_panic() arada çalıştıysa
            # _last_panic_event None'dır — yayımlanan kayıt diriltilmez
            # (history kaydı yine de son değerleri taşır; history
            # reset'te korunur).
            if (
                self._panic_triggered
                and self._last_panic_event is not None
                and self._last_panic_event.get("timestamp")
                == event_record.get("timestamp")
            ):
                self._last_panic_event.update(event_record)
            self._last_panic_mono = time.monotonic()

        return dict(event_record)

    def check_watchdog_conditions(
        self,
        input_ctrl: Any = None,
        character_dead: bool = False,
        prompt_stuck_msg: Optional[str] = None,
        gm_whisper_detected: bool = False,
        play_siren: bool = True,
        game_window_present: Optional[bool] = None,
        game_window_minimized: Optional[bool] = None,
        paused: Optional[bool] = None,
        training_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Periyodik olarak çağrılarak 4 ana acil durum koşulunu denetler:
        - 15 dakika boyunca hiçbir şey toplayamama
        - 2 dakikadan uzun süren savaş
        - Karakter ölümü
        - "Prompt stuck for 10s" veya GM kontrolü

        F-0041: Panik zaten aktifken koşul sürse bile yeniden tetiklenmez
        (None döner); kilit reset_panic() ile açıldıktan sonra yeniden
        denetlenir. F-0163: üretimde periyodik çağrı için
        start_watchdog_polling() kullanılır; main.py'daki bağlantı (D4)
        ayrıca yapılmalıdır.
        sweep-21 B-13: eşzamanlı doğru olan TÜM koşullar değerlendirilir —
        ilk sebep tetikler, diğerleri olay kaydının concurrent_reasons
        alanına girer (eskiden yalnızca ilk sebep görünürdü).
        sweep-21 B-06: kilitliyken gelen yeni sebep history'ye bastırılmış
        kayıt olarak yazılır (epizot başına her sebep bir kez).
        sweep-21 B-05: son kilitleme başarısız kaldıysa (inputs_halted
        False) kilit 30 sn aralıkla yeniden denenir.
        fix/panik-kilit-oyun-yok: game_window_present=False iken
        (oyun penceresi kapalı) idle sayacı sıfırlanır/durur — oyun
        kapalıyken "toplayamadı" diye IDLE_TIMEOUT_15M tetiklenmez.
        fix/v20-panic-combat: oyun yokken SAVAŞ sayacı da sıfırlanır/
        durur — oyun kapalıyken kare gelmez, combat_recovery.in_combat
        bayat True'da takılır ve yoksa COMBAT_TIMEOUT_2M haksız yere
        tetikler. Aynı donma bot duraklatılıyken de geçerlidir
        (paused=True veya set_paused(True)): duraklatılmış bot savaşmaz,
        savaş sayacı işlemez. Ölüm/GM/istem yolları aynen çalışır
        (oyun yokken de, duraklatılıyken de).
        fix/v20-rss-panik: oyun penceresi simge durumundayken de
        (game_window_minimized=True) idle VE savaş sayacı donar —
        simge durumundaki oyunun ekranı okunamaz, toplanamaz,
        savaşılamaz. Watchdog pencereyi "var" sayar
        (is_game_window_present True döner), o yüzden simge durumu
        ayrı sinyalle gelir; oyun-yokluğu ile aynı donma uygulanır.
        Ölüm/GM/istem yolları simge durumundayken de aynen çalışır.
        fix/v20-egitim-idle: eğitim sürerken de (training_active=True)
        YALNIZCA idle sayacı donar — canlı 22:09:42: bot eğitim için
        duraklatılmış, oyun görünürken IDLE_TIMEOUT_15M haksız yere
        attı. Savaş sayacı duraklatmada zaten donar; istem sayacı da
        duraklatmada donar; ölüm/GM aynen çalışır. Her duraklatma
        değil, yalnızca eğitim donmayı açar (Selçuk kararı).
        None (bilinmiyor / verilmedi) = eski davranış.
        """
        now = time.monotonic()

        with self._lock:
            panicked = self._panic_triggered
            last_event = dict(self._last_panic_event) if self._last_panic_event else None
            halt_retry_due = self._halt_retry_due_mono
            idle_elapsed = now - self._last_gather_ts
            combat_elapsed = (now - self._combat_start_ts) if self._combat_start_ts is not None else 0.0
            prompt_elapsed = (now - self._prompt_appear_ts) if self._prompt_appear_ts is not None else 0.0
            # fix/panik-kilit-oyun-yok: oyun yokken idle sayacı durur.
            # _last_gather_ts "şimdi"ye çekilirse idle_elapsed 0 olur ve
            # aşağıdaki idle koşulu tutmaz; oyun geri gelince sayaç
            # sıfırdan başlar. Diğer zamanlayıcılara dokunulmaz.
            # fix/v20-rss-panik: simge durumu da aynı donmayı uygular.
            # fix/v20-egitim-idle: eğitim de YALNIZCA idle donmasını
            # uygular (savaş/istem duraklatma kuralında zaten donar).
            window_absent = game_window_present is False
            window_minimized = game_window_minimized is True
            training = training_active is True
            idle_paused = window_absent or window_minimized or training
            if idle_paused:
                self._last_gather_ts = now
                idle_elapsed = 0.0
            was_idle_paused = (
                self._idle_paused_for_game_window or self._idle_paused_for_minimized
                or self._idle_paused_for_training
            )
            self._idle_paused_for_game_window = window_absent
            self._idle_paused_for_minimized = window_minimized
            self._idle_paused_for_training = training
            # fix/v20-panic-combat: savaş sayacı donar — oyun yokken
            # VEYA bot duraklatılıyken. _combat_start_ts temizlenir
            # (sıfırlanır); oyun geri gelince / devam edilince sayaç
            # sıfırdan başlar, eski birikmiş süreyle anında tetiklenmez.
            # Bu temizlik poll'un set_combat_state(True) çağrısından
            # SONRA çalışır (poll sırası: detektör -> set_combat_state
            # -> check_watchdog_conditions), o yüzden bayat True poll'u
            # sayacı yeniden kuramaz. paused argümanı verilmezse
            # set_paused() ile bildirilen durum kullanılır.
            paused_effective = self._paused if paused is None else bool(paused)
            combat_frozen = paused_effective or window_absent or window_minimized
            if combat_frozen:
                self._combat_start_ts = None
                self._combat_poll_streak = 0
                combat_elapsed = 0.0
            was_combat_frozen_pause = self._combat_frozen_for_pause
            was_combat_frozen_window = self._combat_frozen_for_game_window
            was_combat_frozen_minimized = self._combat_frozen_for_minimized
            self._combat_frozen_for_pause = paused_effective
            self._combat_frozen_for_game_window = window_absent
            self._combat_frozen_for_minimized = window_minimized
            # fix/v20-kalan-eksikler: duraklatılıyken istem sayacı donar
            # (canlı 19:26:11 — Ctrl+Alt+F10 sonrası PROMPT_STUCK yanlış pozitif).
            if paused_effective:
                self._prompt_appear_ts = None
                prompt_elapsed = 0.0

        if idle_paused and not was_idle_paused:
            logger.debug(
                "[PANIC] %s — boşta sayacı duraklatıldı "
                "(IDLE_TIMEOUT_15M tetiklenmez).",
                "Eğitim sürüyor"
                if training
                else (
                    "Oyun penceresi simge durumunda"
                    if window_minimized
                    else "Oyun penceresi yok"
                ),
            )
        elif not idle_paused and was_idle_paused:
            logger.debug("[PANIC] Boşta sayacı yeniden çalışıyor (sıfırdan).")
        _was_combat_frozen = (
            was_combat_frozen_pause or was_combat_frozen_window or was_combat_frozen_minimized
        )
        if combat_frozen and not _was_combat_frozen:
            logger.debug(
                "[PANIC] Savaş sayacı donduruldu (%s) — COMBAT_TIMEOUT_2M tetiklenmez.",
                "bot duraklatıldı"
                if paused_effective
                else ("oyun penceresi simge durumunda" if window_minimized else "oyun penceresi yok"),
            )
        elif not combat_frozen and _was_combat_frozen:
            logger.debug("[PANIC] Savaş sayacı yeniden çalışıyor (sıfırdan).")

        # Öncelik sırası korunur: ölüm → GM → istem → savaş → idle.
        hits: List[Tuple[str, str]] = []
        if character_dead:
            hits.append(("CHARACTER_DEAD", "Karakter ölüm ekranı algılandı!"))
        if gm_whisper_detected:
            hits.append(("GM_INSPECTION_ALERT", "Olası GM mesajı veya ışınlanma anomalisi algılandı!"))
        prompt_frozen = paused_effective or self._prompt_stuck_in_grace(now)
        prompt_hit_by_message = False
        if (
            prompt_stuck_msg
            and "Prompt stuck for 10s" in prompt_stuck_msg
            and not prompt_frozen
        ):
            with self._lock:
                self._prompt_stuck_confirm_streak += 1
                streak = self._prompt_stuck_confirm_streak
            if streak >= max(1, int(self.PROMPT_STUCK_CONFIRM_POLLS)):
                hits.append(("PROMPT_STUCK_10S", prompt_stuck_msg))
                prompt_hit_by_message = True
        elif not prompt_frozen and prompt_elapsed >= self.prompt_stuck_sec:
            with self._lock:
                self._prompt_stuck_confirm_streak += 1
                streak = self._prompt_stuck_confirm_streak
            if streak >= max(1, int(self.PROMPT_STUCK_CONFIRM_POLLS)):
                hits.append(
                    (
                        "PROMPT_STUCK_10S",
                        f"Prompt stuck for 10s ({prompt_elapsed:.1f}s boyunca etkileşim istemi kapanmadı).",
                    )
                )
        else:
            with self._lock:
                if not prompt_stuck_msg and prompt_elapsed < self.prompt_stuck_sec:
                    self._prompt_stuck_confirm_streak = 0
        if combat_elapsed >= self.combat_limit_sec:
            hits.append(
                (
                    "COMBAT_TIMEOUT_2M",
                    f"Savaş süresi 2 dakika sınırını aştı ({combat_elapsed:.1f}s).",
                )
            )
        if idle_elapsed >= self.idle_limit_sec:
            hits.append(
                (
                    "IDLE_TIMEOUT_15M",
                    f"Karakter {idle_elapsed / 60.0:.1f} dakikadır hiçbir kaynak toplayamadı.",
                )
            )

        if panicked:
            # sweep-21 B-05: kilit başarısız kaldıysa yeniden dene.
            if last_event is not None and not last_event.get("inputs_halted", True):
                if now >= halt_retry_due:
                    retry_ok = self.emergency_halt_inputs(
                        input_ctrl,
                        reason=str(last_event.get("reason") or ""),
                        details=str(last_event.get("details") or ""),
                    )
                    with self._lock:
                        self._halt_retry_due_mono = now + 30.0
                        if retry_ok:
                            self._inputs_halted = True
                            if self._last_panic_event is not None:
                                self._last_panic_event["inputs_halted"] = True
                                self._last_panic_event["inputs_halt_error"] = None
                            if self._panic_history:
                                self._panic_history[-1]["inputs_halted"] = True
                                self._panic_history[-1]["inputs_halt_error"] = None
                            logger.info("[PANIC] Girdi kilidi yeniden denemede kuruldu.")
                        else:
                            logger.warning(
                                "[PANIC] Girdi kilidi yeniden denemede de kurulamadı; 30 sn sonra tekrar denenecek."
                            )
            # sweep-21 B-06: kilitliyken gelen yeni sebepler kaybolmasın.
            # Kalıcı alarm dosyasına yazılır (RAM history'sine değil —
            # F-0041 kilit epizodu başına tek history kaydı ister).
            active_reason = str((last_event or {}).get("reason") or "")
            for hit_reason, hit_details in hits:
                if hit_reason == active_reason:
                    continue
                with self._lock:
                    already = hit_reason in self._suppressed_reasons_logged
                    if not already:
                        self._suppressed_reasons_logged.add(hit_reason)
                if not already:
                    self._write_local_alarm_file(
                        hit_reason,
                        hit_details,
                        False,
                        extra={
                            "suppressed_while_panicked": True,
                            "active_reason": active_reason,
                        },
                    )
                    logger.warning(
                        "[PANIC] Kilitliyken yeni sebep bastırıldı (alarm dosyasına yazıldı): %s",
                        hit_reason,
                    )
            return None

        if not hits:
            return None

        first_reason, first_details = hits[0]
        # F-0049 / smoke Başlat-Devam: PROMPT_STUCK panik kilidi değil —
        # soft-pause + grace. Diğer sebepler (ölüm/GM/savaş/idle) tam panik.
        if first_reason == "PROMPT_STUCK_10S":
            soft = self._handle_prompt_stuck_soft(
                details=first_details, input_ctrl=input_ctrl
            )
            # Eşzamanlı daha ağır sebepler varsa onlar panik kilidine gider.
            hard = [(r, d) for r, d in hits[1:] if r != "PROMPT_STUCK_10S"]
            if hard:
                return self.trigger_panic(
                    reason=hard[0][0],
                    details=hard[0][1],
                    input_ctrl=input_ctrl,
                    play_siren=play_siren,
                    extra_reasons=hard[1:] + [("PROMPT_STUCK_10S", first_details)],
                )
            return soft
        # sweep-21 B-13: aynı turda doğru olan diğer koşullar olay
        # kaydının concurrent_reasons alanına girer.
        return self.trigger_panic(
            reason=first_reason,
            details=first_details,
            input_ctrl=input_ctrl,
            play_siren=play_siren,
            extra_reasons=hits[1:],
        )

    def register_watchdog_detectors(
        self,
        character_dead_detector: Optional[Callable[[], bool]] = None,
        gm_whisper_detector: Optional[Callable[[], bool]] = None,
        prompt_stuck_detector: Optional[Callable[[], Optional[str]]] = None,
        gather_heartbeat_detector: Optional[Callable[[], bool]] = None,
        combat_state_detector: Optional[Callable[[], bool]] = None,
        prompt_visible_detector: Optional[Callable[[], bool]] = None,
        game_window_present_detector: Optional[Callable[[], Optional[bool]]] = None,
        game_window_minimized_detector: Optional[Callable[[], Optional[bool]]] = None,
        training_active_detector: Optional[Callable[[], Optional[bool]]] = None,
    ) -> None:
        """
        F-0163: _poll_loop'un her turda danışacağı dış sinyal detektörlerini
        kaydeder. Yalnızca verilen (None olmayan) detektörler değişir;
        öncekiler silinmez. Detektör sözleşmeleri:
        - character_dead_detector / gm_whisper_detector: bool döner.
        - prompt_stuck_detector: takılı istem mesajını (str) veya None döner.
        - gather_heartbeat_detector: son turdan beri başarılı gather olduysa
          True döner — poll bunu record_successful_gather() ile tüketir.
        - combat_state_detector: savaşta True, değilse False döner — poll
          bunu set_combat_state() ile tüketir.
        - prompt_visible_detector: etkileşim istemi görünürse True döner —
          poll bunu set_interact_prompt_visible() ile tüketir.
        - game_window_present_detector (fix/panik-kilit-oyun-yok,
          fix/v20-panic-combat): oyun penceresi açıksa True, kapalıysa
          False, belirlenemiyorsa None döner — poll bunu
          check_watchdog_conditions'a geçirir; False iken idle sayacı
          VE savaş sayacı durur, diğer tetikleyiciler aynen çalışır.
        - game_window_minimized_detector (fix/v20-rss-panik): oyun
          penceresi simge durumundaysa True döner — poll bunu
          check_watchdog_conditions'a geçirir; True iken idle sayacı
          VE savaş sayacı durur (oyun-yokluğu ile aynı kural).
        - training_active_detector (fix/v20-egitim-idle): süreç içinde
          RTX eğitimi sürüyorsa True döner — poll bunu
          check_watchdog_conditions'a geçirir; True iken YALNIZCA
          idle sayacı durur (savaş/istem/ölüm/GM aynen çalışır).
        """
        with self._lock:
            if character_dead_detector is not None:
                self._character_dead_detector = character_dead_detector
            if gm_whisper_detector is not None:
                self._gm_whisper_detector = gm_whisper_detector
            if prompt_stuck_detector is not None:
                self._prompt_stuck_detector = prompt_stuck_detector
            if gather_heartbeat_detector is not None:
                self._gather_heartbeat_detector = gather_heartbeat_detector
            if combat_state_detector is not None:
                self._combat_state_detector = combat_state_detector
            if prompt_visible_detector is not None:
                self._prompt_visible_detector = prompt_visible_detector
            if game_window_present_detector is not None:
                self._game_window_present_detector = game_window_present_detector
            if game_window_minimized_detector is not None:
                self._game_window_minimized_detector = game_window_minimized_detector
            if training_active_detector is not None:
                self._training_active_detector = training_active_detector

    def bind_production_watchdog_sources(
        self,
        worker: Any = None,
        combat_recovery: Any = None,
        gm_whisper_detector: Optional[Callable[[], bool]] = None,
        training_active_detector: Optional[Callable[[], Optional[bool]]] = None,
    ) -> Dict[str, Any]:
        """
        F-0163 (2. tur): üretimdeki canlı kaynakları (BotWorker /
        CombatRecovery) watchdog detektörlerine bağlar. main.py'deki
        start_watchdog_polling(input_ctrl=...) çağrısının yanına tek satır:
            _panic.bind_production_watchdog_sources(worker=self.worker)

        Ördek-tipi (duck-typed) bağlama: bu dosya üretim modüllerini import
        etmez; yalnızca verilen nesnelerin özniteliklerini okur:
        - character_dead: combat_recovery.is_dead — worker'ın
          check_character_dead(frame) çağrılarıyla güncellenen bayrak.
        - combat_state: combat_recovery.in_combat.
        - gather_heartbeat: worker.gathered_count farkı — sayı arttıysa
          son turdan beri gather olmuş demektir (record_successful_gather
          ile tüketilir). İlk çağrı yalnızca taban sayıyı alır.
        - prompt_visible: worker._detect_gather_prompt() — worker'ın kendi
          görsel probu (F/E etkileşim istemi); hata verirse detektör None
          döner, poll o turda istem durumunu güncellemez.
        - prompt_stuck: worker.behavioral_guard.check_prompt_stuck() —
          10 sn takılı istemi worker'ın kendi guard'ı izler; detektör
          "Prompt stuck for 10s" içeren mesajı döner (sweep-21 B-04
          düzeltmesi: bind'de prompt_stuck hiç yoktu).
        - gm_whisper: repo'da üretim kaynağı YOK (GM/fısıltı algılayan
          modül bulunamadı) — yalnızca açıkça verilen bir detektör
          bağlanır; verilmezse kayıtsız kalır ve wiring status'te eksik
          görünür. gm_whisper "zorunlu" sayılmaz (aşağıdaki
          missing_required_detectors hesabına girmez) çünkü yapısal olarak
          yok; kanala dair karar Selçuk'undur (AÇIK SORU).
        - game_window_present (fix/panik-kilit-oyun-yok): worker.watchdog
          (BotWatchdog).is_game_window_present — True/False/None döner
          (None = Windows dışı / hedef başlık yok, belirlenemiyor).
          "Zorunlu" sayılmaz (missing_required'a girmez): Linux/VM'de
          belirlenemediği için yoksa fully_wired asla True olamaz ve
          "degraded" kalıcı alarm yorgunluğuna dönerdi (gm_whisper ile
          aynı gerekçe).
        - game_window_minimized (fix/v20-rss-panik): worker.watchdog
          .is_game_window_minimized — simge durumundaysa True döner.
          "Zorunlu" sayılmaz (aynı gerekçe).
        - training_active (fix/v20-egitim-idle): süreç-içi eğitim
          slotu (dataset_factory.rtx_trainer_engine.is_training_active)
          — tembel import ile okunur, bu dosya üretim modüllerini
          üstte import etmez. Açıkça detektör verilirse o kullanılır;
          verilmezse slot okuyucusu bağlanır. Import/okuma hatası
          None döner (bilinmiyor = eski davranış). "Zorunlu"
          sayılmaz (aynı gerekçe).

        Döner: {"wired": {sinyal: bool, ...}, "missing": [sinyal, ...]}.
        Eksik sinyal varsa warning loglanır — sessiz eksik bağlama yok.
        """
        cr = combat_recovery
        if cr is None and worker is not None:
            cr = getattr(worker, "combat_recovery", None)

        wired: Dict[str, bool] = {
            "character_dead": False,
            "gm_whisper": False,
            "gather_heartbeat": False,
            "combat_state": False,
            "prompt_visible": False,
            "prompt_stuck": False,
            "game_window_present": False,
            "game_window_minimized": False,
            "training_active": False,
        }

        # fix/v20-egitim-idle: eğitim sinyali worker'dan bağımsızdır —
        # süreç-içi slot modül düzeyinde tutulur. Açık detektör
        # verilmezse tembel slot okuyucusu bağlanır (import hatası
        # None döner, poll o turda eski davranışla denetler).
        if training_active_detector is not None:
            self.register_watchdog_detectors(
                training_active_detector=training_active_detector
            )
            wired["training_active"] = True
        else:

            def _training_active() -> Optional[bool]:
                try:
                    import importlib

                    mod = importlib.import_module(
                        "dataset_factory.rtx_trainer_engine"
                    )
                    fn = getattr(mod, "is_training_active", None)
                    if not callable(fn):
                        return None
                    return bool(fn())
                except Exception:
                    return None

            self.register_watchdog_detectors(
                training_active_detector=_training_active
            )
            wired["training_active"] = True

        if cr is not None and hasattr(cr, "is_dead"):
            self.register_watchdog_detectors(
                character_dead_detector=lambda: bool(getattr(cr, "is_dead", False))
            )
            wired["character_dead"] = True
        if cr is not None and hasattr(cr, "in_combat"):
            self.register_watchdog_detectors(
                combat_state_detector=lambda: bool(getattr(cr, "in_combat", False))
            )
            wired["combat_state"] = True

        if worker is not None and hasattr(worker, "gathered_count"):
            _state = {"last_count": None}

            def _gather_heartbeat() -> bool:
                count = getattr(worker, "gathered_count", None)
                if not isinstance(count, int):
                    return False
                last = _state["last_count"]
                _state["last_count"] = count
                if last is None:
                    return False
                return count > last

            self.register_watchdog_detectors(gather_heartbeat_detector=_gather_heartbeat)
            wired["gather_heartbeat"] = True

        if worker is not None:
            _prompt_detector = getattr(worker, "_detect_gather_prompt", None)
            if callable(_prompt_detector):

                def _gathering_context() -> bool:
                    # Yürürken yanlış pozitif F-istemi panik üretmesin —
                    # yalnızca aktif toplama / seçili düğümde say.
                    if bool(getattr(worker, "_is_gathering_active", False)):
                        return True
                    if getattr(worker, "active_node", None) is not None:
                        return True
                    state = str(getattr(worker, "state", "") or "").upper()
                    return state in ("GATHERING", "HARVESTING", "INTERACTING")

                def _prompt_visible() -> Optional[bool]:
                    if not _gathering_context():
                        return False
                    result = _prompt_detector()
                    if isinstance(result, (tuple, list)) and result:
                        return bool(result[0])
                    if isinstance(result, bool):
                        return result
                    return None

                self.register_watchdog_detectors(prompt_visible_detector=_prompt_visible)
                wired["prompt_visible"] = True

            pause_fn = getattr(worker, "pause", None)
            if callable(pause_fn):
                self.set_prompt_stuck_pause_callback(
                    lambda reason: pause_fn(reason or "Prompt stuck for 10s")
                )

        # sweep-21 B-04: prompt_stuck bind'de hiç yoktu — worker'ın
        # behavioral_guard.check_prompt_stuck() üretim kaynağıdır. Poll
        # bu detektörü her turda çağırır; 10 sn takılı istem "Prompt stuck
        # for 10s" mesajıyla döner (check_watchdog_conditions'un aradığı
        # tam alt-dize budur). Soft-pause yoluna gider (F-0049).
        if worker is not None:
            _guard = getattr(worker, "behavioral_guard", None)
            _prompt_fn = getattr(worker, "_detect_gather_prompt", None)
            _stuck_fn = getattr(_guard, "check_prompt_stuck", None) if _guard is not None else None
            if callable(_stuck_fn) and callable(_prompt_fn):

                def _prompt_stuck() -> Optional[str]:
                    try:
                        if not bool(getattr(worker, "_is_gathering_active", False)) and getattr(
                            worker, "active_node", None
                        ) is None:
                            return None
                        _vis = _prompt_fn()
                        if isinstance(_vis, (tuple, list)) and _vis:
                            visible = bool(_vis[0])
                        elif isinstance(_vis, bool):
                            visible = _vis
                        else:
                            return None
                        _node = getattr(worker, "active_node", None)
                        _nid: Optional[int] = None
                        if isinstance(_node, dict):
                            _raw = _node.get("id")
                            _nid = _raw if isinstance(_raw, int) else None
                        elif hasattr(_node, "id"):
                            _raw = getattr(_node, "id")
                            _nid = _raw if isinstance(_raw, int) else None
                        stuck, msg = _stuck_fn(
                            visible, current_node_id=_nid, max_stuck_sec=10.0
                        )
                        if stuck:
                            return f"Prompt stuck for 10s: {msg}" if msg else "Prompt stuck for 10s"
                        return None
                    except Exception:
                        return None

                self.register_watchdog_detectors(prompt_stuck_detector=_prompt_stuck)
                wired["prompt_stuck"] = True

        if gm_whisper_detector is not None:
            self.register_watchdog_detectors(gm_whisper_detector=gm_whisper_detector)
            wired["gm_whisper"] = True
        elif worker is not None:
            _grab_screen = getattr(worker, "grab_screen", None)
            if callable(_grab_screen):

                def _gm_whisper_from_chat() -> bool:
                    try:
                        frame = _grab_screen()
                    except Exception:
                        return False
                    if frame is None:
                        return False
                    try:
                        from core.ocr_distance import detect_gm_whisper_chat

                        return bool(detect_gm_whisper_chat(frame))
                    except Exception:
                        return False

                self.register_watchdog_detectors(gm_whisper_detector=_gm_whisper_from_chat)
                wired["gm_whisper"] = True

        # fix/panik-kilit-oyun-yok: oyun penceresi varlığı — worker.watchdog
        # (BotWatchdog, main.py'de worker.watchdog = self.watchdog ile
        # bağlanır).is_game_window_present. Yoksa (eski/test worker'ı)
        # kayıtsız kalır; poll o durumda idle'ı eskisi gibi denetler.
        if worker is not None:
            _bot_watchdog = getattr(worker, "watchdog", None)
            _is_present_fn = (
                getattr(_bot_watchdog, "is_game_window_present", None)
                if _bot_watchdog is not None
                else None
            )
            if callable(_is_present_fn):

                def _game_window_present() -> Optional[bool]:
                    try:
                        result = _is_present_fn()
                    except Exception:
                        return None
                    if result is None:
                        return None
                    return bool(result)

                self.register_watchdog_detectors(
                    game_window_present_detector=_game_window_present
                )
                wired["game_window_present"] = True

            # fix/v20-rss-panik: simge durumu — worker.watchdog
            # (BotWatchdog).is_game_window_minimized. Yoksa (eski/test
            # worker'ı) kayıtsız kalır; poll o durumda idle'ı eskisi
            # gibi denetler.
            _is_minimized_fn = (
                getattr(_bot_watchdog, "is_game_window_minimized", None)
                if _bot_watchdog is not None
                else None
            )
            if callable(_is_minimized_fn):

                def _game_window_minimized() -> Optional[bool]:
                    try:
                        result = _is_minimized_fn()
                    except Exception:
                        return None
                    if result is None:
                        return None
                    return bool(result)

                self.register_watchdog_detectors(
                    game_window_minimized_detector=_game_window_minimized
                )
                wired["game_window_minimized"] = True

        missing = [name for name, ok in wired.items() if not ok]
        if missing:
            logger.warning(
                "Watchdog üretim kaynakları eksik bağlandı; kayıtsız sinyaller: %s "
                "(gm_whisper için repo'da üretim kaynağı yok).",
                ", ".join(missing),
            )
        return {"wired": wired, "missing": missing}

    def get_watchdog_wiring_status(self) -> Dict[str, Any]:
        """
        F-0163 (2. tur): watchdog'un hangi dış sinyalleri görebildiğini
        dürüstçe raporlar — poll'un kör olduğu sinyal gizlenmez.
        """
        with self._lock:
            detectors = {
                "character_dead": self._character_dead_detector is not None,
                "gm_whisper": self._gm_whisper_detector is not None,
                "prompt_stuck": self._prompt_stuck_detector is not None,
                "gather_heartbeat": self._gather_heartbeat_detector is not None,
                "combat_state": self._combat_state_detector is not None,
                "prompt_visible": self._prompt_visible_detector is not None,
                "game_window_present": self._game_window_present_detector is not None,
                "game_window_minimized": self._game_window_minimized_detector is not None,
                "training_active": self._training_active_detector is not None,
            }
            polling = self._watchdog_thread is not None and self._watchdog_thread.is_alive()
            failures = dict(self._detector_failures)
        missing = [name for name, ok in detectors.items() if not ok]
        # sweep-21 B-04: gm_whisper'ın repo'da üretim kaynağı yok — yapısal
        # eksiktir, "zorunlu" sayılmaz; yoksa fully_wired asla True olamaz
        # ve "degraded" kalıcı alarm yorgunluğuna dönerdi.
        # fix/panik-kilit-oyun-yok: game_window_present de "zorunlu"
        # sayılmaz — Windows dışı platformda (Linux/VM) belirlenemez;
        # aynı gerekçeyle missing_required'a girmez.
        # fix/v20-rss-panik: game_window_minimized de aynı gerekçeyle
        # "zorunlu" sayılmaz.
        # fix/v20-egitim-idle: training_active de aynı gerekçeyle
        # "zorunlu" sayılmaz — detektör kayıtlı olsa bile sinyal
        # bilinmiyor (None) dönebilir; yokluğu alarm yorgunluğu
        # yapmaz.
        required = [
            name
            for name in detectors
            if name not in ("gm_whisper", "game_window_present", "game_window_minimized", "training_active")
        ]
        missing_required = [name for name in required if not detectors[name]]
        faulted = [name for name, count in failures.items() if count >= 3]
        return {
            "module": "opsec_crypto",
            "watchdog_polling": polling,
            "detectors": detectors,
            "missing_detectors": missing,
            "missing_required_detectors": missing_required,
            "faulted_detectors": faulted,
            "detector_failures": failures,
            "fully_wired": not missing_required,
            # Dış sinyallerin ikisi de (ölüm + GM fısıltısı) bağlıysa poll
            # bunları da gözleyebilir; biri bile eksikse poll kısmen kördür.
            "external_signals_observable": bool(
                detectors["character_dead"] and detectors["gm_whisper"]
            ),
        }

    @staticmethod
    def _detector_timeout_sec() -> float:
        """Sweep-21 B-11: tek detektör çağrısı için zaman aşımı (sn)."""
        return 2.0

    def _call_detector(self, detector: Optional[Callable[[], Any]], name: str) -> Any:
        """Bir detektörü istisna- ve timeout-güvenli çağırır.

        sweep-21 B-11: eskiden timeout yoktu — yavaş/bozulan detektör poll
        turunu uzatır, istisna yutulup None döner ve "temiz" sayılırdı.
        Artık her çağrı 2 sn timeout'lu ayrı thread'de koşar; timeout/istisna
        ardışık hata sayacını artırır, 3 ardışık hatada detektör
        "faulted" sayılır (get_watchdog_wiring_status'te görünür).
        Yoksa/hatalıysa None döner.
        """
        if detector is None:
            return None
        box: Dict[str, Any] = {}

        def _run() -> None:
            try:
                box["value"] = detector()
            except Exception as exc:  # noqa: BLE001 — detektör hatası poll'u öldürmez
                box["error"] = exc

        # S-Leak: Prevent spawning unbounded threads if a previous check is still running/blocked
        if not hasattr(self, "_active_detector_threads"):
            self._active_detector_threads = {}
        prev = self._active_detector_threads.get(name)
        if prev is not None and prev.is_alive():
            with self._lock:
                count = int(self._detector_failures.get(name, 0)) + 1
                self._detector_failures[name] = count
            if count == 1 or count % 6 == 0:
                logger.warning(
                    "Watchdog detektörü '%s' hala önceki turdan yanıt bekliyor — yeni thread açılmadı, atlandı.",
                    name,
                )
            return None

        thread = threading.Thread(target=_run, name=f"WatchdogDetector-{name}", daemon=True)
        self._active_detector_threads[name] = thread
        thread.start()
        thread.join(timeout=self._detector_timeout_sec())
        if thread.is_alive():
            with self._lock:
                count = int(self._detector_failures.get(name, 0)) + 1
                self._detector_failures[name] = count
            if count == 1 or count % 3 == 0:
                logger.warning(
                    "Watchdog detektörü '%s' %.1f sn'de yanıt vermedi "
                    "(ardışık hata: %d) — bu tur yok sayıldı, temiz sayılmaz.",
                    name,
                    self._detector_timeout_sec(),
                    count,
                )
            return None
        if "error" in box:
            with self._lock:
                count = int(self._detector_failures.get(name, 0)) + 1
                self._detector_failures[name] = count
            if count == 1 or count % 3 == 0:
                logger.warning(
                    "Watchdog detektörü '%s' hata verdi: %s (ardışık hata: %d)",
                    name,
                    box["error"],
                    count,
                )
            return None
        with self._lock:
            self._detector_failures[name] = 0
        return box.get("value")

    def start_watchdog_polling(
        self,
        input_ctrl: Any = None,
        interval_sec: float = 5.0,
        play_siren: bool = True,
        character_dead_detector: Optional[Callable[[], bool]] = None,
        gm_whisper_detector: Optional[Callable[[], bool]] = None,
        prompt_stuck_detector: Optional[Callable[[], Optional[str]]] = None,
        gather_heartbeat_detector: Optional[Callable[[], bool]] = None,
        combat_state_detector: Optional[Callable[[], bool]] = None,
        prompt_visible_detector: Optional[Callable[[], bool]] = None,
        game_window_present_detector: Optional[Callable[[], Optional[bool]]] = None,
        game_window_minimized_detector: Optional[Callable[[], Optional[bool]]] = None,
        training_active_detector: Optional[Callable[[], Optional[bool]]] = None,
    ) -> bool:
        """
        F-0163 (dilim-içi kısım): check_watchdog_conditions'ı periyodik
        çağıran daemon thread'i başlatır; böylece watchdog üretimde de
        çalışır. Üretim bağlantısı: main.py PanicAlertSystem örneğinde bu
        metodu bir kez çağırmalıdır (main.py bu dilimin dosyası değildir);
        dış sinyaller için önce bind_production_watchdog_sources() çağrılır.
        Zaten çalışıyorsa False döner.
        Detektörler verilmezse poll yalnızca iç zamanlayıcılara (boşta
        kalma / savaş süresi / istem görünürlüğü) dayanır; karakter ölümü
        ve GM fısıltısı gibi dış sinyaller o durumda poll'dan gözlenemez —
        bu eksiklik artık sessiz değil, warning ile loglanır.
        """
        if (
            character_dead_detector is not None
            or gm_whisper_detector is not None
            or prompt_stuck_detector is not None
            or gather_heartbeat_detector is not None
            or combat_state_detector is not None
            or prompt_visible_detector is not None
            or game_window_present_detector is not None
            or game_window_minimized_detector is not None
            or training_active_detector is not None
        ):
            self.register_watchdog_detectors(
                character_dead_detector=character_dead_detector,
                gm_whisper_detector=gm_whisper_detector,
                prompt_stuck_detector=prompt_stuck_detector,
                gather_heartbeat_detector=gather_heartbeat_detector,
                combat_state_detector=combat_state_detector,
                prompt_visible_detector=prompt_visible_detector,
                game_window_present_detector=game_window_present_detector,
                game_window_minimized_detector=game_window_minimized_detector,
                training_active_detector=training_active_detector,
            )
        with self._lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return False
            self._watchdog_stop.clear()

        # F-0163 (2. tur): eksik detektörle başlayan poll sessizce kör
        # kalmasın — hangi dış sinyalin gözlenemediği açıkça loglanır.
        _wiring = self.get_watchdog_wiring_status()
        if _wiring["missing_detectors"]:
            logger.warning(
                "Watchdog poll dış sinyalsiz başlıyor; kayıtsız detektörler: %s — "
                "bu sinyaller poll'dan gözlenemez "
                "(bind_production_watchdog_sources ile bağlayın).",
                ", ".join(_wiring["missing_detectors"]),
            )

        def _poll_loop() -> None:
            while not self._watchdog_stop.is_set():
                try:
                    # F-0163: argümansız (sabit False/None) çağrı yok — her
                    # turda kayıtlı detektörler danışılır; detektör yoksa
                    # ilgili sinyal None/False kalır (eski varsayılan).
                    if self._call_detector(
                        self._gather_heartbeat_detector, "gather_heartbeat"
                    ):
                        self.record_successful_gather()
                    combat_state = self._call_detector(
                        self._combat_state_detector, "combat_state"
                    )
                    # fix/v20-panic-combat: tek poll'lük True sayacı
                    # kurmaz — _combat_poll_confirmed ardışık 2 True
                    # ister; False streak'i sıfırlar ve sayacı temizler.
                    # None (detektör yok/hatalı) dokunmaz.
                    if combat_state is not None:
                        if self._combat_poll_confirmed(bool(combat_state)):
                            self.set_combat_state(True)
                    prompt_visible = self._call_detector(
                        self._prompt_visible_detector, "prompt_visible"
                    )
                    if prompt_visible is not None:
                        self.set_interact_prompt_visible(bool(prompt_visible))
                        # PROMPT_STUCK panik kilidi varken istem kaybolduysa
                        # otomatik unlock (Başlat/Devam beklemeden recovery).
                        try:
                            self.maybe_auto_recover_prompt_stuck(
                                bool(prompt_visible), input_ctrl=input_ctrl
                            )
                        except Exception as _rec_exc:
                            logger.debug("prompt_stuck auto-recover skipped: %s", _rec_exc)
                    stuck_msg = self._call_detector(
                        self._prompt_stuck_detector, "prompt_stuck"
                    )
                    # fix/panik-kilit-oyun-yok: oyun penceresi yokken idle
                    # sayacı durur (check_watchdog_conditions içinde).
                    # Detektör yok/hatalı/belirsiz (None) ise eski davranış.
                    game_window_present = self._call_detector(
                        self._game_window_present_detector, "game_window_present"
                    )
                    # fix/v20-rss-panik: simge durumu da idle/savaş
                    # sayacını dondurur (oyun-yokluğu ile aynı kural).
                    game_window_minimized = self._call_detector(
                        self._game_window_minimized_detector, "game_window_minimized"
                    )
                    # fix/v20-egitim-idle: eğitim sürerken YALNIZCA
                    # idle sayacı donar (check_watchdog_conditions).
                    training_active = self._call_detector(
                        self._training_active_detector, "training_active"
                    )
                    self.check_watchdog_conditions(
                        input_ctrl=input_ctrl,
                        character_dead=bool(
                            self._call_detector(
                                self._character_dead_detector, "character_dead"
                            )
                        ),
                        prompt_stuck_msg=stuck_msg if isinstance(stuck_msg, str) else None,
                        gm_whisper_detected=bool(
                            self._call_detector(
                                self._gm_whisper_detector, "gm_whisper"
                            )
                        ),
                        play_siren=play_siren,
                        game_window_present=(
                            None
                            if game_window_present is None
                            else bool(game_window_present)
                        ),
                        game_window_minimized=(
                            None
                            if game_window_minimized is None
                            else bool(game_window_minimized)
                        ),
                        training_active=(
                            None if training_active is None else bool(training_active)
                        ),
                    )
                except Exception as exc:
                    logger.error("Watchdog poll hatası: %s", exc)
                self._watchdog_stop.wait(timeout=max(0.5, float(interval_sec)))

        thread = threading.Thread(
            target=_poll_loop, name="PanicWatchdogPollThread", daemon=True
        )
        with self._lock:
            self._watchdog_thread = thread
        thread.start()
        return True

    def stop_watchdog_polling(self) -> bool:
        """Watchdog poll thread'ini durdurur."""
        with self._lock:
            thread = self._watchdog_thread
            if thread is None or not thread.is_alive():
                self._watchdog_thread = None
                return False
        self._watchdog_stop.set()
        thread.join(timeout=2.0)
        with self._lock:
            if thread.is_alive():
                # sweep-21 B-12: thread detektörde takılı kaldıysa referans
                # DÜŞMEZ — yoksa sonraki start_watchdog_polling() ikinci
                # poll thread'i açar (çift poll; sayaç güncellemeleri yarışır).
                logger.warning(
                    "Watchdog poll thread'i 2 sn'de durmadı; referans korunuyor, "
                    "çift poll açılmayacak."
                )
                return False
            self._watchdog_thread = None
        return True

    def is_watchdog_polling(self) -> bool:
        with self._lock:
            return self._watchdog_thread is not None and self._watchdog_thread.is_alive()

    def get_opsec_crypto_diagnostic_snapshot(self) -> dict:
        # S-2201: sabit "ok"/True değil — gerçek panik + watchdog durumu.
        # status: "panic" (panik aktif) / "degraded" (detektör eksik) / "ok".
        with self._lock:
            panicked = self._panic_triggered
            halted = self._inputs_halted
        wiring = self.get_watchdog_wiring_status()
        # sweep-21 B-10: "ready" artık watchdog kablolamasını da hesaba
        # katar — eskiden status="degraded" iken ready=True dönüyordu.
        # gm_whisper yapısal eksik sayılmaz (missing_required'a girmez);
        # arızalı (faulted) detektör de hazır sayılmaz.
        missing_required = wiring.get(
            "missing_required_detectors", wiring["missing_detectors"]
        )
        faulted = wiring.get("faulted_detectors", [])
        if panicked:
            status = "panic"
        elif missing_required or faulted:
            status = "degraded"
        else:
            status = "ok"
        # fix/panic-lock: HUD/diagnostic tüketicisi kilidin sebebini de görür.
        panic_status = self.get_panic_status()
        return {
            "module": "opsec_crypto",
            "status": status,
            "ready": (not panicked)
            and validate_opsec_crypto_contract()
            and not missing_required
            and not faulted,
            "panic_triggered": panicked,
            "inputs_halted": halted,
            "panic_reason": panic_status["reason"],
            "panic_details": panic_status["details"],
            "panic_timestamp": panic_status["timestamp"],
            "panic_elapsed_sec": panic_status["elapsed_sec"],
            "watchdog_polling": wiring["watchdog_polling"],
            "missing_detectors": wiring["missing_detectors"],
            "missing_required_detectors": missing_required,
            "faulted_detectors": faulted,
        }


if __name__ == "__main__":
    import tempfile

    print("=== Katman V (opsec_crypto.py) Doğrulama Testleri Başlatılıyor ===")

    # 1. LocalVaultCrypto Testi
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_file = os.path.join(tmpdir, "settings.enc")
        salt_file = os.path.join(tmpdir, ".vault_salt")
        vault = LocalVaultCrypto(vault_path=vault_file, salt_path=salt_file)

        sample_cfg = {
            "gathering": {"min_f_interval": 1.5, "max_f_interval": 2.8},
            "route_waypoints": [{"x": 120, "y": 340}, {"x": 180, "y": 410}],
            "profile_name": "safe_gatherer_v5",
        }
        vault.save_encrypted_config(sample_cfg)
        loaded_cfg = vault.load_encrypted_config()
        assert loaded_cfg == sample_cfg, "Şifreli konfigürasyon çözme doğrulaması başarısız!"
        sig = vault.get_unique_build_signature()
        assert "build_signature" in sig and len(sig["build_signature"]) == 32
        print(f"[OK] LocalVaultCrypto doğrulandı -> İmza: {sig['build_signature']} ({sig['cipher_suite']})")

    # 2. NativeWin32HotkeyManager Testi
    hk_mgr = NativeWin32HotkeyManager()
    triggered_events: List[str] = []
    reg_res = hk_mgr.register_default_f8_f9(
        on_f8=lambda: triggered_events.append("F8_PRESSED"),
        on_f9=lambda: triggered_events.append("F9_PRESSED"),
    )
    hk_mgr.simulate_hotkey_event(NativeWin32HotkeyManager.HOTKEY_ID_F8)
    hk_mgr.simulate_hotkey_event(NativeWin32HotkeyManager.HOTKEY_ID_F9)
    time.sleep(0.08)
    hk_mgr.stop()
    assert "F8_PRESSED" in triggered_events and "F9_PRESSED" in triggered_events
    print(f"[OK] NativeWin32HotkeyManager doğrulandı -> Tetiklenenler: {triggered_events}, Kayıt: {reg_res}")

    # 3. PanicAlertSystem Testi
    class _DummyInputCtrl:
        def __init__(self) -> None:
            self.halted = False

        def press_key(self, key: str) -> bool:
            return not self.halted

    dummy_ctrl = _DummyInputCtrl()
    panic_sys = PanicAlertSystem(enable_sound=False)
    alert_res = panic_sys.check_watchdog_conditions(
        input_ctrl=dummy_ctrl,
        prompt_stuck_msg="Prompt stuck for 10s",
        play_siren=False,
    )
    assert alert_res is not None and alert_res["inputs_halted"] is True
    assert dummy_ctrl.press_key("f") is False
    print(f"[OK] PanicAlertSystem doğrulandı -> Sebep: {alert_res['reason']}, Girdi Kilidi: {alert_res['inputs_halted']}")


def get_opsec_crypto_capabilities() -> list:
    return ["opsec_crypto", "token_encryption", "diagnostic_snapshot"]


# S-2201: sözleşme probu için süreç-içi tekil kasa (temel dizinde değil,
# geçici dizinde kurulur; anahtar türetme maliyeti süreçte bir kez ödenir).
_CONTRACT_PROBE_VAULT: Optional["LocalVaultCrypto"] = None
_CONTRACT_PROBE_LOCK = threading.Lock()


def _get_opsec_crypto_contract_probe_vault() -> "LocalVaultCrypto":
    """Sözleşme probunun kullandığı geçici-dizin kasasını döndürür."""
    global _CONTRACT_PROBE_VAULT
    with _CONTRACT_PROBE_LOCK:
        if _CONTRACT_PROBE_VAULT is None:
            probe_dir = tempfile.mkdtemp(prefix="opsec_crypto_contract_")
            _CONTRACT_PROBE_VAULT = LocalVaultCrypto(
                vault_path=os.path.join(probe_dir, "settings.enc"),
                salt_path=os.path.join(probe_dir, ".vault_salt"),
            )
        return _CONTRACT_PROBE_VAULT


def get_opsec_crypto_contract_probe() -> dict:
    """
    S-2201: gerçek sözleşme probu — sabit dizge listesinin varlığına değil,
    bellek-içi şifrele/çöz round-trip'ine + anahtar malzemesine + yetenek
    listesinin boş olmamasına bakar. Kasa bozulursa / round-trip
    başarısız olursa contract_valid False döner (False dalı ölü değil).
    """
    capabilities = get_opsec_crypto_capabilities()
    caps_ok = bool(capabilities) and all(
        isinstance(c, str) and bool(c) for c in capabilities
    )
    roundtrip_ok = False
    cipher = None
    error = None
    try:
        vault = _get_opsec_crypto_contract_probe_vault()
        health = vault.get_opsec_crypto_health_summary()
        roundtrip_ok = bool(health.get("roundtrip_ok"))
        cipher = health.get("cipher")
        error = health.get("error")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "module": "opsec_crypto",
        "contract_valid": bool(caps_ok and roundtrip_ok),
        "capabilities_ok": bool(caps_ok),
        "capability_count": len(capabilities),
        "roundtrip_ok": roundtrip_ok,
        "cipher": cipher,
        "error": error,
    }


def validate_opsec_crypto_contract() -> bool:
    """
    S-2201: artık gerçek prob — bool(get_opsec_crypto_capabilities())
    totolojisi değil; round-trip başarısızsa False döner. Ondan türeyen
    readiness/audit/seal/certificate zinciri de böylece gerçek bir
    kontrolün sonucunu yansıtır.
    """
    try:
        return bool(get_opsec_crypto_contract_probe()["contract_valid"])
    except Exception as exc:
        logger.warning("opsec_crypto sözleşme probu hata verdi: %s", exc)
        return False


def get_opsec_crypto_baseline_metrics() -> dict:
    return {
        "module": "opsec_crypto",
        "contract_valid": validate_opsec_crypto_contract(),
        "capability_count": len(get_opsec_crypto_capabilities()),
    }


def get_opsec_crypto_readiness_status() -> dict:
    return {
        "module": "opsec_crypto",
        "ready": validate_opsec_crypto_contract(),
        "metrics": get_opsec_crypto_baseline_metrics(),
        # S-2201: hazır/uygunluk iddiasının dayandığı gerçek prob ayrıntısı.
        "probe": get_opsec_crypto_contract_probe(),
    }


def get_opsec_crypto_audit_record() -> dict:
    return {
        "module": "opsec_crypto",
        "audit_passed": validate_opsec_crypto_contract(),
        "readiness": get_opsec_crypto_readiness_status(),
    }


def get_opsec_crypto_operational_digest() -> dict:
    return {
        "module": "opsec_crypto",
        "operational": validate_opsec_crypto_contract(),
        "audit": get_opsec_crypto_audit_record(),
    }


def get_opsec_crypto_telemetry_payload() -> dict:
    return {
        "module": "opsec_crypto",
        "payload": get_opsec_crypto_operational_digest(),
    }


def get_opsec_crypto_integrity_seal() -> dict:
    return {
        "module": "opsec_crypto",
        "sealed": validate_opsec_crypto_contract(),
        "telemetry": get_opsec_crypto_telemetry_payload(),
    }


def get_opsec_crypto_compliance_certificate() -> dict:
    return {
        "module": "opsec_crypto",
        "certified": validate_opsec_crypto_contract(),
        "seal": get_opsec_crypto_integrity_seal(),
    }


def get_opsec_crypto_readiness_probe() -> dict:
    return {
        "module": "opsec_crypto",
        "ready": validate_opsec_crypto_contract(),
        "certificate": get_opsec_crypto_compliance_certificate(),
    }


def get_opsec_crypto_audit_summary() -> dict:
    return {
        "module": "opsec_crypto",
        "audit_passed": validate_opsec_crypto_contract(),
        "probe": get_opsec_crypto_readiness_probe(),
    }


