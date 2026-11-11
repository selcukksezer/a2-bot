# -*- coding: utf-8 -*-
"""D23 (opsec_crypto) iyileştirme kilitleme testleri.

Kapsar:
- I1: _ctr_xor_sha256_stream int-tabanlı XOR, eski bayt bayt uygulamayla
  birebir aynı çıktıyı verir (referans oracle test içinde).
- I2: makine parmak izi __init__ başına 1 kez toplanır.
- I3: eski (legacy) anahtar init'te türetilmez; yalnızca eski formatlı
  kasa çözülürken bir kez türetilir ve önbelleğe alınır.
- I4: Telegram sendMessage başarısı JSON "ok" alanıyla anlaşılır;
  ok:false gövdesi, description içinde '"ok":true' geçse bile
  "gönderildi" sayılmaz.
"""

import base64
import hashlib
import hmac
import json
import os
import struct
import tempfile
import unittest
from unittest import mock

from core.opsec_crypto import LocalVaultCrypto, PanicAlertSystem


def _reference_ctr_xor(key: bytes, nonce: bytes, payload: bytes) -> bytes:
    """D23 öncesi bayt bayt uygulamanın birebir kopyası (oracle)."""
    out = bytearray(len(payload))
    total_len = len(payload)
    offset = 0
    counter = 0
    while offset < total_len:
        counter_bytes = struct.pack(">Q", counter)
        keystream_block = hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest()
        chunk_len = min(32, total_len - offset)
        for i in range(chunk_len):
            out[offset + i] = payload[offset + i] ^ keystream_block[i]
        offset += chunk_len
        counter += 1
    return bytes(out)


def _make_vault(tmpdir: str) -> LocalVaultCrypto:
    return LocalVaultCrypto(
        vault_path=os.path.join(tmpdir, "settings.enc"),
        salt_path=os.path.join(tmpdir, ".vault_salt"),
    )


class TestCtrStreamEquivalence(unittest.TestCase):
    def test_matches_reference_oracle_all_sizes(self):
        key = b"K" * 32
        nonce = b"N" * 16
        for size in (0, 1, 15, 31, 32, 33, 63, 64, 65, 1000, 65536):
            payload = os.urandom(size)
            got = LocalVaultCrypto._ctr_xor_sha256_stream(key, nonce, payload)
            want = _reference_ctr_xor(key, nonce, payload)
            self.assertEqual(got, want, f"size={size}")

    def test_ctr_roundtrip_large_payload_via_vault(self):
        with tempfile.TemporaryDirectory() as td:
            vault = _make_vault(td)
            self.assertFalse(vault.prefer_aes_gcm)  # cryptography yok: CTR yolu canlı
            data = {"blob": "x" * (256 * 1024), "n": 42}
            token = vault.encrypt_json_payload(data)
            self.assertEqual(vault.decrypt_json_payload(token), data)

    def test_tampered_token_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            vault = _make_vault(td)
            token = vault.encrypt_json_payload({"a": 1})
            # Ortadaki bir baytı boz (MAC yakalamalı).
            bad = token[:20] + ("A" if token[20] != "A" else "B") + token[21:]
            with self.assertRaises(ValueError):
                vault.decrypt_json_payload(bad)


class TestFingerprintCollectedOnce(unittest.TestCase):
    def test_init_collects_fingerprint_once(self):
        orig = LocalVaultCrypto._collect_machine_fingerprint
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                LocalVaultCrypto,
                "_collect_machine_fingerprint",
                staticmethod(counting),
            ):
                _make_vault(td)
        self.assertEqual(len(calls), 1)


class TestLegacyKeyLazy(unittest.TestCase):
    def _build_legacy_token(self, vault, data):
        """Eski (100k iterasyon, gizlisiz) anahtarla CTR-HMAC v1 paketi üret."""
        legacy_master = hashlib.pbkdf2_hmac(
            "sha256",
            vault._machine_fingerprint,
            vault._local_salt,
            100_000,
            dklen=32,
        )
        enc = hmac.new(legacy_master, b"ENC-KEY-V1", hashlib.sha256).digest()
        mac = hmac.new(legacy_master, b"MAC-KEY-V1", hashlib.sha256).digest()
        plaintext = json.dumps(
            data, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        nonce = os.urandom(16)
        ciphertext = LocalVaultCrypto._ctr_xor_sha256_stream(enc, nonce, plaintext)
        header = bytes([1]) + nonce + ciphertext
        packet = header + hmac.new(mac, header, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(packet).decode("ascii")

    def test_legacy_key_not_derived_at_init_or_for_new_vaults(self):
        orig = LocalVaultCrypto._derive_machine_key
        calls = []

        def counting(self, *args, **kwargs):
            calls.append(1)
            return orig(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                LocalVaultCrypto, "_derive_machine_key", counting
            ):
                vault = _make_vault(td)
                self.assertEqual(len(calls), 1)  # yalnızca master (600k)
                self.assertIsNone(vault._legacy_key_pair)
                # Yeni formatlı kasa çözümü legacy türetimi tetiklemez.
                data = {"route": [1, 2, 3]}
                vault.decrypt_json_payload(vault.encrypt_json_payload(data))
                self.assertEqual(len(calls), 1)
                self.assertIsNone(vault._legacy_key_pair)

    def test_legacy_vault_still_decrypts_and_caches_key(self):
        orig = LocalVaultCrypto._derive_machine_key
        calls = []

        def counting(self, *args, **kwargs):
            calls.append(1)
            return orig(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                LocalVaultCrypto, "_derive_machine_key", counting
            ):
                vault = _make_vault(td)
                self.assertEqual(len(calls), 1)
                data = {"legacy": True, "v": 1}
                token = self._build_legacy_token(vault, data)
                self.assertEqual(vault.decrypt_json_payload(token), data)
                self.assertEqual(len(calls), 2)  # tembel legacy türetimi: 1 kez
                self.assertIsNotNone(vault._legacy_key_pair)
                # İkinci eski-kasa çözümü yeniden türetmez.
                self.assertEqual(vault.decrypt_json_payload(token), data)
                self.assertEqual(len(calls), 2)


class TestTelegramResponseOk(unittest.TestCase):
    def test_ok_true(self):
        self.assertTrue(
            PanicAlertSystem._telegram_response_ok(
                '{"ok": true, "result": {"message_id": 1}}'
            )
        )

    def test_ok_false(self):
        self.assertFalse(
            PanicAlertSystem._telegram_response_ok(
                '{"ok": false, "error_code": 403, "description": "Forbidden"}'
            )
        )

    def test_ok_false_with_ok_true_nested_is_not_sent(self):
        # Eski alt-dize kontrolü bunu True sayıyordu (yanlış pozitif):
        # iç içe "ok": true, üst seviye ok:false'ı gizliyordu.
        body = '{"ok": false, "result": {"ok": true, "message_id": 1}}'
        self.assertIn('"ok":true', body.replace(" ", "").lower())
        self.assertFalse(PanicAlertSystem._telegram_response_ok(body))

    def test_non_json_falls_back_to_substring(self):
        self.assertTrue(PanicAlertSystem._telegram_response_ok('plain "ok":true text'))
        self.assertFalse(PanicAlertSystem._telegram_response_ok("plain text"))

    def test_no_ok_field(self):
        self.assertFalse(
            PanicAlertSystem._telegram_response_ok('{"result": {"message_id": 1}}')
        )


class TestSendPanicNotificationTelegram(unittest.TestCase):
    def _fake_urlopen(self, body: str, status: int = 200):
        resp = mock.MagicMock()
        resp.status = status
        resp.read.return_value = body.encode("utf-8")
        cm = mock.MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        return cm

    def test_error_body_returns_false(self):
        panic = PanicAlertSystem(
            telegram_token="tok", telegram_chat_id="123", enable_sound=False
        )
        body = (
            '{"ok": false, "error_code": 403, "description": "Forbidden", '
            '"result": {"ok": true}}'
        )
        with mock.patch(
            "core.opsec_crypto.urllib.request.urlopen",
            return_value=self._fake_urlopen(body),
        ):
            self.assertFalse(panic.send_panic_notification("TEST", "detay"))

    def test_success_body_returns_true(self):
        panic = PanicAlertSystem(
            telegram_token="tok", telegram_chat_id="123", enable_sound=False
        )
        body = '{"ok": true, "result": {"message_id": 7}}'
        with mock.patch(
            "core.opsec_crypto.urllib.request.urlopen",
            return_value=self._fake_urlopen(body),
        ):
            self.assertTrue(panic.send_panic_notification("TEST", "detay"))

    def test_no_channel_returns_false_without_network(self):
        panic = PanicAlertSystem(enable_sound=False)
        with mock.patch(
            "core.opsec_crypto.urllib.request.urlopen",
            side_effect=AssertionError("ağ çağrısı olmamalı"),
        ):
            self.assertFalse(panic.send_panic_notification("TEST", "detay"))


if __name__ == "__main__":
    unittest.main()
