# -*- coding: utf-8 -*-
"""
Auto-Gather Module (Gatherion Auto-Record Style)
Detects 'F Gather' prompt in real time and automatically gathers resources.
"""

import time
import random
import winsound
import keyboard
from core.vision import Vision
from core.input_driver import InputDriver
from version import __version__


def _load_gather_settings() -> dict:
    """Reads config/settings.json (gathering section); {} on any failure."""
    try:
        from config.schema import SettingsManager

        return SettingsManager().load().to_dict().get("gathering", {}) or {}
    except Exception:
        return {}


class AutoGatherBot:
    def __init__(self, match_threshold=None, min_f_interval=None, max_f_interval=None):
        self.vision = Vision()
        self.driver = InputDriver()
        # Settings first (config/settings.json -> gathering), explicit args win,
        # historical defaults last — the bot used to ignore settings entirely.
        _settings = _load_gather_settings()
        self.match_threshold = 0.75 if match_threshold is None else match_threshold
        self.min_f_interval = (
            float(_settings.get("f_interval_min", 1.5))
            if min_f_interval is None
            else min_f_interval
        )
        self.max_f_interval = (
            float(_settings.get("f_interval_max", 3.0))
            if max_f_interval is None
            else max_f_interval
        )

        self.running = False
        self.paused = False
        self.gathered_count = 0
        self.f_pressed_count = 0
        self._hotkey_names: list = []

    def _register_hotkeys(self) -> None:
        """Registers hotkeys and verifies registration; a failed hotkey must
        not silently leave the bot without its pause control."""
        try:
            keyboard.add_hotkey("ctrl+alt+f9", self.toggle_pause, suppress=False)
            self._hotkey_names.append("ctrl+alt+f9")
        except Exception as exc:
            print(f"[Bot] Uyarı: Duraklatma kısayolu kaydedilemedi: {exc}")

    def _unregister_hotkeys(self) -> None:
        """Removes only this bot's hotkeys (never keyboard.unhook_all(),
        which would rip out every other module's hotkeys too). A failed
        removal only logs a warning — the remaining hotkeys are still
        attempted and the registry is cleared either way."""
        for name in list(self._hotkey_names):
            try:
                keyboard.remove_hotkey(name)
            except Exception as exc:
                print(f"[Bot] Uyarı: Kısayol kaldırılamadı ({name}): {exc}")
        self._hotkey_names.clear()

    def _sleep_interruptible(self, seconds: float, chunk: float = 0.2) -> bool:
        """Sleep in small chunks; return False as soon as ESC is pressed or
        the bot is stopped (same rationale as RouteRunner: a single
        time.sleep() after a harvest used to delay ESC by up to 1.5 s)."""
        deadline = time.monotonic() + max(0.0, seconds)
        while self.running:
            try:
                if keyboard.is_pressed("esc"):
                    return False
            except Exception:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(chunk, remaining))
        return False

    def start(self):
        self.running = True
        self.paused = False
        print("\n" + "=" * 60)
        print("AION 2 AUTO-GATHER (AUTO-RECORD) MODU BAŞLATILDI")
        print("Kontroller: [Ctrl+Alt+F9] Duraklat / Devam Et | [ESC] Çıkış")
        print("Karakterinle kaynakların yanına git, bot otomatik toplayacaktır.")
        print("=" * 60 + "\n")

        # Setup hotkeys
        self._register_hotkeys()

        last_f_time = 0
        is_gathering = False
        consecutive_not_found = 0
        # F interval is drawn once per gather session and redrawn after each
        # press — never re-drawn every tick (that would make the effective
        # interval random per tick instead of per press).
        f_interval = random.uniform(self.min_f_interval, self.max_f_interval)
        f_pressed_at_gather_start = 0

        try:
            while self.running:
                if keyboard.is_pressed('esc'):
                    print("\n[Bot] ESC basıldı, çıkış yapılıyor...")
                    break

                if self.paused:
                    time.sleep(0.2)
                    continue

                # Check for 'F Gather' on screen
                detected, match_pct, loc = self.vision.check_gather_prompt(
                    threshold=self.match_threshold,
                    use_roi=True
                )

                now = time.time()

                if detected:
                    consecutive_not_found = 0
                    if not is_gathering:
                        is_gathering = True
                        f_pressed_at_gather_start = self.f_pressed_count
                        f_interval = random.uniform(self.min_f_interval, self.max_f_interval)
                        print(f"\n[GATHERING] Kaynak tespit edildi! (Eşleşme: %{match_pct})")
                        try:
                            winsound.Beep(1200, 150) # Gentle prompt beep
                        except:
                            pass

                    # Press F at humanized intervals
                    if now - last_f_time >= f_interval:
                        self.driver.send_gather_key()
                        self.f_pressed_count += 1
                        last_f_time = now
                        f_interval = random.uniform(self.min_f_interval, self.max_f_interval)
                        print(f"[{time.strftime('%H:%M:%S')}] F basıldı (#{self.f_pressed_count}) | Eşleşme: %{match_pct}")

                else:
                    consecutive_not_found += 1
                    # If prompt missing for 3 consecutive checks (~0.5s), verify
                    # before counting: re-check once, and count only if F was
                    # actually pressed during this gather session. A prompt that
                    # merely flickered (or a session with no press) is not a
                    # completed harvest.
                    if is_gathering and consecutive_not_found > 3:
                        re_detected, _, _ = self.vision.check_gather_prompt(
                            threshold=self.match_threshold,
                            use_roi=True
                        )
                        if re_detected:
                            consecutive_not_found = 0
                        else:
                            is_gathering = False
                            if self.f_pressed_count > f_pressed_at_gather_start:
                                self.gathered_count += 1
                                print(f"[{time.strftime('%H:%M:%S')}] Toplama tamamlandı! (Toplam Toplanan: {self.gathered_count})")
                            else:
                                print(f"[{time.strftime('%H:%M:%S')}] Prompt kayboldu ama bu oturumda F basılmadı — toplandı sayılmadı.")
                            if not self._sleep_interruptible(random.uniform(0.8, 1.5)):
                                break

                # Sleep a tiny amount to keep CPU near 0-1%
                time.sleep(0.12)

        finally:
            self.running = False
            self._unregister_hotkeys()
            print("\nBot durduruldu.")

    def toggle_pause(self):
        self.paused = not self.paused
        state = "DURAKLATILDI" if self.paused else "DEVAM EDİYOR"
        print(f"\n>>> [F9] BOT {state} <<<")
        try:
            freq = 800 if self.paused else 1500
            winsound.Beep(freq, 150)
        except:
            pass

    def get_auto_gather_ready_flag(self) -> dict:
        """Real readiness probe (S-2703): vision templates/capture, the
        driver's trusted (hardware HID / Interception / FakerInput) key
        path, and the gather timing config are all checked. Returns
        ready=False with reasons instead of an unconditional True.

        Driver note: soft SendInput is dropped by AION 2 Raw Input, so a
        non-test-mode driver without a trusted backend is NOT ready —
        there is no silent soft fallback. In test_mode the simulated
        input path counts as ready.
        """
        problems: list = []

        # --- config sanity -------------------------------------------------
        try:
            threshold = float(self.match_threshold)
            if not 0.0 < threshold <= 1.0:
                problems.append(f"match_threshold_out_of_range={self.match_threshold!r}")
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            problems.append(f"match_threshold_error={exc}")
        try:
            f_min = float(self.min_f_interval)
            f_max = float(self.max_f_interval)
            if not 0.0 < f_min <= f_max:
                problems.append(
                    f"f_interval_out_of_range=({self.min_f_interval!r}, {self.max_f_interval!r})"
                )
        except Exception as exc:  # noqa: BLE001 - probe must not raise
            problems.append(f"f_interval_error={exc}")

        # --- vision: gather templates loaded + capture dimensions sane ----
        vision_ready = False
        vision = getattr(self, "vision", None)
        if vision is None:
            problems.append("vision_missing")
        else:
            if not callable(getattr(vision, "check_gather_prompt", None)):
                problems.append("vision_missing_check_gather_prompt")
            try:
                flag_fn = getattr(vision, "get_vision_engine_ready_flag", None)
                if callable(flag_fn):
                    vision_ready = bool(flag_fn().get("ready", False))
                elif callable(getattr(vision, "_is_engine_ready", None)):
                    vision_ready = bool(vision._is_engine_ready())
                else:
                    templates_ok = bool(
                        getattr(vision, "_template_variants", None)
                        or getattr(vision, "gather_template", None) is not None
                    )
                    capture_ok = (
                        getattr(vision, "screen_w", 0) > 0
                        and getattr(vision, "screen_h", 0) > 0
                    )
                    vision_ready = bool(templates_ok and capture_ok)
            except Exception as exc:  # noqa: BLE001 - probe must not raise
                problems.append(f"vision_probe_error={exc}")
            if not vision_ready:
                problems.append("vision_not_ready")

        # --- driver: trusted hardware key path (or simulated test mode) ----
        driver_ready = False
        driver_backend = None
        driver = getattr(self, "driver", None)
        if driver is None:
            problems.append("driver_missing")
        else:
            if not callable(getattr(driver, "send_gather_key", None)):
                problems.append("driver_missing_send_gather_key")
            try:
                driver_backend = getattr(driver, "active_backend_name", None)
            except Exception:  # noqa: BLE001 - probe must not raise
                driver_backend = None
            if getattr(driver, "test_mode", False):
                # Simulated input: no hardware backend needed.
                driver_ready = True
            else:
                trusted = getattr(driver, "keyboard_inject_trusted", None)
                if isinstance(trusted, bool):
                    driver_ready = trusted
                elif callable(getattr(driver, "is_keyboard_trusted", None)):
                    try:
                        driver_ready = bool(driver.is_keyboard_trusted())
                    except Exception as exc:  # noqa: BLE001 - probe must not raise
                        problems.append(f"driver_probe_error={exc}")
                if not driver_ready:
                    problems.append(
                        "no_trusted_keyboard_backend="
                        f"{driver_backend or 'unknown'}"
                    )

        return {
            "module": "auto_gather",
            "ready": not problems,
            "problems": problems,
            "vision_ready": vision_ready,
            "driver_ready": driver_ready,
            "driver_backend": driver_backend,
        }

    def get_auto_gather_version_info(self) -> dict:
        """Version info with REAL readiness (S-6701): ready comes from the
        live probe (get_auto_gather_ready_flag — vision templates/capture
        + trusted hardware key path), never an unconditional True. Mirrors
        core/vision.py get_vision_engine_version_info. Probe failure is
        fail-closed (ready=False) and never raises."""
        try:
            flag = self.get_auto_gather_ready_flag()
        except Exception as exc:  # noqa: BLE001 - diagnostic must not raise
            return {
                "module": "auto_gather",
                "version": __version__,
                "ready": False,
                "problems": [f"ready_probe_error={exc}"],
            }
        return {
            "module": "auto_gather",
            "version": __version__,
            "ready": bool(flag.get("ready", False)),
            "problems": list(flag.get("problems", [])),
        }

    def get_auto_gather_diagnostic_snapshot(self) -> dict:
        """Diagnostic snapshot with REAL readiness (S-6701): status is
        'ok' only when the live probe passes, else 'degraded'. Mirrors
        core/vision.py get_vision_engine_diagnostic_snapshot. Never raises."""
        try:
            flag = self.get_auto_gather_ready_flag()
        except Exception as exc:  # noqa: BLE001 - diagnostic must not raise
            return {
                "module": "auto_gather",
                "status": "degraded",
                "ready": False,
                "problems": [f"ready_probe_error={exc}"],
            }
        ready = bool(flag.get("ready", False))
        return {
            "module": "auto_gather",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "problems": list(flag.get("problems", [])),
            "vision_ready": bool(flag.get("vision_ready", False)),
            "driver_ready": bool(flag.get("driver_ready", False)),
        }

if __name__ == "__main__":
    bot = AutoGatherBot()
    bot.start()


def get_auto_gather_capabilities() -> list:
    return ["auto_gather", "proximity_harvesting", "diagnostic_snapshot"]


def validate_auto_gather_contract() -> bool:
    """Real contract check (S-2703, Karar 3 — proba bağlama): capabilities
    are present, the required vision/driver APIs exist on the real
    classes, and the gather timing settings are sane. Fail-closed on
    any error. The live vision/driver state probe (templates loaded,
    trusted hardware key path) is AutoGatherBot.get_auto_gather_ready_flag,
    which needs an instance; this module-level check verifies the
    contract that probe depends on, without constructing heavy objects
    on every call (the wrappers below call this repeatedly)."""
    try:
        if not get_auto_gather_capabilities():
            return False
        for cls, names in (
            (
                AutoGatherBot,
                ("get_auto_gather_ready_flag", "start", "toggle_pause"),
            ),
            (
                Vision,
                ("check_gather_prompt", "get_vision_engine_ready_flag"),
            ),
            (
                InputDriver,
                ("send_gather_key", "keyboard_inject_trusted"),
            ),
        ):
            for name in names:
                if getattr(cls, name, None) is None:
                    return False
        settings = _load_gather_settings()
        f_min = float(settings.get("f_interval_min", 1.5))
        f_max = float(settings.get("f_interval_max", 3.0))
        if not 0.0 < f_min <= f_max:
            return False
        return True
    except Exception:
        return False


def get_auto_gather_baseline_metrics() -> dict:
    return {
        "module": "auto_gather",
        "contract_valid": validate_auto_gather_contract(),
        "gather_ready": validate_auto_gather_contract(),
    }


def get_auto_gather_readiness_status() -> dict:
    return {
        "module": "auto_gather",
        "ready": validate_auto_gather_contract(),
        "metrics": get_auto_gather_baseline_metrics(),
    }


def get_auto_gather_audit_record() -> dict:
    return {
        "module": "auto_gather",
        "audit_passed": validate_auto_gather_contract(),
        "readiness": get_auto_gather_readiness_status(),
    }


def get_auto_gather_operational_digest() -> dict:
    return {
        "module": "auto_gather",
        "operational": validate_auto_gather_contract(),
        "audit": get_auto_gather_audit_record(),
    }


def get_auto_gather_telemetry_payload() -> dict:
    return {
        "module": "auto_gather",
        "payload": get_auto_gather_operational_digest(),
    }


def get_auto_gather_integrity_seal() -> dict:
    return {
        "module": "auto_gather",
        "sealed": validate_auto_gather_contract(),
        "telemetry": get_auto_gather_telemetry_payload(),
    }


def get_auto_gather_compliance_certificate() -> dict:
    return {
        "module": "auto_gather",
        "certified": validate_auto_gather_contract(),
        "seal": get_auto_gather_integrity_seal(),
    }


def get_auto_gather_readiness_probe() -> dict:
    return {
        "module": "auto_gather",
        "ready": validate_auto_gather_contract(),
        "certificate": get_auto_gather_compliance_certificate(),
    }


def get_auto_gather_audit_summary() -> dict:
    return {
        "module": "auto_gather",
        "audit_passed": validate_auto_gather_contract(),
        "probe": get_auto_gather_readiness_probe(),
    }


