# -*- coding: utf-8 -*-
"""
Full Route Runner Module (State Machine)
Orchestrates:
1. Ready node selection (5-min cooldown)
2. Map opening and navigation click
3. Walking & arrival monitoring
4. Auto-gathering when 'F Gather' is detected
5. Repeating along the route loop
"""

import time
import random
import weakref
import keyboard
import winsound
from enum import Enum
from core.input_driver import InputDriver
from core.vision import Vision
from core.route_manager import RouteManager
from core.navigator import MapNavigator
from modules.recorder import NodeRecorder
from version import __version__


# S-6603: modül düzeyi sarmalayıcıların canlı proba bağlanabilmesi için
# yaşayan RouteRunner örneğine zayıf referans (main.py'deki
# _LIVE_ORCHESTRATOR_REF deseni). Zayıf referans: örnek GC edilince
# kendiliğinden boşalır, örneği hayatta tutmaz.
_LIVE_ROUTE_RUNNER_REF = None


def _load_route_settings() -> dict:
    """Reads config/settings.json (route section); {} on any failure."""
    try:
        from config.schema import SettingsManager

        return SettingsManager().load().to_dict().get("route", {}) or {}
    except Exception:
        return {}


class BotState(Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    WALKING = "WALKING"
    GATHERING = "GATHERING"
    PAUSED = "PAUSED"

class RouteRunner:
    def __init__(self, rest_duration_sec=None, walk_timeout_sec=None):
        self.routes = RouteManager()
        self.vision = Vision()
        self.driver = InputDriver()
        self.navigator = MapNavigator(self.routes, self.vision, self.driver)
        self.recorder = NodeRecorder(self.routes, self.vision, self.driver, self.navigator)

        # Settings first (config/settings.json -> route: rest_minutes,
        # time_to_reach_sec), explicit args win, historical defaults last —
        # the runner used to ignore settings entirely.
        _settings = _load_route_settings()
        if rest_duration_sec is None:
            rest_duration_sec = float(_settings.get("rest_minutes", 5)) * 60.0
        if walk_timeout_sec is None:
            walk_timeout_sec = float(_settings.get("time_to_reach_sec", 60))
        self.rest_duration_sec = rest_duration_sec
        self.walk_timeout_sec = walk_timeout_sec

        self.state = BotState.IDLE
        self.running = False
        self.active_node = None
        self.gathered_total = 0
        self._state_before_pause: BotState | None = None
        self._hotkey_names: list = []

        # S-6603: tam kurulmuş örneği modül düzeyi canlı proba kaydet
        # (yalnızca __init__ başarıyla biterse; yarım örnek kaydedilmez).
        global _LIVE_ROUTE_RUNNER_REF
        try:
            _LIVE_ROUTE_RUNNER_REF = weakref.ref(self)
        except Exception:
            pass

    def _register_hotkeys(self) -> None:
        """Registers hotkeys and verifies registration."""
        for name, callback in (
            ("ctrl+alt+f8", self.recorder.record_current_node),
            ("ctrl+alt+f9", self.toggle_pause),
        ):
            try:
                keyboard.add_hotkey(name, callback, suppress=False)
                self._hotkey_names.append(name)
            except Exception as exc:
                print(f"[Runner] Uyarı: Kısayol kaydedilemedi ({name}): {exc}")

    def _unregister_hotkeys(self) -> None:
        """Removes only this runner's hotkeys (never keyboard.unhook_all(),
        which would rip out every other module's hotkeys too) — not even on
        the error path: a failed removal only logs and moves on to the next
        hotkey (S-2705)."""
        for name in list(self._hotkey_names):
            try:
                keyboard.remove_hotkey(name)
            except Exception as exc:
                print(f"[Runner] Uyarı: Kısayol kaldırılamadı ({name}): {exc}")
        self._hotkey_names.clear()

    def _sleep_interruptible(self, seconds: float, chunk: float = 0.2) -> bool:
        """Sleep in small chunks; return False as soon as ESC is pressed or
        the runner is stopped. A single time.sleep(30) used to make ESC
        unresponsive for up to 30 s while waiting for nodes to rest; the
        0.2 s chunk keeps the reaction time under ~0.2 s at the cost of
        ~150 wakeups over a full 30 s wait."""
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
        self.state = BotState.NAVIGATING

        print("\n" + "=" * 65)
        print("AION 2 ROTA VE OTO-TOPLAMA MOTORU (ROUTE MODE)")
        print("Kontroller:")
        print("  [Ctrl+Alt+F8]  - Bulunduğun yerdeki kaynağı anında rotaya kaydet")
        print("  [Ctrl+Alt+F9]  - Duraklat / Devam Et")
        print("  [ESC] - Çıkış")
        print("=" * 65 + "\n")

        # Global hotkeys
        self._register_hotkeys()

        try:
            walk_start_time = 0
            last_f_press = 0
            gathering_lost_counter = 0
            # F interval is drawn once per gather session and redrawn after
            # each press — never re-drawn every tick.
            f_interval = random.uniform(1.6, 2.8)

            while self.running:
                try:
                    if keyboard.is_pressed('esc'):
                        print("\n[Runner] ESC basıldı, kapatılıyor...")
                        break

                    if self.state == BotState.PAUSED:
                        time.sleep(0.2)
                        continue

                    # -----------------------------------------------------------------
                    # STATE: NAVIGATING -> Pick next ready node & click on map
                    # -----------------------------------------------------------------
                    if self.state == BotState.NAVIGATING:
                        ready_node = self.routes.get_next_ready_node(self.rest_duration_sec)
                        if not ready_node:
                            print(f"[{time.strftime('%H:%M:%S')}] Dinlenen tüm kaynaklar hazır olana kadar bekleniyor (30s)...")
                            if not self._sleep_interruptible(30):
                                print("\n[Runner] ESC basıldı, kapatılıyor...")
                                break
                            continue

                        self.active_node = ready_node
                        # P1-10 (K-14): navigate_to_node yalnızca harita
                        # tıklaması yapar; seyahat izleme + zaman aşımı
                        # aşağıdaki WALKING durumunda (walk_timeout_sec).
                        success = self.navigator.navigate_to_node(ready_node)
                        if success:
                            self.state = BotState.WALKING
                            walk_start_time = time.time()
                        else:
                            print(f"[Runner] {ready_node['name']} navigasyonu başarısız, atlanıyor.")
                            self.routes.mark_node_failed(
                                ready_node["id"],
                                active_route=getattr(self.routes, "active_route", None),
                            )
                            if not self._sleep_interruptible(2.0):
                                print("\n[Runner] ESC basıldı, kapatılıyor...")
                                break

                    # -----------------------------------------------------------------
                    # STATE: WALKING -> Monitoring movement towards node
                    # -----------------------------------------------------------------
                    elif self.state == BotState.WALKING:
                        # Check if 'F Gather' prompt appeared on screen
                        detected, match_pct, _ = self.vision.check_gather_prompt(threshold=0.75, use_roi=True)
                        if detected:
                            print(f"\n[{time.strftime('%H:%M:%S')}] Hedefe ulaşıldı! Prompt eşleşmesi: %{match_pct}")
                            self.state = BotState.GATHERING
                            gathering_lost_counter = 0
                            last_f_press = 0
                            f_interval = random.uniform(1.6, 2.8)
                            continue

                        # Check walking timeout (e.g. character got stuck on geometry)
                        if time.time() - walk_start_time > self.walk_timeout_sec:
                            print(f"\n[Runner] Zaman aşımı ({self.walk_timeout_sec}s): Düğüm {self.active_node.get('name', self.active_node.get('id'))} ulaşılamadı veya toplanacak kaynak yok!")
                            self.routes.mark_node_failed(
                                self.active_node["id"],
                                active_route=getattr(self.routes, "active_route", None),
                            )
                            self.state = BotState.NAVIGATING
                            if not self._sleep_interruptible(1.0):
                                print("\n[Runner] ESC basıldı, kapatılıyor...")
                                break
                            continue

                        time.sleep(0.15)

                    # -----------------------------------------------------------------
                    # STATE: GATHERING -> Pressing F until harvest finishes
                    # -----------------------------------------------------------------
                    elif self.state == BotState.GATHERING:
                        detected, match_pct, _ = self.vision.check_gather_prompt(threshold=0.75, use_roi=True)
                        now = time.time()

                        if detected:
                            gathering_lost_counter = 0
                            # Humanized press interval: 1.6s to 2.8s (fixed for
                            # this gather session; redrawn after each press)
                            if now - last_f_press >= f_interval:
                                self.driver.send_gather_key()
                                last_f_press = now
                                f_interval = random.uniform(1.6, 2.8)
                                print(f"[{time.strftime('%H:%M:%S')}] F basıldı...")
                        else:
                            gathering_lost_counter += 1
                            if gathering_lost_counter > 4: # Prompt gone for ~0.6s
                                print(f"\n[{time.strftime('%H:%M:%S')}] Kaynak başarıyla toplandı!")
                                self.gathered_total += 1
                                self.routes.mark_node_visited(self.active_node["id"])
                                print(f"[Runner] Toplam toplanan: {self.gathered_total} kaynak. Sıradaki hedefe geçiliyor...")
                                self.state = BotState.NAVIGATING
                                if not self._sleep_interruptible(random.uniform(1.0, 2.0)):
                                    print("\n[Runner] ESC basıldı, kapatılıyor...")
                                    break

                        time.sleep(0.12)
                except Exception as exc:
                    # One failing tick must never kill the engine: log it and
                    # keep running from the current state.
                    print(f"[Runner] Hata (motor çalışmaya devam ediyor): {exc!r}")
                    time.sleep(1.0)
                    continue

        finally:
            self.running = False
            self._unregister_hotkeys()
            print("\nRota motoru durduruldu.")

    def toggle_pause(self):
        if self.state == BotState.PAUSED:
            # Resume into the state we paused from — never reset to
            # NAVIGATING (that would abandon an in-progress walk/gather).
            self.state = self._state_before_pause or BotState.NAVIGATING
            self._state_before_pause = None
            print("\n>>> [F9] BOT DEVAM EDİYOR <<<")
            try: winsound.Beep(1500, 150)
            except: pass
        else:
            self._state_before_pause = self.state
            self.state = BotState.PAUSED
            print("\n>>> [F9] BOT DURAKLATILDI <<<")
            try: winsound.Beep(700, 200)
            except: pass

    def _probe_components(self) -> dict:
        """Probe real routes/vision/driver state (S-2702).

        routes: RouteManager.load_all_regions() reads
            routes/*/region_data.json from disk; ok when at least one region
            loads.
        vision: VisionEngine._is_engine_ready() — gather templates loaded
            and capture dimensions sane; no screen capture is taken here.
        driver: InputDriver.keyboard_inject_trusted — a kernel/HID keyboard
            path (Interception/FakerInput/Serial HID) exists. Soft SendInput
            is dropped by AION 2 Raw Input, so an untrusted driver means the
            runner cannot actually gather.
        Any probe raising is treated as that component not ready.
        """
        try:
            data = self.routes.load_all_regions()
            regions = data.get("regions", {}) if isinstance(data, dict) else {}
            routes_ok = bool(regions)
            routes_detail = {
                "regions_loaded": len(regions),
                "active_region": getattr(self.routes, "active_region", None),
            }
        except Exception as exc:
            routes_ok = False
            routes_detail = {"error": repr(exc)}
        try:
            is_ready = getattr(self.vision, "_is_engine_ready", None)
            if callable(is_ready):
                vision_ok = bool(is_ready())
            else:
                templates_ok = bool(
                    getattr(self.vision, "gather_template", None)
                    or getattr(self.vision, "_template_variants", None)
                )
                capture_ok = int(getattr(self.vision, "screen_w", 0) or 0) > 0 and int(
                    getattr(self.vision, "screen_h", 0) or 0
                ) > 0
                vision_ok = bool(templates_ok and capture_ok)
        except Exception:
            vision_ok = False
        try:
            driver_ok = bool(getattr(self.driver, "keyboard_inject_trusted", False))
        except Exception:
            driver_ok = False
        try:
            driver_backend = str(getattr(self.driver, "active_backend_name", "") or "")
        except Exception:
            driver_backend = ""
        ready = bool(routes_ok and vision_ok and driver_ok)
        return {
            "ready": ready,
            "routes_ok": routes_ok,
            "vision_ok": vision_ok,
            "driver_ok": driver_ok,
            "driver_backend": driver_backend,
            "routes": routes_detail,
        }

    def get_route_runner_ready_flag(self) -> dict:
        probe = self._probe_components()
        return {"module": "route_runner", **probe}

    def get_route_runner_version_info(self) -> dict:
        probe = self._probe_components()
        return {"module": "route_runner", "version": __version__, "ready": probe["ready"]}

    def get_route_runner_diagnostic_snapshot(self) -> dict:
        probe = self._probe_components()
        return {
            "module": "route_runner",
            "status": "ok" if probe["ready"] else "degraded",
            "ready": probe["ready"],
            "routes_ok": probe["routes_ok"],
            "vision_ok": probe["vision_ok"],
            "driver_ok": probe["driver_ok"],
        }

if __name__ == "__main__":
    # Standalone entry point by design (F-1097 / decision 5): RouteRunner is
    # an alternative entry point, not wired into the main bot flow.
    runner = RouteRunner()
    runner.start()


def get_route_runner_capabilities() -> list:
    return ["route_runner", "sequential_execution", "diagnostic_snapshot"]


def validate_route_runner_contract() -> bool:
    """Real contract check (not an unconditional True): capabilities are
    present, the public methods the runner depends on exist on
    RouteRunner/RouteManager/Vision/InputDriver, and the route settings
    source reads as a dict. No component is instantiated here — live
    instance state is probed by RouteRunner.get_route_runner_ready_flag()."""
    try:
        if not get_route_runner_capabilities():
            return False
        for cls, names in (
            (RouteRunner, ("start", "toggle_pause", "get_route_runner_ready_flag", "_unregister_hotkeys")),
            (RouteManager, ("get_next_ready_node", "load_all_regions", "get_route_nodes")),
            (Vision, ("check_gather_prompt", "is_map_open")),
            (InputDriver, ("send_gather_key",)),
        ):
            for name in names:
                if not callable(getattr(cls, name, None)):
                    return False
        if not isinstance(_load_route_settings(), dict):
            return False
        return True
    except Exception:
        return False


def _get_live_route_runner():
    """Kayıtlı yaşayan RouteRunner örneğini döndürür (yoksa None).

    S-6603 / S-0204 deseni: modül düzeyi sarmalayıcılar canlı probu bu
    kayıt üzerinden çözer; kayıt yoksa canlı prob yoktur.
    """
    ref = _LIVE_ROUTE_RUNNER_REF
    if ref is None:
        return None
    try:
        return ref()
    except Exception:
        return None


def _live_route_runner_probe() -> tuple:
    """(probe, live): canlı örneğin _probe_components() sonucu.

    live=False ise kayıtlı yaşayan örnek yoktur ve probe None'dır.
    Probun kendisi hata verirse bileşenler hazır sayılmaz (ready=False),
    ama live=True kalır — örnek var, probu başarısız.
    """
    runner = _get_live_route_runner()
    if runner is None:
        return None, False
    try:
        probe = runner._probe_components()
    except Exception as exc:
        return {
            "ready": False,
            "routes_ok": False,
            "vision_ok": False,
            "driver_ok": False,
            "driver_backend": "",
            "routes": {"error": repr(exc)},
        }, True
    if not isinstance(probe, dict) or "ready" not in probe:
        return {
            "ready": False,
            "routes_ok": False,
            "vision_ok": False,
            "driver_ok": False,
            "driver_backend": "",
            "routes": {"error": "probe did not return a readiness dict"},
        }, True
    return probe, True


def _route_runner_live_status() -> dict:
    """Sözleşme + canlı prob birleşimi (S-6603).

    Canlı örnek varsa hazır olma = sözleşme VE canlı prob (bölge
    yüklenmemiş / sürücü güvensiz ise ready=False). Canlı örnek yoksa
    yalnızca statik sözleşme raporlanır ve "live": False ile işaretlenir
    — çalışan bir örneğin sağlığı "hazır" sayılmaz; bu geri dönüş
    main.py'deki S-0204 kararıyla aynıdır ve
    test_validate_contract_true_with_real_classes'ın sabitlediği
    sözleşme metriğini korur.
    """
    contract_valid = validate_route_runner_contract()
    probe, live = _live_route_runner_probe()
    live_ready = bool(probe.get("ready")) if isinstance(probe, dict) else False
    ready = bool(contract_valid and live_ready) if live else bool(contract_valid)
    return {
        "contract_valid": contract_valid,
        "live": live,
        "live_ready": live_ready,
        "ready": ready,
        "probe": probe,
    }


def get_route_runner_baseline_metrics() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "contract_valid": status["contract_valid"],
        # S-6603: canlı örnek varken runner_ready canlı proba bağlıdır;
        # canlı örnek yokken sözleşme metriğidir ("live": False).
        "runner_ready": status["ready"],
        "live": status["live"],
        "live_ready": status["live_ready"],
        "probe": status["probe"],
    }


def get_route_runner_readiness_status() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "ready": status["ready"],
        "live": status["live"],
        "live_ready": status["live_ready"],
        "probe": status["probe"],
        "metrics": get_route_runner_baseline_metrics(),
    }


def get_route_runner_audit_record() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "audit_passed": status["ready"],
        "live": status["live"],
        "readiness": get_route_runner_readiness_status(),
    }


def get_route_runner_operational_digest() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "operational": status["ready"],
        "live": status["live"],
        "audit": get_route_runner_audit_record(),
    }


def get_route_runner_telemetry_payload() -> dict:
    return {"module": "route_runner", "payload": get_route_runner_operational_digest()}


def get_route_runner_integrity_seal() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "sealed": status["ready"],
        "live": status["live"],
        "telemetry": get_route_runner_telemetry_payload(),
    }


def get_route_runner_compliance_certificate() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "certified": status["ready"],
        "live": status["live"],
        "seal": get_route_runner_integrity_seal(),
    }


def get_route_runner_readiness_probe() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "ready": status["ready"],
        "live": status["live"],
        "certificate": get_route_runner_compliance_certificate(),
    }


def get_route_runner_audit_summary() -> dict:
    status = _route_runner_live_status()
    return {
        "module": "route_runner",
        "audit_passed": status["ready"],
        "live": status["live"],
        "probe": get_route_runner_readiness_probe(),
    }


