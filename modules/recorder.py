# -*- coding: utf-8 -*-
"""
Node Recorder Module (F8 Recording)
Captures player location on map and saves it as a route node.
"""

import time
import winsound
import cv2
import numpy as np
from core.input_driver import InputDriver
from core.vision import Vision
from core.route_manager import RouteManager
from core.navigator import MapNavigator
from version import __version__

class NodeRecorder:
    def __init__(self, route_manager: RouteManager, vision: Vision, driver: InputDriver, navigator: MapNavigator):
        self.routes = route_manager
        self.vision = vision
        self.driver = driver
        self.navigator = navigator

    def record_current_node(self):
        """
        Triggered when user presses F8:
        1. Opens map (M)
        2. Detects player position arrow on the map (returns None when
           the arrow is not found — see Vision.find_player_on_map;
           there is NO centre fallback since Sweep G, so a missing
           arrow must fail the recording with a clear message instead
           of unpacking None)
        3. Crops small grayscale map patch (80x80, matching RouteManager)
        4. Closes map (M) — always, via try/finally, even on error
        5. Saves to RouteManager and plays pleasant confirmation beep
        """
        print("\n[Recorder] F8 algılandı! Düğüm kaydediliyor...")

        # 1. Open map
        opened = self.navigator.open_map()
        if not opened:
            print("[Recorder] Hata: Harita açılamadı!")
            try:
                winsound.Beep(500, 300) # Low error beep
            except:
                pass
            return None

        try:
            time.sleep(0.2) # Allow map rendering

            # 2. Grab map screen and locate the player arrow on it
            map_gray = self.vision.grab_screen(grayscale=True)
            if map_gray is None or getattr(map_gray, "size", 0) == 0:
                print("[Recorder] Hata: Harita görüntüsü alınamadı!")
                return None
            h, w = map_gray.shape

            # Detect the player arrow instead of assuming it is at the centre.
            # find_player_on_map() tries the arrow template + cyan segmentation
            # and returns None when neither finds the arrow (Sweep G: no
            # centre fallback — the centre is not the player position).
            # P0-9 (K-15): None doğrudan unpack ediliyordu
            # (center_x, center_y = None -> TypeError) ve F8 kaydı
            # çöküyordu. Guard: ok bulunamazsa kayıt anlamlı mesaj +
            # hata bipiyle başarısız olur, çökmez (harita finally ile
            # yine kapanır).
            player_pos = self.vision.find_player_on_map(map_gray)
            if not isinstance(player_pos, (tuple, list)) or len(player_pos) < 2:
                print("[Recorder] Hata: Oyuncu oku haritada bulunamadı — kayıt iptal edildi.")
                try:
                    winsound.Beep(500, 300)  # Low error beep
                except Exception:
                    pass
                return None
            center_x, center_y = int(player_pos[0]), int(player_pos[1])

            # 3. Crop 80x80 patch around player location (RouteManager expects
            # 80x80 node patches; a 60x60 patch mismatches its matcher)
            patch_size = 80
            half = patch_size // 2
            min_y = max(0, center_y - half)
            max_y = min(h, center_y + half)
            min_x = max(0, center_x - half)
            max_x = min(w, center_x + half)
            patch = map_gray[min_y:max_y, min_x:max_x].copy()

            # 4.5. Kara Liste (Blacklist) kontrolü
            if getattr(self.routes, "blacklist_manager", None) is not None:
                is_bl, bl_entry = self.routes.blacklist_manager.is_blacklisted(
                    getattr(self.routes, "active_region", "Altgard"),
                    center_x,
                    center_y,
                )
                if is_bl:
                    print(f"[Recorder] ⛔ Bu konum ({center_x}, {center_y}) KARA LİSTEDE! Kayıt engellendi.")
                    try:
                        winsound.Beep(600, 250)
                    except Exception:
                        pass
                    return None

            # 5. Save to database
            node = self.routes.add_node(
                map_x=center_x,
                map_y=center_y,
                patch_img=patch
            )

            if node is None:
                print(f"[Recorder] ⛔ Düğüm eklenemedi (kara liste engeli veya kayıt hatası).")
                try:
                    winsound.Beep(600, 250)
                except Exception:
                    pass
                return None

            print(f"[Recorder] {node['name']} başarıyla kaydedildi! (Konum: {center_x}, {center_y})")
            try:
                winsound.Beep(1800, 180) # High success beep
            except:
                pass

            return node
        finally:
            # 4. Close map immediately — always, even when a step above failed,
            # so an error never leaves the map open on screen.
            self.navigator.close_map()

    def _components_ready(self) -> bool:
        """True when all four collaborators the recorder needs are present
        and each one reports itself ready.

        Mirrors MapNavigator._components_ready(): presence alone is the
        minimum, and where a collaborator exposes its own ready-flag
        method that flag is honoured too (a missing method falls back to
        presence). A collaborator whose probe raises is not ready.
        """
        collaborators = (
            (self.routes, ("get_route_manager_ready_flag",)),
            (self.vision, ("get_vision_engine_ready_flag",)),
            (self.driver, ("get_input_controller_ready_flag",)),
            (self.navigator, ("get_navigator_ready_flag",)),
        )
        for component, flag_methods in collaborators:
            if component is None:
                return False
            for method_name in flag_methods:
                probe = getattr(component, method_name, None)
                if probe is None:
                    continue
                try:
                    flag = probe()
                except Exception:
                    return False
                if isinstance(flag, dict):
                    if not flag.get("ready", False):
                        return False
                elif not flag:
                    return False
        return True

    def get_recorder_ready_flag(self) -> dict:
        return {"module": "recorder", "ready": self._components_ready()}

    def get_recorder_version_info(self) -> dict:
        return {"module": "recorder", "version": __version__, "ready": self._components_ready()}

    def get_recorder_diagnostic_snapshot(self) -> dict:
        ready = self._components_ready()
        return {"module": "recorder", "status": "ok" if ready else "degraded", "ready": ready}


def get_recorder_capabilities() -> list:
    return ["route_recorder", "waypoint_capture", "diagnostic_snapshot"]


def validate_recorder_contract() -> bool:
    """
    Real contract check (not an unconditional True): capabilities are
    present, NodeRecorder exposes the recording entry point and its own
    readiness probe, every collaborator class exposes the methods
    record_current_node() actually calls, and the instance readiness
    probe reports ready with all collaborators present but not ready
    when one is missing.
    """
    try:
        if not get_recorder_capabilities():
            return False
        for name in ("record_current_node", "get_recorder_ready_flag", "_components_ready"):
            if not callable(getattr(NodeRecorder, name, None)):
                return False
        for cls, names in (
            (Vision, ("grab_screen", "find_player_on_map")),
            (RouteManager, ("add_node",)),
            (MapNavigator, ("open_map", "close_map")),
        ):
            for name in names:
                if not callable(getattr(cls, name, None)):
                    return False
        # Behavioural probe of the instance readiness check (no I/O:
        # mock collaborators, never a real driver/vision call).
        from unittest.mock import MagicMock

        ready = NodeRecorder(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        if not ready._components_ready():
            return False
        missing_one = NodeRecorder(MagicMock(), MagicMock(), MagicMock(), None)
        if missing_one._components_ready():
            return False
        return True
    except Exception:
        return False


def get_recorder_baseline_metrics() -> dict:
    contract_valid = validate_recorder_contract()
    return {"module": "recorder", "contract_valid": contract_valid, "recorder_ready": contract_valid}


def get_recorder_readiness_status() -> dict:
    return {"module": "recorder", "ready": validate_recorder_contract(), "metrics": get_recorder_baseline_metrics()}


def get_recorder_audit_record() -> dict:
    return {"module": "recorder", "audit_passed": validate_recorder_contract(), "readiness": get_recorder_readiness_status()}


def get_recorder_operational_digest() -> dict:
    return {"module": "recorder", "operational": validate_recorder_contract(), "audit": get_recorder_audit_record()}


def get_recorder_telemetry_payload() -> dict:
    return {"module": "recorder", "payload": get_recorder_operational_digest()}


def get_recorder_integrity_seal() -> dict:
    return {"module": "recorder", "sealed": validate_recorder_contract(), "telemetry": get_recorder_telemetry_payload()}


def get_recorder_compliance_certificate() -> dict:
    return {"module": "recorder", "certified": validate_recorder_contract(), "seal": get_recorder_integrity_seal()}


def get_recorder_readiness_probe() -> dict:
    return {"module": "recorder", "ready": validate_recorder_contract(), "certificate": get_recorder_compliance_certificate()}


def get_recorder_audit_summary() -> dict:
    return {"module": "recorder", "audit_passed": validate_recorder_contract(), "probe": get_recorder_readiness_probe()}


