# -*- coding: utf-8 -*-
"""
PyQt6 Settings Window (Gatherion Style Split Pane)
Left Pane: Gathering, Route, and Break settings.
Right Pane: Nodes table with mini map patch thumbnails, Region/Route selector, and Map viewer button.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QImage
from core.config import ConfigManager
from core.route_manager import RouteManager
from version import __version__

class SettingsDialog(QDialog):
    def __init__(self, config_manager: ConfigManager, route_manager: RouteManager, parent=None,
                 mobile_server=None, mobile_token_auth=None):
        super().__init__(parent)
        self.cfg = config_manager
        self.routes = route_manager
        # S-0401: live mobile server / shared token authenticator wiring.
        # Both are optional (dialog also works standalone); when present,
        # rotate_mobile_token() propagates the new token to the live
        # server instead of only writing it to disk. Callers that own a
        # MobileRemoteServer (e.g. main.py) should pass it here.
        self.mobile_server = mobile_server
        self.mobile_token_auth = mobile_token_auth
        self.setWindowTitle("Gatherion A2 - Settings & Routes")
        self.setFixedSize(900, 560)
        self.setStyleSheet("""
            QDialog {
                background-color: #12161c;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                border: 1px solid #2d3748;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #aab4c2;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 12px;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #1e242e;
                border: 1px solid #334155;
                border-radius: 4px;
                color: #f8fafc;
                padding: 4px 8px;
                min-height: 28px;
            }
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QTableWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                gridline-color: #21262d;
                color: #c9d1d9;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #0d1117;
                color: #8b949e;
                border: none;
                padding: 6px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.init_ui()
        self.load_settings()
        self.load_nodes_table()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # -------------------------------------------------------------
        # LEFT PANE: Bot Settings
        # -------------------------------------------------------------
        left_pane = QVBoxLayout()
        left_pane.setSpacing(10)

        # Gathering Box
        box_gather = QGroupBox("GATHERING")
        lay_gather = QVBoxLayout(box_gather)

        row_f = QHBoxLayout()
        row_f.addWidget(QLabel("Interval between F presses:"))
        self.spin_f_min = QDoubleSpinBox()
        self.spin_f_min.setRange(0.5, 10.0); self.spin_f_min.setSingleStep(0.1)
        self.spin_f_max = QDoubleSpinBox()
        self.spin_f_max.setRange(0.5, 10.0); self.spin_f_max.setSingleStep(0.1)
        row_f.addWidget(self.spin_f_min)
        row_f.addWidget(QLabel("to"))
        row_f.addWidget(self.spin_f_max)
        row_f.addWidget(QLabel("s"))
        lay_gather.addLayout(row_f)

        self.chk_test = QCheckBox("Test mode (detect only, never press keys)")
        self.chk_overlay = QCheckBox("Show overlay over the game")
        lay_gather.addWidget(self.chk_test)
        lay_gather.addWidget(self.chk_overlay)
        left_pane.addWidget(box_gather)

        # Route Box
        box_route = QGroupBox("ROUTE")
        lay_route = QVBoxLayout(box_route)

        row_rest = QHBoxLayout()
        row_rest.addWidget(QLabel("Rest after a visit:"))
        self.spin_rest = QSpinBox()
        self.spin_rest.setRange(1, 120)
        row_rest.addWidget(self.spin_rest)
        row_rest.addWidget(QLabel("minutes"))
        row_rest.addStretch()
        lay_route.addLayout(row_rest)

        row_reach = QHBoxLayout()
        row_reach.addWidget(QLabel("Time to reach a node:"))
        self.spin_reach = QSpinBox()
        self.spin_reach.setRange(10, 300)
        row_reach.addWidget(self.spin_reach)
        row_reach.addWidget(QLabel("seconds"))
        row_reach.addStretch()
        lay_route.addLayout(row_reach)

        self.chk_combat = QCheckBox("Fight back when attacked (1,2,3/click)")
        lay_route.addWidget(self.chk_combat)
        left_pane.addWidget(box_route)

        # Bottom buttons on left
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_and_close)
        btn_row.addWidget(self.btn_save)
        self.btn_rotate_token = QPushButton("Rotate Mobile Token")
        self.btn_rotate_token.clicked.connect(self.rotate_mobile_token)
        btn_row.addWidget(self.btn_rotate_token)
        left_pane.addLayout(btn_row)
        # Inline validation message (non-modal: a modal popup would block
        # automated runs and hide the dialog behind the game window).
        self.lbl_save_warn = QLabel("")
        self.lbl_save_warn.setStyleSheet("color: #f87171; font-weight: 600;")
        self.lbl_save_warn.setWordWrap(True)
        left_pane.addWidget(self.lbl_save_warn)

        main_layout.addLayout(left_pane, stretch=4)

        # -------------------------------------------------------------
        # RIGHT PANE: Nodes Table & Region/Route Selector
        # -------------------------------------------------------------
        right_pane = QVBoxLayout()
        right_pane.setSpacing(10)
        box_nodes = QGroupBox("NODES")
        lay_nodes = QVBoxLayout(box_nodes)

        # Region/Route Selector Row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Region:"))
        self.cb_region = QComboBox()
        self.cb_region.currentTextChanged.connect(self.on_region_changed)
        sel_row.addWidget(self.cb_region)

        sel_row.addWidget(QLabel("Route:"))
        self.cb_route = QComboBox()
        self.cb_route.currentTextChanged.connect(self.load_nodes_table)
        sel_row.addWidget(self.cb_route)
        lay_nodes.addLayout(sel_row)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setHorizontalHeaderLabels(["PATCH", "NODE", "MAP (X,Y)", "GATHERED", "STATUS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        lay_nodes.addWidget(self.table)

        # Node action buttons
        act_row = QHBoxLayout()
        self.btn_delete_node = QPushButton("Delete Node")
        self.btn_delete_node.setStyleSheet("background-color: #991b1b;")
        self.btn_delete_node.clicked.connect(self.delete_selected_node)
        act_row.addWidget(self.btn_delete_node)
        act_row.addStretch()
        lay_nodes.addLayout(act_row)

        right_pane.addWidget(box_nodes)
        main_layout.addLayout(right_pane, stretch=6)

    def load_settings(self):
        # Canonical schema keys (config/schema.py). The legacy aliases
        # (min_f_interval, rest_after_visit_min, ...) no longer exist in
        # the schema, so reading them here silently returned defaults.
        self.spin_f_min.setValue(self.cfg.get("gathering", "f_interval_min", 1.5))
        self.spin_f_max.setValue(self.cfg.get("gathering", "f_interval_max", 3.0))
        self.chk_test.setChecked(self.cfg.get("gathering", "test_mode", False))
        self.chk_overlay.setChecked(self.cfg.get("gathering", "show_overlay", True))
        self.spin_rest.setValue(self.cfg.get("route", "rest_minutes", 5))
        self.spin_reach.setValue(self.cfg.get("route", "time_to_reach_sec", 60))
        self.chk_combat.setChecked(self.cfg.get("route", "fight_back", True))

        # Populate regions
        self.cb_region.clear()
        for r in self.routes.data.get("regions", {}).keys():
            self.cb_region.addItem(r)

    def on_region_changed(self, region_name):
        self.cb_route.clear()
        region = self.routes.data.get("regions", {}).get(region_name, {})
        for r in region.get("routes", {}).keys():
            self.cb_route.addItem(r)
        self.load_nodes_table()

    def load_nodes_table(self):
        self.table.setRowCount(0)
        region_name = self.cb_region.currentText()
        nodes = self.routes.get_route_nodes(
            region_name=region_name,
            route_name=self.cb_route.currentText()
        )
        for row, n in enumerate(nodes):
            self.table.insertRow(row)

            # Patch thumbnail: resolve from the NODE's own patch_path
            # (which carries its region), never from the active region's
            # patch directory.
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            p_path = None
            try:
                if hasattr(self.routes, "_resolve_patch_disk_path"):
                    p_path = self.routes._resolve_patch_disk_path(n)
            except Exception:
                p_path = None
            if not p_path:
                patch_file = n.get("patch_file")
                if patch_file:
                    try:
                        base = self.routes._get_region_patches_dir(region_name)
                    except Exception:
                        base = self.routes.patch_dir
                    p_path = os.path.join(base, patch_file)
            if p_path and os.path.exists(p_path):
                pix = QPixmap(p_path).scaled(36, 36, Qt.AspectRatioMode.KeepAspectRatio)
                lbl_img.setPixmap(pix)
            self.table.setCellWidget(row, 0, lbl_img)

            # Node Name
            self.table.setItem(row, 1, QTableWidgetItem(n.get("name", f"Node {n['id']}")))
            # Map Coords
            self.table.setItem(row, 2, QTableWidgetItem(f"({n.get('map_x')}, {n.get('map_y')})"))
            # Gathered Count
            self.table.setItem(row, 3, QTableWidgetItem(f"{n.get('visit_count', 0)}x"))
            # Status: real readiness, not a constant "Ready".
            try:
                ready = n.is_ready() if callable(getattr(n, "is_ready", None)) else None
            except Exception:
                ready = None
            if ready is True:
                status = "Ready"
            elif ready is False:
                if n.get("not_gatherable"):
                    status = "Not gatherable"
                elif str(n.get("state", "")).lower() == "resting":
                    status = "Resting"
                else:
                    status = "Not ready"
            else:
                status = str(n.get("state", "active") or "active").capitalize()
            self.table.setItem(row, 4, QTableWidgetItem(status))

    def delete_selected_node(self):
        row = self.table.currentRow()
        if row < 0:
            return
        # The table shows the FILTERED route nodes: resolve the selected
        # row to the node id first, then delete by id. Deleting
        # region["nodes"][row] removed the wrong node whenever a filter
        # was active.
        nodes = self.routes.get_route_nodes(
            region_name=self.cb_region.currentText(),
            route_name=self.cb_route.currentText(),
        )
        if row >= len(nodes):
            return
        node = nodes[row]
        try:
            node_id = int(node.get("id"))
        except (TypeError, ValueError):
            return
        region_name = node.get("region") or self.cb_region.currentText()
        if hasattr(self.routes, "delete_node"):
            self.routes.delete_node(node_id, region_name)
        self.load_nodes_table()

    def _resolve_mobile_server(self):
        """Return the live MobileRemoteServer, if one is reachable.

        Priority: explicitly wired server, then the parent-widget chain
        (a main window owns ``mobile_server``). Returns None when no
        live server is known — rotation then only affects the on-disk
        token, which the server picks up at its next start/bind.
        """
        server = getattr(self, "mobile_server", None)
        if server is not None:
            return server
        widget = self.parentWidget()
        seen = set()
        while widget is not None and id(widget) not in seen:
            seen.add(id(widget))
            server = getattr(widget, "mobile_server", None)
            if server is not None:
                return server
            widget = widget.parentWidget() if hasattr(widget, "parentWidget") else None
        return None

    def _resolve_mobile_token_auth(self):
        """Return the shared MobileTokenAuthenticator, if one is reachable.

        S-4301: _resolve_mobile_server walked the parent chain for the
        server but there was no equivalent discovery for the auth, so a
        dialog instantiated without an explicit ``mobile_token_auth``
        always minted a fresh authenticator and the shared instance
        (main.py's, bound to the server at :583 and shared with the
        worker at :541) kept holding the old token. Priority: explicitly
        wired auth, then the parent-widget chain (a main window owns
        ``mobile_token_auth``). Returns None when none is known.
        """
        auth = getattr(self, "mobile_token_auth", None)
        if auth is not None and hasattr(auth, "rotate_token"):
            return auth
        widget = self.parentWidget()
        seen = set()
        while widget is not None and id(widget) not in seen:
            seen.add(id(widget))
            auth = getattr(widget, "mobile_token_auth", None)
            if auth is not None and hasattr(auth, "rotate_token"):
                return auth
            widget = widget.parentWidget() if hasattr(widget, "parentWidget") else None
        return None

    def _propagate_token_to_mobile_server(self, auth) -> bool:
        """Push ``auth``'s (freshly rotated) token onto the live server.

        Uses the authenticator's own ``bind_to_mobile_server()`` — the
        same call main.py makes once at startup — and verifies the
        server actually holds the new token afterwards. Deliberately
        does NOT call ``server.rotate_access_token()``: that generates
        a *different* 32-char token and would desync the server from
        the 256-bit authenticator. Returns False when no live server
        is known or the push could not be verified.
        """
        server = self._resolve_mobile_server()
        if server is None:
            return False
        try:
            bound = False
            if hasattr(auth, "bind_to_mobile_server"):
                try:
                    bound = bool(auth.bind_to_mobile_server(server))
                except Exception:
                    bound = False
            if not bound and hasattr(server, "access_token"):
                # Fallback: direct assignment of the rotated token.
                server.access_token = auth.token
                if hasattr(server, "require_token"):
                    server.require_token = True
            return bool(
                getattr(server, "access_token", None) is not None
                and str(getattr(server, "access_token")) == str(auth.token)
            )
        except Exception:
            return False

    def rotate_mobile_token(self):
        """Generate a fresh 256-bit mobile token (core/incident_recorder.py).

        S-0401: rotating only the on-disk token left the live
        MobileRemoteServer accepting the OLD token (its access_token was
        set once at startup via bind_to_mobile_server()). The new token
        is therefore also pushed to the live server — via the shared
        authenticator when one was wired in, so its in-memory token
        stays in sync too — and the push is verified. When no live
        server is reachable the user is told the new token only takes
        effect after the server restarts, instead of being left to
        believe the old token is already dead.
        Returns True when the token was rotated (and, if a live server
        was known, verified on it); False on failure.
        """
        try:
            from core.incident_recorder import MobileTokenAuthenticator

            # Prefer the shared authenticator (main.py's instance) so its
            # in-memory token does not keep the old value after rotation.
            # S-4301: discovery now also walks the parent-widget chain
            # (_resolve_mobile_token_auth), mirroring the server lookup.
            auth = self._resolve_mobile_token_auth()
            if auth is None or not hasattr(auth, "rotate_token"):
                auth = MobileTokenAuthenticator()
            auth.rotate_token()
            server_updated = self._propagate_token_to_mobile_server(auth)
            server_known = self._resolve_mobile_server() is not None
            if server_known and server_updated:
                QMessageBox.information(
                    self,
                    "Mobile Token",
                    "Yeni 256-bit mobil token üretildi ve canlı mobil "
                    "sunucuya uygulandı.\n"
                    f"Parmak izi: {auth.token_fingerprint}\n"
                    "Eski token artık geçersiz. QR kodunu yeniden oluşturun.",
                )
                return True
            if server_known:
                QMessageBox.warning(
                    self,
                    "Mobile Token",
                    "Yeni 256-bit mobil token üretildi (diske yazıldı),\n"
                    f"Parmak izi: {auth.token_fingerprint}\n"
                    "AMA canlı mobil sunucuya uygulanamadı — sunucu hâlâ "
                    "eski tokenı kabul ediyor olabilir. Sunucuyu yeniden "
                    "başlatın ve QR kodunu yeniden oluşturun.",
                )
                return False
            QMessageBox.information(
                self,
                "Mobile Token",
                "Yeni 256-bit mobil token üretildi.\n"
                f"Parmak izi: {auth.token_fingerprint}\n"
                "Canlı mobil sunucu bulunamadı; yeni token sunucu yeniden "
                "başlatıldığında geçerli olur. QR kodunu yeniden oluşturun.",
            )
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Mobile Token", f"Token döndürülemedi: {exc}")
            return False

    def save_and_close(self):
        # Validate the interval before saving: min must not exceed max.
        if self.spin_f_min.value() > self.spin_f_max.value():
            self.lbl_save_warn.setText("F aralığı hatalı: minimum değer maksimumdan büyük olamaz.")
            return
        self.lbl_save_warn.setText("")
        self.cfg.set("gathering", "f_interval_min", self.spin_f_min.value())
        self.cfg.set("gathering", "f_interval_max", self.spin_f_max.value())
        self.cfg.set("gathering", "test_mode", self.chk_test.isChecked())
        self.cfg.set("gathering", "show_overlay", self.chk_overlay.isChecked())
        self.cfg.set("route", "rest_minutes", self.spin_rest.value())
        self.cfg.set("route", "time_to_reach_sec", self.spin_reach.value())
        self.cfg.set("route", "fight_back", self.chk_combat.isChecked())
        self.accept()

    def get_settings_dialog_health_summary(self) -> dict:
        # S-0405 (Karar 3: proba bağlama, silme yok): eskiden sabit
        # {"ready": True} dönerdi. Şimdi gerçek kontroller çalışır:
        # dosyanın sözleşme zinciri + config okunabilirliği + rota
        # verisi + widget'ların tutarlı durumu. ready, kontrollerin
        # tamamı geçerse True'dur.
        checks: dict = {}
        try:
            checks["contract_valid"] = bool(validate_settings_dialog_contract())
        except Exception:
            checks["contract_valid"] = False
        try:
            checks["config_readable"] = (
                self.cfg is not None
                and self.cfg.get("gathering", "f_interval_min", None) is not None
            )
        except Exception:
            checks["config_readable"] = False
        try:
            routes_data = getattr(self.routes, "data", None)
            checks["routes_loaded"] = bool(
                routes_data and isinstance(routes_data, dict)
                and routes_data.get("regions")
            )
        except Exception:
            checks["routes_loaded"] = False
        try:
            checks["widgets_ready"] = bool(
                hasattr(self, "spin_f_min")
                and hasattr(self, "spin_f_max")
                and hasattr(self, "table")
                and self.spin_f_min.value() <= self.spin_f_max.value()
            )
        except Exception:
            checks["widgets_ready"] = False
        return {
            "module": "settings_dialog",
            "ready": bool(checks) and all(checks.values()),
            "checks": checks,
        }

    def get_settings_dialog_summary(self) -> dict:
        # S-4302: was a constant {"settings_dialog_ready": True}. Now
        # delegates to the real health probe (health_summary); any probe
        # failure reports not-ready.
        try:
            ready = bool(self.get_settings_dialog_health_summary().get("ready", False))
        except Exception:
            ready = False
        return {"settings_dialog_ready": ready}

    def get_settings_dialog_ready_flag(self) -> dict:
        # S-4302: "ready" now reflects the real health probe.
        try:
            ready = bool(self.get_settings_dialog_health_summary().get("ready", False))
        except Exception:
            ready = False
        return {"module": "settings_dialog", "ready": ready}

    def get_settings_dialog_version_info(self) -> dict:
        # S-4302: "ready" now reflects the real health probe.
        try:
            ready = bool(self.get_settings_dialog_health_summary().get("ready", False))
        except Exception:
            ready = False
        return {"module": "settings_dialog", "version": __version__, "ready": ready}

    def get_settings_dialog_diagnostic_snapshot(self) -> dict:
        # S-4302: "status"/"ready" now reflect the real health probe.
        try:
            ready = bool(self.get_settings_dialog_health_summary().get("ready", False))
        except Exception:
            ready = False
        return {
            "module": "settings_dialog",
            "status": "ok" if ready else "degraded",
            "ready": ready,
        }


def get_settings_dialog_capabilities() -> list:
    return ["settings_dialog", "preferences_editor", "diagnostic_snapshot"]


def validate_settings_dialog_contract() -> bool:
    # S-4302: was bool(get_settings_dialog_capabilities()) — the
    # capabilities list is a constant 3-element list, so this was
    # always True, and the whole baseline/readiness/audit/telemetry/
    # integrity/compliance/probe/summary chain derived from it always
    # reported pass. Now a real probe: every expected capability must
    # be present, the dialog class must actually carry its load-bearing
    # methods, and the config schema contract (the settings subsystem
    # this dialog edits) must hold. Any failure makes it False.
    try:
        expected = ("settings_dialog", "preferences_editor", "diagnostic_snapshot")
        caps = get_settings_dialog_capabilities()
        if not all(c in caps for c in expected):
            return False
        if not all(
            hasattr(SettingsDialog, name)
            for name in (
                "rotate_mobile_token",
                "save_and_close",
                "get_settings_dialog_health_summary",
            )
        ):
            return False
        from config.schema import validate_config_schema_contract

        return bool(validate_config_schema_contract())
    except Exception:
        return False


def get_settings_dialog_baseline_metrics() -> dict:
    # S-4302: "dialog_ready" was a hardcoded True; it now reflects the
    # real contract probe above.
    ready = validate_settings_dialog_contract()
    return {
        "module": "settings_dialog",
        "contract_valid": ready,
        "dialog_ready": ready,
    }


def get_settings_dialog_readiness_status() -> dict:
    try:
        return {
            "module": "settings_dialog",
            "ready": validate_settings_dialog_contract(),
            "metrics": get_settings_dialog_baseline_metrics(),
        }
    except Exception:
        return {
            "module": "settings_dialog",
            "ready": False,
            "metrics": {},
        }


def get_settings_dialog_audit_record() -> dict:
    return {
        "module": "settings_dialog",
        "audit_passed": validate_settings_dialog_contract(),
        "readiness": get_settings_dialog_readiness_status(),
    }


def get_settings_dialog_operational_digest() -> dict:
    return {
        "module": "settings_dialog",
        "operational": validate_settings_dialog_contract(),
        "audit": get_settings_dialog_audit_record(),
    }


def get_settings_dialog_telemetry_payload() -> dict:
    return {
        "module": "settings_dialog",
        "payload": get_settings_dialog_operational_digest(),
    }


def get_settings_dialog_integrity_seal() -> dict:
    return {
        "module": "settings_dialog",
        "sealed": validate_settings_dialog_contract(),
        "telemetry": get_settings_dialog_telemetry_payload(),
    }


def get_settings_dialog_compliance_certificate() -> dict:
    return {
        "module": "settings_dialog",
        "certified": validate_settings_dialog_contract(),
        "seal": get_settings_dialog_integrity_seal(),
    }


def get_settings_dialog_readiness_probe() -> dict:
    return {
        "module": "settings_dialog",
        "ready": validate_settings_dialog_contract(),
        "certificate": get_settings_dialog_compliance_certificate(),
    }


def get_settings_dialog_audit_summary() -> dict:
    return {
        "module": "settings_dialog",
        "audit_passed": validate_settings_dialog_contract(),
        "probe": get_settings_dialog_readiness_probe(),
    }


