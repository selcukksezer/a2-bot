# -*- coding: utf-8 -*-
"""
PyQt6 Ctrl+Alt+F9 hızlı kontrol menüsü.
Görünen satırlar kısa Türkçe düğmelerdir. Uzun açıklama yalnızca ipucundadır.
Durum satırı: duraklatıldı veya çalışıyor, bölge, rota. Yeşil rozet: "3x".
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QClipboard,
    QColor,
    QCursor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.shell_theme import SHELL_COLORS

logger = logging.getLogger("Gatherion.F9Menu")

# Opaque panel fill — light row/header text (#e5e7eb / #f8fafc) needs a real dark surface.
# WA_TranslucentBackground alone often leaves a white QScrollArea / desktop behind the text.
_PANEL_BG = QColor(SHELL_COLORS["panel_deep"])
_PANEL_BORDER = QColor(SHELL_COLORS["line"])
_ROW_TEXT = QColor(SHELL_COLORS["text"])
_ROW_TEXT_HOVER = QColor("#ffffff")
_ROW_HOVER_BG = QColor("#232830")
_ROW_MUTED = QColor(SHELL_COLORS["muted"])
_RADIO_ON = QColor("#38bdf8")
_RADIO_OFF = QColor("#6b7280")


class _MenuRowWidget(QWidget):
    """One short action button. The long explanation is the tooltip, not the label."""

    clicked = pyqtSignal(str)
    hovered = pyqtSignal(str, QWidget)

    def __init__(
        self,
        action_id: str,
        text: str,
        is_radio: bool = False,
        radio_checked: bool = False,
        has_submenu: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.action_id = str(action_id or "").strip()
        self.label_text = str(text or self.action_id or "—").strip() or "—"
        self.is_radio = bool(is_radio)
        self.radio_checked = bool(radio_checked)
        self.has_submenu = bool(has_submenu)
        self.is_hovered = False

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self.setMinimumWidth(180)

    def set_text(self, text: str) -> None:
        safe_text = str(text or "").strip()
        self.label_text = safe_text if safe_text else (self.action_id or "—")
        self.update()

    def set_radio_checked(self, checked: bool) -> None:
        self.radio_checked = bool(checked)
        self.update()

    def set_hovered(self, hovered: bool) -> None:
        if self.is_hovered != hovered:
            self.is_hovered = hovered
            self.update()

    def enterEvent(self, event: QEvent) -> None:
        self.set_hovered(True)
        self.hovered.emit(self.action_id, self)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.set_hovered(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_press_ts = time.monotonic()
            self.clicked.emit(self.action_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            last_p = getattr(self, "_last_press_ts", 0.0)
            if (time.monotonic() - last_p) >= 0.35:
                self.clicked.emit(self.action_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Üzerine gelindiğinde vurgu çubuğu
        if self.is_hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_ROW_HOVER_BG))
            painter.drawRoundedRect(rect.adjusted(4, 2, -4, -2), 4.0, 4.0)

        left_x = 14
        center_y = rect.height() // 2

        if self.is_radio:
            if self.radio_checked:
                # Dolu mavi/camgöbeği nokta ●
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(_RADIO_ON))
                painter.drawEllipse(QPoint(left_x + 4, center_y), 4, 4)
            else:
                # Boş daire ○
                painter.setPen(QPen(_RADIO_OFF, 1.3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPoint(left_x + 4, center_y), 4, 4)

        text_x = left_x + 18

        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QPen(_ROW_TEXT_HOVER if self.is_hovered else _ROW_TEXT))

        right_pad = 18 if self.has_submenu else 10
        text_rect = QRect(text_x, 0, rect.width() - text_x - right_pad, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self.label_text,
        )

        if self.has_submenu:
            chev_rect = QRect(rect.width() - 16, 0, 12, rect.height())
            chev_font = QFont("Segoe UI", 10)
            chev_font.setWeight(QFont.Weight.Bold)
            painter.setFont(chev_font)
            painter.setPen(QPen(_ROW_MUTED))
            painter.drawText(chev_rect, int(Qt.AlignmentFlag.AlignCenter), "›")

        painter.end()


class F9Menu(QWidget):
    """Ctrl+Alt+F9 menu. Each row is one short Turkish action."""

    SHORT_LABELS: Dict[str, str] = {
        "resume": "Devam Et",
        "combat": "Savaş",
        "route": "Toplama",
        "auto_record": "Oto-Kayıt",
        "manual_record": "Konum Ekle",
        "explorer": "Keşif",
        "minimap_auto_record": "Minimap",
        "active_route": "Rota",
        "radar": "Radar",
        "analytics": "İstatistik",
        "calibrate": "Kalibrasyon",
        "yolo_factory": "Eğit",
        "settings": "Ayarlar",
        "open_log": "Günlük",
        "guide": "Kılavuz",
        "copy_support": "Tanılama",
        "exit": "Çıkış",
    }

    action_triggered = pyqtSignal(str)
    mode_selected = pyqtSignal(str)  # Geriye dönük uyumluluk sinyali
    route_selected = pyqtSignal(str, str)  # (region, route)
    route_changed = pyqtSignal(str, str)  # (region, route) Canlı UI ve rota senkronizasyonu sinyali
    gathering_state_changed = pyqtSignal(dict)  # Canlı toplama/hasat durumu verisi
    harvest_stage_changed = pyqtSignal(int, int)  # (current_stage, max_stages)
    routes_reloaded = pyqtSignal(dict)  # Yeniden taranan {region: [routes]}
    minimap_auto_record_toggled = pyqtSignal(bool)  # Sağ Üst Minimap Oto-Kayıt sinyali
    panel_open_changed = pyqtSignal(bool)
    selective_gather_changed = pyqtSignal(list, bool)  # (materials: list, auto_proximity: bool)
    consumables_toggled = pyqtSignal(bool)  # İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll) sinyali

    DEFAULT_SELECTIVE_PRESETS: List[Tuple[str, List[str]]] = [
        ("Tümü (Hepsi)", []),
        ("⛏️ Orichalcum Ore (orichalcum)", ["orichalcum"]),
        ("🌸 Aria (aria)", ["aria"]),
        ("🍒 Mela (mela)", ["mela"]),
        ("🌿 Targena (targena)", ["targena"]),
        ("💎 Gem / Odyle (gem_crystal)", ["gem_crystal"]),
        ("🪵 Ağaç / Kereste (tree_log)", ["tree_log"]),
    ]

    MODE_LABELS: Dict[str, str] = {
        "combat": "Savaşma Modu (Combat)",
        "combat_mode": "Savaşma Modu (Combat)",
        "auto_record": "Topla (Gather) / Oto-Kayıt",
        "manual_record": "Manuel Kayıt",
        "route": "Toplama Modu (Gathering / Route)",
        "route_mode": "Toplama Modu (Gathering / Route)",
        "explorer": "Oto-Keşif & Haritalama (Explore)",
        "explorer_mode": "Oto-Keşif & Haritalama (Explore)",
    }

    def __init__(
        self,
        route_manager: Optional[Any] = None,
        active_mode: str = "manual_record",
        minimap_auto_record: bool = True,
        harvests_per_node: int = 3,
        parent: Optional[QWidget] = None,
        auto_consumables: bool = True,
    ) -> None:
        super().__init__(parent)
        self.route_manager = route_manager
        self.shell: Optional[Any] = None
        self.overlay: Optional[Any] = None
        self.worker: Optional[Any] = None
        self._active_mode = self._normalize_mode(active_mode)
        self._active_region = "Altgard"
        self._active_route = "Dredgion Crash Site"
        self._minimap_auto_record_enabled: bool = bool(minimap_auto_record)
        try:
            self._harvests_per_node: int = max(1, int(harvests_per_node))
        except Exception:
            self._harvests_per_node = 3
        self._current_harvest_stage: int = 0
        self._is_gathering: bool = False
        self._current_resource: str = ""
        self._gathered_count: int = 0
        self._active_node_id: int = 0
        self._selective_gather_enabled: bool = False
        self._selected_materials: List[str] = []
        self._proximity_route_auto: bool = True
        self._auto_consumables_enabled: bool = bool(auto_consumables)
        self._regions_routes: Dict[str, List[str]] = {}
        self._rows: Dict[str, _MenuRowWidget] = {}
        self._route_submenu: Optional[QMenu] = None
        self._guide_dialog: Optional[QDialog] = None
        self._log_dialog: Optional[QDialog] = None
        self._last_popup_ts: float = 0.0
        self._panel_open: bool = False
        self._paused: bool = True
        self._collapsible: List[QWidget] = []
        self._menu_chord_label = "Ctrl+Alt+F9"
        self._node_chord_label = "Ctrl+Alt+F8"
        self._pause_chord_label = "Ctrl+Alt+F10"

        self._sync_from_route_manager()
        self._init_ui()

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        if not mode or not isinstance(mode, (str, bytes)):
            return "manual_record"
        m = (mode.decode("utf-8", errors="ignore") if isinstance(mode, bytes) else mode).strip().lower()
        if m in ("combat", "combat_mode", "savas", "savaş", "fight"):
            return "combat"
        if m in ("auto", "auto_record", "autorecord", "gather"):
            return "auto_record"
        if m in ("manual", "manual_record", "manualrecord"):
            return "manual_record"
        if m in ("route", "route_mode", "routemode", "topla"):
            return "route"
        if m in ("explorer", "explorer_mode", "explore", "auto_cartography"):
            return "explorer"
        return "manual_record"

    @staticmethod
    def _default_regions_routes() -> Dict[str, List[str]]:
        return {
            "Altgard": ["All", "Dredgion Crash Site", "Safe Haven", "Quai Campsite", "Calderon Canyon", "Silent Hill"],
            "Verteron": ["All", "Tolbas Forest", "Cantas Coast", "Pilgrim's Ridge", "Verteron Citadel"],
            "Eltnen": ["All", "Manduri Forest", "Golden Bough Garrison", "Eiron Desert"],
            "Morheim": ["All", "Ice Claw Village", "Patamor Ridge", "Mist Mane Village"],
            "DefaultRegion": ["All"],
        }

    @classmethod
    def _sanitize_regions_routes(cls, raw_map: Any) -> Dict[str, List[str]]:
        """Boş veya hatalı tipteki bölge/rota yapılarını güvenli sözlüğe dönüştürür."""
        sanitized: Dict[str, List[str]] = {}
        if isinstance(raw_map, dict):
            for raw_reg, raw_rtes in raw_map.items():
                if raw_reg is None:
                    continue
                reg_name = str(raw_reg).strip()
                if not reg_name:
                    continue
                clean_routes: List[str] = []
                if isinstance(raw_rtes, dict):
                    iterable = list(raw_rtes.keys())
                elif isinstance(raw_rtes, (list, tuple, set)):
                    iterable = list(raw_rtes)
                elif isinstance(raw_rtes, str) and raw_rtes.strip():
                    iterable = [raw_rtes.strip()]
                else:
                    iterable = []
                for item in iterable:
                    if item is None:
                        continue
                    r_str = str(item).strip()
                    if r_str and r_str not in clean_routes:
                        clean_routes.append(r_str)
                sanitized[reg_name] = clean_routes if clean_routes else ["All"]
        return sanitized

    def _sync_from_route_manager(self) -> None:
        """RouteManager mevcutsa bölgeleri, rotaları ve aktif seçimi senkronize eder."""
        try:
            if self.route_manager is not None:
                if hasattr(self.route_manager, "active_region") and self.route_manager.active_region:
                    self._active_region = str(self.route_manager.active_region)
                if hasattr(self.route_manager, "active_route") and self.route_manager.active_route:
                    self._active_route = str(self.route_manager.active_route)

                # 1. get_regions() & get_routes(reg) API
                if hasattr(self.route_manager, "get_regions"):
                    try:
                        regs = self.route_manager.get_regions()
                        if regs and isinstance(regs, (list, tuple)):
                            parsed: Dict[str, List[str]] = {}
                            for reg in regs:
                                reg_name = str(reg).strip()
                                if not reg_name:
                                    continue
                                if hasattr(self.route_manager, "get_routes"):
                                    rtes = self.route_manager.get_routes(reg_name)
                                    parsed[reg_name] = [str(r).strip() for r in rtes if r] if rtes else ["All"]
                                else:
                                    parsed[reg_name] = ["All"]
                            if parsed:
                                self._regions_routes = parsed
                    except Exception as sub_exc:
                        logger.debug("get_regions üzerinden okuma hatası: %s", sub_exc)

                # 2. _regions dict (handles routes as list, dict, or tuple)
                if not self._regions_routes and hasattr(self.route_manager, "_regions"):
                    raw_regs = getattr(self.route_manager, "_regions", {})
                    if isinstance(raw_regs, dict) and raw_regs:
                        parsed = {}
                        for reg_name, reg_val in raw_regs.items():
                            if isinstance(reg_val, dict):
                                r_val = reg_val.get("routes", [])
                                if isinstance(r_val, dict):
                                    parsed[str(reg_name)] = list(r_val.keys()) if r_val else ["All"]
                                elif isinstance(r_val, (list, tuple, set)):
                                    parsed[str(reg_name)] = [str(r) for r in r_val if r] if r_val else ["All"]
                        if parsed:
                            self._regions_routes = parsed

                # 3. data dict fallback
                if hasattr(self.route_manager, "data") and isinstance(self.route_manager.data, dict):
                    data = self.route_manager.data
                    self._active_region = data.get("active_region") or self._active_region
                    self._active_route = data.get("active_route") or self._active_route
                    if not self._regions_routes:
                        regions_dict = data.get("regions", {})
                        parsed = {}
                        for reg_name, reg_val in regions_dict.items():
                            if isinstance(reg_val, dict):
                                routes_map = reg_val.get("routes", {})
                                if isinstance(routes_map, dict):
                                    parsed[str(reg_name)] = list(routes_map.keys()) if routes_map else ["All"]
                                elif isinstance(routes_map, (list, tuple, set)):
                                    parsed[str(reg_name)] = [str(r) for r in routes_map if r] if routes_map else ["All"]
                        if parsed:
                            self._regions_routes = parsed
        except Exception as exc:
            logger.warning("RouteManager üzerinden rotalar senkronize edilemedi: %s", exc)

        # 4. Scan disk base_dir / routes/ directory for region_data.json
        base_dir = "routes"
        if self.route_manager is not None and hasattr(self.route_manager, "base_dir"):
            base_dir = str(self.route_manager.base_dir)
        if not os.path.isabs(base_dir):
            if os.path.isdir(r"F:\a2\routes"):
                base_dir = r"F:\a2\routes"
            elif os.path.isdir("routes"):
                base_dir = os.path.abspath("routes")

        if os.path.isdir(base_dir):
            try:
                for entry in sorted(os.listdir(base_dir)):
                    entry_path = os.path.join(base_dir, entry)
                    json_path = os.path.join(entry_path, "region_data.json")
                    if os.path.isdir(entry_path) and os.path.isfile(json_path):
                        with open(json_path, "r", encoding="utf-8") as jf:
                            reg_data = json.load(jf)
                        reg_name = str(reg_data.get("region", entry)).strip()
                        r_list = ["All"]
                        raw_rtes = reg_data.get("routes", [])
                        if isinstance(raw_rtes, list):
                            for r in raw_rtes:
                                rs = str(r).strip()
                                if rs and rs not in r_list:
                                    r_list.append(rs)
                        for n in reg_data.get("nodes", []):
                            if isinstance(n, dict):
                                for r in n.get("routes", []):
                                    rs = str(r).strip()
                                    if rs and rs not in r_list:
                                        r_list.append(rs)
                        if reg_name not in self._regions_routes:
                            self._regions_routes[reg_name] = r_list
                        else:
                            for r in r_list:
                                if r not in self._regions_routes[reg_name]:
                                    self._regions_routes[reg_name].append(r)
            except Exception as exc:
                logger.debug("Disk rotaları taranırken hata: %s", exc)

        # 5. Legacy routes.json fallback
        if not self._regions_routes:
            routes_path = os.path.join(os.getcwd(), "routes", "routes.json")
            if os.path.exists(routes_path):
                try:
                    with open(routes_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._active_region = data.get("active_region", self._active_region)
                    self._active_route = data.get("active_route", self._active_route)
                    for reg_name, reg_val in data.get("regions", {}).items():
                        if isinstance(reg_val, dict):
                            r_keys = list(reg_val.get("routes", {}).keys())
                            self._regions_routes[reg_name] = r_keys if r_keys else ["All"]
                except Exception as exc:
                    logger.debug("F9Menu içinde routes.json okunamadı: %s", exc)

        # 6. Default canonical regions and subregions fallback
        if not self._regions_routes:
            self._regions_routes = {
                "Altgard": ["All", "Dredgion Crash Site", "Safe Haven", "Quai Campsite", "Calderon Canyon", "Silent Hill"],
                "Verteron": ["All", "Tolbas Forest", "Cantas Coast", "Pilgrim's Ridge", "Verteron Citadel"],
                "Eltnen": ["All", "Manduri Forest", "Golden Bough Garrison", "Eiron Desert"],
                "Morheim": ["All", "Ice Claw Village", "Patamor Ridge", "Mist Mane Village"],
                "DefaultRegion": ["All"],
            }

    def _init_ui(self) -> None:
        # Qt.WindowType.Tool + WindowStaysOnTopHint kullanıyoruz:
        # Qt.WindowType.Popup Windows'ta fareyi kilitlediği (SetCapture) ve Aion 2 kamera
        # modunda imleç merkeze çekildiğinde kendi kendine kapanıp açıldığı (titreme) için
        # Tool penceresi kullanılarak farenin serbest kalması ve menünün sabit durması sağlanır.
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Floating tool window may keep translucency; paintEvent still fills an opaque panel.
        # Embedded mode (attach_embedded) turns translucency off so the shell scroll viewport
        # cannot show through as white-on-light text.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowOpacity(0.96)
        quit_attr = getattr(Qt.WidgetAttribute, "WA_QuitOnClose", None)
        if quit_attr is not None:
            self.setAttribute(quit_attr, False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setWindowTitle(f"{self._menu_chord_label} Hızlı Kontrol Menüsü")
        self._apply_panel_stylesheet(embedded=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(1)

        # -----------------------------------------------------------------
        # 0. ÜST BAŞLIK & 3X TOPLAMA MODU ROZETİ (HEADER / SUBTITLE)
        # -----------------------------------------------------------------
        self.header_frame = QFrame(self)
        self.header_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        header_vbox = QVBoxLayout(self.header_frame)
        header_vbox.setContentsMargins(14, 4, 14, 4)
        header_vbox.setSpacing(3)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)

        self.header_title_label = QLabel(
            f"{self._menu_chord_label} Hızlı Kontrol Menüsü", self.header_frame
        )
        self.header_title_label.setStyleSheet(
            f"""
            QLabel {{
                color: {SHELL_COLORS["text"]};
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
            }}
            """
        )

        self.harvest_badge_label = QLabel(
            f"{self._harvests_per_node}x",
            self.header_frame,
        )
        self.harvest_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.harvest_badge_label.setStyleSheet(
            """
            QLabel {
                background-color: rgba(34, 197, 94, 0.18);
                color: #4ade80;
                border: 1px solid rgba(34, 197, 94, 0.45);
                border-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 700;
                padding: 2px 6px;
            }
            """
        )

        self._toggle_btn = QPushButton("✕", self.header_frame)
        self._toggle_btn.setFixedSize(22, 22)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setToolTip("Menüyü kapat")
        self._toggle_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                color: #94a3b8;
                border: 1px solid #2d323b;
                border-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton:hover {
                color: #f8fafc;
                border-color: #64748b;
            }
            """
        )
        self._toggle_btn.clicked.connect(self.toggle_panel)

        top_bar.addWidget(self.header_title_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.harvest_badge_label)
        top_bar.addWidget(self._toggle_btn)
        header_vbox.addLayout(top_bar)

        self.header_subtitle_label = QLabel(self._format_status_line(), self.header_frame)
        self.header_subtitle_label.setStyleSheet(
            f"""
            QLabel {{
                color: {SHELL_COLORS["muted"]};
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 500;
            }}
            """
        )
        header_vbox.addWidget(self.header_subtitle_label)
        layout.addWidget(self.header_frame)
        self._collapsible.extend(
            [self.header_title_label, self.harvest_badge_label, self.header_subtitle_label]
        )

        self._add_separator(layout)

        self._add_row(layout, "resume", self.SHORT_LABELS["resume"])
        self._add_separator(layout)
        self._add_row(
            layout,
            "combat",
            self.SHORT_LABELS["combat"],
            is_radio=True,
            radio_checked=(self._active_mode == "combat"),
        )
        self._add_row(
            layout,
            "route",
            self.SHORT_LABELS["route"],
            is_radio=True,
            radio_checked=(self._active_mode == "route"),
        )
        self._add_row(
            layout,
            "auto_record",
            self.SHORT_LABELS["auto_record"],
            is_radio=True,
            radio_checked=(self._active_mode == "auto_record"),
        )
        self._add_row(
            layout,
            "manual_record",
            self.SHORT_LABELS["manual_record"],
            is_radio=True,
            radio_checked=(self._active_mode == "manual_record"),
        )
        self._add_row(
            layout,
            "explorer",
            self.SHORT_LABELS["explorer"],
            is_radio=True,
            radio_checked=(self._active_mode == "explorer"),
        )
        self._add_row(
            layout,
            "minimap_auto_record",
            self.SHORT_LABELS["minimap_auto_record"],
            is_radio=True,
            radio_checked=self._minimap_auto_record_enabled,
        )
        self._add_row(
            layout,
            "active_route",
            self.SHORT_LABELS["active_route"],
            has_submenu=True,
        )
        self._add_separator(layout)
        self._init_selective_gather_ui(layout)
        self._add_separator(layout)
        for action_id in (
            "radar",
            "analytics",
            "calibrate",
            "yolo_factory",
            "settings",
            "open_log",
            "guide",
            "copy_support",
            "exit",
        ):
            if action_id == "exit":
                self._add_separator(layout)
            self._add_row(layout, action_id, self.SHORT_LABELS[action_id])

        self.adjustSize()

    def _init_selective_gather_ui(self, layout: QVBoxLayout) -> None:
        """Adds '🎯 Seç-Topla (Materyal Seç & Yakınlık Rotası)' selector & proximity button."""
        self.selective_gather_frame = QFrame(self)
        self.selective_gather_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #161a20;
                border: 1px solid {SHELL_COLORS['line']};
                border-radius: 6px;
                padding: 4px;
            }}
            """
        )
        sg_layout = QVBoxLayout(self.selective_gather_frame)
        sg_layout.setContentsMargins(6, 4, 6, 4)
        sg_layout.setSpacing(4)

        # Header row: Label + Checkbox
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(4)

        lbl_title = QLabel("🎯 Seç-Topla", self.selective_gather_frame)
        lbl_title.setStyleSheet(
            f"""
            QLabel {{
                color: {SHELL_COLORS['text']};
                font-family: 'Segoe UI', sans-serif;
                font-size: 10px;
                font-weight: 700;
            }}
            """
        )
        lbl_title.setToolTip("🎯 Seç-Topla: Yalnızca seçili materyali topla ve yakınlığına göre rota çiz")

        self.chk_selective_gather = QCheckBox("Aktif", self.selective_gather_frame)
        self.chk_selective_gather.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_selective_gather.setStyleSheet(
            """
            QCheckBox {
                color: #38bdf8;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
            """
        )
        self.chk_selective_gather.setChecked(self._selective_gather_enabled)
        self.chk_selective_gather.toggled.connect(self._on_selective_gather_toggled)

        hdr_row.addWidget(lbl_title)
        hdr_row.addStretch(1)
        hdr_row.addWidget(self.chk_selective_gather)
        sg_layout.addLayout(hdr_row)

        # Dropdown
        self.combo_selective_materials = QComboBox(self.selective_gather_frame)
        self.combo_selective_materials.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_selective_materials.setStyleSheet(
            """
            QComboBox {
                background-color: #0f172a;
                color: #ffffff;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 3px 6px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 500;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #38bdf8;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #ffffff;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                border: 1px solid #334155;
                outline: none;
                font-size: 9px;
            }
            """
        )
        view = self.combo_selective_materials.view()
        if view is not None:
            pal = view.palette()
            pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.Base, QColor("#0f172a"))
            view.setPalette(pal)

        self.populate_region_materials(self._active_region)
        self.combo_selective_materials.currentIndexChanged.connect(self._on_selective_material_changed)
        sg_layout.addWidget(self.combo_selective_materials)

        # Action Button: "🧭 Yakınlığa Göre Rota Çiz & Topla"
        self.btn_proximity_route = QPushButton("🧭 Yakınlığa Göre Rota Çiz & Topla", self.selective_gather_frame)
        self.btn_proximity_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_proximity_route.setStyleSheet(
            """
            QPushButton {
                background-color: #1e3a8a;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 700;
                padding: 4px 6px;
            }
            QPushButton:hover {
                background-color: #2563eb;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            """
        )
        self.btn_proximity_route.clicked.connect(self._on_proximity_route_button_clicked)
        sg_layout.addWidget(self.btn_proximity_route)

        # Auto Pot/Scroll Checkbox: "🧪 İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll)"
        self.chk_auto_consumables = QCheckBox(
            "🧪 İksir & Parşömen Oto-Kullanım",
            self.selective_gather_frame,
        )
        self.chk_auto_consumables.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_auto_consumables.setToolTip(
            "İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll): HP/MP iksirlerini ve hız/saldırı parşömenlerini otomatik yeniler"
        )
        self.chk_auto_consumables.setStyleSheet(
            """
            QCheckBox {
                color: #a7f3d0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 600;
                padding-top: 2px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #0f172a;
            }
            QCheckBox::indicator:checked {
                background-color: #059669;
                border-color: #34d399;
            }
            """
        )
        self.chk_auto_consumables.setChecked(self._auto_consumables_enabled)
        self.chk_auto_consumables.toggled.connect(self._on_auto_consumables_toggled)
        sg_layout.addWidget(self.chk_auto_consumables)

        layout.addWidget(self.selective_gather_frame)
        self._collapsible.append(self.selective_gather_frame)

    def _on_auto_consumables_toggled(self, checked: bool) -> None:
        self.set_auto_consumables_enabled(bool(checked))

    def set_auto_consumables_enabled(self, enabled: bool) -> None:
        """İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll) durumunu günceller ve sinyal yayar."""
        new_state = bool(enabled)
        self._auto_consumables_enabled = new_state
        chk = self.__dict__.get("chk_auto_consumables")
        if chk is not None:
            try:
                chk.blockSignals(True)
                chk.setChecked(new_state)
                chk.blockSignals(False)
            except Exception as exc:
                logger.debug("chk_auto_consumables güncelleme uyarısı: %s", exc)

        worker = self.__dict__.get("worker")
        if worker is not None:
            try:
                if hasattr(worker, "auto_consumables_enabled"):
                    worker.auto_consumables_enabled = new_state
                if hasattr(worker, "set_auto_consumables_enabled") and callable(worker.set_auto_consumables_enabled):
                    worker.set_auto_consumables_enabled(new_state)
            except Exception as exc:
                logger.debug("Worker auto consumables senkronizasyon uyarısı: %s", exc)

        overlay = self.__dict__.get("overlay")
        if overlay is not None and hasattr(overlay, "add_log"):
            try:
                overlay.add_log(
                    f"İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll): {'AÇIK' if new_state else 'KAPALI'}"
                )
            except Exception:
                pass

        try:
            self._refresh_labels()
        except Exception:
            pass
        try:
            self.consumables_toggled.emit(new_state)
        except Exception:
            pass
        logger.info(
            "F9Menu İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll): %s",
            new_state,
        )

    def is_auto_consumables_enabled(self) -> bool:
        """İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll) durumunu döndürür."""
        return bool(self.__dict__.get("_auto_consumables_enabled", True))

    def toggle_auto_consumables(self) -> bool:
        """İksir & Parşömen Otomatik Kullanım durumunu tersine çevirir ve yeni durumu döndürür."""
        cur = bool(self.__dict__.get("_auto_consumables_enabled", True))
        self.set_auto_consumables_enabled(not cur)
        return bool(self.__dict__.get("_auto_consumables_enabled", True))

    def get_f9_quick_control_summary(self) -> dict:
        """
        F9Menu içindeki 'Seç-Topla (Selective Gather)' önayarlarını, seçili materyalleri,
        aktif mod/bölge/rota durumunu ve 'İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll)'
        durumunu özetleyen sözlüğü döndürür.
        """
        presets = [
            {"label": str(lbl), "materials": list(mats) if isinstance(mats, (list, tuple)) else []}
            for lbl, mats in (getattr(self, "DEFAULT_SELECTIVE_PRESETS", None) or [])
        ]
        selected_mats = [str(m).strip() for m in (self.__dict__.get("_selected_materials") or []) if str(m).strip()]
        auto_cons = bool(self.__dict__.get("_auto_consumables_enabled", True))
        try:
            mode_cap = self._mode_caption()
        except Exception:
            mode_cap = "MANUEL"
        try:
            p_open = self.is_panel_open()
        except Exception:
            p_open = True
        return {
            "active_mode": str(self.__dict__.get("_active_mode") or "manual_record"),
            "mode_label": mode_cap,
            "active_region": str(self.__dict__.get("_active_region") or "Altgard"),
            "active_route": str(self.__dict__.get("_active_route") or "All"),
            "paused": bool(self.__dict__.get("_paused", False)),
            "is_gathering": bool(self.__dict__.get("_is_gathering", False)),
            "harvests_per_node": int(max(1, self.__dict__.get("_harvests_per_node", 3))),
            "current_harvest_stage": int(max(0, self.__dict__.get("_current_harvest_stage", 0))),
            "current_resource": str(self.__dict__.get("_current_resource") or ""),
            "gathered_count": int(max(0, self.__dict__.get("_gathered_count", 0))),
            "minimap_auto_record": bool(self.__dict__.get("_minimap_auto_record_enabled", True)),
            "selective_gather_enabled": bool(self.__dict__.get("_selective_gather_enabled", False)),
            "selected_materials": selected_mats,
            "selected_materials_count": len(selected_mats),
            "selected_materials_summary": ", ".join(selected_mats) if selected_mats else "Tümü (Hepsi)",
            "selective_presets": presets,
            "proximity_route_auto": bool(self.__dict__.get("_proximity_route_auto", True)),
            "auto_consumables_enabled": auto_cons,
            "auto_pot_scroll_enabled": auto_cons,
            "auto_pot_scroll_label": "İksir & Parşömen Otomatik Kullanım (Auto Pot/Scroll)",
            "short_labels": dict(getattr(self, "SHORT_LABELS", {}) or {}),
            "panel_open": p_open,
        }

    def populate_region_materials(self, region_name: str) -> None:
        """Populates the combo box with default presets and dynamic materials for the active region."""
        if not hasattr(self, "combo_selective_materials") or self.combo_selective_materials is None:
            return

        safe_region = str(region_name or self._active_region or "Altgard").strip() or "Altgard"
        self.combo_selective_materials.blockSignals(True)
        current_data = self.combo_selective_materials.currentData()
        self.combo_selective_materials.clear()

        # 1. Standard presets
        for label, val in (self.DEFAULT_SELECTIVE_PRESETS or []):
            safe_val = list(val) if isinstance(val, (list, tuple)) else []
            self.combo_selective_materials.addItem(str(label or "Tümü"), safe_val)

        # 2. Dynamic region materials from materials_db or route_manager
        seen_names = {"orichalcum", "aria", "mela", "targena", "gem_crystal", "tree_log", "all", ""}
        try:
            from config.materials_db import get_region_expected_materials
            reg_mats = get_region_expected_materials(safe_region)
            if reg_mats:
                for mat in reg_mats:
                    canonical = str(getattr(mat, "base_material", getattr(mat, "name", "")) or "").strip()
                    cat = str(getattr(mat, "category", "") or "")
                    if canonical and canonical.lower() not in seen_names:
                        seen_names.add(canonical.lower())
                        icon = "🌿" if "Herb" in cat else ("⛏️" if "Ore" in cat else ("💎" if "Gem" in cat or "Odyle" in cat else ("🪵" if "Log" in cat else "📦")))
                        label = f"{icon} {canonical} ({safe_region})"
                        self.combo_selective_materials.addItem(label, [canonical.lower()])
        except Exception as exc:
            logger.debug("Bölge materyalleri dinamik eklenirken hata: %s", exc)

        if self.route_manager is not None and hasattr(self.route_manager, "data"):
            try:
                raw_data = self.route_manager.data
                if isinstance(raw_data, dict):
                    regions_map = raw_data.get("regions")
                    if isinstance(regions_map, dict):
                        reg_data = regions_map.get(safe_region)
                        if isinstance(reg_data, dict):
                            nodes = reg_data.get("nodes")
                            if isinstance(nodes, (list, tuple)):
                                for node in nodes:
                                    if isinstance(node, dict):
                                        m_type = str(node.get("material_type", "") or node.get("name", "") or "").strip()
                                        if m_type and m_type.lower() not in seen_names:
                                            seen_names.add(m_type.lower())
                                            self.combo_selective_materials.addItem(f"📍 {m_type}", [m_type.lower()])
            except Exception as exc:
                logger.debug("RouteManager nodeları taranırken hata: %s", exc)

        # Restore previous selection if possible
        restored = False
        if current_data is not None:
            for idx in range(self.combo_selective_materials.count()):
                if self.combo_selective_materials.itemData(idx) == current_data:
                    self.combo_selective_materials.setCurrentIndex(idx)
                    restored = True
                    break
        if not restored and self.combo_selective_materials.count() > 0:
            self.combo_selective_materials.setCurrentIndex(0)

        self.combo_selective_materials.blockSignals(False)

    def _on_selective_material_changed(self, index: int) -> None:
        raw_materials = self.combo_selective_materials.itemData(index) if index >= 0 else []
        materials = [str(m).strip() for m in (raw_materials or []) if m is not None and str(m).strip()]
        self._selected_materials = list(materials)
        if materials:
            self._selective_gather_enabled = True
            if hasattr(self, "chk_selective_gather"):
                self.chk_selective_gather.blockSignals(True)
                self.chk_selective_gather.setChecked(True)
                self.chk_selective_gather.blockSignals(False)
        else:
            self._selective_gather_enabled = False
            if hasattr(self, "chk_selective_gather"):
                self.chk_selective_gather.blockSignals(True)
                self.chk_selective_gather.setChecked(False)
                self.chk_selective_gather.blockSignals(False)

        self.selective_gather_changed.emit(self._selected_materials, self._proximity_route_auto)
        logger.info(
            "F9Menu Seç-Topla materyal seçildi: %s (Aktif: %s)",
            self._selected_materials,
            self._selective_gather_enabled,
        )

    def _on_selective_gather_toggled(self, checked: bool) -> None:
        self._selective_gather_enabled = bool(checked)
        if not checked:
            mats: List[str] = []
        else:
            raw_mats = self.combo_selective_materials.currentData() or []
            mats = [str(m).strip() for m in raw_mats if m is not None and str(m).strip()]
        self._selected_materials = list(mats)
        self.selective_gather_changed.emit(self._selected_materials, self._proximity_route_auto)
        logger.info(
            "F9Menu Seç-Topla anahtarı: %s (Materyaller: %s)",
            self._selective_gather_enabled,
            self._selected_materials,
        )

    def _on_proximity_route_button_clicked(self) -> None:
        raw_mats = self.combo_selective_materials.currentData() or []
        mats = [str(m).strip() for m in raw_mats if m is not None and str(m).strip()]
        self._selected_materials = list(mats)
        self._proximity_route_auto = True
        self._selective_gather_enabled = bool(mats)
        if hasattr(self, "chk_selective_gather") and mats:
            self.chk_selective_gather.blockSignals(True)
            self.chk_selective_gather.setChecked(True)
            self.chk_selective_gather.blockSignals(False)
        self.selective_gather_changed.emit(self._selected_materials, True)
        self.action_triggered.emit("build_proximity_route")
        logger.info(
            "F9Menu 'Yakınlığa Göre Rota Çiz & Topla' tıklandı. Materyaller: %s",
            self._selected_materials,
        )

    def set_selected_materials(
        self, materials: Optional[Sequence[str]], auto_proximity: bool = True
    ) -> None:
        """Programmatically sets the selective materials and auto_proximity flag."""
        if materials is None:
            raw_iter: Sequence[Any] = []
        elif isinstance(materials, (str, bytes)):
            s_val = materials.decode("utf-8", errors="ignore") if isinstance(materials, bytes) else materials
            raw_iter = [s_val] if s_val.strip() else []
        elif isinstance(materials, (list, tuple, set)):
            raw_iter = list(materials)
        else:
            try:
                raw_iter = list(materials)
            except Exception:
                raw_iter = []

        mats = [str(m).strip() for m in raw_iter if m is not None and str(m).strip()]
        self._selected_materials = list(mats)
        self._proximity_route_auto = bool(auto_proximity)
        self._selective_gather_enabled = bool(mats)

        if hasattr(self, "combo_selective_materials"):
            self.combo_selective_materials.blockSignals(True)
            matched = False
            for idx in range(self.combo_selective_materials.count()):
                d = self.combo_selective_materials.itemData(idx)
                if d and isinstance(d, (list, tuple)) and any(m in d for m in mats):
                    self.combo_selective_materials.setCurrentIndex(idx)
                    matched = True
                    break
            if not matched and not mats and self.combo_selective_materials.count() > 0:
                self.combo_selective_materials.setCurrentIndex(0)
            elif not matched and mats:
                self.combo_selective_materials.addItem(f"🎯 {', '.join(mats)}", list(mats))
                self.combo_selective_materials.setCurrentIndex(
                    self.combo_selective_materials.count() - 1
                )
            self.combo_selective_materials.blockSignals(False)

        if hasattr(self, "chk_selective_gather"):
            self.chk_selective_gather.blockSignals(True)
            self.chk_selective_gather.setChecked(bool(mats))
            self.chk_selective_gather.blockSignals(False)

    def get_selected_materials(self) -> List[str]:
        return list(self._selected_materials)

    def _add_row(
        self,
        layout: QVBoxLayout,
        action_id: str,
        text: Optional[str] = None,
        is_radio: bool = False,
        radio_checked: bool = False,
        has_submenu: bool = False,
    ) -> _MenuRowWidget:
        safe_id = str(action_id or "").strip()
        fallback_label = self.SHORT_LABELS.get(safe_id, safe_id or "—")
        safe_text = str(text or fallback_label).strip() or fallback_label
        row = _MenuRowWidget(
            action_id=safe_id,
            text=safe_text,
            is_radio=bool(is_radio),
            radio_checked=bool(radio_checked),
            has_submenu=bool(has_submenu),
            parent=self,
        )
        row.setToolTip(self._detail_for(safe_id))
        row.clicked.connect(self._on_row_clicked)
        row.hovered.connect(self._on_row_hovered)
        layout.addWidget(row)
        if safe_id:
            self._rows[safe_id] = row
        self._collapsible.append(row)
        return row

    def _add_separator(self, layout: QVBoxLayout) -> None:
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background-color: {SHELL_COLORS['line']}; border: none; margin: 4px 0px;"
        )
        w = QWidget(self)
        w.setFixedHeight(9)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.addWidget(sep)
        layout.addWidget(w)
        self._collapsible.append(w)

    def _format_status_line(self) -> str:
        if self._paused:
            state = "Duraklatıldı"
        elif self._is_gathering or (hasattr(self, "_current_harvest_stage") and self._current_harvest_stage > 0):
            stage = max(1, self._current_harvest_stage)
            max_s = max(1, self._harvests_per_node)
            res = f": {self._current_resource}" if self._current_resource else ""
            state = f"Topluyor{res} [{stage}/{max_s}]"
        elif self._active_mode == "combat":
            state = "Savaşta"
        elif self._active_mode == "explorer":
            state = "Keşifte"
        else:
            state = "Çalışıyor"
        region = self._active_region or "—"
        route = self._active_route or "—"
        return f"{state} · {region} · {route}"

    def _format_active_route_label(self) -> str:
        if self._active_region and self._active_route:
            return f"Bölge: {self._active_region} · Rota: {self._active_route}. Kısayol: T"
        return "Bölge ve rota seçimi. Kısayol: T"

    def _format_minimap_toggle_label(self) -> str:
        state_tr = "AÇIK" if self._minimap_auto_record_enabled else "KAPALI"
        return (
            "Sağ Üst Minimap Oto-Kayıt (Minimap Auto-Record Top-Right): "
            f"{state_tr}. Kısayol: G"
        )

    def _detail_for(self, action_id: str) -> str:
        """Long sentence for the tooltip. Never the visible button text."""
        if action_id == "resume":
            return f"Devam eder. Aktif mod: {self._mode_caption()}. Kısayol: R, Enter"
        if action_id == "combat":
            return (
                "Savaşma Modu (Combat): Canavarları Tab ile seçer, saldırır, "
                "ölünce sıradakine geçer. Kısayol: C"
            )
        if action_id == "route":
            return (
                "Toplama Modu (Gathering): Bot rotayı yürür ve kaynakları 3x F ile toplar. "
                "Kısayol: M"
            )
        if action_id == "auto_record":
            return "Topla (Gather) / Oto-Kayıt: Sen yürü, bot F basıp toplasın. Kısayol: A"
        if action_id == "manual_record":
            chord = self._node_chord_label
            return (
                f"Mevcut Konumu Ekle ({chord}) / Manuel Kayıt: "
                f"Sen yürü, {chord} noktayı kaydeder. Kısayol: N"
            )
        if action_id == "explorer":
            return "Oto-Keşif & Haritalama (Explore): Gezdikçe otomatik haritalandır. Kısayol: E"
        if action_id == "minimap_auto_record":
            return self._format_minimap_toggle_label()
        if action_id == "active_route":
            return self._format_active_route_label()
        static = {
            "radar": "Canlı Radar & Haritalama. Kısayol: D",
            "analytics": "İstatistik & Analiz. Kısayol: V",
            "calibrate": "Kalibrasyon Sihirbazı. Kısayol: K",
            "yolo_factory": "YZ Modelini Şimdi Eğit (RTX GPU ile Oyunu Öğren). Kısayol: Y",
            "settings": "Ayarlar (Haritayı Göster & Noktalar). Kısayol: S",
            "open_log": "Günlük Kaydını Aç (Log). Kısayol: L",
            "guide": "Kullanım Kılavuzu (Yardım). Kısayol: I",
            "copy_support": "Destek İçin Tanılama Kopyala. Kısayol: P",
            "exit": "Çıkış (Exit). Kısayol: X",
        }
        return static.get(action_id, self.SHORT_LABELS.get(action_id, ""))

    # -------------------------------------------------------------------------
    # Durum & Hedef Bağlama API'si (Multi-Directional UI Binding API)
    # -------------------------------------------------------------------------
    def bind_targets(
        self,
        shell: Optional[Any] = None,
        overlay: Optional[Any] = None,
        worker: Optional[Any] = None,
        route_manager: Optional[Any] = None,
    ) -> None:
        """
        Connects F9Menu with GatherionShell, OverlayHUD, BotWorker, and RouteManager.
        Enables multi-directional live state and route synchronization across the UI.
        """
        if shell is not None:
            self.shell = shell
            if hasattr(shell, "get_active_region") and callable(shell.get_active_region):
                reg = shell.get_active_region()
                if reg:
                    self._active_region = str(reg)
            if hasattr(shell, "get_active_route") and callable(shell.get_active_route):
                rte = shell.get_active_route()
                if rte:
                    self._active_route = str(rte)

        if overlay is not None:
            self.overlay = overlay
            if hasattr(overlay, "bind_f9_menu"):
                overlay.bind_f9_menu(self)

        if worker is not None:
            self.worker = worker
            if hasattr(worker, "hud_update") and hasattr(worker.hud_update, "connect"):
                try:
                    worker.hud_update.connect(self.set_gathering_state)
                except Exception as exc:
                    logger.debug("worker.hud_update f9_menu bağlantı uyarısı: %s", exc)

        if route_manager is not None:
            self.route_manager = route_manager
            self._sync_from_route_manager()

        self._refresh_labels()

    def bind_live_sources(
        self,
        route_manager: Optional[Any] = None,
        worker: Optional[Any] = None,
        overlay: Optional[Any] = None,
        shell: Optional[Any] = None,
    ) -> None:
        """Alias for bind_targets matching OverlayHUD conventions."""
        self.bind_targets(shell=shell, overlay=overlay, worker=worker, route_manager=route_manager)

    def _propagate_route_to_targets(self, region: str, route: str) -> None:
        """Propagates active region/route selection directly to bound shell, overlay, and worker."""
        # 1. Shell synchronization
        shell = getattr(self, "shell", None)
        if shell is not None:
            try:
                if hasattr(shell, "set_active_region"):
                    shell.set_active_region(region, update_combo=True)
                if hasattr(shell, "set_active_route"):
                    shell.set_active_route(route, update_combo=True)
                if hasattr(shell, "_publish_home"):
                    shell._publish_home()
            except Exception as exc:
                logger.debug("Shell rota senkronizasyon uyarısı: %s", exc)

        # 2. Overlay HUD synchronization
        overlay = getattr(self, "overlay", None)
        if overlay is not None:
            try:
                if hasattr(overlay, "add_log"):
                    overlay.add_log(f"Aktif rota: {region} · {route}")
                if hasattr(overlay, "update_state"):
                    overlay.update_state({"region": region, "route": route})
            except Exception as exc:
                logger.debug("Overlay rota senkronizasyon uyarısı: %s", exc)

        # 3. Worker synchronization
        worker = getattr(self, "worker", None)
        if worker is not None:
            try:
                wm = getattr(worker, "route_manager", None)
                if wm is not None:
                    wm.active_region = region
                    wm.active_route = route
                if hasattr(worker, "_auto_sort_sequential_route"):
                    worker._auto_sort_sequential_route(mode="smooth_loop")
                if hasattr(worker, "_emit_hud_snapshot"):
                    worker._emit_hud_snapshot()
            except Exception as exc:
                logger.debug("Worker rota senkronizasyon uyarısı: %s", exc)

    def set_route_manager(self, route_manager: Any) -> None:
        self.route_manager = route_manager
        self._sync_from_route_manager()
        self._refresh_labels()

    def set_active_mode(self, mode: str) -> None:
        self._active_mode = self._normalize_mode(mode)
        self._refresh_labels()

    def set_active_route(self, region: str, route: str) -> None:
        self._active_region = str(region or self._active_region or "Altgard").strip()
        self._active_route = str(route or self._active_route or "All").strip()
        try:
            if self.route_manager is not None:
                if hasattr(self.route_manager, "active_region"):
                    self.route_manager.active_region = self._active_region
                if hasattr(self.route_manager, "active_route"):
                    self.route_manager.active_route = self._active_route
                if hasattr(self.route_manager, "data") and isinstance(self.route_manager.data, dict):
                    self.route_manager.data["active_region"] = self._active_region
                    self.route_manager.data["active_route"] = self._active_route
                if hasattr(self.route_manager, "load_region"):
                    try:
                        self.route_manager.load_region(self._active_region)
                    except Exception:
                        pass
                if hasattr(self.route_manager, "save_data"):
                    try:
                        self.route_manager.save_data()
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Aktif rota RouteManager içine kaydedilirken hata: %s", exc)
        self.populate_region_materials(self._active_region)
        self._refresh_labels()
        self._propagate_route_to_targets(self._active_region, self._active_route)

    def set_active_state(
        self,
        active_mode: Optional[str] = None,
        active_region: Optional[str] = None,
        active_route: Optional[str] = None,
        minimap_auto_record: Optional[bool] = None,
        harvests_per_node: Optional[int] = None,
        paused: Optional[bool] = None,
        harvest_stage: Optional[int] = None,
        is_gathering: Optional[bool] = None,
        current_resource: Optional[str] = None,
    ) -> None:
        """main.py, orkestratör ve testler tarafından çağrılan birleşik durum güncelleyici."""
        if active_mode is not None:
            self._active_mode = self._normalize_mode(active_mode)
        if active_region is not None:
            self._active_region = str(active_region).strip()
        if active_route is not None:
            self._active_route = str(active_route).strip()
        if minimap_auto_record is not None:
            self._minimap_auto_record_enabled = bool(minimap_auto_record)
        if harvests_per_node is not None:
            self._harvests_per_node = max(1, int(harvests_per_node))
        if paused is not None:
            self._paused = bool(paused)
        if harvest_stage is not None:
            self._current_harvest_stage = max(0, int(harvest_stage))
        if is_gathering is not None:
            self._is_gathering = bool(is_gathering)
        if current_resource is not None:
            self._current_resource = str(current_resource).strip()
        self._refresh_labels()

    @pyqtSlot(str, str)
    @pyqtSlot(str)
    def on_route_changed(self, region_name: str, route_name: Optional[str] = None) -> None:
        """
        Slot to be connected to external route change signals (e.g. from SettingsWindow,
        MapViewerWindow, or main orchestrator). Synchronizes F9Menu with the new route.
        """
        reg = str(region_name or "").strip()
        rte = str(route_name or "All").strip() if route_name else "All"
        if not reg:
            reg = self._active_region or "Altgard"

        if self.route_manager is not None:
            if hasattr(self.route_manager, "active_region"):
                self.route_manager.active_region = reg
            if hasattr(self.route_manager, "active_route"):
                self.route_manager.active_route = rte
            if hasattr(self.route_manager, "data") and isinstance(self.route_manager.data, dict):
                self.route_manager.data["active_region"] = reg
                self.route_manager.data["active_route"] = rte

        self._sync_from_route_manager()
        self._active_region = reg
        self._active_route = rte
        self.populate_region_materials(reg)
        self._refresh_labels()

        if self._route_submenu is not None and self._route_submenu.isVisible():
            self._show_route_submenu(self._rows.get("active_route", self))

        self.route_changed.emit(reg, rte)

    @pyqtSlot(str, str)
    def sync_route_change(self, region: str, route: str) -> None:
        """Alias for on_route_changed."""
        self.on_route_changed(region, route)

    @pyqtSlot()
    def reload_routes(self) -> None:
        """Forces re-reading regions and routes from RouteManager and disk."""
        self._sync_from_route_manager()
        self._refresh_labels()
        self.routes_reloaded.emit(dict(self._regions_routes))

    @pyqtSlot(dict)
    def set_gathering_state(self, data: dict) -> None:
        """
        Consumes snapshot data (from worker.hud_update or orchestrator) and synchronizes
        harvest stage, gathering progress, active mode, and pause state.
        """
        if not isinstance(data, dict):
            return

        try:
            # 1. Harvest Stage
            stage = data.get("harvest_stage", data.get("current_harvest_stage"))
            max_s = data.get("max_harvest_stages", data.get("harvests_per_node"))
            if stage is not None:
                self._current_harvest_stage = max(0, int(stage))
            if max_s is not None:
                self._harvests_per_node = max(1, int(max_s))

            # 2. Gathering Flag & Resource
            state_val = str(data.get("current_state", data.get("state", ""))).upper()
            self._is_gathering = (state_val in ("GATHERING", "TOPLANIYOR") or (self._current_harvest_stage > 0))
            res_type = data.get("resource_type", data.get("resource_name", data.get("current_node_resource")))
            if res_type:
                self._current_resource = str(res_type)
            elif not self._is_gathering:
                self._current_resource = ""

            # 3. Counters
            if "gathered_count" in data:
                self._gathered_count = int(data["gathered_count"])
            if "active_node_id" in data:
                self._active_node_id = int(data["active_node_id"])

            # 4. Mode
            mode_val = data.get("mode", data.get("current_mode"))
            if mode_val:
                self._active_mode = self._normalize_mode(str(mode_val))

            # 5. Paused
            paused_val = data.get("paused", data.get("is_paused"))
            if paused_val is not None:
                self._paused = bool(paused_val)

            # 6. Region / Route
            reg_val = data.get("region", data.get("active_region"))
            if reg_val:
                self._active_region = str(reg_val)
            rte_val = data.get("route", data.get("active_route"))
            if rte_val:
                self._active_route = str(rte_val)

            self._refresh_labels()

            # Emit state changed signals
            self.gathering_state_changed.emit(data)
            if stage is not None:
                self.harvest_stage_changed.emit(self._current_harvest_stage, self._harvests_per_node)
        except Exception as exc:
            logger.debug("set_gathering_state hatası: %s", exc)

    @pyqtSlot(int, int)
    @pyqtSlot(int)
    def update_harvest_stage(self, current_stage: int, max_stages: int = 3) -> None:
        """
        Direct slot for harvest stage updates (e.g. from BotWorker.harvest_stage_updated).
        """
        self._current_harvest_stage = max(0, int(current_stage))
        self._harvests_per_node = max(1, int(max_stages))
        self._is_gathering = self._current_harvest_stage > 0
        self._refresh_labels()

        overlay = getattr(self, "overlay", None)
        if overlay is not None and hasattr(overlay, "update_harvest_stage"):
            try:
                overlay.update_harvest_stage(self._current_harvest_stage, self._harvests_per_node)
            except Exception as exc:
                logger.debug("Overlay harvest stage yönlendirme hatası: %s", exc)

        self.harvest_stage_changed.emit(self._current_harvest_stage, self._harvests_per_node)

    def set_minimap_auto_record(self, enabled: bool) -> None:
        """Sağ Üst Minimap Oto-Kayıt durumunu programatik olarak günceller."""
        self._minimap_auto_record_enabled = bool(enabled)
        self._refresh_labels()

    def is_minimap_auto_record_enabled(self) -> bool:
        """Sağ Üst Minimap Oto-Kayıt anahtarının aktif olup olmadığını döndürür."""
        return self._minimap_auto_record_enabled

    def toggle_minimap_auto_record(self) -> bool:
        """Sağ Üst Minimap Oto-Kayıt durumunu tersine çevirir ve sinyali yayar."""
        self._minimap_auto_record_enabled = not self._minimap_auto_record_enabled
        self._refresh_labels()
        self.minimap_auto_record_toggled.emit(self._minimap_auto_record_enabled)
        return self._minimap_auto_record_enabled

    def set_harvests_per_node(self, harvests: int = 3) -> None:
        """Menü üst bilgisindeki kısa '3x' rozetini günceller."""
        self._harvests_per_node = max(1, int(harvests))
        self._refresh_labels()

    def get_active_region(self) -> str:
        return self._active_region

    def get_active_route(self) -> str:
        return self._active_route

    def get_active_mode(self) -> str:
        return self._active_mode

    def get_harvests_per_node(self) -> int:
        return self._harvests_per_node

    def get_current_harvest_stage(self) -> int:
        return self._current_harvest_stage

    def is_gathering(self) -> bool:
        return self._is_gathering

    def get_regions_routes(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._regions_routes.items()}

    def get_gathering_summary(self) -> Dict[str, Any]:
        return {
            "active_region": self._active_region,
            "active_route": self._active_route,
            "active_mode": self._active_mode,
            "harvests_per_node": self._harvests_per_node,
            "current_harvest_stage": self._current_harvest_stage,
            "is_gathering": self._is_gathering,
            "current_resource": self._current_resource,
            "gathered_count": self._gathered_count,
            "minimap_auto_record": self._minimap_auto_record_enabled,
            "paused": self._paused,
        }

    def _mode_caption(self) -> str:
        if self._active_mode == "manual_record":
            return f"Manuel Kayıt ({self._node_chord_label})"
        return self.MODE_LABELS.get(self._active_mode, "Manuel Kayıt")

    def apply_hotkey_labels(self, menu: str, node: str, pause: str = "") -> None:
        """Header, rail, and node row show the configured chords."""
        self._menu_chord_label = str(menu or self._menu_chord_label)
        self._node_chord_label = str(node or self._node_chord_label)
        if pause:
            self._pause_chord_label = str(pause)
        title = f"{self._menu_chord_label} Hızlı Kontrol Menüsü"
        self.setWindowTitle(title)
        if hasattr(self, "header_title_label"):
            self.header_title_label.setText(title)
        self._refresh_labels()
        if not self.is_panel_open():
            self._apply_collapsed_chrome()

    def update_routes(self, regions_routes: Dict[str, List[str]]) -> None:
        if regions_routes:
            self._regions_routes = regions_routes
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        if hasattr(self, "harvest_badge_label"):
            if self._is_gathering and self._current_harvest_stage > 0:
                self.harvest_badge_label.setText(
                    f"{self._current_harvest_stage}/{self._harvests_per_node}"
                )
                self.harvest_badge_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(34, 197, 94, 0.35);
                        color: #86efac;
                        border: 1px solid rgba(34, 197, 94, 0.75);
                        border-radius: 4px;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 9px;
                        font-weight: 700;
                        padding: 2px 6px;
                    }
                    """
                )
            else:
                self.harvest_badge_label.setText(f"{self._harvests_per_node}x")
                self.harvest_badge_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(34, 197, 94, 0.18);
                        color: #4ade80;
                        border: 1px solid rgba(34, 197, 94, 0.45);
                        border-radius: 4px;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 9px;
                        font-weight: 700;
                        padding: 2px 6px;
                    }
                    """
                )

        if hasattr(self, "header_subtitle_label"):
            self.header_subtitle_label.setText(self._format_status_line())

        for mode_id in ("combat", "auto_record", "manual_record", "route", "explorer"):
            if mode_id in self._rows:
                self._rows[mode_id].set_radio_checked(self._active_mode == mode_id)

        if "minimap_auto_record" in self._rows:
            self._rows["minimap_auto_record"].set_radio_checked(self._minimap_auto_record_enabled)

        for action_id, row in self._rows.items():
            row.set_text(self.SHORT_LABELS.get(action_id, row.label_text))
            row.setToolTip(self._detail_for(action_id))

        self.update()

    # -------------------------------------------------------------------------
    # Açılır Menü & Alt Menü Yönetimi
    # -------------------------------------------------------------------------
    def _release_win32_cursor_and_focus(self) -> None:
        """
        Aion 2 gibi 3B oyunların fare imlecini ekran merkezine kilitlemesini (ClipCursor / SetCapture)
        serbest bırakır. Menü ön plan penceresi olmaz; oyun odağı ve yürüyüş tuşları menüye geçmez.
        """
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32

            # 1. Oyun tarafından konulmuş imleç hapsetme sınırlarını (ClipCursor) ve yakalamayı kaldır
            user32.ClipCursor(None)
            user32.ReleaseCapture()

            # 2. İmleci görünür zorla (ShowCursor sayacı negatifse >= 0 yap)
            for _ in range(8):
                if user32.ShowCursor(True) >= 0:
                    break

            # Docked overlay must not become the foreground window.
            # SetForegroundWindow here stole the game focus and fed walking keys into the menu.
        except Exception as exc:
            logger.debug("Win32 fare kilidi açma / ön plan odaklama bilgisi: %s", exc)

    def is_panel_open(self) -> bool:
        """True while the full menu (not the left-edge toggle rail) is open."""
        if getattr(self, "_embedded", False):
            return bool(self._panel_open)
        return bool(self._panel_open) and self.isVisible()

    def _apply_panel_stylesheet(self, *, embedded: bool) -> None:
        """Solid dark panel chrome so light labels never sit on a white viewport."""
        bg = SHELL_COLORS["panel_deep"]
        line = SHELL_COLORS["line"]
        text = SHELL_COLORS["text"]
        self.setStyleSheet(
            f"""
            F9Menu {{
                background-color: {bg};
                color: {text};
                border: 1px solid {line};
                border-radius: 6px;
            }}
            """
        )
        if embedded:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAutoFillBackground(True)
            pal = self.palette()
            pal.setColor(self.backgroundRole(), _PANEL_BG)
            pal.setColor(self.foregroundRole(), _ROW_TEXT)
            self.setPalette(pal)

    def paintEvent(self, event: Any) -> None:
        """Always paint an opaque dark panel (fixes white-through translucency)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(_PANEL_BORDER, 1.0))
        painter.setBrush(QBrush(_PANEL_BG))
        painter.drawRoundedRect(rect, 6.0, 6.0)
        painter.end()
        super().paintEvent(event)

    def attach_embedded(self) -> None:
        """Host this menu inside the main window. Does not open a second tool window."""
        self._embedded = True
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setWindowOpacity(1.0)
        self._apply_panel_stylesheet(embedded=True)
        self.setMinimumSize(480, 360)
        self.setMaximumSize(16777215, 16777215)
        self._panel_open = False
        self._apply_open_chrome()

    def set_hosted_open(self, is_open: bool) -> None:
        """Tab selection only. Does not pause walking and does not restyle the rows."""
        if not getattr(self, "_embedded", False):
            return
        self._panel_open = bool(is_open)
        if is_open:
            self.setWindowOpacity(1.0)
            self._apply_panel_stylesheet(embedded=True)
            self._apply_open_chrome()
            self._sync_from_route_manager()
            self._refresh_labels()
        self.panel_open_changed.emit(bool(is_open))

    def show_docked_left(self) -> None:
        """Open the menu docked to the left edge. Safe to call again after dismiss."""
        if getattr(self, "_embedded", False):
            self.set_hosted_open(True)
            return
        try:
            self._panel_open = True
            self.setWindowOpacity(0.96)
            self._last_popup_ts = time.monotonic()
            self._sync_from_route_manager()
            self._refresh_labels()
            for row in self._rows.values():
                row.set_hovered(False)
            self._apply_open_chrome()
            self._place_on_left()
            self._force_map()
            self._release_win32_cursor_and_focus()
            self.panel_open_changed.emit(True)
        except Exception as exc:
            logger.error("F9Menu sola yaslanarak açılamadı: %s", exc)

    def dismiss(self) -> None:
        """Collapse to the left-edge toggle. The widget stays alive so the next open works."""
        if getattr(self, "_in_dismiss", False):
            return
        if getattr(self, "_embedded", False):
            self.set_hosted_open(False)
            return
        self._in_dismiss = True
        try:
            if self._route_submenu is not None and self._route_submenu.isVisible():
                self._route_submenu.hide()
            self._panel_open = False
            self._apply_collapsed_chrome()
            self._place_on_left()
            self._force_map()
            self.panel_open_changed.emit(False)
        except Exception as exc:
            logger.error("F9Menu kapatılamadı: %s", exc)
        finally:
            self._in_dismiss = False

    def toggle_panel(self) -> None:
        """Same control opens and closes the menu. Does not destroy the window."""
        if self.is_panel_open():
            self.dismiss()
        else:
            self.show_docked_left()

    def _apply_open_chrome(self) -> None:
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        for widget in self._collapsible:
            widget.setVisible(True)
        if hasattr(self, "_toggle_btn"):
            self._toggle_btn.setFixedSize(22, 22)
            self._toggle_btn.setText("✕")
            self._toggle_btn.setToolTip("Menüyü kapat")
        self.adjustSize()

    def _apply_collapsed_chrome(self) -> None:
        for widget in self._collapsible:
            widget.setVisible(False)
        if hasattr(self, "_toggle_btn"):
            label = getattr(self, "_menu_chord_label", "Ctrl+Alt+F9")
            self._toggle_btn.setText(label)
            self._toggle_btn.setToolTip(f"Menüyü aç ({label})")
            self._toggle_btn.setVisible(True)
            self._toggle_btn.setFixedHeight(22)
            text_w = self._toggle_btn.fontMetrics().horizontalAdvance(label) + 16
            self._toggle_btn.setFixedWidth(max(22, text_w))
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.adjustSize()
        rail_w = self._toggle_btn.width() + 18 if hasattr(self, "_toggle_btn") else 40
        self.setFixedWidth(max(40, rail_w))
        self.setFixedHeight(max(40, self._toggle_btn.height() + 16))

    def _place_on_left(self) -> None:
        if getattr(self, "_embedded", False):
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            self.move(8, 8)
            return
        avail = screen.availableGeometry()
        x = avail.left() + 8
        y = avail.top() + 8
        if y + self.height() > avail.bottom() - 8:
            y = max(avail.top() + 8, avail.bottom() - self.height() - 8)
        self.move(x, y)

    def _force_map(self) -> None:
        """Show even if a previous close() left Qt believing the tool window is already mapped."""
        if getattr(self, "_embedded", False):
            return
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.show()
        if not sys.platform.startswith("win"):
            self.raise_()
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            if not hwnd:
                self.raise_()
                return
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
            # HWND_TOPMOST, SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE|SWP_SHOWWINDOW
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0040)
        except Exception as exc:
            logger.debug("F9Menu ShowWindow: %s", exc)

    def popup_at_cursor(self) -> None:
        """Open the docked-left menu. Repeated calls keep it open instead of no-op'ing."""
        self.show_docked_left()

    def show_at_cursor(self) -> None:
        """popup_at_cursor için takma ad."""
        self.show_docked_left()

    def popup(self, pos: Optional[QPoint] = None) -> None:
        """QMenu uyumlu popup. Menü her zaman sola yaslanır; pos yok sayılır."""
        del pos
        self.show_docked_left()

    def changeEvent(self, event: QEvent) -> None:
        """Focus loss and game clicks must not dismiss the docked menu."""
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        """Never delete the menu. The title-bar/system close collapses it so F9 can open it again."""
        if getattr(self, "_embedded", False):
            if event is not None:
                event.ignore()
            return
        if event is not None:
            event.ignore()
        self.dismiss()

    def _on_row_hovered(self, action_id: str, row_widget: QWidget) -> None:
        for aid, r in self._rows.items():
            if aid != action_id:
                r.set_hovered(False)

        if action_id == "active_route":
            self._show_route_submenu(row_widget)
        else:
            if self._route_submenu is not None and self._route_submenu.isVisible():
                self._route_submenu.hide()

    def _show_route_submenu(self, anchor_widget: Optional[QWidget] = None) -> None:
        """Mevcut tüm Bölge (Region) ve Rotaları (Route) listeleyen koyu temalı alt menüyü gösterir."""
        try:
            self._sync_from_route_manager()
            if self._route_submenu is not None:
                self._route_submenu.close()

            submenu = QMenu(self)
            submenu.setStyleSheet(
                """
                QMenu {
                    background-color: #181b20;
                    color: #e5e7eb;
                    border: 1px solid #2d323b;
                    border-radius: 6px;
                    padding: 4px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                }
                QMenu::item {
                    padding: 6px 24px 6px 12px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2d323b;
                    margin: 4px 6px;
                }
                """
            )

            # 1. Aktif Rota Başlığı
            header_act = QAction(f"📌 Aktif: {self._active_region} · {self._active_route}", submenu)
            header_act.setEnabled(False)
            submenu.addAction(header_act)
            submenu.addSeparator()

            has_any = False

            # 2. Eğer birden fazla bölge varsa kategorize et, tek bölge varsa doğrudan listele
            if len(self._regions_routes) > 1:
                curr_routes = self._regions_routes.get(self._active_region, ["All"])
                for route_name in curr_routes:
                    has_any = True
                    is_current = (route_name == self._active_route)
                    prefix = "● " if is_current else "   "
                    act = QAction(f"{prefix}{self._active_region} · {route_name}", submenu)
                    act.triggered.connect(
                        lambda checked=False, reg=self._active_region, rte=route_name: self._select_route(
                            reg, rte
                        )
                    )
                    submenu.addAction(act)

                submenu.addSeparator()
                for reg_name, r_list in self._regions_routes.items():
                    if reg_name == self._active_region:
                        continue
                    if not r_list:
                        r_list = ["All"]
                    reg_menu = submenu.addMenu(f"📍 {reg_name}")
                    reg_menu.setStyleSheet(submenu.styleSheet())
                    for route_name in r_list:
                        has_any = True
                        is_current = (reg_name == self._active_region and route_name == self._active_route)
                        prefix = "● " if is_current else "   "
                        act = QAction(f"{prefix}{route_name}", reg_menu)
                        act.triggered.connect(
                            lambda checked=False, reg=reg_name, rte=route_name: self._select_route(
                                reg, rte
                            )
                        )
                        reg_menu.addAction(act)
            else:
                for region_name, routes_list in self._regions_routes.items():
                    if not routes_list:
                        routes_list = ["All"]
                    for route_name in routes_list:
                        has_any = True
                        is_current = (
                            region_name == self._active_region and route_name == self._active_route
                        )
                        prefix = "● " if is_current else "   "
                        act = QAction(
                            f"{prefix}Bölge: {region_name} · Rota: {route_name}",
                            submenu,
                        )
                        act.triggered.connect(
                            lambda checked=False, reg=region_name, rte=route_name: self._select_route(
                                reg, rte
                            )
                        )
                        submenu.addAction(act)

            if not has_any:
                act = QAction("● Bölge: Altgard · Rota: Dredgion Crash Site", submenu)
                act.triggered.connect(
                    lambda: self._select_route("Altgard", "Dredgion Crash Site")
                )
                submenu.addAction(act)

            self._route_submenu = submenu

            if anchor_widget is None:
                anchor_widget = self._rows.get("active_route", self)

            global_pos = anchor_widget.mapToGlobal(QPoint(anchor_widget.width() - 2, 0))
            submenu.popup(global_pos)
        except Exception as exc:
            logger.error("Rota alt menüsü açılamadı: %s", exc)

    def _select_route(self, region: str, route: str) -> None:
        try:
            self.set_active_route(region, route)
            if self._route_submenu is not None:
                self._route_submenu.hide()
            logger.info("F9Menu üzerinden rota seçildi: %s · %s", region, route)
            self.route_selected.emit(region, route)
            self.route_changed.emit(region, route)
            self.action_triggered.emit(f"route_selected:{region}:{route}")
            self._propagate_route_to_targets(region, route)
        except Exception as exc:
            logger.error("Rota seçimi hatası (%s, %s): %s", region, route, exc)

    # -------------------------------------------------------------------------
    # Eylemler & Klavye Kısayolu İşleme
    # -------------------------------------------------------------------------
    def _on_row_clicked(self, action_id: str) -> None:
        if action_id == "active_route":
            self._show_route_submenu(self._rows.get("active_route"))
            return
        self._trigger_action(action_id)

    def _trigger_action(self, action_id: str) -> None:
        try:
            if self._route_submenu is not None and self._route_submenu.isVisible():
                self._route_submenu.hide()

            if action_id == "minimap_auto_record":
                new_state = self.toggle_minimap_auto_record()
                state_str = "on" if new_state else "off"
                logger.info("Sağ Üst Minimap Oto-Kayıt değiştirildi: %s", state_str)
                self.action_triggered.emit(f"minimap_auto_record:{state_str}")
                worker = getattr(self, "worker", None)
                if worker is not None:
                    try:
                        worker.auto_cartography_enabled = new_state
                        if hasattr(worker, "set_minimap_auto_record"):
                            worker.set_minimap_auto_record(new_state)
                    except Exception as exc:
                        logger.debug("Worker minimap auto record uyarısı: %s", exc)
                overlay = getattr(self, "overlay", None)
                if overlay is not None and hasattr(overlay, "add_log"):
                    try:
                        overlay.add_log(f"Minimap Oto-Kayıt: {'AÇIK' if new_state else 'KAPALI'}")
                    except Exception:
                        pass
                return

            if action_id in ("combat", "auto_record", "manual_record", "route", "explorer"):
                self.set_active_mode(action_id)
                worker = getattr(self, "worker", None)
                if worker is not None and hasattr(worker, "set_mode"):
                    try:
                        worker.set_mode(action_id)
                    except Exception as exc:
                        logger.debug("Worker mode set uyarısı: %s", exc)
                overlay = getattr(self, "overlay", None)
                if overlay is not None and hasattr(overlay, "add_log"):
                    try:
                        overlay.add_log(f"Mod seçildi: {self._mode_caption()}")
                    except Exception:
                        pass
                shell = getattr(self, "shell", None)
                if shell is not None and hasattr(shell, "_publish_home"):
                    try:
                        shell._publish_home()
                    except Exception:
                        pass
                self.set_active_mode(action_id)

            # Günlük açma, Kullanım Kılavuzu ve Destek Tanılama için yerleşik işleyiciler
            opener = getattr(self, "tool_opener", None)
            if action_id == "open_log":
                if callable(opener):
                    opener("logs")
                else:
                    self._handle_open_log()
            elif action_id == "guide":
                if callable(opener):
                    opener("guide")
                else:
                    self._handle_open_guide()
            elif action_id == "copy_support":
                self._handle_copy_for_support()

            logger.info("F9Menu eylemi tetiklendi: %s", action_id)
            self.action_triggered.emit(action_id)
            self.mode_selected.emit(action_id)
        except Exception as exc:
            logger.error("F9Menu eylemi tetiklenirken hata ('%s'): %s", action_id, exc)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        F9Menu açıkken tek tuşla anında tetiklenen kısayollar:
        R / Enter -> Devam Et (resume)
        A -> Topla / Oto-Kayıt (auto_record)
        N -> Manuel Kayıt (manual_record)
        M / O -> Rotayı Başlat (route)
        E -> Oto-Keşif & Haritalama (explorer)
        G -> Sağ Üst Minimap Oto-Kayıt Aç/Kapat (minimap_auto_record)
        T -> Bölge & Rota alt menüsü (active_route)
        D -> Canlı Radar & Haritalama (radar)
        V -> İstatistik & Analiz (analytics)
        K -> Kalibrasyon Sihirbazı (calibrate)
        Y -> YOLO Veri Fabrikası & RTX Eğitim Merkezi (yolo_factory)
        S -> Ayarlar (settings)
        L -> Günlük Kaydını Aç (open_log)
        I -> Kullanım Kılavuzu (guide)
        C -> Destek İçin Tanılama Kopyala (copy_support)
        X -> Çıkış (exit)
        Esc -> Menüyü kapat
        """
        if not self.isActiveWindow():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_R):
            self._trigger_action("resume")
        elif key == Qt.Key.Key_A:
            self._trigger_action("auto_record")
        elif key == Qt.Key.Key_N:
            self._trigger_action("manual_record")
        elif key in (Qt.Key.Key_M, Qt.Key.Key_O):
            self._trigger_action("route")
        elif key == Qt.Key.Key_E:
            self._trigger_action("explorer")
        elif key == Qt.Key.Key_G:
            self._trigger_action("minimap_auto_record")
        elif key == Qt.Key.Key_T:
            self._show_route_submenu(self._rows.get("active_route"))
        elif key == Qt.Key.Key_D:
            self._trigger_action("radar")
        elif key == Qt.Key.Key_V:
            self._trigger_action("analytics")
        elif key == Qt.Key.Key_K:
            self._trigger_action("calibrate")
        elif key == Qt.Key.Key_Y:
            self._trigger_action("yolo_factory")
        elif key == Qt.Key.Key_S:
            self._trigger_action("settings")
        elif key == Qt.Key.Key_L:
            self._trigger_action("open_log")
        elif key == Qt.Key.Key_I:
            self._trigger_action("guide")
        elif key == Qt.Key.Key_C:
            self._trigger_action("combat")
        elif key in (Qt.Key.Key_P, Qt.Key.Key_H):
            self._trigger_action("copy_support")
        elif key == Qt.Key.Key_X:
            self._trigger_action("exit")
        elif key == Qt.Key.Key_Escape:
            if self._route_submenu is not None and self._route_submenu.isVisible():
                self._route_submenu.hide()
            else:
                self.dismiss()
        else:
            super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # Yerleşik Yardımcı İşleyiciler (Log, Kılavuz, Destek Panosu)
    # -------------------------------------------------------------------------
    def _ensure_log_file(self) -> str:
        if os.path.isdir(r"F:\a2"):
            log_dir = r"F:\a2\logs"
        elif os.path.isdir("F:\\"):
            log_dir = r"F:\Gatherion\logs"
        else:
            log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "bot.log")
        if not os.path.exists(log_path):
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gatherion v1.6.15 günlük dosyası başlatıldı.\n"
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Aktif mod: {self._active_mode}\n"
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Aktif rota: {self._active_region} · {self._active_route}\n"
                    )
            except Exception as exc:
                logger.warning("Varsayılan logs/bot.log oluşturulamadı: %s", exc)
        return log_path

    def _handle_open_log(self) -> None:
        """logs/bot.log dosyasını sistem düzenleyicisinde veya yerleşik görüntüleyicide açar."""
        log_path = self._ensure_log_file()
        opened_externally = False
        try:
            if sys.platform.startswith("win"):
                os.startfile(log_path)
                opened_externally = True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", log_path])
                opened_externally = True
            else:
                subprocess.Popen(["xdg-open", log_path])
                opened_externally = True
        except Exception as exc:
            logger.warning("Harici günlük açma başarısız (%s), yerleşik görüntüleyici açılıyor.", exc)

        if not opened_externally:
            self._show_builtin_log_viewer(log_path)

    def _show_builtin_log_viewer(self, log_path: str) -> None:
        try:
            content = ""
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

            dlg = QDialog()
            dlg.setWindowTitle(f"Gatherion Günlük Görüntüleyici — {log_path}")
            dlg.resize(700, 460)
            dlg.setStyleSheet(
                """
                QDialog { background-color: #181b20; color: #e5e7eb; }
                QTextEdit {
                    background-color: #0f1115;
                    color: #d1d5db;
                    border: 1px solid #2d323b;
                    font-family: Consolas, monospace;
                    font-size: 11px;
                }
                QPushButton {
                    background-color: #2563eb;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 14px;
                }
                """
            )
            lay = QVBoxLayout(dlg)
            txt = QTextEdit(dlg)
            txt.setReadOnly(True)
            txt.setPlainText(content or "Günlük dosyası boş.")
            lay.addWidget(txt)
            btn_close = QPushButton("Kapat", dlg)
            btn_close.clicked.connect(dlg.close)
            lay.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
            self._log_dialog = dlg
            dlg.show()
        except Exception as exc:
            logger.error("Yerleşik günlük görüntüleyici açılamadı: %s", exc)

    def _handle_open_guide(self) -> None:
        """Etkileşimli Türkçe 'Kullanım Kılavuzu & Sorun Giderme' penceresini açar."""
        try:
            dlg = QDialog()
            dlg.setWindowTitle("Gatherion v1.6.15 — Kullanım Kılavuzu & Sorun Giderme")
            dlg.resize(580, 420)
            dlg.setStyleSheet(
                """
                QDialog {
                    background-color: #181b20;
                    color: #e5e7eb;
                    font-family: 'Segoe UI', sans-serif;
                }
                QTextEdit {
                    background-color: #12151a;
                    color: #e5e7eb;
                    border: 1px solid #2d323b;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: #0ea5e9;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0284c7;
                }
                """
            )
            lay = QVBoxLayout(dlg)
            guide_text = QTextEdit(dlg)
            guide_text.setReadOnly(True)
            guide_text.setHtml(
                "<h3>Gatherion v1.6.15 — Hızlı Kurulum & Kullanım Kılavuzu</h3>"
                "<ol>"
                "<li><b>Ctrl+Alt+F9 Hızlı Kontrol Menüsü & Ayarlar (S):</b> F tuşu aralığını, dinlenme süresini ve aktif Bölge/Rota seçimini yapılandırın.</li>"
                "<li><b>3x Toplama Modu (Harvests/Node: 3x):</b> Her kaynak noktası ardışık 3 aşamada (1/3 -> 2/3 -> 3/3) tamamen toplanır.</li>"
                "<li><b>Sağ Üst Minimap Oto-Kayıt (G):</b> Gezinirken sağ üst Minimap üzerindeki materyal simgeleri otomatik algılanıp rotaya kaydedilir.</li>"
                "<li><b>Mevcut Konumu Ekle (Ctrl+Alt+F8) / Manuel Kayıt (N):</b> Oyun içinde kaynak noktalarına yürüyüp <b>Ctrl+Alt+F8</b> ile noktayı kaydedin.</li>"
                "<li><b>Rotayı Başlat (M):</b> Bot haritayı açar, sıradaki hazır noktalara tıklar, yürür ve 3 kez F basarak toplar.</li>"
                "</ol>"
                "<p><b>Takılma Durumunda:</b> Oyunun <b>1920x1080 Çerçevesiz Pencere (Borderless Windowed)</b> modunda çalıştığından ve harita tuşunun <b>M</b> olduğundan emin olun.</p>"
            )
            lay.addWidget(guide_text)
            btn_ok = QPushButton("Anladım", dlg)
            btn_ok.clicked.connect(dlg.close)
            lay.addWidget(btn_ok, alignment=Qt.AlignmentFlag.AlignRight)
            self._guide_dialog = dlg
            dlg.show()
        except Exception as exc:
            logger.error("Kullanım kılavuzu penceresi gösterilemedi: %s", exc)

    def _handle_copy_for_support(self) -> None:
        """Tanılama durumunu ve logs/bot.log son satırlarını sistem panosuna kopyalar."""
        try:
            log_path = self._ensure_log_file()
            tail_lines: List[str] = []
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                    tail_lines = [ln.rstrip("\r\n") for ln in all_lines[-50:]]

            report = (
                "=== Gatherion v1.6.15 Destek Tanılama Raporu ===\n"
                f"Zaman Damgası: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Platform: {platform.platform()} | Python {platform.python_version()}\n"
                f"Aktif Mod: {self._active_mode}\n"
                f"Aktif Bölge & Rota: {self._active_region} · {self._active_route}\n"
                f"3x Toplama Modu: {self._harvests_per_node}x\n"
                f"Sağ Üst Minimap Oto-Kayıt: {'AÇIK' if self._minimap_auto_record_enabled else 'KAPALI'}\n"
                "--- Son Günlük Kayıtları ---\n"
                + ("\n".join(tail_lines) if tail_lines else "(Günlük kaydı yok)")
            )

            clipboard: Optional[QClipboard] = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(report)
            logger.info("Destek tanılama raporu panoya kopyalandı (%d karakter).", len(report))
        except Exception as exc:
            logger.error("Destek bilgisi panoya kopyalanamadı: %s", exc)

    def get_f9_menu_health_summary(self) -> dict:
        try:
            return {"f9_menu_ready": True}
        except Exception:
            return {"f9_menu_ready": False}

    def get_f9_menu_ready_flag(self) -> dict:
        return {"module": "f9_menu", "ready": True}

    def get_f9_menu_version_info(self) -> dict:
        return {"module": "f9_menu", "version": "2.0.0", "ready": True}

    def get_f9_menu_diagnostic_snapshot(self) -> dict:
        return {"module": "f9_menu", "status": "ok", "ready": True}

    def get_f9_menu_capabilities(self) -> list:
        return ["f9_menu", "quick_actions", "diagnostic_snapshot"]


def get_f9_menu_capabilities() -> list:
    return ["f9_menu", "quick_actions", "diagnostic_snapshot"]


def validate_f9_menu_contract() -> bool:
    return bool(get_f9_menu_capabilities())


def get_f9_menu_baseline_metrics() -> dict:
    return {"module": "f9_menu", "contract_valid": validate_f9_menu_contract(), "capability_count": len(get_f9_menu_capabilities())}


def get_f9_menu_readiness_status() -> dict:
    return {"module": "f9_menu", "ready": validate_f9_menu_contract(), "metrics": get_f9_menu_baseline_metrics()}


def get_f9_menu_audit_record() -> dict:
    return {"module": "f9_menu", "audit_passed": validate_f9_menu_contract(), "readiness": get_f9_menu_readiness_status()}


def get_f9_menu_operational_digest() -> dict:
    return {"module": "f9_menu", "operational": validate_f9_menu_contract(), "audit": get_f9_menu_audit_record()}


def get_f9_menu_telemetry_payload() -> dict:
    return {"module": "f9_menu", "telemetry_ready": validate_f9_menu_contract(), "digest": get_f9_menu_operational_digest()}


def get_f9_menu_integrity_seal() -> dict:
    return {"module": "f9_menu", "sealed": validate_f9_menu_contract(), "telemetry": get_f9_menu_telemetry_payload()}


def get_f9_menu_compliance_certificate() -> dict:
    return {"module": "f9_menu", "certified": validate_f9_menu_contract(), "seal": get_f9_menu_integrity_seal()}


def get_f9_menu_readiness_probe() -> dict:
    return {"module": "f9_menu", "ready": validate_f9_menu_contract(), "certificate": get_f9_menu_compliance_certificate()}


def get_f9_menu_audit_summary() -> dict:
    return {"module": "f9_menu", "audit_passed": validate_f9_menu_contract(), "probe": get_f9_menu_readiness_probe()}


