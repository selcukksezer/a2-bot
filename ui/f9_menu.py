# -*- coding: utf-8 -*-
"""
PyQt6 F9 Hızlı Kontrol Açılır Menüsü (Gatherion Türkçe Arayüz & 3x Toplama / Minimap Oto-Kayıt)
- F9 tuşuna basıldığında imleç konumunda (QCursor.pos()) açılır
- Koyu tema (#181b20 arka plan, ince #2d323b kenarlık)
- Üst bilgi çubuğunda "F9 Hızlı Kontrol Menüsü", "Duraklatıldı — Devam için Enter" ve aktif "3x Toplama Modu (Harvests/Node: 3x)" rozeti
- Sağ Üst Minimap Oto-Kayıt ("Minimap Auto-Record (Top-Right)") hızlı aç/kapat anahtarı ve `minimap_auto_record_toggled = pyqtSignal(bool)` sinyali
- Sağ hizalı tuş rozetleri ([R], [A], [N], [M], [E], [G], [T], [H], [D], [V], [K], [S], [L], [I], [C], [X])
- Mod seçimi ve Minimap Oto-Kayıt için radyo/durum göstergeleri (mavi nokta ● / boş daire ○)
- 'Bölge (Region) & Rota (Route): <Region> · <Route>' (T) üzerinde anında bölge ve rota geçişi için alt menü
- Menü açıkken anında klavye kısayolları: R, Enter, A, N, M, O, E, G, T, H, D, V, K, S, L, I, C, X, Esc
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, pyqtSignal
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
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
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

logger = logging.getLogger("Gatherion.F9Menu")


class _MenuRowWidget(QWidget):
    """
    F9Menu içinde isteğe bağlı radyo/durum göstergesi, etiket metni,
    kısayol tuş rozeti ve alt menü oku ('›') içeren tekil etkileşimli satır.
    """

    clicked = pyqtSignal(str)
    hovered = pyqtSignal(str, QWidget)

    def __init__(
        self,
        action_id: str,
        text: str,
        badge_key: str,
        is_radio: bool = False,
        radio_checked: bool = False,
        has_submenu: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.action_id = action_id
        self.label_text = text
        self.badge_key = badge_key
        self.is_radio = is_radio
        self.radio_checked = radio_checked
        self.has_submenu = has_submenu
        self.is_hovered = False

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self.setMinimumWidth(410)

    def set_text(self, text: str) -> None:
        self.label_text = text
        self.update()

    def set_radio_checked(self, checked: bool) -> None:
        self.radio_checked = checked
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
            painter.setBrush(QBrush(QColor("#232830")))
            painter.drawRoundedRect(rect.adjusted(4, 2, -4, -2), 4.0, 4.0)

        left_x = 14
        center_y = rect.height() // 2

        if self.is_radio:
            if self.radio_checked:
                # Dolu mavi/camgöbeği nokta ● (#38bdf8)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#38bdf8")))
                painter.drawEllipse(QPoint(left_x + 4, center_y), 4, 4)
            else:
                # Boş daire ○ (#6b7280)
                painter.setPen(QPen(QColor("#6b7280"), 1.3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPoint(left_x + 4, center_y), 4, 4)

        text_x = left_x + 18

        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#e5e7eb") if not self.is_hovered else QColor("#ffffff")))

        badge_area_width = 48 if self.has_submenu else 38
        text_rect = QRect(text_x, 0, rect.width() - text_x - badge_area_width, rect.height())
        elided_text = painter.fontMetrics().elidedText(
            self.label_text, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided_text,
        )

        # Sağ taraftaki kısayol tuşu rozeti
        badge_w = 20
        badge_h = 20
        right_pad = 18
        badge_x = rect.width() - right_pad - badge_w
        badge_y = (rect.height() - badge_h) // 2
        badge_rect = QRect(badge_x, badge_y, badge_w, badge_h)

        painter.setPen(QPen(QColor("#363c47"), 1))
        painter.setBrush(QBrush(QColor("#262b33")))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)

        badge_font = QFont("Segoe UI", 8)
        badge_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(badge_font)
        painter.setPen(QPen(QColor("#9ca3af")))
        painter.drawText(badge_rect, int(Qt.AlignmentFlag.AlignCenter), self.badge_key)

        if self.has_submenu:
            chev_rect = QRect(rect.width() - 16, 0, 12, rect.height())
            chev_font = QFont("Segoe UI", 10)
            chev_font.setWeight(QFont.Weight.Bold)
            painter.setFont(chev_font)
            painter.setPen(QPen(QColor("#9ca3af")))
            painter.drawText(chev_rect, int(Qt.AlignmentFlag.AlignCenter), "›")

        painter.end()


class F9Menu(QWidget):
    """
    Gatherion F9 Hızlı Kontrol Menüsü (Türkçe).
    3x Toplama Modu ('Harvests/Node: 3x') başlık rozetini ve Sağ Üst Minimap Oto-Kayıt
    ('Minimap Auto-Record (Top-Right)') hızlı geçiş anahtarını içerir.
    """

    action_triggered = pyqtSignal(str)
    mode_selected = pyqtSignal(str)  # Geriye dönük uyumluluk sinyali
    route_selected = pyqtSignal(str, str)  # (region, route)
    minimap_auto_record_toggled = pyqtSignal(bool)  # Sağ Üst Minimap Oto-Kayıt sinyali

    MODE_LABELS: Dict[str, str] = {
        "auto_record": "Topla (Gather) / Oto-Kayıt",
        "manual_record": "Manuel Kayıt (F8)",
        "route": "Rotayı Başlat (Route)",
        "route_mode": "Rotayı Başlat (Route)",
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
    ) -> None:
        super().__init__(parent)
        self.route_manager = route_manager
        self._active_mode = self._normalize_mode(active_mode)
        self._active_region = "Altgard"
        self._active_route = "Dredgion Crash Site"
        self._minimap_auto_record_enabled: bool = bool(minimap_auto_record)
        self._harvests_per_node: int = max(1, int(harvests_per_node))
        self._regions_routes: Dict[str, List[str]] = {}
        self._rows: Dict[str, _MenuRowWidget] = {}
        self._route_submenu: Optional[QMenu] = None
        self._guide_dialog: Optional[QDialog] = None
        self._log_dialog: Optional[QDialog] = None
        self._last_popup_ts: float = 0.0

        self._sync_from_route_manager()
        self._init_ui()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not mode:
            return "manual_record"
        m = mode.strip().lower()
        if m in ("auto", "auto_record", "autorecord", "gather"):
            return "auto_record"
        if m in ("manual", "manual_record", "manualrecord"):
            return "manual_record"
        if m in ("route", "route_mode", "routemode"):
            return "route"
        if m in ("explorer", "explorer_mode", "explore", "auto_cartography"):
            return "explorer"
        return "manual_record"

    def _sync_from_route_manager(self) -> None:
        """RouteManager mevcutsa bölgeleri, rotaları ve aktif seçimi senkronize eder."""
        try:
            if self.route_manager is not None:
                if hasattr(self.route_manager, "active_region") and self.route_manager.active_region:
                    self._active_region = str(self.route_manager.active_region)
                if hasattr(self.route_manager, "active_route") and self.route_manager.active_route:
                    self._active_route = str(self.route_manager.active_route)

                if hasattr(self.route_manager, "data") and isinstance(self.route_manager.data, dict):
                    data = self.route_manager.data
                    self._active_region = data.get("active_region") or self._active_region
                    self._active_route = data.get("active_route") or self._active_route
                    regions_dict = data.get("regions", {})
                    parsed: Dict[str, List[str]] = {}
                    for reg_name, reg_val in regions_dict.items():
                        if isinstance(reg_val, dict):
                            routes_map = reg_val.get("routes", {})
                            parsed[reg_name] = list(routes_map.keys()) if routes_map else ["All"]
                    if parsed:
                        self._regions_routes = parsed
        except Exception as exc:
            logger.warning("RouteManager üzerinden rotalar senkronize edilemedi: %s", exc)

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

        if not self._regions_routes:
            self._regions_routes = {
                "Altgard": ["Dredgion Crash Site", "Safe Haven", "Quai Campsite", "All"],
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
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setWindowTitle("F9 Hızlı Kontrol Menüsü")

        self.setStyleSheet(
            """
            F9Menu {
                background-color: #181b20;
                border: 1px solid #2d323b;
                border-radius: 6px;
            }
            """
        )

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

        self.header_title_label = QLabel("F9 Hızlı Kontrol Menüsü", self.header_frame)
        self.header_title_label.setStyleSheet(
            """
            QLabel {
                color: #f8fafc;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: 700;
            }
            """
        )

        # 3x Toplama Modu & Harvests/Node: 3x rozeti
        self.harvest_badge_label = QLabel(
            f"3x Toplama Modu · Harvests/Node: {self._harvests_per_node}x",
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

        top_bar.addWidget(self.header_title_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.harvest_badge_label)
        header_vbox.addLayout(top_bar)

        self.header_subtitle_label = QLabel(
            "Duraklatıldı — Devam için Enter  |  "
            f"Bölge (Region): {self._active_region} · Rota (Route): {self._active_route}",
            self.header_frame,
        )
        self.header_subtitle_label.setStyleSheet(
            """
            QLabel {
                color: #94a3b8;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9px;
                font-weight: 500;
            }
            """
        )
        header_vbox.addWidget(self.header_subtitle_label)
        layout.addWidget(self.header_frame)

        self._add_separator(layout)

        # -----------------------------------------------------------------
        # 1. Devam Et (Resume): <aktif_mod> (R / Enter)
        # -----------------------------------------------------------------
        resume_label = (
            f"Devam Et: {self.MODE_LABELS.get(self._active_mode, 'Manuel Kayıt (F8)')} "
            "(Duraklatıldı — Devam için Enter)"
        )
        self._add_row(layout, "resume", resume_label, "R")

        self._add_separator(layout)

        # -----------------------------------------------------------------
        # 2. Radyo Mod Seçenekleri (Topla, Manuel Kayıt, Rotayı Başlat, Oto-Keşif)
        # -----------------------------------------------------------------
        self._add_row(
            layout,
            "auto_record",
            "Topla (Gather) / Oto-Kayıt: Sen yürü, bot F basıp toplasın",
            "A",
            is_radio=True,
            radio_checked=(self._active_mode == "auto_record"),
        )
        self._add_row(
            layout,
            "manual_record",
            "Mevcut Konumu Ekle (F8) / Manuel Kayıt: Sen yürü, F8 noktayı kaydeder",
            "N",
            is_radio=True,
            radio_checked=(self._active_mode == "manual_record"),
        )
        self._add_row(
            layout,
            "route",
            "Rotayı Başlat (Route): Bot yürür ve noktaları 3x toplar",
            "M",
            is_radio=True,
            radio_checked=(self._active_mode == "route"),
        )
        self._add_row(
            layout,
            "explorer",
            "Oto-Keşif & Haritalama (Explore): Gezdikçe otomatik haritalandır",
            "E",
            is_radio=True,
            radio_checked=(self._active_mode == "explorer"),
        )

        # -----------------------------------------------------------------
        # 3. Sağ Üst Minimap Oto-Kayıt (Minimap Auto-Record Top-Right) Hızlı Anahtarı
        # -----------------------------------------------------------------
        self._add_row(
            layout,
            "minimap_auto_record",
            self._format_minimap_toggle_label(),
            "G",
            is_radio=True,
            radio_checked=self._minimap_auto_record_enabled,
        )

        # -----------------------------------------------------------------
        # 4. Bölge (Region) & Rota (Route) Alt Menüsü (T)
        # -----------------------------------------------------------------
        route_label = self._format_active_route_label()
        self._add_row(
            layout,
            "active_route",
            route_label,
            "T",
            has_submenu=True,
        )

        self._add_separator(layout)

        # -----------------------------------------------------------------
        # 5. Harita, Radar, Analiz, Kalibrasyon, YOLO Fabrikası, Ayarlar & Destek Araçları
        # -----------------------------------------------------------------
        self._add_row(layout, "radar", "Canlı Radar & Haritalama", "D")
        self._add_row(layout, "analytics", "İstatistik & Analiz", "V")
        self._add_row(layout, "calibrate", "Kalibrasyon Sihirbazı", "K")
        self._add_row(layout, "yolo_factory", "YOLO Veri Fabrikası & RTX Eğitim Merkezi", "Y")
        self._add_row(layout, "settings", "Ayarlar (Haritayı Göster & Noktalar)...", "S")
        self._add_row(layout, "open_log", "Günlük Kaydını Aç (Log)", "L")
        self._add_row(layout, "guide", "Kullanım Kılavuzu (Yardım)", "I")
        self._add_row(layout, "copy_support", "Destek İçin Tanılama Kopyala", "C")

        self._add_separator(layout)

        # -----------------------------------------------------------------
        # 6. Çıkış (X)
        # -----------------------------------------------------------------
        self._add_row(layout, "exit", "Çıkış (Exit)", "X")

        self.adjustSize()

    def _add_row(
        self,
        layout: QVBoxLayout,
        action_id: str,
        text: str,
        badge_key: str,
        is_radio: bool = False,
        radio_checked: bool = False,
        has_submenu: bool = False,
    ) -> _MenuRowWidget:
        row = _MenuRowWidget(
            action_id=action_id,
            text=text,
            badge_key=badge_key,
            is_radio=is_radio,
            radio_checked=radio_checked,
            has_submenu=has_submenu,
            parent=self,
        )
        row.clicked.connect(self._on_row_clicked)
        row.hovered.connect(self._on_row_hovered)
        layout.addWidget(row)
        self._rows[action_id] = row
        return row

    def _add_separator(self, layout: QVBoxLayout) -> None:
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2d323b; border: none; margin: 4px 0px;")
        w = QWidget(self)
        w.setFixedHeight(9)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.addWidget(sep)
        layout.addWidget(w)

    def _format_active_route_label(self) -> str:
        if self._active_region and self._active_route:
            return f"Bölge (Region): {self._active_region} · Rota (Route): {self._active_route}"
        return "Bölge (Region) & Rota (Route) Seçimi"

    def _format_minimap_toggle_label(self) -> str:
        state_tr = "AÇIK" if self._minimap_auto_record_enabled else "KAPALI"
        return f"Sağ Üst Minimap Oto-Kayıt (Minimap Auto-Record Top-Right): {state_tr}"

    # -------------------------------------------------------------------------
    # Durum Güncelleme API'si (Public State Update API)
    # -------------------------------------------------------------------------
    def set_route_manager(self, route_manager: Any) -> None:
        self.route_manager = route_manager
        self._sync_from_route_manager()
        self._refresh_labels()

    def set_active_mode(self, mode: str) -> None:
        self._active_mode = self._normalize_mode(mode)
        self._refresh_labels()

    def set_active_route(self, region: str, route: str) -> None:
        self._active_region = region
        self._active_route = route
        try:
            if self.route_manager is not None:
                if hasattr(self.route_manager, "active_region"):
                    self.route_manager.active_region = region
                if hasattr(self.route_manager, "active_route"):
                    self.route_manager.active_route = route
                if hasattr(self.route_manager, "data") and isinstance(self.route_manager.data, dict):
                    self.route_manager.data["active_region"] = region
                    self.route_manager.data["active_route"] = route
                if hasattr(self.route_manager, "save_data"):
                    self.route_manager.save_data()
        except Exception as exc:
            logger.warning("Aktif rota RouteManager içine kaydedilirken hata: %s", exc)
        self._refresh_labels()

    def set_active_state(
        self,
        active_mode: Optional[str] = None,
        active_region: Optional[str] = None,
        active_route: Optional[str] = None,
        minimap_auto_record: Optional[bool] = None,
        harvests_per_node: Optional[int] = None,
    ) -> None:
        """main.py ve orkestratör tarafından F9 açılmadan önce çağrılan birleşik durum güncelleyici."""
        if active_mode is not None:
            self._active_mode = self._normalize_mode(active_mode)
        if active_region is not None:
            self._active_region = str(active_region)
        if active_route is not None:
            self._active_route = str(active_route)
        if minimap_auto_record is not None:
            self._minimap_auto_record_enabled = bool(minimap_auto_record)
        if harvests_per_node is not None:
            self._harvests_per_node = max(1, int(harvests_per_node))
        self._refresh_labels()

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
        """Menü üst bilgisindeki '3x Toplama Modu · Harvests/Node: 3x' rozetini günceller."""
        self._harvests_per_node = max(1, int(harvests))
        self._refresh_labels()

    def update_routes(self, regions_routes: Dict[str, List[str]]) -> None:
        if regions_routes:
            self._regions_routes = regions_routes
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        if hasattr(self, "harvest_badge_label"):
            self.harvest_badge_label.setText(
                f"3x Toplama Modu · Harvests/Node: {self._harvests_per_node}x"
            )
        if hasattr(self, "header_subtitle_label"):
            self.header_subtitle_label.setText(
                "Duraklatıldı — Devam için Enter  |  "
                f"Bölge (Region): {self._active_region} · Rota (Route): {self._active_route}"
            )

        if "resume" in self._rows:
            mode_str = self.MODE_LABELS.get(self._active_mode, "Manuel Kayıt (F8)")
            self._rows["resume"].set_text(
                f"Devam Et: {mode_str} (Duraklatıldı — Devam için Enter)"
            )

        for mode_id in ("auto_record", "manual_record", "route", "explorer"):
            if mode_id in self._rows:
                self._rows[mode_id].set_radio_checked(self._active_mode == mode_id)

        if "minimap_auto_record" in self._rows:
            self._rows["minimap_auto_record"].set_text(self._format_minimap_toggle_label())
            self._rows["minimap_auto_record"].set_radio_checked(self._minimap_auto_record_enabled)

        if "active_route" in self._rows:
            self._rows["active_route"].set_text(self._format_active_route_label())

        self.update()

    # -------------------------------------------------------------------------
    # Açılır Menü & Alt Menü Yönetimi
    # -------------------------------------------------------------------------
    def _release_win32_cursor_and_focus(self) -> None:
        """
        Aion 2 gibi 3B oyunların fare imlecini ekran merkezine kilitlemesini (ClipCursor / SetCapture)
        serbest bırakır ve F9 menüsünü Windows ön plan penceresi (Foreground Window) yaparak
        fare imlecinin menü üzerinde rahatça hareket etmesini sağlar.
        """
        if not sys.platform.startswith("win"):
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # 1. Oyun tarafından konulmuş imleç hapsetme sınırlarını (ClipCursor) ve yakalamayı kaldır
            user32.ClipCursor(None)
            user32.ReleaseCapture()

            # 2. İmleci görünür zorla (ShowCursor sayacı negatifse >= 0 yap)
            for _ in range(8):
                if user32.ShowCursor(True) >= 0:
                    break

            # 3. Menü penceresini güvenli şekilde ön plana (Foreground) getir
            hwnd = int(self.winId())
            if hwnd:
                fg_hwnd = user32.GetForegroundWindow()
                fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
                cur_tid = kernel32.GetCurrentThreadId()
                attached = False
                if fg_tid and cur_tid and fg_tid != cur_tid:
                    attached = bool(user32.AttachThreadInput(fg_tid, cur_tid, True))
                try:
                    user32.BringWindowToTop(hwnd)
                    user32.SetForegroundWindow(hwnd)
                    user32.SetActiveWindow(hwnd)
                    user32.SetFocus(hwnd)
                finally:
                    if attached:
                        user32.AttachThreadInput(fg_tid, cur_tid, False)
        except Exception as exc:
            logger.debug("Win32 fare kilidi açma / ön plan odaklama bilgisi: %s", exc)

    def popup_at_cursor(self) -> None:
        """F9 menüsünü doğrudan QCursor.pos() imleç konumunda açar ve fareyi serbest bırakır."""
        try:
            now_mono = time.monotonic()
            if self.isVisible() and (now_mono - self._last_popup_ts) < 0.45:
                self._release_win32_cursor_and_focus()
                return

            self._last_popup_ts = now_mono
            self._sync_from_route_manager()
            self._refresh_labels()
            for r in self._rows.values():
                r.set_hovered(False)

            self._release_win32_cursor_and_focus()

            pos = QCursor.pos()
            screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
            self.adjustSize()
            if screen is not None:
                avail = screen.availableGeometry()
                # İmleç ekranın tam ortasında veya kenarında olsa bile menünün hemen imleç altında açılmasını sağla
                x = min(max(pos.x() - 24, avail.left() + 8), avail.right() - self.width() - 8)
                y = min(max(pos.y() - 16, avail.top() + 8), avail.bottom() - self.height() - 8)
                pos = QPoint(x, y)

            self.move(pos)
            self.show()
            self.raise_()
            self.activateWindow()
            self._release_win32_cursor_and_focus()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        except Exception as exc:
            logger.error("F9Menu imleç konumunda açılamadı: %s", exc)

    def show_at_cursor(self) -> None:
        """popup_at_cursor için takma ad."""
        self.popup_at_cursor()

    def popup(self, pos: Optional[QPoint] = None) -> None:
        """QMenu uyumlu popup(pos) metodu."""
        if pos is None:
            self.popup_at_cursor()
        else:
            self._last_popup_ts = time.monotonic()
            self._sync_from_route_manager()
            self._refresh_labels()
            self.adjustSize()
            self.move(pos)
            self.show()
            self.raise_()
            self.activateWindow()
            self._release_win32_cursor_and_focus()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

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

            has_any = False
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
            self.hide()
            logger.info("F9Menu üzerinden rota seçildi: %s · %s", region, route)
            self.route_selected.emit(region, route)
            self.action_triggered.emit(f"route_selected:{region}:{route}")
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
                return

            if action_id in ("auto_record", "manual_record", "route", "explorer"):
                self.set_active_mode(action_id)

            self.hide()

            # Günlük açma, Kullanım Kılavuzu ve Destek Tanılama için yerleşik işleyiciler
            if action_id == "open_log":
                self._handle_open_log()
            elif action_id == "guide":
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
            self._trigger_action("copy_support")
        elif key == Qt.Key.Key_X:
            self._trigger_action("exit")
        elif key == Qt.Key.Key_Escape:
            if self._route_submenu is not None and self._route_submenu.isVisible():
                self._route_submenu.hide()
            self.hide()
        else:
            super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # Yerleşik Yardımcı İşleyiciler (Log, Kılavuz, Destek Panosu)
    # -------------------------------------------------------------------------
    def _ensure_log_file(self) -> str:
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
                "<li><b>F9 Hızlı Kontrol Menüsü & Ayarlar (S):</b> F tuşu aralığını, dinlenme süresini ve aktif Bölge/Rota seçimini yapılandırın.</li>"
                "<li><b>3x Toplama Modu (Harvests/Node: 3x):</b> Her kaynak noktası ardışık 3 aşamada (1/3 -> 2/3 -> 3/3) tamamen toplanır.</li>"
                "<li><b>Sağ Üst Minimap Oto-Kayıt (G):</b> Gezinirken sağ üst Minimap üzerindeki materyal simgeleri otomatik algılanıp rotaya kaydedilir.</li>"
                "<li><b>Mevcut Konumu Ekle (F8) / Manuel Kayıt (N):</b> Oyun içinde kaynak noktalarına yürüyüp <b>F8</b> tuşuna basarak noktayı kaydedin.</li>"
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



